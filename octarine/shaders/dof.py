"""Depth-of-field post-processing for the viewer.

Objects near a focal plane are rendered sharp while everything closer or
farther is progressively blurred, similar to a photographic lens. This is
implemented as a pygfx ``EffectPass`` (a screen-space post-processing pass
with access to the depth buffer): per pixel we compute a "circle of
confusion" from the distance to the focal plane and gather colors over a
golden-angle disk of that radius.

Note that as a screen-space effect this applies to the entire rendered
image - including overlay elements such as messages - and objects that do
not write depth (e.g. meshes with transparent alpha modes) are blurred by
whatever is behind them.
"""

import math
import time

import numpy as np
import pygfx as gfx
import wgpu

from pygfx.renderers.wgpu import EffectPass


class _AutofocusDepthReader:
    """Find the autofocus target in the depth buffer.

    Searches the pixel disk of a given radius around the view center for
    the (spatially) closest pixel that holds any geometry and returns its
    depth and pixel offset. wgpu does not allow partial CPU readbacks of
    depth textures, so the search runs in a minimal compute shader and
    only its 16-byte result is read back.
    """

    _WGSL = """
        struct Result {
            found: f32,
            depth: f32,
            dx: f32,
            dy: f32,
        }

        @group(0) @binding(0) var depthTex: texture_depth_2d;
        @group(0) @binding(1) var<storage, read_write> result: Result;
        @group(0) @binding(2) var<uniform> u_radius: f32;

        var<private> best_d2: f32 = 1e30;
        var<private> best_depth: f32 = 1.0;
        var<private> best_off: vec2f = vec2f(0.0, 0.0);

        fn consider(offset: vec2i, center: vec2i, dims: vec2i, r2: f32) {
            let p = center + offset;
            if (p.x < 0 || p.y < 0 || p.x >= dims.x || p.y >= dims.y) {
                return;
            }
            let d2 = f32(offset.x * offset.x + offset.y * offset.y);
            if (d2 > r2 || d2 >= best_d2) {
                return;
            }
            let depth = textureLoad(depthTex, p, 0);
            if (depth >= 1.0) {  // no geometry at this pixel
                return;
            }
            best_d2 = d2;
            best_depth = depth;
            best_off = vec2f(offset);
        }

        @compute @workgroup_size(1)
        fn main() {
            let dims = vec2i(textureDimensions(depthTex));
            let center = dims / 2;
            let r = i32(u_radius);
            let r2 = u_radius * u_radius;

            // Search in expanding square rings; ring k has a minimum
            // (Euclidean) distance of k, so we can stop as soon as no
            // closer hit is possible
            for (var k = 0; k <= r; k += 1) {
                if (f32(k * k) > best_d2) {
                    break;
                }
                if (k == 0) {
                    consider(vec2i(0, 0), center, dims, r2);
                    continue;
                }
                for (var i = -k; i <= k; i += 1) {
                    consider(vec2i(i, -k), center, dims, r2);
                    consider(vec2i(i, k), center, dims, r2);
                }
                for (var i = -k + 1; i <= k - 1; i += 1) {
                    consider(vec2i(-k, i), center, dims, r2);
                    consider(vec2i(k, i), center, dims, r2);
                }
            }

            result.found = select(0.0, 1.0, best_d2 < 1e29);
            result.depth = best_depth;
            result.dx = best_off.x;
            result.dy = best_off.y;
        }
    """

    def __init__(self, device):
        self._device = device
        module = device.create_shader_module(code=self._WGSL)
        self._pipeline = device.create_compute_pipeline(
            layout="auto",
            compute={"module": module, "entry_point": "main"},
        )
        self._result_buffer = device.create_buffer(
            size=16, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )
        self._radius_buffer = device.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

    def read(self, depth_texture, radius=0.0):
        """Find the closest geometry within `radius` px of the view center.

        Returns a (depth, dx, dy) tuple - the depth (0-1) plus the pixel
        offset relative to the center - or None if there is no geometry
        within the search radius.
        """
        self._device.queue.write_buffer(
            self._radius_buffer, 0, np.array([radius, 0, 0, 0], dtype=np.float32)
        )
        bind_group = self._device.create_bind_group(
            layout=self._pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": depth_texture.create_view()},
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self._result_buffer,
                        "offset": 0,
                        "size": 16,
                    },
                },
                {
                    "binding": 2,
                    "resource": {
                        "buffer": self._radius_buffer,
                        "offset": 0,
                        "size": 16,
                    },
                },
            ],
        )
        encoder = self._device.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self._pipeline)
        cpass.set_bind_group(0, bind_group)
        cpass.dispatch_workgroups(1)
        cpass.end()
        self._device.queue.submit([encoder.finish()])

        found, depth, dx, dy = np.frombuffer(
            self._device.queue.read_buffer(self._result_buffer), np.float32
        )
        if found < 0.5:
            return None
        return float(depth), float(dx), float(dy)


class DepthOfFieldPass(EffectPass):
    """A depth-of-field (focal blur) post-processing pass.

    Parameters
    ----------
    camera :    pygfx.Camera
                The camera used to render the scene. Needed to map depth
                values back to world-unit distances.
    focus :     float, optional
                Distance of the focal plane from the camera in world units.
                For orthographic cameras this may be negative: pygfx places
                the camera in the middle of the scene. If None (default),
                continuously auto-focuses on whatever is at the center of
                the view (if that is empty space, the image is left sharp).
    aperture :  float
                Blur strength: the blur radius in physical pixels of a
                point at 100% relative defocus - relative to the focus
                distance for perspective cameras, and to the visible height
                of the view for orthographic ones. Typical values are
                50-300.
    max_radius : float
                Upper limit for the blur radius in physical pixels.
    num_taps :  int
                Number of samples in the blur kernel. More taps give a
                smoother blur at a higher rendering cost.
    smooth :    float | bool
                Only relevant for autofocus (`focus=None`): if > 0, changes
                in focus are eased over approximately this many seconds
                instead of snapping instantly (True = 0.2s). While the view
                center is over empty space the last focus is held. Note
                that when rendering reactively something needs to keep
                triggering re-renders until a re-focus transition has
                settled (see `Viewer.set_depth_of_field`, which handles
                this).
    snap_radius : float
                Only relevant for autofocus (`focus=None`): search radius
                in physical pixels around the view center. The autofocus
                targets the object closest to the view center within that
                radius, instead of only what is exactly under the center
                pixel. 0 (default) disables snapping.

    """

    USES_DEPTH = True

    uniform_type = dict(
        EffectPass.uniform_type,
        projection_transform_inv="4x4xf4",
        focus="f4",
        autofocus="f4",
        aperture="f4",
        max_radius="f4",
        ortho_scale="f4",
    )

    wgsl = """
        fn linear_dist(depth: f32, texCoord: vec2f) -> f32 {
            // Signed view-space depth; can be negative for ortho cameras,
            // which pygfx places in the middle of the scene
            let ndc = vec4f(texCoord.x * 2.0 - 1.0, 1.0 - texCoord.y * 2.0, depth, 1.0);
            let h = u_effect.projection_transform_inv * ndc;
            return -h.z / h.w;  // the camera looks down -z in view space
        }

        fn coc_radius(dist: f32, focus: f32) -> f32 {
            // Perspective uses the thin-lens ratio; for an orthographic
            // camera (ortho_scale > 0) the distance to the camera is
            // arbitrary, so we normalize by the visible height of the view
            let denom = select(max(dist, 1e-6), u_effect.ortho_scale, u_effect.ortho_scale > 0.0);
            let c = u_effect.aperture * abs(dist - focus) / denom;
            return clamp(c, 0.0, u_effect.max_radius);
        }

        const GOLDEN_ANGLE: f32 = 2.39996323;
        const NUM_TAPS: i32 = {{ num_taps }};

        @fragment
        fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
            let texIndex = vec2i(varyings.position.xy);
            let dims = vec2f(textureDimensions(colorTex));
            let max_i = vec2i(dims) - 1;

            // Focal distance: fixed, or read from the pixel at the center
            // of the view (autofocus)
            var focus = u_effect.focus;
            if (u_effect.autofocus > 0.5) {
                let c_idx = vec2i(dims * 0.5);
                let c_depth = textureLoad(depthTex, c_idx, 0);
                if (c_depth >= 1.0) {
                    // Nothing under the center of the view -> keep the image sharp
                    return textureLoad(colorTex, texIndex, 0);
                }
                focus = linear_dist(c_depth, (vec2f(c_idx) + 0.5) / dims);
            }

            let center_dist = linear_dist(textureLoad(depthTex, texIndex, 0), varyings.texCoord);
            let center_coc = coc_radius(center_dist, focus);

            var acc = textureLoad(colorTex, texIndex, 0);
            var w_acc = 1.0;

            // Gather over at least half the max radius so that blurred
            // foreground objects can bleed onto in-focus regions behind them
            let gather_r = max(center_coc, u_effect.max_radius * 0.5);
            for (var i = 0; i < NUM_TAPS; i += 1) {
                let r = gather_r * sqrt((f32(i) + 0.5) / f32(NUM_TAPS));
                let ang = f32(i) * GOLDEN_ANGLE;
                let offs = vec2f(cos(ang), sin(ang)) * r;

                let idx = clamp(vec2i(vec2f(texIndex) + offs), vec2i(0), max_i);
                let uv = (vec2f(idx) + 0.5) / dims;

                let s_dist = linear_dist(textureLoad(depthTex, idx, 0), uv);
                var s_coc = coc_radius(s_dist, focus);
                // Background may not bleed over a sharper foreground: clamp
                // the blur of samples behind the center pixel to the
                // center's own blur
                if (s_dist > center_dist) {
                    s_coc = min(s_coc, center_coc);
                }
                // Scatter-as-gather: a sample only contributes if its own
                // blur disk reaches this pixel
                let w = smoothstep(r - 1.0, r + 1.0, s_coc);
                acc += textureLoad(colorTex, idx, 0) * w;
                w_acc += w;
            }
            return acc / w_acc;
        }
    """

    def __init__(
        self,
        camera,
        focus=None,
        aperture=100.0,
        max_radius=16.0,
        num_taps=64,
        smooth=0.0,
        snap_radius=0.0,
    ):
        super().__init__()
        self.camera = camera
        self.focus = focus
        self.aperture = aperture
        self.max_radius = max_radius
        self.num_taps = num_taps
        self.smooth = smooth
        self.snap_radius = snap_radius
        self._depth_reader = None
        # State for CPU-driven (smoothed and/or snapping) autofocus
        self._smooth_value = None
        self._smooth_settled = True
        self._smooth_last_t = None
        self._autofocus_point_view = None  # view-space target point

    @property
    def focus(self):
        """Focal distance in world units; None means autofocus."""
        return self._focus

    @focus.setter
    def focus(self, value):
        # The uniforms are set at render time
        self._focus = float(value) if value is not None else None

    @property
    def aperture(self):
        """Blur strength (px of blur radius at 100% relative defocus)."""
        return float(self._uniform_data["aperture"])

    @aperture.setter
    def aperture(self, value):
        value = float(value)
        if value < 0:
            raise ValueError(f"aperture must be >= 0, got {value}")
        self._uniform_data["aperture"] = value

    @property
    def max_radius(self):
        """Upper limit for the blur radius in physical pixels."""
        return float(self._uniform_data["max_radius"])

    @max_radius.setter
    def max_radius(self, value):
        value = float(value)
        if value < 0:
            raise ValueError(f"max_radius must be >= 0, got {value}")
        self._uniform_data["max_radius"] = value

    @property
    def num_taps(self):
        """Number of samples in the blur kernel."""
        return self._template_vars["num_taps"]

    @num_taps.setter
    def num_taps(self, value):
        value = int(value)
        if value < 1:
            raise ValueError(f"num_taps must be >= 1, got {value}")
        self._set_template_var(num_taps=value)

    @property
    def smooth(self):
        """Approximate autofocus re-focus time in seconds; 0 = instant."""
        return self._smooth

    @smooth.setter
    def smooth(self, value):
        if isinstance(value, bool):
            value = 0.2 if value else 0.0
        value = float(value)
        if value < 0:
            raise ValueError(f"smooth must be >= 0, got {value}")
        self._smooth = value

    @property
    def snap_radius(self):
        """Autofocus search radius around the view center in px; 0 = off."""
        return self._snap_radius

    @snap_radius.setter
    def snap_radius(self, value):
        value = float(value)
        if value < 0:
            raise ValueError(f"snap_radius must be >= 0, got {value}")
        self._snap_radius = value

    def get_focus_position(self, renderer):
        """Return the world position of the current focal point.

        With a fixed `focus` this is the point at the focus distance along
        the view axis; with autofocus (`focus=None`) it is the surface point
        at the center of the view, read back from the depth buffer of the
        last rendered frame.

        Parameters
        ----------
        renderer :  pygfx.WgpuRenderer
                    The renderer used to render the scene.

        Returns
        -------
        (3,) array | None
                    None if the position cannot be determined - i.e. with
                    autofocus when nothing has been rendered yet or there
                    is no object under the center of the view.

        """
        if self.focus is not None:
            # The focal point on the view axis, in view space
            p = self.camera.world.matrix @ np.array([0.0, 0.0, -self.focus, 1.0])
            return p[:3] / p[3]

        if self.smooth > 0 or self.snap_radius > 0:
            # CPU-driven autofocus: use the state maintained by the render
            # pass. The focal point sits on the ray towards the (possibly
            # snapped) target, at the (possibly still easing) focal distance.
            target_pt = self._autofocus_point_view
            if target_pt is None or self._smooth_value is None:
                return None
            target_dist = -target_pt[2]
            if getattr(self.camera, "fov", 0) == 0:
                view_pt = np.array(
                    [target_pt[0], target_pt[1], -self._smooth_value]
                )
            else:
                view_pt = np.asarray(target_pt) * (
                    self._smooth_value / max(target_dist, 1e-9)
                )
            p = self.camera.world.matrix @ np.append(view_pt, 1.0)
            return p[:3] / p[3]

        tex = renderer._blender.get_texture("depth")
        if tex is None:  # nothing rendered yet
            return None
        if self._depth_reader is None:
            self._depth_reader = _AutofocusDepthReader(renderer._device)
        res = self._depth_reader.read(tex, 0.0)
        if res is None:  # nothing under the center of the view
            return None

        # Unproject the center pixel back into world space
        ndc = np.array([0.0, 0.0, res[0], 1.0])
        p = self.camera.world.matrix @ (self.camera.projection_matrix_inverse @ ndc)
        return p[:3] / p[3]

    def _autofocus_step(self, depth_tex):
        """Determine this frame's autofocus distance on the CPU.

        Used whenever snapping and/or smoothing are enabled - both need
        frame-to-frame state that the (stateless) fragment shader cannot
        provide.
        """
        if self._depth_reader is None:
            self._depth_reader = _AutofocusDepthReader(self._device)
        res = self._depth_reader.read(depth_tex.texture, self.snap_radius)

        now = time.perf_counter()
        dt = 0.0 if self._smooth_last_t is None else now - self._smooth_last_t
        # Long gaps between renders (e.g. with reactive rendering) should
        # not make the transition snap
        dt = min(max(dt, 0.0), 0.05)
        self._smooth_last_t = now

        if res is not None:  # something within reach of the view center
            depth, dx, dy = res
            w, h = depth_tex.size[0], depth_tex.size[1]
            ndc = np.array(
                [
                    ((w // 2 + dx) + 0.5) / w * 2.0 - 1.0,
                    1.0 - ((h // 2 + dy) + 0.5) / h * 2.0,
                    depth,
                    1.0,
                ]
            )
            pt = self.camera.projection_matrix_inverse @ ndc
            self._autofocus_point_view = pt[:3] / pt[3]
            target = -self._autofocus_point_view[2]

            if self._smooth_value is None or self.smooth <= 0:
                self._smooth_value = target  # no easing: snap directly
            else:
                tau = self.smooth / 3.0  # ~95% converged after `smooth` s
                self._smooth_value += (target - self._smooth_value) * (
                    1.0 - math.exp(-dt / tau)
                )
            # The transition has settled once the remaining focus error
            # would cause less than ~0.05 px of blur
            scale = float(self._uniform_data["ortho_scale"]) or max(
                abs(target), 1e-9
            )
            if self.aperture * abs(target - self._smooth_value) / scale < 0.05:
                self._smooth_value = target
                self._smooth_settled = True
            else:
                self._smooth_settled = False
        else:
            self._smooth_settled = True
            if self.smooth <= 0:
                # Mirror the plain autofocus: nothing in reach -> all sharp
                self._smooth_value = None
                self._autofocus_point_view = None
            # else: hold the current focus while over empty space

        if self._smooth_value is None:
            # No focus target (yet): fall back to the (per-fragment)
            # shader autofocus, which leaves the image sharp on background
            self._uniform_data["autofocus"] = 1.0
        else:
            self._uniform_data["autofocus"] = 0.0
            self._uniform_data["focus"] = self._smooth_value

    def render(self, command_encoder, color_tex, depth_tex, target_tex):
        # The camera matrices are only final at render time
        self._uniform_data["projection_transform_inv"] = (
            self.camera.projection_matrix_inverse.T
        )
        if getattr(self.camera, "fov", 0) == 0:
            # Orthographic: normalize blur by the visible view height
            zoom = getattr(self.camera, "zoom", 1) or 1
            self._uniform_data["ortho_scale"] = (
                abs(float(self.camera.height) / zoom) or 1.0
            )
        else:
            self._uniform_data["ortho_scale"] = 0.0

        # Focus: fixed, CPU-driven autofocus (snapping and/or smoothing),
        # or plain per-fragment shader autofocus
        if self._focus is not None:
            self._uniform_data["focus"] = self._focus
            self._uniform_data["autofocus"] = 0.0
            self._smooth_value = None
            self._smooth_settled = True
            self._autofocus_point_view = None
        elif self.smooth > 0 or self.snap_radius > 0:
            self._autofocus_step(depth_tex)
        else:
            self._uniform_data["autofocus"] = 1.0
            self._smooth_value = None
            self._smooth_settled = True
            self._autofocus_point_view = None

        super().render(command_encoder, color_tex, depth_tex, target_tex)
