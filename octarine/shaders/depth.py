"""Normalized depth rendering for the viewer.

pygfx's built-in ``DepthPass`` renders the raw depth buffer which is of
limited use for typical scenes: with a perspective projection the depth
buffer is highly non-linear and virtually all visible geometry maps to
values very close to 1.0 - i.e. everything shows up as the same shade of
grey.

The pass implemented here instead linearizes depth (to view-space
distance) and normalizes it to the depth range actually covered by the
geometry on screen: the nearest visible surface renders black, the
farthest light grey, and the (empty) background stays white. In
``overlay`` mode the scene's own colors are kept and darkened with
distance instead (depth cueing). The min/max depth is determined each
frame by a small compute-shader reduction over the depth buffer.
"""

import numpy as np
import wgpu

from pygfx.renderers.wgpu import EffectPass


class _DepthMinMaxReader:
    """Find the min/max depth of the geometry in the depth buffer.

    wgpu does not allow partial CPU readbacks of depth textures, so the
    reduction runs in a compute shader and only its 8-byte result is read
    back. The atomic min/max operate on bit-cast depth values, which works
    because non-negative IEEE 754 floats compare like their u32 bit
    patterns.
    """

    _WGSL = """
        struct MinMax {
            dmin: atomic<u32>,
            dmax: atomic<u32>,
        }

        @group(0) @binding(0) var depthTex: texture_depth_2d;
        @group(0) @binding(1) var<storage, read_write> result: MinMax;

        @compute @workgroup_size(16, 16)
        fn main(@builtin(global_invocation_id) gid: vec3u) {
            let dims = textureDimensions(depthTex);
            if (gid.x >= dims.x || gid.y >= dims.y) {
                return;
            }
            let depth = textureLoad(depthTex, vec2i(gid.xy), 0);
            if (depth >= 1.0) {  // no geometry at this pixel
                return;
            }
            let bits = bitcast<u32>(depth);
            atomicMin(&result.dmin, bits);
            atomicMax(&result.dmax, bits);
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
            size=8,
            usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
        )

    def read(self, depth_texture):
        """Return the geometry's (min, max) depth (0-1), or None if there is none."""
        # Reset to dmin = bits of 1.0, dmax = bits of 0.0
        self._device.queue.write_buffer(
            self._result_buffer, 0, np.array([0x3F800000, 0], dtype=np.uint32)
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
                        "size": 8,
                    },
                },
            ],
        )
        encoder = self._device.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self._pipeline)
        cpass.set_bind_group(0, bind_group)
        cpass.dispatch_workgroups(
            (depth_texture.size[0] + 15) // 16, (depth_texture.size[1] + 15) // 16
        )
        cpass.end()
        self._device.queue.submit([encoder.finish()])

        dmin, dmax = np.frombuffer(
            self._device.queue.read_buffer(self._result_buffer), dtype=np.uint32
        ).view(np.float32)
        if dmax < dmin:  # buffer untouched -> no geometry on screen
            return None
        return float(dmin), float(dmax)


class NormalizedDepthPass(EffectPass):
    """Render normalized depth as shades of grey.

    Unlike pygfx's ``DepthPass`` (which renders the raw - highly
    non-linear - depth buffer), this linearizes depth to view-space
    distance and stretches it over the depth range actually covered by
    the geometry on screen: the nearest visible surface renders black,
    the farthest light grey, and the (empty) background stays white.

    Parameters
    ----------
    camera :    pygfx.Camera, optional
                The camera used to render the scene; needed to linearize
                depth values. If None, the raw depth values are
                normalized instead - with a perspective camera this
                exaggerates depth differences close to the camera.
    overlay :   bool
                If True, the scene's own colors are kept and darkened
                with distance (depth cueing) instead of being replaced
                by greyscale; the background is left untouched.
    strength :  float
                How dark the farthest geometry is rendered, from 0 (not
                at all) to 1 (black in greyscale mode; fully darkened in
                overlay mode). The default of 0.9 keeps the farthest
                geometry distinguishable from the (white) background in
                greyscale mode.

    """

    USES_DEPTH = True

    uniform_type = dict(
        EffectPass.uniform_type,
        projection_transform_inv="4x4xf4",
        lin_min="f4",
        lin_range="f4",
        overlay="f4",
        strength="f4",
    )

    wgsl = """
        @fragment
        fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
            let texIndex = vec2i(varyings.position.xy);
            let depth = textureLoad(depthTex, texIndex, 0);
            let overlay = u_effect.overlay > 0.5;
            if (depth >= 1.0) {  // no geometry at this pixel
                let color = textureLoad(colorTex, texIndex, 0);
                return select(vec4f(1.0, 1.0, 1.0, 1.0), color, overlay);
            }
            // View-space distance (with camera=None this is just the raw
            // depth again, see render())
            let h = u_effect.projection_transform_inv * vec4f(0.0, 0.0, depth, 1.0);
            let lin = -h.z / h.w;
            let t = clamp((lin - u_effect.lin_min) / u_effect.lin_range, 0.0, 1.0);
            if (overlay) {
                // Darken the scene's own colors with distance
                let color = textureLoad(colorTex, texIndex, 0);
                return vec4f(color.rgb * (1.0 - t * u_effect.strength), color.a);
            }
            let grey = t * u_effect.strength;
            return vec4f(grey, grey, grey, 1.0);
        }
    """

    def __init__(self, camera=None, overlay=False, strength=0.9):
        super().__init__()
        self.camera = camera
        self.overlay = overlay
        self.strength = strength
        self._minmax_reader = None

    @property
    def overlay(self):
        """Whether to darken the scene's colors instead of rendering greyscale."""
        return bool(self._uniform_data["overlay"])

    @overlay.setter
    def overlay(self, value):
        self._uniform_data["overlay"] = float(bool(value))

    @property
    def strength(self):
        """How dark the farthest geometry is rendered (0-1)."""
        return float(self._uniform_data["strength"])

    @strength.setter
    def strength(self, value):
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"strength must be in [0, 1], got {value}")
        self._uniform_data["strength"] = value

    @staticmethod
    def _linearize(proj_inv, depth):
        h = proj_inv @ np.array([0.0, 0.0, depth, 1.0])
        return -h[2] / h[3]  # the camera looks down -z in view space

    def render(self, command_encoder, color_tex, depth_tex, target_tex):
        # The camera matrices are only final at render time
        if self.camera is not None:
            proj_inv = self.camera.projection_matrix_inverse
        else:
            # No camera: -h.z/h.w recovers the raw depth value
            proj_inv = np.diag([1.0, 1.0, -1.0, 1.0])
        self._uniform_data["projection_transform_inv"] = proj_inv.T

        # Normalize to the depth range of the on-screen geometry. The
        # linearization is monotonic in depth, so the raw min/max map to
        # the linear min/max (modulo ordering, which `sorted` fixes).
        if self._minmax_reader is None:
            self._minmax_reader = _DepthMinMaxReader(self._device)
        minmax = self._minmax_reader.read(depth_tex.texture)
        if minmax is None:  # no geometry: all pixels take the background branch
            lmin, lmax = 0.0, 1.0
        else:
            lmin, lmax = sorted(self._linearize(proj_inv, d) for d in minmax)
        if (lmax - lmin) < 1e-9:
            # Only a single depth value on screen: render it mid-grey
            lmin, lmax = lmin - 1.0, lmin + 1.0
        self._uniform_data["lin_min"] = lmin
        self._uniform_data["lin_range"] = lmax - lmin

        super().render(command_encoder, color_tex, depth_tex, target_tex)
