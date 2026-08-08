"""Screen-space ambient occlusion (SSAO) for the viewer.

Ambient light is normally applied uniformly, which leaves creases, cavities
and the contact points between objects looking flat. This pass estimates how
much of the surrounding hemisphere is blocked at each pixel and darkens the
image accordingly - the cheap screen-space stand-in for the shadowing that
ambient light would produce.

`pygfx` has no ambient occlusion of its own (only baked `ao_map` textures,
which require UVs and a precomputed map), so the effect is implemented here
as an ``EffectPass`` that runs three steps of its own:

1. An occlusion pass reconstructs view-space positions and normals from the
   depth buffer and samples a hemisphere around each pixel: every sample
   that ends up behind existing geometry counts as occluded. The result goes
   into an off-screen texture.
2. Two separable bilateral blur passes smooth the (necessarily noisy)
   occlusion without bleeding across depth discontinuities.
3. A composite pass multiplies the occlusion into the rendered image.

Note that as a screen-space effect this applies to the entire rendered
image - including overlay elements such as messages - and that objects
which do not write depth (e.g. meshes with transparent alpha modes) neither
cast nor receive occlusion.
"""

import wgpu

from pygfx.renderers.wgpu import CopyPass, EffectPass
from pygfx.renderers.wgpu.engine.effectpasses import FullQuadPass

# Depth reading, view-space positions and (reconstructed) view-space normals.
# Shared with the outline pass; both bind `depthTex` and carry a
# `projection_transform_inv` uniform.
from .depth_wgsl import DEPTH_WGSL, VIEW_POS_WGSL, VIEW_NORMAL_WGSL


class AmbientOcclusionPass(EffectPass):
    """A screen-space ambient occlusion (SSAO) post-processing pass.

    Estimates how much of the hemisphere above each pixel is blocked by
    nearby geometry and darkens the image accordingly, which brings out
    creases, cavities and the contact points between objects.

    Parameters
    ----------
    camera :    pygfx.Camera
                The camera used to render the scene. Needed to map depth
                values back to view-space positions.
    radius :    float
                How far - in world units - to look for occluders. This is
                the one parameter that has to match the scene: too small
                and the effect disappears, too large and it turns into a
                dark haze. A good starting point is a few percent of the
                size of the scene (see `Viewer.set_ambient_occlusion`,
                which derives such a default from the scene bounds).
    intensity : float
                Strength of the darkening, from 0 (no effect) to 1 (fully
                occluded pixels turn black).
    bias :      float
                Occluders closer to the surface than this - as a fraction
                of `radius` - are ignored. Raise it if flat surfaces show
                occlusion of their own ("self-occlusion", caused by depth
                buffer precision), lower it if fine detail is lost.
    samples :   int
                Number of hemisphere samples per pixel. More samples mean
                less noise at a higher rendering cost.
    power :     float
                Exponent applied to the occlusion; values > 1 restrict the
                effect to the darkest areas, values < 1 spread it out.
    blur :      bool | int
                Radius (in pixels) of the bilateral blur that removes the
                sampling noise; the kernel spans twice as many taps. True
                (default) uses 2, i.e. exactly one tile of the 4x4 sampling
                pattern - other values leave some of the pattern visible.
                False (or 0) disables the blur, which is mostly useful for
                debugging.
    debug :     bool
                If True, render the occlusion itself as greyscale instead
                of darkening the scene. Useful for tuning `radius`.

    """

    USES_DEPTH = True

    class _OcclusionPass(FullQuadPass):
        """Estimate the occlusion at each pixel from the depth buffer."""

        uniform_type = dict(
            projection_transform="4x4xf4",
            projection_transform_inv="4x4xf4",
            radius="f4",
            bias="f4",
        )

        wgsl = (
            DEPTH_WGSL
            + VIEW_POS_WGSL
            + VIEW_NORMAL_WGSL
            + """
            const NUM_SAMPLES: i32 = {{ num_samples }};
            const GOLDEN_ANGLE: f32 = 2.39996323;
            const TAU: f32 = 6.28318531;

            // Ordered 4x4 dither (a Bayer matrix, built recursively). Each
            // pixel of a 4x4 tile gets one of 16 sample configurations, so
            // a blur spanning a whole tile averages all 16 of them - the
            // classic interleaved sampling trick. Anything with a longer
            // period (e.g. interleaved gradient noise) leaves streaks the
            // blur cannot remove.
            fn bayer2(a: vec2f) -> f32 {
                let b = floor(a);
                return fract(b.x * 0.5 + b.y * b.y * 0.75);
            }

            fn bayer4(a: vec2f) -> f32 {
                return bayer2(a * 0.5) * 0.25 + bayer2(a);
            }

            @fragment
            fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
                let dims = vec2f(textureDimensions(depthTex));
                let tex_index = vec2i(varyings.position.xy);
                let depth = textureLoad(depthTex, tex_index, 0);
                if (depth >= 1.0) {
                    return vec4f(1.0, 0.0, 0.0, 1.0);  // no geometry here
                }

                let pos = to_view_pos(varyings.texCoord, depth);
                let normal = view_normal_at(tex_index, depth, pos);

                // Tangent frame around the normal, rotated per pixel. The
                // same dither value also offsets the sample distances, so
                // each pixel of a tile really does sample differently.
                let dither = bayer4(varyings.position.xy);
                let angle0 = dither * TAU;
                let jitter = dither;
                var up = vec3f(0.0, 0.0, 1.0);
                if (abs(normal.z) > 0.9) {
                    up = vec3f(1.0, 0.0, 0.0);
                }
                let tangent = normalize(cross(up, normal));
                let bitangent = cross(normal, tangent);

                let radius = max(u_effect.radius, 1e-9);
                let bias = u_effect.bias * radius;

                var occlusion = 0.0;
                for (var i = 0; i < NUM_SAMPLES; i += 1) {
                    // Cosine-weighted direction in the hemisphere around the
                    // normal, taken from a golden-angle spiral
                    let u = (f32(i) + 0.5) / f32(NUM_SAMPLES);
                    let r = sqrt(u);
                    let angle = f32(i) * GOLDEN_ANGLE + angle0;
                    let dir = tangent * (cos(angle) * r)
                            + bitangent * (sin(angle) * r)
                            + normal * sqrt(max(1.0 - u, 0.0));

                    // Sample distance, squared so that samples bunch up near
                    // the surface where contact shadows matter most. It runs
                    // over a sequence of its own: deriving it from `u` as
                    // well would tie every grazing direction to the far end
                    // of the range, which biases the estimate.
                    let t = fract(jitter + f32(i) * 0.61803399);
                    let sample_pos = pos + dir * (radius * mix(0.1, 1.0, t * t));

                    // Where does that point land on screen?
                    let clip = u_effect.projection_transform * vec4f(sample_pos, 1.0);
                    if (clip.w <= 0.0) {
                        continue;  // behind the camera
                    }
                    let ndc = clip.xyz / clip.w;
                    let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
                    if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) {
                        continue;  // off screen: no information
                    }

                    // Note that a uv of exactly 1.0 passes the test above
                    let s_index = clamp(vec2i(uv * dims), vec2i(0), vec2i(dims) - 1);
                    let s_depth = textureLoad(depthTex, s_index, 0);
                    if (s_depth >= 1.0) {
                        continue;  // background: nothing to occlude with
                    }
                    let s_pos = to_view_pos((vec2f(s_index) + 0.5) / dims, s_depth);

                    // The sample is occluded if the geometry actually drawn
                    // at that pixel sits in front of it
                    if (s_pos.z - sample_pos.z > bias) {
                        // Only count occluders within `radius`, else objects
                        // far in front cast a dark halo onto what is behind
                        occlusion += smoothstep(
                            0.0, 1.0, radius / max(abs(pos.z - s_pos.z), 1e-9)
                        );
                    }
                }

                let ao = clamp(1.0 - occlusion / f32(NUM_SAMPLES), 0.0, 1.0);
                return vec4f(ao, 0.0, 0.0, 1.0);
            }
        """
        )

    class _BlurPass(FullQuadPass):
        """Blur the occlusion along one axis, keeping depth edges intact."""

        uniform_type = dict(
            projection_transform_inv="4x4xf4",
            direction="2xf4",
            depth_sigma="f4",
        )

        wgsl = (
            DEPTH_WGSL
            + """
            const RADIUS: i32 = {{ blur_radius }};

            @fragment
            fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
                let dims = vec2i(textureDimensions(aoTex));
                let tex_index = vec2i(varyings.position.xy);
                let center = textureLoad(aoTex, tex_index, 0).r;
                let center_depth = depth_at(tex_index);
                if (center_depth >= 1.0) {
                    return vec4f(center, 0.0, 0.0, 1.0);  // no geometry here
                }
                let center_z = to_view_depth(center_depth);

                let dir = vec2i(u_effect.direction);
                // The depth difference at which a neighbour counts as a
                // different surface
                let sigma_z = max(u_effect.depth_sigma, 1e-9);

                // Spatially this is a plain box filter over 2 * RADIUS taps
                // rather than a Gaussian: only averaging whole tiles of the
                // 4x4 sampling pattern cancels it exactly.
                var acc = 0.0;
                var w_acc = 0.0;
                for (var i = -RADIUS; i < RADIUS; i += 1) {
                    let idx = clamp(tex_index + dir * i, vec2i(0), dims - 1);
                    let s_depth = depth_at(idx);
                    if (s_depth >= 1.0) {
                        continue;  // background
                    }
                    let dz = (to_view_depth(s_depth) - center_z) / sigma_z;
                    let w = exp(-0.5 * dz * dz);
                    acc += textureLoad(aoTex, idx, 0).r * w;
                    w_acc += w;
                }
                // The center pixel is never background, so w_acc > 0
                return vec4f(acc / w_acc, 0.0, 0.0, 1.0);
            }
        """
        )

        def __init__(self, direction):
            super().__init__()
            self._uniform_data["direction"] = direction

    class _CompositePass(FullQuadPass):
        """Darken the rendered image by the occlusion."""

        uniform_type = dict(
            intensity="f4",
            power="f4",
            debug="f4",
        )

        wgsl = """
            @fragment
            fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
                let tex_index = vec2i(varyings.position.xy);
                let color = textureLoad(colorTex, tex_index, 0);
                let ao_raw = clamp(textureLoad(aoTex, tex_index, 0).r, 0.0, 1.0);
                let ao = pow(ao_raw, u_effect.power);
                if (u_effect.debug > 0.5) {
                    return vec4f(ao, ao, ao, 1.0);
                }
                let f = clamp(mix(1.0, ao, u_effect.intensity), 0.0, 1.0);
                return vec4f(color.rgb * f, color.a);
            }
        """

    def __init__(
        self,
        camera,
        radius=1.0,
        intensity=1.0,
        bias=0.01,
        samples=16,
        power=1.0,
        blur=True,
        debug=False,
    ):
        super().__init__()
        self.camera = camera
        self.radius = radius
        self.intensity = intensity
        self.bias = bias
        self.power = power
        self.debug = debug

        # Each sub-pass keeps its own uniform buffer. That matters: they are
        # all recorded into a single command encoder, so two passes sharing a
        # buffer would both see whatever was written to it last.
        self._occlusion_pass = self._OcclusionPass()
        self._blur_passes = (
            self._BlurPass((1.0, 0.0)),  # horizontal
            self._BlurPass((0.0, 1.0)),  # vertical
        )
        self._composite_pass = self._CompositePass()
        self._copy_pass = None  # only used if there is no depth buffer

        self.samples = samples
        self.blur = blur

        self._ao_textures = ()
        self._ao_views = ()
        self._ao_size = None

    @property
    def radius(self):
        """How far to look for occluders, in world units."""
        return self._radius

    @radius.setter
    def radius(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError(f"radius must be > 0, got {value}")
        self._radius = value

    @property
    def intensity(self):
        """Strength of the darkening (0 = off, 1 = black where fully occluded)."""
        return self._intensity

    @intensity.setter
    def intensity(self, value):
        value = float(value)
        if value < 0:
            raise ValueError(f"intensity must be >= 0, got {value}")
        self._intensity = value

    @property
    def bias(self):
        """Occluders nearer than this fraction of `radius` are ignored."""
        return self._bias

    @bias.setter
    def bias(self, value):
        value = float(value)
        if value < 0:
            raise ValueError(f"bias must be >= 0, got {value}")
        self._bias = value

    @property
    def samples(self):
        """Number of hemisphere samples per pixel."""
        return self._occlusion_pass._template_vars["num_samples"]

    @samples.setter
    def samples(self, value):
        value = int(value)
        if value < 1:
            raise ValueError(f"samples must be >= 1, got {value}")
        self._occlusion_pass._set_template_var(num_samples=value)

    @property
    def power(self):
        """Exponent applied to the occlusion."""
        return self._power

    @power.setter
    def power(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError(f"power must be > 0, got {value}")
        self._power = value

    @property
    def blur(self):
        """Radius of the bilateral blur in pixels; 0 disables it."""
        return self._blur

    @blur.setter
    def blur(self, value):
        if isinstance(value, bool):
            value = 2 if value else 0
        value = int(value)
        if value < 0:
            raise ValueError(f"blur must be >= 0, got {value}")
        self._blur = value
        if value:
            for p in self._blur_passes:
                p._set_template_var(blur_radius=value)

    @property
    def debug(self):
        """Whether to render the occlusion itself instead of the scene."""
        return self._debug

    @debug.setter
    def debug(self, value):
        self._debug = bool(value)

    def _create_textures(self, size):
        """(Re-)create the two off-screen occlusion textures."""
        self._ao_textures = tuple(
            self._device.create_texture(
                size=(size[0], size[1], 1),
                format=wgpu.TextureFormat.r16float,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT
                | wgpu.TextureUsage.TEXTURE_BINDING,
                dimension=wgpu.TextureDimension.d2,
            )
            for _ in range(2)
        )
        self._ao_views = tuple(t.create_view() for t in self._ao_textures)
        self._ao_size = size

    def render(self, command_encoder, color_tex, depth_tex, target_tex):
        if depth_tex is None:
            # Nothing was rendered with depth - pass the image through
            if self._copy_pass is None:
                self._copy_pass = CopyPass()
            self._copy_pass.render(command_encoder, color_tex, None, target_tex)
            return

        size = (color_tex.texture.size[0], color_tex.texture.size[1])
        if size != self._ao_size:
            self._create_textures(size)

        # The camera matrices are only final at render time
        proj = self.camera.projection_matrix.T
        proj_inv = self.camera.projection_matrix_inverse.T

        occlusion = self._occlusion_pass
        occlusion._uniform_data["projection_transform"] = proj
        occlusion._uniform_data["projection_transform_inv"] = proj_inv
        occlusion._uniform_data["radius"] = self.radius
        occlusion._uniform_data["bias"] = self.bias
        occlusion.render(
            command_encoder, depthTex=depth_tex, targetTex=self._ao_views[0]
        )

        # Separable blur: horizontal into the second texture, vertical back
        ao_view = self._ao_views[0]
        if self.blur:
            for p, dst in zip(self._blur_passes, self._ao_views[::-1]):
                p._uniform_data["projection_transform_inv"] = proj_inv
                # Surfaces further apart than the sampling radius are not
                # part of the same occlusion neighbourhood either
                p._uniform_data["depth_sigma"] = self.radius
                p.render(
                    command_encoder, aoTex=ao_view, depthTex=depth_tex, targetTex=dst
                )
                ao_view = dst

        composite = self._composite_pass
        composite._uniform_data["intensity"] = self.intensity
        composite._uniform_data["power"] = self.power
        composite._uniform_data["debug"] = float(self.debug)
        composite.render(
            command_encoder,
            colorTex=color_tex,
            aoTex=ao_view,
            targetTex=target_tex,
        )

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} radius={self.radius} "
            f"intensity={self.intensity} samples={self.samples} at {hex(id(self))}>"
        )
