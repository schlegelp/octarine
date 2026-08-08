"""Screen-space outlines for the viewer.

Draws a line wherever the geometry has an edge - around the silhouette of
each object, and along the creases within it - the way a technical
illustration or a comic would. Beyond looking good, this does real work in a
crowded scene: overlapping objects of similar color become individually
readable, because every one of them is now bounded by a line.

`pygfx` has nothing of the sort, so the effect is implemented here as an
``EffectPass``. Like the ambient occlusion pass it works purely from the
depth buffer, and detects two kinds of edge:

1. **Depth edges.** For each neighbouring pixel we ask how far it lies from
   the tangent plane of the surface under the center pixel. On a flat (or
   smoothly curved) surface that distance is tiny no matter how steeply the
   surface is tilted away from the camera; where one surface ends and
   another begins, it jumps. Comparing against the plane rather than
   against the depth itself is what keeps floors and other grazing surfaces
   from being outlined along their whole length.
2. **Normal edges.** Where the surface normal turns sharply but the depth
   stays continuous - the fold of a crease. Normals are reconstructed from
   the depth buffer (see `octarine.shaders.depth_wgsl`).

Note that as a screen-space effect this applies to the entire rendered image
- including overlay elements such as messages - and that objects which do
not write depth (e.g. meshes with transparent alpha modes) are neither
outlined nor occlude an outline.
"""

import pygfx as gfx

from pygfx.renderers.wgpu import CopyPass, EffectPass

from .depth_wgsl import DEPTH_WGSL, VIEW_POS_WGSL, VIEW_NORMAL_WGSL


class OutlinePass(EffectPass):
    """An outline (edge detection) post-processing pass.

    Finds silhouettes and creases in the depth buffer and draws them in
    `color`.

    Parameters
    ----------
    camera :    pygfx.Camera
                The camera used to render the scene. Needed to map depth
                values back to view-space positions.
    color :     str | tuple
                Color of the outline. The alpha channel doubles as the
                strength of the effect, i.e. "#0004" gives a subtle line.
    thickness : float
                Width of the outline in physical pixels. Values above ~4
                start to look chunky rather than drawn.
    depth_threshold : float
                How far a neighbouring pixel has to lie off the tangent
                plane of the surface to count as a separate object,
                relative to its distance from the camera. Lower it to
                outline shallower steps, raise it if surfaces get outlined
                across their interior.
    normal_threshold : float
                How sharply the surface has to fold to count as a crease,
                as `1 - cos(angle)`: 0.3 is roughly 45 degrees. Set to 0 to
                switch creases off and outline only silhouettes.
    debug :     bool
                If True, render the edges themselves as white on black
                instead of drawing them over the scene.

    """

    USES_DEPTH = True

    uniform_type = dict(
        EffectPass.uniform_type,
        projection_transform_inv="4x4xf4",
        color="4xf4",
        ortho_scale="f4",
        depth_threshold="f4",
        normal_threshold="f4",
        debug="f4",
    )

    wgsl = (
        DEPTH_WGSL
        + VIEW_POS_WGSL
        + VIEW_NORMAL_WGSL
        + """
        const RADIUS: i32 = {{ radius }};

        // The eight directions the outline grows in. Every radius from 1 to
        // RADIUS is tested along each of them, so a thick outline is a
        // filled disc rather than a ring of dots.
        const DIRECTIONS = array<vec2i, 8>(
            vec2i(1, 0), vec2i(-1, 0), vec2i(0, 1), vec2i(0, -1),
            vec2i(1, 1), vec2i(1, -1), vec2i(-1, 1), vec2i(-1, -1),
        );

        @fragment
        fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
            let dims = vec2i(textureDimensions(depthTex));
            let tex_index = vec2i(varyings.position.xy);
            let color = textureLoad(colorTex, tex_index, 0);
            let depth = textureLoad(depthTex, tex_index, 0);

            let is_background = depth >= 1.0;
            var edge = 0.0;

            // How much of a step counts as an edge. For a perspective
            // camera that scales with the distance to the camera (things
            // twice as far away have half the depth resolution on screen);
            // for an orthographic one - where pygfx parks the camera in
            // the middle of the scene, so the distance is meaningless -
            // it scales with the visible height of the view instead.
            var pos = vec3f(0.0);
            var normal = vec3f(0.0);
            var scale = u_effect.ortho_scale;
            if (!is_background) {
                pos = to_view_pos(varyings.texCoord, depth);
                normal = view_normal_at(tex_index, depth, pos);
                if (scale <= 0.0) {
                    scale = max(abs(pos.z), 1e-6);
                }
            }
            let depth_threshold = u_effect.depth_threshold * scale;

            for (var d = 0; d < 8; d += 1) {
                let dir = DIRECTIONS[d];
                for (var r = 1; r <= RADIUS; r += 1) {
                    let idx = clamp(tex_index + dir * r, vec2i(0), dims - 1);
                    let n_depth = textureLoad(depthTex, idx, 0);
                    let n_is_background = n_depth >= 1.0;

                    // Geometry against empty space: the outer silhouette,
                    // and the one edge that is always worth drawing
                    if (n_is_background != is_background) {
                        edge = 1.0;
                        continue;
                    }
                    if (is_background) {
                        continue;  // both are background
                    }

                    let n_pos = to_view_pos(
                        (vec2f(idx) + 0.5) / vec2f(dims), n_depth);

                    // Distance of the neighbour from the tangent plane at
                    // the center. Near zero across a smooth surface however
                    // steeply it is tilted; large where a surface ends.
                    let plane_dist = abs(dot(normal, n_pos - pos));
                    edge = max(edge, smoothstep(
                        depth_threshold, depth_threshold * 2.0, plane_dist));

                    // Creases. Only tested at the outer end of each
                    // direction: reconstructing a normal costs a handful of
                    // depth samples, and a fold thick enough to matter is
                    // picked up there just as well.
                    if (u_effect.normal_threshold > 0.0 && r == RADIUS) {
                        let n_normal = view_normal_at(idx, n_depth, n_pos);
                        let fold = 1.0 - dot(normal, n_normal);
                        edge = max(edge, smoothstep(
                            u_effect.normal_threshold,
                            u_effect.normal_threshold * 2.0,
                            fold));
                    }
                }
            }

            if (u_effect.debug > 0.5) {
                return vec4f(edge, edge, edge, 1.0);
            }

            // `u_effect.color` is already physical (see the `color` setter);
            // effect passes run on the linear image, before gamma
            let strength = edge * u_effect.color.a;
            return vec4f(mix(color.rgb, u_effect.color.rgb, strength), color.a);
        }
    """
    )

    def __init__(
        self,
        camera,
        color="#000",
        thickness=1.0,
        depth_threshold=0.02,
        normal_threshold=0.3,
        debug=False,
    ):
        super().__init__()
        self.camera = camera
        self.color = color
        self.thickness = thickness
        self.depth_threshold = depth_threshold
        self.normal_threshold = normal_threshold
        self.debug = debug
        self._copy_pass = None  # only used if there is no depth buffer

    @property
    def color(self):
        """Color of the outline; its alpha is the strength of the effect."""
        return self._color

    @color.setter
    def color(self, value):
        self._color = gfx.Color(value)
        # Effect passes see the linear (physical) image - the sRGB encode
        # only happens in pygfx's final composition step - so the color has
        # to go into the uniform linearized. The shaders' `srgb2physical()`
        # is not available here; this is the same conversion.
        rgb = [
            c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
            for c in self._color.rgb
        ]
        self._uniform_data["color"] = (*rgb, self._color.a)

    @property
    def thickness(self):
        """Width of the outline in physical pixels."""
        return self._thickness

    @thickness.setter
    def thickness(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError(f"thickness must be > 0, got {value}")
        self._thickness = value
        # The shader walks out one pixel at a time, so the radius has to be
        # a (small) integer; changing it recompiles the shader
        self._set_template_var(radius=max(1, min(int(round(value)), 8)))

    @property
    def depth_threshold(self):
        """Relative step in depth that counts as a separate object."""
        return float(self._uniform_data["depth_threshold"])

    @depth_threshold.setter
    def depth_threshold(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError(f"depth_threshold must be > 0, got {value}")
        self._uniform_data["depth_threshold"] = value

    @property
    def normal_threshold(self):
        """How sharp a fold counts as a crease, as `1 - cos(angle)`; 0 = off."""
        return float(self._uniform_data["normal_threshold"])

    @normal_threshold.setter
    def normal_threshold(self, value):
        value = float(value)
        if not 0 <= value <= 2:
            raise ValueError(f"normal_threshold must be in [0, 2], got {value}")
        self._uniform_data["normal_threshold"] = value

    @property
    def debug(self):
        """Whether to render the edges themselves instead of the scene."""
        return bool(self._uniform_data["debug"])

    @debug.setter
    def debug(self, value):
        self._uniform_data["debug"] = float(bool(value))

    def render(self, command_encoder, color_tex, depth_tex, target_tex):
        if depth_tex is None:
            # Nothing was rendered with depth - pass the image through
            if self._copy_pass is None:
                self._copy_pass = CopyPass()
            self._copy_pass.render(command_encoder, color_tex, None, target_tex)
            return

        # The camera matrices are only final at render time
        self._uniform_data["projection_transform_inv"] = (
            self.camera.projection_matrix_inverse.T
        )
        if getattr(self.camera, "fov", 0) == 0:
            # Orthographic: scale the threshold by the visible view height
            zoom = getattr(self.camera, "zoom", 1) or 1
            self._uniform_data["ortho_scale"] = (
                abs(float(self.camera.height) / zoom) or 1.0
            )
        else:
            self._uniform_data["ortho_scale"] = 0.0

        super().render(command_encoder, color_tex, depth_tex, target_tex)

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} color={self.color} "
            f"thickness={self.thickness} at {hex(id(self))}>"
        )
