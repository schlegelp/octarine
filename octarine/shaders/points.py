"""Points material with more flexible size/edge handling.

Extends pygfx's marker points with three features:

1. ``edge_size_space``: express ``edge_width`` in a different coordinate
   space than ``size``. E.g. points sized in world units but with an edge
   that is always 10 *screen* pixels wide.
2. ``min_size``/``max_size``: clamp the final on-screen point size (in
   logical pixels). E.g. "100 world units, but at least 10 pixels".
3. ``min_edge_width``: a floor for the final on-screen edge width (in
   logical pixels).
4. These combine freely: the space conversion happens per quantity,
   the clamps apply to the converted values.

Implemented by subclassing pygfx's points material/shader and rewriting the
small size/edge conversion block in pygfx's stock ``points.wgsl``. Importing
this module registers the shader with pygfx.
"""

import pygfx as gfx

from pygfx.renderers.wgpu import register_wgpu_render_function
from pygfx.renderers.wgpu.shaders.pointsshader import PointsShader
from pygfx.utils.enums import CoordSpace

# Largest finite f32; used as the "no maximum" sentinel for max_size.
_F32_MAX = 3.4028235e38


class FlexPointsMaterial(gfx.PointsMarkerMaterial):
    """A marker points material with flexible size/edge-width spaces.

    In addition to the properties of `pygfx.PointsMarkerMaterial`, this
    material supports:

    - `edge_size_space`: coordinate space for `edge_width` ("screen",
      "world" or "model"). If None (default), `edge_width` uses the same
      space as `size` (i.e. `size_space`), matching stock pygfx behavior.
    - `min_size` / `max_size`: clamp the final on-screen point size, in
      logical pixels. Useful with `size_space="world"` to keep far-away
      points visible ("world-sized but at least N pixels"). The clamp is
      applied after the size has been converted to screen pixels, so it
      also works with per-vertex sizes. If both are set, `max_size` wins.
    - `min_edge_width`: a floor for the final on-screen edge width, in
      logical pixels. Only applies when `edge_width` > 0 (i.e. it does
      not conjure up an edge that was disabled).

    """

    uniform_type = dict(
        gfx.PointsMarkerMaterial.uniform_type,
        min_size="f4",
        max_size="f4",
        min_edge_width="f4",
    )

    def __init__(
        self,
        *,
        edge_size_space=None,
        min_size=None,
        max_size=None,
        min_edge_width=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.edge_size_space = edge_size_space
        self.min_size = min_size
        self.max_size = max_size
        self.min_edge_width = min_edge_width

    @property
    def edge_size_space(self):
        """The coordinate space in which the edge width is expressed.

        One of "screen", "world" or "model" (see
        :obj:`pygfx.utils.enums.CoordSpace`), or None to use the same
        space as the point size (``size_space``).
        """
        return self._store.edge_size_space

    @edge_size_space.setter
    def edge_size_space(self, value):
        if value is not None and value not in CoordSpace:
            raise ValueError(
                f"edge_size_space must be None or a string in {CoordSpace}, "
                f"not {value!r}"
            )
        self._store.edge_size_space = value

    @property
    def min_size(self):
        """Minimum on-screen point size in logical pixels (0 = no minimum)."""
        return float(self.uniform_buffer.data["min_size"])

    @min_size.setter
    def min_size(self, value):
        value = 0.0 if value is None else float(value)
        if value < 0:
            raise ValueError(f"min_size must be >= 0, got {value}")
        self.uniform_buffer.data["min_size"] = value
        self.uniform_buffer.update_full()

    @property
    def max_size(self):
        """Maximum on-screen point size in logical pixels (None = no maximum)."""
        value = float(self.uniform_buffer.data["max_size"])
        return None if value >= 1e38 else value

    @max_size.setter
    def max_size(self, value):
        value = _F32_MAX if value is None else float(value)
        if value <= 0:
            raise ValueError(f"max_size must be > 0, got {value}")
        self.uniform_buffer.data["max_size"] = value
        self.uniform_buffer.update_full()

    @property
    def min_edge_width(self):
        """Minimum on-screen edge width in logical pixels (0 = no minimum)."""
        return float(self.uniform_buffer.data["min_edge_width"])

    @min_edge_width.setter
    def min_edge_width(self, value):
        value = 0.0 if value is None else float(value)
        if value < 0:
            raise ValueError(f"min_edge_width must be >= 0, got {value}")
        self.uniform_buffer.data["min_edge_width"] = value
        self.uniform_buffer.update_full()


# The three places in pygfx's points.wgsl (vs_main) that we rewrite. All are
# jinja-templated source; the replacements introduce a new `edge_size_space`
# template variable which our shader subclass defines.

# 1) The block converting `size_space` units to logical pixels via a single
#    shared `size_ratio`. Replaced with per-quantity ratios so size and edge
#    width can live in different spaces.
_RATIO_ANCHOR = """\
    $$ if size_space == 'screen'
        let size_ratio = 1.0;
    $$ else
        // The size is expressed in world space. So we first check where a point, moved shift_factor logical pixels away
        // from the node, ends up in world space. We actually do that for both x and y, in case there's anisotropy.
        let shift_factor = 1000.0;
        let pos_s_shiftedx = pos_s + vec2<f32>(shift_factor, 0.0);
        let pos_s_shiftedy = pos_s + vec2<f32>(0.0, shift_factor);
        let pos_n_shiftedx = vec4<f32>((pos_s_shiftedx / screen_factor - 1.0) * pos_n.w, pos_n.z, pos_n.w);
        let pos_n_shiftedy = vec4<f32>((pos_s_shiftedy / screen_factor - 1.0) * pos_n.w, pos_n.z, pos_n.w);
        let pos_w_shiftedx = u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv * pos_n_shiftedx;
        let pos_w_shiftedy = u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv * pos_n_shiftedy;
        $$ if size_space == 'model'
            // Transform back to model space
            let pos_m_shiftedx = u_wobject.world_transform_inv * pos_w_shiftedx;
            let pos_m_shiftedy = u_wobject.world_transform_inv * pos_w_shiftedy;
            // Distance in model space
            let size_ratio = (1.0 / shift_factor) * 0.5 * (distance(pos_m.xyz, pos_m_shiftedx.xyz) + distance(pos_m.xyz, pos_m_shiftedy.xyz));
        $$ else
            // Distance in world space
            let size_ratio = (1.0 / shift_factor) * 0.5 * (distance(pos_w.xyz, pos_w_shiftedx.xyz) + distance(pos_w.xyz, pos_w_shiftedy.xyz));
        $$ endif
    $$ endif"""

_RATIO_WGSL = """\
    $$ if size_space != 'screen' or edge_size_space != 'screen'
        // Check where a point, moved shift_factor logical pixels away from the
        // node, ends up in world space. Both x and y, in case of anisotropy.
        let shift_factor = 1000.0;
        let pos_s_shiftedx = pos_s + vec2<f32>(shift_factor, 0.0);
        let pos_s_shiftedy = pos_s + vec2<f32>(0.0, shift_factor);
        let pos_n_shiftedx = vec4<f32>((pos_s_shiftedx / screen_factor - 1.0) * pos_n.w, pos_n.z, pos_n.w);
        let pos_n_shiftedy = vec4<f32>((pos_s_shiftedy / screen_factor - 1.0) * pos_n.w, pos_n.z, pos_n.w);
        let pos_w_shiftedx = u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv * pos_n_shiftedx;
        let pos_w_shiftedy = u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv * pos_n_shiftedy;
    $$ endif
    $$ if size_space == 'model' or edge_size_space == 'model'
        // Transform back to model space; ratio is distance in model space
        let pos_m_shiftedx = u_wobject.world_transform_inv * pos_w_shiftedx;
        let pos_m_shiftedy = u_wobject.world_transform_inv * pos_w_shiftedy;
        let model_ratio = (1.0 / shift_factor) * 0.5 * (distance(pos_m.xyz, pos_m_shiftedx.xyz) + distance(pos_m.xyz, pos_m_shiftedy.xyz));
    $$ endif
    $$ if size_space == 'world' or edge_size_space == 'world'
        // Distance in world space
        let world_ratio = (1.0 / shift_factor) * 0.5 * (distance(pos_w.xyz, pos_w_shiftedx.xyz) + distance(pos_w.xyz, pos_w_shiftedy.xyz));
    $$ endif
    $$ if size_space == 'model'
        let size_ratio = model_ratio;
    $$ elif size_space == 'world'
        let size_ratio = world_ratio;
    $$ else
        let size_ratio = 1.0;
    $$ endif
    $$ if edge_size_space == 'model'
        let edge_ratio = model_ratio;
    $$ elif edge_size_space == 'world'
        let edge_ratio = world_ratio;
    $$ else
        let edge_ratio = 1.0;
    $$ endif"""

# 2) The edge width conversion: use the edge's own ratio and floor the
#    result at min_edge_width - but only when there is an edge at all, so a
#    floor cannot conjure up an edge that was disabled with edge_width=0.
_EDGE_ANCHOR = "        let edge_width = u_material.edge_width / size_ratio;  // expressed in logical screen pixels"
_EDGE_WGSL = "        let edge_width = select(0.0, max(u_material.edge_width / edge_ratio, u_material.min_edge_width), u_material.edge_width > 0.0);  // expressed in logical screen pixels"

# 3) The size conversion (aa and non-aa variants): clamp the converted size.
#    min(max(...)) instead of clamp() so that max_size deterministically wins
#    if min_size > max_size.
_SIZE_AA_ANCHOR = "        let size:f32 = size_ref / size_ratio;  // Logical pixels"
_SIZE_AA_WGSL = "        let size:f32 = min(max(size_ref / size_ratio, u_material.min_size), u_material.max_size);  // Logical pixels"

_SIZE_NOAA_ANCHOR = "        let size:f32 = max(min_size_for_pixel, size_ref / size_ratio);  // non-aa don't get smaller."
_SIZE_NOAA_WGSL = "        let size:f32 = max(min_size_for_pixel, min(max(size_ref / size_ratio, u_material.min_size), u_material.max_size));  // non-aa don't get smaller."

_REPLACEMENTS = [
    (_RATIO_ANCHOR, _RATIO_WGSL),
    (_EDGE_ANCHOR, _EDGE_WGSL),
    (_SIZE_AA_ANCHOR, _SIZE_AA_WGSL),
    (_SIZE_NOAA_ANCHOR, _SIZE_NOAA_WGSL),
]


@register_wgpu_render_function(gfx.Points, FlexPointsMaterial)
class FlexPointsShader(PointsShader):
    """Points shader with per-quantity size spaces and a pixel-size clamp."""

    def __init__(self, wobject):
        super().__init__(wobject)
        material = wobject.material
        # None means "follow size_space" (= stock pygfx behavior)
        self["edge_size_space"] = material.edge_size_space or material.size_space

    def get_code(self):
        code = super().get_code()
        for anchor, replacement in _REPLACEMENTS:
            if code.count(anchor) != 1:
                import pygfx

                raise RuntimeError(
                    "octarine's flexible points shader could not find its "
                    f"injection point in pygfx {pygfx.__version__}'s points "
                    "shader - the two are incompatible. Please open an issue "
                    "at https://github.com/schlegelp/octarine/issues"
                )
            code = code.replace(anchor, replacement)
        return code
