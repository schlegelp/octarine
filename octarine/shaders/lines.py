"""Line material with per-node line widths.

pygfx's `LineMaterial` has a single, uniform `thickness`. This module adds a
`thickness_mode` ("uniform" or "vertex") that mirrors what
`pygfx.PointsMaterial.size_mode` does for points: in "vertex" mode the width
is read per node from `geometry.thicknesses`, which gives lines that taper
(think neuron radii or streamlines).

The width is interpolated along each segment: a segment between two nodes of
different width is drawn as a trapezoid, and both the geometry and the
anti-aliasing in the fragment shader follow from the same (linear) varying,
so the edge stays crisp. Joins and caps use the width of the node they sit
on. Dashes are the one thing that keeps using the uniform `thickness`: the
dash pattern is expressed in units of line width and a per-node unit would
make the pattern stretch and squeeze along the line.

Implemented by subclassing pygfx's line material/shader and rewriting the
three lines in pygfx's stock ``line.wgsl`` that read the thickness from the
material uniform. Importing this module registers the shader with pygfx.
"""

import pygfx as gfx

from pygfx.renderers.wgpu import Binding, register_wgpu_render_function
from pygfx.renderers.wgpu.shaders.lineshader import LineShader


class FlexLineMaterial(gfx.LineMaterial):
    """A line material that can take its width from the geometry.

    In addition to the properties of `pygfx.LineMaterial`, this material
    supports:

    - `thickness_mode`: "uniform" (default, i.e. stock pygfx behavior) uses
      the material's `thickness` for the whole line; "vertex" takes the
      width per node from `geometry.thicknesses` (a 1d float32 buffer with
      one value per position). Widths are in the same units as `thickness`,
      i.e. they follow `thickness_space`.

    Note that `thickness` is still used to scale the dash pattern, also in
    "vertex" mode.

    """

    def __init__(self, *, thickness_mode="uniform", **kwargs):
        super().__init__(**kwargs)
        self.thickness_mode = thickness_mode

    @property
    def thickness_mode(self):
        """How the line width is determined: "uniform" or "vertex"."""
        return self._store.thickness_mode

    @thickness_mode.setter
    def thickness_mode(self, value):
        value = "uniform" if value is None else str(value)
        if value not in ("uniform", "vertex"):
            raise ValueError(
                f"thickness_mode must be 'uniform' or 'vertex', got {value!r}"
            )
        self._store.thickness_mode = value


# The three places in pygfx's line.wgsl (vs_main) that we rewrite. All are
# jinja-templated source; the replacements introduce a new `thickness_mode`
# template variable which our shader subclass defines.

# 1) Introduce `thickness_ref`: either the uniform or the node's own width.
#    Injected just before the thickness is first used. This mirrors what
#    pygfx's points.wgsl does with `size_ref`/`load_s_sizes`.
_REF_ANCHOR = "    let min_size_for_pixel = 1.415 / l2p;  // For minimum pixel coverage. Use sqrt(2) to take diagonals into account."
_REF_WGSL = """\
    $$ if thickness_mode == 'vertex'
    let thickness_ref = load_s_thicknesses(node_index);
    $$ else
    let thickness_ref = u_material.thickness;
    $$ endif
    let min_size_for_pixel = 1.415 / l2p;  // For minimum pixel coverage. Use sqrt(2) to take diagonals into account."""

# 2) + 3) The aa and non-aa thickness conversions. Everything downstream
#    (joins, caps, the thickness varying) is derived from these, so these two
#    lines are all it takes to make the width per-node.
_AA_ANCHOR = (
    "    let thickness:f32 = u_material.thickness / thickness_ratio;  // Logical pixels"
)
_AA_WGSL = "    let thickness:f32 = thickness_ref / thickness_ratio;  // Logical pixels"

_NOAA_ANCHOR = "    let thickness:f32 = max(min_size_for_pixel, u_material.thickness / thickness_ratio);  // non-aa lines get no thinner than 1 px"
_NOAA_WGSL = "    let thickness:f32 = max(min_size_for_pixel, thickness_ref / thickness_ratio);  // non-aa lines get no thinner than 1 px"

_REPLACEMENTS = [
    (_REF_ANCHOR, _REF_WGSL),
    (_AA_ANCHOR, _AA_WGSL),
    (_NOAA_ANCHOR, _NOAA_WGSL),
]


@register_wgpu_render_function(gfx.Line, FlexLineMaterial)
class FlexLineShader(LineShader):
    """Line shader that can read the line width per node."""

    def __init__(self, wobject):
        super().__init__(wobject)
        self["thickness_mode"] = wobject.material.thickness_mode

    def get_bindings(self, wobject, shared, scene):
        bindings_by_group = super().get_bindings(wobject, shared, scene)

        if self["thickness_mode"] == "vertex":
            thicknesses = getattr(wobject.geometry, "thicknesses", None)
            if thicknesses is None:
                raise ValueError(
                    "thickness_mode='vertex' requires the geometry to have a "
                    "`thicknesses` buffer (one width per position)."
                )
            if thicknesses.nitems != wobject.geometry.positions.nitems:
                raise ValueError(
                    f"Got {thicknesses.nitems} line widths for "
                    f"{wobject.geometry.positions.nitems} positions."
                )
            # Append to the bindings pygfx already set up (group 0)
            bindings = bindings_by_group[0]
            index = len(bindings)
            binding = Binding(
                "s_thicknesses", "buffer/read_only_storage", thicknesses, "VERTEX"
            )
            bindings[index] = binding
            self.define_binding(0, index, binding)

        return bindings_by_group

    def get_code(self):
        code = super().get_code()
        for anchor, replacement in _REPLACEMENTS:
            if code.count(anchor) != 1:
                import pygfx

                raise RuntimeError(
                    "octarine's per-node line width shader could not find its "
                    f"injection point in pygfx {pygfx.__version__}'s line "
                    "shader - the two are incompatible. Please open an issue "
                    "at https://github.com/schlegelp/octarine/issues"
                )
            code = code.replace(anchor, replacement)
        return code
