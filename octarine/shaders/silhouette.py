"""Silhouette rendering for meshes, as popularized by Neuroglancer.

The effect multiplies each fragment's color and alpha by
``pow(1 - |dot(normal, view_dir)|, silhouette)``: face-on regions become
transparent while edges/creases light up, giving an x-ray-like view of the
mesh's outline. A silhouette of 0 disables the effect; typical values are
1-8 (same exponent semantics as Neuroglancer's "silhouette" property).

Implemented by subclassing pygfx's Phong material/shader and injecting the
silhouette term into pygfx's stock ``mesh.wgsl`` right before the final
color is composed. Importing this module registers the shader with pygfx.
"""

import pygfx as gfx

from pygfx.renderers.wgpu import register_wgpu_render_function
from pygfx.renderers.wgpu.shaders.meshshader import MeshPhongShader


class SilhouetteMeshMaterial(gfx.MeshPhongMaterial):
    """A Phong mesh material with a Neuroglancer-style silhouette effect.

    In addition to the properties of `pygfx.MeshPhongMaterial`, this material
    has a `silhouette` exponent: fragments are multiplied by
    ``pow(1 - |dot(normal, view_dir)|, silhouette)``, making face-on regions
    transparent and emphasizing edges/creases. 0 disables the effect;
    typical values are 1-8.

    Note that for the transparency to composite sensibly the material should
    use a transparent `alpha_mode` (octarine defaults to "weighted_blend").

    """

    uniform_type = dict(
        gfx.MeshPhongMaterial.uniform_type,
        silhouette="f4",
    )

    def __init__(self, silhouette=1.0, **kwargs):
        super().__init__(**kwargs)
        self.silhouette = silhouette

    @property
    def silhouette(self):
        """Silhouette exponent; 0 disables the effect."""
        return float(self.uniform_buffer.data["silhouette"])

    @silhouette.setter
    def silhouette(self, value):
        value = float(value)
        if value < 0:
            raise ValueError(f"silhouette must be >= 0, got {value}")
        self.uniform_buffer.data["silhouette"] = value
        self.uniform_buffer.update_full()


# The line in pygfx's mesh.wgsl (fs_main) where the final fragment color is
# composed; `physical_color`, `diffuse_color`, `surface_normal` and
# `varyings.world_pos` are all live at this point.
_ANCHOR = "    let out_color = vec4<f32>(physical_color, diffuse_color.a);"

_SILHOUETTE_WGSL = """\
    var out_color = vec4<f32>(physical_color, diffuse_color.a);
    if (u_material.silhouette > 0.0) {
        let sil_view = select(
            normalize(u_stdinfo.cam_transform_inv[3].xyz - varyings.world_pos),
            (u_stdinfo.cam_transform_inv * vec4<f32>(0.0, 0.0, 1.0, 0.0)).xyz,
            is_orthographic());
        let sil_cos = abs(dot(normalize(surface_normal), sil_view));
        out_color = out_color * pow(1.0 - sil_cos, u_material.silhouette);
    }"""


@register_wgpu_render_function(gfx.Mesh, SilhouetteMeshMaterial)
class SilhouetteMeshShader(MeshPhongShader):
    """Phong mesh shader with the silhouette term spliced in."""

    def get_code(self):
        code = super().get_code()
        if code.count(_ANCHOR) != 1:
            import pygfx

            raise RuntimeError(
                "octarine's silhouette shader could not find its injection "
                f"point in pygfx {pygfx.__version__}'s mesh shader - the two "
                "are incompatible. Please open an issue at "
                "https://github.com/schlegelp/octarine/issues"
            )
        return code.replace(_ANCHOR, _SILHOUETTE_WGSL)
