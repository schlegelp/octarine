"""Subsurface scattering (translucency) for meshes.

Approximates light that enters a surface, bounces around inside it and
leaves again towards the viewer. Two terms are added on top of pygfx's
Phong shading:

 - *Transmission* (Barre-Brisebois & Bouchard, GDC 2011): light travelling
   roughly *through* the object, which shows up when a light sits behind
   what you are looking at. This is what makes ears, leaves, wax and thin
   neurites glow when backlit.
 - *Wrapped diffuse*: light is allowed to bleed past the terminator (the
   `dot(normal, light) = 0` line), so shading eases into shadow instead of
   ending abruptly - the characteristic soft look of skin or marble.

Both are tinted by `scatter_color`, the color light picks up on its way
through the material (a deep red for flesh, for instance).

Thickness is a *constant* here, not a per-vertex quantity, so the effect
cannot know that an ear is thinner than a skull. That is a deliberate
trade: it needs no preprocessing, no extra vertex buffers and no extra
render passes - just one more loop over the lights.

The material derives from `SilhouetteMeshMaterial` so that the two effects
compose (both are off by default and cost nothing when off); see
`_SSS_HELPER_WGSL` below for how the terms are spliced into pygfx's stock
`mesh.wgsl`. Importing this module registers the shader with pygfx.
"""

import pygfx as gfx

from pygfx.renderers.wgpu import register_wgpu_render_function

from .silhouette import SilhouetteMeshMaterial, SilhouetteMeshShader

#: The tunable properties of `SubsurfaceMeshMaterial`, excluding the master
#: `subsurface` strength. Shared with `Viewer.set_subsurface` and
#: `visuals.mesh2gfx` so that all three validate the same names.
SUBSURFACE_PROPERTIES = (
    "scatter_color",
    "thickness",
    "distortion",
    "falloff",
    "wrap",
    "glow",
)


class SubsurfaceMeshMaterial(SilhouetteMeshMaterial):
    """A Phong mesh material with subsurface scattering (translucency).

    In addition to the properties of `pygfx.MeshPhongMaterial` (and the
    `silhouette` exponent of `SilhouetteMeshMaterial`, which is off by
    default here), this material scatters light through the surface.

    Parameters
    ----------
    subsurface :    float
                    Master strength of the effect; 0 disables it entirely.
                    Typical values are 0.5-2.
    scatter_color : str | tuple
                    The color light picks up while travelling through the
                    material. Defaults to a warm red.
    thickness :     float
                    How much material the light has to cross, in [0, 1].
                    Constant across the mesh - lower it for thin, papery
                    objects, raise it for chunky ones.
    distortion :    float
                    How far the transmitted light direction is bent along
                    the surface normal, in [0, 1]. 0 makes the glow appear
                    only under near-exact backlighting; higher values wrap
                    it around the silhouette.
    falloff :       float
                    Exponent of the transmission lobe (>= 1). Higher values
                    tighten the glow around the light direction.
    wrap :          float
                    How far diffuse light bleeds past the terminator, in
                    [0, 1]. 0 leaves pygfx's Lambertian shading untouched.
    glow :          float
                    A view-independent floor added to the transmission,
                    giving translucent objects a slight overall lift.
                    (Called "ambient" in the original formulation; renamed
                    to avoid confusion with `pygfx.AmbientLight`.)

    Notes
    -----
    The transmission term deliberately ignores shadows: light that scatters
    through an object is exactly the light that did *not* reach the surface
    directly, so attenuating it by a shadow lookup would cancel the effect.

    """

    uniform_type = dict(
        SilhouetteMeshMaterial.uniform_type,
        sss_color="4xf4",
        sss_weight="f4",
        sss_thickness="f4",
        sss_distortion="f4",
        sss_falloff="f4",
        sss_wrap="f4",
        sss_glow="f4",
    )

    def __init__(
        self,
        subsurface=1.0,
        scatter_color="#a03028",
        thickness=0.5,
        distortion=0.2,
        falloff=4.0,
        wrap=0.5,
        glow=0.0,
        silhouette=0.0,
        **kwargs,
    ):
        # Note the silhouette default of 0: inherited as-is it would be 1,
        # and asking for subsurface scattering should not quietly also turn
        # the mesh into an x-ray view of itself
        super().__init__(silhouette=silhouette, **kwargs)
        self.subsurface = subsurface
        self.scatter_color = scatter_color
        self.thickness = thickness
        self.distortion = distortion
        self.falloff = falloff
        self.wrap = wrap
        self.glow = glow

    def _set_positive(self, field, name, value, minimum=0.0):
        value = float(value)
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}, got {value}")
        self.uniform_buffer.data[field] = value
        self.uniform_buffer.update_full()

    @property
    def subsurface(self):
        """Master strength of the scattering; 0 disables the effect."""
        return float(self.uniform_buffer.data["sss_weight"])

    @subsurface.setter
    def subsurface(self, value):
        self._set_positive("sss_weight", "subsurface", value)

    @property
    def scatter_color(self):
        """The color light picks up while travelling through the material."""
        return gfx.Color(self.uniform_buffer.data["sss_color"])

    @scatter_color.setter
    def scatter_color(self, color):
        self.uniform_buffer.data["sss_color"] = gfx.Color(color)
        self.uniform_buffer.update_full()

    @property
    def thickness(self):
        """How much material the light has to cross, in [0, 1]."""
        return float(self.uniform_buffer.data["sss_thickness"])

    @thickness.setter
    def thickness(self, value):
        self._set_positive("sss_thickness", "thickness", value)

    @property
    def distortion(self):
        """How far transmitted light bends along the normal, in [0, 1]."""
        return float(self.uniform_buffer.data["sss_distortion"])

    @distortion.setter
    def distortion(self, value):
        self._set_positive("sss_distortion", "distortion", value)

    @property
    def falloff(self):
        """Exponent of the transmission lobe; higher tightens the glow."""
        return float(self.uniform_buffer.data["sss_falloff"])

    @falloff.setter
    def falloff(self, value):
        # pow() with an exponent below 1 blows the lobe up across the whole
        # surface, which reads as a washed-out fog rather than translucency
        self._set_positive("sss_falloff", "falloff", value, minimum=1.0)

    @property
    def wrap(self):
        """How far diffuse light bleeds past the terminator, in [0, 1]."""
        return float(self.uniform_buffer.data["sss_wrap"])

    @wrap.setter
    def wrap(self, value):
        self._set_positive("sss_wrap", "wrap", value)

    @property
    def glow(self):
        """View-independent floor added to the transmission."""
        return float(self.uniform_buffer.data["sss_glow"])

    @glow.setter
    def glow(self, value):
        self._set_positive("sss_glow", "glow", value)


# --- WGSL ------------------------------------------------------------------
#
# Three splices into pygfx's stock mesh.wgsl. Note that `get_code()` runs
# *before* the `{$ include ... $}` directives are resolved, so we cannot
# reach into `light_phong.wgsl`'s RE_Direct(); instead we add our own pass
# over the lights next to pygfx's. Our snippets are templated along with
# everything else, hence the `$$ if num_*_lights` guards.

# Module-scope helper. Placed before pygfx's own ReflectedLight struct - it
# needs the types and helpers from light_common.wgsl, which mesh.wgsl has
# included by this point.
_HELPER_ANCHOR = "struct ReflectedLight {"

_SSS_HELPER_WGSL = """\
struct SssParams {
    tint: vec3<f32>,
    thickness: f32,
    distortion: f32,
    falloff: f32,
    wrap: f32,
    glow: f32,
};

fn sss_contribution(
    light: IncidentLight,
    geometry: GeometricContext,
    albeido: vec3<f32>,
    p: SssParams,
) -> vec3<f32> {
    // Transmission: light that entered the far side and scattered through
    // towards the viewer. Pushing the light vector along the normal
    // ("distortion") widens the lobe so it wraps around the silhouette
    // instead of only lighting up under exact backlighting.
    let lt_dir = normalize(light.direction + geometry.normal * p.distortion);
    let lt_dot = pow(saturate(dot(geometry.view_dir, -lt_dir)), p.falloff);
    let transmitted = (lt_dot + p.glow) * p.thickness;

    // Wrapped diffuse, expressed as the *extra* light relative to the
    // Lambertian term pygfx has already accumulated for this light, so the
    // two add up to a wrapped diffuse rather than double-counting it.
    let dot_nl = dot(geometry.normal, light.direction);
    let wrapped = saturate((dot_nl + p.wrap) / (1.0 + p.wrap)) - saturate(dot_nl);

    return light.color * p.tint * (
        albeido * transmitted + BRDF_Lambert(albeido) * wrapped
    );
}

struct ReflectedLight {"""

# Our pass over the lights, right after pygfx's. Deliberately not merged
# into pygfx's loop: that one attenuates each light by its shadow lookup,
# and transmitted light must not be.
_LIGHTS_ANCHOR = """\
        // Punctual light
        {$ include 'pygfx.light_punctual.wgsl' $}"""

_SSS_LIGHTS_WGSL = """\
        // Punctual light
        {$ include 'pygfx.light_punctual.wgsl' $}

        // Subsurface scattering (octarine)
        var sss_color = vec3<f32>(0.0);
        if (u_material.sss_weight > 0.0) {
            var sss_params: SssParams;
            sss_params.tint = srgb2physical(u_material.sss_color.rgb);
            sss_params.thickness = u_material.sss_thickness;
            sss_params.distortion = u_material.sss_distortion;
            sss_params.falloff = u_material.sss_falloff;
            sss_params.wrap = u_material.sss_wrap;
            sss_params.glow = u_material.sss_glow;

            $$ if num_point_lights > 0
            for (var i = 0; i < {{num_point_lights}}; i ++ ) {
                let sss_light = getPointLightInfo(u_point_lights[i], geometry);
                if (sss_light.visible) {
                    sss_color += sss_contribution(sss_light, geometry, physical_albeido, sss_params);
                }
            }
            $$ endif

            $$ if num_spot_lights > 0
            for (var i = 0; i < {{num_spot_lights}}; i ++ ) {
                let sss_light = getSpotLightInfo(u_spot_lights[i], geometry);
                if (sss_light.visible) {
                    sss_color += sss_contribution(sss_light, geometry, physical_albeido, sss_params);
                }
            }
            $$ endif

            $$ if num_dir_lights > 0
            for (var i = 0; i < {{num_dir_lights}}; i ++ ) {
                let sss_light = getDirectionalLightInfo(u_directional_lights[i], geometry);
                if (sss_light.visible) {
                    sss_color += sss_contribution(sss_light, geometry, physical_albeido, sss_params);
                }
            }
            $$ endif

            sss_color *= u_material.sss_weight;
        }"""

# Where the lit color is composed. `sss_color` is declared by the snippet
# above, which lives in the same `$$ if lighting` branch - hence the guard.
_COMPOSE_ANCHOR = (
    "    var physical_color = reflected_light.direct_diffuse + "
    "reflected_light.direct_specular + reflected_light.indirect_diffuse + "
    "reflected_light.indirect_specular;"
)

_SSS_COMPOSE_WGSL = f"""\
{_COMPOSE_ANCHOR}
    $$ if lighting
    physical_color += sss_color;
    $$ endif"""


@register_wgpu_render_function(gfx.Mesh, SubsurfaceMeshMaterial)
class SubsurfaceMeshShader(SilhouetteMeshShader):
    """Phong mesh shader with subsurface scattering (and silhouette) spliced in."""

    def get_code(self):
        # The silhouette shader has already spliced its own term in
        code = super().get_code()
        for anchor, replacement in (
            (_HELPER_ANCHOR, _SSS_HELPER_WGSL),
            (_LIGHTS_ANCHOR, _SSS_LIGHTS_WGSL),
            (_COMPOSE_ANCHOR, _SSS_COMPOSE_WGSL),
        ):
            if code.count(anchor) != 1:
                import pygfx

                raise RuntimeError(
                    "octarine's subsurface scattering shader could not find its "
                    f"injection point in pygfx {pygfx.__version__}'s mesh shader - "
                    "the two are incompatible. Please open an issue at "
                    "https://github.com/schlegelp/octarine/issues"
                )
            code = code.replace(anchor, replacement)
        return code
