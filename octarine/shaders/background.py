"""Custom pygfx material/shader for radial ("studio") gradient backgrounds.

`pygfx`'s stock `BackgroundMaterial` only does uniform colors and linear
(bottom-to-top or corner-to-corner) gradients. This module adds a radial
gradient with a freely positioned center, a three-stop color ramp and a
vignette - the kind of backdrop product/hero renders are usually shot
against. `BACKGROUND_PRESETS` holds a handful of ready-made looks.

Importing this module registers the shader with pygfx (via the
`register_wgpu_render_function` decorator).
"""

import pygfx as gfx
import wgpu  # only for flags/enums

from pygfx.renderers.wgpu import (
    register_wgpu_render_function,
    BaseShader,
    Binding,
    load_wgsl,
)

from ..utils import as_color_list

# Ready-made gradients. Colors are (inner, mid, outer), i.e. the center of the
# glow, the lift half-way out, and the color the gradient settles into. The
# `description` is what the GUI shows as tooltip; `from_preset` strips it.
BACKGROUND_PRESETS = {
    "graphite": dict(
        description="Neutral studio grey; the all-rounder that imposes no color identity",
        colors=("#34383A", "#171A1B", "#050607"),
        center=(0.43, 0.39),
        radius=0.65,
        falloff=3.0,
        vignette=0.55,
    ),
    "cinematic": dict(
        description='Desaturated blue-black; reads as "cool atmosphere", not "blue background"',
        colors=("#28343B", "#111A20", "#030507"),
        center=(0.42, 0.36),
        radius=0.70,
        falloff=2.5,
        vignette=0.2,
    ),
    "warm": dict(
        description="Warm charcoal; flatters brass, bronze, walnut, leather and warm whites",
        colors=("#3A3530", "#171411", "#070605"),
        center=(0.57, 0.38),
        radius=0.62,
        falloff=3.5,
        vignette=0.6,
    ),
    "olive": dict(
        # Kept bright because olive disappears into black surprisingly quickly
        description="Muted olive; architecture, furniture, ceramics, natural materials",
        colors=("#30362E", "#151914", "#050705"),
        center=(0.46, 0.34),
        radius=0.72,
        falloff=2.0,
        vignette=0.25,
    ),
    "burgundy": dict(
        description="Dusty burgundy; editorial/photographic without an obvious red cast",
        colors=("#3A292C", "#180F12", "#070405"),
        center=(0.55, 0.42),
        radius=0.62,
        falloff=2.5,
        vignette=0.4,
    ),
    "halo": dict(
        description="Near-black halo; just enough light to separate the silhouette",
        colors=("#252A2A", "#0B0D0D", "#010202"),
        center=(0.48, 0.35),
        radius=0.45,
        falloff=3.0,
        vignette=0.15,
    ),
}


class GradientBackgroundMaterial(gfx.Material):
    """A radial gradient background.

    The color ramp runs from `color_inner` at the `center`, through
    `color_mid` half-way out, to `color_outer` at (and beyond) `radius`.
    Distances are measured in units of the canvas *width*, so the gradient
    stays circular whatever the window's aspect ratio.

    Parameters
    ----------
    colors :    tuple
                Two or three colors: `(inner, mid, outer)` or - with the
                mid stop interpolated - `(inner, outer)`.
    center :    (x, y) tuple
                Center of the gradient in relative image coordinates:
                `(0, 0)` is the top left, `(1, 1)` the bottom right corner.
    radius :    float
                Distance at which the gradient reaches `color_outer`, as a
                fraction of the canvas width.
    falloff :   float
                Shape of the ramp. Values > 1 keep the core bright and push
                the transition towards the rim (3 ~ "the outer 30% of the
                radius"), 1 is linear, values < 1 drop off right at the
                center and trail out.
    vignette :  float
                Strength (0-1) of the additional darkening towards the
                corners of the frame. 0 disables it.
    kwargs :    Any
                Additional kwargs are passed to `pygfx.Material`.

    """

    uniform_type = dict(
        gfx.Material.uniform_type,
        color_inner="4xf4",
        color_mid="4xf4",
        color_outer="4xf4",
        center="2xf4",
        radius="f4",
        falloff="f4",
        vignette="f4",
    )

    def __init__(
        self,
        colors=("#34383A", "#171A1B", "#050607"),
        *,
        center=(0.5, 0.4),
        radius=0.65,
        falloff=3.0,
        vignette=0.5,
        alpha_mode="blend",
        depth_write=False,
        render_queue=1000,
        **kwargs,
    ):
        super().__init__(
            alpha_mode=alpha_mode,
            depth_write=depth_write,
            render_queue=render_queue,
            **kwargs,
        )
        self.set_colors(colors)
        self.center = center
        self.radius = radius
        self.falloff = falloff
        self.vignette = vignette
        # Name of the preset this material was made from; see `from_preset`
        self.preset = None

    @classmethod
    def from_preset(cls, preset, **kwargs):
        """Create a material from one of the `BACKGROUND_PRESETS`.

        Parameters
        ----------
        preset :    str | dict
                    Name of a preset (see `BACKGROUND_PRESETS`) or a dict of
                    parameters.
        kwargs :    Any
                    Override individual preset parameters.

        """
        if isinstance(preset, dict):
            params = dict(preset)
        else:
            if preset not in BACKGROUND_PRESETS:
                raise ValueError(
                    f"Unknown background preset '{preset}'. Available presets: "
                    f"{', '.join(BACKGROUND_PRESETS)}"
                )
            params = dict(BACKGROUND_PRESETS[preset])
        params.pop("description", None)  # docs/GUI only
        overrides = {k: v for k, v in kwargs.items() if v is not None}
        params.update(overrides)

        mat = cls(**params)
        # Remember the preset - the GUI uses this to reflect the current
        # state. Overriding any of its parameters makes it a custom gradient.
        mat.preset = preset if isinstance(preset, str) and not overrides else None
        return mat

    def set_colors(self, *colors):
        """Set the gradient's colors.

        Accepts three colors (inner, mid, outer) or two (inner, outer), in
        which case the mid stop is the average of the two. Colors can be
        passed as separate arguments or as a single sequence.
        """
        colors = as_color_list(*colors)
        if len(colors) == 3:
            inner, mid, outer = colors
        elif len(colors) == 2:
            inner, outer = colors
            # Blending in the middle of the ramp is a reasonable stand-in for
            # a hand-picked secondary color
            mid = gfx.Color([(a + b) / 2 for a, b in zip(inner.rgba, outer.rgba)])
        else:
            raise ValueError(f"Need 2 or 3 colors, got {len(colors)}.")
        self.color_inner = inner
        self.color_mid = mid
        self.color_outer = outer

    @property
    def colors(self):
        """The gradient's (inner, mid, outer) colors."""
        return (self.color_inner, self.color_mid, self.color_outer)

    @colors.setter
    def colors(self, colors):
        self.set_colors(colors)

    @property
    def color_inner(self):
        """Color at the center of the gradient."""
        return gfx.Color(self.uniform_buffer.data["color_inner"])

    @color_inner.setter
    def color_inner(self, color):
        self.uniform_buffer.data["color_inner"] = gfx.Color(color)
        self.uniform_buffer.update_full()

    @property
    def color_mid(self):
        """Color half-way between the center and `radius`."""
        return gfx.Color(self.uniform_buffer.data["color_mid"])

    @color_mid.setter
    def color_mid(self, color):
        self.uniform_buffer.data["color_mid"] = gfx.Color(color)
        self.uniform_buffer.update_full()

    @property
    def color_outer(self):
        """Color at (and beyond) `radius`."""
        return gfx.Color(self.uniform_buffer.data["color_outer"])

    @color_outer.setter
    def color_outer(self, color):
        self.uniform_buffer.data["color_outer"] = gfx.Color(color)
        self.uniform_buffer.update_full()

    @property
    def center(self):
        """Center of the gradient; (0, 0) is the top left corner."""
        return tuple(float(v) for v in self.uniform_buffer.data["center"])

    @center.setter
    def center(self, center):
        center = tuple(float(v) for v in center)
        if len(center) != 2:
            raise ValueError(f"center must be an (x, y) tuple, got {center}")
        self.uniform_buffer.data["center"] = center
        self.uniform_buffer.update_full()

    @property
    def radius(self):
        """Radius of the gradient as a fraction of the canvas width."""
        return float(self.uniform_buffer.data["radius"])

    @radius.setter
    def radius(self, radius):
        radius = float(radius)
        if radius <= 0:
            raise ValueError(f"radius must be > 0, got {radius}")
        self.uniform_buffer.data["radius"] = radius
        self.uniform_buffer.update_full()

    @property
    def falloff(self):
        """Shape of the radial ramp; > 1 keeps the core bright."""
        return float(self.uniform_buffer.data["falloff"])

    @falloff.setter
    def falloff(self, falloff):
        falloff = float(falloff)
        if falloff <= 0:
            raise ValueError(f"falloff must be > 0, got {falloff}")
        self.uniform_buffer.data["falloff"] = falloff
        self.uniform_buffer.update_full()

    @property
    def vignette(self):
        """Strength (0-1) of the darkening towards the corners."""
        return float(self.uniform_buffer.data["vignette"])

    @vignette.setter
    def vignette(self, vignette):
        vignette = float(vignette)
        if not 0 <= vignette <= 1:
            raise ValueError(f"vignette must be between 0 and 1, got {vignette}")
        self.uniform_buffer.data["vignette"] = vignette
        self.uniform_buffer.update_full()


@register_wgpu_render_function(gfx.Background, GradientBackgroundMaterial)
class GradientBackgroundShader(BaseShader):
    type = "render"

    def get_bindings(self, wobject, shared, scene):
        bindings = {
            0: Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            1: Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            2: Binding("u_material", "buffer/uniform", wobject.material.uniform_buffer),
        }
        self.define_bindings(0, bindings)

        return {
            0: bindings,
        }

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_strip,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        # A single screen-filling quad
        return {
            "indices": (4, 1),
        }

    def get_code(self):
        return load_wgsl("background_gradient.wgsl", "octarine.shaders.wgsl")
