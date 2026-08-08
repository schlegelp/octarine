"""Tone mapping and exposure for the viewer.

pygfx renders into a floating point buffer, so the colors reaching the screen
are not limited to [0, 1]: a specular highlight, an emissive surface or - in
particular - anything lit by an environment map (see
`octarine.shaders.environment`) routinely goes well above white. The final
composition step simply clips those values, which is why bright regions turn
into flat white blobs and lose all their color: as soon as the red channel
clips first, a warm highlight reads as pure red before it reads as white.

Tone mapping is the fix. It maps the whole (open-ended) range of rendered
values into what the display can show, with a curve that compresses the
highlights gradually instead of cutting them off. `exposure` scales the image
before the curve is applied, i.e. it is the photographic exposure control:
raise it to lift a dark scene, lower it to recover detail in blown-out
highlights.

The pass runs late - after effects such as bloom, which want the untouched
high dynamic range values, and before pygfx's own anti-aliasing and gamma
handling, which want display-referred ones.
"""

import numpy as np

from pygfx.renderers.wgpu import EffectPass

#: The available tone mapping curves.
TONEMAP_MODES = ("none", "reinhard", "aces", "filmic")

# Each mode maps a linear color `c` (already multiplied by the exposure) to a
# display-referred one. They differ mostly in how much contrast they keep
# while doing it - see the class docstring.
_CURVES = {
    "none": """
        // Clip only. Useful as a baseline, and for when the tone mapping is
        // wanted purely as an exposure control.
        return clamp(c, vec3f(0.0), vec3f(1.0));
    """,
    "reinhard": """
        // Extended Reinhard (Reinhard et al. 2002): the gentlest of the
        // three. `white_point` is the value that maps to exactly 1.0, so
        // anything brighter is what actually clips.
        let w = max(u_effect.white_point, 1e-3);
        let mapped = c * (1.0 + c / (w * w)) / (1.0 + c);
        return clamp(mapped, vec3f(0.0), vec3f(1.0));
    """,
    "aces": """
        // Narkowicz' (2015) curve fit to the ACES filmic response. Punchier
        // than Reinhard - it darkens the shadows and saturates as it rolls
        // off, the way film does.
        let a = 2.51; let b = 0.03; let cc = 2.43; let d = 0.59; let e = 0.14;
        return clamp((c * (a * c + b)) / (c * (cc * c + d) + e),
                     vec3f(0.0), vec3f(1.0));
    """,
    "filmic": """
        // Hable's "Uncharted 2" curve. Similar in spirit to ACES but with a
        // longer toe, i.e. it holds on to shadow detail for longer.
        let scale = 1.0 / hable(vec3f(max(u_effect.white_point, 1e-3))).x;
        return clamp(hable(c) * scale, vec3f(0.0), vec3f(1.0));
    """,
}

_HABLE_WGSL = """
    fn hable(x: vec3f) -> vec3f {
        let A = 0.15; let B = 0.50; let C = 0.10;
        let D = 0.20; let E = 0.02; let F = 0.30;
        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
    }
"""


class ToneMappingPass(EffectPass):
    """A tone mapping (and exposure) post-processing pass.

    Compresses the rendered high dynamic range image into the range the
    display can show, so that bright regions roll off smoothly instead of
    clipping to flat white.

    Parameters
    ----------
    mode :      str
                The tone mapping curve:
                 - "aces" (default): a fit to the ACES filmic response.
                   Contrasty and saturated; the usual choice.
                 - "filmic": Hable's "Uncharted 2" curve. Like ACES but
                   holds more shadow detail.
                 - "reinhard": the gentlest option. Keeps colors closest to
                   the original at the cost of looking flatter.
                 - "none": clip only, i.e. exposure control on its own.
    exposure :  float
                Scales the image before the curve is applied: 2 is one stop
                brighter, 0.5 one stop darker.
    white_point : float
                The input value that maps to white. Only used by "reinhard"
                and "filmic"; raising it holds on to more highlight detail
                (and darkens the image overall).

    """

    uniform_type = dict(
        EffectPass.uniform_type,
        exposure="f4",
        white_point="f4",
    )

    # `{{ curve }}` is filled in by the `mode` setter below
    wgsl = (
        _HABLE_WGSL
        + """
        fn tonemap(c: vec3f) -> vec3f {
            {{ curve }}
        }

        @fragment
        fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
            let tex_index = vec2i(varyings.position.xy);
            let color = textureLoad(colorTex, tex_index, 0);
            let exposed = max(color.rgb * u_effect.exposure, vec3f(0.0));
            return vec4f(tonemap(exposed), color.a);
        }
    """
    )

    def __init__(self, mode="aces", exposure=1.0, white_point=4.0):
        super().__init__()
        self.mode = mode
        self.exposure = exposure
        self.white_point = white_point

    @property
    def mode(self):
        """The tone mapping curve; one of `TONEMAP_MODES`."""
        return self._mode

    @mode.setter
    def mode(self, value):
        value = str(value).lower()
        if value not in _CURVES:
            raise ValueError(
                f"Unknown tone mapping mode '{value}'. Must be one of "
                f"{', '.join(TONEMAP_MODES)}."
            )
        self._mode = value
        # Swapping the curve recompiles the shader, which is why this is a
        # template variable rather than a uniform: the curve is picked once
        # and then runs without a branch per pixel.
        self._set_template_var(curve=_CURVES[value])

    @property
    def exposure(self):
        """Scales the image before the curve is applied."""
        return float(self._uniform_data["exposure"])

    @exposure.setter
    def exposure(self, value):
        value = float(value)
        if value < 0:
            raise ValueError(f"exposure must be >= 0, got {value}")
        self._uniform_data["exposure"] = value

    @property
    def stops(self):
        """The exposure expressed in stops; 0 leaves the image unchanged."""
        return float(np.log2(self.exposure)) if self.exposure > 0 else -np.inf

    @stops.setter
    def stops(self, value):
        self.exposure = 2.0 ** float(value)

    @property
    def white_point(self):
        """The input value that maps to white ("reinhard" and "filmic" only)."""
        return float(self._uniform_data["white_point"])

    @white_point.setter
    def white_point(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError(f"white_point must be > 0, got {value}")
        self._uniform_data["white_point"] = value

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} mode={self.mode!r} "
            f"exposure={self.exposure} at {hex(id(self))}>"
        )
