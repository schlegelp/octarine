"""Matcap ("material capture") shading for meshes.

A matcap is a picture of a sphere, shaded however you like, that is used as
a lookup table: the surface normal at each pixel - expressed in *view* space -
picks a point on that sphere, and whatever color is there becomes the color of
the pixel. Everything the sphere shows (the diffuse falloff, the highlights,
the rim light, even the illusion of reflections) comes along for the ride,
without a single light in the scene.

That makes matcaps a staple of scientific and sculpting viewers:

 - Shading is completely independent of the scene's lights, so meshes look
   the same in every scene and cannot end up under- or overlit.
 - Surface shape reads extremely well, because the whole normal hemisphere is
   mapped onto a hand-tuned gradient rather than a Lambert term.
 - It costs one texture lookup - no lights are evaluated at all.

The trade-off is that the shading is locked to the camera: it rotates with
the view, so a matcap cannot express where a light actually *is* in the
scene, and it casts and receives nothing.

`pygfx` has no matcap material, so this module adds one by subclassing
`MeshBasicMaterial` (which has no lighting to get in the way) and splicing a
matcap lookup into pygfx's stock ``mesh.wgsl``. The matcap images themselves
are generated procedurally - `MATCAP_PRESETS` are recipes, not pictures - by
lighting a virtual sphere with the same analytic environment model that
`octarine.shaders.environment` uses for image-based lighting, so a matcap and
an environment built from the same preset agree on where the light is coming
from. A preset can equally carry a lighting rig of its own, which is what the
ones reproducing Blender's matcaps do.

Importing this module registers the shader with pygfx.
"""

import numpy as np
import pygfx as gfx

from pygfx.renderers.wgpu import register_wgpu_render_function
from pygfx.renderers.wgpu.shaders.meshshader import MeshShader

from ..utils import as_color_list
from .environment import (
    environment_radiance,
    _light_solid_angle,
    _physical2srgb,
    _srgb2physical,
    _rgb,
    _resolve_params,
)

# Ready-made matcaps. `environment` is the lighting setup the sphere is lit
# with - either the name of one of
# `octarine.shaders.environment.ENVIRONMENT_PRESETS` or a rig of its own -
# and everything else describes the surface. `tint` is how much of an object's
# color survives - 1 keeps it (so a scene of differently colored objects still
# reads as such), 0 lets the matcap's own color take over, which is what the
# strongly colored presets want. `description` is the GUI's tooltip.
MATCAP_PRESETS = {
    "pearl": dict(
        description="Neutral glossy white; shows shape without imposing a color",
        environment="studio",
        base_color="#e6e7ea",
        specular=0.5,
        shininess=70,
        rim=0.22,
        rim_color="#ffffff",
        rim_power=4.0,
        tint=1.0,
    ),
    "clay": dict(
        description="Matte modelling clay; no highlights, pure form",
        environment="soft",
        base_color="#c08a70",
        specular=0.0,
        shininess=8,
        rim=0.08,
        rim_color="#ffd9c6",
        rim_power=3.0,
        tint=0.6,
    ),
    "metal": dict(
        description="Brushed steel; hard highlight and a strong rim",
        environment="studio",
        base_color="#70767e",
        specular=1.8,
        shininess=200,
        rim=0.8,
        rim_color="#cfe0ff",
        rim_power=5.0,
        tint=0.3,
    ),
    "gold": dict(
        description="Warm polished metal, lit by a low sun",
        environment="sunset",
        base_color="#c8912e",
        specular=0.9,
        shininess=150,
        rim=0.35,
        rim_color="#ffd98a",
        rim_power=4.0,
        tint=0.0,
    ),
    "jade": dict(
        description="Deep green stone with a translucent glowing rim",
        environment="soft",
        base_color="#1e5c42",
        specular=0.6,
        shininess=90,
        rim=0.55,
        rim_color="#7dffb5",
        rim_power=3.0,
        tint=0.0,
    ),
    "neon": dict(
        description="Near-black surface with magenta/cyan edges; for dark backgrounds",
        environment="neon",
        base_color="#20222c",
        specular=1.2,
        shininess=120,
        rim=1.4,
        rim_color="#ff59c7",
        rim_power=2.5,
        tint=0.4,
    ),
    # The five below are reproductions of matcaps from Blender's Workbench
    # renderer (basic_side, ceramic_lightbulb, ceramic_dark, toon_dark and
    # toon_light). Nothing is copied - the parameters were fitted to the
    # originals, so these are the closest this model gets to them rather than
    # pixel-exact matches. They bring their own lighting rig instead of naming
    # one of the shared environments, because a rig built to light a single
    # sphere is not necessarily a good environment to light a scene with.
    "sidelit": dict(
        description="Plain grey under one big side light; unfussy and very readable",
        environment=dict(
            sky="#8a8a8a",
            horizon="#535353",
            ground="#010101",
            gradient=2.3,
            lights=(
                dict(direction=(0.6, 0.72, 0.36), color="#ffffff", intensity=12.5,
                     angle=22, softness=0.95),
            ),
        ),
        base_color="#e3e3e3",
        specular=0.1,
        shininess=8,
        rim=0.0,
        tint=1.0,
    ),
    "ceramic": dict(
        description="Cool glazed ceramic with long strip-light reflections",
        environment=dict(
            sky="#ebe8ff",
            horizon="#52373a",
            ground="#000000",
            gradient=1.05,
            lights=(
                # The two strip lights whose reflections make the streaks
                dict(direction=(0.59, 0.74, -0.33), color="#ffffff", intensity=5.9,
                     angle=15, length=38, roll=78, softness=1.0),
                dict(direction=(-0.6, 0.79, -0.17), color="#ffffff", intensity=10.2,
                     angle=1.5, length=10, roll=66, softness=1.0),
                # Bounce off the floor in front, for the bright lower edge
                dict(direction=(0.17, -0.48, 0.86), color="#ffffff", intensity=0.4,
                     angle=72, softness=1.0),
            ),
        ),
        base_color="#beccce",
        specular=0.16,
        shininess=280,
        rim=0.0,
        tint=0.8,
    ),
    "slate": dict(
        description="Muted blue-grey ceramic with a small warm key light",
        environment=dict(
            sky="#766350",
            horizon="#49535c",
            ground="#4e463c",
            gradient=1.15,
            lights=(
                dict(direction=(-0.54, 0.59, 0.59), color="#a3a3ad", intensity=6.2,
                     angle=16, softness=0.3),
                dict(direction=(-0.71, 0.22, 0.66), color="#a9b2c3", intensity=3.4,
                     angle=50, softness=1.0),
                # The warm highlight, from behind the shoulder
                dict(direction=(-0.54, 0.83, -0.13), color="#fac587", intensity=14.7,
                     angle=11, softness=0.75),
            ),
        ),
        base_color="#aeb1b4",
        specular=0.09,
        shininess=80,
        rim=0.0,
        tint=0.5,
    ),
    "toon": dict(
        description="Cel shading: flat tones, hard terminators and an ink outline",
        environment=dict(
            sky="#2a2a2a",
            horizon="#2a2a2a",
            ground="#111111",
            gradient=0.4,
            # One small, hard light: with the tones quantized it is where the
            # terminators land that matters, not how smooth the falloff is
            lights=(
                dict(direction=(-0.34, 0.5, 0.79), color="#ffffff", intensity=65.0,
                     angle=7.0, softness=0.2),
            ),
        ),
        base_color="#dedede",
        specular=0.01,
        shininess=300,
        rim=0.35,
        rim_color="#3fe3ee",
        rim_power=9.0,
        bands=3,
        band_softness=0.03,
        edge=0.95,
        edge_width=0.03,
        tint=1.0,
    ),
    "toon_light": dict(
        description="Pale cel shading, for light backgrounds",
        environment=dict(
            sky="#999999",
            horizon="#999999",
            ground="#3d3d3d",
            gradient=3.0,
            lights=(
                dict(direction=(0.53, 0.11, 0.84), color="#ffffff", intensity=0.8,
                     angle=85, softness=0.05),
            ),
        ),
        base_color="#e6e6e6",
        specular=0.0,
        rim=0.0,
        # High-key lighting only covers the top of the range, so ten bands
        # show up as the three barely separated tones this one lives on
        bands=10,
        band_softness=0.03,
        edge=0.2,
        edge_width=0.03,
        tint=1.0,
    ),
}

#: The properties of a matcap recipe, i.e. what `make_matcap` accepts as
#: overrides on top of a preset. Shared with `Viewer.set_matcap` so that both
#: validate the same names.
MATCAP_PROPERTIES = (
    "environment",
    "base_color",
    "specular",
    "shininess",
    "rim",
    "rim_color",
    "rim_power",
    "bands",
    "band_softness",
    "edge",
    "edge_width",
    "tint",
)


def _sky_irradiance(elevations, environment, samples=2048):
    """Irradiance of an environment's sky gradient, per normal elevation.

    The gradient part of the environment model depends only on how high up a
    direction points, so the irradiance it produces depends only on how high
    up the *normal* points. That makes this a one-dimensional problem: we
    integrate over the sphere once for each of a few hundred elevations and
    interpolate in between, instead of integrating for every pixel.

    Parameters
    ----------
    elevations : (N,) array
                The y components of the normals to evaluate, in [-1, 1].
    environment : dict
                Environment parameters, with the softboxes already removed
                (they are handled analytically, see `make_matcap`).
    samples :   int
                Number of directions used for the integration.

    Returns
    -------
    (N, 3) array
                Linear irradiance.

    """
    # Fibonacci sphere: N roughly evenly spaced directions, each covering the
    # same solid angle (4*pi/N)
    i = np.arange(samples, dtype=np.float64) + 0.5
    y = 1 - 2 * i / samples
    radius = np.sqrt(np.maximum(1 - y * y, 0))
    theta = np.pi * (1 + 5**0.5) * i
    directions = np.stack(
        [np.cos(theta) * radius, y, np.sin(theta) * radius], axis=-1
    ).astype(np.float32)

    radiance = environment_radiance(directions, environment)  # (samples, 3)
    solid_angle = 4 * np.pi / samples

    # The normals only need to differ in elevation, so put them all in the
    # y/z plane; `cos_weights` is then (len(elevations), samples)
    elevations = np.clip(np.asarray(elevations, dtype=np.float32), -1, 1)
    normals = np.stack(
        [np.zeros_like(elevations), elevations, np.sqrt(1 - elevations**2)], axis=-1
    )
    cos_weights = np.clip(normals @ directions.T, 0, None)
    return (cos_weights @ radiance) * solid_angle


def _softbox_irradiance(light, cos_theta, samples=512):
    """Irradiance from one softbox, per cosine of the angle to its axis.

    Treating a softbox as a point light and multiplying by the cosine is
    only right for a small one. A big source keeps lighting a surface after
    its center has dropped below the horizon, which is exactly what makes
    the terminator of a studio-lit sphere so much softer than a Lambert
    falloff - so we integrate over the source instead.

    The result depends only on the angle between the normal and the light's
    axis, which makes this a one-dimensional table, the same trick
    `_sky_irradiance` uses.

    Parameters
    ----------
    light :     dict
                A softbox; see `octarine.shaders.environment`.
    cos_theta : (N,) array
                Cosines of the angle between normal and light axis.
    samples :   int
                Number of directions used for the integration.

    Returns
    -------
    (N,) array
                Irradiance per unit radiance, i.e. multiply by the light's
                color and intensity to get the contribution.

    """
    # Tubes are treated as the round softbox that covers the same amount of
    # sky. The shape of an area source barely shows in its diffuse falloff -
    # only its size does - and it does show in the reflection, which is
    # handled by the environment model itself.
    angle = float(np.arccos(np.clip(1 - _light_solid_angle(light) / (2 * np.pi), -1, 1)))
    softness = float(np.clip(light.get("softness", 0.8), 0, 1))

    # Directions covering the cap, each carrying the same solid angle
    i = np.arange(samples, dtype=np.float64) + 0.5
    cos_a = 1 - (i / samples) * (1 - np.cos(angle))
    sin_a = np.sqrt(np.maximum(1 - cos_a * cos_a, 0))
    phi = np.pi * (1 + 5**0.5) * i
    directions = np.stack([sin_a * np.cos(phi), sin_a * np.sin(phi), cos_a], axis=-1)

    # The same soft edge `environment_radiance` gives the disc
    cos_outer, cos_inner = np.cos(angle), np.cos(angle * (1 - softness))
    if cos_inner - cos_outer < 1e-6:
        falloff = np.ones(samples)
    else:
        t = np.clip((cos_a - cos_outer) / (cos_inner - cos_outer), 0, 1)
        falloff = t * t * (3 - 2 * t)
    weights = falloff * (2 * np.pi * (1 - np.cos(angle)) / samples)

    cos_theta = np.clip(np.asarray(cos_theta, dtype=np.float64), -1, 1)
    sin_theta = np.sqrt(np.maximum(1 - cos_theta * cos_theta, 0))
    normals = np.stack([sin_theta, np.zeros_like(cos_theta), cos_theta], axis=-1)
    return np.clip(normals @ directions.T, 0, None) @ weights


#: Rec. 709 luminance weights; used to posterize without shifting hues.
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _posterize(shading, bands, softness):
    """Quantize the shading into `bands` flat tones - the cel-shading step.

    This runs on the *shading* rather than on the finished color, because
    that is what a cel painter quantizes: the light, not the paint. The
    surface keeps its own color and the top tone is whatever that color is,
    instead of always being white.

    The quantization itself happens on the luminance (so hues survive) and
    in display space rather than in linear space, because equal steps of the
    sRGB value are what read as equally spaced tones. `softness` is the
    width of the transition as a fraction of a band: 0 gives the hard
    terminator of a cartoon, 1 blurs the steps into each other.
    """
    bands = int(bands)
    if bands < 2:
        raise ValueError(f"bands must be >= 2 (or 0 for smooth), got {bands}")

    lum = shading @ _LUMA
    encoded = _physical2srgb(lum) * (bands - 1)
    step = np.floor(encoded)
    frac = encoded - step

    half = np.clip(float(softness), 0, 1) * 0.5
    if half <= 0:
        frac = (frac >= 0.5).astype(np.float32)
    else:
        frac = _smoothstep((frac - (0.5 - half)) / (2 * half))

    # Clamped, so that `bands` really does mean that many tones however
    # bright the lighting is. The highlight and the rim are added afterwards
    # and stay open-ended, which is where the dynamic range lives.
    quantized = _srgb2physical(np.clip(step + frac, 0, bands - 1) / (bands - 1))
    return shading * (quantized / np.maximum(lum, 1e-6))[..., None]


def make_matcap(preset="pearl", *, size=256, environment=None, **overrides):
    """Render a matcap image by lighting a virtual sphere.

    Parameters
    ----------
    preset :    str | dict
                Name of an entry in `MATCAP_PRESETS` (e.g. "pearl", "clay",
                "metal", "gold", "jade" or "neon"), or a dict of the
                properties below.
    size :      int
                Width/height of the generated image in pixels. 256 is ample -
                a matcap is a very smooth image.
    environment : str | dict, optional
                Lighting setup to shade the sphere with; see
                `octarine.shaders.environment.ENVIRONMENT_PRESETS`. Defaults
                to whatever the matcap preset asks for.
    **overrides
                Individual `base_color`, `specular`, `shininess`, `rim`,
                `rim_color`, `rim_power`, `bands`, `band_softness`, `edge`,
                `edge_width` or `tint` values overriding the preset's.

    Returns
    -------
    (size, size, 4) float32 array
                Linear (physical) RGBA. Highlights may well exceed 1 - the
                image is high dynamic range, like the environments it is
                lit with.

    """
    params = _resolve_params(preset, presets=MATCAP_PRESETS,
                             properties=MATCAP_PROPERTIES,
                             kind="matcap", environment=environment, **overrides)

    size = int(size)
    if size < 8:
        raise ValueError(f"size must be >= 8, got {size}")

    # The normals of a sphere seen head-on. Row 0 is the top of the image,
    # so y runs from +1 down to -1.
    t = (np.arange(size, dtype=np.float32) + 0.5) / size * 2 - 1
    x, y = np.meshgrid(t, -t)
    r2 = x * x + y * y

    # Outside the disc we keep going with the silhouette normal (z = 0)
    # rather than leaving the corners undefined: bilinear filtering reaches
    # just past the rim, and would otherwise drag background into it.
    scale = np.where(r2 > 1, 1 / np.sqrt(np.maximum(r2, 1e-12)), 1.0)
    nx, ny = x * scale, y * scale
    nz = np.sqrt(np.maximum(1 - np.minimum(r2, 1), 0))
    normals = np.stack([nx, ny, nz], axis=-1)

    # We look at the sphere head-on, i.e. along -z
    view_dir = np.array([0, 0, 1], dtype=np.float32)

    env = _resolve_params(params.get("environment", "studio"))
    lights = env.pop("lights", ())
    env_intensity = float(env.get("intensity", 1.0))

    # --- Diffuse -----------------------------------------------------------
    # Both halves of the environment are integrated numerically, but each
    # along the one axis it actually varies over: the sky gradient by
    # elevation, a softbox by the angle to its own axis.
    cosines = np.linspace(-1, 1, 257, dtype=np.float32)
    table = _sky_irradiance(cosines, env)
    irradiance = np.stack(
        [np.interp(normals[..., 1], cosines, table[:, c]) for c in range(3)],
        axis=-1,
    ).astype(np.float32)

    for light in lights:
        direction = np.asarray(light["direction"], dtype=np.float32)
        direction = direction / np.linalg.norm(direction)
        radiance = (
            _rgb(light.get("color", "#ffffff"))
            * float(light.get("intensity", 1.0))
            * env_intensity
        )
        shading = np.interp(
            normals @ direction, cosines, _softbox_irradiance(light, cosines)
        )
        irradiance = irradiance + radiance * shading[..., None]

    shading = irradiance / np.pi

    # --- Cel shading -------------------------------------------------------
    # Posterizing the diffuse (and only the diffuse - the highlight and the
    # rim stay smooth, as they do in a hand-painted cel) turns the gradient
    # into the flat tones with hard terminators of a cartoon.
    bands = params.get("bands", 0)
    if bands:
        shading = _posterize(shading, bands, params.get("band_softness", 0.1))

    color = _rgb(params.get("base_color", "#e6e7ea")) * shading

    # --- Specular ----------------------------------------------------------
    # Rather than adding a Blinn-Phong dot per light, we reflect the
    # environment itself: what a shiny surface shows is the *shape* of the
    # softboxes and the sky behind them, which is the difference between a
    # matcap that looks photographed and one that looks like three white
    # dots. `shininess` blurs the reflection by widening each softbox (and
    # dimming it to match, so that the total energy stays put).
    specular = float(params.get("specular", 0.5))
    if specular:
        shininess = max(float(params.get("shininess", 70)), 1.0)
        # Angular width of a Blinn-Phong lobe of this exponent, doubled
        # because a half-vector spread of x spreads reflections by 2x
        blur = 2 * np.arccos(0.5 ** (1 / shininess))

        blurred = dict(env, lights=[])
        for light in lights:
            angle = np.radians(float(light.get("angle", 25.0)))
            # Combine source size and lobe width in quadrature
            wide = float(np.hypot(angle, blur))
            spread = dict(light, angle=np.degrees(wide))
            blurred["lights"].append(
                dict(
                    spread,
                    # Spreading the same light over more sky dims it
                    intensity=float(light.get("intensity", 1.0))
                    * _light_solid_angle(light)
                    / _light_solid_angle(spread),
                    softness=float(
                        np.clip(max(light.get("softness", 0.8), blur / wide), 0, 1)
                    ),
                )
            )

        reflection = 2 * (normals @ view_dir)[..., None] * normals - view_dir
        color = color + environment_radiance(reflection, blurred) * specular

    # --- Rim ---------------------------------------------------------------
    # The bright edge where the surface turns away from us. Matcaps lean on
    # this heavily: it is what separates an object from the background when
    # there is no lighting to do it.
    rim = float(params.get("rim", 0.3))
    if rim:
        rim_color = _rgb(params.get("rim_color", "#ffffff"))
        rim_power = max(float(params.get("rim_power", 4.0)), 0.1)
        facing = np.clip(normals @ view_dir, 0, 1)[..., None]
        color = color + rim_color * rim * (1 - facing) ** rim_power

    # --- Ink line ----------------------------------------------------------
    # A dark band right at the silhouette, where the normals turn away from
    # the camera. It reads as a drawn outline around the object - the classic
    # companion to cel shading, and useful on its own to separate pale
    # objects from a pale background.
    edge = float(params.get("edge", 0.0))
    if edge:
        edge_width = float(params.get("edge_width", 0.05))
        if not 0 < edge_width <= 1:
            raise ValueError(f"edge_width must be in (0, 1], got {edge_width}")
        ink = _smoothstep((np.sqrt(r2) - (1 - edge_width)) / edge_width)
        color = color * (1 - edge * ink)[..., None]

    out = np.ones((size, size, 4), dtype=np.float32)
    out[..., :3] = color
    return out


def matcap_texture(matcap="pearl", *, size=256, **kwargs):
    """Build a matcap texture, ready to be assigned to `MatcapMeshMaterial`.

    Parameters
    ----------
    matcap :    str | dict | array | pygfx.Texture | pygfx.TextureMap
                A preset name or recipe dict (see `make_matcap`), or an
                image to use directly: an (N, M, 3) or (N, M, 4) array of
                floats (linear) or uint8 (sRGB), which is what an
                off-the-shelf matcap PNG looks like once loaded.
    size :      int
                Size of the generated image; ignored if `matcap` is already
                an image.
    **kwargs
                Passed on to `make_matcap`; ignored if `matcap` is already
                an image.

    Returns
    -------
    pygfx.TextureMap

    """
    if isinstance(matcap, gfx.TextureMap):
        return matcap
    if isinstance(matcap, gfx.Texture):
        return gfx.TextureMap(matcap, filter="linear", wrap="clamp")

    if isinstance(matcap, (str, dict)):
        image = make_matcap(matcap, size=size, **kwargs)
        preset = matcap if isinstance(matcap, str) and not kwargs else None
    else:
        image = np.asarray(matcap)
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(
                "A matcap image must be an (N, M, 3) or (N, M, 4) array, got "
                f"shape {image.shape}."
            )
        if image.dtype == np.uint8:
            # An 8-bit image is sRGB-encoded by convention; the shader
            # decodes it again, so it can be uploaded as-is
            image = image.astype(np.float32) / 255
            image = np.where(image <= 0.04045, image / 12.92,
                             ((image + 0.055) / 1.055) ** 2.4)
        image = image.astype(np.float32)
        if image.shape[2] == 3:
            image = np.concatenate(
                [image, np.ones(image.shape[:2] + (1,), np.float32)], axis=-1
            )
        preset = None

    # As with the environment maps, pygfx's shaders run every color through
    # `srgb2physical()`, so we hand it sRGB-encoded data. Keeping it float
    # means highlights can exceed 1 and still feed bloom and tone mapping.
    data = np.empty(image.shape, dtype=np.float16)
    data[..., :3] = _physical2srgb(image[..., :3])
    data[..., 3] = image[..., 3]

    texture = gfx.Texture(np.ascontiguousarray(data), dim=2)
    # "clamp" keeps the rim from wrapping around to the opposite edge
    tex_map = gfx.TextureMap(texture, filter="linear", wrap="clamp")
    tex_map.preset = preset
    return tex_map


class MatcapMeshMaterial(gfx.MeshBasicMaterial):
    """A mesh material shaded by a matcap instead of by the scene's lights.

    The surface normal in view space looks up a color in the `matcap` image
    (a picture of a shaded sphere), which becomes the color of the fragment.
    The scene's lights, shadows and ambient occlusion play no part in it.

    In addition to the properties of `pygfx.MeshBasicMaterial`:

    Parameters
    ----------
    matcap :    str | dict | array | pygfx.Texture | pygfx.TextureMap
                The matcap. A preset name or recipe dict is rendered on the
                spot (see `make_matcap` and `MATCAP_PRESETS`); an image is
                used as-is.
    tint :      float
                How much of the material's own `color` (and any vertex
                colors) tints the matcap, from 0 (the matcap's colors win)
                to 1 (fully multiplied in). Neutral matcaps are usually
                worth tinting - it keeps differently colored objects
                distinguishable - while strongly colored ones are not.
    kwargs :    Any
                Additional kwargs are passed to `pygfx.MeshBasicMaterial`.

    """

    uniform_type = dict(
        gfx.MeshBasicMaterial.uniform_type,
        matcap_tint="f4",
    )

    def __init__(self, matcap="pearl", tint=1.0, **kwargs):
        super().__init__(**kwargs)
        self.matcap = matcap
        self.tint = tint

    @property
    def matcap(self):
        """The matcap image, as a `pygfx.TextureMap`."""
        return self._store.matcap

    @matcap.setter
    def matcap(self, matcap):
        self._store.matcap = None if matcap is None else matcap_texture(matcap)

    @property
    def tint(self):
        """How much the material's own color tints the matcap (0-1)."""
        return float(self.uniform_buffer.data["matcap_tint"])

    @tint.setter
    def tint(self, value):
        value = float(value)
        if not 0 <= value <= 1:
            raise ValueError(f"tint must be in [0, 1], got {value}")
        self.uniform_buffer.data["matcap_tint"] = value
        self.uniform_buffer.update_full()


# The line in pygfx's mesh.wgsl (fs_main) where the final fragment color is
# composed. For a basic material `physical_color` is just the albedo at this
# point, which is exactly what we want to replace (and to tint by).
_ANCHOR = "    let out_color = vec4<f32>(physical_color, diffuse_color.a);"

_MATCAP_WGSL = """\
    $$ if use_matcap is defined
    // Matcap (octarine): look the shading up in a picture of a sphere,
    // indexed by the surface normal in view space.
    var mc_normal = normalize(surface_normal);
    mc_normal = select(-mc_normal, mc_normal, is_front);
    let mc_normal_view = normalize(
        (u_stdinfo.cam_transform * vec4<f32>(mc_normal, 0.0)).xyz);

    // Direction from the surface to the camera, in view space. Building the
    // lookup frame around it (rather than around the view axis) keeps the
    // highlight from sliding off objects at the edge of a wide-angle view.
    let mc_view_pos = (u_stdinfo.cam_transform * vec4<f32>(varyings.world_pos, 1.0)).xyz;
    let mc_view_dir = select(
        normalize(-mc_view_pos), vec3<f32>(0.0, 0.0, 1.0), is_orthographic());
    let mc_x = normalize(vec3<f32>(mc_view_dir.z, 0.0, -mc_view_dir.x));
    let mc_y = cross(mc_view_dir, mc_x);

    // 0.495 rather than 0.5 keeps the lookup a hair inside the sphere, so
    // that filtering at the very rim has something to interpolate with
    let mc_uv = vec2<f32>(dot(mc_x, mc_normal_view), dot(mc_y, mc_normal_view))
        * 0.495 + 0.5;
    let mc_color = textureSample(t_matcap, s_matcap, vec2<f32>(mc_uv.x, 1.0 - mc_uv.y));

    let mc_tint = mix(vec3<f32>(1.0), physical_color, u_material.matcap_tint);
    physical_color = srgb2physical(mc_color.rgb) * mc_tint;
    $$ endif

    let out_color = vec4<f32>(physical_color, diffuse_color.a);"""


@register_wgpu_render_function(gfx.Mesh, MatcapMeshMaterial)
class MatcapMeshShader(MeshShader):
    """Basic mesh shader with the matcap lookup spliced in."""

    def get_bindings(self, wobject, shared, scene):
        result = super().get_bindings(wobject, shared, scene)

        matcap = wobject.material.matcap
        if matcap is not None:
            # Bind group 2 is free for mesh materials (pygfx's own standard
            # material uses it for the same purpose). `check=False` because
            # the lookup is computed from the normal, not from texcoords.
            bindings = self._define_texture_map(
                wobject.geometry, matcap, "matcap", check=False
            )
            bindings = {i: binding for i, binding in enumerate(bindings)}
            self.define_bindings(2, bindings)
            result[2] = bindings
            self["use_matcap"] = True

        return result

    def get_code(self):
        code = super().get_code()
        if code.count(_ANCHOR) != 1:
            import pygfx

            raise RuntimeError(
                "octarine's matcap shader could not find its injection point "
                f"in pygfx {pygfx.__version__}'s mesh shader - the two are "
                "incompatible. Please open an issue at "
                "https://github.com/schlegelp/octarine/issues"
            )
        return code.replace(_ANCHOR, _MATCAP_WGSL)
