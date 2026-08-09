"""Procedural environment maps for image-based lighting (IBL).

Lighting a scene with a handful of point/directional lights leaves surfaces
looking flat: every pixel is lit from two or three directions and from
nowhere else. Real objects are lit from *every* direction - the sky, the
ground, the walls of the room - which is what gives them their gradients and
their reflections. Image-based lighting captures that by wrapping the scene
in an environment map and treating the whole thing as a light source.

Environment maps are normally photographs (HDRIs). This module instead
*synthesizes* one from an analytic model, so nothing has to be downloaded or
shipped with the package:

 - a three-band **sky gradient** (zenith -> horizon -> ground), which is
   what produces the smooth top-lit-to-bottom-shaded falloff, and
 - a handful of **softboxes**: bright discs at a given direction, angular
   size and softness, standing in for studio lights, a sun, or the window
   of a room. Giving one a `length` stretches it into a tube - a strip
   light or a fluorescent fixture - which is what produces the long, drawn
   out streaks a glossy surface shows. These are what produce the
   highlights and the crisp reflections.

`ENVIRONMENT_PRESETS` holds a few ready-made lighting setups built from
those two ingredients. The result is uploaded as a `rgba16float` cube map,
i.e. it is genuinely high dynamic range: a softbox can be many times
brighter than white, which is what makes highlights read as highlights (and
what gives the bloom and tone mapping passes something to work with).

pygfx does the actual lighting: `Viewer.set_environment` assigns the cube
map to `scene.environment`, from where every `MeshStandardMaterial` picks it
up as full diffuse + specular IBL. Materials that are not PBR (e.g. the
Phong default) only get the specular reflection - see `Viewer.set_environment`.
"""

import numpy as np
import pygfx as gfx

from ..utils import as_color_list

# Ready-made lighting setups. `sky`/`horizon`/`ground` define the background
# gradient, `lights` the softboxes on top of it. A softbox is a disc of
# angular radius `angle` degrees, fading out over the outer `softness`
# fraction of it; `length` (also degrees, default 0) stretches it into a
# tube and `roll` turns that tube about its own axis. Directions are in world
# space (+y is up, +z towards the default camera), and do not need to be
# normalized. `description` is what the GUI shows as tooltip.
#
# The `intensity` of each preset is calibrated rather than chosen: it is set
# so that the brightest direction of the environment lights a white diffuse
# surface to just below clipping. Without that, the environment alone blows
# out anything pale - environment maps light a surface from every direction
# at once, so they add up to much more than a light of the same brightness.
ENVIRONMENT_PRESETS = {
    "studio": dict(
        description="Neutral three-point studio; the all-rounder",
        intensity=0.55,
        sky="#8a9099",
        horizon="#6e7278",
        ground="#2b2c2e",
        gradient=1.0,
        lights=(
            # Key: large, soft, slightly warm, high up on the left
            dict(direction=(-0.55, 0.75, 0.45), color="#fff2e2", intensity=9.0,
                 angle=32, softness=0.85),
            # Fill: dimmer and cooler, from the right, to keep the shadow
            # side from going dead
            dict(direction=(0.85, 0.15, 0.5), color="#dfe8ff", intensity=2.2,
                 angle=45, softness=1.0),
            # Rim: small and hot, from behind and above, for the edge
            # highlight that separates the object from the background
            dict(direction=(0.25, 0.6, -0.9), color="#ffffff", intensity=14.0,
                 angle=14, softness=0.5),
        ),
    ),
    "soft": dict(
        description="Overcast dome; near-shadowless and neutral, for figures",
        intensity=0.4,
        sky="#b8bcc2",
        horizon="#9aa0a6",
        ground="#4a4c50",
        gradient=0.7,
        lights=(
            # A single very large, very soft source straight overhead: enough
            # to give shape without casting a highlight anyone would notice
            dict(direction=(-0.1, 1.0, 0.25), color="#ffffff", intensity=2.6,
                 angle=70, softness=1.0),
        ),
    ),
    "sky": dict(
        description="Outdoor daylight; blue zenith, warm sun, earthy bounce",
        intensity=0.72,
        sky="#4a7ec4",
        horizon="#c3d4e6",
        ground="#4c4437",
        gradient=1.6,
        lights=(
            # The sun: small and very bright, the classic hard key light
            dict(direction=(-0.5, 0.62, 0.6), color="#fff4dd", intensity=42.0,
                 angle=7, softness=0.35),
            # Broad sky fill opposite the sun
            dict(direction=(0.6, 0.5, -0.5), color="#a8c8f0", intensity=1.6,
                 angle=60, softness=1.0),
        ),
    ),
    "sunset": dict(
        description="Low warm sun against a violet sky; long, dramatic gradients",
        intensity=1.5,
        sky="#3c3363",
        horizon="#d98a52",
        ground="#221c26",
        gradient=2.4,
        lights=(
            dict(direction=(-0.9, 0.12, 0.35), color="#ffb066", intensity=26.0,
                 angle=11, softness=0.6),
            dict(direction=(0.5, 0.55, -0.6), color="#7d7ad6", intensity=3.0,
                 angle=50, softness=1.0),
        ),
    ),
    "neon": dict(
        description="Near-black room with magenta/cyan rims; for dark backgrounds",
        intensity=0.75,
        sky="#101322",
        horizon="#0b0d16",
        ground="#05060a",
        gradient=1.0,
        lights=(
            dict(direction=(-0.7, 0.35, 0.55), color="#ff2f9e", intensity=16.0,
                 angle=26, softness=0.9),
            dict(direction=(0.75, 0.3, -0.5), color="#25e8ff", intensity=13.0,
                 angle=26, softness=0.9),
            # A dim top light so the geometry does not disappear entirely
            dict(direction=(0.0, 1.0, 0.1), color="#c8d4ff", intensity=1.2,
                 angle=60, softness=1.0),
        ),
    ),
}

#: The parameters that make up an environment, i.e. what `procedural_env_map`
#: accepts as overrides on top of a preset. Shared with `Viewer.set_environment`
#: so that both validate the same names.
ENVIRONMENT_PROPERTIES = ("intensity", "sky", "horizon", "ground", "gradient", "lights")


def cube_directions(size):
    """World-space direction of every texel of a cube map.

    Parameters
    ----------
    size :  int
            Width/height of one cube face.

    Returns
    -------
    (6, size, size, 3) array
            Unit vectors, one per texel, in the face order wgpu expects
            (+X, -X, +Y, -Y, +Z, -Z).

    Notes
    -----
    Cube maps are specified in a left-handed coordinate system while pygfx
    is right-handed, so pygfx negates the x component when it samples one
    (both for the skybox background and for `env_map` lookups). The
    directions returned here have that flip applied already, i.e. they are
    the *world* direction that will end up sampling the corresponding texel.

    """
    # Texel centers across a face, in [-1, 1]
    t = (np.arange(size, dtype=np.float32) + 0.5) / size * 2 - 1
    # `u` runs along the columns (to the right), `v` along the rows (down)
    u, v = np.meshgrid(t, t)
    one = np.ones_like(u)

    # The standard cube map face parametrization
    faces = np.stack(
        [
            np.stack([one, -v, -u], -1),  # +X
            np.stack([-one, -v, u], -1),  # -X
            np.stack([u, one, v], -1),  # +Y
            np.stack([u, -one, -v], -1),  # -Y
            np.stack([u, -v, one], -1),  # +Z
            np.stack([-u, -v, -one], -1),  # -Z
        ]
    )
    faces /= np.linalg.norm(faces, axis=-1, keepdims=True)

    # Undo the x flip pygfx applies when sampling, so that callers can think
    # in plain world directions
    faces[..., 0] *= -1
    return faces


def _physical2srgb(x):
    """Encode linear (physical) values the way pygfx's shaders expect them.

    Every environment lookup in pygfx runs the sampled texel through
    `srgb2physical()`, so the data we upload has to be sRGB-*encoded* even
    though the texture itself is float. The transfer function is monotonic
    above 1.0 as well, which is what keeps the map high dynamic range.
    """
    x = np.maximum(x, 0.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def _srgb2physical(x):
    """Inverse of `_physical2srgb`; matches pygfx's `srgb2physical()`."""
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _resolve_params(preset, *, presets=None, properties=None, kind="environment",
                    **overrides):
    """Merge a preset name (or dict) with explicit overrides.

    Shared with `octarine.shaders.matcap`, which resolves its own recipes the
    same way; `presets`/`properties`/`kind` say which table to look in.
    Overrides of `None` are ignored, so callers can pass their (optional)
    keyword arguments straight through.
    """
    presets = ENVIRONMENT_PRESETS if presets is None else presets
    properties = ENVIRONMENT_PROPERTIES if properties is None else properties

    if isinstance(preset, dict):
        params = dict(preset)
    else:
        if preset not in presets:
            raise ValueError(
                f"Unknown {kind} preset '{preset}'. Available presets: "
                f"{', '.join(presets)}"
            )
        params = dict(presets[preset])
    params.pop("description", None)  # docs/GUI only

    params.update({k: v for k, v in overrides.items() if v is not None})

    if unknown := set(params) - set(properties):
        raise ValueError(
            f"Unknown {kind} propert{'y' if len(unknown) == 1 else 'ies'}: "
            f"{', '.join(sorted(unknown))}. Valid: {', '.join(properties)}."
        )
    return params


def _rgb(color):
    """A color spec as a (3,) array of linear (physical) values."""
    return _srgb2physical(as_color_list(color)[0].rgb).astype(np.float32)


def _light_tangent(direction, roll):
    """The axis a tube light is stretched along.

    Perpendicular to the light's direction, horizontal at `roll` = 0 and
    turning about the direction from there.
    """
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(direction @ up)) > 0.999:  # pointing straight up or down
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    tangent = np.cross(up, direction)
    tangent /= np.linalg.norm(tangent)
    roll = np.radians(float(roll))
    return np.cos(roll) * tangent + np.sin(roll) * np.cross(direction, tangent)


def _light_solid_angle(light):
    """How much of the sky a softbox covers, in steradians."""
    angle = np.radians(float(light.get("angle", 25.0)))
    # Past a full turn a tube just laps itself, and so does the arc that
    # `environment_radiance` clamps against - keep the two in agreement
    length = min(np.radians(float(light.get("length", 0.0))), 2 * np.pi)
    # A disc, plus the band swept out when it is stretched into a tube
    return 2 * np.pi * (1 - np.cos(angle)) + 2 * np.sin(angle) * length


def environment_radiance(directions, preset="studio", *, rotation=0.0, **overrides):
    """Evaluate the analytic environment model in the given directions.

    This is what `procedural_env_map` bakes into a cube map; it is exposed
    separately because the matcap generator lights its sphere with the same
    model (see `octarine.shaders.matcap`).

    Parameters
    ----------
    directions : (..., 3) array
                World-space directions. Must be normalized.
    preset :    str | dict
                Name of an entry in `ENVIRONMENT_PRESETS`, or a dict of the
                properties below.
    rotation :  float
                Rotation of the environment about the vertical axis, in
                degrees. Moves the highlights around without having to
                redefine the lights.
    **overrides
                Individual `intensity`, `sky`, `horizon`, `ground`,
                `gradient` or `lights` values overriding the preset's.

    Returns
    -------
    (..., 3) array
                Linear (physical) radiance. Values well above 1 are normal -
                that is what a light source looks like.

    """
    params = _resolve_params(preset, **overrides)
    directions = np.asarray(directions, dtype=np.float32)

    if rotation:
        # Rotate the *lookup* direction the other way, so that a positive
        # rotation moves the environment (and hence the highlights) as the
        # user would expect
        a = np.radians(-float(rotation))
        cos_a, sin_a = np.cos(a), np.sin(a)
        x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
        directions = np.stack(
            [cos_a * x + sin_a * z, y, -sin_a * x + cos_a * z], axis=-1
        )

    sky = _rgb(params.get("sky", "#8a9099"))
    horizon = _rgb(params.get("horizon", "#6e7278"))
    ground = _rgb(params.get("ground", "#2b2c2e"))
    gradient = float(params.get("gradient", 1.0))
    if gradient <= 0:
        raise ValueError(f"gradient must be > 0, got {gradient}")

    # Sky gradient: the elevation drives a ramp from the horizon color to
    # `sky` above and to `ground` below. `gradient` > 1 keeps the horizon
    # color spread out and confines `sky` to the zenith.
    elevation = directions[..., 1:2]
    up = np.clip(elevation, 0, 1) ** gradient
    down = np.clip(-elevation, 0, 1) ** gradient
    radiance = horizon + (sky - horizon) * up + (ground - horizon) * down

    for light in params.get("lights", ()):
        direction = np.asarray(light["direction"], dtype=np.float32)
        direction = direction / np.linalg.norm(direction)
        angle = np.radians(float(light.get("angle", 25.0)))
        softness = float(np.clip(light.get("softness", 0.8), 0, 1))

        # A disc of angular radius `angle`, fading out over the outer
        # `softness` fraction of it. Comparing cosines avoids an arccos per
        # texel; note that cos decreases as the angle grows, so the *outer*
        # edge is the *lower* cosine.
        cos_outer = np.cos(angle)
        cos_inner = np.cos(angle * (1 - softness))

        length = np.radians(float(light.get("length", 0.0)))
        if length < 0:
            raise ValueError(f"light length must be >= 0, got {light['length']}")
        if length:
            # A tube: the disc is swept along an arc of `length` centered on
            # the light direction, so the distance to measure is the one to
            # the *nearest point of that arc*. Finding it is a matter of
            # projecting into the plane the arc lies in and clamping the
            # angle to the arc's extent.
            tangent = _light_tangent(direction, light.get("roll", 0.0))
            axis = directions @ direction
            along = directions @ tangent
            nearest = np.clip(np.arctan2(along, axis), -length / 2, length / 2)
            cos_angle = axis * np.cos(nearest) + along * np.sin(nearest)
        else:
            cos_angle = directions @ direction

        if cos_inner - cos_outer < 1e-6:  # a hard-edged disc
            falloff = (cos_angle >= cos_outer).astype(np.float32)
        else:
            t = np.clip((cos_angle - cos_outer) / (cos_inner - cos_outer), 0, 1)
            falloff = t * t * (3 - 2 * t)  # smoothstep

        color = _rgb(light.get("color", "#ffffff")) * float(light.get("intensity", 1.0))
        radiance = radiance + falloff[..., None] * color

    intensity = float(params.get("intensity", 1.0))
    if intensity < 0:
        raise ValueError(f"intensity must be >= 0, got {intensity}")
    return radiance * intensity


def procedural_env_map(preset="studio", *, resolution=128, rotation=0.0, **overrides):
    """Build an environment cube map from the analytic model.

    Parameters
    ----------
    preset :    str | dict
                Name of an entry in `ENVIRONMENT_PRESETS` (e.g. "studio",
                "soft", "sky", "sunset" or "neon"), or a dict of the
                properties below.
    resolution : int
                Width/height of one cube face. 128 is plenty for the diffuse
                lighting; raise it if a mirror-like material shows the
                softboxes as visibly polygonal.
    rotation :  float
                Rotation about the vertical axis in degrees; moves the
                highlights without redefining the lights.
    **overrides
                Individual `intensity`, `sky`, `horizon`, `ground`,
                `gradient` or `lights` values overriding the preset's.

    Returns
    -------
    pygfx.TextureMap
                A mipmapped `rgba16float` cube map, ready to be assigned to
                `scene.environment`, `material.env_map` or a
                `pygfx.BackgroundSkyboxMaterial`.

    """
    resolution = int(resolution)
    if resolution < 4:
        raise ValueError(f"resolution must be >= 4, got {resolution}")

    radiance = environment_radiance(
        cube_directions(resolution), preset, rotation=rotation, **overrides
    )

    data = np.empty((6, resolution, resolution, 4), dtype=np.float16)
    data[..., :3] = _physical2srgb(radiance)
    data[..., 3] = 1.0

    # The mipmaps are not just an optimization here: pygfx reads the diffuse
    # irradiance off the smallest mip level and picks the level for the
    # specular reflection by roughness, so without them every material would
    # mirror the environment.
    texture = gfx.Texture(
        data,
        dim=2,
        size=(resolution, resolution, 6),
        generate_mipmaps=True,
    )
    env_map = gfx.TextureMap(texture, filter="linear", mipmap_filter="linear")
    # Remember what this was made from - the GUI uses it to reflect the
    # current state, `Viewer.set_environment` to report it back. Overriding
    # any of its parameters makes it a custom environment.
    is_plain = isinstance(preset, str) and not overrides and not rotation
    env_map.preset = preset if is_plain else None
    return env_map
