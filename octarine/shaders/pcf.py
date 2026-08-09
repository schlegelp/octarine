"""Wider percentage-closer filtering (PCF) for shadows.

A shadow map is a depth image rendered from the light. It is sampled on a grid
that is fixed *in the light's frame*, so when the light turns - which the
headlight does on every camera rotation, see `Viewer._update_headlight` - that
grid turns with it and the depth samples land on different world positions each
frame. The reconstructed shadow edge therefore wobbles by up to a texel, which
reads as a crawling, shimmering outline while you orbit.

Nothing can stop the grid from turning, so the lever is the filter: the wider
the kernel the shadow lookup averages over, the gentler the gradient across the
edge, and the fewer levels a one-texel wobble moves. Softer shadows, less
crawl - it is one trade, not two effects.

pygfx samples a fixed 17-tap kernel reaching ~2.5 texels (and, for point
lights, a single tap with no filtering at all). This module swaps both tap
loops - and only those, the rest of pygfx' snippet is left alone - for a Vogel
disk. Measured on a sphere over a plane, going from pygfx' kernel to 25 taps at
5 texels widens the edge from 10 to 19 pixels and drops the pixels that swing
by more than 10/255 under a rotating grid from 5079 to 1478.

Note that tap *density* is what matters, not reach: spreading the same 36 taps
over 12 texels instead of 8 measures worse than pygfx' stock kernel, because
the disk starts to undersample and the banding comes back. Keep the taps above
roughly 0.2 per texel^2 if you change these numbers.

The kernel is baked into the WGSL when `install()` first runs (from
`Viewer.__init__`), so it can only be changed before that - see `configure()`.

"""

import math
import re

import jinja2

from .. import config

#: Number of samples in the disk, and its radius in shadow-map texels, for the
#: directional (head)light. Change with `configure()`, not by assignment.
PCF_TAPS = 25
PCF_RADIUS = 5.0

#: The same for point lights, which get their own numbers because their texels
#: are much coarser: a point light's shadow camera is a 90 degree perspective
#: camera, so at `SHADOW_LIGHT_DISTANCE` it spans some six scene radii where
#: the directional light's ortho box spans 2.2 - i.e. ~2.7x the world size per
#: texel. The same radius in texels would make their shadows far softer than
#: the headlight's, so it is scaled down to match (and needs fewer taps to
#: cover the smaller disk).
PCF_TAPS_CUBE = 12
PCF_RADIUS_CUBE = 2.5

logger = config.get_logger(__name__)

# The snippet we patch. pygfx' mesh.wgsl pulls this in with
# `{$ include 'pygfx.light_shadow.wgsl' $}` and light_punctual.wgsl calls into
# the two functions we reach into.
_SNIPPET = "light_shadow.wgsl"
_CONTEXT = "pygfx"

# pygfx' directional/spot kernel: an accumulator, a run of fixed-offset taps,
# and the divide. `return shadow;` follows and is left in place.
_STOCK_DISK_TAPS = re.compile(
    r"[ \t]*var shadow: ?f32 = 0\.0;\n"
    r"(?:[ \t]*shadow \+= textureSampleCompareLevel\(.*\n)+"
    r"[ \t]*shadow /= [\d.]+;\n"
)

# pygfx' point-light kernel: one unfiltered tap along the cube-face direction
_STOCK_CUBE_TAP = re.compile(
    r"[ \t]*var dir = uv_to_direction\(faceIndex, light_local\);\n"
    r"[ \t]*var shadow = textureSampleCompareLevel\("
    r"t_shadow, u_shadow_sampler, dir, layer_index, depth\);\n"
)

_DISK_TAPS = """\
    let texel = 1.0 / vec2<f32>(textureDimensions(t_shadow));
    var shadow: f32 = 0.0;
    for (var i = 0; i < PCF_TAPS; i += 1) {
        shadow += textureSampleCompareLevel(
            t_shadow, u_shadow_sampler,
            light_local + PCF_DISK[i] * texel, layer_index, depth
        );
    }
    shadow /= f32(PCF_TAPS);
"""

# A cube map is sampled by direction, not by face + uv, so the taps are laid
# out on this face and turned back into directions. Taps that fall off the edge
# of the face simply point into the neighbouring one, which is where they
# belong.
_CUBE_TAPS = """\
    let texel = 1.0 / vec2<f32>(textureDimensions(t_shadow));
    var shadow: f32 = 0.0;
    for (var i = 0; i < PCF_CUBE_TAPS; i += 1) {
        let dir = uv_to_direction(faceIndex, light_local + PCF_CUBE_DISK[i] * texel);
        shadow += textureSampleCompareLevel(t_shadow, u_shadow_sampler, dir, layer_index, depth);
    }
    shadow /= f32(PCF_CUBE_TAPS);
"""

_installed = False


def _disk_wgsl(name, n, radius):
    """A WGSL const array of `n` points spread over a disk, on a golden spiral."""
    golden_angle = math.pi * (3 - math.sqrt(5))
    taps = ",\n".join(
        f"    vec2<f32>({radius * math.sqrt((i + 0.5) / n) * math.cos(i * golden_angle): .6f},"
        f"{radius * math.sqrt((i + 0.5) / n) * math.sin(i * golden_angle): .6f})"
        for i in range(n)
    )
    return f"const {name}_TAPS: i32 = {n};\n\nconst {name}_DISK = array<vec2<f32>, {n}>(\n{taps}\n);"


def _patch(stock):
    """Swap the tap loops in pygfx' shadow snippet for our wider kernels.

    Everything else in `stock` - the projection, the cube-face selection, the
    bias handling - is left as pygfx wrote it, so upstream fixes to any of that
    still reach us. Raises `RuntimeError` if either kernel is not where we
    expect it, which is the whole compatibility contract with pygfx.

    """
    code = stock
    for what, pattern, replacement in (
        ("directional", _STOCK_DISK_TAPS, _DISK_TAPS),
        ("point-light", _STOCK_CUBE_TAP, _CUBE_TAPS),
    ):
        code, n = pattern.subn(lambda _, r=replacement: r, code)
        if n != 1:
            raise RuntimeError(
                f"expected exactly one {what} tap kernel in pygfx' {_SNIPPET}, found {n}"
            )

    header = "\n\n".join(
        (
            "// Shadow filtering widened by octarine - see octarine/shaders/pcf.py",
            _disk_wgsl("PCF", PCF_TAPS, PCF_RADIUS),
            _disk_wgsl("PCF_CUBE", PCF_TAPS_CUBE, PCF_RADIUS_CUBE),
        )
    )
    return f"{header}\n\n{code}"


def configure(taps=None, radius=None, taps_cube=None, radius_cube=None):
    """Change the shadow filter kernel (see the module docstring for the units).

    Must be called *before* the first `Viewer` is created: the kernel is baked
    into the shader when `install()` runs, and pygfx caches compiled shaders for
    the life of the process, so there is no way to change it afterwards. Raises
    if it is already too late rather than silently doing nothing.

    """
    if _installed:
        raise RuntimeError(
            "Shadow filtering is already installed - `configure()` has to be "
            "called before the first Viewer is created."
        )

    global PCF_TAPS, PCF_RADIUS, PCF_TAPS_CUBE, PCF_RADIUS_CUBE
    if taps is not None:
        PCF_TAPS = int(taps)
    if radius is not None:
        PCF_RADIUS = float(radius)
    if taps_cube is not None:
        PCF_TAPS_CUBE = int(taps_cube)
    if radius_cube is not None:
        PCF_RADIUS_CUBE = float(radius_cube)


def install():
    """Patch our wider kernel into pygfx' light_shadow.wgsl.

    Called from `Viewer.__init__`, before anything is rendered. Note that this
    is process-wide - it patches a snippet every pygfx shader shares, not
    something owned by one viewer.

    Idempotent, and a no-op (with a warning) if pygfx' shader plumbing does not
    look the way we expect - shadows then simply keep pygfx' own filtering
    rather than the viewer failing to render at all. That is the opposite of
    what e.g. `silhouette.py` does, deliberately: those are effects the user
    asked for by name, where quietly rendering something else would be wrong,
    whereas this is an always-on quality tweak whose fallback is still correct.

    """
    global _installed
    if _installed:
        return
    # Set up front so that every failure below degrades exactly once
    _installed = True

    try:
        from pygfx.renderers.wgpu.shader import templating
        from pygfx.renderers.wgpu.wgsl import load_wgsl

        original = templating.root_loader.mapping[_CONTEXT]
        patched = _patch(load_wgsl(_SNIPPET))
    except Exception as e:  # pragma: no cover - depends on the pygfx version
        import pygfx

        logger.warning(
            f"Could not widen pygfx {pygfx.__version__}'s shadow filtering "
            f"({e}); shadows will use pygfx' own, which shimmers more as the "
            "camera moves. Please report at "
            "https://github.com/schlegelp/octarine/issues"
        )
        return

    # `register_wgsl_loader` refuses to re-register the "pygfx" context, so
    # wrap it: our snippet is found first, everything else falls through
    templating.root_loader.mapping[_CONTEXT] = jinja2.ChoiceLoader(
        [jinja2.DictLoader({_SNIPPET: patched}), original]
    )
    templating.jinja_env.cache.clear()
