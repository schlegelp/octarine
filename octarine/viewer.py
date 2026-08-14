import os
import sys
import time
import cmap
import uuid
import random
import inspect

import numpy as np
import pygfx as gfx
import trimesh as tm
import wgpu  # only for flags/enums

from pathlib import Path
from collections import OrderedDict
from functools import wraps, lru_cache, partial
from pygfx.renderers.wgpu.engine.update import ensure_wgpu_object
from pygfx.renderers.wgpu.engine.edl import EDLPass
from pygfx.renderers.wgpu.engine.effectpasses import (
    NoisePass,
    FogPass,
    NormalPass,
    PPAAPass,
)
from pygfx.renderers.wgpu.engine.bloom import PhysicalBasedBloomPass

from rendercanvas.offscreen import OffscreenRenderCanvas

from .visuals import (
    mesh2gfx,
    volume2gfx,
    points2gfx,
    lines2gfx,
    text2gfx,
    sparsevolume2gfx,
    tubes2gfx,
)
from .conversion import get_converter
from . import utils, config


__all__ = ["Viewer", "viewers"]

logger = config.get_logger(__name__)

# This keeps track of open viewers
viewers = []

AUTOSTART_EVENT_LOOP = True

EFFECT_CLASSES = {
    "edl": EDLPass,
    "noise": NoisePass,
    "fog": FogPass,
    # `None` = resolved lazily inside `add_effect` (octarine's own passes;
    # avoids importing .shaders at module load)
    "depth": None,
    "ao": None,
    "outline": None,
    "tonemap": None,
    "normal": NormalPass,
    "bloom": PhysicalBasedBloomPass,
}

# Post-processing passes run in a fixed order, no matter in which order they
# were switched on. An effect's stage says where in the chain it belongs:
#
#  0/1 - part of how the scene is shaded. These have to come before anything
#        that moves pixels around, so that e.g. a depth-of-field blur blurs
#        the occlusion and the outlines along with the image.
#    2 - the default: lens and image effects (depth of field, bloom, fog, ...)
#    3 - tone mapping, which needs the finished high dynamic range image
#    4 - anti-aliasing, which wants the tone mapped one: run on values that
#        are still allowed to exceed white it smears highlights into their
#        surroundings. pygfx adds its own AA pass at renderer creation, i.e.
#        *before* everything else, so we move it to the end.
EFFECT_STAGES = {"ao": 0, "outline": 1, "tonemap": 3}
DEFAULT_EFFECT_STAGE = 2
PPAA_EFFECT_STAGE = 4

# Effect-pass parameters that are expressed in physical pixels, and hence have
# to be scaled when a screenshot is supersampled: that renders the frame at N
# times the output resolution, so a one-pixel outline drawn into it would come
# out 1/N of a pixel wide in the final image (i.e. invisible). The value is an
# upper limit for the scaled parameter, or `None` for "no limit" - it is there
# for the ones that are sample counts rather than sizes: those pay for the
# larger frame *and* the wider kernel, and are not worth taking to 8x.
# Keyed by class name so that we need not import every pass here.
PIXEL_SCALED_EFFECT_PARAMS = {
    "OutlinePass": {"thickness": None},
    "AmbientOcclusionPass": {"blur": None},
    "DepthOfFieldPass": {
        "aperture": None,
        "max_radius": None,
        "snap_radius": None,
        "num_taps": 256,  # the blur disk grows with the radius, so sample it denser
    },
    "EDLPass": {"radius": None},
    "DDAAPass": {"max_edge_iters": 20},  # length of the edge search, in pixels
}

# TODO
# - add styles for viewer (lights, background, etc.) - e.g. .set_style(dark)
#   - e.g. material.metalness = 2 looks good for background meshes
#   - metalness = 1 with roughness = 0 makes for funky looking neurons
#   - m.material.side = "FRONT" makes volumes look better
# - make Viewer reactive (see reactive_rendering.py) to save
#   resources when not actively using the viewer - might help in Jupyter?
# [/] add specialised methods for adding neurons, volumes, etc. to the viewer

# The named views `Viewer.set_view` and `Viewer.get_view` understand, as
# ``name -> (view_dir, up)``. Prefixing a name with "-" looks from the far side.
NAMED_VIEWS = {
    "XY": ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),  # frontal
    "-XY": ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),  # from the back
    "XZ": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),  # lateral
    "-XZ": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),  # the other lateral
    "YZ": ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),  # top
    "-YZ": ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),  # bottom
}

# Positions of the two static lights (see `Viewer.__init__`). They are parked
# far away from the scene so that their light arrives virtually parallel, no
# matter how large the scene is. See `Viewer._fit_shadows` for what happens to
# them when shadows are switched on.
STATIC_LIGHT_POSITIONS = np.array([(-1, -1, -1), (1, 1, 1)], dtype=float) * 1e6

# How far (in scene radii) from the scene's center the static lights are placed
# when shadows are on. Anything much larger costs depth precision in the shadow
# map; anything much smaller makes the light noticeably divergent.
SHADOW_LIGHT_DISTANCE = 3

# Slack (in scene radii) left around the scene when sizing a shadow camera's
# frustum. Keeps the scene off the very edge of the shadow map.
SHADOW_MARGIN = 1.1

# pygfx can only render meshes, lines and points into a shadow map; it raises
# a `RuntimeError` for anything else (volumes, text, images, ...)
SHADOW_CASTERS = (gfx.Mesh, gfx.Line, gfx.Points)


def update_viewer(legend=True, bounds=True):
    def outer(func):
        """Decorator to update legend and other properties."""

        @wraps(func)
        def inner(*args, **kwargs):
            # Run function first
            func(*args, **kwargs)
            update_helper(viewer=args[0], legend=legend, bounds=bounds)

        return inner

    return outer


def _brighten_color(color, amount=0.3):
    """Lift a color's HSL lightness (or lower it if already bright)."""
    h, s, lightness = gfx.Color(color).to_hsl()
    if lightness <= (1 - amount):
        lightness = min(lightness + amount, 1)
    else:
        lightness = max(lightness - amount, 0)
    return gfx.Color.from_hsl(h, s, lightness)


def _nice_number(x):
    """Round `x` down to the nearest 1, 2 or 5 times a power of ten."""
    exponent = np.floor(np.log10(x))
    fraction = x / 10**exponent  # this is somewhere in [1, 10)
    for m in (5, 2):
        if fraction >= m:
            return m * 10.0**exponent
    return 10.0**exponent


def _format_number(x):
    """Format a number for display without trailing zeros."""
    if float(x).is_integer() and abs(x) < 1e9:
        return f"{int(x):d}"
    return f"{x:g}"


# Fields of a pygfx camera state (see `gfx.PerspectiveCamera.get_state`). Note
# that "position" is accepted as an alias for "x", "y" and "z" combined.
CAMERA_STATE_FIELDS = {
    "x",
    "y",
    "z",
    "scale",
    "rotation",
    "reference_up",
    "fov",
    "width",
    "height",
    "depth",
    "zoom",
    "maintain_aspect",
    "depth_range",
}


def _parse_state_fields(fields):
    """Parse camera state field names (see `Viewer.link`)."""
    if fields is None:
        return None

    if isinstance(fields, str):
        fields = {fields}
    fields = set(fields)

    if not fields:
        return None

    # "position" is a single field in the state dict but the filters work on
    # the individual coordinates (same as in pygfx)
    if "position" in fields:
        fields.discard("position")
        fields.update({"x", "y", "z"})

    unknown = fields - CAMERA_STATE_FIELDS
    if unknown:
        raise ValueError(
            f"Unknown camera state field(s): {', '.join(sorted(unknown))}. "
            f"Must be 'position' or one of {sorted(CAMERA_STATE_FIELDS)}."
        )

    return fields


def _flatten_viewers(args, func):
    """Flatten (potentially nested) viewer arguments into a flat list."""
    viewers_ = []
    for arg in args:
        arg = arg if isinstance(arg, (list, tuple, set)) else [arg]
        for v in arg:
            if not isinstance(v, Viewer):
                raise TypeError(
                    f"`Viewer.{func}` expected Viewer(s), got '{type(v).__name__}'"
                )
            if v not in viewers_:
                viewers_.append(v)
    return viewers_


def _filter_camera_state(state, include=None, exclude=None):
    """Drop camera state fields that should not be synchronised.

    This mirrors what pygfx' controllers do for partially linked cameras.
    """
    if include is None and exclude is None:
        return state

    # Split "position" into its components so the filters can address them
    if "position" in state:
        state = state.copy()
        state["x"], state["y"], state["z"] = state.pop("position")

    if include is not None:
        state = {k: v for k, v in state.items() if k in include}
    if exclude is not None:
        state = {k: v for k, v in state.items() if k not in exclude}

    return state


def update_helper(viewer, legend=True, bounds=True):
    """Helper function to update legend and other properties."""
    # Always clear the cached objects dictionary
    viewer._objects.cache_clear()

    if legend:
        if getattr(viewer, "controls", None):
            viewer.controls.update_legend()
        if getattr(viewer, "widget", None):
            if viewer.widget.toolbar:
                viewer.widget.toolbar.update_legend()
    if bounds:
        # Everything that is fitted to the scene as a whole - the bounding box
        # visual, the shadow-casting lights, the ambient occlusion radius, the
        # environment - is now out of date. Each of those walks every visual on
        # the canvas, so bringing them up to date here would make a loop of
        # `add` calls quadratic in the number of objects. Instead we only flag
        # the scene and catch up once, right before the next frame is drawn
        # (see `Viewer._refresh_scene`).
        viewer._scene_stale = True

    # Any time we update the viewer, we should set it to stale
    viewer._render_stale = True
    viewer.canvas.request_draw()


class Viewer:
    """PyGFX 3D viewer.

    Parameters
    ----------
    offscreen : bool
                If True, will use an offscreen Canvas. Useful if you only
                want a screenshot.
    title :     str
                Title of the viewer window.
    max_fps :   int
                Maximum frames per second to render.
    size :      tuple, optional
                Size of the viewer window.
    camera :    "ortho" | "perspective"
                Type of camera to use. Defaults to "ortho". Note you can always
                change the camera type by adjust the `Viewer.camera.fov` attribute
                (0 = ortho, >0 = perspective).
    control :   "trackball" | "panzoom" | "fly" | "orbit"
                Controller type to use. Defaults to "trackball".
    headlight : bool
                If True (default), objects are lit by a single light source that
                is linked to the camera - i.e. they are always lit from the front.
                If False, we use two fixed light sources which means the lighting
                changes as you move the camera. Can also be changed at any time via
                the `Viewer.headlight` property.
    shadows :   bool
                Whether objects cast shadows onto each other. On by default; see
                the `Viewer.shadows` property for details.
    ambient_occlusion : bool
                Whether to darken creases, cavities and the points where objects
                touch. On by default with settings derived from the scene; use
                `Viewer.set_ambient_occlusion` to tune them.
    show :      "auto" (default) | bool
                Whether to immediately show the viewer. When set to "auto" (default),
                will immmediately show the viewer if:
                 - we are in a Jupyter environment
                 - we are in an iPython session and we can hook into an iPython event loop
                If neither of the above applies or `show=False`, you will have to manually run
                `Viewer.show()`. This gives you the chance to add objects to the viewer
                before it is shown and the blocking event loop is started.
                The `show` parameter is ignored if `offscreen` is True.
    **kwargs
                Keyword arguments are passed through to ``WgpuCanvas``.

    """

    # Palette used for assigning colors to objects
    palette = "seaborn:tab10"
    highlight_color = "yellow"

    def __init__(
        self,
        offscreen=False,
        title="Octarine Viewer",
        max_fps=30,
        camera="ortho",
        control="trackball",
        size=None,
        show=True,
        headlight=True,
        shadows=True,
        ambient_occlusion=True,
        **kwargs,
    ):
        # We need to import WgpuCanvas before we (potentially) start the event loop
        # If we don't, we get a segfault.
        if not offscreen:
            from rendercanvas.auto import RenderCanvas

        # Check if we're running in an IPython environment
        if utils._type_of_script() == "ipython" and not offscreen:
            ip = get_ipython()  # noqa: F821
            if not ip.active_eventloop:
                if AUTOSTART_EVENT_LOOP:
                    try:
                        ip.enable_gui("qt")
                        logger.debug(
                            "Looks like you're running in an IPython environment but haven't "
                            "started a GUI event loop. We've started one for you using the "
                            "Qt backend."
                        )
                    except (ModuleNotFoundError, ImportError):
                        raise ValueError(
                            "Looks like you're running in an IPython environment but haven't "
                            "started a GUI event loop. We tried to start one for you using the "
                            "Qt6 backend (via %gui qt6) but that failed. You may have to start "
                            "the event loop manually. See "
                            "https://ipython.readthedocs.io/en/stable/config/eventloops.html"
                            "for details."
                        )
                else:
                    raise ValueError(
                        'IPython event loop not running. Please use e.g. "%gui qt6" to hook into the event loop.'
                    )

            # ipython is running multiple event loops and recent versions which seems to confuse rendercanvas
            # (see https://github.com/pygfx/rendercanvas/issues/211). Here, we force it to use asyncio
            # from rendercanvas.asyncio import loop
            # RenderCanvas.select_loop(loop)

        self._title = title

        # Update some defaults as necessary
        defaults = {"title": title, "max_fps": max_fps}
        if size is not None:
            defaults["size"] = size
        defaults.update(kwargs)

        # If we're running in headless mode (primarily for tests on CI) we will
        # simply not initialize the gfx objects. Not ideal but it turns
        # out to be very annoying to correctly setup on Github Actions.
        if getattr(config, "HEADLESS", False):
            return

        if not offscreen:
            self.canvas = RenderCanvas(**defaults)
        else:
            self.canvas = OffscreenRenderCanvas(**defaults)

        # There is a bug in pygfx 0.1.18 that causes the renderer to crash
        # when using a Jupyter canvas without explicitly setting the pixel_ratio.
        # This is already fixed in main but for now:
        if self._is_jupyter:
            self.renderer = gfx.renderers.WgpuRenderer(
                self.canvas, show_fps=False, pixel_ratio=2
            )
        else:
            self.renderer = gfx.renderers.WgpuRenderer(self.canvas, show_fps=False)

        # Set up a default scene
        self.scene = gfx.Scene()

        # A minor ambient light
        self.scene.add(gfx.AmbientLight(intensity=0.5))

        # A strong point light form front/top/left
        key_light = gfx.PointLight(intensity=4)
        key_light.shadow.bias = 0.0000005  # this helps with shadow acne
        key_light.local.position = STATIC_LIGHT_POSITIONS[0]  # left, up, forward

        # A weaker point light from the back
        back_light = gfx.PointLight(intensity=1)
        back_light.shadow.bias = 0.0000005  # this helps with shadow acne
        back_light.local.position = STATIC_LIGHT_POSITIONS[1]  # right, down, back

        # These two lights are fixed in world space, i.e. the lighting changes as
        # the camera moves. They are switched off when the (camera-linked)
        # headlight is switched on - see `Viewer.headlight`
        self._static_lights = [key_light, back_light]
        self.scene.add(key_light, back_light)

        # Set up a default background (see also `set_bgcolor` and
        # `set_bg_gradient`)
        self._bgcolor = [gfx.Color("black")]
        self._background = gfx.Background(None, gfx.BackgroundMaterial((0, 0, 0)))
        self.scene.add(self._background)

        # Add camera
        if camera == "ortho":
            self.camera = gfx.OrthographicCamera()
        elif camera == "perspective":
            self.camera = gfx.PerspectiveCamera()
        else:
            raise ValueError(f"Unknown camera type: {camera}")

        # The headlight no longer hangs off the camera, but keep the camera in
        # the scene graph so that anything a user parents to it still renders
        self.scene.add(self.camera)

        # A light that follows the camera and hence always shines from wherever
        # we are looking from (see `Viewer.headlight`). It is *not* parented to
        # the camera but re-aimed on every frame from `_update_headlight` - see
        # there for why.
        self._headlight = gfx.DirectionalLight(intensity=4)
        self._headlight.shadow.bias = 0.0000005  # this helps with shadow acne
        # Offsetting the light from the camera's axis (here: up and to the left)
        # keeps some variation in the shading - a light shining exactly along the
        # view direction makes objects look very flat
        self._set_headlight_offset((-0.5, 0.5, 0))
        self.scene.add(self._headlight)

        # This also takes care of switching off the static lights (if required)
        self._headlight_enabled = False
        self.headlight = headlight

        # Add controller
        controller = {
            "trackball": gfx.TrackballController,
            "panzoom": gfx.PanZoomController,
            "fly": gfx.FlyController,
            "orbit": gfx.OrbitController,
        }.get(control, None)
        if controller is None:
            raise ValueError(f"Unknown controller type: {control}")

        self.controller = controller(self.camera, register_events=self.renderer)

        # Setup overlay
        self.overlay_camera = gfx.NDCCamera()
        self.overlay_scene = gfx.Scene()

        # Setup transform gizmo
        self.transform_gizmo = None

        # Stats
        self.stats = gfx.Stats(self.renderer)
        self._show_fps = False

        # Setup key events
        self._key_events = {}
        self._key_events["1"] = lambda: self.set_view("XY")  # frontal view
        self._key_events["2"] = lambda: self.set_view("XZ")  # lateral view
        self._key_events["3"] = lambda: self.set_view("YZ")  # top view
        self._key_events[("1", ("Shift",))] = lambda: self.set_view("-XY")  # back view
        self._key_events[("2", ("Shift",))] = lambda: self.set_view(
            "-XZ"
        )  # other lateral view
        self._key_events[("3", ("Shift",))] = lambda: self.set_view(
            "-YZ"
        )  # bottom view
        self._key_events["f"] = lambda: self._toggle_fps()
        self._key_events["c"] = lambda: self._toggle_controls()

        def _keydown(event):
            """Handle key presses."""
            if not event.modifiers:
                if event.key in self._key_events:
                    self._key_events[event.key]()
            else:
                tup = (event.key, tuple(event.modifiers))
                if tup in self._key_events:
                    self._key_events[tup]()

        # Register events
        self.renderer.add_event_handler(_keydown, "key_down")

        # Finally, setting some variables
        self._show_bounds = False
        self._shadows = False
        self._shadow_fit = None  # (center, radius) of the scene; see `_fit_shadows`
        # Whether the scene's contents changed since we last fitted anything to
        # them, and whether an `add` asked for the camera to be re-centered;
        # see `Viewer._refresh_scene`
        self._scene_stale = False
        self._center_pending = False
        self._centered_camera_sig = None
        self._refreshing_scene = False
        self._ao_pass = None
        self._ao_auto_radius = True  # see `Viewer._update_ao_radius`
        self._outline_pass = None
        self._tonemap_pass = None
        # Image-based lighting; see `Viewer.set_environment`
        self._env_map = None
        self._env_settings = {}
        self._env_background = False
        self._pre_env_light_intensities = None
        self._animations = {}
        self._animations_flagged_for_removal = []
        self._animations_frame_counter = 0
        self._on_double_click = None
        self._on_hover = None
        self._objects_pickable = False
        self._selected = []
        self._render_trigger = "continuous"

        # Camera links (see `Viewer.link`)
        self._linked = []
        self._link_filter = (None, None)
        self._last_camera_sig = None

        # Widen the shadow filter. The kernel is baked into the shader, so this
        # has to run before the first compile - see `octarine.shaders.pcf`,
        # including for what happens without octarine's custom shaders.
        try:
            from .shaders.pcf import install as _install_pcf

            _install_pcf()
        except ImportError as e:
            logger.warning(f"Shadow filtering left at pygfx' default: {e}")

        # Effects that are on by default. These have to come last because they
        # need the variables above (plus the scene, camera and renderer)
        self.shadows = shadows
        if ambient_occlusion:
            self.set_ambient_occlusion()

        viewers.append(self)

        # This starts the animation loop
        if show and not self._is_jupyter:
            self.show(start_loop=show == "start_loop")

    def _animate(self):
        """Run the rendering loop."""
        rm = self.render_trigger

        # Objects may have come or gone since the last frame - catch up on
        # everything that is fitted to the scene as a whole before anything
        # else in this frame gets to look at it. In particular this has to
        # happen before the animations below: several of them (the scale bar,
        # the depth-of-field focus tracker) read the camera, which we may be
        # about to re-center.
        self._refresh_scene()

        # First run the user animations
        self._animations_frame_counter += 1
        if self._animations_frame_counter == sys.maxsize:  # reset to avoid overflow
            self._animations_frame_counter = 0
        # N.B. we're iterating over the list because the user might add / remove
        # animations during the loop
        for i, (func, (on_error, run_every, req_render)) in enumerate(
            list(self._animations.items())
        ):
            # Skip if we're not supposed to run this frame
            if run_every and (self._animations_frame_counter % run_every) != 0:
                continue
            try:
                func()
                if req_render:
                    self._render_stale = True
            except BaseException as e:
                if on_error == "raise":
                    raise e
                elif on_error == "log":
                    logger.error(f"Error in animation function '{func}': {e}")
                elif on_error == "remove":
                    logger.error(
                        f"Removing animation function '{func}' because of error: {e}"
                    )
                    # Flag animation for removal
                    self._animations_flagged_for_removal.append(func)

        # Check if any animations need to be removed
        for f in self._animations_flagged_for_removal:
            try:
                _ = self._animations.pop(f)
            except KeyError:
                pass  # already removed (e.g. by index or by function)
        self._animations_flagged_for_removal = []

        # Now check if we need to render the scene
        if rm == "active_window":
            if not self._window_is_active():
                self.canvas.request_draw()
                return
        elif rm == "reactive":
            # If we're linked to another viewer, our camera may have been moved
            # by that viewer's controller - in which case none of our own events
            # fired and nothing flagged us as stale (see `Viewer.link`)
            if self._linked and self._camera_sig() != self._last_camera_sig:
                self._render_stale = True
            # If the scene is not stale, we can skip rendering
            if not getattr(self, "_render_stale", False):
                self.canvas.request_draw()
                return

        # The headlight follows the camera but is not parented to it, so it has
        # to be re-aimed whenever we've moved (see `_update_headlight`)
        if self._headlight_enabled:
            self._update_headlight()

        # Now render the scene
        if self._show_fps:
            with self.stats:
                self.renderer.render(self.scene, self.camera, flush=False)
                if self.transform_gizmo:
                    self.renderer.render(self.transform_gizmo, self.camera, flush=False)
                self.renderer.render(
                    self.overlay_scene, self.overlay_camera, flush=False
                )
            self.stats.render()
        else:
            self.renderer.render(self.scene, self.camera, flush=False)
            if self.transform_gizmo:
                self.renderer.render(self.transform_gizmo, self.camera, flush=False)
            self.renderer.render(self.overlay_scene, self.overlay_camera)

        # Set stale to False
        self._render_stale = False
        self._last_camera_sig = self._camera_sig() if self._linked else None

        self.canvas.request_draw()

    def _window_is_active(self):
        """Whether the canvas' window currently has the focus.

        Note to self: we need to explore how to do this with different backends
        / window managers. Only the Qt canvas can tell us at all; everything
        else (offscreen canvases in particular) counts as active. Not sure if
        this will work with e.g. Jupyter (does it know when the notebook is
        active?).

        """
        if not hasattr(self.canvas, "isActiveWindow"):
            return True
        return self.canvas.isActiveWindow()

    def _refresh_scene(self):
        """Re-fit everything that is derived from the scene as a whole.

        The bounding box visual, the camera (if an `add` asked to be centered),
        the shadow-casting lights (with their shadow cameras), the ambient
        occlusion radius and the environment maps all have to follow the scene
        as objects come and go - and each of them walks every visual on the
        canvas. Doing that for every object added would make filling a viewer
        quadratic in the number of objects, which is why `update_helper` and
        `Viewer.add` merely flag what is out of date and we catch up here
        instead: once, immediately before the next frame is drawn. Objects are
        typically added in a loop, so this collapses N sweeps into one.

        Note that this does *not* affect `Viewer.bounds`, which always reports
        the scene as it currently stands, nor an explicit call to
        `Viewer.center_camera`, which centers there and then.

        """
        if not (self._scene_stale or self._center_pending) or self._refreshing_scene:
            return

        # `update_bounds` takes the previous bounding box off the scene, which
        # comes back through `update_helper` and flags us as stale again -
        # hence both the re-entrancy guard and clearing the flags only at the end
        self._refreshing_scene = True
        try:
            if self._show_bounds:
                self.update_bounds()

            # N.B. this has to come *after* the bounding box visual was
            # re-fitted: `center_camera` frames the whole scene graph, box
            # included, and a stale box still sticking out of the scene would
            # widen the view. It is also skipped if the camera was moved since
            # (see `Viewer._request_center`).
            if self._center_pending and self._camera_sig() == self._centered_camera_sig:
                self.center_camera()

            # New visuals have to pick up the shadow state, and both the lights
            # and the ambient occlusion radius have to be re-fitted to the new
            # extents of the scene. Walking those extents is O(number of
            # objects), so do it once and share.
            fit_shadows = self._shadows
            fit_ao = self._ao_pass is not None
            if fit_shadows or fit_ao:
                world_bounds = self.bounds
                if fit_shadows:
                    self._update_shadows(bounds=world_bounds)
                if fit_ao:
                    self._update_ao_radius(bounds=world_bounds)

            # ... and new meshes have to be lit by the environment like the rest
            if self._env_map is not None:
                self._update_environment()
        finally:
            self._refreshing_scene = False
            self._scene_stale = False
            self._center_pending = False

    def _next_color(self):
        """Return next color in the colormap."""
        # Cache the full palette. N.B. that ordering of colors in cmap depends on
        # the number of colors requested - i.e. we can't just grab the last color.
        if not hasattr(self, "_cached_palette") or self.palette != self._cached_palette:
            self._cached_colors = list(cmap.Colormap(self.palette).iter_colors())
            self._cached_palette = self.palette

        if not hasattr(self, "_palette_index"):
            self._palette_index = -1
        self._palette_index += 1

        return self._cached_colors[self._palette_index % len(self._cached_colors)]

    def _next_label(self, prefix="Object"):
        """Return next label."""
        existing = [o for o in self.objects if str(o).startswith(prefix)]
        if len(existing) == 0:
            return prefix
        return f"{prefix}.{len(existing) + 1:03}"

    def __getitem__(self, key):
        """Get item."""
        return self.objects[key]

    def __contains__(self, key):
        """Check if object is on canvas."""
        return key in self.objects

    def __len__(self):
        """Return number of objects on canvas."""
        return len(self._object_ids)

    @property
    def blend_mode(self):
        """Deprecated! Render blend mode.

        This property has been deprecated. Please use `Viewer.set_alpha_mode()` instead.

        """
        raise DeprecationWarning(
            "The 'blend_mode' property is deprecated. Please use 'Viewer.set_alpha_mode()' instead."
        )

    @property
    def render_trigger(self):
        """Determines when the scene is (re)rendered.

        By default, we leave it to the renderer to decide when to render the scene.
        You can adjust that behaviour by setting render mode to:
         - "continuous" (default): leave it to the renderer to decide when to render the scene
         - "reactive": rendering is only triggered when the scene changes
         - "active_window": rendering is only done when the window is active; this currently
           only works with the PySide backend

        """
        return self._render_trigger

    @render_trigger.setter
    def render_trigger(self, mode):
        valid = ("continuous", "active_window", "reactive")
        if mode not in valid:
            raise ValueError(f"Unknown render mode: {mode}. Must be one of {valid}.")

        # No need to do anything if the value is the same
        if mode == getattr(self, "_render_trigger", None):
            return

        # Add/remove event handlers as necessary
        if mode == "reactive":
            self._set_stale_func = lambda event: setattr(self, "_render_stale", True)
            self.renderer.add_event_handler(
                self._set_stale_func,
                "pointer_down",
                "pointer_move",
                "pointer_up",
                "wheel",
                # "before_render",
            )
        elif self._render_trigger == "reactive":
            self.renderer.remove_event_handler(
                self._set_stale_func,
                "pointer_down",
                "pointer_move",
                "pointer_up",
                "wheel",
                # "before_render",
            )

        self._render_trigger = mode

    @property
    def controls(self):
        """Return the controls widget."""
        return getattr(self, "_controls", None)

    @property
    def visible(self):
        """List IDs of currently visible objects."""
        objects = self.objects  # grab this only once to speed things up
        return [s for s in objects if objects[s][0].visible]

    @property
    def invisible(self):
        """List IDs of currently visible objects."""
        objects = self.objects  # grab this only once to speed things up
        return [s for s in objects if not objects[s][0].visible]

    @property
    def pinned(self):
        """List IDs of currently pinned objects."""
        objects = self.objects  # grab this only once to speed things up
        return [s for s in objects if getattr(objects[s][0], "_pinned", False)]

    @property
    def selected(self):
        """Return IDs of or set selected objects."""
        return self._selected

    @selected.setter
    def selected(self, val):
        val = utils.make_iterable(val) if val is not None else []

        objects = self.objects  # grab once to speed things up
        logger.debug(f"{len(val)} objects selected ({len(self.selected)} previously)")
        # First un-highlight neurons which aren't selected anymore
        for s in [s for s in self._selected if s not in val]:
            for v in objects[s]:
                v.material.color = v._stored_color

        # Highlight new additions
        for s in val:
            if s not in self._selected:
                for v in objects[s]:
                    # Keep track of old colour
                    v._stored_color = v.material.color
                    v.material.color = gfx.Color(self.highlight_color)
        self._selected = list(val)

        # Update legend and set render stale (if applicable)
        update_helper(self, legend=True, bounds=False)

    @property
    def size(self):
        """Return size of the canvas."""
        return self.canvas.get_logical_size()

    @size.setter
    def size(self, size):
        """Set size of the canvas."""
        assert len(size) == 2
        self.canvas.set_logical_size(*size)

    @property
    def lights(self):
        """List of all light sources illuminating the scene.

        This includes the headlight, which is a scene child like the others but
        gets re-aimed from the camera on every frame (see `Viewer.headlight`).

        """
        return list(self.scene.iter(lambda x: isinstance(x, gfx.Light)))

    @property
    def headlight(self):
        """Whether the scene is lit by a light linked to the camera.

        If True (default), a single light source follows the camera, which means
        objects are always lit from the front, no matter where you move the
        camera. If False, we use two point lights that are fixed in
        world space, i.e. the lighting changes as the camera moves. Providing
        either a float or a tuple of 2 or 3 floats will switch the headlight on
        and set the light's offset from the camera's axis: a single float `x`
        is shorthand for `(-x, x, 0)`, i.e. moves the light left and up. The
        default offset is (-0.5, 0.5, 0) and is kept when you switch the
        headlight off and on again.

        Note that the ambient light is unaffected by this setting.

        """
        return self._headlight_enabled

    @headlight.setter
    def headlight(self, v):
        offset = None  # `None` means: keep the current offset
        if isinstance(v, bool):
            # N.B. this check must come first because `bool` is a subclass of
            # `int` and would otherwise be interpreted as an offset
            pass
        elif isinstance(v, (int, float)):
            offset = (-float(v), float(v), 0)
            v = True
        elif isinstance(v, (tuple, list)):
            if len(v) == 2:
                offset = (float(v[0]), float(v[1]), 0)
            elif len(v) == 3:
                offset = (float(v[0]), float(v[1]), float(v[2]))
            else:
                raise ValueError(
                    f"Expected 2 or 3 values for headlight offset, got {len(v)}"
                )
            v = True
        else:
            raise TypeError(f"Expected bool, float or tuple, got {type(v)}")

        self._headlight_enabled = v
        self._headlight.visible = v
        if offset is not None:
            self._set_headlight_offset(offset)
        for light in self._static_lights:
            light.visible = not v

        self._render_stale = True

    def _set_headlight_offset(self, offset):
        """Set the headlight's offset from the camera's axis.

        Also caches the direction it implies - from the offset towards the point
        one unit in front of the camera, i.e. the light's own -z in camera space
        - because `_update_headlight` needs it on every frame and it changes
        only here.

        """
        self._headlight_offset = np.asarray(offset, dtype=float)

        direction = np.array((0, 0, -1), dtype=float) - self._headlight_offset
        norm = np.linalg.norm(direction)
        # A degenerate offset (i.e. sitting on the target) - shine straight ahead
        self._headlight_direction = direction / norm if norm else direction + (0, 0, -1)

    def toggle_headlight(self):
        """Toggle the camera-linked headlight."""
        self.headlight = not self.headlight

    @property
    def shadows(self):
        """Whether objects cast shadows onto each other (on by default).

        Note that only meshes can *receive* shadows - lines and points can cast
        them but are never shaded themselves. Volumes and text take no part in
        shadows at all.

        The lights and their shadow cameras are automatically fitted to the
        scene while this is on, and re-fitted whenever objects are added or
        removed (see `Viewer._fit_shadows`). Because that moves the static
        lights in much closer than they normally sit, expect the shading to
        change slightly as well.

        """
        return self._shadows

    @shadows.setter
    def shadows(self, v):
        """Set shadow state."""
        if not isinstance(v, bool):
            raise TypeError(f"Expected bool, got {type(v)}")

        if v == self._shadows:
            return

        self._shadows = v
        self._update_shadows()

        self._render_stale = True

    def _update_shadows(self, bounds=None):
        """Apply the current shadow state to the scene.

        This is called whenever shadows are toggled and - via `update_helper` -
        whenever objects are added to or removed from the scene, so that new
        visuals pick up the shadow state and the lights stay fitted to the
        scene as it grows. `bounds` is the scene's extents if the caller has
        them at hand already (see `Viewer.bounds`).

        """
        state = self._shadows

        for vis in self.visuals:
            # pygfx can only render some object types into a shadow map (and
            # raises for the rest). The bounding box is viewer chrome and has no
            # business casting shadows either.
            casts = (
                state
                and isinstance(vis, SHADOW_CASTERS)
                and getattr(vis, "_object_type", None) != "boundingbox"
            )
            # Only meshes ever receive shadows - for everything else pygfx
            # ignores the flag
            receives = state and isinstance(vis, gfx.Mesh)

            # N.B. `receive_shadow` recompiles the object's shader, so don't
            # touch either flag unless it actually changes
            if vis.cast_shadow != casts:
                vis.cast_shadow = casts
            if vis.receive_shadow != receives:
                vis.receive_shadow = receives

        for light in self.lights:
            if isinstance(light, (gfx.PointLight, gfx.DirectionalLight, gfx.SpotLight)):
                light.cast_shadow = state

        self._fit_shadows(bounds=bounds)

    def _fit_shadows(self, bounds=None):
        """Fit the lights and their shadow cameras to the scene.

        Shadow maps are rendered from the light's point of view, through a
        camera whose frustum pygfx sizes neither to the scene nor to the light's
        position: point lights get a fixed far plane of 1e5 units and the
        directional (head)light a fixed 1000x1000 ortho box. Our static lights
        sit a million units out, so out of the box the entire scene falls behind
        their shadow camera's far plane and nothing is ever drawn into the map -
        i.e. no shadows, at any scene scale.

        So while shadows are on we pull the static lights in to just outside the
        scene and size all the frusta to match. Switching shadows off parks the
        lights back where they were, which keeps their light parallel.

        `bounds` is the scene's extents if the caller has them at hand already
        (see `Viewer.bounds`).

        """
        if not self._shadows:
            self._shadow_fit = None
            for light, position in zip(self._static_lights, STATIC_LIGHT_POSITIONS):
                light.local.position = position
            return

        if bounds is None:
            bounds = self.bounds
        if bounds is None:  # nothing on the canvas to fit to
            self._shadow_fit = None
            return

        center = bounds.mean(axis=1)
        # Radius of the scene's bounding sphere. The fallback catches scenes
        # without any extent, e.g. a single point.
        radius = float(np.linalg.norm(bounds[:, 1] - bounds[:, 0])) / 2
        if not radius:
            radius = 1.0

        # Cache this: the headlight is re-aimed on every frame (see
        # `_update_headlight`) and we don't want to walk the whole scene graph
        # for its bounding box each time
        self._shadow_fit = (center, radius)

        # The static lights: move them to just outside the scene and clip their
        # shadow camera (a 90 degree perspective camera) to the slab the scene
        # actually occupies. A tight near/far range is what keeps the depth map
        # precise enough to not produce shadow acne.
        distance = radius * SHADOW_LIGHT_DISTANCE
        for light, position in zip(self._static_lights, STATIC_LIGHT_POSITIONS):
            direction = position / np.linalg.norm(position)
            light.local.position = center + direction * distance
            light.shadow.camera.depth_range = (
                (distance - radius) / 2,
                distance + radius * 2,
            )

        # The headlight sits the same distance out, on the axis through the
        # scene's center (see `_update_headlight`). A bounding *sphere* fit
        # keeps its frustum the same size from every direction, so orbiting
        # doesn't resize it either.
        margin = radius * SHADOW_MARGIN
        camera = self._headlight.shadow.camera
        camera.width = camera.height = 2 * margin
        camera.depth_range = (distance - margin, distance + margin)

        self._update_headlight()

    def _update_headlight(self):
        """Point the headlight at the scene from wherever the camera is.

        The obvious way to have a light follow the camera is to parent it to the
        camera, which is what we used to do. That works for the shading but not
        for the shadows: pygfx puts a light's shadow camera *at the light* and
        aims it at the light's target, and for a camera-parented light that
        target is pinned to the point one unit in front of the camera. The
        light's offset from the camera's axis (see `Viewer.headlight`) therefore
        turns into a tilt of ~35 degrees, and the only way for the shadow camera
        to still cover the scene is to grow its (ortho) frustum until it reaches
        - a box several times larger than the scene, growing with the distance
        to the camera. Aiming through the scene's center instead would mean no
        offset at all, i.e. exactly the flat lighting the offset exists to avoid.

        That is what made shadows flicker: the frustum was re-derived per frame,
        so panning or zooming - neither of which changes the light's *direction*,
        and so neither of which may change the shadows - changed the world size
        of a shadow map texel and re-quantised the depth map onto a different
        grid every frame.

        So the light is a plain scene child instead and we aim it ourselves:
        same direction as before (only that matters for a directional light, so
        the shading is unchanged), but parked in front of the scene's center
        rather than riding along with the camera. Its frustum is then sized once
        per scene, in `_fit_shadows`, and camera movement leaves shadows alone.

        """
        # Rotating the cached camera-space direction by the camera gives what
        # pygfx used to derive for us. N.B. this comes out of the camera's
        # orientation alone, not of where it is - and the rotation matrix does
        # it ~20x faster than `pylinalg.vec_transform_quat`.
        direction = (
            self.camera.world.rotation_matrix[:3, :3] @ self._headlight_direction
        )

        if self._shadow_fit is None:
            # Nothing to fit to (an empty canvas, or shadows are off). Only the
            # direction matters for the shading, so leave the light on the
            # camera and its shadow camera alone.
            anchor, distance = self.camera.world.position, 1.0
        else:
            center, radius = self._shadow_fit
            anchor, distance = center, radius * SHADOW_LIGHT_DISTANCE

        self._headlight.local.position = anchor - direction * distance
        self._headlight.target.local.position = anchor

    @property
    def visuals(self):
        """List of all visuals on this canvas."""
        return [c for c in self.scene.children if hasattr(c, "_object_id")]

    @property
    def bounds(self):
        """Bounds of all current visuals (visible and invisible).

        See [`Viewer.get_bounds`][octarine.Viewer.get_bounds] to ask for the
        bounds of individual objects.

        Returns
        -------
        bounds :    (3, 2) array | None
                    ``[[xmin, xmax], [ymin, ymax], [zmin, zmax]]`` in world
                    space, or ``None`` if there is nothing on the canvas.

        """
        return self.get_bounds()

    def get_bounds(self, objects=None):
        """Bounds of the given objects (visible and invisible).

        Parameters
        ----------
        objects :   str | int | list | visual, optional
                    Object(s) to measure: name(s)/ID(s), index(es) in the list
                    of visuals, or the visual(s) themselves. If ``None``
                    (default), uses everything on the canvas.

        Returns
        -------
        bounds :    (3, 2) array | None
                    ``[[xmin, xmax], [ymin, ymax], [zmin, zmax]]`` in world
                    space, or ``None`` if nothing takes up any space.

        """
        if objects is None:
            visuals = self.visuals
        else:
            visuals = self._resolve_visuals(objects)

        bounds = []
        for vis in visuals:
            # Skip the bounding box itself
            if getattr(vis, "_object_type", None) == "boundingbox":
                continue

            # N.B. this is `None` for visuals that don't take up any space
            aabb = vis.get_world_bounding_box()
            if aabb is not None:
                bounds.append(aabb)

        if not bounds:
            return None

        bounds = np.stack(bounds)  # (N, 2, 3)

        mn = bounds[:, 0, :].min(axis=0)
        mx = bounds[:, 1, :].max(axis=0)

        return np.vstack((mn, mx)).T

    def _resolve_visuals(self, obj):
        """Turn name(s)/index(es)/visual(s) into a flat list of visuals."""
        objects = obj if utils.is_iterable(obj) else [obj]

        all_objects = self.objects  # grab once to speed things up

        visuals = []
        for ob in objects:
            if ob in all_objects:
                visuals += list(all_objects[ob])
            elif isinstance(ob, int):
                visuals += list(list(all_objects.values())[ob])
            elif isinstance(ob, gfx.WorldObject):
                visuals.append(ob)
            else:
                raise ValueError(f"Unable to find object(s) for {ob}")
        return visuals

    @property
    def max_fps(self):
        """Maximum frames per second to render."""
        return self.canvas._subwidget._BaseRenderCanvas__scheduler._max_fps

    @max_fps.setter
    def max_fps(self, v):
        assert isinstance(v, int)
        self.canvas._subwidget._BaseRenderCanvas__scheduler._max_fps = v

    @property
    def moveable_object(self):
        """Get/Set the object that can be moved via the transform gizmo."""
        if self.transform_gizmo is None:
            return None
        return self.transform_gizmo._object_to_control

    @moveable_object.setter
    def moveable_object(self, obj):
        if obj is None:
            if self.transform_gizmo:
                self.transform_gizmo._object_to_control = None
            return

        if isinstance(obj, str):
            if obj not in self.objects:
                raise ValueError(f"Object '{obj}' not found.")
            elif len(self.objects[obj]) > 1:
                raise ValueError(f"Object '{obj}' consists of multiple WorldObjects.")
            obj = self.objects[obj][0]
        elif not isinstance(obj, gfx.WorldObject):
            raise TypeError(f"Expected pygfx object, got {type(obj)}")

        if self.transform_gizmo is None:
            # The transform gizmo is rendered independent of the scene (so it always stay on top)
            self.transform_gizmo = gfx.TransformGizmo(obj)
            self.transform_gizmo.add_default_event_handlers(self.renderer, self.camera)
        else:
            self.transform_gizmo._object_to_control = obj

    @property
    def _is_jupyter(self):
        """Check if Viewer is using Jupyter canvas."""
        return "Jupyter" in str(type(self.canvas))

    @property
    def _is_offscreen(self):
        """Check if Viewer is using offscreen canvas."""
        return isinstance(self.canvas, OffscreenRenderCanvas)

    @property
    def _window_manager(self):
        """Which window manager is being used."""
        try:
            return type(self.canvas).__module__.split(".")[-1]
        except BaseException:
            return "na"

    @property
    def _object_ids(self):
        """All object IDs on this canvas in order of addition."""
        obj_ids = []
        for v in self.visuals:
            if hasattr(v, "_object_id"):
                obj_ids.append(v._object_id)
        return sorted(set(obj_ids), key=lambda x: obj_ids.index(x))

    @property
    def objects(self):
        return self._objects()

    @lru_cache(maxsize=1)
    def _objects(self):
        """Ordered dictionary {name->[visuals]} of all objects in order of addition."""
        objects = OrderedDict()
        for v in self.visuals:
            if hasattr(v, "_object_id"):
                if v._object_id in objects:
                    objects[v._object_id].append(v)
                else:
                    objects[v._object_id] = [v]

        return objects

    @property
    def objects_grouped(self):
        """Ordered dictionary {group_name: [object_ids]} of all groups. Ungrouped objects are omitted."""
        groups = OrderedDict()
        for obj_id, visuals in self.objects.items():
            group_name = getattr(visuals[0], "_object_group", None)
            if group_name is not None:
                if group_name not in groups:
                    groups[group_name] = []
                groups[group_name].append(obj_id)
        return groups

    @property
    def objects_pickable(self):
        return self._objects_pickable

    @objects_pickable.setter
    def objects_pickable(self, v):
        if not isinstance(v, bool):
            raise TypeError(f"Expected bool, got {type(v)}")

        # No need to do anything if the value is the same
        if v == self._objects_pickable:
            return

        self._objects_pickable = v

        # Set pick_write to new value for all materials
        for objects in self.objects.values():
            for ob in objects:
                try:
                    ob.material.pick_write = v
                except AttributeError:
                    pass

    @property
    def highlighted(self):
        """Return IDs of currently highlighted objects."""
        highlighted = []
        for obj in self.objects:
            if any([getattr(v, "_highlighted", False) for v in self.objects[obj]]):
                highlighted.append(obj)
        return highlighted

    @property
    def on_hover(self):
        """Determines what to do when hovering over objects.

        Can be set to:
         - `None`: do nothing
         - "highlight": hide object

        """
        return self._on_hover

    @on_hover.setter
    def on_hover(self, v):
        valid = (None, "highlight")
        if v not in valid:
            raise ValueError(
                f"Unknown value for on_hover: {v}. Must be one of {valid}."
            )

        # No need to do anything if the value is the same
        if v == self._on_hover:
            return

        if v:
            # Make objects pickable
            self.objects_pickable = True

            # Add the event handler
            self.scene.add_event_handler(self._highlight_on_hover_event, "pointer_move")
        else:
            self.scene.remove_event_handler(
                self._highlight_on_hover_event, "pointer_move"
            )
            current_hover = getattr(self, "_current_hover_object", None)

            # Make sure to unhighlight the current hover object
            if current_hover:
                self.unhighlight_objects(current_hover)
                self._current_hover_object = None

        self._on_hover = v

    def _highlight_on_hover_event(self, event):
        """This is the event callback for highlighting objects on hover."""
        # If any buttons are pressed (e.g. mouse left during panning) ignore the event
        if event.buttons:
            return

        # Parse the current object
        new_hover = event.pick_info["world_object"]
        current_hover = getattr(self, "_current_hover_object", None)

        # Break early if there is nothing to do
        if new_hover is None and current_hover is None:
            # print("  No hover")
            return

        new_hover_id = [k for k, v in self.objects.items() if new_hover in v]
        new_hover_id = new_hover_id[0] if new_hover_id else None

        # See if we need to de-highlight the current hover object
        if current_hover:
            # If the new object is the same as the current one, we don't need to do anything
            if current_hover == new_hover_id:
                return
            if current_hover in self.objects:
                self.unhighlight_objects(current_hover)
            self._current_hover_object = None

        # Highlight the new object
        if new_hover_id:
            self.highlight_objects(
                new_hover_id, color=getattr(self, "_highlight_on_hover_color", 0.2)
            )
            self._current_hover_object = new_hover_id

    @property
    def on_double_click(self):
        """Determines what to do when double clicking on objects.

        Can be set to:
         - `None`: do nothing
         - "hide": hide object
         - "remove": remove object
         - "select": select object
         - callable: a custom function that takes as input `event` and `viewer`

        See `octarine.viewer.handle_object_event` for an example of how to write a custom function for this.

        """
        return self._on_double_click

    @on_double_click.setter
    def on_double_click(self, v):
        valid = (None, "hide", "remove", "select")
        if v not in valid and not callable(v):
            raise ValueError(
                f"Unknown value for on_double_click: {v}. Must be one of {valid}."
            )

        # No need to do anything if the value is the same
        if v == self._on_double_click:
            return

        # First try to remove the current event handler for double clicks
        try:
            self.scene.remove_event_handler(
                getattr(self, "_on_double_click_func", None), "double_click"
            )
        except KeyError:
            pass

        if v:
            # Make objects pickable
            self.objects_pickable = True

            # Now add the new event handler
            if not callable(v):
                func = partial(handle_object_event, viewer=self, actions=(v,))
            else:
                func = partial(v, viewer=self)
            self.scene.add_event_handler(func, "double_click")
            self._on_double_click_func = func
        else:
            self._on_double_click_func = None

        self._on_double_click = v

    def add_animation(self, x, on_error="remove", run_every=None, req_render=True):
        """Add animation function to the Viewer.

        Parameters
        ----------
        x :         callable
                    Function to add to the animation loop.
        on_error :  "remove" | "ignore" | "raise" | "log"
                    What to do if the function throws an error. If "remove",
                    the function will be removed from the animation loop. If
                    "ignore", the error will be ignored and the function will
                    continue to be called.
        run_every : int, optional
                    Use to run the function every n frames.
        req_render : bool, optional
                    Whether this animation requires a re-render of the scene.
                    This is mainly a flag to help the viewer to decide
                    whether/when to trigger a render. See also the `render_trigger`
                    property.

        """
        if not callable(x):
            raise TypeError(f"Expected callable, got {type(x)}")

        assert on_error in ["remove", "ignore", "raise", "log"]

        self._animations[x] = (on_error, run_every, req_render)

    def remove_animation(self, x):
        """Remove animation function from the Viewer.

        Parameters
        ----------
        x :     callable | int
                Either the function itself or its index
                in the list of animations.

        """
        if callable(x):
            self._animations_flagged_for_removal.append(x)
        elif isinstance(x, int):
            self._animations_flagged_for_removal.append(
                list(self._animations.keys())[x]
            )
        else:
            raise TypeError(f"Expected callable or index (int), got {type(x)}")

    @staticmethod
    def _effect_stage(effect_pass):
        """Where in the chain a post-processing pass belongs."""
        stage = getattr(effect_pass, "_octarine_stage", None)
        if stage is not None:
            return stage
        if isinstance(effect_pass, PPAAPass):
            return PPAA_EFFECT_STAGE
        return DEFAULT_EFFECT_STAGE

    def _add_effect_pass(self, effect_pass, stage=DEFAULT_EFFECT_STAGE):
        """Insert a post-processing pass at its place in the chain.

        Passes are ordered by `stage` (see `EFFECT_STAGES`) rather than by
        the order in which they were switched on, so that e.g. switching
        ambient occlusion on after depth of field still occludes first and
        blurs second. Passes we did not add ourselves - anything the user
        put into `renderer.effect_passes` directly - count as the default
        stage and keep their relative order.

        """
        effect_pass._octarine_stage = stage
        passes = list(self.renderer.effect_passes)
        index = len(passes)
        for i, other in enumerate(passes):
            if self._effect_stage(other) > stage:
                index = i
                break
        passes.insert(index, effect_pass)
        self.renderer.effect_passes = tuple(passes)

    def add_effect(self, effect, disable=False, **kwargs):
        """Add post-processing effect to the renderer.

        You can also use this method to adjust the parameters of an existing
        effect or to remove an effect (see the `disable` parameter).

        Parameters
        ----------
        effect :   str
                    Name of the effect to add. Currently supported:
                     - "edl" (Eye-Dome Lighting)
                       This effect enhances depth perception for complex
                       geometries by darkening edges based on depth differences.
                     - "noise"
                       Adds noise to the full image.
                     - "fog"
                       Adds fog to the full image, using the depth buffer.
                     - "depth"
                       Renders scene depth as shades of grey (near = dark,
                       far = light), normalized to the depth range of the
                       visible geometry; the background stays white. With
                       `overlay=True` the objects' own colors are kept and
                       darkened with distance instead (depth cueing).
                     - "ao"
                       Screen-space ambient occlusion: darkens creases,
                       cavities and the contact points between objects.
                       See also `Viewer.set_ambient_occlusion`.
                     - "outline"
                       Draws a line around silhouettes and along creases,
                       the way a technical illustration would. See also
                       `Viewer.set_outline`.
                     - "tonemap"
                       Compresses the rendered high dynamic range image into
                       what the display can show, so bright regions roll off
                       instead of clipping to white; also provides the
                       exposure control. See also `Viewer.set_tonemapping`.
                     - "normal"
                       Renders normals reconstructed from the depth buffer.
                     - "bloom"
                       Physically-based bloom effect; makes bright regions
                       glow. Best suited for HDR rendering pipelines.

        disable :   bool
                    If True, the effect is removed from the renderer instead
                    of added. Any `**kwargs` are ignored in that case.

        **kwargs
                    Keyword arguments passed to the effect constructor:
                    - edl:
                      - strength (default 5): EDL strength; typical range ~ [0.5, 10.0].
                      - radius (default 1.5): sampling radius in pixels
                      - depth_edge_threshold (default 0.0)
                    - noise:
                      - noise (default 0.1): amount of noise to add
                    - fog:
                      - color (default "#fff"): fog color
                      - power (default 1.0): how quickly fog thickens with depth
                    - depth:
                      - camera (default: the viewer's camera): used to
                        linearize depth values
                      - overlay (default False): darken the objects' own
                        colors by depth instead of rendering greyscale
                      - strength (default 0.9): how dark the farthest
                        geometry gets, from 0 (not at all) to 1 (black /
                        fully darkened)
                    - ao:
                      - radius (default: 4% of the scene's diagonal): how
                        far to look for occluders, in world units
                      - intensity (default 1): strength of the darkening
                      - bias (default 0.01): fraction of `radius` below
                        which occluders are ignored
                      - samples (default 16): samples per pixel
                      - power (default 1): exponent applied to the occlusion
                      - blur (default True): radius of the bilateral blur
                      - debug (default False): render the occlusion itself
                    - outline:
                      - camera (default: the viewer's camera)
                      - color (default "#000"): outline color; its alpha is
                        the strength of the effect
                      - thickness (default 1): width in physical pixels
                      - depth_threshold (default 0.02): relative step in
                        depth that counts as a separate object
                      - normal_threshold (default 0.3): how sharp a fold
                        counts as a crease; 0 outlines silhouettes only
                      - debug (default False): render the edges themselves
                    - tonemap:
                      - mode (default "aces"): "aces", "filmic", "reinhard"
                        or "none"
                      - exposure (default 1): scales the image before the
                        tone mapping curve is applied
                      - white_point (default 4): input value that maps to
                        white ("reinhard" and "filmic" only)
                    - normal: no parameters
                    - bloom:
                      - bloom_strength (default 0.04): strength of the bloom
                      - max_mip_levels (default 6): number of mip levels used
                      - filter_radius (default 0.005): upsampling filter radius
                      - use_karis_average (default False): reduces fireflies


        """
        if effect not in EFFECT_CLASSES:
            raise ValueError(f"Unknown effect: {effect}")

        effect_cls = EFFECT_CLASSES[effect]
        if effect == "depth":
            # Our own normalized-depth shader; imported lazily because
            # custom shaders require pygfx>=0.17
            from .shaders import NormalizedDepthPass

            effect_cls = NormalizedDepthPass
            kwargs.setdefault("camera", self.camera)
        elif effect == "ao":
            from .shaders import AmbientOcclusionPass

            effect_cls = AmbientOcclusionPass
            kwargs.setdefault("camera", self.camera)
            # An explicit radius pins the effect to it, otherwise we keep
            # deriving it from the scene (see `Viewer._update_ao_radius`)
            self._ao_auto_radius = "radius" not in kwargs
            kwargs.setdefault("radius", self._default_ao_radius())
        elif effect == "outline":
            from .shaders import OutlinePass

            effect_cls = OutlinePass
            kwargs.setdefault("camera", self.camera)
        elif effect == "tonemap":
            from .shaders import ToneMappingPass

            effect_cls = ToneMappingPass

        # Check if we already have this effect
        p = None
        for e in self.renderer.effect_passes:
            if isinstance(e, effect_cls):
                p = e
                break

        if disable:
            if p is not None:
                self.renderer.effect_passes = tuple(
                    e for e in self.renderer.effect_passes if e is not p
                )
                for attr in ("_ao_pass", "_outline_pass", "_tonemap_pass"):
                    if p is getattr(self, attr, None):
                        setattr(self, attr, None)
            return

        if p is None:
            # Overwrite the default of 1 (seems too weak in my hands)
            if (effect_cls is EDLPass) and "strength" not in kwargs:
                kwargs["strength"] = 5.0

            p = effect_cls(**kwargs)
            self._add_effect_pass(p, EFFECT_STAGES.get(effect, DEFAULT_EFFECT_STAGE))
            # Keep the dedicated `set_*` methods and `add_effect` on the same
            # pass instead of each adding one of their own
            if effect == "ao":
                self._ao_pass = p
            elif effect == "outline":
                self._outline_pass = p
            elif effect == "tonemap":
                self._tonemap_pass = p
        else:
            # Update parameters
            for k, v in kwargs.items():
                if hasattr(p, k):
                    setattr(p, k, v)
                else:
                    raise ValueError(f"Effect '{effect}' has no parameter '{k}'")
            if effect == "ao":
                # May have been switched off via `set_ambient_occlusion(False)`
                p.enabled = True

    def show(self, use_sidecar=False, toolbar=False, start_loop=False):
        """Show viewer.

        Parameters
        ----------
        use_sidecar : bool
                      Jupyter lab only: if True, will use the Sidecar extension
                      to display the viewer outside the notebooks. Will throw
                      an error if Sidecar is not installed.
        toolbar :     bool
                      Jupyter lab only: if True, will show a toolbar. You can
                      always show/hide the toolbar with ``viewer.show_controls()``
                      and ``viewer.hide_controls()``, or the `c` hotkey.
        start_loop :  bool
                      Scripts & standard REPL only:
                      If True, will start the blocking (!) event loop. This is
                      the recommended way to show the viewer when using it in a script.
                      From an interactive REPL such as IPython you should be able to
                      just call ``Viewer.show()`` and the interactive viewer will appear
                      while still allowing you to interact with the REPL.

        """
        # This is for e.g. headless testing
        if getattr(config, "HEADLESS", False):
            logger.info("Viewer widget not shown - running in headless mode.")
            return

        # Start the animation loop
        self.canvas.request_draw(self._animate)

        # If this is an offscreen canvas, we don't need to do anything else
        if isinstance(self.canvas, OffscreenRenderCanvas):
            return

        # In terminal we can just show the window
        if not self._is_jupyter:
            # Not all backends have a show method (e.g. GLFW does not)
            if hasattr(self.canvas, "show"):
                self.canvas.show()

            if start_loop:
                from rendercanvas.auto import loop

                loop.run()
            elif utils._type_of_script() in ("terminal", "script") and os.environ.get(
                "OCTARINE_CHECK_LOOP", "1"
            ) in ("1", "true", "True"):
                logger.warning(
                    "Running in a (potentially) non-interactive terminal or script "
                    "environment. You may have to manually start the event loop "
                    "for the canvas to render:\n\n"
                    "  >>> v = octarine.Viewer(show=False)\n"
                    "  >>> ...  # setup your viewer\n"
                    "  >>> v.show(start_loop=True)\n\n"
                    "Alternatively, use the loop.run() function:\n\n"
                    "  >>> from rendercanvas.auto import loop\n"
                    "  >>> ...  # setup your viewer\n"
                    "  >>> v.show()\n"
                    "  >>> loop.run()\n\n"  # do not remove the \n\n here
                )
        else:
            # if not hasattr(self, 'widget'):
            from .jupyter import JupyterOutput
            from IPython.display import display

            # Construct the widget
            if not hasattr(self, "widget"):
                self.widget = JupyterOutput(
                    self,
                    use_sidecar=use_sidecar,
                    toolbar=toolbar,
                    sidecar_kwargs={"title": self._title},
                )

            # This will display the viewer right here and there
            display(self.widget)

    def show_message(
        self, message, position="top-right", font_size=20, color=None, duration=None
    ):
        """Show message on canvas.

        Parameters
        ----------
        message :   str | None
                    Message to show. Set to `None` to remove the existing message.
        position :  "top-left" | "top-right" | "bottom-left" | "bottom-right" | "center"
                    Position of the message on the canvas.
        font_size : int, optional
                    Font size of the message.
        color :     str | tuple, optional
                    Color of the message. If `None`, will use white.
        duration :  int, optional
                    Number of seconds after which to fade the message.

        """
        if message is None and hasattr(self, "_message_text"):
            if self._message_text.parent:
                self.overlay_scene.remove(self._message_text)
            del self._message_text
            return

        _positions = {
            "top-left": (-0.95, 0.95, 0),
            "top-right": (0.95, 0.95, 0),
            "bottom-left": (-0.95, -0.95, 0),
            "bottom-right": (0.95, -0.95, 0),
            "center": (0, 0, 0),
        }
        if position not in _positions:
            raise ValueError(f"Unknown position: {position}")

        if not hasattr(self, "_message_text"):
            self._message_text = text2gfx(
                message, color="white", font_size=font_size, screen_space=True
            )

        # Make sure the text is in the scene
        if self._message_text not in self.overlay_scene.children:
            self.overlay_scene.add(self._message_text)

        self._message_text.set_text(message)
        self._message_text.font_size = font_size
        self._message_text.anchor = position
        if color is not None:
            self._message_text.material.color = cmap.Color(color).rgba
        self._message_text.material.opacity = 1
        self._message_text.local.position = _positions[position]

        # When do we need to start fading out?
        if duration:
            self._fade_out_time = time.time() + duration

            def _fade_message():
                if not hasattr(self, "_message_text"):
                    self.remove_animation(_fade_message)
                else:
                    if time.time() > self._fade_out_time:
                        # This means the text will fade fade over 1/0.02 = 50 frames
                        self._message_text.material.opacity = max(
                            self._message_text.material.opacity - 0.02, 0
                        )

                    if self._message_text.material.opacity <= 0:
                        if self._message_text.parent:
                            self.overlay_scene.remove(self._message_text)
                        self.remove_animation(_fade_message)

            self.add_animation(_fade_message)

    def set_scalebar(
        self,
        size="auto",
        units=None,
        position="bottom-right",
        color="w",
        width=3,
        font_size=14,
        label=True,
        margin=20,
    ):
        """Add (or remove) a scale bar overlay.

        The scale bar is drawn on top of the scene and indicates a given
        distance in world units. It automatically tracks zoom level and
        canvas size.

        Note that this requires an orthographic camera (the default): with a
        perspective camera the scale depends on the distance from the camera
        and a single bar would be meaningless. If the camera is (or becomes)
        perspective, the scale bar is hidden until it is orthographic again.

        Parameters
        ----------
        size :      float | "auto" | False
                    Length of the scale bar in world units. If "auto"
                    (default), the bar is dynamically re-sized as you zoom
                    to a "nice" round number spanning roughly a quarter of
                    the canvas. Use `viewer.set_scalebar(False)` to remove
                    an existing scale bar.
        units :     str, optional
                    Units to append to the label, e.g. "nm" or "µm". Note
                    that Octarine has no notion of the units of your data -
                    this is simply used for the label.
        position :  "bottom-right" | "bottom-left" | "top-right" | "top-left"
                    Corner of the canvas to place the scale bar in.
        color :     str | tuple
                    Color of the bar and its label.
        width :     float
                    Thickness of the bar in pixels.
        font_size : int
                    Font size of the label in pixels.
        label :     bool | str
                    Whether to label the bar with its size. Set to `False`
                    for a bare bar, or pass a string to use a fixed custom
                    label instead of the size.
        margin :    int
                    Distance (in pixels) of the bar from the canvas edges.

        Examples
        --------
        >>> import octarine as oc
        >>> v = oc.Viewer()
        >>> # A bar that adjusts to the zoom level
        >>> v.set_scalebar(units="nm")
        >>> # A fixed 1000 nm bar in the top-left corner
        >>> v.set_scalebar(1000, units="nm", position="top-left")
        >>> # Remove the scale bar again
        >>> v.set_scalebar(False)

        """
        # Skip if running in headless mode
        if getattr(config, "HEADLESS", False):
            return

        if size is False or size is None:
            if getattr(self, "_scalebar", None) is not None:
                self.overlay_scene.remove(self._scalebar)
                self.remove_animation(self._update_scalebar)
                self._scalebar = None
                self._render_stale = True
            return

        if isinstance(size, str):
            if size != "auto":
                raise ValueError(f'Expected a number, "auto" or False, got "{size}"')
        else:
            size = float(size)
            if size <= 0:
                raise ValueError(f"Scale bar size must be positive, got {size}")

        if position not in ("bottom-right", "bottom-left", "top-right", "top-left"):
            raise ValueError(f"Unknown position: {position}")

        if self.camera.fov != 0:
            raise ValueError(
                "Scale bars require an orthographic camera but this viewer's "
                f"camera has a field of view of {self.camera.fov}. Set "
                "`Viewer.camera.fov = 0` to make it orthographic."
            )

        sb = getattr(self, "_scalebar", None)
        if sb is None:
            sb = self._scalebar = gfx.Group()
            # The bar is a unit quad that `_update_scalebar` positions and
            # scales in NDC coordinates. Note that we're using a mesh rather
            # than a line because pygfx lines have caps which would make the
            # bar wider than the distance it represents.
            sb._bar = gfx.Mesh(gfx.plane_geometry(1, 1), gfx.MeshBasicMaterial())
            sb._text = text2gfx(
                "", font_size=font_size, anchor="bottom-center", screen_space=True
            )
            sb._label = ""
            sb._state = None
            sb.add(sb._bar, sb._text)
            self.overlay_scene.add(sb)

        sb._bar.material.color = gfx.Color(color)
        sb._text.font_size = font_size
        sb._text.material.color = gfx.Color(color)
        sb._text.visible = bool(label)

        self._scalebar_config = {
            "size": size,
            "units": units,
            "position": position,
            "width": width,
            "label": label,
            "margin": margin,
        }

        # This keeps the bar in sync with the camera and the canvas size.
        # Note that removals are deferred to the next frame, so we have to
        # cancel any pending one in case the bar was removed and re-added
        # in between two frames.
        if self._update_scalebar in self._animations_flagged_for_removal:
            self._animations_flagged_for_removal.remove(self._update_scalebar)
        self.add_animation(self._update_scalebar, on_error="log", req_render=False)
        self._update_scalebar()
        self._render_stale = True

    def _update_scalebar(self):
        """Animation hook: sync scale bar with the camera and canvas size."""
        sb = getattr(self, "_scalebar", None)
        if sb is None:
            return

        conf = self._scalebar_config

        # A scale bar is meaningless for a perspective camera because the
        # scale depends on the distance from the camera
        if self.camera.fov != 0:
            if sb.visible:
                logger.warning(
                    "Hiding scale bar: this requires an orthographic camera."
                )
                sb.visible = False
                self._render_stale = True
            return
        sb.visible = True

        # We run before the renderer, which is what normally syncs the camera
        # with the canvas. Do it here so that the bar has the correct length
        # already on the very first frame (e.g. for offscreen screenshots).
        width_px, height_px = self.renderer.logical_size
        if not width_px or not height_px:
            return
        self.camera.set_view_size(width_px, height_px)

        # NDC units per world unit along the horizontal axis of the screen:
        # the orthographic projection maps the visible width (which already
        # accounts for zoom and aspect ratio) onto the -1 to 1 NDC range
        ndc_per_world = float(self.camera.projection_matrix[0, 0])
        if ndc_per_world <= 0:
            return

        size = conf["size"]
        if size == "auto":
            # Aim for a bar spanning a quarter of the canvas (= 0.5 in NDC)
            # and round that down to the nearest nice number
            size = _nice_number(0.5 / ndc_per_world)

        length = size * ndc_per_world
        margin_x = 2 * conf["margin"] / width_px
        margin_y = 2 * conf["margin"] / height_px

        top = conf["position"].startswith("top")
        if conf["position"].endswith("right"):
            x1 = 1 - margin_x
            x0 = x1 - length
        else:
            x0 = -1 + margin_x
            x1 = x0 + length
        y = 1 - margin_y if top else -1 + margin_y
        thickness = 2 * conf["width"] / height_px

        # Only touch the visuals (and trigger a re-render) if anything changed
        state = (x0, x1, y, thickness, top)
        if state != sb._state:
            sb._state = state
            sb._bar.local.position = ((x0 + x1) / 2, y, 0)
            sb._bar.local.scale = (length, thickness, 1)
            # Label sits just above the bar (below it if the bar is at the top)
            offset = thickness / 2 + 8 / height_px
            sb._text.local.position = (
                (x0 + x1) / 2,
                y - offset if top else y + offset,
                0,
            )
            sb._text.anchor = "top-center" if top else "bottom-center"
            self._render_stale = True

        label = conf["label"]
        if not isinstance(label, str):
            label = _format_number(size)
            if conf["units"]:
                label = f"{label} {conf['units']}"
        if label != sb._label:
            sb._text.set_text(label)
            sb._label = label
            self._render_stale = True

    def show_controls(self):
        """Show controls."""
        if self._is_jupyter:
            if self.widget.toolbar:
                self.widget.toolbar.show()
        else:
            if not hasattr(self, "_controls"):
                from .controls import Controls

                self._controls = Controls(self)
            self._controls.show()

    def hide_controls(self):
        """Hide controls."""
        if self._is_jupyter:
            if self.widget.toolbar:
                self.widget.toolbar.hide()
        else:
            if hasattr(self, "_controls"):
                self._controls.hide()

    def _toggle_controls(self):
        """Switch controls on and off."""
        if self._is_jupyter:
            if self.widget.toolbar:
                self.widget.toolbar.toggle()
        else:
            if not hasattr(self, "_controls"):
                self.show_controls()
            elif self._controls.isVisible():
                self.hide_controls()
            else:
                self.show_controls()

    @update_viewer(legend=True, bounds=True)
    def clear(self):
        """Clear canvas of objects (expects lights and background)."""
        # Skip if running in headless mode
        if getattr(config, "HEADLESS", False):
            return

        # Remove everything but the lights and backgrounds
        self.scene.remove(*self.visuals)

        # Rset the transform gizmo
        self.transform_gizmo = None

    @update_viewer(legend=True, bounds=True)
    def remove_objects(self, to_remove):
        """Remove given neurons/visuals from canvas."""
        to_remove = utils.make_iterable(to_remove)

        for vis in self.scene.children:
            if vis in to_remove:
                self.scene.remove(vis)
            elif hasattr(vis, "_object_id"):
                if vis._object_id in to_remove:
                    self.scene.remove(vis)

    @update_viewer(legend=True, bounds=True)
    def pop(self, N=1):
        """Remove the most recently added N visuals."""
        for vis in list(self.objects.values())[-N:]:
            self.remove_objects(vis)

    @property
    def show_bounds(self):
        """Set to ``True`` to show bounding box."""
        return self._show_bounds

    @property
    def show_fps(self):
        """Show frames per second."""
        return self._show_fps

    @show_fps.setter
    def show_fps(self, v):
        if not isinstance(v, bool):
            raise TypeError(f"Expected bool, got {type(v)}")
        self._show_fps = v
        self._render_stale = True

    def toggle_bounds(self):
        """Toggle bounding box."""
        self.show_bounds = not self.show_bounds

    @show_bounds.setter
    def show_bounds(self, v):
        if not isinstance(v, bool):
            raise TypeError(f"Need bool, got {type(v)}")

        self._show_bounds = v

        if self.show_bounds:
            self.update_bounds()
        else:
            self.remove_bounds()

    def remove_bounds(self):
        """Remove bounding box visual."""
        self._show_bounds = False
        for v in self.visuals:
            if getattr(v, "_object_type", "") == "boundingbox":
                self.remove_objects(v)

    def resize(self, size):
        """Resize canvas.

        Parameters
        ----------
        size :  (width, height) tuple
                New size of the canvas.
        """
        assert len(size) == 2
        self.canvas.set_logical_size(*size)

    def update_bounds(self, color="w", width=1):
        """Update bounding box visual."""
        # Remove any existing visual
        self.remove_bounds()

        self._show_bounds = True

        # Skip if no visual on canvas
        bounds = self.scene.get_bounding_box()
        if isinstance(bounds, type(None)):
            return

        # Create box visual
        box = gfx.BoxHelper()
        box.set_transform_by_aabb(bounds)

        # Add custom attributes
        box._object_type = "boundingbox"
        box._object_id = uuid.uuid4()

        self.scene.add(box)

    def _request_center(self):
        """Ask for the camera to be centered on the scene before the next frame.

        Centering frames the entire scene graph, which is O(number of objects)
        - doing it inside every `add` is what made filling a viewer quadratic
        (see `Viewer._refresh_scene`). We remember the camera as we leave it so
        that the deferred centering can tell whether anybody has taken the
        camera over in the meantime - a `camera.show_object` of their own, say,
        or a drag of the controller. Centering there and then would have been
        overruled by that just the same, so in that case we skip it.

        """
        self._center_pending = True
        self._centered_camera_sig = self._camera_sig()

    def center_camera(self):
        """Center camera on visuals."""
        # Adding objects only asks for this to happen before the next frame
        # (see `Viewer._refresh_scene`) - doing it now makes that redundant
        self._center_pending = False
        if len(self):
            self.camera.show_object(
                self.scene, scale=1, view_dir=(0.0, 0.0, 1.0), up=(0.0, -1.0, 0.0)
            )
            self._sync_linked()

    @property
    def linked(self):
        """Viewers this viewer's camera is linked with (see `Viewer.link`)."""
        return tuple(self._linked)

    def _camera_sig(self):
        """Cheap fingerprint of the camera state - used to detect movement."""
        return (
            *self.camera.local.position,
            *self.camera.local.rotation,
            self.camera.width,
            self.camera.height,
            self.camera.zoom,
            self.camera.fov,
        )

    def _sync_linked(self):
        """Push this viewer's camera state to any linked viewers."""
        if not self._linked:
            return

        state = _filter_camera_state(self.camera.get_state(), *self._link_filter)
        for v in self._linked:
            v.camera.set_state(state)
            # The other viewers don't know that anything happened, so we have to
            # ask them to re-render themselves
            v._render_stale = True
            v.canvas.request_draw()

    def link(self, *others, sync=None, exclude=None):
        """Keep the camera synchronised with (an)other viewer(s).

        Panning, rotating or zooming in any of the linked viewers moves the
        cameras in all the others as well. The same goes for programmatic
        changes via [`Viewer.set_view`][octarine.Viewer.set_view] and
        [`Viewer.center_camera`][octarine.Viewer.center_camera] (including the
        implicit centering when adding objects) but not for changes made
        directly on the `Viewer.camera` object.

        Links are symmetrical and transitive: linking `A` to `B` and then `B` to
        `C` means that all three viewers move together. On linking, the other
        viewers immediately adopt this viewer's current view.

        Parameters
        ----------
        *others :   Viewer | list thereof
                    Viewer(s) to link with this one.
        sync :      str | list of str, optional
                    Which fields of the camera state to synchronise. If `None`
                    (default) everything is synchronised. The three interactive
                    controls map onto "position" (panning; can also be addressed
                    as the individual "x", "y" and "z"), "rotation" (rotating)
                    and "width" + "height" (zooming).
        exclude :   str | list of str, optional
                    The inverse of `sync`: which fields of the camera state to
                    keep independent. Can be combined with `sync`.
                    Note that `sync`/`exclude` apply to the entire group, i.e.
                    linking a new viewer into an existing group also (re-)sets
                    the filter for the viewers that were already in it.

        See Also
        --------
        [`Viewer.unlink`][octarine.Viewer.unlink]
                    Break the link again.
        [`Viewer.linked`][octarine.Viewer.linked]
                    The viewers currently linked with this one.

        Examples
        --------
        >>> import octarine as oc
        >>> v1, v2 = oc.Viewer(), oc.Viewer()
        >>> v1.link(v2)                      # fully link the two viewers
        >>> v1.unlink()                      # ... and unlink them again
        >>> v1.link(v2, sync='rotation')     # only synchronise the rotation
        >>> v1.link(v2, exclude=['width', 'height'])  # ... or zoom separately

        """
        others = _flatten_viewers(others, "link")
        if not others:
            raise ValueError("Must provide at least one viewer to link with.")
        if any(v is self for v in others):
            raise ValueError("Can not link a viewer with itself.")

        sync = _parse_state_fields(sync)
        exclude = _parse_state_fields(exclude)

        # Collect the full group: the viewers to link plus whatever they (and
        # we) were already linked with
        group = []
        for v in [self] + others:
            for w in [v] + list(v._linked):
                if w not in group:
                    group.append(w)

        # A group that mixes orthographic with perspective cameras must not
        # synchronise the field of view: the orthographic cameras have theirs
        # locked to zero and would flatten the perspective ones (pygfx clamps
        # the fov in the other direction). Unless explicitly asked to, that is.
        is_ortho = {isinstance(v.camera, gfx.OrthographicCamera) for v in group}
        if len(is_ortho) > 1 and (sync is None or "fov" not in sync):
            exclude = (exclude or set()) | {"fov"}

        # Let every viewer's controller drive every other viewer's camera. Note
        # that each controller already has its own camera registered first which
        # is important because that's the one it reads the current state from.
        for v in group:
            v._linked = [w for w in group if w is not v]
            v._link_filter = (sync, exclude)
            for w in v._linked:
                v.controller.add_camera(
                    w.camera, include_state=sync, exclude_state=exclude
                )

        # Make the others adopt our view so we start out in sync
        self._sync_linked()

    def unlink(self, *others):
        """Unlink viewers such that their cameras move independently again.

        Parameters
        ----------
        *others :   Viewer | list thereof, optional
                    Viewer(s) to remove from this viewer's link group. If not
                    provided, this viewer itself is removed and any other
                    viewers in the group stay linked with each other. Viewers
                    that aren't part of this viewer's group are silently
                    ignored.

        See Also
        --------
        [`Viewer.link`][octarine.Viewer.link]
                    Link viewers in the first place.

        """
        others = _flatten_viewers(others, "unlink")

        group = [self] + list(self._linked)
        drop = [v for v in (others or [self]) if v in group]
        if not drop:
            return
        keep = [v for v in group if v not in drop]

        for v in drop:
            for w in group:
                if w is not v:
                    v.controller.remove_camera(w.camera)
                    w.controller.remove_camera(v.camera)
            v._linked = []
            v._link_filter = (None, None)

        for v in keep:
            v._linked = [w for w in keep if w is not v]
            if not v._linked:
                v._link_filter = (None, None)

    @update_viewer(legend=True, bounds=True)
    def add(self, x, name=None, group=None, center=True, clear=False, **kwargs):
        """Add object to canvas.

        This function is a general entry point for adding objects to the canvas.
        It will look at the type of the input and try to find an appropriate
        function to convert the input to visuals.

        Use `octarine.register_converter` to add custom converters.

        Parameters
        ----------
        x
                    Object(s) to add to the canvas.
        name :      str, optional
                    Name for the visual(s).
        group :     str, optional
                    Group for the visual(s).
        center :    bool, optional
                    If True, re-center camera to all objects on canvas.
        clear :     bool, optional
                    If True, clear canvas before adding new objects.
        **kwargs
                    Keyword arguments passed to the conversion functions when
                    generating visuals.

        Returns
        -------
        None

        """
        if clear:
            self.clear()

        converter = get_converter(x, raise_missing=False)
        if utils.is_iterable(x) and not converter:
            for xx in x:
                self.add(xx, center=False, clear=False, name=name, **kwargs)
            if center:
                self._request_center()
            return

        if converter is None:
            raise NotImplementedError(f"No converter found for {x} ({type(x)})")

        # Check if we have to provide a color
        if "color" not in kwargs and "color" in inspect.signature(converter).parameters:
            kwargs["color"] = tuple(self._next_color().rgba)

        visuals = utils.make_iterable(converter(x, **kwargs))

        for v in visuals:
            # If we have a name, assign it to the visual
            if name is not None:
                v._object_id = name
            # If not we either use existing ID or generate a new one
            else:
                # Give visuals an _object_id if they don't already have one
                if not hasattr(v, "_object_id"):
                    new_id = self._next_label("Object")
                    for v2 in visuals:
                        v._object_id = new_id
                elif not isinstance(v._object_id, str):
                    v._object_id = str(v._object_id)

            v._object_group = group

            self.scene.add(v)

        # Note this is deferred to just before the next frame rather than done
        # here - see `Viewer._request_center`
        if center:
            self._request_center()

    @update_viewer(legend=True, bounds=True)
    def _add_to_scene(self, visual, center=True):
        """Add visual to scene.

        This is just a convenient collection point for us to trigger a bunch of updates in one go,
        """
        # If we need objects to be pickable, set the material accordingly
        if self.objects_pickable:
            try:
                visual.material.pick_write = True
            except AttributeError:
                pass

        self.scene.add(visual)

        # Note this is deferred to just before the next frame rather than done
        # here - see `Viewer._request_center`
        if center:
            self._request_center()

    def add_mesh(
        self,
        mesh,
        name=None,
        group=None,
        color=None,
        alpha=None,
        silhouette=None,
        subsurface=None,
        shader=None,
        matcap=None,
        center=True,
    ):
        """Add mesh to canvas.

        Parameters
        ----------
        mesh :      Mesh-like
                    Mesh to plot. If this is a pygfx.Mesh, it will be added
                    directly to the scene without modification (i.e. `color`,
                    `alpha`, etc. will be ignored).
        name :      str, optional
                    Name for the visual.
        group :     str, optional
                    Group for the visual.
        color :     str | tuple, optional
                    Color to use for plotting. If multiple colors,
                    must be a list of colors with the same length as
                    the number of faces or vertices.
        alpha :     float, optional
                    Opacity value [0-1]. If provided, will override
                    the alpha channel of the color.
        silhouette : float, optional
                    If provided (and > 0), render the mesh with a
                    Neuroglancer-style silhouette effect: face-on regions
                    become transparent while edges/creases are emphasized.
                    Typical values are 1-8 (same exponent semantics as
                    Neuroglancer). Use `Viewer.set_silhouette` to toggle
                    the effect on existing meshes. Only works with the
                    default "phong" shader.
        subsurface : float | dict, optional
                    If provided (and > 0), render the mesh with subsurface
                    scattering: light bleeds through the surface so that
                    backlit and thin regions glow, as with skin, wax or
                    leaves. A float sets the strength (typical values are
                    0.5-2); pass a dict to also set `scatter_color`,
                    `thickness`, `distortion`, `falloff`, `wrap` or `glow`
                    - e.g. `{"subsurface": 1.5, "scatter_color": "#c33"}`.
                    Use `Viewer.set_subsurface` to toggle the effect on
                    existing meshes. Only works with the default "phong"
                    shader.
        shader :    str | pygfx.Material subclass, optional
                    The shader (i.e. material) to use for the mesh.
                    Defaults to "phong". Any mesh material available in
                    the installed pygfx can be selected by name - e.g.
                    "basic", "standard", "physical", "toon", "normal",
                    "normal_lines" or "slice". Alternatively, pass a
                    `pygfx.Material` subclass directly. See
                    `octarine.visuals.available_shaders()` for the full
                    list of options.
        matcap :    str | dict | array, optional
                    If provided, shade the mesh with a matcap instead of
                    with the scene's lights: a picture of a shaded sphere
                    indexed by the surface normal. Pass the name of a
                    preset ("pearl", "clay", "metal", "gold", "jade" or
                    "neon"), a recipe dict, or a matcap image. Use
                    `Viewer.set_matcap` to apply one to existing meshes.
                    A matcap replaces the material, so it cannot be
                    combined with `silhouette`, `subsurface` or `shader`.
        center :    bool, optional
                    If True, re-center camera to all objects on canvas.

        """
        if isinstance(mesh, tm.Scene):
            for _, ob in mesh.geometry.items():
                self.add_mesh(
                    ob,
                    name=name,
                    color=color,
                    alpha=alpha,
                    silhouette=silhouette,
                    subsurface=subsurface,
                    shader=shader,
                    matcap=matcap,
                    center=False,
                )
            return

        if not utils.is_mesh_like(mesh):
            raise TypeError(f"Expected mesh-like object, got {type(mesh)}")
        if color is None:
            color = self._next_color()
        if name is None:
            name = self._next_label("Mesh")
        elif not isinstance(name, str):
            name = str(name)

        if not isinstance(mesh, gfx.Mesh):
            visual = mesh2gfx(
                mesh,
                color=color,
                alpha=alpha,
                silhouette=silhouette,
                subsurface=subsurface,
                shader=shader,
                matcap=matcap,
            )
        else:
            visual = mesh

        visual._object_id = name if name else uuid.uuid4()
        visual._object_group = group

        self._add_to_scene(visual, center)

    def add_points(
        self,
        points,
        name=None,
        group=None,
        color=None,
        marker=None,
        size=2,
        size_space="screen",
        edge_size_space=None,
        min_size=None,
        max_size=None,
        min_edge_width=None,
        edge_width=None,
        edge_color=None,
        edge_mode=None,
        center=True,
    ):
        """Add points plot to canvas.

        Parameters
        ----------
        points :    (N, 3) array
                    Points to plot.
        name :      str, optional
                    Name for the visual.
        group :     str, optional
                    Group for the visual.
        color :     str | tuple, optional
                    Color to use for plotting. Can be the name of
                    a colormap or a single color.
        marker :    str, optional
                    Marker to use for plotting. By default (None), will
                    use a point. Other options include e.g. "circle", "ring"
                    or "diamond". See `pygfx.MarkerShape` for the definitive
                    list of options. Please note that you may have to
                    increase the size of the marker to see some of the shapes.
        size :      int | float
                    Marker size. Can be a single value or an array of
                    sizes for each point.
        size_space : "screen" | "world" | "model", optional
                    Units to use for the marker size. "screen" (default)
                    will keep the line width constant on the screen, while
                    "world" and "model" will keep it constant in world and
                    model coordinates, respectively. In the latter two cases,
                    `size` corresponds to the diameter (not radius) of the
                    marker!
        edge_size_space : "screen" | "world" | "model", optional
                    Units to use for the marker's edge width. By default
                    (None) the edge width uses `size_space`. E.g. combine
                    ``size_space="world"`` with ``edge_size_space="screen"``
                    for world-sized markers with a constant on-screen edge.
        min_size :  float, optional
                    Minimum on-screen marker size in (logical) pixels.
                    Useful with ``size_space="world"`` to keep far-away
                    points visible: "100 world units but at least 10 pixels".
        max_size :  float, optional
                    Maximum on-screen marker size in (logical) pixels.
        min_edge_width : float, optional
                    Minimum on-screen edge width in (logical) pixels. Useful
                    with ``edge_size_space="world"`` to keep the edge visible
                    when zoomed out. Only applies when the edge is enabled
                    (edge_width > 0).
        edge_width : float, optional
                    Width of the marker's edge (in `edge_size_space` units).
                    Defaults to pygfx's default (currently 1).
        edge_color : str | tuple, optional
                    Color of the marker's edge. Defaults to pygfx's default
                    (currently black).
        edge_mode : "centered" | "inner" | "outer", optional
                    How the edge is drawn relative to the marker's outline:
                    straddling it, inside it, or outside it. Defaults to
                    pygfx's default (currently "centered").
        center :    bool, optional
                    If True, re-center camera to all objects on canvas.

        """
        if not isinstance(points, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(points)}")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) array, got {points.shape}")
        if color is None:
            color = self._next_color()
        if name is None:
            name = self._next_label("Scatter")
        elif not isinstance(name, str):
            name = str(name)

        visual = points2gfx(
            points,
            color=color,
            size=size,
            size_space=size_space,
            marker=marker,
            edge_size_space=edge_size_space,
            min_size=min_size,
            max_size=max_size,
            min_edge_width=min_edge_width,
            edge_width=edge_width,
            edge_color=edge_color,
            edge_mode=edge_mode,
        )
        visual._object_id = name if name else uuid.uuid4()
        visual._object_group = group
        self._add_to_scene(visual, center)

    def add_lines(
        self,
        lines,
        name=None,
        group=None,
        color=None,
        linewidth=1,
        linewidth_space="screen",
        linestyle="solid",
        center=True,
    ):
        """Add lines to canvas.

        Parameters
        ----------
        lines :     list of (N, 3) arrays | (N, 3) array
                    Lines to plot. If a list of arrays, each array
                    represents a separate line. If a single array,
                    each row represents a point in the line. You can
                    introduce breaks in the line by inserting NaNs.
        name :      str, optional
                    Name for the visual.
        group :     str, optional
                    Group for the visual.
        color :     str | tuple, optional
                    Color to use for plotting. Can be a single color
                    or one for every point in the line(s).
        linewidth : float | array, optional
                    Line width. Can also be an array with one width for
                    every point in the line(s), in which case the line
                    tapers from point to point. Note that with per-point
                    widths, `linestyle` dashes are still scaled by the
                    mean width.
        linewidth_space : "screen" | "world" | "model", optional
                    Units to use for the line width. "screen" (default)
                    will keep the line width constant on the screen, while
                    "world" and "model" will keep it constant in world and
                    model coordinates, respectively.
        linestyle : "solid" | "dashed" | "dotted" | "dashdot" | tuple, optional
                    Line style to use. If a tuple, must define the on/off
                    sequence.
        center :    bool, optional
                    If True, re-center camera to all objects on canvas.

        """
        # TODO:
        # - allow providing a tuple of (positions, edges) for lines

        if isinstance(lines, np.ndarray):
            if lines.ndim != 2 or lines.shape[1] != 3:
                raise ValueError(f"Expected (N, 3) array, got {lines.shape}")
        elif isinstance(lines, list):
            if not all([l.ndim == 2 and l.shape[1] == 3 for l in lines]):
                raise ValueError("Expected list of (N, 3) arrays.")
        else:
            raise TypeError(f"Expected numpy array or list, got {type(lines)}")

        if color is None:
            color = self._next_color()
        if name is None:
            name = self._next_label("Lines")
        elif not isinstance(name, str):
            name = str(name)

        visual = lines2gfx(
            lines,
            linewidth=linewidth,
            linewidth_space=linewidth_space,
            color=color,
            dash_pattern=linestyle,
        )
        visual._object_id = name if name else uuid.uuid4()
        visual._object_group = group
        self._add_to_scene(visual, center)

    def add_volume(
        self,
        volume,
        spacing=(1, 1, 1),
        name=None,
        group=None,
        color=None,
        opacity=1.0,
        offset=(0, 0, 0),
        clim="data",
        slice=False,
        interpolation="linear",
        hide_zero=True,
        center=True,
    ):
        """Add image volume to canvas.

        Note that the default blend mode for the renderer may cause objects
        behind or inside the volume to look funny. You can change the blend
        mode by setting e.g. `viewer.blend_mode='additive'`.

        Parameters
        ----------
        volume :    (N, M, K) array
                    Volume to plot.
        spacing :   tuple
                    Spacing between voxels.
        name :      str, optional
                    Name for the visual.
        group :     str, optional
                    Group for the visual.
        color :     color | list of colors | pygfx.Texture, optional
                    Colormap to render the volume. This can be:
                      - name of a colormap (e.g. "viridis" or "magma")
                      - a single color (name, hex, rgb, rgba)
                      - a list of colors
                      - a 1D pygfx.Texture
                    Note that single colors typically don't look good and
                    it's better to define at least two colors. For example,
                    instead of "red" use ["red", "yellow"]. If `None` will
                    use one of the built-in pygfx colormaps.
        opacity :   float, optional
                    Overall opacity of the volume. Must be between 0 and 1.
        offset :    tuple, optional
                    (x, y, z) offset for the volume. If None, will use (0, 0, 0).
        clim :      "data" | "datatype" | tuple, optional
                    The contrast limits to scale the data values with.
                      - "data" (default) will use the min/max of the data
                      - "datatype" will use (0, theoretical max of data type)
                        for integer data, e.g. (0, 255) for int8 and uint8,
                        and (0, 1) for float data assuming the data has been
                        normalized
                      - tuple of min/max values or combination of "data" and
                        "datatype" strings
        slice :         bool | tuple, optional
                        Render volume slices instead of the full volume:
                        - True: render slices along all three dimensions
                        - tuple of bools, e.g. `(True, True, False)`: render slices
                          in the respective dimensions
                        - tuple of floats, e.g. `(0.5, 0.5, 0.5)`: render slices
                          at the respective positions (relative to the volume size)
        interpolation : "linear" | "nearest"
                    Interpolation to use when rendering the volume. "linear"
                    (default) looks better but is slower.
        hide_zero : bool
                    If True, will hide voxels with lowest value according to `cmin`.
        center :    bool, optional
                    If True, re-center camera to all objects on canvas.

        """
        if not isinstance(volume, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(volume)}")
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D array, got {volume.ndim}")
        if name is None:
            name = self._next_label("Volume")
        elif not isinstance(name, str):
            name = str(name)

        visuals = volume2gfx(
            volume,
            spacing=spacing,
            offset=offset,
            color=color,
            opacity=opacity,
            clim=clim,
            slice=slice,
            interpolation=interpolation,
            hide_zero=hide_zero,
        )
        name = name if name else uuid.uuid4()
        for vis in visuals:
            vis._object_id = name if name else uuid.uuid4()
            vis._object_group = group
            self._add_to_scene(vis, center)

    def add_sparse_volume(
        self,
        voxels,
        values=None,
        name=None,
        group=None,
        color=None,
        opacity=1.0,
        spacing=(1, 1, 1),
        offset=(0, 0, 0),
        clim=None,
        mode="mip",
        step_size=0.5,
        threshold=0.5,
        density=0.1,
        smoothing=0.0,
        brick_size=16,
        interpolation=None,
        hide_zero=True,
        method="auto",
        center=True,
    ):
        """Add sparse volumetric data to canvas.

        In contrast to `add_volume`, this accepts voxel coordinates (or runs)
        instead of a dense 3D grid. The data is rendered with a custom
        raycasting shader whose memory footprint scales with the number of
        occupied 16^3 bricks rather than with the bounding box - tens of
        millions of voxels are feasible.

        Run-length encoded voxels take a separate, bit-per-voxel path which
        uses roughly 23x less GPU memory but is binary occupancy only.

        Parameters
        ----------
        voxels :    (N, 3) array | (N, 4) array | VoxelCloud | VoxelRuns
                    Either voxel coordinates (xyz; floats are floored to
                    integers) or run-length encoded voxels as
                    (x, y, z, x_run_length) - the layout returned by
                    `dvid.get_sparsevol(..., voxels=False)`.
        values :    (N,) array, optional
                    Per-voxel scalar values to map onto the colormap. If not
                    provided, the volume is rendered as binary occupancy.
                    Not supported for run-length encoded input.
        name :      str, optional
                    Name for the visual.
        group :     str, optional
                    Group for the visual.
        color :     color | list of colors | pygfx.Texture, optional
                    Colormap to render the volume (see `add_volume`).
        opacity :   float
                    Opacity of the volume.
        spacing :   tuple | float
                    (x, y, z) side lengths of a single voxel.
        offset :    tuple
                    (x, y, z) world offset for the volume.
        clim :      (min, max) tuple, optional
                    Range used to scale `values`; defaults to their min/max.
        mode :      "mip" | "density" | "surface"
                    Render as maximum-intensity projection, with
                    front-to-back emission/absorption (cloud-like) or as a
                    shaded isosurface.
        step_size : float
                    Ray-march step (in voxels) inside occupied bricks.
                    Smaller values miss fewer small structures but render
                    slower.
        threshold : float
                    "surface" mode only: the level at which the surface
                    sits, as a fraction of `clim`.
        density :   float
                    "density" mode only: extinction per voxel at the top of
                    `clim`. Higher values render more opaque.
        smoothing : float
                    "surface" mode only: width (in voxels) of an extra
                    filter applied to the field the surface *normal* is
                    taken from. 0 (the default) is off; ~1-2 removes the
                    voxel-scale stipple from the shading. The surface
                    itself is not moved, so no thin structures are lost.
        brick_size : int
                    Edge length (in voxels) of the bricks used to pack the
                    data. Must be a power of two.
        interpolation : "linear" | "nearest", optional
                    Interpolation used when sampling the volume. Defaults
                    to "nearest" for binary occupancy (no `values`) and
                    "linear" when `values` are given or in "surface" mode.
        hide_zero : bool
                    Whether to hide empty space / the lowest value.
        method :    "auto" | "shader" | "bitmask" | "dense"
                    "shader" uses the byte-per-voxel sparse-volume shader,
                    "bitmask" the bit-per-voxel one (binary data only, ~23x
                    smaller on the GPU), "dense" bins the points into a
                    (downsampled) dense grid rendered through the regular
                    volume pipeline. "auto" picks "bitmask" for runs and
                    "shader" for coordinates, falling back to "dense" if the
                    data occupies too many bricks.
        center :    bool, optional
                    If True, re-center camera to all objects on canvas.

        """
        if name is None:
            name = self._next_label("SparseVolume")
        elif not isinstance(name, str):
            name = str(name)

        visuals = utils.make_iterable(
            sparsevolume2gfx(
                voxels,
                values=values,
                color=color,
                opacity=opacity,
                spacing=spacing,
                offset=offset,
                clim=clim,
                mode=mode,
                step_size=step_size,
                threshold=threshold,
                density=density,
                smoothing=smoothing,
                brick_size=brick_size,
                interpolation=interpolation,
                hide_zero=hide_zero,
                method=method,
            )
        )
        for vis in visuals:
            vis._object_id = name if name else uuid.uuid4()
            vis._object_group = group
            self._add_to_scene(vis, center)

    def add_tubes(
        self,
        profile,
        edges=None,
        name=None,
        group=None,
        color=None,
        alpha=None,
        axial_lod=0,
        n_theta=32,
        k=None,
        k_normal=1,
        offset=(0, 0, 0),
        center=True,
    ):
        """Add parametric tubes to canvas.

        Tubes are skeletons with a per-node radial profile

            r(theta) = a0 + sum_k [a_k cos(k*theta) + b_k sin(k*theta)]

        rendered with a custom vertex-pulling shader: the surface is generated
        in the vertex shader straight from the coefficients, so no mesh is
        ever built. `n_theta` and `k` are uniforms, which makes angular level
        of detail a re-draw rather than a re-upload.

        Parameters
        ----------
        profile :   TubeProfile | (M, 8 + 2K) array
                    Either an object with a `to_gpu_buffer()` method and an
                    `edges` attribute (e.g. `sparsecubes.TubeProfile`), or the
                    raw coefficient array in its Cartesian form: position (3),
                    frame quaternion xyzw (4), mean radius a0 (1), then K
                    cosine and K sine coefficients. Positions are expected in
                    physical units.
        edges :     (E, 2) array, optional
                    Index pairs into the nodes. Required if `profile` is a raw
                    coefficient array; otherwise taken from `profile.edges`.
        name :      str, optional
                    Name for the visual.
        group :     str, optional
                    Group for the visual.
        color :     str | tuple | (M, 3) array | (M, 4) array, optional
                    Color for the tubes. An array with one color per node is
                    rendered as per-node colors.
        alpha :     float, optional
                    Opacity value [0-1]; overrides the color's alpha channel.
        axial_lod : int
                    Axial level of detail: keep every 2**axial_lod-th node
                    along each unbranched run. 0 is full resolution, 1 halves,
                    2 quarters, and so on. Branch points and tips are always
                    kept, so no arm can go missing. This is a cost lever
                    rather than a quality one: the intersecting-discs problem
                    it looks like it should fix is not really an axial one at
                    all (see "What has been tried" in
                    `octarine.shaders.tubes`).
        n_theta :   int
                    Number of angular samples around the tube. 32 is smooth,
                    8 still gives a reasonable silhouette at a quarter of the
                    vertices.
        k :         int, optional
                    Number of harmonics to evaluate for the surface position.
                    Defaults to all that are present in the buffer; 0 renders
                    circular tubes of radius a0.
        k_normal :  int
                    Number of harmonics to evaluate for the *normal*, clamped
                    to `k`. Deliberately much lower: dr/dtheta weights
                    harmonic k by k, so the harmonics that still sharpen the
                    silhouette already make the shading look like sandpaper -
                    and dark wherever the normal tilts past the view
                    direction. 0 is the smooth-tube floor.
        offset :    tuple
                    (x, y, z) world offset for the tubes.
        center :    bool, optional
                    If True, re-center camera to all objects on canvas.

        """
        if name is None:
            name = self._next_label("Tubes")
        elif not isinstance(name, str):
            name = str(name)

        visuals = utils.make_iterable(
            tubes2gfx(
                profile,
                color=color,
                alpha=alpha,
                edges=edges,
                axial_lod=axial_lod,
                n_theta=n_theta,
                k=k,
                k_normal=k_normal,
                offset=offset,
            )
        )
        for vis in visuals:
            vis._object_id = name if name else uuid.uuid4()
            vis._object_group = group
            self._add_to_scene(vis, center)

    def close(self):
        """Close the viewer."""
        # Skip if this is headless mode
        if getattr(config, "HEADLESS", False):
            return

        # Clear first to free all visuals
        self.clear()

        # Make sure we don't leave a dangling camera behind in another viewer's
        # controller
        self.unlink()

        # Remove from config if this is the primary viewer
        if self == getattr(config, "PRIMARY_VIEWER", None):
            del config.PRIMARY_VIEWER

        # Close if not already closed
        if not self.canvas.get_closed():
            self.canvas.close()

        if hasattr(self, "_controls"):
            self._controls.close()

        # Close the Jupyter widget
        if hasattr(self, "widget") and not getattr(self.widget, "_is_closed", False):
            self.widget.close(close_viewer=False)

        try:
            viewers.remove(self)
        except ValueError:
            pass

    # N.B. `bounds=False`: `Viewer.bounds` deliberately covers invisible visuals
    # too, so hiding an object can never change the extents of the scene and
    # there is nothing for `_refresh_scene` to re-fit.
    @update_viewer(legend=True, bounds=False)
    def hide_objects(self, obj):
        """Hide given object(s).

        Parameters
        ----------
        obj :   str | list
                Object(s) to hide.

        """
        objects = self.objects  # grab once to speed things up
        for ob in utils.make_iterable(obj):
            if ob not in objects:
                logger.warning(f'Object "{ob}" not found on canvas.')
                continue
            for v in objects[ob]:
                if getattr(v, "_pinned", False):
                    continue
                if v.visible:
                    v.visible = False

    def hide_selected(self):
        """Hide currently selected object(s)."""
        # N.B. no decorator here - `hide_objects` already updates the viewer
        self.hide_objects(self.selected)

    @update_viewer(legend=True, bounds=False)
    def unhide_objects(self, obj=None):
        """Unhide given object(s).

        Parameters
        ----------
        obj :   str | list | None
                Object(s) to unhide. If None, will unhide all objects.

        """
        objects = self.objects  # grab once to speed things up
        if obj is not None:
            ids = utils.make_iterable(obj)
        else:
            ids = list(objects.keys())

        for ob in ids:
            if ob not in objects:
                logger.warning(f"Object {ob} not found on canvas.")
                continue
            for v in objects[ob]:
                if getattr(v, "_pinned", False):
                    continue
                if not v.visible:
                    v.visible = True

    def highlight_objects(self, obj, color=0.3):
        """Highlight given object(s) by increasing their brightness.

        Parameters
        ----------
        obj :   str | int | list | visual
                Object(s) to highlight. Can be the name(s) or ID(s) of
                the object(s), their index(es) in the list of visuals,
                or the visual(s) themselves. Objects already highlighted
                will be silently ignored.
        color : float | tuple
                Color to use for highlighting. If a float, will change
                the HSV value of the current color. If a tuple, will
                use the RGB(A) color.

        See Also
        --------
        Viewer.unhighlight_objects
                Use to remove highlights.

        """
        if not utils.is_iterable(obj):
            objects = [obj]
        else:
            objects = obj

        all_objects = self.objects  # grab once to speed things up

        for ob in objects:
            if ob in all_objects:
                list_ = all_objects[ob]
            elif isinstance(ob, int):
                list_ = list(self.objects.values())[ob]
            elif isinstance(ob, gfx.WorldObject):
                list_ = [ob]
            else:
                raise TypeError(f"Unknown object type: {type(ob)}")

            for o in list_:
                # Skip if object is pinned
                if getattr(o, "_pinned", False):
                    continue
                # Skip if object is already highlighted
                if getattr(o, "_highlighted", False):
                    continue

                if isinstance(color, (float, int)):
                    new_color = _brighten_color(o.material.color, color)
                else:
                    # See if pygfx can handle the color
                    new_color = gfx.Color(color)

                o.material._original_color = o.material.color
                o.material.color = new_color
                o._highlighted = True
                # Remember the style so e.g. set_colors can re-apply it
                o._highlight_style = color

    def unhighlight_objects(self, obj=None):
        """Unhighlight given object(s).

        Parameters
        ----------
        obj :   str | int | list | visual
                Object(s) to unhighlight. Can be the name(s) or ID(s) of
                the object(s), their index(es) in the list of visuals,
                or the visual(s) themselves. If None, will unhighlight all
                objects. Objects that aren't highlighted will be silently
                ignored.

        See Also
        --------
        Viewer.highlight_objects
                Use to highlight objects

        """
        # Important note: it looks like any attribute we added previously
        # will (at some point) have been silently renamed to "_Viewer{attribute}"
        if obj is None:
            obj = [v for v in self.visuals if getattr(v, "_highlighted", False)]

        if not utils.is_iterable(obj):
            objects = [obj]
        else:
            objects = obj

        all_objects = self.objects  # grab once to speed things up

        for ob in objects:
            if ob in all_objects:
                list_ = all_objects[ob]
            elif isinstance(ob, int):
                list_ = list(self.visuals.values())[ob]
            elif isinstance(ob, gfx.WorldObject):
                list_ = [ob]
            else:
                raise TypeError(f"Unknown object type: {type(ob)}")

            for o in list_:
                # Skip if object is pinned
                if getattr(o, "_pinned", False):
                    continue

                # Skip if object isn't actually highlighed
                if not getattr(o, "_highlighted", False):
                    continue
                o.material.color = o.material._original_color
                del o.material._original_color
                del o._highlighted
                if hasattr(o, "_highlight_style"):
                    del o._highlight_style

    def pin_objects(self, obj):
        """Pin given object(s).

        Changes to the color or visibility of pinned neurons are silently
        ignored. You can use this to keep specific neurons visible while
        cycling through the rest - useful for comparisons.

        """
        obj = utils.make_iterable(obj)
        objects = self.objects  # grab only once to speed things up

        for ob in obj:
            if ob not in objects:
                logger.warning(f"Object {ob} not found on canvas.")
                continue
            for v in objects[ob]:
                v._pinned = True

    def unpin_objects(self, obj=None):
        """Unpin given object(s).

        Use ``obj`` to unhide specific neurons.

        """
        objects = self.objects  # grab once to speed things up
        if obj is None:
            obj = objects
        else:
            obj = utils.make_iterable(obj)

        for ob in obj:
            if ob not in objects:
                logger.warning(f"Object {ob} not found on canvas.")
                continue
            for v in objects[ob]:
                v._pinned = False

    @update_viewer(legend=False, bounds=False)
    def set_alpha_mode(self, mode, objects=None):
        """Defines how objects' colors are blended.

        With version v0.13.0 pygfx replaced the single renderer.blend_mode property with
        customizable per-material alpha modes. The Viewer.set_alpha_mode function provides
        a high-level interface to these settings. If you need more fine-grained control,
        see the material.alpha_mode property of individual objects.

        Parameters
        ----------
        mode :      str
                    The mode to set. Please see the pygfx documentation for details:
                      >>> import pygfx
                      >>> help(pygfx.Material.alpha_mode)
        objects :   list, optional
                    Objects to set the alpha mode for. If None, will set for all objects.

        """
        if objects is None:
            objects = list(self.objects)

        for n in objects:
            for v in self.objects[n]:
                if getattr(v, "_pinned", False):
                    continue
                if not hasattr(v, "material"):
                    continue
                v.material.alpha_mode = mode

    @update_viewer(legend=False, bounds=False)
    def set_silhouette(self, silhouette, objects=None):
        """Set Neuroglancer-style silhouette rendering for meshes.

        Fragments are multiplied by `pow(1 - |dot(normal, view_dir)|, silhouette)`:
        face-on regions become transparent while edges/creases are emphasized,
        giving an x-ray-like view of the mesh's outline.

        Parameters
        ----------
        silhouette : float
                    The silhouette exponent: 0 disables the effect, typical
                    values are 1-8 (same semantics as Neuroglancer's
                    "silhouette" property).
        objects :   list, optional
                    Objects to set the silhouette for. If None, will set for
                    all (mesh) objects. Non-mesh objects are silently skipped.

        """
        silhouette = float(silhouette)
        if silhouette < 0:
            raise ValueError(f"silhouette must be >= 0, got {silhouette}")

        # This import registers the shader with pygfx
        from .shaders import SilhouetteMeshMaterial

        if objects is None:
            objects = list(self.objects)
        else:
            objects = utils.make_iterable(objects)

        for n in objects:
            for v in self.objects[n]:
                if getattr(v, "_pinned", False):
                    continue
                if not isinstance(v, gfx.Mesh):
                    continue
                mat = v.material
                if isinstance(mat, SilhouetteMeshMaterial):
                    mat.silhouette = silhouette
                    if silhouette > 0:
                        if not hasattr(mat, "_pre_silhouette_alpha_mode"):
                            mat._pre_silhouette_alpha_mode = mat.alpha_mode
                        mat.alpha_mode = "weighted_blend"
                    elif hasattr(mat, "_pre_silhouette_alpha_mode"):
                        mat.alpha_mode = mat._pre_silhouette_alpha_mode
                        del mat._pre_silhouette_alpha_mode
                elif isinstance(mat, gfx.MeshPhongMaterial):
                    if silhouette == 0:
                        continue
                    # Swap in a silhouette material, carrying over the
                    # relevant properties of the old one
                    props = {
                        p: getattr(mat, p)
                        for p in (
                            "color",
                            "color_mode",
                            "map",
                            "opacity",
                            "pick_write",
                            "side",
                            "flat_shading",
                            "shininess",
                            "specular",
                            "emissive",
                            "alpha_test",
                        )
                        if getattr(mat, p, None) is not None
                    }
                    new_mat = SilhouetteMeshMaterial(silhouette=silhouette, **props)
                    new_mat._pre_silhouette_alpha_mode = mat.alpha_mode
                    new_mat.alpha_mode = "weighted_blend"
                    v.material = new_mat
                elif silhouette > 0:
                    logger.warning(
                        f'Skipped mesh "{n}": silhouette rendering requires a '
                        f"Phong-based material, got {type(mat).__name__}."
                    )

    @update_viewer(legend=False, bounds=False)
    def set_subsurface(self, subsurface=1.0, objects=None, **kwargs):
        """Set subsurface scattering (translucency) for meshes.

        Light is allowed to bleed through the surface instead of stopping
        at it: regions with a light behind them glow, and shading eases
        past the terminator rather than dropping off abruptly. This is what
        gives skin, wax, marble, leaves and thin neurites their translucent
        look.

        Note that the effect uses a *constant* thickness (see below) rather
        than the real local thickness of the mesh, so it cannot on its own
        tell a thin part from a thick one.

        Parameters
        ----------
        subsurface : float
                    Master strength of the effect: 0 disables it, typical
                    values are 0.5-2.
        objects :   list, optional
                    Objects to set the scattering for. If None, will set
                    for all (mesh) objects. Non-mesh objects are silently
                    skipped.
        **kwargs
                    Further properties of
                    `octarine.shaders.SubsurfaceMeshMaterial` to set:
                    `scatter_color`, `thickness`, `distortion`, `falloff`,
                    `wrap` and `glow`. Anything not given is left at its
                    current (or default) value.

        """
        subsurface = float(subsurface)
        if subsurface < 0:
            raise ValueError(f"subsurface must be >= 0, got {subsurface}")

        # This import registers the shader with pygfx
        from .shaders import (
            SUBSURFACE_PROPERTIES,
            SilhouetteMeshMaterial,
            SubsurfaceMeshMaterial,
        )

        if unknown := set(kwargs) - set(SUBSURFACE_PROPERTIES):
            raise ValueError(
                f"Unknown subsurface propert{'y' if len(unknown) == 1 else 'ies'}: "
                f"{', '.join(sorted(unknown))}. "
                f"Valid: {', '.join(sorted(SUBSURFACE_PROPERTIES))}."
            )

        if objects is None:
            objects = list(self.objects)
        else:
            objects = utils.make_iterable(objects)

        for n in objects:
            for v in self.objects[n]:
                if getattr(v, "_pinned", False):
                    continue
                if not isinstance(v, gfx.Mesh):
                    continue
                mat = v.material
                if not isinstance(mat, SubsurfaceMeshMaterial):
                    if subsurface == 0:
                        continue
                    if not isinstance(mat, gfx.MeshPhongMaterial):
                        logger.warning(
                            f'Skipped mesh "{n}": subsurface scattering requires a '
                            f"Phong-based material, got {type(mat).__name__}."
                        )
                        continue
                    # Swap in a subsurface material, carrying over the
                    # relevant properties of the old one. Note that
                    # SubsurfaceMeshMaterial derives from the silhouette
                    # material, so an already-silhouetted mesh keeps its
                    # silhouette (and its pre-silhouette alpha mode).
                    props = {
                        p: getattr(mat, p)
                        for p in (
                            "color",
                            "color_mode",
                            "map",
                            "opacity",
                            "pick_write",
                            "side",
                            "flat_shading",
                            "shininess",
                            "specular",
                            "emissive",
                            "alpha_test",
                            "alpha_mode",
                        )
                        if getattr(mat, p, None) is not None
                    }
                    if isinstance(mat, SilhouetteMeshMaterial):
                        props["silhouette"] = mat.silhouette
                    new_mat = SubsurfaceMeshMaterial(**props)
                    if hasattr(mat, "_pre_silhouette_alpha_mode"):
                        new_mat._pre_silhouette_alpha_mode = (
                            mat._pre_silhouette_alpha_mode
                        )
                    v.material = new_mat
                    mat = new_mat

                mat.subsurface = subsurface
                for prop, value in kwargs.items():
                    setattr(mat, prop, value)

    @update_viewer(legend=False, bounds=False)
    def set_matcap(self, matcap="pearl", objects=None, *, tint=None, **overrides):
        """Shade meshes with a matcap instead of with the scene's lights.

        A matcap ("material capture") is a picture of a shaded sphere used
        as a lookup table: the surface normal - as seen from the camera -
        picks a point on that sphere, and its color becomes the color of the
        pixel. Everything the sphere shows (the falloff, the highlights, the
        rim light) comes along with it, without a single light being
        evaluated.

        This is a staple of scientific and sculpting viewers because surface
        shape reads exceptionally well and the result cannot be under- or
        overlit. The trade-off is that the shading is locked to the camera:
        it turns with the view, and the mesh takes no part in shadows,
        ambient occlusion or anything else the lights drive.

        Octarine generates its matcaps procedurally, so they are recipes
        rather than images - see `octarine.shaders.matcap.MATCAP_PRESETS`:

        | Preset       | Description                                       |
        |--------------|---------------------------------------------------|
        | `pearl`      | Neutral glossy white; the default                 |
        | `clay`       | Matte modelling clay; no highlights, pure form    |
        | `metal`      | Brushed steel; hard highlight and a strong rim    |
        | `gold`       | Warm polished metal, lit by a low sun             |
        | `jade`       | Deep green stone with a translucent glowing rim   |
        | `neon`       | Near-black with magenta/cyan edges; dark scenes   |
        | `sidelit`    | Plain grey under one big side light; readable     |
        | `ceramic`    | Cool glaze with long strip-light reflections      |
        | `slate`      | Muted blue-grey ceramic with a small warm key     |
        | `toon`       | Cel shading: flat tones and an ink outline        |
        | `toon_light` | Pale cel shading, for light backgrounds           |

        The last five reproduce matcaps that ship with Blender's Workbench
        renderer; their parameters were fitted to the originals rather than
        copied from them.

        Parameters
        ----------
        matcap :    str | dict | array | None
                    Name of a preset (see table above), a dict of the
                    properties below, or an image to use as-is: an
                    (N, M, 3) or (N, M, 4) array of floats (linear) or
                    uint8 (sRGB), which is what an off-the-shelf matcap PNG
                    looks like once loaded. Use `None` to go back to the
                    material the meshes had before.
        objects :   list, optional
                    Objects to set the matcap for. If None, will set for
                    all (mesh) objects. Non-mesh objects are silently
                    skipped.
        tint :      float, optional
                    How much of an object's own color tints the matcap,
                    from 0 (the matcap's colors win) to 1 (fully multiplied
                    in). Tinting keeps differently colored objects
                    distinguishable, which is why the neutral presets ask
                    for it and the strongly colored ones do not. Defaults
                    to whatever the preset asks for.
        **overrides
                    Individual properties of the recipe to override:
                    `environment` (the lighting setup the sphere is lit
                    with - the name of one of `Viewer.set_environment`'s, or
                    a rig of its own), `base_color`, `specular`,
                    `shininess`, `rim`, `rim_color`, `rim_power`, and
                    `bands` / `band_softness` / `edge` / `edge_width` for
                    cel shading.

        Examples
        --------
        >>> import octarine as oc
        >>> v = oc.Viewer()
        >>> v.set_matcap("clay")

        Presets are starting points - every property can be overridden:

        >>> v.set_matcap("pearl", base_color="#b0c4de", rim=0.6)
        >>> v.set_matcap("metal", environment="sunset")

        `bands` quantizes the shading into that many flat tones and `edge`
        draws an ink line around the silhouette, which turns any preset
        into a cel-shaded one:

        >>> v.set_matcap("jade", bands=4, band_softness=0, edge=0.8)

        Back to the regular lit materials:

        >>> v.set_matcap(None)

        """
        # These imports register the shader with pygfx
        from .shaders import MATCAP_PRESETS, MatcapMeshMaterial, matcap_texture

        if objects is None:
            objects = list(self.objects)
        else:
            objects = utils.make_iterable(objects)

        if matcap is None:
            restored = []
            for n in objects:
                for v in self.objects[n]:
                    previous = getattr(v, "_pre_matcap_material", None)
                    if previous is not None:
                        v.material = previous
                        del v._pre_matcap_material
                        restored.append(v)
            # A matcap is exempt from the environment; now that it is gone,
            # these meshes have to be lit like the rest again
            self._update_environment(objects=restored)
            return

        tex_map = matcap_texture(matcap, **overrides)
        if tint is None:
            # The recipe's own tint. An image we were handed directly says
            # nothing about tinting, so leave the object's color in place.
            recipe = (
                MATCAP_PRESETS.get(matcap, {}) if isinstance(matcap, str) else matcap
            )
            tint = recipe.get("tint", 1.0) if isinstance(recipe, dict) else 1.0

        for n in objects:
            for v in self.objects[n]:
                if getattr(v, "_pinned", False):
                    continue
                if not isinstance(v, gfx.Mesh):
                    continue
                mat = v.material
                if isinstance(mat, MatcapMeshMaterial):
                    mat.matcap = tex_map
                    mat.tint = tint
                    continue

                # Swap in a matcap material, carrying over the relevant
                # properties of the old one. The old material is kept so
                # that `set_matcap(None)` can put it back - including any
                # silhouette or subsurface settings it may have had.
                props = {
                    p: getattr(mat, p)
                    for p in (
                        "color",
                        "color_mode",
                        "map",
                        "opacity",
                        "pick_write",
                        "side",
                        "flat_shading",
                        "alpha_test",
                        "alpha_mode",
                        "wireframe",
                    )
                    if getattr(mat, p, None) is not None
                }
                new_mat = MatcapMeshMaterial(matcap=tex_map, tint=tint, **props)
                v._pre_matcap_material = mat
                v.material = new_mat

    @update_viewer(legend=False, bounds=False)
    def set_environment(
        self,
        preset="studio",
        *,
        resolution=128,
        rotation=0.0,
        show_background=False,
        pbr=True,
        roughness=0.4,
        metalness=0.0,
        reflectivity=0.35,
        dim_lights=0.5,
        **overrides,
    ):
        """Light the scene with a procedural environment map (IBL).

        A handful of lights leaves surfaces looking flat: every pixel is lit
        from two or three directions and from nowhere else. Real objects are
        lit from *every* direction - sky, ground, the walls of the room -
        which is what gives them their gradients and their reflections.
        Image-based lighting captures that by wrapping the scene in an
        environment map and treating the whole thing as a light source.

        Octarine synthesizes its environments rather than loading HDRI
        photographs, so nothing has to be downloaded: each one is a sky
        gradient plus a few "softboxes" (see
        `octarine.shaders.environment.ENVIRONMENT_PRESETS`):

        | Preset   | Description                                           |
        |----------|-------------------------------------------------------|
        | `studio` | Neutral three-point studio; the all-rounder (default) |
        | `soft`   | Overcast dome; near-shadowless, for figures           |
        | `sky`    | Outdoor daylight; blue zenith, warm sun               |
        | `sunset` | Low warm sun against a violet sky; dramatic           |
        | `neon`   | Near-black room with magenta/cyan rims; dark scenes   |

        Only physically-based (`shader="standard"` or `"physical"`) meshes
        can be lit by an environment in full. By default the plain Phong
        meshes octarine creates are therefore converted to PBR ones (see
        `pbr` below); meshes with a silhouette, subsurface or matcap
        material keep theirs and receive only a reflection on top of their
        normal shading.

        Because an environment lights a surface from all directions at once
        it adds up to a lot of light, and the scene's own lights are dimmed
        to compensate (see `dim_lights`). Pairing this with
        `Viewer.set_tonemapping` is recommended: environments produce values
        well above white, which are otherwise simply clipped.

        Parameters
        ----------
        preset :    str | dict | None
                    Name of a preset (see table above) or a dict of the
                    properties below. Use `None` to switch the environment
                    off again, which also undoes everything below.
        resolution : int
                    Size of one cube map face. 128 is plenty for the
                    lighting itself; raise it if a mirror-like material
                    shows the softboxes as visibly polygonal.
        rotation :  float
                    Rotation of the environment about the vertical axis in
                    degrees; moves the highlights without having to
                    redefine the lights.
        show_background : bool
                    If True, also show the environment as the background,
                    so that reflections and backdrop agree.
        pbr :       bool
                    Whether to convert plain Phong meshes to physically
                    based ones, which is what lets them pick the
                    environment up as full (diffuse + specular) lighting.
                    Their previous materials are restored by
                    `set_environment(None)`.
        roughness : float
                    Roughness of the converted materials, from 0 (a mirror)
                    to 1 (completely matte).
        metalness : float
                    Metalness of the converted materials, from 0 (a
                    dielectric - plastic, stone, tissue) to 1 (bare metal,
                    which takes its color entirely from its reflections).
        reflectivity : float
                    How strong an environment reflection non-PBR materials
                    (Phong, toon, ...) get on top of their normal shading.
        dim_lights : float | bool
                    Factor the scene's own lights are scaled by while the
                    environment is on, so that the two do not add up to a
                    washed-out image. Pass 1 (or False) to leave them
                    alone; the original intensities are restored by
                    `set_environment(None)`.
        **overrides
                    Individual properties of the environment to override:
                    `intensity`, `sky`, `horizon`, `ground`, `gradient` and
                    `lights`.

        Examples
        --------
        >>> import octarine as oc
        >>> v = oc.Viewer()
        >>> v.set_environment("studio")

        For the full effect, show the environment and tone map the result:

        >>> v.set_environment("sunset", show_background=True)
        >>> v.set_tonemapping("aces")

        Presets are starting points - every property can be overridden:

        >>> v.set_environment("studio", rotation=90, intensity=1.5)
        >>> v.set_environment("neon", roughness=0.15, metalness=0.9)

        Back to the plain lights:

        >>> v.set_environment(None)

        """
        if preset is None:
            self._clear_environment()
            return

        # These imports register the shaders with pygfx
        from .shaders import procedural_env_map

        self._env_map = procedural_env_map(
            preset, resolution=resolution, rotation=rotation, **overrides
        )
        self._env_settings = dict(
            pbr=bool(pbr),
            roughness=float(roughness),
            metalness=float(metalness),
            reflectivity=float(reflectivity),
        )

        # Physically-based materials pick this up on their own - including
        # any that are added later
        self.scene.environment = self._env_map

        if show_background:
            self._background.material = gfx.BackgroundSkyboxMaterial(map=self._env_map)
            self._env_background = True
        elif getattr(self, "_env_background", False):
            self.set_bgcolor(self._bgcolor)
            self._env_background = False

        self._dim_lights(dim_lights)
        self._update_environment()

    def _dim_lights(self, factor):
        """Scale the scene's own lights while an environment is lighting it.

        An environment lights a surface from every direction at once, so
        leaving the punctual lights at full strength on top of it washes the
        image out. The original intensities are remembered so that
        `set_environment(None)` can put them back.
        """
        if factor is False:
            factor = 1.0
        factor = float(factor)
        if factor < 0:
            raise ValueError(f"dim_lights must be >= 0, got {factor}")

        if self._pre_env_light_intensities is None:
            self._pre_env_light_intensities = {
                id(light): light.intensity for light in self.lights
            }
        for light in self.lights:
            original = self._pre_env_light_intensities.get(id(light))
            if original is not None:
                light.intensity = original * factor

    def _update_environment(self, objects=None):
        """Apply the current environment to (new) mesh materials.

        Called via `update_helper` whenever objects are added, so that
        meshes added after `set_environment` are lit the same way.
        """
        if self._env_map is None:
            return

        from .shaders import MatcapMeshMaterial

        settings = self._env_settings
        for vis in self.visuals if objects is None else objects:
            if not isinstance(vis, gfx.Mesh) or getattr(vis, "_pinned", False):
                continue
            mat = vis.material
            if isinstance(mat, MatcapMeshMaterial):
                continue  # a matcap deliberately ignores the scene's lighting
            if isinstance(mat, gfx.MeshStandardMaterial):
                # Takes the environment straight off the scene. If it is one
                # we made, keep it in step with the current settings - a
                # material the user brought themselves is left alone.
                created = getattr(vis, "_pre_env_material", (None, None))[0]
                if mat is created:
                    mat.roughness = settings["roughness"]
                    mat.metalness = settings["metalness"]
                continue

            if settings["pbr"] and type(mat) is gfx.MeshPhongMaterial:
                # Only *plain* Phong materials are converted: the silhouette
                # and subsurface materials derive from it, and swapping them
                # out would throw their effect away.
                props = {
                    p: getattr(mat, p)
                    for p in (
                        "color",
                        "color_mode",
                        "map",
                        "opacity",
                        "pick_write",
                        "side",
                        "flat_shading",
                        "emissive",
                        "alpha_test",
                        "alpha_mode",
                        "wireframe",
                    )
                    if getattr(mat, p, None) is not None
                }
                new_mat = gfx.MeshStandardMaterial(
                    roughness=settings["roughness"],
                    metalness=settings["metalness"],
                    **props,
                )
                # Both the material we made and the one it replaced: the
                # first tells `_clear_environment` whether ours is still the
                # one in place, the second is what it puts back
                vis._pre_env_material = (new_mat, mat)
                vis.material = new_mat
                continue

            # Not physically based: pygfx can still give it a plain
            # reflection of the environment on top of its own shading
            if hasattr(mat, "env_map"):
                mat.env_map = self._env_map
                mat.env_combine_mode = "ADD"
                mat.reflectivity = settings["reflectivity"]

    def _clear_environment(self):
        """Undo everything `set_environment` did."""
        env_map, self._env_map = self._env_map, None
        self.scene.environment = None

        for vis in self.visuals:
            if not isinstance(vis, gfx.Mesh):
                continue
            saved = getattr(vis, "_pre_env_material", None)
            if saved is not None:
                created, original = saved
                if vis.material is created:
                    vis.material = original
                elif getattr(vis, "_pre_matcap_material", None) is created:
                    # A matcap was applied on top of the material we made, so
                    # ours is not the one to swap out - but taking the matcap
                    # off later must not put it back either
                    vis._pre_matcap_material = original
                del vis._pre_env_material
            elif getattr(vis.material, "env_map", None) is env_map:
                # Only ours - an env_map the user assigned themselves stays
                vis.material.env_map = None

        if self._pre_env_light_intensities is not None:
            for light in self.lights:
                original = self._pre_env_light_intensities.get(id(light))
                if original is not None:
                    light.intensity = original
            self._pre_env_light_intensities = None

        if getattr(self, "_env_background", False):
            self.set_bgcolor(self._bgcolor)
            self._env_background = False

    @property
    def environment(self):
        """The environment map lighting the scene, if any (read-only).

        Set it with `Viewer.set_environment`.
        """
        return self._env_map

    @update_viewer(legend=False, bounds=False)
    def set_depth_of_field(
        self,
        enabled=True,
        *,
        focus=None,
        aperture=100.0,
        max_radius=16.0,
        smooth=False,
        snap_radius=0,
    ):
        """Set a depth-of-field (focal blur) effect for the viewer.

        Objects near a focal plane are rendered sharp while everything
        closer or farther is progressively blurred, similar to a
        photographic lens.

        Note that this is a screen-space post-processing effect: it applies
        to the entire rendered image (including overlay elements such as
        messages), and objects that do not write depth (e.g. meshes with a
        transparent alpha mode) are blurred by whatever is behind them.

        Parameters
        ----------
        enabled :   bool
                    Use `viewer.set_depth_of_field(False)` to turn the
                    effect off again.
        focus :     float, optional
                    Distance of the focal plane from the camera in world
                    units (note that for orthographic cameras this can be
                    negative because pygfx places the camera in the middle
                    of the scene). If None (default), continuously
                    auto-focuses on whatever is at the center of the view
                    (if that is empty space, the image is left sharp).
        aperture :  float
                    Blur strength: the blur radius in physical pixels of a
                    point at 100% relative defocus - relative to the focus
                    distance for perspective cameras, and to the visible
                    height of the view for orthographic ones. Typical
                    values are 50-300.
        max_radius : float
                    Upper limit for the blur radius in physical pixels.
        smooth :    bool | float
                    Only relevant for autofocus (`focus=None`): if truthy,
                    changes in focus are eased over approximately this many
                    seconds (True = 0.2s) instead of snapping instantly.
                    While the center of the view is over empty space, the
                    last focus is held.
        snap_radius : float
                    Only relevant for autofocus (`focus=None`): search
                    radius in physical pixels around the view center. The
                    autofocus targets the object closest to the view center
                    within that radius, instead of only what is exactly
                    under the center pixel. 0 (default) disables snapping.

        """
        if not enabled:
            if getattr(self, "_dof_pass", None) is not None:
                self._dof_pass.enabled = False
                self.remove_animation(self._dof_smooth_tick)
            return

        from .shaders import DepthOfFieldPass

        if getattr(self, "_dof_pass", None) is None:
            self._dof_pass = DepthOfFieldPass(
                self.camera,
                focus=focus,
                aperture=aperture,
                max_radius=max_radius,
                smooth=smooth,
                snap_radius=snap_radius,
            )
            # A lens effect: after the shading passes (occlusion, outlines),
            # before the tone map
            self._add_effect_pass(self._dof_pass)
        else:
            self._dof_pass.focus = focus
            self._dof_pass.aperture = aperture
            self._dof_pass.max_radius = max_radius
            self._dof_pass.smooth = smooth
            self._dof_pass.snap_radius = snap_radius
        self._dof_pass.enabled = True
        # This keeps re-rendering (in "reactive" mode) while a smooth
        # re-focus transition is still settling
        self.add_animation(self._dof_smooth_tick, on_error="log", req_render=False)

    def _dof_smooth_tick(self):
        """Animation hook: re-render while a smooth re-focus is settling."""
        dof_pass = getattr(self, "_dof_pass", None)
        if dof_pass is not None and dof_pass.enabled and not dof_pass._smooth_settled:
            self._render_stale = True

    def _default_ao_radius(self, fraction=0.04, bounds=None):
        """A sensible ambient occlusion radius for the current scene."""
        if bounds is None:
            bounds = self.bounds
        if bounds is None:  # nothing on the canvas (yet)
            return 1.0
        diagonal = float(np.linalg.norm(bounds[:, 1] - bounds[:, 0]))
        return (diagonal * fraction) or 1.0

    def _update_ao_radius(self, bounds=None):
        """Re-derive the ambient occlusion radius from the scene.

        Unlike the other occlusion parameters the radius is in world units and
        hence has to match the scene - which we only know once there is
        something on the canvas. This is called via `update_helper` whenever
        objects are added or removed, unless the user has pinned a radius by
        passing one explicitly (see `Viewer.set_ambient_occlusion`). `bounds`
        is the scene's extents if the caller has them at hand already.

        """
        if self._ao_pass is None or not self._ao_auto_radius:
            return

        radius = self._default_ao_radius(bounds=bounds)
        if radius == self._ao_pass.radius:
            return

        self._ao_pass.radius = radius
        self._render_stale = True

        # Keep the GUI's radius slider in step with it
        if self.controls is not None:
            self.controls.sync_ao_radius()

    @update_viewer(legend=False, bounds=False)
    def set_ambient_occlusion(
        self,
        enabled=True,
        *,
        radius=None,
        intensity=1.0,
        bias=0.01,
        samples=16,
        power=1.0,
        blur=True,
        debug=False,
    ):
        """Set a screen-space ambient occlusion (SSAO) effect for the viewer.

        Ambient light is otherwise applied uniformly, which leaves creases,
        cavities and the points where objects touch looking flat. This
        estimates how much of the surrounding hemisphere is blocked at each
        pixel and darkens the image accordingly.

        Note that this is a screen-space post-processing effect: it applies
        to the entire rendered image (including overlay elements such as
        messages), and objects that do not write depth (e.g. meshes with a
        transparent alpha mode) neither cast nor receive occlusion.

        Parameters
        ----------
        enabled :   bool
                    Use `viewer.set_ambient_occlusion(False)` to turn the
                    effect off again.
        radius :    float, optional
                    How far to look for occluders, in world units. This is
                    the one parameter that has to match the scene: too
                    small and the effect disappears, too large and it turns
                    into a dark haze. If None (default), 4% of the diagonal
                    of the scene bounds is used and kept up-to-date as
                    objects are added or removed; passing a value pins the
                    radius to it.
        intensity : float
                    Strength of the darkening, from 0 (no effect) to 1
                    (fully occluded pixels turn black).
        bias :      float
                    Occluders closer to the surface than this - as a
                    fraction of `radius` - are ignored. Raise it if flat
                    surfaces show occlusion of their own, lower it (down to
                    0) for more contrast in tight creases.
        samples :   int
                    Number of hemisphere samples per pixel. More samples
                    mean less noise at a higher rendering cost.
        power :     float
                    Exponent applied to the occlusion; values > 1 restrict
                    the effect to the darkest areas, values < 1 spread it
                    out.
        blur :      bool | int
                    Radius (in pixels) of the bilateral blur that removes
                    the sampling noise. True (default) uses 2, which is
                    exactly one tile of the sampling pattern; False (or 0)
                    disables it.
        debug :     bool
                    If True, render the occlusion itself as greyscale
                    instead of darkening the scene. Useful for finding a
                    `radius` that suits the scene.

        """
        if not enabled:
            if getattr(self, "_ao_pass", None) is not None:
                self._ao_pass.enabled = False
            return

        from .shaders import AmbientOcclusionPass

        # Without an explicit radius we keep deriving it from the scene
        # (see `Viewer._update_ao_radius`)
        self._ao_auto_radius = radius is None
        if radius is None:
            radius = self._default_ao_radius()

        if getattr(self, "_ao_pass", None) is None:
            self._ao_pass = AmbientOcclusionPass(
                self.camera,
                radius=radius,
                intensity=intensity,
                bias=bias,
                samples=samples,
                power=power,
                blur=blur,
                debug=debug,
            )
            # Occlusion is part of the shading, so it has to run before the
            # anti-aliasing and any lens effects (e.g. depth of field)
            self._add_effect_pass(self._ao_pass, EFFECT_STAGES["ao"])
        else:
            self._ao_pass.radius = radius
            self._ao_pass.intensity = intensity
            self._ao_pass.bias = bias
            self._ao_pass.samples = samples
            self._ao_pass.power = power
            self._ao_pass.blur = blur
            self._ao_pass.debug = debug
        self._ao_pass.enabled = True

    @update_viewer(legend=False, bounds=False)
    def set_outline(
        self,
        enabled=True,
        *,
        color="#000",
        thickness=1.0,
        depth_threshold=0.02,
        normal_threshold=0.3,
        debug=False,
    ):
        """Draw outlines around silhouettes and along creases.

        This gives the scene the look of a technical illustration, and does
        real work in a crowded one: objects of similar color that overlap
        become individually readable, because each of them is bounded by a
        line.

        Note that this is a screen-space post-processing effect: it applies
        to the entire rendered image (including overlay elements such as
        messages), and objects that do not write depth (e.g. meshes with a
        transparent alpha mode) are neither outlined nor occlude an outline.

        Parameters
        ----------
        enabled :   bool
                    Use `viewer.set_outline(False)` to turn the effect off
                    again.
        color :     str | tuple
                    Color of the outline. Its alpha channel doubles as the
                    strength of the effect, so e.g. "#0004" gives a subtle
                    line rather than a hard one.
        thickness : float
                    Width of the outline in physical pixels. Values above
                    about 4 start to look chunky rather than drawn.
        depth_threshold : float
                    How far a neighbouring pixel has to lie off the surface
                    under the current one to count as a separate object,
                    relative to its distance from the camera. Lower it to
                    outline shallower steps, raise it if surfaces are
                    outlined across their interior.
        normal_threshold : float
                    How sharply the surface has to fold to count as a
                    crease, as `1 - cos(angle)`: 0.3 is roughly 45 degrees.
                    0 switches creases off and outlines silhouettes only.
        debug :     bool
                    If True, render the detected edges as white on black
                    instead of drawing them over the scene. Useful for
                    tuning the two thresholds.

        Examples
        --------
        >>> import octarine as oc
        >>> v = oc.Viewer()
        >>> v.set_outline()

        A thicker, softer line - and silhouettes only:

        >>> v.set_outline(color="#0008", thickness=2, normal_threshold=0)

        """
        if not enabled:
            if getattr(self, "_outline_pass", None) is not None:
                self._outline_pass.enabled = False
            return

        from .shaders import OutlinePass

        if getattr(self, "_outline_pass", None) is None:
            self._outline_pass = OutlinePass(
                self.camera,
                color=color,
                thickness=thickness,
                depth_threshold=depth_threshold,
                normal_threshold=normal_threshold,
                debug=debug,
            )
            # Outlines are part of the shading: they belong under a lens
            # effect (and have to be blurred by it), not on top of it
            self._add_effect_pass(self._outline_pass, EFFECT_STAGES["outline"])
        else:
            self._outline_pass.color = color
            self._outline_pass.thickness = thickness
            self._outline_pass.depth_threshold = depth_threshold
            self._outline_pass.normal_threshold = normal_threshold
            self._outline_pass.debug = debug
        self._outline_pass.enabled = True

    @update_viewer(legend=False, bounds=False)
    def set_tonemapping(self, mode="aces", *, exposure=1.0, white_point=4.0):
        """Set tone mapping (and exposure) for the viewer.

        The scene is rendered into a floating point buffer, so colors are
        not limited to [0, 1]: highlights, emissive surfaces and anything
        lit by an environment map (see `Viewer.set_environment`) routinely
        go well above white. Without tone mapping those values are simply
        clipped, which turns bright regions into flat white blobs and skews
        their color - a warm highlight reads as pure red once the red
        channel clips and the others have not.

        Tone mapping maps that open-ended range onto what the display can
        show, rolling the highlights off gradually instead. `exposure`
        scales the image before the curve is applied, i.e. it is the
        photographic exposure control.

        The pass runs last, after effects such as bloom (which want the
        untouched high dynamic range values) and before pygfx's own
        anti-aliasing and gamma handling.

        Parameters
        ----------
        mode :      str | None
                    The tone mapping curve:
                     - "aces" (default): a fit to the ACES filmic response.
                       Contrasty and saturated; the usual choice.
                     - "filmic": Hable's "Uncharted 2" curve. Like ACES but
                       holds on to more shadow detail.
                     - "reinhard": the gentlest option. Stays closest to the
                       original colors, at the cost of looking flatter.
                     - "none": clip only, i.e. exposure control on its own.
                    Use `None` to remove the tone mapping altogether.
        exposure :  float
                    Scales the image before the curve is applied: 2 is one
                    stop brighter, 0.5 one stop darker. See also the
                    `Viewer.exposure` property, which sets this on its own.
        white_point : float
                    The input value that maps to white. Only used by
                    "reinhard" and "filmic"; raising it holds on to more
                    highlight detail (and darkens the image overall).

        Examples
        --------
        >>> import octarine as oc
        >>> v = oc.Viewer()
        >>> v.set_environment("studio")   # gives it something to roll off
        >>> v.set_tonemapping("aces")
        >>> v.exposure = 1.5              # brighten by ~0.6 stops

        Back to plain clipping:

        >>> v.set_tonemapping(None)

        """
        if mode is None or mode is False:
            if getattr(self, "_tonemap_pass", None) is not None:
                self.renderer.effect_passes = tuple(
                    e
                    for e in self.renderer.effect_passes
                    if e is not self._tonemap_pass
                )
                self._tonemap_pass = None
            return

        from .shaders import ToneMappingPass

        if getattr(self, "_tonemap_pass", None) is None:
            self._tonemap_pass = ToneMappingPass(
                mode=mode, exposure=exposure, white_point=white_point
            )
            # Has to see the finished image, so it goes last
            self._add_effect_pass(self._tonemap_pass, EFFECT_STAGES["tonemap"])
        else:
            self._tonemap_pass.mode = mode
            self._tonemap_pass.exposure = exposure
            self._tonemap_pass.white_point = white_point
        self._tonemap_pass.enabled = True

    @property
    def exposure(self):
        """Exposure of the rendered image; 1 leaves it unchanged.

        Setting this switches tone mapping on if it is not already (see
        `Viewer.set_tonemapping`) - without a curve to roll the highlights
        off, raising the exposure would only clip them.

        """
        pass_ = getattr(self, "_tonemap_pass", None)
        return 1.0 if pass_ is None else pass_.exposure

    @exposure.setter
    def exposure(self, value):
        if getattr(self, "_tonemap_pass", None) is None:
            self.set_tonemapping(exposure=value)
        else:
            self._tonemap_pass.exposure = value
            self._render_stale = True

    @update_viewer(legend=True, bounds=False)
    def set_colors(self, c, alpha_mode="auto"):
        """Set object color.

        Parameters
        ----------
        c :     tuple | dict
                RGB color(s) to apply. Values must be 0-1. Accepted:
                   1. Tuple of single color. Applied to all visible objects.
                   2. Dictionary names/IDs to colors.
        alpha_mode : str
                If "auto" (default), will set the alpha mode to "add" if the
                opacity is < 1, and "opaque" otherwise. Set `alpha_mode` to `None` to
                skip this adjustment.

        """
        objects = self.objects  # grab once to speed things up
        if isinstance(c, (tuple, list, np.ndarray, str)):
            cmap = {s: c for s in objects}
        elif isinstance(c, dict):
            cmap = c
        else:
            raise TypeError(f'Unable to use colors of type "{type(c)}"')

        for n in objects:
            if n in cmap:
                for v in objects[n]:
                    if getattr(v, "_pinned", False):
                        continue
                    if not hasattr(v, "material"):
                        continue
                    # Note: there is currently a bug where removing or adding an alpha
                    # channel from a color will break the rendering pipeline
                    if len(v.material.color) == 4:
                        new_c = gfx.Color(cmap[n]).rgba
                    else:
                        new_c = gfx.Color(cmap[n]).rgb

                    if n in self._selected and hasattr(v, "_stored_color"):
                        # Selected objects wear the selection highlight;
                        # update the color they revert to on deselection
                        # instead of overwriting the highlight.
                        v._stored_color = gfx.Color(new_c)
                    elif getattr(v, "_highlighted", False):
                        # Hover-highlighted objects wear a brightened color;
                        # update the underlying color and re-apply the
                        # highlight so it survives un-highlighting.
                        v.material._original_color = gfx.Color(new_c)
                        style = getattr(v, "_highlight_style", 0.3)
                        if isinstance(style, (float, int)):
                            v.material.color = _brighten_color(new_c, style)
                        else:
                            v.material.color = gfx.Color(style)
                    else:
                        v.material.color = gfx.Color(new_c)

                    # Determine if we consider this transparent
                    if len(new_c) == 4 and new_c[3] < 1:
                        is_transparent = True
                    elif v.material.opacity < 1:
                        is_transparent = True
                    else:
                        is_transparent = False

                    if alpha_mode == "auto":
                        if is_transparent:
                            v.material.alpha_mode = "add"
                        else:
                            v.material.alpha_mode = "solid"
                    elif alpha_mode:
                        v.material.alpha_mode = alpha_mode

    def colorize(self, palette="seaborn:tab10", objects=None, randomize=True):
        """Colorize objects using a color palette.

        Parameters
        ----------
        palette :   str | cmap Colormap
                    Name of the `cmap` palette to use. See
                    https://cmap-docs.readthedocs.io/en/latest/catalog/#colormaps-by-category
                    for available options.
        objects :   list, optional
                    Objects to colorize. If None, will colorize all objects.
        randomize : bool
                    If True (default), will randomly shuffle the colors.

        """
        if objects is None:
            objects = self.objects  # grab once to speed things up

        if not isinstance(palette, cmap._colormap.Colormap):
            palette = cmap.Colormap(palette)

        if randomize:
            # Note: can't use numpy here because it claims array is not 1d
            colors = random.choices(list(palette.iter_colors()), k=len(objects))
        else:
            colors = list(palette.iter_colors(len(objects)))

        colormap = {s: tuple(colors[i].rgba) for i, s in enumerate(objects)}

        self.set_colors(colormap)

    def set_bgcolor(self, c, *more):
        """Set background color.

        Parameters
        ----------
        c :     tuple | str | list
                RGB(A) color to use for the background. Pass two or four
                colors - either as separate arguments or as a single list -
                for a linear gradient: two colors run bottom to top, four
                colors set the bottom left, bottom right, top left and top
                right corner, respectively.

        See Also
        --------
        [`octarine.Viewer.set_bg_gradient`][]
                    Radial ("studio") gradient backgrounds, incl. presets.

        Examples
        --------
        >>> import octarine as oc
        >>> v = oc.Viewer()
        >>> v.set_bgcolor("white")
        >>> v.set_bgcolor("black", "#1B2838")  # vertical gradient

        """
        colors = utils.as_color_list(c, *more)
        if len(colors) not in (1, 2, 4):
            raise ValueError(f"Need 1, 2 or 4 colors, got {len(colors)}.")

        # Remember for when a gradient background is switched off again
        self._bgcolor = colors

        # If a gradient background is currently in place we have to swap the
        # material rather than just re-color it
        if isinstance(self._background.material, gfx.BackgroundMaterial):
            self._background.material.set_colors(*colors)
        else:
            self._background.material = gfx.BackgroundMaterial(*colors)

    def set_bg_gradient(
        self,
        preset="graphite",
        *,
        colors=None,
        center=None,
        radius=None,
        falloff=None,
        vignette=None,
    ):
        """Set a radial ("studio") gradient as background.

        This is the kind of backdrop product or hero renders are typically
        shot against: a soft pool of light behind the object that fades into
        near-black towards the edges of the frame. The gradient is fixed to
        the canvas, i.e. it does not move with the camera.

        Available presets:

        | Preset      | Description                                            |
        |-------------|--------------------------------------------------------|
        | `graphite`  | Neutral studio grey; the all-rounder (default)         |
        | `cinematic` | Desaturated blue-black; dark metals, tech, sci-fi      |
        | `warm`      | Warm charcoal; flatters brass, bronze, wood, leather   |
        | `olive`     | Muted olive; organic and natural materials             |
        | `burgundy`  | Dusty burgundy; editorial/photographic                 |
        | `halo`      | Near-black halo; dramatic, minimal                     |

        Parameters
        ----------
        preset :    str | dict | None
                    Name of a preset (see table above) or a dict of the
                    parameters below. Use `None` to switch the gradient off
                    again and go back to a plain background.
        colors :    tuple, optional
                    Three colors `(inner, mid, outer)` - the center of the
                    glow, the lift half-way out, and the color the gradient
                    settles into. Two colors `(inner, outer)` also work, in
                    which case the mid stop is interpolated.
        center :    (x, y) tuple, optional
                    Center of the gradient in relative image coordinates:
                    `(0, 0)` is the top left, `(1, 1)` the bottom right
                    corner.
        radius :    float, optional
                    Distance at which the gradient reaches its outer color,
                    as a fraction of the canvas width.
        falloff :   float, optional
                    Shape of the ramp: values > 1 keep the core bright and
                    push the transition towards the rim (3 confines it to
                    roughly the outer 30% of the radius), 1 is linear, and
                    values < 1 drop off right at the center.
        vignette :  float, optional
                    Strength (0-1) of the additional darkening towards the
                    corners of the frame. 0 disables it.

        Examples
        --------
        >>> import octarine as oc
        >>> v = oc.Viewer()
        >>> v.set_bg_gradient("cinematic")

        Presets are just starting points - every parameter can be overridden:

        >>> v.set_bg_gradient("cinematic", radius=0.5, vignette=0.4)
        >>> v.set_bg_gradient(colors=("#3A292C", "#070405"), falloff=2)

        Back to a plain background:

        >>> v.set_bg_gradient(None)

        """
        if preset is None:
            self.set_bgcolor(self._bgcolor)
            return

        # This import registers the shader with pygfx
        from .shaders import GradientBackgroundMaterial

        self._background.material = GradientBackgroundMaterial.from_preset(
            preset,
            colors=colors,
            center=center,
            radius=radius,
            falloff=falloff,
            vignette=vignette,
        )

    def _toggle_fps(self):
        """Switch FPS measurement on and off."""
        self.show_fps = not self.show_fps

    def screenshot(
        self,
        filename="screenshot.png",
        size=None,
        pixel_ratio=None,
        alpha=True,
        supersample=2,
    ):
        """Save a screenshot of the canvas.

        Parameters
        ----------
        filename :      str | pathlib.Path, optional
                        Filename to save to. If ``None``, will return image array.
                        Note that this will always save a PNG file, no matter
                        the extension.
        size :          tuple, optional
                        Size of the screenshot. If provided, will temporarily
                        change the canvas size.
        pixel_ratio :   int, optional
                        Factor by which to scale canvas. Determines image
                        dimensions: the image comes out at `size` (or the
                        current canvas size) times this factor. Defaults to
                        the renderer's current pixel ratio.
        alpha :         bool, optional
                        If True, will export transparent background.
        supersample :   int, optional
                        Render the frame at this factor above the output
                        resolution and filter it back down - i.e. supersampling
                        anti-aliasing, the one knob that actually resolves
                        sub-pixel detail rather than smoothing over it. The
                        image dimensions are unaffected. 2 (the default) takes
                        care of most of what the renderer's own anti-aliasing
                        leaves behind, 4 is as good as it realistically gets;
                        1 switches it off. Memory and render time grow with the
                        square of the factor, and it is capped to whatever
                        still fits the GPU's maximum texture size.

                        The filter used to resample the frame is the renderer's
                        `pixel_filter` - 'mitchell' by default, which is sharp
                        but rings slightly at high-contrast edges; 'tent' or
                        'bspline' trade sharpness for no ringing at all.

        Examples
        --------
        A high quality 4k screenshot, no matter the size of the window:

        >>> v.screenshot("figure.png", size=(3840, 2160), pixel_ratio=1,
        ...              supersample=4)

        """
        im = self._screenshot(
            alpha=alpha, size=size, pixel_ratio=pixel_ratio, supersample=supersample
        )
        if filename:
            filename = Path(filename)
            if filename.suffix != ".png":
                filename = filename.parent / f"{filename.name}.png"
            utils.write_png(im, filename.resolve())
        else:
            return im

    def _screenshot(self, alpha=True, size=None, pixel_ratio=None, supersample=1):
        """Return image array for screenshot."""
        supersample = int(supersample)
        if supersample < 1:
            raise ValueError(f"supersample must be >= 1, got {supersample}")

        if alpha:
            vis = self._background.visible
            self._background.visible = False
        if size:
            os = self.size
            self.size = size

        # The image comes out at logical size x pixel ratio. Supersampling
        # renders it larger than that and filters it back down, so the ratio we
        # render at and the one that defines the output size are not the same.
        out_ratio = pixel_ratio if pixel_ratio else self.renderer.pixel_ratio
        supersample = self._clamp_supersample(supersample, self._output_size(out_ratio))

        # Both of these are plain attributes (the properties only ever set one
        # of them), so this restores "auto" pixel ratio as well as a fixed one
        opr = (self.renderer._pixel_scale, self.renderer._pixel_ratio)
        scaled = []

        # In the non-continuous trigger modes `_animate` skips the render
        # itself - nothing has flagged the scene as stale, or the window is not
        # active (which is exactly the case when the screenshot is taken from
        # the controls panel). The forced draw below would then simply hand us
        # back the frame that is already there: at the old size and pixel ratio
        # and with the background we just hid still in it. So take the trigger
        # out of the loop for the duration of the capture. N.B. we bypass the
        # property, which would add/remove event handlers.
        trigger = self._render_trigger
        self._render_trigger = "continuous"

        try:
            self.renderer.pixel_ratio = out_ratio * supersample
            if supersample > 1:
                scaled = self._scale_pixel_effects(supersample)

            # Make sure a frame with the (potentially) updated size, pixel ratio
            # and effect parameters is drawn before we read the image back.
            # Note: this has to happen _after_ adjusting those!
            self.canvas.force_draw()

            if supersample == 1:
                im = self.renderer.snapshot()
            else:
                # Note we ask the canvas for its size again: a resize (see
                # `size` above) may only have gone through with the draw
                im = self._downsampled_snapshot(self._output_size(out_ratio))
        finally:
            for effect_pass, param, value in scaled:
                setattr(effect_pass, param, value)
            self.renderer._pixel_scale, self.renderer._pixel_ratio = opr
            if alpha:
                self._background.visible = vis
            if size:
                self.size = os

            # What the canvas is showing now is the frame we just captured,
            # hidden background and all. Flag it as stale so that the next tick
            # draws a normal one over it - and where there will be no such tick
            # (an inactive window never redraws), draw it here and now, i.e.
            # while the trigger is still overridden.
            self._render_stale = True
            if trigger == "active_window" and not self._window_is_active():
                self.canvas.force_draw()
            self._render_trigger = trigger

        return im

    def _output_size(self, pixel_ratio):
        """Size (in pixels) of a screenshot taken at the given pixel ratio."""
        w, h = self.renderer.logical_size
        return max(1, round(w * pixel_ratio)), max(1, round(h * pixel_ratio))

    def _clamp_supersample(self, supersample, out_size):
        """Reduce the supersample factor to what the GPU can still allocate."""
        max_size = self.renderer.device.limits["max-texture-dimension-2d"]
        max_supersample = max(1, int(max_size // max(out_size)))
        if supersample > max_supersample:
            logger.warning(
                f"Supersampling a {out_size[0]}x{out_size[1]} screenshot {supersample}x "
                f"exceeds this GPU's maximum texture size ({max_size} px). "
                f"Falling back to {max_supersample}x."
            )
            supersample = max_supersample
        return supersample

    def _scale_pixel_effects(self, factor):
        """Scale the effect parameters that are given in physical pixels.

        See `PIXEL_SCALED_EFFECT_PARAMS` for the why. Returns a list of
        `(pass, parameter, old value)` for the caller to restore.

        """
        scaled = []
        for effect_pass in self.renderer.effect_passes:
            for klass in type(effect_pass).__mro__:
                params = PIXEL_SCALED_EFFECT_PARAMS.get(klass.__name__)
                if params is None:
                    continue
                for param, limit in params.items():
                    value = getattr(effect_pass, param, None)
                    if value is None:
                        continue
                    new_value = value * factor
                    if limit is not None:
                        new_value = min(new_value, limit)
                    setattr(effect_pass, param, type(value)(new_value))
                    scaled.append((effect_pass, param, value))
                break  # a pass is only ever listed once
        return scaled

    def _downsampled_snapshot(self, size):
        """Filter the frame that was last drawn down to `size` pixels.

        `renderer.snapshot()` reads the renderer's internal texture as it is,
        so with supersampling it would simply hand us a larger image. Flushing
        into a texture of the intended size instead takes the same route as
        rendering to a screen with a pixel ratio > 1: the reconstruction filter
        selected by `renderer.pixel_filter` ('mitchell' by default) does the
        downsampling on the GPU, and in linear light rather than on the sRGB
        encoded values.

        """
        w, h = size
        texture = gfx.Texture(
            size=(w, h, 1),
            dim=2,
            # Same format as the renderer's internal texture, so that the flush
            # is a filter and nothing else. `colorspace` only matters when a
            # texture is *sampled* - this one is a render target - but saying
            # "srgb" twice makes pygfx complain.
            format="rgba8unorm-srgb",
            colorspace="physical",
            usage=(
                wgpu.TextureUsage.RENDER_ATTACHMENT
                | wgpu.TextureUsage.TEXTURE_BINDING
                | wgpu.TextureUsage.COPY_SRC
            ),
        )

        # `flush` gamma-corrects for canvases whose format is not sRGB. Ours is,
        # so that correction has to sit out this one flush.
        gamma_correction_srgb = self.renderer._gamma_correction_srgb
        self.renderer._gamma_correction_srgb = 1.0
        try:
            self.renderer.flush(target=texture)
        finally:
            self.renderer._gamma_correction_srgb = gamma_correction_srgb

        data = self.renderer.device.queue.read_texture(
            {
                "texture": ensure_wgpu_object(texture),
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            {"offset": 0, "bytes_per_row": 4 * w, "rows_per_image": h},
            (w, h, 1),
        )
        return np.frombuffer(data, np.uint8).reshape(h, w, 4)

    def set_view(self, view):
        """(Re-)set camera position.

        Parameters
        ----------
        view :      XY | XZ | YZ | dict
                    View to set. Can be inverted to e.g. "-XY" to show view from back.
                    If a dictionary, should describe the state of the camera. Typically,
                    this is obtained by calling `viewer.get_view()`.

        """
        if isinstance(view, dict):
            self.camera.set_state(view)
        elif isinstance(view, str) and view in NAMED_VIEWS:
            view_dir, up = NAMED_VIEWS[view]
            self.camera.show_object(self.scene, view_dir=view_dir, up=up)
        else:
            raise TypeError(f"Unable to set view from {view!r}")

        self._sync_linked()

    def get_view(self, view=None):
        """Get camera state.

        Parameters
        ----------
        view :      XY | XZ | YZ, optional
                    If given, return the camera state that `set_view(view)`
                    would produce instead of the current one - without actually
                    moving the camera.

        Returns
        -------
        dict
                    Camera state, as accepted by
                    [`Viewer.set_view`][octarine.Viewer.set_view].

        """
        if view is None:
            return self.camera.get_state()

        if not isinstance(view, str) or view not in NAMED_VIEWS:
            raise TypeError(f"Unable to make a view from {view!r}")

        # Let the camera work out what this view means for the current scene,
        # then put it back where it was. N.B. this deliberately does not go
        # through `set_view`: merely asking what a view looks like must not
        # push the intermediate state to linked viewers.
        before = self.camera.get_state()
        try:
            view_dir, up = NAMED_VIEWS[view]
            self.camera.show_object(self.scene, view_dir=view_dir, up=up)
            return self.camera.get_state()
        finally:
            self.camera.set_state(before)

    def bind_key(self, key, func, modifiers=None):
        """Bind a function to a key press.

        Note that any existing keybindings for `key` + `modifiers` will be
        silently overwritten.

        Parameters
        ----------
        key :       str
                    Key to bind to. Can be any key on the keyboard.
        func :      callable
                    Function to call when key is pressed.
        modifiers : str | list thereof, optional
                    Modifier(s) to use with the key. Can be "Shift", "Control",
                    "Alt" or "Meta".

        """
        if not callable(func):
            raise TypeError("`func` needs to be callable")

        if not isinstance(key, str):
            raise TypeError(f"Expected `key` to be a string, got {type(key)}")

        if modifiers is None:
            self._key_events[key] = func
        else:
            # We need to make `modifiers` is hashable
            if isinstance(modifiers, str):
                modifiers = (modifiers,)
            elif isinstance(modifiers, (set, list)):
                modifiers = tuple(modifiers)

            if not isinstance(modifiers, tuple):
                raise TypeError(
                    f"Unexpected datatype for `modifiers`: {type(modifiers)}"
                )

            self._key_events[(key, modifiers)] = func


def handle_object_event(event, viewer, actions):
    """Handle object events.

    Note that `actions` is only used here and will not be passed to
    custom on-doubleclick / on-hover functions!
    """
    # Parse the object (this will be e.g. a Mesh visual)
    obj = event.pick_info["world_object"]

    # Get the ID of the object
    new_hover_id = [k for k, v in viewer.objects.items() if obj in v]
    new_hover_id = new_hover_id[0] if new_hover_id else None

    if new_hover_id:
        if "hide" in actions:
            viewer.hide_objects(new_hover_id)
        if "unhide" in actions:
            viewer.unhide_objects(new_hover_id)
        if "highlight" in actions:
            viewer.highlight_objects(new_hover_id)
        if "unhighlight" in actions:
            viewer.unhighlight_objects(new_hover_id)
        if "pin" in actions:
            viewer.pin_objects(new_hover_id)
        if "unpin" in actions:
            viewer.unpin_objects(new_hover_id)
        if "remove" in actions:
            viewer.remove_objects(new_hover_id)
        if "select" in actions:
            if new_hover_id in viewer.selected:
                viewer.selected = [i for i in viewer.selected if i != new_hover_id]
            else:
                viewer.selected = np.append(viewer.selected, new_hover_id)

        logger.debug(f"Object: {new_hover_id}, Action: {actions}")


def start_ipython_event_loop(gui):
    ip = get_ipython()  # noqa
    if not ip.active_eventloop:
        try:
            ip.enable_gui(gui)
            logger.debug(
                "Looks like you're running in an IPython environment but haven't "
                "started a GUI event loop. We've started one for you using the "
                f"{gui} backend."
            )
        except (ModuleNotFoundError, ImportError):
            logger.warning(
                "Looks like you're running an IPython environment but haven't "
                "started a GUI event loop. We tried to start one for you using the "
                f"{gui} backend (via %gui {gui}) but that failed. If you want a"
                "non-blocking Octarine viewer, you may have to start the event loop "
                "manually (see https://ipython.readthedocs.io/en/stable/config/eventloops.html)."
                "Otherwise just call `Viewer.show()` to start a blocking viewer."
            )
            return False
    return True
