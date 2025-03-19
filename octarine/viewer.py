import png
import sys
import time
import cmap
import uuid
import random
import inspect
import warnings

import numpy as np
import pygfx as gfx
import trimesh as tm

from functools import wraps, lru_cache, partial
from collections import OrderedDict

from wgpu.gui.offscreen import WgpuCanvas as WgpuCanvasOffscreen

from .visuals import mesh2gfx, volume2gfx, points2gfx, lines2gfx, text2gfx
from .conversion import get_converter
from . import utils, config


__all__ = ["Viewer", "viewers"]

logger = config.get_logger(__name__)

# This keeps track of open viewers
viewers = []

AUTOSTART_EVENT_LOOP = True

# TODO
# - add styles for viewer (lights, background, etc.) - e.g. .set_style(dark)
#   - e.g. material.metalness = 2 looks good for background meshes
#   - metalness = 1 with roughness = 0 makes for funky looking neurons
#   - m.material.side = "FRONT" makes volumes look better
# - make Viewer reactive (see reactive_rendering.py) to save
#   resources when not actively using the viewer - might help in Jupyter?
# [/] add specialised methods for adding neurons, volumes, etc. to the viewer
# - move lights to just outside the scene's bounding box (maybe use decorator?)
#   whenever we add/remove objects


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
        if getattr(viewer, "show_bounds", False):
            viewer.update_bounds()

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
        **kwargs,
    ):
        # We need to import WgpuCanvas before we (potentially) start the event loop
        # If we don't we get a segfault.
        if not offscreen:
            from wgpu.gui.auto import WgpuCanvas

        # Check if we're running in an IPython environment
        if utils._type_of_script() == "ipython":
            ip = get_ipython()  # noqa: F821
            if not ip.active_eventloop:
                if AUTOSTART_EVENT_LOOP:
                    try:
                        ip.enable_gui("qt6")
                        logger.debug(
                            "Looks like you're running in an IPython environment but haven't "
                            "started a GUI event loop. We've started one for you using the "
                            "Qt6 backend."
                        )
                    except ModuleNotFoundError:
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

        self._title = title

        # Update some defaults as necessary
        defaults = {"title": title, "max_fps": max_fps, "size": size}
        defaults.update(kwargs)

        # If we're running in headless mode (primarily for tests on CI) we will
        # simply not initialize the gfx objects. Not ideal but it turns
        # out to be very annoying to correctly setup on Github Actions.
        if getattr(config, "HEADLESS", False):
            return

        if not offscreen:
            self.canvas = WgpuCanvas(**defaults)
        else:
            self.canvas = WgpuCanvasOffscreen(**defaults)

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
        self.scene.add(gfx.PointLight(intensity=4))
        self.scene.children[-1].shadow.bias = 0.0000005  # this helps with shadow acne
        self.scene.children[-1].local.x = -1000000  # move to the left
        self.scene.children[-1].local.y = -1000000  # move up
        self.scene.children[-1].local.z = -1000000  # move light forward

        # A weaker point light from the back
        self.scene.add(gfx.PointLight(intensity=1))
        self.scene.children[-1].shadow.bias = 0.0000005  # this helps with shadow acne
        self.scene.children[-1].local.x = 1000000  # move to the left
        self.scene.children[-1].local.y = 1000000  # move up
        self.scene.children[-1].local.z = 1000000  # move light forward

        # Set up a default background
        self._background = gfx.BackgroundMaterial((0, 0, 0))
        self.scene.add(gfx.Background(None, self._background))

        # Add camera
        if camera == "ortho":
            self.camera = gfx.OrthographicCamera()
        elif camera == "perspective":
            self.camera = gfx.PerspectiveCamera()
        else:
            raise ValueError(f"Unknown camera type: {camera}")

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
        self._animations = {}
        self._animations_flagged_for_removal = []
        self._animations_frame_counter = 0
        self._on_double_click = None
        self._on_hover = None
        self._objects_pickable = False
        self._selected = []
        self._render_trigger = "continuous"

        viewers.append(self)

        # This starts the animation loop
        if show and not self._is_jupyter:
            self.show(start_loop=show == "start_loop")

    def _animate(self):
        """Run the rendering loop."""
        rm = self.render_trigger

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
                elif on_error == "remove":
                    logger.error(
                        f"Removing animation function '{func}' because of error: {e}"
                    )
                    # Flag animation for removal
                    self._animations_flagged_for_removal.append(i)

        # Check if any animations need to be removed
        for f in self._animations_flagged_for_removal[::-1]:
            self._animations.pop(f)
        self._animations_flagged_for_removal = []

        # Now check if we need to render the scene
        if rm == "active_window":
            # Note to self: we need to explore how to do this with different backends / Window managers
            # Not sure if this will work with e.g. Jupyter (does it know when the notebook is active?)
            if hasattr(self.canvas, "isActiveWindow"):
                if not self.canvas.isActiveWindow():
                    self.canvas.request_draw()
                    return
        elif rm == "reactive":
            # If the scene is not stale, we can skip rendering
            if not getattr(self, "_render_stale", False):
                self.canvas.request_draw()
                return

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

        self.canvas.request_draw()

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
        """Render blend mode.

        This is a simple shortcut, see `Viewer.renderer.blend_mode` for more information.

        """
        return self.renderer.blend_mode

    @blend_mode.setter
    def blend_mode(self, mode):
        if mode == "additive" and self.transform_gizmo is not None:
            logger.warning(
                "Setting blend mode to 'additive' may break interaction with the transform gizmo."
            )
        self.renderer.blend_mode = mode

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
    def shadows(self):
        """Return shadow state."""
        return self._shadows

    @shadows.setter
    def shadows(self, v):
        """Set shadow state."""
        if not isinstance(v, bool):
            raise TypeError(f"Expected bool, got {type(v)}")

        def set_shadow(obj, state):
            if hasattr(obj, "cast_shadow"):
                obj.cast_shadow = state
            if hasattr(obj, "receive_shadow"):
                obj.receive_shadow = state

        if v != self._shadows:
            self._shadows = v
            for vis in self.visuals:
                set_shadow(vis, v)

            for ch in self.scene.children:
                if isinstance(ch, gfx.PointLight):
                    ch.cast_shadow = v

            # self.scene.traverse(lambda x: set_shadow(x, v))

    @property
    def visuals(self):
        """List of all visuals on this canvas."""
        return [c for c in self.scene.children if hasattr(c, "_object_id")]

    @property
    def bounds(self):
        """Bounds of all currently visuals (visible and invisible)."""
        bounds = []
        for vis in self.visuals:
            # Skip the bounding box itself
            if getattr(vis, "_object_id", "") == "boundingbox":
                continue

            try:
                bounds.append(vis._bounds)
            except BaseException:
                pass

        if not bounds:
            return None

        bounds = np.dstack(bounds)

        mn = bounds[:, 0, :].min(axis=1)
        mx = bounds[:, 1, :].max(axis=1)

        return np.vstack((mn, mx)).T

    @property
    def max_fps(self):
        """Maximum frames per second to render."""
        return self.canvas._subwidget._max_fps

    @max_fps.setter
    def max_fps(self, v):
        assert isinstance(v, int)
        self.canvas._subwidget._max_fps = v

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
        return "JupyterWgpuCanvas" in str(type(self.canvas))

    @property
    def _is_offscreen(self):
        """Check if Viewer is using offscreen canvas."""
        return isinstance(self.canvas, WgpuCanvasOffscreen)

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

        """
        return self._on_double_click

    @on_double_click.setter
    def on_double_click(self, v):
        valid = (None, "hide", "remove", "select")
        if v not in valid:
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
            func = partial(handle_object_event, viewer=self, actions=(v,))
            self.scene.add_event_handler(func, "double_click")
            self.__on_double_click_func = func

        self._on_double_click = v

    def add_animation(self, x, on_error="remove", run_every=None, req_render=True):
        """Add animation function to the Viewer.

        Parameters
        ----------
        x :         callable
                    Function to add to the animation loop.
        on_error :  "remove" | "ignore" | "raise"
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

        assert on_error in ["remove", "ignore", "raise"]

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

    def show(self, use_sidecar=False, toolbar=False, start_loop=False):
        """Show viewer.

        Parameters
        ----------

        For Jupyter lab only:

        use_sidecar : bool
                      If True, will use the Sidecar extension to display the
                      viewer outside the notebooks. Will throw an error if
                      Sidecar is not installed.
        toolbar :     bool
                      If True, will show a toolbar. You can always show/hide
                      the toolbar with ``viewer.show_controls()`` and
                      ``viewer.hide_controls()``, or the `c` hotkey.

        For scripts & standard REPL:

        start_loop :  bool
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
        if isinstance(self.canvas, WgpuCanvasOffscreen):
            return

        # In terminal we can just show the window
        if not self._is_jupyter:
            # Not all backends have a show method (e.g. GLFW does not)
            if hasattr(self.canvas, "show"):
                self.canvas.show()

            if start_loop:
                from wgpu.gui.auto import run
                run()
            elif utils._type_of_script() in ("terminal", "script"):
                logger.warning(
                    "Running in a (potentially) non-interactive terminal or script "
                    "environment. You may have to manually start the event loop "
                    "for the canvas to render:\n\n"
                    "  >>> v = octarine.Viewer(show=False)\n"
                    "  >>> ...  # setup your viewer\n"
                    "  >>> v.show(start_loop=True)\n\n"
                    "Alternatively, use WGPU's run() function:\n\n"
                    "  >>> from wgpu.gui.auto import run\n"
                    "  >>> ...  # setup your viewer\n"
                    "  >>> v.show()\n"
                    "  >>> run()\n\n"  # do not remove the \n\n here
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

    def center_camera(self):
        """Center camera on visuals."""
        if len(self):
            self.camera.show_object(
                self.scene, scale=1, view_dir=(0.0, 0.0, 1.0), up=(0.0, -1.0, 0.0)
            )

    @update_viewer(legend=True, bounds=True)
    def add(self, x, name=None, center=True, clear=False, **kwargs):
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

        if utils.is_iterable(x) and not isinstance(x, np.ndarray):
            for xx in x:
                self.add(xx, center=False, clear=False, name=name, **kwargs)
            if center:
                self.center_camera()
            return

        converter = get_converter(x, raise_missing=False)
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

            self.scene.add(v)

        if center:
            self.center_camera()

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

        if center:
            self.center_camera()

    def add_mesh(self, mesh, name=None, color=None, alpha=None, center=True):
        """Add mesh to canvas.

        Parameters
        ----------
        mesh :      Mesh-like
                    Mesh to plot. If this is a pygfx.Mesh, it will be added
                    directly to the scene without modification (i.e. `color`,
                    `alpha`, etc. will be ignored).
        name :      str, optional
                    Name for the visual.
        color :     str | tuple, optional
                    Color to use for plotting. If multiple colors,
                    must be a list of colors with the same length as
                    the number of faces or vertices.
        alpha :     float, optional
                    Opacity value [0-1]. If provided, will override
                    the alpha channel of the color.
        center :    bool, optional
                    If True, re-center camera to all objects on canvas.

        """
        if isinstance(mesh, tm.Scene):
            for _, ob in mesh.geometry.items():
                self.add_mesh(ob, name=name, color=color, alpha=alpha, center=False)
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
            visual = mesh2gfx(mesh, color=color, alpha=alpha)
        else:
            visual = mesh

        visual._object_id = name if name else uuid.uuid4()

        self._add_to_scene(visual, center)

    def add_points(
        self,
        points,
        name=None,
        color=None,
        marker=None,
        size=2,
        size_space="screen",
        center=True,
    ):
        """Add points plot to canvas.

        Parameters
        ----------
        points :    (N, 3) array
                    Points to plot.
        name :      str, optional
                    Name for the visual.
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
            points, color=color, size=size, size_space=size_space, marker=marker
        )
        visual._object_id = name if name else uuid.uuid4()

        self._add_to_scene(visual, center)

    def add_lines(
        self,
        lines,
        name=None,
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
        color :     str | tuple, optional
                    Color to use for plotting. Can be a single color
                    or one for every point in the line(s).
        linewidth : float, optional
                    Line width.
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
        self._add_to_scene(visual, center)

    def add_volume(
        self,
        volume,
        spacing=(1, 1, 1),
        name=None,
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
            self._add_to_scene(vis, center)

    def close(self):
        """Close the viewer."""
        # Skip if this is headless mode
        if getattr(config, "HEADLESS", False):
            return

        # Clear first to free all visuals
        self.clear()

        # Remove from config if this is the primary viewer
        if self == getattr(config, "PRIMARY_VIEWER", None):
            del config.PRIMARY_VIEWER

        # Close if not already closed
        if not self.canvas.is_closed():
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

    @update_viewer(legend=True, bounds=True)
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

    @update_viewer(legend=True, bounds=True)
    def hide_selected(self):
        """Hide currently selected object(s)."""
        self.hide_neurons(self.selected)

    @update_viewer(legend=True, bounds=True)
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

    def highlight_objects(self, obj, color=0.2):
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
                    # Work in HSL space
                    h, s, l = o.material.color.to_hsl()
                    # If the value is not maxed yet, increase it
                    if l < 1:
                        l = min(l + color, 1)
                    else:
                        l = max(l - color, 0)

                    new_color = gfx.Color.from_hsl(h, s, l)
                else:
                    # See if pygfx can handle the color
                    new_color = gfx.Color(color)

                o.material._original_color = o.material.color
                o.material.color = new_color
                o._highlighted = True

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

    @update_viewer(legend=True, bounds=False)
    def set_colors(self, c):
        """Set object color.

        Parameters
        ----------
        c :      tuple | dict
                 RGB color(s) to apply. Values must be 0-1. Accepted:
                   1. Tuple of single color. Applied to all visible objects.
                   2. Dictionary names/IDs to colors.

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
                    v.material.color = gfx.Color(new_c)

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

    def set_bgcolor(self, c):
        """Set background color.

        Parameters
        ----------
        c :     tuple | str
                RGB(A) color to use for the background.

        """
        self._background.set_colors(gfx.Color(c).rgba)

    def _toggle_fps(self):
        """Switch FPS measurement on and off."""
        self.show_fps = not self.show_fps

    def screenshot(
        self, filename="screenshot.png", size=None, pixel_ratio=None, alpha=True
    ):
        """Save a screenshot of the canvas.

        Parameters
        ----------
        filename :      str, optional
                        Filename to save to. If ``None``, will return image array.
                        Note that this will always save a PNG file, no matter
                        the extension.
        size :          tuple, optional
                        Size of the screenshot. If provided, will temporarily
                        change the canvas size.
        pixel_ratio :   int, optional
                        Factor by which to scale canvas. Determines image
                        dimensions. Note that this seems to have no effect
                        on offscreen canvases.
        alpha :         bool, optional
                        If True, will export transparent background.

        """
        im = self._screenshot(alpha=alpha, size=size, pixel_ratio=pixel_ratio)
        if filename:
            if not filename.endswith(".png"):
                filename += ".png"
            png.from_array(
                im.reshape(im.shape[0], im.shape[1] * im.shape[2]), mode="RGBA"
            ).save(filename)
        else:
            return im

    def _screenshot(self, alpha=True, size=None, pixel_ratio=None):
        """Return image array for screenshot."""
        if alpha:
            oc = [
                self._background.color_bottom_left,
                self._background.color_bottom_right,
                self._background.color_top_left,
                self._background.color_top_right,
            ]
            self._background.set_colors((0, 0, 0, 0))
            op = self._background.opacity
            self._background.opacity = 0
        if size:
            os = self.size
            self.size = size
        if pixel_ratio:
            opr = self.renderer.pixel_ratio
            self.renderer.pixel_ratio = pixel_ratio

        # If this is an offscreen canvas, we need to manually trigger a draw first
        # Note: this has to happen _after_ adjust parameters!
        if isinstance(self.canvas, WgpuCanvasOffscreen):
            self.canvas.draw()
        else:
            # This is a bit of a hack to make sure a new frame with the (potentially)
            # updated size, pixel ratio, etc. is drawn before taking the screenshot.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.canvas.draw_frame()

        try:
            im = self.renderer.snapshot()
        except BaseException:
            raise
        finally:
            if alpha:
                self._background.set_colors(*oc)
                self._background.opacity = op
            if size:
                self.size = os
            if pixel_ratio:
                self.renderer.pixel_ratio = opr

        return im

    def set_view(self, view):
        """(Re-)set camera position.

        Parameters
        ----------
        view :      XY | XZ | YZ | dict
                    View to set. Can be inverted to e.g. "-XY" to show view from back.
                    If a dictionary, should describe the state of the camera. Typically,
                    this is obtained by calling `viewer.get_view()`.

        """
        if view == "XY":
            self.camera.show_object(
                self.scene, view_dir=(0.0, 0.0, 1.0), up=(0.0, -1.0, 0.0)
            )
        elif view == "-XY":
            self.camera.show_object(
                self.scene, view_dir=(0.0, 0.0, -1.0), up=(0.0, -1.0, 0.0)
            )
        elif view == "XZ":
            self.camera.show_object(
                self.scene, scale=1, view_dir=(0.0, 1.0, 0.0), up=(0.0, 0.0, 1.0)
            )
        elif view == "-XZ":
            self.camera.show_object(
                self.scene, scale=1, view_dir=(0.0, -1.0, 0.0), up=(0.0, 0.0, 1.0)
            )
        elif view == "YZ":
            self.camera.show_object(
                self.scene, scale=1, view_dir=(1.0, 0.0, 0.0), up=(0.0, -1.0, 0.0)
            )
        elif view == "YZ":
            self.camera.show_object(
                self.scene, scale=1, view_dir=(-1.0, 0.0, 0.0), up=(0.0, -1.0, 0.0)
            )
        elif isinstance(view, dict):
            self.camera.set_state(view)
        else:
            raise TypeError(f"Unable to set view from {type(view)}")

    def get_view(self):
        """Get current camera position."""
        return self.camera.get_state()

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
    """Handle object events."""
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
        except ModuleNotFoundError:
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
