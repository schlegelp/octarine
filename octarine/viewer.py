import png
import cmap
import uuid
import wgpu
import random
import inspect

import numpy as np
import pygfx as gfx

from functools import wraps
from collections import OrderedDict

from wgpu.gui.auto import WgpuCanvas
from wgpu.gui.offscreen import WgpuCanvas as WgpuCanvasOffscreen

from .visuals import mesh2gfx, volume2gfx, points2gfx, lines2gfx
from .conversion import get_converter
from . import utils, config


__all__ = ['Viewer']

logger = config.get_logger(__name__)

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


def update_legend(func):
    """Decorator to update legend after function call."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        if args[0].controls:
            args[0].controls.update_legend()
    return wrapper


class Viewer:
    """PyGFX 3D viewer.

    Parameters
    ----------
    offscreen : bool, optional
                If True, will use an offscreen Canvas. Useful if you only
                want a screenshot.
    title :     str, optional
                Title of the viewer window.
    max_fps :   int, optional
                Maximum frames per second to render.
    size :      tuple, optional
                Size of the viewer window.
    show :      bool, optional
                Whether to immediately show the viewer.
    show_controls : bool, optional
                If True, will show the controls widget.
                You can always show/hide the controls with
                ``viewer.show_controls()`` and ``viewer.hide_controls()``.
    **kwargs
                Keyword arguments are passed through to ``WgpuCanvas``.

    """
    # Palette used for assigning colors to objects
    palette='seaborn:tab10'

    def __init__(self,
                 offscreen=False,
                 title='Octarine Viewer',
                 max_fps=30,
                 size=None,
                 show=True,
                 show_controls=False,
                 **kwargs):
        # Check if we're running in an IPython environment
        if utils._type_of_script() == 'ipython':
            ip = get_ipython()  # noqa: F821
            if not ip.active_eventloop:
                # ip.enable_gui('qt6')
                raise ValueError('IPython event loop not running. Please use e.g. "%gui qt" to hook into the event loop.')

        self._title = title

        # Update some defaults as necessary
        defaults = {'title': title, 'max_fps': max_fps, 'size': size}
        defaults.update(kwargs)

        # If we're running in headless mode (primarily for tests on CI) we will
        # simply not initialize the gfx objects. Not ideal but it turns
        # out to be very annoying to correctly setup on Github Actions.
        if getattr(config, 'HEADLESS', False):
            return

        if not offscreen:
            self.canvas = WgpuCanvas(**defaults)
        else:
            self.canvas = WgpuCanvasOffscreen(**defaults)

        # There is a bug in pygfx 0.1.18 that causes the renderer to crash
        # when using a Jupyter canvas without explicitly setting the pixel_ratio.
        # This is already fixed in main but for now:
        if self._is_jupyter:
            self.renderer = gfx.renderers.WgpuRenderer(self.canvas, show_fps=False, pixel_ratio=2)
        else:
            self.renderer = gfx.renderers.WgpuRenderer(self.canvas, show_fps=False)

        # Set up a default scene
        self.scene = gfx.Scene()
        self.scene.add(gfx.AmbientLight(intensity=0.5))
        self.scene.add(gfx.PointLight(intensity=4))
        # Adjust shadow bias (this helps with shadow acne)
        self.scene.children[-1].shadow.bias = 0.0000005

        # Modify the light
        light = self.scene.children[-1]
        light.local.z = -10000  # move light forward
        light.local.euler_x = 2.5 # rotate light

        # Set up a default background
        self._background = gfx.BackgroundMaterial((0, 0, 0))
        self.scene.add(gfx.Background(None, self._background))

        # Add camera
        self.camera = gfx.OrthographicCamera()
        #self.camera.show_object(scene, scale=1.4)

        # Add controller
        self.controller = gfx.TrackballController(self.camera, register_events=self.renderer)

        # Stats
        self.stats = gfx.Stats(self.renderer)
        self._show_fps = False

        # Setup key events
        self.key_events = {}
        self.key_events['1'] = lambda : self.set_view('XY')
        self.key_events['2'] = lambda : self.set_view('XZ')
        self.key_events['3'] = lambda : self.set_view('YZ')
        self.key_events['f'] = lambda : self._toggle_fps()
        self.key_events['c'] = lambda : self._toggle_controls()

        def _keydown(event):
            """Handle key presses."""
            if event.key in self.key_events:
                self.key_events[event.key]()

        # Register events
        self.renderer.add_event_handler(_keydown, "key_down")

        # Finally, setting some variables
        self._show_bounds = False
        self._shadows = False
        self._animations = []

        # This starts the animation loop
        if show:
            self.show()

            # Add controls
            if show_controls:
                self.show_controls()

    def _animate(self):
        """Animate the scene."""
        to_remove = []
        for i, func in enumerate(self._animations):
            try:
                func()
            except BaseException as e:
                logger.error(f'Removing animation function {func} because of error: {e}')
                to_remove.append(i)
        for i in to_remove[::-1]:
            self.remove_animation(i)

        if self._show_fps:
            with self.stats:
                self.renderer.render(self.scene, self.camera, flush=False)
            self.stats.render()
        else:
            self.renderer.render(self.scene, self.camera)
        self.canvas.request_draw()

    def _next_color(self):
        """Return next color in the colormap."""
        # Cache the full palette. N.B. that ordering of colors in cmap depends on
        # the number of colors requested - i.e. we can't just grab the last color.
        if not hasattr(self, '__cached_palette') or self.palette != self.__cached_palette:
            self.__cached_colors = list(cmap.Colormap(self.palette).iter_colors())
            self.__cached_palette = self.palette

        return self.__cached_colors[len(self.objects) % len(self.__cached_colors)]

    def _next_label(self, prefix='Object'):
        """Return next label."""
        existing = [o for o in self.objects if o.startswith(prefix)]
        if len(existing) == 0:
            return prefix
        return f'{prefix}.{len(existing) + 1:03}'

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
    def controls(self):
        """Return the controls widget."""
        return getattr(self, '_controls', None)

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
        return [s for s in objects if getattr(objects[s][0], '_pinned', False)]

    @property
    def selected(self):
        """Return IDs of or set selected objects."""
        return self.__selected

    @selected.setter
    def selected(self, val):
        val = utils.make_iterable(val)

        objects = self.objects  # grab once to speed things up
        logger.debug(f'{len(val)} objects selected ({len(self.selected)} previously)')
        # First un-highlight neurons no more selected
        for s in [s for s in self.__selected if s not in val]:
            for v in objects[s]:
                if isinstance(v, gfx.Mesh):
                    v.color = v._stored_color
                else:
                    v.set_data(color=v._stored_color)

        # Highlight new additions
        for s in val:
            if s not in self.__selected:
                for v in objects[s]:
                    # Keep track of old colour
                    v.unfreeze()
                    v._stored_color = v.color
                    v.freeze()
                    if isinstance(v, gfx.Mesh):
                        v.color = self.highlight_color
                    else:
                        v.set_data(color=self.highlight_color)

        self.__selected = list(val)

        # Update legend
        if self.show_legend:
            self.update_legend()

        # Update data text
        # Currently only the development version of vispy supports escape
        # character (e.g. \n)
        t = '| '.join([f'{objects[s][0]._name} - #{s}' for s in self.__selected])
        self._data_text.text = t

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
            raise TypeError(f'Expected bool, got {type(v)}')

        def set_shadow(obj, state):
            if hasattr(obj, 'cast_shadow'):
                obj.cast_shadow = state
            if hasattr(obj, 'receive_shadow'):
                obj.receive_shadow = state

        if v != self._shadows:
            self._shadows = v
            for vis in self.visuals:
                set_shadow(vis, v)

            for ch in self.scene.children:
                if isinstance(ch, gfx.PointLight):
                    ch.cast_shadow = v

            #self.scene.traverse(lambda x: set_shadow(x, v))

    @property
    def visuals(self):
        """List of all visuals on this canvas."""
        return [c for c in self.scene.children if hasattr(c, '_object_id')]

    @property
    def bounds(self):
        """Bounds of all currently visuals (visible and invisible)."""
        bounds = []
        for vis in self.visuals:
            # Skip the bounding box itself
            if getattr(vis, '_object_id', '') == 'boundingbox':
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
    def _is_jupyter(self):
        """Check if Viewer is using Jupyter canvas."""
        return "JupyterWgpuCanvas" in str(type(self.canvas))

    @property
    def _is_offscreen(self):
        """Check if Viewer is using offscreen canvas."""
        return isinstance(self.canvas, WgpuCanvasOffscreen)

    @property
    def _object_ids(self):
        """All object IDs on this canvas in order of addition."""
        obj_ids = []
        for v in self.visuals:
            if hasattr(v, '_object_id'):
                obj_ids.append(v._object_id)
        return sorted(set(obj_ids), key=lambda x: obj_ids.index(x))

    @property
    def objects(self):
        """Ordered dictionary {name->[visuals]} of all objects in order of addition."""
        objects = OrderedDict()
        for ob in self._object_ids:
            objects[ob] = [v for v in self.visuals if getattr(v, '_object_id', None) == ob]

        return objects

    def add_animation(self, x):
        """Add animation function to the Viewer.

        Parameters
        ----------
        x :     callable
                Function to add to the animation loop.

        """
        if not callable(x):
            raise TypeError(f'Expected callable, got {type(x)}')

        self._animations.append(x)

    def remove_animation(self, x):
        """Remove animation function from the Viewer.

        Parameters
        ----------
        x :     callable | int
                Either the function itself or its index
                in the list of animations.

        """
        if callable(x):
            self._animations.remove(x)
        elif isinstance(x, int):
            self._animations.pop(x)
        else:
            raise TypeError(f'Expected callable or index (int), got {type(x)}')

    def show(self, use_sidecar=False):
        """Show viewer.

        Parameters
        ----------
        use_sidecar : bool
                      For Jupyter lab only: if True, will use the Sidecar
                      extension to display the viewer outside the notebooks.
                      Will throw an error if Sidecar is not installed.

        """
        # This is for e.g. headless testing
        if getattr(config, 'HEADLESS', False):
            logger.info("Viewer widget not shown - running in headless mode.")
            return

        # Start the animation loop
        self.canvas.request_draw(self._animate)

        # If this is an offscreen canvas, we don't need to show anything
        if isinstance(self.canvas, WgpuCanvasOffscreen):
            return
        # In terminal we can just show the window
        elif not self._is_jupyter:
            self.canvas.show()
        # For Jupyter we need to wrap the canvas in a widget
        else:
            if not hasattr(self, 'widget'):
                from .jupyter import JupyterOutput
                # Construct the widget
                self.widget = JupyterOutput(self, use_sidecar=use_sidecar, sidecar_kwargs={'title': self._title})
            return self.widget

    def show_controls(self):
        """Show controls."""
        if self._is_jupyter:
            logger.warning('Controls are not (yet) supported in Jupyter.')
            return

        if not hasattr(self, '_controls'):
            from .controls import Controls
            self._controls = Controls(self)
        self._controls.show()

    def hide_controls(self):
        """Hide controls."""
        if not hasattr(self, '_controls'):
            return
        self._controls.hide()

    def _toggle_controls(self):
        """Switch controls on and off."""
        if not hasattr(self, '_controls'):
            self.show_controls()
        elif self._controls.isVisible():
            self.hide_controls()
        else:
            self.show_controls()

    @update_legend
    def clear(self):
        """Clear canvas of objects (expects lights and background)."""
        # Skip if running in headless mode
        if getattr(config, 'HEADLESS', False):
            return

        # Remove everything but the lights and backgrounds
        self.scene.remove(*self.visuals)

    @update_legend
    def remove_objects(self, to_remove):
        """Remove given neurons/visuals from canvas."""
        to_remove = utils.make_iterable(to_remove)

        for vis in self.scene.children:
            if vis in to_remove:
                self.scene.children.remove(vis)
            elif hasattr(vis, '_object_id'):
                if vis._object_id in to_remove:
                    self.scene.children.remove(vis)

        if self.show_bounds:
            self.update_bounds()

    @update_legend
    def pop(self, N=1):
        """Remove the most recently added N visuals."""
        for vis in list(self.objects.values())[-N:]:
            self.remove_objects(vis)

    @property
    def show_bounds(self):
        """Set to ``True`` to show bounding box."""
        return self._show_bounds

    def toggle_bounds(self):
        """Toggle bounding box."""
        self.show_bounds = not self.show_bounds

    @show_bounds.setter
    def show_bounds(self, v):
        if not isinstance(v, bool):
            raise TypeError(f'Need bool, got {type(v)}')

        self._show_bounds = v

        if self.show_bounds:
            self.update_bounds()
        else:
            self.remove_bounds()

    def remove_bounds(self):
        """Remove bounding box visual."""
        self._show_bounds = False
        for v in self.visuals:
            if getattr(v, '_object_type', '') == 'boundingbox':
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

    def update_bounds(self, color='w', width=1):
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
        box._object_type = 'boundingbox'
        box._object_id = uuid.uuid4()

        self.scene.add(box)

    def center_camera(self):
        """Center camera on visuals."""
        if len(self):
            self.camera.show_object(
                self.scene,
                scale=1,
                view_dir=(0., 0., 1.),
                up=(0., -1., 0.)
                )

    @update_legend
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
            raise NotImplementedError(f'No converter found for {x} ({type(x)})')

        # Check if we have to provide a color
        if 'color' not in kwargs and 'color' in inspect.signature(converter).parameters:
            kwargs['color'] = tuple(self._next_color().rgba)

        visuals = utils.make_iterable(converter(x, **kwargs))

        for v in visuals:
            # If we have a name, assign it to the visual
            if name is not None:
                v._object_id = name
            # If not we either use existing ID or generate a new one
            else:
                # Give visuals an _object_id if they don't already have one
                if not hasattr(v, '_object_id'):
                    new_id = self._next_label('Object')
                    for v2 in visuals:
                        v._object_id = new_id
                elif not isinstance(v._object_id, str):
                    v._object_id = str(v._object_id)

            self.scene.add(v)

        if center:
            self.center_camera()

    @update_legend
    def add_mesh(self, mesh, name=None, color=None, center=True):
        """Add mesh to canvas.

        Parameters
        ----------
        mesh :      Mesh-like
                    Mesh to plot.
        name :      str, optional
                    Name for the visual.
        color :     str | tuple, optional
                    Color to use for plotting. Can be the name of
                    a colormap or a single color.
        center :    bool, optional
                    If True, re-center camera to all objects on canvas.

        """
        if not utils.is_mesh_like(mesh):
            raise TypeError(f'Expected mesh-like object, got {type(mesh)}')
        if color is None:
            color = self._next_color()
        if name is None:
            name = self._next_label('Mesh')
        elif not isinstance(name, str):
            name = str(name)

        visual = mesh2gfx(mesh, color=color)
        visual._object_id = name if name else uuid.uuid4()
        self.scene.add(visual)

        if center:
            self.center_camera()

    @update_legend
    def add_points(self, points, name=None, color=None, size=2, center=True):
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
        size :      int | float
                    Marker size.
        center :    bool, optional
                    If True, re-center camera to all objects on canvas.

        """
        if not isinstance(points, np.ndarray):
            raise TypeError(f'Expected numpy array, got {type(points)}')
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f'Expected (N, 3) array, got {points.shape}')
        if color is None:
            color = self._next_color()
        if name is None:
            name = self._next_label('Scatter')
        elif not isinstance(name, str):
            name = str(name)

        visual = points2gfx(points, color=color, size=size)
        visual._object_id = name if name else uuid.uuid4()
        self.scene.add(visual)

        if center:
            self.center_camera()

    @update_legend
    def add_lines(self, lines, name=None, color=None, linewidth=1, center=True):
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
        center :    bool, optional
                    If True, re-center camera to all objects on canvas.

        """
        # TODO:
        # - allow providing a tuple of (positions, edges) for lines

        if isinstance(lines, np.ndarray):
            if lines.ndim != 2 or lines.shape[1] != 3:
                raise ValueError(f'Expected (N, 3) array, got {lines.shape}')
        elif isinstance(lines, list):
            if not all([l.ndim == 2 and l.shape[1] == 3 for l in lines]):
                raise ValueError('Expected list of (N, 3) arrays.')
        else:
            raise TypeError(f'Expected numpy array or list, got {type(lines)}')

        if color is None:
            color = self._next_color()
        if name is None:
            name = self._next_label('Lines')
        elif not isinstance(name, str):
            name = str(name)

        visual = lines2gfx(lines, linewidth=linewidth, color=color)
        visual._object_id = name if name else uuid.uuid4()
        self.scene.add(visual)

        if center:
            self.center_camera()

    @update_legend
    def add_volume(self, volume, dims, name=None, color=None, offset=(0, 0, 0), cmin=None, cmax='auto', center=True):
        """Add image volume to canvas.

        Parameters
        ----------
        volume :    (N, M, K) array
                    Volume to plot.
        dims :      tuple
                    Scale factors for the volume.
        name :      str, optional
                    Name for the visual.
        color :     tuple, optional
                    Color to use for plotting. Can be the name of
                    a colormap or a single color.
        offset :    tuple, optional
                    Offset for the volume.
        cmin/cmax : float | "auto", optional
                    Min/max values for the colormap. If "auto", will
                    use the min/max of the volume. If `None` will determine
                    the min/max based on the data type of `volume`.
        center :    bool, optional
                    If True, re-center camera to all objects on canvas.

        """
        if not isinstance(volume, np.ndarray):
            raise TypeError(f'Expected numpy array, got {type(volume)}')
        if volume.ndim != 3:
            raise ValueError(f'Expected 3D array, got {volume.ndim}')
        if color is None:
            color = self._next_color()
        if name is None:
            name = self._next_label('Volume')
        elif not isinstance(name, str):
            name = str(name)

        visual = volume2gfx(volume, dims=dims, offset=offset, color=color, cmin=cmin, cmax=cmax)
        visual._object_id = name if name else uuid.uuid4()
        self.scene.add(visual)

        if center:
            self.center_camera()

    def close(self):
        """Close the viewer."""
        # Skip if this is headless mode
        if getattr(config, 'HEADLESS', False):
            return

        # Clear first to free all visuals
        self.clear()

        # Remove from config if this is the primary viewer
        if self == getattr(config, 'PRIMARY_VIEWER', None):
            del config.PRIMARY_VIEWER

        # Close if not already closed
        if not self.canvas.is_closed():
            self.canvas.close()

        if hasattr(self, '_controls'):
            self._controls.close()

        # Close the Jupyter widget
        if hasattr(self, 'widget') and not getattr(self.widget, '_is_closed', False):
            self.widget.close(close_viewer=False)

    def hide_objects(self, obj):
        """Hide given object(s).

        Parameters
        ----------
        obj :   str | list
                Object(s) to hide.

        """
        objects = self.objects   # grab once to speed things up
        for ob in utils.make_iterable(obj):
            if ob not in objects:
                logger.warning(f'Object "{ob}" not found on canvas.')
                continue
            for v in objects[ob]:
                if getattr(v, '_pinned', False):
                    continue
                if v.visible:
                    v.visible = False

    def hide_selected(self):
        """Hide currently selected object(s)."""
        self.hide_neurons(self.selected)

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
                logger.warning(f'Object {ob} not found on canvas.')
                continue
            for v in objects[ob]:
                if getattr(v, '_pinned', False):
                    continue
                if not v.visible:
                    v.visible = True

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
                logger.warning(f'Object {ob} not found on canvas.')
                continue
            for v in objects[ob]:
                v._pinned = True

    def unpin_objects(self, obj=None):
        """Unpin given object(s).

        Use ``obj`` to unhide specific neurons.

        """
        objects = self.objects  # grab once to speed things up
        if not isinstance(obj, type(None)):
            obj = utils.make_iterable(obj)
        else:
            obj = objects

        for ob in obj:
            if ob not in objects:
                logger.warning(f'Object {ob} not found on canvas.')
                continue
            for v in objects[ob]:
                v.unfreeze()
                v._pinned = False
                v.freeze()

    @update_legend
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
                    if getattr(v, '_pinned', False):
                        continue
                    if not hasattr(v, 'material'):
                        continue
                    # Note: there is currently a bug where removing or adding an alpha
                    # channel from a color will break the rendering pipeline
                    if len(v.material.color) == 4:
                        new_c = gfx.Color(cmap[n]).rgba
                    else:
                        new_c = gfx.Color(cmap[n]).rgb
                    v.material.color = gfx.Color(new_c)

    def colorize(self, palette='seaborn:tab10', objects=None, randomize=True):
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
        self._show_fps = not self._show_fps

    def screenshot(self,
                   filename='screenshot.png',
                   size=None,
                   pixel_ratio=None,
                   alpha=True):
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
            if not filename.endswith('.png'):
                filename += '.png'
            png.from_array(im.reshape(im.shape[0], im.shape[1] * im.shape[2]), mode='RGBA').save(filename)
        else:
            return im

    def _screenshot(self, alpha=True, size=None, pixel_ratio=None):
        """Return image array for screenshot."""
        if alpha:
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

        try:
            im = self.renderer.snapshot()
        except BaseException:
            raise
        finally:
            if alpha:
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
        view :      XY | XZ | YZ
                    View to set.

        """
        if view == 'XY':
            self.camera.show_object(self.scene, view_dir=(0., 0., 1.), up=(0., -1., 0.))
        elif view == 'XZ':
            self.camera.show_object(self.scene, scale=1, view_dir=(0., 1., 0.), up=(0., 0., 1.))
        elif view == 'YZ':
            self.camera.show_object(self.scene, scale=1, view_dir=(-1., 0., 0.), up=(0., -1., 0.))
        else:
            raise TypeError(f'Unable to set view from {type(view)}')
