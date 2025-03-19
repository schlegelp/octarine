"""
A selection gizmo for the 3d viewer.
"""

import numpy as np
import pylinalg as la
import pygfx as gfx

from pygfx.objects import WorldObject
from pygfx.utils.viewport import Viewport
from pygfx.utils.transform import AffineTransform

from .visuals import lines2gfx


class BaseSelectionGizmo(WorldObject):
    """Gizmo to manage a Selection Box.

    To invoke the Gizmo shift-click on the canvas (see also `modifier`) and start dragging
    to draw a selection rectangle.

    Parameters
    ----------
    viewer :  Viewer
        The viewer to which the gizmo is attached.
    modifier : str
        The modifier key to use to activate the gizmo. Default "Shift".
    outline_color : str, optional
        The color to use for the edge of the selection box. Set to None to disable outline.
    outline_color : str, optional
        The color to use for the fill of the selection box. Set to None to disable fill.
    outline_width : int
        The width of the selection box lines.
    outline_style : str
        The style of the selection box lines.
    force_square : bool
        Whether to force the selection box to be square.
    show_info : bool
        Whether to show a small box with additional infos on the selection box.
    debug : bool
        Whether to print debug information.
    leave : bool
        Whether to leave the selection box after the selection is done.
    callback_after : callable
        A function to call after the selection is done.
    callback_during : callable
        A function to call during the selection.
    name : str
        An identifier for the gizmo.

    """

    def __init__(
        self,
        viewer,
        modifier="Shift",
        outline_color="w",
        outline_opacity=0.7,
        outline_width=1,
        outline_style="dashed",
        fill_color=None,
        fill_opacity=0.3,
        force_square=False,
        show_info=False,
        info_font_size=0.1,
        debug=False,
        leave=False,
        callback_after=None,
        callback_during=None,
        name="SelectionGizmo",
    ):
        assert modifier in ("Shift", "Ctrl", "Alt", None)

        super().__init__()

        self._viewer = viewer
        self._viewport = Viewport.from_viewport_or_renderer(self._viewer.renderer)
        self._camera = self._viewer.overlay_camera  # this is the NDC camera
        self._scene = self._viewer.overlay_scene  # we draw on the overlay scene
        self._scene.add(self)
        self._ndc_to_screen = None
        self._add_default_event_handlers()  # this sets up the event handlers

        # Init - N.B. we're prefixing _ to avoid conflicts with WorldObject properties
        self._debug = debug
        self._show_info = show_info
        self._info_font_size = info_font_size
        self._outline_style = outline_style
        self._outline_width = outline_width
        self._modifier = modifier
        self._outline_color = outline_color
        self._outline_opacity = outline_opacity
        self._fill_color = fill_color
        self._fill_opacity = fill_opacity
        self._force_square = force_square
        self._leave = leave
        self._callback_after = callback_after
        self._callback_during = callback_during
        self._name = name

        self.visible = False  # N.B. this is the WorldObject visibility (no underscore)!
        self._active = False
        self._disable = False  # Can be set to True to (temporarily) disable the gizmo
        self._sel_info = {}
        self._elements = {}

        # Generate the visuals
        self.create_visuals()

    @property
    def bounds(self):
        """Return bounds based on selection.

        Importantly, this always returns the sorted bounds while
        self._sel_info may not be sorted.
        """
        if not self._sel_info:
            return None
        sel = np.vstack([self._sel_info["start"], self._sel_info["end"]])
        return np.vstack([sel.min(axis=0), sel.max(axis=0)])

    def __del__(self):
        """Clean-up."""
        self._clear()
        super().__del__()

    def _clear(self):
        """Remove self from viewer."""
        try:
            self._remove_default_event_handlers()
        except KeyError:
            pass
        self._scene.remove(self)

    def create_visuals(self):
        """Create selection elements."""
        self._visual_fill = None
        self._visual_outline = None

        # Generate fill
        if self._fill_color:
            self._visual_fill = gfx.Mesh(
                gfx.Geometry(
                    positions=np.array(
                        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)],
                        dtype=np.float32,
                    ),
                    indices=np.array([[0, 1, 2, 3]], dtype=np.int32),
                ),
                gfx.MeshBasicMaterial(color=self._fill_color),
            )
            self._visual_fill.material.opacity = self._fill_opacity
            self.add(self._visual_fill)

        # Generate outline
        if self._outline_color:
            self._visual_outline = lines2gfx(
                np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)]),
                self._outline_color,
                dash_pattern=self._outline_style,
                linewidth=self._outline_width,
            )
            self._visual_outline.material.opacity = self._outline_opacity
            self.add(self._visual_outline)

        # Generate info text
        if self._show_info:
            self._show_info = (
                gfx.Text(
                    markdown="",
                    font_size=self._info_font_size,
                    anchor="bottomright",
                    material=gfx.TextMaterial(
                        color=self._outline_color
                        if self._outline_color
                        else self._fill_color
                    ),
                ),
                gfx.Text(
                    markdown="",
                    font_size=self._info_font_size,
                    anchor="topleft",
                    material=gfx.TextMaterial(
                        color=self._outline_color
                        if self._outline_color
                        else self._fill_color
                    ),
                ),
            )
            self.add(self._show_info[0])
            self.add(self._show_info[1])

    def _update_gizmo(self, event):
        """Update the transform."""
        if event.type != "before_render":
            return

        # Only update if gizmo is actually active
        if self._active and not self._disable:
            self._update_ndc_screen_transform()

    def _update_ndc_screen_transform(self):
        """Update the NDC (Normalized Device Coordinates) to screen transform."""
        # Note: screen origin is at top left corner of NDC with Y-axis pointing down
        x_dim, y_dim = self._viewport.logical_size
        screen_space = AffineTransform()
        screen_space.position = (-1, 1, 0)
        screen_space.scale = (2 / x_dim, -2 / y_dim, 1)
        self._ndc_to_screen = screen_space.inverse_matrix
        self._screen_to_ndc = screen_space.matrix

    def _screen_to_world(self, pos):
        """Translate screen position to world coordinates."""
        if not self._viewport.is_inside(*pos):
            return None

        # Get position relative to viewport
        pos_rel = (
            pos[0] - self._viewport.rect[0],
            pos[1] - self._viewport.rect[1],
        )

        vs = self._viewport.logical_size

        # Convert position to NDC
        x = pos_rel[0] / vs[0] * 2 - 1
        y = -(pos_rel[1] / vs[1] * 2 - 1)
        pos_ndc = (x, y, 0)

        pos_ndc += la.vec_transform(
            self._camera.world.position, self._camera.camera_matrix
        )
        pos_world = la.vec_unproject(pos_ndc[:2], self._camera.camera_matrix)

        return pos_world

    def _add_default_event_handlers(self):
        """Register Gizmo callbacks."""
        # Add handler for gizmo events
        self._viewport.renderer.add_event_handler(
            self._process_event,
            "pointer_down",
            "pointer_move",
            "pointer_up",
        )
        # Add handler to update gizmo (i.e. the transform)
        self._viewport.renderer.add_event_handler(self._update_gizmo, "before_render")

    def _remove_default_event_handlers(self):
        """Remove Gizmo callbacks."""
        self._viewport.renderer.remove_event_handler(
            self._process_event,
            "pointer_down",
            "pointer_move",
            "pointer_up",
        )
        self._viewport.renderer.remove_event_handler(
            self._update_gizmo, "before_render"
        )

    def _process_event(self, event):
        """Callback to handle gizmo-related events."""
        # If gizmo is disabled, just return
        if self._disable:
            return

        # Triage over event type
        has_mod = self._modifier is None or (self._modifier in event.modifiers)
        if event.type == "pointer_down" and has_mod:
            self._selection_start(event)
            self._viewport.renderer.request_draw()
        elif event.type == "pointer_up" and self._active:
            self._selection_stop(event)
            self._viewport.renderer.request_draw()
        elif event.type == "pointer_move" and self._active:
            self._selection_move(event)
            self._viewport.renderer.request_draw()

    def _selection_start(self, event):
        """Initialize the drag."""
        # Set the positions of the selection rectangle
        world_pos = self._screen_to_world((event.x, event.y))

        if world_pos is None:
            return

        # Set the rectangle to visible
        self.visible = True
        self._active = True
        self._event_modifiers = event.modifiers  # store the modifiers

        if self._visual_outline:
            self._visual_outline.geometry.positions.data[:, 0] = world_pos[0]
            self._visual_outline.geometry.positions.data[:, 1] = world_pos[1]
            self._visual_outline.geometry.positions.update_range()
        if self._visual_fill:
            self._visual_fill.geometry.positions.data[:, 0] = world_pos[0]
            self._visual_fill.geometry.positions.data[:, 1] = world_pos[1]
            self._visual_fill.geometry.positions.update_range()

        # In debug mode we will add points
        if self._debug:
            print("Starting at ", world_pos)
            self.remove(*[c for c in self.children if isinstance(c, gfx.Points)])
            point = gfx.Points(
                gfx.Geometry(
                    positions=np.array(
                        [[world_pos[0], world_pos[1], 0]], dtype=np.float32
                    )
                ),
                material=gfx.PointsMaterial(color="r", size=10),
            )
            self.add(point)

        # Store the selection box coordinates
        self._sel_info = {
            "start": world_pos,
            "end": world_pos,
        }

        # Update info text (if applicable)
        self._update_info()

    def _selection_stop(self, event):
        """Stop the drag on pointer up."""
        # Set the rectangle to invisible
        self._active = False
        if not self._leave:
            self.visible = False

        if self._debug:
            world_pos = self._screen_to_world((event.x, event.y))
            point = gfx.Points(
                gfx.Geometry(
                    positions=np.array(
                        [[world_pos[0], world_pos[1], 0]], dtype=np.float32
                    )
                ),
                material=gfx.PointsMaterial(color="g", size=10),
            )
            self.add(point)
            print("Stopping with Selection box: ", self._sel_info)

        if self._callback_after:
            self._callback_after(self)

    def _selection_move(self, event):
        """Move the selection box with the cursor."""
        # Set the positions of the rectangle
        world_pos = self._screen_to_world((event.x, event.y))

        # If we're outside viewport, return
        if world_pos is None:
            return

        # Make square if necessary
        if self._force_square:
            dx, dy, dz = world_pos - self._sel["start"]
            dmin = min(abs(dx), abs(dy))
            world_pos[0] = self._sel_info["start"][0] + np.sign(dx) * dmin
            world_pos[1] = self._sel_info["start"][1] + np.sign(dy) * dmin

        if self._visual_outline:
            # The first and the last point on the line remain on the origin
            # The second point goes to (origin, new_y), the third to (new_x, new_y)
            # The fourth to (new_x, origin)
            self._visual_outline.geometry.positions.data[1, 1] = world_pos[1]
            self._visual_outline.geometry.positions.data[2, 0] = world_pos[0]
            self._visual_outline.geometry.positions.data[2, 1] = world_pos[1]
            self._visual_outline.geometry.positions.data[3, 0] = world_pos[0]
            self._visual_outline.geometry.positions.update_range()

        if self._visual_fill:
            self._visual_fill.geometry.positions.data[1, 1] = world_pos[1]
            self._visual_fill.geometry.positions.data[2, 0] = world_pos[0]
            self._visual_fill.geometry.positions.data[2, 1] = world_pos[1]
            self._visual_fill.geometry.positions.data[3, 0] = world_pos[0]
            self._visual_fill.geometry.positions.update_range()

        # Store the selection box coordinates
        self._sel_info["end"] = world_pos

        # Update info text (if applicable)
        self._update_info()

        if self._debug:
            print(f"{self._name} moving to {world_pos}")
            point = gfx.Points(
                gfx.Geometry(
                    positions=np.array(
                        [[world_pos[0], world_pos[1], 0]], dtype=np.float32
                    )
                ),
                material=gfx.PointsMaterial(color="w", size=10),
            )
            self.add(point)

    def _update_info(self):
        """Update the info text."""
        if not self._show_info:
            return

        if not self._sel_info:
            return

        self._show_info[0].set_text(
            f"({self._sel_info['start'][0]:.2f}, {self._sel_info['start'][1]:.2f})"
        )
        self._show_info[1].set_text(
            f"({self._sel_info['end'][0]:.2f}, {self._sel_info['end'][1]:.2f})"
        )

        self._show_info[0].local.position = self._sel_info["start"]
        self._show_info[1].local.position = self._sel_info["end"]


# We're wrapping the BaseSelectionGizmo in another class to provide a more user-friendly interface
class SelectionGizmo:
    """Selection gizmo for the 3D viewer.

    This gizmo allows to select objects in the 3d viewer by drawing a rectangle.

    Parameters
    ----------
    viewer : Viewer
        The viewer to which the gizmo is attached.
    callback : callable, optional
        A function to call when the selection is done. Must accept the selection as a dictionary (see
        [`SelectionGizmo._find_selected_objects`][octarine.selection.SelectionGizmo._find_selected_objects]
        for details). Can also be added post-hoc using `SelectionGizmo.add_callback().
    ignore_invisible :  bool
        Whether to ignore invisible objects. Default is True.
    **kwargs
        Keyword arguments passed to BaseSelectionGizmo.

    """

    def __init__(self, viewer, callback=None, ignore_invisible=True, **kwargs):
        self.callbacks = []
        if not isinstance(callback, type(None)):
            self.add_callback(callback)

        if "callback_after" in kwargs:
            raise ValueError(
                "The `callback_after` keyword argument is reserved for internal use!"
            )
        kwargs["callback_after"] = self.handle_post_selection
        self.gizmo = BaseSelectionGizmo(viewer=viewer, **kwargs)
        self.ignore_invisible = ignore_invisible

    def add_callback(self, callback):
        if not callable(callback):
            raise ValueError("`callback` must be a single callable")
        self.callbacks.append(callback)

    def handle_post_selection(self, gizmo):
        """Handle things after selection."""
        # If no callback is set, we're done here
        if not self.callbacks and not self.gizmo._debug:
            return

        # Find the selected objects
        selection = self.find_selected_objects()
        if self.gizmo._debug:
            n_clipped = len([k for k in selection if selection[k]["clipped"]])
            n_contained = len([k for k in selection if selection[k]["contained"]])
            print(f"Selected objects: {n_clipped} clipped, {n_contained} contained:")
            print(selection)

        # Run the callbacks
        for func in self.callbacks:
            func(selection)

    def find_selected_objects(self):
        """Find objects that are selected by the selection box.

        Notes
        -----
        - currently only Meshes and Points are supported; other objects are returns as `None`

        Returns
        -------
        dict
                Returns a dictionary matching `Viewer.objects`, i.e. keyed by object groups:

                    {
                        "group_name": {
                            "clipped": bool,  # whether the object(s) is at least clipped
                            "contained": bool,  # whether the object(s) is fully contained
                            "objects": [  # an entry for each object in the object group
                                {
                                "clipped": bool,
                                "contained": bool,
                                "data": np.ndarray | None (if not clipped or fully contained)
                                }, ...
                            ]
                        }
                    }

        """
        # If no selection box, return right away
        if not self.gizmo._sel_info:
            return {}

        # Next, we need to iterate over all objects, (1) project their data to screen space
        # and (2) determine whether they are inside the selection box
        selection = {}
        for name in (
            self.gizmo._viewer.visible
            if self.ignore_invisible
            else self.gizmo._viewer.objects
        ):
            selection[name] = {"clipped": False, "contained": True, "objects": []}
            objects = self.gizmo._viewer.objects[name]
            for ob in objects:
                # Extract the data depending on the type of object
                if not isinstance(ob, (gfx.Mesh, gfx.Points, gfx.Line)):
                    # Note to self: we could use object boundaries where no data is available
                    if self.gizmo._debug:
                        print(f"Object {ob} not supported")
                    selection[name]["contained"] = None
                    selection[name]["objects"].append(
                        {"clipped": None, "contained": None, "data": None}
                    )
                    continue

                # Get the data representing the object
                data = ob.geometry.positions.data

                # Apply world transform
                data = la.vec_transform(data, ob.world.matrix)

                # Project the data to screen space
                # (N.B. were using the viewer's normal camera, not the overlay's NDC camera)
                data_screen = la.vec_transform(
                    data, self.gizmo._viewer.camera.camera_matrix
                )

                # Check whether the object is in the selection box
                mask = self.is_data_in_selection_box(data_screen)

                # Store the results
                is_clipped = bool(np.any(mask))  # avoid getting np.True_/np.False_

                if isinstance(ob, gfx.Line):
                    # Lines can have breaks where the data will be `nan` - these will
                    # always count as not inside the selection box. We need to ignore
                    # these when checking for full containment.
                    data_not_nan = np.all(~np.isnan(data), axis=1)
                    is_contained = bool(np.all(mask[data_not_nan]))
                else:
                    is_contained = bool(np.all(mask))

                # If the object is either fully contained or fully outside, we don't
                # really need to pass the mask
                if is_contained or not is_clipped:
                    mask = None

                selection[name]["objects"].append(
                    {"clipped": is_clipped, "contained": is_contained, "data": mask}
                )
                # Propagate results to parent group
                if not is_contained:
                    selection[name]["contained"] = False
                if is_clipped:
                    selection[name]["clipped"] = True

        return selection

    def is_data_in_selection_box(self, data_screen):
        """Check whether selected obejct is inside selection box.

        Parameters
        ----------
        data_screen : (N, 3) array
            The data in screen coordinates.

        Returns
        -------
        bool
            Whether the object is inside the selection box.

        """
        if not self.gizmo._sel_info:
            return np.zeros(data_screen.shape[0], dtype=bool)

        # Check whether the object is in the selection box
        return np.all(
            (data_screen[:, [0, 1]] >= self.gizmo.bounds[0, :-1])
            & (data_screen[:, [0, 1]] <= self.gizmo.bounds[1, :-1]),
            axis=1,
        )
