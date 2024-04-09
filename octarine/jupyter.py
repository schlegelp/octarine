import pygfx as gfx

from ipywidgets import VBox
from IPython.display import display

from ipywidgets.widgets import (
    SelectMultiple,
    FloatSlider,
    ColorPicker,
    Checkbox,
    HBox,
    Tab,
    Dropdown,
    Layout,
    Button,
)

try:
    from sidecar import Sidecar
except ImportError:
    Sidecar = None


class JupyterOutput(HBox):
    """Wrap Viewer (and potentially a toolbar) in Vbox for display."""

    def __init__(self, viewer, toolbar: bool, use_sidecar: bool, sidecar_kwargs: dict):
        """

        Parameters
        ----------
        viewer  :       Viewer
                        The Viewer instance to display.
        toolbar :       bool
                        Whether to display a toolbar. If False, the toolbar
                        will still be created but will hidden.
        use_sidecar :   bool
                        Whether to use Sidecar for display. Will throw an
                        error if Sidecar is not installed.
        sidecar_kwargs: dict
                        optional kwargs passed to Sidecar

        """
        self.viewer = viewer
        self.sidecar = None

        self.use_sidecar = use_sidecar

        if self.use_sidecar and Sidecar is None:
            raise ImportError(
                "Sidecar extention is not installed: `pip install sidecar`\n"
                "Note that you will have to restart Jupyter and deep-reload "
                "the page after installation."
            )

        self.toolbar = JupyterToolbar(viewer)
        self.output = (
            viewer.canvas,
            self.toolbar,
        )

        if not toolbar:  # just stack canvas in VBox
            self.toolbar.hide()

        if use_sidecar:  # instantiate sidecar if requested
            self.sidecar = Sidecar(**sidecar_kwargs)

        # stack all objects in the HBox
        super().__init__(self.output)

    def _repr_mimebundle_(self, *args, **kwargs):
        """
        This is what jupyter hook into when this output context instance is returned at the end of a cell.
        """
        if self.use_sidecar:
            with self.sidecar:
                return display(VBox(self.output))
        else:
            # just display VBox contents in cell output
            return super()._repr_mimebundle_(*args, **kwargs)

    def close(self, close_viewer=True):
        """Closes the output context, cleanup all the stuff"""
        self._is_closed = True  #  we need this to avoid recursion in close

        if close_viewer:
            self.viewer.close()

        if self.toolbar is not None:
            self.toolbar.close()

        if self.sidecar is not None:
            self.sidecar.close()

        super().close()  # ipywidget VBox cleanup


class JupyterToolbar(VBox):
    """Basic toolbar using ipywidgets"""

    def __init__(self, viewer, width="150px"):
        self.viewer = viewer

        self.legend = SelectMultiple(
            options=[*"abcdefghijklmonp"],
            value=[],
            # rows=10,
            description="",
            disabled=False,
            layout=Layout(
                width="auto", height="200px"
            ),  # make sure this doesn't take too much space
        )
        self.update_legend()  # update legend entries
        self.color_picker = ColorPicker(
            concise=False,
            description="",
            value="white",
            disabled=False,
            layout=Layout(width="auto"),
            tooltip="Set color for selected objects",
        )
        self.actions_dropdown = Dropdown(
            options=["Action", "Remove selected", "Hide selected", "Unhide all"],
            value="Action",
            description="",
            disabled=False,
            layout=Layout(width="auto"),
            tooltip="Perform action on selected objects",
        )
        self.center_camera_button = Button(
            description="Center Camera",
            value=False,
            disabled=False,
            icon="align-center",
            layout=Layout(width="auto"),
            tooltip="Auto-center camera",
        )
        self.screenshot_button = Button(
            description="Screenshot",
            value=False,
            disabled=False,
            icon="camera",
            layout=Layout(width="auto"),
            tooltip="Take a screenshot",
        )
        self.flat_shading_toggle = Checkbox(
            value=False,
            description="Flat shading",
            disabled=False,
            indent=False,
            layout=Layout(width="auto"),
        )
        self.wireframe_toggle = Checkbox(
            value=False,
            description="Wireframe",
            disabled=False,
            indent=False,
            layout=Layout(width="auto"),
        )
        self.ambient_light_toggle = Checkbox(
            value=True,
            description="Ambient Light",
            disabled=False,
            indent=False,
            layout=Layout(width="auto"),
        )
        self.ambient_light_slider = FloatSlider(
            value=0.5,
            min=0,
            max=5,
            step=0.1,
            description="",
            disabled=False,
            continuous_update=True,
            orientation="horizontal",
            readout=True,
            readout_format=".1f",
            layout=Layout(width="auto"),
        )
        self.bounds_toggle = Checkbox(
            value=False,
            description="Show Bounds",
            disabled=False,
            indent=False,
            layout=Layout(width="auto"),
        )

        # self._panzoom_controller_button.observe(self.panzoom_handler, "value")
        self.center_camera_button.on_click(self.center_camera)
        self.screenshot_button.on_click(self.screenshot)
        self.legend.observe(self.object_selected, names="value")
        self.color_picker.observe(self.change_color, names="value")
        self.actions_dropdown.observe(self.action_selected, names="value")
        self.flat_shading_toggle.observe(self.set_shading, names="value")
        self.wireframe_toggle.observe(self.set_wireframe, names="value")
        self.ambient_light_toggle.observe(self.toggle_ambient_light, names="value")
        self.ambient_light_slider.observe(self.set_ambient_light, names="value")
        self.bounds_toggle.observe(self.toggle_bounds, names="value")

        # Split into object and scene widgets
        object_widgets = [
            self.legend,
            self.actions_dropdown,
            self.color_picker,
            self.center_camera_button,
            self.screenshot_button,
        ]
        scene_widgets = [
            self.flat_shading_toggle,
            self.wireframe_toggle,
            self.ambient_light_toggle,
            self.ambient_light_slider,
            self.bounds_toggle,
        ]

        # Combine object and scene widgets in tabs
        self.tabs = Tab()
        self.tabs.children = [
            VBox(object_widgets, layout=Layout(width="auto")),
            VBox(scene_widgets, layout=Layout(width="auto")),
        ]
        self.tabs.titles = ["Objects", "Scene"]

        super().__init__([self.tabs], layout=Layout(width=width))

    def hide(self):
        """Hide the toolbar."""
        self.layout.display = "none"

    def show(self):
        """Show the toolbar."""
        self.layout.display = "block"

    def toggle(self):
        """Toggle visibility of the toolbar."""
        if self.layout.display == "none":
            self.show()
        else:
            self.hide()

    def center_camera(self, obj):
        """Center camera on scene."""
        self.viewer.center_camera()

    def screenshot(self, obj):
        pass

    def update_legend(self):
        """Update entries in the legend widget."""
        self.legend.options = list(self.viewer.objects.keys())

    def object_selected(self, change):
        """This is triggered when an object is selected in the legend widget."""
        # Change is a tuple of (item1, item2, ..) from the legend widget
        if len(change["new"]) == 1:
            # Get this object
            ob = self.viewer[change["new"][0]][0]
            # Get the color and assign to the color picker widget
            self.color_picker.value = ob.material.color.hex

    def action_selected(self, change):
        """Perform an action on selected objects."""
        # Reset the button value to "Action"
        self.actions_dropdown.value = "Action"

        sel = self.legend.value
        if len(sel):
            if change["new"] == "Remove selected":
                self.viewer.remove_objects(sel)
            elif change["new"] == "Hide selected":
                self.viewer.hide_objects(sel)
            elif change["new"] == "Unhide all":
                self.viewer.unhide_objects()

    def change_color(self, change):
        """This is triggered when the color picker is changed."""
        self.viewer.set_colors({ob: change["new"] for ob in self.legend.value})

    def set_shading(self, change):
        """Set flat shading for all meshes."""
        for ob in self.viewer.scene.children:
            if isinstance(ob, gfx.Mesh):
                ob.material.flat_shading = change["new"]

    def set_wireframe(self, change):
        """Set wireframe mode for all meshes."""
        for ob in self.viewer.scene.children:
            if isinstance(ob, gfx.Mesh):
                ob.material.wireframe = change["new"]

    def set_ambient_light(self, change):
        """Set intensity of ambient lights."""
        for ob in self.viewer.scene.children:
            if isinstance(ob, gfx.AmbientLight):
                ob.intensity = change["new"]

    def toggle_ambient_light(self, change):
        """Toggle visibility of ambient lights."""
        for ob in self.viewer.scene.children:
            if isinstance(ob, gfx.AmbientLight):
                ob.visible = change["new"]

    def toggle_bounds(self, change):
        """Toggle visibility of scene bounds."""
        self.viewer.show_bounds = change["new"]
