import numpy as np
import pygfx as gfx

try:
    from PySide6 import QtWidgets, QtCore
except ImportError:
    raise ImportError(
        "Showing controls requires PySide6. Please install it via:\n `pip install PySide6`."
    )

# TODOs:
# - add custom legend formatting (e.g. "{object.name}")
# - show type of object in legend
# - add dropdown to manipulate all selected objects
# - add filter to legend (use item.setHidden(True/False) to show/hide items)
# - highlight object in legend when hovered over in scene
# - make legend tabbed (QTabWidget)


def set_viewer_stale(func):
    """Decorator to set the viewer stale after a function call."""

    def wrapper(self, *args, **kwargs):
        self.viewer._render_stale = True
        return func(self, *args, **kwargs)

    return wrapper


class Controls(QtWidgets.QWidget):
    def __init__(self, viewer, width=300, height=400):
        super().__init__()
        self.viewer = viewer
        self.setWindowTitle("Controls")
        self.resize(width, height)

        self.tab_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.tab_layout)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QtWidgets.QTabWidget.West)
        self.tabs.setMovable(True)

        self.tab_layout.addWidget(self.tabs)

        self.tab1 = QtWidgets.QWidget()
        self.tab2 = QtWidgets.QWidget()
        self.tab1_layout = QtWidgets.QVBoxLayout()
        self.tab2_layout = QtWidgets.QVBoxLayout()
        self.tab1.setLayout(self.tab1_layout)
        self.tab2.setLayout(self.tab2_layout)

        self.tabs.addTab(self.tab1, "Legend")
        self.tabs.addTab(self.tab2, "Controls")

        # self.btn_layout = QtWidgets.QVBoxLayout()
        # self.setLayout(self.btn_layout)

        # # Build legend
        # self.build_gui()

        # Build gui
        self.build_legend_gui()
        self.build_controls_gui()

        # Populate legend
        self.update_legend()

        # This determines the target for color changes
        self.active_objects = None
        self.active_volume = None

        self.color_picker = QtWidgets.QColorDialog(parent=self)
        self.color_picker.setOption(QtWidgets.QColorDialog.ShowAlphaChannel, on=True)
        self.color_picker.currentColorChanged.connect(self.set_color)
        self.color_picker.colorSelected.connect(self.reset_active_objects)

    def build_legend_gui(self):
        """Build the legend GUI."""
        # Add legend (i.e. a list widget)
        self.legend = self.create_legend()

        # Add the dropdown to action all selected objects
        self.sel_action = QtWidgets.QPushButton(text="Action")
        self.sel_action_menu = QtWidgets.QMenu(self)
        self.sel_action_menu.addAction("Invert Visibility")
        self.sel_action_menu.actions()[-1].triggered.connect(self.invert_visibility)
        self.sel_action_menu.addAction("Hide All")
        self.sel_action_menu.actions()[-1].triggered.connect(self.hide_all)
        self.sel_action_menu.addAction("Show All")
        self.sel_action_menu.actions()[-1].triggered.connect(self.show_all)
        self.sel_action_menu.addAction("Delete All")
        self.sel_action_menu.actions()[-1].triggered.connect(self.delete_all)
        self.sel_action_menu.addAction("Select All")
        self.sel_action_menu.actions()[-1].triggered.connect(self.select_all)
        self.sel_action_menu.addAction("Clear Selection")
        self.sel_action_menu.actions()[-1].triggered.connect(self.select_none)
        self.sel_action_menu.addAction("Invert Selection")
        self.sel_action_menu.actions()[-1].triggered.connect(self.invert_selection)
        self.sel_action_menu.addAction("Hide selected")
        self.sel_action_menu.actions()[-1].triggered.connect(self.hide_selected)
        self.sel_action_menu.addAction("Show selected")
        self.sel_action_menu.actions()[-1].triggered.connect(self.show_selected)
        self.sel_action_menu.addAction("Delete selected")
        self.sel_action_menu.actions()[-1].triggered.connect(self.delete_selected)
        self.sel_action_menu.addAction("Color selected")
        self.sel_action_menu.actions()[-1].triggered.connect(self.color_selected)
        self.sel_action.setMenu(self.sel_action_menu)
        self.tab1_layout.addWidget(self.sel_action)

    def build_controls_gui(self):
        """Build the legend GUI."""

        # Add dropdown to determine what's happening on hover
        self.on_hover_label = QtWidgets.QLabel("On Hover:")
        self.tab2_layout.addWidget(self.on_hover_label)
        self.on_hover_dropdown = QtWidgets.QComboBox()
        self.on_hover_dropdown.addItems(["Nothing", "Highlight"])
        self.on_hover_dropdown.setToolTip(
            "Action to perform when hovering over an object."
        )
        self.on_hover_dropdown.currentIndexChanged.connect(
            lambda x: setattr(
                self.viewer,
                "on_hover",
                self.on_hover_dropdown.currentText().lower()
                if self.on_hover_dropdown.currentText() != "Nothing"
                else None,
            )
        )
        self.tab2_layout.addWidget(self.on_hover_dropdown)

        # Add dropdown to determine what's happening on double click
        self.on_dclick_label = QtWidgets.QLabel("On Double Click:")
        self.tab2_layout.addWidget(self.on_dclick_label)
        self.on_dclick_dropdown = QtWidgets.QComboBox()
        self.on_dclick_dropdown.addItems(["Nothing", "Hide", "Remove", "Select"])
        self.on_dclick_dropdown.setToolTip(
            "Action to perform when double clicking on an object."
        )
        self.on_dclick_dropdown.currentIndexChanged.connect(
            lambda x: setattr(
                self.viewer,
                "on_double_click",
                self.on_dclick_dropdown.currentText().lower()
                if self.on_dclick_dropdown.currentText() != "Nothing"
                else None,
            )
        )
        self.tab2_layout.addWidget(self.on_dclick_dropdown)

        # Horizontal divider
        self.tab2_layout.addWidget(QHLine())

        # Add button to toggle wireframe
        self.mesh_wireframe_checkbox = QtWidgets.QCheckBox("Wireframe")
        self.mesh_wireframe_checkbox.setChecked(False)

        # Add button to toggle flat shading
        self.mesh_flat_checkbox = self.create_checkbox(
            "Flat Shading", self.tab2_layout, gfx.Mesh, "material.flat_shading"
        )

        # Add button to toggle wireframe
        self.mesh_wireframe_checkbox = QtWidgets.QCheckBox("Wireframe")
        self.mesh_wireframe_checkbox.setChecked(False)

        def set_wireframe(*args):
            for vis in self.viewer.scene.children:
                if isinstance(vis, gfx.Mesh):
                    vis.material.wireframe = self.mesh_wireframe_checkbox.isChecked()

        self.mesh_wireframe_checkbox.toggled.connect(set_wireframe)
        self.tab2_layout.addWidget(self.mesh_wireframe_checkbox)

        # Add button to toggle FPS
        self.fps_checkbox = QtWidgets.QCheckBox("Show FPS")
        self.fps_checkbox.setChecked(self.viewer._show_fps)

        self.fps_checkbox.toggled.connect(
            lambda x: setattr(
                self.viewer,
                "show_fps",
                self.fps_checkbox.isChecked(),
            )
        )
        self.tab2_layout.addWidget(self.fps_checkbox)

        # Horizontal divider
        self.tab2_layout.addWidget(QHLine())

        # Add controls to adjust ambient light
        self.ambient_light_checkbox = QtWidgets.QCheckBox("Ambient Light")
        self.ambient_light_checkbox.setChecked(True)

        def toggle_ambient_light(*args):
            for vis in self.viewer.scene.children:
                if isinstance(vis, gfx.AmbientLight):
                    vis.visible = self.ambient_light_checkbox.isChecked()

        self.ambient_light_checkbox.toggled.connect(toggle_ambient_light)
        self.tab2_layout.addWidget(self.ambient_light_checkbox)

        self.ambient_light_slider = self.create_slider(
            "Intensity",
            0,
            10,
            gfx.AmbientLight,
            "intensity",
            step=0.01,
            parent_layout=self.tab2_layout,
        )

        # Horizontal divider
        self.tab2_layout.addWidget(QHLine())

        # Add dropdown to determine render mode
        self.render_mode_label = QtWidgets.QLabel("Render trigger:")
        self.tab2_layout.addWidget(self.render_mode_label)
        self.render_mode_dropdown = QtWidgets.QComboBox()
        self.render_mode_dropdown.setToolTip(
            "Set trigger for re-rendering the scene. See documentation for details."
        )
        self.render_mode_dropdown.addItems(["Continuous", "Reactive", "Active Window"])
        self.render_mode_dropdown.setItemData(
            0, "Continuously render the scene.", QtCore.Qt.ToolTipRole
        )
        self.render_mode_dropdown.setItemData(
            1,
            "Render only when the scene changes.",
            QtCore.Qt.ToolTipRole,
        )
        self.render_mode_dropdown.setItemData(
            2, "Render only when the window is active.", QtCore.Qt.ToolTipRole
        )
        render_trigger_vals = ["continuous", "reactive", "active_window"]
        self.render_mode_dropdown.currentIndexChanged.connect(
            lambda x: setattr(
                self.viewer,
                "render_trigger",
                render_trigger_vals[self.render_mode_dropdown.currentIndex()],
            )
        )
        # Set default item to whatever the currently set render trigger is
        self.render_mode_dropdown.setCurrentIndex(
            render_trigger_vals.index(self.viewer.render_trigger)
        )
        self.tab2_layout.addWidget(self.render_mode_dropdown)

        # This would make it so the legend does not stretch when
        # we resize the window vertically
        self.tab2_layout.addStretch(1)

    def create_legend(self, spacing=0, index=None):
        """Generate the legend widget."""
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Legend"))
        list_widget = QtWidgets.QListWidget()
        list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        layout.addWidget(list_widget)
        list_widget.setSpacing(spacing)

        # Add some example items (for debugging only)
        # for i, c in enumerate(["red", "green", "blue"]):
        #     item, item_widget = self.make_legend_entry(f"Item {i}", color=c)
        #     list_widget.addItem(item)
        #     list_widget.setItemWidget(item, item_widget)

        if index is not None:
            self.tab1_layout.insertWidget(index, list_widget)
        else:
            self.tab1_layout.addWidget(list_widget)

        return list_widget

    def make_legend_entry(self, name, color=None, type=None):
        """Generate a legend entry.

        Parameters
        ----------
        name :      str
                    Name of the entry.
        color :     str | tuple | array
                    Color of the entry.

        Returns
        -------
        item :      QtWidgets.QListWidgetItem
                    List item.
        item_widget : QtWidgets.QWidget
                    List item widget.

        """
        # Initialize widget and item
        item_widget = QtWidgets.QWidget()
        item_widget.setObjectName(str(name))
        item = QtWidgets.QListWidgetItem()
        item._id = name  # this helps to identify the item

        # Generate the label
        line_text = QtWidgets.QLabel(f"{name}")
        line_text.setToolTip("Click to select")

        # Generate the checkbox
        line_checkbox = QtWidgets.QCheckBox()
        line_checkbox.setObjectName(str(name))  # this helps to identify the checkbox
        line_checkbox.setMaximumWidth(40)
        line_checkbox.setToolTip("Toggle visibility")
        line_checkbox.setChecked(True)

        def set_property(*args):
            for vis in self.viewer.objects.get(name, []):
                # Navigate to the correct property
                vis.visible = line_checkbox.isChecked()

        line_checkbox.toggled.connect(set_property)

        # Generate the button
        if type == gfx.Volume:
            line_push_button = self.create_volume_btn(name, callback=None)
        else:
            line_push_button = self.create_color_btn(name, color=color, callback=None)

        # Generate item layout
        item_layout = QtWidgets.QHBoxLayout()
        item_layout.setContentsMargins(0, 0, 0, 0)  # make layout tight
        item_layout.setSpacing(0)

        # Add text and button to layout
        item_layout.addWidget(line_text)
        item_layout.addWidget(line_checkbox)
        item_layout.addWidget(line_push_button)

        # Set layout
        item_widget.setLayout(item_layout)
        item.setSizeHint(item_widget.sizeHint())

        return item, item_widget

    def update_legend(self):
        """Update legend with objects in current scene."""
        # Get objects in scene
        objects = self.viewer.objects

        # Go over existing items
        N_items = self.legend.count()
        present = []
        for i in list(range(N_items))[::-1]:
            # Get this item
            item = self.legend.item(i)

            # Clear item if not present anymore
            if item._id not in objects:
                self.legend.takeItem(i)
                continue
            else:
                present.append(item._id)

            # Update color
            try:
                color = objects[item._id][0].material.color
            except BaseException:
                color = gfx.Color("k")
            # Find the button in this widget
            item_widget = self.legend.itemWidget(item)
            line_push_button = item_widget.findChild(QtWidgets.QPushButton)
            # Update color
            if line_push_button:
                line_push_button.setStyleSheet(f"background-color: {color.css}")

        # Add new items
        for obj in objects:
            if obj not in present:
                try:
                    color = objects[obj][0].material.color
                except BaseException:
                    # Note to self: need to make sure we also cater for color arrays
                    # which are in the geometry object
                    color = "k"
                item, item_widget = self.make_legend_entry(
                    obj, color=color, type=type(objects[obj][0])
                )
                self.legend.addItem(item)
                self.legend.setItemWidget(item, item_widget)

        # # Now check if the visibility of the objects has changed
        visible = self.viewer.visible
        for i in range(self.legend.count()):
            item = self.legend.item(i)
            item_widget = self.legend.itemWidget(item)
            line_checkbox = item_widget.findChild(QtWidgets.QCheckBox)
            if item._id in visible:
                line_checkbox.setChecked(True)
            else:
                line_checkbox.setChecked(False)

    def color_button_clicked(self):
        """Set the active object to be the buttons target."""
        sender = self.sender()
        push_button = self.findChild(QtWidgets.QPushButton, sender.objectName())
        # print(f'click: {push_button.objectName()}')
        self.active_objects = push_button._id
        self.color_picker.show()

    def volume_button_clicked(self):
        """Set the active object to be the buttons target."""
        sender = self.sender()
        push_button = self.findChild(QtWidgets.QPushButton, sender.objectName())
        # print(f'click: {push_button.objectName()}')
        self.active_volume = push_button.objectName()
        self.volume_controls.show()

    @set_viewer_stale
    def set_color(self, color):
        """Color current active object(s). This is the callback for the color picker."""
        if self.active_objects is None:
            return
        elif self.active_objects == "selected":
            targets = self.get_selected()
        elif not isinstance(self.active_objects, list):
            targets = [self.active_objects]

        # Convert QColor to [0-1] RGB
        color = np.array(color.toTuple()) / 255

        self.viewer.set_colors({name: color for name in targets})

    def select_all(self):
        """Select all objects."""
        self.legend.selectAll()

    def select_none(self):
        """Select no objects."""
        self.legend.clearSelection()

    def invert_selection(self):
        """Invert selection."""
        for i in range(self.legend.count()):
            item = self.legend.item(i)
            if item.isSelected():
                self.legend.setItemSelected(item, False)
            else:
                self.legend.setItemSelected(item, True)

    def get_selected(self):
        """Get selected items."""
        sel = []
        for item in self.legend.selectedItems():
            sel.append(item._id)
        return sel

    def color_selected(self):
        """Set the active object to be the selected objects."""
        self.active_objects = "selected"
        self.color_picker.show()

    @set_viewer_stale
    def hide_all(self):
        """Hide all objects."""
        self.viewer.hide_objects(self.viewer.objects)

    @set_viewer_stale
    def hide_selected(self):
        """Hide selected objects."""
        sel = self.get_selected()
        if sel:
            self.viewer.hide_objects(self.get_selected())

    @set_viewer_stale
    def show_all(self):
        """Show all objects."""
        self.viewer.unhide_objects(None)

    @set_viewer_stale
    def show_selected(self):
        """Show selected objects."""
        sel = self.get_selected()
        if sel:
            self.viewer.unhide_objects(self.get_selected())

    @set_viewer_stale
    def invert_visibility(self):
        """Invert visibility of all objects."""
        vis = self.viewer.visible
        invis = self.viewer.invisible
        self.viewer.hide_objects(vis)
        self.viewer.unhide_objects(invis)

    @set_viewer_stale
    def delete_all(self):
        """Delete all objects."""
        self.viewer.remove_objects(self.viewer.objects)

    @set_viewer_stale
    def delete_selected(self):
        """Delete selected objects."""
        sel = self.get_selected()
        if sel:
            self.viewer.remove_objects(self.get_selected())

    def reset_active_objects(self):
        """Reset active objects."""
        self.color_target = None

    def create_color_btn(self, name, color=None, callback=None):
        """Generate a colorize button ."""
        # Generate button
        color_btn = QtWidgets.QPushButton()

        # Make sure it doesn't take up too much space
        color_btn.setMaximumWidth(20)
        color_btn.setMaximumHeight(20)
        color_btn.setObjectName(str(name))
        color_btn._id = name  # this helps to identify the associated object

        # Set tooltip
        color_btn.setToolTip("Click to change color")

        # Set color (will be updated subsequently via controls.update_legend())
        if color is None:
            color = "w"
        color = gfx.Color(color)
        color_btn.setStyleSheet(f"background-color: {color.css}")

        # Connect callback (this just sets the active object)
        color_btn.clicked.connect(self.color_button_clicked)

        return color_btn

    def create_volume_btn(self, name, callback=None, color="w"):
        """Generate a button to bring up the volume control."""
        # Generate button
        volume_btn = QtWidgets.QPushButton()

        # Make sure it doesn't take up too much space
        volume_btn.setMaximumWidth(20)
        volume_btn.setMaximumHeight(20)
        volume_btn.setObjectName(str(name))

        # Set tooltip
        volume_btn.setToolTip("Click to adjust volume")

        # Set color (will be updated subsequently via controls.update_legend())
        if color is None:
            color = "w"
        color = gfx.Color(color)
        volume_btn.setStyleSheet(f"background-color: {color.css}")

        # Connect callback (this just sets the active object)
        volume_btn.clicked.connect(self.volume_button_clicked)

        return volume_btn

    def create_checkbox(
        self,
        name,
        parent_layout,
        targets=None,
        property=None,
        callback=None,
        toggle=[],
        index=None,
        default_value=False,
    ):
        """Create a checkbox to toggle a property."""
        checkbox = QtWidgets.QCheckBox(name)

        checkbox.setChecked(bool(default_value))

        def set_property(*args):
            path = property.split(".")
            for vis in self.viewer.scene.children:
                if isinstance(vis, targets):
                    # Navigate to the correct property
                    for p in path[:-1]:
                        vis = getattr(vis, p)
                    setattr(vis, path[-1], checkbox.isChecked())
            for e in toggle:
                e.setEnabled(checkbox.isChecked())
            if callback:
                callback(checkbox.isChecked())

        checkbox.toggled.connect(set_property)

        # set_property()
        if index is not None:
            parent_layout.insertWidget(index, checkbox)
        else:
            parent_layout.addWidget(checkbox)
        return checkbox

    def create_slider(
        self, name, min, max, targets, property, parent_layout, step=1, callback=None
    ):
        """Generate a slider to adjust a property."""
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel(name))
        slide = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slide.setMinimum(min / step)
        slide.setMaximum(max / step)
        # slide.setSingleStep(step)
        val = 0
        for vis in self.viewer.scene.children:
            if isinstance(vis, targets):
                val = getattr(vis, property)
                break
        slide.setValue(val / step)

        if isinstance(step, float):
            val_label = QtWidgets.QLabel(f"{float(val):3.2f}")
        else:
            val_label = QtWidgets.QLabel(f"{int(val):03d}")

        layout.addWidget(val_label)

        def set_value(value):
            value = value * step
            if isinstance(step, float):
                val_label.setText(f"{float(value):3.2f}")
            else:
                val_label.setText(f"{int(value):03d}")

            for target in self.viewer.scene.children:
                if not isinstance(target, targets):
                    continue
                setattr(target, property, value)
            if callback:
                callback(value)

        slide.valueChanged.connect(set_value)

        layout.addWidget(slide)

        parent_layout.addLayout(layout)
        return slide

    def close(self):
        """Close the controls."""
        # This makes sure to also close the color picker, not just the controls window
        self.color_picker.close()
        super().close()


class QHLine(QtWidgets.QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class QVLine(QtWidgets.QFrame):
    def __init__(self):
        super(QVLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.VLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
