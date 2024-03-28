import numpy as np
import pygfx as gfx

try:
    from PySide6 import QtWidgets, QtCore
except ImportError:
    raise ImportError("Showing controls requires PySide6. Please install it via:\n `pip install PySide6`.")

# TODOs:
# - add custom legend formatting (e.g. "{object.name}")
# - show type of object in legend
# - add dropdown to manipulate all selected objects
# - add filter to legend (use item.setHidden(True/False) to show/hide items)
# - highlight object in legend when hovered over in scene
# - make legend tabbed (QTabWidget)

class Controls(QtWidgets.QWidget):
    def __init__(self, viewer, width=200, height=400):
        super().__init__()
        self.viewer = viewer
        self.setWindowTitle("Legend")
        self.resize(width, height)

        self.btn_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.btn_layout)

        # Build legend
        self.build_gui()

        # Populate legend
        self.update_legend()

        # This determines the target for color changes
        self.active_objects = None

        self.color_picker = QtWidgets.QColorDialog(parent=self)
        self.color_picker.currentColorChanged.connect(self.set_color)
        self.color_picker.colorSelected.connect(self.reset_active_objects)

    def build_gui(self):
        """Build the GUI."""
        # Add legend (i.e. a list widget)
        self.legend = self.create_legend()

        # Add the dropdown to action all selected objects
        self.sel_action = QtWidgets.QPushButton(text="Action")
        self.sel_action_menu = QtWidgets.QMenu(self)
        self.sel_action_menu.addAction("Hide")
        self.sel_action_menu.actions()[-1].triggered.connect(self.hide_selected)
        self.sel_action_menu.addAction("Show")
        self.sel_action_menu.actions()[-1].triggered.connect(self.show_selected)
        self.sel_action_menu.addAction("Delete")
        self.sel_action_menu.actions()[-1].triggered.connect(self.delete_selected)
        self.sel_action_menu.addAction("Color")
        self.sel_action_menu.actions()[-1].triggered.connect(self.color_selected)
        self.sel_action.setMenu(self.sel_action_menu)
        self.btn_layout.addWidget(self.sel_action)

        # Add horizontal divider
        self.add_split()

        # First: add button to toggle flat shading
        self.mesh_flat_checkbox = self.create_checkbox(
            "Flat Shading", gfx.Mesh, "material.flat_shading"
        )

        # Second: add button to toggle wireframe
        self.mesh_wireframe_checkbox = QtWidgets.QCheckBox("Wireframe")
        self.mesh_wireframe_checkbox.setChecked(False)

        def set_wireframe(*args):
            for vis in self.viewer.scene.children:
                if isinstance(vis, gfx.Mesh):
                    vis.material.wireframe = self.mesh_wireframe_checkbox.isChecked()

        self.mesh_wireframe_checkbox.toggled.connect(set_wireframe)
        self.btn_layout.addWidget(self.mesh_wireframe_checkbox)

        # Add horizontal divider
        self.add_split()

        # Third: add controls to adjust ambient light
        self.ambient_light_checkbox = QtWidgets.QCheckBox("Ambient Light")
        self.ambient_light_checkbox.setChecked(True)

        def toggle_ambient_light(*args):
            for vis in self.viewer.scene.children:
                if isinstance(vis, gfx.AmbientLight):
                    vis.visible = self.ambient_light_checkbox.isChecked()

        self.ambient_light_checkbox.toggled.connect(toggle_ambient_light)
        self.btn_layout.addWidget(self.ambient_light_checkbox)

        self.ambient_light_slider = self.create_slider(
                    "Intensity", 0, 10, gfx.AmbientLight, "intensity", step=0.01
                )

        # Add horizontal divider
        # self.add_split()

        # This would make it so the legend does not stretch when
        # we resize the window vertically
        # self.btn_layout.addStretch(1)

        return

    def add_split(self):
        """Add horizontal divider."""
        # self.btn_layout.addSpacing(5)
        self.btn_layout.addWidget(QHLine())
        # self.btn_layout.addSpacing(5)

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
            self.btn_layout.insertWidget(index, list_widget)
        else:
            self.btn_layout.addWidget(list_widget)

        return list_widget

    def make_legend_entry(self, name, color=None):
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
        item_widget.setObjectName(name)
        item = QtWidgets.QListWidgetItem()
        item._id = name  # this helps to identify the item

        # Generate the label
        line_text = QtWidgets.QLabel(f"{name}")
        line_text.setToolTip("Click to select")

        # Generate the checkbox
        line_checkbox = QtWidgets.QCheckBox()
        line_checkbox.setObjectName(name)  # this helps to identify the checkbox
        line_checkbox.setMaximumWidth(40)
        line_checkbox.setToolTip("Toggle visibility")
        line_checkbox.setChecked(True)

        def set_property(*args):
            for vis in self.viewer.objects.get(name, []):
                # Navigate to the correct property
                vis.visible = line_checkbox.isChecked()

        line_checkbox.toggled.connect(set_property)

        # Generate the button
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
                color = gfx.Color('k')
            # Find the button in this widget
            item_widget = self.legend.itemWidget(item)
            line_push_button = item_widget.findChild(QtWidgets.QPushButton, item._id)
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
                    color = 'k'
                item, item_widget = self.make_legend_entry(obj, color=color)
                self.legend.addItem(item)
                self.legend.setItemWidget(item, item_widget)

    def color_button_clicked(self):
        """Set the active object to be the buttons target."""
        sender = self.sender()
        push_button = self.findChild(QtWidgets.QPushButton, sender.objectName())
        # print(f'click: {push_button.objectName()}')
        self.active_objects = push_button.objectName()
        self.color_picker.show()

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

    def hide_selected(self):
        """Hide selected objects."""
        sel = self.get_selected()
        if sel:
            self.viewer.hide_objects(self.get_selected())

    def show_selected(self):
        """Show selected objects."""
        sel = self.get_selected()
        if sel:
            self.viewer.unhide_objects(self.get_selected())

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
        color_btn.setObjectName(name)

        # Set tooltip
        color_btn.setToolTip("Click to change color")

        # Set color (will be updated subsequently via controls.update_legend())
        if color is None:
            color = 'w'
        color = gfx.Color(color)
        color_btn.setStyleSheet(f"background-color: {color.css}")

        # Connect callback (this just sets the active object)
        color_btn.clicked.connect(self.color_button_clicked)

        return color_btn

    def create_checkbox(self, name, targets=None, property=None, callback=None, toggle=[], index=None, default_value=False):
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
            self.btn_layout.insertWidget(index, checkbox)
        else:
            self.btn_layout.addWidget(checkbox)
        return checkbox

    def create_slider(self, name, min, max, targets, property, step=1, callback=None):
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

        self.btn_layout.addLayout(layout)
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