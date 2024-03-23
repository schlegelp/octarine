import pygfx as gfx

try:
    from PySide6 import QtWidgets, QtGui, QtCore
except ImportError:
    raise ImportError("Showing controls requires PySide6. Please install it via:\n `pip install PySide6`.")

# TODOs:
# - add custom legend formatting (e.g. "{object.name}")
# - show type of object in legend
# - add dropdown to manipulate all selected objects
# - add filter to legend (use item.setHidden(True/False) to show/hide items)
# - highlight object in legend when hovered over in scene

class Controls(QtWidgets.QWidget):
    def __init__(self, viewer, width=200, height=400):
        super().__init__()
        self.viewer = viewer
        self.setWindowTitle("Legend")
        self.resize(width, height)

        self.btn_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.btn_layout)

        self.build_gui()

    def build_gui(self):
        """Build the GUI."""

        # Add legend
        self.legend = self.create_legend()

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

        self.add_split()

        self.btn_layout.addStretch(1)

        return

    def add_split(self):
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

        # Add some example items
        for i, c in enumerate(["red", "green", "blue"]):
            item, item_widget = self.make_legend_entry(f"Item {i}", color=c)
            list_widget.addItem(item)
            list_widget.setItemWidget(item, item_widget)

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
        item = QtWidgets.QListWidgetItem()
        item._id = name  # this helps to identify the item

        # Generate a button
        line_text = QtWidgets.QLabel(f"{name}")
        line_push_button = QtWidgets.QPushButton()
        line_push_button.setMaximumWidth(20)
        line_push_button.setMaximumHeight(20)
        line_push_button.setObjectName(name)  # this helps to identify the button
        line_push_button.clicked.connect(self._clicked)  # connect button to function

        if color is not None:
            color = gfx.Color(color).css
            line_push_button.setStyleSheet(f"background-color: {color}")

        # Generate item layout
        item_layout = QtWidgets.QHBoxLayout()
        item_layout.setContentsMargins(0, 0, 0, 0)  # make layout tight
        item_layout.setSpacing(0)
        # Add text and button to layout
        item_layout.addWidget(line_text)
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

    def _clicked(self):
        sender = self.sender()
        push_button = self.findChild(QtWidgets.QPushButton, sender.objectName())
        # print(f'click: {push_button.objectName()}')

    def create_color_btn(self, name, target, property, callback=None):
        layout = QtWidgets.QHBoxLayout()

        layout.addWidget(QtWidgets.QLabel(name))

        color_btn = QtWidgets.QPushButton()
        color_btn.setStyleSheet("background-color: %s" % getattr(target, property).hex)

        def set_color():
            color = QtWidgets.QColorDialog.getColor(
                QtGui.QColor(getattr(target, property).hex)
            )
            if color.isValid():
                color_btn.setStyleSheet("background-color: %s" % color.name())
                setattr(target, property, color.name())
                if callback:
                    callback(color.name())

        color_btn.clicked.connect(set_color)

        layout.addWidget(color_btn)
        self.btn_layout.addLayout(layout)
        return color_btn

    def create_checkbox(
        self, name, targets=None, property=None, callback=None, toggle=[], index=None, default_value=False
    ):
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


    #     self.hello = ["Hallo Welt", "Hei maailma", "Hola Mundo", "Привет мир"]

    #     self.button = QtWidgets.QPushButton("Click me!")
    #     self.text = QtWidgets.QLabel("Hello World",
    #                                  alignment=QtCore.Qt.AlignCenter)

    #     self.layout = QtWidgets.QVBoxLayout(self)
    #     self.layout.addWidget(self.text)
    #     self.layout.addWidget(self.button)

    #     self.button.clicked.connect(self.magic)

    # @QtCore.Slot()
    # def magic(self):
    #     self.text.setText(random.choice(self.hello))


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