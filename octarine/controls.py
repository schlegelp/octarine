import numpy as np
import pygfx as gfx

from functools import wraps

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


def connect_color_picker(func):
    """Decorator to mark this controls as the active color picker target.

    On some backends (notably macOS) Qt uses a shared native color dialog.
    We therefore route color events through a class-level dispatcher and
    update the active controls instance whenever a color action is triggered.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        """Wrapper to activate this controls for subsequent color events."""
        Controls._active_color_controls = self
        return func(self, *args, **kwargs)

    return wrapper


def set_viewer_stale(func):
    """Decorator to set the viewer stale after a function call."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.viewer._render_stale = True
        return func(self, *args, **kwargs)

    return wrapper


class Controls(QtWidgets.QWidget):
    _shared_color_picker = None
    _active_color_controls = None

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

        self.color_picker = self._get_shared_color_picker()

    @classmethod
    def _get_shared_color_picker(cls):
        """Return the shared QColorDialog instance used by all controls."""
        if cls._shared_color_picker is None:
            picker = QtWidgets.QColorDialog()
            picker.setOption(QtWidgets.QColorDialog.ShowAlphaChannel, on=True)
            picker.currentColorChanged.connect(cls._dispatch_color_changed)
            picker.colorSelected.connect(cls._dispatch_color_selected)
            cls._shared_color_picker = picker
        return cls._shared_color_picker

    @classmethod
    def _dispatch_color_changed(cls, color):
        """Forward color changes to the currently active controls instance."""
        active_controls = cls._active_color_controls
        if active_controls is not None:
            active_controls.set_color(color)

    @classmethod
    def _dispatch_color_selected(cls, *args):
        """Reset active selection for the currently active controls instance."""
        active_controls = cls._active_color_controls
        if active_controls is not None:
            active_controls.reset_active_objects()

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
                    self.viewer._render_stale = True

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
        line_text = QtWidgets.QPushButton(f"{name}", flat=True)
        line_text.setToolTip("Click to select")
        line_text.setProperty("legend_role", "label")
        # Allow long names to shrink first so toggle + color controls stay visible.
        line_text.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred
        )
        line_text.setMinimumWidth(0)

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
                self.viewer._render_stale = True
            self.update_legend()

        line_checkbox.toggled.connect(set_property)

        # Generate the button
        if type == gfx.Volume:
            line_push_button = self.create_volume_btn(name, callback=None)
        else:
            line_push_button = self.create_color_btn(name, color=color, callback=None)
        line_push_button.setProperty("legend_role", "control")

        # Generate item layout
        item_layout = QtWidgets.QHBoxLayout()
        item_layout.setContentsMargins(0, 0, 0, 0)  # make layout tight
        item_layout.setSpacing(0)

        # Add text and button to layout
        item_layout.addWidget(line_text)
        item_layout.addWidget(line_checkbox)
        item_layout.addSpacing(15)
        item_layout.addWidget(line_push_button)
        item_layout.setStretch(0, 1)

        # Set layout
        item_widget.setLayout(item_layout)
        item.setSizeHint(item_widget.sizeHint())

        return item, item_widget

    def make_grouped_legend_entry(self, names, group_name=None):
        """Generate a collapsible legend entry for grouped objects.

        Parameters
        ----------
        names : list[str] | tuple[str]
                    Object names that belong to this group.
        group_name : str | None
                    Label shown in the collapsible header.

        Returns
        -------
        item_widget : QtWidgets.QWidget
                    Collapsible widget with one row per grouped member.
        """
        if not names:
            raise ValueError("`names` must contain at least one object name.")

        if group_name is None:
            group_name = f"group ({len(names)})"

        # Outer container that can be used as the list item widget.
        item_widget = QtWidgets.QWidget()
        item_widget.setObjectName(str(group_name))
        item_layout = QtWidgets.QVBoxLayout(item_widget)
        item_layout.setContentsMargins(0, 0, 0, 0)
        item_layout.setSpacing(0)

        # Header button to toggle child visibility.
        header_row = QtWidgets.QWidget()
        header_row_layout = QtWidgets.QHBoxLayout(header_row)
        header_row_layout.setContentsMargins(0, 0, 0, 0)
        header_row_layout.setSpacing(0)

        header = QtWidgets.QToolButton()
        # Prefix with a thin gap so text does not crowd the arrow indicator.
        header.setText(f"  {group_name}")
        header.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        header.setArrowType(QtCore.Qt.RightArrow)
        header.setCheckable(True)
        header.setChecked(False)
        header.setStyleSheet(
            "QToolButton {"
            " text-align: left;"
            " font-size: 13px;"
            " background: transparent;"
            " border: none;"
            " padding: 0px;"
            " }"
            "QToolButton:checked { background: transparent; border: none; }"
            "QToolButton:pressed { background: transparent; border: none; }"
            "QToolButton:hover { background: transparent; border: none; }"
            "QToolButton::menu-indicator { width: 8px; height: 8px; }"
        )
        header.setAutoRaise(True)
        header.setIconSize(QtCore.QSize(8, 8))
        header.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        header.setToolTip("Click to expand/collapse grouped legend entries")

        group_checkbox = QtWidgets.QCheckBox()
        group_checkbox.setProperty("legend_role", "group_visibility")
        group_checkbox.setTristate(True)
        group_checkbox.setMaximumWidth(40)
        group_checkbox.setToolTip("Toggle visibility for all group members")

        def toggle_group_visibility(*args):
            is_visible = group_checkbox.isChecked()
            for member_name in names:
                for vis in self.viewer.objects.get(member_name, []):
                    vis.visible = is_visible

                # Mirror the group toggle on each member checkbox without
                # retriggering each member's callback.
                member_checkbox = item_widget.findChild(
                    QtWidgets.QCheckBox, str(member_name)
                )
                if member_checkbox is not None:
                    member_checkbox.blockSignals(True)
                    member_checkbox.setChecked(is_visible)
                    member_checkbox.blockSignals(False)
            self.viewer._render_stale = True

        def coerce_group_checkbox_click_state(*args):
            # Partial is an indicator-only state; user clicks should skip it.
            if group_checkbox.checkState() == QtCore.Qt.PartiallyChecked:
                group_checkbox.blockSignals(True)
                group_checkbox.setCheckState(QtCore.Qt.Checked)
                group_checkbox.blockSignals(False)
                toggle_group_visibility()

        group_checkbox.toggled.connect(toggle_group_visibility)
        group_checkbox.clicked.connect(coerce_group_checkbox_click_state)

        group_color = None
        first_name = names[0]
        if first_name in self.viewer.objects and self.viewer.objects[first_name]:
            try:
                group_color = self.viewer.objects[first_name][0].material.color
            except BaseException:
                group_color = "k"

        group_color_button = self.create_color_btn(
            f"group::{group_name}", color=group_color, callback=None
        )
        group_color_button.setProperty("legend_role", "group_control")
        group_color_button.setToolTip("Click to change color for entire group")
        group_color_button._id = list(names)

        header_row_layout.addWidget(header)
        header_row_layout.addWidget(group_checkbox)
        header_row_layout.addSpacing(15)
        header_row_layout.addWidget(group_color_button)
        header_row_layout.setStretch(0, 1)
        item_layout.addWidget(header_row)

        # Child container with one standard legend row per member.
        content_widget = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content_widget)
        content_layout.setContentsMargins(18, 0, 0, 0)
        content_layout.setSpacing(0)
        content_widget.setVisible(False)

        for name in names:
            color = None
            vis_type = None
            if name in self.viewer.objects and self.viewer.objects[name]:
                try:
                    color = self.viewer.objects[name][0].material.color
                except BaseException:
                    color = "k"
                vis_type = type(self.viewer.objects[name][0])

            _, child_widget = self.make_legend_entry(name, color=color, type=vis_type)
            child_widget.setProperty("legend_role", "group_member")

            child_layout = child_widget.layout()
            if child_layout is not None:
                child_layout.setContentsMargins(0, 0, 0, 0)
                child_layout.setSpacing(0)

            # Keep rows compact but let Qt compute a non-clipping height.
            child_widget.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
            )

            # Apply mild compression: reduce excess row padding without clipping text.
            compact_height = max(20, child_widget.sizeHint().height() - 8)
            child_widget.setMinimumHeight(compact_height)
            child_widget.setMaximumHeight(compact_height)
            content_layout.addWidget(child_widget)

        item_layout.addWidget(content_widget)

        def toggle_group(expanded):
            header.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
            content_widget.setVisible(expanded)

            # Keep parent QListWidgetItem height in sync with expanded/collapsed state.
            if hasattr(self, "legend"):
                for i in range(self.legend.count()):
                    legend_item = self.legend.item(i)
                    if self.legend.itemWidget(legend_item) is item_widget:
                        legend_item.setSizeHint(item_widget.sizeHint())
                        break

        header.toggled.connect(toggle_group)

        return item_widget

    def update_legend(self):
        """Update legend with objects in current scene."""
        # Get visuals in scene
        visuals = self.viewer.visuals

        # Collect ungrouped visuals by object id and grouped visuals by group name.
        object_ids = {}
        grouped_ids = {}
        for vis in visuals:
            if not hasattr(vis, "_object_id"):
                continue

            object_id = vis._object_id
            object_group = getattr(vis, "_object_group", None)

            if object_group is None:
                object_ids[object_id] = object_ids.get(object_id, []) + [vis]
                continue

            grouped_ids[object_group] = grouped_ids.get(object_group, {})
            grouped_ids[object_group][object_id] = (
                grouped_ids[object_group].get(object_id, []) + [vis]
            )

        # Build a unified legend spec for add/update/remove bookkeeping.
        legend_entries = {}
        for object_id, obj_visuals in object_ids.items():
            legend_entries[object_id] = dict(
                kind="single",
                members=[object_id],
                visuals=obj_visuals,
            )
        for group_name, member_dict in grouped_ids.items():
            group_key = f"group::{group_name}"
            members = list(member_dict.keys())
            flattened = []
            for member in members:
                flattened.extend(member_dict[member])
            legend_entries[group_key] = dict(
                kind="group",
                members=members,
                member_visuals=member_dict,
                visuals=flattened,
                group_name=group_name,
            )

        # Go over existing items
        N_items = self.legend.count()
        present = []
        for i in list(range(N_items))[::-1]:
            # Get this item
            item = self.legend.item(i)

            # Clear item if not present anymore
            if item._id not in legend_entries:
                self.legend.takeItem(i)
                continue
            else:
                present.append(item._id)

            entry = legend_entries[item._id]
            item_widget = self.legend.itemWidget(item)

            if entry["kind"] == "single":
                try:
                    color = entry["visuals"][0].material.color
                except BaseException:
                    color = gfx.Color("k")

                line_push_button = next(
                    (
                        button
                        for button in item_widget.findChildren(QtWidgets.QPushButton)
                        if button.property("legend_role") == "control"
                    ),
                    None,
                )
                if line_push_button:
                    line_push_button.setStyleSheet(f"background-color: {color.css}")
            else:
                expected_members = {str(member) for member in entry["members"]}
                current_members = {
                    w.objectName()
                    for w in item_widget.findChildren(QtWidgets.QWidget)
                    if w.property("legend_role") == "group_member"
                }

                # Keep grouped rows in sync when members are added/removed.
                if current_members != expected_members:
                    header = item_widget.findChild(QtWidgets.QToolButton)
                    is_expanded = bool(header and header.isChecked())
                    item_widget = self.make_grouped_legend_entry(
                        entry["members"], group_name=entry["group_name"]
                    )
                    self.legend.setItemWidget(item, item_widget)
                    item.setSizeHint(item_widget.sizeHint())

                    if is_expanded:
                        new_header = item_widget.findChild(QtWidgets.QToolButton)
                        if new_header:
                            new_header.setChecked(True)

                for member_name in entry["members"]:
                    try:
                        color = entry["member_visuals"][member_name][0].material.color
                    except BaseException:
                        color = gfx.Color("k")

                    member_widget = item_widget.findChild(
                        QtWidgets.QWidget, str(member_name)
                    )
                    if not member_widget:
                        continue
                    line_push_button = next(
                        (
                            button
                            for button in member_widget.findChildren(QtWidgets.QPushButton)
                            if button.property("legend_role") == "control"
                        ),
                        None,
                    )
                    if line_push_button:
                        line_push_button.setStyleSheet(
                            f"background-color: {color.css}"
                        )

                group_color_btn = next(
                    (
                        button
                        for button in item_widget.findChildren(QtWidgets.QPushButton)
                        if button.property("legend_role") == "group_control"
                    ),
                    None,
                )
                if group_color_btn:
                    try:
                        group_color = entry["visuals"][0].material.color
                    except BaseException:
                        group_color = gfx.Color("k")
                    group_color_btn.setStyleSheet(
                        f"background-color: {group_color.css}"
                    )

        # Add new items
        for entry_id, entry in legend_entries.items():
            if entry_id not in present:
                try:
                    color = entry["visuals"][0].material.color
                except BaseException:
                    # Note to self: need to make sure we also cater for color arrays
                    # which are in the geometry object
                    color = "k"

                if entry["kind"] == "group":
                    item = QtWidgets.QListWidgetItem()
                    item._id = entry_id
                    item_widget = self.make_grouped_legend_entry(
                        entry["members"], group_name=entry["group_name"]
                    )
                    item.setSizeHint(item_widget.sizeHint())
                else:
                    item, item_widget = self.make_legend_entry(
                        entry_id,
                        color=color,
                        type=type(entry["visuals"][0]),
                    )

                self.legend.addItem(item)
                self.legend.setItemWidget(item, item_widget)

        # Check visibility and selected status of objects
        visible = self.viewer.visible
        for i in range(self.legend.count()):
            item = self.legend.item(i)
            item_widget = self.legend.itemWidget(item)
            entry = legend_entries.get(item._id)

            if entry is None:
                continue

            if entry["kind"] == "group":
                member_ids = entry["members"]

                group_checkbox = next(
                    (
                        checkbox
                        for checkbox in item_widget.findChildren(QtWidgets.QCheckBox)
                        if checkbox.property("legend_role") == "group_visibility"
                    ),
                    None,
                )
                if group_checkbox is not None:
                    visible_count = sum(member_id in visible for member_id in member_ids)
                    is_mixed = 0 < visible_count < len(member_ids)

                    group_checkbox.blockSignals(True)
                    if is_mixed:
                        group_checkbox.setCheckState(QtCore.Qt.PartiallyChecked)
                    elif visible_count > 0:
                        group_checkbox.setCheckState(QtCore.Qt.Checked)
                    else:
                        group_checkbox.setCheckState(QtCore.Qt.Unchecked)

                    if is_mixed:
                        group_checkbox.setToolTip(
                            "Some objects in this group are hidden"
                        )
                    else:
                        group_checkbox.setToolTip(
                            "Toggle visibility for all group members"
                        )
                    group_checkbox.blockSignals(False)
            else:
                member_ids = [item._id]

            for member_id in member_ids:
                line_checkbox = item_widget.findChild(QtWidgets.QCheckBox, str(member_id))
                if line_checkbox:
                    line_checkbox.setChecked(member_id in visible)

                member_widget = item_widget.findChild(QtWidgets.QWidget, str(member_id))
                if member_widget is None:
                    member_widget = item_widget

                line_text = next(
                    (
                        button
                        for button in member_widget.findChildren(QtWidgets.QPushButton)
                        if button.property("legend_role") == "label"
                    ),
                    None,
                )
                if not line_text:
                    continue

                if self.viewer.selected and member_id in self.viewer.selected:
                    line_text.setStyleSheet("color: yellow; text-align: left;")
                else:
                    line_text.setStyleSheet("color: white; text-align: left;")

    @connect_color_picker
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
        elif not isinstance(self.active_objects, (list, tuple)):
            targets = [self.active_objects]
        else:
            targets = list(self.active_objects)

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

    @connect_color_picker
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
        self.active_objects = None

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
        # Keep shared picker alive but hide it when this controls closes.
        if Controls._active_color_controls is self:
            Controls._active_color_controls = None
        self.color_picker.hide()
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
