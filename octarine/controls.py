import ctypes
import sys

import numpy as np
import pygfx as gfx

from functools import wraps
from pathlib import Path

try:
    from PySide6 import QtWidgets, QtCore, QtGui
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Showing controls requires PySide6. Please install it via:\n `pip install PySide6`."
    )

# TODOs:
# - add custom legend formatting (e.g. "{object.name}")
# - show type of object in legend
# - add dropdown to manipulate all selected objects
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


def fix_native_color_picker(picker, qcolor):
    """Re-set the native macOS color panel's color as a proper sRGB color.

    Qt's Cocoa dialog helper hands QColors to the shared NSColorPanel via
    ``colorWithCalibratedRed`` (Generic RGB, gamma 1.8), so sRGB values
    display noticeably washed-out (e.g. #1F77B4 shows as #238BC1). After the
    dialog is shown, overwrite the panel's color with the same components
    tagged as sRGB. No-op off macOS, for non-native dialogs and on failure.
    """
    if sys.platform != "darwin" or qcolor is None:
        return
    if picker.testOption(QtWidgets.QColorDialog.DontUseNativeDialog):
        return
    try:
        c_void_p, c_double = ctypes.c_void_p, ctypes.c_double
        objc = ctypes.CDLL("/usr/lib/libobjc.dylib")
        objc.objc_getClass.restype = c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]

        def send(restype, receiver, sel, *args, argtypes=()):
            fn = ctypes.cast(
                objc.objc_msgSend,
                ctypes.CFUNCTYPE(restype, c_void_p, c_void_p, *argtypes),
            )
            return fn(receiver, objc.sel_registerName(sel), *args)

        # Messaging nil is a safe no-op, so a missing class/panel falls through.
        panel = send(c_void_p, objc.objc_getClass(b"NSColorPanel"), b"sharedColorPanel")
        color = send(
            c_void_p,
            objc.objc_getClass(b"NSColor"),
            b"colorWithSRGBRed:green:blue:alpha:",
            qcolor.redF(),
            qcolor.greenF(),
            qcolor.blueF(),
            qcolor.alphaF(),
            argtypes=[c_double] * 4,
        )
        if not panel or not color:
            return
        # The panel change echoes back into the dialog; keep it from
        # dispatching a (no-op) set_color.
        picker.blockSignals(True)
        send(None, panel, b"setColor:", color, argtypes=[c_void_p])
        picker.blockSignals(False)
    except Exception:
        pass


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

    # Point-size slider range for scatter visuals. The slider is integer-valued,
    # so size = tick * _SIZE_STEP.
    _SIZE_MIN = 0.5
    _SIZE_MAX = 50.0
    _SIZE_STEP = 0.5

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
        self.tab3 = QtWidgets.QWidget()
        self.tab4 = QtWidgets.QWidget()
        self.tab1_layout = QtWidgets.QVBoxLayout()
        self.tab2_layout = QtWidgets.QVBoxLayout()
        self.tab3_layout = QtWidgets.QVBoxLayout()
        self.tab4_layout = QtWidgets.QVBoxLayout()
        self.tab1.setLayout(self.tab1_layout)
        self.tab2.setLayout(self.tab2_layout)
        self.tab3.setLayout(self.tab3_layout)
        self.tab4.setLayout(self.tab4_layout)

        self.tabs.addTab(self.tab1, "Legend")
        self.tabs.addTab(self.tab2, "Controls")
        self.tabs.addTab(self.tab3, "Screenshot")
        self.tabs.addTab(self.tab4, "Effects")

        # self.btn_layout = QtWidgets.QVBoxLayout()
        # self.setLayout(self.btn_layout)

        # # Build legend
        # self.build_gui()

        # Build gui
        self.build_legend_gui()
        self.build_controls_gui()
        self.build_screenshot_gui()
        self.build_effects_gui()

        # Populate legend
        self.update_legend()

        # This determines the target for color changes
        self.active_objects = None
        self.active_volume = None

        self.color_picker = self._get_shared_color_picker()

    def showEvent(self, event):
        """Re-layout legend rows whenever the window is (re)shown.

        Rows added while the window was hidden keep the pre-show viewport
        width, clipping the visibility checkbox and color button off the right
        edge until the view lays out its item widgets again (e.g. on resize).
        """
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self.legend.doItemsLayout)

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
        # Add a search box on top of the legend. Searches both group names and
        # individual entries; see `filter_legend` for the behavior.
        self.legend_filter = QtWidgets.QLineEdit()
        self.legend_filter.setPlaceholderText("Filter legend…")
        self.legend_filter.setClearButtonEnabled(True)
        self.legend_filter.setToolTip("Search groups and entries")
        self.legend_filter.textChanged.connect(self.filter_legend)
        self.tab1_layout.addWidget(self.legend_filter)

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

    def build_screenshot_gui(self):
        """Build the GUI for the screenshot tab."""
        # Checkbox for transparent background
        self.screenshot_alpha_checkbox = QtWidgets.QCheckBox("Transparent")
        self.screenshot_alpha_checkbox.setChecked(True)
        self.screenshot_alpha_checkbox.setToolTip(
            "Hide the background to export a transparent PNG."
        )
        self.tab3_layout.addWidget(self.screenshot_alpha_checkbox)

        # Horizontal divider
        self.tab3_layout.addWidget(QHLine())

        # Show the current (physical) canvas size, i.e. what the screenshot
        # size will be unless a custom size is set below
        self.screenshot_current_size_label = QtWidgets.QLabel("")
        # Allow wrapping so the label does not add to the tab's minimum width
        self.screenshot_current_size_label.setWordWrap(True)
        self.screenshot_current_size_label.setToolTip(
            "Size of the screenshot unless a custom size is set below. This is "
            "the canvas size times the render pixel ratio."
        )
        self.tab3_layout.addWidget(self.screenshot_current_size_label)
        self._update_screenshot_size_label()

        # The canvas emits no resize signal we could hook into, so poll
        # (cheap; skipped entirely while the tab is not visible)
        self._screenshot_size_timer = QtCore.QTimer(self)
        self._screenshot_size_timer.setInterval(500)
        self._screenshot_size_timer.timeout.connect(self._update_screenshot_size_label)
        self._screenshot_size_timer.start()

        # Checkbox + spinboxes for a custom screenshot size
        self.screenshot_size_checkbox = QtWidgets.QCheckBox("Custom size")
        self.screenshot_size_checkbox.setChecked(False)
        self.screenshot_size_checkbox.setToolTip(
            "Render the screenshot at a custom size. The canvas is temporarily "
            "resized during the capture."
        )
        self.tab3_layout.addWidget(self.screenshot_size_checkbox)

        self.screenshot_width_spinbox = QtWidgets.QSpinBox()
        self.screenshot_width_spinbox.setToolTip("Width in pixels.")
        self.screenshot_height_spinbox = QtWidgets.QSpinBox()
        self.screenshot_height_spinbox.setToolTip("Height in pixels.")
        for spinbox, value in zip(
            (self.screenshot_width_spinbox, self.screenshot_height_spinbox),
            self.viewer.size,
        ):
            spinbox.setRange(1, 8192)
            spinbox.setValue(int(round(value)))
            spinbox.setEnabled(False)

        # One row per dimension - side by side they would dictate a fairly
        # large minimum width for the whole controls window
        size_layout = QtWidgets.QGridLayout()
        size_layout.addWidget(QtWidgets.QLabel("Width:"), 0, 0)
        size_layout.addWidget(self.screenshot_width_spinbox, 0, 1)
        size_layout.addWidget(QtWidgets.QLabel("Height:"), 1, 0)
        size_layout.addWidget(self.screenshot_height_spinbox, 1, 1)
        self.tab3_layout.addLayout(size_layout)

        def toggle_custom_size(checked):
            self.screenshot_width_spinbox.setEnabled(checked)
            self.screenshot_height_spinbox.setEnabled(checked)
            # Grey out the current-size label while it is overridden
            self.screenshot_current_size_label.setEnabled(not checked)

        self.screenshot_size_checkbox.toggled.connect(toggle_custom_size)

        # Horizontal divider
        self.tab3_layout.addWidget(QHLine())

        # Filename + browse button
        self.tab3_layout.addWidget(QtWidgets.QLabel("File:"))
        self.screenshot_filename_edit = QtWidgets.QLineEdit("screenshot.png")
        self.screenshot_browse_button = QtWidgets.QPushButton("Browse...")
        self.screenshot_browse_button.clicked.connect(self._screenshot_browse)

        filename_layout = QtWidgets.QHBoxLayout()
        filename_layout.addWidget(self.screenshot_filename_edit)
        filename_layout.addWidget(self.screenshot_browse_button)
        self.tab3_layout.addLayout(filename_layout)

        # Save + copy-to-clipboard buttons; stacked vertically to keep
        # the tab's minimum width small
        self.screenshot_save_button = QtWidgets.QPushButton("Save screenshot")
        self.screenshot_save_button.clicked.connect(self._save_screenshot)
        self.tab3_layout.addWidget(self.screenshot_save_button)
        self.screenshot_clipboard_button = QtWidgets.QPushButton("Copy to clipboard")
        self.screenshot_clipboard_button.clicked.connect(self._screenshot_to_clipboard)
        self.tab3_layout.addWidget(self.screenshot_clipboard_button)

        # Label for transient status messages (e.g. "Saved ...")
        self.screenshot_status_label = QtWidgets.QLabel("")
        self.screenshot_status_label.setWordWrap(True)
        self.tab3_layout.addWidget(self.screenshot_status_label)

        # This would make it so the widgets do not stretch when
        # we resize the window vertically
        self.tab3_layout.addStretch(1)

    def _update_screenshot_size_label(self):
        """Refresh the label showing the current canvas (=screenshot) size."""
        # `isVisible` also covers the controls window itself being hidden
        if self.screenshot_current_size_label.text() and not self.tab3.isVisible():
            return
        try:
            # The renderer snapshot comes out at logical size x pixel ratio
            w, h = self.viewer.size
            ratio = self.viewer.renderer.pixel_ratio
        except Exception:
            return
        w, h = int(round(w * ratio)), int(round(h * ratio))
        self.screenshot_current_size_label.setText(f"Current size: {w} x {h} px")

    def _screenshot_kwargs(self):
        """Collect screenshot options from the GUI."""
        kwargs = dict(alpha=self.screenshot_alpha_checkbox.isChecked())
        if self.screenshot_size_checkbox.isChecked():
            kwargs["size"] = (
                self.screenshot_width_spinbox.value(),
                self.screenshot_height_spinbox.value(),
            )
            # Without this, the image dimensions would be the requested size
            # times the renderer's pixel ratio (e.g. 2 on HiDPI screens)
            kwargs["pixel_ratio"] = 1
        return kwargs

    def _screenshot_browse(self):
        """Open a file dialog to pick the screenshot filename."""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save screenshot",
            self.screenshot_filename_edit.text() or "screenshot.png",
            "PNG image (*.png)",
        )
        if filename:
            self.screenshot_filename_edit.setText(filename)

    def _save_screenshot(self):
        """Save a screenshot with the current GUI settings."""
        filename = self.screenshot_filename_edit.text().strip()
        if not filename:
            self._set_screenshot_status("Please choose a filename.")
            return

        try:
            self.viewer.screenshot(filename=filename, **self._screenshot_kwargs())
        except Exception as e:
            self._set_screenshot_status(f"Error: {e}")
            return

        # Viewer.screenshot always writes PNG files - report the actual path
        path = Path(filename)
        if path.suffix != ".png":
            path = path.parent / f"{path.name}.png"
        self._set_screenshot_status(f"Saved {path.resolve()}")

    def _screenshot_to_clipboard(self):
        """Copy a screenshot to the system clipboard."""
        try:
            im = self.viewer.screenshot(filename=None, **self._screenshot_kwargs())
        except Exception as e:
            self._set_screenshot_status(f"Error: {e}")
            return

        im = np.ascontiguousarray(im)
        h, w = im.shape[:2]
        qimg = QtGui.QImage(im.data, w, h, w * 4, QtGui.QImage.Format_RGBA8888)
        # The QImage merely wraps the numpy buffer - copy so the clipboard
        # owns the data even after `im` is garbage-collected
        QtWidgets.QApplication.clipboard().setImage(qimg.copy())
        self._set_screenshot_status(f"Copied {w}x{h} image to clipboard.")

    def _set_screenshot_status(self, text, timeout=5000):
        """Show a transient status message in the screenshot tab."""
        self.screenshot_status_label.setText(text)
        self.screenshot_status_label.setToolTip(text)

        def clear():
            # Only clear if no newer message has replaced this one
            if self.screenshot_status_label.text() == text:
                self.screenshot_status_label.setText("")
                self.screenshot_status_label.setToolTip("")

        QtCore.QTimer.singleShot(timeout, clear)

    def build_effects_gui(self):
        """Build the GUI for the effects tab."""
        # --- Eye-Dome Lighting ---
        # This comes straight from pygfx, i.e. unlike the effects further
        # down it does not require octarine's custom shaders.
        # If EDL was already enabled via the API, reflect that here
        from pygfx.renderers.wgpu.engine.edl import EDLPass

        edl_pass = None
        for e in self.viewer.renderer.effect_passes:
            if isinstance(e, EDLPass):
                edl_pass = e
                break

        self.edl_checkbox = QtWidgets.QCheckBox("Eye-Dome Lighting")
        self.edl_checkbox.setToolTip(
            "Enhance depth perception for complex geometries by darkening "
            "edges based on depth differences between neighboring pixels."
        )
        self.edl_checkbox.setChecked(edl_pass is not None)
        self.tab4_layout.addWidget(self.edl_checkbox)

        def update_edl(*args):
            if not self.edl_checkbox.isChecked():
                return
            strength = self.edl_strength_slider
            radius = self.edl_radius_slider
            self.viewer.add_effect(
                "edl",
                strength=strength.value() * strength._step,
                radius=radius.value() * radius._step,
            )
            self.viewer._render_stale = True

        self.edl_strength_slider = self.create_effect_slider(
            "Strength",
            min=0.0,
            max=400.0,
            step=0.1,
            value=edl_pass.strength if edl_pass else 5.0,
            parent_layout=self.tab4_layout,
            callback=update_edl,
        )
        self.edl_radius_slider = self.create_effect_slider(
            "Radius",
            min=0.1,
            max=10.0,
            step=0.1,
            value=edl_pass.radius if edl_pass else 1.5,
            parent_layout=self.tab4_layout,
            callback=update_edl,
        )
        self.edl_radius_slider.setToolTip("Sampling radius in pixels.")
        for widget in (self.edl_strength_slider, self.edl_radius_slider):
            widget.setEnabled(edl_pass is not None)

        def toggle_edl(checked):
            if checked:
                update_edl()
            else:
                self.viewer.add_effect("edl", disable=True)
                self.viewer._render_stale = True
            self.edl_strength_slider.setEnabled(checked)
            self.edl_radius_slider.setEnabled(checked)

        self.edl_checkbox.toggled.connect(toggle_edl)

        # Horizontal divider
        self.tab4_layout.addWidget(QHLine())

        # The effects below require octarine's custom shaders (pygfx >=
        # 0.16); if those are unavailable we show the controls greyed-out.
        try:
            from . import shaders  # noqa: F401

            shaders_available = True
        except ImportError:
            shaders_available = False

        # --- Silhouette ---
        # If silhouette was already enabled via the API, reflect that here
        sil_power = 0.0
        if shaders_available:
            from .shaders import SilhouetteMeshMaterial

            for vis in self.viewer.scene.children:
                if isinstance(vis, gfx.Mesh) and isinstance(
                    vis.material, SilhouetteMeshMaterial
                ):
                    sil_power = max(sil_power, vis.material.silhouette)

        self.silhouette_checkbox = QtWidgets.QCheckBox("Silhouette")
        self.silhouette_checkbox.setToolTip(
            "Emphasize the edges/creases of meshes and make face-on regions "
            "transparent, giving an x-ray-like view of their outlines "
            "(same effect as Neuroglancer's 'silhouette' setting)."
        )
        self.silhouette_checkbox.setChecked(sil_power > 0)
        self.tab4_layout.addWidget(self.silhouette_checkbox)

        self.silhouette_slider = self.create_effect_slider(
            "Power",
            min=0.5,
            max=8.0,
            step=0.1,
            value=sil_power if sil_power > 0 else 2.0,
            parent_layout=self.tab4_layout,
            callback=lambda v: self.silhouette_checkbox.isChecked()
            and self.viewer.set_silhouette(v),
        )
        self.silhouette_slider.setEnabled(sil_power > 0)

        def toggle_silhouette(checked):
            slider = self.silhouette_slider
            self.viewer.set_silhouette(
                slider.value() * slider._step if checked else 0
            )
            slider.setEnabled(checked)

        self.silhouette_checkbox.toggled.connect(toggle_silhouette)

        # Horizontal divider
        self.tab4_layout.addWidget(QHLine())

        # --- Depth of field ---
        # If depth of field was already enabled via the API, reflect that here
        dof_pass = getattr(self.viewer, "_dof_pass", None)
        dof_on = dof_pass is not None and dof_pass.enabled

        self.dof_checkbox = QtWidgets.QCheckBox("Depth of Field")
        self.dof_checkbox.setToolTip(
            "Blur objects that are closer or farther than the focal plane, "
            "similar to a photographic lens."
        )
        self.dof_checkbox.setChecked(dof_on)
        self.tab4_layout.addWidget(self.dof_checkbox)

        def update_dof_params(*args):
            dof_pass = getattr(self.viewer, "_dof_pass", None)
            if dof_pass is None or not self.dof_checkbox.isChecked():
                return
            strength = self.dof_strength_slider
            radius = self.dof_radius_slider
            dof_pass.aperture = strength.value() * strength._step
            dof_pass.max_radius = radius.value() * radius._step
            self.viewer._render_stale = True

        self.dof_strength_slider = self.create_effect_slider(
            "Strength",
            min=0,
            max=300,
            step=1,
            value=dof_pass.aperture if dof_pass else 100,
            parent_layout=self.tab4_layout,
            callback=update_dof_params,
        )
        self.dof_radius_slider = self.create_effect_slider(
            "Max Blur",
            min=1,
            max=40,
            step=1,
            value=dof_pass.max_radius if dof_pass else 16,
            parent_layout=self.tab4_layout,
            callback=update_dof_params,
        )
        # Focus mode: autofocus or a fixed distance
        self._updating_focus_range = False  # guards apply_fixed_focus
        focus_layout = QtWidgets.QHBoxLayout()
        focus_layout.addWidget(QtWidgets.QLabel("Focus:"))
        self.dof_focus_dropdown = QtWidgets.QComboBox()
        self.dof_focus_dropdown.addItems(["Auto", "Fix"])
        self.dof_focus_dropdown.setToolTip("How to determine the focal distance.")
        self.dof_focus_dropdown.setItemData(
            0,
            "Focus on whatever is at the center of the view.",
            QtCore.Qt.ToolTipRole,
        )
        self.dof_focus_dropdown.setItemData(
            1, "Focus at a fixed distance from the camera.", QtCore.Qt.ToolTipRole
        )
        focus_layout.addWidget(self.dof_focus_dropdown)
        self.tab4_layout.addLayout(focus_layout)

        # In "Fix" mode: a slider to set the focal distance
        def apply_fixed_focus(value):
            dof_pass = getattr(self.viewer, "_dof_pass", None)
            if (
                dof_pass is None
                or self._updating_focus_range
                or not self.dof_checkbox.isChecked()
                or self.dof_focus_dropdown.currentText() != "Fix"
            ):
                return
            dof_pass.focus = value
            self.viewer._render_stale = True

        # Wrapped in a QWidget so the whole row can be hidden in "Auto" mode
        self.dof_focus_row = QtWidgets.QWidget()
        focus_row_layout = QtWidgets.QVBoxLayout(self.dof_focus_row)
        focus_row_layout.setContentsMargins(0, 0, 0, 0)
        self.dof_focus_slider = self.create_effect_slider(
            "Distance",
            min=0.0,
            max=100.0,
            step=0.5,
            value=50.0,
            parent_layout=focus_row_layout,
            callback=apply_fixed_focus,
        )
        self.tab4_layout.addWidget(self.dof_focus_row)

        def update_focus_range():
            """(Re-)fit the focus slider range to the scene and camera.

            This only moves the slider (and its label) - it does not touch
            the focus of the render pass itself.
            """
            cam = self.viewer.camera
            sphere = self.viewer.scene.get_world_bounding_sphere()
            if sphere is None:
                center, radius = np.zeros(3), 1.0
            else:
                center = np.asarray(sphere[:3], dtype=float)
                radius = max(float(sphere[3]), 1e-9)
            # View-space distance of the scene center from the camera
            p = cam.world.inverse_matrix @ np.append(center, 1.0)
            dist = -p[2] / p[3]

            dof_pass = getattr(self.viewer, "_dof_pass", None)
            current = dof_pass.focus if dof_pass is not None else None
            if current is None:
                current = dist

            # Make sure the range covers the current focus
            lo = min(dist - radius, current)
            hi = max(dist + radius, current)

            slider = self.dof_focus_slider
            self._updating_focus_range = True
            try:
                slider._step = (hi - lo) / 200
                slider.setMinimum(round(lo / slider._step))
                slider.setMaximum(round(hi / slider._step))
                slider.setValue(round(current / slider._step))
                # Force a label update even if the tick did not change
                slider.valueChanged.emit(slider.value())
            finally:
                self._updating_focus_range = False

        # In "Auto" mode: a slider for how far around the view center to
        # look for a focus target...
        def update_snap(value):
            dof_pass = getattr(self.viewer, "_dof_pass", None)
            if dof_pass is None or not self.dof_checkbox.isChecked():
                return
            dof_pass.snap_radius = value
            self.viewer._render_stale = True

        self.dof_snap_row = QtWidgets.QWidget()
        snap_row_layout = QtWidgets.QVBoxLayout(self.dof_snap_row)
        snap_row_layout.setContentsMargins(0, 0, 0, 0)
        self.dof_snap_slider = self.create_effect_slider(
            "Snap",
            min=0,
            max=100,
            step=1,
            value=int(dof_pass.snap_radius) if dof_pass else 0,
            parent_layout=snap_row_layout,
            callback=update_snap,
        )
        self.dof_snap_slider.setToolTip(
            "Focus on the closest object within this many pixels of the "
            "view center (0 = exact center only)."
        )
        self.tab4_layout.addWidget(self.dof_snap_row)

        # ... a checkbox to ease re-focusing over time...
        self.dof_smooth_checkbox = QtWidgets.QCheckBox("Smooth")
        self.dof_smooth_checkbox.setToolTip(
            "Re-focus gradually (over ~200 ms) instead of instantly."
        )
        self.tab4_layout.addWidget(self.dof_smooth_checkbox)

        def toggle_smooth(checked):
            if (
                getattr(self.viewer, "_dof_pass", None) is None
                or not self.dof_checkbox.isChecked()
            ):
                return  # picked up when depth of field is enabled
            strength = self.dof_strength_slider
            radius = self.dof_radius_slider
            snap = self.dof_snap_slider
            # Route through the API so the keep-alive animation is managed
            self.viewer.set_depth_of_field(
                focus=None,
                aperture=strength.value() * strength._step,
                max_radius=radius.value() * radius._step,
                smooth=checked,
                snap_radius=snap.value() * snap._step,
            )

        # ... and a checkbox to mark the current focal point
        self.dof_focus_marker_checkbox = QtWidgets.QCheckBox("Show focus point")
        self.dof_focus_marker_checkbox.setToolTip(
            "Show a marker that tracks the point the camera is focused on."
        )
        self.tab4_layout.addWidget(self.dof_focus_marker_checkbox)

        def switch_focus_mode(*args):
            fix = self.dof_focus_dropdown.currentText() == "Fix"
            self.dof_focus_row.setVisible(fix)
            self.dof_snap_row.setVisible(not fix)
            self.dof_smooth_checkbox.setVisible(not fix)
            self.dof_focus_marker_checkbox.setVisible(not fix)
            enabled = self.dof_checkbox.isChecked()
            if fix:
                self._set_dof_focus_marker(False)
                update_focus_range()
                slider = self.dof_focus_slider
                apply_fixed_focus(slider.value() * slider._step)
            else:
                dof_pass = getattr(self.viewer, "_dof_pass", None)
                if dof_pass is not None and enabled:
                    dof_pass.focus = None
                    self.viewer._render_stale = True
                self._set_dof_focus_marker(
                    enabled and self.dof_focus_marker_checkbox.isChecked()
                )

        # Initialize the focus widgets (before connecting any signals)
        dof_fix = dof_pass is not None and dof_pass.focus is not None
        self.dof_focus_dropdown.setCurrentIndex(1 if dof_fix else 0)
        self.dof_focus_row.setVisible(dof_fix)
        self.dof_snap_row.setVisible(not dof_fix)
        self.dof_smooth_checkbox.setChecked(dof_pass is not None and dof_pass.smooth > 0)
        self.dof_smooth_checkbox.setVisible(not dof_fix)
        self.dof_focus_marker_checkbox.setVisible(not dof_fix)
        if dof_fix:
            update_focus_range()

        self.dof_focus_dropdown.currentIndexChanged.connect(switch_focus_mode)
        self.dof_smooth_checkbox.toggled.connect(toggle_smooth)
        self.dof_focus_marker_checkbox.toggled.connect(
            lambda checked: self._set_dof_focus_marker(
                checked and self.dof_checkbox.isChecked()
            )
        )

        dof_widgets = (
            self.dof_strength_slider,
            self.dof_radius_slider,
            self.dof_focus_dropdown,
            self.dof_focus_slider,
            self.dof_snap_slider,
            self.dof_smooth_checkbox,
            self.dof_focus_marker_checkbox,
        )
        for widget in dof_widgets:
            widget.setEnabled(dof_on)

        def toggle_dof(checked):
            if checked:
                fix = self.dof_focus_dropdown.currentText() == "Fix"
                if fix:
                    update_focus_range()
                strength = self.dof_strength_slider
                radius = self.dof_radius_slider
                focus = self.dof_focus_slider
                snap = self.dof_snap_slider
                # focus=None -> auto-focus on whatever is at the view center
                self.viewer.set_depth_of_field(
                    focus=focus.value() * focus._step if fix else None,
                    aperture=strength.value() * strength._step,
                    max_radius=radius.value() * radius._step,
                    smooth=self.dof_smooth_checkbox.isChecked(),
                    snap_radius=snap.value() * snap._step,
                )
                self._set_dof_focus_marker(
                    not fix and self.dof_focus_marker_checkbox.isChecked()
                )
            else:
                self.viewer.set_depth_of_field(False)
                self._set_dof_focus_marker(False)
            for widget in dof_widgets:
                widget.setEnabled(checked)

        self.dof_checkbox.toggled.connect(toggle_dof)

        if not shaders_available:
            import pygfx

            msg = (
                "Effects require pygfx >= 0.16 "
                f"(you have {pygfx.__version__}). Please update pygfx."
            )
            for widget in (self.silhouette_checkbox, self.silhouette_slider) + (
                self.dof_checkbox,
            ) + dof_widgets:
                widget.setEnabled(False)
                widget.setToolTip(msg)

        self.tab4_layout.addStretch(1)

    def _set_dof_focus_marker(self, visible):
        """Show/hide a marker tracking the depth-of-field focal point."""
        # The animation hook checks this flag: `remove_animation` is
        # deferred, so the hook runs one more time after being removed and
        # would otherwise re-show the marker we just hid.
        self._dof_focus_marker_active = visible
        marker = getattr(self, "_dof_focus_marker", None)
        if not visible:
            if marker is not None:
                self.viewer.remove_animation(self._track_dof_focus)
                if marker.visible:
                    marker.visible = False
                    self.viewer._render_stale = True
            return

        if marker is None:
            marker = gfx.Points(
                gfx.Geometry(positions=np.zeros((1, 3), dtype=np.float32)),
                gfx.PointsMarkerMaterial(
                    marker="ring",
                    size=15,
                    color="#ff00ff",
                    edge_color="#ffffff",
                    edge_width=1.5,
                ),
            )
            # The marker must neither occlude anything nor write to the
            # depth buffer (which would confuse the autofocus itself)
            marker.material.depth_write = False
            marker.material.depth_test = False
            marker.visible = False
            self._dof_focus_marker = marker
            # N.B. we add the marker directly to the scene (not via
            # viewer.add) so it does not show up in the legend
            self.viewer.scene.add(marker)

        # The animation runs every frame and moves the marker to wherever
        # the focus currently is
        self.viewer.add_animation(
            self._track_dof_focus, on_error="log", req_render=False
        )

    def _track_dof_focus(self):
        """Animation hook: move the focus marker to the current focal point."""
        marker = self._dof_focus_marker
        dof_pass = getattr(self.viewer, "_dof_pass", None)
        pos = None
        if (
            getattr(self, "_dof_focus_marker_active", False)
            and dof_pass is not None
            and dof_pass.enabled
        ):
            pos = dof_pass.get_focus_position(self.viewer.renderer)

        if pos is None:
            if marker.visible:
                marker.visible = False
                self.viewer._render_stale = True
            return

        pos = pos.astype(np.float32)
        if marker.visible and np.array_equal(marker.geometry.positions.data[0], pos):
            return
        marker.geometry.positions.data[0] = pos
        marker.geometry.positions.update_full()
        marker.visible = True
        self.viewer._render_stale = True

    def create_legend(self, spacing=0, index=None):
        """Generate the legend widget."""
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Legend"))
        list_widget = QtWidgets.QListWidget()
        list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        # Long names elide rather than scroll, so the row never needs to scroll
        # sideways; keep the horizontal scrollbar off as a safeguard.
        list_widget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        layout.addWidget(list_widget)
        list_widget.setSpacing(spacing)

        # Watch the list and its viewport so hover-highlighting clears when the
        # cursor moves onto empty space or leaves the legend entirely.
        list_widget.installEventFilter(self)
        list_widget.viewport().installEventFilter(self)

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

        # Generate the label. ElidedLabel truncates long names with an ellipsis so
        # the visibility toggle + color control are never pushed out of view; the
        # full name is available via the tooltip.
        line_text = ElidedLabel(f"{name}")
        line_text.setProperty("legend_role", "label")

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

        # Highlight the underlying object while hovering this row.
        self._install_hover_highlight(item_widget, name)

        return item, item_widget

    def _make_group_member_row(self, name):
        """Build one compact group-member row widget for `name`.

        Returns the child QWidget ready to be inserted into a group's content
        layout. Used both when first building a group and when incrementally
        adding members in `update_legend`.
        """
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

        return child_widget

    def _materialize_group_members(self, item_widget):
        """Build a group's member rows on first expand (lazy materialization).

        Rows are created from `item_widget._member_names` and seeded with the
        current visibility/selection state so they match the viewer without
        waiting for the next `update_legend` call. No-op if already materialized.
        """
        if getattr(item_widget, "_materialized", False):
            return

        content_widget = item_widget.findChild(QtWidgets.QWidget, "group_content")
        if content_widget is None:
            return

        visible = self.viewer.visible
        selected = self.viewer.selected
        for member_name in item_widget._member_names:
            row = self._make_group_member_row(member_name)
            content_widget.layout().addWidget(row)

            # Seed checkbox state without retriggering the per-row callback.
            checkbox = row.findChild(QtWidgets.QCheckBox, str(member_name))
            if checkbox is not None:
                checkbox.blockSignals(True)
                checkbox.setChecked(member_name in visible)
                checkbox.blockSignals(False)

            # Seed selection highlight on the label.
            label = next(
                (
                    widget
                    for widget in row.findChildren(ElidedLabel)
                    if widget.property("legend_role") == "label"
                ),
                None,
            )
            if label is not None:
                label.setSelected(bool(selected and member_name in selected))

        item_widget._materialized = True

    def _install_hover_highlight(self, widget, target):
        """Highlight `target` object(s) while the cursor is over `widget`.

        `target` is a raw object id or a (live) list of ids accepted by
        `Viewer.highlight_objects`. The filter is installed on `widget` and all
        of its current children so hovering any sub-control (label, checkbox,
        color button) is treated as hovering the row.
        """
        widget._hover_target = target
        widget.installEventFilter(self)
        for child in widget.findChildren(QtWidgets.QWidget):
            child.installEventFilter(self)

    def _find_hover_target(self, widget):
        """Walk up from `widget` to the nearest row carrying a hover target."""
        while widget is not None:
            target = getattr(widget, "_hover_target", None)
            if target is not None:
                return target
            widget = widget.parent()
        return None

    def _set_hover_highlight(self, target):
        """Move the hover highlight to `target` (or clear it when None)."""
        current = getattr(self, "_legend_hover_target", None)
        if target == current:
            return

        # Highlighting touches material colors; guard so a stale/removed id
        # never breaks hover handling.
        if current is not None:
            try:
                self.viewer.unhighlight_objects(current)
            except BaseException:
                pass
        if target is not None:
            try:
                self.viewer.highlight_objects(target)
            except BaseException:
                pass

        self._legend_hover_target = target
        self.viewer._render_stale = True

    def eventFilter(self, obj, event):
        """Drive legend hover-highlighting from widget enter/leave events."""
        etype = event.type()
        if etype == QtCore.QEvent.Enter:
            self._set_hover_highlight(self._find_hover_target(obj))
        elif etype == QtCore.QEvent.Leave:
            # Child leaves are followed by a sibling enter, so only clear when
            # the cursor leaves the list (or moves onto its empty background).
            if obj is self.legend or obj is self.legend.viewport():
                self._set_hover_highlight(None)
        return super().eventFilter(obj, event)

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
        # Canonical, mutable member list shared with closures and the group color
        # button so incremental updates stay in sync without rebuilding the widget.
        item_widget._member_names = list(names)
        # Bare group label kept around so the header text (which also shows the
        # member count) can be rebuilt when membership changes.
        item_widget._group_name = group_name
        # Member rows are built lazily on first expand; see _materialize_group_members.
        item_widget._materialized = False
        item_layout = QtWidgets.QVBoxLayout(item_widget)
        item_layout.setContentsMargins(0, 0, 0, 0)
        item_layout.setSpacing(0)

        # Header button to toggle child visibility.
        header_row = QtWidgets.QWidget()
        header_row_layout = QtWidgets.QHBoxLayout(header_row)
        header_row_layout.setContentsMargins(0, 0, 0, 0)
        header_row_layout.setSpacing(0)

        # Arrow-only toggle button. The group name lives in a separate ElidedLabel
        # so a long name truncates instead of pushing the controls out of view.
        header = QtWidgets.QToolButton()
        header.setArrowType(QtCore.Qt.RightArrow)
        header.setCheckable(True)
        header.setChecked(False)
        header.setStyleSheet(
            "QToolButton {"
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
        header.setFixedWidth(16)
        header.setToolTip("Click to expand/collapse grouped legend entries")

        # Group name + member count. Clicking it toggles the group, mirroring the
        # arrow button.
        header_label = ElidedLabel(f"{group_name} ({len(names)})")
        header_label.setProperty("legend_role", "group_label")
        header_label.clicked.connect(header.toggle)

        group_checkbox = QtWidgets.QCheckBox()
        group_checkbox.setProperty("legend_role", "group_visibility")
        group_checkbox.setTristate(True)
        group_checkbox.setMaximumWidth(40)
        group_checkbox.setToolTip("Toggle visibility for all group members")

        def toggle_group_visibility(*args):
            is_visible = group_checkbox.isChecked()
            for member_name in item_widget._member_names:
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
        # Share the canonical list so the color picker always targets the current
        # members; incremental updates mutate this list in place.
        group_color_button._id = item_widget._member_names

        header_row_layout.addWidget(header)
        header_row_layout.addWidget(header_label)
        header_row_layout.addWidget(group_checkbox)
        header_row_layout.addSpacing(15)
        header_row_layout.addWidget(group_color_button)
        header_row_layout.setStretch(1, 1)
        item_layout.addWidget(header_row)

        # Hovering the group header highlights all members. Pass the live member
        # list so the highlight always reflects the current membership.
        self._install_hover_highlight(header_row, item_widget._member_names)

        # Child container with one standard legend row per member.
        content_widget = QtWidgets.QWidget()
        content_widget.setObjectName("group_content")
        content_layout = QtWidgets.QVBoxLayout(content_widget)
        content_layout.setContentsMargins(18, 0, 0, 0)
        content_layout.setSpacing(0)
        content_widget.setVisible(False)

        # Rows are intentionally NOT built here: they are materialized on first
        # expand to avoid creating widgets for groups the user never opens.
        item_layout.addWidget(content_widget)

        def toggle_group(expanded):
            header.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
            if expanded:
                self._materialize_group_members(item_widget)
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

    def filter_legend(self, text):
        """Filter legend rows by `text` (case-insensitive substring).

        Searches both group names and individual entries (top-level singles and
        group members):

        - Single entry matching -> shown, otherwise hidden.
        - Group name matching -> group shown collapsed with all members (member
          matching is not applied; a group-name match wins).
        - Group member matching (group name does not) -> group shown expanded
          with only the matching member rows visible.
        - No match in a group -> group hidden.
        - Empty text -> everything shown, all member rows restored, all groups
          collapsed.
        """
        text = text.strip().lower()
        for i in range(self.legend.count()):
            item = self.legend.item(i)
            item_widget = self.legend.itemWidget(item)
            is_group = hasattr(item_widget, "_member_names")

            # Cleared filter -> reset everything and collapse all groups.
            if not text:
                item.setHidden(False)
                if is_group:
                    self._set_group_filter_state(
                        item, item_widget, expanded=False, visible_members=None
                    )
                continue

            # Single entry.
            if not is_group:
                item.setHidden(text not in str(item._id).lower())
                continue

            # Group: match the group name first (wins over member matching).
            group_name = str(item._id)[len("group::") :]
            if text in group_name.lower():
                item.setHidden(False)
                self._set_group_filter_state(
                    item, item_widget, expanded=False, visible_members=None
                )
                continue

            matches = [m for m in item_widget._member_names if text in str(m).lower()]
            if matches:
                item.setHidden(False)
                self._set_group_filter_state(
                    item,
                    item_widget,
                    expanded=True,
                    visible_members={str(m) for m in matches},
                )
            else:
                item.setHidden(True)

    def _set_group_filter_state(self, item, item_widget, expanded, visible_members):
        """Expand/collapse a group and set per-member-row visibility for filtering.

        `visible_members` is a set of stringified member ids to keep visible, or
        ``None`` to show all members. Driven directly (signals blocked) rather
        than via the header's ``toggled`` signal, which would not fire when the
        checked state is unchanged.
        """
        header = item_widget.findChild(QtWidgets.QToolButton)
        if header is not None:
            header.blockSignals(True)
            header.setChecked(expanded)
            header.setArrowType(
                QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow
            )
            header.blockSignals(False)

        # Member rows are built lazily on first expand; ensure they exist before
        # we try to toggle their visibility.
        if expanded:
            self._materialize_group_members(item_widget)

        content_widget = item_widget.findChild(QtWidgets.QWidget, "group_content")
        if content_widget is not None:
            content_widget.setVisible(expanded)

        for row in item_widget.findChildren(QtWidgets.QWidget):
            if row.property("legend_role") != "group_member":
                continue
            row.setVisible(
                visible_members is None or row.objectName() in visible_members
            )

        item.setSizeHint(item_widget.sizeHint())

    def _update_group_header_text(self, item_widget):
        """Refresh a group header's label, including its member count."""
        label = next(
            (
                widget
                for widget in item_widget.findChildren(ElidedLabel)
                if widget.property("legend_role") == "group_label"
            ),
            None,
        )
        if label is not None:
            count = len(item_widget._member_names)
            label.setText(f"{item_widget._group_name} ({count})")

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
                # Object ids may be non-strings (e.g. uuid.UUID), so keep raw ids
                # for building rows / highlighting and only stringify for matching
                # against widget objectNames.
                expected_members = {str(member) for member in entry["members"]}

                if not item_widget._materialized:
                    # Rows haven't been built yet (group never expanded). Just track
                    # the current membership; the rows will be built from this list
                    # on first expand, so no widget work is needed here.
                    item_widget._member_names[:] = list(entry["members"])
                else:
                    current_members = {
                        w.objectName()
                        for w in item_widget.findChildren(QtWidgets.QWidget)
                        if w.property("legend_role") == "group_member"
                    }

                    # Keep grouped rows in sync when members are added/removed by
                    # editing only the changed rows in place, rather than rebuilding
                    # the entire group widget (expensive for large groups).
                    if current_members != expected_members:
                        content_widget = item_widget.findChild(
                            QtWidgets.QWidget, "group_content"
                        )

                        # Remove rows for members no longer in the group.
                        for member_name in current_members - expected_members:
                            row = content_widget.findChild(
                                QtWidgets.QWidget, member_name
                            )
                            if row is not None:
                                content_widget.layout().removeWidget(row)
                                row.deleteLater()

                        # Insert new rows (raw id) at their canonical index.
                        for idx, member in enumerate(entry["members"]):
                            if str(member) not in current_members:
                                content_widget.layout().insertWidget(
                                    idx, self._make_group_member_row(member)
                                )

                        # Mutate the shared list in place (raw ids) so the group
                        # color button's `_id` and `toggle_group_visibility` stay
                        # in sync.
                        item_widget._member_names[:] = list(entry["members"])

                        item.setSizeHint(item_widget.sizeHint())

                # Keep the header's member count in sync with the current members.
                self._update_group_header_text(item_widget)

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
                        widget
                        for widget in member_widget.findChildren(ElidedLabel)
                        if widget.property("legend_role") == "label"
                    ),
                    None,
                )
                if not line_text:
                    continue

                line_text.setSelected(
                    bool(self.viewer.selected and member_id in self.viewer.selected)
                )

        # Re-apply an active filter so rows added/removed above stay consistent
        # with the current search text.
        if getattr(self, "legend_filter", None) is not None and self.legend_filter.text():
            self.filter_legend(self.legend_filter.text())

    @connect_color_picker
    def color_button_clicked(self):
        """Set the active object to be the buttons target."""
        sender = self.sender()
        push_button = self.findChild(QtWidgets.QPushButton, sender.objectName())
        # print(f'click: {push_button.objectName()}')
        self.active_objects = push_button._id
        self._sync_and_show_size_popup(anchor=push_button)
        self._show_color_picker()

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
        targets = self._resolve_targets()
        if not targets:
            return

        # Convert QColor to [0-1] RGBA
        rgba = np.array(color.toTuple()) / 255

        # The picker is pre-filled fully opaque (see `_sync_color_picker`), so
        # alpha = 1 means the slider was not touched: keep each object's
        # current alpha instead of making it opaque. Lowering the slider
        # (even to 99%) explicitly sets the alpha.
        if rgba[3] < 1:
            cmap = {name: tuple(rgba) for name in targets}
        else:
            cmap = {
                name: tuple(rgba[:3]) + (self._current_alpha(name),)
                for name in targets
            }

        self.viewer.set_colors(cmap)

    def _current_alpha(self, name):
        """Return the current alpha of the (first) visual of an object."""
        color = self._object_color(name)
        return tuple(color.rgba)[3] if color is not None else 1.0

    def _sync_color_picker(self):
        """Pre-set the picker to the current color of the (first) target object.

        The color is pre-filled fully opaque: with the object's alpha baked in,
        the dialog swatch blends towards its background and a translucent blue
        object would show up as a washed-out white-blue. `set_color` preserves
        the objects' alpha as long as the dialog's alpha slider stays at 100%.

        Returns the pre-filled QColor, or None if no target color resolved.
        """
        for name in self._resolve_targets():
            color = self._object_color(name)
            if color is None:
                continue
            qcolor = QtGui.QColor.fromRgbF(
                *(min(max(c, 0.0), 1.0) for c in color.rgb), 1.0
            )
            # Block signals so pre-filling doesn't fire set_color, which
            # would flatten a mixed-color selection to this one color.
            self.color_picker.blockSignals(True)
            self.color_picker.setCurrentColor(qcolor)
            self.color_picker.blockSignals(False)
            return qcolor
        return None

    def _object_color(self, name):
        """Return an object's true color, looking through any active highlight.

        Both selection (yellow) and hover (brightened) highlights swap
        `material.color` and stash the real color away. The legend entry is
        typically still hovered when the color picker is opened, so reading
        `material.color` directly would sync the highlight color instead.
        """
        selected = self.viewer.selected or []
        for vis in self.viewer.objects.get(name, []):
            try:
                if name in selected and hasattr(vis, "_stored_color"):
                    return gfx.Color(vis._stored_color)
                if getattr(vis, "_highlighted", False):
                    return gfx.Color(vis.material._original_color)
                return gfx.Color(vis.material.color)
            except BaseException:
                # E.g. per-vertex colors; try the next visual.
                continue
        return None

    def _show_color_picker(self):
        """Pre-fill the picker with the active objects' color and show it."""
        qcolor = self._sync_color_picker()
        self.color_picker.show()
        # Fix up the native panel's colorspace both right away and once the
        # event loop has settled, in case Qt pushes its own (mis-converted)
        # color to the panel asynchronously while showing.
        fix_native_color_picker(self.color_picker, qcolor)
        QtCore.QTimer.singleShot(
            0, lambda: fix_native_color_picker(self.color_picker, qcolor)
        )

    def _resolve_targets(self):
        """Flatten self.active_objects into a list of object names."""
        if self.active_objects is None:
            return []
        if self.active_objects == "selected":
            return self.get_selected()
        if not isinstance(self.active_objects, (list, tuple)):
            return [self.active_objects]
        return list(self.active_objects)  # copy: group _id is a live list

    def _iter_points_visuals(self):
        """Yield active gfx.Points visuals whose size can be set (skip per-vertex)."""
        for name in self._resolve_targets():
            for vis in self.viewer.objects.get(name, []):
                if not isinstance(vis, gfx.Points):
                    continue
                # Per-vertex sizing ignores material.size; don't pretend to set it.
                if getattr(vis.material, "size_mode", None) == "vertex":
                    continue
                yield vis

    @set_viewer_stale
    def set_point_size(self, size):
        """Set material.size on all active Points visuals (size-slider callback)."""
        for vis in self._iter_points_visuals():
            vis.material.size = float(size)

    def _ensure_point_size_popup(self):
        """Lazily build the per-instance point-size popup and return it."""
        if getattr(self, "_point_size_popup", None) is not None:
            return self._point_size_popup

        # Qt.Tool: floats beside the (native) color dialog without stealing focus
        # or auto-closing on outside clicks.
        popup = QtWidgets.QWidget(self, QtCore.Qt.Tool)
        popup.setWindowTitle("Point size")
        layout = QtWidgets.QHBoxLayout(popup)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.addWidget(QtWidgets.QLabel("Point size"))

        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(int(round(self._SIZE_MIN / self._SIZE_STEP)))
        slider.setMaximum(int(round(self._SIZE_MAX / self._SIZE_STEP)))
        label = QtWidgets.QLabel(f"{self._SIZE_MIN:3.2f}")
        label.setMinimumWidth(40)

        def on_change(tick):
            size = tick * self._SIZE_STEP
            label.setText(f"{size:3.2f}")
            self.set_point_size(size)

        slider.valueChanged.connect(on_change)
        layout.addWidget(slider, stretch=1)
        layout.addWidget(label)

        self._point_size_popup = popup
        self._point_size_slider = slider
        self._point_size_label = label
        return popup

    def _sync_and_show_size_popup(self, anchor=None):
        """Show the size popup near `anchor` if any active target is a Points visual."""
        first = next(self._iter_points_visuals(), None)
        if first is None:
            if getattr(self, "_point_size_popup", None) is not None:
                self._point_size_popup.hide()
            return

        popup = self._ensure_point_size_popup()
        size = float(getattr(first.material, "size", self._SIZE_MIN))
        size = max(self._SIZE_MIN, min(self._SIZE_MAX, size))
        tick = int(round(size / self._SIZE_STEP))

        # Block signals so the pre-fill doesn't re-quantize the stored size.
        self._point_size_slider.blockSignals(True)
        self._point_size_slider.setValue(tick)
        self._point_size_slider.blockSignals(False)
        self._point_size_label.setText(f"{tick * self._SIZE_STEP:3.2f}")

        if anchor is not None:
            popup.move(anchor.mapToGlobal(QtCore.QPoint(0, anchor.height())))
        popup.show()
        popup.raise_()

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
        self._sync_and_show_size_popup(anchor=self)
        self._show_color_picker()

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

    def create_effect_slider(
        self, name, min, max, step, value, parent_layout, callback
    ):
        """Generate a slider for a viewer-level (effect) property.

        Unlike `create_slider` this does not target scene objects but simply
        reports values to a callback; the initial value is passed explicitly.
        Note that the underlying QSlider is integer-valued: `slider.value()`
        returns ticks which need to be multiplied by `step`.
        """
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel(name))

        if isinstance(step, float):
            val_label = QtWidgets.QLabel(f"{float(value):3.2f}")
        else:
            val_label = QtWidgets.QLabel(f"{int(value):03d}")
        layout.addWidget(val_label)

        slide = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slide.setMinimum(round(min / step))
        slide.setMaximum(round(max / step))
        slide.setValue(round(value / step))
        # For convenience: slider.value() * slider._step = actual value.
        # `_step` may be re-assigned (together with new min/max ticks) to
        # re-range the slider.
        slide._step = step

        def set_value(tick):
            value = tick * slide._step
            if isinstance(slide._step, float):
                val_label.setText(f"{float(value):3.2f}")
            else:
                val_label.setText(f"{int(value):03d}")
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
        # Stop tracking the depth-of-field focus point (if we were)
        self._set_dof_focus_marker(False)
        self.color_picker.hide()
        if getattr(self, "_point_size_popup", None) is not None:
            self._point_size_popup.hide()
        super().close()


class ElidedLabel(QtWidgets.QLabel):
    """A QLabel that elides its text with an ellipsis instead of growing the row.

    Used for legend names so the visibility checkbox and color button always stay
    visible: the label's width hint is zeroed, and the (full) text is painted
    elided to the right at render time so it adapts to resizes without any
    size-hint recomputation. The full text is shown in the tooltip.
    """

    clicked = QtCore.Signal()

    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self._full_text = ""
        self._selected = False
        # Width hint is ignored so a long name never pushes the controls out.
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred
        )
        self.setMinimumWidth(0)
        self.setText(text)

    def setText(self, text):
        self._full_text = "" if text is None else str(text)
        self.setToolTip(self._full_text)
        super().setText(self._full_text)
        self.update()

    def text(self):
        return self._full_text

    def setSelected(self, selected):
        """Toggle the selection highlight (yellow vs. white text)."""
        if selected != self._selected:
            self._selected = selected
            self.update()

    def sizeHint(self):
        return QtCore.QSize(0, self.fontMetrics().height())

    def minimumSizeHint(self):
        return QtCore.QSize(0, self.fontMetrics().height())

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        rect = self.contentsRect()
        elided = self.fontMetrics().elidedText(
            self._full_text, QtCore.Qt.ElideRight, rect.width()
        )
        painter.setPen(
            QtGui.QColor("yellow") if self._selected else QtGui.QColor("white")
        )
        painter.drawText(
            rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, elided
        )

    def mousePressEvent(self, event):
        # Accept the press so the matching release (and our `clicked` signal) is
        # delivered here rather than propagating to the list item.
        if event.button() == QtCore.Qt.LeftButton:
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.rect().contains(
            event.pos()
        ):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


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
