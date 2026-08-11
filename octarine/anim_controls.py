"""The "Animation" tab of the controls window.

This lives in its own module (mixed into `octarine.controls.Controls`) mostly
to keep `controls.py` from growing another few hundred lines. It is a GUI on
top of `octarine.anim_utils.Animation`: everything here either builds an
`Animation` from what the widgets say or drives one - nothing about how
animations work belongs in this file.

Recording is driven frame by frame off a timer rather than in a loop, so the
window stays responsive (and cancellable) while it renders.

"""

from pathlib import Path

from PySide6 import QtWidgets, QtCore

from .anim_utils import EASINGS, OUTPUT_FORMATS, Animation, frame_count

# Axis choices for the orbit mode: label -> what `Animation.orbit` wants
ORBIT_AXES = {
    "Up (turntable)": "up",
    "X": "x",
    "Y": "y",
    "Z": "z",
    "View (roll)": "view",
}


class AnimationControlsMixin:
    """Animation tab for `Controls`.

    Expects `self.viewer` and `self.tab5_layout`, the legend selection
    (`self.get_selected`) for the "selected objects" orbit target, plus the
    shared widget helpers on `Controls` (`make_scrolling_tab`,
    `create_size_widgets`, `open_file_dialog`, `set_status`).

    """

    def build_animation_gui(self):
        """Build the GUI for the animation tab."""
        # State of the keyframe mode: one entry per waypoint, in order. The
        # `duration`/`easing` of an entry describe the move *into* it (the
        # first entry's are only used when looping back to it at the end).
        self._anim_keyframes = []
        self._anim_preview = None
        self._anim_recorder = None
        self._anim_record_timer = None
        self._anim_syncing_keyframe = False

        # There is more in this tab than in any of the others - putting it in a
        # scroll area keeps it from dictating the height of the whole controls
        # window (a tab is only ever as small as its tallest tab's contents)
        self.anim_layout = self.make_scrolling_tab(self.tab5_layout)
        self.anim_layout.addWidget(self._build_anim_animation_group())
        self.anim_layout.addWidget(self._build_anim_render_group())
        # Keeps the groups at the top when the window is taller than they are
        self.anim_layout.addStretch(1)

        self._anim_update_summary()

    def _build_anim_animation_group(self):
        """The "Animation" group: what to animate, and playing it back."""
        group = QtWidgets.QGroupBox("Animation")
        layout = QtWidgets.QVBoxLayout(group)

        # Mode selector: the two ways of putting an animation together
        self.anim_mode_dropdown = QtWidgets.QComboBox()
        self.anim_mode_dropdown.addItems(["Orbit", "Keyframes"])
        self.anim_mode_dropdown.setToolTip(
            "Orbit: circle the camera around the scene or a selection.\n"
            "Keyframes: fly through views you captured yourself."
        )
        layout.addWidget(self.anim_mode_dropdown)

        # One page per mode. These are shown/hidden rather than stacked: a
        # stacked widget is always as tall as its tallest page, which would
        # leave a gap under the (much shorter) orbit page
        self.anim_pages = [
            self._build_anim_orbit_page(),
            self._build_anim_keyframe_page(),
        ]
        for page in self.anim_pages:
            layout.addWidget(page)
        self.anim_mode_dropdown.currentIndexChanged.connect(self._anim_show_page)
        self.anim_mode_dropdown.currentIndexChanged.connect(
            lambda *_: self._anim_update_summary()
        )
        self._anim_show_page(self.anim_mode_dropdown.currentIndex())

        self.anim_preview_button = QtWidgets.QPushButton("Preview")
        self.anim_preview_button.setToolTip(
            "Play the animation in the viewer (on a loop) without recording."
        )
        self.anim_preview_button.clicked.connect(self._anim_toggle_preview)
        layout.addWidget(self.anim_preview_button)

        return group

    def _anim_show_page(self, index):
        """Show the settings for the selected mode, hide the other's."""
        for i, page in enumerate(self.anim_pages):
            page.setVisible(i == index)

    def _build_anim_orbit_page(self):
        """Page with the settings for a plain orbit."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setLabelAlignment(QtCore.Qt.AlignLeft)
        layout.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)

        self.anim_orbit_target = QtWidgets.QComboBox()
        self.anim_orbit_target.addItems(["Everything", "Selected objects"])
        self.anim_orbit_target.setToolTip(
            "What to orbit around. 'Selected objects' uses the selection in "
            "the legend tab."
        )
        layout.addRow("Around:", self.anim_orbit_target)

        self.anim_orbit_axis = QtWidgets.QComboBox()
        self.anim_orbit_axis.addItems(list(ORBIT_AXES))
        self.anim_orbit_axis.setToolTip(
            "Axis to rotate about. 'Up' is the scene's up direction and gives "
            "a turntable rotation whichever way you are looking."
        )
        layout.addRow("Axis:", self.anim_orbit_axis)

        self.anim_orbit_turns = QtWidgets.QDoubleSpinBox()
        self.anim_orbit_turns.setRange(-100.0, 100.0)
        self.anim_orbit_turns.setSingleStep(0.25)
        self.anim_orbit_turns.setDecimals(2)
        self.anim_orbit_turns.setValue(1.0)
        self.anim_orbit_turns.setToolTip(
            "Number of turns; negative values orbit the other way."
        )
        layout.addRow("Turns:", self.anim_orbit_turns)

        self.anim_orbit_duration = QtWidgets.QDoubleSpinBox()
        self.anim_orbit_duration.setRange(0.1, 3600.0)
        self.anim_orbit_duration.setSingleStep(0.5)
        self.anim_orbit_duration.setDecimals(1)
        self.anim_orbit_duration.setSuffix(" s")
        self.anim_orbit_duration.setValue(6.0)
        self.anim_orbit_duration.valueChanged.connect(
            lambda *_: self._anim_update_summary()
        )
        layout.addRow("Duration:", self.anim_orbit_duration)

        self.anim_orbit_recenter = QtWidgets.QCheckBox("Re-centre first")
        self.anim_orbit_recenter.setChecked(True)
        self.anim_orbit_recenter.setToolTip(
            "Frame the target before orbiting. Uncheck to orbit around "
            "whatever the camera is looking at right now, i.e. to animate "
            "exactly the view you have set up."
        )
        layout.addRow(self.anim_orbit_recenter)

        return page

    def _build_anim_keyframe_page(self):
        """Page for assembling a fly-through from captured views."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        hint = QtWidgets.QLabel("Move the camera, then add the view as a keyframe.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.anim_keyframe_list = QtWidgets.QListWidget()
        self.anim_keyframe_list.setToolTip(
            "The camera visits these views in order. Double-click one to jump "
            "back to it."
        )
        self.anim_keyframe_list.setMaximumHeight(140)
        self.anim_keyframe_list.currentRowChanged.connect(
            lambda *_: self._anim_sync_keyframe_editor()
        )
        self.anim_keyframe_list.itemDoubleClicked.connect(
            lambda *_: self._anim_goto_keyframe()
        )
        layout.addWidget(self.anim_keyframe_list)

        # Two rows of buttons - side by side they would push up the minimum
        # width of the whole controls window
        button_layout = QtWidgets.QGridLayout()
        buttons = [
            (
                "Add",
                "Capture the current view as a new keyframe.",
                self._anim_add_keyframe,
            ),
            (
                "Update",
                "Replace the selected keyframe with the current view.",
                self._anim_update_keyframe,
            ),
            (
                "Go to",
                "Set the camera to the selected keyframe.",
                self._anim_goto_keyframe,
            ),
            ("Remove", "Delete the selected keyframe.", self._anim_remove_keyframe),
        ]
        for i, (label, tooltip, callback) in enumerate(buttons):
            button = QtWidgets.QPushButton(label)
            button.setToolTip(tooltip)
            button.clicked.connect(callback)
            button_layout.addWidget(button, i // 2, i % 2)
        layout.addLayout(button_layout)

        # Settings for the move *into* the selected keyframe
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)

        self.anim_keyframe_duration = QtWidgets.QDoubleSpinBox()
        self.anim_keyframe_duration.setRange(0.1, 3600.0)
        self.anim_keyframe_duration.setSingleStep(0.5)
        self.anim_keyframe_duration.setDecimals(1)
        self.anim_keyframe_duration.setSuffix(" s")
        self.anim_keyframe_duration.setValue(2.0)
        self.anim_keyframe_duration.setToolTip(
            "How long the camera takes to get to the selected keyframe."
        )
        self.anim_keyframe_duration.valueChanged.connect(
            lambda value: self._anim_edit_keyframe("duration", value)
        )
        form.addRow("Move in:", self.anim_keyframe_duration)

        self.anim_keyframe_easing = QtWidgets.QComboBox()
        self.anim_keyframe_easing.addItems(list(EASINGS))
        self.anim_keyframe_easing.setCurrentText("in_out")
        self.anim_keyframe_easing.setToolTip(
            "How the move accelerates. 'in_out' eases out of the previous "
            "keyframe and into this one."
        )
        self.anim_keyframe_easing.currentTextChanged.connect(
            lambda value: self._anim_edit_keyframe("easing", value)
        )
        form.addRow("Easing:", self.anim_keyframe_easing)
        layout.addLayout(form)

        self.anim_keyframe_loop = QtWidgets.QCheckBox("Return to first keyframe")
        self.anim_keyframe_loop.setToolTip(
            "Add a final move back to the first keyframe so the animation "
            "loops seamlessly. Its duration is the one set on that keyframe."
        )
        self.anim_keyframe_loop.toggled.connect(lambda *_: self._anim_update_summary())
        self.anim_keyframe_loop.toggled.connect(
            lambda *_: self._anim_sync_keyframe_editor()
        )
        layout.addWidget(self.anim_keyframe_loop)

        self._anim_sync_keyframe_editor()
        return page

    def _build_anim_render_group(self):
        """The "Render" group: frame rate, size, output file and recording."""
        group = QtWidgets.QGroupBox("Render")
        layout = QtWidgets.QVBoxLayout(group)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)

        self.anim_fps = QtWidgets.QSpinBox()
        self.anim_fps.setRange(1, 240)
        self.anim_fps.setValue(30)
        self.anim_fps.setToolTip("Frames per second of the recording.")
        self.anim_fps.valueChanged.connect(lambda *_: self._anim_update_summary())
        form.addRow("FPS:", self.anim_fps)

        self.anim_supersample = QtWidgets.QSpinBox()
        self.anim_supersample.setRange(1, 4)
        self.anim_supersample.setValue(1)
        self.anim_supersample.setToolTip(
            "Anti-aliasing quality (see the screenshot tab). The cost is paid "
            "for every single frame, so 1 or 2 is usually the sweet spot."
        )
        form.addRow("Supersample:", self.anim_supersample)

        self.anim_format = QtWidgets.QComboBox()
        self.anim_format.addItems(list(OUTPUT_FORMATS))
        self.anim_format.setToolTip(
            "MP4 and GIF need `imageio` (plus `imageio-ffmpeg` for MP4). A PNG "
            "sequence writes one numbered image per frame into a folder and "
            "needs nothing extra."
        )
        self.anim_format.currentTextChanged.connect(self._anim_format_changed)
        form.addRow("Format:", self.anim_format)
        layout.addLayout(form)

        (
            self.anim_size_checkbox,
            self.anim_width_spinbox,
            self.anim_height_spinbox,
        ) = self.create_size_widgets(
            layout,
            "Record at a custom size. The canvas is resized for the recording "
            "and put back afterwards.",
            minimum=16,
        )

        layout.addWidget(QtWidgets.QLabel("File:"))
        self.anim_filename_edit = QtWidgets.QLineEdit("animation.mp4")
        self.anim_browse_button = QtWidgets.QPushButton("Browse...")
        self.anim_browse_button.clicked.connect(self._anim_browse)
        filename_layout = QtWidgets.QHBoxLayout()
        filename_layout.addWidget(self.anim_filename_edit)
        filename_layout.addWidget(self.anim_browse_button)
        layout.addLayout(filename_layout)

        # What the current settings add up to
        self.anim_summary_label = QtWidgets.QLabel("")
        self.anim_summary_label.setWordWrap(True)
        layout.addWidget(self.anim_summary_label)

        self.anim_record_button = QtWidgets.QPushButton("Record")
        self.anim_record_button.clicked.connect(self._anim_toggle_record)
        layout.addWidget(self.anim_record_button)

        self.anim_progress = QtWidgets.QProgressBar()
        self.anim_progress.setVisible(False)
        layout.addWidget(self.anim_progress)

        self.anim_status_label = QtWidgets.QLabel("")
        self.anim_status_label.setWordWrap(True)
        layout.addWidget(self.anim_status_label)

        return group

    # -------------------------------------------------------------- keyframes

    def _anim_add_keyframe(self):
        """Capture the current view as a keyframe."""
        self._anim_keyframes.append(
            {
                "state": self.viewer.get_view(),
                "duration": self.anim_keyframe_duration.value(),
                "easing": self.anim_keyframe_easing.currentText(),
            }
        )
        self._anim_refresh_keyframes(select=len(self._anim_keyframes) - 1)

    def _anim_update_keyframe(self):
        """Point the selected keyframe at the current view."""
        row = self.anim_keyframe_list.currentRow()
        if row < 0:
            return self._anim_set_status("No keyframe selected.")
        # N.B. no need to refresh the list - the labels don't show the view
        self._anim_keyframes[row]["state"] = self.viewer.get_view()
        self._anim_set_status(f"Keyframe {row + 1} updated.")

    def _anim_remove_keyframe(self):
        row = self.anim_keyframe_list.currentRow()
        if row < 0:
            return self._anim_set_status("No keyframe selected.")
        self._anim_keyframes.pop(row)
        self._anim_refresh_keyframes(select=min(row, len(self._anim_keyframes) - 1))

    def _anim_goto_keyframe(self):
        """Set the camera to the selected keyframe."""
        row = self.anim_keyframe_list.currentRow()
        if row < 0:
            return
        self._anim_stop_preview()
        self.viewer.set_view(self._anim_keyframes[row]["state"])

    def _anim_edit_keyframe(self, key, value):
        """Apply an edit from the duration/easing widgets."""
        row = self.anim_keyframe_list.currentRow()
        # Also fires while we are filling the widgets in `_anim_sync_keyframe_editor`
        if row < 0 or self._anim_syncing_keyframe:
            return
        self._anim_keyframes[row][key] = value
        # Only this one entry's label can have changed - re-building the list
        # here would also write back into the widget being edited
        self.anim_keyframe_list.item(row).setText(self._anim_keyframe_label(row))
        self._anim_update_summary()

    def _anim_keyframe_label(self, index):
        """Text for a keyframe's entry in the list."""
        if index == 0:
            return "1 · start"
        keyframe = self._anim_keyframes[index]
        return f"{index + 1} · {keyframe['duration']:.1f} s · {keyframe['easing']}"

    def _anim_refresh_keyframes(self, select=None):
        """Rebuild the keyframe list widget from `self._anim_keyframes`."""
        self.anim_keyframe_list.blockSignals(True)
        self.anim_keyframe_list.clear()
        for i in range(len(self._anim_keyframes)):
            self.anim_keyframe_list.addItem(self._anim_keyframe_label(i))
        if select is not None and 0 <= select < len(self._anim_keyframes):
            self.anim_keyframe_list.setCurrentRow(select)
        self.anim_keyframe_list.blockSignals(False)
        self._anim_sync_keyframe_editor()
        self._anim_update_summary()

    def _anim_sync_keyframe_editor(self):
        """Show the selected keyframe's settings in the editor widgets."""
        row = self.anim_keyframe_list.currentRow()
        # The first keyframe has nothing leading into it - unless we loop back
        # to it at the end, in which case its settings describe that last move
        editable = row > 0 or (row == 0 and self.anim_keyframe_loop.isChecked())

        self._anim_syncing_keyframe = True
        try:
            if 0 <= row < len(self._anim_keyframes):
                keyframe = self._anim_keyframes[row]
                self.anim_keyframe_duration.setValue(keyframe["duration"])
                self.anim_keyframe_easing.setCurrentText(keyframe["easing"])
            self.anim_keyframe_duration.setEnabled(editable)
            self.anim_keyframe_easing.setEnabled(editable)
        finally:
            self._anim_syncing_keyframe = False

    # --------------------------------------------------------- the animation

    def _anim_build(self):
        """Build an `Animation` from the current settings."""
        anim = Animation(self.viewer, fps=self.anim_fps.value())

        if self.anim_mode_dropdown.currentText() == "Orbit":
            objects = None
            if self.anim_orbit_target.currentText() == "Selected objects":
                objects = self.get_selected()
                if not objects:
                    raise ValueError("No objects selected in the legend.")
            anim.orbit(
                objects,
                turns=self.anim_orbit_turns.value(),
                duration=self.anim_orbit_duration.value(),
                axis=ORBIT_AXES[self.anim_orbit_axis.currentText()],
                recenter=self.anim_orbit_recenter.isChecked(),
            )
        else:
            if len(self._anim_keyframes) < 2:
                raise ValueError("Add at least two keyframes to animate between.")
            anim.start_at(self._anim_keyframes[0]["state"])
            for keyframe in self._anim_keyframes[1:]:
                anim.move_to(
                    keyframe["state"],
                    duration=keyframe["duration"],
                    easing=keyframe["easing"],
                )
            if self.anim_keyframe_loop.isChecked():
                first = self._anim_keyframes[0]
                anim.move_to(
                    first["state"],
                    duration=first["duration"],
                    easing=first["easing"],
                )

        return anim

    def _anim_duration(self):
        """Length of the animation the current settings describe (0 if none)."""
        if self.anim_mode_dropdown.currentText() == "Orbit":
            return self.anim_orbit_duration.value()

        keyframes = self._anim_keyframes
        if len(keyframes) < 2:
            return 0.0
        duration = sum(k["duration"] for k in keyframes[1:])
        if self.anim_keyframe_loop.isChecked():
            duration += keyframes[0]["duration"]
        return duration

    def _anim_update_summary(self):
        """Update the "X s, N frames" label under the settings."""
        duration = self._anim_duration()
        if not duration:
            self.anim_summary_label.setText("Nothing to animate yet.")
            return
        # N.B. the same rule the recording itself uses, not a rounding of our own
        n_frames = frame_count(duration, self.anim_fps.value())
        self.anim_summary_label.setText(f"{duration:.1f} s · {n_frames} frames")

    # ------------------------------------------------------------- preview

    def _anim_toggle_preview(self):
        if self._anim_preview is not None:
            return self._anim_stop_preview()

        try:
            anim = self._anim_build()
        except ValueError as e:
            return self._anim_set_status(str(e))

        self._anim_preview = anim
        anim.play(loop=True)
        self.anim_preview_button.setText("Stop preview")
        self._anim_set_status("Previewing...", timeout=None)

    def _anim_stop_preview(self):
        if self._anim_preview is None:
            return
        self._anim_preview.stop()
        self._anim_preview = None
        self.anim_preview_button.setText("Preview")
        self._anim_set_status("")

    # ------------------------------------------------------------- recording

    def _anim_toggle_record(self):
        if self._anim_recorder is not None:
            return self._anim_cancel_record()

        filename = self.anim_filename_edit.text().strip()
        if not filename:
            return self._anim_set_status("Please provide a file name.")

        self._anim_stop_preview()

        kwargs = dict(
            supersample=self.anim_supersample.value(),
            # Only a PNG sequence can carry the transparency
            alpha=not self._anim_suffix(),
            **self._size_kwargs(
                self.anim_size_checkbox,
                self.anim_width_spinbox,
                self.anim_height_spinbox,
            ),
        )

        try:
            anim = self._anim_build()
            self._anim_recorder = anim.recorder(Path(filename).expanduser(), **kwargs)
        except BaseException as e:
            self._anim_recorder = None
            return self._anim_set_status(f"Could not start recording: {e}")

        self.anim_record_button.setText("Cancel")
        self.anim_preview_button.setEnabled(False)
        self.anim_progress.setRange(0, self._anim_recorder.n_frames)
        self.anim_progress.setValue(0)
        self.anim_progress.setVisible(True)
        self._anim_set_status("Recording...", timeout=None)

        # One frame per timer tick keeps the event loop - and with it the
        # cancel button and the viewer itself - alive while we render
        self._anim_record_timer = QtCore.QTimer(self)
        self._anim_record_timer.setInterval(0)
        self._anim_record_timer.timeout.connect(self._anim_record_step)
        self._anim_record_timer.start()

    def _anim_record_step(self):
        """Render one frame; wrap up when there are none left."""
        recorder = self._anim_recorder
        if recorder is None:
            return

        try:
            more = recorder.step()
        except BaseException as e:
            return self._anim_cancel_record(f"Recording failed: {e}")

        self.anim_progress.setValue(recorder.frame)
        if more:
            return

        self._anim_stop_record_timer()
        self._anim_recorder = None
        try:
            result = recorder.finish()
        except BaseException as e:
            self._anim_reset_record_gui()
            return self._anim_set_status(f"Could not write the animation: {e}")

        self._anim_reset_record_gui()
        self._anim_set_status(f"Saved {Path(result).resolve()}", timeout=10000)

    def _anim_cancel_record(self, status="Recording cancelled."):
        """Abort a running recording and remove what was written so far."""
        self._anim_stop_record_timer()
        if self._anim_recorder is not None:
            self._anim_recorder.cancel()
            self._anim_recorder = None
        self._anim_reset_record_gui()
        self._anim_set_status(status)

    def _anim_stop_record_timer(self):
        if self._anim_record_timer is not None:
            self._anim_record_timer.stop()
            self._anim_record_timer.deleteLater()
            self._anim_record_timer = None

    def _anim_reset_record_gui(self):
        self.anim_record_button.setText("Record")
        self.anim_preview_button.setEnabled(True)
        self.anim_progress.setVisible(False)

    # ------------------------------------------------------------- odds & ends

    def _anim_suffix(self):
        """File suffix for the selected output format ('' for a PNG folder)."""
        return OUTPUT_FORMATS[self.anim_format.currentText()]

    def _anim_format_changed(self, label):
        """Keep the file name's extension in step with the chosen format."""
        filename = Path(self.anim_filename_edit.text().strip() or "animation")
        # A PNG sequence goes into a folder, i.e. a name without an extension
        self.anim_filename_edit.setText(
            str(filename.with_suffix(OUTPUT_FORMATS[label]))
        )

    def _anim_browse(self):
        """Pick the output file (or folder, for a PNG sequence)."""
        current = Path(self.anim_filename_edit.text().strip() or "animation")
        label = self.anim_format.currentText()
        suffix = self._anim_suffix()

        if suffix:
            self.open_file_dialog(
                "Save animation",
                self.anim_filename_edit.setText,
                initial=current,
                filters=[f"{label} (*{suffix})", "All files (*)"],
                # Without this a name typed without an extension would quietly
                # be written as a PNG sequence instead
                default_suffix=suffix.lstrip("."),
            )
        else:
            self.open_file_dialog(
                "Folder for the PNG sequence",
                self.anim_filename_edit.setText,
                initial=current.parent,
                directory=True,
            )

    def _anim_set_status(self, text, timeout=5000):
        """Show a message under the record button, cleared after `timeout` ms."""
        self.set_status(self.anim_status_label, text, timeout=timeout)

    def _anim_cleanup(self):
        """Stop preview and recording - called when the controls close."""
        self._anim_stop_preview()
        # N.B. guarded so that closing the panel does not leave a stray
        # "Recording cancelled." behind
        if self._anim_recorder is not None:
            self._anim_cancel_record()
