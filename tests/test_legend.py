"""Tests for the legend in the (Qt) controls panel."""

import os

import pytest
import numpy as np
import octarine as oc

# N.B. the Qt binding is pinned in tests/conftest.py, which has to run before
# anything else can load one - see the comment there. Importing a submodule
# (and allowing a plain ImportError) is deliberate: when another binding has
# got in first, `PySide6` itself imports fine and only its extension modules
# fail to load.
pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)


@pytest.fixture()
def viewer():
    v = oc.Viewer(offscreen=True, size=(200, 150))
    yield v
    v.close()


@pytest.fixture()
def controls(viewer):
    """A shown controls panel, so legend refreshes actually run."""
    from PySide6 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from octarine.controls import Controls

    c = Controls(viewer)
    c.show()
    yield c
    c.close()


def _points(n=3):
    return np.random.rand(n, 3).astype(np.float32)


def _ids(controls):
    return [controls.legend.item(i)._id for i in range(controls.legend.count())]


def test_legend_tracks_objects(viewer, controls):
    """Rows must follow objects being added and removed."""
    viewer.add(_points(), name="a")
    viewer.add(_points(), name="b")
    controls.flush_legend()
    assert _ids(controls) == ["a", "b"]

    viewer.remove_objects(["a"])
    controls.flush_legend()
    assert _ids(controls) == ["b"]
    # The bookkeeping must not hold on to the removed row
    assert set(controls._legend_rows) == {"b"}

    viewer.clear()
    controls.flush_legend()
    assert _ids(controls) == []


def test_legend_refreshes_are_coalesced(viewer, controls):
    """Adding N objects must cost one refresh, not N - see `update_legend`."""
    from PySide6 import QtWidgets

    refreshes = []
    refresh = controls._refresh_legend

    def counted_refresh():
        refreshes.append(1)
        refresh()

    controls._refresh_legend = counted_refresh

    for i in range(25):
        viewer.add(_points(), name=f"obj{i}")
    assert refreshes == []  # nothing ran while the adds were in flight

    # Returning to the event loop is what actually drives the refresh
    QtWidgets.QApplication.instance().processEvents()
    assert refreshes == [1]
    assert controls.legend.count() == 25

    # ... and a flush with nothing pending must not redo the work
    controls.flush_legend()
    assert refreshes == [1]


def test_legend_syncs_color_visibility_and_selection(viewer, controls):
    """A row must show its object's color, visibility and selection state."""
    viewer.add(_points(), name="a")
    viewer.add(_points(), name="b")
    viewer.set_colors({"a": (1.0, 0.0, 0.0, 1.0)})
    viewer.hide_objects(["b"])
    viewer.selected = ["a"]
    controls.flush_legend()

    row_a = controls._legend_rows["a"].widget
    row_b = controls._legend_rows["b"].widget

    assert "255,0,0" in row_a._legend_control.styleSheet().replace(" ", "")
    assert row_a._legend_checkbox.isChecked()
    assert not row_b._legend_checkbox.isChecked()
    assert row_a._legend_label._selected
    assert not row_b._legend_label._selected


def test_legend_row_checkbox_toggles_visibility(viewer, controls):
    """Unticking a row must hide the object it stands for."""
    viewer.add(_points(), name="a")
    controls.flush_legend()

    row = controls._legend_rows["a"].widget
    row._legend_checkbox.setChecked(False)
    assert viewer.visible == []

    row._legend_checkbox.setChecked(True)
    assert viewer.visible == ["a"]


def test_legend_groups_are_lazy(viewer, controls):
    """A group's member rows must only be built once it is expanded."""
    viewer.add(_points(), name="g1", group="G")
    viewer.add(_points(), name="g2", group="G")
    controls.flush_legend()

    assert _ids(controls) == ["group::G"]
    group = controls._legend_rows["group::G"].widget
    assert group._member_rows == {}
    assert group._group_label.text() == "G (2)"

    group._group_header.setChecked(True)  # expand
    assert set(group._member_rows) == {"g1", "g2"}


def test_legend_group_membership_updates_in_place(viewer, controls):
    """Members added/removed must edit the group's rows, not rebuild them."""
    from PySide6 import QtCore

    viewer.add(_points(), name="g1", group="G")
    viewer.add(_points(), name="g2", group="G")
    controls.flush_legend()
    group = controls._legend_rows["group::G"].widget
    group._group_header.setChecked(True)
    kept = group._member_rows["g1"]

    viewer.add(_points(), name="g3", group="G")
    controls.flush_legend()
    assert group._member_names == ["g1", "g2", "g3"]
    assert set(group._member_rows) == {"g1", "g2", "g3"}
    assert group._group_label.text() == "G (3)"
    assert group._member_rows["g1"] is kept  # existing rows are left alone
    # Recoloring the group must act on the members it has now
    assert sorted(group._group_control._id) == ["g1", "g2", "g3"]

    viewer.remove_objects(["g3"])
    controls.flush_legend()
    assert set(group._member_rows) == {"g1", "g2"}
    assert group._group_label.text() == "G (2)"

    # Group checkbox reflects partial visibility
    viewer.hide_objects(["g1"])
    controls.flush_legend()
    assert group._group_checkbox.checkState() == QtCore.Qt.PartiallyChecked

    # ... and clicking through it drives every member. N.B. the partial
    # state is indicator-only, so a click has to be used rather than
    # `setChecked` - see `toggle_group_visibility`.
    group._group_checkbox.click()
    assert sorted(viewer.visible) == ["g1", "g2"]
    assert all(r._legend_checkbox.isChecked() for r in group._member_rows.values())


def test_legend_group_member_rows_follow_member_order(viewer, controls):
    """Rows must be laid out in member order, not just appended as they arrive.

    Removing an object and adding it back lands it at the end of the group's
    members while its row is still sitting where it was, so the ordering pass
    has to cover the rows that were already there.
    """
    for name in ("g1", "g2", "g3"):
        viewer.add(_points(), name=name, group="G")
    controls.flush_legend()
    group = controls._legend_rows["group::G"].widget
    group._group_header.setChecked(True)  # expand

    def layout_order():
        layout = group._group_content.layout()
        return [layout.itemAt(i).widget().objectName() for i in range(layout.count())]

    assert layout_order() == group._member_names == ["g1", "g2", "g3"]

    # Re-adding moves g1 to the end of the group
    viewer.remove_objects(["g1"])
    viewer.add(_points(), name="g1", group="G")
    controls.flush_legend()
    assert group._member_names == ["g2", "g3", "g1"]
    assert layout_order() == ["g2", "g3", "g1"]

    # A plain append still goes where it belongs ...
    viewer.add(_points(), name="g4", group="G")
    controls.flush_legend()
    assert layout_order() == group._member_names == ["g2", "g3", "g1", "g4"]

    # ... and so does a removal from the middle
    viewer.remove_objects(["g3"])
    controls.flush_legend()
    assert layout_order() == group._member_names == ["g2", "g1", "g4"]

    # No row may be left in the layout twice, or orphaned from the bookkeeping
    assert sorted(layout_order()) == sorted(group._member_rows)


def test_legend_group_checkbox_does_not_get_stuck(viewer, controls):
    """Toggling a group must leave the tri-state in step with its members.

    A group toggle updates its member rows itself, so it has to invalidate the
    snapshot a refresh compares against - otherwise the next change that
    happens to restore the snapshotted state is skipped as a no-op.
    """
    from PySide6 import QtCore

    viewer.add(_points(), name="g1", group="G")
    viewer.add(_points(), name="g2", group="G")
    controls.flush_legend()
    group = controls._legend_rows["group::G"].widget
    group._group_header.setChecked(True)  # expand

    viewer.hide_objects(["g1"])
    controls.flush_legend()
    assert group._group_checkbox.checkState() == QtCore.Qt.PartiallyChecked

    group._group_checkbox.click()  # show everything again
    controls.flush_legend()
    assert group._group_checkbox.checkState() == QtCore.Qt.Checked
    assert sorted(viewer.visible) == ["g1", "g2"]

    # Back to where we started: the group must notice a second time
    group._member_rows["g1"]._legend_checkbox.setChecked(False)
    controls.flush_legend()
    assert sorted(viewer.visible) == ["g2"]
    assert group._group_checkbox.checkState() == QtCore.Qt.PartiallyChecked


def test_legend_filter(viewer, controls):
    """Filtering must match group names and members, and be undoable."""
    viewer.add(_points(), name="apple")
    viewer.add(_points(), name="banana")
    viewer.add(_points(), name="cherry", group="fruit")
    viewer.add(_points(), name="date", group="fruit")
    controls.flush_legend()
    group = controls._legend_rows["group::fruit"].widget

    def hidden():
        return {
            controls.legend.item(i)._id: controls.legend.item(i).isHidden()
            for i in range(controls.legend.count())
        }

    # A member match expands its group and shows only the matching rows
    controls.legend_filter.setText("cherry")
    assert hidden() == {"apple": True, "banana": True, "group::fruit": False}
    assert group._member_rows["cherry"].isVisible()
    assert not group._member_rows["date"].isVisible()

    # A group-name match wins and keeps the group collapsed
    controls.legend_filter.setText("fruit")
    assert hidden() == {"apple": True, "banana": True, "group::fruit": False}
    assert not group._group_content.isVisible()

    # A single entry
    controls.legend_filter.setText("app")
    assert hidden() == {"apple": False, "banana": True, "group::fruit": True}

    # Clearing restores everything
    controls.legend_filter.setText("")
    assert not any(hidden().values())
    # The group is collapsed again, so its rows are not on screen - but none
    # of them is left explicitly hidden by the filter.
    assert not any(r.isHidden() for r in group._member_rows.values())


def test_legend_filter_survives_new_objects(viewer, controls):
    """An active filter must still apply to rows added after it was typed."""
    viewer.add(_points(), name="apple")
    controls.flush_legend()
    controls.legend_filter.setText("ban")

    viewer.add(_points(), name="banana")
    controls.flush_legend()
    by_id = {
        controls.legend.item(i)._id: controls.legend.item(i)
        for i in range(controls.legend.count())
    }
    assert by_id["apple"].isHidden()
    assert not by_id["banana"].isHidden()


def test_legend_hover_target_resolves_through_subwidgets(viewer, controls):
    """Hovering any control in a row must highlight that row's object(s)."""
    viewer.add(_points(), name="a")
    viewer.add(_points(), name="g1", group="G")
    controls.flush_legend()

    row = controls._legend_rows["a"].widget
    assert controls._find_hover_target(row._legend_checkbox) == "a"
    assert controls._find_hover_target(row._legend_control) == "a"

    group = controls._legend_rows["group::G"].widget
    assert controls._find_hover_target(group._group_checkbox) is group._member_names


def test_legend_skips_refreshes_while_hidden(viewer, controls):
    """A hidden panel must do no work, then catch up when it is shown."""
    from PySide6 import QtWidgets

    controls.hide()
    viewer.add(_points(), name="a")
    QtWidgets.QApplication.instance().processEvents()
    assert controls.legend.count() == 0
    assert controls._legend_dirty

    controls.show()
    assert not controls._legend_dirty
    assert _ids(controls) == ["a"]


def test_legend_rows_are_sized_for_the_font_they_are_shown_in(viewer, controls):
    """Row heights must track the legend's font, not the application default.

    The hint is measured once, on a row that is not in the list yet, and
    reused for every row - so it has to be taken with the legend's own font
    and thrown away when that font changes.
    """
    from PySide6 import QtGui

    viewer.add(_points(), name="a")
    controls.flush_legend()
    row = controls._legend_rows["a"].widget
    # Measured before the row was in the list, but it must match the row as
    # it is actually laid out now that it is.
    assert controls._legend_row_hint.height() == row.sizeHint().height()
    small = controls._legend_row_hint.height()

    font = QtGui.QFont(controls.font())
    font.setPointSize(font.pointSize() + 14)
    controls.setFont(font)
    assert controls._legend_row_hint is None  # the old metrics are dropped

    viewer.add(_points(), name="b")
    controls.flush_legend()
    bigger_row = controls._legend_rows["b"].widget
    assert controls._legend_row_hint.height() > small
    assert controls._legend_row_hint.height() == bigger_row.sizeHint().height()


def test_legend_bulk_insert_leaves_list_visible(viewer, controls):
    """The bulk-insert path hides the list; it must put it back."""
    for i in range(controls._LEGEND_BULK_INSERT + 5):
        viewer.add(_points(), name=f"obj{i}")
    controls.flush_legend()

    assert controls.legend.isVisible()
    assert controls.legend.count() == controls._LEGEND_BULK_INSERT + 5

    # ... and the rows it added must have had their hover filters installed,
    # which the bulk path defers until the rows are in place
    from PySide6 import QtCore, QtGui

    row = controls._legend_rows["obj0"].widget
    QtCore.QCoreApplication.sendEvent(
        row._legend_checkbox, QtGui.QEnterEvent(*(QtCore.QPointF(),) * 3)
    )
    assert controls._legend_hover_target == "obj0"


def test_controls_explain_a_clashing_qt_binding():
    """Importing the controls next to another Qt binding must say so.

    Only one Qt binding can be loaded per process, and the failure otherwise
    surfaces as a bare "symbol not found" naming libraries the caller never
    asked for. Run in a subprocess: loading PyQt6 here would break every other
    Qt test in this session, which is the very thing being tested.
    """
    import subprocess
    import sys

    pytest.importorskip("PyQt6")

    done = subprocess.run(
        [sys.executable, "-c", "import PyQt6.QtWidgets; import octarine.controls"],
        capture_output=True,
        text=True,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert done.returncode != 0
    assert "PyQt6 is already loaded in this process" in done.stderr
