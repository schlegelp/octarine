"""Merged objects must behave exactly like unmerged ones.

`test_merge.py` covers the merging primitives. This covers the promise the
viewer makes on top of them: that turning merging on changes how many pygfx
objects the renderer walks and nothing else. Most tests here therefore build
the same scene twice - once merged, once not - and assert that the two agree,
either on what they draw or on what the viewer reports.

"""

import pytest
import numpy as np
import trimesh as tm
import pygfx as gfx

import octarine as oc
from octarine.merge import sub_visual_at
from conftest import bands as _bands, frame_viewer

RED, LIME, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)

# Alpha modes that draw an opaque color opaquely (see `_OPAQUE_ALPHA_MODES`)
_OPAQUE_MODES = ("auto", "opaque", "solid")

_LINE = np.zeros((5, 3), dtype=np.float32)
_LINE[:, 0] = np.linspace(-70, 70, 5)
_POINT = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)


@pytest.fixture()
def qt_controls():
    """A merged viewer with a shown controls panel, so legend refreshes run.

    Built the way `tests/test_legend.py` does it: the Qt binding is pinned in
    `tests/conftest.py` and a `QApplication` has to exist before `Controls`.

    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from PySide6 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from octarine.controls import Controls

    v = _viewer(merge=True)
    controls = Controls(v)
    controls.show()
    yield v, controls
    controls.close()
    v.close()


@pytest.fixture()
def qt_controls_unmerged():
    """The same, but with merging off, for exercising the toggle itself."""
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from PySide6 import QtWidgets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from octarine.controls import Controls

    v = _viewer(merge=False)
    controls = Controls(v)
    controls.show()
    yield v, controls
    controls.close()
    v.close()


def _viewer(merge=False, **kwargs):
    """An offscreen viewer set up so that rendered bands are crisp."""
    v = oc.Viewer(offscreen=True, size=(200, 200), merge=merge, **kwargs)
    v.renderer.pixel_ratio = 1
    v.renderer.ppaa = "none"
    return v


def _frame(v):
    """Settle the scene, then point the camera at a fixed window on it.

    Done *after* the objects are in, and with `center=False` on every add:
    adding asks the viewer to re-center, which runs on the next refresh and
    would throw this away.

    """
    v._refresh_scene()
    return frame_viewer(v)


def _colors(v):
    """Just the band colors, in the order they appear down the image.

    Heights are left out: what these tests are about is which object is drawn
    and in what color, and a band's exact pixel height depends on the renderer.

    """
    return [tuple(int(c) for c in color) for _, color in _bands(v)]


def _bars(v, colors=("red", "lime", "blue"), shader="basic"):
    """Add one horizontal bar per color and settle the scene.

    `bar0` is placed highest so that the bars come out of `_bands` in name
    order - the image's first row is the top of the scene.

    """
    step = 160 // (len(colors) + 1)
    for i, color in enumerate(colors):
        mesh = tm.creation.box((160, 20, 20))
        mesh.vertices = mesh.vertices + (0, 60 - i * step, 0)
        v.add_mesh(mesh, name=f"bar{i}", color=color, shader=shader, center=False)
    return _frame(v)


# ------------------------------------------------------------------ basics


def test_merging_reduces_visuals_but_not_objects():
    """Merging must change the object count not at all and the visual count a lot."""
    plain = _bars(_viewer(merge=False))
    merged = _bars(_viewer(merge=True))

    assert list(plain.objects) == list(merged.objects) == ["bar0", "bar1", "bar2"]
    assert len(plain.visuals) == 3
    assert len(merged.visuals) == 1

    stats = merged.merge_stats
    assert (stats.objects, stats.visuals, stats.members) == (3, 1, 3)
    assert stats.split_outs == 0

    plain.close()
    merged.close()


def test_merged_scene_renders_identically():
    """The whole point: same picture, fewer objects."""
    plain = _bars(_viewer(merge=False))
    merged = _bars(_viewer(merge=True))

    # N.B. non-empty: an empty render would make this pass for the wrong reason
    assert _bands(plain) == [(20, RED), (20, LIME), (20, BLUE)]
    assert _bands(merged) == _bands(plain)

    plain.close()
    merged.close()


def test_merge_off_by_default():
    """Merging must not happen unless it was asked for."""
    v = _bars(_viewer())
    assert len(v.visuals) == 3
    assert v.merge_stats.merged == 0
    v.close()


@pytest.mark.parametrize("threshold,expected", [(2, 1), (3, 1), (4, 3)])
def test_auto_merge_threshold(threshold, expected):
    """An integer policy must only start merging once the canvas is that full."""
    v = _bars(_viewer(merge=threshold))
    assert len(v.visuals) == expected
    v.close()


def test_merge_false_keeps_an_object_out():
    """`add(merge=False)` must leave that object as a visual of its own."""
    v = _viewer(merge=True)
    # Top to bottom: bar0 (red), the excluded one (blue), bar1 (lime)
    for i, color in enumerate(("red", "lime")):
        mesh = tm.creation.box((160, 20, 20))
        mesh.vertices = mesh.vertices + (0, 40 - i * 80, 0)
        v.add_mesh(mesh, name=f"bar{i}", color=color, shader="basic", center=False)

    loner = tm.creation.box((160, 20, 20))
    v.add_mesh(
        loner, name="loner", color="blue", shader="basic", merge=False, center=False
    )
    _frame(v)

    # The two mergeable bars merged; the excluded one stands on its own
    assert v.merge_stats.members == 2
    assert type(v.objects["loner"][0]).__name__ == "Mesh"
    assert _colors(v) == [RED, BLUE, LIME]
    v.close()


# ------------------------------------------------------- per-object control


def test_hide_and_unhide_round_trip():
    """Hiding a merged object must take it off the screen and nothing else."""
    v = _bars(_viewer(merge=True))
    everything = _bands(v)
    assert everything == [(20, RED), (20, LIME), (20, BLUE)]

    v.hide_objects("bar1")
    assert v.visible == ["bar0", "bar2"]
    assert v.invisible == ["bar1"]
    assert _colors(v) == [RED, BLUE]

    v.unhide_objects("bar1")
    assert v.visible == ["bar0", "bar1", "bar2"]
    assert _bands(v) == everything
    # Hiding is not a reason to come apart
    assert v.merge_stats.split_outs == 0
    v.close()


def test_set_colors_per_object():
    """Recoloring one merged object must leave the others alone."""
    v = _bars(_viewer(merge=True))

    v.set_colors({"bar1": (1, 0, 1, 1)})
    assert _colors(v) == [RED, (255, 0, 255), BLUE]
    assert v.merge_stats.split_outs == 0
    v.close()


def test_set_colors_everything_keeps_the_merge():
    """Coloring the whole canvas must not take it apart.

    `Viewer.set_colors` also sets `alpha_mode`, which a merged visual can only
    hold once - the bulk path exists so that does not split every object on
    the canvas. See `MergeManager.bulk_shared_write`.

    """
    v = _bars(_viewer(merge=True))
    v.set_colors("lime")

    assert v.merge_stats.split_outs == 0
    assert len(v.visuals) == 1
    assert _colors(v) == [LIME] * 3
    v.close()


def test_selection_highlight_round_trip():
    """Selecting and deselecting must restore the object's own color."""
    v = _bars(_viewer(merge=True))
    before = _bands(v)

    v.selected = ["bar1"]
    assert _colors(v)[1] == (255, 255, 0)  # highlight_color

    v.selected = []
    assert _bands(v) == before
    assert v.merge_stats.split_outs == 0
    v.close()


def test_highlight_objects_round_trip():
    """Hover highlighting must brighten one object and come back off cleanly."""
    v = _bars(_viewer(merge=True))
    before = _bands(v)

    v.highlight_objects("bar2")
    assert v.highlighted == ["bar2"]
    assert _bands(v) != before

    v.unhighlight_objects()
    assert v.highlighted == []
    assert _bands(v) == before
    v.close()


def test_pinned_objects_ignore_changes():
    """A pinned merged object must ignore hiding, as an unmerged one does."""
    v = _bars(_viewer(merge=True))
    v.pin_objects("bar1")

    v.hide_objects(["bar0", "bar1", "bar2"])
    assert v.visible == ["bar1"]

    v.unpin_objects()
    v.hide_objects("bar1")
    assert v.visible == []
    v.close()


def test_member_bounds_match_unmerged():
    """A merged object must report the same bounds as an unmerged one.

    `Viewer.get_bounds` covers visible *and* invisible objects, so this has to
    hold for a hidden object too - whose vertices are NaN in the merged buffer.

    """
    plain = _bars(_viewer(merge=False))
    merged = _bars(_viewer(merge=True))

    for name in ("bar0", "bar1", "bar2"):
        np.testing.assert_allclose(
            merged.get_bounds(name), plain.get_bounds(name), atol=1e-5
        )

    merged.hide_objects("bar1")
    np.testing.assert_allclose(
        merged.get_bounds("bar1"), plain.get_bounds("bar1"), atol=1e-5
    )

    plain.close()
    merged.close()


# ------------------------------------------------------------ splitting out


def test_styling_a_subset_splits_it_out():
    """A style a merged visual cannot hold per member must split that member."""
    v = _bars(_viewer(merge=True), shader="phong")
    assert len(v.visuals) == 1

    v.set_silhouette(3, objects=["bar1"])

    assert v.merge_stats.split_outs == 1
    assert len(v.visuals) == 2
    # ... and the canvas still holds the same three objects
    assert list(v.objects) == ["bar0", "bar1", "bar2"]
    v.close()


def test_styling_everything_keeps_the_merge():
    """The same style applied to all of them needs no splitting."""
    v = _bars(_viewer(merge=True), shader="phong")
    v.set_silhouette(3)

    assert v.merge_stats.split_outs == 0
    assert len(v.visuals) == 1
    v.close()


def test_unmerge_objects_restores_plain_visuals():
    """Unmerging must leave a canvas indistinguishable from an unmerged one."""
    plain = _bars(_viewer(merge=False))
    v = _bars(_viewer(merge=True))

    v.unmerge_objects()

    assert v.merge_stats.merged == 0
    assert len(v.visuals) == 3
    assert all(isinstance(x[0], gfx.Mesh) for x in v.objects.values())
    assert _bands(v) == _bands(plain)

    plain.close()
    v.close()


def test_remove_merged_object():
    """Removing a merged object must take it off the canvas and the screen."""
    v = _bars(_viewer(merge=True))

    v.remove_objects("bar1")

    assert list(v.objects) == ["bar0", "bar2"]
    assert _colors(v) == [RED, BLUE]
    v.close()


def test_pop_removes_merged_objects():
    """`Viewer.pop` hands `remove_objects` whatever `Viewer.objects` holds."""
    v = _bars(_viewer(merge=True))

    v.pop(1)
    assert list(v.objects) == ["bar0", "bar1"]

    v.pop(2)
    assert list(v.objects) == []
    v.close()


def test_clear_forgets_merge_bookkeeping():
    """A cleared canvas must not hold on to handles or the old ordering."""
    v = _bars(_viewer(merge=True))
    v.clear()

    assert list(v.objects) == []
    assert v.merge_stats == oc.handles.MergeStats()

    # An object reusing a name must start at the front of the order again
    _bars(v, colors=("blue", "lime", "red"))
    assert list(v.objects) == ["bar0", "bar1", "bar2"]
    assert _colors(v) == [BLUE, LIME, RED]
    v.close()


def test_objects_keep_their_order_through_a_split():
    """`Viewer.objects` is documented as being in order of addition.

    Splitting puts the new visual on the end of the scene, so without the
    order bookkeeping the split-out object would jump to the back.

    """
    v = _bars(_viewer(merge=True), shader="phong")
    v.set_silhouette(3, objects=["bar1"])

    assert v.merge_stats.split_outs == 1
    assert list(v.objects) == ["bar0", "bar1", "bar2"]
    v.close()


def test_hidden_state_survives_a_rebuild():
    """A member hidden before a rebuild must still be hidden after it.

    Rebuilding reads the merged buffers as they stand, and hiding overwrites
    a member's positions with NaN - so without the stash a hidden member would
    either reappear or be baked in as NaN for good. This is the invariant the
    whole feature leans on.

    """
    v = _bars(_viewer(merge=True), shader="phong")
    v.hide_objects("bar1")
    hidden = _bands(v)

    # Splitting another object out rebuilds the merged visual bar1 lives in
    v.set_silhouette(3, objects=["bar0"])

    assert v.invisible == ["bar1"]
    assert "bar1" not in v.visible

    # ... and unhiding still puts it back where it belongs
    v.set_silhouette(0, objects=["bar0"])
    v.unhide_objects("bar1")
    assert len(_bands(v)) == len(hidden) + 1
    v.close()


# ----------------------------------------------------------------- picking


def test_pick_resolves_to_the_right_object():
    """A pick on a merged visual must report the object, not the visual."""
    v = _bars(_viewer(merge=True))
    (merged,) = v.visuals

    found = {
        sub_visual_at(merged, {"face_index": s.face_offset}).object_id
        for s in merged._sub_visuals
    }
    assert found == {"bar0", "bar1", "bar2"}
    v.close()


def test_double_click_action_hits_one_object():
    """The viewer's object-event path must act on the picked object alone."""
    from octarine.viewer import handle_object_event

    v = _bars(_viewer(merge=True))
    (merged,) = v.visuals
    sub = merged._sub_visuals[1]

    class _Event:
        pick_info = {"world_object": merged, "face_index": sub.face_offset}

    handle_object_event(_Event(), v, ["hide"])

    assert v.invisible == ["bar1"]
    v.close()


# ------------------------------------------------------------------ legend


def test_legend_lists_merged_objects(qt_controls):
    """The legend must show one row per object, not one per merged visual."""
    v, controls = qt_controls
    _bars(v)
    controls.flush_legend()

    rows = [controls.legend.item(i)._id for i in range(controls.legend.count())]
    assert rows == ["bar0", "bar1", "bar2"]

    # ... and its visibility toggle must act on that object alone
    row = controls._legend_rows["bar1"].widget
    row._legend_checkbox.setChecked(False)
    assert v.visible == ["bar0", "bar2"]
    assert _colors(v) == [RED, BLUE]


# ------------------------------------------------------- lines and points


def _line_bars(v, colors=("red", "lime", "blue"), width=6):
    """One horizontal line per color, topmost first, in world-space widths."""
    step = 160 // (len(colors) + 1)
    for i, color in enumerate(colors):
        pts = np.zeros((5, 3), dtype=np.float32)
        pts[:, 0] = np.linspace(-70, 70, 5)
        pts[:, 1] = 60 - i * step
        v.add_lines(
            pts,
            name=f"ln{i}",
            color=color,
            linewidth=width,
            linewidth_space="world",
            center=False,
        )
    return _frame(v)


def _point_rows(v, colors=("red", "lime", "blue"), size=12):
    """One point per color, topmost first, in world-space sizes."""
    step = 160 // (len(colors) + 1)
    for i, color in enumerate(colors):
        v.add_points(
            np.array([[0, 60 - i * step, 0]], dtype=np.float32),
            name=f"pt{i}",
            color=color,
            size=size,
            size_space="world",
            center=False,
        )
    return _frame(v)


@pytest.mark.parametrize("build,names", [(_line_bars, "ln"), (_point_rows, "pt")])
def test_lines_and_points_merge_and_render(build, names):
    """Lines and points must merge and draw exactly as they did apart."""
    plain = build(_viewer(merge=False))
    merged = build(_viewer(merge=True))

    assert len(plain.visuals) == 3
    assert len(merged.visuals) == 1
    assert _colors(plain) == [RED, LIME, BLUE]  # i.e. it really drew something
    assert _colors(merged) == _colors(plain)

    plain.close()
    merged.close()


@pytest.mark.parametrize("build,prefix", [(_line_bars, "ln"), (_point_rows, "pt")])
def test_lines_and_points_per_object_control(build, prefix):
    """Hiding and recoloring must work per object for lines and points too."""
    v = build(_viewer(merge=True))
    everything = _colors(v)
    assert everything == [RED, LIME, BLUE]

    v.hide_objects(f"{prefix}1")
    assert v.visible == [f"{prefix}0", f"{prefix}2"]
    assert _colors(v) == [RED, BLUE]

    v.unhide_objects(f"{prefix}1")
    assert _colors(v) == everything

    v.set_colors({f"{prefix}1": (1, 0, 1, 1)})
    assert _colors(v) == [RED, (255, 0, 255), BLUE]
    assert v.merge_stats.split_outs == 0
    v.close()


def test_merged_lines_keep_their_own_widths():
    """Differing line widths must be absorbed, not averaged away."""
    v = _viewer(merge=True)
    for i, (color, width) in enumerate((("red", 4), ("lime", 10))):
        pts = np.zeros((5, 3), dtype=np.float32)
        pts[:, 0] = np.linspace(-70, 70, 5)
        pts[:, 1] = 40 - i * 80
        v.add_lines(
            pts,
            name=f"ln{i}",
            color=color,
            linewidth=width,
            linewidth_space="world",
            center=False,
        )
    _frame(v)

    assert len(v.visuals) == 1
    assert [height for height, _ in _bands(v)] == [4, 10]
    v.close()


# -------------------------------------------------------------- stability


def test_idle_refresh_does_not_rebuild_merged_visuals():
    """A refresh with nothing new to merge must leave the scene alone.

    `merge_visuals` buckets two already-merged visuals that share a merge key
    together, so without a guard an idle refresh would decompose and rebuild
    them - replacing every GPU buffer - on every single frame, forever. More
    than one merged visual is the normal case at scale, since `max_size` and
    the vertex ceiling both force it.

    """
    v = _viewer(merge=True)
    v._merge.max_size = 2
    _bars(v, colors=("red", "lime", "blue", "white"))

    before = [id(vis) for vis in v.visuals]
    assert len(before) == 2  # max_size=2 split them across two visuals

    v._scene_stale = True
    v._refresh_scene()

    assert [id(vis) for vis in v.visuals] == before
    v.close()


def test_removal_does_not_count_as_erosion():
    """`split_outs` reports a merge being eaten away, which removal is not."""
    v = _bars(_viewer(merge=True))

    v.remove_objects("bar1")

    assert list(v.objects) == ["bar0", "bar2"]
    assert v.merge_stats.split_outs == 0
    # ... and the order bookkeeping does not keep the removed object
    assert "bar1" not in v._merge._order
    v.close()


def test_hidden_handles_are_not_retained_after_a_split():
    """A hidden member's stashed positions must not outlive its handle."""
    v = _bars(_viewer(merge=True), shader="phong")
    v.hide_objects("bar1")
    assert len(v._merge._hidden) == 1

    # Splitting bar1 out ends its life as a member
    v.unmerge_objects("bar1")

    assert all(h._object_id != "bar1" for h in v._merge._hidden)
    v.close()


def test_objects_keep_their_order_through_a_merge():
    """Merging must not reorder `Viewer.objects`.

    Merging appends its new visuals to the end of the scene, exactly as
    splitting does, so a canvas whose objects do not all land in one visual
    stops being in scene order the moment it is merged.

    """
    v = _viewer(merge=True)
    v.add_lines(_LINE, name="line1", color="red", center=False)
    v.add_points(_POINT, name="pts1", color="red", center=False)
    v.add_lines(_LINE, name="line2", color="red", center=False)
    v.add_points(_POINT, name="pts2", color="red", center=False)
    _frame(v)

    # Lines and points cannot share a visual, so this merged into two
    assert v.merge_stats.merged == 2
    assert list(v.objects) == ["line1", "pts1", "line2", "pts2"]
    v.close()


def test_len_counts_objects_not_visuals():
    """`len(viewer)` must report objects, which merging does not change."""
    plain = _bars(_viewer(merge=False))
    merged = _bars(_viewer(merge=True))

    assert len(merged.visuals) == 1 and len(plain.visuals) == 3
    assert len(merged) == len(plain) == 3
    assert merged._object_ids == plain._object_ids == ["bar0", "bar1", "bar2"]

    plain.close()
    merged.close()


def test_later_adds_fold_into_the_existing_merge():
    """Objects added after a merge must join it, not start a second visual.

    They only merge if they agree on the shared object state, which the
    viewer applies to new visuals (`cast_shadow` and friends) as part of the
    same refresh - so merging has to run after that, not before.

    """
    v = _viewer(merge=True)
    for batch in range(3):
        for i in range(3):
            v.add_lines(_LINE, name=f"b{batch}_{i}", color="red", center=False)
        _frame(v)

    assert v.merge_stats.members == 9
    assert len(v.visuals) == 1
    v.close()


def test_unmerge_survives_a_later_add():
    """An explicit unmerge must not be undone by the next object added."""
    v = _bars(_viewer(merge=True))
    v.unmerge_objects()
    assert v.merge_stats.merged == 0

    v.add_lines(_LINE, name="late", color="red", center=False)
    _frame(v)
    assert v.merge_stats.merged == 0

    # ... but asking to merge explicitly overrides it
    v.merge_objects()
    assert v.merge_stats.merged == 1
    v.close()


def test_mixed_transparency_keeps_each_object_right():
    """Colouring some objects transparent must not restyle the opaque ones.

    `alpha_mode` is shared by a merged visual's members, so the transparent
    one has to leave the merge rather than drag everyone into its blend mode.

    """
    plain = _bars(_viewer(merge=False))
    merged = _bars(_viewer(merge=True))

    colors = {"bar0": (1, 0, 0, 0.5), "bar1": (0, 1, 0, 1.0), "bar2": (0, 0, 1, 1.0)}
    for v in (plain, merged):
        v.set_colors(colors)

    # The transparent one blends; the opaque ones draw opaquely either way
    assert merged.objects["bar0"][0].material.alpha_mode == "add"
    assert plain.objects["bar0"][0].material.alpha_mode == "add"
    for name in ("bar1", "bar2"):
        assert merged.objects[name][0].material.alpha_mode in _OPAQUE_MODES
        assert plain.objects[name][0].material.alpha_mode in _OPAQUE_MODES

    plain.close()
    merged.close()


def test_opaque_objects_keep_their_blend_mode():
    """One transparent object must not drag its merged siblings into "add".

    `alpha_mode` is shared by a merged visual, so the transparent object has
    to leave the merge. The subtle part is the opaque ones: `set_colors` has
    nothing to write for them (they already draw opaquely), and if that
    silence is taken as indifference the transparent one claims the shared
    blend mode for everybody.

    """
    plain = _bars(_viewer(merge=False))
    merged = _bars(_viewer(merge=True))

    colors = {"bar0": "red", "bar1": (0, 0, 1, 0.5), "bar2": "lime"}
    for v in (plain, merged):
        v.set_colors(colors)
        assert v.objects["bar1"][0].material.alpha_mode == "add"
        for name in ("bar0", "bar2"):
            assert v.objects[name][0].material.alpha_mode in _OPAQUE_MODES

    # Only the transparent one had to leave the merge
    assert merged.merge_stats.split_outs == 1

    plain.close()
    merged.close()


def test_recoloring_one_object_does_not_split_it():
    """The commonest per-object operation must not erode the merge."""
    v = _bars(_viewer(merge=True))

    v.set_colors({"bar1": (1, 0, 1, 1)})

    assert v.merge_stats.split_outs == 0
    assert len(v.visuals) == 1
    assert _colors(v) == [RED, (255, 0, 255), BLUE]
    v.close()


def test_hidden_object_stays_hidden_when_split_out():
    """Splitting a hidden object out must not bring it back on screen.

    It stops being a member, so the NaN stash that was expressing its
    hiddenness no longer applies to it - the visual has to carry it instead.

    """
    v = _bars(_viewer(merge=True))
    v.hide_objects("bar1")
    hidden = _colors(v)

    v.unmerge_objects("bar1")

    assert v.invisible == ["bar1"]
    assert v.objects["bar1"][0].visible is False
    assert _colors(v) == hidden
    v.close()


def test_remove_works_after_an_explicit_merge():
    """`merge_objects()` merges whatever the policy says, so removal must too."""
    v = _bars(_viewer(merge=False))
    v.merge_objects()
    assert v.merge_stats.merged == 1

    v.remove_objects("bar1")

    assert list(v.objects) == ["bar0", "bar2"]
    assert _colors(v) == [RED, BLUE]
    v.close()


def test_styling_a_subset_works_on_the_first_call():
    """A style that splits must be applied to the visual the split produced.

    `Viewer.objects` is memoised, so the entry held across the split still
    describes the scene as it was - handles pointing into a merged visual
    that has just left the canvas. Styling those writes into a discarded
    object and looks like nothing happened.

    """
    v = _bars(_viewer(merge=True), shader="phong")

    v.set_silhouette(3, objects=["bar1"])

    from octarine.shaders import SilhouetteMeshMaterial

    assert isinstance(v.objects["bar1"][0].material, SilhouetteMeshMaterial)
    # ... and the untouched ones are still merged and still phong
    assert v.merge_stats.members == 2
    v.close()


def test_hidden_state_survives_a_remainder_of_one():
    """A one-member remainder is a plain visual again, and must stay hidden.

    `merge._strip_if_whole` takes `_sub_visuals` off a visual holding a single
    member, so the handle expressing its hiddenness goes away with it.

    """
    v = _viewer(merge=True)
    for i, color in enumerate(("red", "lime")):
        mesh = tm.creation.box((160, 20, 20))
        mesh.vertices = mesh.vertices + (0, 40 - i * 80, 0)
        v.add_mesh(mesh, name=f"b{i}", color=color, shader="basic", center=False)
    _frame(v)

    v.hide_objects("b0")
    v.unmerge_objects("b1")

    assert v.invisible == ["b0"]
    assert _colors(v) == [LIME]
    v.close()


def test_objects_hidden_before_merging_can_be_unhidden():
    """Hiding then merging must not make an object permanently invisible.

    Visibility is part of the merge key, so objects hidden while they were
    still visuals of their own merge into a visual that is itself invisible -
    which a per-member handle cannot undo.

    """
    v = _viewer(merge=True)
    for i, color in enumerate(("red", "lime", "blue")):
        mesh = tm.creation.box((160, 20, 20))
        mesh.vertices = mesh.vertices + (0, 60 - i * 40, 0)
        v.add_mesh(mesh, name=f"b{i}", color=color, shader="basic", center=False)
    v.hide_objects(["b0", "b1"])
    _frame(v)

    assert v.invisible == ["b0", "b1"]
    assert _colors(v) == [BLUE]

    v.unhide_objects()
    assert v.invisible == []
    assert _colors(v) == [RED, LIME, BLUE]
    v.close()


def test_pinned_objects_are_not_dragged_by_a_shared_write():
    """Pinning must hold even when the property is shared by a merged visual.

    The styling loops skip pinned objects, so they are not getting the value
    the rest are - and a bulk write that counted them as covered would reach
    them through the shared material anyway.

    """
    plain = _bars(_viewer(merge=False))
    merged = _bars(_viewer(merge=True))

    for v in (plain, merged):
        v.pin_objects("bar0")
        v.set_colors((1, 0, 0, 0.5))
        assert v.objects["bar0"][0].material.alpha_mode in _OPAQUE_MODES
        assert v.objects["bar1"][0].material.alpha_mode == "add"

    plain.close()
    merged.close()


# ------------------------------------------------------- switching it on/off


def test_merge_can_be_set_after_init():
    """`Viewer.merge` must take effect when set, not at some later frame."""
    v = _bars(_viewer(merge=False))
    before = _colors(v)
    assert v.merge is False and len(v.visuals) == 3

    v.merge = True
    assert v.merge is True
    assert len(v.visuals) == 1
    assert list(v.objects) == ["bar0", "bar1", "bar2"]
    assert _colors(v) == before

    # Turning it off must not leave the canvas half merged
    v.merge = False
    assert len(v.visuals) == 3
    assert all(isinstance(x[0], gfx.Mesh) for x in v.objects.values())
    assert _colors(v) == before
    v.close()


def test_merge_threshold_can_be_set_after_init():
    """A count set afterwards must be honoured like one passed to `Viewer`."""
    v = _bars(_viewer(merge=False))

    v.merge = 10  # more objects than the canvas holds
    assert v.merge_stats.merged == 0

    v.merge = 3
    assert v.merge_stats.merged == 1
    v.close()


def test_per_object_state_survives_being_merged():
    """Pinning and selection are kept on the object, which merging replaces.

    An object stops being a visual of its own and becomes a handle, so the
    bookkeeping the viewer hung off it has to come along - otherwise pinning
    silently stops holding and a selected object can never be restored to the
    colour it had.

    """
    v = _viewer(merge=True)
    for i, color in enumerate(("red", "lime")):
        mesh = tm.creation.box((160, 20, 20))
        mesh.vertices = mesh.vertices + (0, 40 - i * 80, 0)
        v.add_mesh(mesh, name=f"b{i}", color=color, shader="basic", center=False)

    v.pin_objects("b0")
    v.selected = ["b1"]
    _frame(v)  # the first merge pass happens here

    assert v.merge_stats.merged == 1
    assert v.pinned == ["b0"]

    v.selected = []
    assert _colors(v) == [RED, LIME]  # b1 back to its own colour, not the highlight
    v.close()


def test_per_object_state_survives_being_split_out():
    """... and the same on the way back out of a merged visual."""
    v = _bars(_viewer(merge=True))
    v.highlight_objects("bar1")
    assert v.highlighted == ["bar1"]

    v.unmerge_objects("bar1")

    assert v.highlighted == ["bar1"]
    v.unhighlight_objects()
    assert v.highlighted == []
    assert _colors(v) == [RED, LIME, BLUE]
    v.close()


# ---------------------------------------------------------------- controls


def test_controls_toggle_merging(qt_controls_unmerged):
    """The controls checkbox must switch merging on and off."""
    v, controls = qt_controls_unmerged
    _bars(v)
    controls.flush_legend()

    assert controls.merge_checkbox.isChecked() is False
    assert len(v.visuals) == 3

    controls.merge_checkbox.setChecked(True)
    assert v.merge is True
    assert len(v.visuals) == 1
    # The objects are still listed one per row
    assert list(v.objects) == ["bar0", "bar1", "bar2"]

    controls.merge_checkbox.setChecked(False)
    assert v.merge is False
    assert len(v.visuals) == 3


def test_controls_toggle_is_reversible(qt_controls_unmerged):
    """Switching merging off must not stop it being switched back on.

    Turning it off splits everything out, and the flag that keeps a *user*
    unmerge from being undone by the next `add` would make that permanent.

    """
    v, controls = qt_controls_unmerged
    _bars(v)

    for _ in range(3):
        controls.merge_checkbox.setChecked(True)
        assert v.merge_stats.merged == 1
        controls.merge_checkbox.setChecked(False)
        assert v.merge_stats.merged == 0


def test_controls_merge_readout_follows_the_canvas(qt_controls):
    """The readout must track merging, splitting and objects coming and going."""
    v, controls = qt_controls
    _bars(v)
    controls.flush_legend()
    assert controls.merge_status_label.text() == "3 objects drawn as 1"

    # A style a merged object cannot hold erodes the merge, and says so
    v.set_silhouette(3, objects=["bar1"])
    controls.flush_legend()
    assert controls.merge_status_label.text() == "3 objects drawn as 2 (1 split out)"

    v.remove_objects("bar0")
    controls.flush_legend()
    assert controls.merge_status_label.text().startswith("2 objects")


def test_controls_checkbox_follows_the_viewer(qt_controls_unmerged):
    """Setting `Viewer.merge` from code must move the checkbox with it."""
    v, controls = qt_controls_unmerged
    _bars(v)

    v.merge = True
    controls.flush_legend()
    assert controls.merge_checkbox.isChecked() is True
    assert controls.merge_status_label.text() == "3 objects drawn as 1"

    v.merge = False
    controls.flush_legend()
    assert controls.merge_checkbox.isChecked() is False
    assert controls.merge_status_label.text() == "3 objects, not merged"


def test_unmerging_by_hand_is_not_counted_as_erosion(qt_controls):
    """`split_outs` reports a merge being eaten away, not deliberate splits."""
    v, _ = qt_controls
    _bars(v)

    v.unmerge_objects()
    v.merge_objects()
    v.merge = False
    v.merge = True
    assert v.merge_stats.split_outs == 0

    # ... whereas a style that cannot be held per object is erosion
    v.set_silhouette(3, objects=["bar1"])
    assert v.merge_stats.split_outs == 1
