"""Tests for `octarine.Animation` and the animation controls."""

import os
import png
import pytest
import numpy as np
import trimesh as tm
import octarine as oc

from octarine import anim_utils

# Anything Qt (the controls tab below) must not try to open a window
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _make_viewer(camera="ortho"):
    v = oc.Viewer(offscreen=True, size=(120, 90), camera=camera)
    v.add_mesh(
        tm.creation.icosphere(subdivisions=2, radius=8), color="orange", name="ball"
    )
    v.add_mesh(
        tm.creation.box(extents=(4, 4, 4)).apply_translation((40, 0, 0)),
        color="cyan",
        name="cube",
    )
    v.center_camera()
    return v


@pytest.fixture()
def viewer():
    v = _make_viewer()
    yield v
    v.close()


@pytest.fixture()
def persp_viewer():
    """A perspective camera, which - unlike an ortho one - stands off its pivot.

    Anything to do with where the camera *is* (rather than where it looks) is
    only really tested here: an orthographic camera sits exactly at the point
    it is looking at, so it never moves when orbiting.

    """
    v = _make_viewer(camera="perspective")
    yield v
    v.close()


def _pivot_distance(state):
    """Distance from the camera to the point it is looking at."""
    return np.linalg.norm(anim_utils._pivot(state) - np.asarray(state["position"]))


def test_timeline_layout(viewer):
    """Segments must line up one after the other."""
    anim = oc.Animation(viewer, fps=10)
    anim.orbit(duration=2).hold(1).move_to("XZ", duration=3)

    assert [(s.start, s.end) for s in anim.camera] == [(0, 2), (2, 3), (3, 6)]
    assert anim.duration == 6
    assert anim.n_frames == 60
    # The last frame stops one short of the end so a full turn loops seamlessly
    assert anim.times()[-1] == pytest.approx(6 - 1 / 10)


def test_orbit_is_a_closed_loop(viewer):
    """A full turn must come back to exactly where it started."""
    anim = oc.Animation(viewer).orbit(turns=1, duration=4)

    anim.set_time(0)
    start = viewer.get_view()
    anim.set_time(anim.duration)
    end = viewer.get_view()

    assert np.allclose(start["position"], end["position"], atol=1e-6)
    # Quaternions are double covers - q and -q are the same rotation
    assert abs(np.dot(start["rotation"], end["rotation"])) == pytest.approx(1, abs=1e-6)


def test_orbit_keeps_its_distance(persp_viewer):
    """The camera must circle the pivot, not spiral in or out."""
    anim = oc.Animation(persp_viewer).orbit(turns=1, duration=4)
    orbit = anim.camera[0]

    distances, positions = [], []
    for t in np.linspace(0, orbit.duration, 9):
        anim.set_time(t)
        state = persp_viewer.get_view()
        positions.append(np.asarray(state["position"]).copy())
        distances.append(np.linalg.norm(positions[-1] - orbit.pivot))
        # ... and keep looking at the pivot all the way round
        assert np.allclose(anim_utils._pivot(state), orbit.pivot, atol=1e-6)

    assert distances[0] > 1  # i.e. the camera stands off the pivot at all
    assert np.allclose(distances, distances[0], atol=1e-6)
    # Half way round we must be on the opposite side of the pivot
    assert np.allclose(
        positions[4] - orbit.pivot, orbit.pivot - positions[0], atol=1e-6
    )


def test_orbit_axis(viewer):
    """The rotation axis must be the one asked for."""
    anim = oc.Animation(viewer)
    anim.orbit(axis="x", duration=1)
    assert np.allclose(anim.camera[0].axis, (1, 0, 0))

    # "up" is the scene's up direction, i.e. what gives a turntable rotation
    anim = oc.Animation(viewer)
    anim.orbit(axis="up", duration=1)
    up = viewer.get_view()["reference_up"]
    assert np.allclose(anim.camera[0].axis, up / np.linalg.norm(up))

    # ... and can be flipped
    anim = oc.Animation(viewer)
    anim.orbit(axis="-y", duration=1)
    assert np.allclose(anim.camera[0].axis, (0, -1, 0))

    with pytest.raises(ValueError):
        oc.Animation(viewer).orbit(axis="nope")


def test_orbit_recenter(viewer):
    """`recenter` decides whether we frame the objects or keep the view."""
    # Framed on the cube, which sits off to one side of the scene
    anim = oc.Animation(viewer).orbit("cube", duration=1)
    assert np.allclose(anim.camera[0].pivot, (40, 0, 0), atol=1e-6)
    anim.set_time(0)
    assert viewer.get_view()["width"] < 20  # zoomed in on a small object

    # Without re-centering the current view is kept, whatever it is
    before = viewer.get_view()
    anim = oc.Animation(viewer).orbit(duration=1, recenter=False)
    anim.set_time(0)
    after = viewer.get_view()
    assert np.allclose(before["position"], after["position"])
    assert before["width"] == pytest.approx(after["width"])


def test_orbit_needs_something_to_orbit(viewer):
    viewer.clear()
    with pytest.raises(ValueError):
        oc.Animation(viewer).orbit()
    with pytest.raises(ValueError):
        oc.Animation(viewer).orbit("no-such-object")


def test_move_to_paths(persp_viewer):
    """An arc swings around the pivot, a linear path cuts straight across."""
    persp_viewer.set_view("XY")
    front = persp_viewer.get_view()
    persp_viewer.set_view("-XY")
    back = persp_viewer.get_view()
    persp_viewer.set_view(front)

    distance = _pivot_distance(front)
    assert distance > 1  # the two views are on opposite sides of the scene

    # Half way round the arc we are still the same distance out ...
    anim = oc.Animation(persp_viewer).start_at(front)
    anim.move_to(back, duration=2, path="arc")
    anim.set_time(1)
    assert _pivot_distance(persp_viewer.get_view()) == pytest.approx(distance, rel=1e-6)

    # ... whereas the straight line goes right through the middle of it all
    anim = oc.Animation(persp_viewer).start_at(front)
    anim.move_to(back, duration=2, path="linear")
    anim.set_time(1)
    assert np.allclose(
        persp_viewer.get_view()["position"],
        (np.asarray(front["position"]) + np.asarray(back["position"])) / 2,
        atol=1e-6,
    )

    # Ends up exactly at the target view either way
    for path in ("arc", "linear"):
        anim = oc.Animation(persp_viewer).start_at(front)
        anim.move_to(back, duration=2, path=path)
        anim.set_time(2)
        assert np.allclose(
            persp_viewer.get_view()["position"], back["position"], atol=1e-6
        )

    with pytest.raises(ValueError):
        oc.Animation(persp_viewer).start_at(front).move_to(back, path="wiggly")


def test_easing(viewer):
    """Easing must not change where a segment starts and ends."""
    anim = oc.Animation(viewer).start_at("XY")
    anim.move_to("XZ", duration=2, easing="in_out")

    segment = anim.camera[0]
    assert segment.progress(0) == 0
    assert segment.progress(2) == 1
    # Slow at the ends, fast in the middle
    assert segment.progress(0.2) < 0.1
    assert segment.progress(1) == pytest.approx(0.5)

    with pytest.raises(ValueError):
        oc.Animation(viewer).start_at("XY").move_to("XZ", easing="nope")

    # Callables work just as well
    anim = oc.Animation(viewer).start_at("XY")
    anim.move_to("XZ", duration=2, easing=lambda t: t**3)
    assert anim.camera[0].progress(1) == pytest.approx(0.125)


def test_zoom(persp_viewer):
    """Zooming must not move the point we are looking at."""
    viewer = persp_viewer
    anim = oc.Animation(viewer)
    before = viewer.get_view()
    anim.zoom(2, duration=1)

    anim.set_time(1)
    after = viewer.get_view()
    assert after["width"] == pytest.approx(before["width"] / 2)
    assert np.allclose(anim_utils._pivot(before), anim_utils._pivot(after), atol=1e-6)


def test_set_time_is_random_access(viewer):
    """Any point in time must render the same, whatever we did before."""
    anim = oc.Animation(viewer, fps=4)
    anim.orbit(duration=1).move_to("XZ", duration=1)

    forwards = []
    for t in anim.times():
        anim.set_time(t)
        forwards.append(viewer.get_view()["position"].copy())

    backwards = []
    for t in reversed(anim.times()):
        anim.set_time(t)
        backwards.append(viewer.get_view()["position"].copy())

    assert np.allclose(forwards, backwards[::-1])


def test_render_frames(viewer):
    """Rendering without a filename must return the frames themselves."""
    anim = oc.Animation(viewer, fps=8).orbit(duration=0.5)
    before = viewer.get_view()

    frames = anim.render(None, pixel_ratio=1, progress=False)

    assert len(frames) == 4
    assert frames[0].shape == (90, 120, 4)
    assert not np.array_equal(frames[0], frames[2])  # something actually moved
    # The camera must be back where it was
    assert np.allclose(viewer.get_view()["position"], before["position"])
    assert viewer.size == (120, 90)


def test_render_size_and_supersample(viewer):
    anim = oc.Animation(viewer, fps=4).orbit(duration=0.5)
    frames = anim.render(
        None, size=(64, 48), pixel_ratio=1, supersample=2, progress=False
    )
    assert frames[0].shape == (48, 64, 4)
    assert viewer.size == (120, 90)  # canvas back to its old size


def test_render_png_sequence(viewer, tmp_path):
    """A path without an extension gets one PNG per frame."""
    anim = oc.Animation(viewer, fps=6).orbit(duration=0.5)
    out = anim.render(tmp_path / "frames", pixel_ratio=1, progress=False)

    files = sorted(out.glob("*.png"))
    assert len(files) == 3
    w, h, *_ = png.Reader(filename=str(files[0])).read()
    assert (w, h) == (120, 90)


def test_render_video(viewer, tmp_path):
    imageio = pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")

    anim = oc.Animation(viewer, fps=8).orbit(duration=0.5)
    out = anim.render(tmp_path / "orbit.mp4", pixel_ratio=1, progress=False)

    reader = imageio.get_reader(str(out))
    try:
        assert reader.get_meta_data()["size"] == (120, 90)
    finally:
        reader.close()


def test_render_unknown_format(viewer, tmp_path):
    anim = oc.Animation(viewer, fps=4).orbit(duration=0.5)
    with pytest.raises(ValueError):
        anim.render(tmp_path / "orbit.tiff")


def test_render_progress_callback(viewer):
    seen = []
    anim = oc.Animation(viewer, fps=4).orbit(duration=1)
    anim.render(None, pixel_ratio=1, progress=lambda i, n: seen.append((i, n)))
    assert seen == [(i + 1, 4) for i in range(4)]


def test_empty_animation(viewer):
    anim = oc.Animation(viewer)
    assert anim.duration == 0
    with pytest.raises(ValueError):
        anim.render(None)
    with pytest.raises(ValueError):
        anim.play()


def test_recorder_cancel(viewer, tmp_path):
    """A cancelled recording must leave nothing behind."""
    anim = oc.Animation(viewer, fps=8).orbit(duration=1)
    before = viewer.get_view()

    rec = anim.recorder(tmp_path / "frames", pixel_ratio=1)
    rec.step()
    rec.step()
    assert len(list((tmp_path / "frames").glob("*.png"))) == 2

    rec.cancel()
    assert list((tmp_path / "frames").glob("*.png")) == []
    assert np.allclose(viewer.get_view()["position"], before["position"])


def test_recorder_parks_the_draw_loop(viewer, monkeypatch):
    """Recording must park the viewer's own draw loop - and put it back.

    Anything driving the recorder from an event loop (the controls panel) would
    otherwise have the canvas paint the scene in between our own frames.

    """
    # The offscreen canvas has no scheduler, so stand one in. N.B. it hands the
    # rate back as a float although the setter only takes ints.
    fps = {"value": 30.0}
    monkeypatch.setattr(
        type(viewer),
        "max_fps",
        property(
            lambda self: fps["value"],
            lambda self, value: fps.__setitem__("value", value),
        ),
    )

    rec = oc.Animation(viewer, fps=4).orbit(duration=0.5).recorder(None, pixel_ratio=1)
    rec.step()
    assert fps["value"] == 1

    rec.finish()
    assert fps["value"] == 30


def test_play_and_stop(viewer):
    """Playback must hook into (and unhook from) the viewer's animation loop."""
    anim = oc.Animation(viewer, fps=10).orbit(duration=1)
    assert not anim.playing

    anim.play(loop=True)
    assert anim.playing
    assert anim._playback in viewer._animations

    anim.stop()
    assert not anim.playing
    # Removal is deferred to the next frame, hence the flag rather than the dict
    assert (
        anim._playback is None
        or anim._playback in viewer._animations_flagged_for_removal
    )


def test_start_at(viewer):
    anim = oc.Animation(viewer).start_at("XY")
    xy = viewer.get_view()
    # `start_at` must not move the camera itself, just the timeline's origin
    anim.hold(1)
    anim.set_time(0)
    assert np.allclose(viewer.get_view()["rotation"], xy["rotation"], atol=1e-6)

    with pytest.raises(ValueError):
        anim.start_at("XZ")  # timeline is no longer empty


def test_animation_controls(viewer, tmp_path):
    """The animation tab must build and record what its widgets say."""
    try:
        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    except Exception as e:  # not installed, clashing bindings, no plugin, ...
        pytest.skip(f"no usable PySide6: {e}")

    from octarine.controls import Controls

    controls = Controls(viewer)
    try:
        assert "Animation" in [
            controls.tabs.tabText(i) for i in range(controls.tabs.count())
        ]

        # The tab must be able to squeeze down to well below the height it
        # would like, i.e. it scrolls instead of stretching the whole window
        tab = controls.tab5
        assert tab.minimumSizeHint().height() < tab.sizeHint().height() / 2

        controls.anim_fps.setValue(6)
        controls.anim_orbit_duration.setValue(0.5)
        assert "3 frames" in controls.anim_summary_label.text()

        # Only the current mode's settings take up any space
        assert [p.isHidden() for p in controls.anim_pages] == [False, True]

        # Keyframes: two waypoints plus the loop back to the first
        controls.anim_mode_dropdown.setCurrentText("Keyframes")
        assert [p.isHidden() for p in controls.anim_pages] == [True, False]
        controls._anim_add_keyframe()
        viewer.set_view("XZ")
        controls.anim_keyframe_duration.setValue(1.0)
        controls._anim_add_keyframe()
        controls.anim_keyframe_loop.setChecked(True)
        assert len(controls._anim_build().camera) == 2
        assert controls._anim_build().duration == pytest.approx(2.0)

        # A single keyframe is not an animation
        controls._anim_keyframes.pop()
        controls._anim_refresh_keyframes()
        controls._anim_toggle_preview()
        assert controls._anim_preview is None
        assert "at least two" in controls.anim_status_label.text()

        # Browsing must not spin a nested event loop: those do not survive
        # every way of hosting the Qt loop (IPython's input hook quits them,
        # taking the dialog with it). The watchdog turns a regression into a
        # failed assertion rather than a test that hangs forever.
        from PySide6 import QtCore

        QtCore.QTimer.singleShot(
            3000,
            lambda: [
                w.reject()
                for w in app.topLevelWidgets()
                if isinstance(w, QtWidgets.QFileDialog)
            ],
        )
        controls.anim_browse_button.click()
        assert len(controls._file_dialogs) == 1
        dialog = controls._file_dialogs[0]
        assert not dialog.isHidden()
        # ... and the answer comes back through the signal
        dialog.fileSelected.emit(str(tmp_path / "picked.mp4"))
        assert controls.anim_filename_edit.text() == str(tmp_path / "picked.mp4")
        # Closing it must also let go of it, so they don't pile up
        dialog.close()
        assert controls._file_dialogs == []

        # Record an orbit as a PNG sequence, one frame per timer tick
        controls.anim_mode_dropdown.setCurrentText("Orbit")
        controls.anim_format.setCurrentText("PNG sequence")
        controls.anim_filename_edit.setText(str(tmp_path / "seq"))
        controls._anim_toggle_record()
        assert controls.anim_record_button.text() == "Cancel"
        for _ in range(100):
            app.processEvents()
            if controls._anim_recorder is None:
                break
        assert controls._anim_recorder is None
        assert controls.anim_record_button.text() == "Record"
        assert len(list((tmp_path / "seq").glob("*.png"))) == 3
    finally:
        controls.close()
