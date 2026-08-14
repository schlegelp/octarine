"""Tests for `Viewer.screenshot`, in particular its supersampling."""

import os
import png
import pytest
import numpy as np
import trimesh as tm
import octarine as oc

# The on-screen canvas below must not try to open a window
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture()
def viewer():
    v = oc.Viewer(offscreen=True, size=(200, 150))
    yield v
    v.close()


@pytest.fixture()
def scene_viewer(viewer):
    """A sphere plus a helix - curved silhouettes and thin, near-diagonal lines."""
    viewer.add_mesh(tm.creation.icosphere(subdivisions=3, radius=8), color="orange")
    t = np.linspace(0, 6 * np.pi, 200)
    viewer.add_lines(
        np.c_[np.cos(t) * 14, np.sin(t) * 14, t - 9], color="cyan", linewidth=1
    )
    viewer.center_camera()
    return viewer


def _shot(v, **kwargs):
    return np.asarray(v.screenshot(filename=None, **kwargs))


def test_screenshot_size(scene_viewer):
    """Supersampling must not change the image dimensions."""
    for supersample in (1, 2, 4):
        img = _shot(scene_viewer, pixel_ratio=1, supersample=supersample)
        assert img.shape == (150, 200, 4)

    # `size` and `pixel_ratio` still define the dimensions between them
    assert _shot(scene_viewer, size=(400, 300), pixel_ratio=1).shape == (300, 400, 4)
    assert _shot(scene_viewer, size=(400, 300), pixel_ratio=2).shape == (600, 800, 4)


def test_screenshot_file(scene_viewer, tmp_path):
    """A supersampled screenshot must save as a valid PNG of the right size."""
    filename = tmp_path / "shot.png"
    scene_viewer.screenshot(filename, pixel_ratio=1, supersample=2)
    w, h, _, _ = png.Reader(filename=str(filename)).read()
    assert (w, h) == (200, 150)


def test_screenshot_without_supersampling_is_a_plain_snapshot(scene_viewer):
    """`supersample=1` must take the renderer's snapshot as-is."""
    scene_viewer.renderer.pixel_ratio = 1
    scene_viewer.canvas.draw()
    plain = scene_viewer.renderer.snapshot()
    assert np.array_equal(
        plain, scene_viewer._screenshot(alpha=False, pixel_ratio=1, supersample=1)
    )


def test_screenshot_supersampling_antialiases(scene_viewer):
    """Supersampling must bring the image closer to a well-resolved render."""
    # An 8x render, downsampled by hand, is as close to ground truth as we get
    reference = _shot(scene_viewer, pixel_ratio=1, supersample=8).astype(float)

    def error(supersample):
        img = _shot(scene_viewer, pixel_ratio=1, supersample=supersample)
        return np.abs(img.astype(float) - reference).mean()

    # Each step has to be a clear improvement, not a wash. The first one is
    # worth a lot more than the second: 2x already resolves most of what a
    # single sample per pixel misses (here ~8x less error, against ~2x for 4x)
    assert error(2) < 0.25 * error(1)
    assert error(4) < 0.75 * error(2)


def test_screenshot_scales_pixel_effects(scene_viewer):
    """Effects sized in pixels must come out the same size in the image.

    Rendering the frame at N times the output resolution would otherwise
    shrink e.g. a one-pixel outline to 1/N of a pixel.

    """
    scene_viewer.set_outline(True, thickness=2)

    def ink(img):
        """How much darker than the background the image's dark pixels are."""
        return (img[..., :3].astype(float).mean(-1) < 40).sum()

    plain = ink(_shot(scene_viewer, pixel_ratio=1, supersample=1, alpha=False))
    for supersample in (2, 4):
        img = _shot(scene_viewer, pixel_ratio=1, supersample=supersample, alpha=False)
        assert ink(img) == pytest.approx(plain, rel=0.2)


def test_screenshot_alpha(scene_viewer):
    """A transparent screenshot must have its edges anti-aliased too."""
    img = _shot(scene_viewer, pixel_ratio=1, supersample=4, alpha=True)
    alpha = img[..., 3]
    assert alpha.min() == 0 and alpha.max() == 255
    # Partially covered pixels along the silhouette
    assert ((alpha > 0) & (alpha < 255)).sum() > 100


def test_screenshot_restores_state(scene_viewer):
    """Nothing the screenshot turns up may survive it."""
    scene_viewer.set_outline(True, thickness=1.5)
    scene_viewer.set_ambient_occlusion(blur=2)

    # An automatic pixel ratio must stay automatic
    scene_viewer.renderer.pixel_scale = None
    scene_viewer.screenshot(filename=None, supersample=4)
    assert scene_viewer.renderer._pixel_scale is None
    assert scene_viewer.renderer._pixel_ratio is None

    # ... and an explicit one must come back unchanged
    scene_viewer.renderer.pixel_ratio = 1.5
    scene_viewer.screenshot(filename=None, pixel_ratio=3, supersample=2)
    assert scene_viewer.renderer.pixel_ratio == 1.5

    assert scene_viewer._outline_pass.thickness == 1.5
    assert scene_viewer._ao_pass.blur == 2
    assert scene_viewer._background.visible


@pytest.mark.parametrize("trigger", ("reactive", "active_window"))
def test_screenshot_forces_a_render(scene_viewer, trigger):
    """A non-continuous render trigger must not hand back a stale frame."""
    # Draw an ordinary frame, then put the viewer in the state where neither
    # trigger would render again on its own: nothing is flagged as stale and
    # the window is not active (an offscreen canvas is always "active", so
    # that one we have to fake)
    scene_viewer.canvas.force_draw()
    scene_viewer._render_stale = False
    scene_viewer.canvas.isActiveWindow = lambda: False
    scene_viewer.render_trigger = trigger

    img = _shot(scene_viewer, pixel_ratio=1, supersample=1, alpha=True)
    assert img[..., 3].min() == 0  # the background really is gone

    # The trigger must come back untouched - and the canvas must end up showing
    # an ordinary frame again rather than the one we just captured
    assert scene_viewer.render_trigger == trigger
    scene_viewer.canvas.force_draw()
    assert np.asarray(scene_viewer.renderer.snapshot())[..., 3].min() == 255


def test_screenshot_controls_collect_screenshots_in_a_folder(viewer, tmp_path):
    """A folder in the control panel's filename box must number the files."""
    try:
        from PySide6 import QtWidgets

        QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    except Exception as e:  # not installed, clashing bindings, no plugin, ...
        pytest.skip(f"no usable PySide6: {e}")

    from octarine.controls import Controls

    controls = Controls(viewer)
    try:
        # A file name is still used as it stands (bar the enforced suffix)
        controls.screenshot_filename_edit.setText(str(tmp_path / "shot"))
        assert controls._screenshot_target() == tmp_path / "shot.png"

        controls.screenshot_filename_edit.setText(str(tmp_path))
        for i in (1, 2, 3):
            assert controls._screenshot_target() == tmp_path / f"screenshot_{i:04d}.png"
            controls._save_screenshot()
            assert f"screenshot_{i:04d}.png" in controls.screenshot_status_label.text()

        # Numbering must continue past what is already in there, gaps and all
        (tmp_path / "screenshot_0009.png").touch()
        assert controls._screenshot_target() == tmp_path / "screenshot_0010.png"
    finally:
        controls.close()


def test_screenshot_supersample_is_capped(scene_viewer):
    """A factor that would blow past the GPU's texture limit must be reduced."""
    max_size = scene_viewer.renderer.device.limits["max-texture-dimension-2d"]
    size = (max_size // 3, max_size // 4)
    assert scene_viewer._clamp_supersample(8, size) == 3
    assert scene_viewer._clamp_supersample(2, size) == 2


def test_screenshot_errors(scene_viewer):
    with pytest.raises(ValueError):
        scene_viewer.screenshot(filename=None, supersample=0)


def test_screenshot_on_screen_canvas():
    """Screenshots must also work off an actual (i.e. on-screen) canvas.

    That path reads back a frame the canvas drew for the screen rather than
    one we asked for ourselves, and is easy to break without noticing.

    """
    try:
        # Whichever backend `rendercanvas.auto` picks (Qt, glfw, ...)
        v = oc.Viewer(offscreen=False, show=False, size=(200, 150))
    except Exception as e:  # no GUI toolkit, no display, clashing Qt bindings
        pytest.skip(f"no on-screen canvas available: {e}")

    try:
        v.add_mesh(tm.creation.icosphere(subdivisions=3, radius=8), color="orange")
        v.center_camera()
        v.show()  # this is what hooks up the draw callback
        for supersample in (1, 2):
            img = _shot(v, pixel_ratio=1, supersample=supersample, alpha=False)
            assert img.shape == (150, 200, 4)
            assert img[..., :3].max() > 0  # something was actually rendered
    finally:
        v.close()
