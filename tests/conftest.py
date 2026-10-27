"""Shared test setup.

The only thing here is a Qt binding guard, and it has to run before anything
else - see below.
"""

import os
import warnings

# Anything Qt in the test suite must not try to open a window
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Load PySide6 (the binding `octarine.controls` needs) before any other Qt
# binding can get in first. Qt bindings each ship their own copy of the Qt
# libraries under the same names, so whichever is loaded first is what every
# later binding's extension modules get resolved against - and PySide6 then
# dies on a missing symbol. An unrelated pytest plugin is enough to trigger
# this: napari, for instance, pulls in PyQt6 and is auto-registered whenever
# it happens to be installed in the same environment, which silently skipped
# every Qt test in this suite until this guard was added.
#
# Winning the race is only best-effort - pytest imports `pytest11` entry-point
# plugins before it loads any conftest, so a plugin that pulls in its binding
# at *import* time would still beat us. The ones seen in the wild (napari
# among them) import theirs lazily, which is late enough for this to work.
try:
    import PySide6.QtWidgets  # noqa: F401
except ModuleNotFoundError:
    # No PySide6 at all: the Qt tests skip themselves, nothing to report.
    pass
except ImportError as e:
    # PySide6 is installed but will not load, i.e. the clash above happened
    # anyway. Say so: the Qt tests are about to vanish from the run, and not
    # noticing that is exactly how a broken one went unseen for months. A
    # warning rather than an error, because the ~300 tests that need no Qt
    # are still perfectly runnable.
    warnings.warn(
        "PySide6 is installed but failed to load, so every Qt test will be "
        "skipped. Another Qt binding has most likely been pulled into the "
        f"process first - see tests/conftest.py. PySide6 said: {e}",
        stacklevel=1,
    )


# ---------------------------------------------------------------- rendering

import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture()
def flat_viewer():
    """An offscreen viewer framed so that `bands` reads crisp rows.

    Shared by the merging tests, which check what a scene *draws* rather than
    what it holds: a flat scene, a fixed 200x200 window on the origin, and no
    anti-aliasing, so each object shows up as a run of identically coloured
    rows. See `bands`.

    """
    import octarine as oc

    v = make_flat_viewer(oc.Viewer(offscreen=True, size=(200, 200)))
    yield v
    v.close()


def make_flat_viewer(viewer):
    """Strip a viewer down so its output is readable row by row."""
    viewer.renderer.pixel_ratio = 1
    viewer.renderer.ppaa = "none"  # keep the band edges crisp so we can count rows
    return frame_viewer(viewer)


def frame_viewer(viewer):
    """Point the camera at a fixed 200x200 window on the origin.

    Call this *after* the objects are in, with `center=False` on every add:
    adding asks the viewer to re-center, which runs on the next refresh and
    would throw this away.

    """
    cam = viewer.camera
    cam.width = cam.height = 200
    cam.zoom = 1
    cam.depth_range = (1, 1000)
    cam.local.position = (0, 0, 120)
    cam.look_at((0, 0, 0))
    return viewer


def bands(viewer):
    """Contiguous runs of lit rows, as (height, color) pairs.

    The colour is sampled in the middle of the run, well away from the caps
    and outlines that the renderer's supersampling leaves half-covered.

    """
    img = np.asarray(viewer.screenshot(filename=None))[..., :3].astype(int)
    lit = img.sum(2) > 40

    runs = []
    for row in np.where(lit.any(1))[0]:
        if runs and row == runs[-1][1] + 1:
            runs[-1][1] = row
        else:
            runs.append([row, row])

    out = []
    for lo, hi in runs:
        mid = (lo + hi) // 2
        cols = np.where(lit[mid])[0]
        out.append((hi - lo + 1, tuple(img[mid, cols[len(cols) // 2]])))
    return out
