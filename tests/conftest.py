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
