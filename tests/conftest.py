"""Shared test setup.

The only thing here is a Qt binding guard, and it has to run before anything
else - see below.
"""

import os

# Anything Qt in the test suite must not try to open a window
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Load PySide6 (the binding `octarine.controls` needs) before any other Qt
# binding can get in first. Qt bindings each ship their own copy of the Qt
# libraries under the same names, so whichever is loaded first is what every
# later binding's extension modules get resolved against - and PySide6 then
# dies on a missing symbol. An unrelated pytest plugin is enough to trigger
# this: napari, for instance, pulls in PyQt6 and is auto-registered whenever
# it happens to be installed in the same environment, which would silently
# skip every Qt test in this suite.
#
# Nothing here fails if PySide6 is missing; the Qt tests skip themselves.
try:
    import PySide6.QtWidgets  # noqa: F401
except ImportError:
    pass
