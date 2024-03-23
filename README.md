![cocoa](docs/_static/octarine_logo_banner.png)
<p align="center">
<i>
Octarine is the eighth color of the Discworld's spectrum, which is described as the color of magic itself. Only wizards and cats can see it.
</i>
</p>

# Octarine
A high-performance, easy-to-use 3D viewer. `Octarine` is build on top of the excellent
[`pygfx`](https://github.com/pygfx/pygfx) WGPU rendering engine which does most of the heavy lifting - we're simply
abstracting away some of the boiler plate code for you.

## Rationale :thought_balloon:
Why another 3d viewer? There are plenty options out there:
[`vedo`](https://github.com/marcomusy/vedo), [`ipygany`](https://github.com/jupyter-widgets-contrib/ipygany), [`polyscope`](https://github.com/nmwsharp/polyscope), [`napari`](https://github.com/napari/napari), [`fury`](https://github.com/fury-gl/fury) or [`pyvista`](https://github.com/pyvista/pyvista) to name but a few. All of these are great in their own right but I wanted something (a) without heavy dependencies (i.e. no VTK), (b) that lets me interactively explore my data in both REPL and Jupyter and (c) is very performant. None of the above ticked all those boxes for me.

`Octarine` tries to fill that gap:
1. _Lightweight_ with very few direct or indirect dependencies.
2. Works in both _Jupyter_ and _REPL_.
3. _High performance_: a mesh with 15M faces renders with 80 fps at 1080p (2023 MacBook Pro).

## ToDo :ballot_box_with_check:
This is still a prototype but basic stuff already works (mostly because `pygfx` makes it so ridiculously easy).

- [x] basic datatypes: meshes, points, scatter, volumes
- [x] custom keyboard shortcuts
- [x] rudamentary controls + legend
- [ ] user-defined animations
- [ ] proper docs
- [ ] tests

## Installation :rocket:

```bash
pip install octarine3d
```

In addition you will need to install at least one window manager supported by [wgpu-py](https://github.com/pygfx/wgpu-py):
- qt: PySide6, PyQt6, PySide2, PyQt5 all work but I recommend PySide6 (see below)
- glfw: a lightweight GUI for the desktop
- jupyter_rfb: only needed if you plan on using `Octarine` in Jupyter
- wx

Please note that at this point, `Octarine`'s controls panel requires `PySide6`. So if you need GUI controls you have to use `PySide6`.

## Quickstart :fire:

```python
# Create a Viewer instance
from octarine import Viewer
v = Viewer()

# Add random points as scatter
import numpy as np
points = np.random.rand(10, 3)  # 10 random points
v.add(points)

# Clear scene
v.clear()

# Add a mesh
# See also `Viewer.add_mesh`
import pygfx as gfx
m = gfx.geometries.mobius_strip_geometry()
v.add(m, color='b')

# Close the viewer
v.close()
```

Other selected `Viewer` methods:
- `add()`: generic method to add stuff to the viewer; will call respective specialised methods
- `add_lines()`: add line plot
- `add_mesh()`: add meshes (anything that has `.vertices` and `.faces` goes)
- `add_scatter()`: add a scatter plot
- `add_volume()`: add an image volume
- `center_camera()`: center camera on scene
- `clear()`: clear scene
- `close()`: close viewer
- `colorize()`: cycle colors for all objects
- `pop()`: remove last added object
- `remove()`: remove a given object(s) from the scene
- `screenshot()`: take (and save) a screenshot
- `set_bgcolor()`: set background color
- `set_colors()`: set object colors

### Hotkeys
The following keyboard shortcuts are hard-coded:
- `1`: reset view to XY (frontal)
- `2`: reset view to XZ (dorsal)
- `3`: reset view to YZ (lateral)
- `f`: show FPS
- `c`: show control panel

You can also bind custom functions to keys:

```python
# Bind `x` key to the cycle-colors method
v.key_events['x'] = lambda : v.colorize()
```
