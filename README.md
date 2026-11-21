![octarine banner](https://schlegelp.github.io/octarine/_static/octarine_logo_banner.png)
<p align="center">
<i>
Octarine is the eighth color of the Discworld's spectrum, which is described as the color of magic itself. Only wizards and cats can see it.
</i>
</p>

[![docs](https://github.com/schlegelp/octarine/actions/workflows/docs.yml/badge.svg)](https://schlegelp.github.io/octarine/)

# Octarine
A minimalist, easy-to-use, high-performance 3D viewer. `Octarine` is built on top of the excellent
[`pygfx`](https://github.com/pygfx/pygfx) WGPU rendering engine which does most of the heavy lifting - we're simply
abstracting away some of the boiler plate code for you.

## Rationale :thought_balloon:
Why another 3D viewer? There are plenty options out there:
[`vedo`](https://github.com/marcomusy/vedo), [`ipygany`](https://github.com/jupyter-widgets-contrib/ipygany), [`polyscope`](https://github.com/nmwsharp/polyscope), [`napari`](https://github.com/napari/napari), [`fury`](https://github.com/fury-gl/fury), [`plotly`](https://github.com/plotly/plotly.py) or [`pyvista`](https://github.com/pyvista/pyvista) to name but a few. All of these are great in their own right but I wanted something *(a)* without heavy dependencies (i.e. no VTK), *(b)* that lets me interactively explore my data in both REPL and Jupyter and *(c)* is very performant. None of the existing solutions ticked all those boxes for me.

`Octarine` tries to fill that gap:
1. _Lightweight_ with very few direct or indirect dependencies.
2. Works in both _Jupyter_ and _REPL_.
3. _High performance_: a mesh with 15M faces renders with 80 fps at 1080p on a 2023 MacBook Pro.

## Features :sparkles:
- **Objects**: meshes, points, lines, volumes, tubes and sparse volumes (tens of millions of
  voxels, including run-length encoded ones) - see [Adding Objects](https://schlegelp.github.io/octarine/objects/)
- **Looks good out of the box**: shadows, ambient occlusion and a light that follows the camera
  are on by default; environment lighting, matcaps, outlines, silhouettes, subsurface scattering,
  depth of field, tone mapping and gradient backgrounds are one call away - see
  [Effects & Shading](https://schlegelp.github.io/octarine/effects/)
- **Animations**: orbit, fly between views and zoom, then play it in the viewer or render it to
  `.mp4`, `.gif` or PNGs - see [Animations](https://schlegelp.github.io/octarine/animations/)
- **Many objects**: optionally merge objects into fewer draw calls for scenes made of thousands
  of small objects, while still hiding, colouring and picking them one by one - see
  [Managing Objects](https://schlegelp.github.io/octarine/manage/)
- **Controls**: a control panel with legend, effects, screenshots and animations; keep the
  cameras of several viewers in sync - see [Viewer Controls](https://schlegelp.github.io/octarine/controls/)

## Installation :rocket:

```bash
pip install "octarine3d[all]"
```

`Octarine` requires Python 3.11 or later. The above will install the minimal requirements plus `PySide6` and
`jupyter_rfb` as window managers for IPython/shell and Jupyter, respectively. Check out the [**Install Instructions**](https://schlegelp.github.io/octarine/install/)
for details.

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

# Add a mesh (see also `Viewer.add_mesh`)
import pygfx as gfx
m = gfx.geometries.mobius_strip_geometry()
v.add(m, color='b')

# Close the viewer
v.close()
```

> [!NOTE]
> The above example will work in interactive environments such as IPython and Jupyter. When using from the standard
> REPL or when running as a script you will have to additionally start the event loop. Please see the corresponding
> section in the [Introduction](https://schlegelp.github.io/octarine/intro/).

![demo gif](docs/_static/octarine_demo_720p.gif)

## Want to learn more?
Head over to the [**Documentation**](https://schlegelp.github.io/octarine/)!

## Want to contribute?
We welcome all kinds of contributions. For example:

- reports of bugs, broken examples, etc.
- feature requests
- pull requests with bug fixes or new features

If you already know what needs doing, feel free to open a pull request
right away. When in doubt please open an [issue](https://github.com/schlegelp/octarine/issues)
so we can discuss the best way to address the issue.

## Development :dash:

### Tests

To run the test suite:

```bash
pip install -e .[dev]
pytest -v
```

Most tests render with an offscreen canvas, so they need a working GPU driver but no window. Tests for the Qt
control panel are run if `PySide6` is installed.

### Docs

To generate the documentation:

```bash
pip install -e .[docs]
mkdocs build
```