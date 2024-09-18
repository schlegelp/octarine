![octarine banner](https://schlegelp.github.io/octarine/_static/octarine_logo_banner.png)
<p align="center">
<i>
Octarine is the eighth color of the Discworld's spectrum, which is described as the color of magic itself. Only wizards and cats can see it.
</i>
</p>

[![docs](https://github.com/schlegelp/octarine/actions/workflows/docs.yml/badge.svg)](https://schlegelp.github.io/octarine/)

# Octarine
A minimalist, easy-to-use, high-performance 3D viewer. `Octarine` is build on top of the excellent
[`pygfx`](https://github.com/pygfx/pygfx) WGPU rendering engine which does most of the heavy lifting - we're simply
abstracting away some of the boiler plate code for you.

## Rationale :thought_balloon:
Why another 3D viewer? There are plenty options out there:
[`vedo`](https://github.com/marcomusy/vedo), [`ipygany`](https://github.com/jupyter-widgets-contrib/ipygany), [`polyscope`](https://github.com/nmwsharp/polyscope), [`napari`](https://github.com/napari/napari), [`fury`](https://github.com/fury-gl/fury), [`plotly`](https://github.com/plotly/plotly.py) or [`pyvista`](https://github.com/pyvista/pyvista) to name but a few. All of these are great in their own right but I wanted something *(a)* without heavy dependencies (i.e. no VTK), *(b)* that lets me interactively explore my data in both REPL and Jupyter and *(c)* is very performant. None of the existing solutions ticked all those boxes for me.

`Octarine` tries to fill that gap:
1. _Lightweight_ with very few direct or indirect dependencies.
2. Works in both _Jupyter_ and _REPL_.
3. _High performance_: a mesh with 15M faces renders with 80 fps at 1080p on a 2023 MacBook Pro.

## Installation :rocket:

```bash
pip install "octarine3d[all]"
```

This will install the minimal requirements plus `PySide6` and `jupyter_rfb` as window managers for IPython/shell
and Jupyter, respectively. Check out the [**Install Instructions**](https://schlegelp.github.io/octarine/install/)
for details.

## Status :ballot_box_with_check:
All basic components have been implemented but this is a very young project, which means that the API can still change with each version. We'd love for you to take it for a spin and let us know what you think though!

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
> REPL or when running as a script you will have to additionally start the event loop. Please see corresponding the
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
TODO

### Docs

To generate the documentation:

```bash
pip install -e .[docs]
mkdocs build
```