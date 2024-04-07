![banner](_static/octarine_logo_banner.png)

# Overview
A minimalist, easy-to-use, high-performance 3D viewer. `Octarine` is build on top of the excellent
[`pygfx`](https://github.com/pygfx/pygfx) WGPU rendering engine which does most of the heavy lifting - we're simply
abstracting away some of the boiler plate code for you.

## Rationale :thought_balloon:
Why another 3d viewer? There are plenty options out there:
[`vedo`](https://github.com/marcomusy/vedo), [`ipygany`](https://github.com/jupyter-widgets-contrib/ipygany), [`polyscope`](https://github.com/nmwsharp/polyscope), [`napari`](https://github.com/napari/napari), [`fury`](https://github.com/fury-gl/fury), [`plotly`](https://github.com/plotly/plotly.py) or [`pyvista`](https://github.com/pyvista/pyvista) to name but a few. All of these are great in their own right but I wanted something *(a)* without heavy dependencies (i.e. no VTK), *(b)* that lets me interactively explore my data in both REPL and Jupyter and *(c)* is very performant. None of the existing solutions ticked all those boxes for me.

`Octarine` tries to fill that gap:

1. _Lightweight_ with very few direct or indirect dependencies.
2. Works in both _Jupyter_ and _REPL_.
3. _High performance_: a mesh with 15M faces renders with 80 fps at 1080p on a 2023 MacBook Pro.

## Example

```python
import octarine as oc
import trimesh as tm

# Load an example from trimesh
meshes = tm.load_remote(
    'https://github.com/mikedh/trimesh/raw/main/models/CesiumMilkTruck.glb'
    )

# Open a fresh viewer
v = oc.Viewer()

# Add mesh to viewer
v.add(meshes)
```

![example](_static/milktruck_example.png)

## Installation

`Octarine` is published as a [Python package] and can be installed with
`pip`, ideally by using a [virtual environment]. Open up a terminal and install
`Octarine` with:

=== "Latest"

    ``` sh
    pip install octarine3d
    ```

=== "dev"

    ``` sh
    pip install git+https://github.com/schlegelp/octarine.git
    ```

!!! tip

    If you don't have prior experience with Python, we recommend reading
    [Using Python's pip to Manage Your Projects' Dependencies], which is a
    really good introduction on the mechanics of Python package management and
    helps you troubleshoot if you run into errors.

  [Python package]: https://pypi.org/project/octarine3d/
  [virtual environment]: https://realpython.com/what-is-pip/#using-pip-in-a-python-virtual-environment
  [semantic versioning]: https://semver.org/
  [upgrade to the next major version]: upgrade.md
  [Markdown]: https://python-markdown.github.io/
  [Pygments]: https://pygments.org/
  [Python Markdown Extensions]: https://facelessuser.github.io/pymdown-extensions/
  [Using Python's pip to Manage Your Projects' Dependencies]: https://realpython.com/what-is-pip/