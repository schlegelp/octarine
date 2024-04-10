![banner](_static/octarine_logo_banner.png)

# Overview
A minimalist, easy-to-use, high-performance 3D viewer. `Octarine` is build on top of the excellent
[`pygfx`](https://github.com/pygfx/pygfx) WGPU rendering engine which does most of the heavy lifting - we're simply
abstracting away some of the boiler plate code for you.

## Rationale :thought_balloon:
Why another 3D viewer? There are already plenty options out there:
[`vedo`](https://github.com/marcomusy/vedo), [`ipygany`](https://github.com/jupyter-widgets-contrib/ipygany), [`polyscope`](https://github.com/nmwsharp/polyscope), [`napari`](https://github.com/napari/napari), [`fury`](https://github.com/fury-gl/fury), [`plotly`](https://github.com/plotly/plotly.py) or [`pyvista`](https://github.com/pyvista/pyvista) to name but a few. All of these are great in their own right but I wanted something *(a)* without heavy dependencies (i.e. no VTK), *(b)* that lets me interactively explore my data in both REPL and Jupyter and *(c)* is very performant. None of the existing solutions ticked all those boxes for me.

`Octarine` tries to fill that gap:

1. _Lightweight_ with very few direct or indirect dependencies
2. Works in both _Jupyter_ and Python shell (e.g. `IPython`)
3. _High performance_: a mesh with 15M faces renders with 80 fps at 1080p on a 2023 MacBook Pro

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

## What next?

<div class="grid cards" markdown>

-   :material-progress-wrench:{ .lg .middle } __Install__

    ---

    Instructions on how to install `Octarine`.

    [:octicons-arrow-right-24: Installation](install.md)

-   :material-eye-arrow-right:{ .lg .middle } __Viewer Basics__

    ---

    Learn about using `Octarine` in different environments.

    [:octicons-arrow-right-24: The Basics](intro.md)

-   :material-cube:{ .lg .middle } __Objects__

    ---

    Check out the guide on different object types.

    [:octicons-arrow-right-24: Adding Objects](objects.md)

-   :material-format-font:{ .lg .middle } __Animations__

    ---

    Add movement to the viewer.

    [:octicons-arrow-right-24: Animations](animations.md)

-   :material-camera-control:{ .lg .middle } __Control__

    ---

    Learn how to control the viewer, adjust colors, etc.

    [:octicons-arrow-right-24: Controls](controls.md)

</div>