# The Basics

You can use the `octarine.Viewer` in Jupyter environments, (interactive) shells such as [IPython](https://github.com/ipython/ipython) and Python scripts.
The functionality is the same but the handling differs slightly.

## Jupyter Lab/Notebook

```python
import octarine as oc
import pygfx as gfx

# Initialize a new Viewer
v = oc.Viewer()

# Make a simple cube
cube = gfx.Mesh(
    gfx.box_geometry(200, 200, 200),
    gfx.MeshPhongMaterial(color="#336699"),
)

# Add to viewer
v.add(cube)
```

!!! important

    In Jupyter, the `.show()` method generates and displays a widget wrapping the `Viewer`. In a script/terminal,
    the `.show()` message will cause the window to appear. By default, the viewer is displayed right away but you
    can decide to belay that like so:

    ```python
    v = oc.Viewer(show=False)  # set show=False when initializing the viewer
    ...
    do stuff
    ```

    Then in another cell:

    ```python
    v.show()
    ```

![cube example](_static/cube_example_jupyter.png)

!!! tip "Resizing the widget"

    You can adjust the size of the widget either when creating the viewer via
    the `size` argument, or afterwards by clicking + dragging the lower right corner of
    the widget.

### Sidecar

[Sidecar](https://github.com/jupyter-widgets/jupyterlab-sidecar) is a Jupyter widget that
lets you put widgets such as the `Viewer` to the side of your Jupyter window.

To install:

```
pip install sidecar
```

Then in the notebook:

```python
# Initialize a new Viewer
v = oc.Viewer()

# Add the cube we already made
v.add(cube)

# Show widget but put it in the sidecar
v.show(use_sidecar=True)
```

![cube example](_static/cube_example_sidecar.png)

## IPython and other interactive shells

Start IPython by running the `ipython` command from the terminal. Once you're in
the Python shell:

```Python
>>> import octarine as oc
>>> import pygfx as gfx

>>> # Start the main event loop for qt
>>> # (see below for explanation)
>>> %gui qt

>>> # This will immediately open an empty viewer window
>>> v = oc.Viewer()

>>> # Make a simple cube
>>> cube = gfx.Mesh(
...     gfx.box_geometry(200, 200, 200),
...     gfx.MeshPhongMaterial(color="#336699"),
... )

>>> # Add to viewer
>>> v.add(cube)
```

![cube example](_static/cube_example.png)

!!! warning "The main event loop"

    `octarine.Viewer` will need to hook into IPython's main event loop. There are two
    catches here:

    1. You may have to start the event loop manually either via the `%gui` magic command,
    via start-up arguments (e.g. `ipython --gui qt`) or via [config files](https://ipython.org/ipython-doc/dev/config/intro.html).
    2. The active main event loop has to be compatible with the `pygfx` backend you're
    using. At this point, the safest bet is to use `%gui qt`.

    `octarine` is mainly tested with IPython but there are of course other interactive REPLs out
    there such as e.g. [`ptpython`](https://github.com/prompt-toolkit/ptpython). To use
    `octarine` interactively, you may have to figure out how to start the (correct)
    event loop yourself. For example. in case of `ptpython` you need to start it
    with a `--asyncio` flag:

    ```shell
    $ ptpython --asyncio
    ```


## Scripts & non-interactive shell

You can also use `octarine` in Python scripts or from non-interactive shells (like the default `python` shell).
In those scenarios you will have to additionally start the event loop:

```python
import octarine as oc

# Initialize the viewer but don't show yet
v = oc.Viewer(show=False)

# Add random points as scatter
import numpy as np
points = np.random.rand(10, 3)  # 10 random points
v.add(points)

# Show and start the event loop in one go
v.show(start_loop=True)
```

## Offscreen

What if you just want to take a quick snapshot without bothering with a
window/widget? No problem:

```python
import octarine as oc

# Open a fresh offscreen viewer
# (this will not spawn a window)
v = oc.Viewer(offscreen=True)

# Add a mesh to viewer
# (reusing the cube from examples above)
v.add(cube)

# Adjust camera view to frontal
v.set_view('XY')

# Take a snapshot
v.screenshot('cube.png', alpha=True)
```

![cube screenshot](_static/cube_screenshot.png)