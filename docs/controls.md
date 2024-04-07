# Controlling the Viewer

In [The Basics](intro.md), you already learned how to open a new
[`octarine.Viewer`][] and add a simple mesh.

Here we will demonstrate various ways to programmatically control the viewer.

## Closing the viewer
Use [`octarine.Viewer.close`][]`()` to close the viewer:

```python
>>> import octarine as oc
>>> v = oc.Viewer()
>>> v.close()
```

## Adjust size
Use [`octarine.Viewer.resize`][]`()` to adjust the size of the viewer:

```python
>>> v = oc.Viewer()
>>> v.resize((1000, 1000))
```

## Camera
Use [`octarine.Viewer.center_camera`][]`()` to center the camera onto objects in the scene:

```python
>>> v = oc.Viewer()
>>> v.center_camera()
```

Use [`octarine.Viewer.set_view`][]`()` to set the camera view:

```python
>>> v = oc.Viewer()
>>> v.set_view('XY')  # set view to frontal
```

## Colors
Use [`octarine.Viewer.colorize`][]`()` to randomize colors:

```python
>>> v = oc.Viewer()
>>> v.colorize(palette='seaborn:tab10')
```

Use [`octarine.Viewer.set_colors`][]`()` to set colors for given objects:

```python
>>> v = oc.Viewer()
>>> v.add(cube, name='cube')
>>> v.set_colors('w')  # set color for all objects
>>> v.set_colors({'cube': 'r'})  # set colors for individual objects
```

Use [`octarine.Viewer.set_bgcolor`][]`()` to change the background color:

```python
>>> v = oc.Viewer()
>>> v.set_bgcolor("white")
```

## Hotkeys
While the viewer or widget is active you can use a set of hotkeys to control the viewer:

| Hotkey | Description                                  |
|--------|----------------------------------------------|
| `1`    | Set frontal (XY) view                        |
| `2`    | Set dorsal (XZ) view                         |
| `3`    | Set lateral (YZ) view                        |
| `f`    | Show/hide frames per second                  |
| `c`    | Show/hide control panel (requires PySide6)   |

## GUI Controls

`Octarine` provides basic GUI controls when run from the shell. These currently
require [PySide6](https://pypi.org/project/PySide6/) to be installed and you
may have to use `%gui qt6` when inside `IPython`.

Then you can either press the `c` hotkey or:

```
>>> v = oc.Viewer()
>>> v.show_controls()
```

<center><img src="https://schlegelp.github.io/octarine/_static/controls_example.png" alt="controls" width="300"/></center>