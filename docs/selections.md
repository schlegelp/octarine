# Selecting objects

This section will demonstrate how to select objects in your viewer.
`Octarine` offers two options:

1. "Picking" by clicking on individual objects
2. Drawing a selection rectangle to select objects

## Picking

`Octarine` wraps `pygfx`'s basic picking system. This allows you to easily select objects
by double clicking on them:

```python
v = oc.Viewer()

# Make 3 cubes that are slightly offset against each other
import pygfx as gfx
for i in range(3):
    color = ['red', 'green', 'blue'][i]
    cube = gfx.Mesh(
        gfx.box_geometry(200, 200, 200),
        gfx.MeshPhongMaterial(color=color),
    )
    cube.local.x = 300 * i
    v.add(cube, name=f"{color} cube")

# Tell viewer to select objects on double click
v.on_double_click = "select"
```

<video controls>
<source src="../_static/picking_example1.mov" type="video/mp4">
</video>

See the help for `v.on_double_click` for other presets.

!!! tip "Selection color"

    You can change the color used to highlight selected objects:

    ```python
    v.highlight_color = "purple"
    ```

Alternatively, you can also adjust the behavior when hovering over objects:

```python
# Highlight objects when hovering over them
v.on_hover = "highlight"
```

If you want a custom behaviour, have a look at the logic behind the
`Viewer.on_double_click` setter - `pygfx` makes it ridiculously easy to
hook into the event system. We're also happy to extend the current
list of options - please open a [Github Issue](https://github.com/schlegelp/octarine/issues/new)
if you have suggestions!

Last but not least, these setting can also be accessed through the control panel:

![picking controls](_static/picking_controls.png)


## Drawing Selections
In this section you will learn to hook up a selection widget to a viewer to make more complex selections.

We'll start by importing `octarine` and the [`SelectionGizmo`][octarine.selection.SelectionGizmo]:

```python
import octarine as oc
from octarine.selection import SelectionGizmo
```

Next, we will instantiate the viewer and add a few cubes:

```python
v = oc.Viewer()

# Make 3 cubes that are slightly offset against each other
import pygfx as gfx
for i in range(3):
    color = ['red', 'green', 'blue'][i]
    cube = gfx.Mesh(
        gfx.box_geometry(200, 200, 200),
        gfx.MeshPhongMaterial(color=color),
    )
    cube.local.x = 300 * i
    v.add(cube, name=f"{color} cube")
```

![three cubes](_static/selection_example1.png)

Note that we have given our cubes some useful names:
```python
>>> v.objects
OrderedDict([('red cube', [<pygfx.Mesh  at 0x3565dca10>]),
             ('green cube', [<pygfx.Mesh  at 0x356441f10>]),
             ('blue cube', [<pygfx.Mesh  at 0x14ef4c450>])])
```

Next, we will add a [`SelectionGizmo`][octarine.selection.SelectionGizmo] to the viewer:

```python
# Instantiate the gizmo and link it to our viewer
sel = SelectionGizmo(viewer=v)
```

Now if you hold `Shift` and click & drag on the viewer, you should see a selection rectangle being drawn:

![drawing a selection](_static/selection_example2.gif)

Other than that nothing happens. That's because we haven't told the `SelectionGizmo` what do do
once a selection has been made. For that, we need to attach a callback function that accepts
a dictionary describing the selection. Let's write a little function that just prints the content
of that dictionary and attach as callback:

```python
from pprint import pprint

def print_selection(sel_dict):
    pprint("Selection made:")
    pprint(sel_dict)

sel.add_callback(print_selection)
```

Now if we make a selection, you should see something like this in your console:

```python
'Selection made:'
{'blue cube': {'clipped': False,
               'contained': False,
               'objects': [{'clipped': False,
                            'contained': False,
                            'data': None}]},
 'green cube': {'clipped': True,
                'contained': False,
                'objects': [{'clipped': True,
                             'contained': False,
                             'data': array([False, False, False, False, True, True, True, True, True, False, True, False, False, True, False, True, True, False, True, False, False, True, False, True])}]},
 'red cube': {'clipped': True,
              'contained': True,
              'objects': [{'clipped': True, 'contained': True, 'data': None}]}}
```

If you look closely, you will see that the selection rectangle:

1. missed the blue cube
2. clipped the green cube
3. fully contained the red cube

For the clipped green cube, the dictionary gives us an array (`data`) that tells us which vertices of the
cube were in- (`True`) and which were outside (`False`) of the selection box.

Let's do something more elaborate and add a callback that highlight cubes that we selected:

```python
def highlight_selection(sel_dict):
    # First unhighlight all currently highlighted objects
    v.unhighlight_objects()

    # Now highlight the selected neurons
    v.highlight_objects([ob for ob, props in sel_dict.items() if props['clipped'] is True])

sel.add_callback(highlight_selection)
```

![highlighting a selection](_static/selection_example3.gif)