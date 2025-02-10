# Managing objects

As you add objects to the [octarine.Viewer][], you might want to keep track of
them so you can e.g. colorize or remove them at a later point.

Unless specified, each object gets a generic identifier:

```python
>>> duck = tm.load_remote(
...         'https://github.com/mikedh/trimesh/raw/main/models/Duck.glb'
...     )

>>> # Add the duck
>>> v.add_mesh(duck)
```

![bunny example](_static/duck_example.png)

Let's check what objects are there:

```python
>>> v.objects
OrderedDict([('Object', [<pygfx.Mesh at 0x37f4f8b90>])])
```

The `.objects` property will return a dictionary mapping IDs to `pygfx` visuals.

Alternatively, you can also do this:

```python
>>> v['Object']
[<pygfx.Mesh at 0x37f4f8b90>]
```

Instead of generic IDs, we can also explicitly set the ID:

```python
>>> v.add_mesh(duck, name='Duck')
>>> v.objects
OrderedDict([('Duck', [<pygfx.Mesh at 0x37f5f8a12>])])
```

This can also be used to combine multiple objects under the same ID.

Why are these IDs relevant? Well, they help you manipulate objects after
they've been added:

```python
>>> v.set_colors({'Duck': 'w'})
>>> v.hide_objects('Duck')
>>> v.remove_objects('Duck')
```

## What next?

<div class="grid cards" markdown>

-   :material-cube:{ .lg .middle } __Objects__

    ---

    Manipulate viewer and objects (color, size, visibility, etc).

    [:octicons-arrow-right-24: Viewer Controls](controls.md)

-   :material-select:{ .lg .middle } __Selection__

    ---

    Selecting objects on the viewer.

    [:octicons-arrow-right-24: Selection](selection.md)

</div>

