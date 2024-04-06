# Adding Objects to the Viewer

Off the bat `Octarine` supports four types of objects all of
which have dedicated `Viewer` methods:

|   | Type          | Viewer method                  |
|---|---------------|--------------------------------|
| 1.| Meshes        | [octarine.Viewer.add_mesh][]   |
| 2.| Points        | [octarine.Viewer.add_points][] |
| 3.| Lines         | [octarine.Viewer.add_lines][]  |
| 4.| Image Volumes | [octarine.Viewer.add_volume][] |

As a general entry point you can use the [octarine.Viewer.add][] method
which will pass an object to the respective specialized function:

```python
>>> v = oc.Viewer()
>>> # This ...
>>> v.add(mesh)
>>> # ... is effectively the same as this
>>> v.add_mesh(mesh)
```

## Meshes

`Octarine` will happily work with anything that's mesh-like - i.e. anything that
has `.vertices` and `.faces`. In practice, I'd recommend you use
[`trimesh`](https://github.com/mikedh/trimesh) to e.g. load meshes from files:

```python
>>> import octarine as oc
>>> import trimesh as tm

>>> mesh = tm.load_remote(
...         'https://github.com/mikedh/trimesh/raw/main/models/bunny.ply'
...     )
>>> type(mesh)
<trimesh.Trimesh(vertices.shape=(8146, 3), faces.shape=(16301, 3), name=`bunny.ply`)

>>> v = oc.Viewer()
>>> v.add_mesh(mesh, name='bunny')
```

![bunny example](_static/bunny_example.png)


## Points
TODO

## Lines
TODO

## Image Volumes

Image volumes are expected to be 3d `numpy` arrays or `trimesh.VoxelGrids`.

TODO

## Managing objects

As you add objects to the [octarine.Viewer][], you might want to keep track of
them so you can e.g. colorize or remove them.

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

Instead of generic IDs, we can also explicitly set the ID:

```python
>>> v.add_mesh(duck, name='Duck')
>>> v.objects
OrderedDict([('Duck', [<pygfx.Mesh at 0x37f4f8b90>])])
```

This can also be used to combine multiple objects under the same ID.

Why are these IDs relevant? Well, they help you manipulate objects after
they've been added:

```
>>> v.set_colors({'Duck': 'w'})
>>> v.hide_objects('Duck')
```

## Extending `Octarine`

TODO