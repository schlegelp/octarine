# Adding Objects to the Viewer

Off the bat `Octarine` supports 4 types of objects, all of
which have dedicated `Viewer` methods:

|   | Object Type                       | Viewer method                  |
|---|-----------------------------------|--------------------------------|
| 1.| [Meshes](#meshes)                 | [octarine.Viewer.add_mesh][]   |
| 2.| [Points](#points)                 | [octarine.Viewer.add_points][] |
| 3.| [Lines](#lines)                   | [octarine.Viewer.add_lines][]  |
| 4.| [Image Volumes](#image-volumes)   | [octarine.Viewer.add_volume][] |

As a general entry point you can use the [octarine.Viewer.add][]`()` method
which will pass an object to the respective specialized function:

```python
>>> v = oc.Viewer()
>>> # This ...
>>> v.add(mesh)
>>> # ... is effectively the same as this
>>> v.add_mesh(mesh)
```
!!! tip

    The specialised methods may offer more ways to customize the visual.

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

See [octarine.Viewer.add_mesh][]`()` for details!

## Points

Points are expected to be 2d `(N, 3)` numpy arrays:

```python
>>> import octarine as oc
>>> v = oc.Viewer()

>>> # Add random points as scatter
>>> import numpy as np
>>> points = np.random.rand(10, 3)  # 10 random points
>>> v.add_points(points color='r')
```

![points example](_static/points_example.png)

See [octarine.Viewer.add_points][]`()` for details!

## Lines

Lines are expected to be either a `(N, 3)` numpy array
representing a single contiguous line or a list thereof:


```python
>>> import octarine as oc
>>> v = oc.Viewer()

>>> import numpy as np
>>> line1 = np.random.rand(10, 3)  # points for line 1
>>> line2 = np.random.rand(5, 3)  # points for line 2
>>> v.add_lines([line1, line2], color='y')
```

![lines example](_static/lines_example.png)

See [octarine.Viewer.add_lines][]`()` for details!

## Image Volumes

Image volumes are expected to be 3d `numpy` arrays or `trimesh.VoxelGrids`.

In this example, we're using [pynrrd](https://pypi.org/project/pynrrd/) to read
an image stack of a _Drosophila_ brain downloaded from figshare
([link](https://figshare.com/s/43ea65ba938e64312f32)):

```python
>>> import nrrd
>>> vol, meta = nrrd.read('JRC2018_UNISEX_38um_iso_16bit.nrrd')
>>> # Dimensions along each axis are 0.38 microns
>>> meta['sizes]
array([[0.38, 0.  , 0.  ],
       [0.  , 0.38, 0.  ],
       [0.  , 0.  , 0.38]])

>>> import octarine as oc
>>> v = oc.Viewer()
>>> v.add_volume(vol, spacing=(.38, .38, .38))
>>> v.show_bounds = True
```

![brain volume](_static/brain_volume_example.png)

Note that the default blend mode for the renderer may cause objects
behind or inside the volume to look funny. You can change the blend
mode by setting e.g. `v.blend_mode='additive'`.

Alternatively, you can also add slices through the volume:

```python
>>> import cmap
>>> v.add_volume(
...     vol,
...     color=cmap.Colormap('Greys'),  # use a different colormap
...     spacing=.38,  # single value for isometric data
...     slice=True  # can also be a tuple, e.g. (True, False, True)
... )
```

![brain volume](_static/brain_volume_example2.png)

See [octarine.Viewer.add_volume][]`()` for details!

## Custom Objects

What if you have want to visualize something not currently supported
by `Octarine`?

Go check out the [Extending Octarine](extending.md) tutorial to learn how!

