# Extending `Octarine`

In [Adding Objects](objects.md) you learned how to use the built-in
object types. But what if you have want to visualize something not currently supportedby `Octarine`?

Well, in the first instance you can just generate the
`pygfx` visual (e.g. a `pygfx.Mesh` or a `pygfx.Line`) yourself and
use the [octarine.Viewer.add][]`()` method to add them to the scene.

That's probably good enough for the odd one-off but what if you want
to use the `Octarine` viewer for your specialised data on a regular
basis? Easy: you extend `Octarine`'s functionality to include your
data!

Probably easiest to illustrate using an example:

First we will define a custom dummy class

```python
class Point3d:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
```

Next, we need to write a function that converts a `Point3d` into
a `pygfx` object:

```python
import pygfx as gfx

def convert_point3d(point, radius=1, color='red'):
    """Convert a Point3d to a pygfx Sphere."""
    assert isinstance(point, Point3d)

    # The geometry
    geometry = gfx.geometries.sphere_geometry(radius=radius)

    # The material
    material = gfx.MeshPhongMaterial(color=color)

    # Combine
    mesh = gfx.Mesh(geometry, material)

    # Set coordinates
    mesh.local.position = (point.x, point.y, point.z)

    # Return the mesh
    return [mesh]
```

Now that we have a function that converts from `Point3d` to a `pygfx`
visual, we need to tell `Octarine` about it:

```python
import octarine as oc

oc.register_converter(Point3d, convert_point3d)
```

With that we add a `Point3d` to any viewer:

```
p = Point3d(1, 1, 1)

v = oc.Viewer()
v.add(p)
```

![sphere example](_static/sphere_example.png)


If you like can even add a specialized `.add_...()` method:


```python
@oc.viewer.update_legend
def add_point3d(self, point, name=None, color=None, radius=1, center=True):
    """Add Point3d to canvas.

    Parameters
    ----------
    point :     Point3D
                Point to plot.
    name :      str, optional
                Name for the visual.
    color :     str | tuple, optional
                Color to use for the visual.
    center :    bool, optional
                If True, re-center camera to all objects on canvas.

    """
    # This grabs the next available color
    if color is None:
        color = self._next_color()
    # Make sure we have a sensible name for our visual
    if name is None:
        name = self._next_label('Point')
    elif not isinstance(name, str):
        name = str(name)

    visual = convert_point3d(point, color=color, radius=radius)
    visual._object_id = name if name else uuid.uuid4()
    self.scene.add(visual)

    if center:
        self.center_camera()

oc.Viewer.add_point3d = add_point3d
```

Now this should work:

```python
>>> p = Point3d(1, 1, 1)
>>> v = oc.Viewer()
>>> v.add_point3d(p)
```
