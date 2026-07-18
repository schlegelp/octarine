# Effects & Shading

`Octarine` offers a number of ways to change how the scene is rendered.
They roughly fall into two categories:

1. Per-object material settings: [transparency](#transparency-alpha-modes)
   and [silhouette rendering](#silhouette-rendering)
2. Screen-space post-processing passes applied to the rendered image:
   [`add_effect`](#post-processing-effects) and
   [depth of field](#depth-of-field)

Most of what follows requires `pygfx>=0.16` (which is what recent versions
of `Octarine` install anyway).

## Transparency (alpha modes)

Whenever objects are semi-transparent, the renderer has to decide how to
blend their colors with whatever is behind them. `pygfx` (`>=0.13`) handles
this via per-material "alpha modes" and
[`octarine.Viewer.set_alpha_mode`][] provides a high-level interface to
these:

```python
>>> import octarine as oc
>>> v = oc.Viewer()
>>> v.add_mesh(mesh, name='bunny', alpha=0.5)

>>> # Set the alpha mode for all objects...
>>> v.set_alpha_mode('weighted_blend')

>>> # ... or only for specific ones
>>> v.set_alpha_mode('add', objects=['bunny'])
```

See `pygfx.Material.alpha_mode` (e.g. via `help()`) for the available
modes. Note that `Octarine` will automatically pick a sensible alpha mode
when you change an object's opacity: `"add"` for semi-transparent objects,
`"opaque"` otherwise. If you need full control, you can always set
`material.alpha_mode` on individual `pygfx` objects.

## Silhouette rendering

[`octarine.Viewer.set_silhouette`][] enables
[Neuroglancer](https://github.com/google/neuroglancer)-style silhouette
rendering for meshes: face-on regions become transparent while edges and
creases are emphasized, giving an x-ray-like view of the mesh's outline.

```python
>>> v = oc.Viewer()
>>> v.add_mesh(mesh, name='bunny')

>>> # Enable silhouette rendering for all meshes
>>> v.set_silhouette(2)

>>> # Adjust the strength (typical values are 1-8)
>>> v.set_silhouette(6)

>>> # Disable again
>>> v.set_silhouette(0)
```

![silhouette off](_static/effects_silhouette_before.png){ width="49%" }
![silhouette on](_static/effects_silhouette_after.png){ width="49%" }

Under the hood, fragments are multiplied by
`pow(1 - |dot(normal, view_dir)|, silhouette)` - i.e. the `silhouette`
exponent has the same semantics as Neuroglancer's "silhouette" property.

Use the `objects` parameter to apply the effect to only some meshes. Note
that this only works for meshes with Phong-based materials - other objects
are silently skipped.

You can also enable the effect for a mesh right when adding it:

```python
>>> v.add_mesh(mesh, silhouette=2)
```

## Post-processing effects

[`octarine.Viewer.add_effect`][] adds a post-processing pass to the
renderer - i.e. an effect applied to the fully rendered image:

```python
>>> v = oc.Viewer()
>>> v.add_mesh(mesh)

>>> # Add Eye-Dome Lighting
>>> v.add_effect('edl')

>>> # Call again to adjust parameters of an existing effect
>>> v.add_effect('edl', strength=8)

>>> # Remove the effect again
>>> v.add_effect('edl', disable=True)
```

Currently supported effects:

| Effect     | Description                                                    | Parameters |
|------------|----------------------------------------------------------------|------------|
| `"edl"`    | Eye-Dome Lighting: darkens edges based on depth differences, enhancing depth perception for complex geometries. | `strength` (default 5), `radius`, `depth_edge_threshold` |
| `"noise"`  | Adds noise to the image.                                       | `noise` (default 0.1) |
| `"fog"`    | Adds fog based on the depth buffer.                            | `color` (default `"#fff"`), `power` (default 1) |
| `"depth"`  | Renders scene depth as shades of grey (near = dark, far = light), normalized to the visible geometry. With `overlay=True` the objects' own colors are kept and darkened with distance instead (depth cueing). | `overlay` (default False), `strength` (default 0.9) |
| `"normal"` | Renders normals reconstructed from the depth buffer.           | - |
| `"bloom"`  | Physically-based bloom: makes bright regions glow.             | `bloom_strength`, `max_mip_levels`, `filter_radius`, `use_karis_average` |

See the [`octarine.Viewer.add_effect`][] reference for details on the
individual parameters.

Here is what these effects look like (click to enlarge; the bloom example
uses `bloom_strength=0.5` to make the effect more obvious):

![no effect](_static/effects_baseline.png){ width="49%" }
![edl](_static/effects_edl.png){ width="49%" }
![noise](_static/effects_noise.png){ width="49%" }
![fog](_static/effects_fog.png){ width="49%" }
![depth](_static/effects_depth.png){ width="49%" }
![bloom](_static/effects_bloom.png){ width="49%" }

## Depth of field

[`octarine.Viewer.set_depth_of_field`][] adds a photographic focal blur:
objects near the focal plane are rendered sharp while everything closer or
farther is progressively blurred.

```python
>>> v = oc.Viewer()
>>> v.add_mesh(mesh)

>>> # Enable with default settings: continuously auto-focus on
>>> # whatever is at the center of the view
>>> v.set_depth_of_field()

>>> # Stronger blur, eased focus transitions
>>> v.set_depth_of_field(aperture=200, smooth=True)

>>> # Fix the focal plane at a given distance from the camera
>>> v.set_depth_of_field(focus=1000)

>>> # Disable again
>>> v.set_depth_of_field(False)
```

![no effect](_static/effects_baseline.png){ width="49%" }
![depth of field](_static/effects_dof.png){ width="49%" }

The most important parameters:

- `focus`: distance of the focal plane from the camera in world units; if
  `None` (default) the effect continuously auto-focuses on whatever is at
  the center of the view (over empty space the image is left sharp)
- `aperture`: blur strength - the blur radius in physical pixels of a point
  at 100% relative defocus; typical values are 50-300
- `max_radius`: upper limit for the blur radius in physical pixels
- `smooth`: if truthy, autofocus changes are eased over approximately this
  many seconds (`True` = 0.2s) instead of snapping instantly
- `snap_radius`: autofocus search radius in physical pixels around the view
  center - the effect focuses on the object closest to the view center
  within that radius

!!! note

    Depth of field is a screen-space effect: it applies to the entire
    rendered image (including overlay elements such as messages), and
    objects that don't write depth (e.g. meshes with a transparent alpha
    mode) are blurred by whatever is behind them.

## Effects in the GUI

Silhouette and depth of field can also be toggled and tuned interactively
from the "Effects" tab of the [control panel](controls.md#gui-controls).

## Under the hood

The custom shaders and render passes powering these features live in the
`octarine.shaders` module - see the [API reference](api/shaders.md). The
module is imported lazily and requires `pygfx>=0.16`.
