# :simple-keepachangelog: Changelog

This is a selection of features added, changes made and bugs fixed with each version.
For a full list of changes please see the [commits history](https://github.com/schlegelp/octarine/commits/main)
on the Github repository.

## dev
_Date: ongoing_

#### Breaking
- `Octarine` now requires `pygfx>=0.17` - `0.17` renamed two symbols in the shader API that
  `octarine`'s custom shaders build on (`lighting_phong()` no longer takes `is_front`, and
  `physical_albeido` is now spelled `physical_albedo`)

#### Improvements
- new [`Viewer.set_ambient_occlusion`][octarine.Viewer.set_ambient_occlusion] method (also
  available as `add_effect('ao')`): screen-space ambient occlusion, i.e. the shadowing that
  ambient light would produce in creases, cavities and where objects touch - `pygfx` has no
  ambient occlusion of its own beyond baked `ao_map` textures. Occlusion is estimated from the
  depth buffer, so no preprocessing and no extra geometry pass is needed, and the whole thing
  costs well under a millisecond per frame at the default 16 samples. `radius` (defaulting to
  4% of the scene's diagonal) is the parameter that has to match the scene; `intensity`, `bias`,
  `samples`, `power` and `blur` tune the look, and `debug=True` renders the raw occlusion
  (see [Ambient occlusion](effects.md#ambient-occlusion)). With an automatic radius the effect
  now also tracks the scene, i.e. the radius is re-derived as objects are added or removed
- control panel: the "Effects" tab now has one collapsible section per effect (the checkbox
  switches the effect on, the arrow expands its parameters) and gained controls for
  [`Viewer.shadows`][octarine.Viewer.shadows] and
  [`Viewer.set_ambient_occlusion`][octarine.Viewer.set_ambient_occlusion]
  (see [GUI Controls](controls.md#gui-controls))
- new [`Viewer.link`][octarine.Viewer.link] / [`Viewer.unlink`][octarine.Viewer.unlink] methods
  (plus a [`Viewer.linked`][octarine.Viewer.linked] property): keep the camera synchronised
  between two or more viewers, so that panning, rotating or zooming in one of them does the
  same in all the others - handy for comparing things side by side. Links are symmetrical and
  transitive, and `sync`/`exclude` let you share only part of the camera state (e.g. rotate
  together but zoom separately). Viewers need not show the same data or use the same controls
  (see [Linking viewers](controls.md#linking-viewers))
- new [`Viewer.set_subsurface`][octarine.Viewer.set_subsurface] method (plus a `subsurface`
  parameter for [`Viewer.add_mesh`][octarine.Viewer.add_mesh]): render meshes as translucent,
  so that backlit regions glow and shading eases past the terminator instead of dropping off
  abruptly - the look of skin, wax, marble or thin neurites. Tunable via `scatter_color`,
  `thickness`, `distortion`, `falloff`, `wrap` and `glow`; composes with `set_silhouette`
  (see [Subsurface scattering](effects.md#subsurface-scattering))
- new [`Viewer.set_bg_gradient`][octarine.Viewer.set_bg_gradient] method: use a radial ("studio")
  gradient as background - a soft pool of light behind the object fading into near-black towards
  the edges of the frame. Comes with six presets (`graphite`, `cinematic`, `warm`, `olive`,
  `burgundy`, `halo`), each of which can be tweaked via `colors`, `center`, `radius`, `falloff`
  and `vignette` (see [Background](controls.md#background)). The presets are also available from
  a new `Background` dropdown in the GUI controls (both the Qt panel and the Jupyter toolbar)
- [`Viewer.set_bgcolor`][octarine.Viewer.set_bgcolor] now also takes two colors (for a
  bottom-to-top gradient) or four colors (one per corner)
- new [`Viewer.add_tubes`][octarine.Viewer.add_tubes] method: render skeletons with a per-node
  radial profile as tubes, straight from the coefficients with no mesh in between. The surface is
  generated in the vertex shader, so angular level of detail via `n_theta`, `k` and `k_normal` is
  a re-draw rather than a re-upload; `axial_lod` thins the skeleton without touching its topology.
  Frames are realigned onto one rotation-minimizing chain over the whole tree, so the quads
  spanning a branch point are no longer twisted. Note that a swept surface self-intersects
  wherever the radius outruns the centreline's curvature, which rasterised skeletons routinely
  trigger; dropping harmonics via `k` is currently the only effective lever against it
  (see [Tubes](objects.md#tubes))
- new [`Viewer.headlight`][octarine.Viewer.headlight] property (plus a `headlight` parameter for
  [`octarine.Viewer`][] and a checkbox in the GUI controls): link the light to the camera such
  that objects are always lit from the front; accepts a float or tuple to also set the light's
  offset from the camera's axis (see [Lighting](controls.md#lighting))
- new `Viewer.lights` property: list all light sources illuminating the scene
- [`Viewer.add_sparse_volume`][octarine.Viewer.add_sparse_volume]: new `mode="surface"` which
  renders the volume as a shaded isosurface (normals from the volume's gradient) instead of a
  flat blob; the level is set via the new `threshold` parameter (see
  [Sparse Volumes](objects.md#surface-rendering))
- [`Viewer.add_sparse_volume`][octarine.Viewer.add_sparse_volume]: new `smoothing` parameter for
  `mode="surface"` which widens the filter the surface normal is taken from, removing the
  voxel-scale stipple from the shading. The isosurface itself is left alone, so the silhouette
  is unchanged and no thin structures are lost (see
  [Smoothing](objects.md#smoothing))
- [`Viewer.add_sparse_volume`][octarine.Viewer.add_sparse_volume]: `mode="density"` now
  accumulates properly instead of saturating on the first voxel; the extinction per voxel is
  set via the new `density` parameter (previously it was tied to `opacity`, whose default of
  `1` made the volume opaque at the first sample)

- [`Viewer.add_sparse_volume`][octarine.Viewer.add_sparse_volume] now accepts run-length encoded
  voxels as an `(N, 4)` array of `(x, y, z, x_run_length)` - the layout DVID's `sparsevol`
  endpoint returns (plus the [`octarine.VoxelRuns`][] container). Runs are rendered from a
  bit-per-voxel bitmask with a sparse two-level index, which uses ~23x less GPU memory than the
  byte-per-voxel atlas (14 MB vs 328 MB for a 31M-voxel neuron) and never needs the coordinates
  materialized. Binary occupancy only - passing `values` alongside runs raises. Binary `(N, 3)`
  coordinates can opt in with `method="bitmask"` (see
  [Sparse Volumes](objects.md#run-length-encoded-voxels))

#### Fixes
- adding many objects one by one is no longer quadratic in the number of objects: the bounding
  box visual, the camera centering, the fit of the shadow-casting lights, the ambient occlusion
  radius and the environment all have to follow the scene, and each of them walks every visual
  on the canvas. They are now re-fitted once, immediately before the next frame, rather than on
  every single `add`. Filling a viewer with 800 objects went from ~11 s to ~0.3 s; the effect is
  most pronounced with shadows and ambient occlusion (now on by default), but the centering
  alone accounted for most of it. Note that this means those properties only catch up on the
  next draw - `Viewer.bounds` is unaffected and always reports the scene as it currently stands,
  as is an explicit [`Viewer.center_camera`][octarine.Viewer.center_camera], which centers there
  and then. A camera you set up yourself after an `add` (e.g. via `camera.show_object`) also
  still wins, i.e. the deferred centering does not overrule it
- `add_volume`/`add_sparse_volume`: a hex color string (e.g. `"#ff9955"`) raised
  `Colormap '#ff9955' not found` instead of being used as a single color
- fixed colormaps wrapping around: values at the very top of `clim` sampled the colormap at
  texcoord `1.0`, where pygfx's default `wrap="repeat"` blended the last color with the
  (transparent, if `hide_zero=True`) first one - halving both color and alpha. Affects volumes
  and sparse volumes

#### Breaking
- shadows, the camera-linked headlight and ambient occlusion are now **on by default**, i.e.
  scenes look different out of the box: objects cast shadows onto each other, are always lit
  from the front and get contact shadows in their creases and cavities. Each can be switched
  off individually via the new `shadows` / `ambient_occlusion` parameters and the existing
  `headlight` parameter (or the matching properties/methods), e.g.
  `oc.Viewer(shadows=False, headlight=False, ambient_occlusion=False)` gets you the previous
  look. Note that fitting the shadows and the occlusion radius to the scene costs an extra
  pass over the objects, which is made once per frame in which the scene changed
- `opacity` no longer scales the extinction per voxel in `add_sparse_volume(mode="density")`;
  it is now a plain global opacity in all modes and `density` takes its place

To install the current `dev` version of `octarine`:

```shell
pip uninstall octarine3d -y
pip install git+https://github.com/schlegelp/octarine.git
```

## Version `0.7.0` { data-toc-label="0.7.0" }
_Date: 02/08/26_

#### Improvements
- new [`Viewer.set_scalebar`][octarine.Viewer.set_scalebar] method: overlay a scale bar that
  either shows a fixed distance or dynamically adjusts to the zoom level (see
  [Scale bar](controls.md#scale-bar))
- [`Viewer.add_mesh`][octarine.Viewer.add_mesh] now exposes pygfx's mesh shaders via the new `shader` parameter
- [`Viewer.add_lines`][octarine.Viewer.add_lines] now accepts an array of line widths (one per point)

#### Fixes
- fixed issue in [`Viewer.hide_selected`][octarine.Viewer.hide_selected]

## Version `0.6.0` { data-toc-label="0.6.0" }
_Date: 18/07/26_

#### Breaking
- `Octarine` now requires `pygfx>=0.16` (needed for the new custom shaders)

#### Improvements
- new [`Viewer.add_sparse_volume`][octarine.Viewer.add_sparse_volume] method (plus the [`octarine.VoxelCloud`][]
  container): render tens of millions of voxels from an `(N, 3)` array of coordinates using a custom
  brick-based raycasting shader (see [Sparse Volumes](objects.md#sparse-volumes))
- new [`Viewer.set_silhouette`][octarine.Viewer.set_silhouette] method (plus a `silhouette` parameter
  for `add_mesh`): Neuroglancer-style silhouette rendering for meshes (see [Effects & Shading](effects.md#silhouette-rendering))
- new [`Viewer.set_depth_of_field`][octarine.Viewer.set_depth_of_field] method: focal-blur post-processing
  with continuous autofocus (see [Effects & Shading](effects.md#depth-of-field))
- [`Viewer.add_effect`][octarine.Viewer.add_effect]: new effects `"noise"`, `"fog"`, `"depth"`
  (normalized depth / depth cueing), `"normal"` and `"bloom"`; new `disable` parameter to remove effects
- [`Viewer.add_points`][octarine.Viewer.add_points]: new marker options (`marker`, `size_space`,
  `edge_size_space`, `min_size`, `max_size`, `edge_width`, `edge_color`, `edge_mode`, `min_edge_width`)
  backed by a custom points shader (see [Markers, sizes and edges](objects.md#markers-sizes-and-edges))
- control panel: new "Screenshot" (save to file or clipboard) and "Effects" tabs
- IPython: the event loop is now started with `%gui qt` instead of `%gui qt6`

#### Fixes
- legend: color button and visibility checkbox are shown from the start; the color picker is synced
  to the active object

**Full Changelog**: [v0.5.0...v0.6.0](https://github.com/schlegelp/octarine/compare/v0.5.0...v0.6.0)

## Version `0.5.0` { data-toc-label="0.5.0" }
_Date: 12/06/26_

#### Improvements
- `Viewer.on_double_click` now also accepts a custom callable (see
  [Custom callbacks](selections.md#custom-callbacks))
- new `Viewer.objects_grouped` property
- legend improvements: group contents are created lazily, new filter field, hovering over an entry
  highlights the object in the viewer, group labels show member counts

**Full Changelog**: [v0.4.1...v0.5.0](https://github.com/schlegelp/octarine/compare/v0.4.1...v0.5.0)

## Version `0.4.1` { data-toc-label="0.4.1" }
_Date: 03/06/26_

#### Fixes
- don't import the `controls` module (and hence `PySide6`) at top level

**Full Changelog**: [v0.4.0...v0.4.1](https://github.com/schlegelp/octarine/compare/v0.4.0...v0.4.1)

## Version `0.4.0` { data-toc-label="0.4.0" }
_Date: 10/04/26_

#### Improvements
- objects can now be assigned to a `group` (e.g. `Viewer.add(x, group="my group")`); groups show up
  as collapsible entries in the legend (see [Grouping objects](manage.md#grouping-objects))
- `Viewer.add_animation`: new `on_error="log"` option

#### Fixes
- legend: the color button now follows programmatic color changes; long labels are truncated
- fixed the color picker when multiple viewers are open

**Full Changelog**: [v0.3.5...v0.4.0](https://github.com/schlegelp/octarine/compare/v0.3.5...v0.4.0)

## Version `0.3.5` { data-toc-label="0.3.5" }
_Date: 24/03/26_

#### Improvements
- new `octarine.video_helpers` module with [`make_rotation_video`][octarine.video_helpers.make_rotation_video]
  (see [Recording videos](animations.md#recording-videos))
- the event-loop warning can now be suppressed

#### Fixes
- fixed Jupyter detection

**Full Changelog**: [v0.3.4...v0.3.5](https://github.com/schlegelp/octarine/compare/v0.3.4...v0.3.5)

## Version `0.3.4` { data-toc-label="0.3.4" }
_Date: 27/02/26_

#### Improvements
- relaxed the pinned `pygfx` version requirement

#### Fixes
- `Viewer.add`: check for a converter before trying to iterate (e.g. don't treat `gfx.Geometry`
  as an iterable)

**Full Changelog**: [v0.3.3...v0.3.4](https://github.com/schlegelp/octarine/compare/v0.3.3...v0.3.4)

## Version `0.3.3` { data-toc-label="0.3.3" }
_Date: 22/10/25_

#### Breaking
- dropped support for Python 3.9 - `Octarine` now requires Python `>=3.10`

#### Improvements
- new [`Viewer.add_effect`][octarine.Viewer.add_effect] method for post-processing effects such as
  Eye-Dome Lighting

#### Fixes
- fixed setting/getting `Viewer.max_fps`
- fixed an issue in the screenshot function

**Full Changelog**: [v0.3.2...v0.3.3](https://github.com/schlegelp/octarine/compare/v0.3.2...v0.3.3)

## Version `0.3.2` { data-toc-label="0.3.2" }
_Date: 17/10/25_

#### Breaking
- adapted to `pygfx` `0.14`: the `Viewer.blend_mode` property is deprecated in favour of the new
  [`Viewer.set_alpha_mode`][octarine.Viewer.set_alpha_mode] method (alpha modes are also set
  automatically based on object opacity)

#### Fixes
- `Viewer.screenshot`: fixed transparency when `alpha=True`

**Full Changelog**: [v0.3.1...v0.3.2](https://github.com/schlegelp/octarine/compare/v0.3.1...v0.3.2)

## Version `0.3.1` { data-toc-label="0.3.1" }
_Date: 27/05/25_

#### Fixes
- follow changes to `Texture`/`Map` in newer `pygfx` versions
- `Viewer.screenshot` now resolves file paths (e.g. `~/screenshot.png`)
- don't try to start an event loop for offscreen canvases
- handle `ImportError` gracefully when initializing the GUI event loop

**Full Changelog**: [v0.3.0...v0.3.1](https://github.com/schlegelp/octarine/compare/v0.3.0...v0.3.1)

## Version `0.3.0` { data-toc-label="0.3.0" }
_Date: 19/03/25_

#### Breaking
- drop support for Python 3.8 (follows `pygfx`)

#### Improvements
- bumps minimum version of pygfx to `0.9.0`.
- added render trigger options (see [Render Triggers](triggers.md))
- new selection widget (see [Selecting Objects](selections.md))
- new transform widget (see [Moving Objects](manage.md#moving-objects-interactively))
- various improvements in the documentation

**Full Changelog**: [v0.2.5...v0.3.0](https://github.com/schlegelp/octarine/compare/v0.2.5...v0.3.0)

## Version `0.2.5` { data-toc-label="0.2.5" }
_Date: 31/09/24_

#### Fixes
- fixed an segfault issue

**Full Changelog**: [v0.2.4...v0.2.5](https://github.com/schlegelp/octarine/compare/v0.2.4...v0.2.5)

## Version `0.2.4` { data-toc-label="0.2.4" }
_Date: 28/09/24_

#### Fixes
- fix an issue when trimesh is installed without the optional scipy dependency

**Full Changelog**: [v0.2.3...v0.2.4](https://github.com/schlegelp/octarine/compare/v0.2.3...v0.2.4)

## Version `0.2.3` { data-toc-label="0.2.3" }
_Date: 27/09/24_

#### Fixes
- fixes an issue with requirements

**Full Changelog**: [v0.2.2...v0.2.3](https://github.com/schlegelp/octarine/compare/v0.2.2...v0.2.3)

## Version `0.2.2` { data-toc-label="0.2.2" }
_Date: 27/09/24_

#### Improvements
- existing viewers are tracked in `octarine.viewers`
- allow using matplotlib-style line patterns (`-`, `--`, etc.)

**Full Changelog**: [v0.2.1...v0.2.2](https://github.com/schlegelp/octarine/compare/v0.2.1...v0.2.2)

## Version `0.2.1` { data-toc-label="0.2.1" }
_Date: 19/09/24_

#### Fixes
- fixes an issue with `importlib-metadata` dependency

**Full Changelog**: [v0.2.0...v0.2.1](https://github.com/schlegelp/octarine/compare/v0.2.0...v0.2.1)

## Version `0.2.0` { data-toc-label="0.2.0" }
_Date: 19/09/24_

#### Improvements
- added a basic picking system
- color picker now shows alpha channel
- general improvements to volume rendering
- use `Viewer.blend_mode` to set blend mode
- `Viewer.set_view` now also accepts a dictionary with camera state

#### Fixes
- fixes an issue with `Viewer.screenshot`

**Full Changelog**: [v0.1.4...v0.2.0](https://github.com/schlegelp/octarine/compare/v0.1.4...v0.2.0)

## Version <`0.2.0` { data-toc-label="older versions" }

For earlier versions, please see the [commit history](https://github.com/schlegelp/octarine/commits/main/).

