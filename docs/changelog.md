# :simple-keepachangelog: Changelog

This is a selection of features added, changes made and bugs fixed with each version.
For a full list of changes please see the [commits history](https://github.com/schlegelp/octarine/commits/main)
on the Github repository.

## dev
_Date: ongoing_

To install the current `dev` version of `octarine`:

```shell
pip uninstall octarine3d -y
pip install git+https://github.com/schlegelp/octarine.git
```

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

