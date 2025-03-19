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

## Version `0.3.0` { data-toc-label="0.2.5" }
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

