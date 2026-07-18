# Shaders

The `octarine.shaders` module contains the custom materials and
post-processing passes powering some of `Octarine`'s features - see
[Effects & Shading](../effects.md) for an introduction. Requires
`pygfx>=0.16`.

Note that you will rarely have to touch these directly: they are
typically employed via the respective [octarine.Viewer][] methods
(e.g. `add_sparse_volume`, `set_silhouette` or `set_depth_of_field`).

::: octarine.shaders.SparseVolume
    options:
      show_root_heading: true
      filters:
       - "!^_"
       - "^__init__$"

::: octarine.shaders.SparseVolumeMaterial
    options:
      show_root_heading: true
      filters:
       - "!^_"
       - "^__init__$"

::: octarine.shaders.pack_sparse_voxels
    options:
      show_root_heading: true

::: octarine.shaders.SilhouetteMeshMaterial
    options:
      show_root_heading: true
      filters:
       - "!^_"
       - "^__init__$"

::: octarine.shaders.FlexPointsMaterial
    options:
      show_root_heading: true
      filters:
       - "!^_"
       - "^__init__$"

::: octarine.shaders.DepthOfFieldPass
    options:
      show_root_heading: true
      filters:
       - "!^_"
       - "^__init__$"

::: octarine.shaders.NormalizedDepthPass
    options:
      show_root_heading: true
      filters:
       - "!^_"
       - "^__init__$"
