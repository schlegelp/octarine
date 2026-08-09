# Shaders

The `octarine.shaders` module contains the custom materials and
post-processing passes powering some of `Octarine`'s features - see
[Effects & Shading](../effects.md) for an introduction. Requires
`pygfx>=0.17`.

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

::: octarine.shaders.BitmaskVolume
    options:
      show_root_heading: true
      filters:
       - "!^_"
       - "^__init__$"

::: octarine.shaders.BitmaskVolumeMaterial
    options:
      show_root_heading: true
      filters:
       - "!^_"
       - "^__init__$"

::: octarine.shaders.pack_voxel_runs
    options:
      show_root_heading: true

::: octarine.shaders.runs_from_voxels
    options:
      show_root_heading: true

::: octarine.shaders.TubeVisual
    options:
      show_root_heading: true
      filters:
       - "!^_"
       - "^__init__$"

::: octarine.shaders.TubeMaterial
    options:
      show_root_heading: true
      filters:
       - "!^_"
       - "^__init__$"

::: octarine.shaders.decimate_edges
    options:
      show_root_heading: true

::: octarine.shaders.align_frames
    options:
      show_root_heading: true

::: octarine.shaders.rotate_profile
    options:
      show_root_heading: true

::: octarine.shaders.split_runs
    options:
      show_root_heading: true

::: octarine.shaders.SilhouetteMeshMaterial
    options:
      show_root_heading: true
      filters:
       - "!^_"
       - "^__init__$"

::: octarine.shaders.SubsurfaceMeshMaterial
    options:
      show_root_heading: true
      filters:
       - "!^_"
       - "^__init__$"

::: octarine.shaders.MatcapMeshMaterial
    options:
      show_root_heading: true
      filters:
       - "!^_"
       - "^__init__$"

::: octarine.shaders.make_matcap
    options:
      show_root_heading: true

::: octarine.shaders.matcap_texture
    options:
      show_root_heading: true

::: octarine.shaders.GradientBackgroundMaterial
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

::: octarine.shaders.AmbientOcclusionPass
    options:
      show_root_heading: true
      filters:
       - "!^_"
       - "^__init__$"

::: octarine.shaders.OutlinePass
    options:
      show_root_heading: true
      filters:
       - "!^_"
       - "^__init__$"

::: octarine.shaders.ToneMappingPass
    options:
      show_root_heading: true
      filters:
       - "!^_"
       - "^__init__$"

::: octarine.shaders.procedural_env_map
    options:
      show_root_heading: true

::: octarine.shaders.environment_radiance
    options:
      show_root_heading: true

::: octarine.shaders.cube_directions
    options:
      show_root_heading: true
