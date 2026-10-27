# Merging

Merging many small objects into a few shared pygfx objects, which is much
faster to render. See [Merging objects](../manage.md#merging-objects) for how
to use it; this is the machinery underneath.

The entry points on the viewer are `Viewer.merge_objects`,
`Viewer.unmerge_objects` and `Viewer.merge_stats`, plus the `merge` parameter
of `Viewer` and of the `add` methods.

::: octarine.handles.MergeStats
    options:
      show_source: true
      show_root_heading: true

::: octarine.handles.SubVisualHandle
    options:
      show_source: true
      show_root_heading: true
      filters:
       - "!^_"
       - "^__init__$"

::: octarine.merge.merge_visuals
    options:
      show_source: true
      show_root_heading: true

::: octarine.merge.split_visual
    options:
      show_source: true
      show_root_heading: true

::: octarine.merge.sub_visual_at
    options:
      show_source: true
      show_root_heading: true

::: octarine.merge.SubVisual
    options:
      show_source: true
      show_root_heading: true
