import pytest
import octarine as oc

import pygfx as gfx
import trimesh as tm
import numpy as np

from octarine.merge import merge_visuals, split_visual, sub_visual_at
from conftest import bands as _bands, make_flat_viewer

np.random.seed(0)


def _line(n=4, **kwargs):
    """A small line visual for the merging tests."""
    from octarine.visuals import lines2gfx

    pts = np.zeros((n, 3), dtype=np.float32)
    pts[:, 0] = np.linspace(-10, 10, n)
    kwargs.setdefault("color", "red")
    return lines2gfx(pts, **kwargs)


def _points(n=4, **kwargs):
    """A small points visual for the merging tests."""
    from octarine.visuals import points2gfx

    kwargs.setdefault("color", "red")
    return points2gfx(np.random.rand(n, 3).astype(np.float32), **kwargs)


def _box(y, color):
    """A wide, flat box at a given height - one readable band per box."""
    from octarine.visuals import mesh2gfx

    mesh = tm.creation.box((160, 20, 20))
    mesh.vertices = mesh.vertices + (0, y, 0)
    return mesh2gfx(mesh, color=color, shader="basic")


def _mesh(**kwargs):
    """A small mesh visual for the merging tests."""
    from octarine.visuals import mesh2gfx

    kwargs.setdefault("color", "red")
    return mesh2gfx(tm.creation.box((1, 1, 1)), **kwargs)


# ------------------------------------------------------------------- lines


def test_merge_lines():
    """Lines that only differ in color and width should end up in one object."""
    from octarine.shaders import FlexLineMaterial

    a = _line(4, color="red", linewidth=1)
    b = _line(5, color="blue", linewidth=3)
    a._object_id, b._object_id = "a", "b"

    (merged,) = merge_visuals([a, b])

    assert isinstance(merged, gfx.Line)
    # 4 + 5 vertices plus one NaN vertex separating them
    assert merged.geometry.positions.nitems == 10
    assert np.isnan(merged.geometry.positions.data[4]).all()

    # Both differing properties moved into the geometry
    assert isinstance(merged.material, FlexLineMaterial)
    assert merged.material.thickness_mode == "vertex"
    assert str(merged.material.color_mode).endswith("vertex")

    subs = merged._sub_visuals
    assert [s.object_id for s in subs] == ["a", "b"]
    assert [(s.offset, s.count) for s in subs] == [(0, 4), (5, 5)]

    # ... and the members' own values landed in their slices
    assert list(merged.geometry.thicknesses.data[subs[0].slice]) == [1] * 4
    assert list(merged.geometry.thicknesses.data[subs[1].slice]) == [3] * 5
    assert merged.geometry.colors.data[subs[0].slice] == pytest.approx(
        np.tile([1, 0, 0, 1], (4, 1))
    )
    assert merged.geometry.colors.data[subs[1].slice] == pytest.approx(
        np.tile([0, 0, 1, 1], (5, 1))
    )


def test_merge_keeps_uniforms_uniform():
    """Values that don't differ must not be promoted to a per-vertex buffer."""
    (merged,) = merge_visuals([_line(color="red", linewidth=2) for _ in range(3)])

    assert "colors" not in merged.geometry.keys()
    assert "thicknesses" not in merged.geometry.keys()
    assert merged.material.thickness == 2
    assert tuple(merged.material.color) == (1, 0, 0, 1)

    # Uniform widths need nothing octarine's own material provides, so a scene
    # of stock pygfx lines must stay on stock pygfx after merging
    assert type(merged.material) is gfx.LineMaterial


@pytest.mark.parametrize(
    "kwargs",
    [
        {"linewidth_space": "world"},  # different thickness space
        {"dash_pattern": "dashed"},  # different dash pattern
        {"linewidth": 0},  # thin lines draw with another topology
    ],
)
def test_merge_lines_respects_the_key(kwargs):
    """Lines that would need a different pipeline must not be merged."""
    from octarine.merge import _line_key

    assert len(merge_visuals([_line(), _line(**kwargs)])) == 2

    # ... but two of a kind still merge with each other
    src = _line(**kwargs)
    (merged,) = merge_visuals([src, _line(**kwargs)])

    # Everything the key covers has to survive the merge - otherwise the merged
    # visual would render differently, or refuse to merge with its own kind
    assert _line_key(merged) == _line_key(src)


def test_merge_dashed_lines_keep_their_dash_scale():
    """Dashes are sized in units of the uniform thickness - it must not drift."""
    a = _line(color="red", linewidth=2, dash_pattern="dashed")
    b = _line(color="blue", linewidth=2, dash_pattern="dashed")

    (merged,) = merge_visuals([a, b])
    assert merged.material.thickness == 2

    # Per-vertex widths make the uniform thickness a mean, so dashed lines only
    # merge with others that agree on it
    c = _line(color="red", linewidth=[1, 2, 3, 4], dash_pattern="dashed")
    d = _line(color="blue", linewidth=[4, 3, 2, 1], dash_pattern="dashed")
    (merged,) = merge_visuals([c, d])
    assert merged.material.thickness == c.material.thickness == 2.5
    assert list(merged.geometry.thicknesses.data) == [1, 2, 3, 4, 0, 4, 3, 2, 1]


# ------------------------------------------------------------------ points


def test_merge_points():
    """Points concatenate without separators; size and color move per-vertex."""
    a = _points(4, color="red", size=2)
    b = _points(6, color="blue", size=8)
    a._object_id, b._object_id = "a", "b"

    (merged,) = merge_visuals([a, b])

    assert isinstance(merged, gfx.Points)
    assert merged.geometry.positions.nitems == 10  # no separator
    assert not np.isnan(merged.geometry.positions.data).any()

    assert str(merged.material.size_mode).endswith("vertex")
    assert str(merged.material.color_mode).endswith("vertex")
    assert list(merged.geometry.sizes.data) == [2] * 4 + [8] * 6

    subs = merged._sub_visuals
    assert [(s.object_id, s.offset, s.count) for s in subs] == [("a", 0, 4), ("b", 4, 6)]
    assert merged.geometry.positions.data[subs[1].slice] == pytest.approx(
        b.geometry.positions.data
    )


def test_merge_points_respects_the_key():
    """Marker materials draw differently, so they stay in their own group."""
    plain = _points(color="red", size=2)
    marker = _points(color="red", size=2, marker="square")

    assert type(plain.material) is not type(marker.material)
    assert len(merge_visuals([plain, marker])) == 2
    assert len(merge_visuals([_points(marker="square"), _points(marker="square")])) == 1


# ------------------------------------------------------------------ meshes


def test_merge_meshes():
    """Mesh faces have to be shifted to index into the merged vertex buffer."""
    a, b = _mesh(color="red"), _mesh(color="blue")
    a._object_id, b._object_id = "a", "b"
    n_verts = a.geometry.positions.nitems
    n_faces = a.geometry.indices.nitems

    (merged,) = merge_visuals([a, b])

    assert isinstance(merged, gfx.Mesh)
    assert merged.geometry.positions.nitems == 2 * n_verts
    assert merged.geometry.indices.nitems == 2 * n_faces
    assert str(merged.material.color_mode).endswith("vertex")

    subs = merged._sub_visuals
    assert [(s.offset, s.count) for s in subs] == [(0, n_verts), (n_verts, n_verts)]
    assert [(s.face_offset, s.face_count) for s in subs] == [
        (0, n_faces),
        (n_faces, n_faces),
    ]

    # The second mesh's faces point at the second mesh's vertices
    faces = merged.geometry.indices.data
    assert faces[subs[0].face_slice].max() < n_verts
    assert faces[subs[1].face_slice].min() >= n_verts
    assert np.array_equal(faces[subs[1].face_slice] - n_verts, b.geometry.indices.data)

    # ... and every face still spans the vertices it did before
    assert merged.geometry.positions.data[faces[subs[1].face_slice]] == pytest.approx(
        b.geometry.positions.data[b.geometry.indices.data]
    )


def test_merge_meshes_respects_the_key():
    """Meshes needing a different shader or buffer set must not be merged."""
    from octarine.merge import _mesh_key

    assert len(merge_visuals([_mesh(), _mesh(shader="basic")])) == 2
    assert len(merge_visuals([_mesh(), _mesh(silhouette=2)])) == 2

    # Normals are only carried over if everyone has them
    with_normals = _mesh()
    with_normals.geometry.normals = gfx.Buffer(
        np.zeros((with_normals.geometry.positions.nitems, 3), np.float32)
    )
    assert len(merge_visuals([_mesh(), with_normals])) == 2

    src = _mesh(shader="basic")
    (merged,) = merge_visuals([src, _mesh(shader="basic", color="blue")])
    assert _mesh_key(merged) == _mesh_key(src)


def test_merge_meshes_with_auto_vertex_colors():
    """Mesh shaders read `geometry.colors` under "auto" - don't drop them.

    `trimesh2gfx` leaves `color_mode` at "auto" and hands the colours over as
    a buffer, which the mesh shader multiplies into the uniform colour. Two
    live colour sources are more than one absorbed buffer can express, so
    those meshes must be refused rather than merged down to the uniform.

    """
    from octarine.visuals import trimesh2gfx

    def colored(rgba):
        mesh = tm.creation.box((1, 1, 1))
        mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1))
        return trimesh2gfx(mesh)

    a, b = colored([255, 0, 0, 255]), colored([0, 0, 255, 255])
    assert str(a.material.color_mode) == "auto" and "colors" in a.geometry.keys()
    assert len(merge_visuals([a, b])) == 2

    # Lines and points resolve "auto" to the uniform colour and ignore any
    # colors buffer, so for them there is only ever one source to absorb
    assert len(merge_visuals([_line(), _line(color="blue")])) == 1


def test_merge_meshes_keeps_material_settings():
    """A merged material must carry over the source's own knobs."""
    a = _mesh(color="red", silhouette=3)
    b = _mesh(color="blue", silhouette=3)

    (merged,) = merge_visuals([a, b])
    assert type(merged.material) is type(a.material)
    assert merged.material.silhouette == 3
    assert merged.material.alpha_mode == a.material.alpha_mode

    # Meshes differing in a material uniform stay apart
    assert len(merge_visuals([_mesh(silhouette=3), _mesh(silhouette=5)])) == 2


# ------------------------------------------------------------------ shared


def test_merge_passes_through_and_keeps_order():
    """Anything we can't merge stays put, in place and untouched."""
    from octarine.visuals import volume2gfx

    a, b = _line(color="red"), _line(color="blue")
    vol = volume2gfx(np.random.rand(4, 4, 4).astype(np.float32), color="magma")
    box = gfx.BoxHelper()  # a Line subclass - must not be swallowed

    grouped = merge_visuals([a, vol, b, box])

    assert len(grouped) == 3
    assert grouped[0]._sub_visuals  # a and b merged, at a's position
    assert grouped[1] is vol
    assert grouped[2] is box

    # A lone visual is passed through as-is rather than rebuilt
    assert merge_visuals([a]) == [a]

    # Different visual types never mix
    merged = merge_visuals([_line(), _line(), _points(), _points(), _mesh(), _mesh()])
    assert [type(v).__name__ for v in merged] == ["Line", "Points", "Mesh"]


def test_merge_respects_object_state():
    """Object-level state the merged visual can hold only once must match."""
    a, b = _line(), _line()
    b.visible = False
    assert len(merge_visuals([a, b])) == 2

    a, b = _line(), _line()
    b.local.position = (5, 0, 0)
    assert len(merge_visuals([a, b])) == 2

    a, b = _line(), _line()
    b.render_order = 3
    assert len(merge_visuals([a, b])) == 2

    # Members that agree keep that state on the merged visual
    a, b = _line(), _line()
    a.render_order = b.render_order = 3
    a.local.position = b.local.position = (5, 0, 0)
    (merged,) = merge_visuals([a, b])
    assert merged.render_order == 3
    assert tuple(merged.local.position) == (5, 0, 0)


@pytest.mark.parametrize("make", [_line, _points, _mesh])
def test_merge_batching_and_regrouping(make):
    """Groups can be size-capped, and grown one visual at a time."""
    visuals = [make(color=c) for c in ("red", "blue", "green", "yellow")]

    batched = merge_visuals(visuals, max_size=2)
    assert len(batched) == 2
    assert [len(v._sub_visuals) for v in batched] == [2, 2]

    n = visuals[0].geometry.positions.nitems
    assert len(merge_visuals(visuals, max_nodes=2 * n)) == 2

    # Merged visuals are taken apart again when re-merged, so adding to an
    # existing group doesn't nest or duplicate anything
    (regrouped,) = merge_visuals(batched + [make(color="white")])
    assert len(regrouped._sub_visuals) == 5
    assert sum(s.count for s in regrouped._sub_visuals) == 5 * n


def test_merge_object_ids():
    """The merged visual reports one id only where its members agree."""
    a, b = _line(), _line()
    a._object_id = b._object_id = "same"
    a._object_group = b._object_group = "grp"
    (merged,) = merge_visuals([a, b])
    assert merged._object_id == "same"
    assert merged._object_group == "grp"

    a, b = _line(), _line()
    a._object_id, b._object_id = "a", "b"
    a._object_group, b._object_group = "one", "two"
    (merged,) = merge_visuals([a, b])
    assert merged._object_id not in ("a", "b")
    assert merged._object_group is None
    assert [s.object_id for s in merged._sub_visuals] == ["a", "b"]
    assert [s.group for s in merged._sub_visuals] == ["one", "two"]


# ------------------------------------------------------------------ render


def _flat_viewer():
    """An offscreen viewer whose output can be read row by row."""
    return make_flat_viewer(oc.Viewer(offscreen=True, size=(200, 200)))


def _hide(merged, index):
    """Hide one member by NaN-ing the vertex slice it owns."""
    sub = merged._sub_visuals[index]
    merged.geometry.positions.data[sub.slice] = np.nan
    merged.geometry.positions.update_range(sub.offset, sub.count)


def test_merge_lines_render():
    """A merged line must draw its members with their own color and width."""
    from octarine.visuals import lines2gfx

    def horizontal(y, color, width):
        pts = np.zeros((10, 3), dtype=np.float32)
        pts[:, 0] = np.linspace(-80, 80, 10)
        pts[:, 1] = y
        return lines2gfx(pts, color=color, linewidth=width, linewidth_space="world")

    v = _flat_viewer()
    (merged,) = merge_visuals([horizontal(40, "red", 4), horizontal(-40, "blue", 10)])
    v.scene.add(merged)

    # Two bands, the lower one thicker, each in its own color
    assert _bands(v) == [(4, (255, 0, 0)), (10, (0, 0, 255))]

    # NaN-ing a member's position slice hides just that member
    _hide(merged, 0)
    assert _bands(v) == [(10, (0, 0, 255))]

    v.close()


def test_merge_meshes_render():
    """A merged mesh must draw its members in their own place and color."""
    v = _flat_viewer()
    (merged,) = merge_visuals([_box(40, "red"), _box(-40, "blue")])
    v.scene.add(merged)

    assert _bands(v) == [(20, (255, 0, 0)), (20, (0, 0, 255))]

    _hide(merged, 0)
    assert _bands(v) == [(20, (0, 0, 255))]

    v.close()


def test_merge_points_render():
    """A merged points visual must draw its members at their own size."""
    from octarine.visuals import points2gfx

    def dot(y, color, size):
        return points2gfx(
            np.array([[0, y, 0]], dtype=np.float32),
            color=color,
            size=size,
            size_space="world",
        )

    v = _flat_viewer()
    (merged,) = merge_visuals([dot(40, "red", 10), dot(-40, "blue", 30)])
    v.scene.add(merged)

    bands = _bands(v)
    assert [c for _, c in bands] == [(255, 0, 0), (0, 0, 255)]
    assert bands[0][0] == pytest.approx(10, abs=2)
    assert bands[1][0] == pytest.approx(30, abs=2)

    _hide(merged, 0)
    assert [c for _, c in _bands(v)] == [(0, 0, 255)]

    v.close()


# ------------------------------------------------------------------- split


def _ids(visual):
    """The object ids a visual holds, merged or not."""
    subs = getattr(visual, "_sub_visuals", None)
    if subs is None:
        return [visual._object_id]
    return [sub.object_id for sub in subs]


@pytest.mark.parametrize("make", [_line, _points, _mesh])
def test_split_roundtrip(make):
    """Splitting every member out must give back what was merged."""
    originals = []
    for i, color in enumerate(("red", "blue", "green")):
        vis = make(color=color)
        vis._object_id = f"obj{i}"
        originals.append(vis)

    (merged,) = merge_visuals(list(originals))
    remainder, extracted = split_visual(merged)

    assert remainder is None
    assert [_ids(v) for v in extracted] == [["obj0"], ["obj1"], ["obj2"]]

    for original, split in zip(originals, extracted):
        # Same geometry, in the same place ...
        np.testing.assert_allclose(
            split.geometry.positions.data, original.geometry.positions.data
        )
        # ... and the same color, back on the material where it started
        assert split.material.color_mode == "uniform"
        np.testing.assert_allclose(
            np.asarray(split.material.color.rgba),
            np.asarray(gfx.Color(original.material.color).rgba),
        )
        # A split-out visual must be indistinguishable from an unmerged one
        assert not hasattr(split, "_sub_visuals")


@pytest.mark.parametrize("make", [_line, _points, _mesh])
def test_split_middle_member(make):
    """Splitting one member must leave the others merged and intact."""
    visuals = []
    for i, color in enumerate(("red", "blue", "green")):
        vis = make(color=color)
        vis._object_id = f"obj{i}"
        visuals.append(vis)

    (merged,) = merge_visuals(list(visuals))
    remainder, extracted = split_visual(merged, "obj1")

    assert _ids(remainder) == ["obj0", "obj2"]
    assert [_ids(v) for v in extracted] == [["obj1"]]

    # The remainder still disagrees on color, so it keeps a vertex buffer ...
    assert remainder.material.color_mode == "vertex"
    np.testing.assert_allclose(
        remainder.geometry.positions.data[remainder._sub_visuals[1].slice],
        visuals[2].geometry.positions.data,
    )
    # ... and the extracted one is back to a uniform blue
    assert extracted[0].material.color_mode == "uniform"
    np.testing.assert_allclose(
        np.asarray(extracted[0].material.color.rgba), (0.0, 0.0, 1.0, 1.0)
    )


def test_split_mesh_faces_are_rebased():
    """A split-out mesh's faces must index into its own vertices."""
    a, b = _mesh(color="red"), _mesh(color="blue")
    a._object_id, b._object_id = "a", "b"

    (merged,) = merge_visuals([a, b])
    _, extracted = split_visual(merged, ["a", "b"])

    for split, original in zip(extracted, (a, b)):
        indices = split.geometry.indices.data
        assert indices.max() < len(split.geometry.positions.data)
        np.testing.assert_array_equal(indices, original.geometry.indices.data)


def test_split_keeps_shared_state():
    """A split must carry the merged visual's shared material state over."""
    a, b = _mesh(color="red"), _mesh(color="blue")
    a._object_id, b._object_id = "a", "b"
    for vis in (a, b):
        vis.material.flat_shading = True
        vis.material.opacity = 0.5

    (merged,) = merge_visuals([a, b])
    remainder, (extracted,) = split_visual(merged, "a")

    for vis in (remainder, extracted):
        assert vis.material.flat_shading is True
        assert vis.material.opacity == 0.5
        assert vis._object_type == "mesh"


def test_split_groups_members_by_object_id():
    """Members sharing an object id must split out together."""
    a, b, c = _mesh(color="red"), _mesh(color="blue"), _mesh(color="green")
    a._object_id = b._object_id = "pair"
    c._object_id = "lonely"

    (merged,) = merge_visuals([a, b, c])
    remainder, extracted = split_visual(merged, "pair")

    assert _ids(remainder) == ["lonely"]
    # Both members stay in one visual, which therefore keeps its sub-visuals
    assert len(extracted) == 1
    assert _ids(extracted[0]) == ["pair", "pair"]
    assert extracted[0].material.color_mode == "vertex"


def test_split_unknown_ids_and_errors():
    """Unknown ids are ignored; an unmerged visual cannot be split."""
    a, b = _mesh(color="red"), _mesh(color="blue")
    a._object_id, b._object_id = "a", "b"
    (merged,) = merge_visuals([a, b])

    remainder, extracted = split_visual(merged, "nope")
    assert extracted == []
    assert _ids(remainder) == ["a", "b"]

    with pytest.raises(ValueError, match="not a merged visual"):
        split_visual(_mesh())


def test_split_restores_line_widths():
    """Widths absorbed into the geometry must come back onto the material."""
    a = _line(color="red", linewidth=2, linewidth_space="world")
    b = _line(color="red", linewidth=7, linewidth_space="world")
    a._object_id, b._object_id = "a", "b"

    (merged,) = merge_visuals([a, b])
    assert merged.material.thickness_mode == "vertex"

    _, extracted = split_visual(merged, ["a", "b"])
    assert [v.material.thickness for v in extracted] == [2, 7]
    # With the width back on the material there is nothing left that needs
    # octarine's own material, so a split-out line returns to stock pygfx -
    # which is also what `lines2gfx` built it as in the first place.
    assert all(type(v.material) is gfx.LineMaterial for v in extracted)
    assert all("thicknesses" not in v.geometry.keys() for v in extracted)


# ------------------------------------------------------------------ picking


def test_sub_visual_at_mesh():
    """A mesh pick must resolve to the member owning that face."""
    a, b, c = _mesh(color="red"), _mesh(color="blue"), _mesh(color="green")
    a._object_id, b._object_id, c._object_id = "a", "b", "c"
    (merged,) = merge_visuals([a, b, c])

    for sub in merged._sub_visuals:
        # First, last and a middle face of every member
        for face in (sub.face_offset, sub.face_offset + sub.face_count - 1,
                     sub.face_offset + sub.face_count // 2):
            hit = sub_visual_at(merged, {"face_index": face})
            assert hit.object_id == sub.object_id

    # Past the end of the last member
    total = sum(s.face_count for s in merged._sub_visuals)
    assert sub_visual_at(merged, {"face_index": total}) is None


@pytest.mark.parametrize("make", [_points, _line])
def test_sub_visual_at_vertex(make):
    """Points and lines must resolve a pick by vertex index."""
    a, b = make(color="red"), make(color="blue")
    a._object_id, b._object_id = "a", "b"
    (merged,) = merge_visuals([a, b])

    for sub in merged._sub_visuals:
        for vertex in (sub.offset, sub.offset + sub.count - 1):
            hit = sub_visual_at(merged, {"vertex_index": vertex})
            assert hit.object_id == sub.object_id


def test_sub_visual_at_line_gap():
    """The NaN vertex between two merged lines belongs to no member."""
    a, b = _line(color="red"), _line(color="blue")
    a._object_id, b._object_id = "a", "b"
    (merged,) = merge_visuals([a, b])

    first = merged._sub_visuals[0]
    gap = first.offset + first.count
    assert np.all(np.isnan(merged.geometry.positions.data[gap]))
    assert sub_visual_at(merged, {"vertex_index": gap}) is None


def test_sub_visual_at_non_merged():
    """Picking something that was never merged must simply report nothing."""
    assert sub_visual_at(_mesh(), {"face_index": 0}) is None
    assert sub_visual_at(_points(), {"vertex_index": 0}) is None


def test_sub_visual_at_without_index():
    """A pick carrying no usable index must not raise."""
    a, b = _mesh(color="red"), _mesh(color="blue")
    a._object_id, b._object_id = "a", "b"
    (merged,) = merge_visuals([a, b])

    assert sub_visual_at(merged, {}) is None


def test_split_meshes_render():
    """Splitting a member out must not change what the scene looks like."""
    a, b = _box(40, "red"), _box(-40, "blue")
    a._object_id, b._object_id = "a", "b"

    v = _flat_viewer()
    (merged,) = merge_visuals([a, b])
    v.scene.add(merged)
    before = _bands(v)
    assert before == [(20, (255, 0, 0)), (20, (0, 0, 255))]

    # Split "a" back out and put both halves on the scene in its place
    remainder, (extracted,) = split_visual(merged, "a")
    v.scene.remove(merged)
    v.scene.add(remainder, extracted)

    assert _bands(v) == before

    # And the split-out mesh is addressable the plain way again
    extracted.material.color = "lime"
    assert _bands(v) == [(20, (0, 255, 0)), (20, (0, 0, 255))]

    v.close()
