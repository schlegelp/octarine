"""Merging visuals into fewer pygfx objects.

Every object in a scene costs the renderer a fixed slice of CPU time per
frame: it is walked, its transform and uniforms are refreshed, it is
depth-sorted and it is submitted as a draw call of its own. None of that
scales with how much geometry the object holds, so a few hundred small
objects cost far more than one large one. Five hundred neurons added as five
hundred visuals spend ~5 ms per frame before the GPU has drawn anything;
merged, the same geometry costs well under 1 ms.

`merge_visuals` does that merging, for lines, points and meshes. Two visuals
can share one pygfx object when they agree on everything that object can hold
only once - the material's uniforms, its blend and depth state, the object's
transform and render order. The properties that usually *do* differ - colour,
line width, point size - are moved into the geometry instead, where pygfx can
carry one value per vertex. Which of those a material supports is what
decides how much can be merged in practice: octarine's `FlexLineMaterial`
exists precisely so that lines of differing width still qualify.

Those absorbable properties are declared per visual type - see `_LINE`,
`_POINTS` and `_MESH` - as the triple of (material value, material mode,
geometry buffer) that pygfx uses to express "one value for the object" versus
"one value per vertex". Everything else about merging is generic over that
table.

What merging costs is the per-object handle. `material.color`,
`visual.visible` and one `_object_id` per object are gone; each merged-in
visual becomes a range of vertices recorded in `visual._sub_visuals`, and
per-object operations become writes to that range. See `SubVisual`.

"""

import uuid

import pygfx as gfx
import numpy as np

from collections import namedtuple
from dataclasses import dataclass
from functools import lru_cache

from . import utils

__all__ = ["merge_visuals", "SubVisual"]


# Ceiling on the size of a single merged visual, deliberately conservative:
# the widest buffer is 16 bytes per vertex for colours, and a merged mesh
# carries its indices on top of that, so this stays well under the 128 MiB
# storage binding size GPUs commonly report.
_MAX_MERGED_NODES = 4_000_000

# Material properties the generic copy below leaves alone. `alpha_*`,
# `depth_write` and `render_queue` because pygfx derives them from one another
# and their setters have side effects, so `_copy_material_state` handles them
# by hand; `vertex_colors` because it is a deprecated alias of `color_mode`,
# which every visual type absorbs into the geometry - leaving it in would put
# the absorbed property back into the key through the back door, and a merged
# visual would then refuse to merge with the plain ones it was built from.
_SKIP_MATERIAL_PROPS = frozenset(
    {
        "alpha_mode",
        "alpha_config",
        "alpha_method",
        "depth_write",
        "render_queue",
        "vertex_colors",
    }
)

# Object-level state that a merged visual can hold only once, and that is
# therefore both part of the merge key and copied onto the merged visual.
# (The transform and `_object_type` need special handling and are separate.)
_SHARED_OBJECT_PROPS = ("visible", "render_order", "cast_shadow", "receive_shadow")

# A property that pygfx can express either as one value for the whole object
# or as one per vertex, and that merging therefore absorbs into the geometry
# when members disagree: the material property holding the single value, the
# material property switching between the two, the geometry buffer, and how
# many components each value has.
_Absorbable = namedtuple("_Absorbable", ["value", "mode", "buffer", "channels"])

_COLOR = _Absorbable("color", "color_mode", "colors", 4)
_THICKNESS = _Absorbable("thickness", "thickness_mode", "thicknesses", 1)
_SIZE = _Absorbable("size", "size_mode", "sizes", 1)

_Member = namedtuple("_Member", ["object_id", "group", "positions", "attrs"])
_Merger = namedtuple("_Merger", ["key", "members", "merge"])


@dataclass
class SubVisual:
    """One of the original visuals inside a visual merged by `merge_visuals`.

    Merging turns N objects into one, which means per-object operations
    (hiding, recolouring, picking) can no longer address an object as a whole.
    Instead each original visual becomes a contiguous range of vertices in the
    merged buffers, and this is the record of where it went::

        sub = merged._sub_visuals[3]
        merged.geometry.colors.data[sub.slice] = (1, 0, 0, 1)
        merged.geometry.colors.update_range(sub.offset, sub.count)

    Hiding works the same way, by writing NaN into the position slice - those
    vertices then drop out for lines, points and meshes alike, exactly like
    the NaN breaks that separate merged lines. Note that this is destructive,
    so callers that want to unhide again have to keep a copy of the slice.

    Merged meshes additionally own a range of *faces*, which is what picking
    reports and what the merged `indices` buffer is laid out by.

    Attributes
    ----------
    object_id : str | uuid
                The `_object_id` the original visual carried.
    group :     str | None
                The `_object_group` the original visual carried.
    offset :    int
                Index of the member's first vertex in the merged buffers.
    count :     int
                Number of vertices the member occupies. For lines this
                excludes the NaN vertex that separates it from the next
                member.
    face_offset : int | None
                Index of the member's first face. Meshes only.
    face_count : int | None
                Number of faces the member occupies. Meshes only.

    """

    object_id: object
    group: object
    offset: int
    count: int
    face_offset: int = None
    face_count: int = None

    @property
    def slice(self):
        """The member's vertex range, as a slice into the merged buffers."""
        return slice(self.offset, self.offset + self.count)

    @property
    def face_slice(self):
        """The member's face range, or None if this isn't a mesh."""
        if self.face_offset is None:
            return None
        return slice(self.face_offset, self.face_offset + self.face_count)


# ---------------------------------------------------------------- material


def _hashable(value):
    """Turn a property value into something we can put in a merge key."""
    if isinstance(value, np.ndarray):
        return value.tobytes()
    if utils.is_hashable(value):
        return value

    # Colours, clipping planes and other small sequences are unhashable but
    # compare perfectly well by value - and `gfx.Color` is all over the mesh
    # and marker materials, so falling back to identity here would mean they'd
    # essentially never merge.
    try:
        return tuple(_hashable(v) for v in value)
    except TypeError:
        # Textures, images and the like. Identity is the honest comparison for
        # those, and it errs towards *not* merging.
        return id(value)


@lru_cache(maxsize=None)
def _candidate_material_props(cls, absorbed):
    """Settable properties of `cls` that may have to match for a merge.

    Introspected rather than listed out, so that material subclasses - pygfx's,
    octarine's own and any a user brings - are covered without this module
    having to know about them. Some of these raise on access (see
    `_material_props`), which is why this is only the candidate list.

    """
    return tuple(
        sorted(
            name
            for name in dir(cls)
            if not name.startswith("_")
            and name not in absorbed
            and name not in _SKIP_MATERIAL_PROPS
            and isinstance(getattr(cls, name, None), property)
            and getattr(cls, name).fset is not None
        )
    )


def _material_props(material, absorbed):
    """The (name, value) pairs that describe `material` for merging purposes.

    Both the merge key and the copy onto the merged material go through here,
    so the two cannot disagree about which properties matter. Properties that
    raise are skipped: pygfx leaves optional maps (`emissive_map`,
    `normal_map`) off the material's store until they are set, so reading one
    is an `AttributeError` rather than `None`. Skipping is sound because a
    material that *has* one contributes an extra pair and so keys differently
    from one that doesn't.

    """
    for name in _candidate_material_props(type(material), absorbed):
        try:
            yield name, getattr(material, name)
        except AttributeError:
            continue


def _material_state(material, absorbed):
    """Hashable digest of everything about `material` that a merge preserves."""
    state = [(name, _hashable(value)) for name, value in _material_props(material, absorbed)]
    state.append(("alpha_mode", material.alpha_mode))
    state.append(("alpha_config", repr(material.alpha_config)))
    state.append(("depth_write", material.depth_write))
    state.append(("render_queue", material.render_queue))
    return tuple(state)


def _copy_material_state(source, target, absorbed):
    """Copy everything `_material_state` covers from one material onto another."""
    for name, value in _material_props(source, absorbed):
        setattr(target, name, value)

    # Alpha goes first: setting it resets the derived depth and queue settings.
    # Those two are only pinned on the merged material if they were pinned on
    # the source - otherwise we let them derive, as they did before.
    if source.alpha_mode == "custom":
        target.alpha_config = source.alpha_config
    else:
        target.alpha_mode = source.alpha_mode
    if source.depth_write_is_set:
        target.depth_write = source.depth_write
    if source.render_queue_is_set:
        target.render_queue = source.render_queue


# --------------------------------------------------------- absorbed values


def _mode(material, name):
    """Read an absorbed property's "uniform" / "vertex" mode."""
    # pygfx returns a plain str today but templates on `str(...).split(".")`
    # itself, so we stay defensive in the same way.
    return str(getattr(material, name, "uniform")).split(".")[-1]


def _buffer(geometry, name):
    """Read a geometry buffer's data, or None if it isn't there."""
    buffer = getattr(geometry, name, None)
    return None if buffer is None else buffer.data


def _as_rgba(colors):
    """Normalise a per-vertex color array to (N, 4) float32 RGBA.

    Mirrors the way the shaders expand 1- to 4-channel color buffers, so that
    members with differently shaped colors can share one buffer.

    """
    colors = np.asarray(colors, dtype=np.float32)
    if colors.ndim == 1:
        colors = colors[:, None]

    n, channels = colors.shape
    if channels == 4:
        return colors

    rgba = np.ones((n, 4), dtype=np.float32)
    if channels == 3:
        rgba[:, :3] = colors
    elif channels == 2:  # grayscale + alpha
        rgba[:, :3] = colors[:, :1]
        rgba[:, 3] = colors[:, 1]
    else:  # grayscale
        rgba[:, :3] = colors[:, :1]
    return rgba


def _absorbable_values(vis, attr):
    """This visual's values for one absorbable property.

    Always shaped (n_vertices, channels) when the visual carries them per
    vertex and (channels,) when it carries a single value, so that the two
    cases can be told apart by `ndim` and broadcast alike.

    """
    if _mode(vis.material, attr.mode) == "vertex":
        values = _buffer(vis.geometry, attr.buffer)
        if attr.channels == 4:
            return _as_rgba(values)
        return np.asarray(values, dtype=np.float32).reshape(-1, 1)

    value = getattr(vis.material, attr.value)
    if attr.channels == 4:
        return np.asarray(gfx.Color(value).rgba, dtype=np.float32)
    return np.asarray([value], dtype=np.float32)


def _differs(members, attr):
    """Whether an absorbable property varies across members.

    A member that already carries per-vertex values forces the per-vertex
    buffer even if all of its values happen to be equal - we don't inspect
    them.

    """
    first = members[0].attrs[attr.value]
    for member in members:
        values = member.attrs[attr.value]
        if values.ndim == 2 or values.tobytes() != first.tobytes():
            return True
    return False


def _absorb(members, subs, n_vertices, attrs):
    """Route each absorbable property into the geometry or the material.

    Returns the geometry and material kwargs for the merged visual: a buffer
    plus `mode="vertex"` where the members disagree, the shared single value
    plus `mode="uniform"` where they don't.

    """
    geometry_kwargs, material_kwargs = {}, {}

    for attr in attrs:
        if _differs(members, attr):
            # The gap vertices between members keep their zero. They are never
            # drawn (their position is NaN), and unlike NaN a zero cannot leak
            # into a shader's interpolation.
            values = np.zeros((n_vertices, attr.channels), dtype=np.float32)
            for member, sub in zip(members, subs):
                values[sub.slice] = member.attrs[attr.value]
            geometry_kwargs[attr.buffer] = (
                values if attr.channels > 1 else values.reshape(-1)
            )
            material_kwargs[attr.mode] = "vertex"
        else:
            value = members[0].attrs[attr.value]
            material_kwargs[attr.value] = (
                gfx.Color(tuple(value)) if attr.channels == 4 else float(value[0])
            )
            material_kwargs[attr.mode] = "uniform"

    return geometry_kwargs, material_kwargs


# ------------------------------------------------------------------ shared


def _object_state(vis):
    """Hashable digest of the object-level state a merged visual holds once."""
    return (
        getattr(vis, "_object_type", None),
        tuple(_hashable(getattr(vis, name)) for name in _SHARED_OBJECT_PROPS),
        vis.local.matrix.tobytes(),
    )


def _mergeable(vis, kind):
    """Whether this visual's geometry and colour setup is one we can merge."""
    geometry, material = vis.geometry, vis.material

    positions = _buffer(geometry, "positions")
    if not utils.is_points(positions) or not len(positions):
        return False
    # Anything else on the geometry - a texcoords buffer on a line, say -
    # would be silently dropped, so leave such visuals alone instead.
    if any(not k.startswith("_") and k not in kind.buffers for k in geometry.keys()):
        return False

    # A colormap or texture brings state we can't merge
    if material.map is not None:
        return False

    mode = _mode(material, "color_mode")
    if mode not in ("auto", "uniform", "vertex"):
        return False  # face colors index by face, not by vertex
    if mode == "auto" and _buffer(geometry, "colors") is not None and kind.auto_reads_colors:
        # Mesh shaders multiply the uniform colour by the vertex colours when
        # the mode is "auto" (`meshshader.py`, `mesh.wgsl`), where line and
        # point shaders fall back to the uniform colour alone. Two live colour
        # sources is more than one absorbed buffer can express, so we refuse
        # rather than quietly drop one of them.
        return False

    # Whatever else is absorbed has to actually be there if it says it is
    for attr in kind.attrs:
        if _mode(material, attr.mode) == "vertex":
            if _buffer(geometry, attr.buffer) is None:
                return False
    return True


def _sub_visuals_of(vis, count, face_count=None):
    """The members `vis` is made up of.

    A visual that has already been merged carries several; anything else is a
    single member spanning all of its vertices.

    """
    subs = getattr(vis, "_sub_visuals", None)
    if subs:
        return subs
    return [
        SubVisual(
            getattr(vis, "_object_id", None),
            getattr(vis, "_object_group", None),
            0,
            count,
            None if face_count is None else 0,
            face_count,
        )
    ]


def _members(vis, kind, face_count=None, extra=None):
    """Split `vis` into its members, slicing every absorbed property along.

    `extra(sub)` supplies whatever else a visual type carries per member (the
    mesh path uses it for faces).

    """
    positions = _buffer(vis.geometry, "positions")
    values = {attr.value: _absorbable_values(vis, attr) for attr in kind.attrs}

    for sub in _sub_visuals_of(vis, len(positions), face_count):
        attrs = {
            name: (value[sub.slice] if value.ndim == 2 else value)
            for name, value in values.items()
        }
        if extra is not None:
            attrs.update(extra(sub))
        yield _Member(sub.object_id, sub.group, positions[sub.slice], attrs)


def _layout(members, gap):
    """Concatenate the members' positions and record where each one landed.

    `gap` is the number of filler vertices to leave between neighbouring
    members: one for lines, whose NaN position keeps them from being joined up
    into one long line, and none for points and meshes, whose primitives are
    independent to begin with.

    """
    counts = [len(m.positions) for m in members]
    positions = np.empty(
        (sum(counts) + gap * (len(members) - 1), 3), dtype=np.float32
    )

    subs, at = [], 0
    for member, count in zip(members, counts):
        if gap and at:
            positions[at - gap : at] = np.nan
        positions[at : at + count] = member.positions
        subs.append(SubVisual(member.object_id, member.group, at, count))
        at += count + gap

    return subs, positions


def _finalize(visual, template, subs):
    """Give the merged visual the octarine and pygfx state of its members."""
    # The merged visual can carry only one id and one group. Where the members
    # disagree we fall back to a fresh id - `_sub_visuals` is what identifies
    # the individual objects from here on.
    ids = {s.object_id for s in subs}
    groups = {s.group for s in subs}
    visual._object_id = subs[0].object_id if len(ids) == 1 else uuid.uuid4()
    visual._object_group = subs[0].group if len(groups) == 1 else None
    visual._sub_visuals = subs
    if hasattr(template, "_object_type"):
        visual._object_type = template._object_type

    for name in _SHARED_OBJECT_PROPS:
        setattr(visual, name, getattr(template, name))
    matrix = template.local.matrix
    if not np.array_equal(matrix, np.eye(4)):
        visual.local.matrix = matrix

    return visual


# ------------------------------------------------------------------- lines


def _line_key(vis):
    """Merge key for a line visual, or None if it can't be merged."""
    from .shaders import FlexLineMaterial

    material = vis.material
    # Deliberately not `LineSegmentMaterial` & co: those pair up vertices, so
    # the NaN vertex we insert between members would shift the pairing.
    if type(material) not in (gfx.LineMaterial, FlexLineMaterial, gfx.LineThinMaterial):
        return None
    if not _mergeable(vis, _LINE):
        return None

    return (
        # Thin lines draw with another topology, so they only merge with each
        # other. Thick ones merge regardless of their material class, since
        # `FlexLineMaterial` can express what `LineMaterial` does.
        "thin" if isinstance(material, gfx.LineThinMaterial) else "thick",
        # The dash pattern is expressed in units of the (uniform) thickness,
        # so with dashes that thickness has to match too. Without them it is
        # absorbed into the geometry like any other differing width.
        float(material.thickness) if material.dash_pattern else None,
        _object_state(vis),
        _material_state(material, _LINE.absorbed),
    )


def _line_members(vis):
    """Split a line visual into its members."""
    return _members(vis, _LINE)


def _merge_lines(template, members):
    """Merge line members into a single visual, styled after `template`."""
    source = template.material

    subs, positions = _layout(members, gap=_LINE.gap)
    geometry_kwargs, material_kwargs = _absorb(
        members, subs, len(positions), _LINE.attrs
    )

    if isinstance(source, gfx.LineThinMaterial):
        # Thin lines are one physical pixel wide whatever the material says,
        # so drop the width the generic absorb produced rather than leaving a
        # buffer on the geometry that nothing reads.
        geometry_kwargs.pop(_THICKNESS.buffer, None)
        material_kwargs.pop(_THICKNESS.value, None)
        material_kwargs.pop(_THICKNESS.mode, None)
        material = gfx.LineThinMaterial(**material_kwargs)
    elif material_kwargs["thickness_mode"] == "vertex":
        # Only per-vertex widths need octarine's own material, so a scene of
        # stock pygfx lines stays on stock pygfx after merging
        from .shaders import FlexLineMaterial

        if source.dash_pattern:
            # The uniform thickness scales the dash pattern, and the merge key
            # made sure every member agrees on it - keep it exactly.
            material_kwargs["thickness"] = source.thickness
        else:
            # Otherwise it goes unused, and the mean is simply the most
            # descriptive value to report (as in `lines2gfx`). The gap
            # vertices are zero, so they only dilute; discount them.
            widths = geometry_kwargs["thicknesses"]
            mean_width = float(widths.sum()) / sum(s.count for s in subs)
            material_kwargs["thickness"] = mean_width if mean_width > 0 else 1
        material = FlexLineMaterial(**material_kwargs)
    else:
        material_kwargs.pop("thickness_mode")
        material = gfx.LineMaterial(**material_kwargs)

    _copy_material_state(source, material, _LINE.absorbed)
    visual = gfx.Line(gfx.Geometry(positions=positions, **geometry_kwargs), material)
    return _finalize(visual, template, subs)


# ------------------------------------------------------------------ points


def _points_key(vis):
    """Merge key for a points visual, or None if it can't be merged."""
    material = vis.material
    if not isinstance(material, gfx.PointsMaterial):
        return None
    if not _mergeable(vis, _POINTS):
        return None

    # Unlike lines, the material class stays exactly as it is: plain points,
    # markers and `FlexPointsMaterial` draw differently enough that promoting
    # one to another would change what the user sees.
    return (
        type(material),
        _object_state(vis),
        _material_state(material, _POINTS.absorbed),
    )


def _points_members(vis):
    """Split a points visual into its members."""
    return _members(vis, _POINTS)


def _merge_points(template, members):
    """Merge point members into a single visual, styled after `template`."""
    source = template.material

    # Points are independent primitives, so they simply concatenate
    subs, positions = _layout(members, gap=_POINTS.gap)
    geometry_kwargs, material_kwargs = _absorb(
        members, subs, len(positions), _POINTS.attrs
    )

    material = type(source)(**material_kwargs)
    _copy_material_state(source, material, _POINTS.absorbed)
    visual = gfx.Points(gfx.Geometry(positions=positions, **geometry_kwargs), material)
    return _finalize(visual, template, subs)


# ------------------------------------------------------------------ meshes


def _mesh_key(vis):
    """Merge key for a mesh visual, or None if it can't be merged."""
    material = vis.material
    if not isinstance(material, gfx.MeshAbstractMaterial):
        return None
    if not _mergeable(vis, _MESH):
        return None

    indices = _buffer(vis.geometry, "indices")
    if indices is None or indices.ndim != 2 or indices.shape[1] != 3:
        return None  # not a triangle mesh

    return (
        type(material),
        # Normals and texcoords are only carried over when every member has
        # them, so their presence has to match. Note that octarine leaves
        # normals off where it can, letting pygfx compute them.
        _buffer(vis.geometry, "normals") is not None,
        _buffer(vis.geometry, "texcoords") is not None,
        _object_state(vis),
        _material_state(material, _MESH.absorbed),
    )


def _mesh_members(vis):
    """Split a mesh visual into its members."""
    geometry = vis.geometry
    indices = _buffer(geometry, "indices")
    normals = _buffer(geometry, "normals")
    texcoords = _buffer(geometry, "texcoords")

    def extra(sub):
        return {
            # Kept as a view plus the offset it is currently relative to, so
            # that `_merge_meshes` can rebase it in a single pass
            "indices": indices[sub.face_slice],
            "vertex_offset": sub.offset,
            "normals": None if normals is None else normals[sub.slice],
            "texcoords": None if texcoords is None else texcoords[sub.slice],
        }

    return _members(vis, _MESH, face_count=len(indices), extra=extra)


def _merge_meshes(template, members):
    """Merge mesh members into a single visual, styled after `template`."""
    source = template.material

    # Triangles are independent, so the vertices simply concatenate ...
    subs, positions = _layout(members, gap=_MESH.gap)
    geometry_kwargs, material_kwargs = _absorb(
        members, subs, len(positions), _MESH.attrs
    )

    # ... but the faces index into that concatenated buffer and so have to be
    # rebased onto it, by however far the member moved.
    n_faces = sum(len(m.attrs["indices"]) for m in members)
    indices = np.empty((n_faces, 3), dtype=np.int32)
    at = 0
    for member, sub in zip(members, subs):
        local = member.attrs["indices"]
        sub.face_offset, sub.face_count = at, len(local)
        np.add(local, sub.offset - member.attrs["vertex_offset"], out=indices[at : at + len(local)])
        at += len(local)
    geometry_kwargs["indices"] = indices

    for name in ("normals", "texcoords"):
        if members[0].attrs[name] is None:
            continue
        geometry_kwargs[name] = np.concatenate(
            [m.attrs[name] for m in members], axis=0, dtype=np.float32
        )

    material = type(source)(**material_kwargs)
    _copy_material_state(source, material, _MESH.absorbed)
    visual = gfx.Mesh(gfx.Geometry(positions=positions, **geometry_kwargs), material)
    return _finalize(visual, template, subs)


# ---------------------------------------------------------------- dispatch


class _Kind:
    """What merging needs to know about one type of visual.

    `attrs` are the properties absorbed into the geometry when members
    disagree, `extra_buffers` whatever else the geometry may legitimately
    carry, `gap` the filler vertices to leave between members, and
    `auto_reads_colors` whether this type's shader reads `geometry.colors`
    under `color_mode="auto"` (see `_mergeable`).

    """

    def __init__(self, attrs, extra_buffers=(), gap=0, auto_reads_colors=False):
        self.attrs = attrs
        self.gap = gap
        self.auto_reads_colors = auto_reads_colors
        self.absorbed = frozenset(
            name for attr in attrs for name in (attr.value, attr.mode)
        )
        self.buffers = ("positions", *(a.buffer for a in attrs), *extra_buffers)


_LINE = _Kind((_COLOR, _THICKNESS), gap=1)
_POINTS = _Kind((_COLOR, _SIZE))
_MESH = _Kind(
    (_COLOR,),
    extra_buffers=("indices", "normals", "texcoords"),
    auto_reads_colors=True,
)

# Keyed by exact type: subclasses (`BoxHelper`, `InstancedMesh`, octarine's
# `TubeVisual`) come with semantics of their own and are left alone.
_MERGERS = {
    gfx.Line: _Merger(_line_key, _line_members, _merge_lines),
    gfx.Points: _Merger(_points_key, _points_members, _merge_points),
    gfx.Mesh: _Merger(_mesh_key, _mesh_members, _merge_meshes),
}


def _batch_members(members, max_size, max_nodes):
    """Chop `members` into batches that respect the two size ceilings."""
    batch, nodes = [], 0
    for member in members:
        count = len(member.positions)
        if batch and (
            (max_size and len(batch) >= max_size)
            or (max_nodes and nodes + count > max_nodes)
        ):
            yield batch
            batch, nodes = [], 0
        batch.append(member)
        nodes += count
    if batch:
        yield batch


def merge_visuals(visuals, max_size=None, max_nodes=_MAX_MERGED_NODES):
    """Merge compatible visuals into fewer pygfx objects.

    Lines, points and meshes are merged; everything else - volumes, text,
    tubes, and any visual whose material or geometry we can't concatenate - is
    passed through untouched and in place, as are visuals that end up alone in
    their group.

    Visuals are merged only if they agree on everything the merged object can
    hold just once: the material's uniforms and blend/depth state, the
    object's transform, visibility and render order. Colour is the exception,
    along with line width and point size - where members disagree on those,
    they move into the geometry as per-vertex buffers.

    Merging is destructive to the per-object handles the viewer relies on
    (`material.color`, `visual.visible`, one `_object_id` per object); those
    are replaced by the vertex ranges in `visual._sub_visuals`. See
    `SubVisual` for how to address an individual member.

    Parameters
    ----------
    visuals :   list of pygfx visuals
                Visuals to merge. Visuals that have already been through this
                function are taken apart member by member and re-merged, so a
                group can be grown one visual at a time.
    max_size :  int, optional
                Maximum number of members per merged visual. `None` (default)
                puts every compatible visual in one object. The gain flattens
                out below ~50 objects, so capping this keeps the buffers small
                enough to rebuild cheaply without giving much up.
    max_nodes : int, optional
                Maximum number of vertices per merged visual. Guards against
                producing buffers larger than the GPU will bind.

    Returns
    -------
    list of pygfx visuals
                The merged visuals. Merged ones carry a `_sub_visuals` list.

    See Also
    --------
    SubVisual
                Addressing an individual object inside a merged visual.

    """
    # Groups in first-appearance order. Visuals we can't merge get a group of
    # their own, which the loop below passes straight through.
    order, buckets = [], {}

    for vis in visuals:
        merger = _MERGERS.get(type(vis))
        key = merger.key(vis) if merger is not None else None
        group = buckets.get((type(vis), key)) if key is not None else None
        if group is None:
            group = []
            order.append(group)
            if key is not None:
                buckets[type(vis), key] = group
        group.append(vis)

    merged = []
    for group in order:
        # A lone visual is already as merged as it is going to get
        if len(group) == 1:
            merged.append(group[0])
            continue

        merger = _MERGERS[type(group[0])]
        members = [m for vis in group for m in merger.members(vis)]
        for batch in _batch_members(members, max_size, max_nodes):
            merged.append(merger.merge(group[0], batch))

    return merged
