"""Addressing one object inside a merged visual.

`merge.merge_visuals` trades N pygfx objects for one, which is what makes
large scenes fast - and which takes away the handles the viewer reaches for
when it wants to act on a single object. `visual.visible`, `material.color`
and `visual.geometry.positions` all belong to the merged object now, and
there are hundreds of originals behind it.

A `SubVisualHandle` puts those handles back. It is shaped like a visual, so
the viewer's per-object code paths - `hide_objects`, `set_colors`,
`highlight_objects`, the legend, box selection - keep working on it
unchanged, but every write lands in the vertex range the member owns rather
than on an object of its own::

    handle = viewer.objects["neuron_1"][0]
    handle.visible = False            # NaNs its slice of the positions
    handle.material.color = "red"     # writes its slice of the colors

What a handle cannot fake is state the merged object can only hold once - the
material's blend and depth settings, the transform, the render order. Writing
one of those asks the manager to split the member back out into a visual of
its own first (see `MergeManager.split_out`), so the write lands somewhere it
can actually be expressed. Reads of those come straight off the merged object,
which is accurate: the merge key made every member agree on them.

Note that only *material* writes are routed that way today, because only
`_HandleMaterial` intercepts them. A handle has no `__setattr__` of its own,
so an object-level write - `handle.render_order = 2` - sets an attribute on
the handle that nothing ever reads, silently. `_write_shared` is already
parameterised by owner and would route it, but nothing passes `"visual"`.
Driving a merged object with the transform gizmo is likewise out: pygfx wants
a real `WorldObject` and refuses a handle. Both are documented as limitations
in `docs/manage.md`; unmerging the object first is the way round them.

Handles are persistent. The viewer stores bookkeeping on its objects
(`_pinned`, `_highlighted`, `_stored_color`), and `Viewer.objects` is a cache
that is cleared constantly, so the manager hands out the same handle for an
object every time rather than building one per lookup.

"""

import weakref

import pygfx as gfx
import numpy as np
import pylinalg as la

from contextlib import contextmanager
from dataclasses import dataclass

from pygfx.utils.bounds import points_to_bounds

from . import utils
from .config import get_logger
from .merge import (
    _COLOR,
    _color_from_physical,
    _color_to_physical,
    _mode,
    colors_are_physical,
    merge_visuals,
    split_visual,
)

logger = get_logger(__name__)

__all__ = ["SubVisualHandle", "MergeManager", "MergeStats", "as_visual"]

# Hoisted: `get_world_bounding_box` runs per object per bounds refresh and
# building a 4x4 identity each time is most of what it does.
_IDENTITY = np.eye(4)

# Distinguishes "no value recorded" from a recorded `None`.
_UNSET = object()


def as_visual(obj):
    """The real pygfx visual behind `obj`.

    A `SubVisualHandle` stands in for an object that lives inside a merged
    visual, so code that needs the actual scene-graph object - to test what
    kind of visual it is, or to reach its transform - goes through here rather
    than knowing the handle type itself.

    """
    return obj.visual if isinstance(obj, SubVisualHandle) else obj


# The bookkeeping the viewer hangs off whatever `Viewer.objects` hands out:
# pinning, the hover highlight and the colour a selected object reverts to.
# It lives on the object, so it has to follow one across the boundary between
# being a visual of its own and being a member of a merged visual - in both
# directions - or pinning and selection quietly stop working the first time a
# canvas is merged.
_CARRIED = ("_pinned", "_highlighted", "_highlight_style", "_stored_color")
_CARRIED_MATERIAL = ("_original_color",)


def _capture_state(obj):
    """Snapshot the viewer's per-object bookkeeping."""
    state = {}
    for name in _CARRIED:
        value = getattr(obj, name, _UNSET)
        if value is not _UNSET:
            state[name] = value
    material = getattr(obj, "material", None)
    for name in _CARRIED_MATERIAL:
        value = getattr(material, name, _UNSET)
        if value is not _UNSET:
            state["material", name] = value
    return state


def _apply_state(obj, state):
    """Put a `_capture_state` snapshot back onto an object."""
    for name, value in state.items():
        if isinstance(name, tuple):
            setattr(obj.material, name[1], value)
        else:
            setattr(obj, name, value)


def member_ids(visual):
    """The object ids `visual` holds: its members if merged, else its own.

    A generator rather than a set: callers ask "is any/are all of these
    covered", and a merged visual can hold hundreds of thousands of members.

    """
    subs = getattr(visual, "_sub_visuals", None)
    if subs:
        return (sub.object_id for sub in subs)
    return iter((getattr(visual, "_object_id", None),))


class _HandleBuffer:
    """One of a member's geometry buffers, as a view into the merged one."""

    __slots__ = ("_handle", "_name")

    def __init__(self, handle, name):
        self._handle = handle
        self._name = name

    @property
    def data(self):
        return self._handle._buffer_data(self._name)

    def update_range(self, offset=0, size=None):
        """Flag this member's range on the merged buffer as changed."""
        self._handle._update_buffer(self._name, offset, size)


class _HandleGeometry:
    """A member's slice of the merged geometry.

    Only the buffers a member actually owns a range of are exposed. Normals
    and texcoords follow the vertices, `indices` the faces; anything the
    merged geometry does not carry is absent here too, so `getattr(geometry,
    name, None)` keeps working the way it does on a real geometry.

    """

    __slots__ = ("_handle", "_buffers")

    def __init__(self, handle):
        self._handle = handle
        # A buffer view is stateless over (handle, name) and there are only a
        # handful of names, so they are made once. Box selection reads
        # `geometry.positions.data` once per object, and at the scale this
        # module exists for that allocation shows up.
        self._buffers = {}

    def __getattr__(self, name):
        buffer = self._buffers.get(name)
        if buffer is None:
            if getattr(self._handle.visual.geometry, name, None) is None:
                raise AttributeError(name)
            buffer = self._buffers[name] = _HandleBuffer(self._handle, name)
        return buffer


class _HandleMaterial:
    """A member's view of the merged material.

    `color` is per-member: the merged visual carries one per vertex, so the
    handle writes the member's slice. Everything else is shared, and a write
    that would actually change it splits the member out first.

    Attributes whose name starts with an underscore are kept on the handle
    rather than forwarded - the viewer stores `_original_color` on the
    material of a highlighted object, and that must not leak onto the
    hundreds of members sharing the merged material.

    """

    def __init__(self, handle):
        # Bypass our own __setattr__, which would treat these as shared writes
        object.__setattr__(self, "_handle", handle)

    @property
    def color(self):
        return gfx.Color(self._handle._read_color())

    @color.setter
    def color(self, value):
        self._handle._write_color(value)

    @property
    def color_mode(self):
        # A member always has a colour of its own, however the merged visual
        # happens to be storing it at the moment.
        return "vertex"

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._handle.visual.material, name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        # A property defined here is one this class intercepts because it is
        # per-member (`color`); everything else belongs to the merged material
        # and is shared by every member of it.
        if isinstance(getattr(type(self), name, None), property):
            object.__setattr__(self, name, value)
            return
        self._handle._write_shared("material", name, value)


class SubVisualHandle:
    """One object inside a merged visual, addressable as if it were its own.

    Parameters
    ----------
    manager :   MergeManager
                The manager that owns this handle. Used to split the member
                back out when a write cannot be expressed on a merged object.
    visual :    pygfx visual
                The merged visual the member currently lives in.
    sub :       merge.SubVisual
                Which vertex (and face) range the member owns.

    See Also
    --------
    merge.SubVisual
                The record this wraps.

    """

    # Defaults for the bookkeeping the viewer hangs off its objects. Without
    # them every `getattr(obj, "_pinned", False)` probe - several per object
    # per styling or legend pass - costs a raised-and-swallowed AttributeError
    # from `__getattr__`.
    _pinned = False
    _highlighted = False

    def __init__(self, manager, visual, sub):
        self._manager = manager
        self.visual = visual
        self.sub = sub
        self._geometry = _HandleGeometry(self)
        self._material = _HandleMaterial(self)
        # Positions are destroyed to hide a member (see `visible`), so the
        # originals are kept here while it is hidden - and only while it is,
        # which keeps the cost proportional to how much is hidden.
        self._stashed_positions = None

    def __repr__(self):
        return (
            f"<SubVisualHandle {self._object_id!r} "
            f"in {type(self.visual).__name__} at {self.sub.offset}:"
            f"{self.sub.offset + self.sub.count}>"
        )

    # ------------------------------------------------------------ identity

    @property
    def _object_id(self):
        return self.sub.object_id

    @property
    def _object_group(self):
        return self.sub.group

    @property
    def _object_type(self):
        return getattr(self.visual, "_object_type", None)

    @property
    def material(self):
        return self._material

    @property
    def geometry(self):
        return self._geometry

    # ------------------------------------------------------------ retarget

    def _retarget(self, visual, sub):
        """Point this handle at a rebuilt merged visual.

        Merging, splitting and compacting all produce *new* pygfx objects, so
        the handles for the members that moved have to be pointed at them -
        the viewer and the legend hold on to handles across all of that.

        """
        self.visual = visual
        self.sub = sub

    # ---------------------------------------------------------- visibility

    @property
    def visible(self):
        return self._stashed_positions is None and self.visual.visible

    @visible.setter
    def visible(self, value):
        if bool(value) == (self._stashed_positions is None):
            return

        buffer = self.visual.geometry.positions
        if value:
            buffer.data[self.sub.slice] = self._stashed_positions
            self._stashed_positions = None
            self._manager._hidden.discard(self)
        else:
            # A copy, not a view: the next line overwrites what it points at.
            self._stashed_positions = buffer.data[self.sub.slice].copy()
            buffer.data[self.sub.slice] = np.nan
            self._manager._hidden.add(self)
        buffer.update_range(self.sub.offset, self.sub.count)
        self._manager._touch()

    # -------------------------------------------------------------- bounds

    def get_bounding_box(self):
        """This member's axis-aligned bounds, in the merged visual's space.

        Computed from the member's own positions rather than the merged
        object's, and from the stash while it is hidden - `Viewer.get_bounds`
        documents itself as covering visible *and* invisible objects, and a
        hidden member's vertices are NaN in the buffer.

        Deliberately pygfx's own `points_to_bounds`: it is what every real
        visual's bounds go through, it already drops non-finite vertices (which
        a merged line's gap vertices and a hidden member's are), and going
        through it is what keeps a handle's bounds identical to those of the
        unmerged object it stands in for.

        """
        aabb, _ = points_to_bounds(self._positions(), calculate_radius=False)
        return aabb

    def get_world_bounding_box(self):
        """This member's axis-aligned bounds, in world space."""
        aabb = self.get_bounding_box()
        if aabb is None:
            return None
        matrix = self.visual.world.matrix
        if np.array_equal(matrix, _IDENTITY):
            return aabb
        return la.aabb_transform(aabb, matrix)

    # ------------------------------------------------------- shared fields

    def __getattr__(self, name):
        # Only reached for names this class does not define, which is where
        # the shared object-level state (`render_order`, `cast_shadow`, ...)
        # and anything a plugin reads off a visual ends up.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.visual, name)

    # ------------------------------------------------------------ internals

    def _positions(self):
        """The member's positions, from the stash while it is hidden."""
        if self._stashed_positions is not None:
            return self._stashed_positions
        return self.visual.geometry.positions.data[self.sub.slice]

    def _buffer_data(self, name):
        if name == "positions":
            return self._positions()
        buffer = getattr(self.visual.geometry, name)
        if name == "indices":
            return buffer.data[self.sub.face_slice]
        return buffer.data[self.sub.slice]

    def _update_buffer(self, name, offset, size):
        buffer = getattr(self.visual.geometry, name)
        base = self.sub.face_offset if name == "indices" else self.sub.offset
        count = self.sub.face_count if name == "indices" else self.sub.count
        buffer.update_range(base + offset, count if size is None else size)
        self._manager._touch()

    def _read_color(self):
        colors = getattr(self.visual.geometry, "colors", None)
        if colors is None:
            # Still a single colour for the whole merged visual
            return self.visual.material.color
        stored = colors.data[self.sub.offset]
        # Whether the buffer holds physical or sRGB values depends on the
        # visual type - see `merge.colors_are_physical`.
        if colors_are_physical(self.visual):
            return _color_from_physical(stored)
        return gfx.Color(stored)

    def _write_color(self, value):
        if colors_are_physical(self.visual):
            rgba = _color_to_physical(value)
        else:
            rgba = np.asarray(gfx.Color(value).rgba, dtype=np.float32)
        # Writing one member's colour is what forces a merged visual off its
        # shared colour, so the buffer may have to be created here.
        colors = self._manager._ensure_vertex_colors(self.visual)
        colors.data[self.sub.slice] = rgba
        colors.update_range(self.sub.offset, self.sub.count)
        self._manager._touch()

    def _write_shared(self, owner, name, value):
        """Apply a write that a merged object can only hold once.

        Inside a bulk write this member stakes its claim on the shared value
        first (see `MergeManager.bulk_shared_write`), *before* the "nothing
        would change" shortcut below. Taking the shortcut first would let a
        member that is happy with the current value stay silent, and a later
        member asking for a different one would then find the visual
        unclaimed and write its value to everybody.

        Outside a bulk, an unchanged value is simply dropped: splitting an
        object out to write what is already there would be pure loss.

        """
        target = self.visual.material if owner == "material" else self.visual
        try:
            current, known = getattr(target, name), True
        except AttributeError:
            current, known = None, False

        if self._manager._covered_by_bulk(self.visual, name, value):
            # Every member of this visual is in this bulk and none of them has
            # asked for anything else, so it can go on the shared object.
            if not (known and _same(current, value)):
                setattr(target, name, value)
                self._manager._touch()
            return

        if known and _same(current, value):
            return

        split = self._manager.split_out([self._object_id])[self._object_id]
        self._manager.note_erosion()
        setattr(getattr(split, owner) if owner == "material" else split, name, value)


def _same(current, value):
    """Whether writing `value` over `current` would change anything."""
    if current is value:
        return True
    if isinstance(current, gfx.Color):
        try:
            return tuple(current.rgba) == tuple(gfx.Color(value).rgba)
        except Exception:
            return False
    try:
        return bool(np.array_equal(current, value))
    except Exception:
        return current == value


# ------------------------------------------------------------------ manager


@dataclass
class MergeStats:
    """What merging currently costs and saves on a canvas.

    Attributes
    ----------
    objects :       int
                    Objects on the canvas, merged or not.
    visuals :       int
                    pygfx objects the renderer actually walks. This is the
                    number merging exists to bring down.
    merged :        int
                    How many of those visuals hold more than one member.
    members :       int
                    Objects currently living inside a merged visual.
    largest :       int
                    Members in the biggest merged visual.
    split_outs :    int
                    Members that have been split back out since the canvas
                    was created, because something was asked of them that a
                    merged object cannot express. A number that climbs during
                    normal use means the merge is being eroded - see
                    `SubVisualHandle._write_shared`.

    """

    objects: int = 0
    visuals: int = 0
    merged: int = 0
    members: int = 0
    largest: int = 0
    split_outs: int = 0


class MergeManager:
    """Keeps a canvas's merged visuals and their per-object handles in step.

    `merge.merge_visuals` and `merge.split_visual` are one-shot functions over
    a list of visuals. This is the live layer on top: it decides when to merge,
    owns the handle for every merged-in object, and holds the one invariant
    that the rest of the feature depends on - that a member hidden before a
    rebuild is still hidden after it.

    Parameters
    ----------
    viewer :        Viewer
                    The canvas this manages. Held weakly: the viewer owns the
                    manager.
    auto_merge :    bool | int
                    `False` to never merge, `True` to merge whenever there is
                    anything to merge, or a count to only start merging once
                    the canvas holds that many objects. Merging only pays off
                    in bulk, so a threshold keeps small scenes on the simple
                    path.
    max_size :      int, optional
                    Members per merged visual, passed to `merge_visuals`.

    """

    def __init__(self, viewer, auto_merge=False, max_size=None):
        self._viewer = weakref.ref(viewer)
        self.auto_merge = auto_merge
        self.max_size = max_size

        # object_id -> [handle], one per member that object has in a merged
        # visual. Persistent: the viewer hangs `_pinned`, `_highlighted` and
        # friends off these, and `Viewer.objects` is a cache that gets cleared.
        self._handles = {}
        # id(merged visual) -> its handles in buffer order, so that
        # `handles_for` is a lookup rather than a scan. Rebuilt by `sync`.
        self._by_visual = {}
        # The handles whose members are currently hidden. Hiding is rare and
        # the rebuild paths only care about hidden members, so tracking them
        # keeps those paths proportional to what is hidden rather than to how
        # many objects the canvas holds.
        self._hidden = set()
        self._split_outs = 0
        # The object ids a bulk shared write currently covers, if one is in
        # progress - see `bulk_shared_write`.
        self._bulk = None
        # (visual, property) -> value already written there by the bulk in
        # progress, so that a differing one is not quietly applied on top.
        # Keyed by the visual itself rather than its id: a split frees visuals
        # mid-bulk and an address can be reused by the one built to replace it.
        self._bulk_values = {}
        # object_id -> the order it was added in. Merging and splitting both
        # replace visuals, and a replacement goes on the end of the scene, so
        # the scene's own order stops reflecting the order things were added
        # in as soon as anything is split. `Viewer.objects` documents itself as
        # being in order of addition, and the legend is built from it.
        self._order = {}
        self._next_order = 0
        # Whether the scene's merged visuals have changed since `sync` last
        # ran, and whether anything has been split (which is the only thing
        # that puts the scene out of addition order). Both let the hot path in
        # `Viewer._objects` skip work that would otherwise be paid per update.
        self._handles_stale = True
        self._order_stale = False

    # -------------------------------------------------------------- policy

    @property
    def enabled(self):
        """Whether merging is switched on at all."""
        return self.auto_merge is not False and self.auto_merge is not None

    def register(self, visual, merge=None):
        """Note a visual on its way into the scene.

        `merge=False` is recorded on the visual itself, the way `_object_id`
        and `_object_group` are: it belongs to that one object, it never has
        to survive a merge (an excluded object is never merged), and it is
        collected with the visual instead of lingering in a set by name.

        """
        if merge is False:
            # `_never_merge` marks it as the object's own setting, which an
            # explicit `Viewer.merge_objects` leaves alone (unlike the flag
            # `unmerge_objects` sets).
            visual._no_merge = visual._never_merge = True
        self.order_key(visual._object_id)

    def reset(self):
        """Forget everything about the canvas's objects.

        For `Viewer.clear`: the handles point at visuals that are no longer in
        the scene, and an object added after a clear should start at the front
        of the order again even if it reuses a name.

        """
        self._handles.clear()
        self._by_visual.clear()
        self._hidden.clear()
        self._order.clear()
        self._next_order = 0
        self._handles_stale = True
        self._order_stale = False

    def forget(self, object_ids):
        """Drop bookkeeping for objects that have left the canvas.

        Without this a session that cycles objects in and out keeps one order
        entry per object ever added.

        """
        for object_id in utils.make_iterable(object_ids):
            self._order.pop(object_id, None)

    def order_key(self, object_id):
        """Sort key putting an object back where it was added.

        Objects are numbered on their way in (`Viewer._add_to_scene`); an
        object that somehow arrives without one is numbered on first sight,
        which is the same answer as long as nothing has been split yet.

        N.B. the counter only ever climbs. Numbering from `len(self._order)`
        would reuse the number of an object that has since been removed, and
        the two would then sort equal.

        """
        if object_id not in self._order:
            self._order[object_id] = self._next_order
            self._next_order += 1
        return self._order[object_id]

    @contextmanager
    def bulk_shared_write(self, object_ids):
        """Apply a shared property to many objects without taking them apart.

        A property a merged object can hold only once normally splits a member
        out when it is written (see `SubVisualHandle._write_shared`) - but the
        viewer's styling methods walk their targets one at a time, and most of
        them are called for *every* object on the canvas. `Viewer.set_colors`
        alone sets `alpha_mode` on everything it touches, so without this a
        single call would take a whole merged scene apart.

        Inside this context a write is applied to the merged material as it
        stands whenever every member of that merged visual is among
        `object_ids` *and* every member is getting the same value. Members only
        partly covered still split, and so does a member that wants a different
        value from the one its merged visual has already been given: a shared
        property is shared, so "all of them want this" is the only case where
        writing it once is the same thing as writing it per object.

        That second condition is not hypothetical - `Viewer.set_colors` writes
        `alpha_mode="add"` for the objects it is making transparent and
        `"solid"` for the rest, in one pass over the canvas. Without it the
        last member written would decide the blend mode for all of them.

        """
        previous, previous_values = self._bulk, self._bulk_values
        self._bulk = set(utils.make_iterable(object_ids))
        # (visual, property) -> the value this bulk already wrote there
        self._bulk_values = {}
        try:
            yield
        finally:
            self._bulk_values = previous_values
            self._bulk = previous

    def note_shared_intent(self, obj, name, value):
        """Record what `obj` wants a shared property to be, without writing it.

        For a caller that has decided the write is unnecessary - the value it
        wants is already what the object renders as - but whose silence would
        otherwise be taken as indifference. Inside a bulk write the first
        member to speak claims the shared value for the whole visual, so a
        member that says nothing lets a later one with a *different* value
        claim it and apply that to everybody. See `bulk_shared_write`.

        """
        if self._bulk is not None and isinstance(obj, SubVisualHandle):
            self._covered_by_bulk(obj.visual, name, value)

    def _covered_by_bulk(self, visual, name, value):
        """Whether this write can go straight onto `visual`'s shared object.

        Only when the bulk covers every member *and* nothing else in it has
        already written a different value here - see `bulk_shared_write`.

        """
        if self._bulk is None:
            return False

        written = self._bulk_values.get((visual, name), _UNSET)
        if written is not _UNSET:
            # Already decided for this visual: the same value can go on the
            # shared object again (it is a no-op), a different one cannot.
            return _same(written, value)

        # Short-circuits on the first member that isn't covered, and allocates
        # nothing - a merged visual can hold hundreds of thousands of members.
        # The answer is then remembered, so this is paid once per visual.
        covered = all(object_id in self._bulk for object_id in member_ids(visual))
        if covered:
            self._bulk_values[visual, name] = value
        return covered

    # --------------------------------------------------------------- hooks

    def _touch(self):
        """Flag the canvas as needing a redraw."""
        viewer = self._viewer()
        if viewer is not None:
            viewer._render_stale = True

    def _candidates(self):
        """The scene's visuals that merging is allowed to touch."""
        viewer = self._viewer()
        if viewer is None:
            return []
        return [
            vis
            for vis in viewer.visuals
            if getattr(vis, "_object_type", None) != "boundingbox"
            and not getattr(vis, "_no_merge", False)
        ]

    def merged_visuals(self):
        """The scene's merged visuals, i.e. those standing for several objects."""
        viewer = self._viewer()
        if viewer is None:
            return []
        return [v for v in viewer.scene.children if getattr(v, "_sub_visuals", None)]

    # -------------------------------------------------------------- merging

    def maybe_apply(self):
        """Run a merge pass if the policy calls for one.

        Called once per frame from `Viewer._refresh_scene`, which is where
        everything derived from the scene as a whole catches up - so a loop of
        `add` calls triggers one merge rather than one per object.

        """
        if not self.enabled:
            return

        candidates = self._candidates()
        if isinstance(self.auto_merge, int) and not isinstance(self.auto_merge, bool):
            if len(candidates) < self.auto_merge:
                return
        if len(candidates) < 2:
            return

        self.apply(candidates)

    def apply(self, candidates=None):
        """Merge what can be merged, now."""
        viewer = self._viewer()
        if viewer is None:
            return
        if candidates is None:
            candidates = self._candidates()
        if len(candidates) < 2:
            return

        loose = [vis for vis in candidates if not getattr(vis, "_sub_visuals", None)]
        if not loose:
            # Nothing new to fold in. Without this, a scene that merges into
            # more than one visual - which `max_size` and the vertex ceiling
            # guarantee at the scale this exists for - would be taken apart and
            # rebuilt on every refresh, replacing every GPU buffer, forever:
            # `merge_visuals` buckets two already-merged visuals that share a
            # key together, and the "nothing changed" check below only notices
            # after the work.
            return

        # N.B. the whole scene is re-merged, not just the loose objects.
        # Merging only those among themselves would avoid rebuilding what is
        # already merged, but it leaves a new visual behind per batch and the
        # canvas fragments - which is the cost this exists to avoid.
        #
        # The price is that `merge_visuals` rebuilds every visual it is given,
        # so folding one new object into a merged visual of N costs O(N):
        # adding objects one at a time *with a frame drawn between each* is
        # quadratic. Measured, 100-vertex lines: 800 objects cost one merge
        # pass and ~0.3 s added in a loop, against 799 passes and ~4 s with a
        # frame in between - a cell per object in Jupyter, or a GUI being
        # interacted with. Fixing it properly means appending into
        # over-allocated buffers instead of rebuilding, which is a larger
        # change than this one; `docs/manage.md` records it as a limitation.
        candidates = list(candidates)
        # Snapshot the viewer's bookkeeping off the objects that are about to
        # stop being visuals of their own; already-merged ones keep theirs on
        # handles, which `sync` retargets rather than replaces.
        carried = {
            vis._object_id: _capture_state(vis)
            for vis in candidates
            if not getattr(vis, "_sub_visuals", None)
        }
        self._restore_hidden()
        merged = merge_visuals(candidates, max_size=self.max_size)

        if merged == candidates:
            # Nothing merged after all, so the restore above has to be undone
            # here - no rebuild is coming to re-apply it.
            self._rehide()
            return

        kept = set(map(id, merged))
        for vis in candidates:
            if id(vis) not in kept:
                viewer.scene.remove(vis)
        for vis in merged:
            if vis.parent is None:
                viewer.scene.add(vis)

        # Merging appends the new visuals to the end of the scene, so the
        # scene's order is no longer the order objects were added in.
        self._order_stale = True
        self._after_rebuild()

        for object_id, state in carried.items():
            for handle in self._handles.get(object_id, ()):
                _apply_state(handle, state)

    def drop(self, object_ids):
        """Take objects out of their merged visuals without rebuilding them.

        For `Viewer.remove_objects`: the members are going away, so the
        remainder is all that has to be built.

        """
        self.split_out(object_ids, removing=True)

    def note_erosion(self, count=1):
        """Record that objects left a merge because of something asked of them.

        This is what `MergeStats.split_outs` reports, so it counts only the
        splits nobody asked for directly - a style a merged object cannot hold
        per object. Removing objects, unmerging on request and switching the
        policy off are all deliberate, and none of them is the merge being
        eaten away.

        """
        self._split_outs += count

    def split_out(self, object_ids=None, *, removing=False, keep_apart=False):
        """Split the given object(s) out into visuals of their own.

        This is what happens when something is asked of a single member that a
        merged object cannot express - a silhouette, a blend mode, a transform
        of its own. The member stops being a member; its handle is dropped and
        `Viewer.objects` reports the real visual again.

        Parameters
        ----------
        object_ids :    hashable | list thereof | None
                        Objects to split out. `None` splits every member of
                        every merged visual, leaving nothing merged. Ids that
                        are not in a merged visual are ignored.
        keep_apart :    bool
                        Whether the split-out visuals should stay out of
                        later merges. True when the user asked for the split
                        (`Viewer.unmerge_objects`), which would otherwise be
                        undone by the next `add`.
        removing :      bool
                        Whether these objects are leaving the canvas. Then
                        there is no point building the split-out visuals, only
                        the remainder. See `drop`.

        Returns
        -------
        dict
                    ``{object_id: visual}`` for the objects actually split out.

        """
        viewer = self._viewer()
        merged = self.merged_visuals()
        remaining = (
            {object_id for vis in merged for object_id in member_ids(vis)}
            if object_ids is None
            else set(utils.make_iterable(object_ids))
        )
        out, touched = {}, False
        hidden_ids = {handle._object_id for handle in self._hidden}
        carried = {
            object_id: _capture_state(handles[0])
            for object_id, handles in self._handles.items()
            if object_id in remaining and handles
        }

        for visual in merged:
            present = remaining.intersection(member_ids(visual))
            if not present:
                continue

            self._restore_hidden(visual)
            remainder, extracted = split_visual(visual, present, extract=not removing)

            viewer.scene.remove(visual)
            for vis in ([remainder] if remainder is not None else []) + extracted:
                viewer.scene.add(vis)
                # Anything that comes out of this *not* merged has no handle
                # any more, so the NaN stash can no longer express it being
                # hidden - the visual has to carry it. Note this catches the
                # remainder too: a remainder of one member is stripped back to
                # a plain visual by `merge._strip_if_whole`.
                if not getattr(vis, "_sub_visuals", None):
                    if vis._object_id in hidden_ids:
                        vis.visible = False
                    _apply_state(vis, carried.get(vis._object_id, {}))
            # A split inside a bulk write replaces the visual, and the values
            # that bulk has already written to it have to follow - otherwise
            # the next differing write sees a clean slate and lands on the
            # shared material after all, which is what splitting avoided.
            if self._bulk_values and remainder is not None:
                for (vis, name), value in list(self._bulk_values.items()):
                    if vis is visual:
                        self._bulk_values[remainder, name] = value
            for vis in extracted:
                out[vis._object_id] = vis
                if keep_apart:
                    vis._no_merge = True
            for object_id in present:
                self._handles.pop(object_id, None)
            remaining -= present
            touched = True

        if touched:
            # Splitting puts the new visuals on the end of the scene, so the
            # scene's order is no longer the order objects were added in.
            self._order_stale = True
            self._after_rebuild()
            logger.debug("Split objects out of their merged visual(s)")

        return out

    # ---------------------------------------------------------- invariants

    def _restore_hidden(self, visual=None):
        """Put hidden members' real positions back, before a rebuild.

        Hiding a member overwrites its positions with NaN, so a rebuild that
        read the buffers as they stand would bake the NaN in. The stash is the
        single source of truth for what a hidden member's geometry actually
        is; this puts it back, and `_rehide` - via `_after_rebuild` once the
        handles point at the rebuilt visuals - takes it away again.

        `visual` limits it to one merged visual; `None` restores every hidden
        member, which is what a whole-canvas merge pass wants.

        """
        for handle in self._hidden:
            if visual is None or handle.visual is visual:
                buffer = handle.visual.geometry.positions
                buffer.data[handle.sub.slice] = handle._stashed_positions

    def _rehide(self):
        """Re-apply every hidden member's NaN, wherever its handle now points."""
        for handle in self._hidden:
            buffer = handle.visual.geometry.positions
            buffer.data[handle.sub.slice] = np.nan
            buffer.update_range(handle.sub.offset, handle.sub.count)

    def _after_rebuild(self):
        """Re-point handles at the rebuilt visuals and re-apply hidden state."""
        self._handles_stale = True
        self.sync()
        self._rehide()
        self._normalise_visibility()
        # Whatever `Viewer.objects` is holding describes the scene as it was:
        # visuals that have left the canvas, handles pointing into them.
        # Anything read from it now would be acting on discarded objects. The
        # legend is rebuilt from that same dictionary, so it goes too - a
        # rebuild is exactly when what it has to show changes.
        viewer = self._viewer()
        if viewer is not None:
            # Imported here rather than at module scope: `viewer` imports this
            # module, so the other direction can only be taken on demand.
            from .viewer import update_helper

            update_helper(viewer, legend=True, bounds=False)
        self._touch()

    def _normalise_visibility(self):
        """Express a merged visual's hiddenness per member instead.

        Visibility is part of the merge key, so objects that were hidden
        before they were merged - while they were still visuals of their own -
        end up together in a merged visual that is itself invisible. A handle
        cannot work with that: its member is hidden by something it does not
        own, so unhiding it would have to make every other member visible too,
        and `SubVisualHandle.visible` would simply report False forever.

        So the merged visual is made visible again and each of its members is
        hidden the way a member is hidden, through the stash.

        """
        for visual in self.merged_visuals():
            if visual.visible:
                continue
            visual.visible = True
            for handle in self.handles_for(visual):
                handle.visible = False

    # -------------------------------------------------------------- handles

    def sync(self):
        """Bring the handle registry in line with the scene.

        Handles are matched to members by object id and position within that
        id, so an object whose member count has not changed keeps the handle
        it already had - along with everything the viewer stored on it.

        Skipped unless something has actually replaced a merged visual.
        `Viewer.objects` is rebuilt on every viewer update and asks for this
        each time, but merged visuals only change through `apply` and
        `split_out`, which both flag it.

        """
        if not self._handles_stale:
            return
        self._handles_stale = False

        seen, by_visual = {}, {}
        for visual in self.merged_visuals():
            # Built here rather than derived on demand: `Viewer.objects` wants
            # one merged visual's handles at a time, and picking them out of
            # the registry per visual would make that quadratic in the number
            # of objects - which is the number this whole module exists for.
            handles = []
            for sub in visual._sub_visuals:
                existing = self._handles.get(sub.object_id, [])
                index = len(seen.get(sub.object_id, []))
                if index < len(existing):
                    handle = existing[index]
                    handle._retarget(visual, sub)
                else:
                    handle = SubVisualHandle(self, visual, sub)
                seen.setdefault(sub.object_id, []).append(handle)
                handles.append(handle)
            by_visual[id(visual)] = handles

        self._handles = seen
        self._by_visual = by_visual
        # Handles that were replaced rather than retargeted are no longer the
        # ones the registry hands out, so a stale one must not keep a hidden
        # member hidden (or hold its positions alive).
        if self._hidden:
            live = {handle for handles in seen.values() for handle in handles}
            self._hidden &= live

    def handles_for(self, visual):
        """The handles for one merged visual's members, in buffer order."""
        return self._by_visual.get(id(visual), [])

    def _ensure_vertex_colors(self, visual):
        """Give a merged visual a per-vertex color buffer, if it hasn't one.

        Members that agreed on their color were merged with a single uniform
        color (see `merge._absorb`); the first per-object color write is what
        forces them apart, and it happens here rather than at merge time so
        that a scene nobody recolors never pays for the buffer.

        """
        geometry, material = visual.geometry, visual.material
        colors = getattr(geometry, "colors", None)

        if colors is None:
            n = len(geometry.positions.data)
            if colors_are_physical(visual):
                rgba = _color_to_physical(material.color)
            else:
                rgba = np.asarray(gfx.Color(material.color).rgba, dtype=np.float32)
            colors = gfx.Buffer(np.tile(rgba, (n, 1)))
            geometry.colors = colors

        # Must be explicit: under "auto" a mesh shader multiplies the uniform
        # color into the vertex colors instead of replacing it. Read through
        # `merge._mode`, which normalises what pygfx hands back.
        if _mode(material, _COLOR.mode) != "vertex":
            material.color_mode = "vertex"

        return colors

    # ---------------------------------------------------------------- stats

    @property
    def stats(self):
        """A `MergeStats` snapshot of this canvas."""
        viewer = self._viewer()
        if viewer is None:
            return MergeStats()

        visuals = viewer.visuals
        members = [len(v._sub_visuals) for v in visuals if getattr(v, "_sub_visuals", None)]

        return MergeStats(
            # The viewer's own count, rather than arithmetic that could drift
            # from it (`objects` is cached, so this is not a new walk).
            objects=len(viewer.objects),
            visuals=len(visuals),
            merged=len(members),
            members=sum(members),
            largest=max(members, default=0),
            split_outs=self._split_outs,
        )
