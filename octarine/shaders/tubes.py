"""Custom pygfx WorldObject/Material/Shader for parametric tubes.

Renders a skeleton with a per-node radial profile

    r(theta) = a0 + sum_k [ a_k cos(k*theta) + b_k sin(k*theta) ]

as a smooth tube surface, without ever materializing a mesh: the shader pulls
every vertex out of a coefficient storage buffer, indexed by `vertex_index`
alone (see `wgsl/tube.wgsl`).

The point of that is level of detail. The number of angular samples
(`n_theta`) and the number of harmonics evaluated (`k_max`) live in the
material's uniform buffer, so changing either is a uniform write plus a
smaller draw call - no re-upload, no reallocation, and the (comparatively
large) coefficient buffer never moves. Axial LOD is not free in the same way:
it needs a different edge list, but that is a small index buffer swap while
the coefficients stay put.

Importing this module registers the shader with pygfx (via the
`register_wgpu_render_function` decorator).


What has been tried
-------------------

Arbors traced from voxel data look jagged when swept - the "pile of
intersecting discs" - and the obvious reading is that the *sweep* is at fault.
Four attempts at fixing it that way have now been made and three were reverted.
Notes here so the next person does not repeat them.

Numbers below are from a 12,604-node arbor at 16-unit voxel spacing with K = 16
harmonics. Its relevant property is that it is heavily oversampled along the
axis relative to its own radius: reach / edge length is 3.5 at the median, and
94% of edges are shorter than the tube is wide. Two metrics are referenced:

  punch-through   fraction of ring samples that cross a neighbour's
                  cross-section plane while still inside that neighbour's
                  profile, i.e. rings that have stabbed through each other
  lost detail     p95 distance from the full-resolution surface to the
                  candidate one, as a percentage of the median a0

1. Union of solids (raycast).  One round cone per edge, unioned, so that
   neither self-intersection nor junction seams can show at all. Worked, but
   was only a marginal improvement on raising `axial_lod` while costing per
   pixel rather than per vertex, and it carried a spatial index of its own.
   Note for anyone tempted to revisit it: a *hard* union is itself creased -
   the intersection curve of two tilted cones is a genuine sharp feature, so
   the union boundary is exactly the jagged thing it was supposed to remove. A
   smooth union (`smin`) would not be, but that is still per pixel.

2. Axial decimation (`axial_lod`).  Kept, but as a *cost* lever only. It is a
   weak quality lever and an expensive one: `axial_lod=1` takes punch-through
   from 4.3% to 3.2% while losing 83% of a0 in detail. Its docstring used to
   claim it was the main quality knob; it is not. Decimating a jittery axis
   also raises the turn angle roughly as fast as it raises the chord length,
   which is why the fold criterion barely improves.

3. Centreline smoothing (Laplacian, with the displacement absorbed into k=1).
   Reverted. The centreline is a free parameter - the k = 1 harmonic *is* a
   translation of the cross-section's centre, so moving a node by `delta` and
   setting `a_1 -= delta . u`, `b_1 -= delta . v` leaves the surface where it
   was to first order. That works exactly as advertised (one pass: punch-
   through 4.3% -> 2.0% at 25% lost detail, i.e. the same self-intersection
   reduction as `axial_lod=1` for a third of the damage) and it still did not
   look meaningfully better - and introduced artefacts of its own, most likely
   where the pinned run ends kink against their smoothed neighbours, and where
   `delta` grows comparable to `r` so the first-order absorption stops holding.

4. Cubic B-spline along the axis.  Reverted. Each edge swept as a segment of a
   uniform cubic B-spline over four consecutive rings (an (E, 4) control table,
   since a tree cannot give "the previous node" from the node index), blending
   position, frame, a0 and every harmonic, with `n_axial` sub-quads per edge as
   a uniform. An approximating spline never interpolates its control rings, so
   this is C2 and removes the lerp's normal discontinuity - which is real and
   large: a median 20 degrees at *every* ring, above 30 degrees at a third of
   them, p99 at 177 degrees. It was verified correct (a straight constant-
   radius tube is its own B-spline, and both paths agreed to within a pixel on
   silhouette, outward normals and taper shading). It still did not look
   meaningfully better.

What that leaves.  Three independent axial treatments - decimating, moving the
axis, and making the sweep C2 - all moved the metrics and none moved the look.
The evidence points at the *angular* profile instead:

  - `|m_k|` changes by 50-65% of its own run-mean between adjacent nodes, at
    every k. Consecutive cross-sections are genuinely different shapes, so the
    surface ripples between them however the sweep is done.
  - Rendering the same data at `k=3` instead of the full K = 16 is dramatically
    cleaner under *any* axial treatment. That is the one change so far that
    visibly fixes it.

So the next thing to try is on the coefficient side, not the sweep: low-passing
`a_k`/`b_k` along each run, tapering the harmonics with k, or fitting the
profile with a smoothness penalty upstream in sparse-cubes. Note that unlike
the k = 1 absorption above, any of those genuinely changes shape - that is a
decision about what the profile should represent, not a rendering trick.

What did help, and stays: `align_frames`. The stored frames are rotation-
minimizing only within an unbranched run, and the chain restarts at every
branch point - 461 edges here had more than 45 degrees of twist between their
two rings and every single one of them touched a node of degree != 2. Rebuilding
one chain over the whole tree is an exact rotation of the profile (see
`rotate_profile`), so it costs no shape at all.
"""

from collections import deque

import numpy as np
import pygfx as gfx
import wgpu  # only for flags/enums

from pygfx.renderers.wgpu import (
    register_wgpu_render_function,
    BaseShader,
    Binding,
    load_wgsl,
)

# Floats per node before the harmonics: position (3), quaternion (4), a0 (1)
HEADER_FLOATS = 8


def _split_coefficients(coefs):
    """Validate a (M, 8 + 2K) coefficient array and return (array, K)."""
    coefs = np.ascontiguousarray(coefs, dtype=np.float32)
    if coefs.ndim != 2:
        raise ValueError(f"Expected a (M, 8 + 2K) coefficient array, got {coefs.shape}")
    n_extra = coefs.shape[1] - HEADER_FLOATS
    if n_extra < 0 or n_extra % 2:
        raise ValueError(
            "Coefficient array must have 8 + 2K columns (position, quaternion, "
            f"a0, then K cosine and K sine coefficients), got {coefs.shape[1]}"
        )
    return coefs, n_extra // 2


def node_reach(coefs, k=None):
    """Per-node upper bound on r(theta), i.e. a0 + sum_k hypot(a_k, b_k)."""
    if k is None:
        _, k = _split_coefficients(coefs)
    a = coefs[:, HEADER_FLOATS : HEADER_FLOATS + k]
    b = coefs[:, HEADER_FLOATS + k : HEADER_FLOATS + 2 * k]
    return coefs[:, 7] + np.hypot(a, b).sum(axis=1)


def _adjacency(edges, n_nodes):
    """CSR-style adjacency: for each node, its neighbours and the edges reaching
    them. `edges[:, ::-1].ravel()` is the *other* endpoint of each half-edge."""
    deg = np.bincount(edges.ravel(), minlength=n_nodes)
    order = np.argsort(edges.ravel(), kind="stable")
    nbr = edges[:, ::-1].ravel()[order]
    via = np.repeat(np.arange(len(edges)), 2)[order]
    indptr = np.concatenate([[0], np.cumsum(deg)])
    return deg, indptr, nbr, via


def split_runs(edges, n_nodes):
    """Split an edge list into its unbranched runs.

    Each run is a list of node indices `[start, ...interior..., end]` whose
    ends have degree != 2 - branch points and tips - and whose interior is all
    degree 2. Every edge belongs to exactly one run. A closed loop, which has
    no node of degree != 2 to start from, is broken open at an arbitrary node
    and comes back with `run[0] == run[-1]`.

    Parameters
    ----------
    edges :     (E, 2) array
                Index pairs into the nodes.
    n_nodes :   int
                Number of nodes the indices refer to.

    Returns
    -------
    list of list of int

    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if not len(edges):
        return []

    deg, indptr, nbr, via = _adjacency(edges, n_nodes)
    essential = deg != 2
    seen = np.zeros(len(edges), dtype=bool)
    runs = []

    def walk(start, first_edge, first_node):
        """Follow a run leaving `start`; return its interior nodes and its end."""
        interior = []
        cur, e = first_node, first_edge
        while True:
            seen[e] = True
            if essential[cur] or cur == start:
                return interior, cur
            interior.append(cur)
            nxt, nxt_e = -1, -1
            for m in range(indptr[cur], indptr[cur + 1]):
                if via[m] != e:
                    nxt, nxt_e = nbr[m], via[m]
                    break
            if nxt < 0:  # a degree-2 node whose two half-edges are one edge
                return interior, cur
            cur, e = nxt, nxt_e

    for a in np.flatnonzero(essential):
        for m in range(indptr[a], indptr[a + 1]):
            if not seen[via[m]]:
                interior, end = walk(a, via[m], nbr[m])
                runs.append([int(a), *map(int, interior), int(end)])

    # Runs with no essential node at all (a closed loop) are not reached above;
    # break each one open at an arbitrary node.
    for e in np.flatnonzero(~seen):
        if seen[e]:
            continue
        a = int(edges[e, 0])
        interior, end = walk(a, e, int(edges[e, 1]))
        runs.append([a, *map(int, interior), int(end)])

    return runs


def decimate_edges(edges, n_nodes, step):
    """Thin an edge list `step`-fold along its unbranched runs.

    Nodes of degree != 2 - branch points and tips - are always kept, so the
    topology is untouched and no arm can go missing; only the interior of each
    run is thinned. Striding the node array instead would silently reconnect
    unrelated branches.

    Note that this leaves the surviving nodes' *frames* describing a centreline
    that no longer exists: the stored tangent is the local direction at full
    resolution, and the chord it is now swept along spans `step` times as much
    of the centreline's jitter. Follow with `align_frames(..., retangent=True)`
    or the decimated tube's rings sit further off their sweep than the
    undecimated ones did.

    Parameters
    ----------
    edges :     (E, 2) array
                Index pairs into the nodes.
    n_nodes :   int
                Number of nodes the indices refer to.
    step :      int
                Keep every `step`-th interior node. 1 is a no-op.

    Returns
    -------
    (E', 2) int32 array

    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if step <= 1 or not len(edges):
        return np.ascontiguousarray(edges, dtype=np.int32)

    kept = []
    for run in split_runs(edges, n_nodes):
        chain = [run[0], *run[1:-1][step - 1 :: step], run[-1]]
        for a, b in zip(chain[:-1], chain[1:]):
            if a != b:
                kept.append((a, b))

    if not kept:
        return np.zeros((0, 2), dtype=np.int32)
    return np.ascontiguousarray(kept, dtype=np.int32)


def _frame_uvt(quat):
    """Unpack (M, 4) xyzw quaternions into the frame's (u, v, t) columns.

    The CPU mirror of the shader's `frame_uv` / `node_tangent`.
    """
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    u = np.stack(
        [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], axis=1
    )
    v = np.stack(
        [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)], axis=1
    )
    t = np.stack(
        [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)], axis=1
    )
    return u, v, t


def _quat_from_uvt(u, v, t):
    """Inverse of `_frame_uvt`: the xyzw quaternion whose columns are (u, v, t).

    Shepperd's method - the branch with the largest divisor is taken so that
    the square root never lands near zero.
    """
    m = np.stack([u, v, t], axis=2)  # m[:, row, column]
    m00, m11, m22 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    trace = m00 + m11 + m22
    q = np.zeros((len(m), 4), dtype=np.float64)

    c0 = trace > 0
    c1 = ~c0 & (m00 >= m11) & (m00 >= m22)
    c2 = ~c0 & ~c1 & (m11 >= m22)
    c3 = ~(c0 | c1 | c2)

    def root(sel, expr):
        return 2.0 * np.sqrt(np.maximum(expr[sel], 1e-30))

    s = root(c0, trace + 1.0)
    q[c0] = np.stack(
        [
            (m[c0, 2, 1] - m[c0, 1, 2]) / s,
            (m[c0, 0, 2] - m[c0, 2, 0]) / s,
            (m[c0, 1, 0] - m[c0, 0, 1]) / s,
            0.25 * s,
        ],
        axis=1,
    )
    s = root(c1, 1.0 + m00 - m11 - m22)
    q[c1] = np.stack(
        [
            0.25 * s,
            (m[c1, 0, 1] + m[c1, 1, 0]) / s,
            (m[c1, 0, 2] + m[c1, 2, 0]) / s,
            (m[c1, 2, 1] - m[c1, 1, 2]) / s,
        ],
        axis=1,
    )
    s = root(c2, 1.0 + m11 - m00 - m22)
    q[c2] = np.stack(
        [
            (m[c2, 0, 1] + m[c2, 1, 0]) / s,
            0.25 * s,
            (m[c2, 1, 2] + m[c2, 2, 1]) / s,
            (m[c2, 0, 2] - m[c2, 2, 0]) / s,
        ],
        axis=1,
    )
    s = root(c3, 1.0 + m22 - m00 - m11)
    q[c3] = np.stack(
        [
            (m[c3, 0, 2] + m[c3, 2, 0]) / s,
            (m[c3, 1, 2] + m[c3, 2, 1]) / s,
            0.25 * s,
            (m[c3, 1, 0] - m[c3, 0, 1]) / s,
        ],
        axis=1,
    )
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _transport(u, t_from, t_to):
    """Carry `u` from one tangent to another by the minimal rotation.

    Rodrigues about `t_from x t_to`. Where the two tangents are (anti)parallel
    there is no such rotation - and a 180 degree turn in the centreline has no
    meaningful transport either - so `u` is passed through unchanged.
    """
    axis = np.cross(t_from, t_to)
    s = np.linalg.norm(axis, axis=1)
    ang = np.arctan2(s, (t_from * t_to).sum(1))
    k = axis / np.maximum(s, 1e-12)[:, None]
    ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
    out = u * ca + np.cross(k, u) * sa + k * ((k * u).sum(1))[:, None] * (1 - ca)
    return np.where((s > 1e-12)[:, None], out, u)


def _orthonormal(u, t):
    """Project `u` into the plane perpendicular to `t` and normalize.

    Falls back to an arbitrary perpendicular where `u` is (anti)parallel to
    `t`, which only happens if the incoming frame was already degenerate.
    """
    u = u - t * (u * t).sum(1)[:, None]
    n = np.linalg.norm(u, axis=1)
    alt = np.cross(
        t, np.where(np.abs(t[:, :1]) < 0.9, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    )
    u = np.where((n > 1e-9)[:, None], u, alt)
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def _chord_tangents(pos, edges, t0, deg):
    """Re-derive the per-node tangent from the chords actually being swept.

    At a degree-2 node this is the bisector of its two chords, which is the
    orientation that puts the ring symmetrically between the quads meeting at
    it; at a tip it is the single chord. Branch points keep their stored
    tangent - three or more arms leave them and no single direction means
    anything there.

    Each half-edge contributes its chord direction flipped to agree with the
    stored tangent's sense, so the result does not depend on which way round
    the edge list happens to pair its nodes.
    """
    i, j = edges[:, 0], edges[:, 1]
    d = pos[j] - pos[i]
    n = np.linalg.norm(d, axis=1, keepdims=True)
    d = np.divide(d, n, out=np.zeros_like(d), where=n > 1e-12)

    acc = np.zeros_like(pos)
    for node in (i, j):
        # `np.sign` would drop the contribution of an exactly perpendicular
        # chord, so resolve the tie towards +1 instead
        sense = np.where((d * t0[node]).sum(1) >= 0, 1.0, -1.0)
        np.add.at(acc, node, d * sense[:, None])

    t = np.where((deg[:, None] >= 3) | (deg[:, None] == 0), t0, acc)
    n = np.linalg.norm(t, axis=1)
    # A hairpin - two chords that double back on each other - cancels out
    t = np.where((n > 1e-9)[:, None], t, t0)
    t = t / np.linalg.norm(t, axis=1, keepdims=True)
    # Preserve the stored sense: the shader reads the tangent's sign to orient
    # each quad, and flipping it would flip the frame's handedness with it
    return t * np.where((t * t0).sum(1) >= 0, 1.0, -1.0)[:, None]


def rotate_profile(coefs, psi):
    """Rotate each node's profile by `psi` (M,) radians about its own tangent.

    Exact, and the reason the frame can be rebuilt without losing any shape:
    substituting theta -> theta + psi into

        r(theta) = a0 + sum_k [a_k cos(k theta) + b_k sin(k theta)]

    and re-collecting gives

        a'_k = a_k cos(k psi) + b_k sin(k psi)
        b'_k = b_k cos(k psi) - a_k sin(k psi)

    i.e. each harmonic rotates by `k psi`, and a0 is untouched.

    Parameters
    ----------
    coefs :     (M, 8 + 2K) float array
                Modified in place.
    psi :       (M,) array
                Angle from the old `u` to the new one, in the old frame.

    """
    k = (coefs.shape[1] - HEADER_FLOATS) // 2
    if not k:
        return coefs
    a = coefs[:, HEADER_FLOATS : HEADER_FLOATS + k]
    b = coefs[:, HEADER_FLOATS + k : HEADER_FLOATS + 2 * k]
    kpsi = np.arange(1, k + 1)[None, :] * psi[:, None]
    c, s = np.cos(kpsi), np.sin(kpsi)
    a[:], b[:] = a * c + b * s, b * c - a * s
    return coefs


def align_frames(coefs, edges, retangent=True):
    """Rebuild the per-node frame so that it is consistent along `edges`.

    Two things about the frames as they arrive show up as jagged geometry, and
    this fixes both:

    - The stored theta = 0 reference is rotation-minimizing only *within* an
      unbranched run. The chain restarts at every branch point, so an edge
      spanning a junction connects theta = 0 on one ring to an unrelated angle
      on the other and the quad between them is twisted. The runs also drift
      against a true rotation-minimizing frame in the interior. One chain is
      propagated over the whole tree here instead.
    - The stored tangent is the centreline's *local* direction. For a skeleton
      traced from voxels that sits well off the chord the surface is actually
      swept along, which tilts the ring relative to its own sweep and is what
      makes consecutive rings punch through each other; `decimate_edges` makes
      it worse, not better, because the chord then spans more of the jitter.
      `retangent` re-derives the tangent from the edge list.

    The profile is rotated to follow the new frame, which is exact (see
    `rotate_profile`), so no shape is lost to the realignment itself.
    Re-tangenting does move the surface - the profile was measured in the old
    cross-section plane and is applied in a slightly tilted one - but that
    error is second order in the tilt while the punch-through it removes is
    first order.

    Parameters
    ----------
    coefs :     (M, 8 + 2K) array
                Per-node coefficients. Not modified; a new array is returned.
    edges :     (E, 2) array
                The edge list the surface will actually be swept along - i.e.
                after any decimation, not before.
    retangent : bool
                Whether to re-derive the tangents as well as the phase. Phase
                alignment alone leaves the geometry untouched.

    Returns
    -------
    (M, 8 + 2K) float32 array

    """
    coefs = np.array(coefs, dtype=np.float32, copy=True)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    n_nodes = len(coefs)
    if not len(edges) or not n_nodes:
        return coefs

    pos = coefs[:, :3].astype(np.float64)
    u0, v0, t0 = _frame_uvt(coefs[:, 3:7].astype(np.float64))
    deg = np.bincount(edges.ravel(), minlength=n_nodes)

    t = _chord_tangents(pos, edges, t0, deg) if retangent else t0
    # The stored reference, carried onto the new tangent. Where the tangent is
    # unchanged this is exactly `u0`, so `retangent=False` starts from the
    # frames as given and only corrects their phase below.
    u_loc = _orthonormal(_transport(u0, t0, t), t)

    # Per *ordered* pair of adjacent nodes, the twist between the transport of
    # one node's reference and the next node's own. Computed for every run's
    # pairs at once; only the accumulation below is sequential.
    runs = split_runs(edges, n_nodes)
    pairs_a = np.concatenate([r[:-1] for r in runs]) if runs else np.zeros(0, int)
    pairs_b = np.concatenate([r[1:] for r in runs]) if runs else np.zeros(0, int)
    p = _transport(u_loc[pairs_a], t[pairs_a], t[pairs_b])
    delta = np.arctan2(
        (u_loc[pairs_b] * np.cross(t[pairs_b], p)).sum(1),
        (u_loc[pairs_b] * p).sum(1),
    )

    # phi is the correction that turns `u_loc` into the transported reference.
    # Along an edge a -> b it obeys phi_b = phi_a - delta_ab, so within a run it
    # is a cumulative sum - and only one scalar per run has to cross a branch
    # point, which is a walk over the ~1e2 runs rather than the ~1e4 nodes.
    offsets = np.cumsum([0] + [len(r) - 1 for r in runs])
    at_node = {}
    for ri, run in enumerate(runs):
        at_node.setdefault(run[0], []).append(ri)
        at_node.setdefault(run[-1], []).append(ri)

    phi = np.zeros(n_nodes)
    assigned = np.zeros(n_nodes, dtype=bool)
    done = np.zeros(len(runs), dtype=bool)

    for seed in range(len(runs)):
        if done[seed]:
            continue
        if not assigned[runs[seed][0]]:
            assigned[runs[seed][0]] = True
        queue = deque([seed])
        while queue:
            ri = queue.popleft()
            if done[ri]:
                continue
            run = runs[ri]
            d = delta[offsets[ri] : offsets[ri + 1]]
            if assigned[run[0]]:
                nodes, step = run, d
            elif assigned[run[-1]]:
                # Traversed from the far end: reversed, and each step is the
                # inverse rotation
                nodes, step = run[::-1], -d[::-1]
            else:
                continue  # unreachable until a neighbouring run assigns an end
            done[ri] = True
            walked = phi[nodes[0]] - np.cumsum(step)
            # A closed loop comes back to its own start; its accumulated twist
            # need not be zero, so leave the start alone and take the seam
            tail = nodes[1:]
            fresh = ~assigned[tail]
            phi[np.asarray(tail)[fresh]] = walked[fresh]
            assigned[tail] = True
            for end in (nodes[0], nodes[-1]):
                for rj in at_node.get(end, ()):
                    if not done[rj]:
                        queue.append(rj)

    c, s = np.cos(phi)[:, None], np.sin(phi)[:, None]
    u = u_loc * c + np.cross(t, u_loc) * s  # u_loc _|_ t, so no third term

    # Nodes no edge reaches are never drawn; leave their frames as they were
    idle = deg == 0
    u[idle], t[idle] = u0[idle], t0[idle]

    # The angle the profile has to follow, measured in the frame it is written
    # in - i.e. against the *old* u and v
    psi = np.arctan2((u * v0).sum(1), (u * u0).sum(1))
    rotate_profile(coefs, psi)
    coefs[:, 3:7] = _quat_from_uvt(u, np.cross(t, u), t).astype(np.float32)
    return coefs


class TubeVisual(gfx.WorldObject):
    """A tube surface defined by per-node radial profile coefficients.

    Parameters
    ----------
    coefs :     (M, 8 + 2K) array
                Per-node coefficients, as produced by
                `sparsecubes.TubeProfile.to_gpu_buffer()` in its Cartesian
                form: position (3), frame quaternion xyzw (4), mean radius
                a0 (1), then K cosine and K sine coefficients. Positions are
                expected in physical units (i.e. with any voxel spacing
                already applied).
    edges :     (E, 2) array
                Index pairs into the nodes. A skeleton is a tree, so the
                surface is swept per edge - consecutive node indices are not
                assumed to be connected.
    material :  TubeMaterial
                The material defining the appearance of the tube.
    colors :    (M, 4) array, optional
                Per-node RGBA colors. Requires the material's `color_mode`
                to be "vertex".

    """

    def __init__(self, coefs, edges, material, colors=None, **kwargs):
        # Initialize the base object first: it claims the process-global id
        # that `WorldObject.__del__` needs, so that a validation error below
        # does not leave a half-built object that blows up on garbage collection
        super().__init__(**kwargs)

        coefs, k = _split_coefficients(coefs)

        edges = np.ascontiguousarray(edges, dtype=np.int32).reshape(-1, 2)
        n_nodes = len(coefs)
        n_edges = len(edges)
        if n_edges and (edges.min() < 0 or edges.max() >= n_nodes):
            raise ValueError(
                f"Edge indices must be in [0, {n_nodes}), got "
                f"[{edges.min()}, {edges.max()}]"
            )

        # The corner positions let pygfx derive the bounding box from the
        # geometry. r(theta) is bounded from above by a0 + sum_k m_k, with
        # m_k = hypot(a_k, b_k) the (frame-independent) harmonic magnitude.
        if n_nodes:
            a = coefs[:, HEADER_FLOATS : HEADER_FLOATS + k]
            b = coefs[:, HEADER_FLOATS + k :]
            reach = float((coefs[:, 7] + np.hypot(a, b).sum(axis=1)).max())
            positions = coefs[:, :3]
            corners = np.array(
                [positions.min(axis=0) - reach, positions.max(axis=0) + reach],
                dtype=np.float32,
            )
        else:
            corners = np.zeros((2, 3), dtype=np.float32)

        buffers = {}
        if colors is not None:
            colors = np.ascontiguousarray(colors, dtype=np.float32).reshape(-1, 4)
            if len(colors) != n_nodes:
                raise ValueError(f"Got {len(colors)} colors for {n_nodes} nodes.")
            buffers["colors"] = _at_least_one(colors, (1, 4), np.float32)

        self.geometry = gfx.Geometry(
            positions=corners,
            # Storage buffers must be flat: pygfx infers the buffer format from
            # the last dimension and supports only 1-4 channels, so an (M, 16)
            # array would raise "Unexpected vertex format '16xf4'" at pipeline
            # build time
            coefs=_at_least_one(coefs, (1, HEADER_FLOATS + 2 * k), np.float32).ravel(),
            tube_edges=_at_least_one(edges, (1, 2), np.int32),
            **buffers,
        )
        self.material = material

        self.k = int(k)
        self.stride = HEADER_FLOATS + 2 * self.k
        self.n_nodes = int(n_nodes)
        self.n_edges = int(n_edges)


def _at_least_one(arr, shape, dtype):
    """Substitute a dummy element for an empty array (zero-sized buffers are
    not bindable). The draw count is derived from `n_edges`, not from the
    buffer sizes, so the dummy is never touched."""
    if arr.size:
        return arr
    return np.zeros(shape, dtype=dtype)


class TubeMaterial(gfx.MeshPhongMaterial):
    """Material for rendering a TubeVisual.

    In addition to the properties of `pygfx.MeshPhongMaterial`, this material
    has the two angular level-of-detail knobs:

     - `n_theta`: number of angular samples around the tube (>= 3). This is
       both a uniform and the draw-call size; changing it neither re-uploads
       nor reallocates the coefficients.
     - `k_max`: number of harmonics evaluated for the surface position.
       Clamped in the shader to the number actually present in the buffer;
       0 renders circular tubes of radius a0.
     - `k_normal`: number of harmonics evaluated for the *normal*, clamped to
       `k_max`. Deliberately separate and usually much lower - dr/dtheta
       weights harmonic k by k, so the harmonics that still sharpen the
       silhouette already make the shading look like sandpaper (and dark
       where the normal tilts past the view direction). 1 is the default,
       0 the smooth-tube floor.

    `color_mode` may be "uniform" (the material's `color`) or "vertex"
    (per-node RGBA from `geometry.colors`).

    """

    uniform_type = dict(
        gfx.MeshPhongMaterial.uniform_type,
        n_theta="i4",
        k_max="i4",
        k_normal="i4",
    )

    def __init__(self, n_theta=32, k_max=8, k_normal=1, **kwargs):
        super().__init__(**kwargs)
        self.n_theta = n_theta
        self.k_max = k_max
        self.k_normal = k_normal

    @property
    def n_theta(self):
        """Number of angular samples around the tube."""
        return self._store.n_theta

    @n_theta.setter
    def n_theta(self, value):
        value = int(value)
        if value < 3:
            raise ValueError(f"n_theta must be >= 3, got {value}")
        self.uniform_buffer.data["n_theta"] = value
        self.uniform_buffer.update_full()
        # Also kept on the (tracked) store: the shader reads it from the
        # uniform, but `get_render_info` needs to see the change to resize
        # the draw call, and uniform buffer writes are not tracked
        self._store.n_theta = value

    @property
    def k_max(self):
        """Number of harmonics evaluated for the position; 0 gives circular tubes."""
        return int(self.uniform_buffer.data["k_max"])

    @k_max.setter
    def k_max(self, value):
        value = int(value)
        if value < 0:
            raise ValueError(f"k_max must be >= 0, got {value}")
        self.uniform_buffer.data["k_max"] = value
        self.uniform_buffer.update_full()

    @property
    def k_normal(self):
        """Number of harmonics evaluated for the normal; clamped to `k_max`."""
        return int(self.uniform_buffer.data["k_normal"])

    @k_normal.setter
    def k_normal(self, value):
        value = int(value)
        if value < 0:
            raise ValueError(f"k_normal must be >= 0, got {value}")
        self.uniform_buffer.data["k_normal"] = value
        self.uniform_buffer.update_full()


@register_wgpu_render_function(TubeVisual, TubeMaterial)
class TubeShader(BaseShader):
    type = "render"

    def __init__(self, wobject, **kwargs):
        super().__init__(wobject, **kwargs)

        self["stride"] = wobject.stride
        self["k_buf"] = wobject.k

    def get_bindings(self, wobject, shared, scene):
        geometry = wobject.geometry
        material = wobject.material

        color_mode = material.color_mode
        colors = getattr(geometry, "colors", None)
        if color_mode == "auto":
            color_mode = "vertex" if colors is not None else "uniform"
        elif color_mode == "vertex" and colors is None:
            raise ValueError(
                "color_mode='vertex' requires the geometry to have a `colors` "
                "buffer (one RGBA color per node)."
            )
        elif color_mode not in ("uniform", "vertex"):
            raise ValueError(
                f"TubeMaterial supports color_mode 'uniform' or 'vertex', got "
                f"{color_mode!r}"
            )
        self["color_mode"] = color_mode

        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
            Binding("s_coefs", "buffer/read_only_storage", geometry.coefs, "VERTEX"),
            Binding(
                "s_edges", "buffer/read_only_storage", geometry.tube_edges, "VERTEX"
            ),
        ]
        if color_mode == "vertex":
            bindings.append(
                Binding("s_colors", "buffer/read_only_storage", colors, "VERTEX")
            )

        bindings = {i: b for i, b in enumerate(bindings)}
        self.define_bindings(0, bindings)

        return {
            0: bindings,
        }

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            # The quad winding follows the frame handedness, which we do not
            # control; the analytic normal and the two-sided lighting make
            # either winding render correctly, so nothing is culled.
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        # Two triangles per (edge, angular sector) cell. Reading `n_theta` off
        # the material's store (rather than its uniform) registers it with
        # pygfx's change tracker, so that changing it resizes the draw call.
        n_theta = wobject.material.n_theta
        return {
            "indices": (6 * wobject.n_edges * n_theta, 1),
        }

    def get_code(self):
        return load_wgsl("tube.wgsl", "octarine.shaders.wgsl")
