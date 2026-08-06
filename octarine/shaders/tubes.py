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
"""

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


def decimate_edges(edges, n_nodes, step):
    """Thin an edge list `step`-fold along its unbranched runs.

    Nodes of degree != 2 - branch points and tips - are always kept, so the
    topology is untouched and no arm can go missing; only the interior of each
    run is thinned. Striding the node array instead would silently reconnect
    unrelated branches.

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

    # CSR-style adjacency: for each node, the neighbours and the edges reaching
    # them. `edges[:, ::-1].ravel()` is the *other* endpoint of each half-edge.
    deg = np.bincount(edges.ravel(), minlength=n_nodes)
    order = np.argsort(edges.ravel(), kind="stable")
    nbr = edges[:, ::-1].ravel()[order]
    via = np.repeat(np.arange(len(edges)), 2)[order]
    indptr = np.concatenate([[0], np.cumsum(deg)])

    essential = deg != 2
    seen = np.zeros(len(edges), dtype=bool)
    kept = []

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

    def emit(start, interior, end):
        chain = [start, *interior[step - 1 :: step], end]
        for a, b in zip(chain[:-1], chain[1:]):
            if a != b:
                kept.append((a, b))

    for a in np.flatnonzero(essential):
        for m in range(indptr[a], indptr[a + 1]):
            if not seen[via[m]]:
                emit(a, *walk(a, via[m], nbr[m]))

    # Runs with no essential node at all (a closed loop) are not reached above;
    # break each one open at an arbitrary node.
    for e in np.flatnonzero(~seen):
        if seen[e]:
            continue
        a = int(edges[e, 0])
        emit(a, *walk(a, e, int(edges[e, 1])))

    if not kept:
        return np.zeros((0, 2), dtype=np.int32)
    return np.ascontiguousarray(kept, dtype=np.int32)


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
