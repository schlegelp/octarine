"""Tests for the parametric tube shader.

The fixtures are synthetic profiles with analytically known silhouettes, so
the assertions do not depend on the producer side (sparse-cubes).
"""

import pytest
import octarine as oc

import numpy as np

# The frame quaternion is the rotation whose columns are (u, v, t). This one
# gives u = +y, v = +z, t = +x, i.e. a tube running along +x whose theta = 0
# direction is +y. That makes the two views below read directly:
#   "XY" (camera looks along +z) measures the extent along u
#   "XZ" (camera looks along +y) measures the extent along v
QUAT_X = (0.5, 0.5, 0.5, 0.5)

# Camera window (in world units) used by `render` below; the fixtures are sized
# to fit comfortably inside it
CAM_WIDTH = 12.0


def straight_tube(a0=2.0, k=2, length=10.0, n=11, a0_end=None, **harmonics):
    """A straight tube along +x. Harmonics are given as e.g. `a2=0.5`.

    `a0_end` tapers the mean radius linearly from `a0` to `a0_end`.
    """
    coefs = np.zeros((n, 8 + 2 * k), dtype=np.float32)
    coefs[:, 0] = np.linspace(0, length, n)
    coefs[:, 3:7] = QUAT_X
    coefs[:, 7] = a0 if a0_end is None else np.linspace(a0, a0_end, n)
    for name, value in harmonics.items():
        which, idx = name[0], int(name[1:])
        assert which in "ab" and 1 <= idx <= k
        coefs[:, 8 + (idx - 1) + (0 if which == "a" else k)] = value
    edges = np.column_stack([np.arange(n - 1), np.arange(1, n)]).astype(np.int32)
    return coefs, edges


def render(coefs, edges, view="XY", size=256, width=CAM_WIDTH, **kwargs):
    """Render at a fixed world-units-per-pixel scale, return the RGBA image."""
    v = oc.Viewer(offscreen=True, size=(size, size))
    v.add_tubes(coefs, edges=edges, **kwargs)
    img = shoot(v, view=view, size=size, width=width)
    v.close()
    return img


def shoot(v, view="XY", size=256, width=CAM_WIDTH):
    v.set_view(view)
    # Override the auto-fit so that pixels map to world units predictably
    v.camera.width = width
    v.camera.height = width
    return np.asarray(v.screenshot(filename=None, size=(size, size)))


def srgb2linear(u8):
    c = np.asarray(u8) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lit_luminance(coefs, edges, width=16.0, **kwargs):
    """Linear luminance of every lit fragment, with the specular term off.

    `lighting_phong` reduces to `albeido * (ambient + diffuse * lambert)` once
    `shininess` is 0, so in linear space the image is an affine function of
    lambert = |dot(view, normal)| - which lets the tests below read the normal
    off the render without depending on the light constants.
    """
    v = oc.Viewer(offscreen=True, size=(256, 256))
    v.add_tubes(coefs, edges=edges, color=(0.5, 0.5, 0.5), **kwargs)
    (vis,) = [o for objs in v.objects.values() for o in objs]
    vis.material.shininess = 0.0
    img = shoot(v, width=width)
    v.close()
    mask = img[..., 3] > 0
    return srgb2linear(img[..., :3].max(axis=-1).astype(float))[mask], mask


def extent(img, width=CAM_WIDTH):
    """Extent (in world units) of the covered rows.

    The fixtures run along the image columns, so this is the tube's silhouette
    across its axis. Measured as a fraction of the image so that it does not
    depend on the screenshot coming back at 1x or 2x.
    """
    mask = img[..., 3] > 0
    rows = np.where(mask.any(axis=1))[0]
    if not len(rows):
        return 0.0
    return (rows[-1] - rows[0] + 1) / img.shape[0] * width


def silhouette(a0, a, b, axis="u", n=100_000):
    """CPU mirror of the shader's profile loop: the cross-section's extent.

    Evaluates r(theta) = a0 + sum_k [a_k cos(k theta) + b_k sin(k theta)] and
    projects onto `u` (theta = 0) or `v` (theta = pi/2). Note that this is
    max - min rather than 2 * max: odd harmonics offset the cross-section from
    its skeleton point, so it is not symmetric in general.
    """
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    k = np.arange(1, len(a) + 1)[:, None]
    r = a0 + (np.asarray(a)[:, None] * np.cos(k * theta)).sum(0)
    r += (np.asarray(b)[:, None] * np.sin(k * theta)).sum(0)
    proj = r * (np.cos(theta) if axis == "u" else np.sin(theta))
    return proj.max() - proj.min()


# One pixel of slack, plus a little, on either edge of the silhouette
def tol(size=256, width=CAM_WIDTH):
    return 3 * width / size


def test_straight_circular_tube():
    """a0 alone must give a circular tube: same width from any side."""
    coefs, edges = straight_tube(a0=2.0)

    xy = extent(render(coefs, edges, view="XY"))
    xz = extent(render(coefs, edges, view="XZ"))

    assert xy == pytest.approx(4.0, abs=tol())
    assert xz == pytest.approx(4.0, abs=tol())


def test_cosine_harmonic_is_aligned_with_u():
    """`a_2` alone makes an ellipse whose long axis is at theta = 0, i.e. `u`.

    Catches the harmonic loop and the `u` half of the frame unpack: with `u`
    and `v` swapped the two extents trade places.
    """
    a0, a2 = 2.0, 0.5
    coefs, edges = straight_tube(a0=a0, a2=a2)

    along_u = extent(render(coefs, edges, view="XY"))
    along_v = extent(render(coefs, edges, view="XZ"))

    assert along_u > along_v
    assert along_u == pytest.approx(2 * (a0 + a2), abs=tol())
    assert along_v == pytest.approx(
        silhouette(a0, [0, a2], [0, 0], axis="v"), abs=tol()
    )


def test_sine_harmonic_rotates_the_ellipse():
    """`b_2` alone puts the long axis at 45 degrees, so `u` and `v` tie.

    Catches the `b_k` half of the coefficient layout: reading the sine
    coefficients from the wrong offset (or not at all) leaves a circle.
    """
    a0, b2 = 2.0, 0.5
    coefs, edges = straight_tube(a0=a0, b2=b2)

    along_u = extent(render(coefs, edges, view="XY"))
    along_v = extent(render(coefs, edges, view="XZ"))

    assert along_u == pytest.approx(along_v, abs=tol())
    # ... and it is not simply a circle of radius a0
    assert along_u > 2 * a0 + tol()
    assert along_u == pytest.approx(
        silhouette(a0, [0, 0], [0, b2], axis="u"), abs=tol()
    )


def test_profile_matches_cpu_reference():
    """The full harmonic series must agree with the same recurrence in numpy."""
    a = [0.30, -0.20, 0.10, 0.05]
    b = [-0.15, 0.25, -0.05, 0.10]
    a0 = 2.0
    coefs, edges = straight_tube(
        a0=a0, k=4, **{f"a{i + 1}": v for i, v in enumerate(a)}
    )
    coefs[:, 12:16] = b

    assert extent(render(coefs, edges, view="XY")) == pytest.approx(
        silhouette(a0, a, b, axis="u"), abs=tol()
    )
    assert extent(render(coefs, edges, view="XZ")) == pytest.approx(
        silhouette(a0, a, b, axis="v"), abs=tol()
    )


def test_normals_are_outward():
    """Lit face-on, dark at the silhouette.

    Getting the normal inverted is invisible in a wireframe and is the most
    likely single bug: it leaves the whole tube at flat ambient brightness.
    """
    coefs, edges = straight_tube(a0=2.0)
    img = render(coefs, edges, view="XY", color="w")

    lum = img[..., :3].max(axis=-1).astype(float)
    mask = img[..., 3] > 0
    rows = np.where(mask.any(axis=1))[0]
    mid = (rows[0] + rows[-1]) // 2

    def band(sl):
        return lum[sl][mask[sl]]

    core = band(np.s_[mid - 2 : mid + 3, :])
    rim = np.concatenate(
        [band(np.s_[rows[0] : rows[0] + 2, :]), band(np.s_[rows[-1] - 1 :, :])]
    )

    assert core.mean() > 200
    assert rim.mean() < 0.7 * core.mean()


def test_edge_direction_does_not_matter():
    """An edge list may pair its nodes either way round.

    Swapping a pair reverses the traversal of that quad, and hence its winding
    and `front_facing` - so without orienting off the stored tangent, a
    mixed-orientation edge list renders half its tubes inside-out.
    """
    coefs, edges = straight_tube(a0=1.0, a0_end=3.0, n=21, a2=0.3)
    forward = render(coefs, edges, width=16.0, color="w").astype(int)

    rng = np.random.default_rng(0)
    for _ in range(3):
        flipped = edges.copy()
        reverse = rng.random(len(edges)) < 0.5
        flipped[reverse] = flipped[reverse][:, ::-1]
        got = render(coefs, flipped, width=16.0, color="w").astype(int)
        # Allow a rounding step: the arithmetic is not associative
        assert np.abs(got - forward).max() <= 2

    assert (
        np.abs(
            render(coefs, edges[:, ::-1].copy(), width=16.0, color="w").astype(int)
            - forward
        ).max()
        <= 2
    )


def test_taper_tilts_the_normal_by_the_taper_angle():
    """On a cone the normal must tilt off radial by exactly the taper angle.

    This is the fixture where the shader's one-sided axial difference and the
    CPU mirror's centred one agree exactly - a cone differenced either way is
    the same cone - so it can be asserted to tolerance. It is also the test
    that fails if anyone "simplifies" the normal to use the stored tangent:
    that would leave the normal exactly radial and every cone reading 1.0.
    """
    # A cylinder calibrates the scale: its brightest fragment is lambert = 1
    # (normal straight at the camera) and its darkest is lambert = 0 (normal
    # at the silhouette), whatever the light constants happen to be
    cylinder, _ = lit_luminance(*straight_tube(a0=2.0))
    floor, ceiling = cylinder.min(), cylinder.max()

    def peak_lambert(coefs, edges):
        lum, _ = lit_luminance(coefs, edges)
        return (lum.max() - floor) / (ceiling - floor)

    assert peak_lambert(*straight_tube(a0=2.0)) == pytest.approx(1.0, abs=0.04)

    for r0, r1, length in [(1.0, 3.0, 10.0), (1.0, 5.0, 10.0), (1.0, 6.0, 5.0)]:
        slope = (r1 - r0) / length
        coefs, edges = straight_tube(a0=r0, a0_end=r1, length=length, n=41)
        # cos(taper angle); the normal is (-slope, cos(theta), sin(theta))
        # normalized, so its peak alignment with the view is 1 / hypot(1, slope)
        assert peak_lambert(coefs, edges) == pytest.approx(
            1 / np.hypot(1, slope), abs=0.04
        ), f"taper slope {slope}"


def test_k_normal_is_separate_from_k_max():
    """The normal is truncated harder than the position, and separately.

    dr/dtheta weights harmonic k by k, so the harmonics that still sharpen the
    silhouette already make the shading noisy. Rendering the same fixture at
    two `k_normal` values must therefore leave the silhouette pixel-identical
    and the shading different - assert it, or the two uniforms will get quietly
    collapsed back into one.
    """
    coefs, edges = straight_tube(a0=2.0, k=4, a2=0.35, a3=-0.25, a4=0.2)
    coefs[:, 12:16] = [0.2, -0.3, 0.15, -0.2]  # b_1 .. b_4

    smooth, mask_smooth = lit_luminance(coefs, edges, k=4, k_normal=1)
    rough, mask_rough = lit_luminance(coefs, edges, k=4, k_normal=4)

    assert np.array_equal(mask_smooth, mask_rough)
    assert not np.allclose(smooth, rough)

    # k_normal can only truncate, never add: above k_max it must be a no-op
    flat, _ = lit_luminance(coefs, edges, k=0, k_normal=0)
    clamped, _ = lit_luminance(coefs, edges, k=0, k_normal=4)
    assert np.array_equal(flat, clamped)


def test_n_theta_is_a_uniform():
    """Angular LOD must be a re-draw, not a re-upload.

    This is the property the whole design exists for, so it is asserted
    explicitly: changing `n_theta` on a live visual changes the silhouette
    (i.e. it reaches the shader *and* resizes the draw call) while leaving the
    coefficient buffer object untouched.
    """
    coefs, edges = straight_tube(a0=2.0)

    v = oc.Viewer(offscreen=True, size=(256, 256))
    v.add_tubes(coefs, edges=edges, n_theta=64)
    (vis,) = [o for objs in v.objects.values() for o in objs]
    buf_before = vis.geometry.coefs

    smooth = extent(shoot(v))
    assert smooth == pytest.approx(4.0, abs=tol())

    # A 3-gon cross-section inscribed in the circle, with a vertex at theta = 0:
    # the extent along u collapses to 1.5 * R
    vis.material.n_theta = 3
    coarse = extent(shoot(v))
    assert coarse == pytest.approx(3.0, abs=tol())

    # No upload, no reallocation - the coefficients never moved
    assert vis.geometry.coefs is buf_before
    v.close()


def test_n_theta_silhouettes_converge():
    """Coarse and fine angular sampling must agree on the silhouette."""
    coefs, edges = straight_tube(a0=2.0)

    coarse = render(coefs, edges, n_theta=8)[..., 3] > 0
    fine = render(coefs, edges, n_theta=64)[..., 3] > 0

    disagree = np.logical_xor(coarse, fine).sum()
    assert disagree < 0.02 * fine.sum()


def test_k_max_truncation():
    """`k_max=0` must render a circular tube of radius a0, harmonics or not."""
    a0, a2 = 2.0, 0.5
    coefs, edges = straight_tube(a0=a0, a2=a2)

    def ext(**kw):
        return extent(render(coefs, edges, **kw))

    assert ext(k=0) == pytest.approx(2 * a0, abs=tol())
    assert ext(k=2) == pytest.approx(2 * (a0 + a2), abs=tol())
    # k_max is clamped to what the buffer actually holds
    assert ext(k=99) == pytest.approx(2 * (a0 + a2), abs=tol())

    # ... and it can be changed on a live visual
    v = oc.Viewer(offscreen=True, size=(256, 256))
    v.add_tubes(coefs, edges=edges, k=2)
    (vis,) = [o for objs in v.objects.values() for o in objs]
    assert extent(shoot(v)) == pytest.approx(2 * (a0 + a2), abs=tol())
    vis.material.k_max = 0
    assert extent(shoot(v)) == pytest.approx(2 * a0, abs=tol())
    v.close()


def y_skeleton(k=2, radius=1.0):
    """A Y with deliberately scrambled node indices."""
    pts = np.array(
        [[0, 0, 0], [4, 0, 0], [8, 0, 0], [12, 4, 0], [12, -4, 0]], dtype=np.float32
    )
    order = np.array([3, 0, 4, 2, 1])
    inv = np.argsort(order)

    coefs = np.zeros((len(pts), 8 + 2 * k), dtype=np.float32)
    coefs[:, :3] = pts[order]
    coefs[:, 3:7] = QUAT_X
    coefs[:, 7] = radius
    edges = inv[np.array([[0, 1], [1, 2], [2, 3], [2, 4]])].astype(np.int32)
    return coefs, edges


def arms_at(mask, frac):
    column = mask[:, int(mask.shape[1] * frac)].astype(int)
    return int((np.diff(np.concatenate([[0], column, [0]])) == 1).sum())


def test_branch_topology():
    """A Y-shaped skeleton must render three arms.

    The node order is deliberately scrambled, so an implementation that built
    geometry between consecutive node *indices* (rather than along edges)
    would stitch the arms together and fail this.
    """
    coefs, edges = y_skeleton()
    mask = render(coefs, edges, view="XY", width=24.0)[..., 3] > 0

    assert arms_at(mask, 0.3) == 1  # trunk
    assert arms_at(mask, 0.5) == 1  # trunk
    assert arms_at(mask, 0.7) == 2  # past the bifurcation


def test_degenerate_inputs():
    """Empty / single node / single edge must not crash or divide by zero."""
    k = 2
    cases = {
        "empty": (np.zeros((0, 8 + 2 * k), np.float32), np.zeros((0, 2), np.int32)),
        "single node": (
            np.zeros((1, 8 + 2 * k), np.float32),
            np.zeros((0, 2), np.int32),
        ),
        "single edge": (straight_tube(n=2)[0], np.array([[0, 1]], np.int32)),
    }
    for coefs, edges in cases.values():
        v = oc.Viewer(offscreen=True, size=(128, 128))
        v.add_tubes(coefs, edges=edges)
        v.canvas.draw()  # force the shader to actually compile/render
        v.close()


def test_per_node_colors():
    """An (M, 4) color array is rendered per node."""
    coefs, edges = straight_tube(a0=2.0)
    colors = np.zeros((len(coefs), 4), dtype=np.float32)
    colors[:, 1] = 1.0  # green
    colors[:, 3] = 1.0

    img = render(coefs, edges, color=colors)
    lit = img[..., :3].reshape(-1, 3)[(img[..., 3] > 0).ravel()].mean(axis=0)

    assert lit[1] > 3 * max(lit[0], lit[2])

    with pytest.raises(ValueError, match="one color per node"):
        render(coefs, edges, color=colors[:-1])


def test_decimate_edges():
    """Axial LOD must thin runs exactly, and never touch the topology."""
    from octarine.shaders import decimate_edges

    # A plain chain thins by exactly the step, endpoints kept
    chain = np.column_stack([np.arange(16), np.arange(1, 17)])
    for step, expect in [(1, 16), (2, 8), (4, 4), (8, 2)]:
        got = decimate_edges(chain, 17, step)
        assert len(got) == expect
        kept = sorted(set(got.ravel().tolist()))
        assert kept[0] == 0 and kept[-1] == 16

    # A Y keeps its branch point and all three tips at every level, so no arm
    # can go missing however hard it is decimated
    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 9],
            [6, 10],
            [10, 11],
            [11, 12],
        ]
    )
    order = np.random.default_rng(0).permutation(13)
    scrambled = order[edges]
    for step in (1, 2, 4, 8):
        got = decimate_edges(scrambled, 13, step)
        degree = np.bincount(got.ravel(), minlength=13)
        assert degree[order[6]] == 3, "branch point lost"
        for tip in (0, 9, 12):
            assert degree[order[tip]] == 1, "tip lost"

    # A closed loop has no node of degree != 2 to start from; it must still
    # terminate and stay closed
    loop = np.column_stack([np.arange(8), (np.arange(8) + 1) % 8])
    assert len(decimate_edges(loop, 8, 2)) == 4

    assert decimate_edges(np.zeros((0, 2), int), 0, 2).shape == (0, 2)


def test_axial_lod():
    """`axial_lod=n` keeps every 2**n-th node, and keeps every arm."""
    coefs, edges = y_skeleton(radius=1.0)

    # Subdivide each arm so there is something to decimate
    dense_pos, dense_edges = [], []
    for a, b in edges:
        pa, pb = coefs[a, :3], coefs[b, :3]
        base = len(dense_pos)
        for t in np.linspace(0, 1, 9):
            dense_pos.append(pa + t * (pb - pa))
        dense_edges += [(base + i, base + i + 1) for i in range(8)]
    dense = np.zeros((len(dense_pos), coefs.shape[1]), np.float32)
    dense[:, :3] = dense_pos
    dense[:, 3:7] = QUAT_X
    dense[:, 7] = 1.0
    dense_edges = np.array(dense_edges, np.int32)

    previous = None
    for lod in (0, 1, 2):
        v = oc.Viewer(offscreen=True, size=(256, 256))
        v.add_tubes(dense, edges=dense_edges, axial_lod=lod)
        (vis,) = [o for objs in v.objects.values() for o in objs]
        mask = shoot(v, width=24.0)[..., 3] > 0
        v.close()

        # Roughly halves each level (the arms' shared endpoints are duplicated
        # in this fixture, so this is not exactly len/2**lod)
        if previous is not None:
            assert vis.n_edges < previous
        previous = vis.n_edges

        # ... and all three arms survive, whatever the level
        assert arms_at(mask, 0.3) == 1
        assert arms_at(mask, 0.7) == 2

    # The node buffer is never thinned - only the edge list is. (The frames in
    # it are realigned against the decimated chords, but no node is dropped.)
    v = oc.Viewer(offscreen=True, size=(128, 128))
    v.add_tubes(dense, edges=dense_edges, axial_lod=2)
    (vis,) = [o for objs in v.objects.values() for o in objs]
    assert vis.n_nodes == len(dense)
    v.close()


def test_lod_validation():
    coefs, edges = straight_tube()
    with pytest.raises(ValueError, match="axial_lod"):
        render(coefs, edges, axial_lod=-1)


# --------------------------------------------------------------------------
# Frame realignment
# --------------------------------------------------------------------------


def ring_points(coefs, n_theta=64):
    """CPU mirror of the swept surface, sampled per node."""
    from octarine.shaders.tubes import _frame_uvt

    k = (coefs.shape[1] - 8) // 2
    u, v, _ = _frame_uvt(coefs[:, 3:7].astype(np.float64))
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    kk = np.arange(1, k + 1)[None, :, None]
    a = coefs[:, 8 : 8 + k].astype(np.float64)
    b = coefs[:, 8 + k : 8 + 2 * k].astype(np.float64)
    r = coefs[:, 7][:, None].astype(np.float64) + (
        a[:, :, None] * np.cos(kk * theta) + b[:, :, None] * np.sin(kk * theta)
    ).sum(1)
    e_r = (
        np.cos(theta)[None, :, None] * u[:, None, :]
        + np.sin(theta)[None, :, None] * v[:, None, :]
    )
    return (coefs[:, :3][:, None, :].astype(np.float64) + r[:, :, None] * e_r).reshape(
        -1, 3
    )


def twist_per_edge(coefs, edges):
    """Angle between one node's theta = 0 carried onto the next, and the next's.

    Zero everywhere is a rotation-minimizing frame - i.e. the quad between the
    two rings connects corresponding points rather than twisting between them.
    """
    from octarine.shaders.tubes import _frame_uvt, _transport

    u, v, t = _frame_uvt(coefs[:, 3:7].astype(np.float64))
    i, j = np.asarray(edges)[:, 0], np.asarray(edges)[:, 1]
    p = _transport(u[i], t[i], t[j])
    return np.abs(np.degrees(np.arctan2((p * v[j]).sum(1), (p * u[j]).sum(1))))


def twisted_y(k=2, radius=1.0, seed=0):
    """A Y whose stored frames are rolled by a random angle at every node."""
    from octarine.shaders.tubes import _frame_uvt, _quat_from_uvt

    coefs, edges = y_skeleton(k=k, radius=radius)
    u, v, t = _frame_uvt(coefs[:, 3:7].astype(np.float64))
    roll = np.random.default_rng(seed).uniform(-np.pi, np.pi, len(coefs))
    c, s = np.cos(roll)[:, None], np.sin(roll)[:, None]
    u2 = u * c + v * s
    coefs[:, 3:7] = _quat_from_uvt(u2, np.cross(t, u2), t)
    # An off-centre, non-circular cross-section, so a mis-rotated frame shows
    coefs[:, 8] = 0.4 * radius
    coefs[:, 8 + k] = 0.25 * radius
    return coefs, edges


def test_split_runs():
    """Every edge belongs to exactly one run, ends have degree != 2."""
    from octarine.shaders import split_runs

    # Node 2 is the branch point, 0/4/6 the tips: three runs of three nodes
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [2, 5], [5, 6]])
    runs = split_runs(edges, 7)
    assert sorted(len(r) for r in runs) == [3, 3, 3]
    seen = sorted(tuple(sorted(p)) for run in runs for p in zip(run[:-1], run[1:]))
    assert seen == sorted(tuple(sorted(e)) for e in edges), "edges not covered once"

    # A closed loop has no degree != 2 node; it comes back broken open
    loop = np.column_stack([np.arange(6), (np.arange(6) + 1) % 6])
    (run,) = split_runs(loop, 6)
    assert run[0] == run[-1] and len(run) == 7

    assert split_runs(np.zeros((0, 2), int), 0) == []


def test_rotate_profile_matches_reevaluation():
    """Rotating the coefficients by psi must equal evaluating at theta + psi."""
    from octarine.shaders import rotate_profile

    rng = np.random.default_rng(1)
    k = 5
    coefs = np.zeros((7, 8 + 2 * k), np.float32)
    coefs[:, 3:7] = QUAT_X
    coefs[:, 7] = 3.0
    coefs[:, 8:] = rng.uniform(-0.4, 0.4, (7, 2 * k))
    psi = rng.uniform(-np.pi, np.pi, 7)

    theta = np.linspace(0, 2 * np.pi, 37, endpoint=False)
    kk = np.arange(1, k + 1)[None, :, None]

    def radius(c, th):
        a = c[:, 8 : 8 + k].astype(np.float64)[:, :, None]
        b = c[:, 8 + k :].astype(np.float64)[:, :, None]
        return c[:, 7][:, None].astype(np.float64) + (
            a * np.cos(kk * th) + b * np.sin(kk * th)
        ).sum(1)

    expect = radius(coefs, (theta[None, None, :] + psi[:, None, None]))
    got = radius(rotate_profile(coefs.copy(), psi), theta[None, None, :])
    assert np.abs(got - expect).max() < 1e-5


def test_align_frames_phase_only_preserves_the_surface():
    """Realigning the phase rotates the parametrisation, not the geometry."""
    from octarine.shaders import align_frames

    coefs, edges = twisted_y(radius=2.0)
    before, after = (
        ring_points(coefs, 2048),
        ring_points(align_frames(coefs, edges, retangent=False), 64),
    )
    # Compared as point sets: sample i of one is not sample i of the other
    d = np.linalg.norm(after[:, None, :] - before[None, :, :], axis=-1).min(axis=1)
    assert d.max() < 0.02, "phase realignment moved the surface"


def test_align_frames_removes_twist():
    """One rotation-minimizing chain over the whole tree, branch points included."""
    from octarine.shaders import align_frames

    coefs, edges = twisted_y()
    assert twist_per_edge(coefs, edges).max() > 45, "fixture is not twisted"
    # Floor is the float32 quaternion the frame round-trips through
    assert twist_per_edge(align_frames(coefs, edges), edges).max() < 1e-3


def test_align_frames_retangents_onto_the_chords():
    """With `retangent`, a node's tangent follows the chords it is swept along."""
    from octarine.shaders import align_frames
    from octarine.shaders.tubes import _frame_uvt

    # A right-angle bend whose stored tangents all point along +x, i.e. as
    # wrong as they can be for the second arm
    pts = np.array([[0, 0, 0], [4, 0, 0], [4, 4, 0], [4, 8, 0]], np.float32)
    coefs = np.zeros((4, 8 + 2 * 2), np.float32)
    coefs[:, :3] = pts
    coefs[:, 3:7] = QUAT_X  # t = +x everywhere
    coefs[:, 7] = 1.0
    edges = np.array([[0, 1], [1, 2], [2, 3]], np.int32)

    _, _, t = _frame_uvt(align_frames(coefs, edges)[:, 3:7].astype(np.float64))
    # The corner bisects +x and +y; the far arm is pure +y
    assert np.allclose(t[0], [1, 0, 0], atol=1e-6)
    assert np.allclose(t[1], np.array([1, 1, 0]) / np.sqrt(2), atol=1e-6)
    assert np.allclose(t[3], [0, 1, 0], atol=1e-6)

    # ... and the frame stays orthonormal and right-handed throughout
    u, v, t = _frame_uvt(align_frames(coefs, edges)[:, 3:7].astype(np.float64))
    assert np.abs((u * v).sum(1)).max() < 1e-6
    assert np.abs((u * t).sum(1)).max() < 1e-6
    assert np.allclose(np.cross(u, v), t, atol=1e-6)


def test_material_validation():
    from octarine.shaders import TubeMaterial, TubeVisual

    mat = TubeMaterial(n_theta=16, k_max=2)
    assert mat.n_theta == 16
    assert mat.k_max == 2
    assert mat.k_normal == 1  # the normal is truncated harder by default

    with pytest.raises(ValueError, match="n_theta"):
        mat.n_theta = 2
    with pytest.raises(ValueError, match="k_max"):
        mat.k_max = -1
    with pytest.raises(ValueError, match="k_normal"):
        mat.k_normal = -1

    coefs, edges = straight_tube()
    # 8 + 2K columns, so an odd number of harmonic columns is a mistake
    with pytest.raises(ValueError, match="8 \\+ 2K"):
        TubeVisual(coefs[:, :-1], edges, TubeMaterial())
    with pytest.raises(ValueError, match="Edge indices"):
        TubeVisual(coefs, np.array([[0, 999]], np.int32), TubeMaterial())


def test_add_via_duck_typing():
    """`Viewer.add` must pick up anything that quacks like a TubeProfile."""
    buf, node_edges = straight_tube(a0=2.0, k=4)

    class FakeProfile:
        """Mimics `sparsecubes.TubeProfile` without importing sparse-cubes."""

        a0 = buf[:, 7]
        mag = np.zeros((len(buf), 4), np.float32)
        phase = np.zeros((len(buf), 4), np.float32)
        frame = buf[:, 3:7]
        edges = node_edges

        def __init__(self, form="cartesian"):
            self._form = form

        def to_gpu_buffer(self):
            return buf, {
                "K": 4,
                "n_nodes": len(buf),
                "stride_floats": buf.shape[1],
                "form": self._form,
            }

    from octarine.utils import is_tube_profile

    assert is_tube_profile(FakeProfile())
    assert not is_tube_profile(np.zeros((10, 16)))

    v = oc.Viewer(offscreen=True, size=(128, 128))
    v.add(FakeProfile())
    (vis,) = [o for objs in v.objects.values() for o in objs]
    assert vis._object_type == "tubes"
    assert vis.n_edges == len(node_edges)
    v.canvas.draw()

    # The polar form puts m_k / phi_k in the same slots and is indistinguishable
    # by shape, so it must be rejected rather than silently mis-rendered
    with pytest.raises(ValueError, match="Cartesian"):
        v.add(FakeProfile(form="polar"))
    v.close()


def test_bounding_box_covers_the_surface():
    """The auto-centering must frame the tube, not just its skeleton."""
    from octarine.shaders import TubeMaterial, TubeVisual

    a0, a2, b2 = 2.0, 0.5, 0.25
    coefs, edges = straight_tube(a0=a0, a2=a2, b2=b2, length=10.0)
    vis = TubeVisual(coefs, edges, TubeMaterial())

    bbox = vis.get_world_bounding_box()
    reach = a0 + np.hypot(a2, b2)

    assert bbox[0] == pytest.approx([-reach, -reach, -reach], abs=1e-4)
    assert bbox[1] == pytest.approx([10.0 + reach, reach, reach], abs=1e-4)
