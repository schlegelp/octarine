import time
import pytest
import octarine as oc

import trimesh as tm
import numpy as np

# Set random state
np.random.seed(0)


@pytest.fixture
def mesh():
    return tm.creation.icosphere()


@pytest.fixture()
def line_single():
    return np.random.rand(10, 3)


@pytest.fixture()
def line_stack():
    return [np.random.rand(i, 3) for i in np.random.randint(2, 10, 10)]


@pytest.fixture()
def points():
    return np.random.rand(10, 3)


@pytest.fixture()
def points_colors():
    return np.random.rand(10, 3), np.random.rand(10, 3)


def test_adding_generic_objects(mesh, line_single, line_stack, points, points_colors):
    v = oc.Viewer(offscreen=True)

    # Test adding objects generically
    for ob in [mesh, line_single, line_stack, points, points_colors]:
        v.add(ob)
        v.clear()

    v.close()


@pytest.mark.parametrize("color", [None, "red", np.random.rand(3)])
def test_adding_mesh(mesh, color):
    v = oc.Viewer(offscreen=True)
    v.add_mesh(mesh, color=color)
    v.close()


def test_showing_messsage():
    v = oc.Viewer(offscreen=True)
    v.show_message("test", color="red")
    v.show_message(None, color="red")
    v.show_message("test", color="red", duration=1)
    time.sleep(2)
    v.close()


@pytest.fixture()
def sphere_shell():
    """~50k voxel coordinates on a sphere shell (sparse volumetric data)."""
    v = np.random.normal(size=(50_000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * 40 + 50


def test_pack_sparse_voxels():
    from octarine.shaders.packing import pack_sparse_voxels

    # Three hand-placed points spanning two bricks (brick_size=16)
    pts = np.array(
        [
            [3, 4, 5],  # -> brick (0, 0, 0), local (0, 0, 0) after origin shift
            [3, 4, 6],  # -> same brick, local (0, 0, 1)
            [20, 4, 5],  # -> brick (1, 0, 0), local (1, 0, 0)
        ]
    )
    p = pack_sparse_voxels(pts)

    assert (p.origin == [3, 4, 5]).all()
    assert p.shape == (18, 1, 2)
    assert p.n_bricks == 2
    # Coarse index is zyx-ordered; slots are 1-based (0 = empty)
    assert p.coarse.shape == (1, 1, 2)
    assert p.coarse[0, 0, 0] == 1 and p.coarse[0, 0, 1] == 2
    # Atlas is zyx-ordered with cells of brick + 1-voxel apron (18^3),
    # packed along x; payload voxels are offset by +1 for the apron
    assert p.atlas.shape == (18, 18, 36)
    assert p.atlas[1, 1, 1] == 255  # point 1
    assert p.atlas[2, 1, 1] == 255  # point 2
    assert p.atlas[1, 1, 20] == 255  # point 3 (slot 1 -> atlas x offset 18)
    assert p.atlas.sum() == 3 * 255

    # Values are quantized into 1-255 (0 = empty); duplicates keep the max
    p2 = pack_sparse_voxels(pts, values=[0.0, 5.0, 10.0])
    assert p2.atlas[1, 1, 1] == 1
    assert p2.atlas[2, 1, 1] == 128
    assert p2.atlas[1, 1, 20] == 255


def test_pack_sparse_voxels_apron():
    from octarine.shaders.packing import pack_sparse_voxels

    # Two points on either side of a brick border (x=15|16) plus an anchor
    # at the origin; border voxels must be mirrored into the neighboring
    # brick's apron layer so trilinear sampling is seamless across bricks.
    pts = np.array([[0, 0, 0], [15, 0, 0], [16, 0, 0]])
    p = pack_sparse_voxels(pts)

    assert p.n_bricks == 2
    # Own-brick payload writes
    assert p.atlas[1, 1, 1] == 255  # (0,0,0)
    assert p.atlas[1, 1, 16] == 255  # (15,0,0) -> local x 15 (+1 apron)
    assert p.atlas[1, 1, 19] == 255  # (16,0,0) -> brick 1, local x 0
    # Apron copies: (16,0,0) into brick 0's far-x apron layer, and
    # (15,0,0) into brick 1's near-x apron layer
    assert p.atlas[1, 1, 17] == 255
    assert p.atlas[1, 1, 18] == 255
    assert p.atlas.sum() == 5 * 255


@pytest.mark.parametrize("mode", ["mip", "density"])
def test_adding_sparse_volume(sphere_shell, mode):
    v = oc.Viewer(offscreen=True)
    v.add_sparse_volume(sphere_shell, mode=mode)
    (obj,) = [o for objs in v.objects.values() for o in objs]
    assert obj._object_type == "sparsevolume"
    v.canvas.draw()  # force the shader to actually compile/render
    v.close()


def test_adding_voxel_cloud(sphere_shell):
    # VoxelCloud routes through viewer.add + the converter registry
    v = oc.Viewer(offscreen=True)
    v.add(oc.VoxelCloud(sphere_shell, values=sphere_shell[:, 2]))
    (obj,) = [o for objs in v.objects.values() for o in objs]
    assert obj._object_type == "sparsevolume"
    v.canvas.draw()
    v.close()


def test_sparse_volume_dense_fallback(sphere_shell):
    v = oc.Viewer(offscreen=True)
    v.add_sparse_volume(sphere_shell, method="dense")
    (obj,) = [o for objs in v.objects.values() for o in objs]
    assert obj._object_type == "volume"
    v.close()


def test_sparse_volume_render_correctness(sphere_shell):
    """The rendered MIP silhouette must match the analytic projection."""
    v = oc.Viewer(offscreen=True, size=(400, 400))
    v.add_sparse_volume(sphere_shell)
    img = np.asarray(v.screenshot(filename=None, size=(400, 400)))
    v.close()

    mask = img[..., :3].max(axis=-1) > 20
    assert mask.any()

    # The shell projects to a disk inscribed in the volume's square
    # silhouette; the camera looks down +z and centers the volume.
    ys, xs = np.where(mask)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    radius = (ys.max() - ys.min() + 1) / 2
    yy, xx = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2

    iou = (mask & disk).sum() / (mask | disk).sum()
    assert iou > 0.9, f"Rendered silhouette does not match projection (IoU={iou:.3f})"


def test_sparse_volume_rolled_camera(sphere_shell):
    """Corner-on camera orientations with roll must not make the volume vanish.

    Regression test: pygfx's volume_ray.wgsl derives the camera handedness
    from the product of the view-matrix diagonal, which is <= 0 for some pure
    rotations (corner-on views with roll) and makes the volume disappear.
    The sparse-volume shader uses the actual determinant instead.
    """
    v = oc.Viewer(offscreen=True, size=(200, 200))
    v.add_sparse_volume(sphere_shell)

    d = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    base = np.cross(d, [0, 0, 1.0])
    base /= np.linalg.norm(base)
    other = np.cross(d, base)
    for i in range(8):
        phi = 2 * np.pi * i / 8
        up = np.cos(phi) * base + np.sin(phi) * other
        v.camera.show_object(v.scene, scale=1, view_dir=tuple(d), up=tuple(up))
        img = np.asarray(v.screenshot(filename=None, size=(200, 200)))
        px = (img[..., :3].max(axis=-1) > 20).sum()
        assert px > 1000, f"Volume vanished at roll {i * 45} deg ({px} px)"
    v.close()
