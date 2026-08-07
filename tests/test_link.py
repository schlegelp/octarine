import os
import pytest
import numpy as np
import trimesh as tm
import octarine as oc

# The GUI tests below must not try to open a window
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SIZE = (400, 300)
RECT = (0, 0, *SIZE)  # the viewport the controller gestures act on


@pytest.fixture()
def viewers():
    """Three offscreen viewers, each with an object to look at."""
    vv = [oc.Viewer(offscreen=True, size=SIZE) for _ in range(3)]
    for v in vv:
        v.add(tm.primitives.Box())
    yield vv
    for v in vv:
        v.close()


def _state(viewer):
    """The parts of the camera state the controllers actually move."""
    cam = viewer.camera
    return (
        np.asarray(cam.local.position),
        np.asarray(cam.local.rotation),
        np.array([cam.width, cam.height]),
    )


def _same_view(a, b):
    return all(np.allclose(x, y) for x, y in zip(_state(a), _state(b)))


def _move(viewer):
    """Pan, rotate and zoom via the controller (i.e. as a user would)."""
    viewer.controller.pan((17, -23), RECT)
    viewer.controller.rotate((0.4, 0.2), RECT)
    viewer.controller.zoom(0.5, RECT)


def test_link_syncs_current_view(viewers):
    v1, v2, _ = viewers

    # Move v1 away so the two viewers start out with different views
    _move(v1)
    assert not _same_view(v1, v2)

    # Linking should make v2 adopt v1's view
    v1.link(v2)
    assert _same_view(v1, v2)


def test_link_is_symmetrical(viewers):
    v1, v2, _ = viewers
    v1.link(v2)

    assert v1.linked == (v2,)
    assert v2.linked == (v1,)

    # Both controllers now drive both cameras...
    for v in (v1, v2):
        assert set(map(id, v.controller.cameras)) == {id(v1.camera), id(v2.camera)}
        # ... but each still reads its own camera first
        assert v.controller.cameras[0] is v.camera

    # Moving either viewer moves the other
    _move(v1)
    assert _same_view(v1, v2)
    _move(v2)
    assert _same_view(v1, v2)


def test_link_is_transitive(viewers):
    v1, v2, v3 = viewers

    v1.link(v2)
    v2.link(v3)

    for v in viewers:
        assert len(v.linked) == 2
        assert len(v.controller.cameras) == 3

    # v1 and v3 were never linked directly but still move together
    _move(v1)
    assert _same_view(v1, v3)
    _move(v3)
    assert _same_view(v1, v2)


def test_link_multiple_at_once(viewers):
    v1, v2, v3 = viewers

    # Both a list and separate arguments should work
    v1.link([v2, v3])
    assert len(v1.linked) == 2

    v1.unlink()
    v1.link(v2, v3)
    assert len(v1.linked) == 2

    _move(v2)
    assert _same_view(v1, v2) and _same_view(v1, v3)


def test_link_propagates_set_view(viewers):
    v1, v2, _ = viewers
    v1.link(v2)

    v1.set_view("XZ")
    assert _same_view(v1, v2)

    # A saved view (i.e. a state dict) should propagate as well
    state = v1.get_view()
    _move(v1)
    v1.set_view(state)
    assert _same_view(v1, v2)

    # ... and so should re-centering
    v1.center_camera()
    assert _same_view(v1, v2)


def test_reactive_viewer_rerenders_when_driven_by_link(viewers):
    """A linked viewer in reactive mode must notice that its camera moved."""
    v1, v2, _ = viewers
    v1.link(v2)
    v2.render_trigger = "reactive"

    renders = []
    v2.renderer.render = lambda *args, **kwargs: renders.append(1)

    # Bring v2 up to date, then check that it does indeed sit still
    v2._animate()
    n = len(renders)
    assert n > 0
    v2._animate()
    assert len(renders) == n

    # Now move v1 through its controller: this moves v2's camera without v2
    # ever seeing an event of its own
    _move(v1)
    v2._animate()
    assert len(renders) > n


@pytest.mark.parametrize(
    "kwargs,expect_pos,expect_rot,expect_zoom",
    [
        ({}, True, True, True),
        ({"sync": "rotation"}, False, True, False),
        ({"sync": ["position", "rotation"]}, True, True, False),
        ({"exclude": ["width", "height"]}, True, True, False),
        ({"exclude": "rotation"}, True, False, True),
        ({"sync": ["x", "y", "z"]}, True, False, False),
    ],
)
def test_link_partial(viewers, kwargs, expect_pos, expect_rot, expect_zoom):
    v1, v2, _ = viewers
    v1.link(v2, **kwargs)

    before = _state(v2)
    _move(v1)
    after = _state(v2)

    for i, expected in enumerate((expect_pos, expect_rot, expect_zoom)):
        if expected:
            # v2 followed v1
            assert np.allclose(after[i], _state(v1)[i])
        else:
            # v2 stayed where it was
            assert np.allclose(after[i], before[i])


def test_link_mixed_camera_types():
    """Linking an ortho with a perspective viewer must not flatten the latter."""
    ortho = oc.Viewer(offscreen=True, size=SIZE)
    persp = oc.Viewer(offscreen=True, size=SIZE, camera="perspective")
    for v in (ortho, persp):
        v.add(tm.primitives.Box())

    fov = persp.camera.fov
    assert fov > 0

    ortho.link(persp)
    assert persp.camera.fov == fov

    _move(ortho)
    assert np.allclose(ortho.camera.local.rotation, persp.camera.local.rotation)
    assert persp.camera.fov == fov

    # ... unless explicitly asked for
    ortho.unlink()
    ortho.link(persp, sync=["rotation", "fov"])
    assert persp.camera.fov == ortho.camera.fov == 0

    for v in (ortho, persp):
        v.close()


def test_link_two_perspective_cameras_syncs_fov():
    v1 = oc.Viewer(offscreen=True, size=SIZE, camera="perspective")
    v2 = oc.Viewer(offscreen=True, size=SIZE, camera="perspective")
    for v in (v1, v2):
        v.add(tm.primitives.Box())

    v1.link(v2)
    v1.camera.fov = 70
    v1.set_view(v1.get_view())
    assert v2.camera.fov == 70

    for v in (v1, v2):
        v.close()


def test_unlink_self(viewers):
    v1, v2, v3 = viewers
    v1.link(v2, v3)

    # Removing v1 leaves v2 and v3 linked with each other
    v1.unlink()
    assert v1.linked == ()
    assert v1.controller.cameras == (v1.camera,)
    assert set(map(id, v2.linked)) == {id(v3)}
    assert set(map(id, v3.linked)) == {id(v2)}

    view = _state(v1)
    _move(v2)
    assert _same_view(v2, v3)
    assert all(np.allclose(x, y) for x, y in zip(_state(v1), view))


def test_unlink_other(viewers):
    v1, v2, v3 = viewers
    v1.link(v2, v3)

    v1.unlink(v3)
    assert v3.linked == ()
    assert v3.controller.cameras == (v3.camera,)
    assert set(map(id, v1.linked)) == {id(v2)}

    view = _state(v3)
    _move(v1)
    assert _same_view(v1, v2)
    assert all(np.allclose(x, y) for x, y in zip(_state(v3), view))


def test_unlink_is_idempotent(viewers):
    v1, v2, v3 = viewers

    # Unlinking an unlinked viewer, or one from another group, is a no-op
    v1.unlink()
    v1.unlink(v2)

    v1.link(v2)
    v1.unlink(v3)
    assert v1.linked == (v2,)

    v1.unlink(v2)
    v1.unlink(v2)
    assert v1.linked == () and v2.linked == ()


def test_relink_updates_filter(viewers):
    v1, v2, _ = viewers
    v1.link(v2, sync="rotation")
    v1.link(v2)  # re-link, this time fully

    _move(v1)
    assert _same_view(v1, v2)


def test_close_unlinks(viewers):
    v1, v2, v3 = viewers
    v1.link(v2, v3)

    v2.close()
    assert v2.linked == ()
    assert set(map(id, v1.linked)) == {id(v3)}
    for cam in v1.controller.cameras:
        assert cam is not v2.camera

    # The remaining viewers must still work
    _move(v1)
    assert _same_view(v1, v3)


def test_link_errors(viewers):
    v1, v2, _ = viewers

    with pytest.raises(ValueError):
        v1.link(v1)

    with pytest.raises(ValueError):
        v1.link()

    with pytest.raises(TypeError):
        v1.link("not a viewer")

    with pytest.raises(TypeError):
        v1.unlink(42)

    with pytest.raises(ValueError):
        v1.link(v2, sync="not-a-field")

    with pytest.raises(ValueError):
        v1.link(v2, exclude=["rotation", "not-a-field"])

    # None of the above should have left a partial link behind
    assert v1.linked == () and v2.linked == ()
