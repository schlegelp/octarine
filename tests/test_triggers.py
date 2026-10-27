import os
import pytest
import pygfx as gfx
import trimesh as tm
import octarine as oc

# The GUI tests below must not try to open a window
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SIZE = (400, 300)


def _make_viewer():
    """An offscreen viewer with an object to look at, rendering reactively."""
    v = oc.Viewer(offscreen=True, size=SIZE)
    v.add(tm.primitives.Box(), name="box")
    v.render_trigger = "reactive"
    v._animate()  # let it settle (the first frame is always drawn)
    v._render_stale = False
    return v


@pytest.fixture()
def viewer():
    v = _make_viewer()
    yield v
    v.close()


@pytest.fixture()
def viewers():
    """Two reactive viewers, linked to each other."""
    vv = [_make_viewer() for _ in range(2)]
    vv[0].link(vv[1])
    for v in vv:
        v._animate()
        v._render_stale = False
    yield vv
    for v in vv:
        v.close()


def _drag(viewer):
    """Rotate the camera the way a user would: by dragging with the mouse."""
    for event in (
        gfx.objects.PointerEvent(
            type="pointer_down", x=100, y=100, button=1, buttons=(1,), modifiers=()
        ),
        gfx.objects.PointerEvent(
            type="pointer_move", x=160, y=140, button=1, buttons=(1,), modifiers=()
        ),
        gfx.objects.PointerEvent(
            type="pointer_up", x=160, y=140, button=1, buttons=(), modifiers=()
        ),
    ):
        viewer.renderer.dispatch_event(event)


def _run_loop(viewer, requests, max_frames=100):
    """Draw frames for as long as the viewer asks for them.

    This is what the canvas does for us in a live viewer: a frame is drawn
    only if something requested one. `requests` is the list a previous
    `_count_draw_requests` hands out. Returns the number of frames drawn - if
    the viewer never goes back to sleep, that is `max_frames`.

    """
    frames = 0
    while requests and frames < max_frames:
        requests.clear()
        viewer._animate()
        frames += 1
    return frames


def _count_renders(viewer, monkeypatch):
    """Make the viewer count the frames it actually renders."""
    renders = []
    render = viewer.renderer.render

    def counting_render(*args, **kwargs):
        # The viewer renders the scene, the gizmo and the overlay per frame;
        # count the first of those only
        if args[0] is viewer.scene:
            renders.append(1)
        return render(*args, **kwargs)

    monkeypatch.setattr(viewer.renderer, "render", counting_render)
    return renders


def _count_draw_requests(viewer, monkeypatch):
    """Make the viewer count the frames it asks the canvas for."""
    requests = []
    request_draw = viewer.canvas.request_draw

    def counting_request_draw(*args, **kwargs):
        requests.append(1)
        return request_draw(*args, **kwargs)

    monkeypatch.setattr(viewer.canvas, "request_draw", counting_request_draw)
    return requests


def test_idle_viewer_lets_the_canvas_sleep(viewer, monkeypatch):
    requests = _count_draw_requests(viewer, monkeypatch)

    for _ in range(3):
        viewer._animate()

    assert not requests


def test_continuous_viewer_keeps_the_loop_running(viewer, monkeypatch):
    viewer.render_trigger = "continuous"
    requests = _count_draw_requests(viewer, monkeypatch)

    for _ in range(3):
        viewer._animate()

    assert len(requests) == 3


def test_stale_scene_wakes_the_canvas(viewer, monkeypatch):
    requests = _count_draw_requests(viewer, monkeypatch)

    viewer._render_stale = True

    assert len(requests) == 1


@pytest.mark.parametrize("req_render", [True, False])
def test_animations_keep_the_loop_running(viewer, monkeypatch, req_render):
    # Animations are called once per frame, so they need frames to keep coming
    # whether or not running them dirties the scene: a `req_render=False` one
    # is typically polling for work that will (loading objects in the
    # background, say) and must not be left asleep
    requests = _count_draw_requests(viewer, monkeypatch)

    # Adding the first animation has to wake the sleeping canvas...
    viewer.add_animation(lambda: None, req_render=req_render)
    assert len(requests) == 1

    # ... and from then on the loop has to keep itself going
    for _ in range(3):
        requests.clear()
        viewer._animate()
        assert requests


def test_camera_movement_from_code_asks_for_a_frame(viewer, monkeypatch):
    # Named views go past none of the event handlers, so `set_view` has to ask
    # for the frame that shows the new view itself
    requests = _count_draw_requests(viewer, monkeypatch)
    viewer.set_view("XZ")
    assert len(requests) == 1

    # Reaching for the camera directly cannot wake a sleeping canvas - but it
    # must not be missed by a frame that is drawn for whatever other reason
    renders = _count_renders(viewer, monkeypatch)
    viewer._animate()
    viewer.camera.zoom = 2.0
    viewer._animate()
    assert len(renders) == 2


def test_held_key_keeps_moving_the_camera(viewer, monkeypatch):
    # A fly controller moves the camera for as long as a key is held - which
    # it can only do from the frames the viewer asks for
    viewer.controller = gfx.FlyController(
        viewer.camera, register_events=viewer.renderer
    )
    before = viewer._camera_sig()
    requests = _count_draw_requests(viewer, monkeypatch)

    viewer.renderer.dispatch_event(
        gfx.objects.KeyboardEvent(type="key_down", key="w", modifiers=())
    )
    frames = _run_loop(viewer, requests, max_frames=10)

    assert viewer._camera_sig() != before
    # The key is still down, so the viewer must not stop asking for frames
    assert frames == 10


def test_drag_is_rendered_until_the_motion_settles(viewer, monkeypatch):
    # Gestures are damped: the controller keeps moving the camera for a few
    # frames after the gesture itself is over, and the viewer has to keep
    # asking for frames until it has come to a rest
    before = viewer._camera_sig()
    requests = _count_draw_requests(viewer, monkeypatch)

    _drag(viewer)
    frames = _run_loop(viewer, requests, max_frames=10)

    assert viewer._camera_sig() != before
    # Damping settles in wall-clock time, so in a loop as tight as this one
    # the motion is still going: the viewer has to keep asking for frames
    assert frames == 10


def test_viewer_goes_back_to_sleep_once_the_motion_stops(viewer, monkeypatch):
    # Without damping the gesture is over as soon as the button comes up, so
    # the viewer has to stop asking for frames again
    viewer.controller.damping = 0
    requests = _count_draw_requests(viewer, monkeypatch)

    _drag(viewer)
    frames = _run_loop(viewer, requests)

    assert frames < 5


def test_linked_viewer_is_woken_by_the_other_one(viewers, monkeypatch):
    # The controller of a linked viewer drives our camera directly, so none of
    # our own handlers fire - and asleep, nothing of ours is running to notice
    v1, v2 = viewers
    before = v2._camera_sig()
    requests = _count_draw_requests(v2, monkeypatch)

    _drag(v1)
    for _ in range(3):
        v1._animate()

    assert v2._camera_sig() != before
    assert requests


def test_passive_animations_do_not_keep_the_canvas_awake(viewer, monkeypatch):
    # The scale bar only keeps itself in step with frames that are drawn
    # anyway (`_passive=True`), so it must not hold the loop open
    viewer.set_scalebar(True)
    viewer._animate()
    requests = _count_draw_requests(viewer, monkeypatch)

    for _ in range(3):
        viewer._animate()

    assert not requests

    # ... but it still follows the camera
    before = viewer._scalebar._state
    viewer.camera.zoom = 3.0
    viewer._animate()
    assert viewer._scalebar._state != before


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v.set_bgcolor("white"),
        lambda v: v.add_effect("noise"),
        lambda v: v.show_message("hello"),
        lambda v: v.highlight_objects("box"),
        lambda v: setattr(v, "show_bounds", True),
        lambda v: setattr(v, "moveable_object", "box"),
    ],
)
def test_changes_to_the_scene_ask_for_a_frame(viewer, monkeypatch, change):
    requests = _count_draw_requests(viewer, monkeypatch)

    change(viewer)

    assert requests
