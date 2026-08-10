import time
import pytest
import octarine as oc

import trimesh as tm
import numpy as np
import pylinalg as la

from functools import partial

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


def test_bounds(points):
    v = oc.Viewer(offscreen=True)

    # Nothing on the canvas
    assert v.bounds is None

    # A box of (2, 4, 6) centered on the origin
    v.add_mesh(tm.creation.box((2, 4, 6)), name="box")
    assert v.bounds.shape == (3, 2)
    assert v.bounds == pytest.approx(np.array([[-1, 1], [-2, 2], [-3, 3]]))

    # Bounds are the union over all visuals...
    v.add_points(np.array([[10.0, 0, 0], [0, -20, 0]]), name="points")
    assert v.bounds == pytest.approx(np.array([[-1, 10], [-20, 2], [-3, 3]]))

    # ... in world space, i.e. transforms count
    v.objects["box"][0].local.position = (100, 0, 0)
    assert v.bounds == pytest.approx(np.array([[0, 101], [-20, 2], [-3, 3]]))

    # Invisible visuals still count
    v.objects["box"][0].visible = False
    assert v.bounds == pytest.approx(np.array([[0, 101], [-20, 2], [-3, 3]]))

    # The bounding box visual is viewer chrome and must not count itself
    before = v.bounds.copy()
    v.show_bounds = True
    assert v.bounds == pytest.approx(before)

    v.clear()
    assert v.bounds is None

    v.close()


@pytest.mark.parametrize("color", [None, "red", np.random.rand(3)])
def test_adding_mesh(mesh, color):
    v = oc.Viewer(offscreen=True)
    v.add_mesh(mesh, color=color)
    v.close()


def test_adding_mesh_shader(mesh):
    from octarine.visuals import available_shaders

    v = oc.Viewer(offscreen=True)
    for name, material in available_shaders().items():
        v.add_mesh(mesh, color="red", shader=name)
        (obj,) = [o for objs in v.objects.values() for o in objs]
        assert isinstance(obj.material, material)
        v.canvas.draw()  # force the shader to actually compile/render
        v.clear()

    # Passing a material class directly must also work
    import pygfx as gfx

    v.add_mesh(mesh, color="red", shader=gfx.MeshToonMaterial)
    (obj,) = [o for objs in v.objects.values() for o in objs]
    assert isinstance(obj.material, gfx.MeshToonMaterial)

    with pytest.raises(ValueError, match="Unknown shader"):
        v.add_mesh(mesh, shader="does-not-exist")

    # Silhouette requires the default phong shader
    with pytest.raises(ValueError, match="silhouette"):
        v.add_mesh(mesh, shader="toon", silhouette=2.0)

    # ... and so does subsurface scattering
    with pytest.raises(ValueError, match="subsurface"):
        v.add_mesh(mesh, shader="toon", subsurface=1.0)

    v.close()


def test_mesh_silhouette_render(mesh):
    """Silhouette must make face-on regions transparent but keep the rim bright."""

    def render(silhouette):
        v = oc.Viewer(offscreen=True, size=(300, 300))
        v.add_mesh(mesh, color="red", silhouette=silhouette)
        img = np.asarray(v.screenshot(filename=None, size=(300, 300)))
        v.close()
        return img[..., :3].max(axis=-1).astype(float)

    plain = render(None)
    sil = render(2.0)

    # The camera centers the object, so the image center is face-on
    # (note: the screenshot may come back at 2x on hidpi displays)
    h, w = plain.shape
    c = np.s_[h // 2 - 10 : h // 2 + 10, w // 2 - 10 : w // 2 + 10]
    assert plain[c].mean() > 50
    assert sil[c].mean() < 0.2 * plain[c].mean()

    # A ~4 px band just inside the sphere's outline must stay bright
    mask = plain > 20
    eroded = mask.copy()
    for _ in range(4):
        for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            eroded &= np.roll(eroded, shift, axis=axis)
    rim = mask & ~eroded
    assert rim.any()
    assert sil[rim].mean() > 5 * sil[c].mean()


def test_set_silhouette_toggle(mesh):
    from octarine.shaders import SilhouetteMeshMaterial

    v = oc.Viewer(offscreen=True, size=(300, 300))
    v.add_mesh(mesh, color="red")
    (obj,) = [o for objs in v.objects.values() for o in objs]
    color_before = tuple(obj.material.color)
    mode_before = obj.material.alpha_mode

    # Turning silhouette on swaps in the custom material (keeping the color)
    v.set_silhouette(2.0)
    assert isinstance(obj.material, SilhouetteMeshMaterial)
    assert obj.material.silhouette == 2.0
    assert obj.material.alpha_mode == "weighted_blend"
    assert tuple(obj.material.color) == color_before
    v.canvas.draw()

    # Turning it off restores the previous alpha mode
    v.set_silhouette(0)
    assert obj.material.silhouette == 0
    assert obj.material.alpha_mode == mode_before
    v.canvas.draw()
    v.close()


def _render_lit(mesh, material, light_z):
    """Render `mesh` with a single point light at (0, 0, light_z).

    Octarine's camera looks down +z, so a light at positive z sits *behind*
    the object as seen from the camera.
    """
    import pygfx as gfx

    v = oc.Viewer(offscreen=True, size=(200, 200))
    v.add_mesh(mesh, color="red")
    (obj,) = [o for objs in v.objects.values() for o in objs]
    if material is not None:
        obj.material = material
    for light in list(v.lights):  # drop the default lighting rig
        light.parent.remove(light)
    light = gfx.PointLight(intensity=30)
    light.local.position = (0, 0, light_z)
    v.scene.add(light)
    img = np.asarray(v.screenshot(filename=None, size=(200, 200)))[..., :3]
    v.close()
    return img.astype(float)


def test_subsurface_off_is_plain_phong(mesh):
    """A subsurface material with the effect off must render as plain Phong."""
    import pygfx as gfx
    from octarine.shaders import SubsurfaceMeshMaterial

    plain = _render_lit(mesh, gfx.MeshPhongMaterial(color="red"), -5)
    off = _render_lit(mesh, SubsurfaceMeshMaterial(subsurface=0, color="red"), -5)
    assert np.abs(plain - off).max() == 0


def test_subsurface_backlight_render(mesh):
    """Transmission must light up a backlit mesh, but not a front-lit one."""
    import pygfx as gfx
    from octarine.shaders import SubsurfaceMeshMaterial

    def sss():
        return SubsurfaceMeshMaterial(
            subsurface=3.0, color="red", thickness=1.0, wrap=0.0, glow=0.0
        )

    # Backlit: plain Phong leaves the visible side unlit, subsurface does not
    plain_back = _render_lit(mesh, gfx.MeshPhongMaterial(color="red"), 5)
    sss_back = _render_lit(mesh, sss(), 5)
    lit = sss_back.max(axis=-1) > 5
    assert lit.any()
    assert plain_back.max() < 5
    assert sss_back[lit].mean() > 20

    # Front-lit: the transmission lobe points away from the viewer, so the
    # image must be unchanged
    plain_front = _render_lit(mesh, gfx.MeshPhongMaterial(color="red"), -5)
    sss_front = _render_lit(mesh, sss(), -5)
    assert np.abs(plain_front - sss_front).max() == 0


def test_subsurface_wrap_softens_terminator(mesh):
    """Wrap must brighten the mesh past the terminator without moving the lit side."""
    import pygfx as gfx
    from octarine.shaders import SubsurfaceMeshMaterial

    # Light from the side, so the terminator runs across the visible face
    def render(material):
        v = oc.Viewer(offscreen=True, size=(200, 200))
        v.add_mesh(mesh, color="red")
        (obj,) = [o for objs in v.objects.values() for o in objs]
        obj.material = material
        for light in list(v.lights):
            light.parent.remove(light)
        light = gfx.PointLight(intensity=30)
        light.local.position = (-5, 0, -2)
        v.scene.add(light)
        img = np.asarray(v.screenshot(filename=None, size=(200, 200)))[..., :3]
        v.close()
        return img.astype(float).mean(axis=-1)

    plain = render(gfx.MeshPhongMaterial(color="red"))
    wrapped = render(
        SubsurfaceMeshMaterial(
            subsurface=1.0, color="red", wrap=1.0, thickness=0.0, glow=0.0
        )
    )

    # The shadowed half must gain light...
    dark = (plain > 0) & (plain < 10)  # on the mesh, but only dimly lit
    assert dark.any()
    assert wrapped[dark].mean() > plain[dark].mean() + 2

    # ... while wrapping never *removes* light. Checked away from the
    # outline only: pygfx resolves its 2x supersampling with a Mitchell
    # filter, whose negative lobes undershoot by ~1/255 next to an edge
    # that got brighter, which has nothing to do with the shading.
    interior = plain > 0
    for _ in range(3):
        for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            interior &= np.roll(interior, shift, axis=axis)
    assert interior.any()
    assert (wrapped[interior] >= plain[interior] - 1e-3).all()


def test_set_subsurface_toggle(mesh):
    from octarine.shaders import SubsurfaceMeshMaterial

    v = oc.Viewer(offscreen=True, size=(300, 300))
    v.add_mesh(mesh, color="red")
    (obj,) = [o for objs in v.objects.values() for o in objs]
    color_before = tuple(obj.material.color)
    mode_before = obj.material.alpha_mode

    # Turning it on swaps in the custom material, keeping color and, unlike
    # the silhouette, the alpha mode (the mesh stays opaque)
    v.set_subsurface(1.5, thickness=0.8, scatter_color="#00ff00")
    assert isinstance(obj.material, SubsurfaceMeshMaterial)
    assert obj.material.subsurface == pytest.approx(1.5)
    assert obj.material.thickness == pytest.approx(0.8)
    assert tuple(obj.material.scatter_color)[:3] == pytest.approx((0, 1, 0))
    assert obj.material.alpha_mode == mode_before
    assert tuple(obj.material.color) == color_before
    assert obj.material.silhouette == 0  # not turned on behind the user's back
    v.canvas.draw()

    v.set_subsurface(0)
    assert obj.material.subsurface == 0
    v.canvas.draw()
    v.close()


def test_set_subsurface_validation(mesh):
    v = oc.Viewer(offscreen=True, size=(300, 300))
    v.add_mesh(mesh, color="red")

    with pytest.raises(ValueError, match="subsurface must be >= 0"):
        v.set_subsurface(-1)
    with pytest.raises(ValueError, match="Unknown subsurface"):
        v.set_subsurface(1.0, sccatter_color="red")
    with pytest.raises(ValueError, match="Unknown subsurface"):
        v.add_mesh(mesh, subsurface={"subsurface": 1.0, "thicknes": 0.5})

    v.set_subsurface(1.0)
    (obj,) = [o for objs in v.objects.values() for o in objs]
    with pytest.raises(ValueError, match="falloff must be >= 1"):
        obj.material.falloff = 0.5
    with pytest.raises(ValueError, match="thickness must be >= 0"):
        obj.material.thickness = -0.1

    v.close()


def test_subsurface_controls_qt(mesh):
    """The subsurface checkbox/slider in the (Qt) controls panel."""
    # Note that the import can also fail if PySide6 *is* installed but clashes
    # with another Qt binding in the same environment - skip either way
    QtWidgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from octarine.controls import Controls
    from octarine.shaders import SubsurfaceMeshMaterial

    v = oc.Viewer(offscreen=True, size=(300, 300))
    v.add_mesh(mesh, color="red")
    (obj,) = [o for objs in v.objects.values() for o in objs]

    ctrl = Controls(v)
    assert not ctrl.subsurface_checkbox.isChecked()
    assert not ctrl.subsurface_slider.isEnabled()

    ctrl.subsurface_checkbox.setChecked(True)
    assert isinstance(obj.material, SubsurfaceMeshMaterial)
    assert obj.material.subsurface > 0
    assert ctrl.subsurface_slider.isEnabled()

    slider = ctrl.subsurface_slider
    slider.setValue(int(round(2.0 / slider._step)))
    assert obj.material.subsurface == pytest.approx(2.0, abs=0.05)

    ctrl.subsurface_checkbox.setChecked(False)
    assert obj.material.subsurface == 0
    assert not ctrl.subsurface_slider.isEnabled()

    # A panel built while the effect is active must reflect that
    v.set_subsurface(1.2)
    assert Controls(v).subsurface_checkbox.isChecked()

    v.close()


@pytest.mark.parametrize("order", ["sss_first", "silhouette_first"])
def test_subsurface_and_silhouette_compose(mesh, order):
    """The two effects must not clobber each other, in either order."""
    from octarine.shaders import SubsurfaceMeshMaterial

    v = oc.Viewer(offscreen=True, size=(300, 300))
    v.add_mesh(mesh, color="red")
    (obj,) = [o for objs in v.objects.values() for o in objs]
    mode_before = obj.material.alpha_mode

    if order == "sss_first":
        v.set_subsurface(1.5)
        v.set_silhouette(2.0)
    else:
        v.set_silhouette(2.0)
        v.set_subsurface(1.5)

    assert isinstance(obj.material, SubsurfaceMeshMaterial)
    assert obj.material.subsurface == pytest.approx(1.5)
    assert obj.material.silhouette == pytest.approx(2.0)
    assert obj.material.alpha_mode == "weighted_blend"
    v.canvas.draw()

    # Dropping the silhouette keeps the scattering and restores the alpha mode
    v.set_silhouette(0)
    assert obj.material.subsurface == pytest.approx(1.5)
    assert obj.material.silhouette == 0
    assert obj.material.alpha_mode == mode_before
    v.canvas.draw()
    v.close()


def _sharpness(img, mask):
    """Mean gradient magnitude over a masked region - a proxy for sharpness."""
    gy, gx = np.gradient(img.astype(float).mean(axis=-1))
    return np.hypot(gy, gx)[mask].mean()


def test_depth_of_field_render():
    """DoF must blur out-of-focus spheres but keep the focal plane sharp."""
    v = oc.Viewer(offscreen=True, size=(300, 300))
    for x, z, color, name in [
        (0, 0, "red", "near"),
        (-4, -30, "cyan", "far1"),
        (4, -30, "green", "far2"),
    ]:
        s = tm.creation.icosphere(subdivisions=3)
        s.apply_translation((x, 0, z))
        v.add_mesh(s, color=color, name=name)
    v.camera.show_object(v.scene, view_dir=(0, 0, -1), up=(0, 1, 0))

    def shot():
        return np.asarray(v.screenshot(filename=None))[..., :3]

    def dilate(mask, iters=3):
        for _ in range(iters):
            for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
                mask = mask | np.roll(mask, shift, axis=axis)
        return mask

    # Build a footprint mask for each sphere from the plain render (dilated
    # a little so they include the spheres' edges)
    plain = shot()
    r, g, b = plain[..., 0], plain[..., 1], plain[..., 2]
    near = dilate((r > 100) & (b < 80))  # red sphere
    far1 = dilate((b > 100) & (r < 80))  # cyan sphere
    far2 = dilate((g > 100) & (r < 80) & (b < 80))  # green sphere
    assert near.any() and far1.any() and far2.any()

    # Focus on the near sphere (at the origin). Note: for ortho cameras
    # pygfx places the camera mid-scene, so the focus distance is signed.
    p = v.camera.world.inverse_matrix @ np.array([0, 0, 0, 1.0])
    focus = -p[2] / p[3]
    # Ortho blur is normalized by the visible view height: aperture equal
    # to that height gives 1 px of blur per world unit of defocus - the far
    # spheres sit 30 units behind the focal plane, the near sphere spans <2.
    aperture = v.camera.height / v.camera.zoom
    v.set_depth_of_field(focus=focus, aperture=aperture, max_radius=20)
    dof = shot()

    assert _sharpness(dof, near) > 0.7 * _sharpness(plain, near)
    assert _sharpness(dof, far1) < 0.5 * _sharpness(plain, far1)
    assert _sharpness(dof, far2) < 0.5 * _sharpness(plain, far2)

    # Autofocus (focus=None) targets the center of the view, i.e. the near
    # sphere - so the result should be the same
    v.set_depth_of_field(aperture=aperture, max_radius=20)
    auto = shot()
    assert _sharpness(auto, near) > 0.7 * _sharpness(plain, near)
    assert _sharpness(auto, far1) < 0.5 * _sharpness(plain, far1)
    assert _sharpness(auto, far2) < 0.5 * _sharpness(plain, far2)

    # Disabling restores the original render
    v.set_depth_of_field(False)
    off = shot()
    assert np.allclose(off, plain, atol=2)
    v.close()


def test_depth_of_field_toggle(mesh):
    from octarine.shaders import DepthOfFieldPass

    v = oc.Viewer(offscreen=True, size=(200, 200))
    v.add_mesh(mesh, color="red")

    v.set_depth_of_field(focus=10, aperture=50, max_radius=8)
    passes = [p for p in v.renderer.effect_passes if isinstance(p, DepthOfFieldPass)]
    assert len(passes) == 1
    dof = passes[0]
    assert dof.enabled
    assert dof.focus == 10
    assert dof.aperture == 50
    assert dof.max_radius == 8
    v.canvas.draw()

    # Re-enabling updates the existing pass instead of adding a second one
    v.set_depth_of_field(aperture=25)
    assert [
        p for p in v.renderer.effect_passes if isinstance(p, DepthOfFieldPass)
    ] == [dof]
    assert dof.focus is None  # back to the autofocus default
    assert dof.aperture == 25

    # Disabling keeps the pass around but turns it off
    v.set_depth_of_field(False)
    assert not dof.enabled
    v.canvas.draw()

    with pytest.raises(ValueError):
        v.set_depth_of_field(aperture=-1)
    v.close()


def test_depth_of_field_focus_position():
    """get_focus_position must return the point the camera is focused on."""
    v = oc.Viewer(offscreen=True, size=(200, 200))
    s = tm.creation.icosphere(subdivisions=3)
    s.apply_translation((0, 0, -5))
    v.add_mesh(s, color="red")
    v.camera.show_object(v.scene, view_dir=(0, 0, -1), up=(0, 1, 0))

    # Autofocus: the focal point is the surface under the view center,
    # i.e. the front pole of the sphere at (0, 0, -4)
    v.set_depth_of_field()
    v.canvas.draw()  # the depth buffer only exists after a draw
    pos = v._dof_pass.get_focus_position(v.renderer)
    assert pos is not None
    assert np.allclose(pos, [0, 0, -4], atol=0.05)

    # Fixed focus: the focal point lies on the view axis at that distance
    p = v.camera.world.inverse_matrix @ np.array([0, 0, -4, 1.0])
    v.set_depth_of_field(focus=-p[2] / p[3])
    pos = v._dof_pass.get_focus_position(v.renderer)
    assert np.allclose(pos, [0, 0, -4], atol=1e-4)
    v.close()


def test_depth_of_field_smooth():
    """Smooth autofocus must ease towards a new target instead of snapping."""
    v = oc.Viewer(offscreen=True, size=(200, 200))
    s = tm.creation.icosphere(subdivisions=3)
    s.apply_translation((0, 0, -5))
    v.add_mesh(s, color="red")
    v.camera.show_object(v.scene, view_dir=(0, 0, -1), up=(0, 1, 0))

    v.set_depth_of_field(smooth=True)
    dof = v._dof_pass
    assert dof.smooth == 0.2
    assert dof.focus is None  # still reports autofocus

    # First frame: snaps straight to the initial target
    v.canvas.draw()
    start = dof._smooth_value
    assert start is not None and dof._smooth_settled
    assert dof._uniform_data["autofocus"] == 0.0

    # Move the sphere 10 units back -> the autofocus target jumps by 10;
    # after a single frame the focus must be strictly in between
    (obj,) = [o for objs in v.objects.values() for o in objs]
    obj.local.z -= 10
    v.canvas.draw()
    assert not dof._smooth_settled
    assert start < dof._smooth_value < start + 9.9

    # ... and settle on the new target if we keep rendering
    for _ in range(500):
        v.canvas.draw()
        if dof._smooth_settled:
            break
    assert dof._smooth_settled
    assert abs(dof._smooth_value - (start + 10)) < 0.1

    # Turning smoothing off reverts to per-fragment shader autofocus
    v.set_depth_of_field(smooth=False)
    v.canvas.draw()
    assert dof._uniform_data["autofocus"] == 1.0
    v.close()


def test_depth_of_field_snap():
    """Snapping autofocus must lock onto the closest object near the center."""
    v = oc.Viewer(offscreen=True, size=(200, 200))
    v.renderer.pixel_ratio = 1  # physical == logical pixels
    s = tm.creation.icosphere(subdivisions=3)
    v.add_mesh(s, color="red")
    v.camera.show_object(v.scene, view_dir=(0, 0, -1), up=(0, 1, 0))

    # Shift the sphere so its edge sits ~30 px right of the view center
    wpp = (v.camera.height / v.camera.zoom) / 200  # world units per pixel
    (obj,) = [o for objs in v.objects.values() for o in objs]
    shift = 1 + 30 * wpp
    obj.local.x = shift

    # Plain autofocus: nothing under the exact center -> stays sharp
    v.set_depth_of_field()
    v.canvas.draw()
    assert v._dof_pass._uniform_data["autofocus"] == 1.0
    assert v._dof_pass.get_focus_position(v.renderer) is None

    # Snapping: the closest point of the sphere is within reach ...
    v.set_depth_of_field(snap_radius=50)
    v.canvas.draw()
    dof = v._dof_pass
    assert dof._uniform_data["autofocus"] == 0.0
    pos = dof.get_focus_position(v.renderer)
    assert pos is not None
    # ... namely the sphere's rim nearest to the view center
    assert abs(pos[0] - (shift - 1)) < 5 * wpp
    assert abs(pos[1]) < 5 * wpp
    assert -0.05 < pos[2] < 0.45  # front surface right at the silhouette

    # A search radius smaller than the gap finds nothing -> sharp again
    v.set_depth_of_field(snap_radius=10)
    v.canvas.draw()
    assert dof._uniform_data["autofocus"] == 1.0
    assert dof.get_focus_position(v.renderer) is None
    v.close()


def _srgb_to_linear(img):
    """Screenshots are sRGB-encoded; occlusion is a linear-light factor."""
    c = np.asarray(img, dtype=float)[..., :3] / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).mean(axis=-1)


def _crease_scene(size=(400, 400)):
    """A floor and a wall meeting at a right angle, seen from above."""
    # The occlusion (and the shadows, which darken the same crease) are what
    # these tests switch on themselves, so start from a plain render
    v = oc.Viewer(offscreen=True, size=size, ambient_occlusion=False, shadows=False)
    # Anti-aliasing runs after the occlusion and reshuffles silhouette
    # pixels, which would muddy the comparisons below
    v.renderer.ppaa = "none"
    floor = tm.creation.box(extents=(20, 0.5, 20))
    floor.apply_translation((0, -0.25, 0))
    v.add_mesh(floor, color="w", name="floor")
    wall = tm.creation.box(extents=(20, 10, 0.5))
    wall.apply_translation((0, 5, -5))
    v.add_mesh(wall, color="w", name="wall")
    v.camera.show_object(v.scene, view_dir=(0, -0.6, -1), up=(0, 1, 0))
    return v


def test_ambient_occlusion_render():
    """AO must darken a right-angle crease and leave flat surfaces alone."""
    v = _crease_scene()
    # Screenshots are supersampled by default, and the filter that resamples
    # them has slight negative lobes: it overshoots at the crease, i.e. a few
    # pixels next to it come out *brighter* than in the plain render. That is
    # the filter's doing, not the occlusion's, so compare unfiltered renders
    shot = partial(v.screenshot, filename=None, supersample=1)
    plain = np.asarray(shot())[..., :3]

    # `debug` renders the occlusion itself: 1 = unoccluded
    v.set_ambient_occlusion(radius=3.0, samples=32, debug=True)
    ao = _srgb_to_linear(np.asarray(shot()))

    column = ao[:, ao.shape[1] // 2]
    # Half the hemisphere is blocked at a right-angle crease, so the
    # occlusion there tends to 0.5 (a finite radius keeps it above that)
    assert 0.45 < column.min() < 0.75
    # The open floor in the foreground must not occlude itself at all
    assert column[-60:].min() > 0.99

    # Without `debug` the scene is darkened but never brightened
    v.set_ambient_occlusion(radius=3.0, samples=32)
    delta = _srgb_to_linear(plain) - _srgb_to_linear(np.asarray(shot())[..., :3])
    assert delta.max() > 0.1  # the crease is visibly darker ...
    assert delta.min() >= 0  # ... and nothing got brighter

    # Disabling restores the original render
    v.set_ambient_occlusion(False)
    assert np.allclose(np.asarray(shot())[..., :3], plain, atol=2)
    v.close()


def test_ambient_occlusion_intensity():
    """More intensity (and more `power`) must mean more darkening."""
    v = _crease_scene(size=(200, 200))
    plain = _srgb_to_linear(np.asarray(v.screenshot(filename=None)))

    darkening = []
    for intensity in (0.0, 0.5, 1.0):
        v.set_ambient_occlusion(radius=3.0, intensity=intensity)
        shot = _srgb_to_linear(np.asarray(v.screenshot(filename=None)))
        darkening.append((plain - shot).sum())
    assert darkening[0] == pytest.approx(0, abs=1e-6)
    assert darkening[0] < darkening[1] < darkening[2]

    # `power` > 1 restricts the effect to the areas that are darkest anyway
    v.set_ambient_occlusion(radius=3.0, power=3.0)
    strong = _srgb_to_linear(np.asarray(v.screenshot(filename=None)))
    assert (plain - strong).sum() > darkening[2]
    v.close()


def test_ambient_occlusion_toggle(mesh):
    from octarine.shaders import AmbientOcclusionPass

    v = oc.Viewer(offscreen=True, size=(200, 200))
    v.add_mesh(mesh, color="red")

    v.set_ambient_occlusion(radius=0.5, intensity=0.8, samples=8, blur=False)
    passes = [
        p for p in v.renderer.effect_passes if isinstance(p, AmbientOcclusionPass)
    ]
    assert len(passes) == 1
    ao = passes[0]
    assert ao.enabled
    assert ao.radius == 0.5
    assert ao.intensity == 0.8
    assert ao.samples == 8
    assert ao.blur == 0
    # Occlusion is part of the shading, so it has to run before the
    # anti-aliasing and any lens effects
    assert v.renderer.effect_passes[0] is ao
    v.canvas.draw()

    # Re-enabling updates the existing pass instead of adding a second one
    v.set_ambient_occlusion(radius=0.25)
    assert [
        p for p in v.renderer.effect_passes if isinstance(p, AmbientOcclusionPass)
    ] == [ao]
    assert ao.radius == 0.25
    assert ao.blur == 2  # back to the default

    # Disabling keeps the pass around but turns it off
    v.set_ambient_occlusion(False)
    assert not ao.enabled
    v.canvas.draw()

    # `add_effect` shares the pass with `set_ambient_occlusion`
    v.add_effect("ao", intensity=0.4)
    assert v._ao_pass is ao
    assert ao.enabled and ao.intensity == 0.4
    v.canvas.draw()

    v.add_effect("ao", disable=True)
    assert v._ao_pass is None
    assert not [
        p for p in v.renderer.effect_passes if isinstance(p, AmbientOcclusionPass)
    ]

    # ... and can create it, too (with a radius derived from the scene)
    v.add_effect("ao")
    ao = v.renderer.effect_passes[0]
    assert isinstance(ao, AmbientOcclusionPass)
    assert ao.radius == pytest.approx(v._default_ao_radius())
    v.canvas.draw()

    for kwargs in ({"radius": 0}, {"intensity": -1}, {"samples": 0}, {"power": 0}):
        with pytest.raises(ValueError):
            v.set_ambient_occlusion(**kwargs)
    v.close()


def test_default_effects(mesh):
    """Shadows, headlight and ambient occlusion are on out of the box."""
    from octarine.shaders import AmbientOcclusionPass

    v = oc.Viewer(offscreen=True, size=(200, 200))
    assert v.shadows is True
    assert v.headlight is True
    assert isinstance(v._ao_pass, AmbientOcclusionPass)
    assert v._ao_pass.enabled
    assert v.renderer.effect_passes[0] is v._ao_pass

    # There is nothing on the canvas yet, so the occlusion radius has to catch
    # up with the scene as it fills (and shrink back again). Note that this
    # happens on the next frame, not on `add` itself - see
    # `Viewer._refresh_scene`.
    assert v._ao_pass.radius == 1.0
    v.add_mesh(mesh, name="small")
    v.canvas.draw()
    assert v._ao_pass.radius == pytest.approx(v._default_ao_radius())
    big = mesh.copy()
    big.apply_scale(100)
    v.add_mesh(big, name="big")
    v.canvas.draw()
    assert v._ao_pass.radius == pytest.approx(v._default_ao_radius())
    assert v._ao_pass.radius > 10
    v.pop()
    v.canvas.draw()
    assert v._ao_pass.radius == pytest.approx(v._default_ao_radius())

    # An explicit radius pins it, i.e. the scene no longer overrides it
    v.set_ambient_occlusion(radius=7.0)
    v.add_mesh(big, name="big2")
    v.canvas.draw()
    assert v._ao_pass.radius == 7.0
    v.close()

    # ... and all three can be switched off right from the start
    v = oc.Viewer(
        offscreen=True,
        size=(200, 200),
        shadows=False,
        headlight=False,
        ambient_occlusion=False,
    )
    assert v.shadows is False
    assert v.headlight is False
    assert v._ao_pass is None
    assert not any(
        isinstance(p, AmbientOcclusionPass) for p in v.renderer.effect_passes
    )
    v.add_mesh(mesh)
    assert not any(vis.cast_shadow for vis in v.visuals)
    v.canvas.draw()
    v.close()


def test_ambient_occlusion_default_radius(mesh):
    """The default radius must scale with the scene."""
    v = oc.Viewer(offscreen=True, size=(200, 200))
    assert v._default_ao_radius() == 1.0  # nothing on the canvas yet

    v.add_mesh(mesh, color="red", name="small")
    small = v._default_ao_radius()
    diagonal = float(np.linalg.norm(v.bounds[:, 1] - v.bounds[:, 0]))
    assert small == pytest.approx(diagonal * 0.04)

    big = mesh.copy()
    big.apply_scale(10)
    v.add_mesh(big, color="blue", name="big")
    assert v._default_ao_radius() > 5 * small
    v.close()


def test_effects_controls_qt(mesh):
    """The shadow + ambient occlusion controls in the (Qt) controls panel."""
    # Note that the import can also fail if PySide6 *is* installed but clashes
    # with another Qt binding in the same environment - skip either way
    QtWidgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from octarine.controls import Controls

    v = oc.Viewer(offscreen=True, size=(200, 200), shadows=False,
                  ambient_occlusion=False)
    v.add_mesh(mesh, color="red")

    ctrl = Controls(v)

    # Shadows are a plain on/off switch
    assert not ctrl.shadows_checkbox.isChecked()
    ctrl.shadows_checkbox.setChecked(True)
    assert v.shadows
    ctrl.shadows_checkbox.setChecked(False)
    assert not v.shadows

    # Ambient occlusion: settings are disabled (and the section collapsed)
    # until the effect is switched on
    assert not ctrl.ao_section.isExpanded()
    assert not ctrl.ao_radius_slider.isEnabled()

    ctrl.ao_checkbox.setChecked(True)
    assert ctrl.ao_section.isExpanded()
    assert ctrl.ao_radius_slider.isEnabled()
    ao = v._ao_pass
    assert ao is not None and ao.enabled
    # The radius is in world units, so it must be fitted to the scene
    assert ao.radius == pytest.approx(v._default_ao_radius(), rel=0.02)

    # ... and it must follow the scene as long as the slider is untouched
    slider = ctrl.ao_radius_slider
    big = mesh.copy()
    big.apply_scale(100)
    v.add_mesh(big, name="big")
    assert ao.radius == pytest.approx(v._default_ao_radius())
    assert slider.value() * slider._step == pytest.approx(ao.radius, rel=0.02)
    v.pop()  # ... and shrink back with it
    assert ao.radius == pytest.approx(v._default_ao_radius())

    ctrl.ao_intensity_slider.setValue(
        round(0.5 / ctrl.ao_intensity_slider._step)
    )
    assert ao.intensity == pytest.approx(0.5)
    ctrl.ao_samples_slider.setValue(round(32 / ctrl.ao_samples_slider._step))
    assert ao.samples == 32
    ctrl.ao_blur_checkbox.setChecked(False)
    assert ao.blur == 0
    ctrl.ao_debug_checkbox.setChecked(True)
    assert ao.debug
    v.canvas.draw()

    # The extremes of the radius slider must both be usable (a radius of 0
    # is not a thing)
    slider = ctrl.ao_radius_slider
    slider.setValue(slider.minimum())
    assert ao.radius > 0
    slider.setValue(slider.maximum())
    assert ao.radius == pytest.approx(slider.maximum() * slider._step)
    radius = ao.radius

    # A hand-picked radius must survive an off/on cycle
    ctrl.ao_checkbox.setChecked(False)
    assert not ao.enabled
    assert not ctrl.ao_radius_slider.isEnabled()
    ctrl.ao_checkbox.setChecked(True)
    assert ao.enabled and ao.radius == pytest.approx(radius, rel=0.02)
    v.canvas.draw()

    # A panel built while the effects are active must reflect that
    v.shadows = True
    ctrl2 = Controls(v)
    assert ctrl2.shadows_checkbox.isChecked()
    assert ctrl2.ao_checkbox.isChecked()
    assert ctrl2.ao_radius_slider.isEnabled()
    assert ctrl2.ao_samples_slider.value() * ctrl2.ao_samples_slider._step == 32
    assert not ctrl2.ao_blur_checkbox.isChecked()
    assert ctrl2.ao_debug_checkbox.isChecked()
    # The radius was hand-picked, so it must not be overwritten from the scene
    assert ctrl2._ao_radius_touched
    assert ao.radius == pytest.approx(radius, rel=0.02)

    v.close()

    # With the effects on by default the panel comes up switched on - but
    # still collapsed, i.e. the state of an effect must not expand its section
    v = oc.Viewer(offscreen=True, size=(200, 200))
    v.add_mesh(mesh, color="red")
    ctrl = Controls(v)
    assert ctrl.shadows_checkbox.isChecked()
    assert ctrl.ao_checkbox.isChecked()
    assert not ctrl.ao_section.isExpanded()
    assert ctrl.ao_radius_slider.isEnabled()
    # The viewer derives the radius itself, so the panel must not pin it
    assert not ctrl._ao_radius_touched
    slider = ctrl.ao_radius_slider
    assert slider.value() * slider._step == pytest.approx(
        v._ao_pass.radius, rel=0.02
    )
    v.close()


def _render_point_view(v, cam_height):
    """Render with an ortho camera showing `cam_height` world units.

    With a 300x300 canvas (pixel_ratio 1) the world-to-pixel ratio
    is 300 / cam_height.
    """
    cam = v.camera
    cam.width = cam_height
    cam.height = cam_height
    cam.zoom = 1
    cam.depth_range = (1, 1000)
    cam.local.position = (0, 0, 50)
    cam.look_at((0, 0, 0))
    return np.asarray(v.screenshot(filename=None, size=(300, 300)))


def _color_mask(img, channel):
    """Boolean mask of pixels dominated by the given RGB channel."""
    rgb = img[..., :3].astype(int)
    other = [c for c in range(3) if c != channel]
    return (
        (rgb[..., channel] > 120)
        & (rgb[..., other[0]] < 100)
        & (rgb[..., other[1]] < 100)
    )


def test_points2gfx_material_selection():
    """Only requests for flex features should switch to the custom material."""
    import pygfx as gfx
    from octarine.visuals import points2gfx
    from octarine.shaders import FlexPointsMaterial

    pts = np.zeros((2, 3), dtype=np.float32)

    # Stock materials when no flex feature is requested
    assert type(points2gfx(pts, color="red").material) is gfx.PointsMaterial
    assert (
        type(points2gfx(pts, color="red", marker="ring").material)
        is gfx.PointsMarkerMaterial
    )

    # Edge styling alone upgrades to a (stock) marker material
    mat = points2gfx(pts, color="red", edge_width=5).material
    assert type(mat) is gfx.PointsMarkerMaterial
    assert mat.edge_width == 5

    # ... and edge_mode is a stock passthrough as well
    mat = points2gfx(pts, color="red", marker="ring", edge_mode="outer").material
    assert type(mat) is gfx.PointsMarkerMaterial
    assert mat.edge_mode == "outer"

    # Flex features switch to the custom material; plain points get an
    # invisible edge so they still look like gfx.PointsMaterial points
    mat = points2gfx(pts, color="red", min_size=10).material
    assert isinstance(mat, FlexPointsMaterial)
    assert mat.min_size == 10
    assert mat.max_size is None
    assert mat.edge_width == 0

    mat = points2gfx(
        pts,
        color="red",
        marker="ring",
        size_space="world",
        edge_size_space="screen",
        max_size=100,
        min_edge_width=2,
    ).material
    assert isinstance(mat, FlexPointsMaterial)
    assert mat.edge_size_space == "screen"
    assert mat.max_size == 100
    assert mat.min_edge_width == 2


def test_points_min_max_size():
    """min_size/max_size must clamp the on-screen size of world-space points."""
    from octarine.shaders import FlexPointsMaterial

    v = oc.Viewer(offscreen=True, size=(300, 300))
    v.renderer.pixel_ratio = 1
    v.add_points(
        np.zeros((1, 3), dtype=np.float32),
        color="red",
        size=4,
        size_space="world",
        min_size=0,
        center=False,
    )
    (obj,) = [o for objs in v.objects.values() for o in objs]
    assert isinstance(obj.material, FlexPointsMaterial)

    def diameter(img):
        return 2 * np.sqrt(_color_mask(img, 0).sum() / np.pi)

    # At 1 px per world unit the point covers ~4 px
    assert abs(diameter(_render_point_view(v, 300)) - 4) < 2

    # ... but never fewer than min_size (a live uniform update, no recompile)
    obj.material.min_size = 40
    assert abs(diameter(_render_point_view(v, 300)) - 40) < 3

    # ... and never more than max_size (at 10 px/world-unit it would be 40 px)
    obj.material.min_size = None
    obj.material.max_size = 10
    assert abs(diameter(_render_point_view(v, 30)) - 10) < 3
    v.close()


def _edge_runs(img):
    """Lengths of contiguous red runs along the horizontal centerline."""
    red = _color_mask(img, 0)[img.shape[0] // 2].astype(int)
    steps = np.diff(np.r_[0, red, 0])
    return list(np.where(steps == -1)[0] - np.where(steps == 1)[0])


def test_points_edge_size_space():
    """A screen-space edge must keep its pixel width as the marker scales."""

    def make_viewer(edge_size_space):
        v = oc.Viewer(offscreen=True, size=(300, 300))
        v.renderer.pixel_ratio = 1
        v.add_points(
            np.zeros((1, 3), dtype=np.float32),
            color="blue",
            marker="circle",
            size=60,
            size_space="world",
            edge_size_space=edge_size_space,
            edge_width=8,
            edge_color="red",
            center=False,
        )
        return v

    # Screen-space edge: 8 px whether the disc is 60 or 120 px wide
    v = make_viewer("screen")
    for cam_height in (300, 150):
        runs = _edge_runs(_render_point_view(v, cam_height))
        assert len(runs) == 2  # crossing the left and right rim
        assert all(abs(r - 8) <= 2 for r in runs)
    v.close()

    # Control: a world-space edge doubles along with the disc
    v = make_viewer("world")
    for cam_height, expected in ((300, 8), (150, 16)):
        runs = _edge_runs(_render_point_view(v, cam_height))
        assert len(runs) == 2
        assert all(abs(r - expected) <= 2 for r in runs)
    v.close()


def test_points_min_edge_width():
    """min_edge_width must floor the on-screen width of a world-space edge."""
    v = oc.Viewer(offscreen=True, size=(300, 300))
    v.renderer.pixel_ratio = 1
    v.add_points(
        np.zeros((1, 3), dtype=np.float32),
        color="blue",
        marker="circle",
        size=120,
        size_space="world",
        edge_width=8,
        min_edge_width=6,
        edge_color="red",
        center=False,
    )
    (obj,) = [o for objs in v.objects.values() for o in objs]

    # At 1 px/world-unit the edge is well above the floor -> 8 px
    runs = _edge_runs(_render_point_view(v, 300))
    assert len(runs) == 2
    assert all(abs(r - 8) <= 1 for r in runs)

    # At 0.5 px/world-unit it would shrink to 4 px -> floored at 6 px
    runs = _edge_runs(_render_point_view(v, 600))
    assert len(runs) == 2
    assert all(abs(r - 6) <= 1 for r in runs)

    # Removing the floor (a live uniform update) lets it shrink to 4 px
    obj.material.min_edge_width = None
    runs = _edge_runs(_render_point_view(v, 600))
    assert len(runs) == 2
    assert all(abs(r - 4) <= 1 for r in runs)
    v.close()


def test_points_edge_mode():
    """edge_mode must place the edge inside/astride/outside the outline."""
    for mode, expected in (("inner", 60), ("centered", 70), ("outer", 80)):
        v = oc.Viewer(offscreen=True, size=(300, 300))
        v.renderer.pixel_ratio = 1
        v.add_points(
            np.zeros((1, 3), dtype=np.float32),
            color="blue",
            marker="circle",
            size=60,
            edge_width=10,
            edge_color="red",
            edge_mode=mode,
            center=False,
        )
        img = _render_point_view(v, 300)
        footprint = _color_mask(img, 0) | _color_mask(img, 2)
        diam = 2 * np.sqrt(footprint.sum() / np.pi)
        assert abs(diam - expected) < 3, mode
        v.close()


def test_points_size_space_combos():
    """All size_space x edge_size_space combos must compile and render."""
    v = oc.Viewer(offscreen=True, size=(300, 300))
    v.renderer.pixel_ratio = 1
    centers = []
    for i, size_space in enumerate(("screen", "world", "model")):
        for j, edge_space in enumerate((None, "screen", "world", "model")):
            pos = np.array([[j * 60 - 90, i * 60 - 60, 0]], dtype=np.float32)
            v.add_points(
                pos,
                color="red",
                marker="diamond",
                size=20,
                size_space=size_space,
                edge_size_space=edge_space,
                edge_width=3,
                edge_color="white",
                min_size=5,
                center=False,
            )
            centers.append(pos[0, :2])

    # The camera shows 300 world units -> 1 px per world unit
    red = _color_mask(_render_point_view(v, 300), 0)
    h, w = red.shape
    for cx, cy in centers:
        px, py = int(w / 2 + cx), int(h / 2 - cy)
        assert red[py - 15 : py + 15, px - 15 : px + 15].sum() > 20
    v.close()


def _scalebar_box(v, size=(400, 300)):
    """Return (canvas_size, (row, x0, x1)) of the rendered scale bar.

    Expects the scene itself to contain nothing white so that the bar can be
    told apart from the objects.

    """
    img = np.asarray(v.screenshot(filename=None, size=size, alpha=False))
    rgb = img[..., :3].astype(int)
    white = (rgb.min(axis=-1) > 200) & (np.ptp(rgb, axis=-1) < 30)
    h, w = white.shape
    # The bar is the longest uninterrupted horizontal run of white pixels
    best = None
    for r in range(h):
        cols = np.where(white[r])[0]
        if len(cols) > 20 and (cols.max() - cols.min()) == len(cols) - 1:
            if best is None or (cols.max() - cols.min()) > (best[2] - best[1]):
                best = (r, int(cols.min()), int(cols.max()))
    return (w, h), best


def test_lines2gfx_material_selection():
    """Only per-point widths should switch to the custom line material."""
    import pygfx as gfx
    from octarine.visuals import lines2gfx
    from octarine.shaders import FlexLineMaterial

    line = np.zeros((4, 3), dtype=np.float32)

    # Stock materials for a single width
    assert type(lines2gfx(line, color="red", linewidth=2).material) is gfx.LineMaterial
    assert (
        type(lines2gfx(line, color="red", linewidth=0).material)
        is gfx.LineThinMaterial
    )

    # An array of widths switches to the custom material. Its uniform
    # thickness is the mean width - that's what dashes are scaled with.
    mat = lines2gfx(line, color="red", linewidth=[1, 2, 3, 4]).material
    assert isinstance(mat, FlexLineMaterial)
    assert mat.thickness_mode == "vertex"
    assert mat.thickness == 2.5

    # Widths for a stack of lines are padded at the NaN breaks
    lines = [np.zeros((3, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.float32)]
    vis = lines2gfx(lines, color="red", linewidth=[1, 2, 3, 4, 5])
    assert vis.geometry.thicknesses.nitems == vis.geometry.positions.nitems == 6
    assert list(vis.geometry.thicknesses.data) == [1, 2, 3, 0, 4, 5]

    # ... but widths that already include the breaks are fine too
    vis = lines2gfx(lines, color="red", linewidth=np.ones(6))
    assert vis.geometry.thicknesses.nitems == 6

    with pytest.raises(ValueError):
        lines2gfx(lines, color="red", linewidth=[1, 2, 3])
    with pytest.raises(ValueError):
        lines2gfx(line, color="red", linewidth=[-1, 1, 1, 1])


def test_lines_per_point_width_render():
    """Per-point widths must taper the rendered line."""
    from octarine.shaders import FlexLineMaterial

    v = oc.Viewer(offscreen=True, size=(400, 400))
    v.renderer.pixel_ratio = 1

    # A horizontal line from x=-150 to x=150, tapering from 2 to 40 units
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[:, 0] = np.linspace(-150, 150, 21)
    v.add_lines(
        pts,
        color="red",
        linewidth=np.linspace(2, 40, 21),
        linewidth_space="world",
        center=False,
    )
    (obj,) = [o for objs in v.objects.values() for o in objs]
    assert isinstance(obj.material, FlexLineMaterial)

    cam = v.camera
    cam.width = cam.height = 400  # 1 px per world unit
    cam.zoom = 1
    cam.depth_range = (1, 1000)
    cam.local.position = (0, 0, 50)
    cam.look_at((0, 0, 0))

    def heights(img):
        """Number of red pixels per column."""
        return _color_mask(img, 0).sum(axis=0)

    h = heights(np.asarray(v.screenshot(filename=None, size=(400, 400))))
    # The line spans columns 50-350; width at column c is 2 + 38 * (c - 50) / 300
    for col in (60, 200, 340):
        expected = 2 + 38 * (col - 50) / 300
        assert abs(h[col] - expected) <= 2, f"{h[col]} px at column {col}"

    # Switching back to a uniform width renders a constant-width line
    obj.material.thickness_mode = "uniform"
    h = heights(np.asarray(v.screenshot(filename=None, size=(400, 400))))
    assert abs(h[60] - h[340]) <= 2
    assert abs(h[200] - obj.material.thickness) <= 2
    v.close()


def test_scalebar(mesh):
    v = oc.Viewer(offscreen=True, size=(400, 300))
    v.add_mesh(mesh, color="red")

    # A fixed-size bar must span the expected fraction of the canvas
    v.set_scalebar(0.5, units="nm")
    (w, h), (row, x0, x1) = _scalebar_box(v)
    visible_width = 2 / v.camera.projection_matrix[0, 0]
    assert abs((x1 - x0) - 0.5 / visible_width * w) < 3
    assert v._scalebar._label == "0.5 nm"
    # ... and sit `margin` (logical) pixels from the bottom-right corner
    assert abs((w - x1) - 20 * w / 400) < 3
    assert abs((h - row) - 20 * h / 300) < 5

    # The other corners must mirror that
    v.set_scalebar(0.5, position="top-left")
    (w2, h2), (row2, x2, x3) = _scalebar_box(v)
    assert (w2, h2) == (w, h)
    assert abs(x2 - (w - x1)) < 3 and abs((x3 - x2) - (x1 - x0)) < 3
    assert abs(row2 - (h - row)) < 5

    # An "auto" bar picks a nice round number covering ~a quarter of the canvas
    v.set_scalebar(units="nm")
    for zoom, expected in ((1, "1 nm"), (2, "0.5 nm"), (10, "0.1 nm")):
        v.camera.zoom = zoom
        _, (_, x0, x1) = _scalebar_box(v)
        assert v._scalebar._label == expected, zoom
        assert 0.1 < (x1 - x0) / w < 0.3, zoom
    v.camera.zoom = 1

    # Labels can be switched off or overridden
    v.set_scalebar(0.5, label=False)
    assert not v._scalebar._text.visible
    v.set_scalebar(0.5, label="custom")
    assert v._scalebar._text.visible and v._scalebar._label == "custom"

    # Removing (also repeatedly) must clear the overlay
    v.set_scalebar(False)
    assert v._scalebar is None
    assert not len(v.overlay_scene.children)
    v.set_scalebar(False)

    # Removing and re-adding between two frames must not kill the animation
    v.set_scalebar(0.5)
    v.set_scalebar(False)
    v.set_scalebar(0.5)
    v.canvas.draw()
    v.canvas.draw()
    assert v._update_scalebar in v._animations
    assert v._scalebar._bar.geometry.positions.data.any()

    with pytest.raises(ValueError, match="auto"):
        v.set_scalebar("nope")
    with pytest.raises(ValueError, match="positive"):
        v.set_scalebar(-1)
    with pytest.raises(ValueError, match="Unknown position"):
        v.set_scalebar(0.5, position="middle")

    v.close()


def test_scalebar_requires_orthographic_camera(mesh):
    v = oc.Viewer(offscreen=True, size=(400, 300), camera="perspective")
    v.add_mesh(mesh)

    with pytest.raises(ValueError, match="orthographic"):
        v.set_scalebar(0.5)

    # Switching to a perspective camera after the fact hides the bar
    v.camera.fov = 0
    v.set_scalebar(0.5)
    v.canvas.draw()
    assert v._scalebar.visible
    v.camera.fov = 50
    v.canvas.draw()
    assert not v._scalebar.visible
    v.camera.fov = 0
    v.canvas.draw()
    assert v._scalebar.visible

    v.close()


def test_headlight_toggle(mesh):
    v = oc.Viewer(offscreen=True, size=(200, 200))
    v.add_mesh(mesh)

    # The headlight is on by default, which switches the static lights off
    assert v.headlight is True
    assert v._headlight.visible is True
    assert not any(light.visible for light in v._static_lights)

    # The headlight follows the camera but is a plain scene child (see
    # `Viewer._update_headlight`)
    assert v._headlight.parent is v.scene
    assert v._headlight in v.lights

    v.headlight = False
    assert v._headlight.visible is False
    assert all(light.visible for light in v._static_lights)

    # Shadows must reach the headlight, too
    v.shadows = False
    assert not v._headlight.cast_shadow
    v.shadows = True
    assert v._headlight.cast_shadow is True
    assert all(light.cast_shadow for light in v._static_lights)

    v.toggle_headlight()
    assert v.headlight is True
    assert v._headlight.visible is True
    assert not any(light.visible for light in v._static_lights)

    with pytest.raises(TypeError):
        v.headlight = "yes"

    # ... and it can be switched off right from the start
    v2 = oc.Viewer(offscreen=True, size=(200, 200), headlight=False)
    assert v2.headlight is False
    assert all(light.visible for light in v2._static_lights)
    v2.close()

    v.close()


def _shadow_scene(scale=1.0):
    """A ground plane with a sphere floating above it."""
    plane = tm.creation.box((20 * scale, 0.2 * scale, 20 * scale))
    plane.apply_translation((0, -3 * scale, 0))
    sphere = tm.creation.icosphere(radius=2 * scale, subdivisions=3)
    sphere.apply_translation((0, 2 * scale, 0))
    return plane, sphere


def _shadow_viewer(shadows=True, scale=1.0, headlight=False, camera="ortho",
                   size=(200, 200), plain=False):
    """An open viewer on the scene above, framed so the shadow is in view.

    `plain` switches off the occlusion and the anti-aliasing, for tests that
    measure the shadow itself rather than the whole picture.

    """
    plane, sphere = _shadow_scene(scale)
    v = oc.Viewer(offscreen=True, size=size, camera=camera,
                  ambient_occlusion=not plain)
    if plain:
        v.renderer.ppaa = "none"
    v.headlight = headlight
    v.add_mesh(plane, color="white")
    v.add_mesh(sphere, color="red")
    v.shadows = shadows
    # Look at the plane at a slant so the sphere's shadow falls into view
    v.camera.show_object(v.scene, view_dir=(0, -0.6, -1), up=(0, 1, 0))
    return v


def _render_shadow_scene(shadows, scale=1.0, headlight=False, camera="ortho"):
    """Render the scene above and return the greyscale image."""
    v = _shadow_viewer(shadows, scale, headlight, camera)
    img = np.asarray(v.screenshot(filename=None, size=(200, 200)))[..., :3]
    v.close()
    return img.astype(float).mean(axis=-1)


@pytest.mark.parametrize("camera", ["ortho", "perspective"])
@pytest.mark.parametrize("headlight", [False, True])
# The scene scale matters: pygfx sizes neither the lights nor their shadow
# cameras to the scene, so without our own fitting the shadow map is empty
@pytest.mark.parametrize("scale", [0.01, 1.0, 1000.0])
def test_shadow_render(scale, headlight, camera):
    """Switching shadows on must actually darken part of the plane."""
    without = _render_shadow_scene(False, scale, headlight, camera)
    with_ = _render_shadow_scene(True, scale, headlight, camera)

    # The sphere's shadow has to show up as a decent patch of the plane going
    # properly dark - the generous threshold keeps the faint (<10/255) banding
    # you get from PCF on a plane lit edge-on out of the mask
    darkened = (without - with_) > 20
    assert darkened.sum() > 100

    # ... and it has to be a solid patch rather than the speckle you get from
    # shadow acne, so it has to survive an erosion. A solid blob only loses its
    # perimeter here (~20%), while acne - which is a checkerboard - loses
    # basically all of itself.
    eroded = darkened.copy()
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        eroded &= np.roll(darkened, shift, axis=axis)
    assert eroded.sum() > darkened.sum() * 0.6


def _resolved_shadow_wgsl():
    """What `{$ include 'pygfx.light_shadow.wgsl' $}` currently resolves to."""
    from pygfx.renderers.wgpu.shader.templating import apply_templating

    return apply_templating("{$ include 'pygfx.light_shadow.wgsl' $}")


def test_shadow_pcf_installed():
    """Creating a viewer must swap in our wider shadow filter."""
    from octarine.shaders import pcf

    v = oc.Viewer(offscreen=True, size=(50, 50))
    code = _resolved_shadow_wgsl()
    v.close()

    # Both light paths now sample a disk of taps ...
    assert f"const PCF_DISK = array<vec2<f32>, {pcf.PCF_TAPS}>" in code
    assert f"const PCF_CUBE_DISK = array<vec2<f32>, {pcf.PCF_TAPS_CUBE}>" in code
    # ... and none of pygfx' fixed texel offsets survive
    assert "vec2<i32>" not in code

    # Installing again must not stack another loader on top
    pcf.install()
    assert _resolved_shadow_wgsl() == code


def test_shadow_pcf_patch_leaves_the_rest_of_pygfx_alone():
    """We must swap the tap loops only, so upstream fixes still reach us."""
    from pygfx.renderers.wgpu.wgsl import load_wgsl
    from octarine.shaders import pcf

    stock = load_wgsl(pcf._SNIPPET)
    patched = pcf._patch(stock)

    # Everything pygfx' light_punctual.wgsl calls into is still declared, and
    # the projection/face-selection code around the taps is untouched
    for line in stock.splitlines():
        if line.startswith("fn ") or "proj_coords" in line or "faceIndex =" in line:
            assert line in patched


def test_shadow_pcf_rejects_unrecognised_pygfx():
    """A shadow shader we can't find the tap loops in has to fail, not silently pass."""
    from octarine.shaders import pcf

    with pytest.raises(RuntimeError, match="directional tap kernel"):
        pcf._patch("fn get_shadow() { return 1.0; }")


def test_shadow_pcf_skips_incompatible_pygfx(monkeypatch, caplog):
    """... and `install()` must degrade rather than break rendering."""
    from octarine.shaders import pcf
    from pygfx.renderers.wgpu.shader import templating

    before = templating.root_loader.mapping[pcf._CONTEXT]
    monkeypatch.setattr(pcf, "_installed", False)
    monkeypatch.setattr(
        "pygfx.renderers.wgpu.wgsl.load_wgsl", lambda *a, **kw: "fn something_else() {}"
    )

    with caplog.at_level("WARNING"):
        pcf.install()

    assert "shadow filtering" in caplog.text.lower()
    # The loader must be left exactly as it was, and we must not try again
    assert templating.root_loader.mapping[pcf._CONTEXT] is before
    assert pcf._installed is True


def test_shadow_pcf_configure_is_too_late_once_installed():
    """A knob that silently does nothing is worse than no knob."""
    from octarine.shaders import pcf

    oc.Viewer(offscreen=True, size=(50, 50)).close()
    with pytest.raises(RuntimeError, match="before the first Viewer"):
        pcf.configure(taps=8)


@pytest.mark.parametrize("headlight", [False, True])
def test_shadow_edge_is_filtered(headlight):
    """The shadow must fade over several pixels rather than step from lit to dark.

    That gradient is the whole point of the wider PCF kernel: it is what keeps
    the edge from crawling as the shadow map's texel grid turns with the light
    (see `octarine.shaders.pcf`).

    """
    plane, sphere = _shadow_scene()
    v = _shadow_viewer(headlight=headlight, size=(400, 400), plain=True)
    v.camera.zoom = 4
    v.screenshot(filename=None, size=(400, 400))  # settles the lights

    # Follow the light from the sphere's centre down to the plane's top face to
    # find where its shadow lands, then centre on that. Panning the camera does
    # not turn it, so the lighting is untouched.
    centre = sphere.bounds.mean(axis=0)
    light = v._headlight if headlight else v._static_lights[0]
    d = centre - np.asarray(light.world.position)
    d /= np.linalg.norm(d)
    target = centre + d * ((plane.bounds[1][1] - centre[1]) / d[1])
    pos = np.asarray(v.camera.local.position, dtype=float)
    fwd = np.asarray(v.camera.world.forward, dtype=float)
    to_target = target - pos
    v.camera.local.position = pos + (to_target - np.dot(to_target, fwd) * fwd)

    img = np.asarray(v.screenshot(filename=None, size=(400, 400)))[..., :3]
    img = img.astype(float).mean(axis=-1)
    v.close()

    # Walk out of the shadow: the row through the middle now starts inside it
    row = img[img.shape[0] // 2][200:]
    dark, lit = row.min(), row.max()
    # There has to be a shadow to look at in the first place. The contrast is
    # lower with the static lights because each one lights what the other
    # shadows, so this threshold is deliberately modest.
    assert lit - dark > 25
    lo, hi = dark + 0.2 * (lit - dark), dark + 0.8 * (lit - dark)

    outside = np.where(row >= hi)[0]
    assert len(outside)
    inside = np.where(row[: outside[0]] <= lo)[0]
    assert len(inside)

    # Unfiltered this transition is 1-2 px and pygfx' own kernel gives ~5 at
    # this scale. Ours has to be clearly wider, without smearing the shadow
    # away entirely.
    width = outside[0] - inside[-1]
    assert 8 < width < 40


def test_shadow_flags(mesh, points, line_single):
    v = oc.Viewer(offscreen=True, size=(200, 200), shadows=False)
    v.add_mesh(mesh)
    v.add_points(points)
    v.add_lines(line_single)
    # pygfx cannot render a volume into a shadow map and raises if we ask it to
    v.add_volume(np.random.rand(10, 10, 10).astype(np.float32))
    v.canvas.draw()  # the flags are applied per frame; see `_refresh_scene`

    assert v.shadows is False
    assert not any(vis.cast_shadow for vis in v.visuals)

    v.shadows = True
    by_type = {type(vis).__name__: vis for vis in v.visuals}
    assert by_type["Mesh"].cast_shadow is True
    assert by_type["Points"].cast_shadow is True
    assert by_type["Line"].cast_shadow is True
    assert by_type["Volume"].cast_shadow is False

    # Only meshes ever receive shadows
    assert by_type["Mesh"].receive_shadow is True
    assert not any(
        vis.receive_shadow for name, vis in by_type.items() if name != "Mesh"
    )

    # This must not raise (it does if a volume ends up in the shadow pass)
    v.screenshot(filename=None, size=(200, 200))

    v.shadows = False
    assert not any(vis.cast_shadow or vis.receive_shadow for vis in v.visuals)

    with pytest.raises(TypeError):
        v.shadows = "yes"

    v.close()

    # Shadows are on by default, and objects have to pick that up as they are
    # added rather than only when the flag is flipped
    v = oc.Viewer(offscreen=True, size=(200, 200))
    assert v.shadows is True
    v.add_mesh(mesh)
    v.canvas.draw()
    (vis,) = v.visuals
    assert vis.cast_shadow is True
    assert vis.receive_shadow is True
    v.close()


def test_shadow_lights_track_the_scene(mesh):
    """The lights must be re-fitted as objects come and go."""
    v = oc.Viewer(offscreen=True, size=(200, 200), shadows=False)
    v.add_mesh(mesh)  # unit sphere at the origin

    # While shadows are off the static lights stay parked far away, which is
    # what makes their light effectively parallel
    parked = np.array([tuple(light.local.position) for light in v._static_lights])
    assert np.abs(parked).min() == pytest.approx(1e6)

    v.shadows = True
    near = np.array([tuple(light.local.position) for light in v._static_lights])
    radius = np.linalg.norm(np.ptp(v.scene.get_bounding_box(), axis=0)) / 2
    assert np.linalg.norm(near, axis=1) == pytest.approx(
        [radius * oc.viewer.SHADOW_LIGHT_DISTANCE] * 2
    )

    # Growing the scene has to move the lights out with it... (on the next
    # frame, that is - see `Viewer._refresh_scene`)
    big = tm.creation.icosphere(radius=1000)
    big.apply_translation((5000, 0, 0))
    v.add_mesh(big)
    v.canvas.draw()
    grown = np.array([tuple(light.local.position) for light in v._static_lights])
    assert np.linalg.norm(grown, axis=1).min() > np.linalg.norm(near, axis=1).max()
    # ... and the lights have to follow the scene's center, not sit at the origin
    center = np.asarray(v.scene.get_bounding_box(), dtype=float).mean(axis=0)
    assert grown.mean(axis=0) == pytest.approx(center)

    # ... and shrinking it again has to bring them back
    v.pop()
    v.canvas.draw()
    shrunk = np.array([tuple(light.local.position) for light in v._static_lights])
    assert shrunk == pytest.approx(near)

    # Switching shadows off parks them where they started
    v.shadows = False
    assert np.array(
        [tuple(light.local.position) for light in v._static_lights]
    ) == pytest.approx(parked)

    v.close()


def test_scene_refit_is_deferred(mesh):
    """The scene is re-fitted once per frame, not once per object added.

    Every one of those fits walks all visuals, so doing it per object made
    filling a viewer quadratic in the number of objects.
    """
    v = oc.Viewer(offscreen=True, size=(200, 200))

    fits = []
    wrapped = v._update_shadows

    def _counting_update_shadows(*args, **kwargs):
        fits.append(1)
        return wrapped(*args, **kwargs)

    v._update_shadows = _counting_update_shadows

    for i in range(5):
        v.add_mesh(mesh.copy(), name=f"mesh{i}")
    assert v._scene_stale and not fits  # nothing fitted yet

    v.canvas.draw()
    assert not v._scene_stale and len(fits) == 1

    # ... and a frame that did not change the scene must not re-fit at all
    v.canvas.draw()
    assert len(fits) == 1

    v.close()


def test_center_camera_is_deferred(mesh):
    """Centering happens on the next frame, not inside `add`.

    Like the rest of the re-fitting it frames the whole scene graph, so doing
    it per object made filling a viewer quadratic.
    """
    v = oc.Viewer(offscreen=True, size=(200, 200))
    off_center = mesh.copy()
    off_center.apply_translation((500, 0, 0))

    v.add_mesh(off_center, name="mesh")
    assert v._center_pending
    assert tuple(v.camera.local.position) == (0, 0, 0)  # not yet

    v.canvas.draw()
    assert not v._center_pending
    assert v.camera.local.position[0] == pytest.approx(500, rel=1e-3)

    # `center=False` must leave the camera alone entirely
    before = tuple(v.camera.local.position)
    v.add_mesh(mesh.copy(), name="second", center=False)
    v.canvas.draw()
    assert tuple(v.camera.local.position) == before

    v.close()


def test_center_camera_yields_to_the_user(mesh):
    """A view set up after `add` must survive the deferred centering.

    Centering used to happen inside `add`, so anything the user did to the
    camera afterwards won. Deferring it must not turn that around.
    """
    v = oc.Viewer(offscreen=True, size=(200, 200))
    v.add_mesh(mesh)

    # N.B. `local.rotation` is a live view onto the camera's own buffer, so
    # these have to be copies rather than `np.asarray`
    def rotation():
        return np.array(v.camera.local.rotation, dtype=float)

    # This is what the deferred centering would produce ...
    v.center_camera()
    centered = rotation()

    v.add_mesh(mesh.copy(), name="second")
    v.camera.show_object(v.scene, view_dir=(0, -0.6, -1), up=(0, 1, 0))
    mine = rotation()
    assert not np.allclose(mine, centered)

    v.canvas.draw()
    assert rotation() == pytest.approx(mine)

    # ... but a *later* `add` centers again, just as it did before
    v.add_mesh(mesh.copy(), name="third")
    v.canvas.draw()
    assert rotation() == pytest.approx(centered)

    v.close()


def test_scene_refit_with_bounding_box(mesh):
    """The bounding box visual must not keep the scene stale forever.

    Re-drawing it removes the old one, which comes back through
    `update_helper` and would otherwise flag the scene as stale on every frame.
    """
    v = oc.Viewer(offscreen=True, size=(200, 200))
    v.add_mesh(mesh)
    v.show_bounds = True
    v.canvas.draw()

    assert not v._scene_stale
    assert sum(getattr(x, "_object_type", None) == "boundingbox" for x in v.visuals) == 1

    # A new object still has to grow the box
    big = mesh.copy()
    big.apply_scale(10)
    v.add_mesh(big, name="big")
    v.canvas.draw()
    (box,) = [x for x in v.visuals if getattr(x, "_object_type", None) == "boundingbox"]
    assert tuple(box.local.scale) == pytest.approx(np.ptp(v.bounds, axis=1), rel=1e-3)
    assert not v._scene_stale

    v.close()


def test_shadow_applies_to_objects_added_later(mesh):
    """Visuals added after the toggle must pick up the shadow state."""
    v = oc.Viewer(offscreen=True, size=(200, 200))
    v.shadows = True

    v.add_mesh(mesh)
    v.canvas.draw()  # the flags are applied per frame; see `_refresh_scene`
    (vis,) = v.visuals
    assert vis.cast_shadow is True
    assert vis.receive_shadow is True

    v.close()


def test_headlight_offset():
    v = oc.Viewer(offscreen=True, size=(200, 200))

    assert tuple(v._headlight_offset) == (-0.5, 0.5, 0)

    # A float/tuple switches the headlight on and sets the offset
    v.headlight = 1
    assert v.headlight is True
    assert tuple(v._headlight_offset) == (-1, 1, 0)

    v.headlight = (0.2, 0.3)
    assert v.headlight is True
    assert tuple(v._headlight_offset) == (0.2, 0.3, 0)

    v.headlight = (0.2, 0.3, 0.4)
    assert tuple(v._headlight_offset) == (0.2, 0.3, 0.4)

    # Switching off and on again must keep the offset
    v.headlight = False
    assert v.headlight is False
    v.headlight = True
    assert tuple(v._headlight_offset) == (0.2, 0.3, 0.4)

    with pytest.raises(ValueError):
        v.headlight = (0.2, 0.3, 0.4, 0.5)

    v.close()


def test_headlight_direction():
    """The offset must translate into the light direction it always did."""
    v = oc.Viewer(offscreen=True, size=(50, 50))
    v.add_mesh(tm.creation.icosphere())

    for offset in ((-0.5, 0.5, 0), (1, 0, 0), (0.2, 0.3, 0.4)):
        v.headlight = offset
        for view in ("XY", "XZ", "-YZ"):
            v.set_view(view)
            v.screenshot(filename=None, size=(50, 50))

            # This is what pygfx derives for a camera-parented light: from the
            # light's offset towards the point one unit in front of the camera
            position = la.vec_transform(offset, v.camera.world.matrix)
            target = la.vec_transform((0, 0, -1), v.camera.world.matrix)
            expected = target - position
            expected /= np.linalg.norm(expected)

            assert np.asarray(v._headlight.world.forward) == pytest.approx(
                expected, abs=1e-6
            )

    v.close()


def test_headlight_shadow_is_camera_invariant():
    """Moving the camera must not change the shadows the headlight casts.

    With an ortho camera, dollying along the view axis changes neither the
    projection nor the light's direction, so the render has to come out
    identical. It did not before the headlight was taken off the camera: its
    shadow frustum was sized from the camera's *position*, so every dolly step
    re-quantised the shadow map and the shadows flickered.

    """
    def render(dolly):
        # `plain` also switches off the occlusion, which samples the depth
        # buffer - and its precision does depend on where the camera is, which
        # has nothing to do with the shadows
        v = _shadow_viewer(headlight=True, plain=True)
        v.camera.local.position = (
            np.asarray(v.camera.local.position) + v.camera.world.forward * dolly
        )
        img = np.asarray(v.screenshot(filename=None, size=(200, 200)))[..., :3]
        cam = v._headlight.shadow.camera
        v.close()
        return img.astype(float), (cam.width, cam.height, *cam.depth_range)

    ref, ref_frustum = render(0)
    # There have to be shadows in the first place for this to mean anything
    assert (ref.mean(axis=-1) < 100).sum() > 100

    for dolly in (-5, -50, -500):
        img, frustum = render(dolly)
        assert frustum == pytest.approx(ref_frustum)
        assert img == pytest.approx(ref)


def test_headlight_render(mesh):
    """The headlight must light the object the same way from any angle."""

    def render(headlight):
        v = oc.Viewer(offscreen=True, size=(200, 200), headlight=headlight)
        v.add_mesh(mesh, color="white")
        brightness = []
        for view in ("XY", "-XY"):
            v.set_view(view)
            img = np.asarray(v.screenshot(filename=None, size=(200, 200)))[..., :3]
            img = img.astype(float)
            # Mean over the object (the background is black)
            brightness.append(img[img.max(axis=-1) > 5].mean())
        v.close()
        return brightness

    # With fixed lights, the sphere is noticeably darker from the back
    front, back = render(False)
    assert back < 0.9 * front

    # With the headlight, front and back view are lit identically
    front, back = render(True)
    assert abs(back - front) < 0.01 * front


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


@pytest.fixture()
def solid_ball():
    """A solid ball of voxels (radius 20) centered at (25, 25, 25)."""
    g = np.stack(np.meshgrid(*[np.arange(-22, 23)] * 3, indexing="ij"), -1)
    g = g.reshape(-1, 3)
    return g[np.linalg.norm(g, axis=1) <= 20] + 25


@pytest.mark.parametrize("mode", ["mip", "density", "surface"])
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


def _radial_luminance(img):
    """Per-pixel luminance of the rendered object plus its normalized radius."""
    mask = img[..., :3].max(axis=-1) > 5
    ys, xs = np.nonzero(mask)
    r = np.hypot(ys - ys.mean(), xs - xs.mean())
    return img[..., :3].mean(axis=-1)[ys, xs], r / r.max()


def test_sparse_volume_surface_shading(solid_ball):
    """The isosurface must be lit by its gradient, not rendered flat.

    A ball lit by the headlight follows Lambert's cosine: the surface normal
    faces the camera at the center of the silhouette and turns away toward
    the rim. Regression test for the normal (gradient) computation - a
    constant-luminance blob means the normals are degenerate.
    """
    v = oc.Viewer(offscreen=True, size=(200, 200))
    v.add_sparse_volume(solid_ball, mode="surface", color=["k", "w"])
    v.camera.show_object(v.scene, scale=1, view_dir=(0, 0, -1), up=(0, 1, 0))
    img = np.asarray(v.screenshot(filename=None, size=(200, 200)))
    v.close()

    lum, r = _radial_luminance(img)
    cos_theta = np.sqrt(np.clip(1 - r**2, 0, 1))
    corr = np.corrcoef(lum, cos_theta)[0, 1]
    assert corr > 0.5, f"Surface shading does not follow the normals (corr={corr:.2f})"
    assert np.median(lum[r < 0.2]) > np.median(lum[r > 0.8]) + 40


def test_sparse_volume_surface_anisotropic(solid_ball):
    """Normals must be transformed into world space, not used in voxel space.

    With anisotropic `spacing` the two are not related by a rotation, so
    skipping the inverse-transpose transform visibly changes the shading.
    """
    lums = []
    for spacing in ((1, 1, 1), (1, 1, 4)):
        v = oc.Viewer(offscreen=True, size=(200, 200))
        v.add_sparse_volume(
            solid_ball, mode="surface", color=["k", "w"], spacing=spacing
        )
        v.camera.show_object(v.scene, scale=1, view_dir=(1, 0, 0), up=(0, 0, 1))
        img = np.asarray(v.screenshot(filename=None, size=(200, 200)))
        v.close()
        lum, r = _radial_luminance(img)
        lums.append(np.median(lum[r > 0.7]))

    # Stretching along the view-perpendicular axis turns the rim of the
    # silhouette more towards the camera, i.e. it must get brighter
    assert lums[1] > lums[0] + 10, f"Shading ignores `spacing` ({lums})"


def test_sparse_volume_surface_threshold(solid_ball):
    """A higher threshold must carve out a smaller isosurface."""
    center = np.array([25, 25, 25])
    # Values ramp from 0 at the surface of the ball to 1 at its center
    values = 1 - np.linalg.norm(solid_ball - center, axis=1) / 20

    radii = []
    for threshold in (0.2, 0.7):
        v = oc.Viewer(offscreen=True, size=(200, 200))
        v.add_sparse_volume(
            solid_ball,
            values=values,
            mode="surface",
            color=["k", "w"],
            threshold=threshold,
        )
        v.camera.show_object(v.scene, scale=1, view_dir=(0, 0, -1), up=(0, 1, 0))
        img = np.asarray(v.screenshot(filename=None, size=(200, 200)))
        v.close()
        ys, _ = np.nonzero(img[..., :3].max(axis=-1) > 5)
        radii.append((ys.max() - ys.min() + 1) / 2)

    # The iso-radius shrinks from 0.8 to 0.3 of the ball's radius
    assert radii[0] > radii[1] * 2, f"threshold has too little effect ({radii})"


def test_sparse_volume_density(solid_ball):
    """Density mode must accumulate rather than saturate on the first voxel."""
    stats = {}
    for kwargs in (dict(mode="mip"), dict(mode="density", density=0.02),
                   dict(mode="density", density=1.0)):
        v = oc.Viewer(offscreen=True, size=(200, 200))
        v.add_sparse_volume(solid_ball, color=["k", "w"], **kwargs)
        v.camera.show_object(v.scene, scale=1, view_dir=(0, 0, -1), up=(0, 1, 0))
        img = np.asarray(v.screenshot(filename=None, size=(200, 200)))
        v.close()
        lum, r = _radial_luminance(img)
        # Ignore the antialiased silhouette, which varies in every mode
        lum = lum[r < 0.7]
        stats[kwargs.get("density", "mip")] = (lum.mean(), lum.std())

    # MIP of a binary volume is flat by construction; density must vary with
    # the path length through the ball
    assert stats[0.02][1] > stats["mip"][1] * 5, f"density mode is flat: {stats}"
    # ... and a higher extinction must render more opaque
    assert stats[1.0][0] > stats[0.02][0] + 20, f"`density` has no effect: {stats}"


def test_sparse_volume_material_properties():
    from octarine.shaders import SparseVolumeMaterial

    m = SparseVolumeMaterial()
    assert m.render_mode == "mip"
    # "surface" is an alias for pygfx's "iso"
    m.render_mode = "surface"
    assert m.render_mode == "iso"

    m.threshold = 0.25
    m.density = 2.0
    m.gradient_delta = 1.5
    m.smoothing = 2.0
    m.shininess = 10
    m.emissive = "#f00"
    assert m.threshold == 0.25
    assert m.density == 2.0
    assert m.gradient_delta == 1.5
    assert m.smoothing == 2.0
    assert m.shininess == 10
    assert tuple(m.emissive)[:3] == (1, 0, 0)

    for prop, value in (
        ("render_mode", "bogus"),
        ("step_size", 0),
        ("density", -1),
        ("gradient_delta", 0),
        ("smoothing", -1),
    ):
        with pytest.raises(ValueError):
            setattr(m, prop, value)


def test_runs_from_voxels_roundtrip(solid_ball):
    """Coordinates -> runs must be lossless and actually compress."""
    from octarine.shaders import runs_from_voxels

    runs = runs_from_voxels(solid_ball)
    assert len(runs) < len(solid_ball) / 10, "runs did not compress"

    expanded = np.concatenate(
        [
            np.column_stack([np.arange(x, x + n), np.full(n, y), np.full(n, z)])
            for x, y, z, n in runs
        ]
    )
    assert np.array_equal(
        np.unique(solid_ball, axis=0), np.unique(expanded, axis=0)
    )

    # Duplicates and negative coordinates must survive too
    dupes = np.repeat(solid_ball - 100, 3, axis=0)
    runs2 = runs_from_voxels(dupes)
    assert runs2[:, 3].sum() == len(np.unique(dupes, axis=0))


def test_pack_voxel_runs():
    from octarine.shaders import pack_voxel_runs

    # Two runs, the second crossing a brick border (brick_size=16)
    runs = np.array([[0, 0, 0, 4], [14, 5, 0, 6]])
    p = pack_voxel_runs(runs, brick_size=16)

    assert p.n_voxels == 10
    assert p.n_bricks == 2  # the second run straddles bricks 0 and 1
    assert p.shape == (20, 6, 1)
    # 1 bit per voxel, no apron
    assert p.bits.nbytes == p.n_bricks * 16**3 // 8
    assert int(np.unpackbits(p.bits.view(np.uint8)).sum()) == 10

    with pytest.raises(ValueError):
        pack_voxel_runs(runs, brick_size=7)
    with pytest.raises(ValueError):
        pack_voxel_runs(np.zeros((0, 4)))


@pytest.mark.parametrize("mode", ["mip", "density", "surface"])
def test_adding_voxel_runs(solid_ball, mode):
    from octarine.shaders import runs_from_voxels

    runs = runs_from_voxels(solid_ball)
    v = oc.Viewer(offscreen=True, size=(200, 200))
    v.add_sparse_volume(oc.VoxelRuns(runs), mode=mode)
    (obj,) = [o for objs in v.objects.values() for o in objs]
    assert obj._object_type == "sparsevolume"
    v.canvas.draw()  # force the shader to actually compile/render
    v.close()


def test_voxel_runs_match_coordinates(solid_ball):
    """The bit- and byte-per-voxel paths must render the same volume."""
    from octarine.shaders import runs_from_voxels

    def silhouette(obj):
        v = oc.Viewer(offscreen=True, size=(200, 200))
        v.add_sparse_volume(obj, mode="surface", color=["k", "w"])
        v.camera.show_object(v.scene, scale=1, view_dir=(0.3, 1, -0.4), up=(0, 0, 1))
        img = np.asarray(v.screenshot(filename=None, size=(200, 200)))
        v.close()
        return img[..., :3].max(axis=-1) > 5

    a = silhouette(solid_ball)
    b = silhouette(oc.VoxelRuns(runs_from_voxels(solid_ball)))
    iou = (a & b).sum() / (a | b).sum()
    assert iou > 0.95, f"bitmask and atlas paths disagree (IoU={iou:.3f})"


def test_voxel_runs_are_smaller(solid_ball):
    """The whole point of the bitmask path is the GPU footprint."""
    from octarine.shaders import runs_from_voxels, pack_voxel_runs, pack_sparse_voxels

    bits = pack_voxel_runs(runs_from_voxels(solid_ball), brick_size=16)
    atlas = pack_sparse_voxels(solid_ball, brick_size=16)
    assert bits.nbytes * 8 < atlas.atlas.nbytes, (
        f"bitmask ({bits.nbytes}) not much smaller than atlas ({atlas.atlas.nbytes})"
    )


def test_voxel_runs_reject_values(solid_ball):
    from octarine.shaders import runs_from_voxels

    runs = runs_from_voxels(solid_ball)
    v = oc.Viewer(offscreen=True)
    with pytest.raises(ValueError, match="binary occupancy"):
        v.add_sparse_volume(runs, values=np.ones(len(runs)))
    v.close()

    with pytest.raises(ValueError):
        oc.VoxelRuns(solid_ball)  # (N, 3) is not runs


@pytest.mark.parametrize("method", ["shader", "bitmask"])
def test_sparse_volume_surface_smoothing(solid_ball, method):
    """`smoothing` must change the shading but never the geometry.

    It widens the filter the *normal* is taken from, leaving the isosurface
    itself on the unsmoothed field - so the set of pixels the surface covers
    has to come out bit-identical. `emissive` makes every hit pixel bright
    regardless of its normal, which is what lets us compare hit sets rather
    than brightness.
    """
    from octarine.shaders import runs_from_voxels

    obj = (
        oc.VoxelRuns(runs_from_voxels(solid_ball))
        if method == "bitmask"
        else solid_ball
    )

    def render(smoothing, emissive):
        v = oc.Viewer(offscreen=True, size=(200, 200))
        v.add_sparse_volume(
            obj, mode="surface", color=["k", "w"], method=method, smoothing=smoothing
        )
        if emissive:
            v.objects["SparseVolume"][0].material.emissive = "#ffffff"
        v.camera.show_object(v.scene, scale=1, view_dir=(0.3, 1, -0.4), up=(0, 0, 1))
        img = np.asarray(v.screenshot(filename=None, size=(200, 200)))
        v.close()
        return img[..., :3].astype(float)

    off, on = render(0.0, False), render(2.0, False)
    assert np.abs(off - on).mean() > 0.05, "smoothing did not change the shading"

    hit_off = render(0.0, True).sum(axis=-1) > 12
    hit_on = render(2.0, True).sum(axis=-1) > 12
    assert (hit_off == hit_on).all(), "smoothing moved the surface"


def test_sparse_volume_smoothing_toggles_shader():
    """Flipping `smoothing` at runtime must re-resolve the shader variant."""
    g = np.arange(24) - 12
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    vox = np.stack(np.nonzero((X**2 + Y**2 + Z**2) <= 64), axis=1).astype(np.int32)

    v = oc.Viewer(offscreen=True, size=(150, 150))
    v.add_sparse_volume(vox, mode="surface", color=["k", "w"], method="bitmask")
    v.camera.show_object(v.scene, scale=1, view_dir=(0.3, 1, -0.4), up=(0, 0, 1))

    def shot():
        for _ in range(2):
            v.canvas.draw()
        return np.asarray(v.canvas.draw())[..., :3].astype(float)

    material = v.objects["SparseVolume"][0].material
    off = shot()
    material.smoothing = 2.0
    on = shot()
    material.smoothing = 0.0
    back = shot()
    v.close()

    assert np.abs(off - on).mean() > 0.05, "turning smoothing on had no effect"
    assert (back == off).all(), "turning smoothing back off did not restore"


def test_bitmask_material_properties():
    from octarine.shaders import BitmaskVolumeMaterial

    m = BitmaskVolumeMaterial()
    assert m.render_mode == "mip"
    m.render_mode = "surface"
    assert m.render_mode == "iso"

    m.threshold = 0.25
    m.density = 2.0
    m.gradient_delta = 1.5
    m.smoothing = 2.0
    m.color = "#ff0000"
    assert m.threshold == 0.25
    assert m.density == 2.0
    assert m.gradient_delta == 1.5
    assert m.smoothing == 2.0
    assert tuple(m.color)[:3] == (1, 0, 0)

    for prop, value in (
        ("render_mode", "bogus"),
        ("step_size", 0),
        ("density", -1),
        ("smoothing", -1),
    ):
        with pytest.raises(ValueError):
            setattr(m, prop, value)


def test_colormap_accepts_hex_color():
    """A hex string is a color, not the name of a colormap."""
    from octarine.visuals import to_colormap

    assert tuple(to_colormap("#ff9955", hide_zero=True).texture.data[-1][:3]) == pytest.approx(
        (1.0, 0.6, 1 / 3), abs=0.01
    )


def test_colormap_is_clamped():
    """Colormaps must not wrap around.

    pygfx's TextureMap defaults to `wrap="repeat"`, which makes a value at
    the top of `clim` (texcoord 1.0) blend the last texel with the first -
    which `hide_zero` made fully transparent - halving both color and alpha.
    """
    from octarine.visuals import to_colormap

    for hide_zero in (True, False):
        tm = to_colormap("red", hide_zero=hide_zero)
        assert tm.wrap_s == "clamp"
