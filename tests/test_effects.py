"""Tests for the environment (IBL), tone mapping, outline and matcap effects."""

import os
import pytest
import numpy as np
import pygfx as gfx
import trimesh as tm
import octarine as oc

from octarine.shaders import (
    ENVIRONMENT_PRESETS,
    MATCAP_PRESETS,
    TONEMAP_MODES,
    MatcapMeshMaterial,
    OutlinePass,
    ToneMappingPass,
    cube_directions,
    environment_radiance,
    make_matcap,
    matcap_texture,
    procedural_env_map,
)

# The GUI tests below must not try to open a window
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture()
def viewer():
    v = oc.Viewer(offscreen=True, size=(300, 300))
    yield v
    v.close()


@pytest.fixture()
def scene_viewer(viewer):
    """A viewer with two overlapping meshes of similar color."""
    sphere = tm.creation.icosphere(subdivisions=3, radius=0.8)
    viewer.add_mesh(sphere, color=(0.6, 0.65, 0.8), name="sphere")
    box = tm.creation.box((1.0, 1.0, 1.0))
    box.apply_translation([0.9, -0.5, 0.3])
    viewer.add_mesh(box, color=(0.6, 0.62, 0.75), name="box")
    return viewer


def _render(viewer):
    """Render the scene and return the image as (H, W, 3) floats."""
    return np.asarray(viewer.screenshot(filename=None, alpha=False))[..., :3].astype(
        float
    )


# --- Environment ------------------------------------------------------------


def test_cube_directions():
    dirs = cube_directions(8)
    assert dirs.shape == (6, 8, 8, 3)
    assert np.allclose(np.linalg.norm(dirs, axis=-1), 1)

    # Every face must point predominantly along its own axis. Note the x
    # flip: pygfx negates x when sampling, so world +x lives on the -X face.
    axes = [(0, -1), (0, 1), (1, 1), (1, -1), (2, 1), (2, -1)]
    for face, (axis, sign) in enumerate(axes):
        assert np.all(np.sign(dirs[face, ..., axis]) == sign), face
        others = [a for a in range(3) if a != axis]
        assert np.all(
            np.abs(dirs[face][..., axis, None]) >= np.abs(dirs[face][..., others])
        ), face


def test_environment_radiance_directions():
    """Softboxes must show up in the direction they were placed."""
    env = dict(
        sky="#000",
        horizon="#000",
        ground="#000",
        lights=(
            dict(direction=(0, 1, 0), color="#f00", intensity=1.0, angle=20),
            dict(direction=(1, 0, 0), color="#0f0", intensity=1.0, angle=20),
            dict(direction=(0, 0, 1), color="#00f", intensity=1.0, angle=20),
        ),
    )
    probes = np.array(
        [(0, 1, 0), (1, 0, 0), (0, 0, 1), (0, -1, 0)], dtype=np.float32
    )
    radiance = environment_radiance(probes, env)
    assert np.argmax(radiance[0]) == 0  # up   -> red
    assert np.argmax(radiance[1]) == 1  # right-> green
    assert np.argmax(radiance[2]) == 2  # front-> blue
    assert np.allclose(radiance[3], 0)  # down -> nothing


def test_environment_gradient():
    """The sky gradient must run from `ground` through `horizon` to `sky`."""
    env = dict(sky="#fff", horizon="#888", ground="#000", lights=())
    radiance = environment_radiance(
        np.array([(0, 1, 0), (0, 0, 1), (0, -1, 0)], dtype=np.float32), env
    )
    assert radiance[0].mean() > radiance[1].mean() > radiance[2].mean()


@pytest.mark.parametrize("preset", list(ENVIRONMENT_PRESETS))
def test_environment_presets_are_calibrated(preset):
    """No preset may blow out a white diffuse surface on its own.

    pygfx reads the diffuse irradiance off the smallest mip level, i.e. the
    average of each cube face, so that is what has to stay below 1.
    """
    radiance = environment_radiance(cube_directions(32), preset)
    face_means = radiance.mean(axis=(1, 2)) @ [0.2126, 0.7152, 0.0722]
    assert face_means.max() < 1.0
    # ... but it does have to be high dynamic range somewhere, else there is
    # nothing for the highlights (or bloom, or tone mapping) to work with
    assert radiance.max() > 1.0


def test_procedural_env_map():
    env_map = procedural_env_map("studio", resolution=32)
    assert isinstance(env_map, gfx.TextureMap)
    assert env_map.texture.size == (32, 32, 6)
    assert env_map.texture.generate_mipmaps
    assert env_map.preset == "studio"

    # Overriding a property makes it a custom environment
    assert procedural_env_map("studio", resolution=32, intensity=2).preset is None
    assert procedural_env_map("studio", resolution=32, rotation=90).preset is None


def test_procedural_env_map_errors():
    with pytest.raises(ValueError, match="Unknown environment preset"):
        procedural_env_map("nope")
    with pytest.raises(ValueError, match="Unknown environment propert"):
        procedural_env_map("studio", bogus=1)
    with pytest.raises(ValueError, match="resolution"):
        procedural_env_map("studio", resolution=2)


def test_environment_rotation():
    """Rotating the environment must move the highlight, not remove it."""
    plain = procedural_env_map("studio", resolution=32)
    turned = procedural_env_map("studio", resolution=32, rotation=90)
    a = np.asarray(plain.texture.data, dtype=np.float32)
    b = np.asarray(turned.texture.data, dtype=np.float32)
    assert not np.allclose(a, b)
    assert a.max() == pytest.approx(b.max(), rel=0.2)


def test_set_environment(scene_viewer):
    scene_viewer.set_environment("studio")
    assert scene_viewer.environment is not None
    assert scene_viewer.scene.environment is scene_viewer.environment

    # Plain Phong meshes are converted so they can be lit in full ...
    material = scene_viewer.objects["sphere"][0].material
    assert isinstance(material, gfx.MeshStandardMaterial)
    assert material.roughness == pytest.approx(0.4)
    # ... and keep their color
    assert material.color.rgba == pytest.approx((0.6, 0.65, 0.8, 1.0), abs=1e-6)

    # Meshes added afterwards get the same treatment
    scene_viewer.add_mesh(tm.creation.icosphere(subdivisions=2), color="w", name="late")
    assert isinstance(scene_viewer.objects["late"][0].material, gfx.MeshStandardMaterial)

    # Switching it off restores everything
    scene_viewer.set_environment(None)
    assert scene_viewer.environment is None and scene_viewer.scene.environment is None
    assert isinstance(scene_viewer.objects["sphere"][0].material, gfx.MeshPhongMaterial)
    assert not isinstance(scene_viewer.objects["sphere"][0].material, gfx.MeshStandardMaterial)


def test_set_environment_no_pbr(scene_viewer):
    """Without the conversion, meshes only get a reflection."""
    scene_viewer.set_environment("studio", pbr=False, reflectivity=0.5)
    material = scene_viewer.objects["sphere"][0].material
    assert isinstance(material, gfx.MeshPhongMaterial)
    assert material.env_map is scene_viewer.environment
    assert material.reflectivity == pytest.approx(0.5)

    scene_viewer.set_environment(None)
    assert scene_viewer.objects["sphere"][0].material.env_map is None


def test_set_environment_updates_settings(scene_viewer):
    """Calling it again must re-tune the materials, not just swap the map."""
    scene_viewer.set_environment("studio", roughness=0.2, metalness=0.1)
    material = scene_viewer.objects["sphere"][0].material
    assert material.roughness == pytest.approx(0.2)

    scene_viewer.set_environment("neon", roughness=0.8, metalness=0.9)
    material = scene_viewer.objects["sphere"][0].material
    assert material.roughness == pytest.approx(0.8)
    assert material.metalness == pytest.approx(0.9)


def test_environment_and_matcap_interact(scene_viewer):
    """The two both swap materials, so they have to hand over cleanly."""
    # Environment first, then a matcap on top of it
    scene_viewer.set_environment("studio")
    scene_viewer.set_matcap("gold")
    assert isinstance(scene_viewer.objects["sphere"][0].material, MatcapMeshMaterial)

    # Removing the environment must leave the matcap alone ...
    scene_viewer.set_environment(None)
    assert isinstance(scene_viewer.objects["sphere"][0].material, MatcapMeshMaterial)
    # ... and removing the matcap must land back on the original material,
    # not on the one the environment installed in between
    scene_viewer.set_matcap(None)
    assert type(scene_viewer.objects["sphere"][0].material) is gfx.MeshPhongMaterial

    # The other way round: a mesh coming out of a matcap while an
    # environment is on has to be lit by it
    scene_viewer.set_matcap("clay")
    scene_viewer.set_environment("neon")
    assert isinstance(scene_viewer.objects["sphere"][0].material, MatcapMeshMaterial)
    scene_viewer.set_matcap(None)
    assert isinstance(
        scene_viewer.objects["sphere"][0].material, gfx.MeshStandardMaterial
    )


def test_effects_survive_cycling(scene_viewer):
    """Switching everything on and off repeatedly must not accumulate."""
    lights = [light.intensity for light in scene_viewer.lights]
    for _ in range(3):
        scene_viewer.set_environment("studio")
        scene_viewer.set_matcap("gold")
        scene_viewer.set_outline()
        scene_viewer.set_tonemapping()
        scene_viewer.set_environment(None)
        scene_viewer.set_matcap(None)
        scene_viewer.set_outline(False)
        scene_viewer.set_tonemapping(None)

    assert type(scene_viewer.objects["sphere"][0].material) is gfx.MeshPhongMaterial
    assert [light.intensity for light in scene_viewer.lights] == pytest.approx(lights)
    names = [type(p).__name__ for p in scene_viewer.renderer.effect_passes]
    assert names.count("OutlinePass") == 1
    assert "ToneMappingPass" not in names


def test_set_environment_keeps_effect_materials(scene_viewer):
    """Silhouette/subsurface materials must not be swapped out from under us."""
    from octarine.shaders import SilhouetteMeshMaterial, SubsurfaceMeshMaterial

    scene_viewer.set_silhouette(3, objects="sphere")
    scene_viewer.set_subsurface(1.0, objects="box")
    scene_viewer.set_environment("studio")
    assert isinstance(scene_viewer.objects["sphere"][0].material, SilhouetteMeshMaterial)
    assert isinstance(scene_viewer.objects["box"][0].material, SubsurfaceMeshMaterial)
    # ... but they do pick up a reflection
    assert scene_viewer.objects["sphere"][0].material.env_map is scene_viewer.environment


def test_set_environment_dims_lights(scene_viewer):
    before = [light.intensity for light in scene_viewer.lights]
    scene_viewer.set_environment("studio", dim_lights=0.5)
    assert [light.intensity for light in scene_viewer.lights] == pytest.approx(
        [i * 0.5 for i in before]
    )

    # Changing environment must not dim them a second time
    scene_viewer.set_environment("neon", dim_lights=0.5)
    assert [light.intensity for light in scene_viewer.lights] == pytest.approx(
        [i * 0.5 for i in before]
    )

    scene_viewer.set_environment(None)
    assert [light.intensity for light in scene_viewer.lights] == pytest.approx(before)


def test_set_environment_background(scene_viewer):
    scene_viewer.set_environment("sunset", show_background=True)
    assert isinstance(scene_viewer._background.material, gfx.BackgroundSkyboxMaterial)
    # The background must actually be drawn, not just set
    assert _render(scene_viewer)[5, 5].sum() > 10

    scene_viewer.set_environment(None)
    assert isinstance(scene_viewer._background.material, gfx.BackgroundMaterial)


def test_set_environment_changes_the_image(scene_viewer):
    before = _render(scene_viewer)
    scene_viewer.set_environment("neon", roughness=0.2, metalness=0.8)
    assert not np.allclose(before, _render(scene_viewer))


# --- Tone mapping -----------------------------------------------------------


@pytest.mark.parametrize("mode", TONEMAP_MODES)
def test_tonemapping_modes_render(scene_viewer, mode):
    scene_viewer.set_environment("studio", show_background=True)
    scene_viewer.set_tonemapping(mode, exposure=2.0)
    assert scene_viewer._tonemap_pass.mode == mode
    image = _render(scene_viewer)
    assert np.isfinite(image).all() and image.max() > 0


def test_tonemapping_prevents_clipping(scene_viewer):
    """The whole point: highlights must stop being flat white."""
    scene_viewer.set_environment("studio", roughness=0.15, metalness=0.4)
    scene_viewer.exposure = 4.0  # deliberately far too bright

    scene_viewer.set_tonemapping("none", exposure=4.0)
    clipped_off = (_render(scene_viewer).max(axis=-1) >= 255).mean()

    scene_viewer.set_tonemapping("aces", exposure=4.0)
    clipped_on = (_render(scene_viewer).max(axis=-1) >= 255).mean()

    assert clipped_off > 0.02
    assert clipped_on < clipped_off / 10


def test_exposure_property(viewer):
    assert viewer.exposure == 1.0

    # Setting it switches tone mapping on
    viewer.exposure = 2.0
    assert viewer._tonemap_pass is not None
    assert viewer.exposure == 2.0
    assert viewer._tonemap_pass.stops == pytest.approx(1.0)

    viewer.set_tonemapping(None)
    assert viewer._tonemap_pass is None
    assert viewer.exposure == 1.0


def test_tonemapping_errors(viewer):
    with pytest.raises(ValueError, match="Unknown tone mapping mode"):
        viewer.set_tonemapping("nope")
    with pytest.raises(ValueError, match="exposure"):
        ToneMappingPass(exposure=-1)
    with pytest.raises(ValueError, match="white_point"):
        ToneMappingPass(white_point=0)


def test_exposure_brightens(scene_viewer):
    scene_viewer.set_tonemapping("none", exposure=0.5)
    dark = _render(scene_viewer).mean()
    scene_viewer.set_tonemapping("none", exposure=1.5)
    assert _render(scene_viewer).mean() > dark


# --- Outline ----------------------------------------------------------------


def test_set_outline(scene_viewer):
    before = _render(scene_viewer)
    scene_viewer.set_outline(color="#fff", thickness=2)
    after = _render(scene_viewer)
    assert not np.allclose(before, after)
    assert scene_viewer._outline_pass.thickness == 2

    # A white outline must add bright pixels the plain render does not have
    assert (after.min(axis=-1) > 240).sum() > (before.min(axis=-1) > 240).sum()

    scene_viewer.set_outline(False)
    assert not scene_viewer._outline_pass.enabled
    assert np.allclose(before, _render(scene_viewer))


def test_outline_debug_mode(scene_viewer):
    """In debug mode the image is the edge mask: white lines on black."""
    scene_viewer.set_outline(debug=True)
    image = _render(scene_viewer)
    assert image.max() > 200  # there are edges
    assert (image.max(axis=-1) < 10).mean() > 0.5  # and mostly background
    # The mask is greyscale
    assert np.allclose(image[..., 0], image[..., 1])


def test_outline_thickness(scene_viewer):
    """A thicker outline must cover more pixels."""
    counts = []
    for thickness in (1, 3):
        scene_viewer.set_outline(thickness=thickness, debug=True)
        counts.append((_render(scene_viewer).max(axis=-1) > 128).sum())
    assert counts[1] > counts[0]


def test_outline_creases(scene_viewer):
    """Switching creases off must leave fewer edges than leaving them on."""
    scene_viewer.set_outline(debug=True, normal_threshold=0.3)
    with_creases = (_render(scene_viewer).max(axis=-1) > 128).sum()
    scene_viewer.set_outline(debug=True, normal_threshold=0)
    assert (_render(scene_viewer).max(axis=-1) > 128).sum() <= with_creases


@pytest.mark.parametrize("camera", ["ortho", "perspective"])
def test_outline_cameras(camera):
    """Both camera types must produce outlines (they scale differently)."""
    v = oc.Viewer(offscreen=True, size=(200, 200), camera=camera)
    v.add_mesh(tm.creation.icosphere(subdivisions=3), color="w")
    v.set_outline(debug=True)
    assert (_render(v).max(axis=-1) > 128).sum() > 50
    v.close()


def test_outline_errors(viewer):
    with pytest.raises(ValueError, match="thickness"):
        OutlinePass(viewer.camera, thickness=0)
    with pytest.raises(ValueError, match="depth_threshold"):
        OutlinePass(viewer.camera, depth_threshold=0)
    with pytest.raises(ValueError, match="normal_threshold"):
        OutlinePass(viewer.camera, normal_threshold=3)


# --- Matcap -----------------------------------------------------------------


@pytest.mark.parametrize("preset", list(MATCAP_PRESETS))
def test_make_matcap(preset):
    image = make_matcap(preset, size=64)
    assert image.shape == (64, 64, 4)
    assert np.isfinite(image).all() and (image >= 0).all()
    # The sphere is lit from above, so its top must be brighter than its
    # bottom (the disc spans the whole image; row 0 is the top)
    assert image[8, 32, :3].mean() > image[56, 32, :3].mean()


def test_make_matcap_overrides():
    plain = make_matcap("pearl", size=32)
    tinted = make_matcap("pearl", size=32, base_color="#f00")
    assert tinted[16, 16, 0] > tinted[16, 16, 1]
    assert not np.allclose(plain, tinted)

    # Relighting it with another environment must change it too
    assert not np.allclose(plain, make_matcap("pearl", size=32, environment="neon"))


def test_matcap_texture_from_image():
    """An off-the-shelf matcap PNG (uint8 sRGB) must be usable directly."""
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    tex_map = matcap_texture(image)
    assert isinstance(tex_map, gfx.TextureMap)
    assert tex_map.texture.size[:2] == (16, 16)

    with pytest.raises(ValueError, match="matcap image"):
        matcap_texture(np.zeros((16, 16)))


def test_matcap_errors():
    with pytest.raises(ValueError, match="Unknown matcap preset"):
        make_matcap("nope")
    with pytest.raises(ValueError, match="Unknown matcap propert"):
        make_matcap("pearl", bogus=1)
    with pytest.raises(ValueError, match="tint"):
        MatcapMeshMaterial(tint=2)


def test_set_matcap(scene_viewer):
    before = _render(scene_viewer)
    scene_viewer.set_matcap("clay")

    material = scene_viewer.objects["sphere"][0].material
    assert isinstance(material, MatcapMeshMaterial)
    assert material.matcap.preset == "clay"
    assert material.tint == pytest.approx(MATCAP_PRESETS["clay"]["tint"])
    assert not np.allclose(before, _render(scene_viewer))

    # Switching it off restores the original material
    scene_viewer.set_matcap(None)
    assert isinstance(scene_viewer.objects["sphere"][0].material, gfx.MeshPhongMaterial)
    assert np.allclose(before, _render(scene_viewer))


def test_set_matcap_subset(scene_viewer):
    scene_viewer.set_matcap("jade", objects="sphere")
    assert isinstance(scene_viewer.objects["sphere"][0].material, MatcapMeshMaterial)
    assert not isinstance(scene_viewer.objects["box"][0].material, MatcapMeshMaterial)


def test_set_matcap_tint(scene_viewer):
    """Tinting must let an object's own color through."""
    scene_viewer.set_matcap("pearl", tint=1.0)
    tinted = _render(scene_viewer)
    scene_viewer.set_matcap("pearl", tint=0.0)
    plain = _render(scene_viewer)
    # The meshes are bluish, so tinting has to cost them red
    lit = plain.max(axis=-1) > 20
    assert tinted[lit][:, 0].mean() < plain[lit][:, 0].mean()


def test_matcap_ignores_lights(scene_viewer):
    """A matcap is shaded by its image alone."""
    scene_viewer.set_matcap("pearl", tint=0)
    before = _render(scene_viewer)
    scene_viewer.headlight = not scene_viewer.headlight
    assert np.allclose(before, _render(scene_viewer))


def test_add_mesh_matcap(viewer):
    viewer.add_mesh(tm.creation.icosphere(), color="w", matcap="gold", name="m")
    material = viewer.objects["m"][0].material
    assert isinstance(material, MatcapMeshMaterial)
    assert material.tint == pytest.approx(MATCAP_PRESETS["gold"]["tint"])

    with pytest.raises(ValueError, match="cannot be combined"):
        viewer.add_mesh(tm.creation.icosphere(), matcap="gold", silhouette=2)
    with pytest.raises(ValueError, match="cannot be combined"):
        viewer.add_mesh(tm.creation.icosphere(), matcap="gold", shader="toon")


def test_set_matcap_skips_environment(scene_viewer):
    """A matcap mesh must keep its material when an environment is set."""
    scene_viewer.set_matcap("clay", objects="sphere")
    scene_viewer.set_environment("studio")
    assert isinstance(scene_viewer.objects["sphere"][0].material, MatcapMeshMaterial)
    assert isinstance(scene_viewer.objects["box"][0].material, gfx.MeshStandardMaterial)


# --- Effect ordering --------------------------------------------------------


def test_effect_pass_order(viewer):
    """Passes must run in their fixed order, whatever order they are added in."""
    viewer.set_tonemapping("aces")
    viewer.set_depth_of_field(focus=5)
    viewer.set_outline()
    viewer.set_ambient_occlusion()

    names = [type(p).__name__ for p in viewer.renderer.effect_passes]
    order = [
        "AmbientOcclusionPass",
        "OutlinePass",
        "DepthOfFieldPass",
        "ToneMappingPass",
    ]
    assert [n for n in names if n in order] == order
    # pygfx' anti-aliasing wants the tone mapped image, so it goes last
    assert names[-1].endswith("AAPass")


def test_add_effect_route(viewer):
    """The new passes are also reachable through `add_effect`."""
    viewer.add_effect("outline", thickness=3)
    viewer.add_effect("tonemap", mode="reinhard", exposure=1.5)
    assert viewer._outline_pass.thickness == 3
    assert viewer._tonemap_pass.mode == "reinhard"
    assert viewer._tonemap_pass.exposure == pytest.approx(1.5)

    # ... and updating an existing one goes to the same pass
    viewer.add_effect("outline", thickness=1)
    assert viewer._outline_pass.thickness == 1
    assert sum(isinstance(p, OutlinePass) for p in viewer.renderer.effect_passes) == 1

    viewer.add_effect("outline", disable=True)
    assert viewer._outline_pass is None
    viewer.add_effect("tonemap", disable=True)
    assert viewer._tonemap_pass is None


# --- GUI --------------------------------------------------------------------


@pytest.fixture()
def controls(scene_viewer):
    # Note that the import can also fail if PySide6 *is* installed but clashes
    # with another Qt binding in the same environment - skip either way
    QtWidgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from octarine.controls import Controls

    return Controls(scene_viewer)


def test_environment_gui(controls, scene_viewer):
    assert not controls.env_checkbox.isChecked()

    controls.env_checkbox.setChecked(True)
    assert scene_viewer.environment is not None

    presets = list(ENVIRONMENT_PRESETS)
    controls.env_dropdown.setCurrentIndex(presets.index("sky"))
    assert scene_viewer.environment.preset == "sky"

    controls.env_background_checkbox.setChecked(True)
    assert isinstance(scene_viewer._background.material, gfx.BackgroundSkyboxMaterial)

    controls.env_checkbox.setChecked(False)
    assert scene_viewer.environment is None


def test_outline_gui(controls, scene_viewer):
    controls.outline_checkbox.setChecked(True)
    assert scene_viewer._outline_pass.enabled

    controls.outline_thickness_slider.setValue(3)
    assert scene_viewer._outline_pass.thickness == 3

    controls.outline_color_dropdown.setCurrentIndex(1)  # white
    assert scene_viewer._outline_pass.color.hex.lower() == "#ffffff"

    controls.outline_checkbox.setChecked(False)
    assert not scene_viewer._outline_pass.enabled


def test_matcap_gui(controls, scene_viewer):
    controls.matcap_checkbox.setChecked(True)
    assert isinstance(scene_viewer.objects["sphere"][0].material, MatcapMeshMaterial)

    presets = list(MATCAP_PRESETS)
    controls.matcap_dropdown.setCurrentIndex(presets.index("metal"))
    assert scene_viewer.objects["sphere"][0].material.matcap.preset == "metal"

    controls.matcap_checkbox.setChecked(False)
    assert isinstance(scene_viewer.objects["sphere"][0].material, gfx.MeshPhongMaterial)


def test_tonemap_gui(controls, scene_viewer):
    controls.tonemap_checkbox.setChecked(True)
    assert scene_viewer._tonemap_pass.mode == "aces"

    modes = list(TONEMAP_MODES)
    controls.tonemap_dropdown.setCurrentIndex(modes.index("filmic"))
    assert scene_viewer._tonemap_pass.mode == "filmic"

    slider = controls.tonemap_exposure_slider
    slider.setValue(round(1.0 / slider._step))  # +1 stop
    assert scene_viewer._tonemap_pass.exposure == pytest.approx(2.0)

    controls.tonemap_checkbox.setChecked(False)
    assert scene_viewer._tonemap_pass is None


def test_gui_reflects_api_state(scene_viewer):
    """A panel built while the effects are on must reflect them."""
    QtWidgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from octarine.controls import Controls

    scene_viewer.set_environment("neon")
    scene_viewer.set_outline(thickness=2)
    scene_viewer.set_matcap("gold")
    scene_viewer.set_tonemapping("filmic", exposure=2.0)

    controls = Controls(scene_viewer)
    assert controls.env_checkbox.isChecked()
    assert controls.env_dropdown.currentText() == "Neon"
    assert controls.outline_checkbox.isChecked()
    assert controls.outline_thickness_slider.value() == 2
    assert controls.matcap_checkbox.isChecked()
    assert controls.matcap_dropdown.currentText() == "Gold"
    assert controls.tonemap_checkbox.isChecked()
    assert controls.tonemap_dropdown.currentText() == "Filmic"
    assert controls.tonemap_exposure_slider.value() == pytest.approx(
        1.0 / controls.tonemap_exposure_slider._step
    )
