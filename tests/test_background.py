import os
import pytest
import numpy as np
import pygfx as gfx
import octarine as oc

from octarine.shaders import BACKGROUND_PRESETS, GradientBackgroundMaterial

# The GUI tests below must not try to open a window
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture()
def viewer():
    v = oc.Viewer(offscreen=True, size=(600, 400))
    yield v
    v.close()


def _render(viewer):
    """Render the (empty) scene and return the image as (H, W, 3) floats."""
    return np.asarray(viewer.screenshot(filename=None, alpha=False))[..., :3].astype(
        float
    )


def _px(img, x, y):
    """Sample the image at relative coordinates; (0, 0) is the top left."""
    h, w = img.shape[:2]
    return img[min(int(y * h), h - 1), min(int(x * w), w - 1)]


def test_set_bgcolor_solid(viewer):
    viewer.set_bgcolor("white")
    assert isinstance(viewer._background.material, gfx.BackgroundMaterial)
    assert _px(_render(viewer), 0.5, 0.5).mean() > 250

    viewer.set_bgcolor((0, 0, 0))
    assert _px(_render(viewer), 0.5, 0.5).mean() < 5


@pytest.mark.parametrize("as_list", [True, False])
def test_set_bgcolor_linear_gradient(viewer, as_list):
    colors = ["black", "white"]
    if as_list:
        viewer.set_bgcolor(colors)
    else:
        viewer.set_bgcolor(*colors)

    img = _render(viewer)
    assert _px(img, 0.5, 0.02).mean() > _px(img, 0.5, 0.98).mean()

    # Four colors set the corners
    viewer.set_bgcolor("black", "black", "white", "white")
    img = _render(viewer)
    assert _px(img, 0.5, 0.02).mean() > _px(img, 0.5, 0.98).mean()

    with pytest.raises(ValueError):
        viewer.set_bgcolor("black", "white", "red")


@pytest.mark.parametrize("preset", list(BACKGROUND_PRESETS))
def test_bg_gradient_presets(viewer, preset):
    viewer.set_bg_gradient(preset)
    assert isinstance(viewer._background.material, GradientBackgroundMaterial)

    params = BACKGROUND_PRESETS[preset]
    img = _render(viewer)

    # The center of the gradient must show its inner color
    inner = np.array(gfx.Color(params["colors"][0]).rgba[:3]) * 255
    assert np.allclose(_px(img, *params["center"]), inner, atol=2)

    # ... and the far corner the outer color, darkened by the vignette
    outer = np.array(gfx.Color(params["colors"][-1]).rgba[:3]) * 255
    corner = _px(img, 0.99, 0.99)
    assert (corner <= outer + 2).all()
    if params["vignette"] > 0:
        assert corner.sum() < outer.sum()


def test_bg_gradient_unknown_preset(viewer):
    with pytest.raises(ValueError):
        viewer.set_bg_gradient("not-a-preset")


def test_bg_gradient_overrides(viewer):
    viewer.set_bg_gradient("graphite", radius=0.3, falloff=1.5, vignette=0)
    mat = viewer._background.material
    assert mat.radius == pytest.approx(0.3)
    assert mat.falloff == pytest.approx(1.5)
    assert mat.vignette == 0
    # Everything else comes from the preset
    assert mat.center == pytest.approx(BACKGROUND_PRESETS["graphite"]["center"])

    # Two colors: the mid stop is interpolated
    viewer.set_bg_gradient(colors=("#FFFFFF", "#000000"))
    mat = viewer._background.material
    assert mat.color_inner.hex == "#ffffff"
    assert mat.color_outer.hex == "#000000"
    assert mat.color_mid.rgba[:3] == pytest.approx((0.5, 0.5, 0.5))

    with pytest.raises(ValueError):
        viewer.set_bg_gradient(colors=("#FFFFFF",))


def test_bg_gradient_is_circular(viewer):
    """Iso-lines must be circles, not ellipses, whatever the aspect ratio."""
    viewer.set_bg_gradient(center=(0.5, 0.5), radius=0.3, falloff=2, vignette=0)
    img = _render(viewer)

    # The viewer is 600x400, i.e. distances are measured in units of the width
    # and a dy of 0.3 in image coordinates equals a dx of 0.2
    assert np.allclose(_px(img, 0.7, 0.5), _px(img, 0.5, 0.8), atol=1)
    assert not np.allclose(_px(img, 0.7, 0.5), _px(img, 0.5, 0.7), atol=1)


def test_bg_gradient_falloff(viewer):
    """A higher falloff must keep the core brighter."""

    def at_half_radius(falloff):
        viewer.set_bg_gradient(
            center=(0.5, 0.5), radius=0.4, falloff=falloff, vignette=0
        )
        return _px(_render(viewer), 0.7, 0.5).mean()

    assert at_half_radius(4) > at_half_radius(2) > at_half_radius(0.5)


def test_bg_gradient_vignette(viewer):
    """The vignette must darken the corners but leave the center alone."""

    def render(vignette):
        viewer.set_bg_gradient(
            "graphite", center=(0.5, 0.5), radius=0.4, vignette=vignette
        )
        img = _render(viewer)
        return _px(img, 0.5, 0.5).mean(), _px(img, 0.99, 0.99).mean()

    center_off, corner_off = render(0)
    center_on, corner_on = render(0.8)

    assert center_on == pytest.approx(center_off, abs=1)
    assert corner_on < corner_off


def test_bg_gradient_toggle(viewer):
    viewer.set_bgcolor("#123456")
    viewer.set_bg_gradient("halo")
    assert isinstance(viewer._background.material, GradientBackgroundMaterial)

    # Switching the gradient off must restore the plain background color
    viewer.set_bg_gradient(None)
    assert isinstance(viewer._background.material, gfx.BackgroundMaterial)
    assert viewer._background.material.color_top_left.hex == "#123456"

    # As must setting a color while a gradient is active
    viewer.set_bg_gradient("halo")
    viewer.set_bgcolor("white")
    assert isinstance(viewer._background.material, gfx.BackgroundMaterial)
    assert _px(_render(viewer), 0.5, 0.5).mean() > 250


def test_bg_gradient_transparent_screenshot(viewer):
    """The gradient must not end up in a transparent screenshot."""
    viewer.set_bg_gradient("graphite")
    img = np.asarray(viewer.screenshot(filename=None, alpha=True))
    assert (img[..., 3] == 0).all()


def test_background_options(viewer):
    """The options behind the GUI dropdowns."""
    labels, values, current = oc.utils.background_options(viewer)
    assert labels[0] == "Plain" and values[0] is None
    assert values[1:] == list(BACKGROUND_PRESETS)
    assert labels[1:] == [n.title() for n in BACKGROUND_PRESETS]
    assert current == 0  # plain background

    # A preset must be reflected...
    viewer.set_bg_gradient("olive")
    labels, values, current = oc.utils.background_options(viewer)
    assert values[current] == "olive"

    # ... and a custom gradient (here: a preset with an override) must get
    # its own entry that restores it
    viewer.set_bg_gradient("olive", radius=0.9)
    labels, values, current = oc.utils.background_options(viewer)
    assert labels[current] == "Custom"
    viewer.set_bg_gradient(values[current])
    assert viewer._background.material.radius == pytest.approx(0.9)


def test_background_dropdown_qt(viewer):
    """The dropdown in the (Qt) controls panel."""
    # Note that the import can also fail if PySide6 *is* installed but clashes
    # with another Qt binding in the same environment - skip either way
    QtWidgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from octarine.controls import Controls

    dropdown = Controls(viewer).background_dropdown
    labels = [dropdown.itemText(i) for i in range(dropdown.count())]
    assert labels == ["Plain"] + [n.title() for n in BACKGROUND_PRESETS]
    assert dropdown.currentIndex() == 0

    # Picking an entry must set the background
    dropdown.setCurrentIndex(1)
    mat = viewer._background.material
    assert isinstance(mat, GradientBackgroundMaterial)
    assert mat.preset == labels[1].lower()

    # ... and "Plain" must switch it off again
    dropdown.setCurrentIndex(0)
    assert isinstance(viewer._background.material, gfx.BackgroundMaterial)

    # A panel built while a preset is active must reflect that preset
    viewer.set_bg_gradient("burgundy")
    assert Controls(viewer).background_dropdown.currentText() == "Burgundy"


def test_background_dropdown_jupyter(viewer):
    """The dropdown in the (ipywidgets) Jupyter toolbar."""
    pytest.importorskip("ipywidgets")

    from octarine.jupyter import JupyterToolbar

    dropdown = JupyterToolbar(viewer).background_dropdown
    labels = [label for label, _ in dropdown.options]
    assert labels == ["Plain"] + [n.title() for n in BACKGROUND_PRESETS]
    assert dropdown.value == 0

    dropdown.value = 1
    mat = viewer._background.material
    assert isinstance(mat, GradientBackgroundMaterial)
    assert mat.preset == labels[1].lower()

    dropdown.value = 0
    assert isinstance(viewer._background.material, gfx.BackgroundMaterial)

    viewer.set_bg_gradient("burgundy")
    dropdown = JupyterToolbar(viewer).background_dropdown
    assert dropdown.options[dropdown.value][0] == "Burgundy"


def test_gradient_material_validation():
    mat = GradientBackgroundMaterial()

    mat.center = (0.1, 0.2)
    assert mat.center == pytest.approx((0.1, 0.2))

    for prop, value in (
        ("radius", 0),
        ("falloff", -1),
        ("vignette", 1.5),
        ("center", (0.1, 0.2, 0.3)),
    ):
        with pytest.raises(ValueError):
            setattr(mat, prop, value)
