"""Generate the labeled screenshots for the "Effects & Shading" docs page.

Renders a demo scene (5 Stanford bunnies receding into the distance) with an
offscreen viewer and captures it once per effect. Output goes to docs/_static/.

Run from anywhere:

    python docs/generate_screenshots.py

Requires `octarine` with an offscreen-capable backend plus `trimesh` and
`pillow`; downloads the bunny mesh from the trimesh repository.

Note: the "normal" effect is currently skipped because pygfx' NormalPass
renders black in offscreen mode (as of pygfx 0.16.0).
"""

from pathlib import Path

import numpy as np
import trimesh as tm
import trimesh.transformations as tf
import octarine as oc
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "_static"
W, H = 640, 400  # final image size
RW, RH = 800, 500  # requested canvas size (hidpi displays double this)

bunny = tm.load_remote("https://github.com/mikedh/trimesh/raw/main/models/bunny.ply")
if hasattr(bunny, "to_geometry"):
    bunny = bunny.to_geometry()
bunny.apply_translation(-bunny.centroid)
bunny.apply_transform(tf.rotation_matrix(-np.pi / 2, [1, 0, 0]))  # stand upright
ext = bunny.extents.max()


def get_font(size=22):
    for candidate in ("/System/Library/Fonts/Helvetica.ttc", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def crop_window(img_arr, pad_frac=0.18):
    """Fixed-aspect crop window around non-black content."""
    nonblack = img_arr[..., :3].max(axis=2) > 10
    rows, cols = np.any(nonblack, axis=1), np.any(nonblack, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    h, w = img_arr.shape[:2]
    ch, cw = r1 - r0, c1 - c0
    r0, r1 = max(0, r0 - int(ch * pad_frac)), min(h, r1 + int(ch * pad_frac))
    c0, c1 = max(0, c0 - int(cw * pad_frac)), min(w, c1 + int(cw * pad_frac))
    # Expand the window to the target aspect ratio
    ch, cw = r1 - r0, c1 - c0
    aspect = W / H
    if cw / ch < aspect:  # too narrow -> widen
        extra = int(ch * aspect - cw)
        c0, c1 = max(0, c0 - extra // 2), min(w, c1 + extra // 2)
    else:  # too wide -> heighten
        extra = int(cw / aspect - ch)
        r0, r1 = max(0, r0 - extra // 2), min(h, r1 + extra // 2)
    return r0, r1, c0, c1


def finalize(img_arr, window, text, name):
    """Crop, downscale, label and save a screenshot."""
    r0, r1, c0, c1 = window
    im = Image.fromarray(img_arr[r0:r1, c0:c1]).convert("RGB")
    im = im.resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(im, "RGBA")
    font = get_font()
    pad = 8
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = 12, H - th - 2 * pad - 12
    d.rounded_rectangle([x, y, x + tw + 2 * pad, y + th + 2 * pad], radius=6, fill=(0, 0, 0, 160))
    d.text((x + pad, y + pad - bbox[1]), text, font=font, fill=(255, 255, 255, 255))
    im.save(OUT / f"{name}.png", optimize=True)
    print(f"{name}.png saved")


def grab(v, n_predraw=2):
    for _ in range(n_predraw):
        v.canvas.draw()
    return np.asarray(v.screenshot(filename=None, alpha=False))


def make_scene():
    """Five bunnies stepping right and receding into the distance."""
    v = oc.Viewer(offscreen=True, size=(RW, RH))
    colors = ["#e15759", "#f28e2b", "#59a14f", "#4e79a7", "#b07aa1"]
    for i, c in enumerate(colors):
        m = bunny.copy()
        m.apply_translation((i * ext * 0.75, i * ext * 0.12, -i * ext * 1.5))
        v.add_mesh(m, name=f"bunny{i}", color=c)
    v.center_camera()
    return v


if __name__ == "__main__":
    # Baseline (also defines the crop window shared by all scene shots)
    v = make_scene()
    base = grab(v)
    window = crop_window(base)
    finalize(base, window, "no effect", "effects_baseline")
    v.close()

    # Post-processing effects (bloom cranked up - the default is barely visible)
    for eff, kwargs in [
        ("edl", {}),
        ("noise", {}),
        ("fog", {}),
        ("depth", {}),
        ("bloom", {"bloom_strength": 0.5}),
    ]:
        v = make_scene()
        v.add_effect(eff, **kwargs)
        finalize(grab(v), window, eff, f"effects_{eff}")
        v.close()

    # Depth of field: autofocus snaps to the object nearest the view center;
    # extra pre-draws give the autofocus a depth frame to sample from
    v = make_scene()
    v.set_depth_of_field(aperture=140, snap_radius=250)
    finalize(grab(v, n_predraw=4), window, "depth of field", "effects_dof")
    v.close()

    # Silhouette: single bunny, before/after
    v = oc.Viewer(offscreen=True, size=(RW, RH))
    v.add_mesh(bunny.copy(), name="bunny", color="#4e79a7")
    v.center_camera()
    before = grab(v)
    sil_window = crop_window(before)
    finalize(before, sil_window, "silhouette off", "effects_silhouette_before")
    v.set_silhouette(3)
    finalize(grab(v), sil_window, "silhouette = 3", "effects_silhouette_after")
    v.close()

    print("done")
