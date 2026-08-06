"""Custom pygfx WorldObject/Material/Shader for binary sparse volumetric data.

Renders run-length-encoded voxels packed into a bitmask brick map (see
`.rle_packing`) using two-level raycasting. Compared to `.sparse_volume` this
trades per-voxel values and hardware filtering for ~23x less GPU memory.

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
    GfxTextureView,
    load_wgsl,
)

from .rle_packing import PackedBitmask


class BitmaskVolume(gfx.WorldObject):
    """A binary sparse volume defined by bitmask-packed voxel runs.

    Parameters
    ----------
    packed :    PackedBitmask
                Bitmask-packed voxel data (see `pack_voxel_runs`).
    material :  BitmaskVolumeMaterial
                The material defining the appearance of the volume.

    """

    def __init__(self, packed, material, **kwargs):
        if not isinstance(packed, PackedBitmask):
            raise TypeError(f"Expected PackedBitmask, got {type(packed)}")

        shape = tuple(int(s) for s in packed.shape)
        # The corner positions let pygfx derive the bounding box from the
        # geometry (voxel centers sit at integer positions, hence the 0.5)
        corners = np.array(
            [[-0.5, -0.5, -0.5], [shape[0] - 0.5, shape[1] - 0.5, shape[2] - 0.5]],
            dtype=np.float32,
        )
        geometry = gfx.Geometry(
            positions=corners,
            super_index=gfx.Texture(packed.super_index, dim=3),
            bricktab=packed.bricktab,
            bits=packed.bits,
        )
        super().__init__(geometry, material, **kwargs)

        self.packed = packed
        self.shape = shape
        self.brick_size = int(packed.brick_size)
        self.n_bricks = int(packed.n_bricks)


class BitmaskVolumeMaterial(gfx.Material):
    """Material for rendering a BitmaskVolume.

    The data is binary occupancy, so there is nothing to map onto a colormap:
    the volume is drawn in a single `color`.

    Parameters
    ----------
    render_mode :   "mip" | "density" | "iso"
                    - "mip": flat silhouette (for binary data the first hit
                      *is* the maximum)
                    - "density": front-to-back absorption, so thicker parts
                      render more opaque
                    - "iso" (alias "surface"): shaded isosurface
    color :         Color
                    Color of the volume.
    step_size :     float
                    Ray-march step (in voxels) inside occupied bricks.
    threshold :     float
                    "iso" mode only: level at which the surface sits, between
                    0 (just outside a voxel) and 1 (a voxel center).
    density :       float
                    "density" mode only: extinction per voxel.
    gradient_delta : float
                    "iso" mode only: half-width (in voxels) of the central
                    differences used to derive the surface normal.
    smoothing :     float
                    "iso" mode only: width (in voxels) of an extra filter
                    applied to the field the *normal* is taken from. 0 (the
                    default) is off. Removes the voxel-scale stipple from the
                    shading without moving the surface.
    shininess :     float
                    "iso" mode only: size of the specular highlight.
    emissive :      Color
                    "iso" mode only: color emitted regardless of lighting.

    """

    uniform_type = dict(
        gfx.Material.uniform_type,
        color="4xf4",
        step_size="f4",
        threshold="f4",
        density="f4",
        gradient_delta="f4",
        smoothing="f4",
        shininess="f4",
        emissive_color="4xf4",
    )

    def __init__(
        self,
        render_mode="mip",
        color="#ffffff",
        step_size=0.5,
        threshold=0.5,
        density=0.1,
        gradient_delta=1.0,
        smoothing=0.0,
        shininess=30,
        emissive="#000",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._store.smoothing_on = float(smoothing) > 0
        self.render_mode = render_mode
        self.color = color
        self.step_size = step_size
        self.threshold = threshold
        self.density = density
        self.gradient_delta = gradient_delta
        self.smoothing = smoothing
        self.shininess = shininess
        self.emissive = emissive

    @property
    def render_mode(self):
        """The render mode: "mip", "density" or "iso"."""
        return self._store.render_mode

    @render_mode.setter
    def render_mode(self, value):
        if value == "surface":
            value = "iso"
        if value not in ("mip", "density", "iso"):
            raise ValueError(
                f"render_mode must be 'mip', 'density' or 'iso', got {value!r}"
            )
        self._store.render_mode = value

    @property
    def color(self):
        """Color of the volume."""
        return gfx.Color(self.uniform_buffer.data["color"])

    @color.setter
    def color(self, value):
        self.uniform_buffer.data["color"] = gfx.Color(value)
        self.uniform_buffer.update_full()

    @property
    def step_size(self):
        """Ray-march step (in voxels) inside occupied bricks."""
        return float(self.uniform_buffer.data["step_size"])

    @step_size.setter
    def step_size(self, value):
        value = float(value)
        if not 0 < value <= 16:
            raise ValueError(f"step_size must be in (0, 16], got {value}")
        self.uniform_buffer.data["step_size"] = value
        self.uniform_buffer.update_full()

    @property
    def threshold(self):
        """Isosurface level ("iso" mode)."""
        return float(self.uniform_buffer.data["threshold"])

    @threshold.setter
    def threshold(self, value):
        self.uniform_buffer.data["threshold"] = float(value)
        self.uniform_buffer.update_full()

    @property
    def density(self):
        """Extinction per voxel ("density" mode)."""
        return float(self.uniform_buffer.data["density"])

    @density.setter
    def density(self, value):
        value = float(value)
        if value < 0:
            raise ValueError(f"density must be >= 0, got {value}")
        self.uniform_buffer.data["density"] = value
        self.uniform_buffer.update_full()

    @property
    def gradient_delta(self):
        """Half-width (in voxels) of the normal's central differences."""
        return float(self.uniform_buffer.data["gradient_delta"])

    @gradient_delta.setter
    def gradient_delta(self, value):
        value = float(value)
        if not 0 < value <= 16:
            raise ValueError(f"gradient_delta must be in (0, 16], got {value}")
        self.uniform_buffer.data["gradient_delta"] = value
        self.uniform_buffer.update_full()

    @property
    def smoothing(self):
        """Width (in voxels) of the extra filter applied to the normal."""
        return float(self.uniform_buffer.data["smoothing"])

    @smoothing.setter
    def smoothing(self, value):
        value = float(value)
        if not 0 <= value <= 16:
            raise ValueError(f"smoothing must be in [0, 16], got {value}")
        # Toggling smoothing on or off swaps shader variants
        was_on = float(self.uniform_buffer.data["smoothing"]) > 0
        self.uniform_buffer.data["smoothing"] = value
        self.uniform_buffer.update_full()
        if (value > 0) != was_on:
            self._store.smoothing_on = value > 0

    @property
    def shininess(self):
        """Size of the specular highlight on the isosurface."""
        return float(self.uniform_buffer.data["shininess"])

    @shininess.setter
    def shininess(self, value):
        self.uniform_buffer.data["shininess"] = float(value)
        self.uniform_buffer.update_full()

    @property
    def emissive(self):
        """Color the isosurface emits regardless of lighting."""
        return gfx.Color(self.uniform_buffer.data["emissive_color"])

    @emissive.setter
    def emissive(self, color):
        self.uniform_buffer.data["emissive_color"] = gfx.Color(color)
        self.uniform_buffer.update_full()


@register_wgpu_render_function(BitmaskVolume, BitmaskVolumeMaterial)
class BitmaskVolumeShader(BaseShader):
    type = "render"

    def __init__(self, wobject, **kwargs):
        super().__init__(wobject, **kwargs)

        packed = wobject.packed
        self["brick_size"] = int(packed.brick_size)
        self["words"] = int(packed.brick_size) ** 3 // 32
        self["shape_x"], self["shape_y"], self["shape_z"] = packed.shape
        self["gbx"], self["gby"], self["gbz"] = packed.grid_bricks
        self["mode"] = wobject.material.render_mode
        self["smooth"] = wobject.material.smoothing > 0

    def get_bindings(self, wobject, shared, scene):
        geometry = wobject.geometry
        material = wobject.material

        self["mode"] = material.render_mode
        self["smooth"] = material._store.smoothing_on

        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
            # r32uint, so non-filterable and read with textureLoad - no sampler
            Binding(
                "t_super",
                "texture/auto",
                GfxTextureView(geometry.super_index),
                "FRAGMENT",
            ),
            Binding(
                "s_bricktab", "buffer/read_only_storage", geometry.bricktab, "FRAGMENT"
            ),
            Binding("s_bits", "buffer/read_only_storage", geometry.bits, "FRAGMENT"),
        ]

        bindings = {i: b for i, b in enumerate(bindings)}
        self.define_bindings(0, bindings)

        return {0: bindings}

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.front,  # the back planes are the ref
        }

    def get_render_info(self, wobject, shared):
        return {"indices": (36, 1)}

    def get_code(self):
        return load_wgsl("bitmask_volume.wgsl", "octarine.shaders.wgsl")
