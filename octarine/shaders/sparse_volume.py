"""Custom pygfx WorldObject/Material/Shader for sparse volumetric data.

Renders sparse (N, 3) voxel data packed into a brick map (see `.packing`)
using two-level raycasting: a dense low-resolution index texture over the
bounding box, plus an atlas texture holding only the occupied bricks.

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
    GfxSampler,
    GfxTextureView,
    load_wgsl,
)

from .packing import PackedBricks


class SparseVolume(gfx.WorldObject):
    """A sparse volume defined by brick-packed voxel data.

    Parameters
    ----------
    packed :    PackedBricks
                Brick-packed voxel data (see `pack_sparse_voxels`).
    material :  SparseVolumeMaterial
                The material defining the appearance of the volume.

    """

    def __init__(self, packed, material, **kwargs):
        if not isinstance(packed, PackedBricks):
            raise TypeError(f"Expected PackedBricks, got {type(packed)}")

        shape = tuple(int(s) for s in packed.shape)
        # The corner positions let pygfx derive the bounding box from the
        # geometry (voxel centers sit at integer positions, hence the 0.5)
        corners = np.array(
            [[-0.5, -0.5, -0.5], [shape[0] - 0.5, shape[1] - 0.5, shape[2] - 0.5]],
            dtype=np.float32,
        )
        geometry = gfx.Geometry(
            coarse=gfx.Texture(packed.coarse, dim=3),
            atlas=gfx.Texture(packed.atlas, dim=3),
            positions=corners,
        )
        super().__init__(geometry, material, **kwargs)

        self.shape = shape
        self.brick_size = int(packed.brick_size)
        self.n_bricks = int(packed.n_bricks)


class SparseVolumeMaterial(gfx.VolumeBasicMaterial):
    """Material for rendering a SparseVolume.

    In addition to the properties of `pygfx.VolumeBasicMaterial` (`clim`,
    `map`, `gamma`, `interpolation`, `opacity`), this material has a
    `render_mode`:

     - "mip": maximum intensity projection (like `pygfx.VolumeMipMaterial`)
     - "density": front-to-back emission/absorption for a cloud-like look;
       `opacity` scales the extinction per voxel

    In addition, `step_size` (in voxels, default 0.5) controls the ray-march
    step inside occupied bricks: smaller = fewer misses of small structures
    but slower rendering.

    """

    uniform_type = dict(
        gfx.VolumeBasicMaterial.uniform_type,
        step_size="f4",
    )

    def __init__(self, render_mode="mip", step_size=0.5, **kwargs):
        super().__init__(**kwargs)
        self.render_mode = render_mode
        self.step_size = step_size

    @property
    def render_mode(self):
        """The render mode: "mip" or "density"."""
        return self._store.render_mode

    @render_mode.setter
    def render_mode(self, value):
        if value not in ("mip", "density"):
            raise ValueError(f"render_mode must be 'mip' or 'density', got {value!r}")
        self._store.render_mode = value

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


@register_wgpu_render_function(SparseVolume, SparseVolumeMaterial)
class SparseVolumeShader(BaseShader):
    type = "render"

    def __init__(self, wobject, **kwargs):
        super().__init__(wobject, **kwargs)

        self["brick_size"] = wobject.brick_size
        self["shape_x"], self["shape_y"], self["shape_z"] = wobject.shape

        # The atlas is r8unorm: values arrive in the shader as 0-1 floats.
        # These template vars make pygfx's `sampled_value_to_color()` (from
        # image_sample.wgsl) treat them like uint8 values, matching how the
        # packer quantizes into 1-255.
        self["img_format"] = "f32"
        self["img_nchannels"] = 1
        self["climcorrection"] = " * 255.0"

        material = wobject.material
        self["colorspace"] = "srgb"
        if material.map is not None:
            self["colorspace"] = material.map.texture.colorspace

    def get_bindings(self, wobject, shared, scene):
        geometry = wobject.geometry
        material = wobject.material

        self["mode"] = material.render_mode

        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
        ]

        # The coarse index is r32uint and therefore non-filterable; it is
        # read with textureLoad so it needs no sampler.
        coarse_view = GfxTextureView(geometry.coarse)
        bindings.append(Binding("t_coarse", "texture/auto", coarse_view, "FRAGMENT"))

        atlas_view = GfxTextureView(geometry.atlas)
        sampler = GfxSampler(material.interpolation, "clamp")
        bindings.append(Binding("s_atlas", "sampler/filtering", sampler, "FRAGMENT"))
        bindings.append(Binding("t_atlas", "texture/auto", atlas_view, "FRAGMENT"))

        if material.map is not None:
            bindings.extend(self.define_img_colormap(material.map))

        bindings = {i: b for i, b in enumerate(bindings)}
        self.define_bindings(0, bindings)

        return {
            0: bindings,
        }

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.front,  # the back planes are the ref
        }

    def get_render_info(self, wobject, shared):
        return {
            "indices": (36, 1),
        }

    def get_code(self):
        return load_wgsl("sparse_volume.wgsl", "octarine.shaders.wgsl")
