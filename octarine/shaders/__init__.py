"""Custom shaders that extend pygfx.

This subpackage hooks into pygfx's (semi-public) shader API, which can change
between pygfx releases — hence the version guard below. It is imported lazily
(from `Viewer.__init__`, which needs `pcf`, and on first use of e.g.
`Viewer.add_sparse_volume`) so that importing octarine itself does not depend
on it.
"""


def _check_pygfx_version():
    import pygfx

    version = tuple(int(x) for x in pygfx.__version__.split(".")[:2] if x.isdigit())
    if version < (0, 16):
        raise ImportError(
            "octarine's custom shaders require pygfx>=0.16.0, you have "
            f"{pygfx.__version__}. Please update: pip install -U pygfx"
        )


_check_pygfx_version()

from .packing import pack_sparse_voxels, PackedBricks, AtlasCapacityError  # noqa: E402
from .rle_packing import (  # noqa: E402
    pack_voxel_runs,
    runs_from_voxels,
    PackedBitmask,
    BitmaskCapacityError,
)
from .sparse_volume import SparseVolume, SparseVolumeMaterial  # noqa: E402
from .bitmask_volume import BitmaskVolume, BitmaskVolumeMaterial  # noqa: E402
from .tubes import (  # noqa: E402
    TubeVisual,
    TubeMaterial,
    align_frames,
    decimate_edges,
    rotate_profile,
    split_runs,
)
from .silhouette import SilhouetteMeshMaterial  # noqa: E402
from .sss import SubsurfaceMeshMaterial, SUBSURFACE_PROPERTIES  # noqa: E402
from .background import (  # noqa: E402
    GradientBackgroundMaterial,
    BACKGROUND_PRESETS,
)
from .dof import DepthOfFieldPass  # noqa: E402
from .depth import NormalizedDepthPass  # noqa: E402
from .ao import AmbientOcclusionPass  # noqa: E402
from .outline import OutlinePass  # noqa: E402
from .tonemap import ToneMappingPass, TONEMAP_MODES  # noqa: E402
from .environment import (  # noqa: E402
    ENVIRONMENT_PRESETS,
    ENVIRONMENT_PROPERTIES,
    cube_directions,
    environment_radiance,
    procedural_env_map,
)
from .matcap import (  # noqa: E402
    MATCAP_PRESETS,
    MATCAP_PROPERTIES,
    MatcapMeshMaterial,
    make_matcap,
    matcap_texture,
)
from .points import FlexPointsMaterial  # noqa: E402
from .lines import FlexLineMaterial  # noqa: E402

__all__ = [
    "pack_sparse_voxels",
    "PackedBricks",
    "AtlasCapacityError",
    "pack_voxel_runs",
    "runs_from_voxels",
    "PackedBitmask",
    "BitmaskCapacityError",
    "SparseVolume",
    "SparseVolumeMaterial",
    "BitmaskVolume",
    "BitmaskVolumeMaterial",
    "TubeVisual",
    "TubeMaterial",
    "align_frames",
    "decimate_edges",
    "rotate_profile",
    "split_runs",
    "SilhouetteMeshMaterial",
    "SubsurfaceMeshMaterial",
    "SUBSURFACE_PROPERTIES",
    "GradientBackgroundMaterial",
    "BACKGROUND_PRESETS",
    "DepthOfFieldPass",
    "NormalizedDepthPass",
    "AmbientOcclusionPass",
    "OutlinePass",
    "ToneMappingPass",
    "TONEMAP_MODES",
    "ENVIRONMENT_PRESETS",
    "ENVIRONMENT_PROPERTIES",
    "cube_directions",
    "environment_radiance",
    "procedural_env_map",
    "MATCAP_PRESETS",
    "MATCAP_PROPERTIES",
    "MatcapMeshMaterial",
    "make_matcap",
    "matcap_texture",
    "FlexPointsMaterial",
    "FlexLineMaterial",
]
