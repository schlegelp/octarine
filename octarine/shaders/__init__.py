"""Custom shaders that extend pygfx.

This subpackage hooks into pygfx's (semi-public) shader API, which can change
between pygfx releases — hence the version guard below. It is imported lazily
(on first use of e.g. `Viewer.add_sparse_volume`) so that importing octarine
itself does not depend on it.
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
from .sparse_volume import SparseVolume, SparseVolumeMaterial  # noqa: E402
from .silhouette import SilhouetteMeshMaterial  # noqa: E402

__all__ = [
    "pack_sparse_voxels",
    "PackedBricks",
    "AtlasCapacityError",
    "SparseVolume",
    "SparseVolumeMaterial",
    "SilhouetteMeshMaterial",
]
