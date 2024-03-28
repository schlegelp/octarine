import pygfx as gfx

from .utils import is_hashable, is_points, is_lines, is_volume, is_mesh_like, is_pygfx_visual, is_pygfx_geometry
from .visuals import points2gfx, lines2gfx, mesh2gfx, volume2gfx


CONVERTERS = {
    is_mesh_like : mesh2gfx,
    is_points: points2gfx,
    is_lines: lines2gfx,
    is_volume: volume2gfx,
    is_pygfx_visual: lambda x: x,  # pass-through
    is_pygfx_geometry: lambda x: gfx.Mesh(x, gfx.MeshPhongMaterial()),  # add default material and return
}


def get_converter(t, raise_missing=True):
    """Get the converter for a given data type."""
    # First check if we have a direct match
    if is_hashable(t) and t in CONVERTERS:
        return CONVERTERS[t]

    # If not, check if we have a converter for the type of `t`
    if type(t) in CONVERTERS:
        return CONVERTERS[type(t)]

    # If that also failed, see if we any of the converters keys are functions
    for k, v in CONVERTERS.items():
        if callable(k):
            try:
                if k(t):
                    return v
            except Exception:
                pass

    if not raise_missing:
        return None

    raise NotImplementedError(f"No converter found for {t} ({type(t)}).")


def register_converter(t, converter):
    """Register a converter for a given data type.

    Parameters
    ----------
    t :         type | hashable | callable
                Data type to register the converter for. If a function
                it is expected to take a single argument `x` and return
                True if `x` can be converted using `converter`.
    converter : callable
                Function that converts `x` to pygfx visuals. Must accept
                at least a single argument and return either a single
                visual or a list thereof.

    """
    if not callable(converter):
        raise ValueError("Converter must be callable.")

    if not callable(t) and not is_hashable(t):
        raise ValueError("Type must be hashable or callable.")

    CONVERTERS[t] = converter