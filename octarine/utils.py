import six

import pygfx as gfx
import numpy as np

from collections.abc import Iterable

from . import config

# Set up logging
logger = config.get_logger(__name__)


def parse_objects(x, include_geometries=True):
    """Categorize objects e.g. for plotting.

    Returns
    -------
    meshes :       list of mesh-likes
    volumes :      list of (N, M, K) arrays
    points :       list of (N, 3) arrays
    visual :       list of pygfx visuals

    """
    # Make sure this is a list.
    if not isinstance(x, (list, tuple)):
        x = [x]

    # If any list in x, flatten first
    if any([isinstance(i, list) for i in x]):
        # We need to be careful to preserve order
        # to not break assignment of colors
        y = []
        for i in x:
            y += i if isinstance(i, list) else [i]
        x = y

    # Collect visuals
    visuals = [ob for ob in x if "pygfx.objects" in str(type(ob))]

    if include_geometries:
        visuals += [ob for ob in x if "pygfx.geometries" in str(type(ob))]

    # Collect scatter points
    scatter = [
        ob
        for ob in x
        if isinstance(ob, np.ndarray) and (ob.ndim == 2) and (ob.shape[1] == 3)
    ]

    # Collect dataframes with X/Y/Z coordinates
    dataframes = [ob for ob in x if _is_pandas_dataframe(ob)]
    if [d for d in dataframes if False in np.isin(["x", "y", "z"], d.columns)]:
        logger.warning("DataFrames must have x, y and z columns.")
    dataframes = [d for d in dataframes if all(np.isin(["x", "y", "z"], d.columns))]
    scatter += [d[["x", "y", "z"]].values for d in dataframes]

    # Collect volumes
    volumes = [ob for ob in x if isinstance(ob, np.ndarray) and (ob.ndim == 3)]

    # Collect meshes
    meshes = [ob for ob in x if is_mesh_like(ob)]

    # Collect arrays
    arrays = [ob.copy() for ob in x if isinstance(ob, np.ndarray)]
    # Remove arrays with wrong dimensions
    if [ob for ob in arrays if ob.shape[1] != 3 and ob.shape[0] != 2]:
        logger.warning(
            "Arrays need to be of shape (N, 3) for scatter or (2, N)" " for line plots."
        )
    arrays = [ob for ob in arrays if any(np.isin(ob.shape, [2, 3]))]

    points = dataframes + arrays

    return meshes, volumes, points, visuals


def _is_pandas_dataframe(x):
    """Check if object is a pandas DataFrame."""
    # We're doing this without the use of isinstance() to avoid
    # needing pandas as a dependency
    if not hasattr(x, "__class__"):
        return False
    # Check if any of the parent classes is a pandas DataFrame
    for b in x.__class__.__mro__:
        if b.__module__.startswith("pandas") and b.__name__ == "DataFrame":
            return True
    return False


def make_iterable(x, force_type=None):
    """Force input into a numpy array.

    For dicts, keys will be turned into array.

    Examples
    --------
    >>> import octarine as oc
    >>> oc.utils.make_iterable(1)
    array([1])
    >>> oc.utils.make_iterable([1])
    array([1])
    >>> oc.utils.make_iterable({'a': 1})
    array(['a'], dtype='<U1')

    """
    if (
        not isinstance(x, Iterable)
        or isinstance(x, six.string_types)
        or isinstance(x, gfx.Geometry)
    ):
        x = [x]

    if isinstance(x, (dict, set)):
        x = list(x)

    return np.asarray(x, dtype=force_type)


def as_color_list(*colors):
    """Turn one or more color specs into a list of `pygfx.Color`.

    Both `as_color_list("r", "g")` and `as_color_list(["r", "g"])` give the
    same result, i.e. colors can be passed as separate arguments or as a
    single sequence. Note that a sequence of numbers - e.g. `(1, 0, 0)` - is
    always treated as a single RGB(A) color.

    Examples
    --------
    >>> import octarine as oc
    >>> oc.utils.as_color_list("red")
    [Color(1.0, 0.0, 0.0, 1.0)]
    >>> oc.utils.as_color_list([(1, 0, 0), "blue"])
    [Color(1.0, 0.0, 0.0, 1.0), Color(0.0, 0.0, 1.0, 1.0)]

    """
    if len(colors) == 1 and _is_color_sequence(colors[0]):
        colors = tuple(colors[0])
    return [gfx.Color(c) for c in colors]


def background_options(viewer):
    """Options for the background dropdown in the GUI controls.

    Returns a list of labels, the corresponding values to pass to
    `Viewer.set_bg_gradient()`, and the index of the viewer's current
    background.

    """
    labels, values = ["Plain"], [None]

    # The gradients require octarine's custom shaders (pygfx >= 0.16)
    try:
        from .shaders import BACKGROUND_PRESETS, GradientBackgroundMaterial
    except ImportError:
        return labels, values, 0

    labels.extend(name.title() for name in BACKGROUND_PRESETS)
    values.extend(BACKGROUND_PRESETS)

    mat = viewer._background.material
    if not isinstance(mat, GradientBackgroundMaterial):
        return labels, values, 0
    elif mat.preset in BACKGROUND_PRESETS:
        return labels, values, values.index(mat.preset)

    # A gradient set via the API with custom parameters: add an entry for it
    # so that we can switch back to it
    labels.append("Custom")
    values.append(
        dict(
            colors=mat.colors,
            center=mat.center,
            radius=mat.radius,
            falloff=mat.falloff,
            vignette=mat.vignette,
        )
    )
    return labels, values, len(values) - 1


def _is_color_sequence(x):
    """Check if `x` is a sequence of colors rather than a single color."""
    if isinstance(x, (str, gfx.Color)) or not isinstance(x, (list, tuple, np.ndarray)):
        return False
    # A single color is a sequence of numbers - anything that isn't must be
    # a color in its own right
    return len(x) > 0 and all(
        isinstance(c, (str, list, tuple, np.ndarray, gfx.Color)) for c in x
    )


def is_iterable(x) -> bool:
    """Test if input is iterable (but not str).

    Examples
    --------
    >>> import octarine as oc
    >>> oc.utils.is_iterable(['a'])
    True
    >>> oc.utils.is_iterable('a')
    False
    >>> oc.utils.is_iterable({'a': 1})
    True

    """
    if (
        isinstance(x, Iterable)
        and not isinstance(x, six.string_types)
        and not _is_pandas_dataframe(x)
        and not isinstance(x, gfx.Geometry)
    ):
        return True
    else:
        return False


def is_hashable(x) -> bool:
    """Check if object is hashable."""
    try:
        hash(x)
        return True
    except TypeError:
        return False


def is_mesh_like(x):
    """Check if object is mesh (i.e. contains vertices and faces)."""
    if hasattr(x, "vertices") and hasattr(x, "faces"):
        return True

    return False


def is_points(x):
    """Check if object could be points (i.e. contains 3D coordinates)."""
    if isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] == 3:
        return True

    return False


def is_lines(x):
    """Check if object could be lines (i.e. contains 3D coordinates)."""
    if isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] == 3:
        return True

    return False


def is_volume(x):
    """Check if object could be a volume (i.e. 3D array)."""
    if isinstance(x, np.ndarray) and x.ndim == 3:
        return True

    return False


class VoxelCloud:
    """Container for sparse volumetric data.

    Wrapping voxel coordinates in this class tells `Viewer.add` to render
    them as a (sparse) volume instead of as points.

    Parameters
    ----------
    coords :    (N, 3) array
                Voxel coordinates (xyz). Floats are floored to integers.
    values :    (N,) array, optional
                Per-voxel scalar values.

    """

    def __init__(self, coords, values=None):
        coords = np.asarray(coords)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) array, got {coords.shape}")
        if values is not None:
            values = np.asarray(values).ravel()
            if len(values) != len(coords):
                raise ValueError(
                    f"Got {len(values)} values for {len(coords)} voxels."
                )
        self.coords = coords
        self.values = values

    def __len__(self):
        return len(self.coords)

    def __repr__(self):
        return f"<VoxelCloud with {len(self):,} voxels>"


class VoxelRuns:
    """Container for run-length-encoded sparse volumetric data.

    Wrapping voxel runs in this class tells `Viewer.add` to render them as a
    (sparse) volume. Runs are rendered as binary occupancy from a bitmask,
    which uses roughly 23x less GPU memory than the same data given as
    `VoxelCloud` coordinates - see [Sparse Volumes](../objects.md#sparse-volumes).

    Parameters
    ----------
    runs :      (N, 4) array
                Runs as (x, y, z, x_run_length), i.e. the layout returned by
                `dvid.get_sparsevol(..., voxels=False)`. Runs extend along x
                and the length is an inclusive voxel count.

    """

    def __init__(self, runs):
        runs = np.asarray(runs)
        if runs.ndim != 2 or runs.shape[1] != 4:
            raise ValueError(f"Expected (N, 4) array, got {runs.shape}")
        self.runs = runs

    def __len__(self):
        return len(self.runs)

    @property
    def n_voxels(self):
        """Number of occupied voxels."""
        return int(np.asarray(self.runs)[:, 3].astype(np.int64).sum())

    def __repr__(self):
        return f"<VoxelRuns with {len(self):,} runs ({self.n_voxels:,} voxels)>"


def is_pygfx_visual(x):
    """Check if object is a pygfx visual."""
    if isinstance(x, gfx.WorldObject):
        return True
    return False


def is_pygfx_geometry(x):
    """Check if object is a pygfx geometry."""
    if isinstance(x, gfx.Geometry):
        return True
    return False


def _type_of_script() -> str:
    """Return context (terminal, jupyter, colab, iPython) in which navis is run."""
    try:
        ipy_str = str(type(get_ipython()))  # noqa: F821
        if "zmqshell" in ipy_str:
            return "jupyter"
        elif "colab" in ipy_str:
            return "colab"
        else:  # if 'terminal' in ipy_str:
            return "ipython"
    except BaseException:
        try:
            # This is not perfect but should work in most cases
            if hasattr(__main__.__file__):  # noqa: F821
                return "script"
        except BaseException:
            pass
        return "terminal"


def is_jupyter() -> bool:
    """Test if navis is run in a Jupyter notebook.

    Also returns True if inside Google colaboratory!

    Examples
    --------
    >>> from navis.utils import is_jupyter
    >>> # If run outside a Jupyter environment
    >>> is_jupyter()
    False

    """
    return _type_of_script() in ("jupyter", "colab")
