import six

import pygfx as gfx
import numpy as np
import pandas as pd

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
    visuals = [ob for ob in x if 'pygfx.objects' in str(type(ob))]

    if include_geometries:
        visuals += [ob for ob in x if 'pygfx.geometries' in str(type(ob))]

    # Collect scatter points
    scatter = [ob for ob in x if isinstance(ob, np.ndarray) and (ob.ndim == 2) and (ob.shape[1] == 3)]

    # Collect dataframes with X/Y/Z coordinates
    dataframes = [ob for ob in x if isinstance(ob, pd.DataFrame)]
    if [d for d in dataframes if False in np.isin(['x', 'y', 'z'], d.columns)]:
        logger.warning('DataFrames must have x, y and z columns.')
    dataframes = [d for d in dataframes if all(np.isin(['x', 'y', 'z'], d.columns))]
    scatter += [d[['x', 'y', 'z']].values for d in dataframes]

    # Collect volumes
    volumes = [ob for ob in x if isinstance(ob, np.ndarray) and (ob.ndim == 3)]

    # Collect meshes
    meshes = [ob for ob in x if is_mesh_like(ob)]

    # Collect dataframes with X/Y/Z coordinates
    dataframes = [ob for ob in x if isinstance(ob, pd.DataFrame)]
    if [d for d in dataframes if False in np.isin(['x', 'y', 'z'], d.columns)]:
        logger.warning('DataFrames must have x, y and z columns.')
    # Filter to and extract x/y/z coordinates
    dataframes = [d for d in dataframes if False not in [c in d.columns for c in ['x', 'y', 'z']]]
    dataframes = [d[['x', 'y', 'z']].values for d in dataframes]

    # Collect arrays
    arrays = [ob.copy() for ob in x if isinstance(ob, np.ndarray)]
    # Remove arrays with wrong dimensions
    if [ob for ob in arrays if ob.shape[1] != 3 and ob.shape[0] != 2]:
        logger.warning('Arrays need to be of shape (N, 3) for scatter or (2, N)'
                       ' for line plots.')
    arrays = [ob for ob in arrays if any(np.isin(ob.shape, [2, 3]))]

    points = dataframes + arrays

    return meshes, volumes, points, visuals


def make_iterable(x, force_type = None):
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
    if not isinstance(x, Iterable) or isinstance(x, six.string_types):
        x = [x]

    if isinstance(x, (dict, set)):
        x = list(x)

    return np.asarray(x, dtype=force_type)


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
    if isinstance(x, Iterable) and not isinstance(x, (six.string_types, pd.DataFrame)):
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
    if hasattr(x, 'vertices') and hasattr(x, 'faces'):
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
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        elif 'colab' in ipy_str:
            return 'colab'
        else:  # if 'terminal' in ipy_str:
            return 'ipython'
    except BaseException:
        return 'terminal'


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
    return _type_of_script() in ('jupyter', 'colab')