import uuid
import cmap

import pygfx as gfx
import numpy as np
import trimesh as tm

from importlib.util import find_spec

from . import config, utils

logger = config.get_logger(__name__)


def mesh2gfx(mesh, color, alpha=None):
    """Convert generic mesh to pygfx visuals.

    Parameters
    ----------
    mesh :          mesh-like
                    Mesh to convert.
    color :         str | tuple | array
                    Color to use for plotting. If multiple colors,
                    must be a list of colors with the same length as
                    the number of faces or vertices.
    alpha :         float, optional
                    Opacity value [0-1]. If provided, will override
                    the alpha channel of the color.

    """
    # Skip empty meshes
    if not len(mesh.faces):
        return

    # Parse color
    mat_color_kwargs, obj_color_kwargs = parse_mesh_color(mesh, color, alpha)
    # In theory we should be able to change pick_write on-the-fly since pygfx 0.3.0
    # But that doesn't seem to be the case.
    mat_color_kwargs['pick_write'] = True

    vis = gfx.Mesh(
        gfx.Geometry(
            indices=mesh.faces.astype(np.int32, copy=False),
            positions=mesh.vertices.astype(np.float32, copy=False),
            **obj_color_kwargs,
        ),
        gfx.MeshPhongMaterial(**mat_color_kwargs),
    )

    # Add custom attributes
    vis._object_type = "mesh"
    vis._object_id = uuid.uuid4()

    return vis


def geometry2gfx(geometry, color, alpha=None):
    """Convert a pygfx.Geometry to a pygfx.Mesh.

    Parameters
    ----------
    geometry :      pygfx.Geometry
                    Geometry to convert.
    color :         str | tuple | array
                    Color to use for plotting. If multiple colors,
                    must be a list of colors with the same length as
                    the number of faces or vertices.
    alpha :         float, optional
                    Opacity value [0-1]. If provided, will override
                    the alpha channel of the color.

    """
    # Parse color
    mat_color_kwargs, obj_color_kwargs = parse_mesh_color(geometry, color, alpha)

    if "colors" in obj_color_kwargs:
        geometry.colors = obj_color_kwargs["colors"]

    # In theory we should be able to change pick_write on-the-fly since pygfx 0.3.0
    # But that doesn't seem to be the case.
    mat_color_kwargs['pick_write'] = True

    vis = gfx.Mesh(geometry, gfx.MeshPhongMaterial(**mat_color_kwargs))

    # Add custom attributes
    vis._object_type = "mesh"
    vis._object_id = uuid.uuid4()

    return vis


def parse_mesh_color(mesh, color, alpha=None):
    """Parse color for mesh plotting."""
    mat_color_kwargs = dict()
    obj_color_kwargs = dict()
    if isinstance(color, np.ndarray) and color.ndim == 2:
        if alpha is not None:
            if color.shape[1] == 3:
                color = np.hstack((color, np.ones((color.shape[0], 1))))
            elif color.shape[1] != 4:
                raise ValueError("Expected colors to have 3 or 4 channels.")
            color[:, -1] = alpha

        # Make sure the color is what pygfx expects
        if color.dtype in (np.float64,):
            color = color.astype(np.float32, copy=False)

        if len(color) == len(mesh.vertices):
            obj_color_kwargs = dict(colors=color)
            mat_color_kwargs = dict(color_mode="vertex")
        elif len(color) == len(mesh.faces):
            obj_color_kwargs = dict(colors=color)
            mat_color_kwargs = dict(color_mode="face")
        else:
            raise ValueError(
                "Expected colors to have the same length as the number of vertices or faces."
            )
    else:
        if alpha is not None:
            color = gfx.Color(color).rgba
            color = (color[0], color[1], color[2], alpha)

        mat_color_kwargs["color"] = color

    return mat_color_kwargs, obj_color_kwargs


def color_to_texture(color, N=256, gamma=1.0, fade=True):
    """Convert a given color to a pygfx Texture."""
    # First force RGB
    stop = gfx.Color(color).rgba
    stop = (stop.r, stop.g, stop.b, 1.0)  # make sure alpha is 1
    start = gfx.Color(color if not fade else "k").rgba
    start = (start.r, start.g, start.b, 0)  # make sure alpha is 0

    # Need to double check that pygfx properly interpolates the color
    colormap_data = np.vstack((start, stop)).astype(np.float32, copy=False)

    # Important note:
    # It looks that as things stand now, pygfx expects the colormap to be only
    # rgb, not rgba. So we need to remove the alpha channel.
    colormap_data = colormap_data[:, :3]

    # Convert to vispy cmap
    return gfx.Texture(colormap_data, dim=1)


def volume2gfx(
    vol,
    color,
    opacity=1.0,
    spacing=(1, 1, 1),
    offset=(0, 0, 0),
    clim="data",
    slice=False,
    interpolation="linear",
    hide_zero=True,
):
    """Convert volume (i.e. 3d arrays) to pygfx visual.

    Parameters
    ----------
    vol :           np.ndarray
                    3D array representing the volume.
    spacing :       tuple
                    Spacing between voxels in the volume.
    color :         color | list of colors | pygfx.Texture, optional
                    Colormap to render the volume. This can be:
                      - name of a colormap (e.g. "viridis" or "magma")
                      - a single color (name, hex, rgb, rgba)
                      - a list of colors
                      - a 1D pygfx.Texture
                    Note that single colors typically don't look good and
                    it's better to define at least two colors. For example,
                    instead of "red" use ["red", "yellow"]. If `None` will
                    use one of the built-in pygfx colormaps.
    opacity :       float, optional
                    Opacity of the volume.
    offset :        tuple, optional
                    Offset to apply to the volume.
    clim :          "data" | "datatype" | tuple, optional
                    The contrast limits to scale the data values with.
                      - "data" (default) will use the min/max of the data
                      - "datatype" will use (0, theoretical max of data type)
                        for integer data, e.g. (0, 255) for int8 and uint8,
                        and (0, 1) for float data assuming the data has been
                        normalized
                      - tuple of min/max values or combination of "data" and
                        "datatype" strings
    slice :         bool | tuple, optional
                    Render volume slices instead of the full volume:
                      - True: render slices in all dimensions
                      - tuple of bools, e.g. `(True, True, False)`: render slices
                        in the respective dimensions
                      - tuple of floats, e.g. `(0.5, 0.5, 0.5)`: render slices
                        at the respective positions (relative to the volume size)
    interpolation : str, optional
                    Interpolation method to use. Either "linear" or "nearest".
    hide_zero :     bool, optional
                    If True, will set the alpha for the lowest value to 0.

    Returns
    -------
    vis :           gfx.Volume
                    Pygfx visual representing the volume.

    """
    # TODOs:
    # - add support for other Volume materials (e.g. gfx.VolumeMipMaterial)

    assert isinstance(vol, np.ndarray), "Expected 3D numpy array."
    assert vol.ndim == 3, "Expected 3D numpy array."
    assert isinstance(spacing, (tuple, list, np.ndarray, int, float))
    if isinstance(spacing, (int, float)):
        spacing = [spacing] * 3
    assert len(spacing) == 3, "Expected spacing as tuple of length 3."

    # Similar to vispy, pygfx seems to expect zyx coordinate space
    grid = vol.T

    # Convert to data type that pygfx can handle:
    # Convert non-native byte order to native; e.g. >u4 -> u4 = uint64
    if grid.dtype.byteorder in (">", "<"):
        grid = grid.astype(grid.dtype.str.replace(grid.dtype.byteorder, ""))
    # Convert boolean matrices to uint16; I tried uint4 but that renders as
    # uniform volume and uint8 looks fuzzy
    elif grid.dtype == bool:
        grid = grid.astype(np.uint16)

    # Find the potential min/max value of the volume
    if isinstance(clim, str):
        if clim == "datatype":
            cmin = cmax = "datatype"
        elif isinstance(clim, str) and clim == "data":
            cmin = cmax = "data"
        else:
            raise ValueError(f"Invalid value for clim: {clim}")
    else:
        cmin, cmax = clim

    if cmin == "datatype":
        cmin = 0
    elif cmin == "data":
        cmin = grid.min()

    if cmax == "datatype":
        # If float, assume that the data is normalized
        if grid.dtype.kind == "f":
            cmax = 1
        # Otherwise, use the maximum value of the data type
        else:
            cmax = np.iinfo(grid.dtype).max
    elif cmax == "data":
        cmax = grid.max()

    # Initialize texture
    tex = gfx.Texture(grid, dim=3)

    # Initialize colormap (and make copy of the data to avoid issues
    # with the original colormap data being modified)
    cmap = to_colormap(color, hide_zero=hide_zero)

    # Initialize the volume
    visuals = []
    if slice in (False, None):
        vis = gfx.Volume(
            gfx.Geometry(grid=tex),
            gfx.VolumeMipMaterial(
                clim=(cmin, cmax),
                map=cmap,
                interpolation=interpolation,
            ),
        )
        visuals.append(vis)
    else:
        if isinstance(slice, bool):
            slice = (0.5, 0.5, 0.5)
        elif not isinstance(slice, (list, tuple)):
            raise ValueError("Expected `slice` as bool or tuple.")
        elif len(slice) != 3:
            raise ValueError("Expected `slice` as bool or tuple of length 3.")

        slice = list(slice)
        for ix, dim in enumerate([2, 1, 0]):  # xyz
            # Skip if we don't want to render this slice
            if isinstance(slice[ix], bool):
                if slice[ix]:
                    slice[ix] = 0.5
                else:
                    continue

            abcd = [0, 0, 0, 0]
            abcd[dim] = -1
            abcd[-1] = grid.shape[2 - dim] / (1 / slice[ix]) * spacing[dim]
            material = gfx.VolumeSliceMaterial(clim=(cmin, cmax), plane=abcd)
            visuals.append(gfx.Volume(gfx.Geometry(grid=tex), material))

    # Set scales and offset
    for vis in visuals:
        vis.material.opacity = opacity
        (
            vis.local.scale_x,
            vis.local.scale_y,
            vis.local.scale_z,
        ) = spacing
        (vis.local.x, vis.local.y, vis.local.z) = offset

        # Add custom attributes
        vis._object_type = "volume"
        vis._object_id = uuid.uuid4()

        # Note: to trigger an update of the colormap data later:
        # vis.material.data[:, 1] = 0
        # vis.material.map.update_range((0, 0, 0), vis.material.map.size)

    return visuals


def to_colormap(x, hide_zero):
    """Convert `x` to a gfx.Texture that can be used for Volumes."""
    # If this is a texture
    if x is None:
        tex = gfx.cm.cividis
    elif isinstance(x, gfx.Texture):
        if x.dim != 1:
            raise ValueError("Expected 1D texture.")
        tex = x
    elif isinstance(x, str) and hasattr(gfx.cm, x):
        tex = getattr(gfx.cm, x)
    elif isinstance(x, gfx.Color):
        # cmap needs a list of colors (even if len == 1)
        tex = cmap.Colormap([x.rgba]).to_pygfx()
    elif isinstance(x, cmap.Colormap):
        tex = x.to_pygfx()
    elif isinstance(x, (dict, list)):
        # cmap can interpret dict and list of colors
        tex = cmap.Colormap(x).to_pygfx()
    elif isinstance(x, str):
        tex = cmap.Colormap(x).to_pygfx()
    else:
        # Last ditch effort: see if cmap can handle it
        c = cmap.Colormap([x])

        # If x is a single (RGB) color, cmap will create a colormap
        # with the first color being `None` and the second being `x`.
        # We need to set the first color to black
        if len(c.color_stops) == 2 and c.color_stops[0].color == "none":
            c = cmap.Colormap(["k", x])

        tex = c.to_pygfx()

    if hide_zero:
        # Add an alpha column if needed
        if tex.data.shape[1] == 3:
            np_ver = [int(i) for i in  np.__version__.split('.')]
            # Prior to version 1.24.0, numpy's hstack did not accept a `dtype`
            # parameter directly
            if np_ver[0] <= 1 and np_ver[1] < 24:
                colors = np.hstack(
                    (tex.data, np.ones((tex.data.shape[0], 1)))
                ).astype(tex.data.dtype)
            else:
                colors = np.hstack(
                    (tex.data, np.ones((tex.data.shape[0], 1))), dtype=tex.data.dtype
                )
            tex = gfx.Texture(colors, dim=1)
        # Otherwise make a copy to avoid modifying the original data
        else:
            tex = gfx.Texture(tex.data.copy(), dim=1)

        # Set alpha channel for first color to 0
        tex.data[0, 3] = 0

    return tex


def points2gfx(points, color, size=2, marker=None, size_space="screen"):
    """Convert points to pygfx visuals.

    Parameters
    ----------
    points :        (N, 3) array
                    Points to plot.
    color :         tuple | array
                    Color to use for plotting. If multiple colors,
                    must be a list of colors with the same length as
                    the number of points.
    size :          int, optional
                    Marker size.
    marker :        str, optional
                    See gfx.MarkerShape for available markers.
    size_space :    "screen" | "world" | "model", optional
                    Units to use for the marker size. "screen" (default)
                    will keep the line width constant on the screen, while
                    "world" and "model" will keep it constant in world and
                    model coordinates, respectively.

    Returns
    -------
    vis :           gfx.Points
                    Pygfx visual for points.

    """
    # TODOs:
    # - add support for per-vertex sizes and colors
    assert isinstance(points, np.ndarray), "Expected numpy array."
    assert points.ndim == 2, "Expected 2D numpy array."
    assert points.shape[1] == 3, "Expected (N, 3) array."

    points = points.astype(np.float32, copy=False)

    # Make sure coordinates are c-contiguous
    if not points.flags["C_CONTIGUOUS"]:
        points = np.ascontiguousarray(points)

    geometry_kwargs = {}
    material_kwargs = {}
    # In theory we should be able to change pick_write on-the-fly since pygfx 0.3.0
    # But that doesn't seem to be the case.
    material_kwargs['pick_write'] = True

    # Parse sizes
    if utils.is_iterable(size):
        if len(size) != len(points):
            raise ValueError(
                "Expected `size` to be a single value or "
                "an array of the same length as `points`."
            )
        geometry_kwargs["sizes"] = np.asarray(size).astype(np.float32, copy=False)
        material_kwargs["size_mode"] = "vertex"
    else:
        material_kwargs["size"] = size

    # Parse color(s)
    if isinstance(color, np.ndarray) and color.ndim == 2:
        # If colors are provided for each node we have to make sure
        # that we also include `None` for the breaks in the segments
        n_points = len(points)
        if len(color) != n_points:
            raise ValueError(f"Got {len(color)} colors for {n_points} points.")
        color = color.astype(np.float32, copy=False)
        geometry_kwargs["colors"] = color
        material_kwargs["color_mode"] = "vertex"
    else:
        if isinstance(color, np.ndarray):
            color = color.astype(np.float32, copy=False)
        material_kwargs["color"] = color

    if marker is None:
        material = gfx.PointsMaterial(size_space=size_space, **material_kwargs)
    else:
        material = gfx.PointsMarkerMaterial(
            marker=marker, size_space=size_space, **material_kwargs
        )

    vis = gfx.Points(gfx.Geometry(positions=points, **geometry_kwargs), material)

    # Add custom attributes
    vis._object_type = "points"
    vis._object_id = uuid.uuid4()

    return vis


def lines2gfx(lines, color, linewidth=1, linewidth_space="screen", dash_pattern=None):
    """Convert lines into pygfx visuals.

    Parameters
    ----------
    lines :     list of (N, 3) arrays | (N, 3) array
                Lines to plot. If a list of arrays, each array
                represents a separate line. If a single array,
                each row represents a point in the line. You can
                introduce breaks in the line by inserting NaNs.
    color :     str | tuple, optional
                Color to use for plotting. Can be a single color
                or one for every point in the line(s).
    linewidth : float, optional
                Line width. Set to 0 to use thin lines which can speed
                up rendering.
    linewidth_space : "screen" | "world" | "model", optional
                Units to use for the line width. "screen" (default)
                will keep the line width constant on the screen, while
                "world" and "model" will keep it constant in world and
                model coordinates, respectively.
    dash_pattern : "solid" | "dashed" | "dotted" | "dashdot" | tuple, optional
                Line style to use. If a tuple, must define the on/off
                sequence.

    Returns
    -------
    vis :           gfx.Line
                    Pygfx visuals for lines.

    """
    if isinstance(lines, np.ndarray):
        assert lines.ndim == 2
        assert lines.shape[1] == 3
        assert len(lines) > 1
    elif isinstance(lines, list):
        assert all([isinstance(l, np.ndarray) for l in lines])
        assert all([l.ndim == 2 for l in lines])
        assert all([l.shape[1] == 3 for l in lines])
        assert all([len(l) > 1 for l in lines])

        # Convert to the (N, 3) format
        if len(lines) == 1:
            lines = lines[0]
        else:
            lines = np.insert(
                np.vstack(lines),
                np.cumsum([len(l) for l in lines[:-1]]),
                np.nan,
                axis=0,
            )
    else:
        raise TypeError("Expected numpy array or list of numpy arrays.")

    if dash_pattern is None:
        dash_pattern = ()  # pygfx expects an empty tuple for solid lines
    elif isinstance(dash_pattern, str):
        if dash_pattern in ("solid", '-'):
            dash_pattern = ()
        elif dash_pattern in ("dashed", "--"):
            dash_pattern = (5, 2)
        elif dash_pattern in ("dotted", ":"):
            dash_pattern = (1, 2)
        elif dash_pattern in ("dashdot", "-."):
            dash_pattern = (5, 2, 1, 2)
        else:
            raise ValueError(f"Unknown dash pattern: {dash_pattern}")

    geometry_kwargs = {}
    material_kwargs = {}

    # In theory we should be able to change pick_write on-the-fly since pygfx 0.3.0
    # But that doesn't seem to be the case.
    material_kwargs['pick_write'] = True

    # Parse color(s)
    if isinstance(color, np.ndarray) and color.ndim == 2:
        # If colors are provided for each node we have to make sure
        # that we also include `None` for the breaks in the segments

        # See if we can rescue this if there is now a mismatch in the
        # number of colors and points
        if len(color) != len(lines):
            # Count the number of non-NaN points
            n_points = (~np.isnan(lines[:, 0])).sum()
            if n_points != len(lines):
                if len(color) == n_points:
                    breaks = np.where(np.isnan(lines[:, 0]))[0]
                    offset = np.arange(len(breaks))
                    color = np.insert(color, breaks - offset, np.nan, axis=0)
                else:
                    raise ValueError(f"Got {len(color)} colors for {n_points} points.")
        color = color.astype(np.float32, copy=False)
        geometry_kwargs["colors"] = color
        material_kwargs["color_mode"] = "vertex"
    else:
        if isinstance(color, np.ndarray):
            color = color.astype(np.float32, copy=False)
        material_kwargs["color"] = color

    if linewidth > 0:
        mat = gfx.LineMaterial(
            thickness=linewidth,
            thickness_space=linewidth_space,
            dash_pattern=dash_pattern,
            **material_kwargs,
        )
    else:
        mat = gfx.LineThinMaterial(
            **material_kwargs,
        )

    vis = gfx.Line(
        gfx.Geometry(positions=lines.astype(np.float32, copy=False), **geometry_kwargs),
        mat,
    )

    # Add custom attributes
    vis._object_type = "lines"
    vis._object_id = uuid.uuid4()

    return vis


def trimesh2gfx(mesh, color=None, alpha=None, use_material=True):
    """Convert trimesh to pygfx visual.

    Importantly, this function will also try to extract textures
    if applicable.

    """
    assert isinstance(mesh, tm.Trimesh), f"Expected trimesh.Trimesh, got {type(mesh)}."

    kwargs = dict(
        positions=np.ascontiguousarray(mesh.vertices, dtype="f4"),
        indices=np.ascontiguousarray(mesh.faces, dtype="i4"),
    )
    # trimesh needs scipy to compute normals
    if find_spec("scipy"):
        kwargs['normals'] = np.ascontiguousarray(mesh.vertex_normals, dtype="f4")

    if mesh.visual.kind == "texture" and getattr(mesh.visual, "uv", None) is not None:
        # convert the uv coordinates from opengl to wgpu conventions.
        # wgpu uses the D3D and Metal coordinate systems.
        # the coordinate origin is in the upper left corner, while the opengl coordinate
        # origin is in the lower left corner.
        # trimesh loads textures according to the opengl coordinate system.
        wgpu_uv = mesh.visual.uv * np.array([1, -1]) + np.array(
            [0, 1]
        )  # uv.y = 1 - uv.y
        kwargs["texcoords"] = np.ascontiguousarray(wgpu_uv, dtype="f4")
    elif mesh.visual.kind == "vertex":
        kwargs["colors"] = np.ascontiguousarray(mesh.visual.vertex_colors, dtype="f4")

    # Generate the geometry
    vis = gfx.Mesh(gfx.Geometry(**kwargs), gfx.MeshPhongMaterial())

    # If we have a material (including a texture)
    if hasattr(mesh.visual, "material") and use_material:
        material = mesh.visual.material
        # The material can be a PBRMaterial or a SimpleMaterial
        # pygfx' helper method only supports PBRMaterials
        if isinstance(material, tm.visual.material.PBRMaterial):
            vis.material = gfx.material_from_trimesh(material)
        elif isinstance(material, tm.visual.material.SimpleMaterial):
            vis.material = simple_material_from_trimesh(material)

    return vis


def simple_material_from_trimesh(material):
    """Convert a Trimesh SimpleMaterial object into a pygfx material.

    Parameters
    ----------
    material : trimesh.Material
        The material to convert.

    Returns
    -------
    converted : Material
        The converted material.

    """
    if not isinstance(material, tm.visual.material.SimpleMaterial):
        raise NotImplementedError()

    gfx_material = gfx.MeshPhongMaterial(
        color=material.ambient / 255,
        pick_write=True,  # we can't seem to change this on-the-fly
        )

    gfx_material.shininess = material.glossiness
    gfx_material.specular = gfx.Color(*(material.specular / 255))

    if hasattr(material, "image"):
        gfx_material.map = texture_from_pillow_image(material.image)

    gfx_material.side = "FRONT"
    return gfx_material


def texture_from_pillow_image(image, dim=2, **kwargs):
    """Pillow Image texture.

    Create a Texture from a PIL.Image.

    Parameters
    ----------
    image : Image
        The `PIL.Image
        <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image>`_
        to convert into a texture.
    dim : int
        The number of spatial dimensions of the image.
    kwargs : Any
        Additional kwargs are forwarded to :class:`pygfx.Texture`.

    Returns
    -------
    image_texture : Texture
        A texture object representing the given image.

    """
    # If this is a palette image, convert it to RGBA
    if getattr(image, "mode", None) == "P":
        image = image.convert("RGBA")

    m = memoryview(image.tobytes())

    im_channels = len(image.getbands())
    buffer_shape = image.size + (im_channels,)

    m = m.cast(m.format, shape=buffer_shape)
    return gfx.Texture(m, dim=dim, **kwargs)


# Monkey-patch the pygfx texture_from_pillow_image function
gfx.materials._compat.texture_from_pillow_image = texture_from_pillow_image


def scene2gfx(scene):
    """Convert trimesh Scene to pygfx visuals."""
    assert isinstance(
        scene, tm.scene.scene.Scene
    ), f"Expected trimesh scene, got {type(scene)}."

    # Get all the geometry names
    gfx_geometries = {}
    visuals = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]

        if geometry_name not in gfx_geometries:
            gfx_geometries[geometry_name] = trimesh2gfx(scene.geometry[geometry_name])

        vis = gfx.Mesh(
            gfx_geometries[geometry_name].geometry,
            gfx_geometries[geometry_name].material,
        )
        vis.local.matrix = transform

        visuals.append(vis)

    return visuals


def text2gfx(
    text,
    position=(0, 0, 0),
    color="w",
    font_size=1,
    anchor="top-right",
    screen_space=False,
    markdown=False,
):
    """Convert text to pygfx visuals.

    Parameters
    ----------
    text :          str
                    Text to plot.
    position :      tuple
                    Position of the text.
    color :         tuple | str
                    Color to use for plotting.
    font_size :     int, optional
                    Font size.
    anchor :        str, optional
                    Anchor point of the text. Combination of vertical and
                    horizontal alignment (e.g. "top-center"):
                     - vertical: "top", "bottom", "middle", "baseline"
                     - horizontal: "left", "right", "center"
    screen_space :  bool, optional
                    Whether to use screen space coordinates.
    markdown :      bool, optional
                    Whether the text should be interpreted as markdown.

    Returns
    -------
    text :          gfx.Text
                    Pygfx visual for text.

    """
    assert isinstance(text, str), "Expected string."
    assert isinstance(position, (list, tuple, np.ndarray)), "Expected list or tuple."
    assert len(position) == 3, "Expected (x, y, z) position."

    defaults = {
        "font_size": font_size,
        "anchor": anchor,
        "screen_space": screen_space,
    }
    if markdown:
        defaults["markdown"] = text
    else:
        defaults["text"] = text

    text = gfx.Text(
        **defaults,
        material=gfx.TextMaterial(color=color),
    )
    text.local.position = position
    return text


def visual_passthrough(x, *args, **kwargs):
    """Pass-through converter."""
    if any(args) or any(kwargs):
        logger.info(
            "Pygfx visuals are passed-through as is. Any additional arguments are ignored."
        )
    return x
