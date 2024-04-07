import uuid

import pygfx as gfx
import numpy as np
import trimesh as tm

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


def parse_mesh_color(mesh, color, alpha=None):
    """Parse color for mesh plotting."""
    mat_color_kwargs = dict()
    obj_color_kwargs = dict()
    if isinstance(color, np.ndarray) and color.ndim == 2:
        if alpha is not None:
            color[:, -1] = alpha

        if len(color) == len(mesh.vertices):
            obj_color_kwargs = dict(colors=color)
            mat_color_kwargs = dict(color_mode="vertex")
        elif len(color) == len(mesh.faces):
            obj_color_kwargs = dict(colors=color)
            mat_color_kwargs = dict(color_mode="face")
        else:
            mat_color_kwargs["color"] = color
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


def volume2gfx(vol, dims, color, offset=(0, 0, 0), cmin=None, cmax=None, **kwargs):
    """Convert volume (i.e. 3d arrays) to pygfx visual.

    Parameters
    ----------
    vol :           np.ndarray
                    3D array representing the volume.
    dims :          tuple
                    Dimensions of the volume along the (x, y, z) axes.
    color :         str | tuple
                    CURRENTLY NOT USED.

    """
    # TODOs:
    # - add support for custom color maps (see cmap's to_pygfx method)
    # - add support for other Volume materials (e.g. gfx.VolumeMipMaterial)

    assert isinstance(vol, np.ndarray), "Expected 3D numpy array."
    assert vol.ndim == 3, "Expected 3D numpy array."
    assert isinstance(dims, (tuple, list, np.ndarray, int, float))
    if isinstance(dims, (int, float)):
        dims = [dims] * 3
    assert len(dims) == 3, "Expected dimensions as tuple of length 3."

    # Similar to vispy, pygfx seems to expect zyx coordinate space
    grid = vol.T

    # Avoid boolean matrices here
    if grid.dtype == bool:
        grid = grid.astype(int)
    elif grid.dtype.str.contains('>u') or grid.dtype.str.contains('<u'):
        grid = grid.astype(np.uint32)
    elif grid.dtype.str.contains('>f') or grid.dtype.str.contains('<f'):
        grid = grid.astype(np.float32)

    # Find the potential min/max value of the volume
    if cmax is None:
        cmax = np.iinfo(grid.dtype).max
    elif cmax == "auto":
        cmax = grid.max()

    if cmin is None:
        cmin = 0
    elif cmin == "auto":
        cmin = grid.min()

    # Initialize texture
    tex = gfx.Texture(grid, dim=3)

    # Initialize the volume
    vis = gfx.Volume(
        gfx.Geometry(grid=tex),
        gfx.VolumeRayMaterial(clim=(0, cmax), map=gfx.cm.cividis),
    )

    # Set scales and offset
    (
        vis.local.scale_x,
        vis.local.scale_y,
        vis.local.scale_z,
    ) = dims
    (vis.local.x, vis.local.y, vis.local.z) = offset

    # Add custom attributes
    vis._object_type = "volume"
    vis._object_id = uuid.uuid4()

    return vis


def points2gfx(points, color, size=2):
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

    Returns
    -------
    list
                    Contains pygfx visuals for points.

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
    if utils.is_iterable(size):
        if len(size) != len(points):
            raise ValueError(
                "Expected `size` to be a single value or "
                "an array of the same length as `points`."
            )
        geometry_kwargs['sizes'] = np.asarray(size).astype(np.float32, copy=False)
        material_kwargs['vertex_sizes'] = True
    else:
        material_kwargs['size'] = size

    vis = gfx.Points(
        gfx.Geometry(positions=points, **geometry_kwargs),
        gfx.PointsMaterial(color=color, **material_kwargs),
    )

    # Add custom attributes
    vis._object_type = "points"
    vis._object_id = uuid.uuid4()

    return vis


def lines2gfx(lines, color, linewidth=1):
    """Convert lines into pygfx visuals."""
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
            lines = np.insert(np.vstack(lines), np.cumsum([len(l) for l in lines[:-1]]), np.nan, axis=0)
    else:
        raise TypeError("Expected numpy array or list of numpy arrays.")

    geometry_kwargs = {}
    material_kwargs = {}

    # Parse color(s)
    if isinstance(color, np.ndarray) and color.ndim == 2:
        # If colors are provided for each node we have to make sure
        # that we also include `None` for the breaks in the segments
        n_points = (~np.isnan(lines[:, 0])).sum()
        if n_points != len(lines):
            # See if we can rescue this
            if len(color) == n_points:
                breaks = np.where(np.isnan(lines[:, 0]))[0]
                for b in breaks:
                    color = np.insert(color, b, np.nan, axis=0)
            else:
                raise ValueError(f"Got {len(color)} colors for {n_points} points.")
        color = color.astype(np.float32, copy=False)
        geometry_kwargs['colors'] = color
        material_kwargs['color_mode'] = 'vertex'
    else:
        if isinstance(color, np.ndarray):
            color = color.astype(np.float32, copy=False)
        material_kwargs['color'] = color

    vis = gfx.Line(
        gfx.Geometry(positions=lines.astype(np.float32, copy=False),
                     **geometry_kwargs),
        gfx.LineMaterial(thickness=linewidth,
                         **material_kwargs),
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
        normals=np.ascontiguousarray(mesh.vertex_normals, dtype="f4"),
    )
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
    vis = gfx.Mesh(gfx.Geometry(**kwargs),
                   gfx.MeshBasicMaterial())

    # If we have a material (including a texture)
    if hasattr(mesh.visual, 'material') and use_material:
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

    gfx_material = gfx.MeshPhongMaterial(color=material.ambient / 255)

    gfx_material.shininess = material.glossiness
    gfx_material.specular = gfx.Color(*(material.specular / 255))

    if hasattr(material, 'image'):
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
    if getattr(image, 'mode', None) == 'P':
        image = image.convert('RGBA')

    m = memoryview(image.tobytes())

    im_channels = len(image.getbands())
    buffer_shape = image.size + (im_channels,)

    m = m.cast(m.format, shape=buffer_shape)
    return gfx.Texture(m, dim=dim, **kwargs)

# Monkey-patch the pygfx texture_from_pillow_image function
gfx.materials._compat.texture_from_pillow_image = texture_from_pillow_image


def scene2gfx(scene):
    """Convert trimesh Scene to pygfx visuals."""
    assert isinstance(scene, tm.scene.scene.Scene), f"Expected trimesh scene, got {type(scene)}."

    # Get all the geometry names
    gfx_geometries = {}
    visuals = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]

        if geometry_name not in gfx_geometries:
            gfx_geometries[geometry_name] = trimesh2gfx(scene.geometry[geometry_name])

        vis = gfx.Mesh(gfx_geometries[geometry_name].geometry,
                       gfx_geometries[geometry_name].material)
        vis.local.matrix = transform

        visuals.append(vis)

    return visuals
