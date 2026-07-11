"""CPU-side packing of sparse (N, 3) voxel coordinates into a brick map.

The brick map consists of two textures:
 - a dense, low-resolution "coarse" index (r32uint) over the data's bounding
   box where each texel holds `atlas slot + 1` for occupied bricks and 0 for
   empty ones
 - a 3D "atlas" (r8unorm) in which the occupied bricks are packed side by side

This keeps GPU memory proportional to the number of *occupied* bricks instead
of the full bounding box, which is what makes 50M+ scattered voxels feasible.

Each atlas cell is (brick_size + 2)^3: the brick payload plus a 1-voxel apron
that duplicates the bordering voxels of neighboring bricks. The apron lets the
shader sample with (hardware) trilinear interpolation right up to the brick
borders without bleeding into unrelated bricks - without it, structures that
straddle brick boundaries would render visibly thinned at the seams.
"""

import warnings

from dataclasses import dataclass

import numpy as np


class AtlasCapacityError(ValueError):
    """Raised when the data contains more occupied bricks than fit in the atlas."""

    pass


@dataclass
class PackedBricks:
    """Result of `pack_sparse_voxels`.

    Attributes
    ----------
    coarse :        (gz, gy, gx) uint32 array
                    Coarse brick index in zyx order; 0 = empty, otherwise
                    atlas slot + 1.
    atlas :         (az * (B+2), ay * (B+2), ax * (B+2)) uint8 array
                    Brick atlas in zyx order; each cell is the brick payload
                    plus a 1-voxel apron of neighboring bricks' voxels.
    origin :        (3,) int array
                    xyz voxel coordinate of the volume's corner (subtracted
                    from the input coordinates).
    shape :         (3,) tuple
                    xyz extent of the volume in voxels.
    atlas_bricks :  (3,) tuple
                    xyz dimensions of the atlas in bricks.
    brick_size :    int
                    Edge length of a brick in voxels.
    n_bricks :      int
                    Number of occupied bricks.
    fill_ratio :    float
                    Points per allocated atlas voxel; low values indicate
                    scattered data for which the atlas is wasteful.

    """

    coarse: np.ndarray
    atlas: np.ndarray
    origin: np.ndarray
    shape: tuple
    atlas_bricks: tuple
    brick_size: int
    n_bricks: int
    fill_ratio: float


def pack_sparse_voxels(
    voxels,
    values=None,
    clim=None,
    brick_size=16,
    max_atlas_dim=2048,
    fill_ratio_warn=0.005,
):
    """Pack sparse (N, 3) voxel coordinates into a brick map.

    Parameters
    ----------
    voxels :        (N, 3) array
                    Voxel coordinates (xyz). Floats are floored to integers.
    values :        (N,) array, optional
                    Per-voxel scalar values. If given, values are quantized
                    into the 1-255 range (0 is reserved for empty space)
                    according to `clim`; voxels hit by multiple points keep
                    the maximum. Without values, occupied voxels are set
                    to 255.
    clim :          (min, max) tuple, optional
                    Range used to quantize `values`. Defaults to the data
                    min/max.
    brick_size :    int
                    Brick edge length in voxels; must be a power of two.
    max_atlas_dim : int
                    Maximum edge length of the atlas texture. The default
                    matches WebGPU's default `max_texture_dimension_3d`.
    fill_ratio_warn : float
                    Warn if the fraction of atlas voxels actually hit by a
                    point falls below this value.

    Returns
    -------
    PackedBricks

    """
    voxels = np.asarray(voxels)
    if voxels.ndim != 2 or voxels.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) array, got {voxels.shape}")
    if len(voxels) == 0:
        raise ValueError("Cannot pack empty array of voxels.")
    if brick_size < 2 or (brick_size & (brick_size - 1)):
        raise ValueError(f"`brick_size` must be a power of two, got {brick_size}")
    if values is not None:
        values = np.asarray(values).ravel()
        if len(values) != len(voxels):
            raise ValueError(
                f"Got {len(values)} values for {len(voxels)} voxels."
            )

    B = brick_size
    shift = int(np.log2(B))

    ijk = np.floor(voxels).astype(np.int64, copy=False)
    origin = ijk.min(axis=0)
    ijk = ijk - origin
    shape = tuple(int(s) for s in (ijk.max(axis=0) + 1))

    # Brick coordinate and coordinate within the brick, per point
    bx, by, bz = (ijk[:, 0] >> shift, ijk[:, 1] >> shift, ijk[:, 2] >> shift)
    lx, ly, lz = (ijk[:, 0] & (B - 1), ijk[:, 1] & (B - 1), ijk[:, 2] & (B - 1))

    # Coarse grid dimensions (xyz)
    gx, gy, gz = (int(-(-s // B)) for s in shape)

    # Linear index of each point's brick in the (zyx-ordered) coarse grid
    keys = (bz * gy + by) * gx + bx
    ubids, inv = np.unique(keys, return_inverse=True)
    m = len(ubids)

    # Coarse index texture: 0 = empty, otherwise atlas slot + 1
    coarse = np.zeros(gz * gy * gx, dtype=np.uint32)
    coarse[ubids] = np.arange(1, m + 1, dtype=np.uint32)
    coarse = coarse.reshape(gz, gy, gx)

    # Atlas cells are the brick plus a 1-voxel apron on each side
    A = B + 2

    # Choose atlas dimensions (in cells); fill x first, then y, then z
    cap = max_atlas_dim // A
    ax = min(m, cap)
    ay = min(-(-m // ax), cap)
    az = -(-m // (ax * ay))
    if az > cap:
        raise AtlasCapacityError(
            f"Data occupies {m:,} bricks of size {B}, exceeding the "
            f"{cap ** 3:,} bricks that fit in a {max_atlas_dim}^3 atlas. "
            "Increase `brick_size`, raise the texture size limit (see "
            "pygfx.renderers.wgpu.set_wgpu_limits) or use the dense fallback."
        )

    if values is None:
        quant = None
    else:
        if clim is None:
            clim = (values.min(), values.max())
        cmin, cmax = float(clim[0]), float(clim[1])
        scale = 254.0 / (cmax - cmin) if cmax > cmin else 0.0
        # Quantize into 1-255; 0 is reserved for empty space
        quant = np.clip((values - cmin) * scale, 0, 254).astype(np.uint8) + 1

    atlas = np.zeros((az * A, ay * A, ax * A), dtype=np.uint8)

    def scatter(slot, cx, cy, cz, q):
        """Write values at (cx, cy, cz) within the A^3 cell of `slot`."""
        vx = (slot % ax) * A + cx
        vy = ((slot // ax) % ay) * A + cy
        vz = (slot // (ax * ay)) * A + cz
        if q is None:
            atlas[vz, vy, vx] = 255
        else:
            # Voxels hit multiple times keep the maximum
            np.maximum.at(atlas, (vz, vy, vx), q)

    # Each point goes into its own brick (offset +1 for the apron) ...
    slot = inv.astype(np.int64)  # 0-based atlas slot per point
    scatter(slot, lx + 1, ly + 1, lz + 1, quant)

    # ... and, for points on a brick border, also into the apron layer of the
    # (occupied) neighboring bricks. Per axis a point touches the -1 neighbor
    # iff it sits on the first voxel layer of its brick, and the +1 neighbor
    # iff it sits on the last.
    coarse_flat = coarse.ravel()
    border = {
        -1: (lx == 0, ly == 0, lz == 0),
        1: (lx == B - 1, ly == B - 1, lz == B - 1),
    }
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                mask = None
                for d, axis in ((dx, 0), (dy, 1), (dz, 2)):
                    if d == 0:
                        continue
                    cond = border[d][axis]
                    mask = cond if mask is None else (mask & cond)
                idx = np.flatnonzero(mask)
                if len(idx) == 0:
                    continue
                tbx, tby, tbz = bx[idx] + dx, by[idx] + dy, bz[idx] + dz
                inb = (
                    (tbx >= 0) & (tbx < gx)
                    & (tby >= 0) & (tby < gy)
                    & (tbz >= 0) & (tbz < gz)
                )
                idx, tbx, tby, tbz = idx[inb], tbx[inb], tby[inb], tbz[inb]
                if len(idx) == 0:
                    continue
                # Only occupied neighbors have an atlas cell to write into
                tslot = coarse_flat[(tbz * gy + tby) * gx + tbx]
                occ = tslot != 0
                idx = idx[occ]
                if len(idx) == 0:
                    continue
                scatter(
                    tslot[occ].astype(np.int64) - 1,
                    lx[idx] - dx * B + 1,
                    ly[idx] - dy * B + 1,
                    lz[idx] - dz * B + 1,
                    None if quant is None else quant[idx],
                )

    fill_ratio = len(voxels) / (m * A**3)
    # Only warn when the waste is substantial (> ~32 MB of atlas)
    if fill_ratio < fill_ratio_warn and m * A**3 > 32e6:
        warnings.warn(
            f"Sparse volume has a low fill ratio ({fill_ratio:.2%}): the "
            f"{len(voxels):,} points occupy {m:,} bricks "
            f"({m * A**3 * 1e-6:.0f} MB of atlas). For very scattered data "
            "consider a larger `brick_size` or rendering as points instead."
        )

    return PackedBricks(
        coarse=coarse,
        atlas=atlas,
        origin=origin,
        shape=shape,
        atlas_bricks=(ax, ay, az),
        brick_size=B,
        n_bricks=m,
        fill_ratio=fill_ratio,
    )
