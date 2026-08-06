"""CPU-side packing of run-length-encoded (N, 4) voxels into a bitmask brick map.

Where `.packing` stores one *byte* per voxel in a dense atlas (so that it can
carry per-voxel values and be sampled with hardware trilinear filtering), this
packs binary occupancy into one *bit* per voxel. Runs never have values, so
there is nothing to interpolate and the byte is pure waste.

The representation is:

  super       dense r32uint grid over the bounding box at `brick_size * 8`
              resolution; texel = super-brick slot + 1, 0 = empty
  bricktab    u32, per occupied super-brick: 8^3 entries of brick slot + 1
  bits        u32, per occupied brick: brick_size^3 / 32 words, 1 bit per voxel

The two-level index matters as much as the bitmask does. A dense index over the
bounding box costs `(bbox / brick_size)^3 * 4` bytes regardless of how much of
it is occupied, and neurons have very empty bounding boxes: for a 31M-voxel
neuron spanning 3366x10137x5332 that is 179 MB at brick_size=16 - more than ten
times the payload. Making the index sparse brings it down to 1.8 MB.

No apron is needed: each of the eight taps of a trilinear sample resolves its
own brick through the index, so samples cross brick borders correctly. That
saves the (B+2)^3 / B^3 overhead the dense atlas pays on top of the 8x.

Measured on a 31.3M-voxel / 1.16M-run DVID neuron, in bytes per occupied voxel:

    (N, 3) int32 coordinates            12.000
    r8 atlas + dense index (`.packing`) 10.475
    bitmask + two-level index (here)     0.451
    RLE + two-level index, brick 64      0.215

So this is ~23x smaller than the dense atlas. A brick-clipped RLE payload would
be a further ~2.1x smaller still, but each sample then costs a binary search
over the brick's run list instead of a single bit test, which measured 1.7x
slower at equal memory and 2.9x slower at its own memory optimum. If GPU memory
ever becomes the binding constraint, that is the knob to reach for; until then
the bit test is the better trade.

brick_size 16 is the measured optimum: 8 makes the index dominate, 32+ wastes
bits on empty space inside partly-filled bricks.
"""

import warnings

from dataclasses import dataclass

import numpy as np

#: Bricks per super-brick edge. The brick table is dense within a super-brick,
#: so this trades index granularity against the size of that table.
SUPER = 8


class BitmaskCapacityError(ValueError):
    """Raised when the packed data exceeds what can be bound on the GPU."""

    pass


@dataclass
class PackedBitmask:
    """Result of `pack_voxel_runs`.

    Attributes
    ----------
    super_index :   (sz, sy, sx) uint32 array
                    Dense super-brick index in zyx order; 0 = empty, otherwise
                    super-brick slot + 1.
    bricktab :      (n_supers * 8^3,) uint32 array
                    Per occupied super-brick, a dense 8^3 table of brick slots
                    (+1; 0 = empty).
    bits :          (n_bricks * brick_size^3 / 32,) uint32 array
                    One bit per voxel, x fastest.
    origin :        (3,) int array
                    xyz voxel coordinate of the volume's corner.
    shape :         (3,) tuple
                    xyz extent of the volume in voxels.
    grid_bricks :   (3,) tuple
                    xyz size of the brick grid.
    grid_supers :   (3,) tuple
                    xyz size of the super-brick grid.
    brick_size :    int
                    Edge length of a brick in voxels.
    n_bricks :      int
                    Number of occupied bricks.
    n_supers :      int
                    Number of occupied super-bricks.
    n_voxels :      int
                    Number of occupied voxels.
    fill_ratio :    float
                    Occupied voxels per allocated brick voxel.

    """

    super_index: np.ndarray
    bricktab: np.ndarray
    bits: np.ndarray
    origin: np.ndarray
    shape: tuple
    grid_bricks: tuple
    grid_supers: tuple
    brick_size: int
    n_bricks: int
    n_supers: int
    n_voxels: int
    fill_ratio: float

    @property
    def nbytes(self):
        """Total GPU footprint in bytes."""
        return self.super_index.nbytes + self.bricktab.nbytes + self.bits.nbytes


def runs_from_voxels(voxels):
    """Convert (N, 3) voxel coordinates to (N, 4) runs along x.

    Parameters
    ----------
    voxels :    (N, 3) array
                Voxel coordinates (xyz). Floats are floored to integers.

    Returns
    -------
    (M, 4) int64 array
                Runs as (x, y, z, length), sorted by (z, y, x).

    """
    voxels = np.asarray(voxels)
    if voxels.ndim != 2 or voxels.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) array, got {voxels.shape}")
    if not len(voxels):
        raise ValueError("Cannot pack an empty array of voxels.")

    ijk = np.floor(voxels).astype(np.int64, copy=False)
    origin = ijk.min(axis=0)
    ijk = ijk - origin
    extent = ijk.max(axis=0) + 1

    # Pack into a single key with x fastest, then unique: this both drops
    # duplicates and sorts by (z, y, x), which is the order runs need.
    # (`np.unique(..., axis=0)` would sort x-major instead and find no runs.)
    key = np.unique((ijk[:, 2] * extent[1] + ijk[:, 1]) * extent[0] + ijk[:, 0])
    x, rest = key % extent[0], key // extent[0]
    y, z = rest % extent[1], rest // extent[1]

    # A new run starts wherever x is not one past the previous, or y/z change
    brk = np.r_[True, (np.diff(x) != 1) | (np.diff(y) != 0) | (np.diff(z) != 0)]
    starts = np.flatnonzero(brk)
    lengths = np.diff(np.r_[starts, len(key)])
    return np.column_stack([x[starts] + origin[0], y[starts] + origin[1],
                            z[starts] + origin[2], lengths])


def pack_voxel_runs(runs, brick_size=16, max_buffer_bytes=128 * 1024**2):
    """Pack (N, 4) run-length-encoded voxels into a bitmask brick map.

    Parameters
    ----------
    runs :          (N, 4) array
                    Runs as (x, y, z, x_run_length), i.e. the layout DVID's
                    `sparsevol` endpoint returns (see `dvid.get_sparsevol`
                    with ``voxels=False``). Runs extend along x and the length
                    is an inclusive voxel count.
    brick_size :    int
                    Brick edge length in voxels; must be a power of two in
                    8-64. The default of 16 is the measured optimum.
    max_buffer_bytes : int
                    Storage-buffer size to check the payload against; matches
                    WebGPU's default `maxStorageBufferBindingSize`.

    Returns
    -------
    PackedBitmask

    """
    runs = np.asarray(runs)
    if runs.ndim != 2 or runs.shape[1] != 4:
        raise ValueError(f"Expected (N, 4) array of runs, got {runs.shape}")
    if not len(runs):
        raise ValueError("Cannot pack an empty array of runs.")
    if brick_size not in (8, 16, 32, 64):
        raise ValueError(f"`brick_size` must be 8, 16, 32 or 64, got {brick_size}")

    B = int(brick_size)
    sh = int(np.log2(B))

    # DVID hands these over as uint32; unsigned arithmetic would wrap when we
    # subtract the origin below, so widen to a signed type first
    runs = runs.astype(np.int64, copy=True)
    runs = runs[runs[:, 3] > 0]
    if not len(runs):
        raise ValueError("All runs have zero length.")

    origin = runs[:, :3].min(axis=0)
    runs[:, :3] -= origin
    x, y, z, length = runs[:, 0], runs[:, 1], runs[:, 2], runs[:, 3]
    shape = np.array(
        [(x + length - 1).max() + 1, y.max() + 1, z.max() + 1], dtype=np.int64
    )
    n_voxels = int(length.sum())

    grid_bricks = (shape + B - 1) // B
    grid_supers = (grid_bricks + SUPER - 1) // SUPER

    # ---- split runs at brick borders ------------------------------------
    # A run starting at x with length L touches this many bricks along x
    n_span = ((x & (B - 1)) + length + B - 1) // B
    rep = np.repeat(np.arange(len(runs)), n_span)
    step = np.arange(int(n_span.sum())) - np.repeat(np.cumsum(n_span) - n_span, n_span)

    bx = (x[rep] >> sh) + step
    by, bz = y[rep] >> sh, z[rep] >> sh
    x0 = np.maximum(x[rep], bx * B)
    x1 = np.minimum(x[rep] + length[rep], (bx + 1) * B)
    lx, ly, lz = x0 - bx * B, y[rep] & (B - 1), z[rep] & (B - 1)
    ln = x1 - x0

    # ---- two-level index -------------------------------------------------
    sup_key = ((bz // SUPER) * grid_supers[1] + (by // SUPER)) * grid_supers[0] + (
        bx // SUPER
    )
    usup = np.unique(sup_key)
    n_supers = len(usup)

    super_index = np.zeros(int(np.prod(grid_supers)), np.uint32)
    super_index[usup] = np.arange(1, n_supers + 1, dtype=np.uint32)
    super_index = super_index.reshape(
        int(grid_supers[2]), int(grid_supers[1]), int(grid_supers[0])
    )

    brick_key = (bz * grid_bricks[1] + by) * grid_bricks[0] + bx
    ubrick, slot = np.unique(brick_key, return_inverse=True)
    n_bricks = len(ubrick)
    slot = slot.astype(np.int64)

    words = B**3 // 32
    n_bytes = n_bricks * words * 4
    if n_bytes > max_buffer_bytes:
        raise BitmaskCapacityError(
            f"Data occupies {n_bricks:,} bricks of size {B} = "
            f"{n_bytes / 1e6:.0f} MB, exceeding the "
            f"{max_buffer_bytes / 1e6:.0f} MB storage-buffer limit. Increase "
            "`brick_size`, raise the limit (see "
            "pygfx.renderers.wgpu.set_wgpu_limits) or render at a coarser "
            "scale."
        )
    if max(super_index.shape) > 2048:
        raise BitmaskCapacityError(
            f"Super-brick grid {super_index.shape[::-1]} exceeds the 2048 "
            "texture limit; increase `brick_size`."
        )

    # Brick slot table, indexed by (super slot, brick within super)
    ub_z, ub_rest = np.divmod(ubrick, grid_bricks[0] * grid_bricks[1])
    ub_y, ub_x = np.divmod(ub_rest, grid_bricks[0])
    ub_sup = np.searchsorted(
        usup,
        ((ub_z // SUPER) * grid_supers[1] + (ub_y // SUPER)) * grid_supers[0]
        + (ub_x // SUPER),
    )
    within = ((ub_z % SUPER) * SUPER + (ub_y % SUPER)) * SUPER + (ub_x % SUPER)
    bricktab = np.zeros(n_supers * SUPER**3, np.uint32)
    bricktab[ub_sup * SUPER**3 + within] = np.arange(1, n_bricks + 1, dtype=np.uint32)

    # ---- bitmask payload -------------------------------------------------
    bits = np.zeros(n_bricks * words, np.uint32)
    bit0 = (lz * B + ly) * B + lx                    # first bit of the run
    bit1 = bit0 + ln                                 # one past the last
    # A row of B bits straddles a u32 boundary once B > 32, so split per word
    w0, w1 = bit0 >> 5, (bit1 - 1) >> 5
    n_words = w1 - w0 + 1
    rep2 = np.repeat(np.arange(len(bit0)), n_words)
    step2 = np.arange(int(n_words.sum())) - np.repeat(
        np.cumsum(n_words) - n_words, n_words
    )
    word = w0[rep2] + step2
    lo = np.maximum(bit0[rep2], word * 32) - word * 32
    hi = np.minimum(bit1[rep2], (word + 1) * 32) - word * 32
    mask = ((np.int64(1) << (hi - lo)) - 1) << lo
    np.bitwise_or.at(bits, slot[rep2] * words + word, mask.astype(np.uint32))

    fill_ratio = n_voxels / (n_bricks * B**3)
    if fill_ratio < 0.02 and n_bytes > 32e6:
        warnings.warn(
            f"Sparse volume has a low fill ratio ({fill_ratio:.2%}): the "
            f"{n_voxels:,} voxels occupy {n_bricks:,} bricks "
            f"({n_bytes / 1e6:.0f} MB). Consider a smaller `brick_size`."
        )

    return PackedBitmask(
        super_index=super_index,
        bricktab=bricktab,
        bits=bits,
        origin=origin,
        shape=tuple(int(s) for s in shape),
        grid_bricks=tuple(int(g) for g in grid_bricks),
        grid_supers=tuple(int(g) for g in grid_supers),
        brick_size=B,
        n_bricks=n_bricks,
        n_supers=n_supers,
        n_voxels=n_voxels,
        fill_ratio=fill_ratio,
    )
