"""Candidate pairs of a periodic cell on the GPU (the screen of ``SkinNeighbourList.build_gpu``).

``candidate_pairs(wrapped, image_vectors, radius)`` returns ``(image, sender, receiver)`` for
every pair with ``|wrapped[receiver] - wrapped[sender] + image_vectors[image]|`` below
``radius + SCREEN_MARGIN`` (float32 screen; the caller decides in float64). On CUDA one Triton
kernel tests all sender x receiver tiles over all images in registers and appends the hits of
a tile with one atomic add; the order of the appended hits depends on scheduling, so the caller
must sort them. The CPU uses chunked PyTorch. ``half=True`` returns only pairs with
``sender <= receiver`` (the D3 half list) and skips the tiles below the diagonal.
"""

from __future__ import annotations

import functools

import torch

BLOCK = 32  # atoms per tile side (fixed for every crystal and GPU)
# CUDA's largest grid y dimension.
MAX_GRID_Y = 65535
# The most blocks one screen launch may have. Triton's launcher multiplies gridX * gridY * gridZ as 32-bit ints and
# launches only if the product is positive: above 2**31 - 1 blocks it overflowed and the screen silently did not
# run (no pairs, no error: NbCl5 at 1.73M and 2.0M atoms, 31 October 2026).
MAX_GRID_BLOCKS = 2**31 - 1
SCREEN_MARGIN = 0.01  # Å, float32 screen above the float64 decision
CPU_CHUNK = 1 << 22  # elements of the (images, rows, atoms) distance block per CPU chunk


@functools.cache
def _triton_kernel():
    """The screening kernel, built on first use (Triton is imported only on CUDA)."""
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(pos_ptr, image_ptr, out_ptr, counter_ptr, launched_ptr, natoms, nimages, screen, capacity, grid_y,
               rounds, BLOCK: tl.constexpr, HALF: tl.constexpr):
        if (tl.program_id(0) == 0) & (tl.program_id(1) == 0):  # the launch ran (``check_launched``)
            tl.store(launched_ptr, 1)
        senders = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        sx = tl.load(pos_ptr + senders * 3, mask=senders < natoms, other=0.0)
        sy = tl.load(pos_ptr + senders * 3 + 1, mask=senders < natoms, other=0.0)
        sz = tl.load(pos_ptr + senders * 3 + 2, mask=senders < natoms, other=0.0)
        pairs = natoms.to(tl.int64) * natoms  # int64: image * N^2 overflows int32 for large lists
        for k in range(rounds):  # receiver tiles program_id(1) + k * grid_y (``_grid``)
            tile = tl.program_id(1) + k * grid_y
            receivers = tile * BLOCK + tl.arange(0, BLOCK)
            valid = (senders < natoms)[:, None] & (receivers < natoms)[None, :]
            loops = nimages
            if HALF:  # tiles below the diagonal hold no sender <= receiver pair: no image loop
                valid = valid & (senders[:, None] <= receivers[None, :])
                loops = tl.where(tl.program_id(0) > tile, 0, nimages)
            rx = tl.load(pos_ptr + receivers * 3, mask=receivers < natoms, other=0.0)
            ry = tl.load(pos_ptr + receivers * 3 + 1, mask=receivers < natoms, other=0.0)
            rz = tl.load(pos_ptr + receivers * 3 + 2, mask=receivers < natoms, other=0.0)
            pair = senders.to(tl.int64)[:, None] * natoms + receivers.to(tl.int64)[None, :]
            for image in range(loops):
                ix = tl.load(image_ptr + image * 3)
                iy = tl.load(image_ptr + image * 3 + 1)
                iz = tl.load(image_ptr + image * 3 + 2)
                dx = rx[None, :] - sx[:, None] + ix
                dy = ry[None, :] - sy[:, None] + iy
                dz = rz[None, :] - sz[:, None] + iz
                hit = ((dx * dx + dy * dy + dz * dz) < screen) & valid
                flat = tl.reshape(hit.to(tl.int32), [BLOCK * BLOCK])
                count = tl.sum(flat, axis=0)
                if count > 0:
                    base = tl.atomic_add(counter_ptr, count)
                    slot = (base + tl.cumsum(flat, axis=0) - flat).to(tl.int64)
                    code = tl.reshape(image * pairs + pair, [BLOCK * BLOCK])
                    tl.store(out_ptr + slot, code, mask=(flat > 0) & (slot < capacity))

    return kernel


def _grid(natoms, max_grid_y=MAX_GRID_Y):
    """(grid, grid_y, rounds) of the screen: sender tiles on x; a program on y screens receiver tiles
    ``y + k * grid_y`` for ``k < rounds`` (tiles past the end hold no receiver: masked).

    ``grid_y`` is the receiver tile count unless that exceeds CUDA's y limit (``max_grid_y``) or would make the
    launch more than ``MAX_GRID_BLOCKS`` blocks, which Triton's launcher would skip without an error. Up to 46,340
    tiles (1.48M atoms) the grid is the 2-D one, ``(tiles, tiles, 1)`` with one round, as before.
    """
    tiles = max(1, -(-natoms // BLOCK))
    grid_y = max(1, min(tiles, int(max_grid_y), MAX_GRID_BLOCKS // tiles))
    if tiles * grid_y > MAX_GRID_BLOCKS or tiles > MAX_GRID_BLOCKS:
        raise ValueError(f"the pair screen of {natoms} atoms needs {tiles} x {grid_y} blocks, more than one launch "
                         f"takes ({MAX_GRID_BLOCKS})")
    return (tiles, grid_y, 1), grid_y, -(-tiles // grid_y)


def _candidate_pairs_cpu(wrapped, image_vectors, screen, half=False):
    natoms, nimages = wrapped.shape[0], image_vectors.shape[0]
    positions, images = wrapped.float(), image_vectors.float()
    rows = max(1, CPU_CHUNK // max(1, natoms * nimages))
    found = []
    for start in range(0, natoms, rows):
        squared = None
        for axis in range(3):
            d = (positions[None, None, :, axis] - positions[None, start:start + rows, None, axis]
                 + images[:, None, None, axis])
            squared = d * d if squared is None else squared.add_(d * d)
        hit = squared < screen
        if half:  # sender (row) <= receiver (column)
            rows_here = torch.arange(start, start + hit.shape[1], device=hit.device)
            hit &= rows_here[None, :, None] <= torch.arange(natoms, device=hit.device)[None, None, :]
        hits = torch.nonzero(hit)
        hits[:, 1] += start
        found.append(hits)
    hits = torch.cat(found)
    return hits[:, 0], hits[:, 1], hits[:, 2]


def check_launched(launched, natoms):
    """Raise unless the screen's first program set ``launched`` (a host synchronisation): a launch that did not
    run would otherwise look like a list without pairs (31 October 2026: Triton's int32 grid product)."""
    if int(launched.reshape(-1)[0]) != 1:
        raise RuntimeError(f"the GPU pair screen of {natoms} atoms did not run (its launch was skipped); "
                           "the neighbour list would have no pairs")


def screen_into(wrapped, image_vectors, radius, out, counter, launched, half=False, max_grid_y=MAX_GRID_Y):
    """Launch the CUDA screen into ``out`` (int64 codes) behind ``counter`` (int32, zeroed by the caller); no
    host synchronisation. Codes past ``out.numel()`` are dropped while ``counter`` still counts them, so
    ``counter > out.numel()`` means the buffer was too small (``SkinNeighbourList.build_gpu_async``).
    ``launched`` (int32, zeroed by the caller) is set to 1 by the launch; ``check_launched`` reads it."""

    natoms, nimages = wrapped.shape[0], image_vectors.shape[0]
    screen = (radius + SCREEN_MARGIN) ** 2
    grid, grid_y, rounds = _grid(natoms, max_grid_y)
    _triton_kernel()[grid](wrapped.float().contiguous(), image_vectors.float().contiguous(), out, counter, launched,
                           natoms, nimages, screen, out.numel(), grid_y, rounds, BLOCK=BLOCK, HALF=bool(half),
                           num_warps=4)


def decode_codes(code, natoms):
    """(image, sender, receiver) int64 tensors of screen codes ``image * N^2 + sender * N + receiver``."""
    pairs = natoms * natoms
    image = code // pairs
    rest = code - image * pairs
    return image, rest // natoms, rest % natoms


def screen_codes(wrapped, image_vectors, radius, state=None, half=False, max_grid_y=MAX_GRID_Y):
    """The screened pairs as int64 codes (``decode_codes``), unordered on CUDA: a view of the kept output buffer.

    Arguments as ``candidate_pairs``; ``max_grid_y`` (CUDA) caps the receiver programs, each then screening
    several receiver tiles (``_grid``; smaller in tests). The codes take 8 bytes per pair; decoding them in
    chunks bounds the caller's temporaries (``SkinNeighbourList.build_gpu``).
    """
    natoms, nimages = wrapped.shape[0], image_vectors.shape[0]
    screen = (radius + SCREEN_MARGIN) ** 2
    if not wrapped.is_cuda:
        image, sender, receiver = _candidate_pairs_cpu(wrapped, image_vectors, screen, half)
        return (image * natoms + sender) * natoms + receiver

    state = {} if state is None else state
    positions = wrapped.float().contiguous()
    images = image_vectors.float().contiguous()
    counter = torch.zeros(1, dtype=torch.int32, device=wrapped.device)
    grid, grid_y, rounds = _grid(natoms, max_grid_y)
    while True:
        out = state.get("buffer")
        if out is None:
            out = state["buffer"] = torch.empty(max(1 << 16, 128 * natoms), dtype=torch.int64,
                                                device=wrapped.device)
        launched = torch.zeros(1, dtype=torch.int32, device=wrapped.device)
        _triton_kernel()[grid](positions, images, out, counter, launched, natoms, nimages, screen, out.numel(),
                               grid_y, rounds, BLOCK=BLOCK, HALF=bool(half), num_warps=4)
        check_launched(launched, natoms)
        found = int(counter.item())  # host sync (rebuild steps only)
        if found <= out.numel():
            break
        out = state["buffer"] = None  # the old buffer is freed before its larger replacement
        state["buffer"] = torch.empty(int(found * 1.25) + 1, dtype=torch.int64, device=wrapped.device)
        counter.zero_()
    return out[:found]


def candidate_pairs(wrapped, image_vectors, radius, state=None, half=False, max_grid_y=MAX_GRID_Y):
    """(image, sender, receiver) int64 tensors of the screened pairs (unordered on CUDA).

    ``wrapped``: (N, 3) positions inside the cell; ``image_vectors``: (M, 3) Cartesian image
    translations; ``state``: optional dict that keeps the CUDA output buffer between calls;
    ``half``: only pairs with ``sender <= receiver``.
    """
    return decode_codes(screen_codes(wrapped, image_vectors, radius, state, half, max_grid_y), wrapped.shape[0])
