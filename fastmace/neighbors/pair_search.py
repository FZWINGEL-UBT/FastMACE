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

import numpy as np
import torch

from fastmace.kernels.gpu_models import GPUModels

BLOCK = 32  # atoms per tile side (fixed for every crystal and GPU)
# CUDA's largest grid y dimension.
MAX_GRID_Y = 65535
# The most blocks one screen launch may have. Triton's launcher multiplies gridX * gridY * gridZ as 32-bit ints and
# launches only if the product is positive: above 2**31 - 1 blocks it overflowed and the screen silently did not
# run (no pairs, no error: NbCl5 at 1.73M and 2.0M atoms, 31 October 2026).
MAX_GRID_BLOCKS = 2**31 - 1
SCREEN_MARGIN = 0.01  # Å, float32 screen above the float64 decision
CPU_CHUNK = 1 << 22  # elements of the (images, rows, atoms) distance block per CPU chunk
# Neighbour-list follow-up (6 November 2026): BINNED_SCREEN screens pairs through a cell list. Atoms are binned in
# fractional slabs at least the screen radius thick, and each receiver bin is tested against its 27 neighbour bins,
# the image read off the wrap. That is O(N) instead of the tile screen's O(N^2) (SrTiO3 16x16x16: 17.8 ms of a 22.4 ms
# build on an RTX 4000 Ada). It finds every pair within the screen radius, so the build's float64 decision and
# unique sort key give the same list: bitwise. Needs a cell given (``cell=``), at least 3 bins along every axis and
# the full list (not ``half``); otherwise the tile screen runs. None: on for BINNED_SCREEN_GPUS; True / False: an
# A/B side's choice.
BINNED_SCREEN = None
BINNED_SCREEN_GPUS = GPUModels(())
BIN_BLOCK = 32  # atoms per tile side of the binned screen


@functools.cache
def _triton_kernel():
    """The screening kernel, built on first use (Triton is imported only on CUDA)."""
    import triton
    import triton.language as tl

    @triton.jit
    def screen_tile(pos_ptr, image_ptr, out_ptr, counter_ptr, natoms, nimages, screen, capacity, tile,
                    BLOCK: tl.constexpr, HALF: tl.constexpr):
        """Screen sender tile ``program_id(0)`` against receiver tile ``tile`` over every image."""
        senders = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        receivers = tile * BLOCK + tl.arange(0, BLOCK)
        valid = (senders < natoms)[:, None] & (receivers < natoms)[None, :]
        loops = nimages
        if HALF:  # tiles below the diagonal hold no sender <= receiver pair: no image loop
            valid = valid & (senders[:, None] <= receivers[None, :])
            loops = tl.where(tl.program_id(0) > tile, 0, nimages)
        sx = tl.load(pos_ptr + senders * 3, mask=senders < natoms, other=0.0)
        sy = tl.load(pos_ptr + senders * 3 + 1, mask=senders < natoms, other=0.0)
        sz = tl.load(pos_ptr + senders * 3 + 2, mask=senders < natoms, other=0.0)
        rx = tl.load(pos_ptr + receivers * 3, mask=receivers < natoms, other=0.0)
        ry = tl.load(pos_ptr + receivers * 3 + 1, mask=receivers < natoms, other=0.0)
        rz = tl.load(pos_ptr + receivers * 3 + 2, mask=receivers < natoms, other=0.0)
        pair = senders.to(tl.int64)[:, None] * natoms + receivers.to(tl.int64)[None, :]
        pairs = natoms.to(tl.int64) * natoms  # int64: image * N^2 overflows int32 for large lists
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

    @triton.jit
    def kernel(pos_ptr, image_ptr, out_ptr, counter_ptr, natoms, nimages, screen, capacity, grid_y, rounds,
               BLOCK: tl.constexpr, HALF: tl.constexpr, SINGLE: tl.constexpr):
        if SINGLE:  # one receiver tile per program (up to 46,340 tiles): the 2-D grid's code
            screen_tile(pos_ptr, image_ptr, out_ptr, counter_ptr, natoms, nimages, screen, capacity,
                        tl.program_id(1), BLOCK, HALF)
        else:  # receiver tiles program_id(1) + k * grid_y (``_grid``)
            for k in range(rounds):
                screen_tile(pos_ptr, image_ptr, out_ptr, counter_ptr, natoms, nimages, screen, capacity,
                            tl.program_id(1) + k * grid_y, BLOCK, HALF)

    return kernel


@functools.cache
def _binned_kernel():
    """The cell-list screen (``BINNED_SCREEN``), built on first use."""
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(pos_ptr, image_ptr, order_ptr, start_ptr, out_ptr, counter_ptr, natoms, nb0, nb1, nb2, k0, k1, k2,
               screen, capacity, BLOCK: tl.constexpr):
        """Receiver bin ``program_id(0)`` against its neighbour bin ``program_id(1)`` (offsets -1..1 per axis)."""
        rbin = tl.program_id(0)
        stencil = tl.program_id(1)
        b0 = rbin // (nb1 * nb2)
        b1 = (rbin // nb2) % nb1
        b2 = rbin % nb2
        t0 = b0 + stencil // 9 - 1
        t1 = b1 + (stencil // 3) % 3 - 1
        t2 = b2 + stencil % 3 - 1
        w0 = tl.where(t0 < 0, -1, tl.where(t0 >= nb0, 1, 0))  # the sender bin's wrap along each axis
        w1 = tl.where(t1 < 0, -1, tl.where(t1 >= nb1, 1, 0))
        w2 = tl.where(t2 < 0, -1, tl.where(t2 >= nb2, 1, 0))
        sbin = ((t0 - w0 * nb0) * nb1 + (t1 - w1 * nb1)) * nb2 + (t2 - w2 * nb2)
        # The sender's copy in the neighbour bin is wrapped[sender] + w . cell: the image is -w.
        image = ((k0 - w0) * (2 * k1 + 1) + (k1 - w1)) * (2 * k2 + 1) + (k2 - w2)
        ix = tl.load(image_ptr + image * 3)
        iy = tl.load(image_ptr + image * 3 + 1)
        iz = tl.load(image_ptr + image * 3 + 2)
        r_lo = tl.load(start_ptr + rbin)
        r_hi = tl.load(start_ptr + rbin + 1)
        s_lo = tl.load(start_ptr + sbin)
        s_hi = tl.load(start_ptr + sbin + 1)
        pairs = natoms.to(tl.int64) * natoms
        lane = tl.arange(0, BLOCK)
        for r0 in range(r_lo, r_hi, BLOCK):
            rslot = r0 + lane
            rmask = rslot < r_hi
            receivers = tl.load(order_ptr + rslot, mask=rmask, other=0)
            rx = tl.load(pos_ptr + receivers * 3, mask=rmask, other=0.0)
            ry = tl.load(pos_ptr + receivers * 3 + 1, mask=rmask, other=0.0)
            rz = tl.load(pos_ptr + receivers * 3 + 2, mask=rmask, other=0.0)
            for s0 in range(s_lo, s_hi, BLOCK):
                sslot = s0 + lane
                smask = sslot < s_hi
                senders = tl.load(order_ptr + sslot, mask=smask, other=0)
                sx = tl.load(pos_ptr + senders * 3, mask=smask, other=0.0)
                sy = tl.load(pos_ptr + senders * 3 + 1, mask=smask, other=0.0)
                sz = tl.load(pos_ptr + senders * 3 + 2, mask=smask, other=0.0)
                dx = rx[None, :] - sx[:, None] + ix  # the tile screen's float32 expression
                dy = ry[None, :] - sy[:, None] + iy
                dz = rz[None, :] - sz[:, None] + iz
                hit = ((dx * dx + dy * dy + dz * dz) < screen) & smask[:, None] & rmask[None, :]
                flat = tl.reshape(hit.to(tl.int32), [BLOCK * BLOCK])
                count = tl.sum(flat, axis=0)
                if count > 0:
                    base = tl.atomic_add(counter_ptr, count)
                    slot = (base + tl.cumsum(flat, axis=0) - flat).to(tl.int64)
                    pair = senders.to(tl.int64)[:, None] * natoms + receivers.to(tl.int64)[None, :]
                    code = tl.reshape(image.to(tl.int64) * pairs + pair, [BLOCK * BLOCK])
                    tl.store(out_ptr + slot, code, mask=(flat > 0) & (slot < capacity))

    return kernel


@torch.compiler.assume_constant_result
def _binned_listed(index: int) -> bool:
    """Whether CUDA device ``index`` is one of ``BINNED_SCREEN_GPUS`` (a Dynamo constant)."""
    return torch.cuda.get_device_name(index) in BINNED_SCREEN_GPUS


def binned_screen(device) -> bool:
    """``BINNED_SCREEN`` on ``device``: the constant when set, else on for ``BINNED_SCREEN_GPUS``."""
    if BINNED_SCREEN is not None:
        return bool(BINNED_SCREEN)
    device = torch.device(device)
    if device.type != "cuda":
        return False
    return _binned_listed(device.index if device.index is not None else torch.cuda.current_device())


def _bin_plan(wrapped, image_vectors, radius, cell, half):
    """(grid, args) of the binned screen for these positions, or None where the tile screen runs.

    ``cell``: the host (3, 3) cell whose images ``image_vectors`` are, enumerated as ``SkinNeighbourList`` does
    (a meshgrid of -k..k per axis, ``ij`` order). Bins are fractional slabs of thickness width / nb >= the screen
    radius, so two atoms within it differ by at most one bin along every axis.
    """
    if cell is None or half or not wrapped.is_cuda or not binned_screen(wrapped.device):
        return None
    cell = np.asarray(cell, dtype=float)
    reach = radius + SCREEN_MARGIN
    inverse = np.linalg.inv(cell)
    widths = 1.0 / np.linalg.norm(inverse, axis=0)
    nb = [int(np.floor(w / reach)) for w in widths]
    ks = [int(np.floor(radius / w + 1.0)) for w in widths]
    if min(nb) < 3 or image_vectors.shape[0] != int(np.prod([2 * k + 1 for k in ks])):
        return None
    device = wrapped.device
    pos = wrapped.to(torch.float64)
    bins = None
    for a in range(3):  # host scalars only: no copy to the device, so no host synchronisation
        frac = (pos[:, 0] * float(inverse[0, a]) + pos[:, 1] * float(inverse[1, a])
                + pos[:, 2] * float(inverse[2, a]))
        idx = torch.clamp(torch.floor(frac * nb[a]), 0, nb[a] - 1).to(torch.int64)
        bins = idx if bins is None else bins * nb[a] + idx
    nbins = nb[0] * nb[1] * nb[2]
    order = torch.argsort(bins, stable=True).to(torch.int32)
    start = torch.zeros(nbins + 1, dtype=torch.int32, device=device)
    counts = torch.zeros(nbins, dtype=torch.int32, device=device).index_add_(0, bins, torch.ones_like(order))
    start[1:] = torch.cumsum(counts, 0).to(torch.int32)  # not bincount: it reads its input's max on the host
    return (nbins, 27), (order, start, nb, ks)


def _launch_binned(plan, positions, images, out, counter, natoms, screen):
    grid, (order, start, nb, ks) = plan
    _binned_kernel()[grid](positions, images, order, start, out, counter, natoms, nb[0], nb[1], nb[2], ks[0], ks[1],
                           ks[2], screen, out.numel(), BLOCK=BIN_BLOCK, num_warps=4)


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


def check_found(found, wrapped, image_vectors, radius):
    """Raise if the screen found no pair although atom 0 has a neighbour within ``radius`` (a brute-force test
    of atom 0 against every atom and image, run only for an empty result). A launch that did not run looks like
    a list without pairs (31 October 2026: Triton's launcher skipped grids above 2**31 blocks without an error);
    ``_grid`` keeps launches below that, and this catches any other way the screen could come back empty."""
    if found or wrapped.shape[0] == 0:
        return
    screen = (radius + SCREEN_MARGIN) ** 2
    d = wrapped[None, :, :].float() - wrapped[None, 0:1, :].float() + image_vectors[:, None, :].float()
    squared = (d * d).sum(dim=-1)
    squared[:, 0] = torch.where((image_vectors.abs().sum(dim=1) == 0), float("inf"), squared[:, 0])  # no self pair
    if bool((squared < screen).any()):
        raise RuntimeError(f"the GPU pair screen of {wrapped.shape[0]} atoms found no pairs, but atom 0 has "
                           "neighbours within range: the screen's launch did not run")


def screen_into(wrapped, image_vectors, radius, out, counter, half=False, max_grid_y=MAX_GRID_Y, cell=None):
    """Launch the CUDA screen into ``out`` (int64 codes) behind ``counter`` (int32, zeroed by the caller); no
    host synchronisation. Codes past ``out.numel()`` are dropped while ``counter`` still counts them, so
    ``counter > out.numel()`` means the buffer was too small (``SkinNeighbourList.build_gpu_async``). The caller
    checks an empty result with ``check_found`` when it reads the count."""

    natoms, nimages = wrapped.shape[0], image_vectors.shape[0]
    screen = (radius + SCREEN_MARGIN) ** 2
    plan = _bin_plan(wrapped, image_vectors, radius, cell, half)
    if plan is not None:
        _launch_binned(plan, wrapped.float().contiguous(), image_vectors.float().contiguous(), out, counter, natoms,
                       screen)
        return
    grid, grid_y, rounds = _grid(natoms, max_grid_y)
    _triton_kernel()[grid](wrapped.float().contiguous(), image_vectors.float().contiguous(), out, counter, natoms,
                           nimages, screen, out.numel(), grid_y, rounds, BLOCK=BLOCK, HALF=bool(half),
                           SINGLE=rounds == 1, num_warps=4)


def decode_codes(code, natoms):
    """(image, sender, receiver) int64 tensors of screen codes ``image * N^2 + sender * N + receiver``."""
    pairs = natoms * natoms
    image = code // pairs
    rest = code - image * pairs
    return image, rest // natoms, rest % natoms


def screen_codes(wrapped, image_vectors, radius, state=None, half=False, max_grid_y=MAX_GRID_Y, cell=None):
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
    plan = _bin_plan(wrapped, image_vectors, radius, cell, half)
    while True:
        out = state.get("buffer")
        if out is None:
            out = state["buffer"] = torch.empty(max(1 << 16, 128 * natoms), dtype=torch.int64,
                                                device=wrapped.device)
        if plan is not None:
            _launch_binned(plan, positions, images, out, counter, natoms, screen)
        else:
            _triton_kernel()[grid](positions, images, out, counter, natoms, nimages, screen, out.numel(), grid_y,
                                   rounds, BLOCK=BLOCK, HALF=bool(half), SINGLE=rounds == 1, num_warps=4)
        found = int(counter.item())  # host sync (rebuild steps only)
        check_found(found, wrapped, image_vectors, radius)
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
