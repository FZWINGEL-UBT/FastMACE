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
SCREEN_MARGIN = 0.01  # Å, float32 screen above the float64 decision
CPU_CHUNK = 1 << 22  # elements of the (images, rows, atoms) distance block per CPU chunk


@functools.cache
def _triton_kernel():
    """The screening kernel, built on first use (Triton is imported only on CUDA)."""
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(pos_ptr, image_ptr, out_ptr, counter_ptr, natoms, nimages, screen, capacity,
               BLOCK: tl.constexpr, HALF: tl.constexpr):
        senders = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        receivers = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        valid = (senders < natoms)[:, None] & (receivers < natoms)[None, :]
        loops = nimages
        if HALF:  # tiles below the diagonal hold no sender <= receiver pair: no image loop
            valid = valid & (senders[:, None] <= receivers[None, :])
            loops = tl.where(tl.program_id(0) > tl.program_id(1), 0, nimages)
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

    return kernel


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


def screen_into(wrapped, image_vectors, radius, out, counter, half=False):
    """Launch the CUDA screen into ``out`` (int64 codes) behind ``counter`` (int32, zeroed by the caller); no
    host synchronisation. Codes past ``out.numel()`` are dropped while ``counter`` still counts them, so
    ``counter > out.numel()`` means the buffer was too small (``SkinNeighbourList.build_gpu_async``)."""
    import triton

    natoms, nimages = wrapped.shape[0], image_vectors.shape[0]
    screen = (radius + SCREEN_MARGIN) ** 2
    grid = (triton.cdiv(natoms, BLOCK), triton.cdiv(natoms, BLOCK))
    _triton_kernel()[grid](wrapped.float().contiguous(), image_vectors.float().contiguous(), out, counter, natoms,
                           nimages, screen, out.numel(), BLOCK=BLOCK, HALF=bool(half), num_warps=4)


def candidate_pairs(wrapped, image_vectors, radius, state=None, half=False):
    """(image, sender, receiver) int64 tensors of the screened pairs (unordered on CUDA).

    ``wrapped``: (N, 3) positions inside the cell; ``image_vectors``: (M, 3) Cartesian image
    translations; ``state``: optional dict that keeps the CUDA output buffer between calls;
    ``half``: only pairs with ``sender <= receiver``.
    """
    natoms, nimages = wrapped.shape[0], image_vectors.shape[0]
    screen = (radius + SCREEN_MARGIN) ** 2
    if not wrapped.is_cuda:
        return _candidate_pairs_cpu(wrapped, image_vectors, screen, half)
    import triton

    state = {} if state is None else state
    positions = wrapped.float().contiguous()
    images = image_vectors.float().contiguous()
    counter = torch.zeros(1, dtype=torch.int32, device=wrapped.device)
    grid = (triton.cdiv(natoms, BLOCK), triton.cdiv(natoms, BLOCK))
    while True:
        out = state.get("buffer")
        if out is None:
            out = state["buffer"] = torch.empty(max(1 << 16, 128 * natoms), dtype=torch.int64,
                                                device=wrapped.device)
        _triton_kernel()[grid](positions, images, out, counter, natoms, nimages, screen, out.numel(),
                               BLOCK=BLOCK, HALF=bool(half), num_warps=4)
        found = int(counter.item())  # host sync (rebuild steps only)
        if found <= out.numel():
            break
        state["buffer"] = torch.empty(int(found * 1.25) + 1, dtype=torch.int64, device=wrapped.device)
        counter.zero_()
    code = out[:found]
    pairs = natoms * natoms
    image = code // pairs
    rest = code - image * pairs
    return image, rest // natoms, rest % natoms
