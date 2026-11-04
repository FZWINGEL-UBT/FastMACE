"""Per-species row products of the generic species skip (``SpeciesSkip``, ``SPECIES_GEMM``).

``species_mm(x, w, plan)`` is ``y[a, j] = x[a, j] @ w[species[a]]`` for ``x [atoms, size, K]`` (any strides) and the
stacked matrices ``w [species, K, N]``: every atom's rows through its own species' matrix only, instead of through
all present species' concatenated matrices followed by the selection of the atom's block (a third of the products
for three species). Its input gradient is the same product with the transposed matrices.

The grid is fixed for a structure's atom count, so the call can be captured in CUDA graphs and survives atoms
changing species: ``species_plan`` sorts the atoms by species on the device (stable) and cuts each species' rows
(atom, j) into tiles of ``BLOCK_M`` rows, at most ``ceil(atoms * size / BLOCK_M) + species`` of them; the tiles past
the last are empty. A tile gathers its rows from ``x`` and scatters its results to ``y``; every row is written once.

Each output element is the same K-long dot product as in the concatenated product, reduced in ``BLOCK_K`` steps;
``precision`` "ieee" (FP32 FMA) or "tf32x3" (``tf32x3_gemms``). On the CPU the product runs in PyTorch.
"""

from __future__ import annotations

from functools import cache

import torch

BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
NUM_WARPS, NUM_STAGES = 4, 2


@cache
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gemm(x_ptr, w_ptr, y_ptr, order_ptr, tile_ptr, size, N, K, sxa, sxj, sxk, swk, swn, sya, syj, syn,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PRECISION: tl.constexpr):
        t, pn = tl.program_id(0), tl.program_id(1)
        species = tl.load(tile_ptr + 3 * t)
        start = tl.load(tile_ptr + 3 * t + 1)
        end = tl.load(tile_ptr + 3 * t + 2)
        if start >= end:
            return
        r = start + tl.arange(0, BLOCK_M)
        live = r < end
        i = tl.where(live, r // size, 0)
        j = r - i * size
        atom = tl.load(order_ptr + i, mask=live, other=0).to(tl.int64)
        xrow = atom * sxa + j.to(tl.int64) * sxj
        rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        wbase = w_ptr + species.to(tl.int64) * K * N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + rk
            a = tl.load(x_ptr + xrow[:, None] + kk[None, :] * sxk, mask=live[:, None] & (kk < K)[None, :], other=0.0)
            b = tl.load(wbase + kk[:, None] * swk + rn[None, :] * swn, mask=(kk < K)[:, None] & (rn < N)[None, :],
                        other=0.0)
            acc += tl.dot(a, b, input_precision=PRECISION)
        yrow = atom * sya + j.to(tl.int64) * syj
        tl.store(y_ptr + yrow[:, None] + rn[None, :] * syn, acc, mask=live[:, None] & (rn < N)[None, :])

    return gemm


def tiles(atoms, size, species):
    """The fixed tile count of ``atoms`` atoms with ``size`` rows each over ``species`` species."""
    return -(-atoms * size // BLOCK_M) + species


def species_plan(index, species, size):
    """(order [atoms] int32, tiles [T, 3] int32: species, first row, end row) for the atoms' species ``index``
    ([atoms], values in ``range(species)``) and ``size`` rows per atom; fixed shapes, on ``index``'s device."""
    atoms = index.shape[0]
    order = torch.argsort(index, stable=True).to(torch.int32)
    counts = torch.zeros(species, dtype=torch.int64, device=index.device)  # no bincount: it syncs (max)
    counts = counts.index_add(0, index.long(), torch.ones_like(index, dtype=torch.int64)) * size
    ends = torch.cumsum(counts, 0)
    per = (counts + BLOCK_M - 1) // BLOCK_M
    tile_ends = torch.cumsum(per, 0)
    t = torch.arange(tiles(atoms, size, species), device=index.device)
    group = torch.searchsorted(tile_ends, t, right=True)
    valid = group < species
    g = group.clamp(max=species - 1)
    first = ends[g] - counts[g] + (t - (tile_ends[g] - per[g])) * BLOCK_M
    end = torch.where(valid, ends[g], first)  # empty past the last tile
    return order, torch.stack([g, first, end], dim=1).to(torch.int32).contiguous()


def _launch(x, w, order, tile, out, precision):
    import triton

    atoms, size, k = x.shape
    n = w.shape[2]
    grid = (tile.shape[0], triton.cdiv(n, BLOCK_N))
    _kernel()[grid](x, w, out, order, tile, size, n, k, x.stride(0), x.stride(1), x.stride(2), w.stride(1),
                    w.stride(2), out.stride(0), out.stride(1), out.stride(2), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                    BLOCK_K=BLOCK_K, PRECISION=precision, num_warps=NUM_WARPS, num_stages=NUM_STAGES)


def _reference(x, w, index):
    return torch.einsum("ajk,akn->ajn", x, w[index.long()])


@torch.library.custom_op("fastmace::species_mm", mutates_args=())
def species_mm(x: torch.Tensor, w: torch.Tensor, w_t: torch.Tensor, index: torch.Tensor, order: torch.Tensor,
               tile: torch.Tensor, precision: str) -> torch.Tensor:
    """``x[a] @ w[index[a]]`` for ``x [atoms, size, K]`` and ``w [species, K, N]`` (``w_t``: w transposed, for the
    gradient; ``order``, ``tile``: ``species_plan``'s); PyTorch on the CPU and for other dtypes."""
    if x.device.type != "cuda" or x.dtype != torch.float32:
        return _reference(x, w, index)
    out = x.new_empty((x.shape[0], x.shape[1], w.shape[2]))
    if out.numel():
        _launch(x, w, order, tile, out, precision)
    return out


@species_mm.register_fake
def _(x, w, w_t, index, order, tile, precision):
    return x.new_empty((x.shape[0], x.shape[1], w.shape[2]))


def _setup_context(ctx, inputs, output):
    x, w, w_t, index, order, tile, precision = inputs
    ctx.save_for_backward(w, w_t, index, order, tile)
    ctx.precision = precision


def _backward(ctx, grad):
    w, w_t, index, order, tile = ctx.saved_tensors
    grad_x = species_mm(grad, w_t, w, index, order, tile, ctx.precision) if ctx.needs_input_grad[0] else None
    return grad_x, None, None, None, None, None, None


species_mm.register_autograd(_backward, setup_context=_setup_context)


__all__ = ["BLOCK_K", "BLOCK_M", "BLOCK_N", "species_mm", "species_plan", "tiles"]
