"""Piecewise Chebyshev radial tables: edge length r -> the folded radial layer's input and the edge density.

``radial_tables`` (opt-6 OPT6-T05) replaces, per (sender species, receiver species) pair of a factored MH-1
interaction, the chain from the edge length to the folded last radial layer's input ``[h c, c]`` and to the
edge density ``tanh(density(r))^2 c`` by tables fitted at setup (``fastmace.rewrites.radial``). Every pair
shares one set of pieces; a piece holds a degree-``D`` Chebyshev series per output.

- **Lookup:** ``n_bins`` uniform bins over ``[r_lo, r_max)``, each inside exactly one piece (the pieces are a
  dyadic refinement of the bins' grid), give the piece of ``r`` in one gather: ``bins[(r - r_lo) * bin_scale]``.
  With ``x = (r - start) * scale - 1`` in ``[-1, 1]``, Clenshaw's recurrence sums the series.
- **Outside the domain:** ``r >= r_max`` (beyond the cutoff, also the engine's padded edges) gives exactly zero
  values and derivatives, as the cutoff does. ``r < r_lo`` is evaluated at ``r_lo`` with a zero derivative; the
  domain's lower end is set below any distance an MD run reaches (the fit's ``R_LO``).
- **Backward:** ``dL/dr = sum_j dL/dg_j g_j'(r)`` from the derivative series (``dcoef``, the Chebyshev
  derivative times the piece's ``scale``), recomputed: nothing is saved but ``r`` and the pair index.

Layouts: ``coef [P, K, D + 1, J]``, ``dcoef [P, K, D, J]`` (float32), ``starts``, ``scales [K]`` (float32),
``bins [n_bins]`` (int32); ``J = Q + 1``, the folded input's width ``Q`` and the edge density. On the CPU (and as
the certification's float32 reference) ``evaluate`` runs the same recurrence in PyTorch.
"""

from __future__ import annotations

from functools import cache

import torch

BLOCK_E = 32


def evaluate(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max, derivative=False):
    """PyTorch values ``[E, J]`` (or their r-derivatives, with ``dcoef`` as ``coef``) in ``coef``'s dtype."""
    r = r.reshape(-1).to(coef.dtype)
    inside = r < r_max
    clamped = torch.clamp(r, min=r_lo)
    index = torch.clamp(((clamped - r_lo) * bin_scale).to(torch.int64), 0, bins.shape[0] - 1)
    piece = bins[index].to(torch.int64)
    x = (clamped - starts[piece]) * scales[piece] - 1.0
    series = coef[pair.to(torch.int64), piece]  # [E, terms, J]
    b1 = torch.zeros_like(series[:, 0])
    b2 = torch.zeros_like(b1)
    for k in range(series.shape[1] - 1, 0, -1):
        b1, b2 = series[:, k] + 2.0 * x[:, None] * b1 - b2, b1
    values = series[:, 0] + x[:, None] * b1 - b2
    keep = inside if not derivative else inside & (r >= r_lo)
    return torch.where(keep[:, None], values, torch.zeros_like(values))


@cache
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def _series(coef_ptr, pair, piece, x, rows, j, column_mask, K, TERMS: tl.constexpr, J: tl.constexpr,
                BLOCK_E: tl.constexpr, BLOCK_J: tl.constexpr):
        base = ((pair.to(tl.int64) * K + piece.to(tl.int64)) * TERMS) * J
        mask = rows[:, None] & column_mask[None, :]
        b1 = tl.zeros((BLOCK_E, BLOCK_J), dtype=tl.float32)
        b2 = tl.zeros((BLOCK_E, BLOCK_J), dtype=tl.float32)
        for k in tl.static_range(TERMS - 1, 0, -1):
            c = tl.load(coef_ptr + base[:, None] + k * J + j[None, :], mask=mask, other=0.0)
            b = c + 2.0 * x[:, None] * b1 - b2
            b2 = b1
            b1 = b
        c0 = tl.load(coef_ptr + base[:, None] + j[None, :], mask=mask, other=0.0)
        return c0 + x[:, None] * b1 - b2

    @triton.jit
    def _locate(r_ptr, pair_ptr, starts_ptr, scales_ptr, bins_ptr, e, rows, r_lo, bin_scale, n_bins):
        r = tl.load(r_ptr + e, mask=rows, other=0.0)
        pair = tl.load(pair_ptr + e, mask=rows, other=0)
        clamped = tl.maximum(r, r_lo)
        index = tl.minimum(((clamped - r_lo) * bin_scale).to(tl.int32), n_bins - 1)
        piece = tl.load(bins_ptr + index, mask=rows, other=0)
        start = tl.load(starts_ptr + piece, mask=rows, other=0.0)
        scale = tl.load(scales_ptr + piece, mask=rows, other=1.0)
        x = (clamped - start) * scale - 1.0
        return r, pair, piece, x

    @triton.jit
    def forward(r_ptr, pair_ptr, coef_ptr, starts_ptr, scales_ptr, bins_ptr, g_ptr, density_ptr, E, K, n_bins,
                r_lo, bin_scale, r_max, TERMS: tl.constexpr, J: tl.constexpr, BLOCK_E: tl.constexpr,
                BLOCK_J: tl.constexpr):
        e = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
        rows = e < E
        r, pair, piece, x = _locate(r_ptr, pair_ptr, starts_ptr, scales_ptr, bins_ptr, e, rows, r_lo, bin_scale,
                                    n_bins)
        j = tl.arange(0, BLOCK_J)
        values = _series(coef_ptr, pair, piece, x, rows, j, j < J, K, TERMS, J, BLOCK_E, BLOCK_J)
        values = tl.where((r < r_max)[:, None], values, 0.0)
        e64 = e.to(tl.int64)
        tl.store(g_ptr + e64[:, None] * (J - 1) + j[None, :], values, mask=rows[:, None] & (j < J - 1)[None, :])
        density = tl.sum(tl.where((j == J - 1)[None, :], values, 0.0), axis=1)
        tl.store(density_ptr + e64, density, mask=rows)

    @triton.jit
    def backward(r_ptr, pair_ptr, dcoef_ptr, starts_ptr, scales_ptr, bins_ptr, grad_g_ptr, grad_density_ptr,
                 grad_r_ptr, E, K, n_bins, r_lo, bin_scale, r_max, TERMS: tl.constexpr, J: tl.constexpr,
                 BLOCK_E: tl.constexpr, BLOCK_J: tl.constexpr):
        e = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
        rows = e < E
        r, pair, piece, x = _locate(r_ptr, pair_ptr, starts_ptr, scales_ptr, bins_ptr, e, rows, r_lo, bin_scale,
                                    n_bins)
        j = tl.arange(0, BLOCK_J)
        slopes = _series(dcoef_ptr, pair, piece, x, rows, j, j < J, K, TERMS, J, BLOCK_E, BLOCK_J)
        e64 = e.to(tl.int64)
        grad = tl.load(grad_g_ptr + e64[:, None] * (J - 1) + j[None, :],
                       mask=rows[:, None] & (j < J - 1)[None, :], other=0.0)
        grad_density = tl.load(grad_density_ptr + e64, mask=rows, other=0.0)
        grad = tl.where((j == J - 1)[None, :], grad_density[:, None], grad)
        total = tl.sum(grad * slopes, axis=1)
        total = tl.where((r < r_max) & (r >= r_lo), total, 0.0)
        tl.store(grad_r_ptr + e64, total, mask=rows)

    return forward, backward


def _block_j(width):
    import triton

    return max(16, triton.next_power_of_2(int(width)))


@torch.library.custom_op("fastmace::radial_table_forward", mutates_args=())
def radial_table_forward(r: torch.Tensor, pair: torch.Tensor, coef: torch.Tensor, starts: torch.Tensor,
                         scales: torch.Tensor, bins: torch.Tensor, r_lo: float, bin_scale: float,
                         r_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(``[E, J - 1]`` folded input, ``[E, 1]`` edge density) of edge lengths ``r`` and pair indices ``pair``."""
    r, pair = r.reshape(-1).contiguous(), pair.contiguous()
    edges, width = r.shape[0], coef.shape[-1]
    if not r.is_cuda:
        values = evaluate(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max)
        return values[:, :-1].contiguous(), values[:, -1:].contiguous()
    g = r.new_empty((edges, width - 1))
    density = r.new_empty((edges, 1))
    if edges:
        forward, _ = _kernels()
        forward[(-(-edges // BLOCK_E),)](
            r, pair, coef, starts, scales, bins, g, density, edges, coef.shape[1], bins.shape[0], r_lo, bin_scale,
            r_max, TERMS=coef.shape[2], J=width, BLOCK_E=BLOCK_E, BLOCK_J=_block_j(width))
    return g, density


@radial_table_forward.register_fake
def _(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max):
    edges = r.numel()
    return r.new_empty((edges, coef.shape[-1] - 1)), r.new_empty((edges, 1))


@torch.library.custom_op("fastmace::radial_table_backward", mutates_args=())
def radial_table_backward(r: torch.Tensor, pair: torch.Tensor, dcoef: torch.Tensor, starts: torch.Tensor,
                          scales: torch.Tensor, bins: torch.Tensor, grad_g: torch.Tensor, grad_density: torch.Tensor,
                          r_lo: float, bin_scale: float, r_max: float) -> torch.Tensor:
    """``dL/dr`` ``[E, 1]`` from the gradients of the folded input and the edge density."""
    shape = r.shape
    r, pair = r.reshape(-1).contiguous(), pair.contiguous()
    grad_g, grad_density = grad_g.contiguous(), grad_density.reshape(-1).contiguous()
    edges, width = r.shape[0], dcoef.shape[-1]
    if not r.is_cuda:
        slopes = evaluate(r, pair, dcoef, starts, scales, bins, r_lo, bin_scale, r_max, derivative=True)
        grad = torch.cat([grad_g, grad_density[:, None]], dim=1)
        return (grad * slopes).sum(dim=1).reshape(shape)
    grad_r = r.new_empty((edges,))
    if edges:
        _, backward = _kernels()
        backward[(-(-edges // BLOCK_E),)](
            r, pair, dcoef, starts, scales, bins, grad_g, grad_density, grad_r, edges, dcoef.shape[1], bins.shape[0],
            r_lo, bin_scale, r_max, TERMS=dcoef.shape[2], J=width, BLOCK_E=BLOCK_E, BLOCK_J=_block_j(width))
    return grad_r.reshape(shape)


@radial_table_backward.register_fake
def _(r, pair, dcoef, starts, scales, bins, grad_g, grad_density, r_lo, bin_scale, r_max):
    return torch.empty_like(r)


class _RadialTable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, pair, coef, dcoef, starts, scales, bins, r_lo, bin_scale, r_max):
        ctx.save_for_backward(r, pair, dcoef, starts, scales, bins)
        ctx.constants = (r_lo, bin_scale, r_max)
        return radial_table_forward(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max)

    @staticmethod
    def backward(ctx, grad_g, grad_density):
        r, pair, dcoef, starts, scales, bins = ctx.saved_tensors
        if grad_g is None:
            grad_g = r.new_zeros((r.numel(), dcoef.shape[-1] - 1))
        if grad_density is None:
            grad_density = r.new_zeros((r.numel(), 1))
        grad_r = radial_table_backward(r, pair, dcoef, starts, scales, bins, grad_g, grad_density, *ctx.constants)
        return grad_r, None, None, None, None, None, None, None, None, None


def radial_table(r, pair, coef, dcoef, starts, scales, bins, r_lo, bin_scale, r_max):
    """``(folded input [E, J - 1], edge density [E, 1])``, differentiable in ``r`` (``[E, 1]`` or ``[E]``)."""
    return _RadialTable.apply(r, pair, coef, dcoef, starts, scales, bins, float(r_lo), float(bin_scale),
                              float(r_max))


__all__ = ["evaluate", "radial_table", "radial_table_backward", "radial_table_forward"]
