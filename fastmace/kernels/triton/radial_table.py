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

Layouts: ``coef [P, K, D + 1, J]``, ``dcoef [P, K, D, J]``, ``starts``, ``scales [K]`` (all in the tables'
dtype: float32, or float64 for float64 plans, which then take float64 lengths), ``bins [n_bins]`` (int32);
``J = Q + 1``, the folded input's width ``Q`` and the edge density. On the CPU (and as the certification's
reference) ``evaluate`` runs the same recurrence in PyTorch.
"""

from __future__ import annotations

from functools import cache

import torch

BLOCK_E = 32
# Columns per program of the forward (None: all J in one program). The forward's register tile is [BLOCK_E,
# next_power_of_2(J)]: OMAT-0's J = 66 pads to 128 (212 registers, 13 % occupancy on the A100). With 64, a second
# column program takes the last two columns. Bitwise: every column is its own Clenshaw sum. A module constant for the
# A/B (OMAT-0 A100 profile, 3 November 2026); the backward sums over the columns and keeps one program.
COLUMN_BLOCK = None
WARPS = None  # the forward's num_warps (None: Triton's default, 4); a module constant for the lab (round 4)


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
        b1 = tl.zeros((BLOCK_E, BLOCK_J), dtype=coef_ptr.dtype.element_ty)  # float32, or float64 tables
        b2 = tl.zeros((BLOCK_E, BLOCK_J), dtype=coef_ptr.dtype.element_ty)
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
                BLOCK_J: tl.constexpr, DENSITY: tl.constexpr, G: tl.constexpr):
        e = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
        rows = e < E
        r, pair, piece, x = _locate(r_ptr, pair_ptr, starts_ptr, scales_ptr, bins_ptr, e, rows, r_lo, bin_scale,
                                    n_bins)
        j = tl.program_id(1) * BLOCK_J + tl.arange(0, BLOCK_J)  # this program's columns
        values = _series(coef_ptr, pair, piece, x, rows, j, j < J, K, TERMS, J, BLOCK_E, BLOCK_J)
        values = tl.where((r < r_max)[:, None], values, 0.0)
        e64 = e.to(tl.int64)
        tl.store(g_ptr + e64[:, None] * G + j[None, :], values, mask=rows[:, None] & (j < G)[None, :])
        if DENSITY:
            density = tl.sum(tl.where((j == J - 1)[None, :], values, 0.0), axis=1)
            last = (tl.program_id(1) + 1) * BLOCK_J >= J  # the program holding column J - 1
            tl.store(density_ptr + e64, density, mask=rows & last)

    @triton.jit
    def backward(r_ptr, pair_ptr, dcoef_ptr, starts_ptr, scales_ptr, bins_ptr, grad_g_ptr, grad_density_ptr,
                 grad_r_ptr, E, E_DENSITY, K, n_bins, r_lo, bin_scale, r_max, TERMS: tl.constexpr,
                 J: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_J: tl.constexpr, G: tl.constexpr):
        e = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
        rows = e < E
        r, pair, piece, x = _locate(r_ptr, pair_ptr, starts_ptr, scales_ptr, bins_ptr, e, rows, r_lo, bin_scale,
                                    n_bins)
        j = tl.arange(0, BLOCK_J)
        slopes = _series(dcoef_ptr, pair, piece, x, rows, j, j < J, K, TERMS, J, BLOCK_E, BLOCK_J)
        e64 = e.to(tl.int64)
        grad = tl.load(grad_g_ptr + e64[:, None] * G + j[None, :], mask=rows[:, None] & (j < G)[None, :], other=0.0)
        # E_DENSITY = 0 without a density gradient: nothing is read, and the same code as a zero gradient runs (a
        # compile-time branch let the compiler reorder the sum: last-bit differences).
        grad_density = tl.load(grad_density_ptr + e64, mask=e < E_DENSITY, other=0.0)
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
                         r_max: float, density: bool = True, columns: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    """(``[E, J - 1]`` folded input, ``[E, 1]`` edge density) of edge lengths ``r`` and pair indices ``pair``.

    ``density=False`` (a residual block, which has no density): the density is ``[E, 0]``, neither computed nor
    stored. ``columns`` (-1: J - 1): the folded input's leading columns only, ``[E, columns]``."""
    r, pair = r.reshape(-1).contiguous(), pair.contiguous()
    edges, width = r.shape[0], coef.shape[-1]
    columns = width - 1 if columns < 0 else int(columns)
    if not r.is_cuda:
        values = evaluate(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max)
        return values[:, :columns].contiguous(), values[:, width - 1:width - 1 + int(density)].contiguous()
    g = r.new_empty((edges, columns))
    out = r.new_empty((edges, int(density)))
    if edges:
        forward, _ = _kernels()
        block_j = _block_j(width) if COLUMN_BLOCK is None else min(_block_j(width), int(COLUMN_BLOCK))
        forward[(-(-edges // BLOCK_E), -(-width // block_j))](
            r, pair, coef, starts, scales, bins, g, out if density else g, edges, coef.shape[1], bins.shape[0], r_lo,
            bin_scale, r_max, TERMS=coef.shape[2], J=width, BLOCK_E=BLOCK_E, BLOCK_J=block_j,
            DENSITY=density, G=columns, **({} if WARPS is None else {"num_warps": int(WARPS)}))
    return g, out


@radial_table_forward.register_fake
def _(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max, density=True, columns=-1):
    edges = r.numel()
    columns = coef.shape[-1] - 1 if columns < 0 else int(columns)
    return r.new_empty((edges, columns)), r.new_empty((edges, int(density)))


@torch.library.custom_op("fastmace::radial_table_backward", mutates_args=())
def radial_table_backward(r: torch.Tensor, pair: torch.Tensor, dcoef: torch.Tensor, starts: torch.Tensor,
                          scales: torch.Tensor, bins: torch.Tensor, grad_g: torch.Tensor, grad_density: torch.Tensor,
                          r_lo: float, bin_scale: float, r_max: float) -> torch.Tensor:
    """``dL/dr`` ``[E, 1]`` from the gradients of the folded input (its leading ``grad_g.shape[1]`` columns; the
    others' are zero) and the edge density (``[E, 0]``: none)."""
    shape = r.shape
    r, pair = r.reshape(-1).contiguous(), pair.contiguous()
    density = grad_density.shape[-1] > 0
    grad_g, grad_density = grad_g.contiguous(), grad_density.reshape(-1).contiguous()
    edges, width = r.shape[0], dcoef.shape[-1]
    if not r.is_cuda:
        slopes = evaluate(r, pair, dcoef, starts, scales, bins, r_lo, bin_scale, r_max, derivative=True)
        grad_g = torch.nn.functional.pad(grad_g, (0, width - 1 - grad_g.shape[1]))
        grad = torch.cat([grad_g, grad_density[:, None] if density else grad_g.new_zeros((edges, 1))], dim=1)
        return (grad * slopes).sum(dim=1).reshape(shape)
    grad_r = r.new_empty((edges,))
    if edges:
        _, backward = _kernels()
        backward[(-(-edges // BLOCK_E),)](
            r, pair, dcoef, starts, scales, bins, grad_g, grad_density if density else grad_g, grad_r, edges,
            edges if density else 0, dcoef.shape[1], bins.shape[0], r_lo, bin_scale, r_max, TERMS=dcoef.shape[2],
            J=width, BLOCK_E=BLOCK_E, BLOCK_J=_block_j(width), G=grad_g.shape[1])
    return grad_r.reshape(shape)


@radial_table_backward.register_fake
def _(r, pair, dcoef, starts, scales, bins, grad_g, grad_density, r_lo, bin_scale, r_max):
    return torch.empty_like(r)


class _RadialTable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, pair, coef, dcoef, starts, scales, bins, r_lo, bin_scale, r_max, density, columns):
        ctx.save_for_backward(r, pair, dcoef, starts, scales, bins)
        ctx.constants = (r_lo, bin_scale, r_max)
        ctx.density = density
        ctx.columns = dcoef.shape[-1] - 1 if columns < 0 else columns
        return radial_table_forward(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max, density, columns)

    @staticmethod
    def backward(ctx, grad_g, grad_density):
        r, pair, dcoef, starts, scales, bins = ctx.saved_tensors
        if grad_g is None:
            grad_g = r.new_zeros((r.numel(), ctx.columns))
        if grad_density is None:
            grad_density = r.new_zeros((r.numel(), int(ctx.density)))
        grad_r = radial_table_backward(r, pair, dcoef, starts, scales, bins, grad_g, grad_density, *ctx.constants)
        return grad_r, None, None, None, None, None, None, None, None, None, None, None


def radial_table(r, pair, coef, dcoef, starts, scales, bins, r_lo, bin_scale, r_max, density=True, columns=-1):
    """``(folded input [E, J - 1], edge density [E, 1])``, differentiable in ``r`` (``[E, 1]`` or ``[E]``);
    ``density=False``: the density ``[E, 0]`` (not computed); ``columns`` >= 0: the folded input's leading
    ``columns`` only (its backward then reads no gradient of the others)."""
    return _RadialTable.apply(r, pair, coef, dcoef, starts, scales, bins, float(r_lo), float(bin_scale),
                              float(r_max), bool(density), int(columns))


__all__ = ["evaluate", "radial_table", "radial_table_backward", "radial_table_forward"]
