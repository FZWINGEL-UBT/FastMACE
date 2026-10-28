"""Per-piece w-tables: the folded radial layer's per-edge weights straight from the radial tables (ideas D2).

With ``radial_tables`` (opt-6 OPT6-T05) the second MH-1 interaction computes ``a = [h c, c]`` (65 wide) from a
piecewise Chebyshev table and then ``w = a W^T`` (``W [1280, 65]``, the folded layer); the backward forms
``da = dw W`` and the table's input gradient. Within one table piece ``a(r) = sum_k T_k(x) C_k``, so

    w(r) = sum_k T_k(x) G_k,   G = C W^T  ([terms, 1280] per (pair, piece)),

and ``dL/dr = sum_c dw_c w_c'(r)`` with ``G' = C' W^T`` from the table's derivative series. In float64 (17 terms,
degree 16) that is 17 multiply-adds per output instead of 65, and the same in the backward: ``tools/bench/
ideas_d2_wtable.py`` measured both kernels at 0.63-0.64 of the two float64 products on the RTX 5060 Ti.

``G`` and ``G'`` (174 KB per (pair, piece) in float64) do not fit the L2, so the edges are sorted by (pair, piece)
for these kernels, and each edge's ``w`` row is written to (and its ``dw`` row read from) its receiver-ordered
position. Edges at or beyond the cutoff (and padding) sort into one extra all-zero group: their ``w`` is exactly
zero and so is their slope, as with the tables. Below ``r_lo`` the table is evaluated at ``r_lo`` with a zero
slope (the tables' own rule).

The values equal the chain ``table -> a -> a W^T`` up to float64 rounding (the series and the product are
reassociated: ``sum_k T_k (C_k W^T)`` against ``(sum_k T_k C_k) W^T``).
"""

from __future__ import annotations

from functools import cache

import torch

FORWARD_TILES = (16, 128)  # BLOCK_E, BLOCK_Q (ideas_d2_wtable.py: within 1 % of the best of four on sm_120)
BACKWARD_TILES = (32, 32)


@cache
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def forward(x_ptr, group_ptr, perm_ptr, g_ptr, w_ptr, E, Q, TERMS: tl.constexpr, BLOCK_E: tl.constexpr,
                BLOCK_Q: tl.constexpr):
        e = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
        q = tl.program_id(1) * BLOCK_Q + tl.arange(0, BLOCK_Q)
        rows, cols = e < E, q < Q
        mask = rows[:, None] & cols[None, :]
        x = tl.load(x_ptr + e, mask=rows, other=0.0)
        group = tl.load(group_ptr + e, mask=rows, other=0).to(tl.int64)
        dest = tl.load(perm_ptr + e, mask=rows, other=0).to(tl.int64)
        base = group * TERMS * Q
        t0 = tl.full((BLOCK_E,), 1.0, x.dtype)
        t1 = x
        acc = t0[:, None] * tl.load(g_ptr + base[:, None] + q[None, :], mask=mask, other=0.0)
        acc += t1[:, None] * tl.load(g_ptr + base[:, None] + Q + q[None, :], mask=mask, other=0.0)
        for k in tl.static_range(2, TERMS):
            t0, t1 = t1, 2.0 * x * t1 - t0
            acc += t1[:, None] * tl.load(g_ptr + base[:, None] + k * Q + q[None, :], mask=mask, other=0.0)
        tl.store(w_ptr + dest[:, None] * Q + q[None, :], acc, mask=mask)

    @triton.jit
    def backward(x_ptr, group_ptr, perm_ptr, dg_ptr, dw_ptr, out_ptr, E, Q, TERMS: tl.constexpr,
                 BLOCK_E: tl.constexpr, BLOCK_Q: tl.constexpr):
        e = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
        rows = e < E
        x = tl.load(x_ptr + e, mask=rows, other=0.0)
        group = tl.load(group_ptr + e, mask=rows, other=0).to(tl.int64)
        src = tl.load(perm_ptr + e, mask=rows, other=0).to(tl.int64)
        base = group * TERMS * Q
        total = tl.zeros((BLOCK_E,), dtype=x.dtype)
        for start in range(0, Q, BLOCK_Q):
            q = start + tl.arange(0, BLOCK_Q)
            mask = rows[:, None] & (q < Q)[None, :]
            t0 = tl.full((BLOCK_E,), 1.0, x.dtype)
            t1 = x
            d = t0[:, None] * tl.load(dg_ptr + base[:, None] + q[None, :], mask=mask, other=0.0)
            d += t1[:, None] * tl.load(dg_ptr + base[:, None] + Q + q[None, :], mask=mask, other=0.0)
            for k in tl.static_range(2, TERMS):
                t0, t1 = t1, 2.0 * x * t1 - t0
                d += t1[:, None] * tl.load(dg_ptr + base[:, None] + k * Q + q[None, :], mask=mask, other=0.0)
            g = tl.load(dw_ptr + src[:, None] * Q + q[None, :], mask=mask, other=0.0)
            total += tl.sum(g * d, axis=1)
        tl.store(out_ptr + e, total, mask=rows)

    return forward, backward


def fold_tables(coef, dcoef, weight):
    """``(G, G')`` from a table's ``coef [P, K, D + 1, J]``, ``dcoef [P, K, D', J]`` and the folded weight
    ``[Q, J - 1]`` (``[W, b]``; the table's last column is the edge density): both ``[P * K + 1, D + 1, Q]``,
    each with a final all-zero group for edges outside the domain. The derivative series has ``D' = D`` terms
    when it is the value series' derivative (float32 tables) and ``D + 1`` when it is fitted itself (float64
    tables, f64-T01); shorter series are zero-padded. Computed in float64 on the tables' device."""
    pairs, pieces, terms, _ = coef.shape
    width = weight.shape[1]
    w64 = weight.detach().to(torch.float64)
    g = torch.einsum("pktj,qj->pktq", coef[..., :width].to(torch.float64), w64)
    d = torch.einsum("pktj,qj->pktq", dcoef[..., :width].to(torch.float64), w64)
    if d.shape[2] < terms:
        d = torch.cat([d, d.new_zeros((*d.shape[:2], terms - d.shape[2], d.shape[3]))], dim=2)
    zero = torch.zeros((1, terms, weight.shape[0]), dtype=torch.float64, device=coef.device)
    g = torch.cat([g.reshape(pairs * pieces, terms, -1), zero]).to(coef.dtype).contiguous()
    d = torch.cat([d.reshape(pairs * pieces, terms, -1), zero]).to(coef.dtype).contiguous()
    return g, d


def _locate(r, pair, pieces, starts, scales, bins, r_lo, bin_scale, r_max):
    """Sorted Chebyshev arguments, groups and the sort order; and the slope mask in edge order."""
    r = r.reshape(-1).to(starts.dtype)
    inside = r < r_max
    clamped = torch.clamp(r, min=r_lo)
    index = torch.clamp(((clamped - r_lo) * bin_scale).to(torch.int64), 0, bins.shape[0] - 1)
    piece = bins[index].to(torch.int64)
    x = (clamped - starts[piece]) * scales[piece] - 1.0
    group = torch.where(inside, pair.to(torch.int64) * pieces + piece, torch.full_like(piece, -1))  # -1: outside
    order = torch.argsort(group)
    slope = (inside & (r >= r_lo)).to(starts.dtype)
    return x, group, order, slope


@torch.library.custom_op("fastmace::w_table_forward", mutates_args=())
def w_table_forward(r: torch.Tensor, pair: torch.Tensor, g: torch.Tensor, starts: torch.Tensor,
                    scales: torch.Tensor, bins: torch.Tensor, r_lo: float, bin_scale: float,
                    r_max: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``w [E, Q]`` (receiver order) and, for the backward, the sorted arguments, groups, order and slope mask."""
    import triton

    groups, terms, q = g.shape
    pieces = starts.shape[0]
    x, group, order, slope = _locate(r, pair, pieces, starts, scales, bins, r_lo, bin_scale, r_max)
    group = torch.where(group < 0, torch.full_like(group, groups - 1), group)  # the all-zero group
    xs = x[order].contiguous()
    gs = group[order].to(torch.int32).contiguous()
    perm = order.to(torch.int32).contiguous()
    edges = xs.shape[0]
    w = g.new_empty((edges, q))
    if edges:
        be, bq = FORWARD_TILES
        _kernels()[0][(triton.cdiv(edges, be), triton.cdiv(q, bq))](xs, gs, perm, g, w, edges, q, TERMS=terms,
                                                                    BLOCK_E=be, BLOCK_Q=bq)
    return w, xs, gs, perm, slope


@w_table_forward.register_fake
def _(r, pair, g, starts, scales, bins, r_lo, bin_scale, r_max):
    edges = r.reshape(-1).shape[0]
    return (g.new_empty((edges, g.shape[2])), g.new_empty((edges,)), pair.new_empty((edges,), dtype=torch.int32),
            pair.new_empty((edges,), dtype=torch.int32), g.new_empty((edges,)))


@torch.library.custom_op("fastmace::w_table_backward", mutates_args=())
def w_table_backward(dw: torch.Tensor, dg: torch.Tensor, xs: torch.Tensor, gs: torch.Tensor, perm: torch.Tensor,
                     slope: torch.Tensor) -> torch.Tensor:
    """``dL/dr [E]`` (edge order) from the weights' gradient ``dw [E, Q]`` (receiver order)."""
    import triton

    _, terms, q = dg.shape
    edges = xs.shape[0]
    dw = dw.contiguous()
    sorted_dr = xs.new_empty((edges,))
    if edges:
        be, bq = BACKWARD_TILES
        _kernels()[1][(triton.cdiv(edges, be),)](xs, gs, perm, dg, dw, sorted_dr, edges, q, TERMS=terms,
                                                 BLOCK_E=be, BLOCK_Q=bq)
    dr = torch.empty_like(sorted_dr)
    dr[perm.to(torch.int64)] = sorted_dr
    return dr * slope


@w_table_backward.register_fake
def _(dw, dg, xs, gs, perm, slope):
    return xs.new_empty((xs.shape[0],))


def w_table_reference(r, pair, g, starts, scales, bins, r_lo, bin_scale, r_max):
    """PyTorch values ``[E, Q]`` of the w-table (CPU tests and checks)."""
    groups, terms, _ = g.shape
    x, group, _, _ = _locate(r, pair, starts.shape[0], starts, scales, bins, r_lo, bin_scale, r_max)
    group = torch.where(group < 0, torch.full_like(group, groups - 1), group)
    cheb = [torch.ones_like(x), x]
    for _ in range(2, terms):
        cheb.append(2.0 * x * cheb[-1] - cheb[-2])
    basis = torch.stack(cheb[:terms], dim=1)  # [E, terms]
    return torch.einsum("et,etq->eq", basis, g[group])


class _WTableConvolution(torch.autograd.Function):
    """``streaming_convolution(x, sh, w(r), ...)`` with ``w`` from the w-table; gradients to r, x and sh."""

    @staticmethod
    def forward(ctx, r, pair, g, dg, starts, scales, bins, r_lo, bin_scale, r_max, x, sh, senders, receivers,
                row_ptr, kernel_id):
        from fastmace.kernels.oeq_streaming.runtime import streaming_conv_forward

        w, xs, gs, perm, slope = w_table_forward(r, pair, g, starts, scales, bins, r_lo, bin_scale, r_max)
        message = streaming_conv_forward(x, sh, w, senders, row_ptr, kernel_id)
        ctx.save_for_backward(x, sh, w, senders, receivers, row_ptr, dg, xs, gs, perm, slope)
        ctx.kernel_id, ctx.r_shape = kernel_id, r.shape
        return message

    @staticmethod
    def backward(ctx, dout):
        from fastmace.kernels.oeq_streaming.runtime import streaming_conv_backward

        x, sh, w, senders, receivers, row_ptr, dg, xs, gs, perm, slope = ctx.saved_tensors
        dx, dsh, dw = streaming_conv_backward(x, sh, w, senders, receivers, row_ptr, dout, ctx.kernel_id)
        dr = w_table_backward(dw, dg, xs, gs, perm, slope).reshape(ctx.r_shape)
        return dr, None, None, None, None, None, None, None, None, None, dx, dsh, None, None, None, None


class WTable(torch.nn.Module):
    """One interaction's w-table, built from its radial table and folded weight (``fold_tables``)."""

    def __init__(self, table, weight):
        super().__init__()
        g, dg = fold_tables(table.coef, table.dcoef, weight)
        self.register_buffer("g", g, persistent=False)
        self.register_buffer("dg", dg, persistent=False)
        for name in ("starts", "scales", "bins", "columns"):
            self.register_buffer(name, getattr(table, name), persistent=False)
        self.species = table.species
        self.r_lo, self.bin_scale, self.r_max = table.r_lo, table.bin_scale, table.r_max

    def pair(self, node_attrs, edge_index):
        species = node_attrs[:, self.columns].argmax(dim=1).to(torch.int32)
        return species[edge_index[0]] * self.species + species[edge_index[1]]

    def convolution(self, lengths, pair, x, sh, senders, receivers, row_ptr, kernels):
        return _WTableConvolution.apply(lengths, pair, self.g, self.dg, self.starts, self.scales, self.bins,
                                        self.r_lo, self.bin_scale, self.r_max, x, sh, senders, receivers, row_ptr,
                                        kernels.id)

    def extra_repr(self):
        return f"groups={self.g.shape[0]}, terms={self.g.shape[1]}, outputs={self.g.shape[2]}"
