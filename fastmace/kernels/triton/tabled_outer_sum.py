"""The first interaction's radial table and outer-product sums with one fused backward (round 2, 5 November 2026).

The reassociated first interaction (tabled generic family, OMAT-0) evaluates its folded input ``a = g(r) [E, K]`` from
the radial table (``radial_table``), then the per-slot sums ``S[key] += y a^T`` (``edge_outer_sum``). Its backward
ran three passes: ``edge_outer_sum_backward`` wrote ``grad_a [E, K]``, a compiled pass assembled the table's gradient
from it, and ``radial_table_backward`` read that back for ``dL/dr = sum_j grad_g_j g_j'(r)``. ``tabled_outer_sum``
keeps the forward and fuses the backward into one walk over the edges (``edge_outer_sum``'s ``BACKWARD_WALK``): per
edge, ``grad_a = G[key]^T y`` stays in registers and is contracted at once with the table's derivative row, so
neither ``grad_a`` nor the assembled gradient is written. The edge density's gradient joins the same sum.

Rounding: the sums are regrouped (tree sums over d and j). Any key order is correct; runs of equal keys (the segmented
first sum's receiver and sender-species order) load each slot's block once. Float32 CUDA tensors only; elsewhere the
reference composition (``edge_outer_sum_backward`` and ``radial_table_backward``) runs.
"""

from __future__ import annotations

from functools import cache

import torch

from fastmace.kernels.gpu_models import GPUModels

CHUNK = 16  # edges per program (the walk's A100 optimum, tools/eos_backward_lab.py)
WARPS = 1
# Phase 2: the forward evaluates a = g(r) inside the segmented sum's walk (no [E, K] table output written and read
# back) and the backward evaluates it again next to the derivative row (nothing of a saved). Rounding (its own
# Clenshaw sums, the same recurrence). None: on for TABLED_FORWARD_GPUS (decision 66, after the MD-step A/Bs); True /
# False: an A/B side's choice on every GPU. MD step b/a (ab / ba) with FUSED_FIRST_BACKWARD against the default,
# OMAT-0 medium, SrTiO3 10x10x10 and NbCl5: RTX 5060 Ti 0.945 / 0.945, 0.948 / 0.955; RTX 4000 Ada 0.949 / 0.944,
# 0.978 / 0.968 (FUSED_FIRST_BACKWARD alone 0.962-0.979); A100-SXM4-80GB 0.966 / 0.962, 0.970 / 0.982 (alone
# 0.966-0.984; every A100 model).
TABLED_FORWARD = None
TABLED_FORWARD_GPUS = GPUModels(("NVIDIA GeForce RTX 5060 Ti", "NVIDIA RTX 4000 Ada Generation",
                                 "NVIDIA A100-SXM4-80GB"))
FORWARD_CHUNK = 32  # edges per program of the tabled forward (edge_outer_sum.FORWARD_CHUNK)
FORWARD_WARPS = 1


def tabled_forward(device):
    """``TABLED_FORWARD`` on ``device``: the constant when set, else on for ``TABLED_FORWARD_GPUS``."""
    if TABLED_FORWARD is not None:
        return bool(TABLED_FORWARD)
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_name(device) in TABLED_FORWARD_GPUS


@cache
def _kernel():
    import triton
    import triton.language as tl

    from fastmace.kernels.triton.edge_outer_sum import _base_fn

    _base = _base_fn()

    @triton.jit
    def kernel(grad_ptr, a_ptr, y_ptr, key_ptr, r_ptr, pair_ptr, dcoef_ptr, starts_ptr, scales_ptr, bins_ptr,
               grad_density_ptr, dr_ptr, grad_y_ptr, num_edges, a_stride, E_DENSITY, n_pieces, n_bins, r_lo,
               bin_scale, r_max, K: tl.constexpr, D: tl.constexpr, INNER: tl.constexpr, TERMS: tl.constexpr,
               J: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
               RECOMPUTE: tl.constexpr, coef_ptr, VTERMS: tl.constexpr):
        start = tl.program_id(0) * CHUNK
        d = tl.arange(0, BLOCK_D)
        k = tl.arange(0, BLOCK_K)
        mask = (d < D)[:, None] & (k < K)[None, :]
        offsets = d[:, None] * (INNER * K) + k[None, :]
        current = tl.load(key_ptr + start).to(tl.int64)
        block = tl.load(grad_ptr + _base(current, D, K, INNER) + offsets, mask=mask, other=0.0)
        stop = tl.minimum(start + CHUNK, num_edges)
        for edge in range(start, stop):
            key = tl.load(key_ptr + edge).to(tl.int64)
            if key != current:  # a new run: its slot's block, once
                block = tl.load(grad_ptr + _base(key, D, K, INNER) + offsets, mask=mask, other=0.0)
                current = key
            row = edge.to(tl.int64)
            # The table's lookup at r (radial_table's rule).
            r = tl.load(r_ptr + row)
            pair = tl.load(pair_ptr + row).to(tl.int64)
            clamped = tl.maximum(r, r_lo)
            index = tl.minimum(((clamped - r_lo) * bin_scale).to(tl.int32), n_bins - 1)
            piece = tl.load(bins_ptr + index).to(tl.int64)
            x = (clamped - tl.load(starts_ptr + piece)) * tl.load(scales_ptr + piece) - 1.0
            if RECOMPUTE:  # a = g(r) again from the value series (coef [P, pieces, VTERMS, J])
                vbase = (pair * n_pieces + piece) * VTERMS * J
                v1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
                v2 = tl.zeros((BLOCK_K,), dtype=tl.float32)
                for t in tl.static_range(VTERMS - 1, 0, -1):
                    c = tl.load(coef_ptr + vbase + t * J + k, mask=k < K, other=0.0)
                    v = c + 2.0 * x * v1 - v2
                    v2 = v1
                    v1 = v
                a = tl.load(coef_ptr + vbase + k, mask=k < K, other=0.0) + x * v1 - v2
                a = tl.where(r < r_max, a, 0.0)
            else:
                a = tl.load(a_ptr + row * a_stride + k, mask=k < K, other=0.0)
            y = tl.load(y_ptr + row * D + d, mask=d < D, other=0.0)
            tl.store(grad_y_ptr + row * D + d, tl.sum(block * a[None, :], axis=1), mask=d < D)
            grad_a = tl.sum(block * y[:, None], axis=0)
            # The derivative row (dcoef [P, pieces, TERMS, J]).
            base = (pair * n_pieces + piece) * TERMS * J
            b1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
            b2 = tl.zeros((BLOCK_K,), dtype=tl.float32)
            c1 = 0.0 * x  # the density column J - 1 (a scalar Clenshaw sum)
            c2 = 0.0 * x
            for t in tl.static_range(TERMS - 1, 0, -1):
                c = tl.load(dcoef_ptr + base + t * J + k, mask=k < K, other=0.0)
                b = c + 2.0 * x * b1 - b2
                b2 = b1
                b1 = b
                cd = tl.load(dcoef_ptr + base + t * J + (J - 1))
                cn = cd + 2.0 * x * c1 - c2
                c2 = c1
                c1 = cn
            slope = tl.load(dcoef_ptr + base + k, mask=k < K, other=0.0) + x * b1 - b2
            slope_density = tl.load(dcoef_ptr + base + (J - 1)) + x * c1 - c2
            grad_density = tl.load(grad_density_ptr + row, mask=edge < E_DENSITY, other=0.0)
            total = tl.sum(grad_a * slope, axis=0) + grad_density * slope_density
            total = tl.where((r < r_max) & (r >= r_lo), total, 0.0)
            tl.store(dr_ptr + row, total)

    return kernel


@cache
def _forward_kernel():
    import triton
    import triton.language as tl

    from fastmace.kernels.triton.edge_outer_sum import _base_fn

    _base = _base_fn()

    @triton.jit
    def kernel(y_ptr, key_ptr, r_ptr, pair_ptr, coef_ptr, starts_ptr, scales_ptr, bins_ptr, out_ptr, density_ptr,
               num_edges, n_pieces, n_bins, r_lo, bin_scale, r_max, K: tl.constexpr, D: tl.constexpr,
               INNER: tl.constexpr, TERMS: tl.constexpr, J: tl.constexpr, CHUNK: tl.constexpr, BLOCK_K: tl.constexpr,
               BLOCK_D: tl.constexpr):
        start = tl.program_id(0) * CHUNK
        d = tl.arange(0, BLOCK_D)
        k = tl.arange(0, BLOCK_K)
        mask = (d < D)[:, None] & (k < K)[None, :]
        offsets = d[:, None] * (INNER * K) + k[None, :]
        acc = tl.zeros((BLOCK_D, BLOCK_K), dtype=tl.float32)
        current = tl.load(key_ptr + start).to(tl.int64)
        stop = tl.minimum(start + CHUNK, num_edges)
        for edge in range(start, stop):
            key = tl.load(key_ptr + edge).to(tl.int64)
            if key != current:  # a run ends: add its sum to its slot
                tl.atomic_add(out_ptr + _base(current, D, K, INNER) + offsets, acc, mask=mask, sem="relaxed")
                acc = tl.zeros((BLOCK_D, BLOCK_K), dtype=tl.float32)
                current = key
            row = edge.to(tl.int64)
            r = tl.load(r_ptr + row)
            pair = tl.load(pair_ptr + row).to(tl.int64)
            clamped = tl.maximum(r, r_lo)
            index = tl.minimum(((clamped - r_lo) * bin_scale).to(tl.int32), n_bins - 1)
            piece = tl.load(bins_ptr + index).to(tl.int64)
            x = (clamped - tl.load(starts_ptr + piece)) * tl.load(scales_ptr + piece) - 1.0
            base = (pair * n_pieces + piece) * TERMS * J
            v1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
            v2 = tl.zeros((BLOCK_K,), dtype=tl.float32)
            c1 = 0.0 * x
            c2 = 0.0 * x
            for t in tl.static_range(TERMS - 1, 0, -1):
                c = tl.load(coef_ptr + base + t * J + k, mask=k < K, other=0.0)
                v = c + 2.0 * x * v1 - v2
                v2 = v1
                v1 = v
                cd = tl.load(coef_ptr + base + t * J + (J - 1))
                cn = cd + 2.0 * x * c1 - c2
                c2 = c1
                c1 = cn
            inside = r < r_max
            a = tl.where(inside, tl.load(coef_ptr + base + k, mask=k < K, other=0.0) + x * v1 - v2, 0.0)
            density = tl.where(inside, tl.load(coef_ptr + base + (J - 1)) + x * c1 - c2, 0.0)
            tl.store(density_ptr + row, density)
            y = tl.load(y_ptr + row * D + d, mask=d < D, other=0.0)
            acc += y[:, None] * a[None, :]
        tl.atomic_add(out_ptr + _base(current, D, K, INNER) + offsets, acc, mask=mask, sem="relaxed")

    return kernel


@torch.library.custom_op("fastmace::tabled_outer_sum_forward", mutates_args=())
def tabled_outer_sum_forward(r: torch.Tensor, pair: torch.Tensor, coef: torch.Tensor, starts: torch.Tensor,
                             scales: torch.Tensor, bins: torch.Tensor, r_lo: float, bin_scale: float, r_max: float,
                             columns: int, y: torch.Tensor, key: torch.Tensor, slots: int,
                             inner: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """(S [slots, D * K], edge density [E, 1]) with a = the table's leading ``columns`` values evaluated in the walk
    (``TABLED_FORWARD``); float32 CUDA (elsewhere: the table, then ``edge_outer_sum_forward``)."""
    import triton

    from fastmace.kernels.triton.edge_outer_sum import edge_outer_sum_forward
    from fastmace.kernels.triton.radial_table import radial_table_forward
    edges, d, k = r.numel(), y.shape[1], int(columns)
    if not (r.is_cuda and coef.dtype == torch.float32 and y.dtype == torch.float32):
        a, density = radial_table_forward(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max, True, k)
        return edge_outer_sum_forward(a, y, key, slots, inner), density
    r, pair, y, key = r.reshape(-1).contiguous(), pair.contiguous(), y.contiguous(), key.contiguous()
    out = torch.zeros((slots, d * k), dtype=y.dtype, device=y.device)
    density = r.new_empty((edges, 1))
    if edges:
        _forward_kernel()[(triton.cdiv(edges, FORWARD_CHUNK),)](
            y, key, r, pair, coef, starts, scales, bins, out, density, edges, coef.shape[1], bins.shape[0],
            float(r_lo), float(bin_scale), float(r_max), K=k, D=d, INNER=inner, TERMS=coef.shape[2],
            J=coef.shape[-1], CHUNK=FORWARD_CHUNK, BLOCK_K=triton.next_power_of_2(k),
            BLOCK_D=triton.next_power_of_2(d), num_warps=FORWARD_WARPS)
    return out, density


@tabled_outer_sum_forward.register_fake
def _(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max, columns, y, key, slots, inner=1):
    return y.new_empty((slots, y.shape[1] * int(columns))), r.new_empty((r.numel(), 1))


@torch.library.custom_op("fastmace::tabled_outer_sum_backward", mutates_args=())
def tabled_outer_sum_backward(grad: torch.Tensor, a: torch.Tensor, y: torch.Tensor, key: torch.Tensor,
                              r: torch.Tensor, pair: torch.Tensor, dcoef: torch.Tensor, starts: torch.Tensor,
                              scales: torch.Tensor, bins: torch.Tensor, grad_density: torch.Tensor, r_lo: float,
                              bin_scale: float, r_max: float, inner: int = 1,
                              coef: torch.Tensor | None = None, columns: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """(dL/dr [E], grad_y [E, D]) from the slots' gradient ``grad [slots, D * K]`` (``inner`` layout), the table rows
    ``a [E, K]`` (the table's leading K columns) and the edge density's gradient (``[E, 0]``: none). With ``coef``
    (``TABLED_FORWARD``): ``a`` is evaluated again from the value table's leading ``columns`` (``a`` unused)."""
    if coef is not None:
        if not (grad.is_cuda and coef.dtype == torch.float32):
            from fastmace.kernels.triton.radial_table import radial_table_forward
            a, _ = radial_table_forward(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max, False,
                                        int(columns))
            coef = None
        else:
            a = r.new_empty((r.numel(), int(columns)))  # shape only: the kernel evaluates a
    num_edges, k = a.shape
    d = y.shape[1]
    grad_density = grad_density.reshape(-1)
    fused = (grad.is_cuda and grad.dtype == torch.float32 and a.dtype == torch.float32 and dcoef.dtype == torch.float32
             and k <= dcoef.shape[-1] - 1)
    if not fused:
        from fastmace.kernels.triton.edge_outer_sum import edge_outer_sum_backward
        from fastmace.kernels.triton.radial_table import radial_table_backward

        grad_a, grad_y = edge_outer_sum_backward(grad, a, y, key, inner)
        density = grad_density.reshape(-1, 1) if grad_density.numel() else grad_density.reshape(-1, 0)
        dr = radial_table_backward(r.reshape(-1), pair, dcoef, starts, scales, bins, grad_a, density, r_lo, bin_scale,
                                   r_max)
        return dr.reshape(-1), grad_y
    import triton

    from fastmace.kernels.triton.edge_outer_sum import _rows
    grad, a, y, key = grad.contiguous(), _rows(a), y.contiguous(), key.contiguous()
    r, pair, grad_density = r.reshape(-1).contiguous(), pair.contiguous(), grad_density.contiguous()
    dr = r.new_empty((num_edges,))
    grad_y = torch.empty_like(y)
    if num_edges:
        _kernel()[(triton.cdiv(num_edges, CHUNK),)](
            grad, a, y, key, r, pair, dcoef, starts, scales, bins, grad_density if grad_density.numel() else r, dr,
            grad_y, num_edges, a.stride(0), grad_density.numel(), dcoef.shape[1], bins.shape[0], float(r_lo),
            float(bin_scale), float(r_max), K=k, D=d, INNER=inner, TERMS=dcoef.shape[2], J=dcoef.shape[-1],
            CHUNK=CHUNK, BLOCK_K=triton.next_power_of_2(k), BLOCK_D=triton.next_power_of_2(d),
            RECOMPUTE=coef is not None, coef_ptr=coef if coef is not None else dcoef,
            VTERMS=coef.shape[2] if coef is not None else 1, num_warps=WARPS)
    return dr, grad_y


@tabled_outer_sum_backward.register_fake
def _(grad, a, y, key, r, pair, dcoef, starts, scales, bins, grad_density, r_lo, bin_scale, r_max, inner=1,
      coef=None, columns=0):
    return r.new_empty((r.numel(),)), torch.empty_like(y)


class _TabledOuterSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, pair, coef, dcoef, starts, scales, bins, r_lo, bin_scale, r_max, columns, y, key, slots,
                inner):
        from fastmace.kernels.triton.edge_outer_sum import edge_outer_sum_forward
        from fastmace.kernels.triton.radial_table import radial_table_forward

        ctx.constants = (r_lo, bin_scale, r_max)
        ctx.inner, ctx.columns = inner, int(columns)
        ctx.shape, ctx.slots = r.shape, int(slots)
        ctx.tabled = tabled_forward(r.device) and r.dtype == torch.float32
        if ctx.tabled:  # nothing of a kept: the backward evaluates it again
            ctx.save_for_backward(coef, y, key, r, pair, dcoef, starts, scales, bins)
            return tabled_outer_sum_forward(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max, columns, y,
                                            key, slots, inner)
        a, density = radial_table_forward(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max, True, columns)
        ctx.save_for_backward(a, y, key, r, pair, dcoef, starts, scales, bins)
        return edge_outer_sum_forward(a, y, key, slots, inner), density

    @staticmethod
    def backward(ctx, grad, grad_density):
        first, y, key, r, pair, dcoef, starts, scales, bins = ctx.saved_tensors
        if grad is None:
            grad = y.new_zeros((ctx.slots, y.shape[1] * ctx.columns))
        if grad_density is None:
            grad_density = y.new_zeros((0,))
        if ctx.tabled:  # first is coef
            dr, grad_y = tabled_outer_sum_backward(grad, y.new_empty((0, 0)), y, key, r, pair, dcoef, starts, scales,
                                                   bins, grad_density, *ctx.constants, ctx.inner, first, ctx.columns)
        else:  # first is a
            dr, grad_y = tabled_outer_sum_backward(grad, first, y, key, r, pair, dcoef, starts, scales, bins,
                                                   grad_density, *ctx.constants, ctx.inner)
        return (dr.reshape(ctx.shape), None, None, None, None, None, None, None, None, None, None, grad_y, None,
                None, None)


def tabled_outer_sum(r, pair, coef, dcoef, starts, scales, bins, r_lo, bin_scale, r_max, columns, y, key, slots,
                     inner=1):
    """``(S [slots, D * K], edge density [E, 1])``: the table's leading ``columns`` = K values ``a`` of each edge
    length ``r``, summed as ``S[key] += y a^T`` (``edge_outer_sum``'s segmented forward, d major), with the fused
    backward (module docstring). Differentiable in ``r`` and ``y``."""
    return _TabledOuterSum.apply(r, pair, coef, dcoef, starts, scales, bins, float(r_lo), float(bin_scale),
                                 float(r_max), int(columns), y, key, int(slots), int(inner))


__all__ = ["CHUNK", "FORWARD_CHUNK", "TABLED_FORWARD", "TABLED_FORWARD_GPUS", "WARPS", "tabled_forward",
           "tabled_outer_sum", "tabled_outer_sum_backward", "tabled_outer_sum_forward"]
