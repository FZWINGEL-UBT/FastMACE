"""``main``'s kernels before the fifth list's changes (plans/capacity-ideas-20261030.md), for ``ab_swap.py``'s side a.

Copied from ``main`` at ``e021a5a7``; each name replaces one module attribute of the library (``ab_swap.py --swap``).
"""

from __future__ import annotations

from functools import cache

import torch

from fastmace.kernels.triton.edge_outer_sum import edge_outer_sum
from fastmace.kernels.triton.segment_gemm import joined_products
from fastmace.rewrites.mh1.modules import FirstInteractionReassociation

# The library forward, captured at import (before ab_swap replaces it with its dispatcher).
_LIBRARY_FIRST_FORWARD = FirstInteractionReassociation.forward


@cache
def _edge_outer_sum_backward_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(grad_ptr, a_ptr, y_ptr, key_ptr, grad_a_ptr, grad_y_ptr, num_edges,
               K: tl.constexpr, D: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_K: tl.constexpr):
        edges = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
        edge_mask = edges < num_edges
        edges = edges.to(tl.int64)
        k = tl.arange(0, BLOCK_K)
        mask = edge_mask[:, None] & (k < K)[None, :]
        key = tl.load(key_ptr + edges, mask=edge_mask, other=0).to(tl.int64)
        a = tl.load(a_ptr + edges[:, None] * K + k[None, :], mask=mask, other=0.0)
        rows = grad_ptr + key[:, None] * (D * K) + k[None, :]
        grad_a = a * 0
        for d in tl.static_range(D):
            g = tl.load(rows + d * K, mask=mask, other=0.0)
            y = tl.load(y_ptr + edges * D + d, mask=edge_mask, other=0.0)
            grad_a += g * y[:, None]
            tl.store(grad_y_ptr + edges * D + d, tl.sum(g * a, axis=1), mask=edge_mask)
        tl.store(grad_a_ptr + edges[:, None] * K + k[None, :], grad_a, mask=mask)

    return kernel


class _Launcher:
    """``kernel[grid](...)`` without the launch arguments the old kernel does not take."""

    def __init__(self, kernel, drop):
        self.kernel, self.drop = kernel, drop

    def __getitem__(self, grid):
        launch = self.kernel[grid]
        return lambda *args, **kwargs: launch(*args, **{k: v for k, v in kwargs.items() if k not in self.drop})


def edge_outer_sum_backward():
    """For ``fastmace.kernels.triton.edge_outer_sum:_triton_backward``: D strided stores of grad_y per edge block."""
    return _Launcher(_edge_outer_sum_backward_kernel(), ("BLOCK_D",))


@cache
def _gated_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gated_input_grads(g_ptr, wt_ptr, gated_ptr, gate_ptr, dgated_ptr, dgate_ptr, atoms_n, N, K, g_n, g_m,
                          gate_n, dgate_n, D: tl.constexpr, BLOCK_A: tl.constexpr, BLOCK_N: tl.constexpr,
                          BLOCK_K: tl.constexpr, PRECISION: tl.constexpr):
        pa, pn = tl.program_id(0), tl.program_id(1)
        atom = (pa * BLOCK_A + tl.arange(0, BLOCK_A)).to(tl.int64)
        j = (pn * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        amask, jmask = atom < atoms_n, j < N
        tile = amask[:, None] & jmask[None, :]
        gate = tl.load(gate_ptr + atom[:, None] * gate_n + j[None, :], mask=tile, other=0.0)
        dgate = tl.zeros((BLOCK_A, BLOCK_N), dtype=tl.float32)
        kk = tl.arange(0, BLOCK_K)
        for m in tl.static_range(D):
            acc = tl.zeros((BLOCK_A, BLOCK_N), dtype=tl.float32)
            for start in range(0, K, BLOCK_K):
                k = start + kk
                a = tl.load(g_ptr + (atom * g_n + m * g_m)[:, None] + k[None, :],
                            mask=amask[:, None] & (k < K)[None, :], other=0.0)
                b = tl.load(wt_ptr + k[:, None] * N + j[None, :], mask=(k < K)[:, None] & jmask[None, :], other=0.0)
                acc += tl.dot(a, b, input_precision=PRECISION)
            row = atom * D + m
            gated = tl.load(gated_ptr + row[:, None] * N + j[None, :], mask=tile, other=0.0)
            tl.store(dgated_ptr + row[:, None] * N + j[None, :], acc * gate, mask=tile)
            dgate += acc * gated
        tl.store(dgate_ptr + atom[:, None] * dgate_n + j[None, :], dgate, mask=tile)

    return gated_input_grads


def gated_fused(grad, offset, mul_out, wt, gated, gates, gate_offset, dgated, dgates, dim, precision):
    """For ``fastmace.kernels.triton.gated_products:_fused``: the m-outer, K-inner kernel (OPT10-T01)."""
    import triton

    block_a, block_n, block_k, num_warps, num_stages = 32, 64, 32, 4, 3
    atoms, width = grad.shape
    mul_in = int(wt.shape[1])
    grid = (triton.cdiv(atoms, block_a), triton.cdiv(mul_in, block_n))
    _gated_kernel()[grid](grad.reshape(-1)[offset:], wt, gated, gates.reshape(-1)[gate_offset:],
                          dgated, dgates.reshape(-1)[gate_offset:], atoms, mul_in, mul_out, width, mul_out,
                          gates.stride(0), dgates.stride(0), D=dim, BLOCK_A=block_a, BLOCK_N=block_n,
                          BLOCK_K=block_k, PRECISION=precision, num_warps=num_warps, num_stages=num_stages)


def first_reassociation_joined(self, node_feats, node_attrs, edge_attrs, augmented, edge_index, weight):
    """Prototype for ``fastmace.rewrites.mh1.modules:FirstInteractionReassociation.forward`` (fifth list, D): the
    per-path products in ``segment_gemm``'s ``joined_products`` (one Triton launch per path where the shape rule
    picks Triton, written into the joined output) instead of one matmul per path and a cat. ir_mul layout only;
    otherwise the library forward. Not ``main``'s code: side a of this A/B is the candidate. Imported only after
    the hybrid runtime is prepared (``ab_swap.install``), which captures the library forward first."""
    if not self.ir_mul or self.tf32x3 or not augmented.is_cuda or augmented.dtype != torch.float32:
        return _LIBRARY_FIRST_FORWARD(self, node_feats, node_attrs, edge_attrs, augmented, edge_index, weight)
    onehot = node_attrs[:, self.columns]
    species = onehot.argmax(dim=1)
    x = self.species_feats
    nodes, slots = onehot.shape
    width, dim = augmented.shape[1], edge_attrs.shape[1]
    key = edge_index[1] * slots + species[edge_index[0]]
    summed = edge_outer_sum(augmented, edge_attrs, key, nodes * slots, segmented=self.segmented).view(
        nodes, slots, dim, width)
    rows, weights, offsets, dims = [], [], [], []
    start, offset, c = 0, 0, self.channels
    for path, (l, gamma) in enumerate(zip(self.ls, self.gammas)):  # noqa: E741
        size = 2 * l + 1
        rows.append(summed[:, :, start:start + size].transpose(1, 2).reshape(nodes * size, slots * width))
        weights.append((gamma * x[:, None, :] * weight[path * c:(path + 1) * c].t()[None]).reshape(slots * width, c))
        offsets.append(offset)
        dims.append(size)
        start += size
        offset += size * c
    return joined_products(rows, weights, [w.t().contiguous() for w in weights], offsets, dims, offset)


__all__ = ["edge_outer_sum_backward", "first_reassociation_joined", "gated_fused"]
