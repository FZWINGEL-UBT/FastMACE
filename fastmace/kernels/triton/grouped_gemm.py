"""A block linear's small per-l products in one launch (plan opt-1, OPT-T03 candidate 2).

Design: ``docs/design/opt-T03-small-m.md``.

A block linear multiplies each irrep segment's rows ``[atoms (2l + 1), K]`` by its weight. On small cells each
of those products leaves most of the GPU idle (NaCl 3x3x3: 0.8–2.7 waves). ``grouped_products(lefts, weights,
weights_t)`` runs up to ``MAX_GROUPS`` of them in one Triton launch, a tile scheduler over the groups, and the
input gradients the same way with the contiguous transposes. The products are unchanged: IEEE ``tl.dot``
over K in a fixed order, no atomics, so the result is rounding-level against cuBLAS and deterministic.

The groups' operands stay where they are. The kernel reads every group's A and weight through its offset from
the first group's tensor (in elements), passed as scalar arguments, so no pointer table is built at run time
and a CUDA graph replays the launch as captured. The outputs are one flat buffer, which the wrapper cuts into
the groups' ``[M_i, N_i]`` views. On the CPU and for other dtypes the products run in PyTorch.
"""

from __future__ import annotations

from functools import cache
from typing import List

import torch

from fastmace.kernels.triton import tile_rule

MAX_GROUPS = 4
_BIG = 2 ** 62  # tile start of an unused group: no program reaches it


@cache
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["ao1", "ao2", "ao3", "wo1", "wo2", "wo3", "co1", "co2", "co3",
                                   "t1", "t2", "t3"])
    def gemm(a_ptr, w_ptr, c_ptr, ao1, ao2, ao3, wo1, wo2, wo3, co1, co2, co3, t1, t2, t3,
             lda0, lda1, lda2, lda3, m0, m1, m2, m3, k0, k1, k2, k3, n0, n1, n2, n3,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        group = (pid >= t1).to(tl.int32) + (pid >= t2).to(tl.int32) + (pid >= t3).to(tl.int32)
        first, second, third = group == 0, group == 1, group == 2
        ao = tl.where(first, 0, tl.where(second, ao1, tl.where(third, ao2, ao3)))
        wo = tl.where(first, 0, tl.where(second, wo1, tl.where(third, wo2, wo3)))
        co = tl.where(first, 0, tl.where(second, co1, tl.where(third, co2, co3)))
        ts = tl.where(first, 0, tl.where(second, t1, tl.where(third, t2, t3)))
        lda = tl.where(first, lda0, tl.where(second, lda1, tl.where(third, lda2, lda3)))
        M = tl.where(first, m0, tl.where(second, m1, tl.where(third, m2, m3)))
        K = tl.where(first, k0, tl.where(second, k1, tl.where(third, k2, k3)))
        N = tl.where(first, n0, tl.where(second, n1, tl.where(third, n2, n3)))
        local = pid - ts
        tiles_n = (N + BLOCK_N - 1) // BLOCK_N
        tm, tn = local // tiles_n, local % tiles_n
        r = (tm * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        j = (tn * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        kk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        a_base, w_base = a_ptr + ao, w_ptr + wo
        for start in range(0, K, BLOCK_K):
            k = start + kk
            a = tl.load(a_base + r[:, None] * lda + k[None, :], mask=(r < M)[:, None] & (k < K)[None, :], other=0.0)
            w = tl.load(w_base + k[:, None] * N + j[None, :], mask=(k < K)[:, None] & (j < N)[None, :], other=0.0)
            acc += tl.dot(a, w, input_precision="ieee")
        tl.store(c_ptr + co + r[:, None] * N + j[None, :], acc, mask=(r < M)[:, None] & (j < N)[None, :])

    return gemm


def _offset(tensor, base):
    diff = tensor.data_ptr() - base.data_ptr()
    if diff % tensor.element_size():
        raise ValueError("grouped_products: operands must be aligned to their element size")
    return diff // tensor.element_size()


def _launch(lefts, weights, out, out_offsets):
    """One launch for ``lefts[i] @ weights[i]`` written at ``out_offsets[i]`` of the flat ``out``."""
    import triton

    bm, bn, bk = tile_rule.BLOCK_M, tile_rule.BLOCK_N, tile_rule.BLOCK_K
    count = len(lefts)
    lefts = [x if x.stride(1) == 1 else x.contiguous() for x in lefts]
    tiles, starts = 0, []
    for x, w in zip(lefts, weights):
        starts.append(tiles)
        tiles += triton.cdiv(x.shape[0], bm) * triton.cdiv(w.shape[1], bn)
    pad = MAX_GROUPS - count
    a0, w0 = lefts[0], weights[0]
    ao = [_offset(x, a0) for x in lefts[1:]] + [0] * pad
    wo = [_offset(w, w0) for w in weights[1:]] + [0] * pad
    co = [int(o) for o in out_offsets[1:]] + [0] * pad
    ts = starts[1:] + [_BIG] * pad
    lda = [x.stride(0) for x in lefts] + [0] * pad
    ms = [x.shape[0] for x in lefts] + [0] * pad
    ks = [x.shape[1] for x in lefts] + [0] * pad
    ns = [w.shape[1] for w in weights] + [1] * pad
    _kernel()[(tiles,)](a0, w0, out[out_offsets[0]:], *ao, *wo, *co, *ts, *lda, *ms, *ks, *ns,
                        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=tile_rule.NUM_WARPS,
                        num_stages=tile_rule.NUM_STAGES)


def _products(lefts, weights):
    sizes = [int(x.shape[0]) * int(w.shape[1]) for x, w in zip(lefts, weights)]
    like = lefts[0]
    out = like.new_empty((sum(sizes),))
    offsets, total = [], 0
    for size in sizes:
        offsets.append(total)
        total += size
    if like.is_cuda and like.dtype == torch.float32:
        for start in range(0, len(lefts), MAX_GROUPS):
            stop = start + MAX_GROUPS
            base = offsets[start]
            _launch(lefts[start:stop], weights[start:stop], out[base:],
                    [o - base for o in offsets[start:stop]])
    else:
        for x, w, offset, size in zip(lefts, weights, offsets, sizes):
            out[offset:offset + size] = (x @ w).reshape(-1)
    return out


@torch.library.custom_op("fastmace::grouped_products", mutates_args=())
def grouped_products_op(lefts: List[torch.Tensor], weights: List[torch.Tensor]) -> torch.Tensor:
    """The flat concatenation of ``lefts[i] @ weights[i]`` (each ``[M_i, N_i]``, row-major)."""
    return _products(lefts, weights)


@grouped_products_op.register_fake
def _(lefts, weights):
    return lefts[0].new_empty((sum(x.shape[0] * w.shape[1] for x, w in zip(lefts, weights)),))


class _GroupedProducts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, count, *tensors):
        lefts, weights, weights_t = tensors[:count], tensors[count:2 * count], tensors[2 * count:]
        ctx.count = count
        ctx.shapes = [(x.shape[0], w.shape[1]) for x, w in zip(lefts, weights)]
        ctx.save_for_backward(*weights_t)
        return grouped_products_op(list(lefts), list(weights))

    @staticmethod
    def backward(ctx, grad):
        weights_t = ctx.saved_tensors
        grads, offset = [], 0
        for m, n in ctx.shapes:
            grads.append(grad.narrow(0, offset, m * n).view(m, n))
            offset += m * n
        flat = grouped_products_op(grads, list(weights_t))
        out, offset = [], 0
        for (m, _), wt in zip(ctx.shapes, weights_t):
            size = m * wt.shape[1]
            out.append(flat.narrow(0, offset, size).view(m, wt.shape[1]))
            offset += size
        return (None, *out, *([None] * (2 * ctx.count)))


def grouped_products(lefts, weights, weights_t):
    """``[lefts[i] @ weights[i]]`` in one launch per ``MAX_GROUPS`` products; ``weights_t[i]`` is the
    contiguous transpose of ``weights[i]`` (the input gradient's operand). The weights get no gradient."""
    count = len(lefts)
    flat = _GroupedProducts.apply(count, *lefts, *weights, *weights_t)
    out, offset = [], 0
    for x, w in zip(lefts, weights):
        size = x.shape[0] * w.shape[1]
        out.append(flat.narrow(0, offset, size).view(x.shape[0], w.shape[1]))
        offset += size
    return out


__all__ = ["MAX_GROUPS", "grouped_products", "grouped_products_op"]
