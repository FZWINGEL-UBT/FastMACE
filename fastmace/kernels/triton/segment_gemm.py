"""Block-linear products written straight into their joined output (plan opt-1, OPT-T02 stage B).

A block linear multiplies each irrep segment's rows ``[atoms (2l + 1), mul_in]`` by its weight and then
joins the results into one ``[atoms, width]`` tensor in the ir_mul layout, where component m of channel u
of a segment at ``offset`` sits at ``offset + m * mul + u``. The join is a copy of the whole output. Its
backward slices the gradient into rows again, which is another copy.

``joined_products(rows, weights, weights_t, offsets, dims, width)`` does both without the copies, for
products whose shape rule (``tile_rule.segment_gemm_choice``) picks Triton:

- **forward:** a gathered-row Triton GEMM stores the rows of ``rows[i] @ weights[i]`` at their place in the
  joined output, row ``(atom, m)`` at ``atom * width + offset_i + m * mul_out_i``;
- **backward:** the same kernel reads the gradient's rows of segment i in place and multiplies them by the
  contiguous ``weights_t[i]``.

Products the rule leaves to cuBLAS keep today's path (cuBLAS, then a copy into the output). On the CPU and
for other dtypes everything runs in PyTorch with the block linears' own operations. The Triton GEMM
accumulates in IEEE float32 (``tl.dot(input_precision="ieee")``) over K in a fixed order, with no atomics:
deterministic, rounding-level against cuBLAS. The weights get no gradient (MD forces only).
"""

from __future__ import annotations

from functools import cache
from typing import List

import torch

from fastmace.kernels.triton import tile_rule


@cache
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gemm(a_ptr, b_ptr, c_ptr, rows, N, K, D, a_n, a_m, b_k, b_n, c_n, c_m,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PRECISION: tl.constexpr):
        # Row r of A and C is (atom r // D, component r % D); A's and C's rows are gathered and scattered.
        pm, pn = tl.program_id(0), tl.program_id(1)
        r = (pm * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        j = (pn * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        atom, comp = r // D, r % D
        a_row = atom * a_n + comp * a_m
        kk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for start in range(0, K, BLOCK_K):
            k = start + kk
            a = tl.load(a_ptr + a_row[:, None] + k[None, :], mask=(r < rows)[:, None] & (k < K)[None, :], other=0.0)
            b = tl.load(b_ptr + k[:, None] * b_k + j[None, :] * b_n, mask=(k < K)[:, None] & (j < N)[None, :],
                        other=0.0)
            acc += tl.dot(a, b, input_precision=PRECISION)
        tl.store(c_ptr + (atom * c_n + comp * c_m)[:, None] + j[None, :], acc,
                 mask=(r < rows)[:, None] & (j < N)[None, :])

    return gemm


def _gemm(a, a_offset, a_n, a_m, b, c, c_offset, c_n, c_m, rows, dim, choice, precision="ieee"):
    """C rows (atom, m) <- A rows (atom, m) @ B, A and C addressed from their flat storage."""
    import triton

    k, n = b.shape
    grid = (triton.cdiv(rows, choice.block_m), triton.cdiv(n, choice.block_n))
    _kernel()[grid](a.reshape(-1)[a_offset:], b, c.reshape(-1)[c_offset:], rows, n, k, dim, a_n, a_m,
                    b.stride(0), b.stride(1), c_n, c_m, BLOCK_M=choice.block_m, BLOCK_N=choice.block_n,
                    BLOCK_K=choice.block_k, PRECISION=precision, num_warps=choice.num_warps,
                    num_stages=choice.num_stages)


def _choices(rows_per_product, weights, device, dtype, precision="ieee"):
    sms = tile_rule.device_sms(device) if dtype == torch.float32 else None
    if precision != "ieee" and sms:  # tf32x3_gemms: 3xTF32 where K, N >= BLOCK_K, SIMT IEEE below (OPT4-T03, OPT5-T05)
        return [
            tile_rule.GemmChoice("triton", reason=f"{precision}: K, N >= {tile_rule.BLOCK_K}")
            if int(w.shape[0]) >= tile_rule.BLOCK_K and int(w.shape[1]) >= tile_rule.BLOCK_K
            else tile_rule.GemmChoice("cublas", reason=f"{precision}: sub-tile K={int(w.shape[0])}, "
                                                       f"N={int(w.shape[1])} kept on SIMT IEEE")
            for w in weights
        ]
    return [tile_rule.segment_gemm_choice(rows, int(w.shape[1]), int(w.shape[0]), sms)
            for rows, w in zip(rows_per_product, weights)]


@torch.library.custom_op("fastmace::joined_products", mutates_args=())
def joined_products_op(rows: List[torch.Tensor], weights: List[torch.Tensor], offsets: List[int], dims: List[int],
                       width: int, precision: str) -> torch.Tensor:
    """``[atoms, width]``: ``rows[i] @ weights[i]`` (rows ``[atoms * dims[i], K_i]``) at column ``offsets[i]``;
    columns no product reaches are zero. ``precision`` is the Triton GEMM's ``tl.dot`` input precision: "ieee",
    or "tf32x3" (``tf32x3_gemms``), which runs every product on CUDA in Triton."""
    atoms = rows[0].shape[0] // dims[0]
    covered = sum(dim * int(w.shape[1]) for dim, w in zip(dims, weights))
    like = rows[0]
    out = like.new_zeros((atoms, width)) if covered != width else like.new_empty((atoms, width))
    choices = _choices([atoms * d for d in dims], weights, like.device, like.dtype, precision)
    for x, w, offset, dim, choice in zip(rows, weights, offsets, dims, choices):
        mul = int(w.shape[1])
        if choice.backend == "triton" and atoms:
            x = x.contiguous()
            _gemm(x, 0, dim * x.shape[1], x.shape[1], w, out, offset, width, mul, atoms * dim, dim, choice, precision)
        else:
            out[:, offset:offset + dim * mul] = (x @ w).reshape(atoms, dim * mul)
    return out


@joined_products_op.register_fake
def _(rows, weights, offsets, dims, width, precision):
    return rows[0].new_empty((rows[0].shape[0] // dims[0], width))


@torch.library.custom_op("fastmace::joined_products_input_grads", mutates_args=())
def joined_products_input_grads(grad: torch.Tensor, weights_t: List[torch.Tensor], offsets: List[int],
                                dims: List[int], precision: str) -> List[torch.Tensor]:
    """Per product i: ``grad``'s rows of its segment ``@ weights_t[i]``, as ``[atoms * dims[i], K_i]``."""
    atoms, width = grad.shape
    choices = _choices([atoms * d for d in dims], [w.t() for w in weights_t], grad.device, grad.dtype, precision)
    out = []
    for wt, offset, dim, choice in zip(weights_t, offsets, dims, choices):
        mul_out, mul_in = int(wt.shape[0]), int(wt.shape[1])
        if choice.backend == "triton" and atoms:
            g = grad.contiguous()
            dx = grad.new_empty((atoms * dim, mul_in))
            _gemm(g, offset, width, mul_out, wt, dx, 0, dim * mul_in, mul_in, atoms * dim, dim, choice, precision)
        else:
            dx = grad[:, offset:offset + dim * mul_out].reshape(atoms * dim, mul_out) @ wt
        out.append(dx)
    return out


@joined_products_input_grads.register_fake
def _(grad, weights_t, offsets, dims, precision):
    atoms = grad.shape[0]
    return [grad.new_empty((atoms * dim, int(wt.shape[1]))) for wt, dim in zip(weights_t, dims)]


class _JoinedProducts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, offsets, dims, width, precision, count, *tensors):
        rows, weights, weights_t = tensors[:count], tensors[count:2 * count], tensors[2 * count:]
        ctx.offsets, ctx.dims, ctx.count, ctx.precision = offsets, dims, count, precision
        ctx.save_for_backward(*weights_t)
        return joined_products_op(list(rows), list(weights), offsets, dims, width, precision)

    @staticmethod
    def backward(ctx, grad):
        weights_t = ctx.saved_tensors
        grads = joined_products_input_grads(grad, list(weights_t), ctx.offsets, ctx.dims, ctx.precision)
        return (None, None, None, None, None, *grads, *([None] * (2 * ctx.count)))


def joined_products(rows, weights, weights_t, offsets, dims, width, precision="ieee"):
    """The joined ``[atoms, width]`` output of the products ``rows[i] @ weights[i]``; see the module docstring.

    ``weights_t[i]`` is the contiguous transpose of ``weights[i]`` (the input gradient's operand). Neither
    weight may require a gradient. ``precision``: see ``joined_products_op``."""
    count = len(rows)
    return _JoinedProducts.apply(list(offsets), list(dims), int(width), str(precision), count, *rows, *weights,
                                 *weights_t)


__all__ = ["joined_products", "joined_products_input_grads", "joined_products_op"]
