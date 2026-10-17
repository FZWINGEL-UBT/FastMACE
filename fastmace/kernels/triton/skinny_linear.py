"""Folded radial last layer with a Triton input gradient for the [h c, c] layout.

folded_linear(a, weight) returns a @ weight.T, like torch.nn.functional.linear
without bias, for a = [h c, c] of width P + 1 (the folded radial cutoff). The
forward stays the ordinary GEMM. The input gradient
    grad_a = grad_out @ weight,   grad_out [E, Q], weight [Q, P + 1],
is a skinny GEMM whose P + 1 = 65 output columns waste half of cuBLAS's
64-wide tiles (NaNbO3 3x3x3, Q = 1280: 2.0 ms, against 1.6 ms for 64
columns). The Triton kernel computes the first P columns with a FP32 (IEEE,
no TF32) block product and the last column as a row sum in the same pass, so
grad_out, the largest per-edge tensor of the step, is read once. Same FP32
arithmetic; only the summation order differs. The CPU uses PyTorch. The
weight gets no gradient (MD forces only); a weight that requires one falls
back to PyTorch.
"""

from functools import cache

import torch

# One fixed configuration for every crystal and GPU (no autotuning).
BLOCK_M, BLOCK_K, NUM_WARPS, NUM_STAGES = 64, 32, 4, 3
MAX_TRITON_COLUMNS = 128  # P above this falls back to PyTorch (the block of P columns lives in registers)


@cache
def _triton_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(grad_ptr, main_ptr, last_ptr, out_ptr, rows, Q, P,
               BLOCK_M: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_K: tl.constexpr):
        m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = m < rows
        m = m.to(tl.int64)
        p = tl.arange(0, BLOCK_P)
        k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_P), dtype=tl.float32)
        last = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for start in range(0, Q, BLOCK_K):
            kk = start + k
            g = tl.load(grad_ptr + m[:, None] * Q + kk[None, :],
                        mask=row_mask[:, None] & (kk < Q)[None, :], other=0.0)
            w = tl.load(main_ptr + kk[:, None] * P + p[None, :],
                        mask=(kk < Q)[:, None] & (p < P)[None, :], other=0.0)
            b = tl.load(last_ptr + kk, mask=kk < Q, other=0.0)
            acc += tl.dot(g, w, input_precision="ieee")
            last += tl.sum(g * b[None, :], axis=1)
        tl.store(out_ptr + m[:, None] * (P + 1) + p[None, :], acc,
                 mask=row_mask[:, None] & (p < P)[None, :])
        tl.store(out_ptr + m * (P + 1) + P, last, mask=row_mask)

    return kernel


@torch.library.custom_op("fastmace::folded_linear_input_grad", mutates_args=())
def folded_linear_input_grad(grad: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """grad @ weight for grad [E, Q] and weight [Q, P + 1]."""
    if not grad.is_cuda or grad.dtype != torch.float32 or weight.shape[1] - 1 > MAX_TRITON_COLUMNS:
        return grad @ weight
    import triton
    grad = grad.contiguous()
    rows, q = grad.shape
    p = weight.shape[1] - 1
    out = torch.empty((rows, p + 1), dtype=grad.dtype, device=grad.device)
    if rows:
        main = weight[:, :p].contiguous()
        last = weight[:, p].contiguous()
        _triton_kernel()[(triton.cdiv(rows, BLOCK_M),)](
            grad, main, last, out, rows, q, p, BLOCK_M=BLOCK_M,
            BLOCK_P=max(16, triton.next_power_of_2(p)), BLOCK_K=BLOCK_K,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    return out


@folded_linear_input_grad.register_fake
def _(grad, weight):
    return grad.new_empty((grad.shape[0], weight.shape[1]))


class _FoldedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, weight, contiguous_weight):
        ctx.save_for_backward(weight)
        if contiguous_weight:
            # a @ (W^T stored contiguously): cuBLAS's non-transposed kernels, 1.86 -> 1.52 ms for
            # [E, 65] @ [65, 1280] on NaNbO3 shapes (the contiguous folded weight screen).
            return a @ weight.t().contiguous()
        return a @ weight.t()

    @staticmethod
    def backward(ctx, grad):
        (weight,) = ctx.saved_tensors
        return folded_linear_input_grad(grad, weight), None, None


def folded_linear(a, weight, contiguous_weight=False):
    """a @ weight.T with the fused input gradient; the weight must not require a gradient.

    contiguous_weight=True multiplies by a contiguous copy of weight.T (same products).
    """
    if weight.requires_grad:
        return torch.nn.functional.linear(a, weight)
    return _FoldedLinear.apply(a, weight, bool(contiguous_weight))
