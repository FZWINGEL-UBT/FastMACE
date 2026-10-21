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

Tiles (opt-2 OPT2-T03): ``tile_rule="fixed"`` launches the one configuration
below; ``tile_rule="waves"`` (the opt-in record ``folded_backward_tiles``)
picks 128-row tiles with 8 warps where the 64-row grid is a few waves over
the SMs (small cells: 2 to 8 waves), the fixed tiles elsewhere. BLOCK_K stays
32 in both, so every row is reduced in the same order: the results are bitwise
the same (``tile_choice``).
"""

from functools import cache

import torch

# One fixed configuration for every crystal and GPU (no autotuning).
BLOCK_M, BLOCK_K, NUM_WARPS, NUM_STAGES = 64, 32, 4, 3
MAX_TRITON_COLUMNS = 128  # P above this falls back to PyTorch (the block of P columns lives in registers)
TILE_RULES = ("fixed", "waves")
# The waves rule (OPT2-T03): 128-row tiles, 8 warps, where the 64-row grid has WAVES_MIN to WAVES_MAX waves
# over the SMs. Measured on the RTX 5060 Ti (36 SMs), kernel alone: 0.71x at 3.6 waves (NaCl 3x3x3), 0.90x at
# 7.1, 1.01x at 1.8 and 1.03x at 14; at 0.9 waves 1.67x. Same BLOCK_K, so the same reduction order.
WAVES_TILES = (128, BLOCK_K, 8, NUM_STAGES)
WAVES_MIN, WAVES_MAX = 2, 8


def tile_choice(rows: int, sms: int | None, rule: str = "fixed") -> tuple[int, int, int, int]:
    """(BLOCK_M, BLOCK_K, warps, stages) of the input-gradient kernel for ``rows`` rows on ``sms`` SMs."""
    if rule not in TILE_RULES:
        raise ValueError(f"tile_rule must be one of {TILE_RULES}, got {rule!r}")
    if rule == "waves" and sms:
        tiles = -(-int(rows) // BLOCK_M)
        if WAVES_MIN * sms <= tiles <= WAVES_MAX * sms:
            return WAVES_TILES
    return BLOCK_M, BLOCK_K, NUM_WARPS, NUM_STAGES


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
def folded_linear_input_grad(grad: torch.Tensor, weight: torch.Tensor, tile_rule: str = "fixed") -> torch.Tensor:
    """grad @ weight for grad [E, Q] and weight [Q, P + 1]; ``tile_rule`` picks the tiles (``tile_choice``)."""
    if not grad.is_cuda or grad.dtype != torch.float32 or weight.shape[1] - 1 > MAX_TRITON_COLUMNS:
        return grad @ weight
    import triton

    from fastmace.kernels.triton import tile_rule as rules
    grad = grad.contiguous()
    rows, q = grad.shape
    p = weight.shape[1] - 1
    out = torch.empty((rows, p + 1), dtype=grad.dtype, device=grad.device)
    if rows:
        main = weight[:, :p].contiguous()
        last = weight[:, p].contiguous()
        block_m, block_k, warps, stages = tile_choice(rows, rules.device_sms(grad.device), tile_rule)
        _triton_kernel()[(triton.cdiv(rows, block_m),)](
            grad, main, last, out, rows, q, p, BLOCK_M=block_m,
            BLOCK_P=max(16, triton.next_power_of_2(p)), BLOCK_K=block_k,
            num_warps=warps, num_stages=stages)
    return out


@folded_linear_input_grad.register_fake
def _(grad, weight, tile_rule="fixed"):
    return grad.new_empty((grad.shape[0], weight.shape[1]))


class _FoldedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, weight, contiguous_weight, tile_rule):
        ctx.save_for_backward(weight)
        ctx.tile_rule = tile_rule
        if contiguous_weight:
            # a @ (W^T stored contiguously): cuBLAS's non-transposed kernels, 1.86 -> 1.52 ms for
            # [E, 65] @ [65, 1280] on NaNbO3 shapes (the contiguous folded weight screen).
            return a @ weight.t().contiguous()
        return a @ weight.t()

    @staticmethod
    def backward(ctx, grad):
        (weight,) = ctx.saved_tensors
        return folded_linear_input_grad(grad, weight, ctx.tile_rule), None, None, None


def folded_linear(a, weight, contiguous_weight=False, tile_rule="fixed"):
    """a @ weight.T with the fused input gradient; the weight must not require a gradient.

    contiguous_weight=True multiplies by a contiguous copy of weight.T (same products); ``tile_rule``
    ("fixed" or "waves", ``tile_choice``) picks the input gradient's tiles, with the same results.
    """
    if tile_rule not in TILE_RULES:
        raise ValueError(f"tile_rule must be one of {TILE_RULES}, got {tile_rule!r}")
    if weight.requires_grad:
        return torch.nn.functional.linear(a, weight)
    return _FoldedLinear.apply(a, weight, bool(contiguous_weight), tile_rule)
