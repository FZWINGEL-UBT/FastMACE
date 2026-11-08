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

``RADIAL_TF32X3`` (an A/B constant, off-polar-alex 6 November 2026; ``radial_tf32x3_default``): the forward and
the input gradient as 3xTF32 products (``tf32x3_gemm.tf32x3_mm``, CUTLASS on the A100), K and N padded with zeros
to the kernels' alignment. MACE-OFF23-large's [E, 64] @ [64, 3808] and its [E, 3808] @ [3808, 64] gradient run
compute-bound on SIMT FP32 (1.0 and 1.3 ms of a 12.1 ms force call on the A100-SXM4-80GB). Rounding. The GPU
list applies only where the caller marks the product eligible (``folded_linear(..., tf32x3=True)``): MACE-OFF's
radial networks (``generic.apply_fused_radial_backward``, the stock first block's table), the only models measured.

``SPLIT_COLUMNS`` (an A/B constant, Codex R1, off-polar-alex 7 November 2026; ``split_columns_default``): the input
gradient with a second grid axis over ``SPLIT_BLOCK_P``-column tiles of P where the one-axis grid leaves SMs idle
and the split grid still fits in one wave (``split_columns``): MACE-OFF23-large's 512-row chunks of the streamed
backward launch 8 CTAs for a 3808-long reduction. Tile 0 also computes the last column. Every output keeps its
BLOCK_K reduction order, so the results are bitwise the one-axis grid's.
"""

from functools import cache

import torch

from fastmace.kernels.gpu_models import GPUModels, dynamo_constant

# One fixed configuration for every crystal and GPU (no autotuning).
BLOCK_M, BLOCK_K, NUM_WARPS, NUM_STAGES = 64, 32, 4, 3
MAX_TRITON_COLUMNS = 128  # P above this falls back to PyTorch (the block of P columns lives in registers)
TILE_RULES = ("fixed", "waves")
# RADIAL_TF32X3: None follows RADIAL_TF32X3_GPUS for the eligible products (MACE-OFF's radial networks; decision 66);
# True / False: an A/B side's choice for every folded product. A100-SXM4-80GB (off-polar-alex, 7 November 2026,
# results/off-polar-alex-20261106): MD step b/a (b = on), water_216, abab / abba: OFF23-large 0.919 / 0.919; with
# the OFF tables and the stock first skip, all five OFF models 0.56-0.65. Force gates pass under the 3x rule.
RADIAL_TF32X3 = None
RADIAL_TF32X3_GPUS = GPUModels(("NVIDIA A100-SXM4-80GB",))
_ALIGN = 4  # cutlass.tf32x3.ALIGNMENT: K and N padded to it
# SPLIT_COLUMNS: None follows SPLIT_COLUMNS_GPUS; True / False: an A/B side's choice. RTX 5060 Ti (36 SMs), MD step
# b/a, water_216, abab / abba: MACE-OFF23-large 0.951 / 0.953 (16-column tiles; 32-column tiles 1.07 / 1.06);
# the kernel alone on [512, 3808] @ [3808, 65] 0.88x, on 896 rows (the split grid over one wave) 1.07x. With the
# one-wave rule, 8 November: OFF23-large 0.950 / 0.954; OFF23-small, -medium, OFF23b and OFF24 0.993-1.006 (their
# chunks are over 576 rows, so the split never fires). Exact (tests/gpu/test_folded_tiles_cuda.py, bitwise), so on
# by decision 66 on the 5060 Ti. A100-SXM4-80GB (Alex 4961878; OFF23-large's 640-row chunks split on 108 SMs):
# OFF23-large 0.999 / 0.998, OFF23-small 0.993 / 1.001, -medium 0.999 / 1.005, OFF23b 1.000 / 1.004, OFF24
# 1.002 / 0.998: a tie, so measured and left off (it changes the tiling and launch grid, so decision 66 needs
# a measured speed-up in both orders, not the pure-removal tie; conductor 8 November). The Ada is not measured.
SPLIT_COLUMNS = None
SPLIT_COLUMNS_GPUS = GPUModels(("NVIDIA GeForce RTX 5060 Ti",))
SPLIT_BLOCK_P = 16


@dynamo_constant
def _tf32x3_listed(index: int) -> bool:
    """Whether CUDA device ``index`` is one of ``RADIAL_TF32X3_GPUS`` (a Dynamo constant, as
    ``receiver_sum._listed``)."""
    return torch.cuda.get_device_name(index) in RADIAL_TF32X3_GPUS


def radial_tf32x3_default(device, eligible: bool = False) -> bool:
    """``RADIAL_TF32X3`` on ``device``: the constant when set, else on for an ``eligible`` product on
    ``RADIAL_TF32X3_GPUS``; never off CUDA."""
    device = torch.device(device)
    if device.type != "cuda":
        return False
    if RADIAL_TF32X3 is not None:
        return bool(RADIAL_TF32X3)
    return bool(eligible) and _tf32x3_listed(device.index if device.index is not None else torch.cuda.current_device())


@dynamo_constant
def _split_listed(index: int) -> bool:
    """Whether CUDA device ``index`` is one of ``SPLIT_COLUMNS_GPUS`` (a Dynamo constant)."""
    return torch.cuda.get_device_name(index) in SPLIT_COLUMNS_GPUS


def split_columns_default(device) -> bool:
    """``SPLIT_COLUMNS`` on ``device``: the constant when set, else ``SPLIT_COLUMNS_GPUS``; never off CUDA."""
    device = torch.device(device)
    if device.type != "cuda":
        return False
    if SPLIT_COLUMNS is not None:
        return bool(SPLIT_COLUMNS)
    return _split_listed(device.index if device.index is not None else torch.cuda.current_device())


def split_columns(rows: int, p: int, sms: int | None, block_m: int) -> int:
    """The column tile of the split grid (``SPLIT_BLOCK_P``) when the one-axis grid of ``rows`` rows in
    ``block_m``-row tiles, split into ``SPLIT_BLOCK_P``-column tiles of P, still fits in one wave over ``sms``
    SMs and P has more than one tile; else 0 (the one-axis grid)."""
    if not sms or rows <= 0 or p <= SPLIT_BLOCK_P:
        return 0
    tiles = -(-int(rows) // block_m) * -(-int(p) // SPLIT_BLOCK_P)
    return SPLIT_BLOCK_P if tiles <= sms else 0


def _padded(t, rows=0, columns=0):
    return torch.nn.functional.pad(t, (0, columns, 0, rows)) if rows or columns else t


def tf32x3_forward(a, weight):
    """``a @ weight.T`` (a [E, K], weight [Q, K]) as one 3xTF32 product, K padded to the alignment with zeros."""
    from fastmace.kernels.triton.tf32x3_gemm import tf32x3_mm

    pad = -a.shape[1] % _ALIGN
    return tf32x3_mm(_padded(a, columns=pad), _padded(weight, columns=pad).t())
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


@cache
def _split_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(grad_ptr, main_ptr, last_ptr, out_ptr, rows, Q, P,
               BLOCK_M: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_K: tl.constexpr):
        m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        tile = tl.program_id(1)
        row_mask = m < rows
        m = m.to(tl.int64)
        p = tile * BLOCK_P + tl.arange(0, BLOCK_P)
        k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_P), dtype=tl.float32)
        last = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for start in range(0, Q, BLOCK_K):
            kk = start + k
            g = tl.load(grad_ptr + m[:, None] * Q + kk[None, :],
                        mask=row_mask[:, None] & (kk < Q)[None, :], other=0.0)
            w = tl.load(main_ptr + kk[:, None] * P + p[None, :],
                        mask=(kk < Q)[:, None] & (p < P)[None, :], other=0.0)
            acc += tl.dot(g, w, input_precision="ieee")
            if tile == 0:  # the last column once, in the one-axis kernel's order
                b = tl.load(last_ptr + kk, mask=kk < Q, other=0.0)
                last += tl.sum(g * b[None, :], axis=1)
        tl.store(out_ptr + m[:, None] * (P + 1) + p[None, :], acc,
                 mask=row_mask[:, None] & (p < P)[None, :])
        if tile == 0:
            tl.store(out_ptr + m * (P + 1) + P, last, mask=row_mask)

    return kernel


def _input_grad(grad, main, last, tile_rule, out=None):
    """The Triton input gradient [E, P + 1] from the contiguous halves ``main`` [Q, P] and ``last`` [Q]; into
    ``out`` (contiguous rows, such as a row slice) when given (``chunked_conv_backward``, opt-6 OPT6-T02)."""
    import triton

    from fastmace.kernels.triton import tile_rule as rules
    grad = grad.contiguous()
    rows, q = grad.shape
    p = main.shape[1]
    if out is None:
        out = torch.empty((rows, p + 1), dtype=grad.dtype, device=grad.device)
    if rows:
        sms = rules.device_sms(grad.device)
        block_m, block_k, warps, stages = tile_choice(rows, sms, tile_rule)
        split = split_columns(rows, p, sms, block_m) if split_columns_default(grad.device) else 0
        if split:
            _split_kernel()[(triton.cdiv(rows, block_m), triton.cdiv(p, split))](
                grad, main, last, out, rows, q, p, BLOCK_M=block_m, BLOCK_P=split, BLOCK_K=block_k,
                num_warps=warps, num_stages=stages)
            return out
        _triton_kernel()[(triton.cdiv(rows, block_m),)](
            grad, main, last, out, rows, q, p, BLOCK_M=block_m,
            BLOCK_P=max(16, triton.next_power_of_2(p)), BLOCK_K=block_k,
            num_warps=warps, num_stages=stages)
    return out


def input_grad_into(grad, weight, main, last, out, tile_rule="fixed"):
    """``out[:] = grad @ weight`` for grad [rows, Q] and weight [Q, P + 1] (its halves ``main``, ``last``): the
    Triton kernel where it applies, PyTorch otherwise; ``out`` is a contiguous [rows, P + 1] buffer."""
    if _uses_triton(grad, weight):
        _input_grad(grad, main, last, tile_rule, out)
    else:
        torch.mm(grad, weight, out=out)
    return out


def _uses_triton(grad, weight):
    return grad.is_cuda and grad.dtype == torch.float32 and weight.shape[1] - 1 <= MAX_TRITON_COLUMNS


@torch.library.custom_op("fastmace::folded_linear_input_grad", mutates_args=())
def folded_linear_input_grad(grad: torch.Tensor, weight: torch.Tensor, tile_rule: str = "fixed") -> torch.Tensor:
    """grad @ weight for grad [E, Q] and weight [Q, P + 1]; ``tile_rule`` picks the tiles (``tile_choice``)."""
    if not _uses_triton(grad, weight):
        return grad @ weight
    p = weight.shape[1] - 1
    return _input_grad(grad, weight[:, :p].contiguous(), weight[:, p].contiguous(), tile_rule)


@torch.library.custom_op("fastmace::folded_linear_input_grad_parts", mutates_args=())
def folded_linear_input_grad_parts(grad: torch.Tensor, weight: torch.Tensor, main: torch.Tensor, last: torch.Tensor,
                                   tile_rule: str = "fixed") -> torch.Tensor:
    """``folded_linear_input_grad`` with the contiguous halves of ``weight`` given (``folded_constants``,
    OPT3-T05): the same kernel on the same values, without slicing the weight on every call."""
    if not _uses_triton(grad, weight):
        return grad @ weight
    return _input_grad(grad, main, last, tile_rule)


@torch.library.custom_op("fastmace::folded_linear_input_grad_tf32x3", mutates_args=())
def folded_linear_input_grad_tf32x3(grad: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``grad @ weight`` (grad [E, Q], weight [Q, P + 1]) as one 3xTF32 product (``RADIAL_TF32X3``): weight's
    columns padded to the alignment with zeros, the padding cut off."""
    from fastmace.kernels.triton.tf32x3_gemm import tf32x3_mm

    columns = weight.shape[1]
    out = tf32x3_mm(grad.contiguous(), _padded(weight, columns=-columns % _ALIGN).contiguous())
    return out if out.shape[1] == columns else out[:, :columns].contiguous()


@folded_linear_input_grad_tf32x3.register_fake
def _(grad, weight):
    return grad.new_empty((grad.shape[0], weight.shape[1]))


@folded_linear_input_grad_parts.register_fake
def _(grad, weight, main, last, tile_rule="fixed"):
    return grad.new_empty((grad.shape[0], weight.shape[1]))


@folded_linear_input_grad.register_fake
def _(grad, weight, tile_rule="fixed"):
    return grad.new_empty((grad.shape[0], weight.shape[1]))


class _FoldedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, weight, contiguous_weight, tile_rule, main=None, last=None, eligible=False):
        ctx.save_for_backward(weight, main, last)
        ctx.tile_rule = tile_rule
        ctx.tf32x3 = a.dtype == torch.float32 and radial_tf32x3_default(a.device, eligible)
        if ctx.tf32x3:
            return tf32x3_forward(a, weight)
        if contiguous_weight:
            # a @ (W^T stored contiguously): cuBLAS's non-transposed kernels, 1.86 -> 1.52 ms for
            # [E, 65] @ [65, 1280] on NaNbO3 shapes (the contiguous folded weight screen).
            return a @ weight.t().contiguous()
        return a @ weight.t()

    @staticmethod
    def backward(ctx, grad):
        weight, main, last = ctx.saved_tensors
        if ctx.tf32x3:
            return folded_linear_input_grad_tf32x3(grad, weight), None, None, None, None, None, None
        if main is not None:
            return (folded_linear_input_grad_parts(grad, weight, main, last, ctx.tile_rule),
                    None, None, None, None, None, None)
        return folded_linear_input_grad(grad, weight, ctx.tile_rule), None, None, None, None, None, None


def folded_linear(a, weight, contiguous_weight=False, tile_rule="fixed", parts=None, tf32x3=False):
    """a @ weight.T with the fused input gradient; the weight must not require a gradient.

    contiguous_weight=True multiplies by a contiguous copy of weight.T (same products); ``tile_rule``
    ("fixed" or "waves", ``tile_choice``) picks the input gradient's tiles, with the same results; ``parts``
    (``folded_constants``) are weight's contiguous halves (W [Q, P], b [Q]) for the input gradient, so that it
    does not slice them out of weight on every call. ``tf32x3`` marks the product eligible for
    ``RADIAL_TF32X3_GPUS`` (``radial_tf32x3_default``).
    """
    if tile_rule not in TILE_RULES:
        raise ValueError(f"tile_rule must be one of {TILE_RULES}, got {tile_rule!r}")
    if weight.requires_grad:
        return torch.nn.functional.linear(a, weight)
    main, last = parts if parts is not None else (None, None)
    return _FoldedLinear.apply(a, weight, bool(contiguous_weight), tile_rule, main, last, bool(tf32x3))
