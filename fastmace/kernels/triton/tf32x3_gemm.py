"""3xTF32 atom GEMMs through Triton's ``tf32x3`` (plan opt-4, OPT4-T03 stage 1: accuracy first).

``tf32x3_matmul(a, b)`` is ``a @ b`` with every product split into three TF32 tensor-core products
(big * big + big * small + small * big, Triton's ``tl.dot(input_precision="tf32x3")``), set per kernel. No
global TF32 flag is touched: cuBLAS stays IEEE everywhere else. The input gradients run through the same
kernel, ``grad @ b.T`` (with ``b_t``, the contiguous transpose, where the caller keeps one) and ``a.T @ grad``.

The split is FP32-class on typical operands but not IEEE-equivalent under cancellation: the dropped
small * small term costs a few float32 roundoffs of the sum of |a b| per output
(``docs/validation/opt-T01-tensor-cores.md`` §6-§7). Stage 1 measures whether that matters for forces
(``docs/validation/opt4-T03.md``); speed does not count here.

One fixed tile configuration (``tile_rule``'s), no autotuning (SPEC §9.1). K is reduced in a fixed order with
no atomics, so the result is deterministic. Triton's ``tf32x3`` needs compute capability 8.0 or newer
(``supported``). On the CPU and for other dtypes the product runs in PyTorch.
"""

from __future__ import annotations

from functools import cache

import torch

from fastmace.kernels.triton import tile_rule

PRECISION = "tf32x3"
MIN_CAPABILITY = (8, 0)  # Triton's rule for tf32 dots (capability // 10 >= 8)


@cache
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gemm(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PRECISION: tl.constexpr):
        pm, pn = tl.program_id(0), tl.program_id(1)
        rm = (pm * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        rn = (pn * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        rk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for start in range(0, K, BLOCK_K):
            kk = start + rk
            a = tl.load(a_ptr + rm[:, None] * sam + kk[None, :] * sak,
                        mask=(rm < M)[:, None] & (kk < K)[None, :], other=0.0)
            b = tl.load(b_ptr + kk[:, None] * sbk + rn[None, :] * sbn,
                        mask=(kk < K)[:, None] & (rn < N)[None, :], other=0.0)
            acc += tl.dot(a, b, input_precision=PRECISION)
        tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc, mask=(rm < M)[:, None] & (rn < N)[None, :])

    return gemm


def supported(device) -> tuple[bool, str]:
    """(True, "ok") if Triton's tf32x3 runs on ``device``: CUDA with compute capability >= 8.0."""
    device = torch.device(device)
    if device.type != "cuda":
        return False, f"{device.type} device: tf32x3 needs CUDA"
    index = device.index if device.index is not None else torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(index)
    if tuple(capability) < MIN_CAPABILITY:
        return False, (f"compute capability {capability[0]}.{capability[1]} < "
                       f"{MIN_CAPABILITY[0]}.{MIN_CAPABILITY[1]}: no TF32 tensor cores")
    return True, "ok"


def uses_tf32x3_tile(k: int, n: int) -> bool:
    """Whether ``[M, k] @ [k, n]`` fills at least one ``BLOCK_K x BLOCK_K`` (32 x 32) tile along K and N
    (OPT5-T05). Below ``BLOCK_K`` (e.g. K = 6 species inputs or N = 1..6 readout heads), a 64x64x32 tensor-core
    tile is >= 50-98 % zero-padded, slower than cuBLAS SIMT FP32, and up to 6.9x less accurate than IEEE
    (OPT4-T03 §3); those sub-tile products run on SIMT IEEE FP32."""
    return int(k) >= tile_rule.BLOCK_K and int(n) >= tile_rule.BLOCK_K


def _launch(a, b, out):
    import triton

    m, k = a.shape
    n = b.shape[1]
    if not uses_tf32x3_tile(k, n):
        torch.mm(a.contiguous(), b.contiguous(), out=out)
        return
    grid = (triton.cdiv(m, tile_rule.BLOCK_M), triton.cdiv(n, tile_rule.BLOCK_N))
    _kernel()[grid](a, b, out, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1), out.stride(0),
                    out.stride(1), BLOCK_M=tile_rule.BLOCK_M, BLOCK_N=tile_rule.BLOCK_N, BLOCK_K=tile_rule.BLOCK_K,
                    PRECISION=PRECISION, num_warps=tile_rule.NUM_WARPS, num_stages=tile_rule.NUM_STAGES)

@torch.library.custom_op("fastmace::tf32x3_mm", mutates_args=())
def tf32x3_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` (``[M, K] @ [K, N]``, any strides) as 3xTF32 on CUDA float32; PyTorch's product otherwise."""
    if a.device.type != "cuda" or a.dtype != torch.float32 or b.dtype != torch.float32:
        return a @ b
    out = a.new_empty((a.shape[0], b.shape[1]))
    if out.numel():
        if a.shape[1] == 0:
            out.zero_()
        else:
            _launch(a, b, out)
    return out


@tf32x3_mm.register_fake
def _(a, b):
    return a.new_empty((a.shape[0], b.shape[1]))


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_a = tf32x3_mm(grad, b.t()) if ctx.needs_input_grad[0] else None
    grad_b = tf32x3_mm(a.t(), grad) if ctx.needs_input_grad[1] else None
    return grad_a, grad_b


# The op's own gradients (through itself, so differentiable again): a module that fx traces records the op, not
# Tf32x3Matmul (OMOL-0's joint embedding block), and the compiled force call differentiates that trace twice.
tf32x3_mm.register_autograd(_backward, setup_context=_setup_context)


class Tf32x3Matmul(torch.autograd.Function):
    """``a @ b`` through ``tf32x3_mm``, both input gradients through it as well (``b_t``: b's contiguous
    transpose for the gradient of ``a``, or None)."""

    @staticmethod
    def forward(ctx, a, b, b_t):
        ctx.save_for_backward(a if b.requires_grad else None, b_t if b_t is not None else b)
        ctx.transposed = b_t is not None
        return tf32x3_mm(a, b)

    @staticmethod
    def backward(ctx, grad):
        a, saved = ctx.saved_tensors
        grad_a = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a = tf32x3_mm(grad, saved if ctx.transposed else saved.t())
        if ctx.needs_input_grad[1]:
            grad_b = tf32x3_mm(a.t(), grad)
        return grad_a, grad_b, None


def tf32x3_matmul(a, b, b_t=None):
    """``a @ b`` as 3xTF32 (see the module docstring); ``b_t`` is b's contiguous transpose, if kept."""
    return Tf32x3Matmul.apply(a, b, b_t)


__all__ = ["MIN_CAPABILITY", "PRECISION", "Tf32x3Matmul", "supported", "tf32x3_matmul", "tf32x3_mm", "uses_tf32x3_tile"]
