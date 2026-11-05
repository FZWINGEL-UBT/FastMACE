"""A block linear's per-irrep products on the [atoms, width] tensor itself (round 5, 5 November 2026; A100 profile at
2a016fd4, "GEMM layout glue").

``BlockLinear`` multiplies each irrep segment's rows ``x[:, o:o + d * k].reshape(n * d, k)`` by its weight and joins the
results with ``torch.cat``. The reshape of a column slice is a copy (Inductor's ``clone_slice`` kernels, about 160 us
per A100 force call on Benchmark_loup) and the join another (``cat_view``, about 125 us). CUTLASS's gathered and
scattered 3xTF32 entry (``tf32x3.species_into``, one species) reads row (a, j) of segment ``x[:, o:o + d * k]`` at
``(a * width + o + j * k) / lda`` rows of ``lda`` floats and writes its result straight into the joined output the same
way, so neither copy remains. The arithmetic is that of ``species_mm``'s CUTLASS kernel (64x64 tiles, 3xTF32): rounding
against the copy path's ``mm_into``, not bitwise (both FP32-class).

``segment_rows(x, weights, weights_t, plan, width_out)``: ``plan`` holds (input offset, irrep dim, output offset) per
block, block b mapping ``x[:, o_in:o_in + d * k_b]`` to ``out[:, o_out:o_out + d * n_b]`` with ``weights[b]`` [k_b,
n_b]; every output segment and every input segment at most once (no accumulation: the kernel has no beta). Columns no
block writes are zero. Its input gradient is the same op with the transposed weights and the plan reversed. Products
whose shape ``tf32x3.takes`` refuses, or operands that do not fit the kernel, run as the copy path does (``mm_into``
or cuBLAS IEEE into a temporary, then copied into place).

``SEGMENT_ROWS``: None: on for ``SEGMENT_ROWS_GPUS`` (decision 66, after the MD-step A/Bs); True / False: an A/B
side's choice. Only where the module's tf32x3 backend is CUTLASS (compute capability 8.0).
"""

from __future__ import annotations

import math

import torch

from fastmace.kernels.gpu_models import GPUModels

SEGMENT_ROWS = None
SEGMENT_ROWS_GPUS = GPUModels(())

_INDICES = {}  # (device index, rows, size, row stride, dim stride, ld) -> int32 row indices (built outside captures)


def segment_rows_default(device) -> bool:
    """``SEGMENT_ROWS`` for ``device``: the constant when set, else on for a ``SEGMENT_ROWS_GPUS`` CUDA device."""
    if SEGMENT_ROWS is not None:
        return bool(SEGMENT_ROWS)
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_name(device) in SEGMENT_ROWS_GPUS


def plan_of(segments_in, segments_out, block_ends, weights):
    """The flat plan [o_in, d, o_out, ...] of a BlockLinear's blocks, or None when a segment would be read or written
    by two blocks (grouped products over several input segments included) or the dims disagree."""
    plan, seen_in, seen_out = [], set(), set()
    for (i, j), weight in zip(block_ends, weights):
        if isinstance(i, tuple):
            if len(i) != 1:
                return None
            i = i[0]
        if i in seen_in or j in seen_out:
            return None
        seen_in.add(i)
        seen_out.add(j)
        o_in, k, d = segments_in[i]
        o_out, m, d_out = segments_out[j]
        if d != d_out or tuple(weight.shape) != (k, m):
            return None
        plan += [int(o_in), int(d), int(o_out)]
    return plan


def _ld(view):
    """The row unit of a [rows, size, columns] view (row (a, j) at a * s0 + j * s1 = index * ld), or 0."""
    if view.stride(2) != 1 or view.shape[2] % 4:
        return 0
    ld = math.gcd(view.stride(0), view.stride(1))
    if ld <= 0 or ld % 4 or view.data_ptr() % 16:
        return 0
    return ld


def _indices(view, ld):
    rows, size, _ = view.shape
    key = (view.device.index, rows, size, view.stride(0), view.stride(1), ld)
    cached = _INDICES.get(key)
    if cached is not None:
        return cached
    a = torch.arange(rows, device=view.device, dtype=torch.int64)[:, None]
    j = torch.arange(size, device=view.device, dtype=torch.int64)
    index = ((a * view.stride(0) + j * view.stride(1)) // ld).to(torch.int32).flatten()
    if not torch.cuda.is_current_stream_capturing():  # a capture's tensors hold values only once it replays
        _INDICES[key] = index
    return index


def _counts(rows, device):
    key = (device.index, "counts", rows)
    cached = _INDICES.get(key)
    if cached is not None:
        return cached
    counts = torch.full((1,), rows, dtype=torch.int32, device=device)
    if not torch.cuda.is_current_stream_capturing():
        _INDICES[key] = counts
    return counts


def _product(a, w, y):
    """``y[:, j] = a[:, j] @ w`` for views a [n, d, k] and y [n, d, m] of larger row-major tensors."""
    from fastmace.kernels.cutlass import tf32x3
    from fastmace.kernels.triton import species_gemm
    from fastmace.kernels.triton.tf32x3_gemm import _launch

    n, d, k = a.shape
    m = w.shape[1]
    lda, ldy = _ld(a), _ld(y)
    if (lda and ldy == m and w.is_contiguous() and w.data_ptr() % 16 == 0 and k % 4 == 0
            and tf32x3.takes(n * d, k, m)):
        tf32x3.species_into(a, lda, _indices(a, lda), w.unsqueeze(0), y, _indices(y, ldy), _counts(n, a.device), d,
                            species_gemm.tiles(n, d, 1))
        return
    out = a.new_empty((n * d, m))
    _launch(a.reshape(n * d, k), w, out)
    y.copy_(out.view(n, d, m))


def _apply(x, weights, plan, width_out):
    n = x.shape[0]
    out = x.new_empty((n, width_out))
    written = []
    for b, w in enumerate(weights):
        o_in, d, o_out = plan[3 * b:3 * b + 3]
        k, m = w.shape
        written.append((o_out, o_out + d * m))
        if n:
            _product(x[:, o_in:o_in + d * k].view(n, d, k), w, out[:, o_out:o_out + d * m].view(n, d, m))
    start = 0
    for begin, end in sorted(written) + [(width_out, width_out)]:
        if begin > start:
            out[:, start:begin].zero_()
        start = max(start, end)
    return out


@torch.library.custom_op("fastmace::tf32x3_segment_rows", mutates_args=())
def segment_rows(x: torch.Tensor, weights: list[torch.Tensor], weights_t: list[torch.Tensor], plan: list[int],
                 width_out: int) -> torch.Tensor:
    """The joined [n, width_out] output of a BlockLinear's 1:1 blocks (see the module docstring); ``x`` contiguous."""
    if x.device.type != "cuda" or x.dtype != torch.float32:
        out = x.new_zeros((x.shape[0], width_out))
        for b, w in enumerate(weights):
            o_in, d, o_out = plan[3 * b:3 * b + 3]
            k, m = w.shape
            out[:, o_out:o_out + d * m] = (x[:, o_in:o_in + d * k].reshape(-1, k) @ w).reshape(-1, d * m)
        return out
    return _apply(x.contiguous(), weights, plan, width_out)


@segment_rows.register_fake
def _(x, weights, weights_t, plan, width_out):
    return x.new_empty((x.shape[0], width_out))


def reversed_plan(plan):
    return [v for b in range(len(plan) // 3) for v in (plan[3 * b + 2], plan[3 * b + 1], plan[3 * b])]


def _setup_context(ctx, inputs, output):
    x, weights, weights_t, plan, width_out = inputs
    ctx.save_for_backward(*weights, *weights_t)
    ctx.count, ctx.plan, ctx.width_in = len(weights), plan, x.shape[1]


def _backward(ctx, grad):
    saved = ctx.saved_tensors
    weights, weights_t = list(saved[:ctx.count]), list(saved[ctx.count:])
    grad_x = segment_rows(grad.contiguous(), weights_t, weights, reversed_plan(ctx.plan), ctx.width_in)
    return grad_x, None, None, None, None


segment_rows.register_autograd(_backward, setup_context=_setup_context)

__all__ = ["SEGMENT_ROWS", "SEGMENT_ROWS_GPUS", "plan_of", "reversed_plan", "segment_rows", "segment_rows_default"]
