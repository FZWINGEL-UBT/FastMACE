"""Block-diagonal (per-irrep) linear layer without layout copies (INPLACE_BLOCK_LINEAR).

The ir_mul core layout stores component m of channel u of a segment at
offset + m * mul + u of each row. The per-l product of _BlockLinear needs the
segment as an [n * dim, mul] matrix, which is not a view of an [n, width]
row (copy), and it concatenates the per-segment results (copy). Here each
segment is read as the [dim, n, mul] view with strides (mul, width, 1) and the
product written into the same view of a preallocated output: one
strided-batched GEMM per block (lda = ldc = row width, batch stride = mul, the
weight shared across the batch), no copies. Same dot products as the
[n * dim, mul] GEMM; exact up to rounding. The backward is the same operation
with the transposed weights (the weights are constants in MD).
"""
from typing import List

import torch


def _segment(t, offset, mul, dim):
    n = t.shape[0]
    return t[:, offset:offset + dim * mul].view(n, dim, mul).transpose(0, 1)


def _apply(x, weights, seg_in, seg_out, ends, width_out):
    if x.stride(-1) != 1:
        x = x.contiguous()
    n = x.shape[0]
    out = x.new_empty((n, width_out))
    written = set()
    for (i, j), weight in zip(ends, weights):
        offset_in, mul_in, dim = seg_in[i]
        offset_out, mul_out, _ = seg_out[j]
        a = _segment(x, offset_in, mul_in, dim)
        o = _segment(out, offset_out, mul_out, dim)
        b = weight.expand(dim, mul_in, mul_out)
        if j in written:
            torch.baddbmm(o, a, b, out=o)
        else:
            torch.bmm(a, b, out=o)
            written.add(j)
    for j, (offset, mul, dim) in enumerate(seg_out):
        if j not in written:
            out[:, offset:offset + dim * mul].zero_()
    return out


def _unflatten(flat, size):
    return [tuple(flat[k:k + size]) for k in range(0, len(flat), size)]


@torch.library.custom_op("fastmace::inplace_block_linear", mutates_args=())
def inplace_block_linear(x: torch.Tensor, weights: List[torch.Tensor], seg_in: List[int],
                         seg_out: List[int], ends: List[int], width_out: int) -> torch.Tensor:
    return _apply(x, weights, _unflatten(seg_in, 3), _unflatten(seg_out, 3), _unflatten(ends, 2), width_out)


@inplace_block_linear.register_fake
def _(x, weights, seg_in, seg_out, ends, width_out):
    return x.new_empty((x.shape[0], width_out))


class _InplaceBlockLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, spec, weights, weights_t):
        seg_in, seg_out, ends, width_in, width_out = spec
        ctx.spec = spec
        ctx.save_for_backward(*weights_t)
        return inplace_block_linear(x, list(weights), seg_in, seg_out, ends, width_out)

    @staticmethod
    def backward(ctx, grad):
        seg_in, seg_out, ends, width_in, _ = ctx.spec
        swapped = [v for k in range(0, len(ends), 2) for v in (ends[k + 1], ends[k])]
        dx = inplace_block_linear(grad, list(ctx.saved_tensors), seg_out, seg_in, swapped, width_in)
        return dx, None, None, None


def block_linear(x, segments_in, segments_out, block_ends, weights, weights_t):
    """x @ blockdiag(weights) in the ir_mul layout; segments (offset, mul, dim), block_ends (i, j)."""
    spec = ([v for s in segments_in for v in s], [v for s in segments_out for v in s],
            [v for e in block_ends for v in e], sum(m * d for _, m, d in segments_in),
            sum(m * d for _, m, d in segments_out))
    return _InplaceBlockLinear.apply(x, spec, tuple(weights), tuple(weights_t))
