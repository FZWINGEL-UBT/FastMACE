"""Per-slot sums of edge outer products for the reassociated first interaction.

edge_outer_sum(a, y, key, slots) returns S with
    S[s, d, k] = sum over edges e with key[e] = s of y[e, d] a[e, k],
flattened to [slots, D * K]. The forward is ordinary PyTorch (index_add of the
outer products, which Inductor fuses into one scatter kernel). The backward
    grad_a[e, k] = sum_d G[key[e], d, k] y[e, d],
    grad_y[e, d] = sum_k G[key[e], d, k] a[e, k]
is one Triton kernel on CUDA that reads each gathered row of G once for both
results; Inductor's own backward read it twice in two reduction kernels
(research roadmap item 11, experiment 2). The CPU uses the same formulas in
PyTorch. Only first derivatives are supported (forces, not force derivatives).

With segmented=True the forward is a Triton kernel too: each program walks a
chunk of consecutive edges, accumulates the outer products of a run of equal
keys in registers and adds the run's sum to its slot once (an atomic add only
where a run ends), instead of one atomic add per edge and element. Correct for
any key order; it pays off when equal keys are contiguous, i.e. with edges
sorted by receiver and then by sender species (the segmented first sum).

With inner=I > 1 the slots are I-wide groups (key = group * I + i) and each group's
block is laid out [D, I, K] instead of [I, D, K]: S viewed as [slots / I, D, I, K].
The reassociated first interaction then reads each (l, m) range of the sums as
one contiguous [rows * (2l + 1), I * K] matrix, without a transposing copy before
its GEMMs (or their backward). inner=1 is the plain layout. The CUDA backward's
grad_y differs from the plain layout's by rounding (the compiler orders its sum
over K differently for the other addresses); grad_a and the sums are the same.
"""

from functools import cache

import torch

FORWARD_CHUNK = 32  # edges per program of the segmented forward (fixed for every crystal and GPU)
# The kernels read ``a`` at its row stride where its rows are unit-stride (False: a contiguous copy first, as before
# 3 November 2026; kept for the A/B).
STRIDED_ROWS = True
BACKWARD_BLOCK_EDGES = 32  # edges per program of the backward (fixed for every crystal and GPU)


@cache
def _base_fn():
    import triton

    @triton.jit
    def base(key, D, K, INNER):
        """Offset of slot ``key``'s element (0, 0): group ``key // INNER``, then ``i = key % INNER`` in [D, I, K]."""
        return (key // INNER) * (D * INNER * K) + (key % INNER) * K

    return base


@cache
def _triton_forward():
    import triton
    import triton.language as tl

    _base = _base_fn()

    @triton.jit
    def kernel(a_ptr, y_ptr, key_ptr, out_ptr, num_edges, a_stride, K: tl.constexpr, D: tl.constexpr,
               INNER: tl.constexpr,
               CHUNK: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr):
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
            a = tl.load(a_ptr + row * a_stride + k, mask=k < K, other=0.0)
            y = tl.load(y_ptr + row * D + d, mask=d < D, other=0.0)
            acc += y[:, None] * a[None, :]
        tl.atomic_add(out_ptr + _base(current, D, K, INNER) + offsets, acc, mask=mask, sem="relaxed")

    return kernel


def _rows(a):
    """``a`` itself where its rows are unit-stride (a column slice such as the generic tables' ``augmented[:, :-1]``,
    read in place at its row stride), else a contiguous copy."""
    return a if STRIDED_ROWS and a.stride(1) == 1 else a.contiguous()


@torch.library.custom_op("fastmace::edge_outer_sum_forward", mutates_args=())
def edge_outer_sum_forward(a: torch.Tensor, y: torch.Tensor, key: torch.Tensor,
                           slots: int, inner: int = 1) -> torch.Tensor:
    """[slots, D * K] per-slot sums of y[e, d] a[e, k] by key runs (segmented forward), in the ``inner`` layout."""
    num_edges, k = a.shape
    d = y.shape[1]
    out = torch.zeros((slots, d * k), dtype=a.dtype, device=a.device)
    if not a.is_cuda or a.dtype != torch.float32:
        return _to_inner(out.index_add(0, key, (y[:, :, None] * a[:, None, :]).reshape(num_edges, d * k)), d, k,
                         inner)
    import triton
    a, y, key = _rows(a), y.contiguous(), key.contiguous()
    if num_edges:
        _triton_forward()[(triton.cdiv(num_edges, FORWARD_CHUNK),)](
            a, y, key, out, num_edges, a.stride(0), K=k, D=d, INNER=inner, CHUNK=FORWARD_CHUNK,
            BLOCK_K=triton.next_power_of_2(k), BLOCK_D=triton.next_power_of_2(d), num_warps=1)
    return out


@edge_outer_sum_forward.register_fake
def _(a, y, key, slots, inner=1):
    return a.new_empty((slots, y.shape[1] * a.shape[1]))


def _to_inner(summed, d, k, inner):
    """The plain layout [slots, D * K] as the ``inner`` layout (a copy; no-op for inner = 1)."""
    if inner == 1:
        return summed
    return summed.view(-1, inner, d, k).transpose(1, 2).reshape(summed.shape)


def _from_inner(summed, d, k, inner):
    """The ``inner`` layout as the plain one, [slots, D, K] (a copy; a view for inner = 1)."""
    if inner == 1:
        return summed.reshape(-1, d, k)
    return summed.reshape(-1, d, inner, k).transpose(1, 2).reshape(-1, d, k)


@cache
def _triton_backward():
    import triton
    import triton.language as tl

    _base = _base_fn()

    @triton.jit
    def kernel(grad_ptr, a_ptr, y_ptr, key_ptr, grad_a_ptr, grad_y_ptr, num_edges, a_stride,
               K: tl.constexpr, D: tl.constexpr, INNER: tl.constexpr, BLOCK_E: tl.constexpr,
               BLOCK_K: tl.constexpr):
        edges = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
        edge_mask = edges < num_edges
        edges = edges.to(tl.int64)
        k = tl.arange(0, BLOCK_K)
        mask = edge_mask[:, None] & (k < K)[None, :]
        key = tl.load(key_ptr + edges, mask=edge_mask, other=0).to(tl.int64)
        a = tl.load(a_ptr + edges[:, None] * a_stride + k[None, :], mask=mask, other=0.0)
        rows = grad_ptr + _base(key, D, K, INNER)[:, None] + k[None, :]
        grad_a = a * 0
        for d in tl.static_range(D):
            g = tl.load(rows + d * (INNER * K), mask=mask, other=0.0)
            y = tl.load(y_ptr + edges * D + d, mask=edge_mask, other=0.0)
            grad_a += g * y[:, None]
            tl.store(grad_y_ptr + edges * D + d, tl.sum(g * a, axis=1), mask=edge_mask)
        tl.store(grad_a_ptr + edges[:, None] * K + k[None, :], grad_a, mask=mask)

    return kernel


@torch.library.custom_op("fastmace::edge_outer_sum_backward", mutates_args=())
def edge_outer_sum_backward(grad: torch.Tensor, a: torch.Tensor, y: torch.Tensor,
                            key: torch.Tensor, inner: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """(grad_a, grad_y) for the upstream gradient grad [slots, D * K] (in the ``inner`` layout)."""
    num_edges, k = a.shape
    d = y.shape[1]
    if not grad.is_cuda:
        rows = _from_inner(grad, d, k, inner)[key]
        return torch.einsum("edk,ed->ek", rows, y), torch.einsum("edk,ek->ed", rows, a)
    import triton
    grad, a, y, key = grad.contiguous(), _rows(a), y.contiguous(), key.contiguous()
    grad_a = a.new_empty(a.shape)
    grad_y = torch.empty_like(y)
    if num_edges:
        _triton_backward()[(triton.cdiv(num_edges, BACKWARD_BLOCK_EDGES),)](
            grad, a, y, key, grad_a, grad_y, num_edges, a.stride(0),
            K=k, D=d, INNER=inner, BLOCK_E=BACKWARD_BLOCK_EDGES, BLOCK_K=triton.next_power_of_2(k))
    return grad_a, grad_y


@edge_outer_sum_backward.register_fake
def _(grad, a, y, key, inner=1):
    return a.new_empty(a.shape), torch.empty_like(y)


class _EdgeOuterSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, y, key, slots, segmented, inner):
        ctx.save_for_backward(a, y, key)
        ctx.inner = inner
        if segmented:
            return edge_outer_sum_forward(a, y, key, slots, inner)
        outer = (y[:, :, None] * a[:, None, :]).reshape(a.shape[0], y.shape[1] * a.shape[1])
        summed = torch.zeros((slots, outer.shape[1]), dtype=outer.dtype, device=outer.device)
        return _to_inner(summed.index_add(0, key, outer), y.shape[1], a.shape[1], inner)

    @staticmethod
    def backward(ctx, grad):
        a, y, key = ctx.saved_tensors
        grad_a, grad_y = edge_outer_sum_backward(grad, a, y, key, ctx.inner)
        return grad_a, grad_y, None, None, None, None


def edge_outer_sum(a, y, key, slots, segmented=False, inner=1):
    """[slots, D * K]: per-slot sums of y[e, d] a[e, k] (d major), with a fused backward.

    segmented=True: run-segmented Triton forward; inner=I: groups of I slots laid out [D, I, K] (see the module
    docstring; ``slots`` a multiple of I).
    """
    return _EdgeOuterSum.apply(a, y, key, slots, bool(segmented), int(inner))
