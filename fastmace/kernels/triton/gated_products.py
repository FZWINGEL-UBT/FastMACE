"""The gate's backward folded into ``linear_2``'s input-gradient GEMM (plans/opt-10.md, OPT10-T01, ``gate_epilogue``).

In an MH-1-class ``GatedRows`` chain with ``segment_gemm``, ``linear_2``'s inputs are the scalars and, per gated
segment l (D = 2l + 1 components per atom), the rows ``gated * gate`` (``[atoms * D, mul]``, the atom's gates
broadcast over its D rows). The stock backward writes ``drows = grad_l @ W_l^T`` and then an elementwise kernel
reads ``drows``, the gated rows and the gates again for ``dgated = drows * gate`` and
``dgate = sum_m drows * gated``.

``gated_joined_products(scalars, gated, gates, ...)`` has the same forward (the elementwise products, then
``joined_products_op``). Its backward runs, for every gated segment whose product the shape rule
(``tile_rule.segment_gemm_choice``) gives to Triton, one atom-tiled kernel: per (atom block, column block) it runs
the K-loop of each component m, multiplies by the gate tile (loaded once), stores ``dgated`` and sums ``dgate`` over
m in registers. ``drows`` is never written, there are no atomics, and the m-sum needs no partial tensor. Segments
the rule leaves to cuBLAS, the scalars' product, the CPU and other dtypes keep the stock operations.

Not bitwise the stock backward: the K-sums run in ``tl.dot`` with another tile shape, and the m-sum is in another
order (7-8e-7 relative on the RTX 4000 Ada, docs/validation/opt10-T01.md). Deterministic. The weights get no
gradient (MD forces only).
"""

from __future__ import annotations

from functools import cache
from typing import List

import torch

from fastmace.kernels.triton import segment_gemm as sg

# The fused kernel's tile: atoms x columns x K, warps, stages; the fastest of four on OPT10-T01's quick pass on the
# RTX 4000 Ada for NbCl5, NaNbO3 and NaCl alike (one fixed configuration, SPEC §9.1).
BLOCK_A, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_STAGES = 32, 64, 32, 4, 3


@cache
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gated_input_grads(g_ptr, wt_ptr, gated_ptr, gate_ptr, dgated_ptr, dgate_ptr, atoms_n, N, K, g_n, g_m,
                          gate_n, dgate_n, D: tl.constexpr, BLOCK_A: tl.constexpr, BLOCK_N: tl.constexpr,
                          BLOCK_K: tl.constexpr, PRECISION: tl.constexpr):
        # Row (atom, m) of the gradient starts at atom * g_n + m * g_m; gated and dgated rows are atom * D + m.
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


def _fused(grad, offset, mul_out, wt, gated, gates, gate_offset, dgated, dgates, dim, precision):
    """dgated and dgates[:, gate_offset:gate_offset + mul_in] of one gated segment in one launch."""
    import triton

    atoms, width = grad.shape
    mul_in = int(wt.shape[1])
    grid = (triton.cdiv(atoms, BLOCK_A), triton.cdiv(mul_in, BLOCK_N))
    _kernel()[grid](grad.reshape(-1)[offset:], wt, gated, gates.reshape(-1)[gate_offset:],
                    dgated, dgates.reshape(-1)[gate_offset:], atoms, mul_in, mul_out, width, mul_out,
                    gates.stride(0), dgates.stride(0), D=dim, BLOCK_A=BLOCK_A, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                    PRECISION=precision, num_warps=NUM_WARPS, num_stages=NUM_STAGES)


def _gated_rows(scalars, gated, gates, gate_offsets):
    """``linear_2``'s input rows: the scalars, then per gated segment its rows times the atom's gates."""
    n = scalars.shape[0]
    rows = [scalars]
    for x, offset in zip(gated, gate_offsets):
        mul = x.shape[1]
        rows.append((x.view(n, -1, mul) * gates.narrow(-1, offset, mul).unsqueeze(-2)).reshape(-1, mul))
    return rows


@torch.library.custom_op("fastmace::gated_products_input_grads", mutates_args=())
def gated_products_input_grads(grad: torch.Tensor, gated: List[torch.Tensor], gates: torch.Tensor,
                               weights_t: List[torch.Tensor], offsets: List[int], dims: List[int], inputs: List[int],
                               gate_offsets: List[int], precision: str) -> List[torch.Tensor]:
    """``[dscalars, *dgated, dgates]`` of ``joined_products`` over the gated rows (``gated_joined_products``).

    Product p reads ``grad``'s columns from ``offsets[p]`` (``dims[p]`` components) and multiplies by
    ``weights_t[p]``; its input is ``inputs[p]``: 0 the scalars, i > 0 gated segment i - 1, whose gates are
    ``gates``' columns from ``gate_offsets[i - 1]``."""
    atoms = grad.shape[0]
    gates = gates.contiguous()  # the kernel addresses the gates by row stride from a flat view
    choices = sg._choices([atoms * d for d in dims], [w.t() for w in weights_t], grad.device, grad.dtype, precision)
    covered = sum(int(x.shape[1]) for x in gated)
    dgates = gates.new_empty((atoms, gates.shape[1])) if covered == gates.shape[1] else gates.new_zeros(
        (atoms, gates.shape[1]))
    out: List[torch.Tensor] = [grad.new_empty(0)] * (1 + len(gated))
    g = grad.contiguous()
    for wt, offset, dim, index, choice in zip(weights_t, offsets, dims, inputs, choices):
        mul_out, mul_in = int(wt.shape[0]), int(wt.shape[1])
        if index == 0:
            if choice.backend == "triton" and atoms:
                dx = grad.new_empty((atoms * dim, mul_in))
                sg._gemm(g, offset, g.shape[1], mul_out, wt, dx, 0, dim * mul_in, mul_in, atoms * dim, dim,
                         choice, precision)
            else:
                dx = grad[:, offset:offset + dim * mul_out].reshape(atoms * dim, mul_out) @ wt
            out[0] = dx
            continue
        x, gate_offset = gated[index - 1], gate_offsets[index - 1]
        if choice.backend == "triton" and atoms:
            dgated = grad.new_empty((atoms * dim, mul_in))
            _fused(g, offset, mul_out, wt, x.contiguous(), gates, gate_offset, dgated, dgates, dim, precision)
        else:
            drows = (grad[:, offset:offset + dim * mul_out].reshape(atoms * dim, mul_out) @ wt).view(atoms, dim, mul_in)
            gate = gates.narrow(-1, gate_offset, mul_in)
            dgated = (drows * gate.unsqueeze(-2)).reshape(atoms * dim, mul_in)
            dgates[:, gate_offset:gate_offset + mul_in] = (drows * x.view(atoms, dim, mul_in)).sum(-2)
        out[index] = dgated
    out.append(dgates)
    return out


@gated_products_input_grads.register_fake
def _(grad, gated, gates, weights_t, offsets, dims, inputs, gate_offsets, precision):
    atoms = grad.shape[0]
    scalars = next(int(wt.shape[1]) * dim for wt, dim, index in zip(weights_t, dims, inputs) if index == 0)
    return [grad.new_empty((atoms, scalars)), *[x.new_empty(x.shape) for x in gated],
            gates.new_empty((atoms, gates.shape[1]))]


class _GatedJoinedProducts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, offsets, dims, inputs, gate_offsets, width, precision, count, scalars, gates, *tensors):
        gated, weights, weights_t = tensors[:count], tensors[count:count + len(dims)], tensors[count + len(dims):]
        rows = _gated_rows(scalars, gated, gates, gate_offsets)
        ctx.args = (offsets, dims, inputs, gate_offsets, precision, count)
        ctx.save_for_backward(gates, *gated, *weights_t)
        return sg.joined_products_op([rows[i] for i in inputs], list(weights), offsets, dims, width, precision)

    @staticmethod
    def backward(ctx, grad):
        offsets, dims, inputs, gate_offsets, precision, count = ctx.args
        gates, *rest = ctx.saved_tensors
        gated, weights_t = rest[:count], rest[count:]
        grads = gated_products_input_grads(grad, list(gated), gates, list(weights_t), offsets, dims, inputs,
                                           gate_offsets, precision)
        dscalars, dgated, dgates = grads[0], grads[1:-1], grads[-1]
        return (None,) * 7 + (dscalars, dgates, *dgated, *([None] * (2 * len(dims))))


def gated_joined_products(scalars, gated, gates, gate_offsets, weights, weights_t, offsets, dims, inputs, width,
                          precision="ieee"):
    """``joined_products`` of ``linear_2`` over the gated rows, with the gate's backward in its input-gradient GEMM.

    ``scalars`` ``[atoms, S]``; ``gated[s]`` ``[atoms * D_s, mul_s]`` and its gates ``gates[:, gate_offsets[s]:
    gate_offsets[s] + mul_s]``; product p is ``rows[inputs[p]] @ weights[p]`` at column ``offsets[p]`` (rows: the
    scalars, then the gated rows in order). See the module docstring."""
    return _GatedJoinedProducts.apply(list(offsets), list(dims), list(inputs), list(gate_offsets), int(width),
                                      str(precision), len(gated), scalars, gates, *gated, *weights, *weights_t)


__all__ = ["BLOCK_A", "BLOCK_K", "BLOCK_N", "gated_joined_products", "gated_products_input_grads"]
