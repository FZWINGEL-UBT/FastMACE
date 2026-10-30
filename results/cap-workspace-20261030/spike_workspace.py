"""Spike: joined products written into a workspace view through an out-argument custom op, under torch.compile.

Checks (CPU, cuBLAS branch of the op):
1. eager and compiled forward + backward give the same values as the existing op, bitwise;
2. the product returned is a view of the workspace (no clone by functionalization);
3. the input gradients are written into the workspace too.
"""
import os
from typing import List

import torch

import fastmace  # noqa: F401
from fastmace.kernels.triton import segment_gemm as sg


@torch.library.custom_op("spike::joined_products_into", mutates_args=("out",))
def joined_products_into(rows: List[torch.Tensor], weights: List[torch.Tensor], offsets: List[int], dims: List[int],
                         out: torch.Tensor) -> None:
    atoms = rows[0].shape[0] // dims[0]
    for x, w, offset, dim in zip(rows, weights, offsets, dims):
        mul = int(w.shape[1])
        out[:, offset:offset + dim * mul] = (x @ w).reshape(atoms, dim * mul)


@torch.library.custom_op("spike::input_grads_into", mutates_args=("outs",))
def input_grads_into(grad: torch.Tensor, weights_t: List[torch.Tensor], offsets: List[int], dims: List[int],
                     outs: List[torch.Tensor]) -> None:
    atoms, width = grad.shape
    for wt, offset, dim, dx in zip(weights_t, offsets, dims, outs):
        mul_out = int(wt.shape[0])
        dx.copy_(grad[:, offset:offset + dim * mul_out].reshape(atoms * dim, mul_out) @ wt)


@torch.library.custom_op("spike::input_grads_into_space", mutates_args=("space",))
def input_grads_into_space(grad: torch.Tensor, weights_t: List[torch.Tensor], offsets: List[int], dims: List[int],
                           space: torch.Tensor) -> None:
    atoms, width = grad.shape
    flat, start = space.view(-1), 0
    for wt, offset, dim in zip(weights_t, offsets, dims):
        mul_out, mul_in = int(wt.shape[0]), int(wt.shape[1])
        n = atoms * dim
        flat[start:start + n * mul_in].view(n, mul_in).copy_(
            grad[:, offset:offset + dim * mul_out].reshape(n, mul_out) @ wt)
        start += n * mul_in


class Reuse(torch.autograd.Function):
    """Variant B: the input gradients are written into the product's own storage (the product is dead by then)."""

    @staticmethod
    def forward(ctx, offsets, dims, count, out, *tensors):
        rows, weights, weights_t = tensors[:count], tensors[count:2 * count], tensors[2 * count:]
        ctx.offsets, ctx.dims, ctx.count = offsets, dims, count
        ctx.shapes = [(r.shape[0], r.shape[1]) for r in rows]
        joined_products_into(list(rows), list(weights), offsets, dims, out)
        ctx.mark_dirty(out)
        ctx.save_for_backward(out, *weights_t)
        return out

    @staticmethod
    def backward(ctx, grad):
        out, *weights_t = ctx.saved_tensors
        input_grads_into_space(grad, list(weights_t), ctx.offsets, ctx.dims, out)
        flat, outs, start = out.view(-1), [], 0
        for n, k in ctx.shapes:
            outs.append(flat[start:start + n * k].view(n, k))
            start += n * k
        return (None, None, None, None, *outs, *([None] * (2 * ctx.count)))


class Into(torch.autograd.Function):
    @staticmethod
    def forward(ctx, offsets, dims, count, out, grad_space, *tensors):
        rows, weights, weights_t = tensors[:count], tensors[count:2 * count], tensors[2 * count:]
        ctx.offsets, ctx.dims, ctx.count = offsets, dims, count
        ctx.shapes = [(r.shape[0], r.shape[1]) for r in rows]
        ctx.save_for_backward(*weights_t)
        ctx.grad_space = grad_space
        joined_products_into(list(rows), list(weights), offsets, dims, out)
        ctx.mark_dirty(out)
        return out

    @staticmethod
    def backward(ctx, grad):
        weights_t = ctx.saved_tensors
        outs, start = [], 0
        for n, k in ctx.shapes:
            outs.append(ctx.grad_space.view(-1)[start:start + n * k].view(n, k))
            start += n * k
        input_grads_into(grad, list(weights_t), ctx.offsets, ctx.dims, outs)
        return (None, None, None, None, None, *outs, *([None] * (2 * ctx.count)))


def model(into, ws=None):
    def f(x0, x1, w0, w1, w0t, w1t):
        rows = [x0, x1]
        if into == "reuse":
            p = Reuse.apply([0, w0.shape[1]], [1, 3], 2, ws[0], *rows, w0, w1, w0t, w1t)
        elif into:
            out = ws[0]
            p = Into.apply([0, w0.shape[1]], [1, 3], 2, out, ws[1], *rows, w0, w1, w0t, w1t)
        else:
            p = sg.joined_products(rows, [w0, w1], [w0t, w1t], [0, w0.shape[1]], [1, 3], w0.shape[1] + 3 * w1.shape[1])
        return (torch.sin(p) * p).sum(), p
    return f


def run(f, inputs):
    xs = [t.detach().clone().requires_grad_(i < 2) for i, t in enumerate(inputs)]
    energy, p = f(*xs)
    g = torch.autograd.grad(energy, xs[:2])
    return energy.detach(), p, g


torch.manual_seed(0)
n, k0, k1, m0, m1 = 64, 16, 8, 32, 8
inputs = [torch.randn(n, k0, dtype=torch.float64), torch.randn(3 * n, k1, dtype=torch.float64),
          torch.randn(k0, m0, dtype=torch.float64), torch.randn(k1, m1, dtype=torch.float64)]
inputs += [inputs[2].t().contiguous(), inputs[3].t().contiguous()]
VARIANT = os.environ.get("VARIANT", "into")
VARIANT = True if VARIANT == "into" else VARIANT
for compiled in (False, True):
    base = model(False)
    ref = run(torch.compile(base, backend="inductor", fullgraph=True) if compiled else base, inputs)
    ws = (torch.full((n, m0 + 3 * m1), float("nan"), dtype=torch.float64),
          torch.full((n * k0 + 3 * n * k1,), float("nan"), dtype=torch.float64))
    f = model(VARIANT, ws)
    if compiled:
        f = torch.compile(f, backend="inductor", fullgraph=True)
    energy, p, g = run(f, inputs)
    same = torch.equal(energy, ref[0]) and torch.equal(p.detach(), ref[1].detach()) and all(
        torch.equal(a, b) for a, b in zip(g, ref[2]))
    in_ws = p.untyped_storage().data_ptr() == ws[0].untyped_storage().data_ptr()
    grads_in_ws = any(gg.untyped_storage().data_ptr() in (ws[0].untyped_storage().data_ptr(),
                                                           ws[1].untyped_storage().data_ptr()) for gg in g)
    print("energy", float(energy), float(ref[0]), "p equal", torch.equal(p.detach(), ref[1].detach()),
          "p nan", bool(torch.isnan(p).any()), "max |p - ref|", float((p.detach() - ref[1].detach()).abs().max()),
          "ws0 == ref", torch.equal(ws[0], ref[1].detach()), p.shape, ref[1].shape, p.stride(), ref[1].stride())
    diff = max(float((a - b).abs().max()) for a, b in zip(g, ref[2]))
    print(f"compiled={compiled}: max grad diff {diff:.3g} bitwise={same} product_in_workspace={in_ws} "
          f"grads_in_workspace={grads_in_ws}")
