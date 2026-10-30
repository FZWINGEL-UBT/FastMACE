"""Prototype (measurement only): ``segment_gemm``'s joined products and their input gradients in a per-interaction
workspace, for ``memory_ledger.py --workspace-prototype`` (docs/validation/capacity-reinplace.md).

- ``joined_products_into`` writes ``joined_products_op``'s output into ``out``, with its kernels (``_choices``,
  ``_gemm``), so the values are bitwise the same.
- ``input_grads_into`` writes ``joined_products_input_grads``' outputs into consecutive views of ``space``.
- ``_IntoWorkspace`` (an ``autograd.Function``) writes the product into the interaction's workspace and, in the
  backward, the input gradients into the same storage: the product is dead by then (its only consumer, the symmetric
  contraction, has run its backward).
- ``InPlaceIntoSaved``, an Inductor post-grad pass: in a compiled backward the mutation of the saved product is
  functionalized as a clone plus a copy back into the workspace input, which it aliases; the pass calls the op on
  the saved tensor directly instead.
- ``install(natoms, device, dtype)``: patches ``GatedRows.forward`` and ``compile_model`` (each compiled model's
  ``GatedRows`` with joined products gets a workspace attribute ``(natoms, width)`` before its first call), and sets
  the pass in Inductor's config. Process-wide; a tool's experiment, not a library path.
"""

from __future__ import annotations

from typing import List

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass

from fastmace.kernels.triton import segment_gemm as sg


@torch.library.custom_op("fastmace_proto::joined_products_into", mutates_args=("out",))
def joined_products_into(rows: List[torch.Tensor], weights: List[torch.Tensor], offsets: List[int], dims: List[int],
                         width: int, precision: str, out: torch.Tensor) -> None:
    """``joined_products_op`` written into ``out`` ``[atoms, width]``."""
    atoms = rows[0].shape[0] // dims[0]
    covered = sum(dim * int(w.shape[1]) for dim, w in zip(dims, weights))
    if covered != width:
        out.zero_()
    choices = sg._choices([atoms * d for d in dims], weights, out.device, out.dtype, precision)
    for x, w, offset, dim, choice in zip(rows, weights, offsets, dims, choices):
        mul = int(w.shape[1])
        if choice.backend == "triton" and atoms:
            x = x.contiguous()
            sg._gemm(x, 0, dim * x.shape[1], x.shape[1], w, out, offset, width, mul, atoms * dim, dim, choice,
                     precision)
        else:
            out[:, offset:offset + dim * mul] = (x @ w).reshape(atoms, dim * mul)


@joined_products_into.register_fake
def _(rows, weights, offsets, dims, width, precision, out):
    return None


@torch.library.custom_op("fastmace_proto::input_grads_into", mutates_args=("space",))
def input_grads_into(grad: torch.Tensor, weights_t: List[torch.Tensor], offsets: List[int], dims: List[int],
                     precision: str, space: torch.Tensor) -> None:
    """``joined_products_input_grads`` written into consecutive views of ``space``."""
    atoms, width = grad.shape
    choices = sg._choices([atoms * d for d in dims], [w.t() for w in weights_t], grad.device, grad.dtype, precision)
    flat, start = space.view(-1), 0
    for wt, offset, dim, choice in zip(weights_t, offsets, dims, choices):
        mul_out, mul_in = int(wt.shape[0]), int(wt.shape[1])
        n = atoms * dim
        dx = flat[start:start + n * mul_in].view(n, mul_in)
        if choice.backend == "triton" and atoms:
            g = grad.contiguous()
            sg._gemm(g, offset, width, mul_out, wt, dx, 0, dim * mul_in, mul_in, n, dim, choice, precision)
        else:
            dx.copy_(grad[:, offset:offset + dim * mul_out].reshape(n, mul_out) @ wt)
        start += n * mul_in


@input_grads_into.register_fake
def _(grad, weights_t, offsets, dims, precision, space):
    return None


class _IntoWorkspace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, offsets, dims, width, precision, count, out, *tensors):
        rows, weights, weights_t = tensors[:count], tensors[count:2 * count], tensors[2 * count:]
        ctx.offsets, ctx.dims, ctx.count, ctx.precision = offsets, dims, count, precision
        ctx.shapes = [(r.shape[0], r.shape[1]) for r in rows]
        joined_products_into(list(rows), list(weights), offsets, dims, width, precision, out)
        ctx.mark_dirty(out)
        ctx.save_for_backward(out, *weights_t)
        return out

    @staticmethod
    def backward(ctx, grad):
        out, *weights_t = ctx.saved_tensors
        input_grads_into(grad, list(weights_t), ctx.offsets, ctx.dims, ctx.precision, out)
        flat, outs, start = out.view(-1), [], 0
        for n, k in ctx.shapes:
            outs.append(flat[start:start + n * k].view(n, k))
            start += n * k
        return (None, None, None, None, None, None, *outs, *([None] * (2 * ctx.count)))


class InPlaceIntoSaved(CustomGraphPass):
    """Post-grad: ``auto_functionalized_v2(input_grads_into, space=<base>)`` becomes a direct call on the base, and the
    copy of its new value back into the forward's workspace input (which the base aliases) is dropped."""

    def __call__(self, graph):
        from torch._higher_order_ops.auto_functionalize import auto_functionalized_v2

        target = torch.ops.fastmace_proto.input_grads_into.default
        for node in list(graph.nodes):
            if node.target is not auto_functionalized_v2 or node.args[0] is not target:
                continue
            kwargs = dict(node.kwargs)
            bases = kwargs.pop("_all_bases")
            index = kwargs.pop("_space_base_index")
            if any(k.startswith("_space_") for k in kwargs) or len(bases) != 1:
                continue  # the space is a view of its base: not handled
            base = bases[index]
            with graph.inserting_before(node):
                graph.call_function(target, (), {**kwargs, "space": base})
            for user in list(node.users):
                for use in list(user.users):
                    if use.target is torch.ops.aten.copy_.default and use.args[1] is user:
                        graph.erase_node(use)
                user.replace_all_uses_with(base)
                graph.erase_node(user)
            graph.erase_node(node)
        graph.lint()

    def uuid(self):
        return "fastmace-proto-in-place-into-saved-1"


def _workspace_shape(module, natoms):
    """(natoms, width) if the module's product can hold its input gradients too, else None."""
    second = module.second
    width = sum(mul * dim for _, mul, dim in second.segments_out)
    grads = sum(natoms * dim * int(getattr(second, f"weight_{k}").shape[0]) for _, _, dim, k in module.joined)
    return (natoms, width) if grads <= natoms * width else None


def install(natoms, device, dtype):
    """Patch GatedRows and compile_model for this process (see the module docstring); returns the workspaces."""
    from fastmace.model import compile as compiling
    from fastmace.rewrites.fusion import modules

    workspaces = []
    torch._inductor.config.post_grad_custom_post_pass = InPlaceIntoSaved()
    original_compile = compiling.compile_model

    def compile_model(*args, **kwargs):
        compiled = original_compile(*args, **kwargs)
        for module in getattr(compiled, "_orig_mod", compiled).modules():
            if isinstance(module, modules.GatedRows) and module.joined is not None:
                shape = _workspace_shape(module, natoms)
                if shape is not None:
                    module._fastmace_ws = torch.empty(shape, device=device, dtype=dtype)
                    workspaces.append(module._fastmace_ws)
        return compiled

    compiling.compile_model = compile_model
    original_forward = modules.GatedRows.forward

    def forward(self, x, y):
        ws = getattr(self, "_fastmace_ws", None)
        if ws is None or self.joined is None or self.gate_epilogue:
            return original_forward(self, x, y)
        outputs, n, like = self.first.output_rows(x, y)
        if ws.shape[0] != n or ws.dtype != like.dtype or ws.device != like.device:
            return original_forward(self, x, y)
        index, start, length = self.scalars
        scalars = outputs[index].narrow(1, start, length)
        if self.act_scalar is not None:
            scalars = self.act_scalar(scalars) * self.scalar_cst
        rows = [scalars]
        if self.gated:
            index, start, length = self.gates
            gates = outputs[index].narrow(1, start, length)
            if self.act_gate is not None:
                gates = self.act_gate(gates) * self.gate_cst
            for segment, offset, mul, dim in self.gated:
                gated = outputs[segment].view(n, dim, mul)
                gate = gates.narrow(-1, offset, mul).unsqueeze(-2)
                rows.append((gated * gate).reshape(n * dim, mul))
        if not self._any_triton(rows, n):
            return original_forward(self, x, y)
        second = self.second
        width = sum(mul * dim for _, mul, dim in second.segments_out)
        count = len(self.joined)
        out = _IntoWorkspace.apply(
            [offset for _, offset, _, _ in self.joined], [dim for _, _, dim, _ in self.joined], int(width),
            "tf32x3" if self.tf32x3 else "ieee", count, ws, *[rows[i] for i, _, _, _ in self.joined],
            *[getattr(second, f"weight_{k}") for _, _, _, k in self.joined],
            *[getattr(second, f"weight_t_{k}") for _, _, _, k in self.joined])
        return second.transpose_out(out)

    modules.GatedRows.forward = forward
    return workspaces
