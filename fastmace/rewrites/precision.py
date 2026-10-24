"""The ``tf32x3_gemms`` record: every atom GEMM as 3xTF32 (plan opt-4, OPT4-T03 stage 1; opt-in, accuracy first).

In every module that multiplies atom rows by a weight, the product and its input gradients run through
``fastmace.kernels.triton.tf32x3_gemm`` (Triton ``tl.dot(input_precision="tf32x3")``) instead of cuBLAS IEEE or
the IEEE Triton GEMMs:

- block linears (``BlockLinear``, ``ResidualBlockLinear``, also inside the ``gated_rows`` chains);
- ``segment_gemm``'s joined products (``GatedRows``), all of them in Triton, since the cuBLAS products the
  shape rule would keep are IEEE;
- the generic species skip (``SpeciesSkip``) and the reassociated first interaction's per-path products
  (``FirstInteractionReassociation``).

The edge-level radial networks keep their GEMMs. No global TF32 flag is set. The record is ``rounding``, and
not IEEE-equivalent under cancellation (``docs/validation/opt-T01-tensor-cores.md`` §7); ``deterministic`` (one
fixed tile, no atomics). It is applied last of the module records, after the species family has built its
composed block linears. Opt-in only (``RewriteConfig.tf32x3_gemms``): the planner refuses it off float32 and
below compute capability 8.0; making it a default would need the owner's capability rule (``plans/opt-4.md`` §4).
On the CPU every product runs in PyTorch, bitwise the stock path.
"""

from __future__ import annotations

from fastmace.rewrites.registry import CaseErrors, Rewrite

MACE_VERSIONS = ">=0.3.16,<0.3.17"
_TYPES = ("BlockLinear", "ResidualBlockLinear", "GatedRows", "SpeciesSkip", "FirstInteractionReassociation")


def _targets(model):
    return [(name, module) for name, module in model.named_modules() if type(module).__name__ in _TYPES]


def route(model):
    """Set ``tf32x3`` on every module that has atom GEMMs; returns their names."""
    names = []
    for name, module in _targets(model):
        module.tf32x3 = True
        names.append(name)
    return names


def _support(model):
    from fastmace.rewrites import linear

    return linear.RECORDS["block_linear"].applies_to(model)


def _selfcheck(original, rewritten, context):
    """Each rerouted product against the IEEE product of the same operands, values and input gradients: the
    block linears' products (``block_products``), ``segment_gemm``'s joined products and the species skip's."""
    import torch

    from fastmace.rewrites.linear.modules import ConstantMatmul, block_products

    generator = torch.Generator().manual_seed(context.seed + 5)
    cases, dtype = CaseErrors(), None

    def probe(rows, columns, like):
        return torch.randn((rows, columns), generator=generator, dtype=torch.float64).to(like)

    for name, module in _targets(rewritten):
        kind = type(module).__name__
        if kind in ("BlockLinear", "ResidualBlockLinear"):
            weights = [getattr(module, f"weight_{k}") for k in range(len(module.block_ends))]
            dtype = dtype or weights[0].dtype
            lefts = [probe(7 * (k + 1), w.shape[0], w).requires_grad_() for k, w in enumerate(weights)]
            grads = [probe(7 * (k + 1), w.shape[1], w) for k, w in enumerate(weights)]

            def stock(xs, module=module, weights=weights):
                return [ConstantMatmul.apply(x, w, getattr(module, f"weight_t_{k}"))
                        for k, (x, w) in enumerate(zip(xs, weights))]

            with torch.enable_grad():
                values = []
                for function in (stock, lambda xs, module=module: block_products(module, xs)):
                    outs = function(lefts)
                    total = sum((o * g).sum() for o, g in zip(outs, grads))
                    values.append((*[o.detach() for o in outs], *torch.autograd.grad(total, lefts)))
            for a, b in zip(*values):
                cases.add(f"{name}.tf32x3", a, b)
        elif kind == "GatedRows" and module.joined is not None:
            from fastmace.kernels.triton.segment_gemm import joined_products

            second = module.second
            weights = [getattr(second, f"weight_{k}") for _, _, _, k in module.joined]
            weights_t = [getattr(second, f"weight_t_{k}") for _, _, _, k in module.joined]
            offsets = [o for _, o, _, _ in module.joined]
            dims = [d for _, _, d, _ in module.joined]
            width = sum(mul * dim for _, mul, dim in second.segments_out)
            rows = [probe(7 * d, w.shape[0], w).requires_grad_() for d, w in zip(dims, weights)]
            grad = probe(7, width, weights[0])
            with torch.enable_grad():
                values = []
                for precision in ("ieee", "tf32x3"):
                    out = joined_products(rows, weights, weights_t, offsets, dims, width, precision)
                    values.append((out.detach(), *torch.autograd.grad(out, rows, grad)))
            for a, b in zip(*values):
                cases.add(f"{name}.joined.tf32x3", a, b)
        elif kind == "SpeciesSkip":
            from fastmace.kernels.triton.tf32x3_gemm import tf32x3_matmul

            for index in range(len(module.pairs)):
                weight = getattr(module, f"w{index}")
                left = probe(7, weight.shape[0], weight).requires_grad_()
                grad = probe(7, weight.shape[1], weight)
                with torch.enable_grad():
                    values = []
                    for function in (torch.matmul, tf32x3_matmul):
                        out = function(left, weight)
                        values.append((out.detach(), *torch.autograd.grad(out, (left,), grad)))
                for a, b in zip(*values):
                    cases.add(f"{name}.w{index}.tf32x3", a, b)
    return cases.result(dtype)


def _apply(model, context):
    return route(model)


RECORDS = {
    "tf32x3_gemms": Rewrite(
        name="tf32x3_gemms", applies_to=_support, requires=("block_linear",), conflicts=("grouped_launch",),
        mace_versions=MACE_VERSIONS, source_hashes={}, exactness="rounding", default="off", deterministic=True,
        workload_constraints=(), apply=_apply, selfcheck=_selfcheck),
}

__all__ = ["RECORDS", "route"]
