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
not IEEE-equivalent under cancellation (``docs/validation/speed/opt-T01-tensor-cores.md`` §7); ``deterministic`` (one
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


def route(model, cueq_projected="auto", sc_generated_config="auto"):
    """Set ``tf32x3`` on every module that has atom GEMMs, and ``tf32x3_backend``, the backend ``tf32x3_mm`` uses on
    the module's device ("cutlass" on compute capability 8.0, else "triton"; ``GatedRows`` then takes the block
    products instead of its joined Triton GEMM); returns their names."""
    import torch

    from fastmace.kernels.triton.tf32x3_gemm import backend

    names = []
    for name, module in _targets(model):
        tensor = next(iter(module.parameters()), None)
        if tensor is None:
            tensor = next(iter(module.buffers()), None)
        device = tensor.device if tensor is not None else torch.device("cpu")
        module.tf32x3 = True
        module.tf32x3_backend = backend(device)
        if type(module).__name__ == "FirstInteractionReassociation":  # idea 31: kernels.cutlass.first_div
            from fastmace.kernels.cutlass import first_div

            module.first_div = (module.tf32x3_backend == "cutlass"
                                and first_div.first_div_default(torch.cuda.get_device_name(device)))
        if type(module).__name__ == "SpeciesSkip":  # idea 31: kernels.cutlass.skip_select
            from fastmace.kernels.cutlass import skip_select

            module.skip_select = (module.tf32x3_backend == "cutlass"
                                  and skip_select.skip_select_default(torch.cuda.get_device_name(device)))
        if type(module).__name__ == "BlockLinear" and module.tf32x3_backend == "cutlass":
            from fastmace.kernels.cutlass import linear_div, segment_rows

            module.linear_div = linear_div.linear_div_default(torch.cuda.get_device_name(device))

            if segment_rows.segment_rows_default(device):
                count = len(module.block_ends)
                module.segment_plan = segment_rows.plan_of(
                    module.segments_in, module.segments_out, module.block_ends,
                    [getattr(module, f"weight_{k}") for k in range(count)])
        names.append(name)
    _route_compose_skip(model)
    from fastmace.rewrites import cueq_projection

    cueq_projection.route(model, cueq_projected)  # CUEQ_PROJECTED, RewriteConfig.cueq_projected
    from fastmace.rewrites import sc_generated

    sc_generated.route(model, sc_generated_config)  # SC_GENERATED, RewriteConfig.sc_generated
    return names


def _route_compose_skip(model):
    """``kernels.triton.compose_skip`` on every generic interaction that qualifies (``compose_skip_products``), on
    CUDA where ``compose_skip_default`` says so for the GPU."""
    import torch

    from fastmace.kernels.triton import compose_skip
    from fastmace.rewrites.generic.modules import compose_skip_products

    for module in model.modules():
        reassociation = getattr(module, "first_reassociation", None)
        if reassociation is None or not hasattr(reassociation, "compose_skip"):
            continue
        reassociation.compose_skip = None
        reassociation.compose_radial = None
        if getattr(reassociation, "nested_rotation", None) is not None:  # routed again: the plain table back
            from fastmace.rewrites.radial import nested

            nested.release([module])
        reassociation.compose_joined = False
        reassociation.count_plan = False
        reassociation.joined_layout = False
        device = reassociation.columns.device
        if device.type != "cuda" or not compose_skip.compose_skip_default(torch.cuda.get_device_name(device)):
            continue
        products = compose_skip_products(module)
        if products is not None:
            linear, skip = products
            reassociation.compose_skip = (module.linear, linear, module.skip_tp, skip)
            reassociation.count_plan = compose_skip.count_plan_default(torch.cuda.get_device_name(device))
            _route_nested(module, reassociation, device)
            if not compose_skip.compose_fast_default(torch.cuda.get_device_name(device)):
                continue
            reassociation.compose_joined = True
            reassociation.joined_layout = compose_skip.joined_layout_default(torch.cuda.get_device_name(device))
            reassociation.compose_radial = getattr(module, "radial_last", None)
            table = reassociation._buffers.get("species_feats")
            if table is not None and reassociation.compose_radial is not None:
                reassociation.bind(table)


def _route_nested(module, reassociation, device):
    """``rewrites.radial.nested`` (``NESTED_RADIAL``) on a first interaction that composes its skip: where the GPU
    says so, the tabled forward runs, the interaction's table is per species pair over the reassociation's and the
    skip's superset columns with 64 folded channels, and every table row passes ``nested.install``'s check. Sets
    ``reassociation.nested_rotation`` (the maps composed into the products); the table keeps the rotated copies."""
    import sys

    import torch

    from fastmace.kernels.triton.tabled_outer_sum import tabled_forward
    from fastmace.rewrites.generic import modules as gm
    from fastmace.rewrites.generic.modules import fc_last_weight
    from fastmace.rewrites.radial import nested

    if not hasattr(reassociation, "nested_rotation"):
        return
    if not nested.nested_default(torch.cuda.get_device_name(device)) or not tabled_forward(device):
        return
    # every whole-system call of the interaction must take the fused walk (fused_first_applies), the only reader
    if gm.FIRST_FULL_ROWS or not reassociation.segmented or not gm.fused_first_backward(device):
        return
    table, last = getattr(module, "radial_table", None), getattr(module, "radial_last", None)
    skip = reassociation.compose_skip[2]
    if table is None or last is None or getattr(table, "coef", None) is None or table.coef.dtype != torch.float32:
        return
    columns = reassociation.columns.cpu().tolist()
    if (fc_last_weight(last).shape[-1] != nested.COLUMNS or table.columns.cpu().tolist() != columns
            or getattr(skip, "columns", None) is None or skip.columns.cpu().tolist() != columns):
        return
    if nested.NESTED_RADIAL is True:
        nested.require(table)  # forced on: a table that needs more far columns fails loudly
    if not nested.install(table):
        import logging

        worst = max((max(e) for e in table.nested_errors), default=None)
        logging.getLogger(__name__).warning("NESTED_RADIAL refused: far error %s > %s", worst, nested.TOLERANCE)
        return
    reassociation.nested_rotation = nested.rotation(table, len(columns), len(columns))
    nested.activate(table)  # one table on the device: the rotated rows
    worst = max(max(e) for e in table.nested_errors)
    print(f"fastmace: NESTED_RADIAL on ({table.coef.shape[0]} rows, far error {worst:.2e}, r_far {table.nested_r_far})",
          file=sys.stderr, flush=True)


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
    return route(model, context.config.cueq_projected, context.config.sc_generated)


RECORDS = {
    "tf32x3_gemms": Rewrite(
        name="tf32x3_gemms", applies_to=_support, requires=("block_linear",), conflicts=("grouped_launch",),
        mace_versions=MACE_VERSIONS, source_hashes={}, exactness="rounding", default="off", deterministic=True,
        workload_constraints=(), apply=_apply, selfcheck=_selfcheck),
}

__all__ = ["RECORDS", "route"]
