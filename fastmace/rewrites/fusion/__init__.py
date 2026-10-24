"""The fusion family of optimisation round 1 (``plans/opt-1.md``, OPT-T02; ``docs/design/opt-T02-fusion.md``).

``RECORDS`` holds ``gated_rows``, ``segment_gemm`` and ``grouped_launch``:

- ``gated_rows`` (stage A): in every interaction whose forward reads ``gated_rows`` (the folded
  MH-1 forward, ``fastmace.rewrites.mh1.modules.cutoff_folded_forward``) and that has merged
  residual products, MACE's ``GatedEquivariantBlock`` in the ir_mul layout and a block linear
  ``linear_2``, the chain residual products -> gate -> ``linear_2`` runs on per-segment rows
  (``modules.GatedRows``). It removes the joins and copies between them and their backward,
  about 1.9 ms of a 27 ms NbCl5 3x3x3 step on the RTX 5060 Ti (2-T20a). The products and the
  elementwise formulas are unchanged, so the record is ``exact``. On by default wherever
  ``merged_residual`` is (owner decision, 20 October 2026, after the A/B on both GPUs).

- ``segment_gemm`` (stage B, first part): ``linear_2``'s products in those chains write straight into its
  joined output, and their input gradients read the gradient's rows in place
  (``fastmace.kernels.triton.segment_gemm``), where the shared shape rule
  (``fastmace.kernels.triton.tile_rule``) picks the Triton GEMM; cuBLAS otherwise. It removes the join
  before the symmetric contraction and its backward. ``rounding`` (another GEMM), deterministic (no
  atomics). On by default with ``gated_rows`` in float32 plans that are not reproducible (owner decision,
  20 October 2026); float64 and ``reproducible=True`` keep the stock products and join.

- ``grouped_launch`` (OPT-T03 candidate 2, ``docs/design/opt-T03-small-m.md``): in every block linear
  (``BlockLinear``, ``ResidualBlockLinear``, also inside the ``gated_rows`` chains), the products that the
  shared shape rule calls small run together in one Triton launch (``fastmace.kernels.triton.grouped_gemm``),
  forward and input gradient; the others keep cuBLAS. ``rounding`` (another GEMM), deterministic. Opt-in
  (``RewriteConfig.grouped_launch``).

The self-check compares each chain with the unrewritten interaction's ``linear_2(gate(linear_1(x)
+ linear_res(y)))``, values and both input gradients, in the model's dtype (SPEC §6.4). The gate's
forward is pinned by sha256: the rows reproduce its arithmetic.
"""

from __future__ import annotations

from fastmace.model.capabilities import FACTORED_INTERACTIONS
from fastmace.rewrites import memory as memory_terms
from fastmace.rewrites.registry import CaseErrors, Rewrite, Support, check_tensors

MACE_VERSIONS = ">=0.3.16,<0.3.17"
GATE_FORWARD = "mace.modules.gate:GatedEquivariantBlock.forward"
GATE_FORWARD_SHA256 = "a4343a8ad2ae966f97d49a0bad4577f677dcbb4bca77dd54f89b3f5cbee48d27"
RESHAPE_FORWARD = "mace.modules.irreps_tools:reshape_irreps.forward"
RESHAPE_FORWARD_SHA256 = "abe2111a66a143e905fbe9a382120623c95c959c36967dd7e76fc0352301b010"
# Set by a forward function that reads interaction.gated_rows (the folded MH-1 forward).
READS_GATED_ROWS = "fastmace_gated_rows"
_PROBE_ROWS = 7


def _modules():
    from fastmace.rewrites.fusion import modules

    return modules


def _reads_gated_rows(interaction):
    forward = getattr(getattr(interaction, "forward", None), "__func__", None)
    return bool(getattr(forward, READS_GATED_ROWS, False))


def gate_rows(model):
    """Attach ``gated_rows`` to every interaction that has the chain; returns their names."""
    m = _modules()
    names = []
    for index, interaction in enumerate(getattr(model, "interactions", ())):
        if not _reads_gated_rows(interaction):
            continue
        chain, _ = m.gated_rows(getattr(interaction, "residual_linear", None),
                                getattr(interaction, "equivariant_nonlin", None),
                                getattr(interaction, "linear_2", None))
        if chain is not None:
            interaction.gated_rows = chain
            names.append(f"interactions.{index}")
    return names


def _support(model):
    for interaction in getattr(model, "interactions", ()):
        if (type(interaction).__name__ in FACTORED_INTERACTIONS
                and type(getattr(interaction, "equivariant_nonlin", None)).__name__ == "GatedEquivariantBlock"):
            return Support(True, "ok")
    return Support(False, f"no {FACTORED_INTERACTIONS} interaction with MACE's GatedEquivariantBlock")


def _chain_cases(original, rewritten, context):
    """(errors, dtype): each chain against the unrewritten interaction, values and both input gradients, as a
    ``CaseErrors`` (each case's tensors are freed before the next interaction's probe)."""
    import torch

    generator = torch.Generator().manual_seed(context.seed)
    cases, dtype = CaseErrors(), None
    for index, interaction in enumerate(getattr(rewritten, "interactions", ())):
        chain = getattr(interaction, "gated_rows", None)
        if chain is None:
            continue
        reference = original.get_submodule(f"interactions.{index}")
        first = chain.first
        weight = first.weight_0
        widths = [sum(m * d for _, m, d in part) for part in
                  (first.segments_in[:first.first_count], first.segments_in[first.first_count:])]
        x, y = (torch.randn((_PROBE_ROWS, width), generator=generator, dtype=torch.float64)
                .to(device=weight.device, dtype=weight.dtype).requires_grad_() for width in widths)
        width_out = sum(m * d for _, m, d in chain.second.segments_out)
        grad = torch.randn((_PROBE_ROWS, width_out), generator=generator, dtype=torch.float64).to(x)

        def stock(a, b, reference=reference):
            return reference.linear_2(reference.equivariant_nonlin(reference.linear_1(a) + reference.linear_res(b)))

        with torch.enable_grad():
            values = []
            for function in (stock, chain):
                out = function(x, y)
                values.append((out.detach(), *torch.autograd.grad(out, (x, y), grad)))
        for a, b in zip(*values):
            cases.add(f"interactions.{index}.gated_rows", a, b)
        del values, x, y, grad
        dtype = dtype or weight.dtype
    return cases, dtype


def _selfcheck(original, rewritten, context):
    errors, dtype = _chain_cases(original, rewritten, context)
    return errors.result(dtype)


def join_in_place(model):
    """Let every ``gated_rows`` chain's ``linear_2`` write into its joined output, and make the interaction's
    ``reshape`` a view where it is one (without it the reshape copies the joined output again); returns the
    names of the interactions."""
    m = _modules()
    names = []
    for index, interaction in enumerate(getattr(model, "interactions", ())):
        chain = getattr(interaction, "gated_rows", None)
        if chain is None or not chain.join_in_place()[0]:
            continue
        view, _ = m.irreps_view(getattr(interaction, "reshape", None))
        if view is not None:
            interaction.reshape = view
        names.append(f"interactions.{index}")
    return names


def _segment_support(model):
    return _support(model)


def _segment_selfcheck(original, rewritten, context):
    """The chains against the unrewritten interactions (as ``gated_rows``), and each ``linear_2``'s joined
    products against its own products and join on enough atoms that the shape rule picks Triton for every
    product on this device, values and the input gradients."""
    import torch

    from fastmace.kernels.triton import tile_rule
    from fastmace.kernels.triton.segment_gemm import joined_products
    from fastmace.rewrites.linear.modules import join_rows

    cases, dtype = _chain_cases(original, rewritten, context)
    generator = torch.Generator().manual_seed(context.seed + 1)
    for index, interaction in enumerate(getattr(rewritten, "interactions", ())):
        view = getattr(interaction, "reshape", None)
        if type(view).__name__ != "IrrepsView":
            continue
        stock = original.get_submodule(f"interactions.{index}").reshape
        probe = torch.randn((5, view.components * view.mul), generator=generator, dtype=torch.float64)
        probe = probe.to(device=interaction.linear_2.weight_0.device, dtype=interaction.linear_2.weight_0.dtype)
        cases.add(f"interactions.{index}.reshape", stock(probe), view(probe))
    for index, interaction in enumerate(getattr(rewritten, "interactions", ())):
        chain = getattr(interaction, "gated_rows", None)
        if chain is None or chain.joined is None:
            continue
        second = chain.second
        weight = second.weight_0
        sms = tile_rule.device_sms(weight.device) or 1
        dims = [dim for _, _, dim, _ in chain.joined]
        atoms = -(-tile_rule.MIN_TRITON_WAVES * sms * tile_rule.BLOCK_M // min(dims))
        rows = [torch.randn((atoms * second.segments_in[i][2], second.segments_in[i][1]), generator=generator,
                            dtype=torch.float64).to(weight).requires_grad_() for i, _, _, _ in chain.joined]
        width = sum(mul * dim for _, mul, dim in second.segments_out)
        grad = torch.randn((atoms, width), generator=generator, dtype=torch.float64).to(weight)
        full = [None] * len(second.segments_in)
        for (i, _, _, _), x in zip(chain.joined, rows):
            full[i] = x

        def stock(*xs, second=second):
            return join_rows(second.segments_out, second.products([*xs]), atoms, xs[0])

        def joined(*xs, second=second, chain=chain):
            return joined_products(list(xs), [getattr(second, f"weight_{k}") for _, _, _, k in chain.joined],
                                   [getattr(second, f"weight_t_{k}") for _, _, _, k in chain.joined],
                                   [o for _, o, _, _ in chain.joined], [d for _, _, d, _ in chain.joined], width)

        with torch.enable_grad():
            values = []
            for function, inputs in ((stock, full), (joined, rows)):
                out = function(*inputs)
                values.append((out.detach(), *torch.autograd.grad(out, rows, grad)))
        for a, b in zip(*values):
            cases.add(f"interactions.{index}.gated_rows.joined", a, b)
        # Each interaction's probe rows, products and gradients are sized to fill the device (several GB on a
        # wide model); free them before the next interaction's (5-T03, rule 12's build estimate).
        del values, rows, full, grad
    return cases.result(dtype)


def _block_modules(model):
    from fastmace.rewrites.linear.modules import BlockLinear, ResidualBlockLinear

    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, (BlockLinear, ResidualBlockLinear))]


def group_launches(model):
    """Let every block linear run its small products in one launch; returns the names of the modules."""
    import torch

    from fastmace.kernels.triton import tile_rule

    names = []
    for name, module in _block_modules(model):
        weight = module.weight_0
        module.grouped_launch = True
        # Read once at setup: the forward evaluates the shape rule while torch.compile traces it.
        module.sms = tile_rule.device_sms(weight.device) if weight.dtype == torch.float32 else None
        names.append(name)
    return names


def _grouped_launch_support(model):
    """Where block_linear applies (it is asked before block_linear has converted the model)."""
    from fastmace.rewrites import linear

    return linear.RECORDS["block_linear"].applies_to(model)


def _grouped_launch_selfcheck(original, rewritten, context):
    """Each block linear's products through ``block_products`` with the small ones grouped, against one
    product at a time, on 7 atoms' rows (every product small), values and input gradients."""
    import torch

    from fastmace.rewrites.linear.modules import ConstantMatmul, block_products

    generator = torch.Generator().manual_seed(context.seed + 3)
    cases, dtype = [], None
    for name, module in _block_modules(rewritten):
        count = len(module.block_ends)
        weights = [getattr(module, f"weight_{k}") for k in range(count)]
        dtype = dtype or weights[0].dtype
        lefts = [torch.randn((7 * (k + 1), w.shape[0]), generator=generator, dtype=torch.float64)
                 .to(w).requires_grad_() for k, w in enumerate(weights)]
        grads = [torch.randn((7 * (k + 1), w.shape[1]), generator=generator, dtype=torch.float64).to(w)
                 for k, w in enumerate(weights)]

        def single(xs, module=module, weights=weights):
            return [ConstantMatmul.apply(x, w, getattr(module, f"weight_t_{k}")) for k, (x, w) in
                    enumerate(zip(xs, weights))]

        with torch.enable_grad():
            values = []
            for function in (single, lambda xs, module=module: block_products(module, xs)):
                outs = function(lefts)
                total = sum((o * g).sum() for o, g in zip(outs, grads))
                values.append((*[o.detach() for o in outs], *torch.autograd.grad(total, lefts)))
        cases += [(f"{name}.grouped_launch", a, b) for a, b in zip(*values)]
    return check_tensors(cases, dtype)


def _apply_grouped_launch(model, context):
    return group_launches(model)


def _apply(model, context):
    return gate_rows(model)


def _apply_segment(model, context):
    return join_in_place(model)


RECORDS = {
    "gated_rows": Rewrite(
        name="gated_rows", applies_to=_support, requires=("block_linear", "merged_residual"), conflicts=(),
        mace_versions=MACE_VERSIONS, source_hashes={GATE_FORWARD: GATE_FORWARD_SHA256}, exactness="exact",
        default="auto", deterministic=True, workload_constraints=(), apply=_apply, selfcheck=_selfcheck),
    "segment_gemm": Rewrite(
        name="segment_gemm", applies_to=_segment_support, requires=("gated_rows",), conflicts=(),
        mace_versions=MACE_VERSIONS, source_hashes={RESHAPE_FORWARD: RESHAPE_FORWARD_SHA256}, exactness="rounding",
        default="auto", deterministic=True, workload_constraints=(), apply=_apply_segment,
        selfcheck=_segment_selfcheck, selfcheck_memory=memory_terms.segment_gemm_selfcheck),
    "grouped_launch": Rewrite(
        name="grouped_launch", applies_to=_grouped_launch_support, requires=("block_linear",), conflicts=(),
        mace_versions=MACE_VERSIONS, source_hashes={}, exactness="rounding", default="off", deterministic=True,
        workload_constraints=(), apply=_apply_grouped_launch, selfcheck=_grouped_launch_selfcheck),
}

__all__ = ["GATE_FORWARD", "GATE_FORWARD_SHA256", "READS_GATED_ROWS", "RECORDS", "gate_rows", "join_in_place"]
