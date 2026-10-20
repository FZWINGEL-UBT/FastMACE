"""The fusion family of optimisation round 1 (``plans/opt-1.md``, OPT-T02; ``docs/design/opt-T02-fusion.md``).

``RECORDS`` holds:

- ``gated_rows`` (stage A): in every interaction whose forward reads ``gated_rows`` (the folded
  MH-1 forward, ``fastmace.rewrites.mh1.modules.cutoff_folded_forward``) and that has merged
  residual products, MACE's ``GatedEquivariantBlock`` in the ir_mul layout and a block linear
  ``linear_2``, the chain residual products -> gate -> ``linear_2`` runs on per-segment rows
  (``modules.GatedRows``). It removes the joins and copies between them and their backward,
  about 1.9 ms of a 27 ms NbCl5 3x3x3 step on the RTX 5060 Ti (2-T20a). The products and the
  elementwise formulas are unchanged, so the record is ``exact``. It is opt-in
  (``RewriteConfig.gated_rows``, default off) until the A/B on both GPUs.

The self-check compares each chain with the unrewritten interaction's ``linear_2(gate(linear_1(x)
+ linear_res(y)))``, values and both input gradients, in the model's dtype (SPEC §6.4). The gate's
forward is pinned by sha256: the rows reproduce its arithmetic.
"""

from __future__ import annotations

from fastmace.model.capabilities import FACTORED_INTERACTIONS
from fastmace.rewrites.registry import Rewrite, Support, check_tensors

MACE_VERSIONS = ">=0.3.16,<0.3.17"
GATE_FORWARD = "mace.modules.gate:GatedEquivariantBlock.forward"
GATE_FORWARD_SHA256 = "a4343a8ad2ae966f97d49a0bad4577f677dcbb4bca77dd54f89b3f5cbee48d27"
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
                                getattr(interaction, "equivariant_nonlin", None), getattr(interaction, "linear_2", None))
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


def _selfcheck(original, rewritten, context):
    import torch

    generator = torch.Generator().manual_seed(context.seed)
    cases, dtype = [], None
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
        cases += [(f"interactions.{index}.gated_rows", a, b) for a, b in zip(*values)]
        dtype = dtype or weight.dtype
    return check_tensors(cases, dtype)


def _apply(model, context):
    return gate_rows(model)


RECORDS = {
    "gated_rows": Rewrite(
        name="gated_rows", applies_to=_support, requires=("block_linear", "merged_residual"), conflicts=(),
        mace_versions=MACE_VERSIONS, source_hashes={GATE_FORWARD: GATE_FORWARD_SHA256}, exactness="exact",
        default="off", deterministic=True, workload_constraints=(), apply=_apply, selfcheck=_selfcheck),
}

__all__ = ["GATE_FORWARD", "GATE_FORWARD_SHA256", "READS_GATED_ROWS", "RECORDS", "gate_rows"]
