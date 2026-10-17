"""The linear family: ``block_linear``, ``grouped_linear``, ``merged_residual`` (SPEC §6.2).

``RECORDS`` holds the three registry records (``fastmace.rewrites.registry.Rewrite``). Each acts
on the model in place and is checked against the unrewritten model by name (SPEC §6.4): values
and input gradients of every replaced module, on random inputs in the module's own dtype (the
weights are extracted in that dtype, so a float64 check of float32 weights would measure the
extraction's rounding, not the rewrite).

- ``block_linear`` replaces every cuEquivariance Linear (internal shared weights) by a
  ``BlockLinear``.
- ``grouped_linear`` regroups every ungrouped ``BlockLinear``; the result is the module the
  research engine builds with ``GROUPED_BLOCK_LINEAR`` set.
- ``merged_residual`` sets ``residual_linear`` on every interaction whose forward reads it
  (the forward function carries ``fastmace_residual_linear = True``; plan task 1a-T07's folded
  forward does), that divides by the density before ``linear_1`` (``density_before_linear``)
  and whose ``linear_1`` and ``linear_res`` are block linears. It requires ``density_first``,
  so it acts only where that rewrite applies.

No source is pinned: these records rely on cuEquivariance's ``Linear`` (its ``f``, the layout
transposes and the ir_mul core layout), not on MACE's code, and the self-check covers that.
The research engine's in-place variant (``INPLACE_BLOCK_LINEAR``; opt-in, slower) is not ported.
"""

from __future__ import annotations

from fastmace.model.capabilities import FACTORED_INTERACTIONS
from fastmace.rewrites.registry import Rewrite, Support, check_tensors

MACE_VERSIONS = ">=0.3.16,<0.3.17"
# Set by a forward function that reads interaction.residual_linear (1a-T07's folded forward).
READS_RESIDUAL_LINEAR = "fastmace_residual_linear"
_PROBE_ROWS = 7


def _modules(model):
    named = getattr(model, "named_modules", None)
    return list(named()) if named is not None else []


def _block_type():
    from fastmace.rewrites.linear import modules

    return modules


# --------------------------------------------------------------------------- apply


def block_linears(model):
    """Replace the applicable cuEquivariance Linear layers of ``model``; returns the replaced names."""
    lm = _block_type()
    replaced = []
    for name, module in _modules(model):
        for child_name, child in list(module.named_children()):
            replacement = lm.block_linear(child)
            if replacement is not None:
                setattr(module, child_name, replacement)
                replaced.append(f"{name}.{child_name}" if name else child_name)
    return replaced


def group_linears(model):
    """Regroup every ungrouped BlockLinear of ``model``; returns the regrouped names."""
    lm = _block_type()
    grouped = []
    for name, module in _modules(model):
        for child_name, child in list(module.named_children()):
            if isinstance(child, lm.BlockLinear) and not child.grouped:
                setattr(module, child_name, lm.regroup(child))
                grouped.append(f"{name}.{child_name}" if name else child_name)
    return grouped


def _reads_residual_linear(interaction):
    forward = getattr(getattr(interaction, "forward", None), "__func__", None)
    return bool(getattr(forward, READS_RESIDUAL_LINEAR, False))


def _mergeable(interaction):
    lm = _block_type()
    return (_reads_residual_linear(interaction) and bool(getattr(interaction, "density_before_linear", False))
            and isinstance(getattr(interaction, "linear_1", None), lm.BlockLinear)
            and isinstance(getattr(interaction, "linear_res", None), lm.BlockLinear))


def merge_residual_linears(model):
    """Attach ``residual_linear`` to every mergeable interaction of ``model``; returns their names."""
    lm = _block_type()
    merged = []
    for index, interaction in enumerate(getattr(model, "interactions", ())):
        if _mergeable(interaction):
            interaction.residual_linear = lm.ResidualBlockLinear(interaction.linear_1, interaction.linear_res)
            merged.append(f"interactions.{index}")
    return merged


# --------------------------------------------------------------------------- applies_to


def _block_linear_support(model):
    lm = _block_type()
    if any(lm.is_cueq_linear(module) or isinstance(module, lm.BlockLinear) for _, module in _modules(model)):
        return Support(True, "ok")
    return Support(False, "no cuEquivariance Linear with internal shared weights (an e3nn model, or not MACE)")


def _merged_residual_support(model):
    lm = _block_type()
    for interaction in getattr(model, "interactions", ()):
        if type(interaction).__name__ not in FACTORED_INTERACTIONS:
            continue
        layers = [getattr(interaction, name, None) for name in ("linear_1", "linear_res")]
        if all(layer is not None and (lm.is_cueq_linear(layer) or isinstance(layer, lm.BlockLinear))
               for layer in layers):
            return Support(True, "ok")
    return Support(False, f"no {FACTORED_INTERACTIONS} interaction with cuEquivariance linear_1 and linear_res")


# --------------------------------------------------------------------------- self-checks


def _probe(module, generator, width):
    import torch

    weight = module.weight_0
    x = torch.randn((_PROBE_ROWS, width), generator=generator, dtype=torch.float64)
    return x.to(device=weight.device, dtype=weight.dtype).requires_grad_()


def _values_and_gradients(function, inputs, grad):
    import torch

    with torch.enable_grad():
        out = function(*inputs)
        return (out.detach(), *torch.autograd.grad(out, inputs, grad))


def _compare(name, expected, got):
    """(name, expected, got) for the value and each input gradient; one module per name."""
    return [(name, a, b) for a, b in zip(expected, got)]


def _width(segments):
    return sum(mul * dim for _, mul, dim in segments)


def _reference(original, name):
    """The unrewritten module that the block linear at ``name`` replaced.

    An earlier rewrite may have wrapped that module: the MH-1 family's ``factor_species`` keeps
    each species embedding as ``<path>.embedding`` of its ``ProjectedEmbedding``. The reference is
    then the nearest enclosing path of the unrewritten model that is a cuEquivariance Linear.
    """
    lm = _block_type()
    parts = name.split(".")
    while parts:
        try:
            module = original.get_submodule(".".join(parts))
        except AttributeError:
            module = None
        if module is not None and (len(parts) == len(name.split(".")) or lm.is_cueq_linear(module)):
            return module
        parts.pop()
    raise AttributeError(f"the unrewritten model has no module {name} (nor a cuEquivariance Linear enclosing it)")


def _check_block_modules(original, rewritten, context, grouped_only):
    import torch

    lm = _block_type()
    generator = torch.Generator().manual_seed(context.seed)
    cases, dtype = [], None
    for name, module in _modules(rewritten):
        if not isinstance(module, lm.BlockLinear) or (grouped_only and not module.grouped):
            continue
        reference = _reference(original, name)
        x = _probe(module, generator, _width(module.segments_in))
        grad = torch.randn((_PROBE_ROWS, _width(module.segments_out)), generator=generator,
                           dtype=torch.float64).to(x)
        expected = _values_and_gradients(reference, (x,), grad)
        got = _values_and_gradients(module, (x,), grad)
        cases += _compare(name, expected, got)
        dtype = dtype or module.weight_0.dtype
    return check_tensors(cases, dtype)


def _block_linear_selfcheck(original, rewritten, context):
    return _check_block_modules(original, rewritten, context, grouped_only=False)


def _grouped_linear_selfcheck(original, rewritten, context):
    return _check_block_modules(original, rewritten, context, grouped_only=True)


def _merged_residual_selfcheck(original, rewritten, context):
    import torch

    generator = torch.Generator().manual_seed(context.seed)
    cases, dtype = [], None
    for index, interaction in enumerate(getattr(rewritten, "interactions", ())):
        merged = getattr(interaction, "residual_linear", None)
        if merged is None:
            continue
        reference = original.get_submodule(f"interactions.{index}")
        first_width = _width(merged.segments_in[:merged.first_count])
        x = _probe(merged, generator, first_width)
        y = _probe(merged, generator, _width(merged.segments_in) - first_width)
        grad = torch.randn((_PROBE_ROWS, _width(merged.segments_out)), generator=generator,
                           dtype=torch.float64).to(x)
        expected = _values_and_gradients(lambda a, b: reference.linear_1(a) + reference.linear_res(b), (x, y), grad)
        got = _values_and_gradients(merged, (x, y), grad)
        cases += _compare(f"interactions.{index}.residual_linear", expected, got)
        dtype = dtype or merged.weight_0.dtype
    return check_tensors(cases, dtype)


# --------------------------------------------------------------------------- records


def _apply_block_linear(model, context):
    return block_linears(model)


def _apply_grouped_linear(model, context):
    return group_linears(model)


def _apply_merged_residual(model, context):
    return merge_residual_linears(model)


RECORDS = {
    "block_linear": Rewrite(
        name="block_linear", applies_to=_block_linear_support, requires=(), conflicts=(),
        mace_versions=MACE_VERSIONS, source_hashes={}, exactness="rounding", default="auto",
        workload_constraints=(), apply=_apply_block_linear, selfcheck=_block_linear_selfcheck),
    "grouped_linear": Rewrite(
        name="grouped_linear", applies_to=_block_linear_support, requires=("block_linear",), conflicts=(),
        mace_versions=MACE_VERSIONS, source_hashes={}, exactness="rounding", default="auto",
        workload_constraints=(), apply=_apply_grouped_linear, selfcheck=_grouped_linear_selfcheck),
    "merged_residual": Rewrite(
        name="merged_residual", applies_to=_merged_residual_support, requires=("block_linear", "density_first"),
        conflicts=(), mace_versions=MACE_VERSIONS, source_hashes={}, exactness="rounding", default="auto",
        workload_constraints=(), apply=_apply_merged_residual, selfcheck=_merged_residual_selfcheck),
}

__all__ = ["READS_RESIDUAL_LINEAR", "RECORDS", "block_linears", "group_linears", "merge_residual_linears"]
