"""The generic family: rewrites for MACE-MP-0, the density family and MACE-OFF's later blocks (SPEC §6.2).

``RECORDS`` holds five registry records (``fastmace.rewrites.registry``), each a dict of the §6.1
fields plus the ``RewriteConfig`` field that switches it (docs/design/switch-map.md):

- ``generic_species_skip`` (``species_skip``): every generic skip connection restricted to the
  structure's species (``SpeciesSkip``); exact, fixed species superset;
- ``generic_reassociate`` (``reassociate_first``): the first interaction summed per receiver and
  sender species before its last radial layer; only when the first interaction is a generic
  class and its inputs are species-only;
- ``generic_segmented_first_sum`` (``segmented_first_sum``): that sum as MH-1's run-segmented
  Triton kernel, with each receiver's edges grouped by sender species; the research engine's
  GENERIC_REASSOCIATE builds its reassociation with ``segmented=SEGMENTED_FIRST_SUM``;
- ``generic_fused_radial_backward`` (``fused_folded_backward``): the last radial layer through
  ``folded_linear`` (Triton input gradient on CUDA, PyTorch on the CPU), in every generic
  interaction that is not reassociated;
- ``generic_streaming`` (``streaming_conv``): the receiver-streaming OEq convolution in every
  generic interaction with an OEq-fused convolution that is not reassociated (hybrid backend).

Every record gives each generic interaction (``modules.GENERIC_INTERACTIONS``, source pinned by
sha256) the replacement forward ``modules.generic_interaction_forward`` and sets its own
attribute; the forward reads them all. Records that depend on the structure read the one-hot node
attributes from ``Context.inputs["node_attrs"]`` (rows of the structure, or of the declared
species superset; only which columns are present matters).

The self-checks compare every interaction a record changed with the original interaction of the
same name (SPEC §6.4): both outputs (message and skip connection) and the input gradients (node
features, or for species-only first inputs the per-species feature table; edge attributes, edge
features and the cutoff), with and without a cutoff, on random inputs in the module's dtype.

MACE-OFF's first block (``RealAgnosticInteractionBlock``) is not generic, so
``generic_reassociate`` never applies to MACE-OFF; the other three act on its later blocks.

The reassociation and streaming modules come from the MH-1 family
(``fastmace.rewrites.mh1.modules``, plan task 1a-T07); they are imported when a record is
applied. ``attach_streaming_kernels`` registers a ``streaming_count`` buffer that counts every
edge until the engine binds its edge counter (``fastmace.rewrites.mh1.bind_streaming_count``).
"""

from __future__ import annotations

import hashlib
import inspect
import types

from fastmace.model.capabilities import first_inputs_species_only
from fastmace.rewrites.generic import modules as gm
from fastmace.rewrites.registry import Support, check_tensors

MACE_VERSIONS = ">=0.3.16,<0.3.17"
NODE_ATTRS = "node_attrs"  # Context.inputs key of the structure's one-hot node attributes
_NODES, _EDGES = 9, 40


def _generic_interactions(model):
    """[(index, interaction, kind)] of the model's GENERIC_INTERACTIONS blocks."""
    out = []
    for index, interaction in enumerate(getattr(model, "interactions", ())):
        entry = gm.GENERIC_INTERACTIONS.get(type(interaction).__name__)
        if entry is not None:
            out.append((index, interaction, entry[0]))
    return out


def _prepare(interaction, kind):
    """Give ``interaction`` the generic forward (once); refuses a forward whose source differs from the pin."""
    if getattr(interaction, "generic_kind", None) == kind:
        return
    expected = gm.GENERIC_INTERACTIONS[type(interaction).__name__][1]
    digest = hashlib.sha256(inspect.getsource(type(interaction).forward).encode()).hexdigest()
    if digest != expected:
        raise ValueError(f"the generic rewrites support the mace-torch 0.3.16 {type(interaction).__name__}.forward "
                         f"only (source sha256 {digest})")
    interaction.generic_kind = kind
    interaction.forward = types.MethodType(gm.generic_interaction_forward, interaction)


def _columns(context, name):
    """The superset's node_attrs columns (``Context.inputs["species_columns"]``, SPEC §8.5), else the
    columns of the species in the structure's one-hot node attributes."""
    from fastmace.rewrites.mh1 import superset_columns

    node_attrs = context.inputs.get(NODE_ATTRS)
    if node_attrs is None:
        raise ValueError(f"{name} needs the structure's one-hot node attributes in Context.inputs[{NODE_ATTRS!r}]")
    return superset_columns(node_attrs, context.inputs.get("species_columns"))


def _device(module):
    return next(module.parameters()).device


def _has_fc_radial(interaction):
    try:
        gm.fc_split(interaction.conv_tp_weights)
    except (AttributeError, ValueError):
        return False
    return True


def _oeq_fused(interaction):
    return hasattr(interaction, "conv_fusion") and hasattr(getattr(interaction, "conv_tp", None), "input_args")


# --------------------------------------------------------------------------- apply


def apply_species_skip(model, context):
    """Restrict every generic skip connection to the structure's species; returns the changed names."""
    columns = _columns(context, "generic_species_skip")
    changed = []
    for index, interaction, kind in _generic_interactions(model):
        _prepare(interaction, kind)
        replacement = gm.species_skip(interaction.skip_tp, columns.to(_device(interaction)))
        if replacement is not None:
            interaction.skip_tp = replacement
            changed.append(f"interactions.{index}")
    return changed


def apply_reassociate(model, context):
    """Reassociate the first interaction if it is generic and its inputs are species-only; returns the names."""
    from fastmace.rewrites.mh1 import species_table
    from fastmace.rewrites.mh1.modules import FirstInteractionReassociation, probe_first_tensor_product

    columns = _columns(context, "generic_reassociate")
    generic = _generic_interactions(model)
    if not generic or generic[0][0] != 0:
        return []
    if not first_inputs_species_only(model):
        raise ValueError("generic_reassociate requires species-only first-interaction inputs")
    _, interaction, kind = generic[0]
    _prepare(interaction, kind)
    interaction.radial_hidden, interaction.radial_last = gm.fc_split(interaction.conv_tp_weights)
    channels, ls, gammas, ir_mul = probe_first_tensor_product(interaction)
    table = species_table(model, interaction, columns, len(context.inputs[NODE_ATTRS]),
                          context.inputs.get("joint_features"))
    interaction.first_reassociation = FirstInteractionReassociation(
        columns.to(_device(interaction)), channels, ls, gammas, ir_mul, species_feats=table)
    return ["interactions.0"]


def apply_segmented_first_sum(model, context):
    """Sum the reassociated generic first interaction with the run-segmented Triton kernel; returns its name."""
    from fastmace.rewrites.mh1 import segment_first_sum

    generic = _generic_interactions(model)
    reassociation = getattr(model.interactions[0], "first_reassociation", None) if generic else None
    if reassociation is None:
        raise ValueError("generic_segmented_first_sum requires the reassociated generic first interaction "
                         "(generic_reassociate)")
    return segment_first_sum(model)


def apply_fused_radial_backward(model, context):
    """Route the last radial layer of every non-reassociated generic interaction through folded_linear."""
    changed = []
    for index, interaction, kind in _generic_interactions(model):
        if getattr(interaction, "first_reassociation", None) is not None:
            continue
        _prepare(interaction, kind)
        interaction.radial_hidden, interaction.radial_last = gm.fc_split(interaction.conv_tp_weights)
        interaction.fused_radial_backward = True
        changed.append(f"interactions.{index}")
    return changed


def apply_streaming(model, context):
    """Attach the receiver-streaming kernels to every non-reassociated generic OEq-fused convolution."""
    changed = []
    for index, interaction, kind in _generic_interactions(model):
        if getattr(interaction, "first_reassociation", None) is not None or not _oeq_fused(interaction):
            continue
        from fastmace.rewrites.mh1.modules import attach_streaming_kernels

        _prepare(interaction, kind)
        attach_streaming_kernels(interaction)
        changed.append(f"interactions.{index}")
    return changed


# --------------------------------------------------------------------------- applies_to

_NOT_GENERIC = "no MACE-MP-0, density-family or MACE-OFF residual interaction"


def _support_species_skip(model):
    generic = _generic_interactions(model)
    if not generic:
        return Support(False, _NOT_GENERIC)
    if any(gm.scalar_second_input(getattr(i, "skip_tp", None)) for _, i, _ in generic):
        return Support(True, "ok")
    return Support(False, "no generic skip connection with one-hot scalar node attributes")


def _support_reassociate(model):
    interactions = list(getattr(model, "interactions", ()))
    if not _generic_interactions(model):
        return Support(False, _NOT_GENERIC)
    first = type(interactions[0]).__name__
    if first not in gm.GENERIC_INTERACTIONS:
        return Support(False, f"the first interaction {first} is not a generic class "
                              f"(MACE-OFF: the generic rewrites act on its later blocks only)")
    if not first_inputs_species_only(model):
        return Support(False, "first-interaction inputs are not species-only")
    if not _has_fc_radial(interactions[0]):
        return Support(False, "the first radial network is not an e3nn FullyConnectedNet without output activation")
    return Support(True, "ok")


def _support_fused_radial_backward(model):
    generic = _generic_interactions(model)
    if not generic:
        return Support(False, _NOT_GENERIC)
    if any(_has_fc_radial(i) for _, i, _ in generic):
        return Support(True, "ok")
    return Support(False, "no generic radial network is an e3nn FullyConnectedNet without output activation")


def _support_streaming(model):
    generic = _generic_interactions(model)
    if not generic:
        return Support(False, _NOT_GENERIC)
    if any(_oeq_fused(i) for _, i, _ in generic):
        return Support(True, "ok")
    return Support(False, "no generic interaction has an OpenEquivariance-fused convolution (hybrid backend only)")


# --------------------------------------------------------------------------- self-checks


def _probe_inputs(module, columns, generator, species_only):
    """Leaf inputs of one interaction (float64 draws in the module's dtype) and the gradient leaves."""
    import torch
    from e3nn import o3

    parameter = next(module.parameters())
    dims = {name: o3.Irreps(str(getattr(module, f"{name}_irreps"))).dim
            for name in ("node_attrs", "node_feats", "edge_attrs", "edge_feats")}

    def draw(*shape, rand=False):
        values = (torch.rand if rand else torch.randn)(shape, generator=generator, dtype=torch.float64)
        return values.to(device=parameter.device, dtype=parameter.dtype)

    picks = columns.cpu()[torch.randint(len(columns), (_NODES,), generator=generator)]
    node_attrs = torch.zeros((_NODES, dims["node_attrs"]), dtype=torch.float64)
    node_attrs[torch.arange(_NODES), picks] = 1
    node_attrs = node_attrs.to(device=parameter.device, dtype=parameter.dtype)
    table = draw(dims["node_attrs"], dims["node_feats"]).requires_grad_() if species_only else None
    feats = None if species_only else draw(_NODES, dims["node_feats"]).requires_grad_()
    edge_attrs = draw(_EDGES, dims["edge_attrs"]).requires_grad_()
    edge_feats = draw(_EDGES, dims["edge_feats"]).requires_grad_()
    cutoff = (draw(_EDGES, 1, rand=True) + 0.1).requires_grad_()
    senders = torch.randint(_NODES, (_EDGES,), generator=generator)
    receivers = torch.sort(torch.randint(_NODES, (_EDGES,), generator=generator)).values  # streaming needs sorting
    edge_index = torch.stack([senders, receivers]).to(parameter.device)
    return dict(node_attrs=node_attrs, table=table, node_feats=feats, edge_attrs=edge_attrs,
                edge_feats=edge_feats, edge_index=edge_index, cutoff=cutoff)


def _run(module, probe, with_cutoff, cotangents, generator):
    import torch

    reassociation = getattr(module, "first_reassociation", None)
    own = None if reassociation is None else reassociation.species_feats
    with torch.enable_grad():
        feats = probe["node_attrs"] @ probe["table"] if probe["table"] is not None else probe["node_feats"]
        cutoff = probe["cutoff"] if with_cutoff else None
        if reassociation is not None:  # the probe's species table in place of the structure's (this check only)
            reassociation.bind(module.linear_up(probe["table"][reassociation.columns]))
        try:
            outputs = [out for out in module(probe["node_attrs"], feats, probe["edge_attrs"], probe["edge_feats"],
                                             probe["edge_index"], cutoff=cutoff) if out is not None]
        finally:
            if reassociation is not None:
                reassociation.bind(own)
        if not cotangents:
            cotangents.extend(torch.randn(out.shape, generator=generator, dtype=torch.float64).to(out)
                              for out in outputs)
        leaves = [probe["table"] if probe["table"] is not None else probe["node_feats"],
                  probe["edge_attrs"], probe["edge_feats"]] + ([probe["cutoff"]] if with_cutoff else [])
        grads = torch.autograd.grad(outputs, leaves, cotangents[:len(outputs)], allow_unused=True)
    grads = [torch.zeros_like(leaf) if grad is None else grad for leaf, grad in zip(leaves, grads)]
    return [out.detach() for out in outputs] + grads


def _check_interactions(original, rewritten, context, selected):
    """CheckResult over the interactions of ``rewritten`` that ``selected(interaction)`` picks."""
    import torch

    generator = torch.Generator().manual_seed(context.seed)
    node_attrs = context.inputs.get(NODE_ATTRS)
    cases, dtype = [], None
    for index, interaction in enumerate(getattr(rewritten, "interactions", ())):
        if not selected(interaction):
            continue
        name = f"interactions.{index}"
        reference = original.get_submodule(name)
        if node_attrs is not None:
            columns = _columns(context, name)
        else:
            from e3nn import o3
            columns = torch.arange(o3.Irreps(str(interaction.node_attrs_irreps)).dim)
        species_only = getattr(interaction, "first_reassociation", None) is not None
        probe = _probe_inputs(interaction, columns, generator, species_only)
        added = False
        if getattr(interaction, "streaming_kernels", None) is not None and not hasattr(interaction, "streaming_count"):
            interaction.streaming_count = torch.tensor(_EDGES, device=probe["edge_index"].device)
            added = True
        try:
            for with_cutoff in (False, True):
                cotangents = []
                expected = _run(reference, probe, with_cutoff, cotangents, generator)
                got = _run(interaction, probe, with_cutoff, cotangents, generator)
                cases += [(name, a, b) for a, b in zip(expected, got)]
        finally:
            if added:
                del interaction.streaming_count
        dtype = dtype or next(interaction.parameters()).dtype
    return check_tensors(cases, dtype)


def _selfcheck_species_skip(original, rewritten, context):
    return _check_interactions(original, rewritten, context, lambda i: isinstance(i.skip_tp, gm.SpeciesSkip))


def _selfcheck_reassociate(original, rewritten, context):
    return _check_interactions(original, rewritten, context,
                               lambda i: getattr(i, "first_reassociation", None) is not None)


def _selfcheck_segmented_first_sum(original, rewritten, context):
    return _check_interactions(original, rewritten, context,
                               lambda i: getattr(getattr(i, "first_reassociation", None), "segmented", False))


def _selfcheck_fused_radial_backward(original, rewritten, context):
    return _check_interactions(original, rewritten, context, lambda i: getattr(i, "fused_radial_backward", False))


def _selfcheck_streaming(original, rewritten, context):
    return _check_interactions(original, rewritten, context,
                               lambda i: getattr(i, "streaming_kernels", None) is not None)


# --------------------------------------------------------------------------- records


def _record(name, config_field, applies_to, exactness, constraints, apply, selfcheck, requires=(),
            deterministic=True):
    # deterministic as the MH-1 records (SPEC §9.3, 2-T16): the streaming and segmented variants are not.
    return dict(name=name, config_field=config_field, applies_to=applies_to, requires=requires, conflicts=(),
                mace_versions=MACE_VERSIONS, source_hashes=gm.source_hashes(), exactness=exactness,
                default="auto", workload_constraints=constraints, apply=apply, selfcheck=selfcheck,
                deterministic=deterministic)


# In the research engine's order (generic_rewrites): skip, reassociation, fused backward, streaming.
RECORDS = {
    "generic_species_skip": _record(
        "generic_species_skip", "species_skip", _support_species_skip, "exact", ("fixed_species_superset",),
        apply_species_skip, _selfcheck_species_skip),
    "generic_reassociate": _record(
        "generic_reassociate", "reassociate_first", _support_reassociate, "rounding",
        ("single_graph", "fixed_species_superset"), apply_reassociate, _selfcheck_reassociate),
    # GENERIC_REASSOCIATE builds its reassociation with segmented=SEGMENTED_FIRST_SUM.
    "generic_segmented_first_sum": _record(
        "generic_segmented_first_sum", "segmented_first_sum", _support_reassociate, "exact",
        ("single_graph", "fixed_species_superset", "sort_receivers"), apply_segmented_first_sum,
        _selfcheck_segmented_first_sum, requires=("generic_reassociate",), deterministic=False),
    "generic_fused_radial_backward": _record(
        "generic_fused_radial_backward", "fused_folded_backward", _support_fused_radial_backward, "rounding", (),
        apply_fused_radial_backward, _selfcheck_fused_radial_backward),
    "generic_streaming": _record(
        "generic_streaming", "streaming_conv", _support_streaming, "exact",
        ("sort_receivers", "per_engine_edge_counter"), apply_streaming, _selfcheck_streaming, deterministic=False),
}

__all__ = ["NODE_ATTRS", "RECORDS", "apply_fused_radial_backward", "apply_reassociate", "apply_species_skip",
           "apply_streaming"]
