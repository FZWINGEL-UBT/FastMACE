"""The MH-1 family: rewrites of MH-1-class interactions (SPEC §6.2; plan task 1a-T07).

``RECORDS`` holds one registry record (``fastmace.rewrites.registry.Rewrite``) per §6.2 row,
in the order the research engine applies them. Every record acts in place on the
interactions of ``FACTORED_INTERACTIONS`` (``RealAgnosticResidualNonLinearInteractionBlock``)
and sets the attributes that ``modules.cutoff_folded_forward`` reads; the research engine's
module switches became these records (docs/design/switch-map.md, section 1).

- ``factor_species``: the species embeddings' share of both radial networks' first layers is
  computed per atom (``ProjectedEmbedding``) and gathered per edge (``FactoredRadialMLP``).
- ``fold_cutoff``: the cutoff applied inside the last radial layer, (h W + b) c = [h c, c] [W; b];
  replaces the interaction forward by ``cutoff_folded_forward`` (source hash-checked).
- ``density_first``: linear_1(m) / s = linear_1(m / s) for a bias-free ``linear_1``.
- ``joint_radial_input``: one first layer for both radial networks.
- ``fused_folded_backward``: the folded last layer's input gradient as one Triton pass.
- ``fused_radial_hidden``: the radial hidden block as Triton kernels; opt-in (no gain), off.
- ``streaming_conv``: the receiver-streaming OEq convolution (OEq PR 223 kernels) in every
  interaction with an OEq-fused convolution.
- ``reassociate_first``: the first interaction summed per receiver and sender species before
  its last radial layer; needs the structure's one-hot species as ``Context.inputs["node_attrs"]``.
- ``segmented_first_sum``: that sum as a run-segmented Triton kernel.

Self-checks (SPEC §6.4) compare each rewritten interaction with the unrewritten one by name on
a random graph in the model's dtype and device: message, skip features and the gradients with
respect to the edge inputs (spherical harmonics, radial features, cutoff) and the node
features. Receivers are sorted (the streaming kernels need it) and every species of the
structure occurs. A reassociated first interaction gets species-only node features and no
node-feature gradient: in the model its node features are the species embedding, which does
not depend on the positions.

Workload constraints, for the planner: ``single_graph`` (one structure per evaluation),
``fixed_species_superset`` (atoms may move or swap, but not to species absent at setup),
``sort_receivers`` (receiver-sorted edges, ``NeighbourConfig.sort_receivers``), ``oeq`` (the
hybrid backend) and ``per_engine_edge_counter`` (the engine binds its neighbour list's GPU edge
count with ``bind_streaming_count``; until then every edge counts).
"""

from __future__ import annotations

from fastmace.model.capabilities import FACTORED_INTERACTIONS, first_inputs_species_only, radial_cutoff_passed
from fastmace.model.capabilities import rewrite_support as _model_support
from fastmace.rewrites.registry import Rewrite, RewriteError, Support, check_tensors, source_sha256

MACE_VERSIONS = ">=0.3.16,<0.3.17"
# sha256 of inspect.getsource(RealAgnosticResidualNonLinearInteractionBlock.forward) in mace-torch 0.3.16
# (the research engine's _NONLINEAR_FORWARD_SHA256): modules.cutoff_folded_forward copies it with one change.
NONLINEAR_FORWARD = "mace.modules.blocks:RealAgnosticResidualNonLinearInteractionBlock.forward"
NONLINEAR_FORWARD_SHA256 = "727c37b796de7d3b33793ae6317a6d09c1f4c8677b28c04963ffbb34899b0b44"
RADIAL_MLP_FORWARD = "mace.modules.radial:RadialMLP.forward"
RADIAL_MLP_FORWARD_SHA256 = "a6f06e6c4a4fece818030d4b977f41d369273f2c33da75ccf1f06036ca5ff9eb"

SINGLE_GRAPH = "single_graph"
FIXED_SPECIES_SUPERSET = "fixed_species_superset"
SORT_RECEIVERS = "sort_receivers"
OEQ = "oeq"
PER_ENGINE_EDGE_COUNTER = "per_engine_edge_counter"

_NODES, _EDGES = 7, 29  # self-check graph


def _modules():
    from fastmace.rewrites.mh1 import modules

    return modules


def _factored(model):
    """(index, interaction) of every FACTORED_INTERACTIONS interaction."""
    return [(index, interaction) for index, interaction in enumerate(getattr(model, "interactions", ()))
            if type(interaction).__name__ in FACTORED_INTERACTIONS]


def _folded(interaction):
    return getattr(getattr(interaction, "forward", None), "__func__", None) is _modules().cutoff_folded_forward


# --------------------------------------------------------------------------- apply


def factor_species_radial(model):
    """Rewrite the radial input layers of every factored interaction in place; returns their names.

    Applies to interactions of FACTORED_INTERACTIONS whose radial networks read
    [edge_feats, source_embedding[sender], target_embedding[receiver]]
    (mace-torch 0.3.16). Each interaction's source/target embedding is
    replaced by its projection through the species columns of both
    ``conv_tp_weights`` and ``density_fn``; both networks then read only the
    radial columns plus their projected blocks. Raises if an interaction of
    that class does not have the expected layout.
    """
    import torch

    m = _modules()
    names = []
    for index, interaction in _factored(model):
        problem = _layout_problem(interaction)
        if problem:
            raise RewriteError(f"factor_species: {problem} in interactions.{index}")
        radial_dim = interaction.edge_feats_irreps.num_irreps
        nets = (interaction.conv_tp_weights, interaction.density_fn)
        firsts = [net.net[0] for net in nets]
        species_dim = (firsts[0].in_features - radial_dim) // 2
        widths = [first.out_features for first in firsts]
        total = sum(widths)
        source_columns = slice(radial_dim, radial_dim + species_dim)
        target_columns = slice(radial_dim + species_dim, radial_dim + 2 * species_dim)
        interaction.source_embedding = m.ProjectedEmbedding(
            interaction.source_embedding, [first.weight[:, source_columns] for first in firsts])
        interaction.target_embedding = m.ProjectedEmbedding(
            interaction.target_embedding, [first.weight[:, target_columns] for first in firsts],
            torch.cat([first.bias for first in firsts]))
        replaced = []
        offset = 0
        for net, width in zip(nets, widths):
            source = (radial_dim + offset, radial_dim + offset + width)
            target = (radial_dim + total + offset, radial_dim + total + offset + width)
            replaced.append(m.FactoredRadialMLP(net, radial_dim, source, target))
            offset += width
        interaction.conv_tp_weights, interaction.density_fn = replaced
        names.append(f"interactions.{index}")
    return names


def fold_radial_cutoff(model):
    """Make every factored interaction apply its cutoff inside conv_tp_weights; returns their names."""
    import types

    import torch

    m = _modules()
    names = []
    for index, interaction in _factored(model):
        digest = source_sha256(f"{type(interaction).__module__}:{type(interaction).__qualname__}.forward")
        if digest != NONLINEAR_FORWARD_SHA256:
            raise RewriteError("fold_cutoff supports the mace-torch 0.3.16 interaction forward only "
                               f"(source sha256 {digest})")
        if not isinstance(interaction.conv_tp_weights, m.FactoredRadialMLP):
            raise RewriteError(f"fold_cutoff needs factor_species first (interactions.{index})")
        last = interaction.conv_tp_weights.rest[-1]
        if not isinstance(last, torch.nn.Linear) or last.bias is None:
            raise RewriteError(f"Unexpected last radial layer {last} in interactions.{index}")
        interaction.forward = types.MethodType(m.cutoff_folded_forward, interaction)
        names.append(f"interactions.{index}")
    return names


def _folded_interactions(model, name):
    out = []
    for index, interaction in _factored(model):
        if not _folded(interaction):
            raise RewriteError(f"{name} requires the folded interaction forward (fold_cutoff), "
                               f"interactions.{index} has none")
        out.append((index, interaction))
    return out


def _linear_biases(linear):
    # e3nn keeps an empty "bias" buffer when a Linear has no biases.
    return [name for name, tensor in (*linear.named_parameters(), *linear.named_buffers())
            if "bias" in name and tensor.numel() > 0]


def density_before_linear(model):
    """Divide by the density before the bias-free linear_1 in every folded interaction; returns their names."""
    names = []
    for index, interaction in _folded_interactions(model, "density_first"):
        biases = _linear_biases(interaction.linear_1)
        if biases:
            raise RewriteError(f"density_first needs a bias-free linear_1 (found {biases} in interactions.{index})")
        interaction.density_before_linear = True
        names.append(f"interactions.{index}")
    return names


def joint_radial_input(model):
    """One first layer for both radial networks of every folded interaction; returns their names."""
    import torch

    names = []
    for index, interaction in _folded_interactions(model, "joint_radial_input"):
        nets = (interaction.conv_tp_weights, interaction.density_fn)
        # Blocks of the projected embeddings are ordered [conv_tp_weights, density_fn].
        if nets[0].source[0] + nets[0].weight.shape[0] != nets[1].source[0]:
            raise RewriteError(f"joint_radial_input expects adjacent conv/density species blocks "
                               f"(interactions.{index})")
        interaction.joint_radial_weight = torch.nn.Parameter(
            torch.cat([net.weight for net in nets], 0).detach().clone(), requires_grad=False)
        interaction.joint_radial_widths = [int(net.weight.shape[0]) for net in nets]
        names.append(f"interactions.{index}")
    return names


def fused_folded_backward(model):
    """Triton input gradient of the folded last radial layer of every folded interaction; returns their names."""
    names = []
    for index, interaction in _folded_interactions(model, "fused_folded_backward"):
        interaction.conv_tp_weights.fused_backward = True
        names.append(f"interactions.{index}")
    return names


def fuse_radial_hidden(model):
    """Fuse every LayerNorm, SiLU, Linear, LayerNorm, SiLU run of both radial networks; returns the networks."""
    from fastmace.kernels.triton.fused_radial import hidden_plan

    names = []
    for index, interaction in _folded_interactions(model, "fused_radial_hidden"):
        for attribute in ("conv_tp_weights", "density_fn"):
            net = getattr(interaction, attribute)
            plan = hidden_plan(net.rest[:-1])
            if plan is not None:
                net.fused_hidden = plan
                names.append(f"interactions.{index}.{attribute}")
    return names


def _streamable(interaction):
    return hasattr(interaction, "conv_fusion") and hasattr(getattr(interaction, "conv_tp", None), "input_args")


def bind_streaming_count(model, count):
    """Point every streaming interaction of ``model`` at ``count``, the engine's GPU edge count (a 0-d tensor).

    The streaming kernels leave every edge at or after the count out of the receivers' rows,
    so the engine's padding beyond the cutoff contributes nothing (the research engine's
    ``MACEForces.bind_streaming_count``). Returns the number of interactions bound.
    """
    bound = 0
    for module in model.modules():
        if getattr(module, "streaming_kernels", None) is not None:
            module.register_buffer("streaming_count", count, persistent=False)
            bound += 1
    return bound


def streaming_convolutions(model):
    """Receiver-streaming kernels for every folded interaction with an OEq-fused convolution; returns their names.

    Every edge counts until the engine binds its edge counter (``bind_streaming_count``).
    """
    m = _modules()
    names = []
    for index, interaction in _folded_interactions(model, "streaming_conv"):
        if _streamable(interaction):
            m.attach_streaming_kernels(interaction)
            names.append(f"interactions.{index}")
    return names


def reassociate_first_interaction(model, node_attrs):
    """Attach the reassociated first interaction to a folded model; returns the species columns.

    Requires species-only first-interaction inputs (first_inputs_species_only):
    a linear node embedding of the one-hot species, plus at most per-graph
    joint embeddings (one structure per evaluation). ``node_attrs`` are
    the structure's one-hot species; atoms may change places later, but not
    to species absent at setup.
    """
    import torch

    m = _modules()
    interactions = getattr(model, "interactions", ())
    interaction = interactions[0] if len(interactions) else None
    if interaction is None or not _folded(interaction):
        raise RewriteError("reassociate_first requires the folded first interaction (fold_cutoff)")
    if not first_inputs_species_only(model):
        raise RewriteError("reassociate_first requires species-only first-interaction inputs")
    if node_attrs is None:
        raise RewriteError("reassociate_first needs the structure's one-hot species (Context.inputs['node_attrs'])")
    channels, ls, gammas, ir_mul = m.probe_first_tensor_product(interaction)
    columns = torch.nonzero(torch.as_tensor(node_attrs).sum(dim=0) > 0).flatten()
    interaction.first_reassociation = m.FirstInteractionReassociation(
        columns.to(next(interaction.parameters()).device), channels, ls, gammas, ir_mul)
    return tuple(int(c) for c in columns)


def segment_first_sum(model):
    """Run-segmented sum in the reassociated first interaction; returns its name."""
    reassociation = getattr(model.interactions[0], "first_reassociation", None)
    if reassociation is None:
        raise RewriteError("segmented_first_sum requires the reassociated first interaction (reassociate_first)")
    reassociation.segmented = True
    return ["interactions.0"]


# --------------------------------------------------------------------------- applies_to


def _layout_problem(interaction):
    """Why an interaction does not have the factorable radial layout (mace-torch 0.3.16), or None."""
    import torch

    radial_dim = interaction.edge_feats_irreps.num_irreps
    nets = [getattr(interaction, name, None) for name in ("conv_tp_weights", "density_fn")]
    if any(getattr(net, "net", None) is None for net in nets):
        return "radial networks without .net (already factored, or not MACE's RadialMLP)"
    firsts = [net.net[0] for net in nets]
    species_dim = (getattr(firsts[0], "in_features", 0) - radial_dim) // 2
    for first in firsts:
        if (not isinstance(first, torch.nn.Linear) or first.bias is None
                or first.in_features != radial_dim + 2 * species_dim):
            return f"unexpected radial input layer {first}"
    return None


def _factor_support(model):
    verdict = Support(*_model_support(model)["factor_species"])
    if not verdict.supported:
        return verdict
    for index, interaction in _factored(model):
        problem = _layout_problem(interaction)
        if problem:
            return Support(False, f"interactions.{index}: {problem}")
    return verdict


def _fold_support(model):
    return Support(*_model_support(model)["fold_cutoff"])


def _density_support(model):
    verdict = _fold_support(model)
    if not verdict.supported:
        return verdict
    for index, interaction in _factored(model):
        biases = _linear_biases(interaction.linear_1)
        if biases:
            return Support(False, f"interactions.{index}: linear_1 has biases {biases}; density_first needs "
                                  "a bias-free linear_1")
    return verdict


def fuse_radial_mlp(model):
    """The whole hidden stack of every qualifying radial network in one kernel per direction (opt-2 OPT2-T02).

    Returns the networks fused. A network whose hidden run is not LayerNorm, SiLU, (Linear, LayerNorm, SiLU) x K
    (MH-1's density networks: LayerNorm, SiLU only) keeps its modules.
    """
    from fastmace.kernels.triton.radial_mlp import RadialStack, stack_layers

    names = []
    for index, interaction in _folded_interactions(model, "fused_radial_mlp"):
        for attribute in ("conv_tp_weights", "density_fn"):
            net = getattr(interaction, attribute)
            modules = list(net.rest[:-1])
            if not isinstance(stack_layers(modules), str):
                net.fused_stack = RadialStack(modules)
                names.append(f"interactions.{index}.{attribute}")
    return names


def _mlp_support(model):
    from fastmace.kernels.triton.radial_mlp import stack_layers

    verdict = _fold_support(model)
    if not verdict.supported:
        return verdict
    reasons = []
    for index, interaction in _factored(model):
        for attribute in ("conv_tp_weights", "density_fn"):
            net = getattr(interaction, attribute)
            modules = net.rest[:-1] if hasattr(net, "rest") else net.net[1:-1]
            why = stack_layers(modules)
            if not isinstance(why, str):
                return verdict
            reasons.append(f"interactions.{index}.{attribute}: {why}")
    return Support(False, "no radial network with a fusable hidden stack: " + "; ".join(reasons))


def _hidden_support(model):
    from fastmace.kernels.triton.fused_radial import hidden_plan

    verdict = _fold_support(model)
    if not verdict.supported:
        return verdict
    for _, interaction in _factored(model):
        for net in (interaction.conv_tp_weights, interaction.density_fn):
            modules = net.rest[:-1] if hasattr(net, "rest") else net.net[1:-1]
            if hidden_plan(modules) is not None:
                return verdict
    return Support(False, "no LayerNorm, SiLU, Linear, LayerNorm, SiLU run in the radial networks")


def _streaming_support(model):
    from fastmace.kernels import streaming_available

    verdict = _fold_support(model)
    if not verdict.supported:
        return verdict
    if not any(_streamable(interaction) for _, interaction in _factored(model)):
        return Support(False, "no factored interaction with an OEq-fused convolution (hybrid backend)")
    available = streaming_available()
    if not available:
        return Support(False, available.reason)
    return verdict


def _reassociate_support(model):
    verdict = _fold_support(model)
    if not verdict.supported:
        return verdict
    if not first_inputs_species_only(model):
        return Support(False, "first-interaction inputs are not species-only")
    interactions = getattr(model, "interactions", ())
    if type(interactions[0]).__name__ not in FACTORED_INTERACTIONS:
        return Support(False, f"the first interaction {type(interactions[0]).__name__} is not "
                              f"{FACTORED_INTERACTIONS}")
    try:  # one edge through the model's own convolution
        _modules().probe_first_tensor_product(interactions[0])
    except ValueError as error:
        return Support(False, f"the first convolution does not have the reassociated form: {error}")
    return verdict


# --------------------------------------------------------------------------- self-checks


def _graph(model, interaction, context, generator):
    """Random inputs of one interaction (float64 on the CPU): a dict of the forward's arguments."""
    import torch

    def randn(*shape):
        return torch.randn(shape, generator=generator, dtype=torch.float64)

    elements = interaction.node_attrs_irreps.dim
    node_attrs = context.inputs.get("node_attrs") if context.inputs else None
    columns = (torch.nonzero(torch.as_tensor(node_attrs).sum(dim=0) > 0).flatten().cpu() if node_attrs is not None
               else torch.arange(elements))
    order = torch.randperm(_NODES, generator=generator)
    species = columns[torch.arange(_NODES) % len(columns)][order]
    onehot = torch.nn.functional.one_hot(species, elements).to(torch.float64)
    width = interaction.node_feats_irreps.dim
    if getattr(interaction, "first_reassociation", None) is not None:
        node_feats = randn(elements, width)[species]  # species-only, as the node embedding gives
    else:
        node_feats = randn(_NODES, width)
    receivers = torch.sort(torch.randint(0, _NODES, (_EDGES,), generator=generator)).values
    senders = torch.randint(0, _NODES, (_EDGES,), generator=generator)
    inputs = {
        "node_attrs": onehot, "node_feats": node_feats,
        "edge_attrs": randn(_EDGES, interaction.edge_attrs_irreps.dim),
        "edge_feats": randn(_EDGES, interaction.edge_feats_irreps.dim),
        "edge_index": torch.stack([senders, receivers]),
        "cutoff": (torch.rand((_EDGES, 1), generator=generator, dtype=torch.float64)
                   if radial_cutoff_passed(model) else None),
    }
    return inputs


def _evaluate(interaction, inputs, grads, differentiate):
    import torch

    with torch.enable_grad():
        leaves = {name: inputs[name].detach().clone().requires_grad_() for name in differentiate}
        arguments = {**inputs, **leaves}
        outputs = [out for out in interaction(**arguments) if out is not None]
        values = [out.detach() for out in outputs]
        pairs = [(out, grad) for out, grad in zip(outputs, grads) if out.requires_grad]
        gradients = (torch.autograd.grad([out for out, _ in pairs], list(leaves.values()), [g for _, g in pairs],
                                         allow_unused=True) if pairs else [None] * len(leaves))
    gradients = [torch.zeros_like(leaf) if g is None else g for leaf, g in zip(leaves.values(), gradients)]
    return values + gradients


def _check_interactions(original, rewritten, context):
    """CheckResult over every factored interaction of ``rewritten`` against ``original`` (see the module docstring)."""
    import torch

    generator = torch.Generator().manual_seed(context.seed)
    cases, dtype = [], None
    for index, interaction in _factored(rewritten):
        reference = original.get_submodule(f"interactions.{index}")
        parameter = next(interaction.parameters())
        device, dtype = parameter.device, dtype or parameter.dtype
        inputs = _graph(rewritten, interaction, context, generator)
        inputs = {name: (value.to(device=device, dtype=parameter.dtype) if value.is_floating_point()
                         else value.to(device)) if value is not None else None for name, value in inputs.items()}
        differentiate = [name for name in ("edge_attrs", "edge_feats", "cutoff") if inputs[name] is not None]
        if getattr(interaction, "first_reassociation", None) is None:
            differentiate.append("node_feats")
        with torch.no_grad():
            shapes = [out.shape for out in reference(**inputs) if out is not None]
        grads = [torch.randn(shape, generator=generator, dtype=torch.float64).to(device=device, dtype=parameter.dtype)
                 for shape in shapes]
        expected = _evaluate(reference, inputs, grads, differentiate)
        got = _evaluate(interaction, inputs, grads, differentiate)
        cases += [(f"interactions.{index}", a, b) for a, b in zip(expected, got)]
    return check_tensors(cases, dtype)


def _selfcheck(original, rewritten, context):
    return _check_interactions(original, rewritten, context)


# --------------------------------------------------------------------------- records


def _apply(function):
    def apply(model, context):
        return function(model)

    apply.__name__ = f"apply_{function.__name__}"
    return apply


def _apply_reassociate(model, context):
    return reassociate_first_interaction(model, (context.inputs or {}).get("node_attrs"))


def _record(name, applies_to, apply, requires=(), exactness="rounding", default="auto", constraints=(),
            hashes=None, deterministic=True, conflicts=()):
    # deterministic (SPEC §9.3, 2-T16): torch reformulations, MACE's scatter_add_ and FastMACE's Triton
    # kernels without atomics are; the streaming convolution and the segmented forward are not.
    return Rewrite(name=name, applies_to=applies_to, requires=requires, conflicts=conflicts,
                   mace_versions=MACE_VERSIONS,
                   deterministic=deterministic,
                   source_hashes=hashes or {}, exactness=exactness, default=default,
                   workload_constraints=constraints, apply=apply, selfcheck=_selfcheck)


_FORWARD_PIN = {NONLINEAR_FORWARD: NONLINEAR_FORWARD_SHA256}
_FIRST = (SINGLE_GRAPH, FIXED_SPECIES_SUPERSET)

RECORDS = {
    "factor_species": _record("factor_species", _factor_support, _apply(factor_species_radial),
                              hashes={**_FORWARD_PIN, RADIAL_MLP_FORWARD: RADIAL_MLP_FORWARD_SHA256}),
    "fold_cutoff": _record("fold_cutoff", _fold_support, _apply(fold_radial_cutoff), ("factor_species",),
                           hashes=_FORWARD_PIN),
    "density_first": _record("density_first", _density_support, _apply(density_before_linear), ("fold_cutoff",)),
    "joint_radial_input": _record("joint_radial_input", _fold_support, _apply(joint_radial_input), ("fold_cutoff",)),
    "fused_folded_backward": _record("fused_folded_backward", _fold_support, _apply(fused_folded_backward),
                                     ("fold_cutoff",)),
    "fused_radial_hidden": _record("fused_radial_hidden", _hidden_support, _apply(fuse_radial_hidden),
                                   ("fold_cutoff",), default="off"),
    # opt-2 OPT2-T02: the whole hidden stack per direction; covers what fused_radial_hidden fuses.
    "fused_radial_mlp": _record("fused_radial_mlp", _mlp_support, _apply(fuse_radial_mlp), ("fold_cutoff",),
                                default="off", conflicts=("fused_radial_hidden",)),
    "streaming_conv": _record("streaming_conv", _streaming_support, _apply(streaming_convolutions),
                              ("fold_cutoff",), exactness="exact",
                              constraints=(OEQ, SORT_RECEIVERS, PER_ENGINE_EDGE_COUNTER),
                              deterministic=False),  # OEq kernels add sender gradients atomically
    "reassociate_first": _record("reassociate_first", _reassociate_support, _apply_reassociate, ("fold_cutoff",),
                                 constraints=_FIRST),
    "segmented_first_sum": _record("segmented_first_sum", _reassociate_support, _apply(segment_first_sum),
                                   ("reassociate_first",), exactness="exact", constraints=(*_FIRST, SORT_RECEIVERS),
                                   deterministic=False),  # float32 Triton forward: relaxed atomics across chunks
}

__all__ = [
    "FIXED_SPECIES_SUPERSET", "MACE_VERSIONS", "OEQ", "PER_ENGINE_EDGE_COUNTER", "RECORDS", "SINGLE_GRAPH",
    "SORT_RECEIVERS", "bind_streaming_count", "density_before_linear",
    "factor_species_radial", "fold_radial_cutoff", "fuse_radial_hidden", "fuse_radial_mlp", "fused_folded_backward",
    "joint_radial_input", "reassociate_first_interaction", "segment_first_sum", "streaming_convolutions",
]
