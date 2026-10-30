"""The species family of optimisation round 1 (OPT-T08; ``docs/design/opt-T08-linear-audit.md``).

``RECORDS`` holds two opt-in records (default off). Both are rounding class and deterministic.

- ``species_tables`` (candidate C1). In MD the one-hot ``node_attrs`` never change, so every
  atom-level product that reads only them is a function of the atom's species. These products
  become per-species tables, computed once by the replaced modules on the identity rows and
  gathered by species at each step (``modules.SpeciesTable``):
  - the node embedding (it then hands the one-hot on, ``modules.OneHotPassthrough``);
  - each interaction's source and target embeddings (``ProjectedEmbedding`` after
    ``factor_species``);
  - the first interaction's ``skip_tp`` and ``linear_up``, whose input is the embedding.

  The step loses their products and backward: 4.1 % of NaCl's B stage, 2.1 % of NbCl5's (2-T20a
  profiles). Pinned sources:
  - ``ScaleShiftMACE.forward``: the embedding is read only by the first interaction;
  - the MH-1-class interaction forward: the first interaction reads it only through ``skip_tp`` and
    ``linear_up``.

  Refused for a joint charge/spin embedding (OMOL-0): there the first input is not species-only.
  Species swaps (phase 3) change only the gathered index.
- ``fold_readout_tail`` (candidate C2). The last product's ``linear`` and the last interaction's
  ``skip_tp`` feed only the last readout, whose first layer is linear. That layer is folded into
  both (``modules.compose``) and replaced by the identity. Products of 512 columns become
  products of the readout's width (96 on MH-1, all heads kept): 2.0 % of NaCl's B stage, 1.2 % of
  NbCl5's. MACE's ``node_feats`` descriptor output changes meaning; FastMACE never reads it.
  Needs ``block_linear``; MH-1 class only (the other classes' skip is element-dependent).

Each self-check compares the rewritten modules with what they replaced, values and input gradients,
in the model's dtype (SPEC §6.4).
"""

from __future__ import annotations

from fastmace.model.capabilities import FACTORED_INTERACTIONS
from fastmace.rewrites.registry import Rewrite, Support, check_tensors

MACE_VERSIONS = ">=0.3.16,<0.3.17"
MODEL_FORWARD = "mace.modules.models:ScaleShiftMACE.forward"
MODEL_FORWARD_SHA256 = "7f4daea32cb9b55c59db5ee623f8310ef65e41cf0f717278f743ae8bd2782e1d"
INTERACTION_FORWARD = "mace.modules.blocks:RealAgnosticResidualNonLinearInteractionBlock.forward"
INTERACTION_FORWARD_SHA256 = "727c37b796de7d3b33793ae6317a6d09c1f4c8677b28c04963ffbb34899b0b44"
# fold_readout_tail: the last interaction must return skip_tp(node_feats) only as the product's skip term, with
# skip_tp a linear layer. That holds for the MH-1 class (MH-1, OMOL-0). MACE-MP-0, the density family and
# MACE-OFF use an element-dependent skip (a tensor product with node_attrs): no linear layer to fold into.
TAIL_FORWARDS = {
    "RealAgnosticResidualNonLinearInteractionBlock": (INTERACTION_FORWARD, INTERACTION_FORWARD_SHA256),
}
_PROBE_ROWS = 7
# The interaction forward that passes the one-hot node_attrs as the residual input when ``species_residual`` is set
# (fastmace.rewrites.mh1.modules.cutoff_folded_forward); a forward without the marker is refused.
READS_SPECIES_RESIDUAL = "fastmace_species_residual"
# Prototype (fp32 profile, 29 October 2026; not a RewriteConfig field): with an int, fold_readout_tail also keeps only
# that head's hidden columns of a multi-head nonlinear readout (MH-1: 16 of 96), so the folded products are 16 wide.
# Exact for that head only: the readout's other output columns become meaningless. None (the default) keeps all.
TAIL_HEAD = None


def _modules():
    from fastmace.rewrites.species import modules

    return modules


def _one_hot_rows(model):
    """The identity rows [n_elements, n_elements] in the model's dtype and device."""
    import torch

    tensor = next(iter(model.buffers()), None)
    tensor = next(iter(model.parameters()), tensor)
    width = len(model.atomic_numbers)
    return torch.eye(width, dtype=tensor.dtype, device=tensor.device)


# --------------------------------------------------------------------------- species_tables


def _tables_support(model):
    if type(model).__name__ != "ScaleShiftMACE":
        return Support(False, f"{type(model).__name__}: only ScaleShiftMACE's forward is pinned")
    if hasattr(model, "joint_embedding"):
        return Support(False, "a joint (charge/spin) embedding is added to the first input, which is then not "
                              "species-only")
    if getattr(getattr(model, "node_embedding", None), "linear", None) is None:
        return Support(False, "no linear node embedding")
    interactions = getattr(model, "interactions", ())
    if not interactions or type(interactions[0]).__name__ not in FACTORED_INTERACTIONS:
        return Support(False, f"the first interaction is not of {FACTORED_INTERACTIONS}")
    first = interactions[0]
    if not hasattr(first, "skip_tp") or not hasattr(first, "linear_up"):
        return Support(False, "the first interaction has no skip_tp and linear_up")
    return Support(True, "ok")


def tabulate(model):
    """Replace the species-only modules of ``model`` by species tables; returns the replaced names."""
    import torch

    m = _modules()
    onehot = _one_hot_rows(model)
    names = []
    with torch.no_grad():
        embedding = model.node_embedding
        embedded = embedding(onehot)
        first = model.interactions[0]
        for attribute in ("skip_tp", "linear_up"):
            module = getattr(first, attribute)
            setattr(first, attribute, m.SpeciesTable(module(embedded), m.Chain(embedding, module)))
            names.append(f"interactions.0.{attribute}")
        for index, interaction in enumerate(model.interactions):
            for attribute in ("source_embedding", "target_embedding"):
                module = getattr(interaction, attribute, None)
                if module is None:
                    continue
                setattr(interaction, attribute, m.SpeciesTable(module(onehot), module))
                names.append(f"interactions.{index}.{attribute}")
        model.node_embedding = m.OneHotPassthrough()
        names.append("node_embedding")
    return names


def _probe_attrs(model, context, generator):
    """One-hot rows: the structure's (context) or random ones.

    A structure with more than ``_PROBE_ROWS`` atoms gives its distinct rows, one per species present, so that
    the checks' memory does not grow with the atom count (the tiled capacity path builds 300k+ atom cells).
    """
    import torch

    attrs = (context.inputs or {}).get("node_attrs")
    width = len(model.atomic_numbers)
    if attrs is None:
        species = torch.randint(0, width, (_PROBE_ROWS,), generator=generator)
        attrs = torch.nn.functional.one_hot(species, width)
    elif attrs.shape[0] > _PROBE_ROWS:
        attrs = torch.unique(attrs, dim=0)
    rows = _one_hot_rows(model)
    return attrs.to(dtype=rows.dtype, device=rows.device)


def _tables_selfcheck(original, rewritten, context):
    import torch

    m = _modules()
    generator = torch.Generator().manual_seed(context.seed)
    attrs = _probe_attrs(rewritten, context, generator)
    cases, dtype = [], None
    with torch.no_grad():
        for name, module in rewritten.named_modules():
            if isinstance(module, m.SpeciesTable):
                cases.append((name, module.reference(attrs), module(attrs)))
                dtype = dtype or module.table.dtype
    return check_tensors(cases, dtype)


# --------------------------------------------------------------------------- fold_readout_tail


def _readout_first(readout):
    """(attribute, module) of the readout's first linear layer, or (None, None)."""
    for attribute in ("linear_1", "linear"):
        module = getattr(readout, attribute, None)
        if module is not None:
            return attribute, module
    return None, None


def _tail_support(model):
    if type(model).__name__ != "ScaleShiftMACE":
        return Support(False, f"{type(model).__name__}: only ScaleShiftMACE's forward is pinned")
    interactions, products, readouts = (getattr(model, n, ()) for n in ("interactions", "products", "readouts"))
    if not interactions or not readouts or len(readouts) not in (1, len(interactions)):
        return Support(False, "the last readout does not read the last layer's features")
    kind = type(readouts[-1]).__name__
    if kind not in ("NonLinearReadoutBlock", "LinearReadoutBlock"):
        return Support(False, f"last readout {kind} is not MACE's linear or nonlinear readout")
    if not getattr(products[-1], "use_sc", True):
        return Support(False, "the last product has no skip connection")
    last = type(interactions[-1]).__name__
    if last not in TAIL_FORWARDS:
        return Support(False, f"last interaction {last} is not one of {tuple(TAIL_FORWARDS)} (its skip_tp is "
                              "not a linear layer of the features alone)")
    skip = type(getattr(interactions[-1], "skip_tp", None)).__name__
    if skip not in ("Linear", "BlockLinear", "SpeciesTable"):
        return Support(False, f"the last skip_tp is a {skip}, not a linear layer")
    return Support(True, "ok")


def fold_tail(model):
    """Fold the last readout's first linear layer into the last product's linear and the last skip_tp."""
    import torch

    from fastmace.rewrites.linear.modules import BlockLinear

    m = _modules()
    readout = model.readouts[-1]
    attribute, layer = _readout_first(readout)
    product, last = model.products[-1], model.interactions[-1]
    if not isinstance(layer, BlockLinear) or not isinstance(product.linear, BlockLinear):
        raise ValueError("fold_readout_tail needs block linears (block_linear)")
    with torch.no_grad():
        product.linear = m.compose(product.linear, layer)
        skip = last.skip_tp
        if isinstance(skip, m.SpeciesTable):  # a one-interaction model under species_tables
            last.skip_tp = m.SpeciesTable(layer(skip.table), m.Chain(skip, layer))
        elif isinstance(skip, BlockLinear):
            last.skip_tp = m.compose(skip, layer)
        else:
            raise ValueError("fold_readout_tail: the last skip_tp is neither a block linear nor a species table")
        setattr(readout, attribute, torch.nn.Identity())
        if TAIL_HEAD is not None and getattr(readout, "num_heads", 1) > 1:
            _slice_head(readout, product, last, int(TAIL_HEAD), m)
    index = len(model.interactions) - 1
    return [f"products.{index}.linear", f"interactions.{index}.skip_tp",
            f"readouts.{len(model.readouts) - 1}.{attribute}"]


def _slice_head(readout, product, last, head, m):
    """TAIL_HEAD: keep one head's hidden columns in the folded products and the readout (exact for that head)."""
    import torch

    if type(readout).__name__ != "NonLinearReadoutBlock":
        raise ValueError("TAIL_HEAD needs MACE's NonLinearReadoutBlock")
    hidden = product.linear.segments_out[0][1]
    width = hidden // readout.num_heads
    cols = slice(head * width, (head + 1) * width)
    weight = product.linear.weight_0
    eye = torch.eye(hidden, dtype=weight.dtype, device=weight.device)
    if torch.any(readout.linear_2(torch.zeros_like(eye[:1])) != 0):
        raise ValueError("TAIL_HEAD: the readout's linear_2 has a bias")
    out = readout.linear_2(eye)  # [hidden, heads]: the effective matrix (normalisation included)
    probe = torch.randn((7, hidden), generator=torch.Generator().manual_seed(5), dtype=torch.float64).to(weight)
    heads = torch.full((7,), head, dtype=torch.long, device=weight.device)
    before = readout(probe, heads)[:, head]
    lin = torch.nn.Linear(width, out.shape[1], bias=False).to(device=weight.device, dtype=weight.dtype)
    lin.weight.copy_(out[cols].t())
    act = readout.non_linearity
    readout.non_linearity = m.HeadActivation(act.acts[0])
    readout.linear_2 = lin
    readout.num_heads = 1
    after = readout(probe[:, cols])[:, head]
    if not torch.allclose(before, after, rtol=1e-5, atol=1e-6 * float(before.abs().max())):
        raise ValueError(f"TAIL_HEAD: head {head} differs after slicing ({float((before - after).abs().max())})")
    product.linear = m.compose(product.linear, m.slice_columns(product.linear, cols.start, width))
    last.skip_tp = m.compose(last.skip_tp, m.slice_columns(last.skip_tp, cols.start, width))


def _tail_selfcheck(original, rewritten, context):
    import torch

    from fastmace.rewrites.linear import _compare, _probe, _values_and_gradients, _width
    from fastmace.rewrites.linear.modules import BlockLinear

    m = _modules()
    generator = torch.Generator().manual_seed(context.seed)
    cases, dtype = [], None
    index = len(rewritten.interactions) - 1
    for name in (f"products.{index}.linear", f"interactions.{index}.skip_tp"):
        module = rewritten.get_submodule(name)
        if not isinstance(module, BlockLinear):
            continue
        reference = m.reference_of(module)
        x = _probe(module, generator, _width(module.segments_in))
        grad = torch.randn((_PROBE_ROWS, _width(module.segments_out)), generator=generator,
                           dtype=torch.float64).to(x)
        cases += _compare(name, _values_and_gradients(reference, (x,), grad),
                          _values_and_gradients(module, (x,), grad))
        dtype = dtype or module.weight_0.dtype
    return check_tensors(cases, dtype)


# --------------------------------------------------------------------------- fold_product_linear


def _consumers(model):
    """(owner, attribute) of every linear consumer of layer 0's output: the second interaction's
    ``skip_tp`` and ``linear_up`` and, with one readout per layer, the first readout's first linear."""
    second = model.interactions[1]
    out = [(second, "skip_tp"), (second, "linear_up")]
    if len(model.readouts) == len(model.interactions):
        attribute, _ = _readout_first(model.readouts[0])
        if attribute is not None:
            out.append((model.readouts[0], attribute))
    return out


def _product_support(model):
    supported, reason = _tables_support(model)
    if not supported:
        return Support(False, reason)
    if len(model.interactions) < 2:
        return Support(False, "one interaction: nothing reads layer 0's output but the readout")
    if type(model.interactions[1]).__name__ not in FACTORED_INTERACTIONS:
        return Support(False, f"the second interaction is not of {FACTORED_INTERACTIONS}")
    if not getattr(model.products[0], "use_sc", True):
        return Support(False, "the first product has no skip connection")
    if len(model.readouts) == len(model.interactions):
        kind = type(model.readouts[0]).__name__
        if kind not in ("NonLinearReadoutBlock", "LinearReadoutBlock"):
            return Support(False, f"first readout {kind} is not MACE's linear or nonlinear readout")
    return Support(True, "ok")


def fold_product(model):
    """Fold ``products.0.linear`` into the linear consumers of layer 0's output (needs species_tables)."""
    import torch

    from fastmace.rewrites.linear.modules import BlockLinear

    m = _modules()
    product, skip = model.products[0].linear, model.interactions[0].skip_tp
    if not isinstance(product, BlockLinear):
        raise ValueError("fold_product_linear needs block linears (block_linear)")
    if not isinstance(skip, m.SpeciesTable):
        raise ValueError("fold_product_linear needs the first skip_tp as a species table (species_tables)")
    width = sum(mul * dim for _, mul, dim in product.segments_in)
    elements = len(model.atomic_numbers)
    names = []
    with torch.no_grad():
        for owner, attribute in _consumers(model):
            consumer = getattr(owner, attribute)
            if not isinstance(consumer, BlockLinear):
                raise ValueError(f"fold_product_linear: {attribute} is not a block linear")
            folded = m.compose(product, consumer)
            setattr(owner, attribute, m.FoldedConsumer(
                folded, width, consumer(skip.table), m.ConsumerReference(product, consumer, skip.table, width)))
            names.append(attribute)
        carried = torch.cat([skip.table.new_zeros((elements, width)),
                             torch.eye(elements, dtype=skip.table.dtype, device=skip.table.device)], dim=1)
        model.interactions[0].skip_tp = m.SpeciesTable(carried, m.SpeciesCarrier(width))
        model.products[0].linear = m.PadColumns(elements)
    return ["products.0.linear", "interactions.0.skip_tp", *names]


def _product_selfcheck(original, rewritten, context):
    import torch

    from fastmace.rewrites.linear import _compare, _values_and_gradients

    m = _modules()
    generator = torch.Generator().manual_seed(context.seed)
    attrs = _probe_attrs(rewritten, context, generator)
    cases, dtype = [], None
    for name, module in rewritten.named_modules():
        if not isinstance(module, m.FoldedConsumer):
            continue
        dtype = dtype or module.table.dtype
        left = torch.randn((attrs.shape[0], module.width), generator=generator, dtype=torch.float64)
        left = left.to(dtype=module.table.dtype, device=module.table.device).requires_grad_()
        grad = torch.randn((attrs.shape[0], module.table.shape[1]), generator=generator,
                           dtype=torch.float64).to(left)
        expected = _values_and_gradients(lambda a, f=module.reference: f(torch.cat([a, attrs], dim=1)), (left,), grad)
        got = _values_and_gradients(lambda a, f=module: f(torch.cat([a, attrs], dim=1)), (left,), grad)
        cases += _compare(name, expected, got)
    return check_tensors(cases, dtype)


# --------------------------------------------------------------------------- records


def _apply_tables(model, context):
    return tabulate(model)


# --------------------------------------------------------------------------- species_residual


def _residual_support(model):
    found = _tables_support(model)
    if not found.supported:
        return found
    if getattr(model.interactions[0], "linear_res", None) is None:
        return Support(False, "the first interaction has no linear_res")
    return Support(True, "ok")


def residual_rows(model):
    """Make the first interaction's residual products per-species rows; returns the replaced names."""
    m = _modules()
    first = model.interactions[0]
    forward = getattr(getattr(first, "forward", None), "__func__", None)
    if not getattr(forward, READS_SPECIES_RESIDUAL, False):
        raise ValueError("species_residual needs the interaction forward of fold_cutoff (cutoff_folded_forward)")
    table = getattr(first, "linear_up", None)
    if not isinstance(table, m.SpeciesTable):
        raise ValueError("species_residual needs the first linear_up as a species table (species_tables)")
    merged = getattr(first, "residual_linear", None)
    if merged is None:
        raise ValueError("species_residual needs the merged residual product (merged_residual)")
    rows = m.SpeciesResidualBlockLinear(merged, table.table)
    first.residual_linear = rows
    names = ["interactions.0.residual_linear"]
    chain = getattr(first, "gated_rows", None)
    if chain is not None and chain.first is merged:
        chain.first = rows
        names.append("interactions.0.gated_rows.first")
    first.species_residual = True
    return names


def _residual_selfcheck(original, rewritten, context):
    """The new rows against the merged product on the gathered table rows, values and the input gradient."""
    import torch

    m = _modules()
    generator = torch.Generator().manual_seed(context.seed)
    first = rewritten.interactions[0]
    rows = first.residual_linear
    if not isinstance(rows, m.SpeciesResidualBlockLinear):
        return check_tensors([], None)
    merged, table = rows.reference
    attrs = _probe_attrs(rewritten, context, generator)
    weight = rows.weight_0
    width = sum(mul * dim for _, mul, dim in rows.segments_in[:rows.first_count])
    x = torch.randn((attrs.shape[0], width), generator=generator, dtype=torch.float64).to(weight).requires_grad_()
    y = table[attrs.argmax(dim=-1)]
    cases = []
    with torch.enable_grad():
        expected = merged(x, y)
        got = rows(x, attrs)
        grad = torch.randn(expected.shape, generator=generator, dtype=torch.float64).to(expected)
        (gx_expected,) = torch.autograd.grad(expected, x, grad)
        (gx_got,) = torch.autograd.grad(got, x, grad)
    cases.append(("interactions.0.residual_linear", expected.detach(), got.detach()))
    cases.append(("interactions.0.residual_linear (input gradient)", gx_expected, gx_got))
    return check_tensors(cases, weight.dtype)


def _apply_residual(model, context):
    return residual_rows(model)


def _apply_tail(model, context):
    return fold_tail(model)


def _apply_product(model, context):
    return fold_product(model)


RECORDS = {
    "species_tables": Rewrite(
        name="species_tables", applies_to=_tables_support, requires=(), conflicts=(),
        mace_versions=MACE_VERSIONS,
        source_hashes={MODEL_FORWARD: MODEL_FORWARD_SHA256, INTERACTION_FORWARD: INTERACTION_FORWARD_SHA256},
        exactness="rounding", default="off", workload_constraints=("fixed_species_superset",),
        apply=_apply_tables, selfcheck=_tables_selfcheck, deterministic=True),
    "fold_readout_tail": Rewrite(
        name="fold_readout_tail", applies_to=_tail_support, requires=("block_linear",), conflicts=(),
        mace_versions=MACE_VERSIONS,
        source_hashes={MODEL_FORWARD: MODEL_FORWARD_SHA256, **dict(TAIL_FORWARDS.values())},
        exactness="rounding", default="off", workload_constraints=(),
        apply=_apply_tail, selfcheck=_tail_selfcheck, deterministic=True),
    "fold_product_linear": Rewrite(
        name="fold_product_linear", applies_to=_product_support, requires=("block_linear", "species_tables"),
        conflicts=(), mace_versions=MACE_VERSIONS,
        source_hashes={MODEL_FORWARD: MODEL_FORWARD_SHA256, INTERACTION_FORWARD: INTERACTION_FORWARD_SHA256},
        exactness="rounding", default="off", workload_constraints=("fixed_species_superset",),
        apply=_apply_product, selfcheck=_product_selfcheck, deterministic=True),
    "species_residual": Rewrite(
        name="species_residual", applies_to=_residual_support,
        requires=("species_tables", "merged_residual"), conflicts=("tf32x3_gemms",),
        mace_versions=MACE_VERSIONS,
        source_hashes={MODEL_FORWARD: MODEL_FORWARD_SHA256, INTERACTION_FORWARD: INTERACTION_FORWARD_SHA256},
        exactness="rounding", default="off", workload_constraints=("fixed_species_superset",),
        apply=_apply_residual, selfcheck=_residual_selfcheck, deterministic=True),
}

__all__ = ["READS_SPECIES_RESIDUAL", "RECORDS", "fold_product", "fold_tail", "residual_rows", "tabulate"]
