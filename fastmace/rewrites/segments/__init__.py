"""``recompute_node_segments``: the node-local segments recomputed in the backward (plans/capacity.md CAP-T08).

**Where the memory is.** The force call keeps per-atom state for the backward, and most of it comes from each
interaction's node-local segments, which do not mix atoms (docs/validation/capacity-T08-chunks.md):

- the ``gated_rows`` chain (residual products, gate and ``linear_2``);
- the product block (symmetric contraction, linear and skip connection).

At NbCl5 21,600 atoms they held 168 of the eager force call's 199 kB per atom.

**What the record does.** Each of them runs under non-reentrant ``torch.utils.checkpoint``
(``modules.recomputed_forward``): the forward keeps only the segments' inputs, and the backward rebuilds a
segment's intermediates when it reaches it. AOT autograd traces the recomputation, so the compiled model stays one
graph with Inductor's fusions.

**Measured** in the compiled default plan (NbCl5, ``recompute_conv_weights`` on):

- the force call's peak fell from 4.79 to 3.50 GiB at 21,600 atoms, 64 kB per atom, at 1.10 times the call's time;
- 59 kB per atom at 9,216 atoms.

**Chunking was tried and left out.** Running the segments on 4, 8 or 16 atom chunks saved no more; 2 chunks once
saved 0.24 GiB more, at 1.08 times this record's time. So the record recomputes each segment whole.

**One region per interaction was tried and left out** (``SEGMENT_REGION``): checkpointing ``gated_rows`` through the
product block as one region kept 62 instead of 97 kB per atom, but the compiled peak stayed at 3.5 GiB (the region's
recomputation sets it), at 1.24 times the step time, and NbCl5 67,392 atoms no longer fitted.

**Exactness and planning.** The record is ``rounding``: Inductor fuses the recomputed segments differently. It is
deterministic.

Off by default. Rule 12 turns it on only when the plan does not fit with ``recompute_conv_weights`` on
(``planner.memory.MEMORY_RESCUE``, in that order).

**Self-check.** Each recomputed segment against its stored forward, values and the input gradients, in the
model's dtype.
"""

from __future__ import annotations

import math

from fastmace.rewrites import memory as memory_terms
from fastmace.rewrites.registry import CaseErrors, Rewrite, Support

MACE_VERSIONS = ">=0.3.16,<0.3.17"
_PROBE_ROWS = 7
# "separate": each segment its own region; "combined" (A/Bs): one checkpoint region per interaction from its gated_rows
# chain through its product block (modules.join_region). Combined kept 35 kB per atom fewer, but its recomputation set
# the same compiled peak, at 1.24 times separate's step time, and fitted fewer atoms (capacity-T08-chunks.md).
SEGMENT_REGION = "separate"


def _modules():
    from fastmace.rewrites.segments import modules

    return modules


def segments(model):
    """``(name, module)`` of every node-local segment the record recomputes: each interaction's ``gated_rows`` chain
    and each product block, in the model's order."""
    found = []
    for index, interaction in enumerate(getattr(model, "interactions", ())):
        chain = getattr(interaction, "gated_rows", None)
        if chain is not None:
            found.append((f"interactions.{index}.gated_rows", chain))
    for index, product in enumerate(getattr(model, "products", ())):
        found.append((f"products.{index}", product))
    return found


def _support(model):
    interactions = getattr(model, "interactions", ())
    products = getattr(model, "products", ())
    if not interactions or len(products) != len(interactions):
        return Support(False, "no interaction and product blocks to recompute")
    kinds = {type(product).__name__ for product in products}
    if kinds != {"EquivariantProductBasisBlock"}:
        return Support(False, f"product blocks {sorted(kinds)} are not MACE's EquivariantProductBasisBlock")
    return Support(True, "ok")


def recompute_segments(model):
    """Recompute every node-local segment of ``model`` in the backward, each its own region (or, with
    ``SEGMENT_REGION`` "combined", one region per interaction that has a ``gated_rows`` chain); returns their names."""
    m = _modules()
    names = []
    joined = set()
    if SEGMENT_REGION == "combined":
        for index, interaction in enumerate(getattr(model, "interactions", ())):
            if getattr(interaction, "gated_rows", None) is not None:
                m.join_region(interaction, model.products[index])
                joined.add(index)
                names.append(f"interactions.{index}.gated_rows..products.{index} (one region)")
    for name, module in segments(model):
        if int(name.split(".")[1]) in joined:
            continue
        m.recompute(module)
        names.append(name)
    if not names:
        raise ValueError("recompute_node_segments: no node-local segment")
    return names


def _probe(generator, rows, width, like):
    import torch

    return torch.randn((rows, width), generator=generator, dtype=torch.float64).to(like)


def _region_cases(rewritten, generator, cases):
    """Each combined region against its stored chain, reshape and product block, on probe rows: the output and the
    gradients of the chain's inputs and the skip term."""
    import torch

    elements = len(getattr(rewritten, "atomic_numbers", ())) or 1
    dtype = None
    for index, product in enumerate(getattr(rewritten, "products", ())):
        if getattr(product, "region_widths", None) is None:
            continue
        like = rewritten.interactions[index].gated_rows.first.weight_0
        x, y = (_probe(generator, _PROBE_ROWS, width, like) for width in product.region_widths)
        species = torch.randint(0, elements, (_PROBE_ROWS,), generator=generator)
        attrs = torch.nn.functional.one_hot(species, elements).to(like)

        def stock(x, y, sc, product=product):
            return product.recomputed_inner(product.region_reshape(product.region_chain(x, y)), sc, attrs)

        with torch.no_grad():
            shape = stock(x, y, None).shape
        sc = _probe(generator, _PROBE_ROWS, math.prod(shape[1:]), like).view(shape)
        leaves = [t.detach().requires_grad_() for t in (x, y, sc)]
        grad = torch.randn(shape, generator=generator, dtype=torch.float64).to(like)
        values = []
        with torch.enable_grad():
            joined = torch.cat(leaves[:2], dim=1)
            for out in (stock(*leaves), product(node_feats=joined, sc=leaves[2], node_attrs=attrs)):
                values.append((out.detach(), *torch.autograd.grad(out, leaves, grad)))
        for a, b in zip(*values):
            cases.add(f"interactions.{index}.gated_rows..products.{index}", a, b)
        dtype = dtype or like.dtype
        del values, leaves
    return dtype


def _selfcheck(original, rewritten, context):
    """Each recomputed segment against its stored forward on probe rows: the output and the gradients of its floating
    inputs. A product block's input is the reshaped output of its interaction's ``gated_rows`` probe."""
    import torch

    m = _modules()
    generator = torch.Generator().manual_seed(context.seed + 41)
    cases, dtype = CaseErrors(), None
    elements = len(getattr(rewritten, "atomic_numbers", ())) or 1
    dtype = _region_cases(rewritten, generator, cases)
    for name, module in segments(rewritten):
        index = int(name.split(".")[1])
        if (getattr(module, "recomputed_inner", None) is None
                or getattr(rewritten.products[index], "region_widths", None) is not None):
            continue
        chain = getattr(rewritten.interactions[index], "gated_rows", None)
        if chain is None:
            continue  # a product block's probe comes from its interaction's gated_rows chain
        like = chain.first.weight_0
        if name.startswith("interactions."):
            first = chain.first
            widths = [sum(mul * dim for _, mul, dim in part)
                      for part in (first.segments_in[:first.first_count], first.segments_in[first.first_count:])]
            inputs = [_probe(generator, _PROBE_ROWS, width, like) for width in widths]
        else:
            width_out = sum(mul * dim for _, mul, dim in chain.second.segments_out)
            node_feats = rewritten.interactions[index].reshape(_probe(generator, _PROBE_ROWS, width_out, like))
            species = torch.randint(0, elements, (_PROBE_ROWS,), generator=generator)
            attrs = torch.nn.functional.one_hot(species, elements).to(like)
            with torch.no_grad():
                shape = m.stored(module)(node_feats, None, attrs).shape
            sc = _probe(generator, _PROBE_ROWS, math.prod(shape[1:]), like).view(shape)
            inputs = [node_feats, sc, attrs]
        wants = [x.is_floating_point() and not (name.startswith("products.") and j == 2) for j, x in enumerate(inputs)]
        inputs = [x.detach().requires_grad_(want) for x, want in zip(inputs, wants)]
        leaves = [x for x in inputs if x.requires_grad]
        values = []
        with torch.enable_grad():
            for function in (m.stored(module), module):
                out = function(*inputs)
                if not values:
                    grad = torch.randn(out.shape, generator=generator, dtype=torch.float64).to(out)
                values.append((out.detach(), *torch.autograd.grad(out, leaves, grad)))
        for a, b in zip(*values):
            cases.add(name, a, b)
        dtype = dtype or like.dtype
        del values, inputs, leaves
    return cases.result(dtype)


def _apply(model, context):
    return recompute_segments(model)


RECORDS = {
    "recompute_node_segments": Rewrite(
        name="recompute_node_segments", applies_to=_support, requires=("gated_rows",), conflicts=(),
        mace_versions=MACE_VERSIONS, source_hashes={}, exactness="rounding", default="off", deterministic=True,
        workload_constraints=(), apply=_apply, selfcheck=_selfcheck, memory=memory_terms.recompute_node_segments),
}

__all__ = ["RECORDS", "SEGMENT_REGION", "recompute_segments", "segments"]
