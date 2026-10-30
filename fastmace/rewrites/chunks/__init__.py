"""``atom_chunks``: the node-local segments on atom chunks with recomputation (plans/capacity.md CAP-T08).

The force call keeps per-atom state for the backward, and most of it comes from each interaction's node-local
segments, which do not mix atoms (docs/validation/capacity-T08-chunks.md):

- the ``gated_rows`` chain (residual products, gate and ``linear_2``);
- the product block (symmetric contraction, linear and skip connection).

At NbCl5 21,600 atoms they held 168 of the eager force call's 199 kB per atom.

With the record, each of them runs on ``RewriteConfig.atom_chunk_count`` atom chunks, each chunk under
non-reentrant ``torch.utils.checkpoint`` (``modules.chunked_forward``):

- the forward keeps only the chunks' inputs;
- the backward rebuilds one chunk's intermediates at a time.

AOT autograd traces the chunks and their recomputation, so the compiled model stays one graph with Inductor's
fusions. Measured in the compiled default plan (NbCl5 21,600 atoms, ``recompute_conv_weights`` on): the force
call's peak fell from 4.79 to 3.49 GiB with 8 chunks, at 1.17 times the call's time.

The record is ``rounding``, because the GEMMs see fewer rows. It is deterministic: the chunks run one after
another, without atomics.

Off by default. Rule 12 turns it on only when the plan does not fit with ``recompute_conv_weights`` on
(``planner.memory.MEMORY_RESCUE``, in that order), with the fewest chunks that fit (``chunk_count``).

The self-check compares each chunked segment with its own forward unchunked, on probe rows that split unevenly:
values and the input gradients, in the model's dtype.
"""

from __future__ import annotations

import math

from fastmace.rewrites import memory as memory_terms
from fastmace.rewrites.registry import CaseErrors, Rewrite, Support

MACE_VERSIONS = ">=0.3.16,<0.3.17"
# The chunk count when RewriteConfig.atom_chunk_count is None and no plan chose one (a forced "on").
DEFAULT_CHUNKS = 4
# The counts rule 12 chooses from, fewest first (planner.minimal; memory_terms.atom_chunks).
CHUNK_COUNTS = (2, 4, 8, 16)


def _modules():
    from fastmace.rewrites.chunks import modules

    return modules


def segments(model):
    """``(name, module)`` of every node-local segment the record chunks: each interaction's ``gated_rows`` chain and
    each product block, in the model's order."""
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
        return Support(False, "no interaction and product blocks to chunk")
    kinds = {type(product).__name__ for product in products}
    if kinds != {"EquivariantProductBasisBlock"}:
        return Support(False, f"product blocks {sorted(kinds)} are not MACE's EquivariantProductBasisBlock")
    return Support(True, "ok")


def chunk_segments(model, chunks):
    """Chunk every node-local segment of ``model`` into ``chunks`` atom chunks; returns their names."""
    m = _modules()
    names = []
    for name, module in segments(model):
        m.chunk(module, chunks)
        names.append(f"{name} ({chunks} chunks)")
    if not names:
        raise ValueError("atom_chunks: no node-local segment to chunk")
    return names


def _probe(generator, rows, width, like):
    import torch

    return torch.randn((rows, width), generator=generator, dtype=torch.float64).to(like)


def _selfcheck(original, rewritten, context):
    """Each chunked segment against its own forward unchunked, on ``count * MIN_ATOMS_PER_CHUNK + 5`` probe rows (so
    every chunk runs and the split is uneven): the output and the gradients of its floating inputs."""
    import torch

    m = _modules()
    generator = torch.Generator().manual_seed(context.seed + 41)
    cases, dtype = CaseErrors(), None
    elements = len(getattr(rewritten, "atomic_numbers", ())) or 1
    outputs = {}
    for name, module in segments(rewritten):
        count = int(getattr(module, "atom_chunk_count", 0))
        if not count:
            continue
        rows = count * m.MIN_ATOMS_PER_CHUNK + 5
        if name.startswith("interactions."):
            first = module.first
            like = first.weight_0
            widths = [sum(mul * dim for _, mul, dim in part)
                      for part in (first.segments_in[:first.first_count], first.segments_in[first.first_count:])]
            inputs = [_probe(generator, rows, width, like) for width in widths]
            index = int(name.split(".")[1])
            outputs[index] = (rows, rewritten.interactions[index].reshape)
        else:
            index = int(name.split(".")[1])
            if index not in outputs:
                continue  # the product's input comes from the gated_rows probe
            rows_in, reshape = outputs[index]
            chain = rewritten.interactions[index].gated_rows
            like = chain.first.weight_0
            width_out = sum(mul * dim for _, mul, dim in chain.second.segments_out)
            node_feats = reshape(_probe(generator, rows_in, width_out, like))
            species = torch.randint(0, elements, (rows_in,), generator=generator)
            attrs = torch.nn.functional.one_hot(species, elements).to(like)
            reference = m.unchunked(module)
            with torch.no_grad():
                shape = reference(node_feats, None, attrs).shape
            sc = _probe(generator, rows_in, math.prod(shape[1:]), like).view(shape)
            inputs = [node_feats, sc, attrs]
        wants = [x.is_floating_point() and not (name.startswith("products.") and j == 2) for j, x in enumerate(inputs)]
        inputs = [x.detach().requires_grad_(want) for x, want in zip(inputs, wants)]
        leaves = [x for x in inputs if x.requires_grad]
        values = []
        with torch.enable_grad():
            for function in (m.unchunked(module), module):
                out = function(*inputs)
                if not values:
                    grad = torch.randn(out.shape, generator=generator, dtype=torch.float64).to(out)
                values.append((out.detach(), *torch.autograd.grad(out, leaves, grad)))
        for a, b in zip(*values):
            cases.add(name, a, b)
        dtype = dtype or inputs[0].dtype
        del values, inputs, leaves
    return cases.result(dtype)


def _apply(model, context):
    chunks = getattr(context.config, "atom_chunk_count", None) or DEFAULT_CHUNKS
    return chunk_segments(model, chunks)


RECORDS = {
    "atom_chunks": Rewrite(
        name="atom_chunks", applies_to=_support, requires=("gated_rows",), conflicts=(),
        mace_versions=MACE_VERSIONS, source_hashes={}, exactness="rounding", default="off", deterministic=True,
        workload_constraints=(), apply=_apply, selfcheck=_selfcheck, memory=memory_terms.atom_chunks),
}

__all__ = ["CHUNK_COUNTS", "DEFAULT_CHUNKS", "RECORDS", "chunk_segments", "segments"]
