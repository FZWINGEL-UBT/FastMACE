"""The recomputed forwards of ``recompute_node_segments`` (CAP-T08; ``fastmace.rewrites.segments``)."""

from __future__ import annotations

import types


def recomputed_forward(self, *args, **kwargs):
    """``self.recomputed_inner`` (the module's forward before the record) under non-reentrant
    ``torch.utils.checkpoint``: the backward keeps only the segment's inputs and rebuilds its intermediates when it
    reaches the segment. AOT autograd traces the recomputation into the compiled graph."""
    from torch.utils.checkpoint import checkpoint

    return checkpoint(self.recomputed_inner, *args, use_reentrant=False, preserve_rng_state=False, **kwargs)


def recompute(module):
    """Give ``module`` the recomputed forward; its forward before is kept as ``recomputed_inner`` (a bound method,
    so a deep copy rebinds both to the copy)."""
    module.recomputed_inner = module.forward
    module.forward = types.MethodType(recomputed_forward, module)
    return module


def stored(module):
    """``module``'s forward before ``recompute`` or ``join_region`` (the self-check's reference)."""
    return getattr(module, "recomputed_inner", module.forward)


# --------------------------------------------------------------------------- one region per interaction

def joined_inputs(self, x, y):
    """The ``gated_rows`` chain deferred to the product block's region: its two inputs, joined by columns."""
    import torch

    return torch.cat([x, y], dim=1)


def passed_through(self, tensor):
    """The interaction's reshape deferred to the product block's region."""
    return tensor


def region(self, x, y, sc, node_attrs):
    """``gated_rows`` -> reshape -> product block, the interaction's whole node-local tail."""
    return self.recomputed_inner(self.region_reshape(self.region_chain(x, y)), sc, node_attrs)


def region_forward(self, node_feats, sc, node_attrs):
    """The product block's forward with the region: ``node_feats`` is the deferred chain's joined inputs; the chain,
    the reshape and the product run under one non-reentrant checkpoint, so the backward keeps only the chain's
    inputs, the skip term and the species, and rebuilds the rest when it reaches the region."""
    import torch
    from torch.utils.checkpoint import checkpoint

    x, y = node_feats.split(self.region_widths, dim=1)
    body = types.MethodType(region, self)
    chunks = int(getattr(self, "region_chunks", 1))
    if chunks <= 1:
        return checkpoint(body, x, y, sc, node_attrs, use_reentrant=False, preserve_rng_state=False)
    # The region does not mix atoms: each atom chunk is its own checkpoint, so the backward rebuilds one chunk's
    # intermediates at a time (the recomputation's working set divided by the chunk count).
    # A plain loop: Dynamo cannot lift a comprehension's closure into checkpoint's higher-order op.
    xs, ys = torch.tensor_split(x, chunks, dim=0), torch.tensor_split(y, chunks, dim=0)
    attrs = torch.tensor_split(node_attrs, chunks, dim=0)
    scs = torch.tensor_split(sc, chunks, dim=0) if sc is not None else (None,) * chunks
    outs = []
    for i in range(chunks):
        outs.append(checkpoint(body, xs[i], ys[i], scs[i], attrs[i], use_reentrant=False, preserve_rng_state=False))
    return torch.cat(outs, dim=0)


def join_region(interaction, product, chunks=1):
    """One region from ``interaction.gated_rows`` through ``product``: the chain returns its joined inputs, the
    reshape passes them through, and the product block runs the chain, the reshape and itself under one checkpoint.
    The forwards before are kept as bound methods (a deep copy rebinds them to the copy)."""
    chain, reshape = interaction.gated_rows, interaction.reshape
    first = chain.first
    product.region_widths = tuple(sum(mul * dim for _, mul, dim in part)
                                  for part in (first.segments_in[:first.first_count],
                                               first.segments_in[first.first_count:]))
    product.region_chain, product.region_reshape = chain.forward, reshape.forward
    product.region_chunks = max(1, int(chunks))
    product.recomputed_inner = product.forward
    chain.forward = types.MethodType(joined_inputs, chain)
    reshape.forward = types.MethodType(passed_through, reshape)
    product.forward = types.MethodType(region_forward, product)
    return product


__all__ = ["join_region", "joined_inputs", "passed_through", "recompute", "recomputed_forward", "region",
           "region_forward", "stored"]
