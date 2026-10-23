"""Rule 12's memory terms of the rewrite records (``Rewrite.memory``; SPEC §9.3 rule 12, plan task 5-T03).

A term is ``term(dims, width) -> (bytes per edge, bytes per node)``: what the rewrite adds to stock MACE's
force call at its peak (the start of the force backward, where every forward tensor is kept), from the
model's widths (``model.capabilities.ModelDims``) and the force precision's bytes per float. Negative:
a stock tensor the rewrite does not keep. The planner sums the enabled records' terms
(``planner.memory``) and, by decision 51, turns off those with a positive per-edge term first when the
estimate does not fit.

The terms count saved tensors, not transients that end before the peak (the reassociation's
forward outer product, the streaming convolution's sort keys). ``tests/cpu/test_memory_estimate.py``
checks each term the CPU runs with the same tensors against a tiny-model measurement; the CUDA-only
kernels' terms (the streaming convolution) are checked by 5-T03's GPU peaks.
"""

from __future__ import annotations

# int64 per edge: the reassociation's sort key (saved for its backward).
_KEY_BYTES = 8
# int32 senders and receivers per edge, saved by the streaming convolution for its backward.
_STREAMING_INDEX_BYTES = 2 * 4


def _first_hidden(block) -> int:
    """The first hidden widths of an interaction's radial networks, summed (the columns factor_species
    projects the species embeddings into)."""
    networks = (block.radial, *(widths for widths, _ in block.other_radials))
    return sum(widths[1] for widths in networks if len(widths) > 2)


def factor_species(dims, width):
    """The first radial layer reads [radial basis, projected source, projected target] (2 x the networks'
    first hidden widths) instead of [radial basis, source and target embeddings]."""
    first = dims.interactions[0]
    embeddings = first.radial[0] - first.edge_feats
    return (2 * _first_hidden(first) - embeddings) * width, 0


def fold_cutoff(dims, width):
    """Per interaction the folded last layer keeps [h c, c] (last hidden width + 1) instead of the cutoff's
    copy of the tensor product weights."""
    return sum(block.radial[-2] + 1 - block.weight_numel for block in dims.interactions
               if len(block.radial) > 2) * width, 0


def joint_radial_input(dims, width):
    """The concatenated radial input is never formed: its first layer's output is one linear plus the
    gathered species terms."""
    first = dims.interactions[0]
    return -(first.edge_feats + _first_hidden(first)) * width, 0


def fused_radial_hidden(dims, width):
    """A fused LayerNorm, SiLU, Linear, LayerNorm, SiLU run keeps its input and output, not its four
    intermediates: three tensors of the run's width fewer per interaction."""
    return -sum(3 * block.radial[2] for block in dims.interactions if len(block.radial) > 3) * width, 0


def reassociate_first(dims, width):
    """The first interaction keeps [h c, c] and the sort key instead of the per-edge tensor product weights
    (MH-1; the sender features and messages it also avoids are not counted, which errs high)."""
    first = dims.interactions[0]
    return (first.radial[-2] + 1 - first.weight_numel) * width + _KEY_BYTES, 0


def generic_reassociate(dims, width):
    """As ``reassociate_first`` for the generic interactions, which keep [h c] (no bias column)."""
    first = dims.interactions[0]
    return (first.radial[-2] - first.weight_numel) * width + _KEY_BYTES, 0


def streaming_conv(dims, width):
    """Against the fused convolution it replaces: the int32 senders and receivers it saves."""
    return _STREAMING_INDEX_BYTES, 0


__all__ = ["factor_species", "fold_cutoff", "fused_radial_hidden", "generic_reassociate", "joint_radial_input",
           "reassociate_first", "streaming_conv"]
