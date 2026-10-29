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
# MH-1's fused path, measured on the GPU (decision 67, plans/capacity.md CAP-T02; docs/validation/capacity-T02.md):
# the share of the estimate's stock per-edge floats (``_stock_edge_floats``; 8,265 for MH-1) each record does not
# keep. chunked_conv_backward and radial_tables: the change in the peak when the record alone is turned off (NbCl5,
# 14,976 atoms, RTX 5060 Ti, float32: 2,086 and 2,306 bytes per edge, 521 and 576 of MH-1's floats). The
# streaming convolution's: the remainder of the fused path (2,000 and 673 floats), fitted so that the estimate
# stays 1.05-1.34x above the capacity scan's measured peaks (tests/cpu/test_memory_fit.py's capacity cases); per
# precision, as the fused paths keep different tensors in float32 and float64. Shares, so a smaller model's terms
# shrink with its widths.
_CHUNKED_BACKWARD_SHARE = 521 / 8265
_RADIAL_TABLES_SHARE = 576 / 8265
_MH1_STREAMING_SHARE = {4: 2000 / 8265, 8: 673 / 8265}


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


def _stock_edge_floats(dims) -> int:
    """The floats per edge the estimate counts for stock MACE with a fused convolution (``planner.memory``): every
    interaction's per-edge tensors, the widest backward transient and the shared edge geometry."""
    from fastmace.planner import memory as estimate  # planner.memory imports nothing from torch or MACE

    if not dims.interactions:
        return 0
    first = dims.interactions[0]
    return (sum(edge for edge, _ in estimate.interaction_floats(dims, fused_conv=True))
            + max(block.weight_numel for block in dims.interactions)
            + estimate._GEOMETRY_FLOATS + first.edge_attrs + first.edge_feats)


def mh1_streaming_conv(dims, width):
    """MH-1's streaming convolution with its fused folded path: its int32 senders and receivers, less the share of
    the stock per-edge floats the fused path does not keep (``_MH1_STREAMING_SHARE``, fitted on the capacity
    scan's GPU peaks, decision 67)."""
    share = _MH1_STREAMING_SHARE.get(width, _MH1_STREAMING_SHARE[4])
    return _STREAMING_INDEX_BYTES - round(share * _stock_edge_floats(dims)) * width, 0


def chunked_conv_backward(dims, width):
    """The chunked convolution backward keeps no per-edge weight gradient of the whole edge set (measured)."""
    return -round(_CHUNKED_BACKWARD_SHARE * _stock_edge_floats(dims)) * width, 0


def radial_tables(dims, width):
    """Tabulated radial networks keep no hidden activations per edge (measured)."""
    return -round(_RADIAL_TABLES_SHARE * _stock_edge_floats(dims)) * width, 0


def block_linear_resident(dims, width):
    """Each converted linear keeps its extracted weight and a contiguous transposed copy as buffers
    (``BlockLinear.weight_k``, ``weight_t_k``), beside the execution copy's own parameters."""
    return 2 * dims.linear_weights * width


def merged_residual_resident(dims, width):
    """The merged layer stacks ``linear_1``'s and ``linear_res``'s weights again, with their transpose
    (``ResidualBlockLinear``)."""
    return 2 * dims.residual_weights * width


def segment_gemm_selfcheck(dims, width, sms):
    """``segment_gemm``'s self-check (rewrites/fusion): per interaction it probes enough atoms that every product
    takes Triton on this device, ``MIN_TRITON_WAVES x sms x BLOCK_M`` rows for the scalar segment (irrep
    dimension 1), with ``linear_2``'s width (the interaction's target irreps) in and out. Probes, the upstream
    gradient, both paths' outputs and input gradients and their intermediates: 8 such tensors at once, one
    interaction at a time (measured on the tiny MH-1 model, tests/cpu/test_memory_estimate.py)."""
    from fastmace.kernels.triton import tile_rule

    atoms = tile_rule.MIN_TRITON_WAVES * max(int(sms or 1), 1) * tile_rule.BLOCK_M
    return _SEGMENT_SELFCHECK_TENSORS * atoms * max(block.target for block in dims.interactions) * width


# segment_gemm's self-check: tensors of [probe atoms, linear_2's width] live at once. 8.0-8.1 measured on the tiny
# MH-1 model (CPU); one more covers the reshape probes and the small tensors beside them.
_SEGMENT_SELFCHECK_TENSORS = 9


__all__ = ["block_linear_resident", "chunked_conv_backward", "factor_species", "fold_cutoff", "fused_radial_hidden",
           "generic_reassociate", "joint_radial_input", "merged_residual_resident", "mh1_streaming_conv",
           "radial_tables", "reassociate_first", "segment_gemm_selfcheck", "streaming_conv"]
