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
# MH-1's fused path, measured on the GPU (decision 67, plans/capacity.md CAP-T02; docs/validation/capacity/capacity-T02.md):
# the share of the estimate's stock per-edge floats (``_stock_edge_floats``; 8,265 for MH-1) each record does not
# keep. chunked_conv_backward and radial_tables: the change in the peak when the record alone is turned off (NbCl5,
# 14,976 atoms, RTX 5060 Ti, float32: 2,086 and 2,306 bytes per edge, 521 and 576 of MH-1's floats). The
# streaming convolution's: the remainder of the fused path (2,000 and 673 floats), fitted so that the estimate
# stays 1.05-1.34x above the capacity scan's measured peaks (tests/cpu/test_memory_fit.py's capacity cases); per
# precision, as the fused paths keep different tensors in float32 and float64. Shares, so a smaller model's terms
# shrink with its widths. In float32 291 more floats (2,291): the rescue matrix's default stage
# (docs/validation/capacity/estimate-band.md) measured 5.3 kB per edge against the estimate's 8.4, an overstatement 1.58 times
# against 1.36 per atom, so crystals with many edges per atom (SrTiO3, NaTaO3) came out 9 % higher; the extra share
# brings the per-edge overstatement to the per-atom one.
_CHUNKED_BACKWARD_SHARE = 521 / 8265
_RADIAL_TABLES_SHARE = 576 / 8265
_MH1_STREAMING_SHARE = {4: 2291 / 8265, 8: 673 / 8265}
# The memory-rescue records and gate_epilogue as fractions of the remaining activations of their own side they keep
# (``planner.memory.MemoryTerm.edge_keep`` and ``node_keep``): conv_weights and first_table recompute per-edge
# tensors, the node segments, the node region and gate_epilogue per-node ones. Each is the ratio of the measured live
# peak's per-edge or per-node slope (max allocated = fixed + per atom x N + per edge x E, least squares per stage of
# the rescue ladder over NbCl5, NaCl, SrTiO3 and NaTaO3; within 1 % at every point) between consecutive stages
# (docs/validation/capacity/estimate-band.md). As fractions of the whole remainder (estimate-refit) they mixed the two sides,
# so a crystal with more edges per atom came out 1.2 times higher at the node-side stages.
# Cells of at least 15,000 atoms (below it the build's fixed transient sets the live peak), five crystals.
# conv_weights keeps 0.43 of 5.3 kB per edge (the second interaction's [E, 1280] weights go); first_table 0.17 of 0.43
# (260 B per edge, the table output; what remains is the radial basis and the indices).
_CONV_WEIGHTS_KEEP = 0.080
_FIRST_TABLE_KEEP = 0.382
# Per atom: 226 kB with the separate gate, 140 with the node segments, 90 with the node region (of the segments').
_NODE_SEGMENTS_KEEP = 0.644
_NODE_REGION_KEEP = 0.646
# gate_epilogue keeps more: 12.3 kB per atom without the node segments (1.057), 36 with the separate segments
# (1.257 of the segments' 140), nothing inside recompute_node_region's chunks (planner.memory.CONTEXT_TERMS).
_GATE_EPILOGUE_KEEP = 1.057
_GATE_EPILOGUE_SEGMENTS_KEEP = 1.257


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


def _generic_folded(dims):
    """(weight_numel, folded input width) of the widest interaction: its tensor product weights per edge and the last
    radial layer's input ``[h c, c]`` (its hidden width and the cutoff column)."""
    widest = max(dims.interactions, key=lambda block: block.weight_numel, default=None)
    if widest is None:
        return 0, 0
    return widest.weight_numel, (widest.radial[-2] if len(widest.radial) > 1 else 0) + 1


def generic_chunked_conv_backward(dims, width):
    """The generic chunked backward keeps no per-edge weight gradient of the whole edge set (the estimate's widest
    backward transient, ``weight_numel`` floats per edge, is not formed), and the folded convolution saves its
    input ``[h c, c]`` for the backward: per edge, ``weight_numel`` less that width (OMAT-0 medium: 1,280 - 65)."""
    weights, folded = _generic_folded(dims)
    return -(weights - folded) * width, 0


def generic_recompute_conv_weights(dims, width):
    """Without stored per-edge weights (the folded path keeps its input ``[h c, c]``, counted with
    ``generic_chunked_conv_backward``): the widest interaction's ``weight_numel`` per edge."""
    weights, _ = _generic_folded(dims)
    return -weights * width, 0


def chunked_conv_backward(dims, width):
    """The chunked convolution backward keeps no per-edge weight gradient of the whole edge set (measured)."""
    return -round(_CHUNKED_BACKWARD_SHARE * _stock_edge_floats(dims)) * width, 0


def recompute_conv_weights(dims, width):
    """wprime_backward without the stored per-edge weights: the streaming interactions' ``[E, W]`` weights are not
    kept (MH-1: the second interaction's, 1,280 floats per edge; the first takes the reassociated path). With the
    fused forward the weights never exist in memory (CAP-T07b: built on chip in both passes). It keeps
    ``_CONV_WEIGHTS_KEEP`` of the remaining per-edge activations."""
    return 0, 0, _CONV_WEIGHTS_KEEP, 1.0


def recompute_first_table(dims, width):
    """The first interaction's table output ``[h c, c]`` is not kept for the backward (the region keeps the lengths,
    which the table keeps anyway; 260 B per edge at NbCl5 and NaCl). It keeps ``_FIRST_TABLE_KEEP`` of the remaining
    per-edge activations."""
    return 0, 0, _FIRST_TABLE_KEEP, 1.0


def recompute_node_segments(dims, width):
    """The node-local segments recomputed in the backward keep only their inputs: ``_NODE_SEGMENTS_KEEP`` of the
    remaining per-node activations (CAP-T08)."""
    return 0, 0, 1.0, _NODE_SEGMENTS_KEEP


def recompute_node_region(dims, width):
    """The combined node region in atom chunks, with the message divided inside its chunks, keeps
    ``_NODE_REGION_KEEP`` of the separate segments' remaining per-node activations (ninth list item 2, M3
    target 1)."""
    return 0, 0, 1.0, _NODE_REGION_KEEP


def gate_epilogue(dims, width):
    """The gate's backward in ``linear_2``'s epilogue keeps more for the backward than the separate gate:
    ``_GATE_EPILOGUE_KEEP`` of the remaining per-node activations (docs/validation/capacity/region-norm.md)."""
    return 0, 0, 1.0, _GATE_EPILOGUE_KEEP


def gate_epilogue_in_segments(dims, width):
    """``gate_epilogue`` with the separate node segments (``recompute_node_segments``), whose recomputation keeps
    the epilogue's extra buffers live at the peak: ``_GATE_EPILOGUE_SEGMENTS_KEEP`` of the remaining per-node
    activations (planner.memory.CONTEXT_TERMS)."""
    return 0, 0, 1.0, _GATE_EPILOGUE_SEGMENTS_KEEP


def radial_tables(dims, width):
    """Tabulated radial networks keep no hidden activations per edge (measured)."""
    return -round(_RADIAL_TABLES_SHARE * _stock_edge_floats(dims)) * width, 0


def generic_radial_tables(dims, width):
    """Tabulated generic radial networks keep, per edge, none of what the estimate counts for them besides their
    outputs: each network's input and its saved hidden units (``planner.memory._network_floats`` less the output,
    for the radial network and the interaction's other ones, the density's). The table's output ``[h c, c,
    density]`` stays (the folded path saves it; generic_chunked_conv_backward's term counts it). Not the real saved
    tensors (two per hidden unit): the estimate's stock count holds one, and removing more made rule 12 predict a
    third of the measured peak (docs/validation/omat0-capacity-2.md)."""
    from fastmace.planner import memory as estimate  # planner.memory imports nothing from torch or MACE

    def network(widths, saved):
        return estimate._network_floats(widths, saved) - (widths[-1] if len(widths) > 1 else 0)

    removed = sum(network(i.radial, i.radial_saved) + sum(network(w, s) for w, s in i.other_radials)
                  for i in dims.interactions)
    return -removed * width, 0


def _table_output(interaction):
    """The floats per edge of a generic interaction's table output ``[h c, c]``: the last radial layer's input and
    the cutoff column."""
    return interaction.radial[-2] + 1


def generic_recompute_first_table(dims, width):
    """The reassociated first interaction's table output is not kept for the backward (it keeps the lengths, which
    the table keeps anyway). With stored weights the other interactions keep the weights, not the table output."""
    return -_table_output(dims.interactions[0]) * width, 0


def generic_recompute_first_table_with_weights(dims, width):
    """``generic_recompute_first_table`` with ``generic_recompute_conv_weights``: no interaction keeps its table
    output (the chunked convolution evaluates it again)."""
    return -sum(_table_output(i) for i in dims.interactions) * width, 0


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


__all__ = ["block_linear_resident", "chunked_conv_backward", "factor_species",
           "fold_cutoff", "fused_radial_hidden", "gate_epilogue", "gate_epilogue_in_segments", "generic_reassociate",
           "joint_radial_input",
           "merged_residual_resident", "mh1_streaming_conv", "radial_tables", "reassociate_first",
           "recompute_conv_weights", "recompute_first_table", "recompute_node_segments", "segment_gemm_selfcheck",
           "streaming_conv"]
