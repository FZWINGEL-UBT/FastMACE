"""Modules and forward of the generic family: MACE-MP-0, the density family and MACE-OFF's later blocks.

``generic_interaction_forward`` reproduces the mace-torch 0.3.16 forward of every class in
``GENERIC_INTERACTIONS`` (its source is pinned by sha256) and reads the attributes the generic
records set on an interaction:

- ``first_reassociation`` (``generic_reassociate``): the first interaction summed per receiver
  and sender species before its last radial layer;
- ``fused_radial_backward`` (``generic_fused_radial_backward``): the last radial layer as
  ``hidden @ W.T`` through ``folded_linear``, whose Triton input gradient replaces cuBLAS's;
- ``streaming_kernels`` (``generic_streaming``): the receiver-streaming OEq convolution;
- ``chunked_backward_edges`` and ``generic_folded`` (``generic_chunked_conv_backward``): the streaming convolution
  takes the last radial layer's input ``[h c, c]`` and its weight ``[W, 0]`` (MH-1's folded layer,
  ``GenericFoldedRadial``), so its backward runs in L2-sized edge chunks and the ``[E, W]`` weight gradient is
  never formed in full;
- ``skip_tp`` replaced by a ``SpeciesSkip`` (``generic_species_skip``), which needs no change of
  the forward;
- ``radial_table`` (``generic_radial_tables``, fastmace.rewrites.radial): with edge lengths ``[E, 1]`` as
  ``edge_feats`` (the record's ``LengthsEmbedding``), the certified table gives ``[h c, c]`` and the edge density in
  place of the radial network, the cutoff product and the density network.

``density_before_linear`` (the research engine's ``GENERIC_DENSITY_FIRST``) is still read, so the
forward stays a faithful copy, but no record sets it: it was not carried over (phase 1a
decision after wave 0).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import torch

# Class name -> (variant, sha256 of inspect.getsource(<class>.forward) in mace-torch 0.3.16);
# generic_interaction_forward reproduces each of them.
GENERIC_INTERACTIONS: Mapping[str, tuple[str, str]] = MappingProxyType({
    "RealAgnosticResidualInteractionBlock":
        ("residual", "5148c60b11fd7dcb8fca624aa0447b9e508fcb4d7869a2b3d79e1ac14f768b62"),
    "RealAgnosticDensityInteractionBlock":
        ("density", "80fd7bc7ea53c17b3212db3ff42ac87b6e42ee8e986c9b408aec06922ad91dad"),
    "RealAgnosticDensityResidualInteractionBlock":
        ("density_residual", "228610bc97daa842d6ecc8770142d1ad4bce24cc3060fecebc13e3df726a8316"),
})


def source_hashes():
    """{"mace.modules.blocks:<class>.forward": sha256} for Rewrite.source_hashes."""
    return {f"mace.modules.blocks:{name}.forward": digest for name, (_, digest) in GENERIC_INTERACTIONS.items()}


def fc_split(network):
    """(hidden layers, last layer) of a bias-free e3nn FullyConnectedNet without output activation."""
    layers = list(network.children())
    last = layers[-1] if layers else None
    if type(network).__name__ != "FullyConnectedNet" or getattr(last, "act", True) is not None:
        raise ValueError(f"needs an e3nn FullyConnectedNet without output activation, got {type(network).__name__}")
    return torch.nn.Sequential(*layers[:-1]), last


# The reassociated first interaction with radial tables reads the whole table row [h c, c] against [W, 0] (as MH-1
# reads [h c, c] against [W, b]) instead of the slice h c against W: no strided copy of [E, 64] and no zero-padding
# slice backward; the cutoff column only adds zero products. Exact to rounding (the per-path products' K grows by
# one zero row per species slot). A module constant for the A/B (OMAT-0 A100 profile, 3 November 2026).
FIRST_FULL_ROWS = False
# The tabled first interaction's table evaluated for the reassociation's columns only (the folded input without its
# cutoff column, which the reassociation drops): no [E, h c + 1] output sliced, and its backward reads the [E, h c]
# gradient as it is, without the zero-padded copy (OMAT-0 A100 profile, 4 November 2026: 0.11 ms per force call).
# Bitwise: the dropped column's gradient is zero. A module constant for the A/B.
FIRST_TABLE_COLUMNS = False


def _first_inputs(augmented, last):
    """(table rows, last-layer weight) for the reassociation of a tabled generic first interaction."""
    weight = fc_last_weight(last)
    if FIRST_FULL_ROWS:
        return augmented, torch.nn.functional.pad(weight, (0, 1))
    if augmented.shape[1] == weight.shape[-1]:  # FIRST_TABLE_COLUMNS: the table returned these columns only
        return augmented, weight
    return augmented[:, :-1], weight


def fc_last_weight(last):
    """Effective [out, in] matrix of an e3nn last layer: x @ w / sqrt(h_in var_in / var_out) == x @ result.T."""
    return (last.weight / (last.h_in * last.var_in / last.var_out) ** 0.5).t()


def bias_free(linear):
    return not [name for name, tensor in (*linear.named_parameters(), *linear.named_buffers())
                if "bias" in name and tensor.numel() > 0]


def first_table_region(self, edge_feats, node_feats, node_attrs, edge_attrs, edge_index):
    """``(message, edge density)`` of a reassociated generic interaction with a radial table: the table, then the
    reassociation of its ``h c``. ``recompute_first_table`` runs it under non-reentrant checkpoint, so the backward
    keeps the lengths instead of the table's ``[E, h c + 1]`` output and evaluates the table again (MH-1's
    ``modules.first_table_region``); the reassociation's backward needs nothing else of the region, so the compiled
    backward rebuilds only the table."""
    columns = fc_last_weight(self.radial_last).shape[-1] if FIRST_TABLE_COLUMNS and not FIRST_FULL_ROWS else -1
    augmented, edge_density = self.radial_table(edge_feats, node_attrs, edge_index, columns=columns)
    rows, weight = _first_inputs(augmented, self.radial_last)
    message = self.first_reassociation(node_feats, node_attrs, edge_attrs, rows, edge_index, weight)
    return message, edge_density


def recomputed_first_table(self, edge_feats, node_feats, node_attrs, edge_attrs, edge_index):
    """``first_table_region`` under non-reentrant checkpoint (``generic_recompute_first_table``)."""
    from torch.utils.checkpoint import checkpoint

    return checkpoint(first_table_region, self, edge_feats, node_feats, node_attrs, edge_attrs, edge_index,
                      use_reentrant=False, preserve_rng_state=False)


def generic_interaction_forward(self, node_attrs, node_feats, edge_attrs, edge_feats, edge_index,
                                cutoff=None, lammps_class=None, lammps_natoms=(0, 0), first_layer=False):
    """RealAgnosticResidual / Density / DensityResidual InteractionBlock.forward (mace-torch 0.3.16)
    with the generic rewrites: streaming convolution, reassociated first interaction, fused radial backward."""
    from mace.tools.scatter import scatter_sum
    kind = self.generic_kind
    num_nodes = node_feats.shape[0]
    n_real = lammps_natoms[0] if lammps_class is not None else None
    # The skip connection of the real rows only: the ghost (halo) rows' are truncated below (a tile's halo is about
    # 40 % of its rows).
    real = slice(None) if n_real is None else slice(0, n_real)
    sc = self.skip_tp(node_feats[real], node_attrs[real]) if kind != "density" else None
    node_feats = self.linear_up(node_feats)
    node_feats = self.handle_lammps(node_feats, lammps_class=lammps_class, lammps_natoms=lammps_natoms,
                                    first_layer=first_layer)
    reassociation = getattr(self, "first_reassociation", None)
    # generic_chunked_conv_backward: the streaming convolution builds the weights from the last radial layer's input.
    chunked = reassociation is None and getattr(self, "chunked_backward_edges", 0) > 0
    table = getattr(self, "radial_table", None)
    tabled = table is not None and edge_feats.shape[1] == 1  # generic_radial_tables: edge_feats are the lengths
    # generic_recompute_first_table: the table's output is not kept for the backward, which evaluates it again.
    recompute_table = tabled and getattr(self, "recompute_table", False)
    table_lengths = (edge_feats if recompute_table and chunked and getattr(self, "generic_recompute", None) is not None
                     else None)
    # generic_wprime_backward: the table, the folded layer and the convolution in one op, its backward without the
    # per-edge weight gradient.
    wprime_lengths = (edge_feats if tabled and chunked and getattr(self, "generic_recompute", None) is None
                      and getattr(self, "wprime_backward", None) is not None else None)
    message = None
    if recompute_table and reassociation is not None:
        message, table_density = recomputed_first_table(self, edge_feats, node_feats, node_attrs, edge_attrs,
                                                        edge_index)
    elif table_lengths is not None:
        from fastmace.rewrites.mh1.modules import streaming_message
        message, table_density = streaming_message(self, node_feats, edge_attrs, None, edge_index, num_nodes,
                                                   node_attrs=node_attrs, table_lengths=table_lengths)
    elif wprime_lengths is not None:
        from fastmace.rewrites.mh1.modules import streaming_message
        message, table_density = streaming_message(self, node_feats, edge_attrs, None, edge_index, num_nodes,
                                                   node_attrs=node_attrs, wprime_lengths=wprime_lengths)
    elif tabled:
        augmented, table_density = table(edge_feats, node_attrs, edge_index,
                                         density=kind != "residual")  # [h c, c] and the density
    if tabled and (chunked or message is not None):
        pass
    elif tabled and reassociation is None:
        from fastmace.kernels.triton.skinny_linear import folded_linear
        tp_weights = folded_linear(augmented[:, :-1].contiguous(), fc_last_weight(self.radial_last).contiguous())
    elif chunked:
        hidden = self.radial_hidden(edge_feats)
        augmented = (torch.cat([hidden, torch.ones_like(hidden[:, :1])], dim=1) if cutoff is None
                     else torch.cat([hidden * cutoff, cutoff], dim=1))  # (h W) c = [h c, c] [W, 0]
    elif reassociation is None:
        if getattr(self, "fused_radial_backward", False):
            from fastmace.kernels.triton.skinny_linear import folded_linear
            tp_weights = folded_linear(self.radial_hidden(edge_feats), fc_last_weight(self.radial_last).contiguous())
        else:
            tp_weights = self.conv_tp_weights(edge_feats)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff
    density = None
    if kind != "residual":
        if tabled:
            edge_density = table_density
        else:
            edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)
            if cutoff is not None:
                edge_density = edge_density * cutoff
        density = scatter_sum(src=edge_density, index=edge_index[1], dim=0,
                              dim_size=num_nodes if n_real is None else n_real)  # every receiver is real
    if message is not None:
        pass
    elif reassociation is not None:
        if tabled:
            hidden, weight = _first_inputs(augmented, self.radial_last)  # h c and W, or [h c, c] and [W, 0]
        else:
            hidden, weight = self.radial_hidden(edge_feats), fc_last_weight(self.radial_last)
            if cutoff is not None:  # (h W) c = (h c) W
                hidden = hidden * cutoff
        message = reassociation(node_feats, node_attrs, edge_attrs, hidden, edge_index, weight,
                                rows=n_real)  # the tiles' receivers only
    elif getattr(self, "streaming_kernels", None) is not None:
        from fastmace.rewrites.mh1.modules import streaming_message
        message = streaming_message(self, node_feats, edge_attrs, None if chunked else tp_weights, edge_index,
                                    num_nodes, augmented=augmented if chunked else None, rows=n_real)
    elif hasattr(self, "conv_fusion"):
        message = self.conv_tp(node_feats, edge_attrs, tp_weights, edge_index)
    else:
        mji = self.conv_tp(node_feats[edge_index[0]], edge_attrs, tp_weights)
        message = scatter_sum(src=mji, index=edge_index[1], dim=0, dim_size=num_nodes)
    if message.shape[0] != n_real:  # (the streaming convolution may have built the real rows only)
        message = self.truncate_ghosts(message, n_real)
    node_attrs = self.truncate_ghosts(node_attrs, n_real)
    if kind == "residual":
        sc = self.truncate_ghosts(sc, n_real)
        message = self.linear(message) / self.avg_num_neighbors
        return self.reshape(message), sc
    if getattr(self, "density_before_linear", False):  # GENERIC_DENSITY_FIRST (not carried over); bias-free linear
        message = self.linear(message / (density + 1))
    else:
        message = self.linear(message) / (density + 1)
    if kind == "density":
        message = self.skip_tp(message, node_attrs)
        return self.reshape(message), None
    sc = self.truncate_ghosts(sc, n_real)
    return self.reshape(message), sc


class GenericFoldedRadial(torch.nn.Module):
    """A generic interaction's last radial layer as MH-1's folded layer (``generic_chunked_conv_backward``): the
    weight ``[W, 0]`` applied to ``[h c, c]`` (``(h W) c``; the zero column keeps MH-1's ``[W, b]`` layout for a
    bias-free layer), with its two halves as buffers for the fused input gradient (as ``folded_constants``)."""

    def __init__(self, last):
        super().__init__()
        weight = fc_last_weight(last).detach()
        zero = weight.new_zeros(weight.shape[0])
        self.register_buffer("folded_constant", torch.cat([weight, zero[:, None]], dim=1).contiguous())
        self.register_buffer("folded_main", weight.contiguous())
        self.register_buffer("folded_last", zero)
        self.folded_tiles = "fixed"

    def folded_weight(self):
        return self.folded_constant

    @property
    def folded_parts(self):
        return self.folded_main, self.folded_last


# Every atom's rows through its own species' matrix only (``fastmace.kernels.triton.species_gemm``), instead of
# through all present species' matrices followed by the selection of its block: a third of the products and no
# species-expanded intermediates for three species. Exact (the same dot products; bitwise in the species skip lab).
# CUDA float32 only. A module constant for the A/B (4 November 2026, Benchmark_loup speed goal).
SPECIES_GEMM = False


class SpeciesSkip(torch.nn.Module):
    """Element-dependent skip connection restricted to the structure's species (``generic_species_skip``).

    MACE's skip_tp is a fully connected tensor product of the node features with the
    one-hot node attributes (89x0e for the MP/MPA/OMAT/MatPES tables). For a one-hot
    input of element z it is a linear map L_z per irrep block, and cuEquivariance
    evaluates it for all elements (an [atoms, channels x 89] expansion). This keeps
    only the species present at setup: per block one GEMM against the concatenated
    [channels, species x channels] matrices, then the atom's own column block.
    Matrices are measured from the original module; atoms may change places later,
    but not to species absent at setup.
    """

    def __init__(self, columns, segments_in, segments_out, blocks, layout):
        super().__init__()
        self.register_buffer("columns", columns)
        self.segments_in = [tuple(s) for s in segments_in]
        self.segments_out = [tuple(s) for s in segments_out]
        self.layout = layout
        self.pairs = []
        for index, (i, j, weight) in enumerate(blocks):  # weight [species, mul_in, mul_out]
            species, mul_in, mul_out = weight.shape
            self.register_buffer(f"w{index}", weight.permute(1, 0, 2).reshape(mul_in, species * mul_out).contiguous())
            self.register_buffer(f"ws{index}", weight.contiguous(), persistent=False)  # [species, mul_in, mul_out]
            self.register_buffer(f"wt{index}", weight.transpose(1, 2).contiguous(), persistent=False)
            self.pairs.append((i, j))
        self.tf32x3 = False  # tf32x3_gemms: the products as 3xTF32 (OPT4-T03)

    def _view(self, x, segment):
        offset, mul, dim = segment
        block = x[:, offset:offset + dim]
        if self.layout == "ir_mul":  # component m of channel u at m * mul + u
            return block.reshape(x.shape[0], dim // mul, mul)
        return block.reshape(x.shape[0], mul, dim // mul).transpose(1, 2)

    def forward(self, x1, x2):
        onehot = x2[:, self.columns]  # [atoms, species]
        atoms, species = onehot.shape
        outputs = [None] * len(self.segments_out)
        grouped = SPECIES_GEMM and x1.is_cuda and x1.dtype == torch.float32
        if grouped:
            from fastmace.kernels.triton.species_gemm import species_mm, species_plan
            kind = onehot.argmax(dim=1)
            precision = "tf32x3" if self.tf32x3 else "ieee"
            plans = {}
        for index, (i, j) in enumerate(self.pairs):
            weight = getattr(self, f"w{index}")
            xi = self._view(x1, self.segments_in[i])  # [atoms, size, mul_in]
            if grouped:
                size = xi.shape[1]
                if size not in plans:
                    plans[size] = species_plan(kind, species, size)
                y = species_mm(xi, getattr(self, f"ws{index}"), getattr(self, f"wt{index}"), kind, *plans[size],
                               precision)
                outputs[j] = y if outputs[j] is None else outputs[j] + y
                continue
            size, mul_out = xi.shape[1], weight.shape[1] // species
            left = xi.reshape(atoms * size, -1)
            if self.tf32x3:
                from fastmace.kernels.triton.tf32x3_gemm import tf32x3_matmul

                y = tf32x3_matmul(left, weight).view(atoms, size, species, mul_out)
            else:
                y = (left @ weight).view(atoms, size, species, mul_out)
            y = (y * onehot[:, None, :, None]).sum(dim=2)  # the atom's own species block
            outputs[j] = y if outputs[j] is None else outputs[j] + y
        blocks = []
        for j, (offset, mul, dim) in enumerate(self.segments_out):
            y = outputs[j]
            if y is None:
                y = x1.new_zeros(atoms, dim // mul, mul)
            if self.layout == "ir_mul":
                blocks.append(y.reshape(atoms, dim))
            else:
                blocks.append(y.transpose(1, 2).reshape(atoms, dim))
        return torch.cat(blocks, dim=1)


def irrep_segments(irreps):
    from e3nn import o3
    segments, offset = [], 0
    for mul, ir in o3.Irreps(str(irreps)):
        segments.append((offset, mul, mul * ir.dim, str(ir)))
        offset += mul * ir.dim
    return segments


def scalar_second_input(skip):
    """The element count of a skip_tp whose second input is all scalars (one-hot attributes), else None."""
    from e3nn import o3
    irreps_in2 = getattr(skip, "irreps_in2", None)
    if irreps_in2 is None:
        return None
    irreps = o3.Irreps(str(irreps_in2))
    if any(ir != o3.Irrep("0e") for _, ir in irreps):
        return None
    return irreps.dim


def species_skip(skip, columns):
    """A SpeciesSkip measured from ``skip`` (fully connected TP with one-hot scalar second input), or None."""
    elements = scalar_second_input(skip)
    if elements is None:
        return None
    segments_in, segments_out = irrep_segments(skip.irreps_in1), irrep_segments(skip.irreps_out)
    dim_in = sum(s[2] for s in segments_in)
    parameter = next(skip.parameters())
    device, dtype = parameter.device, parameter.dtype
    tolerance = 1e-4 if dtype == torch.float32 else 1e-10
    with torch.no_grad():
        for layout in ("ir_mul", "mul_ir"):
            blocks = []
            for i, (offset, mul, dim, ir) in enumerate(segments_in):
                size = dim // mul
                basis = torch.zeros((mul, dim_in), device=device, dtype=dtype)
                position = offset + (torch.arange(mul) if layout == "ir_mul" else torch.arange(mul) * size)
                basis[torch.arange(mul), position] = 1  # component m = 0 of every channel
                for j, (offset_out, mul_out, dim_out, ir_out) in enumerate(segments_out):
                    if ir_out != ir:
                        continue
                    weights = []
                    for column in columns.tolist():
                        attrs = torch.zeros((mul, elements), device=device, dtype=dtype)
                        attrs[:, column] = 1
                        response = skip(basis, attrs)
                        out = (response[:, offset_out:offset_out + mul_out] if layout == "ir_mul" else
                               response[:, offset_out:offset_out + dim_out:dim_out // mul_out])
                        weights.append(out.clone())
                    blocks.append((i, j, torch.stack(weights)))
            replacement = SpeciesSkip(columns.to(device), [s[:3] for s in segments_in],
                                      [s[:3] for s in segments_out], blocks, layout).to(device)
            generator = torch.Generator(device="cpu").manual_seed(0)
            probe = torch.randn((11, dim_in), generator=generator, dtype=dtype).to(device)
            picks = columns[torch.randint(len(columns), (11,), generator=generator)]
            attrs = torch.zeros((11, elements), device=device, dtype=dtype)
            attrs[torch.arange(11), picks.to(device)] = 1
            expected, got = skip(probe, attrs), replacement(probe, attrs)
            scale = float(expected.abs().max()) + 1e-30
            if got.shape == expected.shape and float((got - expected).abs().max()) <= tolerance * scale:
                return replacement
    raise ValueError(f"generic_species_skip: replacement of {type(skip).__name__} does not reproduce it")


__all__ = ["GENERIC_INTERACTIONS", "SpeciesSkip", "bias_free", "fc_last_weight", "fc_split",
           "generic_interaction_forward", "irrep_segments", "scalar_second_input", "source_hashes", "species_skip"]
