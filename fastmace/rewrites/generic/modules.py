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

from fastmace.kernels.gpu_models import GPUModels

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
# The tabled first interaction's table and outer sums with one fused backward (round 2, 5 November 2026;
# kernels.triton.tabled_outer_sum): per edge dr and the edge attributes' gradient from the slots' gradient in one walk,
# without grad_a [E, h c], its assembled copy and the separate table backward. Rounding (regrouped sums). Needs the
# segmented first sum and float32. None: on for FUSED_FIRST_BACKWARD_GPUS (decision 66, after the MD-step A/Bs);
# True / False: an A/B side's choice on every GPU. MD step b/a (ab / ba), OMAT-0 medium, SrTiO3 10x10x10 and NbCl5:
# RTX 5060 Ti 0.962 / 0.963 and 0.972 / 0.969; RTX 4000 Ada 0.962 / 0.963 and 0.979 / 0.975 (with TABLED_FORWARD:
# 0.945 / 0.945, 0.948 / 0.955; 0.949 / 0.944, 0.978 / 0.968); A100-SXM4-80GB 0.969 / 0.966 and 0.983 / 0.984
# (0.966 / 0.962, 0.970 / 0.982), its NbCl5 stress error 1.55-1.60e-7, under the default plan's accepted 1.8e-7
# (docs/current/omat0-a100.md).
FUSED_FIRST_BACKWARD = None
# NVIDIA H200 (5 November 2026, results/h200-defaults-20261105): MD step b/a 1.051 / 1.052 (b = off; OMAT-0 medium,
# SrTiO3 10x10x10, abab / abba, every other A100 switch on).
FUSED_FIRST_BACKWARD_GPUS = GPUModels(("NVIDIA GeForce RTX 5060 Ti", "NVIDIA RTX 4000 Ada Generation",
                                       "NVIDIA A100-SXM4-80GB", "NVIDIA H200"))


def fused_first_backward(device):
    """``FUSED_FIRST_BACKWARD`` on ``device``: the constant when set, else on for ``FUSED_FIRST_BACKWARD_GPUS``."""
    if FUSED_FIRST_BACKWARD is not None:
        return bool(FUSED_FIRST_BACKWARD)
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_name(device) in FUSED_FIRST_BACKWARD_GPUS


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


def fused_first_applies(self, edge_feats):
    """Whether ``fused_first_message`` serves this tabled, reassociated interaction (``FUSED_FIRST_BACKWARD``)."""
    table, reassociation = getattr(self, "radial_table", None), getattr(self, "first_reassociation", None)
    return (table is not None and reassociation is not None and not FIRST_FULL_ROWS and reassociation.segmented
            and table.coef.dtype == torch.float32 and edge_feats.dtype == torch.float32
            and fused_first_backward(table.coef.device))


def fused_first_message(self, edge_feats, node_feats, node_attrs, edge_attrs, edge_index, rows=None,
                        density_of=None):
    """``(message, edge density)`` through the tabled outer sum (the table read inside the reassociation's per-slot
    sums, one fused backward; ``kernels.triton.tabled_outer_sum``), or None where ``fused_first_applies`` is false.
    ``density_of(edge density) -> density``: the reassociation divides its blocks by ``density + 1`` in the products'
    epilogue (idea 31, ``kernels.cutlass.first_div``) and returns them as a tuple where it can; the result is then
    ``(message, edge density, density)``."""
    if not fused_first_applies(self, edge_feats):
        return None
    from fastmace.kernels.triton.tabled_outer_sum import tabled_outer_sum

    table, reassociation = self.radial_table, self.first_reassociation
    weight = fc_last_weight(self.radial_last)
    width = weight.shape[-1]
    pair = table.pair(node_attrs, edge_index)
    density = []

    def summed_fn(key, slots):
        summed, edge_density = tabled_outer_sum(edge_feats, pair, table.coef, table.dcoef, table.starts, table.scales,
                                                table.bins, table.r_lo, table.bin_scale, table.r_max, width,
                                                edge_attrs.contiguous(), key, slots)
        density.append(edge_density)
        return summed, width

    computed = []

    def divided_by():
        computed.append(density_of(density[0]))
        return computed[0]

    message = reassociation(node_feats, node_attrs, edge_attrs, None, edge_index, weight, rows=rows,
                            summed_fn=summed_fn, divided_by=divided_by if density_of is not None else None)
    if density_of is not None:
        return message, density[0], computed[0] if computed else None
    return message, density[0]


def first_table_region(self, edge_feats, node_feats, node_attrs, edge_attrs, edge_index):
    """``(message, edge density)`` of a reassociated generic interaction with a radial table: the table, then the
    reassociation of its ``h c``. ``recompute_first_table`` runs it under non-reentrant checkpoint, so the backward
    keeps the lengths instead of the table's ``[E, h c + 1]`` output and evaluates the table again (MH-1's
    ``modules.first_table_region``); the reassociation's backward needs nothing else of the region, so the compiled
    backward rebuilds only the table."""
    fused = fused_first_message(self, edge_feats, node_feats, node_attrs, edge_attrs, edge_index)
    if fused is not None:
        return fused
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


def first_div_applies(self, kind, n_real):
    """Whether the first-interaction blocks go through ``kernels.cutlass.first_div`` (``first_div`` set at build by
    ``precision.route``): a density interaction without halo rows whose linear is the identity (the generic fold)
    and whose skip is a ``SpeciesSkip`` in ir_mul with one input segment per reassociated path."""
    reassociation = getattr(self, "first_reassociation", None)
    skip = getattr(self, "skip_tp", None)
    if (kind != "density" or n_real is not None or not getattr(reassociation, "first_div", False)
            or not isinstance(self.linear, torch.nn.Identity) or not isinstance(skip, SpeciesSkip)
            or skip.layout != "ir_mul" or len(skip.segments_in) != len(reassociation.ls)):
        return False
    c = reassociation.channels
    return all(mul == c and dim == c * (2 * degree + 1)
               for (_, mul, dim), degree in zip(skip.segments_in, reassociation.ls))


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
    early_density = None  # first_div: the density summed before the products
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
    elif tabled and reassociation is not None and fused_first_applies(self, edge_feats) and first_div_applies(self,
                                                                                                         kind, n_real):
        # idea 31 (kernels.cutlass.first_div): the density first, the blocks divided in the epilogue, kept apart
        message, table_density, early_density = fused_first_message(
            self, edge_feats, node_feats, node_attrs, edge_attrs, edge_index, rows=n_real,
            density_of=lambda edge_density: scatter_sum(src=edge_density, index=edge_index[1], dim=0,
                                                        dim_size=num_nodes))
    elif tabled and reassociation is not None and fused_first_applies(self, edge_feats):
        message, table_density = fused_first_message(self, edge_feats, node_feats, node_attrs, edge_attrs, edge_index,
                                                     rows=n_real)
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
    density = early_density
    if kind != "residual" and density is None:
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
    if isinstance(message, tuple) and (n_real is not None or kind == "residual"
                                       or getattr(self, "density_before_linear", False)):
        from fastmace.kernels.oeq_streaming.wprime_backward import dense_message
        wprime = self.wprime_backward  # BLOCK_MESSAGE on a path that needs the whole message
        message = dense_message(message, wprime.block[1], wprime.kernels.schedule.channels,
                                wprime.kernels.schedule.output_dim)
    if isinstance(message, tuple):  # BLOCK_MESSAGE: linear_1's input groups, every row real
        pass
    elif message.shape[0] != n_real:  # (the streaming convolution may have built the real rows only)
        message = self.truncate_ghosts(message, n_real)
    node_attrs = self.truncate_ghosts(node_attrs, n_real)
    if kind == "residual":
        sc = self.truncate_ghosts(sc, n_real)
        message = self.linear(message) / self.avg_num_neighbors
        return self.reshape(message), sc
    divided = None
    if early_density is not None and isinstance(message, tuple):
        pass  # first_div: the blocks are divided already (the linear is the identity)
    elif getattr(self, "density_before_linear", False):  # GENERIC_DENSITY_FIRST (not carried over); bias-free linear
        message = self.linear(message / (density + 1))
    else:
        # idea 31 (kernels.cutlass.linear_div): the division and the join in the products' epilogue, bitwise
        divide = getattr(self.linear, "divided", None)
        divided = divide(message, density) if divide is not None else None
        message = divided if divided is not None else self.linear(message) / (density + 1)
    if kind == "density":
        message = self.skip_tp(message, node_attrs)
        return self.reshape(message), None
    sc = self.truncate_ghosts(sc, n_real)
    if divided is not None:  # the epilogue's output is reshape_irreps' [n, irreps, mul] tensor already
        from fastmace.rewrites.linear.modules import reshape_view

        view = reshape_view(self.reshape, message)
        if view is not None:
            return view, sc
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
# CUDA float32 only (4 November 2026, Benchmark_loup speed goal). None: on for SPECIES_GEMM_GPUS (decision 66;
# owner, 4 November 2026: "Yes both"); True / False: an A/B side's choice on every GPU. Benchmark_loup (OMAT-0
# medium, SrTiO3 5,000 atoms) MD step b/a in both orders: RTX 4000 Ada 0.955 / 0.952, RTX 5060 Ti 0.950 / 0.953;
# A100-SXM4-80GB 0.992 / 0.991 (not adopted there: its skip runs as 3xTF32, where the Triton dot is about 3x slower
# per product than the concatenated CUTLASS GEMM; our own mma.sync kernel, species_gemm.TENSOR_CORES, did no better).
SPECIES_GEMM = None
SPECIES_GEMM_GPUS = ("NVIDIA GeForce RTX 5060 Ti", "NVIDIA RTX 4000 Ada Generation")


# Round 5 (5 November 2026): the edge spherical harmonics (e3nn, l <= 3, normalised, component) by
# fastmace.kernels.triton.edge_sh.edge_sh3, one Triton kernel each way, instead of Inductor's compiled e3nn
# polynomials, which save about 16 per-edge intermediates for autograd (A100 profile at 5b6c5550: about 230 µs of
# edge geometry per force call). Rounding (the backward's sums regrouped; 5.5e-7 / 1.4e-7 of float64 e3nn).
# Applied by generic_streaming (the model-level module; float32 CUDA vectors only, anything else falls through).
# None: on for EDGE_SH_GPUS (decision 66; every A100 model through GPUModels); True / False: an A/B side's choice on
# every GPU. Benchmark_loup (OMAT-0 medium) MD step b/a (ab / ba), SrTiO3 10x10x10 and NbCl5, gates pass (the A100's
# NbCl5 stress exception as before): RTX 5060 Ti 0.985 / 0.961, 0.990 / 0.993; RTX 4000 Ada 0.989 / 0.987,
# 0.998 / 0.997; A100 80GB PCIe 0.983 / 0.985, 0.995 / 0.988 (results/round5-20261105).
EDGE_SH = None
# NVIDIA H200 (5 November 2026, results/h200-defaults-20261105): MD step b/a 1.019 / 1.018 (b = off; OMAT-0 medium,
# SrTiO3 10x10x10, abab / abba, every other A100 switch on).
EDGE_SH_GPUS = GPUModels(("NVIDIA GeForce RTX 5060 Ti", "NVIDIA RTX 4000 Ada Generation", "NVIDIA A100-SXM4-80GB",
                          "NVIDIA A100 80GB PCIe", "NVIDIA H200"))


def edge_sh_default(device=None):
    """``EDGE_SH`` on ``device`` (a torch.device, a device name or None): the constant when set, else on for
    ``EDGE_SH_GPUS``."""
    if EDGE_SH is not None:
        return bool(EDGE_SH)
    if isinstance(device, torch.device):
        if device.type != "cuda":
            return False
        device = torch.cuda.get_device_name(device)
    return device is not None and device in EDGE_SH_GPUS


class EdgeSH(torch.nn.Module):
    """``o3.SphericalHarmonics([0, 1, 2, 3], normalize=True, normalization="component")`` for float32 CUDA
    vectors through ``edge_sh3``; any other input through the original module."""

    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, x):
        if x.is_cuda and x.dtype == torch.float32 and x.dim() == 2 and x.shape[1] == 3:
            from fastmace.kernels.triton.edge_sh import edge_sh3
            return edge_sh3(x)
        return self.original(x)


def edge_sh_replaceable(module):
    """Whether ``module`` is e3nn's spherical harmonics that ``EdgeSH`` reproduces."""
    if type(module).__name__ != "SphericalHarmonics" or not getattr(module, "normalize", False):
        return False
    if getattr(module, "normalization", None) != "component":
        return False
    return [int(l) for l in getattr(module, "_ls_list", [])] == [0, 1, 2, 3]


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
        # SPECIES_GEMM None: the device's default, fixed when the module is built on it
        self.species_gemm_gpu = (columns.is_cuda and torch.cuda.get_device_name(columns.device) in SPECIES_GEMM_GPUS)

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
        blocks_in = x1 if isinstance(x1, tuple) else None  # first_div: each input segment as [atoms, size, mul]
        x1 = x1[0] if blocks_in is not None else x1
        on = self.species_gemm_gpu if SPECIES_GEMM is None else SPECIES_GEMM
        grouped = on and x1.is_cuda and x1.dtype == torch.float32
        if grouped:
            from fastmace.kernels.triton.species_gemm import species_mm, species_plan
            kind = onehot.argmax(dim=1)
            precision = "tf32x3" if self.tf32x3 else "ieee"
            order, counts = species_plan(kind, species)
        for index, (i, j) in enumerate(self.pairs):
            weight = getattr(self, f"w{index}")
            xi = blocks_in[i] if blocks_in is not None else self._view(x1, self.segments_in[i])  # [atoms, size, mul]
            if grouped:
                y = species_mm(xi, getattr(self, f"ws{index}"), getattr(self, f"wt{index}"), kind, order, counts,
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
