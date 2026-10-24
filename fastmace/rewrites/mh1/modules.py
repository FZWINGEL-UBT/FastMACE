"""Modules and the folded interaction forward of the MH-1 rewrites (SPEC §6.2).

Copied from the research engine (``langevin/gpu_langevin.py``) with its module switches
replaced by attributes that the rewrite records of ``fastmace.rewrites.mh1`` set on each
interaction or radial network:

- ``FactoredRadialMLP``: ``fused_backward`` (fused_folded_backward), ``folded_tiles``
  (folded_backward_tiles), ``fused_hidden`` (fused_radial_hidden);
- the interaction: ``density_before_linear`` (density_first), ``joint_radial_weight`` and
  ``joint_radial_widths`` (joint_radial_input), ``first_reassociation`` (reassociate_first),
  ``streaming_kernels`` and ``streaming_count`` (streaming_conv), ``residual_linear``
  (merged_residual, ``fastmace.rewrites.linear``), ``gated_rows`` (gated_rows,
  ``fastmace.rewrites.fusion``: the residual products, gate and linear_2 on per-segment rows);
- ``FirstInteractionReassociation.segmented`` (segmented_first_sum).

The research engine's screen options that were not carried over (docs/design/switch-map.md
section 7: ``CONTIGUOUS_FOLDED_WEIGHT``, ``JOINT_FUSED_BACKWARD``, ``STREAMING_MUL_IR``,
``STREAMING_RECEIVER_BACKWARD``) are fixed at their research-engine value, off.
"""

from __future__ import annotations

import torch

from fastmace.kernels.triton.edge_outer_sum import edge_outer_sum
from fastmace.kernels.triton.fused_radial import run_hidden_plan
from fastmace.kernels.triton.skinny_linear import folded_linear
from fastmace.rewrites.fusion import READS_GATED_ROWS
from fastmace.rewrites.linear import READS_RESIDUAL_LINEAR


class ProjectedEmbedding(torch.nn.Module):
    """A species embedding followed by the species columns of several radial input layers.

    Returns [n_nodes, sum of outputs]: one block per radial network, each the
    product of the embedding with that network's columns (plus its first-layer
    bias for the target side, so each edge adds the bias exactly once).
    """

    def __init__(self, embedding, weights, bias=None):
        super().__init__()
        self.embedding = embedding
        self.weight = torch.nn.Parameter(torch.cat(weights, 0).detach().clone(), requires_grad=False)
        self.bias = (None if bias is None else
                     torch.nn.Parameter(bias.detach().clone(), requires_grad=False))

    def forward(self, node_attrs):
        return torch.nn.functional.linear(self.embedding(node_attrs), self.weight, self.bias)


class FactoredRadialMLP(torch.nn.Module):
    """MACE RadialMLP whose first layer reads the radial columns plus gathered species terms.

    Input columns: [radial (radial_dim), projected source blocks, projected
    target blocks]; ``source`` and ``target`` give this network's column
    ranges. Later layers (LayerNorm, SiLU, linears) are the original modules.
    """

    def __init__(self, mlp, radial_dim, source, target):
        super().__init__()
        first = mlp.net[0]
        self.weight = torch.nn.Parameter(first.weight[:, :radial_dim].detach().clone(),
                                         requires_grad=False)
        self.rest = mlp.net[1:]
        self.hs = getattr(mlp, "hs", None)
        self.radial_dim = int(radial_dim)
        self.source = tuple(int(i) for i in source)
        self.target = tuple(int(i) for i in target)
        self.fused_backward = False  # fused_folded_backward
        self.folded_tiles = "fixed"  # folded_backward_tiles: "waves" (opt-2 OPT2-T03)
        self.fused_hidden = None  # fused_radial_hidden: a fused_radial.hidden_plan

    def first(self, inputs):
        """Output of the first linear layer (before its activation) for the concatenated inputs."""
        hidden = torch.nn.functional.linear(inputs[:, :self.radial_dim], self.weight)
        return (hidden + inputs[:, self.source[0]:self.source[1]]
                + inputs[:, self.target[0]:self.target[1]])

    def forward(self, inputs, cutoff=None):
        return self.after_first(self.first(inputs), cutoff)

    def after_first(self, hidden, cutoff=None):
        """The network from its first layer's output on (joint_radial_input computes that output)."""
        if cutoff is None:
            if self.fused_hidden is not None:
                return self.rest[-1](run_hidden_plan(hidden, self.fused_hidden))
            return self.rest(hidden)
        # Cutoff folded into the last layer: one GEMM over [h c, c], (h W + b) c = [h c, c] [W; b].
        augmented = self.augmented_after_first(hidden, cutoff)
        if self.fused_backward:
            return folded_linear(augmented, self.folded_weight(), tile_rule=self.folded_tiles,
                                 parts=self.folded_parts)
        return torch.nn.functional.linear(augmented, self.folded_weight())

    def augmented(self, inputs, cutoff):
        """[h c, c]: the folded last layer's input; [h, 1] without a cutoff."""
        return self.augmented_after_first(self.first(inputs), cutoff)

    def augmented_after_first(self, hidden, cutoff):
        if self.fused_hidden is not None:
            hidden = run_hidden_plan(hidden, self.fused_hidden)
        else:
            hidden = self.rest[:-1](hidden)
        if cutoff is None:
            return torch.cat([hidden, torch.ones_like(hidden[:, :1])], dim=1)
        return torch.cat([hidden * cutoff, cutoff], dim=1)

    def folded_weight(self):
        """[W, b] of the last layer, applied to [h c, c]; the buffer of ``folded_constants`` where it is set."""
        constant = getattr(self, "folded_constant", None)
        if constant is not None:
            return constant
        last = self.rest[-1]
        return torch.cat([last.weight, last.bias[:, None]], dim=1)

    @property
    def folded_parts(self):
        """(W, b) as contiguous buffers for the fused input gradient (``folded_constants``), else None."""
        main = getattr(self, "folded_main", None)
        return None if main is None else (main, self.folded_last)

    def hold_folded_constants(self):
        """``folded_constants`` (OPT3-T05): [W, b] and its two halves as buffers, built once instead of a
        concatenation and two copies per force call. The weights are constant, so the values are the same."""
        weight = self.folded_weight().detach().clone().contiguous()
        self.register_buffer("folded_constant", weight)
        self.register_buffer("folded_main", weight[:, :-1].contiguous())
        self.register_buffer("folded_last", weight[:, -1].contiguous())


class FirstInteractionReassociation(torch.nn.Module):
    """First-interaction message without per-edge tensor-product weights (reassociate_first).

    With species-only scalar sender features x[z, u] (C channels), the folded
    last radial layer W (one C-row block per path) applied to a = [h c, c],
    and spherical harmonics Y, the message of receiver i is
        m[i, l, m, u] = gamma_l sum_z x[z, u] sum_k W[path(l) C + u, k] S[i, z, k, (l, m)],
        S[i, z, k, (l, m)] = sum over edges j -> i with species(j) = z of a[e, k] Y[e, (l, m)].
    gamma_l and the output layout are measured from the model's own convolution.
    ``columns`` are the node_attrs columns of the species superset (SPEC §8.5), present or not.
    ``species_feats`` is x: the [S, C] table of the first interaction's sender features after
    ``linear_up``, one row per superset column, from the identity basis (``identity_species_feats``),
    never from the atoms present. A force field binds its own table before every call (a model may
    be shared); the forward does not read ``node_feats``.
    """

    def __init__(self, columns, channels, ls, gammas, ir_mul, segmented=False, species_feats=None):
        super().__init__()
        self.register_buffer("columns", columns)
        self.register_buffer("species_feats", species_feats, persistent=False)
        self.segmented = bool(segmented)  # segmented_first_sum
        self.channels = int(channels)
        self.ls = tuple(int(l) for l in ls)  # noqa: E741
        self.gammas = tuple(float(g) for g in gammas)
        self.ir_mul = bool(ir_mul)
        self.tf32x3 = False  # tf32x3_gemms: the per-path products as 3xTF32 (OPT4-T03)

    def bind(self, species_feats):
        """Point the forward at ``species_feats`` (a host-side pointer swap; captured graphs keep their address)."""
        if self._buffers.get("species_feats") is not species_feats:
            self.register_buffer("species_feats", species_feats, persistent=False)

    def forward(self, node_feats, node_attrs, edge_attrs, augmented, edge_index, weight):
        onehot = node_attrs[:, self.columns]
        species = onehot.argmax(dim=1)
        x = self.species_feats  # identity basis of the superset (SPEC §8.5), not the first atom of each species
        if x is None:
            raise RuntimeError("FirstInteractionReassociation has no species_feats table: bind one "
                               "(identity_species_feats) before calling it")
        nodes, slots = onehot.shape
        width, dim = augmented.shape[1], edge_attrs.shape[1]
        key = edge_index[1] * slots + species[edge_index[0]]
        # [nodes, slots, (l, m), k]; fused backward (edge_outer_sum)
        summed = edge_outer_sum(augmented, edge_attrs, key, nodes * slots,
                                segmented=self.segmented).view(nodes, slots, dim, width)
        blocks = []
        start = 0
        c = self.channels
        for path, (l, gamma) in enumerate(zip(self.ls, self.gammas)):  # noqa: E741
            size = 2 * l + 1
            # [nodes * size, slots * width] @ [slots * width, C]
            left = summed[:, :, start:start + size].transpose(1, 2).reshape(nodes * size, slots * width)
            right = (gamma * x[:, None, :] * weight[path * c:(path + 1) * c].t()[None]).reshape(slots * width, c)
            if self.tf32x3:
                from fastmace.kernels.triton.tf32x3_gemm import tf32x3_matmul

                block = tf32x3_matmul(left, right).view(nodes, size, c)
            else:
                block = (left @ right).view(nodes, size, c)
            if not self.ir_mul:
                block = block.transpose(1, 2)
            blocks.append(block.reshape(nodes, size * c))
            start += size
        return torch.cat(blocks, dim=1)


def identity_attrs(columns, width, rows, dtype, device):
    """The identity basis of the superset ``columns`` as one-hot node attributes, [max(rows, S), width].

    Row s < S is the one-hot row of column ``columns[s]``; the rest are zero. ``rows`` is the
    structure's atom count: ``identity_species_feats`` evaluates the basis in a batch of the model's own
    row count (see there).
    """
    columns = torch.as_tensor(columns, dtype=torch.int64).cpu()
    attrs = torch.zeros((max(int(rows), len(columns)), int(width)), dtype=dtype)
    attrs[torch.arange(len(columns)), columns] = 1
    return attrs.to(device)


def identity_species_feats(model, interaction, attrs, count, joint=None):
    """x of ``FirstInteractionReassociation``: [count, C] sender features of the superset species.

    ``model.node_embedding`` of the identity basis ``attrs`` (``identity_attrs``), plus the per-graph
    joint embedding when the model has one (OMOL-0: ``joint`` is ``(batch, features)``, the zero graph
    index of every row and the structure's total charge and spin), then the first interaction's
    ``linear_up``; the first ``count`` rows. The batch has the row count of the model's own call: the
    GEMMs' row values depend on the row count but not on a row's position, so a present species gets
    bitwise the features its atoms have in the model (CPU float64, plan task 3-T02). No gradient: in the
    model these features are the species embedding, which does not depend on the positions.
    """
    with torch.no_grad():
        feats = model.node_embedding(attrs)
        if joint is not None and hasattr(model, "joint_embedding"):
            feats = feats + model.joint_embedding(*joint)  # MACE: node_feats += joint_embedding(...)
        return interaction.linear_up(feats)[:count]


def cutoff_folded_forward(self, node_attrs, node_feats, edge_attrs, edge_feats, edge_index,
                          cutoff=None, lammps_class=None, lammps_natoms=(0, 0), first_layer=False):
    """RealAgnosticResidualNonLinearInteractionBlock.forward (mace-torch 0.3.16), except that
    conv_tp_weights receives the cutoff and applies it itself (fold_cutoff).

    fastmace.rewrites.mh1 pins the sha256 of the copied forward's source (NONLINEAR_FORWARD_SHA256).
    """
    from mace.tools.scatter import scatter_sum
    num_nodes = node_feats.shape[0]
    n_real = lammps_natoms[0] if lammps_class is not None else None
    sc = self.skip_tp(node_feats)
    node_feats = self.linear_up(node_feats)
    residual_linear = getattr(self, "residual_linear", None)  # merged_residual
    node_feats_res = self.linear_res(node_feats) if residual_linear is None else None
    if lammps_class is None and getattr(self, "direct_node_feats", False):  # direct_node_feats (OPT3-T05)
        pass  # handle_lammps is the identity without LAMMPS: no join, so no strided slices to copy
    else:
        node_feats_attrs = torch.cat([node_feats, node_attrs], dim=-1)
        node_feats_attrs = self.handle_lammps(node_feats_attrs, lammps_class=lammps_class,
                                              lammps_natoms=lammps_natoms, first_layer=first_layer)
        node_feats = node_feats_attrs[:, : node_feats.shape[-1]]
        node_attrs = node_feats_attrs[:, node_feats.shape[-1]:]
    source_embedding = self.source_embedding(node_attrs)
    target_embedding = self.target_embedding(node_attrs)
    reassociation = getattr(self, "first_reassociation", None)  # reassociate_first
    joint = getattr(self, "joint_radial_weight", None)  # joint_radial_input
    if joint is not None:
        # Both first layers at once; the gathered species terms join inside the add.
        first = torch.nn.functional.linear(edge_feats, joint)
        hidden = (first + source_embedding[edge_index[0]]
                  + target_embedding[edge_index[1]])
        conv_hidden, density_hidden = hidden.split(self.joint_radial_widths, dim=1)
        if reassociation is None:
            tp_weights = self.conv_tp_weights.after_first(conv_hidden, cutoff)
        edge_density = torch.tanh(self.density_fn.after_first(density_hidden) ** 2)
    else:
        edge_feats = torch.cat([edge_feats, source_embedding[edge_index[0]],
                                target_embedding[edge_index[1]]], dim=-1)
        if reassociation is None:
            tp_weights = self.conv_tp_weights(edge_feats, cutoff)  # changed: cutoff applied inside
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)
    if cutoff is not None:
        edge_density = edge_density * cutoff
    density = scatter_sum(src=edge_density, index=edge_index[1], dim=0, dim_size=num_nodes)
    if reassociation is not None:
        augmented = (self.conv_tp_weights.augmented_after_first(conv_hidden, cutoff) if joint is not None
                     else self.conv_tp_weights.augmented(edge_feats, cutoff))
        message = reassociation(node_feats, node_attrs, edge_attrs, augmented, edge_index,
                                self.conv_tp_weights.folded_weight())
    elif getattr(self, "streaming_kernels", None) is not None:  # streaming_conv
        message = streaming_message(self, node_feats, edge_attrs, tp_weights, edge_index, num_nodes)
    elif hasattr(self, "conv_fusion"):
        message = self.conv_tp(node_feats, edge_attrs, tp_weights, edge_index)
    else:
        mji = self.conv_tp(node_feats[edge_index[0]], edge_attrs, tp_weights)
        message = scatter_sum(src=mji, index=edge_index[1], dim=0, dim_size=num_nodes)
    message = self.truncate_ghosts(message, n_real)
    density = self.truncate_ghosts(density, n_real)
    sc = self.truncate_ghosts(sc, n_real)
    gated_rows = getattr(self, "gated_rows", None)  # gated_rows: residual products, gate and linear_2 on rows
    if gated_rows is not None:
        message = gated_rows(message / (density * self.beta + self.alpha), node_feats)
        return (self.reshape(message), sc)
    if residual_linear is not None:  # with density_first (merge_residual_linears)
        message = residual_linear(message / (density * self.beta + self.alpha), node_feats)
    else:
        node_feats_res = self.truncate_ghosts(node_feats_res, n_real)
        if getattr(self, "density_before_linear", False):  # density_first; linear_1 is bias-free
            message = self.linear_1(message / (density * self.beta + self.alpha))
        else:
            message = self.linear_1(message) / (density * self.beta + self.alpha)
        message = message + node_feats_res
    message = self.equivariant_nonlin(message)
    message = self.linear_2(message)
    return (self.reshape(message), sc)


setattr(cutoff_folded_forward, READS_RESIDUAL_LINEAR, True)
setattr(cutoff_folded_forward, READS_GATED_ROWS, True)


def streaming_message(self, node_feats, edge_attrs, tp_weights, edge_index, num_nodes):
    """The interaction's OEq convolution through the receiver-streaming kernels (attached as streaming_kernels)."""
    from fastmace.kernels.oeq_streaming import streaming_convolution
    sender, receiver = edge_index[0], edge_index[1]
    # The compacted count (the engine registers the neighbour list's GPU counter) ends the
    # receiver-sorted edges within the cutoff; the padding after it goes past every row. (A float32
    # cutoff test is not enough: pairs at the cutoff can disagree with the float64 compaction.)
    position = torch.arange(receiver.shape[0], device=receiver.device)
    key = torch.where(position < self.streaming_count, receiver, torch.full_like(receiver, num_nodes))
    row_ptr = torch.searchsorted(key, torch.arange(num_nodes + 1, device=key.device)).to(torch.int32)
    # The kernels read and write the model's own node layout (ir_mul with cuEquivariance), so
    # OEq's layout transposes around the convolution are not needed.
    conv = self.conv_tp
    transpose = self.streaming_kernels.layout == "mul_ir" and conv.layout_transpose_in is not None
    feats = conv.layout_transpose_in(node_feats) if transpose else node_feats
    message = streaming_convolution(feats, edge_attrs, tp_weights, sender.to(torch.int32),
                                    receiver.to(torch.int32), row_ptr, self.streaming_kernels)
    if transpose:
        message = conv.layout_transpose_out(message)
    return message


def attach_streaming_kernels(interaction):
    """Receiver-streaming kernels for an interaction's OEq-fused convolution, in the model's layout.

    Sets ``interaction.streaming_kernels`` and a ``streaming_count`` buffer under which every edge
    counts, until the engine binds its neighbour list's GPU edge count
    (``fastmace.rewrites.mh1.bind_streaming_count``).
    """
    import numpy as np

    from fastmace.kernels.oeq_streaming import StreamingKernels
    weight = next(interaction.parameters())
    conv = interaction.conv_tp
    ir_mul = conv.layout_transpose_in is not None  # OEq transposes around its kernel
    interaction.streaming_kernels = StreamingKernels(
        conv.input_args["problem"], np.float32 if weight.dtype == torch.float32 else np.float64,
        layout="ir_mul" if ir_mul else "mul_ir", receiver_backward=False)
    interaction.register_buffer("streaming_count", torch.tensor(torch.iinfo(torch.int64).max, device=weight.device),
                                persistent=False)


def probe_first_tensor_product(interaction):
    """(channels, ls, gammas, ir_mul) of a scalar-input channelwise first convolution.

    Measured from the interaction's own conv_tp (e3nn, cuEquivariance or
    OEq), so its path coefficients and output layout are used as they are.
    Raises unless every output equals gamma_l x[u] Y[l, m] w[path, u].
    """
    from e3nn import o3
    edge_irreps = o3.Irreps(str(interaction.edge_irreps))
    attr_irreps = o3.Irreps(str(interaction.edge_attrs_irreps))
    if len(edge_irreps) != 1 or edge_irreps[0].ir != o3.Irrep("0e"):
        raise ValueError(f"First-interaction sender features {edge_irreps} are not one scalar block")
    if any(mul != 1 for mul, _ in attr_irreps):
        raise ValueError(f"Unexpected edge attributes {attr_irreps}")
    channels = edge_irreps[0].mul
    ls = [ir.l for _, ir in attr_irreps]
    if interaction.conv_tp.weight_numel != channels * len(ls):
        raise ValueError("First convolution is not channelwise with one weight per path")
    parameter = next(interaction.parameters())
    device, dtype = parameter.device, parameter.dtype
    edge_index = torch.tensor([[1], [0]], device=device)

    def convolve(x, y, w):
        with torch.no_grad():
            if hasattr(interaction, "conv_fusion"):
                out = interaction.conv_tp(x, y, w, edge_index)
            else:
                out = torch.zeros((2, sum(channels * (2 * ell + 1) for ell in ls)), dtype=dtype, device=device)
                out = out.index_add(0, edge_index[1], interaction.conv_tp(x[edge_index[0]], y, w))
        return out[0].double().cpu()

    dim = attr_irreps.dim
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    ones = convolve(torch.ones((2, channels), dtype=dtype, device=device),
                    torch.ones((1, dim), dtype=dtype, device=device),
                    torch.ones((1, channels * len(ls)), dtype=dtype, device=device))
    gammas, start = [], 0
    for l in ls:  # noqa: E741
        block = ones[start:start + channels * (2 * l + 1)]
        # Equal up to rounding: cuEq's CPU (naive) kernel gives them to within 1e-15 (the research
        # engine demanded exact equality, which only its GPU convolutions met).
        if not torch.allclose(block, block[0].expand_as(block), rtol=tolerance, atol=0):
            raise ValueError("First convolution coefficients differ within a path")
        gammas.append(float(block[0]))
        start += block.numel()
    generator = torch.Generator().manual_seed(11)
    x = torch.rand((2, channels), generator=generator, dtype=torch.float64) + 0.5
    y = torch.rand((1, dim), generator=generator, dtype=torch.float64) + 0.5
    w = torch.rand((1, channels * len(ls)), generator=generator, dtype=torch.float64) + 0.5
    reference = convolve(x.to(device, dtype), y.to(device, dtype), w.to(device, dtype))
    for ir_mul in (False, True):
        blocks, start = [], 0
        for path, (l, gamma) in enumerate(zip(ls, gammas)):  # noqa: E741
            size = 2 * l + 1
            block = gamma * (x[1] * w[0, path * channels:(path + 1) * channels])[:, None] * y[0, start:start + size]
            blocks.append((block.t() if ir_mul else block).reshape(-1))
            start += size
        if torch.allclose(torch.cat(blocks), reference, rtol=tolerance, atol=0):
            return channels, ls, gammas, ir_mul
    raise ValueError("First convolution does not match the reassociated form in either layout")


__all__ = [
    "FactoredRadialMLP", "FirstInteractionReassociation", "ProjectedEmbedding", "attach_streaming_kernels",
    "cutoff_folded_forward", "identity_attrs", "identity_species_feats", "probe_first_tensor_product",
    "streaming_message",
]
