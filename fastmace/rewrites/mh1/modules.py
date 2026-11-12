"""Modules and the folded interaction forward of the MH-1 rewrites (SPEC §6.2).

Copied from the research engine (``langevin/gpu_langevin.py``) with its module switches
replaced by attributes that the rewrite records of ``fastmace.rewrites.mh1`` set on each
interaction or radial network:

- ``FactoredRadialMLP``: ``fused_backward`` (fused_folded_backward), ``folded_tiles``
  (folded_backward_tiles), ``fused_hidden`` (fused_radial_hidden);
- the interaction: ``density_before_linear`` (density_first), ``joint_radial_weight`` and
  ``joint_radial_widths`` (joint_radial_input), ``first_reassociation`` (reassociate_first),
  ``streaming_kernels`` and ``streaming_count`` (streaming_conv), ``chunked_backward_edges``
  (chunked_conv_backward: edges per chunk, 0 = off), ``residual_linear``
  (merged_residual, ``fastmace.rewrites.linear``), ``gated_rows`` (gated_rows,
  ``fastmace.rewrites.fusion``: the residual products, gate and linear_2 on per-segment rows);
- ``FirstInteractionReassociation.segmented`` (segmented_first_sum).

The research engine's screen options that were not carried over (docs/current/switch-map.md
section 7: ``CONTIGUOUS_FOLDED_WEIGHT``, ``JOINT_FUSED_BACKWARD``, ``STREAMING_MUL_IR``,
``STREAMING_RECEIVER_BACKWARD``) are fixed at their research-engine value, off.
"""

from __future__ import annotations

import torch

from fastmace.kernels.gpu_models import GPUModels
from fastmace.kernels.triton.edge_outer_sum import edge_outer_sum
from fastmace.kernels.triton.fused_radial import run_hidden_plan
from fastmace.kernels.triton.skinny_linear import folded_linear
from fastmace.rewrites.fusion import READS_GATED_ROWS
from fastmace.rewrites.linear import READS_RESIDUAL_LINEAR
from fastmace.rewrites.species import READS_SPECIES_RESIDUAL


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
        return self.folded_layer(self.augmented_after_first(hidden, cutoff))

    def folded_layer(self, augmented):
        """The folded last layer on its input ``[h c, c]`` (also the radial tables' output, radial_tables)."""
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


# Proof counter (compose-outside-graph, 12 November 2026): binds inside a CUDA-graph capture that kept a table's cached
# composed weights instead of composing them into the graph (``FirstInteractionReassociation.bind``).
_CAPTURE_ADOPTS = [0]


def CAPTURE_ADOPTS():  # noqa: N802 (the proof counter: --b-expect fastmace.rewrites.mh1.modules:CAPTURE_ADOPTS)
    """Binds inside a capture so far that kept the cached composed weights."""
    return _CAPTURE_ADOPTS[0]

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
        self.first_div = False  # idea 31 (kernels.cutlass.first_div), set by precision.route
        # kernels.triton.compose_skip: (the interaction's linear, its product index per path (empty: the identity),
        # its SpeciesSkip, the skip's product index per path), set by precision.route; the forward then returns the
        # skip's output before the division, and the interaction skips its linear and skip_tp
        self.compose_skip = None
        # kernels.triton.compose_skip.COMPOSE_FAST, set by precision.route: the interaction's radial last layer
        # (``bind`` composes the weights once per table), and the products through one joined species product
        self.compose_radial = None
        self.compose_joined = False
        self.count_plan = False
        self.joined_layout = False  # compose_skip.JOINED_LAYOUT: the sums d-major, the paths read in place  # compose_skip.COUNT_PLAN: the species plan by a counting sort
        # rewrites.radial.nested (NESTED_RADIAL), set by precision.route with compose_skip: [receiver species,
        # slots * 64, slots * 64] float64 maps composed into the weights of the calls whose sums come from the
        # rotated table (``summed_fn.nested``)
        self.register_buffer("nested_rotation", None, persistent=False)

    def bind(self, species_feats):
        """Point the forward at ``species_feats`` (a host-side pointer swap; captured graphs keep their address).
        With ``compose_skip``, also at that table's composed weights (``refresh_composed``)."""
        if self._buffers.get("species_feats") is not species_feats:
            self.register_buffer("species_feats", species_feats, persistent=False)
        if species_feats is None or self.compose_skip is None or self.compose_radial is None:
            return
        # Inside a capture, a table whose weights are cached keeps them although its version moved: the capture's
        # snapshot restore copied the same values back (bumping the version), and the force field's derived-state
        # registry restores the composed weights with the table (``composed_buffers``). Recomposing there put the four
        # float64 GEMMs into the captured step, replayed every step for constant inputs (compose-outside-graph,
        # 12 November 2026). A new table, new parameters or a nested change still compose.
        entry = self.refresh_composed(species_feats, adopt=torch.cuda.is_available()
                                      and torch.cuda.is_current_stream_capturing())
        for k, tensor in enumerate(entry[2]):
            if self._buffers.get(f"composed_{k}") is not tensor:
                self.register_buffer(f"composed_{k}", tensor, persistent=False)

    def refresh_composed(self, table, adopt=False):
        """The composed weights of ``table`` (``compose_skip``), computed on first use and again in place (same
        addresses) after the table or a composed parameter changed (their versions): ``[w_0, w_t_0, w_1, ...]`` per
        path, as the forward would compose them from the same parameters. Returns the cache entry ``[table ref,
        version, weights, parameter versions]``. ``adopt``: a cached entry whose parameters are unchanged takes the
        table's new version without composing (``bind`` inside a capture: the caller keeps them consistent)."""
        import weakref

        cache = self.__dict__.setdefault("_composed_cache", {})
        nested = self.nested_rotation is not None
        if self.__dict__.get("_composed_nested", False) != nested:  # nested_rotation set or cleared: compose again
            cache.clear()
            self.__dict__["_composed_nested"] = nested
        entry = cache.get(id(table))
        params = self._compose_versions()
        if entry is not None and entry[0]() is table and entry[3] == params:
            if entry[1] == table._version:
                return entry
            if adopt:
                _CAPTURE_ADOPTS[0] += 1
                entry[1] = table._version
                return entry
        with torch.no_grad():
            weights = self._compose_weights(table, self._radial_weight())
            if self.nested_rotation is not None:  # the rotated set after the plain one (``_composed(nested=True)``)
                weights += self._compose_weights(table, self._radial_weight(), self.nested_rotation)
        if entry is None or entry[0]() is not table:
            # Entries of tables that are gone go first: the tiled call binds a new table per tile, and their composed
            # weights (two [3, 192, 128] float32 per path) held 64 GB of an A100 by the end of a 149M-atom call.
            for key in [key for key, held in cache.items() if held[0]() is None]:
                del cache[key]
            entry = [weakref.ref(table), table._version, weights, params]
            cache[id(table)] = entry
        else:
            for dst, src in zip(entry[2], weights):
                dst.copy_(src)
            entry[1], entry[3] = table._version, params
        return entry

    def plan_columns(self):
        """The node_attrs columns of the species plan ``_composed`` builds (the skip's), or None when it builds none:
        the force field builds the plan outside the step and binds it (``species_gemm.bind_plan``)."""
        return self.compose_skip[2].columns if self.compose_skip is not None else None

    def composed_buffers(self, table):
        """The cached composed weights of ``table`` (``refresh_composed``), or ``()`` when none are cached: the
        force field registers them as derived state beside the table, so that a snapshot restore (a capture's
        warm-up, a rejected transaction) brings both back together."""
        entry = self.__dict__.get("_composed_cache", {}).get(id(table))
        return tuple(entry[2]) if entry is not None and entry[0]() is table else ()

    def _compose_versions(self):
        """The versions of every tensor the composition reads besides the table: the radial last layer, the
        interaction's linear and skip (``compose_skip``) and the nested rotation. A reload or an optimiser step
        writes them in place and moves their versions; nothing in an MD step does."""
        from fastmace.kernels.triton.species_gemm import PLAN_BUFFERS

        tensors = []
        modules = [self.compose_radial] + ([self.compose_skip[0], self.compose_skip[2]] if self.compose_skip else [])
        for module in modules:
            if isinstance(module, torch.nn.Module):
                # Not the species plan bound onto the skip (step-layout-cleanup): no input of the composition, and a
                # capture's snapshot restore moves its versions, which recomposed inside the captured step.
                tensors += list(module.parameters()) + [buffer for name, buffer in module.named_buffers()
                                                        if name.rpartition(".")[2] not in PLAN_BUFFERS]
            elif module is not None:  # a plain record of tensors (the CPU tests' radial layer)
                tensors += [value for value in vars(module).values() if isinstance(value, torch.Tensor)]
        if self.nested_rotation is not None:
            tensors.append(self.nested_rotation)
        versions = []
        for tensor in tensors:
            try:
                versions.append(tensor._version)
            except RuntimeError:  # an inference tensor has no version counter (and is never written in place)
                versions.append(None)
        return tuple(versions)

    def _radial_weight(self):
        """The radial last layer's matrix as the generic first interaction passes it (``_first_inputs``)."""
        from fastmace.rewrites.generic import modules as generic

        weight = generic.fc_last_weight(self.compose_radial)
        return torch.nn.functional.pad(weight, (0, 1)) if generic.FIRST_FULL_ROWS else weight

    def _compose_weights(self, x, weight, rotation=None):
        """Per path ``compose_skip.composed(right, ws, linear, rotation)`` as ``[w_0, w_t_0, w_1, w_t_1, ...]``."""
        from fastmace.kernels.triton import compose_skip

        linear, linear_products, skip, products = self.compose_skip
        c = self.channels
        out = []
        for path, gamma in enumerate(self.gammas):
            right = (gamma * x[:, None, :] * weight[path * c:(path + 1) * c].t()[None]).reshape(-1, c)
            mid = getattr(linear, f"weight_{linear_products[path]}") if linear_products else None
            out += compose_skip.composed(right, getattr(skip, f"ws{products[path]}"), mid, rotation)
        return out

    def forward(self, node_feats, node_attrs, edge_attrs, augmented, edge_index, weight, rows=None, summed_fn=None,
                divided_by=None):
        """``rows``: the message of the first ``rows`` nodes only, when every edge's receiver is one of them (the
        tiled call's tiles: the halo's rows are senders only, and the padding edges join a receiver slot); the
        senders' species still come from every row. None: every node's. ``summed_fn(key, slots) -> (sums [slots,
        D * K], K)``: the per-slot sums from elsewhere (``augmented`` unused; the tabled outer sum,
        ``generic.FUSED_FIRST_BACKWARD``)."""
        onehot = node_attrs[:, self.columns]
        species = onehot.argmax(dim=1)
        x = self.species_feats  # identity basis of the superset (SPEC §8.5), not the first atom of each species
        if x is None:
            raise RuntimeError("FirstInteractionReassociation has no species_feats table: bind one "
                               "(identity_species_feats) before calling it")
        nodes, slots = onehot.shape
        if rows is not None:
            nodes = int(rows)
        dim = edge_attrs.shape[1]
        key = edge_index[1] * slots + species[edge_index[0]]
        # [nodes, slots, (l, m), k]; fused backward (edge_outer_sum)
        joined = (summed_fn is not None and self.joined_layout and self.compose_skip is not None and rows is None
                  and self.compose_joined and self.tf32x3
                  and getattr(self, "tf32x3_backend", None) in ("cutlass", "triton"))  # Triton: strided launches
        if joined:  # JOINED_LAYOUT: [nodes, (l, m), slots, k]; _composed reads each path's rows in place
            summed, width = summed_fn(key, nodes * slots, inner=slots)
            return self._composed(summed.view(nodes, dim, slots, width), node_attrs, x, weight, nodes, slots, width,
                                  joined=True, nested=getattr(summed_fn, "nested", False))
        if summed_fn is not None:
            summed, width = summed_fn(key, nodes * slots)
            summed = summed.view(nodes, slots, dim, width)
        else:
            width = augmented.shape[1]
            summed = edge_outer_sum(augmented, edge_attrs, key, nodes * slots,
                                    segmented=self.segmented).view(nodes, slots, dim, width)
        blocks = []
        start = 0
        c = self.channels
        nested = getattr(summed_fn, "nested", False)  # the sums of the rotated table (rewrites.radial.nested)
        if self.compose_skip is not None and rows is None:
            return self._composed(summed, node_attrs, x, weight, nodes, slots, width, nested=nested)
        if nested:
            raise RuntimeError("nested radial sums need the composed products (compose_skip, every row)")
        if divided_by is not None and self.tf32x3 and self.ir_mul and rows is None:
            # idea 31 (kernels.cutlass.first_div): the blocks divided by (density + 1) in the epilogue, kept apart
            lefts, rights = [], []
            for path, (l, gamma) in enumerate(zip(self.ls, self.gammas)):  # noqa: E741
                size = 2 * l + 1
                lefts.append(summed[:, :, start:start + size].transpose(1, 2).reshape(nodes * size, slots * width))
                rights.append((gamma * x[:, None, :] * weight[path * c:(path + 1) * c].t()[None])
                              .reshape(slots * width, c))
                start += size
            density = divided_by()
            from fastmace.kernels.cutlass import first_div

            if first_div.applies(lefts, rights, density):
                outs = first_div.blocks_div(lefts, rights, density)
                return tuple(out.view(nodes, 2 * degree + 1, c) for out, degree in zip(outs, self.ls))
            start = 0
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

    def _composed(self, summed, node_attrs, x, weight, nodes, slots, width, joined=False, nested=False):
        """The message through the interaction's linear and the receivers' species skip
        (``kernels.triton.compose_skip``): per path ``species_mm(left, right L B_z)``, joined as the skip joins its
        output blocks (ir_mul, path order). ``joined`` (``JOINED_LAYOUT``): ``summed`` is [nodes, (l, m), slots,
        k], and the paths are read in place as row groups (``species_mm_rows``)."""
        from fastmace.kernels.triton import compose_skip
        from fastmace.kernels.triton.species_gemm import (bound_plan, joined_rows, species_mm, species_mm_joined,
                                                         species_mm_rows, species_plan)

        if torch.is_grad_enabled() and (weight.requires_grad or x.requires_grad):
            raise RuntimeError("compose_skip has no gradient for the weights: inference only (frozen parameters)")
        linear, linear_products, skip, products = self.compose_skip
        plan = bound_plan(self, nodes) if node_attrs.shape[0] == nodes else None  # the force field's (outside the step)
        if plan is not None:
            kind, order, counts = plan
        else:
            onehot = node_attrs[:nodes, skip.columns]
            kind = onehot.argmax(dim=1)
            order, counts = species_plan(kind, onehot.shape[1], counting=self.count_plan)
        if not self.tf32x3:
            precision = "ieee"
        else:
            precision = "tf32x3-cutlass" if getattr(self, "tf32x3_backend", None) == "cutlass" else "tf32x3"
        c = self.channels
        rotation = None
        if nested:  # rewrites.radial.nested: the rotated set, bound after the plain one
            rotation = self.nested_rotation
            if rotation is None or rotation.shape[1] != slots * width:
                raise RuntimeError("nested radial sums without a matching nested_rotation")
        # the table's composed weights from ``bind`` where they match these shapes, else composed here
        first = 2 * len(self.ls) if nested else 0
        bound = [self._buffers.get(f"composed_{k}") for k in range(first, first + 2 * len(self.ls))]
        if any(t is None for t in bound) or bound[0].shape[1] != slots * width:
            bound = None
        lefts, ws, w_ts, start = [], [], [], 0
        for path, (l, gamma) in enumerate(zip(self.ls, self.gammas)):  # noqa: E741
            size = 2 * l + 1
            if not joined:
                lefts.append(summed[:, :, start:start + size].transpose(1, 2).reshape(nodes, size, slots * width))
            if bound is not None:
                w, w_t = bound[2 * path], bound[2 * path + 1]
            else:
                right = (gamma * x[:, None, :] * weight[path * c:(path + 1) * c].t()[None]).reshape(slots * width, c)
                mid = getattr(linear, f"weight_{linear_products[path]}") if linear_products else None
                w, w_t = compose_skip.composed(right, getattr(skip, f"ws{products[path]}"), mid, rotation)
            ws.append(w)
            w_ts.append(w_t)
            start += size
        if joined:  # JOINED_LAYOUT: the rows of every path in place, one gradient tensor back
            sizes = [2 * l + 1 for l in self.ls]  # noqa: E741
            _, scatters = joined_rows(order, sizes)
            flat = summed.reshape(nodes * sum(sizes), slots * width)
            return species_mm_rows(flat, sizes, ws, w_ts, kind, order, counts, scatters, precision).reshape(nodes, -1)
        if precision == "tf32x3-cutlass" and self.compose_joined:
            # one product per path into the joined output, its rows built here (fused by a compiled caller)
            gathers, scatters = joined_rows(order, [left.shape[1] for left in lefts])
            lefts = [left.contiguous() for left in lefts]
            return species_mm_joined(lefts, ws, w_ts, kind, order, counts, gathers, scatters,
                                     precision).reshape(nodes, -1)
        blocks = [species_mm(left, w, w_t, kind, order, counts, precision).reshape(nodes, -1)
                  for left, w, w_t in zip(lefts, ws, w_ts)]
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


# The generic family's FUSED_FIRST_BACKWARD (tabled outer sum: the table read inside the reassociation's per-slot
# sums, one fused backward; kernels.triton.tabled_outer_sum) in MH-1's reassociated first interaction (polar-T01):
# the folded weight [W, b] reads the table's leading h + 1 columns, the bias column included. Switched by
# generic.modules.FUSED_FIRST_BACKWARD (the A/B constant) or, when it is None, on for these GPUs; none measured on
# MH-1 or POLAR-1 yet (decision 66).
FUSED_FIRST_MH1_GPUS = GPUModels(())


def fused_first_message(self, edge_feats, node_feats, node_attrs, edge_attrs, edge_index, rows=None):
    """``(message, edge density)`` of a tabled, reassociated, segmented float32 first interaction through the
    tabled outer sum, or None where it does not apply (``generic.modules.fused_first_applies``)."""
    from fastmace.rewrites.generic import modules as gm

    if not gm.fused_first_applies(self, edge_feats, FUSED_FIRST_MH1_GPUS):
        return None
    from fastmace.kernels.triton import tabled_outer_sum as tos

    table = self.radial_table
    weight = self.conv_tp_weights.folded_weight()
    width = weight.shape[-1]
    pair = table.pair(node_attrs, edge_index)
    density = []

    def summed_fn(key, slots, inner=1):
        summed, edge_density = tos.tabled_outer_sum(edge_feats, pair, table.coef, table.dcoef, table.starts,
                                                    table.scales, table.bins, table.r_lo, table.bin_scale,
                                                    table.r_max, width, edge_attrs.contiguous(), key, slots, inner)
        density.append(edge_density)
        return summed, width

    message = self.first_reassociation(node_feats, node_attrs, edge_attrs, None, edge_index, weight, rows=rows,
                                       summed_fn=summed_fn)
    return message, density[0]


def first_table_region(self, edge_feats, node_feats, node_attrs, edge_attrs, edge_index, rows=None):
    """``(message, edge density)`` of a reassociated interaction with a radial table: the table, then the
    reassociation. ``recompute_first_table`` runs it under non-reentrant checkpoint, so the backward keeps the
    lengths instead of the table's ``[E, h c + 1]`` output and evaluates the table again; the reassociation's
    backward needs nothing else of the region, so the compiled backward rebuilds only the table."""
    folded_input, edge_density = self.radial_table(edge_feats, node_attrs, edge_index)
    message = self.first_reassociation(node_feats, node_attrs, edge_attrs, folded_input, edge_index,
                                       self.conv_tp_weights.folded_weight(), rows)
    return message, edge_density


def recomputed_first_table(self, edge_feats, node_feats, node_attrs, edge_attrs, edge_index, rows=None):
    """``first_table_region`` under non-reentrant checkpoint (``recompute_first_table``)."""
    from torch.utils.checkpoint import checkpoint

    return checkpoint(first_table_region, self, edge_feats, node_feats, node_attrs, edge_attrs, edge_index, rows,
                      use_reentrant=False, preserve_rng_state=False)


def cutoff_folded_forward(self, node_attrs, node_feats, edge_attrs, edge_feats, edge_index,
                          cutoff=None, lammps_class=None, lammps_natoms=(0, 0), first_layer=False):
    """RealAgnosticResidualNonLinearInteractionBlock.forward (mace-torch 0.3.16), except that
    conv_tp_weights receives the cutoff and applies it itself (fold_cutoff).

    fastmace.rewrites.mh1 pins the sha256 of the copied forward's source (NONLINEAR_FORWARD_SHA256).
    """
    from mace.tools.scatter import scatter_sum
    num_nodes = node_feats.shape[0]
    n_real = lammps_natoms[0] if lammps_class is not None else None
    sc = self.skip_tp(node_feats if n_real is None else node_feats[:n_real])  # the real rows' (ghosts truncated)
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
    reassociation = getattr(self, "first_reassociation", None)  # reassociate_first
    # chunked_conv_backward: the streaming convolution takes the folded layer's input and applies its weight itself.
    chunked = reassociation is None and getattr(self, "chunked_backward_edges", 0) > 0
    tp_weights = augmented = None
    table = getattr(self, "radial_table", None)  # radial_tables: edge_feats are then the edge lengths [E, 1]
    w_lengths = wprime_lengths = message = None
    wprime = getattr(self, "wprime_backward", None)  # wprime_backward (OPT9-T03)
    if table is not None and edge_feats.shape[-1] == 1 and chunked and wprime is not None:
        wprime_lengths = edge_feats  # the message and the edge density from one op
    elif table is not None and edge_feats.shape[-1] == 1 and reassociation is not None and getattr(
            self, "recompute_first_table", False):  # recompute_first_table: the table's output is not kept
        message, edge_density = recomputed_first_table(self, edge_feats, node_feats, node_attrs, edge_attrs, edge_index,
                                                       n_real)  # the tiles' receivers only (n_real rows)
    elif (table is not None and edge_feats.shape[-1] == 1 and reassociation is not None
          and (fused := fused_first_message(self, edge_feats, node_feats, node_attrs, edge_attrs, edge_index,
                                            n_real)) is not None):
        message, edge_density = fused  # FUSED_FIRST_BACKWARD on MH-1's class
    elif table is not None and edge_feats.shape[-1] == 1:
        folded_input, edge_density = table(edge_feats, node_attrs, edge_index)  # the cutoff is in both
        if getattr(self, "w_table", None) is not None and chunked and reassociation is None:  # w_tables
            w_lengths = edge_feats
        elif chunked or reassociation is not None:
            augmented = folded_input
        else:
            tp_weights = self.conv_tp_weights.folded_layer(folded_input)
    else:
        source_embedding = self.source_embedding(node_attrs)
        target_embedding = self.target_embedding(node_attrs)
        joint = getattr(self, "joint_radial_weight", None)  # joint_radial_input
        if joint is not None:
            # Both first layers at once; the gathered species terms join inside the add.
            first = torch.nn.functional.linear(edge_feats, joint)
            hidden = (first + source_embedding[edge_index[0]]
                      + target_embedding[edge_index[1]])
            conv_hidden, density_hidden = hidden.split(self.joint_radial_widths, dim=1)
            if chunked or reassociation is not None:
                augmented = self.conv_tp_weights.augmented_after_first(conv_hidden, cutoff)
            else:
                tp_weights = self.conv_tp_weights.after_first(conv_hidden, cutoff)
            edge_density = torch.tanh(self.density_fn.after_first(density_hidden) ** 2)
        else:
            edge_feats = torch.cat([edge_feats, source_embedding[edge_index[0]],
                                    target_embedding[edge_index[1]]], dim=-1)
            if chunked or reassociation is not None:
                augmented = self.conv_tp_weights.augmented(edge_feats, cutoff)
            else:
                tp_weights = self.conv_tp_weights(edge_feats, cutoff)  # changed: cutoff applied inside
            edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)
        if cutoff is not None:
            edge_density = edge_density * cutoff
    if wprime_lengths is not None:
        message, edge_density = streaming_message(self, node_feats, edge_attrs, None, edge_index, num_nodes,
                                                  node_attrs=node_attrs, wprime_lengths=wprime_lengths)
    density = scatter_sum(src=edge_density, index=edge_index[1], dim=0, dim_size=num_nodes)
    if wprime_lengths is not None or message is not None:
        pass
    elif reassociation is not None:
        message = reassociation(node_feats, node_attrs, edge_attrs, augmented, edge_index,
                                self.conv_tp_weights.folded_weight(), rows=n_real)  # the tiles' receivers only
    elif getattr(self, "streaming_kernels", None) is not None:  # streaming_conv
        message = streaming_message(self, node_feats, edge_attrs, tp_weights, edge_index, num_nodes, augmented,
                                    w_lengths, node_attrs, rows=n_real)
    elif hasattr(self, "conv_fusion"):
        message = self.conv_tp(node_feats, edge_attrs, tp_weights, edge_index)
    else:
        mji = self.conv_tp(node_feats[edge_index[0]], edge_attrs, tp_weights)
        message = scatter_sum(src=mji, index=edge_index[1], dim=0, dim_size=num_nodes)
    if message.shape[0] != n_real:  # (the streaming convolution may have built the real rows only)
        message = self.truncate_ghosts(message, n_real)
    density = self.truncate_ghosts(density, n_real)
    sc = self.truncate_ghosts(sc, n_real)
    # species_residual: the residual products are per-species rows, read from the one-hot node_attrs.
    residual_input = node_attrs if getattr(self, "species_residual", False) else node_feats
    # Ghost rows (fastmace.model.tiled's halo) are cut as stock MACE cuts node_feats_res.
    residual_input = self.truncate_ghosts(residual_input, n_real)
    gated_rows = getattr(self, "gated_rows", None)  # gated_rows: residual products, gate and linear_2 on rows
    if gated_rows is not None:
        divisor = density * self.beta + self.alpha
        if getattr(gated_rows, "region_divides", False):  # recompute_node_region: divided inside the region's chunks
            message = gated_rows(message, residual_input, divisor)
        else:
            message = gated_rows(message / divisor, residual_input)
        return (self.reshape(message), sc)
    if residual_linear is not None:  # with density_first (merge_residual_linears)
        message = residual_linear(message / (density * self.beta + self.alpha), residual_input)
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
setattr(cutoff_folded_forward, READS_SPECIES_RESIDUAL, True)


def streaming_message(self, node_feats, edge_attrs, tp_weights, edge_index, num_nodes, augmented=None,
                      w_lengths=None, node_attrs=None, wprime_lengths=None, table_lengths=None, rows=None,
                      block_rows=None):
    """The interaction's OEq convolution through the receiver-streaming kernels (attached as streaming_kernels).

    With ``augmented`` (chunked_conv_backward) the per-edge weights are ``augmented @ [W, b].T`` of the folded
    radial layer, built inside the convolution, whose backward runs in L2-sized edge chunks; with
    ``fused_folded_forward`` attached (opt-8 OPT8-T02) one kernel builds them and the message. With
    ``wprime_lengths`` (wprime_backward, opt-9 OPT9-T03) the radial table runs inside the op too, which returns
    ``(message, edge density)``; its backward forms the lengths' gradient without the weight gradient. With
    ``table_lengths`` (generic_recompute_first_table on generic_recompute_conv_weights' path) the op evaluates the
    radial table itself and keeps the lengths instead of its output; it also returns ``(message, edge density)``.

    ``rows`` (a tile's receiver count, every receiver below it): the message of those rows only, without the
    halo rows' zeros, on the paths whose kernels allow it (``StreamingKernels.rows_fit``); else all ``num_nodes``.
    ``block_rows`` (wprime_lengths with BLOCK_MESSAGE, ``rewrites.generic.modules.TILE_BLOCK_ROWS``): the block
    groups of that many leading rows (views), their gradients as short.
    """
    from fastmace.kernels.oeq_streaming import folded_streaming_convolution, streaming_convolution
    from fastmace.rewrites.generic.modules import edge_vectors
    sender, receiver = edge_index[0], edge_index[1]
    plain = wprime_lengths is None and w_lengths is None and table_lengths is None and (
        augmented is None or (getattr(self, "generic_recompute", None) is None
                              and getattr(self, "fused_folded_forward", None) is None))
    if rows is not None and not (plain and self.streaming_kernels.rows_fit(rows)):
        rows = None
    rows = num_nodes if rows is None else rows
    # The compacted count (the engine registers the neighbour list's GPU counter) ends the
    # receiver-sorted edges within the cutoff; the padding after it goes past every row. (A float32
    # cutoff test is not enough: pairs at the cutoff can disagree with the float64 compaction.)
    position = torch.arange(receiver.shape[0], device=receiver.device)
    key = torch.where(position < self.streaming_count, receiver, torch.full_like(receiver, num_nodes))
    row_ptr = torch.searchsorted(key, torch.arange(rows + 1, device=key.device)).to(torch.int32)
    # The kernels read and write the model's own node layout (ir_mul with cuEquivariance), so
    # OEq's layout transposes around the convolution are not needed.
    conv = self.conv_tp
    transpose = self.streaming_kernels.layout == "mul_ir" and conv.layout_transpose_in is not None
    feats = conv.layout_transpose_in(node_feats) if transpose else node_feats
    density = None
    if wprime_lengths is not None:
        radial = getattr(self, "generic_folded", None) or self.conv_tp_weights  # generic_wprime_backward
        message, density = self.wprime_backward.convolution(
            wprime_lengths, self.radial_table.pair(node_attrs, edge_index), self.radial_table, radial.folded_weight(),
            feats, edge_attrs, sender.to(torch.int32), receiver.to(torch.int32), row_ptr, self.streaming_kernels,
            getattr(self, "fused_folded_forward", None), getattr(self, "recompute_weight_rows", 0),
            linear=getattr(self, "linear", None),  # BLOCK_MESSAGE: a tuple of linear_1's input groups
            block_rows=block_rows, vectors=edge_vectors(self, edge_attrs))  # EDGE_FORCE
    elif w_lengths is not None:  # w_tables: the per-edge weights from the edge lengths, in float64
        message = self.w_table.convolution(w_lengths, self.w_table.pair(node_attrs, edge_index), feats, edge_attrs,
                                           sender.to(torch.int32), receiver.to(torch.int32), row_ptr,
                                           self.streaming_kernels)
    elif table_lengths is not None:
        from fastmace.kernels.oeq_streaming import tabled_recompute_convolution
        radial = self.generic_folded
        message, density = tabled_recompute_convolution(
            table_lengths, self.radial_table.pair(node_attrs, edge_index), self.radial_table, radial.folded_weight(),
            feats, edge_attrs, sender.to(torch.int32), receiver.to(torch.int32), row_ptr, self.streaming_kernels,
            self.chunked_backward_edges, self.generic_recompute, radial.folded_tiles, radial.folded_parts)
    elif augmented is not None and getattr(self, "generic_recompute", None) is not None:
        from fastmace.kernels.oeq_streaming import folded_recompute_convolution
        radial = self.generic_folded  # generic_recompute_conv_weights: no per-edge weights kept
        message = folded_recompute_convolution(augmented, radial.folded_weight(), feats, edge_attrs,
                                               sender.to(torch.int32), receiver.to(torch.int32), row_ptr,
                                               self.streaming_kernels, self.chunked_backward_edges,
                                               self.generic_recompute, radial.folded_tiles, radial.folded_parts)
    elif augmented is not None:
        radial = getattr(self, "generic_folded", None) or self.conv_tp_weights  # generic_chunked_conv_backward
        message = folded_streaming_convolution(augmented, radial.folded_weight(), feats, edge_attrs,
                                               sender.to(torch.int32), receiver.to(torch.int32), row_ptr,
                                               self.streaming_kernels, self.chunked_backward_edges,
                                               radial.folded_tiles, radial.folded_parts,
                                               getattr(self, "fused_folded_forward", None))
    else:
        message = streaming_convolution(feats, edge_attrs, tp_weights, sender.to(torch.int32),
                                        receiver.to(torch.int32), row_ptr, self.streaming_kernels)
    if transpose:
        message = conv.layout_transpose_out(message)
    return message if density is None else (message, density)


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
    "cutoff_folded_forward", "first_table_region", "fused_first_message", "identity_attrs", "recomputed_first_table",
    "identity_species_feats",
    "probe_first_tensor_product", "streaming_message",
]
