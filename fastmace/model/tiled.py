"""Receiver-tiled, three-sweep force call: a capacity prototype (docs/design/tiled-backend.md).

``TiledForceField`` wraps an eager ``MACEForceField`` of a two-interaction ``ScaleShiftMACE`` whose first
interaction reads species-only sender features (MH-1 with ``reassociate_first``, or stock: the first
interaction's input is the species embedding). One force call runs the rewritten modules on tiles: a sorted set
of receivers, the edges into them and the halo of their senders, laid out as MACE's LAMMPS ghosts so that the
interactions cut their outputs to the receivers themselves. Three sweeps:

1. interaction 0 forward without autograd. Of the inter-layer features h1 only interaction 1's two reads of
   them are kept globally: x = linear_up(h1) and sc = skip_tp(h1), both linear (MH-1: 512 columns each, against
   about 2,100 for h1);
2. interaction 1 forward and backward on x and sc (its linear_up and skip_tp stand in for the stored values): the
   adjoints dE/dx, scattered over the tile and its halo, and dE/dsc of the receivers, and the edge gradients of
   the second interaction;
3. interaction 0 again, with autograd, backward of its own energy terms plus <dE/dx, x> + <dE/dsc, sc> of the
   recomputed h1: the remaining edge gradients.

Only x, sc, their adjoints and per-atom outputs grow with the atom count (MH-1: 8 kB per atom); every other
intermediate is bounded by the tile caps. Forces come from the edge-vector gradients (MACE's edge forces), the
stress from the edge virial on a symmetric strain of the edge vectors (stress = dE/d(strain) / V, MACE's sign).
The result equals the
whole-system call up to float32 rounding (the tiles' GEMMs see fewer rows; the force scatter is atomic).

Prototype scope: eager, no CUDA graphs, fixed cell per call, no PolarMACE, D3, joint embedding or embedding
readout. Tiles are planned on the host from a spatial binning of the fractional coordinates.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import math
from typing import Any

from fastmace.model.forces import ForceOutput


@dataclasses.dataclass(frozen=True)
class TileConfig:
    """Tile caps of the prototype: at most ``receivers`` receiver atoms and ``edges`` edges per tile.

    ``edges=None`` caps the edges at ``receivers`` times twice the structure's mean degree, so a tile is split
    where its receivers are unusually crowded. ``mode``: ``"eager"`` (each tile at its own shape), ``"padded"``
    (every tile in one padded slot, eagerly) or ``"graphs"`` (the padded slot, one CUDA graph per sweep).
    ``headroom``: a padded slot's halo and edge counts above the largest tile's, as a fraction.
    """

    receivers: int = 8192
    edges: int | None = None
    mode: str = "eager"
    headroom: float = 0.1

    def __post_init__(self):
        if self.mode not in ("eager", "padded", "graphs"):
            raise ValueError(f"TileConfig.mode must be 'eager', 'padded' or 'graphs', got {self.mode!r}")
        if not 0 <= float(self.headroom) <= 10:
            raise ValueError(f"TileConfig.headroom must be in [0, 10], got {self.headroom!r}")
        if int(self.receivers) < 1:
            raise ValueError(f"TileConfig.receivers must be positive, got {self.receivers!r}")
        if self.edges is not None and int(self.edges) < 1:
            raise ValueError(f"TileConfig.edges must be positive or None, got {self.edges!r}")


class _Ghosts:
    """``lammps_class`` of the tile calls: not None, so the interactions cut their outputs to the receivers
    (``truncate_ghosts``); with ``first_layer=True`` their ``handle_lammps`` exchanges nothing."""


_GHOSTS = _Ghosts()


def _given(value):
    """Stand-in for a module of interaction 1 in sweep 2: returns ``value``, whatever its input."""
    import torch

    class Given(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return value

    return Given()


def plan_tiles(positions, cell, row_ptr, receivers, edges):
    """Receiver tiles: a list of sorted int64 device index tensors covering every atom once.

    Atoms are binned on a k1 x k2 x k3 grid of fractional coordinates, with k_i from the cell's plane spacings so
    that a block is about cubic and holds about ``receivers`` atoms; a block above either cap is split into
    consecutive pieces (in index order) below both. A rule on the cell geometry and the caps only.
    """
    import torch

    n = positions.shape[0]
    cell64 = torch.as_tensor(cell, dtype=torch.float64, device=positions.device)
    frac = torch.remainder(positions.to(torch.float64) @ torch.linalg.inv(cell64), 1.0)
    volume = abs(float(torch.linalg.det(cell64)))
    recip = torch.linalg.inv(cell64).T
    spacings = [1.0 / float(torch.linalg.norm(recip[i])) for i in range(3)]  # plane spacings, Angstrom
    side = (volume * min(receivers, n) / n) ** (1 / 3)
    k = [max(1, round(s / side)) for s in spacings]
    bins = torch.clamp((frac * torch.tensor(k, dtype=torch.float64, device=positions.device)).long(), min=0)
    bins = torch.minimum(bins, torch.tensor([x - 1 for x in k], device=positions.device))
    block = (bins[:, 0] * k[1] + bins[:, 1]) * k[2] + bins[:, 2]
    order = torch.sort(block, stable=True).indices  # within a block: increasing atom index
    counts = torch.bincount(block, minlength=k[0] * k[1] * k[2]).cpu().tolist()
    degree = (row_ptr[1:] - row_ptr[:-1])[order].cpu()
    tiles = []
    start = 0
    for count in counts:
        stop = start + count
        piece = start
        edge_sum = 0
        for i in range(start, stop):
            d = int(degree[i])
            if i > piece and (i - piece >= receivers or edge_sum + d > edges):
                tiles.append(order[piece:i])
                piece, edge_sum = i, 0
            edge_sum += d
        if stop > piece:
            tiles.append(order[piece:stop])
        start = stop
    return tiles, tuple(k)


class TiledForceField:
    """The tiled three-sweep force call around an eager ``MACEForceField`` (``field``, ``compile_mode="none"``).

    ``field`` keeps its neighbour list, edges, species tables and rewritten model; this class only schedules the
    model's modules. ``evaluate(atoms, stress)`` is the one-off evaluation of SPEC §7.1; ``__call__(positions)``
    the force call on the field's current edges (after ``rebuild``, ``compact`` and ``use_edges``).

    ``TileConfig.mode``: ``"eager"`` runs each tile at its own shape; ``"padded"`` and ``"graphs"`` run every tile in
    one padded slot of fixed receiver, halo and edge counts (``_Slot``), eagerly or as one captured CUDA graph per
    sweep that reads the slot's static buffers. The slot grows, and the graphs are captured again, only when a tile
    does not fit it.
    """

    def __init__(self, field, tiles: TileConfig | None = None):
        import torch

        self.field = field
        self.config = tiles or TileConfig()
        model = getattr(field.model, "_orig_mod", field.model)
        if field.compile_mode != "none":
            raise ValueError("TiledForceField needs an eager force field (compile_mode='none')")
        if type(model).__name__ != "ScaleShiftMACE" or len(model.interactions) != 2:
            raise ValueError("TiledForceField supports two-interaction ScaleShiftMACE models only")
        if hasattr(model, "joint_embedding") or getattr(field, "is_polar", False):
            raise ValueError("TiledForceField does not support joint embeddings or PolarMACE")
        if self.config.mode == "graphs" and field.device.type != "cuda":
            raise ValueError("TileConfig.mode='graphs' needs a CUDA device")
        self.model = model
        self.device = field.device
        self.dtype = field.dtype
        self._big = torch.tensor(torch.iinfo(torch.int64).max, device=self.device)
        skip = model.interactions[1].skip_tp
        required = [p for p in inspect.signature(skip.forward).parameters.values()
                    if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        self._skip_takes_attrs = len(required) >= 2
        # Padding edges: a vector beyond the cutoff, so their radial envelope, and every term it multiplies, is 0.
        r_max = float(getattr(model, "r_max", 10.0))
        self._far = torch.tensor([2.0 * r_max + 1.0, 0.0, 0.0], dtype=self.dtype, device=self.device)
        self._slot = None
        self._g_slot = None
        self._graphs = {}
        self._graph_key = None
        self._pool = None
        self._stream = None
        self.last = {}

    @property
    def compute_stress(self):
        return self.field.compute_stress

    # ------------------------------------------------------------------ one tile

    def _tile_graph(self, receivers, row_ptr, edge_index, local_of, halo_start=None):
        """Local graph of one tile: (nodes [R + H], edge ids [Ec], local edge_index [2, Ec], R, halo size).

        The halo's local indices start at ``halo_start`` (default: after the receivers; a padded slot puts them
        after its receiver slots)."""
        import torch

        r = receivers.shape[0]
        start = r if halo_start is None else int(halo_start)
        starts = row_ptr[receivers]
        counts = row_ptr[receivers + 1] - starts
        total = int(counts.sum())
        offsets = torch.cumsum(counts, 0) - counts
        edge_ids = (torch.repeat_interleave(starts - offsets, counts, output_size=total)
                    + torch.arange(total, device=self.device))
        senders = edge_index[0, edge_ids]
        local_of[receivers] = torch.arange(r, device=self.device)
        outside = senders[local_of[senders] < 0]
        halo = torch.unique(outside)
        local_of[halo] = torch.arange(start, start + halo.shape[0], device=self.device)
        local = torch.stack([local_of[senders], torch.repeat_interleave(
            torch.arange(r, device=self.device), counts, output_size=total)])
        local_of[receivers] = -1
        local_of[halo] = -1
        return torch.cat([receivers, halo]), edge_ids, local, r, halo.shape[0]

    def _edges(self, vectors, node_attrs, local, strain):
        """Spherical harmonics, radial features and cutoff of the tile's edges from their vectors."""
        import torch

        model = self.model
        if strain is not None:
            vectors = vectors + vectors @ (0.5 * (strain + strain.transpose(0, 1)))
        lengths = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
        edge_attrs = model.spherical_harmonics(vectors)
        edge_feats, cutoff = model.radial_embedding(lengths, node_attrs, local, model.atomic_numbers)
        return lengths, edge_attrs, edge_feats, cutoff

    def _layer(self, index, node_feats, node_attrs, edge_attrs, edge_feats, cutoff, local, r):
        """Interaction ``index`` and its product on a tile: the receivers' output features [R, ...]."""
        model = self.model
        feats, sc = model.interactions[index](
            node_attrs=node_attrs, node_feats=node_feats, edge_attrs=edge_attrs, edge_feats=edge_feats,
            edge_index=local, cutoff=cutoff, first_layer=True, lammps_class=_GHOSTS,
            lammps_natoms=(r, node_feats.shape[0] - r))
        return model.products[index](node_feats=feats, sc=sc, node_attrs=node_attrs[:r])

    def _consumers(self, h1, attrs):
        """Interaction 1's two reads of h1: ``(linear_up(h1), skip_tp(h1))`` (``skip_tp`` with the species when
        it takes them)."""
        interaction = self.model.interactions[1]
        sc = interaction.skip_tp(h1, attrs) if self._skip_takes_attrs else interaction.skip_tp(h1)
        return interaction.linear_up(h1), sc

    @staticmethod
    def _masked(value, mask):
        return value if mask is None else value * mask.unsqueeze(-1) if value.dim() > 1 else value * mask

    def _first(self, tile, state, adjoint=False, strain=None):
        """Interaction 0 on a tile. Sweep 1 (``adjoint`` False, no autograd): ``(es, x, sc)``, the receivers' node
        energies before scale and shift and interaction 1's inputs. Sweep 3 (``adjoint`` True): ``(es, edge-vector
        gradient, strain gradient or None)`` of interaction 0's energy plus <dE/dx, x> + <dE/dsc, sc>. Padding
        receivers (``recv_mask`` 0) give exact zeros."""
        import torch

        model = self.model
        nodes, local, r, vectors, recv_mask, _ = tile
        attrs = self.node_attrs[nodes]
        receivers = nodes[:r]
        with torch.enable_grad() if adjoint else torch.no_grad():
            leaf = vectors.detach().requires_grad_(adjoint)
            eps = (torch.zeros((3, 3), dtype=self.dtype, device=self.device, requires_grad=True)
                   if adjoint and strain is not None else None)
            lengths, edge_attrs, edge_feats, cutoff = self._edges(leaf, attrs, local, eps)
            node_feats = model.node_embedding(attrs)
            h1 = self._layer(0, node_feats, attrs, edge_attrs, edge_feats, cutoff, local, r)
            heads = self.heads[:r]
            es = model.readouts[0](h1, heads)[self.arange[:r], heads]
            if hasattr(model, "pair_repulsion"):
                es = model.pair_repulsion_fn(lengths, attrs, local, model.atomic_numbers)[:r] + es
            x, sc = self._consumers(h1, attrs[:r])
            if not adjoint:
                return self._masked(es, recv_mask), self._masked(x, recv_mask), self._masked(sc, recv_mask)
            objective = (self._masked(model.scale_shift(es, heads), recv_mask).sum()
                         + (self._masked(state["gx"][receivers], recv_mask) * x).sum()
                         + (self._masked(state["gsc"][receivers], recv_mask) * sc).sum())
            wanted = [leaf] + ([eps] if eps is not None else [])
            grads = torch.autograd.grad(objective, wanted)
        return self._masked(es.detach(), recv_mask), grads[0], grads[1] if eps is not None else None

    def _second(self, tile, state, strain):
        """Interaction 1 forward and backward on the stored ``linear_up(h1)`` and ``skip_tp(h1)`` (sweep 2):
        ``(es, edge-vector gradient, dE/dx [R + H], dE/dsc [R], strain gradient or None)``."""
        import torch

        model = self.model
        interaction = model.interactions[1]
        nodes, local, r, vectors, recv_mask, node_mask = tile
        attrs = self.node_attrs[nodes]
        with torch.enable_grad():
            leaf = vectors.detach().requires_grad_(True)
            eps = (torch.zeros((3, 3), dtype=self.dtype, device=self.device, requires_grad=True)
                   if strain is not None else None)
            x = state["x"][nodes].detach().requires_grad_(True)
            sc = state["sc"][nodes[:r]].detach().requires_grad_(True)
            _, edge_attrs, edge_feats, cutoff = self._edges(leaf, attrs, local, eps)
            modules = interaction._modules
            saved = modules["linear_up"], modules["skip_tp"]
            modules["linear_up"], modules["skip_tp"] = _given(x), _given(sc)
            try:  # their real counterparts ran in sweep 1, on the h1 that is not kept
                out = self._layer(1, x, attrs, edge_attrs, edge_feats, cutoff, local, r)
            finally:
                modules["linear_up"], modules["skip_tp"] = saved
            heads = self.heads[:r]
            es = model.readouts[1](out, heads)[self.arange[:r], heads]
            objective = self._masked(model.scale_shift(es, heads), recv_mask).sum()
            wanted = [leaf, x, sc] + ([eps] if eps is not None else [])
            grads = torch.autograd.grad(objective, wanted)
        return (self._masked(es.detach(), recv_mask), grads[0], self._masked(grads[1], node_mask),
                self._masked(grads[2], recv_mask), grads[3] if eps is not None else None)

    # ------------------------------------------------------------------ padded slot

    def _new_slot(self, sizes):
        """Static buffers of one padded tile: ``sizes = (R, H, E)`` receiver slots (one more than any tile, so a
        padding receiver always exists), halo rows and edges."""
        import torch

        receivers, halo, edges = sizes
        i64 = torch.int64
        new = functools.partial(torch.zeros, device=self.device)
        return dict(sizes=sizes, nodes=new(receivers + halo, dtype=i64), node_mask=new(receivers + halo,
                    dtype=self.dtype), recv_mask=new(receivers, dtype=self.dtype), edge_ids=new(edges, dtype=i64),
                    local=new((2, edges), dtype=i64), edge_mask=new(edges, dtype=self.dtype), count=new((), dtype=i64),
                    out={})

    def _fill(self, slot, piece):
        """Copy one tile (``_tile_graph``'s output at the slot's halo start) into the slot's static buffers.

        Padding: receivers and halo rows read atom 0 and are masked; padding edges join the last receiver slot
        (always padding) to itself, after the tile's edges (so the receivers stay sorted), beyond the cutoff and
        beyond the tile's edge count (the streaming kernels' ``streaming_count``)."""
        nodes, edge_ids, local, r, h = piece
        receivers, _, edges = slot["sizes"]
        e = edge_ids.shape[0]
        slot["nodes"].zero_()
        slot["nodes"][:r] = nodes[:r]
        slot["nodes"][receivers:receivers + h] = nodes[r:]
        slot["node_mask"].zero_()
        slot["node_mask"][:r] = 1
        slot["node_mask"][receivers:receivers + h] = 1
        slot["recv_mask"].zero_()
        slot["recv_mask"][:r] = 1
        slot["edge_ids"].zero_()
        slot["edge_ids"][:e] = edge_ids
        slot["local"].fill_(receivers - 1)
        slot["local"][:, :e] = local
        slot["edge_mask"].zero_()
        slot["edge_mask"][:e] = 1
        slot["count"].fill_(e)

    def _body(self, sweep, slot, g):
        """One sweep on the slot's tile, reading only static buffers; writes ``slot["out"]`` (graph-capturable)."""
        import torch

        receivers = slot["sizes"][0]
        nodes, eids = slot["nodes"], slot["edge_ids"]
        ei = g["edge_index"][:, eids]
        vectors = g["pos"][ei[1]] - g["pos"][ei[0]] + g["shifts"][eids]
        vectors = torch.where(slot["edge_mask"].unsqueeze(1) > 0, vectors, self._far)
        tile = (nodes, slot["local"], receivers, vectors, slot["recv_mask"], slot["node_mask"])
        strain = True if g["stress"] else None
        if sweep == 1:
            values = self._first(tile, g["state"])
            names = ("es", "x", "sc")
        elif sweep == 2:
            es, g_vec, g_x, g_sc, g_eps = self._second(tile, g["state"], strain)
            values = (es, self._masked(g_vec, slot["edge_mask"]), g_x, g_sc) + ((g_eps,) if strain else ())
            names = ("es", "g_vec", "g_x", "g_sc", "g_eps")[:len(values)]
        else:
            _, g_vec, g_eps = self._first(tile, g["state"], adjoint=True, strain=strain)
            values = (self._masked(g_vec, slot["edge_mask"]),) + ((g_eps,) if strain else ())
            names = ("g_vec", "g_eps")[:len(values)]
        out = slot["out"].setdefault(sweep, {})
        for name, value in zip(names, values):
            if name not in out:
                out[name] = torch.empty_like(value)
            out[name].copy_(value)

    def _scatter(self, sweep, slot, g):
        """Add the slot's outputs into the global buffers (outside the graphs: warm-ups and captures have no side
        effects). Padding rows read atom 0 and add exact zeros to it."""
        out = slot["out"][sweep]
        receivers = slot["sizes"][0]
        recv = slot["nodes"][:receivers]
        state = g["state"]
        if sweep == 1:
            state = self._state(g, out["x"], out["sc"])
            g["node_es"].index_add_(0, recv, out["es"])
            state["x"].index_add_(0, recv, out["x"])
            state["sc"].index_add_(0, recv, out["sc"])
            return
        if sweep == 2:
            g["node_es"].index_add_(0, recv, out["es"])
            state["gx"].index_add_(0, slot["nodes"], out["g_x"])
            state["gsc"].index_add_(0, recv, out["g_sc"])
        ei = g["edge_index"][:, slot["edge_ids"]]
        g["forces"].index_add_(0, ei[1], out["g_vec"], alpha=-1)
        g["forces"].index_add_(0, ei[0], out["g_vec"])
        if "g_eps" in out:
            g["strain_grad"].add_(out["g_eps"].double())

    def _run_slot(self, sweep, slot, g):
        """The sweep's body on the filled slot: eagerly, or by replaying its graph (captured on first use)."""
        import torch

        if self.config.mode != "graphs":
            self._body(sweep, slot, g)
            return
        graph = self._graphs.get(sweep)
        if graph is None:
            if self._stream is None:
                self._stream = torch.cuda.Stream()
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                for _ in range(2):  # library workspaces and the allocator, outside the capture
                    self._body(sweep, slot, g)
            torch.cuda.current_stream().wait_stream(self._stream)
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool, stream=self._stream):
                self._body(sweep, slot, g)
            self._graphs[sweep] = graph
        graph.replay()

    # ------------------------------------------------------------------ the call

    def _bind(self, count=None):
        field = self.field
        field._bind_streaming_count()  # species tables; then every tile edge is a real edge
        saved = []
        for module in field._streaming:
            saved.append((module, module._buffers.get("streaming_count")))
            module.register_buffer("streaming_count", self._big if count is None else count, persistent=False)
        return saved

    def __call__(self, positions):
        """Energy [1] (float64), forces [N, 3] and stress [3, 3] or None at float64 ``positions``."""
        import torch

        field = self.field
        data = field.data
        stress = field.compute_stress
        padded = self.config.mode != "eager"
        with torch.no_grad():
            field.positions.copy_(positions)
        pos = field.positions.detach()
        edge_index = data["edge_index"]
        n = pos.shape[0]
        self.node_attrs = data["node_attrs"]
        heads = (data["head"][data["batch"]] if "head" in data else torch.zeros_like(data["batch"])).long()
        receiver = edge_index[1]
        row_ptr = torch.searchsorted(receiver, torch.arange(n + 1, device=self.device))
        num_edges = receiver.shape[0]
        caps = self.config
        edge_cap = caps.edges if caps.edges is not None else max(
            1, int(math.ceil(2 * caps.receivers * num_edges / max(n, 1))))
        cell = data["cell"].reshape(3, 3) if "cell" in data else torch.as_tensor(field.neighbours.cell)
        tiles, grid = plan_tiles(pos, cell, row_ptr, caps.receivers, edge_cap)
        local_of = torch.full((n,), -1, dtype=torch.int64, device=self.device)
        stats = dict(tiles=len(tiles), grid=grid, edge_cap=edge_cap, max_receivers=0, max_halo=0, max_edges=0,
                     mode=caps.mode)
        if padded:
            sizes = self._slot_sizes(tiles, row_ptr, edge_index, local_of, stats)
            if self._slot is None or any(a > b for a, b in zip(sizes, self._slot["sizes"])):
                self._slot = self._new_slot(sizes)
            stats["slot"] = self._slot["sizes"]
        saved = self._bind(self._slot["count"] if padded else None)
        try:
            # The neighbour list's full buffers (fixed addresses between capacity changes) for the padded path.
            base_index = field.neighbours.edge_index if padded else edge_index
            base_shifts = field.neighbours.shifts if padded else data["shifts"]
            g = self._globals(n, stress, pos, base_index, base_shifts, heads)
            self.heads, self.arange = g["heads"], g["arange"]
            if padded:
                key = (n, self._slot["sizes"], base_index.data_ptr(), base_shifts.data_ptr(), pos.data_ptr(),
                       self.node_attrs.data_ptr(), stress)
                if key != self._graph_key:
                    self._graphs = {}
                    self._graph_key = key
            forces, node_es, strain_grad, state = g["forces"], g["node_es"], g["strain_grad"], g["state"]
            if padded and self._g_slot is not self._slot:  # a new slot: its graphs are captured again
                self._graphs = {}
                self._g_slot = self._slot
            for sweep in (1, 2, 3):
                for receivers in tiles:
                    piece = self._tile_graph(receivers, row_ptr, edge_index, local_of,
                                             self._slot["sizes"][0] if padded else None)
                    self._note(stats, piece)
                    if padded:
                        self._fill(self._slot, piece)
                        self._run_slot(sweep, self._slot, g)
                        self._scatter(sweep, self._slot, g)
                    else:
                        self._eager_tile(sweep, piece, g)
                    del piece
            stats["global_bytes"] = int(sum(t.numel() * t.element_size() for t in state.values()))
            with torch.no_grad():
                node_inter = self.model.scale_shift(node_es, self.heads[:n])
                from fastmace.model import energy as energy64

                if energy64.NODE_E0_KEY in data:
                    e0 = data[energy64.NODE_E0_KEY]
                else:  # a float64 model summed natively: its own table
                    e0 = self.model.atomic_energies_fn(self.node_attrs)[self.arange[:n], self.heads[:n]].double()
                energy = (e0 + node_inter.double()).sum().reshape(1)
                out_stress = None
                if stress:
                    volume = torch.linalg.det(cell.to(torch.float64)).abs()
                    out_stress = (strain_grad / volume).to(self.dtype)  # MACE: dE/d(strain) / V
            self.last = stats
            return ForceOutput(energy, forces.clone(), out_stress)
        finally:
            for module, count in saved:
                if count is not None:
                    module.register_buffer("streaming_count", count, persistent=False)

    def _globals(self, n, stress, pos, edge_index, shifts, heads):
        """The call's global buffers, kept between calls at fixed addresses (the graphs read them) and zeroed.

        A new set (another atom count, or the neighbour list's buffers moved) drops the captured graphs. The
        state (``x``, ``sc`` and their adjoints) is allocated from sweep 1's first outputs."""
        import torch

        g = getattr(self, "_g", None)
        if g is None or g["n"] != n or g["edge_index"] is not edge_index or g["shifts"] is not shifts:
            # n + 1 rows: a padded slot has at most n + 1 receiver rows (one padding row past the largest tile).
            g = dict(n=n, pos=pos, edge_index=edge_index, shifts=shifts, heads=torch.cat([heads, heads[-1:]]),
                     arange=torch.arange(n + 1, device=self.device),
                     forces=torch.zeros((n, 3), dtype=self.dtype, device=self.device),
                     node_es=torch.zeros((n,), dtype=self.dtype, device=self.device),
                     strain_grad=torch.zeros((3, 3), dtype=torch.float64, device=self.device), state={})
            self._g = g
            self._graphs = {}
        else:
            g["heads"].copy_(torch.cat([heads, heads[-1:]]))
        g["stress"] = stress
        for tensor in [g["forces"], g["node_es"], g["strain_grad"], *g["state"].values()]:
            tensor.zero_()
        return g

    def _state(self, g, x, sc):
        """Allocate ``x``, ``sc`` and their adjoints [N, width] on sweep 1's first outputs."""
        import torch

        state = g["state"]
        if "x" not in state:
            n = g["n"]
            for name, like in (("x", x), ("sc", sc), ("gx", x), ("gsc", sc)):
                state[name] = torch.zeros((n, like.shape[1]), dtype=self.dtype, device=self.device)
        return state

    @staticmethod
    def _note(stats, piece):
        nodes, edge_ids, _, r, h = piece
        stats["max_receivers"] = max(stats["max_receivers"], r)
        stats["max_halo"] = max(stats["max_halo"], h)
        stats["max_edges"] = max(stats["max_edges"], edge_ids.shape[0])

    def _slot_sizes(self, tiles, row_ptr, edge_index, local_of, stats):
        """(R, H, E) of a slot that holds every tile: one receiver slot more than the largest tile (the padding
        receiver), halo and edges with ``TileConfig.headroom`` above the largest, so later calls (atoms move, the
        list is rebuilt) rarely need a larger slot."""
        largest = [0, 0, 0]
        for receivers in tiles:
            nodes, edge_ids, _, r, h = self._tile_graph(receivers, row_ptr, edge_index, local_of)
            largest = [max(largest[0], r), max(largest[1], h), max(largest[2], edge_ids.shape[0])]
        head = 1.0 + float(self.config.headroom)
        return (largest[0] + 1, max(1, math.ceil(largest[1] * head)), max(1, math.ceil(largest[2] * head)))

    def _eager_tile(self, sweep, piece, g):
        """One sweep on one unpadded tile (``mode="eager"``)."""
        nodes, edge_ids, local, r, _ = piece
        edge_index, state = g["edge_index"], g["state"]
        ei = edge_index[:, edge_ids]
        vectors = g["pos"][ei[1]] - g["pos"][ei[0]] + g["shifts"][edge_ids]
        tile = (nodes, local, r, vectors, None, None)
        receivers = nodes[:r]
        strain = True if g["stress"] else None
        g_eps = None
        if sweep == 1:
            es, x, sc = self._first(tile, state)
            state = self._state(g, x, sc)
            g["node_es"][receivers] += es
            state["x"][receivers] = x
            state["sc"][receivers] = sc
            return
        if sweep == 2:
            es, g_vec, g_x, g_sc, g_eps = self._second(tile, state, strain)
            g["node_es"][receivers] += es
            state["gx"].index_add_(0, nodes, g_x)
            state["gsc"][receivers] = g_sc
        else:
            _, g_vec, g_eps = self._first(tile, state, adjoint=True, strain=strain)
        g["forces"].index_add_(0, ei[1], g_vec, alpha=-1)
        g["forces"].index_add_(0, ei[0], g_vec)
        if g_eps is not None:
            g["strain_grad"].add_(g_eps.double())

    def evaluate(self, atoms: Any, stress: bool = False):
        """Energy (eV), forces (eV/Å, float64) and the 3x3 stress (eV/Å³) or None of ``atoms`` (one-off)."""
        import torch

        if stress and not self.field.compute_stress:
            raise ValueError("stress=True needs a force field built with compute_stress=True")
        out = self(self.field._prepare_one_off(atoms))
        energy = float(out.energy.reshape(-1)[0])
        forces = out.forces.detach().to(torch.float64).cpu().numpy()
        values = out.stress.detach().to(torch.float64).cpu().numpy() if stress else None
        return energy, forces, values
