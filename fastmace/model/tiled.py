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
    (every tile in one padded slot, eagerly), ``"graphs"`` (the padded slot, one CUDA graph per sweep) or
    ``"compiled"`` (the padded slot, each sweep's forward compiled by Inductor at the slot's shapes; the force
    derivative outside the compiled frame; needs a compiled force field, whose model is traceable).
    ``headroom``: a padded slot's halo count above the largest of its tiles', as a fraction. ``bucket``: the padded
    modes keep one slot per edge bucket, ``bucket`` times the largest tile's edges (rounded up to 256) wide, so a
    small tile runs in a small slot; 1 or more gives one slot for every tile.
    ``budget_receivers``: larger receiver caps the call may take after its first call (at ``receivers``), from
    the memory model: the first call measures its workspace per receiver (the peak above what stays allocated
    after it: the tiled state, the field and the outputs), and the call then takes the largest candidate whose
    predicted workspace, with 10 % headroom, fits the memory left on the device. Larger tiles run faster (fewer
    launches, a smaller halo fraction; docs/validation/tiled-screen.md), and at the capacity limit the
    workspace does not fit, so the cap stays at ``receivers``. Chosen once; needs a CUDA device (elsewhere the
    cap stays).
    ``state``: where x, sc and their adjoints live, ``"device"`` or ``"host"`` (pinned host memory on CUDA; each
    tile copies its rows to the device and its results back, so the device holds no per-atom state of the
    sweeps; not with ``mode="graphs"``, whose captured bodies read the state themselves).
    """

    receivers: int = 8192
    edges: int | None = None
    mode: str = "eager"
    headroom: float = 0.1
    bucket: float = 0.15
    # "global": the force field's neighbour list; "tiles": each tile's edges from a cell binning of the current
    # positions (``fastmace.neighbors.tile_list``), bitwise the global list's, without it (T3).
    neighbours: str = "global"
    budget_receivers: tuple[int, ...] = ()
    state: str = "device"

    def __post_init__(self):
        if self.mode not in ("eager", "padded", "graphs", "compiled"):
            raise ValueError(f"TileConfig.mode must be 'eager', 'padded', 'graphs' or 'compiled', got {self.mode!r}")
        if not 0 <= float(self.headroom) <= 10:
            raise ValueError(f"TileConfig.headroom must be in [0, 10], got {self.headroom!r}")
        if self.neighbours not in ("global", "tiles"):
            raise ValueError(f"TileConfig.neighbours must be 'global' or 'tiles', got {self.neighbours!r}")
        if not 0 < float(self.bucket) <= 10:
            raise ValueError(f"TileConfig.bucket must be in (0, 10], got {self.bucket!r}")
        if int(self.receivers) < 1:
            raise ValueError(f"TileConfig.receivers must be positive, got {self.receivers!r}")
        if self.state not in ("device", "host"):
            raise ValueError(f"TileConfig.state must be 'device' or 'host', got {self.state!r}")
        if self.state == "host" and self.mode == "graphs":
            raise ValueError("TileConfig.state='host' cannot run in mode 'graphs': the captured bodies read the "
                             "state from device memory")
        if any(int(r) < 1 for r in self.budget_receivers):
            raise ValueError(f"TileConfig.budget_receivers must be positive, got {self.budget_receivers!r}")
        if self.edges is not None and int(self.edges) < 1:
            raise ValueError(f"TileConfig.edges must be positive or None, got {self.edges!r}")


class _State:
    """x, sc and their adjoints [N, width] (sweep 1 allocates them from its first outputs), on the device or in
    host memory (``TileConfig.state``). Tiles read rows to the device and put or add rows; on the device these
    are the plain indexing and ``index_add_``, in host memory the same on the host after one copy."""

    def __init__(self, device, dtype, host):
        self.device, self.dtype, self.host = device, dtype, host
        self.tensors = {}

    def allocate(self, n, widths):
        import torch

        if self.tensors:
            return
        pin = self.host and self.device.type == "cuda"
        where = "cpu" if self.host else self.device
        for name, width in widths.items():
            self.tensors[name] = torch.zeros((n, width), dtype=self.dtype, device=where, pin_memory=pin)

    def zero(self):
        for tensor in self.tensors.values():
            tensor.zero_()

    def bytes(self):
        return int(sum(t.numel() * t.element_size() for t in self.tensors.values()))

    def rows(self, name, index):
        """``state[name][index]`` on the device."""
        if not self.host:
            return self.tensors[name][index]
        return self.tensors[name].index_select(0, index.cpu()).to(self.device)

    def put(self, name, index, values):
        """``state[name][index] = values``."""
        if not self.host:
            self.tensors[name][index] = values
        else:
            self.tensors[name][index.cpu()] = values.detach().cpu()

    def add(self, name, index, values):
        """``state[name].index_add_(0, index, values)``."""
        if not self.host:
            self.tensors[name].index_add_(0, index, values)
        else:
            self.tensors[name].index_add_(0, index.cpu(), values.detach().cpu())


class _Ghosts:
    """``lammps_class`` of the tile calls: not None, so the interactions cut their outputs to the receivers
    (``truncate_ghosts``); with ``first_layer=True`` their ``handle_lammps`` exchanges nothing."""


_GHOSTS = _Ghosts()


@functools.cache
def _holder_class():
    import torch

    class Holder(torch.nn.Module):
        """Stand-in for a module of interaction 1 in sweep 2: returns its ``value`` buffer, whatever its input
        (a buffer, so a compiled tile body takes it as an input)."""

        def __init__(self):
            super().__init__()
            self.register_buffer("value", None, persistent=False)

        def forward(self, *args, **kwargs):
            return self.value

    return Holder


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
        # A compiled force field's model is the simplified, traceable copy (``compile_model``): the tile bodies
        # run its modules (``_orig_mod``), eagerly or, in mode "compiled", under their own torch.compile. An eager
        # field's e3nn modules cannot be compiled (their TorchScript parts would differentiate to zero).
        if self.config.mode == "compiled" and field.compile_mode == "none":
            raise ValueError("TileConfig.mode='compiled' needs a compiled force field (compile_mode='default')")
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
        self._holders = (_holder_class()(), _holder_class()())
        # The reassociated first interactions read species tables evaluated at the tile's own row count (the
        # field's ``species_tables``): bitwise the features the tile's embedding gives its atoms.
        self._reassociations = [entry[1] for entry in field._tables]
        self._compiled = {}
        self._slots = {}  # edge bucket -> padded slot (with its graphs)
        self._edge_step = None
        self.receivers = int(self.config.receivers)  # the receiver cap (budget_receivers may raise it once)
        self._released = False  # the allocator's cache, released once before the first tile
        self._budgeted = not self.config.budget_receivers
        self.slot_changes = 0  # slots created or grown (each change compiles and captures again)
        self._graph_key = None
        self._pool = None
        self._stream = None
        self.last = {}

    @property
    def compute_stress(self):
        return self.field.compute_stress

    # ------------------------------------------------------------------ one tile

    def _tile_graph(self, receivers, source, local_of, halo_start=None):
        """Local graph of one tile: ``(nodes [R + H], ei [2, Ec], shifts [Ec, 3], local edge_index [2, Ec], R, H)``.

        ``source(receivers)`` gives the edges into the receivers, receiver-sorted: ``(ei, shifts)`` with ``ei`` the
        global (sender, receiver) pairs. The halo's local indices start at ``halo_start`` (default: after the
        receivers; a padded slot puts them after its receiver slots)."""
        import torch

        r = receivers.shape[0]
        start = r if halo_start is None else int(halo_start)
        ei, shifts = source(receivers)
        senders = ei[0]
        local_of[receivers] = torch.arange(r, device=self.device)
        outside = senders[local_of[senders] < 0]
        halo = torch.unique(outside)
        local_of[halo] = torch.arange(start, start + halo.shape[0], device=self.device)
        local = torch.stack([local_of[senders], local_of[ei[1]]])
        local_of[receivers] = -1
        local_of[halo] = -1
        return torch.cat([receivers, halo]), ei, shifts, local, r, halo.shape[0]

    def _source(self, positions, n):
        """``(source, row_ptr, edges per atom)`` of this call's edges: the force field's list, or (neighbours
        "tiles") a cell binning of ``positions`` (float64) that builds each tile's edges."""
        import torch

        data = self.field.data
        if self.config.neighbours == "tiles":
            from fastmace.neighbors.tile_list import TileNeighbours

            bins = TileNeighbours.like(self.field.neighbours)
            bins.bin(positions)

            def tiles(receivers):
                sender, receiver, _, shifts = bins.edges(receivers)
                return torch.stack([sender, receiver]), shifts

            # No degrees before the tiles' builds: the tiles are planned by receivers only.
            return tiles, torch.zeros(n + 1, dtype=torch.int64, device=self.device), None
        edge_index, shifts = data["edge_index"], data["shifts"]
        row_ptr = torch.searchsorted(edge_index[1], torch.arange(n + 1, device=self.device))

        def global_list(receivers):
            starts = row_ptr[receivers]
            counts = row_ptr[receivers + 1] - starts
            total = int(counts.sum())
            offsets = torch.cumsum(counts, 0) - counts
            edge_ids = (torch.repeat_interleave(starts - offsets, counts, output_size=total)
                        + torch.arange(total, device=self.device))
            return edge_index[:, edge_ids], shifts[edge_ids]

        return global_list, row_ptr, edge_index.shape[1] / max(n, 1)

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
            es, x, sc = self._forward(3 if adjoint else 1)(leaf, eps, attrs, local, r)
            heads = self.heads[:r]
            if not adjoint:
                return self._masked(es, recv_mask), self._masked(x, recv_mask), self._masked(sc, recv_mask)
            objective = (self._masked(model.scale_shift(es, heads), recv_mask).sum()
                         + (self._masked(state.rows("gx", receivers), recv_mask) * x).sum()
                         + (self._masked(state.rows("gsc", receivers), recv_mask) * sc).sum())
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
            x = state.rows("x", nodes).detach().requires_grad_(True)
            sc = state.rows("sc", nodes[:r]).detach().requires_grad_(True)
            modules = interaction._modules
            saved = modules["linear_up"], modules["skip_tp"]
            holder_x, holder_sc = self._holders
            holder_x.value, holder_sc.value = x, sc
            modules["linear_up"], modules["skip_tp"] = holder_x, holder_sc
            try:  # their real counterparts ran in sweep 1, on the h1 that is not kept
                objective, es = self._forward(2)(leaf, x, eps, attrs, local, recv_mask, r)
            finally:
                modules["linear_up"], modules["skip_tp"] = saved
                holder_x.value = holder_sc.value = None
            wanted = [leaf, x, sc] + ([eps] if eps is not None else [])
            grads = torch.autograd.grad(objective, wanted)
        return (self._masked(es.detach(), recv_mask), grads[0], self._masked(grads[1], node_mask),
                self._masked(grads[2], recv_mask), grads[3] if eps is not None else None)

    def _first_forward(self, leaf, eps, attrs, local, r):
        """Interaction 0's forward on a tile: ``(es, x, sc)`` (sweeps 1 and 3; compiled in mode "compiled")."""
        model = self.model
        lengths, edge_attrs, edge_feats, cutoff = self._edges(leaf, attrs, local, eps)
        node_feats = model.node_embedding(attrs)
        h1 = self._layer(0, node_feats, attrs, edge_attrs, edge_feats, cutoff, local, r)
        heads = self.heads[:r]
        es = model.readouts[0](h1, heads)[self.arange[:r], heads]
        if hasattr(model, "pair_repulsion"):
            es = model.pair_repulsion_fn(lengths, attrs, local, model.atomic_numbers)[:r] + es
        x, sc = self._consumers(h1, attrs[:r])
        return es, x, sc

    def _second_forward(self, leaf, x, eps, attrs, local, recv_mask, r):
        """Sweep 2's forward up to its objective (interaction 1 reads the holders' x and sc): ``(objective, es)``.
        Compiled as one function in mode "compiled"."""
        model = self.model
        _, edge_attrs, edge_feats, cutoff = self._edges(leaf, attrs, local, eps)
        out = self._layer(1, x, attrs, edge_attrs, edge_feats, cutoff, local, r)
        heads = self.heads[:r]
        es = model.readouts[1](out, heads)[self.arange[:r], heads]
        return self._masked(model.scale_shift(es, heads), recv_mask).sum(), es

    def _forward(self, sweep):
        """The sweep's forward: plain, or (mode "compiled") compiled on first use, once per sweep (sweeps 1 and 3
        apart: without and with autograd); each slot's shapes are one specialisation."""
        plain = {1: self._first_forward, 2: self._second_forward, 3: self._first_forward}[sweep]
        if self.config.mode != "compiled":
            return plain
        compiled = self._compiled.get(sweep)
        if compiled is None:
            import torch

            from fastmace.rewrites import inductor as inductor_records

            # One specialisation per slot (edge bucket) and sweep; dynamo would run eager beyond its limit.
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            options = inductor_records.inductor_options(self.model)
            compiled = torch.compile(plain, dynamic=False, fullgraph=False, **({"options": options} if options else {}))
            self._compiled[sweep] = compiled
        return compiled

    # ------------------------------------------------------------------ padded slot

    def _new_slot(self, sizes):
        """Static buffers of one padded tile: ``sizes = (R, H, E)`` receiver slots (one more than any tile, so a
        padding receiver always exists), halo rows and edges."""
        import torch

        receivers, halo, edges = sizes
        i64 = torch.int64
        new = functools.partial(torch.zeros, device=self.device)
        tables = self.field.species_tables(receivers + halo) if self._reassociations else []
        return dict(sizes=sizes, nodes=new(receivers + halo, dtype=i64), tables=tables,
                    node_mask=new(receivers + halo, dtype=self.dtype), recv_mask=new(receivers, dtype=self.dtype),
                    ei=new((2, edges), dtype=i64), shifts=new((edges, 3), dtype=self.dtype),
                    local=new((2, edges), dtype=i64), edge_mask=new(edges, dtype=self.dtype),
                    count=new((), dtype=i64), out={}, graphs={})

    def _fill(self, slot, piece):
        """Copy one tile (``_tile_graph``'s output at the slot's halo start) into the slot's static buffers.

        Padding: receivers and halo rows read atom 0 and are masked; padding edges join the last receiver slot
        (always padding) to itself, after the tile's edges (so the receivers stay sorted), beyond the cutoff and
        beyond the tile's edge count (the streaming kernels' ``streaming_count``)."""
        nodes, ei, shifts, local, r, h = piece
        receivers, _, edges = slot["sizes"]
        e = ei.shape[1]
        slot["nodes"].zero_()
        slot["nodes"][:r] = nodes[:r]
        slot["nodes"][receivers:receivers + h] = nodes[r:]
        slot["node_mask"].zero_()
        slot["node_mask"][:r] = 1
        slot["node_mask"][receivers:receivers + h] = 1
        slot["recv_mask"].zero_()
        slot["recv_mask"][:r] = 1
        slot["ei"].zero_()
        slot["ei"][:, :e] = ei
        slot["shifts"].zero_()
        slot["shifts"][:e] = shifts
        slot["local"].fill_(receivers - 1)
        slot["local"][:, :e] = local
        slot["edge_mask"].zero_()
        slot["edge_mask"][:e] = 1
        slot["count"].fill_(e)

    def _body(self, sweep, slot, g):
        """One sweep on the slot's tile, reading only static buffers; writes ``slot["out"]`` (graph-capturable)."""
        import torch

        receivers = slot["sizes"][0]
        nodes, ei = slot["nodes"], slot["ei"]
        vectors = g["pos"][ei[1]] - g["pos"][ei[0]] + slot["shifts"]
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
            state.add("x", recv, out["x"])
            state.add("sc", recv, out["sc"])
            return
        if sweep == 2:
            g["node_es"].index_add_(0, recv, out["es"])
            state.add("gx", slot["nodes"], out["g_x"])
            state.add("gsc", recv, out["g_sc"])
        ei = slot["ei"]
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
        graph = slot["graphs"].get(sweep)
        if graph is None:
            from fastmace import _patches

            _patches.set_capture_safe_mask_head(True)  # MACE's readout mask assigns by index (as GraphEngine)
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
            slot["graphs"][sweep] = graph
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
        n = pos.shape[0]
        self.node_attrs = data["node_attrs"]
        heads = (data["head"][data["batch"]] if "head" in data else torch.zeros_like(data["batch"])).long()
        source, row_ptr, degree = self._source(positions, n)
        caps = self.config
        edge_cap = caps.edges if caps.edges is not None else (
            2**62 if degree is None else max(1, int(math.ceil(2 * self.receivers * degree))))
        cell = data["cell"].reshape(3, 3) if "cell" in data else torch.as_tensor(field.neighbours.cell)
        tiles, grid = plan_tiles(pos, cell, row_ptr, self.receivers, edge_cap)
        local_of = torch.full((n,), -1, dtype=torch.int64, device=self.device)
        measure = not self._budgeted and self.device.type == "cuda"
        if measure:  # the workspace of this call: its peak above what stays allocated after it
            torch.cuda.reset_peak_memory_stats(self.device)
        stats = dict(tiles=len(tiles), grid=grid, edge_cap=edge_cap, max_receivers=0, max_halo=0, max_edges=0,
                     receivers=self.receivers,
                     edges=0, mode=caps.mode, neighbours=caps.neighbours)
        keys = self._slot_keys(tiles, source, local_of, n) if padded else None
        if padded:
            stats["slots"] = {key: self._slots[key]["sizes"] for key in sorted(set(keys))}
            stats["slot_changes"] = self.slot_changes
        if not self._released and self.device.type == "cuda":
            # The force field's construction leaves its peak in PyTorch's cache; libraries that allocate outside
            # it (cuEquivariance's cudaMallocAsync) would find the device full in the first tile.
            torch.cuda.empty_cache()
        self._released = True
        saved = self._bind()
        bound = None
        try:
            g = self._globals(n, stress, pos, heads)
            self.heads, self.arange = g["heads"], g["arange"]
            if padded:
                key = (id(g), pos.data_ptr(), self.node_attrs.data_ptr(), stress)
                if key != self._graph_key:  # the graphs read these addresses
                    for slot in self._slots.values():
                        slot["graphs"].clear()
                    self._graph_key = key
            forces, node_es, strain_grad, state = g["forces"], g["node_es"], g["strain_grad"], g["state"]
            for sweep in (1, 2, 3):
                for index, receivers in enumerate(tiles):
                    slot = self._slots[keys[index]] if padded else None
                    piece = self._tile_graph(receivers, source, local_of, slot["sizes"][0] if padded else None)
                    self._note(stats, piece)
                    if sweep == 1:
                        stats["edges"] += piece[1].shape[1]
                    if padded:
                        if bound is not slot:  # the streaming kernels read this slot's edge count and tables
                            for module in field._streaming:
                                module.register_buffer("streaming_count", slot["count"], persistent=False)
                            self._bind_tables(slot["tables"])
                            bound = slot
                        self._fill(slot, piece)
                        self._run_slot(sweep, slot, g)
                        self._scatter(sweep, slot, g)
                    else:
                        self._eager_tile(sweep, piece, g)
                    del piece
            stats["global_bytes"] = state.bytes()
            stats["host_bytes"] = state.bytes() if state.host else 0
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
            out = ForceOutput(energy, forces.clone(), out_stress)
            if not self._budgeted:
                self._budgeted = True
                self._choose_receivers(self._memory())  # None without a CUDA device: the cap stays
            return out
        finally:
            for module, count in saved:
                if count is not None:
                    module.register_buffer("streaming_count", count, persistent=False)

    def _memory(self):
        """``(workspace, available)`` bytes after a measured call, or None without a CUDA device: the call's peak
        above what stays allocated, and the device's free memory plus the allocator's cached, unallocated
        blocks."""
        import torch

        if self.device.type != "cuda":
            return None
        allocated = torch.cuda.memory_allocated(self.device)
        free, _ = torch.cuda.mem_get_info(self.device)
        return (torch.cuda.max_memory_allocated(self.device) - allocated,
                free + torch.cuda.memory_reserved(self.device) - allocated)

    def _choose_receivers(self, memory):
        """``budget_receivers``: the largest cap whose predicted workspace (the measured workspace per receiver,
        times the cap, plus 10 %) fits ``available``; the padded slots are planned again for it."""
        if memory is None:
            return
        workspace, available = memory
        per_receiver = workspace / self.receivers
        fits = [r for r in (self.receivers, *map(int, self.config.budget_receivers))
                if 1.1 * per_receiver * r <= available]
        chosen = max(fits, default=self.receivers)
        if chosen != self.receivers:
            self.receivers = chosen
            self._slots.clear()  # their sizes and edge buckets were for the old cap
            self._edge_step = None
            self._graph_key = None

    def _globals(self, n, stress, pos, heads):
        """The call's global buffers, kept between calls at fixed addresses (the graphs read them) and zeroed.

        A new set (another atom count) drops the captured graphs. The state (``x``, ``sc`` and their adjoints) is
        allocated from sweep 1's first outputs."""
        import torch

        g = getattr(self, "_g", None)
        if g is None or g["n"] != n:
            # n + 1 rows: a padded slot has at most n + 1 receiver rows (one padding row past the largest tile).
            g = dict(n=n, pos=pos, heads=torch.cat([heads, heads[-1:]]),
                     arange=torch.arange(n + 1, device=self.device),
                     forces=torch.zeros((n, 3), dtype=self.dtype, device=self.device),
                     node_es=torch.zeros((n,), dtype=self.dtype, device=self.device),
                     strain_grad=torch.zeros((3, 3), dtype=torch.float64, device=self.device),
                     state=_State(self.device, self.dtype, self.config.state == "host"))
            self._g = g
        else:
            g["heads"].copy_(torch.cat([heads, heads[-1:]]))
        g["stress"] = stress
        for tensor in [g["forces"], g["node_es"], g["strain_grad"]]:
            tensor.zero_()
        g["state"].zero()
        return g

    def _state(self, g, x, sc):
        """Allocate ``x``, ``sc`` and their adjoints [N, width] on sweep 1's first outputs."""
        state = g["state"]
        state.allocate(g["n"], dict(x=x.shape[1], sc=sc.shape[1], gx=x.shape[1], gsc=sc.shape[1]))
        return state

    @staticmethod
    def _note(stats, piece):
        nodes, ei, _, _, r, h = piece
        stats["max_receivers"] = max(stats["max_receivers"], r)
        stats["max_halo"] = max(stats["max_halo"], h)
        stats["max_edges"] = max(stats["max_edges"], ei.shape[1])

    def _slot_keys(self, tiles, source, local_of, n):
        """Each tile's slot: the edge bucket ``ceil(edges / step)``, with ``step`` set on the first call from the
        largest tile (``TileConfig.bucket`` of its edges, at least 256, a multiple of 256). A slot holds one
        receiver row more than the largest of its tiles with ``TileConfig.headroom``, at most the receiver cap (the
        extra row is the padding receiver), their halo with the same headroom and its bucket's edge count. A slot
        grows (and its graphs and compiled shapes go) only when a tile does not fit it."""
        sizes = []
        for receivers in tiles:
            _, ei, _, _, r, h = self._tile_graph(receivers, source, local_of)
            sizes.append((r, h, int(ei.shape[1])))
        if self._edge_step is None:
            largest = max(e for _, _, e in sizes)
            self._edge_step = max(256, 256 * math.ceil(float(self.config.bucket) * largest / 256))
        step = self._edge_step
        keys = [max(1, math.ceil(e / step)) for _, _, e in sizes]
        head = 1.0 + float(self.config.headroom)
        for key in set(keys):
            members = [size for size, k in zip(sizes, keys) if k == key]
            # Receivers with headroom too (tiles are planned again every call, so their sizes move a little), but
            # never past the cap, which no tile exceeds: full tiles keep their slot.
            rows = min(self.receivers, n,  # nor the atom count
                       math.ceil(max(r for r, _, _ in members) * head))
            need = (rows + 1, max(1, math.ceil(max(h for _, h, _ in members) * head)),
                    key * step)
            slot = self._slots.get(key)
            if slot is None or any(a > b for a, b in zip(need, slot["sizes"])):
                if slot is None:
                    grown = need
                else:  # an exceeded size grows by at least a quarter (receivers at most the cap), so a slot changes
                    # a few times, not once per tile a little larger than the last
                    rows_cap = min(self.receivers, n) + 1
                    grown = tuple(b if a <= b else max(a, math.ceil(1.25 * b)) for a, b in zip(need, slot["sizes"]))
                    grown = (min(grown[0], max(rows_cap, need[0])), grown[1], grown[2])
                self._slots[key] = self._new_slot(grown)
                self.slot_changes += 1
        return keys

    def _bind_tables(self, tables):
        for reassociation, table in zip(self._reassociations, tables):
            reassociation.bind(table)

    def _eager_tile(self, sweep, piece, g):
        """One sweep on one unpadded tile (``mode="eager"``)."""
        nodes, ei, shifts, local, r, _ = piece
        if sweep != 2 and self._reassociations:  # interaction 0 runs on the tile's R + H rows
            self._bind_tables(self.field.species_tables(nodes.shape[0]))
        state = g["state"]
        vectors = g["pos"][ei[1]] - g["pos"][ei[0]] + shifts
        tile = (nodes, local, r, vectors, None, None)
        receivers = nodes[:r]
        strain = True if g["stress"] else None
        g_eps = None
        if sweep == 1:
            es, x, sc = self._first(tile, state)
            state = self._state(g, x, sc)
            g["node_es"][receivers] += es
            state.put("x", receivers, x)
            state.put("sc", receivers, sc)
            return
        if sweep == 2:
            es, g_vec, g_x, g_sc, g_eps = self._second(tile, state, strain)
            g["node_es"][receivers] += es
            state.add("gx", nodes, g_x)
            state.put("gsc", receivers, g_sc)
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
