"""Receiver-tiled, three-sweep force call: a capacity prototype (docs/design/tiled-backend.md).

``TiledForceField`` wraps an eager ``MACEForceField`` of a two-interaction ``ScaleShiftMACE`` whose first
interaction reads species-only sender features (MH-1 with ``reassociate_first``, or stock: the first
interaction's input is the species embedding). One force call runs the rewritten modules on tiles: a sorted set
of receivers, the edges into them and the halo of their senders, laid out as MACE's LAMMPS ghosts so that the
interactions cut their outputs to the receivers themselves. Three sweeps:

1. interaction 0 forward without autograd: the inter-layer features h1 of every atom, kept globally;
2. interaction 1 forward and backward: the adjoint dE/dh1, scattered over the tile and its halo, and the edge
   gradients of the second interaction;
3. interaction 0 again, with autograd, backward of its own energy terms plus <dE/dh1, h1>: the remaining edge
   gradients.

Only h1, its adjoint and per-atom outputs grow with the atom count; every other intermediate is bounded by the
tile caps. Forces come from the edge-vector gradients (MACE's edge forces), the stress from the edge virial on a
symmetric strain of the edge vectors. The result equals the whole-system call up to float32 rounding (the tiles'
GEMMs see fewer rows; the force scatter is atomic).

Prototype scope: eager, no CUDA graphs, fixed cell per call, no PolarMACE, D3, joint embedding or embedding
readout. Tiles are planned on the host from a spatial binning of the fractional coordinates.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from fastmace.model.forces import ForceOutput


@dataclasses.dataclass(frozen=True)
class TileConfig:
    """Tile caps of the prototype: at most ``receivers`` receiver atoms and ``edges`` edges per tile.

    ``edges=None`` caps the edges at ``receivers`` times twice the structure's mean degree, so a tile is split
    where its receivers are unusually crowded.
    """

    receivers: int = 8192
    edges: int | None = None

    def __post_init__(self):
        if int(self.receivers) < 1:
            raise ValueError(f"TileConfig.receivers must be positive, got {self.receivers!r}")
        if self.edges is not None and int(self.edges) < 1:
            raise ValueError(f"TileConfig.edges must be positive or None, got {self.edges!r}")


class _Ghosts:
    """``lammps_class`` of the tile calls: not None, so the interactions cut their outputs to the receivers
    (``truncate_ghosts``); with ``first_layer=True`` their ``handle_lammps`` exchanges nothing."""


_GHOSTS = _Ghosts()


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
        self.model = model
        self.device = field.device
        self.dtype = field.dtype
        self._big = torch.tensor(torch.iinfo(torch.int64).max, device=self.device)
        self.last = {}

    @property
    def compute_stress(self):
        return self.field.compute_stress

    # ------------------------------------------------------------------ one tile

    def _tile_graph(self, receivers, row_ptr, edge_index, local_of):
        """Local graph of one tile: (nodes [R + H], edge ids [Ec], local edge_index [2, Ec], R)."""
        import torch

        r = receivers.shape[0]
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
        local_of[halo] = torch.arange(r, r + halo.shape[0], device=self.device)
        local = torch.stack([local_of[senders], torch.repeat_interleave(
            torch.arange(r, device=self.device), counts, output_size=total)])
        nodes = torch.cat([receivers, halo])
        local_of[nodes] = -1
        return nodes, edge_ids, local, r

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

    def _first(self, tile, h1=None, adjoint=None, strain=None):
        """Sweep 1 (``h1`` given: write it, no autograd) or sweep 3 (``adjoint`` given: edge gradients).

        Returns (node energy of the receivers before scale and shift, edge-vector gradient or None,
        strain gradient or None).
        """
        import torch

        model = self.model
        nodes, edge_ids, local, r, vectors = tile
        attrs = self.node_attrs[nodes]
        grad = adjoint is not None
        with torch.enable_grad() if grad else torch.no_grad():
            leaf = vectors.detach().requires_grad_(grad)
            eps = (torch.zeros((3, 3), dtype=self.dtype, device=self.device, requires_grad=True)
                   if grad and strain is not None else None)
            lengths, edge_attrs, edge_feats, cutoff = self._edges(leaf, attrs, local, eps)
            node_feats = model.node_embedding(attrs)
            out = self._layer(0, node_feats, attrs, edge_attrs, edge_feats, cutoff, local, r)
            heads = self.heads[:r]
            arange = self.arange[:r]
            es = model.readouts[0](out, heads)[arange, heads]
            if hasattr(model, "pair_repulsion"):
                es = model.pair_repulsion_fn(lengths, attrs, local, model.atomic_numbers)[:r] + es
            if not grad:
                if h1 is True:  # the first tile: the caller allocates h1 from the output's width
                    return es, out, None
                h1[nodes[:r]] = out
                return es, None, None
            objective = model.scale_shift(es, heads).sum() + (adjoint[nodes[:r]] * out).sum()
            wanted = [leaf] + ([eps] if eps is not None else [])
            grads = torch.autograd.grad(objective, wanted)
        return es.detach(), grads[0], grads[1] if eps is not None else None

    def _second(self, tile, h1, adjoint, strain):
        """Sweep 2: interaction 1 forward and backward; adds dE/dh1 into ``adjoint``."""
        import torch

        model = self.model
        nodes, edge_ids, local, r, vectors = tile
        attrs = self.node_attrs[nodes]
        with torch.enable_grad():
            leaf = vectors.detach().requires_grad_(True)
            eps = (torch.zeros((3, 3), dtype=self.dtype, device=self.device, requires_grad=True)
                   if strain is not None else None)
            feats = h1[nodes].detach().requires_grad_(True)
            _, edge_attrs, edge_feats, cutoff = self._edges(leaf, attrs, local, eps)
            out = self._layer(1, feats, attrs, edge_attrs, edge_feats, cutoff, local, r)
            heads = self.heads[:r]
            es = model.readouts[1](out, heads)[self.arange[:r], heads]
            objective = model.scale_shift(es, heads).sum()
            wanted = [leaf, feats] + ([eps] if eps is not None else [])
            grads = torch.autograd.grad(objective, wanted)
        adjoint.index_add_(0, nodes, grads[1])
        return es.detach(), grads[0], grads[2] if eps is not None else None

    # ------------------------------------------------------------------ the call

    def _bind(self):
        field = self.field
        field._bind_streaming_count()  # species tables; then every tile edge is a real edge
        saved = []
        for module in field._streaming:
            saved.append((module, module._buffers.get("streaming_count")))
            module.register_buffer("streaming_count", self._big, persistent=False)
        return saved

    def __call__(self, positions):
        """Energy [1] (float64), forces [N, 3] and stress [3, 3] or None at float64 ``positions``."""
        import torch

        field = self.field
        data = field.data
        stress = field.compute_stress
        saved = self._bind()
        try:
            with torch.no_grad():
                field.positions.copy_(positions)
            pos = field.positions.detach()
            edge_index = data["edge_index"]
            shifts = data["shifts"]
            n = pos.shape[0]
            self.node_attrs = data["node_attrs"]
            self.heads = (data["head"][data["batch"]] if "head" in data else torch.zeros_like(data["batch"])).long()
            self.arange = torch.arange(n, device=self.device)
            receiver = edge_index[1]
            row_ptr = torch.searchsorted(receiver, torch.arange(n + 1, device=self.device))
            num_edges = receiver.shape[0]
            caps = self.config
            edge_cap = caps.edges if caps.edges is not None else max(
                1, int(math.ceil(2 * caps.receivers * num_edges / max(n, 1))))
            cell = data["cell"].reshape(3, 3) if "cell" in data else torch.as_tensor(field.neighbours.cell)
            tiles, grid = plan_tiles(pos, cell, row_ptr, caps.receivers, edge_cap)
            local_of = torch.full((n,), -1, dtype=torch.int64, device=self.device)
            forces = torch.zeros((n, 3), dtype=self.dtype, device=self.device)
            strain_grad = torch.zeros((3, 3), dtype=torch.float64, device=self.device) if stress else None
            node_es = torch.zeros((n,), dtype=self.dtype, device=self.device)
            stats = dict(tiles=len(tiles), grid=grid, edge_cap=edge_cap, max_receivers=0, max_halo=0, max_edges=0)

            def build(receivers):
                nodes, edge_ids, local, r = self._tile_graph(receivers, row_ptr, edge_index, local_of)
                ei = edge_index[:, edge_ids]
                vectors = pos[ei[1]] - pos[ei[0]] + shifts[edge_ids]
                stats["max_receivers"] = max(stats["max_receivers"], r)
                stats["max_halo"] = max(stats["max_halo"], nodes.shape[0] - r)
                stats["max_edges"] = max(stats["max_edges"], edge_ids.shape[0])
                return nodes, edge_ids, local, r, vectors

            def scatter(tile, g_vec, g_eps):
                _, edge_ids, _, _, _ = tile
                ei = edge_index[:, edge_ids]
                forces.index_add_(0, ei[1], g_vec, alpha=-1)
                forces.index_add_(0, ei[0], g_vec)
                if g_eps is not None:
                    strain_grad.add_(g_eps.to(torch.float64))

            # Sweep 1: h1 of every atom.
            h1 = None
            for receivers in tiles:
                tile = build(receivers)
                if h1 is None:
                    es, out, _ = self._first(tile, h1=True)
                    h1 = torch.empty((n, out.shape[1]), dtype=self.dtype, device=self.device)
                    h1[tile[0][:tile[3]]] = out
                    del out
                else:
                    es, _, _ = self._first(tile, h1=h1)
                node_es[tile[0][:tile[3]]] += es
                del tile
            # Sweep 2: interaction 1 forward and backward.
            adjoint = torch.zeros_like(h1)
            for receivers in tiles:
                tile = build(receivers)
                es, g_vec, g_eps = self._second(tile, h1, adjoint, True if stress else None)
                node_es[tile[0][:tile[3]]] += es
                scatter(tile, g_vec, g_eps)
                del tile, g_vec
            # Sweep 3: interaction 0 replay and backward.
            for receivers in tiles:
                tile = build(receivers)
                _, g_vec, g_eps = self._first(tile, adjoint=adjoint, strain=True if stress else None)
                scatter(tile, g_vec, g_eps)
                del tile, g_vec
            del h1, adjoint
            with torch.no_grad():
                node_inter = self.model.scale_shift(node_es, self.heads)
                from fastmace.model import energy as energy64

                e0 = energy64.node_e0(data, None) if energy64.NODE_E0_KEY in data else None
                if e0 is None:
                    raise ValueError("TiledForceField needs the float64 energy treatment (float64_energy=True)")
                energy = (e0 + node_inter.double()).sum().reshape(1)
                out_stress = None
                if stress:
                    volume = torch.linalg.det(cell.to(torch.float64)).abs()
                    out_stress = (-strain_grad / volume).to(self.dtype)
            self.last = stats
            return ForceOutput(energy, forces, out_stress)
        finally:
            for module, count in saved:
                if count is not None:
                    module.register_buffer("streaming_count", count, persistent=False)

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
