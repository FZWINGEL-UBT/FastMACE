"""Per-tile neighbour lists for the tiled force call (docs/design/tiled-backend.md; plans/capacity-ideas-20261030.md
T3, the list-build tiling of the seventh list).

The tiled force call needs, for one tile at a time, the edges whose receivers are in the tile. ``TileNeighbours``
finds them from a cell binning of the current positions, without the global list: memory per atom is the bins
(about 50 B), and per tile only its candidates.

**Exactness.** A tile's edges are bitwise the edges the global skin list (``SkinNeighbourList``) holds for those
receivers after ``compact`` at the same positions, in the same order:

- the same decision: unit shifts in the list's float dtype, ``times_3x3`` with the list's float64 cell, then
  ``positions[r] - positions[s] + shifts`` in float64 against the squared cutoff, as ``compact`` computes it;
- the same stored values: unit shifts in the float dtype, shifts as the float64 shifts cast to it;
- the same order: the global build's sort key, receiver (and sender group), then sender. Its image term does not
  matter here: the bins need every plane spacing of the cell to hold at least ``MIN_BINS`` bins of the cutoff, so
  a pair has at most one image within the cutoff.

Cells with fewer than ``MIN_BINS`` bins along an axis are refused; the tiled path is for large cells.
"""

from __future__ import annotations

import torch

from .skin import times_3x3

CHUNK = 1 << 22  # atoms per chunk of the binning

MIN_BINS = 3  # bins per axis: a pair then has one image within the cutoff, and +-1 bin covers it
# Bins of half the cutoff and a +-2 stencil (125 bins of (2.5 c)^3 instead of 27 of (3 c)^3: 42 % fewer candidates,
# the search about 90 % of a list rebuild at 1M atoms; tiled-1m-speed.md, "List rebuilds") where every axis holds
# at least 2 * STENCIL_SPLIT + 1 of them, so the stencil does not wrap onto itself (every cell with MIN_BINS
# cutoff-wide bins holds six). The same edges in the same order.
STENCIL_SPLIT = 2


class TileNeighbours:
    """Cell bins of the current positions and each tile's edges within ``cutoff``.

    ``cell`` (3, 3) float64; ``float_dtype``: the list's (the model's) float dtype; ``sender_groups`` (N,) int64
    and ``group_count``: the segmented first sum's sender groups, as the global list sorts them.
    """

    def __init__(self, cutoff, cell, float_dtype, sender_groups=None, group_count=1, device=None, split=None):
        self.cutoff = float(cutoff)
        self.cell = torch.as_tensor(cell, dtype=torch.float64, device=device)
        self.device = self.cell.device
        self.inverse = torch.linalg.inv(self.cell)
        self.float_dtype = float_dtype
        self.groups = None if sender_groups is None else torch.as_tensor(sender_groups, device=self.device)
        self.group_count = int(group_count)
        spacings = 1.0 / torch.linalg.norm(self.inverse, dim=0)  # plane spacing along each lattice vector
        self.bins = [int(torch.floor(s / self.cutoff)) for s in spacings]
        if min(self.bins) < MIN_BINS:
            raise ValueError(f"per-tile neighbour lists need at least {MIN_BINS} cutoff-wide bins along each cell "
                             f"axis; this cell has {self.bins} at a {self.cutoff} Angstrom cutoff")
        # ``split`` bins per cutoff (``STENCIL_SPLIT`` where the cell holds enough of them, else 1) and a +-split
        # stencil: a pair within the cutoff is at most ``split`` bins apart along each axis (a bin is at least
        # cutoff / split wide across its planes), and 2 * split + 1 bins per axis keep each neighbour bin once.
        if split is None:
            fine = [int(torch.floor(s * STENCIL_SPLIT / self.cutoff)) for s in spacings]
            split = STENCIL_SPLIT if min(fine) >= 2 * STENCIL_SPLIT + 1 else 1
        self.split = int(split)
        if self.split > 1:
            self.bins = [int(torch.floor(s * self.split / self.cutoff)) for s in spacings]
            if min(self.bins) < 2 * self.split + 1:
                raise ValueError(f"a +-{self.split} stencil needs {2 * self.split + 1} bins per axis, got {self.bins}")
        d = torch.arange(-self.split, self.split + 1, device=self.device)
        self._neighbour_bins = torch.stack(torch.meshgrid(d, d, d, indexing="ij"), dim=-1).reshape(-1, 3)

    def bin(self, positions):
        """Bin float64 ``positions`` (N, 3) by their wrapped fractional coordinates.

        In chunks of ``CHUNK`` atoms, keeping each atom's integer cell and bin as int32 (the same integer values
        as their float64 and int64 forms, so ``edges`` is unchanged): the whole-system transients of a tiled call
        decide its peak at the largest sizes (tiled-call-memory.md)."""
        k = torch.tensor(self.bins, device=self.device)
        self.positions = positions
        n = positions.shape[0]
        nbins = int(k.prod())
        # The narrowest integer dtypes that hold the values (bins per axis and the atoms' cell images below 2**14:
        # int16; bins and atoms below 2**31: int32 flat bin and order), each value unchanged; 30 bytes per atom
        # against 56 in int32 and int64 (tiled capacity, lever 5).
        narrow = max(self.bins) < 2**14 and n < 2**31 and nbins < 2**31
        self.offsets = torch.empty((n, 3), dtype=torch.int32, device=self.device)  # integer cell of each atom
        self.atom_bin = torch.empty((n, 3), dtype=torch.int16 if narrow else torch.int32, device=self.device)
        flat = torch.empty(n, dtype=torch.int32 if narrow else torch.int64, device=self.device)
        wide = 0  # the largest cell image, for the offsets' dtype
        for start in range(0, n, CHUNK):
            stop = min(start + CHUNK, n)
            frac = times_3x3(positions[start:stop], self.inverse)
            offsets = torch.floor(frac)
            b = torch.minimum(torch.clamp(((frac - offsets) * k).long(), min=0), k - 1)
            self.offsets[start:stop] = offsets
            wide = max(wide, int(offsets.abs().max()) if stop > start else 0)
            self.atom_bin[start:stop] = b
            flat[start:stop] = (b[:, 0] * k[1] + b[:, 1]) * k[2] + b[:, 2]
            del frac, offsets, b
        if narrow and wide < 2**14:
            self.offsets = self.offsets.to(torch.int16)
        self.order = torch.sort(flat, stable=True).indices
        if narrow:
            self.order = self.order.to(torch.int32)
        self.starts = torch.searchsorted(flat[self.order], torch.arange(nbins + 1, dtype=flat.dtype,
                                                                         device=self.device))
        del flat
        self._k = k

    def edges(self, receivers, inner=None):
        """(sender, receiver, unit_shifts, shifts) of the edges into ``receivers`` (sorted int64 atom indices),
        in the global list's order; unit shifts and shifts in the float dtype. ``inner`` (a cutoff below this
        one): also the edges' mask of distances below it, from the same float64 vectors (a kept superset's
        filter on the call that searches it)."""
        k = self._k
        natoms = self.positions.shape[0]
        stencil = self._neighbour_bins.shape[0]  # 27, or 125 with half-cutoff bins
        nb = self.atom_bin[receivers][:, None, :] + self._neighbour_bins[None, :, :]  # (R, stencil, 3)
        t = torch.div(nb, k, rounding_mode="floor")  # the image of the neighbour bin
        nb = nb - t * k
        ids = ((nb[..., 0] * k[1] + nb[..., 1]) * k[2] + nb[..., 2]).reshape(-1)
        counts = self.starts[ids + 1] - self.starts[ids]
        total = int(counts.sum())
        slot = torch.repeat_interleave(torch.arange(ids.shape[0], device=self.device), counts, output_size=total)
        first = torch.cumsum(counts, 0) - counts
        sender = self.order[self.starts[ids][slot] + torch.arange(total, device=self.device) - first[slot]].long()
        receiver = receivers.repeat_interleave(stencil)[slot]
        image = t.reshape(-1, 3)[slot].to(torch.int32)
        # The sender's image at its bin's wrap: x_r - x_s + S cell with S = off_s - off_r - t (small integers: exact
        # in int32 as in float64, half the transient).
        unit = (self.offsets[sender].to(torch.int32) - self.offsets[receiver].to(torch.int32) - image).to(
            self.float_dtype)
        del image
        shifts64 = times_3x3(unit.to(torch.float64), self.cell)
        vectors = self.positions[receiver] - self.positions[sender] + shifts64
        r2 = (vectors * vectors).sum(dim=1)
        del vectors
        keep = (r2 < self.cutoff ** 2) & ~((sender == receiver) & (unit == 0).all(dim=1))
        sender, receiver, unit, shifts64, r2 = sender[keep], receiver[keep], unit[keep], shifts64[keep], r2[keep]
        primary = receiver if self.groups is None else receiver * self.group_count + self.groups[sender]
        order = torch.sort(primary * natoms + sender).indices
        out = sender[order], receiver[order], unit[order], shifts64[order].to(self.float_dtype)
        return out if inner is None else out + (r2[order] < inner ** 2,)

    def shifts(self, unit):
        """The shifts of unit shifts ``unit`` (the float dtype), as ``edges`` gives them: ``times_3x3`` with the
        float64 cell, element-wise per row, then the float dtype."""
        return times_3x3(unit.to(torch.float64), self.cell).to(self.float_dtype)

    @classmethod
    def like(cls, neighbours, cell=None, cutoff=None, split=None):
        """A ``TileNeighbours`` with a ``SkinNeighbourList``'s cutoff (or ``cutoff``: a search radius, e.g. cutoff +
        skin), float dtype, sender groups and current cell (``split``: bins per cutoff, default by the cell)."""
        if cell is None:
            cell = neighbours.cell_t if neighbours.variable_cell else neighbours.build_cell
        groups = neighbours.sender_groups if neighbours.sort_receivers else None
        return cls(neighbours.cutoff if cutoff is None else cutoff, cell, neighbours.float_dtype, groups,
                   neighbours.group_count if groups is not None else 1, device=neighbours.device, split=split)
