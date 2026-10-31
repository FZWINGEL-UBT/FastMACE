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

MIN_BINS = 3  # bins per axis: a pair then has one image within the cutoff, and +-1 bin covers it


class TileNeighbours:
    """Cell bins of the current positions and each tile's edges within ``cutoff``.

    ``cell`` (3, 3) float64; ``float_dtype``: the list's (the model's) float dtype; ``sender_groups`` (N,) int64
    and ``group_count``: the segmented first sum's sender groups, as the global list sorts them.
    """

    def __init__(self, cutoff, cell, float_dtype, sender_groups=None, group_count=1, device=None):
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
        d = torch.arange(-1, 2, device=self.device)
        self._neighbour_bins = torch.stack(torch.meshgrid(d, d, d, indexing="ij"), dim=-1).reshape(-1, 3)

    def bin(self, positions):
        """Bin float64 ``positions`` (N, 3) by their wrapped fractional coordinates."""
        k = torch.tensor(self.bins, device=self.device)
        self.positions = positions
        frac = times_3x3(positions, self.inverse)
        self.offsets = torch.floor(frac)  # integer cell of each atom
        wrapped = frac - self.offsets
        b = torch.minimum(torch.clamp((wrapped * k).long(), min=0), k - 1)
        self.atom_bin = b
        flat = (b[:, 0] * k[1] + b[:, 1]) * k[2] + b[:, 2]
        self.order = torch.sort(flat, stable=True).indices
        nbins = int(k.prod())
        self.starts = torch.searchsorted(flat[self.order], torch.arange(nbins + 1, device=self.device))
        self._k = k

    def edges(self, receivers):
        """(sender, receiver, unit_shifts, shifts) of the edges into ``receivers`` (sorted int64 atom indices),
        in the global list's order; unit shifts and shifts in the float dtype."""
        k = self._k
        natoms = self.positions.shape[0]
        nb = self.atom_bin[receivers][:, None, :] + self._neighbour_bins[None, :, :]  # (R, 27, 3)
        t = torch.div(nb, k, rounding_mode="floor")  # the image of the neighbour bin
        nb = nb - t * k
        ids = ((nb[..., 0] * k[1] + nb[..., 1]) * k[2] + nb[..., 2]).reshape(-1)
        counts = self.starts[ids + 1] - self.starts[ids]
        total = int(counts.sum())
        slot = torch.repeat_interleave(torch.arange(ids.shape[0], device=self.device), counts, output_size=total)
        first = torch.cumsum(counts, 0) - counts
        sender = self.order[self.starts[ids][slot] + torch.arange(total, device=self.device) - first[slot]]
        receiver = receivers.repeat_interleave(27)[slot]
        image = t.reshape(-1, 3)[slot].to(torch.float64)
        # The sender's image at its bin's wrap: x_r - x_s + S cell with S = off_s - off_r - t.
        unit = (self.offsets[sender] - self.offsets[receiver] - image).to(self.float_dtype)
        shifts64 = times_3x3(unit.to(torch.float64), self.cell)
        vectors = self.positions[receiver] - self.positions[sender] + shifts64
        keep = ((vectors * vectors).sum(dim=1) < self.cutoff ** 2) & ~(
            (sender == receiver) & (unit == 0).all(dim=1))
        del vectors
        sender, receiver, unit, shifts64 = sender[keep], receiver[keep], unit[keep], shifts64[keep]
        primary = receiver if self.groups is None else receiver * self.group_count + self.groups[sender]
        order = torch.sort(primary * natoms + sender).indices
        return sender[order], receiver[order], unit[order], shifts64[order].to(self.float_dtype)

    @classmethod
    def like(cls, neighbours, cell=None):
        """A ``TileNeighbours`` with a ``SkinNeighbourList``'s cutoff, float dtype, sender groups and current cell."""
        if cell is None:
            cell = neighbours.cell_t if neighbours.variable_cell else neighbours.build_cell
        groups = neighbours.sender_groups if neighbours.sort_receivers else None
        return cls(neighbours.cutoff, cell, neighbours.float_dtype, groups,
                   neighbours.group_count if groups is not None else 1, device=neighbours.device)
