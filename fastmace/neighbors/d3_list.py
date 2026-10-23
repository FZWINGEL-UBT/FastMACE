"""The D3 pair list: a half list within the D3 cutoff + its own skin, with no per-step compaction.

SPEC §11.2 "Own candidate list with its own skin" and "Periodic images"; plan task 3b-T02.

**What it stores.** Every undirected pair closer than ``cutoff + skin`` at the build, once: pairs
``i < j`` with every periodic image, plus each self-image pair ``(i, i, n)`` once (``n`` lexicographically
positive kept, ``-n`` dropped; the zero image is no pair). The images are every lattice translation in
range, from the cell widths, so a cell narrower than twice the cutoff is valid (no minimum-image
assumption). A D3 kernel iterates over the stored pairs, masks ``r^2 <= cutoff^2`` in registers and adds
each pair's contribution to both atoms (a self-image pair to the same atom twice). No pair moves between
builds.

**Layout** (12 bytes per stored pair; the kernel of 3b-T03 reads it, docs/design/d3.md):

- ``index``: int32 ``[2, capacity]``, ``index[0] = i <= index[1] = j``; sorted by (i, j, shift), so
  the pairs of atom i are one contiguous run (CSR offsets by i);
- ``image``: int32 ``[capacity]``, a row of ``shifts``;
- ``shifts``: float64 ``[shift_capacity, 3]``, integer lattice shifts in lexicographic order. A pair's
  shift absorbs the atoms' whole-cell offsets at the build, so its vector is
  ``positions[j] - positions[i] + shifts[image] @ cell`` with the unwrapped positions and the current
  cell, as ``SkinNeighbourList``'s unit shifts;
- ``shift_vectors``: float64 ``[shift_capacity, 3]``, ``shifts @ cell`` in the current cell (``set_cell``
  refreshes it; a kernel may use it or multiply itself);
- ``count``: int64 device scalar, the stored pairs. Unused capacity holds ``i = j = N`` (one past the
  last atom) and shift row 0.

``set_cell`` and ``needs_rebuild`` are graph-capturable (fixed shapes and addresses, no host
synchronisation). ``build`` runs on the host between replays; it moves buffers only when a capacity
grows or the atom count changes, and returns True then.

**The rebuild test** is ``SkinNeighbourList``'s with the D3 cutoff, against the list's own reference
positions and build cell: with ``F = h0^-1 h`` the deformation since the build, a pair farther than
``cutoff + skin`` at the build stays beyond the cutoff while
``2 max|y - y_build| <= cutoff + skin - cutoff / (1 - |F - I|_F)``, ``y = x h^-1 h0`` (the
displacement test ``max|x - x_build| > skin / 2`` for a fixed cell).

**Build.** The pair screen of ``SkinNeighbourList.build_gpu`` (``pair_search.candidate_pairs``: Triton on
CUDA, chunked PyTorch on the CPU, so the CPU tier builds the same list) restricted to ``sender <=
receiver``, a float64 decision, and a sort on a unique key, so the list is the same on every build at the
same positions, cell and capacities (a canonical restart point, SPEC §10.6, rebuilds it). Builds and their
wall time are counted (``counters``).
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping

import numpy as np
import torch

from .buckets import edge_capacity
from .pair_search import candidate_pairs
from .skin import inverse_3x3, times_3x3

# Stored-pair capacity above the first build (as ``NeighbourConfig.capacity_headroom``).
DEFAULT_HEADROOM = 1.15
# Granularity of the pair capacity, and of the shift table's.
CAPACITY_MULTIPLE = 1024
SHIFT_MULTIPLE = 64
# Bytes per stored pair: two int32 atom indices and an int32 shift row.
BYTES_PER_PAIR = 12
# Transient bytes per screened pair during a build: the int64 screen code, the decoded int64
# (image, sender, receiver), float64 vector and shift, keep mask, sort key and order (an upper estimate).
BUILD_BYTES_PER_CANDIDATE = 136


def image_grid(cell, radius):
    """Integer image shifts (M, 3) that can bring a pair of wrapped positions within ``radius``.

    Wrapped fractional differences lie in (-1, 1): along axis a, the image n can be within ``radius``
    only if |n| < radius / width_a + 1. The grid is symmetric and in lexicographic order, so row
    ``M - 1 - k`` is the negative of row ``k`` and the centre row ``(M - 1) / 2`` is the zero shift.
    """
    widths = 1.0 / np.linalg.norm(np.linalg.inv(np.asarray(cell, dtype=float)), axis=0)
    ranges = [np.arange(-int(np.floor(radius / w + 1.0)), int(np.floor(radius / w + 1.0)) + 1) for w in widths]
    return np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)


def estimated_pairs(natoms: int, volume_A3: float, radius_A: float) -> float:
    """Stored pairs of a half list within ``radius_A`` at uniform density: N^2 (4/3 pi r^3) / (2 V).

    Rule 12's estimate (SPEC §9.3, §11.2): NbCl5 3x3x3 (1944 atoms, 28.8 Å^3 per atom) has about
    1.4 million at 21.5 Å and 1.75 million at 23.2 Å.
    """
    return natoms * natoms * (4.0 / 3.0) * math.pi * radius_A ** 3 / (2.0 * volume_A3)


class D3PairList:
    """The D3 half list with its own skin and rebuild test (module docstring).

    ``cutoff``: the largest D3 cutoff in Å (pair and coordination-number cutoffs); ``skin``: Å, rule 11's
    D3 skin; ``variable_cell=True`` (NPT): the cell is the float64 device tensor ``cell_t``, updated in
    place by ``set_cell`` (or ``needs_rebuild(positions, cell)``).
    """

    def __init__(self, cell, cutoff: float, skin: float, device, *, variable_cell: bool = False,
                 headroom: float = DEFAULT_HEADROOM, capacity_multiple: int = CAPACITY_MULTIPLE):
        if not cutoff > 0 or not skin >= 0:
            raise ValueError(f"the D3 cutoff must be positive and the skin non-negative, got {cutoff!r}, {skin!r}")
        if headroom < 1:
            raise ValueError(f"headroom must be at least 1, got {headroom!r}")
        self.cell = np.asarray(cell, dtype=float)
        self.cutoff = float(cutoff)
        self.skin = float(skin)
        self.device = torch.device(device)
        self.variable_cell = bool(variable_cell)
        self.headroom = float(headroom)
        self.capacity_multiple = int(capacity_multiple)
        self.capacity = 0
        self.shift_capacity = 0
        self.num_pairs = 0
        self.num_shifts = 0
        self.num_images = 0  # the search grid of the last build
        self.natoms = None
        self.builds = 0
        self.build_time_s = 0.0
        self.last_build_time_s = 0.0
        self.cell_t = torch.tensor(self.cell, dtype=torch.float64, device=self.device)  # current cell
        self.build_cell = self.cell_t.clone()  # cell at the last build (h0)
        self.count = torch.zeros((), dtype=torch.int64, device=self.device)
        self.index = self.image = self.shifts = self.shift_vectors = self.reference_positions = None
        self._pair_search_state = {}

    # ------------------------------------------------------------------ host side

    @property
    def radius(self) -> float:
        return self.cutoff + self.skin

    def _allocate_pairs(self, capacity: int) -> None:
        self.capacity = max(int(capacity), 1)
        self.index = torch.zeros((2, self.capacity), dtype=torch.int32, device=self.device)
        self.image = torch.zeros(self.capacity, dtype=torch.int32, device=self.device)

    def _allocate_shifts(self, capacity: int) -> None:
        self.shift_capacity = max(int(capacity), 1)
        self.shifts = torch.zeros((self.shift_capacity, 3), dtype=torch.float64, device=self.device)
        self.shift_vectors = torch.zeros((self.shift_capacity, 3), dtype=torch.float64, device=self.device)

    def _pad(self, start: int) -> None:
        self.index[:, start:].fill_(0 if self.natoms is None else self.natoms)
        self.image[start:].zero_()

    def set_capacity(self, capacity: int, shift_capacity: int | None = None) -> bool:
        """Reallocate to exactly ``capacity`` stored pairs (and ``shift_capacity`` shift rows), a resumed run's.

        SPEC §10.6. True if a buffer moved. The pairs are padding until the next ``build``. A capacity below
        what the last build stored is refused.
        """
        capacity = int(capacity)
        if capacity < max(self.num_pairs, 1):
            raise ValueError(f"D3 pair capacity {capacity} is below the {self.num_pairs} pairs of the last build")
        if shift_capacity is not None and int(shift_capacity) < max(self.num_shifts, 1):
            raise ValueError(f"D3 shift capacity {shift_capacity} is below the {self.num_shifts} shifts of the "
                             "last build")
        moved = False
        if capacity != self.capacity:
            self._allocate_pairs(capacity)
            self._pad(0)
            self.num_pairs = 0
            self.count.zero_()
            moved = True
        if shift_capacity is not None and int(shift_capacity) != self.shift_capacity:
            self._allocate_shifts(int(shift_capacity))
            moved = True
        return moved

    def reset_cell(self, cell) -> None:
        """A new cell for the next ``build``, fixed or variable cell (host, between replays; 4-T05)."""
        self.cell = np.array(cell, dtype=float)
        self.cell_t.copy_(torch.as_tensor(self.cell, dtype=torch.float64, device=self.device))

    def build(self, positions, cell=None) -> bool:
        """Rebuild from (N, 3) float64 positions (NumPy or tensor) at the current cell (``cell``: set it first).

        Returns True if a buffer moved (a capacity grew, or the atom count changed), so captured graphs are
        stale.
        """
        start = time.perf_counter()
        if cell is not None:
            cell = torch.as_tensor(cell, dtype=torch.float64, device=self.device)
            if self.variable_cell:
                self.set_cell(cell)
            elif not torch.equal(cell, self.cell_t):
                raise ValueError("a fixed-cell D3 list was given a different cell; build it with variable_cell=True")
        positions = torch.as_tensor(positions, dtype=torch.float64, device=self.device)
        natoms = positions.shape[0]
        if self.variable_cell:
            self.cell = self.cell_t.cpu().numpy().copy()  # host sync (rebuild steps only)
        self.build_cell.copy_(self.cell_t)
        cell_t = self.build_cell
        grid = torch.as_tensor(image_grid(self.cell, self.radius), dtype=torch.float64, device=self.device)
        nimages = grid.shape[0]
        centre = (nimages - 1) // 2  # the zero shift
        inverse = torch.as_tensor(np.linalg.inv(self.cell), dtype=torch.float64, device=self.device)
        offsets = torch.floor(times_3x3(positions, inverse))  # whole cells of each atom
        wrapped = positions - times_3x3(offsets, cell_t)
        grid_vectors = times_3x3(grid, cell_t)
        radius = self.radius
        # float32 screen of sender <= receiver (one Triton kernel on CUDA), float64 decision
        image, sender, receiver = candidate_pairs(wrapped, grid_vectors, radius, self._pair_search_state, half=True)
        vectors = wrapped[receiver] - wrapped[sender] + grid_vectors[image]
        keep = ((vectors * vectors).sum(dim=1) < radius * radius) & ((sender < receiver) | (image > centre))
        # The screen returns pairs in scheduling order; a unique key makes the order deterministic.
        # Rejected pairs get the largest key and are cut off after the sort.
        key = (sender * natoms + receiver) * nimages + image
        key = torch.where(keep, key, torch.iinfo(torch.int64).max)
        n = int(keep.sum())  # host sync (rebuild steps only)
        order = torch.sort(key).indices[:n]
        sender, receiver, image = sender[order], receiver[order], image[order]
        # The pair's shift for the unwrapped positions; unique() sorts lexicographically, and for one (i, j)
        # the shift is the grid row plus a constant, so the order stays (i, j, shift).
        shifts = grid[image] - offsets[receiver] + offsets[sender]
        if n:
            table, rows = torch.unique(shifts, dim=0, return_inverse=True)
        else:
            table, rows = torch.zeros((1, 3), dtype=torch.float64, device=self.device), image
        moved = natoms != self.natoms
        self.natoms = natoms
        if n > self.capacity or self.capacity == 0:
            self._allocate_pairs(edge_capacity(n, self.headroom, self.capacity_multiple))
            moved = True
        if table.shape[0] > self.shift_capacity:
            self._allocate_shifts(edge_capacity(table.shape[0], self.headroom, SHIFT_MULTIPLE))
            moved = True
        # Fill in place: captured CUDA graphs read these addresses.
        self.index[0, :n].copy_(sender)
        self.index[1, :n].copy_(receiver)
        self.image[:n].copy_(rows)
        self._pad(n)
        self.shifts.zero_()
        self.shifts[:table.shape[0]].copy_(table)
        self.shift_vectors.copy_(times_3x3(self.shifts, cell_t))
        if self.reference_positions is None or self.reference_positions.shape[0] != natoms:
            self.reference_positions = torch.zeros((natoms, 3), dtype=torch.float64, device=self.device)
        self.reference_positions.copy_(positions)
        self.count.fill_(n)
        self.num_pairs = n
        self.num_shifts = int(table.shape[0])
        self.num_images = nimages
        self.builds += 1
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_build_time_s = time.perf_counter() - start
        self.build_time_s += self.last_build_time_s
        return moved

    def canonicalise(self, positions, cell=None) -> bool:
        """A canonical restart point (SPEC §10.6): rebuild from the current positions and cell.

        The result depends only on the positions, the cell and the capacities, so an interrupted and an
        uninterrupted run have the same list after it. Returns ``build``'s flag.
        """
        return self.build(positions, cell)

    def pairs(self):
        """Host ``(i, j, shifts)`` of the stored pairs: the vector of a pair is x[j] - x[i] + shifts @ cell.

        For tests and diagnostics (one host copy).
        """
        n = self.num_pairs
        i, j = self.index[0, :n].long(), self.index[1, :n].long()
        shifts = self.shifts[self.image[:n].long()]
        return i.cpu().numpy(), j.cpu().numpy(), shifts.round().long().cpu().numpy()

    def counters(self) -> dict:
        """Runtime counters (SPEC §9.6, §11.2): builds, their total and last wall time, pairs, capacities."""
        return {"d3_builds": self.builds, "d3_build_time_s": self.build_time_s,
                "d3_last_build_time_s": self.last_build_time_s, "d3_pairs": self.num_pairs,
                "d3_capacity": self.capacity, "d3_shifts": self.num_shifts,
                "d3_shift_capacity": self.shift_capacity, "d3_images": self.num_images, "d3_skin_A": self.skin}

    def buffers(self) -> Mapping[str, torch.Tensor]:
        """Persistent device buffers by name (for the runtime-state registry; reset at a canonical point).

        ``cell`` is derived from the state's cell; everything else is reset by a rebuild.
        """
        names = ["index", "image", "shifts", "shift_vectors", "reference_positions", "count", "build_cell"]
        out = {name: getattr(self, name) for name in names if getattr(self, name, None) is not None}
        if self.variable_cell:
            out["cell"] = self.cell_t
        return out

    # ------------------------------------------------------------------ trial lists (MC displacements)

    def save_list(self, into: dict | None = None) -> dict:
        """Copy the list (``buffers()``, counts, capacities) for a trial build that may be rejected.

        As ``SkinNeighbourList.save_list`` (SPEC §8.5 step 4): the trial build happens in place at the
        buffers' addresses; ``into``, a dict an earlier call returned, is reused when its shapes still match.
        Fixed cell only.
        """
        if self.variable_cell:
            raise RuntimeError("trial lists are for fixed-cell runs (displacements under a variable cell are refused)")
        saved = {} if into is None else into
        with torch.no_grad():
            for name, tensor in self.buffers().items():
                copy = saved.get(name)
                if copy is None or copy.shape != tensor.shape or copy.dtype != tensor.dtype:
                    saved[name] = tensor.detach().clone()
                else:
                    copy.copy_(tensor)
        saved.update(num_pairs=self.num_pairs, num_shifts=self.num_shifts, num_images=self.num_images,
                     capacity=self.capacity, shift_capacity=self.shift_capacity)
        return saved

    def restore_list(self, saved: dict) -> None:
        """Copy a ``save_list`` copy back in place; refused once a buffer moved since (rebuild instead)."""
        if (saved.get("capacity"), saved.get("shift_capacity")) != (self.capacity, self.shift_capacity):
            raise RuntimeError(f"the D3 list's buffers moved since it was saved (capacity {saved.get('capacity')} -> "
                               f"{self.capacity}); rebuild at the committed positions instead")
        with torch.no_grad():
            for name, tensor in self.buffers().items():
                tensor.copy_(saved[name])
        self.num_pairs, self.num_shifts, self.num_images = saved["num_pairs"], saved["num_shifts"], saved["num_images"]

    # ------------------------------------------------------------------ graph-capturable

    def set_cell(self, cell: torch.Tensor) -> None:
        """variable_cell: copy the current cell (float64 tensor) in place and refresh ``shift_vectors``; capturable."""
        if not self.variable_cell:
            raise RuntimeError("set_cell needs a D3PairList built with variable_cell=True")
        self.cell_t.copy_(cell)
        if self.shifts is not None:
            self.shift_vectors.copy_(times_3x3(self.shifts, self.cell_t))

    def needs_rebuild(self, positions: torch.Tensor, cell: torch.Tensor | None = None,
                      fraction: float = 1.0) -> torch.Tensor:
        """Device bool: can a pair beyond ``cutoff + skin`` at the build have come within the cutoff?

        The displacement test (fixed cell) or the deformation-aware test (variable cell) of the module
        docstring, with ``fraction`` scaling the allowed displacement. ``cell`` (variable cell): the current
        cell, taken in as ``set_cell`` does. No host synchronisation; capturable.
        """
        if cell is not None and self.variable_cell:
            self.set_cell(cell)
        if not self.variable_cell:
            squared = ((positions - self.reference_positions) ** 2).sum(dim=1)
            return squared.max() > (fraction * self.skin / 2) ** 2
        to_build = inverse_3x3(self.cell_t) @ self.build_cell  # F^-1: x -> build-cell coordinates
        squared = ((times_3x3(positions, to_build) - self.reference_positions) ** 2).sum(dim=1)
        strain = inverse_3x3(self.build_cell) @ self.cell_t - torch.eye(3, dtype=torch.float64,
                                                                        device=self.cell_t.device)
        lower = 1.0 - torch.sqrt((strain * strain).sum())  # <= smallest singular value of F
        allowed = self.cutoff + self.skin - self.cutoff / torch.clamp(lower, min=1e-6)
        return (2 * torch.sqrt(squared.max()) > fraction * allowed) | (lower <= 0)

    def inside(self, positions: torch.Tensor) -> torch.Tensor:
        """Bool mask over the capacity: stored pairs within the cutoff at ``positions`` and the current cell.

        What the kernel's in-register mask computes, with the padding (``i = j = N``) masked by ``count``;
        for tests (the kernel never materialises it).
        """
        stored = torch.arange(self.capacity, device=self.device) < self.count
        last = positions.shape[0] - 1
        i = torch.clamp(self.index[0].long(), max=last)
        j = torch.clamp(self.index[1].long(), max=last)
        vectors = positions[j] - positions[i] + self.shift_vectors[self.image.long()]
        return stored & ((vectors * vectors).sum(dim=1) <= self.cutoff ** 2)
