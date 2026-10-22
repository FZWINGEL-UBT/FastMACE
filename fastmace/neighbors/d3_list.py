"""The D3 pair list: a half list within the D3 cutoff + its own skin, with no per-step compaction.

SPEC §11.2 "Own candidate list with its own skin" and "Periodic images"; plan task 3b-T02.

**What it stores.** Every undirected pair closer than ``cutoff + skin`` at the build, once: pairs
``i < j`` with every periodic image, plus each self-image pair ``(i, i, n)`` once (``n`` kept, ``-n``
dropped; the zero image is no pair). The images are every lattice translation within range, from the
cell widths, so a cell narrower than twice the cutoff is valid (no minimum-image assumption). A D3
kernel iterates over the stored pairs, masks ``r^2 <= cutoff^2`` in registers and adds each pair's
contribution to both atoms (a self-image pair to the same atom twice). No pair is ever moved between
builds.

**Layout** (12 bytes per stored pair; the kernel of 3b-T03 reads it):

- ``index``: int32 ``[2, capacity]``, ``index[0] <= index[1]``;
- ``image``: int32 ``[capacity]``, a row of ``image_vectors``;
- ``image_vectors``: float64 ``[M + 1, 3]``: the image translations ``images @ cell`` in the current
  cell, and a last, padding row ``PAD_DISTANCE_A`` away from every image;
- ``shifted_positions``: float64 ``[N, 3]``: the positions minus each atom's whole-cell offset at the
  build (``atom_offsets @ cell``), so that the vector of a pair is
  ``shifted_positions[j] - shifted_positions[i] + image_vectors[image]``;
- ``count``: int64 device scalar, the stored pairs. Unused capacity holds the pair ``(0, 0)`` with the
  padding image, which is never within the cutoff.

``prepare(positions)`` writes ``shifted_positions`` and (variable cell) ``image_vectors``; it, ``set_cell``
and ``needs_rebuild`` are graph-capturable (fixed shapes and addresses, no host synchronisation).
``build`` runs on the host between replays; it moves the buffers only when the capacity or the image
count grows (it returns True then).

**The rebuild test** is ``SkinNeighbourList``'s with the D3 cutoff, against the list's own reference
positions and build cell: with ``F = h0^-1 h`` the deformation since the build, a pair farther than
``cutoff + skin`` at the build stays beyond the cutoff while
``2 max|y - y_build| <= cutoff + skin - cutoff / (1 - |F - I|_F)``, ``y = x h^-1 h0`` (the
displacement test ``max|x - x_build| > skin / 2`` for a fixed cell).

**Build.** The pair screen of ``SkinNeighbourList.build_gpu`` (``pair_search.candidate_pairs``, Triton
on CUDA, chunked PyTorch on the CPU) restricted to ``sender <= receiver``, a float64 decision, and a
sort on a unique key ``(i, j, image)``, so the list is the same on every build at the same positions,
cell and capacity (a canonical restart point, SPEC §10.6, rebuilds it). Builds and their wall time are
counted (``counters``).
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

# The padding image's distance, Å: far beyond any D3 cutoff, still exact in float32.
PAD_DISTANCE_A = 1.0e4
# Stored-pair capacity above the first build (as ``NeighbourConfig.capacity_headroom``).
DEFAULT_HEADROOM = 1.15
# Granularity of the capacity.
CAPACITY_MULTIPLE = 1024
# Bytes per stored pair: two int32 atom indices and an int32 image row.
BYTES_PER_PAIR = 12
# Transient bytes per screened pair during a build: the int64 screen code, the decoded int64
# (image, sender, receiver), float64 vector, keep mask and sort key and order (an upper estimate).
BUILD_BYTES_PER_CANDIDATE = 112


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
    place by ``set_cell``; the image vectors follow it in ``prepare``.
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
        self.num_pairs = 0
        self.builds = 0
        self.build_time_s = 0.0
        self.last_build_time_s = 0.0
        self.cell_t = torch.tensor(self.cell, dtype=torch.float64, device=self.device)  # current cell
        self.build_cell = self.cell_t.clone()  # cell at the last build (h0)
        self.count = torch.zeros((), dtype=torch.int64, device=self.device)
        self.images = None  # (M + 1, 3) float64 integer shifts, the last row the padding image
        self.image_vectors = None
        self.atom_offsets = None  # (N, 3) float64 whole-cell offsets at the build
        self.shifted_positions = None
        self.reference_positions = None
        self._pair_search_state = {}

    # ------------------------------------------------------------------ host side

    @property
    def radius(self) -> float:
        return self.cutoff + self.skin

    @property
    def num_images(self) -> int:
        """Image rows without the padding row (M)."""
        return 0 if self.images is None else self.images.shape[0] - 1

    def _allocate(self, needed: int, exact: bool = False) -> None:
        capacity = needed if exact else edge_capacity(needed, self.headroom, self.capacity_multiple)
        self.capacity = max(int(capacity), 1)
        self.index = torch.zeros((2, self.capacity), dtype=torch.int32, device=self.device)
        self.image = torch.zeros(self.capacity, dtype=torch.int32, device=self.device)

    def set_capacity(self, capacity: int) -> bool:
        """Reallocate to exactly ``capacity`` stored pairs (a resumed run's, SPEC §10.6). True if they moved.

        The buffers hold padding until the next ``build``. A capacity below the pairs of the last build
        is refused.
        """
        capacity = int(capacity)
        if capacity < max(self.num_pairs, 1):
            raise ValueError(f"D3 pair capacity {capacity} is below the {self.num_pairs} pairs of the last build")
        if capacity == self.capacity:
            return False
        self._allocate(capacity, exact=True)
        self._pad(0)
        return True

    def _pad(self, start: int) -> None:
        self.index[:, start:].zero_()
        if self.images is not None:
            self.image[start:].fill_(self.images.shape[0] - 1)

    def _set_images(self, cell: np.ndarray) -> bool:
        """The image grid of ``cell`` at the list radius; True if the image buffers moved."""
        grid = image_grid(cell, self.radius)
        rows = np.concatenate([grid, np.zeros((1, 3))]).astype(np.float64)
        moved = self.images is None or self.images.shape[0] != rows.shape[0]
        if moved:
            self.images = torch.zeros(rows.shape, dtype=torch.float64, device=self.device)
            self.image_vectors = torch.zeros(rows.shape, dtype=torch.float64, device=self.device)
        self.images.copy_(torch.from_numpy(rows))
        self._image_vectors(self.build_cell)
        return moved

    def _image_vectors(self, cell: torch.Tensor) -> None:
        vectors = times_3x3(self.images[:-1], cell)
        self.image_vectors[:-1].copy_(vectors)
        self.image_vectors[-1].zero_()
        self.image_vectors[-1, 0] = PAD_DISTANCE_A

    def build(self, positions) -> bool:
        """Rebuild from (N, 3) float64 positions (NumPy or tensor) at the current cell.

        Returns True if a buffer moved (capacity or image count grew), so captured graphs are stale.
        """
        start = time.perf_counter()
        positions = torch.as_tensor(positions, dtype=torch.float64, device=self.device)
        natoms = positions.shape[0]
        if self.variable_cell:
            self.cell = self.cell_t.cpu().numpy().copy()  # host sync (rebuild steps only)
        self.build_cell.copy_(self.cell_t)
        moved = self._set_images(self.cell)
        cell = self.build_cell
        inverse = torch.as_tensor(np.linalg.inv(self.cell), dtype=torch.float64, device=self.device)
        offsets = torch.floor(times_3x3(positions, inverse))  # whole cells of each atom
        wrapped = positions - times_3x3(offsets, cell)
        images, image_vectors = self.images[:-1], self.image_vectors[:-1]
        nimages = images.shape[0]
        centre = (nimages - 1) // 2  # the zero shift
        radius = self.radius
        # float32 screen of sender <= receiver (one Triton kernel on CUDA), float64 decision
        image, sender, receiver = candidate_pairs(wrapped, image_vectors, radius, self._pair_search_state, half=True)
        vectors = wrapped[receiver] - wrapped[sender] + image_vectors[image]
        keep = ((vectors * vectors).sum(dim=1) < radius * radius) & ((sender < receiver) | (image > centre))
        # The screen returns pairs in scheduling order; a unique key makes the order deterministic.
        # Rejected pairs get the largest key and are cut off after the sort.
        key = (sender * natoms + receiver) * nimages + image
        key = torch.where(keep, key, torch.iinfo(torch.int64).max)
        n = int(keep.sum())  # host sync (rebuild steps only)
        order = torch.sort(key).indices[:n]
        if n > self.capacity or self.capacity == 0:
            self._allocate(n)
            moved = True
        # Fill in place: captured CUDA graphs read these addresses.
        self.index[0, :n].copy_(sender[order])
        self.index[1, :n].copy_(receiver[order])
        self.image[:n].copy_(image[order])
        self._pad(n)
        if self.atom_offsets is None or self.atom_offsets.shape[0] != natoms:
            self.atom_offsets = torch.zeros((natoms, 3), dtype=torch.float64, device=self.device)
            self.shifted_positions = torch.zeros((natoms, 3), dtype=torch.float64, device=self.device)
            self.reference_positions = torch.zeros((natoms, 3), dtype=torch.float64, device=self.device)
            moved = True
        self.atom_offsets.copy_(offsets)
        self.reference_positions.copy_(positions)
        self.shifted_positions.copy_(wrapped)
        self.count.fill_(n)
        self.num_pairs = n
        self.builds += 1
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_build_time_s = time.perf_counter() - start
        self.build_time_s += self.last_build_time_s
        return moved

    def canonicalise(self, positions, cell=None) -> bool:
        """A canonical restart point (SPEC §10.6): rebuild from the current positions and cell, then prepare.

        The result depends only on the positions, the cell and the capacity, so an interrupted and an
        uninterrupted run have the same list after it. Returns ``build``'s flag.
        """
        if cell is not None:
            self.set_cell(torch.as_tensor(cell, dtype=torch.float64, device=self.device))
        moved = self.build(positions)
        self.prepare(torch.as_tensor(positions, dtype=torch.float64, device=self.device))
        return moved

    def pairs(self):
        """Host ``(i, j, unit_shifts)`` of the stored pairs: the vector of a pair is x[j] - x[i] + shifts @ cell.

        For tests and diagnostics (one host copy).
        """
        n = self.num_pairs
        i = self.index[0, :n].long()
        j = self.index[1, :n].long()
        shifts = self.images[self.image[:n].long()] - self.atom_offsets[j] + self.atom_offsets[i]
        return i.cpu().numpy(), j.cpu().numpy(), shifts.round().long().cpu().numpy()

    def counters(self) -> dict:
        """Runtime counters (SPEC §9.6, §11.2): builds, their total and last wall time, pairs, capacity, images."""
        return {"d3_builds": self.builds, "d3_build_time_s": self.build_time_s,
                "d3_last_build_time_s": self.last_build_time_s, "d3_pairs": self.num_pairs,
                "d3_capacity": self.capacity, "d3_images": self.num_images, "d3_skin_A": self.skin}

    def buffers(self) -> Mapping[str, torch.Tensor]:
        """Persistent device buffers by name (for the runtime-state registry; reset at a canonical point).

        ``cell`` is derived from the state's cell; everything else is reset by a rebuild.
        """
        names = ["index", "image", "images", "image_vectors", "atom_offsets", "shifted_positions",
                 "reference_positions", "count", "build_cell"]
        out = {name: getattr(self, name) for name in names if getattr(self, name, None) is not None}
        if self.variable_cell:
            out["cell"] = self.cell_t
        return out

    # ------------------------------------------------------------------ graph-capturable

    def set_cell(self, cell: torch.Tensor) -> None:
        """variable_cell: copy the current cell (float64 tensor) in place; CUDA-graph safe."""
        if not self.variable_cell:
            raise RuntimeError("set_cell needs a D3PairList built with variable_cell=True")
        self.cell_t.copy_(cell)

    def prepare(self, positions: torch.Tensor) -> None:
        """Write ``shifted_positions`` (and the image vectors of the current cell) for the kernel; capturable."""
        if self.variable_cell:
            self._image_vectors(self.cell_t)
            self.shifted_positions.copy_(positions - times_3x3(self.atom_offsets, self.cell_t))
        else:
            self.shifted_positions.copy_(positions - times_3x3(self.atom_offsets, self.build_cell))

    def needs_rebuild(self, positions: torch.Tensor, fraction: float = 1.0) -> torch.Tensor:
        """Device bool: can a pair beyond ``cutoff + skin`` at the build have come within the cutoff?

        The displacement test (fixed cell) or the deformation-aware test (variable cell) of the module
        docstring, with ``fraction`` scaling the allowed displacement.
        """
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

    def inside(self) -> torch.Tensor:
        """Bool mask over the capacity: stored pairs within the cutoff at the prepared positions.

        What the kernel's in-register mask computes; for tests (the kernel never materialises it).
        """
        i, j = self.index[0].long(), self.index[1].long()
        vectors = self.shifted_positions[j] - self.shifted_positions[i] + self.image_vectors[self.image.long()]
        return (vectors * vectors).sum(dim=1) <= self.cutoff ** 2
