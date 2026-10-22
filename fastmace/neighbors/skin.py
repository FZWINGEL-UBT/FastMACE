"""Skin neighbour list: candidate pairs within cutoff + skin, compacted to the cutoff on the device.

SPEC §8.1 "Neighbour list". The force field owns one instance (docs/design/interfaces.md):
``needs_rebuild``, ``compact`` and ``set_cell`` are graph-capturable (fixed shapes and
addresses, no host synchronisation); ``build`` runs on the host between replays.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch

from ..config import DEFAULT_SKIN_A, NeighbourConfig
from .buckets import edge_capacity
from .pair_search import candidate_pairs


def times_3x3(rows, matrix):
    """rows @ matrix for [n, 3] rows and a 3 x 3 matrix, as element-wise sums (no FP64 GEMM)."""
    return rows[:, 0:1] * matrix[0] + rows[:, 1:2] * matrix[1] + rows[:, 2:3] * matrix[2]


def det_3x3(m):
    """Determinant of a 3 x 3 tensor from element-wise products (no LAPACK call, CUDA-graph safe)."""
    return (m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
            - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
            + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]))


def inverse_3x3(m):
    """Inverse of a 3 x 3 tensor by cofactors (no LAPACK call or host check, CUDA-graph safe)."""
    cofactors = torch.stack([
        torch.stack([m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1], m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2],
                     m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]]),
        torch.stack([m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2], m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0],
                     m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2]]),
        torch.stack([m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0], m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1],
                     m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]])])
    return cofactors / det_3x3(m)


def _resolved(value, config_value, name):
    """A resolved bool from an explicit argument or an "on"/"off" config field ("auto" is the planner's)."""
    if value is not None:
        return bool(value)
    if config_value == "auto":
        raise ValueError(f"NeighbourConfig.{name} is 'auto'; the planner resolves it, so pass {name}= explicitly")
    return config_value == "on"


class SkinNeighbourList:
    """Periodic pairs within cutoff + skin, compacted to the cutoff on the device.

    ``build`` stores every pair closer than cutoff + skin. That set contains every pair closer
    than the cutoff for as long as no atom has moved more than skin / 2 since the build.
    ``compact`` then reorders the pairs so that those within the cutoff come first and stores
    their number in ``count``. The edges after ``count`` are farther than the cutoff: MACE
    multiplies every edge by a radial envelope that is exactly zero there (the ZBL term too), so
    including some of them changes nothing but work (``padding_is_exact``). Unused capacity
    holds atom 0 and its own periodic image more than 2 (cutoff + skin) away. Tensor addresses
    stay fixed between rebuilds unless the capacity has to grow.

    ``sort_receivers=True`` orders the candidates by receiver at every build (stably, so the
    matscipy order is kept within a receiver; then by ``sender_groups`` when given, the sender
    species of the segmented first-interaction sum). The default keeps matscipy's order.
    ``gpu_build=True`` builds on the device (``build_gpu``) when it is a CUDA device; otherwise
    ``build`` uses matscipy on the host. Both find the same pairs; only their order within a
    receiver differs.

    ``variable_cell=True`` (NPT): the cell is the float64 device tensor ``cell_t``, which the
    integrator updates in place (``set_cell``, graph safe). Pairs keep their integer image
    shifts; ``compact`` converts them with the current cell. With h0 the cell at the last build
    and F = h0^-1 h the deformation since, the distance of a pair is at least sigma_min(F)
    times its distance in build-cell coordinates y = x h^-1 h0, and
    sigma_min(F) >= 1 - |F - I|_F. A pair farther than cutoff + skin at the build therefore
    stays beyond the cutoff while 2 max|y - y_build| <= cutoff + skin - cutoff / (1 - |F - I|_F),
    which is the rebuild test (skin, as for the fixed cell, when F = I).
    """

    def __init__(self, cell, cutoff, device, float_dtype, config: NeighbourConfig | None = None, *,
                 sort_receivers: bool | None = None, gpu_build: bool | None = None, sender_groups=None,
                 group_count: int | None = None, variable_cell: bool = False):
        config = NeighbourConfig() if config is None else config
        self.config = config
        self.cell = np.asarray(cell, dtype=float)
        self.cutoff = float(cutoff)
        # A planned config has a number; "auto" here (a list built without a plan) is rule 11's floor.
        self.skin = DEFAULT_SKIN_A if config.skin == "auto" else float(config.skin)
        self.device = torch.device(device)
        self.float_dtype = float_dtype
        self.edge_multiple = config.edge_multiple
        self.headroom = config.capacity_headroom
        self.sort_receivers = _resolved(sort_receivers, config.sort_receivers, "sort_receivers")
        self.gpu_build = _resolved(gpu_build, config.gpu_neighbour_list, "gpu_neighbour_list")
        # Optional per-atom group (sender species, segmented first sum): secondary sort key. A device
        # tensor at a fixed address that its owner refreshes in place (a derived buffer of the species,
        # SPEC §8.5), read on every build; ``group_count`` is fixed (the superset size), max() + 1 if
        # not given.
        self.sender_groups = (None if sender_groups is None else
                              torch.as_tensor(sender_groups, dtype=torch.int64, device=self.device))
        self.group_count = (1 if self.sender_groups is None else
                            int(group_count) if group_count is not None else int(self.sender_groups.max()) + 1)
        self.capacity = 0
        self.num_candidates = 0
        self.rebuilds = 0
        self.variable_cell = bool(variable_cell)
        if self.variable_cell:
            self.cell_t = torch.tensor(self.cell, dtype=torch.float64, device=self.device)  # current cell
            self.build_cell = self.cell_t.clone()  # cell at the last build (h0)
        self._set_padding()
        self._device_constants = None  # build_gpu's constants of the (build) cell
        self._pair_search_state = {}
        self.reference_positions = None
        self.count = torch.zeros((), dtype=torch.int64, device=self.device)
        # The stream that reads the edge buffers, set while a build runs on another stream beside it
        # (GraphEngine's overlap_rebuild, OPT2-T04): a growth then waits for it and allocates there, so
        # the old buffers are not freed while a force call in flight still reads them.
        self.reader_stream = None

    # ------------------------------------------------------------------ host side

    def _set_padding(self):
        # Padding pair: atom 0 and its image n cells along the longest lattice vector.
        lengths = np.linalg.norm(self.cell, axis=1)
        axis = int(np.argmax(lengths))
        self._pad_unit_shift = np.zeros(3)
        self._pad_unit_shift[axis] = math.ceil(2 * (self.cutoff + self.skin) / lengths[axis]) + 1

    def _take_current_cell(self):
        """variable_cell: rebuild against the current cell (host copy; rebuild steps only)."""
        self.cell = self.cell_t.cpu().numpy().copy()
        self._set_padding()
        self.build_cell.copy_(self.cell_t)

    def _allocate(self, needed, exact=False):
        if self.reader_stream is not None:
            self.reader_stream.synchronize()  # the force call in flight reads the old buffers
            with torch.cuda.stream(self.reader_stream):
                return self._allocate_now(needed, exact)
        return self._allocate_now(needed, exact)

    def _allocate_now(self, needed, exact=False):
        capacity = needed if exact else edge_capacity(needed, self.headroom, self.edge_multiple)
        self.capacity = capacity

        def new(*shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=self.device)

        # Candidates from the last build ...
        self.candidate_index = new(2, capacity, dtype=torch.int64)
        self.candidate_unit_shifts = new(capacity, 3, dtype=self.float_dtype)
        self.candidate_shifts = new(capacity, 3, dtype=self.float_dtype)
        self.candidate_shifts64 = new(capacity, 3, dtype=torch.float64)  # for the cutoff test
        # ... and the same pairs reordered by compact(); this is what the model reads.
        self.edge_index = new(2, capacity, dtype=torch.int64)
        self.unit_shifts = new(capacity, 3, dtype=self.float_dtype)
        self.shifts = new(capacity, 3, dtype=self.float_dtype)

    def set_capacity(self, capacity: int) -> bool:
        """Reallocate the edge buffers to exactly ``capacity`` (a resumed run's, SPEC §10.6). True if they moved.

        The buffers are zero until the next ``build`` fills them, so call ``build`` (and ``compact``)
        before the next force call. A capacity below the candidates of the last build is refused.
        """
        capacity = int(capacity)
        if capacity < max(self.num_candidates, 1):
            raise ValueError(f"neighbour capacity {capacity} is below the {self.num_candidates} candidates of the "
                             "last build")
        if capacity == self.capacity:
            return False
        self._allocate(capacity, exact=True)
        return True

    def build(self, positions) -> bool:
        """Rebuild from (N, 3) float64 positions (NumPy or tensor). Returns True if the capacity grew.

        On a CUDA device with ``gpu_build`` the pairs are found on the device (``build_gpu``),
        otherwise with matscipy on the host (``build_cpu``). A grown capacity moves every
        buffer, so captured graphs are stale.
        """
        if self.gpu_build and self.device.type == "cuda":
            return self.build_gpu(torch.as_tensor(positions, dtype=torch.float64, device=self.device))
        return self.build_cpu(positions)

    def build_cpu(self, positions) -> bool:
        """Rebuild with matscipy from (N, 3) float64 positions. Returns True if the capacity grew."""
        from matscipy.neighbours import neighbour_list

        if torch.is_tensor(positions):
            positions = positions.detach().cpu().numpy()
        positions = np.asarray(positions, dtype=np.float64)
        if self.variable_cell:
            self._take_current_cell()
        # Same call and self-edge rule as mace.data.neighborhood.get_neighborhood.
        sender, receiver, unit_shifts = neighbour_list(
            quantities="ijS", pbc=(True, True, True), cell=self.cell,
            positions=positions, cutoff=self.cutoff + self.skin)
        keep = ~((sender == receiver) & np.all(unit_shifts == 0, axis=1))
        sender, receiver, unit_shifts = sender[keep], receiver[keep], unit_shifts[keep]

        # Keep the complete edge record together while sorting. Stable order within each
        # receiver preserves matscipy's deterministic candidate order and gives compaction a
        # reproducible order after its cutoff partition. Padding is appended below, after
        # this permutation.
        if self.sort_receivers and len(receiver) > 1:
            if self.sender_groups is not None:  # by receiver, then sender group (stable)
                order = np.lexsort((self.sender_groups.cpu().numpy()[sender], receiver))
            else:
                order = np.argsort(receiver, kind="stable")
            sender = sender[order]
            receiver = receiver[order]
            unit_shifts = unit_shifts[order]

        n = len(sender)
        grew = n > self.capacity or self.capacity == 0
        if grew:
            self._allocate(n)

        index = np.zeros((2, self.capacity), dtype=np.int64)
        index[0, :n], index[1, :n] = sender, receiver
        all_unit_shifts = np.tile(self._pad_unit_shift, (self.capacity, 1))
        all_unit_shifts[:n] = unit_shifts
        candidate_shifts = np.asarray(unit_shifts) @ self.cell
        shifts = np.tile(self._pad_unit_shift @ self.cell, (self.capacity, 1))
        shifts[:n] = candidate_shifts
        # copy_ keeps the tensor addresses that captured CUDA graphs read.
        self.candidate_index.copy_(torch.from_numpy(index))
        self.candidate_unit_shifts.copy_(torch.from_numpy(all_unit_shifts))
        self.candidate_shifts.copy_(torch.from_numpy(shifts))
        self.candidate_shifts64.copy_(torch.from_numpy(shifts))
        self._set_reference(torch.from_numpy(positions))
        self.num_candidates = n
        self.rebuilds += 1
        return grew

    def _constants(self, device):
        """Image shifts, cell and padding on the device; recomputed per build for a variable cell."""
        if self._device_constants is not None and not self.variable_cell:
            return self._device_constants
        r = self.cutoff + self.skin
        # Wrapped fractional differences lie in (-1, 1): along axis a, the image n can be within
        # r only if |n| < r / width_a + 1.
        widths = 1.0 / np.linalg.norm(np.linalg.inv(self.cell), axis=0)
        ranges = [np.arange(-int(np.floor(r / w + 1.0)), int(np.floor(r / w + 1.0)) + 1) for w in widths]
        grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
        images = torch.as_tensor(grid, dtype=torch.float64, device=device)
        cell = torch.as_tensor(self.cell, dtype=torch.float64, device=device)
        pad_unit = torch.as_tensor(self._pad_unit_shift, dtype=torch.float64, device=device)
        self._device_constants = {
            "images": images,
            "groups": None if self.sender_groups is None else self.sender_groups.to(device),
            "group_count": self.group_count,
            "cell": cell,
            "inverse": torch.as_tensor(np.linalg.inv(self.cell), dtype=torch.float64, device=device),
            "image_vectors": times_3x3(images, cell),
            "pad_unit": pad_unit,
            "pad_shift": times_3x3(pad_unit[None], cell)[0],
        }
        return self._device_constants

    def build_gpu(self, positions: torch.Tensor) -> bool:
        """``build`` on the positions' device from (N, 3) float64 positions. Returns True if the capacity grew.

        Same pairs as matscipy (distance below cutoff + skin, no self-pair with zero shift), for
        positions anywhere: positions are wrapped into the cell, every image shift that can
        bring a wrapped pair within range is tried, and the shifts are converted back for the
        unwrapped positions. Pairs are screened in float32 with a 0.01 Å margin, then decided
        in float64. Sorting by receiver (and sender group) orders each receiver's pairs by
        sender, then image shift. On a CPU tensor the screen runs in chunked PyTorch.
        """
        device = positions.device
        r = self.cutoff + self.skin
        natoms = positions.shape[0]
        if self.variable_cell:
            self._take_current_cell()
        constants = self._constants(device)
        cell, inverse = constants["cell"], constants["inverse"]
        images, image_vectors = constants["images"], constants["image_vectors"]
        offsets = torch.floor(times_3x3(positions, inverse))  # integer cell of each atom
        wrapped = positions - times_3x3(offsets, cell)
        # float32 screen (pair_search; one Triton kernel on CUDA), float64 decision below
        image, sender, receiver = candidate_pairs(wrapped, image_vectors, r, self._pair_search_state)
        vectors = wrapped[receiver] - wrapped[sender] + image_vectors[image]
        unit_shifts = images[image] - offsets[receiver] + offsets[sender]
        keep = ((vectors * vectors).sum(dim=1) < r * r) & ~(
            (sender == receiver) & (unit_shifts == 0).all(dim=1))
        # The screen returns pairs in scheduling order; a unique key makes the order
        # deterministic. Rejected pairs get the largest key and are cut off after the sort.
        nimages = len(images)
        if self.sort_receivers:
            primary = receiver
            if constants["groups"] is not None:  # receiver, then sender group
                primary = receiver * constants["group_count"] + constants["groups"][sender]
            key = (primary * natoms + sender) * nimages + image
        else:
            key = (image * natoms + sender) * natoms + receiver
        key = torch.where(keep, key, torch.iinfo(torch.int64).max)
        n = int(keep.sum())  # host sync (rebuild steps only)
        order = torch.sort(key).indices[:n]
        sender, receiver, unit_shifts = sender[order], receiver[order], unit_shifts[order]

        grew = n > self.capacity or self.capacity == 0
        if grew:
            self._allocate(n)
        # Fill in place: captured CUDA graphs read these addresses.
        pad_unit, pad_shift = constants["pad_unit"], constants["pad_shift"]
        shifts = times_3x3(unit_shifts, cell)
        self.candidate_index[0, :n].copy_(sender)
        self.candidate_index[1, :n].copy_(receiver)
        self.candidate_index[:, n:].zero_()
        self.candidate_unit_shifts[:n].copy_(unit_shifts)
        self.candidate_unit_shifts[n:].copy_(pad_unit)
        self.candidate_shifts[:n].copy_(shifts)
        self.candidate_shifts[n:].copy_(pad_shift)
        self.candidate_shifts64[:n].copy_(shifts)
        self.candidate_shifts64[n:].copy_(pad_shift)
        self._set_reference(positions)
        self.num_candidates = n
        self.rebuilds += 1
        return grew

    def _set_reference(self, positions):
        if self.reference_positions is None:
            self.reference_positions = torch.zeros(positions.shape, dtype=torch.float64, device=self.device)
        self.reference_positions.copy_(positions)

    def edges(self, num_edges: int):
        """Views of the first ``num_edges`` compacted edges: (edge_index, shifts, unit_shifts); no copy."""
        return self.edge_index[:, :num_edges], self.shifts[:num_edges], self.unit_shifts[:num_edges]

    def buffers(self) -> Mapping[str, torch.Tensor]:
        """Persistent device buffers that build, compact and set_cell write, by name.

        The addresses change only when ``build`` returns True. The engine registers them as
        ``force_field.neighbours.<name>`` (``fastmace.engine.registry.POLICIES``): ``cell`` is
        derived from the state's cell, everything else is reset by a rebuild.
        """
        names = ["candidate_index", "candidate_unit_shifts", "candidate_shifts", "candidate_shifts64",
                 "edge_index", "unit_shifts", "shifts", "count", "reference_positions"]
        out = {name: getattr(self, name) for name in names if getattr(self, name, None) is not None}
        if self.variable_cell:
            out["cell"] = self.cell_t
            out["build_cell"] = self.build_cell
        return out

    # ------------------------------------------------------------------ graph-capturable

    def set_cell(self, cell: torch.Tensor) -> None:
        """variable_cell: copy the current cell (float64 tensor) in place; CUDA-graph safe."""
        if not self.variable_cell:
            raise RuntimeError("set_cell needs a SkinNeighbourList built with variable_cell=True")
        self.cell_t.copy_(cell)

    def needs_rebuild(self, positions: torch.Tensor, fraction: float = 1.0) -> torch.Tensor:
        """Device bool: has any atom moved more than ``fraction * skin / 2`` since the last build?

        variable_cell: the deformation-aware test of the class docstring, with ``fraction``
        scaling the allowed displacement in the same way.
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

    def compact(self, positions: torch.Tensor) -> None:
        """Move pairs within the cutoff to the front (stable order); set ``count``.

        Uses only fixed-shape device operations, so it can be captured in a CUDA graph.
        """
        sender, receiver = self.candidate_index
        if self.variable_cell:  # image shifts in the current cell
            shifts64 = times_3x3(self.candidate_unit_shifts.to(torch.float64), self.cell_t)
            shifts = shifts64.to(self.float_dtype)
        else:
            shifts64, shifts = self.candidate_shifts64, self.candidate_shifts
        vectors = positions[receiver] - positions[sender] + shifts64
        inside = ((vectors * vectors).sum(dim=1) < self.cutoff ** 2).to(torch.int64)
        count = inside.sum()
        destination = torch.where(inside.bool(), torch.cumsum(inside, 0) - 1,
                                  count + torch.cumsum(1 - inside, 0) - 1)
        self.edge_index.index_copy_(1, destination, self.candidate_index)
        self.unit_shifts.index_copy_(0, destination, self.candidate_unit_shifts)
        self.shifts.index_copy_(0, destination, shifts)
        self.count.copy_(count)
