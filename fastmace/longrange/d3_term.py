"""The D3 term of a force field: 3b-T02's pair list and 3b-T03's kernel as one ``DispersionTerm`` (plan task 3b-T04).

SPEC §11.2, docs/design/d3.md. ``D3Term`` owns a ``D3PairList`` with its own skin and rebuild test, and the
``D3Kernel`` that evaluates on it in the force precision. A copy in another precision (the MC energy copy,
``fastmace.mc.evaluator.float64_copy``) adds its own kernel with ``kernel_for``: it reads the same list, and
every build prepares every kernel.

- **Rebuild test.** ``needs_rebuild`` returns the list's device flag. The engine rebuilds when any list's
  flag is set; ``D3Term.build`` then runs only when this list's own test at the rebuild positions asks for
  it (``needs_rebuild_now``, at the smallest fraction the engine tests with, so a pipelined early rebuild
  includes it), and the D3 list keeps its own cadence (``DispersedForceField.rebuild``). No flag is kept
  between steps: a test at an MC trial's positions leaves nothing behind.
- **Addresses.** A build refills the list and the kernels' CSR buffers in place; ``build`` returns True when
  an address moved (a grown capacity), and the engine then captures its graphs again.
- **Species.** The kernels read the C6 references, radii and r2/r4 by atomic number through ``numbers``, a
  device tensor they keep by reference. ``use_numbers`` points them at the state's species, so a species
  event that writes the state needs no refresh of any D3 table (conductor's choice 5 is met by the
  kernel's layout: there is no per-atom D3 table to go stale).
- **Trials.** ``save_trial`` and ``restore_trial`` copy the list and the kernels' buffers, for MC
  displacement trials built around their geometry (``fastmace.mc.displace``).
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from fastmace.longrange import d3
from fastmace.longrange.d3_kernel import D3Kernel
from fastmace.neighbors.d3_list import D3PairList


def cutoff_A(record: d3.D3) -> float:
    """The D3 list's cutoff: the larger of the pair and coordination-number cutoffs, in Å."""
    return max(d3.pair_cutoff_A(record), d3.cn_cutoff_A(record))


def _volume(cell):
    """det(cell) from cross products (``linalg.det`` checks for errors on the host); capturable."""
    return torch.dot(cell[0], torch.linalg.cross(cell[1], cell[2]))


class D3Term(d3.DispersionTerm):
    """``D3Term(record, numbers, cell, *, skin, dtype, variable_cell)``: the D3 term on its own half list.

    ``numbers``: int64 atomic numbers on the device (kept by reference); ``cell``: the (3, 3) cell in Å;
    ``skin``: rule 11's D3 skin in Å; ``dtype``: the force precision of the pair terms; ``variable_cell``:
    NPT and evaluations at other cells (the cell then comes from ``set_cell``).
    """

    def __init__(self, record: d3.D3, numbers, cell, *, skin: float, dtype=torch.float64,
                 variable_cell: bool = False):
        super().__init__(record, numbers)
        device = numbers.device
        self.list = D3PairList(cell, cutoff_A(record), float(skin), device, variable_cell=variable_cell)
        self.kernel = D3Kernel(record, numbers, device=device, dtype=dtype)
        self.kernels = [self.kernel]
        self.fraction = 1.0  # the smallest fraction needs_rebuild was called with (a host constant)

    @property
    def variable_cell(self) -> bool:
        return self.list.variable_cell

    @property
    def cell(self):
        """The current cell (float64 device tensor) the evaluations use."""
        return self.list.cell_t

    # ------------------------------------------------------------------ host side

    def _build(self, positions, cell) -> bool:
        moved = self.list.build(positions, cell if self.variable_cell else None)
        for kernel in self.kernels:
            moved |= self._prepare(kernel)
        return moved

    def _prepare(self, kernel) -> bool:
        listing = self.list
        return bool(kernel.prepare(listing.index[0], listing.index[1], listing.image, listing.shifts))

    def kernel_for(self, dtype) -> D3Kernel:
        """A kernel in ``dtype`` on this list (a copy in another precision); prepared at every build."""
        for kernel in self.kernels:
            if kernel.dtype == dtype:
                return kernel
        kernel = D3Kernel(self.record, self.numbers, device=self.numbers.device, dtype=dtype)
        if self.builds:
            self._prepare(kernel)
        self.kernels.append(kernel)
        return kernel

    def use_numbers(self, numbers) -> None:
        """Read the atomic numbers from ``numbers`` (the state's species, by reference) from now on."""
        if numbers.shape != self.numbers.shape:
            raise ValueError(f"numbers must have shape {tuple(self.numbers.shape)}, got {tuple(numbers.shape)}")
        if not torch.equal(numbers.to(self.numbers.device), self.numbers):
            raise ValueError("the numbers source differs from the D3 term's current numbers")
        self.numbers = numbers
        for kernel in self.kernels:
            kernel.numbers = numbers

    def needs_rebuild_now(self, positions) -> bool:
        """Host answer (one synchronisation): does the list's test at ``positions`` ask for a build, at the
        smallest fraction the engine tests with?"""
        return bool(self.list.needs_rebuild(positions, None, self.fraction))

    # ------------------------------------------------------------------ capturable

    def set_cell(self, cell) -> None:
        if self.variable_cell:
            self.list.set_cell(cell)

    def reset_cell(self, cell) -> None:
        """A new cell between replays, fixed or variable cell (4-T05's ``Simulation.update(cell=)``; host): the
        list's, for the canonical point's rebuild (``DispersedForceField.rebuild_all``). The kernels read the list's
        ``cell_t`` in place. ``GraphEngine.reset_cell`` passes a device tensor; the list takes a host array."""
        if isinstance(cell, torch.Tensor):
            cell = cell.detach().cpu().numpy()
        self.list.reset_cell(cell)

    def needs_rebuild(self, positions, cell=None, fraction: float = 1.0):
        self.fraction = min(self.fraction, float(fraction))
        return self.list.needs_rebuild(positions, cell if self.variable_cell else None, fraction)

    def evaluate(self, positions, cell=None, *, virial: bool = True, kernel=None) -> d3.DispersionResult:
        if cell is not None:
            self.set_cell(cell)
        return (kernel or self.kernel).evaluate(positions, self.list.cell_t, virial=virial)

    def evaluate_energy(self, positions, cell=None, *, kernel=None):
        if cell is not None:
            self.set_cell(cell)
        return (kernel or self.kernel).evaluate_energy(positions, self.list.cell_t)

    def stress(self, result: d3.DispersionResult):
        """``result``'s stress in ASE's sign at the current cell (eV/Å³, float64); capturable."""
        return result.stress(_volume(self.list.cell_t))

    # ------------------------------------------------------------------ registry, counters, trials

    def buffers(self) -> Mapping[str, torch.Tensor]:
        """Device buffers that live across steps: the list's and the kernels' CSR buffers."""
        out = {f"list.{name}": tensor for name, tensor in self.list.buffers().items()}
        for index, kernel in enumerate(self.kernels):
            out.update({f"kernel.{index}.{name}": tensor for name, tensor in kernel.buffers().items()})
        return out

    def counters(self) -> dict:
        return {**super().counters(), **self.list.counters()}

    def save_trial(self, into: dict | None = None) -> dict:
        """Copy the list and the kernels' buffers before a trial build (``D3PairList.save_list``)."""
        saved = {} if into is None else into
        saved["list"] = self.list.save_list(saved.get("list"))
        kernels = saved.setdefault("kernels", {})
        with torch.no_grad():
            for index, kernel in enumerate(self.kernels):
                for name, tensor in kernel.buffers().items():
                    copy = kernels.get((index, name))
                    if copy is None or copy.shape != tensor.shape:
                        kernels[(index, name)] = tensor.detach().clone()
                    else:
                        copy.copy_(tensor)
        return saved

    def restore_trial(self, saved: dict) -> None:
        """Copy a ``save_trial`` copy back in place (refused once a buffer moved since)."""
        self.list.restore_list(saved["list"])
        with torch.no_grad():
            for index, kernel in enumerate(self.kernels):
                for name, tensor in kernel.buffers().items():
                    copy = saved["kernels"][(index, name)]
                    if copy.shape != tensor.shape:
                        raise RuntimeError("the D3 kernel's buffers moved since they were saved; rebuild instead")
                    tensor.copy_(copy)


__all__ = ["D3Term", "cutoff_A"]
