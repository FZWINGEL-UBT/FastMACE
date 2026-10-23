"""The reference tier for MD (SPEC §8.4, §9.3 rule 1, §9.4 terminal sequence).

Plan task 2-T03. ``ReferenceForceField`` evaluates forces, energy and stress through the versioned
reference calculator (``fastmace.reference.calculators``: MACE's ``MACECalculator`` with the
record's corrections) and implements the force-field protocol of ``fastmace.model.forces.ForceField``,
so FastMACE's own integrators, noise stream, output and restart drive it: only the forces come from
the reference. It runs on ``GraphEngine`` with ``capture=False`` and exact edge counts, with no
rewrites.

MACE's calculator builds its own neighbour list at every call (``_atoms_to_batch``), so the
neighbour methods of the protocol do nothing here: no rebuild is ever needed, compaction and
``use_edges`` are no-ops, and the edge capacity is 0. ``edge_count`` is the edge count of the last
evaluation, for the record. Each call reads the positions (and the cell) back to the host, which
is the price of this tier: it is the last resort and the reference for the gate, not a fast path.

PolarMACE: the reference's k-space state is sized for a cell (``ReferenceCalculator.rebase``). A
cell that leaves its headroom raises in the reference; this force field then sizes it for the
current cell and evaluates again, as the gate's references do. ``reset_polar`` (called by every
canonical restart point) sizes it for the current cell, as a fresh build does, so that a resumed
run continues bit for bit.

D3 (3b-T04): with a force field that has D3, the reference is MACE plus ``torch-dftd`` on the same
record (``calculators.load(d3=...)``, SPEC §12.1), labelled non-stock.

The planner builds it for ``mode="reference"`` and as the last candidate of the fallback's terminal
sequence when it fits (§9.4 step 5, ``MinimalPlanner.reference_fits``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DTYPES = ("float64", "float32")


class _NoNeighbours:
    """The neighbour-list surface the engine and the gate read; MACE's calculator has its own list."""

    config = None

    def __init__(self):
        self.rebuilds = 0

    def set_capacity(self, capacity: int) -> bool:
        """A restart's neighbour capacity: nothing to size, so no buffer moves."""
        return False


class ReferenceForceField:
    """A ``fastmace.model.forces.ForceField`` whose evaluations are the reference calculator's.

    ``force_field``: the ``fm.ForceField`` (pristine model, superset, identity); ``atoms``: the run's
    structure; ``device``: where the reference runs (default: CUDA if available, else the CPU);
    ``dtype``: ``"float64"`` (default, the gate's reference) or ``"float32"`` (the yardstick: float32
    forces, energy summed in float64); ``polar``: the plan's ``PolarConfig`` (PR 1762 options of a
    PolarMACE model); ``compute_stress``: whether calls return the stress (variable-cell runs).
    """

    def __init__(self, force_field: Any, atoms: Any, *, device: str | None = None, dtype: str = "float64",
                 polar: Any = None, compute_stress: bool = False):
        import torch

        from fastmace.reference import calculators

        if dtype not in DTYPES:
            raise ValueError(f"the reference tier runs in float64 or float32, got {dtype!r}")
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = force_field.model
        self.source = model
        self.reference = calculators.load(
            model.label or str(model.path), atoms, dtype=dtype, device=device, head=None if model.label else model.head,
            family=model.family, polar=polar, d3=getattr(force_field, "dispersion", None), sha256=model.sha256,
            charge=model.charge,
            spin_multiplicity=model.spin_multiplicity)
        self._device = torch.device(device)
        self._dtype = torch.float64 if dtype == "float64" else torch.float32
        self._compute_stress = bool(compute_stress)
        self.atoms = atoms.copy()
        self.atoms.calc = None
        self.neighbours = _NoNeighbours()
        self.evaluations = 0
        self.polar_rebases = 0
        self.data = {"edge_index": torch.zeros((2, 0), dtype=torch.int64, device=self._device)}
        self._edge_count = torch.zeros((), dtype=torch.int64, device=self._device)
        self._false = torch.zeros((), dtype=torch.bool, device=self._device)

    # ------------------------------------------------------------------ protocol facts

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def compute_stress(self) -> bool:
        return self._compute_stress

    @property
    def padding_exact(self) -> bool:
        return True  # never padded: MACE's calculator evaluates the exact edges

    @property
    def is_polar(self) -> bool:
        return bool(self.reference.is_polar)

    compile_mode = "none"

    # ------------------------------------------------------------------ evaluation

    def _evaluate(self, atoms):
        from fastmace.reference.records import ReferencePolicyError

        try:
            return self.reference.evaluate(atoms)
        except ReferencePolicyError as error:
            if not (self.reference.is_polar and "overflowed" in (error.what or "")):
                raise
            self.reference.rebase(atoms)  # the cell left the k-space headroom: size it for this cell
            self.polar_rebases += 1
            return self.reference.evaluate(atoms)

    def evaluate(self, atoms: Any, stress: bool = False):
        """One-off evaluation of ``atoms``: energy (eV), forces, 3x3 stress or None (host values)."""
        energy, forces, values = self._evaluate(atoms)
        return energy, forces, values if stress else None

    def __call__(self, positions):
        """Energy, forces (and stress) at float64 ``positions`` and the cell of the last ``set_cell``."""
        import torch

        from fastmace.model.forces import ForceOutput

        self.atoms.positions = positions.detach().to("cpu", torch.float64).numpy()
        energy, forces, stress = self._evaluate(self.atoms)
        self.evaluations += 1
        out = ForceOutput(
            energy=torch.tensor([energy], dtype=torch.float64, device=self._device),
            forces=torch.as_tensor(forces, dtype=torch.float64).to(self._device),
            stress=torch.as_tensor(stress, dtype=torch.float64).to(self._device) if self._compute_stress else None)
        return out

    def set_cell(self, cell) -> None:
        import torch

        self.atoms.set_cell(cell.detach().to("cpu", torch.float64).numpy(), scale_atoms=False)

    def reset_polar(self, cell) -> tuple[bool, bool]:
        """A canonical restart point: size a PolarMACE k-space state for ``cell``, as a fresh build does."""
        if self.reference.is_polar:
            self.atoms.set_cell(cell, scale_atoms=False)
            self.reference.rebase(self.atoms)
        return False, False

    # ------------------------------------------------------------------ neighbour surface (no work)

    def needs_rebuild(self, positions, fraction: float = 1.0):
        return self._false

    def compact(self, positions) -> None:
        return None

    def rebuild(self, positions) -> bool:
        return False

    def use_edges(self, num_edges: int) -> None:
        return None

    @property
    def edge_count(self):
        return self._edge_count

    @property
    def edge_capacity(self) -> int:
        return 0

    def buffers(self) -> Mapping[str, Any]:
        return {}

    def info(self) -> Mapping[str, Any]:
        return {"tier": "reference", "evaluations": self.evaluations, "polar_rebases": self.polar_rebases,
                "reference": self.reference.info(), "streaming_interactions": False, "sort_receivers": False}


__all__ = ["DTYPES", "ReferenceForceField"]
