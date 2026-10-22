"""MACE plus D3: the execution force field of a plan with dispersion (plan task 3b-T04; SPEC §11.2).

``DispersedForceField(field, record, atoms, *, skin)`` wraps a plan's ``MACEForceField`` and a
``fastmace.longrange.d3_term.D3Term`` and implements the same force-field protocol
(``fastmace.model.forces.ForceField``). Everything it does not override is the MACE force field's
(attribute access is forwarded), so the engine, the planner's checks and the MC moves see one force
field. Without D3 the planner builds the plain ``MACEForceField`` and nothing here runs, so a D3-free
plan is the code path of before, bit for bit.

- **The force call** (``__call__``) evaluates MACE, then the D3 term's three passes on the same
  positions, and returns the sums: energy (float64, D3's float64 energy added to MACE's), forces (D3's
  in the force precision) and, with a variable cell, the stress (D3's ``-virial / V`` in the stress's
  precision). Both run in the force stage, so a CUDA graph of the force stage captures MACE and D3
  together; the eager and compiled tiers call the same function (only MACE's model is compiled; the
  Triton passes stay outside the compiled region).
- **Energy only** (MC energies, ``energy_only``): MACE's forward plus D3's ``evaluate_energy``.
- **Lists.** ``needs_rebuild`` is the OR of the MACE list's flag and the D3 list's (device ops,
  capturable). ``rebuild`` (host side, after a flag) rebuilds the MACE list as before and the D3 list
  only when its own test asks for it, so the D3 list keeps the cadence its own skin gives (rule 11).
  ``rebuild_all`` rebuilds both: the canonical restart points (SPEC §10.6). A grown capacity of either
  returns True, and the engine captures its graphs again.
- **Cell.** ``set_cell`` gives the cell to both (the D3 list's deformation-aware test and its shifts).
- **Species.** ``register_derived`` registers MACE's derived buffers and points the D3 kernels at the
  state's species (``D3Term.use_numbers``); the kernels index their tables by atomic number, so a
  species event needs no D3 refresh.
- **Copies in another precision** (``following``): a MACE copy that follows this force field
  (``MACEForceField.follow``) gets a D3 kernel in its precision on this force field's D3 list.
"""

from __future__ import annotations

from typing import Any

from fastmace.model.forces import ForceOutput


class DispersedForceField:
    """A ``MACEForceField`` plus a D3 term (module docstring)."""

    def __init__(self, field: Any, record: Any, atoms: Any = None, *, skin: float, term: Any = None):
        import numpy as np
        import torch

        from fastmace.longrange.d3_term import D3Term

        self.__dict__["mace"] = field
        self.dispersion = record
        self._leader = None
        if term is None:
            cell = np.asarray(field.neighbours.cell, dtype=np.float64)
            term = D3Term(record, field.atomic_numbers, cell, skin=skin, dtype=field.dtype,
                          variable_cell=field.compute_stress)
            if atoms is None:
                raise ValueError("a new D3 term needs the structure's atoms for its first build")
            term.build(torch.as_tensor(np.asarray(atoms.get_positions()), dtype=torch.float64, device=field.device),
                       None)
            self.kernel = term.kernel
        else:
            self.kernel = term.kernel_for(field.dtype)
        self.d3 = term

    @classmethod
    def following(cls, copy, leader: DispersedForceField) -> DispersedForceField:
        """``copy`` (a ``MACEForceField`` in another precision) following ``leader``'s MACE force field, with a
        D3 kernel in its precision on ``leader``'s D3 list (the MC energy copy)."""
        copy.follow(leader.mace)
        out = cls(copy, leader.dispersion, skin=leader.d3.list.skin, term=leader.d3)
        out._leader = leader
        return out

    def __getattr__(self, name):
        if name == "mace":  # before __init__ finished (copy, pickle): no recursion
            raise AttributeError(name)
        return getattr(self.mace, name)

    # ------------------------------------------------------------------ evaluation

    def __call__(self, positions):
        """Energy [1], forces [N, 3] (and stress [3, 3]) of MACE plus D3 at float64 ``positions``."""
        out = self.mace(positions)
        stress = out.stress is not None
        result = self.d3.evaluate(positions, virial=stress, kernel=self.kernel)
        energy = out.energy + result.energy.to(out.energy.dtype)
        forces = out.forces + result.forces.to(out.forces.dtype)
        if stress:
            stress = out.stress + self.d3.stress(result).to(out.stress.dtype)
        else:
            stress = None
        return ForceOutput(energy, forces, stress)

    def energy_only(self, positions):
        """The float64 energy [1] of MACE plus D3: forward passes only (``MACEForceField.energy_only``)."""
        import torch

        energy = self.mace.energy_only(positions)
        with torch.no_grad():  # a copy shares its leader's D3 list, so the cell is the leader's
            return energy + self.d3.evaluate_energy(positions, kernel=self.kernel).to(energy.dtype)

    def evaluate(self, atoms, stress=False):
        """Energy (eV), forces (eV/Å, float64) and the 3x3 stress or None of ``atoms``, MACE plus D3 (one-off)."""
        import torch

        if stress and not self.mace.compute_stress:
            raise ValueError("stress=True needs a force field built with compute_stress=True")
        positions = self.mace._prepare_one_off(atoms)
        if self.d3.variable_cell:
            self.d3.set_cell(torch.as_tensor(atoms.cell.array, dtype=torch.float64, device=self.mace.device))
        self.d3.build(positions, None)
        out = self(positions)
        energy = float(out.energy.detach().reshape(-1)[0])
        forces = out.forces.detach().to(torch.float64).cpu().numpy()
        values = out.stress.detach().to(torch.float64).cpu().numpy() if stress else None
        return energy, forces, values

    # ------------------------------------------------------------------ lists and cell

    def set_cell(self, cell):
        self.mace.set_cell(cell)
        self.d3.set_cell(cell)

    def needs_rebuild(self, positions, fraction=1.0):
        return self.mace.needs_rebuild(positions, fraction) | self.d3.needs_rebuild(positions, fraction=fraction)

    def d3_needs_rebuild(self, positions, fraction=1.0):
        """The D3 list's flag alone (device bool): what a fused stage that tests the MACE list itself ORs into the
        engine's flags."""
        return self.d3.needs_rebuild(positions, fraction=fraction)

    def rebuild(self, positions):
        """Rebuild the MACE list, and the D3 list when its own test asks for it; True if a buffer moved."""
        grew = bool(self.mace.rebuild(positions))
        if self.d3.needs_rebuild_now(positions):  # host sync (rebuild steps only)
            # An early rebuild beside the force call in flight (OPT2-T04) runs on a side stream: the force call
            # reads the D3 list in place, so its build waits for it (the MACE build writes candidates only).
            reader = getattr(self.mace.neighbours, "reader_stream", None)
            if reader is not None:
                import torch

                torch.cuda.current_stream().wait_stream(reader)
            grew |= bool(self.d3.build(positions, None))
        return grew

    def rebuild_all(self, positions):
        """Rebuild both lists (a canonical restart point, SPEC §10.6); True if a buffer moved."""
        grew = bool(self.mace.rebuild(positions))
        return bool(self.d3.build(positions, None)) | grew

    # ------------------------------------------------------------------ registry, provenance, trials

    def buffers(self):
        return {**self.mace.buffers(), **{f"d3.{name}": tensor for name, tensor in self.d3.buffers().items()}}

    def register_derived(self, registry, state=None, prefix="force_field"):
        out = self.mace.register_derived(registry, state, prefix=prefix)
        if state is not None and self._leader is None:
            self.d3.use_numbers(state.species)
        return out

    def info(self):
        from fastmace.longrange import d3

        return {**self.mace.info(), "dispersion": d3.describe(self.dispersion), "d3": self.d3.counters()}

    def save_trial(self, into=None):
        """Copy both lists before an MC trial build (``SkinNeighbourList.save_list``, ``D3Term.save_trial``)."""
        saved = {} if into is None else into
        saved["mace"] = self.mace.neighbours.save_list(saved.get("mace"))
        saved["d3"] = self.d3.save_trial(saved.get("d3"))
        return saved

    def restore_trial(self, saved):
        self.mace.neighbours.restore_list(saved["mace"])
        self.d3.restore_trial(saved["d3"])


def wrap(field, record, atoms, *, skin):
    """``field`` with the D3 term of ``record`` (a ``DispersedForceField``), or ``field`` itself without D3."""
    if record is None:
        return field
    return DispersedForceField(field, record, atoms, skin=skin)


__all__ = ["DispersedForceField", "wrap"]
