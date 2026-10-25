"""Corrected reference calculators (SPEC §12.1, §8.4; plan task 1b-T13).

A ``ReferenceCalculator`` evaluates energy, forces and stress with MACE's ``MACECalculator`` as
its reference record states, with the record's corrections applied and checked:

- **float64 reference**: MACE's cuEquivariance model in float64, MACE's own forward;
- **float32 yardstick** of the ratio rule: the same in float32, with the energy reduced into
  float64 (``fastmace.model.energy``; PolarMACE through ``fastmace.longrange.polar``) and the
  checkpoint's float64 E0 table. Forces and stress are MACE's float32 autograd, bit for bit.

The safe order (1a-T19, "Conductor decisions after T17"): the gate's references stored MACE's
bound ``model.forward`` (``configure_md_calculator``) before the corrected PolarMACE forward was
installed, so they computed MACE 0.3.16's uncorrected stress, and nothing noticed. Here:

1. the record is checked against the model (a polar model needs the issue 1642 fix);
2. a model with a forward stored on the instance is refused: it would bypass the class-level
   forward that carries the corrections;
3. the process-wide stock patches of the record are put in force (the one-graph strain);
4. PolarMACE: the PR 1762 options of the record, then ``polar.enable`` (k-space state for the
   cell, headroom 0.1 for strained copies as 1a-T19's references), then, for the yardstick,
   ``polar.set_float64_energy``;
5. every evaluation first asserts that each correction is the active code path, and raises
   ``ReferenceOrderError`` otherwise.

The data preparation and unit handling are ``MACECalculator.calculate``'s (one model, no
padding, no compile), so a stock record gives MACE's numbers. Whether references may run in this
process (none after a production engine, 1a "after T10") is the gate's decision (1b-T16); this
module builds no production engine.
"""

from __future__ import annotations

import numpy as np

from fastmace.errors import FastMACEError
from fastmace.reference import records

ROLES = {"float64": "reference", "float32": "yardstick"}
POLAR_FORWARD = "mace.modules.extensions:PolarMACE.forward"
# The finite-difference self-test of the polar stress fix (1a-T19's tools/force_stress_gate.py).
FD_STRAIN = 1e-4
FD_TOLERANCE = 1e-6  # eV/A^3
VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


class ReferenceOrderError(FastMACEError):
    """A correction of the reference record is not the active code path (SPEC §12.1)."""

    default_next_step = ("Build the reference with fastmace.reference.calculators (load or "
                         "ReferenceCalculator.from_calculator) on a calculator nothing else has configured, "
                         "and do not replace its model's forward afterwards.")


def _dtype_name(model):
    import torch

    dtype = next(model.parameters()).dtype
    return {torch.float64: "float64", torch.float32: "float32"}.get(dtype, str(dtype))


def _polar_config(record):
    from fastmace.config import PolarConfig

    for entry in record.optional_physics:
        if entry["name"] == "pr1762":
            options = entry["options"]
            cap = options["dipole_cap"]
            return PolarConfig(fukui_regularisation=options["fukui_regularisation"],
                               dipole_cap=None if cap is None else tuple(cap))
    return PolarConfig()


class ReferenceCalculator:
    """One corrected reference calculator: ``evaluate(atoms) -> (energy, forces, stress)``.

    Build it with ``load`` (from a checkpoint) or ``from_calculator`` (from a MACE calculator that
    nothing else has configured). ``role`` is "reference" (float64) or "yardstick" (float32).
    """

    def __init__(self, calculator, record, atoms, *, atomic_energies=None, loaded=None):
        from fastmace import _patches
        from fastmace.longrange import polar
        from fastmace.model import energy as energy64

        models = list(getattr(calculator, "models", ()))
        if len(models) != 1:
            raise ValueError(f"a reference calculator holds exactly one MACE model, got {len(models)}")
        if getattr(calculator, "use_compile", False) or getattr(calculator, "pad_num_atoms", 0) \
                or getattr(calculator, "pad_num_edges", 0):
            raise ValueError("a reference calculator is uncompiled and unpadded (MACECalculator defaults)")
        self.calculator = calculator
        self.model = model = models[0]
        self.record = records.require_record_for_model(record, model)
        self.dtype = _dtype_name(model)
        if self.dtype not in ROLES:
            raise ValueError(f"a reference model is float64 or float32, got {self.dtype}")
        self.role = ROLES[self.dtype]
        self.is_polar = polar.is_polar(model)
        numerical = {entry["name"] for entry in record.numerical}
        # 2-T04: an unqualified MACE without the hashed float64 forward keeps MACE's float32 sum (recorded).
        self.float32_sum = self.role == "yardstick" and records.FLOAT32_SUM_UNQUALIFIED["name"] in numerical
        self.float64_energy = self.role == "yardstick" and not self.float32_sum
        if self.float64_energy and records.FLOAT64_ACCUMULATION["name"] not in numerical:
            raise records.ReferencePolicyError(
                f"record {record.id} has no float64 energy accumulation for the float32 yardstick",
                why="without it the ratio rule measures the float32 E0 sum, not the forward pass (SPEC §12.1).")
        self.loaded = loaded
        # 2. Nothing may have stored a forward on the instance before the corrections are in place.
        self._require_class_forward()
        # 3. The record's process-wide stock patches.
        _patches.set_single_graph_strain(bool(record.stock["single_graph_strain"]))
        # 4. PolarMACE: options, k-space state (the issue 1642 fix), float64 energy.
        self.polar_info = None
        if self.is_polar:
            config = _polar_config(record)
            self.polar_info = polar.enable(model, np.asarray(atoms.cell.array, float), config, variable_cell=True)
            if self.float64_energy:
                polar.set_float64_energy(model, True)
        self._float64_forward = None
        if self.float64_energy and not self.is_polar:
            self._float64_forward = energy64.float64_forward(model)  # called as a function, never stored
        self.e0_table = None
        if self.float64_energy:
            import torch

            table = atomic_energies if atomic_energies is not None else getattr(loaded, "atomic_energies", None)
            self.e0_source = "checkpoint" if table is not None else "model (rounded to the model's precision)"
            if table is None:
                table = energy64.model_atomic_energies(model)
            table = torch.as_tensor(table, dtype=torch.float64).cpu().clone()
            expected = tuple(model.atomic_energies_fn.atomic_energies.shape)
            if tuple(table.shape) != expected:
                raise ValueError(f"atomic_energies has shape {tuple(table.shape)}, the model's E0 table {expected}")
            self.e0_table = table
        self.patches = tuple(record_.name for record_ in _patches.applied())
        # 5. D3 (optional physics, 3b-T04): torch-dftd on the record's numbers, summed with MACE's results.
        self.d3 = d3_record(record)
        self._d3_calculator = None
        if self.d3 is not None:
            from fastmace.reference import d3 as d3_reference

            device = str(next(model.parameters()).device)
            self._d3_calculator = d3_reference.torch_dftd3_calculator(self.d3, dtype="float64", device=device)
        self.require_active()

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_calculator(cls, calculator, record, atoms, *, atomic_energies=None):
        """A reference from a MACE calculator that nothing else has configured (the safe order above)."""
        return cls(calculator, record, atoms, atomic_energies=atomic_energies)

    # ------------------------------------------------------------------ checks

    def _require_class_forward(self):
        if "forward" in self.model.__dict__:
            raise ReferenceOrderError(
                "the reference model has a forward stored on the instance",
                why="a stored forward bypasses the class-level forward that carries the record's corrections "
                    "(1a-T19: configure_md_calculator stored MACE's forward before the PolarMACE stress fix was "
                    "installed, so the references ran MACE 0.3.16's stress).")

    def require_active(self):
        """Raise ``ReferenceOrderError`` unless every correction of the record is the active code path."""
        from fastmace import _patches
        from fastmace.longrange import polar

        self._require_class_forward()
        want = bool(self.record.stock["single_graph_strain"])
        if (_patches.record_of("single_graph_strain") is not None) != want:
            raise ReferenceOrderError(f"the one-graph strain is {'off' if want else 'on'}, the record says "
                                      f"{'on' if want else 'off'}")
        if self.is_polar:
            installed = polar.installed()
            if not installed or type(self.model).forward is polar._original(POLAR_FORWARD):
                raise ReferenceOrderError("PolarMACE's corrected forward is not installed (MACE issue 1642)")
            if polar.get_state(self.model) is None:
                raise ReferenceOrderError(
                    "the PolarMACE stress fix (MACE issue 1642) is not active: the model has no k-space state",
                    why="without it MACE's own long-range code runs, whose stress misses the cell dependence.")
            if polar.get_config(self.model) != _polar_config(self.record):
                raise ReferenceOrderError("the PolarMACE options differ from the record's optional physics")
            holder = polar._holder(self.model)
            if bool(holder.float64_energy) != self.float64_energy:
                raise ReferenceOrderError("the PolarMACE float64 energy setting differs from the reference's role")

    def corrections_active(self):
        return ["polar_kspace_stress"] if self.is_polar else []

    # ------------------------------------------------------------------ evaluation

    def evaluate(self, atoms):
        """``(energy, forces, stress)`` of ``atoms``: a Python float (eV), (N, 3) and (3, 3) float64 arrays."""
        import torch
        from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress

        from fastmace.longrange import polar
        from fastmace.model import energy as energy64

        self.require_active()
        calc = self.calculator
        batch = calc._atoms_to_batch(self._with_charge_spin(atoms))
        dtype = next(self.model.parameters()).dtype
        data = {key: (value.to(dtype=dtype) if torch.is_tensor(value) and torch.is_floating_point(value) else value)
                for key, value in batch.to_dict().items()}
        if self.e0_table is not None:
            head = int(data["head"].reshape(-1)[0]) if "head" in data else 0
            data[energy64.NODE_E0_KEY] = energy64.node_e0_float64(self.e0_table, data["node_attrs"], head).to(
                data["node_attrs"].device)
        if self._float64_forward is not None:
            out = self._float64_forward(self.model, data, compute_stress=True, training=False)
        else:
            out = self.model(data, compute_stress=True, training=False)
        if self.is_polar and polar.overflowed(self.model):
            raise records.ReferencePolicyError(
                "the reference's k-vector superset overflowed: this cell is outside the headroom it was built for",
                next_step="Call rebase(atoms) for this structure.")
        energy_unit = calc.energy_units_to_eV
        energy = float(out["energy"].detach().reshape(-1)[0].item()) * energy_unit
        forces = out["forces"].detach().cpu().numpy() * (energy_unit / calc.length_units_to_A)
        stress = out["stress"].detach()[0].cpu().numpy() * (energy_unit / calc.length_units_to_A ** 3)
        # MACECalculator stores Voigt; ASE's get_stress(voigt=False) expands it again.
        stress = voigt_6_to_full_3x3_stress(full_3x3_to_voigt_6_stress(stress))
        forces, stress = np.asarray(forces, dtype=np.float64), np.asarray(stress, dtype=np.float64)
        if self._d3_calculator is not None:  # MACE + torch-dftd, as ASE's SumCalculator sums them
            energy_d3, forces_d3, stress_d3 = self._evaluate_d3(atoms)
            energy, forces, stress = energy + energy_d3, forces + forces_d3, stress + stress_d3
        return energy, forces, stress

    def _evaluate_d3(self, atoms):
        copy = atoms.copy()
        copy.set_constraint()  # every atom's forces: FixAtoms would zero the fixed rows (4-T04)
        copy.calc = self._d3_calculator
        return (float(copy.get_potential_energy()), np.array(copy.get_forces(), dtype=np.float64),
                np.array(copy.get_stress(voigt=False), dtype=np.float64))

    def _with_charge_spin(self, atoms):
        """``atoms`` with the loaded model's total charge and spin in ``info``, where MACE's calculator reads them,
        as the production force field sets them (``MACEForceField``): a configuration rebuilt from numbers and
        positions (the gate's copies and sample moves) carries none, and the calculator then took charge 0 and
        its default spin (found by 3-T11 on a charged model). Keys already set are kept."""
        loaded = self.loaded
        charge = getattr(loaded, "charge", None)
        spin = getattr(loaded, "spin_multiplicity", None)
        if charge is None and spin is None:
            return atoms
        keys = getattr(self.calculator, "info_keys", {}) or {}
        wanted = {keys.get("total_charge", "charge"): charge, keys.get("total_spin", "spin"): spin}
        missing = {key: value for key, value in wanted.items() if value is not None and key not in atoms.info}
        if not missing:
            return atoms
        atoms = atoms.copy()
        atoms.info.update(missing)
        return atoms

    def rebase(self, atoms):
        """Size the PolarMACE k-space state for ``atoms``' cell (another structure); no-op otherwise."""
        from fastmace.longrange import polar

        if self.is_polar:
            self.polar_info = polar.enable(self.model, np.asarray(atoms.cell.array, float), _polar_config(self.record),
                                           variable_cell=True)
            if self.float64_energy:
                polar.set_float64_energy(self.model, True)
        return self.polar_info

    def check_stress_fix(self, atoms, *, tolerance=FD_TOLERANCE, h=FD_STRAIN):
        """The issue 1642 self-test: analytic stress against central finite differences of this energy.

        Returns the result; raises ``ReferencePolicyError`` when they differ by more than ``tolerance``
        (eV/A^3). The finite differences hold the k-vectors included for ``atoms``' cell (``polar.frozen_kset``):
        a k-vector crossing the hard k-space cutoff under the strain steps the energy, and central differences
        across the step go as 1/h (3b-T08: 4.4e-4 eV/A^3 on NbCl5 3x3x3, 8.3e-9 frozen). So the test runs on
        any structure (record entry ``validation``).
        """
        from fastmace.longrange import polar

        if not self.is_polar:
            raise ValueError("the stress fix applies to PolarMACE references only")
        analytic = self.evaluate(atoms)[2]
        with polar.frozen_kset(self.model, atoms.cell.array) as included:
            fd = finite_difference_stress(self, atoms, h=h)
        delta = float(np.abs(analytic - fd).max())
        result = {"max_abs": delta, "tolerance": float(tolerance), "strain": float(h), "kset": "frozen",
                  "k_vectors_included": included, "fd_stress": fd.tolist(), "passed": bool(delta <= tolerance)}
        if delta > tolerance:
            raise records.ReferencePolicyError(
                f"the reference stress differs from finite differences of its energy by {delta:.2e} eV/A^3 "
                f"(tolerance {tolerance:g})",
                why="the PolarMACE k-space stress correction (MACE issue 1642) is not in effect.")
        return result

    def info(self):
        """What provenance and the gate record need about this reference."""
        loaded = self.loaded
        return {
            "record_id": self.record.id, "role": self.role, "dtype": self.dtype,
            "float64_energy": self.float64_energy,
            "energy_bound": records.ENERGY_BOUND_UNQUALIFIED if self.float32_sum else "float64 accumulation",
            "e0_table": getattr(self, "e0_source", None),
            "backend": "cueq" if getattr(self.calculator, "_enable_cueq", False) else "e3nn",
            "cueq_conv_fusion": None if loaded is None else loaded.cueq_conv_fusion,
            "model": None if loaded is None else {"path": str(loaded.path), "sha256": loaded.sha256,
                                                  "label": loaded.label, "head": loaded.head},
            "patches": list(self.patches),
            "corrections_active": self.corrections_active(),
            "optional_physics": [entry["name"] for entry in self.record.optional_physics],
            "gate_floors": records.gate_floors(self.record),
            "stock": self.record.is_stock,
            "polar": self.polar_info,
            "dispersion": None if self.d3 is None else _describe_d3(self.d3),
        }


def d3_record(record):
    """The ``fm.D3`` of a reference record's ``d3`` optional-physics entry, or None."""
    for entry in record.optional_physics:
        if entry["name"] == "d3":
            import dataclasses

            from fastmace.longrange.d3 import D3

            names = {f.name for f in dataclasses.fields(D3)}
            return D3(**{key: value for key, value in entry["parameters"].items() if key in names})
    return None


def _describe_d3(record):
    from fastmace.longrange import d3

    return d3.describe(record) + ", non-stock (MACE plus torch-dftd, SPEC §12.1)"


def finite_difference_stress(reference, atoms, h=FD_STRAIN):
    """3x3 stress (eV/A^3, ASE sign) of ``reference``'s energy by central differences in the strain.

    Diagonal ii: strain e_ii = +-h; shear ij: e_ij = e_ji = +-h/2; sigma = (E(+) - E(-)) / (2 h V)
    (1a-T19's ``tools/force_stress_gate.py``).
    """
    volume = atoms.get_volume()
    stress = np.zeros((3, 3))
    for i, j in VOIGT:
        e = np.zeros((3, 3))
        if i == j:
            e[i, i] = h
        else:
            e[i, j] = e[j, i] = h / 2
        energies = []
        for sign in (1, -1):
            copy = atoms.copy()
            copy.set_cell(atoms.cell.array @ (np.identity(3) + sign * e), scale_atoms=True)
            energies.append(reference.evaluate(copy)[0])
        stress[i, j] = stress[j, i] = (energies[0] - energies[1]) / (2 * h * volume)
    return stress


def load(source, atoms, *, dtype="float64", device="cuda:0", head=None, family=None, polar=None, d3=None,
         sha256=None, trusted=False, charge=None, spin_multiplicity=None, environ=None, checkpoint=None):
    """A reference calculator from a checkpoint, as its record states (``fastmace.model.loading.load``).

    ``family``: the record family (default: the label's, else "polar" or "unlabelled");
    ``polar``: a ``PolarConfig`` (PR 1762 options); ``d3``: an ``fm.D3`` record, whose torch-dftd term is
    added to MACE's results (SPEC §12.1, the record's ``d3`` entry). On the CPU the cuEquivariance model
    takes MACE's naive kernels (the CPU float64 references of 1a-T15). ``checkpoint``: the label's file
    as an explicit path (``loading.resolve_checkpoint``), still checked against ``sha256``.
    """
    from fastmace.model import loading

    version = records.mace_version()
    if d3 is not None:
        records.record_for(version, family or "unlabelled", d3=d3)  # refusals before any model is loaded
    stock = records.STOCK
    loaded = loading.load(source, head, checkpoint=checkpoint, backend=stock["backend"], dtype=dtype, device=device,
                          cueq_conv_fusion=stock["cueq_conv_fusion"], sha256=sha256, trusted=trusted,
                          allow_naive_cueq=str(device).startswith("cpu"), charge=charge,
                          spin_multiplicity=spin_multiplicity, environ=environ)
    record = records.record_for(version, records.family_of(loaded.model, family or loaded.family), polar=polar,
                                d3=d3)
    return ReferenceCalculator(loaded.calculator, record, atoms, loaded=loaded)


__all__ = ["FD_STRAIN", "FD_TOLERANCE", "ROLES", "ReferenceCalculator", "ReferenceOrderError",
           "finite_difference_stress", "load"]
