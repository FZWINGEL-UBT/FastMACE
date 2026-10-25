# SPDX-License-Identifier: LGPL-2.1-or-later
# Provenance: patches and re-derives ASE 3.27's ase/md/langevinbaoab.py (LangevinBAOAB); see NOTICE.md.
"""Independent references for the integrator correction (SPEC §12.1 "Integrator corrections", owner decision D8).

FastMACE's full-cell BAOAB drags the atom velocities with (p_eps + Tr(p_eps)/N_dof · I)/W
(Quigley and Probert 2004, Eq. 5b). ASE 3.27's ``LangevinBAOAB`` writes ``p_eps + trace(p_eps)/Ndof``
without the identity, so NumPy adds the trace term to all nine elements. Two references implement the
published form, independently of ``fastmace.integrators``:

- ``CorrectedLangevinBAOAB``: ASE's ``LangevinBAOAB`` with only the drag replaced. It shares ASE's
  noise stream, so trajectories compare exactly. ``check_ase_source`` refuses an ASE whose BAOAB
  differs from the one this patch was written against.
- ``baoab_step``: one float64 BAOAB step written from the equations of motion, with the noise passed
  in. With ``trace_all_elements=True`` it reproduces ASE, which checks the step itself.

``QUALIFICATION`` records what 1b-T09 showed with them; 1b-T13 enters it in the reference record.
"""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import Callable

import numpy as np
from ase import units
from ase.md.langevinbaoab import LangevinBAOAB
from scipy.linalg import expm

from fastmace.errors import FastMACEError

# inspect.getsource of the ASE methods the corrected reference relies on: the replaced drag, the
# Ornstein-Uhlenbeck step that applies it, and the step order. ASE 3.27.0 and 3.29.0 agree.
ASE_SOURCE_SHA256 = {
    "_barostat_BAOAB_OU_gamma_mod": "946dcb05dede78b17efacab1a054130f1bf2ef461253bbfb42aa79692bdbc5b5",
    "_BAOAB_OU": "0cf4fab9ce33ea72b667a258764e2f08763a21576f2a4e2d196ad50b9793b9fd",
    "step": "1afdedb2cf250921a87362d78efc9ed9496272f666156d97b8c7d545c77c1878",
}

QUALIFICATION_ID = "baoab-drag-quigley-probert-5b/1"


def check_ase_source() -> None:
    """Raise unless the installed ASE's ``LangevinBAOAB`` is the one the corrected reference patches."""
    import ase

    for name, expected in ASE_SOURCE_SHA256.items():
        found = hashlib.sha256(inspect.getsource(getattr(LangevinBAOAB, name)).encode()).hexdigest()
        if found != expected:
            raise FastMACEError(
                f"The corrected BAOAB reference does not fit ASE {ase.__version__}: "
                f"LangevinBAOAB.{name} changed.",
                why="The reference replaces one line of ASE's LangevinBAOAB and was checked against "
                    "ASE 3.27.0 and 3.29.0; a changed method may change what the patch means.",
                next_step="Re-qualify fastmace.reference.integrators against this ASE (plan task 1b-T09) "
                          "and update ASE_SOURCE_SHA256, or install ASE 3.27-3.29.")


def corrected_drag(p_eps, n_dof: int, barostat_mass: float, trace_all_elements: bool = False):
    """The full-cell drag rate on the atom velocities, (p_eps + Tr(p_eps)/N_dof · I)/W.

    ``trace_all_elements`` adds the trace term to all nine elements instead, as ASE 3.27 does.
    """
    p_eps = np.asarray(p_eps, dtype=np.float64)
    trace = np.trace(p_eps) / n_dof
    return (p_eps + (trace if trace_all_elements else trace * np.identity(3))) / barostat_mass


class CorrectedLangevinBAOAB(LangevinBAOAB):
    """ASE's ``LangevinBAOAB`` with the published full-cell drag; everything else, noise included, is ASE's."""

    def __init__(self, *args, **kwargs):
        check_ase_source()
        super().__init__(*args, **kwargs)

    def _barostat_BAOAB_OU_gamma_mod(self):
        if self.hydrostatic:
            return super()._barostat_BAOAB_OU_gamma_mod()
        return corrected_drag(self.p_eps, self.Ndof, self.barostat_mass)


# ---------------------------------------------------------------------------- the step from the equations

@dataclass(frozen=True)
class BAOABSettings:
    """Constants of a full-cell Langevin NPT step (ASE units: eV, Å, amu, ASE time).

    ``externalstress`` is the applied stress tensor in ASE's sign convention (a pressure P is -P · I).
    """

    dt: float
    gamma: float
    barostat_gamma: float
    barostat_mass: float
    kT: float
    externalstress: np.ndarray
    n_dof: int

    @classmethod
    def from_ase(cls, atoms, *, timestep, T_tau, P_tau, externalstress, temperature_K, P_mass_factor=1.0):
        """The settings ASE's ``LangevinBAOAB`` derives from the same arguments (heuristic barostat mass)."""
        n_atoms = len(atoms)
        # ASE's empirical heuristic, W = ((P_tau / 4) / C · N^(1/6))^2 with C = 1.66 fs
        mass = P_mass_factor * ((P_tau / 4.0) / (1.66 * units.fs) * n_atoms ** (1.0 / 6.0)) ** 2
        stress = np.asarray(externalstress, dtype=np.float64)
        if stress.shape == ():
            stress = stress * np.identity(3)
        elif stress.shape == (6,):
            xx, yy, zz, yz, xz, xy = stress
            stress = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        return cls(dt=float(timestep), gamma=1.0 / T_tau, barostat_gamma=1.0 / P_tau, barostat_mass=mass,
                   kT=units.kB * temperature_K, externalstress=stress, n_dof=3 * n_atoms)


@dataclass(frozen=True)
class BAOABStep:
    """Positions, velocities, masses, cell (rows are lattice vectors) and barostat momentum p_eps.

    ``forces``, ``stress`` (3 x 3, ASE sign) and ``force_eps`` belong to the end of the previous step;
    ``None`` evaluates them from this state, as an integrator does at its start.
    """

    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    cell: np.ndarray
    p_eps: np.ndarray
    forces: np.ndarray | None = None
    stress: np.ndarray | None = None
    force_eps: np.ndarray | None = None


def cell_force(velocities, masses, cell, stress, settings: BAOABSettings):
    """G = V (X - P_ext) + (1/N_dof) Σ m v² · I with X = (Σ m v vᵀ - V σ)/V, the barostat's force.

    ``stress`` is the virial stress σ in ASE's sign (positive under tension); ``externalstress`` is
    -P_ext. The kinetic term (1/N_dof) Σ m v² is the Quigley-Probert coupling of the cell to the
    particle momenta, the counterpart of the Tr(p_eps)/N_dof drag on the particles.
    """
    volume = abs(np.linalg.det(cell))
    kinetic = np.einsum("i,ia,ib->ab", masses, velocities, velocities)
    internal = (kinetic - volume * stress) / volume
    return volume * (internal + settings.externalstress) + np.trace(kinetic) / settings.n_dof * np.identity(3)


def baoab_step(step: BAOABStep, settings: BAOABSettings,
               evaluate: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
               xi, zeta, trace_all_elements: bool = False) -> BAOABStep:
    """One full-cell Langevin NPT step, split B A O A B for particles and cell together.

    Equations of motion (Quigley and Probert 2004; W the barostat mass, γ, γ_ε the frictions):
        dr/dt   = p/m + (p_ε/W) r
        dp/dt   = F - ((p_ε + Tr(p_ε)/N_dof · I)/W) p - γ p + noise
        dh/dt   = (p_ε/W) h        (h with lattice vectors as columns)
        dp_ε/dt = G - γ_ε p_ε + noise
    B advances the momenta (v by F/m, p_ε by G) by dt/2; A advances positions by v dt/2 and the cell,
    which scales the positions with it, by exp(p_ε dt/2W); O integrates the friction, the drag and the
    noise exactly over dt. ``xi`` (N x 3) and ``zeta`` (3 x 3) are standard normal numbers; zeta is
    symmetrised so the cell does not rotate. ``evaluate(positions, cell)`` returns forces and stress.
    """
    s = settings
    half = 0.5 * s.dt
    x, v, m = step.positions.copy(), step.velocities.copy(), step.masses
    cell, p_eps = step.cell.copy(), np.array(step.p_eps, dtype=np.float64)
    forces, stress = step.forces, step.stress
    if forces is None:
        forces, stress = evaluate(x, cell)
    force_eps = step.force_eps
    if force_eps is None:
        force_eps = cell_force(v, m, cell, stress, s)

    def cell_half_step(x, cell):
        # h(t + dt/2) = exp(p_ε dt/2W) h(t) in columns, i.e. rows times its transpose; atoms move with the cell
        transform = expm(half * p_eps / s.barostat_mass).T
        return x @ transform, cell @ transform

    # B
    p_eps = p_eps + half * force_eps
    v = v + half * forces / m[:, None]
    # A
    x, cell = cell_half_step(x, cell)
    x = x + half * v
    # O: dv = -(γ I + D) v dt + sqrt(2 γ kT/m) dW, solved exactly; then the barostat's own OU
    drag = corrected_drag(p_eps, s.n_dof, s.barostat_mass, trace_all_elements)
    v = v @ expm(-(s.gamma * np.identity(3) + drag) * s.dt).T
    v = v + np.sqrt(s.kT * (1.0 - np.exp(-2.0 * s.gamma * s.dt))) * xi / np.sqrt(m)[:, None]
    zeta = 0.5 * (zeta + zeta.T)
    p_eps = (p_eps * np.exp(-s.barostat_gamma * s.dt)
             + np.sqrt(s.kT * s.barostat_mass * (1.0 - np.exp(-2.0 * s.barostat_gamma * s.dt))) * zeta)
    # A
    x = x + half * v
    x, cell = cell_half_step(x, cell)
    # forces at the new geometry; the cell force with the velocities before the last B
    forces, stress = evaluate(x, cell)
    force_eps = cell_force(v, m, cell, stress, s)
    # B
    v = v + half * forces / m[:, None]
    p_eps = p_eps + half * force_eps
    return BAOABStep(x, v, m, cell, p_eps, forces, stress, force_eps)


# ---------------------------------------------------------------------------- the qualification record

_TESTS = "tests/cpu/test_integrator_qualification.py"

QUALIFICATION = {
    "id": QUALIFICATION_ID,
    "kind": "integrator correction",
    "integrator": "LangevinBAOAB, full cell",
    "correction": {
        "drag": "(p_eps + Tr(p_eps)/N_dof * I) / W",
        "source": "Quigley and Probert 2004, Eq. 5b",
        "ported_from": "ASE 3.27 LangevinBAOAB (trace term on all nine elements)",
        "hydrostatic": "identical to ASE; no correction",
    },
    "ase_variant": {
        "option": "IntegratorConfig.ase_trace_drag_all_elements=True",
        "reference": "ase.md.langevinbaoab.LangevinBAOAB",
        "cell_modes": ["hydrostatic", "full"],
        "steps": 300,
        "tolerance": 1e-9,
        # largest |difference| of cell, positions and velocities after 300 steps, CPU float64, tiny MACE
        # on NaCl (64 atoms), heating 0 -> 300 K under load (19 October 2026)
        "max_deviation": {"hydrostatic": 1.8e-14, "full": 1.1e-13},
        "tests": [f"{_TESTS}::test_ase_variant_reproduces_ase", "tests/cpu/test_integrators.py (Lennard-Jones)"],
    },
    "production": {
        "option": "IntegratorConfig.ase_trace_drag_all_elements=False (default)",
        "reference": "fastmace.reference.integrators.CorrectedLangevinBAOAB and baoab_step",
        "cell_modes": ["full"],
        "steps": 300,
        "tolerance": 1e-9,
        "max_deviation": {"full": 1.1e-13, "single_step_vs_paper": 1.6e-14},
        "tests": [f"{_TESTS}::test_production_variant_reproduces_the_corrected_reference",
                  f"{_TESTS}::test_production_step_equals_the_paper_step",
                  f"{_TESTS}::test_paper_step_equals_the_corrected_ase"],
    },
    # the two variants after the same 300 steps (N_dof = 192): the size of the correction
    "negative_control": {"test": f"{_TESTS}::test_the_variants_differ_by_the_correction",
                         "variant_difference": 4.3e-4},
    # SPEC §12.2 on the GPU (tests/gpu/test_ase_equivalence_gpu.py): murphy, RTX 4000 Ada, NAS
    # mace_cueq_latest, MH-1 matpes_r2scan, NaCl 3x3x3, plain cuEq float64, FixedPlanner (pipelined),
    # 19 October 2026; results/1b-T09-murphy-20261019/
    "gpu": {
        "langevin": {"protocol": "0 -> 400 -> 0 K in 1 + 8 ps, 18,000 steps, 50 K bins",
                     "reference": "ase.md.langevin.Langevin, MACECalculator cuEq float64",
                     "max_bin_dPE_meV_per_atom": 3.8e-11, "max_bin_dT_K": 3.5e-10,
                     "final_positions_A": 1.0e-11, "limit_dPE_meV_per_atom": 0.01},
        "baoab": {"protocol": "full cell, 300 K, 1 GPa, 2,000 steps",
                  "ase_variant_vs_ase_positions_A": 4.8e-13,
                  "production_vs_corrected_positions_A": 4.7e-13,
                  "variant_difference_positions_A": 6.1e-6},
    },
    "ase_source_sha256": ASE_SOURCE_SHA256,
    "report": "docs/validation/1b-T09-integrators.md",
}
