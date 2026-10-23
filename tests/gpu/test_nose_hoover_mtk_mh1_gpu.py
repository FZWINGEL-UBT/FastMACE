"""Plan task 4-T03 on MH-1: ASE equivalence (conductor's choice 2) and the extended energy over 20 ps.

- **ASE equivalence (acceptance tier):** MH-1 ``matpes_r2scan``, NaCl 3x3x3, plain cuEq float64, 4,000 steps of
  0.5 fs at 300 K from a 300 K zero-momentum start, ``dof="ase"`` and ``fixcm=False``: ``NoseHooverChainNVT``
  (tdamp 50 fs) and ``MTK`` hydrostatic and full (1 GPa, tdamp 50 fs, pdamp 500 fs) against ASE 3.27's
  ``NoseHooverChainNVT``, ``IsotropicMTKNPT`` and ``MTKNPT`` with MACE's calculator. Limits per 500-step
  block: |ΔT| < 1e-6 K and |ΔPE| < 1e-7 meV/atom. The integrators run eagerly with the state on the GPU and
  the forces (and stress) from a second MACE calculator, as 4-T02's test does. These deterministic
  integrators share no noise that would hold the runs together, and plain cuEq float64 differs between
  calculators by about 1e-13 A per 500 steps (``test_engine_gpu.py``, ``FLOAT64_LIMIT``): when chaotic
  growth exceeds the limits before 4,000 steps, the first 1,000 steps are compared at the same limits and
  the growth rate is recorded (checkpoint 2).
- **Extended energy (recorded, plan 4-T03):** ``fm.Simulation`` with the planner's default plan, NaCl 3x3x3,
  MH-1, 20 ps of 1 fs at 300 K (NHC; MTK at 1 GPa): the conserved quantity every 1,000 steps and its drift.

Results go to ``FASTMACE_RESULTS`` when set.
"""

from __future__ import annotations

import importlib.util
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_4t03_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")
needs_mh1 = pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1")

SEED = 20261023
STEPS, SHORT_STEPS, BLOCK = 4000, 1000, 500
DT_FS, TEMPERATURE_K, PRESSURE_GPA = 0.5, 300.0, 1.0
TDAMP_FS, PDAMP_FS = 50.0, 500.0
PE_LIMIT = 1e-10     # eV/atom: 1e-7 meV/atom (SPEC §12.2)
T_LIMIT = 1e-6       # K, per block mean
SCHEMES = ("nhc", "hydrostatic", "full")


def record(name, timestep_fs=DT_FS, dof="ase", fixcm=False):
    from fastmace import integrators

    common = dict(timestep_fs=timestep_fs, temperature_K=TEMPERATURE_K, tdamp_fs=TDAMP_FS, dof=dof, fixcm=fixcm)
    if name == "nhc":
        return integrators.NoseHooverChain(**common)
    return integrators.MTK(pressure_GPa=PRESSURE_GPA, pdamp_fs=PDAMP_FS, cell_mode=name, **common)


def ase_dynamics(name, atoms):
    from ase import units
    from ase.md.nose_hoover_chain import MTKNPT, IsotropicMTKNPT, NoseHooverChainNVT

    common = dict(timestep=DT_FS * units.fs, temperature_K=TEMPERATURE_K, tdamp=TDAMP_FS * units.fs)
    if name == "nhc":
        return NoseHooverChainNVT(atoms, **common)
    cls = IsotropicMTKNPT if name == "hydrostatic" else MTKNPT
    return cls(atoms, pressure_au=PRESSURE_GPA * units.GPa, pdamp=PDAMP_FS * units.fs, **common)


def mh1_calculator():
    from mace.calculators import MACECalculator

    from fastmace.model import labels

    label = labels.LABELS["mh1-r2scan"]
    return MACECalculator(model_paths=str(ENGINE._mh1_path()), head=label.head, device="cuda",
                          default_dtype="float64", enable_cueq=True)


def nacl():
    from ase.build import bulk
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 3))
    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, rng=np.random.default_rng(11))
    Stationary(atoms)
    return atoms


def blocks(rows, steps):
    rows = np.asarray(rows)[:steps]
    return [(rows[i:i + BLOCK, 1].mean(), rows[i:i + BLOCK, 2].mean()) for i in range(0, steps, BLOCK)]


@pytest.mark.slow
@pytest.mark.acceptance
@cuda
@needs_mh1
@pytest.mark.parametrize("name", SCHEMES)
def test_matches_ase_on_mh1(name):
    from ase import units

    from fastmace.integrators.base import CONSERVED_KEY
    from fastmace.integrators.registry import build_integrator
    from fastmace.state import SimulationState

    atoms = nacl()
    n = len(atoms)
    reference = atoms.copy()
    reference.calc = mh1_calculator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ase_dyn = ase_dynamics(name, reference)
    state = SimulationState.from_atoms(atoms, "cuda", SEED, compute_stress=name != "nhc")
    ours = atoms.copy()
    ours.calc = mh1_calculator()

    def forces():
        ours.set_cell(state.cell.cpu().numpy())
        ours.set_positions(state.positions.cpu().numpy())
        state.forces.copy_(torch.from_numpy(ours.get_forces()))
        state.energy.copy_(torch.tensor([ours.get_potential_energy()], dtype=torch.float64))
        if state.stress is not None:
            state.stress.copy_(torch.from_numpy(ours.get_stress(voigt=False)))

    dyn = build_integrator(state, record(name).to_config(seed=SEED))
    forces()
    dyn.start(state)
    ase_rows, our_rows, growth, conserved = [], [], [], []
    start = time.perf_counter()
    for step in range(1, STEPS + 1):
        ase_dyn.run(1)
        dyn.prepare_step(state)
        dyn.first_half(state)
        forces()
        dyn.second_half(state)
        kinetic = float(0.5 * (state.masses * state.velocities ** 2).sum())
        ase_rows.append([step, 2 * reference.get_kinetic_energy() / (3 * n * units.kB),
                         reference.get_potential_energy() / n])
        our_rows.append([step, 2 * kinetic / (3 * n * units.kB), float(state.energy[0]) / n])
        if step % 100 == 0:
            growth.append([step, float(np.abs(state.positions.cpu().numpy() - reference.positions).max()),
                           float(np.abs(state.cell.cpu().numpy() - reference.cell.array).max())])
            conserved.append([step, ase_dyn.get_conserved_energy() / n,
                              float(dyn.output_values(state)[CONSERVED_KEY][0]) / n])
    seconds = time.perf_counter() - start

    def deviations(steps):
        pairs = zip(blocks(ase_rows, steps), blocks(our_rows, steps))
        return [(abs(o[0] - a[0]), abs(o[1] - a[1])) for a, o in pairs]

    full = deviations(STEPS)
    passed_full = all(dt < T_LIMIT and dpe < PE_LIMIT for dt, dpe in full)
    short = deviations(SHORT_STEPS)
    steps_with_growth = [(s, d) for s, d, _ in growth if d > 0]
    rate = None
    if len(steps_with_growth) >= 2:
        (s0, d0), (s1, d1) = steps_with_growth[0], steps_with_growth[-1]
        rate = math.log(d1 / d0) / ((s1 - s0) * DT_FS) if s1 > s0 else None  # per fs
    summary = {"scheme": name, "steps": STEPS, "passed_4000": passed_full,
               "blocks_dT_K": [dt for dt, _ in full], "blocks_dPE_meV_per_atom": [1e3 * dpe for _, dpe in full],
               "position_and_cell_difference_A_every_100": growth, "growth_rate_per_fs": rate,
               "conserved_eV_per_atom_every_100_ase_ours": conserved,
               "final_T_K": [ase_rows[-1][1], our_rows[-1][1]],
               "final_volume_A3": [reference.get_volume(), float(abs(torch.linalg.det(state.cell.cpu())))],
               "seconds": seconds, "gpu": torch.cuda.get_device_name()}
    ENGINE.report(f"4-T03-{name}-ase-equivalence", summary)
    if not passed_full:  # choice 2's shortened horizon
        assert all(dt < T_LIMIT for dt, _ in short), summary
        assert all(dpe < PE_LIMIT for _, dpe in short), summary


@pytest.mark.slow
@pytest.mark.acceptance
@cuda
@needs_mh1
@pytest.mark.parametrize("name", SCHEMES)
def test_extended_energy_over_20_ps(name):
    """Recorded, not acceptance (plan 4-T03): the drift of the conserved quantity in the default plan."""
    import fastmace as fm
    from fastmace.integrators.base import CONSERVED_KEY

    atoms = nacl()
    n = len(atoms)
    sim = fm.Simulation(atoms, fm.ForceField(fm.load("mh1-r2scan"), atoms),
                        integrator=record(name, timestep_fs=1.0, dof="auto", fixcm=True), seed=SEED,
                        workload=fm.Workload(max_temperature_K=TEMPERATURE_K))
    engine = sim._engine

    def sample():
        values = engine.integrator.output_values(engine.state)
        return float(values[CONSERVED_KEY][0]) / n, float(values["kinetic"][0])

    rows = [[0, *sample()]]
    start = time.perf_counter()
    for chunk in range(20):
        sim.run(steps=1000)
        rows.append([1000 * (chunk + 1), *sample()])
    seconds = time.perf_counter() - start
    values = np.array([row[1] for row in rows])
    slope = np.polyfit(np.array([row[0] for row in rows]) * 1e-3, values, 1)[0]  # eV/atom per ps
    summary = {"scheme": name, "ps": 20, "timestep_fs": 1.0, "plan": str(sim.plan.precision),
               "conserved_eV_per_atom_every_ps": rows, "drift_meV_per_atom_per_ps": 1e3 * slope,
               "range_meV_per_atom": 1e3 * float(np.ptp(values)), "seconds": seconds,
               "gpu": torch.cuda.get_device_name()}
    ENGINE.report(f"4-T03-{name}-extended-energy-20ps", summary)
    assert np.all(np.isfinite(values)), summary
