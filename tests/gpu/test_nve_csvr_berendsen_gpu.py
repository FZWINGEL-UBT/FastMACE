"""Plan task 4-T02 on the GPU: NVE, CSVR and Berendsen.

- **Capture (fast tier):** each scheme's half steps captured in CUDA graphs, with a capturable harmonic
  force between them, replay bitwise as the eager steps: no host synchronisation inside a half step
  (SPEC §10.1).
- **ASE equivalence (conductor's choice 2, acceptance tier):** MH-1 ``matpes_r2scan``, NaCl 3x3x3, plain
  cuEq float64, 4,000 steps of 0.5 fs at 300 K from a 300 K zero-momentum start, ``dof="ase"``: each
  integrator against its ASE class with MACE's calculator (NPT Berendsen against ASE's class reordered to
  one force call per step, as this port is; docstring of ``fastmace/integrators/berendsen.py``). Limits per
  500-step block: |ΔT| < 1e-6 K and |ΔPE| < 1e-7 meV/atom. The integrators run eagerly with the state on
  the GPU: the planner's dispatch of the new schemes (4-T01) and the front ends (4-T10) are not there yet,
  so the engine's run of them is 4-T10's and 4-T13's. These deterministic integrators share no noise that
  would hold the two runs together, and plain cuEq float64 differs between calculators by about 1e-13 A
  per 500 steps (``test_engine_gpu.py``, ``FLOAT64_LIMIT``): when chaotic growth exceeds the limits before
  4,000 steps, the test compares the first 1,000 steps at the same limits and records the growth rate
  (checkpoint 2).

Results go to ``FASTMACE_RESULTS`` when set.
"""

from __future__ import annotations

import dataclasses
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


ENGINE = _module("_4t02_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")
needs_mh1 = pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1")

SEED = 20261023
STEPS, SHORT_STEPS, BLOCK = 4000, 1000, 500
DT_FS, TEMPERATURE_K = 0.5, 300.0
PE_LIMIT = 1e-10     # eV/atom: 1e-7 meV/atom (SPEC §12.2)
T_LIMIT = 1e-6       # K, per block mean


def records():
    from fastmace import integrators

    return {
        "nve": integrators.NVE(timestep_fs=DT_FS, dof="ase"),
        "csvr": integrators.CSVR(timestep_fs=DT_FS, temperature_K=TEMPERATURE_K, tau_fs=50.0, dof="ase"),
        "nvt-berendsen": integrators.Berendsen(timestep_fs=DT_FS, temperature_K=TEMPERATURE_K, taut_fs=50.0,
                                               dof="ase"),
        "npt-berendsen": integrators.Berendsen(timestep_fs=DT_FS, temperature_K=TEMPERATURE_K, taut_fs=50.0,
                                               pressure_GPa=1.0, taup_fs=500.0, compressibility_per_GPa=0.04,
                                               dof="ase"),
        "inhomogeneous-berendsen": integrators.Berendsen(timestep_fs=DT_FS, temperature_K=TEMPERATURE_K,
                                                         taut_fs=50.0, pressure_GPa=1.0, taup_fs=500.0,
                                                         compressibility_per_GPa=0.04, mask=(1, 0, 1), dof="ase"),
    }


def build(record, state, **options):
    from fastmace.integrators import registry

    config = record.to_config(seed=SEED)
    if options:
        config = dataclasses.replace(config, **options)
    return registry.build_integrator(state, config, temperature_K=config.temperature_K)


# -------------------------------------------------------------------------- capture

def harmonic(state, x0, v0, k=0.5, bulk_modulus=0.5):
    """A tether and a volume spring, written into the state; graph-capturable."""
    from fastmace.neighbors.skin import det_3x3

    d = state.positions - x0
    state.energy.copy_((0.5 * k * (d * d).sum()).reshape(1))
    state.forces.copy_(-k * d)
    if state.stress is not None:
        volume = det_3x3(state.cell)
        eye = torch.eye(3, dtype=d.dtype, device=d.device)
        state.stress.copy_(k * (d.T @ d) / volume + bulk_modulus * (volume / v0 - 1.0) * eye)


def capture_state(name):
    from ase.build import bulk
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    from fastmace.state import SimulationState

    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2))
    atoms.rattle(0.05, seed=1)
    MaxwellBoltzmannDistribution(atoms, temperature_K=600, rng=np.random.default_rng(2))
    Stationary(atoms)
    variable = "npt" in name or "inhomogeneous" in name
    return SimulationState.from_atoms(atoms, "cuda", SEED, compute_stress=variable)


@cuda
@pytest.mark.parametrize("name", ["nve", "csvr", "nvt-berendsen", "npt-berendsen", "inhomogeneous-berendsen"])
def test_half_steps_replay_in_cuda_graphs_bitwise(name):
    from fastmace.neighbors.skin import det_3x3

    record = records()[name]
    runs = []
    for captured in (False, True):
        state = capture_state(name)
        x0, v0 = state.positions.clone(), det_3x3(state.cell).clone()
        dyn = build(record, state, noise_batch=7)
        harmonic(state, x0, v0)
        dyn.start(state)
        if captured:
            saved = {key: tensor.clone() for key, tensor in dyn.buffers(state).items()}
            saved.update(forces=state.forces.clone(), energy=state.energy.clone())
            if state.stress is not None:
                saved["stress"] = state.stress.clone()
            cursor, generator = state.rng.cursor, state.rng.generator.bit_generator.state
            batch = None if state.rng.batch is None else state.rng.batch.clone()
            dyn.prepare_step(state)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):  # warm-up on a side stream, as the engine does
                dyn.first_half(state)
                harmonic(state, x0, v0)
                dyn.second_half(state)
            torch.cuda.current_stream().wait_stream(stream)
            graphs = []
            for stage in (dyn.first_half, lambda s: harmonic(s, x0, v0), dyn.second_half):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    stage(state)
                graphs.append(graph)
            # undo the warm-up and the capture: restore every written tensor and the noise cursor
            restored = {**dyn.buffers(state), "forces": state.forces, "energy": state.energy}
            if state.stress is not None:
                restored["stress"] = state.stress
            for key, tensor in restored.items():
                tensor.copy_(saved[key])
            if batch is not None:  # the warm-up's prepare_step may have refilled the batch from the generator
                state.rng.batch.copy_(batch)
                state.rng.cursor = cursor
                state.rng.generator.bit_generator.state = generator
        for _ in range(50):
            dyn.prepare_step(state)
            if captured:
                for graph in graphs:
                    graph.replay()
            else:
                dyn.first_half(state)
                harmonic(state, x0, v0)
                dyn.second_half(state)
        torch.cuda.synchronize()
        runs.append({key: tensor.detach().cpu().clone() for key, tensor in dyn.buffers(state).items()})
    eager, captured = runs
    for key in eager:
        assert torch.equal(eager[key], captured[key]), (name, key)


# -------------------------------------------------------------------------- ASE equivalence (choice 2)

class ReorderedStep:
    """ASE 3.27's NPT Berendsen step with one force call: the first kick uses the previous step's forces."""

    def step(self, forces=None):
        from ase.md.nvtberendsen import NVTBerendsen

        if forces is None:
            forces = self.atoms.get_forces(md=True)
        NVTBerendsen.scale_velocities(self)
        self.scale_positions_and_cell()
        p = self.atoms.get_momenta()
        p += 0.5 * self.dt * forces
        if self.fix_com:
            p = p - p.sum(axis=0) / float(len(p))
        self.atoms.set_positions(self.atoms.get_positions() + self.dt * p / self.atoms.get_masses()[:, np.newaxis])
        self.atoms.set_momenta(p)
        forces = self.atoms.get_forces(md=True)
        self.atoms.set_momenta(self.atoms.get_momenta() + 0.5 * self.dt * forces)
        return forces


def ase_dynamics(name, atoms):
    from ase import units
    from ase.md.bussi import Bussi
    from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
    from ase.md.nvtberendsen import NVTBerendsen
    from ase.md.verlet import VelocityVerlet

    dt = DT_FS * units.fs
    if name == "nve":
        return VelocityVerlet(atoms, timestep=dt)
    if name == "csvr":
        return Bussi(atoms, timestep=dt, temperature_K=TEMPERATURE_K, taut=50.0 * units.fs,
                     rng=np.random.default_rng(SEED))
    if name == "nvt-berendsen":
        return NVTBerendsen(atoms, timestep=dt, temperature_K=TEMPERATURE_K, taut=50.0 * units.fs, fixcm=True)
    base = NPTBerendsen if name == "npt-berendsen" else Inhomogeneous_NPTBerendsen
    cls = type(f"Reordered{base.__name__}", (ReorderedStep, base), {})
    options = dict(timestep=dt, temperature_K=TEMPERATURE_K, taut=50.0 * units.fs, pressure_au=1.0 * units.GPa,
                   taup=500.0 * units.fs, compressibility_au=0.04 / units.GPa, fixcm=True)
    if name == "inhomogeneous-berendsen":
        options["mask"] = (1, 0, 1)
    return cls(atoms, **options)


def mh1_calculator():
    from mace.calculators import MACECalculator

    from fastmace.model import labels

    label = labels.LABELS["mh1-r2scan"]
    return MACECalculator(model_paths=str(ENGINE._mh1_path()), head=label.head, device="cuda",
                          default_dtype="float64", enable_cueq=True)


def blocks(rows, steps):
    rows = np.asarray(rows)[:steps]
    return [(rows[i:i + BLOCK, 1].mean(), rows[i:i + BLOCK, 2].mean()) for i in range(0, steps, BLOCK)]


@pytest.mark.slow
@pytest.mark.acceptance
@cuda
@needs_mh1
@pytest.mark.parametrize("name", ["nve", "csvr", "nvt-berendsen", "npt-berendsen", "inhomogeneous-berendsen"])
def test_matches_ase_on_mh1(name):
    from ase import units
    from ase.build import bulk
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    from fastmace.state import SimulationState

    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 3))
    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, rng=np.random.default_rng(11))
    Stationary(atoms)
    n = len(atoms)
    reference = atoms.copy()
    reference.calc = mh1_calculator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ase_dyn = ase_dynamics(name, reference)
    record = records()[name]
    variable = record.to_config(seed=0).variable_cell
    state = SimulationState.from_atoms(atoms, "cuda", SEED, compute_stress=variable)
    ours = atoms.copy()
    ours.calc = mh1_calculator()

    def forces():
        ours.set_cell(state.cell.cpu().numpy())
        ours.set_positions(state.positions.cpu().numpy())
        state.forces.copy_(torch.from_numpy(ours.get_forces()))
        state.energy.copy_(torch.tensor([ours.get_potential_energy()], dtype=torch.float64))
        if state.stress is not None:
            state.stress.copy_(torch.from_numpy(ours.get_stress(voigt=False)))

    dyn = build(record, state)
    forces()
    dyn.start(state)
    ase_rows, our_rows, growth = [], [], []
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
            growth.append([step, float(np.abs(state.positions.cpu().numpy() - reference.positions).max())])
    seconds = time.perf_counter() - start

    def deviations(steps):
        pairs = zip(blocks(ase_rows, steps), blocks(our_rows, steps))
        return [(abs(o[0] - a[0]), abs(o[1] - a[1])) for a, o in pairs]

    full = deviations(STEPS)
    passed_full = all(dt < T_LIMIT and dpe < PE_LIMIT for dt, dpe in full)
    short = deviations(SHORT_STEPS)
    steps_with_growth = [(s, d) for s, d in growth if d > 0]
    rate = None
    if len(steps_with_growth) >= 2:
        (s0, d0), (s1, d1) = steps_with_growth[0], steps_with_growth[-1]
        rate = math.log(d1 / d0) / ((s1 - s0) * DT_FS) if s1 > s0 else None  # per fs
    summary = {"scheme": name, "steps": STEPS, "passed_4000": passed_full,
               "blocks_dT_K": [dt for dt, _ in full], "blocks_dPE_meV_per_atom": [1e3 * dpe for _, dpe in full],
               "position_difference_A_every_100": growth, "growth_rate_per_fs": rate,
               "final_T_K": [ase_rows[-1][1], our_rows[-1][1]], "seconds": seconds,
               "gpu": torch.cuda.get_device_name()}
    ENGINE.report(f"4-T02-{name}-ase-equivalence", summary)
    if not passed_full:  # choice 2's shortened horizon
        assert all(dt < T_LIMIT for dt, _ in short), summary
        assert all(dpe < PE_LIMIT for _, dpe in short), summary
