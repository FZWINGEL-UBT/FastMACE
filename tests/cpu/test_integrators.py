"""SimulationState and the integrators on the CPU: ASE equivalence, the noise stream, the baseline, restarts.

Ported from the research engine's ``tests/core/test_gpu_npt.py`` and the integrator parts of
``tests/core/test_gpu_langevin.py`` (ResearchModule tag research-engine-baseline-20261017), with
their expected values. The integrators never call a force field, so these tests drive the half
steps themselves (``run_md``, the eager order of the engine) with fake force fields: ASE's
non-smooth Lennard-Jones on the host for the ASE comparisons, a harmonic tether in torch elsewhere.

``FASTMACE_BASELINE`` (the export of that tag) adds bitwise comparisons with the research
engine's eager ``GPULangevin`` and ``GPULangevinBAOAB`` steps.
"""

import importlib
import os
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import units
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.constraints import FixAtoms
from ase.md.langevin import Langevin as ASELangevin
from ase.md.langevinbaoab import LangevinBAOAB as ASELangevinBAOAB

from fastmace.config import IntegratorConfig
from fastmace.integrators.baoab import LangevinBAOAB, barostat_mass, expm_3x3, external_stress
from fastmace.integrators.langevin import Langevin
from fastmace.io import restart as restart_io
from fastmace.neighbors.skin import det_3x3
from fastmace.state import SimulationState

SIGMA, EPSILON, RC = 3.4, 0.0104, 7.0
SEED = 7
STEPS = 300


def argon(seed=0, displacement=0.05):
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat((3, 3, 3))
    atoms.rattle(displacement, seed=seed)
    return atoms


def lj_calculator():
    return LennardJones(sigma=SIGMA, epsilon=EPSILON, rc=RC, smooth=False)


def nvt_config(**options):
    # ASE's noise stream: ASE's projection, explicitly (an IntegratorConfig outside a plan reads "auto" so; 3-T22c)
    return IntegratorConfig(ensemble="nvt", timestep_fs=2.0, friction=0.02, **{"fixcm_projection": "ase", **options})


def npt_config(hydrostatic, **options):
    """The settings of the research engine's NPT trajectory tests: T_tau 100 fs, P_tau 400 fs, -0.3 GPa."""
    options.setdefault("pressure_GPa", 0.3)
    return IntegratorConfig(ensemble="npt", timestep_fs=2.0, friction=1.0 / (100 * units.fs),
                            barostat_tau_fs=400.0, cell_mode="hydrostatic" if hydrostatic else "full", **options)


# -- fake force fields: the engine's force stage, writing into the state --

class ASEForces:
    """ASE's Lennard-Jones on a host copy of the state (energy, forces and, if allocated, stress)."""

    def __init__(self, atoms):
        self.atoms = atoms.copy()
        self.atoms.calc = lj_calculator()
        self.calls = 0

    def __call__(self, state):
        self.calls += 1
        self.atoms.set_cell(state.cell.numpy().copy())
        self.atoms.set_positions(state.positions.numpy().copy())
        state.forces.copy_(torch.from_numpy(self.atoms.get_forces()))
        state.energy.copy_(torch.tensor([self.atoms.get_potential_energy()], dtype=torch.float64))
        if state.stress is not None:
            state.stress.copy_(torch.from_numpy(self.atoms.get_stress(voigt=False)))


def harmonic(x, cell, x0, k=0.5, bulk_modulus=0.5, v0=None):
    """A tether to x0 and a volume spring: energy (1,), forces, a 3 x 3 stress; graph-capturable."""
    d = x - x0
    energy = (0.5 * k * (d * d).sum()).reshape(1)
    forces = -k * d
    volume = det_3x3(cell)
    eye = torch.eye(3, dtype=x.dtype, device=x.device)
    stress = k * (d.T @ d) / volume + bulk_modulus * (volume / v0 - 1.0) * eye
    return energy, forces, stress


class HarmonicForces:
    def __init__(self, state):
        self.x0 = state.positions.clone()
        self.v0 = float(det_3x3(state.cell))
        self.calls = 0

    def __call__(self, state):
        self.calls += 1
        energy, forces, stress = harmonic(state.positions, state.cell, self.x0, v0=self.v0)
        state.forces.copy_(forces)
        state.energy.copy_(energy)
        if state.stress is not None:
            state.stress.copy_(stress)


def start(state, integrator, field):
    """What the engine does at construction: the initial force evaluation, then ``integrator.start``."""
    field(state)
    integrator.start(state)


def md_step(state, integrator, field):
    """One eager MD step in the engine's order (docs/design/interfaces.md, "One MD step")."""
    integrator.prepare_step(state)
    integrator.first_half(state)
    field(state)
    integrator.second_half(state)
    state.step += 1


def run_md(state, integrator, field, steps, targets=None):
    for step in range(1, steps + 1):
        if targets is not None:
            integrator.set_temperature(targets(step))
        md_step(state, integrator, field)


def heating(steps):
    return lambda step: 150.0 * step / steps


def new_state(atoms, compute_stress=False, seed=SEED):
    return SimulationState.from_atoms(atoms, "cpu", seed, compute_stress=compute_stress)


# -------------------------------------------------------------------------- SimulationState

def test_state_from_atoms():
    atoms = argon(seed=1)
    atoms.set_velocities(np.random.default_rng(0).standard_normal((len(atoms), 3)) * 0.01)
    state = SimulationState.from_atoms(atoms, "cpu", 5)
    assert state.natoms == len(atoms) == 108
    for name, shape in dict(positions=(108, 3), velocities=(108, 3), masses=(108, 1), cell=(3, 3),
                            forces=(108, 3), energy=(1,)).items():
        tensor = getattr(state, name)
        assert tensor.dtype == torch.float64 and tuple(tensor.shape) == shape, name
    assert state.species.dtype == torch.int64 and torch.equal(state.species, torch.from_numpy(atoms.numbers))
    np.testing.assert_array_equal(state.positions.numpy(), atoms.positions)
    np.testing.assert_array_equal(state.velocities.numpy(), atoms.get_velocities())
    np.testing.assert_array_equal(state.masses.numpy()[:, 0], atoms.get_masses())
    np.testing.assert_array_equal(state.cell.numpy(), atoms.cell.array)
    assert state.stress is None and state.step == 0
    assert not state.forces.any() and not state.energy.any()
    assert state.rng.batch is None
    np.testing.assert_array_equal(state.rng.generator.standard_normal(4), np.random.default_rng(5).standard_normal(4))
    assert torch.equal(state.momenta, state.velocities * state.masses)
    stressed = SimulationState.from_atoms(atoms, "cpu", 5, compute_stress=True)
    assert stressed.stress.shape == (3, 3) and not stressed.stress.any()


def test_from_atoms_refuses_constraints():
    atoms = argon()
    atoms.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(ValueError, match="onstraints"):
        SimulationState.from_atoms(atoms, "cpu", 0)


def test_npt_refuses_a_state_without_stress():
    """The research engine refuses a fixed-cell force field; here the state carries no stress."""
    with pytest.raises(ValueError, match="stress"):
        LangevinBAOAB(new_state(argon()), npt_config(True))


def test_ensembles_are_checked():
    with pytest.raises(ValueError, match="nvt"):
        Langevin(new_state(argon()), npt_config(True))
    with pytest.raises(ValueError, match="npt"):
        LangevinBAOAB(new_state(argon(), compute_stress=True), nvt_config())


# -------------------------------------------------------------------------- BAOAB helpers

def test_expm_matches_scipy():
    from scipy.linalg import expm
    rng = np.random.default_rng(0)
    for scale in (1e-6, 1e-3, 0.1, 1.0):
        a = rng.standard_normal((3, 3)) * scale
        np.testing.assert_allclose(expm_3x3(torch.tensor(a)).numpy(), expm(a), rtol=1e-13, atol=1e-15)


def test_barostat_mass_and_external_stress_match_ase():
    atoms = argon()
    atoms.calc = lj_calculator()
    for hydrostatic, stress in ((True, -0.3 * units.GPa), (False, -0.3 * units.GPa),
                                (False, [1.0, 2.0, 3.0]), (False, [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]),
                                (False, np.arange(9.0).reshape(3, 3))):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dyn = ASELangevinBAOAB(atoms, timestep=units.fs, temperature_K=100, T_tau=100 * units.fs,
                                   P_tau=400 * units.fs, externalstress=stress, hydrostatic=hydrostatic,
                                   P_mass_factor=1.7)
        assert barostat_mass(400 * units.fs, len(atoms), P_mass_factor=1.7) == pytest.approx(dyn.barostat_mass,
                                                                                            rel=1e-15)
        np.testing.assert_array_equal(external_stress(stress, hydrostatic), dyn.externalstress)
    assert barostat_mass(1.0, 10, P_mass=3.0) == 3.0
    with pytest.raises(ValueError):
        barostat_mass(0.0, 10)
    with pytest.raises(ValueError):
        external_stress([1.0, 2.0, 3.0], True)


# -------------------------------------------------------------------------- noise streams

@pytest.mark.parametrize("noise_batch", [1, 7, 1000])
def test_langevin_noise_is_ases_stream(noise_batch):
    """Per step xi, then eta, each (N, 3), in ASE's order, whatever the batch size."""
    state = new_state(argon())
    dyn = Langevin(state, nvt_config(noise_batch=noise_batch))
    assert tuple(state.rng.batch.shape) == (noise_batch, 2, 108, 3)
    assert state.rng.cursor == noise_batch
    reference = np.random.default_rng(SEED)
    for _ in range(20):
        dyn.prepare_step(state)
        np.testing.assert_array_equal(dyn.xi.numpy(), reference.standard_normal(size=(108, 3)))
        np.testing.assert_array_equal(dyn.eta.numpy(), reference.standard_normal(size=(108, 3)))
    assert state.rng.cursor == (20 - 1) % noise_batch + 1


@pytest.mark.parametrize("cell_mode,width", [("hydrostatic", 1), ("full", 9), ("disabled", 0)])
def test_baoab_noise_rows_are_ases_stream(cell_mode, width):
    """BAOAB's rows are 3N + {0, 1, 9} wide: rng.normal((N, 3)), then 1 or (3, 3) cell numbers."""
    state = new_state(argon(), compute_stress=True)
    dyn = LangevinBAOAB(state, npt_config(cell_mode == "hydrostatic", noise_batch=7),
                        disable_cell_langevin=cell_mode == "disabled")
    assert dyn.cell_numbers == width
    assert tuple(state.rng.batch.shape) == (7, 3 * 108 + width)
    reference = np.random.default_rng(SEED)
    for _ in range(16):
        dyn.prepare_step(state)
        np.testing.assert_array_equal(dyn.xi.numpy(), reference.normal(size=(108, 3)))
        if width == 1:
            assert dyn.zeta.numpy()[0] == reference.normal()
        elif width == 9:
            np.testing.assert_array_equal(dyn.zeta.numpy().reshape(3, 3), reference.normal(size=(3, 3)))


# -------------------------------------------------------------------------- ASE trajectories

def run_langevin_pair(noise_batch=256, steps=STEPS):
    reference, test = argon(seed=3), argon(seed=3)
    reference.calc = lj_calculator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # ASE >= 3.28 deprecates fixcm; the research engine uses it
        ase_dyn = ASELangevin(reference, timestep=2 * units.fs, friction=0.02, fixcm=True, temperature_K=0,
                              rng=np.random.default_rng(SEED))
    state = new_state(test)
    field = ASEForces(test)
    dyn = Langevin(state, nvt_config(noise_batch=noise_batch))
    start(state, dyn, field)
    for step in range(1, steps + 1):
        target = 150.0 * step / steps
        ase_dyn.set_temperature(temperature_K=target)
        dyn.set_temperature(target)
        ase_dyn.run(1)
        md_step(state, dyn, field)
    return reference, state, field


@pytest.mark.parametrize("noise_batch", [1, 7, 1000])
def test_langevin_trajectory_matches_ase(noise_batch):
    reference, state, field = run_langevin_pair(noise_batch)
    assert field.calls == STEPS + 1  # one force call per step
    assert state.step == STEPS
    np.testing.assert_allclose(state.positions.numpy(), reference.positions, atol=1e-9)
    np.testing.assert_allclose(state.velocities.numpy(), reference.get_velocities(), atol=1e-9)
    assert float(state.energy[0]) == pytest.approx(reference.get_potential_energy(), abs=1e-8)
    assert reference.get_temperature() > 50  # the thermostat really heated it


def run_baoab_pair(hydrostatic, steps=STEPS, noise_batch=256, trace_all=True, disable_cell_langevin=False,
                   temperature=None):
    """ASE LangevinBAOAB and FastMACE's from the same state and seed (heating, -0.3 GPa or anisotropic)."""
    reference, test = argon(seed=3), argon(seed=3)
    reference.calc = lj_calculator()
    stress = -0.3 * units.GPa
    externalstress = stress if hydrostatic else [stress, 1.5 * stress, stress, 0, 0, 0]
    start_temperature = 0 if temperature is None else temperature
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ASE's heuristic-mass notice
        ase_dyn = ASELangevinBAOAB(reference, timestep=2 * units.fs, T_tau=100 * units.fs, P_tau=400 * units.fs,
                                   hydrostatic=hydrostatic, externalstress=externalstress,
                                   disable_cell_langevin=disable_cell_langevin, temperature_K=start_temperature,
                                   rng=np.random.default_rng(SEED))
    state = new_state(test, compute_stress=True)
    field = ASEForces(test)
    dyn = LangevinBAOAB(state, npt_config(hydrostatic, noise_batch=noise_batch, ase_trace_drag_all_elements=trace_all),
                        externalstress=externalstress, disable_cell_langevin=disable_cell_langevin,
                        temperature_K=start_temperature)
    start(state, dyn, field)
    for step in range(1, steps + 1):
        if temperature is None:
            target = 150.0 * step / steps
            ase_dyn.set_temperature(temperature_K=target)
            dyn.set_temperature(target)
        ase_dyn.run(1)
        md_step(state, dyn, field)
    return reference, state, field, dyn


def check_baoab(reference, state, start_volume):
    np.testing.assert_allclose(state.cell.numpy(), reference.cell.array, atol=1e-9)
    np.testing.assert_allclose(state.positions.numpy(), reference.positions, atol=1e-9)
    np.testing.assert_allclose(state.velocities.numpy(), reference.get_velocities(), atol=1e-9)
    assert float(state.energy[0]) == pytest.approx(reference.get_potential_energy(), abs=1e-8)
    assert reference.get_temperature() > 30                          # the thermostat heated it
    assert abs(reference.get_volume() / start_volume - 1) > 2e-3      # the barostat moved the cell


@pytest.mark.parametrize("noise_batch", [256, 7])
@pytest.mark.parametrize("hydrostatic", [True, False], ids=["hydrostatic", "full"])
def test_baoab_trajectory_matches_ase(hydrostatic, noise_batch):
    """ASE_TRACE_DRAG_ALL_ELEMENTS=True reproduces ASE's LangevinBAOAB (the research engine's tests)."""
    reference, state, field, dyn = run_baoab_pair(hydrostatic, noise_batch=noise_batch)
    assert field.calls == STEPS + 1
    check_baoab(reference, state, argon().get_volume())
    if not hydrostatic:  # the anisotropic load changed the cell shape
        lengths = reference.cell.lengths()
        assert abs(lengths[1] - lengths[0]) > 1e-3


def test_trace_drag_on_the_diagonal_differs_from_ase_only_slightly():
    """The default (Quigley-Probert) full-cell drag: a small departure from ASE's broadcast trace term."""
    reference, state, _, _ = run_baoab_pair(False, steps=20, trace_all=False, temperature=100)
    difference = np.abs(state.positions.numpy() - reference.positions).max()
    assert 1e-12 < difference < 1e-4


def test_trace_drag_is_irrelevant_for_hydrostatic_cells():
    reference, state, _, _ = run_baoab_pair(True, steps=20, trace_all=False, temperature=100)
    np.testing.assert_allclose(state.positions.numpy(), reference.positions, atol=1e-9)


def test_disabled_cell_langevin_draws_no_cell_numbers():
    reference, state, _, dyn = run_baoab_pair(True, steps=50, disable_cell_langevin=True)
    assert dyn.cell_numbers == 0
    np.testing.assert_allclose(state.positions.numpy(), reference.positions, atol=1e-9)
    np.testing.assert_allclose(state.cell.numpy(), reference.cell.array, atol=1e-9)


# -------------------------------------------------------------------------- the integrator contract

def integrators(atoms):
    yield "nvt", Langevin, new_state(atoms), nvt_config(noise_batch=7)
    for hydrostatic in (True, False):
        yield ("npt-hydrostatic" if hydrostatic else "npt-full"), LangevinBAOAB, \
            new_state(atoms, compute_stress=True), npt_config(hydrostatic, noise_batch=7)


def test_addresses_stay_fixed_and_targets_are_device_scalars():
    for name, cls, state, config in integrators(argon(seed=2)):
        dyn = cls(state, config)
        field = HarmonicForces(state)
        start(state, dyn, field)
        tensors = {**dyn.buffers(state), "sqrt_kT": dyn.sqrt_kT, "noise": state.rng.batch}
        pointers = {key: value.data_ptr() for key, value in tensors.items()}
        run_md(state, dyn, field, 10, heating(10))
        dyn.set_temperature(300.0)
        assert float(dyn.sqrt_kT) == pytest.approx(np.sqrt(units.kB * 300.0), rel=1e-15)
        assert {key: value.data_ptr() for key, value in tensors.items()} == pointers, name
        for key in ("positions", "velocities"):
            assert dyn.buffers(state)[key] is getattr(state, key), name
        assert dyn.variable_cell == (cls is LangevinBAOAB)
        if dyn.variable_cell:
            assert dyn.buffers(state)["cell"] is state.cell
        mid = dyn.mid_step_tensors(state)
        assert mid[0] is state.velocities
        assert len(mid) == (2 if dyn.variable_cell else 1)


def test_output_values():
    for name, cls, state, config in integrators(argon(seed=2)):
        dyn = cls(state, config)
        field = HarmonicForces(state)
        start(state, dyn, field)
        run_md(state, dyn, field, 3, heating(3))
        values = dyn.output_values(state)
        expected = {"positions", "momenta", "forces", "energy", "kinetic"} | (
            {"cell", "stress"} if dyn.variable_cell else set())
        assert set(values) == expected, name
        assert torch.equal(values["momenta"], state.velocities * state.masses)
        assert torch.equal(values["kinetic"], (0.5 * (state.masses * state.velocities ** 2).sum()).reshape(1))
        assert values["energy"].shape == (1,)


def test_state_dict_round_trip():
    for name, cls, state, config in integrators(argon(seed=2)):
        dyn = cls(state, config)
        field = HarmonicForces(state)
        start(state, dyn, field)
        run_md(state, dyn, field, 5, heating(5))
        saved = dyn.state_dict()
        if not dyn.variable_cell:
            assert saved == {}
            dyn.load_state_dict({})
            with pytest.raises(ValueError):
                dyn.load_state_dict({"p_eps": np.zeros(())})
            continue
        # 1b: the barostat state is in SimulationState (SPEC §10.6), so BAOAB has no state of its own.
        assert saved == {}
        assert dyn.p_eps is state.p_eps and dyn.force_eps is state.force_eps
        dyn.load_state_dict({})
        # a 1a state_dict (p_eps, force_eps) still loads, into the state's tensors
        legacy = {"p_eps": state.p_eps.numpy().copy(), "force_eps": state.force_eps.numpy().copy()}
        pointer = dyn.p_eps.data_ptr()
        dyn.p_eps.zero_()
        dyn.force_eps.zero_()
        dyn.load_state_dict(legacy)
        assert dyn.p_eps.data_ptr() == pointer and state.p_eps is dyn.p_eps
        np.testing.assert_array_equal(state.p_eps.numpy(), legacy["p_eps"])
        np.testing.assert_array_equal(state.force_eps.numpy(), legacy["force_eps"])
        with pytest.raises(ValueError):
            dyn.load_state_dict({"p_eps": np.zeros((2, 2)), "force_eps": legacy["force_eps"]})
        with pytest.raises(ValueError):
            dyn.load_state_dict({"p_eps": legacy["p_eps"]})


def assert_same_run(a, b):
    for name in ("positions", "velocities", "cell", "forces", "energy"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    if a.stress is not None:
        assert torch.equal(a.stress, b.stress)
    assert a.step == b.step


@pytest.mark.parametrize("which", ["nvt", "npt-hydrostatic", "npt-full"])
def test_restart_round_trip(tmp_path, which):
    """Run, write a restart mid-batch, resume in a fresh state and integrator: bitwise the uninterrupted run."""
    atoms = argon(seed=4)
    (name, cls, state, config), = [case for case in integrators(atoms) if case[0] == which]
    dyn = cls(state, config)
    field = HarmonicForces(state)
    start(state, dyn, field)
    run_md(state, dyn, field, 10, heating(30))
    assert 0 < state.rng.cursor < config.noise_batch
    path = tmp_path / restart_io.RESTART_NAME
    restart_io.write_restart(path, state, engine="gpu", integrator_state=dyn.state_dict(),
                             variable_cell=dyn.variable_cell)

    resumed = new_state(atoms, compute_stress=state.stress is not None, seed=123)
    resumed_dyn = cls(resumed, config)
    resumed_field = HarmonicForces(resumed)
    start(resumed, resumed_dyn, resumed_field)  # the engine's construction, then the restart replaces it
    restart_io.load_state(restart_io.read_restart(path, engine="gpu"), resumed, resumed_dyn)
    assert_same_run(resumed, state)
    if dyn.variable_cell:
        assert torch.equal(resumed_dyn.p_eps, dyn.p_eps) and torch.equal(resumed_dyn.force_eps, dyn.force_eps)

    run_md(state, dyn, field, 20, lambda step: heating(30)(10 + step))
    run_md(resumed, resumed_dyn, resumed_field, 20, lambda step: heating(30)(10 + step))
    assert_same_run(resumed, state)


# -------------------------------------------------------------------------- 1b: barostat state and config

def baoab_1a():
    """``fastmace/integrators/baoab.py`` at tag ``fastmace-1a`` (p_eps on the integrator), from git."""
    import subprocess

    root = Path(__file__).resolve().parents[2]
    try:
        source = subprocess.run(["git", "show", "fastmace-1a:fastmace/integrators/baoab.py"], cwd=root,
                                capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git tag fastmace-1a is not available (shallow checkout)")
    module = types.ModuleType("fastmace.integrators._baoab_1a")
    module.__package__ = "fastmace.integrators"
    exec(compile(source, "baoab_1a.py", "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("hydrostatic", [True, False], ids=["hydrostatic", "full"])
def test_baoab_with_the_barostat_in_the_state_equals_1a_bitwise(hydrostatic):
    """p_eps and force_eps in SimulationState: the same 20 eager steps as phase 1a (CPU float64)."""
    old_module = baoab_1a()
    atoms = baseline_atoms()
    config = npt_config(hydrostatic, noise_batch=7, pressure_GPa=1.5, barostat_mass_factor=0.01)
    old_state, state = new_state(atoms, compute_stress=True), new_state(atoms, compute_stress=True)
    old, dyn = old_module.LangevinBAOAB(old_state, config), LangevinBAOAB(state, config)
    assert old_state.p_eps is None and state.p_eps is dyn.p_eps
    old_field, field = HarmonicForces(old_state), HarmonicForces(state)
    start(old_state, old, old_field)
    start(state, dyn, field)
    for step in range(1, 21):
        old.set_temperature(20.0 * step)
        dyn.set_temperature(20.0 * step)
        md_step(old_state, old, old_field)
        md_step(state, dyn, field)
        for name in ("positions", "velocities", "cell", "forces", "energy", "stress"):
            assert torch.equal(getattr(state, name), getattr(old_state, name)), f"step {step}: {name}"
        assert torch.equal(state.p_eps, old.p_eps) and torch.equal(state.force_eps, old.force_eps), step
    assert torch.equal(state.rng.batch, old_state.rng.batch) and state.rng.cursor == old_state.rng.cursor
    assert not torch.equal(state.cell, torch.tensor(atoms.cell.array))


def test_baoab_keeps_an_allocated_barostat_state():
    """A state that already holds p_eps of the right shape keeps its tensors (fixed addresses)."""
    state = new_state(argon(), compute_stress=True)
    first = LangevinBAOAB(state, npt_config(False))
    first.p_eps.fill_(0.25)
    pointers = state.p_eps.data_ptr(), state.force_eps.data_ptr()
    second = LangevinBAOAB(state, npt_config(False))
    assert (state.p_eps.data_ptr(), state.force_eps.data_ptr()) == pointers and second.p_eps is state.p_eps
    assert float(state.p_eps[0, 0]) == 0.25
    state.rng.batch = None  # another noise layout too
    hydrostatic = LangevinBAOAB(state, npt_config(True))  # another shape: new tensors
    assert tuple(state.p_eps.shape) == () and hydrostatic.p_eps is state.p_eps


def test_six_component_external_stress_equals_ases_externalstress():
    """IntegratorConfig.external_stress_GPa (Voigt, ASE sign) against ASE's externalstress, 20 steps."""
    voigt = (-0.3, -0.45, -0.3, 0.05, -0.02, 0.01)
    reference, test = argon(seed=3), argon(seed=3)
    reference.calc = lj_calculator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ase_dyn = ASELangevinBAOAB(reference, timestep=2 * units.fs, T_tau=100 * units.fs, P_tau=400 * units.fs,
                                   hydrostatic=False, externalstress=np.array(voigt) * units.GPa,
                                   temperature_K=100, rng=np.random.default_rng(SEED))
    state = new_state(test, compute_stress=True)
    field = ASEForces(test)
    config = npt_config(False, pressure_GPa=0.0, external_stress_GPa=voigt, ase_trace_drag_all_elements=True,
                        noise_batch=7)
    dyn = LangevinBAOAB(state, config, temperature_K=100)
    np.testing.assert_array_equal(dyn.externalstress.numpy(), ase_dyn.externalstress)
    start(state, dyn, field)
    ase_dyn.run(20)
    run_md(state, dyn, field, 20)
    np.testing.assert_allclose(state.cell.numpy(), reference.cell.array, atol=1e-9)
    np.testing.assert_allclose(state.positions.numpy(), reference.positions, atol=1e-9)
    np.testing.assert_allclose(state.velocities.numpy(), reference.get_velocities(), atol=1e-9)
    with pytest.raises(ValueError, match="scalar when hydrostatic"):
        LangevinBAOAB(new_state(argon(), compute_stress=True),
                      npt_config(True, pressure_GPa=0.0, external_stress_GPa=voigt))


def test_external_stress_from_the_pressure_is_1as():
    """Without external_stress_GPa the target is -pressure_GPa in ASE units, the 1a value exactly."""
    for hydrostatic in (True, False):
        state = new_state(argon(), compute_stress=True)
        dyn = LangevinBAOAB(state, npt_config(hydrostatic))
        np.testing.assert_array_equal(dyn.externalstress.numpy(), external_stress(-0.3 * units.GPa, hydrostatic))
        scalar = LangevinBAOAB(new_state(argon(), compute_stress=True),
                               npt_config(hydrostatic, pressure_GPa=0.0, external_stress_GPa=-0.3))
        np.testing.assert_array_equal(scalar.externalstress.numpy(), dyn.externalstress.numpy())


def test_disable_cell_langevin_from_the_config():
    config = npt_config(True, disable_cell_langevin=True)
    dyn = LangevinBAOAB(new_state(argon(), compute_stress=True), config)
    assert dyn.disable_cell_langevin and dyn.cell_numbers == 0
    # the 1a keyword still wins for one release (planner/fixed.py passes it)
    keyword = LangevinBAOAB(new_state(argon(), compute_stress=True), config, disable_cell_langevin=False)
    assert not keyword.disable_cell_langevin and keyword.cell_numbers == 1
    stress = LangevinBAOAB(new_state(argon(), compute_stress=True), npt_config(True),
                           externalstress=-0.1 * units.GPa)
    assert float(stress.externalstress) == -0.1 * units.GPa


# -------------------------------------------------------------------------- baseline equality

def baseline_module(name):
    """langevin.<name> from the baseline export, without running langevin/__init__.py."""
    root = os.environ.get("FASTMACE_BASELINE")
    if not root:
        pytest.skip("FASTMACE_BASELINE is not set (export of research-engine-baseline-20261017)")
    directory = Path(root).expanduser() / "Experiments" / "Langevin_Thermostat" / "langevin"
    assert directory.is_dir(), f"FASTMACE_BASELINE={root} has no Experiments/Langevin_Thermostat/langevin"
    if "langevin" not in sys.modules:
        package = types.ModuleType("langevin")
        package.__path__ = [str(directory)]
        sys.modules["langevin"] = package
    return importlib.import_module(f"langevin.{name}")


class _FakeNeighbours:
    """The parts of SkinNeighbourList the baseline integrators touch; never asks for a rebuild."""

    device = "cpu"
    capacity = 1
    variable_cell = True

    def __init__(self):
        self.count = torch.ones((), dtype=torch.int64)
        self.cell_t = torch.zeros((3, 3), dtype=torch.float64)

    def compact(self, positions):
        pass

    def max_displacement_exceeded(self, positions, fraction=1.0):
        return torch.zeros((), dtype=torch.bool)

    def set_cell(self, cell):
        self.cell_t.copy_(cell)


class BaselineHarmonic:
    """``harmonic`` behind the baseline's force-field interface (MACEForces)."""

    padding_exact = True

    def __init__(self, atoms, compute_stress):
        self.compute_stress = compute_stress
        self.neighbours = _FakeNeighbours()
        self.data = {"cell": torch.tensor(atoms.cell.array, dtype=torch.float64)}
        self.x0 = torch.tensor(atoms.positions, dtype=torch.float64)
        self.v0 = float(det_3x3(self.data["cell"]))

    def rebuild(self, positions):
        raise AssertionError("no rebuilds with the fake neighbour list")

    def use_edges(self, num_edges):
        pass

    def set_cell(self, cell):
        self.data["cell"].copy_(cell)

    def __call__(self, positions):
        energy, forces, stress = harmonic(positions, self.data["cell"], self.x0, v0=self.v0)
        return (energy, forces, stress) if self.compute_stress else (energy, forces)


def baseline_atoms():
    atoms = argon(seed=6, displacement=0.1)
    atoms.set_velocities(np.random.default_rng(1).standard_normal((len(atoms), 3)) * 0.005)
    return atoms


@pytest.mark.parametrize("fixcm", [True, False])
def test_langevin_steps_equal_the_baseline_bitwise(fixcm):
    base = baseline_module("gpu_langevin")
    atoms = baseline_atoms()
    config = nvt_config(noise_batch=7, fixcm=fixcm)
    old = base.GPULangevin(atoms.copy(), BaselineHarmonic(atoms, False), timestep=config.timestep_fs * units.fs,
                           friction=config.friction, rng=np.random.default_rng(SEED), temperature_K=0.0,
                           fixcm=fixcm, noise_batch=config.noise_batch)
    state = new_state(atoms)
    dyn = Langevin(state, config)
    field = HarmonicForces(state)
    start(state, dyn, field)
    assert torch.equal(state.forces, old.f) and torch.equal(state.energy, old.energy)
    for step in range(1, 21):
        old.set_temperature(10.0 * step)
        dyn.set_temperature(10.0 * step)
        old.step()
        md_step(state, dyn, field)
        for name, new, ref in (("xi", dyn.xi, old.xi), ("eta", dyn.eta, old.eta), ("x", state.positions, old.x),
                               ("v", state.velocities, old.v), ("f", state.forces, old.f),
                               ("energy", state.energy, old.energy), ("rnd_pos", dyn.rnd_pos, old.rnd_pos),
                               ("rnd_vel", dyn.rnd_vel, old.rnd_vel)):
            assert torch.equal(new, ref), f"step {step}: {name}"
    assert torch.equal(state.rng.batch, old._noise) and state.rng.cursor == old._noise_next
    base_values, values = old._snapshot_values(), dyn.output_values(state)
    for key in base_values:
        assert torch.equal(values[key], base_values[key]), key


@pytest.mark.parametrize("trace_all", [False, True], ids=["corrected", "ase-trace"])
@pytest.mark.parametrize("hydrostatic", [True, False], ids=["hydrostatic", "full"])
def test_baoab_steps_equal_the_baseline_bitwise(hydrostatic, trace_all, monkeypatch):
    base = baseline_module("gpu_npt")
    monkeypatch.setattr(base, "ASE_TRACE_DRAG_ALL_ELEMENTS", trace_all)
    atoms = baseline_atoms()
    config = npt_config(hydrostatic, noise_batch=7, pressure_GPa=1.5, barostat_mass_factor=0.01,
                        ase_trace_drag_all_elements=trace_all)
    # workflow.baoab_settings: the research engine's mapping of main.py's options
    old = base.GPULangevinBAOAB(atoms.copy(), BaselineHarmonic(atoms, True), timestep=config.timestep_fs * units.fs,
                                temperature_K=0.0, T_tau=1.0 / config.friction, rng=np.random.default_rng(SEED),
                                externalstress=-config.pressure_GPa * units.GPa, hydrostatic=hydrostatic,
                                P_tau=config.barostat_tau_fs * units.fs,
                                P_mass_factor=config.barostat_mass_factor, noise_batch=config.noise_batch)
    state = new_state(atoms, compute_stress=True)
    dyn = LangevinBAOAB(state, config)
    assert dyn.barostat_mass == old.barostat_mass
    field = HarmonicForces(state)
    start(state, dyn, field)
    assert torch.equal(dyn.force_eps, old.force_eps)
    for step in range(1, 21):
        old.set_temperature(20.0 * step)
        dyn.set_temperature(20.0 * step)
        old.step()
        md_step(state, dyn, field)
        for name, new, ref in (("xi", dyn.xi, old.xi), ("zeta", dyn.zeta, old.zeta), ("x", state.positions, old.x),
                               ("v", state.velocities, old.v), ("cell", state.cell, old.cell),
                               ("f", state.forces, old.f), ("energy", state.energy, old.energy),
                               ("stress", state.stress, old.stress), ("p_eps", dyn.p_eps, old.p_eps),
                               ("force_eps", dyn.force_eps, old.force_eps)):
            assert torch.equal(new, ref), f"step {step}: {name}"
    assert not torch.equal(state.cell, torch.tensor(atoms.cell.array))  # the barostat moved the cell
    assert torch.equal(state.rng.batch, old._noise) and state.rng.cursor == old._noise_next
    base_values, values = old._snapshot_values(), dyn.output_values(state)
    for key in base_values:
        assert torch.equal(values[key], base_values[key]), key
