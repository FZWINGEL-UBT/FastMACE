"""Nose-Hoover chains and MTK NPT on the CPU (plan task 4-T03): ASE equivalence, D2, restarts, refusals.

The integrators never call a force field, so these tests drive the half steps in the engine's eager order
(as ``test_integrators.py`` does) with ASE's non-smooth Lennard-Jones on a host copy of the state, against
ASE 3.27's ``NoseHooverChainNVT``, ``IsotropicMTKNPT`` and ``MTKNPT`` on the same calculator.
"""

import dataclasses
import warnings

import numpy as np
import pytest
import torch
from ase import units
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.constraints import FixAtoms
from ase.md.nose_hoover_chain import MTKNPT, IsotropicMTKNPT, NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from fastmace import integrators as records
from fastmace.engine.registry import policy_for
from fastmace.integrators.base import CONSERVED_KEY
from fastmace.integrators.mtk import MTK, exp_and_exprel
from fastmace.integrators.nose_hoover import NoseHooverChainNVT as FMNoseHooverChain
from fastmace.integrators.registry import build_integrator, integrator_refusals, registry
from fastmace.io import restart as restart_io
from fastmace.neighbors.skin import det_3x3
from fastmace.state import SimulationState

SIGMA, EPSILON, RC = 3.4, 0.0104, 7.0
STEPS = 300
DT_FS = 2.0
TDAMP_FS, PDAMP_FS = 50.0, 200.0
PRESSURE_GPA = 0.3


def argon(seed=0, temperature_K=120.0):
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat((3, 3, 3))
    atoms.rattle(0.05, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # ASE 3.29 prefers thermalize_momenta; 3.27 lacks it
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=np.random.default_rng(seed))
    Stationary(atoms)
    return atoms


def lj_calculator():
    return LennardJones(sigma=SIGMA, epsilon=EPSILON, rc=RC, smooth=False)


class ASEForces:
    """ASE's Lennard-Jones on a host copy of the state: energy, forces and the stress without the ideal gas."""

    def __init__(self, atoms):
        self.atoms = atoms.copy()
        self.atoms.calc = lj_calculator()
        self.calls = 0

    def __call__(self, state):
        self.calls += 1
        self.atoms.set_cell(state.cell.cpu().numpy().copy())
        self.atoms.set_positions(state.positions.cpu().numpy().copy())
        device = state.positions.device
        state.forces.copy_(torch.from_numpy(self.atoms.get_forces()).to(device))
        state.energy.copy_(torch.tensor([self.atoms.get_potential_energy()], dtype=torch.float64, device=device))
        if state.stress is not None:
            state.stress.copy_(torch.from_numpy(self.atoms.get_stress(voigt=False)).to(device))


def start(state, integrator, field):
    field(state)
    integrator.start(state)


def md_step(state, integrator, field):
    integrator.prepare_step(state)
    integrator.first_half(state)
    field(state)
    integrator.second_half(state)
    state.step += 1


def new_state(atoms, compute_stress=False, device="cpu"):
    return SimulationState.from_atoms(atoms, device, 7, compute_stress=compute_stress)


def nhc_config(fixcm=False, dof="ase", temperature_K=150.0, **options):
    return records.NoseHooverChain(timestep_fs=DT_FS, temperature_K=temperature_K, tdamp_fs=TDAMP_FS,
                                   fixcm=fixcm, dof=dof, **options).to_config(seed=7)


def mtk_config(cell_mode, fixcm=False, dof="ase", temperature_K=150.0, **options):
    return records.MTK(timestep_fs=DT_FS, temperature_K=temperature_K, pressure_GPa=PRESSURE_GPA,
                       tdamp_fs=TDAMP_FS, pdamp_fs=PDAMP_FS, cell_mode=cell_mode, fixcm=fixcm, dof=dof,
                       **options).to_config(seed=7)


def ase_dynamics(which, atoms, temperature_K=150.0, **options):
    common = dict(timestep=DT_FS * units.fs, temperature_K=temperature_K, tdamp=TDAMP_FS * units.fs, **options)
    if which == "nhc":
        return NoseHooverChainNVT(atoms, **common)
    cls = IsotropicMTKNPT if which == "hydrostatic" else MTKNPT
    return cls(atoms, pressure_au=PRESSURE_GPA * units.GPa, pdamp=PDAMP_FS * units.fs, **common)


def build(which, state, **options):
    config = nhc_config(**options) if which == "nhc" else mtk_config(which, **options)
    return build_integrator(state, config)


def set_ase_temperature(dyn, temperature_K):
    """What our set_temperature does, on ASE's objects: the target kT only (conductor's choice 4)."""
    kT = units.kB * temperature_K
    dyn._thermostat._kT = kT
    if hasattr(dyn, "_barostat"):
        dyn._barostat._kT = kT


def run_pair(which, steps=STEPS, schedule=None, **options):
    reference, test = argon(seed=3), argon(seed=3)
    reference.calc = lj_calculator()
    ase_dyn = ase_dynamics(which, reference, **{k: v for k, v in options.items() if k in ("tchain", "tloop",
                                                                                         "pchain", "ploop")})
    state = new_state(test, compute_stress=which != "nhc")
    field = ASEForces(test)
    dyn = build(which, state, **options)
    start(state, dyn, field)
    for step in range(1, steps + 1):
        if schedule is not None and schedule(step) is not None:
            set_ase_temperature(ase_dyn, schedule(step))
            dyn.set_temperature(schedule(step))
        ase_dyn.run(1)
        md_step(state, dyn, field)
    return reference, ase_dyn, state, dyn, field


def check_against_ase(reference, ase_dyn, state, dyn):
    np.testing.assert_allclose(state.positions.numpy(), reference.positions, atol=1e-9)
    np.testing.assert_allclose(state.velocities.numpy(), reference.get_velocities(), atol=1e-9)
    np.testing.assert_allclose(state.cell.numpy(), reference.cell.array, atol=1e-9)
    assert float(state.energy[0]) == pytest.approx(reference.get_potential_energy(), abs=1e-8)
    np.testing.assert_allclose(dyn.thermostat.eta.numpy(), ase_dyn._thermostat._eta, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(dyn.thermostat.p_eta.numpy(), ase_dyn._thermostat._p_eta, rtol=1e-8, atol=1e-12)
    conserved = float(dyn.output_values(state)[CONSERVED_KEY][0])
    assert conserved == pytest.approx(ase_dyn.get_conserved_energy(), abs=1e-8)


# -------------------------------------------------------------------------- records and refusals

def test_records_are_registered():
    nhc, mtk = registry().integrator("nhc"), registry().integrator("mtk")
    assert nhc.ensembles == ("nvt",) and nhc.cell_modes == ("fixed",) and nhc.ase_dof == "3n"
    assert mtk.ensembles == ("npt",) and mtk.cell_modes == ("hydrostatic", "full") and mtk.variable_cell
    assert mtk.report == "experimental: not yet qualified"
    for record in (nhc, mtk):
        assert record.constraints == ("fixcm",) and record.conserved
        assert record.load().__module__.startswith("fastmace.integrators.")
        for name, restart, _ in record.buffers:  # every buffer has its own policy, not another scheme's
            assert policy_for(f"integrator.{name}").restart == restart, name


@pytest.mark.parametrize("which", ["nhc", "hydrostatic", "full"])
def test_fix_atoms_is_refused_with_the_reason(which):
    config = nhc_config(fixcm=False) if which == "nhc" else mtk_config(which)
    reasons = integrator_refusals(config, fix_atoms=True, variable_cell=which != "nhc")
    assert [reason.code for reason in reasons] == ["constraints"]
    assert "does not support constraints" in str(reasons[0]) and "D1" in str(reasons[0])
    assert integrator_refusals(config, variable_cell=which != "nhc") == []


def test_constructors_refuse_what_ase_cannot_do():
    atoms = argon()
    state = new_state(atoms, compute_stress=True)
    with pytest.raises(ValueError, match="positive construction temperature"):
        FMNoseHooverChain(state, nhc_config(temperature_K=0.0))
    with pytest.raises(ValueError, match="'nvt'"):
        FMNoseHooverChain(state, mtk_config("full"))
    with pytest.raises(ValueError, match="stress"):
        MTK(new_state(atoms), mtk_config("full"))
    stressed = dataclasses.replace(mtk_config("full"), pressure_GPa=0.0, external_stress_GPa=(0.1, 0.2, 0.3, 0, 0, 0))
    with pytest.raises(ValueError, match="scalar pressure"):
        MTK(state, stressed)
    with pytest.raises(ValueError, match="tdamp_fs"):
        FMNoseHooverChain(state, dataclasses.replace(nhc_config(), parameters={"tchain": 3, "tloop": 1}))


def test_exp_and_exprel_match_scipy():
    from scipy.linalg import expm
    from scipy.special import exprel

    rng = np.random.default_rng(1)
    for scale in (0.0, 1e-6, 1e-3, 0.5, 2.0):
        a = rng.normal(size=(3, 3)) * scale
        a = a + a.T
        e, r = exp_and_exprel(torch.from_numpy(a))
        w, u = np.linalg.eigh(a)
        np.testing.assert_allclose(e.numpy(), expm(a), rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(r.numpy(), (u * exprel(w)) @ u.T, rtol=1e-14, atol=1e-15)


# -------------------------------------------------------------------------- ASE equivalence

def test_nhc_trajectory_matches_ase():
    reference, ase_dyn, state, dyn, field = run_pair("nhc")
    assert field.calls == STEPS + 1
    check_against_ase(reference, ase_dyn, state, dyn)
    assert abs(ase_dyn._thermostat._eta[0]) > 1e-3  # the thermostat acted


@pytest.mark.parametrize("which", ["nhc", "hydrostatic", "full"])
def test_longer_chains_and_loops_match_ase(which):
    options = dict(tchain=4, tloop=2) if which == "nhc" else dict(tchain=4, tloop=2, pchain=2, ploop=3)
    reference, ase_dyn, state, dyn, _ = run_pair(which, steps=60, **options)
    check_against_ase(reference, ase_dyn, state, dyn)


@pytest.mark.parametrize("which", ["hydrostatic", "full"])
def test_mtk_trajectory_matches_ase(which):
    reference, ase_dyn, state, dyn, field = run_pair(which)
    assert field.calls == STEPS + 1
    check_against_ase(reference, ase_dyn, state, dyn)
    np.testing.assert_allclose(dyn.barostat.eta.numpy(), ase_dyn._barostat._xi, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(dyn.barostat.p_eta.numpy(), ase_dyn._barostat._p_xi, rtol=1e-8, atol=1e-12)
    if which == "hydrostatic":
        assert float(dyn.eps) == pytest.approx(ase_dyn._eps, abs=1e-12)
        assert float(dyn.p_cell) == pytest.approx(ase_dyn._p_eps, rel=1e-8, abs=1e-12)
    else:
        np.testing.assert_allclose(dyn.p_cell.numpy(), ase_dyn._p_g, rtol=1e-8, atol=1e-12)
    assert abs(reference.get_volume() / argon().get_volume() - 1) > 2e-3  # the barostat moved the cell


@pytest.mark.parametrize("which", ["nhc", "hydrostatic", "full"])
def test_a_temperature_schedule_moves_only_the_target(which):
    """Q, R and W stay at the construction temperature; the target kT follows the schedule (choice 4)."""
    schedule = lambda step: 150.0 + 100.0 * step / 100  # noqa: E731
    reference, ase_dyn, state, dyn, _ = run_pair(which, steps=100, schedule=schedule)
    check_against_ase(reference, ase_dyn, state, dyn)
    assert dyn.thermostat.masses[0] == pytest.approx(3 * state.natoms * units.kB * 150.0 * (TDAMP_FS * units.fs) ** 2)


def test_a_pressure_change_matches_ase():
    reference, test = argon(seed=3), argon(seed=3)
    reference.calc = lj_calculator()
    ase_dyn = ase_dynamics("full", reference)
    state = new_state(test, compute_stress=True)
    field = ASEForces(test)
    dyn = build("full", state)
    start(state, dyn, field)
    for step in range(1, 81):
        if step == 40:
            ase_dyn._pressure_au = -0.2 * units.GPa
            dyn.set_pressure(-0.2)
        ase_dyn.run(1)
        md_step(state, dyn, field)
    check_against_ase(reference, ase_dyn, state, dyn)


# -------------------------------------------------------------------------- D2: fixcm and 3N - 3

@pytest.mark.parametrize("which", ["nhc", "hydrostatic", "full"])
def test_fixcm_counts_3n_minus_3_and_keeps_the_centre_of_mass(which):
    atoms = argon(seed=5)
    atoms.set_momenta(atoms.get_momenta() + np.array([0.01, -0.02, 0.005]))  # a drifting start
    state = new_state(atoms, compute_stress=which != "nhc")
    field = ASEForces(atoms)
    dyn = build(which, state, fixcm=True, dof="auto")
    n = state.natoms
    assert dyn.ndof == 3 * n - 3 == dyn.degrees_of_freedom(state)
    kT = units.kB * 150.0
    assert dyn.thermostat.masses[0] == (3 * n - 3) * kT * (TDAMP_FS * units.fs) ** 2
    if which == "hydrostatic":
        assert dyn.W == (3 * n - 3 + 3) * kT * (PDAMP_FS * units.fs) ** 2
    elif which == "full":
        assert dyn.W == ((3 * n - 3) / 3 + 1) * kT * (PDAMP_FS * units.fs) ** 2
    start(state, dyn, field)
    momentum = (state.masses * state.velocities).sum(dim=0)
    assert float(momentum.abs().max()) < 1e-14
    for _ in range(50):
        md_step(state, dyn, field)
    momentum = (state.masses * state.velocities).sum(dim=0)
    assert float(momentum.abs().max()) < 1e-13


def test_ase_dof_without_fixcm_counts_3n():
    state = new_state(argon())
    dyn = build("nhc", state, fixcm=False, dof="auto")
    assert dyn.ndof == 3 * state.natoms
    dyn = build("nhc", state, fixcm=True, dof="ase")
    assert dyn.ndof == 3 * state.natoms


def test_extended_energy_is_conserved():
    state = new_state(argon(seed=6), compute_stress=True)
    field = ASEForces(argon(seed=6))
    dyn = build("full", state, fixcm=True, dof="auto")
    start(state, dyn, field)
    values = []
    for _ in range(200):
        md_step(state, dyn, field)
        values.append(float(dyn.output_values(state)[CONSERVED_KEY][0]))
    kinetic = float(dyn.output_values(state)["kinetic"][0])
    assert np.ptp(values) < 0.05 * kinetic  # the non-smooth LJ cutoff, not the integrator, sets the drift


# -------------------------------------------------------------------------- device state, restarts

@pytest.mark.parametrize("which", ["nhc", "hydrostatic", "full"])
def test_addresses_stay_fixed_and_mid_step_tensors(which):
    state = new_state(argon(), compute_stress=which != "nhc")
    field = ASEForces(argon())
    dyn = build(which, state)
    start(state, dyn, field)
    pointers = {name: tensor.data_ptr() for name, tensor in dyn.buffers(state).items()}
    for name in dyn.buffers(state):
        if name not in ("positions", "velocities", "cell"):
            policy_for(f"integrator.{name}")  # every buffer has a runtime-state policy
    for step in range(5):
        dyn.set_temperature(100.0 + step)
        if which != "nhc":
            dyn.set_pressure(0.1 * step)
        md_step(state, dyn, field)
    assert {name: tensor.data_ptr() for name, tensor in dyn.buffers(state).items()} == pointers
    assert dyn.mid_step_tensors(state)[0] is state.velocities
    assert dyn.variable_cell == (which != "nhc")
    values = dyn.output_values(state)
    assert values[CONSERVED_KEY].shape == (1,)
    if which != "nhc":
        assert values["cell"] is state.cell and values["stress"] is state.stress


def test_targets_of_the_step_in_flight():
    """A target filled between stage A and stage B (pipelining) reaches the next step, not this one."""
    runs = []
    for late in (False, True):
        state = new_state(argon(), compute_stress=True)
        field = ASEForces(argon())
        dyn = build("full", state)
        start(state, dyn, field)
        for step in range(6):
            dyn.prepare_step(state)
            dyn.first_half(state)
            if late and step == 2:
                dyn.set_temperature(400.0)
                dyn.set_pressure(1.0)
            field(state)
            dyn.second_half(state)
            if not late and step == 2:
                dyn.set_temperature(400.0)
                dyn.set_pressure(1.0)
        runs.append(state)
    assert torch.equal(runs[0].positions, runs[1].positions)
    assert torch.equal(runs[0].velocities, runs[1].velocities)


@pytest.mark.parametrize("which", ["nhc", "hydrostatic", "full"])
def test_state_dict_round_trip(which):
    state = new_state(argon(), compute_stress=which != "nhc")
    field = ASEForces(argon())
    dyn = build(which, state)
    start(state, dyn, field)
    for _ in range(5):
        md_step(state, dyn, field)
    saved = dyn.state_dict()
    assert set(saved) == set(dyn.record.state_keys)  # restart format 4 checks exactly these
    other = build(which, new_state(argon(), compute_stress=which != "nhc"), temperature_K=300.0)
    pointers = {name: tensor.data_ptr() for name, tensor in other.buffers(state).items()
                if name not in ("positions", "velocities", "cell")}
    other.load_state_dict(saved)
    assert {key: np.asarray(value).tolist() for key, value in other.state_dict().items()} == \
        {key: np.asarray(value).tolist() for key, value in saved.items()}
    assert other.thermostat.masses == dyn.thermostat.masses
    assert {name: tensor.data_ptr() for name, tensor in other.buffers(state).items()
            if name in pointers} == pointers
    with pytest.raises(ValueError):
        other.load_state_dict({key: value for key, value in saved.items() if key != "p_eta"})


def assert_same_run(a, b):
    for name in ("positions", "velocities", "cell", "forces", "energy"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    if a.stress is not None:
        assert torch.equal(a.stress, b.stress)
    assert a.step == b.step


@pytest.mark.parametrize("fixcm", [True, False])
@pytest.mark.parametrize("which", ["nhc", "hydrostatic", "full"])
def test_restart_round_trip(tmp_path, which, fixcm):
    """Run, write a restart, resume in a fresh state and integrator: bitwise the uninterrupted run."""
    atoms = argon(seed=4)
    dof = "auto" if fixcm else "ase"
    schedule = lambda step: 150.0 + step  # noqa: E731
    state = new_state(atoms, compute_stress=which != "nhc")
    dyn = build(which, state, fixcm=fixcm, dof=dof)
    field = ASEForces(atoms)
    start(state, dyn, field)
    for step in range(1, 11):
        dyn.set_temperature(schedule(step))
        md_step(state, dyn, field)
    path = tmp_path / restart_io.RESTART_NAME
    restart_io.write_restart(path, state, engine="gpu", integrator=dyn, variable_cell=dyn.variable_cell)

    resumed = new_state(atoms, compute_stress=which != "nhc")
    # The engine of a resumed run builds the integrator at the schedule's current target, not the start's.
    resumed_dyn = build(which, resumed, fixcm=fixcm, dof=dof, temperature_K=schedule(10))
    resumed_field = ASEForces(atoms)
    start(resumed, resumed_dyn, resumed_field)
    restart_io.load_state(restart_io.read_restart(path, engine="gpu"), resumed, resumed_dyn)
    assert_same_run(resumed, state)

    for step in range(11, 31):
        dyn.set_temperature(schedule(step))
        resumed_dyn.set_temperature(schedule(step))
        md_step(state, dyn, field)
        md_step(resumed, resumed_dyn, resumed_field)
    assert_same_run(resumed, state)
    assert torch.equal(resumed_dyn.thermostat.p_eta, dyn.thermostat.p_eta)


def test_from_atoms_with_fix_atoms_still_builds_a_state():
    """The refusal is rule 1's (the planner's); the state itself holds the indices (4-T01)."""
    atoms = argon()
    atoms.set_constraint(FixAtoms(indices=[0, 1]))
    state = new_state(atoms)
    assert state.fixed_indices == (0, 1)


def test_detached_volume_is_the_cells():
    state = new_state(argon(), compute_stress=True)
    dyn = build("hydrostatic", state)
    assert float(dyn.volume(state)) == pytest.approx(float(det_3x3(state.cell)))


# -------------------------------------------------------------------------- through Simulation (4-T01's dispatch)

@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    pytest.importorskip("cuequivariance_torch")
    from _support.tiny_models import FAMILIES, tiny_mace_model
    from fastmace.model import loading

    model = tiny_mace_model([11, 17], **FAMILIES["mh1"])
    path = tmp_path_factory.mktemp("models") / "tiny-mh1-11-17.model"
    torch.save(model, path)
    loading._LICENCES_SHOWN.add(str(path))
    return loading.load_pristine(str(path), trusted=True)


@pytest.fixture
def cpu_planner(monkeypatch):
    from fastmace.config import EngineConfig
    from fastmace.engine import constraints
    from fastmace.planner import minimal

    planner = minimal.MinimalPlanner(engine=EngineConfig(device="cpu", compile_mode="none"), environ={},
                                     process=constraints.ProcessRecord())
    monkeypatch.setattr(minimal, "default_planner", lambda force_field: planner)
    return planner


def simulation_record(which):
    common = dict(timestep_fs=1.0, temperature_K=300.0, tdamp_fs=20.0)
    if which == "nhc":
        return records.NoseHooverChain(**common)
    return records.MTK(pressure_GPa=0.2, pdamp_fs=200.0, cell_mode=which, **common)


def tiny_simulation(tiny_model, which):
    from _support.structures import nacl
    from fastmace.model.forcefield import ForceField
    from fastmace.planner.workload import Workload
    from fastmace.simulation import Simulation

    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    return Simulation(atoms, ForceField(tiny_model, atoms, precision="float64"), integrator=simulation_record(which),
                      workload=Workload(max_temperature_K=400.0), seed=4)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("which", ["nhc", "hydrostatic", "full"])
def test_simulation_runs_and_resumes_bitwise(tiny_model, cpu_planner, which, tmp_path):
    """Simulation builds the scheme through the registry; save, resume and run equals the uninterrupted run."""
    from fastmace.protocols import hold, ramp
    from fastmace.simulation import Simulation

    sim = tiny_simulation(tiny_model, which)
    assert type(sim._engine.integrator).__name__ == ("NoseHooverChainNVT" if which == "nhc" else "MTK")
    sim.run(schedule=ramp(300.0, 350.0, ps=0.01))
    sim.save_restart(tmp_path / "restart.npz")
    resumed = Simulation.resume(tmp_path)
    assert resumed._engine.integrator.mass_temperature_K == 300.0  # the masses' temperature, not the target
    sim.run(schedule=hold(350.0, ps=0.01))
    resumed.run(schedule=hold(350.0, ps=0.01))
    for name in ("positions", "velocities", "cell"):
        assert torch.equal(getattr(resumed._engine.state, name), getattr(sim._engine.state, name)), name
    values = sim._engine.integrator.output_values(sim._engine.state)
    assert bool(torch.isfinite(values[CONSERVED_KEY]).all())


def test_the_fused_stage_needs_cuda():
    """4-T03b: the Triton chain stage is the default on CUDA only; on the CPU the PyTorch stage runs."""
    state = new_state(argon(), compute_stress=True)
    assert build("nhc", state).fused is False
    assert build("full", state).fused is False
    config = mtk_config("full")
    with pytest.raises(ValueError, match="need CUDA"):
        MTK(state, config, fused=True)
