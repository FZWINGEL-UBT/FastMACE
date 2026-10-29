"""Plan task 4-T04: ``FixAtoms`` (SPEC §4.4, owner decisions 44 and 45).

ASE's semantics, which the integrators reproduce on the mask path (``state.free_mask``):
``adjust_positions`` resets the fixed atoms to their old positions exactly, ``adjust_forces`` zeroes their
forces in ``get_forces(md=True)``, and ``adjust_momenta`` zeroes their momenta in ``set_momenta``. The noise
is drawn for every atom. ASE's ``VelocityVerlet`` (and ``Bussi``, which steps with it) recomputes every
momentum from the constrained displacement when any constraint is present.

The integrators run eagerly in the engine's order against their ASE classes on the same tiny float64 MACE
model (``test_nve_csvr_berendsen``): positions within 1e-12 Angstrom after 200 steps, fixed atoms bitwise in
place with zero velocities. Without ``FixAtoms`` the step is the unconstrained one bit for bit. Rule 1's
refusals, ``fixcm="auto"``, the degrees of freedom, the restart and resume, ``fastmace.ase`` and the
reference's raw D3 forces follow.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest
import test_nve_csvr_berendsen as t02
from ase import units
from ase.constraints import FixAtoms, FixCom
from ase.md.bussi import Bussi
from ase.md.langevin import Langevin as ASELangevin
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.verlet import VelocityVerlet as ASEVelocityVerlet

from fastmace import integrators as records
from fastmace import optimize
from fastmace.integrators import registry as schemes
from fastmace.state import SimulationState

STEPS = 200
TOLERANCE = 1e-12
FIXED = (0, 5, 9, 17, 30)


def constrained(seed):
    atoms = t02.nacl(seed=seed)
    atoms.set_constraint(FixAtoms(indices=FIXED))
    atoms.set_momenta(atoms.get_momenta())  # ASE's adjust_momenta: the fixed atoms start at rest
    return atoms


class RawForces(t02.Forces):
    """The engine's force stage: every atom's forces, as the force field gives them (no constraint)."""

    def __init__(self, atoms, slot="fastmace"):
        super().__init__(atoms, slot)
        self.atoms.set_constraint()


LANGEVIN = records.Langevin(timestep_fs=1.0, temperature_K=600.0, friction_per_fs=0.02, fixcm=False)
NVE = records.NVE(timestep_fs=1.0, fixcm=False, dof="ase")
CSVR = records.CSVR(timestep_fs=1.0, temperature_K=600.0, tau_fs=20.0, fixcm=False, dof="ase")
NVT_BERENDSEN = records.Berendsen(timestep_fs=1.0, temperature_K=600.0, taut_fs=20.0, fixcm=False, dof="ase")


def ase_dynamics(name, reference):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # ASE >= 3.28 deprecates fixcm
        if name == "langevin":
            return ASELangevin(reference, timestep=units.fs, friction=0.02 / units.fs, fixcm=False,
                               temperature_K=600.0, rng=np.random.default_rng(t02.SEED))
        if name == "nve":
            return ASEVelocityVerlet(reference, timestep=units.fs)
        if name == "csvr":
            return Bussi(reference, timestep=units.fs, temperature_K=600.0, taut=20.0 * units.fs,
                         rng=np.random.default_rng(t02.SEED))
        return NVTBerendsen(reference, timestep=units.fs, temperature_K=600.0, taut=20.0 * units.fs, fixcm=False)


RECORDS = {"langevin": LANGEVIN, "nve": NVE, "csvr": CSVR, "nvt-berendsen": NVT_BERENDSEN}


def pair(name, seed=4, steps=STEPS, noise_batch=7):
    atoms = constrained(seed)
    start_positions = atoms.positions.copy()
    reference = t02.reference_atoms(atoms)
    assert reference.constraints  # ASE runs with the constraint
    ase_dyn = ase_dynamics(name, reference)
    state = t02.new_state(atoms)
    field = RawForces(atoms)
    dyn = t02.build(RECORDS[name], state, noise_batch=noise_batch)
    t02.start(state, dyn, field)
    for _ in range(steps):
        ase_dyn.run(1)
        t02.md_step(state, dyn, field)
    return reference, state, dyn, field, start_positions


# -------------------------------------------------------------------------- against ASE

@pytest.mark.parametrize("name", list(RECORDS))
def test_fix_atoms_matches_ase(name):
    reference, state, dyn, field, start_positions = pair(name)
    assert field.calls == STEPS + 1
    # ASE's Langevin stores momenta and divides by the masses again each step, and its noise amplifies the
    # rounding: 1.5e-12 after 200 steps (test_integrators compares Langevin at 1e-9).
    tolerance = 1e-11 if name == "langevin" else TOLERANCE
    np.testing.assert_allclose(state.positions.numpy(), reference.positions, rtol=0, atol=tolerance)
    np.testing.assert_allclose(state.velocities.numpy(), reference.get_velocities(), rtol=0, atol=tolerance)
    fixed = list(FIXED)
    np.testing.assert_array_equal(state.positions.numpy()[fixed], start_positions[fixed])  # bitwise in place
    assert not state.velocities.numpy()[fixed].any()
    free = [index for index in range(state.natoms) if index not in FIXED]
    assert np.abs(state.positions.numpy()[free] - start_positions[free]).max() > 1e-3  # the others moved


def test_csvr_counts_the_fixed_atoms_and_matches_bussis_transfer():
    reference, state, dyn, field, _ = pair("csvr")
    assert dyn.ndof == 3 * (state.natoms - len(FIXED)) == reference.get_number_of_degrees_of_freedom()


def test_the_noise_is_drawn_for_every_atom():
    state = t02.new_state(constrained(2))
    dyn = t02.build(LANGEVIN, state, noise_batch=3)
    assert tuple(state.rng.batch.shape) == (3, 2, state.natoms, 3)
    reference = np.random.default_rng(t02.SEED)
    dyn.prepare_step(state)
    np.testing.assert_array_equal(dyn.xi.numpy(), reference.standard_normal(size=(state.natoms, 3)))


# -------------------------------------------------------------------------- unchanged without FixAtoms

def _reference_langevin_step(integrator, state, field):
    """Langevin's step as it was before 4-T04 (fixcm off): the no-FixAtoms path must stay this, bit for bit."""
    integrator.prepare_step(state)
    x, v, m = state.positions, state.velocities, state.masses
    sqrt_kT = integrator.sqrt_kT
    rnd_pos = (integrator.c5_unit * sqrt_kT) * integrator.eta
    rnd_vel = (integrator.c3_unit * sqrt_kT) * integrator.xi - (integrator.c4_unit * sqrt_kT) * integrator.eta
    v.add_(integrator.c1 * state.forces / m - integrator.c2 * v + rnd_vel)
    x_old = x.clone()
    x.add_(integrator.dt * v + rnd_pos)
    v.copy_((x - x_old - rnd_pos) / integrator.dt)
    field(state)
    v.add_(integrator.c1 * state.forces / m - integrator.c2 * v + rnd_vel)
    state.step += 1


def _reference_verlet_step(integrator, state, field):
    """``VelocityVerlet``'s step as 4-T02 wrote it (fixcm off)."""
    v = state.velocities
    v.add_(0.5 * integrator.dt * state.forces / state.masses)
    state.positions.add_(integrator.dt * v)
    field(state)
    v.add_(0.5 * integrator.dt * state.forces / state.masses)
    state.step += 1


@pytest.mark.parametrize("name, step", [("langevin", _reference_langevin_step), ("nve", _reference_verlet_step)])
def test_without_fix_atoms_the_step_is_unchanged_bitwise(name, step):
    atoms = t02.nacl(seed=8)
    runs = []
    for reference in (False, True):
        state = t02.new_state(atoms)
        assert state.free_mask is None
        field = RawForces(atoms)
        dyn = t02.build(RECORDS[name], state, noise_batch=5)
        t02.start(state, dyn, field)
        for _ in range(20):
            if reference:
                step(dyn, state, field)
            else:
                t02.md_step(state, dyn, field)
        runs.append(state)
    t02.assert_same_run(*runs)


# -------------------------------------------------------------------------- the state

def test_the_state_zeroes_the_fixed_atoms_velocities():
    atoms = t02.nacl(seed=1)
    velocities = atoms.get_velocities()
    atoms.set_constraint(FixAtoms(indices=FIXED))
    state = SimulationState.from_atoms(atoms, "cpu", 0)
    assert state.fixed_indices == FIXED and state.n_fixed == len(FIXED)
    assert not state.velocities.numpy()[list(FIXED)].any()
    free = [index for index in range(len(atoms)) if index not in FIXED]
    np.testing.assert_array_equal(state.velocities.numpy()[free], velocities[free])


# -------------------------------------------------------------------------- rule 1

def _codes(reasons):
    return sorted(reason.code for reason in reasons)


@pytest.mark.parametrize("record", [LANGEVIN, NVE, CSVR, NVT_BERENDSEN], ids=list(RECORDS))
def test_the_fixed_cell_integrators_take_fix_atoms(record):
    assert "fix_atoms" in schemes.registry().integrator(record.to_config(seed=0).scheme).constraints
    assert schemes.integrator_refusals(record.to_config(seed=0, fixed_atoms=True), fix_atoms=True) == []


@pytest.mark.parametrize("record", [dataclasses.replace(LANGEVIN, fixcm=True), dataclasses.replace(NVE, fixcm=True),
                                    dataclasses.replace(CSVR, fixcm=True),
                                    dataclasses.replace(NVT_BERENDSEN, fixcm=True)], ids=list(RECORDS))
def test_fixcm_with_fix_atoms_is_refused(record):
    reasons = schemes.integrator_refusals(record.to_config(seed=0, fixed_atoms=True), fix_atoms=True)
    assert _codes(reasons) == ["fix-atoms-with-fixcm"]
    assert "fixcm=False" in str(reasons[0])


def test_a_variable_cell_with_fix_atoms_is_refused():
    baoab = records.LangevinBAOAB(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.02, fixcm=False)
    npt_berendsen = dataclasses.replace(t02.NPT_BERENDSEN, fixcm=False)
    inhomogeneous = dataclasses.replace(t02.INHOMOGENEOUS, fixcm=False)
    for record in (baoab, npt_berendsen, inhomogeneous):
        config = record.to_config(seed=0, fixed_atoms=True)
        reasons = schemes.integrator_refusals(config, fix_atoms=True, variable_cell=True)
        assert "fix-atoms-variable-cell" in _codes(reasons) or "constraints" in _codes(reasons), record
        assert any("variable cell" in str(reason) for reason in reasons), record
    assert _codes(schemes.integrator_refusals(npt_berendsen.to_config(seed=0, fixed_atoms=True), fix_atoms=True,
                                              variable_cell=True)) == ["fix-atoms-variable-cell"]


def test_nhc_and_mtk_refuse_fix_atoms_when_registered():
    reg = schemes.registry()
    for scheme in ("nose_hoover", "nhc", "mtk"):
        record = reg.get("integrator", scheme)
        if record is not None:
            assert "fix_atoms" not in record.constraints and "ASE" in record.refusal("fix_atoms")


def test_mc_and_other_constraints_stay_refused():
    config = LANGEVIN.to_config(seed=0, fixed_atoms=True)
    assert _codes(schemes.integrator_refusals(config, fix_atoms=True, mc=True)) == ["mc-with-constraints"]
    assert _codes(schemes.integrator_refusals(config, other_constraints=("FixCom",))) == ["constraints"]


def test_the_optimisers_take_fix_atoms_at_fixed_cell_only():
    for record in (optimize.LBFGS(), optimize.FIRE()):
        assert schemes.optimizer_refusals(record.to_config(), fix_atoms=True) == []
    if schemes.registry().get("cell_filter", "frechet") is not None:  # 4-T07
        config = dataclasses.replace(optimize.LBFGS().to_config(), cell_filter="frechet")
        assert "constraints" in _codes(schemes.optimizer_refusals(config, fix_atoms=True))


# -------------------------------------------------------------------------- fixcm="auto"

@pytest.mark.parametrize("record", [records.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.02),
                                    records.NVE(timestep_fs=1.0),
                                    records.CSVR(timestep_fs=1.0, temperature_K=300.0, tau_fs=20.0),
                                    records.Berendsen(timestep_fs=1.0, temperature_K=300.0, taut_fs=20.0)],
                         ids=list(RECORDS))
def test_fixcm_auto_resolves_from_the_fixed_atoms(record):
    assert record.fixcm == "auto"
    assert record.to_config(seed=0).fixcm is True
    assert record.to_config(seed=0, fixed_atoms=True).fixcm is False
    assert dataclasses.replace(record, fixcm=True).to_config(seed=0, fixed_atoms=True).fixcm is True
    assert dataclasses.replace(record, fixcm=False).to_config(seed=0).fixcm is False


def test_fixcm_rejects_other_values():
    with pytest.raises(ValueError, match="fixcm"):
        records.NVE(timestep_fs=1.0, fixcm="yes")


# -------------------------------------------------------------------------- degrees of freedom

def test_temperatures_count_the_fixed_atoms():
    from fastmace.io.trajectory import degrees_of_freedom

    n = 32
    for record in RECORDS.values():
        config = record.to_config(seed=0, fixed_atoms=True)
        assert degrees_of_freedom(n, config, len(FIXED)) == 3 * (n - len(FIXED))


# -------------------------------------------------------------------------- Simulation, restart and resume

@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    pytest.importorskip("cuequivariance_torch")
    from _support.mc_fixtures import tiny_loaded

    return tiny_loaded(tmp_path_factory.mktemp("models"), (11, 17))


@pytest.fixture
def planner(monkeypatch):
    from _support.mc_fixtures import cpu_planner

    return cpu_planner(monkeypatch)


SIM_FIXED = (0, 3)


def sim_atoms(fixed=SIM_FIXED):
    from _support.structures import nacl

    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    rng = np.random.default_rng(5)
    atoms.set_velocities(rng.normal(scale=0.01, size=(len(atoms), 3)))
    if fixed:
        atoms.set_constraint(FixAtoms(indices=fixed))
    return atoms


def simulation(loaded, integrator, atoms=None, directory=None):
    from fastmace.io.output import Output
    from fastmace.model.forcefield import ForceField
    from fastmace.planner.workload import Workload
    from fastmace.simulation import Simulation

    atoms = sim_atoms() if atoms is None else atoms
    output = None if directory is None else Output(directory, thermo_every=1, trajectory_every=2,
                                                   restart_every=None)
    return Simulation(atoms, ForceField(loaded, atoms, precision="float64"), integrator=integrator,
                      workload=Workload(max_temperature_K=600.0), seed=1, output=output)


SIM_LANGEVIN = records.Langevin(timestep_fs=1.0, temperature_K=600.0, friction_per_fs=0.02)


def test_a_simulation_runs_with_fixed_atoms_and_auto_fixcm(loaded, planner):
    atoms = sim_atoms()
    start = atoms.positions.copy()
    sim = simulation(loaded, SIM_LANGEVIN, atoms)
    assert sim._config.fixcm is False and sim.plan.integrator.fixcm is False
    assert sim.degrees_of_freedom == 3 * (len(atoms) - len(SIM_FIXED))
    sim.run(steps=5)
    committed = sim.committed()
    np.testing.assert_array_equal(committed.positions[list(SIM_FIXED)], start[list(SIM_FIXED)])
    assert not committed.momenta[list(SIM_FIXED)].any()
    sim.close()


def test_a_simulation_with_fixcm_true_and_fixed_atoms_is_refused(loaded, planner):
    from fastmace.errors import PlanRefused

    with pytest.raises(PlanRefused, match="fixcm=False") as error:
        simulation(loaded, dataclasses.replace(SIM_LANGEVIN, fixcm=True))
    assert "fix-atoms-with-fixcm" in {reason.code for reason in error.value.reasons}


def test_a_simulation_with_a_variable_cell_and_fixed_atoms_is_refused(loaded, planner):
    from fastmace.errors import PlanRefused

    baoab = records.LangevinBAOAB(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.02, pressure_GPa=0.1)
    with pytest.raises(PlanRefused, match="variable cell"):
        simulation(loaded, baoab)


def test_restart_holds_the_indices_and_resume_rebuilds_fix_atoms(loaded, planner, tmp_path):
    from fastmace.io import restart as restart_io
    from fastmace.simulation import Simulation

    sim = simulation(loaded, SIM_LANGEVIN, directory=tmp_path / "run")
    sim.run(steps=3)
    sim.save_restart(tmp_path / "run")
    want = sim.committed()
    sim.run(steps=4)
    after = sim.committed()
    sim.close()
    stored = restart_io.read_restart(tmp_path / "run" / restart_io.RESTART_NAME)
    assert restart_io.fixed_atoms(stored) == SIM_FIXED
    resumed = Simulation.resume(tmp_path / "run")
    assert resumed._engine.state.fixed_indices == SIM_FIXED
    assert resumed._config.fixcm is False
    assert resumed.degrees_of_freedom == 3 * (8 - len(SIM_FIXED))
    np.testing.assert_array_equal(resumed.committed().positions, want.positions)
    resumed.run(steps=4)
    np.testing.assert_array_equal(resumed.committed().positions, after.positions)  # bitwise continuation
    np.testing.assert_array_equal(resumed.committed().momenta, after.momenta)
    resumed.close()


# -------------------------------------------------------------------------- fastmace.ase

@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def ase_langevin(loaded, atoms, **options):
    from fastmace.ase import Langevin

    options.setdefault("seed", 1)
    return Langevin(atoms, units.fs, temperature_K=300, friction=0.01 / units.fs, model=loaded, **options)


def test_ase_langevin_takes_fix_atoms_with_fixcm_false(loaded, planner, in_tmp):
    from ase.io import read

    atoms = sim_atoms()
    start = atoms.positions.copy()
    seen = []
    dyn = ase_langevin(loaded, atoms, fixcm=False, trajectory="run.traj", logfile="md.log")
    dyn.attach(lambda: seen.append((atoms.get_forces().copy(), atoms.get_forces(apply_constraint=False).copy())),
               interval=2)
    dyn.run(4)
    dyn.close()
    np.testing.assert_array_equal(atoms.positions[list(SIM_FIXED)], start[list(SIM_FIXED)])
    constrained_forces, raw = seen[-1]
    assert not constrained_forces[list(SIM_FIXED)].any() and raw[list(SIM_FIXED)].any()  # ASE's get_forces
    frames = read("run.traj", index=":")
    assert len(frames) == 5
    for frame in frames:
        assert [type(c).__name__ for c in frame.constraints] == ["FixAtoms"]
        assert tuple(frame.constraints[0].get_indices()) == SIM_FIXED
        assert frame.get_forces(apply_constraint=False)[list(SIM_FIXED)].any()  # raw forces stored
        assert frame.get_number_of_degrees_of_freedom() == 3 * (len(atoms) - len(SIM_FIXED))


def test_ase_langevin_with_default_fixcm_and_fix_atoms_is_refused(loaded, planner, in_tmp):
    from fastmace.errors import PlanRefused

    dyn = ase_langevin(loaded, sim_atoms())
    with pytest.raises(PlanRefused, match="fixcm=False"):
        dyn.run(1)


@pytest.mark.parametrize("change", ["add", "remove", "other"])
def test_an_observer_that_changes_the_constraints_is_refused(loaded, planner, in_tmp, change):
    from fastmace.errors import PlanRefused

    atoms = sim_atoms()

    def observer():
        if change == "add":
            atoms.set_constraint(FixAtoms(indices=(*SIM_FIXED, 5)))
        elif change == "remove":
            atoms.set_constraint()
        else:
            atoms.set_constraint([FixAtoms(indices=SIM_FIXED), FixCom()])

    dyn = ase_langevin(loaded, atoms, fixcm=False)
    dyn.attach(observer, interval=2)
    with pytest.raises(PlanRefused, match="constraints") as error:
        dyn.run(4)
    assert {reason.code for reason in error.value.reasons} == {"observer-constraints"}


def test_an_observer_that_keeps_the_constraints_runs(loaded, planner, in_tmp):
    atoms = sim_atoms()
    dyn = ase_langevin(loaded, atoms, fixcm=False)
    dyn.attach(lambda: atoms.set_constraint(FixAtoms(indices=list(reversed(SIM_FIXED)))), interval=2)
    dyn.run(4)
    dyn.close()


def test_other_ase_constraints_stay_refused(loaded, planner, in_tmp):
    from fastmace.errors import PlanRefused

    atoms = sim_atoms(fixed=())
    atoms.set_constraint(FixCom())
    with pytest.raises(PlanRefused, match="FixCom"):
        ase_langevin(loaded, atoms, fixcm=False)


# -------------------------------------------------------------------------- the reference's D3 forces

def test_reference_d3_forces_are_every_atoms():
    """The gate compares every atom's forces (constraints don't change the force field): the D3 part of the
    reference reads them without the constraint."""
    from ase.calculators.lj import LennardJones

    from fastmace.reference.calculators import ReferenceCalculator

    atoms = sim_atoms()
    stub = type("Stub", (), {"_d3_calculator": LennardJones(sigma=2.5, epsilon=0.01, rc=5.0)})()
    _, forces, _ = ReferenceCalculator._evaluate_d3(stub, atoms)
    raw = atoms.copy()
    raw.set_constraint()
    raw.calc = LennardJones(sigma=2.5, epsilon=0.01, rc=5.0)
    np.testing.assert_array_equal(forces, raw.get_forces())
    assert forces[list(SIM_FIXED)].any()
    assert atoms.constraints  # the caller's atoms keep theirs


def test_torch_dftd3_forces_are_every_atoms(monkeypatch):
    from ase.calculators.lj import LennardJones

    from fastmace.reference import d3

    monkeypatch.setattr(d3, "torch_dftd3_calculator", lambda record, **_: LennardJones(sigma=2.5, epsilon=0.01,
                                                                                         rc=5.0))
    atoms = sim_atoms()
    _, forces, _ = d3.torch_dftd3(atoms, None)
    assert forces[list(SIM_FIXED)].any()


def test_baoab_does_not_claim_a_fixed_centre_of_mass():
    """LangevinBAOAB never removes the total momentum (nor does ASE's), so its record has fixcm=False only."""
    from fastmace.simulation import _integrator

    baoab = records.LangevinBAOAB(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.02)
    assert baoab.fixcm is False and baoab.to_config(seed=0).fixcm is False
    with pytest.raises(ValueError, match="centre-of-mass"):
        records.LangevinBAOAB(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.02, fixcm=True)
    # A restart from before kept fixcm=True, which had no effect on the dynamics: it resumes as False.
    old = {"class": "LangevinBAOAB", "fields": dict(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.02,
                                                     fixcm=True)}
    assert _integrator(old).fixcm is False
