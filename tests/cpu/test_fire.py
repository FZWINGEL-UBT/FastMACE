"""GPU FIRE (plan task 4-T06) on the CPU: ASE 3.27 ``FIRE`` step for step, on the optimiser contract.

The optimiser never calls a force field, so these tests drive it themselves in the driver's eager order
(``optimize/base.py``: propose, force call, refresh, update) with ASE calculators on a host copy of the state:
Lennard-Jones argon, and a tiny MACE model (``tests/_support/tiny_models.py``). ASE runs its own ``FIRE`` on
the same calculator; positions must agree within 1e-12 Angstrom after every step, with the same step count.
ASE 3.29's ``FIRE`` (the CPU environment) is ASE 3.27's line for line.
"""

import numpy as np
import pytest
import torch
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.constraints import FixAtoms
from ase.optimize import FIRE as ASEFIRE

import fastmace as fm
from fastmace.config import OptimizerConfig
from fastmace.integrators.registry import build_dof, build_optimizer, optimizer_refusals, registry
from fastmace.optimize.fire import FIRE
from fastmace.state import SimulationState

ATOL = 1e-12


def argon(rattle=0.1, seed=1):
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat((3, 3, 3))
    atoms.rattle(rattle, seed=seed)
    return atoms


def lj_calculator():
    return LennardJones(sigma=3.4, epsilon=0.0104, rc=7.0, smooth=True)


class HostForces:
    """An ASE calculator on a host copy of the state: energy and forces, constraint-adjusted as ASE's
    ``atoms.get_forces()`` (the copy keeps the constraints)."""

    def __init__(self, atoms, calc):
        self.atoms = atoms.copy()
        self.atoms.calc = calc
        self.calls = 0

    def __call__(self, state):
        self.calls += 1
        self.atoms.set_positions(state.positions.numpy().copy(), apply_constraint=False)
        state.forces.copy_(torch.from_numpy(self.atoms.get_forces()))
        state.energy.copy_(torch.tensor([self.atoms.get_potential_energy()], dtype=torch.float64))


def ase_run(atoms, calc, fmax, steps, **options):
    """ASE's FIRE: the positions and (dt, a, Nsteps) at every convergence check, and the step count."""
    atoms = atoms.copy()
    atoms.calc = calc
    options = {"maxstep" if name == "maxstep_A" else name: value for name, value in options.items()}
    opt = ASEFIRE(atoms, logfile=None, **options)
    positions, scalars = [], []
    for _ in opt.irun(fmax=fmax, steps=steps):
        positions.append(atoms.get_positions())
        scalars.append((opt.dt, opt.a, opt.Nsteps))
    return positions, scalars, opt.nsteps, atoms


def fastmace_run(atoms, calc, fmax, steps, **options):
    """FastMACE's FIRE through the registry in the driver's eager order."""
    state = SimulationState.from_atoms(atoms, "cpu", 0)
    field = HostForces(atoms, calc)
    config = fm.optimize.FIRE(**options).to_config(fmax=fmax, steps=steps)
    dof = build_dof(state, config)
    opt = build_optimizer(dof, config)
    opt.set_fmax(fmax)
    field(state)
    opt.start()
    positions, scalars = [state.positions.numpy().copy()], []
    nsteps = 0
    while not bool(opt.converged) and nsteps < steps:
        opt.propose()
        field(state)
        dof.refresh()
        opt.update()
        nsteps += 1
        positions.append(state.positions.numpy().copy())
        record = opt.state_dict()
        scalars.append((record["dt"], record["a"], record["Nsteps"]))
    return positions, scalars, nsteps, opt, field


def compare(atoms, make_calc, fmax, steps, **options):
    ref_positions, ref_scalars, ref_steps, ref_atoms = ase_run(atoms, make_calc(), fmax, steps, **options)
    positions, scalars, nsteps, opt, field = fastmace_run(atoms, make_calc(), fmax, steps, **options)
    assert nsteps == ref_steps
    assert field.calls == nsteps + 1  # one force call per step
    for step, (mine, theirs) in enumerate(zip(positions, ref_positions)):
        np.testing.assert_allclose(mine, theirs, rtol=0, atol=ATOL, err_msg=f"step {step}")
    # ASE's scalars after each step (its first yield is before any step)
    for step, (mine, theirs) in enumerate(zip(scalars, ref_scalars[1:]), start=1):
        assert mine[2] == theirs[2], f"Nsteps at step {step}"
        assert mine[0] == pytest.approx(theirs[0], rel=1e-14), f"dt at step {step}"
        assert mine[1] == pytest.approx(theirs[1], rel=1e-14), f"a at step {step}"
    return ref_positions, ref_scalars, ref_steps, opt


def step_lengths(positions):
    return [float(np.linalg.norm(b - a)) for a, b in zip(positions, positions[1:])]


# -------------------------------------------------------------------------- ASE equivalence

def test_fire_defaults_match_ase_to_convergence():
    positions, scalars, steps, opt = compare(argon(), lj_calculator, fmax=1e-3, steps=2000)
    assert 30 < steps < 2000 and bool(opt.converged)
    dts = [dt for dt, _, _ in scalars]
    assert max(dts) > 0.1  # the time step grew past Nmin
    assert float(opt.max_force()) < 1e-3


def test_fire_resets_and_maxstep_clip_match_ase():
    """A large time step: the v.f <= 0 reset and the maxstep clip of the whole step both happen."""
    options = dict(dt=0.5, dtmax=3.0, maxstep_A=0.05, Nmin=2)
    positions, scalars, steps, _ = compare(argon(rattle=0.2), lj_calculator, fmax=1e-3, steps=400, **options)
    assert any(later[2] == 0 and earlier[2] > 0 for earlier, later in zip(scalars[1:], scalars[2:]))  # a reset
    assert any(abs(length - 0.05) < 1e-12 for length in step_lengths(positions))  # a clipped step


def test_fire_downhill_check_matches_ase():
    """Large time steps go uphill; ASE resets to the last positions and FastMACE must do the same."""
    resets = []
    options = dict(dt=1.0, dtmax=6.0, finc=1.5, Nmin=1, maxstep_A=0.3, downhill_check=True)
    atoms = argon(rattle=0.15)
    ref = ase_run(atoms, lj_calculator(), 1e-3, 300,
                  position_reset_callback=lambda *args: resets.append(1), **options)
    assert resets, "the case must exercise the uphill reset"
    compare(atoms, lj_calculator, fmax=1e-3, steps=300, **options)
    assert ref[2] > 0


def test_fire_fix_atoms_matches_ase():
    atoms = argon()
    fixed = [0, 5, 17, 40]
    atoms.set_constraint(FixAtoms(indices=fixed))
    positions, _, steps, _ = compare(atoms, lj_calculator, fmax=1e-3, steps=2000)
    assert steps > 10
    np.testing.assert_array_equal(positions[-1][fixed], atoms.positions[fixed])


def test_fire_tiny_mace_matches_ase():
    from _support.tiny_models import tiny_family_calculator

    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.rattle(0.05, seed=3)

    def make_calc():
        calc = tiny_family_calculator("mh1", [11, 17])
        warm = atoms.copy()
        warm.calc = calc
        warm.get_forces()  # TorchScript's first call rounds differently; warm both sides up
        warm.positions += 0.01
        warm.get_forces()
        return calc

    compare(atoms, make_calc, fmax=1e-9, steps=50)


# -------------------------------------------------------------------------- record, refusals, state

def test_fire_record_in_registry():
    record = registry().optimizer("fire")
    assert record.load() is FIRE
    assert record.ase_class == "ase.optimize.fire.FIRE"
    assert {name for name, _, _ in record.buffers} == set(FIRE.BUFFERS)
    assert {(policy, recompute) for _, policy, recompute in record.buffers} == {("reset", "FIRE.start")}
    assert None in record.cell_filters and "frechet" in record.cell_filters
    config = fm.optimize.FIRE().to_config()
    assert optimizer_refusals(config) == []
    assert optimizer_refusals(config, fix_atoms=True) == []  # D1 (a): fixed cell takes FixAtoms


@pytest.mark.parametrize("name, value, match", [
    ("position_reset_callback", "callback", "position_reset_callback"),
    ("maxmove", 0.1, "maxstep_A"),
    ("bogus", 1.0, "bogus"),
])
def test_fire_refuses_unknown_parameters(name, value, match):
    state = SimulationState.from_atoms(argon(), "cpu", 0)
    config = OptimizerConfig(scheme="fire", parameters={name: value})
    with pytest.raises(ValueError, match=match):
        build_optimizer(build_dof(state, config), config)


def test_fire_state_dict_and_buffers():
    atoms = argon()
    _, _, _, opt, _ = fastmace_run(atoms, lj_calculator(), fmax=1e-3, steps=3, downhill_check=True)
    record = opt.state_dict()
    assert set(record) >= {"vel", "dt", "a", "Nsteps", "started", "e_last", "r_last"}
    assert record["started"] is True and record["vel"].shape == (3 * len(atoms),)
    buffers = opt.buffers()
    assert set(buffers) == set(FIRE.BUFFERS)
    assert buffers["vel"].dtype == torch.float64 and buffers["dt"].shape == ()
    assert opt.converged.dtype == torch.bool and opt.converged.shape == ()


def test_fire_converged_at_start_takes_no_step():
    """ASE checks convergence before its first step: a relaxed structure takes none."""
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True)  # forces vanish by symmetry
    _, _, nsteps, opt, field = fastmace_run(atoms, lj_calculator(), fmax=1e-3, steps=10)
    assert nsteps == 0 and bool(opt.converged) and field.calls == 1


def test_fire_status_recheck_and_start():
    atoms = argon()
    _, _, nsteps, opt, field = fastmace_run(atoms, lj_calculator(), fmax=1e-3, steps=5)
    assert nsteps == 5 and not bool(opt.converged)
    converged, force, energy = opt.status.tolist()
    assert converged == 0.0 and force == float(opt.max_force()) and energy == float(opt.dof.energy()[0])
    before = opt.state_dict()
    opt.set_fmax(1.0)  # a looser target: converged at the same x, the state kept (ASE's second run)
    opt.recheck()
    assert bool(opt.converged) and opt.status[0] == 1.0
    after = opt.state_dict()
    assert after["Nsteps"] == before["Nsteps"] and after["dt"] == before["dt"]
    np.testing.assert_array_equal(after["vel"], before["vel"])
    opt.start()  # back to ASE's state at construction
    assert opt.state_dict()["started"] is False and float(opt.dt) == 0.1 and not opt.vel.any()


@pytest.mark.parametrize("downhill_check", [False, True])
def test_fire_converged_propose_is_a_bitwise_no_op(downhill_check):
    """The engine runs stage A before the host reads the flag: once converged, nothing moves or changes."""
    atoms = argon()
    _, _, _, opt, field = fastmace_run(atoms, lj_calculator(), fmax=1e-3, steps=8, downhill_check=downhill_check)
    opt.set_fmax(1.0)
    opt.recheck()
    assert bool(opt.converged)
    saved = {name: tensor.clone() for name, tensor in opt.buffers().items()}
    positions = opt.dof.state.positions.clone()
    for _ in range(3):
        opt.propose()
    assert torch.equal(opt.dof.state.positions, positions)
    for name, tensor in opt.buffers().items():
        assert torch.equal(tensor, saved[name]), name


# -------------------------------------------------------------------------- on 4-T05's driver and GraphEngine
# test_engine's torch Lennard-Jones on FastMACE's skin neighbour list (ASE's non-smooth LennardJones), relaxed
# through OptimizerStepper on GraphEngine, eagerly and through the replay path that stands in for capture.

DRIVER_CASES = {
    "defaults": {},
    "large-dt": dict(dt=1.0, dtmax=4.0, maxstep_A=0.1, Nmin=1),
    "downhill": dict(dt=1.0, dtmax=6.0, finc=1.5, Nmin=1, maxstep_A=0.3, downhill_check=True),
}


def engine_crystal(fixed=None):
    from test_engine import argon as engine_argon

    atoms = engine_argon(seed=11, displacement=0.15)
    if fixed is not None:
        atoms.set_constraint(FixAtoms(indices=fixed))
    return atoms


def engine_build(record=None, *, atoms=None, replay=False):
    from test_engine import ReplayEngine, TorchLJ

    from fastmace.config import EngineConfig, NeighbourConfig
    from fastmace.engine import constraints
    from fastmace.engine.graph import GraphEngine
    from fastmace.optimize import driver

    record = record or fm.optimize.FIRE()
    atoms = engine_crystal() if atoms is None else atoms
    neighbours = NeighbourConfig(skin=0.3, sort_receivers="off", gpu_neighbour_list="off")
    engine = EngineConfig(pipelined="off", async_output="off", compile_mode="none", device="cpu")
    state = SimulationState.from_atoms(atoms, "cpu", 0)
    config = record.to_config()
    optimizer = build_optimizer(build_dof(state, config), config)
    cls = ReplayEngine if replay else GraphEngine
    out = cls(engine, TorchLJ(atoms, neighbours), driver.OptimizerStepper(optimizer), state, neighbours=neighbours,
              capture=replay, constraints=constraints.ProcessRecord())
    if replay:
        out.captures = []
    return out


def engine_ase(atoms, record):
    from test_engine import EPSILON, RC, SIGMA

    atoms.calc = LennardJones(sigma=SIGMA, epsilon=EPSILON, rc=RC, smooth=False)
    options = {name: getattr(record, name) for name in ("dt", "dtmax", "Nmin", "finc", "fdec", "astart", "fa", "a",
                                                        "downhill_check")}
    return ASEFIRE(atoms, logfile=None, maxstep=record.maxstep_A, **options)


def engine_lockstep(record, *, steps=60, fmax=1e-6, fixed=None, replay=False):
    reference = engine_crystal(fixed)
    ase_opt = engine_ase(reference, record)
    engine = engine_build(record, atoms=engine_crystal(fixed), replay=replay)
    engine.integrator.optimizer.set_fmax(fmax)
    run = ase_opt.irun(fmax=fmax, steps=steps)
    next(run)
    worst = 0.0
    for taken in range(1, steps + 1):
        try:
            next(run)
        except StopIteration:
            break
        engine.step()
        assert not engine.halted
        assert engine.state.step == taken == ase_opt.nsteps
        worst = max(worst, float(np.abs(engine.state.positions.numpy() - reference.positions).max()))
    return reference, engine, worst


@pytest.mark.parametrize("replay", [False, True], ids=["eager", "replayed"])
@pytest.mark.parametrize("name", list(DRIVER_CASES))
def test_driver_positions_match_ase_at_every_step(name, replay):
    record = fm.optimize.FIRE(**DRIVER_CASES[name])
    reference, engine, worst = engine_lockstep(record, replay=replay)
    assert worst < ATOL, worst
    assert engine.force_field.calls == 60 + 1
    if name != "defaults":
        assert engine.integrator.optimizer.state_dict()["restarts"] > 0
    if replay:
        assert {key if isinstance(key, str) else key[0] for key in engine.captures} == {"AC", "B", "C"}


def test_driver_fix_atoms():
    fixed = [0, 5, 17]
    reference, engine, worst = engine_lockstep(fm.optimize.FIRE(), steps=40, fixed=fixed)
    assert worst < ATOL, worst
    assert np.array_equal(engine.state.positions.numpy()[fixed], engine_crystal().positions[fixed])


def engine_relax_both(record, *, fmax, steps):
    from fastmace.optimize import driver

    reference = engine_crystal()
    ase_opt = engine_ase(reference, record)
    converged = bool(ase_opt.run(fmax=fmax, steps=steps))
    engine = engine_build(record)
    return reference, ase_opt, converged, engine, driver.relax(engine, fmax=fmax, steps=steps)


@pytest.mark.parametrize("name", list(DRIVER_CASES))
def test_driver_same_step_count_to_fmax(name):
    record = fm.optimize.FIRE(**DRIVER_CASES[name])
    reference, ase_opt, converged, engine, summary = engine_relax_both(record, fmax=0.01, steps=2000)
    assert converged and summary["converged"] and summary["optimizer"] == "fire"
    assert summary["steps"] == ase_opt.nsteps
    np.testing.assert_allclose(engine.state.positions.numpy(), reference.positions, atol=1e-10)
    assert summary["energy_eV_atom"] == pytest.approx(reference.get_potential_energy() / len(reference), abs=1e-12)
    assert summary["fmax_eV_A"] < 0.01
    assert engine.force_field.calls == summary["steps"] + 1  # the halting step evaluates no forces


def test_driver_converged_start_and_step_limit():
    record = fm.optimize.FIRE()
    _, ase_opt, converged, engine, summary = engine_relax_both(record, fmax=10.0, steps=100)
    assert converged and summary["converged"] and summary["steps"] == ase_opt.nsteps == 0
    assert engine.force_field.calls == 1
    reference, ase_opt, converged, engine, summary = engine_relax_both(record, fmax=1e-6, steps=7)
    assert not converged and not summary["converged"] and summary["steps"] == ase_opt.nsteps == 7
    np.testing.assert_allclose(engine.state.positions.numpy(), reference.positions, atol=ATOL)


def test_driver_second_run_with_a_lower_fmax_continues_as_ase():
    from fastmace.optimize import driver

    record = fm.optimize.FIRE()
    reference = engine_crystal()
    ase_opt = engine_ase(reference, record)
    ase_opt.run(fmax=0.1, steps=1000)
    first = ase_opt.nsteps
    ase_opt.run(fmax=0.01, steps=1000)
    engine = engine_build(record)
    one = driver.relax(engine, fmax=0.1, steps=1000)
    two = driver.relax(engine, fmax=0.01, steps=1000)
    assert one["steps"] == first and one["steps"] + two["steps"] == ase_opt.nsteps
    np.testing.assert_allclose(engine.state.positions.numpy(), reference.positions, atol=1e-10)


def test_driver_buffers_registered_and_restored_around_warm_ups():
    from fastmace.engine.registry import unregistered_mutations

    engine = engine_build(fm.optimize.FIRE(downhill_check=True))
    assert {f"optimizer.{name}" for name in FIRE.BUFFERS} <= set(engine.registry.entries())
    assert unregistered_mutations(engine, lambda: [engine.step() for _ in range(5)]) == []
