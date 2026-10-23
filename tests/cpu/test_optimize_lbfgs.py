"""GPU LBFGS and the optimiser driver on the CPU (plan task 4-T05; SPEC §10.3), against ASE 3.27's ``LBFGS``.

The torch Lennard-Jones force field of ``test_engine`` on FastMACE's skin neighbour list, relaxed by
``DeviceLBFGS`` on ``GraphEngine`` (eagerly, and through the replay path that stands in for capture) and by
ASE's ``LBFGS`` on ASE's ``LennardJones`` from the same rattled argon crystal. Positions agree within
1e-12 Angstrom after every step and the runs take the same number of steps to ``fmax``: the two-loop
recursion in matrix form is ASE's to float64 rounding. The ring buffer wraps (``memory=3``), the step is
scaled to ``maxstep`` and damped, ``FixAtoms`` keeps fixed atoms bitwise in place, and a converged start,
a step limit and a second run with a lower ``fmax`` count steps as ASE does.
"""

from __future__ import annotations


import numpy as np
import pytest
import torch
from ase.calculators.lj import LennardJones
from ase.constraints import FixAtoms
from ase.optimize import LBFGS as AseLBFGS
from test_engine import EPSILON, RC, SIGMA, ReplayEngine, TorchLJ, argon

from fastmace import optimize
from fastmace.config import EngineConfig, NeighbourConfig
from fastmace.engine import constraints
from fastmace.engine.graph import GraphEngine
from fastmace.engine.registry import unregistered_mutations
from fastmace.integrators.registry import build_dof, build_optimizer, registry
from fastmace.optimize import driver
from fastmace.optimize.lbfgs import DeviceLBFGS
from fastmace.state import SimulationState

RATTLE = 0.15
POSITION_ATOL = 1e-12


def crystal(fixed=None):
    atoms = argon(seed=11, displacement=RATTLE)
    if fixed is not None:
        atoms.set_constraint(FixAtoms(indices=fixed))
    return atoms


def build(record=None, *, atoms=None, replay=False, engine=None, skin=0.3):
    """A test helper standing in for the planner: force field, state, DOF view, optimiser, engine."""
    record = record or optimize.LBFGS()
    atoms = crystal() if atoms is None else atoms
    neighbours = NeighbourConfig(skin=skin, sort_receivers="off", gpu_neighbour_list="off")
    engine = engine or EngineConfig(pipelined="off", async_output="off", compile_mode="none", device="cpu")
    field = TorchLJ(atoms, neighbours)
    state = SimulationState.from_atoms(atoms, "cpu", 0)
    config = record.to_config()
    optimizer = build_optimizer(build_dof(state, config), config)
    cls = ReplayEngine if replay else GraphEngine
    out = cls(engine, field, driver.OptimizerStepper(optimizer), state, neighbours=neighbours, capture=replay,
              constraints=constraints.ProcessRecord())
    if replay:
        out.captures = []
    return out


def ase_lbfgs(atoms, record=None):
    record = record or optimize.LBFGS()
    atoms.calc = LennardJones(sigma=SIGMA, epsilon=EPSILON, rc=RC, smooth=False)
    return AseLBFGS(atoms, logfile=None, maxstep=record.maxstep_A, memory=record.memory, damping=record.damping,
                    alpha=record.alpha)


def lockstep(record=None, *, steps=50, fmax=1e-6, fixed=None, replay=False):
    """ASE's ``irun`` and the engine side by side; positions compared after every step."""
    reference = crystal(fixed)
    ase_opt = ase_lbfgs(reference, record)
    engine = build(record, atoms=crystal(fixed), replay=replay)
    engine.integrator.optimizer.set_fmax(fmax)
    run = ase_opt.irun(fmax=fmax, steps=steps)
    next(run)  # the initial convergence check
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


# --------------------------------------------------------------------------- the record


def test_lbfgs_is_registered_with_its_support():
    record = registry().optimizer("lbfgs")
    assert record.load() is DeviceLBFGS
    assert record.ase_class == "ase.optimize.lbfgs.LBFGS"
    assert "fix_atoms" in record.constraints
    names = {name for name, _, _ in record.buffers}
    assert set(DeviceLBFGS.BUFFERS) == names


def test_maxstep_above_one_angstrom_is_refused_as_ase_refuses_it():
    with pytest.raises(ValueError, match="much too large"):
        build(optimize.LBFGS(maxstep_A=1.5))


# --------------------------------------------------------------------------- against ASE, step by step


def test_positions_match_ase_at_every_step():
    reference, engine, worst = lockstep()
    assert worst < POSITION_ATOL, worst
    assert engine.force_field.nl.rebuilds > 1  # 0.2 A steps against a 0.3 A skin
    assert engine.force_field.calls == 50 + 1  # one force call per step


def test_the_ring_buffer_wraps_as_ase_pops():
    reference, engine, worst = lockstep(optimize.LBFGS(memory=3), steps=40)
    assert worst < POSITION_ATOL, worst
    assert int(engine.integrator.optimizer.state_dict()["iteration"]) == 40


def test_maxstep_scaling_and_damping():
    record = optimize.LBFGS(maxstep_A=0.004, damping=0.7, alpha=30.0)
    reference, engine, worst = lockstep(record, steps=30)
    assert worst < POSITION_ATOL, worst
    first = build(record)
    start = first.state.positions.clone()
    first.step()
    longest = float(torch.linalg.vector_norm(first.state.positions - start, dim=1).max())
    assert longest == pytest.approx(0.004 * 0.7, rel=1e-12)  # the maxstep branch was taken, then damped


def test_replayed_stages_match_ase():
    reference, engine, worst = lockstep(steps=30, replay=True)
    assert worst < POSITION_ATOL, worst
    kinds = {key if isinstance(key, str) else key[0] for key in engine.captures}
    assert kinds == {"AC", "B", "C"}


def test_fix_atoms_through_the_adjusted_forces():
    fixed = [0, 5, 17]
    reference, engine, worst = lockstep(steps=30, fixed=fixed)
    assert worst < POSITION_ATOL, worst
    start = crystal().positions[fixed]
    assert np.array_equal(engine.state.positions.numpy()[fixed], start)  # bitwise in place


# --------------------------------------------------------------------------- the driver: ASE's step counts


def relax_both(record=None, *, fmax, steps):
    reference = crystal()
    ase_opt = ase_lbfgs(reference, record)
    converged = bool(ase_opt.run(fmax=fmax, steps=steps))
    engine = build(record)
    summary = driver.relax(engine, fmax=fmax, steps=steps)
    return reference, ase_opt, converged, engine, summary


def test_same_step_count_to_fmax():
    reference, ase_opt, converged, engine, summary = relax_both(fmax=0.01, steps=1000)
    assert converged and summary["converged"]
    assert summary["steps"] == ase_opt.nsteps
    np.testing.assert_allclose(engine.state.positions.numpy(), reference.positions, atol=1e-10)
    assert summary["energy_eV_atom"] == pytest.approx(reference.get_potential_energy() / len(reference), abs=1e-12)
    fmax = float(np.linalg.norm(reference.get_forces(), axis=1).max())
    assert summary["fmax_eV_A"] == pytest.approx(fmax, abs=1e-12)
    assert summary["fmax_eV_A"] < 0.01
    assert len(summary["energies_eV"]) == len(summary["fmax_history_eV_A"]) == summary["steps"] + 1
    assert engine.force_field.calls == summary["steps"] + 1  # the halting step evaluates no forces


def test_converged_at_the_start_takes_no_step():
    reference, ase_opt, converged, engine, summary = relax_both(fmax=10.0, steps=100)
    assert converged and summary["converged"] and summary["steps"] == ase_opt.nsteps == 0
    assert engine.force_field.calls == 1
    assert np.array_equal(engine.state.positions.numpy(), crystal().positions)


def test_the_step_limit_is_ases():
    reference, ase_opt, converged, engine, summary = relax_both(fmax=1e-6, steps=7)
    assert not converged and not summary["converged"]
    assert summary["steps"] == ase_opt.nsteps == 7
    np.testing.assert_allclose(engine.state.positions.numpy(), reference.positions, atol=POSITION_ATOL)


def test_a_second_run_with_a_lower_fmax_continues_as_ase():
    reference = crystal()
    ase_opt = ase_lbfgs(reference)
    ase_opt.run(fmax=0.1, steps=1000)
    first = ase_opt.nsteps
    ase_opt.run(fmax=0.01, steps=1000)
    engine = build()
    one = driver.relax(engine, fmax=0.1, steps=1000)
    two = driver.relax(engine, fmax=0.01, steps=1000)
    assert one["steps"] == first and one["steps"] + two["steps"] == ase_opt.nsteps
    np.testing.assert_allclose(engine.state.positions.numpy(), reference.positions, atol=1e-10)


def test_zero_steps_reports_the_start():
    engine = build()
    summary = driver.relax(engine, fmax=1e-6, steps=0)
    assert summary["steps"] == 0 and not summary["converged"]
    assert engine.force_field.calls == 1


# --------------------------------------------------------------------------- refusals


@pytest.mark.parametrize("config", [EngineConfig(pipelined="on", device="cpu", compile_mode="none"),
                                    EngineConfig(pipelined="off", speculative_graph=True, device="cpu",
                                                 compile_mode="none")], ids=["pipelined", "speculative"])
def test_optimisers_run_checked_steps_only(config):
    with pytest.raises(ValueError, match="checked"):
        build(engine=config, replay=config.speculative_graph)


def test_the_stepper_has_no_thermostat():
    engine = build()
    with pytest.raises(NotImplementedError, match="optimiser"):
        engine.integrator.set_temperature(300.0)


def test_optimiser_buffers_are_registered_and_restored_around_warm_ups():
    """Every optimiser buffer has a registry policy; the replay path's warm-ups leave the relaxation ASE's."""
    engine = build()
    assert {f"optimizer.{name}" for name in DeviceLBFGS.BUFFERS} <= set(engine.registry.entries())
    assert unregistered_mutations(engine, lambda: [engine.step() for _ in range(5)]) == []
    reference, replayed, worst = lockstep(steps=10, replay=True)
    assert worst < POSITION_ATOL


def test_matrix_two_loop_equals_ases_loops():
    """The recursion on random history, wrapped ring, against ASE's two loops written out (float64)."""
    generator = np.random.default_rng(3)
    n, memory, iteration = 30, 5, 8
    s = [generator.normal(size=n) for _ in range(memory)]
    y = [si * 2.0 + 0.1 * generator.normal(size=n) for si in s]
    rho = [1.0 / np.dot(yi, si) for yi, si in zip(y, s)]
    q0 = generator.normal(size=n)
    a = np.empty(memory)
    q = q0.copy()
    for i in range(memory - 1, -1, -1):
        a[i] = rho[i] * np.dot(s[i], q)
        q -= a[i] * y[i]
    z = q / 70.0
    for i in range(memory):
        b = rho[i] * np.dot(y[i], z)
        z += s[i] * (a[i] - b)
    # The same history in ring slots, oldest first at slot iteration % memory.
    S, Y = torch.zeros(memory, n, dtype=torch.float64), torch.zeros(memory, n, dtype=torch.float64)
    R = torch.zeros(memory, dtype=torch.float64)
    for k in range(memory):
        slot = (iteration - memory + k) % memory
        S[slot], Y[slot], R[slot] = torch.tensor(s[k]), torch.tensor(y[k]), rho[k]
    out = DeviceLBFGS.two_loop(torch.tensor(q0), S, Y, R, S @ Y.T, torch.tensor(iteration), 1 / 70.0)
    np.testing.assert_allclose(out.numpy(), -z, rtol=1e-13, atol=1e-15)
