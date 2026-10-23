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

from fastmace import optimize  # fm.optimize once 4-T01 lists it in fastmace._PUBLIC
from fastmace.config import OptimizerConfig
from fastmace.integrators.registry import build_dof, build_optimizer, optimizer_refusals, registry
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
    config = optimize.FIRE(**options).to_config(fmax=fmax, steps=steps)
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
    assert record.implementation == "fastmace.optimize.fire:FIRE"
    assert None in record.cell_filters and "frechet" in record.cell_filters
    config = optimize.FIRE().to_config()
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
    for name in ("vel", "dt", "a", "nsteps", "started", "e_last", "r_last", "fmax", "max_force_value",
                 "converged_flag"):
        assert buffers[name].device.type == "cpu", name
    assert buffers["vel"].dtype == torch.float64 and buffers["dt"].shape == ()
    assert opt.converged.dtype == torch.bool and opt.converged.shape == ()


def test_fire_converged_at_start_takes_no_step():
    """ASE checks convergence before its first step: a relaxed structure takes none."""
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True)  # forces vanish by symmetry
    _, _, nsteps, opt, field = fastmace_run(atoms, lj_calculator(), fmax=1e-3, steps=10)
    assert nsteps == 0 and bool(opt.converged) and field.calls == 1
