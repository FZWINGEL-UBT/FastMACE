"""``Simulation(optimizer=)`` and ``sim.relax`` (plan task 4-T05; SPEC §10.3), on the CPU with a tiny MACE model.

The tiny MH-1-like model of ``test_events`` on rattled NaCl, float64: the planned relaxation (``Plan.optimizer``,
checked steps, the optimiser in the integrator's place) against ASE 3.27's ``LBFGS`` on MACE's own calculator
for the same checkpoint, one ``relax(steps=1)`` against one ``run(steps=1)`` at a time: positions within 1e-12
Angstrom after every step for ``HORIZON`` steps. The random tiny model's landscape is chaotic for LBFGS: ASE's
own LBFGS, given forces perturbed by 1e-15 eV/A, drifts 2e-13 A by step 10, 1e-11 A by step 20 and 4e-5 A by
step 50, and does not reach fmax 0.01 in 1,000 steps (4-T05, 23 October 2026). So the 50-step and step-count
criteria run on Lennard-Jones argon (``test_optimize_lbfgs``), and here the first ``HORIZON`` steps.
``FixAtoms`` relaxes with fixed atoms in place (owner decision D1); ``update(cell=, positions=)`` then ``relax``
equals a fresh relaxation of that frame bit for bit; the MD entry points refuse an optimiser Simulation and the
other way round.
"""

from __future__ import annotations

import numpy as np
import pytest
import test_events
from ase.constraints import FixAtoms
from ase.optimize import LBFGS as AseLBFGS

from _support.structures import nacl
from fastmace import optimize

loaded, cpu_planner = test_events.loaded, test_events.cpu_planner
POSITION_ATOL = 1e-12
HORIZON = 12


def structure(seed=3, fixed=None):
    atoms = nacl(repeat=1, rattle=0.12, seed=seed)
    if fixed is not None:
        atoms.set_constraint(FixAtoms(indices=fixed))
    return atoms


def relaxation(loaded, atoms=None, record=None, **options):
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    atoms = structure() if atoms is None else atoms
    return Simulation(atoms, ForceField(loaded, atoms, precision="float64"), optimizer=record or optimize.LBFGS(),
                      seed=1, **options)


def ase_side(loaded, atoms):
    from mace.calculators import MACECalculator

    reference = atoms.copy()
    reference.calc = MACECalculator(model_paths=str(loaded.path), device="cpu", default_dtype="float64")
    return reference, AseLBFGS(reference, logfile=None)


def test_the_plan_is_a_checked_relaxation(loaded):
    sim = relaxation(loaded)
    plan = sim.plan
    assert plan.optimizer.scheme == "lbfgs" and plan.optimizer.cell_filter is None
    assert plan.engine.pipelined == "off" and plan.engine.speculative_graph is False
    assert plan.decision("optimizer")[1] == "lbfgs"
    assert plan.decision("compute_stress")[1] == "off"
    assert sim._engine.info()["optimizer"] == "lbfgs"


def lockstep(loaded, fixed=None):
    sim = relaxation(loaded, structure(fixed=fixed))
    reference, ase_opt = ase_side(loaded, structure(fixed=fixed))
    worst = 0.0
    for taken in range(1, HORIZON + 1):
        relaxed, summary = sim.relax(fmax=1e-5, steps=1)
        ase_opt.run(fmax=1e-5, steps=1)
        assert summary["steps"] == 1 and ase_opt.nsteps == taken
        worst = max(worst, float(np.abs(relaxed.positions - reference.positions).max()))
        assert summary["energy_eV_atom"] == pytest.approx(reference.get_potential_energy() / len(reference),
                                                          abs=1e-12)
    return relaxed, worst


def test_positions_match_ase_at_every_step(loaded):
    relaxed, worst = lockstep(loaded)
    assert worst < POSITION_ATOL, worst
    assert relaxed.calc is None and not relaxed.get_momenta().any()


def test_fix_atoms_stay_in_place(loaded):
    fixed = [0, 3]
    relaxed, worst = lockstep(loaded, fixed)
    assert worst < POSITION_ATOL, worst
    assert np.array_equal(relaxed.positions[fixed], structure(fixed=fixed).positions[fixed])


def test_one_engine_relaxes_another_frame_as_a_fresh_one(loaded):
    """4-T09's quench: frame B on frame A's engine (update with its cell and positions) equals a fresh relaxation."""
    sim = relaxation(loaded)
    sim.relax(fmax=0.01, steps=1000)
    frame = structure(seed=9)
    frame.set_cell(frame.cell.array * 1.02, scale_atoms=True)
    sim.update(cell=frame.cell.array, positions=frame.positions)
    relaxed, summary = sim.relax(fmax=0.01, steps=1000)
    fresh, expected = relaxation(loaded, frame.copy()).relax(fmax=0.01, steps=1000)
    assert summary["steps"] == expected["steps"]
    np.testing.assert_array_equal(relaxed.positions, fresh.positions)
    np.testing.assert_array_equal(relaxed.cell.array, fresh.cell.array)


def test_entry_points_refuse_the_other_kind(loaded):
    from fastmace.simulation import Simulation

    sim = relaxation(loaded)
    with pytest.raises(TypeError, match="relax"):
        sim.run(steps=1)
    with pytest.raises(TypeError, match="restart"):
        sim.save_restart("restart.npz")
    md = test_events.simulation(loaded)
    with pytest.raises(TypeError, match="MD"):
        md.relax(fmax=0.05)
    atoms = structure()
    with pytest.raises(TypeError, match="exactly one"):
        Simulation(atoms, loaded, seed=1)
    with pytest.raises(TypeError, match="exactly one"):
        Simulation(atoms, loaded, integrator=test_events.LANGEVIN, optimizer=optimize.LBFGS(), seed=1)
    with pytest.raises(TypeError, match="fm.optimize"):
        Simulation(atoms, loaded, optimizer=test_events.LANGEVIN, seed=1)


def test_engine_options_that_pipeline_are_refused(loaded):
    from fastmace.config import EngineConfig
    from fastmace.errors import PlanRefused

    with pytest.raises(PlanRefused, match="checked"):
        relaxation(loaded, engine=EngineConfig(device="cpu", compile_mode="none", pipelined="on"))
