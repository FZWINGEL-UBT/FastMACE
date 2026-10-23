"""``Simulation(optimizer=fm.optimize.FIRE())`` and ``sim.relax`` (plan task 4-T06; SPEC §10.3), on the CPU.

The planned FIRE relaxation (``Plan.optimizer``, checked steps) with ``test_relax_simulation``'s tiny MH-1-like
model on rattled NaCl, float64, against ASE 3.27's ``FIRE`` on MACE's own calculator for the same checkpoint:
one ``relax(steps=1)`` against one ``run(steps=1)`` at a time, positions within 1e-12 Angstrom after every step
for ``HORIZON`` steps (the random tiny landscape is chaotic over longer horizons, as ``test_relax_simulation``
records for LBFGS; the step-count criteria run on Lennard-Jones argon in ``test_fire``). ``FixAtoms`` keeps
fixed atoms in place, and ``update(cell=, positions=)`` then ``relax`` equals a fresh relaxation of that frame
bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest
import test_events
from ase.constraints import FixAtoms
from ase.optimize import FIRE as AseFIRE

import fastmace as fm
from _support.structures import nacl

loaded = test_events.loaded
POSITION_ATOL = 1e-12
HORIZON = 12


def structure(seed=3, fixed=None):
    atoms = nacl(repeat=1, rattle=0.12, seed=seed)
    if fixed is not None:
        atoms.set_constraint(FixAtoms(indices=fixed))
    return atoms


def relaxation(loaded, atoms=None, record=None):
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    atoms = structure() if atoms is None else atoms
    return Simulation(atoms, ForceField(loaded, atoms, precision="float64"), optimizer=record or fm.optimize.FIRE(),
                      seed=1)


def ase_side(loaded, atoms, record):
    from mace.calculators import MACECalculator

    reference = atoms.copy()
    reference.calc = MACECalculator(model_paths=str(loaded.path), device="cpu", default_dtype="float64")
    options = {name: getattr(record, name) for name in ("dt", "dtmax", "Nmin", "finc", "fdec", "astart", "fa", "a",
                                                        "downhill_check")}
    return reference, AseFIRE(reference, logfile=None, maxstep=record.maxstep_A, **options)


def test_the_plan_is_a_checked_fire_relaxation(loaded):
    sim = relaxation(loaded)
    plan = sim.plan
    assert plan.optimizer.scheme == "fire" and plan.optimizer.cell_filter is None
    assert plan.engine.pipelined == "off" and plan.engine.speculative_graph is False
    assert plan.decision("optimizer")[1] == "fire"
    assert sim._engine.info()["optimizer"] == "fire"


def lockstep(loaded, record, fixed=None):
    sim = relaxation(loaded, structure(fixed=fixed), record)
    reference, ase_opt = ase_side(loaded, structure(fixed=fixed), record)
    worst = 0.0
    for taken in range(1, HORIZON + 1):
        relaxed, summary = sim.relax(fmax=1e-5, steps=1)
        ase_opt.run(fmax=1e-5, steps=1)
        assert summary["steps"] == 1 and ase_opt.nsteps == taken
        worst = max(worst, float(np.abs(relaxed.positions - reference.positions).max()))
        assert summary["energy_eV_atom"] == pytest.approx(reference.get_potential_energy() / len(reference),
                                                          abs=1e-12)
    return relaxed, worst


@pytest.mark.parametrize("options", [{}, dict(dt=0.5, dtmax=2.0, Nmin=1, maxstep_A=0.05, downhill_check=True)],
                         ids=["defaults", "downhill"])
def test_positions_match_ase_at_every_step(loaded, options):
    relaxed, worst = lockstep(loaded, fm.optimize.FIRE(**options))
    assert worst < POSITION_ATOL, worst


def test_fix_atoms_stay_in_place(loaded):
    fixed = [0, 3]
    relaxed, worst = lockstep(loaded, fm.optimize.FIRE(), fixed)
    assert worst < POSITION_ATOL, worst
    assert np.array_equal(relaxed.positions[fixed], structure(fixed=fixed).positions[fixed])


def test_one_engine_relaxes_another_frame_as_a_fresh_one(loaded):
    """4-T09's quench: frame B on frame A's engine (update with its cell and positions) equals a fresh relaxation,
    FIRE's velocity, dt and a starting afresh."""
    sim = relaxation(loaded)
    sim.relax(fmax=0.01, steps=200)
    frame = structure(seed=9)
    frame.set_cell(frame.cell.array * 1.02, scale_atoms=True)
    sim.update(cell=frame.cell.array, positions=frame.positions)
    relaxed, summary = sim.relax(fmax=0.01, steps=200)
    fresh, expected = relaxation(loaded, frame.copy()).relax(fmax=0.01, steps=200)
    assert summary["steps"] == expected["steps"]
    np.testing.assert_array_equal(relaxed.positions, fresh.positions)
