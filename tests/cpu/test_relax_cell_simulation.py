"""``Simulation(optimizer=...(cell_filter=FrechetCell()))`` on the CPU with a tiny MACE model (plan task 4-T07).

The planned variable-cell relaxation (stress on, the deformation-aware neighbour test, the filter's DOF view in
the optimiser) against ASE 3.27's ``LBFGS`` and ``FIRE`` on ``FrechetCellFilter`` over MACE's own calculator
for the same checkpoint, one ``relax(steps=1)`` against one ``run(steps=1)`` at a time, for ``HORIZON`` steps
(the tiny model's landscape is chaotic for LBFGS, ``test_relax_simulation``). The reference is
``test_optimize_cell.ExactCellRows`` and the limits are that module's. ``FixAtoms`` with a cell filter is
refused at the plan (owner decision D1).
"""

from __future__ import annotations

import numpy as np
import pytest
import test_events
from ase.constraints import FixAtoms
from ase.optimize import FIRE as AseFIRE
from ase.optimize import LBFGS as AseLBFGS
from test_optimize_cell import LBFGS_CELL_ATOL, POSITION_ATOL, ase_filter

from _support.structures import nacl
from fastmace import optimize

loaded, cpu_planner = test_events.loaded, test_events.cpu_planner
HORIZON = 12


def structure(fixed=None):
    atoms = nacl(repeat=1, rattle=0.08, seed=5)
    atoms.set_cell(atoms.cell.array @ np.array([[1.03, 0.01, 0.0], [0.0, 0.99, 0.0], [0.0, 0.0, 1.02]]),
                   scale_atoms=True)
    if fixed is not None:
        atoms.set_constraint(FixAtoms(indices=fixed))
    return atoms


def relaxation(loaded, record, atoms=None):
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    atoms = structure() if atoms is None else atoms
    return Simulation(atoms, ForceField(loaded, atoms, precision="float64"), optimizer=record, seed=1)


def ase_side(loaded, record):
    from mace.calculators import MACECalculator

    reference = structure()
    reference.calc = MACECalculator(model_paths=str(loaded.path), device="cpu", default_dtype="float64")
    target = ase_filter(reference, record.cell_filter, exact=True)
    if isinstance(record, optimize.LBFGS):
        return reference, AseLBFGS(target, logfile=None)
    return reference, AseFIRE(target, logfile=None)


def test_the_plan_is_a_variable_cell_relaxation(loaded):
    sim = relaxation(loaded, optimize.LBFGS(cell_filter=optimize.FrechetCell()))
    plan = sim.plan
    assert plan.optimizer.cell_filter == "frechet" and plan.optimizer.variable_cell
    assert plan.decision("compute_stress")[1] == "on"
    assert sim._engine.variable_cell and sim._engine.integrator.dof.rows == len(structure()) + 3


@pytest.mark.parametrize("record, atol", [(optimize.LBFGS(cell_filter=optimize.FrechetCell()), LBFGS_CELL_ATOL),
                                          (optimize.FIRE(cell_filter=optimize.FrechetCell(scalar_pressure_GPa=0.5)),
                                           POSITION_ATOL)], ids=["lbfgs", "fire-pressure"])
def test_positions_and_cell_match_ase_at_every_step(loaded, record, atol):
    sim = relaxation(loaded, record)
    reference, ase_opt = ase_side(loaded, record)
    worst = 0.0
    for taken in range(1, HORIZON + 1):
        relaxed, summary = sim.relax(fmax=1e-5, steps=1)
        ase_opt.run(fmax=1e-5, steps=1)
        assert summary["steps"] == 1 and ase_opt.nsteps == taken
        worst = max(worst, float(np.abs(relaxed.positions - reference.positions).max()),
                    float(np.abs(relaxed.cell.array - reference.cell.array).max()))
    assert worst < atol, worst
    assert not np.allclose(relaxed.cell.array, structure().cell.array)
    assert summary["volume_A3"] == pytest.approx(reference.get_volume(), rel=1e-9)


def test_fix_atoms_with_a_cell_filter_is_refused_at_the_plan(loaded):
    from fastmace.errors import PlanRefused

    with pytest.raises(PlanRefused, match="D1"):
        relaxation(loaded, optimize.LBFGS(cell_filter=optimize.FrechetCell()), structure(fixed=[0]))
