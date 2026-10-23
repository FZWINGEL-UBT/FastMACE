"""``Simulation.update(cell=)`` at fixed cell (plan task 4-T05, for 4-T09's quench frames on one engine).

A host write of the cell and the positions of another structure equals a fresh ``Simulation`` built on it:
the engine's whole state (``events.check_against``) and the following MD steps, bit for bit on the CPU in
float64 with the tiny MH-1-like model. A variable-cell run, a wrong shape and a left-handed cell are refused.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_events import BAOAB, LANGEVIN, cpu_planner, fresh_simulation, loaded, simulation  # noqa: F401

from _support.structures import nacl
from fastmace import events


def other_frame(scale=1.03, seed=8):
    atoms = nacl(repeat=1, rattle=0.04, seed=seed)
    atoms.set_cell(atoms.cell.array * np.array([scale, 1.0 / scale ** 0.5, 1.01]), scale_atoms=True)
    return atoms


def test_a_new_cell_and_positions_equal_a_fresh_simulation(loaded):
    sim = simulation(loaded)
    sim.run(steps=3)
    frame = other_frame()
    sim.update(positions=frame.positions, cell=frame.cell.array)
    committed = sim.committed()
    np.testing.assert_array_equal(committed.cell, frame.cell.array)
    np.testing.assert_array_equal(committed.positions, frame.positions)
    fresh = fresh_simulation(loaded, sim)
    assert events.check_against(sim._engine, fresh._engine) == []
    for name in ("forces", "energy"):
        np.testing.assert_array_equal(getattr(committed, name), getattr(fresh.committed(), name), err_msg=name)
    sim.run(steps=3)
    for _ in range(3):
        fresh._engine.step()
    for name in ("positions", "momenta", "forces", "energy"):
        np.testing.assert_array_equal(getattr(sim.committed(), name), getattr(fresh.committed(), name), err_msg=name)


def test_the_cell_round_trips_to_the_first_frame(loaded):
    """Back to the starting cell and positions: the forces are the start's again, bit for bit."""
    sim = simulation(loaded)
    start = sim.committed()
    frame = other_frame(scale=0.97, seed=2)
    sim.update(positions=frame.positions, cell=frame.cell.array)
    assert not np.array_equal(sim.committed().forces, start.forces)
    sim.update(positions=start.positions, cell=start.cell)
    np.testing.assert_array_equal(sim.committed().forces, start.forces)
    np.testing.assert_array_equal(sim.committed().energy, start.energy)


def test_a_variable_cell_run_refuses_a_host_cell(loaded):
    sim = simulation(loaded, integrator=BAOAB)
    with pytest.raises(ValueError, match="fixed-cell"):
        sim.update(cell=sim.committed().cell)


@pytest.mark.parametrize("cell, match", [(np.eye(2), "shape"), (-np.eye(3) * 5.6, "positive volume"),
                                         (np.full((3, 3), np.nan), "finite")], ids=["shape", "left-handed", "nan"])
def test_bad_cells_are_refused_before_anything_is_written(loaded, cell, match):
    sim = simulation(loaded)
    before = sim.committed()
    with pytest.raises(ValueError, match=match):
        sim.update(cell=cell, positions=before.positions + 0.01)
    np.testing.assert_array_equal(sim.committed().positions, before.positions)
