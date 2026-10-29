"""``Simulation.update(cell=, positions=)`` on a cell-filter relaxation (plan relax-throughput, task A-T01).

Energy_Hull relaxes hundreds of same-composition frames with a variable cell; a fresh engine per frame
spends most of the wall time on setup. A host write of another frame's cell and positions on a
``FrechetCell`` engine now ends at a canonical point whose ``integrator.start`` restarts the optimiser and
the filter's reference cell, so the next ``relax()`` is a fresh engine's: the same steps, energies, cell and
positions (float64, tiny MH-1-like model on the CPU). A barostat still owns its cell, and a second
``relax()`` without ``update`` still continues the optimiser's history.
"""

from __future__ import annotations

import numpy as np
import pytest
import test_events

from _support.structures import nacl
from fastmace import optimize

loaded, cpu_planner = test_events.loaded, test_events.cpu_planner
RTOL = 1e-10  # the plan's equality: fresh against reused engines, float64


def frame(seed, strain):
    atoms = nacl(repeat=1, rattle=0.08, seed=seed)
    atoms.set_cell(atoms.cell.array @ np.asarray(strain), scale_atoms=True)
    return atoms


FRAME_A = ((1.03, 0.01, 0.0), (0.0, 0.99, 0.0), (0.0, 0.0, 1.02))
FRAME_B = ((0.98, 0.0, 0.02), (0.01, 1.02, 0.0), (0.0, 0.0, 0.99))
FRAME_BIG = ((1.08, 0.0, 0.0), (0.0, 1.07, 0.01), (0.0, 0.0, 1.09))  # more edges at a larger skin: see below


def relaxation(loaded, record, atoms):
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    return Simulation(atoms, ForceField(loaded, atoms, precision="float64"), optimizer=record, seed=1)


def assert_same_relaxation(reused, fresh):
    (relaxed, summary), (expected_atoms, expected) = reused, fresh
    assert summary["steps"] == expected["steps"] and summary["converged"] == expected["converged"]
    for key in ("initial_eV_atom", "energy_eV_atom", "volume_A3"):
        assert summary[key] == pytest.approx(expected[key], rel=RTOL, abs=0), key
    np.testing.assert_allclose(summary["energies_eV"], expected["energies_eV"], rtol=RTOL, atol=0)
    np.testing.assert_allclose(relaxed.cell.array, expected_atoms.cell.array, rtol=RTOL, atol=1e-12)
    np.testing.assert_allclose(relaxed.positions, expected_atoms.positions, rtol=RTOL, atol=1e-12)


RECORDS = [optimize.LBFGS(cell_filter=optimize.FrechetCell()),
           optimize.FIRE(cell_filter=optimize.FrechetCell(scalar_pressure_GPa=0.5))]


# FIRE at 0.5 GPa compresses the tiny model's cell out of the filter's range after about 100 steps: 40 stay inside.
@pytest.mark.parametrize("record, steps", zip(RECORDS, (200, 40)), ids=["lbfgs", "fire-pressure"])
def test_frame_b_on_frame_a_engine_equals_a_fresh_relaxation(loaded, record, steps):
    sim = relaxation(loaded, record, frame(5, FRAME_A))
    sim.relax(fmax=0.01, steps=steps)
    b = frame(9, FRAME_B)
    sim.update(cell=b.cell.array, positions=b.positions)
    committed = sim.committed()
    np.testing.assert_array_equal(committed.cell, b.cell.array)
    np.testing.assert_array_equal(committed.positions, b.positions)
    reused = sim.relax(fmax=0.01, steps=steps)
    assert reused[1]["steps"] > 5  # a real relaxation, not a converged start
    assert_same_relaxation(reused, relaxation(loaded, record, b.copy()).relax(fmax=0.01, steps=steps))


def test_the_filter_starts_from_the_new_cell(loaded):
    """The reference cell (the deformation gradient's origin) is frame B's: ``log F`` is zero and ``x``'s cell rows
    are a fresh filter's."""
    record = RECORDS[0]
    sim = relaxation(loaded, record, frame(5, FRAME_A))
    sim.relax(fmax=0.01, steps=200)
    b = frame(9, FRAME_B)
    sim.update(cell=b.cell.array, positions=b.positions)
    dof = sim._engine.optimizer.dof
    fresh = relaxation(loaded, record, b.copy())._engine.optimizer.dof
    np.testing.assert_array_equal(dof.orig_cell.numpy(), b.cell.array)
    assert not dof.log_deform.any()
    for name in ("x", "gradient", "energy"):
        np.testing.assert_allclose(dof.buffers()[name].numpy(), fresh.buffers()[name].numpy(), rtol=RTOL,
                                   atol=1e-12, err_msg=name)
    state = sim._engine.optimizer.state_dict()
    assert state["iteration"] == 0 and len(state["s"]) == 0


def test_a_larger_cell_with_more_edges_still_matches(loaded):
    """A frame whose list needs more edges (a grown capacity or padded size on the GPU re-captures there)."""
    record = RECORDS[0]
    sim = relaxation(loaded, record, frame(5, FRAME_A))
    edges_before = int(sim._engine.force_field.edge_count)
    sim.relax(fmax=0.05, steps=100)
    big = frame(11, FRAME_BIG)
    sim.update(cell=big.cell.array, positions=big.positions)
    assert int(sim._engine.force_field.edge_count) != edges_before
    reused = sim.relax(fmax=0.05, steps=100)
    assert_same_relaxation(reused, relaxation(loaded, record, big.copy()).relax(fmax=0.05, steps=100))


def test_a_barostat_engine_still_refuses_a_host_cell(loaded):
    sim = test_events.simulation(loaded, integrator=test_events.BAOAB)
    with pytest.raises(ValueError, match="fixed-cell"):
        sim.update(cell=sim.committed().cell)
    with pytest.raises(ValueError, match="barostat"):  # the engine itself, below Simulation's check
        sim._engine.reset_cell(sim.committed().cell)


def test_a_second_relax_without_update_continues_the_history(loaded):
    record = RECORDS[0]
    split = relaxation(loaded, record, frame(5, FRAME_A))
    split.relax(fmax=1e-4, steps=4)
    relaxed, summary = split.relax(fmax=1e-4, steps=4)
    whole_atoms, whole = relaxation(loaded, record, frame(5, FRAME_A)).relax(fmax=1e-4, steps=8)
    assert summary["steps"] == 4 and whole["steps"] == 8
    assert split._engine.optimizer.state_dict()["iteration"] == 8
    np.testing.assert_allclose(relaxed.cell.array, whole_atoms.cell.array, rtol=RTOL, atol=1e-12)
    np.testing.assert_allclose(relaxed.positions, whole_atoms.positions, rtol=RTOL, atol=1e-12)


@pytest.mark.parametrize("name", ["gpu-LBFGS", "gpu-FIRE"])
def test_one_gpu_relaxer_reuses_its_engine_with_a_cell_filter(loaded, name):
    """A-T02: ``GPURelaxer`` moves its variable-cell engine to each next frame of the same atoms."""
    from fastmace.protocols import quench

    frames = [frame(5, FRAME_A), frame(9, FRAME_B), frame(3, FRAME_A)]
    shared = quench.relaxer(name, loaded, precision="float64", cell_filter=optimize.FrechetCell())
    try:
        results = [shared(atoms.copy(), 0.05, 60) for atoms in frames]
    finally:
        shared.close()
    assert shared.builds == 1
    for atoms, (relaxed, summary) in zip(frames, results, strict=True):
        alone = quench.relaxer(name, loaded, precision="float64", cell_filter=optimize.FrechetCell())
        try:
            fresh, expected = alone(atoms.copy(), 0.05, 60)
        finally:
            alone.close()
        assert summary["fixed_cell"] is False
        assert summary["steps"] == expected["steps"]
        assert summary["energy_eV_atom"] == pytest.approx(expected["energy_eV_atom"], rel=RTOL, abs=0)
        np.testing.assert_allclose(relaxed.cell.array, fresh.cell.array, rtol=RTOL, atol=1e-12)
        np.testing.assert_allclose(relaxed.positions, fresh.positions, rtol=RTOL, atol=1e-12)
