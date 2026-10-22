"""3b-T07: an NPT run whose cell diverges stops cleanly; a CPU cell list too large to allocate is refused.

The full NbCl5 melt-quench at 0 GPa vaporised and its cell grew without bound (7e12 Å³/atom) until a
non-finite energy stopped it, and a CPU neighbour search of such a cell exhausted the memory of the
machine (docs/validation/3b-T07.md). Now:
- ``LangevinBAOAB(max_volume_ratio=20, max_cell_length_A=1000)``: a ``Simulation`` checks the cell at
  every thermo row and at the end of every ``run()``, and stops with ``CellDivergedError`` once V/V0
  leaves [1/ratio, ratio] or a cell vector is longer than the limit. It writes ``metadata.json`` and a
  restart first. V0 is the run's starting volume, kept across a resume in ``simulation.json``;
- ``SkinNeighbourList.build_cpu`` refuses a cell whose matscipy cell list would need more than
  ``MAX_CPU_CELL_LIST_BINS`` bins (``CellListTooLargeError``), before matscipy allocates them.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from _support.structures import nacl
from fastmace import integrators
from fastmace.errors import CellDivergedError, CellListTooLargeError, FastMACEError

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

NA_CL = (11, 17)


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    pytest.importorskip("cuequivariance_torch")
    from _support.mc_fixtures import tiny_loaded

    return tiny_loaded(tmp_path_factory.mktemp("models"), NA_CL)


@pytest.fixture
def planner(monkeypatch):
    from _support.mc_fixtures import cpu_planner

    return cpu_planner(monkeypatch)


def _simulation(loaded, atoms, output, **limits):
    from fastmace.io.output import Output
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    # Tension: the cell grows steadily from the first steps.
    npt = integrators.LangevinBAOAB(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01,
                                    pressure_GPa=-2.0, barostat_tau_fs=100.0, **limits)
    field = ForceField(loaded, atoms, precision="float64", species=list(NA_CL))
    return Simulation(atoms, field, integrator=npt, seed=5,
                      output=None if output is None else Output(str(output), trajectory_every=None, thermo_every=2,
                                                                restart_every=None))


# --------------------------------------------------------------------------- the record


def test_the_limits_default_to_20_and_1000_A_and_are_checked():
    record = integrators.LangevinBAOAB(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01)
    assert (record.max_volume_ratio, record.max_cell_length_A) == (20.0, 1000.0)
    off = integrators.LangevinBAOAB(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01,
                                    max_volume_ratio=None, max_cell_length_A=None)
    assert (off.max_volume_ratio, off.max_cell_length_A) == (None, None)
    for bad in ({"max_volume_ratio": 1.0}, {"max_volume_ratio": 0.5}, {"max_volume_ratio": float("inf")},
                {"max_cell_length_A": 0.0}, {"max_cell_length_A": -5.0}, {"max_volume_ratio": True}):
        with pytest.raises(ValueError, match="max_"):
            integrators.LangevinBAOAB(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01, **bad)


def test_the_error_names_the_limit_and_the_next_step():
    error = CellDivergedError("V/V0 = 25.1 at step 120", why="the cell diverged")
    assert isinstance(error, FastMACEError) and "max_volume_ratio" in error.next_step
    assert error.record == "metadata.json"


# --------------------------------------------------------------------------- the stop


def test_a_diverging_npt_run_stops_with_metadata_and_a_restart(loaded, planner, tmp_path):
    atoms = nacl(2, rattle=0.02, seed=1)
    sim = _simulation(loaded, atoms, tmp_path, max_volume_ratio=1.02)
    with pytest.raises(CellDivergedError, match="V/V0") as caught:
        sim.run(steps=400)
    step = caught.value.step
    assert 0 < step < 400
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["timing"]["status"] == "cell-diverged"
    assert (tmp_path / "restart.npz").is_file()
    record = json.loads((tmp_path / "simulation.json").read_text(encoding="utf-8"))
    assert record["reference_volume_A3"] == pytest.approx(atoms.get_volume())
    assert int(sim.committed().step) >= step


def test_the_cell_length_limit_stops_a_run_without_output(loaded, planner):
    atoms = nacl(2, rattle=0.02, seed=1)
    length = float(np.linalg.norm(atoms.cell[:], axis=1).max())
    sim = _simulation(loaded, atoms, None, max_cell_length_A=length * 1.005)
    with pytest.raises(CellDivergedError, match="cell vector"):
        for _ in range(40):
            sim.run(steps=10)


def test_without_limits_the_same_run_continues(loaded, planner):
    atoms = nacl(2, rattle=0.02, seed=1)
    sim = _simulation(loaded, atoms, None, max_volume_ratio=None, max_cell_length_A=None)
    sim.run(steps=10)
    volume = abs(np.linalg.det(np.asarray(sim.committed().cell)))
    assert volume / atoms.get_volume() > 1.02  # past where max_volume_ratio=1.02 stops the same run


def test_a_resume_keeps_the_reference_volume(loaded, planner, tmp_path):
    from fastmace.simulation import Simulation

    atoms = nacl(2, rattle=0.02, seed=1)
    sim = _simulation(loaded, atoms, tmp_path, max_volume_ratio=5.0)
    sim.run(steps=10)
    sim.save_restart(tmp_path)
    grown = sim.committed().cell
    sim.close()
    assert abs(np.linalg.det(np.asarray(grown)) - atoms.get_volume()) > 1e-6  # the cell moved
    resumed = Simulation.resume(tmp_path)
    assert resumed.reference_volume_A3 == pytest.approx(atoms.get_volume())


# --------------------------------------------------------------------------- the CPU cell list


def test_a_cpu_cell_list_that_would_explode_is_refused_before_allocating():
    from fastmace.neighbors.skin import MAX_CPU_CELL_LIST_BINS, SkinNeighbourList, cpu_cell_list_bins

    cutoff = 5.0
    side = 2.4e5  # the vaporised NbCl5 cell's side at the stop
    assert cpu_cell_list_bins(np.eye(3) * side, cutoff) > MAX_CPU_CELL_LIST_BINS
    assert cpu_cell_list_bins(np.eye(3) * 60.0, cutoff) == 12 ** 3
    neighbours = SkinNeighbourList(np.eye(3) * side, cutoff, torch.device("cpu"), torch.float64,
                                   gpu_build=False, sort_receivers=False)
    positions = np.random.default_rng(0).random((8, 3)) * 10.0
    with pytest.raises(CellListTooLargeError, match="bins"):
        neighbours.build_cpu(positions)
