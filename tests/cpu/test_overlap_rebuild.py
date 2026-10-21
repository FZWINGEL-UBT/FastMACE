"""``EngineConfig.overlap_rebuild`` (opt-2 OPT2-T04) on the CPU: carried by the plan, a no-op without CUDA.

An early neighbour rebuild of a pipelined step runs on a side stream beside the force graph in flight.
The CPU has no streams, so here the option must change nothing: the same trajectory bitwise, and the
default engine's ``info()`` without the new counter. The GPU test (tests/gpu/test_overlap_rebuild_gpu.py)
checks the overlapped path itself, bitwise in float64 against the option off.
"""

from __future__ import annotations

import numpy as np
import pytest
import test_minimal_planner as tmp
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace import integrators
from fastmace.config import EngineConfig
from fastmace.planner.workload import Workload

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

quiet_licences = tmp.quiet_licences


@pytest.fixture
def pristine(tmp_path, quiet_licences):
    from fastmace.model import loading

    model = tiny_mace_model([11, 17], **FAMILIES["mh1"])
    path = tmp_path / "tiny-mh1.model"
    torch.save(model, path)
    return loading.load_pristine(str(path), trusted=True)


def simulation(pristine, **engine):
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    atoms = nacl(repeat=1, rattle=0.0, seed=3)
    atoms.set_cell(atoms.cell * 1.01, scale_atoms=True)  # the edge set changes within 50 steps (2-T21d)
    integrator = integrators.Langevin(timestep_fs=1.0, temperature_K=1000.0, friction_per_fs=0.01)
    return Simulation(atoms, ForceField(pristine, atoms, precision="float64"), integrator=integrator, seed=8,
                      workload=Workload(max_temperature_K=1000.0),
                      engine=EngineConfig(device="cpu", compile_mode="none", pipelined="on", **engine))


def test_the_option_is_off_by_default_and_carried_by_the_plan(pristine):
    assert EngineConfig().overlap_rebuild is False
    default, overlapped = simulation(pristine), simulation(pristine, overlap_rebuild=True)
    assert default.plan.engine.overlap_rebuild is False and overlapped.plan.engine.overlap_rebuild is True
    assert overlapped._engine.config.overlap_rebuild is True
    assert "rebuilds_beside" not in default._engine.info()  # the default engine's record is unchanged
    assert overlapped._engine.info()["rebuilds_beside"] == 0
    default.close()
    overlapped.close()


def test_without_cuda_the_option_changes_nothing(pristine):
    """The CPU path: the same rebuilds, on the current stream, and the same trajectory bitwise."""
    runs = []
    for engine in ({}, {"overlap_rebuild": True}):
        sim = simulation(pristine, **engine)
        sim.run(steps=200)
        runs.append((sim.committed(), sim._engine.rebuilds, sim._engine.pipelined_early_rebuilds,
                     sim._engine.rebuilds_beside))
        sim.close()
    (first, rebuilds, early, _), (second, rebuilds_on, early_on, beside) = runs
    assert early > 0, "no early rebuild: the test does not reach the option"
    assert (rebuilds, early) == (rebuilds_on, early_on) and beside == 0
    np.testing.assert_array_equal(first.positions, second.positions)
    np.testing.assert_array_equal(first.forces, second.forces)
    np.testing.assert_array_equal(first.momenta, second.momenta)
