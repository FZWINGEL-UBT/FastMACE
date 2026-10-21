"""``EngineConfig.fused_integrator`` (opt-3 OPT3-T03) on the CPU: carried by the plan, a no-op without CUDA.

The fused AC/ACP stage is Triton on CUDA, so on the CPU the option must change nothing: the PyTorch stage,
the same trajectory bitwise, and ``info()`` saying why. The GPU tests (tests/gpu/test_fused_integrator_gpu.py)
check the fused stage itself, bitwise against the PyTorch stage.
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


def simulation(pristine, pipelined="on", **engine):
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    atoms = nacl(repeat=1, rattle=0.0, seed=3)
    atoms.set_cell(atoms.cell * 1.01, scale_atoms=True)  # the edge set changes within 50 steps (2-T21d)
    integrator = integrators.Langevin(timestep_fs=1.0, temperature_K=1000.0, friction_per_fs=0.01)
    return Simulation(atoms, ForceField(pristine, atoms, precision="float64"), integrator=integrator, seed=8,
                      workload=Workload(max_temperature_K=1000.0),
                      engine=EngineConfig(device="cpu", compile_mode="none", pipelined=pipelined, **engine))


def test_the_option_is_off_by_default_and_carried_by_the_plan(pristine):
    assert EngineConfig().fused_integrator is False
    default, fused = simulation(pristine), simulation(pristine, fused_integrator=True)
    assert default.plan.engine.fused_integrator is False and fused.plan.engine.fused_integrator is True
    assert "fused_integrator" not in default._engine.info()  # the switched-off engine's record is the old one
    assert fused._engine.info()["fused_integrator"] == "not a CUDA engine"
    assert fused._engine._fused is None
    default.close()
    fused.close()


@pytest.mark.parametrize("pipelined", ["on", "off"])
def test_without_cuda_the_option_changes_nothing(pristine, pipelined):
    runs = []
    for engine in ({}, {"fused_integrator": True}):
        sim = simulation(pristine, pipelined=pipelined, **engine)
        sim.run(steps=120)
        runs.append((sim.committed(), sim._engine.rebuilds))
        sim.close()
    (first, rebuilds), (second, rebuilds_on) = runs
    assert rebuilds > 0 and rebuilds == rebuilds_on
    np.testing.assert_array_equal(first.positions, second.positions)
    np.testing.assert_array_equal(first.forces, second.forces)
    np.testing.assert_array_equal(first.momenta, second.momenta)


def test_the_compaction_identity():
    """The scatter kernel's destinations: cumsum(1 - inside) == i + 1 - cumsum(inside) in integers."""
    inside = (torch.rand(10_000, generator=torch.Generator().manual_seed(1)) < 0.3).to(torch.int64)
    cum = torch.cumsum(inside, 0)
    count = inside.sum()
    index = torch.arange(inside.numel())
    reference = torch.where(inside.bool(), cum - 1, count + torch.cumsum(1 - inside, 0) - 1)
    fused = torch.where(inside != 0, cum - 1, count + (index + 1 - cum) - 1)
    assert torch.equal(reference, fused) and count == cum[-1]
