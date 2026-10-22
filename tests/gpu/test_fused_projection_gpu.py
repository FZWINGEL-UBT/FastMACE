"""The fused integrator stage with both fixcm projections on the GPU, bitwise (plan task 3-T22b).

Fast tier, in this process, with ``test_engine_gpu.py``'s deterministic Lennard-Jones force field. Its
replay equals its eager steps bit for bit, so the comparisons are exact. The setup is argon with
alternating masses (40 and 4 amu), with captured graphs:

- **Both projections, fused stage on against off:** 200 steps with rebuilds in the checked and pipelined
  modes give the same positions and velocities, bit for bit. The ``"ase"`` case is OPT3-T03's guarantee,
  kept.
- **``"mass_weighted"`` keeps the total momentum at zero.** A mass event refreshes the fused stage's total
  mass, and the steps after it still equal the PyTorch stage.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from fastmace.config import EngineConfig, IntegratorConfig, NeighbourConfig
from fastmace.engine import constraints
from fastmace.engine.graph import GraphEngine
from fastmace.events import Event, Transaction
from fastmace.integrators.langevin import Langevin
from fastmace.state import SimulationState


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_fused_projection_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")


class LJ(ENGINE.DeterministicLJ):
    """The deterministic Lennard-Jones force field with its skin list under the MACE force field's name, so that
    the fused stage applies (``fused_integrator.applicable``)."""

    @property
    def neighbours(self):
        return self.nl


NEIGHBOURS = NeighbourConfig(skin=0.3, sort_receivers="off", gpu_neighbour_list="on", edge_bucket_fraction=0.0,
                             edge_multiple=64)


def atoms():
    structure = ENGINE.argon()
    masses = structure.get_masses()
    masses[::2] = 4.0
    structure.set_masses(masses)
    return structure


def build(projection, *, fused, mode="pipelined"):
    structure = atoms()
    field = LJ(structure, NEIGHBOURS)
    state = SimulationState.from_atoms(structure, "cuda", 7)
    dyn = Langevin(state, IntegratorConfig(timestep_fs=2.0, friction=0.02, noise_batch=7, fixcm_projection=projection))
    config = EngineConfig(device="cuda:0", pipelined="on" if mode == "pipelined" else "off", fused_integrator=fused,
                          **ENGINE.IN_PROCESS)
    engine = GraphEngine(config, field, dyn, state, capture=True, pad_edges=True,
                         constraints=constraints.ProcessRecord())
    assert (engine._fused is not None) == fused, engine.fused_integrator_reason
    return engine


def run(engine, steps, start=0):
    rows = []
    for step in range(start + 1, start + steps + 1):
        engine.integrator.set_temperature(150.0 * step / steps + 30.0)
        engine.step()
        if step % 50 == 0:
            rows.append(engine.sync())
    return rows


@pytest.mark.parametrize("mode", ["checked", "pipelined"])
@pytest.mark.parametrize("projection", ["ase", "mass_weighted"])
def test_the_fused_stage_equals_the_pytorch_stage_bitwise(projection, mode):
    fused, plain = build(projection, fused=True, mode=mode), build(projection, fused=False, mode=mode)
    got, want = run(fused, 200), run(plain, 200)
    assert fused.info()["neighbour_rebuilds"] >= 1
    ENGINE.assert_bitwise(got, want)
    if projection == "mass_weighted":
        momenta = fused.state.velocities * fused.state.masses
        scale = float(momenta.abs().sum())  # the rounding of 200 steps of sums
        torch.testing.assert_close(momenta.sum(dim=0), torch.zeros(3, dtype=torch.float64, device="cuda"), rtol=0,
                                   atol=1e-13 * scale)


def test_a_mass_event_refreshes_the_fused_total_mass():
    fused, plain = build("mass_weighted", fused=True), build("mass_weighted", fused=False)
    run(fused, 40)
    run(plain, 40)
    masses = fused.state.masses[:, 0].cpu().numpy().copy()
    masses[1::4] *= 3.0
    for engine in (fused, plain):
        Transaction(engine, [Event("masses", values=masses)]).run()
    assert float(fused._fused.total_mass) == float(fused.state.masses.sum())
    got, want = run(fused, 100, start=40), run(plain, 100, start=40)
    ENGINE.assert_bitwise(got, want)
    np.testing.assert_array_equal(fused.sync()["momenta"], plain.sync()["momenta"])
