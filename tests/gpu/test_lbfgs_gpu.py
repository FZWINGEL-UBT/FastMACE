"""GPU LBFGS on captured graphs (plan task 4-T05; SPEC §10.3), in process on the GPU's Lennard-Jones argon.

``test_engine_gpu``'s deterministic Lennard-Jones force field on FastMACE's GPU neighbour list, relaxed by
``DeviceLBFGS``: the captured relaxation (AC and B graphs, the triangular solves and the ring updates inside
them, the converged halt) equals the eager one bit for bit, step for step and to the same step count, and
agrees with ASE's LBFGS on ASE's Lennard-Jones within float64 rounding. A host write of the cell
(``GraphEngine.reset_cell``) keeps the captured graphs when the edge size holds.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.calculators.lj import LennardJones
from ase.optimize import LBFGS as AseLBFGS

from fastmace import optimize
from fastmace.config import EngineConfig, NeighbourConfig
from fastmace.engine import constraints
from fastmace.engine.graph import GraphEngine
from fastmace.integrators.registry import build_dof, build_optimizer
from fastmace.optimize import driver
from fastmace.state import SimulationState

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_lbfgs_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")
FMAX, STEPS = 0.002, 400


class CellLJ(ENGINE.DeterministicLJ):
    """The deterministic Lennard-Jones force field with a host cell write (``reset_cell``)."""

    def reset_cell(self, cell):
        self.nl.reset_cell(cell.detach().cpu().numpy())


def crystal():
    atoms = ENGINE.argon(seed=11)
    atoms.rattle(0.15, seed=11)
    return atoms


def build(capture=True, record=None, atoms=None):
    atoms = crystal() if atoms is None else atoms
    neighbours = NeighbourConfig(skin=0.3, sort_receivers="off", gpu_neighbour_list="on", edge_bucket_fraction=0.0,
                                 edge_multiple=64)
    field = CellLJ(atoms, neighbours)
    state = SimulationState.from_atoms(atoms, "cuda", 0)
    config = (record or optimize.LBFGS()).to_config()
    stepper = driver.OptimizerStepper(build_optimizer(build_dof(state, config), config))
    engine_config = EngineConfig(device="cuda:0", pipelined="off", **ENGINE.IN_PROCESS)
    return GraphEngine(engine_config, field, stepper, state, capture=capture, pad_edges=True,
                       constraints=constraints.ProcessRecord())


def path(engine):
    rows = [engine.state.positions.cpu().numpy().copy()]
    engine.integrator.optimizer.set_fmax(FMAX)
    engine.integrator.optimizer.recheck()
    for _ in range(STEPS):
        engine.step()
        if engine.halted:
            break
        rows.append(engine.state.positions.cpu().numpy().copy())
    return rows


@pytest.mark.parametrize("memory", [100, 4])
def test_captured_relaxation_equals_eager_bitwise(memory):
    record = optimize.LBFGS(memory=memory)
    expected = path(build(capture=False, record=record))
    engine = build(record=record)
    rows = path(engine)
    assert engine.halted and len(rows) == len(expected) < STEPS + 1
    for got, want in zip(rows, expected, strict=True):
        assert np.array_equal(got, want), float(np.abs(got - want).max())
    info = engine.info()
    assert {"AC", "C"} <= set(info["graphs"]) and info["neighbour_rebuilds"] >= 1


def test_matches_ase_on_the_gpu():
    reference = crystal()
    reference.calc = LennardJones(sigma=ENGINE.SIGMA, epsilon=ENGINE.EPSILON, rc=ENGINE.RC, smooth=False)
    ase_opt = AseLBFGS(reference, logfile=None)
    converged = ase_opt.run(fmax=FMAX, steps=STEPS)
    engine = build()
    summary = driver.relax(engine, fmax=FMAX, steps=STEPS)
    assert converged and summary["converged"]
    assert abs(summary["steps"] - ase_opt.nsteps) <= 2, (summary["steps"], ase_opt.nsteps)
    np.testing.assert_allclose(engine.state.positions.cpu().numpy(), reference.positions, atol=1e-6)


def test_a_new_cell_keeps_the_captured_graphs():
    """The cell write and its canonical point drop no graph, and the stages captured before are replayed after."""
    engine = build()
    driver.relax(engine, fmax=0.05, steps=50)
    invalidations, logged = len(engine.invalidations), len(engine.recapture_log)
    atoms = crystal()
    atoms.set_cell(atoms.cell.array * 1.002, scale_atoms=True)
    engine.reset_cell(atoms.cell.array)
    engine.state.positions.copy_(torch.as_tensor(atoms.positions, device="cuda"))
    engine.canonicalise(reevaluate=True)
    reference = build(capture=False, atoms=atoms)
    np.testing.assert_allclose(engine.state.forces.cpu().numpy(), reference.state.forces.cpu().numpy(), atol=1e-12)
    driver.relax(engine, fmax=0.05, steps=5)
    assert len(engine.invalidations) == invalidations
    assert not [entry for entry in engine.recapture_log[logged:] if entry["key"] in ("AC", "C")]
