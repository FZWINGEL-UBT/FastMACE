"""GPU variable-cell relaxations on captured graphs (plan task 4-T07; SPEC §10.3), in process on Lennard-Jones argon.

``test_engine_gpu``'s deterministic Lennard-Jones force field with stress on FastMACE's GPU neighbour list,
relaxed by ``DeviceLBFGS`` and ``FIRE`` over the device ``FrechetCellFilter`` from a strained, rattled crystal:
the captured relaxation (``set_x``'s cell and positions, ``set_cell`` and the deformation-aware neighbour test
in AC, ``refresh``'s Fréchet derivative in B, the converged halt) equals the eager one bit for bit, step for
step and to the same step count, and agrees with ASE's optimiser on ASE's ``FrechetCellFilter`` within 1e-6
Angstrom (the CPU tests hold the step-by-step limits).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.calculators.lj import LennardJones
from ase.filters import FrechetCellFilter as AseFrechet
from ase.optimize import FIRE as AseFIRE
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


ENGINE = _module("_cell_filter_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")
FMAX, STEPS = 0.002, 600
RECORDS = {"lbfgs": optimize.LBFGS(cell_filter=optimize.FrechetCell()),
           "fire": optimize.FIRE(cell_filter=optimize.FrechetCell(scalar_pressure_GPa=0.05))}


def crystal():
    atoms = ENGINE.argon(seed=11)
    atoms.rattle(0.1, seed=11)
    atoms.set_cell(atoms.cell.array @ np.array([[1.03, 0.01, 0.0], [0.0, 0.985, 0.02], [0.0, 0.0, 1.03]]),
                   scale_atoms=True)
    return atoms


def build(record, capture=True):
    atoms = crystal()
    neighbours = NeighbourConfig(skin=0.5, sort_receivers="off", gpu_neighbour_list="on", edge_bucket_fraction=0.0,
                                 edge_multiple=64)
    field = ENGINE.DeterministicLJ(atoms, neighbours, compute_stress=True)
    state = SimulationState.from_atoms(atoms, "cuda", 0, compute_stress=True)
    config = record.to_config()
    stepper = driver.OptimizerStepper(build_optimizer(build_dof(state, config), config))
    engine_config = EngineConfig(device="cuda:0", pipelined="off", **ENGINE.IN_PROCESS)
    return GraphEngine(engine_config, field, stepper, state, capture=capture, pad_edges=True,
                       constraints=constraints.ProcessRecord())


def path(engine):
    rows = [(engine.state.positions.cpu().numpy().copy(), engine.state.cell.cpu().numpy().copy())]
    engine.integrator.optimizer.set_fmax(FMAX)
    engine.integrator.optimizer.recheck()
    for _ in range(STEPS):
        engine.step()
        if engine.halted:
            break
        rows.append((engine.state.positions.cpu().numpy().copy(), engine.state.cell.cpu().numpy().copy()))
    return rows


@pytest.mark.parametrize("name", sorted(RECORDS))
def test_captured_relaxation_equals_eager_bitwise(name):
    expected = path(build(RECORDS[name], capture=False))
    engine = build(RECORDS[name])
    rows = path(engine)
    assert engine.halted and len(rows) == len(expected) < STEPS + 1
    for (positions, cell), (want_positions, want_cell) in zip(rows, expected, strict=True):
        assert np.array_equal(positions, want_positions), float(np.abs(positions - want_positions).max())
        assert np.array_equal(cell, want_cell), float(np.abs(cell - want_cell).max())
    assert not np.allclose(rows[-1][1], rows[0][1])  # the cell moved
    assert {"AC", "C"} <= set(engine.info()["graphs"])


@pytest.mark.parametrize("name", sorted(RECORDS))
def test_matches_ase_on_the_gpu(name):
    from ase import units

    record = RECORDS[name]
    reference = crystal()
    reference.calc = LennardJones(sigma=ENGINE.SIGMA, epsilon=ENGINE.EPSILON, rc=ENGINE.RC, smooth=False)
    target = AseFrechet(reference, scalar_pressure=record.cell_filter.scalar_pressure_GPa * units.GPa)
    ase_opt = (AseLBFGS if name == "lbfgs" else AseFIRE)(target, logfile=None)
    converged = ase_opt.run(fmax=FMAX, steps=STEPS)
    engine = build(record)
    summary = driver.relax(engine, fmax=FMAX, steps=STEPS)
    assert converged and summary["converged"]
    assert abs(summary["steps"] - ase_opt.nsteps) <= 2, (summary["steps"], ase_opt.nsteps)
    np.testing.assert_allclose(engine.state.positions.cpu().numpy(), reference.positions, atol=1e-6)
    np.testing.assert_allclose(engine.state.cell.cpu().numpy(), reference.cell.array, atol=1e-6)
