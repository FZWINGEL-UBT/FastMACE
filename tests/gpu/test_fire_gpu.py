"""FIRE (plan task 4-T06) on CUDA.

- One optimiser step captured with ``torch.cuda.graph`` replays the eager steps bitwise. A harmonic torch force
  field stands in for the model, so the check covers only the optimiser: no host synchronisation inside a
  step, fixed addresses, and ASE's branches (reset, growth, clip, uphill) selected on the device.
- On 4-T05's driver: ``test_engine_gpu``'s deterministic Lennard-Jones on FastMACE's GPU neighbour list,
  relaxed through ``OptimizerStepper`` on ``GraphEngine``. The captured relaxation (AC and B graphs, the
  converged halt) equals the eager one bit for bit, and agrees with ASE's FIRE on ASE's Lennard-Jones.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.constraints import FixAtoms
from ase.optimize import FIRE as AseFIRE

import fastmace as fm
from fastmace.config import EngineConfig, NeighbourConfig
from fastmace.engine import constraints
from fastmace.engine.graph import GraphEngine
from fastmace.integrators.registry import build_dof, build_optimizer
from fastmace.optimize import driver
from fastmace.state import SimulationState

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

STEPS = 40


def atoms(fixed=False):
    structure = bulk("Ar", "fcc", a=5.26, cubic=True).repeat((3, 3, 3))
    structure.rattle(0.1, seed=6)
    if fixed:
        structure.set_constraint(FixAtoms(indices=[0, 7, 30]))
    return structure


class HarmonicForces:
    """An anisotropic tether, written into the state; graph-capturable. Forces are masked as ASE's
    ``atoms.get_forces()`` with ``FixAtoms``."""

    def __init__(self, state):
        self.x0 = state.positions.clone() + 0.05
        self.k = torch.linspace(0.2, 2.0, 3, dtype=torch.float64, device=state.positions.device)

    def __call__(self, state):
        d = state.positions - self.x0
        forces = -self.k * d
        state.forces.copy_(forces if state.free_mask is None else forces * state.free_mask)
        state.energy.copy_((0.5 * (self.k * d * d).sum()).reshape(1))


CASES = {
    "defaults": (dict(), False),
    "large-dt": (dict(dt=0.8, dtmax=4.0, Nmin=1, maxstep_A=0.04), False),
    "downhill": (dict(dt=1.0, dtmax=6.0, finc=1.5, Nmin=1, maxstep_A=0.3, downhill_check=True), False),
    "fix-atoms": (dict(dt=0.5, dtmax=2.0), True),
}


def build(name):
    options, fixed = CASES[name]
    state = SimulationState.from_atoms(atoms(fixed), "cuda", 0)
    config = fm.optimize.FIRE(**options).to_config(fmax=1e-12)  # never converges within STEPS
    dof = build_dof(state, config)
    opt = build_optimizer(dof, config)
    field = HarmonicForces(state)
    field(state)
    opt.start()
    return state, dof, opt, field


def written(state, opt):
    return {**opt.buffers(), "positions": state.positions, "forces": state.forces, "energy": state.energy}


def capture(state, dof, opt, field):
    tensors = written(state, opt)
    saved = {key: value.clone() for key, value in tensors.items()}

    def step():
        opt.propose()
        field(state)
        dof.refresh()
        opt.update()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for key, value in tensors.items():
        value.copy_(saved[key])
    torch.cuda.synchronize()
    return graph


@pytest.mark.parametrize("name", list(CASES))
def test_captured_fire_step_replays_the_eager_steps(name):
    eager_state, eager_dof, eager_opt, eager_field = build(name)
    state, dof, opt, field = build(name)
    graph = capture(state, dof, opt, field)
    pointers = {key: value.data_ptr() for key, value in written(state, opt).items()}
    start = state.positions.clone()
    for _ in range(STEPS):
        eager_opt.propose()
        eager_field(eager_state)
        eager_dof.refresh()
        eager_opt.update()
        graph.replay()
    torch.cuda.synchronize()
    assert {key: value.data_ptr() for key, value in written(state, opt).items()} == pointers
    reference = written(eager_state, eager_opt)
    for key, value in written(state, opt).items():
        assert torch.equal(value, reference[key]), f"{key}: differs from the eager steps"
    assert not torch.equal(state.positions, start)
    assert bool(opt.started) and not bool(opt.converged)
    if CASES[name][1]:
        assert torch.equal(state.positions[[0, 7, 30]], start[[0, 7, 30]])


# -------------------------------------------------------------------------- on 4-T05's driver and GraphEngine


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_fire_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")
FMAX, MAX_STEPS = 0.002, 600
DRIVER_CASES = {
    "defaults": {},
    "large-dt": dict(dt=1.0, dtmax=4.0, maxstep_A=0.1, Nmin=1),
    "downhill": dict(dt=1.0, dtmax=6.0, finc=1.5, Nmin=1, maxstep_A=0.3, downhill_check=True),
}


def crystal():
    atoms = ENGINE.argon(seed=11)
    atoms.rattle(0.15, seed=11)
    return atoms


def build_engine(record, capture=True):
    atoms = crystal()
    neighbours = NeighbourConfig(skin=0.3, sort_receivers="off", gpu_neighbour_list="on", edge_bucket_fraction=0.0,
                                 edge_multiple=64)
    field = ENGINE.DeterministicLJ(atoms, neighbours)
    state = SimulationState.from_atoms(atoms, "cuda", 0)
    config = record.to_config()
    stepper = driver.OptimizerStepper(build_optimizer(build_dof(state, config), config))
    engine_config = EngineConfig(device="cuda:0", pipelined="off", **ENGINE.IN_PROCESS)
    return GraphEngine(engine_config, field, stepper, state, capture=capture, pad_edges=True,
                       constraints=constraints.ProcessRecord())


def path(engine):
    rows = [engine.state.positions.cpu().numpy().copy()]
    engine.integrator.optimizer.set_fmax(FMAX)
    engine.integrator.optimizer.recheck()
    for _ in range(MAX_STEPS):
        engine.step()
        if engine.halted:
            break
        rows.append(engine.state.positions.cpu().numpy().copy())
    return rows


@pytest.mark.parametrize("name", list(DRIVER_CASES))
def test_captured_relaxation_equals_eager_bitwise(name):
    record = fm.optimize.FIRE(**DRIVER_CASES[name])
    expected = path(build_engine(record, capture=False))
    engine = build_engine(record)
    rows = path(engine)
    assert engine.halted and len(rows) == len(expected) < MAX_STEPS + 1
    for got, want in zip(rows, expected, strict=True):
        assert np.array_equal(got, want), float(np.abs(got - want).max())
    info = engine.info()
    assert {"AC", "C"} <= set(info["graphs"]) and info["neighbour_rebuilds"] >= 1


@pytest.mark.parametrize("name", list(DRIVER_CASES))
def test_relaxation_matches_ase_on_the_gpu(name):
    record = fm.optimize.FIRE(**DRIVER_CASES[name])
    reference = crystal()
    reference.calc = LennardJones(sigma=ENGINE.SIGMA, epsilon=ENGINE.EPSILON, rc=ENGINE.RC, smooth=False)
    options = {key: getattr(record, key) for key in ("dt", "dtmax", "Nmin", "finc", "fdec", "astart", "fa", "a",
                                                    "downhill_check")}
    ase_opt = AseFIRE(reference, logfile=None, maxstep=record.maxstep_A, **options)
    converged = ase_opt.run(fmax=FMAX, steps=MAX_STEPS)
    engine = build_engine(record)
    summary = driver.relax(engine, fmax=FMAX, steps=MAX_STEPS)
    assert converged and summary["converged"]
    assert abs(summary["steps"] - ase_opt.nsteps) <= 2, (summary["steps"], ase_opt.nsteps)
    np.testing.assert_allclose(engine.state.positions.cpu().numpy(), reference.positions, atol=1e-6)
