"""Mutation events on the GPU with captured CUDA graphs (plan task 3-T03; SPEC §8.5, §12.2 "Mutation events").

Fast tier, in this process, with ``test_engine_gpu.py``'s deterministic Lennard-Jones force field (no MACE;
its replay equals its eager steps bit for bit, so the comparisons here are bitwise):

- a refresh function of the derived-state registry is capturable: the mass refreshes (Langevin's noise
  coefficients; BAOAB's ``inv_sqrt_masses`` and the cell force ``force_eps``) captured in a CUDA graph and
  replayed after a new mass write give a fresh construction's buffers;
- the first captured MD step after an accepted mass or friction event equals a fresh construction from
  the committed state (checked and pipelined Langevin, hydrostatic and full-cell BAOAB). A friction event
  must drop the captured stages, which bake ``c1``/``c2`` in: a replay of the old graphs would differ;
- a rejected event restores the state bit for bit, and the run continues as one without it.
"""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from fastmace import events
from fastmace.config import EngineConfig, IntegratorConfig, NeighbourConfig
from fastmace.engine import constraints
from fastmace.engine.graph import GraphEngine
from fastmace.events import Event, Transaction
from fastmace.integrators.baoab import LangevinBAOAB
from fastmace.integrators.langevin import Langevin
from fastmace.state import SimulationState


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_events_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")
NEIGHBOURS = NeighbourConfig(skin=0.3, sort_receivers="off", gpu_neighbour_list="on", edge_bucket_fraction=0.0,
                             edge_multiple=64)
CASES = [("checked", False, "hydrostatic"), ("pipelined", False, "hydrostatic"), ("checked", True, "hydrostatic"),
         ("pipelined", True, "full")]


def build(atoms=None, *, mode="checked", npt=False, cell_mode="hydrostatic", friction=0.02):
    atoms = ENGINE.argon() if atoms is None else atoms
    field = ENGINE.DeterministicLJ(atoms, NEIGHBOURS, compute_stress=npt)
    state = SimulationState.from_atoms(atoms, "cuda", 7, compute_stress=npt)
    if npt:
        dyn = LangevinBAOAB(state, IntegratorConfig(ensemble="npt", timestep_fs=2.0, friction=friction, noise_batch=7,
                                                    pressure_GPa=0.5, barostat_mass_factor=0.01,
                                                    cell_mode=cell_mode))
    else:
        dyn = Langevin(state, IntegratorConfig(timestep_fs=2.0, friction=friction, noise_batch=7))
    config = EngineConfig(device="cuda:0", pipelined="on" if mode == "pipelined" else "off", **ENGINE.IN_PROCESS)
    return GraphEngine(config, field, dyn, state, capture=True, pad_edges=True, constraints=constraints.ProcessRecord())


def run(engine, steps, start=0):
    for step in range(start + 1, start + steps + 1):
        engine.integrator.set_temperature(150.0 * step / 200 + 30.0)
        engine.step()


def committed_atoms(engine, masses=None):
    atoms = ENGINE.argon()
    atoms.set_cell(engine.state.cell.cpu().numpy())
    atoms.set_positions(engine.state.positions.cpu().numpy())
    atoms.set_masses(engine.state.masses[:, 0].cpu().numpy() if masses is None else masses)
    return atoms


def take_dynamics(fresh, engine):
    state, source = fresh.state, engine.state
    state.velocities.copy_(source.velocities)
    if source.p_eps is not None:
        state.p_eps.copy_(source.p_eps)
    state.rng.generator = copy.deepcopy(source.rng.generator)
    state.rng.cursor = source.rng.cursor
    state.rng.batch.copy_(source.rng.batch)
    state.step = source.step
    fresh.integrator.set_temperature(engine.integrator.temperature_K)
    fresh.integrator.start(state)


def heavier(engine):
    masses = engine.state.masses[:, 0].cpu().numpy().copy()
    masses[::3] *= 1.7
    masses[1::5] *= 0.6
    return masses


def assert_same(got, want):
    for key in want:
        assert np.array_equal(got[key], want[key]), f"{key}: max |difference| {np.abs(got[key] - want[key]).max()}"


@pytest.mark.parametrize("npt", [False, True])
def test_mass_refreshes_replay_inside_a_cuda_graph(npt):
    engine = build(npt=npt)
    run(engine, 3)
    engine.flush()
    refresh = engine.derived.affected(("masses",))
    assert refresh
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):  # warm-up outside the capture
        engine.derived.refresh(("masses",))
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        engine.derived.refresh(("masses",))
    masses = heavier(engine)
    engine.state.masses.copy_(torch.as_tensor(masses, device="cuda")[:, None])
    graph.replay()
    torch.cuda.synchronize()
    fresh = build(committed_atoms(engine, masses), npt=npt)
    take_dynamics(fresh, engine)
    for name in refresh:
        assert events._same(engine.derived[name].buffer, fresh.derived[name].buffer), name


@pytest.mark.parametrize(("mode", "npt", "cell_mode"), CASES)
@pytest.mark.parametrize("kind", ["masses", "friction"])
def test_first_captured_step_after_an_accepted_event_equals_a_fresh_construction(mode, npt, cell_mode, kind):
    engine = build(mode=mode, npt=npt, cell_mode=cell_mode)
    run(engine, 9)
    assert engine._graphs
    masses = heavier(engine) if kind == "masses" else None
    event = Event("masses", values=masses) if kind == "masses" else Event("friction", values=0.035)
    assert Transaction(engine, [event]).run() == "accepted"
    if kind == "friction":
        assert not engine._graphs  # the stages bake c1, c2 (and BAOAB's constants) in
    fresh = build(committed_atoms(engine, masses), mode=mode, npt=npt, cell_mode=cell_mode,
                  friction=0.035 if kind == "friction" else 0.02)
    take_dynamics(fresh, engine)
    assert events.check_against(engine, fresh) == []
    run(engine, 3, start=9)
    run(fresh, 3, start=9)
    assert engine._graphs and fresh._graphs
    assert_same(engine.sync(), fresh.sync())


@pytest.mark.parametrize(("mode", "npt", "cell_mode"), CASES[1:2] + CASES[3:])
def test_a_rejected_event_restores_the_captured_run(mode, npt, cell_mode):
    engine = build(mode=mode, npt=npt, cell_mode=cell_mode)
    twin = build(mode=mode, npt=npt, cell_mode=cell_mode)
    for one in (engine, twin):
        run(one, 9)
        one.canonicalise()
    changes = [Event("masses", values=heavier(engine)), Event("friction", values=0.035)]

    def evaluate(eng):
        eng.integrator.prepare_step(eng.state)  # uses random numbers
        return "rejected"

    assert Transaction(engine, changes, evaluate=evaluate).run() == "rejected"
    assert events.check_against(engine, twin) == []
    run(engine, 3, start=9)
    run(twin, 3, start=9)
    assert_same(engine.sync(), twin.sync())
