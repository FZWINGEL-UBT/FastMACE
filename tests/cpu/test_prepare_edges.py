"""The stages ``GraphEngine.prepare()`` captures read the edges of their keys (plan task 2-T21d).

A CUDA graph records the tensors its stage reads, the force field's edge views included
(``force_field.use_edges``: views of the first N compacted edges). The step sets those views before
it runs B (``_step_checked``, ``_step_pipelined``) or ACB (``_step_speculative``), so a lazily
captured stage reads its key's edges. 2-T15b's ``prepare()`` captured B and ACB before the first step
without setting them: the graph keyed by the padded initial edge count kept the build's exact count,
and once the compacted edge count grew past it, the extra edges were dropped from the forces (hybrid
MH-1 NaCl heated to 897 K at a 300 K target, 2-T23).

On the CPU nothing is captured and the edges are not padded (``pad_edges`` follows ``capture``), so
the padded and the exact counts agree and the defect cannot show. The tests build the engines with
padded edges, as on CUDA, switch capture on and replace ``GraphEngine._capture`` with a recorder
(monkeypatched here only): it keeps the edge views that are active at capture time and puts them
back for every replay, as a CUDA graph does.
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
from fastmace.engine import graph
from fastmace.planner import minimal
from fastmace.planner.workload import Workload

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

NA_CL = (11, 17)
VIEWS = ("edge_index", "shifts", "unit_shifts")
MODES = {"checked": dict(pipelined="off"), "pipelined": dict(pipelined="on"),
         "speculative": dict(pipelined="off", speculative_graph=True)}

quiet_licences = tmp.quiet_licences


@pytest.fixture
def pristine(tmp_path, quiet_licences):
    from fastmace.model import loading

    model = tiny_mace_model(list(NA_CL), **FAMILIES["mh1"])
    path = tmp_path / "tiny-mh1.model"
    torch.save(model, path)
    return loading.load_pristine(str(path), trusted=True)


class RecordedGraph:
    """A stage with the edge views active at its capture, as a CUDA graph replays them."""

    def __init__(self, engine, key):
        self.stage = engine._stage(key)
        self.field = engine.force_field
        self.views = {name: self.field.data[name] for name in VIEWS}
        self.edges = int(self.views["edge_index"].shape[1])

    def replay(self):
        current = {name: self.field.data[name] for name in VIEWS}
        self.field.data.update(self.views)
        try:
            self.stage()
        finally:
            self.field.data.update(current)


def pad_edges(monkeypatch):
    """Graph engines round their edges up to the bucket, as with CUDA graphs (``pad_edges=True``)."""
    original = graph.GraphEngine.__init__

    def init(self, *args, **kwargs):
        kwargs["pad_edges"] = True
        original(self, *args, **kwargs)

    monkeypatch.setattr(graph.GraphEngine, "__init__", init)


@pytest.fixture
def recorded_capture(monkeypatch):
    """Graph-tier engines pad and capture (as on CUDA) into ``RecordedGraph``; ``captures`` lists
    (key, edges)."""
    pad_edges(monkeypatch)
    original = minimal.MinimalPlanner.build
    captures = []

    def build(planner, plan, force_field, atoms, **kwargs):
        engine = original(planner, plan, force_field, atoms, **kwargs)
        if plan.tier == "graph":
            engine.capture = True
        return engine

    def capture(self, key):
        recorded = RecordedGraph(self, key)
        captures.append((key, recorded.edges))
        self._graphs[key] = recorded

    monkeypatch.setattr(minimal.MinimalPlanner, "build", build)
    monkeypatch.setattr(graph.GraphEngine, "_capture", capture)
    return captures


def simulation(pristine, mode, temperature_K=300.0):
    """The tiny NaCl cell stretched by 1 %: its second shell (3.99 A at rest) sits just outside the 4 A
    cutoff, so the edge count starts at 48 and grows to about 90 within 50 steps at 300 K."""
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    atoms = nacl(repeat=1, rattle=0.0, seed=3)
    atoms.set_cell(atoms.cell * 1.01, scale_atoms=True)
    integrator = integrators.Langevin(timestep_fs=1.0, temperature_K=temperature_K, friction_per_fs=0.01)
    return Simulation(atoms, ForceField(pristine, atoms, precision="float64"), integrator=integrator, seed=8,
                      workload=Workload(max_temperature_K=temperature_K),
                      engine=EngineConfig(device="cpu", compile_mode="none", **MODES[mode]))


def _force_keys(captures):
    return [(key, edges) for key, edges in captures if isinstance(key, tuple) and key[0] in ("B", "ACB")]


@pytest.mark.parametrize("mode", list(MODES))
def test_prepare_captures_each_force_stage_with_its_keys_edges(pristine, recorded_capture, mode):
    sim = simulation(pristine, mode)
    engine = sim._engine
    if mode == "speculative" and not engine.speculative:
        sim.close()
        pytest.skip("no speculative engine on the CPU")
    assert engine.capture and engine.pad_edges and engine.pipelined == (mode == "pipelined")
    forces = _force_keys(recorded_capture)
    assert forces, recorded_capture  # prepare() captured the first step's force stage
    for key, edges in forces:
        assert edges == key[1], f"{key} was captured reading {edges} edges"
    sim.close()


@pytest.mark.parametrize("mode", list(MODES))
def test_prepared_capture_follows_the_uncaptured_run_past_an_edge_count_change(pristine, recorded_capture,
                                                                               monkeypatch, mode):
    """Bitwise equal to the CPU run without capture, with the edge count growing past the build's."""
    captured = simulation(pristine, mode)
    engine = captured._engine
    if mode == "speculative" and not engine.speculative:
        captured.close()
        pytest.skip("no speculative engine on the CPU")
    counts = []
    for _ in range(10):
        captured.run(steps=10)
        counts.append(int(engine.force_field.edge_count))
    assert max(counts) > engine.initial_edges, (f"the edge count {counts} never passed the build's "
                                                f"{engine.initial_edges}: the test does not reach the defect")
    result = captured.committed()
    captured.close()
    monkeypatch.undo()  # no capture: the CPU's eager stages, on the same padded edges
    pad_edges(monkeypatch)
    reference = simulation(pristine, mode)
    for _ in range(10):
        reference.run(steps=10)
    expected = reference.committed()
    reference.close()
    np.testing.assert_array_equal(result.positions, expected.positions)
    np.testing.assert_array_equal(result.forces, expected.forces)
