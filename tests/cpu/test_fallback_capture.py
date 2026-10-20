"""A capture failure before the first step goes to the fallback (plan task 2-T15b; SPEC §9.4, §8.1).

``GraphEngine`` captured lazily, at the first replay of a stage, so a capture failure surfaced in the
first ``sim.run()`` (``check="skip"``) or as a non-gate error inside the gate's replay
(``check="gate"``), outside the fallback window. Now:

- ``GraphEngine.prepare()`` captures the stages the first step replays, before it, and ``Simulation``
  calls it after the build and before the gate;
- an error raised while capturing is tagged with the stage ``"capture"``, wherever it is raised before
  the first step (``prepare``, the gate's replay), and the fallback takes the plan to eager.

On the CPU nothing is captured, so the tests switch capture on for graph-tier engines and inject the
failure into ``GraphEngine._capture`` (monkeypatched here only, never a hook in the library).
"""

from __future__ import annotations

import pytest
import test_minimal_planner as tmp
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace import integrators
from fastmace.config import FallbackConfig
from fastmace.engine import graph
from fastmace.planner import fallback, minimal

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

NA_CL = (11, 17)
LANGEVIN = integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01)
CAPTURE_ERROR = "operation not permitted when stream is capturing (cudaErrorStreamCaptureUnsupported, injected)"


@pytest.fixture
def pristine(tmp_path, quiet_licences):
    from fastmace.model import loading

    model = tiny_mace_model(list(NA_CL), **FAMILIES["mh1"])
    path = tmp_path / "tiny-mh1.model"
    torch.save(model, path)
    return loading.load_pristine(str(path), trusted=True)


quiet_licences = tmp.quiet_licences


def simulation(pristine, **kwargs):
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    return Simulation(atoms, ForceField(pristine, atoms, precision="float64"), integrator=LANGEVIN, seed=8, **kwargs)


@pytest.fixture
def broken_capture(monkeypatch):
    """Graph-tier engines capture (as on CUDA), and every capture fails; ``captured`` lists the keys tried."""
    original = minimal.MinimalPlanner.build
    captured = []

    def build(planner, plan, force_field, atoms, **kwargs):
        engine = original(planner, plan, force_field, atoms, **kwargs)
        if plan.tier == "graph":
            engine.capture = True
        return engine

    def capture(self, key):
        captured.append(key)
        raise RuntimeError(CAPTURE_ERROR)

    monkeypatch.setattr(minimal.MinimalPlanner, "build", build)
    monkeypatch.setattr(graph.GraphEngine, "_capture", capture)
    return captured


def test_prepare_captures_the_first_steps_stages(pristine, monkeypatch):
    """Checked mode: AC and B at the padded initial edge count; nothing without capture."""
    captured = []
    monkeypatch.setattr(graph.GraphEngine, "_capture", lambda self, key: captured.append(key))
    sim = simulation(pristine)
    engine = sim._engine
    assert not engine.capture and captured == []  # the CPU captures nothing: prepare is a no-op
    engine.capture = True
    engine.prepare()
    keys = ["ACP", ("B", engine._pipe_edges)] if engine.pipelined else ["AC", ("B", engine._padded(
        engine.initial_edges))]
    assert captured == keys
    sim.close()


@pytest.mark.parametrize("check", ["skip", "gate"])
def test_a_capture_failure_before_the_first_step_falls_back_to_eager(pristine, broken_capture, check):
    sim = simulation(pristine, check=check)
    plan = sim.plan
    assert broken_capture, "the capture was not tried before the first step"
    assert plan.tier == "eager" and plan.revision == 1 and plan.predecessor.tier == "graph"
    assert plan.reason.startswith("fallback: mode eager: ") and "cudaErrorStreamCaptureUnsupported" in plan.reason
    assert [t.outcome for t in plan.fallbacks] == ["passed", "activated"]
    assert sim.gate.status == ("passed" if check == "gate" else "skipped")
    assert "plan revision 1 (fallback): mode eager" in sim.report.text()
    sim.run(steps=3)
    sim.close()


def test_a_capture_failure_in_the_gates_replay_is_a_capture_failure(pristine, broken_capture, monkeypatch):
    """(a): with no capture before the gate, the gate's replay captures first, and fails: stage capture."""
    monkeypatch.setattr(graph.GraphEngine, "prepare", lambda self: None)
    stages = []
    original = fallback.Fallback.run

    def run(self, failure):
        stages.append(failure.stage)
        return original(self, failure)

    monkeypatch.setattr(fallback.Fallback, "run", run)
    sim = simulation(pristine, check="gate")
    assert stages == ["capture"] and sim.plan.tier == "eager" and sim.gate.status == "passed"
    sim.close()


def test_refuse_raises_the_capture_error_before_the_first_step(pristine, broken_capture):
    with pytest.raises(RuntimeError, match="cudaErrorStreamCaptureUnsupported") as raised:
        simulation(pristine, fallback=FallbackConfig(fallback="refuse"))
    assert fallback.stage_of(raised.value) == "capture"


def test_capture_errors_are_tagged_and_memory_errors_still_propagate(pristine, monkeypatch):
    sim = simulation(pristine)
    engine = sim._engine
    engine.capture = True

    def oom(self, key):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (injected)")

    monkeypatch.setattr(graph.GraphEngine, "_capture", oom)
    with pytest.raises(torch.cuda.OutOfMemoryError) as raised:
        engine.prepare()
    assert fallback._propagates(raised.value)  # out of memory is never a fallback trigger
    sim.close()


def test_the_default_cpu_path_is_unchanged(pristine):
    """No capture on the CPU: prepare changes nothing, and two runs are bitwise equal."""
    import numpy as np

    first, second = simulation(pristine), simulation(pristine)
    second._engine.prepare()  # once more: still a no-op
    for sim in (first, second):
        sim.run(steps=5)
    np.testing.assert_array_equal(first.committed().positions, second.committed().positions)
    np.testing.assert_array_equal(first.committed().forces, second.committed().forces)
    assert first.plan.revision == second.plan.revision == 0
    first.close()
    second.close()
