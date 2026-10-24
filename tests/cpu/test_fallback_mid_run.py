"""Mid-run fallbacks (plan task 5-T08; SPEC §9.4 step 5, §18 decision 54).

- A capture error at a re-capture during ``run()`` goes back to the last committed canonical point (the last
  restart write, else the setup's state), runs the §9.4 search on the committed structure and continues on the
  validated revision: the run reaches its last step, the trajectory equals the unbroken run's (on the CPU the
  compiled tier steps as the graph tier does, 3-T06), the chain holds both revisions with the point's step, and
  ``counters.fallbacks``, the decision ``fallback.mid_run`` (rule 14) and the report's warning record it.
- The output chunks after the point are deleted as on resume: no duplicated or missing thermo rows.
- A compile error of the compiled tier (a Dynamo error at a new edge count) is a trigger too.
- Under ``fallback="refuse"`` the error is raised with the record, and the last restart file stays resumable.
- At most one fallback per canonical interval: a second trigger before the next restart write stops the run.
- Not triggers: out-of-memory, an untagged error, a non-finite energy.
- A resume of the fallback revision rebuilds it as it is.

On the CPU nothing is captured, so the tests switch capture on for graph-tier engines (``recorded_capture``) and
inject the failure into ``GraphEngine._capture`` or ``CompiledEngine.step`` (monkeypatched here only).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import test_adapt_monitor as monitor_test

from fastmace import integrators
from fastmace.config import AdaptationConfig, EngineConfig, FallbackConfig
from fastmace.engine import graph
from fastmace.engine.compiled import CompiledEngine
from fastmace.errors import FallbackExhausted
from fastmace.planner import fallback
from fastmace.planner.workload import Workload

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning"), pytest.mark.rule(14)]

quiet_licences = monitor_test.quiet_licences
pristine = monitor_test.pristine
recorded_capture = monitor_test.recorded_capture

CAPTURE_ERROR = "operation not permitted when stream is capturing (cudaErrorStreamCaptureUnsupported, injected)"


class DynamoError(RuntimeError):
    """A compile error as torch._dynamo raises it (``fallback.stage_of`` reads the module)."""


DynamoError.__module__ = "torch._dynamo.exc"


def simulation(pristine, *, output=None, **options):
    """The monitor tests' tiny stretched NaCl with one graph variant: every new edge size re-captures."""
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    atoms = monitor_test.edges_test.nacl(repeat=1, rattle=0.0, seed=3)
    atoms.set_cell(atoms.cell * 1.01, scale_atoms=True)  # the edge count grows within 50 steps (2-T21d)
    integrator = integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01)
    return Simulation(atoms, ForceField(pristine, atoms, precision="float64"), integrator=integrator, seed=8,
                      workload=Workload(max_temperature_K=300.0), output=output,
                      engine=EngineConfig(device="cpu", compile_mode="none", graph_variant_cap=1),
                      adaptation=AdaptationConfig(adapt="off"), **options)


def output(tmp_path, name, restart_every=None):
    from fastmace.io.output import Output

    return Output(tmp_path / name, trajectory_every=None, restart_every=restart_every, thermo_every=5)


@pytest.fixture
def break_capture(monkeypatch, recorded_capture):
    """``arm(n)``: the n-th capture from now raises (once); ``steps`` lists the step each injected failure hit."""
    recorded = graph.GraphEngine._capture
    state = {"left": None, "steps": []}

    def capture(self, key):
        if state["left"] is not None:
            state["left"] -= 1
            if state["left"] == 0:
                state["left"] = None
                state["steps"].append(int(self.state.step) + 1)
                raise RuntimeError(CAPTURE_ERROR)
        return recorded(self, key)

    monkeypatch.setattr(graph.GraphEngine, "_capture", capture)

    def arm(n=1):
        state["left"] = n

    arm.steps = state["steps"]
    return arm


def after_fallback(engine):
    """Whether ``engine`` runs a mid-run fallback's activated revision (not a candidate being validated)."""
    return any(key == "fallback.mid_run" for key, _, _ in engine.plan.decisions)


def thermo_steps(directory):
    from fastmace.io.output import read_thermo

    header, rows = read_thermo(directory)
    return [int(row[header.index("step")]) for row in rows]


# --------------------------------------------------------------------------- the trigger and the search


def test_a_capture_error_at_a_recapture_falls_back_from_the_setup_state(pristine, break_capture, tmp_path):
    sim = simulation(pristine, output=output(tmp_path, "run"))
    graph_plan = sim.plan
    break_capture()
    sim.run(steps=60)
    (failed,) = break_capture.steps
    assert 0 < failed <= 60
    plan = sim.plan
    assert int(sim._engine.state.step) == 60 and plan.tier != "graph"
    assert fallback.is_mid_run(plan) and plan.step == 0 and plan.predecessor is not None
    assert [entry.revision for entry in plan.chain()][0] == graph_plan.revision
    assert f"from step 0 (the setup state; capture error at step {failed:,}" in str(plan.reason)
    choice, reason = plan.decision("fallback.mid_run")[1:]
    assert choice == f"graph -> {plan.tier}" and reason.rule == 14 and reason.code == fallback.MID_RUN_CODE
    assert any(fallback.MID_RUN_CODE in line and f"step {failed:,}" in line for line in sim.report.warnings())
    assert thermo_steps(tmp_path / "run") == list(range(0, 61, 5))  # the first row once, nothing twice
    sim.close()
    record = json.loads((tmp_path / "run" / "metadata.json").read_text())
    (entry,) = record["counters"]["fallbacks"]
    assert (entry["step"], entry["from"], entry["to"], entry["revision"]) == (0, "graph", plan.tier, plan.revision)
    assert record["counters"]["tier_switches"] == []


def test_the_run_after_the_fallback_keeps_its_physics(pristine, break_capture):
    """The state at the point and its random numbers come from the point: against the unbroken run the
    trajectory agrees to float64 rounding."""
    broken = simulation(pristine)
    break_capture()
    broken.run(steps=60)
    assert break_capture.steps and broken.plan.tier != "graph"
    reference = simulation(pristine)
    reference.run(steps=60)
    assert reference.plan.tier == "graph"
    first, second = broken.committed(), reference.committed()
    assert first.step == second.step == 60
    np.testing.assert_allclose(first.positions, second.positions, rtol=0, atol=1e-9)
    np.testing.assert_allclose(first.momenta, second.momenta, rtol=0, atol=1e-9)
    broken.close()
    reference.close()


def test_the_fallback_starts_from_the_last_restart_point_and_drops_later_chunks(pristine, break_capture, tmp_path):
    sim = simulation(pristine, output=output(tmp_path, "run", restart_every=10))
    sim.run(steps=20)
    assert sim._last_point.step == 20 and sim.plan.tier == "graph"
    break_capture()
    sim.run(steps=40)
    (failed,) = break_capture.steps
    assert failed > 20
    point = sim.plan.step
    assert point == 10 * ((failed - 1) // 10) and point >= 20
    assert f"(the last restart point; capture error at step {failed:,}" in str(sim.plan.reason)
    assert int(sim._engine.state.step) == 60
    assert thermo_steps(tmp_path / "run") == list(range(0, 61, 5))
    assert [entry["step"] for entry in sim._restart_points] == list(range(10, 61, 10))
    sim.close()


def test_a_compile_error_of_the_compiled_tier_is_a_trigger(pristine, monkeypatch):
    original = CompiledEngine.step
    hits = []

    def step(self):
        if int(self.state.step) == 30 and not hits:
            hits.append(True)
            raise DynamoError("backend='inductor' raised: compiling a new edge count (injected)")
        return original(self)

    monkeypatch.setattr(CompiledEngine, "step", step)
    sim = simulation(pristine, mode="compiled")
    assert sim.plan.tier == "compiled"
    sim.run(steps=50)
    assert hits and int(sim._engine.state.step) == 50
    assert fallback.is_mid_run(sim.plan) and sim.plan.step == 0 and "compile error at step 31" in str(sim.plan.reason)
    sim.close()


# --------------------------------------------------------------------------- refuse and the limit


def test_refuse_raises_with_the_record_and_the_restart_stays_resumable(pristine, break_capture, tmp_path):
    from fastmace.simulation import Simulation

    sim = simulation(pristine, output=output(tmp_path, "run", restart_every=10),
                     fallback=FallbackConfig(fallback="refuse"))
    sim.run(steps=20)
    break_capture()
    with pytest.raises(RuntimeError, match="injected") as raised:
        sim.run(steps=40)
    (failed,) = break_capture.steps
    record = raised.value.fastmace_mid_run
    assert record["step"] == failed and record["stage"] == "capture" and record["revision"] == 0
    assert record["point"] == sim._last_point.step
    assert sim.plan.revision == 0
    sim.close()
    resumed = Simulation.resume(tmp_path / "run")
    assert int(resumed._engine.state.step) == record["point"] and resumed.plan.tier == "graph"
    resumed.close()


def test_a_second_failure_in_the_same_interval_stops_the_run(pristine, break_capture, monkeypatch):
    original = CompiledEngine.step

    def step(self):  # the activated revision's step 30, never the candidates' validation
        if int(self.state.step) == 30 and after_fallback(self):
            raise DynamoError("backend='inductor' raised (injected)")
        return original(self)

    monkeypatch.setattr(CompiledEngine, "step", step)
    sim = simulation(pristine)
    break_capture()
    with pytest.raises(FallbackExhausted) as raised:
        sim.run(steps=60)
    assert raised.value.codes == (fallback.MID_RUN_LIMIT_CODE,)
    assert raised.value.previous["revision"] == sim.plan.revision and raised.value.previous["step"] == 0
    assert isinstance(raised.value.__cause__, DynamoError)
    sim.close()


def test_a_restart_write_opens_a_new_interval(pristine, break_capture, monkeypatch, tmp_path):
    original = CompiledEngine.step
    hits = []

    def step(self):
        if int(self.state.step) == 45 and not hits and after_fallback(self):
            hits.append(True)
            raise DynamoError("backend='inductor' raised (injected)")
        return original(self)

    monkeypatch.setattr(CompiledEngine, "step", step)
    sim = simulation(pristine, output=output(tmp_path, "run", restart_every=10))
    break_capture()
    sim.run(steps=60)
    (failed,) = break_capture.steps
    assert failed < 40 and hits  # the capture failure, then a restart write, then the compile failure
    assert int(sim._engine.state.step) == 60
    first, second = fallback.mid_run_fallbacks(sim.plan)
    assert first["from"] == "graph" and second["from"] == "compiled" and second["step"] == 40
    assert sim.plan.decision("fallback.mid_run")[1] == f"compiled -> {sim.plan.tier}"
    assert thermo_steps(tmp_path / "run") == list(range(0, 61, 5))
    sim.close()


# --------------------------------------------------------------------------- not triggers


@pytest.mark.parametrize("kind", ["memory", "untagged"])
def test_out_of_memory_and_untagged_errors_are_not_triggers(pristine, recorded_capture, monkeypatch, kind):
    recorded = graph.GraphEngine._capture
    state = {"armed": False}

    class OutOfMemoryError(RuntimeError):
        pass

    def capture(self, key):
        if state["armed"]:
            state["armed"] = False
            raise OutOfMemoryError("CUDA out of memory (injected)")
        return recorded(self, key)

    original = graph.GraphEngine.step

    def step(self):
        if kind == "untagged" and int(self.state.step) == 10:
            raise RuntimeError("something else (injected)")
        return original(self)

    monkeypatch.setattr(graph.GraphEngine, "_capture", capture)
    monkeypatch.setattr(graph.GraphEngine, "step", step)
    sim = simulation(pristine)
    state["armed"] = kind == "memory"
    with pytest.raises(RuntimeError, match="injected"):
        sim.run(steps=60)
    assert sim.plan.revision == 0 and not fallback.is_mid_run(sim.plan)
    sim.close()


def test_mid_run_stage_classifies_the_triggers():
    tagged = RuntimeError(CAPTURE_ERROR)
    tagged.fastmace_stage = graph.CAPTURE_STAGE
    assert fallback.mid_run_stage(tagged) == "capture"
    assert fallback.mid_run_stage(DynamoError("x")) == "compile"
    assert fallback.mid_run_stage(RuntimeError("x")) is None
    assert fallback.mid_run_stage(MemoryError()) is None
    from fastmace.errors import NonFiniteError, ProcessConstraintError

    assert fallback.mid_run_stage(NonFiniteError("nan")) is None
    assert fallback.mid_run_stage(ProcessConstraintError("compiled twice")) is None


# --------------------------------------------------------------------------- resume


def test_a_resume_rebuilds_the_fallback_revision(pristine, break_capture, tmp_path):
    from fastmace.simulation import Simulation

    sim = simulation(pristine, output=output(tmp_path, "run", restart_every=10))
    break_capture()
    sim.run(steps=60)
    plan = sim.plan
    assert fallback.is_mid_run(plan)
    sim.close()
    resumed = Simulation.resume(tmp_path / "run")
    assert resumed.plan.tier == plan.tier and not isinstance(resumed._engine, graph.GraphEngine)
    assert fallback.mid_run_fallbacks(resumed.plan) == fallback.mid_run_fallbacks(plan)
    resumed.run(steps=10)
    assert int(resumed._engine.state.step) == 70
    resumed.close()


# --------------------------------------------------------------------------- the records


def test_mid_run_fallbacks_reads_the_chain():
    import test_minimal_planner as tm

    plan = tm.plan_label("mh1-r2scan", workload=Workload(max_temperature_K=300.0))
    note = fallback.mid_run_note("graph", 40, 47, "capture", RuntimeError("x (y)"), setup=False)
    off = plan.revise(f"{fallback.PREFIX}rewrites block_linear off: {note}: ddmin", 40)
    moved = off.revise(f"{fallback.PREFIX}mode compiled: {note}: ddmin", 40, tier="compiled")
    assert fallback.mid_run_fallbacks(plan) == []
    assert fallback.mid_run_fallbacks(moved) == [{"step": 40, "from": "graph", "to": "compiled",
                                                 "reason": str(moved.reason), "revision": moved.revision}]
    assert fallback.mid_run_fallbacks(off)[0]["to"] == "graph"
    setup = plan.revise(f"{fallback.PREFIX}mode eager: {fallback.MODE_WHY}", 0, tier="eager")
    assert not fallback.is_mid_run(setup) and fallback.mid_run_fallbacks(setup) == []


def test_counters_fallbacks_are_checked():
    from fastmace import provenance

    assert provenance.FALLBACK_KEYS == ("step", "from", "to", "reason", "revision")
    assert provenance.metadata_record()["counters"]["fallbacks"] == []
    entry = {"step": 40, "from": "graph", "to": "compiled", "reason": "fallback: mode compiled: ...", "revision": 2}
    assert provenance.metadata_record(counters={"fallbacks": [entry]})["counters"]["fallbacks"] == [entry]
    with pytest.raises(ValueError, match="fallbacks"):
        provenance.metadata_record(counters={"fallbacks": [{"step": 1}]})
