"""The tier switch between segments (plan task 5-T07; SPEC §9.6, §18 decisions 52 and 53).

- ``AdaptationConfig``'s defaults are decisions 52 and 53; the options the owner did not choose are refused
  (``adaptation-unsupported``).
- ``Monitor.decide`` is the trigger: the re-capture budget exceeded and more than 20 % of the window's wall
  time spent re-capturing. ``adapt.blocked`` holds the limits: one way, once per run (a resume counts its
  chain), never under ``reproducible=True``, with an explicit ``mode=``, in an MC block or for a relaxation.
- On the CPU, a forced re-capture storm (one graph variant, every capture timed at 2 s) switches the tiny
  NaCl run graph -> compiled at a committed step: a plan revision with the trigger in its reason,
  ``counters.tier_switches``, the report's revision line, the state carried over. A resume after the switch
  runs the compiled revision and does not switch again, and continues bit for bit as the uninterrupted run.
- A candidate that fails its validation leaves the run on the graph revision, with a warning, and no
  second attempt.
- A subprocess runner's child returns the request instead of switching; the parent hands it to the next
  child (``SubprocessRunner.run``).
"""

from __future__ import annotations

import dataclasses
import itertools
import json

import numpy as np
import pytest
import test_adapt_monitor as monitor_test
import test_minimal_planner as tm

from fastmace import config, integrators
from fastmace.config import AdaptationConfig, EngineConfig
from fastmace.engine import graph
from fastmace.engine.compiled import CompiledEngine
from fastmace.errors import PlanRefused
from fastmace.planner import adapt
from fastmace.planner.workload import Workload

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

quiet_licences = monitor_test.quiet_licences
pristine = monitor_test.pristine
recorded_capture = monitor_test.recorded_capture


def window(*, recaptures=5, recapture_s=3.0, wall_s=10.0, exceeded=40, first=0, last=100):
    return adapt.Segment(first, last, "recapture", False, wall_s, recaptures=recaptures, recapture_s=recapture_s,
                         budget_exceeded_at=exceeded)


# --------------------------------------------------------------------------- config (decisions 52, 53)


def test_the_defaults_are_decisions_52_and_53():
    adaptation = AdaptationConfig()
    assert (adaptation.adapt, adaptation.trigger, adaptation.recapture_time_fraction) == ("auto", "recapture-time",
                                                                                          0.2)
    assert (adaptation.skin_replan, adaptation.max_switches, adaptation.switch_back_after,
            adaptation.min_points_between, adaptation.reproducible) == (False, 1, None, None, False)
    assert adaptation.unsupported() == ()
    assert AdaptationConfig(trigger="none").unsupported() == () == AdaptationConfig(max_switches=0).unsupported()


@pytest.mark.parametrize("fields", [{"trigger": "recapture-budget"}, {"skin_replan": True}, {"max_switches": 2},
                                    {"max_switches": None}, {"switch_back_after": 3}, {"min_points_between": 2},
                                    {"reproducible": True}])
def test_options_not_chosen_are_refused(fields):
    from fastmace.simulation import PHASE_5_PENDING_CODES, planner_options

    assert AdaptationConfig(**fields).unsupported() == tuple(fields)
    with pytest.raises(PlanRefused) as refused:
        planner_options(adaptation=AdaptationConfig(**fields))
    assert refused.value.codes == ("adaptation-unsupported",) == (PHASE_5_PENDING_CODES["adaptation"],)
    assert "decisions 52 and 53" in str(refused.value)
    assert "or pass adapt='off'" not in str(refused.value)
    with pytest.raises(PlanRefused):  # adapt="off" does not lift it, and the message says so
        planner_options(adaptation=AdaptationConfig(adapt="off", **fields))


def test_pending_is_no_longer_an_adaptation_value():
    with pytest.raises(ValueError):
        AdaptationConfig(trigger=config.PENDING)
    with pytest.raises(ValueError):
        AdaptationConfig(max_switches=config.PENDING)


# --------------------------------------------------------------------------- the trigger (decision 52)


def test_the_trigger_needs_the_budget_exceeded_and_more_than_a_fifth_of_the_wall_time():
    monitor = adapt.Monitor(AdaptationConfig())
    request = monitor.decide(window(recapture_s=2.5, wall_s=10.0))
    assert request == adapt.SwitchRequest(step=100, recaptures=5, recapture_s=2.5, wall_s=10.0,
                                          budget_exceeded_at=40, threshold=0.2)
    assert "25.0 % of the wall time" in request.text() and "decision 52" in request.text()
    assert adapt.SwitchRequest.from_record(request.record()) == request
    assert monitor.decide(window(recapture_s=2.0, wall_s=10.0)) is None  # 20 %: not above
    assert monitor.decide(window(exceeded=None)) is None  # within the budget
    assert monitor.decide(window(wall_s=0.0)) is None
    assert monitor.decide(None) is None


def test_off_and_no_trigger_never_request():
    for adaptation in (AdaptationConfig(adapt="off"), AdaptationConfig(trigger="none")):
        assert adapt.Monitor(adaptation).decide(window(recapture_s=9.0)) is None


# --------------------------------------------------------------------------- the limits (decision 53)


def graph_plan(**workload):
    return tm.plan_label("mh1-r2scan", workload=Workload(max_temperature_K=300.0, **workload))


def test_a_graph_plan_may_switch_once():
    plan = graph_plan()
    assert plan.tier == "graph" and adapt.blocked(plan, AdaptationConfig(), mode=None) is None
    switched = plan.revise(f"{adapt.PREFIX}graph -> compiled: storm", 120, tier="compiled")
    assert adapt.switches(switched) == 1 and adapt.is_switch(switched)
    assert "one way" in adapt.blocked(switched, AdaptationConfig(), mode=None)
    again = switched.revise(f"{adapt.PREFIX}graph -> compiled: again", 200, tier="graph")  # a chain with one
    assert "at most 1 tier switch per run" in adapt.blocked(again, AdaptationConfig(), mode=None)
    assert "at most 0" in adapt.blocked(plan, AdaptationConfig(max_switches=0), mode=None)
    (switch,) = adapt.tier_switches(switched)
    assert switch.record() == {"step": 120, "from": "graph", "to": "compiled", "reason": str(switched.reason),
                               "revision": 1}


@pytest.mark.parametrize("case", ["off", "none", "mode", "mc-block", "reproducible"])
def test_what_blocks_a_switch(case):
    plan = graph_plan()
    adaptation, mode, in_block = AdaptationConfig(), None, False
    if case == "off":
        adaptation = AdaptationConfig(adapt="off")
    elif case == "none":
        adaptation = AdaptationConfig(trigger="none")
    elif case == "mode":
        mode = "graph"
    elif case == "mc-block":
        in_block = True
    else:
        plan = dataclasses.replace(plan, workload=dataclasses.replace(plan.workload, reproducible=True))
    assert adapt.blocked(plan, adaptation, mode=mode, in_mc_block=in_block) is not None


# --------------------------------------------------------------------------- the switch on the CPU


@pytest.fixture
def slow_captures(monkeypatch):
    """Every read of the capture clock 2 s after the last: each (re-)capture takes 2 s of host time."""
    clock = itertools.count(0, 2)
    monkeypatch.setattr(graph, "_clock", lambda: float(next(clock)))


def storm(pristine, **kwargs):
    """The monitor tests' tiny NaCl run with one graph variant: every new edge size re-captures (budget 1)."""
    return simulation(pristine, variant_cap=1, **kwargs)


def simulation(pristine, *, output=None, variant_cap=None, **options):
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    atoms = monitor_test.edges_test.nacl(repeat=1, rattle=0.0, seed=3)
    atoms.set_cell(atoms.cell * 1.01, scale_atoms=True)  # the edge count grows within 50 steps (2-T21d)
    integrator = integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01)
    engine = {} if variant_cap is None else {"graph_variant_cap": variant_cap}
    return Simulation(atoms, ForceField(pristine, atoms, precision="float64"), integrator=integrator, seed=8,
                      workload=Workload(max_temperature_K=300.0), output=output,
                      engine=EngineConfig(device="cpu", compile_mode="none", **engine), **options)


def test_a_recapture_storm_switches_to_the_compiled_tier(pristine, recorded_capture, slow_captures, tmp_path):
    from fastmace.io.output import Output

    sim = storm(pristine, output=Output(tmp_path / "run", trajectory_every=None, restart_every=None,
                                        thermo_every=10))
    graph_plan = sim.plan
    sim.run(steps=60)
    plan = sim.plan
    assert plan.tier == "compiled" and plan.revision == graph_plan.revision + 1 and adapt.is_switch(plan)
    assert plan.predecessor.tier == "graph" and 0 < plan.step < 60
    assert isinstance(sim._engine, CompiledEngine) and int(sim._engine.state.step) == 60
    assert plan.engine.backend == graph_plan.engine.backend and plan.rewrites == graph_plan.rewrites
    assert plan.precision == graph_plan.precision and plan.integrator == graph_plan.integrator
    choice, reason = plan.decision("adaptation")[1:]
    assert choice == "graph -> compiled" and reason.code == adapt.SWITCH_CODE and reason.rule is None
    assert plan.decision("mode")[2].code == adapt.SWITCH_CODE and ("mode", "compiled") not in plan.requested
    assert "% of the wall time" in str(plan.reason)
    assert sim._monitor._mark == plan.step  # the window restarts at the switch
    assert f"plan revision {plan.revision} (tier switch at step {plan.step:,}): graph -> compiled" in sim.report.text()
    sim.close()
    record = json.loads((tmp_path / "run" / "metadata.json").read_text())
    (switch,) = record["counters"]["tier_switches"]
    assert (switch["step"], switch["from"], switch["to"], switch["revision"]) == (plan.step, "graph", "compiled",
                                                                                   plan.revision)
    assert record["plan"]["tier"] == "compiled"


def test_the_switched_run_keeps_its_physics(pristine, recorded_capture, slow_captures, tmp_path):
    """The state crosses the switch unchanged: against the unswitched run with a canonical point (a restart
    write) at the switch step, the trajectory agrees to float64 rounding (on the CPU without compilation the
    compiled tier steps as the graph tier does, 3-T06)."""
    switched = storm(pristine)
    switched.run(steps=60)
    step = switched.plan.step
    assert switched.plan.tier == "compiled"
    reference = storm(pristine, adaptation=AdaptationConfig(adapt="off"))
    reference.run(steps=step)
    reference.save_restart(tmp_path / "point")  # the same canonical point (§10.6)
    reference.run(steps=60 - step)
    assert reference.plan.tier == "graph"
    first, second = switched.committed(), reference.committed()
    assert first.step == second.step == 60
    np.testing.assert_allclose(first.positions, second.positions, rtol=0, atol=1e-9)
    np.testing.assert_allclose(first.momenta, second.momenta, rtol=0, atol=1e-9)
    assert first.energy == pytest.approx(second.energy, abs=1e-9)
    switched.close()
    reference.close()


def test_no_switch_below_the_threshold_or_when_off(pristine, recorded_capture, monkeypatch):
    monkeypatch.setattr(graph, "_clock", lambda: 0.0)  # captures take no time: never 20 % of the wall time
    sim = storm(pristine)
    sim.run(steps=60)
    assert sim.plan.tier == "graph" and sim._engine.recaptures > 1 and sim.plan.revision == 0
    sim.close()


@pytest.mark.parametrize("options", [{"adaptation": AdaptationConfig(adapt="off")}, {"mode": "graph"}],
                         ids=["off", "explicit-mode"])
def test_no_switch_when_blocked(pristine, recorded_capture, slow_captures, options):
    sim = storm(pristine, **options)
    sim.run(steps=60)
    assert sim.plan.tier == "graph" and sim._engine.recaptures > 1
    assert "re-captures" in sim.report.text()  # 3-T04's warning and 5-T06's line stay
    sim.close()


def test_a_quiet_run_never_reads_the_monitor_in_the_loop(pristine, recorded_capture, slow_captures):
    sim = simulation(pristine)  # the default variant cap: no re-capture past the budget
    sim.run(steps=40)
    assert sim.plan.tier == "graph"
    assert all(segment.reason != "recapture" for segment in sim._monitor.segments)
    sim.close()


def test_a_failed_candidate_stays_on_the_graph_revision(pristine, recorded_capture, slow_captures, monkeypatch):
    from fastmace.errors import GateFailed
    from fastmace.simulation import Simulation

    calls = []

    def fail(self, engine, plan):
        calls.append(plan.tier)
        raise GateFailed("forced")

    monkeypatch.setattr(Simulation, "_validate", fail)
    sim = storm(pristine)
    sim.run(steps=60)
    assert calls == ["compiled"]  # one attempt per run
    assert sim.plan.tier == "graph" and sim.plan.revision == 0 and not isinstance(sim._engine, CompiledEngine)
    assert int(sim._engine.state.step) == 60
    assert any(adapt.SWITCH_FAILED_CODE in line for line in sim.report.warnings())
    sim.close()


def test_a_resume_after_the_switch_runs_compiled_and_continues_bitwise(pristine, recorded_capture, slow_captures,
                                                                       tmp_path):
    """Interrupted after the switch against uninterrupted: bit for bit (conductor's answer, 5-T07 Q5)."""
    from fastmace.io.output import Output
    from fastmace.simulation import Simulation

    def output(name):
        return Output(tmp_path / name, trajectory_every=None, restart_every=40, thermo_every=10)

    whole = storm(pristine, output=output("whole"))
    whole.run(steps=40)
    switched_at = whole.plan.step
    assert whole.plan.tier == "compiled" and switched_at < 40
    whole.run(steps=30)
    expected = whole.committed()
    whole.close()

    first = storm(pristine, output=output("first"))
    first.run(steps=40)
    assert first.plan.tier == "compiled" and first.plan.step == switched_at
    first.close()
    resumed = Simulation.resume(tmp_path / "first")
    assert resumed.plan.tier == "compiled" and adapt.switches(resumed.plan) == 1
    assert isinstance(resumed._engine, CompiledEngine)
    resumed.run(steps=30)
    assert resumed.plan.revision == whole.plan.revision  # no second switch
    got = resumed.committed()
    assert got.step == expected.step == 70
    np.testing.assert_array_equal(got.positions, expected.positions)
    np.testing.assert_array_equal(got.momenta, expected.momenta)
    (switch,) = adapt.tier_switches(resumed.plan)
    assert switch.step == switched_at
    resumed.close()


def test_a_resume_before_the_switch_switches_on_its_own_window(pristine, recorded_capture, monkeypatch, tmp_path):
    """Interrupted before the switch: the resumed run's monitor starts empty (format 4), so it switches at a step
    of its own, once; the two runs agree statistically only (conductor's answer, 5-T07 Q5)."""
    from fastmace.io.output import Output
    from fastmace.simulation import Simulation

    monkeypatch.setattr(graph, "_clock", lambda: 0.0)  # no re-capture time: the first run never switches
    first = storm(pristine, output=Output(tmp_path / "run", trajectory_every=None, restart_every=20,
                                          thermo_every=10))
    first.run(steps=20)
    assert first.plan.tier == "graph"
    first.close()
    clock = itertools.count(0, 2)
    monkeypatch.setattr(graph, "_clock", lambda: float(next(clock)))
    resumed = Simulation.resume(tmp_path / "run")
    resumed.run(steps=40)
    assert resumed.plan.tier == "compiled" and 20 < resumed.plan.step <= 60 and adapt.switches(resumed.plan) == 1
    assert int(resumed._engine.state.step) == 60
    resumed.close()


def test_the_setup_and_each_restart_write_keep_the_last_point(pristine, tmp_path):
    from fastmace.io.output import Output

    sim = simulation(pristine, output=Output(tmp_path / "run", trajectory_every=None, restart_every=10,
                                             thermo_every=10))
    point = sim._last_point
    assert point.step == 0 and point.outputs is None and int(point.restart()["step"]) == 0
    sim.run(steps=25)
    point = sim._last_point
    assert point.step == 20 and point.outputs is not None
    stored = point.restart()
    with np.load(tmp_path / "run" / "restart.npz", allow_pickle=False) as written:
        assert sorted(written.files) == sorted(stored)
        np.testing.assert_array_equal(written["positions"], stored["positions"])
    sim.close()


# --------------------------------------------------------------------------- the subprocess runner


def test_a_child_returns_the_request_instead_of_switching(pristine, recorded_capture, slow_captures):
    sim = storm(pristine)
    sim._child = True
    sim.run(steps=60)
    assert sim.plan.tier == "graph" and sim._switch_request is not None
    assert int(sim._engine.state.step) < 60  # returned early, at a committed step
    step = int(sim._engine.state.step)
    sim._child = False
    sim.switch_tier(sim._switch_request)
    assert sim.plan.tier == "compiled" and sim.plan.step == step
    sim.close()


def test_the_parent_hands_the_request_to_the_next_child(monkeypatch):
    from fastmace.config import RunnerConfig
    from fastmace.engine.subprocess import SubprocessRunner

    runner = SubprocessRunner(RunnerConfig())
    calls, restarts = [], []
    request = adapt.SwitchRequest(step=30, recaptures=4, recapture_s=3.0, wall_s=5.0, budget_exceeded_at=12,
                                  threshold=0.2).record()

    def run_child(sim, *, steps, schedule, check, gate, mc=...):
        calls.append((steps, runner._switch))
        runner._switch = None  # the request goes out with this call
        if len(calls) == 1:
            runner._step += 30
            return {"switch": request}
        runner._step += steps
        return {"switch": None}

    class Sim:
        _queue = type("Q", (), {"drain": staticmethod(lambda: None)})()

    monkeypatch.setattr(runner, "_run_child", run_child)
    monkeypatch.setattr(runner, "restart_worker", lambda: restarts.append(True))
    runner.run(Sim(), steps=100)
    assert calls == [(100, None), (70, request)] and restarts == [True] and runner._step == 100
