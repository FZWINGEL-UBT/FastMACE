"""The runtime monitor (plan task 5-T06; SPEC §9.6, §13).

``planner.adapt.Monitor`` reads the engine's cumulative counters at each canonical point and at the end
of each ``run()`` call, and keeps the difference since the previous one as a ``Segment``: re-captures and
the host time they took, all captures, overflows redone, the re-capture budget's step, and per neighbour
list the rebuilds and build time (MACE: counts only, its build overlaps on a side stream; D3: its own
build time). It decides nothing yet (5-T07 adds the trigger). ``Simulation`` records the segments in
``counters.segments`` and the totals in ``counters.neighbour_lists``; the report adds a line only when
re-captures or overflows happened.
"""

from __future__ import annotations

import json

import pytest
import test_prepare_edges as edges_test

from fastmace import integrators, provenance
from fastmace.config import AdaptationConfig, EngineConfig
from fastmace.engine import graph
from fastmace.planner import adapt
from fastmace.planner.workload import Workload

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

quiet_licences = edges_test.quiet_licences
pristine = edges_test.pristine
recorded_capture = edges_test.recorded_capture


def info(*, rebuilds=0, recaptures=0, recapture_s=0.0, captures=0, capture_s=0.0, overflows=0, exceeded=None,
         d3=None):
    out = {"neighbour_rebuilds": rebuilds, "recaptures": recaptures, "recapture_seconds": recapture_s,
           "captures": captures, "capture_seconds": capture_s, "overflows_redone": overflows,
           "recapture_budget_exceeded_at": exceeded}
    if d3 is not None:
        out["d3"] = {"builds": d3[0] + 1, "rebuilds": d3[0], "regrowths": 1, "build_seconds": d3[1]}
    return out


def started(step=0, wall_s=0.0, **counters):
    monitor = adapt.Monitor(AdaptationConfig())
    monitor.start(info(**counters), step=step, wall_s=wall_s)
    return monitor


# --------------------------------------------------------------------------- segments


def test_a_segment_is_the_difference_since_the_previous_point():
    monitor = started(step=0, wall_s=1.0, rebuilds=1, captures=3, capture_s=0.6)
    segment = monitor.observe(info(rebuilds=5, recaptures=2, recapture_s=0.25, captures=5, capture_s=0.85,
                                   overflows=1), step=100, wall_s=11.0, reason="restart")
    assert segment == monitor.segments[-1]
    assert (segment.first_step, segment.last_step, segment.reason, segment.canonical) == (0, 100, "restart", True)
    assert segment.wall_s == pytest.approx(10.0)
    assert (segment.recaptures, segment.captures, segment.overflows_redone) == (2, 2, 1)
    assert segment.recapture_s == pytest.approx(0.25) and segment.capture_s == pytest.approx(0.25)
    assert segment.budget_exceeded_at is None
    assert segment.lists == {"mace": {"rebuilds": 4, "build_time_s": None}}
    second = monitor.observe(info(rebuilds=6, recaptures=9, recapture_s=1.25, captures=12, capture_s=1.85,
                                  overflows=1, exceeded=150), step=180, wall_s=15.0, reason="run-end",
                             canonical=False)
    assert (second.first_step, second.last_step, second.canonical) == (100, 180, False)
    assert (second.recaptures, second.overflows_redone, second.budget_exceeded_at) == (7, 0, 150)
    assert second.recapture_s == pytest.approx(1.0) and second.wall_s == pytest.approx(4.0)
    assert second.lists["mace"] == {"rebuilds": 1, "build_time_s": None}


def test_no_step_since_the_previous_point_closes_no_segment_and_keeps_the_counters():
    monitor = started(step=10)
    assert monitor.observe(info(captures=1, capture_s=0.5), step=10, wall_s=2.0, reason="host-write") is None
    assert monitor.segments == ()
    segment = monitor.observe(info(captures=1, capture_s=0.5, rebuilds=2), step=20, wall_s=3.0, reason="restart")
    assert (segment.first_step, segment.captures, segment.capture_s, segment.wall_s) == (10, 1, 0.5, 3.0)


def test_d3_keeps_its_build_time_and_missing_counters_read_as_zero():
    monitor = started(d3=(0, 0.01))
    segment = monitor.observe(info(rebuilds=3, d3=(4, 0.05)), step=50, wall_s=1.0, reason="mc-block")
    assert segment.lists["d3"]["rebuilds"] == 4
    assert segment.lists["d3"]["build_time_s"] == pytest.approx(0.04)
    reference = adapt.Monitor(AdaptationConfig())  # the reference tier's info() has none of the counters
    reference.start({"tier": "reference"}, step=0, wall_s=0.0)
    segment = reference.observe({"tier": "reference"}, step=5, wall_s=1.0, reason="run-end", canonical=False)
    assert (segment.recaptures, segment.captures, segment.overflows_redone) == (0, 0, 0)
    assert segment.recapture_s == 0.0 and segment.lists == {"mace": {"rebuilds": 0, "build_time_s": None}}


def test_an_unknown_reason_is_refused():
    monitor = started()
    with pytest.raises(ValueError, match="reason"):
        monitor.observe(info(), step=1, wall_s=1.0, reason="sometimes")
    assert set(adapt.SEGMENT_REASONS) == {"restart", "mc-block", "host-write", "run-end"}


def test_observe_before_start_takes_the_baseline_there():
    monitor = adapt.Monitor(AdaptationConfig())
    assert not monitor.started
    assert monitor.observe(info(captures=4), step=7, wall_s=2.0, reason="host-write") is None
    assert monitor.started and monitor.segments == ()
    segment = monitor.observe(info(captures=5), step=9, wall_s=3.0, reason="restart")
    assert (segment.first_step, segment.captures, segment.wall_s) == (7, 1, 1.0)


# --------------------------------------------------------------------------- windows and totals


def walk(monitor, points):
    for step, recaptures, wall_s in points:
        monitor.observe(info(recaptures=recaptures, recapture_s=0.1 * recaptures, captures=recaptures,
                             capture_s=0.1 * recaptures, rebuilds=step // 10), step=step, wall_s=wall_s,
                        reason="restart")


def test_window_sums_the_segments_since_a_step_or_the_mark():
    monitor = started()
    walk(monitor, [(100, 1, 1.0), (200, 3, 2.0), (300, 6, 3.0)])
    whole = monitor.window()
    assert (whole.first_step, whole.last_step, whole.recaptures, whole.reason) == (0, 300, 6, "restart")
    assert whole.wall_s == pytest.approx(3.0) and whole.recapture_s == pytest.approx(0.6)
    assert whole.lists["mace"]["rebuilds"] == 30
    since = monitor.window(since_step=100)
    assert (since.first_step, since.last_step, since.recaptures) == (100, 300, 5)
    monitor.mark(200)
    marked = monitor.window()
    assert (marked.first_step, marked.recaptures) == (200, 3)
    assert monitor.window(since_step=300) is None
    assert monitor.totals().recaptures == 6


def test_segments_are_bounded_and_the_totals_keep_counting(monkeypatch):
    monkeypatch.setattr(adapt, "SEGMENT_LIMIT", 2)
    monitor = started()
    walk(monitor, [(100, 1, 1.0), (200, 3, 2.0), (300, 6, 3.0)])
    assert [segment.last_step for segment in monitor.segments] == [200, 300]
    assert monitor.closed == 3
    totals = monitor.totals()
    assert (totals.first_step, totals.last_step, totals.recaptures) == (0, 300, 6)


def test_decide_returns_nothing_and_off_still_records():
    monitor = adapt.Monitor(AdaptationConfig(adapt="off"))
    monitor.start(info(), step=0, wall_s=0.0)
    walk(monitor, [(100, 50, 1.0)])
    assert monitor.decide(monitor.window()) is None
    assert monitor.totals().recaptures == 50


def test_export_and_absorb_carry_a_childs_segments_to_the_parent():
    child = started()
    walk(child, [(100, 1, 1.0)])
    cursor = child.cursor()
    walk(child, [(200, 3, 2.0), (300, 6, 3.0)])
    exported = json.loads(json.dumps(child.export(cursor)))  # the reply is JSON
    parent = adapt.Monitor(AdaptationConfig())
    parent.absorb(exported)
    assert [segment.last_step for segment in parent.segments] == [200, 300]
    assert parent.totals().recaptures == 5 and parent.closed == 2
    parent.absorb(None)  # a reply without a monitor (an older child) changes nothing
    assert parent.closed == 2


def test_absorb_counts_segments_the_child_dropped(monkeypatch):
    monkeypatch.setattr(adapt, "SEGMENT_LIMIT", 1)
    child = started()
    cursor = child.cursor()
    walk(child, [(100, 1, 1.0), (200, 3, 2.0)])
    parent = adapt.Monitor(AdaptationConfig())
    parent.absorb(json.loads(json.dumps(child.export(cursor))))
    assert len(parent.segments) == 1 and parent.closed == 2
    assert parent.totals().recaptures == 3 and parent.totals().first_step == 0


def test_a_segment_record_has_the_schema_keys_and_round_trips():
    monitor = started()
    segment = monitor.observe(info(recaptures=1, d3=(1, 0.5)), step=3, wall_s=1.0, reason="restart")
    record = segment.record()
    assert tuple(record) == provenance.SEGMENT_KEYS
    assert adapt.Segment.from_record(json.loads(json.dumps(record))) == segment


# --------------------------------------------------------------------------- provenance and report


def test_counters_segments_default_to_an_empty_list_and_are_checked():
    assert provenance.metadata_record()["counters"]["segments"] == []
    segment = started().observe(info(recaptures=2), step=4, wall_s=1.0, reason="restart")
    record = provenance.metadata_record(counters={"segments": [segment]})
    assert record["counters"]["segments"] == [segment.record()]
    with pytest.raises(ValueError, match="segments"):
        provenance.metadata_record(counters={"segments": [{"first_step": 0}]})
    assert dict(provenance.SPEC_13_KEYS)["runtime monitor segments (5-T06)"] == "counters.segments"


def test_neighbour_lists_carry_the_totals_and_the_skins():
    monitor = started(d3=(0, 0.0))
    monitor.observe(info(rebuilds=7, d3=(2, 0.3)), step=10, wall_s=1.0, reason="restart")
    lists = monitor.neighbour_lists(mace_skin_A=0.3, d3_skin_A=1.5)
    assert lists == {"mace": {"rebuilds": 7, "build_time_s": None, "skin_A": 0.3},
                     "d3": {"rebuilds": 2, "build_time_s": pytest.approx(0.3), "skin_A": 1.5}}
    assert adapt.Monitor(AdaptationConfig()).neighbour_lists(mace_skin_A=0.3) is None  # nothing observed


def test_report_line_only_with_recaptures_or_overflows():
    quiet = started()
    walk(quiet, [(100, 0, 10.0)])
    assert adapt.report_line(quiet.totals()) is None
    assert adapt.report_line(None) is None
    busy = started()
    busy.observe(info(recaptures=14, recapture_s=2.3, overflows=3, exceeded=1200), step=2000, wall_s=46.0,
                 reason="run-end", canonical=False)
    line = adapt.report_line(busy.totals(), budget=8)
    assert line == ("re-captures 14 (2.3 s, 5.0 % of the loop; budget 8 exceeded at step 1,200); "
                    "overflows redone 3")
    overflow = started()
    overflow.observe(info(overflows=2), step=10, wall_s=1.0, reason="run-end", canonical=False)
    assert adapt.report_line(overflow.totals()) == "overflows redone 2"


# --------------------------------------------------------------------------- the graph engine's capture time


def test_graph_engine_times_its_captures(pristine, recorded_capture, monkeypatch):
    clock = iter(range(0, 10_000, 2))  # every read of the capture clock 2 s later
    monkeypatch.setattr(graph, "_clock", lambda: float(next(clock)))
    sim = simulation(pristine)
    engine = sim._engine
    sim.run(steps=30)
    counters = engine.runtime_counters()
    assert counters["captures"] >= 1 and counters["captures"] == len(engine.recapture_log)
    assert counters["capture_seconds"] == pytest.approx(2.0 * counters["captures"])
    assert counters["recapture_seconds"] == pytest.approx(2.0 * counters["recaptures"])
    assert all(entry["s"] == pytest.approx(2.0) for entry in engine.recapture_log)
    for key in ("captures", "capture_seconds", "recapture_seconds"):
        assert engine.info()[key] == counters[key]
    sim.close()


# --------------------------------------------------------------------------- Simulation


def simulation(pristine, *, output=None, variant_cap=None):
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    atoms = edges_test.nacl(repeat=1, rattle=0.0, seed=3)
    atoms.set_cell(atoms.cell * 1.01, scale_atoms=True)  # the edge count grows within 50 steps (2-T21d)
    integrator = integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01)
    engine = {} if variant_cap is None else {"graph_variant_cap": variant_cap}
    return Simulation(atoms, ForceField(pristine, atoms, precision="float64"), integrator=integrator, seed=8,
                      workload=Workload(max_temperature_K=300.0), output=output,
                      engine=EngineConfig(device="cpu", compile_mode="none", **engine))


def test_simulation_closes_segments_at_restarts_and_run_ends(pristine, recorded_capture, tmp_path):
    from fastmace.io.output import Output

    sim = simulation(pristine, output=Output(tmp_path / "run", trajectory_every=None, restart_every=20,
                                             thermo_every=10))
    sim.run(steps=50)
    sim.run(steps=5)
    segments = sim._monitor.segments
    assert [(s.first_step, s.last_step, s.reason, s.canonical) for s in segments] == [
        (0, 20, "restart", True), (20, 40, "restart", True), (40, 50, "run-end", False),
        (50, 55, "run-end", False)]
    assert all(segment.wall_s >= 0.0 for segment in segments)
    assert sum(s.captures for s in segments) <= sim._engine.runtime_counters()["captures"]
    sim.close()
    record = json.loads((tmp_path / "run" / "metadata.json").read_text())
    assert [entry["last_step"] for entry in record["counters"]["segments"]] == [20, 40, 50, 55]
    lists = record["counters"]["neighbour_lists"]
    assert lists["mace"]["rebuilds"] == sum(s.lists["mace"]["rebuilds"] for s in segments)
    assert lists["mace"]["build_time_s"] is None and lists["mace"]["skin_A"] == sim.plan.neighbours.skin


def test_a_recapture_storm_moves_the_counters_and_adds_the_report_line(pristine, recorded_capture):
    sim = simulation(pristine, variant_cap=1)  # every new edge size evicts the one graph there is
    sim.run(steps=50)
    totals = sim._monitor.totals()
    assert totals.recaptures > 0 and totals.recaptures == sim._engine.recaptures
    assert "re-captures" in sim.report.text()
    sim.close()


def test_a_quiet_run_has_no_report_line(pristine):
    sim = simulation(pristine)
    sim.run(steps=5)
    assert sim._monitor.totals().recaptures == 0
    assert "re-captures" not in sim.report.text() and "overflows redone" not in sim.report.text()
    sim.close()


def test_observe_calls_decide_with_the_window(pristine, monkeypatch):
    seen = []
    monkeypatch.setattr(adapt.Monitor, "decide", lambda self, window: seen.append(window))
    sim = simulation(pristine)
    sim.run(steps=5)
    assert len(seen) == 1 and seen[0].last_step == 5
    sim.close()


def test_a_resume_starts_an_empty_monitor_with_the_recorded_adaptation(pristine, tmp_path):
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    atoms = edges_test.nacl(repeat=1, rattle=0.0, seed=3)
    integrator = integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01)
    sim = Simulation(atoms, ForceField(pristine, atoms, precision="float64"), integrator=integrator, seed=8,
                     workload=Workload(max_temperature_K=300.0), adaptation=AdaptationConfig(adapt="off"),
                     engine=EngineConfig(device="cpu", compile_mode="none"))
    sim.run(steps=4)
    sim.save_restart(tmp_path / "restart")
    assert len(sim._monitor.segments) == 1
    sim.close()
    resumed = Simulation.resume(tmp_path / "restart")
    assert resumed._monitor.adaptation == AdaptationConfig(adapt="off") and resumed._monitor.segments == ()
    resumed.run(steps=3)
    assert [(s.first_step, s.last_step) for s in resumed._monitor.segments] == [(4, 7)]
    resumed.close()
