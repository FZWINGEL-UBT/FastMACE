"""Schedules and protocols (plan task 1b-T07; SPEC §7.2 "Schedules and step counts", §7.6, §10.5)."""

from __future__ import annotations

import json
import math
import subprocess
import sys

import pytest

import fastmace.protocols as protocols
from fastmace.protocols import (
    PROFILES,
    Profile,
    Schedule,
    ScheduleState,
    TargetFill,
    hold,
    load_profile,
    output_steps,
    profile_from_data,
    ramp,
)


# The research engine's schedule (langevin/workflow.py:ramp_steps, target_temperature, schedule_phase and
# output_steps at tag research-engine-baseline-20261017), copied as the expected values.
def research_ramp_steps(duration_ps, timestep_fs):
    steps = duration_ps * 1000 / timestep_fs
    if not math.isfinite(steps) or steps < 1 or not math.isclose(steps, round(steps), abs_tol=1e-8):
        raise ValueError("Each ramp must contain a positive integer number of timesteps")
    return round(steps)


def research_target(step, heat_steps, cool_steps, peak_K, final_K=0.0, hold_steps=0):
    if step <= heat_steps:
        return peak_K * step / heat_steps
    if step <= heat_steps + hold_steps:
        return peak_K
    return final_K + (peak_K - final_K) * (heat_steps + hold_steps + cool_steps - step) / cool_steps


def research_phase(step, heat_steps, hold_steps=0):
    if step <= heat_steps:
        return "heating"
    return "hold" if step <= heat_steps + hold_steps else "cooling"


# workflow.PROFILES: (heat_ps, cool_ps, peak_K, final_K, sample_interval, relax_fmax).
RESEARCH_PROFILES = {
    "slow": (30.0, 240.0, 400.0, 0.0, 100, 0.0),
    "fast": (3.0, 24.0, 400.0, 0.0, 100, 0.0),
    "ultrafast": (1.0, 8.0, 400.0, 0.0, 100, 0.0),
    "anneal": (20.0, 250.0, 1000.0, 250.0, 20, 0.01),
}


# --- segments and concatenation ----------------------------------------------------------------------

def test_plus_concatenates_and_refuses_a_temperature_jump():
    schedule = ramp(0, 1000, ps=20) + ramp(1000, 250, rate_K_per_ps=-3)
    assert isinstance(schedule, Schedule) and len(schedule.segments) == 2
    assert schedule.duration_ps == pytest.approx(270.0)
    with pytest.raises(ValueError, match="1000.*900|jump"):
        ramp(0, 1000, ps=20) + ramp(900, 250, ps=10)
    with pytest.raises(ValueError):
        hold(300, ps=1) + ramp(301, 400, ps=1)
    three = ramp(0, 400, ps=1) + hold(400, ps=1) + ramp(400, 0, ps=2)
    assert [s.kind for s in three.segments] == ["ramp", "hold", "ramp"]


@pytest.mark.parametrize("make", [
    lambda: ramp(0, 100),                                   # neither duration nor rate
    lambda: ramp(0, 100, ps=1, rate_K_per_ps=100),          # both
    lambda: ramp(0, 100, rate_K_per_ps=-3),                 # wrong sign
    lambda: ramp(100, 0, rate_K_per_ps=3),
    lambda: ramp(0, 100, rate_K_per_ps=0),
    lambda: ramp(100, 100, rate_K_per_ps=1),                # no change: a rate gives no duration
    lambda: ramp(0, 100, ps=0),
    lambda: ramp(0, 100, ps=-1),
    lambda: ramp(-1, 100, ps=1),                            # negative temperature
    lambda: ramp(0, math.inf, ps=1),
    lambda: ramp(0, 100, ps=math.nan),
    lambda: hold(300, ps=0),
    lambda: hold(-5, ps=1),
])
def test_invalid_segments_raise(make):
    with pytest.raises(ValueError):
        make()


def test_plus_only_takes_schedules():
    with pytest.raises(TypeError):
        ramp(0, 10, ps=1) + 5


# --- rounding, boundaries and totals -----------------------------------------------------------------

def test_anneal_at_half_a_femtosecond_is_540000_steps():
    """SPEC §7.2: ramp(0, 1000, ps=20) + ramp(1000, 250, rate_K_per_ps=-3) is 270 ps, 540,000 steps."""
    resolved = (ramp(0, 1000, ps=20) + ramp(1000, 250, rate_K_per_ps=-3)).resolve(0.5)
    assert resolved.segment_steps == (40_000, 500_000)
    assert resolved.boundaries == (0, 40_000, 540_000)
    assert resolved.total_steps == 540_000
    assert PROFILES["anneal"].schedule.resolve(0.5).total_steps == 540_000


def test_rounding_to_whole_steps_half_up_for_odd_timesteps():
    """Durations round half up to whole steps; the rounded durations and rates are recorded."""
    # 1 ps at 0.3 fs = 3333.33 steps -> 3333; 0.25 ps = 833.33 -> 833; 0.0002 ps = 0.667 -> 1.
    resolved = (ramp(0, 300, ps=1) + hold(300, ps=0.25) + ramp(300, 0, ps=0.0002)).resolve(0.3)
    assert resolved.segment_steps == (3333, 833, 1)  # 833.33 -> 833, 0.667 -> 1
    assert resolved.boundaries == (0, 3333, 4166, 4167)
    assert resolved.total_steps == 4167
    assert resolved.durations_ps == pytest.approx((3333 * 0.3e-3, 833 * 0.3e-3, 0.3e-3), rel=1e-15)
    assert resolved.rates_K_per_ps[0] == pytest.approx(300 / (3333 * 0.3e-3), rel=1e-15)
    assert resolved.rates_K_per_ps[1] == 0.0
    half = ramp(0, 10, ps=0.001).resolve(0.4)  # 2.5 steps: half up, not to even
    assert half.segment_steps == (3,)
    record = resolved.record()
    assert record["timestep_fs"] == 0.3 and record["total_steps"] == 4167
    assert record["boundaries"] == [0, 3333, 4166, 4167]
    assert [s["steps"] for s in record["segments"]] == [3333, 833, 1]
    assert record["segments"][0]["requested_ps"] == 1 and record["segments"][0]["ps"] == pytest.approx(0.9999)
    json.dumps(record)


def test_a_segment_that_rounds_to_zero_steps_is_refused():
    with pytest.raises(ValueError, match="0 steps|zero steps"):
        ramp(0, 10, ps=0.0001).resolve(0.5)  # 0.2 steps
    with pytest.raises(ValueError):
        ramp(0, 10, ps=1).resolve(0)
    with pytest.raises(ValueError):
        ramp(0, 10, ps=1).resolve(-0.5)


# --- the target per step ----------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(RESEARCH_PROFILES))
def test_profile_targets_equal_the_research_engine_step_for_step(name):
    """Bitwise, every step, ultrafast and the others (sparsely for the long ones)."""
    heat_ps, cool_ps, peak, final, _, _ = RESEARCH_PROFILES[name]
    timestep = 0.5
    heat, cool = research_ramp_steps(heat_ps, timestep), research_ramp_steps(cool_ps, timestep)
    resolved = PROFILES[name].schedule.resolve(timestep)
    assert resolved.boundaries == (0, heat, heat + cool)
    stride = 1 if name == "ultrafast" else 97
    steps = sorted(set(range(0, heat + cool + 1, stride)) | {0, 1, heat - 1, heat, heat + 1, heat + cool})
    for step in steps:
        assert resolved.target(step) == research_target(step, heat, cool, peak, final), step
        assert resolved.phase(step) == research_phase(step, heat), step


def test_ultrafast_targets_list_equals_the_research_engine():
    resolved = PROFILES["ultrafast"].schedule.resolve(0.5)
    expected = [research_target(s, 2000, 16000, 400.0) for s in range(1, 18001)]
    assert resolved.targets(1, 18000) == expected


def test_hold_targets_equal_the_research_engine_with_hold_steps():
    """--hold-ps: heat, hold at the peak, cool (workflow.target_temperature with hold_steps)."""
    resolved = (ramp(0, 400, ps=1) + hold(400, ps=0.5) + ramp(400, 0, ps=8)).resolve(0.5)
    assert resolved.boundaries == (0, 2000, 3000, 19000)
    for step in range(0, 19001):
        assert resolved.target(step) == research_target(step, 2000, 16000, 400.0, 0.0, 1000), step
        assert resolved.phase(step) == research_phase(step, 2000, 1000), step


def test_target_endpoints_and_out_of_range():
    resolved = (ramp(100, 300, ps=1) + ramp(300, 50, ps=1)).resolve(1.0)
    assert resolved.target(0) == 100.0
    assert resolved.target(1000) == 300.0
    assert resolved.target(2000) == 50.0
    assert resolved.target(500) == 200.0
    with pytest.raises(ValueError):
        resolved.target(-1)
    with pytest.raises(ValueError):
        resolved.target(2001)


def test_max_and_min_over_a_step_range():
    """The extremes of the per-step targets over first..last inclusive (equal to a brute-force scan)."""
    resolved = (ramp(0, 1000, ps=2) + hold(1000, ps=1) + ramp(1000, 250, ps=3) + ramp(250, 600, ps=1)).resolve(0.7)
    total = resolved.total_steps
    for first, last in [(0, total), (1, 1), (1, 2857), (2000, 5000), (3000, total), (7000, 7400), (5, 4)]:
        if last < first:
            with pytest.raises(ValueError):
                resolved.max_target(first, last)
            continue
        values = [resolved.target(s) for s in range(first, last + 1)]
        assert resolved.max_target(first, last) == max(values)
        assert resolved.min_target(first, last) == min(values)


# --- schedule position: steps, restart data, the fill ------------------------------------------------

def test_steps_to_run_follows_spec_7_2():
    state = ScheduleState(ramp(0, 300, ps=1) + ramp(300, 0, ps=1), timestep_fs=0.5)
    assert state.step == 0 and state.remaining == 4000 and state.target_K == 0.0
    assert state.steps_to_run(None) == 4000
    assert state.steps_to_run(1500) == 1500
    with pytest.raises(ValueError, match="remain"):
        state.steps_to_run(4001)
    for bad in (0, -1, 1.5, True):
        with pytest.raises((ValueError, TypeError)):
            state.steps_to_run(bad)
    state.advance(1500)
    assert state.step == 1500 and state.remaining == 2500
    assert state.target_K == state.schedule.resolve(0.5).target(1500)
    assert state.steps_to_run(None) == 2500
    with pytest.raises(ValueError):
        state.advance(2501)
    state.advance(2500)
    assert state.finished
    with pytest.raises(ValueError, match="finished|new schedule"):
        state.steps_to_run(None)


def test_extremes_of_the_next_steps():
    state = ScheduleState(ramp(0, 300, ps=1) + ramp(300, 0, ps=1), timestep_fs=0.5)
    state.advance(1000)
    assert state.max_target(1000) == 300.0  # steps 1001..2000
    assert state.min_target(1000) == state.resolved.target(1001)
    assert state.min_target(3000) == 0.0
    with pytest.raises(ValueError):
        state.max_target(3001)


def test_schedule_position_round_trips_through_json():
    schedule = ramp(0, 1000, ps=20) + hold(1000, ps=0.3) + ramp(1000, 250, rate_K_per_ps=-3)
    state = ScheduleState(schedule, timestep_fs=0.5)
    state.advance(123_457)
    data = state.to_restart()
    assert set(data) >= {"segments", "timestep_fs", "step", "target_K"}
    text = json.dumps(data)
    back = ScheduleState.from_restart(json.loads(text))
    assert back.schedule == schedule
    assert (back.step, back.timestep_fs, back.target_K) == (123_457, 0.5, state.target_K)
    assert back.resolved == state.resolved
    assert back.to_restart() == data
    assert json.dumps(back.to_restart()) == text


def test_from_restart_refuses_an_inconsistent_record():
    data = ScheduleState(ramp(0, 300, ps=1), timestep_fs=0.5).to_restart()
    with pytest.raises(ValueError):
        ScheduleState.from_restart({**data, "step": 2001})
    with pytest.raises(ValueError):
        ScheduleState.from_restart({**data, "target_K": 1.0, "step": 3})
    with pytest.raises(ValueError):
        ScheduleState.from_restart({**data, "segments": [{**data["segments"][0], "steps": 7}]})


def test_restart_record_is_canonical_json():
    """io.restart stores canonical_json(state.protocol.to_restart()) and reads it back (1b-T05)."""
    pytest.importorskip("torch")
    from fastmace.io import restart
    from fastmace.state import canonical_json
    state = ScheduleState(ramp(0, 5, ps=1) + hold(5, ps=1), 1.0)
    state.advance(10)
    back = ScheduleState.from_restart(restart.protocol_record({"protocol": canonical_json(state.to_restart())}))
    assert back.to_restart() == state.to_restart()


class RecordingIntegrator:
    def __init__(self):
        self.calls = []

    def set_temperature(self, temperature_K):
        self.calls.append(temperature_K)


def test_target_fill_fills_only_where_the_target_changes():
    """Per step on ramps, never again on a hold; the first call always fills (the device target is derived)."""
    integrator = RecordingIntegrator()
    fill = TargetFill(integrator)
    resolved = (ramp(0, 2, ps=0.002) + hold(2, ps=0.003) + ramp(2, 1, ps=0.001)).resolve(1.0)
    for step in range(1, resolved.total_steps + 1):
        fill(resolved.target(step))
    assert integrator.calls == [1.0, 2.0, 1.0]
    fill(1.0)
    assert len(integrator.calls) == 3
    fill.reset()
    fill(1.0)
    assert integrator.calls[-1] == 1.0 and len(integrator.calls) == 4


def test_schedule_state_fill_for_the_next_step():
    integrator = RecordingIntegrator()
    state = ScheduleState(ramp(0, 4, ps=0.004) + hold(4, ps=0.002), timestep_fs=1.0)
    fill = TargetFill(integrator)
    for _ in range(state.remaining):
        fill(state.next_target())
        state.advance(1)
    assert integrator.calls == [1.0, 2.0, 3.0, 4.0]


# --- output steps ------------------------------------------------------------------------------------

def test_output_steps_equals_1a():
    """The 1a function (io/trajectory.output_steps, workflow.output_steps), same signature and values."""
    assert output_steps(20, 2, 10) == list(range(0, 21, 2))
    assert output_steps(20, 4, 10) == [0, 4, 8, 10, 12, 16, 20]
    assert output_steps(30, 4, 10, hold_steps=5) == [0, 4, 8, 10, 12, 15, 16, 20, 24, 28, 30]
    from fastmace.io import trajectory
    for args in [(20, 2, 10), (20, 4, 10), (30, 4, 10, 5), (18000, 100, 2000), (19000, 7, 2000, 1000)]:
        assert output_steps(*args) == trajectory.output_steps(*args)


def test_resolved_output_steps_include_every_boundary():
    resolved = (ramp(0, 400, ps=1) + hold(400, ps=0.5) + ramp(400, 0, ps=8)).resolve(0.5)
    assert resolved.output_steps(7) == output_steps(19000, 7, 2000, 1000)
    four = (ramp(0, 10, ps=0.005) + ramp(10, 20, ps=0.003) + ramp(20, 0, ps=0.004)).resolve(1.0)
    assert four.output_steps(5) == [0, 5, 8, 10, 12]


# --- profiles as data --------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(RESEARCH_PROFILES))
def test_profiles_carry_the_research_settings(name):
    heat_ps, cool_ps, peak, final, interval, fmax = RESEARCH_PROFILES[name]
    profile = PROFILES[name]
    assert isinstance(profile, Profile) and profile.name == name
    assert profile.schedule == ramp(0, peak, ps=heat_ps) + ramp(peak, final, ps=cool_ps)
    assert (profile.sample_interval, profile.trajectory_interval) == (interval, interval)
    assert profile.relax_fmax == fmax
    assert profile.relax_optimizer == "LBFGS"


def test_anneal_quench_settings_are_data():
    quench = PROFILES["anneal"].quench
    assert quench == {"min_K": 500.0, "bin_K": 100.0, "per_bin": 2, "fmax": 0.01, "optimizer": "LBFGS",
                      "max_steps": 5000}
    assert PROFILES["slow"].quench is None


def test_profiles_are_read_only():
    with pytest.raises(TypeError):
        PROFILES["mine"] = PROFILES["slow"]


ANNEAL_TOML = """
name = "anneal"
sample_interval = 20
trajectory_interval = 20
relax_fmax = 0.01

[[segments]]
kind = "ramp"
start_K = 0
end_K = 1000
ps = 20

[[segments]]
kind = "ramp"
start_K = 1000
end_K = 250
rate_K_per_ps = -3

[quench]
min_K = 500.0
bin_K = 100.0
per_bin = 2
fmax = 0.01
optimizer = "LBFGS"
max_steps = 5000
"""


def test_profile_from_toml(tmp_path):
    path = tmp_path / "anneal.toml"
    path.write_text(ANNEAL_TOML)
    profile = load_profile(path)
    resolved, expected = profile.schedule.resolve(0.5), PROFILES["anneal"].schedule.resolve(0.5)
    assert resolved.boundaries == expected.boundaries
    assert resolved.rates_K_per_ps == expected.rates_K_per_ps
    assert all(resolved.target(s) == expected.target(s) for s in range(0, 540_001, 997))
    assert profile.quench == PROFILES["anneal"].quench
    assert profile.sample_interval == 20 and profile.relax_fmax == 0.01


def test_profile_from_yaml_uses_safe_load(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "heat.yaml"
    path.write_text("name: heat\nsegments:\n  - {kind: ramp, start_K: 0, end_K: 400, ps: 1}\n"
                    "  - {kind: hold, temperature_K: 400, ps: 2}\n")
    profile = load_profile(path)
    assert profile.schedule == ramp(0, 400, ps=1) + hold(400, ps=2)
    bad = tmp_path / "bad.yml"
    bad.write_text("!!python/object/apply:os.system ['echo unsafe']\n")
    with pytest.raises(yaml.YAMLError):
        load_profile(bad)


def test_yaml_without_pyyaml_is_a_clear_import_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)
    path = tmp_path / "p.yaml"
    path.write_text("name: x\n")
    with pytest.raises(ImportError, match="PyYAML"):
        load_profile(path)


@pytest.mark.parametrize("data", [
    {"segments": []},
    {"name": "x"},
    {"name": "x", "segments": [{"kind": "spin", "ps": 1}]},
    {"name": "x", "segments": [{"kind": "ramp", "start_K": 0, "end_K": 1, "ps": 1, "colour": "red"}]},
    {"name": "x", "segments": [{"kind": "ramp", "start_K": 0, "end_K": 1, "ps": 1}], "unknown": 1},
    {"name": "x", "segments": [{"kind": "ramp", "start_K": 0, "end_K": 1, "ps": 1}], "sample_interval": 0},
    {"name": "x", "segments": [{"kind": "ramp", "start_K": 0, "end_K": 1, "ps": 1},
                               {"kind": "hold", "temperature_K": 2, "ps": 1}]},
])
def test_invalid_profile_data_raises(data):
    with pytest.raises(ValueError):
        profile_from_data(data)


def test_unknown_profile_suffix(tmp_path):
    path = tmp_path / "p.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="toml|yaml"):
        load_profile(path)


def test_protocols_import_is_light():
    """fm.protocols is a record module (contracts-1b.md): no torch, MACE or ASE."""
    code = ("import json, sys; import fastmace.protocols; "
            "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'mace', 'ase', 'numpy'))))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == []
    assert protocols.ramp is ramp
