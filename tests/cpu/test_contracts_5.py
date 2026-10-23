"""The contracts of phase 5 (plan task 5-T01, docs/design/planner-5.md): names, fields and defaults the later
phase 5 tasks build against.

- ``AdaptationConfig`` and ``CalibrationConfig``: the fields an open owner decision fixes (D2, D3, D5) default
  to ``config.PENDING``, and ``planner_options`` refuses any other value, and ``calibrate=True``, until 5-T07
  and 5-T09 land. The defaults plan and run as before (conductor's choice 7).
- ``PlannerOptions`` records both and reads records from before phase 5.
- ``Plan.calibration`` and ``Plan.adaptation`` default to ``None`` and stay out of the restart record (its
  version stays 2); ``CalibrationRecord`` and ``TierSwitch`` are the records 5-T09 and 5-T07 fill.
"""

import dataclasses
import json

import pytest

from fastmace import config
from fastmace.errors import PlanRefused


def test_adaptation_defaults_wait_for_d2_and_d3():
    adaptation = config.AdaptationConfig()
    assert adaptation.adapt == "auto" and adaptation.trigger == config.PENDING == "pending"
    assert adaptation.recapture_time_fraction == 0.2  # D2 (a)'s threshold, read only by the triggers that use it
    assert config.ADAPTATION_TRIGGERS == ("recapture-time", "recapture-budget", "none")
    assert config.ADAPTATION_DECISION_FIELDS == ("trigger", "skin_replan", "max_switches", "switch_back_after",
                                                 "min_points_between", "reproducible")
    assert all(getattr(adaptation, name) == config.PENDING for name in config.ADAPTATION_DECISION_FIELDS)
    assert adaptation.decided() == ()
    # Every option of D2 and D3 fits the record.
    for fields in ({"trigger": "recapture-time", "skin_replan": False, "max_switches": 1, "switch_back_after": None,
                    "min_points_between": None, "reproducible": False},  # D2 (a), D3 (a)
                   {"trigger": "recapture-time", "skin_replan": True, "max_switches": 2, "switch_back_after": 3},
                   {"trigger": "recapture-budget", "reproducible": True},  # D2 (c), D3 (c)
                   {"trigger": "none", "max_switches": None, "min_points_between": 4}):  # D2 (d), D3 (d)
        assert config.AdaptationConfig(**fields).decided() == tuple(
            name for name in config.ADAPTATION_DECISION_FIELDS if name in fields)


def test_calibration_defaults_wait_for_d5():
    calibration = config.CalibrationConfig()
    assert (calibration.calibrate, calibration.steps, calibration.margin, calibration.choose) == (
        False, 50, 0.05, config.PENDING)
    assert config.CalibrationConfig(calibrate=True, choose=False).choose is False  # D5 (b) fits


def test_planner_options_record_both_and_read_older_records():
    from fastmace.simulation import PlannerOptions, planner_options

    default = planner_options()
    assert default.adaptation == config.AdaptationConfig() and default.calibration == config.CalibrationConfig()
    off = planner_options(adaptation=config.AdaptationConfig(adapt="off"))
    record = json.loads(json.dumps(off.to_record()))
    assert record["adaptation"]["adapt"] == "off" and record["calibration"]["calibrate"] is False
    assert PlannerOptions.from_record(record) == off
    del record["adaptation"], record["calibration"]  # a simulation.json from before phase 5
    assert PlannerOptions.from_record(record) == planner_options()
    with pytest.raises(TypeError, match="AdaptationConfig"):
        planner_options(adaptation="off")


@pytest.mark.parametrize(("kwargs", "code"), [
    ({"adaptation": config.AdaptationConfig(trigger="recapture-time")}, "adaptation-pending"),
    ({"adaptation": config.AdaptationConfig(max_switches=1)}, "adaptation-pending"),
    ({"calibration": config.CalibrationConfig(calibrate=True)}, "calibration-pending"),
    ({"calibration": config.CalibrationConfig(choose=True)}, "calibration-pending"),
])
def test_a_decision_not_taken_yet_is_refused(kwargs, code):
    from fastmace.simulation import PHASE_5_PENDING_CODES, planner_options

    with pytest.raises(PlanRefused) as refused:
        planner_options(**kwargs)
    assert refused.value.codes == (code,) and code in PHASE_5_PENDING_CODES.values()


def test_plan_records_of_phase_5():
    from fastmace.planner.plan import CalibrationCandidate, CalibrationRecord, Plan, TierSwitch

    plan = Plan(tier="graph")
    assert plan.calibration is None and plan.adaptation is None and plan.overrides() == ()
    record = CalibrationRecord(candidates=(CalibrationCandidate("", [4.1, 4.2]),
                                           CalibrationCandidate("backend cueq", (4.6, 4.5))),
                               chosen="", changed_plan=False, steps=50, margin=0.05, setup_s=31.0)
    assert record.candidates[0].step_ms == (4.1, 4.2)
    with pytest.raises(ValueError, match="chosen"):
        dataclasses.replace(record, chosen="backend hybrid")
    switch = TierSwitch(step=20000, from_tier="graph", to_tier="compiled", reason="re-capture storm", revision=1)
    assert switch.record() == {"step": 20000, "from": "graph", "to": "compiled", "reason": "re-capture storm",
                               "revision": 1}
    revised = plan.revise("tier switch", 20000, calibration=record, adaptation=config.AdaptationConfig())
    assert revised.calibration == record and revised.predecessor is plan


def test_the_restart_record_stays_version_2():
    from fastmace.planner import plan as plans
    from fastmace.planner.plan import CalibrationCandidate, CalibrationRecord, Plan

    plan = Plan(tier="graph", engine=config.EngineConfig(), precision=config.PrecisionConfig(),
                calibration=CalibrationRecord((CalibrationCandidate("", (4.0,)),), "", False, 50, 0.05, 1.0),
                adaptation=config.AdaptationConfig())
    record = plan.to_restart(edge_bucket=1024, neighbour_capacity=2048)
    assert plans.PLAN_RECORD_VERSION == 2 and "calibration" not in record and "adaptation" not in record
    again = Plan.from_restart(json.loads(json.dumps(record)))
    assert again.calibration is None and again.adaptation is None
