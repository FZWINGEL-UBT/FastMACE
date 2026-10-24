"""The contracts of phase 5 (plan task 5-T01, docs/design/planner-5.md): names, fields and defaults the later
phase 5 tasks build against.

- ``AdaptationConfig``: the defaults are decisions 52 and 53 (5-T07); the options not chosen are refused
  (``test_adapt_switch.py``). ``CalibrationConfig``: the defaults are decision 55 (5-T09, off by default);
  ``choose=False`` (D5 (b), not chosen) is refused (``test_calibrate.py``). The defaults plan as before
  (conductor's choice 7).
- ``PlannerOptions`` records both and reads records from before phase 5.
- ``Plan.calibration`` and ``Plan.adaptation`` default to ``None`` and stay out of the restart record (its
  version stays 2); ``CalibrationRecord`` and ``TierSwitch`` are the records 5-T09 and 5-T07 fill.
"""

import dataclasses
import json

import pytest

from fastmace import config
from fastmace.errors import PlanRefused


def test_adaptation_defaults_are_decisions_52_and_53():
    adaptation = config.AdaptationConfig()
    assert adaptation.adapt == "auto" and adaptation.trigger == "recapture-time"
    assert adaptation.recapture_time_fraction == 0.2  # decision 52's threshold
    assert config.ADAPTATION_TRIGGERS == ("recapture-time", "recapture-budget", "none")
    assert config.ADAPTATION_DECISION_FIELDS == ("trigger", "skin_replan", "max_switches", "switch_back_after",
                                                 "min_points_between", "reproducible")
    assert set(config.ADAPTATION_SUPPORTED) == set(config.ADAPTATION_DECISION_FIELDS)
    assert all(getattr(adaptation, name) in config.ADAPTATION_SUPPORTED[name]
               for name in config.ADAPTATION_DECISION_FIELDS)
    assert adaptation.unsupported() == ()
    # The options not chosen still fit the record (and are refused at planner_options).
    for fields in ({"trigger": "recapture-time", "skin_replan": True, "max_switches": 2, "switch_back_after": 3},
                   {"trigger": "recapture-budget", "reproducible": True},  # D2 (c), D3 (c)
                   {"trigger": "none", "max_switches": None, "min_points_between": 4}):  # D2 (d), D3 (d)
        assert config.AdaptationConfig(**fields).unsupported() == tuple(
            name for name in config.ADAPTATION_DECISION_FIELDS
            if name in fields and fields[name] not in config.ADAPTATION_SUPPORTED[name])


def test_calibration_defaults_are_decision_55():
    calibration = config.CalibrationConfig()
    assert (calibration.calibrate, calibration.steps, calibration.margin, calibration.choose) == (
        False, 50, 0.05, True)
    assert config.CalibrationConfig(calibrate=True, choose=False).choose is False  # D5 (b) fits the record


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
    ({"adaptation": config.AdaptationConfig(trigger="recapture-budget")}, "adaptation-unsupported"),
    ({"adaptation": config.AdaptationConfig(max_switches=2)}, "adaptation-unsupported"),
    ({"calibration": config.CalibrationConfig(choose=False)}, "calibration-unsupported"),
    ({"calibration": config.CalibrationConfig(calibrate=True, choose=False)}, "calibration-unsupported"),
])
def test_an_option_not_implemented_is_refused(kwargs, code):
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
