"""The optional calibration (SPEC §9.1; plan task 5-T09, owner decision 55; ``fastmace.planner.calibrate``).

- The choice: plain cuEq replaces the rules' hybrid only when faster by more than the margin in every pair of
  the A B A B blocks; a block that fails keeps the rules' plan.
- The tie: hybrid (auto, OEq builds) against plain cuEq where both pass the hard rules. No tie, so no child,
  when the caller set the backend, OEq is unavailable, MACE was imported first, the process compiled the other
  family, ``reproducible=True``, or rule 3 refuses plain cuEq (PolarMACE); planning the tie claims nothing in
  this process. The tiny model's traits with injected ``SoftwareTraits`` and ``Hardware``: no GPU.
- ``Simulation``: MD only (relaxations and MC are refused), ``choose=False`` refused; on the CPU there is no
  tie and the report says why; a forced tie with fake blocks records the measurement in the plan, provenance,
  the report and the setup times, and a changed backend reaches the recorded options, so a resume plans it
  again and never calibrates. Nothing is read from an earlier run (planted records in the cache and model
  directories, watched with an audit hook).
- The child block (``SubprocessRunner.calibrate``) on the CPU: a real fresh child times its steps and leaves
  no file behind; the whole calibration in children writes nothing outside the run directory.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys

import pytest
import test_minimal_planner as tmp
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace import integrators
from fastmace.config import CalibrationConfig, EngineConfig, IntegratorConfig
from fastmace.engine import constraints
from fastmace.errors import PlanRefused
from fastmace.planner import calibrate, minimal
from fastmace.planner.plan import CalibrationRecord
from fastmace.planner.workload import Workload

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

WORKLOAD = tmp.WORKLOAD
CALIBRATE = CalibrationConfig(calibrate=True)
NA_CL = (11, 17)
LANGEVIN = integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01)
CPU_ENGINE = EngineConfig(device="cpu", compile_mode="none")


def plan_with_traits(label="mh1-r2scan", workload=WORKLOAD):
    def plan_with(planner):
        return planner.plan_traits(tmp.tp.traits(label), nacl(), IntegratorConfig(), workload, tmp.GPU,
                                   precision="float32", rewrites="auto")

    return plan_with


def fake_launch(times):
    """``launch`` returning ``times[change]`` in order, and recording the calls."""
    queues = {change: list(values) for change, values in times.items()}
    calls = []

    def launch(change):
        calls.append(change)
        return queues[change].pop(0)

    launch.calls = calls
    return launch


class Clock:
    def __init__(self, step=10.0):
        self.now, self.step = 0.0, step

    def __call__(self):
        self.now += self.step
        return self.now


# --------------------------------------------------------------------------- the choice


def test_blocks_alternate_rules_first():
    assert calibrate.order() == ("", "backend cueq", "", "backend cueq")
    assert calibrate.PAIRS == 2 and calibrate.WARMUP_STEPS == 50


@pytest.mark.parametrize(("rules", "other", "faster"), [
    ((10.0, 10.0), (9.0, 9.0), True),  # 10 % in both pairs
    ((10.0, 10.0), (9.0, 9.7), False),  # 3 % in the second pair: not every pair
    ((10.0, 10.0), (9.5, 9.5), False),  # exactly the margin: not more than it
    ((10.0, 12.0), (9.4, 11.0), True),  # each pair against its own rules block
    ((10.0, 10.0), (11.0, 11.0), False),
    ((), (), False),
])
def test_faster_by_more_than_the_margin_in_every_pair(rules, other, faster):
    assert calibrate.faster_in_every_pair(rules, other, 0.05) is faster


def test_no_tie_runs_no_block():
    found = calibrate.Tie(rules=None, alternative=None, why="the backend is set (EngineConfig.backend='cueq')")

    def launch(change):
        raise AssertionError("no tie: no block")

    outcome = calibrate.calibrate(object(), found, launch, CALIBRATE)
    assert outcome.record is None and not outcome.changed
    assert outcome.note == "not run: no tie, the backend is set (EngineConfig.backend='cueq')"
    assert calibrate.report_line(None, outcome.note) == f"calibration {outcome.note}"
    assert calibrate.report_line(None) is None


def test_a_faster_alternative_in_every_pair_changes_the_backend():
    planner = tmp.minimal()
    found = calibrate.tie(planner, plan_with_traits(), WORKLOAD)
    launch = fake_launch({"": [10.0, 10.2], "backend cueq": [9.0, 9.1]})
    outcome = calibrate.calibrate(planner, found, launch, CALIBRATE, clock=Clock(), setup_s=1.5)
    assert launch.calls == ["", "backend cueq", "", "backend cueq"]
    record = outcome.record
    assert record.changed_plan and record.chosen == "backend cueq" and outcome.changed
    assert [(c.change, c.step_ms) for c in record.candidates] == [("", (10.0, 10.2)), ("backend cueq", (9.0, 9.1))]
    assert (record.steps, record.margin, record.setup_s) == (50, 0.05, 11.5)
    assert outcome.planner.engine.backend == "cueq" and outcome.planner.process is planner.process
    assert planner.engine.backend == "auto"  # not changed in place

    plan = calibrate.apply(outcome.planner.plan_traits(tmp.tp.traits("mh1-r2scan"), nacl(), IntegratorConfig(),
                                                       WORKLOAD, tmp.GPU, precision="float32", rewrites="auto"),
                           outcome)
    key, choice, reason = plan.decision("EngineConfig.backend")
    assert (choice, reason.code, reason.rule) == ("cueq", "backend-calibrated", None)
    assert "measured in this run" in reason and "the rules gave hybrid" in reason
    assert all(item != "EngineConfig.backend" for item, _ in plan.requested)  # not the caller's request
    assert plan.calibration == record and plan.engine.backend == "cueq"
    line = calibrate.report_line(record)
    assert line.startswith("calibration chose plain cuEq; measured in this run, 50 steps per block")
    assert "10, 10.2 ms/step" in line and "9, 9.1 ms/step" in line and "speed-up" not in line


@pytest.mark.parametrize("cueq", [(9.0, 9.7), (10.5, 10.4), (9.6, 9.6)])
def test_otherwise_the_rules_plan_stays(cueq):
    planner = tmp.minimal()
    found = calibrate.tie(planner, plan_with_traits(), WORKLOAD)
    outcome = calibrate.calibrate(planner, found, fake_launch({"": [10.0, 10.0], "backend cueq": list(cueq)}),
                                  CALIBRATE, clock=Clock())
    assert not outcome.changed and outcome.record.chosen == "" and outcome.planner is planner
    plan = calibrate.apply(found.rules, outcome)
    assert plan.calibration == outcome.record and plan.decision("EngineConfig.backend")[2].code == "backend-hybrid"
    assert calibrate.report_line(outcome.record).startswith("calibration kept the rules' plan")


def test_a_failed_block_keeps_the_rules_plan():
    planner = tmp.minimal()
    found = calibrate.tie(planner, plan_with_traits(), WORKLOAD)
    calls = []

    def launch(change):
        calls.append(change)
        if change == "backend cueq":
            raise RuntimeError("out of memory in the child")
        return 10.0

    outcome = calibrate.calibrate(planner, found, launch, CALIBRATE, clock=Clock())
    assert calls == ["", "backend cueq"] and not outcome.changed and outcome.planner is planner
    assert [c.step_ms for c in outcome.record.candidates] == [(10.0,), ()]
    assert "calibration stopped" in outcome.warning and "out of memory in the child" in outcome.warning


# --------------------------------------------------------------------------- the tie


def test_hybrid_against_plain_cueq_is_the_tie_and_claims_nothing():
    process = constraints.ProcessRecord()
    planner = tmp.minimal(process=process)
    found = calibrate.tie(planner, plan_with_traits(), WORKLOAD)
    assert found.why is None
    assert found.rules.engine.backend == "hybrid" and found.alternative.engine.backend == "cueq"
    assert found.rules.decision("EngineConfig.backend")[2].code == "backend-hybrid"
    assert (process.family, process.engines) == (None, 0) and planner.process is process
    # The run's own plan after the probes: the process claims as usual.
    planner.plan_traits(tmp.tp.traits("mh1-r2scan"), nacl(), IntegratorConfig(), WORKLOAD, tmp.GPU,
                        precision="float32")
    assert process.family == constraints.OEQ_FAMILY


@pytest.mark.parametrize(("planner", "label", "why"), [
    (lambda: tmp.minimal(engine=EngineConfig(backend="hybrid")), "mh1-r2scan", "the backend is set"),
    (lambda: tmp.minimal(engine=EngineConfig(backend="cueq")), "mh1-r2scan", "the backend is set"),
    (lambda: tmp.minimal(soft=tmp.software(oeq=(False, "no openequivariance"))), "mh1-r2scan",
     "the rules chose cueq"),
    (lambda: tmp.minimal(soft=tmp.software(mace_imported_first=("mace.modules",))), "mh1-r2scan",
     "the rules chose cueq"),
    (lambda: tmp.minimal(engine=CPU_ENGINE), "mh1-r2scan", "the rules chose cueq"),
    (lambda: tmp.minimal(), "polar1-s", "plain cuEq is refused (polar-plain-compiled-cueq)"),
])
def test_no_tie(planner, label, why):
    found = calibrate.tie(planner(), plan_with_traits(label), WORKLOAD)
    assert found.alternative is None and why in found.why


def test_no_tie_after_the_process_compiled_plain_cueq():
    process = constraints.ProcessRecord()
    process.claim_family(constraints.AUTOGRAD_FAMILY, "an earlier plain cuEq run")
    found = calibrate.tie(tmp.minimal(process=process), plan_with_traits(), WORKLOAD)
    assert found.alternative is None and "backend-cueq-process-family" not in found.why
    assert found.rules.engine.backend == "cueq" and "the rules chose cueq" in found.why


def test_no_tie_for_reproducible_without_planning():
    def plan_with(planner):
        raise AssertionError("a reproducible plan prepares the process: not planned for the calibration")

    found = calibrate.tie(tmp.minimal(), plan_with, Workload(max_temperature_K=300.0, reproducible=True))
    assert found.alternative is None and found.rules is None and "rule 5" in found.why


# --------------------------------------------------------------------------- Simulation


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    pytest.importorskip("cuequivariance_torch")
    from fastmace.model import loading

    model = tiny_mace_model(list(NA_CL), **FAMILIES["mh1"])
    path = tmp_path_factory.mktemp("models") / "tiny-mh1-11-17.model"
    torch.save(model, path)
    loading._LICENCES_SHOWN.add(str(path))
    return loading.load_pristine(str(path), trusted=True)


@pytest.fixture(autouse=True)
def cpu_planner(monkeypatch, tmp_path):
    planner = minimal.MinimalPlanner(engine=CPU_ENGINE, environ={}, process=constraints.ProcessRecord())
    monkeypatch.setattr(minimal, "default_planner", lambda force_field: planner)
    monkeypatch.setenv("FASTMACE_CACHE", str(tmp_path / "cache"))
    return planner


def simulation(loaded, **kwargs):
    from fastmace.simulation import Simulation

    kwargs.setdefault("integrator", LANGEVIN)
    return Simulation(nacl(), loaded, seed=7, **kwargs)


@pytest.fixture
def forced_tie(monkeypatch):
    """A tie on the CPU (both candidates plain cuEq there) and fake blocks: ``set(times)``; the calls."""
    calls = []
    state = {}

    def tie(planner, plan_with, workload):
        return calibrate.Tie(plan_with(calibrate.probe(planner)), plan_with(calibrate.probe(planner, "cueq")))

    class Launcher:
        def __init__(self, simulation, planner, found, config):
            self.launch = fake_launch(state["times"])

        def __call__(self, change):
            calls.append(change)
            return self.launch(change)

        def close(self):
            calls.append("close")

    monkeypatch.setattr(calibrate, "tie", tie)
    monkeypatch.setattr(calibrate, "ChildLauncher", Launcher)

    def set_times(times):
        state["times"] = times
        return calls

    return set_times


def test_relaxations_and_mc_are_refused(loaded):
    from fastmace import optimize

    with pytest.raises(PlanRefused) as refused:
        simulation(loaded, integrator=None, optimizer=optimize.FIRE(), calibration=CALIBRATE)
    assert refused.value.codes == ("calibration-workload",) and "relaxation" in str(refused.value)
    mc = Workload(max_temperature_K=300.0, species_changes=True, mc_moves=("swap",), min_mc_temperature_K=300.0)
    with pytest.raises(PlanRefused) as refused:
        simulation(loaded, workload=mc, calibration=CALIBRATE)
    assert refused.value.codes == ("calibration-workload",) and "MC moves swap" in str(refused.value)
    with pytest.raises(PlanRefused) as refused:
        simulation(loaded, calibration=CalibrationConfig(calibrate=True, choose=False))
    assert refused.value.codes == ("calibration-unsupported",)


def test_no_tie_on_the_cpu_is_reported(loaded):
    sim = simulation(loaded, calibration=CALIBRATE)
    assert sim.plan.calibration is None and "calibration" not in sim.report.setup_s
    line = next(text for text in sim.report.text().splitlines() if text.startswith("backend:"))
    assert "calibration not run: no tie, the rules chose cueq" in line
    assert all(not text.startswith("backend:") for text in simulation(loaded).report.text().splitlines())


def test_a_kept_plan_is_recorded_in_the_plan_provenance_and_report(loaded, forced_tie, tmp_path):
    from fastmace.io.output import Output

    calls = forced_tie({"": [4.0, 4.1], "backend cueq": [4.3, 4.2]})
    sim = simulation(loaded, calibration=CALIBRATE, output=Output(tmp_path / "run"))
    assert calls == ["", "backend cueq", "", "backend cueq", "close"]
    record = sim.plan.calibration
    assert isinstance(record, CalibrationRecord) and not record.changed_plan
    assert sim.report.setup_s["calibration"] == record.setup_s
    assert any(text.startswith("backend: calibration kept the rules' plan") for text in sim.report.text().splitlines())
    assert "calibration: kept the rules' plan" in str(sim.plan)
    sim.run(steps=2)
    sim.close()
    metadata = json.loads((tmp_path / "run" / "metadata.json").read_text(encoding="utf-8"))
    stored = metadata["plan"]["calibration"]
    assert stored["changed_plan"] is False and [c["step_ms"] for c in stored["candidates"]] == [[4.0, 4.1], [4.3, 4.2]]


def test_a_changed_backend_is_recorded_and_resumed_without_calibrating(loaded, forced_tie, tmp_path):
    calls = forced_tie({"": [4.0, 4.1], "backend cueq": [3.0, 3.1]})
    sim = simulation(loaded, calibration=CALIBRATE)
    assert sim.plan.calibration.changed_plan and sim.plan.decision("EngineConfig.backend")[2].code == (
        "backend-calibrated")
    assert sim.options.engine.backend == "cueq" and sim.options.calibration == CALIBRATE
    sim.run(steps=2)
    sim.save_restart(tmp_path / "restart")
    record = json.loads((tmp_path / "restart" / "simulation.json").read_text(encoding="utf-8"))
    assert record["planner_options"]["engine"]["backend"] == "cueq"
    assert record["planner_options"]["calibration"]["calibrate"] is True
    calls.clear()
    from fastmace.simulation import Simulation

    again = Simulation.resume(tmp_path / "restart")
    assert calls == [] and again.plan.engine.backend == "cueq" and again.plan.calibration is None
    assert all(not text.startswith("backend:") for text in again.report.text().splitlines())


def test_no_earlier_record_is_read(loaded, forced_tie, tmp_path, monkeypatch):
    planted = []
    for name in ("cache", "models"):
        directory = tmp_path / name
        directory.mkdir(exist_ok=True)
        for file in ("calibration.json", "calibrate.json", "plan.json"):
            path = directory / file
            path.write_text(json.dumps({"backend": "cueq", "step_ms": 0.001}), encoding="utf-8")
            planted.append(str(path))
    monkeypatch.setenv("FASTMACE_MODELS", str(tmp_path / "models"))
    opened = []
    active = [True]

    def hook(event, arguments):
        if active[0] and event == "open" and arguments and isinstance(arguments[0], (str, os.PathLike)):
            path = os.fspath(arguments[0])
            if any(path.startswith(str(tmp_path / name)) for name in ("cache", "models")):
                opened.append(path)

    sys.addaudithook(hook)
    try:
        forced_tie({"": [4.0, 4.0], "backend cueq": [4.1, 4.1]})
        sim = simulation(loaded, calibration=CALIBRATE)
    finally:
        active[0] = False
    assert not sim.plan.calibration.changed_plan and opened == []


# --------------------------------------------------------------------------- the child block


def _files(root):
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def test_a_child_block_times_its_steps_and_leaves_nothing(loaded, tmp_path):
    from fastmace.engine.subprocess import SubprocessRunner
    from fastmace.io.output import Output

    sim = simulation(loaded, output=Output(tmp_path / "run"))
    before = _files(tmp_path)
    runner = SubprocessRunner(dataclasses.replace(sim.options.runner, persistent=False))
    options = dataclasses.replace(sim.options, engine=dataclasses.replace(CPU_ENGINE, backend="cueq"))
    seconds = runner.calibrate(sim, options, warmup_steps=3, steps=4)
    assert 0 < seconds < 60 and runner.children == 1
    assert runner.root.parent == (tmp_path / "run" / ".fastmace-handoff").resolve()
    assert [path for path in runner.root.iterdir() if path.is_dir()] == []  # the stage is gone
    runner.close()
    assert _files(tmp_path) == before and sim.committed().step == 0  # the run itself did not move


def test_the_calibration_in_children_writes_only_in_the_run_directory(loaded, tmp_path, monkeypatch):
    """The whole path with real children (four CPU blocks of a few steps), the tie forced on the CPU."""
    from fastmace.io.output import Output

    monkeypatch.setattr(calibrate, "WARMUP_STEPS", 2)
    monkeypatch.setattr(calibrate, "tie", lambda planner, plan_with, workload: calibrate.Tie(
        plan_with(calibrate.probe(planner)), plan_with(calibrate.probe(planner, "cueq"))))
    monkeypatch.chdir(tmp_path)
    before = _files(tmp_path)
    sim = simulation(loaded, calibration=CalibrationConfig(calibrate=True, steps=3), output=Output(tmp_path / "run"))
    record = sim.plan.calibration
    assert [len(candidate.step_ms) for candidate in record.candidates] == [2, 2]
    assert all(ms > 0 for candidate in record.candidates for ms in candidate.step_ms)
    assert record.setup_s > 0 and sim.report.setup_s["calibration"] == record.setup_s
    new = [path for path in _files(tmp_path) if path not in before]
    assert new and all(path.startswith("run" + os.sep) or path.startswith("run/") for path in new), new
    assert not (tmp_path / "run" / ".fastmace-handoff").exists()
    sim.close()
