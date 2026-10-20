"""Phase 2 contracts (plan task 2-T01; SPEC §7.6, §8.7, §9.2–§9.5, §12.3).

The new config records and fields, the error classes, ``Rewrite.deterministic``, ``Plan.runner``,
the planner inputs of ``Simulation`` and ``fm.md``, and the stubs that later phase 2 tasks fill.
Everything here is torch-free; ``docs/design/contracts-2.md`` is the page that explains it.
"""

from __future__ import annotations

import dataclasses
import inspect
import pickle
import sys
import typing

import pytest

from fastmace import config, errors


def _frozen(instance, name, value):
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, name, value)


def _keyword_only(signature):
    return [name for name, p in signature.parameters.items() if p.kind is inspect.Parameter.KEYWORD_ONLY]


# --------------------------------------------------------------------------- config records


def test_mode_type_lists_the_four_tiers():
    assert typing.get_args(config.Mode) == ("graph", "compiled", "eager", "reference")
    assert config.MODES == typing.get_args(config.Mode)
    from fastmace.planner import minimal

    assert minimal.MODES == config.MODES


def test_fallback_config_defaults():  # SPEC §9.4, decision 29
    fallback = config.FallbackConfig()
    assert fallback.fallback == "auto"
    assert fallback.diagnosis_budget == 64
    assert fallback.reference_last is True
    _frozen(fallback, "fallback", "refuse")
    assert config.FallbackConfig(fallback="refuse").fallback == "refuse"


@pytest.mark.parametrize("kwargs", [{"fallback": "maybe"}, {"diagnosis_budget": 0}, {"diagnosis_budget": 2.5},
                                    {"diagnosis_budget": True}, {"reference_last": "yes"}])
def test_fallback_config_validation(kwargs):
    with pytest.raises(ValueError):
        config.FallbackConfig(**kwargs)


def test_gate_config_defaults():  # SPEC §12.3 "Resources"
    gate = config.GateConfig()
    assert gate.cpu_float64_time_limit_s == 300.0
    assert gate.reduced_cell == "auto"
    assert gate.reduced_cell_min_height_cutoffs == 2.0
    _frozen(gate, "reduced_cell", "off")
    assert config.GateConfig(reduced_cell="off", cpu_float64_time_limit_s=0).cpu_float64_time_limit_s == 0


@pytest.mark.parametrize("kwargs", [{"cpu_float64_time_limit_s": -1}, {"cpu_float64_time_limit_s": float("inf")},
                                    {"reduced_cell": "on"}, {"reduced_cell_min_height_cutoffs": 0},
                                    {"reduced_cell_min_height_cutoffs": 0.5}])
def test_gate_config_validation(kwargs):
    with pytest.raises(ValueError):
        config.GateConfig(**kwargs)


def test_runner_config_defaults():  # SPEC §8.7
    runner = config.RunnerConfig()
    assert runner.python is None
    assert runner.resolved_python() == sys.executable
    assert runner.environment == ()
    assert runner.inherit_environment is True
    assert runner.timeout_s is None
    assert runner.stall_timeout_s == 1800.0
    _frozen(runner, "python", "/usr/bin/python3")


def test_runner_config_normalises_the_environment():
    runner = config.RunnerConfig(python="/opt/python", environment={"B": "2", "A": "1"})
    assert runner.environment == (("A", "1"), ("B", "2"))
    assert runner.resolved_python() == "/opt/python"
    assert hash(runner) == hash(config.RunnerConfig(python="/opt/python", environment=[("B", "2"), ("A", "1")]))


@pytest.mark.parametrize("kwargs", [{"python": ""}, {"python": 3}, {"environment": {"A": 1}},
                                    {"environment": "A=1"}, {"timeout_s": 0}, {"stall_timeout_s": -5},
                                    {"stall_timeout_s": None}, {"inherit_environment": 1}])
def test_runner_config_validation(kwargs):
    with pytest.raises(ValueError):
        config.RunnerConfig(**kwargs)


def test_new_fields_of_existing_records():
    assert config.EngineConfig().dynamic_edges is False  # 2-T13
    assert config.RewriteConfig().selfcheck_cache == "use"  # 2-T19, inert until then
    assert config.RewriteConfig(selfcheck_cache="refresh").selfcheck_cache == "refresh"
    with pytest.raises(ValueError):
        config.RewriteConfig(selfcheck_cache="always")
    with pytest.raises(ValueError):
        config.EngineConfig(dynamic_edges="yes")


def test_reproducible_stays_on_the_workload():
    from fastmace.planner.workload import Workload

    assert Workload(max_temperature_K=300.0).reproducible is False


# --------------------------------------------------------------------------- errors


def test_fallback_exhausted_is_a_refusal_with_the_revisions():
    from fastmace.errors import FallbackExhausted, PlanRefused, Reason

    reason = Reason("every candidate failed", code="fallback-exhausted")
    error = FallbackExhausted([reason], revisions=("plan 0", "plan 1"))
    assert isinstance(error, PlanRefused)
    assert error.revisions == ("plan 0", "plan 1")
    assert error.codes == ("fallback-exhausted",)
    assert "Record:" in str(error) and "run.plan" in str(error)
    again = pickle.loads(pickle.dumps(error))
    assert again.revisions == error.revisions and again.codes == error.codes
    assert FallbackExhausted([reason]).revisions == ()


def test_subprocess_runner_error():
    from fastmace.errors import FastMACEError, SubprocessRunnerError

    error = SubprocessRunnerError("the child process failed", exit_code=-9, child_message="Killed",
                                  directory="/run/.fastmace-handoff")
    assert isinstance(error, FastMACEError) and isinstance(error, RuntimeError)
    assert (error.exit_code, error.child_message, error.directory) == (-9, "Killed", "/run/.fastmace-handoff")
    assert "Next:" in str(error) and "Record:" in str(error)
    assert pickle.loads(pickle.dumps(error)).exit_code == -9


def test_isolation_error():
    from fastmace.errors import FastMACEError, IsolationError

    error = IsolationError("the polar k-space lease was taken", component="polar k-space", holder="ForceField B")
    assert isinstance(error, FastMACEError) and isinstance(error, RuntimeError)
    assert (error.component, error.holder) == ("polar k-space", "ForceField B")
    assert "rebind()" in str(error)


def test_reproducibility_error():
    from fastmace.errors import FastMACEError, PlanRefused, ReproducibilityError

    error = ReproducibilityError("the family is not qualified", blocking=["cuEq fused convolution"])
    assert isinstance(error, FastMACEError) and not isinstance(error, PlanRefused)
    assert error.blocking == ("cuEq fused convolution",)
    assert ReproducibilityError("x").blocking == ()


def test_new_errors_are_exported():
    for name in ("FallbackExhausted", "SubprocessRunnerError", "IsolationError", "ReproducibilityError"):
        assert name in errors.__all__


# --------------------------------------------------------------------------- Rewrite and Plan


def test_rewrite_deterministic_defaults_to_undeclared():
    from fastmace.rewrites.registry import Rewrite

    field = {f.name: f for f in dataclasses.fields(Rewrite)}["deterministic"]
    assert field.default is None
    assert list(f.name for f in dataclasses.fields(Rewrite))[-1] == "deterministic"


def _toy_fields(name="toy"):
    def always(context):
        return True, "always"

    return dict(name=name, applies_to=always, requires=(), conflicts=(), mace_versions="==0.3.16",
                source_hashes={}, exactness="exact", default="auto", workload_constraints=(),
                apply=lambda model, context: model, selfcheck=lambda original, rewritten, context: None)


def test_rewrite_from_dict_does_not_require_deterministic():
    from fastmace.rewrites.registry import Rewrite

    rewrite, _ = Rewrite.from_dict(_toy_fields())
    assert rewrite.deterministic is None
    rewrite, _ = Rewrite.from_dict({**_toy_fields(), "deterministic": False})
    assert rewrite.deterministic is False
    with pytest.raises(ValueError, match="lacks"):
        Rewrite.from_dict({k: v for k, v in _toy_fields().items() if k != "exactness"})
    with pytest.raises(ValueError, match="deterministic"):
        Rewrite(**{**_toy_fields(), "deterministic": "yes"})


def test_plan_runner_field():
    from fastmace.planner.plan import Plan

    assert Plan(tier="graph").runner == "in-process"
    assert Plan(tier="graph", runner="subprocess").runner == "subprocess"
    with pytest.raises(ValueError, match="runner"):
        Plan(tier="graph", runner="thread")


# --------------------------------------------------------------------------- planner inputs

PLANNER_OPTIONS = ["mode", "allow_reference", "fallback", "gate", "runner", "engine", "neighbours"]


def test_simulation_takes_the_planner_inputs():
    from fastmace.simulation import Simulation

    signature = inspect.signature(Simulation)
    assert _keyword_only(signature)[-len(PLANNER_OPTIONS):] == PLANNER_OPTIONS
    assert signature.parameters["mode"].default == "graph"
    assert signature.parameters["allow_reference"].default is False
    for name in ("fallback", "gate", "runner", "engine", "neighbours"):
        assert signature.parameters[name].default is None


def test_md_takes_the_planner_inputs():
    from fastmace.quick import md

    signature = inspect.signature(md)
    assert set(PLANNER_OPTIONS) <= set(_keyword_only(signature))
    assert signature.parameters["mode"].default == "graph"


@pytest.mark.parametrize("kwargs, message", [
    ({"mode": "fast"}, "mode"),
    ({"allow_reference": "yes"}, "allow_reference"),
    ({"fallback": "refuse"}, "FallbackConfig"),
    ({"gate": {}}, "GateConfig"),
    ({"runner": "python"}, "RunnerConfig"),
    ({"engine": "cueq"}, "EngineConfig"),
    ({"neighbours": 0.3}, "NeighbourConfig"),
])
def test_planner_options_are_validated(kwargs, message):
    from fastmace.simulation import planner_options

    with pytest.raises((TypeError, ValueError), match=message):
        planner_options(**kwargs)


def test_planner_options_default_and_record_round_trip():
    from fastmace.simulation import PlannerOptions, planner_options

    default = planner_options()
    assert default == PlannerOptions()
    assert default.fallback == config.FallbackConfig() and default.gate == config.GateConfig()
    assert default.runner == config.RunnerConfig()
    assert default.engine is None and default.neighbours is None  # None: the default planner's
    options = planner_options(mode="eager", allow_reference=True, fallback=config.FallbackConfig(fallback="refuse"),
                              runner=config.RunnerConfig(environment={"A": "1"}),
                              engine=config.EngineConfig(backend="cueq"),
                              neighbours=config.NeighbourConfig(edge_multiple=128))
    record = options.to_record()
    import json

    assert PlannerOptions.from_record(json.loads(json.dumps(record))) == options
    assert PlannerOptions.from_record(None) == PlannerOptions()


def test_fastmace_run_reads_the_planner_options():
    from fastmace.cli.run import md_arguments

    structure, options = md_arguments({"structure": "NaCl.cif", "model": "mh1", "mode": "graph",
                                       "allow_reference": False, "engine": {"backend": "cueq"},
                                       "neighbours": {"skin": 0.4}, "fallback": {"fallback": "refuse"},
                                       "gate": {"reduced_cell": "off"}, "runner": {"environment": {"A": "1"}}})
    assert options["engine"] == config.EngineConfig(backend="cueq")
    assert options["neighbours"] == config.NeighbourConfig(skin=0.4)
    assert options["fallback"] == config.FallbackConfig(fallback="refuse")
    assert options["gate"] == config.GateConfig(reduced_cell="off")
    assert options["runner"] == config.RunnerConfig(environment={"A": "1"})
    with pytest.raises(ValueError, match="unknown engine keys"):
        md_arguments({"structure": "NaCl.cif", "model": "mh1", "engine": {"turbo": True}})


def test_md_validates_the_options_before_loading(monkeypatch):
    from fastmace import quick

    with pytest.raises(ValueError, match="mode"):
        quick.md("missing.xyz", model="no-such-model", temperature_K=300.0, steps=1, mode="fast")


# --------------------------------------------------------------------------- stubs


def _raises_naming(task, call):
    with pytest.raises(NotImplementedError, match=task):
        call()


def test_subprocess_runner_stub():
    from fastmace.engine import subprocess as runner

    signature = inspect.signature(runner.SubprocessRunner)
    assert list(signature.parameters) == ["config"]
    assert signature.parameters["config"].default is None
    run = inspect.signature(runner.SubprocessRunner.run)
    assert list(run.parameters) == ["self", "simulation", "steps", "schedule"]
    assert runner.HANDOFF_NAME == ".fastmace-handoff"
    # Implemented by 2-T07 (tests/cpu/test_subprocess_runner.py).


def test_reference_force_field_stub():
    from fastmace.engine import reference

    signature = inspect.signature(reference.ReferenceForceField)
    assert list(signature.parameters)[:2] == ["force_field", "atoms"]
    assert _keyword_only(signature) == ["device", "dtype"]
    _raises_naming("2-T03", lambda: reference.ReferenceForceField(None, None))


def test_fallback_stubs():
    from fastmace.planner import fallback

    assert list(inspect.signature(fallback.candidate).parameters) == ["plan", "changes", "force_field", "atoms"]
    assert list(inspect.signature(fallback.activate).parameters)[:1] == ["candidate"]
    _raises_naming("2-T06", lambda: fallback.candidate(None, {}, None, None))
    _raises_naming("2-T06", lambda: fallback.activate(None))
    ddmin = inspect.signature(fallback.ddmin)
    assert list(ddmin.parameters) == ["rewrites", "test", "requires", "budget"]
    _raises_naming("2-T15", lambda: fallback.ddmin(("a",), lambda subset: True, requires={}, budget=None))
    _raises_naming("2-T15", lambda: fallback.diagnose(None, None))
    _raises_naming("2-T15", lambda: fallback.terminal_sequence(None))
    _raises_naming("2-T15", lambda: fallback.Fallback(None, None, None, None, None, config.FallbackConfig()).run(None))


def test_fallback_records():
    from fastmace.planner import fallback

    budget = fallback.Budget(limit=64)
    assert (budget.reduced_gate_runs, budget.full_gate_runs, budget.spent) == (0, 0, False)
    tried = fallback.TriedCandidate(dimension="mode", change="eager", outcome="failed", reason="replay failed",
                                    stage="gate")
    assert tried.revision is None
    with pytest.raises(ValueError):
        fallback.TriedCandidate(dimension="precision", change="x", outcome="failed", reason="r", stage="gate")
    with pytest.raises(ValueError):
        fallback.TriedCandidate(dimension="mode", change="x", outcome="maybe", reason="r", stage="gate")


def test_reproducible_stubs():
    from fastmace.planner import reproducible

    entry = reproducible.AllowListEntry(component="ChannelWiseTensorProduct (unfused)", kind="cueq-kernel",
                                        deterministic=True, reason="summed by scatter_add_ under the flag")
    _frozen(entry, "deterministic", False)
    with pytest.raises(ValueError):
        reproducible.AllowListEntry(component="x", kind="other", deterministic=True, reason="")
    qualification = reproducible.Qualification(backend="cueq", family="mh1", qualified=False,
                                               blocking=["edge_outer_sum (segmented)"])
    assert qualification.blocking == ("edge_outer_sum (segmented)",) and qualification.entries == ()
    assert reproducible.ALLOW_LIST == ()
    _raises_naming("2-T16", lambda: reproducible.qualify(None))


def test_stub_modules_are_torch_free():
    import subprocess

    code = ("import sys; import fastmace.engine.subprocess, fastmace.engine.reference, "
            "fastmace.planner.fallback, fastmace.planner.reproducible; "
            "assert 'torch' not in sys.modules, 'torch imported'")
    subprocess.run([sys.executable, "-c", code], check=True)
