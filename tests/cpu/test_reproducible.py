"""``reproducible=True`` for plain cuEq MH-1 (plan task 2-T16; SPEC §9.3 rule 5, §8.7, decisions 13, 14).

- Every rewrite record declares ``deterministic``: the streaming convolutions (atomic sums) and the
  segmented first sum (relaxed atomics in its float32 Triton forward) are False, the rest True.
- A reproducible MH-1 plan is plain cuEq with the unfused convolution, every hybrid-only rewrite
  off, runs in this process while CUDA is not initialised, and records its qualification: the
  pinned ``cuequivariance_ops`` version, the process environment and the undocumented cuEq
  internal it relies on.
- Rule 5 refuses (``PlanRefused``, rule 5) an explicit hybrid backend or fused convolution, another
  model family, another ``cuequivariance_ops`` version, and a record in use whose
  ``deterministic`` is not True (``None`` included), naming it.
- ``prepare_process`` sets the three variables and the deterministic flags together, and refuses
  after CUDA started; the process record then makes a later non-reproducible plan run in a child
  (decision 14), and the subprocess runner hands the variables to its children.
- ``check_repeatable`` raises ``ReproducibilityError`` naming the kernels when two evaluations differ.
- The default path is untouched: no variable is set and no reproducible decision is made.
"""

import dataclasses
import os

import pytest
import test_minimal_planner as tmp
import torch

from fastmace.config import EngineConfig, RewriteConfig
from fastmace.engine import constraints
from fastmace.errors import PlanRefused, ProcessConstraintError, ReproducibilityError
from fastmace.planner import minimal as minimal_module
from fastmace.planner import reproducible
from fastmace.planner.workload import Workload
from fastmace.rewrites import engine as engine_records
from fastmace.rewrites import registry as reg

REPRODUCIBLE = Workload(max_temperature_K=300.0, reproducible=True)
PINNED = {reproducible.CUEQ_OPS: reproducible.PINNED_CUEQ_OPS}
NONDETERMINISTIC = {"streaming_conv", "chunked_conv_backward", "fused_folded_forward", "generic_streaming",
                    "segmented_first_sum", "generic_segmented_first_sum"}


@pytest.fixture(autouse=True)
def _cuda_not_initialised(monkeypatch):
    monkeypatch.setattr(minimal_module, "_cuda_initialised", lambda: False)


def planner(versions=PINNED, process=None, **configs):
    return tmp.minimal(process=process or constraints.ProcessRecord(), versions=versions, **configs)


def plan(label="mh1-r2scan", workload=REPRODUCIBLE, *, precision="float64", **kwargs):
    kwargs.setdefault("planner", planner())
    return tmp.plan_label(label, workload=workload, precision=precision, **kwargs)


def refusal(**kwargs):
    with pytest.raises(PlanRefused) as caught:
        plan(**kwargs)
    return caught.value


# --------------------------------------------------------------------------- record declarations

def test_every_rewrite_record_declares_deterministic():
    records = {rewrite.name: rewrite for rewrite in reg.default_registry()}
    records.update(engine_records.PROCESS_RECORDS)
    undeclared = sorted(name for name, rewrite in records.items() if rewrite.deterministic is None)
    assert undeclared == []
    assert {name for name, rewrite in records.items() if rewrite.deterministic is False} == NONDETERMINISTIC


# --------------------------------------------------------------------------- the plan's composition

@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_reproducible_mh1_plan_is_plain_unfused_cueq(precision):
    got = plan(precision=precision)
    assert got.engine.backend == "cueq" and got.engine.cueq_conv_fusion == "off"
    assert got.precision.dtype == precision
    assert got.runner == "in-process"
    assert got.decision("EngineConfig.backend")[2].code == "backend-cueq-reproducible"
    assert "reproducible" in str(got.decision("EngineConfig.cueq_conv_fusion")[2])
    for name in ("factor_species", "streaming_conv", "segmented_first_sum", "reassociate_first", "block_linear",
                 "gated_rows", "segment_gemm"):
        assert getattr(got.rewrites, name) == "off", name
    key, choice, reason = got.decision("reproducible")
    assert choice.startswith("qualified") and reason.code == "reproducible-qualified"
    assert reproducible.PINNED_CUEQ_OPS in reason and "undocumented" in reason
    environment = dict(item.split("=", 1) for item in got.decision("reproducible.environment")[1].split())
    assert environment == dict(reproducible.PROCESS_ENVIRONMENT)


def test_reproducible_plan_needs_no_requested_entry():
    """The plan, not the caller, switched the backend and the fusion: nothing shows as requested."""
    requested = dict(plan().requested)
    assert "EngineConfig.backend" not in requested and "EngineConfig.cueq_conv_fusion" not in requested


def test_qualify_lists_the_allow_list():
    got = plan()
    qualification = reproducible.qualify(got, tmp.tp.traits("mh1-r2scan"), versions=PINNED)
    assert qualification.qualified and qualification.blocking == ()
    assert (qualification.backend, qualification.family) == ("cueq", reproducible.FAMILY)
    components = {entry.component for entry in qualification.entries}
    assert {"ChannelWiseTensorProduct (unfused, uniform_1d)", "SymmetricContraction (uniform_1d)",
            "scatter_add_ / index_add / index_select backward"} <= components
    assert all(entry.deterministic is True for entry in qualification.entries)


# --------------------------------------------------------------------------- rule 5 refusals

@pytest.mark.parametrize(("engine", "code"), [
    (EngineConfig(backend="hybrid"), "reproducible-backend"),
    (EngineConfig(backend="cueq", cueq_conv_fusion="on"), "reproducible-conv-fusion"),
])
def test_explicit_conflicts_refuse_rule_5(engine, code):
    error = refusal(planner=planner(engine=engine))
    assert code in tmp.codes(error)
    assert all(reason.rule == 5 for reason in error.reasons if reason.code == code)


@pytest.mark.parametrize("label", ["mp0-medium", "omol0-xl", "off23-small"])  # PolarMACE: rule 3 refuses first
def test_other_families_are_not_qualified(label):
    error = refusal(label=label)
    reason = next(reason for reason in error.reasons if reason.code == "reproducible-family-not-qualified")
    assert reason.rule == 5 and "MH-1" in reason


@pytest.mark.parametrize("version", ["0.13.0", None])
def test_other_cueq_ops_versions_are_not_qualified(version):
    error = refusal(planner=planner(versions={reproducible.CUEQ_OPS: version}))
    reason = next(reason for reason in error.reasons if reason.code == "reproducible-cueq-version")
    assert "uniform_1d" in reason and reproducible.PINNED_CUEQ_OPS in reason


@pytest.mark.parametrize("value", [None, False])
def test_a_record_in_use_that_is_not_deterministic_refuses_naming_it(value):
    registry = reg.default_registry()
    registry._records["single_graph_strain"] = dataclasses.replace(registry["single_graph_strain"],
                                                                   deterministic=value)
    error = refusal(planner=planner(registry=registry))
    reason = next(reason for reason in error.reasons if reason.code == "reproducible-kernel-not-deterministic")
    assert reason.rule == 5 and "single_graph_strain" in reason
    assert ("undeclared" if value is None else "not deterministic") in reason


def test_a_record_switched_off_does_not_block():
    registry = reg.default_registry()
    registry._records["single_graph_strain"] = dataclasses.replace(registry["single_graph_strain"],
                                                                   deterministic=None)
    got = plan(planner=planner(registry=registry), rewrites=RewriteConfig(single_graph_strain="off"))
    assert got.decision("reproducible")[2].code == "reproducible-qualified"


def test_the_reference_tier_is_not_qualified():
    """2-T03's reference tier runs MACE's own calculator, which the allow-list does not cover."""
    error = refusal(mode="reference", allow_reference=True)
    reason = next(reason for reason in error.reasons if reason.code == "reproducible-tier")
    assert reason.rule == 5 and "reference" in reason


def test_an_explicit_nondeterministic_rewrite_refuses():
    error = refusal(rewrites=RewriteConfig(streaming_conv="on"))
    assert {"reproducible-backend", "rewrite-requirement-missing"} & set(tmp.codes(error))


# --------------------------------------------------------------------------- process state

class _FakeTorch:
    def __init__(self, initialised=False):
        self.flag = None
        self.initialised = initialised
        self.cuda = self
        self._inductor = type("I", (), {"config": type("C", (), {"deterministic": False})()})()

    def is_initialized(self):
        return self.initialised

    def use_deterministic_algorithms(self, flag):
        self.flag = flag


def test_prepare_process_sets_everything_together():
    environ, fake, record = {}, _FakeTorch(), constraints.ProcessRecord()
    reproducible.prepare_process(record, environ=environ, torch_module=fake)
    assert environ == dict(reproducible.PROCESS_ENVIRONMENT)
    assert fake.flag is True and fake._inductor.config.deterministic is True
    assert record.reproducible is True
    reproducible.prepare_process(record, environ=environ, torch_module=fake)  # idempotent


def test_prepare_process_refuses_after_cuda_started():
    environ, record = {}, constraints.ProcessRecord()
    with pytest.raises(ProcessConstraintError, match="CUDA"):
        reproducible.prepare_process(record, environ=environ, torch_module=_FakeTorch(initialised=True))
    assert environ == {} and record.reproducible is None


def test_prepare_process_refuses_a_process_that_is_not_reproducible():
    record = constraints.ProcessRecord()
    record.claim_reproducible(False, "a default plan")
    with pytest.raises(ProcessConstraintError, match="not reproducible"):
        reproducible.prepare_process(record, environ={}, torch_module=_FakeTorch())


def test_a_default_plan_in_a_reproducible_process_runs_in_a_child():
    """Decision 14: a process is reproducible or not; the other kind runs in a child (2-T07)."""
    record = constraints.ProcessRecord()
    record.claim_reproducible(True, "a reproducible plan")
    got = plan(workload=tmp.WORKLOAD, planner=planner(process=record))
    assert got.runner == "subprocess"
    assert got.decision("runner")[2].code == "runner-subprocess-not-reproducible"


def test_a_reproducible_plan_in_a_default_process_runs_in_a_child():
    record = constraints.ProcessRecord()
    record.claim_reproducible(False, "a default plan")
    got = plan(planner=planner(process=record))
    assert got.runner == "subprocess"
    assert got.decision("runner")[2].code == "runner-subprocess-reproducible"


def test_reproducible_after_cuda_started_runs_in_a_child(monkeypatch):
    monkeypatch.setattr(minimal_module, "_cuda_initialised", lambda: True)
    got = plan()
    assert got.runner == "subprocess"
    assert got.decision("runner")[2].code == "runner-subprocess-reproducible"
    assert got.decision("reproducible")[2].code == "reproducible-qualified"  # qualified, then handed on


def test_a_reproducible_process_keeps_its_plans_after_cuda_started(monkeypatch):
    """The process was prepared before CUDA started (``plan`` does, before the hardware detection
    initialises CUDA): its reproducible plans stay here."""
    monkeypatch.setattr(minimal_module, "_cuda_initialised", lambda: True)
    record = constraints.ProcessRecord()
    record.claim_reproducible(True, "a reproducible plan")
    assert plan(planner=planner(process=record)).runner == "in-process"


def test_plan_prepares_the_process_before_the_hardware_detection(monkeypatch):
    """``MinimalPlanner.plan`` prepares the process before ``hardware()``, which starts CUDA."""
    calls = []
    record = constraints.ProcessRecord()
    monkeypatch.setattr(reproducible, "prepare_process",
                        lambda process, **kw: (calls.append("prepare"), process.claim_reproducible(True, "test")))
    chosen = planner(process=record)
    monkeypatch.setattr(chosen, "hardware", lambda: calls.append("hardware") or tmp.GPU)

    class _FF:
        traits, precision, rewrites, model = tmp.tp.traits("mh1-r2scan"), "float64", "auto", None

        @staticmethod
        def superset_for(atoms):
            return None

    got = chosen.plan(_FF(), tmp.nacl(), tmp.IntegratorConfig(), REPRODUCIBLE)
    assert calls[:2] == ["prepare", "hardware"] and got.runner == "in-process"


def test_the_runner_hands_the_variables_to_its_children():
    from fastmace.config import RunnerConfig
    from fastmace.engine.subprocess import SubprocessRunner

    runner = SubprocessRunner(RunnerConfig(inherit_environment=False))
    runner._reproducible = True
    _, environment = runner._command("stage")
    for name, value in reproducible.PROCESS_ENVIRONMENT.items():
        assert environment[name] == value
    runner._reproducible = False
    _, environment = runner._command("stage")
    assert not set(reproducible.PROCESS_ENVIRONMENT) & set(environment)


# --------------------------------------------------------------------------- the plan-time check

def test_check_repeatable_passes_bitwise_equal_evaluations():
    values = iter([torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0])])
    reproducible.check_repeatable(lambda: (next(values),), kernels=("uniform_1d",))


def test_check_repeatable_names_the_kernels_when_evaluations_differ():
    values = iter([torch.tensor([1.0, 2.0], dtype=torch.float64),
                   torch.tensor([1.0, 2.0 + 5e-16], dtype=torch.float64)])
    with pytest.raises(ReproducibilityError) as caught:
        reproducible.check_repeatable(lambda: (next(values),), kernels=("uniform_1d",))
    assert caught.value.blocking == ("uniform_1d",)


# --------------------------------------------------------------------------- the default path

def test_the_default_plan_makes_no_reproducible_decision_and_sets_nothing():
    before = {name: os.environ.get(name) for name in reproducible.PROCESS_ENVIRONMENT}
    record = constraints.ProcessRecord()
    got = plan(workload=tmp.WORKLOAD, precision="float32", planner=planner(process=record))
    assert not {"reproducible", "reproducible.environment"} & {entry[0] for entry in got.decisions}
    assert got.runner == "in-process" and got.engine.backend == "hybrid"
    assert record.reproducible is None  # planning alone claims nothing
    assert {name: os.environ.get(name) for name in reproducible.PROCESS_ENVIRONMENT} == before


# --------------------------------------------------------------------------- end to end on the CPU

DRIVER = r"""
import json, sys
sys.path.insert(0, {tests!r})
import fastmace  # noqa: F401  (before torch)
import numpy as np
import torch
from _support.structures import nacl
from fastmace import integrators
from fastmace.config import EngineConfig
from fastmace.model import loading
from fastmace.model.forcefield import ForceField
from fastmace.planner.workload import Workload
from fastmace.simulation import Simulation

model, action, directory, steps, save = {model!r}, {action!r}, {directory!r}, {steps!r}, {save!r}
if action == "resume":
    sim = Simulation.resume(directory)
else:
    loading._LICENCES_SHOWN.add(model)
    loaded = loading.load_pristine(model, trusted=True)
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    sim = Simulation(atoms, ForceField(loaded, atoms, precision="float64"),
                     integrator=integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01),
                     seed=11, engine=EngineConfig(device="cpu", compile_mode="none"),
                     workload=Workload(max_temperature_K=300.0, reproducible=True))
for index, count in enumerate(steps):
    sim.run(steps=count)
    if save and index == 0:
        sim.save_restart(directory)  # the restart cadence: one write after the first run
committed = sim.committed()
print("RESULT " + json.dumps({{
    "step": committed.step, "positions": committed.positions.tobytes().hex(),
    "momenta": committed.momenta.tobytes().hex(), "runner": sim.plan.runner,
    "reproducible": [entry[1] for entry in sim.plan.decisions if entry[0] == "reproducible"],
    "flag": torch.are_deterministic_algorithms_enabled(),
    "environment": {{name: __import__("os").environ.get(name) for name in {names!r}}}}}))
"""


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    pytest.importorskip("cuequivariance_torch")
    from _support.tiny_models import FAMILIES, tiny_mace_model

    model = tiny_mace_model([11, 17], **FAMILIES["mh1"])
    path = tmp_path_factory.mktemp("models") / "tiny-mh1-11-17.model"
    torch.save(model, path)
    return str(path)


def _fresh(tmp_path, model, action, directory, steps, save):
    import json
    import pathlib
    import subprocess
    import sys

    here = pathlib.Path(__file__).resolve()
    tests, root = str(here.parents[1]), str(here.parents[2])  # _support, and this checkout's fastmace
    code = DRIVER.format(tests=tests, model=model, action=action, directory=str(directory), steps=steps, save=save,
                         names=tuple(reproducible.PROCESS_ENVIRONMENT))
    environment = {key: value for key, value in os.environ.items() if key not in reproducible.PROCESS_ENVIRONMENT}
    environment.update(FASTMACE_CACHE=str(tmp_path / "cache"), CUDA_VISIBLE_DEVICES="",
                       PYTHONPATH=os.pathsep.join([root, *filter(None, [environment.get("PYTHONPATH")])]))
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=environment,
                          timeout=600)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    line = next(line for line in done.stdout.splitlines() if line.startswith("RESULT "))
    return json.loads(line[len("RESULT "):])


def test_repeated_and_interrupted_runs_are_bitwise_equal_on_the_cpu(tmp_path, tiny_model):
    """CPU float64, tiny MH-1: two fresh runs of 8 steps (restart write at 4), and the first run's
    restart resumed in a fresh process for the last 4, end bitwise equal; each process is reproducible."""
    first = _fresh(tmp_path, tiny_model, "new", tmp_path / "a", (4, 4), True)
    second = _fresh(tmp_path, tiny_model, "new", tmp_path / "b", (4, 4), True)
    resumed = _fresh(tmp_path, tiny_model, "resume", tmp_path / "a", (4,), False)
    for run in (first, second, resumed):
        assert run["step"] == 8 and run["runner"] == "in-process" and run["flag"] is True
        assert run["environment"] == dict(reproducible.PROCESS_ENVIRONMENT)
        assert run["reproducible"] and run["reproducible"][0].startswith("qualified")
    for run in (second, resumed):
        assert (run["positions"], run["momenta"]) == (first["positions"], first["momenta"])
