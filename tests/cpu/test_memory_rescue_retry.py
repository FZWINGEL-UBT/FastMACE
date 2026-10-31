"""CAP-T07: an out-of-memory error at build or capture rebuilds once with the memory rescue's records on.

The owner's rule (plans/capacity.md, decision 67): a record that trades step time for memory runs only when memory
would not be enough without it. The estimate overstates the peak by up to ``memory.RESCUE_OVERSTATEMENT``, so a run
whose estimate does not fit tries without ``recompute_conv_weights`` first; ``fallback.memory_rescue`` turns it on
after a real out-of-memory error, if the CUDA context is still usable, and otherwise raises the error with the fix.

Faults are injected here only (a monkeypatched build, capture and rescue), never through a hook in the library.
"""

from __future__ import annotations

import gc
import types
import weakref

import pytest
import test_minimal_planner as tmp
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace import integrators
from fastmace.config import RewriteConfig
from fastmace.planner import fallback, minimal

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

NA_CL = (11, 17)
LANGEVIN = integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01)
REWRITES = RewriteConfig(factor_species="on", block_linear="on")
quiet_licences = tmp.quiet_licences


def oom(text="CUDA out of memory. Tried to allocate 2.00 GiB (injected)"):
    return torch.cuda.OutOfMemoryError(text)


# ------------------------------------------------------------------ fallback.memory_rescue


def stub_plan(**rewrites):
    values = {"recompute_conv_weights": "off", "wprime_backward": "on", **rewrites}
    return types.SimpleNamespace(rewrites=types.SimpleNamespace(**values),
                                 engine=types.SimpleNamespace(device="cuda:0"), decisions=())


@pytest.fixture
def stubbed(monkeypatch):
    """The registry's record (its requirements) and ``candidate``, recorded instead of re-planned."""
    calls = []
    monkeypatch.setattr(fallback, "_record", lambda name: types.SimpleNamespace(requires=("wprime_backward",)))

    def candidate(plan, changes, force_field, atoms, *, planner=None, **kwargs):
        calls.append(changes)
        return ("revised", changes)

    monkeypatch.setattr(fallback, "candidate", candidate)
    return calls


def test_a_recoverable_out_of_memory_error_turns_the_rescue_on(stubbed):
    ff = types.SimpleNamespace(rewrites="auto")
    revised = fallback.memory_rescue(stub_plan(), oom(), ff, None, planner=None, stage="capture",
                                     usable=lambda device: True)
    assert revised[0] == "revised"
    assert stubbed == [{"rewrites": {"recompute_conv_weights": "on"},
                        "reason": "out of memory at capture; recompute_conv_weights auto -> on (the memory rescue, "
                                  "rule 12; CAP-T07)"}]


def test_the_rescue_turns_on_one_record_per_rebuild_in_its_order(monkeypatch, stubbed):
    """CAP-T08: recompute_node_segments only after recompute_conv_weights is on, recompute_node_region only after
    recompute_node_segments is on (ninth list item 2), one record per rebuild."""
    requires = {"recompute_conv_weights": ("wprime_backward",), "recompute_node_segments": ("gated_rows",),
                "recompute_node_region": ("recompute_node_segments",)}
    monkeypatch.setattr(fallback, "_record", lambda name: types.SimpleNamespace(requires=requires[name]))
    ff = types.SimpleNamespace(rewrites="auto")
    steps = ((("off", "off"), "recompute_conv_weights"), (("on", "off"), "recompute_node_segments"),
             (("on", "on"), "recompute_node_region"))
    for (conv, nodes), expected in steps:
        plan = stub_plan(recompute_conv_weights=conv, recompute_node_segments=nodes, recompute_node_region="off",
                         gated_rows="on")
        fallback.memory_rescue(plan, oom(), ff, None, planner=None, stage="build", usable=lambda device: True)
        assert stubbed[-1]["rewrites"] == {expected: "on"}
    stubbed.clear()
    plan = stub_plan(recompute_conv_weights="on", recompute_node_segments="on", recompute_node_region="on",
                     gated_rows="on")
    assert fallback.memory_rescue(plan, oom(), ff, None, planner=None, stage="build",
                                  usable=lambda device: True) is None


class AcceleratorError(RuntimeError):
    """As ``torch.AcceleratorError``: the driver's error, not the caching allocator's."""


DRIVER_OOM = ("CUDA error: out of memory\nThe CUDA driver logged these messages, which may provide useful details:\n"
              "Sticky error detected")


def test_the_driver_s_out_of_memory_error_is_never_a_fallback_trigger():
    """CAP-T07, NbCl5 36,720 atoms on WSL2: a sticky driver error went into the fallback search, whose re-plan then
    failed in cudaMemGetInfo."""
    assert fallback._out_of_memory(AcceleratorError(DRIVER_OOM)) and fallback._propagates(AcceleratorError(DRIVER_OOM))
    assert fallback._propagates(oom())
    assert not fallback._propagates(AcceleratorError("CUDA error: an illegal memory access was encountered"))


@pytest.mark.parametrize("error", [oom(), AcceleratorError(DRIVER_OOM)], ids=["allocator", "driver"])
def test_a_sticky_context_raises_the_error_with_the_fix(stubbed, error):
    with pytest.raises(type(error)) as raised:
        fallback.memory_rescue(stub_plan(), error, types.SimpleNamespace(rewrites="auto"), None, planner=None,
                               stage="capture", usable=lambda device: False)
    assert raised.value is error
    text = str(raised.value)
    assert "out of memory" in text.lower() and "not usable" in text
    assert 'RewriteConfig(recompute_conv_weights="on")' in text
    assert stubbed == []


@pytest.mark.parametrize("case", ["not out of memory", "explicit off", "already on", "requirement off"])
def test_no_rescue(stubbed, case):
    error, ff, plan = oom(), types.SimpleNamespace(rewrites="auto"), stub_plan()
    if case == "not out of memory":
        error = RuntimeError("CUDA error: an illegal memory access was encountered")
    elif case == "explicit off":
        ff = types.SimpleNamespace(rewrites=RewriteConfig(recompute_conv_weights="off"))
    elif case == "already on":
        plan = stub_plan(recompute_conv_weights="on")
    else:
        plan = stub_plan(wprime_backward="off")
    assert fallback.memory_rescue(plan, error, ff, None, planner=None, stage="build",
                                  usable=lambda device: True) is None
    assert stubbed == []


@pytest.mark.parametrize("memory", ["warn", "fits"])
def test_an_unusable_context_raises_the_failure_instead_of_searching(memory):
    """CAP-T07, NbCl5 46,656 atoms on WSL2 without blocking launches: the driver's out-of-memory error surfaced as a
    cuBLAS failure, which went into the fallback search and crashed there in cudaMemGetInfo."""
    error = RuntimeError("CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgemm(...)`")
    plan = stub_plan()
    plan.decision = lambda key: ("memory", memory, "rule 12: ...")
    failure = types.SimpleNamespace(plan=plan, stage="build", error=error)
    fallback.raise_if_context_unusable(failure, None, usable=lambda device: True)  # usable: the search runs
    with pytest.raises(RuntimeError) as raised:
        fallback.raise_if_context_unusable(failure, None, usable=lambda device: False)
    assert raised.value is error
    text = str(raised.value)
    assert text.startswith("CUDA error: CUBLAS_STATUS_EXECUTION_FAILED") and "not usable" in text
    assert ('RewriteConfig(recompute_conv_weights="on")' in text) == (memory == "warn")


def test_context_usable(monkeypatch):
    assert fallback._context_usable("cpu")

    def sticky(*args, **kwargs):
        raise RuntimeError("CUDA error: device not ready")

    monkeypatch.setattr(torch.cuda, "synchronize", sticky)
    assert not fallback._context_usable("cuda:0")  # never a crash: the rescue raises the error with its fix


# ------------------------------------------------------------------ Simulation: the rebuild


@pytest.fixture
def pristine(tmp_path, quiet_licences):
    from fastmace.model import loading

    model = tiny_mace_model(list(NA_CL), **FAMILIES["mh1"])
    path = tmp_path / "tiny-mh1.model"
    torch.save(model, path)
    return loading.load_pristine(str(path), trusted=True)


class Injected:
    """The first engine fails at ``stage`` with an out-of-memory error; weakrefs to every built engine."""

    def __init__(self, monkeypatch, stage, *, sticky=False, failures=1):
        self.refs, self.alive_at_build, self.stages = [], [], []
        original = minimal.MinimalPlanner.build
        probe = self

        def build(planner, plan, force_field, atoms, **kwargs):
            gc.collect()
            probe.alive_at_build.append(sum(ref() is not None for ref in probe.refs))
            first = len(probe.refs) < failures
            if first and stage == "build":
                probe.refs.append(lambda: None)
                raise oom()
            engine = original(planner, plan, force_field, atoms, **kwargs)
            probe.refs.append(weakref.ref(engine))
            if first and stage == "capture":
                def prepare():
                    raise oom()

                engine.prepare = prepare
            return engine

        def rescue(plan, error, force_field, atoms, *, planner, stage):
            probe.stages.append(stage)
            if sticky:
                raise error
            turned = ("block_linear", "grouped_linear")[len(probe.stages) - 1]
            return fallback.candidate(plan, {"rewrites": {turned: "off"},
                                             "reason": f"out of memory at {stage} (injected)"},
                                      force_field, atoms, planner=planner)

        monkeypatch.setattr(minimal.MinimalPlanner, "build", build)
        monkeypatch.setattr(fallback, "memory_rescue", rescue)


@pytest.mark.parametrize("stage", ["build", "capture"])
def test_simulation_rebuilds_once_after_a_recoverable_out_of_memory_error(pristine, monkeypatch, stage):
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    probe = Injected(monkeypatch, stage)
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    sim = Simulation(atoms, ForceField(pristine, atoms, precision="float64", rewrites=REWRITES),
                     integrator=LANGEVIN, seed=8, check="skip")
    assert probe.stages == [stage]
    assert len(probe.refs) == 2 and probe.alive_at_build == [0, 0]  # the failed engine is gone before the rebuild
    assert sim.plan.revision == 1 and "block_linear" not in fallback.enabled_rewrites(sim.plan)
    assert f"out of memory at {stage} (injected)" in sim.plan.reason  # the run record keeps which path ran
    sim.run(steps=2)
    sim.close()


def test_simulation_rebuilds_again_while_the_rescue_has_a_record_left(pristine, monkeypatch):
    """CAP-T08: a rebuild that runs out of memory again turns on the rescue's next record."""
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    probe = Injected(monkeypatch, "build", failures=2)
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    sim = Simulation(atoms, ForceField(pristine, atoms, precision="float64", rewrites=REWRITES),
                     integrator=LANGEVIN, seed=8, check="skip")
    assert probe.stages == ["build", "build"] and len(probe.refs) == 3
    assert sim.plan.revision == 2
    sim.run(steps=1)
    sim.close()


def test_simulation_raises_the_error_when_the_context_is_sticky(pristine, monkeypatch):
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    probe = Injected(monkeypatch, "capture", sticky=True)
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        Simulation(atoms, ForceField(pristine, atoms, precision="float64", rewrites=REWRITES),
                   integrator=LANGEVIN, seed=8, check="skip")
    assert probe.stages == ["capture"] and len(probe.refs) == 1
