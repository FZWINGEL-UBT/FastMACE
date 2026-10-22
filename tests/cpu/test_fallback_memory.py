"""A failed candidate's engine is freed before the next candidate is built (plan task 2-T15c; SPEC §9.4).

The fallback keeps every failure's exception for its record. A traceback holds the frames it passed
through, and those frames hold the failed engine (the gate's ``production(engine, ...)``, the
caller's ``engine`` local), so a kept exception kept the engine, and on CUDA its allocations, alive
while the next candidate was built. Now the fallback keeps the type, message, stage and the formatted
traceback text, and drops the frames of the error and its chain.

A weakref per built engine; at every build, every engine built before must already be collected.
Faults are injected here only (a monkeypatched build and gate), never through a hook in the library.
"""

from __future__ import annotations

import gc
import weakref

import pytest
import test_minimal_planner as tmp
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace import integrators
from fastmace.config import IntegratorConfig, RewriteConfig
from fastmace.errors import GateFailed
from fastmace.planner import fallback, minimal, selfcheck

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

NA_CL = (11, 17)
LANGEVIN = integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01)
REWRITES = RewriteConfig(factor_species="on", block_linear="on")


@pytest.fixture
def pristine(tmp_path, quiet_licences):
    from fastmace.model import loading

    model = tiny_mace_model(list(NA_CL), **FAMILIES["mh1"])
    path = tmp_path / "tiny-mh1.model"
    torch.save(model, path)
    return loading.load_pristine(str(path), trusted=True)


quiet_licences = tmp.quiet_licences


class Probe:
    """Weakrefs to every built engine; ``alive_at_build[i]``: engines still alive when build ``i`` started."""

    def __init__(self, monkeypatch, *, fails):
        self.refs, self.alive_at_build, self.tiers = [], [], []
        original = minimal.MinimalPlanner.build
        probe = self

        def build(planner, plan, force_field, atoms, **kwargs):
            gc.collect()
            probe.alive_at_build.append(sum(ref() is not None for ref in probe.refs))
            probe.tiers.append(plan.tier)
            engine = original(planner, plan, force_field, atoms, **kwargs)
            engine._probe_plan = plan
            probe.refs.append(weakref.ref(engine))
            return engine

        def production(engine, references, *, check, quantities=None):
            if fails(engine._probe_plan):
                record = selfcheck.GateRecord(status="failed", check=check, reason="force error (injected)",
                                              quantities={"force_eV_per_A": {"passed": False}})
                raise GateFailed("the gate on this structure failed (injected)", gate_record=record)
            return selfcheck.GateRecord(status="passed", check=check, reason=None)

        monkeypatch.setattr(minimal.MinimalPlanner, "build", build)
        monkeypatch.setattr(selfcheck, "production", production)


def _broken(field):
    return lambda plan: field in fallback.enabled_rewrites(plan)


def test_the_failed_engine_is_collected_before_each_next_build_in_simulation(pristine, monkeypatch):
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    probe = Probe(monkeypatch, fails=_broken("block_linear"))
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    references = selfcheck.References(check="gate", copies=[object()] * 2, values=[{}] * 2,
                                      reference_copies=[object()] * 2, ran="in-process")
    monkeypatch.setattr(selfcheck, "references", lambda *args, **kwargs: references)
    sim = Simulation(atoms, ForceField(pristine, atoms, precision="float64", rewrites=REWRITES),
                     integrator=LANGEVIN, seed=8, check="gate")
    assert sim.plan.revision == 1 and "block_linear" not in fallback.enabled_rewrites(sim.plan)
    assert len(probe.refs) >= 3  # the failed build, the diagnosis runs, the activated candidate
    assert probe.alive_at_build == [0] * len(probe.refs), probe.alive_at_build
    gc.collect()
    assert [ref() is not None for ref in probe.refs] == [False] * (len(probe.refs) - 1) + [True]
    sim.close()


def test_every_failed_candidate_is_collected_before_the_next_build(pristine, monkeypatch):
    """Diagnosis runs and validations that fail inside the search (not only the first failure): the
    graph and compiled tiers fail too, so the mode dimension's compiled candidate (3-T06) and ddmin's
    graph candidate fail their validations and eager follows."""
    probe = Probe(monkeypatch, fails=lambda plan: plan.tier in ("graph", "compiled")
                  or _broken("grouped_linear")(plan))
    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    from fastmace.model.forcefield import ForceField

    ff = ForceField(pristine, atoms, precision="float64", rewrites=REWRITES)
    planner = tmp.cpu_minimal()
    import test_planner as tp

    plan = planner.plan(ff, atoms, IntegratorConfig(temperature_K=300.0), tmp.WORKLOAD, tp.CPU)
    record = selfcheck.GateRecord(status="failed", check="gate", reason="force error",
                                  quantities={"force_eV_per_A": {"passed": False}})
    references = selfcheck.References(check="gate", copies=[object()] * 2, values=[{}] * 2,
                                      reference_copies=[object()] * 2, ran="in-process")
    failure = fallback.Failure(plan=plan, stage="gate", error=GateFailed("failed", gate_record=record), gate=record)
    result = fallback.Fallback(planner, ff, atoms, IntegratorConfig(temperature_K=300.0), references).run(failure)
    assert result.plan.tier == "eager"
    assert any(t.outcome == "failed" and t.full_gate_runs for t in result.plan.fallbacks)  # a failed validation
    assert probe.alive_at_build == [0] * len(probe.refs), probe.alive_at_build


def test_detached_errors_keep_their_diagnostics_without_frames():
    def inner(engine):
        raise RuntimeError("capture of stage B failed")

    def outer():
        engine = object()
        try:
            inner(engine)
        except RuntimeError as error:
            raise ValueError("the build failed") from error

    try:
        outer()
    except ValueError as caught:
        error = caught
    detached = fallback.detach(error)
    assert detached is error and str(error) == "the build failed" and type(error) is ValueError
    assert error.__traceback__ is None and error.__cause__.__traceback__ is None
    text = error.fastmace_traceback
    assert "capture of stage B failed" in text and "in inner" in text and "the build failed" in text
    assert fallback.detach(error) is error and error.fastmace_traceback == text  # idempotent
