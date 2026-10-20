"""The fallback in ``Simulation`` (SPEC §9.4, §9.5, decision 29; plan task 2-T15), on a tiny CPU model.

- no failure: the fallback is never entered, the plan stays revision 0 (acceptance 8's "no fallback
  code is reached");
- ``fallback="refuse"``: a failure raises as in 1b, with its record;
- a broken capture (any check mode): eager passes, the graph mode is dropped, and the report shows
  ``plan revision 1 (fallback): mode eager, …``;
- a deliberately broken rewrite and a broken pair, through the real gate (``check="gate"``): ddmin
  finds them, and they and their dependants are off;
- a resume of a fallback revision rebuilds that revision, without a new fallback.

Faults are injected here only (monkeypatched builds and registry applications), never through a hook
in the library.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import test_minimal_planner as tmp
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace import integrators
from fastmace.config import FallbackConfig, RewriteConfig
from fastmace.errors import GateFailed
from fastmace.planner import fallback, minimal, selfcheck

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

NA_CL = (11, 17)
LANGEVIN = integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01)
REWRITES = RewriteConfig(factor_species="on", block_linear="on")
CAPTURE_ERROR = "CUDA graph capture of stage B failed (injected)"


@pytest.fixture
def pristine(tmp_path, quiet_licences):
    from fastmace.model import loading

    model = tiny_mace_model(list(NA_CL), **FAMILIES["mh1"])
    path = tmp_path / "tiny-mh1.model"
    torch.save(model, path)
    return loading.load_pristine(str(path), trusted=True)


quiet_licences = tmp.quiet_licences


def simulation(pristine, **kwargs):
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    atoms = nacl(repeat=1, rattle=0.05, seed=3)
    ff = ForceField(pristine, atoms, precision="float64", rewrites=kwargs.pop("rewrites", REWRITES))
    return Simulation(atoms, ff, integrator=LANGEVIN, seed=8, **kwargs)


def broken_capture(monkeypatch):
    """The graph tier's build raises a capture error; eager builds."""
    original = minimal.MinimalPlanner.build
    built = []

    def build(planner, plan, force_field, atoms, **kwargs):
        built.append(plan.tier)
        if plan.tier == "graph":
            raise RuntimeError(CAPTURE_ERROR)
        return original(planner, plan, force_field, atoms, **kwargs)

    monkeypatch.setattr(minimal.MinimalPlanner, "build", build)
    return built


def broken_rewrites(monkeypatch, *fields):
    """A model rewritten with every one of ``fields`` on is wrong by 1 % in its readout (after the self-checks)."""
    from fastmace.rewrites import registry as reg

    original = reg.Registry.apply
    wanted = set(fields)
    faulty = []

    def apply(self, model, resolution, *args, **kwargs):
        applied = original(self, model, resolution, *args, **kwargs)
        on = {self.config_field(name) for name in resolution.enabled}
        if wanted <= on:
            with torch.no_grad():
                named = list(model.named_parameters())
                weight = next((p for name, p in named if "readout" in name), named[-1][1])
                weight.mul_(1.01)
            faulty.append(sorted(on))
        return applied

    monkeypatch.setattr(reg.Registry, "apply", apply)
    return faulty


def off_fields(plan):
    return {f.name for f in dataclasses.fields(plan.rewrites) if getattr(plan.rewrites, f.name) in ("off", False)}


def test_no_failure_never_enters_the_fallback(pristine, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the fallback was entered on a run where nothing failed")

    monkeypatch.setattr(fallback, "Fallback", refuse)
    sim = simulation(pristine)
    assert sim.plan.revision == 0 and sim.plan.fallbacks == () and sim.plan.predecessor is None
    assert "(fallback)" not in sim.report.text()
    sim.run(steps=2)
    sim.close()


def test_refuse_raises_the_capture_error(pristine, monkeypatch):
    broken_capture(monkeypatch)
    with pytest.raises(RuntimeError, match="capture of stage B"):
        simulation(pristine, fallback=FallbackConfig(fallback="refuse"))


def test_a_broken_capture_runs_eager_with_a_revision(pristine, monkeypatch):
    built = broken_capture(monkeypatch)
    sim = simulation(pristine)
    plan = sim.plan
    assert plan.tier == "eager" and plan.revision == 1 and plan.predecessor.tier == "graph"
    assert plan.rewrites == plan.predecessor.rewrites
    assert built[0] == "graph" and set(built[1:]) == {"eager"}
    assert [t.outcome for t in plan.fallbacks] == ["passed", "activated"]
    assert sim.gate.status == "skipped"  # the user's check="skip"; the validation is in plan.fallbacks
    assert sim._engine.state.plan_revision == 1
    text = sim.report.text()
    assert "plan revision 1 (fallback): mode eager, reason " in text and CAPTURE_ERROR in text
    sim.run(steps=3)
    assert np.isfinite(sim.committed().forces).all()
    sim.close()


def test_resume_rebuilds_the_fallback_revision(pristine, monkeypatch, tmp_path):
    from fastmace.simulation import Simulation

    broken_capture(monkeypatch)
    sim = simulation(pristine)
    sim.run(steps=2)
    sim.save_restart(tmp_path / "restart")
    sim.close()

    def refuse(*args, **kwargs):
        raise AssertionError("a resume never runs the fallback")

    monkeypatch.setattr(fallback, "Fallback", refuse)
    resumed = Simulation.resume(tmp_path / "restart")
    assert resumed.plan.tier == "eager" and resumed.plan.revision == sim.plan.revision
    np.testing.assert_array_equal(resumed.committed().positions, sim.committed().positions)
    resumed.close()


@pytest.mark.slow
def test_a_broken_rewrite_is_found_by_the_real_gate(pristine, monkeypatch):
    broken_rewrites(monkeypatch, "block_linear")
    with pytest.raises(GateFailed):
        simulation(pristine, check="gate", fallback=FallbackConfig(fallback="refuse"))
    sim = simulation(pristine, check="gate")
    plan = sim.plan
    assert plan.revision == 1 and plan.tier == "graph"
    assert {"block_linear", "grouped_linear", "merged_residual"} <= off_fields(plan)
    assert not {"factor_species", "fold_cutoff", "density_first"} & off_fields(plan)
    assert sim.gate.status == "passed"
    tried = plan.fallbacks
    assert sum(t.reduced_gate_runs for t in tried) <= 64 and sum(t.full_gate_runs for t in tried) == 1
    assert "plan revision 1 (fallback): rewrites " in sim.report.text()
    sim.close()


@pytest.mark.slow
def test_a_broken_pair_is_found_by_the_real_gate(pristine, monkeypatch):
    faulty = broken_rewrites(monkeypatch, "joint_radial_input", "grouped_linear")
    with pytest.raises(GateFailed):
        simulation(pristine, check="gate", fallback=FallbackConfig(fallback="refuse"))
    assert faulty
    sim = simulation(pristine, check="gate")
    off = off_fields(sim.plan)
    assert {"joint_radial_input", "grouped_linear"} <= off
    assert not {"block_linear", "fold_cutoff", "factor_species", "density_first"} & off
    assert sim.gate.status == "passed"
    sim.close()


def test_the_report_lists_every_fallback_revision():
    from fastmace.planner.fallback import TriedCandidate
    from fastmace.planner.plan import Plan
    from fastmace.report import revision_lines

    first = Plan(tier="graph", reason="minimal plan (phase 2)")
    second = first.revise("fallback: rewrites block_linear, grouped_linear off: ddmin: minimal failing set "
                          "block_linear", 0)
    third = second.revise("fallback: mode eager: eager passes", 0, fallbacks=(
        TriedCandidate("mode", "mode eager", "activated", "eager passes", revision=2, full_gate_runs=1),))
    assert revision_lines(first) == []
    assert revision_lines(third) == [
        "plan revision 1 (fallback): rewrites block_linear, grouped_linear off, reason ddmin: minimal failing set "
        "block_linear",
        "plan revision 2 (fallback): mode eager, reason eager passes",
    ]
    assert selfcheck.QUANTITIES  # the module imports
