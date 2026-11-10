"""The near-limit memory probe (``EngineConfig.memory_probe``; ``fastmace.planner.memory_probe``;
docs/validation/capacity/estimate-probe.md).

- When: "near-limit" probes the chosen stage where its estimate is within ``PROBE_SPREAD`` of decision 51's limit,
  and the next-cheaper stage where it might fit; "always" both; "off" and plans without a memory choice nothing.
- The replica: the structure's own repeat unit, repeated to 15,000-20,000 atoms, the most isotropic; the cell
  itself when it is no larger; nothing for a structure without a periodic repeat unit.
- The correction: the fixed part stays, the rest scales by the replica's measured over estimated peak, with
  ``PROBE_HEADROOM``; rule 12 then re-plans with the corrected estimates of the probed stages only, and its
  "short by far" test divides a corrected estimate by the headroom instead of ``RESCUE_OVERSTATEMENT``.
- Rule 12 records its choice (``Plan.memory_ladder``): the chosen stage and the next-cheaper one, the chosen without
  its last rescue rung.
- ``Simulation``: on the CPU nothing is probed; a forced probe with a fake child records its measurement in the plan,
  the plan text, the report, provenance and the setup times, keeps the scales in ``simulation.json``, and a resume
  plans with them and never probes.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import test_calibrate as tc
import test_memory_estimate as tme
import test_minimal_planner as tmp

from _support.structures import nacl, nanbo3
from fastmace.config import EngineConfig, IntegratorConfig, RewriteConfig
from fastmace.planner import memory, memory_probe, minimal

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]
GIB = 2**30


def stage(name, estimate, fixed=GIB // 4):
    return memory.LadderStage(name, (("recompute_conv_weights", "on" if name else "off"),), int(estimate), fixed)


# --------------------------------------------------------------------------- when


@pytest.mark.parametrize(("chosen", "cheaper", "setting", "probed"), [
    (10.0, None, "near-limit", ()),  # far below the limit (18 GiB): unambiguous
    (16.0, None, "near-limit", ("",)),  # within the spread of the limit
    (14.0, 21.0, "near-limit", ("recompute_conv_weights", "")),  # the cheaper stage might fit (21 / 1.25 < 18)
    (12.0, 30.0, "near-limit", ()),  # it cannot, and the chosen is far below
    (10.0, 30.0, "always", ("recompute_conv_weights", "")),
    (16.0, 21.0, "off", ()),
])
def test_the_stages_to_probe(chosen, cheaper, setting, probed):
    name = "" if cheaper is None else "recompute_conv_weights"
    ladder = memory.MemoryLadder(20 * GIB, stage(name, chosen * GIB),
                                 None if cheaper is None else stage("", cheaper * GIB))
    assert ladder.limit == memory.MARGIN * 20 * GIB
    assert tuple(s.stage for s in memory_probe.stages_to_probe(ladder, setting)) == probed
    assert memory_probe.stages_to_probe(None, "always") == ()
    assert memory_probe.stages_to_probe(dataclasses.replace(ladder, free_bytes=0), "always") == ()


# --------------------------------------------------------------------------- the replica


@pytest.mark.parametrize(("make", "unit", "replica_atoms"), [(lambda: nacl(repeat=14), 8, 13**3 * 8),
                                                            (lambda: nanbo3(repeat=16), 5, 15**3 * 5)])
def test_the_replica_is_the_repeat_unit_repeated(make, unit, replica_atoms):
    atoms = make()
    found, repeats = memory_probe.repeat_unit(atoms)
    assert len(found) == unit and len(found) * np.prod(repeats) == len(atoms)
    replica = memory_probe.replica(atoms)
    assert len(replica) == replica_atoms and all(replica.pbc)
    assert memory_probe.REPLICA_ATOMS[0] <= len(replica) <= memory_probe.REPLICA_ATOMS[1]
    assert replica.get_volume() / len(replica) == pytest.approx(atoms.get_volume() / len(atoms), rel=1e-9)
    counts = np.bincount(replica.numbers) / len(replica)
    assert counts == pytest.approx(np.bincount(atoms.numbers) / len(atoms))


def test_a_small_cell_is_its_own_replica_and_a_thermalised_one_has_none():
    small = nacl(repeat=3)
    assert len(memory_probe.replica(small)) == len(small)
    rattled = nacl(repeat=14, rattle=0.05, seed=3)
    assert memory_probe.repeat_unit(rattled)[1] == (1, 1, 1)
    assert memory_probe.replica(rattled) is None


# --------------------------------------------------------------------------- the correction


def test_a_probed_estimate_keeps_its_fixed_part_and_scales_the_rest():
    estimate = memory.MemoryEstimate(terms=(("weights (pristine and execution copy)", GIB // 8),
                                            ("activations", 3 * GIB), ("workspace", GIB // 8)),
                                     edges=1, nodes=1, capacity=0, dtype="float32")
    corrected = memory.probed(estimate, 0.5)
    assert memory.fixed_bytes(estimate) == GIB // 4
    assert corrected.total == pytest.approx(GIB / 4 + 3 * GIB * 0.5 * memory.PROBE_HEADROOM, abs=1)
    assert memory.probe_corrected(corrected) and not memory.probe_corrected(estimate)
    assert corrected.term(memory.PROBE_TERM) == corrected.total - estimate.total
    # "short by far": a corrected estimate overstates by the headroom only, an estimate by RESCUE_OVERSTATEMENT
    free = corrected.total / memory.MARGIN / 1.2
    same = memory.MemoryEstimate(terms=(("activations", corrected.total),), edges=1, nodes=1, capacity=0,
                                 dtype="float32")
    assert memory.short_by_far(corrected, free) and not memory.short_by_far(same, free)


def planner_plan(planner, memory_bytes, rewrites, atoms):
    planner._free_memory_bytes = lambda hardware: hardware.memory_bytes
    hardware = dataclasses.replace(tmp.GPU, memory_bytes=int(memory_bytes))
    loaded = tme.types.SimpleNamespace(label=tme.MH1, head=None, sha256=None, backend=None)
    return planner.plan_traits(tme.mh1_traits(), atoms, IntegratorConfig(), tme.WORKLOAD, hardware,
                               precision="float32", rewrites=rewrites, loaded=loaded)


@pytest.fixture
def rescue(monkeypatch):
    """The tiny MH-1 on a cell where rule 12 takes recompute_conv_weights (as test_memory_estimate's rung tests:
    the overstatement at 1, decision 51 dropping nothing first)."""
    monkeypatch.setattr(memory, "RESCUE_OVERSTATEMENT", 1.0)
    atoms, base = nacl(repeat=16), RewriteConfig(reassociate_first="on")
    full = planner_plan(tmp.minimal(), tmp.GPU.memory_bytes, base, atoms)
    free = full.predicted_memory_bytes / memory.MARGIN * 0.97  # the default plan does not fit
    return atoms, base, full, free


@pytest.mark.rule(12)
def test_rule_12_records_its_choice_and_the_next_cheaper_stage(rescue):
    atoms, base, full, free = rescue
    plan = planner_plan(tmp.minimal(), free, base, atoms)
    assert plan.rewrites.recompute_conv_weights == "on"
    ladder = plan.memory_ladder
    assert ladder.free_bytes == int(free) and ladder.chosen.stage == memory.rescue_stage(
        [name for name, value in ladder.chosen.rewrites if value == "on"]) != ""
    assert ladder.chosen.estimate == plan.predicted_memory_bytes
    assert ladder.cheaper is not None and ladder.cheaper.estimate > ladder.limit >= ladder.chosen.estimate
    assert dict(ladder.cheaper.rewrites)["recompute_conv_weights"] == "off"
    assert full.memory_ladder.cheaper is None and full.memory_ladder.chosen.stage == ""
    # off CUDA there is no choice to record
    assert tmp.plan_label("mh1-r2scan").memory_ladder is None


@pytest.mark.rule(12)
def test_rule_12_plans_with_the_probed_scales(rescue):
    atoms, base, full, free = rescue
    planner = memory_probe.with_scales(tmp.minimal(), [("", 0.5)])
    plan = planner_plan(planner, free, base, atoms)  # the default plan, corrected, fits: no rung
    assert plan.rewrites.recompute_conv_weights == "off" and plan.decision("memory")[1] == "fits"
    assert "memory probe" in str(plan.decision("memory")[2])
    assert plan.memory_ladder.chosen.stage == "" and plan.memory_ladder.chosen.estimate < full.predicted_memory_bytes
    unprobed = memory_probe.with_scales(tmp.minimal(), [("recompute_node_region", 0.5)])
    assert planner_plan(unprobed, free, base, atoms).rewrites.recompute_conv_weights == "on"  # not the stage probed


def test_the_probe_measures_each_stage_and_re_plans(rescue):
    atoms, base, full, free = rescue
    planner = tmp.minimal()
    rules = planner_plan(planner, free, base, atoms)
    ladder = rules.memory_ladder
    calls = []

    def launch(stage):
        calls.append(stage.stage)
        return {"replica_atoms": 17_576, "estimate": stage.fixed + 4 * GIB, "reserved": stage.fixed + 2 * GIB,
                "allocated": stage.fixed + GIB, "seconds": 30.0}

    stages = memory_probe.stages_to_probe(ladder, "always")
    outcome = memory_probe.probe(planner, ladder, stages, launch, clock=tc.Clock(),
                                 plan_with=lambda p: planner_plan(p, free, base, atoms))
    assert calls == [ladder.chosen.stage, ""]
    record = outcome.record
    assert [s.scale for s in record.stages] == [0.5, 0.5] and record.headroom == memory.PROBE_HEADROOM
    chosen = record.stages[0]
    assert chosen.corrected == pytest.approx(ladder.chosen.fixed + (
        ladder.chosen.estimate - ladder.chosen.fixed) * 0.5 * memory.PROBE_HEADROOM, abs=1)
    assert record.before == ladder.chosen.stage and record.chosen == "" and record.changed_plan
    assert record.setup_s == 10.0 and outcome.scales == (("recompute_conv_weights", 0.5), ("", 0.5))
    line = memory_probe.report_line(record)
    assert line.startswith("memory probe chose the default stage, not recompute_conv_weights;")
    assert "replica 17576 atoms measured" in line


@pytest.mark.parametrize(("cheaper", "scale", "after"), [
    (30.0, 0.5, True),  # 30 GiB overstated as the chosen stage was: 15 / 1.25 below the limit (18 GiB)
    (30.0, 0.9, False),  # 27 / 1.25 is not
    (None, 0.5, False),
])
def test_the_cheaper_stage_is_measured_after_the_chosen_one_where_its_scale_says_it_might_fit(cheaper, scale, after):
    """memory_probe.cheaper_after: rule 12 overstated SrTiO3 32^3's default plan 2x (24.4 GiB against 12.0
    measured, results/rung-b-20261109), beyond PROBE_SPREAD, so near-limit probing measured only the rung; the
    chosen stage's measured scale now decides whether the cheaper stage is measured too."""
    ladder = memory.MemoryLadder(20 * GIB, stage("recompute_conv_weights" if cheaper else "", 14 * GIB),
                                 None if cheaper is None else stage("", cheaper * GIB))
    assert memory_probe.cheaper_after(ladder, scale) is after
    assert memory_probe.stages_to_probe(ladder, "near-limit") == ()  # neither near the limit nor a cheaper stage


def test_the_probe_measures_the_cheaper_stage_after_the_chosen_one(rescue):
    atoms, base, full, free = rescue
    planner = tmp.minimal()
    ladder = planner_plan(planner, free, base, atoms).memory_ladder
    calls = []

    def launch(stage):  # the replica measures a quarter of each stage's variable estimate
        calls.append(stage.stage)
        return {"replica_atoms": 17_576, "estimate": stage.fixed + 4 * GIB, "reserved": stage.fixed + GIB,
                "allocated": stage.fixed + GIB, "seconds": 30.0}

    outcome = memory_probe.probe(planner, ladder, (ladder.chosen,), launch, clock=tc.Clock(),
                                 plan_with=lambda p: planner_plan(p, free, base, atoms))
    assert calls == [ladder.chosen.stage, ""]
    assert outcome.record.chosen == "" and outcome.record.changed_plan


def test_a_failed_child_keeps_the_rules_plan(rescue):
    atoms, base, full, free = rescue
    planner = tmp.minimal()
    ladder = planner_plan(planner, free, base, atoms).memory_ladder

    def launch(stage):
        raise RuntimeError("child exited with code 1")

    outcome = memory_probe.probe(planner, ladder, (ladder.chosen,), launch, plan_with=None)
    assert outcome.record is None and outcome.planner is planner and "child exited" in outcome.warning


def test_the_option_and_its_rule():
    with pytest.raises(ValueError):
        EngineConfig(memory_probe="sometimes")
    plan = tme.plan()
    _, choice, reason = plan.decision("EngineConfig.memory_probe")
    assert choice == "near-limit" and "auto (near_limit_for_md_on_cuda)" in str(reason)
    assert plan.engine.memory_probe == "near-limit"


# --------------------------------------------------------------------------- Simulation


loaded = tc.loaded
cpu_planner = tc.cpu_planner


@pytest.fixture
def forced_probe(monkeypatch):
    """A memory choice on the CPU (64 GiB "free"), every plan probed with a fake child: the calls."""
    calls = []
    monkeypatch.setattr(minimal.MinimalPlanner, "_free_memory_bytes", lambda self, hardware: 64 * GIB)
    monkeypatch.setattr(memory_probe, "stages_to_probe", lambda ladder, setting: (ladder.chosen,))
    monkeypatch.setattr(memory_probe, "replica", lambda atoms: atoms.copy())

    class Child:
        def __init__(self, simulation, replica_atoms):
            self.atoms = replica_atoms

        def __call__(self, stage):
            calls.append(stage.stage)
            return {"replica_atoms": len(self.atoms), "estimate": stage.fixed + 4 * GIB,
                    "reserved": stage.fixed + 3 * GIB, "allocated": stage.fixed + 2 * GIB, "seconds": 2.0}

        def close(self):
            calls.append("close")

    monkeypatch.setattr(memory_probe, "ChildProbe", Child)
    return calls


def test_nothing_is_probed_on_the_cpu(loaded):
    sim = tc.simulation(loaded)
    assert sim.plan.engine.memory_probe == "off" and sim.plan.decision("EngineConfig.memory_probe")[1] == "off"
    assert sim.plan.memory_ladder is None and sim.plan.memory_probe is None
    assert "memory_probe" not in sim.report.setup_s
    assert all(not text.startswith("memory:") for text in sim.report.text().splitlines())


def test_a_probe_is_recorded_and_a_resume_plans_with_its_scales(loaded, forced_probe, tmp_path):
    from fastmace.io.output import Output
    from fastmace.simulation import Simulation

    sim = tc.simulation(loaded, output=Output(tmp_path / "run"))
    assert forced_probe == ["", "close"]
    record = sim.plan.memory_probe
    assert record is not None and record.stages[0].scale == 0.75 and not record.changed_plan
    assert sim.report.setup_s["memory_probe"] == record.setup_s
    line = next(text for text in sim.report.text().splitlines() if text.startswith("memory:"))
    assert line.split(":", 1)[1].strip().startswith("memory probe kept the default stage; default: replica 64 atoms")
    assert "memory probe: kept the default" in str(sim.plan)
    assert memory.PROBE_TERM in str(sim.plan.decision("memory")[2])
    sim.run(steps=2)
    sim.save_restart(tmp_path / "restart")
    sim.close()
    metadata = json.loads((tmp_path / "run" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["plan"]["memory_probe"]["stages"][0]["scale"] == 0.75
    stored = json.loads((tmp_path / "restart" / "simulation.json").read_text(encoding="utf-8"))
    assert stored["memory_probe_scales"] == [["", 0.75]]
    forced_probe.clear()
    again = Simulation.resume(tmp_path / "restart")
    assert forced_probe == [] and again.plan.memory_probe is None
    assert again._planner.probe_scales == (("", 0.75),)
    assert memory.PROBE_TERM in str(again.plan.decision("memory")[2])


def test_a_probe_child_builds_the_stage_on_the_replica_and_leaves_nothing(loaded, tmp_path):
    """A real fresh child on the CPU: the replica (a larger cell), the stage's rescue fields set on the force
    field's rewrites, rule 12's estimate of the replica back (the peaks are CUDA's: 0 here); no file left behind."""
    from fastmace.engine.subprocess import SubprocessRunner
    from fastmace.io.output import Output

    sim = tc.simulation(loaded, output=Output(tmp_path / "run"))
    before = tc._files(tmp_path)
    runner = SubprocessRunner(dataclasses.replace(sim.options.runner, persistent=False))
    options = dataclasses.replace(sim.options, engine=dataclasses.replace(tc.CPU_ENGINE, memory_probe="off"))
    small = runner.memory_probe(sim, options, nacl(repeat=2), {"recompute_conv_weights": "off"}, steps=2)
    large = runner.memory_probe(sim, options, nacl(repeat=3), {"recompute_conv_weights": "off"}, steps=2)
    assert large["estimate"] > small["estimate"] > 0 and small["reserved"] == small["allocated"] == 0
    assert runner.children == 2 and [path for path in runner.root.iterdir() if path.is_dir()] == []
    runner.close()
    assert tc._files(tmp_path) == before and sim.committed().step == 0


def test_the_child_probe_takes_the_planner_s_engine(loaded):
    """``ChildProbe`` on a run whose caller passed no ``EngineConfig`` (its options hold none): the child gets the
    planner's engine with the probe off, and one real child returns the replica's estimate."""
    sim = tc.simulation(loaded)
    assert sim.options.engine is None or sim.options.engine == tc.CPU_ENGINE
    child = memory_probe.ChildProbe(sim, nacl(repeat=2), steps=2)
    try:
        result = child(stage("", 0))
    finally:
        child.close()
    assert result["replica_atoms"] == 64 and result["estimate"] > 0 and result["seconds"] > 0
