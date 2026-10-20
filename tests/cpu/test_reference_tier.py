"""The reference tier for MD (plan task 2-T03; SPEC §8.4, §9.3 rule 1, §9.4 step 5, §12.1).

``ReferenceForceField`` evaluates through the versioned reference calculator and feeds FastMACE's own
integrators on ``GraphEngine`` without capture or padding. Tiny MH-1-class model, NaCl, CPU float64:

- one evaluation equals the corrected reference calculator's, bit for bit;
- a reference-tier trajectory equals the graph tier with every rewrite off within 1e-10 Å over 50
  steps (NVT), and a short NPT run within the same bound;
- the planner builds it for ``mode="reference"``: tier "reference", no rewrites, stock backend,
  uncompiled, no compile family claimed; the report says so; the gate runs as usual;
- an interrupted reference-tier run resumes bit for bit;
- "fits" (§9.4 step 5) comes from the gate's reference result, else from one trial evaluation, in a
  child process on a CUDA plan (an out-of-memory poisons the CUDA context);
- ``allow_reference=True`` is no longer refused, and the default path never builds the tier.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from _support.structures import nacl
from _support.tiny_models import FAMILIES, tiny_mace_model
from fastmace import integrators
from fastmace.config import EngineConfig, RewriteConfig
from fastmace.engine import constraints
from fastmace.errors import PlanRefused
from fastmace.io.output import Output
from fastmace.planner import minimal

pytestmark = [pytest.mark.filterwarnings("ignore::UserWarning")]

NA_CL = (11, 17)
LANGEVIN = integrators.Langevin(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01)
BAOAB = integrators.LangevinBAOAB(timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.01, pressure_GPa=0.5,
                                  cell_mode="full")
CPU = EngineConfig(device="cpu", compile_mode="none")
# Every rewrite off: the graph tier the reference tier is compared with (plan task 2-T03).
REWRITES_OFF = RewriteConfig(**{field.name: "off" for field in dataclasses.fields(RewriteConfig)
                                if field.default == "auto"}, polar_uvu=False)


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
def cpu_planner(monkeypatch):
    """The default planner: the CPU without compilation, with a fresh process record per test."""
    record = constraints.ProcessRecord()
    planner = minimal.MinimalPlanner(engine=CPU, environ={}, process=record)
    monkeypatch.setattr(minimal, "default_planner", lambda force_field: planner)
    return planner


def _atoms(repeat=1):
    return nacl(repeat=repeat, rattle=0.05, seed=3)


def _simulation(loaded, *, rewrites="auto", atoms=None, integrator=LANGEVIN, seed=1, **options):
    from fastmace.model.forcefield import ForceField
    from fastmace.simulation import Simulation

    atoms = _atoms() if atoms is None else atoms
    ff = ForceField(loaded, atoms, precision="float64", rewrites=rewrites)
    return Simulation(atoms, ff, integrator=integrator, seed=seed, **options)


def _positions(sim):
    return sim.committed().positions.copy()


# --------------------------------------------------------------------------- the force field


def test_reference_force_field_equals_the_reference_calculator(loaded):
    from fastmace.engine.reference import ReferenceForceField
    from fastmace.model.forcefield import ForceField
    from fastmace.reference import calculators

    atoms = _atoms()
    ff = ForceField(loaded, atoms, precision="float64")
    field = ReferenceForceField(ff, atoms, device="cpu", dtype="float64")
    assert field.device == torch.device("cpu") and field.dtype == torch.float64
    assert field.padding_exact and not field.compute_stress
    out = field(torch.as_tensor(atoms.positions, dtype=torch.float64))
    reference = calculators.load(str(loaded.path), atoms, dtype="float64", device="cpu", sha256=loaded.sha256,
                                 family=loaded.family)
    energy, forces, _ = reference.evaluate(atoms)
    assert out.energy.shape == (1,) and out.energy.dtype == torch.float64
    assert float(out.energy[0]) == energy
    np.testing.assert_array_equal(out.forces.numpy(), forces)
    assert out.stress is None
    assert field.info()["reference"]["record_id"] == reference.record.id


def test_reference_force_field_has_no_neighbour_work(loaded):
    from fastmace.engine.reference import ReferenceForceField
    from fastmace.model.forcefield import ForceField

    atoms = _atoms()
    field = ReferenceForceField(ForceField(loaded, atoms, precision="float64"), atoms, device="cpu")
    positions = torch.as_tensor(atoms.positions, dtype=torch.float64)
    assert not bool(field.needs_rebuild(positions)) and not field.rebuild(positions)
    field.compact(positions)
    field.use_edges(int(field.edge_count))
    assert field.edge_capacity == 0 and dict(field.buffers()) == {}
    assert field.neighbours.set_capacity(0) is False


def test_float32_reference_is_the_yardstick(loaded):
    from fastmace.engine.reference import ReferenceForceField
    from fastmace.model.forcefield import ForceField

    atoms = _atoms()
    field = ReferenceForceField(ForceField(loaded, atoms), atoms, device="cpu", dtype="float32")
    assert field.dtype == torch.float32
    info = field.info()["reference"]
    assert info["role"] == "yardstick" and info["float64_energy"] is True
    out = field(torch.as_tensor(atoms.positions, dtype=torch.float64))
    assert out.energy.dtype == torch.float64


# --------------------------------------------------------------------------- the plan


def test_mode_reference_plans_the_reference_tier(loaded):
    sim = _simulation(loaded, mode="reference")
    plan = sim.plan
    assert plan.tier == "reference"
    assert not [name for name in ("factor_species", "fold_cutoff", "block_linear", "single_graph_strain")
                if getattr(plan.rewrites, name) == "on"]
    assert plan.engine.compile_mode == "none" and plan.engine.backend == "cueq"
    assert plan.engine.pipelined == "off" and not plan.engine.speculative_graph
    decisions = {key: (choice, reason) for key, choice, reason in plan.decisions}
    assert decisions["mode"][0] == "reference"
    assert sim._engine.capture is False and sim._engine.pad_edges is False
    assert type(sim._engine.force_field).__name__ == "ReferenceForceField"
    lines = [line.strip() for line in sim.report.text().splitlines()]
    assert any(line.startswith("plan:") and "reference," in line for line in lines), lines


def test_reference_tier_is_float64_when_the_float64_energy_source_does_not_hash(loaded, monkeypatch):
    """An unknown MACE (2-T04): the precision promotion holds for the reference tier too (conductor answer)."""
    from fastmace.model.forcefield import ForceField
    from fastmace.planner.workload import Workload
    from fastmace.rewrites import versions

    monkeypatch.setattr(versions, "_problems", lambda name: ["source hash differs"] if name == "float64_energy"
                        else [])
    atoms = _atoms()
    ff = ForceField(loaded, atoms, precision="float32")
    plan = minimal.default_planner(ff).plan(ff, atoms, LANGEVIN.to_config(seed=1), Workload(max_temperature_K=300.0),
                                            mode="reference")
    assert plan.tier == "reference" and plan.precision.dtype == "float64"
    assert ("force_precision", "float32") in plan.requested


def test_reference_tier_claims_no_compile_family(loaded, cpu_planner):
    _simulation(loaded, mode="reference")
    assert cpu_planner.process.family is None


def test_allow_reference_is_no_longer_refused(loaded):
    sim = _simulation(loaded, allow_reference=True)
    assert sim.plan.tier == "graph"
    assert "allow_reference" in sim.plan.allowances


def test_rule_1_still_refuses_what_the_shared_integrators_cannot_do(loaded):
    atoms = _atoms()
    atoms.pbc = (True, True, False)
    with pytest.raises(PlanRefused) as refused:
        _simulation(loaded, atoms=atoms, mode="reference", allow_reference=True)
    assert "non-3d-pbc" in refused.value.codes


def test_default_path_never_builds_the_reference_tier(loaded, monkeypatch):
    from fastmace.engine import reference

    def forbidden(*args, **kwargs):
        raise AssertionError("the default path built a ReferenceForceField")

    monkeypatch.setattr(reference.ReferenceForceField, "__init__", forbidden)
    monkeypatch.setattr(minimal.MinimalPlanner, "reference_fits", forbidden)
    sim = _simulation(loaded)
    sim.run(steps=2)
    assert sim.plan.tier == "graph"


# --------------------------------------------------------------------------- trajectories


def test_reference_tier_equals_graph_tier_with_rewrites_off_over_50_steps(loaded):
    """Plan task 2-T03's CPU acceptance: within 1e-10 Å over 50 steps (tiny model, CPU float64)."""
    reference = _simulation(loaded, mode="reference", seed=5)
    graph = _simulation(loaded, rewrites=REWRITES_OFF, seed=5)
    assert graph.plan.tier == "graph"
    reference.run(steps=50)
    graph.run(steps=50)
    assert reference.committed().step == graph.committed().step == 50
    delta = float(np.abs(_positions(reference) - _positions(graph)).max())
    print(f"\nreference tier against the graph tier with rewrites off, 50 steps: max |dx| = {delta:.3e} A")
    assert delta < 1e-10, delta
    assert abs(reference.committed().energy - graph.committed().energy) < 1e-8


def test_reference_tier_npt_follows_the_graph_tier(loaded):
    reference = _simulation(loaded, mode="reference", integrator=BAOAB, seed=6)
    graph = _simulation(loaded, rewrites=REWRITES_OFF, integrator=BAOAB, seed=6)
    reference.run(steps=10)
    graph.run(steps=10)
    assert reference.committed().stress is not None
    assert float(np.abs(_positions(reference) - _positions(graph)).max()) < 1e-10
    assert float(np.abs(reference.committed().cell - graph.committed().cell).max()) < 1e-10


def test_reference_tier_interrupted_equals_uninterrupted(loaded, tmp_path):
    from fastmace.simulation import Simulation

    def output(directory):
        return Output(directory, thermo_every=1, trajectory_every=1, restart_every=2)

    whole = _simulation(loaded, mode="reference", seed=4, output=output(tmp_path / "whole"))
    whole.run(steps=5)
    first = _simulation(loaded, mode="reference", seed=4, output=output(tmp_path / "parts"))
    first.run(steps=3)
    del first
    resumed = Simulation.resume(tmp_path / "parts")
    assert resumed.plan.tier == "reference"
    resumed.run(steps=3)
    np.testing.assert_array_equal(_positions(resumed), _positions(whole))


def test_gate_passes_in_reference_mode(loaded, tmp_path):
    sim = _simulation(loaded, mode="reference", check="gate",
                      output=Output(tmp_path / "run", trajectory_every=None, restart_every=None))
    assert sim.gate.status == "passed"
    sim.run(steps=2)


# --------------------------------------------------------------------------- fits (§9.4 step 5)


def _plan(loaded, cpu_planner, engine=None):
    from fastmace.model.forcefield import ForceField
    from fastmace.planner.workload import Workload

    atoms = _atoms()
    ff = ForceField(loaded, atoms, precision="float64")
    plan = cpu_planner.plan(ff, atoms, LANGEVIN.to_config(seed=1), Workload(max_temperature_K=300.0), mode="reference")
    if engine is not None:
        plan = dataclasses.replace(plan, engine=dataclasses.replace(plan.engine, **engine))
    return cpu_planner, plan, ff, atoms


def test_fits_from_the_gates_reference_result(loaded, cpu_planner):
    from fastmace.planner import selfcheck

    planner, plan, ff, atoms = _plan(loaded, cpu_planner)
    ran = selfcheck.References(check="gate", values=[{"reference": None}], ran="in-process",
                               info={"reference": {"record_id": "x"}})
    fits, reason = planner.reference_fits(plan, ff, atoms, references=ran)
    assert fits and "gate" in reason
    unavailable = selfcheck.References(check="allow_unavailable", unavailable=selfcheck.GateRecord(
        status="unavailable", check="allow_unavailable",
        reason="reference out of memory: OutOfMemoryError: CUDA out of memory"))
    fits, reason = planner.reference_fits(plan, ff, atoms, references=unavailable)
    assert not fits and "out of memory" in reason


def test_fits_by_one_trial_evaluation_on_the_cpu(loaded, cpu_planner):
    planner, plan, ff, atoms = _plan(loaded, cpu_planner)
    fits, reason = planner.reference_fits(plan, ff, atoms)
    assert fits and "trial" in reason


def test_cuda_trial_runs_in_a_child_process_and_any_failure_does_not_fit(loaded, cpu_planner, monkeypatch):
    from fastmace.engine import reference
    from fastmace.planner import selfcheck

    planner, plan, ff, atoms = _plan(loaded, cpu_planner, engine={"device": "cuda:0"})

    def in_process(*args, **kwargs):
        raise AssertionError("a CUDA trial evaluation ran in the production process")

    monkeypatch.setattr(reference.ReferenceForceField, "__init__", in_process)
    jobs = []

    def child(job, **kwargs):
        jobs.append(job)
        raise selfcheck.GateJobError("the reference child process failed (exit code -9)", why="Killed")

    monkeypatch.setattr(selfcheck, "run_child", child)
    fits, reason = planner.reference_fits(plan, ff, atoms)
    assert not fits and "exit code -9" in reason
    assert jobs and jobs[0]["device"] == "cuda:0" and len(jobs[0]["copies"]) == 1
    np.testing.assert_array_equal(np.asarray(jobs[0]["copies"][0]["positions"]), atoms.positions)

    monkeypatch.setattr(selfcheck, "run_child", lambda job, **kwargs: ([{}], {}))
    assert planner.reference_fits(plan, ff, atoms)[0] is True
