"""The fallback on the GPU: acceptance 2's injected cases on MH-1 NaCl (plan task 2-T15; SPEC §9.4).

Each case runs ``fm.Simulation(..., check="gate")`` for MH-1 (``mh1-r2scan``) on NaCl 3x3x3 in a
fresh process, in the hybrid environment when it is there, with one fault injected in the child
(monkeypatched, never a hook in the library):

- **broken rewrite**: a model rewritten with ``block_linear`` on is wrong by 0.1 % in its readout
  (after the self-checks): ddmin finds it, and it and the rewrites that require it are off;
- **broken pair**: the same error only when ``joint_radial_input`` and ``grouped_linear`` are both
  on: ddmin finds the pair. No other rewrite requires either, so the pair is the only 1-minimal
  failing set under the ``requires`` closure (a member that another record requires, such as
  ``density_first`` for ``merged_residual``, admits a second minimal set: murphy, 20 October 2026);
- **broken capture**: the captured stage B scales the forces by 1.001, eager is right: the gate
  fails in graph mode, eager passes, and the plan drops the graph mode.

Every case ends in a validated revision within the budget, recorded as a plan revision with its
reason, and runs 20 steps. Untimed.
"""

import importlib.util
from pathlib import Path

import pytest


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _module("_fallback_gpu_engine_helpers", Path(__file__).parent / "test_engine_gpu.py")

pytestmark = [pytest.mark.slow, pytest.mark.acceptance,
              pytest.mark.skipif(ENGINE._mh1_path() is None, reason="needs FASTMACE_MODELS with MH-1")]

CHILD = """
import dataclasses
import fastmace
import fastmace as fm
import torch
from ase import units
from fastmace.rewrites import registry as reg
from fastmace.engine import graph

CASE = SPEC["case"]
if CASE in ("rewrite", "pair"):
    wanted = set(SPEC["fields"])
    original_apply = reg.Registry.apply
    seen = []

    def apply(self, model, resolution, *args, **kwargs):
        applied = original_apply(self, model, resolution, *args, **kwargs)
        on = {self.config_field(name) for name in resolution.enabled}
        if wanted <= on:
            with torch.no_grad():
                named = list(model.named_parameters())
                weight = next((p for name, p in named if "readout" in name), named[-1][1])
                weight.mul_(1.001)
            seen.append(sorted(on))
        return applied

    reg.Registry.apply = apply
elif CASE == "capture":
    original_stage = graph.GraphEngine._stage

    def _stage(self, key):
        run = original_stage(self, key)
        if not self.capture or isinstance(key, tuple) and key[0] == "ACB" or key in ("C", "AC", "ACP"):
            return run

        def wrong():  # a wrong kernel that only the captured stage runs
            run()
            if torch.cuda.is_current_stream_capturing():
                self.state.forces.mul_(1.001)

        return wrong

    graph.GraphEngine._stage = _stage
elif CASE == "capture_error":
    original_stage = graph.GraphEngine._stage

    def _stage(self, key):
        run = original_stage(self, key)
        if not self.capture:
            return run

        def unsupported():  # a host sync inside the capture: cudaErrorStreamCaptureUnsupported
            run()
            if torch.cuda.is_current_stream_capturing():
                torch.nonzero(self.state.forces)

        return unsupported

    graph.GraphEngine._stage = _stage

memory = []  # CUDA memory allocated when each build starts and ends (2-T15c)
if SPEC.get("probe_memory"):
    from fastmace.planner import fallback as fallback_module, minimal

    if not SPEC.get("detach", True):  # the control: failures keep their frames, as before 2-T15c
        fallback_module.detach = lambda error: error
    original_build = minimal.MinimalPlanner.build

    from fastmace.model import templates

    def template_MiB(force_field):
        # The converted-model templates the force field keeps by design (opt-1 OPT-T06): resident, not a leak.
        cache = force_field.__dict__.get("_templates")
        if cache is None:
            return 0.0
        return sum(templates.nbytes(cache.template(key).model) for key in cache.keys()) / 2**20

    def build(planner, plan, force_field, *args, **kwargs):
        gc.collect()
        torch.cuda.synchronize()
        entry = {"tier": plan.tier, "before_MiB": torch.cuda.memory_allocated() / 2**20,
                 "templates_before_MiB": template_MiB(force_field)}
        memory.append(entry)
        engine = original_build(planner, plan, force_field, *args, **kwargs)
        torch.cuda.synchronize()
        entry["after_MiB"] = torch.cuda.memory_allocated() / 2**20
        return engine

    minimal.MinimalPlanner.build = build

model = fm.load(SPEC["label"])
integrator = fm.integrators.Langevin(timestep_fs=2.0, temperature_K=300.0, friction_per_fs=0.02 * units.fs)
started = time.perf_counter()
sim = fm.Simulation(structure(), model, integrator=integrator, seed=7, check=SPEC.get("check", "gate"))
setup = time.perf_counter() - started
sim.run(steps=20)
plan = sim.plan
committed = sim.committed()
off = sorted(f.name for f in dataclasses.fields(plan.rewrites) if getattr(plan.rewrites, f.name) in ("off", False))
before = plan.predecessor
out = {
    "case": CASE, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "fastmace": fastmace.__version__,
    "backend": plan.engine.backend, "dtype": plan.precision.dtype, "tier": plan.tier, "revision": plan.revision,
    "reason": str(plan.reason), "chain": [str(p.reason) for p in plan.chain()],
    "initial_on": [] if before is None else [f.name for f in dataclasses.fields(before.rewrites)
                                             if getattr(before.rewrites, f.name) in ("on", True)],
    "off": off, "gate": sim.gate.to_metadata(), "setup_s": setup,
    "tried": [dataclasses.asdict(t) for t in plan.fallbacks],
    "reduced_gate_runs": sum(t.reduced_gate_runs for t in plan.fallbacks),
    "full_gate_runs": sum(t.full_gate_runs for t in plan.fallbacks),
    "finite": bool(np.isfinite(committed.forces).all() and np.isfinite(committed.positions).all()),
    "report": sim.report.text(), "check": SPEC.get("check", "gate"), "memory": memory,
}
sim.close()
if SPEC.get("probe_memory"):
    # The template's real footprint (opt-1 OPT-T06): what release_templates() gives back. Registered
    # parameters and buffers (templates.nbytes) undercount it: converted modules hold tensors as attributes.
    force_field = sim.force_field
    del sim, plan
    gc.collect()
    torch.cuda.synchronize()
    held = torch.cuda.memory_allocated()
    released = force_field.release_templates()
    gc.collect()
    torch.cuda.synchronize()
    out["template_release"] = {"allocated_drop_MiB": (held - torch.cuda.memory_allocated()) / 2**20,
                               "nbytes_MiB": released / 2**20}
print(json.dumps(out, default=str))
"""

CASES = {
    "rewrite": {"fields": ["block_linear"]},
    "pair": {"fields": ["joint_radial_input", "grouped_linear"]},
    "capture": {},
    "capture_error": {},
}
DEPENDANTS = {"block_linear": {"grouped_linear", "merged_residual"}}


def _run(case, **extra):
    spec = {"temperature": 300.0, "label": "mh1-r2scan", "case": case, **CASES[case], **extra}
    out = ENGINE.run_fresh(ENGINE.COMMON.format(spec=spec) + CHILD, python=ENGINE.HYBRID_PYTHON)
    name = "-".join([case, *(f"{key}-{value}" for key, value in extra.items())])
    ENGINE.report(f"2-T15-fallback-{name}-mh1-nacl", {key: value for key, value in out.items() if key != "report"})
    print(out["report"])
    return out


def _common(out):
    assert out["revision"] >= 1 and out["reason"].startswith("fallback: ")
    assert out["gate"]["status"] == ("skipped" if out.get("check") == "skip" else "passed"), out["gate"]
    assert out["reduced_gate_runs"] <= 64 and out["full_gate_runs"] >= 1
    assert out["tried"][-1]["outcome"] == "activated"
    assert out["finite"]
    assert f"plan revision {out['revision']} (fallback): " in out["report"]


@pytest.mark.parametrize("case", ["rewrite", "pair"])
def test_ddmin_finds_the_broken_rewrites(case):
    out = _run(case)
    fields = set(CASES[case]["fields"])
    if not fields <= set(out["initial_on"]):
        pytest.skip(f"{sorted(fields)} not all on in this plan ({out['initial_on']})")
    _common(out)
    assert out["tier"] == "graph"
    off_now = set(out["off"]) & set(out["initial_on"])
    expected = set(fields)
    for field in fields:
        expected |= DEPENDANTS.get(field, set()) & set(out["initial_on"])
    assert expected <= off_now, (expected, off_now)
    if case == "pair":  # each alone passes: only the pair is off (nothing requires either member)
        assert off_now == fields, off_now


def test_a_broken_capture_drops_the_graph_mode():
    out = _run("capture")
    _common(out)
    assert out["tier"] == "eager" and out["reason"].startswith("fallback: mode eager: ")
    assert out["gate"]["mode"] == "eager stages C, B"


@pytest.mark.parametrize("check", ["skip", "gate"])
def test_a_capture_error_before_the_first_step_falls_back_to_eager(check):
    """A host sync inside the capture (``torch.nonzero``, as an unhashed MACE block does; 2-T15b): the
    first step's graphs are captured before it, so the fallback takes the plan to eager."""
    out = _run("capture_error", check=check)
    _common(out)
    assert out["tier"] == "eager" and out["reason"].startswith("fallback: mode eager: ")
    assert "captur" in out["reason"].lower()
    assert [t["outcome"] for t in out["tried"]] == ["passed", "activated"]


PREPARE_CHILD = """
import fastmace
import fastmace as fm
import torch
from ase import units
from fastmace.engine import graph

original_prepare = graph.GraphEngine.prepare


def run(prepare):
    graph.GraphEngine.prepare = original_prepare if prepare else (lambda self: None)
    model = fm.load(SPEC["label"])
    integrator = fm.integrators.Langevin(timestep_fs=2.0, temperature_K=300.0, friction_per_fs=0.02 * units.fs)
    sim = fm.Simulation(structure(), model, integrator=integrator, seed=7)
    graphs_after_setup = list(sim._engine.info()["graphs"])
    started = time.perf_counter()
    sim.run(steps=SPEC["steps"])
    loop = time.perf_counter() - started
    committed = sim.committed()
    out = {"positions": committed.positions.tolist(), "energy": committed.energy,
           "momenta": committed.momenta.tolist(), "graphs_after_setup": graphs_after_setup,
           "graphs": list(sim._engine.info()["graphs"]), "setup_s": dict(sim._setup_s), "loop_s": loop,
           "backend": sim.plan.engine.backend, "dtype": sim.plan.precision.dtype}
    sim.close()
    del sim
    gc.collect()
    torch.cuda.empty_cache()
    return out


out = {"lazy": run(False), "prepare": run(True), "lazy_again": run(False)}
out.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__)
print(json.dumps(out))
"""


@pytest.mark.slow
def test_prepare_leaves_the_default_trajectory_unchanged():
    """Acceptance 8 for 2-T15b: capturing before the first step instead of at it, in one process
    (MH-1 hybrid float32, NaCl 3x3x3, 200 steps): bitwise equal to the lazy capture when two lazy runs are."""
    import numpy as np

    spec = {"temperature": 300.0, "label": "mh1-r2scan", "steps": 200}
    out = ENGINE.run_fresh(ENGINE.COMMON.format(spec=spec) + PREPARE_CHILD, python=ENGINE.HYBRID_PYTHON)

    def diff(a, b, key):
        return float(np.abs(np.asarray(out[a][key]) - np.asarray(out[b][key])).max())

    summary = {"gpu": out["gpu"], "torch": out["torch"], "backend": out["lazy"]["backend"],
               "dtype": out["lazy"]["dtype"],
               "prepare_vs_lazy_A": diff("prepare", "lazy", "positions"),
               "lazy_again_vs_lazy_A": diff("lazy_again", "lazy", "positions"),
               "prepare_vs_lazy_momenta": diff("prepare", "lazy", "momenta"),
               "graphs_after_setup": {name: out[name]["graphs_after_setup"] for name in ("lazy", "prepare")},
               "setup_s": {name: out[name]["setup_s"] for name in ("lazy", "prepare", "lazy_again")},
               "loop_s": {name: out[name]["loop_s"] for name in ("lazy", "prepare", "lazy_again")}}
    ENGINE.report("2-T15b-prepare-vs-lazy-mh1-nacl", summary)
    assert out["lazy"]["graphs_after_setup"] == [] and len(out["prepare"]["graphs_after_setup"]) >= 1
    assert out["prepare"]["graphs"] == out["lazy"]["graphs"]  # the same graphs, captured earlier
    if summary["lazy_again_vs_lazy_A"] == 0.0:
        assert summary["prepare_vs_lazy_A"] == 0.0 and summary["prepare_vs_lazy_momenta"] == 0.0, summary
    else:  # the backend is not bitwise even between two lazy runs: no worse than that
        assert summary["prepare_vs_lazy_A"] <= summary["lazy_again_vs_lazy_A"] * 10, summary


PREPARED_FORCES_CHILD = """
import fastmace as fm
import torch
from ase import units
from ase.build import bulk

atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 3))  # from 0 K, as 2-T23's runs
model = fm.load(SPEC["label"])
integrator = fm.integrators.Langevin(timestep_fs=0.5, temperature_K=300.0, friction_per_fs=0.02 * units.fs)
sim = fm.Simulation(atoms, model, integrator=integrator, seed=20261020)
engine = sim._engine
temperatures, counts = [], []
for _ in range(SPEC["steps"] // 50):
    sim.run(steps=50)
    committed = sim.committed()
    kinetic_twice = ((committed.momenta ** 2).sum(axis=1) / atoms.get_masses()).sum()  # 2 E_kin, eV
    temperatures.append(float(kinetic_twice / (3 * len(atoms) * units.kB)))
    counts.append(int(engine.force_field.edge_count))
committed = sim.committed()
graphs = list(engine.info()["graphs"])
initial = int(engine.initial_edges)
backend, dtype = sim.plan.engine.backend, sim.plan.precision.dtype
sim.close()
del sim, engine
torch.cuda.empty_cache()
probe = atoms.copy()
probe.positions = committed.positions
_, forces, _ = fm.ForceField(model, probe, precision="float32").evaluate(probe)
print(json.dumps({"max_force_error": float(np.abs(np.asarray(forces) - committed.forces).max()),
                  "max_force": float(np.abs(committed.forces).max()), "temperatures": temperatures,
                  "counts": counts, "initial_edges": initial, "graphs": graphs, "backend": backend, "dtype": dtype,
                  "gpu": torch.cuda.get_device_name()}))
"""


@pytest.mark.slow
def test_prepared_graphs_keep_every_edge_after_the_edge_count_grows():
    """2-T21d: ``prepare()`` captured B with the build's exact edge views, so a padded B graph read only
    the initial edges and dropped the rest from the forces once the count grew (2-T23: MH-1 hybrid NaCl
    3x3x3 from 0 K heated to 897 K at a 300 K target). After 1,000 steps (0.5 ps, well past the
    count's growth at about 0.2 ps) the committed forces equal a fresh evaluation at the same positions."""
    out = ENGINE.run_fresh(ENGINE.COMMON.format(spec={"temperature": 300.0, "label": "mh1-r2scan", "steps": 1000})
                           + PREPARED_FORCES_CHILD, python=ENGINE.HYBRID_PYTHON)
    ENGINE.report("2-T21d-prepared-forces-mh1-nacl", out)
    assert max(out["counts"]) > out["initial_edges"], out  # the test reaches the defect's condition
    assert out["max_force_error"] <= 1e-3, out  # float32 rounding; dropped edges gave eV/A-sized errors
    assert max(out["temperatures"]) < 500.0, out  # 300 K target from 0 K: no runaway heating


@pytest.mark.slow
def test_a_failed_candidates_memory_is_freed_before_the_next_build():
    """2-T15c: the broken-rewrite case, with ``torch.cuda.memory_allocated`` at every build. With
    ``detach`` each build starts at the first build's level (the failed engines are gone); the control
    without it (failures keep their frames, as before) starts every later build with the first,
    failed engine still allocated.

    The converted-model template that the first build stores (opt-1 OPT-T06, one per conversion, by
    design) stays resident and is not counted as a leak (2-T15d). Its footprint is measured as the
    allocation ``release_templates()`` gives back at the end; the registered tensors
    (``templates.nbytes``, recorded at each build) only show that one template is kept throughout.
    Everything else must not grow: a leaked engine is far above the bound, and a residency that grows
    from build to build fails too."""
    fixed = _run("rewrite", probe_memory=True)
    control = _run("rewrite", probe_memory=True, detach=False)
    for out in (fixed, control):
        _common(out)
        assert len(out["memory"]) >= 3, out["memory"]
    first = fixed["memory"][0]
    engine_MiB = first["after_MiB"] - first["before_MiB"]
    template_MiB = fixed["template_release"]["allocated_drop_MiB"]
    later = [entry["before_MiB"] - template_MiB - first["before_MiB"] for entry in fixed["memory"][1:]]
    resident_templates = [entry["templates_before_MiB"] for entry in fixed["memory"][1:]]
    held = [entry["before_MiB"] - control["memory"][0]["before_MiB"] for entry in control["memory"][1:]]
    summary = {"engine_MiB": engine_MiB, "fixed_extra_at_build_MiB": later, "control_extra_at_build_MiB": held,
               "templates_at_build_MiB": resident_templates, "template_release": fixed["template_release"],
               "fixed": fixed["memory"], "control": control["memory"], "gpu": fixed["gpu"]}
    ENGINE.report("2-T15c-fallback-memory-mh1-nacl", summary)
    assert engine_MiB > 50, summary
    assert max(later) <= 0.1 * engine_MiB + 64, summary  # library workspaces may stay; no engine does
    assert max(later) - min(later) <= 0.1 * engine_MiB, summary  # and nothing grows from build to build
    assert max(resident_templates) - min(resident_templates) == 0, summary  # one template, kept, not repeated
    assert min(resident_templates) > 0 and template_MiB >= fixed["template_release"]["nbytes_MiB"], summary
    assert min(held) >= 0.75 * engine_MiB, summary  # the claim, reproduced on the GPU
