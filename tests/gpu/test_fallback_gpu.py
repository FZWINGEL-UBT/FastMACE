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

model = fm.load(SPEC["label"])
integrator = fm.integrators.Langevin(timestep_fs=2.0, temperature_K=300.0, friction_per_fs=0.02 * units.fs)
started = time.perf_counter()
sim = fm.Simulation(structure(), model, integrator=integrator, seed=7, check="gate")
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
    "report": sim.report.text(),
}
sim.close()
print(json.dumps(out, default=str))
"""

CASES = {
    "rewrite": {"fields": ["block_linear"]},
    "pair": {"fields": ["joint_radial_input", "grouped_linear"]},
    "capture": {},
}
DEPENDANTS = {"block_linear": {"grouped_linear", "merged_residual"}}


def _run(case):
    spec = {"temperature": 300.0, "label": "mh1-r2scan", "case": case, **CASES[case]}
    out = ENGINE.run_fresh(ENGINE.COMMON.format(spec=spec) + CHILD, python=ENGINE.HYBRID_PYTHON)
    ENGINE.report(f"2-T15-fallback-{case}-mh1-nacl", {key: value for key, value in out.items() if key != "report"})
    print(out["report"])
    return out


def _common(out):
    assert out["revision"] >= 1 and out["reason"].startswith("fallback: ")
    assert out["gate"]["status"] == "passed", out["gate"]
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
