"""2-T04 acceptance 1: MH-1 NaCl 3x3x3 NVT on MACE 0.3.15 (fastmace_mace_other), the planner's own path, gate on."""
import fastmace  # noqa: F401  (before torch)

import dataclasses
import json
import logging
import subprocess
import sys
import time

import torch
from ase.build import bulk

import fastmace as fm
from fastmace import _patches
from fastmace.longrange import polar
from fastmace.model import energy
from fastmace.rewrites import registry as reg
from fastmace.rewrites import versions

out = sys.argv[1]
logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((3, 3, 3))
atoms.rattle(0.02, seed=1)
started = time.perf_counter()
run = fm.md(atoms, model="mh-1", head="matpes_r2scan", temperature_K=300.0, steps=200, seed=7,
            output=f"{out}/run", check="gate")
seconds = time.perf_counter() - started
plan = run.plan
registry = reg.default_registry()
ff = run.simulation._engine.force_field


def plan_record(p):
    return {"revision": p.revision, "tier": p.tier, "reason": p.reason, "step": p.step,
            "compile_mode": p.engine.compile_mode, "backend": p.engine.backend,
            "capture": next((c for k, c, _ in p.decisions if k == "capture"), None),
            "mode": [c, str(r), getattr(r, "code", None)] if (k := "mode") and (
                (c := next((c for kk, c, _ in p.decisions if kk == k), None)) is not None and
                (r := next((r for kk, _, r in p.decisions if kk == k), None)) is not None) else None}


patched = {}
for record in _patches.applied():
    if record.name in _patches.SOURCE_PINS:
        patched[record.name] = _patches.patch_problems(record.name) or "hashes"
    elif record.name == "polar_longrange":
        patched[record.name] = polar.source_problems() or "hashes"
    else:
        patched[record.name] = "not a source patch"
result = {
    "machine": "local PC (WSL2)", "gpu": torch.cuda.get_device_name(0),
    "environment": "~/environments/fastmace_mace_other", "mace": reg.installed_mace_version(),
    "torch": torch.__version__,
    "fastmace_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    "model": "mh-1 matpes_r2scan", "structure": "NaCl 3x3x3 (216 atoms), rattle 0.02 seed 1",
    "md": "NVT 300 K, 200 steps, seed 7, defaults otherwise", "check": "gate", "wall_s": seconds,
    "plan": plan_record(plan), "chain": [plan_record(p) for p in plan.chain()],
    "fallbacks": [dataclasses.asdict(t) if dataclasses.is_dataclass(t) else str(t) for t in plan.fallbacks],
    "precision": plan.decision("force_precision")[1],
    "energy_treatment": ff.energy_treatment, "float64_energy_sources": energy.source_problems() or "hash",
    "gate": run.gate.to_metadata(),
    "unhashed_records": {n: registry.qualification(n).reason for n in registry.names()
                         if not registry.qualification(n).supported},
    "enabled_records": list(ff.resolution.enabled),
    "treatments": {t.name: t.decision for t in versions.treatments()},
    "compile_support": tuple(versions.compile_support()),
    "patches_in_force": patched,
}
with open(f"{out}/result.json", "w") as handle:
    json.dump(result, handle, indent=1, default=str)
print(run.summary)
print(plan.describe())
print(json.dumps({k: result[k] for k in ("plan", "chain", "fallbacks", "precision", "energy_treatment",
                                         "enabled_records", "patches_in_force", "wall_s")}, indent=1, default=str))
print("GATE", result["gate"]["status"])
