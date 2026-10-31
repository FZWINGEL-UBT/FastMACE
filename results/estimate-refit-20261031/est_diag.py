"""Rule 12's final estimate, term by term, for MH-1 on NbCl5 / NaCl cells: the default plan on a stand-in
RTX 4000 Ada (or 5060 Ti) with a given free memory, and with given rewrites forced. CPU only (no CUDA)."""

import dataclasses
import json
import sys
from pathlib import Path

import fastmace as fm  # noqa: F401  (before torch)

FM = Path(sys.argv[1])
sys.path.insert(0, str(FM / "tools" / "bench"))
import capacity_scan  # noqa: E402

from fastmace.config import EngineConfig, IntegratorConfig, RewriteConfig  # noqa: E402
from fastmace.planner import memory, minimal  # noqa: E402
from fastmace.planner.plan import Hardware  # noqa: E402
from fastmace.planner.workload import Workload  # noqa: E402

captured = []
original = memory.estimate


def spy(*args, **kwargs):
    out = original(*args, **kwargs)
    captured.append(out)
    return out


memory.estimate = spy
cases = json.loads(sys.argv[2])
loaded = fm.load("mh1-r2scan")
out = []
for case in cases:
    name, cell = capacity_scan.base_cell(case["structure"])
    atoms = cell.repeat(tuple(int(n) for n in case["repeat"].split("x")))
    atoms.set_pbc(True)
    rewrites = RewriteConfig(**case.get("rewrites", {}))
    field = fm.ForceField(loaded, atoms, precision="float32", rewrites=rewrites)
    planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"))
    hardware = Hardware(name=case["gpu"], compute_capability=tuple(case["cc"]), memory_bytes=int(case["free"]),
                        sms=case["sms"])
    planner._free_memory_bytes = lambda hw: hw.memory_bytes
    captured.clear()
    plan = planner.plan(field, atoms, IntegratorConfig(temperature_K=300.0), Workload(max_temperature_K=300.0),
                        hardware)
    est = captured[-1]
    row = dict(case=case, atoms=len(atoms), edges=est.edges, nodes=est.nodes, needed=est.needed,
               terms=[(k, v) for k, v in est.terms], rewrite_terms=list(est.rewrite_terms),
               enabled=[f.name for f in dataclasses.fields(plan.rewrites) if getattr(plan.rewrites, f.name) == "on"])
    out.append(row)
    print(f"== {name} {case['repeat']} {len(atoms)} atoms, {est.edges} edges ({est.edges / len(atoms):.1f}/atom): "
          f"needed {est.needed / 2**30:.2f} GiB = {est.needed / len(atoms) / 1000:.1f} kB/atom")
    for k, v in est.terms:
        print(f"   {k:45s} {v / 2**30:8.2f} GiB {v / len(atoms) / 1000:8.1f} kB/atom")
    for k, v in est.rewrite_terms:
        print(f"     rewrite {k:37s} {v / 2**30:8.2f} GiB {v / len(atoms) / 1000:8.1f} kB/atom")
Path(sys.argv[3]).write_text(json.dumps(out, indent=1, default=str))
