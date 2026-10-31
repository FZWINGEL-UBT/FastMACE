"""Turn est_matrix points into rule-12 fit cases: for each passing point, re-plan it on the CPU with the point's
rewrites and GPU, capture the planner's final estimate inputs (edges, nodes, capacity, enabled records, self-checks,
resident buffers), and keep the measured peak (the largest reserved over build, warm-up and steady).
    python matrix_cases.py FM_SOURCE OUT.json POINTS_DIR...
"""

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

GPUS = {"NVIDIA RTX 4000 Ada Generation": ((8, 9), 48), "NVIDIA RTX A4000": ((8, 6), 48),
        "NVIDIA GeForce RTX 5060 Ti": ((12, 0), 36)}
calls = []
original = memory.estimate


def spy(dims, **kwargs):
    calls.append(kwargs)
    return original(dims, **kwargs)


memory.estimate = spy
loaded = fm.load("mh1-r2scan")
cases = []
for directory in sys.argv[3:]:
    for f in sorted(Path(directory).glob("point-*.json")):
        d = json.loads(f.read_text())
        if d.get("status") != "ok":
            continue
        name, cell = capacity_scan.base_cell(d["structure"])
        atoms = cell.repeat(tuple(int(n) for n in d["repeat"].split("x")))
        atoms.set_pbc(True)
        rewrites = d["rewrites"]
        field = fm.ForceField(loaded, atoms, precision="float32", rewrites=RewriteConfig(**rewrites))
        cc, sms = GPUS[d["gpu"]]
        planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"))
        planner._free_memory_bytes = lambda hw: hw.memory_bytes
        calls.clear()
        plan = planner.plan(field, atoms, IntegratorConfig(temperature_K=300.0), Workload(max_temperature_K=300.0),
                            Hardware(name=d["gpu"], compute_capability=cc, memory_bytes=2**40, sms=sms))
        kw = calls[-1]
        rewrite_terms = [n for n, _ in kw["rewrite_terms"]]
        phases = d["phases"]
        peak = max(int(p.get("max_reserved", 0)) for p in phases.values() if isinstance(p, dict))
        stage = f.stem.split("float32-", 1)[1].split(str(len(atoms)), 1)[1]
        case = dict(label="mh1-r2scan", structure=name, natoms=len(atoms), dtype="float32", stage=stage,
                    edges=int(kw["edges"]), nodes=int(kw["nodes"]), capacity=int(kw["capacity"]),
                    fused_conv=bool(kw["fused_conv"]), rewrites=rewrite_terms,
                    selfchecks=[n for n, _ in kw["selfcheck_terms"]], resident=[n for n, _ in kw["resident_terms"]],
                    sms=sms, gpu=d["gpu"], peak_bytes=peak,
                    measured_edges=int(d["check"]["initial_edges"]) if d.get("check") else None,
                    recorded_prediction=int(d["plan"]["predicted_memory_bytes"]), source=str(f))
        cases.append(case)
        print(f"{name} {len(atoms):7d} {stage:14s} estimate/peak {case['recorded_prediction'] / peak:5.2f} "
              f"peak {peak / 2**30:6.2f} GiB {peak / len(atoms) / 1000:6.1f} kB/atom  edges {case['edges']} "
              f"measured {case['measured_edges']}", flush=True)
Path(sys.argv[2]).write_text(json.dumps(cases, indent=1))
