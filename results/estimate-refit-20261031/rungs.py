"""Which memory-rescue rungs rule 12 plans for MH-1 cells on a given free memory (no GPU work).
    python rungs.py FM_SOURCE FREE_BYTES GPU_NAME STRUCTURE:REPEAT...
"""

import sys
from pathlib import Path

import fastmace as fm  # noqa: F401  (before torch)

FM = Path(sys.argv[1])
sys.path.insert(0, str(FM / "tools" / "bench"))
import capacity_scan  # noqa: E402

from fastmace.config import EngineConfig, IntegratorConfig  # noqa: E402
from fastmace.planner import minimal  # noqa: E402
from fastmace.planner.plan import Hardware  # noqa: E402
from fastmace.planner.workload import Workload  # noqa: E402

free, gpu = int(float(sys.argv[2])), sys.argv[3]
cc, sms = {"NVIDIA RTX 4000 Ada Generation": ((8, 9), 48), "NVIDIA GeForce RTX 5060 Ti": ((12, 0), 36)}[gpu]
loaded = fm.load("mh1-r2scan")
for cell_text in sys.argv[4:]:
    structure, repeat = cell_text.split(":")
    name, cell = capacity_scan.base_cell(structure)
    atoms = cell.repeat(tuple(int(n) for n in repeat.split("x")))
    atoms.set_pbc(True)
    field = fm.ForceField(loaded, atoms, precision="float32")
    planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"))
    planner._free_memory_bytes = lambda hw: hw.memory_bytes
    plan = planner.plan(field, atoms, IntegratorConfig(temperature_K=300.0), Workload(max_temperature_K=300.0),
                        Hardware(name=gpu, compute_capability=cc, memory_bytes=free, sms=sms))
    rungs = [f"{k}={getattr(plan.rewrites, k)}" for k in ("recompute_conv_weights", "recompute_first_table",
                                                            "recompute_node_segments", "gate_epilogue",
                                                            "recompute_node_region")]
    _, choice, reason = plan.decision("memory")
    print(f"{name} {repeat} {len(atoms)}: memory {choice}, predicted {plan.predicted_memory_bytes / 2**30:.2f} GiB | "
          + " ".join(rungs), flush=True)
