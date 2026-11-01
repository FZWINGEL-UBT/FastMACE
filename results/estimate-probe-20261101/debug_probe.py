"""Why a cell is not probed: its replica, and the trigger on the ladder a point recorded.
    python debug_probe.py FM_SOURCE POINT.json STRUCTURE REPEAT
"""

import json
import sys
from pathlib import Path

import fastmace  # noqa: F401  (before torch)

FM = Path(sys.argv[1])
sys.path.insert(0, str(FM / "tools" / "bench"))
import capacity_scan  # noqa: E402

from fastmace.planner import memory, memory_probe  # noqa: E402

point = json.loads(Path(sys.argv[2]).read_text())
_, cell = capacity_scan.base_cell(sys.argv[3])
atoms = cell.repeat(tuple(int(n) for n in sys.argv[4].split("x")))
atoms.set_pbc(True)
found = memory_probe.repeat_unit(atoms)
print("atoms", len(atoms), "unit", None if found is None else (len(found[0]), found[1]))
replica = memory_probe.replica(atoms)
print("replica", None if replica is None else len(replica))
ladder = point["plan"]["memory_ladder"]


def stage(d):
    return None if d is None else memory.LadderStage(d["stage"], tuple(map(tuple, d["rewrites"])), d["estimate"],
                                                     d["fixed"])


ladder = memory.MemoryLadder(ladder["free_bytes"], stage(ladder["chosen"]), stage(ladder["cheaper"]))
print("stages near-limit", [s.stage for s in memory_probe.stages_to_probe(ladder, "near-limit")])

if len(sys.argv) > 5:  # plan it as the Simulation does
    import fastmace as fm
    from fastmace.config import IntegratorConfig
    from fastmace.planner import minimal
    from fastmace.planner.workload import Workload

    field = fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float32")
    plan = minimal.default_planner(field).plan(field, atoms, IntegratorConfig(temperature_K=300.0),
                                               Workload(max_temperature_K=300.0))
    print("memory_probe", plan.engine.memory_probe, "runner", plan.runner, plan.decision("EngineConfig.memory_probe"))
