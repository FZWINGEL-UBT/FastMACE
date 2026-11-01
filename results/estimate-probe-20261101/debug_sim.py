"""Trace the Simulation's memory probe on one cell (GPU): python debug_sim.py FM_SOURCE STRUCTURE REPEAT"""
import sys
from pathlib import Path

import fastmace as fm

FM = Path(sys.argv[1])
sys.path.insert(0, str(FM / "tools" / "bench"))
import capacity_scan  # noqa: E402

from fastmace.planner import memory_probe  # noqa: E402

original = memory_probe.stages_to_probe


def traced(ladder, setting):
    found = original(ladder, setting)
    print("TRACE stages_to_probe", setting, None if ladder is None else (ladder.free_bytes, ladder.chosen.stage,
          ladder.chosen.estimate), [s.stage for s in found], flush=True)
    return found


memory_probe.stages_to_probe = traced
_, cell = capacity_scan.base_cell(sys.argv[2])
atoms = cell.repeat(tuple(int(n) for n in sys.argv[3].split("x")))
atoms.set_pbc(True)
field = fm.ForceField(fm.load("mh1-r2scan"), atoms, precision="float32")
sim = fm.Simulation(atoms, field, seed=1, integrator=fm.integrators.Langevin(timestep_fs=1.0, temperature_K=300.0,
                                                                     friction_per_fs=0.002))
print("TRACE plan probe", sim.plan.memory_probe, flush=True)
print("\n".join(line for line in sim.report.text().splitlines() if "memory" in line or "warn" in line))
