"""compiled_graphs on one tiled field: the memory guard's verdict, the real capture pools, and a same-process A/B.

Call 1 runs uncaptured and measures (the guard). The probe then forces the graphs on (``_graphs_fit = True``) under
``set_per_process_memory_fraction(FRACTION)`` (a refused allocation raises instead of oversubscribing), and alternates
captured (b) and uncaptured (a) calls in ABBA blocks on the same field: an uncaptured call leaves the graphs and their
pools in place, so (a) runs beside them. Records per call: wall, allocated, reserved, peak.
Usage: probe_cg.py --repeat 58x58x58 --receivers 8192 [--blocks 3] --output DIR
"""
import fastmace  # noqa: F401  (before torch: soft file locks)
import argparse
import json
import os
import sys
import time

import torch

import fastmace as fm
from fastmace.model.tiled import TileConfig, TiledForceField
from fastmace.planner import minimal

sys.path.insert(0, "tools/bench")
import tiled_capacity  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--repeat", default="58x58x58")
p.add_argument("--receivers", type=int, default=8192)
p.add_argument("--blocks", type=int, default=3)
p.add_argument("--fraction", type=float, default=0.9)
p.add_argument("--output", required=True)
args = p.parse_args()
os.makedirs(args.output, exist_ok=True)
GiB = 2**30
torch.cuda.set_per_process_memory_fraction(args.fraction)
rep = tuple(int(x) for x in args.repeat.split("x"))
atoms = tiled_capacity.supercell("SrTiO3", rep, 0.05, 0.01)
user = fm.ForceField(fm.load("omat0-medium"), atoms, precision="float32")
planner = minimal.default_planner(user)
field = planner.build_evaluation(planner.plan_evaluation(user, atoms), user, atoms)
tiled = TiledForceField(field, TileConfig(args.receivers, mode="compiled", compiled_graphs=True))
records = []


def call(side):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    energy, forces, stress = tiled.evaluate(atoms, stress=True)
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    last = tiled.last
    row = dict(side=side, wall=wall, energy=float(energy), captured=tiled._captured(),
               state=last.get("compiled_graphs"), receivers=last.get("receivers"), tiles=last.get("tiles"),
               slots=len(last.get("slots") or {}), allocated_gib=torch.cuda.memory_allocated() / GiB,
               reserved_gib=torch.cuda.memory_reserved() / GiB, peak_gib=torch.cuda.max_memory_allocated() / GiB,
               fallback=last.get("budget_fallback"))
    records.append(row)
    print(json.dumps(row), flush=True)
    del forces, stress


print(json.dumps(dict(atoms=len(atoms), device=torch.cuda.get_device_name(), total_gib=torch.cuda.mem_get_info()[1] / GiB,
                      fraction=args.fraction)), flush=True)
call("measure")
guard = dict(state=tiled.graphs_state, memory=tiled._graph_memory)
print(json.dumps(dict(guard=guard)), flush=True)
tiled._graphs_fit = True  # the probe: graphs on whatever the guard said
tiled.graphs_state = "forced by the probe"
call("b-capture")
call("b-warm")
for block in range(args.blocks):
    for side in ("a", "b", "b", "a") if block % 2 == 0 else ("b", "a", "a", "b"):
        tiled._graphs_fit = side == "b"
        call(side)
mean = {s: sum(r["wall"] for r in records if r["side"] == s) / max(1, sum(r["side"] == s for r in records))
        for s in ("a", "b")}
energies = {s: [r["energy"] for r in records if r["side"] == s] for s in ("a", "b")}
summary = dict(guard=guard, mean_wall=mean, ratio_b_over_a=mean["b"] / mean["a"],
               energy_spread=max(abs(x - energies["a"][0]) for x in energies["a"] + energies["b"]),
               pool_gib_after_capture=records[1]["reserved_gib"] - records[0]["reserved_gib"])
with open(os.path.join(args.output, f"probe-cg-{len(atoms)}-R{args.receivers}.json"), "w") as f:
    json.dump(dict(records=records, summary=summary), f, indent=1)
print(json.dumps(summary), flush=True)
