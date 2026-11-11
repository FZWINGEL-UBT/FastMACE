"""Idea 11 cost profile: one steady call with the switch off and on, torch profiler, top ops by CUDA and CPU time.
(1) tiled 40^3 (R 8,192, compiled) with TILE_LOCAL_FRAME off / on; (2) whole system 20^3 (evaluation plan, stress)
with WHOLE_SYSTEM_FLOAT64_EDGES off / on. Usage: prof11.py --output DIR"""
import fastmace  # noqa: F401
import argparse
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

import fastmace as fm
from fastmace.model import forces as forces_module
from fastmace.model import tiled as tiled_module
from fastmace.model.tiled import TileConfig, TiledForceField
from fastmace.planner import minimal

sys.path.insert(0, "tools/bench")
import tiled_capacity  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
args = p.parse_args()
os.makedirs(args.output, exist_ok=True)
torch.cuda.set_per_process_memory_fraction(0.9)
model = fm.load("omat0-medium")


def report(name, prof):
    with open(os.path.join(args.output, f"{name}.txt"), "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40, max_name_column_width=70))
        f.write("\n\n")
        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=40, max_name_column_width=70))
    total_cuda = sum(e.device_time for e in prof.events() if e.device_type.name == "CUDA")
    print(name, "kernel ms", round(total_cuda / 1e3, 1), flush=True)


# (1) tiled
atoms = tiled_capacity.supercell("SrTiO3", (40, 40, 40), 0.0, 0.0)
user = fm.ForceField(model, atoms, precision="float32")
planner = minimal.default_planner(user)
plan = planner.plan_evaluation(user, atoms)
with tiled_capacity.no_global_list(True):
    field = planner.build_evaluation(plan, user, atoms, call_rows=8192, release_node_attrs=True,
                                     release_whole_system=True, global_list=False)
    tf = TiledForceField(field, TileConfig(8192, None, mode="compiled", neighbours="tiles", state="device",
                                           species="index"))
    positions = field._prepare_one_off(atoms)
for frame in (False, True):
    tiled_module.TILE_LOCAL_FRAME = frame
    for _ in range(3):
        tf(positions)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        tf(positions)
        torch.cuda.synchronize()
    report(f"tiled40-frame-{'on' if frame else 'off'}", prof)
del tf, user
torch.cuda.empty_cache()

# (2) whole system
atoms = tiled_capacity.supercell("SrTiO3", (20, 20, 20), 0.0, 0.0)
for switch in (False, True):
    forces_module.WHOLE_SYSTEM_FLOAT64_EDGES = switch
    user = fm.ForceField(model, atoms, precision="float32")
    field = user._evaluator(atoms)[1]
    positions = field._prepare_one_off(atoms)
    print("compile_mode", field.compile_mode, "float64_edges", field.float64_edges, flush=True)
    for _ in range(3):
        field(positions)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        field(positions)
        torch.cuda.synchronize()
    report(f"whole20-f64-{'on' if switch else 'off'}", prof)
    del field, user
    torch.cuda.empty_cache()
print("done")
