"""Kernel table of one steady call: whole-system compiled field or tiled field (device state). Usage:
kprof.py ws|tiled REPEAT OUT.json"""
import json, sys, time, dataclasses
sys.path.insert(0, "tools/bench")
import fastmace  # noqa
import tiled_capacity as tc
from fastmace.model.compile import configure_allocator
configure_allocator()
import torch
import fastmace as fm
from fastmace.planner import minimal
from fastmace.model.tiled import TileConfig, TiledForceField
from torch.profiler import ProfilerActivity, profile

kind, rep, out = sys.argv[1], tuple(int(x) for x in sys.argv[2].split("x")), sys.argv[3]
atoms = tc.supercell("SrTiO3", rep, 0.0, 0.0)
n = len(atoms)
user = fm.ForceField(fm.load("omat0-medium"), atoms, precision="float32")
planner = minimal.default_planner(user)
plan = planner.plan_evaluation(user, atoms)
if kind == "ws":
    field = planner.build_evaluation(plan, user, atoms)
    positions = field._prepare_one_off(atoms)
    call = lambda: field(positions)
else:
    R = int(sys.argv[4]) if len(sys.argv) > 4 else 16384
    with tc.no_global_list(True):
        field = planner.build_evaluation(plan, user, atoms, call_rows=R, release_node_attrs=True,
                                         release_whole_system=True, global_list=False)
        tiled = TiledForceField(field, TileConfig(R, None, mode="auto", neighbours="tiles", budget_receivers=(),
                                                  species="index"))
        positions = field._prepare_one_off(atoms)
    call = lambda: tiled(positions)
times = []
for i in range(4):
    torch.cuda.synchronize(); t = time.perf_counter(); o = call(); torch.cuda.synchronize(); times.append(time.perf_counter() - t)
    del o
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    t = time.perf_counter(); o = call(); torch.cuda.synchronize(); wall = time.perf_counter() - t
ks = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
table = {}
for e in ks:
    us = e.time_range.end - e.time_range.start
    a = table.setdefault(e.name[:150], [0.0, 0])
    a[0] += us; a[1] += 1
iv = sorted((e.time_range.start, e.time_range.end) for e in ks)
busy, end = 0.0, None
for s, f in iv:
    if end is None or s > end: busy += f - s; end = f
    elif f > end: busy += f - end; end = f
rows = sorted(table.items(), key=lambda kv: -kv[1][0])
avg = prof.key_averages()
host = [dict(name=a.key[:120], self_cpu_s=a.self_cpu_time_total / 1e6, calls=a.count)
        for a in sorted(avg, key=lambda a: -a.self_cpu_time_total)[:40]]
rec = dict(kind=kind, atoms=n, times=times, wall=wall, kernel_s=sum(v[0] for _, v in rows) / 1e6, busy_s=busy / 1e6,
           span_s=(iv[-1][1] - iv[0][0]) / 1e6, launches=len(ks),
           kernels=[dict(name=k, ms=v[0] / 1e3, ns_per_atom=v[0] * 1e3 / n, calls=v[1]) for k, v in rows[:120]],
           host=host, last=getattr(locals().get("tiled"), "last", None))
json.dump(rec, open(out, "w"), indent=1, default=str)
print(kind, n, [round(x, 4) for x in times], "wall", round(wall, 4), "kernel", rec["kernel_s"], "busy", rec["busy_s"], "launches", len(ks))
