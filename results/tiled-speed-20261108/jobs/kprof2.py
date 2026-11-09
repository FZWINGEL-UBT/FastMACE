"""Device time of one steady tiled call (device state) grouped by the Python call site that launched it.
Usage: kprof2.py REPEAT OUT.json [R]. Run from a source tree root with PYTHONPATH set."""
import json
import sys
import time

sys.path.insert(0, "tools/bench")
import fastmace  # noqa: F401
import tiled_capacity as tc
from fastmace.model.compile import configure_allocator

configure_allocator()
import torch
from torch.profiler import ProfilerActivity, profile

import fastmace as fm
from fastmace.model.tiled import TileConfig, TiledForceField
from fastmace.planner import minimal

rep, out = tuple(int(x) for x in sys.argv[1].split("x")), sys.argv[2]
R = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
atoms = tc.supercell("SrTiO3", rep, 0.0, 0.0)
n = len(atoms)
user = fm.ForceField(fm.load("omat0-medium"), atoms, precision="float32")
planner = minimal.default_planner(user)
plan = planner.plan_evaluation(user, atoms)
with tc.no_global_list(True):
    field = planner.build_evaluation(plan, user, atoms, call_rows=R, release_node_attrs=True,
                                     release_whole_system=True, global_list=False)
    tiled = TiledForceField(field, TileConfig(R, None, mode="auto", neighbours="tiles", budget_receivers=(),
                                              species="index"))
    positions = field._prepare_one_off(atoms)
times = []
for _ in range(4):
    torch.cuda.synchronize(); t = time.perf_counter(); o = tiled(positions); torch.cuda.synchronize()
    times.append(time.perf_counter() - t); del o
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
    t = time.perf_counter(); o = tiled(positions); torch.cuda.synchronize(); wall = time.perf_counter() - t


def site(stack):
    frames = [f for f in stack if ("fastmace" in f or "tiled_" in f) and "profiler" not in f]
    return " <- ".join(frames[:4]) if frames else (stack[0] if stack else "?")


rows = {}
for a in prof.key_averages(group_by_stack_n=12):
    dev = getattr(a, "self_device_time_total", None)
    if dev is None:
        dev = a.self_cuda_time_total
    if dev <= 0:
        continue
    k = (a.key[:60], site(a.stack))
    r = rows.setdefault(k, [0.0, 0])
    r[0] += dev
    r[1] += a.count
by_site = {}
for (op, s), (dev, cnt) in rows.items():
    r = by_site.setdefault(s, [0.0, 0, {}])
    r[0] += dev
    r[1] += cnt
    r[2][op] = r[2].get(op, 0) + dev
rec = dict(atoms=n, times=times, wall=wall,
           ops=[dict(op=op, site=s, ns_per_atom=dev * 1e3 / n, calls=cnt)
                for (op, s), (dev, cnt) in sorted(rows.items(), key=lambda kv: -kv[1][0])[:150]],
           sites=[dict(site=s, ns_per_atom=v[0] * 1e3 / n, calls=v[1],
                       ops={o: round(t * 1e3 / n, 1) for o, t in sorted(v[2].items(), key=lambda kv: -kv[1])[:8]})
                  for s, v in sorted(by_site.items(), key=lambda kv: -kv[1][0])[:80]])
json.dump(rec, open(out, "w"), indent=1)
print(json.dumps(dict(atoms=n, wall=wall, times=times)))
