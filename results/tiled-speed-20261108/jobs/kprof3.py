"""Device time of one steady tiled call (device state) by labelled region: the tiled call's methods wrapped in
torch.profiler.record_function, each kernel charged to its innermost label. Usage: kprof3.py REPEAT OUT.json [R]."""
import functools
import json
import sys
import time

sys.path.insert(0, "tools/bench")
import fastmace  # noqa: F401
import tiled_capacity as tc
from fastmace.model.compile import configure_allocator

configure_allocator()
import torch
from torch.profiler import ProfilerActivity, profile, record_function

import fastmace as fm
from fastmace.model import tiled as tiled_module
from fastmace.model.tiled import TileConfig, TiledForceField
from fastmace.planner import minimal

LABELS = set()


def label(owner, name, tag=None):
    fn = getattr(owner, name)
    tag = tag or f"{getattr(owner, '__name__', 'mod')}.{name}"
    LABELS.add(tag)

    @functools.wraps(fn)
    def wrapped(*a, **k):
        with record_function(tag):
            return fn(*a, **k)
    setattr(owner, name, staticmethod(wrapped) if isinstance(owner.__dict__.get(name), staticmethod) else wrapped)


T = TiledForceField
for name in ("_fill", "_body", "_scatter", "_first", "_second", "_piece", "_tile_graph", "_slot_keys", "_source",
             "_attrs", "_globals", "_run_slot", "_eager_tile", "_call"):
    if hasattr(T, name):
        label(T, name)
label(tiled_module, "_virial", "virial")
original_forward = T._forward


def forward(self, sweep):
    fn = original_forward(self, sweep)

    def run(*a, **k):
        with record_function(f"fwd{sweep}"):
            return fn(*a, **k)
    return run


T._forward = forward
LABELS.update({"fwd1", "fwd2", "fwd3"})
grad = torch.autograd.grad


def labelled_grad(*a, **k):
    with record_function("autograd.grad"):
        return grad(*a, **k)


torch.autograd.grad = labelled_grad
LABELS.add("autograd.grad")

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
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    t = time.perf_counter(); o = tiled(positions); torch.cuda.synchronize(); wall = time.perf_counter() - t

events = prof.events()
regions, kernels_by = {}, {}
for e in events:
    ks = getattr(e, "kernels", None) or []
    if not ks:
        continue
    path, p = [], e
    while p is not None:
        if p.name in LABELS:
            path.append(p.name)
        p = p.cpu_parent
    key = " < ".join(path[:3]) if path else "(none)"
    dev = sum(k.duration for k in ks)
    r = regions.setdefault(key, [0.0, 0])
    r[0] += dev
    r[1] += len(ks)
    ops = kernels_by.setdefault(key, {})
    ops[e.name[:50]] = ops.get(e.name[:50], 0.0) + dev
rec = dict(atoms=n, times=times, wall=wall,
           regions=[dict(region=k, ns_per_atom=v[0] * 1e3 / n, kernels=v[1],
                         ops={o: round(t * 1e3 / n, 1) for o, t in sorted(kernels_by[k].items(),
                                                                         key=lambda kv: -kv[1])[:10]})
                    for k, v in sorted(regions.items(), key=lambda kv: -kv[1][0])])
json.dump(rec, open(out, "w"), indent=1)
print(json.dumps(dict(atoms=n, wall=wall, times=times, total_ns_per_atom=sum(v[0] for v in regions.values()) * 1e3 / n)))
