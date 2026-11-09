"""Where a wavefront call's time goes (host state, alias, wavefront on): host wall seconds inside the tiled call's
methods and the host state's methods (perf_counter, innermost charged, so the parts add up), and the GPU's busy time
(kernel durations) against the call's wall time. Usage: wprof.py REPEAT OUT.json [R]."""
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
from torch.profiler import ProfilerActivity, profile

import fastmace as fm
from fastmace.model import tiled as tiled_module
from fastmace.model.tiled import TileConfig, TiledForceField
from fastmace.planner import minimal

host = {}
stack = []


def timed(owner, name, tag):
    fn = getattr(owner, name)

    @functools.wraps(fn)
    def wrapped(*a, **k):
        started = time.perf_counter()
        stack.append(0.0)
        try:
            return fn(*a, **k)
        finally:
            inner = stack.pop()
            spent = time.perf_counter() - started
            entry = host.setdefault(tag, [0.0, 0])
            entry[0] += spent - inner
            entry[1] += 1
            if stack:
                stack[-1] += spent
    setattr(owner, name, staticmethod(wrapped) if isinstance(owner.__dict__.get(name), staticmethod) else wrapped)


T, S = TiledForceField, tiled_module._State
for name in ("_fill", "_body", "_scatter", "_piece", "_tile_graph", "_run_slot", "_wavefront", "_source",
             "_globals", "_attrs"):
    if hasattr(T, name):
        timed(T, name, name)
for name in ("rows", "fetch", "put", "add", "flush", "_defer", "_add_now", "_to_host", "touch", "plan", "_close_now",
             "assign", "release"):
    if hasattr(S, name):
        timed(S, name, "state." + name)
grad = torch.autograd.grad


def labelled_grad(*a, **k):
    started = time.perf_counter()
    stack.append(0.0)
    try:
        return grad(*a, **k)
    finally:
        inner = stack.pop()
        spent = time.perf_counter() - started
        entry = host.setdefault("autograd.grad", [0.0, 0])
        entry[0] += spent - inner
        entry[1] += 1
        if stack:
            stack[-1] += spent


torch.autograd.grad = labelled_grad

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
                                              state="host", alias_state=True, wavefront="on", species="index"))
    positions = field._prepare_one_off(atoms)
times = []
for _ in range(4):
    torch.cuda.synchronize(); t = time.perf_counter(); o = tiled(positions); torch.cuda.synchronize()
    times.append(time.perf_counter() - t); del o
host.clear()
torch.cuda.synchronize(); t = time.perf_counter(); o = tiled(positions); torch.cuda.synchronize()
wall_plain = time.perf_counter() - t
del o
host_plain = {k: [round(v[0], 4), v[1]] for k, v in sorted(host.items(), key=lambda kv: -kv[1][0])}
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    t = time.perf_counter(); o = tiled(positions); torch.cuda.synchronize(); wall = time.perf_counter() - t
busy = 0.0
kernels = {}
for e in prof.events():
    if e.device_type == torch.autograd.DeviceType.CUDA:
        spent = e.time_range.elapsed_us()
        busy += spent
        kernels[e.name[:60]] = kernels.get(e.name[:60], 0.0) + spent
rec = dict(atoms=n, times=times, wall_plain=wall_plain, host_plain=host_plain, wall_profiled=wall,
           gpu_busy_s=busy / 1e6, last=dict(tiled.last),
           kernels={k: round(v / 1e6, 4) for k, v in sorted(kernels.items(), key=lambda kv: -kv[1])[:40]})
json.dump(rec, open(out, "w"), indent=1, default=str)
print(json.dumps(dict(atoms=n, times=times, wall=wall_plain, gpu_busy_s=busy / 1e6,
                      host_top=dict(list(host_plain.items())[:12]))))
