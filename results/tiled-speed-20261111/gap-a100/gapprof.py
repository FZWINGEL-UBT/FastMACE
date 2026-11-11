"""Where a steady tiled call (device state) leaves the GPU idle (11 November 2026; the A100's call was 3.7 s with
2.1 s of kernels, plans/ideas-20261111-tiled.md idea 2). Usage: gapprof.py REPEAT OUT.json [R].

- host: perf_counter self time per tiled method on an unprofiled call (innermost charged, the parts add up);
- profiled call: the GPU's busy time (union of kernel intervals) against the call's span; the GPU's idle gaps, each
  charged to the innermost labelled method running on the host at the gap's start (the host then was not launching:
  it was in that method), by label, with the count and total of gaps of 20 us or more;
- CUDA runtime calls (cudaStreamSynchronize, cudaMemcpyAsync, ...) by name and label: count and host time, the
  blocking ones are the host's waits for the GPU.
"""
import bisect
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

host, stack, LABELS = {}, [], set()
METHODS = ("_call", "_run_slot", "_body", "_first", "_second", "_fill", "_scatter", "_piece", "_tile_graph",
           "_slot_keys", "_source", "_attrs", "_globals", "_new_slot", "_captured", "_bind", "_bind_tables",
           "_keep_lists", "_kept_on_host", "_kept_form", "_choose_receivers", "_memory", "_state", "_edges",
           "_compiled_once", "_epilogue", "_forward", "_speed", "_free_bytes", "_graph_budget", "_lanes_on",
           "_lane_slot")


def instrument(owner, name, tag):
    fn = owner.__dict__.get(name)
    if fn is None:
        return
    static = isinstance(fn, staticmethod)
    raw = fn.__func__ if static else fn
    LABELS.add(tag)

    @functools.wraps(raw)
    def wrapped(*a, **k):
        started = time.perf_counter()
        stack.append(0.0)
        try:
            with record_function(tag):
                return raw(*a, **k)
        finally:
            inner = stack.pop()
            spent = time.perf_counter() - started
            entry = host.setdefault(tag, [0.0, 0])
            entry[0] += spent - inner
            entry[1] += 1
            if stack:
                stack[-1] += spent

    setattr(owner, name, staticmethod(wrapped) if static else wrapped)


T = TiledForceField
for name in METHODS:
    instrument(T, name, name)
grad = torch.autograd.grad


def labelled_grad(*a, **k):
    started = time.perf_counter()
    stack.append(0.0)
    try:
        with record_function("autograd.grad"):
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
host.clear()
torch.cuda.synchronize(); t = time.perf_counter(); o = tiled(positions); torch.cuda.synchronize()
wall_plain = time.perf_counter() - t
del o
host_plain = {k: [round(v[0], 4), v[1]] for k, v in sorted(host.items(), key=lambda kv: -kv[1][0])}
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    t = time.perf_counter(); o = tiled(positions); torch.cuda.synchronize(); wall = time.perf_counter() - t

events = prof.events()
kernels, labels, runtime, long_events = [], [], [], []
for e in events:
    start, end = e.time_range.start, e.time_range.end
    if e.device_type == torch.autograd.DeviceType.CUDA:
        if end - start > 50_000:  # a device-side span (not a kernel) longer than 50 ms: reported, not counted
            long_events.append((e.name[:80], round((end - start) / 1e6, 4)))
        else:
            kernels.append((start, end))
    elif e.name in LABELS:
        depth, p = 0, e.cpu_parent
        while p is not None:
            depth += p.name in LABELS
            p = p.cpu_parent
        labels.append((start, end, depth, e.name))
    elif e.name.startswith("cuda") or e.name.startswith("cu"):
        p = e.cpu_parent
        while p is not None and p.name not in LABELS:
            p = p.cpu_parent
        runtime.append((e.name, p.name if p is not None else "(none)", end - start))
kernels.sort()
busy, gaps, cur_s, cur_e = 0.0, [], None, None
for s, e in kernels:
    if cur_e is None or s > cur_e:
        if cur_e is not None:
            busy += cur_e - cur_s
            gaps.append((cur_e, s))
        cur_s, cur_e = s, e
    else:
        cur_e = max(cur_e, e)
if cur_e is not None:
    busy += cur_e - cur_s
span = (max(e for _, e in kernels) - kernels[0][0]) if kernels else 0.0
labels.sort()
starts = [x[0] for x in labels]


def innermost(time_us):
    best = None
    i = bisect.bisect_right(starts, time_us)
    for s, e, depth, name in labels[max(0, i - 4000):i]:
        if s <= time_us < e and (best is None or depth > best[0]):
            best = (depth, name)
    return best[1] if best else "(none)"


by_label = {}
for s, e in gaps:
    length = e - s
    if length < 20:
        key = "(gaps under 20 us)"
    else:
        key = innermost(s)
    g = by_label.setdefault(key, [0, 0.0])
    g[0] += 1
    g[1] += length
calls = {}
for name, label, spent in runtime:
    c = calls.setdefault(f"{name} < {label}", [0, 0.0])
    c[0] += 1
    c[1] += spent
rec = dict(atoms=n, long_device_events=long_events[:20], times=times, wall_plain=wall_plain, wall_profiled=wall, kernel_span_s=span / 1e6,
           gpu_busy_s=busy / 1e6, idle_s=(span - busy) / 1e6, host_plain=host_plain,
           gaps={k: dict(count=v[0], seconds=round(v[1] / 1e6, 4)) for k, v in
                 sorted(by_label.items(), key=lambda kv: -kv[1][1])},
           runtime={k: dict(count=v[0], seconds=round(v[1] / 1e6, 4)) for k, v in
                    sorted(calls.items(), key=lambda kv: -kv[1][1])[:40]},
           last={k: v for k, v in dict(tiled.last).items() if isinstance(v, (int, float, str, bool))})
json.dump(rec, open(out, "w"), indent=1, default=str)
print(json.dumps(dict(atoms=n, times=times, wall=wall_plain, busy=busy / 1e6, idle=(span - busy) / 1e6,
                      gaps_top=dict(list(rec["gaps"].items())[:8]), host_top=dict(list(host_plain.items())[:10]))))
