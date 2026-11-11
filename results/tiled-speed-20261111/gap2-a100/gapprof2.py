"""Where a steady tiled call (device state) leaves the GPU idle, from the profiler's Chrome trace (11 November 2026).
gapprof.py counted the GPU-timeline copies of the record_function annotations as busy time, which hid the A100's
gaps; here the device is busy only in kernel, memcpy and memset events. Usage: gapprof2.py REPEAT OUT.json [R].

- host: perf_counter self time per tiled method on an unprofiled call (as gapprof.py);
- profiled call: the device's busy time (union of kernels, copies and fills) and its idle gaps; each gap charged to
  the innermost labelled method open on the calling thread at the gap's midpoint (what the host was doing), and
  separately to the label of the launch that ended it;
- CUDA runtime and driver calls by name and innermost label (own thread, else the calling thread): count and time.
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

import os
import tempfile

trace_path = os.path.join(tempfile.mkdtemp(), "trace.json")
prof.export_chrome_trace(trace_path)
trace = json.load(open(trace_path))["traceEvents"]
device, launches, spans, calls = [], {}, {}, []
for e in trace:
    cat, args = e.get("cat", ""), e.get("args", {}) or {}
    if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
        device.append((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0.0)), args.get("correlation"), cat))
    elif cat in ("cuda_runtime", "cuda_driver"):
        launches[args.get("correlation")] = (e.get("tid"), float(e["ts"]))
        calls.append((e["name"], e.get("tid"), float(e["ts"]), float(e.get("dur", 0.0))))
    elif cat == "user_annotation" and e.get("name") in LABELS:
        spans.setdefault(e.get("tid"), []).append((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0.0)),
                                                   e["name"]))
main = next((tid for tid, s in spans.items() if any(name == "_call" for _, _, name in s)), None)


def label_at(tid, ts):
    """The innermost label open on ``tid`` at ``ts`` (else on the calling thread)."""
    open_ = [(b - a, name) for a, b, name in spans.get(tid, ()) if a <= ts < b]
    if not open_ and tid != main and main is not None:
        open_ = [(b - a, name) for a, b, name in spans.get(main, ()) if a <= ts < b]
    return min(open_)[1] if open_ else "(none)"


device.sort()
busy, gaps, cur = 0.0, [], None
by_cat = {}
for s, e, corr, cat in device:
    by_cat[cat] = by_cat.get(cat, 0.0) + (e - s)
    if cur is None or s > cur[1]:
        if cur is not None:
            busy += cur[1] - cur[0]
            gaps.append((cur[1], s, corr))
        cur = [s, e]
    else:
        cur[1] = max(cur[1], e)
if cur is not None:
    busy += cur[1] - cur[0]
span = (cur[1] - device[0][0]) if device else 0.0
during, ending = {}, {}
for s, e, corr in gaps:
    length = e - s
    small = length < 20
    key = "(gaps under 20 us)" if small else label_at(main, (s + e) / 2)
    g = during.setdefault(key, [0, 0.0])
    g[0] += 1
    g[1] += length
    launch = launches.get(corr)
    key = "(gaps under 20 us)" if small else (label_at(*launch) if launch else "(no launch)")
    g = ending.setdefault(key, [0, 0.0])
    g[0] += 1
    g[1] += length
runtime = {}
for name, tid, ts, dur in calls:
    c = runtime.setdefault(f"{name} < {label_at(tid, ts)}", [0, 0.0])
    c[0] += 1
    c[1] += dur


def table(d, top=40):
    return {k: dict(count=v[0], seconds=round(v[1] / 1e6, 4))
            for k, v in sorted(d.items(), key=lambda kv: -kv[1][1])[:top]}


rec = dict(atoms=n, times=times, wall_plain=wall_plain, wall_profiled=wall, device_span_s=span / 1e6,
           device_busy_s=busy / 1e6, idle_s=(span - busy) / 1e6,
           device_by_kind_s={k: v / 1e6 for k, v in by_cat.items()}, host_plain=host_plain,
           gaps_during=table(during), gaps_ended_by=table(ending), runtime=table(runtime),
           last={k: v for k, v in dict(tiled.last).items() if isinstance(v, (int, float, str, bool))})
json.dump(rec, open(out, "w"), indent=1, default=str)
print(json.dumps(dict(atoms=n, times=times, wall=wall_plain, busy=busy / 1e6, idle=(span - busy) / 1e6,
                      during=dict(list(rec["gaps_during"].items())[:8]),
                      ended_by=dict(list(rec["gaps_ended_by"].items())[:8]))))
