"""Device time of one steady tiled call (device state) by labelled region, every kernel counted (11 November 2026).
kprof6 charged only kernels attached to torch ops and missed Inductor's Triton launches and cuEquivariance's raw
cuLaunchKernel calls (about half the A100's GPU time). Here each kernel in the Chrome trace is charged, by its
correlation id, to the runtime or driver call that launched it, and that call to the innermost label open on its own
thread at the launch, else on the calling thread (autograd's device threads run the backward while the caller waits
in autograd.grad). Usage: kprof7.py REPEAT OUT.json [R]."""
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
original_epilogue = T._epilogue


def epilogue(self, key):
    fn = original_epilogue(self, key)

    def run(*a, **k):
        with record_function("epilogue"):
            return fn(*a, **k)
    return run


T._epilogue = epilogue
LABELS.update({"fwd1", "fwd2", "fwd3", "fwd1m", "fwd3o", "epilogue"})
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

import os
import tempfile

trace_path = os.path.join(tempfile.mkdtemp(), "trace.json")
prof.export_chrome_trace(trace_path)
trace = json.load(open(trace_path))["traceEvents"]
kernels, launches, spans = [], {}, {}
for e in trace:
    cat, args = e.get("cat", ""), e.get("args", {}) or {}
    if cat == "kernel":
        kernels.append((args.get("correlation"), float(e["ts"]), float(e.get("dur", 0.0)), e["name"]))
    elif cat in ("cuda_runtime", "cuda_driver"):
        launches[args.get("correlation")] = (e.get("tid"), float(e["ts"]))
    elif cat == "user_annotation" and e.get("name") in LABELS:
        spans.setdefault(e.get("tid"), []).append((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0.0)), e["name"]))
main = next((tid for tid, s in spans.items() if any(name == "TiledForceField._call" for _, _, name in s)), None)
for s in spans.values():
    s.sort()


def path_at(tid, ts):
    """The labels open on ``tid`` at ``ts``, innermost first (up to 3)."""
    open_ = [(b - a, name) for a, b, name in spans.get(tid, ()) if a <= ts < b]
    return [name for _, name in sorted(open_)][:3]


regions, kernels_by, unlaunched = {}, {}, 0.0
busy, cur = 0.0, None
for corr, ts, dur, name in sorted(kernels, key=lambda k: k[1]):
    if cur is None or ts > cur[1]:
        if cur is not None:
            busy += cur[1] - cur[0]
        cur = [ts, ts + dur]
    else:
        cur[1] = max(cur[1], ts + dur)
    launch = launches.get(corr)
    if launch is None:
        unlaunched += dur
        key = "(no launch)"
    else:
        path = path_at(launch[0], launch[1])
        if not path and main is not None and launch[0] != main:
            path = ["(device thread)"] + path_at(main, launch[1])
        key = " < ".join(path[:3]) if path else "(none)"
    r = regions.setdefault(key, [0.0, 0])
    r[0] += dur
    r[1] += 1
    ops = kernels_by.setdefault(key, {})
    ops[name[:60]] = ops.get(name[:60], 0.0) + dur
if cur is not None:
    busy += cur[1] - cur[0]
total = sum(v[0] for v in regions.values())
rec = dict(atoms=n, times=times, wall=wall, kernels=len(kernels), kernel_sum_ns_per_atom=total * 1e3 / n,
           busy_ns_per_atom=busy * 1e3 / n, unlaunched_ns_per_atom=unlaunched * 1e3 / n,
           regions=[dict(region=k, ns_per_atom=v[0] * 1e3 / n, kernels=v[1],
                         ops={o: round(t * 1e3 / n, 1) for o, t in sorted(kernels_by[k].items(),
                                                                         key=lambda kv: -kv[1])[:25]})
                    for k, v in sorted(regions.items(), key=lambda kv: -kv[1][0])])
json.dump(rec, open(out, "w"), indent=1)
print(json.dumps(dict(atoms=n, wall=wall, times=times, kernel_sum_ns_per_atom=total * 1e3 / n,
                      busy_ns_per_atom=busy * 1e3 / n,
                      top={r["region"]: round(r["ns_per_atom"]) for r in rec["regions"][:12]})))
