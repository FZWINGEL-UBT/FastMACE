"""One steady tiled call (device state) for Nsight Systems (11 November 2026, the conductor's check of the A100's GPU
idle share without the torch profiler). The tiled call's methods and autograd.grad are NVTX ranges; four calls warm
up, a fifth is timed without capture, a sixth is captured between cudaProfilerStart and cudaProfilerStop (run under
``nsys profile --capture-range=cudaProfilerApi``). Usage: nsysprof.py REPEAT OUT.json [R]."""
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

import fastmace as fm
from fastmace.model.tiled import TileConfig, TiledForceField
from fastmace.planner import minimal

METHODS = ("_call", "_run_slot", "_body", "_first", "_second", "_fill", "_scatter", "_piece", "_tile_graph",
           "_slot_keys", "_source", "_attrs", "_globals", "_new_slot", "_captured", "_bind", "_bind_tables",
           "_keep_lists", "_kept_on_host", "_kept_form", "_choose_receivers", "_memory", "_state", "_edges",
           "_compiled_once", "_epilogue", "_forward", "_speed", "_free_bytes", "_graph_budget", "_lanes_on",
           "_lane_slot")
nvtx = torch.cuda.nvtx


def instrument(owner, name):
    fn = owner.__dict__.get(name)
    if fn is None:
        return
    static = isinstance(fn, staticmethod)
    raw = fn.__func__ if static else fn

    @functools.wraps(raw)
    def wrapped(*a, **k):
        nvtx.range_push(name)
        try:
            return raw(*a, **k)
        finally:
            nvtx.range_pop()

    setattr(owner, name, staticmethod(wrapped) if static else wrapped)


for name in METHODS:
    instrument(TiledForceField, name)
grad = torch.autograd.grad


def labelled_grad(*a, **k):
    nvtx.range_push("autograd.grad")
    try:
        return grad(*a, **k)
    finally:
        nvtx.range_pop()


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
                                              species="index"))
    positions = field._prepare_one_off(atoms)
times = []
for _ in range(5):
    torch.cuda.synchronize(); t = time.perf_counter(); o = tiled(positions); torch.cuda.synchronize()
    times.append(time.perf_counter() - t); del o
torch.cuda.cudart().cudaProfilerStart()
torch.cuda.synchronize(); t = time.perf_counter()
nvtx.range_push("captured_call")
o = tiled(positions)
torch.cuda.synchronize()
nvtx.range_pop()
captured = time.perf_counter() - t
torch.cuda.cudart().cudaProfilerStop()
del o
rec = dict(atoms=n, times=times, captured_wall=captured,
           last={k: v for k, v in dict(tiled.last).items() if isinstance(v, (int, float, str, bool))})
json.dump(rec, open(out, "w"), indent=1, default=str)
print(json.dumps(dict(atoms=n, times=times, captured_wall=captured)))
