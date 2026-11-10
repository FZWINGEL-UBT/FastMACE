"""Stage 0 of the capture arena (results/capture-arena-20261110): feasibility, no engine code.

- ``build OUT``: compile ``arena_shim.c`` with gcc into ``OUT/arena_shim.so`` (no CUDA headers: the driver API).
- ``toy OUT``: a small MLP's forward and backward captured into ``torch.cuda.MemPool`` on the shim
  (``CUDAPluggableAllocator``), replayed, compared bitwise with eager; the shim's requests summarised.
- ``md OUT --structure S --repeat AxBxC``: the real MD plan (``fm.Simulation``, Langevin, the capacity scan's
  setup) with the engine's graph pool on the shim (``torch.cuda.graph_pool_handle`` returns the MemPool's id).
  Writes the shim's request sequence and the caching allocator's block-level events in that pool (the allocator's
  memory history, actions and sizes only) to ``OUT/md-<tag>.json``.
- ``compare A.json B.json``: whether two ``md`` records' sequences are identical, and where they first differ.

Every GPU run goes through ``tools/ci/local_gpu.py run`` locally; on workstations after the idle check.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent.parent
REPO = TOOLS.parent


def build(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    so = out / "arena_shim.so"
    cmd = ["gcc", "-O2", "-shared", "-fPIC", "-o", str(so), str(HERE / "arena_shim.c"), "-ldl"]
    subprocess.run(cmd, check=True)
    version = subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    print(f"built {so} with {version}")
    return so


class Shim:
    def __init__(self, so: Path):
        import torch

        self.so = str(so)
        self.lib = ctypes.CDLL(self.so)
        self.lib.arena_log_count.restype = ctypes.c_int64
        self.lib.arena_mismatches.restype = ctypes.c_int64
        self.lib.arena_fallback_bytes.restype = ctypes.c_int64
        self.lib.arena_planned_served.restype = ctypes.c_int64
        self.lib.arena_base.restype = ctypes.c_uint64
        self.lib.arena_log_get.argtypes = [ctypes.c_int64, ctypes.POINTER(ctypes.c_int64),
                                           ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint64)]
        self.lib.arena_set_plan.argtypes = [ctypes.c_int64, ctypes.POINTER(ctypes.c_int64),
                                            ctypes.POINTER(ctypes.c_int64), ctypes.c_int64]
        self.allocator = torch.cuda.memory.CUDAPluggableAllocator(self.so, "arena_malloc", "arena_free")

    def log(self):
        kind, size, ptr = ctypes.c_int64(), ctypes.c_int64(), ctypes.c_uint64()
        out = []
        for i in range(self.lib.arena_log_count()):
            self.lib.arena_log_get(i, ctypes.byref(kind), ctypes.byref(size), ctypes.byref(ptr))
            out.append(("malloc" if kind.value == 1 else "free", int(size.value), int(ptr.value)))
        return out

    def plan(self, sizes, offsets, total):
        n = len(sizes)
        s = (ctypes.c_int64 * n)(*sizes)
        o = (ctypes.c_int64 * n)(*offsets)
        assert self.lib.arena_set_plan(n, s, o, int(total)) == 0


def _summary(log):
    mallocs = [s for k, s, _ in log if k == "malloc"]
    return dict(mallocs=len(mallocs), frees=sum(1 for k, *_ in log if k == "free"), malloc_bytes=sum(mallocs),
                largest=sorted(mallocs)[-5:])


def toy(out: Path) -> dict:
    import torch

    so = out / "arena_shim.so"
    shim = Shim(so)
    pool = torch.cuda.MemPool(shim.allocator.allocator())
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(256, 1024), torch.nn.SiLU(), torch.nn.Linear(1024, 256)).cuda()
    x = torch.randn(4096, 256, device="cuda", requires_grad=True)

    def step():
        y = model(x).square().sum()
        (g,) = torch.autograd.grad(y, x)
        return g

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    eager = step().clone()
    graph = torch.cuda.CUDAGraph()
    shim.lib.arena_log_clear()
    with torch.cuda.graph(graph, pool=pool.id, stream=side):
        g = step()
    graph.replay()
    torch.cuda.synchronize()
    log = shim.log()
    result = dict(case="toy", bitwise=bool(torch.equal(g, eager)), max_abs=float((g - eager).abs().max()),
                  shim=_summary(log), allocator_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    print(json.dumps(result))
    return result


POOL_ACTIONS = ("alloc", "free_completed", "segment_alloc", "segment_free", "segment_map", "segment_unmap")


def _pool_events(snapshot, pool_id):
    """The pool's block-level trace as (action, size, addr) (needs ``_record_memory_history(enabled="all")``)."""
    events, seen = [], collections.Counter()
    device = torch_device_index()
    for e in snapshot["device_traces"][device]:
        seen[str(e.get("pool_id"))] += 1
        if tuple(e.get("pool_id") or ()) != tuple(pool_id):
            continue
        if e["action"] in POOL_ACTIONS:
            events.append((e["action"], int(e["size"]), int(e["addr"])))
    return events, dict(seen)


def torch_device_index() -> int:
    import torch

    return torch.cuda.current_device()


def segment_bound(events) -> dict:
    """Offline bounds from a pool's block-level trace (the stage-0 question: is segment granularity enough?).

    - ``live_peak``: the peak of live block bytes in the pool.
    - ``reserved``: the bytes of every segment the pool created (what the caching layer asks the shim for).
    - ``segment_arena``: a first-fit offset plan of the segments themselves, each segment occupying the arena only
      over the event intervals in which one of its blocks is live (a segment that is idle between uses may share
      its bytes with another segment then). This is what a shim-level arena can reach without block placement.
    """
    segs = []  # [start, size, intervals]
    live, cur, peak = {}, 0, 0
    open_at = {}
    for i, (action, size, addr) in enumerate(events):
        if action in ("segment_alloc", "segment_map"):
            segs.append([addr, size, []])
        elif action == "alloc":
            live[addr] = size
            cur += size
            peak = max(peak, cur)
            open_at[addr] = i
        elif action == "free_completed" and addr in live:
            cur -= live.pop(addr)
            seg = next((s for s in segs if s[0] <= addr < s[0] + s[1]), None)
            if seg is not None:
                seg[2].append((open_at.pop(addr), i))
    end = len(events)
    for addr, start in open_at.items():
        seg = next((s for s in segs if s[0] <= addr < s[0] + s[1]), None)
        if seg is not None:
            seg[2].append((start, end))

    def merged(intervals):
        out = []
        for a, b in sorted(intervals):
            if out and a <= out[-1][1]:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        return out

    def first_fit(items):  # items: (size, merged intervals); largest first
        placed, total = [], 0
        for size, iv in sorted(items, key=lambda x: -x[0]):
            clashes = sorted((o, o + s) for o, s, jv in placed
                             if any(a < d and c < b for a, b in iv for c, d in jv))
            offset = 0
            for lo, hi in clashes:
                if offset + size <= lo:
                    break
                offset = max(offset, hi)
            placed.append((offset, size, iv))
            total = max(total, offset + size)
        return total

    blocks = []  # every block allocation with its own lifetime: the block-level (placement-aware) arena
    starts = {}
    for i, (action, size, addr) in enumerate(events):
        if action == "alloc":
            starts[addr] = (i, size)
        elif action == "free_completed" and addr in starts:
            a, s = starts.pop(addr)
            blocks.append((s, [[a, i]]))
    blocks += [(s, [[a, end]]) for a, s in starts.values()]
    return dict(live_peak=peak, reserved=sum(s[1] for s in segs), segments=len(segs),
                segment_arena=first_fit([(s[1], merged(s[2])) for s in segs]), block_arena=first_fit(blocks))


def md(out: Path, structure: str, repeat: str, tag: str, warmup: int, no_split: bool = False) -> dict:
    from fastmace.model.compile import configure_allocator

    configure_allocator(enabled=True)
    import torch

    import fastmace as fm

    sys.path.insert(0, str(TOOLS))
    sys.path.insert(0, str(TOOLS / "bench"))
    import capacity_scan as scan
    from ab_frontends import thermalize

    shim = Shim(out / "arena_shim.so")
    pool = torch.cuda.MemPool(shim.allocator.allocator(), no_split=no_split)
    torch.cuda.graph_pool_handle = lambda: pool.id  # the engine's graph pool on the shim
    # The shim's segments are never expandable; the engine's allocator check must skip the arena pool's segments
    # (an engine built while the pool holds segments failed it on the Ada at 36^3; stage 1 exempts the pool there).
    from fastmace.engine import constraints

    check = constraints.check_allocator
    constraints.check_allocator = lambda config, segments, *a, **k: check(
        config, None if segments is None else [s for s in segments
                                               if tuple(s.get("segment_pool_id") or ()) != tuple(pool.id)], *a, **k)
    name, cell = scan.base_cell(structure)
    atoms = cell.repeat(tuple(int(n) for n in repeat.lower().split("x")))
    atoms.set_pbc(True)
    thermalize(atoms, 300.0, 20261110)
    torch.cuda.memory._record_memory_history(enabled="all", context=None, stacks="python", max_entries=2_000_000)
    started = time.perf_counter()
    model = fm.load("omat0-medium")
    field = fm.ForceField(model, atoms, precision="float32")
    sim = fm.Simulation(atoms, field, seed=20261110, integrator=fm.integrators.Langevin(
        timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.002))
    sim.run(steps=warmup)
    torch.cuda.synchronize()
    snapshot = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    log = shim.log()
    events, pools_seen = _pool_events(snapshot, pool.id)
    bound = segment_bound(events)
    shim_sizes = [(k, s) for k, s, _ in log]
    pool_sizes = [(a, s) for a, s, _ in events]
    digest = lambda seq: hashlib.sha256(json.dumps(seq).encode()).hexdigest()[:16]  # noqa: E731
    result = dict(case="md", tag=tag, structure=name, repeat=repeat, natoms=len(atoms),
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  allocator_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                  seconds=time.perf_counter() - started, plan=sim.plan.describe().splitlines()[2:6],
                  no_split=no_split, shim=_summary(log), shim_digest=digest(shim_sizes),
                  pool_events=len(events), pool_digest=digest(pool_sizes), pools_seen=pools_seen,
                  bound={k: v / 2**30 if k != "segments" else v for k, v in bound.items()},
                  max_reserved=torch.cuda.max_memory_reserved() / 2**30,
                  shim_sequence=shim_sizes, pool_sequence=pool_sizes)
    (out / f"md-{tag}.json").write_text(json.dumps(result))
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("_sequence")}))
    return result


def compare(a: Path, b: Path) -> dict:
    ra, rb = json.loads(a.read_text()), json.loads(b.read_text())
    out = {}
    for key in ("shim_sequence", "pool_sequence"):
        sa, sb = ra[key], rb[key]
        first = next((i for i, (x, y) in enumerate(zip(sa, sb)) if x != y), None)
        if first is None and len(sa) != len(sb):
            first = min(len(sa), len(sb))
        out[key] = dict(identical=first is None, length=(len(sa), len(sb)), first_difference=first,
                        at=None if first is None else (sa[first:first + 3], sb[first:first + 3]))
    print(json.dumps(out))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("build", "toy"):
        sub.add_parser(name).add_argument("out", type=Path)
    m = sub.add_parser("md")
    m.add_argument("out", type=Path)
    m.add_argument("--structure", default="SrTiO3")
    m.add_argument("--repeat", default="10x10x10")
    m.add_argument("--tag", required=True)
    m.add_argument("--warmup", type=int, default=20)
    m.add_argument("--no-split", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    args = p.parse_args(argv)
    if args.cmd == "build":
        build(args.out)
    elif args.cmd == "toy":
        toy(args.out)
    elif args.cmd == "md":
        md(args.out, args.structure, args.repeat, args.tag, args.warmup, args.no_split)
    else:
        compare(args.a, args.b)


if __name__ == "__main__":
    main()
