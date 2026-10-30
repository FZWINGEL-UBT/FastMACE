"""Where a large cell's GPU memory sits at the limit, by owner (plans/capacity.md, idea 4 of 30 October 2026).

Builds the default plan's ``fm.Simulation`` for one supercell, as ``capacity_scan.py point`` does, runs the
warm-up and some steady steps, then records:

1. **The allocator at the steady state** (``torch.cuda.memory_snapshot``): the graph pool's segments, and the
   default pool's live blocks matched to their owners by address. The owners are the engine's registered
   buffers (state, integrator, force field with the neighbour list, engine flags), the noise batch, the
   neighbour screen's output buffer, and the model's parameters and buffers. Default-pool memory that is
   reserved but not live is the allocator's cache.
2. **Memory outside the allocator** (``cudaMemGetInfo`` minus the reserved bytes) at the start, after loading,
   after the build and after the steps. It is then measured again after every graph is released and the cache
   emptied: the drop is what the graph executables held outside the allocator (WSL2, CAP-T03).
3. **What fills the graph pool:** the caching allocator's history (Python frames) over one step that captures the
   stages again into a fresh pool. Replaying it finds the moment of the largest live total; the blocks alive then
   are grouped by the innermost frame of the model or FastMACE that allocated them.

Prints a ranked table and writes ``<output>/ledger-<name>-<precision>-<atoms>.json``. Measurement only. Run it
inside a local GPU reservation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import activation_profile  # noqa: E402
import capacity_scan  # noqa: E402

GIB = 2 ** 30
# --second-pool groups: the custom ops (module, attribute) whose CUDA kernels allocate in the second pool.
SECOND_POOL_OPS = {
    "segment_gemm": [("fastmace.kernels.triton.segment_gemm", "joined_products_op"),
                     ("fastmace.kernels.triton.segment_gemm", "joined_products_input_grads")],
    "fused_forward": [("fastmace.kernels.oeq_streaming.fused_forward", "fused_folded_message")],
}


def route_to_second_pool(torch, groups):
    """Re-register the groups' custom ops' CUDA kernels to run inside ``torch.cuda.use_mem_pool(pool)`` (any
    thread: the backward's ops run in autograd's), and return the pool. Experiment only."""
    import importlib as _importlib

    pool = torch.cuda.MemPool()
    for group in groups:
        for module_name, attribute in SECOND_POOL_OPS[group]:
            op = getattr(_importlib.import_module(module_name), attribute)
            original = op._init_fn

            def kernel(*args, _original=original, **kwargs):
                if not torch.cuda.is_current_stream_capturing():  # eager calls (warm-ups) keep the default pool
                    return _original(*args, **kwargs)
                with torch.cuda.use_mem_pool(pool):
                    return _original(*args, **kwargs)

            op.register_kernel("cuda")(kernel)
    return pool


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--precision", choices=("float32", "float64"), default="float32")
    p.add_argument("--repeat", required=True, help="NXxNYxNZ")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--steps", type=int, default=20, help="steady steps after the warm-up")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--refuse", action="store_true",
                   help="FallbackConfig(fallback='refuse'): the first failure is raised")
    p.add_argument("--second-pool", nargs="*", default=[], choices=sorted(SECOND_POOL_OPS),
                   help="experiment: run these ops' CUDA kernels with their allocations in a second MemPool "
                        "(torch.cuda.use_mem_pool), so their large buffers do not fragment the graph pool")
    p.add_argument("--tag", default="", help="suffix of the record's file name")
    return p.parse_args(argv)


def memory_point(torch, label):
    """The allocator's counters and the memory outside it, now."""
    free, total = torch.cuda.mem_get_info()
    reserved = torch.cuda.memory_reserved()
    return dict(label=label, allocated=int(torch.cuda.memory_allocated()), reserved=int(reserved),
                max_allocated=int(torch.cuda.max_memory_allocated()),
                max_reserved=int(torch.cuda.max_memory_reserved()), cuda_free=int(free), cuda_total=int(total),
                outside_allocator=int(total - free - reserved))


def owner_ranges(torch, engine):
    """(start, end, owner) of every named tensor the engine knows, for matching blocks by address."""
    named = {}
    for name, tensor in engine._registered_buffers().items():
        named[name] = tensor
    rng = getattr(engine.state, "rng", None)
    if rng is not None and torch.is_tensor(getattr(rng, "batch", None)):
        named["state.rng.batch (Langevin noise)"] = rng.batch
    neighbours = getattr(engine.force_field, "neighbours", None)
    screen = getattr(neighbours, "_pair_search_state", {}).get("buffer") if neighbours is not None else None
    if screen is not None:
        named["force_field.neighbours.screen_buffer"] = screen
    model = getattr(engine.force_field, "model", None)
    if model is not None:
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            named[f"model.{name}"] = tensor
    out = []
    for name, tensor in named.items():
        if not torch.is_tensor(tensor) or not tensor.is_cuda or tensor.numel() == 0:
            continue
        start = tensor.untyped_storage().data_ptr()
        out.append((start, start + tensor.untyped_storage().nbytes(), name))
    return out


def group(name):
    """The ledger row of a named owner."""
    if name.startswith("model."):
        return "model parameters and buffers"
    if name.startswith("force_field.neighbours."):
        return "neighbour list (buffers and screen)"
    if name.startswith("state.rng.batch"):
        return "Langevin noise batch"
    if name.startswith(("integrator.", "state.", "engine.", "derived.")):
        return "MD state, integrator and engine buffers"
    if name.startswith("force_field."):
        return "force field buffers (model inputs, tables)"
    return "other named"


def allocator_view(torch, engine, second=None):
    """The steady state's segments: graph pool, the experiment's second pool, default pool live blocks by owner,
    and the cache."""
    pool = tuple(engine._pool) if engine._pool is not None else None
    second = tuple(second.id) if second is not None else None
    ranges = owner_ranges(torch, engine)
    rows = defaultdict(int)
    detail = defaultdict(int)
    unattributed = []
    totals = dict(graph_pool_reserved=0, graph_pool_active=0, default_reserved=0, default_active=0,
                  second_pool_reserved=0, second_pool_active=0)
    for segment in torch.cuda.memory_snapshot():
        segment_pool = tuple(segment.get("segment_pool_id", ()))
        in_pool = pool is not None and segment_pool == pool
        key = "graph_pool" if in_pool else "second_pool" if second is not None and segment_pool == second else "default"
        totals[f"{key}_reserved"] += segment["total_size"]
        address = segment["address"]
        for block in segment["blocks"]:
            start, size = address, block["size"]
            address += size
            if block["state"] != "active_allocated":
                continue
            totals[f"{key}_active"] += size
            if key != "default":
                continue
            owners = [name for lo, hi, name in ranges if lo < start + size and hi > start]
            if owners:
                rows[group(owners[0])] += size
                detail[owners[0]] += size
            else:
                rows["default pool, live, not a named buffer"] += size
                frames = block.get("frames") or []
                unattributed.append((size, activation_profile._site(frames)[0] if frames else "no frames"))
    unattributed.sort(reverse=True)
    sites = defaultdict(int)
    for size, site in unattributed:
        sites[site] += size
    return dict(totals=totals, rows=dict(rows), owners=dict(sorted(detail.items(), key=lambda kv: -kv[1])[:60]),
                unattributed_sites=dict(sorted(sites.items(), key=lambda kv: -kv[1])[:30]))


def pool_view(snapshot, engine, trace):
    """The graph pool after a capture: its segments, its free blocks, and the live peak of the allocations made in
    it (the trace's events inside its segments). Reserved above that peak is the pool's fragmentation."""
    pool = tuple(engine._pool) if engine._pool is not None else None
    segments = [s for s in snapshot["segments"] if pool is not None and tuple(s.get("segment_pool_id", ())) == pool]
    ranges = [(s["address"], s["address"] + s["total_size"]) for s in segments]
    free = [b["size"] for s in segments for b in s["blocks"] if b["state"] != "active_allocated"]

    def inside(address):
        return any(lo <= address < hi for lo, hi in ranges)

    events = [e for e in trace if e["action"] in ("alloc", "free_requested", "free_completed") and inside(e["addr"])]
    live_peak, _ = activation_profile.live_at_peak(events)
    held_peak, held = live_until_completed(events)
    streams = defaultdict(int)
    for e in events:
        if e["action"] == "alloc":
            streams[str(e.get("stream"))] += e["size"]
    sizes = sorted((s["total_size"] for s in segments), reverse=True)
    # Every event inside the pool's segments, segment mappings included, compactly: [action, address, size, site]
    # in trace order (pool_packing.py replays them offline).
    compact = [[e["action"], int(e["addr"]), int(e["size"]),
                activation_profile._site(e.get("frames") or [])[0] if e["action"] == "alloc" else ""]
               for e in trace if inside(e["addr"])]
    return dict(reserved=int(sum(sizes)), segments=len(sizes), largest_segments=sizes[:10], live_peak=int(live_peak),
                held_peak=int(held_peak), held_at_peak=held, free=int(sum(free)),
                largest_free=int(max(free, default=0)), free_blocks=len(free), events=len(events),
                alloc_bytes_by_stream=dict(streams), segment_ranges=[[lo, hi] for lo, hi in ranges],
                event_list=compact)


def live_until_completed(events):
    """The pool's live peak when a block counts until its free *completes* (a free deferred behind another stream's
    use is not reusable before then), and how much of that peak was requested-but-not-completed frees."""
    live, requested, total, best, best_requested = {}, set(), 0, 0, 0
    for e in events:
        action, addr, size = e["action"], e["addr"], e["size"]
        if action == "alloc":
            live[addr] = size
            total += size
            if total > best:
                best, best_requested = total, sum(live[a] for a in requested if a in live)
        elif action == "free_requested" and addr in live:
            requested.add(addr)
        elif action == "free_completed" and addr in live:
            total -= live.pop(addr)
            requested.discard(addr)
    return best, int(best_requested)


def main(argv=None):
    args = parse_args(argv)
    from fastmace.model.compile import configure_allocator

    configure_allocator()
    import torch

    import fastmace as fm

    name, cell = capacity_scan.base_cell(args.structure)
    repeat = tuple(int(n) for n in args.repeat.lower().split("x"))
    atoms = cell.repeat(repeat)
    atoms.set_pbc(True)
    spec = importlib.util.spec_from_file_location("_ab_frontends", TOOLS / "ab_frontends.py")
    frontends = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frontends)
    frontends.thermalize(atoms, 300.0, 1)
    points = [dict(memory_point(torch, "start (context created)"), windows=capacity_scan.windows_memory())]
    # Python frames on every block allocated from here on (the trace itself is kept short until step 3).
    torch.cuda.memory._record_memory_history(max_entries=1, stacks="python")
    started = time.perf_counter()
    second = route_to_second_pool(torch, args.second_pool) if args.second_pool else None
    field = fm.ForceField(fm.load(args.model), atoms, precision=args.precision)
    torch.cuda.synchronize()
    points.append(memory_point(torch, "force field loaded"))
    from fastmace.config import FallbackConfig

    fallback = {"fallback": FallbackConfig(fallback="refuse")} if args.refuse else {}
    sim = fm.Simulation(atoms, field, seed=1, **fallback, integrator=fm.integrators.Langevin(
        timestep_fs=1.0, temperature_K=300.0, friction_per_fs=0.002))
    torch.cuda.synchronize()
    points.append(memory_point(torch, "simulation built (first captures)"))
    engine = sim._engine
    sim.run(steps=args.warmup)
    torch.cuda.synchronize()
    points.append(memory_point(torch, f"after {args.warmup} warm-up steps"))
    torch.cuda.reset_peak_memory_stats()
    sim.run(steps=args.steps)
    torch.cuda.synchronize()
    steady = memory_point(torch, f"after {args.steps} steady steps")
    points.append(steady)
    view = allocator_view(torch, engine, second)
    edges = engine.force_field.edge_count
    info = dict(edges_at_cutoff=int(edges), padded_sizes=sorted(int(e) for e in engine.edge_sizes_used),
                capacity=int(engine.force_field.edge_capacity), graphs=[str(k) for k in engine._graphs],
                captures=list(engine.capture_memory), force_graph_releases=engine.force_graph_releases)

    # 2. Release every graph and the pool: what the graph executables held outside the allocator.
    graphs = len(engine._graphs)
    engine.invalidate_graphs("memory ledger: release before the measured re-capture")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    released = memory_point(torch, f"{graphs} graphs released, cache emptied")
    points.append(released)

    # 3. One step that captures the stages again into a fresh pool, with the allocator's history.
    torch.cuda.memory._record_memory_history(enabled=None)
    torch.cuda.memory._record_memory_history(max_entries=2_000_000, stacks="python")
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    sim.run(steps=1)
    torch.cuda.synchronize()
    snapshot = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    recaptured = memory_point(torch, "one step, stages captured again")
    points.append(recaptured)
    trace = [event for device in snapshot["device_traces"] for event in device]
    peak, live = activation_profile.live_at_peak(trace)
    sites = defaultdict(lambda: [0, 0])
    for size, frames in live.values():
        site, _ = activation_profile._site(frames)
        sites[site][0] += size
        sites[site][1] += 1
    capture_sites = sorted(([s, b, c] for s, (b, c) in sites.items()), key=lambda row: -row[1])
    pool_detail = pool_view(snapshot, engine, trace)
    if second is not None:
        pool_detail["second_pool_reserved"] = int(sum(s["total_size"] for s in snapshot["segments"]
                                                      if tuple(s.get("segment_pool_id", ())) == tuple(second.id)))
    natoms = len(atoms)
    padded = max(info["padded_sizes"]) if info["padded_sizes"] else int(edges)

    # The ranked ledger at the steady state's peak reserved.
    totals = view["totals"]
    ledger = [("graph pool (captured stages: the force call's activations and transients)",
               totals["graph_pool_reserved"])]
    if second is not None:
        ledger.append((f"second pool ({', '.join(args.second_pool)} kernels' buffers)", totals["second_pool_reserved"]))
    ledger += sorted(view["rows"].items(), key=lambda kv: -kv[1])
    ledger.append(("default pool, cached (reserved, not live)", totals["default_reserved"] - totals["default_active"]))
    outside = steady["outside_allocator"]
    graph_exec = outside - released["outside_allocator"]
    ledger.append(("outside the allocator: graph executables (dropped on release)", graph_exec))
    ledger.append(("outside the allocator: context, libraries, kernels (and other processes)",
                   released["outside_allocator"]))
    ledger.sort(key=lambda row: -row[1])
    record = dict(tool="memory_ledger", structure=name, precision=args.precision, natoms=natoms, repeat=args.repeat,
                  fastmace_commit=_commit(), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  host=__import__("os").uname().nodename, seconds=time.perf_counter() - started,
                  alloc_conf=__import__("os").environ.get("PYTORCH_CUDA_ALLOC_CONF"), second_pool=args.second_pool,
                  edges=info, points=points, allocator=view,
                  ledger=[dict(row=row, bytes=int(b)) for row, b in ledger],
                  recapture=dict(allocated_before=int(base), peak_live_over_before=int(peak),
                                 sites=[dict(site=s, bytes=int(b), blocks=c, per_edge=b / padded, per_atom=b / natoms)
                                        for s, b, c in capture_sites[:80]], pool=pool_detail),
                  plan=sim.plan.describe())
    sim.close()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"ledger-{name}-{args.precision}-{natoms}{args.tag}.json"
    path.write_text(json.dumps(record, indent=1, default=str) + "\n")
    total = steady["cuda_total"]
    print(f"{name} {args.precision} {natoms} atoms, {int(edges)} edges at the cutoff, padded {padded}; "
          f"device {total / GIB:.2f} GiB, steady max reserved {steady['max_reserved'] / GIB:.2f} GiB")
    for row, size in ledger:
        print(f"  {size / GIB:7.3f} GiB  {row}")
    print(f"graph pool after the re-capture: {pool_detail['reserved'] / GIB:.3f} GiB in {pool_detail['segments']} "
          f"segments; its own allocations peaked at {pool_detail['live_peak'] / GIB:.3f} GiB live "
          f"({pool_detail['held_peak'] / GIB:.3f} GiB counting frees when they complete, "
          f"{pool_detail['held_at_peak'] / GIB:.3f} GiB of it requested-not-completed); "
          f"free blocks {pool_detail['free'] / GIB:.3f} GiB (largest {pool_detail['largest_free'] / GIB:.3f} GiB)")
    print(f"re-capture: peak live {peak / GIB:.2f} GiB above {base / GIB:.2f} GiB; top sites:")
    for site, size, count in capture_sites[:args.top]:
        print(f"  {size / GIB:7.3f} GiB  {size / natoms / 1e3:7.1f} kB/atom  {count:5d} blocks  {site}")
    print(path)
    return record


def _commit():
    sys.path.insert(0, str(TOOLS))
    import ab_step_time

    return ab_step_time.fastmace_commit()


if __name__ == "__main__":
    main()
