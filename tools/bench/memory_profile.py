#!/usr/bin/env python3
"""Per-tensor device-memory profile of one supercell: who holds the memory at the peak, in bytes per atom.

Records the caching allocator's full history (``torch.cuda.memory._record_memory_history``, Python stacks) from
before the model loads, marks the phases in the trace (a marker allocation of a reserved size at each phase start),
runs the case, then replays the trace: every ``alloc`` / ``free_completed`` and every segment map / unmap. At the peak
of *allocated* bytes it lists the live blocks, each with its allocation stack, its memory pool (default or a CUDA
graph's private pool), the phase that allocated it and a scale hint. Owners come from the innermost FastMACE frame
(``OWNERS``, first match); a block allocated by inductor-generated code is marked "(compiled)". Also recorded: the
peak allocated and reserved per phase, the private pools' sizes, and the memory outside the allocator (context,
cuBLAS / cuEq / CUTLASS workspaces) from ``mem_get_info``.

Cases:
- ``md``: ``fm.Simulation`` with Langevin NVT (the capacity scan's setup: graph tier, CUDA graphs, the planner's rescue
  rungs and memory probe), ``--warmup`` then ``--steps`` steps;
- ``single``: one forces + stress call through the ASE calculator (compiled tier, no CUDA graphs, no MD), the shape
  of Symmetrix-XL's capacity probe (one force call).

  python tools/bench/memory_profile.py --case md --structure SrTiO3 --repeat 36x36x36 --output DIR
  python tools/bench/memory_profile.py --diff SMALL.json LARGE.json   # slope per owner and per site, B/atom

Writes ``DIR/memprofile-<case>-<structure>-<atoms>.json``. The scale hint is only a hint (a fixed table can be a
whole multiple of 4 bytes per edge); the slope between two sizes is the attribution.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

TOOLS = Path(__file__).resolve().parents[1]
ROOT = str(Path(fastmace.__file__).resolve().parents[1])

# (owner, substrings of the innermost FastMACE frame's "path:function"); the first match wins.
OWNERS = (
    ("neighbour list", ("fastmace/neighbors/",)),
    ("integrator and state", ("fastmace/integrators/", "fastmace/engine/state", "fastmace/state")),
    ("w' power and value tables", ("wprime_backward.py:power_table", "wprime_backward.py:value_table")),
    ("spline tables", ("spline",)),
    ("composed-skip tables", ("compose_skip", "refresh_composed")),
    ("3xTF32 split copies", ("tf32x3",)),
    ("radial tables", ("rewrites/radial/",)),
    ("model weights and copies", ("fastmace/model/loading", "fastmace/model/compile.py:private_copy",
                                  "fastmace/model/templates")),
    ("rewrites (build)", ("fastmace/rewrites/registry", "fastmace/rewrites/species", "fastmace/rewrites/cueq")),
    ("planner probes", ("fastmace/planner/",)),
    ("model forward/backward", ("fastmace/model/", "fastmace/rewrites/", "fastmace/kernels/")),
    ("engine (graphs, buffers)", ("fastmace/engine/", "fastmace/simulation")),
)
PHASES = ("load", "build", "warmup", "steps", "call", "end")
MARK = 7 << 20  # a phase marker is a uint8 allocation of MARK + 512 * (phase index) bytes


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", choices=("md", "single"), default="md")
    p.add_argument("--structure", default="SrTiO3")
    p.add_argument("--repeat", help="NXxNYxNZ")
    p.add_argument("--model", default="omat0-medium")
    p.add_argument("--precision", choices=("float32", "float64"), default="float32")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--constant", nargs="*", default=[], metavar="MODULE:NAME=VALUE",
                   help="module constants set before the build, as tools/ab_rewrites.py --a-constant takes them")
    p.add_argument("--max-entries", type=int, default=4_000_000, help="allocator trace ring size")
    p.add_argument("--top", type=int, default=40, help="largest live blocks listed")
    p.add_argument("--output", type=Path)
    p.add_argument("--diff", nargs=2, type=Path, metavar=("SMALL", "LARGE"), help="slope between two profiles")
    p.add_argument("--reanalyse", type=Path, metavar="RECORD_JSON",
                   help="analyse the saved snapshot next to this record again (after OWNERS changes)")
    p.add_argument("--phase", help="with --diff: the live set at this phase's peak (warmup, steps, call)")
    return p.parse_args(argv)


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ours(frame):
    """A FastMACE frame, from any source copy (a snapshot taken elsewhere, or a compiled graph cached from another
    copy, names that copy's paths)."""
    name = frame.get("filename", "").replace(os.sep, "/")
    return "/fastmace/" in name and "site-packages" not in name


def _rel(frame):
    name = frame.get("filename", "").replace(os.sep, "/")
    return name[name.rindex("/fastmace/") + 1:] if _ours(frame) else name


def _text(frame):
    return f"{_rel(frame)}:{frame.get('line')} {frame.get('name')}"


def _compiled(frames):
    return any("/inductor/" in f.get("filename", "") and "torch/_inductor" not in f.get("filename", "")
               for f in frames)


def owner_of(frames):
    ours = [f for f in frames if _ours(f)]
    owner = "outside FastMACE"
    if ours:
        key = f"{_rel(ours[0])}:{ours[0].get('name', '')}"
        owner = next((name for name, keys in OWNERS if any(k in key for k in keys)), "other FastMACE")
    return owner + (" (compiled)" if _compiled(frames) else "")


def site(frames):
    """The innermost FastMACE frame, with the generated-code line when a compiled graph allocated the block."""
    ours = [f for f in frames if _ours(f)]
    text = _text(ours[0]) if ours else (_text(frames[0]) if frames else "no stack")
    generated = [f for f in frames if "/inductor/" in f.get("filename", "")
                 and "torch/_inductor" not in f.get("filename", "")]
    if generated:
        text += f" <- {Path(generated[0]['filename']).name[:12]}:{generated[0]['line']}"
    return text


def scale_of(size, natoms, edges):
    """A hint: 'atom' / 'edge' when ``size`` is a whole multiple of 4 bytes per atom / edge to 0.1 %."""
    for name, count in (("edge", edges), ("atom", natoms)):
        if count and size >= 4 * count:
            columns = size / (4 * count)
            if abs(columns - round(columns)) <= 0.001 * columns:
                return name, round(columns)
    return "fixed", 0


def analyse(snapshot, natoms, edges, top, max_entries):
    """Replay the trace: the phases, the peak of allocated bytes overall and per phase, and the live set at the
    overall peak by owner, site, pool and the phase that allocated each block."""
    pools = [(seg["address"], seg["address"] + seg["total_size"], tuple(seg.get("segment_pool_id", (0, 0))))
             for seg in snapshot["segments"]]
    reserved_by_pool = {}
    for lo, hi, pool in pools:
        reserved_by_pool[str(pool)] = reserved_by_pool.get(str(pool), 0) + (hi - lo)

    def pool_of(addr):
        pool = next((pool for lo, hi, pool in pools if lo <= addr < hi), None)
        return "freed by the end" if pool is None else ("default" if pool == (0, 0) else f"private {pool}")

    markers = {MARK + 512 * i: name for i, name in enumerate(PHASES)}
    trace = [e for dev in snapshot["device_traces"] for e in dev]
    phase, sizes, allocated, reserved = "before", {}, 0, 0
    by_phase = {}
    peak, peak_at, peak_reserved, peak_reserved_phase = -1, -1, 0, None
    # per pool (by the address's segment at the end; a private pool keeps its segments): live bytes and their peak
    pool_live, pool_peak, pool_peak_at, private_of = {}, {}, {}, {}
    # each growth of the reserve: how much was free (reserved - allocated) when it grew, and the allocation that
    # needed it (the next alloc); growth while more than the request was free is fragmentation
    growths, pending = [], None
    for i, e in enumerate(trace):
        action, size = e["action"], int(e.get("size", 0))
        if action == "alloc":
            if size in markers:
                phase = markers[size]
                continue
            sizes[e["addr"]] = size
            allocated += size
            pool = private_of.setdefault(e["addr"], pool_of(e["addr"]))
            pool_live[pool] = pool_live.get(pool, 0) + size
            if pool_live[pool] > pool_peak.get(pool, 0):
                pool_peak[pool], pool_peak_at[pool] = pool_live[pool], i
            if pending is not None:
                pending.update(request=size, site=site(e.get("frames", [])), pool=pool)
                growths.append(pending)
                pending = None
        elif action == "free_completed":
            if e["addr"] in sizes:
                pool = private_of.pop(e["addr"], "freed by the end")
                pool_live[pool] = pool_live.get(pool, 0) - sizes[e["addr"]]
            allocated -= sizes.pop(e["addr"], 0)
        elif action in ("segment_alloc", "segment_map"):
            if pending is None:
                pending = dict(at=i, phase=phase, grew=0, free_before=reserved - allocated, reserved_before=reserved)
            pending["grew"] += size
            reserved += size
        elif action in ("segment_free", "segment_unmap"):
            reserved -= size
        row = by_phase.setdefault(phase, dict(allocated=0, reserved=0, at=-1))
        if allocated > row["allocated"]:
            row["allocated"], row["at"] = allocated, i
        row["reserved"] = max(row["reserved"], reserved)
        if allocated > peak:
            peak, peak_at = allocated, i
        if reserved > peak_reserved:
            peak_reserved, peak_reserved_phase = reserved, phase

    def live_at(index, only=None):
        """The live set after trace entry ``index`` (only the blocks in pool ``only``, if given)."""
        live, born, phase = {}, {}, "before"
        for e in trace[:index + 1]:
            if e["action"] == "alloc":
                if int(e["size"]) in markers:
                    phase = markers[int(e["size"])]
                    continue
                live[e["addr"]], born[e["addr"]] = e, phase
            elif e["action"] == "free_completed":
                live.pop(e["addr"], None)
        owners, sites, blocks = {}, {}, []
        for addr, e in live.items():
            frames, size = e.get("frames", []), int(e["size"])
            owner, where, pool = owner_of(frames), site(frames), pool_of(addr)
            if only is not None and pool != only:
                continue
            scale, columns = scale_of(size, natoms, edges)
            row = owners.setdefault(owner, dict(owner=owner, bytes=0, blocks=0, by_phase={}, by_pool={}))
            row["bytes"] += size
            row["blocks"] += 1
            row["by_phase"][born[addr]] = row["by_phase"].get(born[addr], 0) + size
            kind = "private" if pool.startswith("private") else pool
            row["by_pool"][kind] = row["by_pool"].get(kind, 0) + size
            srow = sites.setdefault(where, dict(site=where, owner=owner, bytes=0, blocks=0))
            srow["bytes"] += size
            srow["blocks"] += 1
            blocks.append(dict(bytes=size, owner=owner, scale=scale, columns=columns, pool=pool, phase=born[addr],
                               site=where, frames=[_text(f) for f in frames[:14]]))
        blocks.sort(key=lambda b: -b["bytes"])
        return phase, dict(live_bytes=sum(b["bytes"] for b in blocks),
                           by_owner=sorted(owners.values(), key=lambda r: -r["bytes"]),
                           by_site=sorted(sites.values(), key=lambda r: -r["bytes"])[:200], largest=blocks[:top])

    phase, at_peak = live_at(peak_at)
    # each pool at its own peak: a CUDA graph's private pool is sized by the live peak of its capture
    at_pool_peaks = {pool: dict(peak_allocated_bytes=pool_peak[pool], **live_at(at, only=pool)[1])
                     for pool, at in pool_peak_at.items() if pool_peak[pool] > 0}
    at_phase_peaks = {name: dict(peak_allocated_bytes=row["allocated"], **live_at(row["at"])[1])
                      for name, row in by_phase.items() if name in ("warmup", "steps", "call") and row["at"] >= 0}
    return dict(trace_entries=len(trace), trace_starts_with=trace[0]["action"] if trace else None,
                trace_full=len(trace) >= 0.99 * max_entries, peak_allocated_bytes=peak, peak_at_entry=peak_at,
                peak_phase=phase, peak_reserved_replayed=peak_reserved, peak_reserved_phase=peak_reserved_phase,
                peaks_by_phase={k: dict(allocated=v["allocated"], reserved=v["reserved"]) for k, v in by_phase.items()},
                live_at_peak_bytes=at_peak["live_bytes"], by_owner=at_peak["by_owner"], by_site=at_peak["by_site"],
                largest=at_peak["largest"], at_phase_peaks=at_phase_peaks, at_pool_peaks=at_pool_peaks,
                reserved_by_pool_at_end=reserved_by_pool,
                live_peak_by_pool=pool_peak, growths=dict(
                    count=len(growths), bytes=sum(g["grew"] for g in growths),
                    while_request_fit_in_free=sum(g["grew"] for g in growths
                                                  if g.get("request", 0) <= g["free_before"]),
                    largest=sorted(growths, key=lambda g: -g["grew"])[:60]))


def mark(torch, phase):
    """A phase marker in the allocator trace (allocated and freed at once)."""
    marker = torch.empty(MARK + 512 * PHASES.index(phase), dtype=torch.uint8, device="cuda")
    del marker


def run(args):
    from fastmace.model.compile import configure_allocator

    configure_allocator(enabled=True)
    import torch

    import fastmace as fm

    scan = _module("_capacity_scan", TOOLS / "bench" / "capacity_scan.py")
    frontends = _module("_ab_frontends", TOOLS / "ab_frontends.py")
    name, cell = scan.base_cell(args.structure)
    atoms = cell.repeat(tuple(int(n) for n in args.repeat.lower().split("x")))
    atoms.set_pbc(True)
    frontends.thermalize(atoms, args.temperature_K, args.seed)
    natoms = len(atoms)
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    from ab_rewrites import constant_overrides

    for (module, name_), value in constant_overrides(args.constant).items():
        importlib.import_module(module)
        setattr(sys.modules[module], name_, value)
    record = dict(tool="memory_profile", case=args.case, created_utc=datetime.now(timezone.utc).isoformat(),
                  host=os.uname().nodename, fastmace_commit=_commit(), torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(), model=args.model, structure=name, repeat=args.repeat,
                  natoms=natoms, precision=args.precision, warmup=args.warmup, steps=args.steps,
                  allocator=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), constants=list(args.constant),
                  status="running")
    torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="python",
                                             max_entries=args.max_entries)
    times, sim, edges, extra = {}, None, None, {}
    started = time.perf_counter()
    try:
        mark(torch, "load")
        model = fm.load(args.model)
        torch.cuda.synchronize()
        times["load_s"] = time.perf_counter() - started
        mark(torch, "build")
        if args.case == "md":
            field = fm.ForceField(model, atoms, precision=args.precision)
            sim = fm.Simulation(atoms, field, seed=args.seed, integrator=fm.integrators.Langevin(
                timestep_fs=1.0, temperature_K=args.temperature_K, friction_per_fs=0.002))
            torch.cuda.synchronize()
            times["build_s"] = time.perf_counter() - started
            record["plan_text"] = sim.plan.describe()
            mark(torch, "warmup")
            sim.run(steps=args.warmup)
            torch.cuda.synchronize()
            mark(torch, "steps")
            sim.run(steps=args.steps)
            torch.cuda.synchronize()
            times["run_s"] = time.perf_counter() - started
            engine = sim._engine
            ff = engine.force_field
            nl = ff.neighbours
            edges = int(ff.edge_capacity)
            extra = dict(edge_capacity=edges, edges_at_cutoff=int(ff.edge_count), candidates=int(nl.num_candidates),
                         neighbour_buffers={k: int(t.numel() * t.element_size()) for k, t in nl.buffers().items()},
                         capture_memory=list(getattr(engine, "capture_memory", ())),
                         pool_bytes=engine._pool_bytes() if hasattr(engine, "_pool_bytes") else None,
                         edge_sizes_used=sorted(int(e) for e in getattr(engine, "edge_sizes_used", ())),
                         force_graphs=getattr(engine.config, "force_graphs", None))
        else:
            from fastmace.ase.calculator import FastMACECalculator

            atoms.calc = FastMACECalculator(model, precision=args.precision)
            mark(torch, "call")
            forces = atoms.get_forces()
            atoms.get_stress()
            torch.cuda.synchronize()
            times["call_s"] = time.perf_counter() - started
            record["max_force_eV_per_A"] = float(abs(forces).max())
            plan = atoms.calc.plan
            record["plan_text"] = plan.describe() if plan is not None else None
            evaluator = atoms.calc._evaluator
            edges = int(evaluator.edge_capacity)
            extra = dict(edge_capacity=edges, edges_at_cutoff=int(evaluator.edge_count))
        record["status"] = "ok"
    except Exception as error:  # noqa: BLE001  (a failing size is a result)
        record["status"] = "oom" if "out of memory" in str(error) else "error"
        record["failure"] = f"{type(error).__name__}: {str(error)[:2000]}"
    torch.cuda.synchronize()
    mark(torch, "end")
    record["times"] = times
    record["extra"] = extra
    record["peaks"] = dict(max_allocated=int(torch.cuda.max_memory_allocated()),
                           max_reserved=int(torch.cuda.max_memory_reserved()),
                           allocated=int(torch.cuda.memory_allocated()), reserved=int(torch.cuda.memory_reserved()))
    free, total = torch.cuda.mem_get_info()
    record["peaks"]["outside_allocator"] = int(total - free - torch.cuda.memory_reserved())
    record["peaks"]["total"] = int(total)
    snapshot = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    args.output.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output / f"snapshot-{args.case}-{name}-{natoms}.pickle.gz", "wb") as handle:
        pickle.dump(snapshot, handle)  # for torch.cuda._memory_viz and offline re-analysis (--reanalyse)
    record["analysis"] = analyse(snapshot, natoms, edges or 0, args.top, args.max_entries)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"memprofile-{args.case}-{name}-{natoms}.json"
    path.write_text(json.dumps(record, indent=1, default=str) + "\n")
    print_summary(record)
    print(f"wrote {path}")
    if sim is not None:
        sim.close()


def _commit():
    path = Path(ROOT) / ".source_commit"
    if path.exists():
        return path.read_text().split()[0]
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def print_summary(record):
    n, a, p = record["natoms"], record["analysis"], record["peaks"]
    print(f"{record['case']} {record['structure']} {n} atoms: {record['status']}; peak allocated "
          f"{p['max_allocated'] / 2**30:.2f} GiB ({p['max_allocated'] / n / 1e3:.1f} kB/atom, phase "
          f"{a['peak_phase']}), peak reserved {p['max_reserved'] / 2**30:.2f} GiB "
          f"({p['max_reserved'] / n / 1e3:.1f} kB/atom, phase {a['peak_reserved_phase']}), outside the allocator "
          f"{p['outside_allocator'] / 2**30:.2f} GiB; trace {a['trace_entries']} entries"
          f"{' (FULL: early history lost)' if a['trace_full'] else ''}")
    for phase, row in a["peaks_by_phase"].items():
        print(f"  phase {phase:<8} peak allocated {row['allocated'] / 2**30:6.2f} GiB, reserved "
              f"{row['reserved'] / 2**30:6.2f} GiB")
    g = a["growths"]
    print(f"  reserve grew {g['count']} times, {g['bytes'] / 2**30:.2f} GiB; "
          f"{g['while_request_fit_in_free'] / 2**30:.2f} GiB of it while the request fitted in the free bytes "
          f"(fragmentation); live peak by pool: "
          + ", ".join(f"{k} {v / 2**30:.2f}" for k, v in a["live_peak_by_pool"].items()))
    tables = ([("overall peak", a["by_owner"])] + [(f"{k} peak", v["by_owner"]) for k, v in a["at_phase_peaks"].items()]
              + [(f"peak of pool {k}", v["by_owner"]) for k, v in a.get("at_pool_peaks", {}).items()])
    for title, table in tables:
        print(f" live at the {title}:")
        for row in table:
            pools = ", ".join(f"{k} {v / 2**30:.2f}" for k, v in row["by_pool"].items())
            print(f"  {row['owner']:<44} {row['bytes'] / 2**30:7.2f} GiB {row['bytes'] / n / 1e3:7.1f} kB/atom "
                  f"({row['blocks']} blocks; {pools})")


def diff(small, large, phase=None):
    a, b = (json.loads(Path(p).read_text()) for p in (small, large))
    na, nb = a["natoms"], b["natoms"]
    ta, tb = (r["analysis"] if phase is None else r["analysis"]["at_phase_peaks"][phase] for r in (a, b))
    for title, key in (("owner", "by_owner"), ("site", "by_site")):
        name = "owner" if key == "by_owner" else "site"
        ra = {r[name]: r["bytes"] for r in ta[key]}
        rb = {r[name]: r["bytes"] for r in tb[key]}
        print(f"\nat the {phase or 'overall'} peak, per {title}, between {na} and {nb} atoms: slope (B/atom), "
              f"intercept (MiB), at {nb} (GiB)")
        for k in sorted(set(ra) | set(rb), key=lambda k: -rb.get(k, 0))[:40]:
            slope = (rb.get(k, 0) - ra.get(k, 0)) / (nb - na)
            print(f"  {slope:9.0f} {(rb.get(k, 0) - slope * nb) / 2**20:9.1f} {rb.get(k, 0) / 2**30:7.2f}  {k}")
    print()
    for k in ("max_allocated", "max_reserved", "outside_allocator"):
        slope = (b["peaks"][k] - a["peaks"][k]) / (nb - na)
        print(f"  {k:<20} {slope:9.0f} B/atom, intercept {(b['peaks'][k] - slope * nb) / 2**20:9.1f} MiB")


def main(argv=None):
    args = parse_args(argv)
    if args.diff:
        diff(*args.diff, phase=args.phase)
        return
    if args.reanalyse:
        record = json.loads(args.reanalyse.read_text())
        name = args.reanalyse.name.replace("memprofile-", "snapshot-").replace(".json", ".pickle.gz")
        with gzip.open(args.reanalyse.with_name(name), "rb") as handle:
            snapshot = pickle.load(handle)  # noqa: S301  (our own file)
        record["analysis"] = analyse(snapshot, record["natoms"], record["extra"].get("edge_capacity") or 0,
                                     args.top, args.max_entries)
        args.reanalyse.write_text(json.dumps(record, indent=1, default=str) + "\n")
        print_summary(record)
        return
    if not args.repeat or not args.output:
        raise SystemExit("--repeat and --output are required")
    run(args)


if __name__ == "__main__":
    main()
