"""Capacity scan: the largest supercell that runs NVT MD on this GPU without running out of memory.

Two commands:

- ``point``: one supercell in this process. ``fm.Simulation`` with ``fm.integrators.Langevin`` and the
  planner's default plan at ``--precision`` (graph tier, pipelined where the plan says so), ``--warmup``
  steps (the captures), then ``--steps`` timed steps. Recorded per phase (``load``, ``build``: planning,
  rewrites, self-checks, first forces and captures; ``warmup``; ``steady``): the peak
  ``max_memory_allocated`` / ``max_memory_reserved`` (the peaks are reset after the build), the free
  memory at the start (CUDA's and the Windows ``nvidia-smi`` view, which includes the desktop), the edges
  (within the cutoff at the start, the candidates within cutoff + skin, the pair-buffer capacity, the edge
  bucket and the padded sizes the force graphs ran), rule 12's estimate and decision, any warnings, and
  the ms per step. A failure records its phase, the error, the allocation torch reports and the innermost
  FastMACE frames (the buffer and stage that ran out). Output: ``<output>/point-<name>-<precision>-<atoms>.json``.
- ``scan``: ``point`` in a fresh subprocess per size (a clean CUDA context: an out-of-memory error leaves no
  cached blocks behind) over isotropic supercells of ``--structure`` ordered by atom count. Sizes grow by
  ``--growth`` in atoms from ``--start-atoms`` until one fails, then bisect between the largest passing and
  the smallest failing size. Points already in ``--output`` are reused, so a scan continues across GPU
  reservations; ``--budget-minutes`` stops it before the next point would start past the budget. Prints
  ``scan: done`` with the largest passing size, or ``scan: more``.

Run inside a local GPU reservation (``tools/ci/local_gpu.py run``) on an otherwise idle GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

TOOLS = Path(__file__).resolve().parents[1]
WINDOWS_SMI = "/mnt/c/Windows/System32/nvidia-smi.exe"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("point", "scan"):
        q = sub.add_parser(name)
        q.add_argument("--structure", default="NaCl", help="Benchmark name (under FASTMACE_STRUCTURES) or a file")
        q.add_argument("--model", default="mh1-r2scan", help="Label (its file under FASTMACE_MODELS)")
        q.add_argument("--precision", choices=("float32", "float64"), default="float32")
        q.add_argument("--output", required=True, type=Path, help="Directory (created if missing)")
        q.add_argument("--steps", type=int, default=200, help="timed steps after the warm-up")
        q.add_argument("--warmup", type=int, default=20, help="steps before the timed ones (captures)")
        q.add_argument("--timestep-fs", type=float, default=1.0)
        q.add_argument("--temperature-K", type=float, default=300.0)
        q.add_argument("--seed", type=int, default=1)
    point = sub.choices["point"]
    point.add_argument("--repeat", required=True, help="NXxNYxNZ")
    scan = sub.choices["scan"]
    scan.add_argument("--start-atoms", type=int, default=4000)
    scan.add_argument("--growth", type=float, default=1.5, help="atom ratio between sizes before the first failure")
    scan.add_argument("--max-atoms", type=int, default=2_000_000)
    scan.add_argument("--budget-minutes", type=float, default=13.0)
    scan.add_argument("--point-timeout-s", type=float, default=900.0)
    return p.parse_args(argv)


# --------------------------------------------------------------------------- supercells


def base_cell(structure):
    """(name, unit cell Atoms) of a benchmark name or file."""
    sys.path.insert(0, str(TOOLS))
    import ab_step_time

    return ab_step_time.read_structure(f"{structure}@1x1x1" if "@" not in structure else structure, None)


def supercells(cell_lengths, natoms_cell, max_atoms):
    """Repeats giving roughly isotropic supercells, by increasing atom count: for a growing target length L,
    each axis repeats ``round(L / length)`` times (at least once); one repeat per atom count."""
    found = {}
    length = min(cell_lengths)
    longest = (max_atoms / natoms_cell) ** (1 / 3) * max(cell_lengths) + max(cell_lengths)
    while length <= longest:
        repeat = tuple(max(1, round(length / a)) for a in cell_lengths)
        atoms = natoms_cell * math.prod(repeat)
        if atoms <= max_atoms:
            found.setdefault(atoms, repeat)
        length *= 1.01
    return [(atoms, found[atoms]) for atoms in sorted(found)]


def repeat_text(repeat):
    return "x".join(str(n) for n in repeat)


# --------------------------------------------------------------------------- one point


def windows_memory():
    """(used MiB, total MiB) of the GPU in Windows' nvidia-smi (the desktop included), or None."""
    try:
        out = subprocess.check_output([WINDOWS_SMI, "--query-gpu=memory.used,memory.total",
                                       "--format=csv,noheader,nounits"], text=True, timeout=30)
        used, total = (int(v) for v in out.strip().splitlines()[0].split(","))
        return {"used_MiB": used, "total_MiB": total}
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _fastmace_frames(tb, limit=8):
    """The innermost frames in FastMACE (and the last one outside it): 'file:line function: source'."""
    root = str(Path(fastmace.__file__).resolve().parents[1])
    frames = traceback.extract_tb(tb)
    ours = [f for f in frames if f.filename.startswith(root)]
    picked = ours[-limit:] + ([frames[-1]] if frames and (not ours or frames[-1] is not ours[-1]) else [])
    return [f"{os.path.relpath(f.filename, root) if f.filename.startswith(root) else f.filename}:{f.lineno} "
            f"{f.name}: {(f.line or '').strip()}" for f in picked]


def _tried(message):
    match = re.search(r"Tried to allocate ([\d.]+ [KMGT]iB)", message)
    return match.group(1) if match else None


def run_point(args):
    from fastmace.model.compile import configure_allocator

    configure_allocator()  # before the first CUDA call, as the engine's process constraints require
    import torch

    import fastmace as fm

    name, cell = base_cell(args.structure)
    repeat = tuple(int(n) for n in args.repeat.lower().split("x"))
    atoms = cell.repeat(repeat)
    atoms.set_pbc(True)
    sys.path.insert(0, str(TOOLS))
    import ab_step_time

    spec = importlib.util.spec_from_file_location("_ab_frontends", TOOLS / "ab_frontends.py")
    frontends = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frontends)
    frontends.thermalize(atoms, args.temperature_K, args.seed)
    record = dict(tool="capacity_scan", created_utc=datetime.now(timezone.utc).isoformat(),
                  host=os.uname().nodename, python=sys.executable, fastmace_commit=ab_step_time.fastmace_commit(),
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(), model=args.model, structure=name,
                  precision=args.precision, repeat=repeat_text(repeat), natoms=len(atoms),
                  volume_A3=float(atoms.get_volume()), steps=args.steps, warmup=args.warmup,
                  timestep_fs=args.timestep_fs, temperature_K=args.temperature_K, status="running", phases={})
    free, total = torch.cuda.mem_get_info()
    record["start"] = dict(cuda_free_bytes=int(free), cuda_total_bytes=int(total), windows=windows_memory())
    phase, caught, sim = "load", [], None
    times = {}

    def peaks():
        return dict(max_allocated=int(torch.cuda.max_memory_allocated()),
                    max_reserved=int(torch.cuda.max_memory_reserved()),
                    allocated=int(torch.cuda.memory_allocated()), reserved=int(torch.cuda.memory_reserved()))

    def edges():
        engine = getattr(sim, "_engine", None)
        ff = getattr(engine, "force_field", None)
        nl = getattr(ff, "neighbours", None)
        if nl is None:
            return {}
        return dict(at_cutoff_initial=getattr(engine, "initial_edges", None),
                    at_cutoff_now=int(ff.edge_count), candidates_cutoff_plus_skin=int(nl.num_candidates),
                    capacity=int(ff.edge_capacity), skin_A=float(nl.skin), cutoff_A=float(nl.cutoff),
                    edge_bucket=getattr(engine, "edge_bucket", None),
                    padded_sizes=sorted(int(e) for e in getattr(engine, "edge_sizes_used", ())),
                    rebuilds=int(getattr(nl, "rebuilds", 0)))

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            started = time.perf_counter()
            field = fm.ForceField(fm.load(args.model), atoms, precision=args.precision)
            torch.cuda.synchronize()
            times["load_s"] = time.perf_counter() - started
            record["phases"]["load"] = peaks()
            phase = "build"
            started = time.perf_counter()
            sim = fm.Simulation(atoms, field, seed=args.seed, integrator=fm.integrators.Langevin(
                timestep_fs=args.timestep_fs, temperature_K=args.temperature_K, friction_per_fs=0.002))
            torch.cuda.synchronize()
            times["build_s"] = time.perf_counter() - started
            record["phases"]["build"] = peaks()
            plan = sim.plan
            record["plan"] = dict(engine=type(sim._engine).__name__, predicted_memory_bytes=plan.predicted_memory_bytes,
                                  memory=list(plan.decision("memory")) if _has(plan, "memory") else None,
                                  rule_12=[list(d) for d in plan.decisions if "rule 12" in str(d[2])],
                                  text=plan.describe())
            record["edges"] = edges()
            torch.cuda.reset_peak_memory_stats()
            phase = "warmup"
            started = time.perf_counter()
            sim.run(steps=args.warmup)
            torch.cuda.synchronize()
            times["warmup_s"] = time.perf_counter() - started
            record["phases"]["warmup"] = peaks()
            torch.cuda.reset_peak_memory_stats()
            phase = "steady"
            started = time.perf_counter()
            sim.run(steps=args.steps)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            times["steady_s"] = seconds
            record["phases"]["steady"] = peaks()
            record["ms_per_step"] = 1e3 * seconds / args.steps
            record["edges"] = edges()
            record["status"] = "ok"
    except Exception as error:  # noqa: BLE001  (a failing size is a result)
        message = str(error)
        record["status"] = "oom" if type(error).__name__ == "OutOfMemoryError" or "out of memory" in message else "error"
        record["failure"] = dict(phase=phase, type=f"{type(error).__module__}.{type(error).__name__}",
                                 message=message[:4000], tried_to_allocate=_tried(message),
                                 stage=getattr(error, "fastmace_stage", None),
                                 frames=_fastmace_frames(error.__traceback__), memory=peaks())
        if sim is not None:
            record["edges"] = edges()
    finally:
        record["times"] = times
        record["warnings"] = [f"{w.category.__name__}: {w.message}" for w in caught][:20]
        if sim is not None:
            try:
                sim.close()
            except Exception:  # noqa: BLE001
                pass
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"point-{name}-{args.precision}-{len(atoms)}.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    peak = record["phases"].get("steady", {}).get("max_reserved")
    print(f"point {name} {args.precision} {len(atoms)} atoms ({record['repeat']}): {record['status']}"
          + (f", {record['ms_per_step']:.2f} ms/step, steady peak reserved {peak / 2**30:.2f} GiB" if peak else "")
          + (f", failed in {record['failure']['phase']}: {record['failure']['message'][:200]}"
             if "failure" in record else "") + f" ({path})", flush=True)
    return record


def _has(plan, key):
    return any(entry[0] == key for entry in plan.decisions)


# --------------------------------------------------------------------------- the scan


def point_path(output, name, precision, atoms):
    return output / f"point-{name}-{precision}-{atoms}.json"


def scan(args):
    name, cell = base_cell(args.structure)
    sizes = supercells([float(x) for x in cell.cell.lengths()], len(cell), args.max_atoms)
    deadline = time.monotonic() + 60 * args.budget_minutes
    log = args.output / f"scan-{name}-{args.precision}.log"
    args.output.mkdir(parents=True, exist_ok=True)

    def outcome(index):
        path = point_path(args.output, name, args.precision, sizes[index][0])
        if path.is_file():
            return json.loads(path.read_text())["status"] == "ok"
        if time.monotonic() > deadline:
            return None
        command = [sys.executable, __file__, "point", "--structure", args.structure, "--model", args.model,
                   "--precision", args.precision, "--output", str(args.output), "--steps", str(args.steps),
                   "--warmup", str(args.warmup), "--timestep-fs", str(args.timestep_fs),
                   "--temperature-K", str(args.temperature_K), "--seed", str(args.seed),
                   "--repeat", repeat_text(sizes[index][1])]
        with open(log, "a") as stream:
            stream.write(f"{datetime.now(timezone.utc).isoformat()} start {sizes[index]}\n")
            stream.flush()
            try:
                done = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                      timeout=args.point_timeout_s)
                code = done.returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
            stream.write(f"{datetime.now(timezone.utc).isoformat()} end {sizes[index]} exit={code}\n")
        if not path.is_file():  # the process died before writing (a crash or a timeout): a failing size
            path.write_text(json.dumps(dict(tool="capacity_scan", structure=name, precision=args.precision,
                                            natoms=sizes[index][0], repeat=repeat_text(sizes[index][1]),
                                            status="crash", exit=code), indent=2) + "\n")
        return json.loads(path.read_text())["status"] == "ok"

    low = next((i for i, (atoms, _) in enumerate(sizes) if atoms >= args.start_atoms), len(sizes) - 1)
    passing, failing = None, None
    index = low
    while failing is None:  # grow
        result = outcome(index)
        if result is None:
            print("scan: more", flush=True)
            return
        if not result:
            failing = index
            break
        passing = index
        if index == len(sizes) - 1:
            break
        want = sizes[index][0] * args.growth
        index = next((i for i in range(index + 1, len(sizes)) if sizes[i][0] >= want), len(sizes) - 1)
    if failing is not None and passing is None:  # the start fails: walk down
        index = failing
        while index > 0 and passing is None:
            index = next((i for i in range(index - 1, -1, -1) if sizes[i][0] <= sizes[index][0] / args.growth), 0)
            result = outcome(index)
            if result is None:
                print("scan: more", flush=True)
                return
            if result:
                passing = index
            else:
                failing = index
    while failing is not None and passing is not None and failing - passing > 1:  # bisect
        middle = (passing + failing) // 2
        result = outcome(middle)
        if result is None:
            print("scan: more", flush=True)
            return
        if result:
            passing = middle
        else:
            failing = middle
    best = sizes[passing] if passing is not None else None
    worst = sizes[failing] if failing is not None else None
    summary = dict(structure=name, precision=args.precision, largest_passing=best, smallest_failing=worst)
    (args.output / f"scan-{name}-{args.precision}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"scan: done {name} {args.precision}: largest passing {best}, smallest failing {worst}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "point":
        run_point(args)
    else:
        scan(args)


if __name__ == "__main__":
    main()
