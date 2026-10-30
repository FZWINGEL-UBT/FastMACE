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
        q.add_argument("--structure", default="NaCl",
                       help="Benchmark name (under FASTMACE_STRUCTURES), a file, or SrTiO3 (cubic, a = 3.905 A)")
        q.add_argument("--model", default="mh1-r2scan", help="Label (its file under FASTMACE_MODELS)")
        q.add_argument("--precision", choices=("float32", "float64"), default="float32")
        q.add_argument("--output", required=True, type=Path, help="Directory (created if missing)")
        q.add_argument("--steps", type=int, default=200, help="timed steps after the warm-up")
        q.add_argument("--warmup", type=int, default=20, help="steps before the timed ones (captures)")
        q.add_argument("--timestep-fs", type=float, default=1.0)
        q.add_argument("--temperature-K", type=float, default=300.0)
        q.add_argument("--seed", type=int, default=1)
        q.add_argument("--rewrites", nargs="*", default=[], metavar="FIELD=VALUE",
                       help="RewriteConfig overrides on the default plan (candidates; the scan itself takes none)")
        q.add_argument("--tag", default="", help="suffix of the point files (a candidate's points)")
        q.add_argument("--neighbours", nargs="*", default=[], metavar="FIELD=VALUE",
                       help="NeighbourConfig overrides on the default plan, e.g. edge_bucket_fraction=0.15")
        q.add_argument("--engine", nargs="*", default=[], metavar="FIELD=VALUE",
                       help="EngineConfig overrides on the default plan (candidates), e.g. force_graphs=one")
    point = sub.choices["point"]
    point.add_argument("--repeat", required=True, help="NXxNYxNZ")
    point.add_argument("--trace", action="store_true",
                       help="sample free and reserved device memory at 20 Hz in a thread and log FastMACE's log lines "
                            "with times (CAP-T03: memory outside the allocator per build step)")
    point.add_argument("--diagnose", action="store_true",
                       help="record the allocator's history; on a failure, list the largest live blocks and where "
                            "they were allocated (run with CUDA_LAUNCH_BLOCKING=1 to stop at the failing call); "
                            "the fallback refuses, so the first failure is the one raised")
    scan = sub.choices["scan"]
    scan.add_argument("--start-atoms", type=int, default=4000)
    scan.add_argument("--growth", type=float, default=1.5, help="atom ratio between sizes before the first failure")
    scan.add_argument("--max-atoms", type=int, default=2_000_000)
    scan.add_argument("--budget-minutes", type=float, default=13.0)
    scan.add_argument("--point-timeout-s", type=float, default=900.0)
    return p.parse_args(argv)


# --------------------------------------------------------------------------- supercells


# A capacity point counts only if its list has the structure's edges per atom to within this share, and its
# forces are not all zero (31 October 2026: the GPU pair screen silently did not launch at 1.73M and 2.0M atoms,
# and those points "ran" with no edges).
EDGES_PER_ATOM_TOLERANCE = 0.05


def reference_edges_per_atom(structure, cutoff):
    """Directed edges within ``cutoff`` per atom of ``structure``'s unit cell repeated to at least 2 * cutoff + 1
    Angstrom per lattice plane spacing (ASE's neighbour list on the CPU): the value a correct list must have."""
    import math

    import numpy as np
    from ase.neighborlist import neighbor_list

    _, cell = base_cell(structure)
    spacings = 1.0 / np.linalg.norm(np.linalg.inv(cell.cell.array), axis=0)
    small = cell.repeat([max(1, math.ceil((2 * cutoff + 1) / s)) for s in spacings])
    return len(neighbor_list("i", small, cutoff)) / len(small)


def check_point(structure, cutoff, natoms, edges, max_abs_force, reference=None):
    """``(valid, detail)`` of a capacity point: its edges per atom against ``reference_edges_per_atom`` (within
    ``EDGES_PER_ATOM_TOLERANCE``; thermal or rattled cells differ by a few per cent) and a nonzero largest force."""
    reference = reference_edges_per_atom(structure, cutoff) if reference is None else reference
    per_atom = edges / natoms if natoms else 0.0
    deviation = abs(per_atom - reference) / reference if reference else float("inf")
    valid = deviation <= EDGES_PER_ATOM_TOLERANCE and max_abs_force is not None and max_abs_force > 0
    return valid, dict(edges_per_atom=per_atom, reference_edges_per_atom=reference, deviation=deviation,
                       max_abs_force=max_abs_force, tolerance=EDGES_PER_ATOM_TOLERANCE)


def base_cell(structure):
    """(name, unit cell Atoms) of a benchmark name or file, or ``SrTiO3``: cubic Pm-3m SrTiO3 with a = 3.905 A,
    the cell of symmetrix-xl's capacity probe (benchmarks/srtio3_capacity_probe.py)."""
    if structure == "SrTiO3":
        from ase import Atoms

        cell = Atoms("SrTiO3", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5),
                                                 (0, 0.5, 0.5)], cell=[3.905] * 3, pbc=True)
        return "SrTiO3", cell
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


def _causes(error, limit=6):
    """The errors behind ``error`` (``__cause__``, else ``__context__``): a fallback that fails again hides the
    first failure there."""
    found, seen = [], {id(error)}
    error = error.__cause__ or error.__context__
    while error is not None and id(error) not in seen and len(found) < limit:
        seen.add(id(error))
        found.append(dict(type=f"{type(error).__module__}.{type(error).__name__}", message=str(error)[:2000],
                          stage=getattr(error, "fastmace_stage", None), frames=_fastmace_frames(error.__traceback__)))
        error = error.__cause__ or error.__context__
    return found


class _Tracer:
    """``--trace``: [seconds, free, reserved, allocated] every 50 ms from a thread, and FastMACE's log records
    [seconds, logger, message]. Free memory comes from ``cudaMemGetInfo`` (no stream synchronisation)."""

    def __init__(self, torch):
        import logging
        import threading

        self.torch, self.start, self.samples, self.events = torch, time.perf_counter(), [], []
        self._stop = threading.Event()
        tracer = self

        class Handler(logging.Handler):
            def emit(self, entry):
                tracer.events.append([time.perf_counter() - tracer.start, entry.name, entry.getMessage()[:300]])

        self.handler = Handler(level=logging.INFO)
        self.logger = logging.getLogger("fastmace")
        self.level = self.logger.level
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.handler)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        cuda = self.torch.cuda
        while not self._stop.wait(0.05):
            try:
                free, _ = cuda.mem_get_info()
                self.samples.append([time.perf_counter() - self.start, int(free), int(cuda.memory_reserved()),
                                     int(cuda.memory_allocated())])
            except Exception:  # noqa: BLE001  (after a sticky error)
                return

    def stop(self):
        self._stop.set()
        self.thread.join(timeout=2)
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self.level)
        return dict(samples=self.samples, events=self.events)


def _value(text):
    """A command-line config value: true/false as bools, integers as ints, decimals as floats, else the string."""
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def _tried(message):
    match = re.search(r"Tried to allocate ([\d.]+ [KMGT]iB)", message)
    return match.group(1) if match else None


def _live_blocks(torch, limit=25):
    """The largest live allocations of the caching allocator, with their innermost FastMACE frames, and the
    total live bytes per allocating FastMACE line (needs ``_record_memory_history``)."""
    root = str(Path(fastmace.__file__).resolve().parents[1])
    snapshot = torch.cuda.memory._snapshot()
    blocks, by_site = [], {}
    for segment in snapshot["segments"]:
        for block in segment["blocks"]:
            if block["state"] != "active_allocated":
                continue
            frames = [f for f in block.get("frames", ()) if f["filename"].startswith(root)]
            site = (f"{os.path.relpath(frames[0]['filename'], root)}:{frames[0]['line']} {frames[0]['name']}"
                    if frames else "outside FastMACE")
            by_site[site] = by_site.get(site, 0) + block["size"]
            blocks.append((block["size"], site, [f"{os.path.relpath(f['filename'], root)}:{f['line']} {f['name']}"
                                                 for f in frames[:6]]))
    blocks.sort(key=lambda b: -b[0])
    return dict(largest=[dict(bytes=b[0], site=b[1], frames=b[2]) for b in blocks[:limit]],
                by_site=sorted(([site, size] for site, size in by_site.items()), key=lambda x: -x[1])[:limit],
                live_bytes=sum(b[0] for b in blocks), segments_bytes=sum(s["total_size"] for s in snapshot["segments"]))


def run_point(args):
    from fastmace.model.compile import configure_allocator

    # Before the first CUDA call, as the engine's process constraints require; off with expandable_segments=off.
    configure_allocator(enabled="expandable_segments=off" not in args.engine)
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
    record["rewrites"] = dict(item.split("=", 1) for item in args.rewrites)
    record["engine"] = {key: _value(value) for key, value in (item.split("=", 1) for item in args.engine)}
    record["neighbours"] = {key: _value(value) for key, value in (item.split("=", 1) for item in args.neighbours)}
    record["environment"] = {key: os.environ.get(key) for key in ("CUDA_LAUNCH_BLOCKING", "PYTORCH_CUDA_ALLOC_CONF")}
    if args.diagnose:
        torch.cuda.memory._record_memory_history(max_entries=200_000)
    phase, caught, sim = "load", [], None
    tracer = _Tracer(torch) if args.trace else None
    times = {}

    def peaks():
        found = dict(max_allocated=int(torch.cuda.max_memory_allocated()),
                     max_reserved=int(torch.cuda.max_memory_reserved()),
                     allocated=int(torch.cuda.memory_allocated()), reserved=int(torch.cuda.memory_reserved()))
        stats = torch.cuda.memory_stats()
        found.update({key: int(stats.get(f"{key}.all.peak", stats.get(key, 0)))
                      for key in ("inactive_split_bytes", "active_bytes", "reserved_bytes")},
                     alloc_retries=int(stats.get("num_alloc_retries", 0)),
                     segments=int(stats.get("segment.all.current", 0)))
        try:  # device memory in use outside PyTorch's allocator (libraries, graphs, the context)
            free_now, total_now = torch.cuda.mem_get_info()
            found.update(cuda_free=int(free_now), outside_allocator=int(total_now - free_now - found["reserved"]))
        except Exception:  # noqa: BLE001  (after a sticky error)
            pass
        return found

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
                    rebuilds=int(getattr(nl, "rebuilds", 0)),
                    captures=list(getattr(engine, "capture_memory", ())),
                    force_graphs=getattr(engine.config, "force_graphs", None),
                    force_graph_releases=getattr(engine, "force_graph_releases", None),
                    force_graph_floor=getattr(engine, "force_graph_floor", None),
                    recaptures=getattr(engine, "recaptures", None),
                    force_bytes_per_edge=getattr(engine, "_force_bytes_per_edge", None),
                    pool_bytes=engine._pool_bytes() if hasattr(engine, "_pool_bytes") else None,
                    neighbour_list=neighbour_memory(nl))

    def neighbour_memory(nl):
        """The neighbour list's device bytes: its buffers by name, per slot, and the screen's output buffer."""
        sizes = {name: int(t.numel() * t.element_size()) for name, t in nl.buffers().items()}
        screen = getattr(nl, "_pair_search_state", {}).get("buffer")
        return dict(buffers_bytes=sum(sizes.values()), buffers=sizes,
                    bytes_per_slot=sum(v for k, v in sizes.items() if k.startswith(("candidate", "edge", "unit",
                                                                                      "shifts"))) / nl.capacity,
                    screen_buffer_bytes=0 if screen is None else int(screen.numel() * screen.element_size()))

    def rebuild_transient():
        """One more neighbour build at the committed positions, after the timed steps: its peak above the live
        memory (the build's temporaries) and the reserved memory it added."""
        engine = sim._engine
        nl = engine.force_field.neighbours
        torch.cuda.synchronize()
        allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        grew = bool(nl.build(engine.state.positions))
        torch.cuda.synchronize()
        return dict(peak_above_live_bytes=int(torch.cuda.max_memory_allocated() - allocated),
                    reserved_added_bytes=int(torch.cuda.memory_reserved() - reserved), grew=grew,
                    candidates=int(nl.num_candidates))

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            started = time.perf_counter()
            from fastmace.config import RewriteConfig

            field = fm.ForceField(fm.load(args.model), atoms, precision=args.precision,
                                  **({"rewrites": RewriteConfig(**record["rewrites"])} if args.rewrites else {}))
            torch.cuda.synchronize()
            times["load_s"] = time.perf_counter() - started
            record["phases"]["load"] = peaks()
            phase = "build"
            started = time.perf_counter()
            from fastmace.config import EngineConfig, FallbackConfig, NeighbourConfig

            sim = fm.Simulation(atoms, field, seed=args.seed,
                                **({"fallback": FallbackConfig(fallback="refuse")} if args.diagnose else {}),
                                **({"engine": EngineConfig(**record["engine"])} if args.engine else {}),
                                **({"neighbours": NeighbourConfig(**record["neighbours"])} if args.neighbours else {}),
                                integrator=fm.integrators.Langevin(
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
            ff = sim._engine.force_field
            state_forces = getattr(getattr(sim._engine, "state", None), "forces", None)
            valid, detail = check_point(args.structure, float(ff.neighbours.cutoff), len(atoms), int(ff.edge_count),
                                        None if state_forces is None else float(state_forces.abs().max()))
            record["check"] = detail
            record["status"] = "ok" if valid else "invalid"
            try:
                record["rebuild_transient"] = rebuild_transient()
            except Exception as problem:  # noqa: BLE001  (a measurement after the result)
                record["rebuild_transient"] = f"failed: {type(problem).__name__}: {str(problem)[:300]}"
    except Exception as error:  # noqa: BLE001  (a failing size is a result)
        message = str(error)
        oom = type(error).__name__ == "OutOfMemoryError" or "out of memory" in message
        record["status"] = "oom" if oom else "error"
        record["failure"] = dict(phase=phase, type=f"{type(error).__module__}.{type(error).__name__}",
                                 message=message[:4000], tried_to_allocate=_tried(message),
                                 stage=getattr(error, "fastmace_stage", None),
                                 frames=_fastmace_frames(error.__traceback__), memory=peaks(),
                                 causes=_causes(error))
        if args.diagnose:
            try:
                record["failure"]["live"] = _live_blocks(torch)
            except Exception as problem:  # noqa: BLE001
                record["failure"]["live"] = f"no snapshot: {problem}"
        if sim is not None:
            try:
                record["edges"] = edges()
            except Exception as problem:  # noqa: BLE001  (after a sticky CUDA error)
                record["edges"] = f"not readable: {type(problem).__name__}"
    finally:
        if tracer is not None:
            record["trace"] = tracer.stop()
        record["times"] = times
        record["warnings"] = [f"{w.category.__name__}: {w.message}" for w in caught][:20]
        if sim is not None:
            try:
                sim.close()
            except Exception:  # noqa: BLE001
                pass
    args.output.mkdir(parents=True, exist_ok=True)
    path = point_path(args.output, name, args.precision, len(atoms), args.tag)
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


def point_path(output, name, precision, atoms, tag=""):
    return output / f"point-{name}-{precision}-{atoms}{tag}.json"


def scan(args):
    name, cell = base_cell(args.structure)
    sizes = supercells([float(x) for x in cell.cell.lengths()], len(cell), args.max_atoms)
    deadline = time.monotonic() + 60 * args.budget_minutes
    log = args.output / f"scan-{name}-{args.precision}{args.tag}.log"
    args.output.mkdir(parents=True, exist_ok=True)

    def outcome(index):
        path = point_path(args.output, name, args.precision, sizes[index][0], args.tag)
        if path.is_file():
            return json.loads(path.read_text())["status"] == "ok"
        if time.monotonic() > deadline:
            return None
        command = [sys.executable, __file__, "point", "--structure", args.structure, "--model", args.model,
                   "--precision", args.precision, "--output", str(args.output), "--steps", str(args.steps),
                   "--warmup", str(args.warmup), "--timestep-fs", str(args.timestep_fs),
                   "--temperature-K", str(args.temperature_K), "--seed", str(args.seed),
                   "--repeat", repeat_text(sizes[index][1]), f"--tag={args.tag}"]
        if args.rewrites:
            command += ["--rewrites", *args.rewrites]
        if args.engine:
            command += ["--engine", *args.engine]
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
    summary = dict(structure=name, precision=args.precision, rewrites=args.rewrites, largest_passing=best,
                   smallest_failing=worst)
    (args.output / f"scan-{name}-{args.precision}{args.tag}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"scan: done {name} {args.precision}{args.tag}: largest passing {best}, smallest failing {worst}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "point":
        run_point(args)
    else:
        scan(args)


if __name__ == "__main__":
    main()
