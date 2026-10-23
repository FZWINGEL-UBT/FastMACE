"""The D3 cost on the production ramp (plan task 3b-T06; SPEC §11.2 "Cost target", §9.3 rule 11).

Two subcommands, both on one structure (NbCl5 3x3x3 by default), MH-1 hybrid float32 at 1 fs through the
minimal planner, along a temperature route that follows the anneal profile (0 -> 1000 K in 20 ps, then
3 K/ps) with faster transits between the windows of the cooling ramp (``--route``). Each engine sets its
thermostat target from its own step count before every step (as ``fastmace.protocols.schedule``'s
``TargetFill`` does), so two engines pass the same temperatures at the same steps.

``scan`` (the skin): one D3-on engine runs the route untimed and records every step's positions (float32,
unwrapped, as the engine keeps them) and cell on the GPU. The D3 list's rebuild test
(``D3PairList.needs_rebuild``, deformation-aware under NPT, at the fraction the engine tested with) is
then replayed over that trajectory for each skin of ``--skins``: rebuilds per segment. At frames of the
timed windows, each skin's D3 build (``D3Term.build``: list and kernel preparation) and its three passes
(``D3Term.evaluate`` with the virial, CUDA events) are timed. The amortised D3 cost per step of a skin is
pass time + rebuilds per step x build time. Writes ``<output>.json``.

``ab`` (the cost): a D3-off and a D3-on engine (the D3 skin from ``--d3-skin``, or rule 11's), both MD-only
plans of the same ensemble, seed and starting structure. Every segment of the route runs in blocks of
``--block`` steps that alternate ABBA between the sides, so both sides time the same segment of the ramp
with their rebuilds (MACE's and D3's) and recaptures inside the clock: the builds are amortised as they
fall. Outside the clock, before each block: the counters (MACE rebuilds, D3 builds and build time, pairs),
the cell volume, and whether a pytest process runs (recorded; the reservation is quiet). Per segment:
mean and median ms per step per side and the overhead on/off - 1. Writes ``<output>.json``.

A run is marked invalid when another process used the GPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ab_step_time  # noqa: E402

# label:T_start:T_end:steps. The anneal profile's heating (50 K/ps), its first 8 ps of cooling at 3 K/ps,
# transits at 100 K/ps, and 8 ps windows of the 3 K/ps cooling at 600 K and 300 K.
ANNEAL_ROUTE = ("heat:0:1000:20000,peak:1000:976:8000,transit:976:600:3760,cool600:600:576:8000,"
                "transit:576:300:2760,cool300:300:276:8000")


@dataclasses.dataclass(frozen=True)
class Segment:
    label: str
    start_K: float
    end_K: float
    steps: int

    def target(self, step: int) -> float:
        return self.start_K + (self.end_K - self.start_K) * step / self.steps


def parse_route(text: str) -> list[Segment]:
    out = []
    for item in text.split(","):
        label, start, end, steps = item.split(":")
        out.append(Segment(label, float(start), float(end), int(steps)))
    return out


def route_target(route, step: int) -> float:
    """The thermostat target at ``step`` (0-based) of the route; the last segment's end after it."""
    for segment in route:
        if step < segment.steps:
            return segment.target(step)
        step -= segment.steps
    return route[-1].end_K


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("scan", "ab"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--structure", default="NbCl5")
    p.add_argument("--ensemble", choices=("nvt", "npt"), default="npt")
    p.add_argument("--pressure-GPa", type=float, default=0.0)
    p.add_argument("--timestep-fs", type=float, default=1.0)
    p.add_argument("--route", default=ANNEAL_ROUTE, help="label:T_start:T_end:steps,... (K)")
    p.add_argument("--max-temperature-K", type=float, default=None, help="Workload (default: the route's max)")
    p.add_argument("--d3-skin", type=float, default=None, help="D3 skin, A (default: rule 11's)")
    p.add_argument("--block", type=int, default=1000, help="ab: steps per block")
    p.add_argument("--warmup", type=int, default=50, help="steps at the route's start before anything is recorded")
    p.add_argument("--skins", default="1.0,1.5,2.0,2.5,3.0,4.0", help="scan: D3 skins to replay, A")
    p.add_argument("--timed-segments", default="peak,cool600,cool300", help="scan: segments to time builds at")
    p.add_argument("--build-repeats", type=int, default=5)
    p.add_argument("--pass-repeats", type=int, default=50)
    p.add_argument("--seed", type=int, default=20261022)
    return p.parse_args(argv)


def pytest_running() -> list[int]:
    try:
        out = subprocess.run(["pgrep", "-f", "pytest"], capture_output=True, text=True).stdout.split()
    except OSError:
        return []
    return [int(pid) for pid in out if int(pid) != os.getpid()]


class RouteSide:
    """A GraphEngine following the route: the thermostat target from its own step count before every step."""

    def __init__(self, engine, route, torch):
        self.engine, self.route, self.torch = engine, route, torch
        self.steps = 0

    def step(self):
        self.engine.integrator.set_temperature(route_target(self.route, self.steps))
        self.engine.step()
        self.steps += 1

    def prepare_block(self):
        pass

    def finish_block(self):
        self.engine.flush()

    def synchronize(self):
        self.torch.cuda.synchronize()

    def counters(self) -> dict:
        info = self.engine.info()
        out = dict(neighbour_rebuilds=info["neighbour_rebuilds"], recaptures=info["recaptures"],
                   pool_resets=info["pool_resets"], graphs=len(info["graphs"]))
        d3 = getattr(self.engine.force_field, "d3", None)
        if d3 is not None:
            c = d3.counters()
            out.update(d3_builds=c["d3_builds"], d3_build_time_s=c["d3_build_time_s"],
                       d3_term_build_s=c["build_seconds"], d3_regrowths=c["regrowths"], d3_pairs=c["d3_pairs"],
                       d3_capacity=c["d3_capacity"], d3_skin_A=c["d3_skin_A"], d3_fraction=d3.fraction)
        state = self.engine.state
        cell = state.cell.detach().double().cpu().numpy()
        import numpy as np
        out.update(volume_A3=float(abs(np.linalg.det(cell))), step=self.steps)
        return out


def build(args, loaded, base, *, dispersion: bool, hardware, route):
    """(engine, plan record) of one side."""
    from fastmace.config import EngineConfig, IntegratorConfig, NeighbourConfig
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.planner.workload import Workload

    max_T = args.max_temperature_K or max(max(s.start_K, s.end_K) for s in route)
    field = ForceField(loaded, base, precision="float32", dispersion="d3bj" if dispersion else None)
    neighbours = NeighbourConfig(d3_skin=args.d3_skin) if args.d3_skin else NeighbourConfig()
    planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"), neighbours=neighbours)
    start_K = max(route[0].start_K, 1.0)
    extra = dict(pressure_GPa=args.pressure_GPa) if args.ensemble == "npt" else {}
    integrator = IntegratorConfig(ensemble=args.ensemble, timestep_fs=args.timestep_fs, seed=args.seed,
                                  temperature_K=start_K, **extra)
    plan = planner.plan(field, base, integrator, Workload(max_temperature_K=max_T), hardware)
    engine = planner.build(plan, field, base.copy(), temperature_K=start_K)
    decisions = {key: f"{choice} ({reason})" for key, choice, reason in plan.decisions}
    record = dict(dispersion=dispersion, skin=plan.neighbours.skin, d3_skin=plan.neighbours.d3_skin,
                  max_temperature_K=max_T, ensemble=args.ensemble,
                  d3_skin_decision=decisions.get("NeighbourConfig.d3_skin"))
    return engine, record


def common(args):
    ab_step_time.prepare_fastmace_hybrid()
    import torch

    from fastmace.config import EngineConfig
    from fastmace.model import loading
    from fastmace.planner import minimal

    others = ab_step_time.gpu_processes()
    hardware = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid")).hardware()
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    loaded = loading.load_pristine(args.model, checkpoint=str(args.checkpoint), sha256=sha)
    name, base = ab_step_time.read_structure(args.structure, None)
    route = parse_route(args.route)
    meta = dict(tool="d3_cost", command=args.command, created_utc=datetime.now(timezone.utc).isoformat(),
                host=os.uname().nodename, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                fastmace_commit=ab_step_time.fastmace_commit() or os.environ.get("FASTMACE_SOURCE_COMMIT"),
                model=args.model, checkpoint_sha256=sha,
                structure=name, natoms=len(base), ensemble=args.ensemble, timestep_fs=args.timestep_fs,
                route=[dataclasses.asdict(s) for s in route], args={k: str(v) for k, v in vars(args).items()},
                gpu_processes_at_start=others, pytest_at_start=pytest_running())
    return torch, loaded, base, route, hardware, meta


def write(args, meta):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_name(args.output.name + ".json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


# ---------------------------------------------------------------------------------------------------- scan

def scan(args):
    torch, loaded, base, route, hardware, meta = common(args)
    from fastmace.longrange.d3_term import D3Term

    started = time.perf_counter()
    engine, record = build(args, loaded, base, dispersion=True, hardware=hardware, route=route)
    meta["plan"] = record
    meta["setup_s"] = time.perf_counter() - started
    side = RouteSide(engine, route, torch)
    term = engine.force_field.d3
    total = sum(s.steps for s in route)
    n = len(base)
    positions = torch.empty((total + 1, n, 3), dtype=torch.float32, device="cuda")
    cells = torch.empty((total + 1, 3, 3), dtype=torch.float64, device="cuda")
    state = engine.state
    started = time.perf_counter()
    engine.flush()
    positions[0].copy_(state.positions)
    cells[0].copy_(state.cell)
    before = side.counters()
    for k in range(total):
        side.step()
        engine.flush()
        positions[k + 1].copy_(state.positions)
        cells[k + 1].copy_(state.cell)
    torch.cuda.synchronize()
    after = side.counters()
    meta["run"] = dict(wall_s=time.perf_counter() - started, before=before, after=after,
                       engine_fraction=term.fraction)
    volumes = torch.linalg.det(cells).abs().cpu().numpy()
    bounds, edges = [], 0
    for s in route:
        bounds.append((s.label, edges, edges + s.steps))
        edges += s.steps
    meta["volumes_A3"] = {label: [float(volumes[a]), float(volumes[b])] for label, a, b in bounds}

    numbers = term.numbers
    record_d3 = term.record
    fraction = term.fraction
    timed = [b for b in bounds if b[0] in args.timed_segments.split(",")]
    skins = [float(s) for s in args.skins.split(",")]
    results = {}
    for skin in skins:
        replay = D3Term(record_d3, numbers, cells[0].cpu().numpy(), skin=skin, dtype=torch.float32,
                        variable_cell=args.ensemble == "npt")
        cell_arg = (lambda k: cells[k]) if args.ensemble == "npt" else (lambda k: None)
        replay.build(positions[0].double(), cell_arg(0))
        rebuild_steps = []
        for k in range(1, total + 1):
            p = positions[k].double()
            if bool(replay.list.needs_rebuild(p, cell_arg(k), fraction)):
                replay.list.build(p, cell_arg(k))
                rebuild_steps.append(k)
        per_segment = {}
        for label, a, b in bounds:
            count = sum(1 for k in rebuild_steps if a < k <= b)
            per_segment.setdefault(label, dict(steps=0, rebuilds=0))
            per_segment[label]["steps"] += b - a
            per_segment[label]["rebuilds"] += count
        timings = {}
        for label, a, b in timed:
            frame = (a + b) // 2
            p = positions[frame].double()
            c = cell_arg(frame)
            replay.build(p, c)
            torch.cuda.synchronize()
            build_s = []
            for _ in range(args.build_repeats):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                replay.build(p, c)
                torch.cuda.synchronize()
                build_s.append(time.perf_counter() - t0)
            for _ in range(5):
                replay.evaluate(p, c, virial=True)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            for _ in range(args.pass_repeats):
                replay.evaluate(p, c, virial=True)
            end.record()
            torch.cuda.synchronize()
            pass_ms = start.elapsed_time(end) / args.pass_repeats
            build_ms = 1e3 * sorted(build_s)[len(build_s) // 2]
            seg = per_segment[label]
            rate = seg["rebuilds"] / seg["steps"]
            timings[label] = dict(frame=frame, pass_ms=pass_ms, build_ms=build_ms,
                                  build_ms_all=[1e3 * x for x in build_s],
                                  pairs=int(replay.list.num_pairs), rebuilds_per_step=rate,
                                  amortised_ms=pass_ms + rate * build_ms)
        results[f"{skin:g}"] = dict(skin_A=skin, rebuilds=len(rebuild_steps), segments=per_segment, timed=timings)
        print(f"skin {skin:g}: " + " ".join(
            f"{label} {seg['rebuilds']}/{seg['steps']}" for label, seg in per_segment.items()) + " | " + " ".join(
            f"{label} pass {t['pass_ms']:.2f} build {t['build_ms']:.1f} amortised {t['amortised_ms']:.2f} ms"
            for label, t in timings.items()), flush=True)
        del replay
        torch.cuda.empty_cache()
    meta["skins"] = results
    meta["gpu_processes_at_end"] = [pid for pid in ab_step_time.gpu_processes() if pid != os.getpid()]
    meta["valid"] = not meta["gpu_processes_at_start"] and not meta["gpu_processes_at_end"]
    write(args, meta)


# ---------------------------------------------------------------------------------------------------- ab

def ab(args):
    torch, loaded, base, route, hardware, meta = common(args)
    from torch._dynamo.utils import counters as dynamo_counters

    from fastmace.bench import step_time

    sides, plans, setup = {}, {}, {}
    for name, dispersion in (("off", False), ("on", True)):
        started = time.perf_counter()
        engine, plans[name] = build(args, loaded, base, dispersion=dispersion, hardware=hardware, route=route)
        sides[name] = RouteSide(engine, route, torch)
        setup[name] = time.perf_counter() - started
    meta.update(plans=plans, setup_s=setup)
    for name in sides:  # warm-up: the route's first steps, outside the clock
        for _ in range(args.warmup):
            sides[name].step()
        sides[name].finish_block()
        sides[name].synchronize()
    blocks = []
    pattern = ("off", "on", "on", "off")
    index = 0
    for segment in route:
        remaining = segment.steps - (args.warmup if segment is route[0] else 0)
        while remaining > 0:
            # Each side runs the same steps of the segment: a pair of blocks, the order alternating ABBA.
            size = min(args.block, remaining)
            order = (pattern[2 * (index % 2)], pattern[2 * (index % 2) + 1])
            for name in order:
                side = sides[name]
                compiles = ab_step_time.compile_counters(dynamo_counters)
                before = side.counters()
                pytest = pytest_running()
                ms = step_time.time_block(side, size)
                after = side.counters()
                _, recompiled = ab_step_time.compile_changes(compiles, ab_step_time.compile_counters(dynamo_counters))
                delta = {key: after[key] - before[key] for key in after
                         if isinstance(after[key], (int, float)) and key in before and not key.startswith("d3_skin")}
                blocks.append(dict(segment=segment.label, side=name, pair=index, steps=size, ms_per_step=ms,
                                   start_step=before["step"], volume_A3=after["volume_A3"], delta=delta,
                                   d3_pairs=after.get("d3_pairs"), recompiled=recompiled, pytest=pytest))
            remaining -= size
            index += 1
        summary = segment_summary(blocks, segment.label)
        print(f"{segment.label}: off {summary['off']['mean_ms']:.3f} on {summary['on']['mean_ms']:.3f} ms "
              f"overhead {100 * summary['overhead_mean']:.2f} % "
              f"(median pairs {100 * summary['overhead_pair_median']:.2f} %) "
              f"d3 builds {summary['on']['d3_builds']} volume {summary['on']['volume_end_A3']:.0f}", flush=True)
        meta["blocks"] = blocks
        write(args, meta)
    meta["segments"] = {s.label: segment_summary(blocks, s.label) for s in route}
    meta["gpu_processes_at_end"] = [pid for pid in ab_step_time.gpu_processes() if pid != os.getpid()]
    meta["valid"] = not meta["gpu_processes_at_start"] and not meta["gpu_processes_at_end"]
    meta["pytest_during_blocks"] = sorted({pid for b in blocks for pid in b["pytest"]})
    write(args, meta)


def segment_summary(blocks, label) -> dict:
    """Per side: steps-weighted mean and median ms per step, counter totals; the overhead on/off - 1."""
    import statistics

    out = {}
    for name in ("off", "on"):
        mine = [b for b in blocks if b["segment"] == label and b["side"] == name]
        steps = sum(b["steps"] for b in mine)
        mean = sum(b["ms_per_step"] * b["steps"] for b in mine) / steps
        out[name] = dict(blocks=len(mine), steps=steps, mean_ms=mean,
                         median_ms=statistics.median(b["ms_per_step"] for b in mine),
                         neighbour_rebuilds=sum(b["delta"].get("neighbour_rebuilds", 0) for b in mine),
                         recaptures=sum(b["delta"].get("recaptures", 0) for b in mine),
                         d3_builds=sum(b["delta"].get("d3_builds", 0) for b in mine),
                         d3_build_time_s=sum(b["delta"].get("d3_build_time_s", 0.0) for b in mine),
                         d3_term_build_s=sum(b["delta"].get("d3_term_build_s", 0.0) for b in mine),
                         d3_regrowths=sum(b["delta"].get("d3_regrowths", 0) for b in mine),
                         recompiled_blocks=sum(1 for b in mine if b["recompiled"]),
                         volume_end_A3=mine[-1]["volume_A3"])
    pairs = {}
    for b in blocks:
        if b["segment"] == label:
            pairs.setdefault(b["pair"], {})[b["side"]] = b["ms_per_step"]
    ratios = [p["on"] / p["off"] - 1 for p in pairs.values() if "on" in p and "off" in p]
    out["overhead_mean"] = out["on"]["mean_ms"] / out["off"]["mean_ms"] - 1
    out["overhead_pair_median"] = statistics.median(ratios)
    out["overhead_pairs"] = ratios
    on = out["on"]
    out["d3_build_ms_per_step"] = 1e3 * on["d3_term_build_s"] / on["steps"]
    out["d3_build_share_of_off_step"] = out["d3_build_ms_per_step"] / out["off"]["mean_ms"]
    return out


def main(argv=None):
    args = parse_args(argv)
    (scan if args.command == "scan" else ab)(args)


if __name__ == "__main__":
    main()
