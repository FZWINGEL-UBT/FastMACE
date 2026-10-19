"""1b-T11 probe: the research engine's and FastMACE's CUDA graphs for one model, kernel by kernel, in one process.

Builds both sides as tools/ab_step_time.py does (same structure, seed, options, Inductor custom-pass
slot handling), warms each up right after its build, then:

1. times whole MD steps in alternating blocks (as the A/B tool, fewer blocks);
2. times replays of each side's ACP graph and B graph (current edge size) alone, in alternating
   blocks, with CUDA events;
3. replays each graph under torch.profiler (CUDA activity) and records every kernel, memset and
   memcpy of one replay: name, count per replay, GPU time per replay.

Replaying B repeatedly kicks the velocities again each time, so the graph profiles come last and
the MD state afterwards is not used. Output: <output>/probe.json.

Run like the A/B tool: the hybrid environment through the export's oeq_no_nvcc.sh, with
PYTHONPATH=<FastMACE checkout>; --sides fastmace only needs neither (single-engine control).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

TOOLS = Path(fastmace.__file__).resolve().parents[1] / "tools"


def _ab_tool():
    spec = importlib.util.spec_from_file_location("_ab_step_time", TOOLS / "ab_step_time.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def graphs_of(side, kind):
    """{key: CUDAGraph} and the current B key of one side."""
    if kind == "fastmace":
        engine = side.engine
        return engine._graphs, ("B", int(engine._pipe_edges))
    dyn = side.engine.dyn
    return dyn._graphs, ("B", int(dyn._pipe_edges))


def replay_ms(graph, torch, replays):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / replays


def profile_graph(graph, torch, replays):
    """Per-name kernel/memset/memcpy statistics of one replay, from ``replays`` profiled replays."""
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
    events = [e for e in prof.events() if e.device_type.name == "CUDA"]
    events.sort(key=lambda e: e.time_range.start)
    by_name = defaultdict(lambda: [0, 0.0])
    for e in events:
        by_name[e.name][0] += 1
        by_name[e.name][1] += e.time_range.elapsed_us()
    busy = sum(e.time_range.elapsed_us() for e in events)
    span = events[-1].time_range.end - events[0].time_range.start if events else 0.0
    rows = sorted(({"name": name, "per_replay": count / replays, "us_per_replay": total / replays}
                   for name, (count, total) in by_name.items()), key=lambda r: -r["us_per_replay"])
    return dict(nodes_per_replay=len(events) / replays, busy_us_per_replay=busy / replays,
                span_us_per_replay=span / replays, kernels=rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--sides", nargs="+", choices=("baseline", "fastmace"), default=["baseline", "fastmace"])
    p.add_argument("--model", default="mp0-medium")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--structure", default="NaCl")
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--block", type=int, default=300)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--replay-blocks", type=int, default=10)
    p.add_argument("--replays", type=int, default=200)
    p.add_argument("--profile-replays", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--temperature-K", type=float, default=300.0)
    args = p.parse_args(argv)

    tool = _ab_tool()
    ab_args = tool.parse_args(
        ["--output", str(args.output / "unused"), "--model", args.model, "--checkpoint", str(args.checkpoint),
         "--sides", *args.sides, "--seed", str(args.seed)]
        + (["--baseline", str(args.baseline)] if args.baseline else []))
    from fastmace.bench import ab, step_time

    has_baseline, has_fastmace = "baseline" in args.sides, "fastmace" in args.sides
    oeq_runtime = tool.prepare_fastmace_hybrid() if has_fastmace else None
    experiment = None
    if has_baseline:
        export = ab.load_baseline(args.baseline)
        ab.activate(export)
        from langevin import compile_cache
        from langevin import workflow as experiment
    if tool.gpu_processes():
        raise RuntimeError("GPU must be idle")
    if has_baseline:
        experiment.require_hybrid_wrapper()
    args.output.mkdir(parents=True, exist_ok=False)
    if has_baseline:
        compile_cache.configure(args.output)
        from langevin import gpu_langevin
        gpu_langevin.configure_allocator()
    planner = tool.fastmace_planner(ab_args) if has_fastmace else None
    hardware = planner.hardware() if has_fastmace else None
    import torch

    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    md = None
    if has_baseline:
        md = experiment.parse_args(["--model", args.model, "--checkpoint", str(checkpoint), "--backend", "hybrid"])
        experiment.apply_polar_options(md)
    import copy
    out = dict(created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename, python=sys.executable,
               fastmace_commit=tool.fastmace_commit(), baseline=str(args.baseline), model=args.model,
               structure=args.structure, sides=args.sides, oeq_runtime=oeq_runtime, args=vars(args) | {
                   "output": str(args.output), "baseline": str(args.baseline), "checkpoint": str(args.checkpoint)})
    slot = tool.CustomPassSlot()
    sides, kinds = {}, {}
    structure, base = tool.read_structure(args.structure, experiment)
    out["natoms"] = len(base)
    timing, starting_forces = {}, {}
    for name in args.sides:
        kinds[name] = name
        slot.enter(name)
        started = time.perf_counter()
        if name == "fastmace":
            from fastmace.model.compile import CompiledModelCache
            loaded = planner.load(args.model, checkpoint=str(checkpoint))
            load_s = time.perf_counter() - started
            started = time.perf_counter()
            plan = planner.plan(loaded, base, tool.fastmace_integrator(ab_args), hardware)
            engine = planner.build(plan, loaded, base.copy(), temperature_K=args.temperature_K,
                                   compile_cache=CompiledModelCache())
            sides[name] = tool.FastMACESide(engine, torch)
        else:
            common, calc = experiment.load_md_calculator(md.dtype, True, "cuda:0", checkpoint, label=args.model,
                                                         cueq_conv_fusion=None)
            experiment.configure_md_calculator(calc, compute_stress=False)
            options = copy.copy(md)
            experiment.adapt_options_to_model(options, calc.models[0])
            flags = {flag: getattr(options, flag) for flag in tool.FLAGS}
            load_s = time.perf_counter() - started
            started = time.perf_counter()
            torch.set_default_dtype(torch.float32)
            engine = experiment.GPUEngine(base.copy(), calc, timestep_fs=md.timestep_fs, friction=md.friction,
                                          seed=args.seed, skin=md.skin, compile_mode=md.compile_mode,
                                          device="cuda:0", ensemble="nvt", cell_mode="hydrostatic",
                                          **{key: getattr(md, key) for key in tool.NPT_KEYWORDS}, **flags)
            sides[name] = tool.ResearchEngineSide(engine, args.temperature_K, torch)
            out["baseline_flags"] = flags
        setup_s = time.perf_counter() - started
        starting_forces[name] = (engine.state.forces if name == "fastmace" else engine.dyn.f).detach().double().cpu()
        started = time.perf_counter()
        step_time.warm_up(sides[name], args.warmup)
        timing[name] = dict(load_s=load_s, setup_s=setup_s, warmup_s=time.perf_counter() - started)
        slot.leave(name)
    out["setup"] = timing
    if len(starting_forces) == 2:
        a, b = starting_forces.values()
        out["starting_forces_max_abs_difference_eV_A"] = float((a - b).abs().max())
        out["starting_forces_bitwise_equal"] = bool(torch.equal(a, b))
    from torch._dynamo.utils import counters
    before = tool.compile_counters(counters)

    # 1. Whole steps, alternating blocks.
    protocol = step_time.TimingProtocol(warmup=0, block=args.block, blocks=args.blocks)
    if len(args.sides) == 2:
        result = ab.run_ab(sides, protocol, pattern="abab")
        out["steps"] = result.as_dict()
    else:
        out["steps"] = {args.sides[0]: step_time.time_engine(sides[args.sides[0]], protocol).as_dict()}
    for side in sides.values():
        side.finish_block()
    torch.cuda.synchronize()

    # 2. Graph replays alone, alternating.
    graphs = {name: graphs_of(sides[name], kinds[name]) for name in args.sides}
    out["graph_keys"] = {name: [str(k) for k in g] for name, (g, _) in graphs.items()}
    out["b_key"] = {name: str(key) for name, (_, key) in graphs.items()}
    replays = {}
    for label in ("ACP", "B"):
        samples = {name: [] for name in args.sides}
        for _ in range(args.replay_blocks):
            for name in args.sides:
                table, bkey = graphs[name]
                key = bkey if label == "B" else "ACP"
                if hasattr(sides[name], "prepare_block"):
                    sides[name].prepare_block()
                samples[name].append(replay_ms(table[key], torch, args.replays))
        replays[label] = {name: dict(median_ms=statistics.median(v), blocks_ms=v) for name, v in samples.items()}
        if len(args.sides) == 2:
            a, b = args.sides
            replays[label]["ratio_fastmace_over_baseline"] = (
                replays[label]["fastmace"]["median_ms"] / replays[label]["baseline"]["median_ms"])
    out["replays"] = replays

    # 3. Profiles.
    profiles = {}
    for name in args.sides:
        table, bkey = graphs[name]
        sides[name].prepare_block()
        profiles[name] = {"ACP": profile_graph(table["ACP"], torch, args.profile_replays),
                          "B": profile_graph(table[bkey], torch, args.profile_replays)}
    out["profiles"] = profiles
    changes, recompiled = tool.compile_changes(before, tool.compile_counters(counters))
    out["compile_during_timing"] = dict(changes=changes, recompiled=recompiled)
    out["other_gpu_processes"] = [pid for pid in tool.gpu_processes() if pid != os.getpid()]
    out["gpu"], out["torch"] = torch.cuda.get_device_name(0), torch.__version__
    (args.output / "probe.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    for label, entry in replays.items():
        print(label, {k: (v["median_ms"] if isinstance(v, dict) else v) for k, v in entry.items()})
    print("steps", {k: v for k, v in out["steps"].items() if k in ("ratio_b_over_a",)})
    for name, prof in profiles.items():
        print(name, {g: (p["nodes_per_replay"], round(p["busy_us_per_replay"], 1)) for g, p in prof.items()})


if __name__ == "__main__":
    main()
