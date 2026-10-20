"""Timing breakdown of one MD step of FastMACE's production default (plan task 2-T20a).

The same three measurements as the research engine's
studies/further_optimizations/probes/step_breakdown_probe.py (15 October 2026), in one process,
after equilibrating at --temperature-K, so the tables compare row by row:

1. Wall time: --steps MD steps of the planned GraphEngine (CUDA graphs, pipelined where the plan
   has it), twice; the second window counts (the first may capture graphs for new edge sizes).
2. Stages: --steps further steps with CUDA events around every stage the host launches (graph
   replays ACP or AC, B and C), the neighbour-list rebuilds (``force_field.rebuild``) and the noise
   upload (``integrator.prepare_step``). GPU time not covered by a stage is idle time between stages.
3. Force stage by kernel: --calls eager calls of the force field (the compiled model whose kernels
   the B graph replays) under the profiler. Every kernel is attributed to the CPU operator that
   launched it and to a model part by operator name and tensor shapes (edge-sized against
   atom-sized), with the research engine's categories. FastMACE's Triton and NVRTC kernels run in
   custom ops (``fastmace::...``), so their operator names identify them.
4. Replays of the force graph under the profiler: kernels per replay, kernel time against the
   replay span, the largest gaps.

The engine comes from the minimal planner (what ``Simulation`` and ``fm.md`` build) with every
default of the plan. Velocities start from a Maxwell-Boltzmann distribution at --temperature-K with
the research probe's seed. Measurement only: the engine's methods are wrapped on the instance for
part 2 and restored afterwards. Use an otherwise idle GPU; hold the local reservation
(tools/ci/local_gpu.py run). Writes summary.json, stages.csv, categories.csv and ops.csv.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_step_time  # noqa: E402  (tools/: structures, commit, GPU processes)

STAGE_NAMES = {"ACP": "ACP: Langevin half kick, positions, neighbour check, edge compaction",
               "AC": "AC: Langevin half kick, positions, neighbour check, edge compaction",
               "B": "B: forces + second half kick",
               "C": "C: edge compaction after a rebuild"}
REBUILD = "Neighbour-list rebuild (GPU)"
NOISE = "Noise upload (every step a slice; a new batch every noise_batch)"
B_STAGE = STAGE_NAMES["B"]
PACKAGES = ("torch", "triton", "mace-torch", "cuequivariance", "cuequivariance-torch", "cuequivariance-ops-torch-cu12",
            "cuequivariance-ops-torch-cu13", "openequivariance", "e3nn", "ase", "numpy")


def first_rows(dims):
    """Leading sizes of all 2-D or larger input tensors of an operator."""
    try:
        shapes = json.loads(dims) if dims else []
    except json.JSONDecodeError:
        return []
    rows = []
    for shape in shapes:
        if isinstance(shape, list) and len(shape) >= 2 and all(isinstance(v, int) for v in shape):
            rows.append(shape[0])
    return rows


def category(kernel, op, dims, edges):
    """The research probe's model part of one kernel; FastMACE's custom ops are named ``fastmace::...``."""
    name = f"{op} {kernel}"
    if "fused_radial_hidden" in name:
        return "Radial networks: fused hidden block"
    if "folded_linear_input_grad" in name:
        return "Radial networks: folded last layer, input gradient"
    if "edge_outer_sum" in name:
        return "Interaction 1: reassociated edge sums"
    rows = first_rows(dims)
    edge_sized = edges in rows
    if "jit_conv_forward" in op:
        return "OEq convolution forward"
    if "jit_conv_backward" in op:
        return "OEq convolution backward"
    if "streaming_conv_forward" in name:
        return "Interaction 2: convolution forward (streaming)"
    if "streaming_conv_backward" in name:
        return "Interaction 2: convolution backward (streaming)"
    if op in ("aten::mm", "aten::addmm", "aten::bmm"):
        if edge_sized:
            try:
                shapes = json.loads(dims)
            except json.JSONDecodeError:
                shapes = []
            weight = next((s for s in shapes if isinstance(s, list) and len(s) == 2 and s[0] != edges), None)
            if weight and max(weight) >= 1000:
                return "Radial networks: folded last layer, forward GEMM"
            return "Radial networks: other GEMMs"
        return "Atom-level linear layers (GEMM)"
    if "uniform_1d" in name or "segmented_polynomial" in name:
        return "Atom-level cuEq tensor products (symmetric contraction)"
    if kernel.startswith(("Memcpy", "Memset")) or op in ("aten::copy_", "aten::zero_", "aten::fill_"):
        return "Copies and fills"
    if edge_sized:
        if "layer_norm" in kernel or "silu" in kernel:
            return "Radial networks: elementwise (norms, activations)"
        return "Edge elementwise (geometry, radial basis, gathers, scatters)"
    if "triton" in kernel or "triton" in op:
        return "Atom-level elementwise (gates, products, residuals, readouts)"
    return "Other"


def package_versions():
    from importlib import metadata
    found = {}
    for name in PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return found


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_engine(args, base):
    """The minimal planner's engine for ``base`` (as tools/ab_fastmace.py's 1b side), and its plan."""
    from fastmace.config import EngineConfig, IntegratorConfig
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import fixed, minimal
    from fastmace.planner.workload import Workload

    planner = minimal.MinimalPlanner(engine=EngineConfig(backend=args.backend))
    hardware = planner.hardware()  # the allocator rule, before the first CUDA allocation
    started = time.perf_counter()
    loaded = loading.load_pristine(args.model, checkpoint=str(args.checkpoint), sha256=sha256(args.checkpoint))
    load_s = time.perf_counter() - started
    started = time.perf_counter()
    force_field = ForceField(loaded, base, precision="float32" if args.backend == "hybrid" else "float64")
    integrator = IntegratorConfig(temperature_K=args.temperature_K)
    plan = planner.plan(force_field, base, integrator, Workload(max_temperature_K=args.temperature_K), hardware)
    engine = planner.build(plan, force_field, base.copy(), temperature_K=args.temperature_K)
    options = {name: True for name in fixed.enabled_options(plan)}
    options.update({f"off:{name}": reason for name, reason in fixed.auto_options_off(plan).items()})
    return engine, dict(describe=plan.describe(), options=options, load_s=load_s,
                        setup_s=time.perf_counter() - started, sha256=loaded.sha256)


def stage_name(key):
    kind = key[0] if isinstance(key, tuple) else key
    return STAGE_NAMES.get(kind, str(kind))


def read_trace(path):
    events = json.loads(path.read_text())
    path.unlink()  # large; the tables keep what is needed
    return events.get("traceEvents", events)


def kernel_table(events):
    """{(kernel, op, input dims): [launches, µs]} with each kernel attributed to its innermost CPU operator."""
    ops = defaultdict(list)
    launches, kernels = {}, []
    for e in events:
        cat = e.get("cat")
        if cat in ("cpu_op", "user_annotation") and "dur" in e:
            ops[e["tid"]].append(e)
        elif cat in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {}):
            launches[e["args"]["correlation"]] = e
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels.append(e)
    index = {}
    for tid, items in ops.items():
        items.sort(key=lambda e: e["ts"])
        index[tid] = ([e["ts"] for e in items], items)

    def owner(launch):
        if launch is None or launch["tid"] not in index:
            return "?", ""
        starts, items = index[launch["tid"]]
        best = None
        for e in reversed(items[:bisect.bisect_right(starts, launch["ts"])]):
            if e["ts"] + e["dur"] >= launch["ts"] + launch.get("dur", 0):
                if best is None or e["dur"] < best["dur"]:
                    best = e
                if launch["ts"] - e["ts"] > 2e5:
                    break
        if best is None:
            return "?", ""
        return best["name"], json.dumps(best.get("args", {}).get("Input Dims", ""))[:4000]

    table = defaultdict(lambda: [0, 0.0])
    for k in kernels:
        op, dims = owner(launches.get(k.get("args", {}).get("correlation")))
        row = table[(k["name"][:120], op, dims)]
        row[0] += 1
        row[1] += k["dur"]
    return table, len(kernels)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True, help="Fresh directory")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--structure", default="NaCl",
                   help="Benchmark name under FASTMACE_STRUCTURES (3x3x3), or a file with PATH@NXxNYxNZ")
    p.add_argument("--backend", choices=("hybrid", "cueq"), default="hybrid")
    p.add_argument("--temperature-K", type=float, default=1000.0)
    p.add_argument("--velocity-seed", type=int, default=20261007, help="The research probe's Maxwell-Boltzmann seed")
    p.add_argument("--equilibrate-steps", type=int, default=300)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--calls", type=int, default=10)
    p.add_argument("--allow-busy-gpu", action="store_true", help="CPU checks of the tool only (no timing)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    if args.backend == "hybrid":
        oeq_runtime = ab_step_time.prepare_fastmace_hybrid()  # before anything imports MACE
    else:
        oeq_runtime = None
    import numpy as np
    import torch
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    from torch.profiler import ProfilerActivity, profile

    if not args.allow_busy_gpu and ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=False)
    structure, atoms = ab_step_time.read_structure(args.structure, None)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_K, rng=np.random.default_rng(args.velocity_seed))
    Stationary(atoms)
    engine, plan = build_engine(args, atoms)
    field, integrator = engine.force_field, engine.integrator
    captures = [0]
    original_capture = engine._capture

    def counted_capture(key):
        captures[0] += 1
        return original_capture(key)

    engine._capture = counted_capture
    started = time.perf_counter()
    for _ in range(args.equilibrate_steps):
        engine.step()
    engine.flush()
    torch.cuda.synchronize()
    equilibrate_s = time.perf_counter() - started

    def counters():
        return dict(rebuilds=engine.rebuilds, redos=engine.pipelined_redos,
                    early_rebuilds=engine.pipelined_early_rebuilds, graph_captures=captures[0])

    def delta(before):
        now = counters()
        return {key: now[key] - before[key] for key in now}

    # 1. Wall time of the uninstrumented loop (twice: the first window may capture new graphs).
    windows = []
    for _ in range(2):
        before = counters()
        start = time.perf_counter()
        for _ in range(args.steps):
            engine.step()
        engine.flush()
        torch.cuda.synchronize()
        windows.append(dict(ms_per_step=(time.perf_counter() - start) * 1e3 / args.steps, **delta(before),
                            edge_sizes=sorted(engine.edge_sizes_used)))
    captures_before_stages = captures[0]

    # 2. Stage timing with CUDA events.
    marks = []  # (stage, start event, end event)
    host = defaultdict(float)
    original_run, original_rebuild, original_prepare = engine._run, field.rebuild, integrator.prepare_step

    def timed(kind, fn, *a):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t = time.perf_counter()
        s.record()
        out = fn(*a)
        e.record()
        host[kind] += time.perf_counter() - t
        marks.append((kind, s, e))
        return out

    engine._run = lambda key: timed(stage_name(key), original_run, key)
    field.rebuild = lambda positions: timed(REBUILD, original_rebuild, positions)
    integrator.prepare_step = lambda state: timed(NOISE, original_prepare, state)
    first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    before = counters()
    torch.cuda.synchronize()
    t = time.perf_counter()
    first.record()
    for _ in range(args.steps):
        engine.step()
    engine.flush()
    last.record()
    torch.cuda.synchronize()
    events_wall_ms = (time.perf_counter() - t) * 1e3 / args.steps
    instrumented = delta(before)
    del engine._run, field.rebuild, integrator.prepare_step  # the class methods again
    if engine._run.__func__ is not type(engine)._run or field.rebuild != original_rebuild:
        raise RuntimeError("stage wrappers not removed")
    gpu_span_ms = first.elapsed_time(last) / args.steps
    stages = defaultdict(lambda: [0, 0.0])
    for kind, s, e in marks:
        stages[kind][0] += 1
        stages[kind][1] += s.elapsed_time(e)
    covered = sum(v[1] for v in stages.values()) / args.steps
    with (args.output / "stages.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "calls", "ms_per_call", "ms_per_step", "host_ms_per_step"])
        for kind, (count, total) in sorted(stages.items(), key=lambda item: -item[1][1]):
            writer.writerow([kind, count, f"{total / count:.4f}", f"{total / args.steps:.4f}",
                             f"{host[kind] * 1e3 / args.steps:.4f}"])
        writer.writerow(["GPU idle between stages", "", "", f"{gpu_span_ms - covered:.4f}", ""])

    # 3. Force stage by kernel (eager calls of the compiled model, on the edges B last used).
    positions = engine.state.positions.detach().clone()
    edges = int(field.data["edge_index"].shape[1])
    for _ in range(3):
        field(positions)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(args.calls):
            field(positions)
        torch.cuda.synchronize()
    trace = args.output / "trace.json"
    prof.export_chrome_trace(str(trace))
    table, kernel_count = kernel_table(read_trace(trace))
    categories = defaultdict(lambda: [0.0, 0.0])
    with (args.output / "ops.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["us_per_call", "launches_per_call", "category", "kernel", "op", "input_dims"])
        for (name, op, dims), (count, dur) in sorted(table.items(), key=lambda item: -item[1][1]):
            group = category(name, op, dims, edges)
            categories[group][0] += dur / args.calls
            categories[group][1] += count / args.calls
            writer.writerow([f"{dur / args.calls:.1f}", count / args.calls, group, name, op, dims[:300]])
    kernel_ms = sum(v[0] for v in categories.values()) / 1e3
    with (args.output / "categories.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "ms_per_force_call", "share_of_kernel_time", "kernels_per_call"])
        for group, (us, count) in sorted(categories.items(), key=lambda item: -item[1][0]):
            writer.writerow([group, f"{us / 1e3:.4f}", f"{us / 1e3 / kernel_ms:.4f}", f"{count:g}"])

    # 4. Replays of the force graph under the profiler: kernel time against the replay span.
    graph_key = ("B", int(engine._pipe_edges)) if ("B", int(engine._pipe_edges)) in engine._graphs else None
    if graph_key is None:
        graph_key = next((key for key in engine._graphs if isinstance(key, tuple) and key[0] == "B"
                          and key[1] == edges), None)
    replay_stats = None
    if graph_key is not None:
        graph = engine._graphs[graph_key]
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as replay_prof:
            for _ in range(args.calls):
                graph.replay()
            torch.cuda.synchronize()
        replay_kernels = sorted((e for e in replay_prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
                                key=lambda e: e.time_range.start)
        if replay_kernels:
            busy = sum(e.time_range.elapsed_us() for e in replay_kernels)
            span = replay_kernels[-1].time_range.end - replay_kernels[0].time_range.start
            gaps = sorted((b.time_range.start - a.time_range.end
                           for a, b in zip(replay_kernels, replay_kernels[1:])), reverse=True)
            replay_stats = dict(graph=str(graph_key), kernels_per_replay=len(replay_kernels) / args.calls,
                                kernel_ms_per_replay=busy / 1e3 / args.calls,
                                span_ms_per_replay=span / 1e3 / args.calls,
                                largest_gaps_us=[round(g, 1) for g in gaps[:10]],
                                gaps_over_5us_ms_per_replay=sum(g for g in gaps if g > 5) / 1e3 / args.calls)

    summary = dict(
        tool="step_breakdown", created_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
        python=sys.executable, fastmace_commit=ab_step_time.fastmace_commit(), packages=package_versions(),
        gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda, oeq_runtime=oeq_runtime,
        other_gpu_processes=[pid for pid in ab_step_time.gpu_processes() if pid != os.getpid()],
        model=args.model, checkpoint=str(args.checkpoint), checkpoint_sha256=plan["sha256"], backend=args.backend,
        dtype="float32" if args.backend == "hybrid" else "float64", structure=structure, natoms=len(atoms),
        temperature_K=args.temperature_K, steps=args.steps, equilibrate_steps=args.equilibrate_steps,
        calls=args.calls, padded_edges=edges, plan=plan, engine_mode=engine.mode,
        load_s=plan["load_s"], setup_s=plan["setup_s"], equilibrate_s=equilibrate_s,
        wall_ms_per_step=windows[-1]["ms_per_step"], wall_windows=windows,
        graph_captures_before_stages=captures_before_stages, force_graph_replay_profile=replay_stats,
        instrumented_wall_ms_per_step=events_wall_ms, gpu_span_ms_per_step=gpu_span_ms,
        stage_ms_per_step={k: v[1] / args.steps for k, v in stages.items()},
        stage_calls={k: v[0] for k, v in stages.items()},
        gpu_idle_ms_per_step=gpu_span_ms - covered, instrumented=instrumented,
        force_graph_ms_per_replay=stages[B_STAGE][1] / max(stages[B_STAGE][0], 1),
        eager_force_kernel_ms_per_call=kernel_ms, eager_kernels_per_call=kernel_count / args.calls,
        engine_info=dict(engine.info()), force_field_info=dict(field.info()),
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("wall_ms_per_step", "gpu_span_ms_per_step", "force_graph_ms_per_replay",
                                              "eager_force_kernel_ms_per_call", "stage_ms_per_step")}, indent=2))
    return summary


if __name__ == "__main__":
    main()
