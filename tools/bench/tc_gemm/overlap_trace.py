"""Stream overlap of the MD step's kernels under a GEMM backend (tensor-core GEMM screen, 29 October 2026).

Builds the default MD engine (MH-1 float32, NVT) like ``tools/ab_rewrites.py`` with ``--rewrites`` overrides and the
``cutlass_tc`` backend of ``FASTMACE_SCREEN_TC_GEMM``, warms up, then records ``--steps`` steps under torch.profiler
and exports a Chrome trace. From the trace's GPU kernels (graph-replayed kernels included) it reports per step: the
busy time of each stream, the union, the overlap of the two busiest streams, and for the GEMM kernels (names with
``gemm``/``cutlass``/``Gemm``) their time and the part of it that overlaps a kernel on another stream.

    python run_with_backend.py overlap_trace.py --output DIR --structure NaNbO3 --rewrites tf32x3_gemms=on
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[1]
GEMM_WORDS = ("gemm", "Gemm", "cutlass", "sgemm")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="mh1-r2scan")
    p.add_argument("--structure", default="NaNbO3")
    p.add_argument("--rewrites", nargs="*", default=[], metavar="FIELD=VALUE")
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--steps", type=int, default=20)
    return p.parse_args(argv)


def union(intervals):
    total, end = 0.0, None
    start = None
    for s, e in sorted(intervals):
        if end is None or s > end:
            if end is not None:
                total += end - start
            start, end = s, e
        else:
            end = max(end, e)
    if end is not None:
        total += end - start
    return total


def intersect_total(a, b):
    """Total length of the intersection of two interval sets."""
    def merged(x):
        out = []
        for s, e in sorted(x):
            if out and s <= out[-1][1]:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        return out
    a, b = merged(a), merged(b)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            total += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyse(kernels, steps):
    streams = {}
    for k in kernels:
        streams.setdefault(k["stream"], []).append((k["ts"], k["ts"] + k["dur"]))
    busy = {s: union(v) / steps for s, v in streams.items()}
    top = sorted(busy, key=busy.get, reverse=True)[:2]
    overlap = intersect_total(streams[top[0]], streams[top[1]]) / steps if len(top) == 2 else 0.0
    gemm = [k for k in kernels if any(w in k["name"] for w in GEMM_WORDS)]
    gemm_overlap = 0.0
    for s in streams:
        mine = [(k["ts"], k["ts"] + k["dur"]) for k in gemm if k["stream"] == s]
        others = [iv for t, v in streams.items() if t != s for iv in v]
        if mine and others:
            gemm_overlap += intersect_total(mine, others)
    span = (max(k["ts"] + k["dur"] for k in kernels) - min(k["ts"] for k in kernels)) / steps
    every = union([(k["ts"], k["ts"] + k["dur"]) for k in kernels]) / steps
    kernel_sum = sum(k["dur"] for k in kernels) / steps
    rest = [(k["ts"], k["ts"] + k["dur"]) for k in kernels if not any(w in k["name"] for w in GEMM_WORDS)]
    gemm_any = intersect_total([(k["ts"], k["ts"] + k["dur"]) for k in gemm], rest) / steps
    # Stream-agnostic concurrency (CUPTI may report graph-replayed kernels under the launch stream): the kernel
    # time beyond the union, and the GEMM time during which a non-GEMM kernel also runs.
    return dict(span_us=span, union_us=every, kernel_sum_us=kernel_sum, concurrent_us=kernel_sum - every,
                gemm_overlapping_non_gemm_us=gemm_any,
                stream_busy_us={str(s): v for s, v in busy.items()}, top_two_overlap_us=overlap,
                gemm_us=sum(k["dur"] for k in gemm) / steps, gemm_overlapping_other_streams_us=gemm_overlap / steps,
                gemm_kernels_per_step=len(gemm) / steps,
                gemm_by_stream_us={str(s): sum(k["dur"] for k in gemm if k["stream"] == s) / steps for s in streams})


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(TOOLS))
    import fastmace  # noqa: F401
    import torch
    from torch.profiler import ProfilerActivity, profile

    import hashlib

    import ab_step_time
    import force_stress_gate
    from fastmace.config import EngineConfig, IntegratorConfig, RewriteConfig
    from fastmace.model import loading
    from fastmace.model.forcefield import ForceField
    from fastmace.planner import minimal
    from fastmace.planner.workload import Workload

    args.output.mkdir(parents=True, exist_ok=True)
    rewrites = force_stress_gate.rewrite_overrides(args.rewrites)
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    loaded = loading.load_pristine(args.model, checkpoint=str(args.checkpoint), sha256=sha)
    name, base = ab_step_time.read_structure(args.structure, None)
    field = ForceField(loaded, base, precision="float32", rewrites=RewriteConfig(**rewrites) if rewrites else "auto")
    planner = minimal.MinimalPlanner(engine=EngineConfig(backend="hybrid"))
    plan = planner.plan(field, base, IntegratorConfig(temperature_K=args.temperature_K),
                        Workload(max_temperature_K=args.temperature_K), planner.hardware())
    engine = planner.build(plan, field, base.copy(), temperature_K=args.temperature_K)
    for _ in range(args.warmup):
        engine.step()
    engine.flush()
    torch.cuda.synchronize()
    trace = args.output / f"{args.structure}.trace.json"
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(args.steps):
            engine.step()
        engine.flush()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())["traceEvents"]
    kernels = [dict(name=e["name"], ts=float(e["ts"]), dur=float(e["dur"]), stream=e.get("args", {}).get("stream"))
               for e in events if e.get("cat") == "kernel"]
    result = dict(structure=args.structure, rewrites=args.rewrites, steps=args.steps, gpu=torch.cuda.get_device_name(),
                  fastmace_commit=ab_step_time.fastmace_commit(), kernels=len(kernels), **analyse(kernels, args.steps))
    (args.output / f"{args.structure}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
