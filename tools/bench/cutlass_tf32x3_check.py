"""The CUTLASS 3xTF32 GEMM (fastmace.kernels.cutlass.tf32x3) against Triton's tf32x3 and cuBLAS IEEE on one GPU.

For atom-GEMM-like shapes (M rows, K x N weights) and both B layouts (row-major weight, and the column-major
``b.t()`` of a row-major [N, K], as in the backward): the error of each backend against a float64 product, scaled
by the output's |a| |b| sum (the cancellation-aware metric of opt-T01 §7), a bitwise repeat and a CUDA-graph replay
of the CUTLASS product, and the median time of each backend (CUDA events). Writes one JSON record.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

SHAPES = [(m, k, n) for m in (512, 5000, 40000) for k, n in ((128, 128), (128, 512), (512, 128), (256, 256),
                                                             (2048, 128), (128, 2048), (96, 160))]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reps", type=int, default=50)
    args = p.parse_args(argv)

    import torch

    import fastmace  # noqa: F401  (soft file locks, before CUDA work)
    from fastmace.kernels.cutlass import tf32x3 as cutlass
    from fastmace.kernels.triton import tf32x3_gemm as triton_gemm

    device = torch.device("cuda")
    record = dict(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                  torch=torch.__version__, cutlass_available=cutlass.available(device),
                  backend=triton_gemm.backend(device), cases=[])
    print(json.dumps({k: record[k] for k in ("gpu", "cutlass_available", "backend")}), flush=True)
    if not record["cutlass_available"]:
        args.output.write_text(json.dumps(record, indent=1))
        return 1
    generator = torch.Generator(device="cpu").manual_seed(7)

    def timed(fn):
        for _ in range(3):
            fn()
        times = []
        for _ in range(args.reps):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            stop.record()
            stop.synchronize()
            times.append(start.elapsed_time(stop) * 1e3)
        return statistics.median(times)

    def graph_timed(fn, launches=20):
        """Median µs per call of ``launches`` calls captured in one CUDA graph (no host launch cost)."""
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(launches):
                fn()
        graph.replay()
        times = []
        for _ in range(max(5, args.reps // 5)):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            stop.record()
            stop.synchronize()
            times.append(start.elapsed_time(stop) * 1e3 / launches)
        return statistics.median(times)

    def triton_into(a, b, out):
        saved = triton_gemm.CUTLASS
        triton_gemm.CUTLASS = False
        try:
            triton_gemm._launch(a, b, out)
        finally:
            triton_gemm.CUTLASS = saved

    worst = {}
    for m, k, n in SHAPES:
        for layout in ("rr", "rc"):
            a = torch.randn(m, k, generator=generator).cuda()
            b = (torch.randn(k, n, generator=generator).cuda() if layout == "rr"
                 else torch.randn(n, k, generator=generator).cuda().t())
            reference = a.double() @ b.double()
            scale = a.double().abs() @ b.double().abs()
            out = torch.empty(m, n, device=device)
            assert cutlass.eligible(a, b, out), (m, k, n, layout)
            errors = {}
            for name, fn in (("cutlass", lambda: cutlass.mm_into(a, b, out)), ("triton", lambda: triton_into(a, b, out)),
                             ("cublas_ieee", lambda: torch.mm(a, b, out=out))):
                torch.backends.cuda.matmul.allow_tf32 = False
                fn()
                torch.cuda.synchronize()
                errors[name] = float(((out.double() - reference).abs() / scale.clamp_min(1e-30)).max())
                worst[name] = max(worst.get(name, 0.0), errors[name])
            cutlass.mm_into(a, b, out)
            first = out.clone()
            cutlass.mm_into(a, b, out)
            repeat = bool(torch.equal(first, out))
            graph_out = torch.empty_like(out)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                cutlass.mm_into(a, b, graph_out)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                cutlass.mm_into(a, b, graph_out)
            graph_out.zero_()
            graph.replay()
            torch.cuda.synchronize()
            replay = bool(torch.equal(graph_out, first))
            calls = {"cutlass": lambda: cutlass.mm_into(a, b, out), "triton": lambda: triton_into(a, b, out),
                     "cublas_ieee": lambda: torch.mm(a, b, out=out)}
            times = {name: timed(fn) for name, fn in calls.items()}
            graphed = {name: graph_timed(fn) for name, fn in calls.items()}  # no host launch cost, as in MD
            case = dict(m=m, k=k, n=n, layout=layout, scaled_error=errors, bitwise_repeat=repeat, graph_replay=replay,
                        us=times, graph_us=graphed, cutlass_over_cublas=times["cutlass"] / times["cublas_ieee"],
                        cutlass_over_triton=times["cutlass"] / times["triton"],
                        graph_cutlass_over_cublas=graphed["cutlass"] / graphed["cublas_ieee"])
            record["cases"].append(case)
            print(f"{m:6d} {k:5d} {n:5d} {layout}  err c {errors['cutlass']:.1e} t {errors['triton']:.1e} "
                  f"i {errors['cublas_ieee']:.1e}  us c {times['cutlass']:8.1f} t {times['triton']:8.1f} "
                  f"i {times['cublas_ieee']:8.1f}  graph c {graphed['cutlass']:8.1f} t {graphed['triton']:8.1f} "
                  f"i {graphed['cublas_ieee']:8.1f}  rep {repeat} graph {replay}", flush=True)
    record["worst_scaled_error"] = worst
    record["all_repeat"] = all(c["bitwise_repeat"] for c in record["cases"])
    record["all_graph"] = all(c["graph_replay"] for c in record["cases"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1))
    print(json.dumps({k: record[k] for k in ("worst_scaled_error", "all_repeat", "all_graph")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
