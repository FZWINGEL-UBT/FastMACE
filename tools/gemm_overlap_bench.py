"""Tensor cores beside the CUDA-core FP32 GEMM (plan opt-1, OPT-T01b "tensor-core overlap"; speed only).

Question: does tensor-core work, run concurrently with cuBLAS's SIMT FP32 GEMM, add throughput?

For each shape C[M, N] = A[M, K] @ B[K, N] (float32, contiguous) it measures, warm, in one process:

- alone: cuBLAS FP32 (``fp32_precision="ieee"``), Triton ``tf32x3`` (3xTF32, FP32-class accuracy) and
  cuBLAS 1xTF32 (``fp32_precision="tf32"``, the tensor-core-only reference, not FP32-accurate);
- split: the N columns split between cuBLAS FP32 on one stream (fraction f) and the tensor path on a
  second stream (1 - f), both writing their own columns of C, for f in --splits; the useful FLOPs are
  those of the whole product;
- duplicate: the whole product twice at once, cuBLAS FP32 on one stream and the tensor path on the other
  (independent outputs), as an upper bound on what the second stream can add.

Every variant is captured once in a CUDA graph (two branches forked from and joined to the capture
stream), so launch overhead does not count, and timed as the median of --repeats replays between CUDA
events. The combined rate is the useful FLOPs over the wall time, compared with cuBLAS FP32 alone on the
whole product. Triton tf32x3 uses the fastest of tools/gemm_precision_bench.py's fixed configurations,
chosen alone per shape (a screen's upper bound). Output: <output>/summary.json. Use an idle GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemm_precision_bench as gpb  # noqa: E402  (the Triton GEMM and its configurations)

SHAPES = ("9720x512x512", "13608x512x512", "1944x512x384", "216x512x512")
SPLITS = (0.8, 0.7, 0.6, 0.5)


def parse_shape(text):
    m, k, n = (int(v) for v in text.lower().split("x"))
    return m, k, n


def column_split(n, fraction, multiple=64):
    """Columns for the FP32 side: ``fraction`` of ``n`` rounded to a multiple of 64 (the tile width)."""
    first = int(round(n * fraction / multiple)) * multiple
    return max(0, min(n, first))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True, help="Fresh directory")
    p.add_argument("--shapes", nargs="+", default=list(SHAPES), help="MxKxN")
    p.add_argument("--splits", nargs="+", type=float, default=list(SPLITS), help="FP32 share of the columns")
    p.add_argument("--repeats", type=int, default=50)
    args = p.parse_args(argv)
    import torch
    import triton

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ab_step_time  # noqa: E402

    if ab_step_time.gpu_processes():
        raise SystemExit("GPU must be idle")
    args.output.mkdir(parents=True, exist_ok=False)
    kernel = gpb.triton_kernel()
    generator = torch.Generator(device="cuda").manual_seed(20261020)
    side = torch.cuda.Stream()

    def mm(a, b, out, mode):
        if mode == "triton_tf32x3":
            gpb.run_triton(kernel, a, b, out, configs[mode], "tf32x3")
            return
        torch.backends.cuda.matmul.fp32_precision = "tf32" if mode == "cublas_tf32" else "ieee"
        torch.mm(a, b, out=out)
        torch.backends.cuda.matmul.fp32_precision = "ieee"

    def graph_of(first, second=None):
        """A CUDA graph of ``first()`` on the capture stream and ``second()`` on a forked side stream."""
        for _ in range(3):  # warm (Triton compiles, cuBLAS picks its kernel) before capture
            first()
            if second is not None:
                second()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        capture = torch.cuda.Stream()
        with torch.cuda.stream(capture):
            with torch.cuda.graph(graph, stream=capture):
                if second is not None:
                    fork = torch.cuda.Event()
                    fork.record(capture)
                    side.wait_event(fork)
                    with torch.cuda.stream(side):
                        second()
                    join = torch.cuda.Event()
                    join.record(side)
                first()
                if second is not None:
                    capture.wait_event(join)
        return graph

    def timed(graph):
        graph.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(args.repeats):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            graph.replay()
            e.record()
            e.synchronize()
            times.append(s.elapsed_time(e) * 1e3)
        return statistics.median(times)

    summary = dict(tool="gemm_overlap_bench", created_utc=datetime.now(timezone.utc).isoformat(),
                   host=os.uname().nodename, gpu=torch.cuda.get_device_name(), sms=torch.cuda.get_device_properties(0)
                   .multi_processor_count, torch=torch.__version__, triton=triton.__version__,
                   fastmace_commit=ab_step_time.fastmace_commit(), load=list(os.getloadavg()), repeats=args.repeats,
                   shapes={})
    for text in args.shapes:
        m, k, n = parse_shape(text)
        flops = 2 * m * k * n
        a = torch.randn(m, k, generator=generator, device="cuda")
        b = torch.randn(k, n, generator=generator, device="cuda")
        out = torch.empty(m, n, device="cuda")
        configs = {}
        # The fastest fixed Triton configuration alone (per shape).
        best = None
        for config in gpb.CONFIGS:
            try:
                gpb.run_triton(kernel, a, b, out, config, "tf32x3")
            except Exception:  # noqa: BLE001  (a configuration the compiler refuses)
                continue
            us = gpb.time_us(torch, lambda c=config: gpb.run_triton(kernel, a, b, out, c, "tf32x3"), 20)
            if best is None or us < best[0]:
                best = (us, config)
        configs["triton_tf32x3"] = best[1]
        entry = dict(m=m, k=k, n=n, gflop=flops / 1e9, triton_config=list(best[1]), alone={}, split={}, duplicate={})
        for mode in ("cublas_ieee", "triton_tf32x3", "cublas_tf32"):
            us = timed(graph_of(lambda mode=mode: mm(a, b, out, mode)))
            entry["alone"][mode] = dict(us=us, tflops=flops / (us * 1e-6) / 1e12)
        base_us = entry["alone"]["cublas_ieee"]["us"]
        for tensor_mode in ("triton_tf32x3", "cublas_tf32"):
            results = {}
            for fraction in args.splits:
                cols = column_split(n, fraction)
                if cols in (0, n):
                    continue
                b1, b2 = b[:, :cols].contiguous(), b[:, cols:].contiguous()
                o1 = torch.empty(m, cols, device="cuda")
                o2 = torch.empty(m, n - cols, device="cuda")
                one = timed(graph_of(lambda: mm(a, b1, o1, "cublas_ieee")))
                two = timed(graph_of(lambda tm=tensor_mode: mm(a, b2, o2, tm)))
                both = timed(graph_of(lambda: mm(a, b1, o1, "cublas_ieee"), lambda tm=tensor_mode: mm(a, b2, o2, tm)))
                results[f"{cols / n:.3f}"] = dict(fp32_columns=cols, fp32_part_us=one, tensor_part_us=two,
                                                 concurrent_us=both, serial_us=one + two,
                                                 overlap=(one + two - both) / min(one, two),
                                                 tflops=flops / (both * 1e-6) / 1e12, speedup=base_us / both)
            entry["split"][tensor_mode] = results
            a2 = torch.randn(m, k, generator=generator, device="cuda")
            out2 = torch.empty(m, n, device="cuda")
            both = timed(graph_of(lambda: mm(a, b, out, "cublas_ieee"), lambda tm=tensor_mode: mm(a2, b, out2, tm)))
            entry["duplicate"][tensor_mode] = dict(concurrent_us=both, tflops=2 * flops / (both * 1e-6) / 1e12,
                                                   throughput_vs_fp32_alone=2 * base_us / both)
        summary["shapes"][text] = entry
        best_split = max((r["speedup"], mode, f) for mode, rs in entry["split"].items() for f, r in rs.items())
        print(text, {mode: round(r["tflops"], 2) for mode, r in entry["alone"].items()},
              "best split", [round(best_split[0], 3), best_split[1], best_split[2]],
              "duplicate", {mode: round(r["throughput_vs_fp32_alone"], 3) for mode, r in entry["duplicate"].items()},
              flush=True)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary["load_after"] = list(os.getloadavg())
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
