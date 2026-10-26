#!/usr/bin/env python3
"""OPT5-T08 profile: why the SIMT + 3xFP16 hybrid GEMM loses to cuBLAS IEEE FP32.

Two modes, run under a quiet ``tools/ci/local_gpu.py`` reservation:

* ``--mode ncu``: one launch per (kernel, shape) after warm-up, inside a
  ``cudaProfilerStart``/``Stop`` window, for ``ncu --profile-from-start off``.
  Launch order per shape: ``KERNELS``.
* ``--mode sustain``: each kernel replayed in a CUDA graph for ``--seconds``
  while ``nvidia-smi`` samples SM clock, power and throttle reasons, so the
  per-call time can be set against the clock it actually ran at.

Experimental profiling only; no production code path is touched.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import opt5_t08_large_gemm as large  # imports fastmace before torch
import torch
import triton

baseline = large.baseline
SHAPES = [(1512, 512, 512), (13608, 512, 512), (1944, 1024, 1024)]
KERNELS = ["cublas", "hybrid_gemm_6x2_64_128", "col64_aligned_packed", "col128_aligned_packed"]
ALL_KERNELS = KERNELS + ["cublas_fp16", "triton_fp16x3_rowcol"]


def make_calls(kernels, shape):
    m, k, n = shape
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(k, n, device="cuda")
    out = torch.empty(m, n, device="cuda")
    packed = torch.empty(triton.cdiv(k, 16) * triton.cdiv(n, 8) * 8 * 16, dtype=torch.int32, device="cuda")
    large.pack_b(b, packed)  # constant weights: packed once, not timed
    a16 = a.half()
    b16 = b.half()
    out16 = torch.empty(m, n, device="cuda", dtype=torch.float16)
    calls = {}
    for name in KERNELS:
        if name == "cublas":
            calls[name] = lambda: torch.mm(a, b, out=out)
        elif name == "cublas_fp16":
            calls[name] = lambda: torch.mm(a16, b16, out=out16)
        elif name == "triton_fp16x3_rowcol":
            calls[name] = lambda: baseline.launch_triton_fp16x3_rowcol(a, b, out)
        elif name.startswith("hybrid"):
            calls[name] = lambda info=kernels[name]: baseline.launch_hybrid_nvrtc(info, a, b, out)
        else:
            calls[name] = lambda info=kernels[name]: large.launch(info, a, b, packed, out)
    return calls, (a, b, out, packed, a16, b16, out16)


def ncu_mode():
    for shape in SHAPES:
        calls, keep = make_calls(KERNELS_INFO, shape)
        for fn in calls.values():
            for _ in range(3):
                fn()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        for name, fn in calls.items():
            fn()
            torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        del keep


def smi_sampler(path):
    query = "timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active"
    return subprocess.Popen(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits", "-lms", "50"],
                            stdout=path.open("w"), stderr=subprocess.DEVNULL)


def parse_samples(path):
    rows = []
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            stamp = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            rows.append(dict(t=stamp, sm_mhz=float(parts[1]), mem_mhz=float(parts[2]), power_w=float(parts[3]),
                             temp_c=float(parts[4]), throttle=parts[5]))
        except ValueError:
            continue
    return rows


def sustain_mode(seconds, output):
    samples_path = output.with_suffix(".smi.csv")
    sampler = smi_sampler(samples_path)
    time.sleep(1.0)
    windows = []
    try:
        for shape in SHAPES:
            calls, keep = make_calls(KERNELS_INFO, shape)
            for name, fn in calls.items():
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(80):
                        fn()
                time.sleep(1.0)  # idle gap between kernels: let clock and power settle back
                replays, start = 0, time.time()
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                while time.time() - start < seconds:
                    graph.replay()
                    replays += 1
                    if replays % 20 == 0:
                        torch.cuda.synchronize()
                end.record()
                end.synchronize()
                stop = time.time()
                us = begin.elapsed_time(end) * 1000 / (replays * 80)
                windows.append(dict(kernel=name, shape=list(shape), start=start, stop=stop, us_per_call=us,
                                    tflops=2 * shape[0] * shape[1] * shape[2] / us / 1e6))
                print(json.dumps(windows[-1]), flush=True)
            del keep
            torch.cuda.empty_cache()
    finally:
        time.sleep(0.5)
        sampler.terminate()
        sampler.wait()
    samples = parse_samples(samples_path)
    for w in windows:
        # Skip the first 0.5 s of each window (ramp), use the steady remainder.
        inside = [s for s in samples if w["start"] + 0.5 <= s["t"] <= w["stop"]]
        if inside:
            w["samples"] = len(inside)
            w["sm_mhz_median"] = sorted(s["sm_mhz"] for s in inside)[len(inside) // 2]
            w["power_w_median"] = sorted(s["power_w"] for s in inside)[len(inside) // 2]
            w["temp_c_max"] = max(s["temp_c"] for s in inside)
            w["throttle_reasons"] = sorted({s["throttle"] for s in inside})
            sms = torch.cuda.get_device_properties(0).multi_processor_count
            peak = sms * 256 * w["sm_mhz_median"] * 1e6 / 1e12
            w["fp32_peak_tflops_at_clock"] = peak
            w["fraction_of_fp32_peak"] = w["tflops"] / peak
    return dict(windows=windows, samples=len(samples))


def main():
    global KERNELS_INFO, KERNELS, SHAPES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("ncu", "sustain"), required=True)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--kernels", help="Comma-separated subset of: " + ",".join(ALL_KERNELS))
    parser.add_argument("--shapes", help="Comma-separated MxKxN shapes, for example 13608x512x512")
    args = parser.parse_args()
    torch.zeros(1, device="cuda")
    if args.kernels:
        KERNELS = args.kernels.split(",")
        unknown = set(KERNELS) - set(ALL_KERNELS)
        if unknown:
            parser.error(f"Unknown kernels: {sorted(unknown)}")
    if args.shapes:
        try:
            SHAPES = [tuple(int(x) for x in shape.split("x")) for shape in args.shapes.split(",")]
        except ValueError:
            parser.error("--shapes must be comma-separated MxKxN values")
        if any(len(shape) != 3 for shape in SHAPES):
            parser.error("--shapes must be comma-separated MxKxN values")
    module, KERNELS_INFO = large.compile_kernels()
    if args.mode == "ncu":
        ncu_mode()
        return
    if args.output is None or args.output.exists():
        parser.error("--output is required for sustain and must not exist")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(metadata=dict(
        utc=datetime.utcnow().isoformat() + "Z", gpu=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()), torch=torch.__version__,
        triton=triton.__version__, cuda=torch.version.cuda,
        commit=subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
                                       text=True).strip(),
        protocol=f"CUDA graph of 80 calls replayed for {args.seconds} s per kernel; nvidia-smi every 50 ms"),
        **sustain_mode(args.seconds, args.output))
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
