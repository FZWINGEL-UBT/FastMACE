"""Hybrid GEMM lab (review retry, 11 November 2026): the CUTLASS hybrid kernel (hybrid_sm80.cu) against production's
3xTF32 (fastmace.kernels.cutlass.tf32x3, the same CUTLASS build) and cuBLAS FP32 on tools/fp16x3_gemm_bench.py's
OMAT-0 shapes. Error: max |C - exact| / (|A| |B|) against float64. Time: CUDA events, alternating blocks.

Stop rule (review): the hybrid's error at or below cuBLAS FP32's on each shape, and its shape-weighted time below
3xTF32's (weights: tools/fp16x3_gemm_bench.py has no profile weights, so the sum of per-shape times, each shape once,
and per shape).

    python results/hybrid-gemm-20261111/hybrid_lab.py --output FILE [--reps 50] [--blocks 5]
"""
import argparse
import ctypes
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
SHAPES = ((5000, 2048, 128), (5000, 512, 128), (35000, 128, 128), (35000, 128, 192), (35000, 384, 128),
          (35000, 192, 384), (15000, 1024, 192), (5000, 128, 128), (5000, 128, 384))


class Lab:
    def __init__(self, device):
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check
        self.driver, self.check = driver, _check
        cap = torch.cuda.get_device_capability(device)
        cubin = HERE / ("hybrid_sm80.cubin" if cap[0] == 8 else f"hybrid_sm{cap[0]}{cap[1]}.cubin")
        torch.zeros(1, device=device)
        self.module = _check(driver.cuModuleLoadData(cubin.read_bytes()))
        self.fn = {name: _check(driver.cuModuleGetFunction(self.module, f"fm_{name}_64x64".encode()))
                   for name in ("hyb_rr", "hyb_rc", "ref_rr")}
        pointer, size = _check(driver.cuModuleGetGlobal(self.module, b"fm_hyb_info"))
        info = np.zeros(size // 4, dtype=np.int32)
        _check(driver.cuMemcpyDtoH(info, pointer, size))
        self.threads = int(info[0])
        self.shared = {"hyb_rr": int(info[1]), "hyb_rc": int(info[2]), "ref_rr": int(info[3])}
        for name, f in self.fn.items():
            _check(driver.cuFuncSetAttribute(
                f, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, self.shared[name]))

    def mm(self, name, a, b, c):
        m, k = a.shape
        n = b.shape[1]
        p, i = ctypes.c_void_p, ctypes.c_int
        values = (m, n, k, a.data_ptr(), a.stride(0), b.data_ptr(), b.stride(0), c.data_ptr(), c.stride(0))
        types = (i, i, i, p, i, p, i, p, i)
        stream = self.driver.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
        self.check(self.driver.cuLaunchKernel(self.fn[name], (m + 63) // 64, (n + 63) // 64, 1, self.threads, 1, 1,
                                              self.shared[name], stream, (values, types), 0))


def timed(fns, reps, blocks):
    """Alternating blocks of ``reps`` calls of each fn: the median microseconds per call of each."""
    for fn in fns.values():
        for _ in range(5):
            fn()
    torch.cuda.synchronize()
    times = {k: [] for k in fns}
    for _ in range(blocks):
        for name, fn in fns.items():
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(reps):
                fn()
            e.record()
            torch.cuda.synchronize()
            times[name].append(s.elapsed_time(e) * 1000.0 / reps)
    return {k: float(np.median(v)) for k, v in times.items()}


def error(c, exact, scale):
    return float(((c.double() - exact).abs() / scale.clamp_min(1e-300)).max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--blocks", type=int, default=5)
    p.add_argument("--no-time", action="store_true")
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    lab = Lab(dev)
    tf32x3 = None
    try:
        from fastmace.kernels.cutlass import tf32x3 as t
        if t.available(dev):
            tf32x3 = t
    except Exception as exc:  # noqa: BLE001
        print("tf32x3 unavailable:", exc)
    g = torch.Generator(device=dev).manual_seed(0)
    rows = []
    for m, k, n in SHAPES:
        a = torch.randn((m, k), device=dev, generator=g) * torch.exp(torch.randn((m, 1), device=dev, generator=g))
        b = torch.randn((k, n), device=dev, generator=g) / k ** 0.5
        bt = b.t().contiguous().t()  # column-major B
        exact = a.double() @ b.double()
        scale = a.double().abs() @ b.double().abs()
        outs = {name: torch.empty((m, n), device=dev) for name in ("hyb", "hyb_rc", "ref", "prod", "fp32")}
        calls = {"hyb": lambda: lab.mm("hyb_rr", a, b, outs["hyb"]),
                 "hyb_rc": lambda: lab.mm("hyb_rc", a, bt, outs["hyb_rc"]),
                 "ref": lambda: lab.mm("ref_rr", a, b, outs["ref"]),
                 "fp32": lambda: torch.mm(a, b, out=outs["fp32"])}
        if tf32x3 is not None and tf32x3.eligible(a, b, outs["prod"]):
            calls["prod"] = lambda: tf32x3.mm_into(a, b, outs["prod"])
        for fn in calls.values():
            fn()
        torch.cuda.synchronize()
        row = {"m": m, "k": k, "n": n}
        for name in calls:
            row[f"{name}_err"] = error(outs[name], exact, scale)
        if not args.no_time:
            for name, us in timed(calls, args.reps, args.blocks).items():
                row[f"{name}_us"] = us
            row["hyb_over_ref"] = row["hyb_us"] / row["ref_us"]
            if "prod_us" in row:
                row["hyb_over_prod"] = row["hyb_us"] / row["prod_us"]
        rows.append(row)
        print(json.dumps({key: (round(v, 4) if isinstance(v, float) and v > 1e-3 else v) for key, v in row.items()}),
              flush=True)
    meta = {"gpu": torch.cuda.get_device_name(dev), "torch": torch.__version__, "rows": rows}
    if not args.no_time:
        meta["sum_hyb_us"] = sum(r["hyb_us"] for r in rows)
        meta["sum_ref_us"] = sum(r["ref_us"] for r in rows)
        meta["sum_hyb_over_ref"] = meta["sum_hyb_us"] / meta["sum_ref_us"]
        if all("prod_us" in r for r in rows):
            meta["sum_hyb_over_prod"] = meta["sum_hyb_us"] / sum(r["prod_us"] for r in rows)
    meta["hyb_err_le_fp32_every_shape"] = all(r["hyb_err"] <= r["fp32_err"] for r in rows)
    print(json.dumps({k: v for k, v in meta.items() if k != "rows"}))
    Path(args.output).write_text(json.dumps(meta, indent=1) + "\n")


if __name__ == "__main__":
    main()
