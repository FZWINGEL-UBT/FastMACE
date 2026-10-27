"""f64-T04 step 1 (falsifier A): double-float (df64) FMA-chain throughput and accuracy versus native float64.

Every element runs CHAINS independent recurrences x <- x*a + b for ITERS iterations, with a and b loaded once
from float64 memory. Variants (one CUDA source, NVRTC through cuda.bindings, --fmad=false so nothing is
contracted behind our back; the exact product uses explicit __fmaf_rn, every other op __fadd_rn/__fmul_rn):

  fp64        native float64 fma (the baseline; FP64 pipe)
  fp32        native float32 fma (speed of light of the FP32 pipe, for scale)
  dd_acc      double-float, Joldes-Muller-Popescu 2017 (ACM TOMS 44(2), "Tight and rigorous error bounds for
              basic building blocks of double-word arithmetic"): DWTimesDW3 (Alg. 12, FMA, 9 ops, bound 4u^2)
              then AccurateDWPlusDW (Alg. 6, 20 ops, bound 3u^2+13u^3); u = 2^-24
  dd_sloppy   same product, SloppyDWPlusDW (Alg. 5, 11 ops; no relative bound under cancellation - info only)

float64 -> pair: hi = rn32(x), lo = rn32(x - hi) (x - hi is exact in float64); pair -> float64: hi + lo in fp64.
Accuracy: every variant against an exact (Python decimal, 80 digits) evaluation of the same recurrence on a
sample of elements, and dd against native fp64 on all elements.
"""
import argparse
import ctypes
import decimal
import json
import statistics
import sys

import numpy as np
import torch
from cuda.bindings import driver, nvrtc

SOURCE = r"""
#define ITERS %(iters)d
#define CHAINS %(chains)d
struct df { float hi, lo; };

__device__ __forceinline__ df two_sum(float a, float b) {
    float s = __fadd_rn(a, b); float bb = __fadd_rn(s, -a);
    float e = __fadd_rn(__fadd_rn(a, -__fadd_rn(s, -bb)), __fadd_rn(b, -bb));
    return {s, e};
}
__device__ __forceinline__ df fast_two_sum(float a, float b) {  // |a| >= |b|
    float s = __fadd_rn(a, b); float z = __fadd_rn(s, -a);
    return {s, __fadd_rn(b, -z)};
}
// Joldes, Muller, Popescu 2017, Alg. 12 (DWTimesDW3): relative error <= 4u^2.
__device__ __forceinline__ df dd_mul(df x, df y) {
    float ch = __fmul_rn(x.hi, y.hi);
    float cl1 = __fmaf_rn(x.hi, y.hi, -ch);
    float tl0 = __fmul_rn(x.lo, y.lo);
    float tl1 = __fmaf_rn(x.hi, y.lo, tl0);
    float cl2 = __fmaf_rn(x.lo, y.hi, tl1);
    float cl3 = __fadd_rn(cl1, cl2);
    return fast_two_sum(ch, cl3);
}
// Alg. 6 (AccurateDWPlusDW): relative error <= 3u^2 + 13u^3.
__device__ __forceinline__ df dd_add(df x, df y) {
    df s = two_sum(x.hi, y.hi); df t = two_sum(x.lo, y.lo);
    float c = __fadd_rn(s.lo, t.hi);
    df v = fast_two_sum(s.hi, c);
    float w = __fadd_rn(t.lo, v.lo);
    return fast_two_sum(v.hi, w);
}
// Alg. 5 (SloppyDWPlusDW): no relative bound when x and y have opposite signs.
__device__ __forceinline__ df dd_add_sloppy(df x, df y) {
    df s = two_sum(x.hi, y.hi);
    float v = __fadd_rn(x.lo, y.lo);
    float w = __fadd_rn(s.lo, v);
    return fast_two_sum(s.hi, w);
}
__device__ __forceinline__ df to_df(double x) { float h = __double2float_rn(x); return {h, __double2float_rn(x - (double)h)}; }
__device__ __forceinline__ double from_df(df x) { return (double)x.hi + (double)x.lo; }

extern "C" __global__ void chain_fp64(long long n, const double* x0, const double* a, const double* b, double* out) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= n) return;
    double x[CHAINS], av[CHAINS], bv[CHAINS];
    for (int c = 0; c < CHAINS; ++c) { x[c] = x0[c * n + i]; av[c] = a[c * n + i]; bv[c] = b[c * n + i]; }
    #pragma unroll 8
    for (int k = 0; k < ITERS; ++k) {
        #pragma unroll
        for (int c = 0; c < CHAINS; ++c) x[c] = __fma_rn(x[c], av[c], bv[c]);
    }
    for (int c = 0; c < CHAINS; ++c) out[c * n + i] = x[c];
}
extern "C" __global__ void chain_fp32(long long n, const double* x0, const double* a, const double* b, double* out) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= n) return;
    float x[CHAINS], av[CHAINS], bv[CHAINS];
    for (int c = 0; c < CHAINS; ++c) { x[c] = (float)x0[c * n + i]; av[c] = (float)a[c * n + i]; bv[c] = (float)b[c * n + i]; }
    #pragma unroll 8
    for (int k = 0; k < ITERS; ++k) {
        #pragma unroll
        for (int c = 0; c < CHAINS; ++c) x[c] = __fmaf_rn(x[c], av[c], bv[c]);
    }
    for (int c = 0; c < CHAINS; ++c) out[c * n + i] = (double)x[c];
}
#define DD_KERNEL(NAME, ADD) \
extern "C" __global__ void NAME(long long n, const double* x0, const double* a, const double* b, double* out) { \
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= n) return; \
    df x[CHAINS], av[CHAINS], bv[CHAINS]; \
    for (int c = 0; c < CHAINS; ++c) { x[c] = to_df(x0[c * n + i]); av[c] = to_df(a[c * n + i]); bv[c] = to_df(b[c * n + i]); } \
    _Pragma("unroll 8") \
    for (int k = 0; k < ITERS; ++k) { \
        _Pragma("unroll") \
        for (int c = 0; c < CHAINS; ++c) x[c] = ADD(dd_mul(x[c], av[c]), bv[c]); \
    } \
    for (int c = 0; c < CHAINS; ++c) out[c * n + i] = from_df(x[c]); \
}
DD_KERNEL(chain_dd_acc, dd_add)
DD_KERNEL(chain_dd_sloppy, dd_add_sloppy)
"""

VARIANTS = ("fp64", "fp32", "dd_acc", "dd_sloppy")


def check(result):
    if int(result[0]) != 0:
        raise RuntimeError(f"CUDA/NVRTC call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def compile_module(iters, chains):
    source = SOURCE % {"iters": iters, "chains": chains}
    major, minor = torch.cuda.get_device_capability()
    program = check(nvrtc.nvrtcCreateProgram(source.encode(), b"df64.cu", 0, [], []))
    options = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"--std=c++17", b"--fmad=false"]
    result = nvrtc.nvrtcCompileProgram(program, len(options), options)
    if int(result[0]) != 0:
        size = check(nvrtc.nvrtcGetProgramLogSize(program))
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(program, log)
        raise RuntimeError(log.decode(errors="replace"))
    size = check(nvrtc.nvrtcGetCUBINSize(program))
    cubin = b" " * size
    check(nvrtc.nvrtcGetCUBIN(program, cubin))
    torch.cuda.current_stream()
    torch.zeros(1, device="cuda")
    module = check(driver.cuModuleLoadData(cubin))
    return {v: check(driver.cuModuleGetFunction(module, f"chain_{v}".encode())) for v in VARIANTS}


def launch(function, n, threads, tensors):
    stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
    p = ctypes.c_void_p
    values = (n,) + tuple(t.data_ptr() for t in tensors)
    types = (ctypes.c_longlong, p, p, p, p)
    check(driver.cuLaunchKernel(function, (n + threads - 1) // threads, 1, 1, threads, 1, 1, 0, stream,
                                (values, types), 0))


def exact(x0, a, b, iters):
    """Exact-to-80-digits value of the recurrence (decimal of the float64 inputs)."""
    decimal.getcontext().prec = 80
    x, av, bv = decimal.Decimal(float(x0)), decimal.Decimal(float(a)), decimal.Decimal(float(b))
    for _ in range(iters):
        x = x * av + bv
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1 << 20, help="threads (elements per chain slot)")
    parser.add_argument("--iters", type=int, default=512)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=7, help="alternating timing blocks")
    parser.add_argument("--sample", type=int, default=64, help="elements checked against the exact recurrence")
    parser.add_argument("--json", default=None)
    parser.add_argument("--a-low", type=float, default=0.99, help="a in [a_low, a_low + 0.9 (1 - a_low)]; the "
                        "recurrence's condition number is about 1 / (1 - a)")
    parser.add_argument("--pair-inputs", action="store_true", help="round x0, a, b to values a float32 pair "
                        "represents exactly, so the dd error is the arithmetic's alone")
    args = parser.parse_args()
    torch.manual_seed(0)
    n, total = args.n, args.n * args.chains
    gen = torch.Generator(device="cuda").manual_seed(1)
    # Contractive recurrence x <- a x + b, fixed point b/(1-a) in [-1, 1]; a in [a_low, a_low + 0.9 (1-a_low)]
    # (default [0.99, 0.999]), b of both signs so the dd add sees cancellation.
    x0 = torch.rand(total, device="cuda", dtype=torch.float64, generator=gen) * 2 - 1
    a = args.a_low + 0.9 * (1 - args.a_low) * torch.rand(total, device="cuda", dtype=torch.float64, generator=gen)
    b = (torch.rand(total, device="cuda", dtype=torch.float64, generator=gen) * 2 - 1) * (1 - a)
    if args.pair_inputs:
        def pair(t):
            hi = t.float().double()
            return hi + (t - hi).float().double()
        x0, a, b = pair(x0), pair(a), pair(b)
    fns = compile_module(args.iters, args.chains)
    outs = {v: torch.empty(total, device="cuda", dtype=torch.float64) for v in VARIANTS}
    for v in VARIANTS:
        launch(fns[v], n, args.threads, (x0, a, b, outs[v]))
    torch.cuda.synchronize()
    times = {v: [] for v in VARIANTS}
    reps = 5
    for _ in range(args.blocks):
        for v in VARIANTS:
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(reps):
                launch(fns[v], n, args.threads, (x0, a, b, outs[v]))
            stop.record()
            torch.cuda.synchronize()
            times[v].append(start.elapsed_time(stop) / reps)
    fma = total * args.iters
    result = {"device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
              "n_threads": n, "chains": args.chains, "a_low": args.a_low, "pair_inputs": args.pair_inputs, "iters": args.iters, "fmas": fma, "variants": {}}
    sample = torch.linspace(0, total - 1, args.sample).long().tolist()
    ref = {i: exact(x0[i].item(), a[i].item(), b[i].item(), args.iters) for i in sample}
    scale = max(abs(r) for r in ref.values())
    for v in VARIANTS:
        ms = statistics.median(times[v])
        out = outs[v].cpu().numpy()
        err_exact = max(abs(decimal.Decimal(float(out[i])) - ref[i]) / abs(ref[i]) for i in sample)
        err_exact_scaled = max(abs(decimal.Decimal(float(out[i])) - ref[i]) for i in sample) / scale
        rel_fp64 = float((outs[v] - outs["fp64"]).abs().max() / outs["fp64"].abs().max())
        result["variants"][v] = {"ms_median": ms, "ms_all": times[v], "gfma_per_s": fma / ms / 1e6,
                                 "max_rel_err_vs_exact_sample": float(err_exact),
                                 "max_err_vs_exact_sample_over_max": float(err_exact_scaled),
                                 "max_abs_diff_vs_fp64_over_max": rel_fp64}
    base = result["variants"]["fp64"]["ms_median"]
    for v in VARIANTS:
        result["variants"][v]["speedup_vs_fp64"] = base / result["variants"][v]["ms_median"]
    print(json.dumps(result, indent=1))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=1)


if __name__ == "__main__":
    sys.exit(main())
