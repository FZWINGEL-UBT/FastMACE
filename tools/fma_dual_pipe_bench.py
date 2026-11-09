"""Lab only: can the FP64 pipe act as a second FMA pipe beside FP32 (round 2 idea 1, 9 Nov 2026)?

Each thread runs NF independent FFMA chains and ND independent DFMA chains (fmaf(f, a, c) and fma(d, a, c) with
runtime a, c, so nothing folds). Mixes: FP32 only, FP64 only, and FP32:FP64 instruction ratios 1:1, 2:1 and 3:1.
The gate of the idea: on the A100 a 2:1 mix must reach at least 85 FMA results per clock per SM (FP32 alone peaks at
64), sustained for about 10 s (``--seconds``) so power capping shows. The SM clock is read from NVML during the run
(nvidia-smi), not assumed.

Second round (9 Nov 2026, "a second FP64 pipe on the H200 too"): the FP64 tensor cores. NM independent
mma.sync.m8n8k4 f64 chains per thread (one warp instruction = 256 FP64 multiply-adds on the tensor pipe, 8 per lane),
alone and mixed with FFMA chains (--dmma). On the H200 the FMA work is issue-bound, so only an instruction that does
more per issue can add throughput; DMMA does 8x a DFMA.

Usage: python tools/fma_dual_pipe_bench.py --output DIR [--seconds 10] [--dmma]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import threading
import time
from pathlib import Path

MIXES = {"fp32": (8, 0), "fp64": (0, 8), "1:1": (8, 8), "2:1": (8, 4), "3:1": (12, 4), "4:1": (16, 4)}
# (FFMA chains, DFMA chains, DMMA chains): FFMA instructions per DMMA instruction 0, 4, 8, 16
DMMA_MIXES = {"fp32": (8, 0, 0), "m:dmma4": (0, 0, 4), "m:dmma8": (0, 0, 8), "m:4:1": (8, 0, 2), "m:8:1": (16, 0, 2),
              "m:8:1b": (8, 0, 1), "m:16:1": (16, 0, 1)}
DMMA = ('asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0, %1}, {%2}, {%3}, {%0, %1};" '
        ': "+d"(m{i}x), "+d"(m{i}y) : "d"(ad), "d"(cd));')
INNER = 32


def source(nf, nd, nm=0):
    lines = ['extern "C" __global__ void __launch_bounds__(256) fma_mix(float* out, double* dout, int iters, '
             'float a, float c, double ad, double cd) {',
             "    const float base = threadIdx.x * 1e-6f;"]
    lines += [f"    float f{i} = base + {i}.f;" for i in range(nf)]
    lines += [f"    double d{i} = (double)base + {i}.0;" for i in range(nd)]
    lines += [f"    double m{i}x = (double)base + {i}.0, m{i}y = (double)base - {i}.0;" for i in range(nm)]
    lines.append("    for (int it = 0; it < iters; ++it) {")
    lines.append("        #pragma unroll")
    lines.append(f"        for (int j = 0; j < {INNER}; ++j) {{")
    lines += [f"            f{i} = fmaf(f{i}, a, c);" for i in range(nf)]
    lines += [f"            d{i} = fma(d{i}, ad, cd);" for i in range(nd)]
    lines += ["            " + DMMA.replace("{i}", str(i)) for i in range(nm)]
    lines += ["        }", "    }"]
    fsum = " + ".join(f"f{i}" for i in range(nf)) or "0.f"
    dsum = " + ".join([f"d{i}" for i in range(nd)] + [f"m{i}x + m{i}y" for i in range(nm)]) or "0.0"
    lines.append("    const int t = blockIdx.x * blockDim.x + threadIdx.x;")
    lines.append(f"    out[t] = {fsum};")
    lines.append(f"    dout[t] = {dsum};")
    lines.append("}")
    return "\n".join(lines)


def sm_clock_sampler(samples, stop):
    while not stop.is_set():
        r = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True)
        try:
            mhz, watts, temp = (float(v) for v in r.stdout.strip().split("\n")[0].split(","))
            samples.append((mhz, watts, temp))
        except ValueError:
            pass
        time.sleep(0.2)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--blocks-per-sm", type=int, default=8)
    p.add_argument("--dmma", action="store_true", help="the FP64 tensor-core mixes (DMMA_MIXES) instead of MIXES")
    args = p.parse_args(argv)
    import torch
    from cuda.bindings import driver

    from fastmace.kernels.oeq_streaming.runtime import _check, compile_cubin

    args.output.mkdir(parents=True, exist_ok=True)
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    blocks, threads = sms * args.blocks_per_sm, 256
    out = torch.empty(blocks * threads, device="cuda", dtype=torch.float32)
    dout = torch.empty(blocks * threads, device="cuda", dtype=torch.float64)
    cap = torch.cuda.get_device_capability()
    stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
    results = dict(gpu=torch.cuda.get_device_name(), sms=sms, blocks=blocks, threads=threads, inner=INNER, mixes={})
    print(f"{results['gpu']}: {sms} SMs, {blocks} blocks x {threads} threads", flush=True)
    for name, (nf, nd, *rest) in (DMMA_MIXES if args.dmma else MIXES).items():
        nm = rest[0] if rest else 0
        src = source(nf, nd, nm)
        (args.output / f"fma_mix_{name.replace(':', '_')}.cu").write_text(src)
        module = _check(driver.cuModuleLoadData(compile_cubin(src, cap, ())))
        fn = _check(driver.cuModuleGetFunction(module, b"fma_mix"))
        regs = int(_check(driver.cuFuncGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, fn)))

        def launch(iters):
            _check(driver.cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, stream,
                                         ((out.data_ptr(), dout.data_ptr(), iters, 0.999, 1e-7, 0.999, 1e-7),
                                          (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_float,
                                           ctypes.c_float, ctypes.c_double, ctypes.c_double)), 0))
        iters = 64
        while True:  # size one launch to about 50 ms
            launch(iters)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            launch(iters)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if dt > 0.05:
                break
            iters *= 2
        samples, stop = [], threading.Event()
        sampler = threading.Thread(target=sm_clock_sampler, args=(samples, stop), daemon=True)
        sampler.start()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        launches = max(1, int(args.seconds / dt))
        start.record()
        for _ in range(launches):
            launch(iters)
        end.record()
        torch.cuda.synchronize()
        stop.set()
        sampler.join()
        seconds = start.elapsed_time(end) / 1e3
        steady = samples[len(samples) // 4:] or samples
        mhz = sum(s[0] for s in steady) / max(1, len(steady))
        watts = max((s[1] for s in steady), default=0.0)
        per_thread = iters * INNER * launches
        ffma = nf * per_thread * blocks * threads
        dfma = nd * per_thread * blocks * threads
        dmma = 8 * nm * per_thread * blocks * threads  # 256 multiply-adds per warp instruction, 8 per lane
        clocks = seconds * mhz * 1e6
        rec = dict(nf=nf, nd=nd, nm=nm, registers=regs, dmma_per_clk_sm=dmma / clocks / sms, seconds=seconds, sm_mhz=mhz, max_watts=watts,
                   min_mhz=min((s[0] for s in steady), default=0.0),
                   ffma_per_clk_sm=ffma / clocks / sms, dfma_per_clk_sm=dfma / clocks / sms,
                   fma_per_clk_sm=(ffma + dfma + dmma) / clocks / sms)
        results["mixes"][name] = rec
        print(f"mix {name}: FFMA {rec['ffma_per_clk_sm']:.1f} + DFMA {rec['dfma_per_clk_sm']:.1f} + "
              f"DMMA {rec['dmma_per_clk_sm']:.1f} = "
              f"{rec['fma_per_clk_sm']:.1f} FMA results/clk/SM at {mhz:.0f} MHz (min {rec['min_mhz']:.0f}), "
              f"max {watts:.0f} W, {seconds:.1f} s, registers {regs}", flush=True)
    name = "fma_dmma.json" if args.dmma else "fma_dual_pipe.json"
    (args.output / name).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
