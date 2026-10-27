"""Float64 GEMM: native DGEMM against cuBLAS FP64 fixed-point emulation (Ozaki), on MH-1's real shapes.

The emulation is chosen by environment variables read when cuBLAS creates its handle, so one process per setting.
Accuracy: 64 sampled rows against a long-double (80-bit) CPU reference, error scaled by |A| |B| (the GEMM's own
condition), for two operand distributions: normal, and rows/columns with exponents spread over 2^+-20.
Prints one JSON line.
"""
import json
import os
import sys
import time

import numpy as np
import torch

SHAPES = [(13608, 512, 512), (5832, 512, 512), (1944, 512, 512), (216, 512, 512), (1944, 2048, 384),
          (1944, 384, 2048), (56320, 1280, 65), (56320, 65, 1280), (56320, 64, 64), (7680, 1280, 65)]


def operands(m, k, n, kind, gen):
    a = torch.randn(m, k, generator=gen, dtype=torch.float64)
    b = torch.randn(k, n, generator=gen, dtype=torch.float64) / k ** 0.5
    if kind == "spread":
        a *= torch.exp2(torch.randint(-20, 21, (m, 1), generator=gen).double())
        a *= torch.exp2(torch.randint(-6, 7, (1, k), generator=gen).double())
        b *= torch.exp2(torch.randint(-6, 7, (k, 1), generator=gen).double())
    return a, b


def main():
    label = sys.argv[1]
    torch.backends.cuda.preferred_blas_library("cublas")
    gen = torch.Generator().manual_seed(7)
    out = {"label": label, "env": {k: v for k, v in os.environ.items() if k.startswith("CUBLAS")},
           "gpu": torch.cuda.get_device_name(), "shapes": []}
    for m, k, n in SHAPES:
        row = {"shape": [m, k, n]}
        for kind in ("normal", "spread"):
            a, b = operands(m, k, n, kind, gen)
            ag, bg = a.cuda(), b.cuda()
            c = ag @ bg
            torch.cuda.synchronize()
            rows = torch.randperm(m, generator=gen)[:64]
            ref = a[rows].numpy().astype(np.longdouble) @ b.numpy().astype(np.longdouble)
            scale = np.abs(a[rows].numpy()).astype(np.longdouble) @ np.abs(b.numpy()).astype(np.longdouble)
            err = np.abs(c[rows.cuda()].cpu().numpy().astype(np.longdouble) - ref) / scale
            row[kind] = {"max_err_over_absAB": float(err.max()), "rms": float(np.sqrt((err ** 2).mean()))}
            if kind == "normal":
                for _ in range(3):
                    ag @ bg
                torch.cuda.synchronize()
                reps = max(3, int(2e10 / (2 * m * k * n)))
                t = time.perf_counter()
                for _ in range(reps):
                    ag @ bg
                torch.cuda.synchronize()
                row["us"] = 1e6 * (time.perf_counter() - t) / reps
                row["tflops"] = 2 * m * k * n / row["us"] / 1e6
        out["shapes"].append(row)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        torch.randn(1944, 512, dtype=torch.float64, device="cuda") @ torch.randn(512, 512, dtype=torch.float64,
                                                                                 device="cuda")
        torch.cuda.synchronize()
    out["kernels"] = sorted({e.name[:90] for e in prof.events() if e.device_type.name == "CUDA"})
    print(json.dumps(out))


main()
