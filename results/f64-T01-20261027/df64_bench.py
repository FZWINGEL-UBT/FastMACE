"""Paired-float32 ("double-float") against native float64: a chain of fused multiply-adds per element in Triton.

Each element runs STEPS iterations of y = y * a + b with a, b per element. Native: float64 FMAs. Paired: (hi, lo)
float32 pairs with an exact product (fma-based two-product) and two-sum, about 20 float32 operations per step.
Reports elements per second and the paired result's error against float64. One JSON line.
"""
import json
import time

import torch
import triton
import triton.language as tl

STEPS = 64


@triton.jit
def native(a_ptr, b_ptr, y_ptr, n, STEPS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    a = tl.load(a_ptr + i, mask=m, other=0.0)
    b = tl.load(b_ptr + i, mask=m, other=0.0)
    y = b
    for _ in tl.static_range(STEPS):
        y = tl.fma(y, a, b)
    tl.store(y_ptr + i, y, mask=m)


@triton.jit
def _two_sum(a, b):
    s = a + b
    v = s - a
    e = (a - (s - v)) + (b - v)
    return s, e


@triton.jit
def paired(ah_ptr, al_ptr, bh_ptr, bl_ptr, yh_ptr, yl_ptr, n, STEPS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    ah = tl.load(ah_ptr + i, mask=m, other=0.0)
    al = tl.load(al_ptr + i, mask=m, other=0.0)
    bh = tl.load(bh_ptr + i, mask=m, other=0.0)
    bl = tl.load(bl_ptr + i, mask=m, other=0.0)
    yh, yl = bh, bl
    for _ in tl.static_range(STEPS):
        p = yh * ah
        pe = tl.fma(yh, ah, -p)  # exact product error
        pe = pe + (yh * al + yl * ah)
        ph, pl = _two_sum(p, pe)
        sh, se = _two_sum(ph, bh)
        se = se + pl + bl
        yh, yl = _two_sum(sh, se)
    tl.store(yh_ptr + i, yh, mask=m)
    tl.store(yl_ptr + i, yl, mask=m)


def split(x):
    hi = x.float()
    return hi, (x - hi.double()).float()


def timed(fn, reps=20):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


n, block = 1 << 24, 256
a = (0.9 + 0.1 * torch.rand(n, dtype=torch.float64, device="cuda"))
b = torch.rand(n, dtype=torch.float64, device="cuda")
y = torch.empty_like(a)
ah, al = split(a)
bh, bl = split(b)
yh, yl = torch.empty_like(ah), torch.empty_like(ah)
grid = (triton.cdiv(n, block),)
t64 = timed(lambda: native[grid](a, b, y, n, STEPS=STEPS, BLOCK=block))
t32 = timed(lambda: paired[grid](ah, al, bh, bl, yh, yl, n, STEPS=STEPS, BLOCK=block))
paired_value = yh.double() + yl.double()
err = ((paired_value - y).abs() / y.abs()).max().item()
print(json.dumps({"gpu": torch.cuda.get_device_name(), "elements": n, "steps": STEPS,
                  "native_fp64_gfma_per_s": n * STEPS / t64 / 1e9, "paired_fp32_gfma_per_s": n * STEPS / t32 / 1e9,
                  "speedup": t64 / t32, "paired_max_rel_err_vs_fp64": err}))
