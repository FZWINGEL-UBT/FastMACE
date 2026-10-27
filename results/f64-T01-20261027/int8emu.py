"""Float32 GEMM emulated with INT8 slices and 1-D block scaling (integer Ozaki scheme, as FlashSO(2), arXiv 2610.35164).

A [M, K] and B [K, N] float32. Per row of A (column of B) and per block of 128 entries along K: a power-of-two scale
just above the block's largest |entry|, then p signed INT8 digits (truncated base-128 digits, the last rounded).
C = sum over K blocks of sA * sB * sum_{i+j <= p+1} 2^-7(i+j) (A_i @ B_j), each A_i @ B_j exact in INT32.
B's slices are made once (constant weights in MD); A's slicing is timed separately (torch ops, not fused).

Reports per shape: cuBLAS SGEMM time (TF32 off), INT8 cuBLASLt time (torch._int_mm, one plain product), the
emulated kernel's time for p = 3 and 4, A's slicing time, and the errors of SGEMM and of each emulation against a
float64 product, scaled by |A| |B|. One JSON line.
"""
import json
import time

import torch
import triton
import triton.language as tl

torch.backends.cuda.matmul.allow_tf32 = False
SHAPES = [(13608, 512, 512), (5832, 512, 512), (1944, 512, 512), (216, 512, 512), (1944, 2048, 384),
          (1944, 384, 2048), (56320, 1280, 64), (56320, 128, 1280), (7680, 1280, 64), (56320, 64, 64)]
BK = 128


@triton.jit
def emu_kernel(a_ptr, sa_ptr, b_ptr, sb_ptr, c_ptr, M, N, K, KB,
               P: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BKC: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BKC)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kb in range(0, KB):
        k = kb * BKC + rk
        part = tl.zeros((BM, BN), dtype=tl.float32)
        for i in tl.static_range(P):
            a = tl.load(a_ptr + i * M * K + rm[:, None] * K + k[None, :], mask=(rm[:, None] < M) & (k[None, :] < K),
                        other=0)
            for j in tl.static_range(P - i):
                b = tl.load(b_ptr + j * K * N + k[:, None] * N + rn[None, :],
                            mask=(k[:, None] < K) & (rn[None, :] < N), other=0)
                d = tl.dot(a, b, out_dtype=tl.int32)
                part += d.to(tl.float32) * (2.0 ** (-7.0 * (i + j + 2)))
        sa = tl.load(sa_ptr + rm * KB + kb, mask=rm < M, other=0.0)
        sb = tl.load(sb_ptr + kb * N + rn, mask=rn < N, other=0.0)
        acc += part * sa[:, None] * sb[None, :]
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def slice_rows(x, p):
    """x [R, K] float32 -> (digits [p, R, K] int8, scales [R, K/BK] float32): blocks of BK along K."""
    r, k = x.shape
    kb = -(-k // BK)
    pad = torch.zeros((r, kb * BK), dtype=x.dtype, device=x.device)
    pad[:, :k] = x
    blocks = pad.view(r, kb, BK)
    top = blocks.abs().amax(dim=2).clamp_min(1e-30)
    scale = torch.exp2(torch.floor(torch.log2(top)) + 1.0)  # |x / scale| < 1
    rest = blocks / scale[:, :, None]
    digits = []
    for i in range(p):
        rest = rest * 128.0
        d = torch.round(rest) if i == p - 1 else torch.trunc(rest)
        d = d.clamp(-127, 127)
        digits.append(d)
        rest = rest - d
    out = torch.stack(digits).to(torch.int8).view(p, r, kb * BK)[:, :, :k].contiguous()
    return out, scale.contiguous()


def emulate(a_digits, sa, b_digits, sb, m, n, k, p, bm=64, bn=64):
    c = torch.empty((m, n), dtype=torch.float32, device="cuda")
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    emu_kernel[grid](a_digits, sa, b_digits, sb, c, m, n, k, sa.shape[1], P=p, BM=bm, BN=bn, BKC=BK, num_stages=2)
    return c


def timed(fn, reps=None, flops=None):
    fn()
    torch.cuda.synchronize()
    reps = reps or 10
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return 1e6 * (time.perf_counter() - t) / reps


def main():
    gen = torch.Generator(device="cuda").manual_seed(3)
    rows = []
    for m, k, n in SHAPES:
        a = torch.randn(m, k, device="cuda", generator=gen)
        a *= torch.exp2(torch.randint(-8, 9, (m, 1), device="cuda", generator=gen).float())  # row-scale spread
        b = torch.randn(k, n, device="cuda", generator=gen) / k ** 0.5
        ref = a.double() @ b.double()
        scale = a.double().abs() @ b.double().abs()
        row = {"shape": [m, k, n]}
        reps = max(5, min(200, int(3e9 / (2 * m * k * n))))
        row["sgemm_us"] = timed(lambda: a @ b, reps)
        row["sgemm_err"] = float(((a @ b).double() - ref).abs().div(scale).max())
        if m % 8 == 0 and k % 8 == 0 and n % 8 == 0 and m > 16:
            ai = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda")
            bi = torch.randint(-127, 128, (k, n), dtype=torch.int8, device="cuda")
            row["int8_one_product_us"] = timed(lambda: torch._int_mm(ai, bi), reps)
        bt = b.t().contiguous()
        for p in (3, 4):
            bd, sbt = slice_rows(bt, p)  # columns of B as rows
            bdig = bd.transpose(1, 2).contiguous()  # [p, K, N]
            sb = sbt.t().contiguous()  # [KB, N]
            ad, sa = slice_rows(a, p)
            c = emulate(ad, sa, bdig, sb, m, n, k, p, 32, 64)
            row[f"emu{p}_err"] = float((c.double() - ref).abs().div(scale).max())
            best = None
            for bm, bn in ((32, 64), (64, 64), (128, 64), (64, 128)):
                try:
                    us = timed(lambda: emulate(ad, sa, bdig, sb, m, n, k, p, bm, bn), reps)
                except triton.runtime.errors.OutOfResources:
                    continue
                best = us if best is None or us < best else best
            row[f"emu{p}_us"] = best
            row[f"slice_a{p}_us"] = timed(lambda: slice_rows(a, p), reps)
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "rows": len(rows)}))


main()
