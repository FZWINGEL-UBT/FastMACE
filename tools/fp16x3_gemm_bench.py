"""3xFP16 GEMM upside on the A100 (round 4 idea, 5 November 2026): time and accuracy against the CUTLASS 3xTF32 kernel.

The A100's FP16 tensor cores run at twice its TF32 rate, and FP16 has TF32's 11-bit significand. With A's rows and B's
columns scaled by powers of two into [0.5, 1), each operand splits into an FP16 high part and a low part stored times
2^11 (no underflow for the row's large elements), and one FP16 GEMM over K' = 3K with FP32 accumulation,

    [2^11 Ah, Ah, Al'] @ [Bh; Bl'; Bh] = 2^11 (Ah Bh + Ah Bl + Al Bh),

gives the same three products as 3xTF32. This tool times, per shape of the OMAT-0 Benchmark_loup profile,
(1) ``tf32x3.mm_into`` (CUTLASS, sm_80 only), (2) cuBLAS FP32 (IEEE), (3) the FP16 GEMM alone and (4) its split
prologue (row max, split, concatenation; compiled), and reports each one's error against float64 relative to
|A| |B|. Not part of the library.

    python tools/fp16x3_gemm_bench.py [--shapes M,K,N ...] [--output FILE]
"""

import argparse
import json
import sys

import torch

SHAPES = ((5000, 2048, 128), (5000, 512, 128), (35000, 128, 128), (35000, 128, 192), (35000, 384, 128),
          (35000, 192, 384), (15000, 1024, 192))


def pow2_scale(t, dim):
    """2^-e with max |t| along ``dim`` times it in [0.5, 1) (1 where all zero)."""
    m = t.abs().amax(dim=dim, keepdim=True)
    e = torch.where(m > 0, torch.floor(torch.log2(m)) + 1, torch.zeros_like(m))
    return torch.exp2(-e)


def split_a(a):
    """(A' [M, 3K] fp16, row scale [M, 1]): [2^11 Ah, Ah, 2^11 Al] of the row-scaled A."""
    s = pow2_scale(a, 1)
    x = a * s
    hi = x.half()
    lo = ((x - hi.float()) * 2048.0).half()
    return torch.cat([hi * 2048.0, hi, lo], dim=1), s


def split_b(b):
    """(B' [3K, N] fp16, column scale [1, N]): [Bh; 2^11 Bl; Bh] of the column-scaled B."""
    s = pow2_scale(b, 0)
    x = b * s
    hi = x.half()
    lo = ((x - hi.float()) * 2048.0).half()
    return torch.cat([hi, lo, hi], dim=0), s


def hybrid_parts(a, b):
    """Idea 4 (11 Nov): big*big in TF32, the two cross products in one FP16 GEMM over 2K.

    Returns (Ah fp32 [M, K], Bh fp32 [K, N], A2 fp16 [M, 2K] = [Ah, 2^11 Al], B2 fp16 [2K, N] = [2^11 Bl; Bh],
    row scale, column scale). Ah and Bh have 11-bit significands, so the TF32 product of them is exact per term.
    """
    sa, sb = pow2_scale(a, 1), pow2_scale(b, 0)
    xa, xb = a * sa, b * sb
    ah, bh = xa.half(), xb.half()
    al, bl = ((xa - ah.float()) * 2048.0).half(), ((xb - bh.float()) * 2048.0).half()
    return ah.float(), bh.float(), torch.cat([ah, al], dim=1), torch.cat([bl, bh], dim=0), sa, sb


def hybrid(parts):
    ah, bh, a2, b2, sa, sb = parts
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        big = torch.mm(ah, bh)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = False
    cross = torch.mm(a2, b2, out_dtype=torch.float32)
    return (big + cross * (1.0 / 2048.0)) / sa / sb


def tf32_mm(x, y):
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        return torch.mm(x, y)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = False


def fp16x3(a_split, b_split):
    (a3, ra), (b3, cb) = a_split, b_split
    return torch.mm(a3, b3, out_dtype=torch.float32) * (1.0 / 2048.0) / ra / cb


def timed(fn, reps=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / reps


def error(c, exact, scale):
    return float(((c.double() - exact).abs() / scale.clamp_min(1e-300)).max())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shapes", nargs="*", default=None, metavar="M,K,N")
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)
    shapes = [tuple(int(v) for v in s.split(",")) for s in args.shapes] if args.shapes else SHAPES
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    sys.path.insert(0, ".")
    try:
        from fastmace.kernels.cutlass import tf32x3
        cutlass = tf32x3.available(dev)
    except Exception as exc:  # noqa: BLE001 - the tool reports what it can
        print("tf32x3 unavailable:", exc)
        cutlass = False
    g = torch.Generator(device=dev).manual_seed(0)
    split_a_compiled = torch.compile(split_a)
    rows = []
    for m, k, n in shapes:
        # Realistic magnitudes: a row-varying scale (activations differ by orders of magnitude between rows) and
        # a few small entries.
        a = torch.randn((m, k), device=dev, generator=g) * torch.exp(torch.randn((m, 1), device=dev, generator=g))
        b = torch.randn((k, n), device=dev, generator=g) / k ** 0.5
        exact = a.double() @ b.double()
        scale = a.double().abs() @ b.double().abs()
        row = {"m": m, "k": k, "n": n}
        c = torch.empty((m, n), device=dev)
        if cutlass and tf32x3.eligible(a, b, c):
            row["tf32x3_us"] = timed(lambda: tf32x3.mm_into(a, b, c))
            tf32x3.mm_into(a, b, c)
            row["tf32x3_err"] = error(c, exact, scale)
        row["fp32_us"] = timed(lambda: torch.mm(a, b))
        row["fp32_err"] = error(torch.mm(a, b), exact, scale)
        a_split, b_split = split_a(a), split_b(b)
        a3, b3 = a_split[0], b_split[0]
        row["fp16_gemm_us"] = timed(lambda: torch.mm(a3, b3, out_dtype=torch.float32))
        row["fp16_split_a_us"] = timed(lambda: split_a_compiled(a))
        row["fp16x3_err"] = error(fp16x3(a_split, b_split), exact, scale)
        parts = hybrid_parts(a, b)
        ah, bh, a2, b2 = parts[:4]
        row["hyb_tf32_us"] = timed(lambda: tf32_mm(ah, bh))
        row["hyb_fp16_2k_us"] = timed(lambda: torch.mm(a2, b2, out_dtype=torch.float32))
        row["hyb_sum_us"] = row["hyb_tf32_us"] + row["hyb_fp16_2k_us"]
        if "tf32x3_us" in row:
            row["hyb_over_tf32x3"] = row["hyb_sum_us"] / row["tf32x3_us"]
        row["hybrid_err"] = error(hybrid(parts), exact, scale)
        rows.append(row)
        print(json.dumps({key: (round(v, 2) if isinstance(v, float) and v > 1e-3 else v) for key, v in row.items()}),
              flush=True)
    meta = {"gpu": torch.cuda.get_device_name(dev), "torch": torch.__version__, "rows": rows}
    if args.output:
        with open(args.output, "w") as f:
            json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
