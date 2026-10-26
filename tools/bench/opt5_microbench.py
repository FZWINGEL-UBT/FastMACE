#!/usr/bin/env python3
"""Wave-1 falsifier microbenchmarks for Optimisation Round 5 (OPT5-T02, OPT5-T03, OPT5-T04).

Runs inside a single quiet ``local_gpu.py`` reservation and writes JSON records under
``results/opt5-wave1-20261026/``.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


def graph_time_us(fn, warmup: int = 10, reps: int = 50, blocks: int = 7) -> tuple[float, list[float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(blocks):
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6 / reps)
    return round(statistics.median(times), 2), [round(x, 2) for x in times]


# ==============================================================================
# OPT5-T02: Unbiased cvt.rna.tf32 split + DeepSeek-style SIMT-producer GEMM
# ==============================================================================

NVRTC_T02_SRC = r"""
extern "C" __global__ void split_rna_tf32_kernel(
    const float* __restrict__ x, float* __restrict__ big_out, float* __restrict__ small_out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    unsigned b_bits, s_bits;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(b_bits) : "f"(v));
    float b = __uint_as_float(b_bits);
    float rem = v - b;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(s_bits) : "f"(rem));
    big_out[i] = b;
    small_out[i] = __uint_as_float(s_bits);
  }
}

extern "C" __global__ void split_rz_tf32_kernel(
    const float* __restrict__ x, float* __restrict__ big_out, float* __restrict__ small_out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    unsigned bits = __float_as_uint(x[i]);
    unsigned b_bits = bits & 0xFFFFE000u;
    float b = __uint_as_float(b_bits);
    float rem = x[i] - b;
    unsigned s_bits = __float_as_uint(rem) & 0xFFFFE000u;
    big_out[i] = b;
    small_out[i] = __uint_as_float(s_bits);
  }
}
"""


@triton.jit
def triton_rna_tf32x3_gemm(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    INCLUDE_SMALL_SMALL: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = (pm * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = (pn * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    rk = tl.arange(0, BLOCK_K).to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + rk
        a = tl.load(a_ptr + rm[:, None] * stride_am + k[None, :] * stride_ak,
                    mask=(rm < M)[:, None] & (k < K)[None, :], other=0.0)
        b = tl.load(b_ptr + k[:, None] * stride_bk + rn[None, :] * stride_bn,
                    mask=(k < K)[:, None] & (rn < N)[None, :], other=0.0)
        # Unbiased round-to-nearest-even TF32 split in inline PTX on SIMT cores before MMA:
        a_big = tl.inline_asm_elementwise(
            "cvt.rna.tf32.f32 $0, $1;", "=r,f", [a], dtype=tl.float32, is_pure=True, pack=1
        )
        a_rem = a - a_big
        a_small = tl.inline_asm_elementwise(
            "cvt.rna.tf32.f32 $0, $1;", "=r,f", [a_rem], dtype=tl.float32, is_pure=True, pack=1
        )
        b_big = tl.inline_asm_elementwise(
            "cvt.rna.tf32.f32 $0, $1;", "=r,f", [b], dtype=tl.float32, is_pure=True, pack=1
        )
        b_rem = b - b_big
        b_small = tl.inline_asm_elementwise(
            "cvt.rna.tf32.f32 $0, $1;", "=r,f", [b_rem], dtype=tl.float32, is_pure=True, pack=1
        )
        acc += tl.dot(a_big, b_big, input_precision="tf32")
        acc += tl.dot(a_big, b_small, input_precision="tf32")
        acc += tl.dot(a_small, b_big, input_precision="tf32")
        if INCLUDE_SMALL_SMALL:
            acc += tl.dot(a_small, b_small, input_precision="tf32")

    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc,
             mask=(rm < M)[:, None] & (rn < N)[None, :])


def run_t02() -> dict:
    from fastmace.kernels.triton.tf32x3_gemm import tf32x3_mm

    M, N, K = 5832, 512, 512
    torch.manual_seed(20261026)
    # Test both standard Gaussian operands and positive-mean operands that expose truncation bias in sum(C)
    a = torch.randn((M, K), device="cuda", dtype=torch.float32) + 0.25
    b = torch.randn((K, N), device="cuda", dtype=torch.float32) + 0.25
    ref64 = a.double() @ b.double()
    abs64 = a.double().abs() @ b.double().abs()

    c_cublas = a @ b
    c_triton_tf32x3 = tf32x3_mm(a, b)

    c_rna3 = torch.empty((M, N), device="cuda", dtype=torch.float32)
    c_rna4 = torch.empty((M, N), device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    triton_rna_tf32x3_gemm[grid](
        a, b, c_rna3, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_rna3.stride(0), c_rna3.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, INCLUDE_SMALL_SMALL=False, num_warps=4, num_stages=3
    )
    triton_rna_tf32x3_gemm[grid](
        a, b, c_rna4, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_rna4.stride(0), c_rna4.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, INCLUDE_SMALL_SMALL=True, num_warps=4, num_stages=3
    )

    def stats(c):
        diff = c.double() - ref64
        return {
            "mean_signed_bias": float(diff.mean()),
            "sum_signed_bias": float(diff.sum()),
            "max_abs_err": float(diff.abs().max()),
            "rms_err": float(torch.sqrt(torch.mean(diff ** 2))),
            "cancel_scaled_max": float((diff.abs() / abs64).max()),
        }

    t_cublas, _ = graph_time_us(lambda: torch.mm(a, b, out=c_cublas))
    t_triton_tf32x3, _ = graph_time_us(lambda: tf32x3_mm(a, b))
    t_rna3, _ = graph_time_us(
        lambda: triton_rna_tf32x3_gemm[grid](
            a, b, c_rna3, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c_rna3.stride(0), c_rna3.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, INCLUDE_SMALL_SMALL=False, num_warps=4, num_stages=3
        )
    )
    t_rna4, _ = graph_time_us(
        lambda: triton_rna_tf32x3_gemm[grid](
            a, b, c_rna4, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c_rna4.stride(0), c_rna4.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, INCLUDE_SMALL_SMALL=True, num_warps=4, num_stages=3
        )
    )

    return {
        "shape": [M, N, K],
        "times_us": {
            "cublas_ieee_fp32": t_cublas,
            "triton_builtin_tf32x3": t_triton_tf32x3,
            "triton_ptx_rna_3xTF32": t_rna3,
            "triton_ptx_rna_4xTF32": t_rna4,
        },
        "accuracy": {
            "cublas_ieee_fp32": stats(c_cublas),
            "triton_builtin_tf32x3": stats(c_triton_tf32x3),
            "triton_ptx_rna_3xTF32": stats(c_rna3),
            "triton_ptx_rna_4xTF32": stats(c_rna4),
        },
    }


# ==============================================================================
# OPT5-T03: Radial folded last layer forward [E, 65] @ [1280, 65]^T
# ==============================================================================

@triton.jit
def radial_fwd_64_plus_tail(
    a_ptr, w64_ptr, tail_w_ptr, out_ptr,
    E, Q: tl.constexpr, a_stride_m, w_stride_k,
    TAIL_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    m = (pm * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    n = (pn * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    mask_m = m < E
    k0 = tl.arange(0, 32).to(tl.int64)
    k1 = (32 + tl.arange(0, 32)).to(tl.int64)

    a0 = tl.load(a_ptr + m[:, None] * a_stride_m + k0[None, :], mask=mask_m[:, None], other=0.0)
    w0 = tl.load(w64_ptr + k0[:, None] * w_stride_k + n[None, :])
    acc = tl.dot(a0, w0, input_precision="ieee")

    a1 = tl.load(a_ptr + m[:, None] * a_stride_m + k1[None, :], mask=mask_m[:, None], other=0.0)
    w1 = tl.load(w64_ptr + k1[:, None] * w_stride_k + n[None, :])
    acc += tl.dot(a1, w1, input_precision="ieee")

    for t in tl.static_range(0, TAIL_K):
        a_t = tl.load(a_ptr + m * a_stride_m + (64 + t), mask=mask_m, other=0.0)
        w_t = tl.load(tail_w_ptr + t * Q + n)
        acc += a_t[:, None] * w_t[None, :]

    tl.store(out_ptr + m[:, None] * Q + n[None, :], acc, mask=mask_m[:, None], eviction_policy="evict_first")


def run_t03() -> dict:
    Q = 1280
    out_dict = {}
    cases = [("NaCl_8192", 8192, 150.0), ("NbCl5_56320", 56320, 1000.0), ("NaNbO3_82688", 82688, 1450.0)]
    for name, E, limit_us in cases:
        torch.manual_seed(0)
        a65 = torch.randn((E, 65), device="cuda", dtype=torch.float32)
        w65 = torch.randn((Q, 65), device="cuda", dtype=torch.float32)
        w65_t = w65.t().contiguous()
        w64_t = w65[:, :64].t().contiguous()
        tail1 = w65[:, 64:].t().contiguous()

        a68 = torch.randn((E, 68), device="cuda", dtype=torch.float32)
        w68 = torch.randn((Q, 68), device="cuda", dtype=torch.float32)
        w68_t = w68.t().contiguous()
        w68_64_t = w68[:, :64].t().contiguous()
        tail4 = w68[:, 64:].t().contiguous()

        out = torch.empty((E, Q), device="cuda", dtype=torch.float32)
        ref65 = a65.double() @ w65.double().t()
        c_prod65 = a65 @ w65.t()
        err_prod65 = float((c_prod65.double() - ref65).abs().max())

        row = {"E": E, "falsifier_limit_us": limit_us}
        row["cublas_k65_noncontig_wt_us"], _ = graph_time_us(lambda: torch.mm(a65, w65.t(), out=out))
        row["cublas_k65_contig_wt_us"], _ = graph_time_us(lambda: torch.mm(a65, w65_t, out=out))
        row["cublas_k68_noncontig_wt_us"], _ = graph_time_us(lambda: torch.mm(a68, w68.t(), out=out))
        row["cublas_k68_contig_wt_us"], _ = graph_time_us(lambda: torch.mm(a68, w68_t, out=out))
        row["cublas_k64_only_us"], _ = graph_time_us(lambda: torch.mm(a65[:, :64], w64_t, out=out))
        row["fill_only_us"], _ = graph_time_us(lambda: out.fill_(0.0))

        for bm, bn, nw, ns in [(64, 128, 4, 3), (64, 256, 8, 3), (128, 128, 8, 3), (64, 64, 4, 3)]:
            grid = (triton.cdiv(E, bm), Q // bn)

            def run65(bm=bm, bn=bn, nw=nw, ns=ns, grid=grid):
                radial_fwd_64_plus_tail[grid](
                    a65, w64_t, tail1, out, E, Q, 65, Q, TAIL_K=1, BLOCK_M=bm, BLOCK_N=bn, num_warps=nw, num_stages=ns
                )

            run65()
            err65 = float((out.double() - ref65).abs().max())
            med65, _ = graph_time_us(run65)
            row[f"triton_k65_{bm}x{bn}_w{nw}_us"] = med65
            row[f"triton_k65_{bm}x{bn}_w{nw}_err_ratio"] = round(err65 / err_prod65, 3)

            def run68(bm=bm, bn=bn, nw=nw, ns=ns, grid=grid):
                radial_fwd_64_plus_tail[grid](
                    a68, w68_64_t, tail4, out, E, Q, 68, Q, TAIL_K=4,
                    BLOCK_M=bm, BLOCK_N=bn, num_warps=nw, num_stages=ns
                )

            med68, _ = graph_time_us(run68)
            row[f"triton_k68_{bm}x{bn}_w{nw}_us"] = med68

        out_dict[name] = row
    return out_dict


# ==============================================================================
# OPT5-T04: Fused GatedRows (gated_segment_gemm: gate scale in GEMM prologue)
# ==============================================================================

@triton.jit
def gated_segment_gemm_kernel(
    a_ptr, gate_ptr, b_ptr, c_ptr,
    rows, N, K, D,
    a_n, a_m, gate_n, b_k, b_n, c_n, c_m,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pm, pn = tl.program_id(0), tl.program_id(1)
    r = (pm * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    j = (pn * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    atom, comp = r // D, r % D
    a_row = atom * a_n + comp * a_m
    g_row = atom * gate_n
    kk = tl.arange(0, BLOCK_K).to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in range(0, K, BLOCK_K):
        k = start + kk
        a = tl.load(a_ptr + a_row[:, None] + k[None, :], mask=(r < rows)[:, None] & (k < K)[None, :], other=0.0)
        g = tl.load(gate_ptr + g_row[:, None] + k[None, :], mask=(r < rows)[:, None] & (k < K)[None, :], other=0.0)
        b = tl.load(b_ptr + k[:, None] * b_k + j[None, :] * b_n, mask=(k < K)[:, None] & (j < N)[None, :], other=0.0)
        acc += tl.dot(a * g, b, input_precision="ieee")
    tl.store(c_ptr + (atom * c_n + comp * c_m)[:, None] + j[None, :], acc,
             mask=(r < rows)[:, None] & (j < N)[None, :])


def run_t04() -> dict:
    from fastmace.kernels.triton import tile_rule
    from fastmace.kernels.triton.segment_gemm import _gemm

    # MH-1 interaction block linear_2 segments: L=0 (dim=1, 128->128), L=1 (dim=3, 128->128),
    # L=2 (dim=5, 128->128), L=3 (dim=7, 128->128); total width = 16 * 128 = 2048
    dims = [1, 3, 5, 7]
    mul = 128
    width = sum(d * mul for d in dims)
    sms = tile_rule.device_sms(torch.device("cuda"))
    results = {}
    for name, atoms in [("NaCl_216", 216), ("NbCl5_1944", 1944)]:
        torch.manual_seed(0)
        raw_segs = [torch.randn((atoms, d, mul), device="cuda", dtype=torch.float32) for d in dims]
        gates = [torch.sigmoid(torch.randn((atoms, 1, mul), device="cuda", dtype=torch.float32)) for _ in dims[1:]]
        weights = [torch.randn((mul, mul), device="cuda", dtype=torch.float32) for _ in dims]
        offsets = [0, 128, 512, 1152]
        out_seq = torch.empty((atoms, width), device="cuda", dtype=torch.float32)
        out_fused = torch.empty((atoms, width), device="cuda", dtype=torch.float32)
        choices = [tile_rule.segment_gemm_choice(atoms * d, mul, mul, sms) for d in dims]

        def seq_step():
            # Baseline: elementwise gate multiply producing contiguous [atoms * d, mul] tensors, then segment_gemm
            r0 = torch.nn.functional.silu(raw_segs[0]).reshape(atoms, mul)
            rows = [r0] + [(seg * g).reshape(atoms * d, mul) for seg, g, d in zip(raw_segs[1:], gates, dims[1:])]
            for x, w, offset, dim, choice in zip(rows, weights, offsets, dims, choices):
                if choice.backend == "triton":
                    _gemm(x, 0, dim * mul, mul, w, out_seq, offset, width, mul, atoms * dim, dim, choice, "ieee")
                else:
                    out_seq[:, offset:offset + dim * mul] = (x @ w).reshape(atoms, dim * mul)

        def fused_step():
            # Fused: L=0 uses silu, L=1..3 fuse `seg * g` into the segment_gemm prologue without writing gated rows
            r0 = torch.nn.functional.silu(raw_segs[0]).reshape(atoms, mul)
            c0 = choices[0]
            if c0.backend == "triton":
                _gemm(r0, 0, mul, mul, weights[0], out_fused, 0, width, mul, atoms, 1, c0, "ieee")
            else:
                out_fused[:, :mul] = r0 @ weights[0]
            items = zip(raw_segs[1:], gates, weights[1:], offsets[1:], dims[1:], choices[1:])
            for seg, g, w, offset, dim, choice in items:
                rows = atoms * dim
                grid = (triton.cdiv(rows, choice.block_m), triton.cdiv(mul, choice.block_n))
                gated_segment_gemm_kernel[grid](
                    seg.reshape(-1), g.reshape(-1), w, out_fused.reshape(-1)[offset:],
                    rows, mul, mul, dim,
                    dim * mul, mul, mul, w.stride(0), w.stride(1), width, mul,
                    BLOCK_M=choice.block_m, BLOCK_N=choice.block_n, BLOCK_K=choice.block_k,
                    num_warps=choice.num_warps, num_stages=choice.num_stages,
                )

        seq_step()
        fused_step()
        max_diff = float((out_seq - out_fused).abs().max())
        t_seq, _ = graph_time_us(seq_step)
        t_fused, _ = graph_time_us(fused_step)
        results[name] = {
            "atoms": atoms,
            "choices": [c.backend for c in choices],
            "seq_us": t_seq,
            "fused_prologue_us": t_fused,
            "ratio_fused_over_seq": round(t_fused / t_seq, 4),
            "max_abs_diff": max_diff,
        }
    return results


def main() -> None:
    out_dir = Path("/mnt/d/Margraf/FastMACE/results/opt5-wave1-20261026")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "opt5_t02": run_t02(),
        "opt5_t03": run_t03(),
        "opt5_t04": run_t04(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
