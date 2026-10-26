#!/usr/bin/env python3
"""OPT5-T08: Fused single-kernel SIMT + 3xFP16 NVRTC GEMM (Path B) and row/col-normalized standalone 3xFP16 (Path A).

Design doc: docs/design/2026-10-26-fp16x3-and-hybrid-gemm.md
"""

from __future__ import annotations

import ctypes
import json
import os
import statistics
import time
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks)

SCALE_FLOOR = 2.0 ** -103

import numpy as np  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


def graph_time_us(fn, warmup: int = 10, reps: int = 50, blocks: int = 7) -> float:
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
    return round(statistics.median(times), 2)


def calc_errors(got: torch.Tensor, ref64: torch.Tensor, abs64: torch.Tensor) -> dict:
    diff = (got.double() - ref64).abs()
    scale = float(ref64.abs().max())
    frob = float(ref64.norm())
    normal = abs64 >= SCALE_FLOOR
    cancel = float((diff[normal] / abs64[normal]).max()) if bool(normal.any()) else 0.0
    return {
        "max_rel": float(diff.max()) / scale if scale else 0.0,
        "frob_rel": float(diff.norm()) / frob if frob else 0.0,
        "cancel_scaled": cancel,
    }


# ==============================================================================
# Path A: Standalone Triton 3xFP16 with per-row (A) and per-col (B) tile scaling
# ==============================================================================

@triton.jit
def triton_fp16x3_rowcol_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    sam, sak, sbk, sbn, scm, scn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = (pm * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = (pn * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    rk = tl.arange(0, BLOCK_K).to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + rk
        a = tl.load(a_ptr + rm[:, None] * sam + k[None, :] * sak,
                    mask=(rm < M)[:, None] & (k < K)[None, :], other=0.0)
        b = tl.load(b_ptr + k[:, None] * sbk + rn[None, :] * sbn,
                    mask=(k < K)[:, None] & (rn < N)[None, :], other=0.0)
        # Per-row (axis=1) exponent of A and per-column (axis=0) exponent of B prevent any sparse row/col underflow:
        a_max_bits = tl.max(tl.abs(a).to(tl.int32, bitcast=True), axis=1) & 0x7F800000
        b_max_bits = tl.max(tl.abs(b).to(tl.int32, bitcast=True), axis=0) & 0x7F800000
        a_exp = tl.maximum(tl.minimum(a_max_bits, 0x7B800000), 0x03800000)
        b_exp = tl.maximum(tl.minimum(b_max_bits, 0x7B800000), 0x03800000)
        a_scale = a_exp.to(tl.float32, bitcast=True)[:, None]
        b_scale = b_exp.to(tl.float32, bitcast=True)[None, :]
        a_inv = (0x7F000000 - a_exp).to(tl.float32, bitcast=True)[:, None]
        b_inv = (0x7F000000 - b_exp).to(tl.float32, bitcast=True)[None, :]

        a_norm = a * a_inv
        b_norm = b * b_inv
        ab = a_norm.to(tl.float16)
        as_ = ((a_norm - ab.to(tl.float32)) * 2048.0).to(tl.float16)
        bb = b_norm.to(tl.float16)
        bs = ((b_norm - bb.to(tl.float32)) * 2048.0).to(tl.float16)

        tile_dot = tl.dot(ab, bb, out_dtype=tl.float32) + (
            tl.dot(ab, bs, out_dtype=tl.float32) + tl.dot(as_, bb, out_dtype=tl.float32)
        ) * (1.0 / 2048.0)
        acc += tile_dot * (a_scale * b_scale)

    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc,
             mask=(rm < M)[:, None] & (rn < N)[None, :])


def launch_triton_fp16x3_rowcol(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, bm=64, bn=64, bk=32, nw=4, ns=3):
    m, k = a.shape
    n = b.shape[1]
    if k < 32 or n < 32:
        torch.mm(a.contiguous(), b.contiguous(), out=out)
        return
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    triton_fp16x3_rowcol_kernel[grid](
        a, b, out, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw, num_stages=ns
    )


# ==============================================================================
# Path B: Single-kernel Fused SIMT (FP32) + MMA (3xFP16) NVRTC GEMM on sm_120
# ==============================================================================
# Each 8-warp (256-thread) block computes BM=64 rows x BN=(64 + MMA_N) cols of C = A @ B:
# - Warps 0..3 (128 SIMT threads): compute a 64x64 FP32 sub-tile via FFMA (each thread computes an 8x4 tile = 32 regs).
# - Warps 4..7 (4 MMA warps): compute a 64xMMA_N (MMA_N=32 or 64) sub-tile via Exponent-Shifted 3xFP16 m16n8k16 MMA
#   (each MMA warp computes 16 rows x MMA_N cols).
# Both halves share the global-to-shared memory load of the 64x16 A tile per k-step!

HYBRID_NVRTC_SRC = r"""
__device__ __forceinline__ void mma_fp16(float* d, const unsigned* a, const unsigned* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void split_f16x2(float f0, float f1, unsigned& big, unsigned& small) {
  asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(big) : "f"(f0), "f"(f1));
  float b0, b1;
  asm("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %2; cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
      : "=f"(b0), "=f"(b1) : "r"(big));
  const float r0 = (f0 - b0) * 2048.0f, r1 = (f1 - b1) * 2048.0f;
  asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(small) : "f"(r0), "f"(r1));
}

// Double-buffered, register-prefetched, 1-barrier-per-step fused SIMT (FP32) + MMA (3xFP16) GEMM.
// - Warps 0..3 (128 SIMT threads): compute BM x SIMT_N in IEEE FP32 FFMA.
// - Warps 4..7 (4 MMA warps): compute BM x MMA_N in Exponent-Shifted 3xFP16 (split_f16x2 runs AFTER __syncthreads()
//   on the MMA warps only, overlapped with Warps 0..3's FFMAs!).
template <int BM, int SIMT_N, int MMA_N>
__device__ void hybrid_gemm_pipelined(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K) {
  constexpr int BN = SIMT_N + MMA_N;
  constexpr int BK = 16;
  constexpr int MT = BM / 64;     // m16 tiles per MMA warp (1 for BM=64, 2 for BM=128)
  constexpr int NT = MMA_N / 8;   // n8 tiles per MMA warp (4 for MMA_N=32, 8 for MMA_N=64)
  constexpr int SIMT_RM = BM / 8; // rows per SIMT thread (8 for BM=64, 16 for BM=128)

  // Double-buffered shared memory (2 stages, no pre-split arrays in smem -> only 1 barrier per step!):
  __shared__ __align__(16) float sA[2][BK][BM + 4];
  __shared__ __align__(16) float sB_simt[2][BK][SIMT_N + 4];
  __shared__ __align__(16) float sB_mma[2][MMA_N][BK + 4];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int row0 = blockIdx.x * BM;
  const int col0 = blockIdx.y * BN;

  const int simt_r = (tid & 127) >> 4;   // 0..7
  const int simt_c = (tid & 15) * 4;     // 0, 4, ..., 60
  float simt_acc[SIMT_RM][4];
  float mma_acc[MT][NT][4];

  if (warp < 4) {
#pragma unroll
    for (int i = 0; i < SIMT_RM; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) simt_acc[i][j] = 0.0f;
  } else {
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) mma_acc[m][n][i] = 0.0f;
  }

  const int a_r0 = tid >> 2;          // 0..63
  const int a_k = (tid & 3) * 4;      // 0, 4, 8, 12
  const int bs_k = tid >> 4;          // 0..15
  const int bs_c = (tid & 15) * 4;    // 0..60
  constexpr int BMMA_THREADS = (MMA_N * 16) / 4; // 128 for MMA_N=32, 256 for MMA_N=64
  constexpr int BMMA_ROW_THREADS = MMA_N / 4;    // 8 for MMA_N=32, 16 for MMA_N=64
  const int bm_k = tid / BMMA_ROW_THREADS;
  const int bm_c = (tid % BMMA_ROW_THREADS) * 4;

  auto load_global = [&](int k0, float4 (&pa)[MT], float4& pbs, float4& pbm) {
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const int gm = row0 + m * 64 + a_r0;
      if (gm < M && k0 + a_k + 3 < K) {
        pa[m] = *reinterpret_cast<const float4*>(&A[gm * K + k0 + a_k]);
      } else {
        float4 v = {0.f, 0.f, 0.f, 0.f};
        if (gm < M) {
          if (k0 + a_k + 0 < K) v.x = A[gm * K + k0 + a_k + 0];
          if (k0 + a_k + 1 < K) v.y = A[gm * K + k0 + a_k + 1];
          if (k0 + a_k + 2 < K) v.z = A[gm * K + k0 + a_k + 2];
          if (k0 + a_k + 3 < K) v.w = A[gm * K + k0 + a_k + 3];
        }
        pa[m] = v;
      }
    }
    {
      const int gk = k0 + bs_k;
      const int gn = col0 + bs_c;
      if (gk < K && gn + 3 < N) {
        pbs = *reinterpret_cast<const float4*>(&B[gk * N + gn]);
      } else {
        float4 v = {0.f, 0.f, 0.f, 0.f};
        if (gk < K) {
          if (gn + 0 < N) v.x = B[gk * N + gn + 0];
          if (gn + 1 < N) v.y = B[gk * N + gn + 1];
          if (gn + 2 < N) v.z = B[gk * N + gn + 2];
          if (gn + 3 < N) v.w = B[gk * N + gn + 3];
        }
        pbs = v;
      }
    }
    if (tid < BMMA_THREADS) {
      const int gk = k0 + bm_k;
      const int gn = col0 + SIMT_N + bm_c;
      if (gk < K && gn + 3 < N) {
        pbm = *reinterpret_cast<const float4*>(&B[gk * N + gn]);
      } else {
        float4 v = {0.f, 0.f, 0.f, 0.f};
        if (gk < K) {
          if (gn + 0 < N) v.x = B[gk * N + gn + 0];
          if (gn + 1 < N) v.y = B[gk * N + gn + 1];
          if (gn + 2 < N) v.z = B[gk * N + gn + 2];
          if (gn + 3 < N) v.w = B[gk * N + gn + 3];
        }
        pbm = v;
      }
    }
  };

  auto store_smem = [&](int stg, const float4 (&pa)[MT], const float4& pbs, const float4& pbm) {
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const int r = m * 64 + a_r0;
      sA[stg][a_k + 0][r] = pa[m].x;
      sA[stg][a_k + 1][r] = pa[m].y;
      sA[stg][a_k + 2][r] = pa[m].z;
      sA[stg][a_k + 3][r] = pa[m].w;
    }
    *reinterpret_cast<float4*>(&sB_simt[stg][bs_k][bs_c]) = pbs;
    if (tid < BMMA_THREADS) {
      sB_mma[stg][bm_c + 0][bm_k] = pbm.x;
      sB_mma[stg][bm_c + 1][bm_k] = pbm.y;
      sB_mma[stg][bm_c + 2][bm_k] = pbm.z;
      sB_mma[stg][bm_c + 3][bm_k] = pbm.w;
    }
  };

  float4 pre_a[MT], pre_bs, pre_bm = {0.f, 0.f, 0.f, 0.f};
  load_global(0, pre_a, pre_bs, pre_bm);
  store_smem(0, pre_a, pre_bs, pre_bm);
  __syncthreads();

  int stage = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    if (k0 + BK < K) {
      load_global(k0 + BK, pre_a, pre_bs, pre_bm);
    }

    if (warp < 4) {
#pragma unroll
      for (int k = 0; k < BK; ++k) {
        const float4 b_vec = *reinterpret_cast<const float4*>(&sB_simt[stage][k][simt_c]);
#pragma unroll
        for (int i4 = 0; i4 < SIMT_RM; i4 += 4) {
          const float4 a_vec = *reinterpret_cast<const float4*>(&sA[stage][k][simt_r * SIMT_RM + i4]);
          const float av[4] = {a_vec.x, a_vec.y, a_vec.z, a_vec.w};
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            simt_acc[i4 + r][0] = fmaf(av[r], b_vec.x, simt_acc[i4 + r][0]);
            simt_acc[i4 + r][1] = fmaf(av[r], b_vec.y, simt_acc[i4 + r][1]);
            simt_acc[i4 + r][2] = fmaf(av[r], b_vec.z, simt_acc[i4 + r][2]);
            simt_acc[i4 + r][3] = fmaf(av[r], b_vec.w, simt_acc[i4 + r][3]);
          }
        }
      }
    } else {
      const int group_id = lane >> 2;
      const int tid_in_group = lane & 3;
      const int k_lo = 2 * tid_in_group;
      const int k_hi = k_lo + 8;
      unsigned ab[MT][4], as_[MT][4];
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        const int r0 = m * 64 + (warp - 4) * 16 + group_id;
        const int r1 = r0 + 8;
        split_f16x2(sA[stage][k_lo + 0][r0], sA[stage][k_lo + 1][r0], ab[m][0], as_[m][0]);
        split_f16x2(sA[stage][k_lo + 0][r1], sA[stage][k_lo + 1][r1], ab[m][1], as_[m][1]);
        split_f16x2(sA[stage][k_hi + 0][r0], sA[stage][k_hi + 1][r0], ab[m][2], as_[m][2]);
        split_f16x2(sA[stage][k_hi + 0][r1], sA[stage][k_hi + 1][r1], ab[m][3], as_[m][3]);
      }
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        const int col_n = n * 8 + group_id;
        const float2 blo = *reinterpret_cast<const float2*>(&sB_mma[stage][col_n][k_lo]);
        const float2 bhi = *reinterpret_cast<const float2*>(&sB_mma[stage][col_n][k_hi]);
        unsigned bb[2], bs[2];
        split_f16x2(blo.x, blo.y, bb[0], bs[0]);
        split_f16x2(bhi.x, bhi.y, bb[1], bs[1]);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          float tmp[4] = {0.f, 0.f, 0.f, 0.f};
          mma_fp16(mma_acc[m][n], ab[m], bb);
          mma_fp16(tmp, ab[m], bs);
          mma_fp16(tmp, as_[m], bb);
          mma_acc[m][n][0] = fmaf(tmp[0], 1.0f / 2048.0f, mma_acc[m][n][0]);
          mma_acc[m][n][1] = fmaf(tmp[1], 1.0f / 2048.0f, mma_acc[m][n][1]);
          mma_acc[m][n][2] = fmaf(tmp[2], 1.0f / 2048.0f, mma_acc[m][n][2]);
          mma_acc[m][n][3] = fmaf(tmp[3], 1.0f / 2048.0f, mma_acc[m][n][3]);
        }
      }
    }

    if (k0 + BK < K) {
      store_smem(stage ^ 1, pre_a, pre_bs, pre_bm);
      stage ^= 1;
      __syncthreads();
    }
  }

  if (warp < 4) {
    const int gn = col0 + simt_c;
#pragma unroll
    for (int i = 0; i < SIMT_RM; ++i) {
      const int gm = row0 + simt_r * SIMT_RM + i;
      if (gm < M) {
        if (gn + 3 < N) {
          *reinterpret_cast<float4*>(&C[gm * N + gn]) =
              make_float4(simt_acc[i][0], simt_acc[i][1], simt_acc[i][2], simt_acc[i][3]);
        } else {
          if (gn + 0 < N) C[gm * N + gn + 0] = simt_acc[i][0];
          if (gn + 1 < N) C[gm * N + gn + 1] = simt_acc[i][1];
          if (gn + 2 < N) C[gm * N + gn + 2] = simt_acc[i][2];
        }
      }
    }
  } else {
    const int group_id = lane >> 2;
    const int c_col_off = (lane & 3) * 2;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const int gm0 = row0 + m * 64 + (warp - 4) * 16 + group_id;
      const int gm1 = gm0 + 8;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        const int gn = col0 + SIMT_N + n * 8 + c_col_off;
        const float v0 = mma_acc[m][n][0];
        const float v1 = mma_acc[m][n][1];
        const float v2 = mma_acc[m][n][2];
        const float v3 = mma_acc[m][n][3];
        if (gm0 < M) {
          if (gn + 0 < N) C[gm0 * N + gn + 0] = v0;
          if (gn + 1 < N) C[gm0 * N + gn + 1] = v1;
        }
        if (gm1 < M) {
          if (gn + 0 < N) C[gm1 * N + gn + 0] = v2;
          if (gn + 1 < N) C[gm1 * N + gn + 1] = v3;
        }
      }
    }
  }
}

// 6 SIMT warps (computing 64x96 in IEEE FP32, 32 accumulators/thread) + 2 MMA warps (computing 64x32 in 3xFP16,
// MT=2, NT=4 -> 64 accumulators/thread). Total tile is 64x128 (divides N=128, 256, 512, 1024 evenly with 0 wasted
// blocks!), 3:1 SIMT:MMA work ratio matching sm_120's 178:59 throughput ratio, <= 128 registers, 0 spill, 2 blocks/SM!
__device__ void hybrid_gemm_6x2_64_128_body(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K) {
  constexpr int BM = 64;
  constexpr int SIMT_N = 96;
  constexpr int MMA_N = 32;
  constexpr int BN = 128;
  constexpr int BK = 16;

  __shared__ __align__(16) float sA[2][BK][BM + 4];
  __shared__ __align__(16) float sB_simt[2][BK][SIMT_N + 4];
  __shared__ __align__(16) float sB_mma[2][MMA_N][BK + 4];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int row0 = blockIdx.x * BM;
  const int col0 = blockIdx.y * BN;

  const int simt_r = tid / 24;          // 0..7 (for tid < 192)
  const int simt_c = (tid % 24) * 4;    // 0, 4, ..., 92
  float simt_acc[8][4];
  float mma_acc[2][4][4];

  if (warp < 6) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) simt_acc[i][j] = 0.0f;
  } else {
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int n = 0; n < 4; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) mma_acc[m][n][i] = 0.0f;
  }

  const int a_r = tid >> 2;             // 0..63
  const int a_k = (tid & 3) * 4;        // 0, 4, 8, 12
  const int b_k0 = tid >> 5;            // 0..7 (and b_k0 + 8 -> 8..15)
  const int b_c = (tid & 31) * 4;       // 0, 4, ..., 124 (100% coalesced across all 32 lanes of each warp!)

  auto load_global = [&](int k0, float4& pa, float4& pb0, float4& pb1) {
    const int gm = row0 + a_r;
    if (gm < M) {
      pa = *reinterpret_cast<const float4*>(&A[gm * K + k0 + a_k]);
    } else {
      pa = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    const int gn = col0 + b_c;
    pb0 = *reinterpret_cast<const float4*>(&B[(k0 + b_k0) * N + gn]);
    pb1 = *reinterpret_cast<const float4*>(&B[(k0 + b_k0 + 8) * N + gn]);
  };

  auto store_smem = [&](int stg, const float4& pa, const float4& pb0, const float4& pb1) {
    sA[stg][a_k + 0][a_r] = pa.x;
    sA[stg][a_k + 1][a_r] = pa.y;
    sA[stg][a_k + 2][a_r] = pa.z;
    sA[stg][a_k + 3][a_r] = pa.w;
    if (b_c < SIMT_N) {
      *reinterpret_cast<float4*>(&sB_simt[stg][b_k0][b_c]) = pb0;
      *reinterpret_cast<float4*>(&sB_simt[stg][b_k0 + 8][b_c]) = pb1;
    } else {
      const int mc = b_c - SIMT_N;
      sB_mma[stg][mc + 0][b_k0] = pb0.x;
      sB_mma[stg][mc + 1][b_k0] = pb0.y;
      sB_mma[stg][mc + 2][b_k0] = pb0.z;
      sB_mma[stg][mc + 3][b_k0] = pb0.w;
      sB_mma[stg][mc + 0][b_k0 + 8] = pb1.x;
      sB_mma[stg][mc + 1][b_k0 + 8] = pb1.y;
      sB_mma[stg][mc + 2][b_k0 + 8] = pb1.z;
      sB_mma[stg][mc + 3][b_k0 + 8] = pb1.w;
    }
  };

  float4 pre_a, pre_b0, pre_b1;
  load_global(0, pre_a, pre_b0, pre_b1);
  store_smem(0, pre_a, pre_b0, pre_b1);
  __syncthreads();

  int stage = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    if (k0 + BK < K) {
      load_global(k0 + BK, pre_a, pre_b0, pre_b1);
    }

    if (warp < 6) {
#pragma unroll
      for (int k = 0; k < BK; ++k) {
        const float4 b_vec = *reinterpret_cast<const float4*>(&sB_simt[stage][k][simt_c]);
        const float4 a0 = *reinterpret_cast<const float4*>(&sA[stage][k][simt_r * 8 + 0]);
        const float4 a1 = *reinterpret_cast<const float4*>(&sA[stage][k][simt_r * 8 + 4]);
        const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          simt_acc[i][0] = fmaf(av[i], b_vec.x, simt_acc[i][0]);
          simt_acc[i][1] = fmaf(av[i], b_vec.y, simt_acc[i][1]);
          simt_acc[i][2] = fmaf(av[i], b_vec.z, simt_acc[i][2]);
          simt_acc[i][3] = fmaf(av[i], b_vec.w, simt_acc[i][3]);
        }
      }
    } else {
      const int warp_row_base = (warp - 6) * 32;
      const int group_id = lane >> 2;
      const int tid_in_group = lane & 3;
      const int k_lo = 2 * tid_in_group;
      const int k_hi = k_lo + 8;
      unsigned ab[2][4], as_[2][4];
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const int r0 = warp_row_base + m * 16 + group_id;
        const int r1 = r0 + 8;
        split_f16x2(sA[stage][k_lo + 0][r0], sA[stage][k_lo + 1][r0], ab[m][0], as_[m][0]);
        split_f16x2(sA[stage][k_lo + 0][r1], sA[stage][k_lo + 1][r1], ab[m][1], as_[m][1]);
        split_f16x2(sA[stage][k_hi + 0][r0], sA[stage][k_hi + 1][r0], ab[m][2], as_[m][2]);
        split_f16x2(sA[stage][k_hi + 0][r1], sA[stage][k_hi + 1][r1], ab[m][3], as_[m][3]);
      }
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        const int col_n = n * 8 + group_id;
        const float2 blo = *reinterpret_cast<const float2*>(&sB_mma[stage][col_n][k_lo]);
        const float2 bhi = *reinterpret_cast<const float2*>(&sB_mma[stage][col_n][k_hi]);
        unsigned bb[2], bs[2];
        split_f16x2(blo.x, blo.y, bb[0], bs[0]);
        split_f16x2(bhi.x, bhi.y, bb[1], bs[1]);
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          float tmp[4] = {0.f, 0.f, 0.f, 0.f};
          mma_fp16(mma_acc[m][n], ab[m], bb);
          mma_fp16(tmp, ab[m], bs);
          mma_fp16(tmp, as_[m], bb);
          mma_acc[m][n][0] = fmaf(tmp[0], 1.0f / 2048.0f, mma_acc[m][n][0]);
          mma_acc[m][n][1] = fmaf(tmp[1], 1.0f / 2048.0f, mma_acc[m][n][1]);
          mma_acc[m][n][2] = fmaf(tmp[2], 1.0f / 2048.0f, mma_acc[m][n][2]);
          mma_acc[m][n][3] = fmaf(tmp[3], 1.0f / 2048.0f, mma_acc[m][n][3]);
        }
      }
    }

    if (k0 + BK < K) {
      store_smem(stage ^ 1, pre_a, pre_b0, pre_b1);
      stage ^= 1;
      __syncthreads();
    }
  }

  if (warp < 6) {
    const int gn = col0 + simt_c;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int gm = row0 + simt_r * 8 + i;
      if (gm < M) {
        *reinterpret_cast<float4*>(&C[gm * N + gn]) =
            make_float4(simt_acc[i][0], simt_acc[i][1], simt_acc[i][2], simt_acc[i][3]);
      }
    }
  } else {
    const int warp_row_base = (warp - 6) * 32;
    const int group_id = lane >> 2;
    const int c_col_off = (lane & 3) * 2;
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      const int gm0 = row0 + warp_row_base + m * 16 + group_id;
      const int gm1 = gm0 + 8;
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        const int gn = col0 + SIMT_N + n * 8 + c_col_off;
        const float v0 = mma_acc[m][n][0];
        const float v1 = mma_acc[m][n][1];
        const float v2 = mma_acc[m][n][2];
        const float v3 = mma_acc[m][n][3];
        if (gm0 < M) {
          *reinterpret_cast<float2*>(&C[gm0 * N + gn]) = make_float2(v0, v1);
        }
        if (gm1 < M) {
          *reinterpret_cast<float2*>(&C[gm1 * N + gn]) = make_float2(v2, v3);
        }
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(256, 2) hybrid_gemm_64_32(
    const float* A, const float* B, float* C, int M, int N, int K) {
  hybrid_gemm_pipelined<64, 64, 32>(A, B, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(256, 2) hybrid_gemm_64_64(
    const float* A, const float* B, float* C, int M, int N, int K) {
  hybrid_gemm_pipelined<64, 64, 64>(A, B, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(256, 2) hybrid_gemm_6x2_64_128(
    const float* A, const float* B, float* C, int M, int N, int K) {
  hybrid_gemm_6x2_64_128_body(A, B, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(256) hybrid_gemm_128_32(
    const float* A, const float* B, float* C, int M, int N, int K) {
  hybrid_gemm_pipelined<128, 64, 32>(A, B, C, M, N, K);
}
"""


def compile_hybrid_nvrtc():
    from cuda.bindings import driver, nvrtc

    cap = torch.cuda.get_device_capability()
    _, prog = nvrtc.nvrtcCreateProgram(HYBRID_NVRTC_SRC.encode(), b"hybrid_gemm.cu", 0, [], [])
    opts = [f"-arch=sm_{cap[0]}{cap[1]}".encode(), b"-std=c++17"]
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    _, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
    log = b" " * log_size
    nvrtc.nvrtcGetProgramLog(prog, log)
    if int(err) != 0:
        raise RuntimeError(f"NVRTC failed:\n{log.decode(errors='replace')}")
    _, size = nvrtc.nvrtcGetCUBINSize(prog)
    cubin = b" " * size
    nvrtc.nvrtcGetCUBIN(prog, cubin)
    _, module = driver.cuModuleLoadData(cubin)
    kernels = {}
    attr = driver.CUfunction_attribute
    specs = [
        ("hybrid_gemm_64_32", 64, 96),
        ("hybrid_gemm_64_64", 64, 128),
        ("hybrid_gemm_6x2_64_128", 64, 128),
        ("hybrid_gemm_128_32", 128, 96),
    ]
    for name, bm, bn in specs:
        _, fn = driver.cuModuleGetFunction(module, name.encode())
        _, regs = driver.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_NUM_REGS, fn)
        _, smem = driver.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn)
        _, spill = driver.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn)
        kernels[name] = {
            "fn": fn, "bm": bm, "bn": bn, "regs": int(regs), "smem_bytes": int(smem), "local_bytes": int(spill)
        }
    return module, kernels


def launch_hybrid_nvrtc(kernel_info, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    from cuda.bindings import driver

    m, k = a.shape
    n = b.shape[1]
    bm = kernel_info["bm"]
    bn = kernel_info["bn"]
    grid_x = (m + bm - 1) // bm
    grid_y = (n + bn - 1) // bn
    args = [
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(c.data_ptr()),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.c_int(k),
    ]
    argv = (ctypes.c_void_p * len(args))(*[ctypes.addressof(x) for x in args])
    stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
    driver.cuLaunchKernel(kernel_info["fn"], grid_x, grid_y, 1, 256, 1, 1, 0, stream, ctypes.addressof(argv), 0)


def run_all() -> dict:
    from ase.io import read

    from fastmace.config import RewriteConfig
    from fastmace.kernels.triton import segment_gemm, tf32x3_gemm

    torch.zeros(1, device="cuda")
    _, nvrtc_kernels = compile_hybrid_nvrtc()
    k64_32 = nvrtc_kernels["hybrid_gemm_64_32"]
    k64_64 = nvrtc_kernels["hybrid_gemm_64_64"]
    k6x2_64_128 = nvrtc_kernels["hybrid_gemm_6x2_64_128"]
    k128_32 = nvrtc_kernels["hybrid_gemm_128_32"]

    shapes = [
        ("NaCl_linear_1512x512x512", 1512, 512, 512),
        ("NbCl5_linear_5832x512x512", 5832, 512, 512),
        ("NbCl5_linear_13608x512x512", 13608, 512, 512),
        ("OMOL_linear_1944x1024x1024", 1944, 1024, 1024),
    ]
    bench_rows = {}
    for label, M, K, N in shapes:
        torch.manual_seed(0)
        a = torch.randn((M, K), device="cuda", dtype=torch.float32)
        b = torch.randn((K, N), device="cuda", dtype=torch.float32)
        c = torch.empty((M, N), device="cuda", dtype=torch.float32)
        ref64 = a.double() @ b.double()
        abs64 = a.double().abs() @ b.double().abs()

        t_cublas = graph_time_us(lambda: torch.mm(a, b, out=c))
        err_cublas = calc_errors(a @ b, ref64, abs64)

        launch_triton_fp16x3_rowcol(a, b, c)
        err_rowcol = calc_errors(c, ref64, abs64)
        t_rowcol = graph_time_us(lambda: launch_triton_fp16x3_rowcol(a, b, c))

        launch_hybrid_nvrtc(k64_32, a, b, c)
        torch.cuda.synchronize()
        err_h64_32 = calc_errors(c, ref64, abs64)
        t_h64_32 = graph_time_us(lambda: launch_hybrid_nvrtc(k64_32, a, b, c))
        launch_hybrid_nvrtc(k64_64, a, b, c)
        torch.cuda.synchronize()
        err_h64_64 = calc_errors(c, ref64, abs64)
        t_h64_64 = graph_time_us(lambda: launch_hybrid_nvrtc(k64_64, a, b, c))

        launch_hybrid_nvrtc(k6x2_64_128, a, b, c)
        torch.cuda.synchronize()
        err_h6x2 = calc_errors(c, ref64, abs64)
        t_h6x2 = graph_time_us(lambda: launch_hybrid_nvrtc(k6x2_64_128, a, b, c))

        launch_hybrid_nvrtc(k128_32, a, b, c)
        torch.cuda.synchronize()
        err_h128_32 = calc_errors(c, ref64, abs64)
        t_h128_32 = graph_time_us(lambda: launch_hybrid_nvrtc(k128_32, a, b, c))

        bench_rows[label] = {
            "shape": [M, K, N],
            "cublas_ieee_us": t_cublas,
            "triton_fp16x3_rowcol_us": t_rowcol,
            "nvrtc_hybrid_64_32_us": t_h64_32,
            "nvrtc_hybrid_64_64_us": t_h64_64,
            "nvrtc_hybrid_6x2_64_128_us": t_h6x2,
            "nvrtc_hybrid_128_32_us": t_h128_32,
            "errors": {
                "cublas_ieee": err_cublas,
                "triton_fp16x3_rowcol": err_rowcol,
                "nvrtc_hybrid_64_32": err_h64_32,
                "nvrtc_hybrid_64_64": err_h64_64,
                "nvrtc_hybrid_6x2_64_128": err_h6x2,
                "nvrtc_hybrid_128_32": err_h128_32,
            },
        }

    # Test Path A (row/col normalized Triton 3xFP16) across all 52 real MH-1 NaCl force-call products!
    structures_root = Path(os.environ["FASTMACE_STRUCTURES"]) / "Structures/Benchmark/R2SCAN"
    atoms = read(structures_root / "NaCl.cif").repeat((3, 3, 3))
    atoms.set_pbc(True)
    rng = np.random.default_rng(20261024)
    atoms.positions += rng.uniform(-0.05, 0.05, size=atoms.positions.shape)

    loaded = fastmace.load("mh1-r2scan")
    ff = fastmace.ForceField(loaded, atoms, precision="float32", rewrites=RewriteConfig(tf32x3_gemms="on"))
    ff.evaluate(atoms)
    torch.cuda.synchronize()

    real_ops = []
    orig_launch = tf32x3_gemm._launch
    orig_joined = segment_gemm._gemm

    def hook_launch(a, b, out):
        orig_launch(a, b, out)
        real_ops.append(("dense", a.detach().clone(), b.detach().clone()))

    def hook_joined(a, a_offset, a_n, a_m, b, c, c_offset, c_n, c_m, rows, dim, choice, precision="ieee"):
        orig_joined(a, a_offset, a_n, a_m, b, c, c_offset, c_n, c_m, rows, dim, choice, precision)
        if precision == "tf32x3":
            k, n = b.shape
            nat = rows // dim
            left = a.reshape(-1)[a_offset:].as_strided((nat, dim, k), (a_n, a_m, 1)).reshape(rows, k)
            real_ops.append(("joined", left.detach().clone(), b.detach().clone()))

    tf32x3_gemm._launch = hook_launch
    segment_gemm._gemm = hook_joined
    try:
        ff.evaluate(atoms)
        torch.cuda.synchronize()
    finally:
        tf32x3_gemm._launch = orig_launch
        segment_gemm._gemm = orig_joined

    ratios_cancel = []
    ratios_max_rel = []
    worse_count = 0
    max_cancel_fp16x3 = 0.0
    max_cancel_ieee = 0.0
    for _, a, b in real_ops:
        ref = a.double() @ b.double()
        abs64 = a.double().abs() @ b.double().abs()
        c_ieee = a @ b
        c_fp16x3 = torch.empty_like(c_ieee)
        launch_triton_fp16x3_rowcol(a, b, c_fp16x3)
        e_ieee = calc_errors(c_ieee, ref, abs64)
        e_fp16 = calc_errors(c_fp16x3, ref, abs64)
        max_cancel_ieee = max(max_cancel_ieee, e_ieee["cancel_scaled"])
        max_cancel_fp16x3 = max(max_cancel_fp16x3, e_fp16["cancel_scaled"])
        if e_ieee["cancel_scaled"] > 0:
            r = e_fp16["cancel_scaled"] / e_ieee["cancel_scaled"]
            ratios_cancel.append(r)
            if r > 1.0:
                worse_count += 1
        if e_ieee["max_rel"] > 0:
            ratios_max_rel.append(e_fp16["max_rel"] / e_ieee["max_rel"])

    return {
        "nvrtc_kernels": {k: {x: y for x, y in v.items() if x != "fn"} for k, v in nvrtc_kernels.items()},
        "shape_benchmarks": bench_rows,
        "real_52_mh1_products_rowcol_fp16x3": {
            "products": len(real_ops),
            "fp16x3_max_cancel_scaled": max_cancel_fp16x3,
            "ieee_max_cancel_scaled": max_cancel_ieee,
            "fp16x3_over_ieee_cancel_median": statistics.median(ratios_cancel),
            "fp16x3_over_ieee_cancel_max": max(ratios_cancel),
            "fp16x3_over_ieee_max_rel_median": statistics.median(ratios_max_rel),
            "fp16x3_over_ieee_max_rel_max": max(ratios_max_rel),
            "fp16x3_worse_count": worse_count,
        },
    }


def main() -> None:
    out_dir = Path("/mnt/d/Margraf/FastMACE/results/opt5-T08-hybrid-gemm-20261026")
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_all()
    (out_dir / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
