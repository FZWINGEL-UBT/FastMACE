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

template <int MMA_N>
__device__ void hybrid_gemm_body(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K) {
  constexpr int BM = 64;
  constexpr int SIMT_N = 64;
  constexpr int BN = SIMT_N + MMA_N;
  constexpr int BK = 16;
  constexpr int NT = MMA_N / 8;

  // Shared memory for one 64x16 A tile (FP32 + split FP16x2) and one 16xBN B tile:
  __shared__ __align__(16) float sA[BK][BM + 4];           // [16][68] transposed for conflict-free SIMT loads
  __shared__ __align__(16) unsigned sA_big[BM][BK / 2];    // [64][8] f16x2
  __shared__ __align__(16) unsigned sA_small[BM][BK / 2];  // [64][8] f16x2
  __shared__ __align__(16) float sB_simt[BK][SIMT_N + 4];  // [16][68] for SIMT warps
  __shared__ __align__(16) unsigned sB_big[MMA_N][BK / 2]; // [MMA_N][8] f16x2 (col-major for B)
  __shared__ __align__(16) unsigned sB_small[MMA_N][BK / 2];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int row0 = blockIdx.x * BM;
  const int col0 = blockIdx.y * BN;

  // SIMT thread tile mapping (128 threads = 8x16 grid of 8x4 thread tiles covering 64x64):
  const int simt_r = (tid & 127) >> 4;   // 0..7 -> rows simt_r * 8 .. simt_r * 8 + 7
  const int simt_c = (tid & 15) * 4;     // 0, 4, ..., 60
  float simt_acc[8][4];
  float mma_main[NT][4], mma_cross[NT][4];

  if (warp < 4) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) simt_acc[i][j] = 0.0f;
  } else {
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) { mma_main[n][i] = 0.0f; mma_cross[n][i] = 0.0f; }
  }

  for (int k0 = 0; k0 < K; k0 += BK) {
    // 1. Cooperative load of A [64 x 16 = 1024 floats]: each of the 256 threads loads 4 contiguous K elements (2 f16x2)
    {
      const int a_r = tid >> 2;          // 0..63
      const int a_k = (tid & 3) * 4;     // 0, 4, 8, 12
      const int gm = row0 + a_r;
      float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f;
      if (gm < M && k0 + a_k + 3 < K) {
        const float4 v = *reinterpret_cast<const float4*>(&A[gm * K + k0 + a_k]);
        v0 = v.x; v1 = v.y; v2 = v.z; v3 = v.w;
      } else if (gm < M) {
        if (k0 + a_k + 0 < K) v0 = A[gm * K + k0 + a_k + 0];
        if (k0 + a_k + 1 < K) v1 = A[gm * K + k0 + a_k + 1];
        if (k0 + a_k + 2 < K) v2 = A[gm * K + k0 + a_k + 2];
        if (k0 + a_k + 3 < K) v3 = A[gm * K + k0 + a_k + 3];
      }
      sA[a_k + 0][a_r] = v0;
      sA[a_k + 1][a_r] = v1;
      sA[a_k + 2][a_r] = v2;
      sA[a_k + 3][a_r] = v3;
      unsigned b01, s01, b23, s23;
      split_f16x2(v0, v1, b01, s01);
      split_f16x2(v2, v3, b23, s23);
      sA_big[a_r][(tid & 3) * 2 + 0] = b01;
      sA_small[a_r][(tid & 3) * 2 + 0] = s01;
      sA_big[a_r][(tid & 3) * 2 + 1] = b23;
      sA_small[a_r][(tid & 3) * 2 + 1] = s23;
    }

    // 2. Cooperative load of B_simt [16 x 64 = 1024 floats]: each thread loads 4 contiguous N elements
    {
      const int b_k = tid >> 4;          // 0..15
      const int b_c = (tid & 15) * 4;    // 0..60
      const int gk = k0 + b_k;
      const int gn = col0 + b_c;
      float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f;
      if (gk < K && gn + 3 < N) {
        const float4 v = *reinterpret_cast<const float4*>(&B[gk * N + gn]);
        v0 = v.x; v1 = v.y; v2 = v.z; v3 = v.w;
      } else if (gk < K) {
        if (gn + 0 < N) v0 = B[gk * N + gn + 0];
        if (gn + 1 < N) v1 = B[gk * N + gn + 1];
        if (gn + 2 < N) v2 = B[gk * N + gn + 2];
        if (gn + 3 < N) v3 = B[gk * N + gn + 3];
      }
      sB_simt[b_k][b_c + 0] = v0;
      sB_simt[b_k][b_c + 1] = v1;
      sB_simt[b_k][b_c + 2] = v2;
      sB_simt[b_k][b_c + 3] = v3;
    }

    // 3. Cooperative load & split of B_mma [16 x MMA_N]: threads 0 .. (MMA_N * 8 - 1) load 2 K elements per N col
    if (tid < MMA_N * 8) {
      const int mma_c = tid >> 3;        // 0 .. MMA_N-1
      const int mma_k2 = (tid & 7) * 2;  // 0, 2, ..., 14
      const int gn = col0 + SIMT_N + mma_c;
      float v0 = 0.f, v1 = 0.f;
      if (gn < N) {
        if (k0 + mma_k2 + 0 < K) v0 = B[(k0 + mma_k2 + 0) * N + gn];
        if (k0 + mma_k2 + 1 < K) v1 = B[(k0 + mma_k2 + 1) * N + gn];
      }
      unsigned bg, sm;
      split_f16x2(v0, v1, bg, sm);
      sB_big[mma_c][tid & 7] = bg;
      sB_small[mma_c][tid & 7] = sm;
    }
    __syncthreads();

    // 4. Simultaneous execution: Warps 0..3 on SIMT FFMA, Warps 4..7 on FP16 Tensor Cores!
    if (warp < 4) {
#pragma unroll
      for (int k = 0; k < BK; ++k) {
        float a_reg[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) a_reg[i] = sA[k][simt_r * 8 + i];
        const float4 b_vec = *reinterpret_cast<const float4*>(&sB_simt[k][simt_c]);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          simt_acc[i][0] = fmaf(a_reg[i], b_vec.x, simt_acc[i][0]);
          simt_acc[i][1] = fmaf(a_reg[i], b_vec.y, simt_acc[i][1]);
          simt_acc[i][2] = fmaf(a_reg[i], b_vec.z, simt_acc[i][2]);
          simt_acc[i][3] = fmaf(a_reg[i], b_vec.w, simt_acc[i][3]);
        }
      }
    } else {
      // m16n8k16 fragment layout for A (row-major [16, 16]) and B (col-major [16, 8]):
      const int warp_m = (warp - 4) * 16;
      const int group_id = lane >> 2;       // 0..7
      const int tid_in_group = lane & 3;    // 0..3
      unsigned ab[4], as_[4];
      ab[0] = sA_big[warp_m + group_id][tid_in_group];
      ab[1] = sA_big[warp_m + group_id + 8][tid_in_group];
      ab[2] = sA_big[warp_m + group_id][tid_in_group + 4];
      ab[3] = sA_big[warp_m + group_id + 8][tid_in_group + 4];

      as_[0] = sA_small[warp_m + group_id][tid_in_group];
      as_[1] = sA_small[warp_m + group_id + 8][tid_in_group];
      as_[2] = sA_small[warp_m + group_id][tid_in_group + 4];
      as_[3] = sA_small[warp_m + group_id + 8][tid_in_group + 4];

#pragma unroll
      for (int n = 0; n < NT; ++n) {
        const int col_n = n * 8 + group_id;
        unsigned bb[2], bs[2];
        bb[0] = sB_big[col_n][tid_in_group];
        bb[1] = sB_big[col_n][tid_in_group + 4];
        bs[0] = sB_small[col_n][tid_in_group];
        bs[1] = sB_small[col_n][tid_in_group + 4];

        mma_fp16(mma_main[n], ab, bb);
        mma_fp16(mma_cross[n], ab, bs);
        mma_fp16(mma_cross[n], as_, bb);
      }
    }
    __syncthreads();
  }

  // 5. Store output tiles to C:
  if (warp < 4) {
    const int gn = col0 + simt_c;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int gm = row0 + simt_r * 8 + i;
      if (gm < M) {
        if (gn + 0 < N) C[gm * N + gn + 0] = simt_acc[i][0];
        if (gn + 1 < N) C[gm * N + gn + 1] = simt_acc[i][1];
        if (gn + 2 < N) C[gm * N + gn + 2] = simt_acc[i][2];
        if (gn + 3 < N) C[gm * N + gn + 3] = simt_acc[i][3];
      }
    }
  } else {
    const int warp_m = (warp - 4) * 16;
    const int group_id = lane >> 2;
    const int c_col_off = (lane & 3) * 2;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const int gn = col0 + SIMT_N + n * 8 + c_col_off;
      const int gm0 = row0 + warp_m + group_id;
      const int gm1 = gm0 + 8;
      const float v0 = mma_main[n][0] + mma_cross[n][0] * (1.0f / 2048.0f);
      const float v1 = mma_main[n][1] + mma_cross[n][1] * (1.0f / 2048.0f);
      const float v2 = mma_main[n][2] + mma_cross[n][2] * (1.0f / 2048.0f);
      const float v3 = mma_main[n][3] + mma_cross[n][3] * (1.0f / 2048.0f);
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

extern "C" __global__ void __launch_bounds__(256, 2) hybrid_gemm_64_32(
    const float* A, const float* B, float* C, int M, int N, int K) {
  hybrid_gemm_body<32>(A, B, C, M, N, K);
}

extern "C" __global__ void __launch_bounds__(256, 2) hybrid_gemm_64_64(
    const float* A, const float* B, float* C, int M, int N, int K) {
  hybrid_gemm_body<64>(A, B, C, M, N, K);
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
    for name, bn in [("hybrid_gemm_64_32", 96), ("hybrid_gemm_64_64", 128)]:
        _, fn = driver.cuModuleGetFunction(module, name.encode())
        _, regs = driver.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_NUM_REGS, fn)
        _, smem = driver.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn)
        _, spill = driver.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn)
        kernels[name] = {"fn": fn, "bn": bn, "regs": int(regs), "smem_bytes": int(smem), "local_bytes": int(spill)}
    return module, kernels


def launch_hybrid_nvrtc(kernel_info, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    from cuda.bindings import driver

    m, k = a.shape
    n = b.shape[1]
    bn = kernel_info["bn"]
    grid_x = (m + 63) // 64
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

        bench_rows[label] = {
            "shape": [M, K, N],
            "cublas_ieee_us": t_cublas,
            "triton_fp16x3_rowcol_us": t_rowcol,
            "nvrtc_hybrid_64_32_us": t_h64_32,
            "nvrtc_hybrid_64_64_us": t_h64_64,
            "errors": {
                "cublas_ieee": err_cublas,
                "triton_fp16x3_rowcol": err_rowcol,
                "nvrtc_hybrid_64_32": err_h64_32,
                "nvrtc_hybrid_64_64": err_h64_64,
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
