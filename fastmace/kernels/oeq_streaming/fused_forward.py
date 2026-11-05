"""The folded radial last layer fused into the receiver-streaming forward (``fused_folded_forward``, opt-8 OPT8-T02).

Today's chain builds the per-edge weights ``w = a @ W.T`` with cuBLAS (``W [Q, K]``, ``Q = paths × channels``) and
the streaming forward reads them back. The fused forward builds each batch's ``w`` rows on chip from ``a`` and a
shared-memory slice of ``W``, uses them for the message at once and also stores them, so the chunked backward
(``chunked_conv_backward``) runs unchanged on them. It saves the cuBLAS product and the forward's ``w`` read.
The stored ``w`` and the message are bitwise those of the chain on the measured GPUs: each ``w`` element is the
same k-ordered FP32 sum of fused multiply-adds, and each receiver's message sums its edges in CSR order with the
same angular terms. Results: ``docs/validation/opt8-T01.md`` (T01b) and ``docs/validation/opt8-T02.md``.

Generated, like the vendored kernels, from the streaming schedule (``FactorizedComputationSchedule``) of any uvu
problem whose paths have one edge channel and whose channel count is a multiple of 32; float32 only. Work split:

- a block owns one slice: 32 channels (``blockIdx.y``) × one part of the paths (``blockIdx.z``), and keeps that
  slice of ``W`` in shared memory (row stride 33 floats, conflict-free);
- one warp (lane = channel) takes chunks of ``RECEIVERS_PER_CHUNK`` receivers from a per-slice counter and walks
  their edges in batches of ``BATCH`` across receiver boundaries; the batch's ``a`` rows are staged transposed in
  warp-private shared memory (four edges per float4 broadcast);
- per batch: ``w[part paths][BATCH]`` by a k-loop that reads each ``W`` value once for the batch, stored to ``w``;
  then per edge the angular terms into the receiver's outputs (registers), written when the receiver changes;
- edges past the last CSR row (the neighbour list's padding) get their ``w`` rows the same way, without messages,
  so the backward sees what the chain would have built.

Launch rule (``launch_rule``), from the device and the problem only:
- **paths per part** at most ``W_TILE_REGISTERS / BATCH`` (5): the part's ``w`` tile is at most 40 registers
  per lane;
- **shared memory:** the fewest parts whose ``W`` slice plus ``MIN_WARPS`` warps' staging fit the device's
  opt-in shared memory per block;
- **warps per block:** as many as fit beside the slice, at most ``MAX_WARPS`` (12, the warps for which 168
  registers per thread fit the 64K-register file of every sm_70+ GPU);
- **grid:** ``ceil(2 × SMs / slices)`` blocks per slice.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import torch

from fastmace.kernels.gpu_models import GPUModels

BATCH = 8  # edges per batch: two float4 broadcasts of a per k
RECEIVERS_PER_CHUNK = 2  # receivers per work chunk (OPT8-T01 screen: 2 balanced best on NbCl5 and NaNbO3)
TAIL_EDGES = 64  # padding edges (past the last CSR row) per work chunk
W_TILE_REGISTERS = 40  # paths per part × BATCH
MAX_WARPS = 12
MIN_WARPS = 4
CHANNEL_BLOCK = 32
BLOCKS_PER_SM = 2  # blocks per SM over all slices of the grid (the persistent grid's size)
# A folded weight whose last column is all zero (a last radial layer without bias, such as e3nn's FullyConnectedNet
# in MACE-OMAT-0): the k-loop skips that column, and a and W keep their row stride (OMAT-0 kernel lab, 3 November
# 2026: the fused forward 9.1 % faster on the RTX 4000 Ada, 1.0 % slower on the RTX 5060 Ti). The same FP32 sums:
# the skipped term adds an exact zero. None: on for SKIP_ZERO_COLUMN_GPUS (decision 66; owner, 4 November 2026);
# True / False: an A/B side's choice on every GPU.
SKIP_ZERO_COLUMN = None
# MD step on Benchmark_loup (OMAT-0 medium, SrTiO3 5,000 atoms) on top of degree-5 radial tables: RTX 4000 Ada 0.981,
# A100-SXM4-80GB 0.986 (degree 5 and the skip against degree 5 alone; both orders 0.896 / 0.895 against degree 10).
# The RTX 5060 Ti gained nothing (1.000) and keeps the full k-loop; other GPUs are not measured. The A100 80GB PCIe
# (owner, 4 November 2026: "Yes"): Benchmark_loup 4.61 -> 3.31 us/atom/step with the SXM4's decision-66 switches.
# Every A100 model (GPUModels; owner, 4 November 2026).
SKIP_ZERO_COLUMN_GPUS = GPUModels(("NVIDIA RTX 4000 Ada Generation", "NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"))
# The stored per-edge weights are read only by the backward, long after: store them with the streaming cache hint
# (``__stcs``, evict-first), so they do not displace W, a and x in L2 (OMAT-0 kernel lab, 3 November 2026: the fused
# forward 3.5 % faster on the RTX 4000 Ada, 0.5 % on the RTX 5060 Ti). Bitwise. A module constant for the A/B.
STREAM_STORE_W = False
# The batch's a.W product on tensor cores in 3xTF32 (owner, 4 November 2026: "try out all of the ideas"; OMAT-0 on
# the A100, where the FP32-SIMT product is about 62 GFLOP per call): per warp mma.sync m16n8k8 tiles of 16 channels x
# the BATCH edges, W and a split into TF32 high and low parts on the fly (hi*hi + hi*lo + lo*hi, FP32 accumulation:
# about FP32 accuracy, not bitwise), the result redistributed to lane = channel through a per-warp shared tile.
# Compute capability 8.0 and later only (ignored below). None: on for TENSOR_CORES_GPUS (decision 66; owner,
# 4 November 2026: "Yes", with TC_SPLIT "mask"); True / False: an A/B side's choice on every GPU.
TENSOR_CORES = None
# A100-SXM4-80GB, Benchmark_loup (OMAT-0 medium, SrTiO3 5,000 atoms): MD step b/a 0.933 / 0.933 in both orders;
# loup 3.17 -> 2.92 (NVE), 3.12 -> 2.90 us/atom/step (Langevin). Force gate passes (SrTiO3 energy error 1.39 meV
# against 0.73, floor 10 meV). Slower on the RTX 5060 Ti and the RTX 4000 Ada (TF32 there is 2x FP32).
TENSOR_CORES_GPUS = GPUModels(("NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"))  # every A100 model
TC_STRIDE = 40  # W slice row stride with tensor cores: the fragment loads hit 32 distinct banks
# The W slice split into its TF32 parts once per block (float2 {hi, lo} per weight) instead of in every batch's
# fragment loads: bitwise the same product (the same TC_SPLIT), twice the slice's shared memory. A/B switch.
TC_PRESPLIT = False
TC_PSTRIDE = 36  # the pre-split slice's row stride in float2: the 64-bit fragment loads are conflict-free
TC_RSTRIDE = 36  # the redistribution tile's row stride: conflict-free fragment stores

_REGISTRY = {}


def tensor_cores(capability, name=None):
    """Whether the a.W product runs on tensor cores: ``TENSOR_CORES`` (None: the device ``name`` in
    ``TENSOR_CORES_GPUS``) and compute capability 8.0 or later."""
    on = TENSOR_CORES
    if on is None:
        on = name in TENSOR_CORES_GPUS
    return bool(on) and tuple(capability) >= (8, 0)


def _kpad(k):
    return -(-int(k) // 8) * 8


# How the tensor-core product splits x into TF32 high and low parts. "round": both parts rounded to nearest by an
# integer add and mask (ties away from zero), as accurate as "rna" (cvt.rna.tf32, ties away too; the same errors in
# the GPU test). "mask": the high part truncated (low 13 mantissa bits masked off), the low part the exact remainder
# passed as it is (the tensor core truncates it): 19 instead of 7.6 units of 2^-24 of the row scale at k = 9 on the
# RTX 5060 Ti. A100-SXM4-80GB kernel lab (4 November 2026): the fused forward 3.67 ms with "mask", 4.20 with the
# high part masked and the low part rounded by cvt.rna, 5.13 with "rna", 4.89 for the FP32-SIMT product.
TC_SPLIT = "mask"  # owner, 4 November 2026 ("Yes" to the A100 default with the mask split)
_TC_SPLITS = {
    "mask": """    hi = __float_as_uint(x) & 0xffffe000u;
    lo = __float_as_uint(x - __uint_as_float(hi));""",
    "round": """    hi = (__float_as_uint(x) + 0x1000u) & 0xffffe000u;
    lo = (__float_as_uint(x - __uint_as_float(hi)) + 0x1000u) & 0xffffe000u;""",
    "rna": """    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(hi) : "f"(x));
    const float rest = x - __uint_as_float(hi);
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(lo) : "f"(rest));""",
}


def tc_helpers():
    """The tensor-core kernels' device helpers: ``split_tf32`` (``TC_SPLIT``) and ``mma_tf32``."""
    return TC_HELPERS.replace("    // split\n", _TC_SPLITS[TC_SPLIT] + "\n")


TC_HELPERS = """// 3xTF32: x = hi + lo, both TF32 (TC_SPLIT); a product keeps hi*hi + hi*lo + lo*hi.
__device__ __forceinline__ void split_tf32(float x, unsigned& hi, unsigned& lo) {
    // split
}
__device__ __forceinline__ void mma_tf32(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                 "{%8, %9}, {%0, %1, %2, %3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
"""


# Exponent-shifted 3xFP16 (round 2, 5 November 2026; kernel lab only, ``tc="fp16"``): W split once per block into
# two FP16 planes (big, and the remainder times 2^11), the same 4 bytes per weight as the FP32 slice; a split per
# batch; m16n8k16 MMAs, C = big*big + 2^-11 (big*small + small*big). Power-of-two scales keep both operands in FP16's
# normal range: one per block slice for W, one per edge for a (its row maximum), undone on w. FP16 and TF32 both
# carry 11 significant bits: the 22-bit split matches 3xTF32 with both parts rounded to nearest ("round").
TC_FP16_HELPERS = """// 3xFP16 (tc="fp16"): x = big + 2^-11 small, both FP16 (rounded to nearest); low k packed first.
__device__ __forceinline__ void split_f16x2(float x0, float x1, unsigned& hi, unsigned& lo) {
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(hi) : "f"(x1), "f"(x0));
    float b0, b1;
    asm("{.reg .b16 l, u; mov.b32 {l, u}, %2; cvt.f32.f16 %0, l; cvt.f32.f16 %1, u;}"
        : "=f"(b0), "=f"(b1) : "r"(hi));
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(lo) : "f"((x1 - b1) * 2048.f), "f"((x0 - b0) * 2048.f));
}
__device__ __forceinline__ void mma_f16(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                 "{%8, %9}, {%0, %1, %2, %3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// The power of two that maps a largest magnitude m into [2^13, 2^14) (1 for m = 0).
__device__ __forceinline__ float fp16_scale(float m) {
    if (!(m > 0.f)) return 1.f;
    int e;
    frexpf(m, &e);
    int s = 14 - e;
    s = s < -126 ? -126 : (s > 127 ? 127 : s);
    return __int_as_float((127 + s) << 23);
}
"""


def tc_wfill_fp16(npl, stride):
    """The W slice as two FP16 planes of half2 k-pairs, ``[npl][KP / 2][WS]`` words each (big, then small), scaled by
    one power of two per slice (``wscale``). The slice maximum goes through the warps' staging words (A_BASE)."""
    return f"""            {{
                float m = 0.f;
                for (int i = threadIdx.x; i < {npl} * K * CB; i += blockDim.x) {{
                    const int p = i / (K * CB), rem = i - p * (K * CB), k = rem / CB, l = rem - k * CB;
                    m = fmaxf(m, fabsf(W[(int64_t)(wst[p] + cb * CB + l) * {stride} + k]));
                }}
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(FULL, m, o));
                if (lane == 0) smem[A_BASE + warp * A_STRIDE] = m;
                __syncthreads();
                m = 0.f;
                for (int v = 0; v < (int)(blockDim.x >> 5); ++v) m = fmaxf(m, smem[A_BASE + v * A_STRIDE]);
                wscale = fp16_scale(m);
                __syncthreads();
            }}
            {{
                unsigned* big = reinterpret_cast<unsigned*>(Ws);
                unsigned* small = big + {npl} * (KP / 2) * WS;
                for (int i = threadIdx.x; i < {npl} * (KP / 2) * CB; i += blockDim.x) {{
                    const int p = i / ((KP / 2) * CB), rem = i - p * ((KP / 2) * CB), kk = rem / CB,
                              l = rem - kk * CB;
                    const int64_t row = (int64_t)(wst[p] + cb * CB + l) * {stride};
                    const float x0 = 2 * kk < K ? W[row + 2 * kk] * wscale : 0.f;
                    const float x1 = 2 * kk + 1 < K ? W[row + 2 * kk + 1] * wscale : 0.f;
                    unsigned hi, lo;
                    split_f16x2(x0, x1, hi, lo);
                    big[(p * (KP / 2) + kk) * WS + l] = hi;
                    small[(p * (KP / 2) + kk) * WS + l] = lo;
                }}
            }}"""


def tc_astage_fp16(stride):
    """A warp's a staging [KP][B], each edge's row scaled by its own power of two; lane b keeps edge b's inverse
    scale (``eunscale``)."""
    return f"""                #pragma unroll 1
                for (int b = 0; b < B; ++b) {{
                    float v[(KP + 31) / 32], m = 0.f;
                    #pragma unroll
                    for (int j = 0; j < (KP + 31) / 32; ++j) {{
                        const int k = lane + 32 * j;
                        v[j] = (b < nb && k < K) ? a[(e0 + b) * {stride} + k] : 0.f;
                        m = fmaxf(m, fabsf(v[j]));
                    }}
                    #pragma unroll
                    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(FULL, m, o));
                    const float sc = fp16_scale(m);
                    if (lane == b) eunscale = 1.f / sc;
                    #pragma unroll
                    for (int j = 0; j < (KP + 31) / 32; ++j) {{
                        const int k = lane + 32 * j;
                        if (k < KP) a_s[k * B + b] = v[j] * sc;
                    }}
                }}"""


def tc_product_fp16(npl):
    """``float w[npl][B]`` from the staged, scaled a and the FP16 W planes through 3xFP16 m16n8k16 tiles."""
    return f"""                float wacc[{npl}][2][4];
                #pragma unroll
                for (int g = 0; g < {npl}; ++g)
                    #pragma unroll
                    for (int m = 0; m < 2; ++m)
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) wacc[g][m][i] = 0.f;
                {{
                    const int gid = lane >> 2, tq = lane & 3;
                    const unsigned* big = reinterpret_cast<const unsigned*>(Ws);
                    const unsigned* small = big + {npl} * (KP / 2) * WS;
                    #pragma unroll 2
                    for (int k0 = 0; k0 < KP; k0 += 16) {{
                        unsigned bh[2], bl[2];  // a^T fragment: k = k0 + 2 tq (+1; +8, +9), edge = gid
                        split_f16x2(a_s[(k0 + 2 * tq) * B + gid], a_s[(k0 + 2 * tq + 1) * B + gid], bh[0], bl[0]);
                        split_f16x2(a_s[(k0 + 2 * tq + 8) * B + gid], a_s[(k0 + 2 * tq + 9) * B + gid], bh[1], bl[1]);
                        #pragma unroll
                        for (int g = 0; g < {npl}; ++g)
                            #pragma unroll
                            for (int m = 0; m < 2; ++m) {{  // W fragment: channel m*16 + gid (+8), pair k0/2 + tq (+4)
                                const int o = (g * (KP / 2) + k0 / 2 + tq) * WS + m * 16 + gid;
                                const unsigned ah[4] = {{big[o], big[o + 8], big[o + 4 * WS], big[o + 4 * WS + 8]}};
                                const unsigned al[4] = {{small[o], small[o + 8], small[o + 4 * WS],
                                                        small[o + 4 * WS + 8]}};
                                float cross[4] = {{0.f, 0.f, 0.f, 0.f}};
                                mma_f16(cross, al, bh);
                                mma_f16(cross, ah, bl);
                                mma_f16(wacc[g][m], ah, bh);
                                #pragma unroll
                                for (int i = 0; i < 4; ++i) wacc[g][m][i] = fmaf(cross[i], 0x1p-11f, wacc[g][m][i]);
                            }}
                    }}
                    #pragma unroll
                    for (int g = 0; g < {npl}; ++g)
                        #pragma unroll
                        for (int m = 0; m < 2; ++m) {{  // C fragment: channel m*16 + gid (+8), edge 2 tq (+1)
                            float* rp = r_s + (g * B + 2 * tq) * RS + m * 16 + gid;
                            rp[0] = wacc[g][m][0];
                            rp[RS] = wacc[g][m][1];
                            rp[8] = wacc[g][m][2];
                            rp[RS + 8] = wacc[g][m][3];
                        }}
                }}
                __syncwarp();
                float w[{npl}][B];
                const float wun = 1.f / wscale;
                #pragma unroll
                for (int b = 0; b < B; ++b) {{
                    const float un = __shfl_sync(FULL, eunscale, b) * wun;
                    #pragma unroll
                    for (int g = 0; g < {npl}; ++g) w[g][b] = r_s[(g * B + b) * RS + lane] * un;
                }}"""


def tc_wfill(npl, stride, presplit=False):
    """A block's W slice for the tensor-core product: rows padded to KP with zeros, row stride WS; ``stride``: the
    name of W's row stride in the kernel. ``presplit``: stored as float2 {hi, lo} (``TC_PRESPLIT``)."""
    if presplit:
        return f"""            for (int i = threadIdx.x; i < {npl} * KP * CB; i += blockDim.x) {{
                const int p = i / (KP * CB), rem = i - p * (KP * CB), k = rem / CB, l = rem - k * CB;
                unsigned hi, lo;
                split_tf32(k < K ? W[(int64_t)(wst[p] + cb * CB + l) * {stride} + k] : 0.f, hi, lo);
                reinterpret_cast<float2*>(Ws)[(p * KP + k) * WS + l] = make_float2(__uint_as_float(hi),
                                                                                   __uint_as_float(lo));
            }}"""
    return f"""            for (int i = threadIdx.x; i < {npl} * KP * CB; i += blockDim.x) {{
                const int p = i / (KP * CB), rem = i - p * (KP * CB), k = rem / CB, l = rem - k * CB;
                Ws[(p * KP + k) * WS + l] = k < K ? W[(int64_t)(wst[p] + cb * CB + l) * {stride} + k] : 0.f;
            }}"""


def tc_astage(stride):
    """A warp's a staging [KP][B] for the tensor-core product, zero past K and past the batch."""
    return f"""                for (int i = lane; i < B * KP; i += 32) {{
                    const int b = i / KP, k = i - b * KP;
                    a_s[k * B + b] = (b < nb && k < K) ? a[(e0 + b) * {stride} + k] : 0.f;
                }}"""


def tc_product(npl, presplit=False):
    """``float w[npl][B]`` (lane = channel) from the staged a and the W slice through 3xTF32 mma.sync m16n8k8 tiles
    (16 channels x B = 8 edges) and the warp's redistribution tile ``r_s``; ``presplit``: the slice holds the
    split parts (``tc_wfill``)."""
    if presplit:
        wload = """                                const float2* wp = reinterpret_cast<const float2*>(Ws)
                                                   + (g * KP + k0 + tq) * WS + m * 16 + gid;
                                const float2 v0 = wp[0], v1 = wp[8], v2 = wp[4 * WS], v3 = wp[4 * WS + 8];
                                const unsigned ah[4] = {__float_as_uint(v0.x), __float_as_uint(v1.x),
                                                        __float_as_uint(v2.x), __float_as_uint(v3.x)};
                                const unsigned al[4] = {__float_as_uint(v0.y), __float_as_uint(v1.y),
                                                        __float_as_uint(v2.y), __float_as_uint(v3.y)};"""
    else:
        wload = """                                const float* wp = Ws + (g * KP + k0 + tq) * WS + m * 16 + gid;
                                unsigned ah[4], al[4];
                                split_tf32(wp[0], ah[0], al[0]);
                                split_tf32(wp[8], ah[1], al[1]);
                                split_tf32(wp[4 * WS], ah[2], al[2]);
                                split_tf32(wp[4 * WS + 8], ah[3], al[3]);"""
    return f"""                float wacc[{npl}][2][4];
                #pragma unroll
                for (int g = 0; g < {npl}; ++g)
                    #pragma unroll
                    for (int m = 0; m < 2; ++m)
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) wacc[g][m][i] = 0.f;
                {{
                    const int gid = lane >> 2, tq = lane & 3;
                    #pragma unroll 2
                    for (int k0 = 0; k0 < KP; k0 += 8) {{
                        unsigned bh[2], bl[2];  // a^T fragment: k = k0 + tq (+4), edge = gid
                        split_tf32(a_s[(k0 + tq) * B + gid], bh[0], bl[0]);
                        split_tf32(a_s[(k0 + tq + 4) * B + gid], bh[1], bl[1]);
                        #pragma unroll
                        for (int g = 0; g < {npl}; ++g)
                            #pragma unroll
                            for (int m = 0; m < 2; ++m) {{  // W fragment: channel m*16 + gid (+8), k0 + tq (+4)
{wload}
                                mma_tf32(wacc[g][m], al, bh);
                                mma_tf32(wacc[g][m], ah, bl);
                                mma_tf32(wacc[g][m], ah, bh);
                            }}
                    }}
                    #pragma unroll
                    for (int g = 0; g < {npl}; ++g)
                        #pragma unroll
                        for (int m = 0; m < 2; ++m) {{  // C fragment: channel m*16 + gid (+8), edge 2 tq (+1)
                            float* rp = r_s + (g * B + 2 * tq) * RS + m * 16 + gid;
                            rp[0] = wacc[g][m][0];
                            rp[RS] = wacc[g][m][1];
                            rp[8] = wacc[g][m][2];
                            rp[RS + 8] = wacc[g][m][3];
                        }}
                }}
                __syncwarp();
                float w[{npl}][B];
                #pragma unroll
                for (int g = 0; g < {npl}; ++g)
                    #pragma unroll
                    for (int b = 0; b < B; ++b) w[g][b] = r_s[(g * B + b) * RS + lane];"""


def used_columns(weight):
    """The folded inputs the k-loop reads: ``weight.shape[1]``, one fewer with ``SKIP_ZERO_COLUMN`` (by default on
    ``SKIP_ZERO_COLUMN_GPUS``) when the weight's last column is all zero."""
    k = int(weight.shape[1])
    skip = SKIP_ZERO_COLUMN
    if skip is None:
        skip = weight.device.type == "cuda" and torch.cuda.get_device_name(weight.device) in SKIP_ZERO_COLUMN_GPUS
    return k - 1 if skip and k > 1 and not bool(weight[:, -1].any()) else k


def _lit(value):
    return f"{float(value).hex()}f"


@dataclass(frozen=True)
class LaunchRule:
    parts: int
    warps: int
    blocks_per_slice: int
    shared_bytes: int


def slice_floats(paths_per_part, k, tc=False):
    """Floats of a block's W slice (row stride CHANNEL_BLOCK + 1; with tensor cores ``TC_STRIDE`` and k padded to a
    multiple of 8; ``tc="presplit"``: ``TC_PSTRIDE`` float2), rounded up to keep the staging 16-byte aligned."""
    rows, stride = ((_kpad(k), 2 * TC_PSTRIDE) if tc == "presplit" else (_kpad(k), TC_STRIDE) if tc
                    else (k, CHANNEL_BLOCK + 1))
    return -(-paths_per_part * rows * stride // 4) * 4


def staging_floats(k, edge_dim, tc=False, paths_per_part=0):
    """Floats of one warp's staging: a [K][BATCH] and sh [BATCH][edge_dim]; with tensor cores a [K padded][BATCH]
    and the redistribution tile [paths per part][BATCH][TC_RSTRIDE] after them."""
    if tc:
        return BATCH * (_kpad(k) + edge_dim) + paths_per_part * BATCH * TC_RSTRIDE
    return BATCH * (k + edge_dim)


def launch_rule(paths, k, edge_dim, channels, shared_optin, sms, tc=False):
    """The kernel's parts, warps and grid for ``paths`` uvu paths of ``k``-wide folded inputs on a device with
    ``shared_optin`` bytes of opt-in shared memory per block and ``sms`` SMs (module docstring); ``tc``: the
    tensor-core product's shared layout (``TENSOR_CORES``)."""
    parts = max(1, -(-paths * BATCH // W_TILE_REGISTERS))
    while parts <= paths:
        per = -(-paths // parts)
        if 4 * slice_floats(per, k, tc) + MIN_WARPS * 4 * staging_floats(k, edge_dim, tc, per) <= shared_optin:
            break
        parts += 1
    else:
        raise ValueError(f"fused_folded_forward: no W slice of {k}-wide rows fits {shared_optin} bytes of "
                         "shared memory")
    per = -(-paths // parts)
    parts = -(-paths // per)
    staging = 4 * staging_floats(k, edge_dim, tc, per)
    wbytes = 4 * slice_floats(per, k, tc)
    warps = min(MAX_WARPS, (shared_optin - wbytes) // staging)
    slices = (channels // CHANNEL_BLOCK) * parts
    return LaunchRule(parts, warps, max(1, -(-BLOCKS_PER_SM * sms // slices)), wbytes + warps * staging)


def balanced_parts(costs, parts):
    """Paths split into ``parts`` groups of equal size (the last may be smaller); the angular terms (``costs``)
    balanced greedily, heaviest first; each group sorted."""
    per = -(-len(costs) // parts)
    groups, load = [[] for _ in range(parts)], [0] * parts
    for index in sorted(range(len(costs)), key=lambda i: (-costs[i], i)):
        q = min((q for q in range(parts) if len(groups[q]) < per), key=lambda q: (load[q], q))
        groups[q].append(index)
        load[q] += costs[index]
    return [sorted(g) for g in groups if g]


def check_problem(schedule):
    """Raises ValueError unless the fused forward can run this schedule."""
    if schedule.channels % CHANNEL_BLOCK:
        raise ValueError(f"fused_folded_forward needs channels divisible by {CHANNEL_BLOCK}, "
                         f"not {schedule.channels}")
    if any(p.edge_mul != 1 for p in schedule.paths):
        raise ValueError("fused_folded_forward needs one edge channel per path")
    if schedule.layout not in ("ir_mul", "mul_ir"):
        raise ValueError(f"unknown layout {schedule.layout!r}")


def block_map(blocks, channels):
    """``{segment start: (group, row width, rank offset, component stride)}`` of a block-layout message
    (``wprime_backward.BLOCK_MESSAGE``): ``blocks`` lists per group its irrep dimension and the starts of its
    segments in rank order; group ``g`` is ``[N * dim, P * channels]``, row ``n * dim + component``, column
    ``rank * channels + channel``."""
    out = {}
    for g, (dim, starts) in enumerate(blocks):
        width = int(dim) * len(starts) * channels
        for rank, start in enumerate(starts):
            out[int(start)] = (g, width, rank * channels, len(starts) * channels)
    return out


def block_shapes(blocks, channels, n):
    """The groups' shapes ``[n * dim, P * channels]`` of a block-layout message."""
    return [(n * int(dim), len(starts) * channels) for dim, starts in blocks]


def _node(layout, channels, start, component, dim):
    """The node-feature offset expression of the vendored template (``node_offset``) for channel ``ch``."""
    if layout == "ir_mul":
        return f"{start + component * channels} + ch"
    return f"{start + component} + ch * {dim}"


def render(schedule, k, rule, store_w=True, used=None, stream_store=None, tc=False, blocks=None):
    """CUDA source of ``fused_folded_forward`` for a schedule, the folded input width ``k`` and a launch rule.
    ``store_w=False``: the message alone; ``w`` is neither stored nor built for the padding edges (``wout`` unused;
    plans/capacity.md, the w-rebuild falsifier). ``used``: the leading inputs the k-loop reads (``used_columns``;
    default ``k``), a's and W's rows keep the stride ``k``. ``stream_store`` (default ``STREAM_STORE_W``): ``w``
    stored with ``__stcs``. ``tc``: the a.W product on tensor cores in 3xTF32 (``TENSOR_CORES``, sm_80+; the launch
    rule must be ``launch_rule(..., tc=tc)``'s); ``"presplit"``: with the W slice split once (``TC_PRESPLIT``).
``"fp16"``: exponent-shifted 3xFP16 instead (``TC_FP16_HELPERS``; kernel lab only, used columns padded to 16).
    ``blocks`` (``block_map``'s input; ir_mul, message only): the message written per group, one output
    pointer each (``out0``, ``out1``, ...)."""
    check_problem(schedule)
    if blocks is not None and (store_w or schedule.layout != "ir_mul"):
        raise ValueError("fused_folded_forward: a block-layout message needs store_w=False and ir_mul")
    bmap = None if blocks is None else block_map(blocks, schedule.channels)
    outs_decl = ("float* __restrict__ out" if blocks is None
                 else ", ".join(f"float* __restrict__ out{g}" for g in range(len(blocks))))
    used = int(k) if used is None else int(used)
    if not 1 <= used <= k:
        raise ValueError(f"fused_folded_forward: {used} used columns of {k}")
    stream_store = STREAM_STORE_W if stream_store is None else bool(stream_store)
    store = ("__stcs(&wout[(e0 + b) * Q + wst[g] + ch], w[g][b]);" if stream_store
             else "wout[(e0 + b) * Q + wst[g] + ch] = w[g][b];")
    C, layout, paths = schedule.channels, schedule.layout, schedule.paths
    costs = [sum(len(o.terms) for o in p.outputs) for p in paths]
    parts = balanced_parts(costs, rule.parts)
    owner = {}
    for q, part in enumerate(parts):
        for pi in part:
            for o in paths[pi].outputs:
                if owner.setdefault(o.accumulator.name, q) != q:
                    raise ValueError("fused_folded_forward: an output slot is shared by two path parts")
    slots = {s.name: s for s in schedule.output_slots}
    npp = max(len(p) for p in parts)
    cases = "\n".join(f"    case {q}: {{{_part(schedule, part, slots, C, layout, store, tc, bmap)}    }} break;"
                      for q, part in enumerate(parts))
    if tc == "fp16" and _kpad(used) % 16:
        raise ValueError(f"tc='fp16' needs the used columns padded to a multiple of 16, not {_kpad(used)}")
    helpers = (tc_helpers() + TC_FP16_HELPERS) if tc == "fp16" else tc_helpers() if tc else ""
    ka = used if tc else k  # the staged width of a (tensor cores: the used columns)
    a_base, a_stride = slice_floats(npp, ka, tc), staging_floats(ka, schedule.edge_dim, tc, npp)
    a_offset = "A_BASE + warp * A_STRIDE"
    return f"""// fused_folded_forward (FastMACE opt-8): folded last layer + receiver-streaming forward, w stored.
using int64_t = long long;
constexpr bool STORE_W = {"true" if store_w else "false"};
constexpr int K = {used};  // the folded inputs the k-loop reads
constexpr int KA = {k};  // a's and W's row stride
constexpr int KP = {_kpad(used) if tc else used};  // the k-loop's padded length (tensor cores: steps of 8)
constexpr int RS = {TC_RSTRIDE};
constexpr int CB = {CHANNEL_BLOCK};
constexpr int WS = {TC_PSTRIDE if tc == "presplit" else TC_STRIDE if tc else CHANNEL_BLOCK + 1};
constexpr int ED = {schedule.edge_dim};
constexpr int IN_DIM = {schedule.input_dim};
constexpr int OUT_DIM = {schedule.output_dim};
constexpr int64_t Q = {len(paths) * C};
constexpr int B = {BATCH};
constexpr int RC = {RECEIVERS_PER_CHUNK};
constexpr int TAIL = {TAIL_EDGES};
constexpr unsigned FULL = 0xffffffffu;
constexpr int A_BASE = {a_base};  // the warps' staging: a_s of warp v at A_BASE + v * A_STRIDE
constexpr int A_STRIDE = {a_stride};
{helpers}
extern "C" __global__ void __launch_bounds__({rule.warps * 32}, 1) fused_folded_forward(
    int64_t N, int64_t E, const float* __restrict__ a, const float* __restrict__ W,
    const float* __restrict__ x, const float* __restrict__ sh, const int* __restrict__ senders,
    const int* __restrict__ receivers, const int* __restrict__ row_ptr, int* __restrict__ counters,
    {outs_decl}, float* __restrict__ wout) {{
    extern __shared__ float smem[];
    float* Ws = smem;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float* a_s = Ws + {a_offset};
    float* sh_s = a_s + KP * B;
    float* r_s = sh_s + B * ED;  // tensor cores: the warp's redistribution tile [paths][B][RS]
    const int cb = blockIdx.y;
    const int ch = cb * CB + lane;
    int* ctr = counters + blockIdx.y * gridDim.z + blockIdx.z;
    // Receiver chunks first, then chunks of the padding edges past the last row.
    const int64_t receiver_chunks = (N + RC - 1) / RC;
    int64_t tail_begin = row_ptr[N] < 0 ? 0 : row_ptr[N];
    tail_begin = tail_begin > E ? E : tail_begin;
    switch (blockIdx.z) {{
{cases}
    }}
}}
"""


def _out(slot, C, layout, bmap):
    if bmap is None:
        return f"out[cur * OUT_DIM + {_node(layout, C, slot.start, slot.component, slot.irrep_dim)}]"
    g, width, base, stride = bmap[slot.start]
    return f"out{g}[cur * {width} + {base + slot.component * stride} + ch]"


def _part(schedule, part, slots, C, layout, store, tc=False, bmap=None):
    paths = schedule.paths
    xkeys = sorted({(paths[pi].input_start, t.input_component, paths[pi].input_irrep_dim)
                    for pi in part for o in paths[pi].outputs for t in o.terms})
    xi = {key: i for i, key in enumerate(xkeys)}
    nx = len(xkeys)
    names = list(dict.fromkeys(o.accumulator.name for pi in part for o in paths[pi].outputs))
    si = {n: i for i, n in enumerate(names)}
    n = len(names)
    npl = len(part)
    wst = ", ".join(str(paths[pi].weight_start) for pi in part)
    xloads = "\n".join(f"                    xv[b][{i}] = (snd >= 0) ? x[(int64_t)snd * IN_DIM + "
                       f"{_node(layout, C, s, c, d)}] : 0.f;" for (s, c, d), i in xi.items())
    outs = "\n".join(f"                {_out(slots[name], C, layout, bmap)} = acc[{i}];" for name, i in si.items())
    gemm = []
    for q in range(BATCH // 4):
        gemm.append(f"                    const float4 av{q} ="
                    f" *reinterpret_cast<const float4*>(&a_s[k * B + {4 * q}]);")
    for g in range(npl):
        gemm.append(f"                    {{ const float wk = Ws[({g} * K + k) * WS + lane];")
        for q in range(BATCH // 4):
            for c, comp in enumerate("xyzw"):
                b = 4 * q + c
                gemm.append(f"                      w[{g}][{b}] = fmaf(av{q}.{comp}, wk, w[{g}][{b}]);")
        gemm.append("                    }")
    fp16_decl = "float wscale = 1.f, eunscale = 1.f;" if tc == "fp16" else ""
    if tc == "fp16":
        wfill, astage, product = tc_wfill_fp16(npl, "KA"), tc_astage_fp16("KA"), tc_product_fp16(npl)
    elif tc:
        pre = tc == "presplit"
        wfill, astage, product = tc_wfill(npl, "KA", pre), tc_astage("KA"), tc_product(npl, pre)
    else:
        wfill = f"""            for (int i = threadIdx.x; i < {npl} * K * CB; i += blockDim.x) {{
                const int p = i / (K * CB), rem = i - p * (K * CB), k = rem / CB, l = rem - k * CB;
                Ws[(p * K + k) * WS + l] = W[(int64_t)(wst[p] + cb * CB + l) * KA + k];
            }}"""
        astage = """                for (int i = lane; i < B * K; i += 32) {
                    const int b = i / K, k = i - b * K;
                    a_s[k * B + b] = (b < nb) ? a[(e0 + b) * KA + k] : 0.f;
                }"""
        nl = "\n"
        product = f"""                float w[{npl}][B];
                #pragma unroll
                for (int g = 0; g < {npl}; ++g)
                    #pragma unroll
                    for (int b = 0; b < B; ++b) w[g][b] = 0.f;
                #pragma unroll 5
                for (int k = 0; k < K; ++k) {{
{nl.join(gemm)}
                }}"""
    tp = []
    for g, pi in enumerate(part):
        p = paths[pi]
        tp.append("                        {")
        for j in range(p.edge_irrep_dim):
            tp.append(f"                        const float e{j} = sh_s[b * ED + {p.edge_start + j}];")
        for o in p.outputs:
            terms = " + ".join(f"xv[b][{xi[(p.input_start, t.input_component, p.input_irrep_dim)]}]"
                               f" * e{t.edge_component} * {_lit(t.coefficient)}" for t in o.terms) or "0.f"
            tp.append(f"                        acc[{si[o.accumulator.name]}] = fmaf(w[{g}][b], {terms},"
                      f" acc[{si[o.accumulator.name]}]);")
        tp.append("                        }")
    flush = f"""{{
{outs}
                #pragma unroll
                for (int i = 0; i < {n}; ++i) acc[i] = 0.f;
                ++cur;
            }}"""
    return f"""
        {fp16_decl}
        {{
            const int wst[{npl}] = {{{wst}}};
{wfill}
        }}
        __syncthreads();
        while (true) {{
            int cc = 0;
            if (lane == 0) cc = atomicAdd(ctr, 1);
            cc = __shfl_sync(FULL, cc, 0);
            int64_t r0, r1, ebeg, eend;
            const bool tail = cc >= receiver_chunks;
            if (!tail) {{
                r0 = (int64_t)cc * RC;
                r1 = r0 + RC < N ? r0 + RC : N;
                ebeg = row_ptr[r0] < 0 ? 0 : row_ptr[r0];
                eend = row_ptr[r1] < ebeg ? ebeg : row_ptr[r1];
                eend = eend > E ? E : eend;
            }} else {{
                r0 = r1 = N;
                ebeg = tail_begin + (cc - receiver_chunks) * TAIL;
                if (ebeg >= E || !STORE_W) break;
                eend = ebeg + TAIL < E ? ebeg + TAIL : E;
            }}
            float acc[{n}];
            #pragma unroll
            for (int i = 0; i < {n}; ++i) acc[i] = 0.f;
            int64_t cur = r0;
            for (int64_t e0 = ebeg; e0 < eend; e0 += B) {{
                const int nb = (int)((eend - e0) < B ? (eend - e0) : B);
                __syncwarp();
{astage}
                for (int i = lane; i < B * ED; i += 32) sh_s[i] = (i / ED < nb) ? sh[e0 * ED + i] : 0.f;
                float xv[B][{nx}];
                int rv[B];
                #pragma unroll
                for (int b = 0; b < B; ++b) {{
                    rv[b] = (b < nb && !tail) ? receivers[e0 + b] : 0x7fffffff;
                    int snd = (b < nb && !tail) ? senders[e0 + b] : -1;
                    if (snd < 0 || snd >= N) snd = -1;
{xloads}
                }}
                __syncwarp();
{product}
                if (STORE_W) {{
                    const int wst[{npl}] = {{{wst}}};
                    #pragma unroll
                    for (int b = 0; b < B; ++b)
                        if (b < nb) {{
                            #pragma unroll
                            for (int g = 0; g < {npl}; ++g) {store}
                        }}
                }}
                if (!tail) {{
                    #pragma unroll
                    for (int b = 0; b < B; ++b) {{
                        if (b >= nb) break;
                        while (cur < rv[b]) {flush}
{chr(10).join(tp)}
                    }}
                }}
            }}
            while (cur < r1) {flush}
        }}
"""


class FusedFoldedForward:
    """Compiled ``fused_folded_forward`` for the streaming kernels of one interaction (``StreamingKernels``) and a
    folded input width ``k``; float32 only. ``used``: the inputs the k-loop reads (``used_columns``; default ``k``)."""

    def __init__(self, kernels, k, device=None, store_w=True, used=None, blocks=None):
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check, _device_key, compile_cubin, content_id
        if "using scalar_t = float;" not in kernels.source:
            raise ValueError("fused_folded_forward needs float32 streaming kernels")
        schedule = kernels.schedule
        check_problem(schedule)
        self.schedule = schedule
        device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
        props = torch.cuda.get_device_properties(device)
        self.k, self.channels, self.paths = int(k), schedule.channels, len(schedule.paths)
        self.used = self.k if used is None else int(used)
        self.tc = tensor_cores(torch.cuda.get_device_capability(device), torch.cuda.get_device_name(device))
        self.tc = "presplit" if self.tc and TC_PRESPLIT else self.tc
        self.rule = launch_rule(len(schedule.paths), self.used if self.tc else self.k, schedule.edge_dim,
                                schedule.channels, int(props.shared_memory_per_block_optin),
                                int(props.multi_processor_count), tc=self.tc)
        self.output_dim, self.weight_dim = schedule.output_dim, kernels.weight_dim
        self.store_w = bool(store_w)
        self.blocks = None if blocks is None else tuple((int(d), tuple(int(s) for s in st)) for d, st in blocks)
        self.source = render(schedule, self.k, self.rule, self.store_w, self.used, tc=self.tc, blocks=self.blocks)
        cubin = compile_cubin(self.source, torch.cuda.get_device_capability(device))
        torch.cuda.current_stream(device)
        self._module = _check(driver.cuModuleLoadData(cubin))
        self._function = _check(driver.cuModuleGetFunction(self._module, b"fused_folded_forward"))
        attribute = driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        _check(driver.cuFuncSetAttribute(self._function, attribute, self.rule.shared_bytes))
        self.slices = (self.channels // CHANNEL_BLOCK) * self.rule.parts
        self.id = content_id("FusedFoldedForward", self.source, self.rule, _device_key(device))
        _REGISTRY[self.id] = self

    def __call__(self, a, weight, x, sh, senders, receivers, row_ptr, out, w):
        """Writes the message into ``out [N, output_dim]`` (a list of the groups with ``blocks``) and
        ``a @ weight.T`` into ``w [E, Q]``."""
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check
        nodes, edges = x.shape[0], sh.shape[0]
        counters = torch.zeros(self.slices, dtype=torch.int32, device=x.device)
        p = ctypes.c_void_p
        values = (nodes, edges, a.data_ptr(), weight.data_ptr(), x.data_ptr(), sh.data_ptr(), senders.data_ptr(),
                  receivers.data_ptr(), row_ptr.data_ptr(), counters.data_ptr(),
                  *([o.data_ptr() for o in out] if self.blocks else [out.data_ptr()]), w.data_ptr())
        types = (ctypes.c_longlong, ctypes.c_longlong) + (p,) * (len(values) - 2)
        stream = driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        _check(driver.cuLaunchKernel(self._function, self.rule.blocks_per_slice, self.channels // CHANNEL_BLOCK,
                                     self.rule.parts, self.rule.warps * 32, 1, 1, self.rule.shared_bytes, stream,
                                     (values, types), 0))


@torch.library.custom_op("fastmace::fused_folded_forward", mutates_args=())
def fused_folded_forward(a: torch.Tensor, weight: torch.Tensor, x: torch.Tensor, sh: torch.Tensor,
                         senders: torch.Tensor, receivers: torch.Tensor, row_ptr: torch.Tensor,
                         fused_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(message [N, output_dim], w = a @ weight.T [E, Q]) through the fused kernel ``fused_id``."""
    fused = _REGISTRY[fused_id]
    a, weight, x, sh = a.contiguous(), weight.contiguous(), x.contiguous(), sh.contiguous()
    out = x.new_empty((x.shape[0], fused.output_dim))
    w = x.new_empty((sh.shape[0], weight.shape[0]))
    if sh.shape[0] or x.shape[0]:
        fused(a, weight, x, sh, senders.contiguous(), receivers.contiguous(), row_ptr.contiguous(), out, w)
    return out, w


@fused_folded_forward.register_fake
def _(a, weight, x, sh, senders, receivers, row_ptr, fused_id):
    return x.new_empty((x.shape[0], _REGISTRY[fused_id].output_dim)), x.new_empty((sh.shape[0], weight.shape[0]))


@torch.library.custom_op("fastmace::fused_folded_message", mutates_args=())
def fused_folded_message(a: torch.Tensor, weight: torch.Tensor, x: torch.Tensor, sh: torch.Tensor,
                         senders: torch.Tensor, receivers: torch.Tensor, row_ptr: torch.Tensor,
                         fused_id: int) -> torch.Tensor:
    """The message [N, output_dim] alone, through a kernel built with ``store_w=False`` (``recompute_conv_weights``,
    plans/capacity.md CAP-T07): ``w`` is built on chip and not stored."""
    fused = _REGISTRY[fused_id]
    if fused.store_w:
        raise ValueError("fused_folded_message needs a FusedFoldedForward built with store_w=False")
    a, weight, x, sh = a.contiguous(), weight.contiguous(), x.contiguous(), sh.contiguous()
    out = x.new_empty((x.shape[0], fused.output_dim))
    if sh.shape[0] or x.shape[0]:
        fused(a, weight, x, sh, senders.contiguous(), receivers.contiguous(), row_ptr.contiguous(), out,
              x.new_empty((1,)))
    return out


@fused_folded_message.register_fake
def _(a, weight, x, sh, senders, receivers, row_ptr, fused_id):
    return x.new_empty((x.shape[0], _REGISTRY[fused_id].output_dim))


@torch.library.custom_op("fastmace::fused_folded_message_blocks", mutates_args=())
def fused_folded_message_blocks(a: torch.Tensor, weight: torch.Tensor, x: torch.Tensor, sh: torch.Tensor,
                                senders: torch.Tensor, receivers: torch.Tensor, row_ptr: torch.Tensor,
                                fused_id: int) -> list[torch.Tensor]:
    """``fused_folded_message`` in the block layout (``wprime_backward.BLOCK_MESSAGE``): one ``[N * dim, P * C]``
    tensor per group of the kernel's ``blocks``, what the grouped linear_1 multiplies without a copy."""
    fused = _REGISTRY[fused_id]
    if fused.store_w or not fused.blocks:
        raise ValueError("fused_folded_message_blocks needs a FusedFoldedForward built with store_w=False and blocks")
    a, weight, x, sh = a.contiguous(), weight.contiguous(), x.contiguous(), sh.contiguous()
    out = [x.new_empty(shape) for shape in block_shapes(fused.blocks, fused.channels, x.shape[0])]
    if sh.shape[0] or x.shape[0]:
        fused(a, weight, x, sh, senders.contiguous(), receivers.contiguous(), row_ptr.contiguous(), out,
              x.new_empty((1,)))
    return out


@fused_folded_message_blocks.register_fake
def _(a, weight, x, sh, senders, receivers, row_ptr, fused_id):
    fused = _REGISTRY[fused_id]
    return [x.new_empty(shape) for shape in block_shapes(fused.blocks, fused.channels, x.shape[0])]

