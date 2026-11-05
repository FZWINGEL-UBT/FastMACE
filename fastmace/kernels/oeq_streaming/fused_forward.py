"""The folded radial last layer fused into the receiver-streaming forward (``fused_folded_forward``, opt-8 OPT8-T02).

Today's chain builds the per-edge weights ``w = a @ W.T`` with cuBLAS (``W [Q, K]``, ``Q = paths × channels``) and
the streaming forward reads them back. The fused forward builds each batch's ``w`` rows on chip from ``a`` and a
shared-memory slice of ``W``, uses them for the message at once and also stores them, so the chunked backward
(``chunked_conv_backward``) runs unchanged on them. It saves the cuBLAS product and the forward's ``w`` read.
The stored ``w`` and the message are bitwise those of the chain on the measured GPUs: each ``w`` element is the
same k-ordered FP32 sum of fused multiply-adds, and each receiver's message sums its edges in CSR order with the
same angular terms. Results: ``docs/validation/speed/opt8-T01.md`` (T01b) and ``docs/validation/speed/opt8-T02.md``.

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
# NVIDIA H200 (5 November 2026, results/h200-defaults-20261105): MD step b/a 1.092 / 1.090 (b = off; OMAT-0 medium,
# SrTiO3 10x10x10, abab / abba, every other A100 switch on).
SKIP_ZERO_COLUMN_GPUS = GPUModels(("NVIDIA RTX 4000 Ada Generation", "NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe",
                                   "NVIDIA H200"))
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
# NVIDIA H200 (5 November 2026, results/h200-defaults-20261105): MD step b/a 1.107 / 1.105 (b = off; OMAT-0 medium,
# SrTiO3 10x10x10, abab / abba, every other A100 switch on).
TENSOR_CORES_GPUS = GPUModels(("NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe", "NVIDIA H200"))  # every A100 model
TC_STRIDE = 40  # W slice row stride with tensor cores: the fragment loads hit 32 distinct banks
# The W slice split into its TF32 parts once per block (float2 {hi, lo} per weight) instead of in every batch's
# fragment loads: bitwise the same product (the same TC_SPLIT), twice the slice's shared memory. A/B switch.
TC_PRESPLIT = False
TC_PSTRIDE = 36  # the pre-split slice's row stride in float2: the 64-bit fragment loads are conflict-free
TC_RSTRIDE = 36  # the redistribution tile's row stride: conflict-free fragment stores
# Round 4 (5 November 2026): the staging double-buffered. While a warp computes a batch, cp.async (sm_80+) copies the
# next batch's a and sh into the warp's second buffer, so their load latency is hidden behind the product and the
# tensor product (kernel lab: without a's and sh's loads the tensor-core forward is 18 % and 12 % faster on the
# RTX 5060 Ti). Bitwise: the same values reach the same shared addresses. None: on for PREFETCH_GPUS (decision 66);
# True / False: an A/B side's choice on every sm_80+ GPU (the fp16 lab product excepted). "x": the senders' x rows
# too, gathered with cp.async into one more per-warp tile [BATCH][input_dim / channels][32] after the product (the
# A100 kernel lab: without the x loads the tensor-core forward is 16 % faster).
PREFETCH = None
PREFETCH_GPUS = GPUModels(())
# Round 4 (5 November 2026): the edge harmonics sh staged with float4 loads (one per lane and batch instead of four
# scalar ones; the kernel lab: the RTX 5060 Ti FP32-SIMT forward 0.916, its tensor-core forward 0.865). Bitwise. Needs
# edge_dim % 4 == 0; a sh whose address is not 16-byte aligned is copied first. None: on for VEC_SH_GPUS (decision
# 66); True / False: an A/B side's choice.
VEC_SH = None
# Defaults (decision 66, 5 November 2026): MD step b/a, SrTiO3 10x10x10 and NbCl5, both build orders: RTX 5060 Ti
# 0.977 / 0.977, 0.984 / 0.984 (VEC_SH alone); A100-SXM4-80GB with VEC_A 0.973 / 0.976, 0.976 / 0.981.
# NVIDIA H200 (5 November 2026, results/h200-defaults-20261105): MD step b/a 1.003 / 1.003 (b = off; OMAT-0 medium,
# SrTiO3 10x10x10, abab / abba, every other A100 switch on).
VEC_SH_GPUS = GPUModels(("NVIDIA A100-SXM4-80GB", "NVIDIA GeForce RTX 5060 Ti", "NVIDIA H200"))
# Round 4: the tensor-core kernel stages a with float4 loads from rows padded to a multiple of 4 floats (the radial
# table writes them padded: ``a_columns``; any other a is padded by a copy first) into an edge-major tile, XOR-swizzled
# per 4-float group so the fragment reads stay conflict-free: 4 vector loads per lane and batch instead of 16 scalar
# ones (A100-SXM4-80GB kernel lab, with VEC_SH: 0.903, the bound without any a loads 0.900). Bitwise. Needs the TF32
# product (not fp16) and used columns a multiple of 32. None: on for VEC_A_GPUS (decision 66); True / False: A/B.
VEC_A = None
# NVIDIA H200 (5 November 2026, results/h200-defaults-20261105): MD step b/a 1.047 / 1.046 (b = off; OMAT-0 medium,
# SrTiO3 10x10x10, abab / abba, every other A100 switch on).
VEC_A_GPUS = GPUModels(("NVIDIA A100-SXM4-80GB", "NVIDIA H200"))  # every A100 (see VEC_SH_GPUS; TC_WFRAG_GPUS)
# Round 4: the tensor-core kernel's W slice stored in mma.sync fragment order, [paths][KP / 8][2 m-tiles][32 lanes][4],
# so each lane loads its four W values of a fragment with one float4 shared load (consecutive 16 B per lane: no bank
# conflicts) instead of four scalar loads at row stride TC_STRIDE. The same values reach the same mma: bitwise.
# A100-SXM4-80GB kernel lab: 0.965 alone, 0.860 with VEC_A and VEC_SH (0.905 without). Needs the TF32 product (not
# presplit, not fp16) without prefetch. None: on for TC_WFRAG_GPUS (decision 66); True / False: an A/B side's choice.
TC_WFRAG = None
# Round 4 (5 November 2026): the tensor product on tensor cores too. The a.W product's C fragments hold w as
# (channel, edge) tiles; with the edge pair 2 tq, 2 tq + 1 of a lane as its k columns tq, tq + 4, (w x)[channel,
# (edge, m1)] is directly the A fragment of a second 3xTF32 product against S_e[m1, m3] = sum_m2 CG sh_e[m2] (built per
# edge from sh and the coefficients), and the sum over a receiver's edges is its k dimension: the message stays in
# mma accumulators until the receiver changes. No redistribution tile, no per-lane TP. Rounding (S pre-summed, the
# products in 3xTF32): not bitwise. Needs the TF32 product (not presplit), no prefetch, each path's outputs at most 8
# components of one segment. None: on for TC_TP_GPUS (decision 66); True / False: an A/B side's choice.
TC_TP = None
TC_TP_SPLIT = "round"  # the TP operands' split (``_TC_SPLITS``; "round": both parts rounded to nearest)
TC_TP_GPUS = GPUModels(())
# Default (decision 66): A100-SXM4-80GB with VEC_A and VEC_SH, MD step b/a 0.960 / 0.954 (SrTiO3), 0.970 / 0.969
# (NbCl5), both build orders; force gate as the plan (NbCl5 stress 1.57e-7, the A100's accepted exception).
# NVIDIA H200 (5 November 2026, results/h200-defaults-20261105): MD step b/a 1.022 / 1.024 (b = off; OMAT-0 medium,
# SrTiO3 10x10x10, abab / abba, every other A100 switch on).
TC_WFRAG_GPUS = GPUModels(("NVIDIA A100-SXM4-80GB", "NVIDIA H200"))

_REGISTRY = {}


def prefetch_default(capability, name=None):
    """Whether the kernel double-buffers its staging: ``PREFETCH`` (None: ``name`` in ``PREFETCH_GPUS``), compute
    capability 8.0 or later (cp.async)."""
    on = PREFETCH
    if on is None:
        on = PREFETCH_GPUS.get(name, False) if isinstance(PREFETCH_GPUS, dict) else name in PREFETCH_GPUS
    if not on or tuple(capability) < (8, 0):
        return False
    return "x" if on == "x" else True


def vec_sh_default(edge_dim, name=None):
    """Whether the kernel stages sh with float4 loads: ``VEC_SH`` (None: ``name`` in ``VEC_SH_GPUS``) and
    ``edge_dim % 4 == 0``."""
    on = VEC_SH
    if on is None:
        on = name in VEC_SH_GPUS
    return bool(on) and int(edge_dim) % 4 == 0


def vec_a_default(name=None):
    """``VEC_A`` (None: ``name`` in ``VEC_A_GPUS``); the kernel applies it only where it is built (``FusedFoldedForward``)."""
    on = VEC_A
    if on is None:
        on = name in VEC_A_GPUS
    return bool(on)


def tc_wfrag_default(name=None):
    """``TC_WFRAG`` (None: ``name`` in ``TC_WFRAG_GPUS``); built only where it applies (``FusedFoldedForward``)."""
    on = TC_WFRAG
    if on is None:
        on = name in TC_WFRAG_GPUS
    return bool(on)


def tc_tp_default(name=None):
    """``TC_TP`` (None: ``name`` in ``TC_TP_GPUS``); built only where it applies (``FusedFoldedForward``)."""
    on = TC_TP
    if on is None:
        on = name in TC_TP_GPUS
    return bool(on)


def tc_tp_applies(schedule):
    """Whether ``TC_TP`` can render ``schedule``: every path has at most 8 outputs, of one segment, components 0..d-1
    in order, and each output slot belongs to one path."""
    owners = {}
    for i, path in enumerate(schedule.paths):
        outs = path.outputs
        if not outs or len(outs) > 8 or path.input_irrep_dim > 8:
            return False
        starts = {o.accumulator.start for o in outs}
        if (len(starts) != 1 or [o.accumulator.component for o in outs] != list(range(len(outs)))
                or [o.output_component for o in outs] != list(range(len(outs)))):
            return False
        for o in outs:
            if owners.setdefault(o.accumulator.name, i) != i:
                return False
    return True


def a_columns(fused_id):
    """The columns the radial table should write for kernel ``fused_id``'s a: its padded row stride with ``VEC_A``,
    else -1 (the table's own width)."""
    fused = _REGISTRY.get(int(fused_id))
    return fused.a_stride if fused is not None and fused.vec_a else -1


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


def tc_wfill(npl, stride, presplit=False, wfrag=False):
    """A block's W slice for the tensor-core product: rows padded to KP with zeros, row stride WS; ``stride``: the
    name of W's row stride in the kernel. ``presplit``: stored as float2 {hi, lo} (``TC_PRESPLIT``). ``wfrag``: in
    fragment order (``TC_WFRAG``): element (channel l, column k) at lane (l % 8) * 4 + k % 4, slot (l % 16) / 8 +
    2 ((k % 8) / 4) of tile ((p, k / 8), l / 16)."""
    if wfrag:
        return f"""            for (int i = threadIdx.x; i < {npl} * KP * CB; i += blockDim.x) {{
                const int p = i / (KP * CB), rem = i - p * (KP * CB), k = rem / CB, l = rem - k * CB;
                const int fr = l & 15, fk = k & 7;
                Ws[((p * (KP / 8) + (k >> 3)) * 2 + (l >> 4)) * 128 + ((fr & 7) * 4 + (fk & 3)) * 4 + (fr >> 3)
                   + 2 * (fk >> 2)] = k < K ? W[(int64_t)(wst[p] + cb * CB + l) * {stride} + k] : 0.f;
            }}"""
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


def tc_astage_vec():
    """``tc_astage`` with float4 loads from rows of ``AS`` floats into the edge-major, group-swizzled tile
    (``VEC_A``; K == KP)."""
    return """                for (int i = lane; i < B * (KP / 4); i += 32) {
                    const int b = i / (KP / 4), q = i - b * (KP / 4);
                    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (b < nb) v = __ldg(reinterpret_cast<const float4*>(a + (e0 + b) * AS) + q);
                    *reinterpret_cast<float4*>(&a_s[b * KP + ((q ^ b) << 2)]) = v;
                }"""


def tc_product(npl, presplit=False, vec_a=False, wfrag=False, keep=False):
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
    elif wfrag:  # tc_wfill's fragment order: one float4 per lane
        wload = """                                const float4 wv = reinterpret_cast<const float4*>(Ws)[
                                    ((g * (KP / 8) + (k0 >> 3)) * 2 + m) * 32 + lane];
                                unsigned ah[4], al[4];
                                split_tf32(wv.x, ah[0], al[0]);
                                split_tf32(wv.y, ah[1], al[1]);
                                split_tf32(wv.z, ah[2], al[2]);
                                split_tf32(wv.w, ah[3], al[3]);"""
    else:
        wload = """                                const float* wp = Ws + (g * KP + k0 + tq) * WS + m * 16 + gid;
                                unsigned ah[4], al[4];
                                split_tf32(wp[0], ah[0], al[0]);
                                split_tf32(wp[8], ah[1], al[1]);
                                split_tf32(wp[4 * WS], ah[2], al[2]);
                                split_tf32(wp[4 * WS + 8], ah[3], al[3]);"""
    if vec_a:  # the edge-major, group-swizzled tile of tc_astage_vec
        afrag = """                        split_tf32(a_s[gid * KP + (((k0 >> 2) ^ gid) << 2) + tq], bh[0], bl[0]);
                        split_tf32(a_s[gid * KP + ((((k0 >> 2) + 1) ^ gid) << 2) + tq], bh[1], bl[1]);"""
    else:
        afrag = """                        split_tf32(a_s[(k0 + tq) * B + gid], bh[0], bl[0]);
                        split_tf32(a_s[(k0 + tq + 4) * B + gid], bh[1], bl[1]);"""
    head = f"""                float wacc[{npl}][2][4];
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
{afrag}
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
                }}"""
    if keep:  # TC_TP: w stays in the C fragments
        return head
    return head[:-len("\n                }")] + f"""
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


def staging_floats(k, edge_dim, tc=False, paths_per_part=0, prefetch=False, x_slots=0):
    """Floats of one warp's staging: a [K][BATCH] and sh [BATCH][edge_dim]; with tensor cores a [K padded][BATCH]
    and the redistribution tile [paths per part][BATCH][TC_RSTRIDE] after them (``prefetch``: two a and sh
    buffers, ``PREFETCH``; ``prefetch="x"``: and the x tile, ``x_slots`` values per edge and lane)."""
    xs = BATCH * CHANNEL_BLOCK * int(x_slots) if prefetch == "x" else 0
    if tc:
        return (1 + bool(prefetch)) * BATCH * (_kpad(k) + edge_dim) + paths_per_part * BATCH * TC_RSTRIDE + xs
    return (1 + bool(prefetch)) * BATCH * (k + edge_dim) + xs


def launch_rule(paths, k, edge_dim, channels, shared_optin, sms, tc=False, prefetch=False, x_slots=0):
    """The kernel's parts, warps and grid for ``paths`` uvu paths of ``k``-wide folded inputs on a device with
    ``shared_optin`` bytes of opt-in shared memory per block and ``sms`` SMs (module docstring); ``tc``: the
    tensor-core product's shared layout (``TENSOR_CORES``)."""
    parts = max(1, -(-paths * BATCH // W_TILE_REGISTERS))
    while parts <= paths:
        per = -(-paths // parts)
        if (4 * slice_floats(per, k, tc) + MIN_WARPS * 4 * staging_floats(k, edge_dim, tc, per, prefetch, x_slots)
                <= shared_optin):
            break
        parts += 1
    else:
        raise ValueError(f"fused_folded_forward: no W slice of {k}-wide rows fits {shared_optin} bytes of "
                         "shared memory")
    per = -(-paths // parts)
    parts = -(-paths // per)
    staging = 4 * staging_floats(k, edge_dim, tc, per, prefetch, x_slots)
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


def render(schedule, k, rule, store_w=True, used=None, stream_store=None, tc=False, blocks=None, prefetch=False,
           vec_sh=False, vec_a=False, wfrag=False, tctp=False):
    """CUDA source of ``fused_folded_forward`` for a schedule, the folded input width ``k`` and a launch rule.
    ``store_w=False``: the message alone; ``w`` is neither stored nor built for the padding edges (``wout`` unused;
    plans/capacity.md, the w-rebuild falsifier). ``used``: the leading inputs the k-loop reads (``used_columns``;
    default ``k``), a's and W's rows keep the stride ``k``. ``stream_store`` (default ``STREAM_STORE_W``): ``w``
    stored with ``__stcs``. ``tc``: the a.W product on tensor cores in 3xTF32 (``TENSOR_CORES``, sm_80+; the launch
    rule must be ``launch_rule(..., tc=tc)``'s); ``"presplit"``: with the W slice split once (``TC_PRESPLIT``).
``"fp16"``: exponent-shifted 3xFP16 instead (``TC_FP16_HELPERS``; kernel lab only, used columns padded to 16).
    ``blocks`` (``block_map``'s input; ir_mul, message only): the message written per group, one output
    pointer each (``out0``, ``out1``, ...). ``prefetch`` (sm_80+, not fp16): the staging double-buffered with
    cp.async (``PREFETCH``; the launch rule must be ``launch_rule(..., prefetch=True)``'s). ``vec_sh``: sh staged
    with float4 loads (``VEC_SH``; edge_dim % 4 == 0, sh 16-byte aligned)."""
    if vec_a and (tc not in (True, "presplit") or prefetch or (used if used is not None else k) % 32):
        raise ValueError("fused_folded_forward: vec_a needs the TF32 product without prefetch and used % 32 == 0")
    if wfrag and (tc is not True or prefetch or CHANNEL_BLOCK != 32):
        raise ValueError("fused_folded_forward: wfrag needs the TF32 product (not presplit) without prefetch")
    if tctp and (tc is not True or prefetch or not tc_tp_applies(schedule)):
        raise ValueError("fused_folded_forward: tctp needs the TF32 product (not presplit), no prefetch and "
                         "tc_tp_applies(schedule)")
    if vec_sh and schedule.edge_dim % 4:
        raise ValueError(f"fused_folded_forward: vec_sh needs edge_dim % 4 == 0, not {schedule.edge_dim}")
    check_problem(schedule)
    if prefetch and tc == "fp16":
        raise ValueError("fused_folded_forward: prefetch with the fp16 product is not built")
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
    cases = "\n".join(f"    case {q}: {{{_part(schedule, part, slots, C, layout, store, tc, bmap, prefetch, vec_sh, vec_a, wfrag, tctp)}    }} break;"
                      for q, part in enumerate(parts))
    if tc == "fp16" and _kpad(used) % 16:
        raise ValueError(f"tc='fp16' needs the used columns padded to a multiple of 16, not {_kpad(used)}")
    helpers = (tc_helpers() + TC_FP16_HELPERS) if tc == "fp16" else tc_helpers() if tc else ""
    helpers += CP_ASYNC_HELPERS if prefetch else ""
    if tctp:
        helpers += ("__device__ __forceinline__ void split_tp(float x, unsigned& hi, unsigned& lo) {\n"
                    + _TC_SPLITS[TC_TP_SPLIT] + "\n}\n")
    ka = used if tc else k  # the staged width of a (tensor cores: the used columns)
    x_slots = schedule.input_dim // C
    a_base = slice_floats(npp, ka, tc)
    a_stride = staging_floats(ka, schedule.edge_dim, tc, npp, prefetch, x_slots)
    a_offset = "A_BASE + warp * A_STRIDE"
    return f"""// fused_folded_forward (FastMACE opt-8): folded last layer + receiver-streaming forward, w stored.
using int64_t = long long;
constexpr bool STORE_W = {"true" if store_w else "false"};
constexpr int K = {used};  // the folded inputs the k-loop reads
constexpr int KA = {k};  // W's row stride (and a's, without VEC_A)
constexpr int AS = {-(-k // 4) * 4 if vec_a else k};  // a's row stride
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
{pointers(a_offset, prefetch, x_slots)}
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


CP_ASYNC_HELPERS = """// PREFETCH: 4-byte cp.async global -> shared, zero-filled when bytes = 0 (src then unread).
__device__ __forceinline__ void cp_async4(float* dst, const float* src, int bytes) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\\n" :: "r"(d), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait1() { asm volatile("cp.async.wait_group 1;\\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait0() { asm volatile("cp.async.wait_group 0;\\n" ::: "memory"); }
"""


def pointers(a_offset, prefetch, x_slots=0):
    """The warp's staging pointers: a_s, sh_s and r_s; with ``prefetch`` the two buffers' base ``a_s0`` (a_s and sh_s
    are then set per batch) and SS, one buffer's floats."""
    if prefetch == "x":
        return f"""    constexpr int SS = KP * B + B * ED;  // one staging buffer: a [KP][B], sh [B][ED]
    constexpr int NX = {x_slots};  // x values per edge and lane in the x tile
    float* a_s0 = Ws + {a_offset};
    float* x_s = a_s0 + 2 * SS;  // the warp's x tile [B][NX][32]
    float* r_s = x_s + B * NX * 32;  // the warp's redistribution tile [paths][B][RS]"""
    if prefetch:
        return f"""    constexpr int SS = KP * B + B * ED;  // one staging buffer: a [KP][B], sh [B][ED]
    float* a_s0 = Ws + {a_offset};
    float* r_s = a_s0 + 2 * SS;  // the warp's redistribution tile [paths][B][RS]"""
    return f"""    float* a_s = Ws + {a_offset};
    float* sh_s = a_s + KP * B;
    float* r_s = sh_s + B * ED;  // tensor cores: the warp's redistribution tile [paths][B][RS]"""


def stage_async(first, count, abuf, commit=True):
    """cp.async of the batch at edge ``first`` (``count`` edges) into the buffer at ``abuf``, then a commit."""
    return f"""                {{
                    float* na = {abuf};
                    float* ns = na + KP * B;
                    const int64_t en = {first};
                    const int nn = {count};
                    for (int i = lane; i < B * KP; i += 32) {{
                        const int b = i / KP, k = i - b * KP;
                        const bool ok = b < nn && k < K;
                        cp_async4(&na[k * B + b], ok ? a + (en + b) * KA + k : a, ok ? 4 : 0);
                    }}
                    for (int i = lane; i < B * ED; i += 32) {{
                        const bool ok = i / ED < nn;
                        cp_async4(&ns[i], ok ? sh + en * ED + i : sh, ok ? 4 : 0);
                    }}
                }}""" + ("\n                cp_async_commit();" if commit else "")


def gather_x(first, count, nodes):
    """cp.async of the x rows of the batch at edge ``first`` (``count`` edges; lane b < count holds its sender in
    ``sn``) into the x tile: ``nodes``, each value's offset expression in a row (the lane's channel included)."""
    lines = "\n".join(f"                        cp_async4(&x_s[(b * NX + {i}) * 32 + lane], src + {node}, ok ? 4 : 0);"
                      for i, node in enumerate(nodes))
    return f"""                {{
                    const int nn = {count};
                    #pragma unroll
                    for (int b = 0; b < B; ++b) {{
                        const int s = __shfl_sync(FULL, sn, b);
                        const bool ok = b < nn && !tail && s >= 0 && s < N;
                        const float* src = x + (int64_t)(ok ? s : 0) * IN_DIM;
{lines}
                    }}
                }}"""


def _out(slot, C, layout, bmap):
    if bmap is None:
        return f"out[cur * OUT_DIM + {_node(layout, C, slot.start, slot.component, slot.irrep_dim)}]"
    g, width, base, stride = bmap[slot.start]
    return f"out{g}[cur * {width} + {base + slot.component * stride} + ch]"


def _out_expr(slot, C, layout, bmap, comp, channel):
    """The output address of ``slot``'s segment at component expression ``comp`` and channel expression ``channel``
    (``TC_TP``'s flush)."""
    if bmap is None:
        if layout == "ir_mul":
            return f"out[cur * OUT_DIM + {slot.start} + ({comp}) * {C} + {channel}]"
        return f"out[cur * OUT_DIM + {slot.start} + ({comp}) + ({channel}) * {slot.irrep_dim}]"
    g, width, base, stride = bmap[slot.start]
    return f"out{g}[cur * {width} + {base} + ({comp}) * {stride} + {channel}]"


def _part_tctp(schedule, part, slots, C, layout, store, bmap, vec_sh, vec_a, wfrag):
    """``_part`` with the tensor product on tensor cores (``TC_TP``): see the module constant. One path at a time: its
    a.W product into 8 fragment registers, then its TP over the batch's receiver segments with its own receiver
    cursor (the paths' outputs are independent), so only the TP accumulators stay live across paths."""
    paths = schedule.paths
    npl = len(part)
    wst = ", ".join(str(paths[pi].weight_start) for pi in part)
    xkeys = sorted({(paths[pi].input_start, j, paths[pi].input_irrep_dim) for pi in part
                    for j in range(paths[pi].input_irrep_dim)})
    xi = {key: i for i, key in enumerate(xkeys)}
    nx = len(xkeys)
    nl = "\n"
    wfill = tc_wfill(npl, "KA", False, wfrag)
    astage = tc_astage_vec() if vec_a else tc_astage("KA")
    one = tc_product(1, False, vec_a, wfrag, keep=True)
    if vec_sh:
        shstage = """                for (int i = lane; i < B * ED / 4; i += 32)
                    reinterpret_cast<float4*>(sh_s)[i] = (4 * i / ED < nb)
                        ? __ldg(reinterpret_cast<const float4*>(sh + e0 * ED) + i) : make_float4(0.f, 0.f, 0.f, 0.f);"""
    else:
        shstage = "                for (int i = lane; i < B * ED; i += 32) sh_s[i] = (i / ED < nb) ? sh[e0 * ED + i] : 0.f;"
    xloads = []
    for (st, c, d), i in xi.items():
        for m in range(2):
            for h in range(2):
                chan = f"(cb * CB + {16 * m + 8 * h} + gid)"
                off = f"{st + c * C} + {chan}" if layout == "ir_mul" else f"{st + c} + {chan} * {d}"
                for q in range(2):
                    xloads.append(f"                xt[{q}][{m}][{h}][{i}] = (sq{q} >= 0) ? x[(int64_t)sq{q} * IN_DIM + {off}]"
                                  " : 0.f;")
    wstore = store.replace("w[g][b]", "wacc[0][m][i]").replace("(e0 + b)", "(e0 + eb)").replace(
        "wst[g] + ch]", "wst[GP] + cb * CB + 16 * m + 8 * (i >> 1) + gid]")
    paths_code, flush_all = [], []
    for g, pi in enumerate(part):
        p = paths[pi]
        slot = slots[p.outputs[0].accumulator.name]
        d3 = len(p.outputs)
        outs = []
        for m in range(2):
            for e in range(4):
                comp = f"(2 * tq + {e & 1})"
                chan = f"(cb * CB + {16 * m + 8 * (e >> 1)} + gid)"
                outs.append(f"if (2 * tq + {e & 1} < {d3}) "
                            f"{_out_expr(slot, C, layout, bmap, comp, chan)} = tacc[{g}][{m}][{e}];")
        zero_g = f"for (int m = 0; m < 2; ++m) for (int i = 0; i < 4; ++i) tacc[{g}][m][i] = 0.f;"
        flush_g = " ".join(outs) + " " + zero_g
        flush_all.append(flush_g)
        product = one.replace("(g * KP + k0 + tq)", f"(({g} + g) * KP + k0 + tq)").replace(
            "((g * (KP / 8) + (k0 >> 3))", f"((({g} + g) * (KP / 8) + (k0 >> 3))")
        assert product != one or g == 0
        svals = []
        for j in range(p.input_irrep_dim):
            for q in range(2):
                cases = []
                for o in p.outputs:
                    terms = " + ".join(f"e{q}_{t.edge_component} * {_lit(t.coefficient)}" for t in o.terms
                                       if t.input_component == j)
                    if terms:
                        cases.append((o.output_component, terms))
                expr = "0.f"
                for comp, terms in reversed(cases):
                    expr = f"(gid == {comp} ? ({terms}) : {expr})"
                svals.append((j, q, expr))
        eloads = nl.join(f"                            const float e{q}_{k} = sh_s[(2 * tq + {q}) * ED + {p.edge_start + k}];"
                         for q in range(2) for k in range(p.edge_irrep_dim))
        mmas = []
        for j in range(p.input_irrep_dim):
            i = xi[(p.input_start, j, p.input_irrep_dim)]
            e0x = next(e for jj, q, e in svals if jj == j and q == 0)
            e1x = next(e for jj, q, e in svals if jj == j and q == 1)
            mmas.append(f"""                            {{
                                unsigned bh[2], bl[2];
                                split_tp(ok0 ? {e0x} : 0.f, bh[0], bl[0]);
                                split_tp(ok1 ? {e1x} : 0.f, bh[1], bl[1]);
                                #pragma unroll
                                for (int m = 0; m < 2; ++m) {{
                                    unsigned ah[4], al[4];
                                    split_tp(wacc[0][m][0] * xt[0][m][0][{i}], ah[0], al[0]);
                                    split_tp(wacc[0][m][2] * xt[0][m][1][{i}], ah[1], al[1]);
                                    split_tp(wacc[0][m][1] * xt[1][m][0][{i}], ah[2], al[2]);
                                    split_tp(wacc[0][m][3] * xt[1][m][1][{i}], ah[3], al[3]);
                                    mma_tf32(tacc[{g}][m], al, bh);
                                    mma_tf32(tacc[{g}][m], ah, bl);
                                    mma_tf32(tacc[{g}][m], ah, bh);
                                }}
                            }}""")
        paths_code.append(f"""                {{  // path {pi}
                    constexpr int GP = {g};
{product}
                    if (STORE_W) {{
                        const int wst[{npl}] = {{{wst}}};
                        #pragma unroll
                        for (int m = 0; m < 2; ++m)
                            #pragma unroll
                            for (int i = 0; i < 4; ++i) {{
                                const int eb = 2 * tq + (i & 1);
                                if (eb < nb) {wstore}
                            }}
                    }}
                    if (!tail) {{
                        int64_t cg = cur;
                        int bi = 0;
                        while (bi < nb) {{
                            const int r = __shfl_sync(FULL, rl, bi);
                            while (cg < r) {{ {flush_g.replace("cur * ", "cg * ")} ++cg; }}
                            const bool ok0 = 2 * tq < nb && rq0 == r, ok1 = 2 * tq + 1 < nb && rq1 == r;
{eloads}
{nl.join(mmas)}
                            bi += __popc(__ballot_sync(FULL, lane < nb && rl == r));
                        }}
                    }}
                }}""")
    zero = f"""            #pragma unroll
            for (int g = 0; g < {npl}; ++g)
                #pragma unroll
                for (int m = 0; m < 2; ++m)
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) tacc[g][m][i] = 0.f;"""
    flush = "{ " + " ".join(flush_all) + " ++cur; }"
    return f"""
        {{
            const int wst[{npl}] = {{{wst}}};
{wfill}
        }}
        __syncthreads();
        const int gid = lane >> 2, tq = lane & 3;
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
            float tacc[{npl}][2][4];
{zero}
            int64_t cur = r0;
            for (int64_t e0 = ebeg; e0 < eend; e0 += B) {{
                const int nb = (int)((eend - e0) < B ? (eend - e0) : B);
                __syncwarp();
{astage}
{shstage}
                const int rl = (lane < nb && !tail) ? receivers[e0 + lane] : 0x7fffffff;
                int sl = (lane < nb && !tail) ? senders[e0 + lane] : -1;
                if (sl < 0 || sl >= N) sl = -1;
                const int sq0 = __shfl_sync(FULL, sl, 2 * tq), sq1 = __shfl_sync(FULL, sl, 2 * tq + 1);
                const int rq0 = __shfl_sync(FULL, rl, 2 * tq), rq1 = __shfl_sync(FULL, rl, 2 * tq + 1);
                float xt[2][2][2][{nx}];
{nl.join(xloads)}
                __syncwarp();
{nl.join(paths_code)}
                if (!tail && nb > 0) cur = __shfl_sync(FULL, rl, nb - 1);
            }}
            while (cur < r1) {flush}
        }}
"""


def _part(schedule, part, slots, C, layout, store, tc=False, bmap=None, prefetch=False, vec_sh=False, vec_a=False,
          wfrag=False, tctp=False):
    if tctp:
        return _part_tctp(schedule, part, slots, C, layout, store, bmap, vec_sh, vec_a, wfrag)
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
    before_product = after_product = ""
    if prefetch == "x":
        xloads = "\n".join(f"                    xv[b][{i}] = x_s[(b * NX + {i}) * 32 + lane];" for i in range(nx))
        nodes = [_node(layout, C, s, c, d) for (s, c, d) in xkeys]
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
        wfill, astage, product = (tc_wfill(npl, "KA", pre, wfrag), tc_astage_vec() if vec_a else tc_astage("KA"),
                                  tc_product(npl, pre, vec_a, wfrag))
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
    if vec_sh:
        shstage = """                for (int i = lane; i < B * ED / 4; i += 32)
                    reinterpret_cast<float4*>(sh_s)[i] = (4 * i / ED < nb)
                        ? __ldg(reinterpret_cast<const float4*>(sh + e0 * ED) + i) : make_float4(0.f, 0.f, 0.f, 0.f);"""
    else:
        shstage = "                for (int i = lane; i < B * ED; i += 32) sh_s[i] = (i / ED < nb) ? sh[e0 * ED + i] : 0.f;"
    if prefetch == "x":
        loop_head = f"""            int buf = 0;
            __syncwarp();  // every lane done with the last chunk's buffers
            if (ebeg < eend) {{
                const int64_t ex = ebeg;
{stage_async("ebeg", "(int)((eend - ebeg) < B ? (eend - ebeg) : B)", "a_s0", commit=False)}
                const int sn = (lane < B && ex + lane < eend && !tail) ? senders[ex + lane] : -1;
{gather_x("ex", "(int)((eend - ex) < B ? (eend - ex) : B)", nodes)}
            }}
            cp_async_commit();
            for (int64_t e0 = ebeg; e0 < eend; e0 += B) {{
                const int nb = (int)((eend - e0) < B ? (eend - e0) : B);
                cp_async_wait0();
                __syncwarp();  // the batch's copies of every lane visible; every lane done with the other buffer
                float* a_s = a_s0 + buf * SS;
                float* sh_s = a_s + KP * B;"""
        before_product = f"""                const int64_t ex = e0 + B;
                const int sn = (lane < B && ex + lane < eend && !tail) ? senders[ex + lane] : -1;
                if (ex < eend) {{
{stage_async("ex", "(int)((eend - ex) < B ? (eend - ex) : B)", "a_s0 + (buf ^ 1) * SS", commit=False)}
                }}"""
        after_product = f"""                if (ex < eend) {{
{gather_x("ex", "(int)((eend - ex) < B ? (eend - ex) : B)", nodes)}
                }}
                cp_async_commit();"""
        loop_tail = "                buf ^= 1;\n            }"
    elif prefetch:
        loop_head = f"""            int buf = 0;
            __syncwarp();  // every lane done with the last chunk's buffers
            if (ebeg < eend) {{
{stage_async("ebeg", "(int)((eend - ebeg) < B ? (eend - ebeg) : B)", "a_s0")}
            }}
            for (int64_t e0 = ebeg; e0 < eend; e0 += B) {{
                const int nb = (int)((eend - e0) < B ? (eend - e0) : B);
                __syncwarp();  // every lane done with the buffer the prefetch overwrites
                if (e0 + B < eend) {{
{stage_async("e0 + B", "(int)((eend - e0 - B) < B ? (eend - e0 - B) : B)", "a_s0 + (buf ^ 1) * SS")}
                }} else {{
                    cp_async_commit();  // an empty group: wait_group 1 then always means this batch
                }}
                cp_async_wait1();
                float* a_s = a_s0 + buf * SS;
                float* sh_s = a_s + KP * B;"""
        loop_tail = "                buf ^= 1;\n            }"
    else:
        loop_head = f"""            for (int64_t e0 = ebeg; e0 < eend; e0 += B) {{
                const int nb = (int)((eend - e0) < B ? (eend - e0) : B);
                __syncwarp();
{astage}
{shstage}"""
        loop_tail = "            }"
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
{loop_head}
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
{before_product}
{product}
{after_product}
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
{loop_tail}
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
        self.prefetch = prefetch_default(torch.cuda.get_device_capability(device), torch.cuda.get_device_name(device))
        self.vec_sh = vec_sh_default(schedule.edge_dim, torch.cuda.get_device_name(device)) and not self.prefetch
        self.vec_a = (vec_a_default(torch.cuda.get_device_name(device)) and self.tc in (True, "presplit")
                      and not self.prefetch and self.used % 32 == 0)  # the swizzle stays inside a row
        self.a_stride = -(-self.k // 4) * 4 if self.vec_a else self.k
        self.tctp = (tc_tp_default(torch.cuda.get_device_name(device)) and self.tc is True and not self.prefetch
                     and tc_tp_applies(schedule))
        self.wfrag = (tc_wfrag_default(torch.cuda.get_device_name(device)) and self.tc is True and not self.prefetch
                      and CHANNEL_BLOCK == 32)
        self.rule = launch_rule(len(schedule.paths), self.used if self.tc else self.k, schedule.edge_dim,
                                schedule.channels, int(props.shared_memory_per_block_optin),
                                int(props.multi_processor_count), tc=self.tc, prefetch=self.prefetch,
                                x_slots=schedule.input_dim // schedule.channels)
        self.output_dim, self.weight_dim = schedule.output_dim, kernels.weight_dim
        self.store_w = bool(store_w)
        self.blocks = None if blocks is None else tuple((int(d), tuple(int(s) for s in st)) for d, st in blocks)
        self.source = render(schedule, self.k, self.rule, self.store_w, self.used, tc=self.tc, blocks=self.blocks,
                             prefetch=self.prefetch, vec_sh=self.vec_sh, vec_a=self.vec_a, wfrag=self.wfrag, tctp=self.tctp)
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
        if self.vec_sh and sh.data_ptr() % 16:  # float4 loads need a 16-byte aligned sh
            sh = sh.clone()
        if self.vec_a and (a.shape[1] != self.a_stride or a.stride(0) != self.a_stride or a.data_ptr() % 16):
            a = torch.nn.functional.pad(a[:, :self.k], (0, self.a_stride - self.k)).contiguous()  # rows of AS floats
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

