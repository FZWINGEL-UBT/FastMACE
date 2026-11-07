"""The radial path's r-gradient inside the streaming backward (``wprime_backward``, opt-9 OPT9-T03).

With ``radial_tables`` the folded radial layer's input is a piecewise Chebyshev series in the edge length,
``a(r) = sum_t T_t(x) C_t`` per (species pair, piece), and the per-edge weights are ``w = a [W, b]^T``. Today's
backward (``chunked_conv_backward``) writes the weight gradient ``dw [E, Q]`` in L2-sized chunks, forms ``da =
dw [W, b]`` on a side stream and hands ``da`` to the table's backward, ``dr = <da, a'(r)>``. Within a piece

    dL/dr = sum_c dw_c w'_c(r),   w'(r) = sum_t T_t(x) G'_t,   G' = C' [W, b]^T  ([terms, Q] per (pair, piece)),

so the edge-owned streaming backward can dot each lane's weight gradients with ``w'`` in registers and never
write ``dw``: no chunk stores, no input gradient, no 66-column table backward (``plans/opt-9.md``). ``G'`` is
the derivative half of ``w_table.fold_tables`` without its padding (float32 tables: ``D`` terms).

The kernel is generated from the streaming schedule (``FactorizedComputationSchedule``) with the vendored Jinja
environment. Its body is the vendored ``oeq_projected_backward`` (OpenEquivariance pull request 223, BSD
3-Clause, see NOTICE.md; ``factorized_projected.cuh`` itself stays byte-identical), so ``dx`` and ``dsh`` are the
same sums in the same order; the ``dweights`` store is replaced by the dot. Each edge locates its piece with the
tables' rule (one bin gather; ``x = (max(r, r_lo) - start) * scale - 1``); at or beyond the cutoff (also the
neighbour list's padding) and below ``r_lo`` the slope is zero, as the tables' own, and ``G'`` is not read.

The edge density's gradient still reaches ``r`` through the table's backward on the density column alone
(``density_backward``). Exactness: rounding (the series and the folded product are reassociated); the sender
gradients are added atomically, as in the vendored backward.
"""

from __future__ import annotations

import ctypes

import torch

from fastmace.kernels.gpu_models import GPUModels, lookup

_REGISTRY = {}
# The w-rebuilding backward's launch for recompute_conv_weights: one path part, four edges per batch (the fastest of
# the shapes timed in docs/validation/capacity/capacity-T08-inventory.md section 2).
REBUILD_PARTS = 1
REBUILD_BATCH = 4
# The stored per-edge weights are read once: load them with the streaming cache hint (``__ldcs``, evict-first), so
# they do not displace the derivative table and the receivers' output gradients in L1/L2 (OMAT-0 kernel lab,
# 3 November 2026: the wprime kernel 3.4 % faster on the RTX 5060 Ti). Bitwise. A module constant for the A/B.
STREAM_W = False
# The derivative table's layout for the slope loop. 0: term-major G' [groups, TERMS, Q], one scalar load per term and
# weight. 2 or 4 (float32 tables): q-major [groups, Q, TERMS rounded up to a multiple of SLOPE_VECTOR], each weight's
# terms loaded as float2/float4 (OMAT-0 kernel lab, 3 November 2026: the wprime kernel 8.7 % faster with float2 on
# the RTX 5060 Ti, 2.6 % with float4). The same sums in the same order; the padding terms are zero. The rebuild
# backward (recompute_conv_weights) reads the term-major table. A module constant for the A/B.
SLOPE_VECTOR = 0
# w and its slope from one table (OMAT-0 kernel lab, 4 November 2026). False: the fused forward stores w [E, Q] and the
# backward reads it with the derivative table G'. True (float32 tables): the backward reads the value table
# G = C [W, b]^T instead, q-major with float2 loads ([groups, Q, D + 1 rounded up to even]), and evaluates both
# w = sum_k T_k(x) G_k and w' = sum_k T_k'(x) scale G_k from the one slab, so the forward stores no w (its
# message-only kernel) and the backward reads neither w nor G'. Rounding: the backward's w is the table's series, not
# the forward's a . W product. None: on for the generic family's wprime_backward (OMAT-0, where it was measured) on
# TABLE_W_GPUS (decision 66; owner, 4 November 2026: "Enable it", every A100 model included), off for MH-1;
# True / False: an A/B side's choice on every GPU and model.
TABLE_W = None
# Benchmark_loup (OMAT-0 medium, SrTiO3 5,000 atoms), MD step b/a in both orders: RTX 5060 Ti 0.888 / 0.892, RTX 4000
# Ada 0.928 / 0.931, A100-SXM4-80GB 0.943 / 0.943; force gate passes (results/omat0-kernel-lab-20261103 section 16).
# NVIDIA H200 (5 November 2026, results/h200-defaults-20261105): MD step b/a 1.142 / 1.141 (b = off; OMAT-0 medium,
# SrTiO3 10x10x10, abab / abba, every other A100 switch on).
TABLE_W_GPUS = GPUModels(("NVIDIA GeForce RTX 5060 Ti", "NVIDIA RTX 4000 Ada Generation", "NVIDIA A100-SXM4-80GB",
                          "NVIDIA H200"))
TABLE_W_VECTOR = 2
# The message in linear_1's block layout (OMAT-0 work, 4 November 2026): with TABLE_W and the fused forward, the
# message-only kernel writes each same-irrep group of the grouped BlockLinear's input as its own [N * dim, P * C]
# tensor, the backward reads the groups' gradients through one pointer each, and linear_1 multiplies the groups
# without the relayout copies (RTX 5060 Ti: about 1.15 ms per step). Bitwise the same message. None: on for
# BLOCK_MESSAGE_GPUS (decision 66; owner, 4 November 2026: "Yes, Ada and A100"); True / False: an A/B side's
# choice on every GPU.
BLOCK_MESSAGE = None
# Benchmark_loup MD step b/a (a = the default with TABLE_W) in both orders: RTX 4000 Ada 0.974 / 0.971,
# A100-SXM4-80GB 0.983 / 0.977 (every A100 model, GPUModels), RTX 5060 Ti 0.976 / 0.975 with BLOCK_LDG (1.060 without).
# NVIDIA H200 (5 November 2026, results/h200-defaults-20261105): MD step b/a 1.066 / 1.065 (b = off; OMAT-0 medium,
# SrTiO3 10x10x10, abab / abba, every other A100 switch on).
BLOCK_MESSAGE_GPUS = GPUModels(("NVIDIA GeForce RTX 5060 Ti", "NVIDIA RTX 4000 Ada Generation",
                                "NVIDIA A100-SXM4-80GB", "NVIDIA H200"))
# The block kernel's gradient reads through __ldg (read-only data, written before the launch). The RTX 5060 Ti took
# 9.9 ms with plain loads through the group pointers and 6.6 ms with __ldg, whatever the tensors' placement (the
# kernel alone, Benchmark_loup, 4 November 2026); the RTX 4000 Ada gains nothing (MD step 1.004 / 1.001). With
# FACTORED the RTX 5060 Ti is faster with plain loads (round 2, 5 November 2026: the pack below), so no GPU keeps it.
# None: on for BLOCK_LDG_GPUS (decision 66); True / False: an A/B side's choice on every GPU.
BLOCK_LDG = None
BLOCK_LDG_GPUS = GPUModels(())
# The tensor product's backward factored per path (OMAT-0 TP backward screen, 4 November 2026). False: per
# Clebsch-Gordan term the vendored three updates (weight, input and edge gradient; about 7 floating-point operations).
# True: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i (3 operations per term), then the weight gradient
# sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j. Rounding (the same sums regrouped); every wprime kernel built while
# it is set. A module constant for the A/B: OMAT-0 medium, Benchmark_loup MD step b/a in both orders, RTX 4000 Ada
# 0.983 / 0.980, A100-SXM4-80GB 0.938 / 0.940 (wprime kernel 0.90 / 0.73; docs/validation/omat0/omat0-tp-backward.md).
# None: on for FACTORED_GPUS (owner, 4 November 2026, decision 66; every A100 model, GPUModels); True / False: an A/B
# side's choice on every GPU. RTX 5060 Ti since round 2 (5 November 2026), with plain loads, HORNER and 80 registers:
# MD step b/a SrTiO3 0.979 / 0.982, NbCl5 0.986 / 0.989 (gate 12/12).
FACTORED = None
# NVIDIA H200 (5 November 2026, results/h200-defaults-20261105): MD step b/a 1.160 / 1.159 (b = off; OMAT-0 medium,
# SrTiO3 10x10x10, abab / abba, every other A100 switch on).
FACTORED_GPUS = GPUModels(("NVIDIA RTX 4000 Ada Generation", "NVIDIA A100-SXM4-80GB", "NVIDIA GeForce RTX 5060 Ti",
                           "NVIDIA H200"))
# With FACTORED: each path's most common coefficient magnitude applied once per path (to w and the weight gradient);
# its terms carry only the sign, the others the ratio |c| / alpha (OMAT-0 medium: 50 of 86 terms without a multiply).
# Rounding. Rejected (A100-SXM4-80GB: kernel 0.830 against FACTORED's 0.722, MD step 1.025 / 1.024 against FACTORED);
# kept for the record. A module constant for the A/B.
FACTORED_SCALE = False
# Round 2 (5 November 2026, plans/ideas-20261105-round2.md). HORNER (with TABLE_W): the value table converted to the
# power basis of t once at build (``power_table``, float64), w and its slope by dual Horner instead of the Chebyshev
# basis and its derivative (12 live values per edge). Rounding. SCALAR_PATHS (with FACTORED): l1 = 0 paths add their
# edge gradient per term from w x0, without the v_j sums. Rounding. MAX_REGISTERS: the kernel compiled with
# --maxrregcount (0: no cap). Kernel lab, Benchmark_loup (OMAT-0 medium), the w' kernel against each GPU's default:
# A100-SXM4-80GB HORNER + SCALAR_PATHS 0.940 (HORNER alone 0.997, capped at 80 1.059); RTX 4000 Ada HORNER 0.975
# (+ SCALAR_PATHS 0.987); RTX 5060 Ti FACTORED without BLOCK_LDG 0.929, + HORNER capped at 80 0.887.
# None: on for the *_GPUS sets (decision 66, after the MD-step A/Bs); True / False (an int for MAX_REGISTERS): an A/B
# side's choice on every GPU. MD-step A/Bs b/a (ab / ba), SrTiO3 10x10x10 and NbCl5, OMAT-0 medium, 1 fs:
# A100-SXM4-80GB HORNER + SCALAR_PATHS 0.986 / 0.983 and 0.994 / 0.978 (on); RTX 5060 Ti FACTORED without BLOCK_LDG,
# HORNER and MAX_REGISTERS 80 0.979 / 0.982 and 0.986 / 0.989 (on); RTX 4000 Ada HORNER 1.004 / 1.001 and
# 1.000 / 1.000 (off).
HORNER = None
# NVIDIA H200 (5 November 2026, results/h200-defaults-20261105): MD step b/a 1.046 / 1.046 (b = off; OMAT-0 medium,
# SrTiO3 10x10x10, abab / abba, every other A100 switch on).
HORNER_GPUS = GPUModels(("NVIDIA A100-SXM4-80GB", "NVIDIA GeForce RTX 5060 Ti", "NVIDIA H200"))
SCALAR_PATHS = None
# NVIDIA H200 (5 November 2026, results/h200-defaults-20261105): MD step b/a 1.030 / 1.030 (b = off; OMAT-0 medium,
# SrTiO3 10x10x10, abab / abba, every other A100 switch on).
SCALAR_PATHS_GPUS = GPUModels(("NVIDIA A100-SXM4-80GB", "NVIDIA H200"))
# Idea 39 (plans/h200-ideas-20261105.md, 6 November 2026). ESTRIN (with HORNER, six terms): w and its slope by
# Estrin's scheme (pairs of terms, chains three FMAs deep) instead of dual Horner (five deep). Rounding. None: on for
# ESTRIN_GPUS; True / False: an A/B side's choice.
ESTRIN = None
# NVIDIA A100-SXM4-80GB (6 November 2026, results/ideas-queue-20261106): MD step b/a 0.9961 / 0.9963 (OMAT-0
# medium, SrTiO3 10x10x10, abab / abba); force/stress gate as the default's (worst max-error ratio 1.011, energy).
# Off there all the same: with FOLD_ALPHA it gains nothing (FOLD_ALPHA + ESTRIN vs neither 1.0002 / 1.0028), and
# FOLD_ALPHA alone beats ESTRIN alone (b = FOLD_ALPHA on, ESTRIN off vs ESTRIN on: 0.9967 / 0.9902, 0.9884 / 0.9907).
# NVIDIA H200: 1.0030 / 1.0012, off.
ESTRIN_GPUS = GPUModels(())
# Idea 34 (plans/h200-ideas-20261105.md, 6 November 2026). FOLD_ALPHA (with FACTORED and TABLE_W): FACTORED_SCALE's
# per-path magnitude alpha folded into the w' value table's rows at build (float64), so the kernel's terms carry the
# ratio c / alpha and no per-path multiply is left; scaled paths stay on SCALAR_PATHS. Rounding. None: on for
# FOLD_ALPHA_GPUS; True / False: an A/B side's choice.
FOLD_ALPHA = None
# NVIDIA H200 (6 November 2026, results/ideas-queue-20261106): MD step b/a 0.9963 / 0.9954 (OMAT-0 medium, SrTiO3
# 10x10x10, abab / abba); force/stress gate as the default's (SrTiO3 stress 2.74e-7 vs 2.44e-7, within the limit).
# NVIDIA A100-SXM4-80GB (same day): 0.9911 / 0.9863 and 0.9895 / 0.9893 (two cards); vs ESTRIN 0.9967 / 0.9902,
# 0.9884 / 0.9907;
# gate as the default's (worst max-error ratio 1.049, SrTiO3 forces and stress).
FOLD_ALPHA_GPUS = GPUModels(("NVIDIA H200", "NVIDIA A100-SXM4-80GB"))
MAX_REGISTERS = None
MAX_REGISTERS_GPUS = {"NVIDIA GeForce RTX 5060 Ti": 80}  # device name -> registers
# Round 5 (5 November 2026). BUTTERFLY: the end-of-edge reduction of the EDGE_DIM edge-gradient components as a
# butterfly reduce-scatter. At each step a lane swaps half of its remaining components with its xor partner, so after
# log2(EDGE_DIM) steps lane c << (5 - log2 EDGE_DIM) holds component c and those lanes write dsh coalesced; dr keeps
# its own 5 shuffles. For OMAT-0 (16 components) 21 shuffles per lane and edge instead of 85. Exact: every component
# is summed over the same pairs in the same tree as the shfl_down reduction (a + b == b + a), so dsh and dr are
# bitwise equal. Needs EDGE_DIM a power of two <= 32 and 32-lane cohorts (``butterfly_supported``), else off.
# Kernel lab, Benchmark_loup, the w' kernel against each GPU's default: RTX 5060 Ti 0.965 / 0.963 (no reduction at
# all: 0.675); A100-SXM4-80GB 0.989 (no reduction 0.955).
# None: on for BUTTERFLY_GPUS (decision 66, after the MD-step A/Bs); True / False: an A/B side's choice. MD-step A/Bs
# b/a (ab / ba), SrTiO3 10x10x10 and NbCl5, OMAT-0 medium, 1 fs: RTX 5060 Ti 0.997 / 0.993 and 0.998 / 0.999 (on);
# RTX 4000 Ada 1.008 / 1.007 and 1.006 / 1.000 (off); A100-SXM4-80GB 1.001 / 0.999 and 1.002 / 1.002 (off).
BUTTERFLY = None
# NVIDIA H200: on from 5 November 2026 (results/h200-defaults-20261105, MD step b/a 0.994 / 0.994, b = on), off again
# from 6 November: at d599cc70 (with the later w' defaults) the A/B reversed, b/a 0.9659 / 0.9677 with b = off (kernel
# lab 1,830 us on, 1,612 us off; results/sx012-profile-20261106).
BUTTERFLY_GPUS = GPUModels(("NVIDIA GeForce RTX 5060 Ti",))
# Round 5 (5 November 2026, external idea "integer warp reductions"). REDUX: the end-of-edge sums of the EDGE_DIM
# edge-gradient components and dr in fixed point through the hardware integer reduction (redux.sync.add.s32,
# compute capability 8.0+): per component one redux.sync.max of the lanes' exponents sets a power-of-two scale with
# 2^25 of headroom (32 lanes cannot overflow), the rounded partials are summed as integers, and lane 0 scales back.
# Integer sums are exact and order-free, so dsh and dr are deterministic; each partial is rounded to 2^-25 of its
# component's largest partial in the warp (FP32 class against the shuffle tree's 5 rounded levels): "rounding". A
# NaN or infinity in any partial gives NaN. Float32 only; 32-lane cohorts. Kernel lab, Benchmark_loup, RTX 5060 Ti
# (thermalised capture, the plan's register cap of 80, BUTTERFLY off as the base): 0.966, butterfly 0.984, no reduction
# at all 0.954; dsh 1.4e-7 and dr 2.8e-7 of the max against the shuffle tree. MD-step A/Bs at b809e2bf against each
# GPU's default (SrTiO3 / NbCl5, b/a in both orders): RTX 5060 Ti 0.998 / 0.999, 0.999 / 0.998 (on); RTX 4000 Ada
# 1.009 / 1.008, 1.011 / 1.008 (off).
# None: on for REDUX_GPUS (decision 66, after the MD-step A/Bs); True / False: an A/B side's choice. Takes precedence
# over BUTTERFLY.
REDUX = None
REDUX_GPUS = GPUModels(("NVIDIA GeForce RTX 5060 Ti",))
# TABLE_W's value table interleaved by term pair (H200 ideas plan, idea 1; 5 November 2026): [groups, terms / 2, Q, 2]
# in place of q-major [groups, Q, terms], so a warp's float2 load of one term pair covers 256 contiguous bytes (two
# L1 wavefronts) instead of 768 bytes spread over six lines. The same values in the same registers: bitwise. None: on
# for TABLE_PAIRS_GPUS (decision 66); True / False: an A/B side's choice.
TABLE_PAIRS = None
TABLE_PAIRS_GPUS = GPUModels(())
# HORNER's slope without the piece scale per weight (idea 9): the edge's sum of dweights * dw/dt times the scale once.
# Only dr changes, at rounding level. None: on for HOIST_SCALE_GPUS; True / False: an A/B side's choice.
HOIST_SCALE = None
HOIST_SCALE_GPUS = GPUModels(())
# The receiver's output-gradient row staged in shared memory (H200 ideas plan, idea 38; 6 November 2026,
# results/h200-ideas-20261105/ideas-37-38.md): block b owns edges [b * STAGE_DOUT_CHUNK, (b + 1) * STAGE_DOUT_CHUNK)
# and walks the runs of equal receiver inside them (receiver-sorted edges: runs of about 74 edges on OMAT-0 SrTiO3);
# per run the block copies the receiver's row once, then its warps run the unchanged per-edge body reading dout from
# shared memory. The same values in the same order per edge: dsh and dL/dr bitwise, dx up to the atomics' order.
# H200 lab (descriptor tiles of 32 / 16 edges): 0.875 / 0.883 of the w' kernel. Float32 kernels whose row fits the
# 48 KB of static shared memory (``stage_dout_supported``), else off. None: on for STAGE_DOUT_GPUS; True / False: an
# A/B side's choice. MD-step A/B on the current main (6 November 2026, Modal, SrTiO3 10x10x10): H200 0.9616 / 0.9638,
# A100-SXM4-80GB 0.9896 / 0.9920 (both build orders); bitwise, so no gate.
STAGE_DOUT = None
STAGE_DOUT_GPUS = GPUModels(("NVIDIA H200", "NVIDIA A100-SXM4-80GB"))
STAGE_DOUT_CHUNK = 32  # edges per block (1..32: one ballot finds a run's end)

TEMPLATE = r"""// oeq_projected_backward_wprime (FastMACE opt-9 OPT9-T03), adapted from the vendored
// oeq_projected_backward (OpenEquivariance PR 223, BSD 3-Clause, see NOTICE.md).
using int64_t = signed long long;
using int32_t = int;
using scalar_t = {{ scalar }};
using acc_t = {{ scalar }};

constexpr int INPUT_DIM = {{ schedule.input_dim }};
constexpr int EDGE_DIM = {{ schedule.edge_dim }};
constexpr int OUTPUT_DIM = {{ schedule.output_dim }};
constexpr int WEIGHT_DIM = {{ schedule.weight_numel }};
constexpr int CHANNEL_DIM = {{ schedule.channels }};
constexpr int LOGICAL_GROUP_SIZE =
    {{ schedule.launch_config.logical_cohort_width }};
constexpr int TERMS = {{ terms }};
constexpr int PADDED_TERMS = {{ padded_terms }};  // a q-major G' row: TERMS, then zero terms (SLOPE_VECTOR)
constexpr unsigned int FULL_MASK = 0xffffffffu;

{% macro node_offset(start, component, irrep_dim) -%}
    {%- if layout == "ir_mul" -%}
        {{ start + component * schedule.channels }} + channel
    {%- else -%}
        {{ start + component }} + channel * {{ irrep_dim }}
    {%- endif -%}
{%- endmacro %}

// One logical thread group owns an edge; each lane owns channels separated by LOGICAL_GROUP_SIZE.
// dx and dsh as in oeq_projected_backward; in place of dweights, dr[e] = sum over the edge's weights of
// dweights * w'(r_e), w' from the derivative table gprime [pairs * pieces, TERMS, WEIGHT_DIM] (q-major with
// SLOPE_VECTOR: [pairs * pieces, WEIGHT_DIM, PADDED_TERMS]).
extern "C" __global__ void oeq_projected_backward_wprime(
    int64_t node_count,
    int64_t edge_count,
    const scalar_t* x,
    const scalar_t* sh,
    const scalar_t* weights,
    const int32_t* senders,
    const int32_t* receivers,
{%- if blocks %}
{%- for g in range(n_blocks) %}
    const scalar_t* __restrict__ dout{{ g }},
{%- endfor %}
{%- else %}
    const scalar_t* dout,
{%- endif %}
    scalar_t* dx,
    scalar_t* dsh,
    const scalar_t* lengths,
    const int32_t* pairs,
    const scalar_t* __restrict__ gprime,
    const scalar_t* starts,
    const scalar_t* scales,
    const int32_t* bins,
    scalar_t r_lo,
    scalar_t bin_scale,
    scalar_t r_max,
    int32_t n_bins,
    int32_t pieces,
    scalar_t* dr) {

    const int lane = threadIdx.x % LOGICAL_GROUP_SIZE;
    const int logical_group = threadIdx.x / LOGICAL_GROUP_SIZE;
{%- if stage_dout %}
    // STAGE_DOUT: this block's edges [c0, c1), walked by runs of equal receiver; per run the receiver's dout row is
    // copied to shared memory once, then the warps take the run's edges.
    __shared__ __align__(16) scalar_t stage_row[{{ stage_total }}];
    const int64_t stage_c0 = int64_t(blockIdx.x) * {{ stage_chunk }};
    const int64_t stage_c1 = stage_c0 + {{ stage_chunk }} < edge_count ? stage_c0 + {{ stage_chunk }} : edge_count;
    for (int64_t stage_s = stage_c0; stage_s < stage_c1;) {
    int64_t stage_t1 = stage_c1;
    {
        const int32_t stage_r = receivers[stage_s];
        const int64_t stage_i = stage_s + 1 + lane;
        const unsigned stage_b = __ballot_sync(FULL_MASK, stage_i < stage_c1 && receivers[stage_i] != stage_r);
        if (stage_b) stage_t1 = stage_s + __ffs(stage_b);
        if (stage_r >= 0 && stage_r < node_count) {
            {%- for g, offset, width in stage_rows %}
            {
                const scalar_t* src = {{ "dout" ~ g if blocks else "dout" }} + int64_t(stage_r) * {{ width }};
                if ({{ width % 4 }} == 0 && (reinterpret_cast<unsigned long long>(src) & 15) == 0) {
                    for (int i = threadIdx.x; i < {{ width // 4 }}; i += blockDim.x)
                        reinterpret_cast<float4*>(stage_row + {{ offset }})[i] =
                            __ldg(reinterpret_cast<const float4*>(src) + i);
                } else {
                    for (int i = threadIdx.x; i < {{ width }}; i += blockDim.x)
                        stage_row[{{ offset }} + i] = __ldg(src + i);
                }
            }
            {%- endfor %}
        }
    }
    __syncthreads();
    for (int64_t e = stage_s + logical_group; e < stage_t1; e += blockDim.x / LOGICAL_GROUP_SIZE) {
{%- else %}
    const int64_t e =
        int64_t(blockIdx.x) * (blockDim.x / LOGICAL_GROUP_SIZE) + logical_group;
    if (e >= edge_count) return;
{%- endif %}

    const int32_t sender = senders[e];
    const int32_t receiver = receivers[e];
    if (sender < 0 || sender >= node_count
        || receiver < 0 || receiver >= node_count) {
        // An edge with an out-of-range index contributes nothing: its dsh row and dr are written as zeros (the
        // wrappers allocate dsh with empty_like, so a skipped row would hold garbage; kernel review, 5 November 2026).
        for (int c = lane; c < EDGE_DIM; c += LOGICAL_GROUP_SIZE) dsh[e * EDGE_DIM + c] = scalar_t(0);
        if (lane == 0) dr[e] = scalar_t(0);
        {{ "continue" if stage_dout else "return" }};
    }

    const int64_t input_base = int64_t(sender) * INPUT_DIM;
    const int64_t output_base = int64_t(receiver) * OUTPUT_DIM;
    const int64_t edge_base = e * EDGE_DIM;
    const int64_t weight_base = e * WEIGHT_DIM;

    // The radial tables' piece of r and its Chebyshev argument; no slope at or beyond the cutoff, below r_lo.
    const scalar_t r = lengths[e];
    const bool sloped = r < r_max && r >= r_lo;
    const scalar_t clamped = r > r_lo ? r : r_lo;
    {%- if spline %}
    // SPLINE_W: the cubic spline's interval and t in [0, 1) (rewrites.radial.spline; pieces = its intervals).
{{ spline }}
    const int32_t piece = iv;
    {%- else %}
    int32_t bin = int32_t((clamped - r_lo) * bin_scale);
    bin = bin < n_bins - 1 ? bin : n_bins - 1;
    const int32_t piece = bins[bin];
    const scalar_t t = (clamped - starts[piece]) * scales[piece] - scalar_t(1);
    {%- endif %}
    const scalar_t* g = gprime + (int64_t(pairs[e]) * pieces + piece) * PADDED_TERMS * WEIGHT_DIM;
    {%- if horner %}
    // HORNER: the value table holds power-basis coefficients in t; w and dw/dt by one dual Horner pass per weight.
    const bool inside = r < r_max;
    const scalar_t sc = {{ "sx_inv" if spline else "scales[piece]" }};
    {%- if estrin %}
    // ESTRIN: the powers of t shared by every weight of the edge.
    const scalar_t t2 = t * t, t4 = t2 * t2, two_t = t + t, four_t = two_t + two_t;
    {%- endif %}
    {%- else %}
    scalar_t cheb[PADDED_TERMS];
    cheb[0] = scalar_t(1);
    if (PADDED_TERMS > 1) cheb[1] = t;
    #pragma unroll
    for (int k = 2; k < PADDED_TERMS; ++k) cheb[k] = scalar_t(2) * t * cheb[k - 1] - cheb[k - 2];
    {%- endif %}
    {%- if table_w and not horner %}
    // TABLE_W: w from the value series (zero at or beyond the cutoff) and its r-derivative T_k'(x) * scale,
    // T_k' = k U_(k-1).
    const bool inside = r < r_max;
    scalar_t dcheb[TERMS];
    {
        const scalar_t sc = scales[piece];
        scalar_t um1 = scalar_t(0), u = scalar_t(1);  // U_(k-2), U_(k-1)
        dcheb[0] = scalar_t(0);
        #pragma unroll
        for (int k = 1; k < TERMS; ++k) {
            dcheb[k] = scalar_t(k) * u * sc;
            const scalar_t un = scalar_t(2) * t * u - um1;
            um1 = u;
            u = un;
        }
    }
    {%- endif %}

    acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};
    acc_t radial_gradient = acc_t(0);
    for (int channel = lane; channel < CHANNEL_DIM;
         channel += LOGICAL_GROUP_SIZE) {
        {%- for slot in schedule.input_gradient_slots %}
            acc_t {{ slot.name }} = acc_t(0);
        {%- endfor %}

        {% for path in schedule.paths %}
            {%- set path_index = loop.index0 %}
            {% for edge_channel in range(path.edge_mul) %} {
                const int weight_index =
                    {{ path.weight_start + edge_channel * schedule.channels }} + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                {%- if table_w %}
                scalar_t gv[PADDED_TERMS];
                if (inside) {
                    {%- if table_pairs %}
                    // TABLE_PAIRS: term pair k of weight q at (k * WEIGHT_DIM + q) float2s.
                    const {{ vector_type }}* gq = reinterpret_cast<const {{ vector_type }}*>(g) + weight_index;
                    {%- else %}
                    const {{ vector_type }}* gq =
                        reinterpret_cast<const {{ vector_type }}*>(g + weight_index * PADDED_TERMS);
                    {%- endif %}
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / {{ slope_vector }}; ++k) {
                        const {{ vector_type }} v = __ldg(gq + {{ "k * WEIGHT_DIM" if table_pairs else "k" }});
                        {%- for component in "xyzw"[:slope_vector] %}
                        gv[{{ slope_vector }} * k + {{ loop.index0 }}] = v.{{ component }};
                        {%- endfor %}
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = scalar_t(0);
                }
                {%- if horner and estrin %}
                // ESTRIN: w = (g0 + g1 t) + t^2 (g2 + g3 t) + t^4 (g4 + g5 t);
                // dw/dt = (g1 + 2 g2 t) + t^2 (3 g3 + 4 g4 t) + t^4 (5 g5).
                const scalar_t radial_weight =
                    fmaf(fmaf(fmaf(gv[5], t, gv[4]), t2, fmaf(gv[3], t, gv[2])), t2, fmaf(gv[1], t, gv[0]));
                const scalar_t radial_dt =
                    fmaf(fmaf(gv[5] * scalar_t(5), t2, fmaf(gv[4], four_t, gv[3] * scalar_t(3))), t2,
                         fmaf(gv[2], two_t, gv[1]));
                {%- elif horner %}
                scalar_t radial_weight = gv[TERMS - 1];
                scalar_t radial_dt = scalar_t(0);
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t + radial_weight;
                    radial_weight = radial_weight * t + gv[k];
                }
                {%- else %}
                scalar_t radial_weight = scalar_t(0);
                #pragma unroll
                for (int k = 0; k < TERMS; ++k) radial_weight += cheb[k] * gv[k];
                {%- endif %}
                {%- else %}
                const scalar_t radial_weight = {{ "__ldcs(weights + radial_weight_index)" if stream_w
                                                  else "weights[radial_weight_index]" }};
                {%- endif %}
                acc_t weight_gradient = acc_t(0);
                {%- if factored %}
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                {%- set scalar_path = scalar_paths and path.input_irrep_dim == 1
                                      and (factored[path_index].alpha == 1.0 or fold_alpha) %}
                {%- for i in factored[path_index].inputs %}
                acc_t u_{{ i }} = acc_t(0);
                {%- endfor %}
                {%- if scalar_path %}
                // SCALAR_PATHS: an l1 = 0 path's edge gradient w x0 c g_o added per term, without v_j.
                const scalar_t weighted_input =
                    radial_weight * x[input_base + {{ node_offset(path.input_start, 0, path.input_irrep_dim) }}];
                {%- else %}
                {%- for j in factored[path_index].edges %}
                acc_t v_{{ j }} = acc_t(0);
                {%- endfor %}
                {%- endif %}
                {%- endif %}

                {% for output in path.outputs %} {
                    {%- set slot = output.accumulator %}
                    {%- if blocks and stage_dout %}
                    {%- set bk = blocks[slot.start] %}
                    const scalar_t output_gradient = stage_row[{{ stage_offsets[bk[0]] }}
                        + {{ bk[2] + slot.component * bk[3] }} + channel];
                    {%- elif stage_dout %}
                    const scalar_t output_gradient =
                        stage_row[{{ node_offset(slot.start, slot.component, slot.irrep_dim) }}];
                    {%- elif blocks %}
                    {%- set bk = blocks[slot.start] %}
                    {%- if block_ldg %}
                    const scalar_t output_gradient = __ldg(dout{{ bk[0] }} + int64_t(receiver) * {{ bk[1] }}
                        + {{ bk[2] + slot.component * bk[3] }} + channel);
                    {%- else %}
                    const scalar_t output_gradient = dout{{ bk[0] }}[int64_t(receiver) * {{ bk[1] }}
                        + {{ bk[2] + slot.component * bk[3] }} + channel];
                    {%- endif %}
                    {%- else %}
                    const int64_t output_index =
                        output_base + {{ node_offset(slot.start, slot.component, slot.irrep_dim) }};
                    const scalar_t output_gradient = dout[output_index];
                    {%- endif %}

                    {%- if factored %}
                    {%- for term in output.terms %} {
                        const scalar_t input_value =
                            x[input_base
                              + {{ node_offset(path.input_start, term.input_component, path.input_irrep_dim) }}];
                        const scalar_t edge_value =
                            sh[edge_base
                               + {{ path.edge_start + term.edge_component + edge_channel * path.edge_irrep_dim }}];
                        {%- set fp = factored[path_index] %}
                        {%- set ratio = term.coefficient / fp.alpha %}
                        {%- set op = "-=" if fp.alpha != 1.0 and ratio < 0 else "+=" %}
                        {%- set ratio = ratio | abs if fp.alpha != 1.0 else ratio %}
                        {%- if (ratio - 1.0) | abs < 1e-12 %}
                        const scalar_t scaled = output_gradient;
                        {%- else %}
                        const scalar_t scaled = output_gradient * {{ cpp_scalar_literal(ratio, scalar) }};
                        {%- endif %}
                        u_{{ term.input_component }} {{ op }} scaled * edge_value;
                        {%- if scalar_path %}
                        edge_gradient[{{ path.edge_start + term.edge_component + edge_channel * path.edge_irrep_dim }}]
                            += weighted_input * scaled;
                        {%- else %}
                        v_{{ term.edge_component }} {{ op }} scaled * input_value;
                        {%- endif %}
                    } {%- endfor %}
                    {%- else %}
                    {% for term in output.terms %} {
                        const int64_t input_index = input_base
                            + {{ node_offset(path.input_start, term.input_component, path.input_irrep_dim) }};
                        const int edge_component_index =
                            {{ path.edge_start + term.edge_component }}
                            + {{ edge_channel }} * {{ path.edge_irrep_dim }};

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * {{ cpp_scalar_literal(term.coefficient, scalar) }};
                        {{ term.input_accumulator.name }} += output_gradient
                            * radial_weight * edge_value * {{ cpp_scalar_literal(term.coefficient, scalar) }};
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * {{ cpp_scalar_literal(term.coefficient, scalar) }};
                    } {%- endfor %}
                    {%- endif %}
                } {%- endfor %}
                {%- if factored %}
                {%- set alpha = 1.0 if fold_alpha else factored[path_index].alpha %}
                {%- if alpha != 1.0 %}
                // The path's common coefficient magnitude, applied once (FACTORED_SCALE).
                const scalar_t alpha = {{ cpp_scalar_literal(alpha, scalar) }};
                const scalar_t scaled_weight = radial_weight * alpha;
                {%- else %}
                const scalar_t scaled_weight = radial_weight;
                {%- endif %}
                {%- for i, slot in factored[path_index].inputs.items() %}
                weight_gradient +=
                    x[input_base + {{ node_offset(path.input_start, i, path.input_irrep_dim) }}] * u_{{ i }};
                {{ slot }} += scaled_weight * u_{{ i }};
                {%- endfor %}
                {%- if alpha != 1.0 %}
                weight_gradient *= alpha;
                {%- endif %}
                {%- if not scalar_path %}
                {%- for j in factored[path_index].edges %}
                edge_gradient[{{ path.edge_start + j + edge_channel * path.edge_irrep_dim }}] +=
                    scaled_weight * v_{{ j }};
                {%- endfor %}
                {%- endif %}
                {%- endif %}
                if (sloped) {
                    acc_t slope = acc_t(0);
                    {%- if horner %}
                    slope = {{ "radial_dt" if hoist_scale else "radial_dt * sc" }};
                    {%- elif table_w %}
                    #pragma unroll
                    for (int k = 1; k < TERMS; ++k) slope += dcheb[k] * gv[k];
                    {%- elif slope_vector %}
                    const {{ vector_type }}* gq =
                        reinterpret_cast<const {{ vector_type }}*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / {{ slope_vector }}; ++k) {
                        const {{ vector_type }} v = __ldg(gq + k);
                        {%- for component in "xyzw"[:slope_vector] %}
                        slope += cheb[{{ slope_vector }} * k + {{ loop.index0 }}] * v.{{ component }};
                        {%- endfor %}
                    }
                    {%- else %}
                    #pragma unroll
                    for (int k = 0; k < TERMS; ++k)
                        slope += cheb[k] * __ldg(g + k * WEIGHT_DIM + weight_index);
                    {%- endif %}
                    radial_gradient += weight_gradient * slope;
                }
            } {%- endfor %}
        {%- endfor %}

        {%- for slot in schedule.input_gradient_slots %}
            {{ atomic_add }}(
                dx + input_base + {{ node_offset(slot.start, slot.component, slot.irrep_dim) }},
                scalar_t({{ slot.name }}));
        {%- endfor %}
    }
    {%- if hoist_scale %}
    radial_gradient *= sc;  // HOIST_SCALE: the piece scale once per edge
    {%- endif %}

    {%- if redux %}
    // REDUX: each component's 32 partials summed in fixed point (redux.sync); lane 0 writes dsh[e, :] and dr[e].
    #pragma unroll
    for (int component = 0; component <= EDGE_DIM; ++component) {
        const float v = float(component < EDGE_DIM ? edge_gradient[component] : radial_gradient);
        unsigned ex = __reduce_max_sync(FULL_MASK, __float_as_uint(fabsf(v)) >> 23);
        const bool finite = ex < 255u;
        ex = ex < 30u ? 30u : (finite ? ex : 254u);
        const float scale = __uint_as_float((278u - ex) << 23), inverse = __uint_as_float((ex - 24u) << 23);
        const int total = __reduce_add_sync(FULL_MASK, finite ? __float2int_rn(v * scale) : 0);
        if (lane == 0) {
            const float sum = finite ? float(total) * inverse : __int_as_float(0x7fffffff);
            if (component < EDGE_DIM) dsh[edge_base + component] = scalar_t(sum);
            else dr[e] = scalar_t(sum);
        }
    }
    {%- elif butterfly %}
    // BUTTERFLY: reduce-scatter the edge gradient; lane c << SHIFT writes dsh[e, c], lane 0 dr[e].
    constexpr int SHIFT = {{ 5 - butterfly_steps }};
    #pragma unroll
    for (int step = 0; step < {{ butterfly_steps }}; ++step) {
        const int half = (EDGE_DIM / 2) >> step, offset = 16 >> step;
        const bool upper = (lane & offset) != 0;
        #pragma unroll
        for (int i = 0; i < EDGE_DIM / 2; ++i) {
            if (i < half) {
                const acc_t send = upper ? edge_gradient[i] : edge_gradient[i + half];
                const acc_t keep = upper ? edge_gradient[i + half] : edge_gradient[i];
                edge_gradient[i] = keep + __shfl_xor_sync(FULL_MASK, send, offset, 32);
            }
        }
    }
    #pragma unroll
    for (int offset = (1 << SHIFT) >> 1; offset > 0; offset >>= 1)
        edge_gradient[0] += __shfl_xor_sync(FULL_MASK, edge_gradient[0], offset, 32);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        radial_gradient += __shfl_xor_sync(FULL_MASK, radial_gradient, offset, 32);
    if ((lane & ((1 << SHIFT) - 1)) == 0) dsh[edge_base + (lane >> SHIFT)] = scalar_t(edge_gradient[0]);
    if (lane == 0) dr[e] = scalar_t(radial_gradient);
    {%- else %}
    // Reduce within the logical 32-thread edge owner. Lane 0 writes dsh[e, :] and dr[e].
    for (int offset = 16; offset > 0; offset >>= 1) {
        for (int component = 0; component < EDGE_DIM; ++component)
            edge_gradient[component] +=
                {{ shfl_down_width("edge_gradient[component]", "offset") }};
        radial_gradient += {{ shfl_down_width("radial_gradient", "offset") }};
    }

    if (lane == 0) {
        for (int component = 0; component < EDGE_DIM; ++component)
            dsh[edge_base + component] = scalar_t(edge_gradient[component]);
        dr[e] = scalar_t(radial_gradient);
    }
    {%- endif %}
{%- if stage_dout %}
    }  // STAGE_DOUT: the run's edges
    __syncthreads();  // every warp is done with the row before the next copy
    stage_s = stage_t1;
    }  // STAGE_DOUT: the runs
{%- endif %}
}
"""


def padded_terms(terms, vector):
    """A q-major derivative table's row length: ``terms`` rounded up to a multiple of ``vector`` (0: ``terms``)."""
    return -(-int(terms) // vector) * vector if vector else int(terms)


def factored_paths(schedule, scale=False):
    """Per path of ``schedule`` (``FACTORED``): ``inputs`` {input component: its input-gradient accumulator},
    ``edges`` (the edge components), in first-use order, and ``alpha``: with ``scale`` (``FACTORED_SCALE``) the
    path's most common coefficient magnitude (ties: the largest), which the kernel applies once per path (its terms
    then carry the sign and the ratio |c| / alpha, 1 for most), else 1."""
    result = []
    for path in schedule.paths:
        inputs, edges, magnitudes = {}, {}, {}
        for output in path.outputs:
            for term in output.terms:
                inputs.setdefault(term.input_component, term.input_accumulator.name)
                edges.setdefault(term.edge_component, None)
                key = round(abs(float(term.coefficient)), 12)
                magnitudes[key] = magnitudes.get(key, 0) + 1
        alpha = 1.0
        if scale and magnitudes:
            alpha = max(magnitudes, key=lambda m: (magnitudes[m], m))
            alpha = 1.0 if abs(alpha - 1.0) < 1e-12 else alpha
        result.append(dict(inputs=inputs, edges=tuple(edges), alpha=float(alpha)))
    return result


def render(schedule, scalar, terms, stream_w=None, vector=0, table_w=False, blocks=None, block_ldg=False,
           factored=False, scale=False, horner=False, scalar_paths=False, butterfly=False, redux=False,
           table_pairs=False, hoist_scale=False, stage_dout=False, estrin=False, fold_alpha=False, spline=None):
    """CUDA source of ``oeq_projected_backward_wprime`` for a schedule, ``scalar`` ("float" or "double") and the
    derivative table's ``terms``; ``stream_w`` (default ``STREAM_W``): the stored weights loaded with ``__ldcs``;
    ``vector`` 2 or 4: the q-major table's terms loaded as float2/float4 (``SLOPE_VECTOR``; float only).
    ``table_w`` (``TABLE_W``): the table is the q-major value table (``value_table``, ``terms`` = D + 1), from which
    the kernel evaluates w and its slope; the stored weights are not read (needs ``vector``). ``blocks``
    (``fused_forward.block_map``'s input; ir_mul): the message gradient read per group, one pointer each.
    ``factored`` (``FACTORED``): each path's three trilinear sums regrouped (rounding). Kernel-lab options
    (round 2, 5 November 2026): ``horner`` (with ``table_w``): the value table in the power basis of t
    (``power_table``), w and its slope by dual Horner; ``scalar_paths`` (with ``factored``): l1 = 0 paths add
    their edge gradient per term from w x0, without the v_j sums. ``butterfly`` (``BUTTERFLY``, round 5): the
    end-of-edge reduction as a butterfly reduce-scatter (exact; ``butterfly_supported``). ``table_pairs``
    (``TABLE_PAIRS``, with ``table_w`` and vector 2): the value table interleaved by term pair (``pair_table``).
    ``hoist_scale`` (``HOIST_SCALE``, with ``horner``): the piece scale applied once per edge. ``stage_dout``
    (``STAGE_DOUT``; ``stage_dout_supported``): a block per ``STAGE_DOUT_CHUNK`` edges, each receiver's dout row in
    shared memory (launch ``ceil(edges / STAGE_DOUT_CHUNK)`` blocks). ``spline`` (``SPLINE_W``): the spline's grid
    ``[(start, end, intervals)]``; the table is ``spline_table``'s (4 power-basis terms per interval, HORNER)."""
    from fastmace.kernels.oeq_streaming.fused_forward import block_map

    if blocks is not None and schedule.layout != "ir_mul":
        raise ValueError("a block-layout message needs the ir_mul layout")
    from fastmace.kernels.oeq_streaming.jinja_env import get_jinja_environment

    if schedule.layout not in ("ir_mul", "mul_ir"):
        raise ValueError(f"unknown layout {schedule.layout!r}")
    if int(terms) < 1:
        raise ValueError("the derivative table needs at least one term")
    if vector not in (0, 2, 4) or (vector and scalar != "float"):
        raise ValueError(f"slope vector {vector!r}: 0, or 2 or 4 for float kernels")
    if horner and not table_w:
        raise ValueError("horner evaluates the value table: it needs table_w")
    if scalar_paths and not factored:
        raise ValueError("scalar_paths regroups the factored sums: it needs factored")
    if table_w and not vector:
        raise ValueError("table_w reads the q-major value table: it needs a slope vector of 2 or 4")
    if butterfly and not butterfly_supported(schedule):
        raise ValueError("butterfly needs a power-of-two edge dimension <= 32 and 32-lane cohorts")
    if redux and not redux_supported(schedule, scalar):
        raise ValueError("redux needs float32 and 32-lane cohorts")
    if table_pairs and not (table_w and vector == 2):
        raise ValueError("table_pairs interleaves TABLE_W's value table by float2: it needs table_w and vector 2")
    if hoist_scale and not horner:
        raise ValueError("hoist_scale moves HORNER's piece scale: it needs horner")
    if spline is not None and not (horner and int(terms) == 4 and not table_pairs and not estrin):
        raise ValueError("spline evaluates the spline's four power-basis terms by HORNER: it needs horner and terms 4, "
                         "without table_pairs or estrin")
    if fold_alpha and not (factored and scale and table_w):
        raise ValueError("fold_alpha folds FACTORED_SCALE's alpha into the value table: it needs factored, scale and "
                         "table_w")
    if estrin and not (horner and int(terms) == 6 and scalar == "float"):
        raise ValueError("estrin evaluates HORNER's six-term power basis in float32: it needs horner and terms 6")
    if stage_dout and not stage_dout_supported(schedule, scalar, blocks):
        raise ValueError("stage_dout needs float32, 32-lane cohorts, a chunk of 1..32 edges and a dout row within "
                         "48 KB")
    stage_rows, stage_offsets, stage_total = _stage_layout(schedule, blocks)
    env = get_jinja_environment(backend="cuda", warp_size=schedule.launch_config.logical_cohort_width)
    return env.from_string(TEMPLATE).render(scalar=scalar, schedule=schedule, layout=schedule.layout,
                                            terms=int(terms), stream_w=STREAM_W if stream_w is None else stream_w,
                                            slope_vector=vector, vector_type=f"float{vector}",
                                            padded_terms=padded_terms(terms, vector), table_w=bool(table_w),
                                            blocks=None if blocks is None else block_map(blocks, schedule.channels),
                                            n_blocks=0 if blocks is None else len(blocks), block_ldg=block_ldg,
                                            factored=factored_paths(schedule, scale) if factored else None,
                                            horner=bool(horner), scalar_paths=bool(scalar_paths),
                                            butterfly=bool(butterfly), redux=bool(redux),
                                            table_pairs=bool(table_pairs), hoist_scale=bool(hoist_scale),
                                            stage_dout=bool(stage_dout), stage_chunk=int(STAGE_DOUT_CHUNK),
                                            estrin=bool(estrin), fold_alpha=bool(fold_alpha),
                                            stage_rows=stage_rows, stage_offsets=stage_offsets,
                                            stage_total=stage_total,
                                            butterfly_steps=int(schedule.edge_dim).bit_length() - 1,
                                            spline=None if spline is None else _spline_lookup(spline, scalar))


def _spline_lookup(grid, scalar):
    from fastmace.kernels.oeq_streaming.spline_forward import lookup_code

    return lookup_code(grid, "clamped", "scalar_t" if scalar != "float" else "float", "    ")


def _stage_layout(schedule, blocks):
    """STAGE_DOUT's shared row: ([(group, offset, width)], {group: offset}, total floats). Block layout: each group's
    receiver block (``fused_forward.block_map``'s row width) in group order; otherwise the dense row, one entry."""
    from fastmace.kernels.oeq_streaming.fused_forward import block_map

    if blocks is None:
        width = int(schedule.output_dim)
        return [(0, 0, width)], {0: 0}, width
    widths = {}
    for g, width, _, _ in block_map(blocks, schedule.channels).values():
        widths[int(g)] = int(width)
    rows, offsets, total = [], {}, 0
    for g in sorted(widths):
        rows.append((g, total, widths[g]))
        offsets[g] = total
        total += widths[g]
    return rows, offsets, total


def stage_dout_supported(schedule, scalar, blocks=None):
    """Whether ``STAGE_DOUT`` applies: float32, 32-lane cohorts, ``STAGE_DOUT_CHUNK`` in 1..32 and the receiver's
    dout row within 48 KB of static shared memory."""
    return (scalar == "float" and int(schedule.launch_config.logical_cohort_width) == 32
            and 1 <= int(STAGE_DOUT_CHUNK) <= 32 and _stage_layout(schedule, blocks)[2] * 4 <= 48 * 1024)


def redux_supported(schedule, scalar):
    """Whether ``REDUX`` applies: float32 partials, 32-lane cohorts (redux.sync needs compute capability 8.0+,
    FastMACE's floor for these kernels)."""
    return scalar == "float" and int(schedule.launch_config.logical_cohort_width) == 32


def butterfly_supported(schedule):
    """``BUTTERFLY`` applies: the edge dimension a power of two <= 32 and 32-lane cohorts."""
    d = int(schedule.edge_dim)
    return 0 < d <= 32 and d & (d - 1) == 0 and int(schedule.launch_config.logical_cohort_width) == 32


def power_table(g, terms):
    """A q-major value table ``g [groups, Q, padded]`` whose first ``terms`` entries are Chebyshev coefficients in
    t, converted to power-basis coefficients in t (float64, stored in ``g``'s dtype; the padding stays zero)."""
    import numpy as np

    basis = np.zeros((terms, terms))  # column k: T_k in the power basis
    for k in range(terms):
        unit = np.zeros(terms)
        unit[k] = 1.0
        power = np.polynomial.chebyshev.cheb2poly(unit)  # trailing zeros trimmed
        basis[: power.size, k] = power
    m = torch.as_tensor(basis, dtype=torch.float64, device=g.device)
    out = g.clone()
    out[..., :terms] = (g[..., :terms].to(torch.float64) @ m.T).to(g.dtype)
    return out.contiguous()


def derivative_table(dcoef, weight, vector=0):
    """``G' [pairs * pieces, D', Q]`` from a table's ``dcoef [P, K, D', J]`` and the folded weight ``[Q, J - 1]``
    (the table's last column is the edge density): ``w_table.fold_tables``' derivative half without its padding
    term and zero group. Computed in float64, stored in ``dcoef``'s dtype. ``vector`` 2 or 4 (``SLOPE_VECTOR``):
    q-major, ``[pairs * pieces, Q, D']`` with zero terms up to a multiple of ``vector``, the same values."""
    pairs, pieces, terms, _ = dcoef.shape
    width = weight.shape[1]
    d = torch.einsum("pktj,qj->pktq", dcoef[..., :width].to(torch.float64),
                     weight.detach().to(device=dcoef.device, dtype=torch.float64))
    d = d.reshape(pairs * pieces, terms, -1)
    if vector:
        d = torch.nn.functional.pad(d.transpose(1, 2), (0, padded_terms(terms, vector) - terms))
    return d.to(dcoef.dtype).contiguous()


def value_table(coef, weight, vector=TABLE_W_VECTOR, row_scale=None):
    """``G [pairs * pieces, Q, D + 1 rounded up to a multiple of vector]`` (``TABLE_W``) from a table's ``coef [P, K,
    D + 1, J]`` and the folded weight ``[Q, J - 1]``: ``w_table.fold_tables``' value half without its zero group,
    q-major with zero padding terms. Computed in float64, stored in ``coef``'s dtype. ``row_scale [Q]`` (float64,
    FOLD_ALPHA): each weight's row multiplied by it before rounding."""
    pairs, pieces, terms, _ = coef.shape
    width = weight.shape[1]
    w64 = weight.detach().to(device=coef.device, dtype=torch.float64)
    if row_scale is not None:
        w64 = w64 * torch.as_tensor(row_scale, dtype=torch.float64, device=coef.device)[:, None]
    g = torch.einsum("pktj,qj->pkqt", coef[..., :width].to(torch.float64), w64)
    g = torch.nn.functional.pad(g.reshape(pairs * pieces, -1, terms), (0, padded_terms(terms, vector) - terms))
    return g.to(coef.dtype).contiguous()


def pair_table(g):
    """A q-major value table ``g [groups, Q, padded]`` interleaved by term pair (``TABLE_PAIRS``): ``[groups,
    padded / 2, Q, 2]``, the same values."""
    groups, q, padded = g.shape
    return g.reshape(groups, q, padded // 2, 2).permute(0, 2, 1, 3).contiguous()


def density_column(dcoef):
    """The derivative series of the edge density alone, ``[P, K, D', 1]`` (``density_backward``)."""
    return dcoef[..., -1:].contiguous()


def density_backward(r, pair, density_dcoef, starts, scales, bins, grad_density, r_lo, bin_scale, r_max):
    """``dL/dr [E]`` of the edge density alone: the table's backward on its density column, with an ``[E, 0]``
    folded-input gradient (a view of ``grad_density``)."""
    from fastmace.kernels.triton.radial_table import radial_table_backward

    grad_density = grad_density.reshape(-1, 1).contiguous()
    return radial_table_backward(r, pair, density_dcoef, starts, scales, bins, grad_density[:, :0], grad_density,
                                 float(r_lo), float(bin_scale), float(r_max)).reshape(-1)


def wprime_arguments(r, pair, starts, scales, bins, r_lo, bin_scale, r_max):
    """Per edge the Chebyshev argument ``x`` and the ``G'`` group ``pair * pieces + piece`` (int32), ``-1`` where the
    slope is zero (at or beyond the cutoff, below ``r_lo``): the tables' rule, as the kernel computes it."""
    r = r.reshape(-1).to(starts.dtype)
    clamped = torch.clamp(r, min=r_lo)
    index = torch.clamp(((clamped - r_lo) * bin_scale).to(torch.int64), 0, bins.shape[0] - 1)
    piece = bins[index].to(torch.int64)
    x = (clamped - starts[piece]) * scales[piece] - 1.0
    sloped = (r < r_max) & (r >= r_lo)
    group = torch.where(sloped, pair.reshape(-1).to(torch.int64) * starts.shape[0] + piece,
                        torch.full_like(piece, -1))
    return x.contiguous(), group.to(torch.int32).contiguous()


def wprime_reference(dweights, r, pair, gprime, starts, scales, bins, r_lo, bin_scale, r_max):
    """PyTorch ``dL/dr [E]`` of the per-edge weights' gradient ``dweights [E, Q]`` (CPU tests and checks)."""
    x, group = wprime_arguments(r, pair, starts, scales, bins, r_lo, bin_scale, r_max)
    terms = gprime.shape[1]
    cheb = [torch.ones_like(x), x]
    for _ in range(2, terms):
        cheb.append(2.0 * x * cheb[-1] - cheb[-2])
    basis = torch.stack(cheb[:terms], dim=1)  # [E, terms]
    slopes = torch.einsum("et,etq->eq", basis, gprime[group.clamp(min=0).to(torch.int64)])
    dr = (dweights.to(slopes.dtype) * slopes).sum(dim=1)
    return torch.where(group >= 0, dr, torch.zeros_like(dr))


class WPrimeBackward:
    """Compiled ``oeq_projected_backward_wprime`` for the streaming kernels of one interaction
    (``StreamingKernels``) and a derivative table of ``terms`` terms (``vector``: its q-major layout's load width,
    ``SLOPE_VECTOR``)."""

    def __init__(self, kernels, terms, device=None, vector=0, table_w=False, blocks=None, factored=None,
                 horner=False, table_pairs=False, fold_alpha=False, spline=None):
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check, _device_key, compile_cubin, content_id
        scalar = "float" if "using scalar_t = float;" in kernels.source else "double"
        device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
        self.terms, self.vector = int(terms), int(vector)
        self.scalar = scalar
        self.num_threads, self.cohort, self.shared_memory = kernels.num_threads, kernels.cohort, kernels.shared_memory
        self.output_dim, self.channels = kernels.output_dim, kernels.schedule.channels
        self.table_w = bool(table_w)
        self.blocks = blocks
        self.schedule = kernels.schedule
        self.factored = factored_default(device) if factored is None else bool(factored)
        self.fold_alpha = bool(fold_alpha)  # the caller's table carries each path's alpha (FOLD_ALPHA)
        if self.fold_alpha and not (self.factored and self.table_w):
            raise ValueError("FOLD_ALPHA needs FACTORED and TABLE_W")
        self.horner = bool(horner)  # the caller's table is power_table's (HORNER)
        self.table_pairs = bool(table_pairs)  # the caller's table is pair_table's (TABLE_PAIRS)
        self.hoist_scale = self.horner and hoist_scale_default(device)
        self.scalar_paths = self.factored and scalar_paths_default(device)
        self.max_registers = max_registers_default(device)
        self.redux = redux_default(device) and redux_supported(kernels.schedule, scalar)
        self.butterfly = not self.redux and butterfly_default(device) and butterfly_supported(kernels.schedule)
        self.stage_dout = stage_dout_default(device) and stage_dout_supported(kernels.schedule, scalar, blocks)
        self.stage_chunk = int(STAGE_DOUT_CHUNK)
        self.estrin = self.horner and self.terms == 6 and scalar == "float" and estrin_default(device)
        self.spline = None if spline is None else [tuple(g) for g in spline]  # SPLINE_W: the spline's grid
        self.source = render(kernels.schedule, scalar, self.terms, vector=self.vector, table_w=self.table_w,
                             blocks=blocks, block_ldg=block_ldg_default(device), factored=self.factored,
                             scale=self.factored and (FACTORED_SCALE or self.fold_alpha), horner=self.horner,
                             scalar_paths=self.scalar_paths, butterfly=self.butterfly,
                             redux=self.redux, table_pairs=self.table_pairs, hoist_scale=self.hoist_scale,
                             stage_dout=self.stage_dout, estrin=self.estrin, fold_alpha=self.fold_alpha,
                             spline=self.spline)
        options = ()
        if self.max_registers:
            options = (f"--maxrregcount={self.max_registers}",)
            self.source += f"\n// compiled with --maxrregcount={self.max_registers}\n"  # in the content id
        cubin = compile_cubin(self.source, torch.cuda.get_device_capability(device), options)
        torch.cuda.current_stream(device)
        self._module = _check(driver.cuModuleLoadData(cubin))
        self._function = _check(driver.cuModuleGetFunction(self._module, b"oeq_projected_backward_wprime"))
        self.id = content_id("WPrimeBackward", self.source, _device_key(device))
        _REGISTRY[self.id] = self

    def __call__(self, x, sh, weights, senders, receivers, dout, dx, dsh, r, pair, gprime, starts, scales, bins,
                 r_lo, bin_scale, r_max, dr):
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check
        nodes, edges = x.shape[0], sh.shape[0]
        blocks = (edges * self.cohort + self.num_threads - 1) // self.num_threads
        if self.stage_dout:  # STAGE_DOUT: a block per stage_chunk edges
            blocks = (edges + self.stage_chunk - 1) // self.stage_chunk
        if not blocks:
            return
        p = ctypes.c_void_p
        real = ctypes.c_float if self.scalar == "float" else ctypes.c_double
        values = (nodes, edges, x.data_ptr(), sh.data_ptr(), weights.data_ptr(), senders.data_ptr(),
                  receivers.data_ptr(), *([d.data_ptr() for d in dout] if self.blocks else [dout.data_ptr()]),
                  dx.data_ptr(), dsh.data_ptr(), r.data_ptr(),
                  pair.data_ptr(), gprime.data_ptr(), starts.data_ptr(), scales.data_ptr(), bins.data_ptr(),
                  r_lo, bin_scale, r_max, bins.shape[0], starts.shape[0], dr.data_ptr())
        douts = len(self.blocks) if self.blocks else 1
        types = ((ctypes.c_longlong, ctypes.c_longlong) + (p,) * (13 + douts) + (real, real, real)
                 + (ctypes.c_int, ctypes.c_int, p))
        stream = driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        _check(driver.cuLaunchKernel(self._function, blocks, 1, 1, self.num_threads, 1, 1, self.shared_memory,
                                     stream, (values, types), 0))


@torch.library.custom_op("fastmace::streaming_conv_wprime_backward", mutates_args=())
def streaming_conv_wprime_backward(x: torch.Tensor, sh: torch.Tensor, weights: torch.Tensor, senders: torch.Tensor,
                                   receivers: torch.Tensor, dout: torch.Tensor, r: torch.Tensor, pair: torch.Tensor,
                                   gprime: torch.Tensor, starts: torch.Tensor, scales: torch.Tensor,
                                   bins: torch.Tensor, r_lo: float, bin_scale: float, r_max: float,
                                   wprime_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dx, dsh, dL/dr [E] of the radial path) through the kernel ``wprime_id``; one launch over all edges."""
    kernel = _REGISTRY[wprime_id]
    x, sh, weights, dout = x.contiguous(), sh.contiguous(), weights.contiguous(), dout.contiguous()
    dx = torch.zeros_like(x)  # sender gradients are added atomically
    dsh = torch.empty_like(sh)
    dr = sh.new_empty((sh.shape[0],))
    kernel(x, sh, weights, senders.contiguous(), receivers.contiguous(), dout, dx, dsh,
           r.reshape(-1).contiguous(), pair.contiguous(), gprime, starts, scales, bins, r_lo, bin_scale, r_max, dr)
    return dx, dsh, dr


@streaming_conv_wprime_backward.register_fake
def _(x, sh, weights, senders, receivers, dout, r, pair, gprime, starts, scales, bins, r_lo, bin_scale, r_max,
      wprime_id):
    return torch.empty_like(x), torch.empty_like(sh), sh.new_empty((sh.shape[0],))


@torch.library.custom_op("fastmace::streaming_conv_wprime_backward_recompute", mutates_args=())
def streaming_conv_wprime_backward_recompute(x: torch.Tensor, sh: torch.Tensor, senders: torch.Tensor,
                                             receivers: torch.Tensor, dout: torch.Tensor, r: torch.Tensor,
                                             pair: torch.Tensor, gprime: torch.Tensor, coef: torch.Tensor,
                                             weight: torch.Tensor, starts: torch.Tensor, scales: torch.Tensor,
                                             bins: torch.Tensor, r_lo: float, bin_scale: float, r_max: float,
                                             wprime_id: int, rows: int) -> tuple[torch.Tensor, torch.Tensor,
                                                                                 torch.Tensor]:
    """``streaming_conv_wprime_backward`` without the stored per-edge weights (``recompute_conv_weights``,
    plans/capacity.md CAP-T07): the edges run in chunks of ``rows``, and each chunk's weights are built again from
    its edge lengths (the radial table, then ``a @ weight.T``, as the forward's chain builds them) into a
    chunk-sized buffer just before the kernel reads them. Per edge the kernel computes what the one-launch backward
    does; only the order of the atomic sender additions differs, as between any two runs."""
    from fastmace.kernels.triton.radial_table import radial_table_forward

    kernel = _REGISTRY[wprime_id]
    x, sh, dout = x.contiguous(), sh.contiguous(), dout.contiguous()
    senders, receivers = senders.contiguous(), receivers.contiguous()
    r, pair = r.reshape(-1).contiguous(), pair.contiguous()
    edges = sh.shape[0]
    dx = torch.zeros_like(x)  # sender gradients are added atomically, over all chunks
    dsh = torch.empty_like(sh)
    dr = sh.new_empty((edges,))
    rows = max(1, min(int(rows), edges))
    buffer = sh.new_empty((rows, weight.shape[0]))
    for start in range(0, edges, rows):
        stop = min(edges, start + rows)
        a, _ = radial_table_forward(r[start:stop], pair[start:stop], coef, starts, scales, bins, r_lo, bin_scale,
                                    r_max)
        weights = torch.matmul(a, weight.t(), out=buffer[:stop - start])
        kernel(x, sh[start:stop], weights, senders[start:stop], receivers[start:stop], dout, dx, dsh[start:stop],
               r[start:stop], pair[start:stop], gprime, starts, scales, bins, r_lo, bin_scale, r_max,
               dr[start:stop])
    return dx, dsh, dr


@streaming_conv_wprime_backward_recompute.register_fake
def _(x, sh, senders, receivers, dout, r, pair, gprime, coef, weight, starts, scales, bins, r_lo, bin_scale, r_max,
      wprime_id, rows):
    return torch.empty_like(x), torch.empty_like(sh), sh.new_empty((sh.shape[0],))


@torch.library.custom_op("fastmace::streaming_conv_wprime_backward_blocks", mutates_args=())
def streaming_conv_wprime_backward_blocks(x: torch.Tensor, sh: torch.Tensor, senders: torch.Tensor,
                                          receivers: torch.Tensor, douts: list[torch.Tensor], r: torch.Tensor,
                                          pair: torch.Tensor, gprime: torch.Tensor, starts: torch.Tensor,
                                          scales: torch.Tensor, bins: torch.Tensor, r_lo: float, bin_scale: float,
                                          r_max: float, wprime_id: int) -> tuple[torch.Tensor, torch.Tensor,
                                                                                 torch.Tensor]:
    """``streaming_conv_wprime_backward`` of a ``TABLE_W`` kernel built with ``blocks``: the message gradient as
    one tensor per group (``BLOCK_MESSAGE``)."""
    kernel = _REGISTRY[wprime_id]
    x, sh = x.contiguous(), sh.contiguous()
    douts = [d.contiguous() for d in douts]
    dx = torch.zeros_like(x)  # sender gradients are added atomically
    dsh = torch.empty_like(sh)
    dr = sh.new_empty((sh.shape[0],))
    kernel(x, sh, sh.new_empty((0,)), senders.contiguous(), receivers.contiguous(), douts, dx, dsh,
           r.reshape(-1).contiguous(), pair.contiguous(), gprime, starts, scales, bins, r_lo, bin_scale, r_max, dr)
    return dx, dsh, dr


@streaming_conv_wprime_backward_blocks.register_fake
def _(x, sh, senders, receivers, douts, r, pair, gprime, starts, scales, bins, r_lo, bin_scale, r_max, wprime_id):
    return torch.empty_like(x), torch.empty_like(sh), sh.new_empty((sh.shape[0],))


class _WPrimeBlockConvolution(torch.autograd.Function):
    """``_WPrimeConvolution``'s TABLE_W path with the fused message-only forward, the message in the block layout:
    returns the groups, then the edge density."""

    @staticmethod
    def forward(ctx, r, pair, coef, density_dcoef, starts, scales, bins, r_lo, bin_scale, r_max, weight, gprime, x,
                sh, senders, receivers, row_ptr, message_id, wprime_id):
        from fastmace.kernels.oeq_streaming.fused_forward import a_columns, fused_folded_message_blocks
        from fastmace.kernels.triton.radial_table import radial_table_forward

        a, density = radial_table_forward(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max, True,
                                          a_columns(message_id))  # VEC_A: rows padded for float4 staging
        groups = fused_folded_message_blocks(a, weight, x, sh, senders, receivers, row_ptr, message_id)
        ctx.save_for_backward(x, sh, senders, receivers, r, pair, gprime, density_dcoef, starts, scales, bins)
        ctx.constants, ctx.wprime_id = (r_lo, bin_scale, r_max), wprime_id
        return (*groups, density)

    @staticmethod
    def backward(ctx, *grads):
        x, sh, senders, receivers, r, pair, gprime, density_dcoef, starts, scales, bins = ctx.saved_tensors
        kernel = _REGISTRY[ctx.wprime_id]
        from fastmace.kernels.oeq_streaming.fused_forward import block_shapes
        shapes = block_shapes(kernel.blocks, kernel.channels, x.shape[0])
        douts = [x.new_zeros(shape) if g is None else g for g, shape in zip(grads[:-1], shapes)]
        dx, dsh, dr = streaming_conv_wprime_backward_blocks(x, sh, senders, receivers, douts, r, pair, gprime,
                                                            starts, scales, bins, *ctx.constants, ctx.wprime_id)
        if grads[-1] is not None:
            dr = dr + density_backward(r, pair, density_dcoef, starts, scales, bins, grads[-1], *ctx.constants)
        return (dr.reshape(r.shape), None, None, None, None, None, None, None, None, None, None, None, dx, dsh,
                None, None, None, None, None)


class _SplineBlockConvolution(torch.autograd.Function):
    """SPLINE_W: ``_WPrimeBlockConvolution`` with the message from ``spline_forward`` (w from the cubic spline, no
    a.W product) and the edge density alone from the radial table (its density column, ``columns=0``); the w'
    backward reads the same spline table. Returns the groups, then the edge density."""

    @staticmethod
    def forward(ctx, r, pair, density_coef, density_dcoef, starts, scales, bins, r_lo, bin_scale, r_max, table,
                spline_starts, x, sh, senders, receivers, row_ptr, spline_id, wprime_id):
        from fastmace.kernels.oeq_streaming.spline_forward import spline_message_blocks
        from fastmace.kernels.triton.radial_table import radial_table_forward

        _, density = radial_table_forward(r, pair, density_coef, starts, scales, bins, r_lo, bin_scale, r_max, True,
                                          0)
        groups = spline_message_blocks(x, sh, senders, row_ptr, r, pair, table, spline_id)
        ctx.save_for_backward(x, sh, senders, receivers, r, pair, table, spline_starts, density_dcoef, starts, scales,
                              bins)
        ctx.constants, ctx.wprime_id = (r_lo, bin_scale, r_max), wprime_id
        return (*groups, density)

    @staticmethod
    def backward(ctx, *grads):
        (x, sh, senders, receivers, r, pair, table, spline_starts, density_dcoef, starts, scales,
         bins) = ctx.saved_tensors
        kernel = _REGISTRY[ctx.wprime_id]
        from fastmace.kernels.oeq_streaming.fused_forward import block_shapes
        shapes = block_shapes(kernel.blocks, kernel.channels, x.shape[0])
        douts = [x.new_zeros(shape) if g is None else g for g, shape in zip(grads[:-1], shapes)]
        # The spline kernel's piece count is spline_starts' length (its intervals); scales and bins are not read.
        dx, dsh, dr = streaming_conv_wprime_backward_blocks(x, sh, senders, receivers, douts, r, pair, table,
                                                            spline_starts, scales, bins, *ctx.constants,
                                                            ctx.wprime_id)
        if grads[-1] is not None:
            dr = dr + density_backward(r, pair, density_dcoef, starts, scales, bins, grads[-1], *ctx.constants)
        return (dr.reshape(r.shape),) + (None,) * 11 + (dx, dsh) + (None,) * 5


def spline_table(coef, row_scale=None):
    """SPLINE_W's q-major table ``[pairs * intervals, Q, 4]`` float32 of ``rewrites.radial.spline``'s ``coef [P, I,
    4, Q]``; ``row_scale`` ``[Q]`` (FOLD_ALPHA's ``alpha_rows``) multiplies each weight's row in float64."""
    c = coef.to(torch.float64)
    if row_scale is not None:
        c = c * row_scale.to(device=c.device, dtype=torch.float64)
    pairs, intervals, terms, q = c.shape
    return c.permute(0, 1, 3, 2).reshape(pairs * intervals, q, terms).to(torch.float32).contiguous()


def block_spec(linear, schedule):
    """The block layout of linear_1's grouped input (``BLOCK_MESSAGE``), ``((dim, segment starts), ...)`` in the
    order of its products, or None where the message cannot feed ``linear`` directly: a grouped ``BlockLinear``
    whose input transpose is the identity and whose input segments are exactly the convolution's output segments
    (ir_mul, ``schedule.channels`` channels each)."""
    from fastmace.rewrites.linear.modules import BlockLinear

    if not isinstance(linear, BlockLinear) or not linear.grouped or schedule.layout != "ir_mul":
        return None
    transpose = linear.transpose_in
    if not isinstance(getattr(transpose, "f", transpose), torch.nn.Identity):
        return None
    segments = {(int(s.start), int(s.irrep_dim)) for s in schedule.output_slots}
    blocks, seen = [], set()
    for inputs, _ in linear.block_ends:
        dims = {linear.segments_in[i][2] for i in inputs}
        if len(dims) != 1 or any(linear.segments_in[i][1] != schedule.channels for i in inputs):
            return None
        starts = tuple(int(linear.segments_in[i][0]) for i in inputs)
        if seen & set(starts):
            return None
        seen |= set(starts)
        blocks.append((dims.pop(), starts))
    if {(s, d) for d, st in blocks for s in st} != segments:
        return None
    return tuple(blocks)


def dense_message(groups, blocks, channels, output_dim):
    """The ``[N, output_dim]`` ir_mul message of a block-layout one (the paths that need the whole message)."""
    n = groups[0].shape[0] // blocks[0][0]
    pieces = {}
    for group, (dim, starts) in zip(groups, blocks):
        view = group.reshape(n, dim, len(starts), channels)
        for rank, start in enumerate(starts):
            pieces[start] = view[:, :, rank].reshape(n, dim * channels)
    out = torch.cat([pieces[s] for s in sorted(pieces)], dim=1)
    if out.shape[1] != output_dim:
        raise ValueError(f"dense_message: {out.shape[1]} columns, expected {output_dim}")
    return out


class _WPrimeConvolution(torch.autograd.Function):
    """Edge lengths -> (message, edge density) through the radial table, the folded layer and the streaming
    convolution; the backward is ``streaming_conv_wprime_backward`` plus the density column's."""

    @staticmethod
    def forward(ctx, r, pair, coef, density_dcoef, starts, scales, bins, r_lo, bin_scale, r_max, weight, gprime, x,
                sh, senders, receivers, row_ptr, kernel_id, fused_id, wprime_id, recompute_rows=0, message_id=-1,
                rebuild_id=-1, table_w=False):
        from fastmace.kernels.oeq_streaming.runtime import streaming_conv_forward
        from fastmace.kernels.triton.radial_table import radial_table_forward

        a, density = radial_table_forward(r, pair, coef, starts, scales, bins, r_lo, bin_scale, r_max)
        if rebuild_id >= 0 or (table_w and message_id >= 0):  # w built on chip, never stored
            from fastmace.kernels.oeq_streaming.fused_forward import fused_folded_message
            message = fused_folded_message(a, weight, x, sh, senders, receivers, row_ptr, message_id)
            weights = None
        elif fused_id >= 0:  # fused_folded_forward (opt-8 OPT8-T02): w stored by the message kernel
            from fastmace.kernels.oeq_streaming.fused_forward import fused_folded_forward
            message, weights = fused_folded_forward(a, weight, x, sh, senders, receivers, row_ptr, fused_id)
        else:
            weights = a @ weight.t()
            message = streaming_conv_forward(x, sh, weights, senders, row_ptr, kernel_id)
        if recompute_rows or weights is None or table_w:  # recompute_conv_weights (CAP-T07), TABLE_W: none
            weights = coef.new_empty((0,))
        ctx.save_for_backward(x, sh, weights, senders, receivers, r, pair, gprime, density_dcoef, starts, scales,
                              bins, coef, weight)
        ctx.constants, ctx.wprime_id, ctx.recompute_rows = (r_lo, bin_scale, r_max), wprime_id, int(recompute_rows)
        ctx.rebuild_id = int(rebuild_id)
        return message, density

    @staticmethod
    def backward(ctx, dout, ddensity):
        (x, sh, weights, senders, receivers, r, pair, gprime, density_dcoef, starts, scales, bins, coef,
         weight) = ctx.saved_tensors
        if dout is None:
            dout = x.new_zeros((x.shape[0], _REGISTRY[ctx.wprime_id].output_dim))
        if ctx.rebuild_id >= 0:
            from fastmace.kernels.oeq_streaming.wrebuild_backward import wrebuild_backward
            dx, dsh, dr = wrebuild_backward(x, sh, senders, receivers, dout, r, pair, gprime, coef, weight, starts,
                                            scales, bins, *ctx.constants, ctx.rebuild_id)
        elif ctx.recompute_rows:
            dx, dsh, dr = streaming_conv_wprime_backward_recompute(
                x, sh, senders, receivers, dout, r, pair, gprime, coef, weight, starts, scales, bins, *ctx.constants,
                ctx.wprime_id, ctx.recompute_rows)
        else:
            dx, dsh, dr = streaming_conv_wprime_backward(x, sh, weights, senders, receivers, dout, r, pair, gprime,
                                                         starts, scales, bins, *ctx.constants, ctx.wprime_id)
        if ddensity is not None:
            dr = dr + density_backward(r, pair, density_dcoef, starts, scales, bins, ddensity, *ctx.constants)
        return (dr.reshape(r.shape), None, None, None, None, None, None, None, None, None, None, None, dx, dsh,
                None, None, None, None, None, None, None, None, None, None)


def table_w_default(device, generic):
    """``TABLE_W`` for a WPrime on ``device``: the constant when set, else on for the generic family on
    ``TABLE_W_GPUS``."""
    if TABLE_W is not None:
        return bool(TABLE_W)
    device = torch.device(device)
    return bool(generic) and device.type == "cuda" and torch.cuda.get_device_name(device) in TABLE_W_GPUS


def block_message_default(device):
    """``BLOCK_MESSAGE`` on ``device``: the constant when set, else on for ``BLOCK_MESSAGE_GPUS``."""
    if BLOCK_MESSAGE is not None:
        return bool(BLOCK_MESSAGE)
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_name(device) in BLOCK_MESSAGE_GPUS


def block_ldg_default(device):
    """``BLOCK_LDG`` on ``device``: the constant when set, else on for ``BLOCK_LDG_GPUS``."""
    if BLOCK_LDG is not None:
        return bool(BLOCK_LDG)
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_name(device) in BLOCK_LDG_GPUS


def factored_default(device):
    """``FACTORED`` on ``device``: the constant when set, else on for ``FACTORED_GPUS``."""
    if FACTORED is not None:
        return bool(FACTORED)
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_name(device) in FACTORED_GPUS


def _device_name(device):
    device = torch.device(device)
    return torch.cuda.get_device_name(device) if device.type == "cuda" else None


def horner_default(device):
    """``HORNER`` on ``device``: the constant when set, else on for ``HORNER_GPUS``."""
    return bool(HORNER) if HORNER is not None else _device_name(device) in HORNER_GPUS


def scalar_paths_default(device):
    """``SCALAR_PATHS`` on ``device``: the constant when set, else on for ``SCALAR_PATHS_GPUS``."""
    return bool(SCALAR_PATHS) if SCALAR_PATHS is not None else _device_name(device) in SCALAR_PATHS_GPUS


def table_pairs_default(device):
    """``TABLE_PAIRS`` on ``device``: the constant when set, else on for ``TABLE_PAIRS_GPUS``."""
    return bool(TABLE_PAIRS) if TABLE_PAIRS is not None else _device_name(device) in TABLE_PAIRS_GPUS


def hoist_scale_default(device):
    """``HOIST_SCALE`` on ``device``: the constant when set, else on for ``HOIST_SCALE_GPUS``."""
    return bool(HOIST_SCALE) if HOIST_SCALE is not None else _device_name(device) in HOIST_SCALE_GPUS


def fold_alpha_default(device):
    """``FOLD_ALPHA`` on ``device``: the constant when set, else on for ``FOLD_ALPHA_GPUS``."""
    return bool(FOLD_ALPHA) if FOLD_ALPHA is not None else _device_name(device) in FOLD_ALPHA_GPUS


def alpha_rows(schedule, weights):
    """FOLD_ALPHA's ``[weights]`` float64 row factors: each path's FACTORED_SCALE alpha on its weight rows
    (``path.weight_start`` + edge channel * channels + channel), 1 elsewhere."""
    import numpy as np

    rows = np.ones(int(weights))
    for path, fp in zip(schedule.paths, factored_paths(schedule, True)):
        n = path.edge_mul * schedule.channels
        rows[path.weight_start:path.weight_start + n] = fp["alpha"]
    return torch.as_tensor(rows, dtype=torch.float64)


def estrin_default(device):
    """``ESTRIN`` on ``device``: the constant when set, else on for ``ESTRIN_GPUS``."""
    return bool(ESTRIN) if ESTRIN is not None else _device_name(device) in ESTRIN_GPUS


def stage_dout_default(device):
    """``STAGE_DOUT`` on ``device``: the constant when set, else on for ``STAGE_DOUT_GPUS``."""
    return bool(STAGE_DOUT) if STAGE_DOUT is not None else _device_name(device) in STAGE_DOUT_GPUS


def redux_default(device):
    """``REDUX`` on ``device``: the constant when set, else on for ``REDUX_GPUS``."""
    return bool(REDUX) if REDUX is not None else _device_name(device) in REDUX_GPUS


def butterfly_default(device):
    """``BUTTERFLY`` on ``device``: the constant when set, else on for ``BUTTERFLY_GPUS``."""
    return bool(BUTTERFLY) if BUTTERFLY is not None else _device_name(device) in BUTTERFLY_GPUS


def max_registers_default(device):
    """``MAX_REGISTERS`` on ``device`` (0: no cap): the constant when set, else ``MAX_REGISTERS_GPUS``' entry."""
    return int(MAX_REGISTERS) if MAX_REGISTERS is not None else int(lookup(MAX_REGISTERS_GPUS, _device_name(device), 0))


def prepare_block_messages(model):
    """``WPrime.prepare_blocks`` in every interaction of ``model`` with ``wprime_backward`` (after the rewrites;
    a no-op where ``BLOCK_MESSAGE`` is off). Returns how many interactions write the block layout."""
    count = 0
    if BLOCK_MESSAGE is False:
        return count
    for module in model.modules():
        wprime = module._modules.get("wprime_backward") if hasattr(module, "_modules") else None
        if isinstance(wprime, WPrime):
            wprime.prepare_blocks(module)
            count += wprime.block is not None
    return count


class WPrime(torch.nn.Module):
    """One interaction's ``wprime_backward``: the derivative table ``G'``, the density's derivative series and
    the compiled kernel, built from its radial table, folded weight and streaming kernels."""

    def __init__(self, table, weight, kernels, generic=False):
        super().__init__()
        # TABLE_W: the value table G in place of G' (w and its slope from one slab; the forward stores no w).
        self.table_w = table_w_default(table.coef.device, generic) and table.coef.dtype == torch.float32
        self.horner = self.table_w and horner_default(table.coef.device)
        device = table.coef.device
        self.fold_alpha = (self.table_w and fold_alpha_default(device) and factored_default(device)
                           and not FACTORED_SCALE)
        if self.table_w:
            self.terms, self.vector = int(table.coef.shape[2]), TABLE_W_VECTOR
            scale = alpha_rows(kernels.schedule, weight.shape[0]) if self.fold_alpha else None
            gprime = value_table(table.coef, weight, self.vector, row_scale=scale)
            if self.horner:  # HORNER: the power basis of t, for both kernels
                gprime = power_table(gprime, self.terms)
        else:
            self.terms = int(table.dcoef.shape[2])
            self.vector = SLOPE_VECTOR if table.dcoef.dtype == torch.float32 else 0
            gprime = derivative_table(table.dcoef, weight, self.vector)
        self.table_pairs = self.table_w and self.vector == 2 and table_pairs_default(table.coef.device)
        if self.table_pairs:
            gprime = pair_table(gprime)
        self.register_buffer("gprime", gprime, persistent=False)
        self.register_buffer("density_dcoef", density_column(table.dcoef), persistent=False)
        self.kernel = WPrimeBackward(kernels, self.terms, self.gprime.device, vector=self.vector,
                                     table_w=self.table_w, horner=self.horner, table_pairs=self.table_pairs,
                                     factored=True if self.fold_alpha else None, fold_alpha=self.fold_alpha)
        self.message = None  # TABLE_W with the fused forward: (its id, the message-only FusedFoldedForward)
        self.block = None  # BLOCK_MESSAGE: (fused id, blocks, block FusedFoldedForward, block WPrimeBackward)
        self.spline = None  # SPLINE_W (attach_spline): dict(grid, r_max, certificate, path_scale)
        self.spline_block = None  # SPLINE_W with BLOCK_MESSAGE: (fused id, SplineForward, spline WPrimeBackward)
        self.kernels = kernels
        # recompute_conv_weights on the fused kernels (attach_rebuild): the message-only forward and the backward that
        # builds w on chip, as (FusedFoldedForward(store_w=False), WRebuildBackward); None: the chunked recompute.
        self.rebuild = None

    def attach_spline(self, spline, certificate, table):
        """SPLINE_W: keep ``rewrites.radial.spline``'s spline (``dict(grid, coef [P, I, 4, Q])``, certified) as the
        q-major table both spline kernels read, and the radial table's density column for the forward; the kernels
        are compiled with the block layout (``prepare_blocks``). Returns False where the w' kernel cannot read it
        (needs TABLE_W, HORNER, float2 loads, no TABLE_PAIRS)."""
        if not (self.table_w and self.horner and self.vector == 2 and not self.table_pairs):
            return False
        weights = int(spline["coef"].shape[-1])
        scale = alpha_rows(self.kernels.schedule, weights) if self.fold_alpha else None
        device = self.gprime.device
        self.register_buffer("spline_table", spline_table(spline["coef"], scale).to(device), persistent=False)
        intervals = int(spline["coef"].shape[1])
        self.register_buffer("spline_starts", torch.zeros(intervals, dtype=torch.float32, device=device),
                             persistent=False)
        self.register_buffer("density_coef", table.coef[..., -1:].contiguous(), persistent=False)
        self.spline = dict(grid=[tuple(g) for g in spline["grid"]], r_max=float(table.r_max),
                           certificate=certificate,
                           path_scale=None if scale is None else [float(scale[p.weight_start])
                                                                  for p in self.kernels.schedule.paths])
        return True

    def prepare_blocks(self, interaction):
        """Compile the block-layout kernels (``BLOCK_MESSAGE``) for ``interaction``'s fused forward and linear_1,
        after every rewrite is applied (``prepare_block_messages``); nothing where the layout does not fit."""
        from fastmace.kernels.oeq_streaming.fused_forward import FusedFoldedForward

        self.block = None
        fused, linear = getattr(interaction, "fused_folded_forward", None), getattr(interaction, "linear", None)
        if not (block_message_default(self.gprime.device) and self.table_w and fused is not None
                and linear is not None):
            return
        blocks = block_spec(linear, self.kernels.schedule)
        if blocks is None:
            return
        device = self.gprime.device
        self.block = (fused.id, blocks,
                      FusedFoldedForward(self.kernels, fused.k, device, store_w=False, used=fused.used, blocks=blocks),
                      WPrimeBackward(self.kernels, self.terms, device, vector=self.vector, table_w=True, blocks=blocks,
                                     horner=self.horner, table_pairs=self.table_pairs,
                                     factored=True if self.fold_alpha else None, fold_alpha=self.fold_alpha))
        self.spline_block = None
        from fastmace.kernels.oeq_streaming.spline_forward import SplineForward, spline_w_default
        if self.spline is not None and spline_w_default(device):
            grid = self.spline["grid"]
            self.spline_block = (
                fused.id,
                SplineForward(self.kernels.schedule, blocks, grid, self.spline["r_max"], device,
                              self.spline["path_scale"]),
                WPrimeBackward(self.kernels, 4, device, vector=self.vector, table_w=True, blocks=blocks,
                               horner=True, factored=True if self.fold_alpha else None, fold_alpha=self.fold_alpha,
                               spline=grid))

    def attach_rebuild(self, kernels, fused):
        """Compile the message-only fused forward and the w-rebuilding backward for ``recompute_conv_weights``
        (float32, the interaction's ``fused_folded_forward`` of width ``fused.k``)."""
        if self.fold_alpha:
            raise ValueError("FOLD_ALPHA scales the value table's rows; the w-rebuilding backward reads them unscaled")
        from fastmace.kernels.oeq_streaming.fused_forward import FusedFoldedForward
        from fastmace.kernels.oeq_streaming.wrebuild_backward import WRebuildBackward

        if self.vector or self.table_w:
            raise ValueError("wprime_backward: the rebuild backward (recompute_conv_weights) reads the term-major "
                             "derivative table; it needs SLOPE_VECTOR = 0 and TABLE_W off")
        device = self.gprime.device
        self.rebuild = (FusedFoldedForward(kernels, fused.k, device, store_w=False),
                        WRebuildBackward(kernels, fused.k, self.terms, device, parts=REBUILD_PARTS,
                                         batch=REBUILD_BATCH))

    def convolution(self, lengths, pair, table, weight, x, sh, senders, receivers, row_ptr, kernels, fused=None,
                    recompute_rows=0, linear=None):
        """``(message, edge density)`` of edge lengths ``lengths`` ``[E, 1]`` (differentiable in lengths, x, sh).
        ``recompute_rows`` (``recompute_conv_weights``): keep no per-edge weights for the backward. With the rebuild
        kernels attached (``attach_rebuild``) and ``fused``, the forward stores no ``w`` and the backward builds it on
        chip; otherwise the backward builds it again in chunks of that many edges."""
        if weight.requires_grad:
            raise ValueError("wprime_backward: the folded weight must not require a gradient")
        if (self.spline_block is not None and fused is not None and linear is not None
                and self.spline_block[0] == fused.id and BLOCK_MESSAGE is not False):  # SPLINE_W
            *groups, density = _SplineBlockConvolution.apply(
                lengths, pair, self.density_coef, self.density_dcoef, table.starts, table.scales, table.bins,
                float(table.r_lo), float(table.bin_scale), float(table.r_max), self.spline_table, self.spline_starts,
                x, sh, senders, receivers, row_ptr, self.spline_block[1].id, self.spline_block[2].id)
            return tuple(groups), density
        if (self.block is not None and fused is not None and linear is not None and self.block[0] == fused.id
                and BLOCK_MESSAGE is not False):  # a tuple of linear_1's input groups (prepare_blocks)
            *groups, density = _WPrimeBlockConvolution.apply(
                lengths, pair, table.coef, self.density_dcoef, table.starts, table.scales, table.bins,
                float(table.r_lo), float(table.bin_scale), float(table.r_max), weight, self.gprime, x, sh,
                senders, receivers, row_ptr, self.block[2].id, self.block[3].id)
            return tuple(groups), density
        if self.table_w:  # no w is kept anyway; the fused forward's message-only kernel stores none
            if fused is not None and (self.message is None or self.message[0] != fused.id):
                from fastmace.kernels.oeq_streaming.fused_forward import FusedFoldedForward
                self.message = (fused.id, FusedFoldedForward(kernels, fused.k, self.gprime.device, store_w=False,
                                                             used=fused.used))
            message_id = self.message[1].id if fused is not None else -1
            return _WPrimeConvolution.apply(lengths, pair, table.coef, self.density_dcoef, table.starts, table.scales,
                                            table.bins, float(table.r_lo), float(table.bin_scale), float(table.r_max),
                                            weight, self.gprime, x, sh, senders, receivers, row_ptr, kernels.id,
                                            -1 if fused is None else fused.id, self.kernel.id, 0, message_id, -1, True)
        rebuild = (self.rebuild[0].id, self.rebuild[1].id) if recompute_rows and fused is not None and self.rebuild \
            else None
        return _WPrimeConvolution.apply(lengths, pair, table.coef, self.density_dcoef, table.starts, table.scales,
                                        table.bins, float(table.r_lo), float(table.bin_scale), float(table.r_max),
                                        weight, self.gprime, x, sh, senders, receivers, row_ptr, kernels.id,
                                        -1 if fused is None else fused.id, self.kernel.id, int(recompute_rows),
                                        *(rebuild if rebuild else (-1, -1)))

    def extra_repr(self):
        outputs = self.gprime.shape[1 if self.vector else 2]
        vector = f", q-major float{self.vector}" if self.vector else ""
        table = ", value table (w from the table)" if self.table_w else ""
        table += ", power basis (Horner)" if self.horner else ""
        if self.spline is not None:
            table += f", spline ({sum(g[2] for g in self.spline['grid'])} intervals"
            table += ", in use)" if self.spline_block is not None else ", not compiled)"
        return f"groups={self.gprime.shape[0]}, terms={self.terms}, outputs={outputs}{vector}{table}"


__all__ = ["BLOCK_LDG", "BLOCK_LDG_GPUS", "BLOCK_MESSAGE", "BLOCK_MESSAGE_GPUS", "FACTORED", "FACTORED_GPUS",
           "ESTRIN", "ESTRIN_GPUS", "FOLD_ALPHA", "FOLD_ALPHA_GPUS", "FACTORED_SCALE", "HORNER", "HORNER_GPUS", "MAX_REGISTERS", "MAX_REGISTERS_GPUS", "SCALAR_PATHS",
           "SCALAR_PATHS_GPUS", "TABLE_W", "TABLE_W_GPUS", "WPrime", "WPrimeBackward", "block_ldg_default",
           "block_message_default", "block_spec", "dense_message", "density_backward", "density_column",
           "alpha_rows", "derivative_table", "estrin_default", "fold_alpha_default", "factored_default", "factored_paths", "horner_default", "max_registers_default",
           "padded_terms", "pair_table", "power_table", "prepare_block_messages",
           "render", "spline_table", "STAGE_DOUT", "STAGE_DOUT_CHUNK", "STAGE_DOUT_GPUS", "stage_dout_default", "stage_dout_supported",
           "streaming_conv_wprime_backward", "streaming_conv_wprime_backward_blocks",
           "streaming_conv_wprime_backward_recompute", "scalar_paths_default", "table_w_default", "value_table",
           "wprime_arguments",
           "wprime_reference"]
