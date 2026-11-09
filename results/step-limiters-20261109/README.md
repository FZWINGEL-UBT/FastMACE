# What limits each part of the MD step: H200 and A100 (9 November 2026)

Owner request: "profile the steps in depth and find out what limits each part of each step" on the H200 and A100,
to find further ideas.

## Method

- Job `tools/modal/step_limiters.sh` (Modal, Nsight Compute 2026.4). Model and case: OMAT-0 medium, Benchmark_loup
  SrTiO3 10x10x10 (5,000 atoms, 373,760 padded edges), float32, each GPU's default plan, 300 K, 1 fs.
  - H200: source c520e074.
  - A100: source 18fe4b32. Modal gave an A100 80GB PCIe.
- **Step level:** `tools/step_breakdown.py` main pass gives wall time, stage times, idle time between stages and the
  gaps inside the force graph.
- **Kernel level:** one eager force call (forward and backward) under ncu. Metrics:
  - speed of light for SM, DRAM, L2 and L1;
  - every pipe's utilisation and issue activity;
  - every warp stall reason per issued instruction;
  - occupancy and its limits;
  - global-load sectors per request, shared-memory bank conflicts, threads per instruction.
- **Classification:** `tools/step_limiters.py` labels each kernel with one limiter, in this order:
  1. `tiny`: under 5 µs.
  2. Bound by a unit: the busiest unit is at 60% of its peak or more. The unit is `issue`, `dram`, `l2`, `l1`
     or `pipe:<name>`.
  3. `underfilled`: SMs active for less than 70% of the elapsed cycles.
  4. Otherwise `latency`, labelled with the largest stall reason.
- Each GPU folder holds `limiters.md`, `limiters.csv` (one row per kernel) and `limiters_by_category.csv`.
- Caveats:
  - ncu runs at the running clocks; Modal does not allow clock locking.
  - Caches are not flushed between kernels, so each kernel sees the step's own L2 state.
  - ncu kernel times are 3% (H200) and 0.5% (A100) above the profiler's eager times.

## Step level

| | H200 | A100 80GB PCIe |
| --- | ---: | ---: |
| Wall time per step | 3.425 ms | 7.467 ms |
| Force graph (B), per step in MD | 3.330 ms | 7.359 ms |
| Force graph replayed alone: kernels / span | 3.177 / 3.244 ms | 6.905 / 6.914 ms |
| Gaps inside the graph replay (all under 2 µs) | 63 µs (142 gaps) | 11 µs |
| ACP (half kick, positions, neighbour check, compaction) | 0.054 ms | 0.082 ms |
| Noise upload | 0.016 ms | 0.019 ms |
| Neighbour-list rebuild (GPU work, beside the force graph) | 4.31 ms every 7.7 steps | 8.63 ms every 7.7 steps |

- The force graph is 97% (H200) and 99% (A100) of the wall time.
- The rebuild runs on a side stream beside the force graph (`overlap_rebuild`), so it hides almost entirely.
- In MD, the force graph takes 0.09 ms (H200) and 0.45 ms (A100) longer than when replayed alone. That covers the
  rebuild running beside it and, on the PCIe A100, sustained-load clocks.
- `eng-rebuild-cost` (5 November) measured the rebuild's added time on the A100-SXM4 directly, as noise. A100 PCIe
  clocks are a likely cause of the extra time, so this is not taken as a lead.
- The H200's 63 µs of graph gaps (2%) are about 0.45 µs per kernel boundary, for 143 kernels. Only fewer kernels
  reduce them.

## Kernel level: what limits each part

Kernel time per force call, grouped by component (ncu).

| Component | H200 µs (share) | H200 limiter | A100 µs (share) | A100 limiter |
| --- | ---: | --- | ---: | --- |
| w′ backward (`oeq_projected_backward_wprime`) | 1092 (33.5%) | **issue** 77%, fmalite 60%, FMA 52% | 2265 (33.3%) | **FMA pipe** 85%, issue 61% |
| SPLINE_W forward (`spline_forward0`) | 638 (19.6%) | **issue** 82%, FMA 51% | 1550 (22.8%) | **L2** 70%, long scoreboard 10.4 per issue, issue 38% |
| tf32x3 GEMMs (CUTLASS sm80 64x64, 12 launches) | 609 (18.7%) | **tensor pipe** about 50%, math-pipe throttle; 104 µs underfilled | 1077 (15.8%) | **tensor pipe** about 55%, math-pipe throttle and wait |
| Triton elementwise and reductions | 273 (8.4%) | L2, underfilled, tiny | 443 (6.5%) | DRAM 80–90% |
| tabled outer sum backward | 211 (6.5%) | **L1** 71% (LSU 58%) | 448 (6.6%) | **L1** 64% (LSU 48%) |
| cuEq symmetric contraction | 200 (6.1%) | issue 70–76%, FMA 67–73% | 416 (6.1%) | **FMA pipe** 92–96% |
| tabled outer sum forward | 112 (3.4%) | issue 63%, 64 registers, 44% warps active | 398 (5.8%) | **latency** (long scoreboard 10.0), 128 registers, 23% warps active |
| Other | 123 (3.8%) | tiny, issue | 210 (3.1%) | underfilled, DRAM |
| **Total** | **3258** | | **6808** | |

By limiter:

| Limiter | H200 | A100 |
| --- | ---: | ---: |
| Issue | 62.4% | 1.0% |
| FMA pipe | (counted under issue) | 38.5% |
| L2 | 3.6% | 23.0% |
| Latency | 17.5% (mainly tensor-pipe throttle in the GEMMs) | 21.3% (GEMMs, tabled forward) |
| L1 | 6.9% | 6.6% |
| DRAM | 0% | 5.0% |
| Underfilled | 6.0% | 3.3% |
| Tiny | 3.5% | 1.2% |

## What this means, part by part

- **H200: the edge kernels are issue-bound, not FMA-bound.**
  - w′ issues 77% of cycles and the forward 82%, but the FMA pipes run only 51–60%. So about a third of the issued
    instructions are not FMAs: index math, loads, shuffles and the per-edge reduction.
  - Fewer non-FMA instructions per edge is the only lever there. The best candidate is w′'s end-of-edge reduction:
    MOV 176, SHFL 85, WARPSYNC 88 and ENDCOLLECTIVE 87 per loop in the sm_90 SASS (`../packed-table-lab-20261109`,
    "uniform datapath"). A second FP64 pipe, a second tensor pipe, SMEM spilling, overlap and the uniform datapath
    are all closed.
- **A100: w′ is FMA-bound and the forward is L2-bound.**
  - w′ at 85% FMA pipe is near its roof; only fewer FMAs help.
  - The forward is different on the A100: L2 throughput 70%, 12.4 sectors per load request, about 10 warps stalled
    on global memory per issued instruction, and issue only 38%. The table reads dominate.
  - PACKED8 halves the table bytes. It was adopted on the H200, where the forward is issue-bound and gained only
    0.6%, and had never been measured on the A100. Measured now: 4.5% faster, adopted (below).
- **tf32x3 GEMMs (16–19%) run at the tensor pipe's mma.sync rate.**
  - The tensor pipe is about 50–55% busy (the shared pipe the same), throttled, at 17–19% occupancy (134–158
    registers, 64x64 tiles).
  - On the H200, the M = 5,000 launches leave SMs idle: 104 µs underfilled, 1.2 waves.
  - Earlier rounds closed the wgmma route (`h200-wgmma`, `ideas-r14` item 33) and the grouped launch
    (`SEGMENT_GROUPED`, off).
  - Still open: a CUTLASS 3 TMA warp-specialised 3xTF32 GEMM on the H200. Elsewhere only fewer products or fewer
    launches help.
- **Tabled outer sums.**
  - The backward is L1-bound on both GPUs (LSU 48–58%, 3–4 sectors per request, no bank conflicts): it needs fewer
    loads, not better ones.
  - The forward is 3.5x slower on the A100 than on the H200, against about 2x for everything else. Triton compiles
    the one-warp kernel to 128 registers on sm_80 (64 on sm_90). That allows 16 blocks per SM (23% of warps active)
    on a kernel that waits on global loads.
  - `FORWARD_MAXNREG` (new, off) caps it. The A/B at 64 and 80 registers was a tie (below).
- **cuEq symmetric contraction (6%):** the A100 is FMA-bound at 92–96%, the H200 near issue-bound. That is the
  mandated arithmetic; nothing to gain without fewer FLOPs.
- **Triton elementwise kernels (7–8%):** DRAM- or L2-bound on the A100, for example the species_mm_rows_grad
  reduction at 83% DRAM and `edge_sh3_backward` at 81%.
  - Removing their traffic means fusing them into neighbours, for example the [5000, 2048] reduction into the
    `species_mm_rows_grad` GEMM epilogue.
  - At most about 1–2% each.

## Follow-up runs started from this profile

Both ran on Modal A100-SXM4-80GB, MD-step A/Bs with `tools/modal/round6_ab256.sh` (b/a, abab / abba).

| Run | Result |
| --- | --- |
| PACKED8 on the A100 (`tools/modal/packed8_gate_ab.sh`, `GPU_MATCH=A100`, source 10e23d17; [../packed-table-lab-20261109/a100-packed8-gate-ab/](../packed-table-lab-20261109/a100-packed8-gate-ab/)) | 300 K **0.9557 / 0.9549**, 1500 K **0.9587 / 0.9558** (7.04 → 6.73 ms). The gate passes, with max error / ase_f32's ratios for forces 0.919, 0.843, 0.973 and stress 0.658, 0.917, 1.158 (SrTiO3, NaCl, NbCl5); the default side had 0.965, 0.868, 0.982 and 0.624, 0.936, 1.324. **Adopted on the A100 family** (`PACKED8_GPUS`). As the profile predicted, halving the table bytes pays where the forward is L2-bound: 4.5% on the A100 against 0.6% on the issue-bound H200. |
| `tabled_outer_sum.FORWARD_MAXNREG` (`tools/modal/tabled_maxnreg_ab.sh`, source 31ed59c4; [a100-maxnreg/](a100-maxnreg/)) | Cap 64: 1.0011 / 0.9975. Cap 80: 1.0027 / 0.9993. A tie; rejected (off). The kernel's long-scoreboard waits are not relieved by occupancy alone. The registers per thread at the cap were not checked with ncu. |

## Leads left

1. **H200: w′'s end-of-edge reduction.** About 520 non-FMA instructions per loop (MOV, SHFL, WARPSYNC,
   ENDCOLLECTIVE) in an issue-bound kernel that is 33% of kernel time.
2. **Both GPUs: tf32x3 GEMMs (16–19%).** They run at the mma.sync tensor rate. Not yet tried: a CUTLASS 3 TMA
   warp-specialised 3xTF32 GEMM on the H200. On the H200, 104 µs of the GEMMs are wave tails at M = 5,000.
3. **Both GPUs: tabled outer sum backward (6.5%).** L1/LSU-bound, so it needs fewer loads per edge.
4. **A100: w′ is FMA-bound at 85%.** Only fewer FMAs help; the FP64 and DMMA second pipes are closed.
5. **A100: Triton reductions at 80–90% DRAM.** Each could be fused into its producer GEMM's epilogue, worth 1–2% each.

## Follow-ups on Modal, round 2

### PACKED8 on: the A100 re-profiled ([a100-packed8/](a100-packed8/limiters.md), source 0df22417, A100 80GB PCIe)

The SPLINE_W forward is now FMA-pipe bound: 1,539 → 1,255 µs, FMA pipe 84%, L2 42%, issue 71%. The force stage per
step went 7.36 → 7.13 ms. Both edge kernels are now at the A100's FMA roof (w′ 85%, forward 84%).

### w′ bounds (`tools/modal/kernel_lab.sh`, falsifiers; time without the part / base)

| Part removed | H200 (dad5375b, base 1,098 µs) | A100-SXM4 (0df22417, base 2,319 µs) |
| --- | ---: | ---: |
| Cross-lane reduction (dsh and dr) | 0.932 | 0.950 |
| dsh's reduction (dr kept) | 0.937 | 0.951 |
| dr's reduction (dsh kept) | 0.994 | 0.991 |
| 4 of 5 shuffle levels | 0.947 | 0.956 |
| dx atomics | 0.996 | 0.962 |
| The slope | 0.903 | 0.923 |
| The table's Horner evaluation | 0.853 | 0.781 |

### Reduction variants against today's w′ defaults (H200 kernel lab, dad5375b, base 1,089 µs)

| Variant | Ratio | Exactness |
| --- | ---: | --- |
| `butterfly` | **0.956** | dsh and dr bitwise |
| `redux` | 0.975 | fixed point, 2.8e-7 |
| `F-redux16` | 0.974 | falsifier |
| `redux-pc` | 1.004 | fixed point |

On 6 November (d599cc70) the butterfly was 1.145 and was turned off on the H200. The w′ defaults have changed since:
STAGE_DOUT, STAGE_SH, SPLINE_VECTOR, FOLD_ALPHA and SPLINE_W.

### BUTTERFLY MD-step A/Bs (`tools/modal/round6_ab256.sh`, SKIP_GATE=0; b = on)

| GPU (source) | SrTiO3 10×10×10 | NbCl5 | Gate |
| --- | --- | --- | --- |
| H200 (e3049399) | **0.9863 / 0.9861** (3.416 → 3.369 ms) | 0.9936 / 0.9903 | passes (forces 0.849, 1.006, 0.903) |
| A100-SXM4-80GB (dad5375b, with PACKED8) | **0.9910 / 0.9892** (6.745 → 6.684 ms) | 0.9942 / 0.9973 | passes (forces 0.826, 0.982, 0.890) |

**Adopted on the H200 and the A100 family** (`BUTTERFLY_GPUS`, decision 66). Results: [h200-butterfly/](h200-butterfly/),
[a100-butterfly/](a100-butterfly/).


## Round 3: re-screen of off switches against today's defaults, and the quadratic-table bound

- MD-step A/Bs (`tools/modal/const_screen.sh`, SrTiO3 10×10×10, both orders, no gate) of switches that are off, or
  on only elsewhere, last measured before today's defaults:
  - H200: `TABLE_PAIRS`, `HOIST_SCALE`, `STAGE_QUADS`, w′ `STAGE_EDGES`, forward `PREFETCH_AHEAD` 2, `PACKED_COEF`,
    `SEGMENT_ROWS` + `SEGMENT_GROUPED`.
  - A100: forward `PREFETCH_AHEAD` 0 and 4 (the default 2 was tuned before PACKED8 halved the rows),
    `POWER_BACKWARD`, the tabled forward's `FORWARD_WARPS` 2 and 4 and `FORWARD_CHUNK` 16, `PACKED_COEF`.
- Kernel lab `F-quad-horner` (H200 and A100): w′ with a quadratic's value and slope (3 FMAs) in place of the cubic's
  dual Horner, the same loads. It is the timing bound of a quadratic spline table with twice the pieces.
