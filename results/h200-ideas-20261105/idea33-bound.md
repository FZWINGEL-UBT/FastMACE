# Idea 33 bound: Hopper TF32 wgmma x3 against today's 3xTF32 GEMMs on the H200 (6 November 2026)

**Verdict: stop.** Three single passes of CUTLASS's native sm_90a TF32 wgmma GEMM take **1.07x** today's time on
the six largest shapes (threshold for building: <= 0.6). Over all 24 shapes the ratio is 1.15. A pre-split single
GEMM (K x 3, a correct 3xTF32 sum) comes to 0.67 without the split of A. With the split it is 1.74, because the
split as three torch kernels costs more than the GEMM. Both miss 0.6.

## Setup

- **What is compared.** In one process on one GPU (NVIDIA H200, 132 SMs, sm_90, driver 610.57.04, CUDA 13.3 UMD,
  Modal profile `fourth`):
  - **today:** `fastmace.kernels.cutlass.tf32x3.mm_into`. This is the sm_80 mma.sync `OpMultiplyAddFastF32` mainloop
    (sm_90 cubin), tile `64x64`, as production calls it;
  - **wgmma:** CUTLASS 3.x `CollectiveBuilder` TF32 GEMMs, TMA with warp-specialised mainloops, `SS_TN` GMMA atoms.
    They are built into the lab library `tools/bench/idea33/libidea33_wgmma.so` by `tools/bench/idea33/build.sh`:
    CUTLASS **v4.8.1** (`2c85f497`, the checkout of the production cubins), nvcc 13.3, `-arch=sm_90a`. It is
    prebuilt because the job image has no nvcc.
- **Configs.** Seven configs were timed, and the best one per shape is kept:
  - 128x128x32, 128x64x32 and 128x192x32 cooperative;
  - 128x128x32 cooperative with cluster 2x1;
  - 64x128x32, 64x64x32 and 64x192x32 pingpong.

  **Chosen:** mostly **64x128x32 pingpong** and **64x192x32 pingpong** (cluster 1x1), plus 64x64x32 pingpong for
  35000x384x128.
- **Shapes.** I recorded the shapes from one eager force call at this branch's commit. This is Benchmark_loup:
  OMAT-0 medium, SrTiO3 10x10x10, 5,000 atoms, the minimal plan's engine as `tools/step_breakdown.py` builds it.
  - 33 CUTLASS launches over 24 distinct (M, K, N, B layout) shapes.
  - Measured in isolation, they total 852 µs per force call. The 6 November profile had 866 µs for all
    `fm_tf32x3` launches, and 828 µs for its six largest kernel/grid groups.
  - "Six largest" below means the six distinct shapes with the most time today per force call: 383 µs in total.
- **Timing.** Random operands, median µs per launch over 25 replays of a CUDA graph with 20 launches. Today's side
  was timed before and after the wgmma side, and the smaller value kept. Two fresh processes were run, and the
  table shows their mean (ratios 1.074 and 1.074).
- **Operands.** wgmma TF32 needs both operands K-major, so B is passed as B^T [N, K]. A weight can be stored that
  way for free. The single pass has beta 0 and no C operand, so the x3 estimate is optimistic: a real three-pass
  sequence would also read and accumulate C twice.

## Result (six largest shapes; µs per force call = launches x µs per launch)

| M | K | N | B | launches | today µs/launch | today µs/call | sm_90 1-pass µs/launch (config) | x3 µs/call | x3 / today | K3 pre-split µs/call | K3 / today |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 25000 | 128 | 384 | rr | 2 | 48.9 | 97.8 | 17.5 (64x192x32 pingpong) | 104.8 | 1.07 | 59.3 | 0.61 |
| 35000 | 128 | 384 | rr | 1 | 66.5 | 66.5 | 24.6 (64x192 / 64x128 pingpong) | 73.8 | 1.11 | 41.4 | 0.62 |
| 35000 | 384 | 128 | rc | 1 | 66.6 | 66.6 | 22.0 (64x64x32 pingpong) | 65.9 | 0.99 | 51.8 | 0.78 |
| 15000 | 128 | 384 | rr | 2 | 30.1 | 60.3 | 11.4 (64x192x32 pingpong) | 68.4 | 1.14 | 37.3 | 0.62 |
| 35000 | 128 | 256 | rr | 1 | 46.3 | 46.3 | 18.5 (64x128x32 pingpong) | 55.5 | 1.20 | 31.2 | 0.67 |
| 25000 | 384 | 128 | rc | 1 | 45.3 | 45.3 | 14.2 (64x128x32 pingpong) | 42.5 | 0.94 | 35.7 | 0.79 |
| **Sum** | | | | 8 | | **382.8** | | **410.9** | **1.074** | **256.7** | **0.671** |

"rr" means B is row-major [K, N]; "rc" means it is column-major (the `b.t()` of the backward).

The pre-split forms are one wgmma GEMM over pre-split operands:
- **K x 3:** A' = [A_big | A_big | A_small] and B' = [B_big; B_small; B_big]. This is the correct three-term sum,
  with error 2.7e-6 scaled by |A||B|; today's error is 2.6e-7, and one TF32 pass gives 5.8e-4.
- **K x 2:** A' = [A_big | A_small] and B' = [B_big; B_big]. Only two of the three terms, so it is a timing proxy:
  0.51 on the six shapes.

Splitting A with three torch kernels (mask, copy, subtract) costs 1.07 of today's time on the six shapes, which
gives 1.74 in total. A split fused into A's producer would cost less, but K x 3 alone is already 0.67.

## Why

The single wgmma pass is about 2.7-3x faster than today's 3xTF32 kernel, so the native mainloop does not remove
the three-product cost. Today's kernel already does its three products in one pass over the operands. The wgmma
pass is limited by memory traffic, not by the tensor cores. At 25000x128x384 it moves about 51 MB in 17.5 µs (about
2.9 TB/s), while running at about 140 TFLOPS of TF32. Three passes triple that traffic, and K x 3 triples the A
traffic. Only an in-kernel split could get below 0.6: one read of A and B, with three wgmma products per k-block, as
`OpMultiplyAddFastF32` does with mma.sync. Its floor is the single pass's memory time, 0.36 of today on these
shapes. That needs a custom CUTLASS 3.x mainloop with a register-side split,
because Hopper's TF32 wgmma reads its B operand from shared memory. That is outside this bound and was not built.

## Run

- **Modal run directory:** `~/fm-runs/idea33-h200-2` (WSL), copied to
  `results/h200-ideas-20261105/runs/idea33-h200/`. It holds `idea33-1.json` and `idea33-2.json` with every config's
  time and error, plus logs and `modal.json`.
- **Commit:** source `740f4c6e`. Job `tools/modal/h200_idea33.sh`, bench `tools/bench/idea33/bench.py`.
- **Cost:** about $0.22 metered, and $0.09 for a first attempt (`idea33-h200-1`, source `c12378a7`). That attempt
  failed before any GPU timing because the bench touched CUDA before the engine set the allocator.
- **Plan text:** the idea 33 / G1 wording was not found in this repository; I used the wording from the conductor's
  brief.
