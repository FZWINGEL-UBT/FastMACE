# OPT4-T01: write-bound Triton kernel for the radial last layer's forward; cut at its first measurement

Optimisation round 4 (`plans/opt-4.md`, §3 candidate 1). Date: 24 October 2026. **Result: cut by the
falsifying microbenchmark. No Triton kernel came below cuBLAS. The best one ties cuBLAS and is 31 % above the
plan's limit on NbCl5. No library code, no record, no gate and no A/B.**

## The operation

The folded last radial layer of MH-1's second interaction computes `[h c, c] @ [W; b]^T`. Here `h` is the
64-wide hidden output, `c` the cutoff, and `Q = 1280` the number of output columns
(`FactoredRadialMLP.after_first`, `fastmace/rewrites/mh1/modules.py`).

- **The input width is 65:** 64 hidden columns plus the cutoff column.
- **The K = 68 in the profiles is padding.** Inductor's `pad_mm` widens the operand to 68 columns, a multiple
  of 4, before cuBLAS runs (`gemms.csv` of `results/opt4-baseline-ccab528/local-mh1-multihead/`: operands
  `[E, 68]` and `[68, 1280]`).
- This confirms the layout §3 asked the worker to check: 64 hidden columns plus 4 more, of which 1 is real.

## Microbenchmark

- **Machine:** local RTX 5060 Ti (sm_120, 36 SMs), WSL2, `~/environments/mace_cueq_oeq` (torch 2.14.1+cu130,
  Triton 3.8.0).
- **Source:** FastMACE branch `fastmace/opt4-T01` on main `13a1983`.
- **GPU slots:** two `tools/ci/local_gpu.py` reservations from 09:50 UTC. Untimed: the PC was not quiet, so
  the numbers are microbenchmarks, not A/Bs.
- **Method:** CUDA-graph replays, median of 7 blocks of 50, random operands at the production layout
  (`a = [h c, c]` of `[E, 65]`, `weight [1280, 65]`).
- **Tools:** `tools/radial_forward_bench.py` (tile sweep, cuBLAS and the cheap alternative) and
  `tools/radial_forward_probe.py` (what limits the kernel).
- **Records:** `results/opt4-T01/local-microbench/radial_forward.json` and `radial_forward_probe.json`.
  - The first file's `rule_config` and `rule_us` fields come from a draft library module, since removed.
    That module had the same kernel at one fixed tile (64×128, 4 warps, 1 stage).

**The kernel tried.** All of K fits in one tile. Each program loads its rows of `a` once. It multiplies the
first 64 columns with an IEEE FP32 `tl.dot` (no TF32) and adds the cutoff column as a rank-1 update. Then it
streams its `[BLOCK_M, BLOCK_N]` block to memory. Variants:
- 36 tile configurations: BLOCK_M 32/64/128, BLOCK_N 64/128/256, 4 or 8 warps, 1 or 2 stages;
- K steps of 16, 32 and 64;
- a persistent version that writes its output through a TMA tensor descriptor, so that each tile's store can
  overlap the next tile's product. It ran with 1, 2 and 4 programs per SM.

| µs per call | NaCl (E = 8,192) | NbCl5 (56,320) | NaNbO3 (82,688) |
| --- | ---: | ---: | ---: |
| **Plan's limit** | **≤ 150** | **≤ 1,000** | **≤ 1,450** |
| cuBLAS on the padded `[E, 68]` (production) | 183–186 | 1,270–1,286 | 1,897–1,913 |
| cuBLAS at K = 64 alone (opt-4 §3 evidence, no cutoff column) | 165 | 1,104 | 1,630 |
| Cheap alternative: cuBLAS K = 64 plus a rank-1 update (`addr_`) | 376 | 2,650 | 3,936 |
| Best Triton tile (non-persistent) | 199 (128×64, 4 warps) | 1,331 (128×64, 4 warps) | 1,969 (64×128, 4 warps) |
| Best Triton, persistent with TMA stores | 190 | 1,306 | 1,913 |
| Triton kernel storing zeros (same grid; its write rate alone) | 113–115 | 794–799 | 1,167–1,169 |
| `fill_` of `[E, 1280]` | 113–114 | 793–799 | 1,168–1,171 |

Accuracy against float64: the largest relative error is 3.4–3.7e-7 for every Triton variant. That equals
cuBLAS's error.

## Why it fails

- **The kernel is compute-bound, not write-bound.** Its store alone runs at the `fill_` rate (0.79 ms on
  NbCl5). With the product it takes 1.31–1.36 ms. Triton's IEEE FP32 `tl.dot` (FMA) reaches 7.2–7.5 TFLOPS
  here: 9.8 GFLOP per NbCl5 call, against 16.7 TFLOPS measured peak. That is about cuBLAS's own rate on this
  shape (8.0 TFLOPS).
- **Overlapping the store does not change this.** The persistent TMA version ties cuBLAS (1.306 against
  1.286 ms), because the time it saves on stores goes back into the product.
- **Larger K steps get worse.** A K step of 64 takes 2.6–3.0 ms on NbCl5 (register spills), against
  1.36–1.38 ms for steps of 16 and 32.
- **Even an ideal cutoff column would miss the limit.** cuBLAS's best kernel for this shape (K = 64, no
  cutoff column at all) takes 1.104 ms on NbCl5 and 1.630 ms on NaNbO3, above the limits of 1.00 and 1.45 ms.
- **What the limit needs.** Reaching it takes a SIMT SGEMM more efficient than cuBLAS at K = 64: about 10
  TFLOPS sustained with only two K steps per tile, while streaming 288 MB out. That is CUTLASS-class SIMT
  work, outside this candidate's budget (M).
- **The ceiling of the realistic part is small.** It is the gap between K = 68 and K = 64 in cuBLAS: 0.18 ms
  on NbCl5 (0.8 % of the step) and 0.26 ms on NaNbO3 (1.4 %). Only a kernel at cuBLAS's K = 64 rate that
  also takes the cutoff column for free could reach it, and none of the variants above came near that rate.
- **The cheap alternative is worse than today.** A separate rank-1 update reads and writes the whole
  `[E, 1280]` output again, which doubles the time.

## Decision

Cut, as `plans/opt-4.md` §3 says for a failed first measurement. Not built: the `Rewrite` record, the force
gate, the CPU float64 check, the local A/B in both orders and the Colab L4 A/B. No Colab compute units were
used. The two tools stay in `tools/` so the measurement can be repeated, for example on a GPU where Triton's
FP32 dot reaches a larger share of SIMT peak.
