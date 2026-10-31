# M1 (L2 audit) and M2 (gap trace), 31 October 2026

Eighth list, measurements M1 and M2 (plans/capacity-ideas-20261030.md), plus item 4 of the ninth list. Measured
only; no library change.

| | Verdict |
|---|---|
| **M1** | No kernel re-reads beyond L2 the way `segment_gemm` did (8.5×). Every top kernel with a computable ideal is at 0.7–1.3 times its inputs' bytes. The large ratios are outputs: the stored per-edge weights and the radial table's `a`. **No fix candidate with a re-read ceiling above about 1 %.** |
| **M2 (a)** | GPU idle between and around steps: 0.24–0.50 % of the step at about 9,000 atoms, 0.8 % at 1,000–2,000. Under the 2 % stop rule: **the GPU-resident loop is dead.** |
| **M2 (b)** | Gaps between the force graph's kernels: 0.06–0.07 % at about 9,000 atoms, 0.5–0.8 % at 1,000–2,000. Under the 3 % stop rule: **the megakernel is dead.** |
| **Item 4** | `edge_outer_sum` backward plus interaction 0's radial-table backward: 422–623 µs per step at about 9,000 atoms (0.5–0.57 % of the step), 37–51 µs at 1,000–2,000. Above the 100 µs stop line at 9,000 atoms. Fusing `dr` removes only part of this time; the ceiling is under 0.5 % of the step. |

## Setup

| Item | Value |
|---|---|
| Machine | local PC, WSL2, RTX 5060 Ti 16 GB (sm_120, 36 SMs, 32 MiB L2), driver 610.88 |
| Environment | `~/environments/mace_cueq_oeq` (hybrid) |
| FastMACE | a0992314 = main 5c975d8d (tile order merged) plus this branch's scripts |
| Model and plan | MH-1 `mh1-r2scan`, the minimal planner's default float32 plan (hybrid, CUDA graphs, every `auto` rewrite) |
| Run | Langevin NVT, 1 fs, 1000 K, 100 equilibration steps |
| Cells | NbCl5 4×4×8 (9,216 atoms), NaNbO3 6×6×6 (8,640), NbCl5 3×3×3 (1,944), NaNbO3 3×3×3 (1,080) |
| Tools | `tools/step_breakdown.py` passes, as the fp32 profile used them; launcher `results/l2-gap-audit-20261031/run.sh` |

The passes:
- **main:** wall time and stages.
- **ncu:** one eager force call of the compiled model, with step_breakdown's metrics plus
  `lts__t_sector_hit_rate.pct`.
- **nsys:** force-graph replays with every node traced, for the gaps.
- **nsys-steps:** whole MD steps with NVTX ranges, for the idle holes and the per-kernel time per step. 100 steps on
  the large cells, 300 on the small.

**No Ada leg.**
- leto, murphy, coraline and leeloo all set `RmProfilingAdminOnly: 1`, so Nsight Compute needs root there.
- The Modal image has no Nsight tools.
- A Colab L4 was tried, but the account has 0 compute units: `colab new` failed and no VM was allocated.

## M1: L2 hit rate and DRAM bytes of the top kernels (eager force call, Nsight Compute)

Setup and columns:
- Kernels are grouped by name; "Share" is of the force call's kernel time.
- **Ideal** is each input tensor read once (from the op's input shapes, 4 bytes per element). It is "–" where the
  op's inputs are not recorded: cuBLAS `gemm`, cuEquivariance, and `kernel` (`edge_outer_sum`).
- Nsight Compute ran with `--cache-control none --clock-control none`, so the L2 carries over between kernels as in
  a real step. Its kernel times are longer than in graph replay, so the shares are for ranking.

**NbCl5 4×4×8, 9,216 atoms** (force call 82.6 ms of kernels, 171 kernels):

| Kernel | Launches | ms | Share | L2 hit % | DRAM read / write MB | DRAM / ideal |
|---|---|---|---|---|---|---|
| `gemm` (cuBLAS) | 16 | 23.26 | 28.2 % | 86.8 | 2,775 (sum) | – |
| `cutlass::Kernel2` | 29 | 13.85 | 16.8 % | 72.6 | 2,459 (sum) | 1.06 |
| `oeq_projected_backward_wprime` | 1 | 8.77 | 10.6 % | 87.1 | 1,915 / 81 | 1.23 (reads 1.18) |
| `fused_folded_forward` | 1 | 6.76 | 8.2 % | 43.4 | 380 / 1,570 | output: the w store |
| `joined_products` (Triton, fused) | 8 | 3.30 | 4.0 % | 23.4 | 1,259 (sum) | 0.71 |
| `kernel` (`edge_outer_sum` forward and backward) | 2 | 2.23 | 2.7 % | 38.7 | 384 / 166 | – |
| `magma_sgemmEx_kernel` | 9 | 1.90 | 2.3 % | 80.2 | 348 (sum) | 1.04 |
| cuEq `segmented_polynomial` (bwd algo4, algo6) | 2 | 3.58 | 4.4 % | 6–30 | 756 / 580 | about 1.1 (Y is 302 MB) |
| Triton elementwise and reductions (8 kernels) | | 8.6 | 10 % | 0.5–61 | | 0.98–1.30 |

**NaNbO3 6×6×6, 8,640 atoms** (106.6 ms, 176 kernels):

| Kernel | Launches | ms | Share | L2 hit % | DRAM read / write MB | DRAM / ideal |
|---|---|---|---|---|---|---|
| `gemm` (cuBLAS) | 16 | 21.97 | 20.6 % | 87.0 | 2,560 (sum) | – |
| `oeq_projected_backward_wprime` | 1 | 21.63 | 20.3 % | 88.3 | 4,432 / 185 | 1.24 (reads 1.19) |
| `fused_folded_forward` | 1 | 15.39 | 14.4 % | 46.4 | 894 / 3,598 | output: the w store |
| `cutlass::Kernel2` | 30 | 13.46 | 12.6 % | 72.6 | 2,290 (sum) | 1.03 |
| `kernel` (`edge_outer_sum`) | 2 | 4.66 | 4.4 % | 41.6 | 766 / 318 | – |
| `joined_products` | 8 | 2.92 | 2.7 % | 25.3 | 1,125 (sum) | 0.68 |
| `forward` (`radial_table_forward` ×2) | 2 | 2.41 | 2.3 % | 84.6 | 53 / 332 | output: `a` |
| `magma_sgemmEx_kernel` | 8 | 2.00 | 1.9 % | 84.0 | 319 (sum) | 0.91 |
| cuEq `segmented_polynomial` bwd | 2 | 3.54 | 3.3 % | 12–31 | 733 / 540 | about 1.1 |

**Reading:**
- **`oeq_projected_backward_wprime`** (11–20 % of the call) reads the stored weights once: 1.38 GB of its 1.92 GB
  at 9,216 atoms, 3.43 of 4.43 GB at 8,640. The rest is the node gathers at about 1.2 times their inputs.
  - Its L2 traffic is 15–38 GB at an 87–88 % hit rate. That is L2-bound gather traffic, not a DRAM re-read.
- **`fused_folded_forward`** writes the per-edge weights `w [E, 1,280]`: 1.38 GB at 9,216 atoms, 3.43 GB at 8,640,
  which is most of its DRAM bytes. Its reads are 3.5× its inputs (the `x` gathers per edge), 0.4–0.9 GB.
  - Not storing `w` is `recompute_conv_weights` (CAP-T07), already measured at 1.01–1.04× the step. It is a memory
    mode, not a speed gain.
- **`segment_gemm`** (inside the `joined_products` fusions) runs at 0.7× its inputs' bytes since the tile order
  (merged bb208f2a).
- **cuBLAS `gemm`:** 87 % L2 hit. Its inputs are not recorded, so it has no ideal.
- **cuEquivariance's `segmented_polynomial`:** low L2 hit rates (6–31 %), but its DRAM reads are about 1.1 times
  the contraction input `Y`. A streaming kernel reading its input once has a low hit rate by design.
- **Elementwise and reduction kernels:** at 0.95–1.3 times their inputs.

**Shortlist (fix candidates and ceilings):**

1. **`oeq_projected_backward_wprime` gather locality.** Its L2-bound traffic is the obvious one. But the group
   order, which was the one idea for it, just measured 1.29–1.43× slower on NbCl5 (docs/validation/wprime-group-
   order.md). Its DRAM bytes are at 1.18× the reads, so the DRAM ceiling is at most 0.18 × 11–20 % = 2–4 % of the
   force call, and only if the excess were removable, which receiver order already minimises. Not proposed.
2. **The `w` store of `fused_folded_forward`:** a write, not a re-read. Removing it is CAP-T07's rebuild, which
   costs more than it saves in time. Not proposed for speed.
3. **`edge_outer_sum` forward and backward:** 2.7–4.4 % of the call, with no ideal recorded. Their reads, 384 MB at
   9,216 atoms, are of the order of `[E, 16] + [E, 128]` floats (156 MB per pass). That is at most 1.1–1.2×,
   and the ceiling is under 1 %. See item 4.

## M2: gaps

**(a) GPU idle time per step.**
- Source: Nsight Systems over whole steps, holes of at least 10 µs with no kernel on any stream
  (`tools/bench/fp32_streams.py`).
- "Wall" is the uninstrumented engine loop. "GPU span" is CUDA events over the stages.

| Cell | Wall ms/step | GPU span ms/step | Idle holes µs/step (under nsys) | Share of step | Where |
|---|---|---|---|---|---|
| NbCl5 9,216 | 84.00 | 84.17 | 422.8 | 0.50 % | 326 µs before the noise kernel (0.88 per step: a noise batch), 97 µs after the force graph's scatter |
| NaNbO3 8,640 | 109.49 | 110.57 | 261.2 | 0.24 % | 138 µs noise batch, 119 µs after the scatter |
| NbCl5 1,944 | 17.70 | 17.76 | 137.5 | 0.78 % | after the scatter, then the noise |
| NaNbO3 1,080 | 13.15 | 13.17 | 107.3 | 0.82 % | as above |

- **Overstated, if anything:** Nsight Systems inflates the host part. In the fp32 profile, events gave 0.05–0.08 ms
  of idle time per step where nsys showed 0.22–0.24.
- **Stop rule:** at about 9,000 atoms (a) is 0.24–0.50 %, under 2 %. A GPU-resident loop could at best recover this.
  **Dead.**

**(b) Gaps inside the force graph** (Nsight Systems, every graph node, 10 replays):

| Cell | Kernels per replay | Replay span ms | Gaps µs per replay | Share |
|---|---|---|---|---|
| NbCl5 9,216 | 181 | 82.84 | 59.9 | **0.07 %** |
| NaNbO3 8,640 | 187 | 107.91 | 61.2 | **0.06 %** |
| NbCl5 1,944 | 195 | 17.43 | 89.6 | 0.51 % |
| NaNbO3 1,080 | 208 | 12.91 | 102.8 | 0.80 % |

- **Gap sizes:** almost all are under 5 µs (NbCl5 9,216: 167 under 2 µs, 11 of 2–5 µs, 2 of 5–20 µs per replay).
- **Against 28 October:** the fp32 profile measured about 1 % on the 3×3×3 cells and 5.4 % on NaCl. Current main is
  lower at 1,000–2,000 atoms.
- **Stop rule:** under 3 % on every cell, so a megakernel could recover at most this. **Dead.**

## Item 4: `edge_outer_sum` backward and interaction 0's radial-table backward

Per step, from the whole-step trace:

| Cell | `edge_outer_sum` backward µs | `radial_table_backward` µs | Sum µs | Share of step |
|---|---|---|---|---|
| NbCl5 9,216 | 407.5 | 14.5 | 422.0 | 0.50 % |
| NaNbO3 8,640 | 575.7 | 47.5 | 623.2 | 0.57 % |
| NbCl5 1,944 | 47.5 | 3.6 | 51.1 | 0.29 % |
| NaNbO3 1,080 | 30.6 | 6.1 | 36.7 | 0.28 % |

- **Stop rule:** at about 9,000 atoms the sum is above 100 µs, so the item is not stopped by its rule.
- **Ceiling:** fusing `dr` into the `edge_outer_sum` backward would remove the table backward's launch (15–48 µs)
  and part of the backward's edge reads. That is under 0.5 % of the step, consistent with the list's estimate of
  0.5–1 %.
- **Scaling:** the `edge_outer_sum` backward grows faster than the edge count, 8.6× for 4.6× the edges from 1,944 to
  9,216 atoms. That points at L2 spill in that kernel and could be worth a look on its own. Its whole time is 0.5 %
  of the step, though.

## Records

`results/l2-gap-audit-20261031/local-5060ti/<cell>/` holds, per cell:
- `summary.json` and `stages.csv`;
- `ncu_kernels.csv` and `ncu_categories.csv` (large cells);
- the Nsight Systems kernel and NVTX summaries;
- `nsys_pass.json` and `nsys-steps_pass.json`;
- `streams.txt` (idle holes).

Also: `m1_table.py` (this report's M1 tables) and `run.sh` (the launcher), with the launcher's `run.log` in
`local-5060ti/`. The Nsight reports and SQLite exports (several GB) stay in WSL
`~/fastmace-runs/l2-gap-audit-20261031/local/`.
