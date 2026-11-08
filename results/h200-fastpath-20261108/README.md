# Codex H200 report (8 November): COMPOSE_SKIP fast path, generic readout-tail fold, linear_div split, cuEq projection, staged edges

8 November 2026. Modal H200 (`NVIDIA H200`, driver 610.57.04, sm_90), profile `sixth`, packed jobs. OMAT-0 medium,
SrTiO3 10x10x10 (5,000 atoms), Benchmark_loup settings (Langevin 300 K, 1 fs, skin 0.5, seed 1). A/B: a = the H200
defaults of main `909bc8cd` (plus the switches adopted before the round), b = the variant, `--warmup 200 --block 256
--blocks 6`, both build orders.

Branches: `fastmace/h200-fastpath` (bitwise class: `COMPOSE_FAST`, `LINEAR_DIV_SPLIT`, `CUEQ_PROJECTED`,
`STAGE_EDGES`, `STAGE_SH`) and `fastmace/h200-fold-tail` (rounding class: `GENERIC_FOLD_READOUT_TAIL`). The jobs ran on
a merge of both (`fastmace/h200-fastpath-job`, not for merging).

## Round 1 (`tools/modal/fastpath_job.sh`, source `3c5ed795`, [round1/](round1/run.log))

| Switch | What | MD step b/a (ab / ba) | Class, check | Decision |
| --- | --- | --- | --- | --- |
| `compose_skip.COMPOSE_FAST` | COMPOSE_SKIP's composed weights bound once per table (refreshed in place), the four degree products as one `species_mm_joined` (CUTLASS species_into per path into one buffer), the division returns a view | **0.9614 / 0.9656** (4.015 → 3.860 ms) | bitwise; forces 3.3e-6 against same-build 2.7e-6 eV/Å | **adopted (H200)** |
| `linear_div.LINEAR_DIV_SPLIT` | linear_div's backward writes the divided path blocks contiguously in one Triton kernel | 0.9972 / 1.0025 | bitwise | not faster in both orders: off, closed |
| `cueq_projection.CUEQ_PROJECTED` | cuEq SymmetricContraction's `einsum(weight, projection)` kept as a buffer instead of recomputed per call | **0.9902 / 0.9954** | rounding (einsum order); gate passed: forces 0.922 / 0.900 / 0.979, stress 0.425 / 0.852 / 1.206 | **adopted (H200)** |
| `species.GENERIC_FOLD_READOUT_TAIL` | the generic product's species skip composed with the following readout linear | **0.9855 / 0.9901** | rounding; gate passed: forces 0.848 / 0.920 / 0.968, stress 0.437 / 0.824 / 1.043 | **adopted (H200)** |
| all four | | 0.9497 / 0.9509 (4.018 → 3.816 ms) | | |

The joined species_mm is bitwise against per-path `species_mm` (`tests/gpu/test_species_mm_joined_cuda.py`, eager and
compiled, ieee / tf32x3 / tf32x3-cutlass: 6 passed on the H200, [pytest-joined.log](round1/pytest-joined.log)). The
gate ratios are SrTiO3 / NaCl / NbCl5 against the ASE float32 reference, limit 3
([gate-tail](round1/gate-tail/summary.txt), [gate-cueq](round1/gate-cueq/summary.txt)).

## Round 2 (`tools/modal/fastpath2_job.sh`, source `61d752e6`, [round2/](round2/run.log))

a = the H200 defaults with round 1's three adoptions (3.78 ms per MD step).

| Switch | What | MD step b/a (ab / ba) | Build, check | Decision |
| --- | --- | --- | --- | --- |
| `spline_forward.STAGE_EDGES` | the spline forward (one receiver per block, thread = channel) stages its edges per 32-edge tile: warp 0 does cutoff, sender and spline lookup once per edge with a ballot compaction, all threads copy the edge features coalesced; the channel loop reads shared memory ([source](round2/spline_forward-stage-edges.cu)) | **0.9864 / 0.9856** (3.780 → 3.728 ms) | 80 registers, **0 local bytes** (8 before); forces 3.0e-6 against same-build 2.7e-6 eV/Å, energy diff 1.3e-5 against same-build 2.5e-5 eV | **adopted (H200)** |
| `wprime_backward.STAGE_SH` | each 32-lane w′ cohort copies its edge's features to shared memory once per edge ([source](round2/wprime-stage-sh.cu)) | **0.9955 / 0.9950** | 72 registers, 0 local bytes (unchanged); forces 2.6e-6 against 2.8e-6, energy 3.7e-5 against 4.7e-5 | **adopted (H200)**: bitwise, faster in both orders |
| both | | **0.9819 / 0.9811** (3.776 → 3.704 ms) | | |

Both are bitwise by construction (same arithmetic in the same order; only where the operands are loaded from).

## Step breakdown of the round-1 defaults (before STAGE_EDGES / STAGE_SH)

`tools/step_breakdown.py`, 200 steps after 100 of equilibration, [round2/breakdown/](round2/breakdown/ops.csv).
Per force call, 3.51 ms of kernel time:

| Kernel | µs per call |
| --- | ---: |
| w′ backward (`oeq_projected_backward_wprime`, 72 registers, 44 % occupancy) | 1,106 |
| spline forward (80 registers, 38 %) | 860 |
| tabled outer sum backward (density) | 213 |
| tf32x3 GEMMs in the compiled graph (6 launches) | 155 |
| `tf32x3_linear_div` (4 launches) | 147 |
| `species_mm_joined` + its gradient (COMPOSE_FAST) | 104 + 106 |
| cuEq symmetric contraction (forward + backward, 4 kernels) | 167 |
| tabled outer sum forward | 114 |
| radix sort (8 launches) | 43 |

The two streaming kernels are 56 % of the step; everything at the atom level together is about 0.8 ms.

## Round 3 (`tools/modal/fastpath3_job.sh`, source `7e0711da`, [round3/](round3/run.log))

a = the H200 defaults with rounds 1 and 2 (3.71 ms per MD step). The step breakdown above showed COMPOSE_SKIP's
species plan as an int64 radix sort of 5,000 atoms per force call: 8 onesweep passes plus 10 memsets, 57 µs.

| Switch | What | MD step b/a (ab / ba) | Check | Decision |
| --- | --- | --- | --- | --- |
| `compose_skip.COUNT_PLAN` | `species_plan(..., counting=True)`: the same stable order by a counting sort (one-hot prefix sums, a scatter), fused by the compiled caller | **0.9831 / 0.9818** (3.715 → 3.652 ms) | forces 2.9e-6 against same-build 2.9e-6 eV/Å; bitwise by construction (`tests/cpu/test_compose_skip.py`: equal to the stable argsort, eager and compiled) | **adopted (H200)** |
| `spline_forward` R128 shape with `MIN_BLOCKS` 7 | register cap for 7 blocks per SM | 1.0097 / 1.0103 | no spill reported | slower: closed, the cap stays 6 |
| `MIN_BLOCKS` 8 | | 1.0515 / 1.0507 | | slower: closed |

Breakdown of the round-2 defaults ([round3/breakdown/](round3/breakdown/ops.csv)): 3.44 ms of kernel time per force
call (3.51 before STAGE_EDGES / STAGE_SH); w′ 1,091 µs, spline forward 808 µs (860 before STAGE_EDGES).

## Round 4 (`tools/modal/fastpath4_job.sh`, source `978dbef5`, [round4/](round4/run.log))

a = the H200 defaults on main `65923679` (rounds 1-3 adopted; 3.65 ms per MD step). Two existing w′ switches never
measured on the H200:

| Switch | What | MD step b/a (ab / ba) | Check | Decision |
| --- | --- | --- | --- | --- |
| `wprime_backward.BLOCK_LDG` | the block kernel's gradient read through `__ldg` | 1.0012 / 1.0001 | forces 2.9e-6 against same-build 2.7e-6 eV/Å (bitwise) | tie: closed, stays off |
| `wprime_backward.REDUX` | end-of-edge reduction in fixed point with `redux.sync` (deterministic, rounding) | 1.0206 / 1.0202 | gate run ([round4/gate-redux/](round4/gate-redux/)) | slower: closed, stays off |

Breakdown of the main defaults ([round4/breakdown/](round4/breakdown/ops.csv)): 3.40 ms of kernel time per force
call; w′ 1,099 µs, spline forward 806 µs, tabled outer-sum backward 214 µs, the 3×TF32 GEMMs (18 launches) about
515 µs at 18 % occupancy; the species plan's radix sort is gone (COUNT_PLAN).

## Round 5 (`tools/modal/fastpath5_job.sh`, source `32dff94a`, [round5/](round5/run.log))

| Switch | What | MD step b/a (ab / ba) | Check | Decision |
| --- | --- | --- | --- | --- |
| `cutlass.tf32x3.TILE` = `"32x64"` | two-warp output tile, twice the blocks (632 instead of 316 per product on 132 SMs) | 1.0023 / 1.0030 | forces 3.0e-6 against same-build 2.8e-6 eV/Å (bitwise) | slower: closed |
| `TILE` = `"64x32"` | | 1.0014 / 1.0035 | 2.7e-6 against 2.6e-6 | slower: closed |

As on the A100 (1.4-1.7 % slower there), the 64x64 tile stays.

## Total

MD step on the H200, OMAT-0 medium SrTiO3 10x10x10: 4.018 ms (main `909bc8cd` H200 defaults, round 1's a side) →
3.816 (round 1) → 3.704 (round 2) → about 3.65 ms (round 3): about −9 %. Rounds 4 and 5 found nothing more among the existing switches.
