# Summary of the rejected-optimisation catalogue (11 November 2026)

Rows 288, with an agent record 288, check problems 4 (see `check.txt`).

Source: every index row of `docs/optimisation-ledger.md` (main, 11 November 2026) whose status is rejected, stopped, dropped, cut, closed or not built, or adopted on some GPUs and rejected on others. Eleven Haiku subagents read each row, its fuller entry and its linked report (brief: `INSTRUCTIONS.md`) and recorded what the records say; they did not judge the rejections. `merge.py` parses idea, status, date and links from the ledger itself and compares the agents' file and code checks with its own.

Reliability: kind, stage, GPUs, reason and flags are the agents' reading. One random row per part was compared with the ledger by hand: status and reason matched in all 11, the stage label was debatable in 2. Treat the flags as a filter for what to read, not as findings. Inconsistencies the agents reported in the ledger are in `agent-notes.md`. The full table is `catalogue.md`; the data are `catalogue.jsonl`.

| Kind | Rows |
| --- | ---: |
| REJECTED | 189 |
| NOT_BUILT | 55 |
| CLOSED | 18 |
| PARTIAL | 12 |
| STOPPED | 10 |
| OTHER | 2 |
| DROPPED | 2 |

| Stage where it stopped | Rows |
| --- | ---: |
| lab | 92 |
| ab | 90 |
| bound | 80 |
| gate | 13 |
| unknown | 13 |

| Flag | Rows |
| --- | ---: |
| stand_in | 102 |
| single_gpu | 68 |
| no_build | 67 |
| mixed | 43 |
| tie | 27 |
| resources | 26 |
| retest_noted | 25 |
| accuracy | 20 |
| superseded | 9 |
| unexplained | 4 |

| Month of the decision | Rows |
| --- | ---: |
| 2026-10 | 154 |
| 2026-11 | 131 |
| none | 3 |

| Area | Rows |
| --- | ---: |
| OMAT-0 on the A100 | 113 |
| Precision and tensor cores | 43 |
| Atom-level linears and GEMMs | 26 |
| Interaction-2 convolution | 22 |
| Radial networks and tables | 20 |
| Launches, graphs, scheduling and megakernels | 14 |
| Other models | 13 |
| Integrator, engine and neighbour list | 13 |
| Tiled call | 6 |
| Atom-level elementwise and gates | 5 |
| Symmetric contraction | 5 |
| Setup, compilation and start-up | 4 |
| Alternative backends and external comparisons | 3 |
| Interaction 1 | 1 |

GPUs named in the index row's status or effect (a row can name several; 167 rows name none):

| GPU | Rows |
| --- | ---: |
| RTX 5060 Ti | 58 |
| A100 | 57 |
| H200 | 54 |
| RTX 4000 Ada | 20 |

## Rows by flag, among those that were timed or gated (stage lab, ab or gate)

195 of the 288 rows. The flags come from the agents' reading and are not verified row by row.

### `unexplained` (1)

`model-ffwd-polar`

### `accuracy` (18)

`rad-values-only-fit`, `sc-tf32-bwd`, `prec-tf32`, `prec-cutlass-a100`, `prec-precision-screen`, `prec-fp16-edge-weights`, `prec-delta-gemm`, `prec-channel-split`, `prec-packed-b`, `prec-gemmul8`, `prec-emu-40`, `prec-double-float`, `prec-d1-48`, `launch-pdl`, `model-spin-restricted`, `ext-jax-mliap`, `omat0-fp16x3-gemm`, `tf32-fp16-cross-gemm`

### `resources` (20)

`lin-split-k`, `rad-k64-addr`, `conv-chunked-recompute`, `prec-setmaxnreg`, `ext-lunatp`, `omat0-cutlass-tile`, `omat0-tc-presplit`, `omat0-wprime-piece-order`, `omat0-fp16-tc`, `omat0-wprime-maxreg-a100`, `omat0-hybrid16`, `omat0-atom-order`, `ideas-r9`, `ideas-r11`, `kr-x-nodep`, `wprime-smem-spill`, `deferred-cg-forward`, `fp64-second-pipe`, `wprime-receiver-adjoint`, `cx-n1-cartesian-wprime`

### `tie` (26)

`lin-split-k`, `fp32-followups-20261029` (×5), `rad-radial-entry`, `model-ffwd-polar`, `model-mh1-edge-sh`, `model-polar-shared-geometry`, `model-polar-e3nn-block-linear`, `tiled-piece-start`, `omat0-eos-chunks`, `omat0-first-table-columns`, `kr-redistribute`, `stage-quads`, `packed-coef`, `factor-w`, `tabled-forward-maxnreg`, `cursor32`, `shared-locator`, `linear-div-split`, `h200-cutlass-tile`, `ada-5060-set` (×3)

### `mixed` (43)

`lin-concurrent-branches`, `fp32-followups-20261029` (×2), `rad-fused-radial-mlp`, `rad-f64-reshape`, `conv-own-triton-conv`, `conv-fused-radial-conv`, `prec-cublasdx-int8`, `prec-fp16x3-standalone`, `prec-hybrid-gemm`, `prec-bk32-swizzle`, `prec-fp32-baselines`, `prec-presplit-epilogue`, `prec-setmaxnreg`, `launch-combo-persub`, `launch-two-streams`, `model-off-screens-a100`, `model-polar-phase-recurrence`, `model-small-matmuls`, `eng-skin-rescreens`, `setup-max-autotune`, `omat0-slope-vector`, `omat0-stream-w`, `omat0-producer-consumer`, `omat0-hybrid16`, `omat0-species-gemm-a100-recheck`, `omat0-table-w-forward`, `ideas-r5`, `ideas-r7`, `ideas-r9`, `ideas-r11`, `ideas-r12`, `ideas-r14`, `ideas-r3`, `kr-x-nodep`, `kr-wprime-vector-atomics`, `fusion-receiver-local`, `sx-forward-table-w`, `sx-wprime-reduction`, `mirror-a100`, `forward-ghoist`, `wprime-zero-byte`, `wprime-smem-spill`

### `retest_noted` (23)

`lin-gemm-autotune`, `lin-unused-heads`, `lin-cublaslt`, `rad-contiguous-folded-weight`, `conv-receiver-backward`, `prec-tf32`, `prec-tf32x3-triton`, `prec-cutlass-a100`, `prec-bf16x6`, `model-mh1-edge-sh`, `model-small-matmuls`, `eng-bucket-width`, `tiled-receiver-cap`, `ext-lunatp`, `omat0-producer-consumer`, `omat0-ff-prefetch`, `rescreen-20261109`, `forward-ghoist`, `wprime-smem-spill`, `virial-elementwise`, `omat0-small-fold-alpha`, `ada-5060-set` (×2)

### `single_gpu` (60)

`lin-split-k`, `elt-gate-prologue`, `prec-cutlass-a100`, `prec-deepseek-x3`, `prec-fp16x3-probe`, `prec-fp16x3-standalone`, `prec-int8-f32`, `launch-combo-persub`, `model-mh1-edge-sh`, `model-mh1-fused-first`, `model-polar-wprime-a100`, `model-off-screens-a100`, `model-polar-phase-recurrence`, `model-small-matmuls`, `model-polar-shared-geometry`, `model-polar-e3nn-block-linear`, `tiled-receiver-cap`, `tiled-device-pool`, `omat0-first-full-rows`, `omat0-column-block`, `omat0-eos-chunks`, `omat0-cutlass-tile`, `omat0-wprime-piece-order`, `omat0-wprime-aprime`, `omat0-producer-consumer`, `omat0-wprime-maxreg-a100`, `omat0-mma-chain`, `omat0-radial-table-shapes`, `omat0-tp-factor`, `omat0-fp16x3-gemm`, `omat0-atom-order`, `kr-redistribute`, `kr-tabled-warps`, `kr-species-cutlass-h200`, `sx-wprime-reduction`, `mirror-a100`, `stage-quads`, `packed-coef`, `factor-w`, `bspline-table`, `tabled-forward-maxnreg`, `deferred-cg-forward`, `wprime-split-sums`, `cursor32`, `shared-locator`, `linear-div-split`, `r128-min-blocks`, `h200-ldg-redux`, `h200-cutlass-tile`, `cx-n1-cartesian-wprime`, `cx-n2-fp64-adjoint`, `rungb-first-layer`, `half-system-streams`, `tf32-fp16-cross-gemm`, `ada-5060-set` (×6)

### `stand_in` (86)

`lin-concurrent-branches`, `lin-cublaslt-heuristic`, `fp32-followups-20261029` (×2), `elt-gated-segment-gemm`, `rad-triton-folded-forward`, `rad-cuda-gemm`, `rad-write-bound`, `rad-radial-last-gemm`, `conv-segmented-reduction`, `conv-oeq-weight-prologue`, `conv-fused-radial-conv`, `conv-full-fusion`, `prec-tf32x3-triton`, `prec-cutlass-a100`, `prec-bf16x6`, `prec-cublasdx-int8`, `prec-tc-overlap`, `prec-deepseek-x3`, `prec-fp16x3-probe`, `prec-fp16x3-standalone`, `prec-fp16x3-rowcol`, `prec-fp16x3-other-gpus`, `prec-hybrid-gemm`, `prec-packed-b`, `prec-decoupled`, `prec-two-kernel-split`, `prec-bk32-swizzle`, `prec-tensor-only-sweep`, `prec-presplit-epilogue`, `prec-setmaxnreg`, `prec-cutlass-sm120`, `prec-producer-split`, `prec-gemmul8`, `prec-int8-f32`, `prec-double-float`, `launch-het-megakernel`, `launch-sched-iso`, `launch-cublasdx`, `launch-two-streams`, `eng-species-morton`, `eng-rebuild-cost`, `ext-lunatp`, `omat0-tc-presplit`, `omat0-wprime-piece-order`, `omat0-wprime-aprime`, `omat0-fp16-tc`, `omat0-producer-consumer`, `omat0-wprime-maxreg-a100`, `omat0-wprime-split`, `omat0-mma-chain`, `omat0-hybrid16`, `omat0-ff-prefetch`, `omat0-radial-table-shapes`, `omat0-tp-factor`, `omat0-fp16x3-gemm`, `omat0-tc-tp`, `omat0-ff-hints`, `omat0-atom-order`, `omat0-table-w-forward`, `ideas-r5`, `ideas-r9`, `ideas-q2`, `ideas-r3`, `kr-x-nodep`, `kr-redistribute`, `kr-wprime-restrict`, `kr-wprime-vector-atomics`, `fusion-receiver-local`, `sx-forward-table-w`, `sx-wprime-reduction`, `quad-table-bound`, `tc-edge-tp`, `mirror-a100`, `bulk-slab`, `bspline-table`, `forward-ghoist`, `wprime-zero-byte`, `deferred-cg-forward`, `wprime-split-sums`, `fp64-second-pipe`, `wprime-receiver-adjoint`, `cx-n1-cartesian-wprime`, `cx-n2-fp64-adjoint`, `half-system-streams`, `tf32-fp16-cross-gemm`

## Timed before 5 November 2026 and the switch or tool still exists in the code (25)

These were measured against older defaults (the BUTTERFLY re-screen of 9 November flipped one such rejection).

`lin-inplace-block-linear`, `lin-gemm-autotune`, `lin-split-k`, `rad-contiguous-folded-weight`, `rad-joint-fused-backward`, `conv-receiver-backward`, `fp32-followups-20261029` (×5), `prec-tf32`, `prec-bf16x3-facto`, `eng-bucket-width`, `omat0-first-full-rows`, `omat0-column-block`, `omat0-slope-vector`, `omat0-stream-w`, `omat0-stream-store-w`, `omat0-eos-chunks`, `omat0-tc-presplit`, `omat0-species-gemm-tc`, `omat0-species-gemm-cutlass`, `omat0-wprime-aprime`, `omat0-first-table-columns`

