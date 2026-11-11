# Rejected, stopped and not-built optimisations in the ledger (catalogue, 11 November 2026)

288 index rows of `docs/optimisation-ledger.md` whose status is rejected, stopped, dropped, cut, closed or not built (or adopted on some GPUs and rejected on others). Idea, status and date are parsed from the ledger; stage, GPUs, reason and flags were read from the entries by Haiku subagents and are **not verified** beyond the checks in `check.txt`. Flags are defined in `INSTRUCTIONS.md`.

## Atom-level linears and GEMMs (26)

| Line | ID | Idea | Status | Date | Stage | Tested on | Reason | Flags |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 90 | `lin-inplace-block-linear` | In-place / strided-batched block linear | REJECTED | 2026-10-14 | ab |  | Slower. Rewrites note: cuBLAS batched tiles make the l = 3 product 415 vs 305 µs. |  |
| 91 | `lin-grouped-node-gemms` | Grouped node GEMMs (RE) | NOT BUILT | 2026-10-14 | bound |  | Ceiling too small. | no_build, stand_in |
| 92 | `lin-cueq-fused-tp` | cuEq `fused_tp` for linears | REJECTED | 2026-10-07 | ab |  | No gain; under compilation the linears were already fused. |  |
| 94 | `lin-gemm-autotune` | GEMM-only and broad Inductor autotuning | REJECTED | 2026-10-09 | ab |  | No gain on the rewritten model; per-shape tuning is excluded by the production rule. | retest_noted |
| 98 | `lin-fusion-mechanisms` | Inductor templates, cuBLASLt epilogues, CUTLASS EVT, cuBLASDx | NOT BUILT | 2026-10-20 | bound |  | Dropped on paper: Inductor needs per-shape autotuning; cuBLASLt lacks sigmoid-from-other-operand and gathered rows; cuBLASDx needs MathDx headers. | no_build |
| 99 | `lin-concurrent-branches` | Per-l products on concurrent graph branches | REJECTED | 2026-10-20 | lab | murphy, local | Branches cost more than they overlap. | stand_in, mixed |
| 102 | `lin-split-k` | Split-K for small-cell products | REJECTED | 2026-10-21 | ab | local | Extra launch and partial-sum traffic cost about what the split saves. | single_gpu, tie, resources |
| 103 | `lin-split-k-triton` | Triton split-K with epilogue reduction | NOT BUILT | 2026-10-21 | bound |  | Not worth building. | no_build, stand_in |
| 104 | `lin-checkpoint-scan` | Scan checkpoints for exact zero skips | NOT BUILT | 2026-10-21 | bound | CPU | The only exact skips are worth under 0.5 % together. | no_build |
| 105 | `lin-zero-channels` | Drop MH-1's 24 zero l = 1 channels | NOT BUILT | 2026-10-21 | bound |  | Below the noise floor; needs its own record, self-check and pins. | no_build, retest_noted |
| 106 | `lin-unused-heads` | Single-head model / skip unused readout heads | REJECTED | 2026-10-08 | ab |  | Below noise. | retest_noted, superseded |
| 112 | `lin-t08-c6` | No gradient w.r.t. `node_attrs` | NOT BUILT | 2026-10-21 | bound |  | Comes with C1: tables take no gradient. | no_build, superseded |
| 113 | `lin-t08-c4` | One product for `skip_tp` and `linear_up` | NOT BUILT | 2026-10-21 | bound |  | C2 and C3 replace both products. | no_build, superseded |
| 114 | `lin-t08-c5` | Drop the `node_feats` descriptor concatenation | NOT BUILT | 2026-10-21 | bound |  | Gone with C3. | no_build, superseded |
| 115 | `lin-t08-c7` | Drop unused heads of `readouts.0.linear` | NOT BUILT | 2026-10-21 | bound |  | Absorbed by C3's consumer fold and C2. | no_build, superseded |
| 116 | `lin-t08-headslice` | C2 with a one-head slice | NOT BUILT | 2026-10-21 | bound |  | Needs a fixed head; small. | no_build |
| 117 | `lin-t08-generic` | C1/C2 for residual, density, OFF families | NOT BUILT | 2026-10-21 | bound |  | Those families skip_tp is a tensor product with node_attrs; folding as built would add a product. | no_build |
| 118 | `lin-t08-paper` | Other linear folds checked on paper | NOT BUILT | 2026-10-21 | bound |  | More arithmetic (K = 512 vs 128), non-commuting operations, nonlinear on both sides, or already merged. | no_build |
| 119 | `lin-forward-audit` | Further consecutive-linear folds (round 5) | NOT BUILT | 2026-10-26 | bound |  | Nonlinear boundaries around linear_2 block it. | no_build |
| 120 | `lin-faster-fp32` | A 1.2× faster FP32 SIMT GEMM | NOT BUILT | 2026-10-21 | bound |  | No faster FP32 kernel found without TF32. | no_build, stand_in |
| 121 | `lin-layout-copies` | Remove block-linear row gathers, joins, fallback copy | NOT BUILT | 2026-10-21 | bound |  | Not exact: needs a gathered-row GEMM (rounding; slower on NaCl) or a different cuBLAS kernel choice. | accuracy |
| 122 | `lin-cublaslt` | cuBLASLt as preferred BLAS | REJECTED | 2026-10-08 | ab |  | Faster in isolation, no consistent gain in the step; owner: about 1 % on some cases is not enough. | retest_noted |
| 123 | `lin-cublaslt-heuristic` | Per-shape cuBLASLt algorithm search | REJECTED | 2026-10-26 | lab |  | Unqualified prototype; a shape-keyed cache breaks the generality rule. | stand_in |
| 124 | `lin-svd` | Low-rank atom linears (truncated SVD) | REJECTED | 2026-10-27 | bound |  | Falsified: no truncated-SVD route; rank 509–512 of 512. | no_build |
| 127 | `fp32-followups-20261029` | `segment_gemm` 128-row tiles, 8 warps | REJECTED | 2026-10-29 | ab | RTX 5060 Ti, RTX 4000 Ada | No reason stated separately; the record reports measured ratios 0.999–1.001 on both GPUs. | tie |
| 128 | `fp32-followups-20261029` | int32 addressing in `segment_gemm` (A2) | REJECTED | 2026-10-29 | lab | RTX 5060 Ti, RTX 4000 Ada | No reason stated separately; isolated probe 1.013 Ada and 0.981 local against a 10 % stop rule. | stand_in, mixed |

## Atom-level elementwise and gates (5)

| Line | ID | Idea | Status | Date | Stage | Tested on | Reason | Flags |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 135 | `elt-gate-vjp` | Fused gate/residual VJP | NOT BUILT | 2026-10-10 | bound |  | Already fused at DRAM bandwidth; a custom op would likely lose Inductor neighbouring fusions. | stand_in, resources, unexplained, retest_noted |
| 136 | `elt-gate-prologue` | Gate inside `segment_gemm`'s prologue (stage C) | REJECTED | 2026-10-20 | ab | local | Conductor rule (NbCl5 ≤ 0.985 both orders, both GPUs) not met; per-atom gate gather in the K loop cost the GEMMs more than the removed kernel. | single_gpu |
| 137 | `elt-gate-backward-epilogue` | Gate backward in the input-gradient epilogue (stage D) | REJECTED | 2026-10-20 | bound |  | Per-atom reduction needs atomics or whole-atom tiles, breaking determinism and the float64 bitwise promise. | no_build |
| 138 | `elt-density-prologue` | Density division in the residual GEMM prologue (stage E) | NOT BUILT | 2026-10-20 | bound |  | Absorbed: already fused with the slice copies by Inductor; disappears with A/B. | no_build, superseded |
| 139 | `elt-gated-segment-gemm` | Gate scaling in the `segment_gemm` prologue (round 5) | CUT | 2026-10-26 | lab |  | Below the 5 % bar (−3.7 % < 5 %); the backward still needs the gated rows. | stand_in |

## Radial networks and tables (20)

| Line | ID | Idea | Status | Date | Stage | Tested on | Reason | Flags |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 154 | `rad-weight-sharing` | Same-species radial weight sharing | NOT BUILT | 2026-10-10 | bound |  | Not pursued. | stand_in |
| 155 | `rad-mirror-edge` | Mirror-edge radial sharing | NOT BUILT | 2026-10-15 | bound |  | Slower at the ceiling. | stand_in |
| 156 | `rad-contiguous-folded-weight` | Contiguous folded weight | REJECTED | 2026-10-14 | ab |  | No full-step gain. | retest_noted |
| 157 | `rad-decompose-mm` | Inductor `decompose_mm_pass` | REJECTED | 2026-10-14 | ab |  | No gain. |  |
| 158 | `rad-joint-fused-backward` | Triton input gradient of the joint layer | REJECTED | 2026-10-14 | ab |  | Below 0.1 ms, slower on small cells. |  |
| 159 | `rad-triton-folded-forward` | Triton forward for the folded layer (RE) | REJECTED | 2026-10-14 | lab |  | Slower. | stand_in |
| 160 | `rad-cuda-gemm` | Hand-written CUDA GEMM for the folded forward | REJECTED | 2026-10-14 | lab |  | Slower than cuBLAS. | stand_in |
| 161 | `rad-fused-radial-mlp` | Fused radial MLP, store variant | REJECTED | 2026-10-21 | ab | RTX 5060 Ti | Commit deaa5f90: 'NaCl −1.5 %, NbCl5 +1.0 %: not adopted'; opt-in on its branch, not on main. | mixed |
| 163 | `rad-radial-entry` | Radial MLP entry kernels (OPT2-T02c) | REJECTED | 2026-10-21 | ab |  | Conductor's rule: NbCl5 not below 1.0 in both orders; the stack's isolated −0.34 ms did not reach the step. | tie |
| 164 | `rad-fused-mlp-retry` | Second radial-MLP fusion attempt | NOT BUILT | 2026-10-24 | bound |  | OPT2-T02's cause (opaque op costs Inductor's neighbouring fusions) stands. | no_build |
| 166 | `rad-write-bound` | Write-bound Triton kernel for the folded forward | CUT | 2026-10-24 | lab |  | Compute-bound, not write-bound; even cuBLAS at K = 64 misses the limits. | stand_in |
| 167 | `rad-k64-addr` | cuBLAS K = 64 plus rank-1 update | REJECTED | 2026-10-24 | lab |  | The rank-1 update re-reads and re-writes the whole [E, 1280] output. | resources |
| 168 | `rad-radial-last-gemm` | `64 + tail` streaming-store kernel | CUT | 2026-10-26 | lab |  | Even cuBLAS at K = 64 misses the targets (repeats OPT4-T01). | stand_in |
| 171 | `rad-values-only-fit` | Values-only table fit | REJECTED | 2026-10-27 | gate | CPU | The derivative error failed the certificate. | accuracy, superseded |
| 177 | `rad-w-tables-auto` | D2/D3 by measured FP64 rate instead of GPU names | NOT BUILT | 2026-10-28 | unknown |  | Owner rule 'only where measured'; owner chose the exact GPU-name list (decision 65). | no_build |
| 178 | `rad-w-tables-r1` | Keep only w-table pieces with r ≥ 1 Å | NOT BUILT | 2026-10-28 | bound |  | Memory caveat. | no_build |
| 179 | `rad-low-rank` | Certified low-rank radial / w basis (D4, F1) | REJECTED | 2026-10-27 | bound |  | Full rank. | no_build |
| 181 | `rad-f0` | Float32 input-gradient tiles in the chunked backward (F0) | REJECTED | 2026-10-27 | bound |  | The side stream already hides the input gradient. | stand_in |
| 182 | `rad-f64-reshape` | Reshaped folded forward product in float64 | REJECTED | 2026-10-27 | lab |  | No gain. | mixed |
| 183 | `fp32-followups-20261029` | Backward-only w′ table, real operands (E4) | REJECTED | 2026-10-29 | lab | RTX 5060 Ti |  | mixed, stand_in |

## Interaction 1 (1)

| Line | ID | Idea | Status | Date | Stage | Tested on | Reason | Flags |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 235 | `int1-lm-major` | (l, m)-major outer sums | REJECTED | 2026-10-21 | lab |  | The transpose moved to the join and gradient slices instead of disappearing. |  |

## Symmetric contraction (5)

| Line | ID | Idea | Status | Date | Stage | Tested on | Reason | Flags |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 241 | `sc-oeq-contraction` | OEq 0.7 SymmetricContraction instead of cuEq | REJECTED | 2026-10-16 | unknown |  | Much slower. |  |
| 242 | `sc-reduced-cg` | Reduced Clebsch–Gordan basis (mace-reforge) | REJECTED | 2026-10-16 | bound |  | Nothing to gain. | no_build |
| 245 | `sc-opt3-t02` | Symmetric contraction screen (OPT3-T02) | NOT BUILT | 2026-10-21 | bound | H200 | Memory-bound at its traffic floor. | resources, stand_in |
| 246 | `sc-tf32-bwd` | TF32 backward of interaction 0's contraction | NOT BUILT | 2026-10-24 | gate |  | Fails on another family; production uses cuEquivariance's kernel, which has no TF32 switch. | accuracy |
| 247 | `sc-emulated-gemm` | Contraction as one emulated GEMM (D5) | NOT BUILT | 2026-10-27 | bound |  | Traffic exceeds the time it would save. | no_build, resources |

## Interaction-2 convolution (22)

| Line | ID | Idea | Status | Date | Stage | Tested on | Reason | Flags |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 190 | `conv-segmented-reduction` | Segmented reduction operator (idea 5) | NOT BUILT | 2026-10-09 | lab |  | The permutation cost dominates; sorting was adopted instead. | stand_in |
| 191 | `conv-oeq-schedules` | OEq launch schedules warp4/warp5/blocks2 | REJECTED | 2026-10-09 | lab |  | Slower or no change. |  |
| 193 | `conv-l2-chunks-re` | L2-resident chunks with unpatched OEq | REJECTED | 2026-10-14 | lab |  | Launch and copy overhead with stock OEq. |  |
| 194 | `conv-own-triton-conv` | Own fused Triton convolution | REJECTED | 2026-10-14 | lab |  | Slower on large cells. | mixed |
| 195 | `conv-oeq-weight-prologue` | OEq weight prologue (per-edge weight generation) | NOT BUILT | 2026-10-14 | lab |  | Slower than the GEMM. | stand_in |
| 199 | `conv-bwd-variants` | Streaming backward `__restrict__`, block sizes, register caps | REJECTED | 2026-10-14 | lab |  | Already near the copy floor. |  |
| 200 | `conv-receiver-backward` | Receiver-owned streaming backward | REJECTED | 2026-10-14 | ab |  | One block per receiver serialises the edges of each receiver and loses parallelism. | retest_noted |
| 202 | `conv-fused-radial-conv` | Radial last layer fused into streaming conv (OPT2-T01) | REJECTED | 2026-10-21 | lab |  | Receiver-owned layout forbids a register-tiled GEMM micro-kernel; the saved w traffic is spent again on in-kernel arithmetic. | mixed, stand_in |
| 203 | `conv-dw-in-conv` | Backward-only `dw` contraction inside the conv | NOT BUILT | 2026-10-21 | bound |  | Same E x 1280 x 65 multiply-adds in the layout where OPT2-T01 reached 18 % of FMA peak. | no_build |
| 204 | `conv-oeq-launch-shape` | Streaming forward launch shape | CUT | 2026-10-24 | bound |  | Already DRAM-bound. | resources |
| 205 | `conv-l2-hints` | L2 eviction hints in the streaming forward | NOT BUILT | 2026-10-24 | bound |  | Below any useful gain; template change. | no_build |
| 206 | `conv-radial-conv-reopen` | Radial producer / conv fusion via cuBLASDx | NOT BUILT | 2026-10-26 | bound |  | Its primitive failed gate 1 (launch-cublasdx); the SIMT form had failed twice. | no_build |
| 208 | `conv-chunked-recompute` | Chunked forward with recompute | REJECTED | 2026-10-26 | lab |  | The extra GEMM costs more than the traffic it saves. | resources |
| 214 | `conv-sobek` | Sobek per-receiver reassociation | REJECTED | 2026-10-27 | bound |  | The per-edge weight is a 65-wide linear map, not a short basis. | no_build |
| 216 | `conv-full-fusion` | Table → `w` tile → conv, forward and backward | REJECTED | 2026-10-27 | lab |  | The backward must rebuild w (three FFMA-bound products, issue-bound); the chain backward keeps dw in L2. | stand_in |
| 217 | `conv-edge-count-chunks` | Edge-count chunking instead of receiver chunks | NOT BUILT | 2026-10-27 | unknown |  | Would not close the gap to 0.75. |  |
| 219 | `conv-tc-wtile` | Tensor-core w-tile in the fused backward (F2) | NOT BUILT | 2026-10-27 | bound |  | Below 3 % before issue-slot and accuracy penalties (OPT5-T06/T08 lost on sm_120). | no_build |
| 222 | `conv-source-owned` | Source-owned backward (T01b) | NOT BUILT | 2026-10-27 | bound |  | The atomics it would remove cost at most 5 %; walking by sender gathers rows out of order. | no_build |
| 224 | `conv-path-grouping` | Path-group traffic reduction | NOT BUILT | 2026-10-27 | bound |  | Needs a receiver-owned loop to keep its accumulators, which loses more parallelism than it saves. | no_build |
| 225 | `conv-deriv-activity` | Derivative activity specialisation | NOT BUILT | 2026-10-27 | bound |  | Too small. | no_build |
| 226 | `conv-wide-loads` | `float2` loads in the streaming backward | NOT BUILT | 2026-10-27 | lab | sm_120, Ada | Below the owner's rule: about 1 % on only some cases does not change a default. |  |
| 227 | `fp32-followups-20261029` | `fused_folded_forward` 4 blocks per SM (O7) | REJECTED | 2026-10-29 | ab | RTX 5060 Ti, RTX 4000 Ada |  |  |

## Precision and tensor cores (43)

| Line | ID | Idea | Status | Date | Stage | Tested on | Reason | Flags |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 255 | `prec-tf32` | TF32 matrix products (global, selective) | REJECTED | 2026-10-08 | gate |  | accuracy (failed accuracy; off by design) | accuracy, retest_noted |
| 257 | `prec-fp32-emulation-re` | cuBLAS FP32 emulation on tensor cores (RE) | REJECTED | 2026-10-07 | unknown |  | unavailable on GeForce Blackwell (data-centre Blackwell only) |  |
| 258 | `prec-torch-modes` | torch FP32 matmul precision modes | REJECTED | 2026-10-20 | bound |  | no FP32-accurate mode exists; only ieee, tf32 and none offered | no_build |
| 259 | `prec-bf16x9` | cuBLAS BF16×9 FP32 emulation | REJECTED | 2026-10-20 | bound |  | absent on both production GPUs; cuBLAS documents compute capability 10.0 and 10.3 only | no_build |
| 260 | `prec-tf32x3-triton` | 3×TF32 GEMMs (RE probe; Triton `tf32x3`) | REJECTED | 2026-10-08 | lab | sm_89, sm_120, A100 | three TF32 products cannot beat SIMT FP32 when TF32 is at most 3× FP32 (sm_89 ceiling 38.6 / 3 < 16.3); on the A100 Triton's kernel is the limit | stand_in, retest_noted |
| 261 | `prec-cutlass-ada` | CUTLASS 3×TF32 on sm_89 / sm_120 | NOT BUILT | 2026-10-20 | bound |  | same arithmetic and ceiling as Triton's tf32x3 on these cards | no_build |
| 262 | `prec-cutlass-a100` | CUTLASS 3×TF32 on the A100 (stage 2) | NOT BUILT | 2026-10-24 | lab | A100-SXM4-40GB | fast but only FP32-class on typical data; sm_80 is not a production GPU | accuracy, retest_noted, single_gpu, stand_in |
| 263 | `prec-bf16x6` | Triton BF16×6 / BF16×3 | REJECTED | 2026-10-23 | lab | A100, G4, L4 | needs BF16 dense ≥ 6× FP32 and a better kernel than Triton's | retest_noted, stand_in |
| 265 | `prec-cublasdx-int8` | cuBLASDx INT8-slice FP32 emulation (`RequiredMantissaBits`, MathDx 26.10) | REJECTED | 2026-11-03 | lab | RTX 5060 Ti, RTX 4000 Ada | INT8 throughput on these cards is not 6–10× FP32; the production products are K = N = 512 | mixed, stand_in |
| 266 | `prec-bf16x3-facto` | BF16×3 atom GEMMs in the MD step (FACTO) | REJECTED | 2026-11-03 | ab | RTX 5060 Ti, RTX 4000 Ada, A100 | BF16 tensor cores are only 3–5× FP32 on the production GPUs; on the A100 the GEMM gain is lost with the grouped launch |  |
| 267 | `prec-tc-overlap` | Tensor-core GEMM beside SIMT on two streams | REJECTED | 2026-10-20 | lab |  | both GEMMs fill every SM, so the second stream only gets the tail | stand_in |
| 269 | `prec-hybrid-dispatcher` | Libra hybrid dispatcher speed target | NOT BUILT | 2026-10-26 | bound |  | depended on OPT5-T02's sm_80 kernel, never built; the A100 is not a production GPU | no_build |
| 270 | `prec-deepseek-x3` | SIMT-split producer + pure MMA consumer | CUT | 2026-10-26 | lab | sm_120 | on sm_120 TF32 equals the FP32 rate, so the producer cost exceeds the 3×TF32 gain | single_gpu, stand_in |
| 271 | `prec-precision-screen` | Per-stage bf16/fp16/TF32 sensitivity | REJECTED | 2026-10-24 | gate |  | the error is not concentrated; any 16-bit stage on the force path breaks the gate | accuracy |
| 272 | `prec-zbl-fp16` | ZBL stage in fp16 storage | NOT BUILT | 2026-10-24 | bound |  | costs nothing today (passes the gate; share ≈ 0) |  |
| 273 | `prec-fp16-edge-weights` | FP16 / BF16 storage of per-edge weights | REJECTED | 2026-10-14 | gate |  | accuracy: FP16 and BF16 per-edge weights exceed the allowed error | accuracy |
| 274 | `prec-fp16x3-probe` | Exponent-shifted 3×FP16 co-execution probe | REJECTED | 2026-10-26 | lab | sm_120 | no production kernel realised the probe gain; the fused GEMM lost to cuBLAS | single_gpu, stand_in, superseded |
| 275 | `prec-fp16x3-standalone` | Standalone Triton 3×FP16 GEMM on sm_120 | REJECTED | 2026-10-26 | lab | sm_120 | ties or loses to cuBLAS IEEE; the 3×FP16 ceiling on sm_120 is 0.67× FP32 SIMT | mixed, single_gpu, stand_in |
| 276 | `prec-fp16x3-rowcol` | Row/column-normalised 3×FP16 | REJECTED | 2026-10-26 | lab |  | normalisation overhead (accurate, but 860.3 vs 522.3 µs) | stand_in |
| 277 | `prec-fp16x3-other-gpus` | Row/col 3×FP16 on A100, A4000, Ada | REJECTED | 2026-10-26 | lab | A100, RTX A4000, RTX 4000 Ada | per-K-tile normalisation, split and rescale outweigh the FP16 tensor rate on tall-narrow K = N = 512 shapes | stand_in |
| 278 | `prec-delta-gemm` | Time-step delta GEMM | REJECTED | 2026-10-26 | gate |  | equivariant features change about 21 % per 1 fs step | accuracy |
| 279 | `prec-channel-split` | 50/50 channel-magnitude split | REJECTED | 2026-10-26 | gate |  | MACE weight row norms are nearly uniform; the tail carries about 40 % of the product norm | accuracy |
| 280 | `prec-hybrid-gemm` | Fused SIMT + 3×FP16 NVRTC GEMM | REJECTED | 2026-10-26 | lab |  | wins only on the smallest shape; lost SIMT efficiency and barriers dominate | stand_in, mixed |
| 281 | `prec-packed-b` | Larger tiles, pre-split packed B | REJECTED | 2026-10-26 | lab |  | no general speed win; a fixed FP16 split loses FP32's exponent range | stand_in, accuracy |
| 283 | `prec-decoupled` | Subgroup-barrier decoupling | REJECTED | 2026-10-26 | lab |  | duplicate staging and LSU work outweigh the recovered sync time | stand_in |
| 284 | `prec-mbarrier` | Concurrent kernels / `mbarrier` rings | NOT BUILT | 2026-10-26 | bound |  | the T08d result lowers their value; GEMM-level hybrid work on sm_120 closed | no_build |
| 285 | `prec-two-kernel-split` | Two-kernel FP32/MMA column split | REJECTED | 2026-10-26 | lab |  | the MMA slice is slower per column than FP32 | stand_in |
| 286 | `prec-bk32-swizzle` | BK = 32 staging, block swizzling | REJECTED | 2026-10-26 | lab |  | within noise or shape-dependent; far behind cuBLAS | stand_in, mixed |
| 287 | `prec-fp32-baselines` | FP32 baseline options (cuBLASLt, transposed B) | REJECTED | 2026-10-26 | lab |  | no FP32 alternative beat cuBLAS | mixed |
| 288 | `prec-tensor-only-sweep` | Tensor-only 3×FP16 tile sweep | REJECTED | 2026-10-26 | lab |  | on-the-fly normalisation costs more than tensor cores return | stand_in |
| 289 | `prec-presplit-epilogue` | Producer-side A split + epilogue fusion | REJECTED | 2026-10-26 | lab |  | recurring operand preparation is the barrier | stand_in, mixed |
| 290 | `prec-setmaxnreg` | 4+4 warpgroup `setmaxnreg` | REJECTED | 2026-10-26 | lab |  | the 4 + 4 / BM = 128 layout is inferior regardless of registers | stand_in, mixed, resources |
| 291 | `prec-cutlass-sm120` | CUTLASS SM120 FP16 GEMM as base | REJECTED | 2026-10-26 | lab |  | slower base; at most a few-percent ceiling | stand_in |
| 292 | `prec-producer-split` | Fused producer A split (Triton) | REJECTED | 2026-10-26 | lab |  | even with free emission the GEMM only ties cuBLAS | stand_in |
| 293 | `prec-gemmul8` | GEMMul8 Ozaki II emulation | REJECTED | 2026-10-26 | lab |  | flawed benchmark; no accuracy gate | stand_in, accuracy |
| 294 | `prec-int8-f32` | INT8-slice float32 GEMMs | REJECTED | 2026-10-27 | lab | sm_120 | the INT8:FP32 rate must exceed the product count (≥ 10) for float32 accuracy | stand_in, single_gpu |
| 297 | `prec-emu-performant` | Emulation "performant" strategy | REJECTED | 2026-10-27 | ab |  | eager also emulates the radial 1280×65 products |  |
| 298 | `prec-emu-40` | 40 emulation mantissa bits | REJECTED | 2026-10-27 | gate |  | accuracy is lost fast | accuracy |
| 300 | `prec-f64-triton-rewrites` | `segment_gemm`/`grouped_launch` in float64 | NOT BUILT | 2026-10-27 | bound |  | Triton GEMMs are float32-only; emulated cuBLAS is faster anyway | no_build |
| 302 | `prec-double-float` | Paired-float32 (double-float) arithmetic | REJECTED | 2026-10-27 | lab |  | misses both bars | stand_in, accuracy |
| 303 | `prec-d1-56bits` | Fixed 56 mantissa bits (D1) | SUSPENDED: `auto` is `"dynamic"` everywhere (`dynamic_decis… | 2026-10-28 | ab | RTX 5060 Ti, RTX 4000 Ada | dynamic mode pays an exponent pre-pass or picks more bits than needed |  |
| 305 | `prec-d1-48` | 48 or 52 mantissa bits | REJECTED | 2026-10-27 | gate |  | no longer float64-grade | accuracy |
| 308 | `prec-ozaki-fp8` | FP8/FP4 Ozaki emulation for float64 | REJECTED | 2026-10-27 | unknown |  | not recorded beyond 'rejected or parked earlier' | unexplained, no_build |

## Launches, graphs, scheduling and megakernels (14)

| Line | ID | Idea | Status | Date | Stage | Tested on | Reason | Flags |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 318 | `launch-host-stop` | Remove the last host stop (spin / lagged pipeline) | REJECTED | 2026-10-07 | ab |  | No gain. |  |
| 324 | `launch-combo-persub` | Combo kernels with per-sub-kernel blocks | REJECTED | 2026-10-21 | ab | local | Did not remove the NbCl5 loss. | mixed, single_gpu |
| 325 | `launch-combo-sizerule` | Combo kernels on small cells only | NOT BUILT | 2026-10-21 | bound |  | Would not make the record exact; size-specific; small. | no_build |
| 326 | `launch-memsets` | cuBLAS memsets, per-step constants, eager zero-fills | NOT BUILT | 2026-10-21 | bound |  | (a) needs another GEMM or cuBLAS path; (b) touches MC species changes; (c) overlaps OPT3-T05. | no_build |
| 327 | `launch-pdl` | Inductor programmatic dependent launch | REJECTED | 2026-10-15 | gate |  | Wrong forces in CUDA-graph replays; never enable (SPEC §6.2). | accuracy |
| 328 | `launch-batching` | Disconnected-graph batching (idea 7) | REJECTED | 2026-10-09 | lab |  | Small gain; excluded by the production rule. |  |
| 329 | `launch-megakernel` | Complete force megakernel | NOT BUILT | 2026-10-20 | bound | RTX 5060 Ti | Months of work for a small ceiling; the step is per-size arithmetic, not launch overhead. | no_build, resources, stand_in |
| 330 | `launch-het-megakernel` | Heterogeneous persistent megakernel; co-execution gate | REJECTED | 2026-10-24 | lab | sm_120, A100, L4 | TF32 MMA rate equals FP32 FFMA on sm_120; on A100 tensor cores alone beat the mix; the L4 pass is a no-traffic ceiling. | stand_in |
| 331 | `launch-sched-iso` | Scheduler-isolated co-execution | REJECTED | 2026-10-26 | lab | sm_120 | Each sub-partition owns private FFMA and tensor units; isolation starves them. | stand_in |
| 332 | `launch-regional-persistence` | Regional persistence of the fused atom block (OPT-T05) | NOT BUILT (never started; its precondition, stage C, was re… | 2026-10-20 | bound |  | Stage C was rejected; gaps too small. | no_build |
| 333 | `launch-cublasdx` | cuBLASDx block GEMM as megakernel primitive | REJECTED | 2026-10-26 | lab |  | The primitive is already slower before any fusion. | stand_in |
| 334 | `launch-two-streams` | Independent branches on two streams | REJECTED | 2026-10-26 | lab |  | No reliable critical-path gain. | mixed, stand_in |
| 336 | `fp32-followups-20261029` | Inductor aggressive fusion | REJECTED | 2026-10-29 | ab | local |  | tie |
| 337 | `fp32-followups-20261029` | Save-transcendentals partitioner in float32 | REJECTED | 2026-10-29 | ab | local, RTX 4000 Ada |  |  |

## Other models (13)

| Line | ID | Idea | Status | Date | Stage | Tested on | Reason | Flags |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 411 | `model-generic-density-first` | Density-first for the generic family | REJECTED | 2026-10-15 | unknown |  | No gain. |  |
| 413 | `model-spin-restricted` | Spin-restricted PolarMACE | REJECTED | – | gate |  | Changes the model (never enable); worse model. | accuracy |
| 416 | `model-ffwd-polar` | `fused_folded_forward` on POLAR-1-S | REJECTED | – | ab |  | Neutral. POLAR-1-S has one interaction, which is reassociated, so the kernel had nothing to act on. | tie, unexplained |
| 421 | `model-mh1-edge-sh` | EDGE_SH for MH-1's class (MH-1, POLAR-1) | REJECTED (POLAR-1, A100-SXM4-80GB: tie) | 2026-11-06 | ab | A100-SXM4-80GB | A tie for POLAR-1 on the A100-SXM4-80GB; other GPUs not measured. | tie, single_gpu, retest_noted |
| 422 | `model-mh1-fused-first` | FUSED_FIRST_BACKWARD (tabled outer sum) in MH-1's first interaction | REJECTED (POLAR-1, A100-SXM4-80GB: slower) | 2026-11-06 | ab | A100-SXM4-80GB | Slower on the A100-SXM4-80GB; the 0.970 order was read as noise. | single_gpu |
| 424 | `model-polar-wprime-a100` | `wprime_backward` + `fused_folded_forward` on POLAR-1 on the A100 | DEFAULT (scoped: POLAR-1-M, A100); L REJECTED (NbCl5 slower) | 2026-11-06 | ab | A100-SXM4-80GB | POLAR-1-L rejected by the mixed-cell rule (slower in both orders). | single_gpu |
| 427 | `model-off-screens-a100` | Species GEMM, chunked fused forward, EDGE_SH/HEAD_SLICE/GRAPH_SUM off on MACE-OFF | REJECTED (A100) | 2026-11-06 | ab | A100 | Defaults unchanged; species GEMM and chunked forward slower; other switches ties or under 0.5 %. | mixed, single_gpu |
| 436 | `model-polar-phase-recurrence` | k-space cosines and sines from per-axis phase tables | REJECTED (5060 Ti: tie) | 2026-11-08 | ab | RTX 5060 Ti | Tie; the transcendentals were not the cost on this GPU. | mixed, single_gpu |
| 438 | `model-polar-head-rows` | Irrep blocks kept as rows across the POLAR head chain (Codex R3) | STOPPED (bound: P1 removed its target) | 2026-11-08 | bound | RTX 5060 Ti | Bound: P1 removed its target; the four target kernels are gone with P1. | no_build |
| 439 | `model-small-matmuls` | Small mm/addmm as an Inductor-fused multiply-sum | REJECTED | 2026-11-07 | ab | RTX 5060 Ti | Slower on OFF23-small; no gain on NbCl5; NaCl b side recompiled during timing (invalid). | mixed, single_gpu, retest_noted |
| 440 | `model-polar-shared-geometry` | One unmasked k-space geometry per POLAR call | REJECTED (A100: tie) | 2026-11-07 | ab | A100-SXM4-80GB | A100-SXM4-80GB tie. | tie, single_gpu |
| 441 | `model-polar-e3nn-block-linear` | e3nn Linears as BlockLinear on POLAR-1 | REJECTED (A100: tie) | 2026-11-07 | ab | A100-SXM4-80GB | A100-SXM4-80GB tie. | tie, single_gpu |
| 442 | `model-ffwd-omol` | `fused_folded_forward` on OMOL-0 XL | REJECTED | – | ab |  | The wider a costs the in-kernel product more than the w read saves. |  |

## Integrator, engine and neighbour list (13)

| Line | ID | Idea | Status | Date | Stage | Tested on | Reason | Flags |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 345 | `eng-skin-rescreens` | Skin 0.2 / 0.4 Å, first skin/bucket screen | REJECTED | 2026-10-08 | ab |  | Within noise. | mixed |
| 347 | `eng-bucket-width` | 2 % / 3 % / 10 % buckets; stand-alone padding | REJECTED | 2026-10-09 | ab |  | No MD-step gain. | retest_noted |
| 352 | `eng-empty-cache` | `empty_cache()` after each capture | REJECTED | 2026-10-07 | unknown |  | No effect. |  |
| 354 | `eng-species-morton` | Species + Morton atom order | REJECTED | 2026-10-14 | lab |  | No gain. | stand_in |
| 355 | `eng-force-scatter` | Segmented force scatter | NOT BUILT | 2026-10-14 | bound |  | Too small. | no_build |
| 356 | `eng-baoab-1fs` | BAOAB at 1 fs (as a speed item) | NOT BUILT | 2026-10-14 | unknown |  | Changes the protocol. |  |
| 359 | `eng-noise-in-graph` | Noise selection inside the graph | NOT BUILT | 2026-10-21 | bound |  | Already hidden behind the GPU. | no_build |
| 366 | `eng-incremental-nl` | Incremental neighbour list (F5) | NOT BUILT | 2026-10-27 | bound |  | Not worth a falsifier. | no_build |
| 367 | `eng-rebuild-cost` | Cheaper or rarer neighbour-list rebuilds (skin 1.0) | REJECTED (nothing to gain) | 2026-11-05 | lab | A100-SXM4, 5060 Ti | Nothing to gain. | stand_in |
| 369 | `eng-on-nl` | O(N) neighbour-list rebuild for large cells | NOT BUILT | 2026-10-28 | bound |  | Rebuilds are rare with the skin; matters only for very large cells. | no_build |
| 374 | `fp32-followups-20261029` | Edge buckets of 2.5 % | REJECTED | 2026-10-29 | ab | RTX 5060 Ti, RTX 4000 Ada | Effect ratios 0.990–1.006; the family record gives no separate reason for this candidate. | tie |
| 375 | `fp32-followups-20261029` | Sync-free neighbour rebuild | REJECTED | 2026-10-29 | ab | RTX 4000 Ada, RTX 5060 Ti | Effect ratios show no gain; the family record gives no separate reason for this candidate. | tie |
| 376 | `fp32-followups-20261029` | Pipelined edge margin 0.005 | REJECTED | 2026-10-29 | ab | RTX 4000 Ada | Redos fall from 36 to 0, but Ada ratios are 1.008–1.009. | tie |

## Setup, compilation and start-up (4)

| Line | ID | Idea | Status | Date | Stage | Tested on | Reason | Flags |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 384 | `setup-max-autotune` | `max-autotune` compilation | REJECTED | 2026-10-08 | ab |  | Inductor cudagraph trees conflict with the outer capture; setup cost. | mixed |
| 385 | `setup-activation-budget` | Activation memory budget < 1 | REJECTED | 2026-10-15 | ab |  | Recomputation costs time. |  |
| 392 | `setup-other-items` | OEq build, portable artifacts, self-check cost, load, graph variants | NOT BUILT | 2026-10-24 | bound |  | Below the build rule or covered by another fix. | no_build |
| 393 | `setup-selfcheck-cache` | Qualification cache to skip self-checks | NOT BUILT | 2026-10-20 | bound |  | Too little to save. | no_build |

## Tiled call (6)

| Line | ID | Idea | Status | Date | Stage | Tested on | Reason | Flags |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 756 | `tiled-lazy-keys` | Slot keys taken in sweep 1 from the prefetched graphs instead of a search pass | REJECTED | 2026-11-09 | ab | RTX 5060 Ti | The search pass's builds feed the graph cache, so its time was not idle time. |  |
| 757 | `tiled-piece-start` | Graph cache keeps a tile's index at its slot's halo start | REJECTED | 2026-11-09 | ab | RTX 5060 Ti | That work runs in the prefetch on the side stream, hidden behind the bodies. | tie |
| 758 | `tiled-keep-a0` | Keep interaction 0's node-level A for sweep 3 instead of recomputing it | REJECTED (bound) | 2026-11-09 | bound |  | Keeping the node-level A costs 8 GB at 1M atoms, and a host round trip (about 1.2 s) exceeds the 0.73 s saved. | no_build, resources |
| 760 | `tiled-kept-compact` | Kept lists delta-encoded below their 3 B per edge | REJECTED (bound) | 2026-11-09 | bound |  | Delta-encoding would save at most a third; whole-list dropping when short already saves all. | no_build |
| 761 | `tiled-receiver-cap` | Receiver caps above 16,384 in the compiled budget (32,768, 65,536) | REJECTED | 2026-11-11 | ab | RTX 5060 Ti | 32,768 within noise of 16,384; 65,536 is +3.5 %. | single_gpu, retest_noted |
| 762 | `tiled-device-pool` | The wavefront's row pool (and the alias window) on the device | REJECTED (speed) | 2026-11-11 | ab | RTX 5060 Ti | Slower than the plain device state; the wavefront's cost is its plan, not the copies. | single_gpu |

## Alternative backends and external comparisons (3)

| Line | ID | Idea | Status | Date | Stage | Tested on | Reason | Flags |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1009 | `ext-jax-mliap` | MACE-JAX and LAMMPS ML-IAP | REJECTED | 2026-10-08 | ab |  | Slower or inaccurate. | accuracy |
| 1010 | `ext-symmetrix` | Symmetrix-XL engine; LAMMPS rematch | REJECTED | 2026-10-14 | ab |  | Slower. |  |
| 1011 | `ext-lunatp` | LunaTP tensor product | REJECTED | 2026-10-27 | lab |  | Backward is bound by bytes on this GPU; LunaTP's gains were published against stock OEq on an H100. | stand_in, resources, retest_noted |

## OMAT-0 on the A100 (113)

| Line | ID | Idea | Status | Date | Stage | Tested on | Reason | Flags |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 450 | `omat0-launch-screen` | Launch settings of the fused forward and the streaming kernels | REJECTED | 2026-11-03 | unknown | A100, RTX 4000 Ada | the defaults already sit at the optimum; the time is inside the kernels |  |
| 452 | `omat0-first-full-rows` | Generic reassociation on the full table row `[h c, c]` | REJECTED | 2026-11-03 | ab | RTX 4000 Ada | the wider product costs more than the slice saves. | single_gpu |
| 453 | `omat0-column-block` | 64-column radial-table forward programs | REJECTED | 2026-11-03 | ab | RTX 4000 Ada | no gain. | single_gpu |
| 456 | `omat0-slope-vector` | q-major derivative table, float2/float4 term loads in wprime | REJECTED | 2026-11-04 | ab | RTX 5060 Ti, RTX 4000 Ada, A100 | fewer load instructions; the gain disappears with fewer terms. | mixed |
| 457 | `omat0-stream-w` | wprime reads the stored w with `__ldcs` | REJECTED | 2026-11-04 | ab | RTX 5060 Ti, RTX 4000 Ada, A100 | helps where L2 is small relative to the slabs (sm_120), hurts on sm_89. | mixed |
| 461 | `omat0-stream-store-w` | The fused forward stores w with `__stcs` | REJECTED | 2026-11-04 | ab | RTX 4000 Ada, RTX 5060 Ti |  |  |
| 462 | `omat0-eos-chunks` | `edge_outer_sum` chunk sizes 64 | REJECTED | 2026-11-04 | ab | RTX 4000 Ada |  | single_gpu, tie |
| 464 | `omat0-cutlass-tile` | Two-warp 32x64 / 64x32 tiles for the CUTLASS 3×TF32 GEMMs (A100) | REJECTED | 2026-11-04 | ab | A100 | wave quantisation on 108 SMs; the smaller tiles mainloops lose more than the even load gains. | single_gpu, resources |
| 465 | `omat0-tc-presplit` | W slice split into TF32 parts once per block (tensor-core forward, A100) | REJECTED | 2026-11-04 | lab | A100, RTX 5060 Ti | the doubled slice costs a third of the block's warps; the latency hiding lost outweighs the split instructions saved. | resources, stand_in |
| 467 | `omat0-species-gemm-tc` | The species GEMM in our own mma.sync 3xTF32 kernel (for the A100) | REJECTED | 2026-11-04 | ab | A100, RTX 4000 Ada, RTX 5060 Ti | a simple mma.sync kernel only reaches Triton's rate; CUTLASS 3xTF32 is about 2.7x faster per product, and closing the gap needs a CUTLASS-class grouped GEMM for at most … |  |
| 468 | `omat0-species-gemm-cutlass` | The species GEMM as one gathered CUTLASS 3xTF32 launch (A100) | REJECTED | 2026-11-04 | ab | A100, RTX 4000 Ada | in the captured MD step the A100's concatenated CUTLASS product is cheap; the grouped versions lose the rest to per-call row ordering and indices. |  |
| 469 | `omat0-wprime-piece-order` | wprime backward over edges sorted by (pair, r), so edges of one derivative-table slab run together | REJECTED | 2026-11-04 | lab | RTX 5060 Ti | the kernel streams edges in receiver order; sorted by slab, the output gradients and sender atomics lose locality, which costs more than the whole slope. | resources, single_gpu, stand_in |
| 472 | `omat0-wprime-aprime` | wprime slope as dr = a'(r) . (dw W) instead of the per-edge G' slab | REJECTED | 2026-11-04 | lab | A100 | the reformulation moves 61 GFLOP of per-edge products into the backward; at the A100's 3xTF32 rate that alone costs more than the table reads it saves. | single_gpu, stand_in |
| 473 | `omat0-first-table-columns` | First interaction's radial table stores only the columns it uses | REJECTED | 2026-11-04 | ab | A100, RTX 5060 Ti |  | tie |
| 477 | `omat0-fp16-tc` | 3×FP16 m16n8k16 MMAs in the tensor-core forward; 8 or 10 warps | REJECTED | 2026-11-05 | lab | A100, RTX 5060 Ti | the MMAs are not the limit, and fewer warps lose more latency hiding than the spills cost. | resources, stand_in |
| 478 | `omat0-producer-consumer` | Producer (MMA) and consumer (FP32) warps in the A100 forward, tiles through mbarriers | REJECTED | 2026-11-05 | lab | A100 | closed by the proposal's own rule (more than 10 % of serial): each side keeps only part of the 12 warps for its own latency, which costs what the overlap gains. | mixed, retest_noted, single_gpu, stand_in |
| 484 | `omat0-wprime-maxreg-a100` | Register caps for the A100 w' kernel | REJECTED | 2026-11-05 | lab | A100 | register caps add local memory (cap 72 1.201 with 88 B local, cap 64 1.477 with 128 B local), so the A100 stays uncapped. | resources, single_gpu, stand_in |
| 485 | `omat0-wprime-split` | w' channels split over 2 / 4 warps (falsifier) | REJECTED | 2026-11-05 | lab | A100, RTX 5060 Ti | splitting the channels does not shrink the per-edge register state; it multiplies the per-edge preamble and adds a cross-warp reduction. | stand_in |
| 486 | `omat0-mma-chain` | 3xTF32 projection with separate accumulators for the small products, or the MMA asm without volatile | REJECTED | 2026-11-05 | lab | A100 |  | single_gpu, stand_in |
| 487 | `omat0-hybrid16` | 3xTF32 projection as TF32 big*big plus FP16 m16n8k16 small products (2^11 shift) | REJECTED on the A100 | 2026-11-05 | lab | A100, RTX 5060 Ti | the per-fragment FP16 packs (12 cvt per (g, m, 16 k)) and the register limit eat the third of the MMA time it saves. | mixed, resources, stand_in |
| 490 | `omat0-ff-prefetch` | cp.async double-buffered a / sh (and x) staging in the fused forward | REJECTED on the A100 | 2026-11-05 | lab | A100, RTX 5060 Ti, RTX 4000 Ada | the A100 kernel is at 168 registers with 12 warps; the second buffer costs warps or spills, and the 12 warps already hide the load latency. | retest_noted, stand_in |
| 491 | `omat0-radial-table-shapes` | Radial table launch shapes (BLOCK_E, COLUMN_BLOCK, warps) | REJECTED | 2026-11-05 | lab | A100 | too small to pass an MD A/B. | single_gpu, stand_in |
| 492 | `omat0-tp-factor` | Fewer coefficient multiplies in the fused forward's tensor product | REJECTED (bound) | 2026-11-05 | lab | RTX 5060 Ti |  | single_gpu, stand_in |
| 493 | `omat0-table-in-forward` | Interaction 2's radial table evaluated inside the fused forward | REJECTED (estimate) | 2026-11-05 | bound |  | the table is cheap to store once and expensive to re-evaluate per warp. | no_build |
| 494 | `omat0-fp16x3-gemm` | 3xFP16 GEMMs (one FP16 GEMM over 3K, power-of-two scaling) instead of CUTLASS 3xTF32 | REJECTED (accuracy) | 2026-11-05 | gate | A100 | the FP16 MMA accumulates more coarsely; the split would have to be fused into every producer for any gain. | accuracy, single_gpu, stand_in |
| 495 | `omat0-tc-tp` | The fused forward's tensor product as a second 3xTF32 mma (w x from the product's fragments, S = CG sh, edges… | REJECTED | 2026-11-05 | lab | A100, RTX 5060 Ti | each A element feeds one mma triple (N = 8 outputs, K = 8 edges), and its preparation costs about what the FP32 TP terms cost. | stand_in |
| 496 | `omat0-ff-hints` | Cache hints in the fused forward (x evict_last / ldg, a and sh no_allocate) | REJECTED | 2026-11-05 | lab | A100, RTX 5060 Ti |  | stand_in |
| 497 | `omat0-atom-order` | Morton atom order for the x gathers | REJECTED | 2026-11-05 | lab | RTX 5060 Ti | x (10 MB) stays in L2 in any order; the gathers cost latency, not DRAM traffic. | resources, single_gpu, stand_in |
| 499 | `omat0-species-gemm-a100-recheck` | The species GEMM on the A100 again at the 11.7 ms step | REJECTED | 2026-11-05 | ab | A100-SXM4-80GB, A100 80GB PCIe | SrTiO3 has 3 species and gains; NbCl5 has 2 species and only 6 atoms per cell, where the grouped launch0027s row ordering and index arrays cost more than a third of the … | mixed |
| 500 | `omat0-table-w-forward` | w from the value table inside the fused forward (the backward's TABLE_W) | REJECTED | 2026-11-05 | lab | A100-SXM4-80GB, RTX 5060 Ti | The slab gathers cost more than the tensor-core product they replace; even with the table in cache, the FP32 kernel only ties the tensor-core one. | mixed, stand_in |
| 501 | `ideas-r5` | Owner-pass forward (idea 14A) and warp groups sharing W fragments (idea 17): lab bounds | 17: `W_GROUP` DEFAULT on the H200 (MD 0.9946 / 0.9942) and … | 2026-11-06 | ab | H200, A100-SXM4 | 14(A) is slower in MD (1.045 H200, 1.066 A100) and its gate fails; 17 (W_GROUP) adopted as default. | mixed, stand_in |
| 502 | `ideas-r7` | Owner-pass forward (14A) as a switch; drift audit of the H200 and A100 defaults; lab: stored-w variants, the … | 14A REJECTED; drift: no change; idea 21 ADOPTED on the A100… | 2026-11-06 | ab | H200, A100-SXM4 | The owner pass is slower in MD on both GPUs and the H200 gate fails (NbCl5 stress 2.076 of 2); storing w and reloading it cost more than the k-loop work saved. | mixed |
| 503 | `ideas-r8` | A100 at 12 warps (`WFRAG_REGION`) with `W_GROUP`; `A_SPLIT`; ncu stall reasons (idea 18) | WFRAG_REGION + W_GROUP DEFAULT on the A100-SXM4; A_SPLIT RE… | 2026-11-06 | ab | A100-SXM4, H200 | A_SPLIT is slower on the H200 and off everywhere; idea 18 is closed because instruction-fetch stalls are only 0.18 of 5.40 cycles. |  |
| 504 | `ideas-r9` | 14 and 16 warps (idea 22(ii)), part phases (25), x-load bounds, atom order, persistent w′ (23) | 22(ii), 23, 25 CLOSED; nothing adopted | 2026-11-06 | lab | H200, A100-SXM4, A100 PCIe | ptxas caps the forward at 128 registers and spills at 14 and 16 warps; no better than 12 warps without x loads; persistent w′ slower on every A100 variant. | mixed, resources, stand_in |
| 505 | `ideas-r11` | x gathers after the k-loop (idea 27, `X_LATE`), a/W one k-step ahead (`K_PIPE`), staging one batch ahead (ide… | X_LATE DEFAULT on the H200 and A100-SXM4; K_PIPE off; 26 RE… | 2026-11-06 | ab | H200, A100-SXM4 | Stage-ahead is 1.012–1.034 on the H200 with spills; K_PIPE is worse than X_LATE alone on the H200 and noise on top of it. | mixed, resources |
| 506 | `ideas-r12` | CG factors folded into W (idea 24, `CG_FOLD`), TP inside the next k-loop (22(iii)) | CG_FOLD DEFAULT on the H200 and A100-SXM4; 22(iii) REJECTED | 2026-11-06 | ab | H200, A100-SXM4 | 22(iii) is bitwise but 1.03–1.21 in every form. | mixed |
| 507 | `ideas-r13` | a read with one LDS.128 (17(ii), `A_LD128`), TP without flush checks (29, `TP_NOFLUSH`), further CG folds (28… | A_LD128 DEFAULT on the H200 and A100-SXM4, TP_NOFLUSH on th… | 2026-11-06 | ab | H200, A100-SXM4 | 32 stopped: w shared by mirror pairs at 32-row tiles gives f = 0.22, below the 0.25 stop line; 28 is rounding, about 0.1 % on the H200. |  |
| 508 | `ideas-r14` | GEMM epilogues (31, `LINEAR_DIV`), warp-specialized wgmma forward (30), mirror tiles (32), sm_90 3xTF32 GEMMs… | LINEAR_DIV DEFAULT on the H200 and A100-SXM4; WS_FORWARD DE… | 2026-11-06 | ab | H200, A100-SXM4 | 32, 33 and 36 stopped by their bounds; FIRST_DIV rejected: MD 1.031–1.035 and the density gradient not bitwise on sm_80/sm_90. | mixed |
| 510 | `ideas-q2` | Segmented density sums (5(b), `RECEIVER_SUM`); G1 pre-split ceiling; rcfold on the A100 (28); ZBL config, S f… | RECEIVER_SUM DEFAULT on the A100 (SXM4 and PCIe names); H20… | 2026-11-06 | ab | A100-SXM4, H200 | G1 ceiling about 2 % of the step, real kernel about half; 28 about 0.35 % and rounding; 5(d), 13, 35 bounded or out of proportion to build. | stand_in |
| 511 | `ideas-q1` | Folded α in the w′ table (34, `FOLD_ALPHA`), power-basis tabled backward (40, `POWER_BACKWARD`), Estrin w′ (3… | FOLD_ALPHA DEFAULT on the H200 and A100-SXM4; POWER_BACKWAR… | 2026-11-06 | ab | H200, A100-SXM4 | ESTRIN gains nothing with FOLD_ALPHA (1.0002 / 1.0028) and FOLD_ALPHA alone beats it on the A100; the chunk sweeps were ties or 0.4–1 % slower. |  |
| 512 | `ideas-r3` | w once per mirror pair (idea 14); lane-parallel batch index loads in the fused forward (idea 15a) | idea 14 DROPPED (bound); LANE_INDICES DEFAULT on the H200 | 2026-11-06 | ab | H200, RTX 5060 Ti | Idea 14 dropped at its bound: pair product plus stored-w forward (1,682 µs) against the 1,673 µs forward; LANE_INDICES is off outside the H200. | mixed, stand_in |
| 514 | `kr-x-nodep` | Fused forward: register prefetch of the next batch's x (the senders → x dependency) | REJECTED (bound) | 2026-11-06 | lab | RTX 5060 Ti, H200 | The gathers cost their L2 traffic, not their dependency, so a register prefetch has nothing to hide. | mixed, resources, stand_in |
| 515 | `kr-redistribute` | Fused forward: vectorised C-fragment redistribution | REJECTED (bound) | 2026-11-06 | lab | H200 | Stated only as a bound: H200 0.998. | single_gpu, stand_in, tie |
| 516 | `kr-wprime-restrict` | w′: `__restrict__` on every pointer argument | REJECTED | 2026-11-06 | lab | RTX 5060 Ti, H200 | Slower in the lab on both GPUs (RTX 5060 Ti 1.016, H200 1.031); 96 registers on both. | stand_in |
| 517 | `kr-wprime-vector-atomics` | w′: float4 dx atomics (sm_90) | REJECTED (bound) | 2026-11-06 | lab | H200, RTX 5060 Ti | The atomics are not on the H200 critical path, so vectorising them cannot pay there. | mixed, stand_in |
| 518 | `kr-tabled-warps` | Tabled outer-sum kernels at 2 warps on the H200 | REJECTED | 2026-11-06 | ab | H200 |  | single_gpu |
| 519 | `kr-species-cutlass-h200` | Species GEMM on the CUTLASS backend on the H200 | REJECTED | 2026-11-06 | ab | H200 |  | single_gpu |
| 521 | `fusion-receiver-local` | Receiver-local forward + backward of the last interaction (dout, sh, x on chip; no message store) | REJECTED (fusion and dout staging) | 2026-11-06 | lab | H200, RTX 5060 Ti | Message store and sh/x reads cost nothing measurable; real dout staging (occupancy, exposed fill) measured 1.935–1.986 against a stop rule of under 5 % gain. | mixed, stand_in |
| 522 | `sx-forward-table-w` | Symmetrix-XL-style interaction 2 forward: w per edge from the value table instead of the a·W tensor-core prod… | REJECTED (bound) | 2026-11-06 | lab | H200, RTX 5060 Ti | Every edge reads 30 KB of slab (11.5 GB per call), more than the 3xTF32 product costs; even the L1 case saves only 10 %. | mixed, stand_in |
| 523 | `sx-wprime-reduction` | w′: Symmetrix-XL's Cartesian Jacobian before the reduction, or a cheaper reduction | REJECTED (bound) | 2026-11-06 | lab | H200 | Removing the reduction completely saves 3.6 %; the lab butterfly result conflicts with the H200 MD A/B behind the BUTTERFLY default. | single_gpu, mixed, stand_in |
| 525 | `rescreen-20261109` | Re-screen of off switches against 9 November defaults (H200: TABLE_PAIRS, HOIST_SCALE, STAGE_QUADS, w′ STAGE_… | REJECTED | 2026-11-09 | ab | H200, A100-SXM4 | All candidates miss the 1 % stop rule, which needs at least 1 % in both orders. | retest_noted |
| 526 | `quad-table-bound` | Quadratic spline table (twice the pieces) for w′: lab bound `F-quad-horner` | REJECTED (bound) | 2026-11-09 | lab | H200, A100 PCIe | A quadratic table of twice the pieces would save at most 3.4 % of w′ before its larger loads; on the A100 the table cost is mostly loads, not FMAs. | stand_in |
| 528 | `tc-edge-tp` | The SPLINE_W forward's and w′'s tensor product on 3×TF32 tensor cores: x split once per node as B (channels a… | REJECTED (bound) | 2026-11-09 | lab | A100-SXM4 | Analysis: at most 0.8 of the TP part and at most 3 % of the step, before preparation and register costs; the built TC_TP precedent (1.276) was slower. | stand_in |
| 529 | `mirror-a100` | Mirror-pair w′ (shared table work, mirror order) re-measured on the A100 against today's w′ defaults | REJECTED (kernel lab) | 2026-11-09 | lab | A100-SXM4 | Mirror order breaks STAGE_DOUT's receiver runs (2.9x); skipping half the table loses (1.308) because the branch adds 11 registers; a partner edge would need its w stored… | single_gpu, stand_in, mixed |
| 531 | `stage-quads` | w′: the STAGE_DOUT row read as float4 quartets (one LDS.128 per four staged components) | MEASURED, closed (H200 0.9960 / 0.9984: under 1 %) | 2026-11-07 | ab | H200 | The staged loads do not bound w′, so the A/B gain stays under 1 %. | single_gpu, tie |
| 533 | `packed-coef` | First radial table: two coefficient terms per float4 load | MEASURED, closed: slower (H200 1.0072 / 1.0079) | 2026-11-07 | ab | H200 | Triton issues half the coefficient loads with no register change, but the MD step is 0.7–0.8 % slower. | single_gpu, tie |
| 534 | `factor-w` | Spline forward: w times a path's common factor once | MEASURED, closed (H200 0.9969 / 0.9995: under 1 %) | 2026-11-07 | ab | H200 | Measured H200 ratios 0.9969 / 0.9995 are under 1 %, so it is closed. | tie, single_gpu |
| 535 | `breakthroughs-20261108` | External report: 64-channel h(r) spline, atom megakernel, undirected w′ pairs, fused interaction-1 sum, multi… | REJECTED: h(r) spline (81,920 FMA per edge ≥ 0.91 ms on the… | 2026-11-08 | bound | H200, RTX 5060 Ti | h(r) spline: 81,920 FMA per edge, at least 0.91 ms on the H200; megakernel GEMMs compute-bound; pairs closed by B-mirror-share; fused sum not built (about 50 µs bound). | no_build |
| 536 | `table-compressed` | SPLINE_W table with float16 / bfloat16 coefficients (c0 float32) | REJECTED (certificate) | 2026-11-08 | bound |  | Certificate worst cases (float16 3.16 to 52, bfloat16 20 to 418) exceed the pass level of 1. | accuracy, no_build |
| 537 | `table-per-pair-grid` | SPLINE_W knot grid fitted per species pair | DROPPED (bound) | 2026-11-08 | bound |  | Bytes per edge unchanged and modelled L1 hits rise only 1–3 points. | no_build |
| 538 | `breakthroughs-codex-20261108` | Codex screen: fast spatial operator (mesh/FFT or H²/FMM) for both interaction-2 edge walks; edge-frame / expa… | REJECTED (counting): spatial operator, mesh 0.5 Å forward a… | 2026-11-08 | bound |  | Mesh forward alone takes 6.9 ms at FP32 peak, 5.9× the 1,176.7 µs budget; FMM needs boxes of 6.85 Å or more for rank 27, above the cutoff. | no_build, stand_in |
| 539 | `certified-skip` | Interaction 2 evaluated only where a force/stress error certificate requires it (skip edge / path / channel b… | REJECTED (oracle): a perfect certificate omits about 0.4 % … | 2026-11-08 | bound | CPU | Oracle with a perfect certificate omits about 0.4 % of interaction-2 edges, but 20 % of the step needs 40.6–52.5 % of them. | no_build, stand_in |
| 541 | `wprime-stage-senders` | w′: stage or reorder the sender rows per cohort so the per-edge loop reads only the table | STOPPED (bound) | 2026-11-08 | bound | RTX 5060 Ti, H200 | H200 ceiling about 44 µs = 1.3 % of the step, under twice the 1 % bound. | no_build, stand_in |
| 542 | `bulk-slab` | Spline forward with STAGE_EDGES: each edge's 20 KB slab bulk-copied (cp.async.bulk + mbarrier, 1–2 edges ahea… | REJECTED (lab) | 2026-11-08 | lab | RTX 5060 Ti, H200 | Against equal-occupancy controls it is slower (1.343× / 1.127×) and not bitwise (3.75e-8). | stand_in |
| 543 | `tc-overlap-bound` | Tensor-core GEMMs and other kernels independent of the edge walks run beside the interaction-2 spline forward… | REJECTED (bound) | 2026-11-08 | bound | H200 | Perfect hiding, including launch gaps, is 38 µs = 1.07 % of the step, under the 68 µs lab bar. | stand_in, single_gpu |
| 544 | `grouped-gemm-bound` | One grouped CUTLASS 3xTF32 launch per product group (linear_div, its gradient, the species GEMM and its gradi… | REJECTED (bound) | 2026-11-09 | bound | H200 | Saves 61.0 µs (1.8 % of the step), under the 68 µs lab bar; a real grouped kernel loses part of it. | single_gpu |
| 545 | `codex1109-wprime-locator` | w′ with a precomputed per-edge locator to remove a claimed 7-band branch tree and its 518-instruction `WARPSY… | REJECTED (already measured) | 2026-11-09 | unknown | H200 | Locator is computed at staging under STAGE_EDGES, the band chain is warp-uniform, and the precomputed record was already measured (H200 0.9966 / 0.9974, closed). | tie |
| 546 | `codex1109-spatial-slab-walk` | Spline forward receivers in a Hilbert/SM-aligned spatial order so co-resident warps walk the radial shells in… | REJECTED (already modelled) | 2026-11-09 | bound |  | The 8 Nov L1 model already tried Morton order (hit rate 0.061 to 0.079); the 0.92 target has no model run behind it, and the one-slab bound is not reachable. | no_build |
| 547 | `codex1109-fused-outer-sum` | Interaction 1's outer sums accumulated per receiver in shared memory with the species GEMM applied in the sam… | REJECTED (assessment) | 2026-11-09 | bound | H200 | Outer sum writes its 61 MB in 115 µs, about 0.5 TB/s, so it is not traffic bound; fusing gives up the CUTLASS species GEMM rate. |  |
| 550 | `sc-generated-product1` | `SC_GENERATED` extended to product 1's cuEq symmetric contraction (interaction 1, 128x0e out; no division to … | STOPPED (bound < 1 %) | 2026-11-09 | bound | H200 | Saves about 13 µs (0.4 % of the 3.40 ms step), under the 1 % bar of 34 µs. | single_gpu, stand_in |
| 551 | `sc0-invariant-basis` | Product 0's contraction rewritten in an invariant basis (13 scalar invariants and 16 vector covariants per ch… | STOPPED (lab bar unreachable) | 2026-11-09 | bound | H200 | A lab would have to show 68 µs saved, 83 % of what remains; the estimate saves about 19 µs forward and about 30 µs backward, under the bar's double. | no_build, single_gpu |
| 552 | `bspline-table` | Interaction 2's spline table as per-knot rows shared by neighbouring intervals: a cubic B-spline (one 5.12 KB… | REJECTED (kernel lab) | 2026-11-09 | lab | CPU, RTX 5060 Ti | Kernel lab forward at 0.930 / 0.925 of real: four scalar loads per weight cost what the shared rows save, under half the predicted share, so it stopped before the H200. | single_gpu, stand_in |
| 554 | `wprime-locality-bound` | w′ operand locality of any kind: sender bins / reordering, L2 residency windows, TMA or cp.async operand pipe… | STOPPED (bound) | 2026-11-09 | bound | H200 | Ideal locality of every operand is worth about 42 µs (1.2 % of the step), under the 68 µs lab bar. | single_gpu, stand_in |
| 555 | `forward-ghoist` | Interaction 2 forward with the channel-independent CG sums g_oi = Σ_j c e_j hoisted per edge into shared memo… | REJECTED (kernel lab, 5060 Ti); re-screened on the A100 11 … | 2026-11-09 | lab | RTX 5060 Ti, A100 | Shared g reads and fill cost more than the FMULs removed; stopped before the H200. | mixed, stand_in, retest_noted, superseded |
| 556 | `wprime-zero-byte` | w′'s operand bytes: mirror-pair w′ sharing the table read, 12-byte or packed tables for w′, the receiver's do… | STOPPED (bound) on the H200 | 2026-11-09 | lab | H200, RTX 5060 Ti | w′ is instruction-issue bound and its bytes cost nothing measurable. | mixed, stand_in |
| 557 | `wprime-smem-spill` | w′ under a register cap with CUDA 13's shared-memory spilling (`enable_smem_spilling`, Codex 9 Nov v4) | REJECTED | 2026-11-09 | ab | H200, RTX 5060 Ti, A100 |  | mixed, resources, retest_noted |
| 558 | `wprime-backward-overlap` | Overlap w′ with independent backward kernels, or split w′ into dx/dr and dsh kernels so the layer-1 chain sta… | REJECTED (bound) | 2026-11-09 | bound | H200 | Independent work is only about 20 µs, under 1 %, and the split duplicates about as much work as it would hide. | single_gpu |
| 560 | `tabled-forward-maxnreg` | Register cap on the Triton tabled outer-sum forward (128 registers, 23% warps active on sm_80) | REJECTED (tie) | 2026-11-09 | ab | A100-SXM4 | Stated as a tie: MD step ratios 0.9975 to 1.0027 at caps of 64 and 80 registers. | tie, single_gpu |
| 561 | `first-adjoint-map` | Tabled outer-sum backward: one map `H = G Cᵀ` (`[D, TERMS]`) per (receiver, sender species, piece), built onc… | STOPPED (bound < 1 %) | 2026-11-09 | bound | CPU | Mean group 1.26 edges at 0.07 Å, under the break-even of 2.4 (H200) and 3.7 (A100); at most about 7 µs, 0.2 % of the H200 step. | no_build |
| 563 | `uniform-datapath` | Move per-edge integer work to the uniform datapath (sm_90) | REJECTED (bound) | 2026-11-09 | bound |  | Integer index work is about 11 of 216 forward edge-loop instructions (5 %), partly uniform already. | no_build |
| 564 | `nvrtc-version` | NVRTC kernels compiled with NVRTC 13.5.37 instead of 13.0.88 (Codex 9 Nov v4: Sobek's CUDA 12.9 → 13.3 build … | REJECTED | 2026-11-09 | ab | H200, RTX 5060 Ti |  |  |
| 565 | `report5-20261109` | Report 5 (9 Nov): launch collapse with PDL and persistent glue, mirror-pair w′, deferred CG in the forward, r… | CLOSED except the forward's table on disordered structures | 2026-11-09 | unknown |  |  |  |
| 567 | `deferred-cg-forward` | Deferred CG in the SPLINE_W forward (round 1 proposal 3, Symmetrix compute_Phi1): phi_p[i, j] += (w x_i) e_j … | REJECTED (kernel lab, H200) | 2026-11-09 | lab | H200 | Full form is 2.75× (crystal) and 2.35× (disordered) of real and spills at the 80-register cap; the split sums give 1.02 and 1.10 of real. | single_gpu, stand_in, resources |
| 568 | `wprime-split-sums` | w′ FACTORED u / v sums of 4+ terms in two partial sums (dependency-chain test; the longest chain is 8 FFMAs, … | REJECTED (kernel lab, H200) | 2026-11-09 | lab | H200 | Split sums are slower (1.029 to 1.033); Nsight shows w′ close to issue-bound, not dependency-bound. | single_gpu, stand_in |
| 569 | `fp64-second-pipe` | A100: the FP64 units as a second FMA pipe for w′'s two 21-term paths (round 2 idea 1): u / v sums, scaled and… | REJECTED (kernel lab, A100) | 2026-11-09 | lab | A100 | Register pressure, not conversions, kills it: doubles take register pairs, and any cap spills. | resources, stand_in |
| 570 | `fp64-tensor-pipe` | The FP64 tensor cores (mma.sync f64) beside FFMA: 8–64 multiply-adds per lane per issue on the tensor pipe | REJECTED (bound) | 2026-11-09 | bound | A100, H200 | Bound: product forms inflate multiply-adds 1.8–3.6x through 8-wide tile padding; the unpadded form needs 40 double-to-float conversions per channel and edge. | stand_in |
| 573 | `virial-elementwise` | One-graph strain as elementwise products: the strain gradient a reduction, not a split-K [3, E] x [E, 3] GEMM… | DEFAULT (scoped: RTX 5060 Ti; A100 rejected, bound 0.4 %; H… | 2026-11-10 | ab | RTX 5060 Ti, A100 | A100: cuBLAS runs the GEMM well; the bound is 0.4 % of kernel time at 5,000 atoms, under the 1 % bar. | retest_noted |
| 574 | `i0-copies-zero-fill` | Interaction 0: cat / clone / transpose copies and tabled_outer_sum_forward's zero-fill | NOT BUILT (bound under 1 %) | 2026-11-10 | bound | A100, H200 | Zero-fill is under the 1 % bar on A100 and H200, and the copies are absent on the sm_80 and sm_90 MD plans. | no_build |
| 575 | `wprime-receiver-adjoint` | w′ backward: R_ij = Σ c g per receiver run in shared memory, 13 % less FP32 work | REJECTED (kernel lab) | 2026-11-10 | lab | A100, RTX 5060 Ti | Shared memory 23 to 45 KB halves the CTAs per SM, and the 43 extra shared loads per edge and channel outweigh the 13 % FP32 cut. | resources, stand_in |
| 576 | `i1-moments` | First interaction through polynomial moments | REJECTED (occupancy) | 2026-11-09 | bound | CPU | At thermal displacements groups hold 1.2–1.3 edges against a break-even of 4.4, so moments would save nothing. | no_build |
| 577 | `species-only-kernels` | Species-only work in the force graph precomputed per species | NOT BUILT (sized at about 1.1–1.3 %) | 2026-11-09 | bound | A100, H200 | Upper bound from kernel names and shapes is just above the 1 % rule on both GPUs; the work would need an engine change, not yet built. | no_build, unexplained |
| 586 | `omat0-small-fold-alpha` | w′ FOLD_ALPHA off for OMAT-0 small; also TILE 64x32 and ESTRIN | MEASURED, closed: within the spread | 2026-11-09 | ab | H200, A100 | Within the run-to-run spread of about 0.5 percent: off was faster in two jobs, but on again was faster in a third. | retest_noted |
| 587 | `cursor32` | Spline forward: int loop state | MEASURED, closed: slower (H200 1.0016 / 1.0033) | 2026-11-07 | ab | H200 | Measured and closed, off: same registers and bitwise output, with the H200 A/Bs slower. | single_gpu, tie |
| 588 | `shared-locator` | One spline locator record per edge for the forward and w′ | MEASURED, closed (H200 0.9966 / 0.9974: under 1 %) | 2026-11-07 | ab | H200 | Under the 1 % bar on H200 (0.9966 / 0.9974) and bitwise, so no gain that clears the bar. | single_gpu, tie |
| 590 | `linear-div-split` | linear_div backward: divided path blocks written contiguously by one kernel | MEASURED, closed (H200 0.9972 / 1.0025) | 2026-11-08 | ab | H200 | Closed on H200 at 0.9972 / 1.0025; bitwise; the record gives no further reason. | single_gpu, tie |
| 596 | `r128-min-blocks` | Spline forward with STAGE_EDGES: R128 shape with MIN_BLOCKS 7 / 8 | MEASURED, closed: slower (H200 1.0097 / 1.0103 and 1.0515 /… | 2026-11-08 | ab | H200 | Closed as slower on H200 (1.0097 to 1.0515); the record gives no further reason. | single_gpu |
| 597 | `h200-ldg-redux` | w′ `BLOCK_LDG` and `REDUX` on the H200 (never measured there) | MEASURED, closed (BLOCK_LDG tie 1.0012 / 1.0001; REDUX slow… | 2026-11-08 | ab | H200 | BLOCK_LDG is a tie and REDUX is slower on H200; the record gives no further reason. | single_gpu |
| 598 | `h200-cutlass-tile` | Two-warp 32x64 / 64x32 tiles for the CUTLASS 3×TF32 GEMMs on the H200 | MEASURED, closed: slower (H200 1.0023 / 1.0030, 1.0014 / 1.… | 2026-11-08 | ab | H200 | Closed as slower on H200 and bitwise; the record gives no further reason. | single_gpu, tie |
| 600 | `cx-n1-cartesian-wprime` | w′: Cartesian harmonic adjoints (node converter, then dx and the edge-vector gradient from harmonic polynomia… | REJECTED (bound) | 2026-11-06 | lab | H200 | Fewer registers do not pay for the larger hot loop once STAGE_DOUT is on, and the conversion costs another 4 %. | single_gpu, stand_in, resources |
| 601 | `cx-n2-fp64-adjoint` | w′: receiver-tiled FP64 tensor-core matrix adjoints | REJECTED (bound) | 2026-11-06 | lab | H200 | Even an optimistic floor is 2.38x the whole default kernel, before scalar paths, S preparation, gathers and atomics. | single_gpu, stand_in |
| 606 | `memprofile-shape-carrier` | Release the allocator cache before the capture's warm-up, so Triton module loads find free device memory | REJECTED (no limit gain) | 2026-11-09 | unknown | RTX 5060 Ti | No limit gain on the 5060 Ti: the Triton module-load failure moved to the first eager force call; the switch was reverted. | single_gpu |
| 609 | `memslack-sort-keys` | Graph-pool packing: classic (non-expandable) allocator segments | REJECTED | 2026-11-09 | unknown | RTX 4000 Ada | Out of memory at SrTiO3 36³ in the build (19.17 GiB), so the classic allocator setting was rejected. | single_gpu |
| 612 | `rungb-first-layer` | The layer region inside the first table's rung (always with it) | REJECTED (taken at SrTiO3 32³, far below the limit, 8 % slo… | 2026-11-10 | ab | RTX 4000 Ada | Taken at SrTiO3 32³, far below the limit, and 8 % slower there. | single_gpu |
| 617 | `half-system-streams` | Two half-systems out of phase on two graph branches (B's edge kernels beside A's atom chain) | REJECTED (bound) | 2026-11-11 | lab | A100-SXM4-40GB (Colab) | Even uncoupled halves gain nothing: two streams are no faster than the full graph (1.008). | single_gpu, stand_in |
| 618 | `formulation-search` | Machine search over contraction orders of w' (and the forward's) trilinear sums, scored by FP instructions pe… | REJECTED (bound) | 2026-11-11 | bound |  | Bound: at most 5.1 % of the tensor-product arithmetic, about 1 % of the step; closed without a SASS census. | no_build |
| 619 | `tf32-fp16-cross-gemm` | 3xTF32 GEMMs as TF32 big*big plus the two cross products in one FP16 GEMM over 2K | REJECTED (accuracy) | 2026-11-11 | gate | A100-SXM4-40GB (Colab) | Error 6.3e-7 to 1.4e-6 exceeds the 2.7e-7 stop rule; the split alone is 2.7–5.5e-8 in exact arithmetic. | accuracy, single_gpu, stand_in |
| 623 | `ada-5060-set` | `GRAPH_SUM`, `SCALAR_PATHS`, `BLOCK_ORDER`, `DEGREE_PARTS`, radial 8 warps on the Ada | MEASURED, closed (ties) | 2026-11-11 | ab | RTX 4000 Ada | Ties: each switch within 0.5 % in both orders. | single_gpu, tie |
| 624 | `ada-5060-set` | `PREFETCH_AHEAD`, `FOLD_ALPHA`, w′ `MAX_REGISTERS` 80 on the Ada | MEASURED, closed: slower | 2026-11-11 | ab | RTX 4000 Ada | Closed as slower on the Ada; SrTiO3 ratios run from 1.0037 to 1.0529. | single_gpu |
| 625 | `ada-5060-set` | Spline forward `GEOM` on the Ada | MEASURED, closed: slower | 2026-11-11 | ab | RTX 4000 Ada | Slower on the Ada alone (1.0156 to 1.0226) and on top of PACKED8. | single_gpu |
| 626 | `ada-5060-set` | `VIRIAL_ELEMENTWISE` on the Ada (stress calls, tools/virial_ab.py) | MEASURED, closed: order-dependent | 2026-11-11 | ab | RTX 4000 Ada | Closed as order-dependent: the two build orders disagree (1.0041 and 0.9805). | single_gpu, retest_noted |
| 627 | `ada-5060-set` | `SEGMENT_ROWS` on the Ada through the CUTLASS backend built for sm_89 | MEASURED, closed: tie | 2026-11-11 | ab | RTX 4000 Ada | Tie: not adopted; the CUTLASS cubins are built for sm_80 and sm_90 only, so it cannot run on the 5060 Ti. | single_gpu, tie |
| 630 | `ada-5060-set` | `CUEQ_PROJECTED`, `BLOCK_ORDER`, `GRAPH_SUM`, `DEGREE_PARTS`, `SPLINE_VECTOR` 4, `SYMMETRIC_PAIRS`, SC_GENERA… | MEASURED, closed (no help in the set; SYMMETRIC_PAIRS mixed… | 2026-11-11 | ab | RTX 5060 Ti | No help in the set; SYMMETRIC_PAIRS readings disagree between runs; the SC pair runs only under tf32x3. | single_gpu, tie, retest_noted |
| 633 | `rungb-first-layer` | Chunked backward GEMM gradients: linear_1's input gradient (the four 3×TF32 blocks, 20.5 kB/atom) streamed pe… | REJECTED (bound) | 2026-11-10 | bound |  | Bound only: the streamed blocks give a 7.43 GiB peak, about 1.02 GiB saved; not built. | no_build |
