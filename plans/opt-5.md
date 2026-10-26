# Optimisation round 5: simultaneous CUDA-core and Tensor-core execution

Post-1.0 optimisation plan (`v1.0.0` baseline `6ad8bfaf`, 26 October 2026). This round tests the specific
mechanisms from the literature survey on simultaneous CUDA-core (SIMT FP32) and Tensor-core (MMA) execution that
were **not** covered by round 1's two-stream screen (`OPT-T01b`) or round 4's symmetric single-SM probe
(`OPT4-T07`). Nothing in this plan is implemented yet.

## 1. Context: what the literature establishes vs. what FastMACE already measured

| Source / Experiment | Mechanism | Result | Gap / lesson for FastMACE |
| --- | --- | --- | --- |
| **OPT-T01 / T01b** (`docs/validation/opt-T01-tensor-cores.md`) | Standalone 3×TF32 (Triton `tf32x3`, CUTLASS `OpMultiplyAddFastF32`) and two-stream cuBLAS FP32 + Triton `tf32x3` | 3×TF32 alone loses on `sm_89` and `sm_120` (TF32 rate is 2× and 1× FP32); CUTLASS 3×TF32 wins on `sm_80` (0.75× NaCl, 0.55× NbCl5 GEMM time). Two-stream overlap is 0.77–0.85× on `sm_89`. | Two full independent GEMMs on separate streams compete for SM issue slots, registers and L2/DRAM bandwidth. |
| **OPT4-T03** (`docs/validation/opt4-T03.md`) | 52 position-dependent atom GEMMs routed through `tf32x3` (`fastmace::tf32x3_mm` and `segment_gemm`) | **Passes** the float32 force/stress gate on MH-1 (NaCl 1.09, NbCl5 1.00) and OMOL-0 XL (1.26), and NVE drift on the A100. Worst real-operand cancellation error is 4.7–5.7e-7, below cuBLAS IEEE's 6.4e-7–1.0e-6. | 3×TF32 is numerically valid on MACE operands, except small-$K$ ($K = 6$) species products where IEEE is 6.9× more accurate and a small biased energy offset (2–3e-4 meV/atom) from truncated small-part splits. |
| **OPT4-T07** (`docs/validation/opt4-T07-megakernel-probe.md`) | Single-SM co-execution probe (`tools/megakernel_probe.py`): symmetric `4×4`, `8×4`, `4×8` warp splits, `w % 4` scheduler assignment, `x3` split done inline by the MMA warps | `sm_120`: `keep = 0.91`, `gain = 1.04` (fail). `sm_89`: `keep = 0.92`, `gain = 1.10` (threshold pass). `sm_80`: `keep = 0.58`, `gain = 0.92` (fail; TC alone is 2.17× SIMT). | Inline `cvt.rna.tf32` + `sub.f32` splits on the MMA warps consume SIMT issue slots on the same warp schedulers; uniform 228-register compilation caps occupancy at 256 threads (`4×4`). |
| **Tacker** (HPCA 2022) & **ISPA** (IEEE TC 2023) | Persistent thread blocks, asymmetric warp ratios, named-barrier (`bar.sync`) sub-block synchronization, and per-role register/occupancy control | +18.6 % (Tacker) and +15.3 % (ISPA) throughput on mixed Tensor/CUDA workloads; avoids the 2025 NVIDIA forum anti-pattern (`__syncthreads()` across 256 SIMT + 256 MMA threads → 23.5 ms vs 18.0 + 7.5 ms). | Separate the warp-scheduler issue problem from pipeline overlap: pin MMA warps to dedicated schedulers (`w % 4 < M_sched`), synchronize roles with `bar.sync` instead of `__syncthreads()`, and decouple SIMT register usage from the 128-accumulator MMA frame. |
| **DeepSeek-V3** (2024/2025) | Producer/consumer warpgroup pipeline: Tensor Cores issue back-to-back MMA while CUDA cores execute precision-preserving splits, scaling, accumulation and epilogue corrections in parallel | Sustains Tensor Core utilization without stalling on FP32 correction arithmetic. | In `OPT4-T07`'s `x3` and CUTLASS `OpMultiplyAddFastF32`, the MMA warps stall on their own FP32 operand splits (`cvt.rna.tf32` + `sub.f32`) and cross-term additions. Moving operand splitting and residual correction onto SIMT helper warps removes SIMT instructions from the MMA warp stream. |
| **Libra** (2025), **HC-SpMM** (2025), **BSMR-SDDMM** (2026) | Hybrid operator/region partitioning between Tensor Cores and CUDA Cores based on arithmetic intensity and tile density | 1.33–1.77× over single-pipeline baselines. | Do not split one homogeneous dense GEMM 50/50 on a GPU where one pipeline dominates. Instead: (a) overlap *complementary* MACE operators (e.g. SIMT elementwise gates/skips beside MMA linear tiles), or (b) dispatch large-$M, K$ tiles to accurate Tensor Cores and small-$K$/tail tiles to SIMT FP32. |

## 2. Design principles for Round 5

1. **Complementary roles, not two competing identical GEMMs.**
   - Either CUDA cores perform precision-preserving work (operand splitting into `big`/`small` TF32 fragments in
     shared memory, unbiased rounding, residual/epilogue corrections) while Tensor Cores issue pure `mma.sync`
     chains (**Candidate 1, DeepSeek-style**);
   - or CUDA cores execute MACE's adjacent non-GEMM work (gate activation, `SpeciesSkip` gather/scatter, residual
     add) concurrently with the interaction block's GEMM tiles inside a persistent kernel (**Candidate 3,
     Tacker/Libra-style**).
2. **Scheduler isolation and named barriers (`bar.sync`), never whole-block `__syncthreads()`.**
   - Each SM has 4 warp schedulers (`warp_id % 4`). When an MMA warp and a SIMT warp sit on the same scheduler,
     they compete for its 1 instruction/cycle issue slot, and the MMA warp's address/split arithmetic contends
     with the SIMT warp's FFMA pipeline.
   - Pin MMA warps and SIMT warps to distinct scheduler subsets (e.g. 1 or 2 schedulers for MMA, 2 or 3 for SIMT
     on `sm_120`/`sm_89`; 3 for MMA, 1 for SIMT producer/epilogue on `sm_80`), and synchronize producer/consumer
     shared-memory buffers with PTX `bar.arrive` / `bar.sync` on named barrier IDs `1..4` (**Candidate 2**).
3. **Unbiased rounding of the `small` TF32 residual.**
   - `OPT4-T03 §4` found a systematic +2–3e-4 meV/atom energy offset in Triton's `tf32x3` because Triton truncates
     the low part. Using round-to-nearest-even (`cvt.rna.tf32.f32`) for both `big` and `small = x - big` (plus an
     optional 4th `small_a · small_b` scalar/SIMT residual check) eliminates the one-signed energy bias.
4. **Falsifying first measurement before any library code.**
   - Every candidate starts with a standalone NVRTC / Triton / CUTLASS probe or microbenchmark on production shapes
     from `gemms.csv`. Any candidate that fails its first measurement is cut immediately and recorded in
     `docs/rewrites.md`.

## 3. Candidates, ranked

Ceilings are fractions of the `v1.0.0` step time (`plans/opt-4.md §1`: NaCl 3.08 ms, NbCl5 21.93 ms, NaNbO3
18.41 ms on `sm_120`; OMOL-0 XL 12.74 ms with 78 % in atom GEMMs).

| # | Candidate | Target GPUs | Expected gain | Class | Effort | Risk | Falsifying first measurement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **Pipelined SIMT-split + pure-MMA 3×TF32 (`deepseek_x3`):** SIMT producer warps load FP32 tiles from global/shared memory, compute unbiased `big` and `small` TF32 fragments (`cvt.rna.tf32.f32`) on CUDA cores, and stage them into double-buffered shared memory via `bar.arrive`/`bar.sync`; consumer MMA warps issue back-to-back `ldmatrix.x4` + `mma.sync.aligned.m16n8k8` without inline FP32 split instructions. Optional SIMT accumulation of the 4th term (`small_a · small_b`) or epilogue gate. | `sm_80` (A100), `sm_89` (L4 / Ada), probe on `sm_120` | `sm_80`: −15 to −25 % of force call (MH-1), −30 % on OMOL-0 XL; `sm_89`: −3 to −6 % if producer/consumer beats cuBLAS FP32 on $M \ge 1000$ | rounding, deterministic | M | medium: shared-memory footprint for double-buffered `(big, small)` fragments (2× normal TF32 stage size) | In `tools/megakernel_probe.py` (extended with `x3_pipe`), removing the split from the MMA warps must raise `x3` useful MMA throughput by ≥ 15 % over inline `x3`, and a standalone kernel on NbCl5's `[5832, 512] @ [512, 512]` must beat cuBLAS IEEE FP32 on `sm_89` or beat CUTLASS inline `OpMultiplyAddFastF32` on `sm_80`. |
| 2 | **Tacker-style scheduler-isolated co-execution probe (`sched_iso`):** extend `tools/megakernel_probe.py` to test (a) strict warp-scheduler partitioning (`warp_id % 4 < S_sched` on SIMT vs `>= S_sched` on MMA, rather than interleaved `warp_id < S`), (b) asymmetric ratios (`6×2`, `7×1`, `2×6`), and (c) named PTX barriers (`bar.sync`) instead of `__syncthreads()`. | `sm_120`, `sm_89`, `sm_80` | Gate only (decides whether scheduler isolation unlocks co-execution on `sm_120` or `sm_89`) | – | S (half day) | none | Must reach `keep ≥ 0.90` and `gain ≥ 1.10` (`x3` or `x3_pipe`) on `sm_120`, or ≥ 1.15 on `sm_89`. If no scheduler assignment or warp ratio passes on `sm_120`, dense SIMT+MMA co-execution on `sm_120` remains closed. |
| 3 | **Tacker/Libra complementary operator fusion (MMA/cuBLAS linear + SIMT elementwise/gate/skip):** fuse the post-GEMM SIMT elementwise work of MACE's interaction block (`gated_rows` active-row scaling, `ResidualBlockLinear` skip addition, or segment reductions: 9–13 % of the step, 0.24 ms NaCl / 2.75 ms NbCl5) with the block linear layer using persistent warp specialization (`bar.sync` tile handoff in shared memory, no extra global write/read of the intermediate activation). | `sm_120`, `sm_89`, `sm_80` | NaCl 2–4 %, NbCl5 3–6 % (cuts one `[N, 512]` global round-trip per interaction and overlaps SIMT gate/residual arithmetic with the next tile's GEMM) | rounding, deterministic | M–H | medium: `OPT-T02` stage C regressed NbCl5 by +1.6 % when the gate ran in the prologue of a homogeneous Triton GEMM; warp-specialized epilogue/consumer avoids widening the GEMM loop's register count | Microbenchmark on NbCl5's interaction-1 linear + gate + residual shapes (`[1944, 512]` per irrep block) inside a CUDA-graph replay must beat sequential `segment_gemm` + Inductor elementwise by ≥ 5 % on `sm_120` without regressing NaCl. |
| 4 | **Libra/HC-SpMM hybrid shape dispatcher (`hybrid_tc_simt_gemms`):** combine `OPT4-T03`'s `tf32x3` custom-op routing with a general shape & capability rule `(M, N, K, sm_count, tc_fp32_ratio)`: route only large compute-bound atom GEMMs ($M \ge 512, N \ge 128, K \ge 64$, where `OPT-T01 §5` won 62 of 64 shapes at 0.55× and `OPT4-T03 §3` showed lower error than IEEE) to the pipelined/CUTLASS 3×TF32 kernel, while keeping small-$K$ ($K < 64$, including the $K = 6$ species products) and small-$M$ products on cuBLAS SIMT FP32. Where two independent branches exist (`linear_up` vs `SpeciesSkip`), optionally launch the SIMT small-$K$ skip on a secondary stream overlapped with the Tensor Core `linear_up`. | `sm_80` (A100), `sm_89` (if Candidate 1 passes on large shapes) | `sm_80`: full-step −12 to −22 % on MH-1 and −25 to −35 % on OMOL-0 XL, with zero small-$K$ accuracy loss and < 1e-4 meV/atom energy offset | rounding, deterministic | M | low: reuses `OPT4-T03`'s `tf32x3_gemms` plumbing and `rule 13` self-check | On the 92 shapes of `gemms.csv`, the hybrid rule must beat pure cuBLAS IEEE and pure 3×TF32 on both total time and max cancellation-scaled error ($\le 1.0\times$ IEEE on every shape, fixing $K = 6$). |
| 5 | **Radial folded last layer (`[E, 68] @ [68, 1280]`) SIMT+MMA or write-streaming split:** `plans/opt-4.md` candidate 1 showed cuBLAS runs this layer at 28 % of FP32 peak (1.28 ms on NbCl5, 1.89 ms on NaNbO3) because $K = 68$ spills past a 64-column tile. Test a Libra-style column split inside one kernel: $K_{0..63}$ (`[E, 64] @ [64, 1280]`) on Tensor Cores (3×TF32 or IEEE-equivalent where applicable) or a 64-column tile while SIMT warps accumulate the 4-column tail (`[E, 64:68] @ [64:68, 1280]`, only 6 % of the FLOPs) and stream the combined `[E, 1280]` output with cache-streaming stores (`.cs`). | `sm_120`, `sm_89`, `sm_80` | NaCl ≈ 2 %, NbCl5 ≈ 2 % (0.38 ms), NaNbO3 ≈ 3 % (0.57 ms) | rounding, deterministic | M | low: single opaque custom op replacing `aten::mm` on the folded radial last layer | Graph-replay microbenchmark at $E \in \{8192, 56320, 82688\}$ must reach ≤ 0.15 ms (NaCl), ≤ 1.00 ms (NbCl5) and ≤ 1.45 ms (NaNbO3) on `sm_120` and pass the float32 force gate. |

## 4. Rules (unchanged from Rounds 2–4)

- **General only (SPEC §9.1).** No per-GPU, per-model or per-crystal lookup tables. Any dispatch rule must be a
  closed-form function of `(M, N, K)` and device properties queried at plan time (`sm_count`, compute capability /
  measured TF32-to-FP32 throughput ratio from calibration).
- **Opt-in first.** Every new rewrite or kernel path lands as an opt-in `Rewrite` record (`never_automatic`) with
  an applicability check, self-check (SPEC rule 13), and CPU refusal tests. Promoting any candidate to default is
  an explicit owner decision after same-process A/B evidence on `sm_120` and `sm_89` (or `sm_80` for A100-specific
  tensor-core paths).
- **Accuracy gates.**
  - All rounding candidates must pass `tools/force_stress_gate.py` ($\le 2\times$ the float32 calculator or floor)
    on NaCl and NbCl5 (plus NaNbO3 for radial changes, and OMOL-0 XL for atom-GEMM changes).
  - Tensor-core GEMM candidates (`OPT5-T01`, `OPT5-T04`, `OPT5-T05`) must also pass `tools/tf32x3_accuracy.py`
    (cancellation-scaled error $\max |c - c_{64}| / \sum |a \cdot b|$ on real force-call operands) and the 20 ps
    NVE drift check (`tools/bench/nve_drift.py`).
  - Bitwise determinism on repeated runs and CUDA-graph replays is mandatory (`tests/gpu/`).
- **Timing discipline.**
  - Microbenchmarks use CUDA-graph replay medians under a quiet `tools/ci/local_gpu.py` reservation.
  - End-to-end claims require same-process alternating A/Bs (`tools/ab_rewrites.py`), both build orders (`a,b` and
    `b,a`), 12 blocks, from a pinned source export on ext4.

## 5. Tasks and execution order

| Task | Candidate | Branch | Hardware | Falsifier / Gate | Done when |
| --- | --- | --- | --- | --- | --- |
| **OPT5-T01** | 2, Scheduler-isolated & asymmetric warp probe (`sched_iso`) | `fastmace/opt5-T01` | local `sm_120`, Colab `sm_89` (L4) and `sm_80` (A100) | Extend `tools/megakernel_probe.py` with scheduler-pinned warps (`w % 4 < S_sched`), asymmetric ratios (`6×2`, `7×1`, `2×6`), and `bar.sync`. Cut `sm_120` co-execution if `gain < 1.10`. | Probe JSONs and `docs/validation/opt5-T01-scheduler-probe.md` committed; `docs/design/2026-10-23-heterogeneous-megakernel.md` updated. |
| **OPT5-T02** | 1, DeepSeek-style pipelined SIMT-split + MMA 3×TF32 (`deepseek_x3`) | `fastmace/opt5-T02` | local `sm_120` (accuracy & probe), Colab `sm_89` / `sm_80` (speed) | `x3_pipe` mode in probe + standalone NVRTC/CUTLASS kernel on NbCl5 `[5832, 512] @ [512, 512]`. Cut on `sm_89` if slower than cuBLAS FP32; proceed on `sm_80` if ≥ 1.3× faster than cuBLAS FP32 with unbiased `cvt.rna.tf32` split. | Kernel microbenchmark, real-operand cancellation check (verifying removal of the +2–3e-4 meV/atom truncation bias), and `docs/validation/opt5-T02.md`. |
| **OPT5-T03** | 5, Radial folded last layer (`K = 64 + 4` hybrid / streaming GEMM) | `fastmace/opt5-T03` | local `sm_120`, Colab `sm_89` | Graph-replay microbenchmark on $E \in \{8192, 56320, 82688\}$ must hit ≤ 0.15 / 1.00 / 1.45 ms on `sm_120`. | Opt-in rewrite `radial_last_gemm`, force gate on NaCl/NbCl5/NaNbO3, both-order A/B, and `docs/validation/opt5-T03.md`. |
| **OPT5-T04** | 3, Persistent warp-specialized linear + elementwise/gate epilogue | `fastmace/opt5-T04` | local `sm_120`, Colab `sm_89` | Interaction-block microbenchmark on NbCl5 must beat `segment_gemm` + Inductor elementwise by ≥ 5 % on `sm_120`. | Opt-in rewrite, force gate, both-order A/B on NaCl and NbCl5, and `docs/validation/opt5-T04.md`. |
| **OPT5-T05** | 4, Libra-style shape-dispatched 3×TF32 + SIMT small-$K$ hybrid (`hybrid_tc_simt_gemms`) | `fastmace/opt5-T05` | local `sm_120` (gate & accuracy), Colab `sm_80` (A100 A/B & drift) | Starts after `OPT5-T02`. Must eliminate `OPT4-T03`'s 6.9× $K = 6$ error spike and beat cuBLAS IEEE on `sm_80` in a same-process A/B on MH-1 and OMOL-0 XL. | Opt-in rewrite extending `tf32x3_gemms`, CPU tests, `tf32x3_accuracy.py` + force gate + A100 A/B in `docs/validation/opt5-T05.md`. |
| **OPT5-REP** | Round 5 summary and default recommendations | `fastmace/opt5-report` | – | – | `docs/validation/opt-5.md` and `docs/rewrites.md` updated; owner decisions recorded. |
| **OPT5-T06** | Exponent-shifted `3×FP16` (`c = 3` at `512` FLOP/cycle/SM 16-bit Tensor Core rate; [design](../docs/design/2026-10-26-fp16x3-and-hybrid-gemm.md)) | `fastmace/opt5-T06` | local `sm_120` (probe + Triton bench); workstation `sm_89` (`leto`/`murphy`), Modal L4 & A100 (`sm_80`) when available | Pass single-SM co-execution gate on `sm_120` (`x3 inter fp16 4×4`: **`keep = 0.92`, `gain = 1.22×`**, **PASS**); standalone Triton `triton_fp16x3_kernel` **1.78–1.88× faster than `triton_tf32x3`** on `sm_120`, matching `cuBLAS IEEE` on `NaCl [1512, 512, 512]`. | Probe and standalone Triton kernel benchmarked on `sm_120`; standalone `3×FP16` documented for `sm_80`/`sm_89` adoption regardless of whether the `sm_120` fused megakernel beats `cuBLAS IEEE`. |
| **OPT5-T07** | Time-step delta GEMM ($Y_1 = Y_0 + \Delta A \cdot B$) & 50/50 channel-norm split | `fastmace/opt5-T07` | local `sm_120` | Accuracy $\le 2\times$ IEEE on real `MH-1 NaCl` operands | **Done (falsified):** `20.8 %` per-step feature change and flat weight norms give `411–1,838×` worse error than IEEE. |
| **OPT5-T08** | Single-kernel fused `64 + 32` column-split `SIMT + 3×FP16` NVRTC GEMM on `sm_120` ([design](../docs/design/2026-10-26-fp16x3-and-hybrid-gemm.md)) | `fastmace/opt5-T08` | local `sm_120` | Beat `cuBLAS IEEE FP32` in CUDA-graph replay microbenchmark on `[1512, 512, 512]`, `[5832, 512, 512]`, `[13608, 512, 512]`, `[1944, 1024, 1024]` and pass FP32 accuracy check | Microbenchmark verdict recorded in `docs/validation/opt-5.md`; if it beats `cuBLAS IEEE` on `sm_120`, wire into `BlockLinear`; if not, keep standalone `3×FP16` for `sm_80`/`sm_89`. |

```text
Wave 1 (falsifiers first, no library edits):
  OPT5-T01 (scheduler-isolated & asymmetric bar.sync probe on sm_120 / L4 / A100)
  OPT5-T02 (DeepSeek-style SIMT-split producer + pure-MMA consumer probe & NbCl5 shape bench)
  OPT5-T03 (K = 64 + 4 radial last-layer streaming/hybrid microbenchmark on sm_120)

Wave 2 (library integration & gates for candidates passing Wave 1):
  OPT5-T03 build + gate + A/B (if Wave 1 passes)
  OPT5-T04 warp-specialized linear + gate/residual consumer microbench -> build + A/B
  OPT5-T05 hybrid shape-dispatched 3xTF32 + SIMT small-K (using OPT5-T02's unbiased split)

Wave 3:
  Colab L4 / A100 same-process A/Bs for passing Wave 2 candidates -> OPT5-REP
```
