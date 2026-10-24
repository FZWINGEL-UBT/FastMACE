# Optimisation round 4 (before phase 5's acceptance)

The owner wants one more optimisation round before phase 5's acceptance (5-T11) runs. This plan lists every
idea of rounds 1–3 with its status, measures today's step on main, and proposes the round's tasks. Nothing in
it is implemented yet.

## 1. Baseline: main `ccab528`, 24 October 2026

**Run:** `tools/step_breakdown.py` on MH-1 `matpes_r2scan`, hybrid float32, 3×3×3, NVT at 1000 K with the
research probe's seed. This is OPT-T10's configuration, with every plan default and the same multi-head
checkpoint (`mace-mh-1.model`).
- **Machine:** local RTX 5060 Ti (sm_120, 36 SMs), WSL2, `~/environments/mace_cueq_oeq`.
- **Source:** a pinned export at `~/fastmace-opt4/src-ccab528`.
- **GPU slots:** one `tools/ci/local_gpu.py` reservation, 09:35–09:41 UTC.
- **Untimed:** no quiet stretch, and each number comes from a separate process. Cross-process noise is about
  3 % (AGENTS.md).
- **Records:** `results/opt4-baseline-ccab528/local-mh1-multihead/` and the run scripts.
- **A first run used the single-head file by mistake.** It ran `mace-mh-1.matpes_r2scan.model`, with the nsys
  pass, 09:12–09:28 UTC: 2.899 / 21.161 / 18.353 ms, and 258 / 224 / 236 kernels per replay. Its readout
  computes one head instead of six, so it is not comparable with round 3, and every part below comes from the
  multi-head run.
  - The single-head run is kept in `results/opt4-baseline-ccab528/local/`.
  - Its 5–6 fewer kernels and about 0.05 ms are OPT-T04's "unused heads" skip, screened and not built.

| Per MD step, local, un-profiled | NaCl (216 atoms) | NbCl5 (1,944) | NaNbO3 (1,080) |
| --- | ---: | ---: | ---: |
| Round 3 baseline (OPT-T10, `072fa0a`; opt-3.md cites the same numbers for `62aa6d4`) | 3.028 ms | 21.434 ms | 18.788 ms |
| **Today (`ccab528`)** | **3.083 ms** | **21.928 ms** | **18.409 ms** |
| Change (untimed, cross-process) | +1.8 % | +2.3 % | −2.0 % |
| ACP stage: round 3 → today | 0.144 → 0.051 ms | 0.175 → 0.046 ms | 0.183 → 0.051 ms |
| B stage (forces + half kick): round 3 → today | 2.827 → 2.974 ms | 21.295 → 21.622 ms | 18.556 → 18.294 ms |
| Force-graph replay span (profiler): round 3 → today | 2.874 → 2.915 ms | 21.188 → 21.163 ms | 18.452 → 17.955 ms |
| Force kernels, eager (profiler): round 3 → today | 2.745 → 2.839 ms | 21.013 → 20.837 ms | 18.242 → 17.826 ms |
| Gaps inside a replay (profiler): round 3 → today | 151 → 153 µs | 102 → 102 µs | 115 → 115 µs |
| Kernels per replay: round 3 → today | 264 → 263 | 230 → 229 | 240 → 240 |
| Setup + load, s: round 3 → today | 56.1 → 39.5 | 44.1 → 38.2 | 42.4 → 38.3 |

**What changed since round 3:**
- **The force graph is the same graph.** Kernel counts per part are equal to OPT-T10's within one kernel, and
  so are the gaps.
  - Phases 3–5 added nothing to the step's kernels. The phase 3, 3b and 4 acceptances each measured "no
    regression" with same-process A/Bs (`docs/validation/phase-3.md` §10, `phase-3b.md`, `phase-4.md` §5).
  - Phase 5 changes the planner (rule 12, monitor, tier switch, calibration).
- **`fused_integrator` is the one real change.** It was adopted at the end of round 3 (OPT3-T03), after the
  round-3 baseline was measured. It cut the ACP stage by 0.09–0.13 ms on every cell, about 3 % of NaCl's step.
  Its own same-process A/B gave NaCl −2.4 %.
- **The rest is noise in these untimed, separate processes.**
  - Replay spans and kernel times moved by −2.7 to +3.4 % in both directions.
  - On NbCl5, B ran 0.46 ms above its own replay span; on OPT-T10 it ran 0.11 ms above.
  - With this noise, none of the totals can be read as a regression or a gain. A timed comparison would need
    the same-process tool (`tools/ab_fastmace.py --release`).
- **Setup is 4–17 s shorter** than OPT-T10's. That is cache state and file-system noise, not a measured
  speed-up.

**Where the force call goes today** (local profiler, eager, multi-head checkpoint, ms per call).
`segment_gemm`'s Triton GEMMs appear as "Other" in `categories.csv`; they are counted here with the atom
linears, as OPT-T10 did.

| Part | NaCl | NbCl5 | NaNbO3 |
| --- | ---: | ---: | ---: |
| Atom-level linears (cuBLAS + `segment_gemm`) | 1.359 (48 %) | 9.020 (43 %) | 4.868 (27 %) |
| Radial networks, all | 0.625 (22 %) | 3.840 (18 %) | 5.884 (33 %) |
| of which the folded last layer's forward GEMM | 0.184 | 1.223 | 1.792 |
| Convolution (interaction 1 sums, interaction 2 streaming OEq) | 0.432 (15 %) | 3.507 (17 %) | 4.466 (25 %) |
| Atom-level elementwise | 0.242 (9 %) | 2.749 (13 %) | 1.163 (7 %) |
| Symmetric contraction (cuEq) | 0.107 | 1.156 | 0.575 |
| Edge geometry, basis, gathers; copies | 0.074 | 0.566 | 0.870 |

**Other model families** (NaCl 3×3×3, float32 hybrid, main pass only, 500 steps; same reservations, untimed):

| Label | Step | Force kernels by part | Kernels / gaps per replay | What differs from MH-1 |
| --- | ---: | --- | --- | --- |
| `omat0-medium` (generic MACE, 128 channels) | 1.710 ms | linears 0.39 ms (26 %) at **4.4 TFLOPS (15 % of peak)**, 75 launches; folded last layer forward 0.16 ms and input gradient 0.16 ms (21 % together) | 238 / 90 µs | small atom GEMMs that fill few SMs; the MH-1 rewrites (`gated_rows`, `segment_gemm`, OPT-T08) do not apply |
| `polar1-s` (POLAR-1) | 3.914 ms | linears 2.06 ms in **351 launches**; **"Other" 0.63 ms in 352 launches** (separate `aten::mul/add/div`), copies and fills 0.24 ms in 234 launches; float64 `index_put` backward 0.27 ms | **1,234 / 568 µs** | hundreds of small unfused kernels, about 5× MH-1's launch count; their source is candidate 2's first measurement |
| `omol0-xl` (OMOL-0, 1024 channels) | 12.735 ms | linears 9.60 ms (**78 %**) at 12.0 TFLOPS | 559 / 422 µs | atom-GEMM bound at large shapes; only a faster GEMM moves it. NaCl is not OMOL's domain; the shares are what matter |

## 2. Every idea of rounds 1–3 and its status

| Idea | Round, task | Status | Why |
| --- | --- | --- | --- |
| FP32-accurate tensor cores (Triton `tf32x3`, cuBLAS BF16×9) | 1, OPT-T01 | rejected on sm_89, sm_120 and (Triton `tf32x3`) the A100 | 3×TF32 cannot beat SIMT FP32 on sm_89 and sm_120; Triton's `tf32x3` won 0 of 92 shapes on the A100 (§4); BF16×9 only on compute capability 10.0/10.3 |
| Tensor-core work beside SIMT on two streams | 1, OPT-T01b | rejected | 0.77–0.85× |
| BF16×6 (Triton) | 1, OPT-T01 §6 | rejected on G4 and L4; the A100 would need a CUTLASS-class kernel | |
| **CUTLASS 3×TF32 on the A100** | 1, OPT-T01 §5, §7 | **not built** (owner, 24 October 2026, after OPT4-T03) | 0.75× (NaCl) and 0.55× (NbCl5) of cuBLAS GEMM time, but 7–14× worse than IEEE under synthetic cancellation. OPT4-T03 stage 1 passed the force gate and NVE drift with Triton's `tf32x3` (`docs/validation/opt4-T03.md`). The A100 is Colab-only for FastMACE, so stage 2 waits until it becomes a production target; it can start from the opt-in record `tf32x3_gemms` |
| `gated_rows` (stage A) | 1, OPT-T02 | **adopted** | |
| `segment_gemm` (stage B) | 1, OPT-T02 | **adopted** | |
| Gate in the GEMM prologue (stage C) | 1, OPT-T02 | rejected | NbCl5 +1.6 to +2.0 % |
| Gate backward in the epilogue (stage D) | 1, OPT-T02 | rejected | needs atomics, so not deterministic |
| Per-l products on graph branches | 1, OPT-T03 cand. 1 | rejected | 0.68–1.11× |
| `grouped_launch` | 1, OPT-T03 cand. 2 / T03b | opt-in, fixed | gains on sm_120 only for MH-1 (NaCl 0.977 / 0.982); sm_89 1.005. **Never screened on small-channel models** |
| Exact skips (zero blocks, heads, species) | 1, OPT-T04 | screened, not built | under 0.5 % |
| Regional persistence | 1, OPT-T05 | **not started** | its precondition (stage C landing) failed. Graph gaps are now 100–152 µs per replay (0.5–5 %), below its own 10 % stopping rule. **Recommendation: the owner closes it** (homogeneous persistence; candidate 7 does not decide it) |
| Template cache, persistent worker | 1, OPT-T06/T07 | **adopted** | |
| `species_tables`, `fold_readout_tail`, `fold_product_linear` | 1, OPT-T08 | **adopted** (OPT-DEF-2) | the "pending murphy A/B" was replaced by the Modal L4 and A100 A/Bs (−8 to −10 %) |
| Fused radial last layer + streaming convolution | 2, OPT2-T01 | rejected at gate 1 | in-kernel weight generation at 18 % of FMA peak |
| Radial MLP fusion (`fused_radial_mlp`, T02c entry kernels) | 2, OPT2-T02/T02c | not adopted, opt-in | NaCl −1.5 %, NbCl5 +1.0 %: the opaque op costs Inductor's neighbouring fusions; T02c's NbCl5 A/B 1.001 / 1.006, stopped by the conductor's rule |
| `folded_backward_tiles` | 2, OPT2-T03 | **adopted** | |
| `overlap_rebuild` | 2, OPT2-T04 | **adopted** | |
| `step_graph` | 2, OPT2-T04 | opt-in | neutral |
| `split_k` | 2, OPT2-T05 | rejected | NaCl 1.003 / 1.005 |
| Fused gate + residual VJP | 3, OPT3-T01 | not built | ceiling about 1 % |
| Symmetric contraction | 3, OPT3-T02 | screened, not built | kernels at 72–96 % of memory speed-of-light; every ceiling < 3 % |
| `fused_integrator` | 3, OPT3-T03 | **adopted** | |
| `combo_kernels` | 3, OPT3-T04 | rejected | NbCl5 +1.9 to +2.7 % |
| `direct_node_feats`, `folded_constants` | 3, OPT3-T05 | opt-in | exact; NaCl −0.5 to −0.8 % on sm_120, neutral on the L4. Making them defaults is still the owner's call |
| `lm_major_outer_sum` | 3, OPT3-T05 | rejected, reverted | |
| **Heterogeneous megakernel: single-SM co-execution probe** | design note, 23 October | **never run** | the one-day gate before any megakernel work |
| Research-engine screens of 10–16 October | `docs/rewrites.md` | rejected | not to be repeated without new evidence |

## 3. Candidates, ranked

Ceilings are fractions of today's local step. A candidate is cut as soon as its falsifying first measurement
fails; that measurement comes before any library code.

| # | Candidate | Expected gain | Class | Effort | Risk | Falsifying first measurement |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **Radial last layer, forward:** a write-bound Triton GEMM for `[E, 68] @ [68, 1280]`. All of K fits in one tile, and the output is streamed. Today cuBLAS (`magma_sgemmEx`) runs it at 28 % of FP32 peak. K = 68 is the folded `[h c, c]` input under `joint_radial_input`: 64 hidden columns plus 4 more. The worker confirms that layout first. | NaCl ≈ 2 %, NbCl5 ≈ 2 %, NaNbO3 ≈ 3 %. Generic MACE ≈ 1.5 %: its K is 64, where cuBLAS is already faster. | rounding | M | low: the op is already opaque (`aten::mm`) between the radial MLP and OEq, so no Inductor fusion is lost (the OPT2-T02 failure) | A graph-replay microbenchmark at the production layout must reach ≤ 0.15 ms (NaCl, E = 8,192), ≤ 1.00 ms (NbCl5, E = 56,320) and ≤ 1.45 ms (NaNbO3, E = 82,688). It must also beat the cheap alternative: cuBLAS at K = 64 plus a separate rank-4 update for the other columns. Otherwise stop. |
| 2 | **POLAR-1's unfused small kernels:** 1,234 kernels and 568 µs of gaps per replay, against MH-1's 263 and 153 µs. The source is not yet known. The suspects are the blocks that stay on cuEquivariance's naive backend (`model._untraced_blocks`, left to Dynamo by `simplify_traceable`), graph breaks, and the long-range path. The fix follows the source: a traceable or fused cuEq method, removing the breaks, or a deterministic scatter in place of the float64 `index_put` backward (0.27 ms, `results/opt4-baseline-ccab528/local/models/polar1-s-NaCl/ops.csv`). | POLAR-1 NaCl 15–30 % (ceiling about 1.3 ms of 3.9); large cells smaller | exact where only fusion or launches change; rounding where the cuEq method changes | M–H | medium: PolarMACE's own forward and the long-range k-space path (`longrange/polar.py`) | Profile one POLAR-1-S force call by module (NaCl, NbCl5), with Dynamo's graph-break log. Stop unless the attributable small-kernel work plus the `index_put` backward is ≥ 0.4 ms (≥ 10 %) of NaCl, and has a source that can be changed. |
| 3 | **A100 3×TF32 atom GEMMs, accuracy first:** route every atom GEMM, the cuBLAS products and `segment_gemm`'s Triton GEMM alike, through a custom op on OPT-T01's Triton kernel with `input_precision="tf32x3"`. That is the same three-product split, set per kernel. No global TF32 flag is involved, so there is no module switch, and TF32 stays off everywhere else. Run the force gate and NVE drift. Only if those pass, build the CUTLASS kernel as a custom op, with its own gate. | A100: force call −15 to −25 % (MH-1); OMOL-0 more (78 % atom GEMMs). sm_89 and sm_120: none (never applied) | rounding, **not IEEE-equivalent** under cancellation | stage 1 M (forward and input-gradient custom ops); stage 2 L (CUTLASS toolchain in the package) | high: accuracy; build dependency (nvcc); needs the owner's capability rule | Force gate (≤ 2× the float32 calculator) and NVE drift with the tf32x3 path on MH-1 NaCl and NbCl5 and on OMOL-0. This is an accuracy run, valid on any TF32 GPU, the local card included; its speed does not count. If it fails, try one corrected (Ootomo–Yokota) variant once, then stop. |
| 4 | **`grouped_launch` on small-channel models:** the existing opt-in record, screened on generic MACE, where the atom GEMMs are 216 × 128 × 128 products at 15 % of peak. | generic/OFF NaCl 3–8 % | rounding, deterministic | S (the record exists; extend its applicability check if it refuses) | low; sm_89 lost 0.5 % on MH-1 | Force gate plus a local A/B on `omat0-medium` NaCl: b/a must be ≤ 0.98 in both orders. Otherwise stop. |
| 5 | **Setup and capture time:** about 40 s setup and 22–30 s warm-up (mostly compiles and graph captures) per run, unchanged since round 3. 30–36 s of compile and a 20 s OEq build when cold. Candidates: torch's portable compile cache artifacts for cold machines (Colab, workstations); fewer graph variants captured up front; self-check cost. | seconds per run: tests, acceptance, MC and quench protocols, Colab compute units; nothing per step | exact | M | low | A stage breakdown of a cold and a warm `fm.md` start on current main (2-T19's probe). Build only the items worth ≥ 5 s warm or ≥ 15 s cold. |
| 6 | **OEq streaming forward below its roofline:** NbCl5 at 50 % occupancy and about 47 % of DRAM. (NaNbO3 is already at 93 %.) | NbCl5 ≤ 2 % | exact if only the launch shape changes | H (OEq's generated kernel) | medium | ncu stall reasons and roofline of the forward on NbCl5. Stop if there is no lever a shape rule can set (edges per block, warps per block) that differs from the research engine's rejected register caps and receiver-owned layout. |
| 7 | **Megakernel gate:** the single-SM co-execution probe from the 23 October design note. It decides only the heterogeneous (tensor-core beside SIMT) part. OPT-T01b's 0.77–0.85× on sm_89 already points to a fail there; the probe settles sm_120 and the A100. | none directly; it decides whether heterogeneous megakernel work ever starts | – | S (one day) | none | It is the measurement, with the design note's full pass rule: the SIMT side keeps ≥ 90 % of its standalone rate, **and** the MMA side adds enough useful throughput to pay for an FP32-accurate decomposition. It runs on sm_120, the L4 and the A100. |

**Candidate 1's evidence** (`results/opt4-baseline-ccab528/local/radial_mm.json`, local, one reservation,
CUDA-graph replays, median of 7 blocks of 50):

| E | `torch.mm`, K = 68 | K = 64 | K = 72 / 80 | writing `[E, 1280]` (`fill_`) | nominal DRAM floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8,192 (NaCl) | 183 µs | 165 µs | 183 / 182 µs | 118 µs | 94 µs |
| 56,320 (NbCl5) | 1,284 µs | 1,104 µs | 1,273 / 1,259 µs | 821 µs | 644 µs |
| 82,688 (NaNbO3) | 1,892 µs | 1,630 µs | 1,871 / 1,851 µs | 1,205 µs | 945 µs |

- Padding K to 72 or 80 does not help, so cuBLAS's kernel choice at K > 64 is the cost, not alignment.
- The FP32 compute floor (16.7 TFLOPS measured) is 0.59 ms for NbCl5, below the write.
- A kernel at the `fill_` rate plus about 10 % leaves 0.38 ms (NbCl5) and 0.57 ms (NaNbO3) to gain.

**Not proposed:**
- the rejected ideas in §2 and in `docs/rewrites.md`;
- `combo_kernels` restricted to small cells: a size rule for a 1–2 % NaCl gain;
- a second radial MLP fusion attempt (OPT2-T02's cause stands);
- stand-alone edge padding: the 2 % edge buckets were screened on 15 October.

## 4. Rules (rounds 2 and 3, unchanged)

- **General only.** No per-GPU, per-model or per-shape tables. A shape rule from M, N, K and the SM count is
  allowed. One crystal at a time, 3×3×3, TF32 off.
  - Candidate 3 needs a capability rule (FP32-emulating tensor cores on sm_80). The owner decides whether that
    rule is general enough.
- **Opt-in.** Every candidate is an opt-in `Rewrite` record (`never_automatic`), or an option of its config
  object, with an applicability check and refusal tests.
- **Accuracy.**
  - Exact candidates: bitwise on the CPU and in the reproducible mode.
  - Rounding candidates: the float32 force gate (`tools/force_stress_gate.py`, ≤ 2× the float32 calculator, or
    the gate's floor) on NaCl and NbCl5, plus CPU float64 agreement.
  - Candidate 3 adds NVE drift.
- **Timing.**
  - Same-process A/B (`tools/ab_rewrites.py`), both build orders, 12 blocks, from a pinned source copy on
    ext4.
  - Local noise floor: 1.1 % NaCl, 0.4 % NbCl5.
  - Timed A/Bs on sm_120 run only on the local RTX 5060 Ti with the PC quiet. The conductor schedules them, one
    at a time. The L4 and A100 A/Bs run on Colab, each as a same-process A/B inside one VM (OPT-T10: L4 numbers
    compare only within one container).
- **Defaults.**
  - A default candidate needs a gain on sm_120 and no loss on an L4.
  - The L4 or A100 A/B runs through `tools/colab` (no compute-unit cap). Stop every VM.
  - Candidate 3 cannot gain on sm_120 by construction, so it is an A100-only owner decision.
  - The owner decides all defaults.
- **Resources.**
  - Every local GPU job goes through `tools/ci/local_gpu.py run --wait-minutes 600`, at most 15 min per
    reservation, as a background task of the worker's session.
  - CPU: targeted tests only, one pytest at a time across sessions. Check first with
    `ps -eo args | grep -cE "^[^ ]*python[0-9.]* -m pytest"`, then use `-n 2`. WSL crashed twice on
    23–24 October under concurrent CPU load.
  - Never kill processes by pattern, only your own PIDs.
- **Git.** Never force-push. Commits end with the `Co-Authored-By` line (AGENTS.md).
- **Records.** Every screen, including the ones cut by their first measurement, gets a row in
  `docs/rewrites.md` and a report under `docs/validation/opt4-T<nn>*.md` with run IDs, hardware and
  environment.

## 5. Tasks

| Task | Candidate | Worker | Branch | GPU | Done when |
| --- | --- | --- | --- | --- | --- |
| OPT4-T01 | 1, radial last-layer forward kernel | one worker | `fastmace/opt4-T01` | microbenchmark, gate and A/B local; L4 A/B | the record and its tests are pushed with the gate and both-order A/Bs, or the cut is recorded |
| OPT4-T02 | 2, POLAR-1's small kernels and the float64 scatter | one worker | `fastmace/opt4-T02` | module profile and A/B local (POLAR-1-S NaCl and NbCl5); POLAR-1 gate (S, M; L on the Colab A100) | as above, with the POLAR labels through the gate. **Done 24 October 2026:** `fx_e3nn_linear` (PolarMACE's forward compiles) and `polar_graph_sums`; POLAR-1-S A/B b/a NaCl 0.763/0.759 and NbCl5 0.823/0.823 on sm_120, 0.818/0.811 and 0.840/0.837 on the L4; gate passed for S, M and L. Default on for POLAR-1-S, -M and -L by owner decision 58 (L after its local NbCl5 gate) (`docs/validation/opt4-T02.md`) |
| OPT4-T03 | 3, A100 3×TF32, stage 1 (accuracy through Triton `tf32x3`) | one worker | `fastmace/opt4-T03` | gate locally (TF32 is available on sm_120); NVE drift and OMOL-0 on the Colab A100 | a report with the gate and drift verdict. Stage 2 (CUTLASS) only after a pass **and** the owner's go. **Done:** stage 1 passed (`docs/validation/opt4-T03.md`); stage 2 not built (owner, 24 October 2026); the record `tf32x3_gemms` stays opt-in |
| OPT4-T04 | 4, `grouped_launch` on generic MACE | one worker (small) | `fastmace/opt4-T04` | gate and A/B local; L4 A/B if it gains | the A/B verdict; any applicability change pushed with tests. **Done 24 October 2026:** gate passed; NaCl b/a 0.946–0.955 on sm_120 (five generic labels), 0.957–0.959 on the L4; NbCl5 neutral. Default on for the generic MACE family by owner decision 57 (`docs/validation/opt4-T04.md`) |
| OPT4-T05 | 5, setup and capture time | one worker | `fastmace/opt4-T05` | untimed local runs (cold and warm caches) | the breakdown, and any change ≥ 5 s warm / ≥ 15 s cold built as an option, or none |
| OPT4-T06 | 6, OEq forward (reserve) | the first worker free | `fastmace/opt4-T06` | ncu on NbCl5 | the ncu verdict; build only with a shape-rule lever |
| OPT4-T07 | 7, megakernel probe | one worker | `fastmace/opt4-T07` | local, Colab L4 and A100 (untimed microbenchmark) | the pass/fail table; `docs/design/2026-10-23-heterogeneous-megakernel.md` updated |
| OPT4-T08 | owner's request (24 October): per-stage precision sensitivity, measurement only | one worker | `opt4-T08-precision-screen` | untimed local runs, MH-1 (NaCl, NbCl5, NaNbO3) and omat0-medium (NaCl, NbCl5) | **done 24 October:** no stage with a real share passes the gate alone in bf16, fp16 or TF32; only ZBL (fp16 storage) and MH-1's interaction-0 symmetric-contraction backward (TF32) pass. `docs/validation/opt4-T08-precision-sensitivity.md` |
| OPT4-REP | round report and default proposals | the conductor | `fastmace/opt4-report` | – | `docs/validation/opt-4.md`; the owner's default decisions recorded |

**Order and parallelism** (at most five workers at once; GPU work queues through the reservation):

```text
Day 1  T01 microbench      T02 module profile   T03 tf32x3 gate       T04 gate + A/B      T05 setup breakdown
       (falsifiers first: each worker reports its first measurement to the conductor before building)
Day 1–2  T01 build/gate/A/B   T02 build/gate/A/B   T03 A100 drift + OMOL-0   T07 probe (when T04 or T05 ends)
Day 2  L4 A/Bs (Colab, in parallel)   T06 if a worker is free   report and the owner's defaults
```

- **Timed A/Bs** (T01, T02, T04, any T05 option) take turns on the quiet local GPU. The conductor sets their
  order. Untimed work (microbenchmarks, gates, profiles, the probe) uses short reservations or Colab.
- **File overlaps:** `docs/rewrites.md` and `docs/design/switch-map.md` (all tasks), and `model/compile.py`
  (T02, T05). The conductor merges in task order and resolves them.
  - T01 touches the radial path (`fastmace/kernels/triton/`, `rewrites/fusion`).
  - T02 touches `longrange/polar.py`, `model/compile.py` and possibly the POLAR rewrites.
  - T05 may touch `model/compile.py` (compile cache) and the engine's capture code. Where T05 and T02 both edit
    `model/compile.py`, T02 merges first.
  - T04 touches `rewrites/linear`.
- **Size:** about two days. Phase 5's acceptance (5-T11) starts from the main that includes whatever the owner
  makes default. Records left opt-in do not change any plan, so they can merge without blocking the
  acceptance.
- **Expected outcome.** On the production MH-1 crystals, a further 2–3 % from candidate 1 if it holds. Beyond
  that, the step is bound by FP32 SIMT GEMM throughput and DRAM bandwidth on this card (OPT-T10 §5).
  - Larger gains are possible on other families: POLAR-1 from candidate 2, generic MACE from candidate 4.
  - On the A100, candidate 3, subject to the accuracy gate and the owner's decision.
