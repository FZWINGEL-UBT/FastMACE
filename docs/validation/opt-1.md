# Optimisation round 1: summary

Plan: `plans/opt-1.md`, 20–21 October 2026, between phases 2 and 3. Code freeze for the combined acceptance
(2-T23): main `4f90a92`, recorded in `be92979`. This summary collects the round's records on main, plus the
branch `fastmace/opt-T03b`, which is not in the freeze. Every A/B number below is the same-process `b/a`
step-time ratio: 12 alternating blocks of 300 MD steps, both build orders (a-b with ABAB, b-a with ABBA),
pinned source copies, an idle GPU checked before each job. Below 1 is faster.

Structures: MH-1 `matpes_r2scan`, hybrid cuEq/OEq float32, NVT at 300 K, 3×3×3 cells of NaCl (216 atoms) and
NbCl5 (1944 atoms), from ResearchModule tag `research-engine-baseline-20261017`.

GPUs:
- **murphy:** RTX 4000 Ada (sm_89, 48 SMs), driver 595.58.03, NAS `mace_cueq_oeq` through the hybrid launcher.
  Its 12-block noise floor is 0.16 % (`docs/validation/1a-noise-floor.md`).
- **local:** RTX 5060 Ti (sm_120, 36 SMs), driver 610.88, WSL2, `~/environments/mace_cueq_oeq`. Its noise floor
  is 1.1 % on NaCl and 0.4 % on NbCl5 (`docs/validation/2-T12-local-noise-floor.md`).

## 1. At a glance

| Task | What | murphy (sm_89) | local (sm_120) | Verdict |
| --- | --- | --- | --- | --- |
| 2-T20a | Step breakdown (baseline) | NaCl 3.68, NbCl5 31.58 ms per step | NaCl 3.71, NbCl5 27.43 ms per step; atom GEMMs 43–50 % of the force call | Reference for the round |
| OPT-T01 | FP32-accurate tensor-core GEMMs (Triton 3×TF32; BF16×9 absent) | tf32x3 slower on 89 of 92 shapes; peak FP32 16.3, TF32 38.6, tf32x3 8.4 TFLOPS | tf32x3 faster on 13 of 92; NbCl5 GEMMs 24.9 against 12.5 ms per call; peak 16.7 / 25.0 / 7.1 TFLOPS | **Rejected** |
| OPT-T01b | Tensor-core GEMM beside cuBLAS FP32 on two streams | combined 0.77–0.98× of cuBLAS alone | – | **Rejected** (rule < 1.10×) |
| OPT-T02 A, `gated_rows` | Residual products, gate and `linear_2` on per-segment rows | NaCl **−4.4 %**, NbCl5 **−9.0 %** | NaCl **−3.5 to −3.9 %**, NbCl5 **−10.5 to −10.7 %** | **Default on** (owner, 20 Oct) |
| OPT-T02 B, `segment_gemm` | `linear_2` writes its joined output in place (Triton gathered-row GEMM by the shape rule) | NaCl 0.997 / 0.996, NbCl5 **0.952 / 0.948** | NaCl 0.997 / 0.992, NbCl5 **0.958 / 0.957** | **Default on** (owner, 20 Oct); float64 and reproducible keep the stock path |
| OPT-T02 C, `gate_prologue` | Gate applied inside B's GEMM as it loads A | not measured | NbCl5 **1.020 / 1.016**, NaCl 0.998 / 0.997 | **Rejected** (rule needs both GPUs) |
| OPT-T02 D | Gate backward in the GEMM epilogue | – | – | **Rejected** before building (needs atomics: breaks determinism) |
| OPT-T03 candidates 1–2 (bench) | Per-l products on graph branches / one grouped launch | streams 0.85–1.11×; grouped 1.00/1.16× NaCl, 0.97/1.03× NbCl5 | streams 0.68–0.98×; grouped **1.34/1.30×** NaCl, 0.97/0.98× NbCl5 | Streams rejected; grouped built (below) |
| OPT-T03 candidate 2, `grouped_launch` (first version `cd71b5c`) | Small per-l products in one Triton launch | NaCl **1.070**, NbCl5 1.015 | NaCl **1.018 / 1.010**, NbCl5 1.004 / 0.997 | Rejected, then diagnosed |
| `grouped_launch` fixed (`f2531cb`, branch `opt-T03b`) | Separate outputs; one launch per weight shape | NaCl 1.005 / 1.003, NbCl5 1.001 / 1.002 | NaCl **0.977 / 0.982**, NbCl5 1.004 / 0.996 | **Opt-in**, not a default candidate (gain on sm_120 only) |
| OPT-T04 | Checkpoint scan for exact skips (CPU) | – | – | **Screened, not built** (< 0.5 % in total) |
| OPT-T05 | Regional persistence experiment (optional) | – | – | Not started (it needed C to land) |
| OPT-T06 | Converted-model template cache (setup) | later plans 0.34–0.80 s against 2.6–3.9 s (−2.2 to −3.3 s per plan) | – | **Merged**, default on (`EngineConfig.model_template_cache`) |
| OPT-T07 | Persistent subprocess worker | `run(100)` calls after the first: 9.6–10.2 s against 58–70 s with fresh children | – | **Merged**, default on (`RunnerConfig.persistent`) |
| OPT-T08 | Audit of the atom-level linears (C1 species tables, C2 readout tail, C3 product-linear fold) | estimate C1+C2+C3: −8.3 % NaCl, −5.8 % NbCl5 of the force stage | estimate −9 % NaCl, −5.7 % NbCl5 | Built opt-in, CPU-tested (`opt-T08`, `eecda68`); GPU A/Bs after the acceptance |

**Steady-state step, the defaults in the freeze (A + B), estimated as the product of the separate A/Bs:**

| | murphy | local |
| --- | --- | --- |
| NaCl | ≈ −4.7 % | ≈ −4.2 % |
| NbCl5 | ≈ −13.6 % | ≈ −14.4 % |

This is not a joint measurement. The combined acceptance measures the frozen main end to end against
`fastmace-1b` (runbook item 8).

## 2. Defaults changed by the round

| Field | Rule | Owner decision | Exactness | Off for |
| --- | --- | --- | --- | --- |
| `RewriteConfig.gated_rows` | `on_with_merged_residual` | 20 October 2026, night | exact (bitwise on the CPU and in the reproducible GPU mode) | models without the factored interaction (stage 2) |
| `RewriteConfig.segment_gemm` | `on_with_gated_rows_float32_unless_reproducible` | 20 October 2026, night | rounding, deterministic | float64 and `reproducible=True` plans; the same models as above |
| `EngineConfig.model_template_cache` | on | OPT-T06 (conductor) | copy vs reload within run-to-run noise; bitwise on the CPU | – |
| `RunnerConfig.persistent` | on (POSIX) | OPT-T07 (conductor) | bitwise equal to fresh children where reproducible | Windows (fresh children) |

Both rewrites pass the force gate on NaCl and NbCl5. The 19-label gate in the combined acceptance covers the
other models, and any model that fails there gets the rewrite switched off.
`test_default_plans_equal_main_py` and `test_plans_equal_the_research_engine` now keep an explicit list of
FastMACE-only defaults; every other planner difference from the research engine still fails.

## 3. Tasks

### 2-T20a: where the step goes

`docs/validation/2-T20a-step-breakdown.md`. On the local card, NaCl's force call is 3.32 ms of kernel time.
The atom-level FP32 GEMMs take 1.59 ms of it at 8.7 TFLOPS: the grids are small (0.8–2.7 waves). NbCl5's take
9.88 ms at 12.6 TFLOPS. Fitting fixed + per atom + per edge gives a megakernel ceiling of 8–16 % on NaCl, so a
megakernel is not in this round.

### OPT-T01 and T01b: tensor cores

`docs/validation/opt-T01-tensor-cores.md`, `docs/rewrites.md`.
- On both GPUs, three TF32 products cost more than one SIMT FP32 product. On murphy the 4096³ peak is FP32
  16.3, TF32 38.6 and tf32x3 8.4 TFLOPS, so 3×TF32 cannot reach FP32.
- BF16×9 exists only on compute capability 10.0/10.3.
- Running the tensor-core GEMM beside cuBLAS (T01b) gains nothing. Both GEMMs fill every SM, so only 13–29 %
  of the shorter one overlaps.
- The local rerun (sm_120) confirms the rejection: tf32x3 is faster on 13 of 92 shapes and half as fast per
  force call.

### OPT-T02: fusion around the gate

`docs/validation/opt-T02a-gated-rows.md`, `docs/validation/opt-T02b-segment-gemm.md`,
`docs/design/opt-T02-BC-go-no-go.md`, `docs/design/opt-T02-C-gate-prologue.md`.

**Stage A (`gated_rows`)**: keeps MH-1's residual products, gate and `linear_2` as per-segment rows, removing the
joins and copies between them.

| b/a | murphy, a-b / b-a | local, a-b / b-a |
| --- | --- | --- |
| NaCl | 0.957 / 0.954 | 0.961 / 0.965 |
| NbCl5 | 0.909 / 0.911 | 0.895 / 0.893 |

The local numbers come from the clean rerun (pinned `85d741d`, `~/fastmace-acceptance/opt1-local-rerun-20261020`).
The first local run was disturbed by a desktop video and is invalid. Stage A's report on main still lists
sm_120 as pending; these are those numbers.

**Stage B (`segment_gemm`)**, on top of A:
- NbCl5 −4.2 to −5.2 % on both GPUs; NaCl neutral, since no product reaches the shape rule and the stock path
  runs.
- The rule's tile configuration is within 2–5 % of the pre-check's best per shape.
- Two fixes were needed to get there. A view for `reshape_irreps` removed a second copy; without it B gained
  nothing. And the shape rule is evaluated without string formatting under torch.compile; NbCl5 had crashed.

**Stage C (`gate_prologue`):** values equal B's bitwise, but the gate tile, gathered by atom inside the K loop,
cost the GEMMs more than the elementwise kernel it removed: NbCl5 +1.6 to +2.0 % on sm_120. The code is
unmerged on `fastmace/opt-T02c`.

### OPT-T03: small-M schedules

`docs/rewrites.md`, `docs/design/opt-T03b-diagnosis.md` (branch `opt-T03b`).
- **Graph-branch concurrency (candidate 1):** loses in the bench on both GPUs.
- **Grouped launch (candidate 2), first version:** slower in the step on both GPUs (NaCl +1.0 to +7.0 %).
- **Diagnosis** (local nsys/ncu, owner request), two mistakes in the first version:
  - launches mixed K = 2048 products with K = 128–384 ones, so the long tiles set the launch's length, where
    cuBLAS splits K;
  - one flat output buffer made the gradients and consumer kernels run over the whole buffer (+0.11 ms of
    elementwise work).
- **Fix** (`f2531cb`): separate outputs, one launch per weight shape. In the step, the grouped kernel then runs
  a layer's four 512 × 512 products in about the time cuBLAS takes in sequence (174 against 170 µs, ncu). The
  bench's 1.34× came from operands warm in L2. The gain is the removed launches: 199 → 114 µs of gaps per
  force-graph replay.
- **Fixed result:** NaCl −1.8 to −2.3 % on sm_120, neutral to +0.5 % on sm_89, NbCl5 neutral. It stays opt-in.
  The force gate passes on murphy.

### OPT-T04: exact skips

`docs/validation/opt-T04-checkpoint-scan.md`.
- **Scope:** 13 checkpoints.
- **The zeros:** they sit in the symmetric contractions' fixed coupling tensors (94–99.6 % zero), which
  cuEquivariance already runs sparse.
- **Learned weights:** no zero linear path and no zero species block.
- **The only exact skips:** MH-1's 24 zero l = 1 channels in one product, and the five unused heads' readout
  columns. Together under 0.5 %.

### OPT-T06 and OPT-T07: setup and the subprocess lifecycle

`docs/validation/opt-T06-template-cache.md`, `docs/validation/opt-T07-persistent-worker.md`.
- **OPT-T06:** every plan after the first gets a private copy of the converted model instead of a reload.
  - Each later plan is 2.2–3.3 s faster, at a cost of 26 MB (float32) or 52 MB (float64) per template.
  - Copies are within the GPU's run-to-run noise of a reload, and bitwise equal on the CPU.
- **OPT-T07:** one child keeps its compiled engine across `run()` calls.
  - Later `run(100)` calls take 9.6–10.2 s instead of 58–70 s.
  - It is bitwise equal to fresh children in the reproducible mode.

### OPT-T08: linear audit

`docs/design/opt-T08-linear-audit.md` (branch `opt-T08`).
- **Records:** three opt-in records, C1 species tables, C2 readout tail and C3 product-linear fold. All are
  rounding class.
- **Estimate:** together −9 % of NaCl's force stage and −5.7 % of NbCl5's on the local card, −8.3 % and −5.8 %
  on murphy, from 2-T20a's profiles.
- **Status:** CPU-tested. Its GPU runs did not start before the freeze; they follow the acceptance.

## 4. Lessons for the next round

- **Measure in the step, not only in an isolated bench.** Two candidates (T03 grouped, C) looked good or
  neutral in isolation and lost in the step. The causes were cache residency, Inductor fusion lost at custom-op
  boundaries, and straggler tiles. Every candidate needs the full-step A/B on both GPUs before any decision.
- **sm_89 and sm_120 differ**, in launch-gap cost (WSL2 against native Linux) and in where small grids hurt. A
  gain on one GPU alone does not make a default.
- **Custom ops should return what their consumers want.** Separate outputs, no flat buffers, so autograd and
  Inductor keep per-segment kernels.
- **Timed runs only from pinned source copies.** One first local run was disturbed by desktop load and redone
  with the monitor.

## 5. Open after the acceptance

1. OPT-T08's GPU checks and A/Bs (C1–C3), on both GPUs.
2. `fastmace/opt-T03b` (opt-in `grouped_launch`): trial merge, as an opt-in record.
3. The 19-label gate results for `gated_rows` and `segment_gemm` (combined acceptance): switch off for any
   model that fails.
4. Stage A's validation report: add the sm_120 rows above (it still says "pending").
5. OPT-T05 (persistence) is not started. Its precondition, C landing, failed; it needs a new owner decision.
