# Optimisation round 1 (between phases 2 and 3)

Owner decision, 20 October 2026: before phase 3 starts, one round of speed
optimisations, beginning with the tensor cores. Phase 3 starts after this
round, not in parallel with it. Phase 2's acceptance (2-T23) finishes first:
it holds both GPUs for timed work today.

Background and estimates: the owner's doc "FastMACE: speeding up small-cell
MD steps" (20 October 2026), built from the research engine's step breakdown
of 14–15 October (ResearchModule
`Experiments/Langevin_Thermostat/studies/further_optimizations/2026-10-15-step-breakdown.md`).

**Owner decision, 20 October 2026, evening:** no big accuracy runs on murphy
until this round's optimisations are in. Phase 2's acceptance was stopped
partway (2-T23). It runs once, as a combined acceptance of phase 2 and this
round, on the optimised main. Short non-accuracy jobs (CPU trials, short
speed runs) may still use murphy.

**Owner decision, 20 October 2026, night:** OPT-T02 stage A (`gated_rows`)
becomes the default, with auto = on wherever block linear applies. The
evidence:
- bitwise exact on the CPU and in the reproducible GPU mode;
- the force gate passes;
- the speed gains below.

| | murphy (sm_89) | local (sm_120) |
| --- | --- | --- |
| NaCl | −4.4 % | −3.7 % |
| NbCl5 | −9.0 % | −10.6 % |

The 19-label gate runs in the combined acceptance. Any model that fails there
gets the rewrite switched off.

Results so far in this round:
- **OPT-T01 (tensor cores) and OPT-T01b (overlap):** rejected on sm_89.
- **OPT-T03 candidates 1–2:** missed their rule on sm_89.
- **Stage B (`segment_gemm`):**
  - NbCl5 −4.8 to −5.2 % on murphy and −4.2 to −4.3 % locally, on top of A;
    NaCl neutral (the stock fallback).
  - It is rounding class, not bitwise: Triton replaces cuBLAS on large shapes.
    The force gate passes.
  - **Default on (owner, 20 October 2026)** where the shape rule picks Triton.
    Float64 and the reproducible mode keep the stock path.

## Starting point

- **NaCl 3×3×3 (216 atoms):** 3.81 ms per step on the RTX 5060 Ti. The force
  call is 3.32 ms of kernel time. Of that, the atom-level FP32 GEMMs take
  1.66 ms and the radial-network GEMMs about 0.36 ms. All run as SIMT on the
  CUDA cores: TF32 is off by the production decision.
- **What the GEMM time is:** fitting the three crystals to
  fixed + per atom + per edge gives 0.32 ms fixed, 11.0 µs per atom and
  0.14 µs per edge. So the step is mostly per-size arithmetic, not launch
  overhead.
- **What that rules out:** a megakernel can save at most 8–16 % on NaCl
  (1–2 % on NbCl5), so it is not in this round.
- **Baseline:** 2-T20a, the current FastMACE step breakdown on the local card
  (with the profiling columns: achieved FLOP/s, grid and occupancy, gaps
  inside the graph, bytes per operator).

## Rules for every task

- **Exact enough:** a float32 force path is accepted if its error against
  float64 is at most 2× that of the float32 calculator
  (`tools/validation/roundoff_reference.py` in ResearchModule; the
  rewrite gates in FastMACE). On top of that: the hybrid MD statistics
  against the current default, and all 19 labels of SPEC §12.5 through
  the gate.
- **General:** no per-GPU or per-shape tuning tables, no autotuning (SPEC §1,
  AGENTS.md). A choice made by a rule from the shapes is allowed.
- **Both GPUs:** RTX 5060 Ti (sm_120) and murphy's RTX 4000 Ada (sm_89).
  Timing in alternating A/B blocks in one process, against each machine's
  noise floor; murphy only when its GPU is idle and the host quiet.
- **Precision decision:** SPEC §1 keeps "FP32 with TF32 off" as the
  production decision; lower precision is opt-in only and must pass the
  gate. A split-precision mode that is FP32-accurate is new ground. It can
  be screened and implemented as an option. Making it the default is an
  owner decision with a SPEC amendment, after the A/B.
- **Documentation:** every screen, including ones without gain, is
  documented with run IDs, hardware and environment (AGENTS.md).

## Tasks

### OPT-T01 FP32-accurate tensor-core GEMMs (M, GPU) — first

Goal: run the atom-level and radial FP32 GEMMs on tensor cores with FP32
accuracy.

1. **Survey (CPU, no GPU).** Find which modes our stack exposes:
   - torch 2.14's FP32 matmul precision settings;
   - cuBLAS's FP32 emulation through BF16 (BF16×9) on CUDA 13, and how it is
     switched on (API or environment variable);
   - Triton's `tl.dot(input_precision="tf32x3")` and Inductor's handling of it;
   - CUTLASS 3×TF32 kernels.

   For each: which GPUs (sm_89, sm_120), whether it works inside CUDA-graph
   capture and torch.compile, and its documented accuracy.
   The external research report (`docs/design/2026-10-20-megakernel-research.md`)
   says cuBLAS BF16×9 is limited to compute capabilities 10.0 and 10.3, so
   unavailable on both our GPUs. Verify it against the cuBLAS documentation.
   If it holds, 3×TF32 (Triton `tf32x3`, CUTLASS) is the candidate.
   **Survey result (20 October 2026, `opt-T01-survey.md`):**
   - BF16×9 is confirmed absent on both cards (cuBLAS table: 10.0 and 10.3
     only).
   - torch 2.14 has no FP32-accurate mode (`fp32_precision` accepts only
     ieee, tf32 or none), and Inductor emits only tf32 or ieee.
   - Triton's `tf32x3` is the only usable mode. `bf16x6` has a bug for K < 16
     (triton#12252).
   - On GeForce and workstation Ada/Blackwell, dense TF32 runs at the FP32
     SIMT rate (third-party figures; to be checked with a peak test on the
     card). 3×TF32 therefore tops out near a third of SIMT.
   - The atom GEMMs already reach 11–13 TFLOPS against a practical ceiling of
     about 14–16.

   So step 2 runs narrowly: real shapes, forward and input gradient, against
   cuBLAS SIMT and a Triton ieee control. If tf32x3 is slower everywhere,
   OPT-T01 closes there: the rejection goes in `docs/rewrites.md`, CUTLASS is
   skipped, and OPT-T02 and OPT-T03 move up.
2. **Micro-benchmark (local, then murphy).**
   - Shapes: export the production GEMM shapes and strides from the compiled
     force call. Irrep layouts can fold components into rows, so M is not
     always the atom count. Cover 216 to 2,000 atoms.
   - Comparison: SIMT FP32 against each mode, for the forward products and
     their input gradients. Report speed and max/relative error against
     float64.
   - Precision control: set the precision explicitly per custom (Triton)
     kernel; torch's TF32 flag does not govern them.
3. **Model-level screen.** Switch the mode on for the force call. Measure the
   float32-against-float64 force error (the 2× rule), the step time A/B for
   NaCl, NaNbO3 and NbCl5, and the step breakdown.
4. **Decide and report:** `docs/validation/opt-T01-tensor-cores.md`. If it passes
   and gains, propose it as an option (`ForceField(..., matmul="fp32-tc")` or
   similar) and bring the default question to the owner.

Done when the report gives speed and accuracy per mode and GPU, and a
recommendation.

### Later in this round (ordered by expected gain; each starts after the previous one reports)

- **OPT-T02 Epilogue fusion of linear + elementwise:** gate, product,
  residual and bias in the GEMM epilogue, plus removal of layout copies
  around the atom linears. Target: part of the 1.66 + 0.41 ms; 5–15 % of the
  step (the report ranks this first).
  - Candidates: Triton GEMMs with an epilogue, CUTLASS epilogue visitor trees,
    and cuBLASDx device-callable block GEMMs.
  - Count the backward pass too. Saved preactivations can cancel the forward
    traffic savings.
- **OPT-T03 Small-M schedules and cross-layer grouping:** split-K or stream-K
  chosen by a shape rule; grouped GEMMs across independent layers. Only if
  2-T20a's occupancy columns show underfilled grids.
- **OPT-T04 Exact skips:** a checkpoint scan for zero weight blocks and
  unused paths or heads; species-only work hoisted out of the step. Expected
  0–5 %.

- **OPT-T05 Regional persistence experiment (optional, last).** Only after
  OPT-T02, and only if 2-T20a's Nsight Systems trace shows removable
  transitions between kernels.
  - Method: take OPT-T02's fused atom block and run it as a persistent region
    (tiled tasks, dependency counters, reusable workspace), so that
    persistence's own gain is isolated.
  - Stopping rules (the report's milestones C–D): it must beat OPT-T02's
    block, and needs ≥ 10 % whole-step gain on both GPUs before it grows
    beyond one interaction.

**Added 20 October 2026, from an outside review** (owner: "assess and
implement how you like"). These two target setup and the subprocess
lifecycle, not the steady-state step. Each starts with a short design note
and runs in parallel with B/C, with its own worker.

- **OPT-T06 Converted-model template cache (2-T19 worker).**
  - **Why:** every plan reloads, hashes and converts the checkpoint, 3.1–3.5 s
    for each plan after the first (2-T19 report). Fallback re-plans pay it
    too.
  - **What:** a bounded per-process cache of the pristine converted model,
    keyed by checkpoint hash, backend and dtype, with a private deep copy for
    every plan. All self-checks stay, since checkpoint 2 kept the self-check
    cache out.
  - **Checks:** benchmark clone against reload first. Then A → B → A isolation
    (§8.7), precision and backend changes, fallback attempts, memory
    retention.
  - **Small fix alongside:** call `installed_mace_version()` once per process,
    not once per record.
- **OPT-T07 Persistent subprocess worker (2-T07 worker).**
  - **Why:** every subprocess `run()` starts a fresh child and recompiles,
    68–90 s per `run(100)` (2-T07 report), and ASE observers split runs at
    each callback.
  - **What:** keep one child and its compiled engine for the simulation's
    lifetime. Checkpoint boundaries, host writes, failure recovery and the
    restart semantics stay unchanged.
  - **Checks:** repeated short calls, observer mutations, reproducible
    restart equivalence, child termination, GPU-memory release.

**Overnight queue, 20–21 October 2026** (owner asleep from about 21:30 UTC,
asked for about 10 hours of queued work):

| Who | Until about 03:00 UTC (code freeze for the acceptance) |
| --- | --- |
| 2-T20 | Push the A+B default change (conductor trials and merges it). Then: the stage B report; the local OPT-T03 sm_120 and OPT-T01 reruns; stage C (design note, then build, gate and A/B on both GPUs if it's a go); then an OPT-T04 checkpoint scan, CPU only, report only |
| 2-T19 | OPT-T06: build, tests, murphy bitwise check |
| 2-T07 | OPT-T07: build, tests, murphy checks |
| 2-T23 | Write the combined-acceptance runbook (phase 2 items plus the round's), CPU only |
| 2-T22 | Coraline soft-lock check, only if coraline's GPU has no process of any user |

**Code freeze at about 03:00 UTC:** whatever is merged by then is in the
combined acceptance. Anything later waits for the next round.

**Freeze recorded 21 October 2026, 03:05 UTC:** the code under test is main
at `4f90a92`. This plans-only commit adds the freeze record and changes no
code.
- In the freeze, all merged and trialled: stage A (`gated_rows`) and stage B
  (`segment_gemm`) as defaults, OPT-T06 (template cache), OPT-T07
  (persistent worker), and the records of stage C, OPT-T01/T01b, OPT-T03
  and OPT-T04.
- Not in the freeze, all opt-in and waiting for the next round's trial merge:
  - `fastmace/opt-T03b` (`a741349`): grouped_launch fixed. It gains on sm_120
    only (NaCl b/a 0.977 / 0.982), and on sm_89 it gives 1.005 / 1.003, so
    it is not a default candidate.
  - `fastmace/opt-T08` (`eecda68`): C1 `species_tables`, C2
    `fold_readout_tail`, C3 `fold_product_linear`; CPU-tested. Its GPU runs
    did not start before the freeze, because the worker's permission check
    refused the launch. They follow the acceptance.

**Combined acceptance from about 03:00 UTC (2-T23):** both GPUs, the
runbook's order, including the 19-label gate, the item 4 repeat and item 8
against `fastmace-1b`. The phase 2 report and a round-1 summary are ready
for the owner in the morning.

- **OPT-T08 Atom-level linear audit (2-T16 worker; owner, 21 October
  2026, night: "run the audit tonight and try out everything you can think
  of").**
  - **Tonight, CPU only:** an audit of the compiled force pass for
    multiplies that share an input (merge into one wider GEMM), adjacent
    linears without a nonlinearity between them (fold W2·W1 at load), and
    other exact reductions.
  - **Then:** build every candidate above about 1 % as an opt-in record, and
    A/B it on both GPUs after the acceptance.
- **OPT-T03 candidate 2, diagnosis (2-T20; owner):** why grouped_launch's
  1.3× bench gain becomes a full-step loss. Local nsys/ncu until 02:45 UTC,
  then a fix attempt.

Not in this round:
- a complete force megakernel. Our fit gives a ceiling of 8–16 % on NaCl, and
  the report budgets 0–15 % with regression possible. Months of work.
- batching and larger cells (owner decisions: one crystal at a time,
  3×3×3).
