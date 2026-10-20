# Optimisation round 1 (between phases 2 and 3)

Owner decision, 20 October 2026: before phase 3 starts, one round of speed
optimisations, beginning with the tensor cores. Phase 3 starts after this
round, not in parallel with it. Phase 2's acceptance (2-T23) finishes first:
it holds both GPUs for timed work today.

Background and estimates: the owner's doc "FastMACE: speeding up small-cell
MD steps" (20 October 2026), built from the research engine's step breakdown
of 14–15 October (ResearchModule
`Experiments/Langevin_Thermostat/studies/further_optimizations/2026-10-15-step-breakdown.md`).

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
2. **Micro-benchmark (local, then murphy).** The model's real GEMM shapes
   (M = 216 to 2,000 rows; take K and N from MH-1): SIMT FP32 against each
   mode. Report speed and max/relative error against float64.
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
  residual and bias in the GEMM epilogue. Target: part of the 1.66 + 0.41 ms;
  5–10 % of the step.
- **OPT-T03 Small-M schedules and cross-layer grouping:** split-K or stream-K
  chosen by a shape rule; grouped GEMMs across independent layers. Only if
  2-T20a's occupancy columns show underfilled grids.
- **OPT-T04 Exact skips:** a checkpoint scan for zero weight blocks and
  unused paths or heads; species-only work hoisted out of the step. Expected
  0–5 %.

Not in this round: a persistent megakernel (ceiling 8–16 % on NaCl, months of
work); batching and larger cells (owner decisions: one crystal at a time,
3×3×3).
