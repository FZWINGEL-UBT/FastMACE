# Optimisation round 2 (edge side and small-cell fixed costs)

The owner decided this round on 21 October 2026, after the OPT-T09 breakdown
(`docs/design/opt-T09-step-breakdown.md`). Their words: "Go ahead and try all
5 ideas as much as possible".

**Baseline:** main `eb70cfd`, with the OPT-T08 records on (`species_tables`,
`fold_readout_tail`, `fold_product_linear`). These are the presumed defaults;
their murphy A/B is still pending.

| Structure | Step, un-profiled, local RTX 5060 Ti |
| --- | --- |
| NaCl | 3.196 ms |
| NbCl5 | 21.461 ms |
| NaNbO3 | 18.810 ms |

## Rules (as in round 1)

- **General only.** No per-GPU or per-shape tables; a shape rule from M, N, K
  and the SM count is allowed. One crystal at a time, 3×3×3, TF32 off.
- **Opt-in.** Every candidate is an opt-in `Rewrite` record (`never_automatic`)
  with an applicability check and refusal tests. The owner decides defaults.
- **Accuracy.**
  - Exact candidates: bitwise on CPU and in the reproducible mode.
  - Rounding-class candidates: the float32 force gate (≤ 2× the float32
    calculator, or the gate's floor) on NaCl and NbCl5, plus CPU float64
    agreement.
- **Timing.**
  - Same-process A/B, both build orders, 12 blocks, from a pinned source copy.
  - Local noise floor: 1.1 % NaCl, 0.4 % NbCl5. Murphy's: 0.16 %.
  - Keep the PC quiet. WSL can't see Windows programs on the GPU.
- **GPU sharing.**
  - The local GPU goes through `tools/ci/local_gpu.py` reservations. Short
    jobs; release between jobs.
  - Murphy only when no user has a compute process on it, and not after
    Thursday 22 October 16:00 UTC (power outage on 23 October).
  - OPT-T08's murphy watcher has priority there.
- **CPU.** CPU suites at `-n 2` per worker (five workers share the PC).

## Tasks

| Task | Idea (OPT-T09 rank) | Ceiling | Worker |
| --- | --- | --- | --- |
| OPT2-T01 | Fused radial last layer + streaming convolution, forward and backward, so the [E, 1280] per-edge weights are never stored. Start with a design note that reuses `studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md` (ceiling 16 % NaNbO3 and 9 % NbCl5 then; the Triton attempt fell short; "needs OEq-level kernel engineering"). Then build if the design shows a path. | ~12 % NaCl/NbCl5, ~20 % NaNbO3 | 2-T20 |
| OPT2-T02 | Radial MLP fusion, forward and backward (hidden layers, activation, and their backward in fewer kernels) | 4–8 % | 2-T19 |
| OPT2-T03 | Folded last radial layer's Triton input gradient (25 % occupancy): tile by a shape rule | 2–4 % | 2-T15 |
| OPT2-T04 | NaCl fixed costs: one graph per step (ACP plus force), fewer launches, and the neighbour rebuild and host gaps. Exact. murphy check needed, since launch costs differ between WSL and Linux | 4–6 % NaCl | 2-T21 |
| OPT2-T05 | Small-cell l=0 split-K (or a stream-K rule from M, N, K and the SM count) | ~2 % NaCl | 2-T07 |

**Overlaps:**
- T01 and T03 touch the same kernel (the folded last layer's input gradient):
  if T01 lands, T03's kernel goes away.
- T01, T02 and T03 all touch the radial path. Each stays a separate record;
  combinations get their own A/B at the end.

**Order of work:**
1. Design note.
2. CPU build and tests.
3. Force gate.
4. Local A/B.
5. Murphy A/B when free.
6. A round report (`docs/validation/opt-2.md`).
