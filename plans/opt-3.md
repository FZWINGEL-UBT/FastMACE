# Optimisation round 3 (remaining kernel ideas)

The owner decided this round on 21 October 2026, after OPT-T10
(`docs/design/opt-T10-step-breakdown.md`). Their words: "Go ahead and try all
kernel opportunities".

**Baseline:** main `62aa6d4`, with the round-2 defaults on.

| Structure | Step, local RTX 5060 Ti, un-profiled |
| --- | --- |
| NaCl | 3.028 ms |
| NbCl5 | 21.434 ms |
| NaNbO3 | 18.788 ms |

## Rules

The rules are as in round 2 (`plans/opt-2.md`). In addition:
- **GPUs.** The local GPU uses reservations of at most 15 min, and the memory rules apply (`free -g` first; workers run targeted tests only). Modal L4 (sm_89) and A100 (sm_80) A/Bs go through `tools/modal/fastmace_modal.py`, at a few dollars per task. Murphy only after the outage.
- **Defaults.** A candidate becomes a default candidate only with a gain on sm_120 and no loss on the L4. The owner decides.

## Tasks

| Task | Idea | Estimate | Worker |
| --- | --- | --- | --- |
| OPT3-T01 | Fused gate + residual VJP (the elementwise backward; "not built" since October). Elementwise work is 13 % of NbCl5 | 3–6 % on large cells, rounding | 2-T19 |
| OPT3-T02 | Symmetric contraction (cuEq): profile, roofline, then screen exact/rounding improvements or a better kernel choice | unknown; measure first | 2-T20 |
| OPT3-T03 | Integrator in one kernel (A, C, P with in-kernel Philox and the flags); bitwise if the RNG stream is identical | ~3 % NaCl, exact if possible | 2-T21 |
| OPT3-T04 | NaCl graph gaps: merge the smallest adjacent kernels and cut the launch count inside B (174 µs of gaps) | 3–5 % NaCl, exact | 2-T15 |
| OPT3-T05 | Layout audit: remaining transposes, copies and contiguity conversions in the step trace; remove the exact ones | 1–2 %, exact | 2-T07 |
