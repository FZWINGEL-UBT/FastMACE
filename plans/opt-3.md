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

## Round-3 results and the defaults (21 October 2026)

| Task | Result | Decision |
| --- | --- | --- |
| OPT3-T01 fused gate + residual VJP | the gate's kernels are already fused at DRAM bandwidth; a fused backward could remove 0.6–1.1 % of NbCl5 at best (`docs/design/opt3-T01-gate-vjp-analysis.md`) | not built |
| OPT3-T02 symmetric contraction | 2.5–5.4 % of the step, kernels at 72–96 % of memory speed-of-light; every option's ceiling below 3 % (`docs/design/opt3-T02-symmetric-contraction.md`) | screened, not built |
| OPT3-T03 `fused_integrator` | exact (stage level byte-identical, reproducible trajectories bitwise); NaCl −2.4 % (sm_120) and −1.9 to −2.0 % (L4), NbCl5 −0.3 to −0.7 % / neutral (`docs/validation/opt3-T03.md`) | **default on** (owner, "Yes"; SPEC fourteenth amendment), also in reproducible plans |
| OPT3-T04 `combo_kernels` | rounding; NaCl −0.9 to −1.8 %, NbCl5 +1.9 to +2.7 % on both GPUs (`docs/validation/opt3-T04.md`) | rejected, opt-in record kept |
| OPT3-T05 layout records `direct_node_feats`, `folded_constants` | exact; NaCl −0.5 to −0.8 % (sm_120), neutral on the L4; `lm_major_outer_sum` reverted (`docs/validation/opt3-T05-layout-audit.md`) | opt-in, no gain above noise |
