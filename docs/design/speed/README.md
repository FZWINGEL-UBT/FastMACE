# Speed reports

Design notes and analyses of optimisation rounds 1–3 (20–29 October 2026); their measured results are in [validation/speed](../../validation/speed/README.md).

Columns: file, date added, title, and the report's own verdict line or first sentence.

| File | Added | Title | Verdict / first sentence |
| --- | --- | --- | --- |
| [opt-T02-BC-go-no-go](opt-T02-BC-go-no-go.md) | 2026-10-20 | OPT-T02 stages B and C: go/no-go after stage A | Input: the 2-T20a main pass on murphy (RTX 4000 Ada, sm_89) with `gated_rows` on (`results/opt-T02-trace/murphy/`, 20 October 2026 19:27–19:32 UTC, GPU empty, load … |
| [opt-T02-C-gate-prologue](opt-T02-C-gate-prologue.md) | 2026-10-20 | OPT-T02 stage C: the gate inside `linear_2`'s segment GEMM | Decision (conductor, per §4): rejected. The rule needs both GPUs, and sm_120 alone fails it, so |
| [opt-T02-fusion](opt-T02-fusion.md) | 2026-10-20 | OPT-T02 design: fusion of the atom-level elementwise work and removal of layout copies | Status: reviewed (conductor, 20 October 2026). Stage A is approved as a new registry record that applies |
| [opt-T03-small-m](opt-T03-small-m.md) | 2026-10-20 | OPT-T03 design: small-M GEMMs (fuller grids for small cells) | Status: design for review (20 October 2026), no code. Plan: `plans/opt-1.md`, OPT-T03. Data: 2-T20a |
| [opt-T03b-diagnosis](opt-T03b-diagnosis.md) | 2026-10-21 | OPT-T03 candidate 2 (`grouped_launch`): why the bench gain became a full-step loss, and the fix | Result: two causes, both in my first version, not in the kernel. Fixed, the grouped launch makes the MH-1 |
| [opt-T06-template-cache](opt-T06-template-cache.md) | 2026-10-20 | OPT-T06: converted-model template cache (design note) | Status: design for review, 21 October 2026. Plan `plans/opt-1.md`, OPT-T06. Nothing is implemented yet. |
| [opt-T07-persistent-worker](opt-T07-persistent-worker.md) | 2026-10-20 | OPT-T07: a persistent subprocess worker | Plan task OPT-T07 (`plans/opt-1.md`), a follow-up to 2-T07 (SPEC §8.7, §10.6). |
| [opt-T08-linear-audit](opt-T08-linear-audit.md) | 2026-10-21 | OPT-T08: audit of the atom-level linear layers | Optimisation round 1, task OPT-T08 (owner request, 20–21 October 2026). |
| [opt-T09-step-breakdown](opt-T09-step-breakdown.md) | 2026-10-21 | OPT-T09: where the step goes after optimisation round 1, and what to optimise next | Owner request, 21 October 2026. |
| [opt-T10-step-breakdown](opt-T10-step-breakdown.md) | 2026-10-21 | OPT-T10: where the step goes on the new default, and what to do next | Owner request, 21 October 2026: "After you merged do another detailed kernel timing of the current best configuration". |
| [opt2-T01-fused-radial-conv](opt2-T01-fused-radial-conv.md) | 2026-10-21 | OPT2-T01: the radial last layer fused into the streaming convolution | Plan: `plans/opt-2.md`, OPT2-T01 (owner, 21 October 2026: "Go ahead and try all 5 ideas as much as possible"). |
| [opt2-T02-radial-mlp-fusion](opt2-T02-radial-mlp-fusion.md) | 2026-10-21 | OPT2-T02: radial MLP hidden layers in one kernel per direction (design note) | Status: design, 21 October 2026. Plan `plans/opt-2.md`, OPT2-T02 (OPT-T09 rank 2, ceiling 4–8 %, rounding class). |
| [opt2-T05-split-k](opt2-T05-split-k.md) | 2026-10-21 | OPT2-T05: split-K for small-cell atom products | Plan task OPT2-T05 (`plans/opt-2.md`; candidate 5 of `docs/design/speed/opt-T09-step-breakdown.md`). |
| [opt3-T01-gate-vjp-analysis](opt3-T01-gate-vjp-analysis.md) | 2026-10-21 | OPT3-T01: fused gate and residual VJP, what the trace leaves to win (analysis before building) | Status: analysis, 21 October 2026. Plan `plans/opt-3.md`, OPT3-T01 (estimate 3–6 % on large cells, rounding). |
| [opt3-T02-symmetric-contraction](opt3-T02-symmetric-contraction.md) | 2026-10-21 | OPT3-T02: the cuEquivariance symmetric contraction, measured and screened | Result: not built. The symmetric contraction is 2.5 % (NaCl), 5.4 % (NbCl5) and 3.1 % (NaNbO3) of the step. |
| [opt3-T05-layout-audit](opt3-T05-layout-audit.md) | 2026-10-21 | OPT3-T05: layout audit of the default step | Plan task OPT3-T05 (`plans/opt-3.md`). |
