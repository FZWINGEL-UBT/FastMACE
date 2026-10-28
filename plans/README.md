# FastMACE phase plans

One plan per phase of [SPEC §16](../SPEC.md). Each plan is written when the
phase before it is under way, not all in advance. Worker rules are in
[../AGENTS.md](../AGENTS.md).

| Phase | Plan | Status |
| --- | --- | --- |
| 0. Skeleton | [phase-0.md](phase-0.md) | done (17 October 2026); report `docs/validation/phase-0.md` |
| 1a. Extraction | [phase-1a.md](phase-1a.md) | **all 20 merged; acceptance passed** (18 October 2026, `docs/validation/phase-1a.md`); owner checkpoint 3 decided (19 October: `main.py` defaults to FastMACE) |
| 1b. Contracts | [phase-1b.md](phase-1b.md) | **done** (19–20 October 2026), tag `fastmace-1b`; acceptance report `docs/validation/phase-1b.md` |
| 2. Safety | [phase-2.md](phase-2.md) | **acceptance complete** (21 October 2026, report `docs/validation/phase-2.md`, merged `89ea581`): every item passes except item 7 (deferred by the owner). Results come from the freeze `be92979`, and from `f993713` for what its two defects touched (item 4, the Simulation-driven GPU tests, OMOL-0 XL). The 19-label gate passes 19 of 19. tagged `fastmace-2` (e52c482, owner, 21 October 2026) |
| Optimisation round 1 | [opt-1.md](opt-1.md) | done except OPT-T08's GPU A/B. Defaults since 20 October: `gated_rows`, `segment_gemm`, template cache, persistent worker. Measured end to end against `fastmace-1b`: NaCl −3.6 to −4.1 %, NbCl5 −13.5 to −14.9 % on both GPUs |
| Optimisation round 2 | [opt-2.md](opt-2.md) | started 21 October 2026 (owner: "try all 5 ideas"); OPT2-T01 to T05 with 2-T20, 2-T19, 2-T15, 2-T21 and 2-T07 |
| Optimisation round 3 | [opt-3.md](opt-3.md) | started 21 October 2026 (owner: "try all kernel opportunities"); OPT3-T01 to T05 |
| 3. Dynamic topology and MC | [phase-3.md](phase-3.md) | **complete**, tag `fastmace-3`; [acceptance report](../docs/validation/phase-3.md) records the remaining clauses and deferred workstation check |
| 3b. NPT for the chlorides | [phase-3b.md](phase-3b.md) | **complete**, tag `fastmace-3b`; [report](../docs/validation/phase-3b.md) records the cost target and timing limitations |
| 4. Physics | [phase-4.md](phase-4.md) | **complete**, tag `fastmace-4`; [acceptance report](../docs/validation/phase-4.md) |
| 5. Planner completion | [phase-5.md](phase-5.md) | **complete**, tag `fastmace-5`; [acceptance report](../docs/validation/phase-5.md) |
| 6. Release | [phase-6.md](phase-6.md) | **released `v1.0.0`**, PyPI packages and docs live; Python 3.13 public-index hybrid install verified; [release evidence](../docs/validation/release-1.0.0.md) |
| Optimisation round 5 | [opt-5.md](opt-5.md) | **complete** (26 October 2026); `uses_tf32x3_tile` hybrid SIMT/TC shape dispatch adopted inside `tf32x3_gemms` (`6.90× → 1.52–1.73×` max error ratio vs IEEE, `−31 to −39 %` gate energy offset); `sched_iso` and `x3_pipe` probes recorded; [report](../docs/validation/opt-5.md) |
| Speed-up ideas (27–28 October 2026) | [ideas-20261027.md](ideas-20261027.md) | D1–D3 built and confirmed per GPU (RTX 5060 Ti, RTX 4000 Ada, A100); **decision 65**: all three on by default in float64 MH-1 plans on the two measured GPU models only (exact names), off on the A100 and unmeasured GPUs; branch `fastmace/ideas-20261027` |
| Seven float32 ideas, cheap checks (28 October 2026) | [fp32-ideas-20261028.md](fp32-ideas-20261028.md) | L2-sized atom blocks survive the microbenchmark (0.87x on a four-layer block at 2 chunks, upper bound); recompute, second stream and Strassen–Winograd falsified; BF16x9 emulation inactive on sm_80/sm_120; linear folds already built; contraction fusion dropped on paper; branch `fastmace/fp32-ideas-20261028` |
| Optimisation round 6 | [opt-6.md](opt-6.md) | started 26 October 2026 (owner). OPT6-T02 `chunked_conv_backward` **default on** (decision 60): NbCl5 0.97, NaNbO3 0.97, NaCl 1.01 ([report](../docs/validation/opt6-T02.md)). OPT6-T05 `radial_tables` (new class `certified`, decision 61), opt-in: certificates 0.15–0.30, gate passed, b/a NaCl 0.90, NbCl5 0.92, NaNbO3 0.81 ([report](../docs/validation/opt6-T05.md)); its default awaits the owner |

## Workstation availability (owner, 20 October 2026)

- **Until Thursday 22 October 2026, evening:** jobs may run on the workstations
  (murphy, leto, leeloo, coraline) under the usual rules.
- **After that:** no workstation jobs until a scheduled power outage on Friday
  23 October is over and the owner says they're back. Plan GPU work for the
  local PC in that window.
- **Running jobs:** stop or finish them on Thursday evening, and keep their
  results and restart files.
- **Coraline:** busy with another user's run until about Friday. The coraline
  check 2-T22 is deferred by the owner and carried over.

## Ready to start

All phase 2 tasks (2-T01 to 2-T23) were prestarted on 19 October 2026 and
wait for the conductor's go message (`CONDUCTOR.md`, "Prestarted phase
workers"). 2-T21 works in the ResearchModule worktree
`D:/Margraf/ResearchModule-wt/2-T21` (branch `fastmace/2-T21`).

- Phase 1b: done, tagged `fastmace-1b` (673a8a1, with 1b-T27).
- Merged in phase 2: 2-T11, 2-T01, 2-T14, 2-T09, 2-T12, 2-T10, 2-T18, 2-T04,
  2-T09b, 2-T07, 2-T14c, 2-T02, 2-T06, 2-T08, 2-T13, 2-T05, 2-T14d, 2-T03,
  2-T08b, 2-T19 (report; no cache: owner, checkpoint 2), 2-T04b, 2-T15
  (the fallback; acceptance 2), 2-T04c, 2-T16, 2-T15b, 2-T17, 2-T19b, 2-T04d
  (20 October 2026). 2-T21 is merged in the ResearchModule.
- Combined acceptance, 21 October 2026 (2-T23; freeze `be92979`):
  - **Passed:**
    - item 8 on both machines;
    - the murphy tier, 33 of 33;
    - the local tier, 31 of 33. Its 2 failures were test-side and are fixed in 2-T15d and opt-T07b, both merged.
  - **Gate:** 18 of 19 labels pass. OMOL-0 XL hit a Dynamo `NameError` from per-build strain rebinding. It is fixed in 2-T19b (merged in `314210d`), and the GPU fix check passed on murphy.
  - **Item 4, resolved:** hybrid `fm.Simulation` runs heated and reached NaN because `GraphEngine.prepare()` (2-T15b, `d14be8c`) captured B/ACB with the build's edge views. Edges added later inside the padded bucket were dropped.
    - Fixed by 2-T21d (merged in `5d1848b`), confirmed on both GPUs.
    - Item 4 then passed on `f993713`.
    - ResearchModule's `AGENTS.md` records `d14be8c` up to `5d1848b` as the invalid range for hybrid `main.py --engine fastmace`.
  - **Also merged:** 2-T07c, the heartbeat rename race under load.
  - **Follow-up, not urgent:** a fallback's reason truncates chained compile errors and should carry the innermost message (2-T21d note).
- Working in phase 2:
  - 2-T14e (float64 reuse: duplicate removed; owner decisions of 20 October:
    ASE-equivalence reference 4,000 steps, limits 1e-6 K and 1e-7 meV/atom;
    no reference cache for now);
  - 2-T07b, 2-T15c, 2-T04e (fixes from an outside review, 20 October:
    durable subprocess checkpoint save; failed engine released before the
    fallback's next candidate; hosted CPU CI red on torch 2.13): all merged
    (a8cd333, fe1dac0, 05ecbe6). 2-T14e is merged too (edba9f3);
  - 2-T08c (why `test_mh1_plain_cueq_float64_against_the_research_engine`
    fails on murphy: 12 extra kernels in the stage-B graph, recorded by
    2-T08b; it passes locally).
- Next: 2-T20 and 2-T22 as resources allow.
- Waiting for go: the rest. Answers to all prestart reports are in
  `phase-2.md`, "Conductor answers to the prestart reports".

## Open owner decisions

Open owner decisions that phases wait on (SPEC §18): none open. Decided on
19 October 2026: minimum MACE version (2), gate default (3), CI GPU runners
(5), and 22–30.

The hosted CPU CI (`cpu.yml`) is off for now (owner, 20 October 2026, to
save Actions minutes): it runs by hand only. Trial merges run ruff
(`ruff==0.17.1`, in the NAS `mace_cueq_latest`) and the CPU suite on murphy.
GPU CI is deferred (owner, 20 October 2026; SPEC draft 0.12, eleventh
amendment). It is not part of phase 2's acceptance. `gpu.yml` runs by hand
only until the runners are registered (`docs/development.md`, "Registering
the runners").

2-T14's scratch folders were deleted on the owner's word (20 October 2026).

Task IDs are `<phase>-T<nn>`. Worker branches are `fastmace/<phase>-T<nn>`.

Conductor sessions start with [CONDUCTOR.md](CONDUCTOR.md).
