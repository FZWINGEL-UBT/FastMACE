# FastMACE phase plans

One plan per phase of [SPEC §16](../SPEC.md). Each plan is written when the
phase before it is under way, not all in advance. Worker rules are in
[../AGENTS.md](../AGENTS.md).

| Phase | Plan | Status |
| --- | --- | --- |
| 0. Skeleton | [phase-0.md](phase-0.md) | done (17 October 2026); report `docs/validation/phase-0.md` |
| 1a. Extraction | [phase-1a.md](phase-1a.md) | **all 20 merged; acceptance passed** (18 October 2026, `docs/validation/phase-1a.md`); owner checkpoint 3 decided (19 October: `main.py` defaults to FastMACE) |
| 1b. Contracts | [phase-1b.md](phase-1b.md) | **done** (19–20 October 2026), tag `fastmace-1b`; acceptance report `docs/validation/phase-1b.md` |
| 2. Safety | [phase-2.md](phase-2.md) | **combined acceptance run 21 October 2026, not yet passed** (freeze `be92979`; report `docs/validation/phase-2.md`). Item 8 passes; the gate passes 18 of 19, with OMOL-0 XL fixed by 2-T19b. Item 4 can't be judged: hybrid `Simulation` runs heat up, and the cause is under investigation (2-T21d) |
| Optimisation round 1 | [opt-1.md](opt-1.md) | done except OPT-T08's GPU A/B. Defaults since 20 October: `gated_rows`, `segment_gemm`, template cache, persistent worker. Measured end to end against `fastmace-1b`: NaCl −3.6 to −4.1 %, NbCl5 −13.5 to −14.9 % on both GPUs |
| 3. Dynamic topology and MC | — | after optimisation round 1 |
| 3b. NPT for the chlorides | — | |
| 4. Physics | — | |
| 5. Planner completion | — | |
| 6. Release | — | |

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
  - **Open:** item 4. Hybrid `fm.Simulation` runs (in-process and runner) heat up after about 0.2 ps and reach NaN, while the direct planner-engine path stays smooth.
    - 2-T21d is bisecting on murphy.
    - Its first suspect, `prepare()` capturing B/ACB with the build's edge views, is a real fault. Its fix stays as hardening, but it did not reproduce the heating locally.
    - ResearchModule's `AGENTS.md` warns that hybrid `main.py --engine fastmace` (`--fastmace-api simulation`) runs on FastMACE ≥ `d14be8c` are suspect.
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
