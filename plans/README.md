# FastMACE phase plans

One plan per phase of [SPEC §16](../SPEC.md). Each plan is written when the
phase before it is under way, not all in advance. Worker rules are in
[../AGENTS.md](../AGENTS.md).

| Phase | Plan | Status |
| --- | --- | --- |
| 0. Skeleton | [phase-0.md](phase-0.md) | done (17 October 2026); report `docs/validation/phase-0.md` |
| 1a. Extraction | [phase-1a.md](phase-1a.md) | **all 20 merged; acceptance passed** (18 October 2026, `docs/validation/phase-1a.md`); owner checkpoint 3 decided (19 October: `main.py` defaults to FastMACE) |
| 1b. Contracts | [phase-1b.md](phase-1b.md) | **all tasks merged; acceptance report** `docs/validation/phase-1b.md` (19 October 2026); follow-ups and addendum merged; **done apart from the owner checkpoint** |
| 2. Safety | [phase-2.md](phase-2.md) | planned (19 October 2026); D1–D4 decided; all workers prestarted |
| 3. Dynamic topology and MC | — | |
| 3b. NPT for the chlorides | — | |
| 4. Physics | — | |
| 5. Planner completion | — | |
| 6. Release | — | |

## Ready to start

All phase 2 tasks (2-T01 to 2-T23) were prestarted on 19 October 2026 and
wait for the conductor's go message (`CONDUCTOR.md`, "Prestarted phase
workers"). 2-T21 works in the ResearchModule worktree
`D:/Margraf/ResearchModule-wt/2-T21` (branch `fastmace/2-T21`).

- Phase 1b: done, apart from 1b-T27 (the GPU acceptance tier), which is
  finishing its acceptance-tier run on murphy. 2-T01's go waits for it.
- Merged in phase 2: 2-T11, 2-T01 (20 October 2026).
- Working in phase 2: 2-T12 (local noise floor; local machine exclusive),
  2-T09, 2-T04, 2-T05 (tests on murphy's CPUs meanwhile), 2-T14 (the
  cuEq basis-cache fix, approved).
- 1b-T27 merges after 2-T14's fix.
- Waiting for go: the rest. Answers to all prestart reports are in
  `phase-2.md`, "Conductor answers to the prestart reports".

## Open owner decisions

Open owner decisions that phases wait on (SPEC §18): none open. Decided on
19 October 2026: minimum MACE version (2), gate default (3), CI GPU runners
(5), and 22–30.

Task IDs are `<phase>-T<nn>`. Worker branches are `fastmace/<phase>-T<nn>`.

Conductor sessions start with [CONDUCTOR.md](CONDUCTOR.md).
