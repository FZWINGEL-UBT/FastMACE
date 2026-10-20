# FastMACE phase plans

One plan per phase of [SPEC §16](../SPEC.md). Each plan is written when the
phase before it is under way, not all in advance. Worker rules are in
[../AGENTS.md](../AGENTS.md).

| Phase | Plan | Status |
| --- | --- | --- |
| 0. Skeleton | [phase-0.md](phase-0.md) | done (17 October 2026); report `docs/validation/phase-0.md` |
| 1a. Extraction | [phase-1a.md](phase-1a.md) | **all 20 merged; acceptance passed** (18 October 2026, `docs/validation/phase-1a.md`); owner checkpoint 3 decided (19 October: `main.py` defaults to FastMACE) |
| 1b. Contracts | [phase-1b.md](phase-1b.md) | **done** (19–20 October 2026), tag `fastmace-1b`; acceptance report `docs/validation/phase-1b.md` |
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

- Phase 1b: done, tagged `fastmace-1b` (673a8a1, with 1b-T27).
- Merged in phase 2: 2-T11, 2-T01, 2-T14, 2-T09, 2-T12, 2-T10, 2-T18, 2-T04,
  2-T09b, 2-T07, 2-T14c, 2-T02, 2-T06, 2-T08, 2-T13, 2-T05, 2-T14d, 2-T03,
  2-T08b, 2-T19 (report; no cache: owner, checkpoint 2), 2-T04b, 2-T15
  (the fallback; acceptance 2), 2-T04c, 2-T16, 2-T15b, 2-T17, 2-T19b, 2-T04d
  (20 October 2026). 2-T21 is merged in the ResearchModule.
- Working in phase 2:
  - 2-T23 (phase 2 acceptance; local tier running);
  - 2-T14e (float64 reuse: duplicate removed; owner decisions of 20 October:
    ASE-equivalence reference 4,000 steps, limits 1e-6 K and 1e-7 meV/atom;
    no reference cache for now);
  - 2-T07b, 2-T15c, 2-T04e (fixes from an outside review, 20 October:
    durable subprocess checkpoint save; failed engine released before the
    fallback's next candidate; hosted CPU CI red on torch 2.13).
- Next: 2-T20 and 2-T22 as resources allow.
- Waiting for go: the rest. Answers to all prestart reports are in
  `phase-2.md`, "Conductor answers to the prestart reports".

## Open owner decisions

Open owner decisions that phases wait on (SPEC §18): none open. Decided on
19 October 2026: minimum MACE version (2), gate default (3), CI GPU runners
(5), and 22–30.

Owner action pending: register the two GPU CI runners, murphy and the local
PC (`docs/development.md`, "Registering the runners"; 2-T18 is merged). 2-T18's
done-when, both runners green from a push, waits for it.

2-T14's scratch folders were deleted on the owner's word (20 October 2026).

Task IDs are `<phase>-T<nn>`. Worker branches are `fastmace/<phase>-T<nn>`.

Conductor sessions start with [CONDUCTOR.md](CONDUCTOR.md).
