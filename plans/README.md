# FastMACE phase plans

One plan per phase of [SPEC §16](../SPEC.md). Each plan is written when the
phase before it is under way, not all in advance. Worker rules are in
[../AGENTS.md](../AGENTS.md).

| Phase | Plan | Status |
| --- | --- | --- |
| 0. Skeleton | [phase-0.md](phase-0.md) | done (17 October 2026); report `docs/validation/phase-0.md` |
| 1a. Extraction | [phase-1a.md](phase-1a.md) | **all 20 merged; acceptance passed** (18 October 2026, `docs/validation/phase-1a.md`); owner checkpoint 3 decided (19 October: `main.py` defaults to FastMACE) |
| 1b. Contracts | [phase-1b.md](phase-1b.md) | **all tasks merged; acceptance report** `docs/validation/phase-1b.md` (19 October 2026); follow-ups and addendum merged; **done apart from the owner checkpoint** |
| 2. Safety | [phase-2.md](phase-2.md) | planned (draft, 19 October 2026); owner decisions D1–D4 open |
| 3. Dynamic topology and MC | — | |
| 3b. NPT for the chlorides | — | |
| 4. Physics | — | |
| 5. Planner completion | — | |
| 6. Release | — | |

## Ready to start

Tasks whose dependencies are merged and whose brief is ready. The owner
starts all of them with one message, "start all ready" (see
[CONDUCTOR.md](CONDUCTOR.md)).

All remaining 1b tasks were prestarted on 19 October 2026 (owner request)
and wait for the conductor's go message (`CONDUCTOR.md`, "Prestarted phase
workers"). No "start all ready" is needed in phase 1b.

- Merged: 1b-T01, T02, T03, T05, T04, T06, T06b, T07, T08, T08b, T08c, T09, T10, T11, T12, T13, T14, T15, T16 to T22, T23 tools, T24 to T26 (19 October 2026).
- Merged since: 1b-T28 (software block once per process).
- Working: 1b-T27 (GPU acceptance tier). Owner checkpoint decided (SPEC §18 decisions 22–28).
- Waiting for go: none. Answers to their prestart
  reports, and three items for the owner's review, are in `phase-1b.md`,
  "Conductor answers to the prestart reports".

## Open owner decisions

Open owner decisions that phases wait on (SPEC §18; licence decided: MIT):
minimum MACE version (2, by phase 2), gate default (3, phase 1b), CI GPU
runners (5, phase 2).

Task IDs are `<phase>-T<nn>`. Worker branches are `fastmace/<phase>-T<nn>`.

Conductor sessions start with [CONDUCTOR.md](CONDUCTOR.md).
