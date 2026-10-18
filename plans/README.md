# FastMACE phase plans

One plan per phase of [SPEC §16](../SPEC.md). Each plan is written when the
phase before it is under way, not all in advance. Worker rules are in
[../AGENTS.md](../AGENTS.md).

| Phase | Plan | Status |
| --- | --- | --- |
| 0. Skeleton | [phase-0.md](phase-0.md) | done (17 October 2026); report `docs/validation/phase-0.md` |
| 1a. Extraction | [phase-1a.md](phase-1a.md) | **all 20 merged; acceptance passed** (18 October 2026, `docs/validation/phase-1a.md`); owner checkpoint 3 open |
| 1b. Contracts | [phase-1b.md](phase-1b.md) | planned (draft) |
| 2. Safety | — | |
| 3. Dynamic topology and MC | — | |
| 3b. NPT for the chlorides | — | |
| 4. Physics | — | |
| 5. Planner completion | — | |
| 6. Release | — | |

## Ready to start

Tasks whose dependencies are merged and whose brief is ready. The owner
starts all of them with one message, "start all ready" (see
[CONDUCTOR.md](CONDUCTOR.md)).

- none. Phase 1a is complete except for the owner's checkpoint 3; the phase 1b plan is written next.

## Open owner decisions

Open owner decisions that phases wait on (SPEC §18; licence decided: MIT):
minimum MACE version (2, by phase 2), gate default (3, phase 1b), CI GPU
runners (5, phase 2).

Task IDs are `<phase>-T<nn>`. Worker branches are `fastmace/<phase>-T<nn>`.

Conductor sessions start with [CONDUCTOR.md](CONDUCTOR.md).
