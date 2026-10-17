# FastMACE phase plans

One plan per phase of [SPEC §16](../SPEC.md). Each plan is written when the
phase before it is under way, not all in advance. Worker rules are in
[../AGENTS.md](../AGENTS.md).

| Phase | Plan | Status |
| --- | --- | --- |
| 0. Skeleton | [phase-0.md](phase-0.md) | done (17 October 2026); report `docs/validation/phase-0.md` |
| 1a. Extraction | [phase-1a.md](phase-1a.md) | 14 of 20 merged (T00–T10, T14, T16, T18; 18 October 2026); T07/T08 GPU confirmation pending; next T11 and T12 |
| 1b. Contracts | — | written during 1a |
| 2. Safety | — | |
| 3. Dynamic topology and MC | — | |
| 3b. NPT for the chlorides | — | |
| 4. Physics | — | |
| 5. Planner completion | — | |
| 6. Release | — | |

Open owner decisions that phases wait on (SPEC §18; licence decided: MIT):
minimum MACE version (2, by phase 2), gate default (3, phase 1b), CI GPU
runners (5, phase 2).

Task IDs are `<phase>-T<nn>`. Worker branches are `fastmace/<phase>-T<nn>`.

Conductor sessions start with [CONDUCTOR.md](CONDUCTOR.md).
