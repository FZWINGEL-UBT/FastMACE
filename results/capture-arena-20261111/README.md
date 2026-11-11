# Capture arena, stage 1 on the A100 (11 November 2026)

**Status: shelved. `EngineConfig.capture_arena` stays off and is not merged.**

**Setup:**
- GPU: Alex A100-SXM4-80GB, torch 2.14.1+cu130.
- Model and system: OMAT-0 medium, float32, SrTiO3 36³ (233,280 atoms), default plan.
- Jobs:
  - 4981434, aliased recording without the canary;
  - 4981437, with the capture-only shim and the canary;
  - 4982478, source 3601fa5f: the pool freed with no `MemPool` reference left. Its records are in this directory.

## What works

- **Recording with aliased address ranges** (`record_mode: virtual`). Every range maps one shared granule, so Triton's capture-time pointer check passes.
  - No request came from a stream outside a capture (`uncaptured: 0`).
  - The canary was intact.
  - The recording pool was returned once the caller no longer held the `MemPool`. Before 3601fa5f, 4981434 and 4981437 fell back with "the recording pool kept segments".
- **Planned capture:** it matched the plan exactly (`plan_check: ok`, 21 of 21 requests served on the plan, 0 mismatches, 0 refusals).

## What does not

1. **No saving at the build.** At 36³ the plan is 11,706 MiB for 11,706 MiB of segments (`point-on`, `capture_arena`).
   - The build captures only the first step's keys (ACP and B). Within that one capture, every segment of the pool is busy at the same time.
   - Stage 0 found its 1.79 GiB bound over a build plus 20 MD steps, which includes later captures (`results/capture-arena-20261110`). The sealed arena retires at the first later capture, so it cannot reach that bound.
   - Steady peak reserved memory: 14.42 GiB with the arena off and on (298.95 ms/step for both).
2. **Overlapping segments break the caching allocator.** The MH-1 GPU test, whose plan does overlap segments, aborted with `inserted.second == inserted_by_addr.second INTERNAL ASSERT FAILED` (`CUDACachingAllocator.cpp:290`; `gputest-assert.txt`).
   - Since torch 2.14 each `BlockPool` keeps a second set keyed by (stream, address).
   - Two free blocks of different sizes in the same pool and stream, at one address, are refused by that set and trip the assert. Planned segments that share bytes at equal offsets produce exactly that.
   - A planned mode would need its own virtual range for every segment, mapped onto the arena's physical memory at the planned offset (`cuMemMap` with an offset).
3. **The eager-pair tolerance is too tight for an atomics plan.** The replay against the first eager run differed by:

   | Quantity | Replay vs first eager run | Eager pair |
   |---|---:|---:|
   | Energy | 1.0e-4 | 1.1e-5 |
   | Forces | 2.6e-6 | 2.3e-6 |

   One eager pair underestimates the run-to-run spread, so the build fell back on noise.

**A/B (both sides fell back, so this is the default against itself):** b/a 1.0015 and 0.9987. The arena-on capacity scan stopped at 37³ passing and 38³ failing, the same as the default. The arena-off scan did not finish within the job.

## Decision

The arena is shelved:
- as designed (sealed, first-step keys) it saves nothing at the A100 limit;
- its overlap mode needs a virtual-memory remap that torch's allocator does not promise to tolerate.

Reopening it would need all of these:
- planning across recaptures (an unsealed arena that re-plans);
- per-segment virtual ranges on shared physical memory;
- a noise estimate from several eager runs.

The branch `fastmace/capture-arena` keeps the code and these records.
