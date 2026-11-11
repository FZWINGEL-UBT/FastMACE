# Capture arena, stage 0 (10 November 2026): stopped, then reopened as the segment-level arena (A)

**Question.** Can the CUDA graph's requests be served at planned offsets in one buffer, so the graph pool's
placement loss comes back? The offline bound on the r3 SrTiO3 36³ trace (`results/rung-b-20261109`) said a
lifetime-aware arena would recover about 80 % of it, about 11 kB/atom.

**Answer.** Not through `torch.cuda.MemPool` with a `CUDAPluggableAllocator`. The shim builds and runs everywhere, and
the captures are deterministic. But the caching layer still owns block placement inside the pool, so a shim can only
place whole segments. That recovers 42–52 % of the placement loss. Block-level placement, which would recover 73–99 %,
needs the process allocator replaced. The conductor first stopped the work here (option C). Later the same day the owner
allowed testing our own allocator if it saves memory, carefully, so the segment-level arena (option A) is reopened with
guards (below). The shim and the stage-0 script stay on the branch as lab tools.

## Setup

- FastMACE `fastmace/capture-arena` (stage-0 tools on `a4a44eef`, the allocator-check exemption on `030aba32`; main
  `7d18028d`). torch 2.14.1+cu130 everywhere.
- OMAT-0 medium, float32, SrTiO3 (cubic, a = 3.905 Å) at the repeat shown, thermalised to 300 K. `fm.Simulation`
  with Langevin (1 fs, 0.002 /fs), 20 steps, default plan for the size and GPU.
- The engine's graph pool goes onto the shim: `torch.cuda.graph_pool_handle` returns
  `torch.cuda.MemPool(CUDAPluggableAllocator(shim)).id`. The allocator's memory history (`enabled="all"`) records the
  pool's block-level events.
- Machines:
  - **RTX 5060 Ti:** local WSL, `mace_cueq_oeq`, through `tools/ci/local_gpu.py`.
  - **RTX 4000 Ada:** leto, NAS `mace_cueq_oeq`, after the idle check.
  - **A100-SXM4-80GB:** Alex job 4977416 on node a0537, `fm-a100omat/env`.

## The shim

[`tools/bench/arena/arena_shim.c`](../../tools/bench/arena/arena_shim.c) is plain C with no CUDA headers. Device
memory comes from the driver API (`cuMemAlloc_v2` and `cuMemFree_v2` through `dlopen("libcuda.so.1")`), so the
system gcc builds it (`gcc -O2 -shared -fPIC -ldl`), not `torch.utils.cpp_extension`. It has two modes:

- **Record:** every request goes to `cuMemAlloc` and is logged.
- **Planned:** the k-th malloc of the planned size is served at `base + offset[k]`, and anything else falls back to
  `cuMemAlloc` and is counted as a mismatch.

[`tools/bench/arena/arena_stage0.py`](../../tools/bench/arena/arena_stage0.py) has the `build`, `toy`, `md`
and `compare` steps and the offline bounds (`segment_bound`).

| Check | 5060 Ti | Ada | A100 |
|---|---|---|---|
| Compiler | gcc 13.3 | gcc 11.4 | /usr/bin/gcc |
| Toy capture into the shim pool (MLP forward + `autograd.grad`), replay versus eager | bitwise | bitwise | bitwise |
| Two MD captures, shim request sequence | identical (10³, 30³) | identical (34³; 36³ no_split) | identical (36³) |
| Two MD captures, block-level pool events | identical (10³, 30³) | identical (34³; 36³ no_split) | identical (36³) |

The shim saw these requests in one MD build and 20 steps: 13 mallocs at 10³, 27 at 30³ (5060 Ti) and 24 at 36³
(A100). It saw no frees: the caching layer keeps every segment until the pool is released.

## Granularity: why the shim reaches only 42–52 %

`MemPool(allocator)` keeps the caching allocator's block management. The pool asks the shim for segments, then splits
them and reuses freed blocks itself. So the shim never sees block lifetimes, and the most it can do is overlap
segments that are idle over disjoint intervals.

The bounds below come from the recorded block trace of the same captures (`segment_bound`: first-fit, largest
first):

- **Segment arena:** each segment occupies the arena only while one of its blocks is live.
- **Block arena:** every block is placed by its own lifetime.

| Run | Live peak | Caching allocator (reserved) | Segment arena (a shim's best) | Block arena |
|---|---:|---:|---:|---:|
| A100, 36³ (233,280 atoms) | 10.31 GiB | 14.23 GiB | 12.45 GiB (−1.79, 45 % of the loss) | 10.87 GiB (86 %) |
| 5060 Ti, 30³ (135,000 atoms) | 8.36 GiB | 10.88 GiB | 9.82 GiB (−1.05, 42 %) | 8.39 GiB (99 %) |
| Ada, 34³ (196,520 atoms) | 8.42 GiB | 13.68 GiB | 10.96 GiB (−2.72, 52 %) | 9.82 GiB (73 %) |
| 5060 Ti, 10³ (5,000 atoms) | 0.32 GiB | 0.41 GiB | 0.39 GiB (28 %) | — |

The allocator's own options do not help:

- **`MemPool(no_split=True)`** is worse: A100 segment arena 16.16 GiB, Ada 14.66 GiB. Unsplit segments are reused
  whole by smaller requests.
- **`max_split_size_mb:20` without expandable segments** (5060 Ti, 30³) gives a segment arena of 9.83 GiB, the same
  as the default, and reserves 18.5 GiB.
- **The same setting with `expandable_segments:True`** fails in an eager allocation ("CUDA driver error: device not
  ready").

## The options, and the decision

- **(A) Segment-level arena via the shim.** About 1.8 GiB at 36³ on the A100, about 8 kB/atom, roughly +7–9 % atoms.
  - Overlapping segments is only safe if the caching layer's later block choices match the recording. Its free-block
    order breaks ties by address, and planned addresses differ from recorded ones.
  - So a planned capture would have to be re-recorded and checked, with a fallback. A miss would silently alias live
    tensors on the main MD path.
- **(B) Block-level placement.** Needs the process allocator replaced (`change_current_allocator` before any CUDA use),
  with our own caching for every eager allocation as well. A global change.
- **(C) Stop.** Chosen first by the conductor (10 November 2026): A's gain does not justify a silent-aliasing risk on
  the main MD path, and B is too global.
- **Reopened as (A)**, later on 10 November 2026: the owner allowed testing our own allocator if it saves memory,
  carefully. The conductor's guards:
  - **Native Linux first:** the Alex A100, then leto or murphy after the idle check, where oversubscription is a
    clean out-of-memory error.
  - **The local 5060 Ti only once it works,** and only with all of these:
    - the shim refuses any `cuMemAlloc` beyond `cuMemGetInfo`'s free memory minus 1 GiB;
    - `set_per_process_memory_fraction(0.9)`;
    - a 15 GiB reserved-memory watchdog;
    - sizes from 10³ up, nothing unattended above 20³;
    - no `max_split_size` without expandable segments.
  - **Overlapping segments only behind a build-time plan-match check:** re-record the planned capture's block trace;
    any mismatch falls back to the caching pool.
  - **Forces and stress bitwise equal** to the default capture on the same plan, checked at build and in a GPU test.
  - **Off by default** until the A/B, the bitwise check and the new limits are in on the A100 and the Ada.
  - The `check_allocator` exemption is in (`ProcessRecord.arena_pools`, with a CPU test).

## Finding: the shim crashed the local PC twice (HYPERVISOR_ERROR)

**Never run the shim, this script, or any `max_split_size_mb` / `expandable_segments` / `CUDAPluggableAllocator`
configuration on the local PC (WSL2).**

Both crashes on 10 November 2026, at 14:16 and 17:38 CET, were Windows bugcheck 0x00020001 HYPERVISOR_ERROR, which
took the WSL2 GPU paravirtualisation down. Both happened during the same local job:

- SrTiO3 30³ on the RTX 5060 Ti.
- The shim pool (`cuMemAlloc` through `dlopen("libcuda.so.1")`).
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:20` without expandable segments.
- The allocator's memory history on.

The 14:06 run died in its first record. The 17:08 rerun finished that record (`e`) and died in the second (`f`).

The one record that finished reports 27.9 GiB reserved on the 16 GiB card. The shim pool alone held 18.5 GiB in 52
segments, and the step took 1,260 s against 210 s with the default pool. So device allocations spilled into shared
system memory under WDDM.

There was a warning just before the first crash: the same probe with `expandable_segments:True` had failed in an
eager allocation with "CUDA driver error: device not ready".

The shim's default-pool and `no_split` records at 30³ (12.7 and 14.0 GiB reserved, expandable segments on) did not
crash. Still, the conductor banned every configuration of the shim on the local PC (10 November 2026), and runs on
workstations or Alex are not needed either.

`arena_stage0.py` now refuses to start on WSL, and the WSL launchers (`local0*.sh`, `launch_local0*.sh`) are removed
from the branch.

## What a future attempt needs

- **The allocator-check exemption.** `fastmace/engine/constraints.py:check_allocator` refuses an engine when any
  allocator segment is not expandable. A pluggable pool's segments never are. On the Ada at 36³, an engine built while
  the shim pool held segments failed the check:

  ```text
  30 of 33 allocator segments are not expandable
  ```

  The stage-0 script filters out the arena pool's segments (`segment_pool_id`). An engine-side arena would need the
  same exemption in `check_allocator`.
- **Memory-probe headroom.** A pool on a pluggable allocator cannot use expandable segments. At 36³ on the Ada, the
  default-pool capture ran out of memory: 16.25 GiB in private pools on a 19.5 GiB card, more than the probe budgets
  for.
- **A plan-match check.** Re-record the planned capture and compare it with the plan, as described under (A).

## Files

- `stage0.jsonl`: the stage-0 MD records without their sequences: GPU, repeat, shim summary and digests,
  block-event digest, bounds, `max_reserved`.
  - Tags: `a`/`b` the default pool, `c`/`d` `no_split`, `e` `max_split_size_mb:20`.
- The full records with their sequences stay on the machines:
  - WSL: `~/fm-ada-capacity/arena/stage0-10x10x10-r2`, `stage0-30x30x30`.
  - leto: `/mnt/homes/zwingel/fm-arena/ada-36`, `ada-36b` (the out-of-memory run), `ada-34`.
  - Alex: `/home/atuin/v130cb/v130cb18/fm-arena/runs/stage0-4977416`.
- Job scripts: `tools/bench/arena/stage0_leto.sh`, `stage0b_leto.sh`, `alex_arena0.sh`. The WSL launchers were
  removed after the crashes; they are in the branch's history at `a4a44eef`.
