# The graph pool's packing slack: where it comes from, and what recovers it

The owner's go-ahead of 30 October 2026, through the conductor, after [capacity-ledger.md](capacity-ledger.md):

- **The goal:** recover the 1.88 GiB the graph pool reserves above its live peak at NbCl5 float32, 51,840 atoms, on
  the RTX 5060 Ti, without changing any result.
- **The falsifier comes first,** with a stop below about 0.5 GiB recoverable.
- **Candidates:** a shared workspace for the large buffers, largest-first allocation, and a pre-reserved workspace.

Base: FastMACE `main` `88c9ba95`, with CAP-T06's noise cap.

## Verdict

**The falsifier passes; no mechanism tried keeps the gain at the limit. No library change.**

- **In principle, 1.47 GiB.** Packed offline, largest first, the same blocks with the same lifetimes fit in 11.88 GiB,
  against the allocator's 13.35 GiB. The live peak is 11.49 GiB.
- **The allocator's own settings make it worse.** These were measured, and predicted by a simulation of the
  allocator that reproduces its footprint to 0.03 %:
  - without expandable segments the pool needs 13.93 GiB;
  - with `max_split_size_mb:256` as well, it runs out of memory.
- **A second pool for FastMACE's `segment_gemm` buffers.** Its kernels allocate in a second `torch.cuda.MemPool`,
  through `torch.cuda.use_mem_pool`, in the forward thread and in the autograd thread:
  - the simulation gives 0.76 GiB at 51,840 atoms;
  - it gives nothing at 9,216 atoms (3.68 against 3.61 GiB);
  - the re-capture at 9,216 atoms gives 0.14 GiB: the graph pool 2.93 plus the second pool 0.57, against 3.63;
  - but the steady state is worse, and at 51,840 atoms the build fails inside a capture (`CUDA driver error: device
    not ready`). Two pools cannot reuse each other's free blocks, so near the limit the split costs more than it
    saves.
- **What is left needs a design decision** (see "Options"):
  - a planned workspace that FastMACE's ops write into (estimated at most 0.76–0.86 GiB at this size), or
  - a capture-time allocator that follows an offline plan (up to 1.47 GiB).

## Where the slack comes from

**Tool:** `tools/bench/pool_packing.py`, new and offline. It reads `memory_ledger.py`'s record, which now keeps
every allocator event inside the graph pool during one re-capture of the stages: 581–613 events at these sizes.
It then does three things:

- **Replays** the events per segment.
- **Packs** the same blocks offline: largest first, each at the lowest offset free over its lifetime.
- **Simulates** the caching allocator's large-block policy:
  - expandable segment: smallest free block that fits, split, coalesce on free, extend at the end;
  - without expandable segments: a segment per new block, and oversize blocks reused only within 20 MiB of their
    size.

NbCl5 6×6×20, 51,840 atoms, default plan:

| | GiB |
| --- | ---: |
| Pool reserved | 13.365 |
| Replayed footprint | 13.347 |
| Best-fit simulation of the allocator | 13.343 |
| Offline largest-first packing | **11.881** |
| Live peak (no placement goes below it) | 11.486 |
| **Recoverable in principle** (allocator minus offline packing) | **1.466** |

- **How the pool grows:** 63 growth events. The largest come from blocks of 0.4–1.6 GiB that find only smaller holes
  below the high-water mark. For example:
  - a cuEquivariance `uniform_1d` block of 1.58 GiB grew the pool while 1.86 GiB was free below in 12 holes, the
    largest 0.76 GiB;
  - the two `segment_gemm` joined products of 1.58 GiB each grew it with 0.70 and 0.41 GiB free below.
- **No deferred frees:** a block's live time is the same whether its free counts at the request or at completion, and
  every allocation is on one stream.
- **The blocks:** few and large, from FastMACE's kernels, cuEquivariance and Inductor. The live sites are listed in
  capacity-ledger.md.

At 9,216 atoms (4×4×8) the pool reserves 3.631 GiB against an offline packing of 3.344 GiB and a live peak of
3.274 GiB: 0.27 GiB in principle.

## What was tried

| Candidate | Predicted (simulation) | Measured | Verdict |
| --- | --- | --- | --- |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` | 13.90 GiB | pool 13.93 GiB, steady peak reserved 15.08 GiB (default 14.11) | worse |
| … with `max_split_size_mb:256` | 16.39 GiB | out of memory at build | worse |
| One site at a time in an arena of its own | best 13.29 GiB (the fused forward's message) | — | no gain |
| Shared workspace, packed offline: all FastMACE kernels' blocks | 7.98 + 4.50 = 12.48 GiB | — | 0.86 GiB, if realised |
| … the `segment_gemm` products and input gradients | 9.42 + 3.16 = 12.59 GiB | — | 0.76 GiB, if realised |
| … every block of 256 MiB or more | 1.18 + 11.49 = 12.67 GiB | — | 0.67 GiB |
| Second `MemPool` for `segment_gemm`'s kernels, all calls | 12.59 GiB (as the workspace) | 9,216 atoms: pool 2.93 + second pool 1.13 GiB, peak reserved 4.53 (baseline 4.10); 51,840: build fails | worse |
| … only allocations made while capturing | same | 9,216 atoms: re-capture 2.93 + 0.57 = 3.49 GiB (baseline 3.63); steady peak reserved 4.67 GiB; 51,840: build fails | worse at the limit |

The second pool, in detail:

- `memory_ledger.py --second-pool segment_gemm` re-registers the CUDA kernels of `fastmace::joined_products` and
  `fastmace::joined_products_input_grads` to run inside `torch.cuda.use_mem_pool(pool)`.
- That routes their allocations in whichever thread runs them, including autograd's.
- PyTorch 2.14 supports the context during a capture: the graph retains the pool.
- **With every call routed,** the eager warm-ups also allocate in the second pool, and the capture does not reuse
  those blocks. The second pool doubles.
- **With only capturing calls routed,** the re-capture's blocks leave the graph pool as intended. But the engine runs
  later warm-ups inside the graph pool (`warmup_in_graph_pool`), so their `segment_gemm` buffers stay there.
- **In both cases** the two pools cannot share free blocks. At 51,840 atoms the build fails during a capture with
  `CUDA driver error: device not ready`, the allocator's out-of-memory path inside a capture.

## Options for the conductor

1. **A planned workspace for `segment_gemm`.** The ops write their products and input gradients into views of one
   buffer allocated in the graph pool at the start of the force stage. The views are laid out from the model: the
   number of interactions and the product widths. Nothing is keyed to shapes.
   - **Gain:** estimated at most 0.76 GiB at 51,840 atoms, and nothing at 9,216 atoms.
   - **Exactness:** the same arithmetic, written elsewhere, so the results are bitwise unchanged.
   - **Cost:** it needs out-arguments on two custom ops (`mutates_args`), through Inductor's and AOT autograd's
     tracing of the compiled model. The product's lifetime is set by cuEquivariance, which saves it.
   - **Prerequisite:** the simulation must be checked against a real capture first, since the second pool showed that
     a sound simulation can miss engine behaviour (warm-ups in the pool).
2. **A capture-time allocator following the offline plan,** through `torch.cuda.MemPool` with a pluggable allocator
   for the graph pool, with the plan recorded from a first capture.
   - **Gain:** up to 1.47 GiB.
   - **Cost:** it replaces part of the allocator, a process constraint (SPEC §8.7), and needs a SPEC decision.
3. **Stop here.** About 0.8 GiB at the limit would be one more NbCl5 grid step at most.

My recommendation is option 3, unless the owner wants the last grid step: then option 1, behind a capture check as
described.

## What changed

- `tools/bench/memory_ledger.py`:
  - saves the pool's event list and segment ranges;
  - records the allocator setting;
  - new options `--tag`, `--refuse` (the fallback off, so the first failure is raised) and `--second-pool` (the
    experiment above).
- `tools/bench/pool_packing.py` (new): the replay, the offline packing, and the allocator simulations.
- `results/cap-ledger-20261030/local.sh`: `LEDGER_ARGS` and `TAG` for ledger jobs.
- `results/cap-pool-20261030/`: the ledger records.
- No library code changed.

## Checks

- Ruff is clean on the three tools.
- `test_layering.py` and `test_ab_rewrites_tool.py`, commit `25aebda0`: **34 passed**. They ran on coraline (GPU
  hidden, `-n 2`), because leto had another user's job on its GPU.

## Setup

- **Machine:** Fabi-PC, WSL2, RTX 5060 Ti 16 GB (driver 610.88), `mace_cueq_oeq` (torch 2.14.1+cu130).
- **GPU access:** `tools/ci/local_gpu.py run` reservations, with the idle check before every job.
- **Source:** pinned copies of `fastmace/cap-pool` on `88c9ba95`.
- **Model:** MH-1 (`mh1-r2scan`), hybrid backend, float32, Langevin NVT at 300 K.
- **Allocator variants:** set with `PYTORCH_CUDA_ALLOC_CONF` per process; the engine's `"auto"` then keeps them.
- **Raw records** in `results/cap-pool-20261030/`:
  - `ledger/`: the ledgers, each with its `*.packing.json` from `pool_packing.py`;
  - `jobs.log`.
