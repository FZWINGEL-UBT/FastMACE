# CAP-T03: device memory outside PyTorch's allocator

Plan task CAP-T03 ([plans/capacity.md](../../plans/capacity.md)). The capacity scan
([capacity-16gb.md](capacity-16gb.md)) saw the device memory in use outside PyTorch's caching allocator
(`total − free − reserved`) grow from 1.13 GiB at the start to 2.2 GiB during float32 builds. CAP-T01
([capacity-T01.md](capacity-T01.md)) found the remaining limits to be driver errors ("device not ready / Failed to
create GPU mapping") during captures on a nearly full card.

## Verdict

**The memory outside the allocator is WSL's, not FastMACE's.** The same points were run on the local RTX 5060 Ti
under WSL2 and on leto's RTX 4000 Ada under native Linux, with the same FastMACE commit, after the build:

| Point | WSL2 (local) | Linux (leto) |
| --- | --- | --- |
| NbCl5 float32, 25,200 atoms | 2.11 GiB (1.13 at the start) | **0.28 GiB** (0.17 at the start) |
| SrTiO3 float32, 14,700 atoms | 1.48 GiB, rising to 5.2 GiB at the failure | **0.28 GiB**, passes |
| NaCl float64, 8,000 atoms | 1.41 GiB, rising to 7.0 GiB during the warm-up, fails | **1.10 GiB**, passes |

Under WSL2 the extra memory is tied to live CUDA graphs:

- It appears after the first force-graph capture (traces in `results/cap-t03-20261029/wsl/`).
- It stays through `empty_cache`.
- It drops to about 2 GiB only when the graphs are destroyed, after the failure.
- It grows with the system size: 0.05 GiB above the start at 5,184 NbCl5 atoms, 0.98 GiB at 25,200.

None of this shows on Linux. FastMACE cannot release memory it does not hold.

**The candidate fix is reverted:**

- What it did: wait after `empty_cache` until `cudaMemGetInfo` shows the released bytes free, at most 5 s, before
  captures, after a force-graph release, after the self-checks and when an engine is dropped
  (`fastmace._device_memory`, commit `d62aa68b`).
- Why it was tried: the trace shows a release delay of about 0.7 s under WSL2.
- Result: it changed none of the three failing sizes (NbCl5 30,600, SrTiO3 14,700, NaCl float64 8,000 atoms).
- Commit `5feca148` reverts it.

**What this means for capacity:**

- **Native Linux:** a 16 GB card fits larger cells than this machine does. On leto, SrTiO3 at 14,700 atoms peaks
  at 15.26 GiB reserved plus 0.28 GiB outside. That would just about fit a native 16 GB card. Under WSL2 it fails.
- **Large cells on this machine:** they should run under native Linux, or on a card with more memory.
- **The platform-independent levers** are fragmentation (CAP-T04) and the activations themselves (CAP-T07,
  CAP-T08). Reserved minus allocated at the peak is 1.2–2.0 GiB even on Linux:
  - SrTiO3: 14.02 GiB allocated against 15.26 GiB reserved;
  - NbCl5: 11.0 GiB against 13.0 GiB.

## Measurements

- **`--trace`** (`tools/bench/capacity_scan.py`): a thread samples `cudaMemGetInfo`, `memory_reserved` and
  `memory_allocated` every 50 ms, and FastMACE's log lines (captures, releases) are recorded with their times.
- **WSL2, NbCl5 float32 at 25,200 atoms:**
  - After each `empty_cache` that released about 12 GiB, the free memory caught up about 0.7 s later.
  - After the captures, about 0.9 GiB stayed outside the allocator for the rest of the run.
- **WSL2, NaCl float64 at 8,000 atoms, failing:**
  - Before `fit`'s release: 0.21 GiB free, 8.73 GiB reserved and 6.99 GiB outside the allocator.
  - After the release: 8.68 GiB free, but 6.28 GiB still outside.
  - The new warm-up then reached 8.2 GiB reserved with 7.7 GiB outside and 0 free, and failed.
  - After the failure the outside share fell to 2.06 GiB.
- **Linux:** the outside share stays at 0.28 GiB for float32 and 1.10 GiB for float64 (library workspaces, the
  context) at every phase. The sampled maxima there (8–14 GiB) are single samples during an `empty_cache`, where
  the thread read `reserved` after the release and `free` before it.

## Setup

- **Local:** Fabi-PC, WSL2, RTX 5060 Ti 16 GB (driver 610.88), `mace_cueq_oeq` (torch 2.14.1+cu130),
  `tools/ci/local_gpu.py run` reservations with the idle check. FastMACE:
  - `762527e6`: the size comparison (`-t03`);
  - `f7698e58`: the traces (`-trace`);
  - `332d2a2a`: the wait-after-release candidate (`-settle`).
- **leto:** RTX 4000 Ada 20 GB (driver 595.99.02), NAS `mace_cueq_oeq` (torch 2.14.1+cu130), FastMACE `764ff457`
  (main `b6421afc` plus the candidate and the tools). The other-users and GPU check ran before every job
  (`results/capacity-16gb-20261029/ws.sh`). The candidate only acts after a release, and none of these points
  failed.
- **Model and run:** MH-1 (`mh1-r2scan`), default plan (`force_graphs=fit`), 20 warm-up and 200 timed steps of
  Langevin NVT, one process per point.
- **Raw data:** `results/cap-t03-20261029/wsl/` and `results/cap-t03-20261029/leto/`, one JSON per point. The
  traces are in the `trace` key.
