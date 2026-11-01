# Tiled path: the call's whole-system memory, the MD limit, and the call's prelude

[tiled-compiled-default.md](tiled-compiled-default.md) found that a second call needs about 2 GiB more than the
first. Because of that, at 67.9M NbCl5 atoms both modes ran out of memory on the second call. The single-call
limit was 70.96M, while the repeated-call (MD) limit was only 65-68M. The owner's 70M target should hold for MD.

This report covers:
- where the second call's memory goes;
- the fix, which is exact;
- the limit after the fix;
- a speed finding from the same code: the tile plan's per-atom Python loop.

Unless a section says otherwise, the setup is MH-1 float32 hybrid (`mh1-r2scan`) with `mace_cueq_oeq`, on an
NVIDIA RTX 4000 Ada (20 GB) on leto or murphy. The tiled force field uses:
- tiles of R = 4,096;
- tile lists;
- the state on the host, aliased;
- the species index;
- the force field's own global list left empty.

## Where the memory goes

**Per-call memory snapshot** (`results/tiled-call-memory-20261101/snapshot_calls.py`, 9.26M atoms):
- The two calls' largest allocations are the same tile workspace: the convolution's 0.51 GiB, the gated products
  and the linear layers.
- The tile workspace does not grow between calls.

**Peaks of the whole-system stages** (`stage_peaks.py`, 22.4M atoms):
- Each stage of the call is wrapped, with a running maximum across the resets.
- Before the tiles run, the call holds whole-system transients, and they set its peak at scale:

| Stage | Peak above its start | Left after it |
| --- | ---: | ---: |
| edge source: the tile list's binning | 144 B/atom | 65 B/atom |
| the tile plan | 96 B/atom | 8 B/atom |
| the global buffers (forces, node energies, indices) | 32 B/atom | 32 B/atom |
| the whole call (first / second) | 180 / 169 B/atom | |

- At the second call these transients sit on top of two things:
  - the first call's global buffers;
  - the forces still held by the caller.
- That is the second call's extra 2 GiB at 65M.

## The fix (exact)

**In the binning** (`TileNeighbours.bin`):
- It works in chunks of `CHUNK` = 2^22 atoms.
- It keeps each atom's integer cell and bin as int32: the same integer values as its float64 and int64 forms.

**In the tile plan** (`plan_tiles`):
- It reduces the fractional coordinates in place.
- It computes the block index in chunks.
- It frees each array as soon as it is done with it.
- With per-tile lists it takes no degree arrays (`row_ptr` None). Each block is split into consecutive pieces of
  R, vectorised, which is the split that all-zero degrees gave.

**CPU tests:**
- `test_binning_and_tile_plan_in_chunks_are_bitwise_unchunked` runs with 7-atom chunks. Against one chunk:
  - the offsets, bins, order, tiles and edges are bitwise those of one chunk;
  - the plan without degrees gives the same tiles as all-zero degrees.
- `test_tiled.py`, `test_layering.py` and `test_neighbors.py` on trillian at 31a89edf: 156 passed, 91 skipped.

**Stage peaks after the fix** (leto, 22.4M atoms; `runs/leto-stagepeaks2-*`):

| Stage | Before | After |
| --- | ---: | ---: |
| edge source (peak above / left) | 144 / 65 B/atom | **72 / 33 B/atom** |
| the tile plan (peak above) | 96 B/atom | **48 B/atom** |
| the whole call, second call (peak above) | 169 B/atom | **116 B/atom** |

## The two-call limit after the fix

`limit2.sh` ran on leto at FastMACE 8203e6a7. That is this branch's code before its rebase onto main, with the same
library diff. It used `tiled_capacity.py --calls 2 --mode eager`, R = 4,096. The host guard leaves 20 GB free.

| NbCl5 atoms | Call | Allocated before | Peak above it | Total | Call time | Host peak | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 67,917,312 | 1 / 2 | 6.97 / 9.00 GiB | 6.90 / 5.65 GiB | 13.87 / 14.65 GiB | 1,922 / 1,923 s | 205 / 206 GiB | **valid, two calls** |
| 70,957,944 | 1 / 2 | 7.28 / 9.40 GiB | 7.21 / 5.90 GiB | 14.49 / 15.30 GiB | 1,857 / 1,854 s | 214 / 215 GiB | **valid, two calls** |

The runs are in `runs/leto-limit2-*`.

**Compared with before** (the same 67.9M, [tiled-compiled-default.md](tiled-compiled-default.md)):
- The first call's peak above its start falls from 10.71 to 6.90 GiB.
- The second call, which ran out of memory, now totals 14.65 GiB.

**At 70.96M the repeated call leaves about 4 GiB of the card free.** The edge counts and the forces check are
valid. **The 70M target holds for MD on 20 GB.**

The GPU memory no longer sets the limit at 70.96M; the host does:
- 215 GiB on leto, with the tile-list cache;
- `list_cache=False` saves 19 GB of that.

The compiled two-call run at 70.96M was not measured. Two things point to it fitting:
- compiled holds 0.065 GiB more than eager between calls;
- auto mode compiles when 1.07 GiB is free at the eager peak.

## The call's prelude: a per-atom Python loop in the tile plan

**Tool.** `results/tiled-speed-20261101/gaps.py` (murphy, 884,736 atoms, compiled, overlap on):
- It profiles one call with torch.profiler.
- It merges the device's activities into busy intervals and buckets the gaps between them by length.
- It charges each gap to the stage the host was in when the device ran dry.
- nsys is not installed on the workstations.

**Before the fix** (main 7cca7c27; `runs/murphy-gaps2-*`, `runs/murphy-gaps3-*`):
- **Busy share.** The device is busy 12.95 s of a 15.89 s call, 81 %. The 64 % of
  [tiled-speed.md](tiled-speed.md) was measured with the stage wrappers' synchronisations still in place.
- **The setup gap.** 2.22 s of the 2.94 s idle time is a single gap before the first tile, and 2.26 s of it is
  in `plan_tiles`. With per-tile lists it walked every atom in a Python loop over an all-zero degree tensor
  (`int(degree[i])`, about 2.5 µs per atom).
- **The short gaps.** The 46,354 gaps of up to 50 µs, from launches and Python between kernels, add up to 0.19 s.
- **Eager.** The device is busy 18.5 s of 25.5 s. Its gaps of up to 50 µs add up to 2.6 s.

**After the fix** (this branch with gaps.py; `runs/murphy-gaps4-cm-compiled-*`):

| Measure | Before | After |
| --- | ---: | ---: |
| The tile plan | 2.26 s | 4.8 ms |
| The profiled compiled call | 15.93 s | 13.59 s (−14.7 %) |
| Device busy | 81 % | 95 % (12.88 s of 13.59 s) |
| Idle time | 2.94 s | 0.71 s |

The two runs are separate processes on murphy, not a same-process A/B. The difference is about five times the
cross-process noise (about 3 %).

**CUDA graphs for the padded slots: dropped** (conductor, 1 November).
- The idle time they would remove, the short gaps between kernels, is 0.19 s of a 13.6 s compiled call: 1-2 %.
- Eager would gain more, but compiled is the default.
- Reusing the tile plan across MD steps is not needed for speed either: the plan now takes 5 ms.

## What is left

- With the setup gap gone, the compiled call is 95 % device time. The next levers are inside the kernels:
  interaction 0 (about 7 s) and interaction 1 (about 6 s).
- Building the padded slots' keys (`_slot_keys`) costs 0.74 s of host time per call at 884,736 atoms. It runs
  before the device has work. Most of the remaining 0.71 s of idle time is probably this, but the
  profile does not separate it.
- The plan with degrees (the global-list source) still has the per-atom loop. It is not the default at scale.
