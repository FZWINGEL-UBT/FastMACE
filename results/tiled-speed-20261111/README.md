# Tiled call, 11 November 2026: host heap trim, A100 gate on the merged tip, where the A100 call's time goes

Branch `fastmace/tiled-speed`. Ideas list: `plans/ideas-20261111-tiled.md`.

## Dropped kept lists stayed resident (idea 1): `TILE_HEAP_TRIM`

The A100 397.5M-atom record (`results/tiled-speed-20261110`, job 4981016) had about 110 GiB of host memory that no
live array accounted for. That forced float16 state on the second call and switched the list cache off. A
local census (`census38/`, `census39/`; 5060 Ti host, 39 GiB; SrTiO3 20,098,395 atoms, R = 8,192; `jobs/job38.sh`,
`jobs/dropcensus.py`) forces the first call to drop its kept lists, the 397.5M record's path, and reads the resident
set and glibc's `mallinfo2` around the drop.

| (GiB) | before the drop | after the drop | after `malloc_trim(0)` | live host arrays |
| --- | ---: | ---: | ---: | ---: |
| job 38 (3899b8b0, no trim) | 12.08 | 8.11 | 5.37 | 1.17 |
| job 39 (d4014c75, `TILE_HEAP_TRIM`) | 12.07 | 5.48 | 5.37 | 1.17 |

The kept lists were 7.00 GiB, 4 GiB of them pinned. Without the trim, 2.7 GiB of the freed lists stayed resident
(glibc free 2.77 GiB): the lists' few-MB tensors come from glibc's heap once its mmap threshold has risen, and a
freed heap chunk stays resident until `malloc_trim`. Live arrays did not change, so nothing still referenced the
lists. With the trim the resident set after the drop is the trimmed one to within 0.11 GiB.

`TILE_HEAP_TRIM` (default on, `fastmace/model/tiled.py`) calls `malloc_trim(0)` after the kept lists are dropped and
after the list cache is cleared; stats `heap_trims` and `heap_trim_s` count the calls and their time. It is memory
only (no rounding change).

Not addressed: at the end of a call glibc still holds about 1.5 GiB (16.71 GiB resident against 15.24 trimmed in
both jobs).

### A100 397.5M with the trim (Alex 4981047)

PENDING

## A100 force/stress gate on the merged tip (Alex 4981048)

Source 69a48049 (`fastmace/tiled-speed` with main 4e9388c8 merged), A100-SXM4-80GB, `tools/force_stress_gate.py`
with the production constant; switches all off and all on. Ratios to the ASE float32 yardstick (limit 3):

| system | off: energy / forces / stress | on: energy / forces / stress |
| --- | --- | --- |
| SrTiO3 | 10.59 / 1.049 / 0.660 | 8.80 / 1.085 / 0.660 |
| NaCl | 1.94 / 0.905 / 1.020 | 1.91 / 0.843 / 1.041 |
| NbCl5 | 8.31 / 1.000 / 1.698 | 8.64 / 1.005 / 1.356 |

All pass; the energy ratios above 3 are under the 0.01 meV/atom floor and as high with the switches off. Records in `gate/alex-4981048/`.

## Where the steady device-state call's time goes (idea 2)

### A100 kernel regions (Alex 4980926 `a100prof/`, Alex 4981379 `kprof7-a100/`)

SrTiO3 975,560 atoms, device state, R = 16,384, steady calls 3.77 / 3.73 s (3.8 µs/atom; the whole-system call:
1.96 µs/atom). `kprof7.py` charges every kernel in the profiler's trace, through its correlation id, to the labelled
method that launched it (autograd's device threads to the caller's label): 24,548 kernels, 2,013 ns/atom (union
1,995). The GPU is therefore idle about 47 % of the call; the gap to the whole-system call is host-side.
(`kprof6.py`, 3899b8b0, which counted only kernels attached to torch ops, had 2,115 ns/atom: close.)

| region (ns/atom) | A100 | 5060 Ti (job 42, `kprof7-local/`) |
| --- | ---: | ---: |
| sweep 2 backward (oeq w′ backward 525 on the A100) | 623 | 1,479 |
| sweeps 1 and 3 backward | 298 | 847 |
| sweep 2 forward (spline 218) | 289 | 1,587 |
| sweep 3 forward (`fwd3o`) | 269 | 771 |
| sweep 1 forward (`fwd1m`) | 267 | 752 |
| scatter | 112 | 110 |
| tile graph | 62 | 111 |
| all kernels | 2,013 | 5,917 (wall 6,036) |

A composed interaction 0 in the tile bodies (idea 2) acts on the two interaction-0 forwards, about 535 ns/atom of
kernels on the A100; the host-side idle time (about 1.8 µs/atom) is the larger lever.

### Local idle-gap profile (job 40, `gap-local/`, `jobs/gapprof.py`)

d4014c75, 5060 Ti, the same system: steady 6.06 / 6.13 s. The GPU is busy 6.005 s of the call's 6.037 s kernel span;
the idle gaps total 31 ms (largest: 15 ms charged to `_call`). The 5060 Ti call is GPU-bound, so its host time
(`_first` 1.76 s, `autograd.grad` 1.74 s, `_fill` 1.10 s self time; 2,856 `cudaLaunchKernel` in `_fill`, 0.87 s)
runs under the kernels. (The first run, `gap-local/job39-gapprof-58.json`, counted the profiler's 6 s `_call` device span as
busy; job 40 sets device spans over 50 ms aside.)

### A100 idle-gap profile (Alex 4981351, `gap-a100/`): wrong

`gapprof.py` counted the GPU-timeline copies of the `record_function` annotations (per-tile spans under 50 ms) as
busy time, which covered the gaps: it reported 4.39 s busy of a 4.74 s profiled call. On the 5060 Ti this did not
matter (its GPU is busy).

### A100 idle share without the torch profiler: Nsight Systems (Alex 4981411, `nsys-a100/`)

69a48049, the same call: `jobs/nsysprof.py` under `nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi`, the
tiled methods and `autograd.grad` as NVTX ranges; `jobs/nsysan.py` reads the sqlite export (busy: the union of
kernel, memcpy and memset intervals within the host's `captured_call` range). Uncaptured calls 3.85 / 3.84 / 3.81 s;
captured 4.12 s, of which the GPU is busy 1.957 s in kernels, 0.023 s in copies and 0.002 s in fills: **idle 52 %**
(49 % against the uncaptured wall). `gapprof2.py` (the torch profiler's trace, kernel, copy and fill events only;
Alex 4981405, `gap2-a100/`) agrees: busy 1.95 s.

Idle gaps by the host's innermost NVTX range at each gap's midpoint (s): `_first` 0.84, `autograd.grad` 0.78,
`_tile_graph` 0.09, `_edges` 0.08, `_second` 0.08, `_keep_lists` 0.075, gaps under 20 µs 0.064, every other range
under 0.035. Unprofiled host self time (`gapprof2`): `_first` 1.31 s over 136 calls (about 9.7 ms each),
`autograd.grad` 1.11 s over 136, while the launches cost little (`cuLaunchKernel` in `_first`: 5,228 calls,
0.064 s). The A100 waits on the Python side of the compiled tile bodies (guards, the Inductor wrapper) and on the
autograd engine of their backward. Next: `TileConfig.compiled_graphs` on the A100 (Alex 4981422).

## The wavefront's row pool on the device (idea 3): rejected; the aliased device state: measured

eacf7cfd lets `alias_state` run with `state="device"`: the wavefront pool's row map and free rows and the alias
window live on the device, and the window's rows are planned on the host once after sweep 1. CPU tests on leto
(`test_tiled.py`, `test_layering.py`): 188 passed. 5060 Ti, 975,560 SrTiO3 atoms, compiled tiles, R 16,384 budget,
both orders (`jobs/job41.sh`, `jobs/job43.sh`):

| A/B | ab | ba | device state |
| --- | ---: | ---: | --- |
| host wavefront (a) against device wavefront (b) | 1.012 | 0.994 | 2.52 GB pool both |
| plain device state (a) against device wavefront (b) | 1.263 | 1.181 | 4.99 against 2.52 GB |
| plain device state (a) against aliased device state, no wavefront (b) | 1.034 | 1.021 | 4.99 against 2.52 GB |

The host copies were not the wavefront's cost any more (the copy stream and the host worker hide them); its plan is
(100 tiles with the boundary skins against 68 and a pool of 91 % of the atoms at this size). The aliased device
state halves the device state for 2–3 %: an opt-in capacity lever, not a default. Records: `devpool/`, `devalias/`.

## CUDA graphs for the compiled tiles on the A100 (`TileConfig.compiled_graphs`)

nsys put the A100's idle time in the host's compiled tile bodies and the autograd engine, which
`TileConfig.compiled_graphs` (opt-in since `fe7b0a7b`, 3 November) replays from CUDA graphs. It had never run beside
uncaptured calls, and the A/B, whose sides alternate on one tiled field with their own globals, found two bugs:

- Alex 4981422 (69a48049): a capture after every slot's graphs had been dropped by a new graph key reused the private
  pool that PyTorch releases with its last graph (`CUDACachingAllocator` assertion `use_count > 0`). Fixed in
  35bcd4ab: a new pool handle once no graph holds the old one, and the graph key checked only in calls that capture.
- Alex 4981467 (35bcd4ab): an uncaptured call replaces `slot["out"]`, and a later replay's scatter found no outputs
  (`KeyError`). Fixed in d7d9e7ce: the graphs' outputs live in `slot["graph_out"]`, and `slot["out"]` points at them
  after each replay.

Both are covered by `tests/gpu/test_tiled_speed_cuda.py::test_compiled_graphs_alternate_with_uncaptured_calls`.

A/B (Alex 4981483, d7d9e7ce, A100-SXM4-80GB, 975,560 SrTiO3 atoms, device state, compiled tiles, R 16,384 budget,
off against on in one process, 4 blocks of 2 calls, `jobs/alex_cgab.sh`; records `cgab-a100/`):

| order | off (s) | on (s) | on / off | pair ratios |
| --- | ---: | ---: | ---: | --- |
| ab | 3.589 | 2.042 | 0.569 | 0.566-0.570 |
| ba | 3.561 | 2.035 | 0.571 | 0.568-0.573 |

3.68 → 2.09 µs/atom (the whole-system call: 1.96); the host's time in the tile loop 3.43 → 1.50 s. Energy within
6e-8 meV/atom, forces within 3.8e-6 eV/Å of a largest 1.7e-4 (the atomics' order, as between two calls). The
default (with a memory guard: the graphs keep private pools; the 16 GB A4000 ran out of memory in the first call at
R 8,192 on 3 November) follows in its own merge request after the gate (Alex 4981512) and the 5060 Ti and Ada A/Bs.
