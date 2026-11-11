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
busy time, which covered the gaps: it reported 4.39 s busy of a 4.74 s profiled call, against 1.96 s of kernels in
the call (`kprof7`). On the 5060 Ti this did not matter (its GPU is busy). `gapprof2.py` counts only kernel, copy and
fill events: PENDING (Alex 4981405, local job 44).

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
