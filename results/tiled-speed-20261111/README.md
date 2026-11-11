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

Gate (Alex 4982466, 717b9f92 (4981747 died on a node failure after 35 s), the 3x float32 force/stress gate with
`compiled_graphs` off and on, tiled R 48, compiled; `jobs/alex_gate_cg.sh`; records `gate/alex-4982466-cg/`): every
check passes on both sides. With graphs on: SrTiO3 3x3x3 energy 9.72, forces 1.05, stress 0.63; NaCl 2.27, 0.86,
1.14; NbCl5 8.05, 1.03, 1.59 (energy ratios under the 10 meV floor). So `COMPILED_GRAPH_GPUS` lists the A100 family.

RTX 4000 Ada (murphy, 04de68ef, idle check before each run; `jobs/murphy_cg.sh`, `jobs/probe_cg.py`,
`jobs/murphy_cgab.sh`; records `cg-ada-murphy/`). A probe forcing captures (one process, 4 blocks): 40³ (320,000
atoms) b/a 1.0015, the guard on (3 capture pools and warm-ups of 3.63 GiB fit 16.56 GiB); 58³ (975,560) b/a 1.0013,
the guard off (8.63 GiB x 3 do not fit 12.00). The guarded A/B at 58³ (R 8,192, compiled, off against the guard,
4 blocks of 2 calls): 1.0010 (ab) / 1.0007 (ba) at 5.65 µs/atom, the guard off on both orders (4.7 and 8.6 GiB
warm-ups). The Ada's compiled call is not launch-bound either: off, as on the 5060 Ti.

### The memory guard and the RTX 5060 Ti twin (job 49)

The guard (5c890377, branch `fastmace/tiled-graphs`): the first compiled call runs uncaptured and measures its tile
workspace; captures follow only where one capture pool per slot and the warm-ups fit (`compiled_graphs_fit`, x1.1),
and a captured call that runs out of memory drops its graphs and runs again. `TileConfig.compiled_graphs` defaults
to None (on for `COMPILED_GRAPH_GPUS`). The A4000's out-of-memory case (975,560 atoms, R 8,192) is its CPU test's
must-be-off case.

Local job 46 (unguarded, 975,560 atoms on the 16 GB RTX 5060 Ti) was running when the PC bugchecked at 07:29 CET
(0x133; the WSL kernel logged soft lockups and an RCU stall from 07:28:07). Job 49 reran it on the guarded source
(04de68ef) in steps under a 0.9 memory fraction and a watchdog (`jobs/job49.sh`, `jobs/wrap_frac.py`; records
`cg-5060ti-job49/`). Off against on, one process, 3 blocks of 2 calls, R 8,192 (budget 16,384):

| SrTiO3 | atoms | ab | ba | peak (nvidia-smi) | guard (on side) |
| --- | ---: | ---: | ---: | ---: | --- |
| 30³ | 135,000 | 1.007 | 0.990 | 7.9 / 8.2 GiB | on: 2 pools of 1.7-2.2 GiB fit 13.6-14.3 |
| 40³ | 320,000 | 0.998 | 1.004 | 9.8 / 7.9 GiB | ab on (2 x 2.5 GiB fit 12.7); ba off (3 x 4.35 > 14.25) |
| 50³ | 625,000 | 1.000 | 1.002 | 11.7 / 11.7 GiB | off (3 x 4.2, 2 x 7.2 GiB do not fit) |

With graphs on (30³, 40³ ab) the ratios are within noise at about 6.6 µs/atom: the 5060 Ti's compiled call is not
launch-bound as the A100's is, so it stays off `COMPILED_GRAPH_GPUS`. 58³ was killed by the watchdog 95 s in at
15.8 GiB used on the card (the fraction's 14.3 GiB plus Windows' 1.6 GiB; the rule now counts the process's own
memory). The GPU test of the two fixes passed there (1 passed, 51 s; the guard measured, then captured).

## Idea 11: the residual force on the perfect crystal comes from float32 positions

The capacity records' largest force on the perfect SrTiO3 crystal grows from 1.64e-4 eV/Å at 975,560 atoms (226 Å)
to 1.32e-3 at 149M and 320M (1,210 and 1,562 Å), a factor of 8, as the float32 spacing of the coordinates.
`plans/ideas-20261111-tiled-round2.md` asked whether the positions or the float32 shifts carry it.

**Trace** (numpy, the device's `(pos32[r] - pos32[s]) + shift32` against float64 on a perfect lattice; about 14,700
edges within 6 Å of receivers on both box faces and the middle; script `jobs/idea11_vectors.py`):

| box (Å) | origin (Å) | largest vector error (Å) | positions' part | shifts' part | edges inside the box |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 226.5 | 0 | 1.46e-5 | 1.46e-5 | 5.5e-6 | 1.46e-5 |
| 1,210.5 | 0 | 1.07e-4 | 1.07e-4 | 4.9e-5 | 1.07e-4 |
| 1,562 | 0 | 1.07e-4 | 1.07e-4 | 0 (exact in float32) | 1.07e-4 |
| 23.4 (6³) | 0 | 1.3e-6 | 1.3e-6 | 3e-7 | 1.3e-6 |
| 23.4 (6³) | +1,500 | 1.07e-4 | 1.07e-4 | 3e-7 | 1.07e-4 |

The positions carry it: edges that never cross the boundary carry the whole error, and a 6³ cell moved to 1,500 Å
reproduces it. The shifts' own rounding (up to half the box's float32 spacing) comes second.

**The fix** (`TILE_LOCAL_FRAME`, cf76159a, the conductor's tile-local form): each tile's float32 node rows are built in
float64 relative to its first receiver, each halo row at its first edge's periodic image, and the edge keeps only
the shift that remains (zero unless an atom reaches the tile through two images); the float64 shift is
`round(shift32 @ cell^-1) @ cell`. No stored bytes; the rounding is bounded by the tile's extent. Where the call's
positions are on the host: int16 bins of 32 Å and float32 offsets (`_frame_vectors`, 18 B/atom). CPU tests (leto,
167 passed): the vectors within 8e-6 Å (tile-local) and 6e-6 Å (bins) of float64 in triclinic boxes of 226-1,562 Å
placed 1,500 Å from the origin; the call far from the origin is the whole-system call's to float64 rounding.

**On the GPU** (local job 50, cf76159a, RTX 5060 Ti, OMAT-0 medium float32, SrTiO3 6³ (1,080 atoms) at 0, +200 and
+1,500 Å; tiled mode padded, R 256; `jobs/idea11_gpu.py`; records `idea11-gpu-job50/`):

| offset | path | perfect crystal: largest force (eV/Å) | rattled: largest change from the origin run (eV/Å, relative) |
| ---: | --- | ---: | ---: |
| 0 | whole system | 1.89e-5 | - |
| 0 | tiled, frame off | 1.95e-5 | - |
| 0 | tiled, frame on | 4.58e-6 | - |
| +200 | whole system | 8.08e-5 | 3.15e-4 (1.5e-4) |
| +200 | tiled, frame off | 8.33e-5 | 3.14e-4 (1.5e-4) |
| +200 | tiled, frame on | 3.96e-6 | 2.98e-5 (1.5e-5) |
| +1,500 | whole system | 1.33e-3 | 2.69e-3 (1.3e-3) |
| +1,500 | tiled, frame off | 1.33e-3 | 2.70e-3 (1.3e-3) |
| +1,500 | tiled, frame on | 4.09e-6 | 2.97e-5 (1.5e-5) |

At +1,500 Å the plain call reproduces the 149M/320M records' 1.32e-3 eV/Å. With the frame the residual stays at
4e-6 eV/Å wherever the crystal sits (below the origin run's own 1.9e-5), and the rattled forces stay within
3.0e-5 eV/Å of the origin run, under the gate's 3.57e-5 eV/Å float32 rounding basis. The whole-system path at
+200 Å (the coordinates of the A100-80 whole-system limit, about 57³ = 226 Å) carries 4.3x its origin residual, so
it gets the same treatment (below). Before the frame becomes the default: the 3x gate and a not-slower A/B.

### Both paths by size (conductor, 11 November 2026)

Two switches, both `None` = by size: on for a float32 model when the structure's extent (the largest |coordinate| of
the positions and the cell's corners, `forces.structure_extent`) reaches `FLOAT64_EDGES_EXTENT` = 64 Å, where
float32's spacing is twice a 30 Å box's; off for float64 models and smaller structures, so those calls are unchanged.
`True` / `False` force them (A/B and gates).

- **Whole system** (`forces.WHOLE_SYSTEM_FLOAT64_EDGES`, `engine.float64_edge_vectors`): the force field keeps its
  float64 positions (24 B/atom) and cell in the data dict; the bound forward adds a detached correction to MACE's
  edge vectors (the float64 vector less the float32 one, rounded), so the value is the float64 vector's rounding and
  the gradient MACE's. With a strain the vectors are rebuilt from the unstrained positions and strained as edges
  (`v + v @ sym`, bcb93f76): MACE strains the positions, so the stress's gradient sums position x force over the
  atoms, which cancels in float32 far from the origin (the CPU test's 64 atoms at +1,500 Å: 2.1e-4 relative, 4.7x the
  origin call; the tests/cpu suite on 85532876 caught it).
- **Tiles** (`tiled.TILE_LOCAL_FRAME`, `_slot_local_parts`, inside the tile body and captured with it): each tile's
  rows relative to its first receiver, each halo row at one of its edges' periodic images (the smallest encoded unit
  shift, deterministic), in float64 per row; per edge the integer image `round(shift @ cell^-1)` in float32 and one
  float64 `(unit - row unit) @ cell`. Rows and remaining shifts are kept as float32 high and low halves; the vector
  is `(difference + remaining) + (error + low[r] - low[s] + remaining_low)` with the rows' difference exact by
  TwoSum: the float64 vector's rounding.

**Why the halves and the TwoSum** (local jobs 57-63, `idea11-local/`, `jobs/diag_frame.py`). The accuracy runs above
used R 256 tiles. A same-process A/B of receiver caps (`tiled_ab.py --b-receivers`, job 57) showed 1.2e-4 eV/Å
between R 8,192 and R 16,384 at 20³ with the frame on, 3.8e-6 with it off (job 58); the earlier float64-shift frame
(2f7e8a3f) did the same (jobs 59-60). The slot statistics (job 61, b6dc60d6, SrTiO3 perfect, mode padded):

| structure | R | rows up to (Å) | edges with a remaining shift | largest vector error (Å) | largest force (eV/Å) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20³ (78 Å) | 8,192 | 43 | 0 | 2.0e-6 | 2.1e-5 |
| 20³ (78 Å) | 16,384 | 68 | 68,911 (shift 78 Å) | 8.2e-6 | 1.2e-4 |
| 40³ (156 Å) | 16,384 | 57 | 0 | 3.2e-6 | 4.4e-5 |

Rows of up to the tile's extent carry their float32 spacing, rounded differently in each tile (a non-conservative
error, unlike the plain call's static position rounding), and a tile that meets itself through the boundary adds
whole-box remaining shifts. With the halves (job 62, 4206bff3) 40³ R 16,384 fell to 3.1e-6 eV/Å but the self-meeting
tile kept 4.2e-5 (the rows' difference rounds); with the TwoSum (job 63, f8ea447e) every case is at the float64
vectors' rounding: largest vector error 7.6e-8 Å, largest force 2.8e-6 (20³, R 8,192), 3.4e-6 (20³, R 16,384,
71,785 remaining-shift edges), 3.3e-6 (40³, R 16,384).

**Accuracy on the GPU** (`jobs/idea11_gpu2.py`, SrTiO3 6³ at 0 / +200 / +1,500 Å, OMAT-0 medium float32; largest
force on the perfect crystal, rattled forces' largest change from the origin run):

| run (commit) | whole system off | whole system on | tiles, frame on |
| --- | --- | --- | --- |
| local job 51 (935a7473), RTX 5060 Ti | 1.33e-3 / 2.69e-3 at +1,500 Å | 2.2e-6 / 2.6e-6 | 5.0e-6 / 2.8e-5 |
| local job 56 (8dfe40d9), RTX 5060 Ti | 1.33e-3 / 2.70e-3 | 6.9e-6 / 1.9e-6 | 4.7e-6 / 3.0e-5 |
| Alex 4983424 (ace12ea0), A100-SXM4-80GB | 1.33e-3 / 2.70e-3 | 6.8e-6 / 1.8e-6 | 6.3e-6 / 3.3e-5 |
| Alex 4983752 (f8ea447e), A100-SXM4-80GB | 1.33e-3 / 2.70e-3 | 6.6e-6 / 1.9e-6 | 2.5e-6 / 3.7e-6 |

(R 256 tiles; the rattled tiled change is the tiled call's own rounding against the whole-system origin run, as at
the origin.)

**Gate.** `tools/force_stress_gate.py` 3x, both switches forced on (`--production-constant`; the size rule leaves the
gate's cells off), tiled R 48 compiled, OMAT-0 medium, SrTiO3 3x3x3, NaCl, NbCl5: 12/12 passed on ace12ea0 (Alex
4983424) and on f8ea447e (Alex 4983752).

**Speed** (same-process A/B, both orders; where the size rule leaves them off the calls are unchanged):

| path, GPU, size | commit | b/a (ab / ba) |
| --- | --- | ---: |
| whole system, RTX 5060 Ti, 20³ (40,000 atoms), with stress | 935a7473 (job 51) | 1.016 / 1.015 |
| whole system, A100-SXM4-80GB, 40³ (320,000), with stress | 98ca790e (Alex 4983258) | 1.0158 / 1.0155 |
| whole system, A100-SXM4-80GB, 40³, with stress, strain on the edges | ace12ea0 (Alex 4983424) | 1.0146 / 1.0146 |
| frame, RTX 5060 Ti, 40³, R 8,192 budget, compiled | 935a7473 (job 51: float64 GEMMs) | 1.236 / 1.232 |
| | 0c8da3dd (job 53: per tile, eager) | 1.063 / 1.059 |
| | 98ca790e (job 54: padded slot, compiled call) | 1.014 / 1.019 |
| | 2f7e8a3f (job 55: in the body) | 1.020 / 1.027 |
| | 8dfe40d9 (job 56: float32 units, float64 per row) | 1.003 / 1.003 |
| | 4206bff3 (job 62: halves) | 1.022 / 1.018 |
| | f8ea447e (job 63: halves, TwoSum) | 1.004 / 1.010 |
| frame, A100-SXM4-80GB, 58³ (975,560), R 8,192 budget, compiled graphs | 98ca790e (Alex 4983258) | 1.047 / 1.048 |
| | ace12ea0 (Alex 4983424) | 1.029 / 1.029 |
| | f8ea447e (Alex 4983752) | 1.028 / 1.029 |
| frame, RTX 4000 Ada (murphy), 40³ | f8ea447e (`idea11-ada/`) | 1.0078 / 1.0064 |

On the Ada at 975,560 atoms (226 Å, the frame on by size on both sides of a compiled-graphs A/B that a template slip ran first, `idea11-ada/murphy-frameab-*`) the perfect crystal's largest force is 3.0e-6 eV/Å, against the 1.64e-4 of the capacity record without the frame. The tests/cpu suite on f8ea447e (murphy): 5,610 passed, 203 skipped.

The local job pairs drift by about 1 % between runs (job 63's block pairs 0.93-1.05); the 5060 Ti numbers within 2 %
are within that drift. On the A100 the frame was a separate compiled call per tile and sweep (98ca790e), outside
the captured body: 1.047. The conductor's decision (11 November 2026): on by size on every GPU, the A100 included up
to about 3.5 %, since the A100 is the capacity machine and at 64 Å and above the plain forces reach 37x the gate's
rounding basis; a faster frame on the A100 is a follow-up item. The A100 on f8ea447e: 1.028 / 1.029 (block pairs
1.024-1.031), inside that bar. In the same A/B at 58³ (975,560 atoms, 226 Å) the plain call's largest force on the
perfect crystal is 1.68e-4 eV/Å and the frame's difference from it the same (1.69e-4): the frame removes it.
