# Tiled path: speed (compiled tiles, the profile, the tile-list cache, the budget default)

Capacity for the tiled path is closed at 70.96M NbCl5 atoms ([tiled-alias.md](tiled-alias.md)); the owner's
priority is now speed. This report covers:

- compiled tiles at scale;
- the budget arms;
- a profile of one call;
- the first two levers it showed, built here: each tile's list once per call, and the budget default for compiled
  tiles.

## Setup

MH-1 float32 hybrid (`mh1-r2scan`), `mace_cueq_oeq`, NVIDIA RTX 4000 Ada (20 GB) on murphy and leto.

- Tiles of R = 4,096, tile lists, the state on the host and aliased, the species index.
- The force field's own global list left empty, as `tiled_capacity.py --tile-list` does.
- Same-process A/Bs: one process per pair, both arms warmed up, then alternating blocks, the order rotated every
  block (`results/tiled-speed-20261101/ab_compiled.py`, `ab_arms.py`).

## Compiled tiles and the budget at 9.26M NbCl5 atoms

murphy, 1a021046 (the arms before the cache).

| Pair | Ratio | Medians | First calls |
| --- | ---: | --- | --- |
| compiled / eager | **0.788** | 179.1 s against 227.4 s | compiled 263.0 s then 180.9 s; eager 232.3 s |
| compiled+budget / compiled | **0.957** | 171.1 s against 178.8 s | budget chose R = 16,384; 179.9, 240.6 (the new slots compile), 170.8 s |
| eager+budget / eager | 0.989 | 228.5 s against 231.0 s | budget chose R = 16,384 |

**Forces gate.** Against the first arm of each pair: energy within 4e-5 meV per atom, forces within 8e-6 eV/Å
(largest force 0.09), stress within 4e-9 eV/Å³.

**Compilation.**

- About 80 s for the first slots at 9.26M (the first compiled call's excess over a steady one).
- About 60 s more when the budget switches to R = 16,384 and its slots compile.
- At 51,840 atoms the first compiled call took 60.6 s.

**Graph breaks and the backward** (`TORCH_LOGS=graph_breaks,recompiles` and Dynamo's counters):

- no graph breaks;
- 9 unique graphs, 3 slots × 3 sweeps;
- AOTAutograd compiled 9 of 9.

So the backward is compiled, not split. The recompiles in the log are the slots' shape specialisations.

## The profile (884,736 NbCl5 atoms, 312 tiles; leto)

**Tool:** `results/tiled-speed-20261101/profile_tiles.py`.

- A timed call synchronises the device at every stage boundary and gives each stage's exclusive time.
- A profiled call, without that synchronisation, gives the device's kernels.

**Stages:**

| Stage | Eager (24.9 s) | Compiled (18.8 s) |
| --- | ---: | ---: |
| interaction 0, sweeps 1 and 3 | 10.04 s (40 %) | 7.14 s (38 %) |
| interaction 1, sweep 2 | 9.92 s (40 %) | 5.82 s (31 %) |
| per-tile neighbour list | 1.91 s (7.7 %), 936 searches | 2.55 s (13.6 %), 1,248 searches |
| host state (reads, writes, window, closing) | 1.56 s (6.3 %) | 1.83 s (9.7 %) |
| scatter, tile graph and rest | 1.46 s | 1.45 s |

**Device:**

| | Eager | Compiled |
| --- | ---: | ---: |
| device time, of the profiled call's wall time | 19.7 of 27.5 s (**72 %**) | 14.2 of 21.5 s (**66 %**) |
| elementwise kernels | 8.5 s | 1.7 s, and 2.0 s of fused Triton kernels |
| GEMMs, gated products and the convolution kernels | about 5.7 s | about 5.6 s |
| host↔device copies | 0.8 s | 0.9 s |

- **Eager** spends 43 % of its device time in unfused elementwise kernels, which compiling removes.
- **Compiled** leaves the device idle a third of the call: host work (the tile lists, the host state's indexing on
  the CPU) and per-tile Python.
- **The per-tile list** was searched in every sweep, and once more for the compiled path's slot planning:
  1,248 searches for 312 tiles. Its float64 distance tests show up as about 0.6-0.8 s of double elementwise
  kernels on the device too.

## Lever (a): each tile's list found once per call

`TileConfig.list_cache` (default on; with `neighbours="tiles"`).

- The first time a call needs a tile's edges, `TileNeighbours.edges` searches them as before.
- The call keeps them on the host in a compact form: the sender (int32), the receivers' edge counts, and the unit
  shifts (int16).
- The later passes rebuild the edge index and the shifts from that form, the shifts with the list's own
  `TileNeighbours.shifts` (unit shifts in the float dtype, `times_3x3` with the float64 cell, element-wise per
  row).
- A tile whose unit shifts do not fit int16 is not cached.
- The cache lives for one call.

**CPU tests** (`test_tiled.py`, `test_layering.py`, `test_neighbors.py`, 148 passed):

- the call with the cache is bitwise the call that searches every sweep, eager and padded, with stress;
- the edges are searched once per tile per call, against 3 times eager and 4 times padded.

**Same-process A/B** (leto, c47598b1; the cache on against off; the budget off in both arms):

| Pair | Cache on / off | Medians | Cache (host) |
| --- | ---: | --- | ---: |
| compiled, 884,736 atoms | **0.920** | 17.16 s against 18.65 s | 240 MB |
| eager, 884,736 atoms | 0.972 | 23.95 s against 24.64 s | 240 MB |
| compiled, 9.26M atoms | **0.915** | 167.2 s against 182.7 s | 2.51 GB |

- The cache costs 271 bytes per atom on the host, about 19 GB at 70.96M NbCl5 atoms (NaCl, with more edges, about
  330 bytes).
- At the host limit of [tiled-alias.md](tiled-alias.md) that is 207 + 19 GiB for NbCl5. The scan guard needs
  about 0.3 kB per atom more, so the cache can be turned off (`list_cache=False`) for a run at the very limit.

## Lever (b): the budget by default for compiled tiles

`TileConfig.budget_receivers` defaults to (8192, 16384) with `mode="compiled"` and to none otherwise.

- An explicit value, `()` included, is kept.
- The budget's rule is unchanged: the measured workspace must fit, so at the limit the cap stays at 4,096.
- For eager tiles the budget gains 1 %, too little for a default.
- `tools/bench/tiled_md.py`'s compiled runs now take the budget too. They may record more slot changes than
  before.

## Together, at 9.26M NbCl5 atoms

From the same-process ratios: compiled (0.788) × budget (0.957) × cache (0.915) ≈ **0.69 of the eager call**,
about 157 s against 227 s per call.

## Compiled tiles at the limit

murphy, 3ac65bfb, before the cache: NbCl5 at 70.96M atoms, `--mode compiled`, two calls.

- **The first call:** 1,540 s with compilation (eager 1,937 s), GPU peak 18.47 GiB.
- **The second call** ran out of device memory allocating 1.08 GiB, with 0.83 GiB free. Compiled tiles keep their
  slot buffers and compiled workspaces between calls.

So compiled tiles fit a little below the eager limit. Making `mode="compiled"` the default needs a fallback to
eager at the limit, or a memory rule.

## Next

- (c) The compiled default, with the conductor's gates:
  - an NVE drift check at 884,736 atoms, compiled against eager, 200 steps;
  - the forces gate at the largest compiled size that holds two calls;
  - a rule or fallback for the limit.
- The host state's CPU indexing and the per-tile Python overhead, next by the profile (compiled: 10 % host
  state, and the device idle a third of the call). Overlapping the host-state traffic with the device's work is
  the candidate; profile again after the cache first.
