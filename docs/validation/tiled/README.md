# Tiled-path reports

The tiled (capacity) path and its wavefront mode, 30 October – 4 November 2026 (design: [tiled-backend](../../design/tiled-backend.md), [tiled-wavefront](../../design/tiled-wavefront.md)).

Columns: file, date added, title, and the report's own verdict line or first sentence.

| File | Added | Title | Verdict / first sentence |
| --- | --- | --- | --- |
| [tiled-1m-speed](tiled-1m-speed.md) | 2026-11-03 | Tiled call at one million atoms: where the time goes (3 November 2026) | Owner priority from 3 November 2026: no bigger cells on one GPU; make the tiled call as fast as possible at about 1M atoms. |
| [tiled-alias](tiled-alias.md) | 2026-10-31 | Tiled capacity path: the host state aliased | The neighbour-list session's proposal, through the conductor, after tiled-host-onehot.md. |
| [tiled-build](tiled-build.md) | 2026-10-31 | Tiled capacity path: the build without whole-system intermediates | After tiled-host-state.md, the force field's construction limited the tiled path at 7.76M atoms on 20 GB, at about 2.2-2.4 kB per atom. |
| [tiled-cache-compact](tiled-cache-compact.md) | 2026-11-01 | Tiled path: the compact list cache, MD at 70.96M with the cache on, and the MD proof plot | The 70.96M NVE gate run of tiled-md-host.md had the list cache off. |
| [tiled-call-memory](tiled-call-memory.md) | 2026-11-01 | Tiled path: the call's whole-system memory, the MD limit, and the call's prelude | tiled-compiled-default.md found that a second call needs about 2 GiB more than the first. |
| [tiled-compiled-default](tiled-compiled-default.md) | 2026-11-01 | Tiled path: compiled tiles as the default (the conductor's item (c)) | The gates the conductor set after tiled-speed.md: |
| [tiled-compiled](tiled-compiled.md) | 2026-10-30 | Tiled capacity path: compiled tile bodies | The conductor's next step of 31 October 2026, after T1's graph falsifier (capacity-tiled-t1t2.md). |
| [tiled-concurrent-tiles](tiled-concurrent-tiles.md) | 2026-11-02 | Two tiles at once: stages 0 and 1 (2 November 2026) | Plan: `plans/tiled-concurrent-tiles.md`. |
| [tiled-host-memory](tiled-host-memory.md) | 2026-10-31 | Tiled capacity path: host memory at the call | After tiled-species-index.md, host memory set the tiled path's limit on the workstations (250 GB each). |
| [tiled-host-onehot](tiled-host-onehot.md) | 2026-10-31 | Tiled capacity path: the construction with its one-hot on the host | Lever (2) after tiled-host-memory.md. |
| [tiled-host-state](tiled-host-state.md) | 2026-10-31 | Tiled capacity path: the tiled state on the host (T3 part 2) | After tiled-table-rows.md, the GPU limit was the tiled call's global state: x, sc and their adjoints, 4.75 kB of the call's 5.9 kB per atom. |
| [tiled-leftovers](tiled-leftovers.md) | 2026-11-01 | Tiled path: setup leftovers, the compiled MD limit, the A/B against c8cb1a7f, step times | This report covers the four items the conductor ordered after merging tiled-call-memory (85195b39). |
| [tiled-list-reuse](tiled-list-reuse.md) | 2026-11-01 | Tiled path: tile lists kept across MD steps (`TileConfig.list_reuse`) | What it is. Every tiled call bins all atoms, plans the tiles, and searches each tile's neighbour list once; the list cache serves the later sweeps. |
| [tiled-md-host](tiled-md-host.md) | 2026-11-01 | Tiled path: MD at 70.96M with the integrator's state on the host | Until now, `tools/bench/tiled_md.py` kept the integrator's state on the device: the positions, velocities, forces and masses, as float64. |
| [tiled-omat](tiled-omat.md) | 2026-11-02 | Tiled path with MACE-OMAT-0 medium: SrTiO3 capacity on 20 GB | The owner asked for the tiled path with a plain residual-block model, MACE-OMAT-0 medium (`omat0-medium`). |
| [tiled-onehot](tiled-onehot.md) | 2026-10-31 | Tiled capacity path: one copy of the one-hot species | Step (a) after the build snapshot in tiled-build.md. |
| [tiled-overlap](tiled-overlap.md) | 2026-11-01 | Tiled path: the host-state writes overlapped with the next tile | With the state on the host, every tile copies its results to the host and the host indexes them into the state before the next tile is queued. |
| [tiled-padding](tiled-padding.md) | 2026-11-01 | Tiled path: how much of a padded call is padding | Answer: no. At production sizes the recoverable share is about 2-3 % of the edge work and at most about 1 % from |
| [tiled-r-budget](tiled-r-budget.md) | 2026-10-31 | Tiled call: the receiver cap from the workspace budget | The conductor's decision after the R × bucket screen (tiled-screen.md): R is not fixed at 16,384 but driven by a budget. |
| [tiled-rewrites-1m](tiled-rewrites-1m.md) | 2026-11-09 | Tiled call at one million atoms: whole-system rewrites in the tile bodies (8-9 November 2026) | Verdict: eight switches bring the tiled call with the state on the device from 8.83 to 6.42 µs/atom at 975,560 SrTiO3 atoms on the RTX 5060 Ti (A/B 0.729 / 0.731): 1.34× the whole-system call per atom, was 1.83×. |
| [tiled-screen](tiled-screen.md) | 2026-10-31 | Tiled call: tile size and bucket screen (ninth idea list, item 3) | The compiled tiled call over R ∈ {4,096, 8,192, 16,384} × bucket ∈ {0.15, 0.05}, against the current call (R = 4,096, bucket 0.15). |
| [tiled-slot-growth](tiled-slot-growth.md) | 2026-11-01 | Tiled path: padded slots that settle during MD | Compiled tiles run in padded slots of fixed receiver (R), halo (H) and edge (E) counts. |
| [tiled-species-index](tiled-species-index.md) | 2026-10-31 | Tiled capacity path: the species index instead of the one-hot | Step (b) after tiled-onehot.md, in the conductor's scope: |
| [tiled-speed](tiled-speed.md) | 2026-10-31 | Tiled path: speed (compiled tiles, the profile, the tile-list cache, the budget default) | Capacity for the tiled path is closed at 70.96M NbCl5 atoms (tiled-alias.md); the owner's priority is now speed. |
| [tiled-t3](tiled-t3.md) | 2026-10-31 | Tiled capacity path: T3 (per-tile neighbour lists) | The first part of T3 in plans/capacity-ideas-20261030.md, through the conductor: each tile builds its own edges, and the force field's global list is no longer built. |
| [tiled-table-rows](tiled-table-rows.md) | 2026-10-31 | Tiled capacity path: species tables at the call's row count | The conductor's option (a) after tiled-t3.md. |
| [tiled-wavefront](tiled-wavefront.md) | 2026-11-02 | Tiled path above 78M atoms: wavefront schedule and device trims | Owner decision (2 November 2026): "Go ahead and try to go as big as possible now". |
