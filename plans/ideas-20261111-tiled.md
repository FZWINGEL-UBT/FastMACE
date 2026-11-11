# Tiled and wavefront capacity mode: speed and memory ideas (11 November 2026)

Source: an owner request (11 November 2026) to close the tiled call's per-atom speed gap to the whole-system call
and to find more memory savings. Read: branch `fastmace/tiled-speed` at `96f4aa39` (the newest tiled results),
branch `fastmace/edge-index-int32` at `8ff6f43c`, main `d7dd7e23`; `results/tiled-speed-20261108/` and
`-20261110/`, the records under `results/tiled-capacity-20261109/`, `results/rung-b-20261109/`,
`results/memory-profile-20261109/`, `docs/validation/tiled/`, `docs/design/tiled-*.md`, and the code in
`fastmace/model/tiled.py`, `fastmace/rewrites/mh1/modules.py` and `fastmace/rewrites/generic/modules.py`.

**Nothing below is built or measured.** Two read-only subagents collected the numbers; I checked the ones each idea
rests on in the files named. Novelty was checked by keyword search over the tiled reports, the design notes, the
ledger and `tiled.py`, not by a full read. The companion list for the whole-system kernels is
[ideas-20261111.md](ideas-20261111.md).

## Where the gap is

OMAT-0 medium, SrTiO3, µs per atom per force call (975,560 atoms for the tiled columns):

| GPU | Whole system | Tiled, device state | Wavefront (host state) | Capacity record |
| --- | ---: | ---: | ---: | --- |
| RTX 5060 Ti | 4.79 (29,160 atoms) | 6.27–6.42 (1.3×) | 7.42 (1.55×) | – |
| RTX 4000 Ada | not on file at this size | 5.64 | 6.20 | 9.33 at 179.7M atoms |
| A100-SXM4-80GB | 1.96 (29,160 atoms) | 3.73 (1.9×) | 7.04 (3.6×) | 9.84 at 397.5M atoms |

What the profiles say:

- **5060 Ti, device state** (`results/tiled-speed-20261108/profiles/job17/kprof4-58.json`, 6,168 ns/atom): backward
  of sweeps 2 and 3 2,394, sweep 2 forward 1,648, sweep 3 forward 798, sweep 1 forward 777, tile graph 136, scatter
  114, other glue about 300. The whole-system call is 4,718. The gap is sweep 3's second interaction-0 forward and
  about 650 of glue.
- **5060 Ti, wavefront** (`results/tiled-speed-20261110/profiles/job36/wprof-58.json`): 7.24 s wall, GPU busy
  7.15 s. Its gap to device state (about 1,050 ns/atom) is the state's copies: host to device 576, device to host
  455, device to device 54.
- **A100:** no region profile is on file. Two things are confirmed in the code and the records (ideas 2 and 3).
- **Capacity records** are each run's second call: the first compiled call, which also measures memory, compiles,
  rebuilds the lists and has no graph cache. `results/tiled-scale-20261103/README.md` already notes that second
  calls are not steady. The Ada's 9.33 µs/atom is search pass 1.83 + tiles 7.46; at 1M it is 0.19 + 5.98.

## Ranking

| # | Idea | Helps | Estimated size | First test |
| --- | --- | --- | --- | --- |
| 1 | Give the dropped lists' host memory back, and do not fill lists that will be dropped | A100 capacity and validity | about 110 GiB of host at the 397.5M record | host census before and after a trim |
| 2 | Composed interaction 0 and generated contraction in the tile bodies | A100, H200 speed | 8–20 % of the A100's tiled call | A100 region profile of one tiled call |
| 3 | Wavefront with the row pool on the device | wavefront speed, host memory | 5060 Ti 7–14 %, A100 about 40 % of the wavefront call | none needed for the bound; A/B after the build |
| 4 | Thin wavefront layers | host pool, and room for idea 3 | pool −50 % for about 12 % more halo rows | CPU: pool rows against the grid shape |
| 5 | Wavefront plan from the tile grid, no search pass | capacity-scale speed | Ada record −19.6 %, A100 record −5.8 % | CPU test of the plan against the search pass's |
| 6 | Tiled call instead of the last whole-system rungs | standard mode near its limit | 10–18 % per atom, by unlike measurements | same-process A/B at the limit size |
| 7 | One-pass tiles with a two-cutoff halo | large-memory GPUs; no state at all | 1.15–1.21× the whole system instead of 1.3× (model) | CPU exactness test, then a 65,536-receiver lab |
| 8 | Device per-atom arrays on a diet | Ada capacity | 179.7M to about 240M atoms | device census of per-atom tensors at 20M |

## 1. Give the dropped lists' host memory back

**What the A100 record shows** (`results/tiled-capacity-20261109/alex/cap-4973640/`, 397,535,000 atoms, cgroup limit
231.5 GB):

- Call 1 (eager) filled the kept lists for all 92,000 tiles, then dropped them at the last tile: "the call's dtype
  after dropping the kept lists (115.9 GiB)".
- The sweeps then held 63.6 GiB of state (`global_bytes`), yet `mem.log` shows the cgroup flat at 211,593 MiB
  (221.9 GB) for the whole call and `host_rss` is 197.3 GiB at its end. About 110 GiB is unaccounted for, the size
  of the dropped lists.
- Call 2 (compiled) therefore chose float16 state: "81.3 GiB of state in the call's dtype against 58.2 GiB above
  the floor". The list cache went off too: "16.8 GiB available on the host while the list cache filled".

So float16 state (energy off by 0.54 meV/atom there; the gate fails 12–51× with it) and the missing list cache were
both forced by memory that was no longer in use. Whether the allocator kept the freed pages or a reference kept
the lists alive is not traced. The 10 November fix (`3899b8b0`) returns pinned blocks only.

**The idea.**
- After `_release_kept`, hand the pages back (`malloc_trim(0)`), or hold the lists in blocks the allocator maps
  and unmaps one by one. `MALLOC_MMAP_THRESHOLD_` at 1 MiB already cut the process peak by 7 % at 10M atoms, but as
  an environment setting.
- Decide before filling: the lists need about 4.2 B per edge (115.9 GiB here), which the host budget can compare
  with its room before the first tile. That also saves filling 92,000 tiles' lists for nothing.

**Size.** At 397.5M the A100 keeps float32 state with about 100 GiB to spare. The host limit then moves towards
700M atoms, where the build's host transient (279 B/atom) and the device's per-atom arrays (110 B/atom) both
come close to their limits.

**First test.** `tools/bench/tiled_host_profile.py` at 20–30M atoms with the lists forced to drop: resident memory
before and after a trim. If nothing comes back, look for the reference instead.

## 2. Composed interaction 0 and generated contraction in the tile bodies (A100, H200)

**What is confirmed.** The A100's whole-system plan runs interaction 0 through `COMPOSE_SKIP`, `COMPOSE_FAST`,
`COUNT_PLAN`, `JOINED_LAYOUT` and `SC_GENERATED`. The tile bodies skip all five:
- `mh1/modules.py:251`, `:268`, `:270` take the composed and joined paths only when `rows is None`, and a tile
  passes its receiver count as `rows`;
- `generic/modules.py:432-435` runs the composed division only with no tiles;
- `SC_GENERATED` needs the composed path's `sc` slot (`sc_generated.py:10-13`).

A tile therefore runs the per-path GEMMs and their concatenation, and it runs interaction 0's forward twice
(sweeps 1 and 3). The tiles' interaction 1 does keep the A100's spline path. The 5060 Ti's whole-system plan has
none of the five, which is why its tiles match its whole system kernel for kernel and its gap is only 1.3×.

**Size.** The whole-system A/Bs on the A100: `COMPOSE_SKIP` 0.8027 / 0.9546, `COMPOSE_FAST` off again
1.0240 / 1.0235, `COUNT_PLAN` off again 1.0115 / 1.0130, `JOINED_LAYOUT` 0.9782 / 0.9784, `SC_GENERATED`
0.9685 / 0.9652. Taken together that is 13–27 % of the whole-system step, so roughly 0.3–0.7 µs/atom of the tiled
call's 3.73 (8–20 %). An A100 region profile would replace this estimate; none exists.

**Also there:** the tiles still compose the weights for every tile in `bind` and never read them
(`mh1/modules.py:175-210`).

**First test.** A region profile of one tiled call on the A100 (Alex), as `kprof4.py` on the 5060 Ti.

## 3. Wavefront with the row pool on the device

**The idea.** The wavefront keeps x, sc and their adjoints only for the atoms between their sweeps 1 and 3, in a
pool of rows. That pool is on the host today: `_State.rows` reads a device tensor by atom index, and the pooled
row lookup (`_rows_of`, `assign`, `release`) runs on the host. With the row map on the device (4 B/atom), the pool
can live there whenever it fits, and the host copies go away.

**Why it matters.** On the 5060 Ti the wavefront's whole gap to device state is those copies (1,085 ns/atom of
7,420). On the A100 the wavefront takes 7.04 µs/atom against 3.73 on device state.

**Where it fits.** The pool is 2.6 kB per row. The A100 record left 41 GiB free on the device ("41.15 GiB left at
the first call's peak") while the pool took 63.6 GiB on the host, so at 397.5M it fits only with idea 4; below
about 230M atoms it fits as it is. On the Ada and the 5060 Ti, device state holds 3–5M atoms today; a device pool
reaches roughly 16M and 7M (my estimate from the free memory and the band), and about twice that with idea 4.

**Size.** The wavefront call drops to the device-state call plus its search pass: 5060 Ti 7.42 to about 6.4–6.9
µs/atom, A100 7.04 to about 4.

## 4. Thin wavefront layers

**The idea.** The pool holds about three layers of the tile grid (`docs/design/tiled-wavefront.md`), and
`_grid_shape` makes the blocks about cubic. Blocks that are thinner along the wavefront's axis and wider across it
hold the same receivers but shrink the band in proportion.

**Evidence on file.** The A100 record's pool grew from 26.35M rows at R = 8,192 to 33.06M at R = 16,384: a factor
1.255, the cube root of 2. The pool follows the layer thickness.

**Size** (geometry model for SrTiO3; it overstates the measured halo by about a third, so read the ratios):

| Layer thickness | Pool | Halo rows per receiver |
| --- | ---: | ---: |
| cubic (58 Å at R = 16,384) | 1 | 0.73 |
| half (29 Å) | 0.5 | 0.82 |
| quarter (14.5 Å) | 0.25 | 1.18 |

At half thickness the Ada's pool falls from 48.8 to about 24 GB and the A100's from 63.6 to about 32 GiB, which
then fits on its device (idea 3). The cost is the halo's state traffic: x is read and dE/dx written for the halo
rows. The halo rows' node work is not where the time goes (`tiled-1m-speed.md`).

**First test.** CPU: `plan_tiles(shape=...)` on a 20M-atom lattice with thinner layers, and the pool rows and
largest halo that `wavefront_schedule` reports.

## 5. Wavefront plan from the tile grid, no search pass

**The idea.** Before its sweeps the wavefront builds every tile's graph once to learn which tiles each tile reads
and each atom's first and last tile (`_wavefront`). Without a graph cache nothing of that pass is kept. The same
plan follows from the grid: a tile reads the tiles of its own block and the 26 around it, since a block is wider
than the reach. That is a superset, so sweep 1 runs no later and sweep 3 no earlier than they must, and the pool
grows by at most the tiles that had no real halo atoms. The padded slots' sizes are then taken at each tile's
first build (the `TILE_LAZY_KEYS` code path).

**Size.** `search_s` is 329.1 of the Ada record's 1,676 s (19.6 %) and 226.8 of the A100 record's 3,913 s (5.8 %).
The graph cache cannot hold a capacity run (20 B/edge), so the pass is pure overhead there.

**Closest entry.** `tiled-lazy-keys`, rejected at 1M atoms on device state (1.023 / 1.002), where the pass's
graphs are kept and reused. At capacity they are not.

**First test.** CPU: the grid's plan against the search pass's on the wavefront tests' ring, same forces.

## 6. Tiled call instead of the last whole-system rungs

**The observation.** Near its limit the whole-system call costs more per atom than the tiled call:

| GPU | Whole system at its limit (MD step) | Tiled, device state, 975,560 atoms (force call) |
| --- | --- | --- |
| RTX 5060 Ti | 214,375 atoms, 1,525.9 ms: 7.12 µs/atom | 6.27–6.42 |
| RTX 4000 Ada | 253,265 atoms, 1,594 ms (table rung) or 1,704 ms (layer rung): 6.29 / 6.73 µs/atom | 5.64 |
| RTX 4000 Ada | 296,595 atoms, 2,042.8 ms: 6.89 µs/atom | 5.64 |

The whole-system 5060 Ti call is 4.79 µs/atom at 29,160 atoms, so its own per-atom cost rises by half on the way
to the limit. The tiled call needs 3.45 kB/atom of state and about 1 GiB per tile, far below 16 GB at these sizes.

**The idea.** Treat the tiled call with device state as one more rung, chosen where it is faster than the next
recompute rung, not only where the whole system no longer fits.

**Caveat.** The left column is an MD step (integrator, noise, list rebuilds); the right one a steady force call.

**First test.** Same process, both orders, at 214,375 atoms (5060 Ti) and 253,265 (Ada): the plan the planner
picks there against `TiledForceField` on device state.

## 7. One-pass tiles with a two-cutoff halo

**The idea.** A tile whose halo reaches two cutoffs can run both interactions and one backward in a single body:
interaction 0 on its receivers and the first shell, interaction 1 on its receivers, forces scattered to every
node. There is no state, no pool, no host copy, no second interaction-0 forward and one graph per tile instead of
three bodies. Each site energy is still computed once, so the result is in today's rounding class.

**Why not before.** The design assumed tiles of 8,192 receivers, where the halo is about 0.7 R and a second shell
would double interaction 0. Tile workspace has since fallen from about 505 to 57–95 kB per receiver, so far
larger tiles fit.

**Model** (5060 Ti profile: interaction 1 forward and backward 3,249 ns/atom, interaction 0 forward plus backward
1,755; first-shell ratio h scaled from the measured 0.54 at 16,384 receivers):

| Receivers per tile | h | One pass, ns/atom | Against the whole system (4,718) |
| ---: | ---: | ---: | ---: |
| 16,384 | 0.54 | about 6,080 | 1.29× (three sweeps today: 1.31×) |
| 65,536 | 0.34 | about 5,730 | 1.21× |
| 262,144 | 0.21 | about 5,420 | 1.15× |

It pays where memory allows large tiles: the A100 and H200, and the Ada up to about 65,536 receivers. For capacity
runs the gain is larger than the table shows, because the state copies and the pool disappear as well.

**First test.** CPU: exactness against the whole-system call in float64 with the tiny model, as
`tests/cpu/test_tiled.py`. Then a lab at 65,536 receivers on the Ada. Stop if it is not faster than three sweeps
at the same cap.

## 8. Device per-atom arrays on a diet (Ada)

**What binds the Ada.** At 179.7M atoms the device peak is 18.67 GiB, 111.6 B/atom, almost all of it per-atom
arrays; the tile workspace is under 1 GiB. The 196.5M run failed in `plan_tiles`' stable sort (`tiled.py:774`), a
2.26 GiB request with 17.84 GiB allocated.

**The idea.**
- The plan's order without a device sort: the atoms per block are already counted, so the order can be placed by
  counting, in chunks, as int32. That removes the failing request (about 12 B/atom).
- Tiles as int32 index tensors: 8 to 4 B/atom.
- The float64 positions on the device: with `TILE_FILTER32` only the edges near the cutoff need them, so they can
  stay on the host and be fetched for those edges. Up to 24 B/atom; which float64 copies are live at the peak is
  only partly traced.
- Positions and forces on the host is a design note already (`docs/design/tiled-wavefront.md:111-112`).

**Size.** From 111.6 to about 84 B/atom the Ada reaches about 240M atoms, if the float64 copy is live at the peak;
about 185M from the sort and the int32 tiles alone. Its host (129 GB of 251 GiB at 179.7M) has the room.

**First test.** A census of the device's per-atom tensors at the peak of a 20M-atom call.

## Re-screens and small items

- **Receiver caps above 16,384 on the A100 and the Ada.** Rejected on the 5060 Ti (32,768 inside the drift, 65,536
  3.5 % slower) and not screened elsewhere (`tiled-receiver-cap`). A tile takes about 1 GiB of the A100's 80, and
  fixed per-tile work weighs more on a GPU that is 2.4× faster.
- **`tiled-keep-a0` on the A100.** Rejected because 8 kB/atom did not fit beside the tiles on 16 GB. Up to about 8M
  atoms it fits on 80 GB, and sweep 3's forward is 13 % of the 5060 Ti's tiled call. Idea 7 makes it moot.
- **`grid_fit` at capacity.** The records' tiles hold 8,915 (Ada) and 10,620 (A100) receivers against a cap of
  16,384. `grid_fit` (off by default) cut the tiles by 29 % and the steady call by 2.1 % at 20M atoms.
- **Dead work in interaction 0 on tiles:** `node_embedding` and `linear_up` run on receivers and halo, but the
  reassociation reads the species table. Whether Inductor removes them is not checked.
- **Capacity records:** report a third call as well, so the steady cost is on file.

## Whole-system memory: nothing new beyond idea 6

The peak at the limit is spread out (`results/rung-b-20261109/README.md`, branch `fastmace/edge-index-int32`):
the planned capture arena (10.9 kB/atom, PROPOSED), `edge_index` in int32 (0.8 kB/atom, in progress) and packed
shifts (not taken) are already on the list, and chunking the backward's GEMM gradients is rejected at 4.4 kB/atom.
I found no further exact saving there. Past a few hundred thousand atoms the tiled call is the lever: 3.45 kB/atom
of state against 68.2 kB/atom.
