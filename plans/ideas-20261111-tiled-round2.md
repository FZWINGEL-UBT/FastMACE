# Tiled and wavefront capacity mode, second round of ideas (11 November 2026)

Source: an owner request (11 November 2026) for more speed and capacity ideas, after
[ideas-20261111-tiled.md](ideas-20261111-tiled.md) (ideas 1–8). Read at `fastmace/tiled-speed` `5c890377` and main
`d65fa4b5`: `results/tiled-speed-20261111/README.md`, `results/a100-colab-ideas-20261111/README.md`, the capacity
records under `results/tiled-capacity-20261109/`, and `fastmace/model/tiled.py`, `fastmace/neighbors/tile_list.py`.

**Nothing below is built or measured.** Novelty was checked by keyword search over the ledger, the tiled and
capacity reports, the design notes and the plans.

## What changed since the first list

- **Idea 1** is built: `TILE_HEAP_TRIM`, default. A census at 20.1M atoms confirmed that freed lists stayed
  resident (2.7 of 7.0 GiB). The A100 397.5M rerun is pending.
- **Idea 2:** the A100 region profile exists now. Kernels are 2,013 ns/atom of a 3.8 µs/atom call, so the GPU was
  idle 52 % (Nsight Systems), waiting on the Python side of the compiled tile bodies and on autograd. The two
  interaction-0 forwards are 535 ns/atom.
- **CUDA graphs for the compiled tile bodies** (`compiled_graphs`) on the A100: 0.569 / 0.571, 3.68 to 2.09
  µs/atom against 1.96 for the whole system. Device state only.
- **Idea 3** is rejected at 975,560 atoms: the copy stream and the host worker already hide the host copies. The
  wavefront's cost there is its plan (100 tiles against 68, a pool of 91 % of the atoms).
- **Correction to idea 5.** Its size was overstated. The 329 s search pass in the Ada record belongs to a call
  that rebuilds its lists for a new cap, and that search has to run once in any case. A steady call's pass only
  filters the kept lists: 0.19 µs/atom at 1M, about 3 %. The grid plan saves a full pass only where no lists are
  kept, as in the A100 record (5.8 %).
- **Correction to idea 6 and the first list's last paragraph.** The plain device state is 5.1 kB/atom (4.99 GB at
  975,560 atoms), not 3.45; 3.45 is the aliased host state. The comparison with the whole system's 68.2 kB/atom
  stands.

So the A100's device-state call is now within 7 % of its whole-system call. What is left is the capacity path
(host state, wavefront), kernel work, and memory.

## Ranking

| # | Idea | Helps | Estimated size | First test |
| --- | --- | --- | --- | --- |
| 9 | Captured tile bodies for host state, the wavefront and the aliased device state | A100, H200 capacity path | unknown; device state gained 43 % | Nsight Systems idle share of one A100 wavefront call |
| 10 | The skip input out of the state: 512 columns instead of 640 | pool, device state, copies | state −20 % | CPU test against today's sweeps |
| 11 | Edge vectors in tile-local coordinates | accuracy at capacity scale | residual force 1.3 meV/Å at 149M atoms, 8× the 1M one | CPU: `_tile_vectors` against float64 in a 1,600 Å box |
| 12 | Kept lists as bitmasks over the search stencil | host memory, and calls that drop their lists today | 312 to about 55 B/atom | CPU: bytes and rebuild time on 1M atoms |
| 13 | Spherical harmonics and the edge force inside the edge kernels | whole-system limit, tile workspace | up to 10.7 kB/atom of 68.2 | which of the three edge tensors are live at the peak |
| 14 | Sweep 3 from the kept contraction input where the device has room | A100, H200 speed | 5–13 % of the captured A100 call | A/B at 1M atoms |
| 15 | Host build without the one-hot | A100 host limit after the trim | about 90 of 280–340 B/atom | host census of one build at 20M |
| 16 | Memory measured on the first tiles, not by a whole eager call | first-call time at capacity | A100 record: 11,468 s of 15,381 s | – |
| 17 | Scatter without atomics | speed, mostly on fast GPUs | at most 5.6 % of the A100's kernels | lab of the three `index_add_` forms |

## 9. Captured tile bodies for every state mode

**The evidence.** `compiled_graphs` took the A100's device-state call from 3.68 to 2.09 µs/atom. The config
refuses it with host state, with the aliased device state and with lanes (`tiled.py`: "the captured bodies read
the state"; "the window and the pooled rows are indexed outside the captured bodies"). Capacity runs use exactly
those modes, and the A100's uncaptured wavefront costs 7.04 µs/atom at 1M atoms and 9.84 at 397.5M.

**The idea.** A captured body should read no state at all. Its x, sc, dE/dx and dE/dsc rows come from static
per-slot staging buffers that are filled before the replay: by the prefetch's host-to-device copy with host state,
by a device gather with the device state or its window. `tiled-host-state.md:96` names this need for mode
"graphs"; it was not built.

**Size.** Unknown until the A100's wavefront call is profiled: its host also fetches rows and applies writes,
which a capture does not remove. If the idle share is like the device-state call's 52 %, the call roughly halves.
No gain is expected on the 5060 Ti, whose call is GPU-bound.

## 10. The skip input out of the state

**The idea.** The state holds x = `linear_up(h1)` (512 columns) and sc = `skip_tp(h1)` (128). sc is a scalar
output, so it depends only on h1's 128 scalar channels, and so does x's scalar block. Store those 128 scalars u
once, and x's vector block (384): 512 columns. Sweep 2 forms x's scalar block from u for receivers and halo and sc
for the receivers, two small GEMMs. Its adjoint adds both gradients into u's rows, so sweep 3's objective keeps
its form. The same arithmetic runs one sweep later, so the result stays in today's rounding class.

**Size.** The state shrinks by 20 %: the Ada's pool from 48.8 to 39 GB, the A100's from 63.6 to 51 GiB, the plain
device state from 5.1 to 4.1 kB/atom. sc and its adjoint are also about 16 % of the state's copy traffic (my
count, with a halo of 0.54 R).

**Closest entry.** Adjoint pre-contraction (`tiled-alias.md`): storing all of dE/dh1, rejected because h1 is 2,100
columns wide under the fold. This keeps x for the vector block and narrows only the scalar part.

## 11. Edge vectors in tile-local coordinates

**The evidence.** The records' largest force component on the perfect crystal, which should vanish:

| Atoms | Box side | Largest force component (eV/Å) |
| ---: | ---: | ---: |
| 975,560 | 226 Å | 1.64e-4 |
| 148,955,000 | 1,210 Å | 1.317e-3 |
| 320,000,000 | 1,562 Å | 1.320e-3 |

The ratio is 8.0. float32 spacing is 1.5e-5 Å between 128 and 256 Å and 1.2e-4 Å between 1,024 and 2,048 Å, also
a factor 8. So the residual follows the coordinates' float32 spacing, and at capacity scale it is 37 times the
3.57e-5 eV/Å rounding error the gate is built on. `_tile_vectors` forms `pos[receiver] - pos[sender] + shift`;
whether the positions or the float32 shift vectors carry the rounding is not traced.

**The idea.** Subtract a per-tile origin in float64 before the cast, and build the periodic shift from the unit
shift and the cell in the same frame. Coordinates then stay below about 100 Å in any box.

**Size.** An accuracy fix: the residual falls to that of a 100 Å cell or below. It also removes the need for
whole-system positions on the device (24 B/atom of the Ada's 111.6), since each tile brings its own.

## 12. Kept lists as bitmasks over the search stencil

**The idea.** The kept lists cost 312 B/atom on the host (56 GB on the Ada at 179.7M; 115.9 GiB on the A100 at
397.5M, where they are dropped). The tile search already enumerates each receiver's candidates in a fixed order:
125 bins of half a cutoff (`tile_list.py`, `STENCIL_SPLIT`), about 300 candidates per SrTiO3 atom. One bit per
candidate, set where the pair is in the kept superset, is about 40 B/atom, plus the reference bins (16 B/atom). A
reusing call enumerates the stencil, selects by the bits and runs today's cutoff test; it skips the float64
distance test on every candidate.

**Size.** About 55 instead of 312 B/atom (my estimate from the stencil's volume). Lists that fit are not dropped,
and a call without kept lists is 14–15 % slower (`tiled-host-state-dtype.md`).

**Closest entry.** `tiled-kept-compact`: delta encoding, rejected because it saves a third at most. This is a
different form with a bound of about five times.

**Risk.** Rebuilding from bits costs more than unpacking a stored list (`list reuse` saves 4.3 % per call today).

## 13. Spherical harmonics and the edge force inside the edge kernels

**The idea.** Three per-edge tensors live through most of a whole-system force call
(`results/memory-profile-20261109/`): the spherical harmonics `[E, 16]` (64 B/edge, 4.7 kB/atom) and w′'s outputs
`[E, 16]` and `[E, 7]` (92 B/edge, 6.8 kB/atom), which a separate kernel turns into edge forces at the end.
- Forward: compute the harmonics from the unit vector where the edges are staged. They are l ≤ 3, a few dozen
  operations per edge, against 128 channels of work.
- Backward: after w′'s reduction over the channels, one lane contracts the 16 gradients with the harmonics'
  derivatives and writes the edge force `[E, 3]`.

**Size.** 64 + 92 − 12 = 144 B/edge, 10.7 kB/atom of the 68.2 at the limit, if all three are live at the peak;
that is not traced. The tiles' workspace (57–95 kB per receiver) shrinks by the same amount. It also removes the
A100's DRAM-bound `edge_sh3_backward` kernel (36 µs).

**Closest entry.** `sx-wprime-reduction` applied the Cartesian Jacobian before the channel reduction, in every
thread, and was slower (1.081). After the reduction it runs once per edge.

## 14. Sweep 3 from the kept contraction input where the device has room

`tiled-keep-a0` was rejected because 8.2 kB/atom did not fit beside the tiles on 16 GB. On the A100 the captured
call is GPU-bound and sweep 3's forward is 269 of its 2,013 ns/atom. Up to about 8M atoms the input fits on 80 GB.
Sweep 3 then skips the outer sums and the projection GEMMs, the larger part of that forward: 5–13 % of the call.
The tabled outer-sum backward already recomputes what it needs from the edges.

## 15. Host build without the one-hot

After the heap trim, the build's transient is the A100's next host limit: 280–340 B/atom, MACE's input batch with
float64 positions, a uint8 one-hot of 89 elements and the batch vector (`tiled-host-state-dtype.md:108`). The
tiled path keeps only the species index on the device already. Building the batch from a species index removes
about 90 B/atom of the transient.

## 16. Memory measured on the first tiles

Mode "auto" runs a whole eager call to measure the tile workspace, then compiles. In the A100 record the eager call
took 11,468 s and the compiled one 3,913 s; `compiled_graphs` adds an uncaptured measuring call. A tile's
workspace does not depend on the atom count, so the first few tiles measure it, scaled to the largest slot as the
receiver budget already does. The call can then switch within itself. This only shortens the first calls; for a
single-point capacity run that is most of the time.

## 17. Scatter without atomics

`_scatter` is 112 ns/atom on the A100 and 110 on the 5060 Ti, so it does not scale with the GPU: 5.6 % of the
A100's kernels. Its `index_add_` calls use atomics, yet a tile's dE/dx rows are unique (gather, add, write back)
and the forces' receiver side is sorted (a segment sum). Only the sender side needs atomics.

## Still open from the first list

Ideas 4 (thin wavefront layers), 6 (tiled call instead of the last whole-system rungs), 7 (one-pass tiles) and 8
(device per-atom arrays) have no record yet. Idea 2's target is now known: 535 ns/atom of the A100's 2,013.
