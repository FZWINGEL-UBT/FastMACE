# Tiled call at one million atoms: where the time goes (3 November 2026)

Owner priority from 3 November 2026: no bigger cells on one GPU; make the tiled call as fast as possible at about
1M atoms. This page measures where a 1M call spends its time and which levers are left.

**Setup.**
- **Cell and model:** SrTiO3 58×58×58 (975,560 atoms, 72,191,440 edges in the cutoff, 74 per atom), OMAT-0
  medium, float32, `--tile-list --species-index`, compiled mode, steady third call.
- **GPU:** murphy, RTX 4000 Ada 20 GB, environment `mace_cueq_oeq`.
- **Source:** `b64cf7ba` (profiles) and `95a6f879`/`e7a1a06a` (the new bench options `--sweep-times` and
  `--sweep-profile`, otherwise the same library code) on branch `fastmace/tiled-band-lists`.
- **Records:** `results/tiled-1m-20261103/`.

## Result

The tiled call at 1M runs at about 15 µs per atom (14.5–14.7 s per call), with the GPU busy 97.8 % of the time.
Almost all of that is the model's own per-edge and per-receiver work. Tile size does not change it, and launch or
host overhead is about 2 %. The one tiled-only cost of note is sweep 1, about 10 % of the call.

## Whole-call profile

`--profile-last`, R = 16,384 (68 tiles, the default budget's choice with the state on the device).

| Kernel class | Share |
| --- | --- |
| Convolution (`oeq_projected_backward` 19 %, `oeq_projected_forward` 9 %) | 31 % |
| GEMMs (float32 SIMT) | 20 % |
| Pointwise | 14 % |
| cuEquivariance symmetric contraction | 13 % |
| Copies | 10 % |
| Index | 6 % |

The host spends its time in `cudaMemcpyAsync` waiting for the GPU (11.1 s of self time). So capturing the bodies in
CUDA graphs can remove at most the roughly 2 % of the span when the GPU is idle.

The local RTX 5060 Ti profile (`rtx5060ti-local/`) was taken at R = 8,192, 138 tiles, with both state placements.

## Per sweep

`--sweep-times`: CUDA events around each tile's sweep body, fill and scatter, on the last call.

| Part | State on the device | State on the host (aliased) |
| --- | --- | --- |
| Sweep 1 (interaction 0 forward) | 1.46 s | 1.57 s |
| Sweep 2 (interaction 1 forward and backward) | 8.71 s | 8.85 s |
| Sweep 3 (interaction 0 with autograd) | 3.24 s | 3.34 s |
| Fill and scatter, all sweeps | 0.23 s | 0.40 s |
| Call | 14.5 s | 14.6 s |

`--sweep-profile`: torch.profiler around tiles 4–6 of each sweep. Per tile at R = 16,384, sweep 2 takes about
140 ms:

| Sweep 2 kernel | Share |
| --- | --- |
| `oeq_projected_backward` | 31 % |
| cuEquivariance `kernel` (symmetric contraction) | 15 % |
| `oeq_projected_forward` | 15 % |
| `ampere_sgemm_64x64_tn` | 14 % |
| Other GEMMs | 7 % |
| Slice-backward fusions (halo rows) | about 3 % |

Sweeps 1 and 3 take about 23 and 51 ms per tile. They are mostly GEMMs, the contraction, and copies and
concatenations around the reassociated first interaction.

## Tile size

`--sweep-times --no-budget`: the receiver cap fixed.

| Cap | State | Tiles | Largest halo | GPU peak | Compile call | Steady call |
| --- | --- | --- | --- | --- | --- | --- |
| 16,384 | device | 68 | 8,920 | 12.4 GiB | 76 s | 14.6 s |
| 24,576 | device | 54 | 11,959 | 16.2 GiB | 282 s | 14.6 s |
| 32,768 | device | out of memory | | | | |
| 32,768 | host | 54 | 14,054 | 15.0 GiB | 195 s | 14.7 s |
| 49,152 | host | 27 | 15,134 | 17.2 GiB | 109 s | 15.6 s |

Larger tiles have a smaller halo fraction but are not faster. The sweep times stay the same within 1 %, and at
49,152 they get slightly worse. The halo rows' node work is not where the time goes. The default budget's
R = 16,384 is the right choice at 1M, and it also has the shortest compile.

## What this leaves

- **Sweep-2 conv on the real rows only:** worth about the 3 % of slice-backward fusions. Done: see "Trims".
- **Sweep 3 without interaction 0's forward:** at most the forward part of sweep 3 (about sweep 1's 1.5 s, 10 %).
  Its backward needs the forward's saved activations, though: per-edge tensors, estimated (not measured) at about 30 GB at 1M, which would
  have to round-trip over PCIe each call. Storing only u does not remove the recompute. Not pursued.
- **CUDA graphs for the bodies:** at most the 2 % idle span.
- **The remaining 85 %:** the same OEq convolution, contraction and GEMM kernels as the whole-system engine, which
  runs at 12.5–14.1 µs per atom on this GPU up to about 69k atoms. They have been screened at length
  (`docs/optimisation-ledger.md`: 3×TF32 and 3×FP16 GEMMs, own convolutions, receiver backward and others). Gains
  there are general FastMACE kernel work, not tiled-path work.

## Trims (`aa5dbbfc`, 3 November 2026)

Two of the small items above, measured as an A/B against main `7e3e8cd5` on murphy (same setup, R = 16,384,
`--no-budget --sweep-times`, order base, trim, trim, base; records `results/tiled-1m-20261103/rtx4000ada-murphy/
murphy-trims-*`).

- **Sweep-2 convolution on the real rows.** `streaming_message(rows=n_real)` builds `row_ptr` and the message for
  the tile's receivers only; the senders still index every node. The vendored kernels are unchanged: the launch must
  cover the rows exactly (`StreamingKernels.rows_fit`: rows × channels a multiple of the block size; true for
  128 channels), otherwise every node's rows are built as before. This applies to the plain and the folded streaming
  convolution, not to the fused forward, w-table or w′ paths. The generic density scatter is also sized to the real
  rows.
- **No density in residual radial tables.** `radial_table(density=False)`: the density is `[E, 0]`, not stored, and
  the backward reads no zero gradient. The load stays in the kernel with a run-time count of 0: a compile-time
  branch let the compiler reorder the sum (last-bit differences), the masked load keeps it bit for bit.
- **Not done: a species index instead of the one-hot.** Building and reading the `[R + H, 89]` one-hot costs about
  20 µs per tile and sweep, about 4 ms of the 14.5 s call, and an index would need a side channel past MACE's
  forward signature.

| Run | Steady call | Sweep 1 | Sweep 2 | Sweep 3 | GPU peak | Energy (eV) |
| --- | --- | --- | --- | --- | --- | --- |
| base a | 14.49 s | 1.463 | 8.716 | 3.234 | 12.41 GiB | -7,794,285.0090 |
| trim a | 14.33 s | 1.464 | 8.541 | 3.251 | 12.23 GiB | -7,794,285.0094 |
| trim b | 14.35 s | 1.464 | 8.561 | 3.249 | 12.23 GiB | -7,794,285.0085 |
| base b | 14.58 s | 1.466 | 8.778 | 3.248 | 12.41 GiB | -7,794,285.0083 |

The call is 0.985 of main's (sweep 2 0.978), the GPU peak 0.18 GiB lower and the eager first call about 15 % shorter
(26–27 s against 30–31 s). The energies differ by the same 1 meV as two runs of main (atomic sums). The GPU tests
(`tests/gpu/test_kernels_oeq_streaming_cuda.py` with the new real-rows test, `tests/gpu/test_radial_table_cuda.py`
with the density test) pass on murphy and the local RTX 5060 Ti; the targeted CPU tests (radial tables, generic and
MH-1 rewrites, tiled, streaming kernels, layering, w′ backward) pass on leto.

## Contiguous reassociation layout: rejected (`e2903496`, 3 November 2026)

The idea: the reassociated first interaction slices its per-slot sums `[nodes, species, (l, m), k]` per path and
transposes them, so each path's GEMM input needs a copy (sweeps 1 and 3). `e2903496` (branch
`fastmace/reassoc-layout`, not merged) let `edge_outer_sum` write `[nodes, (l, m), species, k]` instead
(`inner=` argument) so that each path's slice would be a view.

It is not: a slice of the middle axis is not one matrix across nodes (row stride `dim × species × k`, not
`size × species × k`), so Inductor still copies it. The sweep profiles (`murphy-1m-sweep-profile` against
`murphy-reassoc-sweep-profile`, both R = 16,384) show the same copy kernels at the same times
(`clone_slice_transpose_view` became `clone_slice_view`). Measured:

| Run | Steady 1M call (Ada) | MH-1 NaNbO3 step (5060 Ti) | MH-1 NbCl5 step (5060 Ti) |
| --- | --- | --- | --- |
| main `a7159afd` | 14.28 / 14.35 s | 13.31 / 13.35 ms | 17.35 / 17.51 ms |
| `e2903496` | 14.34 / 14.35 s | 14.02 / 14.09 ms | 17.76 / 17.98 ms |

(1M: murphy, order base, new, new, base, records `rtx4000ada-murphy/murphy-reassoc-*`. Whole system:
`tools/ab_step_time.py --sides fastmace`, one process per side, order base, new, new, base, local RTX 5060 Ti,
`rtx5060ti-local/reassoc-local/`.) No gain at 1M, and the whole-system MH-1 step is 2-5 % slower: the segmented sum
and its backward write and read the sums with the larger stride. The CUDA backward's `grad_y` also changed by
rounding (compiled sum order). A layout that would really give views (`[(l, m), nodes, species, k]`, rows ordered
by `m` then node) moves the transpose to the per-path outputs, which `torch.cat` copies anyway, but writes the sums
with an even larger stride; at most about 1.7 % of the 1M call (the copy kernels' share of sweeps 1 and 3), so it
was not tried.

## CUDA graphs for compiled tiles: opt-in (`fe7b0a7b`, 3 November 2026)

`TileConfig(compiled_graphs=True)` (mode `compiled` or `auto`, device state, no lanes; `tiled_capacity.py
--compiled-graphs`) captures each slot's compiled sweep bodies in a CUDA graph after two warm-ups and replays it per
tile, as mode `graphs` does with the eager bodies. The idea: remove the host launch gaps between a compiled tile's
kernels. At 1M atoms those gaps are about 2 % of the call (GPU 97.8 % busy, "Whole-call profile"), so the gain was
expected only for small tiles. Steady calls (calls 2-4, or 3-4 at 1M), SrTiO3, OMAT-0 medium, float32:

| Cell, GPU | Cap R | Without | With graphs | With / without |
| --- | --- | --- | --- | --- |
| 40,000 atoms, RTX 5060 Ti (one run each) | 4,096 | 0.761, 0.761 s | 0.697, 0.678 s | 0.90 |
| 214,375 atoms, RTX 5060 Ti (ABBA) | 8,192 | 3.311-3.392 s, mean 3.333 | 3.157-3.276 s, mean 3.204 | 0.961 |
| 975,560 atoms, RTX 4000 Ada leto (ABBA) | 16,384 | 14.21-14.31 s, mean 14.26 | 14.24-14.27 s, mean 14.26 | 1.000 |
| 975,560 atoms, RTX A4000 trillian (ABBA) | 8,192 | 16.58-16.95 s | out of GPU memory | — |

The energies agree to rounding (atomic sums: -7794285.0088 to -7794285.0092 eV at 1M with and without), and so does
the force check. Memory: the graphs keep each slot's and sweep's private pool. At 1M on the Ada the memory held after
a call rises from 4.97 to 5.21 GiB, and the first compiled call (the captures) peaks at 12.24 GiB as without. On the
16 GB A4000 the first compiled call with graphs ran out of memory even at R = 8,192 (7.15 GiB in private pools,
2.62 GiB more requested), where the run without graphs fits; at R = 16,384 all four trillian runs, with and without,
ran out already in the eager first call. Graphs re-capture when a slot's size or the call's position, species or
stress tensors change (`_graph_key`), so an MD loop must keep its tensors in place to profit.

So the switch stays opt-in (default off): no gain at 1M, less memory headroom, a gain of 4-10 % only where tiles are
small and launch-bound. Records: `results/tiled-1m-20261103/rtx4000ada-leto/leto-graphs-*` (`graphs_ab.sh`),
`rtx-a4000-trillian/trillian-graphs-*` (R = 16,384, all out of memory) and `trillian-graphs-r8-*`,
`rtx5060ti-local/graphs-214k/` and `graphs-40k/` (scripts inside). The source of all runs is `fe7b0a7b`. murphy was
busy with another user's job and was not used.

## MD at 1M: list rebuilds, slots and the remaining queue (3 November 2026)

**Setup.** `tools/bench/tiled_md.py`, SrTiO3 58×58×58 (975,560 atoms), OMAT-0 medium, float32, `--mode auto`, tile
lists, species index, R = 16,384, 60 steps of 1 fs at 600 K, host integrator; skin 0.3 Å; leto (RTX 4000 Ada).
Records `results/tiled-1m-20261103/rtx4000ada-leto/leto-md1m-*`.

**Main (`7430fb51`).** The slots settle in the first call (3 slot changes, none afterwards in 60 steps; the large
slot (16,385, 9,812, 1,275,904) holds). Steps that reuse the kept lists take 14.66 s, the 8 steps that search again
(every 7-8 steps) 16.77 s: a rebuild costs 2.11 s, about 1.9 % of the MD time. Where it goes
(`search_parts.py`, `leto-search-parts`): the 64 tiles' search within cutoff + skin (6.3 Å) 2.25 s; packing the
kept lists 0.09 s; host copies 0.03 s; plan and binning 2 ms each. With cutoff-wide bins the 27-bin stencil tests
about 613 candidates per atom for 93 edges.

**Changes (branch `fastmace/tiled-slot-neighbours`):**

| Change | What it saves | Check |
| --- | --- | --- |
| Half-cutoff bins and a ±2 stencil (`TileNeighbours`, `STENCIL_SPLIT = 2`): 125 bins of (2.5 c)³ instead of 27 of (3 c)³ | search 2.25 → 1.47 s per rebuild at 1M (order 1, 2, 2, 1, 1, 2 in one process, `search_stencil.py`, `leto-search-stencil`) | the same edges in both (digest per tile); `test_tile_neighbours_are_bitwise_the_global_list` with both stencils |
| Senders of the kept lists and the call's list cache as int16 ranks into each tile's int32 senders (`_pack_senders`) | about 3 instead of 5 bytes per edge: kept lists 442 → 281 MiB, cache 347 → 215 MiB at 1M | exact unpacking, CPU test at 1, 65,536 and 65,537 distinct senders |
| The skin check by chunks (`moved_within`) | three whole-system float64 host arrays per call (72 B per atom, about 10 GiB at 149M) | the same decision, CPU test at four chunk sizes |
| A tile whose edge bucket has no slot runs in a neighbouring bucket's slot that fits (`_slot_keys`) | a compile when a tile's edges drift across a bucket's edge (none in these 60 steps) | CPU test: no slot change, the same forces |
| `wavefront="auto"` from `WAVEFRONT_MIN_LAYERS` (4) grid layers where the plain state does not fit the host at all | cells below 12 layers whose plain state exceeds the host (the local PC between about 5M and 17M atoms) | CPU test both ways |

The branch at 1M (`3a83c308`, `leto-md1m-branch2`): reuse steps 14.68 s (main 14.66 s), rebuild steps 16.07 s
(+1.38 s instead of +2.11 s); over steps 2-60 14.87 s against 14.95 s per step (-0.5 %); the same drift (0.033
meV/atom over 60 steps), potential energies equal to 2×10⁻³ eV in 7.75×10⁶; GPU peak 12.23 GiB as main.

**Rejected on the way: edge headroom for the slots** (`5a2dd70f`, `leto-md1m-branch`). Sizing a slot's edges with
`TileConfig.headroom` like its receivers and halo grew the 1M slot to 1,336,832 edges; the padded edges cost every
call: reuse steps 15.03 s (+2.5 %) and GPU peak +0.32 GiB, against a recompile that 60 steps never needed. Removed in
`3a83c308`; the neighbouring-slot rule needs no headroom.

**Not pursued, by the measurements:**
- **32-bit sort keys in the search:** the sort is over the kept edges only (a small part of the 1.47 s, after the
  distance test); per-tile ranks would need another sort first.
- **Per-tile topology cache:** a tile's local topology costs the `fill` phase, 0.08 s of a 14.3 s call (0.6 %).
- **Pencil wavefront with a GPU band:** at 1M the state lives on the device and the wavefront is off; it would
  only matter for host-state runs at capacity sizes, which are parked (no bigger cells).
