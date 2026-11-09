# Tiled call speed: whole-system rewrites in the tile bodies (2026-11-08)

**Task** (conductor, 8 November 2026): bring the tiled call (`fastmace/model/tiled.py`) and its wavefront schedule up
to the whole-system path's speed for MACE-OMAT-0 medium on the local RTX 5060 Ti. Profile at about 1M SrTiO3 atoms,
port the whole-system rewrites into the tile bodies where they are correct per tile, A/B each in both orders, and
keep the wavefront exact.

**Setup.** Local PC, RTX 5060 Ti 16 GB (sm_120, WSL2), `~/environments/mace_cueq_oeq` (torch 2.14.1+cu130), hybrid
backend, float32, TF32 off, `omat0-medium`. Each run went through `tools/ci/local_gpu.py run`, from a
`git archive` tree. Baseline: main `d5de5bf7`. Branch `fastmace/tiled-speed`: `cfec5551` (three switches),
`de85f93f` (graph cache), `b9362cc9` (per-GPU defaults), `887b6b1d` (graph cache budget fix), `29c91cce` (block
rows off under the wavefront); gap work on 9 November: `838930c0` (float32 filter), `b5a59f7d`/`0134304d`
(compiled epilogues), `d5e91441` (fill skip, pinned kept lists), `10a8d1c0` (wavefront scoping), `455facf2` (final
code). Structure: SrTiO3 supercells from `tools/bench/tiled_capacity.py`; 58×58×58 is
975,560 atoms, 72.2M edges, R = 16,384 receivers per tile, 68 tiles (100 under the wavefront).

## Result

| Path, 975,560 atoms | Before (d5de5bf7) | After the first four (8 Nov) | After the gap work (455facf2) | Same-process A/B, every switch off against the defaults, b/a (ab / ba) |
| --- | ---: | ---: | ---: | --- |
| State on the device | 8.62 s, 8.83 µs/atom | 6.67 s, 6.84 µs/atom | 6.26 s, 6.42 µs/atom | 0.729 / 0.731 (8.51 → 6.21 s) |
| State on the host, plain schedule | 10.64 s, 10.90 µs/atom | 8.16 s, 8.37 µs/atom (de85f93f) | not run (see "Host state") | not run |
| State on the host, wavefront | 12.29 s, 12.60 µs/atom | 11.77 s, 12.06 µs/atom (29c91cce) | 11.68 s, 11.98 µs/atom | first four 1.008 / 0.963; gap 0.998 / 0.984 |

**Wavefront round (9 November 2026, d2435c8e).** A profile of the wavefront call (job 19, `jobs/wprof.py`) showed it
host-bound: 13.3 s wall, 8.4 s of GPU work, and the calling thread spent 4.3 s building tile graphs (four builds per
tile) and about 3 s applying the host state's writes. Two exact switches remove that, and the three switches that
had been off under the wavefront are faster there once it is GPU-bound:

| Switch | Class | Wavefront b/a (ab / ba) | Job |
| --- | --- | --- | --- |
| `TILE_GRAPH_CACHE_WAVE`: the graph cache from the search pass to each tile's sweep 3 | exact | 0.824 / 0.817 | 20 (7f193e78) |
| `TILE_HOST_WORKER`: host-state writes on one worker thread, reads wait for the earlier batches | exact | 0.975 / 0.982 (host state, plain schedule: 0.969 / 0.966) | 21 (1d30a163) |
| `TILE_BLOCK_ROWS` under the wavefront | rounding (gated) | 0.931 / 0.937 (was 1.030 / 1.060 host-bound) | 22 (1d30a163) |
| `TILE_EPILOGUE` under the wavefront | rounding (gated) | 0.985 / 0.983 (was 1.012 / 1.009) | 22 |
| `TILE_PINNED_SUPERSET` under the wavefront | exact | 0.989 / 0.990 (was 1.012 / 1.028) | 22 |
| All five, the old wavefront defaults against the new | | **0.766 / 0.774**: 10.10 → 7.74 µs/atom | 23 (d2435c8e) |

The wavefront call is now 7.74 µs/atom (A/B median 7.55 s), 1.61× the whole-system call per atom, down from 2.50×.
Its profile at d2435c8e (job 23): 7.72 s wall, 7.10 s of GPU work (about 1 s of it host-device copies). The host
state's plain schedule is 6.94 µs/atom (A/B median, job 23: 0.968 / 0.972 for the worker). The A side of job 23
(10.10 µs/atom) is faster than the 11.98 µs/atom single call at 455facf2: it already has the filter32 and fill-skip
switches and the B side's warm caches in the same process; the decision measure is the ratio.

The single-call rows are each tool's third call (`tiled_capacity.py --sweep-times`, one process per row; cross-process
noise is about 3 %). The A/B column is the decision 66 measure: one process, both sides on one `TiledForceField`,
3-4 alternating blocks of 2 calls after 3 warm-up calls per side.

**Against the whole-system call.** The largest SrTiO3 cell measured whole is 18×18×18 (29,160 atoms):

| 29,160 atoms | Before | After |
| --- | ---: | ---: |
| Whole system (compiled plan) | 0.1408 s, 4.83 µs/atom | 0.1397 s, 4.79 µs/atom (0.1399 s after the gap work) |
| Tiled (2 tiles) | 0.268 s, 9.18 µs/atom | 0.253 s, 8.69 µs/atom (0.203 s, 6.97 µs/atom after the gap work) |
| Tiled at 975,560 atoms, device state | 8.83 µs/atom | 6.84 µs/atom (6.42 µs/atom after the gap work) |

At 975,560 atoms the tiled call now costs 1.34× the whole-system call per atom (1.83× before, 1.43× after the first
four switches). The wavefront costs 2.50× (2.61× before). At 29,160 atoms the two
tiles carry halos of 4,000 rows each on 16,384 receivers, and the call's fixed costs dominate; the tiled call is not
meant for that size.

## The switches

All four are module constants; the A/B sides set them per call (`tools/bench/tiled_ab.py --a/--b`).

| Switch | What it removes | Class | Device state b/a (ab / ba) | Wavefront b/a (ab / ba) | Default |
| --- | --- | --- | --- | --- | --- |
| `tiled.TILE_PREFETCH_DEVICE` | the GPU idle while the next tile's graph was built after the previous body drained the GPU (`torch.unique`, edge counts): `TileConfig.prefetch` now also runs with the state on the device | exact | 0.939 / 0.941 | – (host state prefetched before) | None: on for `TILED_SPEED_GPUS` |
| `generic.TILE_BLOCK_ROWS` | the dense `[R + H, width]` message, its receiver slice and the gradient padded back to R + H: the w′ convolution's BLOCK_MESSAGE groups are narrowed to the receivers' rows (a view) and `linear_1` multiplies them in the block layout, as on the whole system | rounding (energy 4e-10 relative) | 0.918 / 0.921 | 1.030 / 1.060 (slower) | None: on per call, off under the wavefront |
| `tiled.TILE_VIRIAL` | the strain leaf in sweeps 2 and 3: its backward was a `[3, E] × [E, 3]` split-K GEMM (`sgemm_largek`, ≈ 0.3 µs/atom); now sym(Vᵀ ∂E/∂V) of the tile's edges, padding edges masked | rounding (stress 8.6e-8 eV/Å³ of 0.029) | 0.970 / 0.955 | 0.994 / 0.959 | None: on for `TILED_SPEED_GPUS` |
| `tiled.TILE_GRAPH_CACHE` | three of the four graph builds per tile and call (copy, unpack, `torch.unique`): the slot-key pass keeps each tile's local index (int32) and shifts on the device until the call ends | exact | 0.948 / 0.960 (887b6b1d) | – (off under the wavefront) | None: 25 % of the free memory less the tiles' predicted workspace on `TILED_SPEED_GPUS` |
| all four | | | 0.795 / 0.787 | 1.008 / 0.963 (block rows off) | |

`TILED_SPEED_GPUS` is `GPUModels(("NVIDIA GeForce RTX 5060 Ti",))`. Three switches keep more memory alive or replace a
kernel, so decision 66 scopes them to the GPU family where they were measured. `TILE_BLOCK_ROWS` only removes work,
so it is on for every GPU on the plain schedule. True / False (a fraction for the cache) force a side everywhere.

**Block rows under the wavefront.** The same switch that saves 8 % on the plain schedule cost 3-6 % under the
wavefront (same 100 tiles, same four padded slots on both sides, no recompile warnings). The cause was not found in
this task. The tiled call therefore sets each w′ interaction's `tile_block_rows` per call to `not wave`. With all
switches forced on, the wavefront was 1.036 / 1.002; with block rows off there (29c91cce), 1.008 / 0.963: not slower, and no clear gain. The wavefront is limited by the host state's CPU work (below).

**Graph cache budget.** de85f93f gave the cache 25 % of the free memory at the start of every call. On the first,
measured call the kept graphs counted as tile workspace, so the receiver budget chose 8,192 instead of 16,384 (138
tiles instead of 68; job 4). Later calls could also let the cache take memory the tiles needed. 887b6b1d keeps no
graphs in a measured call and budgets the fraction of the free memory less 1.1 × the measured workspace per
receiver × the cap. After it, 68 tiles again and 6.67 s.

## Where the time went (before)

At 975,560 atoms on device state (8.62 s): sweep 1 0.86 s, sweep 2 3.92 s, sweep 3 1.99 s, the neighbour search
0.73 s, about 1.6 s outside the timed sweeps, and about 1.2 s with the GPU idle. Per receiver, the bodies' kernels
took 753 / 3,757 / 1,823 ns in sweeps 1 / 2 / 3, against 4.72 µs/atom for the whole whole-system call. Sweep 2 was
mostly the fused folded forward (1,170 ns) and w′ (1,122 ns); the rest was the densify and slice copies (removed by
`TILE_BLOCK_ROWS`) and `sgemm_largek` (155 ns per sweep, removed by `TILE_VIRIAL`). Both paths applied the same rewrite
records (generic_species_skip, reassociate, segmented_first_sum, fused_radial_backward, streaming,
chunked_conv_backward, fused_folded_forward, block_linear, grouped_linear, grouped_launch, radial_tables,
wprime_backward, single_graph_strain).

After (887b6b1d, 6.67 s): sweep 1 0.85 s, sweep 2 3.21 s, sweep 3 1.78 s.

**What is left** (not done here, bounds from the profile):

- Sweep 3 recomputes interaction 0's forward, about 0.75 µs/atom. That is the three-sweep design (the activations of
  interaction 0 are not kept for 1M atoms), not a missing rewrite.
- Halo work: each tile computes interaction 0 on R + H rows, H ≈ 9,000 for R = 16,384.
- The list filter in float64 (about 0.35 s per call) and the host list cache's fill, which the graph cache makes
  redundant for tiles it holds.

## Gap work (9 November 2026)

After the four switches the tiled call cost 1.43× the whole-system call per atom. A region profile of one steady
call (`jobs/kprof3.py`, device state, a8d6ffce, record_function labels per method, each kernel charged to its
innermost label) split its 6,479 ns/atom:

| Region | ns/atom | |
| --- | ---: | --- |
| Autograd backward of sweeps 2 and 3 | 2,404 | w′ backward 1,199, GEMMs 297 |
| Sweep 2 forward | 1,635 | fused folded message 1,276 |
| Sweep 1 forward (interaction 0) | 763 | |
| Sweep 3 forward (interaction 0 again) | 804 | the three-sweep recompute |
| `_tile_graph` (slot-key pass) | 327 | H2D copy 69, float64 filter about 105, `unique` 41 |
| `_scatter` (forces, state) | 116 | |
| `_piece`, `_body`, `_first`, `_second`, `_fill`, virial, rest | about 430 | eager glue |

The model's own kernels without the recompute (4,802 ns/atom) match the whole-system call (4,718). The gap is the
recompute plus about 870 ns/atom of glue. The glue was the target:

| Switch | What it removes | Class | Device state b/a (ab / ba) | Wavefront b/a (ab / ba) | Default |
| --- | --- | --- | --- | --- | --- |
| `tiled.TILE_FILTER32` | the kept lists' float64 cutoff test (1/64 rate on this GPU): float32 test, float64 again only within a bound of the cutoff | exact | 0.987 / 0.993 | 0.981 / 1.012 | None: on for `TILED_SPEED_GPUS` |
| `tiled.TILE_EPILOGUE` | about 20 eager kernels per tile and sweep: masks, sweep 3's objective, virial, edge vectors, now in compiled code | rounding | 0.980 / 0.981 | 1.012 / 1.009 | None: `TILED_SPEED_GPUS`, off under the wavefront |
| `tiled.TILE_FILL_SKIP` | the host list cache's fill (copy back and sync) for tiles the graph cache keeps | exact | 0.961 / 0.966 | – (no graph cache there) | True |
| `tiled.TILE_PINNED_SUPERSET` | pageable copies of the kept lists to the device: pinned, up to 4 GiB | exact | 0.986 / 0.990 | 1.012 / 1.028 | None: `TILED_SPEED_GPUS`, off under the wavefront |
| all four, off against the defaults | | | 0.923 / 0.928 | 0.998 / 0.984 | |
| all eight, off against the defaults | | | 0.729 / 0.731 (8.51 → 6.21 s) | | |

`TILE_FILTER32` keeps the same edges in the same order: the band is four times the float32 error bound for the call's
largest coordinates, and CPU and CUDA tests at 200k atoms (small cell; coordinates up to 900 Å) compare it with the
float64 test. It removes work, so the wavefront's mixed A/B (pairs 0.88–1.04) does not scope it off. The epilogue and
the pinned lists are not work-removing switches; their wavefront A/Bs were slower in both orders, so they are off
there. The wavefront A/Bs are noisier than the device-state ones: the host state's CPU work loads 10 or more cores, and
pairs within one A/B spread by ±8 %.

After the gap switches (455facf2, single calls, `tiled_capacity.py --sweep-times`): device state 6.26 s, 6.42 µs/atom (third call; A/B median 6.21 s, 6.37 µs/atom), wavefront
11.68 s, 11.98 µs/atom. Against the whole-system call's 4.79 µs/atom: 1.34× per atom on device state (single call; 1.33× from the A/B median) (was 1.43× after the first four, 1.83× before).
Region profile at the defaults (`jobs/kprof4.py`): 6,168 (`_tile_graph` 222 → 136, the glue regions about 870 → 520) ns/atom.

**Rejected:**

- `TILE_PIECE_START` (the graph cache keeping a tile's index at its slot's halo start, about 60 ns/atom of eager
  work): 0.9999 / 1.0001. That work runs in the prefetch on the side stream, behind the bodies. Removed (455facf2).
- Keeping interaction 0's node-level A for sweep 3 instead of recomputing it: 2,048 floats = 8 kB per atom, 8 GB at
  1M atoms. It does not fit beside the tiles on 16 GB, and the host round trip (about 1.2 s) costs more than the
  0.73 s saved. Rejected at the bound.

**Host state, single calls.** Job 8's host-state call took 12.87 s and job 11's 24.6–33.5 s, against 8.16 s in job 4.
Job 12 ran all switches off against the defaults in one process (both orders): the off side was as slow (13–30 s),
and the pairs spread from 0.59 to 1.31. The slowness was the machine (WSL load up to 24, with the host state's own
`index_add` threads among it), not a regression.

**What is left:** sweep 3's recompute of interaction 0 (about 0.8 µs/atom, the three-sweep design); halo rows
(H ≈ 9,000 per 16,384 receivers in interaction 0); `_scatter` (115 ns/atom, the forces' two `index_add`s over the
edges); and on the wavefront about 0.6 s of host-device copies and launch gaps not hidden behind the GPU work
(wall 7.72 s against 7.10 s busy).

## Correctness

- **CPU tests** (`tests/cpu/test_tiled.py`): the virial equals the strain leaf's gradient (eager and padded), the
  padding edges' mask, prefetch on device state bitwise the call without, the graph cache bitwise the call that
  builds every graph (all tiles kept, some kept), the cache budget rule, the per-GPU defaults, `tile_block_rows`;
  the float32 filter keeps exactly the float64 test's edges (small cell, large coordinates) and the tiled call with
  it is bitwise the call without; the epilogues bitwise the eager ones outside mode "compiled" (strain leaf and
  virial); the fill skip bitwise the call that fills (all and some tiles kept). The full suite runs on leto once the
  Ada capacity scan releases it (queued on 455facf2).
- **GPU test** (`tests/gpu/test_tiled_speed_cuda.py`): each switch on against off and against the whole-system call,
  SrTiO3 6×6×6 rattled and strained, R = 256, eager and padded, and the epilogue also compiled. Passed in job 9
  (a8d6ffce) and job 17 (455facf2; energy within 1e-8 relative, since the forward's sums are float32 atomics too).
  `test_filter32_keeps_exactly_the_float64_tests_edges_on_cuda` (200k atoms, 4M pairs) passed in job 13.
- **Force/stress gate** (`tools/force_stress_gate.py --engine fastmace --model omat0-medium --tiled-receivers 48
  --tiled-mode compiled`, f36adabd, every switch on): 12 of 12 checks pass. Again with `TILE_EPILOGUE` and the
  float32 filter on (b5a59f7d, job 14): 12 of 12, SrTiO3 energy ratio 1.86. Ratios to ASE float32 (energy / forces /
  stress): SrTiO3 3×3×3 1.71 / 0.78 / 0.69, NaCl 1.30 / 0.71 / 0.70, NbCl5 0.96 / 0.95 / 0.56. The energies are far
  below the 0.01 eV floor. The whole-system default gate on this GPU had 1.52 / 0.87 / 0.68, 1.21 / 0.81 / 0.84,
  0.84 / 0.98 / 0.55 (`results/omat0-kernel-lab-20261103`).
- **A/B side agreement at 975,560 atoms:** forces within 4.4e-6 eV/Å for every switch, exact ones included (the
  forces are summed with atomics), energy within 1.2e-6 meV/atom, stress within 8.8e-8 eV/Å³.
- **Wavefront:** the wavefront is unchanged in what it computes. `TILE_GRAPH_CACHE` is off under it, and
  `TILE_PREFETCH_DEVICE` does not apply (host state). Block rows are off there by default. The ported paths run under
  it: the wavefront A/Bs above ran every switch forced on.

## Files

- `jobs/`: the job scripts (`job1.sh` baseline profile, `job2` kernel tables, `job3` A/Bs of three switches, `job4`
  graph cache, `job5`/`job7` gate, GPU test and wavefront A/Bs, `job6` budget fix, `job8` final of 8 November,
  `job12` host-state A/B, `job13` float32 filter, `job14`/`job15` epilogue, fill skip, pinned lists, `job16` piece
  start, `job17` final), `kprof.py` (kernel table), `kprof3.py`/`kprof4.py` (region profiles), `cmp.py`, `summ.py`,
  `collect.sh`.
- `ab/<job>/`: every A/B record (`tiled-ab-SrTiO3-975560-<switch>-<order>.json`).
- `profiles/<job>/`: the `tiled_capacity.py` records (before: job1; after: job4, job6, job8, job11, job17) and the
  kernel and region profiles.
- `gate/<job>/metadata.json`: the force/stress gates (job 7, job 14).
- `gputest/`: the GPU test logs.
- New tool: `tools/bench/tiled_ab.py`.
