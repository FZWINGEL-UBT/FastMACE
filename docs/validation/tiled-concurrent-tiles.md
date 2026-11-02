# Two tiles at once: stages 0 and 1 (2 November 2026)

Plan: `plans/tiled-concurrent-tiles.md`. The owner's idea is to keep the tensor cores and the CUDA cores busy in
different tiles at the same time. The owner said to run stages 0 and 1 on 2 November 2026. They are
measurement only: no library change.

**Setup**
- **Tool:** `tools/bench/tiled_corun.py`.
- **Model and structure:** MACE-OMAT-0 medium, hybrid float32, cubic SrTiO3, the default rewrite set.
- **Tiled call:** `TiledForceField` in mode "compiled", per-tile lists, state on the device (no host state, no
  wavefront), species index.
- **Lane B:** a second force field with `tf32x3_gemms="on"` (and `grouped_launch="off"`, which the planner
  requires beside it).
- **Probe:** it records copies of two tiles' padded slots per sweep from one call, then replays pairs of
  tile-sweeps. Each lane uses its own slot copy and reads the call's device state:
  - `serial`: back to back on one stream;
  - `streams`: on two streams;
  - `split`: on two streams of green contexts holding disjoint halves of the SMs.
  - The time is per pair, `--reps 3` pairs per block, 6 or 10 blocks with the mode order rotated, all in one
    process.
- **GPU and environment:** RTX 5060 Ti (local, WSL2, desktop idle), `~/environments/mace_cueq_oeq`, FastMACE
  branch `fastmace/omat0-wavefront`.
- **Results:** `results/tiled-corun-20261102/`.

## Stage 0: where the time goes (RTX 5060 Ti)

One compiled call (164k atoms, R = 8,192, 27 tiles; and 320k atoms, R = 16,384) under torch.profiler.

| Kernel class | Share of kernel time (R = 8,192) | (R = 16,384) |
| --- | --- | --- |
| OEq convolution (`oeq_projected_*`, streaming) | 27 % | 26 % |
| GEMMs (cuBLAS/CUTLASS SIMT `sgemm`, Inductor) | 23 % | 22 % |
| cuEquivariance JIT kernels (named `kernel`) | 21 % | 21 % |
| Pointwise and reductions (Triton, ATen) | 11 % | 13 % |
| Copies and fills | 9 % | 9 % |
| Gathers, scatters, index | 5 % | 5 % |
| Other | 4 % | 4 % |

**Device occupancy.** The device is busy 70 % of the profiled span at R = 8,192 and 88 % at R = 16,384. That is
the union of kernel intervals, and the profiler's own gaps count against it. The compiled call's
96 % (tiled-omat.md) was measured at R = 16,384 on the Ada without the profiler.

**Bound.** The GEMMs, the only work the tensor cores could take, are 22-23 % of the kernel time. A tensor-core
lane can therefore remove at most that share, and only if it fully overlaps the other lane.

## Stage 1: the co-run probe (RTX 5060 Ti)

**Single sweeps (R = 7,322 receivers, 0.46M edges; ms per tile):**

| Sweep | FP32 | 3×TF32 lane |
| --- | --- | --- |
| 1 | 10.4-10.6 | 11.7-12.0 |
| 2 | 66.2-66.9 | 67.5-67.7 |
| 3 | 22.7-22.8 | 25.6-25.7 |

The 3×TF32 lane is 2-13 % slower per sweep. That matches `tf32-screen.md` on the Ada.

**Pairs.** Each cell gives the median pair time (ms) and its ratio to `serial` in that run. In brackets: the
fastest pair of each mode.

| Pair | serial | streams | split (SM halves) | streams, lane B 3×TF32 |
| --- | --- | --- | --- | --- |
| 2 + 2 | 153.0 (137.5) | 145.6, 1.05× (138.7) | 134.0, 1.08× (131.0); serial 145.3 (140.8) | 148.8, 1.05× (144.8) |
| 2 + 1 | 95.0 (77.1) | 79.9, 1.19× (74.9) | 91.5, 0.84× (88.0); serial 76.4 (76.1) | 98.1, 1.00× (77.4) |
| 2 + 3 | 111.6 (96.2) | 101.9, 1.10× (100.1) | 96.3, 0.94× (94.9); serial 90.7 (87.9) | 103.6, 1.07× (89.4) |
| 1 + 3 | 33.0 (32.9) | 33.3, 0.99× (33.2) | 38.1, 0.86× (37.9); serial 32.6 (32.6) | n/a |

**Repeat with the default allocator.** This was 10 blocks without lane B. The fastest pairs, serial against
streams:
- 2 + 2: 136.3 against 138.6 (0.98×);
- 2 + 1: 76.2 against 76.4 (1.00×);
- 2 + 3: 88.6 against 85.4 (1.04×);
- 1 + 3: 32.5 against 33.3 (0.98×).

**Reading.**
- **Noisy serial pairs.** Whenever streams run in the same process, the serial pairs scatter widely (2 + 1:
  76-109 ms). The run without streams (the split run) has tight serial pairs (76.1-78.3 ms).
  - The likely cause is the caching allocator: blocks freed on a side stream are kept for that stream, so the
    serial runs between them allocate anew under memory pressure.
  - So the streams medians (1.10-1.19×) measure that artefact, not overlap. On the fastest pairs, two streams give
    0.98-1.04×.
- **SM split.** The split helps only two sweep-2 tiles (1.08× in its own run; 131.0 ms against the best serial of
  any run, 136.3: 1.04×). It costs 6-16 % on every mixed pair, because a lane loses half the SMs for its whole
  sweep.
- **3×TF32 lane.** Lane B on the tensor cores gives nothing beyond plain two streams (1.00-1.07×, within the same
  noise). Its own sweeps are slower, and the GEMMs are under a quarter of the work.
- **Memory.** At R = 16,384, two sweep-2 tiles at once do not fit on the 16 GB card. One sweep-2 frame allocates a
  5 GB buffer (985,600 × 1,280 float32) and the call alone peaks at 9.6 GB. The failure is "Failed to create GPU
  mapping" in `results/.../corun-SrTiO3-320000-R16384-pairs-oom.json`.
- **Green contexts.** They need PyTorch's default allocator. With expandable segments, the first tile on a
  green-context stream fails with "CUDA driver error: device not ready" ("Failed to create GPU mapping").

**Stop criterion on the 5060 Ti:** no.
- The criterion is at least 1.05× for (b) or (c), or (d) at least 1.05× over them.
- Plain streams do not reach 1.05× on clean timings.
- The split reaches it only for 2 + 2, and against its own run's baseline.
- The tensor-core lane adds nothing.

## Ada (RTX 4000 Ada Generation, leto, 48 SMs)

**Setup.** The same commands, run through `fm-tiled/corun_chain.sh` on the source copy `src-corun-a6ea39dc`, with
the NAS `mace_cueq_oeq` environment. Each run passed the idle check (`idle_check-*.txt`). Results are in
`results/tiled-corun-20261102/rtx4000ada-leto/`. The serial timings are tight here (spread at most 2 %).

**Stage 0.**
- **R = 8,192:** the device is busy 90 % of the profiled span. Kernel-time shares: convolution 27 %,
  cuEquivariance 23 %, GEMMs 17 %, pointwise 13 %, copies 9 %, index 6 %, other 5 %.
- **R = 16,384:** busy 98 %, with nearly the same shares.

**Single sweeps (R = 7,322; ms):**

| Sweep | FP32 | 3×TF32 lane |
| --- | --- | --- |
| 1 | 11.0-11.4 | 11.7-11.8 |
| 2 | 62.1-63.9 | 63.4-63.8 |
| 3 | 23.6-23.9 | 25.4-25.6 |

At R = 13,845 (FP32): sweep 1 20.8-21.0, sweep 2 121.7-122.6, sweep 3 44.8.

**Pairs (R = 8,192; median ms and its ratio to `serial`, the fastest pair in brackets):**

| Pair | serial | streams | streams, default allocator | split (24 + 24 SMs) | streams, lane B 3×TF32 |
| --- | --- | --- | --- | --- | --- |
| 2 + 2 | 126.6 | 134.0, 0.95× | 134.8, 0.94× | **116.0, 1.09×** (115.8) | 135.0, 0.95× |
| 2 + 1 | 73.7 | 73.6, 1.00× | 74.7, 0.99× | 73.3, 1.02× | 75.1, 0.99× |
| 2 + 3 | 87.2 | 86.6, 1.01× | 86.3, 1.01× | 84.4, 1.03× | 88.3, 1.00× |
| 1 + 3 | 34.6 | 35.2, 0.99× | 35.1, 0.99× | 37.3, 0.93× | n/a |

**R = 16,384 (320k atoms).** Two sweep-2 tiles at once press against the 20 GB. The run log has "OOM ... while
trying to map 20971520 bytes": the allocator frees and maps again.
- 2 + 2 on two streams: 0.72× (342.6 against 245.8 ms).
- The serial 2 + 1 pairs in the same process are inflated to 255 ms, against 143 ms from the single sweeps, so
  that run's "1.76×" is an artefact.
- 2 + 3 gives 1.00× and 1 + 3 gives 0.99×.

## Conclusion

1. **The tensor-core lane is closed.**
   - On both GPUs, a lane with its GEMMs as 3×TF32 beside an FP32 lane gives 0.95-1.07×, never more than plain two
     streams.
   - Its own sweeps are 1-13 % slower.
   - The GEMMs are only 17-23 % of the tiled call's kernel time, and the convolution and cuEquivariance kernels
     (about half) have no tensor-core form.
   - The 9 October precision decision is therefore not needed for this.
2. **Plain two streams give nothing.** The clean Ada timings show 0.94-1.01×. Two kernels from different tiles
   time-slice rather than overlap, because each one fills the device.
3. **Two sweep-2 tiles on disjoint SM halves gain 8-9 %, on both GPUs** (Ada 1.09×, tight; 5060 Ti 1.08×).
   - Per receiver, the split pair at R = 8,192 (115.8 ms for 2 × 7,322) beats a single tile of twice the size at
     R = 16,384 (121.7-122.6 ms for 13,845): 7.9 against 8.8 µs per receiver on the Ada.
   - The memory is about the same: two half-size workspaces against one full-size one.
   - The gain comes from latency-bound stretches of sweep 2 (the device is busy 90 % at this tile size), which the
     other half fills.
   - Mixed pairs gain 1-3 % at best, and 1 + 3 loses 7 %.
4. **What a stage 2 would be.** Pass the stage-1 criterion only for sweep 2: run its tiles in pairs on two
   green-context halves at half the receiver cap, and run sweeps 1 and 3 on the whole device as now.
   - **Expected gain:** sweep 2 is about 60 % of the call, so about 5 % of the call.
   - **Blocker:** green-context streams fail with expandable segments, which FastMACE's allocator setup turns on
     (`configure_allocator`). Without them, the large tiled runs fragment.
   - **Before any build:** a probe of `cudaMallocAsync` as the allocator backend under green contexts, or the
     driver's fix, would come first.
   - Stage 2 is not started. It needs an owner decision, given the small gain and the allocator conflict.

## The asynchronous allocator (owner: "Test another allocator", 2 November 2026)

**Setup.**
- `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` with the probe unchanged (commit `2941aeed`, which records
  the setting).
- Source copy `src-corun-2941aeed`.
- Results: `results/tiled-corun-20261102/rtx4000ada-leto/leto-*-async/`, `.../leto-alloc-ab-*/` and
  `rtx5060ti/corun-SrTiO3-163840-R8192-async.json`.

**Green contexts work with it.** No mapping failure on either GPU. Pairs (median ms, ratio to `serial`):

| GPU, R | 2 + 2 serial | streams | split | 2 + 1 split | 2 + 3 split | 1 + 3 split |
| --- | --- | --- | --- | --- | --- | --- |
| Ada, 8,192 | 125.6 | 133.7, 0.94× | **115.4, 1.09×** | 1.01× | 1.03× | 0.92× |
| Ada, 16,384 | 245.4 | 260.9, 0.94× | **224.9, 1.09×** | 1.00× | 1.01× | 0.94× |
| 5060 Ti, 8,192 | 142.0 | 146.7, 0.97× | 134.9, 1.05× | 0.86× | 0.92× | 0.86× |

- **R = 16,384 on the Ada.** Two sweep-2 tiles now fit side by side without thrashing (with expandable
  segments: 0.72×). The split still gains 9 %, although one tile alone keeps the device 98 % busy. So the gain
  is not only filled gaps: some sweep-2 kernels run more efficiently on half the SMs.
- **The 5060 Ti** gains less (1.05×). Its serial sweep-2 runs are also a little slower under this allocator
  (69.9-78.1 ms against 66.2-66.9 ms).

**Whole tiled calls, allocator A/B** (leto, 5M SrTiO3 atoms, the scan's settings: host state aliased, species
index, wavefront "auto" (off at this size), three calls; serial, expandable first):

| Allocator | eager R 4,096 | compiled R 16,384 (1st, 2nd) | GPU peak (allocated) | reserved | host peak |
| --- | --- | --- | --- | --- | --- |
| expandable segments | 153.4 s | 127.2 s, 89.1 s | 7.89-7.94 GiB | 8.05-8.08 GiB | 20.9 GiB |
| cudaMallocAsync | 154.2 s | 120.8 s, 89.4 s | 7.89-7.94 GiB | 8.38 GiB | 20.9 GiB |

- **Results agree:** both runs are valid, and their energies differ by 1.8 meV over 5M atoms (rounding).
- **Speed and budget unchanged:** the same peaks and the same receiver cap; 0.3 GiB more reserved by the async
  pool.
- **Not yet checked:** whether the async pool behaves as well at the device limit (120M+ atoms).

**Result.** The allocator blocker is gone on the Ada.
- A sweep-2 split under `cudaMallocAsync` gains 9 % per sweep-2 pair at both tile sizes.
- That is about 5 % of a call, since sweep 2 is about 60 % of it.
- It costs a second sweep-2 workspace and an allocator change for the tiled path.
