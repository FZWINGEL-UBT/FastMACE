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

The Ada (RTX 4000 Ada, 48 SMs) still has to be measured before the idea is closed. Section "Ada" below follows
when leto or murphy is free.
