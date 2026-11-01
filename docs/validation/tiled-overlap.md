# Tiled path: the host-state writes overlapped with the next tile

With the state on the host, every tile copies its results to the host and the host indexes them into the state
before the next tile is queued. The profile with the list cache ([tiled-speed.md](tiled-speed.md)) puts this at
10 % of a compiled call (1.71 s of 16.96 s at 884,736 NbCl5 atoms), with the device busy only 64 % of the time.

## The change

`TileConfig.overlap` (default on; only with `state="host"`):

- **Writes.** A tile's host-state writes (`put`, `add`, and the closing of aliased dE/dx rows) are recorded.
  - On CUDA, each is a non-blocking copy into a pinned buffer, plus an event.
  - On the CPU, each is a copy, because padded slots reuse their buffers.
- **When the writes land.** They are applied at the next tile's first host write, by which time the next tile's
  device work is queued. They are applied in order, each after its event.
- **Reads.** Host rows reach the device through pinned staging with a non-blocking copy.
- **End of a sweep.** Every sweep ends with all pending writes applied, before the next sweep reads them.
- **Why no tile reads a row still pending.**
  - Sweep 2 reads sc and x of sweep 1.
  - Its writes go to its own receivers' sc rows and to closed atoms' x rows.

## CPU tests

`test_tiled.py` and `test_layering.py`, 96 passed.

`test_deferred_host_state_writes_are_bitwise_the_immediate_ones` covers eager and padded, plain and aliased state.
In each case the deferred writes are bitwise the immediate ones, and no write is left pending after the call.

## Same-process A/B

Run on murphy, NVIDIA RTX 4000 Ada (20 GB), with `mace_cueq_oeq`, MH-1 float32 hybrid (`mh1-r2scan`), at FastMACE
aa313e49 (`results/tiled-speed-20261101/ab_arms.py`, arms `X` against `X+nooverlap`).

- **Configuration:** tiles of R = 4,096, tile lists with the cache, the state on the host and aliased, the species
  index, and the default budget.
- **Idle check:** the GPU was idle before every job (`idle_check.txt` in each run).

| Pair | Overlap on / off | Medians |
| --- | ---: | --- |
| compiled, 884,736 atoms | **0.973** | 16.50 s against 16.96 s |
| eager, 884,736 atoms | **0.977** | 22.76 s against 23.30 s |
| compiled, 9.26M atoms | **0.958** | 161.2 s against 168.3 s |

**Forces gate.** Against the overlap arm:
- energy within 5.2e-8 meV per atom;
- forces within 1.5e-6 eV/Å;
- stress within 7e-11.

This is the device's run-to-run spread; the CPU path is bitwise.

**Memory.** The pinned buffers hold one tile's rows at a time, beside the slot workspace; the device memory is
unchanged.

The runs are in `results/tiled-speed-20261101/runs/murphy-overlap-*`.

## What is left

- The overlap gains 2–4 %, a quarter to a half of the host-state share; the host's indexing itself remains.
- The device is still idle for much of a compiled call. The next candidate is a CUDA graph per padded slot and
  sweep for the compiled tiles. It goes ahead only after an nsys gap histogram shows that the idle time is bound by
  launches and Python.
