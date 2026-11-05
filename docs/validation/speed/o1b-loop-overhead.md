# O1b: the `fm.md` / `Simulation` loop against the engine loop, measured again (29 October 2026)

Owner asked whether the loop overhead could be removed. The fp32 profile's O1 (`fp32-profile-20261028.md`, 4 rounds)
had blamed the output snapshot. This repeats it with 10 rounds before any change.

- Tool: `tools/bench/sim_loop_overhead.py --repeats 10` (600-step blocks alternating in one process: engine loop,
  `fm.md` with no output, thermo only, default output), MH-1 `mh1-r2scan` float32 default plan, 1000 K, 1 fs.
- Local RTX 5060 Ti, `~/environments/mace_cueq_oeq`, FastMACE main `70593240` (with `wprime_backward` and decision
  66's `grouped_launch` on this GPU), FIFO reservation. Records: `results/fp32-profile-20261028/o1b/`.

| Cell | engine | no output | thermo | default |
| --- | --- | --- | --- | --- |
| NaCl | 2.440 (2.32–2.46) | 2.507 (2.27–2.53) | 2.505 (2.28–2.52) | 2.513 (2.27–2.53) |
| NbCl5 | 17.72 (17.50–18.34) | 17.67 (17.53–18.29) | 17.65 (17.23–18.04) | 17.66 (17.29–18.31) |

ms per step, median and block range.

**Output costs nothing measurable:** default − no output = +0.006 ms (NaCl), −0.01 ms (NbCl5). The snapshot's host
time (0.12 ms per step on NaCl, 0.89 ms on NbCl5) overlaps the GPU. An earlier snapshot (copying a step before its
validation) therefore has nothing to win, and is not built.

**NaCl's loop residual** (no output − engine, +0.07 ms median) lies inside the block spread (the no-output minimum is
below the engine minimum), and NbCl5 shows none. A cProfile of the no-output loop puts `Simulation._advance`'s own
time at about 0.4 µs per step besides `engine.step`. No loop overhead worth removing was found.
