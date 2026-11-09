# Symmetrix-XL 0.1.2 with OMAT-0 small on the H200, Benchmark_loup (9 November 2026)

Owner, 9 November 2026: "one symmetrix xl run using omat 0 small on h200".
- **Hardware:** Modal NVIDIA H200 (sm_90, driver 610.57.04), profile `third`.
- **Source and job:** source `9c8dc65f`, job script `tools/modal/symmetrix_omat0_small.sh`.
- **Symmetrix-XL:** 0.1.2 (`symmetrix-xl` and `symmetrix-xl-cuda13-sm90` wheels) in `sx_image`. Its LAMMPS
  plugin is the 0.1.2 sm_90 build already in the volume (`/data/symmetrix-0.1.2/lmp-sm90`, as in
  [sx012-profile](../sx012-profile-20261106/README.md)).
- **Model:** `mace-omat-0-small.model` (SHA-256 `0abfde07862cf1e9…`), exported with 0.1.2's `symmetrix_extract_mace`
  at its defaults: universal compact radial, 256 spline points, head `omat_pbe` (`extract.log`). The export has
  128 channels, r_cut 6 Å and 89 elements. Run in float32 with the JIT device artifact
  (`jit_device_blocks_per_compute_unit 8`).
- **Protocol:** `tools/bench/sx_loup.py`, as for medium. SrTiO3 10×10×10 (5,000 atoms) from rest, skin 0.5 Å,
  1 fs. NVE, or `fix langevin` at 300 K with damp 0.5 ps. Each run is `run 5` untimed, then `run 20`, timed by
  LAMMPS's "Loop time". Three fresh processes per ensemble.

| Run | Symmetrix-XL ms per step | µs per atom and step |
| --- | --- | --- |
| NVE (3 processes) | 11.33 / 11.29 / 11.32 | 2.26 |
| Langevin (3 processes) | 11.45 / 11.55 / 11.45 | 2.30 |
| Langevin 100 + 300 steps (neighbour rebuilds) | 12.01 | 2.40 |

## Against FastMACE on the same GPU type

FastMACE runs OMAT-0 small on the H200 at **2.12 ms per step** (Langevin 300 K, `tools/step_breakdown.py`,
`../h200-omat0-small-20261108/defaults2/`, source `e2206bb6`). The current defaults (`21cc80e7`) have the same
H200 settings. That is **about 5.4× faster** than Symmetrix-XL's 11.48 ms (Langevin mean).

| Model on the H200 | Symmetrix-XL 0.1.2 | FastMACE | FastMACE speed-up |
| --- | ---: | ---: | ---: |
| OMAT-0 medium (6 November 2026) | 14.57 ms | 5.72 ms (6 November defaults) | 2.5× |
| OMAT-0 small (this run) | 11.48 ms | 2.12 ms | 5.4× |

- Symmetrix-XL gains only 1.27× from medium to small.
- FastMACE gains 2.7× from medium to small.
- The speed-up figures above compare against medium's 6 November FastMACE number. Medium's current H200 step on
  this branch was not measured in this run.

Cost: under $0.10 metered, profile `third` (now $27.88 of $30). The first launch, at a 12-minute timeout, was
refused by the budget guard ($30.17 worst case) and cost nothing.
