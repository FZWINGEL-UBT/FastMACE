# Tiled calls from 1M to 40M atoms, and `fit_grid` (3 November 2026)

Where the per-atom gap between the 1M and the 98–149M tiled calls comes from, and whether a finer tile grid helps.
SrTiO3, MACE-OMAT-0 medium, compiled, float32, one GPU: leto, NVIDIA RTX 4000 Ada Generation, driver 595.99.02, each
run after the idle check (`idle_check.txt`). Sources on the NAS: `fm-tiled/src-prof-b301ee58` (main b301ee58, the
series) and `fm-tiled/src-gridfit-5a162b6c` (branch `fastmace/tile-grid-fit`, the fit runs). Scripts: `scale_series.sh`,
`scale_steady.sh`, `grid_fit_check.py`. Each run folder holds `run.log`, `idle_check.txt` and the JSON record of
`tools/bench/tiled_capacity.py`. This README was written on 4 November 2026 from those records.

Force-call seconds per call; the first call of a run is not comparable across runs, and the second carries compile
time at 1M. Steady µs per atom is the last call over the atoms, given only where a run has a third call.

| Run | Atoms | R | State | Tiles | Grid | Calls (s) | Steady µs/atom |
|---|---|---:|---|---:|---|---|---:|
| leto-scale-1m-dev-r16 | 975,560 | 16,384 | device | 68 | 4×4×4 | 28.2, 65.0, 14.3 | 14.6 |
| leto-scale-1m-dev-r8 | 975,560 | 8,192 | device | 138 | 5×5×5 | 25.2, 103.9, 14.1 | 14.4 |
| leto-scale-1m-host-wave-r8 | 975,560 | 8,192 | host + wavefront | 178 | 7×5×5 | 27.4, 117.2, 15.0 | 15.4 |
| leto-scale-20m-host-r8 | 20,098,395 | 8,192 | host | 4,412 | 15×13×13 | 504.3, 395.9 | |
| leto-scale-20m-host-r8-fit | 20,098,395 | 8,192 | host, `fit_grid` | 3,136 | 16×14×14 | 469.4, 481.0 | |
| leto-scale-20m-host-r11k | 20,098,395 | 11,264 | host | 3,276 | 14×12×12 | 521.0, 452.6 | |
| leto-scale-40m-host-r8 | 40,000,000 | 8,192 | host | 7,858 | 19×17×17 | 1003.8, 686.7 | |
| leto-steady-20m-base | 20,098,395 | 8,192 | host, `--grid-fit off` | 4,412 | 15×13×13 | 503.2, 340.1, 311.0 | 15.5 |
| leto-steady-20m-fit | 20,098,395 | 8,192 | host, `--grid-fit on` | 3,136 | 16×14×14 | 468.1, 353.5, 304.4 | 15.1 |

Reading:
- The receiver cap alone costs nothing at 1M (14.6 against 14.4 µs/atom), host state and the wavefront about 7 %, and
  size from 1M to 20M nothing more once the calls are steady (15.4 against 15.5).
- Second calls are not steady: the same 20M configuration took 395.9 s and 340.1 s, so the second-call differences
  above are not evidence.
- `fit_grid` cuts the tiles by 29 % and the steady call by 2.1 %, in a single unrepeated pair, and the first call by
  7 %. It stays on the unmerged local branch `fastmace/tile-grid-fit` (commit 5a162b6c); no decision was recorded.
