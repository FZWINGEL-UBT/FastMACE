# Setup speed-ups for the billion-atom run (Alex 4985954, 11 November 2026)

Source `0f142386` (branch `fastmace/fast-setup`), one A100-SXM4-80GB (not a whole node), 16 CPUs, the cgroup's
231.5 GB. Job script: `../jobs/alex_setup.sh`. Plan: `plans/billion-atoms-20261111.md`, "Before step 4" 1 and 2.

## 1. Chunked supercell builder (`tools/bench/tiled_capacity.py` `supercell`)

`supercell.jsonl`; SrTiO3, rattle 0.05, seed 0. Max RSS of the process.

| Repeat | Atoms | Builder | ASE's path (`supercell_ase`) | Max RSS |
| --- | ---: | ---: | ---: | ---: |
| 300^3 | 135,000,000 | 18.6 s | 205.9 s, bitwise equal (positions, numbers, cell) | 4.4 GB |
| 430^3 | 397,535,000 | 54.6 s | (record 4973640: about 10 min with plan and checks, one CPU) | 12.6 GB |
| 520^3 | 703,040,000 | 94.7 s | (the 519M segmentation fault came out of this path) | 22.2 GB |
| 585^3 | 1,001,008,125 | 135.0 s | | 31.5 GB |

## 2. `TILE_FIRST_CALL_PROBE` A/B (`probe-off-*`, `probe-on-*`)

`tiled_capacity.py --mode auto --tile-list --host-state --alias-state --species-index --wavefront on --calls 2
--host-positions` (the go/no-go's call), `--first-call-probe off` / `on`, one process each.

| Atoms | Workspace: whole call / probe (bytes) | Ratio | Mode, cap (both) | First call: off / on | Two calls: off / on |
| --- | --- | ---: | --- | ---: | ---: |
| 975,560 (58^3) | 527,284,736 / 526,535,168 | 0.9986 | compiled, 65,536 | 30.9 / 65.5 s | 139.1 / 68.4 s |
| 20,480,000 (160^3) | 503,742,464 / 502,819,328 | 0.9982 | compiled, 65,536 | 351.5 / 215.1 s | 600.6 / 286.5 s |

The probe ran 8 tiles (24 tile sweeps): 8.3 s of tiles and 31.9 s of the wavefront's search pass at 160^3. "First
call" with the probe includes the probe, the call again and its compilation. The forces check passed in all four
runs. The probe run's second call is a second compiled call reusing the kept lists, whose tile-graph cache
(`TILE_GRAPH_CACHE`) took 19.4 GB of its 19.44 GB budget of free device memory (device peak 25.9 GB at 160^3); without
the probe that call is only the third. Adopted: on for the A100 family (`TILE_PROBE_GPUS`).
