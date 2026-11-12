# Paper row: 320M SrTiO3 with float32 host state (Alex 4988550, 12 November 2026)

The go/no-go's call (`../go-alex-4985889/`) at 400^3 = 320,000,000 atoms with `--state-dtype none` (float32 state,
no fall to float16): OMAT-0 medium, `--mode auto --tile-list --host-state --alias-state --species-index --wavefront
on --calls 2 --host-positions`. Job script `../jobs/alex_f32.sh`; the first size ran, so the step-down sizes did not.
Record: `scan/tiled-SrTiO3-320000000-R8192.json`; `run.log`, `mem.log`, `cap-400x400x400.log`. The job ran 2 h 43 min
(build 638 s). It backfilled before the go/no-go ended, so it was not cancelled when the go/no-go came back float32.

**Manifest:** source `e7ac9b42`; checkpoint `mace-omat-0-medium.model` SHA-256
`d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a`; one A100-SXM4-80GB (node a0931, driver
615.71.10, 81,920 MiB), **16 CPUs**, cgroup 231.5 GB; torch 2.14.1+cu130, triton 3.8.0, cuequivariance 0.12.0,
e3nn 0.4.4, mace-torch 0.3.16, ase 3.27.0, numpy 2.2.6 (all of them in `pip-freeze.txt`); switches as above, the
source's defaults otherwise (no first-call probe in `e7ac9b42`).

| Call | Mode, cap | Time | us/atom | Host state | GPU total peak | Host peak (process) |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| 0 | eager, R 8,192 | 6,665 s | 20.83 | float32 | 29.92 GiB (100.4 B/atom) | 183.3 GB |
| 1 | compiled, R 16,384 -> **65,536** | **1,820 s** | **5.69** | float32 | 33.50 GiB (112.4 B/atom) | 209.3 GB |

The compiled call's cap: "48.81 GiB left at the first call's peak >= the compiled residency and margin (1.07 GiB)".
The cgroup's peak was 201,185 MiB (211.0 GB) of 231.5 GB; the first call kept its tile lists (100.0 GB of kept lists,
host state 55.4 GiB), the second dropped them ("20.0 GiB available on the host when keeping tile 5126's lists, below
the floor of 20.0 GiB"). nvidia-smi at most 41,273 MiB.

Valid: 23,680,000,000 edges (74.0 per atom); energy -7.98954680 eV/atom; max |F| 4.40e-6 eV/A on the perfect
crystal; stress diagonal -0.028883; the forces check passed.
