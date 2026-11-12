# Go/no-go at 397.5M SrTiO3 (Alex 4985889, 12 November 2026)

Plan step 3 (`plans/billion-atoms-20261111.md`). The 397.5M record's run again with steps 1-2 in: 430^3 =
397,535,000 atoms, OMAT-0 medium, `--mode auto --tile-list --host-state --alias-state --species-index --wavefront on
--calls 2 --state-dtype auto --host-positions`, source `e7ac9b42` (before the first-call probe and the chunked
supercell). One A100-SXM4-80GB (node a0934, driver 615.71.10), **16 CPUs**, cgroup 231.5 GB, torch 2.14.1+cu130.
Job script `../jobs/alex_go.sh`. Record: `scan/tiled-SrTiO3-397535000-R8192.json`; `run.log`, `mem.log`,
`cap-430.log`. The job took 3 h 57 min (build 785 s).

| Call | Mode, cap | Time | us/atom | Host state | GPU total peak | Before the first tile | Host peak (process) |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| 0 | eager, R 8,192 | 10,301 s | 25.91 | float32 ("the call's dtype after dropping the kept lists (117.4 GiB)") | 36.93 GiB, 99.8 B/atom | 98.4 B/atom | 149.6 GB |
| 1 | compiled, R 16,384 -> **65,536** | 2,449 s | **6.16** | float32 ("the call's dtype: fits") | 31.56 GiB, 85.2 B/atom | 75.7 B/atom | 185.7 GB |

The compiled call's cap: "41.80 GiB left at the first call's peak >= the compiled residency and margin (1.07 GiB)".
The cgroup's peak was 178,395 MiB (187.1 GB) of 231.5 GB. nvidia-smi showed at most 43,193 MiB (the context and the
allocator's reserve included).

Valid: 29,417,590,000 edges (74.0 per atom); energy -7.98954679 eV/atom; **max |F| 4.37e-6 eV/A** on the perfect
crystal (the tile-local frame; the pre-local-frame source showed 1.32e-3, `../heaptrim-alex-4981047/`); stress
diagonal -0.028883; the forces check passed.

## Against the pass line

| Criterion | Line | Measured | |
| --- | --- | --- | --- |
| GPU peak | at most 28 GiB (about 75 B/atom) | 36.93 GiB (99.8 B/atom) in the eager first call; 31.56 GiB (85.2) in the compiled call | **missed** |
| Valid | 74 edges/atom, -7.98955 eV/atom | 74.0, -7.989547 | pass |
| Compiled call | not slower than 9.84 us/atom by more than 3 % | 6.16 us/atom (0.63x the record's time) | pass |
| State dtype | | float32 in both calls (the heap trim; no float16) | |
| Max force on the perfect crystal | about 3e-6 eV/A | 4.37e-6 | pass |

**Where the device memory is** (per atom; the call's ledgers after each call and the peak counters):

- Before each call: 27.4 B/atom (call 0) and 31.4 (call 1). Live: float32 positions 12, `field.neighbours.sender_groups`
  **int64** 8 (the uint8 copy exists beside it), `field._block_order` int32 4, the uint8 species index 2; after call
  0 also `tiled._g.pos` float32 12 (a second positions copy kept by the call) and `tiled._g.node_es` 4.
- Between the call's start and its first tile: +71 B/atom in the eager first call (to 98.4), +44 in the compiled
  call (to 75.7). Step 1 took the compiled call's 101 B/atom (the record) to 75.7, a cut of 25 of the projected 34;
  the eager first call, which the record's census did not trace, is now the device peak.
- The tiles' own workspace: 1.4 B/atom eager (R 8,192), 9.6 compiled (R 65,536, the larger cap).

At 10^9 atoms the eager first call's 99.8 B/atom is about 100 GB, above the A100-80's 85.9 GB; the compiled call's
85.2 is about 85 GB, at the edge. Per the plan: stop, and census the first call's transient (the 71 B/atom between
its start and its first tile, and why the first call holds 23 B/atom more than the second) before step 4. Leads
from the ledgers: the int64 sender groups (8), the second float32 positions (12), the block order (4).

**Against the record (Alex 4973640, 1 CPU, before step 1 and the heap trim):** compiled call 2,449 s against 3,913 s
(the larger receiver cap, 65,536 against 16,384); first call 10,301 s against 11,468 s; host peak 185.7 GB against
229.1 GB; float32 state in both calls against float16 in the compiled call; GPU peak 36.9 GiB against 40.6 GiB.
