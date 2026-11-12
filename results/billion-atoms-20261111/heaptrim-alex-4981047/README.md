# 397.5M SrTiO3 with TILE_HEAP_TRIM (Alex 4981047, 12 November 2026)

The 397.5M-atom capacity record's run again (430^3 = 397,535,000 atoms, OMAT-0 medium, `--mode auto`, tile lists,
aliased host state, wavefront, species index, `--state-dtype auto`, 2 calls) on source `69a48049`, which has
`TILE_HEAP_TRIM` (glibc's free heap returned after the kept lists are dropped). One A100-SXM4-80GB, **16 CPUs**
(`nproc 16`), cgroup 231.5 GB. Positions on the device: this source predates the billion-atom plan's step 1.
Record: `cap-4981047/scan/tiled-SrTiO3-397535000-R8192.json`; `run.log`, `mem.log`, `cap-430.log`.

| Call | Mode, cap | Time | us/atom | Host state | GPU total peak | Host peak |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| 0 | eager, R 8,192 | 9,601 s | 24.15 | float32 ("the call's dtype after dropping the kept lists (117.4 GiB)") | 41.98 GiB | 159.4 GB |
| 1 | compiled, R 16,384 | 3,606 s | 9.07 | float32 ("the call's dtype: fits") | 42.13 GiB | 159.4 GB |

Valid: 29,417,590,000 edges (74.0 per atom, as expected); forces check passed.

- **Against the record (Alex 4973640, source 7e4fbf4c, 1 CPU):** that run fell to float16 state in its compiled call
  (81.3 GiB of state against 58.2 GiB above the floor), with a host peak of 229.1 GB. With the heap trim the host
  peak is 159.4 GB, and both calls keep float32.
- **CPU count:** the eager first call took 24.15 us/atom here with 16 CPUs, against 28.8 us/atom in the record with
  1 CPU (host-state gathers and writes are host work). A paper row from these runs states its CPU count.
- **Accuracy, pre-local-frame:** max |F| on the perfect crystal is **1.32e-3 eV/A**. That is the float32 coordinate
  rounding at about 1,680 A extents in source `69a48049`, which predates `TILE_LOCAL_FRAME` and
  `WHOLE_SYSTEM_FLOAT64_EDGES`. **Do not quote it as the accuracy of the current code** (expected about 3e-6 eV/A
  with the local frame; see the go/no-go, Alex 4985889, on `e7ac9b42`).
