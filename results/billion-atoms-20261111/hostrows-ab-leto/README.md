# Same-process A/B of TILE_HOST_ROWS_KEPT (leto, 13 November 2026)

The host-rows distance test of the kept lists (`../cuts-ab-leto/`, cut 3) against the whole-system float64 device
copy it replaces, in one process: `tools/bench/tiled_ab.py --structure SrTiO3 --model omat0-medium --repeat
160x160x160 --mode auto --host-state --alias-state --wavefront on --host-positions --a
fastmace.model.tiled:TILE_HOST_ROWS_KEPT=False --b fastmace.model.tiled:TILE_HOST_ROWS_KEPT=True --warmup 3
--blocks 3 --calls 2`, both orders, source `34d31aba` (the switch, default on then), leto's RTX 4000 Ada, the NAS
`mace_cueq_oeq` environment. Side a is main's behaviour on this path. Timed calls are compiled (R 16,384) and reuse
the kept lists. Job script `../jobs/ws_hostrows_ab.sh`.

| Order | Median a | Median b | b / a | Block pairs |
| --- | ---: | ---: | ---: | --- |
| ab | 126.00 s | 137.74 s | **1.0932** | 1.0930, 1.0934, 1.0928 |
| ba | 125.81 s | 137.47 s | **1.0927** | 1.0912, 1.0928, 1.0939 |

Energy -1.3e-7 meV/atom apart, forces within 4.6e-6 eV/A, stress within 1.9e-12: the atomics' run-to-run spread (the
same tree twice differs as much, `../cuts-ab-leto/`).

**Decision (conductor's 3 % rule):** 9.3 % slower on reused lists, so not a default everywhere. `TILE_HOST_ROWS_KEPT`
is None by default (`66b9b06d`): on only where the device copy (24 B/atom) and the call's later per-atom transient
(`TILE_HOST_ROWS_KEPT_RESERVE`, 48 B/atom from the census) exceed the device memory free at the call's start; the
decision and its reason in `last["host_rows_kept"]`. At 397.5M on the A100 it stays off (about 28.6 GB needed against
about 64 GB free), at 10^9 it turns on. The 397.5M confirmation against the 28 GiB line sets it True to measure the
cut.
