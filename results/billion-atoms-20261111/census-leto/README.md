# Census of the tiled call's first-call transient (leto, 13 November 2026)

Why the 397.5M go/no-go's eager first call (Alex 4985889, `../go-alex-4985889/`) held +71 B/atom between its start
and its first tile, against +44 in the compiled call. On leto's RTX 4000 Ada (20 GB, driver as in each run's
`idle_check.txt`), the NAS `mace_cueq_oeq` environment, source `65cc05bf` (main after the go/no-go). The go/no-go's
call at SrTiO3 160^3 (20,480,000 atoms): `tiled_capacity.py --structure SrTiO3 --model omat0-medium --mode auto
--tile-list --host-state --alias-state --species-index --wavefront on --calls 2 --state-dtype auto --host-positions
--first-call-probe off --ledger --call-history CALL` (the CUDA allocator's trace in call CALL; `call_peak` in each
record lists the allocations live at that call's peak by source line). Job script `../jobs/ws_census.sh`.

| Run | List reuse | Call traced | Trace | Before the call | Before the first tile (eager / compiled) |
| --- | --- | --- | --- | ---: | ---: |
| `call0/` | on (default) | 0 | overflowed (4M entries): incomplete | 48.8 B/atom | 128.5 / 157.7 |
| `call1/` | on | 1 | complete | 52.9 | 128.5 / 157.7 |
| `nr-call0/` | **off** (as the 397.5M run's host floor left it) | 0 | complete (10.4M entries) | 48.8 | 103.7 / 131.0 |
| `nr-call1/` | off | 1 | complete | 52.9 | 103.7 / 131.0 |

**Finding.** With list reuse on, the call's start holds 24 B/atom more (loop base 119.8 against 96.3 B/atom in the
eager call): `tiled.py` `_source`'s float64 copy of every atom's positions on the device, for the kept lists'
distance test (`call1/`: 24.00 B/atom at `tiled.py:1708`), made even with `--host-positions`. At 397.5M the first
call made it and only then dropped the kept lists at the host floor ("off: the host state (64.8 GiB) fits only
without them"), so its peak carried it: 24 B/atom, plus `node_es` (4, allocated in the first call, before the second).
That is the eager call's +27 over the compiled call's +44.

The rest of the transient, the same in both calls (`nr-call0/`, `nr-call1/`; per atom, not the size-independent
tables and tile workspace that dominate at 20M):

| B/atom | Source |
| ---: | --- |
| 12 | the call's forces (`tiled.py` `_globals`) |
| 8 | the plan's order, int64 (`plan_tiles`) |
| 19.5 | the tile lists' cell binning (`tile_list.py` `bin`: int16 [N, 3] cells and bins, order, starts) |
| 4 | `local_of` (int32, `_call`) |
| 4 | `node_es` (first call) |

43.5 B/atom: the compiled call's +44 at 397.5M. The held-back cuts of step 1 (the plan's order in int32, 4; the
atom bins recomputed per tile, 6) are what is left on this list.

Resident before the call (the ledgers): an int64 `field.neighbours.sender_groups` (8 B/atom) beside the uint8 copy,
and on the A100 the unbound BLOCK_ORDER order `field._block_order` (4; never built on the Ada). `tiled._g.pos` is a
view of `field.positions`, not a copy; the second float32 [N, 3] in the 397.5M ledgers is the call's returned forces.

**Note:** another session ran short GPU work on leto from 23:51:28 to 23:52:35 on 12 November, in `call0/`'s build
(its own report). The allocator trace and peaks are per process; `call0/`'s seconds are not used.
