# Tiled path: tile lists kept across MD steps (`TileConfig.list_reuse`)

**What it is.** Every tiled call bins all atoms, plans the tiles, and searches each tile's neighbour list once; the
list cache serves the later sweeps. In MD, almost the same lists are found again every step.

**Was it worth building?** The conductor approved it if the search is at least 3 % of a call. Measured with
`results/tiled-checks-20261102/checks.py` on murphy at 9.26M atoms, R = 16,384:
- the slots' planning round, which holds each tile's one search, takes **7.50 s of a 134.3 s call (5.6 %)**;
- each sweep that rebuilds the lists from the cache takes 0.67 s.

## The change

`TileConfig.list_reuse` (default off) works with `neighbours="tiles"`:

- **A call that searches.** It bins the atoms and searches each tile within cutoff + skin, using the skin of the
  force field's own list (0.3 Å for the MH-1 plans).
  - The searched superset is kept on the host for each tile, in the cache's compact form: int32 senders, narrow
    counts, one-byte shifts.
  - The tile plan and the call's positions are kept on the host too.
- **A later call reuses them** while the atom count, the cell and the receiver cap match the search, and no atom
  has moved more than skin / 2 since it. That is the whole-system list's rule (`SkinNeighbourList`).
  - The cell is compared every call, so a variable-cell list is allowed. The planner's MD fields have one: they
    compute stress, but their cell is fixed.
  - The receiver cap is compared because the budget raises it after the first call.
- **What a reusing call does.** It skips the binning, the plan and the search. Each tile's superset is filtered at
  the cutoff with a fresh search's own float64 test, on vectors from the current positions and the kept unit
  shifts, keeping its order.
  - So the edges equal a fresh search's on the same tiles, bitwise.
  - The kept tiles can differ from a fresh plan's on the moved positions. The forces are then exact, but with
    another summation order.
- **Cost.** The kept data lives on the host: the superset, about 16 % more edges than at the cutoff, plus 24 bytes
  per atom of reference positions. The device holds nothing between calls.
- **Report.** The call's report gains `list_reused` and `list_rebuilds`.

## Gates

**CPU** (`test_tile_lists_kept_within_half_the_skin_are_bitwise_a_fresh_search`):
- in eager and padded mode, with and without stress;
- after a 0.4 × skin move, pairs cross the cutoff, the lists are reused, and energy and forces are bitwise those of
  a fresh search on the same tiles;
- a 0.6 × skin move, and a new receiver cap, each search again.

`test_tiled.py`, `test_layering.py` and `test_neighbors.py` on trillian at 097eb6ca: 167 passed, 91 skipped.

**GPU** (murphy, at 097eb6ca; runs in `results/tiled-list-reuse-20261101/runs/`):

| Measurement | Without reuse | With reuse | Ratio |
| --- | ---: | ---: | ---: |
| Same-process A/B, 9.26M, compiled + budget: repeated calls (every call reuses) | 133.6 s | 127.4 s | **0.954** |
| MD at 884,736 atoms, 20 compiled steps, mean of steps 2-19 | 13.45 s | 13.06 s | **0.971** |
| The same MD, the steps that reuse | 13.40-13.49 s | 12.85-12.96 s | 0.957 |
| The same MD, the steps that search (every 4-5 steps) | | 13.61-13.71 s | 1.01 |
| NVE drift over the 20 steps | 0.05024 meV/atom | 0.05024 meV/atom | |

- **Forces gate** (the A/B's arms): energy within 7e-8 meV/atom, forces within 1.4e-6 eV/Å, which is the device's
  run-to-run spread.
- **Rebuilds.** At 600 K with the 0.3 Å skin, an atom moves more than 0.15 Å every 4-5 steps (2 fs). So MD gains
  about 3 %, while a call that reuses gains about 4.5 %.
- **A larger skin** would rebuild less often. But it makes the superset larger (about (1 + skin / cutoff)³) and the
  filter more expensive; this was not measured.

## Found on the way

- **The first GPU runs never reused.** The planner's MD field has a variable-cell list, and the first version
  allowed reuse only for a fixed cell. The cell is now compared every call.
- **The kept plan outlived the budget's switch** from R = 4,096 to 16,384. Reuse was then 19 % slower, on the old
  small tiles in new large slots. A new receiver cap now searches and plans again.

## Status

Built, gated and off by default. It gains about 3 % in MD and 4.5 % per call that reuses. Whether to turn it on by
default (for neighbours "tiles"), and whether to try a larger skin, is the conductor's call.

## On by default (conductor's decision, 1 November)

`TileConfig.list_reuse` is on by default for `neighbours="tiles"`, with a host floor.

- **`TileConfig.list_reuse_floor`:** 20 GiB of the host's `MemAvailable`. The call that searches checks it before it
  keeps the reference positions and before each tile's superset.
- **Below the floor** nothing is kept, and reuse stays off from then on. The call's report says why in
  `list_reuse`, for example "off: 12.3 GiB available on the host when keeping tile 4711's lists, below the floor of
  20.0 GiB".
- **No per-call cache beside the kept lists.** While lists are kept, the later sweeps filter the kept supersets
  again; the list cache serves only calls without kept lists. At 70.96M the cache and the supersets together would
  have left less than the floor; without the cache, 27 GiB stay free.
- **The call's report** also gives `list_kept_bytes`.

**CPU** (`test_tiled.py`, `test_layering.py`, `test_neighbors.py` on trillian): 168 passed, 91 skipped.
- New: `test_tile_lists_not_kept_below_the_host_floor`. With a floor no host meets, nothing is kept, the report
  gives the reason, and the results are bitwise the plain call's.
- The tests that count searches now set `list_reuse=False`.

**Tools:** `tiled_md.py --no-list-reuse` (`--list-reuse` is kept as a no-op) and `ab_arms.py +noreuse`.

**The gate at 70.96M** (leto, 6b479d24): `tiled_md.py --mode auto --integrator host --steps 3`, all defaults. Run:
`results/tiled-list-reuse-20261101/runs/leto-nve-reuse-NbCl5-69x69x207`.

| Step | Call | Wall time | GPU peak (of 19.54 GiB) | Host peak (of 251 GiB) | Lists |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | eager, R = 4,096 | 1,771 s | 14.21 GiB | 214.7 GiB | searched (cutoff + skin), kept |
| 1 | compiled, R = 16,384 (the budget's switch) | 1,094 s | 18.28 GiB | 221.4 GiB | searched again (new cap), kept |
| 2 | compiled | 1,000 s | 16.11 GiB | 223.8 GiB | **reused** |
| 3 | compiled | 1,000 s | 16.11 GiB | 223.8 GiB | **reused** |

- **Host memory of the kept lists:** 13.87 GB, 196 bytes per atom. That is 12.17 GB of supersets, 1.26 times the
  9.63 GB list cache at the cutoff, plus 1.70 GB of reference positions. The expectation was (1 + 0.3/6)³ = 1.16
  times the cache: the shell between 6 and 6.3 Å holds more neighbours than the average density predicts.
- **The steps that reuse take 1,000 s.** With the per-call cache and no reuse
  ([tiled-cache-compact.md](tiled-cache-compact.md)) the steps took 1,053-1,112 s, so reuse is about 7 % faster here.
  They also skip the whole-system binning, so their GPU peak is 2.2 GiB lower.
- **Drift:** 0.016 meV/atom over the 3 steps.
