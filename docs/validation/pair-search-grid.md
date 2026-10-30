# The GPU pair screen beyond 2.1M atoms (pair-search grid fix)

The conductor's go of 31 October 2026, after the tiled capacity path's T2 ([capacity-tiled-t1t2.md](capacity-tiled-t1t2.md)).
The GPU neighbour build failed at 2,299,968 NbCl5 atoms with `Triton Error [CUDA]: invalid argument`.

## The cause and the fix

**The cause.** `pair_search.screen_codes` (and `screen_into`, the asynchronous build's) launched the screening
kernel on a 2-D grid of `cdiv(N, 32)` sender tiles × `cdiv(N, 32)` receiver tiles. CUDA's grid y dimension stops at
65,535 blocks, so the launch was invalid from 32 × 65,535 = 2,097,120 atoms. A 1-D grid would not do either:
`cdiv(N, 32)²` blocks pass 2³¹ − 1 near 1.48M atoms.

**The fix.**

- Sender tiles stay on the x axis, which allows 2³¹ − 1 blocks.
- The receiver tile is `program_id(1) + program_id(2) × grid_y`, with `grid_y = min(tiles, 65,535)` and as many z
  layers as needed (`_grid`).
- Tiles past the end of the last layer have no receivers: they are masked, as the last partial tile always was.
- The half list's diagonal test uses the same receiver tile.
- **Below 65,535 receiver tiles** (2.1M atoms) the grid is exactly the old 2-D one, `(tiles, tiles, 1)`.
- **The limit** is now 32 × 65,535² atoms, far beyond memory.
- The pairs are the same. Their order in the output was already that of the atomic append, and every caller sorts
  them.

`screen_codes`, `screen_into` and `candidate_pairs` take `max_grid_y` (default 65,535), so a test can force the
split on a small cell.

## Tests

- **CPU** (`test_neighbors.py::test_pair_search_grid_covers_every_receiver_tile_once`): the grid mapping covers
  every receiver tile exactly once, for 1 to 2,299,968 atoms and for forced limits of 1 to 32 tiles. The grid is
  the old 2-D one up to 65,535 tiles.
- **GPU** (`test_neighbors_gpu.py::test_pair_search_split_grid_finds_the_same_pairs`): NbCl5 2×2×2 (288 atoms, 9
  receiver tiles), full and half screens, with `max_grid_y` 1, 3 and 7. The codes equal the unsplit grid's as a
  sorted set, each once.
- **Runs on murphy:**
  - CPU (GPU hidden, `mace_cueq_latest`): `test_neighbors.py`, `test_layering.py` and `test_d3_list.py`,
    **103 passed** (91 CUDA-only cases skipped).
  - GPU: `test_neighbors_gpu.py`, `test_neighbors_lean_gpu.py`, `test_d3_list_gpu.py` and
    `test_async_rebuild_gpu.py`, with `FASTMACE_BASELINE` set to the research-engine export, **153 passed**.

## Same-process A/B: no slowdown

**Tool:** `tools/bench/pair_search_ab.py`.

- It loads main's `pair_search.py` (main `48af40c6`) beside this tree's, in one process.
- The cells are NbCl5 rattled by up to 0.05 Å, with the 27 nearest images at radius 6.3 Å.
- 10 rounds of ABBA, CUDA events, medians.
- Records: `results/pair-search-grid-20261031/pair-search-ab.json`.
- murphy, RTX 4000 Ada.

| Cell | Atoms | Screened pairs | Main | This branch | Ratio | Same codes | Whole rebuild (this branch) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| NbCl5 4×4×8 | 9,216 | 328,844 | 3.969 ms | 3.965 ms | 0.999 | yes | 4.70 ms |
| NbCl5 6×6×20 | 51,840 | 1,850,420 | 114.79 ms | 114.88 ms | 1.001 | yes | 119.3 ms |

## NbCl5 past 2.1M atoms with the tiled path

(running)
