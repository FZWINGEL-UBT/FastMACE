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

**The run.** `tiled_capacity.py point`, NbCl5 22×22×66 (2,299,968 atoms), R = 4,096, on murphy. The source is
`fastmace/cap-tiled-t1t2` (`f7125bcc`, the tiled path with T2) with this branch's `pair_search.py`
(`results/pair-search-grid-20261031/scan.sh`; records in `scan/`).

**The screen now launches.** The rebuild of the candidate pairs completes at 2,299,968 atoms, where it failed before
with `invalid argument`.

**The build then runs out of memory one step later**, in `SkinNeighbourList.compact`:

- `shifts64 = times_3x3(candidate_unit_shifts.to(float64), cell)` asks for 2.01 GiB with 1.75 GiB free;
- 17.5 GiB is allocated by then.
- `compact` forms float64 shifts and vectors for every candidate pair (within cutoff + skin).

That is the compaction part the conductor left for T3 ("the list build needs tiling too"). So:

- **The NbCl5 limit of the tiled path stays at 2,000,376 atoms on 20 GB.** It is set by the compaction now, not
  the launch.
- **The next step** is a chunked compaction, or one in float32 with a float64 decision near the cutoff only.
  It belongs to T3 and is not done here.

## What changed

- `fastmace/neighbors/pair_search.py`: `MAX_GRID_Y` and `_grid`; the kernel's receiver tile is `y + z × grid_y`.
  `screen_codes`, `screen_into` and `candidate_pairs` take `max_grid_y`.
- Tests: `tests/cpu/test_neighbors.py` (the grid mapping) and `tests/gpu/test_neighbors_gpu.py` (the same pairs
  with the split forced).
- `tools/bench/pair_search_ab.py` (new): the same-process A/B against main's screen.
- `results/pair-search-grid-20261031/`: the job scripts, the A/B record, the idle checks and the scan record.

## Setup

- **Machine:** murphy, RTX 4000 Ada Generation 20 GB (driver 595.99.02).
- **Environments:** NAS `mace_cueq_oeq` for the A/B and the scan, `mace_cueq_latest` for the tests.
- **Idle check** before every job.
- **FastMACE:** branch `fastmace/pair-search-grid` from main `48af40c6`.
- **When:** 31 October 2026, night CET.

## The launch-overflow fix (31 October 2026)

### The bug

**Triton's launcher skipped large screens without an error.** Its C launcher (`triton/backends/nvidia/driver.c`,
`_launch(int gridX, int gridY, int gridZ, ...)`) launches only `if (gridX * gridY * gridZ > 0)`, a product in
32-bit `int`. The screen's `cdiv(N, 32)` × `cdiv(N, 32)` blocks pass 2³¹ above 46,340 tiles (1.48M atoms).

- The product wrapped to zero or a negative number, so the kernel never ran, nothing failed, and the list had no
  pairs.
- At NbCl5 1,728,000 and 2,000,376 atoms the tiled capacity scan therefore "ran" with zero edges and zero forces
  ([capacity-tiled-t1t2.md](capacity-tiled-t1t2.md), correction).
- With the y/z split above (main `734688e6`) the product at 2.3M atoms happened to wrap to a positive value, and
  that run was correct.
- Every GPU-built list above 1.48M atoms was at risk, on any path. No path reached that size before the tiled one.

### The fix (`fastmace/neighbors/pair_search.py`)

- **The grid.** `_grid` keeps every launch at or below `MAX_GRID_BLOCKS` = 2³¹ − 1 blocks, and within CUDA's y
  limit: `grid_y = min(tiles, 65,535, (2³¹ − 1) // tiles)`. Beyond that, a program screens the receiver tiles
  `program_id(1) + k * grid_y` for `k < rounds`. `_grid` raises if a launch would still exceed the bound.
- **The kernel.** The per-tile body is a `@triton.jit` helper. A constexpr picks the single-round program (up to
  46,340 tiles), which is the code of the 2-D grid of before, or the loop.
- **An empty screen is checked** (`check_found`):
  - the check runs only when the screen found no pair;
  - it is a brute-force test of atom 0 against every atom and image, and raises if atom 0 has a neighbour within
    range;
  - the synchronous build checks after its count, the asynchronous one where its count is read;
  - a cell with genuinely no pair in range passes.
- **Why no in-kernel marker.** A marker written by the kernel's first program, the first version, cost 6 % of the
  screen (`marker_cost.py`: 1.061× main with the store, 0.9996× without), so it was dropped.
- **The scan tools check correctness** (`capacity_scan.check_point`, used by `capacity_scan.py` and
  `tiled_capacity.py`):
  - a point counts only if its list has the structure's edges per atom within 5 % (the reference is ASE's list of a
    small copy) and its largest force is nonzero;
  - otherwise its status is "invalid" and the scan stops;
  - `tiled_capacity.py` also records the memory allocated before each call and the call's true peak.

### Audit of the other launches (at 10M atoms and 300M edges)

| Launch | Grid | Largest grid at 10M atoms / 300M edges | Silent-skip risk |
| --- | --- | ---: | --- |
| `pair_search` screen (also the D3 half list) | 2-D, cdiv(N, 32)² | 9.8e10 before; now ≤ 2³¹ − 1 | **fixed here** |
| `segment_gemm` | cdiv(rows, ≥16) × cdiv(n, 64) | rows ≤ 7N: about 1.4e8 | no |
| `tf32x3_gemm` | cdiv(m, 64) × cdiv(n, 64) | about 1e7 | no |
| `gated_products` | cdiv(atoms, 32) × cdiv(512, 64) | 2.5e6 | no |
| `w_table` forward | cdiv(edges, 16) × cdiv(1,280, 128) | 1.9e8 | no (2³¹ at 3.4e9 edges) |
| `w_table` backward, `edge_outer_sum`, `radial_table`, `fused_radial` | 1-D over edges or rows, ≥ 16 per program | ≤ 1.9e7 | no |
| `fused_integrator`, `skinny_linear`, `grouped_gemm`, `d3_kernel` | 1-D over atoms, rows or tiles | ≤ 1.2e8 | no |
| NVRTC kernels (OEq streaming, fused forward, wprime/wrebuild backward) | `cuLaunchKernel`, 1-D | ≤ about 3e8 (edges × cohort / threads) | no: the driver rejects an invalid grid with a checked error |

Only the pair screen could exceed 2³¹ blocks.

### Checks

- **CPU** (murphy, GPU hidden, `mace_cueq_latest`, `-n 8`), `7fa546b4`: **108 passed**, 91 CUDA-only cases skipped.
  The files: `test_capacity_check.py` (new), `test_neighbors.py` (the launch bound from 1 to 10M atoms and the
  empty-screen check), `test_d3_list.py` and `test_layering.py`.
- **GPU** (murphy, research-engine baseline set): `test_neighbors_gpu.py` (the multi-round grid forced with
  `max_grid_y` 1, 3 and 7, and the chunked compact), `test_neighbors_lean_gpu.py`, `test_d3_list_gpu.py` and
  `test_async_rebuild_gpu.py`, **153 passed**.
- **Same-process A/B against main `bacf6e07`** (`pair_search_ab.py`, 10 rounds; records in
  `results/pair-search-grid-20261031/launch/`):

| Cell | Atoms | Main | This branch | Ratio | Codes |
| --- | ---: | ---: | ---: | ---: | --- |
| NbCl5 4×4×8 | 9,216 | 3.956 ms | 3.956 ms | 1.000 | the same 328,844 |
| NbCl5 6×6×20 | 51,840 | 115.42 ms | 115.49 ms | 1.001 | the same 1,850,420 |
| NbCl5 20×20×60 | 1,728,000 | 0.29 ms, **no pairs** (launch skipped) | 203.5 s, 61,682,962 codes | | main's are empty |

