# CAP-T09: a leaner neighbour list

Plan task CAP-T09 ([plans/capacity.md](../../plans/capacity.md)). Owner, 30 October 2026, through the conductor:
improve the neighbour list so that MH-1 float32 fits more atoms on the 16 GB RTX 5060 Ti. The plan had estimated
about 0.3 %, from int32 indices alone.

## Verdict

**Exact, and the default.** Two changes, both bitwise:

- the list stores 60 instead of 104 bytes per slot;
- a rebuild's temporaries drop from about 172 to about 51 bytes per candidate pair.

**New limit for NbCl5: 49,248 → 51,840 atoms (6×6×20), +5.3 %, one grid step.** At 51,840 atoms `main` runs out of
memory in the warm-up, as it did in CAP-T07b. The branch runs at 562 ms per step with at most 14.85 GiB reserved.
54,432 atoms (6×6×21) fails at build.

**NaCl: no new grid point.** The limit is 49,248 atoms (18×18×19) on both sides. CAP-T07b's 46,656 was a grid
artefact: its scan went from 18³ straight to 19³. At 49,248 atoms the branch's peak is 0.52 GiB lower (14.17 against
14.69 GiB reserved). On `main`, one more rebuild after the run ran out of memory. 51,984 atoms (18×19×19) fails in
the warm-up with the change.

**Step time: no change.** In the same-process A/B the ratio is RESULT_AB. The work per step is the compaction, and
it now reads 3 unit-shift floats per slot instead of 3 float64 shifts and 3 float32 shifts.

## Where the neighbour list's memory was

Measured with `tools/bench/capacity_scan.py point`, which now records the list's bytes by buffer, the screen's
output buffer, and one more rebuild at the end with its peak above the live memory. MH-1 float32, default plan,
20 warm-up and 200 timed steps, one fresh process per point. Before is `main` at `c20ddf31`; after is the branch.

| NbCl5 6×6×19, 49,248 atoms, 1.29M edges at the cutoff, 1.92M slots | before | after |
| --- | ---: | ---: |
| `candidate_index` | 30.8 MB (int64) | 15.4 MB (int32) |
| `candidate_unit_shifts` | 23.1 MB | 23.1 MB |
| `candidate_shifts` (float32) | 23.1 MB | — |
| `candidate_shifts64` | 46.1 MB | — |
| `edge_index`, `unit_shifts`, `shifts` (what the model reads) | 76.9 MB | 76.9 MB |
| **List buffers** | **201 MB, 104 B per slot** | **117 MB, 60 B per slot** |
| Screen output buffer (128 codes per atom) | 50 MB | 50 MB |
| **One rebuild, peak above the live memory** | **296 MB** (172 B per pair) | **88 MB** (51 B per pair) |
| Reserved memory a rebuild adds | 273 MB | 84 MB |
| Largest reserved: build / warm-up / steady | 13.91 / 14.24 / 14.32 GiB | 13.80 / 13.90 / 14.11 GiB |

At this size the force call allocates up to 12.0 GiB. Before the change, the list and the screen buffer were
0.25 GB of it (2 %). A rebuild's temporaries came on top: they run between replays, outside the graph pool, so they
stay reserved beside it. That makes about 0.55 GB in all, of which the change removes about 0.29 GB.

| NaCl 18×18×19, 49,248 atoms, 1.58M edges at the cutoff, 3.17M slots | before | after |
| --- | ---: | ---: |
| List buffers | 331 MB | 192 MB |
| One rebuild, peak above the live memory | out of memory | 118 MB |
| Largest reserved | 14.69 GiB | 14.17 GiB |

NaCl holds twice as many slots as edges. Rule 11 sizes the capacity headroom for the thermal spread of the 6.3 Å
shell, so the per-slot saving counts double per edge. That rule is not changed here.

## What changed

- **`SkinNeighbourList` with `LEAN_BUFFERS`** (`fastmace/neighbors/skin.py`, the default):
  - `candidate_index` is int32;
  - `candidate_shifts` and `candidate_shifts64` are gone;
  - each build writes the cell it used into `build_cell`, a new buffer for the fixed cell, registered like the other
    list buffers (reset by a rebuild) and copied by MC trial lists;
  - `compact` computes the float64 shifts as `times_3x3(unit_shifts, build_cell)`. That is the function and the
    operand order the build used to store them, so they are bitwise the stored ones. The variable-cell path already
    worked this way, with the current cell.
  - `LEAN_BUFFERS = False` keeps the layout before. It is read when a list is constructed, for same-process A/Bs
    only.
- **The chunked build** (`build_gpu`):
  - the screen's int64 codes (`pair_search.screen_codes`, `decode_codes`) are decided in float64 in chunks of
    `BUILD_CHUNK` = 2^18 pairs, into one int64 sort key;
  - after the sort, the kept pairs are decoded and written in chunks again;
  - elementwise work per pair, so the result does not depend on the chunk size (tested with 37 pairs).
- **The fused integrator's compaction kernels** (`fastmace/kernels/triton/fused_integrator.py`):
  - a `LEAN` variant reads the int32 indices and computes the shift from the unit shifts and `build_cell`, with FMA
    contraction off, in `times_3x3`'s order;
  - the float32 edge shifts are that value cast, as `compact` does.
- **`pair_search`**: `candidate_pairs` is now `decode_codes(screen_codes(...))`. When its output buffer grows, the
  old one is freed before the new one is allocated.
- **Not changed:**
  - the model's inputs (`edge_index` stays int64);
  - the capacity rule (rule 11);
  - the screen buffer's size;
  - the asynchronous early build's layout of work (only its writes follow the lean buffers).

### Tried in thought and left out

- **Narrower unit shifts** (int8 or int16, 3–6 B per slot):
  - unwrapped positions of a diffusing liquid can make a pair's image shift exceed int8's range in small cells;
  - int16 would need a range check per build and a fallback;
  - about 10 MB at these sizes.
- **Dropping `unit_shifts` from the edge buffers in fixed-cell runs** (12 B per slot): the model's inputs and the MC
  float64 copy (`follow`) read them.
- **Smaller screen buffer** (128 codes per atom against about 35 used by NbCl5): about 35 MB. The asynchronous build
  relies on its size, and resizing it after the first build adds another reallocation path.

## Step time

`tools/ab_rewrites.py --prepare`, same process, MH-1 hybrid float32 NVT at 300 K, default plan on both sides:

- side a: `fastmace.neighbors.skin:LEAN_BUFFERS = False` (the layout before; its build runs in one chunk);
- side b: the default.

Median step in ms. Order `ab` uses the abab block pattern and order `ba` uses abba. Ratios are b/a, with the range over
block pairs in brackets.

RESULT_TABLE

## Exactness and tests

- **Exact by construction**: the same integer unit shifts and the same `times_3x3` arithmetic, done in `compact`
  instead of the build. Every list test compares the compacted edges byte for byte.
- **CPU, on coraline** (GPU hidden, `-n 8`, commit `93a7cc6d`, with `FASTMACE_BASELINE` set): **465 passed, 0
  skipped**. The files: `test_neighbors.py` (including the research-engine baseline comparisons, now on the compacted
  edges, the candidates and the shifts `compact` computes), the new `test_neighbors_lean.py`, `test_d3_list`,
  `test_fused_integrator`, `test_mc_displace`, `test_registry`, `test_engine`, `test_forces`,
  `test_canonical_restart`, `test_superset`, `test_cell_divergence` and `test_rule_11`.
  - `test_neighbors_lean.py`: the lean list against the layout before, for both builds, three sort modes and both
    precisions; the variable cell; chunk independence; MC trial lists.
  - The CPU (matscipy) build stored numpy's `unit_shifts @ cell`. With the lean list, `compact` uses `times_3x3`
    for those builds too. Both were bitwise equal on every test structure, and baseline comparison included.
- **GPU, local RTX 5060 Ti** (`mace_cueq_latest`'s pytest, commits `93a7cc6d` and `b53e0ed9`): **91 passed**, 60
  skipped (the GPU baseline comparisons; no baseline export there). The files: `test_neighbors_lean_gpu.py` (new),
  `test_neighbors_gpu.py`, `test_async_rebuild_gpu.py`, `test_d3_list_gpu.py`, and the fused integrator's stage
  test.
  - **Lean, chunked Triton build against the layout before:** bitwise, on four structures, three sort modes and both
    precisions.
  - **The build's peak** is below 64 B per pair.
  - **The fused stage:** with the lean kernels it writes the PyTorch stage's bytes.
  - **MD trajectories:** MH-1 NaCl 3×3×3, 400 steps with 105 rebuilds, reproducible float64, with the fused stage
    and with the PyTorch stage. Positions and momenta are bitwise equal with `LEAN_BUFFERS` on and off.
  - **One test fixed along the way:** the first run of that MD test used the float32 default plan. Its atomic
    additions differ between processes, so the trajectories differed with the option on and off. The test now uses
    the deterministic process, like `test_fused_integrator_gpu.py`.
- No force/stress gate or NVE drift: nothing changes numerically.

## Setup

- **Machine:** Fabi-PC, WSL2, RTX 5060 Ti 16 GB (driver 610.88), `mace_cueq_oeq` (torch 2.14.1+cu130).
- **GPU access:** `tools/ci/local_gpu.py run` reservations, with the idle check before every job; 14.80 GiB free at
  the start of every point.
- **CPU tests:** coraline. A second user had a login session without jobs, and nothing was on its GPU; the pytest
  busy check was 0.
- **Sources:** pinned copies of FastMACE `main` `c20ddf31` (before) and the branch `b53e0ed9`/`f7f0d176` (after,
  with the branch's `capacity_scan.py` for both sides). No baseline, since both sides are FastMACE.
- **Model:** MH-1 (`mh1-r2scan`), hybrid backend, float32, Langevin NVT at 300 K.
- **Raw records** in `results/cap-nlist-20261030/`:
  - the capacity points: `points/`;
  - the A/Bs: `ab/`;
  - the run script: `local.sh`.
