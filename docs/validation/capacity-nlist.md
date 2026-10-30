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

**Step time: no change.** In the same-process A/B, in both build orders, b/a is 0.997–0.998 at 1,944 NbCl5 atoms and
1.000–1.004 at 9,216. The work per step is the compaction. It now reads 3 unit-shift floats per slot, instead of
3 float64 shifts and 3 float32 shifts, and computes the shifts from them.

**The force/stress gate passed** on NaCl, NbCl5 and NaNbO3 with the default plan.

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

- **`SkinNeighbourList`** (`fastmace/neighbors/skin.py`), unconditionally:
  - `candidate_index` is int32;
  - `candidate_shifts` and `candidate_shifts64` are gone;
  - each build writes the cell it used into `build_cell`, a new buffer for the fixed cell, registered like the other
    list buffers (reset by a rebuild) and copied by MC trial lists;
  - `compact` computes the float64 shifts as `times_3x3(unit_shifts, build_cell)`. That is the function and the
    operand order the build used to store them, so they are bitwise the stored ones. The variable-cell path already
    worked this way, with the current cell.
  - **The A/B switch is removed.** Up to commit `787e44df`, a module constant `LEAN_BUFFERS = False` kept the layout
    before, for the A/B and the trajectory comparison. The A/B and the bitwise comparisons below were measured
    there. The constant was then removed (conductor, 30 October 2026), because the layering test forbids
    module-level switches (`test_layering.py`, `no_module_switches`). There was no reason to keep the old layout,
    so the lean path is unconditional and the old one is deleted, kernels included. Tests at the final commit check
    the lean list against the research-engine baseline and against numpy's products, not against the deleted
    layout.
- **The chunked build** (`build_gpu`):
  - the screen's int64 codes (`pair_search.screen_codes`, `decode_codes`) are decided in float64 in chunks of
    `BUILD_CHUNK` = 2^18 pairs, into one int64 sort key;
  - after the sort, the kept pairs are decoded and written in chunks again;
  - elementwise work per pair, so the result does not depend on the chunk size (tested with 37 pairs).
- **The fused integrator's compaction kernels** (`fastmace/kernels/triton/fused_integrator.py`):
  - they read the int32 indices and compute the shift from the unit shifts and `build_cell`, with FMA contraction
    off, in `times_3x3`'s order;
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
- side b: the default;
- source: commit `f7f0d176`, which still had the constant (removed since; see "What changed").

Median step in ms. Order `ab` uses the abab block pattern and order `ba` uses abba. Ratios are b/a, with the range over
block pairs in brackets.

| Cell | Blocks | order ab: a / b, b/a | order ba: a / b, b/a |
| --- | --- | --- | --- |
| NbCl5 3×3×3, 1,944 atoms | 12 × 300 steps | 17.91 / 17.87, **0.997** (0.992–1.007) | 17.80 / 17.76, **0.998** (0.986–1.016) |
| NbCl5 4×4×8, 9,216 atoms | 8 × 100 steps | 88.25 / 88.24, **1.000** (0.995–1.011) | 88.35 / 88.73, **1.004** (0.993–1.011) |

Every run was valid: nothing compiled while timed, and no other process used the GPU.

- **Rebuilds** happen in the timed steps on both sides, every 10–20 steps at 300 K. Before the timing, the
  constant is restored, so both sides' rebuilds take the chunked path. Each list keeps its layout from
  construction. The rebuild's cost against the one-chunk build was not timed separately; at these sizes it is one
  or two chunks.
- **Setup** (build plus `prepare`, the side built second, one sample each):
  - 3×3×3: 13.0 s lean, 13.3 s before;
  - 4×4×8: 20.0 s lean, 14.8 s before.
  - The 4×4×8 difference is a single pair of samples. The lean list adds no work at build: the chunked build is
    two chunks there, and the captures are the same. It is noted, not explained.

The separate-process step times of the capacity points (625 against 528 ms at 49,248 NbCl5 atoms) are not a
comparison. Those are single processes near the memory limit, and the A/B is the measurement.

## Exactness and tests

- **Exact by construction**: the same integer unit shifts and the same `times_3x3` arithmetic, done in `compact`
  instead of the build. Every list test compares the compacted edges byte for byte.
- **After the switch was removed** (commit `6f802986`):
  - **CPU, on leto** (busy check 0, GPU hidden, `-n 8`, `FASTMACE_BASELINE` set): **438 passed, 0 skipped**. The
    files are those listed below plus `test_layering.py`.
    - `test_neighbors_lean.py` checks the layout (60 B per slot, int32 indices, `build_cell`).
    - It also checks that the compacted shifts equal numpy's `unit_shifts @ cell`, cast, on every test structure, both
      builds and both precisions, padding included.
    - It also checks chunk independence and MC trial lists.
    - The baseline tests in `test_neighbors.py` compare the shifts `compact` computes with the research engine's
      stored ones.
  - **GPU, local RTX 5060 Ti:** **89 passed**, 60 skipped (the GPU baseline comparisons). The fused stage still
    writes the PyTorch stage's bytes. The chunked Triton build equals the build in one chunk, and the build's peak
    stays below 64 B per pair.
- **Up to commit `787e44df`, with the old layout still selectable.**
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
- **Force/stress gate, default plan** (`tools/force_stress_gate.py --engine fastmace`, commit `70830c01`): **passed** on
  all three cells. Error against float64, with its ratio to the float32 ASE calculator's error in brackets:

  | Cell | energy (eV) | forces (eV/Å) | stress (eV/Å³) |
  | --- | ---: | ---: | ---: |
  | NaCl | 1.3e-4 (0.17) | 3.6e-6 (0.92) | 3.5e-8 (0.38) |
  | NbCl5 | 5.6e-5 (0.08) | 8.3e-5 (1.04) | 3.9e-8 (0.37) |
  | NaNbO3 | 6.1e-4 (0.49) | 1.4e-4 (1.02) | 2.1e-7 (0.69) |

- **NVE drift: not run.** The trajectories are bitwise those of the layout before.

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
  - the gate: `gate-metadata.json` and `gate-summary.txt`;
  - the job log of every reservation: `jobs.log`;
  - the A/Bs: `ab/`;
  - the run script: `local.sh`.
