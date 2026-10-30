# Tiled capacity path: T2 (smaller global state) and T1 (one padded CUDA graph per sweep)

The owner's approved next steps T1 and T2 of [plans/capacity-ideas-20261030.md](../../plans/capacity-ideas-20261030.md),
through the conductor, after [capacity-tiled.md](capacity-tiled.md).

## T2: keep interaction 1's reads of h1, not h1

### What changed against the plan

**The plan's T2 would not lower the peak.**

- It was to free h1 after sweep 2, since sweep 3 recomputes interaction 0.
- But the peak is in sweep 2, where h1 (read for every tile's receivers and halo) and its adjoint (accumulated
  over the tiles) are both alive.
- The out-of-memory errors at 1.06M NbCl5 and 1.0M NaCl atoms were in allocating that adjoint at the start of
  sweep 2.

**What is done instead, exact as well.** Interaction 1 reads h1 only through two linear maps, `linear_up` and
`skip_tp`. With MH-1's default `fold_product_linear` both are `FoldedConsumer`s, and h1 is the contraction output
plus the one-hot species: about 2,100 columns.

- **Sweep 1** stores their outputs, `x = linear_up(h1)` and `sc = skip_tp(h1)`, together 608 columns.
- **Sweep 2** runs interaction 1 with its `linear_up` and `skip_tp` standing in for the stored values. It
  accumulates `dE/dx` over each tile and its halo, and writes `dE/dsc` of the receivers.
- **Sweep 3** recomputes h1 per tile and backpropagates `⟨dE/dx, linear_up(h1)⟩ + ⟨dE/dsc, skip_tp(h1)⟩` with interaction
  0's own energy.

The global state falls from about 17 kB to **4.75 kB per atom** (measured: x, sc and their adjoints). This is
`fastmace/model/tiled.py`; the CPU tests compare it against the whole-system call in float64, with the stock
model and with MH-1's rewrites.

### Accuracy

NbCl5 4×4×8, 9,216 atoms, rattled by up to 0.05 Å and strained by up to 2 %, R = 4,096:

| Against | ΔE (meV/atom) | max \|ΔF\| (eV/Å) | max \|Δσ\| (eV/Å³) |
| --- | ---: | ---: | ---: |
| the whole-system eager call | −2.5e-6 | 2.3e-5 | 5.6e-9 |
| the default compiled plan | −7.8e-7 | 2.3e-5 | 6.5e-9 |

That is the same as before T2: 2.3e-5 against the compiled plan
([capacity-tiled.md](capacity-tiled.md)).

### New limits on murphy (RTX 4000 Ada 20 GB)

**Tool:** `tiled_capacity.py point --calls 1`, R = 4,096, growing until a size fails
(`results/cap-tiled-t2-20261030/t2.sh`). Records: `results/cap-tiled-t2-20261030/runs/`.

> **Correction (31 October 2026).** The 1,728,000 and 2,000,376-atom NbCl5 points below are **invalid**: their
> neighbour lists had no pairs (0 edges, all forces and the stress exactly 0). The GPU pair screen's launch had
> 54,000² and 62,512² blocks; Triton's launcher multiplies the grid as 32-bit ints and skipped the launch, without
> an error, once the product passed 2³¹. The scan checked only that the call finished. The **valid NbCl5 limit of
> T2 is 1,481,544 atoms (checked: 26.3 edges per atom, forces as at small sizes)**; the real limit lies between that
> and about 2.2M atoms. The merge message of `218937d2` quotes the invalid 2,000,376. The fix (a bounded launch grid,
> a launch marker that raises, and a correctness check in the capacity tools) is on `fastmace/pair-search-launch`;
> see [pair-search-grid.md](pair-search-grid.md).

| Cell | Atoms | Edges per atom | Call peak (GiB) | Before (GiB) | **Total (GiB)** | Global state (GiB) | Neighbour build peak (GiB) | Call |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NbCl5 16×16×48 | 884,736 | 26.28 | 5.57 | 3.61 | **9.18** (was 19.04) | 4.01 | 3.49 | 25.6 s |
| NbCl5 18×18×54 | 1,259,712 | 26.28 | 7.26 | 5.08 | 12.34 | 5.71 | 4.97 | 33.0 s |
| **NbCl5 19×19×57** | **1,481,544** | 26.28 | 8.25 | 5.95 | **14.20** | 6.71 | 5.85 | 43.8 s |
| NbCl5 20×20×60 | 1,728,000 | **0 (invalid)** | | | | | | |
| NbCl5 21×21×63 | 2,000,376 | **0 (invalid)** | | | | | | |
| NbCl5 22×22×66 | 2,299,968 | | | | the screen's launch fails (`invalid argument`) | | | |
| NaCl 50×50×50 | 1,000,000 | 32.00 | 6.25 | 5.48 | 11.73 | 4.53 | 6.50 | 29.0 s |
| **NaCl 54×54×54** | **1,259,712** | 32.00 | 7.48 | 6.86 | **14.35** | 5.71 | 8.19 | 32.5 s |
| NaCl 57×57×57 | 1,481,544 | | | | out of memory in the neighbour list's compaction | | | |

"Before" is the memory allocated before the call (after the neighbour build); the total is that plus the call's
peak.

- **NbCl5: from 884,736 to at least 1,481,544 atoms (1.7×), at 14.2 of 19.5 GiB.** Between 1.48M and 2.1M atoms
  the screen's launch was silently skipped (above); at 2,299,968 atoms the 2-D grid's y dimension (65,535 blocks)
  made it fail outright. Both are launch limits of `pair_search.screen_codes`, not memory.
- **NaCl: from 778,688 to 1,259,712 atoms (1.6×).** At 1,481,544 atoms the build runs out of memory in
  `SkinNeighbourList.compact`, which forms float64 vectors for every candidate pair: 8.2 GiB peak at 1.26M atoms.
  NaCl has 32 edges per atom within the cutoff, more candidates within the skin, and a denser cell than NbCl5.
- **Both next limits are the neighbour list**, which T3 already names as the next step ("the list build needs
  tiling too"): the pair screen's launch grid and the compaction's whole-candidate float64 vectors.
- **Against main on the same GPU:** NbCl5 at least 1.48M against 60,480-74,088 atoms (20-24×); NaCl 1.26M against
  64,000-74,088 (17-20×).

## T1: one padded CUDA graph per sweep

### What is built

**The slot.** `TileConfig(mode="padded" | "graphs", headroom=0.1)` runs every tile in one slot of fixed size:

- one receiver row more than the largest tile, so a padding receiver always exists;
- halo and edges with 10 % headroom above the largest tile's.

**Padding gives exact zeros:**

- Padding receivers and halo rows read atom 0 and are masked.
- Padding edges join the padding receiver to itself, after the tile's edges, so the receivers stay sorted.
- Their vector lies beyond the cutoff, where the envelope is exactly 0, and they are past the tile's edge count,
  which the slot's device buffer feeds to the streaming kernels as `streaming_count`.

**The bodies and their graphs:**

- **What a body touches.** Each sweep's body reads only the slot's static buffers and the persistent globals, and
  writes the slot's outputs.
- **Scatter after replay.** The scatter into the global buffers runs after each replay, so warm-ups and captures
  have no side effects.
- **One graph per sweep.** It is captured on first use and replayed for every tile. A tile is fed by copying its
  indices and masks into the slot; no kernel parameters are patched.
- **Recapture.** The graphs are captured again only when a tile does not fit the slot (it then grows), or when the
  globals move (atom count, neighbour-list capacity).
- **Readout mask.** The capture-safe readout mask is installed first, as `GraphEngine` does: MACE's `mask_head`
  assigns by index and cannot be captured.

**Tests.** The CPU tests run the padded slot against the whole-system call (float64, stock and MH-1 rewrites, 5,
16 and 1,000 receivers per tile), plus the slot's reuse and growth.

### Falsifier: fails, so T1 stops here

**Tool:** `tools/bench/tiled_graphs.py falsifier`, NbCl5 5×5×12 (21,600 atoms). It runs sweep 2 on the largest tile
in the slot: 4,096 receivers, 2,324 halo atoms, 107,690 edges, in a slot of (4,097, 2,620, 118,460). Each variant
takes 16 iterations after two warm-ups, with CUDA events.

| | Sweep 2 on one tile |
| --- | ---: |
| eager body | 43.38 ms |
| captured graph, replayed | 43.20 ms |
| **speed-up** | **1.00×** (target ≥ 1.8×) |
| largest output difference, graph against eager | 1e-7 (atomic accumulation order) |

**The tile body is GPU-bound, not launch-bound.** At 4,096 receivers and about 108k edges, kernel time dominates,
and a graph saves only the launch gaps.

- The tiled path's gap to main (2.1×) is in the eager kernels themselves, not in Python or launches.
- Main's compiled path fuses the model with Inductor. The whole-system compiled evaluation call is 1.45× faster
  than the eager one at 21,600 atoms (0.26 s against 0.38 s; [capacity-tiled.md](capacity-tiled.md)).

**Whole calls in one process** (`tiled_graphs.py call`, the same cell, the third of three calls):

| Mode | Call | Peak above before | max \|ΔF\| against eager |
| --- | ---: | ---: | ---: |
| eager (tiles at their own shapes) | 0.53 s | 1.28 GiB | |
| padded, eager | 0.75 s | 1.38 GiB | 1.4e-6 eV/Å |
| padded, graphs | 0.73 s | 0.01 GiB (the graph pool, kept since the first call) | 1.4e-6 eV/Å |

**Padding makes the call slower.**

- Every tile pays for the largest tile's slot: the eight tiles here vary in size.
- The slot sizing builds every tile's local graph once more on the host.
- The graphs then save the same 2 % as in the falsifier.

**Not run, since the falsifier failed:** the gate, the NVE check and the timing against main for the graph mode.
The padded and graph modes stay in the code as opt-in modes: the CPU tests cover them, and the GPU run above shows
them correct.

### What could close the gap instead (not tried)

- **Compiled tile bodies** (Inductor at the slot's fixed shapes, the force derivative outside the compiled
  frame as in the split-forces path). The ceiling is about the whole-system compiled/eager ratio, 1.45×. That
  would put the tiled call at about 1.5× main's MD step, against 2.1× now.
- **Less padding waste.** Tiles of more equal size (the binning's last block of each row is short), or two slot
  sizes.
- **No double host planning.** Keep each tile's local graph between the sizing pass and the sweeps.

## Checks

- Ruff is clean on every changed file.
- **CPU tests** on murphy (GPU hidden, `-n 8`, `mace_cueq_latest`, basetemp under `/mnt/homes/zwingel`), branch tip
  `8ad56972`: `test_tiled.py`, `test_layering.py` and `test_acceptance_tools.py`, **113 passed**.

## Setup

- **Machine:** murphy, RTX 4000 Ada Generation 20 GB (driver 595.99.02), NAS `mace_cueq_oeq`.
- **Idle check** before every job (`results/cap-tiled-20261030/run.sh`).
- **Model:** MH-1 float32 hybrid.
- **FastMACE:** branch `fastmace/cap-tiled-t1t2` from main `7185a492`. Sources per run are in the records: T2
  `7ab718dc`, T1 `69d0334f`.
- **When:** 30-31 October 2026, evening CET.
