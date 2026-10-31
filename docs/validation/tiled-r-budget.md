# Tiled call: the receiver cap from the workspace budget

The conductor's decision after the R × bucket screen ([tiled-screen.md](tiled-screen.md)): R is not fixed at
16,384 but driven by a budget.

- The largest R in {4,096, 8,192, 16,384} whose predicted workspace fits the memory left after the tiled state.
- Runs below the limit get the larger tiles' speed; the limit itself is unchanged.
- A rule from the memory model, not a table.

## What changed

**`TileConfig.budget_receivers`** (a tuple, default empty) lists the larger caps the call may take.

- **The first call** runs at `receivers` and measures its own workspace: the peak above what stays allocated
  after it (the tiled state, the field, the outputs). It resets the device's peak statistics at its start.
- **After it**, the call takes the largest of `receivers` and `budget_receivers` that fits:
  `1.1 × workspace per receiver × R ≤ available`.
  - Available memory is the device's free memory plus the allocator's cached, unallocated blocks.
  - The padded slots are planned again for the new cap.
  - The cap is chosen once.
- **Without a CUDA device** the cap stays.

**Why a measured term, not a fixed 0.26 MiB per receiver.** The screen's 0.26 MiB per receiver holds for compiled
tiles only.

- Eager tiles take about 0.6 MiB per receiver at the capacity limit: 2.4 GiB at R = 4,096 and 3.38M atoms.
- A fixed compiled term would choose R = 8,192 there and run out of memory, which would change the limit.
- The first call's own measurement covers every mode and model with one rule.

**Tools:**

- `tiled_capacity.py --budget-receivers`;
- `tiled_graphs.py screen` takes a candidate `R:bucket:R1+R2`, warmed up past the switch, and records the cap it
  chose.

## Tests

`tests/cpu/test_tiled.py`. The device memory is a stub, since CPU has none.

- The choice at three budgets: plenty gives 16; 900 bytes gives 8; 879 bytes gives 4 (8 × 100 B × 1.1 = 880).
- Forces and stress equal the fixed-cap call's; the cap is chosen once.
- Without a measurement the cap stays; non-positive caps are refused.

murphy, GPU hidden, `-n 8`, on bc2e6f51: `test_tiled.py` and `test_layering.py`, 71 passed.

## GPU checks (murphy, RTX 4000 Ada 20 GB, `mace_cueq_oeq`, MH-1 float32 hybrid, `mh1-r2scan`)

**Compiled, NbCl5 51,840 atoms**, against the fixed R = 4,096 in one process (6 alternating blocks of 3 calls,
`tiled_graphs.py screen`, 26011ee1):

| | Chosen R | Time / fixed 4,096 | Workspace | Slot changes |
| --- | ---: | ---: | ---: | ---: |
| budget (8,192, 16,384) | 16,384 | **0.902** | 4.27 GiB | 4 (2 at 4,096, then 2) |

Forces within 2.3e-6 eV/Å of the fixed call.

**Eager, tile lists, capacity points** (`tiled_capacity.py --calls 2`; the second call runs at the chosen cap):

| NbCl5 atoms | Source | Chosen R | Call 1 (R = 4,096) | Call 2 | Result |
| ---: | --- | ---: | --- | --- | --- |
| 2,299,968 | 26011ee1 | 16,384 | 58.6 s, 13.80 GiB | 56.0 s, 17.39 GiB | valid |
| 3,375,000 | bc2e6f51 | 4,096 | 98.5 s, 18.50 GiB | 95.1 s, 18.50 GiB | valid, **the limit unchanged** |

- Below the limit, eager mode gains only about 4 %; compiled mode gains about 10 %.
- At the limit, the cap stays at 4,096, and the limit is that of [tiled-table-rows.md](tiled-table-rows.md).
- The first 3.38M point (26011ee1, a branch cut from main) ran out of memory in its first call, at R = 4,096,
  before any budget choice. Main does not yet have the species tables at the call's row count (that branch is
  unmerged), so the field still kept the N-row tables. This branch is now based on `fastmace/tiled-table-rows`.

Records: `results/tiled-r-budget-20261031/`.
