# Tiled path: compiled tiles as the default (the conductor's item (c))

The gates the conductor set after [tiled-speed.md](tiled-speed.md):

- an NVE drift check at 884,736 atoms, compiled against eager, 200 steps;
- the compiled limit;
- the compiled residency (slots and workspaces) measured as a memory term, at 9.26M atoms and at the limit.

## Setup

MH-1 float32 hybrid (`mh1-r2scan`), `mace_cueq_oeq`, NVIDIA RTX 4000 Ada (20 GB), 250 GB host, on murphy and leto.

- Tiles of R = 4,096, tile lists with the per-call cache, the state on the host and aliased, the species index.
- FastMACE 158127c3 and 32a459ba: main c8cb1a7f with this branch's `tiled_md.py` options.

## NVE drift (murphy, 884,736 NbCl5 atoms, 200 steps of 2 fs from 600 K)

**Tool:** `tools/bench/tiled_md.py --path tiled --tile-list --host-state --alias-state --species-index`.

- The same initial velocities for both runs.
- With per-tile lists the force field's own list is no longer maintained in the loop: the tiles find their edges
  from the positions in every call.

| | Eager | Compiled |
| --- | ---: | ---: |
| total-energy drift | 0.06122 meV/atom | 0.06123 meV/atom |
| total-energy spread | 0.1046 meV/atom | 0.1046 meV/atom |
| time per step | 24.03 s | **17.18 s (0.715×)** |
| slot changes | | 8 (the budget's switch to 16,384, then growth with the edges) |
| edges, step 0 → 200 | 23,248,896 → 24,889,358 | 23,248,896 → 24,889,352 |

The two runs' potential energies agree within 2.2e-5 meV per atom at every step. Their edge counts at step 200
differ by 6, because the trajectories part at rounding level. The drift is the tiled path's own, and the same in
both modes.

## Residency and the limit (leto)

**Tool:** `tiled_capacity.py --calls 2` (`results/tiled-compiled-default-20261101/climit.sh`); the tool sets
`budget_receivers=()` explicitly, so these are R = 4,096 throughout.

| NbCl5 atoms | Mode | Call | Allocated before | Peak above it | Total | Call time | Result |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 9,261,000 | eager | 1 / 2 | 1.075 / 1.351 GiB | 2.27 / 1.99 GiB | 3.34 / 3.34 GiB | 234 / 229 s | valid |
| 9,261,000 | compiled | 1 / 2 | 1.075 / 1.416 GiB | 2.23 / 1.89 GiB | 3.30 / 3.31 GiB | 199 / 167 s | valid |
| 64,964,808 | compiled | 1 / 2 | 6.68 / 8.65 GiB | 10.25 / 10.25 GiB | 16.92 / 18.89 GiB | 1,245 / 1,186 s | valid, two calls |
| 67,917,312 | compiled | 1 / 2 | 6.98 GiB / | 10.71 GiB / | 17.69 GiB / | 1,477 s / | the second call out of memory (1.03 GiB) |
| 67,917,312 | eager | 1 / 2 | 6.98 GiB / | 10.71 GiB / | 17.69 GiB / | 1,990 s / | **the second call out of memory too** (1.03 GiB) |

**What it shows:**

- **The compiled residency is small.** Between calls at 9.26M, compiled holds 0.065 GiB more than eager: the
  padded slots' buffers and outputs. Its workspace within a call is a little smaller than eager's (1.89 against
  1.99 GiB).
- **The compiled first call at 64.96M peaks where eager does.** 16.92 GiB, on the line through eager's
  16.18 GiB at 62.1M and 17.69 GiB at 67.9M (tiled-alias.md).
- **The second call needs more than the first, in both modes.**
  - At 67.9M both the eager and the compiled second call run out of memory.
  - The memory allocated before it is larger by the call's per-atom output buffers (forces, node energies and
    indices, about 32 bytes per atom), and the previous call's returned forces are still held by the caller.
  - The 70.96M of tiled-alias.md was a single call. **The limit for repeated calls, as in MD, is between 64.96M
    and 67.9M NbCl5 atoms on 20 GB, the same for both modes.** In short: single-call limit 70.96M, MD
    (repeated-call) limit 65-68M.

## The default (approved by the conductor; implemented)

`TileConfig.mode` defaults to `"auto"`. The planner has no tiled estimate yet, so "auto" takes the call's own
measurement, as the budget does.

- **An eager force field's tiles** stay eager.
- **A compiled force field's first call** runs eager tiles and measures its peak.
  - The later calls compile when the memory left at that peak holds the compiled residency and a margin:
    `COMPILED_RESIDENCY` (0.07 GiB, measured above) plus `COMPILED_MARGIN` (1 GiB, for the compiled workspaces'
    first allocation).
  - Otherwise they stay eager.
  - Chosen once.
- **The budget** (`budget_receivers` (8192, 16384)) is taken only once the tiles compile.
- **Every call's report** (`TiledForceField.last`) records `mode` and `mode_reason`.
- In the measured cases the memory term is far below what decides the limit, the repeated call's per-atom terms.
  So a call that fits eager tiles fits compiled tiles. No size threshold.

**Tests** (`test_tiled.py` and `test_layering.py`, 92 passed, leto):

- an eager field stays eager, with its reason;
- the choice at a headroom above and below the residency and margin, and that it is made once;
- the budget's default per mode.

**On the GPU** (leto, 7bbabbef, NbCl5 51,840 atoms, `tiled_capacity.py --mode auto --calls 4`):

| Call | Time | Mode | R | Reason |
| ---: | ---: | --- | ---: | --- |
| 1 | 5.66 s | eager | 4,096 | compiled force field: the first call measures its memory |
| 2 | 53.48 s | compiled (compiling) | 16,384 | 17.87 GiB left at the first call's peak >= the compiled residency and margin (1.07 GiB) |
| 3, 4 | 0.89, 0.88 s | compiled | 16,384 | (the same) |

`tiled_capacity.py --budget-receivers` without values now leaves TileConfig's default.

**Compile time per new slot shape:**

| Case | Compile time |
| --- | ---: |
| first call at 51,840 atoms (two slots) | about 60 s |
| first call at 9.26M atoms, three slots ([tiled-speed.md](tiled-speed.md); 199 s here against a steady 167 s) | 30-80 s |
| the budget's switch to R = 16,384 (two new slots) | about 60 s |
| first call at 64.96M (the first call against the second) | about 60 s |

Each new padded slot shape costs about 20-30 s, for its three sweeps.

## Next

- **The repeated call's extra 2 GiB at 65M** limits both modes 6-8M atoms below the single-call limit.
  - Which tensors are alive at the second call's peak is a memory-snapshot question, like the build's in
    [tiled-build.md](tiled-build.md).
  - The previous output's forces (12 bytes per atom) and the per-call buffers are candidates.
  - Profile before changing anything.
- **The profile with the cache**, and whether to overlap the host-state traffic with the device's work.
