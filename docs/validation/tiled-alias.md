# Tiled capacity path: the host state aliased

The neighbour-list session's proposal, through the conductor, after [tiled-host-onehot.md](tiled-host-onehot.md).
Host memory limited the tiled path at about 42M atoms on 250 GB. Of the call's 5.4 kB per atom, the tiled state
(x, sc and their adjoints, float32) was 4.86 kB. This change keeps two of the four tensors, bitwise.

## What changed

**`TileConfig.alias_state`** (with `state="host"`):

- **dE/dsc takes sc's rows.** A receiver's sc is read, and its dE/dsc written, by the same tile in sweep 2, and
  sweep 3 reads only dE/dsc.
  - In the padded path the tile's real receiver rows are assigned. The old `index_add_` onto zeroed rows would
    now add onto the stale sc.
- **dE/dx takes x's rows atom by atom.** Sweep 2 is the only reader of x. An atom's last tile in sweep 2 is the
  last to read its x and the last to add to its dE/dx.
  - Sweep 1 records each atom's first and last tile.
  - In sweep 2, dE/dx sums in a window holding only the atoms open at that moment, with the same additions in the
    same order as the unaliased `index_add_`.
  - After each tile, the atoms it closes copy their dE/dx into their x rows. Sweep 3 reads dE/dx there.
- **Padding rows** (padded path) are left out of the window: they add exact zeros.
- **The window** is sized once per call, from sweep 1's first and last tiles, to the most atoms open at once.

**`tools/bench/tiled_capacity.py --alias-state`.**

**The measurement before building it** (in [tiled-host-onehot.md](tiled-host-onehot.md)): the window is at most
2.9 % of N at 9.26M NbCl5 atoms and 1.8 % at 40.0M, against the conductor's 15 % threshold.

## Tests

CPU (leto, GPU hidden, `-n 8`): `test_tiled.py` and `test_layering.py`, 86 passed.

- The new test: the aliased host state's call is **bitwise** the unaliased one's, for the energy, forces and
  stress of a strained and an unstrained structure.
  - Eager and padded tiles, the stock model and MH-1's rewrites.
  - The state then holds only x and sc.
- `alias_state` without the host state is refused.

## Same-process A/B

**Tool:** `results/tiled-alias-20261101/ab_alias.py`, leto, 7d1b2ebc.

- One field, two tiled force fields (eager tiles, R = 4,096, tile lists, the species index): aliased and plain.
- 6 alternating blocks, the order flipped every block.

| NbCl5 atoms | Aliased / plain | State, plain | State, aliased | max \|ΔF\| |
| ---: | ---: | ---: | ---: | ---: |
| 51,840 | 1.009 | 252 MB | 156 MB | 1.0e-6 eV/Å |
| 884,736 | 0.996 | 4.30 GB | 2.30 GB | 1.2e-6 eV/Å |

The force differences on the GPU are the device's atomic force scatter, as between any two calls. The bitwise
gate is the CPU test.

## Host memory and capacity

leto, RTX 4000 Ada 20 GB, 250 GB host; bcc8d486; `results/tiled-alias-20261101/scan.sh`. Every point passed the
exact capacity check: the initial edges equal ASE's count, and the largest force is nonzero.

**At 9.26M NbCl5 atoms**, against [tiled-host-memory.md](tiled-host-memory.md):

| | Plain host state | Aliased |
| --- | ---: | ---: |
| tiled state (with the window and the index arrays) | 41.95 GiB | 21.63 GiB (2.51 kB/atom) |
| host peak, the call | 48.98 GiB (5.68 kB/atom) | 29.76 GiB (3.45 kB/atom) |
| host peak, the build | 28.14 GiB | 28.17 GiB (3.27 kB/atom) |
| call | 234 s | 235 s |

**The limits:**

| Structure | Atoms | GPU build | GPU call total | Host peak, call | Host peak, build | Call |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NbCl5 | 62,099,136 | 6.39 GiB | 16.18 GiB | 180.5 GiB (3.12 kB/atom) | 176.1 GiB | 1,814 s |
| NbCl5 | 67,917,312 | 6.97 GiB | 17.69 GiB | 198.7 GiB (3.14 kB/atom) | 192.4 GiB | 2,015 s |
| NbCl5 | 70,957,944 | 7.28 GiB | **18.47 GiB** | 207.4 GiB (3.14 kB/atom) | 200.9 GiB | 1,937 s |
| NaCl | 67,917,312 | 6.97 GiB | 17.67 GiB | 221.3 GiB (3.50 kB/atom) | 221.3 GiB | 1,768 s |

**The largest valid structure on a 20 GB GPU with 250 GB of host memory rises from 40.0M to 70.96M atoms.**

- The GPU's call limits next: 18.47 GiB of 19.5 GiB at 70.96M, about 73M at most. The owner's 70M target is
  reached.
- On the host, the build is now as large as the call: MACE's batch with its own neighbour list.
  - For NaCl, whose edges per atom are more, the build sets the host peak (3.5 kB per atom).
  - Lever (1), the build without that neighbour list, would lift NaCl's host ceiling a few million atoms past 70M.
    The owner's priority now is speed, so it stays on the list.
- The guard of the scan predicted 3.6 kB per atom and first skipped 70.96M. The call measured 3.14 kB, so the
  guard is now 3.3 kB.

## Plans

- Rejected, with the conductor: adjoint pre-contraction (storing dE/dh1 instead of dE/dx and dE/dsc). h1 is about
  2,100 columns against 608 for x and sc, so it would need about 8.2 kB per atom instead of 2.5.
