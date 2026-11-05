# Tiled capacity path: the construction with its one-hot on the host

Lever (2) after [tiled-host-memory.md](tiled-host-memory.md). At 40.0M NbCl5 atoms the GPU's largest term was
the construction's peak, 17.43 GiB of the card's 19.5 GiB. Most of it was the one device copy of the one-hot
`node_attrs` [N, 89]. The construction's species check and rewrites read that copy before the tiled path released
it to the host.

## What changed

- **`MACEForceField(release_node_attrs=True)`:** the one-hot stays on the host from the batch on.
  - The species check, the rewrites and E0 read it there. They use only its columns, sums, row count and argmax:
    exact in any float dtype on any device.
  - E0 is looked up on the host and moved to the device: the same float64 values.
  - `data["node_attrs"]` is the host tensor, and the field's own whole-system call refuses (as after
    `release_node_attrs()`).
- **`planner.build_evaluation(..., release_node_attrs=)`** passes it through.
- **`tools/bench/tiled_capacity.py --species-index`** builds with it.

## Tests

leto, `mace_cueq_latest`:

- **GPU:** `test_forces_gpu.py`, 5 passed. The new test checks that with `release_node_attrs=True` the one-hot is
  a float32 host tensor and E0 a float64 CUDA tensor.
- **CPU** (GPU hidden, `-n 8`): `test_tiled.py`, `test_layering.py`, `test_forces.py`, `test_energy.py`,
  `test_superset.py`, `test_planner.py`, `test_rewrites_mh1.py` and `test_rewrites_generic.py`, 375 passed,
  53 skipped.
  - The new test: the tiled call (padded, strained, stress) of a field built with its one-hot on the host is
    bitwise that of a field released after construction, and both refuse their own call.

## Same-process A/B against the current build

**Tool:** `results/tiled-host-onehot-20261101/ab_build.py`, leto, ddd1326c.

- Both builds in one process; each build's device peak is measured.
- Then 6 alternating blocks of tiled calls (tile lists, the state on the host, the species index), the order
  flipped every block.

| NbCl5 atoms | Build device peak, one-hot on the device / on the host | Tiled call, host build / device build | max \|ΔF\| |
| ---: | ---: | ---: | ---: |
| 51,840 | 1.77 / 1.71 GiB | 1.007 | 1.1e-6 eV/Å |
| 884,736 | 4.14 / 3.78 GiB | 0.9995 | 1.2e-6 eV/Å |

- The peak falls by 356 bytes per atom, the one-hot.
- The calls run at the same speed.
- The build times are not comparable within one process (the first build pays the one-time setup), so they are
  in the records only.

## Capacity

murphy, RTX 4000 Ada 20 GB, 250 GB host; 2719ed86 (the same code as this branch before its rebase onto
c90a8fbd); `results/tiled-host-onehot-20261101/scan.sh`.

| NbCl5 atoms | GPU build peak | GPU before the call | GPU call total peak | Host peak (the call) | Host build peak | Call | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 40,001,688 (before, tiled-host-memory.md) | 17.43 GiB | | 10.48 GiB | 199.8 GiB | 114.2 GiB | 1,108 s | valid |
| 40,001,688 | **4.17 GiB** | 4.17 GiB | 10.48 GiB | 200.2 GiB | 114.3 GiB | 1,106 s | valid (exact edge check) |

- The construction's GPU peak falls from 17.43 GiB to 4.17 GiB (0.11 kB per atom).
- The GPU's largest term is now the call: 10.48 GiB at 40.0M, about 0.26 kB per atom plus the tile workspace.
  That is near 70M atoms on 20 GB.
- The host limits first, at about 42M atoms on 250 GB (5.4 kB per atom at the call).

## The gx window of host-state aliasing (measured before building it)

The neighbour-list session's proposal, through the conductor: in host memory, gsc overwrites sc in place, and gx
takes x's rows atom by atom once an atom's last tile is done. Until then its gx accumulates in a window buffer.
The conductor asked for the window's size first.

**Tool:** `results/tiled-host-onehot-20261101/window.py`, leto, no model.

- The real tile plan (R = 4,096) and the per-tile neighbour lists (cutoff 6 Å).
- Each atom is open from its first touching tile, as receiver or halo, to its last one.
- The window is the number of open atoms after each tile.

| NbCl5 atoms | Tiles | Grid | Max window | Fraction of N | Mean halo |
| ---: | ---: | --- | ---: | ---: | ---: |
| 9,261,000 | 2,413 | 14 × 13 × 13 | 264,621 | 2.9 % | 2,295 |
| 40,001,688 | 15,072 | 22 × 21 × 21 | 721,387 | 1.8 % | 1,690 |

The window stays close to one layer of blocks in the plan's order, and its fraction of N falls as N grows: far
below the conductor's 15 % threshold. Aliasing is next.
