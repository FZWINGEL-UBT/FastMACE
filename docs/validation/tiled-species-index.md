# Tiled capacity path: the species index instead of the one-hot

Step (b) after [tiled-onehot.md](tiled-onehot.md), in the conductor's scope:

- the tiled path keeps only the species index and builds each tile's one-hot rows;
- the whole-system path keeps reading its `node_attrs` as before.

## What changed

**`TileConfig.species`:**

- `"one_hot"` (default): unchanged.
- `"index"`: each tile's one-hot rows are `one_hot(species_index[nodes], 89)` in the stored dtype.
  - `species_index` is the field's `_species_index`: each atom's one-hot column, 8 bytes per atom, kept up to date
    by every species refresh.
  - These are exact 0/1 rows, bitwise the stored rows, padding rows (atom 0) included.
  - It needs the field's float64 per-atom E0, which the patched-energy path always has.

**`MACEForceField.release_node_attrs()`:**

- Moves `data["node_attrs"]` to the host, freeing 356 bytes per atom on the device (89 elements, float32).
- `TiledForceField` calls it with `species="index"`.
- The field's own whole-system call then refuses with an error naming the release; by default nothing changes.

**`tools/bench/tiled_capacity.py --species-index`.** The tool refuses it together with `--compare eager`, which
calls the released field.

## Tests

CPU (leto, GPU hidden, `-n 8`): `test_tiled.py`, `test_layering.py`, `test_forces.py`, `test_energy.py` and
`test_superset.py`, 182 passed, 16 skipped. The new tests:

- the rows from the index are bitwise the stored rows, for any nodes;
- the tiled call with the index is bitwise the one-hot call, eager and padded, strained, with stress (after a
  warm-up call: TorchScript's first call in a process rounds differently);
- the released field refuses its own call;
- an unknown `species` value is refused.

## GPU check

The run setup:

- murphy and leto, NVIDIA RTX 4000 Ada (20 GB); `mace_cueq_oeq`;
- MH-1 float32 hybrid, `mh1-r2scan`, eager tiles of R = 4,096, tile lists, the state on the host;
- FastMACE 35cb5101;
- runs `results/tiled-species-index-20261101/scan.sh`.

**Speed and agreement** (`ab.py`, murphy). One process, 6 alternating blocks of 3 calls, the order flipped
every block:

| NbCl5 atoms | One-hot | Index | Index / one-hot | max \|ΔF\| |
| ---: | ---: | ---: | ---: | ---: |
| 9,216 | 0.2541 s | 0.2529 s | 0.995 | 1.0e-6 eV/Å |
| 51,840 | 1.3106 s | 1.3012 s | 0.993 | 1.1e-6 eV/Å |

The force differences are the GPU's unordered sums, as between any two calls.

**Capacity.** Both points pass the conductor's exact check (on main since the capacity worker's change): the
initial edges equal ASE's count exactly, and the largest force is nonzero.

| Structure | Atoms | Host | Before | Build peak | Call total peak | Host state | Call |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| NbCl5 | 32,157,432 | murphy | 3.38 GiB | 14.04 GiB | 8.45 GiB | 146 GiB | 915 s |
| NaCl | 32,157,432 | leto | 3.37 GiB | 14.04 GiB | 8.44 GiB | 146 GiB | 888 s |

The NbCl5 32.16M point ran out of memory in the call with (a) alone. With (b) the call's total peak is 8.45 GiB.

- The device keeps about 0.11 kB per atom between calls: positions, forces and energies, the species index, E0.
- The build's peak is 0.47 kB per atom. The construction still puts the one-hot on the device once, for the
  species check and the rewrites, before it is released.

**The GPU no longer sets the limit on these workstations; host memory does.**

- A point needs about 6.7 kB per atom on the host:
  - the tiled state, 4.86 kB per atom;
  - MACE's host batch with its own neighbour list, about 1.6 kB per atom, which the force field never uses (the
    skin list replaces it).
- murphy and leto have 250 GB. 32.16M atoms is about the largest size that keeps 20 GB free.
- The scan script refuses a size whose predicted need exceeds the available memory less 20 GB. Its first
  version computed the need with mawk's `%d`, which stops at 2^31 bytes. The guard printed 2 GB for 32.16M
  atoms; both machines had enough memory. That is fixed (`%.0f`), and the scan stopped before the next size.
- The two small 9,216-atom runs (`*-small-*`) are marked invalid by the exact edge check: they are rattled, so
  their edge count is not the unrattled cell's. Their forces and energies serve only the one-hot against index
  comparison.

## What limits next

On the host:

- the tiled state, 4.86 kB per atom (x, sc and their adjoints, float32);
- MACE's host batch neighbour list, about 1.6 kB per atom, built by `AtomicData.from_config` and never used.
  Skipping it would need the batch without MACE's edges: a FastMACE-side batch builder, since MACE's
  `from_config` always builds them.

On the GPU, next: the build's one device copy of the one-hot, 0.47 kB per atom at the peak, near 41M atoms on
20 GB. The species check and the rewrites could read it on the host instead.
