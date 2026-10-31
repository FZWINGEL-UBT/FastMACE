# Tiled capacity path: one copy of the one-hot species

Step (a) after the build snapshot in [tiled-build.md](tiled-build.md).

The snapshot was taken at 19.68M NbCl5 atoms. It showed the one-hot `node_attrs` [N, 89] on the device twice
during the force field's construction:

- the local copy, which the species check and the rewrites read;
- `data`'s copy, made by `.to(device, dtype)` from the host batch.

At 356 bytes per atom each, the two copies were 86 % of the build's peak.

## What changed

`fastmace/model/forces.py`: the local `node_attrs` is moved to the device once, in the field's dtype, and put
back into the batch. `data`'s `.to(device, dtype)` then returns that same tensor.

- The species check and the rewrites read only its columns, row count and one-hot sums.
  - The species check tests that each row sums to exactly 1.
  - `superset_columns` and the generic rewrites' columns use only which columns are nonzero.
  - The reassociation uses the row count.
  - E0 uses `argmax`.
- All of these are exact in any float dtype, so the results do not change.

## Tests

- `tests/gpu/test_forces_gpu.py::test_construction_keeps_one_copy_of_the_one_hot_species_on_the_device`:
  - the batch's `node_attrs` after construction is the very CUDA tensor in `data`, in the field's dtype;
  - float32 and float64; the tiny test calculator builds its batch in float64, so float32 covers a dtype change.
  - leto, `mace_cueq_latest`: 4 passed, with the `private_copy` tests of the same file.
- The whole CPU suite (leto, GPU hidden, `-n 8`): 4,833 passed, 205 skipped.

## GPU check

The run setup:

- murphy and leto, NVIDIA RTX 4000 Ada (20 GB); `mace_cueq_oeq`;
- MH-1 float32 hybrid, `mh1-r2scan`, eager tiles of R = 4,096, tile lists, the state on the host;
- FastMACE 5b674003;
- runs `results/tiled-onehot-20261101/scan.sh`.

Every completed point passed the capacity check: edges per atom equal to ASE's (deviation 0) and a nonzero
largest force.

| Structure | Atoms | Host | Before | Build peak | Call total peak | Host state | Call | Result |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| NbCl5 | 25,412,184 | murphy | 11.12 GiB | 11.12 GiB | 15.22 GiB | 115 GiB | 862 s | valid |
| NbCl5 | 28,652,616 | murphy | 12.53 GiB | 12.53 GiB | 17.04 GiB | 130 GiB | 759 s | valid, **limit** |
| NbCl5 | 32,157,432 | murphy | | 14.04 GiB | | | | out of memory in the call's first tile (cuEquivariance's `cudaMallocAsync`) |
| NaCl | 25,412,184 | leto | 11.12 GiB | 11.12 GiB | 15.34 GiB | 115 GiB | 888 s | valid (confirmation) |

**The NbCl5 limit on 20 GB rises from 22.43M to 28.65M atoms.**

- The build no longer peaks above what it keeps: its peak is the memory held before the call, 0.47 kB per atom.
- The call limits now. Its total is about 0.64 kB per atom: what the field keeps plus the tile workspace (about
  4.5 GiB) and the per-atom forces and energies.

## What limits next

What the field keeps on the device, 0.47 kB per atom. Of that, 356 bytes are the one-hot `node_attrs`, which the
tiled path reads only as each tile's rows.

Step (b), in the conductor's scope:

- The tiled path keeps only the species index and builds each tile's one-hot rows (`one_hot(species[nodes])`,
  bitwise the stored rows).
- The whole-system path keeps reading its `node_attrs`.
- That leaves about 0.11 kB per atom on the device.

The next term after it would be host memory:

- the host state is 4.86 kB per atom (130 GiB at 28.65M);
- MACE's host batch is built with its own neighbour list, about 1.6 kB per atom.

murphy had 198 GB free before the 28.65M point.
