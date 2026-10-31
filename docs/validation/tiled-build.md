# Tiled capacity path: the build without whole-system intermediates

After [tiled-host-state.md](tiled-host-state.md), the force field's construction limited the tiled path at
7.76M atoms on 20 GB, at about 2.2-2.4 kB per atom. The conductor's three changes, plus a fourth that the first
rescan showed was needed.

## What changed

1. **E0 by species index** (`fastmace/model/energy.py:node_e0_float64`).
   - Each atom's table entry is looked up through the one-hot row's `argmax`, instead of
     `node_attrs.to(float64) @ table.T`.
   - With exactly one nonzero term per row, the product is the table entry times exactly 1, so the lookup is
     bitwise the product. A test checks this on 500 random one-hot rows, every head, float32 and float64
     one-hot.
   - It drops the [N, 89] float64 copy: 712 B per atom, the 6.14 GiB that failed NaCl at 9.26M.
2. **The batch's edges never reach the device** (`fastmace/model/forces.py`).
   - The skin list's edges replace the batch's (`use_edges`) before anything reads them.
   - The force field now creates empty edge arrays on the device instead of copying the batch's.
3. **The allocator's cache is released once, before the first tile** (`fastmace/model/tiled.py`).
   - The construction leaves its peak in PyTorch's cache.
   - cuEquivariance allocates outside it (`cudaMallocAsync`) and found the device full in the first tile (NbCl5
     9.26M in [tiled-host-state.md](tiled-host-state.md)).
4. **`host_batch`: MACE's input batch left on the host** (`fastmace/model/forces.py`).
   - The first rescan (c88d28d6) ran out of memory at 14.89M NbCl5 atoms inside MACE's own
     `MACECalculator._atoms_to_batch`. Its last step moves the whole batch, MACE's own neighbour list included,
     to the calculator's device.
   - So change 2 had only removed a second copy: about 1.6 kB per atom were still on the device during the
     build.
   - `host_batch` is that function (mace-torch 0.3.16, as in every environment) without the move.
   - No padding graph is needed: the force field takes uncompiled calculators only, and MACE pads only compiled
     ones.
   - The force field moves what it keeps itself.
   - A test checks that `host_batch` gives MACE's batch tensor for tensor, and that construction works with
     meta-tensor edges, which cannot be copied, bitwise equal to a normal construction.

`fastmace.model` and the whole-system path give bitwise the previous results: E0 is bitwise the product, and the
batch's values are the same tensors on the host.

## Tests

murphy, GPU hidden, `-n 8`:

- **The whole CPU suite** on the first three changes: 4,830 passed, 205 skipped, and 1 failure. The failure was
  a configuration error in the new test (an unresolved `NeighbourConfig`); after the fix, the new tests pass.
- **With `host_batch`** (8a2e7b3d): 4,833 passed, 205 skipped.

## GPU check

The run setup:

- murphy and leto, NVIDIA RTX 4000 Ada (20 GB); `mace_cueq_oeq`;
- MH-1 float32 hybrid, `mh1-r2scan`, eager tiles of R = 4,096, tile lists, the state on the host
  (`tiled_capacity.py --tile-list --host-state`);
- runs `results/tiled-build-20261101/scan.sh`.

Every completed point passed the capacity check: edges per atom equal to ASE's (deviation 0) and a nonzero largest
force.

**NbCl5:**

| Atoms | Source, host | Before | Build peak | Call total peak | Call | Result |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 9,261,000 | c88d28d6, murphy | 4.15 GiB | 14.11 GiB | 6.41 GiB | 253 s | valid |
| 10,941,048 | c88d28d6, murphy | 4.87 GiB | 16.37 GiB | 7.33 GiB | 305 s | valid |
| 12,812,904 | c88d28d6, murphy | 5.68 GiB | 18.90 GiB | 8.36 GiB | 392 s | valid |
| 14,886,936 | c88d28d6, murphy | | | | | out of memory in MACE's batch move (changes 1-3 only) |
| 14,886,936 | 8a2e7b3d, murphy | 6.58 GiB | 11.51 GiB | 9.49 GiB | 431 s | valid |
| 17,173,512 | 8a2e7b3d, murphy | 7.56 GiB | 13.26 GiB | 10.72 GiB | 466 s | valid |
| 19,683,000 | 8a2e7b3d, murphy | 8.65 GiB | 15.18 GiB | 12.10 GiB | 546 s | valid |
| 22,425,768 | 8a2e7b3d, murphy | 9.83 GiB | 17.27 GiB | 13.60 GiB | 692 s | valid, **limit** |
| 25,412,184 | 8a2e7b3d, leto | | | | | out of memory in the build (allocating the positions, 292 MiB) |

**NaCl** (8a2e7b3d, murphy):

| Atoms | Before | Build peak | Call total peak | Call | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 9,261,000 | 4.14 GiB | 7.21 GiB | 6.54 GiB | 271 s | valid |
| 12,812,904 | 5.68 GiB | 9.93 GiB | 8.48 GiB | 351 s | valid |
| 14,886,936 | 6.57 GiB | 11.51 GiB | 9.61 GiB | 441 s | valid |
| 17,173,512 | 7.56 GiB | 13.26 GiB | 10.85 GiB | 495 s | valid |
| 19,683,000 | 8.65 GiB | 15.17 GiB | 12.22 GiB | 543 s | valid |
| 22,425,768 | 9.83 GiB | 17.27 GiB | 13.71 GiB | 611 s | valid (the largest size scanned) |

**The NbCl5 limit on 20 GB rises from 7.76M to 22.43M atoms.**

- The build peak falls from about 2.2 kB to 0.77 kB per atom; NaCl's from 2.4 kB to 0.78 kB.
- The memory before the call is 0.44 kB per atom.
- The call is about 31 µs per atom (692 s at 22.4M). It has not changed.

## What limits next

The build's peak, at about 0.77 kB per atom. Reading the code, without a memory snapshot yet:

- The one-hot `node_attrs` [N, 89] is on the device twice during the construction: the local copy that the
  rewrites and the species checks read, and the batch's copy in `data`. That is about 356 B per atom each.
- `data`'s copy stays: 356 B of the 0.44 kB per atom held before the call.
- The tiled path reads only each tile's rows of it.

The conductor asked for a memory snapshot at about 20M atoms to confirm this before anything changes.

### The snapshot (leto, NbCl5 19,683,000 atoms, 8a2e7b3d)

**Tool:** `results/tiled-build-20261101/snapshot.py`.

- It builds as `tiled_capacity.py --tile-list` does, with PyTorch's memory history on.
- It replays the allocation trace to the build's peak and lists what is alive there, with its innermost
  FastMACE or MACE frames.

The build's peak is 15.18 GiB above the start (15.17 GiB traced). 8.21 GiB stays allocated after the build.

| Alive at the peak | GiB | Bytes per atom |
| --- | ---: | ---: |
| the one-hot `node_attrs`, local copy (`forces.py:368`, `batch["node_attrs"].to(device)`) | 6.53 | 356 |
| the one-hot `node_attrs`, `data`'s copy (`forces.py:476`, `.to(device, dtype)` from the host batch) | 6.53 | 356 |
| positions, float64 and float32 (`forces.py:523`, `:521`, `:476`) | 0.88 | 48 |
| E0 (`energy.py:131`), the species arrays (`_setup_species`) and the batch's integer arrays | 1.10 | 60 |
| everything else (the model's tables, a self-check, small allocations) | 0.16 | |

The two copies of the one-hot matrix are 86 % of the peak. That confirms the code reading. Next steps, the
conductor's order:

- (a) one copy on the device;
- (b) the tiled path keeps only the species index and builds each tile's one-hot rows itself.

Records: `results/tiled-build-20261101/runs/` (8a2e7b3d) and `first-c88d28d6/`.
