# Tiled capacity path: the tiled state on the host (T3 part 2)

After [tiled-table-rows.md](tiled-table-rows.md), the GPU limit was the tiled call's global state: x, sc and
their adjoints, 4.75 kB of the call's 5.9 kB per atom. `TileConfig.state="host"` keeps them in pinned host memory.

## What changed

- **`fastmace/model/tiled.py`, `_State`.** x, sc, dE/dx and dE/dsc [N, width], on the device (default) or in
  host memory (`TileConfig.state="host"`, pinned on CUDA).
  - Tiles read their rows to the device (`rows`), and put or add their results (`put`, `add`).
  - On the device these are exactly the previous indexing and `index_add_`.
  - In host memory they are the same operations on the host, after one copy.
  - `mode="graphs"` refuses the host state: its captured bodies read the state on the device.
- `tiled.last["host_bytes"]` reports the host state's size.
- **`tools/bench/tiled_capacity.py`.**
  - New flag `--host-state`.
  - A `RuntimeError` saying "out of memory" (cuEquivariance allocates outside PyTorch's allocator) is now recorded
    as out of memory.
  - Any other `RuntimeError` is recorded and raised.

## Tests

CPU (murphy, GPU hidden): `test_tiled.py` and `test_layering.py`, 76 passed.

- The host state gives bitwise the device state's energy, forces and stress: eager and padded, stock model and
  MH-1 rewrites, unstrained and strained.
- The state tensors are where configured.
- The refusals: an unknown `state`, and `state="host"` with `mode="graphs"`.

## GPU check (murphy, RTX 4000 Ada 20 GB)

The run setup:

- `mace_cueq_oeq`, MH-1 float32 hybrid, `mh1-r2scan`, eager tiles of R = 4,096, tile lists;
- FastMACE c0122a97;
- runs `results/tiled-host-state-20261031/scan.sh`.

**NbCl5 4×4×8, 9,216 atoms, rattled by 0.05 Å, host state against device state:**

| | Value |
| --- | ---: |
| ΔE (total) | 5.0e-6 eV |
| Δ max \|F\| | 0 |
| max \|Δσ\| | 1.2e-9 eV/Å³ |
| call | 0.262 s host, 0.270 s device |

On the GPU the two are not bitwise: the device's `index_add_` of dE/dx sums in an unspecified order, the host's
sequentially.

**Capacity.** Every completed point passed the capacity check: edges per atom equal to ASE's (deviation 0) and a
nonzero largest force.

| Structure | Atoms | Before | Build peak | Call total peak | Host state | Call | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NbCl5 | 3,796,416 | 1.78 GiB | 7.95 GiB | 3.41 GiB | 17.2 GiB | 106 s | valid |
| NbCl5 | 5,832,000 | 2.66 GiB | 12.13 GiB | 4.55 GiB | 26.4 GiB | 167 s | valid |
| NbCl5 | 7,762,392 | 3.50 GiB | 16.10 GiB | 5.60 GiB | 35.2 GiB | 237 s | valid, **limit** |
| NbCl5 | 9,261,000 | | 19.18 GiB | | | | out of memory in the call's first tile (cuEquivariance's `cudaMallocAsync`) |
| NaCl | 4,096,000 | 1.91 GiB | 9.43 GiB | 3.73 GiB | | 119 s | valid |
| NaCl | 5,832,000 | 2.66 GiB | 13.37 GiB | 4.68 GiB | | 181 s | valid |
| NaCl | 7,762,392 | 3.49 GiB | 17.75 GiB | 5.73 GiB | 35.2 GiB | 239 s | valid, **limit** |
| NaCl | 9,261,000 | | | | | | out of memory in the build (6.14 GiB) |

**The valid limits on 20 GB rise from 3.38M to 7.76M atoms for both NbCl5 and NaCl.**

- The call's GPU peak is now about 0.75 kB per atom: 5.6 GiB at 7.76M.
- The state lives on the host: 4.86 kB per atom, 35 GiB of murphy's 250 GB at 7.76M.
- Speed:
  - The call takes about 31 µs per atom: 237 s at 7.76M.
  - With the state on the device it took 29 µs per atom (99 s at 3.38M).
  - The copies cost about 5 % at the limit and nothing measurable at 9,216 atoms.

## What limits next: the force field's construction

The build peak is now the largest term: about 2.2 kB per atom for NbCl5 and 2.4 kB for NaCl, against the call's
0.75 kB.

- **NaCl 9.26M** failed in `fastmace/model/energy.py:node_e0_float64`, at
  `node_attrs.to(torch.float64) @ table.T`. That is the one-hot species matrix [N, 89] converted to float64:
  712 bytes per atom, 6.14 GiB at 9.26M.
  - Each row has a single 1, so the product is exactly the table entry of the atom's species.
  - An index lookup gives the same float64 value without the [N, 89] float64 copy.
- **NbCl5 9.26M** got through the build (19.18 GiB peak). The call's first tile then failed: cuEquivariance
  allocates through `cudaMallocAsync`, outside PyTorch's allocator, which still held the build's memory in its
  cache.
- **The rest of the build peak.** Probably the batch's whole-system tensors on the device, not yet measured:
  - the one-hot `node_attrs` (356 bytes per atom, kept);
  - the batch's edge index and shifts from MACE's `_atoms_to_batch`, moved to the device with the batch.
- **For the conductor:** a short follow-up would build without whole-system intermediates:
  - E0 by index, exact;
  - the batch's edges left on the host when the tiled path builds its own;
  - the allocator's cache released after the build.

## Left over

- `mode="graphs"` with the host state: it would need static per-slot staging buffers, filled outside the graph.
- The host copies are synchronous. Pinned staging buffers with non-blocking copies on a side stream could hide
  them, if the 5 % matters.
