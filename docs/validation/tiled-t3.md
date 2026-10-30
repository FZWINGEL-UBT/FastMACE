# Tiled capacity path: T3 (per-tile neighbour lists)

The first part of T3 in [plans/capacity-ideas-20261030.md](../../plans/capacity-ideas-20261030.md), through the
conductor: each tile builds its own edges, and the force field's global list is no longer built. Host staging of
the global state is the next part of T3. The packed shift code (seventh list) is not needed for this part; see
"Left over".

## What changed

- **`fastmace/neighbors/tile_list.py`, `TileNeighbours`.**
  - It bins the wrapped fractional coordinates of the current positions into cells at least one cutoff wide,
    with at least 3 bins per axis.
  - A tile's receivers search their own bin and the 26 bins around it.
  - The decision is the global list's: shifts in the float dtype, then float64 vectors with the list's cell,
    and the same `< cutoff²` test.
  - Edges come out in the global list's order: by receiver, then the sender's group, then the sender.
  - The CPU tests check that each tile's edges are **bitwise** the global list's edges into those receivers:
    sender, receiver, unit shifts and shifts, on NaCl 3×3×3 rattled and strained.
  - `TileNeighbours.like(list)` takes the cutoff, cell, dtype and sender groups from a `SkinNeighbourList`.
- **`fastmace/model/tiled.py`.**
  - New field `TileConfig.neighbours`: `"global"` (default, unchanged) or `"tiles"`.
  - With `"tiles"`, the tiles are planned by receivers only (no degrees before the build), and the edge cap is
    lifted.
  - Padded slots now hold their tile's `(sender, receiver)` pairs and shifts instead of edge ids into the global
    list. The global buffers are renewed only when the atom count changes.
- **`tools/bench/tiled_capacity.py --tile-list`.**
  - The force field's list allocates one slot and compacts nothing.
  - The tiled call uses `neighbours="tiles"`.
  - The record's `edges_within_cutoff` is then the call's own edge count, which the capacity check compares
    against ASE's.

CPU tests: `tests/cpu/test_tiled.py` adds four tests.

- The bitwise list check above.
- The tiled call with tile lists against the whole-system call, eager and padded.
- The refusal of cells under three bins per axis.

## GPU check (murphy, RTX 4000 Ada 20 GB)

The run setup:

- `mace_cueq_oeq`, MH-1 float32 hybrid, model `mh1-r2scan`, eager tiles of R = 4,096 receivers;
- FastMACE source 6e63ab85 (branch `fastmace/tiled-t3`);
- runs `results/tiled-t3-20261031/scan.sh`.

**NbCl5 4×4×8, 9,216 atoms, rattled by 0.05 Å, global list against tile lists:**

| | Edges | E (tiles − global) | max \|F\| (tiles − global) | max \|Δσ\| | Call |
| --- | ---: | ---: | ---: | ---: | ---: |
| global list | 248,832 | | | | 0.22 s |
| tile lists | 248,832 | 3.8e-6 eV (total) | 4.8e-7 eV/Å | 2.3e-10 eV/Å³ | 0.24 s |

Both lists hold the same edges. The energies differ by float32 rounding of the summation order, as between any
two tilings.

## Capacity

Every point passed the capacity check: edges per atom equal to ASE's on a small copy (deviation 0) and a nonzero
largest force.

"Before" is the memory allocated before the call: the state, the force field and its (now empty) list. "Build" is
the peak above that during the force field's construction.

**Global list (the rescan of 31 October, before T3, source e37a0365):**

| Structure | Atoms | Before | Build peak | Call total peak | Call | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| NbCl5 | 1,728,000 | 6.92 GiB | 9.58 GiB | 16.08 GiB | 53 s | valid |
| NbCl5 | 2,000,376 | 7.99 GiB | 11.08 GiB | 18.38 GiB | 50 s | valid, **limit** |
| NbCl5 | 2,299,968 | | 12.72 GiB | | | OOM in the call |
| NaCl | 1,728,000 | 9.36 GiB | 12.26 GiB | 18.65 GiB | 42 s | valid, **limit** |
| NaCl | 2,000,376 | | 14.17 GiB | | | OOM in the call |

**Tile lists:**

| Structure | Atoms | Before | Build peak | Call total peak | Call | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| NbCl5 | 2,299,968 | 1.90 GiB | 12.72 GiB | 13.80 GiB | 59 s | valid |
| NbCl5 | 2,985,984 | 2.42 GiB | 16.49 GiB | 17.52 GiB | 84 s | valid, **limit** |
| NbCl5 | 3,796,416 | | | | | OOM in the build |
| NaCl | 1,728,000 | 1.46 GiB | 9.95 GiB | 10.84 GiB | 46 s | valid |
| NaCl | 2,299,968 | 1.90 GiB | 13.21 GiB | 13.93 GiB | 68 s | valid |
| NaCl | 2,985,984 | 2.42 GiB | 17.13 GiB | 17.64 GiB | 76 s | valid, **limit** |
| NaCl | 3,511,808 | | | | | OOM in the build |

**The valid limits on 20 GB rise from 2.00M to 2.99M NbCl5 atoms and from 1.73M to 2.99M NaCl atoms.**

The memory before the call falls from about 4.3 kB to about 0.85 kB per atom: the global list and its candidate
buffers are gone. The call's own peak is unchanged: the global state of 4.75 kB per atom plus one tile.

## What limits next: the reassociation table's build

The 3.8M NbCl5 point ran out of memory in the force field's construction, not in the call.

- It failed allocating 7.24 GiB (NaCl 3.51M: 6.70 GiB, the same place) at
  `fastmace/rewrites/mh1/__init__.py:species_table` → `modules.py:identity_species_feats` →
  `model.node_embedding(attrs)`, inside cuEquivariance's naive linear.
- `reassociate_first` evaluates the species table on an identity basis with **as many rows as the structure**
  (`identity_attrs(..., rows=len(node_attrs))`).
  - On purpose: the GEMMs' values depend on the row count, so the table is bitwise the features the atoms get in
    a whole-system call (plan task 3-T02).
  - That costs about 5.9 kB per atom for a moment (16.49 GiB at 2.99M NbCl5 atoms). With tile lists this build
    peak is above the call's own peak at every size measured: it is now the limit.
- In the tiled path the model never runs on N rows: each tile runs on R + H rows. So the N-row table buys no
  bitwise equality there.
- Fix proposals for the conductor (a rewrite and SPEC §8.5 matter, so not done here):
  - (a) `Context.inputs` names the row count of the model's calls, and the tiled path passes its slot's row count.
    Bitwise to the tiled call, then.
  - (b) Evaluate the table in row chunks of a fixed size. Not bitwise to any call, so it needs the spec's
    tolerance instead.

(a) gains little on its own. The call's total peak is about 6.3 kB per atom (17.52 GiB at 2.99M), so the call
would stop at about 3.3M NbCl5 atoms on 20 GB. Most of that is the global state, x, sc and their adjoints at
4.75 kB per atom. Host staging, the next part of T3, takes it off the GPU. With staging, the build would limit
again unless (a) or (b) is done, so both belong in the same step.

## Left over

- Host staging of x, sc and their adjoints (T3 part 2).
- The packed shift code: not needed, since the tile lists keep shifts per tile edge (12 bytes per edge, freed
  after each tile).
- The tile list's build per tile is not timed separately. The call with tile lists is 4 % to 10 % slower at
  9,216 atoms (0.24 s against 0.22 s), and the large calls run 59 s to 84 s.
