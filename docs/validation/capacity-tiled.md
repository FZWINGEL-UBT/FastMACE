# Tiled capacity path: receiver tiles, three sweeps (prototype)

The owner's request of 30 October 2026, through the conductor, after an outside review of Symmetrix-XL: a capacity
execution path that keeps only the inter-layer features and their gradient globally. Every other intermediate
lives in a tile workspace that does not grow with the atom count. Design: [tiled-backend.md](../design/tiled-backend.md).

## Verdict

**The falsifier holds.** At a fixed tile of 4,096 receivers, the force call's workspace stays between 1.2 and
1.7 GiB from 9,216 to 592,704 NbCl5 atoms (64× the atoms). The workspace is the call's peak minus the global h1 and
its adjoint.

- **Per atom**, the whole force call costs 24 kB at 592,704 atoms, including the model and the neighbour list.
  Today's path keeps about 240 kB per atom live.
- **The largest cell so far** is NbCl5 592,704 atoms: 13.2 GiB of murphy's 20 GB. That is 11× the 16 GB limit on main
  and is not yet the 20 GB limit (step 3).

ACCURACY: PENDING. STEP 3: PENDING.

## The prototype

- **`fastmace/model/tiled.py`.** `TiledForceField` wraps an eager `MACEForceField` built from the one-off evaluation
  plan: the default rewrite set, float32 hybrid, with `compile_mode="none"`.
- **Tiles** are planned on the host per call:
  - a spatial binning of the fractional coordinates into about cubic blocks of about R atoms;
  - blocks above R receivers, or above the edge cap of 2 R times the mean degree, are split;
  - each tile is a sorted index list, so the receiver-sorted edge order is kept.
- **The halo.** A tile runs the unchanged rewritten modules on `[receivers, halo]`, in MACE's LAMMPS ghost layout
  (`lammps_natoms`, `first_layer=True`), so each interaction cuts its outputs to the receivers.
- **The three sweeps:**
  1. interaction 0 forward, without autograd, into a global h1 (512x0e+512x1o);
  2. interaction 1 forward and backward, with dE/dh1 scattered into a global adjoint;
  3. interaction 0 again, with the backward of its energy plus ⟨dE/dh1, h1⟩.
- **Forces and stress.** Forces come from the edge-vector gradients, scattered to receivers and senders. The stress
  comes from a symmetric strain of the edge vectors.
- **Library changes found on the way:**
  - `cutoff_folded_forward` now cuts the residual input to the receivers, as stock MACE cuts `node_feats_res`. With
    no ghosts it is the identity. Before, the LAMMPS ghost layout failed in `gated_rows`.
  - The species self-checks (`fold_product_linear`, `species_tables`, `species_residual`) probed every atom's one-hot
    row. With more than 7 atoms they now probe the structure's distinct rows. The old checks ran out of memory in
    the build at 373,248 atoms.

## Falsifier: memory at a fixed tile size

**Tool:** `tools/bench/tiled_capacity.py point`. Its records are in `results/cap-tiled-20261030/falsifier/`.

**Setup:**

- murphy, RTX 4000 Ada 20 GB (driver 595.99.02);
- NAS `mace_cueq_oeq` (torch 2.14.1+cu130);
- MH-1 (`mh1-r2scan`) float32 hybrid, eager;
- NbCl5 from the r2SCAN CIF (72 atoms per cell), unrattled;
- R = 4,096;
- the second of two calls.

**Columns:**

- **Peak:** the peak allocated above what was allocated before the call.
- **Global:** h1 and its adjoint, 16,384 B per atom.
- **Workspace:** peak minus global.
- **Before:** the model, the neighbour list and the batch.

| Atoms | Edges | Tiles | Largest halo | Largest tile's edges | Peak (GiB) | Global (GiB) | **Workspace (GiB)** | Before (GiB) | Total per atom | Call |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9,216 | 242,176 | 4 | 1,356 | 107,620 | 1.375 | 0.141 | **1.23** | 0.18 | 181 kB | 0.2 s |
| 21,600 | 567,600 | 8 | 2,381 | 107,690 | 1.604 | 0.330 | **1.27** | 0.23 | 91 kB | 0.5 s |
| 51,840 | 1,362,240 | 16 | 3,308 | 107,768 | 2.114 | 0.791 | **1.32** | 0.35 | 51 kB | 1.2 s |
| 110,592 | 2,906,112 | 39 | 2,722 | 107,752 | 3.034 | 1.688 | **1.35** | 0.58 | 35 kB | 2.8 s |
| 216,000 | 5,676,000 | 64 | 2,191 | 89,642 | 4.501 | 3.296 | **1.21** | 1.00 | 27 kB | 5.1 s |
| 373,248 | 9,808,128 | 160 | 3,002 | 108,134 | 7.236 | 5.695 | **1.54** | 1.62 | 25 kB | 9.6 s |
| 592,704 | 15,574,944 | 250 | 3,074 | 108,152 | 10.741 | 9.044 | **1.70** | 2.48 | 24 kB | 15.3 s |

- **The workspace is flat to within 0.5 GiB over 64× the atoms.** It follows the largest tile (216,000 atoms: tiles
  of at most 3,408 receivers), plus a small residue that grows with N: about 0.7 kB per atom between 110,592 and
  592,704 atoms. That residue is not traced yet.
- **The global state is h1 and its adjoint.** Together with the neighbour list (about 2.6 kB per atom at 592,704
  atoms) and the batch, the force call costs 24 kB per atom there.
- **Neighbour build.** The rebuild and compaction before the call peak at 2.34 GiB above the call's base at 592,704
  atoms, which is below the call's own peak.

## Accuracy against the current default plan

**The comparison.** `tiled_capacity.py --compare compiled`: the tiled call against the default compiled evaluation
plan, the path of `fm.ForceField.evaluate`.

- Both are built in one process, with the same 22 rewrites.
- The configuration is rattled by up to 0.05 Å per component and strained by a random symmetric strain of up to 2 %.
- R = 4,096; second calls.

| Cell | Atoms | ΔE (meV/atom) | max \|ΔF\| (eV/Å) | rms ΔF (eV/Å) | max \|F\| (eV/Å) | max \|Δσ\| (eV/Å³) | Tiled peak / compiled peak (GiB) | Tiled (eager) / compiled call |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NbCl5 4×4×8 | 9,216 | 8.5e-7 | 2.3e-5 | 8.6e-7 | 2.50 | 2.8e-9 | 1.37 / 3.19 | 0.25 / 0.10 s |
| NbCl5 5×5×12 | 21,600 | 1.9e-6 | 3.3e-5 | 9.0e-7 | 2.74 | 2.6e-9 | 1.61 / 7.47 | 0.52 / 0.26 s |
| NaCl 8×8×8 | 4,096 | 6.9e-6 | 5.0e-6 | 2.1e-7 | 0.25 | 1.9e-9 | 1.37 / 1.54 | 0.42 / 0.05 s |

The differences are float32 rounding: about 1e-5 of the largest force component. The tiles' GEMMs see fewer rows,
and the edge-force scatter order differs. The gate against float64 is below.

**For comparison, the whole-system eager call of the same force field** (`--compare eager`, the same modules on one
graph) peaks at 3.59 GiB at 9,216 atoms and 8.40 GiB at 21,600 atoms: 408 kB per atom. Main's compiled graph path
holds about 240 kB per atom live at its 16 GB limit of 51,840 atoms
([capacity-ledger.md](capacity-ledger.md)).
