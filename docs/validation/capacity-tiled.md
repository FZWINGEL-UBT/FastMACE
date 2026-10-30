# Tiled capacity path: receiver tiles, three sweeps (prototype)

The owner's request of 30 October 2026, through the conductor, after an outside review of Symmetrix-XL: a capacity
execution path that keeps only the inter-layer features and their gradient globally. Every other intermediate
lives in a tile workspace that does not grow with the atom count. Design: [tiled-backend.md](../design/tiled-backend.md).

## Verdict

**All three steps hold.**

- **Memory: flat.** At a fixed tile of 4,096 receivers, the force call's workspace stays between 1.2 and 1.7 GiB
  from 9,216 to 884,736 NbCl5 atoms. The workspace is the call's peak minus the global h1 and its adjoint.
  - **The whole call** costs 23-26 kB per atom at the large cells, including the model and the neighbour list.
  - **Main's MD plan** peaks at about 250 kB per atom allocated, 295 kB per atom reserved, on the same GPU.
- **Accuracy: float32 rounding.** Against the default compiled plan, the tiled call agrees to:
  - at most 3.3 × 10⁻⁵ eV/Å in the forces, about 1e-5 of the largest component;
  - 1 × 10⁻⁵ meV/atom in the energy;
  - 2 × 10⁻⁸ eV/Å³ in the stress.

  It passes the force/stress gate on NaCl, NbCl5 and NaNbO3 with the same errors as the default path: ratios
  0.14-1.04, against a limit of 2.
- **Capacity on murphy (RTX 4000 Ada 20 GB):**
  - **NbCl5: 884,736 atoms** (19.0 GiB), against main's limit between 60,480 and 74,088 atoms. That is at least
    12× and at most 14.6×.
  - **NaCl: 778,688 atoms** (18.2 GiB), against main's limit between 64,000 and 74,088. At least 10.5× and at most
    12.2×.
  - **Both next sizes** (1,061,208 and 1,000,000 atoms) run out of memory allocating the adjoint of h1. At that point
    the 16 kB per atom of global state is the limit.
- **Speed: about 2× slower than main's MD step.** The eager prototype's force call alone takes 2.1× main's whole
  graph-tier MD step on NbCl5 at 21,600 and 51,840 atoms (0.52 / 0.243 s and 1.2 / 0.605 s). At 884,736 atoms the
  force call takes 26 s.
- **MD: same trajectory as the whole-system call.** 300 NVE steps of 2 fs at 600 K, NbCl5 9,216 and NaCl 13,824
  atoms:
  - 47 and 55 neighbour rebuilds, the same count on both paths;
  - final positions within 1 × 10⁻⁵ Å of the whole-system call's;
  - total-energy drift identical to its: 0.063 and 0.033 meV/atom.

  NbCl5 at 373,248 atoms ran 40 steps with 8 rebuilds, with a drift of 0.047 meV/atom.

**What it needs before it is a planner choice:**

- compiled, graph-captured tile bodies (design note §5), to win back the speed;
- an owner decision under SPEC §18: automatic when nothing else fits, or explicit opt-in only (design note §6);
- integration with the engine, the MD state and the restart. This prototype is a force field and a one-off
  evaluation only.

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
| NaNbO3 6×6×6 | 8,640 | −9.1e-6 | 1.8e-5 | 1.1e-6 | 3.77 | 1.9e-8 | 1.73 / 5.24 | 0.45 / 0.15 s |

The differences are float32 rounding: about 1e-5 of the largest force component. The tiles' GEMMs see fewer rows,
and the edge-force scatter order differs.

## Force/stress gate

**The run.** `tools/force_stress_gate.py --engine fastmace --tiled-receivers 128`:

- 3×3×3 cells: NaCl 216, NbCl5 1,944 and NaNbO3 1,080 atoms;
- 3 rattled hydrostatic and 3 rattled full-strain configurations each;
- references in their own processes, the float64 reference and the float32 yardstick with `mace_cueq_latest`;
- R = 128, so every cell has several tiles and halos (NbCl5: about 16).

The default production path (the graph engine, padded edges) ran in the same batch. Records:
`results/cap-tiled-20261030/accuracy/gate-*/gate/metadata.json`.

**Columns.** Each cell is the largest error against float64, then the ratio to the float32 yardstick's error. The
gate's rule: a ratio at most 2, or the error under SPEC §12.3's floor.

| Structure | Path | Energy (meV/atom) | Max force (eV/Å) | Max stress (eV/Å³) | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| NaCl | tiled | 6.9e-5 (0.61) | 3.6e-6 (0.92) | 4.5e-8 (1.04) | pass |
| NaCl | default | 9.0e-5 (0.75) | 3.7e-6 (0.93) | 4.0e-8 (0.96) | pass |
| NbCl5 | tiled | 1.1e-4 (0.14) | 8.3e-5 (1.02) | 3.2e-8 (0.33) | pass |
| NbCl5 | default | 9.6e-5 (0.12) | 8.4e-5 (1.03) | 3.5e-8 (0.35) | pass |
| NaNbO3 | tiled | 2.4e-4 (0.23) | 1.3e-4 (0.99) | 1.9e-7 (0.56) | pass |
| NaNbO3 | default | 3.3e-4 (0.30) | 1.3e-4 (0.99) | 1.8e-7 (0.56) | pass |

**The tiled path is as accurate as the default path.** Its errors against float64 match the default production path's
within the spread of float32 rounding.

**For comparison, the whole-system eager call of the same force field** (`--compare eager`, the same modules on one
graph) peaks at 3.59 GiB at 9,216 atoms and 8.40 GiB at 21,600 atoms: 408 kB per atom. Main's compiled graph path
holds about 240 kB per atom live at its 16 GB limit of 51,840 atoms
([capacity-ledger.md](capacity-ledger.md)).

## Step 3: the largest cells on murphy

**Tool:** `tiled_capacity.py point --calls 1`, R = 4,096, unrattled cells growing until one fails
(`results/cap-tiled-20261030/step3.sh capacity`).

| Cell | Atoms | Edges | Tiles | Call peak (GiB) | Before (GiB) | Total (GiB) | Per atom | Neighbour build peak (GiB) | Call |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NbCl5 15×15×45 | 729,000 | 19,156,500 | 216 | 12.70 | 3.02 | 15.72 | 23.2 kB | 2.88 | 20.7 s |
| **NbCl5 16×16×48** | **884,736** | 23,248,896 | 312 | 15.40 | 3.63 | **19.04** | 23.1 kB | 3.49 | 26.4 s |
| NbCl5 17×17×51 | 1,061,208 | | | out of memory allocating the adjoint of h1 (8.1 GiB) | | | | | |
| NaCl 42×42×42 | 592,704 | 18,966,528 | 249 | 10.87 | 3.32 | 14.19 | 25.7 kB | 3.85 | 18.4 s |
| **NaCl 46×46×46** | **778,688** | 24,918,016 | 216 | 13.84 | 4.32 | **18.16** | 25.0 kB | 5.06 | 21.7 s |
| NaCl 50×50×50 | 1,000,000 | | | out of memory allocating the adjoint of h1 (7.6 GiB) | | | | | |

**Main on the same GPU** (`capacity_scan.py`):

- the default MD plan: graph engine, Langevin, float32;
- source `890a3f0f`: main plus this branch's changes. None of them is active outside the tiled path, except the
  species probe rows.

| Structure | Largest that runs | First out of memory | Peak allocated / reserved per atom at the largest |
| --- | ---: | ---: | --- |
| NbCl5 | 60,480 | 74,088 | about 250 / 295 kB |
| NaCl | 64,000 | 74,088 | 250 / 295 kB (14.9 / 17.6 GiB) |

**So the tiled path takes 12-15× the NbCl5 atoms and 10.5-12× the NaCl atoms of main on murphy.**

- The neighbour build's peak (3.5-5.1 GiB) stays below the call's at every size.
- The limit is the global state: h1 and its adjoint at 16 kB per atom, plus about 4 kB per atom of model inputs and
  the neighbour list.
- The MD state of a real run would come on top. A full NVT simulation needs velocities and the capped noise batch:
  at most a few hundred bytes per atom more.

**Not run:** the local RTX 5060 Ti. Its reservation was held by another session, with a second one queued, at
18:20 CET. By the same accounting (about 24 kB per atom against about 14.7 GiB usable), it would hold about 600-650k
NbCl5 atoms.

## Step 3: step time against main

The same GPU; main's whole MD step (graph tier) against the tiled prototype (eager, no graphs). For the tiled path
the table gives the force call alone, and the NVE loop's step from `tiled_md.py`.

| Cell | Atoms | Main MD step | Tiled force call | Tiled NVE step | Whole-system eager NVE step |
| --- | ---: | ---: | ---: | ---: | ---: |
| NbCl5 4×4×8 | 9,216 | | 0.25 s | 0.248 s | 0.176 s |
| NbCl5 5×5×12 | 21,600 | 0.243 s | 0.52 s | | |
| NbCl5 6×6×20 | 51,840 | 0.605 s | 1.2 s | | |
| NaCl 12×12×12 | 13,824 | | | 0.396 s | 0.272 s |
| NaCl 14×14×14 | 21,952 | 0.289 s | | | |
| NbCl5 12×12×36 | 373,248 | does not fit | 9.6 s | 11.2 s | does not fit |

**The cost of tiling, in eager, is +40-45 % over the whole-system eager call.** The extra work is the halo rows,
the replay of interaction 0 and the smaller GEMMs. The remaining factor of about 1.5 to main is eager against the
compiled, graph-captured path, so a compiled tile body is the first lever.

## Step 3: NVE with neighbour rebuilds

**Tool:** `tools/bench/tiled_md.py`:

- velocity Verlet in float64, starting from Maxwell-Boltzmann velocities at 600 K (seed 0, zero momentum);
- 2 fs steps, with the plan's 0.3 Å skin;
- each step compacts the edges within the cutoff, and rebuilds the candidates when an atom has moved more than half
  the skin;
- `--path both` runs the tiled call and the whole-system call of the same eager force field from the same start.

| Cell | Atoms | Steps | Rebuilds (tiled / whole) | Max \|ΔE_pot\| per atom | Final max \|Δx\| | Drift, tiled / whole (meV/atom) |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| NbCl5 4×4×8 | 9,216 | 300 | 47 / 47 | 7.2e-5 meV | 9.9e-6 Å | 0.063 / 0.063 |
| NaCl 12×12×12 | 13,824 | 300 | 55 / 55 | 1.8e-5 meV | 4.7e-6 Å | 0.033 / 0.033 |
| NbCl5 12×12×36 | 373,248 | 40 | 8 / not run | | | 0.047 / not run |

The tiled trajectory follows the whole-system one to float32 rounding through every rebuild, and conserves energy
as well as it does.

## Open items

- **Speed.** Compiled tile bodies at padded fixed shapes, one CUDA graph per sweep body replayed per tile (design
  note §5).
- **The 0.7 kB per atom residue** in the workspace between 110k and 593k atoms. It is not traced yet, and it is small
  against the 16 kB per atom global state.
- **Capacity beyond 0.9M atoms** needs the global state cut. Sweep 3 recomputes each tile's h1 itself, so h1 could
  be freed after sweep 2, and h1 and its adjoint would never be held together. That is 8 kB per atom, about 1.4-1.5×
  the atoms. Not tried.
- **Planner integration and the owner's decision** (design note §6). The engine, the MD state, restarts and events
  are untouched. `TileConfig` lives in `fastmace.model.tiled` until it becomes a planner option in
  `fastmace.config`.
- **Determinism.** The force scatter is atomic, so a reproducible plan would need a sorted reduction.

## What changed

- `docs/design/tiled-backend.md` (new): the design note.
- `fastmace/model/tiled.py` (new): `TiledForceField`, `TileConfig`, `plan_tiles`.
- `fastmace/rewrites/mh1/modules.py`: `cutoff_folded_forward` cuts the residual input to the receivers. It is the
  identity without ghosts.
- `fastmace/rewrites/species/__init__.py`: a structure with more than 7 atoms probes its distinct one-hot rows in
  the species self-checks.
- `tools/bench/tiled_capacity.py`, `tools/bench/tiled_md.py` (new).
- `tools/force_stress_gate.py`: `--tiled-receivers R`, the tiled path as the production path.
- `tests/cpu/test_tiled.py` (new):
  - the tiled call against the whole-system call in float64, for 5, 16 and 1,000 receivers per tile, with the stock
    model and with MH-1's CPU rewrite package, hydrostatic and strained;
  - without stress;
  - the tile planner's cover and caps;
  - the refusals.
- `results/cap-tiled-20261030/`: the job scripts (`run.sh`, `accuracy.sh`, `step3.sh`) and every record.

## Checks

- Ruff is clean on every changed file.
- **CPU tests** on murphy (GPU hidden, `-n 8`, basetemp under `/mnt/homes/zwingel`, `mace_cueq_latest`), commit
  `07714758`: `test_tiled.py` and `test_layering.py`, **44 passed**.
- The species self-check change, which came later (`f07e7c52`), is covered by the GPU runs only: every build applied
  and passed its 22 self-checks. The CPU species tests have not run on it. The conductor's full suite covers them.

## Setup

- **Machine:** murphy, RTX 4000 Ada Generation 20 GB (sm_89, driver 595.99.02), native Linux.
- **Environments:** NAS `mace_cueq_oeq` (torch 2.14.1+cu130) for production and the tools; `mace_cueq_latest` for the
  gate's references and the CPU tests.
- **Idle check** before every job (`run.sh`): no compute process on the GPU, and no other user's process above
  5 % CPU. Only an idle 4-day-old `python3` of another user, at 0.2 % CPU, was present.
- **Model:** MH-1 (`mh1-r2scan`, `mace-mh-1.model`), float32 hybrid.
- **Structures:** r2SCAN CIFs of the baseline export `1a-T18`.
- **FastMACE:** branch `fastmace/cap-tiled` from main `f3c635b9`. Commits per run are in the records
  (`source_commit`): falsifier `e72c7c06` and `f07e7c52`, accuracy `e0af8dbd` and `890a3f0f`, step 3 `890a3f0f`.
- **When:** 30 October 2026, about 16:00-18:30 CET.
