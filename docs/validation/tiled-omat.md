# Tiled path with MACE-OMAT-0 medium: SrTiO3 capacity on 20 GB

The owner asked for the tiled path with a plain residual-block model, MACE-OMAT-0 medium (`omat0-medium`). The
comparison is symmetrix-xl's tiled numbers: 11.24M atoms on an A100 80 GB and 4.15M on an RTX 5090 32 GB, from one
float32 force call on cubic SrTiO3 (a = 3.905 Å, 6 Å cutoff;
[capacity-16gb.md](capacity-16gb.md)).

**Setup**
- NVIDIA RTX 4000 Ada (20 GB) on murphy and leto, each with a 250 GiB host.
- `mace_cueq_oeq` environment; float32.
- Tile lists, the host state aliased, the species index, the list reuse (default), and auto mode (default): the
  first call is eager and measures its memory, then the calls compile.
- Runs: `results/tiled-omat-20261102/runs/`.

## 1. Smoke: the tiled call against the whole-system call

`results/tiled-omat-20261102/smoke.sh` runs on rattled (0.05 Å) and strained (1 %) supercells. Each row compares a
tiled call with the whole-system call on the same configuration: eager tiles against the eager whole-system call,
and auto tiles against the default compiled plan.

| Structure, model | Atoms | Tiles | Energy diff (meV/atom) | Largest force diff (eV/Å) | Largest force (eV/Å) | Stress diff |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| SrTiO3, OMAT-0, eager | 5,000 | 16 (R = 512) | 3.6e-5 | 1.3e-5 | 2.16 | 2e-8 |
| SrTiO3, OMAT-0, auto | 5,000 | 1 (the budget's R = 16,384) | 4.5e-6 | 1.4e-5 | 2.16 | 2.6e-8 |
| SrTiO3, OMAT-0, auto, `--no-budget` | 8,640 | 16 compiled tiles in 2 slots (R = 1,024) | 8e-6 | 1.6e-5 | 2.14 | 7e-9 |
| NaCl, MH-1, eager / auto | 1,728 | 8 / 1 | 3e-5 / 1e-6 | 5.5e-6 | 0.26 | 2e-9 |
| NbCl5, MH-1, eager / auto | 13,824 | 40 / 1 | 4e-6 / 1e-6 | 3.8e-6 | 2.36 | 4e-9 |

**What it took for a plain residual-block model:** nothing in the tiled code changed. The tiles handled all of
these without changes:
- the generic first-interaction rewrites (`generic_reassociate`, `generic_species_skip`);
- compilation;
- slot sizing;
- the list reuse.

**Two fixes, both in `tools/bench/tiled_capacity.py`:**
- `--compare` with `--tile-list` compared against a whole-system call on an empty list (zero forces). The tool now
  builds the force field's own list at the same positions for the reference.
- `--no-budget` lets compiled tiles run as several tiles at small sizes.

The smoke rows are "invalid" only by the exact edge-count check, which does not apply to rattled cells.

**After main's chunked convolution backward** (6f21ba36, merged as b2e95eca), the OMAT-0 rows still match: forces
within 1.8e-5 eV/Å, energy within 3e-5 meV/atom. The tiles' GPU peak falls:
- eager tiles: 0.55 to 0.40 GiB;
- one compiled tile at 5,000 atoms: 4.31 to 2.57 GiB;
- 16 compiled tiles: 1.03 to 0.68 GiB.

## 2. The size scan

`results/tiled-omat-20261102/scan.sh` runs perfect crystals, so the exact edge check applies: 74 edges per atom.
Each size makes two calls, the first eager and the second compiled. The compiled call's time includes compiling the
slots.

**Before lever 1 (MACE's host neighbour list in the build)**, at 888f1a4c:

| Atoms | Build: host peak | Eager call (R = 4,096) | Compiled call | GPU peak (compiled) | Valid |
| ---: | ---: | ---: | ---: | ---: | --- |
| 975,560 (murphy) | 8.4 GiB | 41 s | 174 s (R = 16,384) | 14.57 GiB | yes |
| 5,000,000 (murphy) | 33.9 GiB | 165 s | 207 s (R = 16,384) | 14.66 GiB | yes |
| 10,001,880 (murphy) | 65.6 GiB | 356 s | 362 s (R = 16,384) | 16.17 GiB | yes |
| 20,098,395 (leto) | 129.6 GiB | 708 s | 483 s (R = 16,384) | 17.36 GiB | yes |
| 40,000,000 (leto) | | | | | skipped: about 277 GB of host |

- **The build's host peak was 6.5-6.9 kB per atom.** After the build the process held about 10 GiB at 20M; the calls
  never came close.
- **The cause:** `MACEForceField`'s batch came from MACE's `AtomicData.from_config`, which runs MACE's host
  neighbour search (`get_neighborhood`, matscipy). At 74 edges per atom that is about 4.7 kB per atom of lists,
  plus matscipy's temporaries.
- **The edges were thrown away unused.** The force field replaced them with empty arrays straight away
  (`BATCH_EDGE_KEYS`), and its own lists (the skin list, or the tiles') give every edge. So the limit before
  lever 1 was about 33-36M atoms, set by a list nothing used.

## 3. Lever 1: the build without MACE's neighbour search

`fastmace.model.forces.host_batch` now builds MACE's batch with `get_neighborhood` replaced by `_no_neighbours`:
- MACE's own cell handling (mace-torch 0.3.16): a zero or missing cell becomes the identity, and a non-periodic axis
  is extended;
- empty edge arrays.

This applies to every force field, not only the tiled path. The batch's edges were always discarded, so the edges
the model sees are unchanged: bitwise, they are the skin list's or the tiles'.

**Tests:**
- `tests/cpu/test_host_batch.py` checks, for a crystal, a slab (two periodic axes) and a molecule, that every entry
  but the edges is MACE's own batch, bitwise, and that the edge arrays are empty with MACE's dtypes.
- `test_construction_never_moves_the_batchs_edges` now expects empty edge arrays.
- The full CPU suite on trillian: 5,016 passed, 202 skipped.
  - The first run had 16 failures, all in `test_colab_driver.py`. That file needs git's `safe.directory` when its
    temporary directory is on the NAS; with it set, all 27 of its tests pass.

**The scan with lever 1** (leto, at 833e537d, before main's chunked backward):

| Atoms | Build: host peak | Eager call (R = 4,096) | Compiled call (R = 8,192) | GPU peak (eager / compiled) | Host peak | Kept lists | Valid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 40,000,000 | 35.3 GiB | 1,526 s | 1,080 s | 10.72 / 14.47 GiB | 139.6 GiB | 19.6 GB | yes |
| 49,691,875 | 43.3 GiB | 1,749 s | 1,249 s | 12.45 / 16.21 GiB | 172.1 GiB | 24.4 GB | yes |
| 56,953,125 | 49.3 GiB | 2,034 s | 1,413 s | 13.76 / 17.51 GiB | 195.0 GiB | 28.0 GB | yes |

- **The build is no longer the peak.** It now takes about 0.9 kB per atom instead of 6.9.
- **The host peak is now the calls':** about 3.7 kB per atom. Measured at 57M:
  - the host state (x, sc and the gradients' windows): 2.62 kB per atom (`host_bytes`);
  - the kept lists: 0.49 kB per atom at 74 edges;
  - the rest: the process, the positions' copies, and the call's transients.
- **The budget picks R = 8,192 here.** The whole-system device state (forces, energies, indices: about 0.1 kB per
  atom) grows with N.

**64.9M atoms** (leto; host guard at 3.68 kB per atom, keeping 20 GB free):
- **First run (3bd9f28f): the compiled second call ran out of GPU memory.**
  - The eager first call ran: 2,399 s, GPU peak 14.02 GiB, host peak 216.8 GiB of 251 GiB.
  - The compiled call failed on a 6.09 GiB allocation (13.30 GiB allocated, 5.89 GiB free).
  - The traceback in the record names the allocation: the compiled forward's radial weights for one tile,
    `empty_strided_cuda((1277696, 1280))`. That is a tile's edges times the radial MLP's output width, in float32.
  - The budget had raised the receiver cap from the eager call's measurement at R = 4,096. A compiled tile at the
    raised cap needs more than that measurement predicts.
- **The fix (f14daf92):** when a call runs out of device memory at a cap the budget raised, the tiled call plans
  again at the next lower cap and retries. The lower cap stays for later calls, and the call's record says why
  (`budget_fallback`). A call at the configured cap still raises the error.
- **Rerun with the fix (8db73835 = f14daf92 rebased; `runs/leto-omat-scan-SrTiO3-235x235x235`): valid.**
  - Eager, R = 4,096: 2,398 s, GPU peak 14.02 GiB, host peak 216.8 GiB.
  - Compiled: `receiver cap 16384 -> 8192 after running out of device memory`. It then took 2,097 s at
    R = 8,192 (12,000 tiles), with a GPU peak of 16.06 GiB and a host peak of 226.7 GiB.
  - 4,801,813,750 initial edges, as expected; forces checked.
  - The list reuse turned itself off: keeping the lists would have left 20.0 GiB on the host, at the floor.

**The limit, then:**

| | Atoms |
| --- | ---: |
| Largest valid size, two calls (eager, then compiled), all defaults | 64,889,375 |
| Host bound without the list cache and kept lists (3.2 kB per atom) | about 70M |

- **What limits it:** the host, at 3.7 kB per atom on 250 GiB, puts the limit near 65M with the 20 GB floor.
- **Against symmetrix-xl's tiled numbers** (one force call, float32): 11.24M on an A100 80 GB and 4.15M on an RTX
  5090 32 GB. This path runs 64.9M with two calls on a 20 GB RTX 4000 Ada with a 250 GiB host.
  - The host memory is what makes the difference; the device holds 14-18 GiB at every size.
  - Symmetrix's runs keep everything on the device.

## 4. MD at 5M atoms (murphy)

`tiled_md.py --mode auto --integrator host --steps 20` on 5,000,000 SrTiO3 atoms at 600 K, with 2 fs steps. Run:
`runs/murphy-omat-md-SrTiO3-100x100x100`, at 888f1a4c.

- **Steady step:** 98.7 s, or 19.7 µs per atom. Every fourth step searches the lists again and takes about 110 s,
  at 600 K with the 0.3 Å skin.
- **Memory:** GPU peak 14.6-14.7 GiB; the host holds 23 GiB per step.
- **Drift:** 0.131 meV/atom over the 20 steps, from the perfect crystal.

## 5. What could carry over from MH-1, with measured bounds

Profiles of one compiled tiled call, with OMAT-0 and with MH-1, on the same 975,560-atom SrTiO3 graph (R = 16,384,
murphy; `runs/murphy-omat-profile-*`):
- OMAT-0: 18.9 s;
- MH-1: 19.0 s;
- the device is busy 96 % of the call in both.

| Candidate | What it does for MH-1 | OMAT-0's measured share | Bound |
| --- | --- | --- | --- |
| Radial tables (`radial_tables`) | The radial network (Bessel features into an MLP with SiLU) becomes tabulated functions of the distance per species pair | The SiLU kernels (`triton_poi_fused_mul_silu*`, forward and backward): 2.3 s of 18.9 s. Part of the 4.3 s of GEMMs is the same network. | 12 % measured, up to about 30 % if most of the GEMMs are radial (to be split by a profile with the radial MLP marked) |
| `wprime_backward` (the convolution's weight gradient in one pass) | MH-1's backward kernel: 2.25 s | OMAT-0's `oeq_projected_backward`: 2.85 s | at most about 0.6 s (3 %) |
| A larger R (a smaller per-edge memory) | | R is at the budget's cap (16,384) up to 20M and 8,192 at 40M+; at R = 16,384 the halo is 27 % of the rows | about 1 % per doubling (MH-1's measurement, [tiled-padding.md](tiled-padding.md)) |
| The gated products' unpadded kernel | MH-1's gated nonlinear products | OMAT-0 has none: its products are symmetric contractions (cuEquivariance, an unnamed `kernel`: 1.83 s) | no counterpart |

**The one worth building: radial tables for OMAT-0-type models.** First, a profile that marks the radial MLP should
split the GEMM time, to make the bound exact.
