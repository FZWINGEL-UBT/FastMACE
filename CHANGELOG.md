# Changelog

All notable changes to FastMACE. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/) (0.x: the API may still change between minor
versions). How versions are set and released: `docs/development.md`, "Releasing".

## [Unreleased]

### Added

- `fm.md(initial_velocities=...)` (and the `fastmace run` key `initial_velocities`), opt-in: a
  structure at rest starts with seeded Maxwell-Boltzmann velocities at the run's starting target
  (`"maxwell_boltzmann"`) or at a given temperature in K; `"keep"` (default) leaves it at rest as before. Drawn
  from `seed` on a stream of its own, never for a structure that already moves, and recorded in the report and
  `metadata.json`'s warnings (owner, 28 October 2026: an option, documented clearly; `docs/guide/md.md`,
  "Starting velocities").
- Three float64 speed-ups, **on by default on the measured GPUs** (owner decision 65; `plans/ideas-20261027.md`):
  `EngineConfig.fp64_emulation_mantissa_bits` ("auto" = 56 fixed mantissa bits in cuBLAS's float64 emulation,
  "dynamic" or a bit count from 8 to 128 to choose), `RewriteConfig.w_tables` (per-piece folded radial tables,
  `fastmace.kernels.triton.w_table`; rounding) and `RewriteConfig.save_transcendentals` (transcendental ops saved
  for the backward instead of recomputed; exact). "auto" turns them on in float64 MH-1 plans that are not
  reproducible on the "NVIDIA GeForce RTX 5060 Ti" and the "NVIDIA RTX 4000 Ada Generation" only (exact device
  names), `w_tables` with at most three species. MH-1 float64 MD step 0.80 (NbCl5), 0.78 (NaNbO3), 0.81 (NaCl) of
  the decision-63 package on the RTX 5060 Ti; forces within 5e-15 eV/Å of it. Off on the A100, where D2 and D3
  measured 1-27 % slower, and on every GPU not measured.
- `RewriteConfig.fused_folded_forward` (opt-8 OPT8-T02, float32; **on by default for MH-1**, owner decision 64): one
  generated kernel builds MH-1's per-edge weights from the folded radial layer's input and the streaming message, and stores the weights for
  the unchanged chunked backward. Exact (weights and message bitwise the cuBLAS + streaming chain's on the
  RTX 5060 Ti). Needs `chunked_conv_backward`; refused in float64 plans: `docs/validation/opt8-T02.md`.
- Float64 package (owner decision 63, `docs/validation/f64-T01.md`): `EngineConfig.fp64_emulation` (cuBLAS's float64
  emulation on INT8 tensor cores where the measured float64 rate is below 1/8 of float32), float64 radial tables,
  and the rounding-class rewrites in float64 plans that are not reproducible. MH-1 float64 MD step 0.57 (NaCl),
  0.42 (NbCl5), 0.49 (NaNbO3) of stock on the RTX 5060 Ti; forces within 4e-14 eV/Å of stock.

- `RewriteConfig.chunked_conv_backward` (opt-6 OPT6-T02): MH-1's streaming convolution builds its per-edge
  weights from the folded radial layer's input and runs its backward in L2-sized edge chunks, so the
  `[E, 1280]` weight gradient never reaches DRAM. Exact (values per edge unchanged; only the order of the
  existing atomic sender sums changes). **On by default** (owner decision 60) with the hybrid streaming
  convolution; cells whose weights fit four L2s keep the unchunked backward. NbCl5 −2.8 to −3.3 %, NaNbO3
  −3.1 to −3.6 % per MD step, NaCl +0.8 % (RTX 5060 Ti): `docs/validation/opt6-T02.md`.
- Exactness class `certified` (SPEC §6.1, owner decision 61), and its first record,
  `RewriteConfig.radial_tables` (opt-6 OPT6-T05, opt-in): per species pair, MH-1's radial chain from the edge
  length to the folded layer's input and the edge density as Chebyshev tables fitted and certified at setup
  against float64; float32 plans only: `docs/validation/opt6-T05.md`. **On by default** (owner decision 62)
  in float32 plans that are not reproducible: NaCl −10 %, NbCl5 −8 %, NaNbO3 −19 % per MD step (RTX 5060 Ti).

### Fixed

- **Decision 65's D1 default is suspended** (owner, 28 October 2026): `EngineConfig.fp64_emulation_mantissa_bits`
  "auto" is cuBLAS's dynamic choice everywhere; an explicit bit count still applies. With 56 fixed mantissa bits
  (D1's default on the RTX 5060 Ti and RTX 4000 Ada), float64 graphs went wrong in two ways, in ResearchModule's
  Energy_Hull relaxations of MH-1:
  - the step's small FP64 products came back about 8 bits accurate: a variable-cell `gpu-LBFGS` relaxation of
    NaCl 3×3×3 set the cell to 16.75000 instead of 16.76443 Å after one step and diverged;
  - a graph re-captured during the run computed wrong energies and forces: NbOCl3 3×3×3 was +0.34 eV/atom from
    the first step after the re-capture on, with the same edges, and never converged.

  The same runs with dynamic bits follow ASE's path. The root cause in cuBLAS's fixed-mantissa emulation is not
  known. As hardening, every small product of the step code outside the model is now element-wise sums
  (`neighbors.skin.matmul_small`, `matvec_small`, `vecmat_small`), which no cuBLAS setting reaches: the LBFGS
  mat-vecs and history, the Frechet cell filter, `expm_3x3`, the BAOAB, MTK and Berendsen barostats, the
  variable-cell neighbour and D3 rebuild tests, the D3 image shifts and the leader shift copy. LBFGS's two
  triangular solves stay TRSM. That hardening alone fixes the first defect, not the second. Rounding changes by a
  few ulps (`logm` within 2e-14 of SciPy, was 1e-14): `docs/validation/fp64-d1-relax.md`.
- A CSVR run from a structure at rest is refused at once, before planning, with `fastmace.errors.InputError`
  (a `ValueError`) that says how to draw velocities. Before, the integrator's refusal counted as a fallback
  candidate failure: each candidate compiled first, and the run ended in a misleading `FallbackExhausted`
  (NaMOCl4 report, 28 October 2026). The fallback search now passes every `InputError` through.
- Documentation pass (`fastmace/docs-20261028`): a feature tour, a first-run tutorial and a complete option
  reference (`docs/reference/`, checked by `tests/cpu/test_docs_options.py`), plus the fixes it found:
  - `fastmace run`: a `steps` key in the file no longer crashes; it is the run's length, `--steps` only
    shortens a piece, and `--resume` continues an unscheduled run to it.
  - `fm.md` declares the largest `Displace` `delta_A` as `Workload.max_displacement_A` (MC with `Displace`
    failed without it).
  - `ForceField(polar=PolarConfig(...))` takes MACE PR 1762's PolarMACE options, recorded in the identity and
    restarts; **not tested**: no run with them has been validated, and the report says so.

### Changed

- `LangevinBAOAB.fixcm` is `False` and only `False`: BAOAB, as ASE's, never removed the total momentum, so the
  record no longer claims a fixed centre of mass. `fixcm=True` is refused; restarts recorded with it resume
  with `False`.
- `RewriteConfig.selfcheck_cache` is removed (it never had an effect); restarts that hold it still load.
- `metadata.json`'s `precision.tf32` records PyTorch's real matmul TF32 flag when the record is made;
  `precision.tf32_requested` holds the plan's `PrecisionConfig.tf32` label (FastMACE never sets the flag).
- Help texts, `fastmace init` and error messages cite no SPEC sections or rule numbers.

## [1.0.0] - 2026-10-25

First stable release (SPEC section 16, phase 6). Acceptance: `docs/validation/phase-6.md`. Qualified stack as
in 0.1.0 (torch 2.14.1+cu130 only, owner decision D3); Linux x86_64, NVIDIA GPUs.

### Added

- `fastmace-oeq-ext` (6-T04): OpenEquivariance 0.7.0's host extension prebuilt for torch 2.14.1+cu130 and
  CPython 3.10-3.13, pulled in by `pip install "fastmace[oeq]"`; no C++ compiler, ninja or nvcc is needed.
  `fastmace doctor` and the `oeq_runtime` patch record say whether the extension is prebuilt or built.
- User documentation and API reference as a MkDocs Material site (6-T05, 6-T06), examples (6-T07),
  benchmark tables (6-T08), licences and notices (6-T02), upstream PR drafts (6-T10).

### Changed

- Plan records name the integrator that runs for the phase 4 schemes (NVE, CSVR, NoseHooverChain, MTK,
  Berendsen) instead of "Langevin"/"LangevinBAOAB" (6-T11); Langevin and BAOAB keep their text.
- The accuracy gate's reference child loads an explicit checkpoint path outside `$FASTMACE_MODELS` (6-T14).
- Development status "Production/Stable".

## [0.1.0] - Planned, not published

The planned name reservation on PyPI (owner decision D2) was superseded by the 1.0.0 release; no `v0.1.0`
tag or PyPI upload was made. The proposal was the library of `fastmace-5` plus the release metadata below
(the `fastmace-5` tree still says `0.0.1.dev0` and `license = "MIT"`). Alpha: the library was
validated on the qualified stack (torch 2.14.1+cu130, mace-torch 0.3.16, cuEquivariance 0.12.0,
OpenEquivariance 0.7.0, ASE 3.27.0) on NVIDIA GPUs. Phase acceptance reports are in `docs/validation/phase-*.md`.

### Release metadata (phase 6)

- Version `0.1.0` in `fastmace/__init__.py`, the single source of the version.
- Release engineering (plan task 6-T12): this changelog, `tools/release/make_release.sh` (build and check from a
  tag, never uploads), the "Releasing" section of `docs/development.md` and a manual-only
  `.github/workflows/release.yml` that builds artifacts without publishing.
- Licence (owner decision D8.2, task 6-T02b): the 13 modules that port ASE 3.27's integrators, optimisers and
  `FrechetCellFilter` are LGPL-2.1-or-later and carry an SPDX header; the package licence expression is
  `MIT AND LGPL-2.1-or-later` (`NOTICE.md`, `docs/licences.md`, `LICENSES/LGPL-2.1-or-later.txt`).

### Phase 0: skeleton

- Package skeleton, import-time soft file locks (no NFS kernel locks), test tiers and layering rules.

### Phase 1a: engine extracted from the research code

- The graph-captured MD engine of ResearchModule's research engine as a library: MH-1 and the other MACE
  foundation models, hybrid cuEquivariance/OpenEquivariance FP32 path, model rewrites, NVT Langevin and NPT
  `LangevinBAOAB`. Step time within 0.5 % of the frozen research engine; force/stress gate passes for all 19
  model labels.

### Phase 1b: planner and front ends

- Planner with a recorded decision per setting, in-library accuracy gate (energy, forces, stress), restart
  files, `fastmace doctor` and `fastmace md`, `fm.md`, the ASE shim (`fastmace.ase`) and `fm.Simulation`.

### Phase 2 and optimisation round 1

- Rewrite registry with hashed patches, unknown-MACE-version fallback, model template cache, persistent
  subprocess worker, `gated_rows` and `segment_gemm` kernels. Fixed: graph pre-capture with stale edge views
  that dropped edges after about 0.2 ps (runs on `d14be8c` up to `5d1848b` were invalid).

### Phase 3: events and Monte Carlo

- Protocol events, swap/transmute Monte Carlo moves with an energy gate, momenta treatments for changed masses,
  the mass-weighted `fixcm` projection (3N − 3 degrees of freedom; decision 37) and 1 fs default timestep
  (0.5 fs with hydrogen).

### Phase 3b: dispersion

- Native GPU D3(BJ), checked against torch-dftd and simple-dftd3, for NVT and NPT; NbCl5 NPT melt–quench end to
  end. The amortised D3 cost (24–30 %) misses the 10 % target (decision 42).

### Phase 4: integrators and optimisers

- Ports of ASE 3.27's `VelocityVerlet`, `Bussi` (CSVR), Berendsen, `NoseHooverChainNVT`, `IsotropicMTKNPT` and
  `MTKNPT`; device `FIRE`, `LBFGS` and `FrechetCellFilter`; fixed atoms; fused Triton integrator kernels; GPU
  compatibility tier.

### Phase 5: adaptation and calibration

- Run-time tier adaptation and calibration, optimisation round 4. Acceptance: CPU suite 4,651 passed, bitwise
  trajectories equal to `fastmace-4`, timed regression within noise.

[Unreleased]: https://github.com/FZWINGEL-UBT/FastMACE/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/FZWINGEL-UBT/FastMACE/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/FZWINGEL-UBT/FastMACE/releases/tag/v0.1.0
