# Changelog

All notable changes to FastMACE. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/) (0.x: the API may still change between minor
versions). How versions are set and released: `docs/development.md`, "Releasing".

## [Unreleased]

### Added

- `RewriteConfig.fused_folded_forward` (opt-8 OPT8-T02, opt-in, float32): one generated kernel builds MH-1's
  per-edge weights from the folded radial layer's input and the streaming message, and stores the weights for
  the unchanged chunked backward. Exact (weights and message bitwise the cuBLAS + streaming chain's on the
  RTX 5060 Ti). Needs `chunked_conv_backward`; refused in float64 plans: `docs/validation/opt8-T02.md`.
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
