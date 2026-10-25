# Changelog

All notable changes to FastMACE. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/) (0.x: the API may still change between minor
versions). How versions are set and released: `docs/development.md`, "Releasing".

## [Unreleased]

Nothing yet.

## [0.1.0] - 2026-10-25

Name reservation on PyPI (owner decision D2): the library of the tag `fastmace-5` plus the release metadata below,
tagged `v0.1.0` (the `fastmace-5` tree still says `0.0.1.dev0` and `license = "MIT"`). Alpha: the library is
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

[Unreleased]: https://github.com/FZWINGEL-UBT/FastMACE/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FZWINGEL-UBT/FastMACE/releases/tag/v0.1.0
