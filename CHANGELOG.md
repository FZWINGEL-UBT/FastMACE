# Changelog

All notable changes to FastMACE. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/) (0.x: the API may still change between minor
versions). How versions are set and released: `docs/development.md`, "Releasing".

## [Unreleased]

### Added

- `RewriteConfig.recompute_first_table` (capacity idea rebuild-`a`, exact, deterministic): the first interaction's
  radial table and reassociated message run under one non-reentrant checkpoint, so the force call keeps the edge
  lengths instead of the table's output (260 bytes per edge for MH-1 float32), and the backward evaluates the table
  again. MH-1 float32 on a 20 GB RTX 4000 Ada: NbCl5 114,048 to 119,232 atoms, NaCl 108,800 to 115,200, at 1.02-1.04
  times the step time. Off; rule 12 turns it on only when the plan does not fit with `recompute_conv_weights` on,
  before `recompute_node_segments` (`planner.memory.MEMORY_RESCUE`), and the out-of-memory rebuild follows the same
  order (`docs/validation/rebuild-a.md`).
- `RewriteConfig.recompute_node_region` (ninth list item 2, rounding, deterministic): `recompute_node_segments`'
  chain and product block of each interaction as one checkpoint region in 4 atom chunks, so the backward rebuilds one
  chunk at a time: the force call's peak is 0.77 times the separate segments' (35 kB per atom less) at 1.22-1.24
  times their force call. MH-1 float32 on a 16 GB RTX 5060 Ti: NbCl5 82,944 to 101,088 atoms, NaCl 70,400 to
  97,336. Off; the last memory-rescue record (rule 12 or the out-of-memory rebuild, only when the plan does not fit
  with `recompute_node_segments` on). The combined region no longer breaks the compiled graph
  (docs/validation/chunked-region.md).
- `RewriteConfig.recompute_conv_weights` (plans/capacity.md CAP-T07 and CAP-T07b, rounding): the per-edge
  tensor-product weights are not kept for the backward (1,280 floats per edge for MH-1, about a third of a large
  cell's force-call memory). With `fused_folded_forward` the forward builds them on chip without storing them and the
  backward builds them on chip again (1-4 % step time); otherwise the backward rebuilds them with cuBLAS in edge
  chunks. MH-1 float32 on a 16 GB RTX 5060 Ti: NbCl5 34,200 to 49,248 atoms, NaCl 27,000 to 46,656. Off; on only when memory would not be enough:
  rule 12 turns it on for a cell that cannot fit (its estimate divided by 1.34 does not fit 90 % of the free memory), and
  `Simulation` rebuilds once with it on after an out-of-memory error at build or capture
  (`fallback.memory_rescue`; a sticky CUDA context raises the error with the fix).
- `RewriteConfig.recompute_node_segments` (plans/capacity.md CAP-T08, rounding, deterministic): each interaction's
  `gated_rows` chain and product block run under non-reentrant `torch.utils.checkpoint` inside the compiled graph,
  so the force call keeps only their inputs (64 kB per atom less for MH-1) at 1.095 times the step time. MH-1 float32
  on a 16 GB RTX 5060 Ti: NbCl5 51,840 to 69,984 atoms, NaCl 49,248 to 67,200. Off; rule 12 turns it on only when
  the plan does not fit with `recompute_conv_weights` on (`planner.memory.MEMORY_RESCUE`), and `Simulation` rebuilds
  with one more rescue record per out-of-memory error while one is left.
- The graph engine keeps the large buffers of its capture and warm-up snapshots in pinned host memory (CAP-T05),
  about 0.5 GiB less at the force stage's peak on large cells.
- Leaner neighbour list (plans/capacity.md CAP-T09, exact): int32 candidate indices, and no stored candidate
  shifts (compaction computes them from the unit shifts and the build cell, bitwise); 60 instead of 104 bytes per
  slot in float32. The GPU build decides its screened pairs in chunks, so a rebuild's temporaries drop from about
  170 to about 50 bytes per pair. MH-1 float32 on a 16 GB RTX 5060 Ti: NbCl5 49,248 to 51,840 atoms.
- The compiled model leaves out outputs no FastMACE evaluation reads (`fastmace.model.compile.UNUSED_OUTPUTS`:
  MACE's concatenated `node_feats`; plans/capacity.md CAP-T12, exact), so Inductor no longer keeps them through the
  force backward: about 9 kB per atom less at the peak. MH-1 float32 on a 16 GB RTX 5060 Ti: NbCl5 51,840 to 54,432 atoms, NaCl 49,248 to
  51,984.

- `EngineConfig.force_graphs` (`"auto"`, `"keep"`, `"fit"`, `"one"`): near the GPU memory limit the graph engine
  releases its captured graphs and their pool before capturing a force graph at a larger padded edge size, so one
  force evaluation's activations live at a time; `"auto"` is `"fit"` with CUDA graphs (plans/capacity.md CAP-T01).

- `RewriteConfig.gate_epilogue` (opt-10 OPT10-T01, float32, opt-in): in MH-1's gated chains the gate's backward
  runs in `linear_2`'s input-gradient GEMM (`fastmace.kernels.triton.gated_products`), one atom-tiled Triton kernel
  per gated segment where the shape rule picks Triton, without the `drows` round trip. Rounding-class,
  deterministic. Needs `segment_gemm`; refused in float64 and beside `tf32x3_gemms`:
  `docs/validation/opt10-gate-epilogue.md`.
- `RewriteConfig.wprime_backward` (opt-9 OPT9-T03, float32; **on by default for MH-1 on the measured GPUs**, owner
  decision 66): the second MH-1 interaction's radial distance gradient inside the streaming convolution's backward,
  each edge's weight gradient dotted with `w'(r)` from per-piece derivative tables; the weight gradient is never
  stored, and the folded input gradient and the radial table's backward drop out. Rounding class, not
  deterministic; needs `radial_tables` and `chunked_conv_backward`. "auto" turns it on in float32 MH-1 plans that
  are not reproducible on "NVIDIA A100-SXM4-80GB", "NVIDIA GeForce RTX 5060 Ti" and "NVIDIA RTX 4000 Ada
  Generation" (exact device names): MD step 0.81-0.89 of the default on the A100, 0.94-0.97 on the RTX 5060 Ti,
  0.93-0.96 on the RTX 4000 Ada; off on every other GPU (`docs/validation/opt9-T03.md`).
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

- `gate_epilogue` is on by default for MH-1 on the RTX 5060 Ti and the RTX 4000 Ada (decision 66, by exact device
  name, `planner.fixed.DECISION_66_GATE_EPILOGUE_GPUS`), where `segment_gemm` is on and `tf32x3_gemms` is off: MD
  steps 1.2-4.7 % faster on NbCl5 and NaNbO3, NaCl neutral (docs/validation/speed-def.md). Off elsewhere; the A100
  was not measured. An explicit value is kept.
- `tools/bench/capacity_scan.py check_point`: no drift tolerance. The list at step 0 must have exactly the small
  copy's edges per atom, and after the MD steps the live count must equal a fresh ASE rebuild at the current
  positions (NbCl5's count grows 5 % in 200 steps at 300 K, which the old 5 % tolerance flagged).
- `gate_epilogue`'s kernel (`fastmace.kernels.triton.gated_products`) runs the K loop outside over a tile that stacks
  every component of the gated segment, so each weight tile is loaded once per K block (fifth list, item E). With
  `gate_epilogue` on, step time is 0.1-2.5 % lower than with the old kernel; `gate_epilogue` on is now 1.2-4.7 %
  faster than off on NbCl5 and NaNbO3 on the RTX 5060 Ti and the RTX 4000 Ada (neutral on NaCl). It stays opt-in
  until the default is decided: docs/validation/speed-def.md.
- Measured, not adopted: `edge_outer_sum`'s backward storing `grad_y` once per edge block (item F, exact, neutral on
  both GPUs) and the first interaction's reassociation through `segment_gemm`'s joined products (item D, 0.25 %
  slower on the RTX 4000 Ada at NaNbO3 8,640 atoms); docs/validation/speed-def.md. No code change.
- `segment_gemm` uses 128-row tiles with 8 warps on the RTX 4000 Ada (decision 66,
  `tile_rule.LARGE_TILE_GPUS`; bitwise equal): MH-1 float32 step 0.986-0.995 times on NbCl5 and NaNbO3. The
  28 October screen that called them neutral had measured the default against itself; the RTX 5060 Ti and the
  A100 keep 64-row tiles (docs/validation/ab-constants-audit-20261030.md).
- `segment_gemm` runs a row tile's column tiles next to each other when the product's A is larger than the
  L2, on the RTX 5060 Ti and the RTX 4000 Ada (decision 66, `tile_rule.COLUMN_ORDER_GPUS`): bitwise equal, A is
  read from DRAM about once instead of once per column tile. MH-1 float32 step at NbCl5 9,216 atoms 0.94 times
  (RTX 5060 Ti) and 0.92 times (RTX 4000 Ada); the A100-SXM4-80GB measured no gain and keeps the old order
  (docs/validation/segment-gemm-order.md).
- `tools/ab_rewrites.py` sets each side's `--a-constant` / `--b-constant` before every step, not only in its build:
  graphs captured again in warm-up had read the other side's value (docs/validation/ab-constants-audit-20261030.md).
- Measured, not adopted: `segment_gemm`'s irrep size D as a compile-time constant (bitwise equal, step time
  unchanged, more compiled variants; docs/validation/speed-small.md). No code change.

- The drawn-ahead Langevin/BAOAB/CSVR noise batch holds at most 64 MiB (CAP-T06): `IntegratorConfig.noise_batch`
  is still the most steps, but above about 5,500 atoms the batch is shorter (26 steps at 51,840 atoms, 0.53 GiB
  less at the peak). The random numbers and their order are unchanged. A restart written with another batch size,
  including every restart from before this change, resumes bitwise.
- Setup across compositions (fixes F1, F2, F3, F5; `docs/validation/setup-fixes.md`):
  - The converted-model templates belong to the `LoadedModel`, so every `ForceField` of one `fm.load` result
    shares them.
  - The OEq kernel ids come from the kernel's content, and radial-table pieces are padded to a bucket of 16.
  - A compiled OEq `ScaleShiftMACE` takes MACE's gradient step after the compiled call, which removes the graph
    break and the Dynamo restart.
  - The edge capacity buffers and the per-atom inputs are marked dynamic.
  - Effect on the RTX 5060 Ti: a later composition in the same process builds in about 11 s, down from 16 s; a
    same-species-count composition on a cold cache in 11–15 s, down from 23–26 s; a new atom count across
    processes in 18.7 s, down from 30.4 s.
  - Step time is unchanged.
- Rule 12 (decision 67, owner): the memory estimate predicts the measured peak of MH-1's fused path (1.0-1.4x,
  was 1.5-2.3x), and a rewrite is turned off for memory only when the plan would not fit with it on and turning it
  off lowers the estimate. Large MH-1 cells keep `streaming_conv` and its dependants: on the RTX 5060 Ti NaCl float32
  reaches 27,000 atoms instead of 21,952, and runs faster per atom (plans/capacity.md CAP-T02).

- **Per-GPU defaults (owner decision 66, 29 October 2026).** `RewriteConfig.grouped_launch` is now on by default
  for MH-1 on the RTX 5060 Ti (planner rule
  `on_with_block_linear_float32_unless_reproducible_for_generic_mace_family_and_mh1_on_decision_66_gpus`,
  `planner.fixed.DECISION_66_GROUPED_LAUNCH_GPUS`): NaCl MD step 0.971 / 0.975 in both build orders, NbCl5 and
  NaNbO3 neutral (`docs/validation/decision66-grouped-launch.md`). Off for MH-1 on every other GPU (the RTX 4000
  Ada measured slower); the generic family keeps it everywhere.
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
