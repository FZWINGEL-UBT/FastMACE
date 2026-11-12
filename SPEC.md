# FastMACE — Specification

Status: draft 0.12, 17 October 2026. Owner: Fabian Zwingel. Written from the
Langevin_Thermostat experiment (`Experiments/Langevin_Thermostat/`), whose GPU
engine this library generalises. Drafts 0.2–0.6 and 0.8–0.10 answer eight
reviews and record the owner's decisions 6–14 (§18, §19); draft 0.7 carried
the research engine's results of 16–17 October 2026 (MACE-OFF, the fused
plain-cuEq convolution, no NFS file locks; §19); draft 0.11 adds the
adoption layer (§7.7), and draft 0.12 answers its review.

Since 17 October 2026 FastMACE has its own repository (§18 decision 1).
Paths in this document that start with `Experiments/`, `Structures/` or
`environment/` refer to the ResearchModule repository.

FastMACE is a standalone Python library that runs pretrained MACE models
(MACE 0.3.x family, including PolarMACE) for atomistic simulation on NVIDIA
GPUs, with exact model-level rewrites, a GPU-resident simulation state, CUDA
graphs and a planner that picks the fastest execution path that is valid for
the user's workload. It must give the same physics as its versioned numerical
reference (§12.1: MACE's own calculator plus documented upstream bug fixes),
within float32 round-off, and fall back safely whenever it cannot.

Three contracts drive extraction and testing, and take precedence over the
breadth of the integrator catalogue:
- **complete simulation state** (§10.6): everything needed to continue a run,
  including exchanged species and prefetched random numbers;
- **transactional mutation and recovery** (§8.5, §9.4): every in-place change
  and every fallback either completes and is validated, or leaves the state
  untouched;
- **versioned numerical reference** (§12.1): what "correct" means for each
  model, MACE version and physics option.

---

## 1. Goals and non-goals

### 1.1 Goals

1. **Speed with exactness.** Default execution is several times faster than
   ASE plus `MACECalculator`. Measured on the research engine: 3–22× in NVT,
   depending on model and crystal; see §3. It uses only rewrites that are
   exact up to floating-point rounding.
2. **General.** Any crystal, composition, cell and GPU, with no per-structure
   tuning tables or shape-keyed caches (project rule, 10 October 2026). Every
   optimisation must hold for other crystals, melt–quench protocols and atom
   exchanges. Per-GPU choices are allowed since decision 66 (29 October 2026):
   a rewrite may be on by default only on the GPU models where same-process
   A/Bs measured it faster.
3. **Broad simulation coverage.** NVE, several thermostats and barostats,
   relaxations, Monte Carlo moves (species swaps at high temperature in
   particular), simulated annealing and quench, and restartable long runs.
4. **Intelligent engine selection.** A planner chooses the execution tier and
   the rewrites from the model, the workload, the hardware and a correctness
   self-check, and records the choice.
5. **Safe by construction.**
   - Each rewrite checks itself against the original module at setup.
   - The accuracy gate runs on the user's own structure, through the path
     that production executes (captured replay included).
   - The planner refuses or falls back on any mismatch; a fallback never
     changes the selected physics.
   - Species changes and other in-place events are transactions (§8.5).
6. **Reproducible provenance.** Every run records the model hash, rewrites,
   precision, tier, hardware, software versions and seeds.
7. **Installable.** `pip install fastmace` works without `nvcc`, with optional
   extras for OpenEquivariance and PolarMACE.

### 1.2 Non-goals for 1.0

- Training or fine-tuning (inference only).
- Multi-GPU domain decomposition. LAMMPS ML-IAP covers that; it could come in
  a later major version.
- Batching many structures in one evaluation. The research production rule is
  one structure at a time. Batched screening is a candidate for 2.x (§17),
  not for 1.0.
- Lower precision than FP32 by default (TF32, BF16, FP16). FP32 with TF32 off
  is the production decision; lower precisions may exist only as explicit
  opt-in and must pass the gate.
- Non-NVIDIA GPUs. CPU runs are for tests and the reference only.

---

## 2. Users and use cases

| Use case | Needs | Typical tier |
| --- | --- | --- |
| Long NVT/NPT MD of one crystal (heating/cooling, annealing) | max speed, restart, async output | graph engine |
| Melt–quench, amorphous structure generation | high T, many neighbour rebuilds, quench relaxations | graph engine, adaptive skin |
| Species exchange / swap Monte Carlo at high T | species change in place, accept/reject on GPU | graph engine with a species superset |
| Insertion/deletion (GCMC), evaporation | atom count changes | not in 1.0 (refused, §9.3 rule 7); 1.x compiled tier with exact atom counts, 2.x ghost-atom buckets (§8.6, §17) |
| Relaxations (fixed or variable cell) | FIRE/LBFGS, stress | compiled or graph engine |
| Drop-in speed-up of ASE scripts | ASE `Calculator` | calculator front end |
| Charged / long-range systems (PolarMACE) | k-space path, per-graph charge/spin | graph engine with polar path |

---

## 3. Starting point: the research engine

### 3.1 Code that becomes the core

All of it is under `Experiments/Langevin_Thermostat/langevin/`, about 6,000
lines:

| Module | Lines | Content | Library destination |
| --- | --- | --- | --- |
| `gpu_langevin.py` | 2,607 | Force wrapper `MACEForces`, all rewrites, skin neighbour list (GPU build, in-graph compaction), padded edge buckets, CUDA-graph Langevin engine, pipelined steps, async output | `fastmace.model`, `fastmace.rewrites`, `fastmace.neighbors`, `fastmace.engine`, `fastmace.integrators.langevin` |
| `gpu_npt.py` | 292 | ASE `LangevinBAOAB` on the GPU (hydrostatic/full cell) | `fastmace.integrators.baoab` |
| `polar_gpu.py` | 658 | PolarMACE on the GPU: capturable k-space, stress fix (MACE issue 1642), PR 1711 uvu GEMM, PR 1762 port | `fastmace.longrange.polar` |
| `edge_outer_sum.py`, `skinny_linear.py`, `fused_radial.py`, `inplace_linear.py` | ~570 | Triton kernels | `fastmace.kernels.triton` |
| `oeq_streaming/` | vendored | OpenEquivariance PR 223 receiver-streaming kernels via NVRTC (BSD-3, LBNL; keep LICENSE and NOTICE) | `fastmace.kernels.oeq_streaming` |
| `pair_search.py` | 102 | GPU pair search | `fastmace.neighbors` |
| `quench.py` | 249 | Frame selection and quench relaxations | `fastmace.protocols.quench` |
| `workflow.py` | 1,499 | CLI, protocol profiles, provenance, resume (output truncation on resume, §10.6) | split: `fastmace.cli`, `fastmace.protocols`, `fastmace.provenance` |
| `upstream_mace.py` | 249 | backports from MACE develop (single-graph strain, PR 1498; plain-cuEq conv fusion on any CUDA device, PR 1641) and the naive-cuEq guard (PR 1614); process-local, installed files untouched | `fastmace.rewrites` (strain), `fastmace.loading` (conv fusion), `fastmace.planner` (kernel trait) |
| `file_locks.py` | 67 | PyTorch's `filelock` locks as soft locks, switched before torch is imported (no NFS lock requests; `docs/NON_LOCKING_DESIGN.md`) | `fastmace` import time (§14) |

Tooling that becomes the validation suite:
- `tools/validation/npt_stress_reference.py` (force/stress gate);
- `tools/validation/roundoff_reference.py`;
- `studies/other_models/probes/model_step_time.py` (benchmark);
- `studies/further_optimizations/probes/step_breakdown_probe.py` (profiler);
- `tests/` (329 CPU tests collected, `test_upstream_mace.py` and
  `test_file_locks.py` included).

### 3.2 Couplings that must be removed

- **Module globals as switches** (`FACTOR_SPECIES_RADIAL`, `BLOCK_LINEAR`,
  `GROUPED_BLOCK_LINEAR`, …, `GENERIC_*`, `polar_gpu.*`), set by
  `workflow.GPUEngine`. They become per-instance configuration (§6.1).
- **Process-wide state.**
  - `_COMPILED_MODELS` cache.
  - The shared streaming edge counter (`MACEForces.bind_streaming_count()`).
  - `torch._dynamo.reset()` per PolarMACE engine.
  - MACE's process-wide autograd-compile policy: hybrid and plain
    cuEquivariance models must not be compiled in one process.
  - `torch._dynamo.config.cache_size_limit` raised to 64.
  - PyTorch's file-lock class (`filelock.FileLock`), switched to soft locks
    when `langevin` is imported (`file_locks.py`), and the environments'
    `nfs-soft-locks` start-up hook (`environment/nfs_soft_locks`) that does
    the same for every interpreter.
- **Model copies lose MACE's convolution adapters.** The engine deep-copies
  the model before compiling or rewriting it. OEq modules serialize only
  their C++ description (`_restore_oeq_wrappers`), and cuEquivariance's
  fused convolution (`cuet.SegmentedPolynomial`, plain backend with conv
  fusion) pickles through `__reduce__` with its constructor arguments only,
  dropping MACE's four-argument adapter and `weight_numel`
  (`_restore_cueq_conv_fusion`, 16 October 2026). Every copy must re-apply
  both, or be replaced by an explicit model builder.

  Each must be encapsulated or documented as a process constraint and enforced
  by the planner.
- **Paths and environment.**
  - NAS paths (`/mnt/homes/zwingel/...`) and the `benchmark_scripts_path()`
    import of `Benchmark_MACE_GPU_OPTIMIZATIONS/scripts/common.py` (model
    table, `load_calculator`).
  - The `oeq_no_nvcc.sh` launcher and the `main.py` refusal without it: OEq
    0.7 lock-file workaround, runtime headers, per-process extension directory,
    `OEQ_EXTENSION_SEED`, and (since 16 October 2026) `CUDA_CACHE_DISABLE=1`
    when the CUDA driver cache directory is on NFS (§14).
- **Version pins.** Source-hash pins of MACE 0.3.16 forwards
  (`_NONLINEAR_FORWARD_SHA256`, the `GENERIC_INTERACTIONS` hashes, the PR 1762
  forward rewrite).
- **Hard-coded protocol choices.** Langevin noise layout, `NOISE_BATCH`,
  restart cadence.

### 3.3 Measured baseline (to be refreshed by the 1.0 benchmark suite)

Hybrid FP32, TF32 off, 3×3×3 cells, NVT 300 K, ms per MD step. These numbers
come from different days and two machines. A uniform table on one machine, run
with the library's own benchmark suite (§12.5), is a release task. A uniform
run of the research engine on 16 October 2026 was stopped after the three MH-1
heads (user decision); its MH-1 values are in the first row. The table is
orientation only: extraction acceptance (§16) compares against a frozen
research-engine revision on the same machine and configuration, never against
these mixed-machine numbers.

| Model | Hardware | NaCl | NbCl5 | vs stock ASE |
| --- | --- | --- | --- | --- |
| MH-1 (three heads; NaNbO3 24.3–24.6 ms) | RTX 5060 Ti | 3.66–3.69 | 28.7–29.9 | NaCl 8.7–10.0×, NaNbO3 6.8–7.1×, NbCl5 5.0–5.8× |
| MP-0 / MPA-0 / OMAT-0 / MatPES (generic rewrites) | RTX 5060 Ti | 1.7–2.6 | 10–13 | up to ~17× |
| POLAR-1-S / M / L | RTX 4000 Ada | 3.9 / 7.7 / 15.4 | 31.6 / 68.3 / 114.6 | 11.8–21.8× |
| OMOL-0 XL | RTX 4000 Ada | 14.2 | 162.6 | 3.7–4.5× |

---

## 4. Supported scope for 1.0

### 4.1 Software matrix

The pins below are the state that has been qualified. The library supports a
declared matrix, tests it in CI and refuses other combinations for the
rewrites it cannot verify.

| Component | Qualified | 1.0 support policy |
| --- | --- | --- |
| Python | 3.10.12; 3.13 clean-install GPU smoke (6-T15) | 3.10–3.13 (owner: add the 3.13 wheel) |
| torch | 2.14.1+cu130 (2.5.1+cu124 reference stack) | each release pins and CI-tests exact builds of the two newest torch minor releases at release time (CUDA 12.x/13.x wheels); other versions run only the rewrites whose self-checks pass |
| MACE (`mace-torch`) | 0.3.16 | per-release rewrite registry (§6.3); unknown versions run the reference path |
| cuEquivariance (+ ops) | 0.12.0 (0.6.1 older stack) | ≥ 0.12 |
| OpenEquivariance | 0.7.0 | optional extra `[oeq]`; ≥ 0.7 |
| e3nn | 0.4.4 | as MACE requires |
| ASE | 3.27.0 | ≥ 3.23 |
| graph-longrange | 0.4.0 (MACE 0.3.16 calls this interface) | extra `[polar]`, pinned per MACE version |
| Triton | as shipped with torch | as shipped with torch |
| torch-dftd | 0.5.1 | extra `[d3]`: the `torch-dftd` reference (§12.1) and the ASE front end; not needed by the native GPU D3 path (§11.2), which ships the same reference tables (§14) |
| simple-dftd3 (`dftd3`) | 1.6.0 | extra `[d3-reference]`, with `torch-dftd`: the second D3 reference of the cross-check tests (§11.2); not in `[full]` |
| OEq host toolchain (fallback build only) | GCC 11.4, ninja, CUDA runtime header wheels (`cuda.h`, `nvrtc.h`, CCCL) | not needed with the prebuilt extension wheels of §14; documented for the fallback build |

### 4.2 Hardware

- **Qualified:** RTX 5060 Ti (sm_120, Blackwell); RTX 4000 Ada (sm_89) on leto
  (14 October 2026) and murphy (15 October 2026); the workstation GPUs of
  coraline (RTX A4000) and leeloo.
- **1.0 target:** sm_80 and newer (Ampere, Ada, Hopper, Blackwell).
- **Older GPUs:** run the compiled tier without the Triton and streaming
  kernels if those fail their self-check.

### 4.3 Models

| Family | Interaction classes (MACE 0.3.16) | Rewrites available |
| --- | --- | --- |
| MH-1 (all heads) | `RealAgnosticResidualNonLinearInteractionBlock`, cutoff passed (`apply_cutoff=False`) | all MH-1 rewrites |
| MACE-OMOL-0 | same class plus per-graph joint embedding (total charge, spin) | MH-1 rewrites incl. first-interaction reassociation |
| MACE-POLAR-1 S/M/L | PolarMACE | MH-1 rewrites + polar path (§11.1) |
| MP-0 medium/large | `RealAgnosticResidualInteractionBlock` | `GENERIC_*` |
| MP-0b3, MPA-0, OMAT-0, MatPES | `RealAgnosticDensity(Residual)InteractionBlock` | `GENERIC_*` |
| MACE-OFF23 small/medium/large, OFF23b medium, OFF24 medium (organic: H, C, N, O, F, P, S, Cl, Br, I) | `RealAgnosticInteractionBlock` + `RealAgnosticResidualInteractionBlock` | `GENERIC_*` (padding exact) |
| Anything else | — | reference tier (MACE calculator), or the compiled plain path if the padding check passes |

**Model-level known issues** (surfaced as warnings by the planner, not as engine
failures):
- compiled plain cuEquivariance PolarMACE returns zero forces, so it is refused;
- POLAR-1-S has a Fukui charge runaway on NaNbO3;
- all POLAR sizes give huge forces and pressures on NaTaO3;
- OMOL-0 is outside its training domain on dense oxides and collapses in NaCl
  MD;
- MACE-OFF23 small/medium/large relax the experimental molecular crystals
  (urea, naphthalene, paracetamol) to 5.5–9.2% below the experimental volume
  (static, float64), as the MACE-OFF paper shows for OFF23(L); OFF24-medium
  is closest (−0.5 to −5.2%); OFF23-small also rearranges naphthalene's cell
  (a +10%, β +7°) (`studies/other_models/README.md`, 16 October 2026);
- MACE-OFF23-large on 2,160 atoms (paracetamol 3×3×3) fits a 16 GB card only
  in the default engine: the stock calculator and the plain engine (both
  with MACE's unfused convolution, 16 October 2026) run out of memory, and
  the float64 reference needs more than 20 GB (§12.3). The plain engine with
  the fused convolution, now its default, was not measured at that size;
- MACE-OFF is a molecular force field, but tiers 1–3 need 3D PBC (§4.4):
  gas-phase molecules run in a periodic box, on the reference tier of an
  `fm.Simulation` (`allow_reference=True`, §9.3 rule 1), or with MACE's own
  `MACECalculator`. `FastMACECalculator` is compiled-tier only, has no
  `allow_reference`, and refuses non-periodic or mixed-PBC atoms (§7.6).
  The measured speed-ups (3.9–16.8× against stock
  ASE) cover periodic crystals and a water box only.

MACE-MDP (`DipolePolarizabilityMACE`) predicts dipoles and polarizabilities,
not energies or forces, and is out of scope for 1.0 (§17).

**Energy scale.** MACE-POLAR-1, MACE-OMOL-0, MACE-OFF and the `omol` and
`spice_wB97M` heads of MH-1 have E0 tables that mix all-electron and
pseudopotential references, classified here by the E0 values themselves.
All-electron references: Cl ≈ −12,522 eV, Na ≈ −4,412 eV, O ≈ −2,048 eV,
Br ≈ −70,042 eV. Pseudopotential (effective core potential) references: Nb
and Ta (≈ −1,545 eV), and iodine. This follows the def2 basis sets, which
are all-electron up to Kr (Z ≤ 36, so Br included) and use effective core
potentials from Rb onwards (Nb 41, I 53, Ta 73). MACE-OFF (all five
checkpoints): H −13.57, C −1,030.57, N −1,486.38, O −2,043.93, F −2,715.32,
P −9,287.41, S −10,834.48, Cl −12,522.65, Br −70,045.28 eV all-electron, and
I −8,102.52 eV with a pseudopotential (−297.75 Ha, a def2 effective-core
valence energy; all-electron iodine is about −6,900 Ha ≈ −1.9 × 10⁵ eV). The
float32 problem below depends on the magnitude of the total, not on the
labelling, so it applies to the mixed tables as well. The other heads
and models use E0 of a few eV (MH-1 `matpes_r2scan`: Cl −4.8 eV, Ta
−40.5 eV). POLAR-1
stores its E0 table in float32, OMOL-0 and MH-1 in float64. NbCl5 3×3×3
with all-electron E0 has a total near −2.08 × 10⁷ eV, where one float32
step is 2 eV. Stock float32 energies of these models are therefore off by
45–102 meV/atom on NbCl5 (gates of 16 October 2026,
`results/new-models-local-rtx5060ti-20261016/`,
`results/omol-opt-rtx5060ti-20261016/`), against 0.001–0.13 meV/atom for the
pseudopotential-referenced models (`results/all-models-rtx5060ti-20261015/`).
MACE-OFF's float32 energies on the organic set are off by 0.1–12.2 meV/atom
in the production path and 0.08–11.4 meV/atom in the stock float32 calculator
(urea 0.1–0.2, naphthalene 3.6–4.0, paracetamol 10–12, water 0.55–0.70;
gates of 16 October 2026, `results/mace-off-20261016/`).
Forces and stress are unaffected (E0 is a constant per species); every
energy-based decision is not, hence §7.3.

**E0 classification rule** (`fastmace.model.capabilities.classify_e0`,
task 1a-T03, 17 October 2026). Each E0 is compared with an all-electron
estimate of the free atom: Thomas–Fermi plus the Scott term,
−0.7687 Z^(7/3) + Z²/2 hartree, accurate to about 15% for Z ≥ 3.
- **all-electron:** E0 / estimate between 0.5 and 3;
- **small** (a few eV, referenced to isolated-atom DFT): otherwise, when
  |E0| < 200 eV;
- **pseudopotential:** otherwise, when E0 < 0 and the ratio is below 0.5;
- **unknown:** anything else, reported as such.

The 200 eV cut lies between the largest small value seen (Ta −40.5 eV) and
the smallest def2 valence energy (Rb, about −650 eV). Every value quoted
above classifies as stated.

**Licences.** Model licences must be shown at load time and are recorded per
label in the label table (§7.6). Per the ACEsuit `mace-foundations` model
table (checked 17 October 2026): MP-0 (a, b, b2, b3) and MPA-0 are MIT;
OMAT-0, MatPES PBE and r2SCAN, OMOL-0 and MH-0/MH-1 are under the Academic
Software License (ASL, non-commercial); MACE-OFF is ASL too (its own
repository). POLAR-1's licence is taken from its release when its label is
added. The same table gives MH-1's training levels of theory as
PBE/r2SCAN/ωB97M-VV10, which fixes the per-head functionals of §7.7's
dispersion rule.

### 4.4 Boundary conditions and constraints

Support is declared per scheme: each integrator, optimiser and cell filter
has a record in `fastmace.integrators.registry` that lists its ensembles, cell
modes and constraint rows (`fixcm`, `fix_atoms`), with a reason for each row it
lacks. Rule 1 (§9.3) reads these records and refuses anything outside them at
plan time, before any compilation, with a reason and a stable code. `FixAtoms`
is held in the state as a device mask and host indices. Every other ASE
constraint is refused, and so are MC moves with any constraint (decision 10).

| Feature | Tiers 1–3 (force field) | Integrators and optimisers | Today |
| --- | --- | --- | --- |
| Periodicity | full 3D PBC only (the force wrapper and the polar path require it) | — | `MACEForces` and `polar_gpu.enable` raise without 3D PBC |
| Non-periodic or mixed PBC (molecules, slabs) | not in 1.0; reference tier only | ASE-driven | — |
| Fixed centre of mass | — | all 1.0 integrators | Langevin only |
| `FixAtoms` | — | NVE, Langevin, CSVR and NVT Berendsen, and FIRE and LBFGS at fixed cell (decision 44); refused for NHC, MTK and every variable cell (BAOAB, MTK, NPT Berendsen, cell filters), and with `fixcm=True`; MC moves with constraints refused in 1.0 (owner decision 10, 17 October 2026) | both GPU integrators raise on any constraint (`gpu_langevin.py:2136`) |
| Other ASE constraints, RATTLE | — | refused in 1.0; RATTLE in 1.x | — |
| Variable cell | stress path | BAOAB, MTK; FIRE/LBFGS with a cell filter | BAOAB |

Temperatures and thermostat targets count 3N − 3 degrees of freedom under
`fixcm` and 3(N − n_fixed) with `FixAtoms` (`dof="auto"`, decision 45), or the
scheme's ASE count with `dof="ase"`. Langevin and BAOAB keep their counts
(decision 37).

---

## 5. Architecture

```
fastmace/
  __init__.py            public API (§7)
  config.py              dataclasses: RewriteConfig, EngineConfig, PrecisionConfig, ...
  errors.py              exception hierarchy (§7.6)
  model/
    loading.py           checkpoint load, head selection, cuEq/OEq conversion (from common.load_calculator),
                         model hash, licence metadata, hybrid converter fix (keep_last_layer_irreps)
    forces.py            ForceField: energy/forces/stress from a GPU state (from MACEForces)
    capabilities.py      rewrite_support, padding_is_exact, first_inputs_species_only, model traits
  rewrites/
    registry.py          Rewrite records, version/hash keys, application order, self-checks (§6)
    mh1/                 factor_species, fold_cutoff, density_first, reassociate_first, joint_radial_input,
                         fused_folded_backward, segmented_first_sum, streaming_conv
    generic/             streaming, reassociate, species_skip, fused_radial_backward
    linear/              block_linear, grouped_linear, merged_residual
    polar/               uvu, pr1762 (opt-in), stress geometry
  kernels/
    triton/              edge_outer_sum, skinny_linear, folded backward, fused_radial
    oeq_streaming/       vendored PR 223 kernels + NVRTC runtime (LICENSE, NOTICE kept)
  neighbors/
    skin.py              SkinNeighbourList (GPU build, receiver sorting, sender groups, variable cell)
    buckets.py           padded edge/atom capacities, headroom policy, bucket growth
  state.py               SimulationState: the complete state of §10.6 (incl. prefetched random numbers)
  events.py              transactional mutations, derived-state registry (§8.5)
  integrators/
    base.py              Integrator protocol: graph-capturable half steps, observables, restart
    nve.py  langevin.py  baoab.py  csvr.py  nose_hoover.py  mtk.py  berendsen.py
  mc/
    moves.py             swap, transmutation, displacement, momentum handling (§10.4); (later) insertion/deletion
    driver.py            hybrid MD/MC schedule, accept/reject on GPU
  optimize/
    fire.py  lbfgs.py    GPU relaxations, fixed and variable cell
  engine/
    graph.py             CUDA-graph engine: capture, replay, pipelining, re-capture (§8.1)
    compiled.py          torch.compile without graphs (§8.2)
    eager.py             eager hybrid (§8.3)
    reference.py         MACE MACECalculator path (§8.4)
    subprocess.py        runs a plan in a child process from a restart write of the last committed step (§8.7, §10.6):
                         backend switches that §8.7 forbids in-process, fallbacks (§9.4)
  planner/
    workload.py          WorkloadSpec: what the user's run will do (§9.2)
    plan.py              Planner, Plan, decision rules, fallback algorithm (§9.3, §9.4)
    selfcheck.py         rewrite and gate checks on the user's structure (§12.3)
  reference/             versioned reference records and corrected reference calculators (§12.1)
  longrange/
    polar.py             PolarMACE GPU path (§11.1)
    d3.py                GPU D3 dispersion, all parameters explicit (§11.2)
  io/
    trajectory.py        async writer, ASE Trajectory/extxyz, every-n-steps, files chunked at restart points (§10.6)
    restart.py           checkpoint/restart format (§10.6)
  provenance.py          run metadata (§13)
  protocols/             schedule algebra (§7.6); heat/cool profiles, anneal, quench, melt–quench (from workflow.py profiles)
  ase/calculator.py      FastMACECalculator (ASE drop-in) and fm.calc (§7.7 level 0)
  ase/md.py              ASE-compatible integrator classes over Simulation, observer sync (§7.7 level 1)
  quick.py               fm.md and RunResult (§7.7 level 2); no engine code
  report.py              the one-screen run report and error-message shape (§7.7)
  cli/                   `fastmace run|md|init|relax|quench|bench|check|info|doctor|models`
  bench/                 step_time, breakdown (from model_step_time / step_breakdown_probe)
tests/  cpu/ gpu/ equivalence/ statistical/ transitions/ mc/ restart/
docs/
```

Layering rules:
- `model` and `rewrites` never import `engine`.
- Integrators see only `SimulationState` and a `ForceField`.
- The planner is the only component that instantiates engines (a minimal
  planner exists from phase 1, §16).
- State changes outside an integrator step go through `events` (§8.5).
- No module-level mutable switches; all options travel in config objects.

---

## 6. Rewrites

### 6.1 The `Rewrite` record

Every optimisation is one record in the registry:

```python
@dataclass(frozen=True)
class Rewrite:
    name: str                        # "grouped_linear"
    applies_to: Callable[[Model], Support]   # (bool, reason); from rewrite_support
    requires: tuple[str, ...]        # e.g. ("block_linear",); ("fold_cutoff",)
    conflicts: tuple[str, ...]
    mace_versions: SpecifierSet      # ">=0.3.16,<0.3.17"
    source_hashes: dict[str, str]    # qualified name -> sha256 of inspect.getsource (as today)
    exactness: Literal["exact", "rounding", "certified", "approximate"]
    default: Literal["on", "off", "auto"]
    workload_constraints: tuple[str, ...]  # e.g. ("fixed_species_superset", "single_graph")
    apply: Callable[[Model, Context], None]
    selfcheck: Callable[[Model, Model, Context], CheckResult]   # vs original module, at setup
```

**Defaults.**
- `"exact"` means bit-for-bit identical math up to summation order.
- `"rounding"` covers reassociation, folded or factored weights and changed
  GEMM grouping; most adopted rewrites are of this kind (§6.2 column).
- `"certified"` (owner decision 61, 27 October 2026) is an approximation
  that carries a certificate, computed at setup for the plan's own model,
  species superset and domain: against a float64 evaluation of the same
  weights, the float32 replacement's error is at most the float32 error of
  the computation it replaces, for its values and their first derivatives,
  each normalised per output, worst case on a dense grid. The record's
  self-check holds the certificate; a ratio above 1 fails it (§9.3 rule 13
  then turns the record off). Float32 plans only: the certificate bounds
  float32 rounding, not float64's. A certified record may become a default
  by owner decision after its gate and A/B, like a rounding-class one.
- `"approximate"` rewrites are never default.

### 6.2 Catalogue for 1.0 (adopted today unless marked; defaults as in `main.py` on 16 October 2026)

| Rewrite | Exactness (§6.1) | Applies to | Step-time change (measured; negative = faster) | Constraint |
| --- | --- | --- | --- | --- |
| `factor_species` (species-factored radial input) | rounding | MH-1 class | part of the MH-1 package | — |
| `fold_cutoff` (cutoff folded into the last radial layer) | rounding | MH-1 class, `apply_cutoff=False` | — | — |
| `sort_receivers` | exact (permutation) | all | −3 to −5% (large crystals) | — |
| `density_first` (division before `linear_1`) | rounding | MH-1 class | ≈ −11% together with `reassociate_first` | bias-free `linear_1` |
| `reassociate_first` (first interaction without per-edge TP weights) + Triton backward | rounding | MH-1 class, `GENERIC_*` | MH-1 package; OMOL −9% NbCl5 | species-only first inputs: linear node embedding, at most per-graph joint embeddings, single graph, fixed species superset |
| `segmented_first_sum` | exact (summation order) | with `reassociate_first` + sorting | −1.6% NaNbO3 | — |
| `joint_radial_input`, `fused_folded_backward`, `gpu_neighbour_list` | rounding, rounding, exact (same pairs) | MH-1 class (folded) | −6 to −15% together | — |
| `block_linear` | rounding (GEMM grouping) | cuEq `Linear` (internal shared weights) | −11.8% NaCl (MH-1) | — |
| `grouped_linear` | rounding | with `block_linear` | MH-1 −2 to −5%, OMOL −4% | — |
| `merged_residual` | rounding | with `block_linear` + `density_first` | included above | — |
| `gated_rows` (residual products, gate and `linear_2` on per-segment rows; FastMACE optimisation round 1, OPT-T02 stage A) | exact (bitwise on the CPU and in the reproducible GPU mode) | with `merged_residual`, MACE's `GatedEquivariantBlock` in the folded MH-1 forward | MH-1: NaCl −4.4 % (sm_89) / −3.7 % (sm_120), NbCl5 −9.0 % / −10.6 %, on top of the rest | default on by owner decision, 20 October 2026; switched off for any model that fails the 19-label gate in the combined acceptance |
| `segment_gemm` (`linear_2`'s products in the `gated_rows` chains write straight into the joined output, Triton gathered-row GEMM where the shared shape rule picks it; OPT-T02 stage B) | rounding (another GEMM; deterministic, no atomics) | with `gated_rows`, float32, not `reproducible=True` | MH-1 NbCl5 −4.8 to −5.2 % (sm_89) / −4.2 to −4.3 % (sm_120) on top of `gated_rows`; NaCl neutral (stock path below the rule's threshold) | default on by owner decision, 20 October 2026; float64 and the reproducible mode keep the stock path; switched off for any model that fails the 19-label gate in the combined acceptance |
| `grouped_launch` (a block linear's small products of equal weight shape in one Triton launch, forward and input gradient; FastMACE optimisation round 1, OPT-T03 candidate 2) | rounding (another GEMM; deterministic) | with `block_linear`, float32, not `reproducible=True`; by default the generic MACE family only (every interaction MP-0's residual or the density family's, not PolarMACE) | generic NaCl −4.5 to −5.4 % (sm_120, five labels) / −4.1 to −4.3 % (sm_89 L4, OMAT-0); NbCl5 neutral (products above the shape rule's threshold); MH-1 NaCl −1.8 to −2.3 % (sm_120), +0.3 to +0.5 % (sm_89) | default on for the generic family by owner decision 57, 24 October 2026; MH-1, MACE-OFF, POLAR-1 and OMOL-0 stay off unless requested |
| `fx_e3nn_linear` (e3nn's `o3.Linear` code as its own fx graph instead of the TorchScript a checkpoint restores, so dynamo traces it; FastMACE optimisation round 4, OPT4-T02) | rounding (eager bitwise on the CPU; compiled, Inductor fuses what ran eagerly; deterministic) | a model with TorchScript `o3.Linear`; by default the qualified POLAR-1 checkpoints (S, M, L), hybrid float32, not `reproducible=True` | POLAR-1-S: PolarMACE's forward compiles (31 graphs → 5, 1,226 → 642 kernels per NaCl force call); with `polar_graph_sums` NaCl −24 % (sm_120) / −19 % (sm_89 L4), NbCl5 −18 % / −16 % | default on for POLAR-1-S, -M and -L by owner decision 58, 24 October 2026 (L after its local NbCl5 gate); other models off unless requested |
| `polar_graph_sums` (PolarMACE's one-graph per-graph sums as a sum over atoms and a broadcast instead of `scatter_add_` and `x[batch]`; OPT4-T02) | rounding (another summation order; deterministic, no atomics) | PolarMACE, one graph per batch; by default as `fx_e3nn_linear` | alone NaCl −8 %, NbCl5 −9 % of the force call (the float64 `index_put` backward); none on top of `fx_e3nn_linear` | default on with `fx_e3nn_linear` by owner decision 58 |
| `streaming_conv` (OEq PR 223 kernels, NVRTC) | exact (summation order; atomic sums, not bitwise reproducible) | folded MH-1 class, `GENERIC_*` with OEq fusion | −7.7% NaNbO3; generic −16 to −28% with reassociation | OEq present; per-engine edge counter (§8.1) |
| `chunked_conv_backward` (the streaming backward in L2-sized edge chunks, each chunk's weight gradient read at once by the folded layer's input gradient on a side stream; optimisation round 6, OPT6-T02) | exact (the same kernels per edge; atomic sender sums as `streaming_conv`, not bitwise reproducible) | with `streaming_conv` and `fused_folded_backward` (MH-1 class); weights of at most 4 L2s run unchunked | NbCl5 −2.0 to −3.3 %, NaNbO3 −3.1 to −3.6 % (sm_120); NaCl +0.8 % | default on by owner decision 60, 27 October 2026 |
| `radial_tables` (per-species-pair Chebyshev tables for the radial chain r → `[h c, c]` and the edge density, fitted at setup; optimisation round 6, OPT6-T04/T05) | certified (§6.1; deterministic) | factored folded MH-1 class, float32 | stub ceiling per force call NbCl5 −7.2 %, NaNbO3 −16 %, NaCl −10 % (sm_120) | default on by owner decision 62 (float32, not reproducible, with `fold_cutoff`); class by owner decision 61, 27 October 2026 |
| `generic_species_skip` | exact (absent columns multiply zeros) | MP-0/density family | −7 to −39% | fixed species superset |
| `generic_fused_radial_backward` | rounding | MP-0/density family | −3 to −5% large cells | — |
| `generic_chunked_conv_backward` (interaction 2's streaming convolution on MH-1's folded path, `[W, 0]` on `[h c, c]`, so the backward runs in L2-sized chunks and the `[E, weight_numel]` weight gradient is never formed; OMAT-0 capacity, 2 November 2026) | rounding (`(h c) Wᵀ` instead of `(h Wᵀ) c`; atomic sums as `streaming_conv`) | generic residual/density family with `streaming_conv` | OMAT-0 MD step −4.7 to −8.4 % (RTX 4000 Ada), −2.8 to −5.1 % (RTX 5060 Ti); RTX A4000 (6 MB L2) 3–4× slower; capacity SrTiO3 4.1×, NaCl 3.2× (Ada) | default on only on "NVIDIA GeForce RTX 5060 Ti" and "NVIDIA RTX 4000 Ada Generation" (decision 66, `planner.fixed.DECISION_66_GENERIC_CHUNKED_GPUS`); `docs/validation/omat0/omat0-capacity.md` |
| `generic_recompute_conv_weights` (no stored weights: the forward builds them on chip, the backward rebuilds each chunk's weights with one GEMM into a slot) | rounding (cuBLAS chunk GEMM vs the on-chip product) | with `generic_chunked_conv_backward`, float32 only | step +4.5 to +9 % (Ada), +11 to +15 % (RTX 5060 Ti) over the chunked path; frees `weight_numel` floats per edge | off; rule 12 rescue rung after MH-1's `recompute_conv_weights` (decision 67); float64 plans refused |
| `generic_radial_tables` (MH-1's certified radial tables, decision 61, for the generic family: one table per interaction, per species pair where the embedding is species-dependent, r → `[h c, c, density]`; OMAT-0 capacity, 2 November 2026) | certified (§6.1; deterministic) | generic residual/density family with `generic_fused_radial_backward`, float32, not `reproducible=True` | OMAT-0 step 0.82–0.86× (whole system), 0.86–0.91× (tiled) on the RTX 4000 Ada and RTX 5060 Ti; forces within 3.5e-5 eV/Å of float32; NVE drift unchanged | default on only on "NVIDIA GeForce RTX 5060 Ti" and "NVIDIA RTX 4000 Ada Generation" (owner, 2 November 2026, decision 66); float64 resolves to off; `docs/validation/omat0/generic-radial-tables.md` |
| `polar_uvu` (PR 1711) | rounding (GEMM form) | PolarMACE | 4–5× faster with the MH-1 rewrites (step time ÷ 4–5) | — |
| `pipelined`, `async_output` | exact (bit-identical) | engine features (§8.1) | −3.0% NaCl (pipelined); asynchronous output bit-identical | — |
| `unfuse_full_addmm`, `first_derivative_only`, expandable segments allocator | exact | engine | avoids 2 × 32 MB copies per step | torch ≥ 2.14 |
| `float64_energy` (energy accumulation, §7.3; new in the library) | reference treatment (§12.1), not a model rewrite | all | correctness, not speed: stock float32 NbCl5 energies off by 45–102 meV/atom with all-electron E0 (POLAR-1, OMOL-0); MACE-OFF organic set 0.08–11.4 meV/atom | always on; applied to the float32 reference too (§12.1) |
| `single_graph_strain` (MACE PR 1498 backport) | rounding | all, stress path | adopted in the research engine on 16 October 2026 (`langevin/upstream_mace.py`, `--single-graph-strain`, `auto` = on for both engines; acts only where stress is computed) | one graph; default `auto` here too |

The exactness column is the initial classification from each rewrite's
algebra; the self-check record of the rewrite (§6.4) is authoritative and
overrides it.

Opt-in only:
- PR 1762 Fukui regularisation and charge-dipole cap (changes the model);
- in-place block linear (slower);
- fused radial hidden (no gain).

Never enable:
- Inductor programmatic dependent launch (wrong forces inside CUDA graphs);
- spin restriction (changes the model, worse).

The experiment's markdown files record the rejected screens (no gain or wrong
answers); copy their summary into `docs/optimisation-ledger.md` so the work is not
repeated. Among them (16 October 2026): OpenEquivariance 0.7's own symmetric
contraction, loaded with MH-1's basis and weights, is exact but 27–79× slower
than cuEquivariance's fused `uniform_1d` kernel even compiled
(`studies/further_optimizations/2026-10-16-oeq-symmetric-contraction.md`); the
hybrid keeps cuEq for the symmetric contraction.

### 6.3 Surviving upstream changes

- **Versioned registry.** Each record carries `mace_versions` and
  `source_hashes`. On a different MACE version, a record whose hashes do not
  match is disabled, the reason is logged, and the planner continues without
  it. The library never silently patches unknown source.
- **Port per MACE release.** A `tools/port_rewrites.py` workflow diffs the new
  MACE forwards against the hashed ones, re-runs the self-checks and the gate,
  and records new hashes.
- **Upstream where possible.** The block linear, grouped linear and merged
  residual rewrites, the PR 1711 GEMM form and the reassociated first
  interaction are generic enough for MACE itself. The less local patching, the
  fewer hashes to maintain.
- **Source rewrites** (the PR 1762 port pattern): `inspect.getsource` plus
  exact-once block replacement plus an `exec` namespace. Allowed only with the
  hash check and the self-check.

### 6.4 Self-checks

Every rewrite runs a numerical check at setup on random inputs, in float64
where possible, as `_block_linear` and `merge_residual_linears` do today.
- **Per module:** relative error ≤ 1e-10 (float64) or 1e-4 (float32).
- **Per model:** after all rewrites, energy, forces and stress on the user's
  structure (§12.3).

---

## 7. Public API

### 7.1 Model and force field

```python
import fastmace as fm

model = fm.load("mh1", head="matpes_r2scan")          # label, path or URL; hashes recorded
model = fm.load("/path/MACE-omol-0-extra-large-1024.model", head="omol",
                charge=0, spin_multiplicity=1)
ff = fm.ForceField(model, atoms, precision="float32", rewrites="auto",
                   species=["Na", "Nb", "Cl", "O"])    # declared superset (§8.5)
e, f, s = ff.evaluate(atoms, stress=True)               # one-off evaluation
```

**Execution copies and templates** (opt-1 OPT-T06). Each plan converts its own copy of the pristine model
(decision 24). On CUDA with the cuEq or hybrid backend, the first conversion of each kind is kept as a template:
checkpoint hash, head, backend, dtype, device, conversion options, process patches and package versions. The
templates belong to the `LoadedModel` that `fm.load` returned (`LoadedModel.template_cache()`, since setup fix
F3), so every `ForceField` made from it shares them, including one built for a new composition. Every later
plan of that kind gets a private deep copy of the template instead of a reload and conversion; this includes the
gate's reduced cell and fallback candidates. No module, tensor or calculator state is shared between plans
(§8.7). Self-checks run on every copy as before.
- A `LoadedModel` keeps at most two templates.
- `ff.release_templates()` frees them for every `ForceField` of that `LoadedModel`.
- `EngineConfig(model_template_cache=False)` turns the cache off.

### 7.2 Simulation

```python
ff = fm.ForceField(model, atoms, species=["Na", "Nb", "Cl", "O"])   # superset and precision live here (§7.1)
sim = fm.Simulation(
    atoms, ff,
    integrator=fm.integrators.Langevin(timestep_fs=0.5, temperature_K=300, friction_per_fs=0.02),
    workload=fm.Workload(species_changes=True, mc_moves=("swap",), max_temperature_K=1000,
                         min_mc_temperature_K=250, host_callbacks_every=1000),  # other fields: defaults (§9.2)
    output=fm.Output("run/", trajectory_every=20, thermo_every=20, restart_every=10_000),
    seed=1,
)
# or, from a model: fm.Simulation(atoms, model, species=["Na", "Nb", "Cl", "O"], integrator=..., ...)  (§7.6)
print(sim.plan)                     # tier, rewrites, reasons, fallbacks (§9)
anneal = fm.protocols.ramp(0, 1000, ps=20) + fm.protocols.ramp(1000, 250, rate_K_per_ps=-3)  # 270 ps, 540,000 steps at 0.5 fs
sim.run(schedule=anneal, steps=200_000)          # the first 200,000 steps, then pause
sim.run(steps=100_000, mc=fm.mc.Swap(pairs=[("Na", "Nb")], every=100, attempts=50,
                                     velocities="exchange"))  # steps 200,000–300,000 of the schedule, with MC (§10.4; the default for swaps)
sim.run()                                        # the remaining 240,000 steps; the swap block stays attached (§7.6)
# or: sim.run(mc=None)                           # the remaining steps without MC
# or, in one call: sim.run(schedule=anneal)      # the whole schedule
sim.save_restart("run/restart.npz"); sim = fm.Simulation.resume("run/")
```

**Schedules and step counts.** A schedule fixes the run's length. `run`
without `steps` runs it to the end. With `steps`, it advances that many steps
along the schedule and pauses; the next `run` (or a resume) continues from
the recorded schedule position. `steps` beyond the remaining schedule raises
before any step; nothing is truncated or extended silently. A finished
schedule has no remaining steps, so a further run needs a new `schedule=`.
Without any schedule, the thermostat target stays at its current value.

### 7.3 Units, conventions and buffer ownership

- **Units:** ASE units throughout: Å, eV, eV/Å, fs at the API (ASE time units
  internally), K, amu, stress in eV/Å³, pressure in GPa only where named
  `_GPa`.
- **Stress:** the force field returns the potential (virial) stress only, as a
  full 3×3 tensor in ASE's sign convention (`pressure = −trace/3`). The
  kinetic contribution is added by the integrator for pressure observables
  and the barostat, as ASE's `get_stress(include_ideal_gas=True)` does. The ASE
  front end returns Voigt form like any ASE calculator.
- **Precision:** two separate settings. *Force precision* (`float32` default,
  `float64`) is the model evaluation. *Integration precision* is always
  float64 for positions, momenta, cell and barostat state; forces and stress
  are cast to float64 before they enter the integrator (as `GPULangevin` and
  `gpu_npt` do today). Provenance records both. MC energy evaluations have
  their own precision (§10.4).
- **Energy:** the scalar potential energy is always accumulated in float64,
  for MD observables, relaxations, quenches, `evaluate()` and MC alike (§4.3:
  float32 totals are off by up to 102 meV/atom for all-electron E0). The E0
  table is kept in float64 (an exact upcast of the checkpoint's table) and
  summed in float64. Every other contribution is computed in the force
  precision, which autograd differentiates, and reduced over atoms, pairs
  or k-vectors into float64: per-atom interaction energies, pair repulsion,
  the embedding readout energy (OMOL-0), PolarMACE's local electron energy,
  electrostatic energy (per-k and self terms) and field term, and D3. The
  energy is never taken from MACE's `out["node_energy"]`, which in 0.3.16
  omits the embedding readout energy of `ScaleShiftMACE` and the electron,
  electrostatic and field energies of `PolarMACE`. Per-atom energies
  (`needs_per_atom`) state which terms they resolve; terms that are not
  atom-resolved (k-space electrostatics) are reported separately.
- **Energy and forces are one scalar.** Forces are −∂E/∂x and stress is the
  strain derivative of this same float64-reduced energy, by autograd through
  the reduction. Taking forces and stress from the autograd of the float32
  sum and reducing the same terms separately into float64 for the energy is
  equivalent and is the implementation (1b-T03): it keeps forces and stress
  bitwise equal to 1a by construction. The gradient of a sum is 1 for every term whatever the
  precision of the accumulator, so the float64 reduction changes no force
  or stress bit compared with a float32 sum. E0 is a constant per species
  and contributes no gradient. What remains is the float32 rounding of the
  forward pass itself. It affects energy, forces and stress alike, as in any
  float32 model, and the gate bounds it (§12.3). The float64 reduction makes
  finite-difference checks and the NVE conserved quantity *more* consistent:
  a float32 total moves in steps of 2 meV (pseudopotential E0) to 2 eV
  (all-electron E0) for NbCl5 3×3×3. Finite-difference tests run against the
  float64 reference, whose energy and forces are consistent to float64
  rounding.
- **Buffer ownership:** engine methods that return tensors from inside a step
  return views of engine-owned buffers, valid until the next step or event,
  and say so in their names (`*_view`). `ForceField.evaluate` and every
  public observable return copies (host arrays by default). Users never write
  to engine buffers; species, masses and charges change only through events
  (§8.5).

### 7.4 ASE front end

```python
atoms.calc = fm.ase.FastMACECalculator(model, rewrites="auto")   # compiled tier, no graphs
```

This front end has the same results contract as the reference (§12.1). ASE
drives the integration, so the speed-up is smaller (no graph engine); document
that. `fm.calc("mh1", ...)` is a one-line wrapper around it (§7.7, level 0).

### 7.5 CLI

- `fastmace run CONFIG.toml`: protocols as today's `main.py` profiles.
- `fastmace md --model mh1 --head matpes_r2scan --structure NaCl.cif
  --ensemble NVT --temperature 300 --steps 1000000 [--repeat 3 3 3]
  --output run/`: the CLI face of `fm.md` (§7.7) for constant-temperature
  runs. Flags: `--model`, `--head`, `--charge`, `--spin-multiplicity`,
  `--structure`, `--ensemble`, `--temperature`, `--pressure` (GPa),
  `--cell-mode`, `--timestep` (fs), `--steps`, `--repeat`, `--dispersion`
  (`none`, `d3bj`) and `--d3-functional`, `--seed`, `--check`, `--output`,
  `--sha256`/`--trusted`. Schedules, MC moves, explicit integrators and the
  full D3 record go through `fastmace init` and `fastmace run`, whose TOML
  holds every `fm.md` argument.
- `fastmace init > run.toml`: a commented TOML for `fastmace run` (model,
  structure, ensemble, temperature or schedule, steps, repeat, output).
- `fastmace relax` and `fastmace quench RUN_DIR`.
- `fastmace check MODEL STRUCTURE`: gate plus self-checks.
- `fastmace bench MODEL STRUCTURE`: step time, breakdown; `--vs-ase` also
  times stock ASE with MACE's calculator on the same structure and GPU, the
  only source of a "× vs stock ASE" figure (§7.7).
- `fastmace info`: environment, GPU, available rewrites and kernels.
- `fastmace doctor`: `info` as a checklist (GPU and driver, torch, MACE,
  cuEquivariance ops and whether its fused kernels load, OEq and which build
  path, Triton, NVRTC, graph-longrange, torch-dftd, cache root, lock mode on
  NFS, model files and their hashes), each failed item with the command or
  document that fixes it. It changes nothing itself.
- `fastmace models`: the versioned label table (labels, heads, file names,
  SHA-256, licences, §7.6).

The CLI prints `FastMACEError` messages (§7.7) without a traceback, exits
non-zero, and points to the run's `metadata.json`; `--debug` shows the
traceback.

### 7.7 Adoption layer

The ASE `Calculator` interface cannot deliver the graph engine's speed: ASE
calls the calculator once per step, so a calculator gets the compiled-tier
gain at most (§8.2). Full speed needs FastMACE to own the MD loop. Three thin
levels over the same `ForceField`, `Workload`, `Simulation`, plan, gate and
restart make that easy to adopt; none is a second engine, and each
exposes the expert objects it built.

**Level 2: one call (`fm.md`).**

```python
import fastmace as fm

run = fm.md("NaCl.cif", model="mh1", head="matpes_r2scan", ensemble="NVT",
            temperature_K=300, timestep_fs=0.5, steps=1_000_000, repeat=(3, 3, 3),
            output="run/", seed=1)
print(run.summary)                 # the one-screen report below
run.plan, run.gate                 # the Plan and gate records (§9.5, §12.3)
sim = run.simulation               # the expert handle, a normal Simulation
```

- Arguments: a structure (path or `Atoms`), `model` (label, path or URL, with
  `head=`, `charge=`, `spin_multiplicity=`, `sha256=`/`trusted=` as for
  `fm.load`), `ensemble` (`"NVE"`, `"NVT"`, `"NPT"`), `temperature_K` or
  `schedule=`, `pressure_GPa` and `cell_mode` for NPT, `steps` (optional
  with a schedule), `mc=` and `min_mc_temperature_K`, `dispersion=`
  (`None`, `"d3bj"` or a full `fm.D3(...)` record, below), `repeat`,
  `output`, `seed`, `check` (as `Simulation`), and `integrator=` to choose
  one explicitly. `ensemble="NVT"` means Langevin (§10.2), `"NPT"` the BAOAB
  barostat.
- Heads: a multi-head checkpoint (MH-1) without `head=` raises `ValueError`
  listing its heads, here and in `fm.load`, `fm.calc` and the
  `fastmace.ase` classes; there is no default head. A label may name a head
  itself, as the research engine's labels do (`mh1-r2scan` = MH-1 with
  `matpes_r2scan`, `common.py` `MODELS`).
- **Dispersion.** "D3(BJ)" alone does not identify the correction (§11.2).
  `dispersion="d3bj"` is shorthand that `fm.md` expands to a complete
  `fm.D3` record before planning: Becke–Johnson damping, the functional
  parameter set of the head's training functional from the label table
  (`pbe` for the PBE-trained models and heads, `r2scan` for `matpes_r2scan`
  and MatPES r2SCAN), and the §11.2 defaults for pair cutoff, CN cutoff, ATM
  term, smoothing, precision and unit constants. The expanded record is
  part of the force-field identity, printed in the report and recorded in
  `metadata.json`. `"d3bj"` raises `ValueError`, naming the reason, when
  the training reference already contains dispersion (MACE-OFF, trained on
  ωB97M-D3(BJ); OMOL-0 and MH-1's `omol` head, ωB97M-V with its VV10
  nonlocal term; any head whose label-table functional includes
  dispersion), since adding D3 would count
  it twice, and when the training functional is unknown (POLAR-1, a model
  loaded from a path or URL), where only an explicit
  `fm.D3(functional=…, ...)` is accepted. An explicit `fm.D3` is used as
  given, and a functional that differs from the training functional is a
  report warning.
- It builds the `ForceField` (float32), `Workload`, integrator, `Output` and
  `Simulation`, then runs. It infers what it can from the arguments, since
  it knows the whole run before planning: the species superset (structure
  plus every species in `mc=`); `max_temperature_K` (maximum of the
  schedule, or `temperature_K`); `mc_moves` and `species_changes` from
  `mc=`. An explicit argument always wins.
- MC temperature window: with a constant `temperature_K`,
  `min_mc_temperature_K` is that temperature, which must then be positive:
  `temperature_K=0` with `mc=` raises `ValueError`, as does any
  `min_mc_temperature_K ≤ 0` in `Workload` (§9.2), since the ΔE floor
  0.01 k_B T_min would be zero (§12.3). With a schedule and `mc=`,
  `min_mc_temperature_K=` is required (`ValueError` naming it otherwise),
  because no value can be inferred safely: the schedule minimum of an
  anneal from 0 K would make the ΔE floor zero (§12.3), and every MC block
  near the start would sit near 0 K. `fm.md` then attaches the move set
  only over the step ranges whose target is at or above it, through
  `run(mc=…)` and `run(mc=None)` (§7.6), and records the ranges.
- Defaults are the production settings: timestep 0.5 fs, Langevin friction
  0.02 per ASE time unit (about 0.001965/fs, `main.py`'s `--friction 0.02`), trajectory and thermo every 20 steps, restart every 10,000,
  seed recorded, `reproducible=False`, D3 off (the report recommends it
  where §11.2 does, as for the molecular chlorides in NPT). Engine, backend,
  rewrites and precision come from the planner (§9). The gate follows
  `Simulation`'s default (§12.3, open decision 3).
- `repeat` defaults to `(1, 1, 1)`: the input cell is never expanded
  silently. The report warns when the cell is small for the cutoff or the
  atom count is small for the GPU.
- Physics is never chosen silently: model, head, charge/spin, D3 and the
  reference corrections are what the user passed or the documented default,
  and the report shows them. Model licences are shown at load (§4.3).

**Level 1: an existing ASE MD script (`fastmace.ase` integrators).**

```python
from ase import units
from fastmace.ase import Langevin

dyn = Langevin(atoms, 0.5 * units.fs, temperature_K=300, friction=0.02 / units.fs,
               model="mh1", head="matpes_r2scan", trajectory="run.traj",
               logfile="run.log", loginterval=20)
dyn.attach(print_energy, interval=1000)
dyn.run(1_000_000)
print(dyn.summary)                 # the same report as fm.md
```

- **Arguments.** The argument names and **units of the ASE 3.27 class**:
  each class accepts an explicit list of the ASE constructor's arguments
  and raises `TypeError`, naming the argument, for any other. For
  `Langevin`: `atoms`, `timestep`, `temperature` or `temperature_K`,
  `friction`, `fixcm`, `trajectory`, `logfile`, `loginterval`,
  `append_trajectory`. Refused: `rng` (the noise comes from FastMACE's
  device generator; pass `seed=`), and `communicator` (`comm`) other than ASE's
  default, and `master` (FastMACE runs one process per structure). The other classes list
  theirs in the API reference, and a CPU test compares each list with
  `inspect.signature` of the ASE class, so an ASE argument that is neither
  supported nor refused fails CI. FastMACE additions are keyword-only:
  `model=` or `force_field=`, `head=`, `seed=`, `output=`, `check=`,
  `dispersion=` (as `fm.md`). `friction` is in ASE's inverse time units as
  in ASE; `friction=0.02` without `/units.fs` would be about 0.002/fs, as
  it is in ASE. The shim never reinterprets an ASE argument.
- Mapping (ASE 3.27 class → FastMACE integrator): `Langevin` → Langevin and
  `LangevinBAOAB` → BAOAB (phase 1b); `VelocityVerlet` → NVE,
  `NoseHooverChainNVT` → NHC, `Bussi` → CSVR, the MTK classes → MTK, and
  `NVTBerendsen`/`NPTBerendsen` → Berendsen (phase 4, with those
  integrators, §10.2). An ASE class without a mapping raises with the list
  of mapped ones; the shim never falls back to ASE's own loop silently.
  Phase 4 maps `VelocityVerlet`, `Bussi`, `NoseHooverChainNVT`,
  `IsotropicMTKNPT`, `MTKNPT`, `NVTBerendsen`, `NPTBerendsen` and
  `Inhomogeneous_NPTBerendsen` with ASE 3.27's arguments and units. Each
  argument list is held as data and checked against `inspect.signature`.
  `rng` is refused in favour of `seed=`.
- Optimisers (decision 48): `fastmace.ase.LBFGS` and `fastmace.ase.FIRE`
  take ASE 3.27's arguments. `use_line_search=True`,
  `position_reset_callback` and the deprecated `maxmove` are refused. They
  take `Atoms` or `ase.filters.FrechetCellFilter`, and the other cell filters
  are refused with the reason. In `Simulation` the same optimisers are
  `fm.optimize.LBFGS` and `FIRE`, and `FrechetCellFilter` is
  `fm.optimize.FrechetCell`.
- ASE members: `run`, `step`, `atoms`, `nsteps`, `dt`, `attach`,
  `observers`, `todict`, `get_time`. `run(n)` advances in graph chunks
  between observer calls. FastMACE members: `summary`, `plan`, `gate`,
  `metadata` (the dict written to `metadata.json`), `simulation` (the
  underlying `Simulation`, for restart and the expert API) and `output`
  (the directory). The plan and gate are made at the first `run` or
  `step`, which prints the report (below).
- **Files.** `trajectory=` and `logfile=` keep their ASE meaning: one ASE
  `Trajectory` file at that path (appended with `append_trajectory`) and
  ASE's log format, so scripts that read them work unchanged; the
  trajectory is written by FastMACE's asynchronous writer (§5 `io`). Every
  other file goes to `output=`, by default `<trajectory stem>.fastmace/`
  next to the trajectory, else `./fastmace-run/`: `metadata.json`, the
  report, and restart files every 10,000 steps. These shim files are not
  the chunked `fm.Output` files of §10.6. A shim run resumes through
  `fm.Simulation.resume(output)`, which continues in the chunked format of
  §10.6 in that directory; the interrupted ASE trajectory is left as it is,
  may hold frames past the restart point, and `metadata.json` records the
  step of its last frame.
- **Observers.** Before each observer call the shim writes the committed
  state into `atoms` (positions, momenta, cell, species, masses, initial
  charges) and attaches a results cache as `atoms.calc`, so
  `get_potential_energy()`, `get_forces()` and `get_stress()` return the
  engine's values for that step without a new evaluation. After the call
  it compares `atoms` with what it wrote. A change of positions, momenta,
  species, masses or charges is applied as one event through the
  transaction contract (§8.5; species must be in the superset, rule 2). A
  change of atom count, PBC, cell (except through a barostat), constraints,
  `info` keys the plan reads (charge, spin) or `atoms.calc` raises, since
  each would invalidate the plan. Observers every few steps keep the graph
  tier (rule 9, decision 38). The report states their interval.
- Constraints follow §4.4: `FixAtoms` is refused until the integrators
  support it in phase 4, with that reason; no other tier is substituted.

**Level 0: a calculator (`fm.calc`).** `atoms.calc = fm.calc("mh1", head="matpes_r2scan")` wraps
`FastMACECalculator` (§7.4). It records, and prints once, "compatibility
tier: compiled, no CUDA graphs; use `fm.md` or `fastmace.ase` integrators for
full speed". It is the right choice for phonopy, NEB, custom optimisers and
any tool that needs only an ASE calculator.

**The report.** `fm.md`, the `fastmace.ase` integrators and the CLI keep
one report in `summary` and `metadata.json`, built in two stages. At setup
they print everything known then: header, physics, plan, gate, setup time,
outputs and warnings, with `speed: pending`. Once the steady-state step
time is measured (a fixed window after warm-up), the speed line is filled
in, printed once as a single line and written to `metadata.json`; at the
end of the run it is replaced by the whole-run mean. For a constant-T NVT
run:

```text
FastMACE 1.0  |  mh1 (head matpes_r2scan)  |  NaCl, 216 atoms (3×3×3), 3D PBC, NVT Langevin 0.5 fs
physics: no dispersion; charge/spin not used by this model
plan:    graph, hybrid cuEq/OEq, float32, 14 rewrites on, 2 off (reasons in run.plan)
gate:    passed (2 rattled; 0.04 meV/atom, 2e-6 eV/Å; floors §12.3)
setup:   42 s (load, rewrites, compile, capture, gate)
speed:   pending → 4.1 ms/step, measured over steps 1,000–2,000 of this run
output:  run/trajectory.000.traj, run/thermo.csv, run/metadata.json
licence: ASL (non-commercial), from the label table (§4.3)
warn:    none
```

- Warnings are conditional on the run: the NPT barostat note only for NPT
  runs, the experimental labels only for the options in use, the model
  issues of §4.3 only for the model and structure that have them. The
  licence line is read from the label table (§7.6) for every model, never
  written by hand.
- Speeds are **measured, never predicted**: the generality rule (§9.1)
  forbids per-GPU tables, so there is no "expected speed-up" figure. The
  speed line is the run's own steady-state step time; until runtime
  calibration (phase 5) there is no comparison between tiers. A ratio against stock ASE appears only when the user
  measured it with `fastmace bench --vs-ase`.
- When the run did not get the graph engine, the report says why and what
  would change it, without estimates: not 3D-periodic (reference tier with
  `allow_reference=True`, or MACE's calculator); a constraint (refused until
  phase 4); OEq unavailable (plain cuEq; `fastmace doctor` names the
  missing piece); a model outside the rewrite registry (plain or reference
  path); padded edges that are not exact (compiled tier, rule 4); a failed rewrite
  self-check (that rewrite off); a fallback after a failed gate (§9.4).
- NPT, MC and D3 carry an "experimental: not yet qualified" line until their
  qualification studies are recorded (§10.2, §10.4, §11.2). `reproducible`
  stays off; requesting it gives the refusal of §9.3.

**Errors.** Python calls raise the exceptions of §7.6, never print-and-return:
scripts must fail visibly and be able to catch them. Every `FastMACEError`
message has the same shape, so no traceback needs reading: what happened;
why, in one or two sentences; what to do next; where the record is
(`run.plan`, `run.gate`, `metadata.json`); and `fastmace doctor` for
environment problems. The CLI prints this message alone (§7.5).

**Acceptance (phase 1b; phase 4 for the remaining integrators).**
- An ASE Langevin script ported by changing the import and adding `model=`
  runs on the graph engine, and gives the same plan as the equivalent
  `Simulation`.
- `fastmace doctor` then `fastmace md` on a structure and a model gives a
  complete run with plan, gate and speed report, without Python.
- `fm.md`, the `fastmace.ase` integrators and `Simulation` give the same plan
  and gate for the same inputs, the same trajectory within the plan's
  reproducibility (statistically for hybrid runs; bitwise where
  `reproducible=True` is qualified, §9.3), and the same steady-state step
  time within run-to-run noise in alternating A/B blocks (a gap is a bug:
  the layer is thin).
- An observer that changes positions, momenta, species, masses or charges
  gives the same state as the same change made through `Simulation`
  events; one that changes the atom count, PBC, cell or constraints raises.
- Each `fastmace.ase` class's argument list matches `inspect.signature` of
  its ASE 3.27 class (every argument supported or explicitly refused); a
  ported script's `trajectory=` file reads back with `ase.io.read` frame for
  frame.
- `dispersion="d3bj"` expands to the full §11.2 record matching the
  label table's functional, and raises for MACE-OFF, OMOL-0, MH-1's `omol`
  head and models with an unknown functional.
- Every path that does not reach the graph engine either runs and reports
  the reason, or refuses with the reason; never silently.

### 7.6 API contracts

Section 7.2 is illustrative; these contracts are binding.

**Exceptions** (`fastmace.errors`), all derived from `FastMACEError`:
- `PlanRefused`: every hard-constraint reason of §9.3 at once, and a `run`
  request the plan does not cover (schedules and MC blocks, below). One
  refusal raises one exception carrying all its reasons;
- `SpeciesNotInSuperset(PlanRefused)`: rule 2, a species outside the
  `ForceField`'s declared superset, in the structure, a declared or
  requested move set, or a host-side mutation (§8.5). When rule 2 is among
  the reasons of a refusal, the exception raised is this subclass, still
  carrying every reason, so `except PlanRefused` catches it too;
- `GateFailed`, `GateUnavailable`: with the gate record (§12.3);
- `ModelElementError`: a declared superset species the model has no
  embedding for (§8.5);
- `RestartMismatch`: a different force-field identity, or a plan change
  without `allow_plan_change` (§10.6);
- `ProcessConstraintError`: §8.7;
- `CheckpointTrustError`: see loading below;
- `NonFiniteError`: §9.6.

Capacity overflows are internal events (§8.1) and never escape to the user.

**ASE front end.** `FastMACECalculator.implemented_properties` is `energy`,
`free_energy` (equal to `energy`), `forces`, and `stress`, in ASE units with
Voigt stress. It is compiled-tier only (§8.2) and needs full 3D PBC (§4.4).
At the first calculation it builds a minimal plan (compiled mode, backend,
rewrites, force precision; no workload, integrator or MC dimension) through
the same planner, so its hard constraints apply: non-periodic or mixed-PBC
atoms raise `PlanRefused` (rule 1), and there is no `allow_reference`.
Gas-phase molecules need `fm.Simulation` with the reference tier or MACE's
`MACECalculator` (§4.3). `energies` (per atom) is listed only for models
whose energy terms are all atom-resolved; PolarMACE's k-space electrostatics
are not (§7.3).

**Loading and checkpoint trust.** MACE checkpoints are Python pickles:
`torch.load(..., weights_only=False)` executes code from the file. So:
- a **label** (`"mh1"`) resolves through a versioned table shipped with the
  package (official URL, file name, SHA-256, licence). A download is
  verified against that hash before it is unpickled, and stored under the
  cache root (`FASTMACE_CACHE`, §14).
- a **path or URL** is unpickled only with `sha256=` matching the file or
  `trusted=True`; otherwise `CheckpointTrustError`. The decision and the
  hash are recorded in provenance.
- a multi-head checkpoint needs `head=` (or a label that names one);
  otherwise `ValueError` listing the heads. The label table records each
  label's licence and the training functional of each head (§4.3, §7.7).

**Schedules.**
- A schedule is a sequence of segments: `ramp(T0, T1, ps=…)` or
  `ramp(T0, T1, rate_K_per_ps=…)`, and `hold(T, ps=…)`. `+` concatenates;
  the end temperature of one segment must equal the start of the next,
  otherwise `ValueError`.
- Each segment's duration is rounded to whole steps at the run's timestep;
  segment boundaries and the total are recorded.
- The target temperature is piecewise linear in the step and reaches the
  integrator as a device-scalar fill before the step, not as an event (§8.5).
- MC blocks are attached to the simulation (every n MD steps), not to
  segments; a block uses the target temperature at its step. A `run(mc=…)`
  attaches the move set, and it stays attached for later `run` calls until
  one passes another `mc=` or `mc=None` (detach); the attached move set is
  restart state, so a resumed simulation continues with it. `run` refuses a
  block whose target temperature is below `Workload.min_mc_temperature_K`
  (§9.2, §12.3), checked over every step the call will run with the
  attached move set. The schedule position is restart state (§10.6).
- `run` checks the whole schedule before the first step and refuses
  (`PlanRefused`) a schedule whose maximum target exceeds
  `Workload.max_temperature_K`, which sized the skin and the buckets (rule
  11); it never re-plans silently (owner decision 9, 17 October 2026).

**Species superset.** Owned by the `ForceField` (`species=`); the planner and
`Workload` read it from there (§9.2).

**ForceField and plans.** A `ForceField` is plan-free (owner decision 12,
17 October 2026): it holds the pristine model, the precision and the
superset, and every plan builds its own execution copy (rewrites, compiled
code, graphs) from it. `ff.evaluate()` plans for a one-off evaluation; a
`Simulation` plans for its workload; both can use the same `ForceField`, in
any order, under the instance isolation of §8.7 and subject to its process
constraints: the first plan in a process fixes its reproducibility, and the
first compiled plan its backend family. A default `ff.evaluate()` first
therefore makes a later `reproducible=True` plan in that process refuse (the
subprocess runner is offered), and a hybrid plan first excludes a compiled
plain-cuEq plan.

**Simulation inputs.** `Simulation` takes a model or a `ForceField`; with a
model it builds the `ForceField` with `species=` (default: the species of the
initial structure) and float32 (owner decision 11, 17 October 2026). The
ensemble, the cell mode and whether stress is needed are those of the
`integrator=` or `optimizer=` object; `Workload` has no copy of them, nor of
the force precision or the superset (§9.2). `run(mc=…)` is checked against
the declared workload before any step: a move type not in `mc_moves` or a
block whose target temperature is below `min_mc_temperature_K` raises
`PlanRefused`, and a swap pair or transmutable set naming a species outside
the superset raises `SpeciesNotInSuperset` (a `PlanRefused`), because the
gate (§12.3) never covered them.

---

## 8. Execution tiers

### 8.1 Tier 1: CUDA-graph engine (from `GPULangevin`)

**State.** All state stays on the GPU: positions, momenta, cell and RNG
stream. Each step replays captured graphs:
- `ACP`: half kick, positions, neighbour check, edge compaction;
- `B`: forces and second half kick;
- `C`: compaction after a rebuild.

**Edges.** Padded edge capacities come in buckets (`EDGE_MULTIPLE = 256`,
bucket fraction 5%). Padding must be exact (`padding_is_exact`); otherwise
this tier is refused. The compiled model sees the edge dimension as dynamic
from its first call (`EngineConfig.dynamic_edges`, on by default, decision
58), so the padded first capture reuses that graph and a start compiles once.

**Neighbour list.** Skin list with a GPU build; rebuilds are triggered by a
displacement test on the GPU. In variable-cell runs the test is
deformation-aware.

**Pipelining.** The force stage is launched before the host reads the flags,
with early neighbour rebuilds. Output is asynchronous and the noise is
uploaded in batches.

**New in the library:**
- **Re-capture policy.** A bucket overflow or a new atom bucket triggers
  re-capture of the affected graphs (a different species superset needs a
  new `ForceField` and a plan revision, §8.5); keep at most N
  captured variants (dynamo limit 64). Log every re-capture; if they happen
  often, the planner downgrades to tier 2.
- **Overflow before commit.** Every captured stage that can overflow a
  capacity (edges, k-vectors, atoms) sets a device flag. The host checks it
  before the step, or an MC decision based on it, is committed; an overflowed
  step is redone after re-capture from the state before it (today's
  speculative path redoes only the force stage). No observable, trajectory
  frame, restart or MC acceptance is ever produced from an overflowed
  evaluation.
- **Graph-safe events.** In-place edits of species, masses and charges, and
  changes of friction, barostat τ or mass, between replays, under the
  transaction contract of §8.5. They never change shapes or addresses.
  Schedule-driven thermostat targets are device-scalar fills, not events
  (§8.5).
- **Encapsulated counters.** The streaming edge counter becomes part of the
  engine, which removes `bind_streaming_count`.
- **Runtime-state registry.** Every buffer that a stage writes and that lives
  across steps registers itself, together with how it is saved, restored or
  reset: positions, velocities, forces, energy, stress, cell, barostat state,
  mid-step copies (`_v_mid`, `_p_mid`), neighbour-list buffers (edge index,
  shifts, count, build cell, reference positions), derived geometry (volume,
  reciprocal cell), pipeline flags and floors, and sticky flags (neighbour
  overflow, k-vector overflow, whose `|=` never clears by itself). Capture,
  events (§8.5) and restart (§10.6) all work from this registry; a CI test
  compares it with the attributes an engine actually mutates.
- **Capture warm-ups leave no trace.** The eager warm-up runs of a capture
  snapshot and restore every registered buffer, sticky flags included. Today
  `GPULangevinBAOAB._graph_state` misses the force field's volume and
  reciprocal cell, `_v_mid`, `_p_mid`, the pipeline flags and the polar
  overflow flag; a test compares the full registered state before and after
  each capture.
- **Energy-only graph.** MC energy evaluations use a separate forward-only
  compiled and captured graph under `torch.no_grad()`. Calling the force
  graph with forces switched off while positions require gradients would run
  AOTAutograd's joint graph, save activations nobody uses and leave
  `grad_fn` references on output buffers, which break later captures (the
  failure recorded in `polar_gpu._Call`). Compiling it costs setup time,
  reported by the benchmark suite (§12.5).

### 8.2 Tier 2: compiled, no graphs

`torch.compile` (`default` mode, no autotuning; compile time matters) with all
valid rewrites. Shapes may change between calls through dynamic or bucketed
shapes. Use it for frequent topology changes, per-step host logic, the ASE
front end and relaxations with many host decisions.

### 8.3 Tier 3: eager hybrid

Rewrites without compilation. Use it when compilation fails or is
disallowed, and for debugging.

### 8.4 Tier 4: reference

MACE's `MACECalculator` (cuEquivariance float64 or float32) with the
corrections of the reference policy (§12.1). Used for the gate, for unknown
models and as the last fallback, but only where it fits in memory (§12.3);
a reference that cannot run is never chosen as a fallback.

### 8.5 Species superset and mutation events

**Species superset.** Today the reassociation, the species-restricted skip
and the compile key use the species present at setup ("atoms may change
places later, but not to species absent at setup"). The library changes
this:
- The `ForceField` takes a declared species superset (default: species in the
  initial structure). Columns and caches are built for the superset. The
  superset must be a subset of the model's element table (`model.elements`,
  MACE's `z_table`): a species the model has no embedding for raises
  `ModelElementError` at `ForceField` construction.
- A species outside the superset raises before stepping
  (`SpeciesNotInSuperset`; a different superset needs a new `ForceField` and
  a plan revision). Membership is checked
  explicitly at setup and on every host-side mutation (atomic number in the
  superset; one-hot row sums to 1). Today an atom outside the columns gets an
  all-zero one-hot row, which `argmax` silently maps to slot 0 in the
  reassociated first interaction and the species skip zeroes.
- Every species-indexed table is built for the superset **without
  depending on which species are present**. Today
  `_FirstInteractionReassociation` takes each species' embedding from
  `node_feats[onehot.argmax(dim=0)]`, the first atom of that species; for a
  superset species with no atoms that is atom 0 of another species. The
  library computes the `[S, C]` table by applying the node embedding (plus a
  per-structure joint embedding, OMOL-0) to the identity basis of the
  superset columns, and registers it as a derived buffer.
- Neighbour-list sender groups (species order for the segmented first sum)
  are a registered derived buffer read on every rebuild, with a fixed group
  count equal to the superset size. Today `build_gpu` uploads them once, on
  the first fixed-cell build, and counts groups as `max() + 1`, which is too
  small when the highest superset species is absent.

**Derived-state registry.** Updating the species and mass buffers alone is not
enough: the integrators cache mass-dependent coefficients (Langevin's
`c3_unit`, `c4_unit`, `c5_unit` from `sqrt(2 fr / m)`; BAOAB's
`inv_sqrt_masses`), and the force path may hold species-derived buffers
(one-hot `node_attrs`, species index, per-edge species pairs of the factored
radial input, segment boundaries, species-skip indices, per-atom charges of
the polar path). Every component that derives a buffer from species, masses
or charges registers it with its refresh function. An event refreshes all
registered buffers; a test enumerates the registry against a fresh
construction so that a forgotten buffer fails CI.

**Transaction contract.** Every in-place mutation (species swap,
transmutation, mass or charge change, displacement move, and a parameter
change that alters cached coefficients: friction, a barostat τ or mass) is
one event. Schedule-driven changes of the target temperature or pressure are
**not** events: the target is a device scalar that the host fills before a
step, as today's `sqrt_kT` (`gpu_langevin.py`, `set_temperature`), and no
cached coefficient depends on it (Langevin's `c3_unit`, `c4_unit`, `c5_unit`
depend on friction and masses only). A ramp therefore changes the target
every step (§7.6) without quiescing the pipeline. An event runs as follows:
1. **Quiesce:** finish pending work (pipelined force stage, asynchronous
   output, speculative steps) so that the state is a committed step.
2. **Snapshot** the affected state: species, masses, charges, momenta of
   touched atoms, cached energy, forces and stress, the registered derived
   buffers they feed, and the random-number cursor. Displacements also
   snapshot the positions and the neighbour-list state of the runtime-state
   registry (edge index, shifts, unit shifts, count, streaming count, flags).
3. **Apply** the mutation and refresh all registered derived buffers;
   displacements also re-compact the neighbour list and run the rebuild test.
4. **Evaluate** what the decision needs (energy for MC; §10.4), with the
   overflow and rebuild check of §8.1 before any decision.
   - **Rebuild needed at the trial positions** (a displacement beyond the
     skin test): the list is built **around the trial geometry**, in trial
     buffers, before the trial is evaluated. This build runs eagerly, outside
     the captured attempt: the GPU build synchronises the host for the pair
     count and may reallocate (§10.4). The committed list stays in the
     snapshot for a rejection; an acceptance commits the trial list.
     Rebuilding at the restored committed positions and replaying the same
     displacement would loop forever: with the default 0.3 Å skin, every
     displacement beyond 0.15 Å fails the test again.
   - **Capacity overflow:** this is not a rejection. The snapshot is
     restored (random-number cursor included), the capacity grows, and the
     same attempt is redone with the same random numbers. It terminates
     because capacities only grow.
5. **Accept or reject.** Reject restores the snapshot exactly (cached forces
   are restored, not recomputed). Accept keeps the new state; if it was
   accepted on an energy-only evaluation, the cached forces and stress are
   marked invalid.
6. **Resume:** before the next MD half step, energy, forces and (NPT) stress
   must be those of the committed state: invalid ones are recomputed once,
   so an MC block with several accepted moves costs one force evaluation at
   its end. The order is fixed: first species and momenta (`rescale`,
   `redraw`, `redraw_all`), then forces and stress, then every **integrator
   cache derived from forces, stress or momenta**. These caches are in the
   derived-state registry; today that is BAOAB's barostat force `force_eps`,
   computed from the stress and the kinetic energy or tensor, and consumed
   at the start of the next NPT step. A test compares the first NPT step
   after an accepted move with a fresh construction, in hydrostatic and
   full-cell mode.

**Random numbers per outcome.** MC uses three separate seeded device
streams: *proposal* (selection keys, new species, displacement vector),
*acceptance* (one uniform) and *velocity* (Maxwell–Boltzmann draws for
`redraw`; `redraw_all` draws 3N numbers once per block). Every attempt
consumes a fixed count from each stream, whatever its outcome, so stream
positions never depend on decisions. That keeps the captured graph
fixed-shape and the restart state simple.

| Outcome | Proposal | Acceptance | Velocity | Stream positions afterwards |
| --- | --- | --- | --- | --- |
| accepted | consumed, used | consumed, used | consumed, used if the move redraws | advanced by the fixed counts |
| rejected | consumed, used | consumed, used | consumed, unused | advanced by the fixed counts |
| impossible (species absent) | consumed, used to detect it | consumed, unused | consumed, unused | advanced by the fixed counts |
| overflow or required rebuild (§8.5 step 4) | restored | restored | restored | reset to the attempt's start; the attempt is redone with the same numbers |

An overflow is not an outcome of the chain but an execution event, which is
why only it rewinds the streams. The stream positions are restart state
(§10.6). A rejected displacement restores the neighbour-list state as well,
so the next MD step or swap sees the list of the committed positions.
Velocity
handling for changed masses is part of the move definition (§10.4), not a
free policy. A test runs the first MD step after an accepted and after a
rejected event and compares it with a fresh construction from the committed
state (bitwise on CPU float64 and in reproducible mode, §9.3; within the gate
otherwise).

### 8.6 Atom-count changes

Atom capacity comes in buckets with padded "ghost" atoms: zero mass (an
inverse mass of zero in the integrator), excluded from neighbour pairs and an
excluded species column. This needs a proof that ghost atoms contribute
exactly zero, checked like `padding_is_exact`. None of this is in 1.0: the
planner refuses `atom_count_changes=True` (§9.3 rule 7), because no 1.0 move,
integrator or restart entry changes the atom count. Insertion and deletion on
the compiled tier with exact atom counts are 1.x (§10.4); ghost-atom buckets
at graph speed are 2.x (§17).

### 8.7 Process constraints (enforced, not documented only)

- **One backend family per process.** MACE's autograd-compile policy is
  process-wide, so the planner refuses to compile hybrid and plain
  cuEquivariance models in one process. It offers a subprocess runner
  instead, which hands the state to the child through a restart write and
  `resume(allow_plan_change=True)` from the last committed step (§10.6),
  never through pickled live objects; the output chunks of §10.6 continue in
  the child.
- **Allocator first.** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` must
  be set before the first CUDA allocation (torch ≥ 2.14). `fastmace` sets it
  on import if CUDA is not yet initialised; otherwise it warns.
- **Dynamo reset.** `torch._dynamo.reset()` is needed per PolarMACE engine;
  the engine does it internally.
- **Dynamo cache limit.** `torch._dynamo.config.cache_size_limit` (raised to
  64) is process-wide; `fastmace` raises it on first use and never lowers it.
- **Reproducible processes.** `CUBLAS_WORKSPACE_CONFIG` must be set before
  CUDA is initialised, and `torch.use_deterministic_algorithms(True)` is
  process-wide. A process is reproducible or not (owner decision 14,
  17 October 2026): the first `reproducible=True` plan sets both, after
  checking that CUDA is not yet initialised (`ProcessConstraintError`
  otherwise); a later plan of the other kind in the same process is refused,
  and the subprocess runner is offered instead.
- **MACE patches.** The research engine installs two backports by patching
  MACE's modules for the whole process (`upstream_mace.use_single_graph_strain`,
  `use_cueq_conv_fusion`). FastMACE makes both per-model: the strain path is
  part of its own graph preparation (a model rewrite), and conv fusion is
  decided when FastMACE converts the model; it patches no MACE module
  (§5).
- **Instance isolation.** Several `ForceField` objects (and engines) may
  coexist in one process and be evaluated in any order, A → B → A, with
  correct results for each. Each instance owns its geometry and capture
  state: k-vector set and mask, neighbour list, edge counters and graphs.
  Today's polar `_Holder` is shared by every copy of a model ("one crystal per
  model at a time"), and the streaming edge counter is shared between engines
  built from one compiled model; neither is ported as is. Where compiled code
  is shared between instances, per-instance state is passed in as data. If a
  shared component cannot be made per-instance, the instance holds an
  exclusive lease on it: using an instance whose lease was taken raises a
  clear error (never a silently wrong result), and `rebind()` re-installs its
  state and re-captures its graphs. The isolation tests run A → B → A for
  two crystals, for the plain, MH-1 and polar paths.

---

## 9. Planner (intelligent engine selection)

### 9.1 Principle

The planner uses rules, not measurement tables: it decides from properties of
the model, workload and hardware, then verifies with self-checks. An optional
short calibration (`CalibrationConfig(calibrate=True)` through `Simulation(calibration=)`, `fm.md(calibrate=True)` or
`fastmace run --calibrate`; decision 55: the backend dimension only, about 50 steps per candidate,
results never persisted across runs) may break ties, but no per-shape table is ever stored. This is the
project's generality rule. Since decision 66 (owner, 29 October 2026) a rule may name GPU models: a default
that holds only on the GPU models where same-process A/Bs in both build orders measured a whole-step speed-up,
matched by exact device name as in decision 65, and off on every other GPU. No measured value is stored or
looked up at run time; the list of GPU names is the only per-GPU data.

### 9.2 Inputs

```python
@dataclass
class Workload:
    max_temperature_K: float                       # required; drives skin and bucket headroom
    species_changes: bool = False                  # swaps/transmutations
    atom_count_changes: bool = False               # reserved; refused in 1.0 (§9.3 rule 7, §8.6)
    min_mc_temperature_K: float | None = None      # lowest MC target temperature; ΔE gate threshold (§12.3); required when mc_moves is non-empty
    host_callbacks_every: int | None = None        # user Python per n steps; None = no callbacks
    needs_per_atom: frozenset[str] = frozenset()   # energies, charges, ...
    mc_moves: tuple[str, ...] = ()                 # "swap", "transmute", "displace"; drives the ΔE gate (§12.3)
    reproducible: bool = False                     # bitwise within the scope of §9.3; refuses what it cannot guarantee
    mc_energy_precision: Literal["auto", "float32", "float64"] = "auto"   # §10.4; auto = force precision if the ΔE gate passes
```

`Workload` validates itself at construction (`ValueError`, before any
planning; `PlanRefused` stays the planner's aggregate of §9.3): a non-empty
`mc_moves` requires `min_mc_temperature_K` (owner decision 8, 17 October
2026; `None` only for non-MC workloads), which must be positive (the ΔE
floor of §12.3 is 0.01 k_B T_min), and `"swap"` or `"transmute"` in
`mc_moves` requires `species_changes=True`; the two can therefore never
disagree.

`Workload` holds only what no other object knows. Periodicity and constraints
come from the `Atoms` object and are checked against §4.4. The ensemble, the
cell mode and whether stress is needed come from the integrator or optimiser
passed to `Simulation` (§7.6). The force precision and the species superset
have one owner, the `ForceField` (`precision=`, `species=`, §7.1); the
planner reads them from there. Draft 0.4 removed the superset copy from
`Workload`, draft 0.6 the ensemble, cell, stress and precision copies, for
the same reason: two owners of one fact disagree.

Model traits come from `capabilities.py`: interaction classes,
`apply_cutoff`, joint embeddings, PolarMACE, heads, `padding_is_exact`,
`first_inputs_species_only`, all-electron or pseudopotential E0 (§4.3).
Hardware and software traits: compute capability, memory, OEq build
availability, Triton availability, NVRTC, and whether cuEquivariance's fused
kernels actually load. cuEquivariance replaces a requested fused kernel by
its naive implementation with only a `UserWarning` when
`cuequivariance_ops_torch` cannot be imported, for example after a cuBLAS
mismatch (MACE PR 1614). The run is then correct but slow. The planner
refuses such a plan unless `allow_naive_cueq=True`, which is recorded. The
research engine has this guard since 16 October 2026
(`langevin/upstream_mace.py`; it refuses when the symmetric contraction or
the convolution fell back; cuEq's `Linear` layers are naive by design and are
not checked).

### 9.3 Decision rules (initial set)

A plan has five independent dimensions: **execution mode** (the tiers of §8:
graph, compiled, eager, reference), **backend** (hybrid cuEq/OEq, plain
cuEq; the plain backend uses cuEq's fused convolution by default on any CUDA
device, MACE PR 1641, exact to rounding, −2 to −6% MD step in float64 and
−29 to −40% in float32 against the unfused path, and the unfused convolution
in `reproducible=True` plans, see below), **rewrite set**, **force precision** and **MC energy precision**. The
physics (model, head, charge/spin, D3 and its parameters, polar options,
reference corrections) is fixed by the user and is never changed by a rule
or a fallback.

Rules are evaluated in this strict order. Rules 1–7 are **hard constraints**
(correctness or an explicit user requirement): a later rule cannot weaken
them, and a user override cannot lift them, except through the `allow_*`
flag named in the rule, which is recorded. When several hard constraints
refuse, all reasons are reported together in one `PlanRefused` error
(§7.6). Rules 8–12 are **preferences**: a user override replaces the
decision, and the overridden plan is self-checked and gated like any other: the same gate policy as an
unoverridden plan, so an override of an MD plan runs no gate unless `check="gate"` (decision 3), and an MC plan
always runs the move-energy part (decision 35).
Rules 13–15 are **reactions** to checks.

| Order | Kind | Condition | Decision |
| --- | --- | --- | --- |
| 1 | hard | workload outside §4.4 (PBC, constraints, MC moves with constraints), or a variable cell with an integrator that has no barostat, or a PolarMACE long-range source without qualification (`polar-unqualified-source`) | refuse; reference tier only with `allow_reference=True` (never for MC with constraints) |
| 2 | hard | a species outside the ForceField's declared superset, in the structure or in a declared move set (swap pairs, transmutable sets) | refuse |
| 3 | hard | PolarMACE with a plain compiled backend | refuse that combination (zero forces); hybrid, or eager plain |
| 4 | hard | `padding_is_exact` false | no graph mode (padded edges would change energies); compiled with exact edge counts |
| 5 | hard | `reproducible=True` | only a qualified (backend, model family) with its deterministic allow-list (see below); refuse otherwise |
| 6 | hard | cuEquivariance's fused kernels do not load (§9.2) | refuse unless `allow_naive_cueq=True` |
| 7 | hard | `atom_count_changes=True` | refuse: not in 1.0 (§8.6, §17) |
| 8 | preference | `species_changes`, graph mode still allowed | graph mode; species-dependent rewrites built for the superset |
| 9 | preference | `host_callbacks_every` set, `fastmace.ase` observers included | graph mode stays. A replay per callback interval is cheap: in 3-T06's measurement a callback every step added 25 % on NaCl and 3 % on NbCl5, while compiled mode was 1.3–4.7× slower. Compiled mode only under rule 4 or with an explicit `mode="compiled"` (decision 38) |
| 10 | preference | the integrator or optimiser and its cell mode (§9.2) | a barostat integrator or a cell-filtered optimiser (FIRE/LBFGS, never an NPT integrator for a relaxation) enables the stress path and the deformation-aware neighbour test; polar: k-vector headroom |
| 11 | preference | `max_temperature_K` high; `"displace"` in `mc_moves`; D3 on | larger skin and bucket headroom (scaled from expected displacement per rebuild interval); a skin that keeps displacement trial rebuilds rare (§10.4); the D3 list's own skin (§11.2); re-capture budget |
| 12 | preference | estimated memory (model dimensions × edge bucket and nodes, forward and backward, plus the D3 list) above 90 % of free GPU memory | drop the rewrites that declare an edge-sized buffer, largest first, then warn (decision 51; e.g. generic-only POLAR-1-L NPT on NbCl5 ran out of memory on 20 GB). Only when the estimate with them on does not fit: a rewrite whose term saves no memory is never dropped (decision 67). The estimate may carry one global safety factor, the same for every model and GPU, fitted on the 19 labels and recorded with its fit (5-T03); per-model, per-GPU or per-shape factors stay forbidden (§9.1). A rewrite's declared term may be measured or fitted on the GPU for its own path (decision 67: MH-1's fused path, fitted on the capacity scan) |
| 13 | reaction | a rewrite self-check fails | disable that rewrite and every rewrite that requires it, record reasons, re-plan |
| 14 | reaction | the gate fails (§12.3) | fallback algorithm (§9.4); never run silently with a failing gate |
| 15 | reaction | known model–structure warning (§4.3) | warn (not refuse), with the documented diagnosis |

**`reproducible=True`.** Switching away from OEq is necessary but not
sufficient.
- **Message passing is not the obstacle.** In default mode MACE 0.3.16's
  `scatter_add_` (`mace/tools/scatter.py`) uses atomics on CUDA. Under
  `torch.use_deterministic_algorithms(True)`, torch 2.14 switches
  `scatter_add_`, `index_add` and the backward passes of `gather` and
  `index_select` on CUDA to deterministic implementations (they are in the
  "will act deterministically" list of its documentation).
- **What remains unknown** is cuEquivariance's kernels, which the flag does
  not govern, and FastMACE's own kernels. They include the fused
  convolution of the plain backend, whose scatter to the receivers happens
  inside the cuEq kernel; repeated float64 force evaluations of the fused
  and of the unfused plain path both differed at about 10⁻¹⁵ eV/Å
  (16 October 2026, default mode, so not yet with the deterministic flag).

The option therefore means:
- **Scope:** bitwise-identical results for the same hardware model, driver,
  software versions, model, plan, seeds and restart cadence (§10.6). No promise
  across GPUs, releases or platforms (PyTorch makes none either).
- **Enforcement:** plain cuEquivariance backend with the **unfused**
  convolution (`cueq_conv_fusion=False`, owner decision 13, 17 October 2026;
  its messages are summed by MACE's `scatter_add_`, which the deterministic
  flag governs, while the fused cuEq kernel scatters internally and is not
  qualified), hybrid-only rewrites off, `torch.use_deterministic_algorithms(True)`
  (with the required cuBLAS workspace setting, §8.7), and a per-component
  allow-list that names the unfused `ChannelWiseTensorProduct` and every
  other cuEq kernel in use. Any operator or kernel
  without a deterministic implementation makes the planner refuse the plan;
  it never falls back to a nondeterministic one. FastMACE's own kernels
  declare their determinism in their rewrite records. Today's
  `edge_outer_sum` (behind `reassociate_first`) is not deterministic in its
  segmented Triton forward, which uses relaxed `tl.atomic_add` where a
  receiver's segment crosses a 32-edge chunk. Its unsegmented forward uses
  `index_add`, which the flag makes deterministic. In reproducible mode the
  segmented forward needs a deterministic variant (one program per receiver
  segment), or the unsegmented form is used. (The streaming convolution is
  hybrid-only and off anyway.) The GPU neighbour-list build is already
  deterministic (it sorts the atomic pair-search output on a unique key).
- **Qualification:** repeated runs and an interrupted-versus-uninterrupted
  run must match bit for bit (§12.2) before the option is offered for a
  backend and model family. **No family is qualified today**, so until phase
  2 qualifies one (§16) the planner refuses `reproducible=True` with that
  reason. Outside this option, results are compared statistically.

### 9.4 Fallback algorithm

A failure (rewrite self-check, gate, replay or transition check, compilation
error) triggers this algorithm, not a fixed "one tier down". It is automatic
by default; `fallback="refuse"` turns it off, and a failure then raises with
the record (decision 29):
1. **Candidates** are points in the five dimensions above, never a change of
   physics. Each candidate is built from a pristine copy of the loaded model
   (the checkpoint state before any rewrite), not by undoing rewrites in
   place.
2. **Diagnose the dimension.** A failure of only the move-energy part of the
   gate (§12.3) is first tried in the MC energy precision dimension: the MC
   energy path switches to float64 while MD forces keep their precision.
   Otherwise, re-run the failing check on the full rewrite set in eager
   mode. If eager passes, the fault lies in compilation or capture: try
   compiled mode without graphs, then eager. If eager fails, the fault lies
   in the rewrites. Faults can interact (two rewrites that pass alone and
   fail together), so the search is delta debugging (ddmin) over the
   rewrite set, respecting `requires`. It finds a minimal failing subset,
   not just a single culprit, and every rewrite in it and every rewrite that
   requires one is disabled.
   **Budget:** diagnosis uses the reduced gate (one rattled copy, the
   failing quantity only), which is cheap, and has its own budget of 64
   reduced-gate runs per fallback (ddmin over the ~16 catalogue entries of
   §6.2 needs about n log n runs typically and is quadratic in the worst
   case; the 12 of draft 0.4 could not finish). When that budget is spent the
   search stops and the planner tries a fixed **terminal sequence** instead:
   (1) the same mode with every rewrite off, (2) eager with every rewrite
   off, (3) the reference tier if it fits (§12.3). Each is validated as in
   step 4; if none passes, the planner refuses with the full record. Full
   gate runs (step 4) are counted separately, one per candidate to activate.
3. **Backend changes respect §8.7.** A plain cuEquivariance candidate after a
   hybrid model was compiled in the process runs eager or in a subprocess
   (through a restart write, §8.7), never as a second compiled backend in the
   same process.
4. **Validate before activation.** A candidate becomes active only after it
   passes the self-checks and the gate in the mode it will run in (replay
   included, §12.3). The failed plan and every tried candidate are recorded.
5. **Stop.** The reference tier is the last candidate only if it fits
   (§12.3); otherwise, or when the terminal sequence of step 2 is exhausted,
   the planner refuses with the full record. Mid-run fallbacks (§9.6, decision 54) follow
   the same steps at a canonical restart point, from the last committed state.

### 9.5 Output

`Plan` is a frozen object: tier, enabled and disabled rewrites with reasons,
bucket and skin settings, kernel availability, the fallback candidates and
predicted memory. `str(plan)` is human-readable, and the plan is written into
provenance (§13). Users can override the preferences of §9.3 (rules 8–12),
never the hard constraints (rules 1–7) except through their `allow_*` flags;
every override is self-checked and gated. A replaced decision reads `choice (override; rule n would give X)`, with
the code `override`, and `Plan.requested` keeps the user's value.

A plan never changes in place. A runtime switch (§9.6) or a fallback creates
a new frozen **revision** with its step, reason and predecessor. Provenance
keeps the whole chain, and the restart file stores the active revision.

### 9.6 Runtime adaptation

During a run the planner monitors:
- re-capture count;
- neighbour rebuild frequency and build time, per list (MACE, D3);
- bucket overflow;
- non-finite observables.

It may switch tiers at a canonical restart point (never mid-step; forced earlier than the run's restart
cadence if needed), as a new plan revision (§9.5) with a logged event (decisions 52–53).
Non-finite energies stop the run with a diagnostic; they are never "fixed".

---

## 10. Integrators and protocols

### 10.1 Contract

Every integrator:
- is written against `SimulationState` and is graph-capturable (no host syncs
  inside a step);
- draws random numbers from a seeded on-device stream, or batches host
  uploads (`NOISE_BATCH`);
- exposes observables (T, E_pot, E_kin, P, V, conserved quantity);
- keeps its targets (temperature, pressure) as device scalars that schedule
  updates fill before a step without quiescing (§8.5);
- serialises its own state for restarts (§10.6);
- registers its mass- and species-derived coefficients (§8.5);
- declares its constraint support (§4.4);
- has an ASE-equivalence test (§12.2).

### 10.2 List

| Integrator | Ensemble | Source / reference | 1.0 |
| --- | --- | --- | --- |
| Velocity Verlet | NVE | standard | yes |
| Langevin (ASE `Langevin` scheme, same noise stream as ASE) | NVT | exists (`GPULangevin`) | yes |
| BAOAB Langevin / `LangevinBAOAB` with barostat | NVT/NPT | exists (`gpu_npt.py`) | yes |
| CSVR (Bussi–Donadio–Parrinello) | NVT | Bussi et al. 2007 | yes |
| Nosé–Hoover chains | NVT | Martyna–Klein–Tuckerman 1992 | yes |
| MTK barostat (with NHC) | NPT | Martyna–Tobias–Klein 1994 | yes |
| Berendsen (thermostat/barostat) | equilibration only | Berendsen 1984 | yes, flagged non-canonical |
| Stochastic cell rescaling (Bussi barostat) | NPT | Bernetti–Bussi 2020 | 1.x |
| Replica exchange (temperature) | NVT | one structure per GPU process; sequential production rule respected | 1.x |

Barostat parameters (τ, mass factor) carry the current warning: the NPT
settings and equilibrium volumes are unqualified. The molecular chlorides need
D3 (§11.2).

### 10.3 Relaxations

FIRE and LBFGS on the GPU, fixed and variable cell, with ASE-compatible
convergence (`fmax`). Today's `quench.py` uses ASE's LBFGS (user decision:
LBFGS found the same minima as BFGS about 25× faster). GPU versions must
reproduce those minima on the annealing benchmark.

### 10.4 Monte Carlo

**Target distribution.** Swaps and displacements sample the canonical
ensemble; transmutations sample the semi-grand canonical ensemble, in the full
phase space (positions, momenta, species) at the schedule's target
temperature of that step. During ramps the chain anneals; equilibrium
statements hold at constant temperature. Insertion and deletion follow in 1.x
(§8.6).

**Proposals.** All moves are symmetric, so no proposal ratio enters the
acceptance:
- *swap:* a pair type chosen uniformly from `pairs`, then atom i uniformly
  among atoms of the first species and atom j uniformly among atoms of the
  second. Counts are conserved, so the reverse move has the same probability.
  If either species is absent, the attempt is recorded as impossible and
  costs nothing.
- *transmutation:* an atom chosen uniformly among atoms whose species is in
  the transmutable set S, then a new species uniformly from S without its
  current one. Symmetric because S is the same for every atom.
- *displacement:* one atom chosen uniformly, displaced uniformly within a
  sphere of radius δ. δ stays fixed while sampling; adaptive δ is allowed
  only during equilibration and is recorded.

**Fixed-shape selection.** Inside a captured graph an atom of species s is
chosen without `torch.nonzero` (a host synchronisation and a dynamic shape):
every atom gets a uniform random key from the device stream, keys of atoms
of other species are set to +∞, and `argmin` picks the atom. The decision
reads the **minimum value**, never the index alone: a minimum of +∞ marks the
attempt impossible (the index of an all-+∞ vector is a valid-looking 0), and
the apply step is masked by that flag.

**Fixed positions versus displacements.** Swaps and transmutations change no
position and no cell: during their attempts the neighbour list needs no
compaction, no rebuild can be triggered, and no edge or k-vector capacity
can overflow. Displacements change distances across the cutoff: each trial
re-compacts the list and runs the rebuild test, and an overflow or required
rebuild is handled as in §8.5 step 4; a rejection restores the neighbour-list
state.

**Momenta of atoms whose mass changes.** Part of the move definition. The
engines store velocities v (p = m v); both forms are given so the square
root cannot be inverted:
- `exchange` (swaps only; default): the two atoms swap velocities along with
  their species, v_i' = v_j and v_j' = v_i, so p_i' = p_j. This is the same
  as exchanging the two atoms' positions, each keeping its own momentum.
  Kinetic energy and total momentum are unchanged.
- `rescale`: p' = p √(m_new / m_old), that is v' = v √(m_old / m_new). This
  keeps each atom's kinetic energy (as LAMMPS `fix atom/swap` does by
  default) but changes the total momentum when masses differ.
- `redraw`: after an accepted move, the velocities of the changed atoms are
  drawn from Maxwell–Boltzmann at T with the new masses (seeded stream).
- `redraw_all`: after an MC block with an accepted move, all velocities are
  drawn from Maxwell–Boltzmann at T and projected to zero total momentum,
  v_i ← v_i − (Σ_j m_j v_j) / M.

Defaults: `exchange` for swaps; for transmutations `rescale`, or
`redraw_all` when the integrator fixes the centre of mass.

**Centre of mass.** `fixcm` (§4.4) concerns the integrator's random terms,
as in ASE: the noise has no centre-of-mass component, while friction acts on
all velocities. An MC move must not change the total momentum behind its
back. `exchange` conserves it exactly. With `fixcm` on, the single-atom
`rescale` and `redraw` are refused, and `redraw_all` restores zero total
momentum exactly (it is a Gibbs step in the zero-momentum subspace).

The projection of the noise is mass-weighted by default, for MD and for MC
(decision 37): `IntegratorConfig.fixcm_projection="auto"` resolves to
`"mass_weighted"` under NVT `fixcm`. Every species then holds the target
temperature, and the reported kinetic temperature counts 3N − 3 degrees of
freedom. ASE's projection subtracts P/(N mᵢ) per atom and scales by
√(N/(N−1)), which is right only for equal masses. It stays available as the
explicit `"ase"`: for ASE noise-stream equivalence, for A/Bs against the
research engine (which uses it), and for the phase 3 no-regression A/B.
- `"ase"` keeps ASE's temperature over 3N degrees of freedom.
- A resumed run keeps the projection its record holds. A record from before
  the option exists reads as `"ase"`.

Keeping velocities unchanged with an energy criterion on the potential alone
is not offered. It changes the kinetic energy (masses (1, 4) and velocities
(2, 1) give E_kin 4 before and 8.5 after the mass swap, at fixed positions)
and breaks detailed balance.

**Acceptance.** For every momentum choice:
`P_acc = min(1, exp(−β (ΔU − Σ_s Δμ_s ΔN_s)))`, with ΔN_s = 0 for swaps and
displacements. **Convention:** Δμ_s are configurational (mass-independent)
chemical-potential differences relative to a reference species, so
equilibrium species populations at fixed positions do not depend on masses.
Why every choice gives this rule:
- `exchange` permutes (mass, momentum) pairs, with Jacobian 1 and unchanged
  kinetic energy.
- `rescale` keeps the kinetic energy; its Jacobian (m_new/m_old)^{3/2} per
  changed atom cancels the mass factor of the momentum partition function
  under this convention (for swaps the Jacobians multiply to 1 under any
  convention).
- `redraw` and `redraw_all` are a Metropolis step on the configurational
  marginal followed by a Gibbs step on the momenta.

Users with full chemical potentials convert with
`fm.mc.configurational_mu(...)`, which applies
Δμ_s^conf = Δμ_s^full + (3/2) k_B T ln(m_s / m_ref). This follows from
μ^full = μ^conf + k_B T ln Λ³ with the thermal wavelength Λ ∝ m^(−1/2). The
exact-enumeration test (§12.2) checks the convention.

**Charge and spin.** In models with total charge and spin inputs (POLAR-1,
OMOL-0), a transmutation at fixed total charge changes the electron count by
ΔZ, and an odd change makes the spin multiplicity invalid (Na ↔ Mg). Each
species in a transmutable set therefore carries a formal charge q_s. The
default requires an isovalent set (Δq = 0). A move updates the total charge
by Δq, which is a registered derived buffer (§8.5), and is allowed only if
ΔZ − Δq is even. The planner refuses sets that violate this (e.g. Na ↔ Mg
without formal charges, or O ↔ Cl with an even Δq). Whether a model handles
the changed charge state physically is a model question (§4.3), not an
engine one.

**Evaluation.** Each attempt is one event under the transaction contract of
§8.5 and costs one energy evaluation of the trial state, on the forward-only
energy graph under `torch.no_grad()` (§8.1). Attempts of one MC block run
back to back; forces and stress are recomputed once at the end of the block
if any move was accepted.

**Baseline energy.** ΔE is always the difference of two energies from the
**same MC evaluator** (graph and precision).
- At the start of every MC block, the current state is evaluated once with
  that evaluator, never taken from the MD energy. The MD energy comes from
  the force graph and, under the float64 MC fallback, from another
  precision; mixing the two would put their offset (up to about 0.1 meV/atom
  for MH-1 r²SCAN in float32, see energy precision below) straight into the
  acceptance.
- The result goes into a separate MC energy cache tagged with evaluator,
  precision and state version. An accepted trial's energy becomes the new
  cached baseline; a rejection keeps it.
- A test checks the first attempt after MD: its baseline equals a fresh
  evaluation with the MC evaluator, and its acceptance probability matches
  the reference ΔE (§12.2).

**Energy precision.** Energies are accumulated in float64 (§7.3). A float32
total cannot resolve MC energy differences: NbCl5 3×3×3 (1,944 atoms) totals
about −20,652 eV with MH-1 `matpes_r2scan` (float32 step 2⁻⁹ eV ≈ 2 meV,
visible in the FP32 relaxation log of
`results/anneal-production-rtx5060ti-20261010/NbCl5`) and about
−2.08 × 10⁷ eV with all-electron E0 (step 2 eV). 0.01 k_B T at 250 K is
0.2 meV. Accumulation alone is not enough: the float32 forward pass carries
per-atom errors of its own. Stock float32 MH-1 r²SCAN energies differ from
float64 by 0.05–0.13 meV/atom on NaNbO3 and NbCl5 (gates of 15 October
2026), most of it not from the summation, which accounts for about
0.01 meV/atom. Whether these errors cancel in ΔE to below 0.01 k_B T has not
been measured. The move-energy gate (§12.3) decides:
`mc_energy_precision="auto"` uses the force precision if the gate passes and
a float64 forward otherwise, while MD forces keep their precision. Float64
runs at a small fraction of the float32 rate on consumer and workstation
GPUs, so phase 3 measures the cost of both (§16).

**Graph capture.** A swap or transmutation attempt (propose → apply → refresh
derived buffers → energy → accept or restore) is one capturable graph with
device-side random numbers and a device-side accept mask, so swaps at high
temperature keep tier-1 speed. A displacement attempt is captured only while
the rebuild test passes: the GPU neighbour build synchronises the host for
the pair count and may reallocate (`build_gpu`, `pair_search.candidate_pairs`),
so a trial that needs a rebuild (§8.5 step 4) leaves the graph, builds
eagerly in trial buffers and then runs the captured energy graph. With δ
near skin/2 (0.15 Å at the default skin) plus the MD drift since the last
build, most trials would take that path, so rule 11 (§9.3) raises the skin
for displacement workloads, and the benchmark reports displacement-attempt
cost with and without a trial rebuild, separately from swap cost (§16,
phase 3).

**Schedule and counters.** n MD steps, then m attempts. Attempted, impossible
and accepted counts per move type are device-side counters, incremented
inside the captured graph and read by the host only with the per-block flags
(§8.1), never per attempt. They are observables and part of the restart
state (§10.6).

**Accuracy prerequisite.** MC is enabled for a plan only after the energy and
move-energy gate (§12.3) passes on the user's structure, including moves to
species that are in the superset but absent at setup.

**Cost.** Each attempt costs one forward evaluation, and each block one more
for the baseline; benchmarks report the evaluation cost as a fraction of an
MD step (§16). With 50 attempts per 100 MD steps (51 evaluations), keeping
90% of MD throughput would need each evaluation to cost at most about 22% of
an MD step, so MC throughput is reported separately, not folded into the
event-overhead target. A local-energy variant (recompute only atoms within
n_layers × r_cut of the change) is a 1.x optimisation and must be exact.

### 10.5 Protocols

Port today's profiles:
- `anneal`: LBFGS fixed-cell fmax 0.01 pre-relaxation; 0 → 1000 K in 20 ps;
  3 K/ps to 250 K; every 20th step saved.
- `quench`: frames per 100 K bin from 500 K, quenched with LBFGS.
- Heating/cooling ramps, holds, and the `fast` and `ultrafast` profiles.

Add melt–quench and a swap-MC annealing protocol. Protocols are data
(TOML/YAML or Python objects), not code paths in a CLI; YAML is read with
`safe_load` only.

### 10.6 Simulation state and restart

The restart file holds the complete state. A resumed run continues exactly
as an uninterrupted run **with the same restart cadence** would: bit for bit
where the plan is reproducible (§9.3), statistically otherwise. Restart
writes are canonicalisation points (below), so a run with a different
cadence is a different, equally valid trajectory; this is the only
definition of continuation the library promises.
- **Physical state:** positions, momenta, cell; current per-atom species
  (atomic numbers after accepted exchanges) and masses; per-atom charges where
  the model takes them, total charge and spin; PBC; constraints with their
  parameters; barostat state (`p_eps`, cell force); energy, forces and stress
  of the committed step.
- **Integrator and protocol state:** integrator parameters and internal
  state, step, schedule and position in it, thermostat target, and the
  protocol recorded at start (today's `LEGACY_PROTOCOL` behaviour).
  Format 4 (phase 4, 4-T01) records the integrator scheme, its
  degree-of-freedom convention and its parameters in the integrator record,
  and the FixAtoms indices in the physical record. A resume with a different
  set of fixed atoms is refused. Each scheme's own state is its `state_dict`,
  whose keys must equal the `state_keys` of the scheme's registry record:
  - NHC: `eta` and `p_eta`;
  - MTK: the cell momenta and the barostat chain;
  - CSVR: the transferred energy (its noise rows and cursor are the random
    state below);
  - NVE and Berendsen: nothing.

  Files of versions 0–3 load unchanged, as Langevin or BAOAB runs without
  fixed atoms.
- **Random state:** every generator (device and host) of the integrator and
  the MC driver, *and* the random numbers already drawn ahead with their
  cursor. Today's restart stores `_noise` and `_noise_next` for exactly this
  reason (`workflow.py`, `state()`); the generator state alone would skip or
  repeat the prefetched batch.
- **MC state:** the attached move set (§7.6), position within the MD/MC
  cycle, counters per move type, adaptive step sizes.
- **Force-field identity:** checkpoint SHA-256, head, species superset, force
  precision, physics options (charge/spin inputs; D3 with all its parameters,
  §11.2; polar options such as the PR 1762 settings; the reference
  corrections of §12.1), MACE and FastMACE versions. A different MACE or
  FastMACE version alone does not refuse a resume: it warns and records
  both versions in `metadata.json` (owner decision 23).
- **Plan:** execution mode, backend, rewrites and MC energy precision; edge
  and capacity buckets.

**Canonical restart points.** The neighbour lists and the pipeline carry
more hidden state than reference positions:
- the build cell h₀ (in variable-cell runs the candidates are built in h₀,
  and the rebuild test measures strain against it);
- the build-time sender groups (species order of the segmented first sum,
  which an MC move since the last rebuild may have changed);
- the pipeline's padded-size floor and next edge bucket (`_pipe_floor`,
  `_pipe_edges`);
- and the same again for every further list, such as D3's.

Rather than serialise all of it, a restart write is a canonicalisation
point, in interrupted and uninterrupted runs alike. It quiesces the
pipeline, rebuilds every neighbour list from the current positions, cell
and species (deterministic: the GPU build sorts on a unique key), and resets
pipeline floors. The resumed run then starts from exactly the state the
uninterrupted run has at that step. Bitwise continuation is defined against
an uninterrupted run with the same restart cadence. The runtime-state
registry (§8.1) lists, for every persistent buffer, whether it is saved or
reset at this point, and a CI test checks that nothing is left over.

Restart files are written atomically (temporary file, then rename, as today),
every 10,000 steps by default and only at committed steps: never inside an MC
block or from an overflowed step. `Simulation.resume` refuses a different
force-field identity, except that a version change alone warns and is
recorded (decision 23). A different plan (another GPU or rewrite set) needs
`allow_plan_change=True` and is recorded: same physics, but no bitwise
continuation.

**Output files at restart points.** The engine state is complete, but the
run has also written files, and asynchronous output (§8.1) writes frames and
thermo rows after the last restart point; on a plain resume they would be
duplicated or forked. Today's resume drops them by rewriting
`temperature.csv` and `md.traj` to the restart step
(`workflow.truncate_outputs`), which rereads and rewrites the whole
trajectory, up to 770 MB on the NAS. The library instead makes every restart
point a chunk boundary of the trajectory and thermo files (§5,
`io/trajectory.py`): a restart write closes the current chunks and records
their indices in the restart file. Resume deletes the chunks after the
restart step and opens new ones; nothing is rewritten. If the outputs are
ahead of the restart and cannot be truncated (read-only, or chunks missing),
`Simulation.resume` refuses with `RestartMismatch`. MC counters are restart
state already. A test crashes a run between a restart write and the next
flush and checks the resumed run's files for duplicate or missing frames and
rows (§12.2).

Tests (§12.2): save → load → save gives identical files; interrupted and
uninterrupted runs crossing accepted and rejected swaps and a partly consumed
noise batch agree bit for bit on CPU float64 and in reproducible mode, and
statistically for hybrid plans.

---

## 11. Physics extensions

### 11.1 Long range (PolarMACE)

Port `polar_gpu.py`:
- fixed-size k-vector list with an exact inclusion mask and headroom for
  strained cells (overflow flag → re-plan, checked before commit, §8.1);
- per-call `_Call` state; the shared `_Holder` is replaced by per-instance
  state or an exclusive lease (§8.7), because today every copy of a model
  shares one structure's k-vectors;
- the stress-geometry fix (MACE issue 1642; fixed upstream in MACE PR 1718,
  after 0.3.16). It is a reference correction (§12.1): for MACE versions
  without PR 1718 the gate's reference calculators take the corrected path
  too, as `tools/validation/npt_stress_reference.py` does today, and no
  fallback restores the uncorrected stress;
- the uvu GEMM (PR 1711).

PR 1762 (Fukui regularisation, dipole cap) stays opt-in. Charge and spin
multiplicity are per-structure inputs with validation.

### 11.2 D3 dispersion on the GPU

The molecular chlorides need D3 for NPT: without it, NbCl5 swells 20% at
300 K and evaporates at 600 K. Today D3 exists only through `torch-dftd` on
the ASE path. Implement D3 inside the force field:
- **Own candidate list with its own skin, no per-step compaction.** A
  second skin list with the D3 pair cutoff. It has its **own skin and its
  own displacement and strain test** against its own reference positions,
  not the MACE list's. With the MACE skin (0.3 Å, test `max > skin/2`) the
  production NbCl5 anneal rebuilt its list 27,697 times in 540,000 steps,
  once per 19.5 steps averaged over the ramp and more often near 1000 K
  (`results/anneal-production-rtx5060ti-20261010/NbCl5/metadata.json`,
  `engine_info/neighbour_rebuilds`). A D3 build enumerates about 40 times
  the MACE list's candidates through the host-synchronising build path
  (`build_gpu`: sort on the unique key, pair count read by the host, buffer
  regrowth), so at that cadence the builds alone would exceed the cost
  target below. A 2 Å skin raises the pair count by only
  ((21.2 + 2)/(21.2 + 0.3))³ ≈ 1.25 and cuts the rebuild rate several-fold;
  rule 11 (§9.3) chooses the D3 skin from `max_temperature_K`, and the plan
  and provenance record it. The list is **stored as a half list**
  (undirected pairs, i < j, plus self-image pairs once). At 40 bohr plus the
  0.3 Å MACE skin, NbCl5 3×3×3 (28.8 Å³/atom; 4/3 π (21.5 Å)³ / 28.8 Å³
  ≈ 1,440 neighbours per atom) has about 2.8 million directed, i.e.
  1.4 million stored, candidate pairs; with a 2 Å skin about 1,800
  neighbours, 3.5 million directed and 1.75 million stored. Either is some
  40 times the MACE list. Compacting that every step (prefix sum and copy)
  would cost more than D3 itself. D3 needs only scalar functions of r_ij, so
  the kernel iterates over the stored pairs directly with an in-register
  r² ≤ r_cut² mask and adds each pair's contribution to both atoms.
- **Periodic images.** Like the MACE list today, the D3 list enumerates
  every periodic image within the cutoff (the image range follows from the
  cutoff and the cell widths, `build_gpu`); it makes no minimum-image
  assumption. A cell narrower than twice the 21.2 Å cutoff, such as the
  three short cell heights of NbCl5 3×3×3 or any small user cell, is
  therefore valid, and no cell-size refusal is needed. Small cells only
  raise the pair count, which the memory estimate (§9.3 rule 12) includes.
- **Cost target.** At most 10% of the MD step time on NbCl5 3×3×3,
  **amortised over a run with its builds included**, measured in phase 3b
  against the same run without D3 on the production ramp (§16). Three
  bandwidth-bound passes over 1.4–1.75 million pairs move on the order of
  150–200 MB per step; the hybrid NbCl5 step takes about 29 ms on the
  RTX 5060 Ti (§3.3), so the per-step budget is about 2.9 ms. The target is
  provisional until that measurement. D3 build time and rebuild count are
  runtime counters (§9.6, §13).
- **Three passes, analytic derivatives.** The C6 coefficients depend on the
  coordination numbers, so the forces need dE/dCN_i of every atom complete
  before they can be propagated through CN. Pass 1: coordination numbers
  CN_i (cutoff `cnthr`). Pass 2: energy, and the direct pair forces and
  virial at fixed CN, while accumulating dE/dCN_i over all pairs of atom i.
  Pass 3, after a kernel boundary (the device-wide synchronisation that
  completes dE/dCN_i): add (dE/dCN_i) ∂CN_i/∂r_ij to the forces and virial
  of both atoms of each pair. `torch-dftd` gets the same term from autograd
  after the energy. Omitting it keeps the energy and changes the forces;
  the review that found this measured about 0.0013 eV/Å in a CPU
  `torch-dftd` probe (not re-measured here). The validation against
  `torch-dftd` compares forces and virial, so it catches the omission. No
  autograd runs over millions of pairs.
- **Graph-capturable.** Masks with `where`, never `torch.nonzero`.
  `torch-dftd` 0.5.1 cannot be captured for exactly this reason: its
  coordination number selects pairs with `torch.nonzero(r <= cutoff)`.
- **Precision.** Pair terms in the force precision; CN, energy and virial
  reductions in float64 (§7.3); the result must pass the gate (§12.3).
- **Unit constants.** Use the reference implementation's constants so that
  validation compares implementations, not constants: `torch-dftd`
  hard-codes `d3_autoang = 0.52917726` and `d3_autoev = 27.21138505`, which
  differ from ASE's `Bohr` and `Hartree` by about 1e-7 relative. The
  constants are recorded with the parameters. (`mace_mp` passes
  `cutoff = 40 * ase.units.Bohr` in Å. `torch-dftd` selects its pairs in Å,
  so they stop at 39.9999963 bohr in its own units, while its masks convert
  the cutoffs with ASE's `Bohr`, exactly 40. FastMACE's reference wrappers
  pass 40 × `d3_autoang` Å, so pairs and coordination numbers both stop at
  exactly 40 bohr (3b-T01). simple-dftd3 1.6.0 agrees with `torch-dftd` on
  the same numbers to 2.2e-8 relative in energy, 4.2e-10 eV/Å in forces
  (r2scan; 2.6e-9 with pbe) and 1.5e-10 eV/Å³ in stress; tad-dftd3 0.7.0 has no periodic boundaries and
  is not a reference.)
- **Stress** included; the D3 list's build state follows the canonical
  restart points (§10.6).

**Parameters.** "D3(BJ)" alone does not identify the correction. The force
field takes and records every choice, and they are part of the force-field
identity (§10.6):

| Parameter | 15 October 2026 study (`studies/npt/probes/d3_check.py`) |
| --- | --- |
| damping | Becke–Johnson (`bj`) |
| functional parameter set | `r2scan` (matching the MH-1 `matpes_r2scan` head; `mace_mp`'s own default is `pbe`) |
| pair cutoff | 40 bohr ≈ 21.2 Å (`mace_mp` default) |
| coordination-number cutoff | 40 bohr (`torch-dftd` 0.5.1 default `cnthr`) |
| three-body (ATM) term | off (`torch-dftd` default `abc=False`) |
| cutoff smoothing | none (`torch-dftd` default) |
| precision | float64 (`mace_mp` passes the calculator's dtype; `TorchDFTD3Calculator`'s own default is float32) |
| unit constants | `torch-dftd`'s `d3_autoang`, `d3_autoev` |

**Correctness criterion.** Implementation correctness and physical stability
are separate questions:
- *Implementation:* energies, forces and stress against `torch-dftd` 0.5.1
  and the reference `dftd3` program with identical parameters, on rattled
  and strained cells; then the GPU engine's NPT trajectories with D3 against
  an independent implementation of the same potential (ASE `LangevinBAOAB`
  with MACE plus `torch-dftd`) under the same conditions: stated pressure,
  temperature, cell, timestep, barostat settings, duration and at least three
  seeds per engine, compared on mean volume, volume fluctuation, mean
  pressure and potential energy within their combined statistical errors.
- *Physics:* whether NbCl5 stays solid at 600 K with MH-1 + D3 is a model
  qualification, not an acceptance test of the library. The 15 October study
  is one seed, 5 ps, 216 atoms, float64 ASE, and leaves the long-run
  behaviour open (`studies/npt/README.md`). A physical failure outside the
  model's training data is an accepted outcome (user decision, 16 October
  2026; §18 decision 6).

### 11.3 Constraints

`FixAtoms`, fixed centre of mass, and (1.x) bond and angle constraints
(RATTLE) for molecular species. Support per integrator and optimiser is
declared in §4.4; anything else is refused at plan time.

---

## 12. Validation and testing

### 12.1 Reference policy (versioned)

"Correct" is defined by a versioned reference record per MACE version and
model family. It has four kinds of entries:
- **Stock equivalence** (default): the reference is MACE's `MACECalculator`
  of that version, cuEquivariance float64 (float32 for the ratio rule).
- **Upstream bug fixes:** corrections of defects that upstream fixed after
  the pinned version. Today one: the PolarMACE stress (MACE issue 1642, fixed
  in PR 1718 after 0.3.16). Each fix is validated independently of FastMACE
  (here: finite differences of the energy,
  `studies/other_models/probes/polar_stress_probe.py`), is applied to the
  reference calculators too (as `tools/validation/npt_stress_reference.py`
  does), and is retired once the pinned MACE contains it; the record then
  states plain stock equivalence.
- **Optional physics changes:** opt-in changes of the model (PR 1762 Fukui
  regularisation and dipole cap; D3). The reference is the stock calculator
  with the same change applied (for D3: MACE plus `torch-dftd` with the
  parameters of §11.2), and results are labelled non-stock.
- **Integrator corrections:** deliberate deviations of a production
  integrator from the implementation it was ported from. Today one:
  full-cell BAOAB. Its drag on the atom velocities is
  (p_ε + Tr(p_ε)/N_dof · I)/W (Quigley and Probert 2004, Eq. 5b);
  `gpu_npt.py` multiplies the trace term by the identity, whereas ASE 3.27
  adds it to all nine elements (an O(1/N_dof) difference of the drag). The
  existing trajectory tests set `ASE_TRACE_DRAG_ALL_ELEMENTS=True` to match
  ASE, so they qualify the ASE variant, not the production one.
  Qualification therefore has two parts. The ASE variant must reproduce ASE
  exactly, which tests the port. The corrected production path is compared
  with an independent reference that implements the same correction: a
  float64 CPU implementation written from the paper, or ASE with the
  correction patched in. Both are recorded in the reference record, and
  fallbacks keep the correction.

**Energy accumulation in the references.** The float32 yardstick of the
ratio rule accumulates its energy in float64, as production does (§7.3).
Stock float32 totals are off by up to 102 meV/atom with all-electron E0
(§4.3; MACE-OFF up to 11.4 meV/atom on paracetamol 3×3×3); against that yardstick a production energy could be off by 200
meV/atom and still pass. With float64 accumulation on both sides, the rule
measures the precision of the forward pass, not of the E0 sum. This is a
numerical treatment of the reference, recorded like a correction; it does
not change the model.

The record's identifier is part of the force-field identity (§10.6) and of
provenance (§13). Fallbacks (§9.4) keep the record: they never drop a
correction or an option, so a correct implementation cannot fail
qualification against a defective reference, and a fallback cannot restore
defective behaviour.

### 12.2 Test classes

| Class | Runs on | Content |
| --- | --- | --- |
| CPU unit | CI (CPU) | rewrite algebra on tiny models (`tiny_mace_calculator`), block linear/grouped/merged, k-vectors, PR 1762 port, neighbour list vs ASE/matscipy, integrator algebra, planner rules and precedence, `run` refusing undeclared moves and blocks below `min_mc_temperature_K` (§7.6), restart round-trip. Today 329 tests (collected); 8 workers locally (WSL RAM limit), `-n 24` on workstations |
| GPU self-check | CI (GPU) + every user run | per-rewrite and per-model checks (§6.4, §12.3) |
| Energy, force and stress gate | CI (GPU), `fastmace check` | today's rule: production path's max error vs float64 ≤ 3 × the float32 calculator's error (2 × until 7 November 2026, owner: "Change the rule to 3x"; `planner.selfcheck.RATIO`), now for energy per atom (float32 yardstick with float64 accumulation, §12.1) as well as forces and stress; rattled and strained cells, hydrostatic and full; all-electron and pseudopotential E0 models; move-energy differences for MC plans (§12.3) |
| Replay and transitions | CI (GPU) | the production path as executed: captured replay, pipelined and speculative steps; across a neighbour rebuild, a capacity overflow with re-capture, a cell deformation past the rebuild threshold, an accepted and a rejected species change, and a restart, each compared with a fresh direct evaluation of the same state |
| Mutation events | CI (CPU, GPU) | derived-state registry against fresh construction (§8.5); first MD step after accept and after reject against fresh construction; a superset species with no atoms (its table row from the identity basis, not from atom 0), the last superset species absent (sender-group count), an atom outside the superset (explicit error); a rejected displacement leaves the neighbour list of the committed positions; a displacement beyond skin/2 (> 0.15 Å at the default skin) is evaluated on a list built eagerly around the trial geometry, accepted and rejected, without looping, and its ΔE equals a direct evaluation; an overflowing displacement is redone with the same random numbers; the first NPT step after an accepted move equals a fresh construction (`force_eps` refreshed), hydrostatic and full cell; a ramp step changes the target without an event (no quiesce, §8.5) |
| Runtime state | CI (GPU) | runtime-state registry against the attributes each stage mutates; full registered state identical before and after every capture warm-up, sticky flags included (§8.1) |
| MC exact enumeration | CI (CPU tiny model; GPU) | a small cell at fixed positions (e.g. 8 sites, 4 + 4 atoms: 70 configurations; transmutation: 256), all configurations evaluated with the reference: per-move acceptance probabilities against the reference ΔE (including the first attempt after MD, whose baseline comes from the MC evaluator), and sampled populations against Boltzmann weights (χ² test, fixed seed). Hybrid MD/MC: kinetic temperature with MC on and off agrees for `exchange`, `rescale`, `redraw` and (with `fixcm`) `redraw_all`, and the total momentum stays zero under `fixcm`; a negative control with unchanged velocities must fail. Charge/spin: a set violating the parity rule is refused (§10.4) |
| Restart | CI (CPU, GPU) | §10.6: round trip; interrupted vs uninterrupted (same restart cadence) across accepted and rejected swaps, a partly consumed noise batch, a variable-cell run with h ≠ h₀ at the restart step, a count-overflow redo before it, and a D3 list; a crash between a restart write and the next output flush leaves the resumed run's trajectory and thermo files without duplicate or missing frames and rows (chunks deleted, nothing rewritten) |
| Reproducibility | CI (GPU) | in reproducible mode, repeated runs and restarts bit for bit (§9.3) |
| Isolation | CI (GPU) | A → B → A with two crystals for the plain, MH-1 and polar paths (§8.7) |
| Fallback | CI (GPU) | deliberately broken rewrite and broken capture: eager diagnosis, bisection, pristine rebuild, validated activation (§9.4) |
| ASE equivalence | CI (GPU) | Langevin: same seed reproduces ASE's noise stream (NaCl 2 ps, 0 → 400 → 0 K: same T and E_pot in every 50 K bin, \|ΔT\| < 1e-6 K, \|ΔPE\| < 1e-7 meV/atom; owner, 20 October 2026); BAOAB: the ASE variant (`ASE_TRACE_DRAG_ALL_ELEMENTS=True`) vs ASE `LangevinBAOAB` exactly, and the corrected production variant vs an independent corrected reference (§12.1); new integrators vs reference implementations |
| Statistical | nightly | long MD: distributions of T, E, P, V vs float64 stock ASE; hybrid runs are compared statistically, not bitwise |
| Performance | nightly | step-time suite (§12.5); regressions > 3% flagged (cross-process noise is about 3%; compare small effects with alternating A/B in one process) |

### 12.3 Gate on the user's structure

The gate is **opt-in** (owner decision 3, 19 October 2026). It is off by
default: the status is `skipped`, recorded and named in the run report.
With `check="gate"`, `Simulation`, `fm.md`, `fastmace.ase` and
`fastmace md --check` run a reduced gate before the first step:
- 2 rattled copies (0.05 Å) and, for NPT, 2 strained copies (±2%);
- production path, **in the mode it will run** (captured replay in graph
  mode), against the reference of §12.1;
- energy per atom, forces and stress: the ratio rule of §12.2 (energy
  against the float32 yardstick with float64 accumulation, §12.1). Each
  quantity has an absolute floor so that the rule stays stable where the
  float32 yardstick happens to be nearly exact: 0.01 meV/atom for energy,
  1 × 10⁻⁵ eV/Å for the maximum force component and 1 × 10⁻⁷ eV/Å³
  (≈ 16 kPa) for the maximum stress component (all accepted, decision 7),
  the latter two well below the float32 errors recorded so far;
- for MC plans, move-energy differences ΔE for sample moves of every enabled
  type (each swap pair, transmutation to every superset species including
  those absent at setup, displacements):
  `|ΔE_prod − ΔE_ref| ≤ 0.01 k_B T_min`, with T_min =
  `Workload.min_mc_temperature_K` (§9.2), declared at construction because
  the gate runs before any schedule is known; `run` refuses an MC block
  whose target temperature is below it (§7.6). Acceptance probabilities are
  then off by at most about 1%. There is deliberately no ratio term: what matters is
  the error in acceptance, not parity with float32. Stock float32 totals are
  spaced about 2 meV apart for NbCl5 3×3×3 (§10.4), and even with float64
  accumulation, twice the float32 forward error bears no relation to
  k_B T. A species-dependent constant energy error, invisible to forces and
  stress, fails here. If only this part fails, the MC energy path moves to
  float64 (§9.4) and the check is repeated.

It takes seconds to a minute. A failing gate triggers the fallback algorithm
(§9.4).

**Status.** Every gate result has one of five states, all recorded:
- `passed`;
- `passed-reduced`: the reference ran on a smaller supercell of the same
  structure, labelled as such. A smaller cell has different periodic
  images, edge counts, capacities and k-vectors. It stands in only for the
  **numerical** comparison with the reference (rewrite algebra, species
  tables, per-edge and per-atom maths, precision). It does not stand in for
  anything that depends on the production cell. The replay and transition
  checks (captured against direct evaluation, rebuilds, overflow and
  re-capture, events) compare the production path with itself, need no
  reference, and therefore always run on the production cell;
- `failed`: triggers §9.4;
- `unavailable`: the reference could not run on any allowed device;
- `skipped`: the default when the gate is not requested (decision 3), or
  explicitly with `check="skip"`.

**Resources.** An FP32 production workload can fit where its unoptimised
float64 reference does not (today: NbCl5 3×3×3 PolarMACE in float64 exceeds
16 GB on the original path, `polar_stress_probe.py`). The gate therefore
evaluates the references first, one configuration and one calculator at a
time, frees them, and only then builds the production engine. On GPU memory
exhaustion it tries CPU float64 if the estimated time is within a
configurable limit, then a reduced cell. If all fail, the status is
`unavailable`: a requested gate refuses the run unless the user passes
`check="allow_unavailable"` (which also requests the gate), and that is
recorded. A more memory-hungry
reference is never a fallback for a plan that failed for memory.

On a CUDA plan, each GPU reference step runs in a child process. A GPU
out-of-memory error can leave a process's CUDA context unusable (sticky
errors, seen on WSL2). A CUDA error, non-zero exit or kill of that child
counts as a device failure for the ladder, and its text is recorded. A CPU
reference child runs under a hard memory cap (`RLIMIT_DATA`, at most the
available memory less a reserve), so an underestimate fails in the child
instead of exhausting the host. Under `passed-reduced`, the replay check on
the production cell is bounded by the gate's own limit per quantity (plan
task 2-T05).

### 12.4 CI matrix

- **CPU:** Linux, Python 3.10–3.12, two torch versions.
- **GPU:** at least one Ada (sm_89) and one Blackwell (sm_120) runner, with
  the hybrid and plain backends.
- Model checkpoints are cached in CI and verified by SHA-256.

### 12.5 Benchmark suite

`fastmace bench` reproduces today's tables:
- models: MH-1 (three heads), MP-0 medium/large, MP-0b3, MPA-0, OMAT-0,
  MatPES-PBE/r²SCAN, POLAR-1-S/M/L, OMOL-0 XL, MACE-OFF23 S/M/L, OFF23b M,
  OFF24 M;
- crystals: NaCl, NaNbO3, NbCl5 (3×3×3), plus the ten benchmark crystals;
  for organic models (MACE-OFF) the organic set of `Structures/Organic`
  (urea, naphthalene and paracetamol 3×3×3 from COD, a 648-atom water box);
- configurations: ASE plus calculator, plain GPU engine, default NVT, default
  NPT;
- timing: median of 4 × 300 steps after 300 warm-up steps.

Every result reports steady-state step time, setup time (load, rewrites,
compilation, capture, gate) and peak GPU memory. MC benchmarks also report the
cost of one energy evaluation as a fraction of an MD step. Comparisons with
the research engine use a frozen revision of it (git commit recorded) on the
same machine and configuration, in alternating A/B blocks. Results go to
versioned JSON. The breakdown mode reproduces `step_breakdown_probe` (stages,
kernel categories, operators).

---

## 13. Provenance

Every run writes `metadata.json` with (schema `fastmace.metadata/1`; additions within a major version raise
`schema_minor`, and readers accept every older minor):
- model path, SHA-256, head, licence;
- the force-field identity (§10.6) and the reference record (§12.1);
- the plan (tier, rewrites on/off with reasons), force and integration
  precision, TF32 flag, backend (for the plain backend: whether the
  convolution is fused), fallbacks tried;
- the file-lock mode (soft or kernel) and whether the CUDA driver cache was
  disabled (§14);
- versions of all packages and kernels (OEq extension build id, Triton
  version), GPU name, driver, compute capability;
- seeds, generator-state hashes and stream positions (cursors), integrator
  parameters, schedule. Full generator states and prefetched random batches
  live only in the restart file (§10.6), not in `metadata.json`;
- gate status (§12.3) and self-check results;
- re-capture counts; rebuild counts and build times per neighbour list
  (MACE, D3) with their skins; overflows redone; tier switches; MC counters;
  restart points with their output chunk indices (§10.6); timing summary;
- code version (package version + git hash when installed from source).

This is today's provenance block, made complete and schema-versioned.

---

## 14. Packaging and distribution

- **Name:** `fastmace`. Check PyPI and GitHub for clashes before the first
  release; the import name and CLI follow.
- **Licence:** MIT (compatible with MACE; owner decision 4, 17 October
  2026). Vendored OEq kernels keep their
  BSD-3 licence and notice. The native D3 path's reference tables
  (`dftd3_params.npz`, byte-identical to `torch-dftd` 0.5.1's, whose sha256
  the identity records) keep `torch-dftd`'s MIT licence and notice; whether
  the release may ship them is checked in phase 6. The cuEquivariance ops wheels are NVIDIA-licensed:
  depend on them, do not vendor them. Model weights are never redistributed;
  `fm.load` downloads from official sources and shows the model licence.
- **Extras:** `fastmace[oeq]`, `fastmace[polar]` (graph-longrange pinned per
  MACE version), `fastmace[d3]` (the `torch-dftd` reference and the ASE front end; the native D3 path does not need it), `fastmace[d3-reference]` (`torch-dftd` and simple-dftd3 for the D3 cross-check tests; not in `[full]`), `fastmace[dev]`,
  and `fastmace[full]`, the union of `[oeq]`, `[polar]` and `[d3]` (no
  packages of its own). The quick start uses `pip install "fastmace[full]"`
  then `fastmace doctor`. Until the prebuilt OEq wheels exist (phase 6),
  `[oeq]` needs the host toolchain below, and `doctor` says what is missing;
  the base install still runs, on the plain backend, and the report says so
  (§7.7).
- **No `nvcc` at install or run time, and no compiler on the default path:**
  - The streaming kernels already compile through NVRTC.
  - "No `nvcc`" is not "no compiler". OEq's host extension today needs a
    host toolchain (`studies/hybrid_oeq/scripts/oeq_no_nvcc.sh`): `g++`,
    ninja (`OEQ_JIT_EXTENSION=1`, `MAX_JOBS`), the CUDA runtime headers
    (`cuda.h`, `nvrtc.h`, CCCL) from the pip runtime wheels, a sentinel in
    place of `nvcc`, a per-process extension directory, `OEQ_EXTENSION_SEED`
    and the lock-file workaround for OEq 0.7/torch 2.14.
  - 1.0 ships prebuilt OEq host-extension wheels per torch/CUDA build, so
    `fastmace[oeq]` needs no compiler. The host-toolchain build stays as a
    documented fallback for other builds (§4.1); `fastmace info` reports
    which of the two is in use and what the fallback is missing. Upstreaming
    the no-`nvcc` build to OEq (§16, phase 6) reduces what the wheels must
    carry.
  - Whichever path is active, `main.py`'s refusal to run without the wrapper
    must disappear.
- **Caches:**
  - Triton and Inductor caches, the OEq extension cache and compile caches go
    under a configurable cache root (`FASTMACE_CACHE`), never inside
    environments. The planner sets Inductor's and Triton's directories before
    a plan's first compilation; a `TORCHINDUCTOR_CACHE_DIR` or
    `TRITON_CACHE_DIR` already set in the environment wins, except Inductor's
    own default, which importing `torch._dynamo` writes there.
  - Respect read-only and NFS homes; pytest's ownership check on the NAS is a
    known problem.
  - **No kernel file locks on network filesystems.** NFS locking on the
    workstations' share (NFS 4.0) broke on coraline (10 October 2026) and
    leto (16 October 2026): the NAS refuses that client's lock requests until
    the mount is reset. Measured sources per run: the CUDA driver's JIT cache
    (POSIX locks on its `index`, 220–440 per run) and PyTorch's `filelock`
    (one, the OEq extension build, fatal on a broken client). FastMACE must
    therefore (a) make `filelock.FileLock` a `SoftFileLock` before torch is
    imported, at `import fastmace`, (b) set `CUDA_CACHE_DISABLE=1` when
    the CUDA driver cache directory lies on NFS (`CUDA_CACHE_PATH`, or the
    driver's default `~/.nv/ComputeCache` when that is unset; NFS detected
    from `/proc/self/mounts`) unless the user set it, and (c) record the
    lock mode in the provenance (§13); an opt-out restores kernel locks. The
    research engine does this since 16–17 October 2026 (zero NFS lock
    requests per run, no start-up cost;
    `Experiments/Langevin_Thermostat/docs/NON_LOCKING_DESIGN.md`).

---

## 15. Documentation

- FastMACE in five minutes: `pip install "fastmace[full]"`, `fastmace
  doctor`, `fastmace md`, `fm.md` (§7.7).
- Port your ASE script: `fm.calc` (compatibility speed), then the
  `fastmace.ase` integrators (full speed, ASE units unchanged, observer
  rules), then `fm.md`.
- Why is it not faster? Each reason the report gives (§7.7): not 3D
  periodic, constraints, model outside the registry, OEq missing, frequent
  callbacks, small system, fallback after a gate, memory.
- What changes physically: float32 with float64 energies, the gate, the
  reference corrections, D3, NPT, MC, the known model issues (§4.3).
- Compatibility table: models × PBC, constraints, NPT, MC, D3, PolarMACE,
  with the experimental labels.
- Example gallery: ASE Langevin port, `fm.md`, NPT chlorides with D3, swap
  MC in an anneal, phonopy and NEB with `fm.calc`, restart, PolarMACE with
  charge, MACE-OFF in a periodic box.
- Quick start for the expert API (`ForceField`, `Simulation`).
- Concepts: tiers, plan and fallback, rewrites, reference policy, gate
  statuses, species superset and events, buckets, complete state and restart.
- How-tos: melt–quench, swap MC at high T, NPT with D3, PolarMACE with charge,
  restart, benchmarking.
- Model notes: per-family support and known model issues (§4.3).
- Rewrite catalogue with exactness, gains and constraints, plus the
  rejected-screens log.
- Reproducibility: non-bitwise hybrid sums, statistical comparison, the scope
  of `reproducible=True`.
- Monte Carlo: move definitions, momentum handling, chemical-potential
  convention.
- API reference (generated).

---

## 16. Migration plan and milestones

Each phase ends with its acceptance criteria met on the RTX 5060 Ti, plus the
GPU compatibility check on every production GPU type (decision 39, amended
23 October 2026).

| Phase | Content | Acceptance | Estimate (1 developer) |
| --- | --- | --- | --- |
| 0. Skeleton | repo layout, packaging, CI (CPU), licence, port CPU tests | `pip install -e .`; CPU suite green | 1 week |
| 1a. Extraction | `model`, `rewrites` (registry with today's catalogue), `neighbors`, `state`, `engine.graph`, `integrators.langevin`/`baoab`, behind the planner interface with one fixed plan (today's `main.py` defaults); no globals, no NAS or experiment imports, no wrapper script | against a frozen research-engine revision on the same machine and configuration, in alternating A/B blocks: steady-state step time within 3%, setup time and peak memory reported; today's force/stress gate passes for all 19 labels of §12.5 (ten benchmark models plus POLAR-1 S/M/L, OMOL-0 XL and five MACE-OFF, the last on the organic set); `Experiments/Langevin_Thermostat/main.py` runs on top of FastMACE | 3–4 weeks |
| 1b. Contracts | rule-based minimal planner (hard constraints, mode and backend choice, `Plan` record with revisions; no runtime adaptation); `Simulation` taking a `ForceField`, the reduced `Workload` and the `run` checks (§7.6, §9.2); float64 energy accumulation (§7.3); runtime-state registry and complete capture restore (§8.1); complete state and restart with canonical restart points and output chunks at restart points (§10.6); reference policy (§12.1) with the BAOAB integrator correction qualified against an independent corrected reference; gate with energy, floors and captured replay (§12.3); exceptions and checkpoint trust (§7.6); adoption layer (§7.7): `fm.md` and `RunResult`, the run report and error-message shape, `fm.calc`, `fastmace.ase` `Langevin` and `LangevinBAOAB` with observer sync, CLI `md`, `init`, `doctor`, `models`, `fastmace[full]` | gate passes for all 19 labels of §12.5, energy included (POLAR-1 and OMOL-0 energies within the gate instead of 45–102 meV/atom off, MACE-OFF instead of 0.1–12 meV/atom); registry and capture-state CI checks pass; restart round trip, interrupted vs uninterrupted with the same cadence (bitwise on CPU float64, statistical for hybrid); a crash between a restart write and the next flush resumes without duplicate frames; the §7.7 acceptance tests for Langevin, BAOAB and the CLI | 4–6 weeks |
| 2. Safety | self-checks, gate statuses and resource policy (§12.3), fallback algorithm with ddmin, diagnosis budget and terminal sequence (§9.4), subprocess runner through a restart write (§8.7), versioned registry, instance isolation (§8.7), provenance schema, GPU CI; qualification of `reproducible=True` for one family (plain cuEq, MH-1) | unknown MACE version runs the reference path cleanly; a deliberately broken rewrite, a broken pair of interacting rewrites and a broken capture each end in the correct validated fallback within the budget, and an exhausted budget ends in the terminal sequence; A → B → A passes; GPU CI deferred by the owner (eleventh amendment of draft 0.12; workflow kept, run by hand); reproducible mode bitwise on repeated and restarted runs, or recorded as not qualifiable with the blocking kernel named | 3–4 weeks |
| 3. Dynamic topology and MC | species superset, derived-state registry, transactional events (§8.5; scheduled targets as device-scalar fills), overflow before commit, re-capture policy, adaptive skin and headroom (rule 11, displacement skin), compiled-tier path; energy-only graph; MC driver (swap, transmutation, displacement; MC-evaluator baseline per block; displacement trials rebuilt eagerly around the trial geometry; fixed-shape selection; `exchange`/`rescale`/`redraw`/`redraw_all`; charge/spin parity), move-energy gate with `min_mc_temperature_K` and MC energy precision, MC restart state | species-change events (no energy evaluation) every 100 steps at 1500 K keep ≥ 90% of fixed-species step speed; MC attempt cost reported separately as a fraction of an MD step, in float32 and float64, for swaps and for displacements with and without a trial rebuild; ΔE errors of float32 forwards measured on MH-1 and an all-electron model; first MD step after accept and reject equals fresh construction; exact-enumeration test passes; interrupted vs uninterrupted across accepted swaps; swap-MC annealing protocol runs end to end | 5–6 weeks |
| 3b. NPT for the chlorides | GPU D3 in the force field (§11.2): own candidate list with its own skin and without per-step compaction, three-pass kernel with analytic derivatives (including the CN chain term) and stress, own deformation-aware rebuild test; D3 in the BAOAB NPT path; constant-pressure melt–quench protocol | D3 against `torch-dftd` and `dftd3`; GPU NPT + D3 against the independent implementation (§11.2); amortised D3 cost with builds included measured against the 10% target on the production ramp; the NbCl5 NPT melt–quench protocol runs end to end. Its physical outcome is recorded, not judged (decision 6, §18) | 3–4 weeks |
| 4. Physics | NVE, CSVR, NHC, MTK, Berendsen; GPU FIRE/LBFGS, fixed and variable cell; `FixAtoms`; their `fastmace.ase` classes (§7.7) | ASE-equivalence tests; GPU LBFGS reproduces the annealing-benchmark minima; §7.7 acceptance for the new classes | 3–4 weeks |
| 5. Planner completion | remaining rules of §9.3 and their precedence, runtime adaptation (§9.6), optional calibration, `str(plan)` | each rule and the precedence order have a test | 1–2 weeks |
| 6. Release | docs, examples, benchmark tables, PyPI, prebuilt OEq host-extension wheels (§14), upstream PRs (MACE: linear rewrites, reassociation; OEq: no-`nvcc` host build) | 1.0 tagged; docs build; `pip install fastmace[oeq]` on a clean machine without a compiler | 2–3 weeks |

The total is about 25–34 weeks (6–8 months) to 1.0, up from 4–5 months in
draft 0.1; draft 0.11's adoption layer (§7.7) added 1–2 weeks to phase 1b. The planner, restart and reference contracts and the MC driver
came before the dynamic-topology milestone in draft 0.2. Draft 0.3 added
float64 energy accumulation, the runtime-state registry, the energy-only
graph, the MC energy-precision measurement and a dedicated D3 kernel.
Draft 0.4 split phase 1 and added the subprocess runner, the API contracts
and the reproducible-mode qualification. Draft 0.6 changed no estimate: its
items (API ownership, output chunks, the terminal fallback sequence, the D3
skin, the displacement path) are small inside the phases that already hold
them, and the prebuilt OEq wheels fall into phase 6. **These estimates are
floors** for one developer: they assume the research engine's code moves
with few surprises.

The first production deliverable is phases 0–3b (19–25 weeks, about 4.5–6
months): NVT and constant-pressure melt–quench and atom exchange on the
fastest tier, including the chlorides with D3 (decision 6). The barostat
settings still need their qualification study (an experiment, not library
work) before NPT defaults are recommended. Phase 0 (skeleton, packaging, CPU
CI) depends on none of the contract wording and can start at any time.

During migration the experiment keeps working: `workflow.py` switches to
FastMACE imports behind a compatibility layer, and results must match the
current engine statistically, or bit for bit where the backend is
deterministic.

---

## 17. Later (2.x candidates)

- Batched evaluation of many small structures for screening. Not used for
  the sequential production runs, and must not slow the single-structure path.
- Ghost-atom buckets for graph-speed GCMC (§8.6).
- Local-energy MC updates.
- Multi-GPU (domain decomposition) or a LAMMPS/ML-IAP export of the
  rewritten model.
- More MACE architectures and other equivariant MLIPs through the same
  registry.
- Enhanced sampling hooks (metadynamics/PLUMED bias forces) on the GPU.
- Property models evaluated along a trajectory, for example MACE-MDP dipoles
  and polarizabilities for IR and Raman spectra (not a force field; needs
  MACE develop's `DipolePolarizabilityMACE`).

---

## 18. Risks and open questions

| Risk / question | Mitigation |
| --- | --- |
| MACE internals change every release; hash-pinned rewrites break | versioned registry, self-checks, upstreaming; reference fallback |
| OEq/cuEq/torch version churn (lock files, `nvcc`, compile policy) | CI matrix, prebuilt extensions, process-constraint enforcement |
| Hybrid OEq sums are not bitwise reproducible | statistical tests; `reproducible=True` plan |
| Plain cuEq float64 is not bitwise reproducible either. cuEquivariance picks segmented-polynomial kernel variants at run time, which changes the summation order: the research engine differs from itself by up to 3.2e-13 Å over 500 MD steps (1a-T13, 18 October 2026) | equivalence tests compare at 1e-12 Å; `reproducible=True` (phase 2) must pin or refuse cuEq's run-time variant choice, or record the blocking kernel (§16 phase 2 acceptance) |
| Species superset increases memory or cost (columns for absent species) | measure; superset defaults to initial species |
| High-T workloads cause frequent re-captures | adaptive skin and headroom, re-capture budget, tier-2 fallback |
| Model-level failures look like engine bugs (POLAR-1-S NaNbO3, OMOL NaCl) | planner warnings with diagnoses; gate against the reference (§12.1) separates the two |
| Barostat settings and equilibrium volumes unqualified | keep warning; qualification study before recommending NPT defaults |
| Hybrid MD/MC samples the wrong distribution (mass changes, proposal asymmetry, chemical-potential convention) | moves defined with their momentum transformation and acceptance (§10.4); exact-enumeration test with a negative control (§12.2) |
| Energies of models with all-electron E0 references (POLAR-1, OMOL-0, MH-1 `omol`/`spice` heads; MACE-OFF H–Br, with a pseudopotential I; Nb, Ta and I are pseudopotential references, §4.3) wrong by up to 102 meV/atom in float32 (MACE-OFF up to 12) | float64 energy accumulation everywhere (§7.3) and in the float32 yardstick (§12.1) |
| float32 forward errors do not cancel in MC ΔE (unmeasured) | move-energy gate; float64 MC energy path as fallback, whose cost on GPUs with slow FP64 is measured in phase 3 (§10.4, §16) |
| Hidden engine state breaks capture, events or restart (stale build cell, sticky flags, frozen sender groups) | runtime-state and derived-state registries with CI checks; canonical restart points (§8.1, §8.5, §10.6) |
| A stale derived buffer after an event (masses, coefficients, species pairs) | derived-state registry checked against fresh construction (§8.5) |
| The float64 reference does not fit where the FP32 production path does | sequential reference evaluation, CPU and reduced-cell options, explicit `unavailable` status (§12.3) |
| A fault appears only inside CUDA graphs (as with Inductor's programmatic dependent launch) | gate and transition tests through captured replay (§12.2, §12.3) |
| Users take the ASE calculator for the fast path, or a ported ASE script behaves differently from ASE | §7.7: the calculator says it is the compatibility tier; the `fastmace.ase` classes keep ASE's signatures and units, sync observers through events and refuse what they cannot honour; the run report states the tier and its reason |
| Positioning vs existing tools (TorchSim batched MLIP MD; MACE's LAMMPS ML-IAP) | FastMACE's distinct value: exact model-level rewrites plus graph-speed single-structure MD with dynamic species; re-check competitors' feature sets before 1.0 and consider offering the force field as a backend to them |
| Licensing of dependencies and models | §14; licence display at load |
| Name clash on PyPI | check before phase 6 |
| D3 list rebuilds at high T cost more than D3 itself (27,697 MACE-list rebuilds in the 540,000-step NbCl5 anneal) | own skin and rebuild test for the D3 list; amortised cost target with builds included; build time counted (§11.2) |
| Output files ahead of the last restart point are duplicated or forked on resume | outputs chunked at restart points; resume deletes chunks, never rewrites (§10.6) |
| A client's NFS lock state breaks (coraline, leto) and stops runs that lock | no kernel file locks on network filesystems: soft locks, CUDA driver cache off on NFS (§14) |
| Displacement MC spends its time in eager trial rebuilds | skin from rule 11; cost reported with and without a rebuild (§10.4) |
| A fallback diagnosis that never ends, or ends nowhere | diagnosis budget and the terminal sequence: rewrites off, eager, reference, refuse (§9.4) |

Open decisions for the owner:
1. Repository — **decided (user, 17 October 2026): its own repository,
   `FZWINGEL-UBT/FastMACE` (private), checked out on an NVMe drive.** This
   replaces the decision of 16 October 2026 (inside ResearchModule as
   `FastMACE/`). Reason: the ResearchModule checkout lives on a USB hard
   disk, and four parallel worktrees of its 17,634 files did not finish
   checking out within the app's five-minute start limit, so the parallel
   worker sessions of phase 0 never started. The history of `FastMACE/` was
   carried over with `git subtree split`. Paths starting with
   `Experiments/`, `Structures/` or `environment/` in this specification
   are paths in ResearchModule, which keeps the research engine and the
   frozen baseline tag `research-engine-baseline-20261017`.
2. Minimum MACE version — **decided (user, 19 October 2026, phase 2 D1):
   0.3.16 is the minimum and the only qualified version.** Other versions
   run through the versioned registry (§6.3): rewrites whose pinned sources
   do not hash are off with their reason, and the always-on treatments
   follow the rule of plan task 2-T04, possibly the reference tier.
   Qualifying a newer release waits until the environments move to
   Python ≥ 3.11, which is a later decision.
3. Gate on the user's structure — **decided (user, 19 October 2026): opt-in**
   (`check="gate"`), off by default and then recorded as `skipped`. A
   requested gate that is `unavailable` refuses the run unless
   `check="allow_unavailable"` is passed (§12.3).
15–21. Decided (user, 19 October 2026, the phase 1b plan's D2–D8):
   - float64 energy accumulation everywhere, including the path `main.py`
     uses (15);
   - a species superset larger than the structure's species is refused until
     phase 3 (16);
   - `fastmace.ase` observers: position and momentum changes are applied as
     a host write at a canonical point, while species, mass and charge
     changes raise until phase 3's events. The §7.7 observer acceptance item
     covers positions and momenta in 1b; the rest moves to phase 3 (17);
   - `fastmace run` is part of 1b (18);
   - without the compiled MD tier (phase 3): inexact padding and
     `allow_reference` are refused; frequent host callbacks stay on the
     graph tier with a sync at each callback, reported (19);
   - `Simulation` without `workload=` uses the integrator's temperature as
     `max_temperature_K` (20);
   - the independent BAOAB reference is ASE 3.27's class with only the drag
     line patched, plus a float64 single-step check from the paper (21).
22–28. Decided (user, 19 October 2026, the phase 1b review list):
   - `fm.md`'s Langevin friction default is ASE's units as in `main.py`:
     0.02 per ASE time unit, about 0.001965/fs (22);
   - resuming with a different MACE or FastMACE version warns and records
     both versions; any other identity change still refuses (23);
   - public `fm.load` returns a pristine model, and each plan converts its
     own copy (24);
   - the float64 energy shift measured in 1b-T03 is accepted. Energies from
     runs before 19 October 2026 carry the float32 summation error and are
     labelled by `float64_energy` (25);
   - the 1e-7 eV/Å³ stress floor stays. The in-library gate refusing
     POLAR-1-S NPT is accepted, since PolarMACE NPT is experimental; the
     precision of the hybrid polar stress path is investigated in phase 3b
     (26). Resolved in part by 3b-T08 and 4-T11: the 4.4e-4 eV/Å³ gap on
     NbCl5 3×3×3 came from the finite-difference reference, not from the
     stress. A Bragg vector crossing graph_longrange's hard k-space cutoff
     under the ±1e-4 strain steps the energy. The finite-difference
     self-test (`ReferenceCalculator.check_stress_fix`) now holds the
     unstrained cell's included k-set and runs on the run's own structure,
     not on NaCl 3×3×3 only. POLAR-1-S float64 gives 8.3e-9 eV/Å³ on
     NbCl5 and 6.9e-9 on NaCl (RTX 5060 Ti, 23 October 2026). The
     PolarMACE stress rule of the gate (phase 4 D6) is still open;
   - NaCl's 1b/1a timing (0.5 % faster on murphy, 0.3 % slower on the RTX
     5060 Ti) is accepted as no regression (27);
   - ResearchModule's `main.py` moves to the 1b API (`Simulation`) as the
     first cross-repository task of phase 2. Conditions: an A/B against
     the current adapter on the ultrafast and anneal profiles (bitwise on
     plain cuEq float64, statistical on the hybrid), `run.json` keeps its
     keys, and `--engine gpu` and the old adapter stay for one phase (28).
4. Licence — **decided (user, 17 October 2026): MIT** (§14).
29–30. Decided (user, 19 October 2026, the phase 2 plan's D3 and D4):
   - the fallback of §9.4 is automatic by default, with `fallback="refuse"`
     to opt out. A fallback is never silent: plan revision, report line and
     metadata (29);
   - `main.py`'s output after its move to `Simulation` (decision 28): crash-safe
     chunks while running, then a single `md.traj` and thermo log assembled
     when the run completes or stops cleanly, so `--quench` and the analysis
     scripts keep working (30).
5. CI GPU runners — **decided (user, 19 October 2026, phase 2 D2): the
   local RTX 5060 Ti (WSL, Blackwell sm_120) and murphy (RTX 4000 Ada,
   sm_89)** as user-level self-hosted GitHub Actions runners. murphy's
   runner waits while another user's process is on the GPU, and keeps its
   files under `/mnt/homes/zwingel`. The owner registers the runners
   themselves, so the token never passes through an agent (plan task 2-T18).
6. Constant-pressure melt–quench of the chlorides — **decided (user,
   16 October 2026): part of the first production deliverable**, so GPU D3
   is on the critical path (phase 3b, §16). If the chlorides fail
   physically because the model's training data do not cover those
   conditions (for example NbCl5 evaporating at 600 K even with D3), that is
   an accepted model limitation, not a library failure: acceptance is
   implementation correctness (§11.2).
7. Gate floors — **accepted (user, 16 October 2026):** 0.01 meV/atom for
   total energy and 0.01 k_B T_min for move-energy differences (§12.3). With
   them, the ΔE rule became the floor alone and the MC path sums energies in
   float64 (§10.4), because float32 totals cannot resolve the floor. The
   floors for forces (1 × 10⁻⁵ eV/Å) and stress (1 × 10⁻⁷ eV/Å³) proposed
   in draft 0.4 were **accepted too (user, 16 October 2026)** (§12.3).
8–14. Decided (user, 17 October 2026, answering the review of draft 0.7):
   `min_mc_temperature_K` is required for MC (8, §9.2); `run` refuses a
   schedule above `max_temperature_K` (9, §7.6); MC with constraints is
   refused in 1.0 (10, §4.4, rule 1); `Simulation(atoms, model, species=…)`
   (11, §7.6); the `ForceField` is plan-free (12, §7.6); reproducible mode
   uses the unfused plain-cuEq convolution (13, §9.3); a process is
   reproducible or not, mixed plans are refused (14, §8.7).
31–36. Decided (user, 22 October 2026, the phase 3 plan's D1–D6, the
   recommendation of each; `plans/phase-3.md`):
   - ResearchModule's old `main.py` adapter (`--fastmace-api adapter`,
     `FixedPlanner` plus the research loop) is removed in phase 3.
     `--engine gpu` stays for A/B baselines and for the options `Simulation`
     refuses (31);
   - production may run on Modal: a documented recipe and a thin wrapper,
     one crystal per job, from a pinned commit, under the monthly budget
     guard (at most the free $30 credit, everything metered included, owner
     22 October 2026) (32);
   - the swap-MC annealing protocol of phase 3 (§10.5) is O ↔ Cl anion swaps
     in NaNbOCl4 (Ma 2026, I4/m, 3×3×3). It uses the anneal profile at 1 fs
     and `Swap(every=100, attempts=50, velocities="exchange")` with
     `min_mc_temperature_K=250`, on MH-1 `matpes_r2scan`. Its physical
     outcome is recorded, not judged (33);
   - phase 3's "one workstation GPU" (§16) is whichever Ada GPU is
     available at acceptance time: leeloo or murphy when idle, else the
     Modal L4 (sm_89). A workstation is preferred for timing items (34);
   - MC plans always run the move-energy part of the gate (§12.3), even
     though the gate is opt-in for MD (decision 3). The energy, force and
     stress parts stay opt-in. A move-energy gate whose reference is
     `unavailable` makes `mc_energy_precision="auto"` choose float64. This
     resolves §10.4's "accuracy prerequisite" against decision 3 (35);
   - the production timestep is 1 fs (user, 22 October 2026). FastMACE's own
     defaults (`fm.md`, CLI) stay 0.5 fs, safe for models with hydrogen, and
     the `init` template for inorganic crystals suggests 1 fs.
     ResearchModule's `main.py` keeps physical output intervals: frames
     every 10 fs, restarts and quench bins in time (36).
37–38. Decided (user, 22 October 2026, phase 3's owner checkpoints 2b and
   2c, the recommendation of each; `plans/phase-3.md`):
   - **The `fixcm` projection.** Under NVT `fixcm`, MD as well as MC uses
     the mass-weighted centre-of-mass projection by default, and the
     temperature output counts 3N − 3 degrees of freedom (§10.4).
     - Why: ASE's projection runs light species hot when the masses differ.
       3-T22 measured at most 0.2 % per species in the 3×3×3 production
       crystals, but Li +233 K in a 16-atom Li/Cs cell at 600 K. The
       mass-weighted projection costs nothing measurable (3-T22b: 0.09 % on
       NaCl).
     - `"ase"` stays as an explicit option for ASE and research-engine
       comparisons. ResearchModule's `--engine gpu` keeps it (37).
   - **Rule 9 is amended** (§9.3): host callbacks, `fastmace.ase` observers
     included, keep the graph tier. 3-T06 measured the compiled tier 4.3–4.7×
     slower than graph on NaCl and 1.3–1.5× slower on NbCl5, with a callback
     every 1, 5 or 10 steps (MH-1 hybrid float32, RTX 5060 Ti) (38).
39. Decided (user, 22 October 2026, before phase 3's acceptance runs):
   acceptance is tested according to risk. This amends decision 34 for the
   phase 3 acceptance; the owner's words: "Apply all three: leeloo's
   acceptance tier is the one Ada spot check (no L4 runs), item 9 as the short
   protocol at the freeze citing 3-T14's full runs, and fewer timing repeats
   where the effect is large".
   - The workstation GPU of §16 is one Ada spot check per phase: the GPU
     acceptance tier on an Ada GPU. Not every item runs on Ada. Items run in
     full on the RTX 5060 Ti.
   - An end-to-end protocol item may run its short protocol at the freeze. It
     cites full-length runs made earlier in the phase, together with the code
     difference between those runs and the freeze.
   - Timing items repeat only as often as the effect's size needs. A 4×
     difference needs one block; a 1 % one needs the full alternating design.
   - Later the same day (user): "Yes: stop the full local tier and run only
     the 7 corrected tests, the GPU slow tier, item 9's short protocol, item 3
     and item 10's tiny-model bitwise check". When the GPU acceptance tier
     has passed on workstation GPUs at the freeze, the local leg reruns only:
     - the tests corrected at acceptance;
     - the GPU slow tier;
     - the items' own runs.
   - *Amended (user, 23 October 2026, "yes" to the conductor's proposal):*
     acceptance splits in two.
     - **Full acceptance**, once per phase, on the reference GPU (the RTX
       5060 Ti): the acceptance and slow tiers, the physics comparisons and
       the timed A/Bs. Its untimed runs may also run elsewhere (a Colab A100)
       to finish sooner; timed runs stay on the reference GPU.
     - **GPU compatibility check** (`-m compat`, about 30–60 min) on every
       production GPU type each phase, and once when a GPU type is added. It
       covers what differs between architectures:
       - kernels compile and attach (every rewrite, OEq, Triton, no naive cuEq fallback);
       - the accuracy gate on MH-1 and a few other models;
       - CUDA-graph replay equal to eager, with the kernel count recorded;
       - reproducible mode bitwise;
       - a short MD smoke (NVE drift, NVT temperature, one NPT run);
       - a restart round trip;
       - the fast GPU tier.
     - It replaces "one Ada spot check". A Colab L4 may stand in for the Ada
       workstations, for correctness only.
     - Production GPU types: Ada (the workstations), Blackwell (the local
       PC), and the Colab A100 if its compatibility check passes.
     - Phase 4-T13 defines the tier.
40–43. Decided (user, 22 October 2026, the phase 3b plan's D1–D4, the
   recommendation of each: "Accept the recommendations for phase 3b's D1-D4
   and prestart the phase 3b workers"; `plans/phase-3b.md`):
   - **The acceptance melt–quench protocol** (40):
     - NbCl5 3×3×3 on MH-1 `matpes_r2scan` with D3-BJ (`r2scan`);
     - hydrostatic NPT at 0.1 GPa: the plan's alternative (b), chosen by the
       owner on 23 October 2026 after the 0 GPa run vaporised (3b-T07);
       the 0 GPa results are kept as context;
     - the anneal profile at 1 fs, with the library's barostat defaults;
     - relaxation and quench at fixed cell, the quench frames at their NPT
       cells;
     - the outcome is recorded, not judged.
   - **PolarMACE in 3b** (41): only the hybrid polar stress precision is
     investigated: POLAR-1-S on NbCl5 3×3×3 against finite differences, as
     decision 26 asked. There is no qualification, and PolarMACE + D3 stays
     refused.
   - **D3's cost target** (42): if D3 misses the 10 % target of §11.2, the
     miss is recorded and the phase continues. Optimisation is a follow-up.
   - **ResearchModule `main.py`** (43) gets `--dispersion off|d3bj` for NVT
     and NPT, default `off`, recorded in `run.json`. Whether the chlorides'
     production NPT defaults to `d3bj` is decided after acceptance.
44–49. Decided (user, 23 October 2026, the phase 4 plan's D1–D6, the
   recommendation (a) of each: "For the other decisions I trust your
   recommendations"; `plans/phase-4.md`):
   - **`FixAtoms`** (44): with NVE, Langevin, CSVR and NVT Berendsen, and
     with FIRE and LBFGS at fixed cell. It is refused, with the reason, for
     NHC and MTK (ASE refuses them too), for every variable cell, and with
     `fixcm=True` (§4.4).
   - **Degrees of freedom** (45): with `fixcm=True` the new integrators remove
     the total momentum at the start, project it out each step and count
     3N − 3. With `FixAtoms` they count 3(N − n_fixed) and do not project.
     `dof="ase"` reproduces ASE's counts, and the ASE-equivalence tests use
     it.
   - **The annealing-benchmark minima** (46): GPU LBFGS (fmax 0.01 eV/Å,
     memory 100, maxstep 0.2 Å, fixed cell, at most 5,000 steps) against
     ASE's LBFGS on `fm.calc` on the 210 `captured.traj` frames. Pass: all
     converge; |ΔE| ≤ 0.025 meV/atom and every atom within 0.1 Å, each in at
     least 20 of 21 frames per crystal. The 13 October research-engine minima
     are compared and recorded, not judged. *Amended (owner, 23 October
     2026):* the energy clause was per frame. 4-T08 missed it on one frame
     (NaTaCl6 frame 12, 0.188 meV/atom), which has two minima that float32
     rounding chooses between; float64 GPU and ASE agree to 1.6e-12. So the
     energy clause takes the structure clause's 20-of-21 allowance.
   - **Protocol relaxations** (47): once 4-T08 passes, `[relax]` and
     `[quench]` use GPU LBFGS by default. `"ase-LBFGS"` and `"ase-BFGS"` stay
     options, and `protocol.json` records which ran.
   - **Optimisers in `fastmace.ase`** (48): `LBFGS` and `FIRE` with ASE
     3.27's arguments, on `Atoms` or `FrechetCellFilter`; other cell filters
     refused (§7.7).
   - **PolarMACE's stress floor** (49): 5e-7 eV/Å³ (0.08 MPa) in the
     reference record, for the stress part only. The energy and force floors
     stay.

50. Decided (user, 23 October 2026, 4-T12b's option (c)): the planner refuses
   PolarMACE with a variable-cell optimiser (`cell_filter="FrechetCell"`,
   code `polar-variable-cell-optimizer`) on the graph and reference tiers.
   Fixed-cell PolarMACE relaxations and other models' cell relaxations still
   plan; ASE's own optimisers on `fm.calc` are not refused. 4-T07's divergence
   on POLAR-1-S NbCl5 comes from the model's own surface (forces up to
   120 eV/Å in the float64 reference, about 70× MH-1's), not from the k-set
   or the stress (`docs/validation/phases/4-T12b.md`). PolarMACE NPT is not refused
   and stays experimental. Lifting the refusal needs a converged reference
   cell relaxation on a sane structure, NaCl first; freezing the k-set per
   rebuild and a smooth cutoff were not chosen.

51–56. Decided (user, 23 October 2026, the phase 5 plan's D1–D6, the
   recommendation of each: "Go ahead and do all recommended decisions for
   D1-D6"). Details are in `plans/phase-5.md`, "Owner decisions".
   - **Rule 12 action** (51, D1 (a)): above 90 % of free GPU memory, turn off
     the rewrites whose records declare an edge-sized buffer, largest term
     first, each with rule 12's reason. If it still does not fit, keep the plan,
     warn with the estimate, and let an out-of-memory error carry the estimate.
     Never shrink rule 11's skin or headroom. Never change the tier, the
     precision or an explicit user setting. The margin and the estimate's
     accuracy come from 5-T03's measurements, and the estimate is never below the
     measured peak on the 19 labels. Amended by decision 67 (29 October 2026): a
     rewrite is turned off only while the estimate with it on does not fit, and
     only if its term lowers the estimate.
   - **Adaptation trigger** (52, D2 (a)): the re-capture budget is exceeded and
     re-capturing takes more than 20 % of wall time over the segments since the
     last switch. The one switch is graph → compiled (same backend, rewrites
     and precision) at the next canonical point. Rebuild rate, build time and
     overflows are recorded, not acted on.
   - **Adaptation limits** (53, D3 (a)): one way, at most one switch per run. No
     switch under `reproducible=True`, with an explicit `mode=`, during an MC
     block, or in a subprocess child without the parent's hand-off. A switch is
     validated before activation (self-checks; the gate when requested; the
     move-energy part for MC) and logged.
   - **Mid-run fallbacks** (54, D4 (a)): a capture error at a re-capture and a
     compile error of the compiled tier during `run()` trigger them. They start
     from the last committed canonical point, redo the steps after it, and
     handle output as on resume. Out-of-memory, non-finite values, refusals and
     process constraints are not triggers.
   - **Calibration** (55, D5 (a)): `calibrate=True`, off by default. It covers
     the backend dimension only (hybrid against plain cuEq, when both pass the
     hard rules and no backend is set). Each candidate runs in a child process,
     50 steps after warm-up, alternating. It changes the choice only when the
     other candidate is faster by more than 5 % in every pair. It is recorded as
     measured, and nothing is read from earlier runs.
   - **Two-stage `auto`** (56, D6 (a)): `FixedPlanner`'s two stages stay as
     they are in phase 5, covered by the conformance suite. Reconsider in phase
     6. Task 5-T10 is closed.

57. Decided (user, 24 October 2026, optimisation round 4, OPT4-T04: "Go
   ahead and do it as you recommended"): `RewriteConfig.grouped_launch` is on
   by default for the generic MACE family only (MP-0, MPA-0, OMAT-0, MatPES
   and the like: every interaction MP-0's residual block or the density
   family's, not PolarMACE). MH-1 and the other families (MACE-OFF, POLAR-1,
   OMOL-0) stay off until each is measured. Float64 and `reproducible=True`
   plans keep the stock products; an explicit `on` or `off` overrides the
   rule for any family. Same-process A/Bs on NaCl gained 4.5–5.4 % on the
   RTX 5060 Ti (five labels, both orders) and 4.1–4.3 % on the L4 (OMAT-0);
   NbCl5 was neutral (`docs/validation/speed/opt4-T04.md`).

58. Decided (user, 24 October 2026, optimisation round 4, OPT4-T02: "Yes",
   answering "both T02 records on by default for POLAR-1-S and -M; POLAR-1-L
   only after a local NbCl5 gate passes"): `RewriteConfig.fx_e3nn_linear` and
   `RewriteConfig.polar_graph_sums` are on by default for the qualified POLAR-1
   checkpoints, POLAR-1-S and POLAR-1-M, in hybrid float32 plans that are not
   reproducible. The planner identifies the checkpoint by its label-table label
   (by label, or by SHA-256 and head, as rule 15 does); PolarMACE with any
   other checkpoint, a model of unknown identity and every other family stay
   off. Float64 and `reproducible=True` plans keep MACE's code, as under
   decision 57 (`polar_graph_sums` is deterministic, but it stays with
   `fx_e3nn_linear` on the compiled hybrid path, where its gain and its
   gate are). An explicit `on` or `off` overrides the rule for any model.
   Same-process A/Bs on POLAR-1-S, both records against the default: NaCl
   −24 % and NbCl5 −18 % on the RTX 5060 Ti, −19 % and −16 % on the L4; the
   force gate passed for S and M on NaCl and NbCl5 2×2×2
   (`docs/validation/speed/opt4-T02.md`). POLAR-1-L met the owner's condition the
   same day: its local NbCl5 2×2×2 gate passed (energy, force and stress
   ratios 0.88, 0.96, 0.75), and it is a qualified checkpoint too.
59. Decided (user, 24 October 2026, optimisation round 4, OPT4-T05: "Yes",
   to "`dynamic_edges` on by default"): `EngineConfig.dynamic_edges` is on by
   default, so a graph-tier MD plan's compiled model sees the edge dimension as
   dynamic from its first call. Before, that call specialised on the exact
   initial edge count, and capturing B at the padded count compiled the whole
   model again. The edges stay padded to buckets and every graph is still
   captured. The steps replay the same compiled code: the default's second
   compile and the option's single compile wrote and read the same Inductor
   and AOT-autograd cache entries. `reproducible=True` trajectories are bitwise
   equal with it off. A start is 7–9 s shorter warm and 16–22 s shorter cold
   (RTX 5060 Ti, MH-1 NaCl and NbCl5, OMAT-0 NaCl;
   `docs/validation/speed/opt4-T05.md`). Plans that compile nothing (eager,
   reference, `compile_mode="none"`) turn it off with a reason; an explicit
   `False` restores the second compile.
62. Decided (user, 27 October 2026, optimisation round 6, OPT6-T05: "yes
   default"): `RewriteConfig.radial_tables` is on by default wherever
   `fold_cutoff` is (the factored MH-1 class), in float32 plans that are not
   reproducible (rule `on_with_fold_cutoff_float32_unless_reproducible`).
   Float64 and reproducible plans keep MACE's radial chain; the record's
   self-check (its certificate at most 1) can still turn it off (rule 13).
   Measured: certificates 0.15–0.30, force gate passed, 20 ps NVE drift
   unchanged, full-step b/a NaCl 0.90, NbCl5 0.92, NaNbO3 0.81 on the RTX
   5060 Ti; on the RTX 4000 Ada (sm_89, leto) NaCl 0.914, NbCl5 0.904,
   NaNbO3 0.832 in both build orders (`docs/validation/speed/opt6-T05.md`).
63. Decided (user, 27 October 2026, f64-T01: "do that", to the float64
   package): float64 plans that are not reproducible take (a) the hybrid
   backend where the planner allows it (already the minimal planner's rule),
   (b) `EngineConfig.fp64_emulation`, cuBLAS's float64 emulation on INT8 tensor
   cores (strategy eager), where the GPU's measured float64 multiply-add rate is
   below 1/8 of its float32 rate (`kernels.triton.fp64_rate`; the compute
   capability cannot tell), set before the first float64 product and checked
   at build, (c) the rounding-class rewrites `species_tables`,
   `fold_readout_tail` and `fold_product_linear`, and (d) float64 radial tables
   (fit and certificate held to float64 rounding: tolerance 3e-13, bound 1e-12).
   Reproducible plans keep native products and MACE's code. Measured: MD step
   0.57 (NaCl), 0.42 (NbCl5), 0.49 (NaNbO3) of stock float64 on the RTX 5060 Ti;
   forces within 4e-14 eV/Å, stress within 1e-16 eV/Å³; 20 ps NVE drift equal
   to stock to seven digits (`docs/validation/speed/f64-T01.md`).
64. Decided (user, 27 October 2026, optimisation round 8, OPT8-T02: "Make it
   default for mh-1"): `RewriteConfig.fused_folded_forward` is on by default
   for MH-1 (the label-table labels of the `mh-1` checkpoint: `mh1-r2scan`,
   `mh1-omat`, `mh1-mp`, found by label or by SHA-256 and head) in float32
   plans that are not reproducible, wherever `chunked_conv_backward` is on
   (rule `on_for_mh1_float32_unless_reproducible`). Every other model keeps
   the cuBLAS product unless asked: POLAR-1-S was neutral (1.004) and OMOL-0
   XL 3.5 % slower (1.035). Measured on MH-1: exact, force gate passed, 20 ps
   NVE drift unchanged; b/a in both build orders NaNbO3 0.928, NbCl5 0.967,
   NaCl 0.996 on the RTX 5060 Ti and NaNbO3 0.958, NbCl5 0.982, NaCl 0.981 on
   the RTX 4000 Ada (murphy) (`docs/validation/speed/opt8-T02.md`).
65. Decided (user, 28 October 2026, ideas-20261027: "Make all 3 default on
   gpu where we measured a speed up but if slowdown or nothing measured dont
   enable it"): in float64 MH-1 plans that are not reproducible, three float64
   speed-ups are on by default on the GPU models where same-process A/Bs in
   both orders measured a speed-up, matched by exact device name
   (`planner.fixed.DECISION_65_GPUS`: "NVIDIA GeForce RTX 5060 Ti", sm_120;
   "NVIDIA RTX 4000 Ada Generation", sm_89). A compute capability would take
   in GPUs never measured (the L4, other sm_89 and sm_120 cards), so it is not
   used. The three:
   (a) D1, `EngineConfig.fp64_emulation_mantissa_bits` "auto" = 56 fixed
   mantissa bits in cuBLAS's float64 emulation (only where decision 63's
   emulation is on; "dynamic" elsewhere);
   (b) D2, `RewriteConfig.w_tables` (per-piece folded radial tables, rounding,
   nondeterministic), which also needs `radial_tables`,
   `chunked_conv_backward` and at most three species (its tables grow with
   the square of the species: 1.3 GB on NaNbO3; `W_TABLE_MAX_BYTES` 2 GiB);
   (c) D3, `RewriteConfig.save_transcendentals` (a custom AOTAutograd
   partitioner that saves transcendental ops for the backward; exact).
   Off everywhere else: on the A100 (measured slower), on every unmeasured
   GPU, on the CPU, in float32 and reproducible plans, and for other models
   (all measurements are MH-1). D3 is deterministic, but reproducible plans
   keep it off too: they take native products and MACE's code, a route the
   speed-up was not measured on. The off reason names the GPU, for example
   "not a measured GPU, decision 65 (NVIDIA A100-SXM4-40GB)". An explicit
   "on", "off", "dynamic" or bit count still overrides. Adding a GPU to the
   list needs its own A/B of all three in both orders. Measured (MD step,
   new/old; D2 and D3 same-process A/Bs in both build orders, D1 ABBA in
   separate processes; `plans/ideas-20261027.md`, "Per-GPU confirmation"):

   | Idea, cell | RTX 5060 Ti | RTX 4000 Ada | A100-SXM4-40GB |
   | --- | ---: | ---: | ---: |
   | D1 NbCl5 / NaNbO3 / NaCl | 0.877 / 0.881 / 0.862 | 0.876 / 0.884 / 0.858 | not applied (emulation off) |
   | D2 NbCl5 / NaNbO3 / NaCl | 0.911 / 0.855 / 0.942 | 0.918 / 0.870 / 0.944 | 1.063 / 1.103 / 1.268 |
   | D3 NbCl5 / NaNbO3 / NaCl | 0.969 / 0.980 / 0.975 | 0.971 / 0.982 / 0.979 | 1.010 / 1.022 / 0.994 |

   All three together: 0.797 (NbCl5), 0.782 (NaNbO3), 0.806 (NaCl) locally.
   Forces within 2–5e-15 eV/Å of the decision-63 package; 20 ps NaCl NVE
   drift unchanged for each idea alone and for all three together.
66. Decided (user, 29 October 2026, on OPT9's A100 ceiling: "Go ahead and make
   it default on gpus where its faster overall the no per gpu optimization
   decision has been overruled per gpu optimization is fine"): the generality
   rule's ban on per-GPU choices (goal 2, §9.1, project rule of 10 October
   2026) is lifted. A rewrite may be on by default on the GPU models where
   same-process A/Bs of the whole MD step, in both build orders, measured it
   faster, matched by device name as in decision 65 (an unmeasured GPU of
   the same capability stays off). Amended (owner, 7 November 2026: "if its
   tested on one variant of a gpu it counts for all gpus (unless the variants
   vary significantly in performance)"): a name stands for every variant of its
   GPU family (`kernels/gpu_models.py`: `GPUModels`, `family`, `lookup`). The
   families are the A100 (SXM4 and PCIe, 40 and 80 GB; owner, 4 November 2026)
   and the H200 (SXM and NVL, same chip, SM count and memory). Variants that
   differ significantly in performance stay separate names: the H100 PCIe (114
   SMs, HBM2e) is not the H100 SXM (132 SMs, HBM3). Amended again (owner,
   7 November 2026, "Yes" to the off-polar worker's rule): a switch that only
   removes work, deleting operations or memory traffic without adding or
   replacing a kernel, changing tiling, launch shape or precision, or keeping
   more memory alive, is on for every GPU (a plain `True` default, no
   `*_GPUS` set) once one same-process MD-step A/B on any GPU shows it not
   slower in both build orders and it is exact or passes
   `tools/force_stress_gate.py` (3 × float32 rounding since 7 November 2026).
   Fusions, kernel substitutions, retiling and
   precision changes stay per GPU family. Everywhere else it stays off or opt-in, and
   the planner's off reason names the GPU. The usual gates still apply per
   record (force/stress gate, NVE drift, exactness class). Per-structure and
   shape-keyed tables stay forbidden, and measured values are never stored or
   looked up at run time. First candidate: OPT9's w'-dot radial backward
   (`plans/opt-9.md`), estimated about 11 % (NaNbO3) and 7 % (NbCl5) on the
   A100 and a loss on the RTX 4000 Ada; it becomes default only where its step
   A/B measures a speed-up. Earlier findings closed only for being per-GPU
   (for example `grouped_launch` for MH-1 on sm_120) may be reopened the same
   way. Applied first (conductor, 29 October 2026, under the owner's "make it
   default on gpus where its faster overall"): `grouped_launch` for MH-1 on
   "NVIDIA GeForce RTX 5060 Ti" (`planner.fixed.DECISION_66_GROUPED_LAUNCH_GPUS`;
   NaCl 0.971 / 0.975, NbCl5 and NaNbO3 neutral, same-process both orders on
   main `30ff3b7c`; `docs/validation/speed/decision66-grouped-launch.md`). Second
   (owner, 29 October 2026): `wprime_backward` (OPT9) for MH-1 in float32
   plans that are not reproducible on "NVIDIA A100-SXM4-80GB", "NVIDIA GeForce
   RTX 5060 Ti" and "NVIDIA RTX 4000 Ada Generation"
   (`planner.fixed.DECISION_66_WPRIME_GPUS`; MD step b/a in both orders 0.81-0.89
   on the A100, 0.94-0.97 on the RTX 5060 Ti, 0.93-0.96 on the RTX 4000 Ada,
   every block pair below 1; `docs/validation/speed/opt9-T03.md`). Third (conductor,
   2 November 2026, under the owner's OMAT-0 capacity request):
   `generic_chunked_conv_backward` for the generic family on "NVIDIA GeForce RTX
   5060 Ti" and "NVIDIA RTX 4000 Ada Generation"
   (`planner.fixed.DECISION_66_GENERIC_CHUNKED_GPUS`; OMAT-0 MD step b/a in both
   orders 0.916-0.953 on the RTX 4000 Ada, 0.949-0.972 on the RTX 5060 Ti; off on
   the RTX A4000, slower there, and on the A100, not measured;
   `docs/validation/omat0/omat0-capacity.md`).
67. Decided (user, 29 October 2026, on the capacity scan, CAP-T02 in
   `plans/capacity.md`: "It should only do this if memory would not be
   enough"): rule 12 turns off `streaming_conv` and the records that require it
   (`chunked_conv_backward`, `fused_folded_forward`, `wprime_backward`) only when
   the estimate with them on does not fit the free memory (decision 51's margin),
   and only if turning them off lowers the estimate. The estimate predicts the
   measured limit: MH-1's fused path declares what it does not keep per edge.
   `chunked_conv_backward` and `radial_tables` carry their measured savings
   (2,086 and 2,306 bytes per edge in float32), and MH-1's streaming convolution
   the remainder of the fused path, fitted per precision on the capacity scan's
   peaks (RTX 5060 Ti under WSL2, RTX 4000 Ada under Linux). The estimate then
   stays between 1.0 and 1.4 times the measured peak on NaCl, NbCl5 and SrTiO3
   up to 30,600 atoms. Before, it was 1.5-2.3 times, so rule 12 turned the fused
   path off from about 13k atoms. That raised the peak, slowed the step and, under
   WSL2, let OEq's own convolution kernels reserve several GiB of stack memory.
   Turning the fused path off never lowers the estimate now, so rule 12 keeps it.
   `tests/cpu/test_memory_fit.py` holds the fit; `docs/validation/capacity/capacity-T02.md`
   the measurements.

---

## 19. Revision history

**Draft 0.2 (16 October 2026)** answers a review of draft 0.1 that checked
the specification against the force wrapper, the NVT and NPT integrators,
the PolarMACE path and the validation scripts. Every finding was verified
against the code before it was adopted; all were adopted.

| Finding | Where answered |
| --- | --- |
| Keeping velocities on mass changes with a potential-only criterion breaks detailed balance; proposals and the chemical-potential convention undefined | §10.4 (moves, `rescale`/`redraw`, acceptance, convention); §12.2 enumeration test |
| Updating species and mass buffers leaves mass-dependent integrator coefficients stale; energy-only MC leaves cached forces stale | §8.5 derived-state registry and transaction contract |
| Restart cannot reconstruct an exchange trajectory; prefetched noise missing | §10.6 complete state and force-field identity |
| Stock-MACE reference conflicts with the PolarMACE stress fix | §12.1 versioned reference policy; §11.1 |
| Energy and move-energy differences not gated | §12.2, §12.3 |
| Gate calls the force field directly, not through captured replay | §8.1 overflow before commit; §12.2 replay and transitions; §12.3 |
| "Fall back one tier" conflates mode, backend and rewrites; rule precedence undefined | §9.3 dimensions and precedence; §9.4 fallback algorithm |
| `reproducible=True` overpromises (MACE's `scatter_add_` is nondeterministic on CUDA) | §9.3 scope, enforcement and qualification |
| Shared PolarMACE `_Holder` couples instances | §8.7 instance isolation; §11.1 |
| No policy when the float64 reference does not fit | §12.3 statuses and resources |
| "NbCl5 NPT with D3 stable at 600 K" is not an acceptance criterion; D3 underspecified | §11.2 parameters and correctness criterion; §16 |
| Milestone order contradicts the planner's role and the MC deliverable | §16 reordered and re-estimated |
| Example requests 200,000 steps of a 540,000-step schedule | §7.2 schedule semantics |
| Units, stress convention, buffer ownership, force vs integration precision | §7.3 |
| PBC and constraint support undeclared | §4.4 |
| Performance target with or without MC evaluations | §10.4 cost; §16 phase 3 |
| Baseline from mixed machines | §3.3; §12.5; §16 phase 1 |

**Owner decisions (16 October 2026, same day).** Decision 6: constant-pressure
melt–quench of the chlorides is part of the first production deliverable;
GPU D3 moved from phase 4 into a new phase 3b, and physical failures outside
the model's training data are accepted (§11.2, §16). Decision 7: the gate
floors were accepted. Checking them against the NbCl5 3×3×3 relaxation log
showed that float32 totals are spaced about 2 meV apart, ten times the ΔE
floor at 250 K, so the ΔE rule lost its ratio term and the MC path sums
energies in float64 (§10.4, §12.3).

**Draft 0.3 (16 October 2026)** answers a second review, of draft 0.2, that
cross-checked it against `gpu_langevin.py`, `gpu_npt.py`, `polar_gpu.py`,
`edge_outer_sum.py`, `pair_search.py`, MACE 0.3.16, `torch-dftd` 0.5.1 and the
recorded gate metadata. Every claim was checked against code, checkpoints and
results; all were adopted, two with a different mechanism and two with
corrected figures (below).

| Finding | Where answered |
| --- | --- |
| All-electron E0 (−2,048 to −70,042 eV per atom) and float32 sums break energies and the ΔE gate; `out["node_energy"]` omits terms of `ScaleShiftMACE` and `PolarMACE` | §4.3 energy scale; §7.3 energy; §6.2 `float64_energy`; §12.1. Checked: POLAR-1/OMOL-0 NbCl5 float32 energies off by 45–102 meV/atom. The review's Br E0 (−74,936 eV) is −70,042 eV in these checkpoints. Beyond the review: float32 forwards also carry 0.05–0.13 meV/atom errors on MH-1 r²SCAN, so the MC energy precision became a plan dimension with a float64 fallback (§9.3, §9.4, §10.4) |
| Absent superset species: first-atom `argmax` in the reassociation table, silent slot 0 for undeclared species, sender groups frozen at the first build with `max() + 1` groups | §8.5 species superset |
| Energy-only evaluation must run under `no_grad`; swaps/transmutations differ from displacements; no `nonzero` in captured selection; displacement rejection must restore the neighbour list; overflowing attempts must be replayable | §8.1 energy-only graph; §8.5 steps 2–4; §10.4 fixed-shape selection, fixed positions versus displacements |
| Restart misses build cell h₀, build-time sender groups, pipeline floor and edge bucket, D3 list state | §10.6 canonical restart points (a rebuild at every restart write, in both runs) instead of serialising each item, plus the runtime-state registry |
| D3: `torch-dftd` uses `nonzero` (not capturable), its own unit constants and float32 by default; per-step compaction of a 40-bohr list is too expensive | §11.2 design bullets and parameter table. Candidate count recomputed as about 3 million for NbCl5 3×3×3 |
| Velocity versus momentum formulas; `fixcm` with MC; incomplete capture warm-up restore | §10.4 momenta (both forms) and centre of mass, with a new `exchange` swap that conserves kinetic energy and total momentum exactly; §8.1 capture warm-ups |
| Reproducible mode and `edge_outer_sum` atomics; aliovalent transmutations in charge/spin models; sign of the μ conversion | §9.3 enforcement; §10.4 charge and spin; §10.4 acceptance (explicit formula) |

Also in draft 0.3: the two upstream MACE changes that the research engine is
adding on 16 October 2026 (`langevin/upstream_mace.py`): the single-graph
strain of MACE PR 1498 as a pending catalogue entry (§6.2), and the check for
cuEquivariance's silent naive fallback (MACE PR 1614) as a planner trait
(§9.2). Phases 1, 3 and 3b grew; the first deliverable is now 17–20 weeks
(§16).

**Draft 0.4 (16 October 2026)** answers a third review, of draft 0.3. Its
points were checked against the spec text and the installed torch 2.14.1.
Adopted:

| Finding | Where answered |
| --- | --- |
| Restart promised "as if not interrupted" but defined bitwise continuation only for the same restart cadence | §10.6 opening (the cadenced definition is the only one); §9.3 scope |
| Planner order numbers repeated (1 three times, 2 three times, 8 twice), so `species_changes` (graph mode) and `padding_is_exact` false (no graph mode) sat at the same level; overrides versus refusals unspecified | §9.3 strict order 1–15 in three kinds (hard, preference, reaction); §9.5 overrides and plan revisions |
| No backend or model family is qualified for `reproducible=True` | §9.3: refused until qualified; qualification of one family in phase 2 (§16) |
| D3: directed count versus undirected kernel; no cost target | §11.2 half list (1.5 million stored pairs for NbCl5 3×3×3); cost target ≤ 10% of the MD step, provisional |
| Phase 1 bundles too much; subprocess runner promised but absent from the architecture | §16 phase 1 split into 1a extraction and 1b contracts, estimates stated as floors (24–32 weeks); §5 `engine/subprocess.py`; phase 2 |
| API illustrative: no exceptions, ASE properties, checkpoint trust (pickles execute code), schedule algebra; species superset in both `ForceField` and `Workload` | §7.6 API contracts; `Workload.species_superset` removed (§9.2) |
| Force and stress ratio rule unstable near zero; `passed-reduced` scope | §12.3 floors (proposed 1e-5 eV/Å, 1e-7 eV/Å³); reduced cells stand in only for the numerical reference comparison, replay and transition checks always run on the production cell |
| Interacting rewrite faults; unbounded gate re-runs | §9.4 ddmin over the rewrite set, reduced gate during diagnosis, at most 12 gate runs |
| Impossible-attempt counters must stay on the device | §10.4 device-side counters read with the block flags |
| RNG states in `metadata.json` | §13: hashes and cursors only; full states in the restart file |
| Gains listed with mixed signs | §6.2 one convention (step-time change, negative = faster) |
| Frozen plan versus mid-run tier switch | §9.5 plan revisions, §9.6 |

Clarified, but the finding itself did not hold:
- **Energy/force mismatch (§7.3).** The review held that a float64-reduced
  energy is not the scalar whose gradient gives the forces. It is: the
  gradient of a sum is 1 per term whatever the accumulator's precision, so
  forces and stress are bit-identical to those of a float32 sum. Float64
  reduction makes finite-difference and NVE checks more consistent, not
  less. §7.3 now states this explicitly.
- **Transaction random numbers (§8.5).** The review called step 4 (rewind on
  overflow) and the rule that rejection keeps consumption a contradiction.
  They are different outcomes, and the text said so ("not a rejection"). The
  per-outcome table now makes it explicit. Every attempt also consumes fixed
  counts from three streams, so positions never depend on decisions.

Not adopted:
- **`reproducible=True` is dead because of MACE's `scatter_add_` atomics.**
  Torch 2.14.1 lists `scatter_add_`, `index_add` and the `gather`/
  `index_select` backward on CUDA among the operations that "will act
  deterministically" under `torch.use_deterministic_algorithms(True)`. The
  open questions are cuEquivariance's and FastMACE's own kernels (the
  segmented `edge_outer_sum` uses `tl.atomic_add`), as §9.3 now states.
- **D3 cell-size refusal for cells narrower than twice 40 bohr.** The GPU
  neighbour list enumerates every periodic image within the cutoff (the
  image range comes from the cutoff and the cell widths in `build_gpu`); it
  has no minimum-image assumption, so no refusal is needed (§11.2).
- **Full-set eager check before bisecting** and **checking the minimum, not
  the index, in MC selection** were already in draft 0.3 (§9.4 step 2,
  §10.4); both were reworded to be unmistakable.
- **The torch version policy contradicts the qualified build.** It does not:
  §4.1 separates the qualified state from the 1.0 support policy. The policy
  row was made more precise anyway.
- **Do not start phase 0 until the contracts are fixed.** Phase 0 is
  packaging and CPU CI and depends on none of the contract wording (§16).

**Draft 0.5 (16 October 2026)** answers a fourth review, of draft 0.4. All
five findings were checked against the spec and the research engine and
adopted:

| Finding | Check | Where answered |
| --- | --- | --- |
| The MC baseline energy may come from the MD force graph (float32) while trials use the float64 MC fallback, so their offset enters the acceptance | the spec named only a trial evaluation | §10.4 baseline energy: one MC-evaluator evaluation per block, a tagged MC energy cache, first-attempt test; cost counts it |
| A displacement that needs a rebuild is replayed after rebuilding at the restored positions, which loops | `max_displacement_exceeded` tests `> skin/2`; default skin 0.3 Å (`workflow.py`), so any move beyond 0.15 Å loops | §8.5 step 4: rebuild around the trial geometry in trial buffers; capacity overflow alone replays; test beyond skin/2 |
| D3 with two passes cannot complete dE/dCN_i before propagating it to forces | C6 depends on CN; the chain term needs all pairs of atom i first | §11.2 three passes with a kernel boundary; cost estimate ~150 MB per step. The reviewer's 0.0013 eV/Å force effect is cited, not re-measured |
| Accepted MC moves leave BAOAB's `force_eps` stale | `_update_force_eps` reads the stress and the kinetic energy or tensor; `_stage_a` consumes it first | §8.5 step 6: fixed refresh order, integrator caches in the derived-state registry; first-NPT-step test, both cell modes |
| Copying the ASE trajectory tests would not qualify production BAOAB | `gpu_npt.py` multiplies the trace term by the identity (Quigley–Probert Eq. 5b); ASE 3.27 adds it to all nine elements; the tests patch `ASE_TRACE_DRAG_ALL_ELEMENTS=True` | §12.1 fourth kind of entry, integrator corrections; §12.2 two-part BAOAB qualification; §16 phase 1b |

**Owner decision (16 October 2026, after draft 0.5).** The force and stress
gate floors of draft 0.4 (1 × 10⁻⁵ eV/Å, 1 × 10⁻⁷ eV/Å³) are accepted
(decision 7, §12.3).

**Draft 0.6 (16 October 2026)** answers a fifth review, of draft 0.5, that
checked the spec against `gpu_langevin.py`, `gpu_npt.py`, `pair_search.py`,
`workflow.py`, `oeq_no_nvcc.sh`, the production annealing metadata and the
uncommitted working tree. The review found the physics and numerical
contracts consistent (MC acceptance algebra and the μ convention, E0 figures
and float32 ulps, the float64-reduction gradient, the kinetic-stress split
in `gpu_npt.py`, the deterministic GPU build, the frozen sender groups) and
twelve gaps between sections; all twelve were adopted (owner decision,
16 October 2026).

| Finding | Check | Where answered |
| --- | --- | --- |
| A. `Simulation` takes a model, so the superset the `ForceField` owns cannot be declared on that path; `Workload.precision`, `ensemble`, `cell`, `needs_stress` duplicate the `ForceField` and the integrator; `run(mc=…)` undefined for undeclared moves | §7.2 example used only species present at setup; rule 10 "selects" an integrator the user already passed | §7.2 example; §7.6 Simulation inputs; §9.2 reduced `Workload`; §9.3 rules 1, 2, 10 |
| B. `atom_count_changes` had a tier and a rule but no 1.0 move, integrator or restart entry | §2, §9.2, rule 7 versus §10.4, §5, §17 | §2 row; §8.6; §9.3 hard rule 7 (refused in 1.0) |
| C. A 12-run gate budget cannot finish ddmin over ~16 rewrites, and an exhausted budget had no defined end | n log n ≈ 64 reduced-gate runs typical | §9.4 step 2: 64 reduced-gate runs, terminal sequence (rewrites off, eager, reference, refuse); step 5 |
| D. Output files written after the last restart point were not reconciled; today's resume rewrites the whole trajectory | `workflow.truncate_outputs` (`workflow.py:824`) rereads and rewrites `md.traj` | §10.6 output chunks at restart points; §5; §8.7 subprocess transport; §12.2 restart row; §13 |
| E. The D3 list shared the MACE skin, so its builds (40× the candidates, host-synchronising) would exceed the 10% target | 27,697 rebuilds in 540,000 steps (`anneal-production-rtx5060ti-20261010/NbCl5`); `build_gpu` syncs on the pair count | §11.2 own skin and test, amortised target, corrected pair counts (≈ 1,440 neighbours, 2.8 million directed at the MACE skin); §9.3 rule 11; §9.6; §13; §16 phase 3b |
| F. "One attempt is a capturable graph" also covered displacement trials that need a rebuild, which the GPU build cannot do inside a capture | `candidate_pairs` reads `counter.item()`, `build_gpu` reads `int(keep.sum())` and may reallocate | §10.4 graph capture scoped to swaps and transmutations; §8.5 step 4; §9.3 rule 11; §12.2; §16 phase 3 |
| G. A thermostat-target change was listed as an event while ramps change the target every step | today `set_temperature` fills the device scalar `sqrt_kT` (`gpu_langevin.py:2204`); `c3_unit`, `c4_unit`, `c5_unit` depend on friction and masses only | §8.5 transaction contract (scheduled targets are device-scalar fills); §7.6; §10.1 |
| H. The move-energy gate needs T_min before `run` provides the schedule | `Workload` had `max_temperature_K` only | §9.2 `min_mc_temperature_K`; §12.3; §7.6 `run` refusal |
| I. "No `nvcc`" promised without listing the host toolchain the OEq extension needs | `oeq_no_nvcc.sh`: `g++`, ninja, runtime header wheels, `nvcc` sentinel | §14 prebuilt wheels as the 1.0 path, toolchain fallback documented; §4.1 row; §16 phase 6 |
| J. §6.1 classified every adopted rewrite as "rounding", against its own definition for `sort_receivers`, `async_output`, `fold_cutoff`; no exactness column | definition: exact = identical up to summation order | §6.1 wording; §6.2 exactness column (initial classification; the self-check record is authoritative) |
| K. Wording overtaken by the working tree: single-graph strain and the naive-cuEq guard are adopted; counts stale | uncommitted AGENTS.md and `langevin/upstream_mace.py`; `workflow.py` 1,440 lines; 299 tests | §6.2 `single_graph_strain` default `auto`; §9.2; §3.1 (new `upstream_mace.py` row) |
| L. Security nits: YAML loader unspecified; subprocess transport unspecified | — | §10.5 `safe_load`; §8.7 and §9.4 restart write as transport |

Estimates unchanged (§16). Four risks added (§18).

**Draft 0.7 (17 October 2026)** carries results of the research engine from
16–17 October 2026 into the spec; no review. All were measured there first:

| Change | Source | Where |
| --- | --- | --- |
| MACE-OFF (five organic models) supported: generic rewrites, gate passes on the organic set, molecules intact in 5 ps at 300 K, 3.9–16.8× faster than stock ASE; static volumes 5.5–9.2% low for OFF23; OFF23-large memory limits; MACE-MDP out of scope | `studies/other_models/README.md` (MACE-OFF section), `results/mace-off-20261016/` | §4.3 table, known issues, licences; §12.5; §17 |
| Plain cuEq uses the fused convolution (MACE PR 1641 backport, default since 16 October 2026); model copies must re-apply MACE's fused-convolution adapter | `studies/further_optimizations/2026-10-16-upstream-mace-survey.md` (item 2) | §3.1; §3.2; §9.3 backend and `reproducible=True` |
| No kernel file locks on network filesystems: soft `filelock`, CUDA driver cache off on NFS, lock mode in provenance | `docs/NON_LOCKING_DESIGN.md` (coraline and leto broke; the driver cache took all but one lock per run) | §3.1; §3.2; §13; §14 caches; §18 |
| OpenEquivariance's symmetric contraction rejected (27–79× slower) | `studies/further_optimizations/2026-10-16-oeq-symmetric-contraction.md` | §6.2 |

Estimates unchanged (§16).

**Draft 0.8 (17 October 2026)** answers a sixth review, of draft 0.7, that
checked it against the tree, the MACE-OFF checkpoints and the 16 October
results. The review found the core contracts consistent (float64 energy and
gradient, transactions and the derived-state registry, canonical restart
points, the three-pass D3, the BAOAB drag correction, the MC acceptance
algebra) and twenty gaps. All twenty were verified and adopted; the seven
design items are owner decisions 8–14 (§18).

| Finding | Check | Where answered |
| --- | --- | --- |
| 1. MACE-OFF uses all-electron E0, missing from §4.3's list | E0 table of all five checkpoints (C −1,030.57 … Br −70,045.28 eV); gates: float32 energy 0.1–12.2 meV/atom (production), 0.08–11.4 (stock) | §4.3 energy scale; §6.2 `float64_energy`; §12.1; §16 phase 1b; §18 |
| 2. MACE-OFF is molecular, tiers 1–3 need 3D PBC | `gpu_langevin.py:1679` refuses `not all(atoms.pbc)` | §4.3 known issues |
| 3. Reproducible mode did not exclude the fused convolution | §9.3 text; the deterministic flag does not govern cuEq | §9.3 enforcement (decision 13) |
| 4. `run` never checked the schedule against `max_temperature_K` | §7.6 checks | §7.6 (decision 9) |
| 5. `min_mc_temperature_K` optional but needed | §9.2 type; §12.3; §7.6 | §9.2 (decision 8) |
| 6. `species_changes` and `mc_moves` could disagree | §9.3 rule 8 vs §10.4, §7.6 | §9.2 validation |
| 7. `Simulation(atoms, model)` could not declare a superset | §7.6, §8.5 | §7.2; §7.6 (decision 11) |
| 8. ForceField/plan lifecycle undefined | §7.1 `evaluate` before §7.2 | §7.6 (decision 12) |
| 9. Superset not checked against the model's elements | §8.5 | §8.5 |
| 10. MC with `FixAtoms` undefined | `gpu_langevin.py:2136` raises on constraints | §4.4; rule 1 (decision 10) |
| 11. Reproducible process-wide state not enforced | `CUBLAS_WORKSPACE_CONFIG`, deterministic flag, dynamo cache limit are process-wide | §8.7 (decision 14) |
| 12. `upstream_mace` switches are module-level patches | §5 forbids module-level switches | §8.7 MACE patches (per-model in FastMACE) |
| 13. `streaming_conv` still listed edge-counter binding | §8.1 encapsulates the counter | §6.2 |
| 14. Re-capture on a changed superset and a "planned rebuild" had no mechanism | rule 2 refuses; `Workload` has no allow field | §8.1; §8.5 |
| 15. Thermostat targets listed as graph-safe events | §8.5: device-scalar fills | §8.1 |
| 16. The example ran the whole schedule, then 200,000 more steps of it | §7.2 | §7.2 alternatives |
| 17. `[d3]` "until native D3 lands" | §11.2, §16 put native D3 in 1.0 | §4.1; §14 |
| 18. Stale counts and hardware | 2,607 and 1,499 lines; 329 tests collected; leto and murphy qualified; 19 model labels | §3.1; §4.2; §12.2; §16 |
| 19. §18's all-electron risk row omitted MACE-OFF | E0 table | §18 |
| 20. The memory note predates the fused plain default | the 16 October benchmark ran the unfused plain path | §4.3 known issues |

Estimates unchanged (§16).

**Draft 0.9 (17 October 2026)** answers a seventh review, of draft 0.8. It
confirmed the twenty draft 0.8 findings closed (counts, hardware, 19 model
labels, the MACE-OFF gate ranges and E0 table) and found one scientific and
seven API or wording issues. All eight were verified and adopted; no owner
decision was needed.

| Finding | Check | Where answered |
| --- | --- | --- |
| 1. MACE-OFF's iodine E0 is not all-electron | I −8,102.52 eV ≈ −298 Ha, a def2 effective-core valence energy; all-electron I ≈ −7,000 Ha; Br −70,045 eV is all-electron | §4.3 energy scale (all-electron and pseudopotential references; Nb, Ta and I pseudopotential); §18 |
| 2. The §7.2 example ran the whole schedule, then 100,000 more MC steps | §7.2 rule: `steps` beyond the remaining schedule raises | §7.2: paused 200,000-step run, MC run inside the schedule, then the rest; a finished schedule needs a new `schedule=` |
| 3. `Workload` example omitted four fields and §9.2 showed no defaults | §7.2 versus §9.2 | §9.2 defaults; §7.2 comment |
| 4. `Workload` construction raised `PlanRefused`, the §9.3 aggregate | §7.6 definition; the same mismatch held for `run`'s schedule and MC checks | §9.2: `ValueError`; §7.6 definition widened to `run` requests the plan does not cover |
| 5. `SpeciesNotInSuperset` used for an element the model lacks | §8.5 versus §7.6 | §7.6 `ModelElementError`; §8.5 |
| 6. "Same `ForceField`, in any order" ignored the process constraints | §8.7 one backend family, reproducible or not | §7.6 |
| 7. The backend dimension said plain cuEq is always fused | §9.3 enforcement uses the unfused form | §9.3 |
| 8. The MACE-OFF non-PBC note covered only `fm.Simulation` | `FastMACECalculator` is compiled-tier, no `allow_reference` | §4.3; §7.6 ASE front end |
| Also: the `species=` comment of §7.2 sat on the `ForceField` argument | §7.6 `Simulation(atoms, model, species=…)` | §7.2 alternative line |

Estimates unchanged (§16).

**Draft 0.10 (17 October 2026)** answers an eighth review, of draft 0.9. It
confirmed the eight draft 0.9 findings closed (iodine −297.75 Ha against
about −6,910 Ha all-electron; the paused example; the API fixes) and found
three wording and exception-contract issues and one note. All were verified
and adopted.

| Finding | Check | Where answered |
| --- | --- | --- |
| 1. "Light" and "heavy" elements misclassified Br (all-electron) and I (pseudopotential) | E0 table; def2 is all-electron up to Kr and uses effective core potentials from Rb | §4.3 (classified by the E0 values: all-electron and pseudopotential references); §18; §19 draft 0.9 row |
| 2. `PlanRefused` and `SpeciesNotInSuperset` overlapped on rule 2 | §7.6 definitions; §9.3 rule 2; §7.6 Simulation inputs | §7.6: `SpeciesNotInSuperset` subclasses `PlanRefused` and is raised, with every reason, when rule 2 is among them |
| 3. The third `run` of §7.2 relied on the MC block persisting | §7.6 said only "attached to the simulation" | §7.6: `mc=` stays attached until another `mc=` or `mc=None`; restart state (§10.6); §7.2 comments |
| Note: `FastMACECalculator` raised `PlanRefused` without a plan | §7.4, §7.6 | §7.6: it builds a minimal plan through the same planner |

Estimates unchanged (§16).

**Draft 0.11 (17 October 2026)** adds the adoption layer (§7.7), from a
proposal reviewed against draft 0.10. Its premise holds: ASE calls a
calculator once per step, so only a layer that owns the MD loop reaches the
graph engine. Adopted: `fm.md`, ASE-compatible integrator classes, `fm.calc`,
the run report, the error-message shape, the CLI additions, `fastmace[full]`
and the documentation pages. Changed on review:

| Proposal item | Problem | Adopted form |
| --- | --- | --- |
| `min_mc_temperature_K` = schedule minimum | an anneal from 0 K gives a zero ΔE floor (§12.3) | §7.7: required with a schedule and `mc=`; MC attached only where the target is at or above it |
| `Langevin(..., friction=0.02)` in the ASE shim | ASE's friction is per ASE time unit (≈ 0.002/fs) | §7.7: ASE signatures and units unchanged; example uses `0.02 / units.fs` |
| `FixAtoms` "forces the compiled tier" | constraints belong to the integrators and are refused until phase 4 (§4.4) | §7.7: refused with that reason |
| Callbacks "run between graph chunks" | ASE observers read and may change `atoms` | §7.7: state synced before, results cache, changes applied as §8.5 events or refused |
| "Expected speed-up" from a 50-step calibration; "≈ 0.6×" and "≈ 1×" in messages | per-GPU tables are forbidden (§9.1); calibration is phase 5 | §7.7: measured step time of the run only; ratios only from `fastmace bench --vs-ase`; reasons without estimates |
| `fastmace[full]` in phases 1a/1b without a compiler | prebuilt OEq wheels are phase 6 (§14) | §14: union extra; `doctor` reports the missing toolchain until then |
| Catch exceptions and print a message | scripts must fail visibly and catch errors | §7.7: exceptions keep propagating with a fixed message shape; only the CLI prints without traceback |
| "Gate on by default" | open decision 3 | `fm.md` follows `Simulation`'s default (§12.3); decision 3 stays open |
| High-level path "within 3%" of the expert path | the layer is thin; a gap is a bug | §7.7: same plan and gate, same step time within noise |
| ASE class names (`NoseHoover`, `MTK`, `Berendsen`) | ASE 3.27 names differ | §7.7 mapping to the ASE 3.27 classes |

Estimates: phase 1b 4–6 weeks (was 3–4), total 25–34 weeks, first
production deliverable 19–25 weeks (§16).

**Draft 0.12 (17 October 2026)** answers a review of draft 0.11, which found
the adoption layer's shape and phase 1b placement right and six details
open, two of them physics-level. All were verified and adopted; the MH-1
licence was checked against the ACEsuit `mace-foundations` model table.

| Finding | Check | Where answered |
| --- | --- | --- |
| 1. `dispersion="d3bj"` did not identify the correction | §11.2 parameter table; the research engine's label table already records each label's training functional (`common.py` `MODELS`) | §7.7: `"d3bj"` expands to the full `fm.D3` record from the label table's functional and is recorded and printed; refused where the reference already contains dispersion (MACE-OFF ωB97M-D3(BJ), OMOL-0 and MH-1 `omol` ωB97M-V) or the functional is unknown (POLAR-1, paths); explicit `fm.D3(...)` otherwise. The review's "OMOL → ωB97M D3" would have double-counted VV10 |
| 2. ASE shim signature and outputs | ASE 3.27 constructors include `fixcm`, `rng`, `communicator`; no report or restart access | §7.7: explicit argument lists checked against `inspect.signature`, `rng` and non-default `communicator` refused; `summary`, `plan`, `gate`, `metadata`, `simulation`, `output`; ASE-format trajectory and log at the given paths, everything else in `output=`, resume through `Simulation.resume` |
| 3. Observer sync missed masses, charges and PBC | §8.5 events include mass and charge changes | §7.7: masses and charges synced and applied as events; PBC, charge/spin `info` keys raise |
| 4. `fastmace md` lacked most `fm.md` arguments | §7.5 | §7.5: flags for head, charge, spin, pressure, cell mode, timestep, dispersion, seed, check, trust; schedules, MC and integrators through `init` + `run` |
| 5. MH-1 example without a head | MH-1 is multi-head | §7.6 loading and §7.7: `head=` required for multi-head checkpoints, no default; examples pass it |
| 6. `temperature_K=0` with `mc=` gave a zero ΔE floor | §12.3 floor 0.01 k_B T_min | §7.7 and §9.2: `min_mc_temperature_K` must be positive |
| Report: NVT example warned about NPT; MH-1 licence | `mace-foundations` table: MH-0/1 ASL | §7.7 conditional warnings, licence from the label table; §4.3 licence list (MP-0, MPA-0 MIT; OMAT-0, MatPES, OMOL-0, MH-1 ASL) |
| Report: "printed at setup" versus "filled in once measured" | §7.7 | §7.7: two-stage report, `speed: pending` at setup, one line when measured, whole-run mean at the end |

Estimates unchanged (§16).

**Draft 0.12, amendment (17 October 2026):** owner decision 4 recorded,
licence MIT (§14, §18). No other change.

**Draft 0.12, second amendment (17 October 2026):** owner decision 1
changed: FastMACE moved from `ResearchModule/FastMACE/` to its own
repository `FZWINGEL-UBT/FastMACE` on an NVMe drive (§18). No content change.

**Draft 0.12, third amendment (17 October 2026):** §14 (b) states which
driver cache directory the NFS rule checks (task 0-T03). No other change.

**Draft 0.12, fourth amendment (18 October 2026):** §18 risk table: plain cuEq float64
is not bitwise reproducible (1a-T13). No other change.

**Draft 0.12, fifth amendment (19 October 2026):** owner decisions 3 (gate opt-in,
§12.3) and 15–21 (the phase 1b plan's D2–D8) recorded in §18. No other change.

**Draft 0.12, sixth amendment (19 October 2026, conductor, from the phase 1b
prestart reports):** §7.7 `fm.md` friction default stated in ASE units as in
`main.py` (0.02 per ASE time unit, not 0.02/fs); §7.3 energy: the float64
energy may be a separate reduction of the float32 terms while forces come
from the float32 autograd (equivalent, 1b-T03); §7.7 ASE shim also refuses
`comm` and `master`. Listed for the owner's review in `plans/phase-1b.md`.

**Draft 0.12, seventh amendment (19 October 2026):** owner decisions 22–28
(the phase 1b review list) recorded in §18; §10.6 and the restart text:
a version change alone warns instead of refusing (decision 23).

**Draft 0.12, eighth amendment (19 October 2026):** owner decisions 2 and 5
decided, and 29–30 added (the phase 2 plan's D1–D4), in §18; §9.4 states the
automatic default and the `fallback="refuse"` opt-out.

**Draft 0.12, ninth amendment (20 October 2026, conductor, from 2-T05):**
§12.3 "Resources" now says three things:
- GPU reference steps run in child processes on CUDA plans;
- CPU reference children run under a hard memory cap;
- the replay check under `passed-reduced` uses the gate's own limit.

**Draft 0.12, tenth amendment (20 October 2026, owner decisions, from 2-T14e):**
§12.2 "ASE equivalence", Langevin: the run is 2 ps (heat and cool 1 ps each,
4,000 steps) instead of 9 ps, and the per-bin limits are 1e-6 K and 1e-7
meV/atom instead of 0.1 K and 0.01 meV/atom. The two runs share ASE's noise
stream and are the same trajectory to rounding (1b-T09: 3.5e-10 K and
3.8e-11 meV/atom), so length adds no power and the limits do the catching
(`docs/validation/phases/2-T14e-reference-reuse.md`). The BAOAB pairs share the
energy limit, now 1e-7 meV/atom (1b-T09 measured 1.1e-11); their other
criteria are unchanged.

**Draft 0.12, eleventh amendment (20 October 2026, owner decision):** GPU CI
is deferred ("for now"). Phase 2's acceptance no longer includes "CI on two
GPU architectures" (§16). The workflow from 2-T18 (`.github/workflows/gpu.yml`)
stays in the repository, but it runs only by hand until the runners are
registered: its push and nightly triggers are off. Until then, GPU coverage
comes from the conductor's trial tiers before merges and from each phase's
acceptance runs. The question comes back when the owner registers the
runners, at the latest in phase 6.

**Draft 0.12, twelfth amendment (20–21 October 2026, owner decisions,
optimisation round 1):** two new default rewrites in §6.2.
- **`gated_rows`:** removes the layout copies around the per-irrep block
  linears. It is exact: bitwise on the CPU and in the reproducible GPU mode.
  It is on wherever the merged residual applies.
- **`segment_gemm`:** a Triton GEMM that writes into the joined layout, chosen
  by a shape rule from M, N, K and the SM count. It is rounding class, and
  on in float32, non-reproducible plans. Float64 and reproducible plans
  resolve it to off.

Measured on murphy (sm_89) and the RTX 5060 Ti (sm_120): A gives NaCl −3.7
to −4.4 % and NbCl5 −9.0 to −10.6 %. B on top gives NbCl5 −4.2 to −5.2 %, and
NaCl is neutral. Both are FastMACE-only defaults, outside the research-engine
plan equality. The combined acceptance's 19-label gate decides per model; a
model that fails gets the rewrite switched off. Also added:
`EngineConfig.model_template_cache` (OPT-T06, on: later plans copy the
converted model). Rejected in the same round and recorded in
`docs/optimisation-ledger.md`:
- FP32-accurate tensor cores (OPT-T01) and their overlap (OPT-T01b);
- OPT-T03's streams;
- OPT-T02 stage C.

**Draft 0.12, thirteenth amendment (21 October 2026, owner decision, "Make all
3 defaults"; OPT-DEF-2):** three more FastMACE-only defaults in §6.2 and §8.
- **The OPT-T08 records** (`species_tables`, `fold_product_linear`,
  `fold_readout_tail`): the species-only atom-level products as per-species
  tables, and two linear folds. They are on where the model takes them
  (`species_tables` and `fold_product_linear`: the MH-1 labels;
  `fold_readout_tail`: also OMOL-0), in float32 plans that are not
  reproducible. All three are rounding class. `species_tables` is bitwise on the
  CPU but not on the GPU (8e-16 eV/Å in the forces in the reproducible mode),
  so float64 and reproducible plans resolve all three to off. Same-process A/B
  of the three together, both orders: RTX 5060 Ti NaCl −9.2 to −9.6 %, NbCl5
  −7.2 to −7.6 %; L4 NaCl −8.0 to −8.2 %, NbCl5 −7.8 to −8.0 %; A100 NaCl −9.3 to
  −10.4 %, NbCl5 −9.3 %. The force gate passed on the RTX 5060 Ti and the L4.
- **`EngineConfig.overlap_rebuild`:** a pipelined fixed-cell step's early
  neighbour rebuild runs beside the force graph. Exact (bitwise in the
  reproducible GPU mode). On by default: NaCl −0.4 to −3.6 % (more at high
  temperature), NbCl5 neutral. `step_graph` stays off.
- **`folded_backward_tiles`:** the folded radial layer's input gradient tiled
  by a waves rule. Exact (same reduction order). On wherever the fused folded
  backward is: NaCl −1.7 to −1.9 % on the RTX 5060 Ti, neutral on the L4.

All five stay outside the research-engine plan equality, each with its own
scope.

**Draft 0.12, fourteenth amendment (21 October 2026, owner decision, "Yes";
optimisation round 3, OPT3-T03):** one more FastMACE-only engine default.
- **`EngineConfig.fused_integrator`:** on a fixed-cell CUDA Langevin engine
  the AC/ACP stage (half kick, drift, neighbour checks, compaction, saved
  velocities and flags) runs as four Triton kernels around PyTorch's
  centre-of-mass sums and cumsum. It reads the same host-drawn noise and
  performs the same float64 operations in the same order, so it is exact:
  every written buffer is byte-identical at stage level, and reproducible-mode
  trajectories are bitwise equal with it off (pipelined, checked, resumed).
  It therefore stays on in reproducible plans. NPT (`LangevinBAOAB`), the CPU
  and installations without Triton keep the PyTorch stage, and
  `info()["fused_integrator"]` gives the reason. Same-process A/B, 300 K,
  both orders: RTX 5060 Ti NaCl −2.4 %, NbCl5 −0.3 to −0.7 %; L4 NaCl −1.9 to
  −2.0 %, NbCl5 neutral.

The rest of round 3 changed no default: OPT3-T01 (fused gate VJP) and
OPT3-T02 (symmetric contraction) were screened and not built, OPT3-T04's
`combo_kernels` was rejected (NbCl5 slower), and OPT3-T05's layout records
stay opt-in (no gain above noise). Details in `plans/opt-3.md` and
`docs/optimisation-ledger.md`.

**Draft 0.12, fifteenth amendment (24 October 2026, owner decision 57, "Go
ahead and do it as you recommended"; optimisation round 4, OPT4-T04):** one
more FastMACE-only rewrite default, scoped to a model family.
- **`RewriteConfig.grouped_launch`:** on with `block_linear` in float32,
  non-reproducible plans for the generic MACE family (every interaction
  MP-0's residual or the density family's, not PolarMACE). Stage 2 turns it
  off for MH-1, MACE-OFF, POLAR-1 and OMOL-0 with the reason "not the generic
  MACE family"; an explicit `on` keeps it for any family. Rounding class,
  deterministic; the record itself is unchanged since OPT-T03b. NaCl −4.5 to
  −5.4 % on the RTX 5060 Ti (five generic labels, both orders), −4.1 to
  −4.3 % on the L4 (OMAT-0), NbCl5 neutral. §6.2's table has its row.

**Draft 0.12, sixteenth amendment (24 October 2026, owner decision 58,
"Yes"; optimisation round 4, OPT4-T02):** two FastMACE-only rewrite defaults,
scoped to qualified checkpoints of one model family.
- **`RewriteConfig.fx_e3nn_linear`:** a TorchScript e3nn `o3.Linear` (as a
  checkpoint restores it) is a graph break, and in a loop it made dynamo run
  PolarMACE's whole forward eagerly. The record gives each such Linear e3nn's
  own fx graph back. Rounding class (eager bitwise on the CPU), deterministic.
- **`RewriteConfig.polar_graph_sums`:** PolarMACE's one-graph per-graph sums
  (Fukui norms, total charges, barycentre) as a sum and a broadcast, without
  `scatter_add_` atomics and the serial float64 `index_put` backward.
  Rounding class, deterministic.
- **Default:** both on for POLAR-1-S, POLAR-1-M and, after its local NbCl5
  2×2×2 gate passed (0.88 / 0.96 / 0.75), POLAR-1-L (the label table's
  `polar1-s`, `polar1-m`, `polar1-l`, identified by label or SHA-256) in hybrid float32,
  non-reproducible plans; stage 2 turns them off for any other model with the
  reason "not a qualified POLAR-1 checkpoint"; float64 and reproducible plans
  keep MACE's code; an explicit `on` or `off` overrides. POLAR-1-S A/B: NaCl
  −24 %, NbCl5 −18 % (RTX 5060 Ti); −19 %, −16 % (L4). §6.2's table has
  their rows.
**Draft 0.12, seventeenth amendment (24 October 2026, owner decision 59, "Yes";
optimisation round 4, OPT4-T05):** one engine default for setup time.
- **`EngineConfig.dynamic_edges`:** on by default on graph-tier MD plans as
  well as on evaluation and compiled-tier plans. The first compiled call marks
  the edge dimension dynamic, and the padded first capture of B reuses its graph
  instead of compiling the model again. Exact: the same compiled code per
  step, and bitwise-equal reproducible-mode trajectories. Warm starts are
  7–9 s shorter and cold starts 16–22 s shorter (local). §8.1 says so. Plans
  that compile nothing record it off (`dynamic-edges-uncompiled`); an explicit
  `False` is recorded as `dynamic-edges-graph-off`.
- **Compiler caches under `FASTMACE_CACHE` (§14, conformance, owner "Yes"):**
  `FixedPlanner.configure_process` now sets `TORCHINDUCTOR_CACHE_DIR` and
  `TRITON_CACHE_DIR` under the cache root when the environment names none.
  Before, nothing called the 1a-T04 helper, so Inductor wrote to
  `$TMPDIR/torchinductor_$USER`, which the workstation rule forbids.

**Release follow-up 6-T15 (25 October 2026, owner: "Go ahead and add the wheel then"):**
extend the prebuilt OEq host-extension wheel matrix to CPython 3.13 and update
the `[oeq]` and `[full]` dependency markers to include it. The torch 2.14.1+cu130,
OpenEquivariance 0.7.0, Linux x86_64 and manylinux_2_28 constraints stay unchanged.
Acceptance is a fresh Python 3.13 installation with the prebuilt extension,
hybrid NaCl MD and no g++/Ninja/nvcc invocation. A C compiler for Triton's
launcher remains a requirement. Results: `docs/validation/phases/6-T15.md`.

**Optimisation round 6 (27 October 2026, owner decisions 60 and 61; `plans/opt-6.md`):**
- **Decision 60 (owner: "yes"), `RewriteConfig.chunked_conv_backward` on by
  default** wherever `streaming_conv` and `fused_folded_backward` are (planner
  rule `on_with_streaming_conv_and_fused_folded_backward`). Exact class. The op
  keeps cells whose per-edge weights fit `UNCHUNKED_L2_MULTIPLE` (4) L2s on the
  unchunked backward. Same-process A/B on the RTX 5060 Ti: NbCl5 0.973 / 0.967,
  NaNbO3 0.969 / 0.964, NaCl 1.008 / 1.009 (`docs/validation/speed/opt6-T02.md`).
- **Decision 61 (owner: "think about a good class"), the `certified` exactness
  class** (§6.1): an approximation with a setup-time certificate that its float32
  error is at most that of the float32 computation it replaces, values and first
  derivatives, against float64 of the same weights; float32 plans only; the
  self-check holds the certificate. First record: `radial_tables` (opt-in,
  `never_automatic`; refused in float64 plans).

**Rule 12 per side, per stage and probed near the limit (1 November 2026, owner: make the memory estimation more
accurate; conductor):** §9.1/§9.3 rule 12 and §13 amended. The rescue records' fractions apply to their own side
(per-edge or per-atom part of the remaining activations), and a per-stage allocator reserve covers what PyTorch
reserves beyond the live peak (no crystal key). `EngineConfig.memory_probe` ("auto" = near-limit for MD with CUDA
graphs, else off; "near-limit", "always", "off"): when the chosen stage or the next-cheaper one is within the
stage band of the free memory, rule 12 measures it in a child on a 15-20k-atom replica of the structure's repeat
unit and re-plans with the measured scale plus 5 % headroom. The ladder and the probe are recorded in the plan,
the report and `metadata.json` (§13 keys `plan.memory_ladder`, `plan.memory_probe`); resumes reuse the scales and
never probe. See `docs/validation/capacity/estimate-band.md` and `docs/validation/capacity/estimate-probe.md`.

**Generic radial tables on by default (2 November 2026, owner: "Yes make it default"; conductor):** §6.2 amended. `generic_radial_tables` is default on for the generic family on the RTX 5060 Ti and the RTX 4000 Ada (decision 66, same-process A/Bs in both orders on both GPUs), float32 only and not in reproducible plans; float64 models resolve "auto" to off, an explicit "on" there is refused. Off on every other GPU (the A100 was not measured). See `docs/validation/omat0/generic-radial-tables.md`.

**Generic radial tables on the A100 80GB cards (3 November 2026, owner: "Yes go ahead"; conductor):** decision 66's list for `generic_radial_tables` adds "NVIDIA A100 80GB PCIe" (SrTiO3 5,000 atoms, MD step b/a 0.902 / 0.901) and "NVIDIA A100-SXM4-80GB" (NaCl 8,000 atoms, 0.917 / 0.917), one structure each, after MargrafGroup's Benchmark_loup ran FastMACE on the PCIe card without them. `generic_chunked_conv_backward` stays off there (1.12 with the tables). See `docs/validation/omat0/loup-a100.md`.

**The tensor-core fused forward on the A100s, and the A100 80GB PCIe in decision 66 (4 November 2026, owner: "Yes
to both"; conductor):** the fused folded forward's a.W product runs in 3xTF32 on tensor cores
(`fused_forward.TENSOR_CORES`, now None = per GPU) on "NVIDIA A100-SXM4-80GB" and "NVIDIA A100 80GB PCIe"
(`TENSOR_CORES_GPUS`), for every model, with the split `TC_SPLIT = "mask"` (the high part truncated to TF32, the low
part the exact remainder passed raw). A precision change of the w product: about 2.5x the rounded split's error,
within 2 chain errors + (3 ceil(k/8) + 52) units of 2^-24 of the row scale |a| |W| (the GPU test). MD step b/a on the
A100-SXM4-80GB, both orders: OMAT-0 medium on Benchmark_loup 0.933 / 0.933; MH-1 NaCl 0.978 / 0.970, NaNbO3 0.966 /
0.967, NbCl5 0.983 / 0.982. Benchmark_loup 3.17 -> 2.92 (NVE) and 3.12 -> 2.90 µs/atom/step (Langevin). Force gates
pass (OMAT-0: SrTiO3, NaCl, NbCl5; MH-1: NaCl, NaNbO3, NbCl5; RTX 5060 Ti); OMAT-0's SrTiO3 energy error 1.39 meV
against 0.73 (floor 10 meV), MH-1's energy errors below the float32 calculator's. Slower on the RTX 5060 Ti and the
RTX 4000 Ada, which keep the FP32-SIMT product. The A100 80GB PCIe joins the SXM4's OMAT-0 switches:
`tf32x3_gemms` for OMAT-0 medium only (`DECISION_66_TF32X3_OMAT0_GPUS`; MH-1 not measured there), the generic
wprime set (`DECISION_66_GENERIC_WPRIME_GPUS`, with chunked_conv_backward as its carrier) and
`SKIP_ZERO_COLUMN_GPUS`: Benchmark_loup there 4.61 -> 3.31 µs/atom/step. See
`results/omat0-kernel-lab-20261103/README.md` sections 9, 11 and 13.

**Factored wprime backward and mirror species pairs per GPU (4 November 2026, owner: "Turn them on and investigate the failure"; conductor):** decision 66 lists for `wprime_backward.FACTORED` (`FACTORED_GPUS`) and `radial.SYMMETRIC_PAIRS` (`SYMMETRIC_PAIRS_GPUS`): "NVIDIA RTX 4000 Ada Generation" and every A100. FACTORED regroups the wprime backward's trilinear sums (rounding), SYMMETRIC_PAIRS lets mirror species pairs read one table row (bitwise). Benchmark_loup MD step b/a, both orders: FACTORED Ada 0.983 / 0.980, A100-SXM4-80GB 0.938 / 0.940; SYMMETRIC_PAIRS on top Ada 0.996 / 0.995, A100 0.998 / 0.990. The RTX 5060 Ti was not measured and keeps both off. Force gates pass on the Ada and, with the new defaults, on the A100 for OMAT-0 SrTiO3 and NaCl and MH-1 NaCl, NaNbO3 and NbCl5. The A100 default plan fails OMAT-0 NbCl5 stress with or without them (1.8e-7 eV/Å³ against 1.2–1.5e-7); a bisection traces it to `tf32x3_gemms` and the tensor-core forward together, each alone passing. The owner accepted the excess and kept both (5 November 2026, "keep it its fine"): an accepted exception of the A100 gate, not a reason to switch either off. See `docs/validation/omat0/omat0-tp-backward.md`.

**The C3 fold for the generic family (5 November 2026, owner: "Yes, port it", then "go ahead"; conductor):** §6.2 and decision 66 amended. Record `generic_fold_product_linear` (field `fold_product_linear`, rounding) folds `products.0.linear` into interaction 1's `linear_up`, `SpeciesSkip` and readout 0's linear on generic models whose first product has no skip term (the density family, MACE-OMAT-0); it needs `block_linear` and `generic_species_skip`. "auto" is on only on `DECISION_66_GENERIC_FOLD_PRODUCT_GPUS`: "NVIDIA RTX 4000 Ada Generation" and every A100. OMAT-0 medium MD step b/a, both orders: Ada 0.9986 / 0.9960 (SrTiO3 5,000 atoms), 0.9911 / 0.9849 (NbCl5 1,944 atoms); A100-SXM4-80GB 0.9928 / 0.9931 and 0.9805 / 0.9828. The RTX 5060 Ti was not measured. See `docs/validation/omat0/generic-fold-product.md`.

**POLAR-1 under decision 66, and a plan-time refusal for one-interaction models (5 November 2026, owner: "yes turn it on for 2 interaction models", "fix it"; worker polar-T01, for the conductor's review):** decision 66 amended. `wprime_backward` with `fused_folded_forward` is "auto" on for the POLAR-1 checkpoints with a second interaction (POLAR-1-M and -L) on `planner.fixed.POLAR_DECISION_66_WPRIME_GPUS`, and `gate_epilogue` for every POLAR-1 checkpoint on `POLAR_DECISION_66_GATE_EPILOGUE_GPUS`: "NVIDIA RTX 4000 Ada Generation" only, the GPU measured (leto, same-process MD step b/a, both orders, NaCl and NbCl5 3×3×3). POLAR-1-M with all three: NaCl 0.966 / 0.966, NbCl5 0.957 / 0.958, force gate passed; POLAR-1-S `gate_epilogue`: NaCl 0.998 / 1.001, NbCl5 0.994 / 0.993. `fused_folded_forward`, `wprime_backward` and `w_tables` act only on the interactions after the reassociated first one; a model with one interaction (POLAR-1-S) and `reassociate_first` on now has them off with that reason ("auto") or refused at plan time (explicit "on", code `rewrite-unsupported-by-model`) instead of failing at build. `species_tables` now also takes PolarMACE (its forward pinned by the polar path's hash in `rewrites.species`; the embedding feeds only the first interaction), so it is on where PolarMACE takes it, as for MH-1 (float32, not reproducible; the whole tiny PolarMACE within float64 rounding); `fold_readout_tail` and `fold_product_linear` stay refused for PolarMACE, whose long-range heads read every layer's features. EDGE_SH can be applied by MH-1's streaming record too (`rewrites.generic.modules.EDGE_SH_MH1_GPUS`, empty: no GPU measured, so off by default). See `docs/validation/polar/polar-reuse.md`.

**Radial tables of degree 5 and the skipped zero column (4 November 2026, owner: "Yes"; conductor):** the float32
radial tables (MH-1's and the generic family's) are fitted with Chebyshev degree 5 instead of 10
(`rewrites.radial.DEGREE`), at the same tolerance and certificate (at most 1; more pieces). MD step against degree
10: OMAT-0 medium on Benchmark_loup 0.903 / 0.905 (A100-SXM4-80GB), 0.908 (RTX 4000 Ada), 0.935 (RTX 5060 Ti);
MH-1 on the Ada neutral (NaCl 0.978, NaNbO3 and NbCl5 within 0.6 %); the force gate passes on SrTiO3, NaCl and
NbCl5. Float64 tables keep `DEGREE_64`. The fused folded forward's k-loop skips the folded weight's all-zero last
column (`fused_forward.SKIP_ZERO_COLUMN`, exact) on "NVIDIA RTX 4000 Ada Generation" and "NVIDIA A100-SXM4-80GB"
(decision 66; 0.981 and 0.986 on top of degree 5; the RTX 5060 Ti 1.000 keeps it off). Benchmark_loup on the
A100-SXM4-80GB: 3.17 µs/atom/step against 3.61 (NVE) and 3.54 (Langevin). See
`results/omat0-kernel-lab-20261103/README.md`.

**Rule 12 for generic models (2 November 2026, owner: test the capacity on OMAT-0 medium and carry over the MH-1
changes; conductor):** §6.2, §9.3 rule 12 and decision 66 amended. Two generic records (§6.2):
`generic_chunked_conv_backward` (default on by decision 66 on the RTX 5060 Ti and RTX 4000 Ada) and
`generic_recompute_conv_weights` (off; float32 only). Rule 12 declares their per-edge terms (the chunked path
drops `weight_numel − (P + 1)` floats per edge; the recompute a further `weight_numel`) and places the recompute
in the rescue ladder after MH-1's `recompute_conv_weights`, with its own allocator reserve. MH-1's radial records
apply only to MH-1's class. Calibrated on SrTiO3 and NaCl (RTX A4000), NaNbO3 held out (RTX 4000 Ada):
estimate/reserved 1.00-1.11. OMAT-0 medium limits, default plan, before → after: RTX 4000 Ada SrTiO3 16,875 →
69,120, NaCl 39,304 → 125,000; RTX 5060 Ti SrTiO3 13,720 → 53,240, NaCl about 27,000 → 85,184. See
`docs/validation/omat0/omat0-capacity.md`. The host batch no longer runs MACE's own neighbour search (its edges were
replaced unread; every force field, batches otherwise bitwise MACE's own), which lets the tiled path reach 57.0M
OMAT-0 SrTiO3 atoms on 20 GB with 251 GiB of host memory (`docs/validation/tiled/tiled-omat.md`).

**Decision 67 (29 October 2026, owner: "It should only do this if memory would not be enough"):** rule 12
(§9.3) turns off the streaming convolution and its dependants only when the plan would not fit with them on and
turning them off lowers the estimate; decision 51 amended to match. MH-1's fused-path rewrites declare their
measured and fitted per-edge savings, so the estimate predicts the capacity scan's peaks (1.0-1.4x, was
1.5-2.3x). See §18, decision 67.

**Decision 66 (29 October 2026, owner):** per-GPU defaults allowed. Goal 2 and §9.1 now permit a default that
names the GPU models where same-process A/Bs measured a whole-step speed-up (exact device names, as in decision
65); per-structure and shape-keyed tables stay forbidden. See §18, decision 66.

**Template ownership (30 October 2026, setup fix F3, conductor):** §7.1 amended. The converted-model templates
belong to the `LoadedModel`, not to each `ForceField`, so a new composition from the same `fm.load` result copies
the conversion instead of loading and converting again (about 3.1 s per composition after the first). Isolation
between plans is unchanged. See `docs/validation/capacity/setup-fixes.md`.

**Decision 66 counts every variant of a GPU family (7 November 2026, owner: "Ok change the decision to that if its tested on one variant of a gpu it counts for all gpus (unless the variants vary significantly in performance)"; conductor):** decision 66 amended. `kernels/gpu_models.py` gains `family()` and `lookup()`: a `GPUModels` entry, and a per-GPU dict entry (`radial_table.WARPS_GPUS`, `wprime_backward.MAX_REGISTERS_GPUS`), now covers every variant of its family. The families are the A100 (unchanged since 4 November) and the H200, which adds the H200 NVL (the same GH100 with 132 SMs and 141 GB of HBM3e; a lower power limit) to every H200 default. The H100 SXM, PCIe and NVL cards stay separate names, as their SM counts and memory differ significantly. No default changes on a measured GPU. The off-polar A100 40GB final checks were cancelled as redundant; the two already measured (MACE-OFF23 small and medium) stay as confirmation.

**Pure work removal goes on for every GPU (7 November 2026, owner: "Yes" to the rule proposed in the off-polar worker; conductor):** decision 66 amended. A switch that only deletes operations or memory traffic, with no new or replaced kernel, no change of tiling, launch shape or precision, and no more memory kept alive, defaults on for every GPU after one not-slower same-process MD-step A/B (both build orders) on any GPU, if it is exact or passes the force/stress gate. The reasoning: every per-GPU split so far was a trade (the fused Fourier transform recomputes cos/sin and was 1.08-1.40x slower on the A100; the chunked conv backward 3-4x slower on the RTX A4000; 128-row tiles lost 2 % on the A100; 3xTF32 wins only on the A100; the gate epilogue tied on the A100), while pure skips have not lost (the zero-column skip measured 1.000 on the RTX 5060 Ti and won elsewhere). Fusions, substitutions, retiling and precision changes stay per GPU family.

**The accuracy gate's ratio is 3 × float32 rounding (7 November 2026, owner: "Change the rule to 3x"; conductor):** §15's gate row and decision 66 amended. `planner.selfcheck.RATIO` (read by `tools/force_stress_gate.py` as `GATE_RATIO` and by the planner's rewrite self-checks) goes from 2.0 to 3.0. Prompted by the MACE-OFF A100 combination (stock first skip, OFF radial tables and radial 3xTF32 together, MD step 0.56-0.64 of the default on all five OFF models): it failed only stress cells at 2.04-2.50 × (OFF23-medium and OFF24-medium paracetamol, OFF23-large naphthalene and paracetamol; the default reaches 1.3-1.9 × there), at most 6e-7 eV/Å³, about 1 bar; forces stayed at 0.9-1.7 ×. Earlier verdicts stand.

**Spline radial weights on the H200 (7 November 2026, owner: "go ahead and build new production version with a695 spline and radial_table_forward and do a new indepth profile on it"; conductor, decision 66):** interaction 2's radial weights w and their slopes come from a per-model cubic Hermite spline (`rewrites/radial/spline.py`, the a695 knot grid with dense knots at short r, fitted in float64 at setup and certified against the float32 network over the whole domain, certificate 0.48), evaluated by a forward kernel in Symmetrix-XL's layout (one thread per receiver and channel, `kernels/oeq_streaming/spline_forward.py`) and by the w' backward; the radial table then serves only the density. Three switches, each on for the H200 family by decision 66: `SPLINE_W` (rounding; Benchmark_loup MD step b/a 0.879 / 0.878, 4.91 to 4.31 ms; force/stress gate passes on SrTiO3, NaCl and NbCl5), `R128` (one receiver per 128-thread block with a register cap of 6 blocks per SM; launch shape only, 0.984 / 0.985 on top) and `SPLINE_VECTOR` = 4 (float4 coefficient loads in the w' kernel, bitwise, 0.974 / 0.979 on top). The A100 was not measured and keeps the table path. See `results/spline-w-h200-20261107/README.md`.

**COMPOSE_SKIP on the H200 (7 November 2026, owner's Codex H200 ideas; conductor, decision 66):** interaction 1's projection, linear and species skip are composed per species and applied through `species_mm` (`COMPOSE_SKIP`, rounding). On for the H200 family: Benchmark_loup MD step b/a 0.9698 / 0.9738 on top of the spline defaults (4.12 to 4.00 ms), force/stress gate passes on SrTiO3, NaCl and NbCl5. Measured and left off: `STAGE_QUADS` 0.996 / 0.998, `PACKED_COEF` 1.007 / 1.008, `FACTOR_W` 0.997 / 1.000, `CURSOR32` 1.002 / 1.003, `SHARED_LOCATOR` 0.997 / 0.997. See `results/h200-compose-quads-20261107/README.md`.


**MACE-OFF and POLAR-1 brought to current speeds (5-8 November 2026, owner: "bring all mace off and mace polar models up to current speeds ... fastest speed on a100"; worker off-polar-alex, conductor, decision 66):** merged from `fastmace/off-polar-alex` (753a05be). Per GPU family: MACE-OFF's stock first block takes the species skip and generic radial tables (`STOCK_FIRST_SKIP`, stock-first tables, exact and certified) on the A100, the RTX 5060 Ti and the RTX 4000 Ada (Ada OFF MD step 0.59-0.75 of the default on all five OFF models; A100 0.56-0.64 with radial 3xTF32, which is on for OFF on the A100 only); `tf32x3_gemms` for every OFF and POLAR-1 checkpoint on the A100 (OFF23-small replaces grouped_launch, 0.966 / 0.947). Two pure-removal switches default on for every GPU (the 7 November rule): `SPIN_BATCHED` (POLAR, the spin channels in one batched pass; 5060 Ti S 0.885-0.951, M 0.949-0.977, L 0.966-0.994; A100 gate passes S/M/L) and `FOLD_UPDATE` (the field-update MLP folded at setup; A100 NaCl 0.965-0.972, NbCl5 1.000; Ada NaCl 0.966-0.977; 5060 Ti 0.95-1.002; gate passes at 3 × on S/M/L, Alex 4957296; `polar_uvu` reverts and self-checks every folded block). Measured and left off: `FUSED_FOURIER` (1.05-1.34 on the 5060 Ti, 1.08-1.40 on the A100), `SMALL_MATMULS` (ties or invalid runs), `SHARED_GEOMETRY`, `JOINT_BACKWARD`. `SPLIT_COLUMNS` (Codex R1, split radial gradient across output columns; OFF23-large water_216 0.951 / 0.953 on the 5060 Ti) is built with an empty GPU set pending its bitwise test. `gpu_models.dynamo_constant` keeps device-list checks from importing Triton. The CPU suite for this merge ran on murphy (5359 passed, 202 skipped) because leto was busy. See `results/off-polar-alex-20261106/` and `docs/validation/polar/polar-reuse.md`.

**The H200 fast path, Codex rounds 1-3 (8 November 2026, owner's Codex H200 ideas; worker sx-layout-lab, conductor, decision 66):** merged from `fastmace/h200-fastpath` (8fde8bc1) and `fastmace/h200-fold-tail` (8f2bb647). On for the H200 family, Benchmark_loup MD step b/a in both build orders: `COMPOSE_FAST` (COMPOSE_SKIP's composed weights bound once per table, the four degree products as one `species_mm_joined`; bitwise, 0.961 / 0.966), `CUEQ_PROJECTED` (cuEq's weight-projection einsum kept as a buffer; rounding, 0.990 / 0.995, gate passed), `GENERIC_FOLD_READOUT_TAIL` (the generic product's species skip composed with the readout linear; rounding, 0.986 / 0.990, 3 × gate passed, worst ratios force 0.968 and stress 1.043 against the limit 3), `STAGE_EDGES` (the spline forward stages its edges per 32-edge tile; 0.986 / 0.986, no local memory), `STAGE_SH` (the w′ cohort stages its edge features in shared memory; bitwise, 0.996 / 0.995) and `COUNT_PLAN` (the species plan by a counting sort; bitwise, 0.983 / 0.982). `LINEAR_DIV_SPLIT` measured 0.997 / 1.003 and is closed. Together the MD step goes from 4.018 to about 3.65 ms. The CPU suite ran on murphy (combined tree 8958994f, identical to the merge: 5379 passed, 202 skipped) because leto was busy. See `results/h200-fastpath-20261108/README.md`.

**POLAR head pruning, the fold self-check on the GPU and the k-space megakernel (8 November 2026, owner: "builds one megakernel for the local gpu for mace polar"; worker polar-megakernel, conductor, decision 66):** merged from `fastmace/polar-megakernel` (f5aa4e0d). Fix: `update_fold.selfcheck` (FOLD_UPDATE, on since the off-polar merge) failed on GPU builds of POLAR-1 with a CUDA/CPU device error inside cuEquivariance's `Linear`, whose float64 naive path keeps constants on the CPU; the check now runs both blocks on the CPU in float64, and the POLAR-1-S gate with the FOLD_UPDATE and SPIN_BATCHED defaults passes on the RTX 4000 Ada (NaCl, NbCl5). `PRUNE_DEAD_DEGREES` (Codex R5: head outputs of degrees no later block reads are not computed; pure removal) defaults on for every GPU: Ada NaCl 0.974 / 0.973, NbCl5 1.000 / 1.001; 5060 Ti L NaCl 0.979; gate passed. Built and off: `COMPOSE_HEADS` (P1; 5060 Ti NaCl S 0.906 / 0.909, M 0.876 / 0.872, NbCl5 ties; gates pass S/M and L NaCl; the decision waits for L), `PHASE_RECURRENCE` (neutral), and the k-space megakernel in CUDA (cubins for sm_80/89/90/120; `MEGAKERNEL_GPUS` empty until its MD A/Bs and gates finish; RTX A4000 kernel time 0.30 × dense at NbCl5, 0.70 × at NaCl) and in Triton (1.3-1.9 × slower). The CPU suite ran on murphy on the merge with main fefb3121 (5438 passed); the last commit's update-fold, polar, head-compose and layering tests pass on leto (97 passed).

**COMPOSE_HEADS on for every GPU (8 November 2026; worker polar-megakernel, conductor, decision 66 and the 7 November pure-removal rule):** POLAR's long-range head linears are composed at setup (P1, `rewrites/polar/head_compose.py`, with `head_compose.revert`; `uvu.revert` undoes it too), the same setup-time reassociation as FOLD_UPDATE, so a plain default. MD step b/a on the RTX 5060 Ti, both orders: NaCl S 0.906 / 0.909, M 0.876 / 0.872, L 0.883 / 0.879; NbCl5 S 1.001 / 1.005, M 0.997 / 1.000, L 1.002 / 0.998. The 3 × gate passes for S and M (NaCl, NbCl5), L NaCl (RTX A4000) and L NbCl5 (Modal A100-80GB, 2baa8634, polar_uvu on, ratios energy 0.86 / forces 0.58 / stress 1.01; the earlier 690d43d0 run measured MACE's forward after rule 13 re-planned polar_uvu away and is void). `PHASE_RECURRENCE` is rejected (a tie). Suite: murphy on 5176ad1d (5438 passed; the 2 failures were in the new revert test, fixed in 8915dca3), then the POLAR test files and test_layering (118 passed).

**SPLIT_COLUMNS on the RTX 5060 Ti (8 November 2026, owner's Codex OFF/POLAR ideas, R1; worker off-polar-alex, conductor, decision 66):** the radial last layer's gradient is split across output columns in 16-column tiles when the split grid fits one wave (`kernels/triton/skinny_linear.py`). Exact (`tests/gpu/test_folded_tiles_cuda.py`, bitwise). MD step b/a, water_216, both orders: MACE-OFF23-large 0.950 / 0.954; OFF23-small, -medium, OFF23b and OFF24 0.993-1.006 (their chunks exceed the one-wave limit of 576 rows on 36 SMs, so the split does not fire). On `SPLIT_COLUMNS_GPUS` = the RTX 5060 Ti only; the A100 (108 SMs, chunks up to 1728 rows) and the Ada are not measured.

**FOLD_UPDATE's self-check for float32 builds (8 November 2026; worker polar-megakernel, conductor):** fix. Since FOLD_UPDATE went on by default, float32 POLAR-1 builds failed `update_fold.selfcheck` (POLAR-1-S 5.6e-8, M 7.6e-7 against the float64 threshold 1e-10, because the fold constants are rounded to float32 once), so rule 13 re-planned `polar_uvu` away. The check now verifies the fold's algebra on a float64 rebuild of the block from the original's weights (with the uvu, dead-degree and head-composition rewrites the block carries) and requires each installed constant to be within 8 eps of its float64 value (POLAR-1-S cueq float32: 3.1e-15). A corrupted constant is still caught (`tests/cpu/test_polar_update_fold.py`). The POLAR, layering, step-breakdown and self-check tests pass on leto (280 passed).

**The CUDA k-space megakernel on the RTX 5060 Ti for large cells, and gates that fail on a revised plan (8 November 2026, owner: "builds one megakernel for the local gpu for mace polar"; worker polar-megakernel, conductor, decision 66):** POLAR's k-space field call runs as one CUDA kernel (`kernels/cuda/kspace.py`, `kspace_megakernel.CUDA` = True) on `MEGAKERNEL_GPUS` = the RTX 5060 Ti, for cells with at least `MEGAKERNEL_MIN_WORK` = 1e6 atoms × k-vectors (a size rule, as the one-wave rule of SPLIT_COLUMNS; smaller cells keep the dense path). MD step b/a on the 5060 Ti, both orders: NbCl5 3×3×3 (1944 atoms × 2989 k) S 0.893 / 0.896, M 0.957 / 0.957, L 0.972 / 0.973; NaCl (216 × 243 k, below the threshold) 0.998-1.012 with the kernel forced, hence the threshold. The kernel alone on the RTX A4000: 0.30 × dense at NbCl5, 0.70 × at NaCl. 3 × gate passes with polar_uvu on: S and M (NaCl, NbCl5) and L NaCl on the RTX 4000 Ada, L NbCl5 on a Modal A100-80GB. Other GPUs are not measured. `tools/force_stress_gate.py` now prints rule 13 revisions on its summary line and fails a gate whose candidate plan was revised (the gates run between 03:04 and 06:10 that way were redone). Codex R3 (POLAR head rows) stopped at the bound: P1 removed its target kernels. The POLAR, k-space, layering and gate tests pass on leto (401 passed). See `docs/design/research/2026-11-07-polar-megakernel.md`.

**The H200 fast path on the A100 family (8 November 2026, owner: "Get a best configuration on a100 now for omat-0-medium based on the h200 results"; worker sx-layout-lab, conductor, decision 66):** the ten H200 switches of 7-8 November (`SPLINE_W`, `R128`, `SPLINE_VECTOR`, `STAGE_EDGES`, `STAGE_SH`, `COMPOSE_SKIP`, `COMPOSE_FAST`, `COUNT_PLAN`, `CUEQ_PROJECTED`, `GENERIC_FOLD_READOUT_TAIL`) are on for the A100 family too. OMAT-0 medium, Benchmark_loup MD step on NHR Alex A100-SXM4-80GB, both orders: 9.85 to 7.36 ms (b/a 0.7471 / 0.7469); each switch turned off again from all ten is slower in both orders (leave-one-out b/a 1.0010-1.0325). The 3 × force/stress gate passes with all ten on (NbCl5 stress ratio 2.74 to 0.97). `POWER_BACKWARD` stays H200-only (1.0006 / 1.0028 on the A100). `tools/jobs/alex_ab_pack.sh` v2 gives each staged job its own OEq cache. CPU suite on leto at b8d977f4: 5443 passed, 202 skipped. MACE-OFF and POLAR-1 on the A100 are measured separately. See `results/a100-omat0-best-20261108/README.md` and `docs/current/omat0-a100.md` §3.8.4.

**OMAT-0 small on the H200, and A/B constants that cover the plan (8 November 2026, owner: OMAT-0 small "with the same settings as medium"; worker omat-0-small, conductor, decision 66):** OMAT-0 small gets the label `omat0-small` (`labels.EXTRA_LABELS`, recognised by SHA-256; not an acceptance label), so it takes medium's label-keyed pair, 3×TF32 GEMMs and the generic w′ backward, on the H200 (`DECISION_66_TF32X3_MODEL_GPUS`, `DECISION_66_GENERIC_WPRIME_LABEL_GPUS`); the A100 waits for its own A/B. Benchmark_loup MD step on the H200: 5.41 to 2.29 ms (b/a 2.360 / 2.370 for the pair turned off); every per-GPU H200 default, the tail fold included, applies to small. 3 × gate passes with small's defaults (forces 0.97 / 0.99 / 1.01, stress 0.50 / 1.20 / 1.22 of 3; SrTiO3 / NaCl / NbCl5). Fix: `tools/ab_rewrites.py` now sets each side's `--a/--b-constant` values around `plan()` as well (label-keyed decisions are taken there), so a side no longer plans under the previous side's constants; no earlier result set a planner constant, so only the small job was affected. H200 round 6 switches `JOINED_LAYOUT` (0.981 / 0.982 on medium, 0.968 / 0.971 on small; bitwise), w′ `STAGE_EDGES` and `PREFETCH_AHEAD` are built and off. CPU suite on leto at c4de72d2: 5459 passed, 203 skipped. See `results/h200-omat0-small-20261108/README.md`.

**JOINED_LAYOUT on the H200 (8 November 2026, owner's Codex H200 round-6 ideas; worker sx-layout-lab, conductor, decision 66):** `compose_skip.JOINED_LAYOUT` on for the H200 family (bitwise): Benchmark_loup MD step b/a 0.9805 / 0.9815 on OMAT-0 medium (0.968 / 0.971 on OMAT-0 small). Measured and left off: w′ `STAGE_EDGES` (a tie) and `PREFETCH_AHEAD` (1.0046 / 1.0067). The A100 verdict follows from Alex. CPU suite on leto at 6b7d9181: 5459 passed, 203 skipped. See `results/h200-fastpath-20261108/README.md`, round 6.

**Edge order for the spline table on the H200 (8 November 2026; worker sx-layout-lab, conductor, decision 66):** the spline forward spends about half its time reading the table (H200: one-slab b/a 0.55). `SORT_BY_LENGTH` (`neighbors.skin`): each receiver's edges are sorted by length bin at the skin-list build (rounding-class: the message sums change order); H200 Benchmark_loup MD step b/a 0.9807 / 0.9791, 3 × gate passes (forces 0.83 / 0.92 / 0.97, stress 0.48 / 0.94 / 1.07). `BLOCK_ORDER` (`spline_forward`): the forward's blocks walk the receivers grouped by species, the order bound by the force field (bitwise); 0.9753 / 0.9724 on top of the sort (3.51 to 3.42 ms). Both on for the H200 family; the A100 follows from Alex. Rejected by bound: float16/bfloat16 spline coefficients (certificate 3.16 against 1), per-pair knot grids (no fewer bytes per edge), Morton order alone. CPU suite on trillian at 9c730498: 5473 passed, 203 skipped. See `results/slab-sorted-h200-20261108/README.md` and `results/omat0-breakthroughs-assessment-20261108/README.md`.

**OMAT-0 medium on the RTX 4000 Ada, and two capacity fixes (8 November 2026, owner: "test the capacity of the newest fastest configuration of omat 0 medium", A/B and gate on the Ada first; worker omat0-ada-capacity, conductor, decision 66):** on the Ada the ten H200/A100 switches were inert: the spline family needs `HORNER` (`WPrime.attach_spline`) and the compose family and `CUEQ_PROJECTED` act only inside `tf32x3_gemms`, both off there. Adopted for "NVIDIA RTX 4000 Ada Generation": `HORNER`, `SPLINE_W`, `R128`, `STAGE_EDGES`, `COMPOSE_SKIP`, `COMPOSE_FAST`, `GENERIC_FOLD_READOUT_TAIL`, and `tf32x3_gemms` for OMAT-0 medium only (`DECISION_66_TF32X3_OMAT0_GPUS`, Triton backend; MH-1 unchanged). Benchmark_loup MD step b/a 0.8219 / 0.8208 (26.2 to 21.5 ms); leave-one-out: spline with HORNER 1.20, COMPOSE_SKIP 1.057, the rest 1.000-1.012. Left off on the Ada: `SPLINE_VECTOR` 4, `STAGE_SH`, `CUEQ_PROJECTED` (slower), `COUNT_PLAN`, `JOINED_LAYOUT`, CUTLASS 3xTF32 on sm_89 (ties). 3 × gate passes (forces 0.80 / 1.00 / 0.93, stress 0.51 / 1.13 / 1.16; SrTiO3 / NaCl / NbCl5; no rule-13 revision). Capacity fixes, every GPU (broken since `generic_wprime_backward`, 3 November): `attach_generic_wprime` now stands aside when rule 12 takes the rescue rung (B) instead of raising, which had dropped the plan to the rewrites-off terminal plan; and rule 12 has a memory term for `generic_wprime_backward` (`rewrites.memory`), which it lacked, so it predicted about 20.8 GiB at SrTiO3 40,000 (real about 4.1) and took rung B beyond about 35,000 atoms. RTX 5060 Ti (16 GB) capacity with the fixes: SrTiO3 163,840 atoms (7.85 µs per atom and step, peak 14.59 GiB; 34,295 before the fix, 53,240 on 2 November), NaCl 195,112 (4.15 µs; 64,000 before, 85,184 on 2 November). The Ada capacity scan is pending. CPU suite on trillian: 5470 passed, 203 skipped, 4 failed in `test_output_chunks` from the suite script's missing PYTHONPATH (25 / 25 with it set). See `results/omat0-ada-capacity-20261108/README.md` and `docs/validation/omat0/omat0-capacity.md`.

**H200 round 6 on the A100 family (8 November 2026; worker sx-layout-lab, conductor, decision 66):** three bitwise switches are on for the A100 family, each faster in both orders on NHR Alex A100-SXM4-80GB (OMAT-0 medium, Benchmark_loup MD step b/a against the A100 defaults): `JOINED_LAYOUT` 0.9782 / 0.9784, w′ `STAGE_EDGES` 0.9643 / 0.9853, `PREFETCH_AHEAD` = 2 0.9810 / 0.9943; all three 0.9590 / 0.9641 (about 7.37 to 7.09 ms). On the H200 w′ `STAGE_EDGES` and `PREFETCH_AHEAD` stay off (a tie and 1.005 there). The changed test files, test_compose_skip and test_layering pass on trillian (99 passed). See `results/h200-fastpath-20261108/round6-a100/`.

**BLOCK_ORDER on the A100 family (8 November 2026; worker sx-layout-lab, conductor, decision 66):** the spline forward's species-grouped receiver block order is on for the A100 family (A100-SXM4-80GB and A100 80GB PCIe). NHR Alex A100-SXM4-80GB, Alex pack 7 (job 4965032), OMAT-0 medium, Benchmark_loup MD step b/a against the A100 defaults: 0.9953 / 0.9945 (7.37 to 7.34 ms), bitwise forward. `SORT_BY_LENGTH` measured 0.9924 / 0.9936 and both together 0.9920 / 0.9904, but it is a rounding switch and stays off on the A100 until an A100 force/stress gate run. Only a one-line test assertion changed (`tests/cpu/test_spline_w.py`); it is not yet run in a CPU suite and goes into the next merge suite on a free workstation. See `results/slab-sorted-h200-20261108/README.md` and its `a100/pack7/`.

**SORT_BY_LENGTH on the A100 family (9 November 2026; worker sx-layout-lab, decision 66):** each receiver's edges are sorted by length at the skin-list build, and this is now on for the A100 family too (A100-SXM4-80GB and A100 80GB PCIe). It is a rounding switch. Its A100 force/stress gate (NHR Alex job 4967026; `tools/force_stress_gate.py`, production with `SORT_BY_LENGTH` and `BLOCK_ORDER`) passes: forces 0.92 / 0.93 / 0.96 and stress 0.64 / 1.02 / 1.15 against the limit of 3, for SrTiO3 / NaCl / NbCl5. The Benchmark_loup MD step with both switches against the A100 defaults (Alex pack 7) is 0.9920 / 0.9904 (7.39 to 7.33 ms). A one-line test assertion was added (`tests/cpu/test_neighbors.py`); it goes into the next merge suite. See `results/slab-sorted-h200-20261108/README.md` and its `a100/gate-sorted/`.

**MACE-OFF and MACE-POLAR on the A100 family (8–9 November 2026; worker off-polar-alex, conductor, decision 66):** `RewriteConfig.cueq_projected` (Toggle, auto) carries the planner's per-model decision for `CUEQ_PROJECTED` (`rewrites.cueq_projection`) under `tf32x3_gemms`. Auto is on where `CUEQ_PROJECTED_GPUS` lists the GPU, except for the labels `planner.fixed.DECISION_66_CUEQ_PROJECTED_EXCLUDED` lists there: OFF23-large and POLAR-1-S/-M on the A100 family, measured not faster (OFF23-large 1.0006 / 0.9996, POLAR-1-S NaCl 1.0042 / 1.0075, POLAR-1-M NaCl 0.9878 / 1.0058). It stays on for OFF23-small, OFF23-medium, OFF23b, OFF24 and POLAR-1-L (0.989–0.999 in both orders; the 3× gate passes on all five, no rule-13 revisions). The module constant still overrides it as an A/B side. `longrange.polar.MEGAKERNEL_GPUS` adds the A100 family (cells from `MEGAKERNEL_MIN_WORK`; NbCl5 MD step S 0.9846 / 0.9854, M 0.9918 / 0.9954, L 0.9855 / 0.9964; the 3× gate passes on S, M and L). `STAGE_SH` reaches POLAR-1-M (the only OFF/POLAR plan with `wprime_backward`): NaCl 0.9811 / 0.9914, NbCl5 0.9968 / 0.9977, bitwise. The other eight H200 switches need the generic family and cannot reach these plans (a forced control pack ties at 0.986–1.007). Net at the tip: about 1 % on POLAR NbCl5, 0.3–1 % elsewhere. On the earlier source 1bd4f3be the OFF23-small A100 default failed the 3× gate on paracetamol stress (3.13); with `CUEQ_PROJECTED`, its default now, it passes at 2.38. Full CPU suite on leto at fabd85f9: 5491 passed, 203 skipped. See `results/off-polar-a100-20261108/README.md`.

**OMAT-0 medium Ada capacity and the composed-skip self-check fix (9 November 2026; worker omat0-ada-capacity, conductor):** a third capacity bug, on every GPU with `COMPOSE_SKIP`: `generic_recompute_first_table`'s self-check (`rewrites.generic._run`) differentiated the probe's species table through `compose_skip`, which is inference only and refuses a weight gradient. The rescue rung therefore failed its self-check and the fallback dropped to the terminal sequence, which ran out of memory; the memory probe's replica of the rung measured the same failure (13.2 GiB at 17,576 atoms instead of 1.3), so it misjudged the rung near the limit on every GPU. With the composed skip on, the probe's table now carries no gradient and is left out of the differentiated leaves, as in production; the outputs and the other leaves' gradients are compared as before (969bfc4d; `test_generic_chunked::test_probe_runs_the_composed_skip_without_a_table_gradient` fails without it). RTX 4000 Ada (leto, every point alone on the GPU) capacity with the new defaults: SrTiO3 233,280 atoms (36³; 1,385 ms per step, 5.94 µs per atom and step, peak 19.15 GiB, rung `recompute_first_table`; 69,120 on 2 November), NaCl 343,000 (35³; 1,533 ms, 4.47 µs, 18.7 GiB; 125,000 on 2 November; 36³ is a genuine out-of-memory in an [atoms, 5120] buffer the rungs do not shrink). At the old limits the step is now 339 ms (SrTiO3, was 401) and 409 ms (NaCl, was 477), with lower peaks (5.82 against 6.78 GiB, 7.08 against 8.91). Rule 12 has no memory terms for the spline and composed-skip paths and overestimates the new Ada plan 1.8–2.15× (a follow-up; the near-limit probe corrects for it). CPU suite at 969bfc4d on trillian: 5,475 passed, 203 skipped. See `results/omat0-ada-capacity-20261108/README.md` and `docs/validation/omat0/omat0-capacity.md`.

**SC_GENERATED on the H200 and the A100 family (9 November 2026; worker bound-a-generated-layer, conductor, decision 66):** interaction 0's density division and cuEquivariance SymmetricContraction are replaced by one custom op whose straight-line NVRTC forward (division fused) and fused backward are generated at plan time from the module's own cuEq descriptor (`rewrites.sc_generated`, `kernels/cuda/sc_generated.py`; `RewriteConfig.sc_generated`, auto on with `tf32x3_gemms` where `SC_GENERATED_GPUS` lists the GPU). Lab on the H200: the four interaction-0 kernels 179.5 → 82.4 µs, about 1,030 / 1,490 FP32 instructions per atom and channel against cuEq's 1,381 / 2,935 (cuEq rebuilds each monomial product per term); outputs within rms 3e-7 (rounding class). OMAT-0 medium, Benchmark_loup MD step b/a: H200 0.9760 / 0.9731 (3.48 → 3.40 ms), A100 80GB PCIe on Modal (Alex unreachable) 0.9685 / 0.9652 (7.50 → 7.27 ms). The 3× float32 force/stress gate passes on both (H200 forces 0.850 / 0.920 / 0.959, stress 0.415 / 0.778 / 1.209; A100 forces 0.972 / 0.854 / 0.977, stress 0.701 / 0.898 / 1.237). CPU suite on leto at bd869591: 5,501 passed, 203 skipped. See `results/sc0-bound-20261109/README.md` and `results/sc0-op-20261109/README.md`.

**OMAT-0 small per-model defaults (8–9 November 2026; worker omat-0-small, conductor, decision 66):** the label `omat0-small` (`labels.EXTRA_LABELS`) gets its own measured defaults. Decision 66 pair (3×TF32 CUTLASS GEMMs and the generic w′ backward) on the H200 and the A100 family: A100 (NHR Alex) 0.3911 / 0.3916 (10.72 → 4.19 ms; 3×TF32 alone 0.824, generic w′ alone 0.559), gate passes (NbCl5 forces 1.03, stress 1.33). H200: w′ `STAGE_EDGES` on (`STAGE_EDGES_MODEL_GPUS`, adds the H200 on top of `STAGE_EDGES_GPUS`; 1.0072 / 1.0114 with it off, bitwise), spline forward R128 `MIN_BLOCKS` 8 (`R128_MIN_BLOCKS_MODEL_GPUS`; 0.9873 / 0.9883, bitwise; medium keeps 6). A100 family: spline forward `PREFETCH_AHEAD` off (0.9902 / 0.9908, confirmation 1.0068 / 1.0073 with it on; bitwise). `FOLD_ALPHA` off measured within the H200 noise and stays on. `tools/ab_rewrites.py` covers plan-time constants and takes dict literals for per-model tables. OMAT-0 small MD step: H200 5.41 → 2.12 ms, A100 10.72 → about 4.17 ms. The Ada and RTX 5060 Ti are not measured for small. See `results/h200-omat0-small-20261108/README.md`.

**Tiled and wavefront-tiled modes up to current speed (8–9 November 2026; worker tiled-speed, conductor, decision 66):** OMAT-0 medium, SrTiO3 975,560 atoms, RTX 5060 Ti, same-process A/Bs in both orders. Device state 8.83 → 6.42 µs per atom and step (all switches off against on 0.729 / 0.731; 1.34× the whole-system call per atom, was 1.83×); wavefront 12.0 → 7.74 µs (old wavefront defaults against new 0.766 / 0.774; 1.61×, was 2.50×). Switches (device b/a): `TILE_PREFETCH_DEVICE` 0.939 / 0.941 (exact), `TILE_BLOCK_ROWS` 0.918 / 0.921 (rounding; wavefront 0.931 / 0.937), `TILE_VIRIAL` 0.970 / 0.955 (rounding), `TILE_GRAPH_CACHE` 0.948 / 0.960 and `TILE_GRAPH_CACHE_WAVE` 0.824 / 0.817 (exact; each tile's graph built once instead of four times), `TILE_FILTER32` 0.987 / 0.993, `TILE_EPILOGUE` 0.980 / 0.981 (rounding), `TILE_FILL_SKIP` 0.961 / 0.966 (exact, every GPU), `TILE_PINNED_SUPERSET` 0.986 / 0.990, `TILE_HOST_WORKER` (host state's deferred writes on one worker thread, reads wait for earlier batches; wavefront 0.975 / 0.982). The scoped switches are on for `TILED_SPEED_GPUS` (the RTX 5060 Ti). Rejected: `TILE_PIECE_START` 0.9999 / 1.0001, `TILE_LAZY_KEYS` 1.023 / 1.002, keeping interaction 0's activations for sweep 3 (8 kB per atom). The force/stress gate (tiled, compiled) passes 12/12 with the rounding switches on. Batch CPU suite on leto (main with OMAT-0 small and tiled-speed, f6e654f1): 5,531 passed, 203 skipped. See `results/tiled-speed-20261108/README.md` and `docs/validation/tiled/tiled-rewrites-1m.md`.

**PACKED8 and BUTTERFLY on the H200 and A100 family (9 November 2026; worker sx-layout-lab, conductor, decision 66):** with `STAGE_EDGES`, the SPLINE_W forward reads an 8-byte-per-weight copy of the spline table instead of the float4 one (`spline_forward.PACKED8`, `PACKED8_GPUS`): per weight the cubic in s = t − 1/2 economised to a quadratic, one shared 6-bit exponent and signed fixed-point coefficients of 22, 20 and 16 bits (format S12, `tools/spline_packed_certificate.py`: value part 0.161 of the certificate budget), decoded exactly by exponent-or tricks. The w′ backward keeps the float32 table for its slopes. Rounding class. Benchmark_loup MD step b/a (b = on): H200 300 K 0.9930 / 0.9945, 1500 K 0.9818 / 0.9768; A100-SXM4-80GB (L2-bound forward) 300 K 0.9557 / 0.9549, 1500 K 0.9587 / 0.9558. The 3× float32 force/stress gate passes on both. `wprime_backward.BUTTERFLY` (end-of-edge butterfly reduce-scatter) is on again for the H200 and now for the A100 family against today's w′ defaults: MD step b/a (b = on) H200 SrTiO3 0.9863 / 0.9861, NbCl5 0.9936 / 0.9903; A100 (with PACKED8) 0.9910 / 0.9892 and 0.9942 / 0.9973; gate passes on both. Closed in the same lane: the FP64 second pipe for w′ (every offload slower on the A100), NVRTC 13.5 (about 2 % slower), the B-spline and 12-byte table formats, and the grouped GEMM (bounded at 61 µs). `SMEM_SPILL`'s earlier A/Bs were void (ptxas ignores `--maxrregcount` under `__launch_bounds__`); it stays off, untested again on the H200. Integrated with tiled-speed's block-rows argument (both spline Functions take `rows` and `forward_table`). CPU suite on leto at 2b421e84: 5,531 passed, 203 skipped. See `results/packed-table-lab-20261109/README.md` and `results/step-limiters-20261109/`.

**Memory profile and the outer-sum shape carrier (9 November 2026; worker omat0-ada-capacity, conductor, decision 66 work removal):** `tools/bench/memory_profile.py` attributes peak device memory by owner from allocator snapshots. At OMAT-0 medium SrTiO3 36³ (233,280 atoms, RTX 4000 Ada, MD) the largest single block was the recomputing tabled outer-sum backward's [E, 64] float buffer, allocated only to carry a shape (18.9 kB per atom). A stride-0 view now carries it (`tabled_outer_sum.SHAPE_CARRIER` = True; gradients bitwise; Benchmark_loup MD step b/a 1.0009 / 1.0002). By owner, before → after, in kB per atom: reserved 86.6 → 75.9; CUDA graph pool 70.4 → 59.7 (live 50.6 → 46.0, packing slack 19.9 → 13.8); default pool 16.1 (neighbour list with its rebuild temporaries 11.9, tables 1.6); single force call 62.7 → 59.0 allocated. With the fix the live set is the four backward GEMM gradients (20.5), the saved [N, 2048] output (8.2), edge spherical harmonics (4.7), w′ blocks (6.8) and the rest (5.8). Ada limits: SrTiO3 233,280 → 253,265 atoms (37³, 1,563 ms per step; 274,360 fails in the neighbour-list rebuild sort beside the 15.4 GiB graph pool); NaCl unchanged at 343,000 (373,248 fails on rung B's dense [N, 5120] message gradient, which Inductor concatenates from the four GEMM gradient blocks: the rescue rung itself adds 20.5 kB per atom, and rule 12 has no term for it). RTX 5060 Ti unchanged (SrTiO3 163,840, NaCl 195,112). Rejected: `EngineConfig.empty_cache_before_warmup` (the failure moved to the first eager call). CPU suite on leto at c4cde0c8: 5,531 passed. See `results/memory-profile-20261109/README.md`.

**OMAT-0 switches screened on MACE-OFF and POLAR-1 on the A100 family (9 November 2026; worker off-polar-alex, conductor, decision 66):** three A100-family defaults are off per model label, all pure removals back to gated or bitwise configurations, so no gate. `SORT_BY_LENGTH` off for OFF23-small and OFF23-large (`skin.SORT_BY_LENGTH_MODEL_OFF_GPUS`; MACE-OFF has no spline tables; water_216, b = off, ab / ba: 0.9830 / 0.9739 and 0.9951 / 0.9900; OFF23-medium, OFF23b-medium, OFF24-medium and POLAR-1 S / M / L tie and keep it). For POLAR-1-M, w′ `STAGE_DOUT` off (`wprime_backward.STAGE_DOUT_MODEL_OFF_GPUS`; NaCl 0.9682 / 0.9770, NbCl5 0.9920 / 0.9911) and fused-forward `DEGREE_PARTS` off (`fused_forward.DEGREE_PARTS_MODEL_OFF_GPUS`; 0.9825 / 0.9911, 0.9967 / 0.9962). `BUTTERFLY` stays on for POLAR-1-M (mixed), and POLAR-1-L's w′ with the fused forward stays off (NbCl5 1.009 / 1.012). The table label now reaches the skin neighbour list (`SkinNeighbourList(model_label=)`), the MH-1-class w′ attach and the MH-1 fused forward; no new config field and no plan-text change. NHR Alex A100-SXM4-80GB screen jobs 4970847–49 on main 6f03bd02; confirmation on the branch (jobs 4972687/88, b = main's defaults forced back): OFF23-small 1.0251 / 1.0074, OFF23-large 0.9972 / 1.0047 (tie), POLAR-1-M NaCl 1.0129 / 1.0306, NbCl5 1.0145 / 1.0142. CPU suite on leto at 04384f8a (code identical to the tip ce6c6cf9): 5,538 passed, 203 skipped. See `results/off-polar-a100-r2-20261109/README.md`.

**Keys-only neighbour-list rebuild sort, and the graph-pool slack (9 November 2026; worker omat0-ada-capacity, conductor, decision 66, capacity item 1):** `skin.SORT_KEYS_ONLY` (default True): `build_gpu` writes each screened pair's unique order key over its screen code, sorts the keys alone (`torch.unique`, no index tensor or key-value sort) and decodes receiver, sender and image from the sorted keys (`SkinNeighbourList._decode_key`). The same pairs in the same order: every list buffer and `compact()`'s edge list are bitwise identical on the RTX 4000 Ada at SrTiO3 36³, and in a new 8-case CPU test. Work removal; Benchmark_loup MD step b/a (b = new): Ada 1.0004 / 0.9999, RTX 5060 Ti 1.0027 / 0.9986. At SrTiO3 233,280 atoms on the Ada the rebuild peak above the list's buffers falls from 1,019 to 351 MiB at the same build time, and the MD peak reserved from 16.49 to 15.82 GiB (75.9 to 72.8 kB/atom). Ada capacity: SrTiO3 253,265 → 274,360 atoms (38³, 1,746 ms per step); NaCl unchanged at 343,000; RTX 5060 Ti unchanged (163,840 and 195,112; its first failing sizes fail in the build on Triton module loads and driver allocations outside the allocator). The graph pool's 13.8 kB/atom packing slack is best-fit placement loss inside the capture's single expandable segment (1.28 GiB of holes at the live peak, the largest 797 MiB from the spline forward's message blocks); `JOINED_LAYOUT` does not change it and `expandable_segments:False` runs out of memory at 36³ (rejected, ledger). Both Ada limits are now set by rescue rung B's dense [N, 5120] gradient (capacity item 2). CPU suite on leto at the merged head: 5,546 passed, 0 failed. See `results/memory-slack-20261109/README.md`.

**Tiled capacity round: 397.5M atoms on one A100-80, host state dtype "auto", A100 build and leak fixes (10 November 2026; worker tiled-speed, conductor, decision 66):** `TileConfig.state_dtype` (default "auto") picks the tiled path's aliased host state: the call's dtype while the host budget allows it, then the kept neighbour lists are dropped (exact, about 14 % slower), then float16. The 16-bit state is for capacity only: it fails the 3× float32 force/stress gate (float16 12–51×, bfloat16 112–310× the float32 spread), so a run that needs the gate sets `state_dtype` to the call's dtype. The host room now also reads the memory cgroup (batch jobs see the whole node in /proc/meminfo). OMAT-0 medium SrTiO3 on one NHR Alex A100-SXM4-80GB (job cgroup 231.5 GB): 397,535,000 atoms at 9.84 µs per atom, 229 GB host and 40.6 GiB device; the limit is host memory. A100 fixes on the way: `cutlass.segment_rows` keeps one index tensor per layout instead of one per row count (64 GB at 149M atoms), and a tensor a CUDA graph capture read is retired rather than freed when more rows replace it (growth at least doubles the rows; CPU test); the species probe checks one-hot rows in chunks; `forcefield.distinct_numbers` and `species._distinct_rows` replace Python loops and row sorts; the MH-1 weakref caches drop dead entries. The 149M-atom build falls from 2,018 to 371 s. New tools: `tools/bench/tiled_host_profile.py`, `tools/bench/tiled_nve.py`. CPU suite on leto at dcabed40 (main a7a8211e merged in): 5,562 passed, 203 skipped, 0 failed. See `results/tiled-capacity-20261109/` and `docs/validation/tiled/`.

**Rung B beside w′, the first-layer recompute rung and the two-replica memory probe (10 November 2026; worker omat0-ada-capacity, conductor, decision 66, capacity item 2):** rescue rung (B) is never taken beside w′ (`planner.memory.RESCUE_NOT_BESIDE`): there it saved nothing and added the dense [N, 5120] message gradient, and rule 12 now has no term for it beside w′. New rescue rung `generic_recompute_first_layer` (`RewriteConfig.recompute_first_layer`, "auto"), taken after the first table's rung: the first interaction and product 0 run under one checkpoint, so product 0's [N, 2048] input (8.2 kB/atom) is not kept. Rounding, float32 only, not deterministic; 3× float32 gate on the RTX 4000 Ada with both rungs forced on (SrTiO3 forces 0.81, stress 0.42; NaCl forces 0.93, stress 0.96, all under 3). Where taken it costs about 9 % (b/a 1.0923 / 1.0917). The memory probe now also measures the first stage on a second, 45k–60k-atom replica and fits an offset and slope per stage (`fitted_offset`, `fitted_scale`; one more child process, about 50 s, only when the probe runs): a single 16,875-atom replica overstated the scale-up (0.70 against 0.52 at 233,280 atoms) and made the default take both rungs at SrTiO3 34³–36³. Against a7a8211e the default plan at 32³ is equal line for line and 34³–36³ tie (separate processes, both orders, 0.9926–1.0003); at 37³ the corrected estimate is 0.2 % above decision 51's limit, so the new tree takes the layer rung there (6.9 % slower, 1.1 GiB less) where a7a8211e ran the table stage at 17.31 GiB. Limits: RTX 4000 Ada SrTiO3 274,360 → 296,595 atoms (39³, 2,043 ms per step), NaCl 343,000 → 405,224 (37³); RTX 5060 Ti SrTiO3 163,840 → 214,375 (35³), NaCl 195,112 → 287,496 (33³). The remaining limiter is the 3×TF32 GEMM gradient blocks (20.5 kB/atom), which no recompute can drop. CPU suite on leto at the merged tip 9b20ddd4: 5,573 passed, 203 skipped, 0 failed. See `results/rung-b-20261109/README.md`.

**Docs refresh: ledger statuses and defaults by GPU against the code (10 November 2026; worker docs-refresh, conductor; docs only):** 35 ledger index rows and 8 appendix statuses were brought in line with the `*_GPUS` sets on main 7d18028d, each citing the commit or §19 entry that changed it. H200-only rows now name the A100 family and the Ada where the sets include them. Rows marked "on branch" that were already merged are now DEFAULT, and the October PENDING rows are resolved or annotated. `docs/current/defaults-by-gpu.md` gains §0, one row per switch set with H200 / A100 / Ada / 5060 Ti cells (on, off with the measured value, not measured, or n/a), and an "Open: not measured per family" list stating whether each open cell needs an A/B only or an A/B plus the 3× float32 gate. No code changes.

**GEOM and SC_FACTORED on the A100 family, VIRIAL_ELEMENTWISE on the RTX 5060 Ti, and the other-ways labs (11 November 2026; worker sx-layout-lab, conductor, decision 66):**
- **GEOM:** the spline forward with selective shared geometry (`spline_forward` GEOM). It needs SPLINE_W and STAGE_EDGES.
- **SC_FACTORED:** the factored generated symmetric contraction (`kernels/cuda/sc_factored.py`). It needs SC_GENERATED.
- Both are default on the A100 family. Both are rounding; the 3× float32 gate passes with both on (NHR Alex A100-SXM4-80GB, job 4973813, source 219602cf). MD step b/a: GEOM 0.9756 / 0.9806, SC_FACTORED 0.9889 / 0.9881.
- **VIRIAL_ELEMENTWISE:** the strain gradient summed elementwise from the edge vectors and their gradient, instead of a split-K [3, E] × [E, 3] GEMM (the tiled path's TILE_VIRIAL, brought to the whole-system plan). Default on the RTX 5060 Ti: stress force call SrTiO3 10×10×10 b/a 0.9675 / 0.9648, gate passes (forces 0.795 / 0.796 / 0.970, stress 0.647 / 0.685 / 0.619). Off on the A100, where the GEMM is 0.4 % of kernel time (Alex 4976350; its gate also passes). Stress calls only.
- **H200 and Ada:** not measured. On the 5060 Ti, GEOM and SC_FACTORED are "n/a" until their prerequisites are adopted there.
- **Labs recorded without adoption:**
  - tensor cores for the edge tensor product, rejected by bound;
  - mirror-pair w′ on the A100 (mirror order 2.87×, half table 1.31×);
  - receiver-adjoint w′ (A100 1.26× slower);
  - interaction 0's copies and zero-fill, stopped under 1 %;
  - step limiters on an Alex A100 (no new 1 % lead).
- CPU suite on murphy at the merged tip a00c07e3: 5,572 passed, 203 skipped, 1 failed. The failure was a test expectation, which counts 4 strain self-check modules since VIRIAL_ELEMENTWISE; it was fixed in 0426f7d7, and test_rewrites_engine.py and test_layering.py were rerun: 59 passed.
- See `results/virial-elementwise-20261110/README.md` and the other-ways README under results/.

**SplineForward registry: one object per equal build (11 November 2026; worker ada-5060-screen, conductor; correctness fix):** `SplineForward.__init__` replaced `_REGISTRY[id]` on every equal build. The id depends only on source and device, so a second force field in the same process overwrote the first's entry. The custom op `spline_message_blocks` launches through the registry, so earlier force fields, and their captured graphs, used the newer object's BLOCK_ORDER `order`. Once that object was replaced again and freed, a replay read freed memory: a CUDA illegal address in a five-engine A/B process on the RTX 4000 Ada (10 November). Forces stayed correct until then, because any receiver permutation gives the same message. The registry now keeps the first object (`_REGISTRY.setdefault`, `spline_forward.shared`). The w′ build binds that registered object, and each force field binds its own `order` on it before every call and capture (`forces._bind_streaming_count`). `launch_order` never reads a bound order of another row count; identity orders are cached per row count. This affects every process with more than one force field and BLOCK_ORDER on (H200 and A100 family defaults): Monte Carlo, replicas, A/B harnesses. Tests: 6 CPU tests, plus a GPU test with three engines of different sizes in one process, one closed mid-run, each matching its alone run (forces ≤ 2.0e-5 eV/Å, RTX 4000 Ada). The GPU test pins the shared-object semantics but did not reproduce the use-after-free on the old code; compute-sanitizer 2026.2 aborts this stack on the host. CPU suite on murphy at the merged tip be193ea0: 5,563 passed, 203 skipped; the 16 failures, all in test_colab_driver.py, come from git's NFS ownership check (all 27 pass with safe.directory set).

**A100 idea screen on Colab (11 November 2026; worker a100-colab-ideas, conductor; records and lab tools, no production change):** five ideas screened on a Colab NVIDIA A100-SXM4-40GB.
- **forward ghoist:** superseded by GEOM. The same regrouping of the forward's sums, built as `spline_forward.GHOIST` and dropped at the merge (source kept at cd7575bb). Its A/B is an independent replication of GEOM's A100 gain on a 40 GB A100: MD step SrTiO3 10×10×10 0.9774 / 0.9806, NbCl5 3×3×3 0.9891 / 0.9886; gate passes.
- **Half-precision far-shell edges (half-far-edges, OPEN):**
  - CPU oracle with float16 far-edge inputs, messages and sums: 0.9–1.8× float32 rounding at r_h 5.0–5.8 Å (gate 3×).
  - HFMA2 is a second pipe on sm_80: 63.8 FFMA, 127.3 HFMA2, 167.7 in a 1:1 mix (FMA results/clk/SM).
  - Offline census of w′ with two far edges per `__half2`: 0.634 of the issue slots per edge, 0.688 at 80 registers.
  - Colab lab: 0.484–0.496 of fp32's time per far edge; estimated 2–4.5 % of the force call.
- **Two half-systems on two streams: rejected** (1.008 of the full graph).
- **Formulation search: closed.** It rediscovers FACTORED, ranking it 2nd of 27; the best tree is about 1 % of the step.
- **TF32 big·big with FP16 cross GEMM: stopped,** error 6.3e-7 to 1.4e-6 against the 2.7e-7 stop rule.

Tools: `tools/bench/cu_census.py`, `tools/bench/half_system_streams.py`, `tools/fma_dual_pipe_bench.py` HFMA2, `tools/fp16x3_gemm_bench.py`. CPU suite on leto at 51634810: 5,579 passed, 203 skipped, 0 failed. Colab spend 14.9 CU. See `results/a100-colab-ideas-20261111/README.md`.

**int32 neighbour-list edge_index (11 November 2026; worker omat0-ada-capacity, conductor, decision 66):** `NeighbourConfig.edge_index_int32` (default "auto") stores the skin list's `edge_index` in int32. The planner turns it on for `planner.fixed.DECISION_66_EDGE_INDEX_INT32_GPUS` (RTX 4000 Ada, RTX 5060 Ti, every A100 model) and records the decision in the plan; the H200 is not measured and stays off. It is a rounding change: on the reproducible MH-1 plan, forces differ by up to 3.8e-5 eV/Å, while energy and cell stay bitwise. The 3× float32 gate passes:

| GPU | SrTiO3 forces / stress | NaCl forces / stress |
|---|---|---|
| RTX 4000 Ada | 0.79 / 0.47 | 0.91 / 1.00 |
| A100-SXM4-80GB (Alex 4981362) | 0.93 / 0.65 | 0.83 / 0.88 |

MD step b/a, SrTiO3 10×10×10 (b = int32), ab / ba: Ada 1.0001 / 0.9987, RTX 5060 Ti 1.0022 / 0.9959, A100 0.9960 / 0.9951. Memory at Ada SrTiO3 34³: neighbour list 1.37 → 1.21 GiB (0.8 kB/atom), peak reserved 13.40 → 12.95 GiB; the cube-step capacity limits do not move. The tiled kept lists are already about 3 B per edge, so they gain nothing. CPU suite on leto at the merged tip 20352f78: 5,580 passed, 203 skipped, 2 failed. Both failures were the new test, fixed in efc24fe0; the changed test files were rerun (77 + 204 passed). See `results/edge-index-int32-20261111/`.

**Tiled speed round (11 November 2026; worker tiled-speed, conductor, decision 66):**
- **Speed switches widened.** `TILED_SPEED_GPUS` now covers the RTX 4000 Ada and every A100 model, as well as the RTX 5060 Ti. It was measured with everything off against everything on at 975,560 SrTiO3 atoms, in both orders:

  | GPU | device state | wavefront |
  |---|---|---|
  | RTX 4000 Ada | 0.868 / 0.868 | 0.772 / 0.769 |
  | A100-SXM4-80GB | 0.792 / 0.789 | 0.782 / 0.785 |

  The gates pass on the Ada (murphy) and the A100 (Alex 4977013, and 4981048 on the merged code).
- **`TILE_COPY_STREAM` (exact).** The host state's deferred device-to-host writes now copy on a side stream from device clones. Wavefront 0.962 / 0.964, host state 0.960 / 0.959. On for `TILED_SPEED_GPUS`.
- **`TILE_HEAP_TRIM` (memory only, default True).** Dropped kept lists and the list cache call `malloc_trim(0)`. A forced drop of 7.0 GiB of lists at 20.1M atoms left 2.77 GiB in glibc's free heap, and the trim returned all of it.
- **Pinned kept lists.** They are counted by the pinned allocator's power-of-two blocks and returned to the system when dropped (`_release_kept`). Without this, the Ada's host peak at 179.7M atoms grew by 7 GB.
- **Idea 3.**
  - The aliased state and the wavefront's row pool on the device (`alias_state` with `state="device"`) are opt-in, not in CUDA graphs and not with lanes.
  - The device row pool as a default is rejected.
  - Receiver caps above 16,384 are rejected on the 5060 Ti.
- **Two `TileConfig.compiled_graphs` correctness fixes** for the opt-in option:
  - 35bcd4ab: a new capture pool once no graph holds the old one. Before, capturing into a private pool PyTorch had released failed an allocator assertion; an MD key change could trigger it.
  - d7d9e7ce: captured outputs live in the slot's `graph_out`, apart from uncaptured calls.

  GPU test 3e37bf5a. The A100 A/B of compiled_graphs (Alex 4981483: 0.569 / 0.571, 3.68 → 2.09 µs/atom) is recorded only; making it a default is a later merge, after its gate and a memory guard.

Records: Ada tiled capacity 179.7M SrTiO3 atoms; the A100's 52 % GPU-idle share by Nsight Systems (host-bound in the compiled bodies' Python and autograd). The 397.5M A100 rerun with the heap trim (Alex 4981047) is still pending; its row follows. CPU suite on murphy at the merged tip ddaaed9c: 5,589 passed, 203 skipped, 0 failed (717b9f92 adds only a README note). See `results/tiled-speed-20261111/README.md`.

**Ada and 5060 Ti re-screen of the H200/A100 switches (11 November 2026; worker ada-5060-screen, conductor, decision 66):** each switch measured alone and left out of the candidate set (OMAT-0 medium, SrTiO3 10×10×10 and NbCl5, both orders), then the adopted set against the old defaults, with the 3× float32 gate.
- **RTX 4000 Ada.**
  - On: SORT_BY_LENGTH, BUTTERFLY, PACKED8, w′ STAGE_DOUT / STAGE_SH / STAGE_EDGES, POWER_BACKWARD, SC_GENERATED, SC_FACTORED, SPLIT_COLUMNS (OFF23-large 0.9556 / 0.9583), the POLAR megakernel (S 0.896, M 0.950, L 0.968).
  - The set against the Ada defaults: SrTiO3 0.8255 / 0.8194 (21.4 → 17.7 ms), NbCl5 0.8546 / 0.8558.
  - Off: GEOM (1.0226 / 1.0209), VIRIAL_ELEMENTWISE, SEGMENT_ROWS (tie), RECEIVER_SUM (structure-dependent), GRAPH_SUM, SCALAR_PATHS, BLOCK_ORDER, DEGREE_PARTS.
  - BUTTERFLY (off 5 Nov) and STAGE_SH (off 8 Nov) flipped on.
  - POLAR-1-M and -L: the set was slower (M NaCl 1.0061 / 1.0077). Leave-one-out found STAGE_DOUT, so the Ada joins `STAGE_DOUT_MODEL_OFF_GPUS` for polar1-m and polar1-l; without it, M and L tie with the defaults. OFF23-medium ties.
- **RTX 5060 Ti.**
  - On: SPLINE_W with R128 and spline STAGE_EDGES, PACKED8, SORT_BY_LENGTH, GENERIC_FOLD_READOUT_TAIL, POWER_BACKWARD, RECEIVER_SUM, w′ STAGE_EDGES / STAGE_DOUT / STAGE_SH, generic fold_product_linear.
  - The set against the 5060 Ti defaults: SrTiO3 0.6427 / 0.6478 (23.9 → 15.4 ms), NbCl5 0.6911 / 0.6906.
  - POLAR: gate_epilogue on for S, M and L; the w′ pair for POLAR-1-M only (NbCl5 0.927 / 0.939). Not for L (NbCl5 1.0098 / 1.0097).
  - Not adopted: the SC pair (tie), SYMMETRIC_PAIRS (mixed), CUEQ_PROJECTED, BLOCK_ORDER, GRAPH_SUM, DEGREE_PARTS, SPLINE_VECTOR 4.
  - The 5060 Ti gates use 2×2×2 cells: the 3×3×3 POLAR-1-M float64 reference failed to map GPU memory under WSL.
- **Gates.** All pass: OMAT-0 on both GPUs, POLAR-1-M and OFF23-medium on the Ada, POLAR S, M and L on the 5060 Ti.

CPU suite on murphy at e63f2a9c: 5,552 passed, 203 skipped, 38 failed. 37 failures came from dftd3_params.npz missing from the tree sent, and 1 from a stale test, fixed in 47cec07f (the only later change). The 8 affected files were rerun with the npz present: 115 passed, 16 skipped, 0 failed. See `results/ada-5060-screen-20261110/README.md` and `docs/current/models-firing.md`.

**Compiled tile bodies in CUDA graphs on the A100, behind a memory guard (11 November 2026; worker tiled-speed, conductor, decision 66):** `TileConfig.compiled_graphs` is now None / True / False, with None as the default.
- **Default:** None turns it on for `COMPILED_GRAPH_GPUS` (the A100 family), only with device state, mode compiled or auto, lanes not "on", and no aliased device state.
- **Memory guard:**
  - The first compiled call runs uncaptured and measures its tile workspace.
  - Captures follow only where (slots + 1) × 1.1 × workspace fits the available memory (`compiled_graphs_fit`), checked again when more slots appear.
  - A captured call that runs out of memory drops its graphs and runs again without them.
- **A100-SXM4-80GB:** 975,560 SrTiO3 atoms, 0.569 / 0.571, 3.68 → 2.09 µs/atom against 1.96 for the whole system (Alex 4981483). The A100 tiled call was host-bound, and the graphs remove the Python and autograd gaps. The 3× gate passes with graphs on (Alex 4982466).
- **RTX 5060 Ti: off.** 30³ 1.007 / 0.990, 40³ 0.998; at 50³ the guard turns it off (local job 49, under the watchdog).
- **RTX 4000 Ada: off.** 40³ 1.0015; 58³ 1.0010 / 1.0007, with the guard off.
- The first idea-11 frame code (`TILE_LOCAL_FRAME`) is included, switched off. The accuracy fix follows in its own merge request.

CPU suite on murphy at the merged tip 0d1ce6d6, collecting `tests/`: 5,600 passed, 873 skipped (203 in tests/cpu as on main, plus 670 GPU tests without CUDA), 1 failed. The failure was the layering allow-list, fixed in 7a62514c (test_layering: 32 passed). See `results/tiled-speed-20261111/`.

**Float64 edge vectors for large boxes (idea 11; 11 November 2026; worker tiled-speed, conductor; accuracy fix):** float32 edge vectors carry the float32 spacing of their coordinates. On the perfect SrTiO3 crystal the largest force therefore grew from 1.64e-4 eV/Å at 1M atoms to 1.32e-3 at 149M and 320M atoms, 37× the gate's rounding basis. A 6³ crystal moved to +1,500 Å reproduces it. The positions carry the error; the shifts' share is at most half.
- **`tiled.TILE_LOCAL_FRAME` (tiles).** Each tile's rows and remaining shifts are built relative to its first receiver in float64, then stored as float32 hi/lo halves with an exact TwoSum. Halo atoms sit at their edge's periodic image. With positions on the host it uses int16 bins plus float32 offsets.
- **`forces.WHOLE_SYSTEM_FLOAT64_EDGES` (whole system).** It adds a detached float64 correction to MACE's float32 edge vectors (`engine.float64_edge_vectors`), with float64 positions in the dict (24 B/atom). With a strain, the edges carry it, which fixes the float32 virial cancellation far from the origin.
- **Default.** Both are None: on for float32 models where the largest |coordinate| of the positions and the cell's corners is at least `FLOAT64_EDGES_EXTENT` = 64 Å. Below that the code path is unchanged.
- **Accuracy.** About 3e-6 eV/Å on the perfect crystal at 0, +200 and +1,500 Å, in production tiles (R 16,384, including a tile that meets itself through the boundary). A100 at +1,500 Å: tiles 2.5e-6, whole system 6.6e-6, against 1.33e-3 off. The 3× gate with both forced on passes on the A100 (Alex 4983752, 12/12).
- **Cost where on.**
  - Frame: A100 at 58³ with compiled graphs 1.028 / 1.029, Ada at 40³ 1.0078 / 1.0064, 5060 Ti at 40³ 1.004 / 1.010.
  - Whole system with stress: A100 at 40³ 1.0146 / 1.0146, 5060 Ti 1.016 / 1.015.
  - The conductor accepted these as an accuracy fix: 2 % bar, 3.5 % for the A100 frame, because the A100 is the capacity GPU. A faster A100 frame is follow-up 11b in `plans/ideas-20261111-tiled-round2.md`.

CPU suite on murphy (tests/cpu) at f8ea447e with main b951e804 merged in: 5,610 passed, 203 skipped. The later commits are docs and records; main's later f31d4f50 touches only plans/, results/ and docs/. See `results/tiled-speed-20261111/README.md`.

**Ledger tidy (11 November 2026; worker docs-refresh, conductor; docs only):**
- **Six new rows.**
  - Open, with estimated sizes, first tests and stop rules: ideas 18, 19 and 20, and the outside ideas "joint rescaling" and "one decode for two edges".
  - Closed by argument: texture interpolation.
- **Four stale statuses corrected to REJECTED:** omat0-slope-vector, -stream-w, -stream-store-w, -eos-chunks.
- **The 27 unverified agent notes on the rejected-row catalogue:** 14 confirmed and fixed, 11 not confirmed, 2 unclear (their records are in ResearchModule). See `plans/notes/rejected-ledger-20261111/verified.md`.

**BINNED_SCREEN on the RTX 5060 Ti and tf32x3 HYBRID on the A100 family (11 November 2026; owner request, workers review and a100-colab-ideas, conductor, decision 66):**
- **`pair_search.BINNED_SCREEN_GPUS` gains the RTX 5060 Ti.**
  - Work removal; the neighbour list is the same bit for bit.
  - MD step SrTiO3 16×16×16: 0.9812 / 0.9769. 20×20×20: 0.9614.
  - Rebuild at 214,375 atoms: 2,131 → 73 ms.
  - See `results/binned-screen-5060ti-20261111/`.
- **`tf32x3.HYBRID` (None: on for `HYBRID_GPUS`, the A100 family).**
  - The cubins are built with `-DFM_HYBRID_F16` (`hybrid_f16.h`): big·big on TF32 tensor cores, both cross products in one FP16 mma.
  - Lab: 0.873 of 3×TF32, error at or below 3×TF32's on every shape.
  - MD step: SrTiO3 10×10×10 0.9896 / 0.9877, NbCl5 0.9832 / 0.9846 (Alex A100-SXM4-80GB). The gate passes.
  - Range: float16 cross parts overflow above 32,752. On the CPU, the worst operand is 778.7 (a 3×3 stress product; feature GEMMs at most 101.5) and the weights' max is 81.8. The cases covered: the gate structures, 30 % compression, a 0.8 Å pair and a 2,000 K MD frame. A CPU test keeps the tenfold margin on record.
  - The sm_90 cubin exists; the H200 is not measured (no Modal budget this month).
  - See `results/hybrid-gemm-20261111/`.
- **Closed with records in the same merge:** half-precision far edges (`HALF_FAR`) fail NaCl's symmetric shell at every radius (F 7.1×, S 13×). Two half-systems out of phase measured 1.016.

CPU suite on murphy at the merged tip 1a80d2e9 (tests/cpu): 5,613 passed, 203 skipped, 0 failed. The range-check commit after it adds one test file: 36 passed, 1 skipped, with test_layering.

**Offline census of C3, C4 and C5 (11 November 2026; worker a100-colab-ideas, conductor; records only):** NVRTC builds for sm_80 and sm_90 of the production w′ and forward sources. All three stop at their bars without GPU time:
- **C3, one spline decode for two edges:** spills on sm_90, and saves 17.9 / 7.7 / −10.4 instructions per edge and channel at the note's pairing rates, against a bar of 27.
- **C4, staging-only joint rescaling:** about 16 instructions per edge and channel, about 97 µs on the A100 and 37 µs on the H200, against bars of 130 and 68.
- **C5, w′ accumulators from the first product:** no drop on sm_90 (477 → 477).

See `results/census-c345-20261111/`.

**First interaction, C1 and C2 (11 November 2026; owner request, worker first-interaction, conductor; records only, both stopped):**
- **C1, the forward by slot with sums in registers.** The falsifier saved 202 µs on the A100 (bar 130). Most of that came from dropping sm_80's scalar atomics: 64 atom instructions, 128 → 72 registers. No correct form keeps that.
  - Best correct form, one warp per slot: −76 µs (1.2 %).
  - The run-aligned walks: 474–555 µs against today's 466.
  - Stopped.
- **C2, a packed first radial table.** The falsifier saved 168 µs on the A100, over the bar. But certified packed slopes need 9.6k–15k pieces from 1 Å, which is 71–91 MB of table against the A100's 40 MB L2. Values alone would fit (1,402 pieces, 6.4 MB) but are worth at most the forward's 76 µs. Stopped by bound.

See `results/first-interaction-20261111/` and `tools/first_interaction_lab.py`.

**Ledger, second outside note (11 November 2026; rev2 queue, worker docs-refresh, conductor; docs only):** three new ledger rows. Nested radial basis is OPEN; the first-interaction lane owns it, and the certificate is next. Compact angular products waits on GEOM's H200 A/B. Shared denominator is stopped by the bound: about 34 µs on the A100 and 7–10 µs on the H200, under both bars. The index rows for ideas 18–20 and the first two outside ideas (C1–C5) now show today's stops, and the original estimates stay in the rows, marked "Before the test".

**BINNED_SCREEN on the RTX 4000 Ada; shuffled atom order (11 November 2026; rev2 queue B2 and B4, worker ada-5060-screen, conductor):**
- **B2, BINNED_SCREEN on the RTX 4000 Ada.** Murphy, OMAT-0 medium MD. SrTiO3 16³ 0.9846 / 0.9810, 20³ 0.9657 / 0.9602. The neighbour list is bit-identical (work removal, decision 66). The Ada is added to `pair_search.BINNED_SCREEN_GPUS`, so the switch is now on for all five GPU families.
- **B4, shuffled against ordered atoms.** Local 5060 Ti, SrTiO3 24³ (69,120 atoms): shuffled is 1.167× slower (1.157 and 1.177 pairwise). A spatial sort at setup is needed for arbitrary inputs. Ledger row `setup-spatial-sort` PROPOSED, not built.
- Suite on the branch tip (= merged tree) on murphy: 5,615 passed, 203 skipped.

**Billion-atom steps 1–2 (11 November 2026; owner request, worker tiled-speed, conductor):**
- **Step 1, 34 B/atom off the device in the tiled call.**
  - Host float64 positions (`TILE_HOST_ROWS`): opt-in per call, by passing host positions. A capacity option, not a default: A100 58³ 1.043 / 1.021.
  - uint8 species maps and no whole-system BLOCK_ORDER order: both on by default. A100 58³ A/B (decision 66, work removal): default tiled call 1.000 / 1.002, capacity call 0.982 / 0.984.
  - Gate 12/12 on Alex.
- **Step 2, overflow audit.** Tile-list sort key guarded (`KEY_LIMIT`); the overflow probe is bitwise at 1M and 750M atoms.
- Suite on the merged tree 9f39b607 (leeloo): 5,631 passed, 204 skipped. The 397.5M go/no-go (step 3) is still pending.

**Nested radial basis (11 November 2026; rev2 queue §1, worker first-interaction, conductor): REJECTED on the A100 and the Ada, kept opt-in.**
- Certificate: far rank at most 29 at 1e-7. With exactly 32 far columns the certificate stays at 0.3011.
- Timing bound: 303 µs on the A100. Real kernels in isolation: forward −137 µs, power backward −144 µs. sm_80 forward at 72 registers.
- MD A/B, both orders. A100: SrTiO3 10³ 0.984 / 0.982, NbCl5 1.057 / 1.057. The A100 split runs clear the power backward: alone it ties (0.998 / 0.996), and nested with the recompute backward also loses NbCl5 (1.063 / 1.065). Ada: SrTiO3 1.026 / 1.021, NbCl5 1.119 / 1.118.
- The loss follows edges per receiver (26 per atom on NbCl5, 74 on SrTiO3), not the far fraction (0.92 / 0.84): the split costs more per run than the skipped columns save on short runs. No size rule, because one structure sits on each side.
- Gates pass on both GPUs. `NESTED_RADIAL_GPUS` is empty.
- Kept: the switch (opt-in), the route fix (nested routing after the radial tables), and proof counters for A/Bs and gates (`ab_rewrites --b-expect`, `force_stress_gate --production-expect`). The H200 is not measured.
- Suite on the merged tip 55685b55 (murphy): 5,623 passed, 204 skipped. The 18 failures there were all in `test_colab_driver.py`, from murphy's git setup; that file passes 32/32 on leeloo for the same tree.

**A100 receiver caps 32,768 / 65,536 (11 November 2026; rev2 queue §3 B3, worker tiled-speed, conductor): ADOPTED on the A100 family.**
- `TILE_LARGE_CAPS` (None = on for `TILE_LARGE_CAP_GPUS`): a host-state tiled call with the default budget also offers receiver caps 32,768 and 65,536. An explicit budget, device state and other GPUs are unchanged.
- MD A/B against R 16,384, b/a both orders (Alex 4984555). 58³: host state 0.886 / 0.893 (32,768) and 0.549 / 0.549 (65,536); wavefront 0.675 / 0.671 and 0.410 / 0.409. 160³: host state 0.732 / 0.724 and 0.597 / 0.599; wavefront 0.722 / 0.723 and 0.573 / 0.583. Energy within 1e-5 meV/atom.
- Peak memory at 160³ (Alex 4985398), device / host: 8,192 2.83 GiB / 24.3 GB; 16,384 3.33 / 28.0; 32,768 4.20 / 31.9; 65,536 6.11 / 43.3. The device fits every cap; the host grows about 0.33 MB per receiver.
- Size rule `TILE_LARGE_CAP_HOST_BYTES` (0.35 MB per added receiver, plus 10 %, above `list_reuse_floor`). Without it, the 397.5M record (229 of 231.5 GB) would have taken 65,536 and run out of host memory.
- Also: `test_colab_driver` takes its git config from the environment (safe.directory, identity), which fixes murphy's 18 "dubious ownership" failures (32/32 there). Billion-atom plan notes (documentation only).
- Suite on the merged tip 16da79e2 (leeloo): 5,643 passed, 204 skipped, 0 failed.

**Billion-atom setup speed-ups (11 November 2026; billion-atom plan "Before step 4", worker tiled-speed, owner "go ahead" in that session, conductor): MERGED.**
- Supercell builder `tools/bench/tiled_capacity.py supercell`, tools only. Builds the cell copies in chunks with ASE's order of repeat, strain, rattle and wrap; `supercell_ase` is kept as the reference. Bitwise to ASE in the CPU tests (SrTiO3, triclinic, chunks that split copies) and at 300³. Alex 4985954: 300³ (135M atoms) 18.6 s against 206 s; 430³ 55 s; 520³ 95 s; 585³ (1.001e9 atoms) 135 s at 31.5 GB max RSS. The 519M segmentation fault came from ASE's per-copy loop.
- `TILE_FIRST_CALL_PROBE` (None = on for `TILE_PROBE_GPUS`, the A100 family; wavefront calls only, exact). Mode auto's measuring first call runs its eager sweeps on the `TILE_PROBE_TILES` = 8 largest tiles only, picks the mode and receiver cap, drops that pass's buffers and kept lists, then runs the whole call in the chosen mode. Basis: the eager workspace is set by the largest tile (526,080,512 bytes at 149M, 320M and 397.5M atoms; Alex 4973640). A/B on the go/no-go call (Alex 4985954): probe workspace 0.9986 of the whole call's at 58³ and 0.9982 at 160³; same mode, cap and reason; forces check passed; two calls at 160³ 287 s against 601 s. At 397.5M it replaces the 11,468 s eager first call.
- Suite on the merged tree 93daefd1 (murphy, idle check incl. our jobs): 5,699 passed, 204 skipped, 0 failed. Also: ledger rows tiled-first-call-probe and capacity-supercell-chunked, a defaults-by-gpu row, the plan's status table.

**Idea 13, edge forces in the w′ backward (11 November 2026; worker omat0-ada-capacity, conductor): REJECTED, kept opt-in. Capture arena: SHELVED (records).**
- `wprime_backward.EDGE_FORCE` (default None, `EDGE_FORCE_GPUS` empty): the w′ backward contracts edge_sh3's Jacobian in its epilogue and writes dL/dv [E, 3] instead of dsh [E, 16]; `EdgeSH` hands its vectors over by weak reference only when the switch is on.
- A100-80 on 8dd5f605: gate passes (SrTiO3 forces 0.90, stress 0.68; NaCl forces 0.85, stress 0.82). MD step b/a SrTiO3 10³ 1.0302 / 1.0313, NbCl5 1.0242 / 1.0212: slower in all four. 36³ steady reserved 14.42 → 16.12 GiB.
- Cause: the w′ dsh is gone (−312 MiB at 24³), but AOT autograd's zero gradients for the compiled region's vector and edge_attrs outputs add 76 B/edge, and interaction 0 keeps edge_sh3's backward. Ceiling −52 B/edge (about −0.8 GiB at 36³) at a 2–3 % step cost. Reopen conditions in the ledger row.
- Capture arena: stage 1 on Alex (4982478) planned 11,706 MiB against 11,706 MiB reserved, so nothing to overlap; torch 2.14's blocks_by_addr assert on overlapping segments. Records and tools only (`tools/bench/arena/`); the code stays on `fastmace/capture-arena`.
- Suite on the merged tip 25d5d2eb (leeloo): 5,706 passed, 204 skipped, 0 failed.
- Records (2a5107c7, Alex 4985845): with EDGE_FORCE on, steady reserved is +12 % at every passing size (36³ 14.42 → 16.12 GiB, 42³ 22.49 → 25.20, 49³ 35.31 → 39.69; step at 49³ 881 → 1,057 ms). Both sides pass 49³ and time out at 53³, so no capacity change can be read.

**H200 packs: A100 switches never measured on the H200 (11–12 November 2026; owner "i approve the h200 pack" and "run it now"; conductor, decision 66): six switches on for the H200.**
- OMAT-0 medium, Benchmark_loup MD step b/a in both build orders (SrTiO3 10³, NbCl5 3³): `tf32x3.HYBRID` 0.9909 / 0.9940, 0.9892 / 0.9885; `spline_forward.GEOM` 0.9857 / 0.9890, 0.9891 / 0.9860; `fused_forward.WFRAG_REGION` (bitwise, re-screened; a tie on 6 November) 0.9877 / 0.9921, 0.9695 / 0.9913. Gates pass for HYBRID (forces ≤ 0.99, stress ≤ 1.04) and GEOM (forces ≤ 0.95, stress ≤ 1.30).
- `TP_NOFLUSH`: 0.9968 / 1.0030, 0.9397 / 1.0038 (the ab baseline ran 5 % slow): mixed, stays off.
- MACE-OFF: radial tables with the stock first block, `STOCK_FIRST_SKIP` and `RADIAL_TF32X3` together, OFF23-medium water_216 0.7417 / 0.7422 (2.43 → 1.80 ms); the gate passes on urea, naphthalene, paracetamol and water_216 (candidate forces ≤ 1.25, stress ≤ 1.51). Measured on OFF23-medium only; taken for the family because the A100 found all five OFF models faster.
- GEOM winning on the H200 keeps compact angular products unbuilt (rev2 queue §4).
- Modal cost: pack 1 about $1.19 (sixth), pack 2 $1.76 (fourth, 20-minute timeout), the OFF candidate gate $0.82 (third). See `results/h200-pack-20261111/README.md`.
- Suite on the branch tip 66cf1ae6 (murphy, idle check incl. our jobs): 5,706 passed, 204 skipped, 0 failed.

**H200 w′ instruction tuning (11–12 November 2026; owner "Can you try the search space and instruction tune on h200?" and "Go ahead" for one H200 job; conductor, decision 66): `wprime_backward.SKIP_OUTSIDE` and `HOIST_DX` on for the H200, together only.**
- Formulation search closed: the best per-path mix saves 45 FP instructions per edge, about 22 µs (0.66 % of the 3.369 ms step), under the 1 % gate; the complex basis and the h-form are worse.
- sm_90 census per edge and lane: base 2,198 instructions (72 registers); skip 2,026 (96), hoist 2,141 (80), skip+hoist 1,990 (96), vec2 2,046 (128), skip+vec2 1,864 (128), vec4 2,489 (141), skip+vec4 1,909 (167); no spills.
- H200 lab (OMAT-0 medium, SrTiO3 10³): skip+hoist 0.973; skip alone 1.021, hoist alone 0.999, so the two share one GPU set and HOIST_DX's default needs SKIP_OUTSIDE. `CHANNEL_VECTOR` lost (1.082–1.606) and stays off.
- MD step b/a both orders: SrTiO3 10³ 0.9889 / 0.9926, NbCl5 3³ 0.9823 / 0.9811. Gate (NaCl, SrTiO3 3³) passes: forces 0.86, 0.88; stress 0.81, 0.46. NbCl5 not in the gate. Rounding only through the compiler's FMA order for the inside edges (dsh 5.5e-10, dr 6.9e-8 relative); accepted by the conductor.
- A100 (Colab, H200 plan via the alias): skip+hoist 1.025, skip 0.997: off.
- Measured on OMAT-0 medium only; the GPU set turns it on for every H200 w′ with TABLE_W (open cell in defaults-by-gpu).
- Modal: one H200 job, 547 s, $0.79 metered. See `results/wprime-h200-20261111/README.md`.
- Suite on the branch tip 45efd9de (murphy, idle check incl. our jobs): 5,714 passed, 204 skipped, 0 failed.

**w′ late pullback (12 November 2026; outside review, conductor; owner "i approve" for one H200 job beyond the Modal cap and "launch now"; decision 66): `wprime_backward.LATE_PULLBACK` on for the H200.**
- Design: every path in N1's Cartesian form (q = M g staged by the STAGE_DOUT prologue). Each lane sums only G = Σ w ∇φ and R = Σ (dw/dt) φ, and the geometry is differentiated once per edge after the channel reduction. The vector paths 1⊗2→1 and 1⊗3→2 use the unit-sphere extension (no Euler term). G/√3 goes into dsh's l = 1 slots and R into dr, so the outputs and the autograd graph are unchanged (`wprime_pullback.py`, a post-render transform before HOIST_DX).
- CPU algebra: float64 6.6e-16, float32 as the direct contraction. sm_90 census per edge 2,026 → 1,826 (0.901), 96 → 56 registers; sm_80 2,010 → 1,799, 128 → 64.
- H200 (OMAT-0 medium, on top of SKIP_OUTSIDE + HOIST_DX): lab 0.859 (1,010.9 → 868.9 µs); MD step b/a SrTiO3 10³ 0.9597 / 0.9612, NbCl5 3³ 0.9848 / 0.9825; gate passes (NaCl, NbCl5, SrTiO3 3³: forces 0.87, 0.94, 0.88; stress 0.85, 1.30, 0.54 of the ASE float32 spread). Rounding (dL/dv 7.6e-7).
- Per-degree hybrids (16 masks) are all dominated by the full form; the irrep-first loop nest costs +31 % instructions: both closed.
- Measured on OMAT-0 medium only; the GPU set turns it on for every H200 variant and every model whose w′ paths it covers (open cell in defaults-by-gpu). A100 off until its A/B (Alex 4988692).
- Modal cost: lab $0.06 (second), gate + A/B $1.01 (fifth, owner override).
- Suite on 082c656c (coraline, idle check incl. our jobs): 5,726 passed, 206 skipped, 0 failed; the tip 8ed2611d adds one line to test_wprime_pullback.py (standalone: 12 passed, 2 skipped). See `results/wprime-late-pullback-20261112/README.md`.

**w′ WEIGHTED_VECTOR and SKIP_OUTSIDE + HOIST_DX on the A100 (12 November 2026; conductor, decision 66): on for the A100 family.**
- `wprime_backward.WEIGHTED_VECTOR`: a path with more than 3 edge sums that is not on SCALAR_PATHS adds `s (w x_i)` per term to dsh, instead of summing v_j and adding `w v_j` (outside review idea 1). Rounding. The template equals the conductor's `wxin_variant.py` on the H200 and A100 renders.
- MD step b/a on Alex A100-SXM4-80GB, the A100's own default plan, OMAT-0 medium, both build orders, SrTiO3 10³ / NbCl5 3³:
  - SKIP_OUTSIDE + HOIST_DX + WEIGHTED_VECTOR: 0.9742 / 0.9730, 0.9710 / 0.9701
  - SKIP_OUTSIDE + HOIST_DX: 0.9849 / 0.9828, 0.9751 / 0.9751
  - WEIGHTED_VECTOR alone: 0.9950 / 0.9914, 0.9971 / 0.9950
  - The 3-structure gate (SrTiO3 3³, NaCl, NbCl5) passes for each. Jobs 4988118 and 4988246.
- Census correction: the A100's own plan (STAGE_EDGES on) already has skip+hoist at 96 registers on sm_80. The 128 → 96 drop applies only to the H200 plan's render. The A100 lab's skip+hoist 1.025 ran the H200 plan through the GPU alias. The A100 gain is therefore instruction count, not occupancy.
- `SKIP_OUTSIDE_GPUS` (shared by `HOIST_DX_GPUS`) and `WEIGHTED_VECTOR_GPUS` gain the A100 family (SXM4-80GB, SXM4-40GB, PCIe through GPUModels' family match).
- WEIGHTED_VECTOR is off whenever LATE_PULLBACK is on (`weighted_vector_on`): LP's block parser is not checked against weighted sources. The A100 is not in `LATE_PULLBACK_GPUS`, so this changes nothing today.
- H200: census only (1,990 → 1,957 instructions per edge, about 0.5 % of the step); WEIGHTED_VECTOR stays off there.
- Measured on OMAT-0 medium only. Other families on the A100 get the switches through the GPU sets; this is an open cell in defaults-by-gpu.
- Suite on the branch tip e907e9dc (murphy, idle check incl. our jobs, 13:27 UTC): 5,730 passed, 206 skipped, 0 failed.
- See `results/wprime-h200-20261111/README.md` Part 5.

**Paper experiment 1: packed forward consistency (12 November 2026; owner "yes go ahead"; conductor): records and tools only, no default changes.**
- Question: the packed forward (`PACKED8`) and the cubic backward are not exact derivatives of one another. Does that inconsistency show up in forces or in MD energy conservation? (a) defaults, (b) PACKED8 off, (c) autograd control (A100).
- Probes (5060 Ti, A100): finite-difference gradients agree to about 3e-5 of |F|, closed-loop work about 1e-8 eV, knot-line jumps about 2e-5 eV: the float32 floor for (a), (b) and (c).
- NVE drift, SrTiO3 10³: the run-to-run SD is 3.7e-6 meV/atom/ps at 300 K (7 runs, 100 ps, three GPU families) and 2.8e-5 at 1500 K (8 runs, 50 ps). The AR(1) standard errors understate this scatter about 4×. (a), (b) and (c) are indistinguishable. The one outlier, Ada 1500 K (a) at −4.8e-5, gave −7.5e-6 when rerun from the same state.
- Cost of consistency (PACKED8 off), MD step b/a for the ab / ba orders, SrTiO3 then NbCl5: A100 1.083 / 1.080 and 1.057 / 1.053; 5060 Ti 1.102 / 1.102 and 1.064 / 1.066; RTX 4000 Ada 1.125 / 1.119 and 1.075 / 1.072.
- Tools: `tools/run_manifest.py`, `tools/consistency_probe.py`, `tools/nve_energy.py`. Suite on c0b14126 (murphy, idle check incl. our jobs): 5,730 passed, 206 skipped, 0 failed.
- See `results/paper-exp1-20261112/README.md`.

**COMPOSE_FAST's weights composed outside the captured step (12 November 2026; conductor, after torch-215-trial §6a; decision 66, work-removal rule): bitwise work removal, no switch, every GPU.**
- Cause: the graph engine's capture restores its registry snapshot right before `torch.cuda.graph`. The `copy_` moves the species table's version, so `FirstInteractionReassociation.bind` (called in the captured region by `ForceField._bind_streaming_count`) recomposed and the four float64 compose GEMMs (`compose_skip.composed`) were replayed every step for constant inputs.
- Change: a bind inside a capture keeps a known table's cached weights when only the table's version moved (`refresh_composed(adopt=True)`, proof counter `mh1.modules.CAPTURE_ADOPTS()`). The force field registers each table's composed weights as derived state (`species_feats.interactions.<i>.composed_<k>`, refreshed by the table's refresh), so every snapshot restore (capture warm-up, rejected transaction) brings table and weights back together. The cache also records the versions of the radial last layer, the linear, the skip and the nested rotation: a reload or an optimiser step recomposes (before: never noticed outside a capture). A new table or a nested change still composes.
- Ada (RTX 4000 Ada, OMAT-0 medium float32, production plan): MD step b/a SrTiO3 10³ 0.9873 / 0.9823, NbCl5 3³ 0.9273 / 0.9341 (same-process, side a = main's bind via `ab_swap.py`); the 2.15 attribution shows the CUTLASS class (301 / 278 µs per step) and every compose site gone. Smoke: no composition inside a capture, composed weights bitwise equal to main's.
- A100 and H200: covered by decision 66's work-removal rule (AGENTS.md idea triage step 2: a bitwise pure work removal is decided by one not-slower A/B on any GPU; conductor, 12 November 2026), not measured. The capture mechanism is the same on every GPU where COMPOSE_FAST is on, so both replayed the compose. Estimate only (not measured): on the H200, where float64 runs on tensor cores, about 25–70 µs per step from the removed graph nodes (about 37 per step on the Ada), i.e. b/a about 0.98–0.99 on SrTiO3 10³ (3.07 ms) and 0.93–0.975 on NbCl5 3³ (1.02 ms).
- Suite on `381ac910` (leto, idle check): 5,738 passed, 206 skipped, 0 failed. See `results/compose-outside-graph-20261112/README.md`.
