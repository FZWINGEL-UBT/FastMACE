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
2. **General.** Any crystal, composition, cell and GPU, with no per-GPU or
   per-structure tuning tables, shape-keyed caches or one-GPU kernel schedules
   (project rule, 10 October 2026). Every optimisation must hold for other
   crystals, melt–quench protocols and atom exchanges.
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
| Python | 3.10.12 | 3.10–3.12 |
| torch | 2.14.1+cu130 (2.5.1+cu124 reference stack) | each release pins and CI-tests exact builds of the two newest torch minor releases at release time (CUDA 12.x/13.x wheels); other versions run only the rewrites whose self-checks pass |
| MACE (`mace-torch`) | 0.3.16 | per-release rewrite registry (§6.3); unknown versions run the reference path |
| cuEquivariance (+ ops) | 0.12.0 (0.6.1 older stack) | ≥ 0.12 |
| OpenEquivariance | 0.7.0 | optional extra `[oeq]`; ≥ 0.7 |
| e3nn | 0.4.4 | as MACE requires |
| ASE | 3.27.0 | ≥ 3.23 |
| graph-longrange | 0.4.0 (MACE 0.3.16 calls this interface) | extra `[polar]`, pinned per MACE version |
| Triton | as shipped with torch | as shipped with torch |
| torch-dftd | 0.5.1 | extra `[d3]`: the `torch-dftd` reference (§12.1) and the ASE front end; not needed by the native GPU D3 path (§11.2) |
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

Support is declared per engine and integrator; the planner refuses anything
outside this table at plan time, before any compilation.

| Feature | Tiers 1–3 (force field) | Integrators and optimisers | Today |
| --- | --- | --- | --- |
| Periodicity | full 3D PBC only (the force wrapper and the polar path require it) | — | `MACEForces` and `polar_gpu.enable` raise without 3D PBC |
| Non-periodic or mixed PBC (molecules, slabs) | not in 1.0; reference tier only | ASE-driven | — |
| Fixed centre of mass | — | Langevin (ASE `fixcm`), BAOAB, NVE, CSVR, NHC, MTK | Langevin only |
| `FixAtoms` | — | all 1.0 integrators and optimisers (phase 4); MC moves with constraints refused in 1.0 (owner decision 10, 17 October 2026) | both GPU integrators raise on any constraint (`gpu_langevin.py:2136`) |
| Other ASE constraints, RATTLE | — | refused in 1.0; RATTLE in 1.x | — |
| Variable cell | stress path | BAOAB, MTK; FIRE/LBFGS with a cell filter | BAOAB |

Degrees of freedom for temperatures follow the active constraints (today
`3N`, which is correct only without constraints).

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
    exactness: Literal["exact", "rounding", "approximate"]
    default: Literal["on", "off", "auto"]
    workload_constraints: tuple[str, ...]  # e.g. ("fixed_species_superset", "single_graph")
    apply: Callable[[Model, Context], None]
    selfcheck: Callable[[Model, Model, Context], CheckResult]   # vs original module, at setup
```

**Defaults.**
- `"exact"` means bit-for-bit identical math up to summation order.
- `"rounding"` covers reassociation, folded or factored weights and changed
  GEMM grouping; most adopted rewrites are of this kind (§6.2 column).
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
| `streaming_conv` (OEq PR 223 kernels, NVRTC) | exact (summation order; atomic sums, not bitwise reproducible) | folded MH-1 class, `GENERIC_*` with OEq fusion | −7.7% NaNbO3; generic −16 to −28% with reassociation | OEq present; per-engine edge counter (§8.1) |
| `generic_species_skip` | exact (absent columns multiply zeros) | MP-0/density family | −7 to −39% | fixed species superset |
| `generic_fused_radial_backward` | rounding | MP-0/density family | −3 to −5% large cells | — |
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
answers); copy their summary into `docs/rewrites.md` so the work is not
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
  the reduction. The gradient of a sum is 1 for every term whatever the
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
  0.02/fs, trajectory and thermo every 20 steps, restart every 10,000,
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
  device generator; pass `seed=`), and `communicator` other than ASE's
  default (FastMACE runs one process per structure). The other classes list
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
  each would invalidate the plan. Observers every few steps make the
  planner choose the compiled tier (rule 9), and the report says so.
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
  path); frequent host callbacks (compiled tier); a failed rewrite
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
this tier is refused.

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
short calibration (`plan(calibrate=True)`: about 50 steps per candidate tier,
results never persisted across runs) may break ties, but no per-GPU or
per-shape table is ever stored. This is the project's generality rule.

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
decision, and the overridden plan is self-checked and gated like any other.
Rules 13–15 are **reactions** to checks.

| Order | Kind | Condition | Decision |
| --- | --- | --- | --- |
| 1 | hard | workload outside §4.4 (PBC, constraints, MC moves with constraints), or a variable cell with an integrator that has no barostat | refuse; reference tier only with `allow_reference=True` (never for MC with constraints) |
| 2 | hard | a species outside the ForceField's declared superset, in the structure or in a declared move set (swap pairs, transmutable sets) | refuse |
| 3 | hard | PolarMACE with a plain compiled backend | refuse that combination (zero forces); hybrid, or eager plain |
| 4 | hard | `padding_is_exact` false | no graph mode (padded edges would change energies); compiled with exact edge counts |
| 5 | hard | `reproducible=True` | only a qualified (backend, model family) with its deterministic allow-list (see below); refuse otherwise |
| 6 | hard | cuEquivariance's fused kernels do not load (§9.2) | refuse unless `allow_naive_cueq=True` |
| 7 | hard | `atom_count_changes=True` | refuse: not in 1.0 (§8.6, §17) |
| 8 | preference | `species_changes`, graph mode still allowed | graph mode; species-dependent rewrites built for the superset |
| 9 | preference | `host_callbacks_every` small (< ~10 steps) | compiled mode (graph replays would be broken up anyway) |
| 10 | preference | the integrator or optimiser and its cell mode (§9.2) | a barostat integrator or a cell-filtered optimiser (FIRE/LBFGS, never an NPT integrator for a relaxation) enables the stress path and the deformation-aware neighbour test; polar: k-vector headroom |
| 11 | preference | `max_temperature_K` high; `"displace"` in `mc_moves`; D3 on | larger skin and bucket headroom (scaled from expected displacement per rebuild interval); a skin that keeps displacement trial rebuilds rare (§10.4); the D3 list's own skin (§11.2); re-capture budget |
| 12 | preference | GPU memory insufficient for estimated edge bucket × feature width | drop memory-heavy rewrites or warn (e.g. generic-only POLAR-1-L NPT on NbCl5 ran out of memory on 20 GB) |
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
error) triggers this algorithm, not a fixed "one tier down":
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
   the planner refuses with the full record. Mid-run fallbacks (§9.6) follow
   the same steps between segments, from the last committed state.

### 9.5 Output

`Plan` is a frozen object: tier, enabled and disabled rewrites with reasons,
bucket and skin settings, kernel availability, the fallback candidates and
predicted memory. `str(plan)` is human-readable, and the plan is written into
provenance (§13). Users can override the preferences of §9.3 (rules 8–12),
never the hard constraints (rules 1–7) except through their `allow_*` flags;
every override is self-checked and gated.

A plan never changes in place. A runtime switch (§9.6) or a fallback creates
a new frozen **revision** with its step, reason and predecessor. Provenance
keeps the whole chain, and the restart file stores the active revision.

### 9.6 Runtime adaptation

During a run the planner monitors:
- re-capture count;
- neighbour rebuild frequency and build time, per list (MACE, D3);
- bucket overflow;
- non-finite observables.

It may switch tiers between segments (never mid-step), as a new plan
revision (§9.5) with a logged event.
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
  corrections of §12.1), MACE and FastMACE versions.
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
force-field identity. A different plan (another GPU or rewrite set) needs
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
  `cutoff = 40 * ase.units.Bohr`, which `torch-dftd` converts back with its
  own constant: 39.9999963 bohr.)
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
| Energy, force and stress gate | CI (GPU), `fastmace check` | today's rule: production path's max error vs float64 ≤ 2 × the float32 calculator's error, now for energy per atom (float32 yardstick with float64 accumulation, §12.1) as well as forces and stress; rattled and strained cells, hydrostatic and full; all-electron and pseudopotential E0 models; move-energy differences for MC plans (§12.3) |
| Replay and transitions | CI (GPU) | the production path as executed: captured replay, pipelined and speculative steps; across a neighbour rebuild, a capacity overflow with re-capture, a cell deformation past the rebuild threshold, an accepted and a rejected species change, and a restart, each compared with a fresh direct evaluation of the same state |
| Mutation events | CI (CPU, GPU) | derived-state registry against fresh construction (§8.5); first MD step after accept and after reject against fresh construction; a superset species with no atoms (its table row from the identity basis, not from atom 0), the last superset species absent (sender-group count), an atom outside the superset (explicit error); a rejected displacement leaves the neighbour list of the committed positions; a displacement beyond skin/2 (> 0.15 Å at the default skin) is evaluated on a list built eagerly around the trial geometry, accepted and rejected, without looping, and its ΔE equals a direct evaluation; an overflowing displacement is redone with the same random numbers; the first NPT step after an accepted move equals a fresh construction (`force_eps` refreshed), hydrostatic and full cell; a ramp step changes the target without an event (no quiesce, §8.5) |
| Runtime state | CI (GPU) | runtime-state registry against the attributes each stage mutates; full registered state identical before and after every capture warm-up, sticky flags included (§8.1) |
| MC exact enumeration | CI (CPU tiny model; GPU) | a small cell at fixed positions (e.g. 8 sites, 4 + 4 atoms: 70 configurations; transmutation: 256), all configurations evaluated with the reference: per-move acceptance probabilities against the reference ΔE (including the first attempt after MD, whose baseline comes from the MC evaluator), and sampled populations against Boltzmann weights (χ² test, fixed seed). Hybrid MD/MC: kinetic temperature with MC on and off agrees for `exchange`, `rescale`, `redraw` and (with `fixcm`) `redraw_all`, and the total momentum stays zero under `fixcm`; a negative control with unchanged velocities must fail. Charge/spin: a set violating the parity rule is refused (§10.4) |
| Restart | CI (CPU, GPU) | §10.6: round trip; interrupted vs uninterrupted (same restart cadence) across accepted and rejected swaps, a partly consumed noise batch, a variable-cell run with h ≠ h₀ at the restart step, a count-overflow redo before it, and a D3 list; a crash between a restart write and the next output flush leaves the resumed run's trajectory and thermo files without duplicate or missing frames and rows (chunks deleted, nothing rewritten) |
| Reproducibility | CI (GPU) | in reproducible mode, repeated runs and restarts bit for bit (§9.3) |
| Isolation | CI (GPU) | A → B → A with two crystals for the plain, MH-1 and polar paths (§8.7) |
| Fallback | CI (GPU) | deliberately broken rewrite and broken capture: eager diagnosis, bisection, pristine rebuild, validated activation (§9.4) |
| ASE equivalence | CI (GPU) | Langevin: same seed reproduces ASE's noise stream (NaCl 9 ps: same T and E_pot in every 50 K bin, \|ΔPE\| < 0.01 meV/atom); BAOAB: the ASE variant (`ASE_TRACE_DRAG_ALL_ELEMENTS=True`) vs ASE `LangevinBAOAB` exactly, and the corrected production variant vs an independent corrected reference (§12.1); new integrators vs reference implementations |
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

Every run writes `metadata.json` with:
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
  BSD-3 licence and notice. The cuEquivariance ops wheels are NVIDIA-licensed:
  depend on them, do not vendor them. Model weights are never redistributed;
  `fm.load` downloads from official sources and shows the model licence.
- **Extras:** `fastmace[oeq]`, `fastmace[polar]` (graph-longrange pinned per
  MACE version), `fastmace[d3]` (the `torch-dftd` reference and the ASE front end; the native D3 path does not need it), `fastmace[dev]`,
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
    environments.
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

Each phase ends with its acceptance criteria met on the RTX 5060 Ti and one
workstation GPU.

| Phase | Content | Acceptance | Estimate (1 developer) |
| --- | --- | --- | --- |
| 0. Skeleton | repo layout, packaging, CI (CPU), licence, port CPU tests | `pip install -e .`; CPU suite green | 1 week |
| 1a. Extraction | `model`, `rewrites` (registry with today's catalogue), `neighbors`, `state`, `engine.graph`, `integrators.langevin`/`baoab`, behind the planner interface with one fixed plan (today's `main.py` defaults); no globals, no NAS or experiment imports, no wrapper script | against a frozen research-engine revision on the same machine and configuration, in alternating A/B blocks: steady-state step time within 3%, setup time and peak memory reported; today's force/stress gate passes for all 19 labels of §12.5 (ten benchmark models plus POLAR-1 S/M/L, OMOL-0 XL and five MACE-OFF, the last on the organic set); `Experiments/Langevin_Thermostat/main.py` runs on top of FastMACE | 3–4 weeks |
| 1b. Contracts | rule-based minimal planner (hard constraints, mode and backend choice, `Plan` record with revisions; no runtime adaptation); `Simulation` taking a `ForceField`, the reduced `Workload` and the `run` checks (§7.6, §9.2); float64 energy accumulation (§7.3); runtime-state registry and complete capture restore (§8.1); complete state and restart with canonical restart points and output chunks at restart points (§10.6); reference policy (§12.1) with the BAOAB integrator correction qualified against an independent corrected reference; gate with energy, floors and captured replay (§12.3); exceptions and checkpoint trust (§7.6); adoption layer (§7.7): `fm.md` and `RunResult`, the run report and error-message shape, `fm.calc`, `fastmace.ase` `Langevin` and `LangevinBAOAB` with observer sync, CLI `md`, `init`, `doctor`, `models`, `fastmace[full]` | gate passes for all 19 labels of §12.5, energy included (POLAR-1 and OMOL-0 energies within the gate instead of 45–102 meV/atom off, MACE-OFF instead of 0.1–12 meV/atom); registry and capture-state CI checks pass; restart round trip, interrupted vs uninterrupted with the same cadence (bitwise on CPU float64, statistical for hybrid); a crash between a restart write and the next flush resumes without duplicate frames; the §7.7 acceptance tests for Langevin, BAOAB and the CLI | 4–6 weeks |
| 2. Safety | self-checks, gate statuses and resource policy (§12.3), fallback algorithm with ddmin, diagnosis budget and terminal sequence (§9.4), subprocess runner through a restart write (§8.7), versioned registry, instance isolation (§8.7), provenance schema, GPU CI; qualification of `reproducible=True` for one family (plain cuEq, MH-1) | unknown MACE version runs the reference path cleanly; a deliberately broken rewrite, a broken pair of interacting rewrites and a broken capture each end in the correct validated fallback within the budget, and an exhausted budget ends in the terminal sequence; A → B → A passes; CI on two GPU architectures; reproducible mode bitwise on repeated and restarted runs, or recorded as not qualifiable with the blocking kernel named | 3–4 weeks |
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
2. Minimum MACE version to support beyond 0.3.16.
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
4. Licence — **decided (user, 17 October 2026): MIT** (§14).
5. Which GPUs to provide as CI runners (local RTX 5060 Ti, a workstation
   RTX 4000 Ada).
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
