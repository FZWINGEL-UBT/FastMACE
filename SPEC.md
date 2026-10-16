# FastMACE — Specification

Status: draft 0.2, 16 October 2026. Owner: Fabian Zwingel. Written from the
Langevin_Thermostat experiment (`Experiments/Langevin_Thermostat/`), whose GPU
engine this library generalises. Draft 0.2 answers a review of draft 0.1
and records the owner's decisions 6 and 7 (§18, §19).

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
| Insertion/deletion (GCMC), evaporation | atom count changes | compiled tier with atom buckets |
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
| `gpu_langevin.py` | 2,568 | Force wrapper `MACEForces`, all rewrites, skin neighbour list (GPU build, in-graph compaction), padded edge buckets, CUDA-graph Langevin engine, pipelined steps, async output | `fastmace.model`, `fastmace.rewrites`, `fastmace.neighbors`, `fastmace.engine`, `fastmace.integrators.langevin` |
| `gpu_npt.py` | 292 | ASE `LangevinBAOAB` on the GPU (hydrostatic/full cell) | `fastmace.integrators.baoab` |
| `polar_gpu.py` | 658 | PolarMACE on the GPU: capturable k-space, stress fix (MACE issue 1642), PR 1711 uvu GEMM, PR 1762 port | `fastmace.longrange.polar` |
| `edge_outer_sum.py`, `skinny_linear.py`, `fused_radial.py`, `inplace_linear.py` | ~570 | Triton kernels | `fastmace.kernels.triton` |
| `oeq_streaming/` | vendored | OpenEquivariance PR 223 receiver-streaming kernels via NVRTC (BSD-3, LBNL; keep LICENSE and NOTICE) | `fastmace.kernels.oeq_streaming` |
| `pair_search.py` | 102 | GPU pair search | `fastmace.neighbors` |
| `quench.py` | 249 | Frame selection and quench relaxations | `fastmace.protocols.quench` |
| `workflow.py` | 1,422 | CLI, protocol profiles, provenance, resume | split: `fastmace.cli`, `fastmace.protocols`, `fastmace.provenance` |

Tooling that becomes the validation suite:
- `tools/validation/npt_stress_reference.py` (force/stress gate);
- `tools/validation/roundoff_reference.py`;
- `studies/other_models/probes/model_step_time.py` (benchmark);
- `studies/further_optimizations/probes/step_breakdown_probe.py` (profiler);
- `tests/` (294 CPU tests).

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

  Each must be encapsulated or documented as a process constraint and enforced
  by the planner.
- **Paths and environment.**
  - NAS paths (`/mnt/homes/zwingel/...`) and the `benchmark_scripts_path()`
    import of `Benchmark_MACE_GPU_OPTIMIZATIONS/scripts/common.py` (model
    table, `load_calculator`).
  - The `oeq_no_nvcc.sh` launcher and the `main.py` refusal without it: OEq
    0.7 lock-file workaround, runtime headers, per-process extension directory,
    `OEQ_EXTENSION_SEED`.
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
| torch | 2.14.1+cu130 (2.5.1+cu124 reference stack) | latest two minor releases with CUDA 12.x/13.x wheels |
| MACE (`mace-torch`) | 0.3.16 | per-release rewrite registry (§6.3); unknown versions run the reference path |
| cuEquivariance (+ ops) | 0.12.0 (0.6.1 older stack) | ≥ 0.12 |
| OpenEquivariance | 0.7.0 | optional extra `[oeq]`; ≥ 0.7 |
| e3nn | 0.4.4 | as MACE requires |
| ASE | 3.27.0 | ≥ 3.23 |
| graph-longrange | 0.4.0 (MACE 0.3.16 calls this interface) | extra `[polar]`, pinned per MACE version |
| Triton | as shipped with torch | as shipped with torch |
| torch-dftd | 0.5.1 | extra `[d3]` until the native GPU D3 lands (§11.2) |

### 4.2 Hardware

- **Qualified:** RTX 5060 Ti (sm_120, Blackwell), RTX 4000 Ada (sm_89),
  workstation GPUs on coraline and leeloo.
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
| Anything else | — | reference tier (MACE calculator), or the compiled plain path if the padding check passes |

**Model-level known issues** (surfaced as warnings by the planner, not as engine
failures):
- compiled plain cuEquivariance PolarMACE returns zero forces, so it is refused;
- POLAR-1-S has a Fukui charge runaway on NaNbO3;
- all POLAR sizes give huge forces and pressures on NaTaO3;
- OMOL-0 is outside its training domain on dense oxides and collapses in NaCl
  MD.

**Licences.** Model licences must be shown at load time: OMOL-0 is under the
ASL licence (non-commercial).

### 4.4 Boundary conditions and constraints

Support is declared per engine and integrator; the planner refuses anything
outside this table at plan time, before any compilation.

| Feature | Tiers 1–3 (force field) | Integrators and optimisers | Today |
| --- | --- | --- | --- |
| Periodicity | full 3D PBC only (the force wrapper and the polar path require it) | — | `MACEForces` and `polar_gpu.enable` raise without 3D PBC |
| Non-periodic or mixed PBC (molecules, slabs) | not in 1.0; reference tier only | ASE-driven | — |
| Fixed centre of mass | — | Langevin (ASE `fixcm`), BAOAB, NVE, CSVR, NHC, MTK | Langevin only |
| `FixAtoms` | — | all 1.0 integrators and optimisers (phase 4) | both GPU integrators raise on any constraint |
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
  planner/
    workload.py          WorkloadSpec: what the user's run will do (§9.2)
    plan.py              Planner, Plan, decision rules, fallback algorithm (§9.3, §9.4)
    selfcheck.py         rewrite and gate checks on the user's structure (§12.3)
  reference/             versioned reference records and corrected reference calculators (§12.1)
  longrange/
    polar.py             PolarMACE GPU path (§11.1)
    d3.py                GPU D3 dispersion, all parameters explicit (§11.2)
  io/
    trajectory.py        async writer, ASE Trajectory/extxyz, every-n-steps, chunked large files
    restart.py           checkpoint/restart format (§10.6)
  provenance.py          run metadata (§13)
  protocols/             heat/cool profiles, anneal, quench, melt–quench (from workflow.py profiles)
  ase/calculator.py      FastMACECalculator (ASE drop-in)
  cli/                   `fastmace run|relax|bench|check|info`
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
- `"rounding"` covers reassociation and changed GEMM grouping; this is all of
  today's adopted rewrites.
- `"approximate"` rewrites are never default.

### 6.2 Catalogue for 1.0 (all adopted today, defaults as in `main.py` on 16 October 2026)

| Rewrite | Applies to | Gain (measured) | Constraint |
| --- | --- | --- | --- |
| `factor_species` (species-factored radial input) | MH-1 class | part of the MH-1 package | — |
| `fold_cutoff` (cutoff folded into the last radial layer) | MH-1 class, `apply_cutoff=False` | — | — |
| `sort_receivers` | all | 3–5% large crystals | — |
| `density_first` (division before `linear_1`) | MH-1 class | ≈11% with `reassociate_first` | bias-free `linear_1` |
| `reassociate_first` (first interaction without per-edge TP weights) + Triton backward | MH-1 class, `GENERIC_*` | MH-1 package; OMOL −9% NbCl5 | species-only first inputs: linear node embedding, at most per-graph joint embeddings, single graph, fixed species superset |
| `segmented_first_sum` | with `reassociate_first` + sorting | 1.6% NaNbO3 | — |
| `joint_radial_input`, `fused_folded_backward`, `gpu_neighbour_list` | MH-1 class (folded) | 6–15% together | — |
| `block_linear` | cuEq `Linear` (internal shared weights) | 11.8% NaCl (MH-1) | — |
| `grouped_linear` | with `block_linear` | MH-1 −2 to −5%, OMOL −4% | — |
| `merged_residual` | with `block_linear` + `density_first` | included above | — |
| `streaming_conv` (OEq PR 223 kernels, NVRTC) | folded MH-1 class, `GENERIC_*` with OEq fusion | 7.7% NaNbO3; generic −16 to −28% with reassociation | OEq present; edge counter binding |
| `generic_species_skip` | MP-0/density family | −7 to −39% | fixed species superset |
| `generic_fused_radial_backward` | MP-0/density family | −3 to −5% large cells | — |
| `polar_uvu` (PR 1711) | PolarMACE | 4–5× with MH-1 rewrites | — |
| `pipelined`, `async_output` | engine features (§8.1) | 3.0% NaCl; bit-identical | — |
| `unfuse_full_addmm`, `first_derivative_only`, expandable segments allocator | engine | avoids 2 × 32 MB copies per step | torch ≥ 2.14 |

Opt-in only:
- PR 1762 Fukui regularisation and charge-dipole cap (changes the model);
- in-place block linear (slower);
- fused radial hidden (no gain).

Never enable:
- Inductor programmatic dependent launch (wrong forces inside CUDA graphs);
- spin restriction (changes the model, worse).

The experiment's markdown files record the rejected screens (no gain or wrong
answers); copy their summary into `docs/rewrites.md` so the work is not
repeated.

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
sim = fm.Simulation(
    atoms, model,
    integrator=fm.integrators.Langevin(timestep_fs=0.5, temperature_K=300, friction_per_fs=0.02),
    workload=fm.Workload(species_changes=True, atom_count_changes=False, cell="fixed",
                         host_callbacks_every=1000),
    output=fm.Output("run/", trajectory_every=20, thermo_every=20, restart_every=10_000),
    seed=1,
)
print(sim.plan)                     # tier, rewrites, reasons, fallbacks (§9)
anneal = fm.protocols.ramp(0, 1000, ps=20) + fm.protocols.ramp(1000, 250, rate_K_per_ps=-3)  # 270 ps
sim.run(schedule=anneal)                         # whole schedule: 540,000 steps at 0.5 fs
sim.run(schedule=anneal, steps=200_000)          # first 200,000 steps, then pause (resumable)
sim.run(steps=100_000, mc=fm.mc.Swap(pairs=[("Na", "Nb")], every=100, attempts=50,
                                     velocities="rescale"))   # §10.4
sim.save_restart("run/restart.npz"); sim = fm.Simulation.resume("run/")
```

**Schedules and step counts.** A schedule fixes the run's length. `run`
without `steps` runs it to the end. With `steps`, it advances that many steps
along the schedule and pauses; the next `run` (or a resume) continues from
the recorded schedule position. `steps` beyond the remaining schedule raises
before any step; nothing is truncated or extended silently. Without a
schedule, the thermostat target stays at its current value.

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
  `gpu_npt` do today). Provenance records both.
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
that.

### 7.5 CLI

- `fastmace run CONFIG.toml`: protocols as today's `main.py` profiles.
- `fastmace relax` and `fastmace quench RUN_DIR`.
- `fastmace check MODEL STRUCTURE`: gate plus self-checks.
- `fastmace bench MODEL STRUCTURE`: step time, breakdown.
- `fastmace info`: environment, GPU, available rewrites and kernels.

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
- **Re-capture policy.** A bucket overflow, a new atom bucket or a changed
  species superset triggers re-capture of the affected graphs; keep at most N
  captured variants (dynamo limit 64). Log every re-capture; if they happen
  often, the planner downgrades to tier 2.
- **Overflow before commit.** Every captured stage that can overflow a
  capacity (edges, k-vectors, atoms) sets a device flag. The host checks it
  before the step, or an MC decision based on it, is committed; an overflowed
  step is redone after re-capture from the state before it (today's
  speculative path redoes only the force stage). No observable, trajectory
  frame, restart or MC acceptance is ever produced from an overflowed
  evaluation.
- **Graph-safe events.** In-place edits of species, masses, charges and
  thermostat targets between replays, under the transaction contract of §8.5.
  They never change shapes or addresses.
- **Encapsulated counters.** The streaming edge counter becomes part of the
  engine, which removes `bind_streaming_count`.

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
  initial structure). Columns and caches are built for the superset.
- A species outside the superset raises before stepping, or triggers a
  planned rebuild if the workload allows it.

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
transmutation, mass or charge change, displacement move, thermostat target
change that touches cached coefficients) is one event:
1. **Quiesce:** finish pending work (pipelined force stage, asynchronous
   output, speculative steps) so that the state is a committed step.
2. **Snapshot** the affected state: species, masses, charges, positions for
   displacements, momenta of touched atoms, cached energy, forces and stress,
   and the registered derived buffers they feed.
3. **Apply** the mutation and refresh all registered derived buffers.
4. **Evaluate** what the decision needs (energy for MC; §10.4), with the
   overflow check of §8.1 before any decision.
5. **Accept or reject.** Reject restores the snapshot exactly (cached forces
   are restored, not recomputed). Accept keeps the new state; if it was
   accepted on an energy-only evaluation, the cached forces and stress are
   marked invalid.
6. **Resume:** before the next MD half step, energy, forces and (NPT) stress
   must be those of the committed state: invalid ones are recomputed once,
   so an MC block with several accepted moves costs one force evaluation at
   its end.

Random numbers consumed by a proposal are not restored on rejection; their
consumption is part of the chain and of the restart state (§10.6). Velocity
handling for changed masses is part of the move definition (§10.4), not a
free policy. A test runs the first MD step after an accepted and after a
rejected event and compares it with a fresh construction from the committed
state (bitwise on CPU float64 and in reproducible mode, §9.3; within the gate
otherwise).

### 8.6 Atom-count changes

Atom capacity comes in buckets with padded "ghost" atoms: zero mass,
excluded from neighbour pairs and an excluded species column. This needs a
proof that ghost atoms contribute exactly zero, checked like
`padding_is_exact`. Until that proof holds, atom-count changes run on tier 2.

### 8.7 Process constraints (enforced, not documented only)

- **One backend family per process.** MACE's autograd-compile policy is
  process-wide, so the planner refuses to compile hybrid and plain
  cuEquivariance models in one process. It offers a subprocess runner
  instead.
- **Allocator first.** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` must
  be set before the first CUDA allocation (torch ≥ 2.14). `fastmace` sets it
  on import if CUDA is not yet initialised; otherwise it warns.
- **Dynamo reset.** `torch._dynamo.reset()` is needed per PolarMACE engine;
  the engine does it internally.
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
    ensemble: Literal["nve", "nvt", "npt", "relax"]
    cell: Literal["fixed", "hydrostatic", "full"]
    species_changes: bool                 # swaps/transmutations
    species_superset: list[str] | None
    atom_count_changes: bool              # insertion/deletion, evaporation removal
    max_temperature_K: float              # drives skin and bucket headroom
    host_callbacks_every: int | None      # user Python per n steps
    needs_stress: bool
    needs_per_atom: set[str]              # energies, charges, ...
    mc_moves: tuple[str, ...]             # "swap", "transmute", "displace"; drives the ΔE gate (§12.3)
    reproducible: bool                    # bitwise within the scope of §9.3; refuses what it cannot guarantee
    precision: Literal["float32", "float64"]   # force precision; integration is always float64 (§7.3)
```

Periodicity and constraints come from the `Atoms` object and are checked
against §4.4.

Model traits come from `capabilities.py`: interaction classes,
`apply_cutoff`, joint embeddings, PolarMACE, heads, `padding_is_exact`,
`first_inputs_species_only`. Hardware traits: compute capability, memory,
OEq build availability, Triton availability, NVRTC.

### 9.3 Decision rules (initial set)

A plan has four independent dimensions: **execution mode** (the tiers of §8:
graph, compiled, eager, reference), **backend** (hybrid cuEq/OEq, plain
cuEq), **rewrite set** and **force precision**. The physics (model, head, charge/spin, D3 and its
parameters, polar options, reference corrections) is fixed by the user and is
never changed by a rule or a fallback. Rules are evaluated in this order; a
later rule cannot override an earlier one.

| Order | Condition | Decision |
| --- | --- | --- |
| 1 | Workload outside §4.4 (PBC, constraints) | refuse, or reference tier if the user allows it |
| 1 | PolarMACE and plain compiled backend | refuse (zero forces); hybrid, or eager plain |
| 1 | species outside the declared superset | refuse |
| 2 | `atom_count_changes` | compiled mode until ghost-atom exactness is proven (§8.6); the species superset still applies if `species_changes` is also set |
| 2 | `species_changes` (no atom-count changes) | graph mode with the declared superset; species-dependent rewrites built for it |
| 2 | `padding_is_exact` false | no graph mode (padded edges would change energies); compiled with exact edge counts |
| 3 | `host_callbacks_every` small (< ~10 steps) | compiled mode (graph replays would be broken up anyway) |
| 4 | `ensemble` and `cell` | `nve`/`nvt` require `cell="fixed"`; `npt` selects a barostat integrator; `relax` with a variable cell selects a cell-filtered optimiser (FIRE/LBFGS), not an NPT integrator; all variable-cell plans enable the stress path and the deformation-aware neighbour test; polar: k-vector headroom |
| 5 | `reproducible=True` | see below |
| 6 | `max_temperature_K` high | larger skin and bucket headroom (scaled from expected displacement per rebuild interval); re-capture budget |
| 7 | GPU memory insufficient for estimated edge bucket × feature width | drop memory-heavy rewrites or warn (e.g. generic-only POLAR-1-L NPT on NbCl5 ran out of memory on 20 GB) |
| 8 | Rewrite self-check fails | disable that rewrite and every rewrite that requires it, record reasons, re-plan |
| 8 | Gate fails (§12.3) | fallback algorithm (§9.4); never run silently with a failing gate |
| 9 | Known model–structure warning (§4.3) | warn (not refuse), with the documented diagnosis |

**`reproducible=True`.** Switching away from OEq is necessary but not
sufficient: MACE 0.3.16 itself sums messages with `scatter_add_`
(`mace/tools/scatter.py`), whose CUDA kernel uses atomics, and cuEquivariance
kernels make their own guarantees. The option therefore means:
- **Scope:** bitwise-identical results for the same hardware model, driver,
  software versions, model, plan, seeds and restart boundaries. No promise
  across GPUs, releases or platforms (PyTorch makes none either).
- **Enforcement:** plain cuEquivariance backend, hybrid-only rewrites off,
  `torch.use_deterministic_algorithms(True)` (with the required cuBLAS
  workspace setting), and a per-component allow-list. Any operator or kernel
  without a deterministic implementation makes the planner refuse the plan;
  it never falls back to a nondeterministic one.
- **Qualification:** repeated runs and an interrupted-versus-uninterrupted
  run must match bit for bit (§12.2) before the option is offered for a
  backend and model family.

### 9.4 Fallback algorithm

A failure (rewrite self-check, gate, replay or transition check, compilation
error) triggers this algorithm, not a fixed "one tier down":
1. **Candidates** are points in the four dimensions above, never a change of
   physics. Each candidate is built from a pristine copy of the loaded model
   (the checkpoint state before any rewrite), not by undoing rewrites in
   place.
2. **Diagnose the dimension.** Re-run the failing check on the same rewrite
   set in eager mode. If eager passes, the fault lies in compilation or
   capture: try compiled mode without graphs, then eager. If eager fails, the
   fault lies in a rewrite: bisect the rewrite set (respecting `requires`),
   disable the culprit and every rewrite that requires it.
3. **Backend changes respect §8.7.** A plain cuEquivariance candidate after a
   hybrid model was compiled in the process runs eager or in a subprocess,
   never as a second compiled backend in the same process.
4. **Validate before activation.** A candidate becomes active only after it
   passes the self-checks and the gate in the mode it will run in (replay
   included, §12.3). The failed plan and every tried candidate are recorded.
5. **Stop.** The reference tier is the last candidate only if it fits
   (§12.3); otherwise the planner refuses with the full record. Mid-run
   fallbacks (§9.6) follow the same steps between segments, from the last
   committed state.

### 9.5 Output

`Plan` is a frozen object: tier, enabled and disabled rewrites with reasons,
bucket and skin settings, kernel availability, the fallback candidates and
predicted memory. `str(plan)` is human-readable, and the plan is written into
provenance (§13). Users can override any choice, but overrides are still
self-checked.

### 9.6 Runtime adaptation

During a run the planner monitors:
- re-capture count;
- neighbour rebuild frequency;
- bucket overflow;
- non-finite observables.

It may switch tiers between segments (never mid-step), with a logged event.
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

**Momenta of atoms whose mass changes.** Part of the move definition, with
two valid choices:
- `rescale` (default): p' = p √(m_new / m_old), which keeps each atom's
  kinetic energy (as LAMMPS `fix atom/swap` does by default);
- `redraw`: after an accepted move, the momenta of the changed atoms are drawn
  from Maxwell–Boltzmann at T with the new masses (seeded stream).

Keeping velocities unchanged with an energy criterion on the potential alone
is not offered. It changes the kinetic energy (masses (1, 4) and velocities
(2, 1) give E_kin 4 before and 8.5 after the mass swap, at fixed positions)
and breaks detailed balance.

**Acceptance.** For both momentum choices:
`P_acc = min(1, exp(−β (ΔU − Σ_s Δμ_s ΔN_s)))`, with ΔN_s = 0 for swaps and
displacements. **Convention:** Δμ_s are configurational (mass-independent)
chemical-potential differences relative to a reference species, so
equilibrium species populations at fixed positions do not depend on masses.
Why both choices give this rule: `rescale` keeps the kinetic energy, and its
Jacobian (m_new/m_old)^{3/2} per changed atom cancels the mass factor of the
momentum partition function under this convention (for swaps the Jacobians
multiply to 1 under any convention). `redraw` is a Metropolis step on the
configurational marginal followed by a Gibbs step on the momenta. Users with
full chemical potentials (including the ideal-gas term
(3/2) k_B T ln m) convert with `fm.mc.configurational_mu(...)`. The
exact-enumeration test (§12.2) checks the convention.

**Evaluation.** Each attempt is one event under the transaction contract of
§8.5 and costs one energy evaluation of the trial state (forward only, no
backward). Attempts of one MC block run back to back; forces and stress are
recomputed once at the end of the block if any move was accepted.
Displacements can change the neighbour list and overflow a capacity; the
overflow check precedes the decision (§8.1).

**Energy precision.** A float32 total energy cannot resolve MC energy
differences in production cells. NbCl5 3×3×3 (1,944 atoms) has a total of
about −20,652 eV; its float32 spacing is 2⁻⁹ eV ≈ 2 meV (visible in the
FP32 relaxation log of `results/anneal-production-rtx5060ti-20261010/NbCl5`),
while 0.01 k_B T at 250 K is 0.2 meV. The MC path therefore keeps per-atom
energies in the force precision but sums them, and forms ΔE, in float64.
Per-atom energies of about 10 eV have a float32 spacing near 1 µeV, so the
summed rounding of all 1,944 atoms stays near 0.04 meV even where hybrid
atomic sums perturb every atom.

**Graph capture.** One attempt (propose → apply → refresh derived buffers →
energy → accept or restore) is a capturable graph with device-side random
numbers and a device-side accept mask, so swaps at high temperature keep
tier-1 speed.

**Schedule and counters.** n MD steps, then m attempts. Attempted, impossible
and accepted counts per move type are observables and part of the restart
state (§10.6).

**Accuracy prerequisite.** MC is enabled for a plan only after the energy and
move-energy gate (§12.3) passes on the user's structure, including moves to
species that are in the superset but absent at setup.

**Cost.** Each attempt costs one forward evaluation; benchmarks report it as
a fraction of an MD step (§16). With 50 attempts per 100 MD steps, keeping
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
(TOML/YAML or Python objects), not code paths in a CLI.

### 10.6 Simulation state and restart

The restart file holds the complete state: resuming must continue the run as
if it had not been interrupted (bit for bit where the plan is reproducible,
§9.3).
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
- **MC state:** position within the MD/MC cycle, counters per move type,
  adaptive step sizes.
- **Force-field identity:** checkpoint SHA-256, head, species superset, force
  precision, physics options (charge/spin inputs; D3 with all its parameters,
  §11.2; polar options such as the PR 1762 settings; the reference
  corrections of §12.1), MACE and FastMACE versions.
- **Plan:** execution mode, backend and rewrites; capacity buckets and the
  neighbour-list reference positions, from which the list is rebuilt
  deterministically, so that a reproducible resume rebuilds at the same steps.

Restart files are written atomically (temporary file, then rename, as today),
every 10,000 steps by default and only at committed steps: never inside an MC
block or from an overflowed step. `Simulation.resume` refuses a different
force-field identity. A different plan (another GPU or rewrite set) needs
`allow_plan_change=True` and is recorded: same physics, but no bitwise
continuation.

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
- uses the same neighbour list, with a longer cutoff in a second list;
- graph-capturable;
- stress included.

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
model family. It has three kinds of entries:
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

The record's identifier is part of the force-field identity (§10.6) and of
provenance (§13). Fallbacks (§9.4) keep the record: they never drop a
correction or an option, so a correct implementation cannot fail
qualification against a defective reference, and a fallback cannot restore
defective behaviour.

### 12.2 Test classes

| Class | Runs on | Content |
| --- | --- | --- |
| CPU unit | CI (CPU) | rewrite algebra on tiny models (`tiny_mace_calculator`), block linear/grouped/merged, k-vectors, PR 1762 port, neighbour list vs ASE/matscipy, integrator algebra, planner rules and precedence, restart round-trip. Today 294 tests; 8 workers locally (WSL RAM limit), `-n 24` on workstations |
| GPU self-check | CI (GPU) + every user run | per-rewrite and per-model checks (§6.4, §12.3) |
| Energy, force and stress gate | CI (GPU), `fastmace check` | today's rule: production path's max error vs float64 ≤ 2 × the float32 calculator's error, now for energy per atom as well as forces and stress; rattled and strained cells, hydrostatic and full; move-energy differences for MC plans (§12.3) |
| Replay and transitions | CI (GPU) | the production path as executed: captured replay, pipelined and speculative steps; across a neighbour rebuild, a capacity overflow with re-capture, a cell deformation past the rebuild threshold, an accepted and a rejected species change, and a restart, each compared with a fresh direct evaluation of the same state |
| Mutation events | CI (CPU, GPU) | derived-state registry against fresh construction (§8.5); first MD step after accept and after reject against fresh construction |
| MC exact enumeration | CI (CPU tiny model; GPU) | a small cell at fixed positions (e.g. 8 sites, 4 + 4 atoms: 70 configurations; transmutation: 256), all configurations evaluated with the reference: per-move acceptance probabilities against the reference ΔE, and sampled populations against Boltzmann weights (χ² test, fixed seed). Hybrid MD/MC: kinetic temperature with MC on and off agrees for `rescale` and `redraw`; a negative control with unchanged velocities must fail |
| Restart | CI (CPU, GPU) | §10.6: round trip; interrupted vs uninterrupted across accepted and rejected swaps and a partly consumed noise batch |
| Reproducibility | CI (GPU) | in reproducible mode, repeated runs and restarts bit for bit (§9.3) |
| Isolation | CI (GPU) | A → B → A with two crystals for the plain, MH-1 and polar paths (§8.7) |
| Fallback | CI (GPU) | deliberately broken rewrite and broken capture: eager diagnosis, bisection, pristine rebuild, validated activation (§9.4) |
| ASE equivalence | CI (GPU) | Langevin: same seed reproduces ASE's noise stream (NaCl 9 ps: same T and E_pot in every 50 K bin, |ΔPE| < 0.01 meV/atom); BAOAB vs ASE `LangevinBAOAB`; new integrators vs reference implementations |
| Statistical | nightly | long MD: distributions of T, E, P, V vs float64 stock ASE; hybrid runs are compared statistically, not bitwise |
| Performance | nightly | step-time suite (§12.5); regressions > 3% flagged (cross-process noise is about 3%; compare small effects with alternating A/B in one process) |

### 12.3 Gate on the user's structure

`Simulation` by default runs a reduced gate before the first step:
- 2 rattled copies (0.05 Å) and, for NPT, 2 strained copies (±2%);
- production path, **in the mode it will run** (captured replay in graph
  mode), against the reference of §12.1;
- energy per atom, forces and stress: the ratio rule of §12.2, with an
  absolute floor of 0.01 meV/atom for energy where float32 happens to be
  nearly exact;
- for MC plans, move-energy differences ΔE for sample moves of every enabled
  type (each swap pair, transmutation to every superset species including
  those absent at setup, displacements):
  `|ΔE_prod − ΔE_ref| ≤ 0.01 k_B T_min`, with T_min the lowest temperature
  at which the run attempts MC moves. Acceptance probabilities are then off
  by at most about 1%. There is deliberately no ratio term: the float32
  calculator forms ΔE from float32 totals, whose spacing (≈ 2 meV for
  NbCl5 3×3×3, §10.4) would let errors of several per cent in acceptance
  pass. A species-dependent constant energy error, invisible to forces and
  stress, fails here.

It takes seconds to a minute. A failing gate triggers the fallback algorithm
(§9.4).

**Status.** Every gate result has one of five states, all recorded:
- `passed`;
- `passed-reduced`: the reference ran on a smaller supercell of the same
  structure, labelled as such;
- `failed`: triggers §9.4;
- `unavailable`: the reference could not run on any allowed device;
- `skipped`: by the user (`check="skip"`).

**Resources.** An FP32 production workload can fit where its unoptimised
float64 reference does not (today: NbCl5 3×3×3 PolarMACE in float64 exceeds
16 GB on the original path, `polar_stress_probe.py`). The gate therefore
evaluates the references first, one configuration and one calculator at a
time, frees them, and only then builds the production engine. On GPU memory
exhaustion it tries CPU float64 if the estimated time is within a
configurable limit, then a reduced cell. If all fail, the status is
`unavailable`: the run is refused unless the user passes
`check="allow_unavailable"`, which is recorded. A more memory-hungry
reference is never a fallback for a plan that failed for memory.

### 12.4 CI matrix

- **CPU:** Linux, Python 3.10–3.12, two torch versions.
- **GPU:** at least one Ada (sm_89) and one Blackwell (sm_120) runner, with
  the hybrid and plain backends.
- Model checkpoints are cached in CI and verified by SHA-256.

### 12.5 Benchmark suite

`fastmace bench` reproduces today's tables:
- models: MH-1 (three heads), MP-0 medium/large, MP-0b3, MPA-0, OMAT-0,
  MatPES-PBE/r²SCAN, POLAR-1-S/M/L, OMOL-0 XL;
- crystals: NaCl, NaNbO3, NbCl5 (3×3×3), plus the ten benchmark crystals;
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
  precision, TF32 flag, backend, fallbacks tried;
- versions of all packages and kernels (OEq extension build id, Triton
  version), GPU name, driver, compute capability;
- seeds and RNG states, integrator parameters, schedule;
- gate status (§12.3) and self-check results;
- re-capture and rebuild counts, overflows redone, tier switches, MC
  counters, timing summary;
- code version (package version + git hash when installed from source).

This is today's provenance block, made complete and schema-versioned.

---

## 14. Packaging and distribution

- **Name:** `fastmace`. Check PyPI and GitHub for clashes before the first
  release; the import name and CLI follow.
- **Licence:** MIT (compatible with MACE). Vendored OEq kernels keep their
  BSD-3 licence and notice. The cuEquivariance ops wheels are NVIDIA-licensed:
  depend on them, do not vendor them. Model weights are never redistributed;
  `fm.load` downloads from official sources and shows the model licence.
- **Extras:** `fastmace[oeq]`, `fastmace[polar]` (graph-longrange pinned per
  MACE version), `fastmace[d3]` (until native D3), `fastmace[dev]`.
- **No `nvcc` at install or run time:**
  - The streaming kernels already compile through NVRTC.
  - OEq's host extension needs a build path without `nvcc` (today:
    `oeq_no_nvcc.sh` with runtime headers, per-process extension directory,
    `OEQ_EXTENSION_SEED`, lock-file workaround for OEq 0.7/torch 2.14).
  - Either ship prebuilt OEq extension wheels per torch/CUDA, or upstream the
    fix to OEq.
  - Whichever is chosen, `main.py`'s refusal to run without the wrapper must
    disappear.
- **Caches:**
  - Triton and Inductor caches, the OEq extension cache and compile caches go
    under a configurable cache root (`FASTMACE_CACHE`), never inside
    environments.
  - Respect read-only and NFS homes: today's NFS-lock failure on coraline and
    pytest's ownership check on the NAS are known problems.

---

## 15. Documentation

- Quick start (ASE calculator; `Simulation`).
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
| 1. Core extraction and contracts | `model`, `rewrites` (registry with today's catalogue), `neighbors`, `state` (complete state, §10.6), `engine.graph`, `integrators.langevin`/`baoab`; a minimal planner (the only component that builds engines: hard refusals, mode and backend choice, `Plan` record; no runtime adaptation); restart format (§10.6); reference policy (§12.1); gate with energy and captured replay; no globals, no NAS or experiment imports, no wrapper script | against a frozen research-engine revision on the same machine and configuration, in alternating A/B blocks: steady-state step time within 3%, setup time and peak memory reported; gate passes for all 14 models; restart round trip, interrupted vs uninterrupted (bitwise on CPU float64, statistical for hybrid); `Experiments/Langevin_Thermostat/main.py` runs on top of FastMACE | 4–5 weeks |
| 2. Safety | self-checks, gate statuses and resource policy (§12.3), fallback algorithm (§9.4), versioned registry, instance isolation (§8.7), provenance schema, GPU CI | unknown MACE version runs the reference path cleanly; a deliberately broken rewrite and a broken capture each end in the correct validated fallback; A → B → A passes; CI on two GPU architectures | 3 weeks |
| 3. Dynamic topology and MC | species superset, derived-state registry, transactional events (§8.5), overflow before commit, re-capture policy, adaptive skin and headroom, compiled-tier path; MC driver (swap, transmutation, displacement; `rescale`/`redraw`), move-energy gate, MC restart state | species-change events (no energy evaluation) every 100 steps at 1500 K keep ≥ 90% of fixed-species step speed; MC attempt cost reported separately as a fraction of an MD step; first MD step after accept and reject equals fresh construction; exact-enumeration test passes; interrupted vs uninterrupted across accepted swaps; swap-MC annealing protocol runs end to end | 4–5 weeks |
| 3b. NPT for the chlorides | GPU D3 in the force field (§11.2) with stress and its second, longer neighbour list, including the deformation-aware rebuild test; D3 in the BAOAB NPT path; constant-pressure melt–quench protocol | D3 against `torch-dftd` and `dftd3`; GPU NPT + D3 against the independent implementation (§11.2); the NbCl5 NPT melt–quench protocol runs end to end. Its physical outcome is recorded, not judged (decision 6, §18) | 2–3 weeks |
| 4. Physics | NVE, CSVR, NHC, MTK, Berendsen; GPU FIRE/LBFGS, fixed and variable cell; `FixAtoms` | ASE-equivalence tests; GPU LBFGS reproduces the annealing-benchmark minima | 3–4 weeks |
| 5. Planner completion | remaining rules of §9.3 and their precedence, runtime adaptation (§9.6), optional calibration, `str(plan)` | each rule and the precedence order have a test | 1–2 weeks |
| 6. Release | docs, examples, benchmark tables, PyPI, upstream PRs (MACE: linear rewrites, reassociation; OEq: no-`nvcc` host build) | 1.0 tagged; docs build; install from PyPI on a clean machine | 2–3 weeks |

The total is about 20–26 weeks (4.5–6 months) to 1.0, up from 4–5 months in
draft 0.1, because the planner, restart and reference contracts and the MC
driver now come before the dynamic-topology milestone. The first production
deliverable is phases 0–3b (14–17 weeks, about 3.5–4 months): NVT and
constant-pressure melt–quench and atom exchange on the fastest tier,
including the chlorides with D3 (decision 6). The barostat settings still
need their qualification study (an experiment, not library work) before NPT
defaults are recommended.

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

---

## 18. Risks and open questions

| Risk / question | Mitigation |
| --- | --- |
| MACE internals change every release; hash-pinned rewrites break | versioned registry, self-checks, upstreaming; reference fallback |
| OEq/cuEq/torch version churn (lock files, `nvcc`, compile policy) | CI matrix, prebuilt extensions, process-constraint enforcement |
| Hybrid OEq sums are not bitwise reproducible | statistical tests; `reproducible=True` plan |
| Species superset increases memory or cost (columns for absent species) | measure; superset defaults to initial species |
| High-T workloads cause frequent re-captures | adaptive skin and headroom, re-capture budget, tier-2 fallback |
| Model-level failures look like engine bugs (POLAR-1-S NaNbO3, OMOL NaCl) | planner warnings with diagnoses; gate against the reference (§12.1) separates the two |
| Barostat settings and equilibrium volumes unqualified | keep warning; qualification study before recommending NPT defaults |
| Hybrid MD/MC samples the wrong distribution (mass changes, proposal asymmetry, chemical-potential convention) | moves defined with their momentum transformation and acceptance (§10.4); exact-enumeration test with a negative control (§12.2) |
| A stale derived buffer after an event (masses, coefficients, species pairs) | derived-state registry checked against fresh construction (§8.5) |
| The float64 reference does not fit where the FP32 production path does | sequential reference evaluation, CPU and reduced-cell options, explicit `unavailable` status (§12.3) |
| A fault appears only inside CUDA graphs (as with Inductor's programmatic dependent launch) | gate and transition tests through captured replay (§12.2, §12.3) |
| Positioning vs existing tools (TorchSim batched MLIP MD; MACE's LAMMPS ML-IAP) | FastMACE's distinct value: exact model-level rewrites plus graph-speed single-structure MD with dynamic species; re-check competitors' feature sets before 1.0 and consider offering the force field as a backend to them |
| Licensing of dependencies and models | §14; licence display at load |
| Name clash on PyPI | check before phase 6 |

Open decisions for the owner:
1. Repository — **decided (user, 16 October 2026): inside ResearchModule as
   `FastMACE/` for now.** A separate repository (simpler CI and release) can
   follow before the first public release.
2. Minimum MACE version to support beyond 0.3.16.
3. Whether the gate on the user's structure is on by default (recommended) or
   opt-in, and whether an `unavailable` gate refuses the run by default
   (proposed, §12.3) or only warns.
4. Licence choice (MIT proposed).
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
   float64 (§10.4), because float32 totals cannot resolve the floor.

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
