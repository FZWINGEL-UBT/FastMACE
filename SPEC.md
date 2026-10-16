# FastMACE — Specification

Status: draft 0.1, 16 October 2026. Owner: Fabian Zwingel. Written from the
Langevin_Thermostat experiment (`Experiments/Langevin_Thermostat/`), whose GPU
engine this library generalises.

FastMACE is a standalone Python library that runs pretrained MACE models
(MACE 0.3.x family, including PolarMACE) for atomistic simulation on NVIDIA
GPUs, with exact model-level rewrites, a GPU-resident simulation state, CUDA
graphs and a planner that picks the fastest execution path that is valid for
the user's workload. It must give the same physics as MACE's own calculator
(within float32 round-off) and fall back safely whenever it cannot.

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
   - The accuracy gate runs on the user's own structure.
   - The planner refuses or falls back on any mismatch.
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
with the library's own benchmark suite (§12.4), is a release task. A uniform
run of the research engine on 16 October 2026 was stopped after the three MH-1
heads (user decision); its MH-1 values are in the first row.

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
  state.py               SimulationState: positions, momenta, masses, cell, species, charges/spin, RNG, step
  integrators/
    base.py              Integrator protocol: graph-capturable half steps, observables, restart
    nve.py  langevin.py  baoab.py  csvr.py  nose_hoover.py  mtk.py  berendsen.py
  mc/
    moves.py             swap, displacement, (later) insertion/deletion
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
    plan.py              Planner, Plan, decision rules, fallback chain (§9)
    selfcheck.py         rewrite and gate checks on the user's structure (§12.2)
  longrange/
    polar.py             PolarMACE GPU path (§11.1)
    d3.py                GPU D3(BJ) dispersion (§11.2)
  io/
    trajectory.py        async writer, ASE Trajectory/extxyz, every-n-steps, chunked large files
    restart.py           checkpoint/restart format (§10.6)
  provenance.py          run metadata (§13)
  protocols/             heat/cool profiles, anneal, quench, melt–quench (from workflow.py profiles)
  ase/calculator.py      FastMACECalculator (ASE drop-in)
  cli/                   `fastmace run|relax|bench|check|info`
  bench/                 step_time, breakdown (from model_step_time / step_breakdown_probe)
tests/  cpu/ gpu/ equivalence/ statistical/
docs/
```

Layering rules:
- `model` and `rewrites` never import `engine`.
- Integrators see only `SimulationState` and a `ForceField`.
- The planner is the only component that instantiates engines.
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
  structure (§12.2).

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
sim.run(steps=200_000, schedule=fm.protocols.ramp(0, 1000, ps=20) + fm.protocols.ramp(1000, 250, rate_K_per_ps=-3))
sim.run(steps=..., mc=fm.mc.Swap(pairs=[("Na", "Nb")], every=100, attempts=50))
sim.save_restart("run/restart.npz"); sim = fm.Simulation.resume("run/")
```

### 7.3 ASE front end

```python
atoms.calc = fm.ase.FastMACECalculator(model, rewrites="auto")   # compiled tier, no graphs
```

This front end has the same results contract as `MACECalculator`. ASE drives
the integration, so the speed-up is smaller (no graph engine); document that.

### 7.4 CLI

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
- **Graph-safe events.** In-place edits of species/one-hot buffers, charges,
  thermostat targets and masses between replays. These never change shapes or
  addresses.
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

MACE's `MACECalculator` (cuEquivariance float64 or float32). Used for the gate,
for unknown models and as the last fallback.

### 8.5 Species superset (required for exchange workloads)

Today the reassociation, the species-restricted skip and the compile key use
the species present at setup ("atoms may change places later, but not to
species absent at setup"). The library changes this:
- The `ForceField` takes a declared species superset (default: species in the
  initial structure). Columns and caches are built for the superset.
- A species change is an in-place write to `node_attrs` (one-hot) and to the
  species-index buffer. No shapes change, so graphs stay valid. A change to a
  species outside the superset raises before stepping, or triggers a planned
  rebuild if the workload allows it.
- Masses are a buffer too. Swapping species swaps masses, and momenta are
  rescaled by policy: keep velocities, or redraw from Maxwell–Boltzmann for
  swapped atoms. The policy is documented and seeded.

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
    reproducible: bool                    # bitwise: forces non-hybrid backend
    precision: Literal["float32", "float64"]
```

Model traits come from `capabilities.py`: interaction classes,
`apply_cutoff`, joint embeddings, PolarMACE, heads, `padding_is_exact`,
`first_inputs_species_only`. Hardware traits: compute capability, memory,
OEq build availability, Triton availability, NVRTC.

### 9.3 Decision rules (initial set)

| Condition | Decision |
| --- | --- |
| `padding_is_exact` false | no tier 1 (padded edges would change energies); tier 2 with exact edge counts |
| `reproducible=True` | plain cuEquivariance backend (hybrid OEq sums are not bitwise reproducible); hybrid-only rewrites off |
| `atom_count_changes` | tier 2 until ghost-atom exactness is proven (§8.6) |
| `species_changes` | tier 1 with declared superset; species-dependent rewrites built for the superset; refuse species outside it |
| `host_callbacks_every` small (< ~10 steps) | tier 2 (graph replays would be broken up anyway) |
| `max_temperature_K` high | larger skin and bucket headroom (scaled from expected displacement per rebuild interval); re-capture budget |
| `cell != fixed` | stress path, variable-cell neighbour test, NPT integrator; polar: k-vector headroom |
| PolarMACE and not hybrid | refuse compiled plain cuEq (zero forces); hybrid or eager |
| GPU memory insufficient for estimated edge bucket × feature width | drop memory-heavy rewrites or warn (e.g. generic-only POLAR-1-L NPT on NbCl5 ran out of memory on 20 GB) |
| Rewrite self-check fails | disable that rewrite, record reason, re-plan |
| Gate fails (§12.2) | fall back one tier (fewer rewrites → plain → reference); never run silently with a failing gate |
| Known model–structure warning (§4.3) | warn (not refuse), with the documented diagnosis |

### 9.4 Output

`Plan` is a frozen object: tier, enabled and disabled rewrites with reasons,
bucket and skin settings, kernel availability, the fallback chain and
predicted memory. `str(plan)` is human-readable, and the plan is written into
provenance (§13). Users can override any choice, but overrides are still
self-checked.

### 9.5 Runtime adaptation

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
- serialises its own state for restarts;
- has an ASE-equivalence test (§12.3).

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

- **Moves:** species swap between atoms of two species; transmutation within
  the superset (semi-grand canonical, with chemical-potential differences);
  displacement. Insertion and deletion follow in 1.x (§8.6).
- **Accept/reject on the GPU:** trial energy from a full force-field
  evaluation (energy-only path, no backward). Metropolis at the current
  thermostat temperature, with a seeded RNG.
- **Hybrid MD/MC schedule:** n MD steps, then m MC attempts. Velocities of
  swapped atoms are handled by the policy of §8.5.
- **Graph capture:** the MC step is a capturable graph (propose → in-place
  swap → energy → accept → revert if rejected), so swaps at high temperature
  keep tier-1 speed.
- **Cost:** each attempt costs one energy evaluation. A later local-energy
  variant (recompute only atoms within n_layers × r_cut of the swap) is a 1.x
  optimisation; it must be exact.

### 10.5 Protocols

Port today's profiles:
- `anneal`: LBFGS fixed-cell fmax 0.01 pre-relaxation; 0 → 1000 K in 20 ps;
  3 K/ps to 250 K; every 20th step saved.
- `quench`: frames per 100 K bin from 500 K, quenched with LBFGS.
- Heating/cooling ramps, holds, and the `fast` and `ultrafast` profiles.

Add melt–quench and a swap-MC annealing protocol. Protocols are data
(TOML/YAML or Python objects), not code paths in a CLI.

### 10.6 Restart

The restart format holds:
- positions, momenta, cell;
- integrator state;
- RNG state (device and host streams);
- step, schedule position;
- plan;
- model hash.

Written every 10,000 steps by default. `Simulation.resume` refuses to resume
with a different model hash or incompatible rewrites unless overridden, and it
keeps the protocol recorded at start (today's `LEGACY_PROTOCOL` behaviour).

---

## 11. Physics extensions

### 11.1 Long range (PolarMACE)

Port `polar_gpu.py`:
- fixed-size k-vector list with an exact inclusion mask and headroom for
  strained cells (overflow flag → re-plan);
- shared `_Holder` and per-call `_Call` state;
- the stress-geometry fix (MACE issue 1642);
- the uvu GEMM (PR 1711).

PR 1762 (Fukui regularisation, dipole cap) stays opt-in. Charge and spin
multiplicity are per-structure inputs with validation.

### 11.2 D3 dispersion on the GPU

The molecular chlorides need D3 for NPT: without it, NbCl5 swells 20% at
300 K and evaporates at 600 K. Today D3 exists only through `torch-dftd` on
the ASE path. Implement D3(BJ) inside the force field:
- uses the same neighbour list, with a longer cutoff in a second list;
- graph-capturable;
- validated against `torch-dftd` 0.5.1 and the reference `dftd3`;
- stress included.

### 11.3 Constraints

`FixAtoms`, fixed centre of mass, and (1.x) bond and angle constraints
(RATTLE) for molecular species.

---

## 12. Validation and testing

### 12.1 Test classes

| Class | Runs on | Content |
| --- | --- | --- |
| CPU unit | CI (CPU) | rewrite algebra on tiny models (`tiny_mace_calculator`), block linear/grouped/merged, k-vectors, PR 1762 port, neighbour list vs ASE/matscipy, integrator algebra, planner rules, restart round-trip. Today 294 tests; 8 workers locally (WSL RAM limit), `-n 24` on workstations |
| GPU self-check | CI (GPU) + every user run | per-rewrite and per-model checks (§6.4, §12.2) |
| Force/stress gate | CI (GPU), `fastmace check` | today's rule: production path's max error vs float64 ≤ 2 × the float32 calculator's error, forces and stress, rattled and strained cells, hydrostatic and full |
| ASE equivalence | CI (GPU) | Langevin: same seed reproduces ASE's noise stream (NaCl 9 ps: same T and E_pot in every 50 K bin, |ΔPE| < 0.01 meV/atom); BAOAB vs ASE `LangevinBAOAB`; new integrators vs reference implementations |
| Statistical | nightly | long MD: distributions of T, E, P, V vs float64 stock ASE; hybrid runs are compared statistically, not bitwise |
| Performance | nightly | step-time suite (§12.4); regressions > 3% flagged (cross-process noise is about 3%; compare small effects with alternating A/B in one process) |

### 12.2 Gate on the user's structure

`Simulation` by default runs a reduced gate before the first step:
- 2 rattled copies (0.05 Å) and, for NPT, 2 strained copies (±2%);
- production path vs the float64 reference;
- the same ratio rule as §12.1.

It takes seconds to a minute. It can be skipped (`check="skip"`), but the
skip is recorded. A failing gate triggers the fallback chain.

### 12.3 CI matrix

- **CPU:** Linux, Python 3.10–3.12, two torch versions.
- **GPU:** at least one Ada (sm_89) and one Blackwell (sm_120) runner, with
  the hybrid and plain backends.
- Model checkpoints are cached in CI and verified by SHA-256.

### 12.4 Benchmark suite

`fastmace bench` reproduces today's tables:
- models: MH-1 (three heads), MP-0 medium/large, MP-0b3, MPA-0, OMAT-0,
  MatPES-PBE/r²SCAN, POLAR-1-S/M/L, OMOL-0 XL;
- crystals: NaCl, NaNbO3, NbCl5 (3×3×3), plus the ten benchmark crystals;
- configurations: ASE plus calculator, plain GPU engine, default NVT, default
  NPT;
- timing: median of 4 × 300 steps after 300 warm-up steps.

Results go to versioned JSON. The breakdown mode reproduces
`step_breakdown_probe` (stages, kernel categories, operators).

---

## 13. Provenance

Every run writes `metadata.json` with:
- model path, SHA-256, head, licence;
- the plan (tier, rewrites on/off with reasons), precision, TF32 flag, backend;
- versions of all packages and kernels (OEq extension build id, Triton
  version), GPU name, driver, compute capability;
- seeds and RNG states, integrator parameters, schedule;
- gate and self-check results;
- re-capture and rebuild counts, tier switches, timing summary;
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
- Concepts: tiers, plan, rewrites, gate, species superset, buckets.
- How-tos: melt–quench, swap MC at high T, NPT with D3, PolarMACE with charge,
  restart, benchmarking.
- Model notes: per-family support and known model issues (§4.3).
- Rewrite catalogue with exactness, gains and constraints, plus the
  rejected-screens log.
- Reproducibility: non-bitwise hybrid sums, statistical comparison.
- API reference (generated).

---

## 16. Migration plan and milestones

Each phase ends with its acceptance criteria met on the RTX 5060 Ti and one
workstation GPU.

| Phase | Content | Acceptance | Estimate (1 developer) |
| --- | --- | --- | --- |
| 0. Skeleton | repo layout, packaging, CI (CPU), licence, port CPU tests | `pip install -e .`; CPU suite green | 1 week |
| 1. Core extraction | `model`, `rewrites` (registry with today's catalogue), `neighbors`, `state`, `engine.graph`, `integrators.langevin`/`baoab`; no globals, no NAS or experiment imports, no wrapper script | `fastmace bench` reproduces §3.3 within 3%; gate passes for all 14 models; `Experiments/Langevin_Thermostat/main.py` can run on top of FastMACE | 3–4 weeks |
| 2. Safety | self-checks, gate on user structure, versioned registry, provenance schema, GPU CI | unknown MACE version runs reference path cleanly; failing rewrite disables itself; CI on two GPU archs | 2–3 weeks |
| 3. Dynamic topology | species superset, graph-safe in-place events, re-capture policy, adaptive skin/headroom, tier 2 path | swap events every 100 steps at 1500 K keep ≥ 90% of fixed-species step speed; energies identical to rebuilding from scratch | 2–3 weeks |
| 4. Physics | NVE, CSVR, NHC, MTK, Berendsen; GPU FIRE/LBFGS; MC swap driver; GPU D3; constraints | ASE-equivalence tests; NbCl5 NPT with D3 stable at 600 K; swap-MC annealing protocol runs end to end | 4–6 weeks |
| 5. Planner | `Workload`, rules of §9.3, fallback chain, runtime adaptation, `str(plan)` | each rule has a test; deliberately broken rewrite or model leads to correct fallback | 2 weeks |
| 6. Release | docs, examples, benchmark tables, PyPI, upstream PRs (MACE: linear rewrites, reassociation; OEq: no-`nvcc` host build) | 1.0 tagged; docs build; install from PyPI on a clean machine | 2–3 weeks |

The total is about 4–5 months to 1.0. Phases 0–3 (about 2 months) already
cover the research production needs: melt–quench and atom exchange on the
fastest tier.

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
| Model-level failures look like engine bugs (POLAR-1-S NaNbO3, OMOL NaCl) | planner warnings with diagnoses; gate against the model's own calculator separates the two |
| Barostat settings and equilibrium volumes unqualified | keep warning; qualification study before recommending NPT defaults |
| Positioning vs existing tools (TorchSim batched MLIP MD; MACE's LAMMPS ML-IAP) | FastMACE's distinct value: exact model-level rewrites plus graph-speed single-structure MD with dynamic species; re-check competitors' feature sets before 1.0 and consider offering the force field as a backend to them |
| Licensing of dependencies and models | §14; licence display at load |
| Name clash on PyPI | check before phase 6 |

Open decisions for the owner:
1. Repository — **decided (user, 16 October 2026): inside ResearchModule as
   `FastMACE/` for now.** A separate repository (simpler CI and release) can
   follow before the first public release.
2. Minimum MACE version to support beyond 0.3.16.
3. Whether the gate on the user's structure is on by default (recommended) or
   opt-in.
4. Licence choice (MIT proposed).
5. Which GPUs to provide as CI runners (local RTX 5060 Ti, a workstation
   RTX 4000 Ada).
