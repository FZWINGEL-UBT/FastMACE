# Phase 1a: extraction

Status: planned, 17 October 2026. Spec: [SPEC.md](../SPEC.md) §3, §4.1,
§4.3, §4.4, §5, §6, §8.1, §9.1, §12.2, §12.5, §16 row 1a. Estimate: 3–4
weeks. Requires phase 0 merged.

Since 17 October 2026 FastMACE has its own repository (SPEC §18 decision 1).
"Source" paths and every path starting with `Experiments/` are in
ResearchModule (`G:\Margraf\ResearchModule`). Workers read them there and
write only here. The one exception is T17, which changes ResearchModule
itself (see that task).

## Goal

Move the research engine's GPU path into `fastmace` with **the same
numerics and the same speed**. It goes behind the planner interface with one
fixed plan (today's `main.py` defaults), with no module globals, no NAS or
experiment imports and no wrapper script. Phase 1a changes structure, not
numbers. Everything that changes results (float64 energy accumulation, the
reference policy, the gate with energy, restart contracts) belongs to 1b.

**Acceptance (SPEC §16):**

1. **Speed.** Against the frozen research engine (tag
   `research-engine-baseline-20261017`), on the same machine and
   configuration, in alternating A/B blocks: steady-state MD step time within
   3%; setup time and peak GPU memory reported. Measured on the RTX 5060 Ti
   and one workstation GPU (murphy or leto).
2. **Accuracy.** Today's force/stress gate
   (`tools/validation/npt_stress_reference.py`'s rule) passes for all 19
   labels of §12.5: the ten benchmark models (MH-1 three heads, MP-0
   medium/large, MP-0b3, MPA-0, OMAT-0, MatPES-PBE/r²SCAN), POLAR-1 S/M/L,
   OMOL-0 XL and the five MACE-OFF models (on the organic set). Energy
   joins the gate in 1b.
3. **Compatibility.** `Experiments/Langevin_Thermostat/main.py` runs on top
   of FastMACE: smoke, `--profile anneal` and `--quench`, results
   statistically equal to the baseline (hybrid) or bit for bit where the
   backend is deterministic.
4. **Rules.** The layering test from phase 0 passes; no `gpu_langevin`-style
   globals; `oeq_no_nvcc.sh` is not needed (or the owner accepted the
   alternative from T00).

## Working principles

- **Copy, then switch.** Code is copied from the baseline into `fastmace`
  and refactored there. The research engine stays untouched until T17.
  `main.py` then gains a FastMACE path, and only after acceptance does the
  owner decide whether the `langevin` modules become thin shims.
- **Same numbers first.** Every task that moves numerical code proves
  equality against the baseline on CPU float64 with the tiny models
  (bitwise where the order of operations is unchanged, otherwise within
  1e-12 relative error), before any GPU check.
- **One owner per file.** Each task lists the files it owns. `config.py`
  belongs to T01 and the interfaces to T02. A task that needs a new config
  field or an interface change reports it to the conductor instead of
  editing those files.
- **GPU time is scheduled.** Workers finish CPU tests, then ask the
  conductor for a GPU slot. Timing runs (T18, T19) need an idle GPU and
  nothing else running.

## Tasks

Each task: ID, size (S < ½ day, M ≈ 1 day, L 2–4 days), dependencies, spec
sections, source at the baseline tag, files owned, tests first, done-when.
"Source" paths are relative to `Experiments/Langevin_Thermostat/` unless
they start with `Experiments/`.

### Wave 0: design (conductor plus two workers)

#### 1a-T00 Spike: OEq runtime without the launcher script (M, GPU)

- Spec: §3.2 "Paths and environment", §14 "No nvcc".
- Source: `studies/hybrid_oeq/scripts/oeq_no_nvcc.sh`,
  `oeq_nvcc_unavailable.sh`, `workflow.require_hybrid_wrapper`,
  `studies/hybrid_oeq/coraline.md`.
- Question: which of the script's settings must exist before the process
  starts (`LD_LIBRARY_PATH`, `PATH` for the `nvcc` sentinel), and which can be
  set in-process before OEq is imported (`CUDA_HOME`, `CPATH`,
  `TORCH_EXTENSIONS_DIR`, `OEQ_JIT_EXTENSION`, `MAX_JOBS`, cache paths,
  `CUDA_CACHE_DISABLE`)? Candidates:
  (a) in-process environment plus `ctypes` preloading of the cu13 runtime
  libraries with `RTLD_GLOBAL`;
  (b) a `fastmace.oeq.prepare()` that re-executes the interpreter once with
  the right environment, only when the hybrid is requested;
  (c) keep a launcher, shipped as a `fastmace` console entry point instead of
  a shell script.
- Files: `docs/design/oeq-runtime.md` (findings and recommendation), and a
  prototype `fastmace/kernels/_oeq_runtime.py` only if (a) works.
- Done when: the recommendation is backed by a hybrid MH-1 NaCl run on the
  RTX 5060 Ti that builds the OEq extension in a fresh cache without the
  script. **Owner checkpoint** if the answer is (b) or (c).

#### 1a-T01 Switch map and config objects (M)

- Spec: §3.2 "Module globals", §5 `config.py`, §6.1, §6.2.
- Source: every module-level switch in `langevin/gpu_langevin.py` (lines
  38–250 and 1221–1240), `langevin/polar_gpu.py`, `langevin/gpu_npt.py`
  (`EXPM_*`, `ASE_TRACE_DRAG_ALL_ELEMENTS`), `langevin/upstream_mace.py`;
  where `workflow.GPUEngine.__init__` (around line 671) sets them; the CLI
  defaults in `workflow.parse_args`; `workflow.adapt_options_to_model` (the
  per-family `auto` resolution).
- Files: `docs/design/switch-map.md` (a table: research-engine global or
  flag → config class and field → default today → `auto` resolution rule →
  catalogue entry of §6.2 or "engine"/"integrator"/"never"), and
  `fastmace/config.py` (frozen dataclasses: `RewriteConfig`, `EngineConfig`,
  `PrecisionConfig`, `NeighbourConfig`, `PolarConfig`, `IntegratorConfig`;
  all fields of the map, defaults equal to today's `main.py` defaults).
  Experimental switches that §6.2 lists as rejected or "never enable"
  (`INDUCTOR_PDL`, `INDUCTOR_COMBO`, `DECOMPOSE_SMALL_MM`,
  `CONTIGUOUS_FOLDED_WEIGHT`, …) are recorded in the map as "not carried
  over" and get no field.
- Tests first: `tests/cpu/test_config.py`: the dataclasses are frozen; the
  defaults equal the values in the map; every `auto` field has a resolution
  rule that T15 will implement (checked against a list in the test).
- Done when: the conductor has reviewed the map. **No other task edits
  `config.py`.**

#### 1a-T02 Interfaces (M, conductor)

- Spec: §5 layering rules, §7.1 (`ForceField`), §8.1 (graph stages ACP, B,
  C), §9.5 (`Plan` output), §10.1 (integrator contract), §10.6 (state).
- Source: how `GPULangevin` (`gpu_langevin.py:2098`) and
  `GPULangevinBAOAB` (`gpu_npt.py:83`) mix integrator maths with capture
  and replay; how `MACEForces` (`gpu_langevin.py:1652`) is called by them.
- Files: `fastmace/model/forces.py` (the `ForceField` protocol only),
  `fastmace/integrators/base.py` (graph-capturable half-step protocol),
  `fastmace/state.py` (the `SimulationState` fields 1a needs: positions,
  momenta, masses, cell, species, RNG stream; the complete §10.6 state comes
  in 1b), `fastmace/engine/base.py`, `fastmace/planner/plan.py` (the `Plan`
  record and a `Planner` protocol), `docs/design/interfaces.md` (one page:
  who calls whom per MD step, mapped onto today's ACP/B/C stages).
- The central design question: `GPULangevin` is integrator and engine at
  once. Decide here where the cut goes, so that T12 and T13 can work in
  parallel. The integrator provides tensor operations for the half steps;
  the engine captures them into graphs.
- Done when: stubs import; the layering test passes; **owner checkpoint**
  on `docs/design/interfaces.md` (one page, 15 minutes).

### Wave 1: leaves (parallel after T01 and T02)

#### 1a-T03 Model capabilities (S)

- Spec: §4.3, §9.2 "Model traits".
- Source: `gpu_langevin.py`: `FACTORED_INTERACTIONS`,
  `_radial_cutoff_passed`, `first_inputs_species_only`, `rewrite_support`,
  `padding_is_exact`, `GENERIC_INTERACTIONS` (class names only; hashes go to
  T06).
- Files: `fastmace/model/capabilities.py` (a frozen `ModelTraits` record:
  interaction classes, `apply_cutoff`, joint embeddings, PolarMACE, heads,
  padding exactness, species-only first inputs, all-electron vs
  pseudopotential E0 classified from the E0 values as §4.3 describes).
- Tests first: port the trait tests of `tests/core/test_gpu_langevin.py`
  and `test_generic_rewrites.py`; one test per tiny-model class; E0
  classification on synthetic tables.

#### 1a-T04 Kernels (M, GPU at the end)

- Spec: §3.1 kernels row, §5 `kernels/`, §14 (OEq licence).
- Source: `langevin/edge_outer_sum.py`, `skinny_linear.py`,
  `fused_radial.py`, `inplace_linear.py` (opt-in only, §6.2), `oeq_streaming/`
  (vendored; keep `LICENSE` and `NOTICE.md` byte-identical), and the cache
  part of `langevin/compile_cache.py`.
- Files: `fastmace/kernels/triton/*.py`, `fastmace/kernels/oeq_streaming/`,
  `fastmace/kernels/__init__.py` (availability probes: Triton importable,
  NVRTC through `cuda.bindings`, with a reason string). Caches go under
  `fastmace._cache`, never under the module directory.
- Tests first: port `tests/core/test_edge_outer_sum.py`,
  `test_skinny_linear.py`, `test_fused_radial.py`, `test_oeq_streaming.py`
  (CPU parts to `tests/cpu`, GPU parts to `tests/gpu`); a test that the
  vendored licence files are present and unchanged.

#### 1a-T05 Model loading (L)

- Spec: §3.2 (paths; adapters lost on copy), §4.1, §4.3 (licences, E0),
  §7.1 (`fm.load`), §9.2 (naive-cuEq guard), §13 (hashes).
- Source: `Experiments/Benchmark_MACE_GPU_OPTIMIZATIONS/scripts/common.py`
  (`MODELS`, `BENCHMARK_MODELS`, `MODEL_TYPES`,
  `_hybrid_keeps_last_layer_irreps`, `configure_cueq`, `validate_cueq`,
  `validate_oeq`, `load_calculator`), `scripts/model_store.py`
  (`checkpoint`, `URLS`), `workflow.load_md_calculator`,
  `langevin/upstream_mace.py` (`use_cueq_conv_fusion`, `conv_fusion_for`,
  `require_fused_cueq`), and the MACE-OFF, POLAR and OMOL label additions in
  `studies/other_models/README.md`.
- Files: `fastmace/model/loading.py`, `fastmace/model/labels.py` (the
  label table: label → source URL or MACE name, head, model type,
  functional, licence; all labels used by §12.5 and the MACE-OFF set).
  Checkpoints resolve from an explicit path, else `FASTMACE_MODELS`, else
  the user cache directory. Never a NAS path; workstation runs set
  `FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models` or pass the
  path. SHA-256 of the checkpoint, licence shown at load (a warning, once).
- Process-wide patches (`use_cueq_conv_fusion`) are installed through one
  idempotent `fastmace/_patches.py` that records what it patched, so that
  provenance can list it. `_patches.py` is shared with T08: T05 creates it,
  T08 adds entries through the conductor.
- Tests first: label table complete for the 19 labels; resolution order;
  hash recorded; naive-cuEq guard refuses a model whose fused kernels fell
  back (port from `tests/core/test_upstream_mace.py`); the hybrid converter
  keeps the last layer's irreps (tiny model).

#### 1a-T09 Neighbour list and buckets (L)

- Spec: §5 `neighbors/`, §8.1 "Edges" and "Neighbour list", §4.4.
- Source: `gpu_langevin.py`: `SkinNeighbourList` (line 638, about 250
  lines), `edge_bucket`, `round_up_edges`, `EDGE_MULTIPLE`,
  `EDGE_BUCKET_FRACTION`, `_times_3x3`, `det_3x3`, `inverse_3x3`, receiver
  sorting; `langevin/pair_search.py`. Options come from `NeighbourConfig`,
  not from `GPU_NEIGHBOUR_LIST`.
- Files: `fastmace/neighbors/skin.py`, `fastmace/neighbors/buckets.py`,
  `fastmace/neighbors/pair_search.py`.
- Tests first: pairs, shifts and counts against ASE's `neighbor_list` for
  the three structures, rattled and strained, CPU float64; GPU build equal to
  the CPU build (same pairs; order may differ); bucket arithmetic; receiver
  sorting is a permutation; the deformation-aware rebuild test of the
  variable-cell path. Port the matching parts of
  `tests/core/test_gpu_langevin.py` and `tests/studies/edge_ordering/test_receiver_sort.py`.

#### 1a-T14 Output and minimal restart (M)

- Spec: §8.1 "async output", §10.6 (only today's behaviour; the complete
  state, canonical restart points and chunked output are 1b).
- Source: the asynchronous writer inside `GPULangevin`,
  `workflow.write_restart`, `workflow.truncate_outputs`, the resume path of
  `workflow.run_crystal`.
- Files: `fastmace/io/trajectory.py`, `fastmace/io/restart.py`.
- Tests first: the async writer gives byte-identical files to synchronous
  writing; restart round trip of the 1a state; resume truncation behaves as
  today (port from `tests/core/test_workflow.py`).

#### 1a-T16 Minimal provenance (S)

- Spec: §13 (1a writes today's block; the schema-versioned version is 1b).
- Source: what `workflow.run_crystal` and `common.model_info` write to
  `run.json` / `metadata.json`, including `cueq_conv_fusion`, environment
  versions and the file-lock mode.
- Files: `fastmace/provenance.py`.
- Tests first: every key today's `run.json` has is produced (compare with a
  stored sample from `results/`), and the values that 1a adds (FastMACE
  version and commit, patches from `_patches.py`) are present.

#### 1a-T18 Validation and A/B tools (M)

- Spec: §12.2 (gate rule), §12.5 (timing: median of 4 × 300 steps after 300
  warm-up steps; setup time; peak memory; versioned JSON; alternating A/B
  against a frozen revision).
- Source: `tools/validation/npt_stress_reference.py`,
  `tools/validation/roundoff_reference.py`,
  `studies/other_models/probes/model_step_time.py`.
- Files: `fastmace/bench/step_time.py`, `fastmace/bench/ab.py` (loads the
  baseline from an exported directory given by path, and FastMACE, in one
  process, and alternates blocks), `tools/force_stress_gate.py`
  (today's rule, both engines). Two hybrid engines in one process are
  allowed; never mix hybrid and plain-cuEq compilation in one process (root
  AGENTS.md).
- Tests first: the timing statistics and the gate rule on synthetic numbers
  (CPU); the A/B loader refuses a baseline directory without a recorded
  commit.
- Until T13 exists, the A/B tool is checked baseline-against-baseline:
  that run measures the noise floor of the 3% criterion. Record it.

### Wave 2: rewrites and long range

#### 1a-T06 Rewrite registry and the linear family (L)

- Spec: §6.1, §6.3, §6.4.
- Source: `gpu_langevin.py`: `_ConstantMatmul`, `_BlockLinear`,
  `_raw_blocks`, `_ResidualBlockLinear`, `merge_residual_linears`,
  `_segments`, `_block_linear`, `block_linears`; the source-hash pins
  (`_NONLINEAR_FORWARD_SHA256`, `GENERIC_INTERACTIONS`).
- Files: `fastmace/rewrites/registry.py` (the `Rewrite` record exactly as
  §6.1, application order, requires/conflicts resolution, hash checks that
  disable a record with a logged reason, self-check runner),
  `fastmace/rewrites/linear/`.
- Tests first: registry rules (order, requires, conflicts, a hash mismatch
  disables with a reason, an `approximate` rewrite is never a default);
  port `tests/core/test_block_linear.py`; per-module self-check thresholds
  of §6.4.

#### 1a-T07 MH-1 rewrites (L, GPU at the end) — after T03, T04, T06

- Source: `gpu_langevin.py` lines 980–1220 and 1481–1650:
  `_ProjectedEmbedding`, `_FactoredRadialMLP`,
  `_FirstInteractionReassociation`, `_cutoff_folded_forward`,
  `_streaming_message`, `_attach_streaming_kernels`, `_fold_radial_cutoff`,
  `_probe_first_tensor_product`, `reassociate_first_interaction`,
  `factor_species_radial`.
- Files: `fastmace/rewrites/mh1/`, one record per §6.2 row.
- Tests first: per-rewrite self-check on tiny MH-1-class models (CPU
  float64); the stacked MH-1 package equal to the baseline's on the same
  tiny model; streaming attachment on the GPU.

#### 1a-T08 Generic, strain and engine-level rewrites (L) — after T03, T06

- Source: `gpu_langevin.py` lines 1221–1480 (`_generic_interaction_forward`,
  `_SpeciesSkip`, `_species_skip`, `_bias_free`, `generic_rewrites`), lines
  261–395 (`capture_safe_mask_head`, `first_derivative_forces*`,
  `use_unfused_full_addmm`); `upstream_mace.py` (`use_single_graph_strain`).
- Files: `fastmace/rewrites/generic/`, `fastmace/rewrites/engine.py`
  (records for `first_derivative_only`, `unfuse_full_addmm`,
  `capture_safe_mask_head`, `single_graph_strain`, installed through
  `_patches.py`).
- Tests first: port `tests/core/test_generic_rewrites.py` and the strain
  parts of `test_upstream_mace.py`; each process-wide patch installs once,
  is recorded and is reported by provenance.

#### 1a-T10 PolarMACE GPU path (L, GPU) — after T03, T05

- Spec: §11.1, §6.2 `polar_uvu`, §4.3 known issues.
- Source: `langevin/polar_gpu.py` (658 lines; its module switches go to
  `PolarConfig`). The per-engine `torch._dynamo.reset()` becomes a
  documented process constraint, enforced by T15.
- Files: `fastmace/longrange/polar.py`, `fastmace/rewrites/polar/`.
- Tests first: port `tests/core/test_polar_gpu.py`; the PR 1762 options stay
  off by default; the compiled plain-cuEq PolarMACE route is refused.

### Wave 3: force field, integrators, engine, planner

#### 1a-T11 ForceField (L, GPU) — after T05, T06–T10

- Spec: §3.2 (adapters lost on copy, process state), §7.1, §8.1
  "Encapsulated counters".
- Source: `MACEForces` (`gpu_langevin.py:1652`), `_simplify_traceable`,
  `_restore_oeq_wrappers`, `_restore_cueq_conv_fusion`, `compile_model`,
  `_COMPILED_MODELS`, `bind_streaming_count`, `configure_allocator`.
- Files: `fastmace/model/forces.py` (implementation of T02's protocol),
  `fastmace/model/compile.py`. The compiled-model cache is an object owned by
  whoever creates force fields (passed in, not a module dict). The streaming
  edge counter belongs to the force field instance, so
  `bind_streaming_count` disappears.
- Tests first: energy, forces and stress equal to `MACEForces` on tiny
  models (CPU float64, rewrites off and on); a deep copy re-applies both the
  OEq and the cuEq fusion adapters; two force fields from one compiled model
  step independently with no rebinding (the old `bind_streaming_count`
  failure as a regression test, GPU).

#### 1a-T12 State and integrators (L) — after T02 (T11 for GPU tests)

- Spec: §10.1, §10.2, §4.4 (fixed centre of mass, constraints refused),
  §12.1 "Integrator corrections" (keep `ASE_TRACE_DRAG_ALL_ELEMENTS` as an
  `IntegratorConfig` field; qualifying it is 1b).
- Source: the Langevin maths in `GPULangevin` (noise layout, `NOISE_BATCH`),
  `gpu_npt.py` (`expm_3x3`, `barostat_mass`, `external_stress`,
  `GPULangevinBAOAB`).
- Files: `fastmace/integrators/langevin.py`, `fastmace/integrators/baoab.py`,
  `fastmace/state.py` (filling T02's fields).
- Tests first: port `tests/core/test_gpu_npt.py` and the integrator parts of
  `test_gpu_langevin.py`; with the same seed the noise stream equals the
  baseline's; one eager step equals the baseline's eager step bitwise (CPU
  float64).

#### 1a-T13 CUDA-graph engine (L, GPU) — after T09, T11, T12, T14

- Spec: §8.1 (stages, pipelining, speculative slots, async output). The
  library's new items in §8.1 (re-capture policy, overflow before commit,
  runtime-state registry) are 1b and phase 2. Keep today's behaviour and
  leave clear seams for them.
- Source: the capture, replay and pipelining part of `GPULangevin`
  (`SPECULATIVE_SLOTS`, `PIPELINE_*`, `EMPTY_CACHE_BEFORE_CAPTURE`,
  `WARMUP_IN_GRAPH_POOL`, `DYNAMO_CACHE_SIZE_LIMIT`), and the BAOAB override.
- Files: `fastmace/engine/graph.py`, `fastmace/engine/constraints.py`
  (process constraints of §3.2 checked at engine creation: allocator setting
  before the first CUDA allocation, the dynamo cache limit, no hybrid and
  plain-cuEq compilation in one process).
- Tests first (GPU): captured replay equals eager steps over 200 steps
  across at least one neighbour rebuild (statistical for the hybrid,
  bitwise for plain cuEq float64); pipelined equals non-pipelined; async
  output bit-identical; the process-constraint checks refuse a mixed
  process.

#### 1a-T15 Fixed-plan planner (M) — after T01, T03, T13

- Spec: §9.1, §9.5, §4.4 (refusals at plan time), §16 ("one fixed plan").
- Source: `workflow.adapt_options_to_model`, `GPUEngine.__init__`, the
  `--backend`/`--dtype` `auto` rules, the MH-1/generic/POLAR/OMOL/MACE-OFF
  switches in `studies/other_models/README.md`.
- Files: `fastmace/planner/fixed.py`: from (model traits, atoms, integrator
  config, hardware) to a `Plan` equal to what `main.py` would run today, with
  a reason string per decision; refuses non-3D-PBC, constraints and models
  whose padding is not exact, before any compilation. The planner is the only
  code that instantiates engines.
- Tests first: for each of the 19 labels (traits from tiny stand-ins or
  recorded trait fixtures), the plan's rewrite set equals the set the
  baseline's `adapt_options_to_model` produces; refusals carry reasons.

### Wave 4: integration and acceptance

#### 1a-T17 `main.py` on FastMACE (M) — after T15, T16

- Spec: §16 ("`main.py` runs on top of FastMACE"), migration paragraph.
- **Cross-repository task.** It runs as a session in ResearchModule, started
  on purpose, under ResearchModule's AGENTS.md, on its own branch there.
  FastMACE comes in through `PYTHONPATH` pointing at a FastMACE checkout of a
  recorded commit, never as a copy or a submodule. `run.json` records that
  commit.
- Files (in ResearchModule): `Experiments/Langevin_Thermostat/langevin/fastmace_engine.py` (a
  `GPUEngine`-compatible adapter that asks FastMACE's planner for the
  engine), a `--engine fastmace` choice in `workflow.parse_args`, and the
  matching tests in `Experiments/Langevin_Thermostat/tests/core/`. The
  default stays `gpu` (baseline code) until the owner switches it after
  acceptance.
- Tests first: the CPU workflow tests of `test_workflow.py` pass with
  `--engine fastmace` where they can run on CPU; `run.json` carries the same
  keys plus FastMACE's.
- The research engine's own docs (`README.md`, `docs/configuration.md`) get
  the new choice.

#### 1a-T19 Acceptance runs and report (L, GPU, conductor-scheduled) — after T17, T18

- Speed: `fastmace.bench.ab` against the baseline export, idle GPU,
  alternating blocks; MH-1 (`matpes_r2scan` head) on NaCl, NaNbO3 and NbCl5
  3×3×3, hybrid FP32 NVT, plus one NPT case; one generic model (MP-0
  medium), POLAR-1-S and OMOL-0 XL on NaCl and NbCl5 as the secondary set.
  Pass: steady-state step time within 3% (or within the noise floor T18
  measured, if that is larger; then report both); setup time and peak memory
  reported, not judged.
- Accuracy: `tools/force_stress_gate.py` for all 19 labels (MACE-OFF on
  the organic set, with `--repeat 1 1 1` for the water box).
- Compatibility: `main.py --engine fastmace` smoke, `--profile anneal` on
  NaCl and `--quench` on its output, compared statistically with the
  baseline (`studies/annealing/README.md` measures).
- Machines: RTX 5060 Ti (WSL) and murphy or leto, sequentially, one job per
  GPU, idle check first.
- Report: `docs/validation/phase-1a.md` with run IDs, machines,
  environments, both commits and every number; JSON in
  `results/<run-id>/`.

## Order and parallelism

```text
Wave 0   T00 (spike, GPU)   T01 (switch map)   T02 (interfaces, conductor)
            │                   │                   │
Wave 1      │        T03  T04  T05  T09  T14  T16  T18     (parallel, CPU first)
            │         │    │    │
Wave 2      │        T06 ──┬── T07 (needs T03, T04)
            │              └── T08 (needs T03)
            │        T10 (needs T03, T05)
Wave 3      └──────► T11 (needs T05–T10)    T12 (needs T02)
                          └──────┬──────────────┘
                                T13 (needs T09, T11, T12, T14) ── T15
Wave 4                          T17 (needs T15, T16) ── T19 (needs T17, T18)
```

- At most four worker sessions at a time. Wave 1 is the widest point
  (seven independent tasks); run four, then the rest.
- The critical path is T02 → T06 → T07 → T11 → T13 → T15 → T17 → T19. Put
  the strongest model and the most review on it.
- GPU slots in order of need: T00 (early, it decides the hybrid route),
  T18's noise-floor run, T04/T07/T10 GPU tests, T11, T13, then T19
  exclusively.

## Checkpoints for the owner

Since 17 October 2026 the owner has delegated routine review to the
conductor ("continue as you see fit"). Checkpoints 1 and 2 became conductor
reviews, reported to the owner. Checkpoint 3 stays the owner's call.

1. After T00 (done, 17 October 2026): the answer is (a), in-process
   preparation, so no re-exec or launcher is needed.
2. After T02 (done, 17 October 2026): `docs/design/interfaces.md` reviewed
   and accepted by the conductor.
3. After T19: the acceptance report; whether `main.py` switches its default
   to `--engine fastmace`, and whether the `langevin` modules become shims
   (they stay as they are until then).

## Conductor decisions after wave 0 (17 October 2026)

From the T00, T01 and T02 reports. Later tasks follow these.

- **OEq runtime (T00): option (a).** `fastmace/kernels/_oeq_runtime.py`
  prepares the OEq build in-process. It must run before anything imports
  MACE: `import mace.calculators` already imports OpenEquivariance and
  `torch.utils.cpp_extension`, which fixes `CUDA_HOME`. So
  `import fastmace` never imports MACE, and loading (T05) calls
  `prepare_in_process` before its first MACE import. If MACE was imported
  first, the hybrid is refused with a reason and the planner falls back to
  plain cuEq. T05 promotes the module: it removes `prepare_by_reexec` and
  the launcher `main`, keeps the `RTLD_GLOBAL` preload as a guard, and owns
  the file from now on. A workstation check (coraline, leto) belongs to T19
  before T17's default switch. A persistent per-build extension cache (about
  15 s per process today) is a later task, not 1a. It would be keyed by
  torch/CUDA/OEq build, not by shape, so it is allowed under the generality
  rule.
- **Switch map (T01) accepted.** On its open questions:
  1. `GENERIC_DENSITY_FIRST` is not carried over: no gain, it stays among
     the rejected screens in `docs/rewrites.md` (phase 6).
  2. Two-stage `auto`: T15 reproduces `parse_args` followed by
     `adapt_options_to_model` exactly, including `--fold-cutoff off` turning
     off reassociation, the fused backward and streaming for the generic
     family. Simplification is deferred to phase 5.
  3. `species_skip` stays on for plain cuEq too, as today. Reproducible mode
     (phase 2) decides separately.
  4. The process-wide options are enforced as process constraints by T13's
     `engine/constraints.py`: a second force field that differs in them is
     refused in-process.
  5. Field placement stays as it is.
  6. `fused_radial_hidden` keeps its `Toggle` for CLI parity.
  7. `float64_energy` gets its field in 1b.
- **Interfaces (T02) accepted.**
  - `SimulationState` stores velocities, as the baseline integrates
    them; momenta are derived.
  - T11's class is `MACEForceField`, and the public `fm.ForceField` becomes
    a factory later (1b).
  - `p_eps` and `force_eps` move into `SimulationState` in 1b.
  - The `Hardware` record in `planner/plan.py` stays.
- **CPU float64 equality (0-T04, `docs/development.md`):** equivalence tests
  compare at `rtol=1e-12` or after a throwaway evaluation.
- **NAS pytest (0-T07):** on the NAS, pytest needs
  `--basetemp=/mnt/homes/zwingel/fastmace-cache/pytest-<name>`, because NFS
  reports a different owner uid.

## Conductor decisions after wave 1 (17 October 2026)

From the T03, T04, T05 and T09 reports. All four are merged (main eff7059
and the commit after it). Equivalence was shown bitwise wherever the baseline
is deterministic: neighbour lists on CPU and CUDA, Triton kernels in
float64, rendered OEq CUDA source, plain cuEq float64 forces (4.8e-16
eV/Å). Hybrid float32 forces agree with the baseline to 2.4–3.0e-7 eV/Å,
within the baseline's own run-to-run scatter (3.3e-7).

- **Tiny polar model (T03 Q1).** `tests/_support/tiny_models.FAMILIES["polar"]`
  keeps MACE's default `apply_cutoff=True`, unlike real POLAR-1. T10 sets
  `apply_cutoff=False` there and updates the polar expectations in
  `tests/cpu/test_capabilities.py` in the same change.
- **MACE-OFF first interaction (T03 Q2).** The research engine reports
  `reassociate_first` as supported for MACE-OFF because some interaction is
  generic, but its first block (`RealAgnosticInteractionBlock`) is not in
  `GENERIC_INTERACTIONS`. T08 checks whether the rewrite applies at all
  there. T15 reproduces the verdict either way, and the result goes into
  `docs/rewrites.md`.
- **E0 rule (T03 Q3)** is recorded in SPEC §4.3.
- **Exceptions (T05 Q1).** `CheckpointTrustError`, `HybridUnavailableError`,
  `BackendError`, `NaiveCueqError` and `ModelLicenceWarning` move to
  `fastmace/errors.py` with the §7.6 hierarchy in 1b. Until then they stay
  in `loading.py`.
- **Model cache (T05 Q3).** `"models"` joins `_cache.CACHE_KINDS` in 1b,
  together with the trust rule. Download stays opt-in and hash-checked.
- **`load()` takes explicit arguments (T05 Q4)**: accepted. The planner
  (T15) resolves `auto` and passes plain values.
- **Head functionals (T05 Q6)** for MH-1's `spice_wB97M` (wb97m-d3bj),
  `oc20_usemppbe` (pbe) and `omol` (wb97m-v): provisional. They are checked
  against the MH-1 release before §7.7's dispersion rule uses them (1b).
- **Hybrid tests in a second interpreter (T05 Q7, T04 Q3).**
  `FASTMACE_HYBRID_PYTHON` names a `mace_cueq_oeq` Python for the hybrid
  child processes; pytest runs in `mace_cueq_latest`. Both T04's streaming
  test and T05's loading test use this mechanism. No pytest is installed into
  the OEq environments.
- **Package data (T04 Q1)** for `fastmace.kernels.oeq_streaming` (`*.cuh`,
  `LICENSE`, `NOTICE.md`) is in `pyproject.toml`; a built wheel contains
  them.
- **`cuda-bindings` (T04 Q2)** is not listed: it arrives with torch's CUDA
  wheels, and a separate pin could mismatch them. Revisit with the prebuilt
  OEq wheels (phase 6).
- **3x3 helpers (T09)** stay in `fastmace.neighbors.skin`; T11 and T12
  import them from there.
- **For T11/T13 (T09 notes):** `ForceField.rebuild` calls `nl.build`,
  `use_edges` uses `nl.edges(n)`, `ForceField.buffers()` includes
  `nl.buffers()`, and padding is
  `round_up_edges(count, nl.capacity, edge_bucket(initial, fraction, multiple))`.
- **Workers in this phase run as background subagents** of the conductor
  session, in worktrees under `D:\Margraf\FastMACE-wt\`, whenever the app
  refuses to start separate sessions. The rules are the same as for sessions.

## Conductor decisions after wave 1b (17 October 2026)

T14, T16 and T18 are merged (main 7545581). Wave 1 is complete.

- **Noise floor (T18, `docs/validation/1a-noise-floor.md`).** Two copies of
  the research engine in one process on murphy agree within 0.32% with the
  4-block protocol and within 0.16% with 12 blocks, about ten times below
  the 3% criterion, and they do not interfere. T19 therefore uses
  same-process A/B with 12 blocks. T19 also:
  - swaps which engine is built first in one repeat (NaNbO3 shows a
    0.1–0.16% offset tied to the build order);
  - measures setup time and peak memory in single-engine processes
    (`--sides baseline` / `--sides fastmace`), because an A/B process mixes
    them;
  - checks that FastMACE's forces in an A/B process equal those from a
    FastMACE-only process, because both engines patch MACE process-wide;
  - uses the fuller baseline export at
    `/mnt/homes/zwingel/fastmace-baseline/1a-T18` (same tag, with
    `oeq_no_nvcc.sh`, `common.py`, structures and a `COMMIT` file). The
    slimmer `20261017` exports stay for the CPU equivalence tests.
- **Restart files (T14).** The extra keys (`velocities`, `masses`,
  `species`, `integrator_keys`, `fastmace_restart`) are accepted: FastMACE
  resumes exactly, and research-engine files still load.
  `Snapshot.wait()` in `engine/base.py` becomes `trajectory.wait(self)` in
  T13. `output_steps` moves to `fastmace.protocols` in 1b.
- **Provenance (T16).** The additions stay under the key `"fastmace"` in
  `run.json` and in the `model` block until the schema-versioned record of
  1b. `code_sha256` stays caller-supplied: T17's `main.py` passes its own
  files, and the git commit covers FastMACE.
- **Git in WSL on Windows worktrees.** WSL's git cannot read a Windows
  worktree, whose `.git` file names a `D:/` path, so `git_commit()` returns
  `(None, None)` there. On the main checkout it works. Runs that need the
  commit in provenance run from the main checkout or the NAS copy, which
  has `.source_commit`. T16's git test skips in that case.

## Conductor decisions after T10 (17 October 2026)

T10 merged (main d057513). Wave 2 status: T10 done; T06 next (critical path),
then T07 and T08.

- **Agreement.** The float64 cuEq polar path equals the research engine's
  `polar_gpu` run alone to 1.7e-13 eV/Å, with the energy equal. The stress
  matches finite differences below 1e-7 eV/Å³; MACE 0.3.16's own stress is
  wrong (issue 1642). The hybrid FP32 error against float64 is the same on
  both sides (NbCl5 1.5e-4 vs 1.8e-4 eV/Å).
- **References run in their own process (T10 Q2), binding on T19.** In a
  research-engine process that has built the hybrid engine first, the
  float64 POLAR-1-S reference on NbCl5 moves by up to 7.8e-7 eV/Å and
  5.4e-9 eV/atom. Run alone, it equals FastMACE and stock MACE to 1e-13.
  The cause is process-wide state in the research engine's setup and was
  not traced further. T19's gate therefore computes every float64 and
  float32 reference in a fresh process that builds no production engine.
  SPEC §12.3 already requires references to run "first, one configuration
  and one calculator at a time"; this adds "in a process that has built no
  production engine". Record the finding in `docs/validation/` with T19.
- **Polar registration (T10 Q1).** T06 registers `rewrites.polar.RECORDS`
  into its registry and adds `polar.installed()` to `_patches` as
  `polar_longrange`. The conductor names this in T06's brief.
- **Charge/spin (T10 Q3).** T11 calls `longrange.polar.validate_charge_spin`
  whenever it passes `total_charge` and `total_spin`.
- **uvu per instance.** Accepted: the rewrite acts on the force field's own
  model only. Deep copies of hybrid models lose MACE's OEq adapters; T11
  re-applies them (`_restore_oeq_wrappers`), as the switch map notes.

## Conductor decisions after T06 (17 October 2026)

T06 merged. The registry is in `fastmace/rewrites/registry.py`; its module
docstring is the API reference. T07 and T08 each add a family module whose
`RECORDS` maps names to §6.1 records (or dicts plus an optional
`config_field`), and place it in `RECORD_SOURCES` where the research engine
applies that family. T15 resolves with `Registry.candidates` and
`Registry.resolve`. An explicit "on" that cannot be honoured raises
`RewriteRefusedError`; it is never silently downgraded. The polar records
and the `polar_longrange` patch entry are registered.

- **cuEq on CPU picks a CUDA-only kernel.** When `cuequivariance_ops_torch`
  is importable, MACE's cuEq conversion builds the channel-wise tensor
  product with the CUDA-only `uniform_1d` kernel even for `device="cpu"`, and
  the first forward pass fails without a GPU. The tests force cuEq's `naive`
  method during conversion (`HAS_CUE_OPS` off). **T11 and T15 must do the
  same** for any CPU cuEq model, including the CPU float64 reference that
  SPEC §12.3 allows as a fallback, or refuse that route with a reason.
  `fastmace.model.loading.load(device="cpu", backend="cueq")` is the first
  place to check.

## New optimisations: screens after T19 (owner, 17 October 2026)

The owner welcomes trying new optimisations and adopting those that make
FastMACE faster while passing the accuracy gates. The general-rule
constraint of SPEC §1.1 still applies: no per-GPU or per-crystal tuning.
Phase 1a itself only reproduces the research engine, so its equivalence
tests stay exact. New ideas are therefore screened only after T19, once
FastMACE has its own engine:

- Measure with `fastmace.bench.ab`, FastMACE against itself (option off
  against on) in alternating blocks in one process, on an idle GPU, with
  12 blocks. A gain counts only if it clearly exceeds the noise floor of
  `docs/validation/1a-noise-floor.md` (0.16–0.32%).
- It must pass the force/stress gate (energy included once 1b adds it) for
  the 19 labels, plus the per-module self-check at the §6.4 thresholds.
- Adopted rewrites start as opt-in records; they become `auto` defaults
  when the gain holds on the three crystals and two GPU architectures.
  Every screen, including those with no gain, is logged in
  `docs/rewrites.md`.

Candidates:
1. **Folded species embedding (T07, 17 October 2026).** Fold the species
   embedding and the per-network linear of `factor_species`'s
   `ProjectedEmbedding` into one per-species table, a gather by species,
   so there is no submodule left for `block_linear`. Exactness: rounding.
   Expected gain is small: per-atom forward kernels only, about 10 µs per
   step, which is about 0.3% on NaCl. The research engine's contiguous
   folded weight gave nothing. It would also make the MH-1 and linear stack
   simpler.

## Risks specific to 1a

| Risk | Mitigation |
| --- | --- |
| OEq runtime cannot be set up in-process | T00 first; options (b) and (c) keep the hybrid working; owner checkpoint |
| Splitting `GPULangevin` into integrator and engine costs speed (extra launches, lost fusion) | T02 cut follows today's stage boundaries; T13 compares captured graphs node counts with the baseline before timing |
| Two engines in one A/B process interfere (compile policy, dynamo cache, streaming counter) | baseline-against-baseline run in T18 measures it; fall back to alternating processes with more blocks, documented |
| Copy drift: the research engine changes during 1a | baseline is the tag; changes to `langevin/` during 1a are listed by the conductor and ported in T17, or deferred |
| A/B noise near 3% | T18 noise floor; same-process alternation; idle desktop |
| Process-wide MACE patches (`_patches.py`) leak between force fields | recorded and idempotent; isolation tests (A → B → A) are phase 2, so 1a only records them |
