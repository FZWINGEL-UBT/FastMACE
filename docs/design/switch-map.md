# Switch map: research-engine globals and `main.py` flags → `fastmace.config`

Plan task 1a-T01. Source: ResearchModule tag `research-engine-baseline-20261017`
(commit d2faa9fd), `Experiments/Langevin_Thermostat/`. Paths below are relative to
that directory.

The research engine keeps its options in module globals
(`langevin/gpu_langevin.py`, `polar_gpu.py`, `gpu_npt.py`), which
`workflow.GPUEngine.__init__` sets from the parsed `main.py` flags before it builds
`MACEForces`, plus two process-wide MACE patches in `upstream_mace.py`. FastMACE
carries each option as a field of a frozen dataclass in
[`fastmace/config.py`](../../fastmace/config.py) (SPEC §3.2, §5, §6.1).

How to read the tables:

- **FastMACE field**: `Class.field`, or *not carried over* with the reason.
- **Default**: the value of today's `main.py` (17 October 2026), which is the
  field's default. For a global that `GPUEngine` sets from a flag, this is the
  flag's default; the global's own value in the module is noted after it.
- **`auto` rule**: the name of the rule (section "Auto resolution rules") that
  the fixed planner (1a-T15) implements. `tests/cpu/test_config.py` checks that
  every field that admits `"auto"` has exactly one rule and that the defaults
  here equal those in `config.py`.
- **Kind**: the SPEC §6.2 catalogue entry; or *engine*, *integrator*,
  *neighbour list*, *precision*, *polar* for options that are not model
  rewrites; or *never* (not carried over).

## 1. Model rewrites → `RewriteConfig`

| Research engine | FastMACE field | Default | `auto` rule | Kind |
| --- | --- | --- | --- | --- |
| `--factor-species`; `gpu_langevin.FACTOR_SPECIES_RADIAL` (passed as `MACEForces(factor_species=...)`) | `RewriteConfig.factor_species` | `"auto"` (global `False`) | `on_for_hybrid_graph_engine_if_model_supports` | §6.2 `factor_species` |
| `--fold-cutoff`; `FOLD_RADIAL_CUTOFF` | `RewriteConfig.fold_cutoff` | `"auto"` (global `False`) | `on_with_factor_species_if_model_supports` | §6.2 `fold_cutoff` |
| `--density-first`; `DENSITY_BEFORE_LINEAR` | `RewriteConfig.density_first` | `"auto"` (global `False`) | `on_with_fold_cutoff_if_model_supports` | §6.2 `density_first` |
| `--reassociate-first`; `REASSOCIATE_FIRST_INTERACTION` | `RewriteConfig.reassociate_first` | `"auto"` (global `False`) | `on_with_stage1_fold_cutoff_if_model_supports` | §6.2 `reassociate_first` |
| `GENERIC_REASSOCIATE` (set from `--reassociate-first`) | `RewriteConfig.reassociate_first` | `"auto"` (global `False`) | `on_with_stage1_fold_cutoff_if_model_supports` | §6.2 `reassociate_first` (`GENERIC_*` row) |
| `--joint-radial-input`; `JOINT_RADIAL_INPUT` | `RewriteConfig.joint_radial_input` | `"auto"` (global `False`) | `on_with_fold_cutoff_if_model_supports` | §6.2 `joint_radial_input` |
| `--fused-folded-backward`; `FUSED_FOLDED_BACKWARD` | `RewriteConfig.fused_folded_backward` | `"auto"` (global `False`) | `on_with_stage1_fold_cutoff_if_model_supports` | §6.2 `fused_folded_backward` |
| `GENERIC_FUSED_RADIAL_BACKWARD` (set from `--fused-folded-backward`) | `RewriteConfig.fused_folded_backward` | `"auto"` (global `False`) | `on_with_stage1_fold_cutoff_if_model_supports` | §6.2 `generic_fused_radial_backward` |
| `--segmented-first-sum`; `SEGMENTED_FIRST_SUM` | `RewriteConfig.segmented_first_sum` | `"auto"` (global `False`) | `on_with_reassociate_first_and_sort_receivers` | §6.2 `segmented_first_sum` |
| `--streaming-conv`; `STREAMING_CONVOLUTION` | `RewriteConfig.streaming_conv` | `"auto"` (global `False`) | `on_for_hybrid_with_stage1_fold_cutoff_and_sort_receivers_if_model_supports` | §6.2 `streaming_conv` |
| `GENERIC_STREAMING` (set from `--streaming-conv`) | `RewriteConfig.streaming_conv` | `"auto"` (global `False`) | `on_for_hybrid_with_stage1_fold_cutoff_and_sort_receivers_if_model_supports` | §6.2 `streaming_conv` |
| `--species-skip`; `GENERIC_SPECIES_SKIP` | `RewriteConfig.species_skip` | `"auto"` (global `False`) | `on_with_graph_engine_if_model_supports` | §6.2 `generic_species_skip` |
| `--fused-radial-hidden`; `FUSED_RADIAL_HIDDEN` | `RewriteConfig.fused_radial_hidden` | `"auto"` (global `False`) | `never_automatic` | §6.2 opt-in "fused radial hidden (no gain)" |
| none: new in the library (opt-2 OPT2-T02) | `RewriteConfig.fused_radial_mlp` | `"auto"` | `never_automatic` | opt-in: the radial networks' whole hidden stack, one Triton kernel per direction; conflicts with `fused_radial_hidden` |
| `--block-linear`; `BLOCK_LINEAR` | `RewriteConfig.block_linear` | `"auto"` (global `False`) | `on_for_hybrid_graph_engine` | §6.2 `block_linear` |
| `--grouped-linear`; `GROUPED_BLOCK_LINEAR` | `RewriteConfig.grouped_linear` | `"auto"` (global `False`) | `on_with_block_linear` | §6.2 `grouped_linear` |
| `--merged-residual`; `MERGED_RESIDUAL_LINEAR` | `RewriteConfig.merged_residual` | `"auto"` (global `False`) | `on_with_block_linear` | §6.2 `merged_residual` |
| — (FastMACE only: optimisation round 1, OPT-T02 stage A) | `RewriteConfig.gated_rows` | `"auto"` | `on_with_merged_residual` | `docs/design/opt-T02-fusion.md`, stage A; default on by owner decision, 20 October 2026 |
| — (FastMACE only: optimisation round 1, OPT-T02 stage B) | `RewriteConfig.segment_gemm` | `"auto"` | `on_with_gated_rows_float32_unless_reproducible` | `docs/design/opt-T02-BC-go-no-go.md`; default on by owner decision, 20 October 2026 |
| — (FastMACE only: optimisation round 1, OPT-T03 candidate 2) | `RewriteConfig.grouped_launch` | `"auto"` | `never_automatic` | `docs/design/opt-T03-small-m.md` |
| — (FastMACE only: optimisation round 1, OPT-T08 C1) | `RewriteConfig.species_tables` | `"auto"` | `never_automatic` | `docs/design/opt-T08-linear-audit.md`, C1; opt-in |
| — (FastMACE only: optimisation round 1, OPT-T08 C2) | `RewriteConfig.fold_readout_tail` | `"auto"` | `never_automatic` | `docs/design/opt-T08-linear-audit.md`, C2; opt-in |
| — (FastMACE only: optimisation round 1, OPT-T08 C3) | `RewriteConfig.fold_product_linear` | `"auto"` | `never_automatic` | `docs/design/opt-T08-linear-audit.md`, C3; opt-in, needs `species_tables` |
| `INPLACE_BLOCK_LINEAR` (no flag) | `RewriteConfig.inplace_block_linear` | `False` | — | §6.2 opt-in "in-place block linear (slower)" |
| `--single-graph-strain`; `upstream_mace.use_single_graph_strain` (process-wide patch of MACE) | `RewriteConfig.single_graph_strain` | `"auto"` | `on` | §6.2 `single_graph_strain` |
| `--polar-uvu`; `polar_gpu.REASSOCIATE_UVU` | `RewriteConfig.polar_uvu` | `True` (flag `on`) | — | §6.2 `polar_uvu` |
| none: the research engine runs every self-check (new in the library, 2-T01; filled by 2-T19) | `RewriteConfig.selfcheck_cache` | `"use"` | — | qualification cache (§6.4; no effect until 2-T19) |

## 2. Engine options → `EngineConfig`

| Research engine | FastMACE field | Default | `auto` rule | Kind |
| --- | --- | --- | --- | --- |
| `--backend` | `EngineConfig.backend` | `"auto"` | `backend_hybrid_for_graph_engine` | engine (§9.3 backend dimension) |
| `--compile-mode` | `EngineConfig.compile_mode` | `"default"` | — | engine |
| `--device` | `EngineConfig.device` | `"cuda:0"` | — | engine |
| `--pipelined` (`GPULangevin(pipelined=...)`) | `EngineConfig.pipelined` | `"auto"` | `on_with_graph_engine_unless_speculative` | §6.2 `pipelined` (engine) |
| `--async-output` | `EngineConfig.async_output` | `"auto"` | `on_with_graph_engine` | §6.2 `async_output` (engine) |
| `--speculative-graph` (`GPULangevin(speculative_graph=...)`) | `EngineConfig.speculative_graph` | `False` (flag `off`) | — | engine (§8.1) |
| `PIPELINE_EDGE_MARGIN` | `EngineConfig.pipeline_edge_margin` | `0.0` | — | engine |
| `PIPELINE_REBUILD_FRACTION` | `EngineConfig.pipeline_rebuild_fraction` | `0.8` | — | engine |
| `--cueq-conv-fusion`; `upstream_mace.use_cueq_conv_fusion` (process-wide patch of MACE's converter) | `EngineConfig.cueq_conv_fusion` | `"auto"` | `on_for_plain_cueq_unless_reproducible` | engine (§9.3 backend; §8.7 "MACE patches") |
| `FIRST_DERIVATIVE_ONLY` | `EngineConfig.first_derivative_only` | `True` | — | §6.2 `first_derivative_only` (engine) |
| `UNFUSE_FULL_ADDMM` | `EngineConfig.unfuse_full_addmm` | `True` | — | §6.2 `unfuse_full_addmm` (engine) |
| `DYNAMO_CACHE_SIZE_LIMIT` | `EngineConfig.dynamo_cache_size_limit` | `64` | — | engine (§3.2 process-wide state) |
| `ALLOCATOR_SETTINGS`, `ALLOCATOR_MIN_TORCH` (`configure_allocator()`) | `EngineConfig.expandable_segments` | `"auto"` (`"expandable_segments:True"` from torch 2.14) | `on_for_torch_2_14_unless_alloc_conf_set` | §6.2 "expandable segments allocator" (engine) |
| `EMPTY_CACHE_BEFORE_CAPTURE` | `EngineConfig.empty_cache_before_capture` | `True` | — | engine |
| `WARMUP_IN_GRAPH_POOL` | `EngineConfig.warmup_in_graph_pool` | `True` | — | engine |
| none: the research engine's `fm.calc` path recompiles on a new edge count (new in the library, 2-T01; filled by 2-T13) | `EngineConfig.dynamic_edges` | `False` | — | engine (evaluation plans; no effect until 2-T13) |
| none: the research engine loads and converts once per process (new in the library, opt-1 OPT-T06) | `EngineConfig.model_template_cache` | `True` | — | planner (`ForceField.execution_model`: later plans copy a converted template; CUDA cuEq and hybrid only) |
| none: the research engine's early rebuild waits for the force call in flight (new in the library, opt-2 OPT2-T04) | `EngineConfig.overlap_rebuild` | `False` | — | engine (`GraphEngine._rebuild(beside=True)`: a pipelined fixed-cell CUDA step's early rebuild runs on a side stream; exact) |
| none: the research engine replays ACP and B as two graphs (new in the library, opt-2 OPT2-T04) | `EngineConfig.step_graph` | `False` | — | engine (`GraphEngine._step_pipelined`: one `("ACPB", edges)` graph per step with the flags' copy and an external event inside; exact) |

## 3. Precision → `PrecisionConfig`

| Research engine | FastMACE field | Default | `auto` rule | Kind |
| --- | --- | --- | --- | --- |
| `--dtype` | `PrecisionConfig.dtype` | `"auto"` | `float32_for_hybrid_float64_for_cueq` | precision |
| TF32 off (`Benchmark_MACE_GPU_OPTIMIZATIONS/scripts/common.py:575`, `allow_tf32 = False`; `main.py` banner "TF32 off") | `PrecisionConfig.tf32` | `False` | — | precision (§1.2: never default) |
| none: the research engine sums the energy in float32 (new in the library, SPEC §6.2, §7.3; 1b-T01) | `PrecisionConfig.float64_energy` | `True` | — | precision (reference treatment, §12.1; `False` only for A/B against the research engine, 1b-T03) |

## 4. Neighbour list → `NeighbourConfig`

| Research engine | FastMACE field | Default | `auto` rule | Kind |
| --- | --- | --- | --- | --- |
| `--skin` | `NeighbourConfig.skin` | `0.3` | — | neighbour list |
| `--sort-receivers` (`MACEForces(sort_by_receiver=...)`) | `NeighbourConfig.sort_receivers` | `"auto"` | `on_for_hybrid_graph_engine` | §6.2 `sort_receivers` |
| `--gpu-neighbour-list`; `GPU_NEIGHBOUR_LIST` | `NeighbourConfig.gpu_neighbour_list` | `"auto"` (global `False`) | `on_with_graph_engine` | §6.2 `gpu_neighbour_list` |
| `EDGE_MULTIPLE` | `NeighbourConfig.edge_multiple` | `256` | — | neighbour list (buckets) |
| `EDGE_BUCKET_FRACTION` (also `GPULangevin(edge_bucket_fraction=...)`) | `NeighbourConfig.edge_bucket_fraction` | `0.05` | — | neighbour list (buckets) |
| `SkinNeighbourList(headroom=1.15)` (constructor default, not a global) | `NeighbourConfig.capacity_headroom` | `1.15` | — | neighbour list (buckets) |

## 5. PolarMACE → `PolarConfig`

| Research engine | FastMACE field | Default | `auto` rule | Kind |
| --- | --- | --- | --- | --- |
| `--polar-fukui-regularisation`; `polar_gpu.FUKUI_REGULARISATION` | `PolarConfig.fukui_regularisation` | `0.0` | — | polar; §6.2 opt-in "PR 1762" (changes the model) |
| `--polar-dipole-cap`; `polar_gpu.DIPOLE_CAP` | `PolarConfig.dipole_cap` | `None` | — | polar; §6.2 opt-in "PR 1762" (changes the model) |
| `polar_gpu.enable(headroom=0.1 if npt else 0.0)` (argument in `GPUEngine.__init__`, not a global) | `PolarConfig.kspace_headroom` | `"auto"` | `variable_cell_0_1_else_0` | polar (§11.1) |

`--polar-uvu` is a rewrite and sits in `RewriteConfig.polar_uvu` (section 1).

## 6. Integrator → `IntegratorConfig`

| Research engine | FastMACE field | Default | `auto` rule | Kind |
| --- | --- | --- | --- | --- |
| `--ensemble` | `IntegratorConfig.ensemble` | `"nvt"` | — | integrator |
| `--timestep-fs` | `IntegratorConfig.timestep_fs` | `0.5` | — | integrator |
| `--friction` | `IntegratorConfig.friction` | `0.02` | — | integrator |
| `--seed` | `IntegratorConfig.seed` | `20261001` | — | integrator |
| `GPULangevin(fixcm=True)` (set in `GPUEngine.__init__`) | `IntegratorConfig.fixcm` | `True` | — | integrator |
| `NOISE_BATCH` (also `GPULangevin(noise_batch=...)`) | `IntegratorConfig.noise_batch` | `256` | — | integrator (§3.2 hard-coded protocol choice) |
| `--pressure-GPa` | `IntegratorConfig.pressure_GPa` | `0.0` | — | integrator |
| `--cell-mode` | `IntegratorConfig.cell_mode` | `"hydrostatic"` | — | integrator |
| `--barostat-tau-fs` | `IntegratorConfig.barostat_tau_fs` | `None` | — | integrator |
| `--barostat-mass-factor` | `IntegratorConfig.barostat_mass_factor` | `1.0` | — | integrator |
| `gpu_npt.ASE_TRACE_DRAG_ALL_ELEMENTS` | `IntegratorConfig.ase_trace_drag_all_elements` | `False` | — | integrator (ASE comparison tests only; `studies/npt/README.md` "Deliberate difference from ASE 3.27") |
| `GPULangevin.set_temperature` from `workflow.run_crystal`'s ramp (no flag; 1b-T01) | `IntegratorConfig.temperature_K` | `None` | — | integrator (`None`: the caller sets the target, as `main.py` does; the public integrator records always set it) |
| `workflow.baoab_settings`: `externalstress = -pressure` (ASE units) (1b-T01) | `IntegratorConfig.external_stress_GPa` | `None` | — | integrator (ASE `externalstress` sign, scalar or six Voigt components; `None` = `-pressure_GPa`) |
| ASE `LangevinBAOAB(disable_cell_langevin=False)`, not passed by `baoab_settings` (1b-T01) | `IntegratorConfig.disable_cell_langevin` | `False` | — | integrator |

## 7. Not carried over: rejected, screen-only and never-enable switches

None of these gets a field. Each was off in every `main.py` run.

| Research engine | FastMACE field | Default | Reason (where the research engine records it) | Kind |
| --- | --- | --- | --- | --- |
| `INDUCTOR_PDL` | not carried over | `False` | Correct without graphs, wrong forces in captured-graph replays (gate ratios 1e4–1e5). `studies/further_optimizations/2026-10-15-fourth-round.md` §3; comment at `gpu_langevin.py:126` | never (§6.2 "Never enable") |
| `INDUCTOR_COMBO` | not carried over | `False` | Combo kernels / aggressive fusion: −0.07 to −0.17 ms on NaCl, +0.64 ms on NaNbO3. `studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md` (Inductor fusion table) | never (rejected screen) |
| `DECOMPOSE_SMALL_MM`, `DECOMPOSE_SMALL_MM_LIMIT` (`use_small_mm_decomposition()`) | not carried over | `False`, `129` | Inductor `decompose_mm_pass` fires but gives no gain. `STREAMED_EDGES_RESULTS.md` (joint radial layer row) | never (rejected screen) |
| `CONTIGUOUS_FOLDED_WEIGHT` | not carried over | `False` | Probe only (1.86 → 1.52 ms in a microbenchmark, never adopted). `studies/further_optimizations/2026-10-14-third-round.md` (table of probes) | never (probe) |
| `STREAMING_MUL_IR` | not carried over | `False` | Screen of the old `mul_ir` layout with transposes; `ir_mul` is the default since 14 October 2026 and the kernel layout follows the model. `STREAMED_EDGES_RESULTS.md` (`ir_mul` vs `mul_ir` table) | never (superseded) |
| `STREAMING_RECEIVER_BACKWARD` | not carried over | `False` | Exact but slower (NaNbO3 backward 4.4 vs 2.7 ms). `STREAMED_EDGES_RESULTS.md` ("receiver-owned backward … Not adopted") | never (rejected screen) |
| `JOINT_FUSED_BACKWARD` | not carried over | `False` | Below 0.1 ms, not adopted. `2026-10-14-third-round.md` §5 | never (rejected screen) |
| `ACTIVATION_MEMORY_BUDGET` | not carried over | `None` | 0.5 and 0.9 were 9–20% slower. `2026-10-15-fourth-round.md` §2 | never (rejected screen) |
| `PREFERRED_BLAS` (`use_preferred_blas()`) | not carried over | `None` | cuBLASLt faster in isolation, not in the full step. `STREAMED_EDGES_RESULTS.md` ("Not adopted") | never (rejected screen) |
| `GENERIC_DENSITY_FIRST` | not carried over | `False` | Never set by `GPUEngine`; "stays a screen option (off)" when the other `GENERIC_*` rewrites were adopted. `studies/other_models/README.md` ("Adopted (user decision, 15 October 2026)"); not in §6.2. See open question 1 | never (screen) |
| `polar_gpu.SPIN_RESTRICTED` | not carried over | `False` | Changes the model and makes results worse (POLAR-1-L NbCl3O max \|F\| 0.47 → 30.5 eV/Å). `studies/other_models/README.md` ("Spin-restricted evaluation") | never (§6.2 "Never enable") |
| `SPECULATIVE_SLOTS` | not carried over | `1` | Engine constant (one speculative step in flight); the speculative path is designed around one slot. Becomes a constant of `fastmace.engine.graph` | engine (constant) |
| `gpu_npt.EXPM_SCALING`, `gpu_npt.EXPM_DEGREE` | not carried over | `2`, `14` | Parameters of the exact 3×3 matrix exponential (float64-exact for \|A\| ≤ 1), not options. Become constants of `fastmace.integrators.baoab` | integrator (constant) |
| `--engine` | not carried over | `"gpu"` | The execution tier is the planner's decision (§8, §9.3); `ase` corresponds to the reference tier (§8.4) | engine (plan) |
| `--stress-mode` | not carried over | `"full"` | ASE engine only; FastMACE computes stress exactly when the integrator needs it (§9.3 rule 10) | engine (plan) |
| `GPULangevin(cuda_graph=device.startswith("cuda"))` | not carried over | `True` | Derived from the device; graph mode is the planner's tier (§8.1) | engine (plan) |

## 8. Not carried over: process state and module data (never config)

These module-level names are caches, saved originals or fixed data, not options.
§3.2 requires the process state to be encapsulated; the data moves to the rewrite
registry (§6.1 `source_hashes`) or stays a constant.

| Research engine | FastMACE field | Default | Where it goes | Kind |
| --- | --- | --- | --- | --- |
| `gpu_langevin._COMPILED_MODELS` | not carried over | `{}` | Process-wide compile cache (§3.2); per-instance in the engine | never |
| `gpu_langevin._MACE_COMPUTE_FORCES`, `_MACE_COMPUTE_FORCES_VIRIALS` | not carried over | `None` | Saved originals of MACE functions patched for `first_derivative_only`; no MACE patching in FastMACE (§8.7) | never |
| `gpu_langevin.FACTORED_INTERACTIONS` | not carried over | `("RealAgnosticResidualNonLinearInteractionBlock",)` | Model trait data for `rewrite_support` (`fastmace.model.capabilities`) | never (data) |
| `gpu_langevin._NONLINEAR_FORWARD_SHA256` | not carried over | `"727c37b7…"` | `Rewrite.source_hashes` (§6.1, §6.3) | never (data) |
| `gpu_langevin.GENERIC_INTERACTIONS` | not carried over | 3 classes with hashes | Model trait data and `Rewrite.source_hashes` | never (data) |
| `polar_gpu._ACTIVE`, `_PR1762_FORWARD`, `_PATCHED` | not carried over | `[]`, `None`, `False` | Process state of the class-level wrapper; per-model state in `fastmace.longrange.polar` | never |
| `polar_gpu._PR1762_EDITS` | not carried over | tuple of edits | Source-rewrite data (§6.3 "Source rewrites") | never (data) |
| `upstream_mace._MACE_SYMMETRIC_DISPLACEMENT`, `_MACE_CUEQ_RUN` | not carried over | `None` | Saved originals of patched MACE functions; FastMACE patches no MACE module (§8.7 "MACE patches") | never |
| `upstream_mace.SYMMETRIC_DISPLACEMENT_PARAMETERS`, `CUEQ_RUN_PARAMETERS`, `FUSED_CUEQ_TYPES` | not carried over | tuples | Signature checks and the naive-cuEq guard data (§9.3 rule 6) | never (data) |

## 9. `main.py` run options outside `config.py`

These flags describe the run, not the engine. They belong to the CLI and
protocols (§7.5, §10.5), the output and restart code (1a-T14) and `main.py`
on FastMACE (1a-T17), and are not fields of the six config classes.

| Research engine | FastMACE field | Default | Where it goes | Kind |
| --- | --- | --- | --- | --- |
| `--profile` | not carried over | `"slow"` | protocol (heat/cool profiles) | protocol |
| `--structures`, `--structure-file`, `--name` | not carried over | ten crystals, `None`, `None` | CLI / `main.py` | run |
| `--heat-ps`, `--cool-ps`, `--hold-ps`, `--peak-K`, `--final-K` | not carried over | profile, profile, `0.0`, profile, profile | protocol schedule | protocol |
| `--relax-fmax`, `--relax-optimizer` | not carried over | profile, `"LBFGS"` | `fastmace.optimize` / protocol | protocol |
| `--repeat` | not carried over | `(3, 3, 3)` | structure preparation | run |
| `--sample-interval`, `--trajectory-interval` | not carried over | profile | output (1a-T14) | run |
| `--restart-interval` | not carried over | `10000` | restart (1a-T14) | run |
| `--initial-displacement-A` | not carried over | `0.0` | structure preparation | run |
| `--model`, `--checkpoint` | not carried over | `"mh1-r2scan"`, NAS path | model loading (1a-T05) | run |
| `--output`, `--resume`, `--dry-run` | not carried over | `None`, `None`, `False` | CLI / `main.py` | run |
| `--quench`, `--quench-min-K`, `--quench-bin-K`, `--quench-per-bin`, `--quench-plot-only`, `--quench-optimizer`, `--quench-threads`, `--quench-fmax` | not carried over | `None`, `500.0`, `100.0`, `2`, `False`, `"LBFGS"`, `16`, `None` | quench post-processing (later phase) | run |
| `LEGACY_PROTOCOL` (resume of runs older than a flag) | not carried over | dict | research `run.json` compatibility only | never |

## 10. Safety options of phase 2 → `FallbackConfig`, `GateConfig`, `RunnerConfig`

New in the library (plan task 2-T01); the research engine has none of them. They
reach the planner through `Simulation(fallback=, gate=, runner=)` and `fm.md`
(`docs/design/contracts-2.md`).

| Research engine | FastMACE field | Default | `auto` rule | Kind |
| --- | --- | --- | --- | --- |
| none (new, SPEC §9.4, decision 29) | `FallbackConfig.fallback` | `"auto"` | `fallback_search_on_failure` | fallback (`"refuse"`: raise as in 1b) |
| none (new, SPEC §9.4 step 2) | `FallbackConfig.diagnosis_budget` | `64` | — | fallback (reduced-gate runs) |
| none (new, SPEC §9.4 step 5) | `FallbackConfig.reference_last` | `True` | — | fallback |
| none (new, SPEC §12.3 "Resources") | `GateConfig.cpu_float64_time_limit_s` | `300.0` | — | gate resources |
| none (new, SPEC §12.3 "Resources") | `GateConfig.reduced_cell` | `"auto"` | `largest_smaller_supercell_at_least_min_height` | gate resources (2-T05) |
| none (new, SPEC §12.3 "Resources") | `GateConfig.reduced_cell_min_height_cutoffs` | `2.0` | — | gate resources |
| none (new, SPEC §8.7) | `RunnerConfig.python` | `None` | — | subprocess runner (`None`: `sys.executable`) |
| none (new, SPEC §8.7) | `RunnerConfig.environment` | `()` | — | subprocess runner |
| none (new, SPEC §8.7) | `RunnerConfig.inherit_environment` | `True` | — | subprocess runner |
| none (new, SPEC §8.7) | `RunnerConfig.timeout_s` | `None` | — | subprocess runner |
| none (new, SPEC §8.7) | `RunnerConfig.stall_timeout_s` | `1800.0` | — | subprocess runner |
| none (new, OPT-T07) | `RunnerConfig.persistent` | `True` | — | subprocess runner: one worker child per `Simulation` (`False`: a fresh child per `run()`) |

## Auto resolution rules

The research engine resolves `auto` in two stages, and the result depends on the
order, so 1a-T15 must reproduce both:

- **Stage 1**, `workflow.parse_args`, before the model is loaded: from
  `--engine`, `--backend` and the other flags only.
- **Stage 2**, `workflow.adapt_options_to_model`, after loading: every option of
  `MODEL_DEPENDENT_OPTIONS` that is still on but that
  `gpu_langevin.rewrite_support(model)` does not support is switched off if it
  was `auto` (reason recorded in `auto_options_off`), and refused if it was
  `on`. Only for the gpu engine.

"Graph engine" below means the research engine's `--engine gpu` (FastMACE's
tier 1, §8.1); "hybrid" the resolved backend. Support keys are those of
`rewrite_support`.

| Rule | Statement | Resolves |
| --- | --- | --- |
| `backend_hybrid_for_graph_engine` | `hybrid` for the graph engine, `cueq` for the reference (ASE) path. | `EngineConfig.backend` |
| `float32_for_hybrid_float64_for_cueq` | `float32` with the hybrid backend, `float64` with plain cuEq. | `PrecisionConfig.dtype` |
| `on_for_hybrid_graph_engine` | On with the graph engine and the hybrid backend; off otherwise. Not model-dependent. | `NeighbourConfig.sort_receivers`, `RewriteConfig.block_linear` |
| `on_with_graph_engine` | On with the graph engine, any backend. | `NeighbourConfig.gpu_neighbour_list`, `EngineConfig.async_output` |
| `on_with_graph_engine_unless_speculative` | On with the graph engine unless `speculative_graph`. | `EngineConfig.pipelined` |
| `on_for_hybrid_graph_engine_if_model_supports` | Stage 1: on with the graph engine and the hybrid backend. Stage 2: off unless support `factor_species` (an interaction of `FACTORED_INTERACTIONS`, MH-1 class). | `RewriteConfig.factor_species` |
| `on_with_factor_species_if_model_supports` | Stage 1: on when `factor_species` is on (stage 1). Stage 2: off unless support `fold_cutoff` (also `apply_cutoff=False`). | `RewriteConfig.fold_cutoff` |
| `on_with_fold_cutoff_if_model_supports` | Stage 1: on when `fold_cutoff` is on (stage 1). Stage 2: off unless support `fold_cutoff`. | `RewriteConfig.density_first`, `RewriteConfig.joint_radial_input` |
| `on_with_stage1_fold_cutoff_if_model_supports` | Stage 1: on when `fold_cutoff` is on **at stage 1**. Stage 2: off unless support `reassociate_first` resp. `fused_folded_backward`, which MP-0 and density-family models have through the `GENERIC_*` variant although their `fold_cutoff` is switched off in stage 2. | `RewriteConfig.reassociate_first`, `RewriteConfig.fused_folded_backward` |
| `on_with_reassociate_first_and_sort_receivers` | Stage 1: on when `reassociate_first` and `sort_receivers` are on. Stage 2: off unless support `reassociate_first`. | `RewriteConfig.segmented_first_sum` |
| `on_for_hybrid_with_stage1_fold_cutoff_and_sort_receivers_if_model_supports` | Stage 1: on with the hybrid backend, `fold_cutoff` (stage 1) and `sort_receivers`. Stage 2: off unless support `streaming_conv` (folded MH-1 class or a `GENERIC_INTERACTIONS` model). | `RewriteConfig.streaming_conv` |
| `on_with_graph_engine_if_model_supports` | Stage 1: on with the graph engine, **any backend**. Stage 2: off unless support `species_skip` (a `GENERIC_INTERACTIONS` model). | `RewriteConfig.species_skip` |
| `never_automatic` | Off; only `"on"` enables it (`fused_radial_hidden` and `fused_radial_mlp` require `fold_cutoff`, `grouped_launch` and `fold_readout_tail` require `block_linear`, `fold_product_linear` requires `block_linear` and `species_tables`). | `RewriteConfig.fused_radial_hidden`, `RewriteConfig.fused_radial_mlp`, `RewriteConfig.grouped_launch`, `RewriteConfig.species_tables`, `RewriteConfig.fold_readout_tail`, `RewriteConfig.fold_product_linear` |
| `on_with_block_linear` | On when `block_linear` is on. `merged_residual` acts only in interactions where `density_first` applies; the rewrite's `requires` handles that, not the rule. | `RewriteConfig.grouped_linear`, `RewriteConfig.merged_residual` |
| `on_with_merged_residual` | On when `merged_residual` is on, so wherever block linear applies with `density_first` (owner decision, 20 October 2026). Stage 2: off unless the model supports `fold_cutoff`; the rewrite's own `applies_to` (MACE's `GatedEquivariantBlock` in a factored interaction) decides where it acts. | `RewriteConfig.gated_rows` |
| `on_with_gated_rows_float32_unless_reproducible` | On when `gated_rows` is on, except in float64 plans and `reproducible=True` plans, which keep the stock products and join (owner decision, 20 October 2026); an explicit `"on"` is kept (in float64 its op runs the PyTorch path). At run time, per force call, the shared shape rule (`fastmace.kernels.triton.tile_rule`) picks the Triton GEMM; where no product qualifies (NaCl-sized cells) the stock products and join run. | `RewriteConfig.segment_gemm` |
| `on` | On for every engine; acts only where stress is computed (NPT, stress-path relaxations). | `RewriteConfig.single_graph_strain` |
| `on_for_plain_cueq_unless_reproducible` | Plain cuEq backend: on, except in a `reproducible=True` plan, which needs the unfused convolution (§9.3, owner decision 13). Hybrid: not applicable (the research engine records `None`). | `EngineConfig.cueq_conv_fusion` |
| `on_for_torch_2_14_unless_alloc_conf_set` | Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before the first CUDA allocation when torch ≥ 2.14 and the variable is unset; otherwise leave the allocator alone. | `EngineConfig.expandable_segments` |
| `variable_cell_0_1_else_0` | `0.1` for a variable-cell integrator (NPT), `0.0` otherwise. | `PolarConfig.kspace_headroom` |
| `fallback_search_on_failure` | On a failing build, compile, capture, self-check or gate before the first step, the §9.4 search (diagnosis, ddmin within the budget, the terminal sequence), each candidate validated before activation (2-T15); nothing happens on a run that does not fail. `"refuse"` raises as in 1b. | `FallbackConfig.fallback` |
| `largest_smaller_supercell_at_least_min_height` | The gate's reduced cell (2-T05): the largest supercell of the same primitive cell that is smaller than the production cell and whose every height is at least `reduced_cell_min_height_cutoffs` × the cutoff; none if no such cell exists. | `GateConfig.reduced_cell` |

An explicit `"on"` keeps the research engine's refusals: a dependency that is off
(for example `fold_cutoff="on"` without `factor_species`) or an unsupported
model is an error with a reason, never a silent downgrade.

## Counts

- Research-engine module-level names: 66 (`gpu_langevin.py` 48, `polar_gpu.py` 8,
  `gpu_npt.py` 3, `upstream_mace.py` 5 names plus its two process-wide patch
  functions). 36 are options carried over into fields (sections 1–6), 15 are
  rejected or constant options (section 7), 15 are process state or data
  (section 8).
- `main.py` flags: 67. 36 map to fields, 2 (`--engine`, `--stress-mode`) are plan
  decisions, 29 are run options outside `config.py` (section 9).
- Fields in `config.py` after phase 1a: 53 (`RewriteConfig` 16, `EngineConfig` 15,
  `PrecisionConfig` 2, `NeighbourConfig` 6, `PolarConfig` 3,
  `IntegratorConfig` 11), 23 of them admit `"auto"`. Four come from neither a
  global nor a flag: `fixcm` and the two headrooms (constructor arguments) and
  `tf32` (set in the benchmark scripts' `common.py`).
- Phase 1b (plan task 1b-T01) added four fields, 57 in all (`PrecisionConfig` 3,
  `IntegratorConfig` 14): `float64_energy` (new in the library),
  `temperature_K` (the caller's ramp in 1a), `external_stress_GPa` (derived from
  `--pressure-GPa` in the research engine) and `disable_cell_langevin` (ASE's
  default there). None admits `"auto"`.
- Phase 2 (plan task 2-T01) added two fields to existing records,
  `EngineConfig.dynamic_edges` and `RewriteConfig.selfcheck_cache`, and three
  records with 11 fields (section 10: `FallbackConfig` 3, `GateConfig` 3,
  `RunnerConfig` 5), 70 in all. OPT-T07 added `RunnerConfig.persistent` (71). Two admit `"auto"` (`FallbackConfig.fallback`,
  `GateConfig.reduced_cell`).
  None comes from the research engine.
- Optimisation round 1 (plan task OPT-T06) added `EngineConfig.model_template_cache`,
  71 in all. It does not come from the research engine, which loaded once per process.
- Optimisation round 2 (plan task OPT2-T04) added `EngineConfig.overlap_rebuild` and
  `EngineConfig.step_graph`, 73 in all (both opt-in).
- Optimisation round 2 (plan task OPT2-T02) added `RewriteConfig.fused_radial_mlp`, opt-in
  (`never_automatic`), 74 in all. It does not come from the research engine.

## Open questions for the conductor

1. **`GENERIC_DENSITY_FIRST`** is a screen option that §6.2 does not mention;
   the research engine keeps it off. Not carried over. Should it become a
   §6.2 opt-in row instead?
2. **Stage-1 coupling of the generic rewrites.** For MP-0 and density-family
   models, `reassociate_first`, `fused_folded_backward` and `streaming_conv`
   follow the stage-1 `fold_cutoff`, which the model then turns off. So
   `fold_cutoff="off"` on an MP-0 model also turns those three off, and with
   `backend="cueq"` they are off because `factor_species` is off at stage 1.
   1a-T15 reproduces this for plan equality; whether FastMACE keeps the
   coupling after phase 1a is a design choice.
3. **`species_skip` on plain cuEq.** Its `auto` is on for the graph engine
   with any backend (unlike the other model rewrites). It is exact; the
   reproducible plan of §9.3 turns "hybrid-only rewrites" off, which does not
   cover it.
4. **Process-wide fields.** `first_derivative_only`, `unfuse_full_addmm`,
   `dynamo_cache_size_limit`, `expandable_segments` and `cueq_conv_fusion`
   act on torch, Inductor or MACE for the whole process in the research
   engine. As config fields they are per instance; the planner must treat
   them as process constraints (§8.7) until each is made per-model.
5. **Placement.** `sort_receivers` and `gpu_neighbour_list` are §6.2 rows but
   live in `NeighbourConfig` (they configure `fastmace.neighbors`);
   `polar_uvu` is a rewrite and lives in `RewriteConfig`, not `PolarConfig`.
6. **`fused_radial_hidden`** keeps `main.py`'s literal `"auto"` (= off). A
   plain `bool = False`, like `inplace_block_linear`, would say the same.
7. **`float64_energy`** (§6.2, always on, new in the library) has no switch in
   the research engine and gets no field. **Revised in phase 1b** (1b-T01): it is
   `PrecisionConfig.float64_energy = True`, so that `False` can reproduce 1a's
   energy for A/B against the research engine (plan phase-1b, task 1b-T03).
