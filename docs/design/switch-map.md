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
| — (FastMACE only: optimisation round 2, OPT2-T03) | `RewriteConfig.folded_backward_tiles` | `"auto"` | `on_with_fused_folded_backward` | `kernels/triton/skinny_linear.py` (`tile_choice`); exact; default on by owner decision, 21 October 2026 (OPT-DEF-2) |
| `--fused-radial-hidden`; `FUSED_RADIAL_HIDDEN` | `RewriteConfig.fused_radial_hidden` | `"auto"` (global `False`) | `never_automatic` | §6.2 opt-in "fused radial hidden (no gain)" |
| — (FastMACE only: optimisation round 3, OPT3-T05) | `RewriteConfig.direct_node_feats` | `"auto"` | `never_automatic` | `docs/design/opt3-T05-layout-audit.md`; exact, opt-in, needs `fold_cutoff` |
| — (FastMACE only: optimisation round 3, OPT3-T05) | `RewriteConfig.folded_constants` | `"auto"` | `never_automatic` | `docs/design/opt3-T05-layout-audit.md`; exact, opt-in, needs `fold_cutoff` |
| — (FastMACE only: optimisation round 6, OPT6-T02) | `RewriteConfig.chunked_conv_backward` | `"auto"` | `on_with_streaming_conv_and_fused_folded_backward` | `docs/validation/opt6-T01.md`, `docs/validation/opt6-T02.md`; exact; default on by owner decision 60 (27 October 2026); cells whose weights fit `UNCHUNKED_L2_MULTIPLE` (4) L2s run unchunked inside the op |
| — (FastMACE only: optimisation round 8, OPT8-T02) | `RewriteConfig.fused_folded_forward` | `"auto"` | `on_for_mh1_float32_unless_reproducible` | `docs/validation/opt8-T01.md` (T01b), `docs/validation/opt8-T02.md`; exact (the stored weights and the message bitwise the chain's on sm_120); default on for MH-1 by owner decision 64 (27 October 2026); needs `chunked_conv_backward`; refused off float32; launch rule from the GPU's shared memory and SM count (`oeq_streaming/fused_forward.py`) |
| — (FastMACE only: optimisation round 6, OPT6-T05) | `RewriteConfig.radial_tables` | `"auto"` | `on_with_fold_cutoff_unless_reproducible` | `docs/validation/opt6-T04.md`, `docs/validation/opt6-T05.md`; certified (owner decision 61); default on by owner decisions 62 (float32) and 63 (float64, `docs/validation/f64-T01.md`), 27 October 2026; needs `fold_cutoff`; float64 tables refused in reproducible plans |
| — (FastMACE only: ideas-20261027 D2, owner decision 65) | `RewriteConfig.w_tables` | `"auto"` | `on_for_mh1_float64_on_measured_gpus_decision_65` | rewrite (`fastmace.rewrites.radial`, record `w_tables`, rounding: the second interaction's per-edge weights straight from per-piece folded tables, `fastmace.kernels.triton.w_table`; float64 plans outside reproducible ones, with `radial_tables` and `chunked_conv_backward`; `plans/ideas-20261027.md`) |
| — (FastMACE only: optimisation round 9, OPT9-T03, owner decision 66) | `RewriteConfig.wprime_backward` | `"auto"` | `on_for_mh1_float32_unless_reproducible_on_decision_66_gpus` | rewrite (`fastmace.rewrites.radial`, record `wprime_backward`, rounding: the second interaction's radial r-gradient inside the streaming backward, each edge's weight gradient dotted with `w'(r)` from per-piece derivative tables, `fastmace.kernels.oeq_streaming.wprime_backward`; drops the `dw` chunk stores, the folded input gradient and the table backward; float32 plans outside reproducible ones, with `radial_tables` and `chunked_conv_backward`; `plans/opt-9.md`, `docs/validation/opt9-T03.md`) |
| — (FastMACE only: plans/capacity.md CAP-T07) | `RewriteConfig.recompute_conv_weights` | `"auto"` | `on_by_rule_12_when_the_plan_does_not_fit_without_it` | rewrite (`fastmace.rewrites.radial`, record `recompute_conv_weights`, exact: `wprime_backward`'s backward builds the per-edge weights again from the edge lengths in edge chunks instead of keeping the `[E, W]` tensor, `fastmace.kernels.oeq_streaming.wprime_backward.streaming_conv_wprime_backward_recompute`; requires `wprime_backward`; `docs/validation/capacity-T07.md`) |
| `--block-linear`; `BLOCK_LINEAR` | `RewriteConfig.block_linear` | `"auto"` (global `False`) | `on_for_hybrid_graph_engine` | §6.2 `block_linear` |
| `--grouped-linear`; `GROUPED_BLOCK_LINEAR` | `RewriteConfig.grouped_linear` | `"auto"` (global `False`) | `on_with_block_linear` | §6.2 `grouped_linear` |
| `--merged-residual`; `MERGED_RESIDUAL_LINEAR` | `RewriteConfig.merged_residual` | `"auto"` (global `False`) | `on_with_block_linear` | §6.2 `merged_residual` |
| — (FastMACE only: optimisation round 1, OPT-T02 stage A) | `RewriteConfig.gated_rows` | `"auto"` | `on_with_merged_residual` | `docs/design/opt-T02-fusion.md`, stage A; default on by owner decision, 20 October 2026 |
| — (FastMACE only: optimisation round 1, OPT-T02 stage B) | `RewriteConfig.segment_gemm` | `"auto"` | `on_with_gated_rows_float32_unless_reproducible` | `docs/design/opt-T02-BC-go-no-go.md`; default on by owner decision, 20 October 2026 |
| — (FastMACE only: optimisation round 10, OPT10-T01) | `RewriteConfig.gate_epilogue` | `"auto"` | `never_automatic` | `fastmace/kernels/triton/gated_products.py`; the gate's backward in `linear_2`'s input-gradient GEMM, one atom-tiled Triton kernel per gated segment where the shape rule picks Triton; rounding; float32 only, not beside `tf32x3_gemms`; requires `segment_gemm` (`docs/validation/opt10-T01.md`) |
| — (FastMACE only: optimisation round 1, OPT-T03 candidate 2) | `RewriteConfig.grouped_launch` | `"auto"` | `on_with_block_linear_float32_unless_reproducible_for_generic_mace_family_and_mh1_on_decision_66_gpus` | `docs/design/opt-T03-small-m.md`; default on for the generic MACE family by owner decision 57, 24 October 2026 (`docs/validation/opt4-T04.md`) |
| — (FastMACE only: optimisation round 3, OPT3-T04) | `RewriteConfig.combo_kernels` | `"auto"` | `never_automatic` | `fastmace/rewrites/inductor.py`; Inductor's pointwise combo kernels, rounding, not adopted (`docs/validation/opt3-T04.md`) |
| — (FastMACE only: ideas-20261027 D3, owner decision 65) | `RewriteConfig.save_transcendentals` | `"auto"` | `on_for_mh1_float64_on_measured_gpus_decision_65` | `fastmace/rewrites/inductor.py`; a custom AOTAutograd partitioner (`custom_partitioner_fn`) that saves transcendental ops for the backward instead of recomputing them; exact (`plans/ideas-20261027.md`) |
| — (FastMACE only: optimisation round 4, OPT4-T03) | `RewriteConfig.tf32x3_gemms` | `"auto"` | `never_automatic` | `fastmace/rewrites/precision.py`; every atom GEMM as 3xTF32 (Triton `tf32x3`), rounding, not IEEE-equivalent under cancellation; float32 on compute capability ≥ 8.0; accuracy screen (`docs/validation/opt4-T03.md`) |
| — (FastMACE only: optimisation round 4, OPT4-T02) | `RewriteConfig.fx_e3nn_linear` | `"auto"` | `on_for_hybrid_float32_unless_reproducible_for_qualified_polar1_checkpoints` | `fastmace/rewrites/codegen.py`; e3nn's `o3.Linear` as fx graphs instead of TorchScript, so dynamo traces PolarMACE's forward; rounding; on for POLAR-1-S, -M and -L by decision 58 (`docs/validation/opt4-T02.md`) |
| — (FastMACE only: optimisation round 4, OPT4-T02) | `RewriteConfig.polar_graph_sums` | `"auto"` | `on_for_hybrid_float32_unless_reproducible_for_qualified_polar1_checkpoints` | `fastmace/rewrites/polar/graph_sums.py`; PolarMACE's one-graph per-graph sums as a sum and a broadcast instead of `scatter_add_` and an index (float64 `index_put` backward); rounding, deterministic; on for POLAR-1-S, -M and -L by decision 58 (`docs/validation/opt4-T02.md`) |
| — (FastMACE only: optimisation round 1, OPT-T08 C1) | `RewriteConfig.species_tables` | `"auto"` | `on_unless_reproducible_if_model_supports` | `docs/design/opt-T08-linear-audit.md`, C1; default on by owner decision, 21 October 2026 (OPT-DEF-2); float64 too by owner decision 63, 27 October 2026 |
| — (FastMACE only: optimisation round 1, OPT-T08 C2) | `RewriteConfig.fold_readout_tail` | `"auto"` | `on_with_block_linear_unless_reproducible_if_model_supports` | `docs/design/opt-T08-linear-audit.md`, C2; default on by owner decision, 21 October 2026 (OPT-DEF-2); float64 too by owner decision 63, 27 October 2026 |
| — (FastMACE only: optimisation round 1, OPT-T08 C3) | `RewriteConfig.fold_product_linear` | `"auto"` | `on_with_species_tables_and_block_linear_if_model_supports` | `docs/design/opt-T08-linear-audit.md`, C3; needs `species_tables`; default on by owner decision, 21 October 2026 (OPT-DEF-2); float64 too by owner decision 63, 27 October 2026 |
| — (FastMACE only: fp32 profile, 28 October 2026) | `RewriteConfig.species_residual` | `"auto"` | `never_automatic` | `docs/validation/fp32-profile-20261028.md`; rounding, opt-in; the first interaction's species-only residual product as per-species rows; needs `species_tables` and `merged_residual` |
| `INPLACE_BLOCK_LINEAR` (no flag) | `RewriteConfig.inplace_block_linear` | `False` | — | §6.2 opt-in "in-place block linear (slower)" |
| `--single-graph-strain`; `upstream_mace.use_single_graph_strain` (process-wide patch of MACE) | `RewriteConfig.single_graph_strain` | `"auto"` | `on` | §6.2 `single_graph_strain` |
| `--polar-uvu`; `polar_gpu.REASSOCIATE_UVU` | `RewriteConfig.polar_uvu` | `True` (flag `on`) | — | §6.2 `polar_uvu` |

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
| none (new in the library: f64-T01, owner decision 63, 27 October 2026) | `EngineConfig.fp64_emulation` | `"auto"` | `on_for_float64_cuda_with_reduced_rate_fp64_unless_reproducible` | engine (`model.compile.configure_fp64_emulation`: `CUBLAS_EMULATE_DOUBLE_PRECISION=1`, `CUBLAS_EMULATION_STRATEGY=eager` before the first float64 product, set by `FixedPlanner.build`; the build warns when a cuBLAS handle created earlier keeps native products; `docs/validation/f64-T01.md`) |
| `on_for_mh1_float64_on_measured_gpus_decision_65` | On in float64 MH-1 plans that are not reproducible, on the GPU models of `planner.fixed.DECISION_65_GPUS` only, matched by exact device name ("NVIDIA GeForce RTX 5060 Ti", "NVIDIA RTX 4000 Ada Generation": same-process A/Bs in both orders measured a speed-up, `plans/ideas-20261027.md`); off on every other GPU (the A100 measured slower; unmeasured GPUs such as the L4 stay off although they share a compute capability), on the CPU, in float32 and reproducible plans, and for other models (owner decision 65, 28 October 2026). `w_tables` also needs `radial_tables` and `chunked_conv_backward` and at most `DECISION_65_MAX_SPECIES` (3) species (its tables grow with the square of the species; 1.3 GB on NaNbO3). The off reason names the GPU. Adding a GPU needs its own A/B of all three in both orders. | `RewriteConfig.w_tables`, `RewriteConfig.save_transcendentals` |
| `dynamic_decision_65_d1_suspended_after_recapture_defect` | `"dynamic"` (cuBLAS's own choice) everywhere. Decision 65's D1 default, 56 fixed mantissa bits in float64 MH-1 plans on `DECISION_65_GPUS`, is suspended (owner, 28 October 2026): with them, a graph captured during a run computed wrong energies and forces, and the step's small products were ~8 bits accurate (fp64-D1 fix, `docs/validation/fp64-d1-relax.md`). An explicit int still applies; no effect where the emulation is off. | `EngineConfig.fp64_emulation_mantissa_bits` |
| none (new in the library: ideas-20261027 D1, owner decision 65, 28 October 2026) | `EngineConfig.fp64_emulation_mantissa_bits` | `"auto"` | `dynamic_decision_65_d1_suspended_after_recapture_defect` | engine (`model.compile.configure_fp64_emulation`: with `fp64_emulation` on and an int resolved, `CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT` before the first float64 product, unless the environment sets it; `"dynamic"` keeps cuBLAS's dynamic choice; an int from 8 to 128 fixes the bits; `plans/ideas-20261027.md`) |
| `EMPTY_CACHE_BEFORE_CAPTURE` | `EngineConfig.empty_cache_before_capture` | `True` | — | engine |
| `WARMUP_IN_GRAPH_POOL` | `EngineConfig.warmup_in_graph_pool` | `True` | — | engine |
| none: the research engine's `fm.calc` path recompiles on a new edge count (new in the library, 2-T01; filled by 2-T13) | `EngineConfig.dynamic_edges` | `True` (owner decision 59, 24 October 2026) | — | engine (`MACEForceField.use_edges` marks the edge dimension dynamic; evaluation and compiled-tier plans always set it, 2-T13 and 3-T06; on graph-tier MD plans the padded first capture reuses the first compiled call's graph, so a start compiles once instead of twice, opt-4 OPT4-T05; plans that compile nothing turn it off) |
| none: the research engine loads and converts once per process (new in the library, opt-1 OPT-T06) | `EngineConfig.model_template_cache` | `True` | — | planner (`ForceField.execution_model`: later plans copy a converted template; CUDA cuEq and hybrid only) |
| none: the research engine's early rebuild waits for the force call in flight (new in the library, opt-2 OPT2-T04) | `EngineConfig.overlap_rebuild` | `True` | — | engine (`GraphEngine._rebuild(beside=True)`: a pipelined fixed-cell CUDA step's early rebuild runs on a side stream; exact) |
| none: the research engine replays ACP and B as two graphs (new in the library, opt-2 OPT2-T04) | `EngineConfig.step_graph` | `False` | — | engine (`GraphEngine._step_pipelined`: one `("ACPB", edges)` graph per step with the flags' copy and an external event inside; exact) |
| none: the research engine runs the integrator stage as PyTorch operations (new in the library, opt-3 OPT3-T03) | `EngineConfig.fused_integrator` | `True` | — | engine (`GraphEngine._stage_ac`/`_stage_acp`: fixed-cell CUDA Langevin's half kick, drift, neighbour checks, compaction and flags in four Triton kernels; exact, also in reproducible plans; default on by owner decision, 21 October 2026; NPT, CPU and no Triton keep the PyTorch stage) |
| none: the research engine re-captures without a budget (new in the library, SPEC §8.1; 3-T01) | `EngineConfig.recapture_budget` | `"auto"` | `recapture_budget_from_workload` | engine (re-capture policy, 3-T04: past the budget the run warns, records and the report recommends the compiled tier; `0`: none) |
| `DYNAMO_CACHE_SIZE_LIMIT` (the same limit, per graph key; new as a cap in the library, SPEC §8.1; 3-T01) | `EngineConfig.graph_variant_cap` | `64` | — | engine (captured variants kept at once, at most 64 and at most `dynamo_cache_size_limit`; 3-T04) |
| none: the research engine keeps every force-graph variant (new in the library, plans/capacity.md CAP-T01) | `EngineConfig.force_graphs` | `"auto"` | `fit_when_capturing_on_cuda_else_keep` | engine (`GraphEngine._limit_force_graphs`: `"one"` releases the captured graphs and the pool before a new force variant, `"fit"` only when it would not fit the free memory; the padded size is kept from then on; exact) |

## 3. Precision → `PrecisionConfig`

| Research engine | FastMACE field | Default | `auto` rule | Kind |
| --- | --- | --- | --- | --- |
| `--dtype` | `PrecisionConfig.dtype` | `"auto"` | `float32_for_hybrid_float64_for_cueq` | precision |
| TF32 off (`Benchmark_MACE_GPU_OPTIMIZATIONS/scripts/common.py:575`, `allow_tf32 = False`; `main.py` banner "TF32 off") | `PrecisionConfig.tf32` | `False` | — | precision (§1.2: never default) |
| none: the research engine sums the energy in float32 (new in the library, SPEC §6.2, §7.3; 1b-T01) | `PrecisionConfig.float64_energy` | `True` | — | precision (reference treatment, §12.1; `False` only for A/B against the research engine, 1b-T03) |
| none (new in the library, SPEC §10.4; 3-T01) | `PrecisionConfig.mc_energy` | `"auto"` | `force_precision_if_move_energy_gate_passes_else_float64` | precision (MC energies; the planner's resolution of `Workload.mc_energy_precision`, 3-T07 and 3-T11) |

## 4. Neighbour list → `NeighbourConfig`

| Research engine | FastMACE field | Default | `auto` rule | Kind |
| --- | --- | --- | --- | --- |
| `--skin` | `NeighbourConfig.skin` | `"auto"` (`main.py` 0.3) | `rule_11_skin_from_workload` | neighbour list (`"auto"` resolves to 0.3 Å, `DEFAULT_SKIN_A`, until rule 11 in 3-T05; an explicit float is kept) |
| `--sort-receivers` (`MACEForces(sort_by_receiver=...)`) | `NeighbourConfig.sort_receivers` | `"auto"` | `on_for_hybrid_graph_engine` | §6.2 `sort_receivers` |
| `--gpu-neighbour-list`; `GPU_NEIGHBOUR_LIST` | `NeighbourConfig.gpu_neighbour_list` | `"auto"` (global `False`) | `on_with_graph_engine` | §6.2 `gpu_neighbour_list` |
| `EDGE_MULTIPLE` | `NeighbourConfig.edge_multiple` | `256` | — | neighbour list (buckets) |
| `EDGE_BUCKET_FRACTION` (also `GPULangevin(edge_bucket_fraction=...)`) | `NeighbourConfig.edge_bucket_fraction` | `0.05` | — | neighbour list (buckets) |
| `SkinNeighbourList(headroom=1.15)` (constructor default, not a global) | `NeighbourConfig.capacity_headroom` | `1.15` | — | neighbour list (buckets) |
| none (new in the library, SPEC §11.2, §9.3 rule 11; 3b-T02) | `NeighbourConfig.d3_skin` | `"auto"` | `rule_11_d3_skin_from_workload` | D3 pair list (its own skin; read only when the force field has D3; an explicit float is kept) |
| none (new in the library, SPEC §9.3 rule 11; 3-T01) | `NeighbourConfig.headroom` | `"auto"` | `rule_11_headroom_from_workload` | neighbour list (buckets; `"fixed"`: `capacity_headroom` and `edge_bucket_fraction` as given; `"auto"` resolves to `"fixed"` until 3-T05) |

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
| ASE's projection, hard-coded in `GPULangevin._stage_a` (no flag; 3-T22) | `IntegratorConfig.fixcm_projection` | `"auto"` | `mass_weighted_under_nvt_fixcm_else_ase` | integrator (`"mass_weighted"`: the canonical projection for unequal masses, docs/design/fixcm-projection.md; `"auto"`: `"mass_weighted"` under NVT `fixcm`, for MD (decision 37, 3-T22c) and MC (3-T22b), else `"ase"`; the research engine and the ASE comparisons pin `"ase"`) |
| none: `GPULangevin` or `GPULangevinBAOAB` by `--ensemble` (4-T01) | `IntegratorConfig.scheme` | `None` | — | integrator (the registry scheme; `None` is the ensemble's default, `config.DEFAULT_SCHEMES`: `"langevin"`, `"baoab"`, `"nve"`; docs/design/physics.md) |
| none: 3N in `GPULangevin` (4-T01) | `IntegratorConfig.dof` | `"auto"` | `owner_decision_d2_default` | integrator (`"fastmace"`: 3N - 3 under `fixcm`, 3(N - n_fixed) with `FixAtoms`; `"ase"`: the scheme's ASE class; `"auto"`: `registry.DOF_AUTO`, `"fastmace"` under D2 (a); Langevin and BAOAB keep their count) |
| none (4-T01) | `IntegratorConfig.parameters` | `()` | — | integrator (scheme parameters from the public records, sorted (name, value) pairs) |
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
| `rule_11_skin_from_workload` | Rule 11 (3-T05): the skin from `max_temperature_K`, the lightest mass and the timestep (the expected displacement per rebuild interval), at least 2δ plus the MD drift for displacement workloads; no tables. Until 3-T05: `DEFAULT_SKIN_A` (0.3 Å). | `NeighbourConfig.skin` |
| `rule_11_headroom_from_workload` | Rule 11 (3-T05): pair-buffer and edge-bucket headroom from the same expected displacement. Until 3-T05: `"fixed"`. | `NeighbourConfig.headroom` |
| `rule_11_d3_skin_from_workload` | Rule 11 with D3 (3b-T02): the D3 list's skin, max(floor, 2 n d + cutoff f / (1 - f)), from the fastest atom's displacement d per step at `max_temperature_K` and the barostat's thermal strain f over a target interval of n steps (`neighbors.adaptive.rule_11_d3`); floor and n provisional until 3b-T06. Without D3 it stays `"auto"`. | `NeighbourConfig.d3_skin` |
| `fit_when_capturing_on_cuda_else_keep` | `"fit"` when the plan captures CUDA graphs: a new force-graph variant (a larger padded edge size) is captured beside the others while the graph pool's largest bytes per edge times the new size, with a 15 % margin, fits the free device memory; otherwise the captured graphs and their pool are released first and the padded size is kept from then on (CAP-T01, docs/validation/capacity-16gb.md). `"keep"` without CUDA graphs. | `EngineConfig.force_graphs` |
| `recapture_budget_from_workload` | The re-capture budget of a run: `graph_variant_cap`; a run that has re-captured as many graphs as it may hold at once has cycled through its whole working set of edge sizes and stages, so the compiled tier is the better plan (3-T04; 3-T05 may refine it with rule 11). | `EngineConfig.recapture_budget` |
| `force_precision_if_move_energy_gate_passes_else_float64` | MC energies in the force precision if the move-energy gate (SPEC §12.3; always run for MC plans, decision 35) passes, float64 otherwise or when the reference is unavailable (3-T11). Plans without MC: the force precision. | `PrecisionConfig.mc_energy` |
| none (new, SPEC §12.3 "Resources") | `GateConfig.reduced_cell_min_height_cutoffs` | `2.0` | — | gate resources |
| none (new, SPEC §8.7) | `RunnerConfig.python` | `None` | — | subprocess runner (`None`: `sys.executable`) |
| none (new, SPEC §8.7) | `RunnerConfig.environment` | `()` | — | subprocess runner |
| none (new, SPEC §8.7) | `RunnerConfig.inherit_environment` | `True` | — | subprocess runner |
| none (new, SPEC §8.7) | `RunnerConfig.timeout_s` | `None` | — | subprocess runner |
| none (new, SPEC §8.7) | `RunnerConfig.stall_timeout_s` | `1800.0` | — | subprocess runner |
| none (new, OPT-T07) | `RunnerConfig.persistent` | `True` | — | subprocess runner: one worker child per `Simulation` (`False`: a fresh child per `run()`) |

## 11. MC driver of phase 3 → `MCConfig`

New in the library (plan task 3-T01); the research engine has no MC. The move records
are `fm.mc.Swap`, `Transmute` and `Displace` (`docs/design/contracts-3.md`), and the
three MC streams' seeds derive from the simulation's `seed` (`config.mc_stream_seeds`).

| Research engine | FastMACE field | Default | `auto` rule | Kind |
| --- | --- | --- | --- | --- |
| none (new, SPEC §10.4) | `MCConfig.max_attempts_per_block` | `10_000` | — | MC driver (3-T09) |
| none (new, SPEC §10.4) | `MCConfig.max_move_sets` | `8` | — | MC driver (3-T09) |

## 12. Runtime adaptation and calibration of phase 5 → `AdaptationConfig`, `CalibrationConfig`

New in the library (plan task 5-T01, `docs/design/planner-5.md`); the research engine has neither. They
reach the planner through `Simulation(adaptation=, calibration=)` (`PlannerOptions`). `AdaptationConfig`'s
defaults are the owner's decisions 52 and 53 (SPEC §18; plan task 5-T07); `Simulation` refuses the options
they did not choose (`adaptation-unsupported`, `config.ADAPTATION_SUPPORTED`). `CalibrationConfig`'s defaults
are decision 55 (plan task 5-T09, `fastmace.planner.calibrate`): off, and `choose=False` (D5 (b), not chosen)
is refused (`calibration-unsupported`). Its warm-up (50 steps) and pairs (2, A B A B) are the module's
constants.

| Research engine | FastMACE field | Default | `auto` rule | Kind |
| --- | --- | --- | --- | --- |
| none (new, SPEC §9.6) | `AdaptationConfig.adapt` | `"auto"` | `adapt_between_segments` | runtime adaptation (`"off"`: the monitor records only) |
| none (new, SPEC §9.6, decision 52) | `AdaptationConfig.trigger` | `"recapture-time"` | — | runtime adaptation (`config.ADAPTATION_TRIGGERS`; `"none"`: no switch) |
| none (new, SPEC §9.6, decision 52) | `AdaptationConfig.recapture_time_fraction` | `0.2` | — | runtime adaptation (trigger threshold) |
| none (new, SPEC §9.6, D2 (b), not chosen) | `AdaptationConfig.skin_replan` | `False` | — | runtime adaptation (only `False`) |
| none (new, SPEC §9.6, decision 53) | `AdaptationConfig.max_switches` | `1` | — | runtime adaptation (`0` or `1`) |
| none (new, SPEC §9.6, D3 (b), not chosen) | `AdaptationConfig.switch_back_after` | `None` | — | runtime adaptation (only `None`: one way) |
| none (new, SPEC §9.6, D3 (d), not chosen) | `AdaptationConfig.min_points_between` | `None` | — | runtime adaptation (only `None`) |
| none (new, SPEC §9.6, D3 (c), not chosen) | `AdaptationConfig.reproducible` | `False` | — | runtime adaptation (only `False`) |
| none (new, SPEC §9.1, decision 55) | `CalibrationConfig.calibrate` | `False` | — | calibration (`True`: MD runs only; `fm.md(calibrate=)`, `fastmace run --calibrate`) |
| none (new, SPEC §9.1) | `CalibrationConfig.steps` | `50` | — | calibration (measured steps per candidate) |
| none (new, SPEC §9.1, decision 55) | `CalibrationConfig.margin` | `0.05` | — | calibration (faster by more, in every pair) |
| none (new, SPEC §9.1, decision 55) | `CalibrationConfig.choose` | `True` | — | calibration (only `True`; `False`, record only, is D5 (b)) |

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
| `never_automatic` | Off; only `"on"` enables it (`fused_radial_hidden`, `direct_node_feats` and `folded_constants` require `fold_cutoff`, `tf32x3_gemms` requires `block_linear`, `species_residual` requires `species_tables` and `merged_residual`, `gate_epilogue` requires `segment_gemm`, `combo_kernels` requires nothing). | `RewriteConfig.fused_radial_hidden`, `RewriteConfig.direct_node_feats`, `RewriteConfig.folded_constants`, `RewriteConfig.combo_kernels`, `RewriteConfig.tf32x3_gemms`, `RewriteConfig.species_residual`, `RewriteConfig.gate_epilogue` |
| `on_with_fold_cutoff_unless_reproducible` | On when `fold_cutoff` is on (the factored MH-1 class) in a float32 or float64 plan that is not reproducible (owner decisions 62 and 63, 27 October 2026; float64 tables are held to float64 rounding, `docs/validation/f64-T01.md`); off in reproducible plans and wherever the model is not factored. The record's self-check (certificate at most 1) can still turn it off (rule 13). | `RewriteConfig.radial_tables` |
| `on_for_mh1_float32_unless_reproducible` | On for MH-1 (the `mh-1` checkpoint's labels `mh1-r2scan`, `mh1-omat`, `mh1-mp`, by label or by SHA-256 and head) in a float32 plan that is not reproducible, wherever `chunked_conv_backward` is on (owner decision 64, 27 October 2026); off for every other model (neutral on POLAR-1-S, 3.5 % slower on OMOL-0 XL), in float64 and reproducible plans. An explicit `"on"` enables it for any float32 model with `chunked_conv_backward`. | `RewriteConfig.fused_folded_forward` |
| `on_for_mh1_float32_unless_reproducible_on_decision_66_gpus` | On for MH-1 (the `mh-1` checkpoint's labels, as `fused_folded_forward`) in a float32 plan that is not reproducible, with `radial_tables` and `chunked_conv_backward` on, on the GPU models of `planner.fixed.DECISION_66_WPRIME_GPUS` only, matched by exact device name ("NVIDIA A100-SXM4-80GB", "NVIDIA GeForce RTX 5060 Ti", "NVIDIA RTX 4000 Ada Generation": same-process MD step A/Bs in both build orders measured 6–19 % faster, `docs/validation/opt9-T03.md`); off on every other GPU (the off reason names it; the A100-SXM4-40GB and the L4 were not measured), for other models, in float64 and reproducible plans (owner decision 66, 29 October 2026). An explicit `"on"` enables it on any GPU in float32 plans outside reproducible ones. | `RewriteConfig.wprime_backward` |
| `on_by_rule_12_when_the_plan_does_not_fit_without_it` | Off, unless rule 12's estimate does not fit 90 % of the free GPU memory after decision 51's drops: then rule 12 turns it on, if `wprime_backward` is on and it lowers the estimate (`planner.memory.MEMORY_RESCUE`, code `rule-12-memory-rescue`; plans/capacity.md CAP-T07). An explicit value is kept. | `RewriteConfig.recompute_conv_weights` |
| `on_with_streaming_conv_and_fused_folded_backward` | On when `streaming_conv` and `fused_folded_backward` are both on (owner decision 60, 27 October 2026); off otherwise, including every plan without the hybrid backend. The size rule is inside the op, not the planner: weights of at most `UNCHUNKED_L2_MULTIPLE` L2s run the unchunked backward. | `RewriteConfig.chunked_conv_backward` |
| `on_with_fused_folded_backward` | On when `fused_folded_backward` is on; stage 2: off unless the model supports `fold_cutoff` (the MH-1 class; the generic families' fused backward is another record). Bitwise the same products (owner decision, 21 October 2026). | `RewriteConfig.folded_backward_tiles` |
| `on_unless_reproducible_if_model_supports` | On in stage 1; stage 2: off unless the model takes it (`rewrite_support["species_tables"]`: ScaleShiftMACE with a factored first interaction and species-only first inputs, i.e. the MH-1 labels); then off in `reproducible=True` plans (its tables round differently from per-atom products on the GPU). An explicit `"on"` is kept (owner decision, 21 October 2026). On in float64 plans since owner decision 63 (27 October 2026). | `RewriteConfig.species_tables` |
| `on_with_block_linear_unless_reproducible_if_model_supports` | On when `block_linear` is on; stage 2: off unless the model takes it (`rewrite_support["fold_readout_tail"]`: ScaleShiftMACE with a factored last interaction, i.e. the MH-1 labels and OMOL-0); off in `reproducible=True` plans (rounding class); on in float64 plans since owner decision 63. | `RewriteConfig.fold_readout_tail` |
| `on_with_species_tables_and_block_linear_if_model_supports` | On when `block_linear` and `species_tables` are on; stage 2 as `species_tables`; off in `reproducible=True` plans (rounding class; float64 since decision 63 follows `species_tables`), and off when `species_tables` is (an explicit `"on"` is then refused). | `RewriteConfig.fold_product_linear` |
| `on_with_block_linear` | On when `block_linear` is on. `merged_residual` acts only in interactions where `density_first` applies; the rewrite's `requires` handles that, not the rule. | `RewriteConfig.grouped_linear`, `RewriteConfig.merged_residual` |
| `on_with_merged_residual` | On when `merged_residual` is on, so wherever block linear applies with `density_first` (owner decision, 20 October 2026). Stage 2: off unless the model supports `fold_cutoff`; the rewrite's own `applies_to` (MACE's `GatedEquivariantBlock` in a factored interaction) decides where it acts. | `RewriteConfig.gated_rows` |
| `on_with_gated_rows_float32_unless_reproducible` | On when `gated_rows` is on, except in float64 plans and `reproducible=True` plans, which keep the stock products and join (owner decision, 20 October 2026); an explicit `"on"` is kept (in float64 its op runs the PyTorch path). At run time, per force call, the shared shape rule (`fastmace.kernels.triton.tile_rule`) picks the Triton GEMM; where no product qualifies (NaCl-sized cells) the stock products and join run. | `RewriteConfig.segment_gemm` |
| `on_with_block_linear_float32_unless_reproducible_for_generic_mace_family_and_mh1_on_decision_66_gpus` | Stage 1: on when `block_linear` is on. Stage 2: off unless every interaction of the model is a `GENERIC_INTERACTIONS` class and the model is not PolarMACE: the generic MACE family (MP-0, MPA-0, OMAT-0, MatPES); or the model is MH-1 on a GPU of `DECISION_66_GROUPED_LAUNCH_GPUS` (decision 66, 29 October 2026: the RTX 5060 Ti, NaCl 0.971 / 0.975 in both build orders, NbCl5 and NaNbO3 neutral). MACE-OFF (a plain first interaction), MH-1, POLAR-1 and OMOL-0 are other families. Then off in float64 and `reproducible=True` plans. An explicit `"on"` is kept for any family (owner decision 57, 24 October 2026). At run time the shared shape rule (`fastmace.kernels.triton.tile_rule`) groups only small products of equal weight shape (NaCl-sized cells); elsewhere the stock products run. | `RewriteConfig.grouped_launch` |
| `on_for_hybrid_float32_unless_reproducible_for_qualified_polar1_checkpoints` | Stage 1: on with the hybrid backend. Stage 2: off unless the model is PolarMACE and its checkpoint is a qualified POLAR-1 checkpoint (`fastmace.planner.fixed.POLAR_GRAPH_LABELS`: `polar1-s`, `polar1-m` and, after its NbCl5 gate, `polar1-l`; the label-table label of the loaded model, found by label or by SHA-256 and head as rule 15 finds it; a model without that identity stays off). Then off in float64 and `reproducible=True` plans, which keep MACE's code (as decision 57). An explicit `"on"` or `"off"` overrides it for any model (owner decision 58, 24 October 2026). | `RewriteConfig.fx_e3nn_linear`, `RewriteConfig.polar_graph_sums` |
| `on` | On for every engine; acts only where stress is computed (NPT, stress-path relaxations). | `RewriteConfig.single_graph_strain` |
| `on_for_plain_cueq_unless_reproducible` | Plain cuEq backend: on, except in a `reproducible=True` plan, which needs the unfused convolution (§9.3, owner decision 13). Hybrid: not applicable (the research engine records `None`). | `EngineConfig.cueq_conv_fusion` |
| `on_for_torch_2_14_unless_alloc_conf_set` | Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before the first CUDA allocation when torch ≥ 2.14 and the variable is unset; otherwise leave the allocator alone. | `EngineConfig.expandable_segments` |
| `on_for_float64_cuda_with_reduced_rate_fp64_unless_reproducible` | On for float64 plans on CUDA that are not reproducible, on GPUs whose measured float64 multiply-add rate is below `FP64_EMULATION_RATE` (1/8) of their float32 rate (`kernels.triton.fp64_rate`, a Triton probe, cached per device; consumer and workstation parts are about 1/64); off on full-rate parts (A100, H100: about 1/2), in float32 and reproducible plans and on the CPU (owner decision 63, 27 October 2026). An explicit `"on"` with `reproducible=True` is refused. | `EngineConfig.fp64_emulation` |
| `variable_cell_0_1_else_0` | `0.1` for a variable-cell integrator (NPT), `0.0` otherwise. | `PolarConfig.kspace_headroom` |
| `fallback_search_on_failure` | On a failing build, compile, capture, self-check or gate before the first step, the §9.4 search (diagnosis, ddmin within the budget, the terminal sequence), each candidate validated before activation (2-T15); nothing happens on a run that does not fail. `"refuse"` raises as in 1b. | `FallbackConfig.fallback` |
| `adapt_between_segments` | Runtime adaptation (SPEC §9.6, decisions 52 and 53; 5-T07): when the re-capture budget is exceeded and re-capturing takes more than `recapture_time_fraction` of the wall time since the last switch, one tier switch graph → compiled at the next committed step outside an MC block (a canonical point, forced there), as a new plan revision, validated before activation. Never under `reproducible=True`, with an explicit `mode=`, or for a relaxation. A run whose trigger does not fire is unchanged (conductor's choice 7). | `AdaptationConfig.adapt` |
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
  None comes from the research engine. `RewriteConfig.selfcheck_cache` was removed on 28 October 2026
  (checkpoint 2 decided no self-check cache; nothing read it).
- Optimisation round 1 (plan task OPT-T06) added `EngineConfig.model_template_cache`,
  71 in all. It does not come from the research engine, which loaded once per process.
- Optimisation round 2 (plan task OPT2-T04) added `EngineConfig.overlap_rebuild` and
  `EngineConfig.step_graph`, 73 in all (both opt-in).
- Optimisation round 3 (plan task OPT3-T03) added `EngineConfig.fused_integrator`, 74 in all; on by default
  since the owner's decision of 21 October 2026.
- Phase 3 (plan task 3-T01) added four fields to existing records (`EngineConfig.recapture_budget` and
  `graph_variant_cap`, `PrecisionConfig.mc_energy`, `NeighbourConfig.headroom`) and let
  `NeighbourConfig.skin` admit `"auto"`, plus the record `MCConfig` with 2 fields (section 11), 80 in
  all. Four more fields admit `"auto"` (`skin`, `headroom`, `recapture_budget`, `mc_energy`). None
  comes from the research engine.
- Phase 3b (plan task 3b-T02) added `NeighbourConfig.d3_skin` (admits `"auto"`), 81 in all. It does not
  come from the research engine, whose D3 ran through torch-dftd on the ASE path.
- Phase 5 (plan task 5-T01) added two records with 12 fields (section 12: `AdaptationConfig` 8,
  `CalibrationConfig` 4). One admits `"auto"` (`AdaptationConfig.adapt`). None comes from the research
  engine.

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
