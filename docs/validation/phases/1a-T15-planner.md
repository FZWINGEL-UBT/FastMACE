# 1a-T15: fixed-plan planner

Plan task 1a-T15, `fastmace/planner/fixed.py`. Baseline: ResearchModule tag
`research-engine-baseline-20261017` (commit d2faa9fd), `workflow.parse_args`,
`adapt_options_to_model`, `GPUEngine.__init__`, `baoab_settings`.

## Plan equality (CPU)

`tests/cpu/test_planner.py`, with `FASTMACE_BASELINE=~/fastmace-baseline/20261017`:

- one stand-in per label of the 19, built from trait facts read from the released
  checkpoints (`model_traits`; `tests/gpu/test_planner_gpu.py` re-reads all 19 from
  `~/models` and finds them equal, including `rewrite_support`);
- for every label, the plan equals the baseline's own `parse_args` +
  `adapt_options_to_model` for 31 `main.py` argument sets (defaults, `--backend cueq`, NPT,
  single options off or on, `generic_only_argv`), refusals included, and
  `Plan.auto_options_off` equals the baseline's returned `{option: reason}`;
- written-out expectations for `--backend auto` and `--backend cueq` (without the baseline).

Two deliberate differences, both cases where the research engine refuses later, at engine
construction: compiled plain-cuEq PolarMACE (zero forces; `GPUEngine.__init__`) is refused at
plan time, and plain-cuEq reassociation (its probe refuses it; conductor decision after T11)
is switched off (`auto`) or refused (`on`) at plan time.

## Plans of the 19 labels

`--backend auto` (hybrid FP32) and `--backend cueq` (float64), NVT. Engine options in every
plan: `gpu_neighbour_list`, `pipelined`, `async_output`, `single_graph_strain` on,
`expandable_segments` on (torch 2.14), `cueq_conv_fusion` on for cueq and not applicable for
the hybrid.

| Labels | Backend | On | Off by stage 2 (reason) |
| --- | --- | --- | --- |
| MH-1 ×3, POLAR-1 S/M/L, OMOL-0 XL | hybrid | factor_species, fold_cutoff, density_first, reassociate_first, joint_radial_input, fused_folded_backward, segmented_first_sum, streaming_conv, block_linear, grouped_linear, merged_residual, sort_receivers | species_skip (no MACE-MP-0 or density-family interaction) |
| MP-0 ×2, MP-0b3, MPA-0, OMAT-0, MatPES ×2, MACE-OFF ×5 | hybrid | reassociate_first, fused_folded_backward, segmented_first_sum, streaming_conv, species_skip, block_linear, grouped_linear, merged_residual, sort_receivers | factor_species, fold_cutoff, density_first, joint_radial_input (interactions … are not MH-1's class) |
| MH-1 ×3, OMOL-0 XL | cueq | (engine options only) | species_skip |
| POLAR-1 S/M/L | cueq | refused compiled (zero forces); with `compile_mode="none"` as MH-1 | species_skip |
| MP-0 ×2, density ×5, MACE-OFF ×5 | cueq | species_skip | — |

At build, the record level decides what acts on the model: `merged_residual` without
`density_first`, `segmented_first_sum` beside `generic_reassociate`, and
`generic_reassociate` on MACE-OFF (non-generic first block) stay off with their reasons,
as the research engine's switches are no-ops there.

## GPU: plan → build → run(500)

Local PC, NVIDIA GeForce RTX 5060 Ti (sm_120, 16 GB), WSL2, 18 October 2026, torch
2.14.1+cu130, MACE 0.3.16. Environments `mace_cueq_latest` (pytest, plain cuEq) and
`mace_cueq_oeq` (`FASTMACE_HYBRID_PYTHON`, hybrid). MH-1 `matpes_r2scan` (`~/models`), NaCl
3×3×3 (216 atoms), NVT 300 K, 0.5 fs, seed 20261018, positions every 50 steps. The planned
engine against 1a-T13's hand-built engine with the same records and options, both
pipelined, each in a fresh process. Results: `results/1a-T15-planner-20261018/`.

| Backend | Records | Max \|Δx\| over 500 steps | Scatter (two hand-built runs) | ms/step planned / hand-built |
| --- | --- | --- | --- | --- |
| cueq float64 | single_graph_strain | 3.9e-13 Å | (limit 1e-12 Å, T13) | 102.17 / 102.17 |
| hybrid float32 | 12 (every MH-1 rewrite, linear family, streaming, strain) | 8.8e-8 Å | 8.6e-8 Å | 3.48 / 3.54, 3.62 |

Edge sizes (7168) and neighbour rebuilds (18) are identical. The allocator was
`expandable_segments:True` in both planned runs.

## Suites

At FastMACE commit e48cdb9 (branch `fastmace/1a-T15`), baseline exports `20261017`
(CPU) and `1a-T18` (GPU):

- CPU, `fastmace_cpu`: 983 passed, 28 skipped; `mace_cueq_latest`: 1009 passed, 2 skipped;
  `ruff check fastmace tests tools` clean.
- The whole `tests/gpu` in one process (`mace_cueq_latest`, hybrid children in
  `mace_cueq_oeq`), RTX 5060 Ti: 219 passed, 2 skipped (37.9 min).
