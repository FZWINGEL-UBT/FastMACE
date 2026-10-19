# Test-suite speed (plan task 1b-T24)

Measured on 19 October 2026, on the local PC only. Raw results are in
[`results/1b-T24-test-speed-20261019/`](../../results/1b-T24-test-speed-20261019/).

## Setup

| Item | Value |
| --- | --- |
| Machine | local PC, WSL2 Ubuntu 24.04, 15 GB RAM |
| GPU | NVIDIA GeForce RTX 5060 Ti (16 GB), driver 610.88. Windows desktop apps held 7–9 GB of it throughout |
| Environments | `mace_cueq_latest` (pytest process: torch 2.14.1+cu130, MACE 0.3.16, cuEquivariance 0.12.0); hybrid children in `mace_cueq_oeq` (`FASTMACE_HYBRID_PYTHON`, OpenEquivariance 0.7.0); CPU suites in `fastmace_cpu` and `mace_cueq_latest` |
| Backends, precision | all the suite's: plain cuEq float64, hybrid float32, eager and compiled |
| Models | MH-1 (`mh1-r2scan`), MP-0 medium, OFF23 small, POLAR-1-S and the other labels from `FASTMACE_MODELS=~/models`; tiny test models |
| Research engine | `FASTMACE_BASELINE=~/fastmace-baseline/1a-T18`, the export of ResearchModule tag `research-engine-baseline-20261017` (d2faa9fd) |
| FastMACE | before: `a5fd779` (`main`); after: `d8b43b0` (branch `fastmace/1b-T24`). Each was run from a `git archive` export, not from the worktree |
| GPU runs | one process, `-p no:xdist`, `TORCHINDUCTOR_COMPILE_THREADS=4` (see "WSL crash") |
| CPU runs | `-n 4`, `OMP_NUM_THREADS=2`, `CUDA_VISIBLE_DEVICES=''`, `TORCHINDUCTOR_COMPILE_THREADS=4` |

**Load during the runs.** The conductor kept other sessions to single test
files at `-n 0` (1b-T03 and 1b-T05 ran some). Another pytest process was
running in 39 of the 147 load samples taken during the GPU after-runs.
The lowest available memory was 5.9 GB (cold full suite) and 6.4 GB in
the baseline run. Durations are wall times from pytest's reports. The
cross-run noise is a few per cent.

**Cold and warm.** "Before" used torch's default compiler cache
(`/tmp/torchinductor_$USER`), warm from earlier runs. "After" uses the
FastMACE cache root (below). It was emptied before the first after-run,
together with the OEq extension cache, so "cold" is the first run after
the change, and "warm" every later one. Warm is what workers see.

## Results

| Suite | Before | After, cold | After, warm |
| --- | ---: | ---: | ---: |
| GPU full (`tests/gpu`, 223 tests) | 33.7 min | 34.8 min | **28.7 min** |
| GPU fast tier (`-m "not slow"`, 178 tests) | (4.3 min)¹ | 5.7 min | **3.4 min** |
| CPU, `fastmace_cpu` (1075 tests after, 1067 before) | 1.7 min | 2.5 min | (2.5 min)² |
| CPU, `mace_cueq_latest` | 2.6 min | 3.0 min | (3.0 min)² |

All runs passed: GPU 221 passed and 2 skipped (full) or 176 passed and
2 skipped (fast). The two skips are `test_kernels_oeq_streaming_cuda.py`,
which runs in the pytest process (`mace_cueq_latest`, no OEq); they were
skipped before as well.

1. The fast tier did not exist before; 4.3 min is the sum of the baseline
   durations of its 178 tests.
2. Both CPU after-runs compiled cold. The CPU compile entries of the
   first after-run were deleted with the cache before the GPU after-runs,
   and the GPU runs add only GPU entries. Almost the whole difference is
   one `torch.compile` test,
   `test_forces.py::test_compiled_forces_match_uncompiled`: under `fastmace_cpu` 71 s before and 115–117 s
   after (cold); under `mace_cueq_latest` about 120 s both times. An A/B of
   that test alone (`-n 0`, 2 threads, `fastmace_cpu`) shows no
   difference once warm. Before, with torch's default cache: 20.8 s and
   17.2 s. After, with the shared cache: 17.9 s and 17.3 s. After, with
   the caller's directory: 125.5 s cold, then 16.9 s. A local GPU check by
   another session ran at the same time. The 8 new CPU tests (persistent
   extension) add about 4 s.

### Where the time went

- **OEq host extension: 17.8 s → 3.0–3.3 s per hybrid process** (probe
  below). That is about 15 s for each hybrid child process in the suite
  (the full suite's test durations drop by 15–17 s for every hybrid test). Examples: `test_mh1_hybrid_float32_in_a_fresh_process`
  28.9 → 12.7 s, `test_two_force_fields…[float32-none]` 29.3 → 13.0 s,
  `test_polar1_s_hybrid_float32_in_a_fresh_process` 28.7 → 13.7 s.
- **The shared compiler cache.** Cold, the compiled MH-1 tests are 20–65 s
  slower than the baseline's warm `/tmp` cache (for example
  `test_mh1_plain_cueq_float64_against_the_research_engine` 337 → 402 s).
  Warm, they are within a few seconds of the baseline (336 s; the
  largest difference is +5 s, `test_mh1_agrees_with_maceforces[cueq-float64-compiled-production]`).
- **The tiers.** The full suite's time is dominated by 9 tests over 60 s,
  all real-checkpoint or research-engine tests. Those, the 500-step
  trajectories and the cold-cache regression tests are `slow` now.

## OEq host extension: build before and after

Probe: `prepare_in_process` and `import openequivariance` in fresh
`mace_cueq_oeq` processes, each with its own fresh `FASTMACE_CACHE`
(`results/…/oeq-probe.jsonl`).

| Case | Prepare | OEq import | Total | Extension | Rebuilt in the process |
| --- | ---: | ---: | ---: | --- | --- |
| before, process 1 | 0.1 s | 17.7 s | 17.8 s | — | `.o`, `.so` |
| before, process 2 | 0.05 s | 17.7 s | 17.8 s | — | `.o`, `.so` |
| after, process 1 (first build) | 19.1 s | 3.5 s | 22.5 s | built | nothing |
| after, process 2 | 0.06 s | 3.2 s | 3.3 s | reused | nothing |
| after, process 3 | 0.09 s | 2.9 s | 3.0 s | reused | nothing |
| after, two processes at once, cold | 17.8 s | 2.5 s | 20.3 s | built / raced | nothing |

The first process with a new build identity pays the build (in a child
interpreter) once, plus about 5 s for that child's torch import. Every
later process copies the 1.8 MB build and ninja finds it up to date. Two
concurrent first builds both succeed: one renames its build into place
("built"), and the other finds it there, discards its own build and uses
the winner's ("raced"). A single published directory remains.

## WSL crash

The first two baseline attempts ended when the WSL VM restarted (01:19,
during `test_rewrites_mh1_gpu.py`; the 01:31 restart was the conductor's
`wsl --shutdown` to recover). The likely cause is host RAM. Inductor
started 24 compile workers per process while two other sessions ran CPU
suites, and WSL has 15 GB. From then on every run used
`TORCHINDUCTOR_COMPILE_THREADS=4` and logged `free -m` every 5 s. No
crash followed, and available memory stayed above 5.9 GB. The setting
changes compile time, not step time. Both before and after ran with it.

## Ten slowest tests per run

The CPU durations are per test summed over its phases. With `-n 4`, they add
up to more than the wall time.

#### GPU full, before (a5fd779): 33.7 min wall, 223 tests (221 passed, 2 skipped)

| s | test |
| ---: | --- |
| 337.1 | `test_engine_gpu.py::test_mh1_plain_cueq_float64_against_the_research_engine` |
| 224.3 | `test_engine_gpu.py::test_mh1_plain_cueq_float64_npt_against_the_research_engine` |
| 169.5 | `test_engine_gpu.py::test_mh1_hybrid_float32_against_the_research_engine` |
| 162.5 | `test_planner_gpu.py::test_mh1_plain_cueq_float64_planned_equals_hand_built` |
| 154.1 | `test_planner_gpu.py::test_mh1_hybrid_float32_planned_equals_hand_built` |
| 98.3 | `test_engine_gpu.py::test_no_compile_at_a_later_edge_bucket_plain_cueq` |
| 94.5 | `test_forces_gpu.py::test_mh1_agrees_with_maceforces[hybrid-float32-compiled-production]` |
| 85.2 | `test_engine_gpu.py::test_no_compile_at_a_later_edge_bucket_hybrid` |
| 67.0 | `test_forces_gpu.py::test_mh1_agrees_with_maceforces[cueq-float64-compiled-production]` |
| 37.4 | `test_forces_gpu.py::test_two_force_fields_from_one_model_step_independently[float32-default]` |

#### GPU full, after, cold caches: 34.8 min wall, 223 tests (221 passed, 2 skipped)

| s | test |
| ---: | --- |
| 402.3 | `test_engine_gpu.py::test_mh1_plain_cueq_float64_against_the_research_engine` |
| 282.0 | `test_engine_gpu.py::test_mh1_plain_cueq_float64_npt_against_the_research_engine` |
| 209.6 | `test_planner_gpu.py::test_mh1_plain_cueq_float64_planned_equals_hand_built` |
| 190.1 | `test_engine_gpu.py::test_mh1_hybrid_float32_against_the_research_engine` |
| 128.3 | `test_forces_gpu.py::test_mh1_agrees_with_maceforces[hybrid-float32-compiled-production]` |
| 123.4 | `test_forces_gpu.py::test_mh1_agrees_with_maceforces[cueq-float64-compiled-production]` |
| 114.2 | `test_planner_gpu.py::test_mh1_hybrid_float32_planned_equals_hand_built` |
| 99.3 | `test_engine_gpu.py::test_no_compile_at_a_later_edge_bucket_plain_cueq` |
| 88.2 | `test_engine_gpu.py::test_no_compile_at_a_later_edge_bucket_hybrid` |
| 49.3 | `test_forces_gpu.py::test_two_force_fields_from_one_model_step_independently[float64-default]` |

#### GPU full, after, warm caches: 28.7 min wall, 223 tests (221 passed, 2 skipped)

| s | test |
| ---: | --- |
| 335.7 | `test_engine_gpu.py::test_mh1_plain_cueq_float64_against_the_research_engine` |
| 223.9 | `test_engine_gpu.py::test_mh1_plain_cueq_float64_npt_against_the_research_engine` |
| 165.9 | `test_planner_gpu.py::test_mh1_plain_cueq_float64_planned_equals_hand_built` |
| 138.2 | `test_engine_gpu.py::test_mh1_hybrid_float32_against_the_research_engine` |
| 107.3 | `test_planner_gpu.py::test_mh1_hybrid_float32_planned_equals_hand_built` |
| 98.9 | `test_engine_gpu.py::test_no_compile_at_a_later_edge_bucket_plain_cueq` |
| 90.7 | `test_engine_gpu.py::test_no_compile_at_a_later_edge_bucket_hybrid` |
| 79.5 | `test_forces_gpu.py::test_mh1_agrees_with_maceforces[hybrid-float32-compiled-production]` |
| 72.1 | `test_forces_gpu.py::test_mh1_agrees_with_maceforces[cueq-float64-compiled-production]` |
| 28.2 | `test_loading_gpu.py::test_hybrid_conversion_keeps_the_last_layer_irreps_of_a_tiny_polar_model` |

#### GPU fast tier, after, cold caches: 5.7 min wall, 178 tests (176 passed, 2 skipped)

| s | test |
| ---: | --- |
| 79.3 | `test_engine_gpu.py::test_process_constraints_refuse_a_mixed_process` |
| 48.3 | `test_forces_gpu.py::test_two_force_fields_from_one_model_step_independently[float32-default]` |
| 28.5 | `test_loading_gpu.py::test_hybrid_conversion_keeps_the_last_layer_irreps_of_a_tiny_polar_model` |
| 20.6 | `test_forces_gpu.py::test_rewritten_force_field_agrees_and_replays_in_a_graph[default-dtype1]` |
| 19.2 | `test_forces_gpu.py::test_rewritten_force_field_agrees_and_replays_in_a_graph[default-dtype0]` |
| 13.8 | `test_forces_gpu.py::test_two_force_fields_from_one_model_step_independently[float32-none]` |
| 11.0 | `test_rewrites_mh1_gpu.py::test_streaming_attaches_on_a_tiny_hybrid_model[float32]` |
| 6.3 | `test_engine_gpu.py::test_replay_equals_eager_bitwise[False]` |
| 5.8 | `test_forces_gpu.py::test_rewritten_force_field_agrees_and_replays_in_a_graph[none-dtype1]` |
| 5.3 | `test_forces_gpu.py::test_private_copy_keeps_the_fused_cueq_adapter[dtype0]` |

#### GPU fast tier, after, warm caches: 3.4 min wall, 178 tests (176 passed, 2 skipped)

| s | test |
| ---: | --- |
| 31.0 | `test_loading_gpu.py::test_hybrid_conversion_keeps_the_last_layer_irreps_of_a_tiny_polar_model` |
| 21.9 | `test_forces_gpu.py::test_two_force_fields_from_one_model_step_independently[float32-default]` |
| 20.0 | `test_engine_gpu.py::test_process_constraints_refuse_a_mixed_process` |
| 12.5 | `test_forces_gpu.py::test_two_force_fields_from_one_model_step_independently[float32-none]` |
| 10.1 | `test_rewrites_mh1_gpu.py::test_streaming_attaches_on_a_tiny_hybrid_model[float32]` |
| 9.6 | `test_forces_gpu.py::test_rewritten_force_field_agrees_and_replays_in_a_graph[default-dtype1]` |
| 6.7 | `test_forces_gpu.py::test_rewritten_force_field_agrees_and_replays_in_a_graph[default-dtype0]` |
| 5.3 | `test_forces_gpu.py::test_private_copy_keeps_the_fused_cueq_adapter[dtype0]` |
| 5.0 | `test_forces_gpu.py::test_private_copy_keeps_the_fused_cueq_adapter[dtype1]` |
| 3.4 | `test_rewrites_mh1_gpu.py::test_package_on_cueq_gpu_models[mh1-cuda-dtype0]` |

#### CPU fastmace_cpu, before: 1.7 min wall, 1067 tests (867 passed, 200 skipped)

| s | test |
| ---: | --- |
| 71.3 | `test_forces.py::test_compiled_forces_match_uncompiled` |
| 4.7 | `test_integrators.py::test_baoab_trajectory_matches_ase[full-7]` |
| 4.7 | `test_integrators.py::test_baoab_trajectory_matches_ase[full-256]` |
| 4.6 | `test_integrators.py::test_baoab_trajectory_matches_ase[hydrostatic-256]` |
| 4.6 | `test_integrators.py::test_baoab_trajectory_matches_ase[hydrostatic-7]` |
| 2.2 | `test_support.py::test_families_use_distinct_interaction_classes` |
| 2.1 | `test_planner.py::test_build_on_the_cpu_equals_a_hand_built_engine[density-False]` |
| 2.0 | `test_rewrites_linear.py::test_merged_residual_applies_to_mh1_interactions` |
| 1.8 | `test_engine.py::test_padded_buckets_match_ase` |
| 1.8 | `test_rewrites_linear.py::test_rewritten_tiny_models_agree_with_the_originals[mh1]` |

#### CPU fastmace_cpu, after, first run (cold cache): 2.5 min wall, 1075 tests (875 passed, 200 skipped)

| s | test |
| ---: | --- |
| 114.8 | `test_forces.py::test_compiled_forces_match_uncompiled` |
| 4.5 | `test_integrators.py::test_baoab_trajectory_matches_ase[full-7]` |
| 4.4 | `test_integrators.py::test_baoab_trajectory_matches_ase[full-256]` |
| 4.4 | `test_integrators.py::test_baoab_trajectory_matches_ase[hydrostatic-7]` |
| 4.3 | `test_integrators.py::test_baoab_trajectory_matches_ase[hydrostatic-256]` |
| 2.4 | `test_support.py::test_families_use_distinct_interaction_classes` |
| 2.3 | `test_capabilities.py::test_support_and_padding_match_the_model[RealAgnosticResidualInteractionBlock-True-False-False-True]` |
| 2.1 | `test_planner.py::test_build_on_the_cpu_equals_a_hand_built_engine[density-False]` |
| 1.8 | `test_rewrites_linear.py::test_merged_residual_applies_to_mh1_interactions` |
| 1.8 | `test_engine.py::test_padded_buckets_match_ase` |

#### CPU fastmace_cpu, after, second run (CPU compile cache cold again): 2.5 min wall, 1075 tests (875 passed, 200 skipped)

| s | test |
| ---: | --- |
| 117.4 | `test_forces.py::test_compiled_forces_match_uncompiled` |
| 4.6 | `test_integrators.py::test_baoab_trajectory_matches_ase[full-7]` |
| 4.5 | `test_integrators.py::test_baoab_trajectory_matches_ase[full-256]` |
| 4.4 | `test_integrators.py::test_baoab_trajectory_matches_ase[hydrostatic-7]` |
| 4.3 | `test_integrators.py::test_baoab_trajectory_matches_ase[hydrostatic-256]` |
| 2.4 | `test_support.py::test_families_use_distinct_interaction_classes` |
| 2.0 | `test_planner.py::test_build_on_the_cpu_equals_a_hand_built_engine[density-False]` |
| 2.0 | `test_capabilities.py::test_support_and_padding_match_the_model[RealAgnosticResidualInteractionBlock-True-False-False-True]` |
| 2.0 | `test_capabilities.py::test_e0_of_a_tiny_model_with_an_all_electron_table` |
| 1.9 | `test_rewrites_linear.py::test_merged_residual_applies_to_mh1_interactions` |

#### CPU mace_cueq_latest, before: 2.6 min wall, 1067 tests (893 passed, 174 skipped)

| s | test |
| ---: | --- |
| 120.4 | `test_forces.py::test_compiled_forces_match_uncompiled` |
| 4.6 | `test_integrators.py::test_baoab_trajectory_matches_ase[full-7]` |
| 4.5 | `test_integrators.py::test_baoab_trajectory_matches_ase[full-256]` |
| 4.4 | `test_integrators.py::test_baoab_trajectory_matches_ase[hydrostatic-7]` |
| 4.3 | `test_integrators.py::test_baoab_trajectory_matches_ase[hydrostatic-256]` |
| 3.8 | `test_forces.py::test_polar_charge_and_spin_are_validated` |
| 3.7 | `test_support.py::test_families_use_distinct_interaction_classes` |
| 3.7 | `test_forces.py::test_polar_force_field_enables_its_own_copy` |
| 2.5 | `test_support.py::test_tiny_model_is_deterministic_in_float64[polar]` |
| 2.1 | `test_planner.py::test_build_on_the_cpu_equals_a_hand_built_engine[density-False]` |

#### CPU mace_cueq_latest, after, first run (cold cache): 3.0 min wall, 1075 tests (901 passed, 174 skipped)

| s | test |
| ---: | --- |
| 125.9 | `test_forces.py::test_compiled_forces_match_uncompiled` |
| 5.4 | `test_integrators.py::test_baoab_trajectory_matches_ase[full-7]` |
| 4.8 | `test_integrators.py::test_baoab_trajectory_matches_ase[hydrostatic-256]` |
| 4.7 | `test_integrators.py::test_baoab_trajectory_matches_ase[full-256]` |
| 4.3 | `test_integrators.py::test_baoab_trajectory_matches_ase[hydrostatic-7]` |
| 4.1 | `test_forces.py::test_polar_force_field_enables_its_own_copy` |
| 3.8 | `test_forces.py::test_polar_charge_and_spin_are_validated` |
| 3.4 | `test_support.py::test_families_use_distinct_interaction_classes` |
| 2.4 | `test_support.py::test_tiny_model_is_deterministic_in_float64[polar]` |
| 2.2 | `test_import_time.py::test_default_makes_pytorch_locks_soft` |

#### CPU mace_cueq_latest, after, second run (CPU compile cache cold again): 3.0 min wall, 1075 tests (901 passed, 174 skipped)

| s | test |
| ---: | --- |
| 122.5 | `test_forces.py::test_compiled_forces_match_uncompiled` |
| 4.6 | `test_integrators.py::test_baoab_trajectory_matches_ase[full-256]` |
| 4.6 | `test_integrators.py::test_baoab_trajectory_matches_ase[full-7]` |
| 4.4 | `test_integrators.py::test_baoab_trajectory_matches_ase[hydrostatic-256]` |
| 4.4 | `test_integrators.py::test_baoab_trajectory_matches_ase[hydrostatic-7]` |
| 3.9 | `test_forces.py::test_polar_charge_and_spin_are_validated` |
| 3.9 | `test_forces.py::test_polar_force_field_enables_its_own_copy` |
| 3.4 | `test_support.py::test_families_use_distinct_interaction_classes` |
| 2.5 | `test_support.py::test_tiny_model_is_deterministic_in_float64[polar]` |
| 2.1 | `test_planner.py::test_build_on_the_cpu_equals_a_hand_built_engine[density-False]` |

## Open

- **No fast GPU test** covers two behaviours: no compile at a later edge
  bucket, and a planned MH-1 engine equal to a hand-built one. Both need a
  real checkpoint today. Tiny-model versions would be new tests, outside
  this task (proposed as a follow-up).
- The research engine's hybrid children still rebuild their OEq extension
  in every process (`oeq_no_nvcc.sh`, fresh `TORCH_EXTENSIONS_DIR`), about
  15 s each, slow tier only. `OEQ_EXTENSION_SEED` could reuse a build made
  for the same launcher cache, but only with a test-side helper that
  seeds it.
- `test_kernels_oeq_streaming_cuda.py` skips in the one-process run,
  because the pytest process has no OEq (as before).
