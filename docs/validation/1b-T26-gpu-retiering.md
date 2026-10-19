# 1b-T26 Re-tiering the GPU suite (19 October 2026)

Since 1b-T24, merged tasks had added GPU tests that run released checkpoints without the `slow`
mark, so the fast tier (`pytest tests/gpu -m "not slow"`) had grown from 3.4 to 24.3 minutes.
This task marked those tests `slow`. It added tiny-model versions where they were cheap, and
lists in [development.md](../development.md) ("Test tiers") the behaviours now tested only in the
full tier. It changed no library code and no assertions.

Machine: local PC, WSL2, NVIDIA GeForce RTX 5060 Ti; pytest in `mace_cueq_latest`
(torch 2.14.1+cu130), hybrid children in `mace_cueq_oeq`; `TORCHINDUCTOR_COMPILE_THREADS=4`,
`-n 0`, warm shared caches. FastMACE: `fastmace/1b-T26` on main 410f4d3.

## Fast tier

| | Tests | Wall time |
| --- | --- | ---: |
| before (main b48f4db, without 1b-T25) | 195 passed, 3 skipped, 52 deselected | 24.3 min |
| after (with 1b-T25 and 1b-T26) | 186 passed, 3 skipped, 65 deselected | **8.0 min** |

After the change, the slowest tests are the tiny-model tests that each start fresh processes:
the 1b-T25 edge bucket (55 s, cold Inductor cache by design), the broken captured graph (39 s),
`two_force_fields[float32-default]` (31 s), `process_constraints` (30 s), the tiny registry
(29 s) and planned-equals-hand-built (29 s). Together they take 3.6 of the 8.0 minutes. A fresh
process costs about 25 s even with warm caches (1b-T25), and that cost sets the floor.

## Marked slow

| Test | Model | Fast-tier time before |
| --- | --- | ---: |
| `test_load_hybrid_gpu::test_mace_imported_first_gives_plain_cueq_with_the_reason` | MH-1 | 310 s |
| `test_canonical_restart_gpu::test_mh1_plain_cueq_float64_resumes_within_1e_12` | MH-1 | 203 s |
| `test_selfcheck_gpu::test_gate_passes_in_graph_mode[mh1-r2scan-nvt]` | MH-1 | 85 s |
| `test_simulation_gpu::test_simulation_reaches_the_main_py_trajectory[hybrid]` | MH-1 | 76 s |
| `test_simulation_gpu::test_simulation_reaches_the_main_py_trajectory[cueq]` | MH-1 | 63 s |
| `test_selfcheck_gpu::test_gate_passes_in_graph_mode[mh1-r2scan-npt-full]` | MH-1 | 57 s |
| `test_selfcheck_gpu::test_a_broken_captured_graph_fails_while_eager_passes` | MH-1 | 53 s |
| `test_ase_calculator_gpu::test_lbfgs_with_fm_calc_reaches_maces_minimum[hybrid]` | MH-1 | 50 s |
| `test_ase_calculator_gpu::test_lbfgs_with_fm_calc_reaches_maces_minimum[pytest]` | MH-1 | 48 s |
| `test_registry_gpu::test_mh1_npt_baoab_full_cell` | MH-1 | 46 s |
| `test_registry_gpu::test_mh1_hybrid_pipelined_nvt` | MH-1 | 46 s |
| `test_registry_gpu::test_mh1_plain_cueq_float64` | MH-1 | 44 s |
| `test_registry_gpu::test_polar_1_s` | POLAR-1-S | 40 s |

`test_selfcheck_gpu`'s parametrisation used to mark every case `slow` except MH-1. It now marks
every case. Tests over 30 s that already use tiny models stay in the fast tier:
`two_force_fields[float32-default]`, `process_constraints` and 1b-T25's two tests.

## Tiny-model versions

Each tiny version runs the MH-1 test's child script, with a tiny MH-1-like model
(`_support.tiny_models`, Na/Cl) loaded from a path with `trusted=True`. Times are warm, from one run:

| Tiny version of | Time | Result | Kept |
| --- | ---: | --- | --- |
| registry, plain cuEq float64 | 29 s | passed | **yes** |
| broken captured graph fails, eager passes | 36 s | passed (eager 1.0e-6, captured 2.7e-5 eV/Å) | **yes** |
| registry, hybrid full-cell NPT | 44 s | passed | no |
| gate passes, full-cell NPT | 60 s | passed | no |
| `fm.Simulation` reaches `FixedPlanner` (cuEq) | 73 s | passed | no |
| registry, POLAR (hybrid) | 77 s | passed | no |
| gate passes, NVT with fresh references | 87 s | passed | no |
| MACE imported first gives plain cuEq | 316 s | passed | no |
| canonical restart, plain cuEq float64 | 84 s | **failed**: 2.5e-11 Å against 1e-12 | no |
| LBFGS with `fm.calc` | 75 s | **failed**: LBFGS does not converge on the random model | no |

The tiny versions cost almost as much as the MH-1 tests (887 s for ten, against about 1,120 s for
the thirteen MH-1 tests). Fresh processes, compiles and the tests' own child processes dominate:
the gate's reference process, three engines in the Simulation test. Only the two cheapest were
kept. The two failures come from the random model, not from the library: its energy surface has no
minimum within LBFGS's 300 steps at `fmax` 2e-3, and it amplifies cuEq's run-time rounding
differences (about 1e-13 Å) over 500 steps at 600 K. The behaviours now tested only in the full tier
are listed in [development.md](../development.md).

## For the conductor

- "MACE imported first" takes about 310 s with either model, tiny or MH-1, while the hybrid case
  (`test_fm_load_then_minimal_planner_gives_the_hybrid`) takes 10 s. The time goes into the plain-cuEq
  path after MACE was imported, in the hybrid interpreter. This was not investigated (no library
  changes in this task). It may be worth a look, because a user who imports MACE first pays it too.
- Getting the fast tier back to about 5 minutes would mean dropping tiny-model tests that are the
  only fast coverage of their behaviour, or sharing one child process between several checks.
