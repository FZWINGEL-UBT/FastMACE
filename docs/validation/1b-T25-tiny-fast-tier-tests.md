# 1b-T25 Tiny-model fast-tier GPU tests (19 October 2026)

Two GPU behaviours were tested only in the full tier (1b-T24), by MH-1 tests marked `slow`.
`tests/gpu/test_tiny_engine_gpu.py` adds tiny-model versions to the fast tier
(`pytest tests/gpu -m "not slow"`). The MH-1 tests are unchanged.

| Fast-tier test | MH-1 test (full tier) | Warm wall time |
| --- | --- | ---: |
| `test_no_compile_at_a_later_edge_bucket_tiny_plain_cueq` | `test_engine_gpu.py::test_no_compile_at_a_later_edge_bucket_plain_cueq` | 58–61 s |
| `test_tiny_plain_cueq_float64_planned_equals_hand_built` | `test_planner_gpu.py::test_mh1_plain_cueq_float64_planned_equals_hand_built` | 30 s |

Machine: local PC, WSL2, NVIDIA GeForce RTX 5060 Ti; `mace_cueq_latest` (torch 2.14.1+cu130),
plain cuEq, float64; tiny MH-1-like model from `tests/_support/tiny_models.py` (`FAMILIES["mh1"]`,
Na and Cl). FastMACE: branch `fastmace/1b-T25` on main 6c40d18. No baseline engine is involved.

## What the tests check

- **Edge bucket.** NaCl 2x2x2 at 900 K. The tiny model's cutoff (4 Å) lies just above the second
  shell (3.99 Å), so the edge count crosses buckets: 1152 initial edges, bucket 256. Warm-up saw
  1024 and 1280. At step 93 the run reached a new padded size (768), and dynamo's counters were the
  same as after warm-up (2 unique graphs in plain cuEq). This used a fresh Inductor cache with a
  foreign capture-safe `mask_head` installed, as the MH-1 test does.
- **Planned equals hand-built.** `FixedPlanner(backend="cueq")` builds the engine with main.py's
  defaults: records `["single_graph_strain"]`, options `single_graph_strain, gpu_neighbour_list`,
  pipelined, `expandable_segments:True`. A second engine is built by hand with 1a-T13's settings.
  After 200 steps the positions differ by 2.2e-15 Å (the limit is 1e-12 Å). Both engines used the
  edge sizes {768, 1024, 1280} and made 7 neighbour rebuilds.

## Why not under 20 s

Profile of one fresh process (tiny model, plain cuEq):

| Phase | Warm caches | Fresh Inductor cache |
| --- | ---: | ---: |
| hardware + load (cuEq conversion) | 7.6 s | 8.3 s |
| build (dynamo trace, compile) | 15.4 s | 21.9 s |
| first step (compiles the padded variant) | 2.3 s | 19.6 s |
| total | 25.7 s | 50.0 s |

A fresh process costs about 25 s even with warm caches, and the edge-bucket test needs an empty
Inductor cache by design. The Triton and FastMACE caches stay shared, because they change no
compile decision. No `dynamo/code_state_*.pkl` profile exists in either cache in this torch build
(local PGO is off without a job id). So a shared Inductor cache would probably detect the same
recompile, but the test keeps the fresh one, as the MH-1 test does.

The tiny **hybrid** edge-bucket test passed too (48 s warm, 68 s cold). It was left out of the fast
tier because of its cost. The hybrid case stays covered by the MH-1 hybrid test in the full tier.

## Fast tier now

`pytest tests/gpu -m "not slow" -n 0` (the command in [development.md](../development.md)):
**195 passed, 3 skipped, 52 deselected in 27.2 min.** The two new tests take about 1.5 min of this.
The rest of the growth from 1b-T24's 3.4 min comes from GPU tests merged after 1b-T24 without the
`slow` mark. The slowest were:

| Test | Time |
| --- | ---: |
| `test_load_hybrid_gpu.py::test_mace_imported_first_gives_plain_cueq_with_the_reason` | 310 s |
| `test_canonical_restart_gpu.py::test_mh1_plain_cueq_float64_resumes_within_1e_12` | 203 s |
| `test_selfcheck_gpu.py::test_gate_passes_in_graph_mode[mh1-r2scan-nvt]` | 123 s |
| `test_registry_gpu.py::test_polar_1_s` | 77 s |
| `test_selfcheck_gpu.py::test_gate_passes_in_graph_mode[mh1-r2scan-npt-full]` | 75 s |
| `test_registry_gpu.py::test_mh1_npt_baoab_full_cell` | 70 s |

They need re-tiering under 1b-T24's rules. That is outside this task.
