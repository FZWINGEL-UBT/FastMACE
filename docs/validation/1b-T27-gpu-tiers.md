# 1b-T27 GPU acceptance tier (19 October 2026)

Goal: the full GPU tier back to 30–40 minutes without losing a check. 1b-T23 measured it at 142 min
locally and 135 min on murphy, plus 25 min of research-engine comparisons.

## Tiers

| Tier | Command | Tests | Local RTX 5060 Ti, warm |
| --- | --- | ---: | ---: |
| fast | `pytest tests/gpu -m "not slow and not acceptance"` | 190 | 7.9 min ¹ |
| full | `pytest tests/gpu -m "not acceptance"` | 244 ² | **38.7 min** ³ (was 142 min) |
| acceptance | `pytest tests/gpu -m acceptance` | 10 | murphy: **103 min**, 10 passed |

1. The sum of the fast-tier tests' times in the full-tier run (the tool's total).
2. Collected; the run had 241 passed and 3 skipped, without the OMOL-0 XL gate case.
3. Measured: 242 passed, 3 skipped, 9 deselected in 41.1 min (wall 41.6 min). That run still had the
   OMOL-0 XL gate case (147 s), which then moved to acceptance; without it, 38.7 min.
   `tools/test_budget.py` on its junit: fast 7.9 min, full 38.3 min of test time, all within budget.

Local run: WSL2, RTX 5060 Ti; `mace_cueq_latest` (torch 2.14.1+cu130), hybrid children in
`mace_cueq_oeq`; `TORCHINDUCTOR_COMPILE_THREADS=4`, `-n 0`, warm shared caches; branch
`fastmace/1b-T27` at 28932d1 plus the selfcheck mark. Junit: `results/1b-T27-local-full-20261019/junit.xml`.

## What moved

**To acceptance** (each keeps `slow`):

| Test | 1b-T23 time | Why at acceptance |
| --- | ---: | --- |
| `test_ase_equivalence_gpu.py::test_langevin_reproduces_ase_in_every_50_K_bin` | 70 min | statistics; the CPU suite checks the integrators against ASE exactly |
| `test_ase_equivalence_gpu.py::test_baoab_variants_reproduce_their_references` | 18 min | statistics; as above |
| `test_canonical_restart_gpu.py::test_mh1_hybrid_resumes_statistically` | 6 min | statistics; the plain cuEq 1e-12 Å restart test stays in full |
| `test_selfcheck_gpu.py::test_gate_time_on_nbcl5` | 1 min | a timing record |
| `test_selfcheck_gpu.py::test_gate_passes_in_graph_mode[omol0-xl]` | 2.5 min | one more model through the gate; MH-1 (NVT, NPT), MP-0 and POLAR-1-S stay in full |
| `test_engine_gpu.py::test_mh1_*_against_the_research_engine` (3) | 16 min ⁴ | research-engine comparisons |
| `test_forces_gpu.py::test_mh1_agrees_with_maceforces[*compiled-production]` (2) | 7 min ⁴ | research-engine comparisons; the eager case stays in full |

4. murphy, from 1b-T23's rerun (`skipped-junit.xml`).

**Merged:** `test_planner_gpu.py::test_mh1_{plain_cueq_float64,hybrid_float32}_planned_equals_hand_built`
and `test_minimal_planner_gpu.py::test_mh1_*_minimal_equals_fixed` (four tests, 639 s locally in 1b-T23)
became `test_planner_gpu.py::test_mh1_*_planned_equals_hand_built_and_minimal` (two tests, 217 s and
78 s). One fresh process per backend builds the `FixedPlanner` engine, the hand-built engine (1a-T13's
settings), the `MinimalPlanner` engine and, for the hybrid, a second `FixedPlanner` engine for the
scatter. Every old assertion is kept. What changed:

- the hybrid comparisons (hand-built against planned, minimal against planned) are now limited by
  three times the scatter between two planned engines **in the same process**. Before, the limits
  came from two hand-built processes (planned against hand-built) and two `FixedPlanner` processes
  (minimal against fixed). The floor of 1e-6 Å is unchanged;
- in the hybrid, the hand-built engine's edge sizes and rebuilds must now equal the planned
  engine's (before, only plain cuEq checked them); the minimal engine's checks are unchanged;
- the hybrid timing assertion (planned less than 1.25 times hand-built) compares with one
  hand-built run in the same process, not the faster of two processes;
- the reports are now named `1b-T27-mh1-*-three-engines`.

**Back to the fast tier:** `test_load_hybrid_gpu.py::test_mace_imported_first_gives_plain_cueq_with_the_reason`,
6.4 s instead of 310 s (below).

## The 310 s test: a stale torch-extension build lock

In `test_mace_imported_first_gives_plain_cueq_with_the_reason`, MACE is imported before FastMACE. A
stack dump every 30 s showed the child waiting in OpenEquivariance's `_wait_for_torch_build_lock`
for 307 of its 308 s. OEq 0.7 has no precompiled extension in `mace_cueq_oeq`, so without FastMACE's
prepared runtime it takes the JIT path in torch's default extensions directory. There it found
`~/.cache/torch_extensions/py310_cu130/libtorch_tp_jit/lock`, an empty lock file left by a build killed
on 8 October 2026 at 20:32. It waits up to 300 s, raises, and OEq imports without its extension. The
test still passed, because the expected outcome is plain cuEq.

The time depended on run order. FastMACE's OEq runtime sets `TORCH_EXTENSIONS_DIR` in the process
environment when it prepares OEq, so once an earlier test in the pytest process has done that, the
children get a fresh directory. That is probably why 1b-T23 measured 10.8 s locally and 5.2 s on
murphy.

- **Test fix:** the test's child gets an empty `TORCH_EXTENSIONS_DIR` of its own (`tmp_path`), which
  makes it independent of the machine's caches: 6.4 s. The other two tests in the file use it too.
- **Environment:** the conductor deleted the stale lock on the local PC. The workstation caches
  had no old lock files.
- **Finding:** a stale torch-extension lock silently costs 300 s in any process that imports MACE
  before FastMACE. The conductor proposes a library follow-up (`fastmace doctor` detecting stale build
  locks).

## Tier rules and tools

- `pyproject.toml` registers `acceptance`. `tests/_support/tiers.py` defines the tiers.
  `tests/cpu/test_tiers.py` fails for an `acceptance` mark without `slow`. When `-m` selects
  acceptance tests, `tests/conftest.py` stops the run in `pytest_configure` (controller only) with a
  usage error if `FASTMACE_BASELINE`, `FASTMACE_HYBRID_PYTHON` or `FASTMACE_MODELS` is unset or
  names nothing.
- **Fixed after review:** the first version checked during collection, in every xdist worker. A run
  without `-m` over `tests/cpu tests/gpu` therefore collected the acceptance tests, and without the
  three variables it crashed every worker. xdist then reported an INTERNALERROR, blaming each
  worker's first test (`test_acceptance_1b.py::test_rename_source[...]`, an ID with newlines, which
  was only the symptom). The conductor found it on murphy at `-n 8`.
- `tools/test_budget.py` prints the tier totals against 10 min (fast) and 40 min (full) as a warning,
  then fails for:
  - a fast-tier test over 60 s;
  - a full-tier test over 120 s that is not on its allowlist;
  - an allowlisted test over 1.5 times its recorded time.
  
  The allowlist holds the plain cuEq canonical restart (205 s) and the merged plain-cuEq planner
  test (217 s).
- CPU tests: `tests/cpu/test_tiers.py`, which checks among other things that no GPU test is in both
  the fast and the acceptance tier, and `tests/cpu/test_test_budget.py`.

Fed 1b-T23's junit files, the tool flags four tiny-model tests in the fast tier at 66–82 s.
1b-T23's runs were slower throughout; in this run they took 31–59 s.

## Acceptance tier on murphy

murphy (NVIDIA RTX 4000 Ada Generation 20 GB; GPU idle before each run, margraf's CPU jobs running),
NAS environments `mace_cueq_latest` with `mace_cueq_oeq` as `FASTMACE_HYBRID_PYTHON`,
`FASTMACE_BASELINE=/mnt/homes/zwingel/fastmace-baseline/1a-T18` (research engine
`research-engine-baseline-20261017`, d2faa9fd), `FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models`,
`TORCHINDUCTOR_COMPILE_THREADS=4`. The detached copy is `/mnt/homes/zwingel/fastmace-t27`; run root
`/mnt/homes/zwingel/fastmace-acceptance/1b-T27-murphy-20261019` (`run.sh`, logs).

| Test | Time |
| --- | ---: |
| `test_ase_equivalence_gpu.py::test_langevin_reproduces_ase_in_every_50_K_bin` | 3749 s |
| `test_ase_equivalence_gpu.py::test_baoab_variants_reproduce_their_references` | 936 s |
| `test_engine_gpu.py::test_mh1_plain_cueq_float64_against_the_research_engine` | 365 s |
| `test_canonical_restart_gpu.py::test_mh1_hybrid_resumes_statistically` | 347 s |
| `test_engine_gpu.py::test_mh1_plain_cueq_float64_npt_against_the_research_engine` | 259 s |
| `test_engine_gpu.py::test_mh1_hybrid_float32_against_the_research_engine` | 202 s |
| `test_forces_gpu.py::test_mh1_agrees_with_maceforces[hybrid-float32-compiled-production]` | 96 s |
| `test_forces_gpu.py::test_mh1_agrees_with_maceforces[cueq-float64-compiled-production]` | 80 s |
| `test_selfcheck_gpu.py::test_gate_passes_in_graph_mode[omol0-xl]` | 73 s |
| `test_selfcheck_gpu.py::test_gate_time_on_nbcl5` | 47 s |

- **Main run** at 28932d1: 9 passed in 101.4 min.
- **OMOL-0 XL case** at 153329e, after it moved to acceptance: 1 passed in 1.3 min.

The three research-engine comparisons in `test_engine_gpu.py` failed in 1b-T23's murphy rerun
(phase-1b.md items 4 and 5) and pass here. Junit: `results/1b-T27-murphy-20261019/`.

## CPU suite under xdist (after the conductor's review)

murphy, NAS `mace_cueq_latest`, `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=2`, `nice`,
`pytest -n 8 tests/cpu tests/gpu`:

| Run | Branch (517207a, main merged in) | main (a2f3159) |
| --- | --- | --- |
| 1 | 4 failed, 2004 passed | 1995 passed |
| 2 | 2008 passed | 1995 passed |
| 3 | 4 failed, 2004 passed | 1995 passed |
| 4 | 2008 passed (`-v`) | |
| 5 | 1 failed (`-v`, reproduction) | |

At 88410c5, before main was merged in, the same command gave 1 failed and 1970 passed.

No INTERNALERROR. The failures are an **order dependence that main has too**. The failing test is
`tests/cpu/test_forces.py::test_polar_force_field_enables_its_own_copy`: one force element off by
5.4e-12 against its 1e-12 tolerance. It fails whenever
`tests/cpu/test_acceptance_1b.py::test_two_simulation_sides_on_the_cpu` ran earlier in the same
process, also on main, without xdist:

```bash
pytest -p no:cacheprovider "tests/cpu/test_acceptance_1b.py::test_two_simulation_sides_on_the_cpu" tests/cpu/test_forces.py::test_polar_force_field_enables_its_own_copy
```

This gives 1 failed, 1 passed on both main and this branch. This branch only changes how xdist
distributes the tests, so the two meet in one worker more often. When the polar test fails, it
leaves its `polar_longrange` patch applied (conftest keeps it as permanent). The three
`test_loading.py::test_conv_fusion_*` tests in the same worker then fail on the extra patch record.
The process state that the CPU `Simulation` test leaves behind has not been found. conftest restores
the default dtype and `fastmace._patches` records after each test, and nothing in `fastmace` sets
torch's thread count, so it is probably something else (not checked by experiment).
