# 1b-T27 GPU acceptance tier (19 October 2026)

Goal: the full GPU tier back to 30–40 minutes without losing a check. 1b-T23 measured it at 142 min
locally and 135 min on murphy, plus 25 min of research-engine comparisons.

## Tiers

| Tier | Command | Tests | Local RTX 5060 Ti, warm |
| --- | --- | ---: | ---: |
| fast | `pytest tests/gpu -m "not slow and not acceptance"` | 190 | 7.9 min ¹ |
| full | `pytest tests/gpu -m "not acceptance"` | 244 ² | **38.7 min** ³ (was 142 min) |
| acceptance | `pytest tests/gpu -m acceptance` | 10 | murphy: ACCEPTANCE-RESULT |

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
  `tests/conftest.py` refuses an `acceptance` mark without `slow`. It stops a run that selects
  acceptance tests with a usage error when `FASTMACE_BASELINE`, `FASTMACE_HYBRID_PYTHON` or
  `FASTMACE_MODELS` is unset or names nothing.
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

ACCEPTANCE-SECTION
