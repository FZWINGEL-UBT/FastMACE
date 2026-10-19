# 1b-T08b: the hybrid after a plain `fm.load` (GPU check)

Plan task 1b-T08b. Question: after `fm.load("mh1-r2scan")` (the pristine e3nn
model, which imports MACE), does the minimal planner still choose the hybrid
backend? `fm.load` prepares the OEq runtime before MACE is imported
(`loading.prepare_hybrid_if_possible`, merged in b4bc2f4).

## Run

- Machine: local PC, WSL2 Ubuntu. GPU: NVIDIA GeForce RTX 5060 Ti (driver
  610.88), otherwise idle (desktop only; a CPU-only 1b-T24 A/B ran at the same
  time; not a timing run).
- FastMACE: main 36cee51 plus the test fix of this branch (the backend decision
  key is `EngineConfig.backend`). No baseline: the check has no research-engine
  side.
- pytest in `mace_cueq_latest` (torch 2.14.1+cu130, MACE 0.3.16,
  cuequivariance-torch 0.12.0, no OpenEquivariance); hybrid children in
  `FASTMACE_HYBRID_PYTHON=~/environments/mace_cueq_oeq/bin/python` (same torch
  and MACE, OpenEquivariance 0.7.0). Model `mh1-r2scan` from
  `FASTMACE_MODELS=~/models`, verified against the label table's SHA-256.
- Command (detached, `TORCHINDUCTOR_COMPILE_THREADS=4`, `OMP_NUM_THREADS=2`):
  `python -m pytest -n 0 -v tests/gpu/test_load_hybrid_gpu.py`.

## Result: 3 passed in 366 s

Each case is a fresh interpreter: `fm.load("mh1-r2scan")`, then
`MinimalPlanner().plan_evaluation` on NaCl 2×2×2.

| Case | Interpreter | `oeq_runtime` | Backend | Reason code |
| --- | --- | --- | --- | --- |
| plain `fm.load` | `mace_cueq_oeq` | `prepared` | `hybrid` | `backend-hybrid` |
| `import mace.calculators` before `fastmace` | `mace_cueq_oeq` | `too late: mace was imported first` | `cueq` | `backend-cueq-mace-imported-first` |
| no OpenEquivariance | `mace_cueq_latest` | `skipped: OpenEquivariance is not installed` | `cueq` | `backend-cueq-oeq-unavailable` |

The first run of the file failed in all three cases on a bug in the test
itself: it looked for a decision named `backend`. The planner records it as
`EngineConfig.backend`. Nothing else changed between the two runs.
