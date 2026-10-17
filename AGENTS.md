# FastMACE agent instructions

These rules apply to every session that works under `FastMACE/`. The
repository-wide rules in the root [AGENTS.md](../AGENTS.md) (environments,
NAS sync, workstation file locations, staging only your own files) still hold.

## What this is

FastMACE is the standalone library specified in [SPEC.md](SPEC.md): pretrained
MACE models on NVIDIA GPUs with exact rewrites, CUDA graphs and a planner. It
is extracted from the research engine in
`Experiments/Langevin_Thermostat/langevin/`. Until phase 1a ends, that engine
is the working production code, and `main.py` must keep running.

## How work is organised

- **One conductor session** owns the plans in [plans/](plans/), the phase
  status and the integration branch (`main`). It assigns tasks and reviews
  and merges them, and writes little code itself.
- **Each worker session does one plan task**, in its own git worktree, on a
  branch named `fastmace/<phase>-<task-id>`, for example `fastmace/1a-T05`.
  The brief names the plan task. The task names the spec sections to read and
  the files to create or change.
- **Read only the spec sections the task names.** SPEC.md is 2,500 lines;
  do not load all of it. Use `grep -n '^##' FastMACE/SPEC.md` to find a
  section.
- **The spec is the contract.** If it is wrong, unclear or contradicts the
  research engine, stop and report to the conductor. Do not improvise around
  it. The conductor amends SPEC.md and adds a §19 revision entry.
- **Write the tests first.** Each task lists its acceptance tests. Write them,
  watch them fail, then write the code.
- **A task ends with a branch and a short report:** what changed, the test
  command and its output, anything left over, and any deviation from the
  plan. Do not merge it yourself.

## Code rules (SPEC §5)

- Package layout as in SPEC §5: `FastMACE/fastmace/`, `FastMACE/tests/`,
  `FastMACE/docs/`.
- `fastmace.model` and `fastmace.rewrites` never import `fastmace.engine`.
- Integrators see only `SimulationState` and a `ForceField`.
- Only the planner instantiates engines.
- State changes outside an integrator step go through `fastmace.events`
  (from phase 3; until then there are no in-place events).
- **No module-level mutable switches.** Every option travels in a config
  object (`fastmace.config`). This is the main difference from the research
  engine, whose `gpu_langevin.FOO = True` globals must not come across.
- No imports from `Experiments/`, `langevin`, `common.py`, `model_store.py`
  or any NAS path. Data paths come from config or environment variables
  (`FASTMACE_CACHE`, `FASTMACE_MODELS`).
- `import fastmace` must happen before `import torch` wherever file locks
  matter (SPEC §14). It switches PyTorch's file locks to soft locks.
- Optimisations must be general: no per-GPU or per-crystal tables,
  shape-keyed caches or one-GPU schedules (SPEC §1.1 goal 2).
- Production is one structure at a time. Do not add batching or concurrent
  structures.
- When code is copied from the research engine, keep its comments where
  they explain why, and cite the source as
  `langevin/<file>.py:<function>` at
  tag `research-engine-baseline-20261017` in the commit message, not in the code.

## Environments and commands

- **CPU development and the install check (WSL2 Ubuntu on the PC):**
  `~/environments/fastmace_cpu`, a fresh venv with CPU torch and
  `pip install -e "FastMACE[dev]"`, the same recipe as the CPU CI (plan task
  0-T02). It is FastMACE's own environment and may be rebuilt at any time.
- **GPU runs, local and on workstations:** never install FastMACE into the
  existing environments. Run it from the checkout with
  `PYTHONPATH=<checkout>/FastMACE` and the environment's Python: locally
  `~/environments/mace_cueq_latest` (plain cuEq) or `mace_cueq_oeq` (hybrid);
  on workstations the NAS environments, following the root AGENTS.md rules
  (idle GPU, files only under `/mnt/homes/zwingel`,
  `PYTHONDONTWRITEBYTECODE=1`). Until plan task 1a-T00 replaces it, hybrid
  runs still go through `oeq_no_nvcc.sh`.
- **CPU suite:** from `FastMACE/`,

  ```bash
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ~/environments/fastmace_cpu/bin/python -m pytest -n 8 tests/cpu
  ```

  At most 8 workers locally (WSL has 15 GB RAM); check free memory first, and
  never run two suites at once.
- **GPU tests and benchmarks** are scheduled by the conductor, one job per
  idle GPU. Timing comparisons count only from alternating A/B blocks in one
  process on an otherwise idle GPU (cross-process noise is about 3%).
- **Baseline for A/B:** the tag `research-engine-baseline-20261017`. Export it
  with `git -c core.autocrlf=false archive research-engine-baseline-20261017 Experiments/Langevin_Thermostat Experiments/Benchmark_MACE_GPU_OPTIMIZATIONS/scripts`
  into a directory outside the checkout (WSL: `~/fastmace-baseline/`;
  workstations: `/mnt/homes/zwingel/fastmace-baseline/`).

## Recording results

- Every GPU result names the machine, GPU, environment, backend, precision,
  model label and the git commit of both FastMACE and the baseline.
- Results and reports go to `FastMACE/docs/validation/` (markdown) and
  `FastMACE/results/<run-id>/` (JSON). Large files stay out of git.
- After each completed task that produced results: update the docs, commit,
  push and sync to the NAS (root AGENTS.md).
