# FastMACE agent instructions

These rules apply to every session in this repository
(`FZWINGEL-UBT/FastMACE`, private; local checkout `D:\Margraf\FastMACE`,
WSL `/mnt/d/Margraf/FastMACE`).

## What this is

FastMACE is the standalone library specified in [SPEC.md](SPEC.md): pretrained
MACE models on NVIDIA GPUs with exact rewrites, CUDA graphs and a planner. It
is extracted from the research engine in the **ResearchModule** repository
(`FZWINGEL-UBT/ResearchModule`, local checkout `G:\Margraf\ResearchModule`,
WSL `/mnt/g/Margraf/ResearchModule`), directory
`Experiments/Langevin_Thermostat/langevin/`. Until phase 1a ends, that engine
is the working production code, and its `main.py` must keep running.

FastMACE was developed inside ResearchModule until 17 October 2026. It moved
to this repository because worktrees of ResearchModule on its USB hard disk
took longer than the app's five-minute start limit (SPEC §18 decision 1).

## The research engine and its repository

- In SPEC.md and the plans, paths starting with `Experiments/`, `Structures/`
  or `environment/` are paths in ResearchModule, not here.
- **Read ResearchModule's sources from `G:\Margraf\ResearchModule`, and only
  read them.** Never create a worktree, branch or commit there from a FastMACE
  task (it is on a slow USB disk, and other sessions work in it). Never import
  from it in FastMACE code; copy code across. (`D:\Margraf\ResearchModule` is
  an old copy from 12 October 2026; do not use it.)
- The research engine frozen for A/B comparisons is ResearchModule's tag
  `research-engine-baseline-20261017` (commit d2faa9fd).
- Tasks that must change ResearchModule (phase 1a T17: `main.py` on FastMACE)
  are done by the conductor, or by a session started there on purpose,
  following ResearchModule's own `AGENTS.md`.
- **ResearchModule's `AGENTS.md` still governs everything outside this
  repository:** environments (which Python on which machine; no installs into
  the NAS environments without its dry-run procedure), workstation rules
  (never while another user is on the GPU or has active jobs, checked right
  before every job; idle leftover sessions are fine; files only under
  `/mnt/homes/zwingel`, no `/tmp`), WSL
  rules (detached launches, 15 GB RAM), and the NAS sync procedure. Read its
  sections "Workstation environment", "NAS checkout" and "Local PC" before
  any run on a GPU or a workstation.

## How work is organised

- **One conductor session** owns the plans in [plans/](plans/), the phase
  status and `main`. It assigns tasks and reviews and merges them, and writes
  little code itself.
- **Each worker session does one plan task**, in its own git worktree of this
  repository. It pushes its work as branch `fastmace/<phase>-T<nn>`, for
  example `fastmace/1a-T05`. The brief names the plan task. The task names
  the spec sections to read and the files to create or change.
- **Read only the spec sections the task names.** SPEC.md is 2,500 lines;
  do not load all of it. Use `grep -n '^##' SPEC.md` to find a section.
- **The spec is the contract.** If it is wrong, unclear or contradicts the
  research engine, stop and report to the conductor. Do not improvise around
  it. The conductor amends SPEC.md and adds a §19 revision entry.
- **Write the tests first.** Each task lists its acceptance tests. Write them,
  watch them fail, then write the code.
- **A task ends with a pushed branch and a short report:** what changed, the
  test command and its output, anything left over, and any deviation from the
  plan. Do not merge it yourself.
- Stage only your own files (never `git add -A` over a broad directory), LF
  line endings, commit messages ending with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Pushing: this repository's `.git/config` has a credential helper that uses
  the `FZWINGEL-UBT` account's `gh` token, so a plain `git push` works from
  any worktree. Do not change `gh` or global git settings, and never print
  the token.

## Code rules (SPEC §5)

- Package layout as in SPEC §5: `fastmace/`, `tests/` and `docs/` at the
  repository root.
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
  they explain why, and cite the source as `langevin/<file>.py:<function>`
  at ResearchModule tag `research-engine-baseline-20261017` in the commit
  message, not in the code.

## Environments and commands

- **CPU development and the install check (WSL2 Ubuntu on the PC):**
  `~/environments/fastmace_cpu`, a fresh venv with CPU torch and
  `pip install -e "/mnt/d/Margraf/FastMACE[dev]"`, the same recipe as the CPU
  CI (plan task 0-T02). It is FastMACE's own environment and may be rebuilt
  at any time.
- **GPU runs, local and on workstations:** never install FastMACE into the
  existing environments. Run it from a checkout with `PYTHONPATH=<checkout>`
  and the environment's Python: locally `~/environments/mace_cueq_latest`
  (plain cuEq) or `mace_cueq_oeq` (hybrid); on workstations the NAS
  environments, with the checkout at `/mnt/homes/zwingel/FastMACE` (plan task
  0-T07). Until plan task 1a-T00 replaces it, hybrid runs still go through
  ResearchModule's
  `Experiments/Langevin_Thermostat/studies/hybrid_oeq/scripts/oeq_no_nvcc.sh`.
- **CPU suite:** from the repository root,

  ```bash
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ~/environments/fastmace_cpu/bin/python -m pytest -n 8 --dist=load --maxschedchunk=1 tests/cpu
  ```

  At most 8 workers locally (WSL has 15 GB RAM); check free memory first, and
  never run two suites at once. While sibling workers are running, use at
  most `-n 2`.
  Keep `--dist=load --maxschedchunk=1`: the CPU suite mixes very short tests
  with long MC statistics cases, and large consecutive batches leave workers idle.
- **GPU tests and benchmarks** are scheduled by the conductor, one job per
  idle GPU. Timing comparisons count only from alternating A/B blocks in one
  process on an otherwise idle GPU (cross-process noise is about 3%).
- **Baseline for A/B:** export ResearchModule's tag, from
  `G:\Margraf\ResearchModule`, with
  `git -c core.autocrlf=false archive research-engine-baseline-20261017 Experiments/Langevin_Thermostat Experiments/Benchmark_MACE_GPU_OPTIMIZATIONS/scripts`
  into a directory outside both checkouts (WSL: `~/fastmace-baseline/`;
  workstations: `/mnt/homes/zwingel/fastmace-baseline/`).

## Recording results

- Every GPU result names the machine, GPU, environment, backend, precision,
  model label and the git commits of both FastMACE and the baseline.
- Results and reports go to `docs/validation/` (markdown) and
  `results/<run-id>/` (JSON). Large files stay out of git.
- After each completed task that produced results: update the docs, commit,
  push, and sync to the NAS checkout `/mnt/homes/zwingel/FastMACE`.
- **NAS and workstations available again (owner, 30 October 2026):** the
  pause of 26 October is lifted. NAS syncs and workstation jobs follow the
  rules above and ResearchModule's `AGENTS.md` (idle check before every job,
  files only under `/mnt/homes/zwingel`).
