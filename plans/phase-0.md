# Phase 0: skeleton

Status: planned, 17 October 2026. Spec: [SPEC.md](../SPEC.md) §5, §12.2
(CPU unit class), §12.4, §14, §16 row 0. Estimate: 1 week.

## Goal

An installable, empty-but-real `fastmace` package with its import-time
behaviour, test infrastructure, layering checks and CPU CI, so that phase 1a
workers can drop extracted code into a structure that already enforces the
rules.

**Acceptance (SPEC §16):** `pip install -e .` works; the CPU suite is green.
In detail:

1. `pip install -e "FastMACE[dev]"` succeeds in a fresh venv with CPU torch.
2. `import fastmace` before `import torch` makes PyTorch's file locks soft
   locks, and the CUDA driver cache rule of §14 holds.
3. The CPU suite passes locally (`-n 8`) and in CI on Python 3.10, 3.11 and
   3.12 with the two newest torch minors.
4. The layering test passes, and fails on a deliberately bad import.
5. On a GPU machine (RTX 5060 Ti and one workstation), `pytest tests/gpu`
   runs its smoke test; on CPU it skips cleanly.

Phase 0 copies no engine code. It depends on none of the contract wording,
so it can run while the phase 1a design (1a-T01, 1a-T02) is being written.

## Owner decision needed first

- **Licence (SPEC §18 decision 4).** §14 proposes MIT. Needed for T01's
  `LICENSE` and `pyproject.toml`. If it is not decided when T01 starts, T01
  leaves `LICENSE` out and marks the `license` field `TODO`; nothing else
  waits on it.

## Tasks

Each task: ID, size (S < ½ day, M ≈ 1 day, L > 1 day), dependencies, spec
sections, files it owns, tests to write first, done-when.

### 0-T01 Package skeleton (M) — no dependencies

- Spec: §5 (layout), §4.1 (dependency pins), §14 (name, extras).
- Files: `FastMACE/pyproject.toml`, `FastMACE/README.md`, `FastMACE/LICENSE`
  (after decision 4), `FastMACE/fastmace/__init__.py` (`__version__ =
  "0.0.1.dev0"`), one `__init__.py` with a one-line docstring per subpackage
  of §5 (`model`, `rewrites`, `kernels`, `neighbors`, `integrators`, `mc`,
  `optimize`, `engine`, `planner`, `reference`, `longrange`, `io`,
  `protocols`, `ase`, `cli`, `bench`), `fastmace/errors.py` with only the
  base class `FastMACEError` (the hierarchy of §7.6 comes in 1b),
  `FastMACE/docs/index.md`, `FastMACE/docs/validation/.gitkeep`.
- Build backend: setuptools (`[tool.setuptools.packages.find]`), flat layout
  as §5 shows. Python `>=3.10`. Base dependencies: `torch`, `mace-torch` (pin
  `==0.3.16`), `ase>=3.23`, `numpy`, `cuequivariance-torch>=0.12`. Take the
  exact package names and current versions from
  `/mnt/homes/zwingel/environments/mace_cueq_latest.requirements.txt` (or the
  local `~/environments/mace_cueq_latest.requirements.txt`). Extras per §14:
  `oeq` (`openequivariance>=0.7`), `polar` (`graph-longrange==0.4.0`), `d3`
  (`torch-dftd==0.5.1`), `full` (union of the three), `dev` (`pytest`,
  `pytest-xdist`, `ruff`).
- pytest configuration in `pyproject.toml`: `testpaths = ["tests"]`,
  `addopts = "-p no:cacheprovider"`, the third-party warning filters from
  `Experiments/Langevin_Thermostat/pytest.ini`, markers `gpu` and `slow`.
- Tests first: `tests/cpu/test_package.py`: version string, every §5
  subpackage importable, `fastmace.errors.FastMACEError` is an `Exception`.
- Done when: the tests pass from an editable install (T02's venv, or any
  venv with CPU torch).

### 0-T02 CPU development venv (S) — after T01

- Files: none in git except a short `FastMACE/docs/development.md` section
  with the recipe.
- In WSL: `python3.10 -m venv ~/environments/fastmace_cpu`, CPU torch
  2.14.1 from the PyTorch CPU index, then `pip install -e "FastMACE[dev]"`
  from the Linux view of the checkout (WSL reads `G:` as `/mnt/g`). Keep pip
  caches in `~/pip-cache` and temporary files in `~/tmp` (root AGENTS.md).
  Run `pip check`, then export `~/environments/fastmace_cpu.requirements.txt`.
- Record the venv in the root AGENTS.md list of local environments.
- Done when: `python -c "import fastmace, torch; print(fastmace.__version__, torch.__version__)"`
  works and T01's tests pass in this venv.

### 0-T03 Import-time behaviour (M) — after T01

- Spec: §14 "Caches" (soft locks, `CUDA_CACHE_DISABLE` on NFS, cache root,
  opt-out), §13 (lock mode in provenance), §3.2 (process-wide state).
- Source: `Experiments/Langevin_Thermostat/langevin/file_locks.py` (67
  lines) and its test `tests/core/test_file_locks.py` (170 lines); the
  NFS rule in `studies/hybrid_oeq/scripts/oeq_no_nvcc.sh`;
  `docs/NON_LOCKING_DESIGN.md`; `environment/nfs_soft_locks/` (the start-up
  hook already installed in our environments; both must coexist).
- Files: `fastmace/_file_locks.py`, `fastmace/_cache.py` (the cache root
  from `FASTMACE_CACHE`, defaulting to the user cache directory, never inside
  `sys.prefix`; subdirectories for Triton, Inductor, OEq, compile caches; it
  sets nothing until an engine asks for it), edits to `fastmace/__init__.py`
  so the lock switch runs first. The opt-out is `FASTMACE_SOFT_FILE_LOCKS=0`;
  `=1` insists (as `LANGEVIN_SOFT_FILE_LOCKS`). The resulting mode is exposed
  read-only (`fastmace.runtime_info()` returns a frozen record) for
  provenance later.
- `CUDA_CACHE_DISABLE=1` is set at import only when the CUDA driver cache
  directory is on NFS and the user has not set it. Detect NFS from
  `/proc/self/mounts`, not by shelling out to `stat`.
- Tests first: port `test_file_locks.py` to `tests/cpu/test_import_time.py`,
  plus: importing `fastmace` after torch still works but reports that the
  switch came late; the NFS rule against a fake mounts table; the cache root
  is never inside `sys.prefix`; importing with the `nfs-soft-locks` hook
  already active is a no-op.
- Done when: the tests pass in a fresh interpreter per test (subprocess),
  because import order is the thing under test.

### 0-T04 Test infrastructure (M) — after T01

- Spec: §12.2 (test classes), §12.4.
- Source: `tiny_mace_calculator` in
  `Experiments/Langevin_Thermostat/tests/core/test_gpu_langevin.py:63` and
  the structure helpers used by `tests/core/test_generic_rewrites.py` and
  `tests/core/test_gpu_npt.py`. Copy them; never import from `Experiments/`.
- Files: `tests/conftest.py` (GPU tests are skipped with a clear reason when
  CUDA is absent; `gpu` and `slow` markers; seed fixture), `tests/_support/`
  (`tiny_models.py` with the tiny model builder for each interaction class
  of §4.3 that MACE 0.3.16 can build; `structures.py` with small NaCl,
  NaNbO3 and NbCl5 cells built in code, so tests need no files from
  `Structures/`), `tests/cpu/`, `tests/gpu/`, `tests/gpu/test_smoke.py`
  (`torch.cuda.is_available()`, one tiny-model forward on the GPU).
- Tests first: `tests/cpu/test_support.py`: each tiny model builds, gives
  finite energy and forces on each structure, and is deterministic in
  float64.
- Done when: the CPU suite passes with `-n 8`, and `tests/gpu` skips on CPU
  and passes on the RTX 5060 Ti (the GPU part is queued by the conductor).

### 0-T05 Layering checks (S) — after T01

- Spec: §5 layering rules; AGENTS.md "Code rules".
- Files: `tests/cpu/test_layering.py`.
- An AST walk over `fastmace/**.py` asserts:
  - `fastmace.model` and `fastmace.rewrites` import nothing from
    `fastmace.engine`;
  - `fastmace.integrators` imports neither `fastmace.engine` nor
    `fastmace.model` internals (only the `ForceField` interface, once it
    exists);
  - no module imports `langevin`, `common`, `model_store`, anything under
    `Experiments`, or contains `/mnt/homes`;
  - no module uses the `global` statement, and no module outside
    `fastmace/config.py` and an explicit allow-list of constants defines a
    module-level UPPER_CASE name bound to a `bool` or `None` (the research
    engine's switch pattern).
- Tests first: the test itself, with a negative fixture (a temporary package
  that breaks each rule) proving that every rule can fail.
- Done when: green on the skeleton, red on each negative fixture.

### 0-T06 CPU CI (M) — after T01, T04

- Spec: §12.4 (CPU: Linux, Python 3.10–3.12, two torch versions).
- Files: `.github/workflows/fastmace-cpu.yml` at the repository root
  (`paths: FastMACE/**`), running `pip install -e "FastMACE[dev]"` with CPU
  torch and `pytest -n auto tests/cpu`.
- Matrix: Python 3.10, 3.11, 3.12 × torch 2.14.1 and the newest 2.13.x.
  First check that MACE 0.3.16, cuEquivariance 0.12 and e3nn 0.4.4 install
  on 3.11 and 3.12. If one does not, drop that cell, record why in
  `docs/development.md` and tell the conductor; this is a §4.1 policy
  question for the owner.
- Pushing a workflow file needs a token with the `workflow` scope. If the
  push is refused, stop and report it; do not change token or `gh`
  settings.
- The repository is private, so Actions minutes count against the account:
  cache pip downloads, and use `concurrency` to cancel superseded runs.
- Done when: a push to the task branch shows the matrix green, or the
  refusal is reported.

### 0-T07 NAS presence (S) — after T01–T06 are merged

- `FastMACE/` is not yet in the NAS checkout (which holds `Experiments/`,
  `Models/`, `Papers/` and `Structures/`). Add it with the fast-sync
  recipe of the root AGENTS.md (LF files from `git archive`, one `rsync
  --files-from`). Nothing on the NAS is overwritten because nothing exists
  there yet. Check that first.
- Run `tests/cpu` there once with the NAS `mace_cueq_latest` Python and
  `PYTHONPATH=/mnt/homes/zwingel/ResearchModule/FastMACE` (`-n 24` on a
  workstation), and `tests/gpu` on one idle workstation GPU (murphy or
  leto).
- Done when: both pass, and the result is recorded in
  `docs/validation/phase-0.md` (machine, GPU, environment, commit).

## Order and parallelism

```text
T01 ──┬── T02
      ├── T03
      ├── T04 ── T06
      └── T05
T02..T06 merged ── T07
```

After T01, the tasks T02, T03, T04 and T05 can run as parallel worker
sessions: their files do not overlap. T01 owns `pyproject.toml` and
`fastmace/__init__.py`. T03 may edit `__init__.py` only to call the lock
switch first. Any other change to a file owned by another task goes back to
the conductor.

## Phase exit

- Conductor merges, runs the CPU suite and `tests/gpu` locally, and writes
  `docs/validation/phase-0.md`.
- Owner checkpoint: acceptance items 1–5 above, then commit, push and NAS
  sync.
