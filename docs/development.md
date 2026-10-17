# Development

## CPU development venv

CPU development and the install check use `~/environments/fastmace_cpu` in
WSL2 Ubuntu on the PC. It is FastMACE's own environment and may be rebuilt at
any time; the CPU CI follows the same recipe. It installs the main checkout
in editable mode, so it outlives any worktree and always sees the code that
is checked out there.

Keep pip caches in `~/pip-cache` and temporary files in `~/tmp`:

```bash
export PIP_CACHE_DIR=~/pip-cache TMPDIR=~/tmp
python3.10 -m venv ~/environments/fastmace_cpu
~/environments/fastmace_cpu/bin/python -m pip install --upgrade pip
~/environments/fastmace_cpu/bin/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
~/environments/fastmace_cpu/bin/python -m pip install -e "/mnt/d/Margraf/FastMACE[dev]"
~/environments/fastmace_cpu/bin/python -m pip check
~/environments/fastmace_cpu/bin/python -m pip freeze > ~/environments/fastmace_cpu.requirements.txt
```

Torch comes from the PyTorch CPU index first, so the editable install keeps
it and does not pull a CUDA build from PyPI. The installed versions are
listed in `~/environments/fastmace_cpu.requirements.txt`.

Check that it works:

```bash
~/environments/fastmace_cpu/bin/python -c "import fastmace, torch; print(fastmace.__version__, torch.__version__)"
```

Run the CPU suite from the repository root (at most 8 workers locally, at
most 2 while other sessions run tests):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ~/environments/fastmace_cpu/bin/python -m pytest -n 8 tests/cpu
```

Never install FastMACE into the GPU environments; GPU runs use
`PYTHONPATH=<checkout>` with the environment's Python (AGENTS.md).

## Bitwise equality in CPU float64 tests

Found in task 0-T04 (17 October 2026, `fastmace_cpu`, torch 2.14.1+cpu,
MACE 0.3.16). On CPU with more than one OpenMP thread, the **first model
evaluated in a process** gives results that differ from every later model by
about 2e-17 relative (1 force component in 192 on a tiny NaCl model). It does
so even when the weights are bitwise equal. Repeated evaluations of one
model are bitwise equal, and all later models agree bitwise with each other.
The difference disappears with `OMP_NUM_THREADS=1`, with TorchScript's
profiling executor disabled (`torch._C._jit_set_profiling_executor(False)`),
or after one throwaway evaluation in the process.

For equivalence tests (FastMACE against the research engine, phase 1a), use
either of these:

- compare at `rtol=1e-12` (forces also `atol=1e-14`) in float64; or
- run one throwaway evaluation in the process before comparing bitwise.

Do not compare the first evaluation of a process bitwise against anything.

## CI

The CPU CI is `.github/workflows/cpu.yml` (SPEC §12.4). It runs on every
branch push and by hand (`workflow_dispatch`); there is no separate
`pull_request` trigger, because pull requests come from branches of this
repository and would run the same commit twice. A newer push to the same
branch cancels the running workflow (`concurrency`), and pip downloads are
cached, since Actions minutes count on a private repository.

Matrix, all on `ubuntu-latest` with CPU torch from the PyTorch CPU index:

| Cell | Python | torch | Other packages |
| --- | --- | --- | --- |
| latest | 3.10, 3.11, 3.12 | 2.14.1 | newest the `pyproject.toml` floors allow |
| previous minor | 3.10, 3.11, 3.12 | 2.13.0 (newest 2.13.x) | newest the floors allow |
| qualified | 3.10 | 2.14.1 | `-c constraints/qualified.txt` (SPEC §4.1 pins) |

`pyproject.toml` keeps the §4.1 floors, so the six plain cells test what a
user gets from a fresh install today. `constraints/qualified.txt` pins the
qualified stack (torch 2.14.1, mace-torch 0.3.16, cuequivariance and
cuequivariance-torch 0.12.0, e3nn 0.4.4, ASE 3.27.0, NumPy 2.2.6); update it
together with SPEC §4.1.

Each cell installs torch first, then `pip install -e ".[dev]"`, runs
`ruff check fastmace tests` and then
`CUDA_VISIBLE_DEVICES='' pytest -n auto tests/cpu tests/gpu`; the GPU tests
skip. The same checks locally, from the repository root:

```bash
~/environments/fastmace_cpu/bin/ruff check fastmace tests
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 ~/environments/fastmace_cpu/bin/python -m pytest -n 8 tests/cpu tests/gpu
```

Ruff's rules are in `pyproject.toml` (`E`, `W`, `F`, `I`, line length 120).
`fastmace` and `_support` are first-party for isort; a module that must
import `fastmace` before `torch` marks that with `# isort: split`
(`tests/conftest.py`).

Install check, 17 October 2026 (`pip install --dry-run` in throwaway venvs):
MACE 0.3.16, e3nn 0.4.4 and cuEquivariance 0.12.0 (pinned) or 0.13.0
(newest) resolve on Python 3.10, 3.11 and 3.12 with torch 2.14.1 and
2.13.0, so no cell is dropped. The newest resolution then was ASE 3.29.0,
NumPy 2.5.4 and filelock 4.2.0.

## NAS checkout

The workstations (murphy, leto, coraline, ...) see FastMACE at
`/mnt/homes/zwingel/FastMACE`. It is a plain directory, not a git
repository, like ResearchModule's NAS copy. `.source_commit` holds the
commit it was copied from. Caches and temporary files of runs there go to
`/mnt/homes/zwingel/fastmace-cache/`. Results and other NAS-only files
must never be deleted, so every sync only adds and overwrites tracked
files. Nobody edits tracked files on the NAS; check first that the NAS
copy matches its `.source_commit`, so an edit made there is not lost:

```bash
C=$(ssh murphy cat /mnt/homes/zwingel/FastMACE/.source_commit)
git -c core.quotepath=false ls-tree -r $C | awk -F'\t' '{split($1,a," "); print a[3]"  "$2}' > local.txt
cut -c43- local.txt | ssh murphy 'cd /mnt/homes/zwingel/FastMACE && git hash-object --stdin-paths' | paste -d' ' - <(cut -c43- local.txt) | diff <(sed 's/  / /' local.txt) -
rm local.txt
```

No output means no NAS-side edits. Then copy `main` with LF line endings
(a plain `git archive` on Windows writes CRLF) and record the commit:

```bash
git -c core.autocrlf=false archive --format=tar main | ssh murphy 'tar -x -C /mnt/homes/zwingel/FastMACE'
ssh murphy "echo $(git rev-parse main) > /mnt/homes/zwingel/FastMACE/.source_commit"
```

`tar -x` never deletes, so a file removed from git stays on the NAS; remove
it by hand after checking that it is not a result. Run tests there with the
NAS environment's Python and `PYTHONPATH=/mnt/homes/zwingel/FastMACE`; the
full command is in `docs/validation/phase-0.md`. pytest needs
`--basetemp=/mnt/homes/zwingel/fastmace-cache/<name>`: under a `TMPDIR` on
the NAS it refuses its `pytest-of-zwingel` directory, because NFS reports
another owner uid (1024) than the workstation login (1012).
