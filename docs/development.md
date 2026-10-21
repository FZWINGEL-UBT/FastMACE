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

## Test tiers

Plan tasks 1b-T24 and 1b-T27 (19 October 2026). The GPU suite has three tiers
(`tests/_support/tiers.py`):

| Tier | Command | Who runs it, when | Local RTX 5060 Ti, warm caches |
| --- | --- | --- | ---: |
| fast (worker) | `pytest tests/gpu -m "not slow and not acceptance"` | every worker, for every change, as often as needed | 7.9 min (190 tests, 1b-T27) |
| full (merge) | `pytest tests/gpu -m "not acceptance"` | the conductor, at merges that touch `model/`, `rewrites/`, `kernels/` or `engine/` | 38.7 min (244 tests; 142 min before 1b-T27) |
| acceptance | `pytest tests/gpu -m acceptance` | phase acceptance, and merges that touch `model/`, `rewrites/` or `kernels/` | murphy: 103 min (10 tests) |

**Merge rule:** the full tier on merges that touch `model/`, `rewrites/`, `kernels/` or `engine/`,
and in addition the acceptance tier on merges that touch `model/`, `rewrites/` or `kernels/`
(they change numerics).

The first run after a change of environment, or with emptied caches, is slower,
because it compiles everything and builds the OEq extension once. Measurements:
[validation/1b-T24-test-suite-speed.md](validation/1b-T24-test-suite-speed.md),
[validation/1b-T27-gpu-tiers.md](validation/1b-T27-gpu-tiers.md).

The CPU suite (`tests/cpu`) has no tiers: everyone runs all of it.

A test is marked `@pytest.mark.slow` when it needs one of these:

- a real checkpoint (MH-1, MP-0, OFF, POLAR-1-S), whose load and compile take tens of seconds;
- a long trajectory (hundreds of steps);
- a cold compiler cache (the edge-bucket regression tests).

A test is also marked `@pytest.mark.acceptance` (and keeps `slow`, so that the older
`-m "not slow"` leaves it out too) when it is:

- a comparison with the research engine (`FASTMACE_BASELINE`) in its own process;
- a statistical check over tens of thousands of steps (ASE equivalence, the hybrid restart);
- a timing record (the gate's time on NbCl5).

Fast- and full-tier tests make no step-time assertions: trials and workers share the GPU, so their
step times are not comparable. A step-time comparison is an acceptance test of its own, which reuses
the numerical test's measurement when both run in one process (2-T14d).

`tests/cpu/test_tiers.py` fails for an `acceptance` mark without `slow`. When a `-m` expression
selects acceptance tests (`-m acceptance`, or `-m slow`), `tests/conftest.py` stops the run with a
usage error if `FASTMACE_BASELINE`, `FASTMACE_HYBRID_PYTHON` or `FASTMACE_MODELS` is unset or names
nothing, so the acceptance tier never skips silently. The check runs in the controller before
collection, so xdist workers never see it. A run without `-m` keeps the tests' own skips.
Run CPU suites under xdist (`-n 2` or more) whenever you change `conftest.py` or a pytest plugin.

Every behaviour tested on the GPU keeps at least one test in the fast tier, usually on a
tiny model or the Lennard-Jones force field in the pytest process, except those listed
below. When you add a GPU test that is slow for one of these reasons, mark it `slow` and
check that a fast test still covers the behaviour. If none can, add the behaviour to that list.

**Budgets.** `tools/test_budget.py` reads junit files (`pytest --junitxml`) and the tiers (by
collecting `tests/gpu`). It prints each tier's total against its budget first (fast 10 min, full
40 min, local RTX 5060 Ti with warm caches; over budget is a warning). It exits 1 if a fast-tier
test takes over 60 s, or a full-tier test over 120 s. The exception is a test that
`FULL_TIER_ALLOWED` in the tool lists with its measured time and a reason; such a test fails only
when it takes more than 1.5 times its recorded time. Run the tool on the junit of a tier run before
you merge new GPU tests:

```bash
python tools/test_budget.py results/<run>/junit.xml
```

GPU command for workers on the local PC (one process; `FASTMACE_HYBRID_PYTHON` runs the
hybrid tests in the OEq environment). `TORCHINDUCTOR_COMPILE_THREADS=4` keeps Inductor from
starting 24 compile workers per process. Without it, WSL's 15 GB ran out on
19 October 2026 while other sessions ran CPU suites, and the WSL VM restarted:

```bash
TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_HYBRID_PYTHON=~/environments/mace_cueq_oeq/bin/python FASTMACE_MODELS=~/models FASTMACE_BASELINE=~/fastmace-baseline/1a-T18 PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 ~/environments/mace_cueq_latest/bin/python -m pytest -m "not slow and not acceptance" tests/gpu
```

Two MH-1 behaviours have tiny-model versions in the fast tier
(`tests/gpu/test_tiny_engine_gpu.py`, plan task 1b-T25): no compile at a later
edge bucket, from a fresh Inductor cache (`test_no_compile_at_a_later_edge_bucket_tiny_plain_cueq`),
and a planned engine equal to a hand-built one
(`test_tiny_plain_cueq_float64_planned_equals_hand_built`). The MH-1 versions stay
in the full tier, and the hybrid edge-bucket case is tested only there.

Plan task 1b-T26 marked `slow` the GPU tests merged after 1b-T24 that run MH-1 or
POLAR-1-S (30 s to 5 min each). Two of them have tiny-model versions in the fast tier
(`tests/gpu/test_tiny_released_gpu.py`): the runtime-state registry in plain cuEq float64
(capture warm-ups leave no trace, the registry covers what the stages mutate), and the gate
failing a broken captured graph while the eager stages pass. Tiny versions of the others were
tried. They cost almost as much as the MH-1 tests, because fresh processes, compiles and the
tests' own child processes dominate, so they were left out
([validation/1b-T26-gpu-retiering.md](validation/1b-T26-gpu-retiering.md)). These behaviours
are tested **only in the full tier**:

- an engine resumed from canonical restart points follows the uninterrupted run within 1e-12 Å
  (`test_canonical_restart_gpu.py`; the CPU suite has tiny-model restart tests);
- the gate passes in graph mode, NVT with fresh references and full-cell NPT
  (`test_selfcheck_gpu.py::test_gate_passes_in_graph_mode`);
- `fm.Simulation` reaches `FixedPlanner`'s trajectory, plain cuEq and hybrid (`test_simulation_gpu.py`);
- LBFGS with `fm.calc` reaches MACE's minimum (`test_ase_calculator_gpu.py`; a tiny random model
  does not converge);
- the registry checks in hybrid pipelined NVT, hybrid full-cell NPT and POLAR-1-S
  (`test_registry_gpu.py`).

Behaviours tested **only in the acceptance tier** (1b-T27):

- Langevin and BAOAB reproduce ASE's integrators statistically (`test_ase_equivalence_gpu.py`,
  88 min; the CPU suite checks both integrators step by step against ASE);
- a hybrid engine resumed from a restart file is statistically the same as the uninterrupted run
  (`test_canonical_restart_gpu.py::test_mh1_hybrid_resumes_statistically`; plain cuEq within
  1e-12 Å stays in the full tier);
- the gate's time on NbCl5 (`test_selfcheck_gpu.py::test_gate_time_on_nbcl5`);
- the planned MH-1 engine less than 25 % slower than the hand-built one, plain cuEq and hybrid
  (`test_planner_gpu.py::*planned_is_as_fast_as_hand_built`, 2-T14d; their numerical comparison stays
  in the full tier);
- FastMACE against the research engine: the MH-1 engine in plain cuEq NVT and NPT and the hybrid
  (`test_engine_gpu.py::*against_the_research_engine`), and the compiled production force fields
  (`test_forces_gpu.py::test_mh1_agrees_with_maceforces[*compiled-production]`; the eager case stays
  in the full tier).

### Caches the suite shares

- **Compiler caches.** `tests/conftest.py` gives the whole run one Inductor/Triton cache,
  inherited by every child process: the FastMACE cache root's (`FASTMACE_CACHE`, else
  `~/.cache/fastmace`, as production runs), unless `TORCHINDUCTOR_CACHE_DIR` is already
  set. `pytest --fresh-compile-cache` compiles everything from scratch in a new directory
  under `<cache root>/test-runs/`. The pytest header names the caches. Tests that need a
  cold cache set their own, explicitly.
- **OEq host extension.** `fastmace.kernels._oeq_runtime` builds OpenEquivariance's host
  extension once per build identity (Python and environment, torch, OpenEquivariance, the
  CUDA runtime wheel, the compiler) in `<cache root>/oeq/extensions/<key>/` and copies it
  into each process's own extension directory. Only the first process after a change of
  environment builds it. Delete that directory to force a rebuild.

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

## GPU CI

Plan task 2-T18, owner decision D2 (SPEC §18 decision 5). `.github/workflows/gpu.yml` runs the GPU
tiers (see "Test tiers") on two self-hosted GitHub Actions runners. They cost no Actions minutes:

| Runner | Machine | GPU | Labels | Directory |
| --- | --- | --- | --- | --- |
| `murphy-sm89` | murphy (shared workstation) | RTX 4000 Ada, sm_89 | `self-hosted, linux, x64, fastmace-gpu, sm_89, murphy` | `/mnt/homes/zwingel/actions-runner/murphy` |
| `local-5060ti-sm120` | the PC, WSL Ubuntu | RTX 5060 Ti, sm_120 | `self-hosted, linux, x64, fastmace-gpu, sm_120, local-5060ti` | `~/actions-runner/local` |

What runs when:

| Event | murphy | local |
| --- | --- | --- |
| push to `main` | fast tier | fast tier |
| nightly, 00:30 UTC | full tier | — (the PC is not always on or idle) |
| by hand (`gh workflow run gpu.yml -f tier=full -f runner=local`) | any tier | any tier |

A push that changes only `plans/`, `docs/`, `results/` or Markdown files starts no GPU run. Each runner
has at most one run in progress and one waiting per ref (`concurrency`, never cancelling a running one).
A burst of merges therefore runs once more, not once per merge.

`gpu.yml` calls `gpu-tier.yml` once per runner. It has two jobs:

- **idle** waits until the GPU is free (`tools/ci/gpu_idle.py check`), for at most 30 minutes, or 150
  minutes at night. The GPU is busy while any compute process is on it: another user's, one of our own
  from another task, or one whose owner `ps` cannot name. It is also busy if `nvidia-smi` fails. WSL's
  `nvidia-smi` lists no processes, so the local runner counts the processes that hold `/dev/dxg`. It
  also needs three consecutive samples under its memory, utilization and RAM limits. The desktop uses
  about 4.3 GB, and WSL needs 10 GB available. Someone else's live local GPU reservation (see the
  next section) also counts as busy. A GPU that stays busy is not a failure: the tests job shows as
  **skipped**, and the idle job's summary names the processes.
- **tests** checks once more, then runs `pytest tests/gpu -m <tier>` with the runner's
  `FASTMACE_CI_PYTHON` and `PYTHONPATH` set to the checkout. Nothing is installed. On the PC the run holds
  the local GPU reservation, and skips if it does not get it within 30 minutes. Then it runs
  `tools/test_budget.py` on the junit file. Over budget fails the job on the local runner, where the
  budgets were measured. On murphy it is informational. While the tests run, a watcher records any
  other user's process that appears on the GPU. It does not stop the run, and the job summary lists
  what it saw. The artefact `gpu-<tier>-<runner>-<attempt>` holds `junit.xml`, `pytest.log`,
  `budget.txt` and the watcher's record for 30 days.

Each runner's `.env` file holds its paths and limits (`tools/ci/runner/env.murphy`,
`tools/ci/runner/env.local`), so the workflow names no machine path. On murphy everything stays under
`/mnt/homes/zwingel`: the runner, its `_work` checkout, `HOME`, `TMPDIR` and a cache of its own
(`fastmace-cache/ci-murphy`).

### The local GPU reservation

WSL on the PC has 15 GB of RAM for every session. A GPU suite started while sibling CPU suites ran
crashed WSL on 19 October 2026. WSL's `nvidia-smi` does not show who is on the GPU. So every local GPU
run holds one reservation, `tools/ci/local_gpu.py`: the GPU CI runner, a worker's GPU suite and a
timing run alike. Every local CPU suite checks it first.

- **Reserve** creates the marker `~/.fastmace-local-gpu` atomically (`mkdir`), with `holder.json`:
  owner, purpose, pid, the pid's start time and the time of the reservation. While another live holder
  has it, reserve waits.
- **Then wait:** holding the marker, it waits until no other `pytest` process runs and at least 10 GiB
  of RAM are available. Then the run starts. New CPU suites do not start meanwhile, because they check
  the marker. If either wait takes longer than `--wait-minutes` (default 60), it releases the marker
  and exits with status 75.
- **Release** removes the marker, only for the pid that holds it.
- **A stale marker** is reported and removed by the next reserve or check. It is stale when its holder
  process is gone, or when its pid now belongs to another process.
- **Check** exits 1 while a live marker exists.

Worker commands on the PC, from the checkout (`python3` is WSL's system Python):

```bash
# A local GPU suite or timing run: reserve, run, release (on any exit; the command's status is returned)
python3 tools/ci/local_gpu.py run --purpose "2-Tnn fast tier" -- <the GPU command>
# The same in a shell script that runs several commands
python3 tools/ci/local_gpu.py reserve --purpose "2-Tnn timing" --pid $$ && trap 'python3 tools/ci/local_gpu.py release --pid $$' EXIT
# Before every local CPU suite: no reservation, and 6 GB available
python3 tools/ci/local_gpu.py check --min-available-gib 6 && <the CPU suite command>
# Who holds it
python3 tools/ci/local_gpu.py status
```

The local runner sets `FASTMACE_LOCAL_GPU_MARKER` to the same marker by its absolute path, because
`keepalive.sh` moves its `HOME`.

### Keeping the runners up without root

GitHub's `svc.sh install` needs `sudo`, and murphy has no linger for systemd user services. So
`tools/ci/runner/keepalive.sh` keeps each runner up. It starts `run.sh` once per runner directory
(`flock`), in a process group of its own, with `HOME` and `TMPDIR` inside the runner directory.

- **murphy:** cron, open to every user there, runs it at `@reboot` and every 10 minutes
  (`tools/ci/runner/crontab.murphy`). A run while the runner is up exits at once.
- **The PC:** a shortcut in the Windows Startup folder runs it at log-on with `--loop`, through
  `wsl.exe`, minimized. The open `wsl.exe` keeps the WSL VM running. Closing that window stops the
  runner.

The runner updates itself (conductor answer to 2-T18, Q8, 20 October 2026). It was installed at v2.338.0.

### Registering the runners (the owner)

The owner runs these from Git Bash in `D:\Margraf\FastMACE`, on `main` after the merge. Both runners can
be registered at once. The local runner starts a job only when the reservation and its idle rule allow
it, so it never runs beside a worker's suite.

`gh` passes the registration token, valid for one hour, straight into a pipe, so it is never shown or
written down. Registering needs admin rights on the repository, which the `FZWINGEL-UBT` account has
(`GH_TOKEN=$(gh auth token --user FZWINGEL-UBT)`, for that one command). The `git show` copies keep LF
line endings.

murphy:

```bash
R=/mnt/homes/zwingel/actions-runner/murphy
ssh murphy "mkdir -p $R"
for f in keepalive.sh env.murphy crontab.murphy; do git -c core.autocrlf=false show main:tools/ci/runner/$f | ssh murphy "cat > $R/$f"; done
ssh murphy "chmod +x $R/keepalive.sh && cd $R && curl -fsSLo runner.tar.gz https://github.com/actions/runner/releases/download/v2.338.0/actions-runner-linux-x64-2.338.0.tar.gz && echo 'af4b794c1bc41d73d40535e3fe092a39f9679cd8d965954c2aca25a05ca41d32  runner.tar.gz' | sha256sum -c - && tar xzf runner.tar.gz && rm runner.tar.gz"
GH_TOKEN=$(gh auth token --user FZWINGEL-UBT) gh api -X POST repos/FZWINGEL-UBT/FastMACE/actions/runners/registration-token --jq .token | ssh murphy "read -r T && cd $R && ./config.sh --unattended --url https://github.com/FZWINGEL-UBT/FastMACE --token \"\$T\" --name murphy-sm89 --labels fastmace-gpu,sm_89,murphy --work _work --replace"
ssh murphy "cd $R && cat env.murphy >> .env && (crontab -l 2>/dev/null; cat crontab.murphy) | crontab - && setsid -f $R/keepalive.sh $R < /dev/null > /dev/null 2>&1"
```

The local PC (WSL). `MSYS_NO_PATHCONV=1` stops Git Bash from rewriting `/home/...` arguments for
`wsl.exe`:

```bash
export MSYS_NO_PATHCONV=1
R=/home/fzwin/actions-runner/local
wsl -e mkdir -p $R
for f in keepalive.sh env.local; do git -c core.autocrlf=false show main:tools/ci/runner/$f | wsl -e sh -c "cat > $R/$f"; done
wsl -e bash -c "chmod +x $R/keepalive.sh && cd $R && curl -fsSLo runner.tar.gz https://github.com/actions/runner/releases/download/v2.338.0/actions-runner-linux-x64-2.338.0.tar.gz && echo 'af4b794c1bc41d73d40535e3fe092a39f9679cd8d965954c2aca25a05ca41d32  runner.tar.gz' | sha256sum -c - && tar xzf runner.tar.gz && rm runner.tar.gz"
GH_TOKEN=$(gh auth token --user FZWINGEL-UBT) gh api -X POST repos/FZWINGEL-UBT/FastMACE/actions/runners/registration-token --jq .token | wsl -e bash -c "read -r T && cd $R && ./config.sh --unattended --url https://github.com/FZWINGEL-UBT/FastMACE --token \"\$T\" --name local-5060ti-sm120 --labels fastmace-gpu,sm_120,local-5060ti --work _work --replace && cat env.local >> .env"
```

Then, in PowerShell, the Startup shortcut. The second line starts the runner now:

```powershell
$l = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\FastMACE GPU runner.lnk"; $s = (New-Object -ComObject WScript.Shell).CreateShortcut($l); $s.TargetPath = "$env:WINDIR\System32\wsl.exe"; $s.Arguments = "-d Ubuntu --exec /home/fzwin/actions-runner/local/keepalive.sh --loop /home/fzwin/actions-runner/local"; $s.WindowStyle = 7; $s.Save()
Start-Process $l
```

Check that both are online and idle:

```bash
gh api repos/FZWINGEL-UBT/FastMACE/actions/runners --jq '.runners[] | [.name, .status, .busy, ([.labels[].name] | join(","))] | @tsv'
```

### Pausing, removing, cleaning up

- **Pause all GPU CI:** `gh workflow disable gpu.yml` (and `enable` to resume). New jobs are not queued.
- **Pause one runner:** create `PAUSED` in its directory, then stop its process group, for example
  `ssh murphy 'R=/mnt/homes/zwingel/actions-runner/murphy; touch $R/PAUSED; kill -- -$(cat $R/.keepalive.pid)'`.
  Never use a broad `pkill`. While `PAUSED` exists, cron and the shortcut start nothing. Remove it to
  resume. The runner shows as offline. Jobs for it wait in the queue, and fail after 24 hours.
- **Remove a runner:** pause it, then
  `GH_TOKEN=$(gh auth token --user FZWINGEL-UBT) gh api -X POST repos/FZWINGEL-UBT/FastMACE/actions/runners/remove-token --jq .token | ssh murphy 'read -r T && cd /mnt/homes/zwingel/actions-runner/murphy && ./config.sh remove --token "$T"'`
  (through `wsl -e bash -c` for the local one). Then delete the cron lines (`crontab -e`) or the Startup
  shortcut, and the runner directory. GitHub also drops a runner that has been offline for 14 days.
- **Clean up:** each job deletes its own pytest temporary directory. `_work` holds one checkout, and
  the CI cache (`fastmace-cache/ci-murphy`, `~/fastmace-cache/ci-local`) holds compiled kernels. Delete
  either between runs if they grow, and `_diag/*.log` too.

### Security

Self-hosted runners run the workflow's code as the runner's user: on murphy as `zwingel`, with
access to the NAS home. So:

- the repository stays private, with no collaborators beyond the owner;
- `gpu.yml` has no `pull_request` trigger, and only `contents: read`;
- the workflows use no secrets;
- jobs reach these runners only through the `fastmace-gpu` label.

Anyone who can push a branch with a changed workflow can run code on murphy, so push access must stay
with the owner. (GitHub's notes: "Secure use reference", self-hosted runner hardening.)

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

## Modal

Plan task MODAL-T01 (owner, 21 October 2026): GPU jobs on [Modal](https://modal.com) while the
workstations are unreachable.
- **L4 (sm_89):** a stand-in for murphy's RTX 4000 Ada.
- **A100 (sm_80):** a third architecture.
- **H100:** also available.

The code is in `tools/modal/`.

**Client:**
- **Venv:** `~/environments/modal_client` in WSL, Modal 1.6.2; the package list is
  `~/environments/modal_client.requirements.txt`.
- **Authentication:** the owner authenticates once with `modal token new`, and the token lives in
  `~/.modal.toml`. Never print, copy or commit it.
- **Profile:** `fzwingel`.

**Image** (`tools/modal/fastmace_modal.py`):
- **Base:** `debian_slim`, Python 3.10, with `g++`, `git` and `procps`.
- **Packages:** `tools/modal/requirements-hybrid.txt`, which pins every package of
  `~/environments/mace_cueq_oeq.requirements.txt` (recipe C/hybrid: torch 2.14.1+cu130 from the PyTorch
  cu130 index, cuEquivariance 0.12.0, OpenEquivariance 0.7.0, MACE 0.3.16, e3nn 0.4.4, ASE 3.27.0,
  NumPy 2.2.6). It drops the local `nfs-soft-locks` hook and adds pytest 9.1.1 and pytest-xdist 3.8.0.
- **OpenEquivariance:** FastMACE builds its host extension in-process (`fastmace/kernels/_oeq_runtime.py`),
  with `g++`; no `nvcc` and no `oeq_no_nvcc.sh` launcher.
- **Records:** every job records the image's `pip freeze` (`pip-freeze.txt` and its sha256 in `modal.json`).

**Volume** `fastmace-data`, mounted at `/data`:
- `models/`: checkpoints from `~/models`;
- `baseline/1a-T18/`: the research-engine export with the benchmark structures (`FASTMACE_STRUCTURES`);
- `cache/<GPU>/`: FastMACE, Inductor, Triton and OEq caches, one set per GPU type. These are committed after
  every job, so later runs skip compilation.

**Entry points**, run from WSL in the worktree with the client's `modal`:

```bash
modal run tools/modal/fastmace_modal.py::upload --models mace-mh-1.model
```

```bash
modal run tools/modal/fastmace_modal.py::smoke --commit <sha> --gpu L4
```

```bash
modal run tools/modal/fastmace_modal.py::job --gpu L4 --timeout-min 80 --commit <sha> --command "bash tools/modal/opt_t08.sh" --out ~/fastmace-acceptance/modal-<run>
```

- **What a job runs:** `--commit` is `git archive`d from `--repo` (default: the main checkout, which sees
  every pushed branch) with `.source_commit`, like the pinned copies. `--tarball` takes a ready archive
  instead.
- **Where the command runs:** from the source root, with `$OUT` as its results directory. The results come
  back as `<out>/out/`, with `modal.json` holding:
  - the GPU, driver and CUDA;
  - the source commit and the command;
  - the exit code and the wall time;
  - an estimated GPU cost at the list prices in `PRICE_PER_HOUR`. CPU, memory and container start are billed
    on top; the dashboard has the exact figure.
- **Cost control:** every function has a timeout, and the container stops when the call returns. Smoke on the
  L4 first.

**Monthly cap** (MODAL-T02; owner: "on modal make sure to in total not use more than the 30$ credit i get for
free per month"). The guard is in `tools/modal/budget.py`, called by every entry point before it launches
anything:
- **Cap:** $30 per calendar month (Modal's free credit), less a $3 reserve. A launch is refused when
  this month's metered cost + the worst case of jobs still running from this checkout + this job's worst case
  exceeds **$27**. The refusal names the numbers.
- **Metered cost:** read from `modal.Workspace.from_context().billing.summary().metered_cost` (as
  `modal billing summary`). It counts GPU, CPU, memory and container start, so it runs above the GPU-list
  estimates in `modal.json`: $3.65 metered on 22 October 2026 against about $2.15 estimated.
- **Worst case:** (function timeout + 300 s start allowance) × (GPU rate + CPU limit × CPU rate + memory
  limit × memory rate). The rates come from `billing.rates()` (as `modal billing rates`).
  - **The limits are hard:** `run_job` has `cpu=(4, 8)` cores and `memory=(16, 32)` GiB, so the bound holds.
  - **Example:** a 60-minute L4 job has a function timeout of 70 min, and its worst case is about $1.79.
- **Unreadable billing refuses:** this covers a network or permission error, a missing rate and an
  unparsable value.
- **Lower limits:** `--limit-usd 10` or `FASTMACE_MODAL_LIMIT_USD=10`.
- **Higher limits:** need `--owner-override "<who>: <why>"`, and the ledger records it. Use it only on the
  owner's word.
- **Jobs still running:** `results/modal-in-flight.json` holds the worst case of each job until it returns,
  or until its worst case has expired. Parallel launches therefore cannot each fit alone and overrun together.
- **Ledger:** after every launch (also a failed one), `results/modal-ledger.csv` gets a row:
  - the date, commit and GPU;
  - the timeout and wall time;
  - the estimated cost at the container limits, and the worst case;
  - the metered cost before and after, and the delta;
  - the limit, the override and the exit code.

  Modal's metering lags by minutes, so the "after" value may not yet contain the job. The next launch reads
  it fresh.
- **No cost multipliers:** `retries=0` on every function and every `with_options`, and `scaledown_window=2`
  (the container stops 2 s after the call). `tests/cpu/test_modal_budget.py` checks both, along with the
  timeouts and that every `.remote` call is behind the guard.
- **Not covered:** image builds (one per change of `requirements-hybrid.txt`, a few minutes of CPU) and jobs
  started outside `fastmace_modal.py`.

**Measured so far:**

| Run | GPU, driver | Result | Wall time, GPU cost (est.) |
| --- | --- | --- | --- |
| Smoke, `724c118` | NVIDIA L4 (sm_89, 23 GB), driver 610.57.04, CUDA UMD 13.3; torch 2.14.1+cu130 sees capability (8, 9) | `tests/gpu/test_smoke.py`: 2 passed | 31 s, ≈ $0.007 |
| Smoke, `340b23c` | NVIDIA A100-SXM4-40GB (sm_80), driver 610.57.04, CUDA UMD 13.3 | `tests/gpu/test_smoke.py`: 2 passed | 30 s, ≈ $0.017 |
| OPT-T08 (`tools/modal/opt_t08.sh`), `3e5decd`, "Modal L4, not murphy" | L4 as above | force gate passed (NaCl ratios: energy 0.33, forces 1.15, stress 0.39; NbCl5: 0.61, 1.00, 0.85). Same-process A/B, all three records on/off, 12 blocks: NaCl 0.918 (off first) / 0.920 (on first), NbCl5 0.922 / 0.920; medians NaCl 4.69 → 4.31 ms, NbCl5 37.29 → 34.37 ms. Results: `results/modal-opt-T08-L4-3e5decd/` | 38 min, ≈ $0.51 |
| Noise floor, `1176db8`, "Modal L4, not murphy" (`round2.sh` noise-ab/ba: a = `gated_rows=auto`, b = `gated_rows=on`, the same plan) | L4 | b/a NaCl 1.0008 / 1.0019, NbCl5 1.0056 / 0.9979: **floor ≈ 0.2 % NaCl, 0.6 % NbCl5**. NbCl5's block-to-block spread is 8–13 % within a run, and absolute times differ by 4 % between containers (36.0 against 37.4 ms) | 21 min, ≈ $0.28 |
| Round 2 on L4, `1176db8` (main `32a0dd9` + this branch), "Modal L4, not murphy", OPT-T08's records on both sides | L4 | `overlap_rebuild` (OPT2-T04): NaCl 300 K 0.996 / 0.990, NbCl5 0.996 / 1.003, NaCl 1000 K 0.988 / 0.987. `folded_backward_tiles` (OPT2-T03): NaCl 1.005 / 1.000, NbCl5 1.000 / 1.001 | 54 min, ≈ $0.72 |
| OPT-T08 on A100, `1176db8`, "Modal A100, not murphy" (records off against on) | A100-SXM4-40GB | NaCl 0.907 / 0.896 (3.11 → 2.82 ms), NbCl5 0.907 / 0.906 (15.07 → 13.66 ms); runs valid with the PID-1 fix | 18 min, ≈ $0.62 |

**Timing on Modal:**
- **Labelling:** label every result "Modal <GPU>, not murphy".
- **Same-process A/Bs only:** a Modal container's GPU is not shared, but its host is, and nothing guarantees
  the same physical GPU or clocks between calls. Compare only within one container, in the same-process A/B
  (both orders).
- **Noise floor (L4, same-process, 12 blocks):** about 0.2 % on NaCl and 0.6 % on NbCl5 (run `modal-noise-L4-1176db8`).
- **The validity flag:** inside Modal's sandbox `nvidia-smi --query-compute-apps` lists every GPU process as PID 1, the
  container's `dumb-init` (`results/modal-opt-T08-L4-3e5decd/pid-check.txt`). `tools/ab_step_time.gpu_processes()`
  therefore drops PID 1 inside a container (`MODAL_IS_REMOTE`, `MODAL_TASK_ID` or `/.dockerenv`). Runs before that
  fix, such as `modal-opt-T08-L4-3e5decd`, show `INVALID` for this reason only.
