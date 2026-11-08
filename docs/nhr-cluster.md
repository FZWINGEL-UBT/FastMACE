# NHR@FAU cluster (Alex) for coding agents

How a coding agent (Claude Code, Codex) runs FastMACE jobs on the NHR@FAU GPU cluster **Alex**. The facts about
the cluster come from [doc.nhr.fau.de](https://doc.nhr.fau.de/) (read 5 November 2026); check that site when
something here does not match. Repository rules ([AGENTS.md](../AGENTS.md)) still apply: no compute outside a
job, record every result with machine, GPU, environment and commits, and use no `/tmp` for project files.

## 1. Account facts (fill in once)

| Item | Value |
| --- | --- |
| HPC account (portal → User → "Active accounts"; not the SSO name) | `v130cb18` (valid until 1 March 2027; PI J. Margraf) |
| Group (`$GROUP`, part of every path) | `v130cb` (`$HOME` = `/home/hpc/v130cb/v130cb18`) |
| Clusters this account may use | not yet checked. Alex: FAU Tier3 accounts need the [Alex access form](https://hpc.fau.de/tier3-access-to-alex/), NHR projects have it. Helma (H100): separate application for every account (§4) |
| Status (5 November 2026) | csnhr host keys verified against the published list and in `known_hosts`; key loaded in the agent; csnhr still answers `Permission denied (publickey)` (key or account not yet distributed; retry after 2 h or the next day, then `ssh -vv` to support) |
| SSH key registered in the [portal](https://portal.hpc.fau.de/) | `~/.ssh/id_ed25519_nhr_fau` on PC-1QCM1255 (Linux, user `guest2`), MD5 `54:dd:56:1a:3b:01:cd:be:2f:6b:b8:46:1f:3f:fa:00`, 5 November 2026; `~/.ssh/config` has `csnhr`, `alex`, `fritz`, `tinyx`, `woody` |

A new account works from the next day. A new key can take **up to two hours** to reach every system.

## 2. SSH from the agent's machine

Every connection goes through the jump host `csnhr.nhr.fau.de`. Use the full host names (`alex.nhr.fau.de`),
not short names. Allowed key types are ED25519, RSA ≥ 4096 bit and ECDSA ≥ 512 bit. NHR **requires a strong
passphrase** on the key.

An agent cannot type a passphrase, so the user unlocks the key once per login session and the agent reuses that:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_nhr_fau -C "nhr-fau"   # once; passphrase required; upload the .pub
ssh-add ~/.ssh/id_ed25519_nhr_fau                                  # user, once per login session
```

`~/.ssh/config`:

```
Host csnhr.nhr.fau.de csnhr
    HostName csnhr.nhr.fau.de
    User v130cb18
    IdentityFile ~/.ssh/id_ed25519_nhr_fau
    IdentitiesOnly yes
    PasswordAuthentication no
    PreferredAuthentications publickey
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 4h

Host alex.nhr.fau.de alex
    HostName alex.nhr.fau.de
    User v130cb18
    ProxyJump csnhr.nhr.fau.de
    IdentityFile ~/.ssh/id_ed25519_nhr_fau
    IdentitiesOnly yes
    PasswordAuthentication no
    PreferredAuthentications publickey
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 4h
```

`ControlMaster` gives the agent's many short `ssh alex '…'` calls a single authenticated connection. The user
opens it once with `ssh alex true`. On the first connection, compare the host keys with
[the published fingerprints](https://doc.nhr.fau.de/access/ssh-hostkeys/) (csnhr ED25519
`SHA256:ZbsDL9koTZ48+trswAeqvjVGZyv/kBUtOjodEN4XMCo`, alex ED25519 `SHA256:53K9MoZ920hbooWthNUv84ubES6kpxjkVSiy0kcoYc8`,
helma ED25519 `SHA256:NxGWy5akU7leaSxjYpS+bh4f149YTW5YjrRWf7giM+k`). An agent can check them with
`ssh-keyscan <host> | ssh-keygen -lf -` before appending them to `~/.ssh/known_hosts`. An unknown or changed key
means stop and ask the user.

`alex.nhr.fau.de` goes to whichever frontend is free. To reach the frontend that holds a tmux session (§6.2), use
its own name: `ssh alex1` / `ssh alex2` (both are in `~/.ssh/config`; check with `hostname`).

Agent rules for SSH:

- Always run non-interactively: `ssh -o BatchMode=yes alex '<command>'`. If that fails with a key or passphrase
  error, stop and ask the user to run `ssh-add` / `ssh alex true`. Never try other keys or a password.
- Quote remote commands in single quotes, so `$WORK`, `$GROUP` and `$TMPDIR` expand on the cluster.
- Never print or copy private keys or tokens to the cluster.

## 3. Where things live

| Path | Quota | Backup | Use for |
| --- | --- | --- | --- |
| `$HOME` = `/home/hpc/$GROUP/$USER` | 100 GB | yes, snapshots | the FastMACE checkout, job scripts, results to keep |
| `$WORK` (`/home/{woody,saturn,titan,janus,atuin}/…`) | 1 TB (Tier3) or project quota | no | Python environments, model cache (`FASTMACE_CACHE`, `FASTMACE_MODELS`), logs |
| `$HPCVAULT` = `/home/vault/$GROUP/$USER` | 1 TB, **tight file count** | yes | archives (`tar`) only; no environments |
| `$FASTTMP` = `/lustre/$GROUP/$USER` | file count only | no | large parallel files; not small files |
| `$TMPDIR` (node-local NVMe) | shared by the jobs on that node | deleted when the job ends | unpacked environments, scratch |

Check usage with `shownicerquota.pl`. Every file system has an inode quota, and a Python environment is many
small files. Keep one environment in `$WORK`, or pack it (`tar -cjf $WORK/envs/fm.tar.bz2 …`) and unpack it to
`$TMPDIR` at job start.

## 4. Nodes and partitions

| GPU | Nodes × GPUs | Slurm request | CPU cores / host RAM per GPU |
| --- | --- | --- | --- |
| A40 48 GB | 44 × 8 | `--gres=gpu:a40:1` | 16 / 60 GB |
| A100 40 GB | 20 × 8 | `-p a100 --gres=gpu:a100:1` | 16 / 120 GB |
| A100 80 GB | 18 × 8 | `-p a100 --gres=gpu:a100:1 -C a100_80` | 16 / 240 GB |
| RTX Pro 6000 96 GB | 12 × 8 | `--gres=gpu:rtxpro6k:1` (needs `cuda/12.8`+) | 32 / 96 GB |
| A100 MIG slices (10 / 20 GB) | – | `-p a100mig` (`a100small`, `a100med`) | 4 |

**H100s are not on Alex; they are on Helma** (`helma.nhr.fau.de`, same SSH pattern, separate access application,
"experienced users"):

| GPU (Helma) | Nodes × GPUs | Slurm request | CPU cores / host RAM per GPU |
| --- | --- | --- | --- |
| H100 94 GB HBM2e | 4 per node | `-p h100 --gres=gpu:h100:1` (1–4 per job) | 32 / 192 GB |
| H200 141 GB HBM3e | 96 × 4 | `-p h200 --gres=gpu:h200:1` | AI.BAY projects only |
| any free GPU | – | `-p preempt` (the default): 2 h guaranteed, then may be killed with 900 s notice | – |

Helma walltime is at most 24 h as well; its frontends are also for compiling only.

- **Frontends (`alex1`, `alex2`) have no GPU** and are for editing, compiling, `git` and `sbatch` only. Never
  run tests, benchmarks or model loads there.
- Walltime is at most 24 h; without `--time` it is **10 minutes**.
- A single-node job shares the host with other users, but its GPUs and their CPU/RAM share are exclusive. That
  makes it valid for the A/B timing rule (alternating blocks in one process), but not for cross-process
  comparisons.
- The A100-SXM4-80GB is in FastMACE's decision-66 GPU sets. Check which A100 model a node has (`nvidia-smi
  --query-gpu=name --format=csv`) and record it: the per-GPU defaults are keyed on the exact device name.

## 5. Software

```bash
module avail cuda; module avail python      # names and versions differ per cluster
module load python cuda/<version>
export http_proxy=http://proxy.nhr.fau.de:80 https_proxy=http://proxy.nhr.fau.de:80   # pip/git from nodes
```

- `module` changes only the current shell, so every job script loads its modules again. Run `module purge` before
  `salloc`.
- Build the FastMACE GPU environment **inside a job on the target GPU type** (CUDA extensions, OEq, Triton), as
  `$WORK/envs/fastmace_alex` (venv) or a conda env with `pkgs_dirs` / `envs_dirs` moved to
  `$WORK/software/private/conda/`. Record the exact recipe in [development.md](development.md) before relying
  on it.
- Getting the code: `rsync -az --delete --exclude .git --exclude results/ <checkout>/ alex:FastMACE/` from the
  agent's machine. Note the synced commit (`git rev-parse HEAD`) in the job's output. Do not put GitHub tokens on
  the cluster.

## 6. Running jobs (the agent's loop)

### 6.1 Batch jobs

Batch job template (`$HOME/FastMACE/jobs/<run-id>.sh`):

```bash
#!/bin/bash -l
#SBATCH --job-name=fm-<run-id>
#SBATCH -p a100 --gres=gpu:a100:1 -C a100_80
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=%x-%j.out
unset SLURM_EXPORT_ENV
module purge; module load python cuda/<version>
source $WORK/envs/fastmace_alex/bin/activate
export FASTMACE_CACHE=$WORK/fastmace-cache FASTMACE_MODELS=$WORK/models
cd $HOME/FastMACE
nvidia-smi --query-gpu=name,driver_version --format=csv
echo "fastmace commit: <sha>"
PYTHONPATH=$PWD python <script> --out results/<run-id>/
```

| Step | Command (from the agent's machine) |
| --- | --- |
| Submit | `ssh alex 'cd FastMACE && sbatch jobs/<run-id>.sh'` → job ID |
| Wait | `ssh alex 'squeue -j <id> -h -o "%T %M %R"'` (empty once finished); poll every few minutes, not in a tight loop, or use `sbatch --wait` in a background command |
| Inspect live | `ssh alex 'srun --jobid=<id> --overlap nvidia-smi'` |
| Outcome | `ssh alex 'sacct -j <id> -o JobID,State,Elapsed,ExitCode'` and the `.out` file |
| Fetch results | `rsync -az alex:FastMACE/results/<run-id>/ results/<run-id>/` |
| Cancel | `ssh alex 'scancel <id>'` (only your own job IDs; never `scancel -u`) |

- Use `sbatch` for long, unattended benchmarks: they survive a frontend reboot.
- Start with a short test job (`--time=00:10:00`) before long runs, and say up front how many GPU-hours a run
  will use. Compute time on Alex is the project's allocation.
- When a job waits, `squeue`'s last column (`%R`) gives the reason. Report it rather than resubmitting.
- Results follow [AGENTS.md](../AGENTS.md) "Recording results": the machine is `alex <node>` (from
  `$SLURM_JOB_NODELIST`), plus the GPU name, environment, backend, precision, model label and both commits.

### 6.2 Working session: tmux + interactive allocation

For a day of iterating (build, test, short A/Bs), hold one GPU in an interactive allocation inside tmux on a
frontend, and run each command into it. There is no queue wait per run, and caches and compiled kernels stay
warm.

```bash
ssh alex1                                   # a named frontend, so the tmux session can be found again
tmux new -A -s fm                           # attach, or create the session
module purge
salloc -p a100 --gres=gpu:a100:1 -C a100_80 --time=24:00:00   # Helma: -p h100 --gres=gpu:h100:1
echo $SLURM_JOB_ID                           # note it; then detach with Ctrl-b d
```

An agent can open the same session without a terminal:
`ssh alex1 'tmux new -d -s fm "module purge; salloc -p a100 --gres=gpu:a100:1 -C a100_80 --time=24:00:00"'`, and
then get the job ID with `ssh alex1 'squeue --me -h -o "%i %j %T %L"'`.

| Step | Command |
| --- | --- |
| Run a command in the allocation (from any frontend) | `ssh alex 'srun --jobid=<id> --overlap bash -lc "cd FastMACE && <command>"'` |
| Long command, detached | `ssh alex1 'tmux new -d -s run1 "srun --jobid=<id> --overlap bash -lc \"cd FastMACE && <cmd> > results/<run-id>/log 2>&1\""'` |
| Time left | `ssh alex 'squeue -j <id> -h -o %L'` |
| Back in by hand | `ssh alex1`, then `tmux attach -t fm` |
| Release the GPU | `ssh alex 'scancel <id>'` (or `exit` in the tmux pane) |

Rules for working sessions:

- **An allocation uses the project's GPU-hours the whole time, even while the GPU is idle.** Release it when the
  work stops (end of day, waiting on a review); never leave a 24 h allocation idle overnight. NHR monitors GPU
  use.
- The tmux session lives only on the frontend that started it. Reattaching needs that frontend (`alex1` or
  `alex2`), but `srun --jobid` works from either one. If the frontend reboots, the tmux session and the `salloc`
  are lost; a running `sbatch` job is not.
- A/B timing inside the allocation follows the usual rule: alternating blocks in one process.
- Each `srun --overlap` step starts a fresh login shell: load modules and activate the environment in the command
  (or in a `jobs/env.sh` it sources).

### 6.3 Setup time and packed A/Bs

Most of a FastMACE A/B job on Alex goes to setup before the GPU does any work: Dynamo tracing, Inductor
compiling and CUDA-graph capture of each side's engine. Two A/Bs timed out at 1.5–1.75 h. The measurements
below are from 7–8 November 2026, on POLAR-1-M with NaCl and NbCl5 (`tools/ab_rewrites.py`, one side a and one side b
per structure).

What a job gets: 16 cores per A100 (`Cpus_allowed_list` 16 cores, `nproc` 16, SMT off), AMD EPYC 7713 (2.0 GHz
base, 3.7 GHz boost, `schedutil` governor). `torch.get_num_threads()` is 16, and `OMP_NUM_THREADS` is unset.
`sbatch` refuses more than 16 CPUs per GPU ("Too many tasks/CPUs per GPU requested").

Setup is single-threaded Python work. In job 4955075's profile, symbolic-shape tracing alone made 3.6 M `SymNode`
calls. A pure-Python microbenchmark (dict and small-object loops, the same uv-built Python 3.10.12) takes
**1.52 s on an Alex A100 node and 0.38 s on the PC (Core Ultra 7 270K)**, so tracing runs about 4× slower than on
the PC. More cores and thread settings do not help much (table).

SETUP_TABLE

Rules that follow:

- **Pack switches into one process.** `tools/jobs/alex_ab_pack.sh` runs `tools/ab_rewrites.py --pack`. Each
  structure builds side a once and each switch once, so a pack of N switches costs 1 + N builds per structure and
  build order. One process per switch costs 2N. Model loading and imports, about 4 s, are not worth packing across
  models, so the template runs one process per model and order.
- **Stage the environment on the node.** `$WORK` (`/home/atuin`) can get very slow. On the evening of 7 November,
  stat-ing about 100 files took 21 s, and starting Python from the venv took 13 min. Copying the venv file by file
  (54 k files, 6.1 GB) takes about 3.5 min; unpacking one tar of it takes 27–55 s. Make the tar once
  (`tar -cf $WORK/<dir>/env-py310.tar -C $WORK/<dir> uv-python env`) and pass it as `ENV_TAR`.
- **Keep the compile cache warm.** A cold cache costs 160–260 s per build instead of about 150 s, and Dynamo tracing
  is paid either way. Do not copy a large cache to `$TMPDIR`: `fm-offpolar/cache` is 7.8 GB, and copying it
  from `$WORK` did not finish within an hour.
- Profile with `cProfile` only to find where time goes. It inflates setup to about 1.7× (250–270 s per build in
  4955075).

Usage:

```bash
cat > $WORK/<dir>/jobs/<run>.env <<'EOF'
FM_ROOT=$WORK/<dir>                      # env/, uv-python/, models/, baseline/, cache/
SRC=$WORK/<dir>/src-<sha>                # tools/jobs/slim_tar.py output, unpacked
OUT=$WORK/<dir>/runs/<run>-$SLURM_JOB_ID
ENV_TAR=$WORK/<dir>/env-py310.tar
PACK="geo|fastmace.longrange.polar:SHARED_GEOMETRY=False;fold|fastmace.rewrites.polar.update_fold:FOLD_UPDATE=True"
MODELS="polar1-m=MACE-POLAR-1-M.model"
EOF
sbatch --output=$WORK/<dir>/logs/%x-%j.out --time=03:00:00 $SRC/tools/jobs/alex_ab_pack.sh $WORK/<dir>/jobs/<run>.env
```

Plan the wall time as about 150 s per build: 2 orders × structures × (1 + switches) builds, plus the timed blocks.

## 7. Help

`ssh -vv alex` output to hpc-support@fau.de. Docs: [SSH](https://doc.nhr.fau.de/access/ssh-command-line/),
[Alex](https://doc.nhr.fau.de/clusters/alex/), [Slurm](https://doc.nhr.fau.de/batch-processing/batch_system_slurm/),
[file systems](https://doc.nhr.fau.de/data/filesystems/), [Python](https://doc.nhr.fau.de/environment/python-env/).
