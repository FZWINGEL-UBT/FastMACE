# 2-T07: the subprocess runner

Plan task 2-T07 (SPEC §8.7, §10.6; phase 2 acceptance item 4). Code:
`fastmace/engine/subprocess.py`, the hand-off in `fastmace/simulation.py`, the
runner offer in `MinimalPlanner._claim_family`. Tests:
`tests/cpu/test_subprocess_runner.py`, `tests/gpu/test_subprocess_runner_gpu.py`.
Results: `results/2-T07-murphy-20261020/`.

## What it does

The planner marks a plan that a process constraint forbids here
`Plan.runner = "subprocess"`, with a `runner` decision. There are two such cases:
- a compiled MD plan of the other compile family (code `runner-subprocess-family`);
- `reproducible=True` after CUDA was initialised (code
  `runner-subprocess-reproducible`). It is stub-tested only, because rule 5
  refuses every reproducible plan until 2-T16.

One-off evaluation plans still raise `ProcessConstraintError`. Default plans
have no `runner` decision.

`Simulation` then builds no engine and runs no gate in this process. Children
do the work, one per step below:
- **setup:** it builds the `Simulation` from the parent's record and `Atoms`,
  which includes the planner, the gate and the build. Its restart at step 0 is
  the first hand-off;
- **each `run()`:** it resumes the last hand-off (`allow_plan_change=True`),
  applies queued host writes, runs the steps and writes a canonical restart.

Every hand-off is a restart point, so it is also a chunk boundary. The output
chunks continue in the child. Sink values are written by the child and
replayed to the parent's sinks.

The parent's `committed()`, `plan`, `gate`, `schedule`, report and
`metadata.json` come from the child. `metadata.json` also gets a `subprocess`
block: python, reason, directory, children, seconds, failed and last child.

Error handling:
- A failing, killed or stalled child raises `SubprocessRunnerError` with the
  child's message.
- A stall is a heartbeat file that does not change for
  `RunnerConfig.stall_timeout_s`.
- A child that runs longer than `timeout_s` is also stopped.
- A stopped child's process group is killed.
- The hand-off directory is kept after an error and removed on a clean
  `close()`.

## CPU (tiny MH-1-type model, float64)

All of these are bitwise:
- two hand-offs against an in-process run with restart writes at the same steps:
  state, thermo rows, frames and restart points;
- sinks, `submit_frame` and host writes;
- resume through the runner;
- the `fastmace.ase` shim with an observer that moves atoms.

The error tests:
- a failing child;
- a killed child (SIGKILL, exit code −9, its stderr message);
- the stall and total-time limits, tested on a sleeping child;
- the default path never reaches the runner.

Command, local `fastmace_cpu`:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ~/environments/fastmace_cpu/bin/python -m pytest -n 0 tests/cpu/test_subprocess_runner.py
```

Results:
- that file: 10 passed in about 100 s. Each child imports torch and MACE, about
  5 s;
- the CPU suite on murphy (`-n 8`, `mace_cueq_latest`, `CUDA_VISIBLE_DEVICES=`):
  2104 passed, 177 skipped.

## GPU (murphy)

Setup:
- **Machine:** murphy, NVIDIA RTX 4000 Ada Generation 20 GB, driver 595.58.03.
- **Who was on the GPU:** no other user's process. `idle_check.txt` lists one
  process, the conductor's 2-T04 trial run (user zwingel). The run is untimed.
- **Environments:** pytest in `mace_cueq_latest`. The drivers and children ran
  in `mace_cueq_oeq` (`FASTMACE_HYBRID_PYTHON`): torch 2.14.1+cu130,
  cuequivariance-torch 0.12.0, OpenEquivariance 0.7.0, mace-torch 0.3.16.
- **FastMACE:** branch `fastmace/2-T07` at `de86261`. The copy on
  murphy is `acb6ef1` with the three files changed in `de86261` copied over. There is no research-engine baseline.
- **Model and run:** MH-1 (`mh1-r2scan`), NaCl 3×3×3 (216 atoms), Langevin at
  300 K, 2 fs, friction 0.02 per ASE time unit.

| Test | Result |
| --- | --- |
| Plain cuEq float64 after a compiled hybrid: setup plus 3 × `run(100)` in children, against a plain-cuEq run in a fresh process with restart writes at the same steps | max position difference **2.56e-13 Å** (limit 1e-12); thermo rows 0–300 every 10 and 31 frames, each once; restart points 0, 100, 200, 300 on both sides; 4 children; hand-off directory removed. 431 s |
| Hybrid float32 after a compiled plain cuEq | plan from the child: hybrid, float32; chunks complete; finite energies; report names the runner. 297 s |

The setup child took 61 s. Each `run(100)` child took 68–90 s, almost all of
it the child's compile. The runner pays a full setup per `run()`, as the
report's warning says.

The statistical comparison of long hybrid runs (as 1b-T14's 20,000 steps) is
left to 2-T23.

The test is marked `slow` and `acceptance`. Command, on murphy from the copy in
`/mnt/homes/zwingel/fastmace-branches/2-T07`, with the environment block of
`docs/validation/phase-0.md`, plus `FASTMACE_MODELS`,
`FASTMACE_HYBRID_PYTHON=/mnt/homes/zwingel/environments/mace_cueq_oeq/bin/python`
and `TORCHINDUCTOR_COMPILE_THREADS=4`:

```bash
$E/bin/python -m pytest -n 0 -p no:cacheprovider -s -v --basetemp=/mnt/homes/zwingel/fastmace-cache/pytest-2-T07-gpu tests/gpu/test_subprocess_runner_gpu.py
```

## Notes

- The first GPU attempt failed in the test's own driver, not in the library.
  The driver read `torch.cuda.get_device_name()` before the first
  `Simulation`, so CUDA was initialised before the planner set the
  expandable-segments allocator, and the engine's process check refused, as
  designed. The driver now reads the GPU name last.
- Records that other tasks read:
  - 2-T17: the `subprocess` block of `metadata.json` is attached by the runner
    (`SubprocessRunner.record()`) until 2-T17 defines the `metadata_record`
    input.
  - 2-T15: produces plans with `runner="subprocess"` (`planner.minimal._in_child`).
  - 2-T16: owns the GPU case of the reproducible trigger.
