# OPT-T07: the persistent subprocess worker

Plan task OPT-T07 (`plans/opt-1.md`). Design: `docs/design/opt-T07-persistent-worker.md`, approved
by the conductor on 20 October 2026 with three answers:
- `RunnerConfig.persistent` is added;
- the provenance block gains `calls` and `persistent`;
- the grace on `close()` is a 30 s module constant.

Code: `fastmace/engine/subprocess.py`, `fastmace/config.py` (`RunnerConfig.persistent`),
`fastmace/provenance.py` (`SUBPROCESS_KEYS`).

Tests:
- `tests/cpu/test_subprocess_worker.py`, new;
- `tests/cpu/test_subprocess_runner.py`, 2-T07's, now counting calls;
- `tests/gpu/test_subprocess_worker_gpu.py`, new;
- `tests/gpu/test_subprocess_runner_gpu.py`, 2-T07's, now through the worker.

GPU results: `results/opt-T07-murphy-20261021/`.

## What changed

- **One worker for many calls.** A `Plan.runner == "subprocess"` simulation keeps its setup child
  as a worker. The worker serves every `run()` as a call, keeping its compiled engine and graphs.
- **The calls are unchanged.** They are 2-T07's requests in the same stage directories, and each
  still ends with the canonical restart write. The worker is therefore SPEC §10.6's uninterrupted
  run with the same restart points, and is bit for bit equal to a fresh child per call where the
  plan is reproducible.
- **Fresh children remain available.** `RunnerConfig(persistent=False)` restores 2-T07's fresh child
  per call. Windows always uses fresh children, because the reply pipe needs POSIX `pass_fds`.
- **Failures end the worker.** A failed call (an error, a crash, a stall or a timeout) ends the
  worker, and the next `run()` starts a fresh child from the last good hand-off.
- **The worker always ends.** It ends:
  - on `close()`;
  - on garbage collection and at the parent's exit, through `weakref.finalize`;
  - when the parent dies while the worker is idle, through EOF on stdin;
  - when the parent dies while a call runs, through the heartbeat thread's parent check, every 1 s.
- **Metadata.** The `subprocess` block of `metadata.json` gains `calls` and `persistent`, and
  `children` counts processes. The schema stays `fastmace.metadata/1`, because keys were added and
  none changed (`docs/design/metadata-schema.md`).

## CPU

`tests/cpu/test_subprocess_worker.py` uses the tiny MH-1-type model in float64. Its tests:
- 10 × `run(1)` plus a `run(3)` in one worker are bitwise equal to an in-process run with restart
  writes after every call: state, thermo rows, frames and restart points. Calls after the first
  took at most 0.18 s; a fresh child takes about 5 s.
- The worker equals fresh children through a four-call sequence: a schedule, a host write,
  `committed()` after it, a frame at step 0, and sinks.
- The `fastmace.ase` shim with observer mutations every 2 steps gives the same results through the
  worker as through fresh children, with one child against six.
- A raise or a SIGKILL in call 2 ends the worker, and the next `run()` resumes fresh from the last
  hand-off, bitwise. The stall and time limits are tested per call.
- The worker ends on `close()`, on garbage collection, at the parent's exit, and when the parent is
  killed, both while the worker is idle and while it is busy.

Results on murphy's CPUs (`mace_cueq_latest`, `CUDA_VISIBLE_DEVICES=`):
- the new and touched files: 151 passed;
- the full CPU suite: 2520 passed, 180 skipped.

Ruff did not run, because local runs are reserved and murphy's environment has no ruff.

## GPU (murphy)

- **Machine:** RTX 4000 Ada Generation 20 GB.
- **GPU before the run:** empty for 60 s (`idle_check.txt`). OPT-T06's probes had run just before.
  Other sessions' CPU suites ran on the host meanwhile. The run is untimed.
- **Environments:** pytest in `mace_cueq_latest`, drivers and children in `mace_cueq_oeq`.
- **Model and run:** MH-1 (`mh1-r2scan`), NaCl 3×3×3, 2 fs, 300 K.
- **FastMACE:** branch `fastmace/opt-T07`, commit 7df2118 plus the GPU test file.

| Test | Result |
| --- | --- |
| Reproducible plain cuEq float64, run in a child after CUDA was initialised: 3 × `run(100)` through the worker against fresh children | positions and momenta **bitwise equal**; 1 child against 4; worker calls 44.5, 10.2, 10.2 s, fresh-child calls 48.6, 61.5, 58.1 s; the worker is listed by `nvidia-smi` while it runs and gone after `close()`; the parent's free device memory comes back to within 0 MiB |
| Parent killed with SIGKILL while its worker is idle | the worker leaves the GPU within the 30 s limit |
| 2-T07's hybrid → plain cuEq float64 case, now through the worker | 2.17e-13 Å against a fresh process (limit 1e-12); `run(100)` calls 37.8, 9.6, 9.6 s, against 90.3, 70.1, 68.3 s with a fresh child per call (2-T07) |
| 2-T07's plain cuEq → hybrid case through the worker | passes: chunks complete, the hybrid float32 plan reported |

The first call compiles and captures. Later calls cost their steps, plus the restart write and a
few file reads. A `run(100)` drops from about 70 s to about 10 s on this reproducible float64 plan.
The rest of that time is the plan's own step time, which these tests do not measure.

Command, on murphy from `/mnt/homes/zwingel/fastmace-branches/opt-T07` with the environment block
of `docs/validation/phase-0.md`, plus `FASTMACE_MODELS`, `FASTMACE_HYBRID_PYTHON` (`mace_cueq_oeq`)
and `TORCHINDUCTOR_COMPILE_THREADS=4`:

```bash
$E/bin/python -m pytest -n 0 -p no:cacheprovider -s -v --basetemp=/mnt/homes/zwingel/fastmace-cache/pytest-opt-T07-gpu tests/gpu/test_subprocess_worker_gpu.py tests/gpu/test_subprocess_runner_gpu.py
```

The GPU tests are marked `slow` and `acceptance`.
