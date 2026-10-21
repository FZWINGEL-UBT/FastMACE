# OPT-T07: a persistent subprocess worker

Plan task OPT-T07 (`plans/opt-1.md`), a follow-up to 2-T07 (SPEC §8.7, §10.6).
This is the design note, written before any code. Branch `fastmace/opt-T07`.

## The problem

Every `run()` of a `Plan.runner == "subprocess"` simulation starts a fresh
child. The child imports torch and MACE, reloads the model, plans, builds and
compiles. It then resumes the last hand-off, runs the steps and exits.

On murphy that cost 68–90 s per `run(100)`, nearly all of it compile (2-T07
report). The `fastmace.ase` shim splits `run()` at every observer step, and
each piece is one child. An observer every 10 steps therefore makes the
runner unusable.

## The idea

The setup child does not exit. It stays as the **worker** for the lifetime of
the `Simulation`, holding its compiled engine and captured graphs, and serves
each later `run()` as a **call**.

Everything else stays as it is:
- the stage directories;
- the restart write at the end of every call;
- the records and the parent's view.

The only thing removed is the rebuild per call.

## Why the trajectory does not change

Today every call ends with `save_restart(stage)`, a canonical restart point
(§10.6: queue drained, chunk closed, lists rebuilt, pipeline floors reset).
The next fresh child resumes that file, and `resume_engine` canonicalises
again.

The worker keeps the same restart write at the end of every call, then simply
goes on with the engine it has.

So the two paths are the two sides of §10.6's own guarantee:

| Today (fresh child per call) | Worker |
| --- | --- |
| interrupted at every restart point and resumed | uninterrupted, with restart writes at the same steps |

§10.6 promises these are equal:
- **bit for bit** where the plan is reproducible. This is tested since 1b-T14
  on CPU float64 (`test_interrupted_equals_uninterrupted_with_the_same_cadence`),
  and for reproducible plain-cuEq MH-1 on the GPU since 2-T16.
- **statistically** otherwise. Plain cuEq float64 matched within 1e-12 Å in
  2-T07, and the hybrid is statistical.

The worker's trajectory is therefore today's wherever today's is defined.
OPT-T07's tests check this directly against the fresh-child path.

The restart write stays at the end of **every** call for three reasons:
- it keeps the restart cadence, and so the trajectory, the same as today;
- it is the recovery point after a failure (below);
- `sim.save_restart` copies it.

Its cost is a canonicalisation, one `.npz` and a few fsyncs: milliseconds,
against the 70 s saved.

## What stays exactly as today

**Checkpoint boundaries and canonical restart points.**
- Each call ends with the same `save_restart(stage)`, so the restart points,
  chunk boundaries, `restart_points` records and the user's `restart_every`
  writes inside a call are unchanged.
- The output writer differs in one detail. Today a child closes the chunk
  that the restart point opened, and the next child's resume reopens it
  (`ChunkedOutput(resume_from=…)`, which deletes later chunks: there are
  none). The worker leaves that chunk open across calls. The files are the
  same: same names, same rows and frames, flushed at the end of each call by
  `Simulation._advance`.

**Host writes and observer mutations between calls.**
- Parent side: no change. `set_positions` and `set_momenta` queue the arrays,
  and the shim's observers call them.
- Today the fresh child resumes (a canonical point) and then applies the
  writes (`_host_write`: quiesce, write, canonicalise and re-evaluate).
- The worker's engine is already at the canonical point of the call's restart
  write, and applies the same writes in the same order.
- The state after the writes is the same, because canonicalising a
  canonicalised state is the identity that §10.6's equality rests on.
- `committed()` after a host write still runs a call, now a cheap one, to get
  the re-evaluated forces.

**Sinks, `submit_frame`, `started`, schedules.** These are the same
request fields as today. The worker drops its previous sinks at the start of
each call and registers the call's intervals, which write into that call's
stage.

**Failure recovery (child crash, OOM, timeout, stall, the fallback path).**
- A call that fails ends the worker:
  - an exception, OOM included: the child writes `status: error` to
    `result.json` and exits with code 1, as today;
  - a crash, such as a signal;
  - a stall (no heartbeat change for `stall_timeout_s`);
  - a call longer than `timeout_s`.
  The parent kills the child's process group.
- After any failure the parent raises `SubprocessRunnerError` with the child's
  message, as today.
- The parent keeps the last good stage (`self.current`); the failed stage
  stays on disk with its path in the record.
- The next `run()` starts a **fresh child**, which resumes the last good
  hand-off. This is today's recovery: the worker never carries state across a
  failure.
- The fallback (2-T15) runs inside the setup child's `Simulation(...)` as
  today. A parent-side fallback that picks `runner="subprocess"`
  (`result.runner`) goes through `_start_runner` → setup, unchanged.

**The subprocess provenance block.** Its keys (`provenance.SUBPROCESS_KEYS`,
2-T17) stay. Their meaning needs one decision; see open question 2.

**Termination and GPU-memory release.** The only way to give a GPU context's
memory back for certain is to end the process, so the worker always ends,
through one of these:
- `close()`: the parent closes the worker's stdin, the child closes its
  `Simulation` writer and exits, and the parent waits up to a grace period
  (30 s) and then kills the process group.
- Garbage collection without `close()`: a `weakref.finalize` on the runner
  does the same without the wait. It holds only the `Popen` object, not the
  runner.
- Parent exit: the same finalizer runs at interpreter exit (finalize is
  atexit-registered).
- Parent killed (SIGKILL, OOM killer):
  - the worker sees EOF on stdin when it is idle;
  - the heartbeat thread sees `os.getppid()` change while a call runs, and
    ends the child with `os._exit`;
  - so the GPU is released within one heartbeat interval.
- The hand-off directory is removed on a clean `close()` and kept after an
  error, as today.

## The protocol

Files stay the medium for state, as in 2-T07. Only the control messages are
new.

**Starting the worker.** It starts as today:
`python -m fastmace.engine.subprocess <stage>`, with the same interpreter,
environment (`PYTHONPATH`, the reproducible variables of 2-T16), process
group and log. Two additions:
- `stdin=PIPE`: the command channel;
- one extra pipe through `pass_fds`, given as `--reply <fd>`: the reply
  channel. stdout and stderr stay the log.

**A call:**
1. The parent writes `request.json` into a new stage directory, as today,
   plus `"worker": true`.
2. The parent writes one line to stdin: `{"stage": "<path>"}`.
3. The worker:
   - reads `request.json`;
   - for the first call (`setup` or a resume), builds or resumes the
     `Simulation`; for a later call, uses the one it holds;
   - applies the host writes, sinks and schedule;
   - runs the steps;
   - makes the restart write into the stage;
   - writes `committed.npz` and `result.json`.
4. The worker writes one reply line, `{"stage": "<path>", "status": "ok"}`,
   and reads the next command.
5. The parent waits on the reply pipe, through a reader thread and a queue,
   while it checks:
   - `process.poll()` (a crash);
   - the heartbeat (a stall);
   - `timeout_s`, timed per call as today.
   It then reads `result.json` and `committed.npz` and takes them as today
   (`_take`, `_replay_sinks`).
6. On `"status": "error"` the worker exits after replying. The parent waits
   for the exit, raises, and starts no further calls on it.

**Shutting down.** EOF on stdin, or the line `{"exit": true}`, makes the
worker close its writer and exit 0.

**Heartbeat.** Its phase reads `idle` between calls. The parent applies the
stall timer only while a call is outstanding.

**Windows.** `pass_fds` is POSIX only. On Windows the runner keeps today's
fresh child per call. CUDA work runs on Linux and WSL, so this costs nothing
in practice.

## When a fresh child is still required

1. **First use:** the setup child becomes the worker. `Simulation.resume`
   gives a child that resumes the given restart, which then becomes the
   worker.
2. **After any failure** of a call, as above.
3. **A worker that is gone:** it exited between calls (killed from outside,
   OOM-killed while idle). The parent sees it at the next call (`poll()`, or
   a broken pipe on the command), records it, and starts a fresh child from
   the last good hand-off.
4. **A changed process:** `os.getpid()` differs from the one that started the
   worker (the parent forked). The child belongs to the original process. The
   copy starts its own fresh child and never writes to the inherited pipe.
5. **A plan revision decided in the parent:** none exists in phase 2.
   Revisions at resume and in the fallback happen inside the child. If a
   later phase revises a running plan from the parent side (§9.6, phase 5),
   that revision ends the worker, and the next call starts a fresh child from
   the hand-off. The runner exposes one method for this (`restart_worker()`).
6. **Opt-out:** with `RunnerConfig.persistent=False` (open question 1), every
   call starts a fresh child, as today.

## Tests (written first)

CPU, tiny model, float64, plans marked `runner="subprocess"` in the parent as
in 2-T07:
- **Repeated short calls:** 10 × `run(1)`, plus a `run(3)`, in one worker.
  - Bitwise against an in-process run with `save_restart` after every call:
    state, thermo rows, frames and restart points.
  - The record shows one child.
  - Each call after the first takes under 2 s on the CPU (a fresh child is
    about 5 s), as a sanity bound, not timing.
- **Against today's path:** the same 4-call sequence with
  `RunnerConfig(persistent=False)` and with the worker. Bitwise equal state,
  rows, frames and restart points. This is CPU float64, a reproducible-style
  case.
- **Observer mutations:** the `fastmace.ase` shim with an observer every 2
  steps that moves atoms and scales momenta, over 10 steps. Bitwise against
  `persistent=False`, with one child.
- **Host writes between calls:** `set_positions`, then `committed()`, then
  `run()`. Forces are re-evaluated, and the result is bitwise equal to the
  fresh-child path.
- **Recovery:** a wrapper interpreter (`RunnerConfig.python`, as in 2-T07)
  makes the worker's second call fail. Three ways are tested: raise, SIGKILL
  and a stall.
  - Each gives `SubprocessRunnerError` with the child's message, and the
    failed stage is kept.
  - The next `run()` starts a fresh child, two in the record, and the
    trajectory equals an in-process run with the same restart points.
- **Termination:** for each of `close()`, `del sim` with `gc.collect()`, and
  the parent's normal exit (the parent run as a subprocess), the worker pid is
  gone within the grace time.
- **Parent killed:** a parent subprocess is killed with SIGKILL while its
  worker is idle, and while a call is running. The worker exits within a
  heartbeat interval or two.
- **Default path:** no runner code is reached, as in 2-T07.

GPU, murphy, untimed, only when the conductor grants it:
- **Reproducible mode (2-T16's plain cuEq MH-1, NaCl, float64):** the plan
  runs in a child after CUDA was initialised in the parent. Three calls with
  the worker against three with `persistent=False`, and the positions must be
  **bitwise** equal. A hybrid → plain-cuEq case within 1e-12 Å, as in 2-T07.
- **Memory release:**
  - During the run, `nvidia-smi --query-compute-apps` lists the worker's pid.
  - After `close()`, and separately after the parent dies, it no longer does.
  - `torch.cuda.mem_get_info` in the parent returns to its value before the
    worker started, within 64 MiB.
- **Speed (recorded, not a gate):** wall time of `run(100)` calls after the
  first, against 2-T07's 68–90 s.

## Files

- `fastmace/engine/subprocess.py` (owner): the worker loop in `main`, the
  command and reply pipes, the parent's wait on a reply, `restart_worker`,
  the finalizer, and the parent-death checks in the heartbeat.
- `fastmace/simulation.py`: no change expected. The runner hooks of 2-T07
  cover it (`close()` already calls `runner.close()`).
- Tests: `tests/cpu/test_subprocess_worker.py` (new), and the 2-T07 tests
  updated where they count children; `tests/gpu/test_subprocess_runner_gpu.py`
  extended.
- This note, and `docs/validation/opt-T07-persistent-worker.md` for the GPU
  results.

## Open questions for the conductor

1. **`RunnerConfig.persistent: bool = True`** (`config.py`, a new field). The
   fresh-child path stays selectable: as the bitwise reference in the tests,
   and as a user's way out (one child per call keeps no GPU memory between
   calls). `config.py` belongs to 2-T01 and then T20. May OPT-T07 add this
   field? The alternative is no option at all, with the fresh-child path kept
   only for Windows and failures. The tests would then need another
   reference, which is weaker.
2. **The provenance block.** Today `children` counts child processes, and one
   is started per call, so it also counts calls. With a worker the two
   differ. My proposal: keep every key, let `children` count processes, and
   add `calls` (calls served) and `persistent` (bool). That changes
   `SUBPROCESS_KEYS`, so `provenance.py` and the metadata schema, which are
   T17's files. The alternative is to keep the keys and let `children` count
   calls: no schema change, but the record would no longer say how many
   processes ran. Which do you want, and who edits `provenance.py`?
3. **Grace time on `close()`:** 30 s before the kill. It is a module
   constant, like 2-T11's wait, and not a config field. Is that acceptable?
