# Phase 2: safety

Status: **planned**, 19 October 2026. The owner accepted the recommendations
for D1–D4 the same day (SPEC §18 decisions 2, 5, 29 and 30), and all workers
were prestarted. Spec: [SPEC.md](../SPEC.md) §16 row
"2. Safety" and the sections it cites: §6.1, §6.3 and §6.4 (registry,
versions, self-checks), §8.7 (process constraints, the subprocess runner,
instance isolation), §9.3 rule 5 and rules 13–14, §9.4 (fallback), §9.5
(plan revisions), §12.2–§12.4 (test classes, gate statuses and resources,
CI) and §13 (provenance). It also carries the phase 1b follow-ups that the
owner's decisions 22–28 (SPEC §18) and the 1b findings left for phase 2.

Estimate: 4–5 weeks. That is SPEC's 3–4 weeks for safety plus about a week
of carry-overs: the move of `main.py`, the coraline check, the speed screens.
There are 23 tasks, about 45 worker-days, and the critical path is about
14 working days plus reviews. Requires phase 1b merged: done (main
`0414323`), follow-up 1b-T28 merged, and 1b-T27 (the GPU acceptance tier),
which T01 waits for.

"Source" names the FastMACE code a task builds on, or a ResearchModule path
(`G:\Margraf\ResearchModule`, read-only except in the two cross-repository
tasks). Research-engine paths are relative to
`Experiments/Langevin_Thermostat/`.

## Goal

Phase 1b gave FastMACE a planner that refuses with reasons, and a gate that
says when a plan is wrong. Phase 2 makes it **safe when something is
wrong**:

- a failing self-check, gate, compile or capture leads to a validated
  fallback plan, found by diagnosis within a budget, or to a refusal with
  the full record, never to a silently wrong run;
- a process-wide constraint (one backend family per process, reproducible
  or not) is met by running the plan in a child process from a restart
  write, not by refusing;
- an unknown MACE version runs correctly, slower if it must, with every
  unqualified rewrite off and the reason logged;
- the gate's references run where memory allows (GPU float64, then CPU
  float64, then a reduced cell), and the status says which;
- several force fields and engines live in one process without corrupting
  each other;
- `reproducible=True` works for one family (plain cuEq, MH-1), or is
  recorded as not qualifiable with the blocking kernel named;
- GPU CI runs the tiers on two GPU architectures.

On a run where nothing fails, phase 2 changes no number. Fallbacks act only
on failure, the subprocess runner only where a constraint forbids the plan
in-process, and reproducible mode only when asked for. The default plans,
trajectories and step times of 1b stay as they are, which acceptance item 8
checks.

Kept out:
- runtime adaptation between segments (§9.6, phase 5), and so mid-run
  fallbacks (§9.4 step 5): phase 2 falls back before the first step;
- the compiled MD tier (phase 3, owner decision D6 of 1b). The fallback's
  mode dimension is graph → eager → reference in phase 2, and compiled joins
  in phase 3;
- the MC energy-precision dimension of the fallback (phase 3, with MC);
- `tools/port_rewrites.py` (§6.3), which is phase 6 with the first port to
  a new MACE release.

**Acceptance (SPEC §16 row 2, made concrete):**

1. **Unknown MACE version.** In an environment with a MACE other than
   0.3.16 (D1):
   - every rewrite whose pinned source no longer hashes is disabled with its
     reason;
   - the always-on treatments (float64 energy, polar forwards) follow the
     rule of 2-T04;
   - an MD run of MH-1 on NaCl completes on the path the planner reports,
     the reference tier if that is what is left, and passes the gate.
   No source is patched that was not hashed.
2. **Fallback.** Each of these, injected in a test (monkeypatched, never a
   hook in the library), ends in the correct validated fallback within the
   budget, recorded as plan revisions with reasons:
   - a deliberately broken rewrite: ddmin finds it, and it and everything
     that requires it are disabled;
   - a broken pair of interacting rewrites that pass alone: ddmin finds the
     pair;
   - a broken capture: eager passes, so the graph mode is dropped;
   - an exhausted diagnosis budget: the terminal sequence runs (rewrites
     off; eager with rewrites off; reference if it fits), each step
     validated, ending in a passing plan or a refusal with the full record.
   Candidates are built from the pristine model (decision 24), never by
   undoing rewrites in place.
3. **Gate statuses and resources.** On a structure whose GPU float64
   reference does not fit (POLAR-1-L or OMOL-0 XL on NbCl5 3×3×3 on the
   16 GB card, as measured in 1b-T03), the gate tries:
   - CPU float64 within the configured time limit;
   - then a reduced cell, giving `passed-reduced` with the replay checks
     still on the production cell;
   - then `unavailable`, under the rules of decision 3.
   A more memory-hungry reference is never tried after a failure for memory.
4. **Subprocess runner.** A plain-cuEq plan requested after a hybrid model
   was compiled in the process, and a `reproducible=True` plan after CUDA
   was initialised, both run in a child process from a restart write of the
   last committed step:
   - output chunks continue without duplicate or missing frames;
   - the parent's `Simulation` reports the child's plan;
   - the trajectory equals a run started in a fresh process: within 1e-12 Å
     on plain cuEq float64 (1b's bound; that backend is not bitwise even in
     one process), statistical for the hybrid.
5. **Instance isolation.** A → B → A with two crystals passes on the plain
   cuEq, MH-1 hybrid and POLAR-1-S paths: each evaluation equals a fresh
   force field's, and engines interleave correctly. A component that cannot
   be made per-instance holds a lease and raises clearly when its lease was
   taken (§8.7).
6. **Reproducible mode.** For plain cuEq MH-1 on NaCl and NbCl5, float64
   and float32: repeated runs and interrupted-versus-uninterrupted runs are
   bitwise equal on the RTX 5060 Ti and on murphy. If not, the planner keeps
   refusing `reproducible=True` and the record names the blocking kernel,
   which §16 also accepts.
7. **GPU CI on two architectures** (D2): the fast tier on every push to
   `main` and the full tier nightly, on an Ada (sm_89) and a Blackwell
   (sm_120) runner, never sharing a GPU with another user.
8. **No regression.** Default plans and steady-state step times equal phase
   1b's (FastMACE tag `fastmace-1b`, which the conductor sets on `main` once
   1b-T27 is merged): MH-1 NaCl and
   NbCl5, hybrid NVT, same-process A/B, 12 blocks, judged against each
   machine's own noise floor (2-T12). Default trajectories are unchanged: no
   fallback, runner or reproducible code is reached.
9. **Carry-overs:**
   - `main.py --engine fastmace` runs on `Simulation` (decision 28, 2-T21);
   - `Simulation.resume` warns on a version-only change (decision 23,
     2-T09);
   - coraline either passes its soft-lock check and leaves the refused-host
     list, or the check is recorded as still waiting for an idle GPU
     (2-T22);
   - `fastmace doctor` reports stale build locks (2-T11);
   - `fm.calc` no longer recompiles when the edge count changes (2-T13).

## Working principles

These are 1b's principles, unchanged where they still apply:

- **Contracts first.** 2-T01 fixes the new config fields, public names and
  stubs, and like 1b-T01 it is the only task that edits `config.py`. Others
  report the fields they need to the conductor.
- **One owner per file**, listed per task, with second editors named (they
  rebase on the owner's merge and touch only the named part).
  `planner/minimal.py`, `simulation.py` and `rewrites/registry.py` each
  have one owner and a named merge order.
- **Same trajectories.** Every phase 2 task leaves default plans, forces,
  stress and trajectories bitwise unchanged on CPU float64 against main.
  Fallback, subprocess and reproducible paths are tested where they act,
  and proven absent from the default path by a test.
- **`main.py` keeps running.** The research-surface test (1b-T01) keeps
  pinning what ResearchModule's adapter imports. 2-T21 moves the adapter to
  `Simulation`; the old adapter and its imports stay for one phase
  (decision 28).
- **New public API only through the planner.** The eager and reference
  tiers, the subprocess runner and reproducible plans are built by the
  planner only (SPEC §5 layering).
- **Fault injection lives in tests only** (monkeypatched records, kernels
  or captures, as 1b-T16 did): no test hook in the library.
- **General, not tuned.** The fallback's budget, the gate's CPU time limit
  and the reduced-cell rule are general rules (SPEC §9.1), never per-GPU
  or per-crystal tables. Speed screens follow the same rule.
- **Tests first,** as in 1a and 1b. GPU tests are written and run on CPU
  where they can be (skipped without CUDA), in the right tier (1b-T27): fast
  for every change, slow for real checkpoints, `acceptance` for long
  statistical runs, checked by `tools/test_budget.py`.

### Resource rules (learned in 1b)

- **Prestarted workers** (`CONDUCTOR.md`, "Prestarted phase workers"):
  every phase 2 task is started at once, prepares read-only, reports
  "prepared, waiting" with its questions, and starts on the conductor's
  "go". The conductor answers in a "Conductor answers" section. At most
  four workers are active at once, one GPU per task, named in the go
  message and in every follow-up.
- **GPUs.**
  - The local RTX 5060 Ti (WSL, 15 GB RAM):
    - `TORCHINDUCTOR_COMPILE_THREADS=4`;
    - at least 10 GB available before a GPU suite and 6 GB before a CPU
      suite;
    - `-n 2` while siblings run;
    - detached launches with `setsid -f` (WSL ends a background job when its
      call returns);
    - no full suites while a timing run is on.
  - murphy and leto (RTX 4000 Ada 20 GB) only when **no other user's
    process** is on the GPU (owner, 19 October 2026). On 19 October leto
    carried cam's training job all day, and margraf ran FHI-aims CPU jobs on
    murphy, which is fine for untimed GPU work but not for timing.
  - coraline and leeloo belong to others (kaifi); only 2-T22 touches
    coraline, and only when its GPU is idle.
- **Timing** needs an otherwise idle machine, judged against that machine's
  own noise floor. Only murphy has one (0.16 %, 1a-T18), so 2-T12 measures
  the local floor first.
- **Workstation files** only under `/mnt/homes/zwingel`, caches included.
  Each task uses its own copy directory there, never the shared NAS
  checkout, and the conductor deletes the copy after the merge, keeping any
  result that is not in git.
- **Trial merges** run on murphy's CPUs with `set -o pipefail` and `-rf`,
  plus the local `fastmace_cpu` suite when the machine is free. The `fast`
  GPU tier runs on engine merges, `full` on merges that touch `engine/` or
  `planner/`, and `acceptance` on merges that change numerics.
- **NFS and git.** Tests that run git in a temporary directory on the NAS
  pass `safe.directory` in the tests only. The library never overrides it
  (1b-T28).
- **Two engines do not share a process**, and two FastMACE backend families
  do not either (§8.7): references run in fresh processes in the
  acceptance tools, and the in-library gate runs its references first
  (1b-T16).
- **Never broad `pkill`.** Kill only your own process group (1b-T24's
  `pkill -f compile_worker` killed a sibling's suite).

## Tasks

Each task gives: ID, size (S < ½ day, M ≈ 1 day, L 2–4 days),
dependencies, spec sections, source, files owned, tests first, and
done-when. A task whose GPU part is named runs its CPU tests first and then
asks the conductor for that GPU.

### Wave 0: contracts

#### 2-T01 Contracts: config fields, names, stubs (M, conductor-delegated) — after 1b-T27

Spec: §7.6, §9.2–§9.5, §12.3, §8.7. Source: `config.py`, `errors.py`,
`planner/plan.py`, `rewrites/registry.py`, `docs/design/contracts-1b.md`.

- `config.py` (only editor in phase 2):
  - a `FallbackConfig`: `fallback="auto" | "refuse"` (D3), the
    reduced-gate diagnosis budget (default 64 runs, §9.4), and whether the
    reference tier may be the last candidate;
  - gate resource fields: the CPU float64 time limit, the reduced-cell rule
    and the minimum reduced-cell size;
  - `mode=` on the planner inputs (graph, eager, reference);
  - reproducible plumbing: `Workload.reproducible` exists, plus the
    allow-list record's type;
  - the subprocess runner's options (the Python executable, which defaults
    to `sys.executable`, and its environment).
- `errors.py` (second editor after 1b-T02, for new classes only):
  `FallbackExhausted` (subclass of `PlanRefused`, carrying the revision
  chain), `SubprocessRunnerError`, `IsolationError` (lease taken),
  `ReproducibilityError`, all in the §7.7 message shape.
- `Rewrite` record (`rewrites/registry.py`, the contract field only): a
  `deterministic: bool | None` field (§9.3 enforcement), default `None` =
  undeclared.
- Stubs with the signatures later tasks fill: `engine/subprocess.py`
  (`SubprocessRunner`), `engine/reference.py` (`ReferenceForceField`),
  `planner/fallback.py` (`Fallback`, `diagnose`, `ddmin`,
  `terminal_sequence`), `planner/reproducible.py` (allow-list, `qualify`).
- `docs/design/contracts-2.md`: names, fields, the plan-revision record of a
  fallback, the gate record's new fields (`device`, `reduced_cell`), and the
  merge order of the shared files.

Tests first: a contract test pinning the new signatures and fields (as
1b-T01's), and the research-surface test unchanged.

Done when both CPU suites and ruff are green, and the stubs raise
`NotImplementedError` naming their task.

### Wave 1: tiers, registry, gate resources, runner, small carry-overs (parallel after T01)

#### 2-T02 Eager tier for MD (M, GPU: murphy, short) — after T01

Spec: §8.3, §9.3 (mode dimension), §9.4 step 2. Source: `engine/graph.py`
(`capture=False` already runs uncaptured; CPU tests use it), `planner/minimal.py`
(`MODE_REFUSALS["eager"]`).

- `mode="eager"` builds the graph engine without capture and without
  compilation, on the same backend and rewrite set: the configuration the
  fallback's eager diagnosis runs in.
- Same integrators, output, restart and gate. The report states
  "eager tier: no CUDA graphs".

Files: `engine/graph.py` (owner in phase 2; T07 and T08 second editors),
`planner/minimal.py` (the mode-eager part; second editor, merge order
below), new tests.

Tests first: eager equals graph on CPU float64 bitwise over 50 steps (tiny
model). On the GPU, eager equals graph within float32 noise for MH-1 hybrid
NaCl (100 steps). The gate passes in eager mode.

Done when `Simulation(..., mode="eager")` runs and is gated, and it is
recorded in the plan.

#### 2-T03 Reference tier for MD (M, GPU: murphy) — after T01

Spec: §8.4, §9.3 rule 1 (`allow_reference`), §9.4 terminal sequence, §12.1.
Source: `reference/calculators.py` (1b-T13: corrected reference calculators,
including the PolarMACE stress fix and float64 energy), the integrators.

- `engine/reference.py`: `ReferenceForceField`, which evaluates through the
  versioned reference calculator (MACE's `MACECalculator` with the
  record's corrections). It feeds FastMACE's integrators, so integrator,
  noise stream, output and restart are the production ones; only forces
  come from the reference.
- `mode="reference"` and `allow_reference=True` (rule 1). This replaces 1b's
  refusal (D6 of 1b).
- A memory estimate decides whether the reference tier "fits" (§9.4 step
  5): the gate's reference memory rule, without tuning tables.

Files: `engine/reference.py` (new), `reference/calculators.py` (second
editor after 1b-T13), `planner/minimal.py` (the mode-reference part).

Tests first: on a tiny model on CPU float64, a reference-tier trajectory
equals the graph tier with rewrites off within 1e-10 Å over 50 steps. On
the GPU, MH-1 NaCl in the reference tier runs 200 steps, and its
temperature and energy agree with the graph tier statistically.

Done when the planner can build a reference-tier `Simulation` and the
report says so.

#### 2-T04 Versioned registry, always-on treatments and the self-check reaction (L) — after T01

Spec: §6.1, §6.3, §6.4, §9.3 rule 13, acceptance 1. Source:
`rewrites/registry.py` (`qualify` already decides by source hashes and only
mentions the version in the reason), `model/energy.py` (1b-T03: source-hash
pinned and refusing on a changed source), `longrange/polar.py` (PR 1762 and
the polar forwards), `rewrites/engine.py` (`mask_head`, single-graph
strain).

- **Unknown MACE version** (D1): every record whose pinned source does not
  hash is disabled with its reason, and the planner continues.
- **Always-on treatments** cannot simply be switched off. Proposal:
  - if `float64_energy`'s source does not hash, the plan falls back to a
    float64 model on the reference tier (2-T03), whose energy sum is float64
    natively, instead of refusing;
  - PolarMACE without a hashing polar forward is refused with the reason,
    because its stress fix (§12.1) needs the patched code;
  - the decision is recorded, and the conductor confirms the rule at T04's
    "prepared" report.
- **Rule 13:** a self-check failure no longer raises. It disables that
  record and every record that requires it, records the reasons, and
  re-plans (a plan revision).
- The per-model check (§6.4) is recorded in the plan.
- A local test environment with another MACE version
  (`~/environments/fastmace_mace_other`, allowed by ResearchModule's
  AGENTS.md for local WSL environments): the newest mace-torch that installs
  on Python 3.10, or 0.3.15 if none newer does.

Files: `rewrites/registry.py` (owner; T19 second editor), `model/energy.py`
(second editor after 1b-T03), `longrange/polar.py` (the pinned-source part;
second editor), `rewrites/engine.py`.

Tests first:
- CPU: fake versions and fake hashes give the documented decisions per
  record and treatment;
- CPU: an injected self-check failure disables the record and its
  dependants and produces a revision;
- in the other-MACE environment, `fm.md` on a tiny model runs on the
  reported path;
- GPU (local, short): MH-1 NaCl in the other-MACE environment, if it
  installs.

Done when acceptance 1 holds on the local machine.

#### 2-T05 Gate resource policy and statuses (L, GPU: local RTX 5060 Ti for the out-of-memory cases) — after T01

Spec: §12.3 "Status" and "Resources". Source: `planner/selfcheck.py`
(1b-T16: `passed-reduced` reserved, `unavailable` on any reference error),
`reference/calculators.py`.

- References in this order: GPU float64; then, on out-of-memory, CPU
  float64 if the estimated time is within the configured limit (the estimate
  comes from atoms × neighbours × a measured per-edge rate, taken by a short
  calibration in the process, never stored); then a reduced cell.
- **Reduced cell:** the largest supercell of the same primitive cell that is
  smaller than the production cell and at least twice the cutoff in every
  height. It gives `passed-reduced`. The replay and transition checks
  always run on the production cell.
- `unavailable` keeps decision 3's rules. A failure for memory never tries
  a more memory-hungry reference.
- The record holds the device, the reduced cell, the times and each tried
  step.
- The report line `gate: passed-reduced (…)`; `report.py` is a second
  editor for that line only.

Files: `planner/selfcheck.py` (owner), `reference/calculators.py` (the
device part; second editor after T03), `report.py` (the gate line only).

Tests first:
- CPU: a simulated out-of-memory at each step walks the order;
- CPU: the reduced-cell rule on several cells, including non-cubic and
  small ones;
- GPU: POLAR-1-L and OMOL-0 XL on NbCl5 3×3×3 on the 16 GB card end in
  CPU float64 or `passed-reduced` (1b-T03 measured 18.6 and 17.1 GB
  references).

Done when acceptance 3 holds on the RTX 5060 Ti.

#### 2-T06 Fallback groundwork: pristine candidates and validated activation (M) — after T01

Spec: §9.4 steps 1 and 4, §9.5. Source: `planner/plan.py` (`Plan.revise`,
`fallbacks`), `model/forcefield.py` (1b-T12: each plan reloads and converts
the checkpoint from the pristine model, decision 24), `planner/selfcheck.py`.

- `candidate(plan, changes)` builds a plan revision from the pristine
  `ForceField` with one dimension changed: mode, backend, the rewrite set,
  or force precision. Physics never changes.
- `activate(candidate)` runs the self-checks and the gate in the mode the
  candidate will run in, and only then makes it active.
- Every tried candidate is recorded in `Plan.fallbacks` and the revision
  chain.

Files: `planner/fallback.py` (owner, until T15 fills the search),
`planner/plan.py` (second editor after 1b-T12).

Tests first: a candidate never shares modules with its predecessor (the
pristine-copy test of 1b-T12); a failing candidate is never activated; the
records' shapes.

Done when T15 can call `candidate` and `activate`.

#### 2-T07 Subprocess runner through a restart write (L, GPU: murphy) — after T01, T09

Spec: §8.7 (one backend family per process; reproducible processes),
§10.6. Source: `simulation.py` (`save_restart`, `resume`,
`allow_plan_change`), `engine/constraints.py` (`ProcessConstraintError`),
`io/output.py` (chunk continuation).

- `SubprocessRunner` writes a canonical restart of the last committed step,
  starts `python -m fastmace.engine.subprocess <directory>` with the
  configured interpreter and environment, and the child resumes with
  `allow_plan_change=True` and runs the requested steps.
- Output chunks continue in the child. The parent reads the child's restart
  and metadata back, so `sim.committed()`, `sim.plan` and the report reflect
  the child.
- The planner offers the runner where §8.7 forbids an in-process plan:
  - a second backend family after a compiled model;
  - `reproducible=True` after CUDA was initialised.
- The parent `Simulation` drives the child per `run()` call: observers and
  the ASE shim's host writes go through restart points (D4 of 1b).
- A child crash ends the run with the child's error message and the record,
  never a hang (timeout, exit code).

Files: `engine/subprocess.py` (new, owner), `simulation.py` (second editor
after T09, for the runner hand-off only), `engine/constraints.py` (the
note).

Tests first:
- CPU: a plan runs in a child and continues bitwise against an in-process
  run (tiny model, CPU float64, two hand-offs);
- CPU: a killed child gives `SubprocessRunnerError` with the child's
  message;
- GPU: hybrid compiled, then a plain-cuEq plan in the child; chunks without
  duplicate or missing frames; plain cuEq float64 against a fresh-process
  run within 1e-12 Å.

Done when acceptance 4 holds.

#### 2-T08 Instance isolation (M, GPU: murphy) — after T01

Spec: §8.7 "Instance isolation", §12.2 "Isolation", acceptance 5. Source:
`model/forces.py`, `model/compile.py` (`CompiledModelCache` shares compiled
code), `longrange/polar.py` (the per-instance `PolarHolder` and k-space
state, already per instance since 1a), `rewrites/engine.py` (the
process-wide `mask_head` patch).

- An audit: every object shared between two force fields or engines built
  from one compiled model, listed with how it is made per-instance (state
  passed as data) or why it needs a lease.
- The A → B → A tests: two crystals, the plain cuEq, MH-1 hybrid and
  POLAR-1-S paths; evaluations and interleaved engine steps against fresh
  instances.
- Leases with a clear `IsolationError` where needed, and `rebind()`.

Files: tests, the audit in `docs/design/isolation.md`, and narrow fixes in
`model/compile.py`, `model/forces.py` or `longrange/polar.py`. Each fix is
named to the conductor first, because the files are shared.

Done when acceptance 5 holds and the audit lists no unhandled shared state.

#### 2-T09 Resume warns on a version change (S) — after T01

Spec: §10.6 as amended (decision 23). Source: `simulation.py` (`resume`),
`state.py` (`ForceFieldIdentity`).

- The identity comparison splits the versions from the rest. A version-only
  change warns (`UserWarning`, with both versions) and records both in
  `metadata.json`. Every other identity change still raises
  `RestartMismatch`.

Files: `simulation.py` (owner of the `resume` part in phase 2; T07 rebases
on it), `state.py` (the comparison helper).

Tests first: version-only → warning and record; a model, physics or
reference change → `RestartMismatch`; the 1b restart tests unchanged.

#### 2-T10 Durable output and restart files (S) — after T01

Spec: §10.6 (atomic restart), §12.2 "Restart". Source: `io/output.py`
(1b-T06: flushed, not fsynced), `io/restart.py`.

- `fsync` the closed chunk files and the restart file before the rename,
  and the directory after it (POSIX). On NFS the cost is measured; restart
  points are 10,000 steps apart.
- A CPU test that the restart sequence fsyncs in the right order (a fake
  `os.fsync` records the calls).

Files: `io/output.py`, `io/restart.py` (owner of both in phase 2).

#### 2-T11 Stale build locks (S) — after T01

Source: the 1b-T27 finding. A lock left by a killed OEq JIT build
(`~/.cache/torch_extensions/<build>/<name>/lock`) made every process that
imports MACE before FastMACE wait 300 s silently in OEq's
`_wait_for_torch_build_lock`. Files: `cli/doctor.py`,
`kernels/_oeq_runtime.py`.

- `fastmace doctor` lists build locks older than a threshold (an hour)
  with no process holding them, and says how to remove them. It never
  removes them itself, because `doctor` changes nothing.
- FastMACE's OEq preparation checks for such a lock before it builds. It
  warns with the path and the fix, and gives up after a short configurable
  wait instead of 300 s silently. The plan then records "OEq unavailable:
  stale build lock" and falls to plain cuEq with that reason.

Tests first: CPU tests with fake lock files and fake ages; `doctor`'s
no-write test still passes.

#### 2-T12 Noise floor of the local RTX 5060 Ti (S, GPU: local, timing) — after T01

Source: 1a-T18's noise-floor tool and its murphy result (0.16 %, 12
blocks). 1b-T23 judged local A/B results without a floor; per-pair ratios
scattered 0.987–1.014.

- The same protocol on the local machine: an idle desktop, same-process
  A/B of identical sides, 12 blocks, MH-1 NaCl and NbCl5 hybrid.
- The result goes in `docs/validation/` and is used by every later local
  timing (T19, T20, T23).

#### 2-T13 `fm.calc` without a second compile when the edge count changes (M, GPU: local) — after T01

Source: 1b-T18 (second call recompiles: 4.5 s plain cuEq, 19 s hybrid) and
1b-T23 (no second compile for same-size copies, so the trigger is the edge
count). Files: `model/compile.py`, `ase/calculator.py`,
`planner/minimal.py` (the `plan_evaluation` part).

- The evaluation plan compiles with the edge dimension marked dynamic from
  the first call (or padded to the neighbour list's capacity buckets, as the
  graph tier does), so relaxations, phonopy and NEB pay one compile.
- The step time per call is measured before and after, because dynamic
  shapes can cost per-call time.

Tests first: a CPU test counting compiles across calls with changing edge
counts. On the GPU, LBFGS on MH-1 NaCl with `fm.calc` must show one compile,
and its time per call is reported.

Done when the recompile is gone without a per-call slowdown beyond the noise
floor (T12), or the trade-off is reported for the conductor's decision.

#### 2-T14 Find the flaky test (S) — after T01

Source: the T20+T25 trial (1 failure in 1964, not reproduced, test
unknown). Files: none in the library (tests only, if a fix is needed).

- Run the CPU suite 20 times on murphy's CPUs: `-n 8`, `-p randomly` (or a
  rotating `-p no:randomly` seed list) and `-rf`, keeping every junit file.
  Name the test, find the cause (order, timing, temp files on NFS), and fix
  the test.

### Wave 2: fallback, reproducible mode, provenance, CI, setup

#### 2-T15 The fallback algorithm (L, GPU: murphy) — after T02, T03, T04, T05, T06, T07; owner decision D3

Spec: §9.4 in full, §9.3 rules 13–14, §9.5, acceptance 2. Source: T06's
`candidate`/`activate`, T04's rule 13, T05's reduced gate, T02 and T03's
modes, T07's runner (backend changes in a child, §9.4 step 3).

- On a failure (self-check, gate, replay or transition check, compile or
  capture error) before the first step:
  1. Diagnose the dimension. Re-run the failing check on the full rewrite
     set in eager mode. If eager passes, the fault is in capture or
     compilation: try the next mode (graph → eager; compiled joins in
     phase 3). If eager fails, the fault is in the rewrites: ddmin over the
     rewrite set, respecting `requires`, with the reduced gate (one rattled
     copy, the failing quantity only).
  2. Keep the budget of 64 reduced-gate runs, counted, with full gate runs
     counted separately.
  3. When the budget is spent, run the terminal sequence: same mode with
     rewrites off; eager with rewrites off; reference if it fits. Each step
     is validated (T06).
  4. A backend change after a compiled model goes through the runner (T07).
  5. Nothing passes: `FallbackExhausted` with the full record.
- `fallback="refuse"` (D3) turns the search off: a failure raises as in 1b,
  with the record.
- The report shows the revision chain: "plan revision 1 (fallback):
  rewrites X, Y off, reason …".

Files: `planner/fallback.py` (owner after T06), `planner/minimal.py`
(owner in phase 2; T02, T03, T13 and T16 merge their parts first),
`simulation.py` (second editor after T07: the fallback call before the first
step), `report.py` (the revision lines only).

Tests first, all on CPU with tiny models and injected faults:
- ddmin finds a single culprit, a culprit pair, and a culprit that another
  rewrite requires;
- the budget is respected; the terminal sequence runs in order;
- a failing candidate is never activated;
- the pristine copy is used;
- the default path never enters the fallback (no failure → no revision).
GPU: the three injected cases of acceptance 2 on MH-1 NaCl.

Done when acceptance 2 holds.

#### 2-T16 Reproducible mode, plain cuEq MH-1 (L, GPU: local RTX 5060 Ti and murphy) — after T01, T07

Spec: §9.3 rule 5 and "`reproducible=True`", §8.7 "Reproducible
processes", decisions 13 and 14, §18 risk "plain cuEq float64 is not bitwise
reproducible", acceptance 6. Source: `planner/minimal.py` (rule 5 refuses),
`kernels/triton/` (`edge_outer_sum`: the segmented forward uses relaxed
atomics where a segment crosses a 32-edge chunk), the rewrite records
(`deterministic` field from T01).

- A reproducible plan:
  - plain cuEq with the **unfused** convolution (decision 13);
  - hybrid-only rewrites off;
  - `torch.use_deterministic_algorithms(True)` with
    `CUBLAS_WORKSPACE_CONFIG` set before CUDA initialises (§8.7), or the
    runner (T07) when CUDA is already up;
  - a per-component allow-list of every cuEq and FastMACE kernel in use.
- `edge_outer_sum`: a deterministic variant of the segmented forward (one
  program per receiver segment), or the unsegmented form in reproducible
  plans, whichever the A/B shows is cheaper.
- **cuEq's run-time kernel-variant choice** (SPEC §18: the research engine
  differs from itself by up to 3.2e-13 Å over 500 steps): pin the variant
  if cuEq allows it. Otherwise the family is recorded as not qualifiable,
  with the kernel named, and rule 5 keeps refusing with that reason.
- Each rewrite record declares `deterministic`.

Files: `planner/reproducible.py` (owner), `planner/minimal.py` (the rule 5
part; second editor before T15), `kernels/triton/edge_outer_sum*` (the
deterministic variant), `rewrites/*` (the `deterministic` declarations only).

Tests first: CPU tests that the plan's composition matches the rules, and
that a kernel with `deterministic=None` makes the plan refuse. GPU on both
machines: repeated runs (3×500 steps) and interrupted-versus-uninterrupted,
bitwise, float64 and float32, NaCl and NbCl5.

Done when acceptance 6 holds either way (qualified, or not qualifiable
with the kernel named).

#### 2-T17 Provenance completion (S) — after T04, T15

Spec: §13. Source: `provenance.py` (1b-T10 and 1b-T28: the schema exists;
several keys are `null` until phase 2: `plan.fallbacks_tried`, self-check
results, the gate device and reduced cell).

- Fill the phase 2 keys: fallbacks tried with reasons, the revision chain,
  per-record self-check results (or "cached", once T19 lands), the gate's
  device and reduced cell, and the runner's child records.
- Keys for later phases stay explicit `null`.

Files: `provenance.py` (owner in phase 2). Test: the §13 bullet-to-key map
of 1b-T10 extended, with no `null` left for a phase 2 key.

#### 2-T18 GPU CI on two architectures (M) — after T01; owner decision D2

Spec: §12.4, acceptance 7. Source: `.github/workflows/cpu.yml`, the tiers
of 1b-T24/T26/T27, `tools/test_budget.py`.

- Self-hosted GitHub Actions runners as D2 decides. The recommendation: the
  local RTX 5060 Ti in WSL (Blackwell sm_120) and murphy (Ada sm_89).
- A wrapper on the workstation runner checks the GPU before each job: no
  other user's compute process, waiting up to a limit, then the job is
  skipped as "busy". It never runs alongside another user.
- Files and caches only under `/mnt/homes/zwingel`; a user-level service,
  no root.
- `.github/workflows/gpu.yml`:
  - the fast tier on every push to `main`;
  - the full tier nightly;
  - the acceptance tier on `workflow_dispatch`;
  - junit artefacts, checked by `tools/test_budget.py`.
- `docs/development.md`: the runners and how to register, pause and remove
  them.

Files: `.github/workflows/gpu.yml` (new), `tools/ci/` (the idle-check
wrapper), `docs/development.md` (the CI section).

Tests first: a CPU test of the wrapper's idle decision, with
`nvidia-smi`/`ps` output mocked.

Done when both runners have run the fast tier green from a push.

#### 2-T19 Setup time: diagnosis and a qualification cache (M, GPU: local, timing) — after T04, T12

Spec: §12.5 (setup time per stage). Source: 1b-T23 (NaCl setup 4–10 s
longer than 1a: murphy 19.5 against 15.3 s, local 24.0 against 14.4 s;
MACE-first 48 against 34 s), 1a open finding "setup longer" (a cached
qualification keyed by checkpoint hash and software versions, general and
not shape-keyed), and 1b-T28 (the software block is already built once).

- Per-stage setup on 1b against 1a on both machines: load, conversion,
  rewrites, self-checks, compile, capture, gate.
- A qualification cache: self-check results keyed by checkpoint SHA-256,
  the record's source hashes, the rewrite set, the software versions and
  the device kind, under `cache_dir("selfcheck")`. A hit skips the
  numerical check and is recorded as "cached"; any key change runs the
  check. This needs the conductor's OK at "prepared", because it is a
  safety trade-off: self-checks run on random inputs, so their result is a
  function of the key.

Files: `_cache.py` (the `selfcheck` kind), `rewrites/registry.py` (the
cache lookup; second editor after T04).

Done when setup per stage is reported, and the cache (if accepted) gives a
measured saving with its key test.

#### 2-T20 Speed screens (M per screen, GPU: local, timing) — after T12; under the owner's standing rule

Spec: §9.1 (generality), §6.1–§6.4. Source: the 1a/1b screen list (folded
species embedding; `inplace_block_linear` timing), the A/B and gate tools
of 1b-T23 (`tools/`, `fastmace/bench/`).

- One branch per screen, each a rewrite record with `exactness`, self-check
  and `deterministic`. Adopted only if:
  - the 19-label gate passes;
  - the same-process A/B gain is beyond the local noise floor (T12) on MH-1
    and one generic model;
  - the gain holds on murphy when murphy is idle.
- Each screen is recorded in `docs/rewrites.md`, adopted or not (SPEC
  §6.2).
- The conductor gives screens free GPU slots only; they never hold up the
  critical path.

### Wave 3: cross-repository

#### 2-T21 `main.py` on `Simulation` (L, GPU: murphy, then local) — after T01, T09, T10; owner decision D4; ResearchModule worktree by the conductor

Spec: decision 28, §16 "During migration the experiment keeps working".
Source: ResearchModule
`Experiments/Langevin_Thermostat/langevin/fastmace_engine.py` (today
`FixedPlanner` plus the research loop `workflow.run_crystal`; relaxations
and `--quench` use MACE's own calculator through `FixedPlanner.load`, not
`fm.calc`), `langevin/workflow.py` (`run_crystal`, profiles, `run.json`,
`--resume`, `--quench`), the adapter tests
(`tests/core/test_fastmace_engine.py`).

- `--engine fastmace` runs the MD through `Simulation`:
  - the profile becomes a `fastmace.protocols` schedule (1b-T07 matches
    the research engine's per-step targets bitwise);
  - output goes through `Output`;
  - restart and resume go through `Simulation.save_restart`/`resume`;
  - the run report is printed and `metadata.json` written.
- Unchanged:
  - the LBFGS pre-relaxation and `--quench` keep MACE's calculator, the
    scientific protocol;
  - `run.json` keeps every key;
  - `--engine gpu` and the old adapter stay for one phase (decision 28).
- The output layout follows D4.
- The A/B condition of decision 28: ultrafast and anneal profiles, plain
  cuEq float64 bitwise against the old adapter (within 1e-12 Å, 1b's bound
  for that backend), hybrid statistical. The adapter tests' energy tolerance
  follows decision 25.
- ResearchModule's AGENTS.md and docs are updated in the same branch. The
  NAS sync follows ResearchModule's fast-sync recipe.

Files (ResearchModule): `langevin/fastmace_engine.py`, the adapter part of
`langevin/workflow.py`, `tests/core/test_fastmace_engine.py`, AGENTS.md,
`docs/`. FastMACE: none, unless the A/B shows a gap, which is reported to
the conductor.

Done when the A/B passes, `main.py --engine fastmace --profile anneal` runs
end to end on murphy, and `--resume` and `--quench` work on the new
outputs.

#### 2-T22 Coraline soft-lock check (S, conductor-scheduled) — when coraline's GPU is idle

Source: 1a open finding "coraline soft-lock check"; ResearchModule
`workflow.DEFAULT_ENGINE_REFUSED_HOSTS = ("coraline",)`.

- When coraline's GPU has no process of any user: one FastMACE-only control
  run, `tools/acceptance/launch.sh speed --machine coraline --only
  control-mh1-nvt-fastmace` or its 1b successor. Check that it takes zero
  NFS kernel lock requests and that the soft-lock mode is recorded.
- If it passes, remove coraline from the refused hosts in ResearchModule,
  with its AGENTS.md line, in a NVMe worktree.
- If coraline never becomes idle in phase 2, this is recorded and carried
  over. It is not an acceptance blocker (acceptance 9 allows "still
  waiting").

### Wave 4: acceptance

#### 2-T23 Acceptance runs and report (L, GPU, conductor-scheduled) — after all except T20 and T22

- Items 1–9 above on the RTX 5060 Ti and murphy (SPEC §16: "each phase ends
  with its acceptance criteria met on the RTX 5060 Ti and one workstation
  GPU"), with the tools of 1b-T23.
- Reproducible-mode and isolation runs on both machines.
- The no-regression A/B against the tag `fastmace-1b`, judged against each
  machine's own noise floor.
- The full and acceptance GPU tiers green on both runners (T18).
- Report: `docs/validation/phase-2.md`, with pass or fail per item, run IDs,
  hardware, environments and open findings.

## Order and parallelism

```text
Wave 0   T01 contracts
             │
Wave 1   T02 eager   T03 reference   T04 registry + rule 13   T05 gate resources   T06 candidates
         T07 runner (after T09)   T08 isolation   T09 resume warning   T10 fsync   T11 stale locks
         T12 local noise floor   T13 fm.calc dynamic   T14 flaky test
             │
Wave 2   T15 fallback (T02–T07; D3)   T16 reproducible (T07)   T17 provenance (T04, T15)
         T18 GPU CI (D2)   T19 setup (T04, T12)   T20 screens (T12; free slots only)
             │
Wave 3   T21 main.py on Simulation (T09, T10; D4)   T22 coraline (when idle)
Wave 4   T23 acceptance
```

- At most four workers active. Start order in wave 1:
  1. T04, T05 and T07 (the long ones on the critical path), with T09 first
     because T07 rebases on it;
  2. T02, T03 and T06 (T15 needs them);
  3. T08, T12, T13, T10, T11 and T14.
- **Critical path:** T01 → T05 or T07 → T15 → T17 → T23, about 14 working
  days. The second path is T01 → T07 → T16 → T23. Give T15, T16 and T07 the
  most review.
- **Merge order of the shared files:**
  - `planner/minimal.py`: T02, T03, T13, T16, then T15 (owner);
  - `simulation.py`: T09, then T07, then T15;
  - `rewrites/registry.py`: T01 (field), T04 (owner), T19;
  - `engine/graph.py`: T02 (owner), then T07 and T08 if they need it;
  - `reference/calculators.py`: T03, then T05;
  - `report.py`: T05, then T15 (one line each).
- **GPU slots:**
  - local RTX 5060 Ti: T12 first, as timing on an idle machine. Then T05's
    out-of-memory cases (16 GB is the point), T13, T16, T19 and T20 timing;
  - murphy (untimed while margraf's CPU jobs run): T02, T03, T07, T08, T15,
    T16 and T21;
  - leto, whenever cam's job is gone: anything untimed;
  - coraline: T22 only.
- **Prepare, then wait:** T15 (after T02–T07), T16 (after T07), T17, T19,
  T21 and T23.

## Checkpoints for the owner

1. Before the go messages: D1–D4 below.
2. At T19's "prepared": whether the self-check qualification cache is
   acceptable (a safety trade-off; the conductor recommends yes, with the
   key in the provenance).
3. After T23: the acceptance report, and whether the old `main.py` adapter
   (`FixedPlanner` plus the research loop) is removed in phase 3, as
   decision 28 implies.

## Owner decisions

**Decided (user, 19 October 2026): the recommendation of each, (a) in D1–D4.**
They are recorded as SPEC §18 decisions 2 and 5 (D1, D2), 29 (D3) and 30 (D4).
For D2 the owner registers the runners themselves (2-T18 gives the commands).


**D1. Minimum MACE version (SPEC §18 open decision 2).** FastMACE is
qualified on MACE 0.3.16. MACE's develop branch drops Python 3.10, and every
environment here is Python 3.10.12, so a newer MACE also means new
environments. Phase 2's acceptance item 1 needs a rule for "unknown
versions".
- **Options:**
  - (a) 0.3.16 is the minimum and the only qualified version. Other versions
    run through the versioned registry (rewrites whose sources don't hash
    are off; always-on treatments per 2-T04; possibly the reference tier),
    and qualification of a newer release waits until the environments move
    to Python ≥ 3.11, which is a later decision;
  - (b) also support older versions (0.3.14/0.3.15) as "runs, not
    qualified";
  - (c) qualify the newest release now, which needs new environments.
- **Recommendation: (a).** It is what the registry already does. It costs
  nothing in phase 2 and keeps the qualified surface small. (b) adds test
  matrix for versions nobody here runs.

**D2. CI GPU runners (SPEC §18 open decision 5).** SPEC §12.4 wants an Ada
(sm_89) and a Blackwell (sm_120) runner. We have the local RTX 5060 Ti
(Blackwell) and the workstations' RTX 4000 Ada (murphy, leto).
Self-hosted GitHub Actions runners cost no minutes on a private repository.
On a shared workstation the runner must wait while another user's process is
on the GPU, and keep its files under `/mnt/homes/zwingel`.
- **Options:**
  - (a) the local RTX 5060 Ti (WSL) plus murphy, as user-level runners with
    the idle-check wrapper;
  - (b) the local machine only, with workstation runs by hand as today;
  - (c) murphy and leto, without the local machine.
- **Recommendation: (a).** It is the cheapest way to meet §12.4.
- **What it needs from you:**
  - the OK to install the GitHub `actions-runner` under
    `/mnt/homes/zwingel` on murphy, as a user service with no root, if the
    workstation admins allow such services;
  - a runner registration token from the `FZWINGEL-UBT` account. Never
    paste it into a chat or file I can see: you run the registration command
    yourself, and I give you the command.
  The local runner runs only while the PC is on.

**D3. Fallback on failure: automatic or refuse?** SPEC §9.4 makes the
fallback automatic: a failing gate or self-check leads to a diagnosed,
validated, slower plan, recorded and reported. In 1b, a failing gate raises
`GateFailed`.
- **What changes:** with automatic fallback, POLAR-1-S NPT with
  `check="gate"` (decision 26) would no longer refuse. It would run on the
  first fallback that passes, likely with some polar or linear rewrites off,
  and the report would say so.
- **Options:**
  - (a) automatic by default, with `fallback="refuse"` to opt out;
  - (b) refuse by default, with `fallback="auto"` to opt in.
- **Recommendation: (a), as SPEC says.** That is the point of phase 2, and a
  fallback is never silent: plan revision, report line, metadata.
  Decision 26 then means "the default fast plan fails there and the fallback
  finds a slower correct one", which is better than refusing.

**D4. `main.py`'s output layout after the move to `Simulation` (2-T21).**
`Simulation` writes crash-safe chunks (`thermo.<n>.csv`,
`trajectory.<n>.traj`, 1b-T06). `main.py`'s research loop writes one
`md.traj` and one thermo log, which `--quench` and the analysis scripts
(`studies/annealing`) read.
- **Options:**
  - (a) chunks plus a single-file export at the end: chunks while running,
    then `md.traj` and the thermo log assembled once the run completes or
    stops cleanly. `--quench` and the scripts read either, through
    `fastmace.io.output.read_frames`/`read_thermo`;
  - (b) single files only, as today, giving up the chunk crash-safety;
  - (c) chunks only, moving every script to the chunk readers.
- **Recommendation: (a).** Crash-safety while running, and nothing
  downstream changes. The assembled file costs one extra copy of the
  trajectory per run, up to about 770 MB for the longest anneal on the NAS.
  `md.traj` is ignored in git anyway.

## Conductor answers to the prestart reports (20 October 2026)

All 23 workers reported, and 22 had questions. These answers are binding, and
each go message points here. The workers' longer notes are in the shared
notes folder of the conductor's session (`2-T0x-…md`).

### Changes to the plan

- **2-T01 grows (all four reports needing it agree).** `Simulation` and
  `fm.md` cannot take the planner's inputs today: no `mode=` (T02, T03),
  and no engine or neighbour options (T21's blocker G1: `main.py
  --backend cueq` cannot reach `Simulation`, so decision 28's bitwise A/B
  cannot run). T01 adds the contract and the plumbing:
  - `Simulation(...)` and `fm.md(...)` take `mode=`, `allow_reference=`,
    `fallback=` (`FallbackConfig`), `gate=` (`GateConfig`), `runner=`
    (`RunnerConfig`), `engine=` (`EngineConfig`) and `neighbours=`
    (`NeighbourConfig`), and pass them to the planner;
  - `simulation.json` records them, and `Simulation.resume` re-plans with
    them;
  - `test_simulation_signature` is updated.
  So T01 is the first editor of `simulation.py` and `quick.py` in phase 2.
- **New fields in T01:**
  - `EngineConfig.dynamic_edges: bool = False` (for T13);
  - `RewriteConfig.selfcheck_cache: Literal["use", "off", "refresh"] =
    "use"` (for T19; inert until T19 lands);
  - `RunnerConfig.stall_timeout_s = 1800` beside `timeout_s = None` (T07:
    no total limit, but a child without progress for 30 minutes is killed);
  - a `Plan.runner: Literal["in-process", "subprocess"] = "in-process"`
    field in `planner/plan.py` (T07), as a contract before T06's edits;
  - stubs `candidate` and `activate` in `planner/fallback.py` (T06 fills
    them).
  T01's own proposals are accepted:
  - the `Mode` type;
  - the default `cpu_float64_time_limit_s = 300`;
  - `Rewrite.from_dict` not requiring the new field;
  - the four error classes;
  - the allow-list record in `planner/reproducible.py`, not `config.py`;
  - `ReproducibilityError` for a failed qualification only, while rule 5
    refusals stay `PlanRefused`.
  `GateRecord`'s new fields are documented by T01 and added by T05.
- **Merge orders for the shared files**, replacing the list in "Order and
  parallelism":

  | File | Editors, in merge order |
  | --- | --- |
  | `simulation.py` | T01 (options), T09 (resume), T10 (`_write_json` only), T07 (runner), T15 (fallback call) |
  | `planner/minimal.py` | T02, T03, T07 (`_claim_family` only), T13, T16, then T15 (owner) |
  | `planner/fixed.py` | T02 (a `capture=` keyword on `build`), T16 (reproducible overrides) |
  | `model/forces.py` | T04 (call sites of `registry.apply` and the float64 energy), T13 (`dynamic_edges` and the cache key), T08 (`rebind`, only if a lease is needed) |
  | `rewrites/registry.py` | T01 (field), T04 (owner), T19 (cache lookup), T20 (screen records) |
  | `config.py` | T01 (owner), then T20 for its screens' fields only |
  | `provenance.py` | T10 (the durable write in `write_metadata` only), T17 (owner) |
  | `reference/calculators.py` | T03, T05 |
  | `reference/records.py` | T04 (the unknown-version record) |
  | `model/forcefield.py` | T04 (identity from the effective precision) |
  | `_patches.py` | T04 (owner: hash gates) |
  | `report.py` | T05 (gate line), T15 (revision lines) |

- **T21 waits for T01, T09 and T10**, not T09 and T10 alone.

### Shared decisions

- **Fallback under the default `check="skip"`** (T06 Q2, T15 Q3). The
  fallback acts on failures that raise: build, compile, capture and
  self-check. On that failure path, a candidate is validated by its
  self-checks **and a reduced gate** (one rattled copy). That runs even under
  `check="skip"`, because §9.4 step 4 requires validation before activation.
  The references are computed once, in a child process if a production
  engine exists, reused for every candidate (the physics is fixed), and the
  run is recorded. ddmin runs only when a gate fails: under `check="gate"`,
  or when that validation fails. The default path, with no failure, never
  computes references.
- **Search dimensions in phase 2** (T06 Q3, T15 Q4): mode (graph → eager →
  reference) and the rewrite set, which is the registry records only, not
  engine options (T15 Q6). Backend changes only in §9.4 step 3's case
  (eager in-process, or `Plan.runner="subprocess"`). Precision is not a
  search dimension.
- **Precision promotion on an unknown MACE** (T04 Q1, T06 Q3). When
  `float64_energy`'s source does not hash, the plan promotes the force
  precision to float64, with a revision and a report line, even if the user
  asked for float32. The run stays correct and gets slower, as acceptance 1
  says. The force-field identity takes the **effective** precision (T04
  edits `model/forcefield.py`), so a resume re-plans the same way and the
  identity matches.
- **Always-on treatments on an unknown MACE** (T04): T04's proposal is
  accepted in full.
  - `float64_energy` runs a native float64 model on the normal tiers, with
    the reference tier as the last resort.
  - PolarMACE is refused without its hashed forward.
  - An unhashed `capture_safe_mask_head` means no capture, so the eager
    tier.
  - Unhashed `first_derivative_only` and `single_graph_strain` are not
    installed.
  - Every patch in `_patches.py` gets a hash gate, including
    `set_cueq_conv_fusion`. T04 owns `_patches.py`.
  - The other environment holds mace-torch 0.3.15, because PyPI has nothing
    newer.
- **Unknown MACE in the reference policy** (T03 Q1): T04 adds an
  "unqualified stock" record for versions outside 0.3.16: stock equivalence
  only; PolarMACE refused. T04 also adds a `content_digest()` without the
  version, for T09.
- **Exceptions the fallback catches** (T15 Q7). Classified by the
  **stage** where they were raised (build, compile, capture, self-check,
  gate), never by type alone. The original exception is always recorded.
  Out-of-memory and non-finite values are not fallback triggers in phase 2,
  and propagate.
- **Restart records** (T06 Q4, T15 Q8): `PLAN_RECORD_VERSION` 2 encodes
  `fallbacks`, and version 1 records still read (T06). No fallback at resume:
  resume rebuilds the stored revision, and a failure there raises (T15 Q9).
- **The resume version rule** (T09 Q2). Decision 23 is read so that an
  upgrade can warn instead of refusing. The reference record's id contains
  the MACE version, so T09 compares the record's `content_digest()`: same
  corrections and only the version differs → warning; different corrections
  → `RestartMismatch`. The versions are recorded through a plan revision and
  the report's warning, as T09 proposed; T17 may add a structured key.
- **Provenance schema** (T17 Q2): T17 bumps it to `fastmace.metadata/2` with
  a schema file and a validation test, and readers accept `/1`. Inputs for
  new blocks come as `metadata_record(...)` arguments that T17 defines, the
  child block among them (T07, T17 Q3). Reproducible data comes through
  `Plan` (T16 → T17 Q4). Per-record self-check results move to
  `plan.selfchecks` (T04 fills, T17 maps), not `gate.selfchecks` (T17 Q5).
  `counters.tier_switches` stays `null` and its text says "phase 5" (T17 Q1).
- **Durable writes** (T10 Q1, Q3): T10 adds `fastmace/io/durable.py`
  (`atomic_write(path, data)`: temporary file, fsync, rename, directory
  fsync). `simulation._write_json` and `provenance.write_metadata` call it
  (T10 edits exactly those two functions). T21's assembled files use it.
  The directory fsync is skipped on `EINVAL`, `ENOTSUP` and `EOPNOTSUPP`,
  and any other error raises (T10 Q2). It is always on, with no switch (T10
  Q4).
- **Speed screens land "off"** (T20 Q2). A passing screen is merged with
  default "off". Turning it on (`auto`) is the owner's decision after
  murphy confirms the gain, so acceptance 8 (default plans unchanged) holds.
  T20 may create `docs/rewrites.md` with the research engine's rejected
  screens (SPEC §6.2).
- **Hybrid on murphy** (T18 Q6). Allowed: ResearchModule's AGENTS.md gives
  murphy the same setup as leto, and the hybrid default passed the
  workstation check there on 15 October 2026.

### Per task

- **T02.**
  - Use `tier="eager"`, built through a `capture=` keyword on
    `FixedPlanner.build` (T02 edits `fixed.py` for that).
  - Rule 4 is lifted for `mode="eager"`, which uses exact edge counts (§9.3
    rule 4's "exact edge counts"), and only there.
  - `pipelined` keeps its resolved value, so CPU eager equals graph.
  - PolarMACE on plain cuEq in eager mode: a short POLAR-1-S check on murphy
    belongs to T02.
- **T03.**
  - The design is accepted: `ReferenceForceField` on `GraphEngine` with
    `capture=False` and `pad_edges=False`; no rewrites; and "fits" decided
    by the gate's reference result, or by one trial evaluation that fails on
    out-of-memory.
  - The gate runs as usual in reference mode.
  - On an unknown MACE whose `float64_energy` does not hash, the tier is
    float64 only.
  - `records.py` belongs to T04 (above).
- **T04.**
  - The split with T15 is accepted: T04 provides
    `Registry.disable_with_dependants(...)` and the reaction, T15 calls them,
    and T04 does not touch `minimal.py`.
  - Test against T03's stub if T03 is not merged.
  - Keep the self-check call in one place for T19's seam.
- **T05.**
  - The ladder, the CPU estimate (with an atoms term for PolarMACE, Q5) and
    the reduced-cell rule are accepted.
  - The order is GPU on the production cell; then CPU on the production cell
    if within the time limit; then GPU on the reduced cell; then CPU on the
    reduced cell (Q6). Memory never grows.
  - On the reduced cell, production is evaluated eagerly by a separate force
    field from the pristine model, not by a second engine (Q1).
  - Under `passed-reduced`, the replay check on the production cell compares
    the captured replay with a direct evaluation of the same force field at
    the same state (Q2).
  - The CPU step checks `MemAvailable` against a calibrated bytes-per-edge
    figure and runs in a child process, so a kill becomes a recorded failure
    (Q3).
  - Both roles use the same device at each step (Q4).
  - T05 parameterises the copy count and the quantities, so T15's reduced
    gate (one copy, the failing quantity) is a call (T15 Q1).
- **T06.**
  - Your shapes for `candidate` and `activate` stand.
  - A `ProcessConstraintError` propagates to T15, which routes it to the
    runner (Q5).
- **T07.**
  - The hand-off design is accepted.
  - The directory is `<output>/.fastmace-handoff/`, or under
    `FASTMACE_CACHE` without output. It is deleted on a clean `close()` and
    kept, with its path recorded, after an error (Q3).
  - The parent's report is rebuilt from the child's plan, with no
    `report.py` edit (Q4).
  - Stub-test the reproducible trigger; T16 adds the GPU case (Q5).
  - The killed-child test uses a wrapper interpreter through
    `RunnerConfig.python` (Q6).
- **T08.**
  - `rebind()` exists on `MACEForceField` (state) and `GraphEngine` (the
    force field's `rebind` plus clearing the graphs), but only where a lease
    turns out to be needed (Q1).
  - The polar `dynamo.reset()` recompile is acceptable as "correct, slower".
    The test counts the recompiles, and a wrong result after it means a
    lease (Q3).
  - Polar A → B → A on the GPU only (Q4). T01 is merged before your go (Q2).
- **T11.**
  - The wait is a keyword argument with a module-constant default, with no
    config field (Q1).
  - `import fastmace` stays quiet: `doctor` reports default-root locks (Q2).
  - The growing per-process extension directories are a finding for later,
    not this task.
- **T12.**
  - Both: identical FastMACE sides, plus one baseline-against-baseline
    process (Q1).
  - NaCl and NbCl5 only (Q2).
  - The conductor gives the go only when nothing else runs locally (Q3).
  - The floor applies to every same-process A/B tool, and the report says so
    (Q4).
- **T13.**
  - Design A, through `EngineConfig.dynamic_edges` (T01), plus a narrow edit
    in `forces.py` (Q1). T13 merges before T08 there (Q2).
  - Judge per-call time against T12's floor. T12 runs first (Q3).
- **T14.**
  - Your own cache root under `/mnt/homes/zwingel/fastmace-2-T14/`. Runs
    1–4 start cold, from a fresh cache each, to provoke your top suspect
    (the OEq first-build race); the rest are warm.
  - Current `main` at go.
  - The order plugin outside the repository is fine.
  - Ask before each batch.
- **T15.**
  - The eager diagnosis and the empty-set test count toward the 64 (Q2).
  - The fallback may switch off a rewrite the user set to "on", recorded
    (Q5). The rest of your sketch stands.
- **T16.**
  - `CUBLAS_WORKSPACE_CONFIG`, `CUEQUIVARIANCE_OPS_U1D_ALGOS` and
    `TORCHINDUCTOR_DETERMINISTIC` are set together before CUDA starts, and
    the child gets them through the runner (Q1).
  - The undocumented cuEq variable is acceptable for qualification only:
    - pinned to the `cuequivariance_ops` version in the allow-list;
    - with a plan-time check that two evaluations are bitwise equal;
    - recorded in the provenance;
    - named in the report as relying on an undocumented internal.
    On any other version the family is not qualified (Q2).
  - `segmented_first_sum` gets `deterministic=False` (Q3).
  - Three fresh processes × 500 steps, plus an interrupted run resumed in a
    fresh process (Q4).
  - Use the unsegmented `edge_outer_sum`, and no deterministic segmented
    variant.
  - T16 may edit `fixed.py`'s reproducible overrides, after T02, if
    `minimal.py` cannot do it cleanly.
  - A reproducible plan has its own compile-cache key.
- **T18.**
  - The design is accepted. Wait for 1b-T27 (Q1).
  - Wait for zwingel's own GPU processes too (Q2).
  - Another user mid-run: record only (Q3).
  - Budget failures fail on the local runner and are informational on
    murphy (Q4).
  - A temporary push trigger on `fastmace/2-T18`, removed before the merge
    (Q5).
  - Read-only `loginctl`/`crontab -l` on murphy is OK (Q7).
  - Automatic runner updates (Q8).
  - The full tier runs nightly on murphy, and on the local runner by
    dispatch only, because the PC is not always on or idle (Q9). This
    adjusts acceptance 7.
  - The owner runs the registration commands.
- **T19.**
  - Measure first. The cache is built only if the measurement shows that
    the self-checks are a material part of the gap. The owner's checkpoint 2
    is taken with that number.
  - Structure-dependent checks stay uncached unless shown species-set
    invariant (Q2).
  - Cache only passes with a margin of threshold/10 or better (Q3).
  - A gap found elsewhere (for example the per-plan reload of decision 24)
    is reported with numbers, not fixed in T19 (Q4).
- **T20.**
  - Write `tools/ab_rewrites.py` (option off against on in one process) or
    extend T12's tool, whichever is smaller (Q1).
  - The 19-label gate on murphy, untimed (Q5).
  - Models (Q6):
    - T20a: MP-0 medium and MPA-0 medium;
    - T20b and T20c: MH-1 and MP-0 medium.
  - The order is 20a, then 20b; 20c is a timing-only check; 20d starts with
    the op map.
- **T21.**
  - G1 goes to T01.
  - G2: `resume` reloads from the recorded source (path and SHA-256), not
    the label alone. That belongs to T09.
  - G3: segments of 1,000 steps with a stop check between them; no library
    change.
  - G4: refuse custom phase ends that are not multiples of the intervals.
  - G5: check the friction round trip first. If it is not bitwise, report it;
    `IntegratorConfig` takes ASE-unit friction.
  - G6: the plan-equality test, yes.
  - Q1: `--fastmace-api simulation|adapter`, default `simulation`. Runs
    started before this task resume on `adapter`. `run.json` gains a key
    and loses none.
  - Q3: delete `simulation/restart.npz` on completion, as the old path does.
  - Q4: the A/B is `ultrafast` for NaCl and NbCl5 plus a full `anneal` of
    NaCl in plain cuEq float64 on murphy (hours are fine), and hybrid
    statistics on NaCl and NbCl5 `ultrafast`, two seeds each.
- **T22.**
  - The read-only lock-mode wrapper and `nfsstat` on coraline are OK at any
    time. The run itself only with the GPU idle (Q1).
  - The conductor syncs the NAS checkout before your run (Q2) and makes the
    ResearchModule change.
- **T23.**
  - The tag `fastmace-1b` goes on main once 1b-T27 is merged (Q1).
  - Item 1 runs locally only (Q2).
  - The hybrid statistics use 1b-T14's measure (Q3).
  - If T16 is "not qualifiable", record the refusal and show the runner on
    the backend case (Q4).
  - Item 8's criteria are as you proposed (Q5).
  - Dispatch runs with `gh`, using the `FZWINGEL-UBT` token for that one
    command and never printing it (Q6).
  - leto for untimed work when no other user's process is there (Q7).
  - `fastmace/bench/versions.py` is yours to edit (Q8).
  - Sanity reruns: the in-library gate for MH-1 and POLAR-1-S, or the
    19-label gate after a merge that changed numerics (Q9).
  - Judge each machine against its own noise floor, as decision 27 did (Q10).

### During wave 1 (20 October 2026)

- **2-T11 merged** (trial on murphy: 2167 passed).
- **1b-T27 held.** Its tier hook raised `UsageError` during collection when
  acceptance tests were collected without their environment variables. That
  crashed every xdist worker, and xdist blamed each worker's first test.
  The fix (c2ac3b8: the check runs in the controller only, and only when
  `-m` selects acceptance tests) is accepted.
- **The flaky test, found.** 1b-T27 found an order-dependent failure that
  `main` has too:
  - running `test_acceptance_1b.py::test_two_simulation_sides_on_the_cpu`
    and then `test_forces.py::test_polar_force_field_enables_its_own_copy`
    in one process makes the polar test fail (5.4e-12 against 1e-12);
  - the polar patch that stays applied after it then fails three
    `test_loading` tests in the same worker.
  2-T14 takes the reproducer and finds the leaked process state: a library
  fix if a `Simulation` or force field leaks state (an isolation defect), a
  test fix if the bound wrongly assumes first-in-process rounding. 1b-T27
  merges after that fix, because with it the `-n 8` suite fails in about 3 of
  5 runs.

## Phase 1b items carried into phase 2

| Item (where found or decided) | Phase 2 task |
| --- | --- |
| `main.py` on `Simulation` (decision 28) | T21 |
| Resume warns on a version-only change (decision 23) | T09 |
| Coraline soft-lock check (1a open finding) | T22 |
| Stale torch-extension build lock costs 300 s (1b-T27) | T11 |
| `fm.calc` recompiles on edge-count change (1b-T18, T23) | T13 |
| Setup 4–10 s longer than 1a; MACE-first 48 against 34 s (1b-T23); cached qualification (1a open finding) | T19 |
| No local noise floor (1b-T23) | T12 |
| Speed screens: folded species embedding, `inplace_block_linear` timing (1a, 1b) | T20 |
| Chunks and restart flushed, not fsynced (1b-T06) | T10 |
| One unidentified flaky CPU test (T20+T25 trial) | T14 |
| `Simulation.run` rebuilt metadata per call (1b-T21b) | done in 1b-T28 |
| Full GPU tier 142 min (1b-T23) | done in 1b-T27 |
| PolarMACE hybrid stress precision; NbCl5 stress against finite differences (decision 26) | phase 3b, not phase 2 |
| Mid-run fallbacks and runtime adaptation (§9.4 step 5, §9.6) | phase 5 |
| `tools/port_rewrites.py` (§6.3) | phase 6, with the first port |

## Risks specific to phase 2

| Risk | Mitigation |
| --- | --- |
| The fallback search is slow or loops | the budget of 64 reduced-gate runs, counted; the terminal sequence; tests for each stop condition (T15) |
| A fallback hides a real defect behind a slower plan | every revision is recorded, printed and kept in the metadata; injected-fault tests; `fallback="refuse"` (D3) |
| The subprocess runner hangs, loses frames, or diverges from an in-process run | restart-write hand-off only (no pickled live objects); a timeout; bitwise and chunk tests (T07) |
| Reproducible mode is not qualifiable (cuEq's run-time kernel choice) | allowed by §16: the record names the kernel and rule 5 keeps refusing (T16) |
| An unknown MACE version silently loses a correction (polar stress, float64 energy) | always-on treatments are decided explicitly (T04), never dropped; PolarMACE refuses without its hashed forward |
| The reduced-cell gate passes a structure-dependent error | `passed-reduced` is labelled; replay and transition checks stay on the production cell (§12.3) |
| GPU contention: leto busy with cam, murphy loaded by margraf, coraline by kaifi | the local GPU for timing; murphy for untimed GPU work; idle checks before every job; T22 not a blocker |
| The `main.py` move changes results the owner relies on | decision 28's A/B; `run.json` unchanged; the old adapter kept for one phase; the scientific protocol unchanged (relaxation and quench on MACE's calculator) |
| A self-hosted runner on a shared workstation disturbs other users | the idle-check wrapper; user-level service; files under `/mnt/homes/zwingel`; the owner's OK (D2) |
