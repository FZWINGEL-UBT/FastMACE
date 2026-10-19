# Phase 1b: contracts

Status: **planned (draft)**, 19 October 2026, for the conductor's review.
Spec: [SPEC.md](../SPEC.md) §16 row 1b and the sections it cites: §7.3,
§7.6, §7.7, §8.1, §9.2, §10.6, §12.1, §12.3; for the minimal planner also
§9.3–§9.5, and §13 and §14 where named. Estimate: 4–6 weeks (23 tasks, about
40 worker-days, critical path about 18 days plus reviews). Requires phase 1a
merged (done, main `bb35363`).

"Source" paths starting with `Experiments/` are in ResearchModule
(`G:\Margraf\ResearchModule`, read only). Unless a path says otherwise,
research-engine sources are at the tag `research-engine-baseline-20261017`,
relative to `Experiments/Langevin_Thermostat/`. Most of phase 1b is new
library code with no research-engine source; there, "Source" names the
FastMACE code of phase 1a that the task builds on.

## Goal

Phase 1a moved the research engine's GPU path into FastMACE with the same
numbers and the same speed. Phase 1b gives it the **contracts** of a
library:

- a planner that decides from rules and refuses with reasons;
- a `Simulation` that takes a `ForceField`, checks every `run` before the
  first step, and restarts completely;
- energies accumulated in float64;
- a gate on the user's structure, energy included;
- typed exceptions;
- the adoption layer (`fm.md`, `fm.calc`, `fastmace.ase`, the CLI) over the
  same objects.

Phase 1b changes numbers in exactly one place: the potential energy, through
float64 accumulation (§7.3). Forces and stress keep every bit (SPEC §7.3
"Energy and forces are one scalar"), so trajectories do not change.
Runtime adaptation, the fallback algorithm, the gate's resource policy,
instance isolation, events, MC and the compiled MD tier stay out (phases 2,
3 and 5).

**Acceptance (SPEC §16 row 1b, made concrete):**

1. **Gate with energy, 19 labels.** The force/stress/energy gate passes for
   all 19 labels of §12.5. It runs on the structures that count, under the
   1a rule ("Conductor decisions after T17": model-level failures are
   warnings, not engine verdicts). Energy uses the ratio rule against the
   float32 yardstick, both sides accumulating in float64, with the
   0.01 meV/atom floor. Concretely:
   - POLAR-1 S/M/L and OMOL-0 XL production energies lie within the gate.
     In 1a they were 45–102 meV/atom off the float64 reference on NbCl5,
     and 2.4–95 meV/atom on the counting structures;
   - MACE-OFF energies lie within the gate (1a gate: 3.9–15 meV/atom
     production error; SPEC §16 quotes 0.1–12).
   The driver is `tools/force_stress_gate.py`, rebuilt on the library's
   reference records and ratio rule (1b-T23): 3 configurations per structure
   and cell mode, references in fresh processes. In addition, the
   in-library reduced gate (§12.3, 1b-T16) reports `passed` for every label
   on one counting structure, in the mode it runs (captured replay).
2. **Registry and capture-state checks.** The runtime-state CI checks pass on
   the GPU for MH-1 hybrid NVT (pipelined), MH-1 NPT BAOAB, POLAR-1-S and
   plain cuEq float64:
   - the registry equals the set of tensors that any stage mutates;
   - the full registered state is identical before and after every capture
     warm-up, sticky flags included.
3. **Restart.** Round trip (save → load → save gives identical files).
   Interrupted against uninterrupted runs with the same restart cadence:
   - bitwise on CPU float64 (tiny models), across a partly consumed noise
     batch, a variable-cell run with h ≠ h₀ at the restart step, and a
     count-overflow redo before it;
   - statistical for the hybrid (MH-1 NaCl, same measures as 1a-T17);
   - within 1e-12 Å for plain cuEq float64 on the GPU, which is not bitwise
     even uninterrupted (SPEC §18).
   A run killed between a restart write and the next flush resumes with no
   duplicate or missing frames and rows: chunks deleted, nothing rewritten.
4. **Adoption layer, §7.7 acceptance for Langevin, BAOAB and the CLI:**
   - an ASE Langevin script ported by changing the import and adding
     `model=` runs on the graph engine with the plan of the equivalent
     `Simulation`;
   - `fastmace doctor`, then `fastmace md`, gives a complete run with plan,
     gate and speed report, without Python;
   - `fm.md`, `fastmace.ase` and `Simulation` give the same plan and gate
     for the same inputs, and the same trajectory statistically (hybrid);
   - in alternating A/B blocks in one process, their steady-state step
     times are equal within the noise floor (1a-T18: 0.16 % for 12 blocks);
   - the observer, signature, trajectory and `dispersion="d3bj"` items of
     §7.7, as scoped by owner decision D4 below;
   - every path that does not reach the graph engine runs and reports why,
     or refuses with the reason.
5. **No regression.** Steady-state step time of the 1b graph engine against
   the 1a engine (FastMACE tag `fastmace-1a`, to be set on `bb35363`) within
   the noise floor: MH-1 NaCl and NbCl5, hybrid NVT, on murphy and the RTX
   5060 Ti, same-process A/B with 12 blocks. Setup time reported per stage,
   gate included (§12.5).
6. **Compatibility.** ResearchModule's `main.py --engine fastmace`, the
   default since 19 October 2026, keeps running on 1b's main:
   - its adapter tests (`tests/core/test_fastmace_engine.py`) pass;
   - a smoke run matches `--engine gpu` as in 1a-T19;
   - the energy differs only by the float64 accumulation (D2).
7. **Rules.** The layering test passes, with no new module-level switches.
   Every public exception derives from `FastMACEError` and has the §7.7
   message shape.

## Working principles

- **Contracts first, then fill.** 1b-T01 fixes the public names, the config
  fields and the stubs every later task fills. Like 1a's T01/T02, it is the
  only task that edits `config.py`. Others report needed fields to the
  conductor.
- **One owner per file**, listed per task. Some files have a second
  editor after the first task merges, named in the task (for example
  `model/forces.py`: T03, then T04). Then the second task rebases and
  touches only the named part.
- **Same trajectories.** Every 1b task leaves forces, stress and
  trajectories bitwise unchanged on CPU float64 against main before the
  task. Only T03 changes energies, and only through the float64 sum.
  Equivalence tests compare at `rtol=1e-12` or after a throwaway evaluation
  (`docs/development.md`).
- **`main.py` keeps running.** ResearchModule's adapter
  (`langevin/fastmace_engine.py`) imports, from FastMACE:
  - `config`;
  - `planner.fixed` (`FixedPlanner`, `PlanRefused`, `REWRITE_FIELDS`,
    `auto_options_off`);
  - `model.compile.CompiledModelCache`;
  - `provenance` (`model_block`, `fastmace_block`);
  - `io.trajectory` and `io.restart` (`state_arrays`, `load_state`,
    `read_restart`, `truncate_outputs`).
  1b-T01 pins this surface in a CPU test. A task that moves one of these
  names leaves a re-export, and removing them is a later cross-repository
  task.
- **New public API only through the planner.** `Simulation`, `fm.md`,
  `fm.calc` and the `fastmace.ase` classes get their engine from the
  minimal planner (1b-T12). None builds a force field or engine itself
  (SPEC §5).
- **Tests first,** as in 1a. GPU tests are written and run on CPU where
  they can be (skipped without CUDA), before the conductor gives the GPU
  slot.

### Resource rules (learned in 1a)

- **One GPU per task, named in the brief and in every follow-up.** The GPUs
  that are ours: the local RTX 5060 Ti (WSL), leto and murphy (both RTX 4000
  Ada 20 GB). **coraline and leeloo are used by others** (user `kaifi`'s
  multi-day jobs since 11 October 2026). Use them only after an idle check,
  never for timing, and only for the coraline soft-lock control below.
- **Idle check first** (`nvidia-smi`; on the PC also Windows'
  `nvidia-smi.exe`). Timing runs (1b-T11, 1b-T23's speed items) need an
  otherwise idle GPU and nothing else running on the machine. While a timing
  run is on, every brief and follow-up lists the reserved machines.
- **References run in fresh processes** in the acceptance gate: one process
  per structure and calculator, which builds no production engine, before
  the production process (1a "after T10", 1a-T19). The in-library gate runs
  them first in-process, under the rule of 1b-T16.
- **Two engines do not share a process.** FastMACE and the research engine
  patch process-wide state: the Inductor custom-pass slot, `mask_head`, and
  the PolarMACE functions. So do two FastMACE backend families (§8.7).
  - A/B against the research engine stays in `fastmace.bench.ab` with its
    handling of the custom-pass slot. Polar models and plain cuEq are timed
    in alternating processes.
  - A/B of FastMACE against FastMACE (1b against 1a, `fm.md` against
    `Simulation`) runs in one process, with one backend family.
- **Ready list, prepare-then-wait** (`plans/CONDUCTOR.md`). In every
  report-driven turn the conductor updates the ready list, and the owner
  starts it with "start all ready", at most four workers at once. Dependent
  tasks start early in an owner turn, prepare, report "prepared, waiting"
  and get a go message after their dependency merges. 1b-T14, T16, T17,
  T20, T21 and T23 are planned that way.
- **Hybrid test children** use `FASTMACE_HYBRID_PYTHON` (a `mace_cueq_oeq`
  Python) while pytest runs in `mace_cueq_latest`. On the NAS, pytest needs
  `--basetemp=/mnt/homes/zwingel/fastmace-cache/pytest-<name>`. While
  siblings run locally, the CPU suite uses at most `-n 2`.

## Tasks

Each task gives: ID, size (S < ½ day, M ≈ 1 day, L 2–4 days),
dependencies, spec sections, source, files owned, tests first, and
done-when. A task whose GPU part is named runs its CPU tests first and then
asks the conductor for that GPU.

### Wave 0: contracts (conductor plus one worker)

#### 1b-T01 Contracts: public names, config fields, stubs (M, conductor)

- Spec: §5, §7.1, §7.2, §7.6, §9.2, §9.5, §10.6.
- Source: `docs/design/interfaces.md` (its open questions 2 and 3),
  `docs/design/switch-map.md` (open question 7), `fastmace/state.py`
  (`FIELDS_ADDED_IN_1B`), the 1a decision sections of `plans/phase-1a.md`.
- Files:
  - `docs/design/contracts-1b.md`: one page, the public objects and who
    builds what (diagram below);
  - `fastmace/config.py`, new fields:
    - `PrecisionConfig.float64_energy: bool = True`;
    - `IntegratorConfig.external_stress_GPa`: `None`, a scalar or 6 Voigt
      components, in ASE's `externalstress` sign; `None` means
      `pressure_GPa`;
    - `IntegratorConfig.disable_cell_langevin: bool = False`;
    - `IntegratorConfig.temperature_K`, needed by the public integrator
      objects;
  - stubs:
    - `fastmace/planner/workload.py` (`Workload` as §9.2, with its
      validation);
    - `fastmace/simulation.py` (`Simulation` signature);
    - `fastmace/integrators/__init__.py` (the public records
      `fm.integrators.Langevin` and `fm.integrators.LangevinBAOAB`, the
      latter ASE's name, mapping onto `IntegratorConfig`);
    - `fastmace/io/output.py` (`Output`);
    - `fastmace/engine/registry.py` (the registered-buffer record and its
      policies, below);
    - `fastmace/quick.py` (`RunResult`);
    - `fastmace/longrange/d3.py` (the `D3` record only, §11.2 fields);
  - `fastmace/__init__.py`: the lazy public names (`load`, `ForceField`,
    `Simulation`, `Workload`, `Output`, `integrators`, `protocols`, `md`,
    `calc`, `D3`), so that `import fastmace` still imports neither torch nor
    MACE;
  - `tests/cpu/test_research_surface.py`: every FastMACE name ResearchModule's
    adapter imports still imports.
- Decisions this task makes and records in the design doc:
  - the registry policies. Each persistent buffer is `save` (in the restart
    file), `reset` (recomputed at a canonical point), or `derived`
    (recomputed from saved state); and `capture` (snapshotted around
    warm-ups), `sticky` or not;
  - the `SimulationState` fields 1b adds (from `FIELDS_ADDED_IN_1B`: `p_eps`,
    `force_eps`, `charges`, `total_charge`, `spin_multiplicity`, `pbc`,
    protocol position, `plan_revision`, force-field identity);
  - where the gate hooks into `Simulation`;
  - the module placement deviations below (§5 lists no module for
    `Simulation`).
- Tests first: config defaults and frozenness; `Workload` validation (§9.2:
  `mc_moves` without `min_mc_temperature_K`, `min_mc_temperature_K ≤ 0`,
  swap without `species_changes`, `atom_count_changes=True` accepted by
  the dataclass and refused by the planner); the public names resolve lazily
  (`tests/cpu/test_import_time.py` extended); the research surface test.
- Done when: stubs import; the layering and research-surface tests pass;
  the conductor has reviewed the design doc (owner checkpoint 1 is
  optional, 15 minutes, on the public names).

#### 1b-T02 Exceptions and the message shape (S) — after T01

- Spec: §7.6 "Exceptions", §7.7 "Errors".
- Source: the 1a exceptions: `CheckpointTrustError`,
  `HybridUnavailableError`, `BackendError`, `NaiveCueqError`,
  `ModelLicenceWarning` (`model/loading.py`); `PlanRefused`
  (`planner/fixed.py`); `ProcessConstraintError` (`engine/constraints.py`);
  `RewriteRefusedError` (`rewrites/registry.py`).
- Files: `fastmace/errors.py` (the whole §7.6 hierarchy plus 1a's, with
  `PlanRefused` carrying every reason, `SpeciesNotInSuperset(PlanRefused)`,
  `GateFailed`/`GateUnavailable` carrying the gate record); the three 1a
  modules (re-exports only: each old name is the `errors` class).
- Message shape: `FastMACEError(what, why, next_step, record=None,
  environment=False)`. `str()` gives the §7.7 shape: what happened; why, in
  one or two sentences; what to do next; where the record is; and
  `fastmace doctor` when `environment` is set. A plain one-argument
  construction still works (1a call sites).
- Tests first: hierarchy (`except PlanRefused` catches
  `SpeciesNotInSuperset`; all derive from `FastMACEError`); one refusal
  with three reasons gives one exception with three reasons; the message
  shape for each class; old import paths give the same class; the existing
  tests that catch the 1a names still pass.

### Wave 1: foundations (parallel after T01)

#### 1b-T03 Float64 energy accumulation (L, GPU at the end: RTX 5060 Ti) — after T01

- Spec: §7.3 "Energy" and "Energy and forces are one scalar", §6.2
  `float64_energy`, §4.3 energy scale, §12.1 "Energy accumulation in the
  references".
- Source: no research-engine code; this is new in the library. MACE 0.3.16
  `mace/modules/models.py`: `MACE.forward`, `ScaleShiftMACE.forward` (its
  `out["node_energy"]` omits the embedding readout energy), the `PolarMACE`
  energy terms. `tools/validation/roundoff_reference.py` measured the
  float32 errors. FastMACE `model/forces.py` (the energy output) and
  `longrange/polar.py` (polar terms).
- Files: `fastmace/model/energy.py`. It holds the float64 reduction: E0
  table upcast exactly to float64 and summed in float64, every other term
  reduced from force precision into float64, per-atom terms with the list
  of terms they resolve. Also edited:
  - `fastmace/model/forces.py`: the energy path;
  - `fastmace/longrange/polar.py`: the electron, electrostatic (per-k and
    self) and field terms reduced into float64;
  - `fastmace/rewrites/registry.py`: only if the reduction needs a
    source-hash pin.
  T04 edits `forces.py` and `polar.py` after this task merges.
- The reduction is a reference treatment, not a rewrite (§6.2). When it
  replaces MACE's forward source, it carries a source-hash pin like the
  rewrites, and a mismatch refuses the model with a reason. It must work in
  captured graphs and under `torch.compile`.
- `PrecisionConfig.float64_energy=False` reproduces 1a's energy, for the
  A/B against the research engine only. The plan records it and the report
  warns.
- Tests first:
  - tiny models of every family (mh1, residual, density, off, omol,
    polar), CPU: float64 model → equal to the stock float64 energy within
    1e-12 relative;
  - float32 model with float64 accumulation → the error against float64
    is that of the forward pass, not of the E0 sum: on synthetic
    all-electron E0 tables it is below the stock float32 error by the E0
    rounding;
  - **forces and stress bitwise equal** with and without the float64 sum
    (float32 and float64, compiled and eager);
  - the energy never comes from `out["node_energy"]`: a regression test
    with OMOL's embedding readout term and PolarMACE's terms;
  - per-atom energies sum to the total and name the terms they omit
    (k-space).
  - GPU: on NbCl5 3×3×3, the float32 production energy of POLAR-1-S and
    OMOL-0 XL against float64 (1a: 45–102 meV/atom off), recorded in
    `docs/validation/1b-T03-energy.md`;
  - GPU: MH-1 `matpes_r2scan` energy change against 1a, which owner
    decision D2 needs;
  - GPU: step time against main before the task, in a same-process A/B with
    12 blocks on NaCl (expected within the noise floor).
- Owner decision **D2** (below) before merge.

#### 1b-T04 Runtime-state registry and complete capture restore (L, GPU: leto) — after T01; edits `forces.py` and `polar.py` after T03

- Spec: §8.1 "Runtime-state registry", "Capture warm-ups leave no trace",
  "Overflow before commit" (only the flags; the redo policy is phase 3);
  §12.2 "Runtime state".
- Source: `GraphEngine.runtime_state` (the seed); `GPULangevinBAOAB._graph_state`
  (`langevin/gpu_npt.py`, the counter-example SPEC §8.1 names); the
  `buffers()` of `MACEForceField`, `SkinNeighbourList`, the integrators and
  `longrange/polar.py`.
- Files:
  - `fastmace/engine/registry.py`: the registry and the snapshot/restore
    used by capture;
  - `fastmace/engine/graph.py`: capture warm-ups through the registry, and
    `runtime_state` replaced by it;
  - `fastmace/engine/base.py`;
  - after T03 merges, the `buffers()`/registration parts of
    `fastmace/model/forces.py` and `fastmace/longrange/polar.py`. The
    k-vector overflow flag is sticky;
  - `fastmace/neighbors/skin.py` (registration only);
  - integrator registration through their `buffers()`. An integrator-side
    change goes to T05, which owns `integrators/`.
- The registry lists for every buffer its owner, its restart policy (from
  T01) and whether it is sticky. A registered `reset` buffer names the
  function that recomputes it, which T14 calls at the canonical point.
- Tests first:
  - CPU (tiny models, eager engine): the registry equals the set of tensors
    whose `_version` counter any stage changes (A, C, B, the pipelined
    redo, a rebuild), over all objects reachable from the engine. An
    unregistered mutated tensor fails with its attribute path;
  - the full registered state is bitwise identical before and after every
    capture warm-up (AC, C, every B bucket, ACP, speculative), sticky flags
    included;
  - GPU: the same on MH-1 NaCl 3×3×3 hybrid pipelined NVT, NPT BAOAB (full
    cell), POLAR-1-S NaCl and plain cuEq float64;
  - trajectories bitwise equal to main before the task (CPU float64) and
    within 1e-12 Å (plain cuEq GPU).

#### 1b-T05 Complete state and restart format (L) — after T01

- Spec: §10.6 (physical, integrator and protocol, random, force-field
  identity and plan state; atomic writes; `RestartMismatch`), §9.5 (the
  restart stores the active revision).
- Source: `workflow.write_restart` (`langevin/workflow.py:880`), `state()`
  in `GPULangevin` (`_noise`, `_noise_next`), `LEGACY_PROTOCOL` and its use
  in `workflow.py`, `GPULangevinBAOAB` state; FastMACE `io/restart.py` and
  `state.py` (1a-T14, T12); ASE 3.27 `ase/md/langevinbaoab.py`
  (`externalstress`, `disable_cell_langevin`).
- Files: `fastmace/state.py` (the 1b fields of T01; `FIELDS_ADDED_IN_1B`
  emptied except `mc`), `fastmace/io/restart.py` (format version 2: every
  §10.6 item except MC, atomic temp-file-and-rename, reads 1a version-1
  files and research-engine files), `fastmace/integrators/baoab.py`
  (`p_eps`, `force_eps` move into the state; reads
  `external_stress_GPa` and `disable_cell_langevin` from the config, the
  constructor keywords stay for one release), `fastmace/integrators/langevin.py`
  and `base.py` (state-dict changes only).
- The force-field identity record (§10.6: checkpoint SHA-256, head,
  superset, force precision, physics options, reference record id, MACE and
  FastMACE versions) is defined here as a frozen record with a stable
  digest. T12 fills it and T17 compares it on resume.
- Tests first: save → load → save byte-identical; every §10.6 field present
  (a test lists them against the spec's bullets); generator state, drawn
  batch and cursor restored, a partly consumed batch continues with the
  same numbers; a BAOAB state with `p_eps` in `SimulationState` gives the
  same 20 eager steps as 1a (CPU float64, bitwise); a 1a restart file and a
  research-engine restart file still load; identity digests differ when any
  identity field differs; `external_stress_GPa` with six components equals
  ASE's `externalstress` in a 20-step BAOAB comparison.

#### 1b-T06 Output chunks at restart points (M) — after T01

- Spec: §10.6 "Output files at restart points", §8.1 async output, §12.2
  restart row (the crash test).
- Source: `workflow.truncate_outputs` and the resume path of
  `workflow.run_crystal` (what is replaced); FastMACE `io/trajectory.py`
  (`OutputWriter`, `AsyncOutput`, `truncate_outputs`).
- Files: `fastmace/io/output.py` (the `Output` record of T01, and the
  chunked writer: `trajectory.<n>.traj` and `thermo.<n>.csv`, chunk
  boundaries exactly at restart steps, chunk indices handed to the restart
  writer), `fastmace/io/trajectory.py` (writer changes; `truncate_outputs`
  stays for `main.py`; `output_steps` re-exported from `fastmace.protocols`
  once T07 merges).
- Resume deletes the chunks after the restart's indices and opens new ones.
  It never rewrites a file. Outputs ahead of the restart that cannot be
  deleted (read-only directory, missing chunk) raise `RestartMismatch`
  before anything is touched.
- Tests first:
  - chunking with the synchronous and the asynchronous writer gives the
    same files;
  - resume after a restart deletes exactly the later chunks;
  - **crash test**: a child process is killed (SIGKILL) after a restart
    write and before the next flush; the resumed run's concatenated
    trajectory and thermo have no duplicate or missing frames and rows
    (CPU, tiny model);
  - a read-only directory gives `RestartMismatch` and leaves the files
    unchanged;
  - `ase.io.read(..., index=":")` over the chunks gives every frame.

#### 1b-T07 Schedules and protocols (M) — after T01

- Spec: §7.2 "Schedules and step counts", §7.6 "Schedules", §10.5.
- Source: `workflow.PROFILES` and `_RAMP` (`langevin/workflow.py:28`), the
  heat/cool/hold logic of `workflow.run_crystal`, `studies/annealing/README.md`;
  FastMACE `io/trajectory.output_steps` (1a-T14; moves here, decision after
  1a wave 1b).
- Files: `fastmace/protocols/__init__.py`, `fastmace/protocols/schedule.py`
  (`ramp`, `hold`, `+`, rounding to whole steps, boundaries and total
  recorded, the target per step, maximum and minimum over a step range, the
  schedule position as restart data), `fastmace/protocols/profiles.py`
  (today's `slow`, `fast`, `ultrafast` and `anneal` as data; TOML and
  YAML through `safe_load` only), `output_steps`.
- The target reaches the integrator through `Integrator.set_temperature`
  (a device-scalar fill between replays, 1a), never as an event. Fill it
  per step only where the target changes. Follow the research engine's
  ramp granularity and record it.
- Tests first: `+` refuses a temperature jump (`ValueError`); rounding,
  boundaries and totals for odd timesteps; `anneal` at 0.5 fs is 540,000
  steps (SPEC §7.2); the target sequence equals the research engine's ramp
  for `ultrafast` step for step; schedule position round-trips; `steps`
  beyond the remainder is refused (the check function T17 calls); `output_steps`
  equals 1a's.

#### 1b-T08 `fm.load`, model cache and checkpoint trust (S) — after T02

- Spec: §7.1, §7.6 "Loading and checkpoint trust", §7.7 "Heads", §14
  caches.
- Source: FastMACE `model/loading.py`, `model/labels.py`, `_cache.py`;
  decisions after 1a wave 1 (`"models"` in the cache kinds, head
  functionals provisional).
- Files: `fastmace/_cache.py` (`"models"` kind), `fastmace/model/loading.py`
  (the public `load`: label, path or URL; `head=` required for
  multi-head checkpoints, else `ValueError` listing the heads; a label may
  name its head; `sha256=`/`trusted=`; `download=True` opt-in into
  `cache_dir("models")`, verified before unpickling; the trust decision and
  hash in provenance), `fastmace/model/labels.py` (head functionals checked
  against the MH-1 release, with the source of each).
- Tests first: path without `sha256`/`trusted` → `CheckpointTrustError`; a
  wrong hash → the same, before unpickling (a pickle that would write a file
  if loaded); a multi-head checkpoint without a head → `ValueError` naming
  the heads; a download into the models cache is verified (local HTTP
  server); every label's head functional is set or explicitly unknown; the
  provenance carries the trust decision.

#### 1b-T09 Integrator qualification: ASE equivalence and the BAOAB correction (M, GPU at the end: murphy) — after T01

- Spec: §12.1 "Integrator corrections", §12.2 "ASE equivalence", §10.1.
- Source: `studies/npt/README.md` ("Deliberate difference from ASE 3.27"),
  `tests/core/test_gpu_npt.py`, `tools/validation/compare_gpu_ase.py`;
  ASE 3.27 `ase/md/langevinbaoab.py`; Quigley and Probert 2004, Eq. 5b;
  FastMACE `integrators/baoab.py` (1a-T12).
- Files: `fastmace/reference/integrators.py`, the independent corrected
  reference (owner decision D8):
  - ASE 3.27's `LangevinBAOAB` with only the drag line replaced by
    (p_ε + Tr(p_ε)/N_dof · I)/W. It shares ASE's noise stream, so it
    allows an exact trajectory comparison;
  - a float64 single-step check written from the paper's equation. The
    worker writes it from the paper, without reading `baoab.py`'s drag
    code first.
  Tests: `tests/cpu/test_integrator_qualification.py` and
  `tests/gpu/test_ase_equivalence_gpu.py`.
- Tests first:
  - ASE variant (`ase_trace_drag_all_elements=True`) against ASE's
    `LangevinBAOAB`, full and hydrostatic cell: equal within 1e-9 over 300
    steps (CPU float64, tiny model);
  - production variant against the corrected reference: equal within 1e-9
    over 300 steps, full cell;
  - the two variants differ by the expected O(1/N_dof) (a negative
    control);
  - GPU (§12.2): Langevin on NaCl over 9 ps with the same seed as ASE: the
    same T and E_pot in every 50 K bin, |ΔPE| < 0.01 meV/atom;
  - GPU: BAOAB, both variants against their references on NaCl.
- Done when: the results are in `docs/validation/1b-T09-integrators.md`;
  T13 enters them in the reference record.

#### 1b-T10 Schema-versioned provenance (M) — after T01

- Spec: §13, §10.6 (restart points with chunk indices), §12.1 (record id).
- Source: FastMACE `provenance.py` (1a-T16); decision after 1a wave 1b
  (additions under `"fastmace"` until the schema-versioned record).
- Files: `fastmace/provenance.py` (`metadata.json` with
  `"schema": "fastmace.metadata/1"`: every §13 item that 1b produces, with
  an explicit `null` where a later phase fills it, such as MC counters and
  tier switches; generator-state hashes and cursors, never full states;
  the plan revision chain; gate record; reference record id; restart
  points with their chunk indices; setup-time breakdown), `docs/design/metadata-schema.md`.
  `model_block`, `fastmace_block` and `run_record` keep their 1a output
  for `main.py`.
- Tests first: every §13 bullet maps to a key (a test lists them); schema
  version present; the 1a keys are unchanged (the stored sample from
  1a-T16); no full generator state or noise batch in `metadata.json`;
  round trip through `json`.
- Phase 2 ("provenance schema") freezes and validates this schema. 1b
  writes version 1.

#### 1b-T11 MP-0 speed gap and setup-time diagnosis (M, GPU: murphy, idle, timing) — no dependencies

- 1a open findings 1 and 2. This is a **parity diagnosis, not a new
  optimisation**:
  - MP-0 medium is 1.4–1.6 % slower than the research engine, about ten
    times the noise floor, with `species_skip` on both sides;
  - setup takes 4–33 s longer.
  A fix belongs to 1b only if it restores what the research engine does
  (same numerics). Anything else goes to the screens after 1b.
- Source:
  - FastMACE's generic path: `rewrites/generic/`, `model/forces.py`,
    `model/compile.py`;
  - the research engine's generic path in `gpu_langevin.py`;
  - its breakdowns in `results/generic-rewrites-rtx5060ti-20261015/breakdown-mp0-medium`;
  - `studies/other_models/probes/model_step_time.py`;
  - 1a-T19's murphy runs (`results/1a-T19-speed-murphy-20261018/`).
- Files: `docs/validation/1b-T11-mp0-setup.md`; a fix only in the file that
  causes the gap, agreed with that file's 1b owner through the conductor.
- Method:
  - speed: compare kernel sequences of the B graph (node counts,
    per-category times) with the research engine's, for MP-0 medium on
    NaCl and NbCl5;
  - setup: time load, conversion, OEq build, each rewrite's self-check,
    compile and capture separately, in single-engine processes.
- Done when: the gap is explained (and fixed with a same-process A/B
  showing parity, or listed as a screen candidate), and the setup budget per
  stage is recorded. Then 1b-T15's report shows the same stages.

### Wave 2: planner, references, canonical restart, report

#### 1b-T12 Minimal planner and the plan-free `fm.ForceField` (L) — after T01, T02, T08

- Spec: §9.1–§9.3 (rules 1–7 and the preferences 8–12 to the extent 1b
  has their tiers), §9.5 (frozen `Plan`, revisions), §7.1, §7.6
  "ForceField and plans", "Species superset", "Simulation inputs", §8.7
  (process constraints at plan time), §4.4.
- Source: FastMACE `planner/fixed.py` (1a-T15: the rewrite resolution,
  refusals and `integrator_keywords`), `planner/plan.py`,
  `kernels/__init__.py` (availability probes), `model/loading.py`
  (`hybrid_blockers`, `require_fused_cueq`), `engine/constraints.py`.
- Files:
  - `fastmace/planner/minimal.py` (`MinimalPlanner`);
  - `fastmace/planner/plan.py` (`Plan.revise(reason, step, **changes)`; the
    software traits record: OEq build path, Triton, NVRTC, fused cuEq
    kernels loading, MACE imported before FastMACE);
  - `fastmace/planner/fixed.py` (shared helpers moved, kept importable;
    `integrator_keywords` reads the new config fields);
  - `fastmace/model/forcefield.py` (public `fm.ForceField`).
- `fm.ForceField(model, precision="float32", rewrites="auto",
  species=None)` holds the pristine loaded model, the force precision and
  the superset (default: the structure's species, at first use). It also
  holds the force-field identity of T05. It is plan-free: each plan builds
  its own `MACEForceField` execution copy through the planner.
  `ff.evaluate(atoms)` plans a one-off evaluation. A superset species
  without an embedding raises `ModelElementError`.
- Rules in 1b:
  - **Hard constraints, all reasons in one `PlanRefused`:**
    - 1: non-3D-PBC; constraints; a variable cell without a barostat
      integrator. `allow_reference` is refused in 1b, owner decision D6;
    - 2: species outside the superset → `SpeciesNotInSuperset`;
    - 3: PolarMACE with plain compiled cuEq;
    - 4: inexact padding refuses the graph tier ("compiled tier: phase 3");
    - 5: `reproducible=True` ("no family qualified", phase 2);
    - 6: naive cuEq unless `allow_naive_cueq=True`, recorded;
    - 7: `atom_count_changes`.
  - **Mode:** graph. Every other mode is refused with the reason and the
    phase that brings it.
  - **Backend:** hybrid when OEq builds and MACE was not imported before
    FastMACE, else plain cuEq with the reason (§9.4, 1a-T00). The
    process's backend family is respected (§8.7).
  - **Preferences:**
    - rule 9, frequent host callbacks: owner decision D6;
    - rule 10: stress path and k-vector headroom from the integrator;
    - rule 11 records `max_temperature_K` but keeps 1a's skin and buckets
      (adaptive skin is phase 3);
    - rule 8 and rule 12 are not in 1b.
  - **Superset beyond the structure's species:** owner decision D3.
  - The rewrite set comes from 1a's resolution, unchanged.
- Tests first:
  - for each of the 19 labels (trait fixtures of 1a-T15) and 1a-T15's 31
    `main.py` argument sets mapped onto 1b inputs, the minimal plan's
    rewrite set and engine options equal `FixedPlanner`'s;
  - each hard rule refuses with its reason, and three violated rules give
    one `PlanRefused` with three reasons;
  - a superset violation is a `SpeciesNotInSuperset`;
  - `Plan.revise` keeps the chain, and the original stays unchanged;
  - `ForceField` is plan-free: two plans from one `ForceField` give two
    execution copies, and the pristine model's parameters are bitwise
    unchanged after both;
  - the process family is enforced: a hybrid plan, then a plain compiled
    plan in one process → `ProcessConstraintError` with the subprocess
    note (the runner itself is phase 2).
- GPU check (local, short): plan → build → 500 steps of MH-1 NaCl 3×3×3
  equals `FixedPlanner`'s engine (as 1a-T15).

#### 1b-T13 Reference policy: records and corrected reference calculators (L) — after T03, T09

- Spec: §12.1 (all four kinds of entries, energy accumulation in the
  references, record id in identity and provenance), §8.4.
- Source:
  - FastMACE's `tools/force_stress_gate.py` (1a-T19): reference processes,
    `enable()` before `configure_md_calculator`, the corrected-forward
    assertion, and the finite-difference self-test;
  - `tools/validation/npt_stress_reference.py`;
  - `studies/other_models/probes/polar_stress_probe.py`;
  - the PR 1762 port in FastMACE `longrange/polar.py`;
  - T03's float64 reduction and T09's qualification.
- Files:
  - `fastmace/reference/records.py`, a versioned record per (MACE version,
    model family): stock equivalence; the upstream fix for MACE issue 1642
    (polar stress), with its validation (finite differences on NaCl 3×3×3);
    optional physics (PR 1762, D3 declared, D3 refused until 3b); the BAOAB
    integrator correction from T09; and float64 accumulation in the float32
    yardstick. Each record has a stable id;
  - `fastmace/reference/calculators.py`: MACE `MACECalculator` cuEq float64,
    and the float32 yardstick with float64 accumulation, the record's
    corrections applied in the safe order. It raises when a correction is
    not the active code path (the 1a "after T17" lesson);
  - `fastmace/reference/__init__.py`.
- Tests first:
  - CPU float64 tiny models: reference energy, forces and stress equal stock
    MACE where the record says stock;
  - the polar stress fix active, checked against finite differences of the
    reference's own energy (tolerance 1e-6 eV/Å³, as 1a-T19);
  - a reference built in the wrong order raises;
  - the yardstick's energy carries the float64 sum, and its forces are
    unchanged;
  - the record id enters the force-field identity (T05) and changes when
    any entry changes;
  - fallbacks cannot drop a correction (a record without the fix for a
    polar model is refused).
- GPU check (leto): POLAR-1-S NaCl reference against 1a-T19's references,
  to 1e-12 (same process type).

#### 1b-T14 Canonical restart points in the engine (M, GPU: RTX 5060 Ti) — after T04, T05, T06

- Spec: §10.6 "Canonical restart points", §8.1 (pipeline state), §12.2
  restart row.
- Source: FastMACE `engine/graph.py` (`_pipe_floor`, `_pipe_edges` and
  their 1a names, the speculative slot, `flush`), the registry of T04.
- Files: `fastmace/engine/graph.py` (`canonicalise()`, called by every
  restart write, interrupted or not: quiesce the pipeline, rebuild every
  neighbour list from positions, cell and species, reset floors and
  `reset`-policy buffers; then the restart write; resume constructs the
  engine to the same state). T04 owns `graph.py` before; this task edits it
  after T04 merges. Also `fastmace/io/restart.py` (only the call; T05
  owns the format).
- Tests first:
  - after `canonicalise()` the registered state equals a freshly built
    engine's at the same positions, cell and species, except the `save`
    buffers (CPU float64);
  - interrupted against uninterrupted with the same cadence, bitwise on
    CPU float64 (tiny models), across a partly consumed noise batch, a
    pipelined count-overflow redo before the restart, a rebuild in the step
    after it, and a full-cell BAOAB run with h ≠ h₀ at the restart step;
  - GPU: plain cuEq float64 MH-1 NaCl within 1e-12 Å over 500 steps with
    restarts every 100;
  - GPU: hybrid MH-1 NaCl statistically (temperature and E_pot means and
    bin means over 20,000 steps, two seeds; the 1a-T17 measures);
  - the runtime-state check of T04: nothing registered as neither saved
    nor reset.

#### 1b-T15 The run report (M) — after T02, T10, T12

- Spec: §7.7 "The report" (two stages, speed pending → measured window →
  whole-run mean; conditional warnings; licence from the label table; the
  reasons when not on the graph engine, without estimates), §12.5 (setup
  stages).
- Source: none in the research engine (its banner lines in
  `workflow.run_crystal` are a starting point for the header); 1b-T11's
  setup stages.
- Files: `fastmace/report.py`. It builds the report from a plan, the gate
  record, the setup timings, the output files and the warnings. It writes
  `metadata.json`'s report block through T10.
- Tests first:
  - golden text for a constant-T NVT run (the §7.7 example's lines);
  - the NPT "experimental: not yet qualified" line only for NPT;
  - the licence line from the label table for every label;
  - a not-graph reason for each refusal or downgrade T12 can produce;
  - the speed line never carries a prediction, and it is replaced by the
    whole-run mean at the end;
  - a `reproducible=True` request shows §9.3's refusal.

### Wave 3: gate, Simulation, calculator, first CLI commands

#### 1b-T16 Gate on the user's structure (L, GPU: leto) — after T04, T12, T13; owner decision D1 before start

- Spec: §12.3 (reduced gate, ratio rule with the floors 0.01 meV/atom,
  1e-5 eV/Å and 1e-7 eV/Å³, captured replay, statuses), §12.2 gate row,
  §9.3 rule 14 (no silent run with a failing gate), §7.6 `GateFailed`,
  `GateUnavailable`.
- Source: FastMACE `tools/force_stress_gate.py` (1a-T19: `summarize`, which
  fails on any non-finite value before taking a maximum; strain and rattle
  generation; seeds). The lessons of 1a's gate driver:
  - references before production;
  - correction asserted active;
  - production process patches must not reach references;
  - OMOL-0 XL's process-to-process float32 scatter of 1.2–1.8e-4 eV/Å.
- Files: `fastmace/planner/selfcheck.py` (the gate).
- What it does:
  - copies: 2 rattled copies (0.05 Å); for a variable-cell integrator also
    2 strained copies (±2 %);
  - references first, one configuration and one calculator at a time,
    freed before the production engine is built;
  - production through the plan's engine in the mode it runs: the registry
    snapshots the state, the gate loads each copy, replays AC/C/B and reads
    the results, then the registry restores the state;
  - statuses: `passed`, `failed`, `unavailable` (reference out of memory or
    failing), `skipped` (`check="skip"`). The CPU and reduced-cell
    fallbacks are phase 2; `passed-reduced` is reserved;
  - a failed gate raises `GateFailed` (no fallback in 1b), and
    `unavailable` follows D1.
- **References and process state.** The gate refuses to compute references
  in-process after a production engine exists in the process (the 1a rule
  "after T10"). It then runs them in a child process in the same
  environment, as 1a-T19's driver does, and records which way they ran. For the hybrid, the OEq runtime is prepared before the
  reference imports MACE.
- Tests first:
  - CPU (tiny models): the ratio rule with floors on synthetic numbers;
    NaN and inf anywhere fail; a deliberately wrong production energy
    offset (species-dependent constant) fails on energy alone; restore
    leaves the registered state bitwise unchanged;
  - `check="skip"` is recorded as `skipped`;
  - GPU: MH-1, MP-0, POLAR-1-S and OMOL-0 XL on NaCl pass in graph mode;
  - GPU: a production path with a broken captured graph (an injected
    wrong-kernel stand-in, like the PDL failure) fails the gate while an
    eager comparison would pass;
  - GPU: in-process references equal fresh-process references within 1e-12
    (float64) and within the model's float32 process scatter (yardstick);
  - GPU: the gate takes "seconds to a minute" on NaCl and NbCl5 (time
    recorded).

#### 1b-T17 `Simulation` (L) — after T07, T12, T14; wires T16's gate when it merges (prepare, then wait)

- Spec: §7.2, §7.6 "Schedules", "Simulation inputs", "ForceField and
  plans", §9.2, §10.6 (save and resume, `allow_plan_change`,
  `RestartMismatch`), §9.6 (non-finite energies stop the run with
  `NonFiniteError`).
- Source: the 1a run loop of ResearchModule's `langevin/fastmace_engine.py`
  (how `main.py` drives `GraphEngine`, async output and restarts), and
  FastMACE `engine/graph.py` and `io/`.
- Files: `fastmace/simulation.py`.
- What it does:
  - `Simulation(atoms, ff_or_model, *, species=None, integrator, workload,
    output=None, seed, check, allow_naive_cueq=False)`. With a model it
    builds the `ForceField` in float32 with `species=`;
  - `sim.plan` and `sim.gate`;
  - `run(steps=None, schedule=None, mc=...)` with every check before the
    first step:
    - a schedule above `max_temperature_K` → `PlanRefused`;
    - `steps` beyond the schedule → `ValueError`;
    - no schedule: the target stays;
    - `mc=` checked against the workload (move type, `min_mc_temperature_K`
      over every step, superset), then refused with "MC: phase 3";
  - output through T06's chunked writer;
  - `save_restart` and `Simulation.resume(directory,
    allow_plan_change=False)`: an identity mismatch raises
    `RestartMismatch`, and a plan change needs the flag and makes a new
    revision;
  - `set_positions` and `set_momenta`: a host write applied at a canonical
    point (owner decision D4);
  - non-finite energy at a committed output step → `NonFiniteError`;
  - workload default: owner decision D7.
- Tests first (CPU, eager engine where possible):
  - every refusal of §7.6 before any step, with the state unchanged;
  - `run(steps=n)` along a schedule, then `run()` finishes it exactly, and
    a finished schedule refuses `run()`;
  - a resume continues the schedule position and the attached state;
  - an identity mismatch → `RestartMismatch`;
  - the plan change creates revision 1 with predecessor 0;
  - `Simulation(atoms, model)` equals `Simulation(atoms, ForceField(model))`;
  - a NaN injected into the energy stops the run with `NonFiniteError`.
  - GPU (local, short): the 1a-T17 smoke run's numbers through `Simulation`
    instead of `main.py`'s loop: MH-1 NaCl, 60 steps, within FP32
    divergence.

#### 1b-T18 `FastMACECalculator` and `fm.calc` (M, GPU: murphy, short) — after T03, T12

- Spec: §7.4, §7.6 "ASE front end", §7.7 level 0, §8.2.
- Source: MACE 0.3.16's `MACECalculator` (the results contract); FastMACE
  `model/forces.py` `evaluate` (compiled model, no graphs).
- Files: `fastmace/ase/calculator.py`, `fastmace/ase/__init__.py` (owned
  jointly with T21: this task creates it, T21 adds its names).
- What it does:
  - at the first calculation, a minimal plan through the planner:
    compiled mode with the 1a `MACEForceField.evaluate` path, no workload;
  - `implemented_properties`: `energy`, `free_energy`, `forces`, `stress`
    (Voigt);
  - `energies` only for models whose terms are all atom-resolved;
  - non-3D-PBC raises `PlanRefused`;
  - prints "compatibility tier: compiled, no CUDA graphs; use `fm.md` or
    `fastmace.ase` integrators for full speed" once;
  - float64 energies.
- Tests first (CPU tiny models): results equal the corrected reference
  calculator (T13) within the gate rule; Voigt order and sign as ASE; a
  molecule without PBC raises `PlanRefused`; the message prints once;
  `energies` is absent for a polar model. GPU: MH-1 NaCl through
  `ase.optimize.LBFGS` with `fm.calc` reaches the same minimum as with
  MACE's calculator.

#### 1b-T19 CLI `doctor` and `models` (S) — after T02, T08

- Spec: §7.5 (`doctor`, `models`, error printing), §14 (lock mode, cache
  root).
- Source: FastMACE `kernels/__init__.py` probes, `model/loading.check_cueq`,
  `check_oeq`, `hybrid_blockers`, `runtime_info()`, `_cache`, `model/labels.py`.
- Files: `fastmace/cli/__init__.py` (`main`, error printing without a
  traceback, `--debug`, non-zero exit), `fastmace/cli/doctor.py`,
  `fastmace/cli/models.py`. The entry point
  `[project.scripts] fastmace = "fastmace.cli:main"` goes into
  `pyproject.toml` through the conductor.
- `doctor` checks: GPU and driver, torch, MACE, cuEquivariance ops and
  whether its fused kernels load, OEq and its build path, Triton, NVRTC,
  graph-longrange, torch-dftd, cache root, lock mode on NFS, model files and
  their hashes. Each failed item names the command or document that fixes
  it. It changes nothing.
- Tests first (CPU): each check's pass and fail text with the probes
  mocked; `doctor` writes no file (temporary home, file listing before and
  after); `models` lists all 19 labels with heads, file names, SHA-256 and
  licences; a `FastMACEError` prints the message shape alone and exits 1,
  and `--debug` shows the traceback.

### Wave 4: adoption layer

#### 1b-T20 `fm.md`, `RunResult` and the dispersion record (M) — after T08, T15, T17

- Spec: §7.7 level 2 (arguments, heads, dispersion, inference, MC window,
  defaults, `repeat`, "physics is never chosen silently"), §11.2 (the D3
  record's fields; the kernel is phase 3b).
- Source: none in the research engine; T08's head functionals.
- Files: `fastmace/quick.py` (`md`, `RunResult` with `summary`, `plan`,
  `gate`, `simulation`, `metadata`, `output`), `fastmace/longrange/d3.py`
  (the `D3` record of T01 and the `"d3bj"` expansion; no kernel).
- What it does:
  - `ensemble="NVT"` means Langevin and `"NPT"` BAOAB; `"NVE"` is refused
    ("phase 4");
  - the superset and `max_temperature_K` are inferred from the arguments;
  - `mc=` runs the §7.7 checks (`ValueError` for `temperature_K=0` with
    `mc=`, `min_mc_temperature_K` required with a schedule), then is
    refused with "MC: phase 3";
  - `dispersion="d3bj"` expands to the full record from the head's
    functional. It raises for MACE-OFF, OMOL-0, MH-1's `omol` head and an
    unknown functional. A complete record is then refused at plan time
    ("D3: phase 3b"), with the record shown;
  - `repeat=(1, 1, 1)` by default, with the small-cell warning.
- Tests first:
  - `fm.md` and the equivalent `Simulation` give equal plans (`Plan`
    equality) and equal gate records on a tiny model;
  - an explicit argument beats an inferred one;
  - each `ValueError` of §7.7 is raised;
  - the `"d3bj"` expansion matches the label table for every label, and
    the refusals name their reason;
  - `RunResult.summary` is the report of T15.

#### 1b-T21 `fastmace.ase` `Langevin` and `LangevinBAOAB` with observer sync (L) — after T15, T17; owner decision D4 before start

- Spec: §7.7 level 1 (arguments and units of ASE 3.27, refusals,
  keyword-only additions, mapping, ASE members, files, observers,
  constraints), §4.4.
- Source: ASE 3.27 `ase/md/langevin.py`, `ase/md/langevinbaoab.py`,
  `ase/md/md.py` (`MolecularDynamics.run`, `attach`, `observers`, `todict`,
  `get_time`), `ase/md/logger.py` (`MDLogger` format), `ase/io/trajectory.py`.
- Files: `fastmace/ase/md.py`, `fastmace/ase/__init__.py` (names added).
- What it does:
  - explicit argument lists per class, checked against `inspect.signature`;
    any other argument raises `TypeError` naming it; `rng` and a
    non-default `communicator` are refused;
  - additions are keyword-only: `model=` or `force_field=`, `head=`,
    `seed=`, `output=`, `check=`, `dispersion=`;
  - an ASE class without a mapping raises with the list of mapped ones;
  - `run(n)` advances in graph chunks between observer calls;
  - observers see the committed state written into `atoms`, with a
    results cache as `atoms.calc`;
  - after an observer: changed positions or momenta go through
    `Simulation.set_positions`/`set_momenta` (D4). Changed species,
    masses or charges raise until phase 3 (D4). A changed atom count, PBC,
    cell (except through the barostat), constraints, plan-relevant `info`
    keys or `atoms.calc` raises;
  - `trajectory=` is one ASE `Trajectory` written by the async writer, and
    `logfile=` uses ASE's format;
  - the other files go to `output=` (default `<trajectory stem>.fastmace/`).
- Tests first:
  - signature lists against ASE 3.27 for both classes; every ASE argument
    is supported or refused, so a new ASE argument fails CI;
  - `friction` in ASE units, never reinterpreted;
  - an ASE Langevin script ported by import change plus `model=` gives the
    plan of the equivalent `Simulation`;
  - the trajectory reads back frame for frame with `ase.io.read`;
  - the log file parses like ASE's;
  - an observer that changes positions gives the same state as
    `Simulation.set_positions` at that step;
  - each forbidden change raises with its reason;
  - `FixAtoms` is refused with the phase-4 reason;
  - `summary` equals `fm.md`'s report for the same run.
- GPU part: none of its own. The A/B timing of acceptance item 4 is
  1b-T23's.

#### 1b-T22 CLI `md`, `init`, `run`; `fastmace[full]` quick start (M) — after T19, T20; owner decision D5

- Spec: §7.5 (`md` flags, `init`, `run`), §14 extras, §15 "FastMACE in five
  minutes".
- Source: `workflow.parse_args` (flag naming habits), `pyproject.toml`
  (the `[full]` extra exists since phase 0).
- Files:
  - `fastmace/cli/md.py`;
  - `fastmace/cli/init.py`: commented TOML holding every `fm.md`
    argument;
  - `fastmace/cli/run.py`: reads that TOML and calls `fm.md`; D5;
  - `docs/quickstart.md`: `pip install "fastmace[full]"`, `fastmace
    doctor`, `fastmace md`, `fm.md`;
  - `tests/cpu/test_cli.py`.
- `md` flags as §7.5. `--dispersion d3bj` and `--d3-functional` are
  accepted and expanded, then refused at plan time until phase 3b. The CLI
  prints `FastMACEError`s through T19's `main`.
- Tests first:
  - `fastmace init | fastmace run` on a tiny model is equivalent to the
    same `fastmace md` flags (same plan, same `metadata.json` except
    timestamps);
  - every §7.5 `md` flag maps to an `fm.md` argument;
  - a TOML with an unknown key is refused naming it;
  - a fresh venv with `pip install ".[full]"` resolves, and `fastmace
    doctor` runs (CPU CI job; the conductor adds it to the CI workflow).

### Wave 5: acceptance

#### 1b-T23 Acceptance runs and report (L, GPU, conductor-scheduled) — after all

- Gate (acceptance 1): `tools/force_stress_gate.py`, rebuilt on
  `fastmace.reference` and `selfcheck`'s ratio rule, with energy gated.
  - all 19 labels on leto: references in fresh processes, MACE-OFF on the
    organic set (`--repeat 1 1 1` for the water box), and the 1a
    counting-structure rule;
  - then the in-library reduced gate once per label on one counting
    structure;
  - a subset on the RTX 5060 Ti: MH-1, POLAR-1-S, OMOL-0 XL and OFF24-M.
- Registry (2) and restart (3): the GPU suite in one process on leto and
  locally; the hybrid restart statistics on the RTX 5060 Ti.
- Adoption layer (4): `fastmace doctor` and `fastmace md` on a clean
  checkout. Then `fm.md`, `fastmace.ase.Langevin` and `Simulation` on MH-1
  NaCl 3×3×3 in one process, in alternating blocks (12 blocks, murphy,
  idle), with the same plan and gate records.
- No regression (5): 1b against the 1a tag, same-process A/B, murphy and
  the RTX 5060 Ti, MH-1 NaCl and NbCl5 hybrid NVT; setup time per stage.
- Compatibility (6): ResearchModule's adapter tests against 1b's main, and
  one `main.py --engine fastmace` smoke run on leto from the NAS copy.
  ResearchModule is only read; the run uses its NAS tree.
- Report: `docs/validation/phase-1b.md` with run IDs, machines, GPUs,
  environments, backend, precision, model labels, both commits and every
  number; JSON in `results/<run-id>/`.
- If coraline is idle during 1b, the conductor runs 1a's open soft-lock
  control there (below) and records it in this report. That control is not
  an acceptance item.

#### 1b-T24 Test-suite speed (M, GPU: local RTX 5060 Ti) — no dependencies (owner, 19 October 2026)

- **Why:** the whole GPU suite takes 38–40 minutes in one process (221 tests; T15 37.9 min, T13b
  40.1 min), and most engine tasks run it two or three times on one GPU. Likely causes:
  - every hybrid test runs in a fresh child process that rebuilds the OEq host extension
    (about 15 s each, 1a-T00);
  - `torch.compile` from scratch in many tests;
  - real-checkpoint agreement tests with research-engine children;
  - 500-step trajectory tests (16 engine tests took 17.5 min on leto);
  - fresh compile caches where only the regression tests need them.
- **Do, in this order:**
  1. **Measure:** run the GPU suite once in one process with `--durations=40` and record the
     table. Do the same for both CPU suites.
  2. **Tiers:** mark the expensive GPU tests `slow` (pyproject's marker already exists): real
     checkpoints, research-engine comparisons, long trajectories, cold-cache regression tests.
     `pytest tests/gpu -m "not slow"` is the worker tier, with a target under 5 minutes. The full
     suite stays the merge and acceptance tier. Every GPU-tested behaviour must keep at least
     one test in the fast tier, for example a short trajectory or a tiny model.
  3. **Persistent OEq extension:** build once per (torch, CUDA, OEq, compiler) key in the
     FastMACE cache and reuse it in later processes, instead of a fresh `TORCH_EXTENSIONS_DIR`
     per process. The key is a build identity, not a shape, so the generality rule allows it
     (decision after wave 0). Concurrent processes must not corrupt it: build into a temporary
     directory, then rename it into place. Respect the soft-lock rules (no kernel locks on NFS).
  4. **Shared compile caches:** test child processes reuse one Inductor/FX cache root per run.
     Only tests that need a cold cache (T13b's bucket regression) get a fresh one, explicitly.
- **Files:** `tests/conftest.py`, `pyproject.toml` (markers, `addopts` only if needed),
  `slow` markers in `tests/gpu/*.py`, `fastmace/kernels/_oeq_runtime.py` (the persistent build),
  `tests/cpu/test_oeq_runtime.py`, `docs/development.md` (a "Test tiers" section). Change no
  test's assertions; only markers, fixtures and cache handling.
- **Tests first:** the persistent extension is reused by a second process with no rebuild, and
  the key changes when torch, CUDA or OEq changes (CPU, faked). Two concurrent first builds both
  succeed with one result.
- **Done when:** the duration tables before and after; the fast GPU tier under about 5 minutes
  locally; the full suite still green in one process and faster than 38 minutes; both CPU
  suites green; and `docs/development.md` tells workers which tier to run when. Workers' briefs
  from then on use the fast tier, and the conductor runs the full suite at merges that touch
  `model/`, `rewrites/`, `kernels/` or `engine/`.

## Order and parallelism

```text
Wave 0   T01 (contracts, conductor) ── T02 (errors)
             │
Wave 1   T03 energy   T04 registry   T05 state/restart   T06 chunks   T07 schedules
         T08 fm.load (T02)   T09 integrator qualification   T10 provenance   T11 MP-0/setup (none)
             │             │              │       │           │
Wave 2   T13 references (T03, T09)   T12 planner (T02, T08)   T14 canonical restart (T04, T05, T06)
         T15 report (T02, T10, T12)
             │                          │                        │
Wave 3   T16 gate (T04, T12, T13)   T17 Simulation (T07, T12, T14; gate wired after T16)
         T18 fm.calc (T03, T12)     T19 CLI doctor/models (T02, T08)
             │                          │
Wave 4   T20 fm.md (T08, T15, T17)   T21 fastmace.ase (T15, T17)   T22 CLI md/init/run (T19, T20)
Wave 5   T23 acceptance (all)
```

- At most four workers at once. Wave 1 is the widest: nine tasks. Start in
  this order:
  1. T03, T04, T05 (the long ones on the critical paths);
  2. T09, which needs a GPU early;
  3. T08 and T07;
  4. T06, T10 and T11.
- **Critical path:** T01 → T03 → T13 → T16 → T17 (gate wiring) → T21 → T23,
  about 18 working days. The second path, T01 → T04 → T14 → T17, is about
  as long. Put the strongest model and most review on T03, T04, T16 and
  T17.
- **Prepare, then wait:** T14 (after T04), T16 (after T13), T17 (after T14,
  then T16), T20, T21 and T23.
- **GPU slots, in order of need:**
  - T11 on murphy (timing: idle, nothing else) and T09 on murphy after it;
  - T04 on leto, T03 on the RTX 5060 Ti;
  - T13 and T16 on leto;
  - T14 on the RTX 5060 Ti;
  - T18 on murphy;
  - then T23 exclusively.
  The owner may reserve the local GPU, as on 18 October 2026. GPU work then
  goes to leto and murphy only.

## Checkpoints for the owner

The owner has delegated routine review (17 October 2026). The conductor
reviews and merges, and the owner decides:

1. After T01 (optional, 15 minutes): the public names in
   `docs/design/contracts-1b.md`.
2. Before T16 starts: D1 (SPEC §18 open decision 3). Before T03 merges: D2.
   Before T12 and T21: D3, D4, D6. Before T17: D7. Before T22: D5.
3. After T23: the acceptance report, and whether `main.py`'s adapter moves
   to the 1b API (`Simulation` instead of `FixedPlanner` and its own loop)
   in a cross-repository task at the start of phase 2.

## Owner decisions

Each is needed by the task named. The recommendation comes first.

**Decided by the owner on 19 October 2026:** D1 as an opt-in, not on by
default; D2–D8 as recommended below. All are recorded in SPEC §18 (3,
15–21) and §12.3.

- **D1, SPEC §18 open decision 3: gate default and `unavailable` (T16, T17,
  T20, T21).** **Decided: opt-in.**
  - The gate is off by default in `Simulation`, `fm.md`, `fastmace.ase` and
    `fastmace md`. Its status is then `skipped`, recorded, and the run
    report says "gate not run (opt-in: check="gate")".
  - `check="gate"` requests it. A requested gate that is `unavailable`
    refuses with `GateUnavailable`, whose message names
    `check="allow_unavailable"` (recorded, and it also requests the gate).
  - T16 builds and tests the gate fully, on the opt-in path. T17, T20 and
    T21 pass `check` through and test both the default (`skipped`) and
    `check="gate"`.
  - The owner's original recommendation (on by default) is kept here for
    the record: the gate separates model faults from engine faults, and in
    1b an `unavailable` gate means no reference ran at all.
  - `main.py`'s adapter is not affected: it builds through `FixedPlanner`
    and has no gate.
- **D2, float64 energy on the `main.py` path (T03).**
  - Yes: `float64_energy=True` everywhere, `FixedPlanner` included, so
    `main.py --engine fastmace` writes float64-accumulated energies from
    the merge on, with `run.json` recording it.
  - Forces, stress and trajectories do not change.
  Reason: the float32 sum is wrong by up to 102 meV/atom for all-electron
  E0 models, and MH-1's pseudopotential E0 makes the change for the
  production label small. T03 measures it; a change below 0.01 meV/atom is
  expected. The alternative, keeping 1a's sum for `main.py` until the
  owner switches, leaves two energies in production.
- **D3, a species superset beyond the structure in 1b (T12).** Accept
  `species=` equal to the structure's species. Refuse a superset with
  absent species with "species superset beyond the structure: phase 3
  (§8.5)". Reason: species-dependent rewrites are built from the
  structure's species in 1a, and §8.5's identity-basis rows and sender
  groups are phase 3; accepting it now would give a plan the gate never
  covered. `ModelElementError` is checked either way.
- **D4, observer changes in `fastmace.ase` before phase 3 (T17, T21).**
  AGENTS.md has no in-place events until phase 3, while §7.7's 1b
  acceptance includes an observer test.
  - Positions and momenta changed by an observer become a host write at a
    canonical point: quiesce, write, rebuild the lists, re-evaluate. This
    is the same `Simulation.set_positions`/`set_momenta` the expert API
    gets, so it is correct and slow; the report says so.
  - Species, masses and charges raise until phase 3 events.
  - The §7.7 observer acceptance item then covers positions and momenta in
    1b; the rest moves to phase 3's acceptance, recorded as a SPEC §19
    amendment.
- **D5, `fastmace run` in 1b (T22).** Yes. `fastmace init` writes a TOML
  for `fastmace run`, which §16 does not list for 1b. Without it, `init` is
  useless. It is small: the TOML holds `fm.md`'s arguments.
- **D6, rules without their tiers (T12).** 1b has the graph tier only for
  MD.
  - Rule 4 (inexact padding): refuse with "compiled MD tier: phase 3".
  - Rule 9 (frequent host callbacks): keep the graph tier with a host
    sync at each callback, and say so in the report; it is a preference,
    so this is allowed.
  - `allow_reference=True` for non-periodic MD: refused in 1b. Reference
    MD needs its own loop over MACE's calculator, outside 1b's scope.
- **D7, `Simulation` without `workload=` (T17).** Default `Workload(max_temperature_K=<integrator temperature_K>)`,
  recorded in the plan; `run(schedule=…)` above it refuses as §7.6 says.
  Reason: §9.2 makes the field required, not the argument, and `fm.md`
  infers the same value; a missing workload would otherwise make the
  one-line expert example fail.
- **D8, BAOAB independent reference (T09).** ASE 3.27's `LangevinBAOAB`
  with the drag line patched, plus a float64 single-step check from the
  paper (§12.1 allows either; with both, the trajectory comparison is exact
  and the formula is checked independently of ASE's code).
- **Conductor (not owner): worker starts.** `plans/CONDUCTOR.md` suggested
  reconsidering a watcher script with an owner-written queue for 1b. The
  recommendation is to keep the ready list ("start all ready") for 1b. It
  worked through 1a with at most four workers, and a watcher would start
  sessions the classifier may treat as a bypass.

## Conductor answers to the prestart reports (19 October 2026)

All 21 waiting workers reported. These answers are binding. Each go message
points here.

**Changed dependencies:**
- **T06 also after T02:** it raises `RestartMismatch`.
- **T12 also after T05:** it holds T05's identity, with no placeholder.
- **T21 also after T20:** its `summary` test compares with `fm.md`.

**T01 contracts, the five points.**
- Points 1–4 are accepted as written: optional `atoms`; `force_field`;
  required `seed`; `mc=...` keeps the move set and `mc=None` detaches it.
- Point 5: `fm.md`'s friction default is `main.py`'s, 0.02 per ASE time
  unit (about 0.001965/fs). SPEC §7.7 is amended (sixth amendment).

### Shared API agreements

- **Reason codes (T02, T12, T15).**
  - T02 adds `Reason(str)` with `.rule` and a stable `.code`, and
    `PlanRefused.from_reasons(reasons)`. That function returns
    `SpeciesNotInSuperset` when a reason has `rule == 2`.
  - T12 raises refusals only through it, and gives every `plan.decisions`
    entry a stable key.
  - T15 matches codes, never text.
- **Errors (T02).**
  - `gate_record` holds the object and `record` the location.
  - The `doctor` line is on by default for the five environment errors.
  - `PolarUnavailableError`, `CacheLocationError` and `BaselineExportError`
    move into `errors.py`, with re-exports where they were.
  - `OEqRuntimeError` becomes `class OEqRuntimeError(FastMACEError,
    RuntimeError)` in `errors.py`, re-exported from
    `kernels/_oeq_runtime.py`.
  - T02 edits only the re-export lines of those files. "Three modules"
    means all four.
- **Restart content (T05, T06, T07, T14, T17).**
  - T05: `ForceFieldIdentity` goes in `state.py`. `plan=` is a JSON
    mapping until T12's `Plan.to_restart()` exists.
  - T07: `state.protocol` is an object with `to_restart() -> dict` and
    `from_restart`. The dict is JSON-safe and holds the segments, the
    resolved timestep, the step and the target.
  - T06: T05 stores T06's `output_chunk`, `output_rows` and
    `output_frames`.
  - T14: T05 also stores the edge bucket and the neighbour capacity under
    the plan record (§10.6, "edge and capacity buckets"). T14 builds a
    resumed engine with them, and may add the capacity option to
    `neighbors/skin.py` after T04 merges.
  - Versions stay in the identity digest, as SPEC §10.6 says. In 1b, resume
    refuses another FastMACE or MACE version (listed for the owner below).
  - `main.py` files are version 2 without `identity`, and
    `Simulation.resume` refuses them.
  - T17 writes `simulation.json` next to the restart, for what T05's format
    does not hold:
    - the model source: label or path, head, SHA-256 and trust;
    - the integrator record, workload, `Output`, seed and
      `allow_naive_cueq`;
    - species, masses and PBC.
- **`output_steps`.** The canonical copy is in `fastmace.protocols` (T07).
  T06 re-exports it in `io/trajectory.py`.
- **TOML.** The conductor adds `tomli; python_version < "3.11"` to
  `pyproject.toml`. T07 and T22 use `tomllib`, falling back to `tomli`.
  PyYAML stays optional: it is imported lazily, only `safe_load` is used,
  and a missing PyYAML raises a clear `ImportError`.
- **Registry (T04, T16).**
  - The policy table lives in `registry.py`, keyed by attribute path. The
    test fails for any buffer without a policy.
  - The host mirrors are engine-owned `derived` buffers with
    `capture=False`. T04 adds that row to `contracts-1b.md`.
  - `p_eps` and `force_eps`: whichever of T04 and T05 merges second renames
    them.
  - For 1b, a sticky polar overflow flag is enough, and it is recorded.
  - `runtime_state` is removed.
  - T04 adds a process-wide `engine.base.production_engine_built()` record,
    set when a `GraphEngine` is built. T16 reads it.
- **Planner entry points (T12, T18).**
  - T12 provides `MinimalPlanner.plan_evaluation(...)`: compiled mode, no
    workload. `fm.calc` and `ForceField.evaluate` use it.
  - Rule 6 is checked when the model is plain cuEq: in `build`, or already
    in `plan` if the model is loaded.
  - Stress uses `effective_external_stress_GPa()` only if the test shows it
    is bitwise equal to 1a.
  - Species are compared as sets.
  - T12 may edit the note's wording in `engine/constraints.py`. If T11's
    docstring edit in `fixed.py` lands first, T12 rebases.
  - Publish the `Plan` shape in `docs/design/` on the branch, not in a
    notes folder.
- **Gate (T13, T16, T17).**
  - T13 exposes `record.id` and tests that it changes; the identity test
    comes with T17.
  - The record states 1a-T19's "stock" exactly: MACE's own cuEq conversion,
    conv fusion off, one-graph strain on.
  - T13 compares against the stored references of run
    `1a-T19-gate-leto-20261018`.
  - T09 exports its qualification results as a constant, and T13 reads that
    constant.
  - T16's proposal is accepted:
    - load the copy, rebuild its list, replay C then B, and read the state;
      the registry then restores it;
    - NVT checks energy and forces only, recording "stress: not in plan";
    - the child entry point is in `selfcheck.py`;
    - a monkeypatched broken kernel in the test is fine.
  - Signatures: `selfcheck.references(plan, force_field, atoms, integrator,
    *, check, seed)` and `selfcheck.production(engine, references, *,
    check) -> GateRecord`.
  - Until T16 merges, `sim.gate` is
    `{status: "skipped", reason: 'opt-in: check="gate"'}`.
- **Simulation (T14, T17, T21).**
  - T14 exposes `canonicalise(reevaluate=True)` for D4's host writes.
  - T17 provides `sim.committed()`: a read-only snapshot of positions,
    momenta, cell, energy, forces, stress and step. It also provides a hook
    that submits a frame to the async writer. T21 uses both.
  - `Simulation` takes a `LoadedModel`, a `ForceField` or a label. A path
    raises `TypeError` pointing to `fm.load`, because the trust decision
    needs one.
  - T17 may delete `test_simulation_stub_says_which_task_fills_it`.
  - `mc` is duck-typed.
  - Energy is checked at every thermo row, and at the end of `run()` when
    there is no output.
- **Report (T15, T20, T21).**
  - T15's object has `setup_text()`, `record_speed(...)`, `finish(mean)`
    and `to_metadata()`, and accepts extra warnings. T17 owns the step
    timing.
  - The speed window is given as arguments. The defaults are 1,000 warm-up
    steps, then 1,000 timed steps. The end-of-run wording is as proposed.
  - In 1b, the §4.3 model warnings are a data table in `report.py`.
  - Setup stages use §12.5's names.
  - The metadata key is `"report"`, written by T10's `write_metadata`.
- **Provenance (T08, T10).**
  - T10 writes the builder and `write_metadata`, and T17 and T20 call them.
  - Inputs are plain mappings with documented keys.
  - `main.py`'s layout stays, and the 1a block nests under `"model"`.
  - The self-check results are part of the gate record.
  - T10 adds `trust` to `model_block`, taken from
    `LoadedModel.provenance()`.
- **Loading (T03, T08).**
  - The internal `load(...)` keeps its signature.
  - Public `fm.load` returns a pristine `LoadedModel` (e3nn, float64, CPU),
    and the plan converts it.
  - A URL downloads only with `download=True`.
  - T03 may add the float64 E0 capture in `loading.py`. Whichever of T03
    and T08 merges second rebases.
- **Energy (T03).**
  - Design A is accepted, and SPEC §7.3 is amended.
  - T03 adds `float64_energy` to `MACEForceField` and the narrow
    pass-through in `planner/fixed.py`; T12 rebases.
  - The conductor adapts ResearchModule's adapter tolerance later, in a
    cross-repository change.

### Per task, the rest

- **T06.** All four proposals are accepted. Every restart write is a chunk
  boundary, the stop request included. An empty chunk still gets its files.
- **T07.**
  - Round half up, refuse 0-step segments and record the rounding.
  - The rate sign rule is as proposed.
  - Pre-relaxation and quench are data only.
- **T09.**
  - Test the installed ASE, with a source guard.
  - The calculator fallback is acceptable.
  - The GPU test runs in float64 and is marked for T24's slow tier.
- **T18.**
  - Always build with stress.
  - When the species or the atom count change, replan and reuse the
    compile cache, so a new species gets a fresh plan.
  - Keep `calc.plan` and `calc.notes`.
  - Until T13 is merged, compare against stock MACE in float64.
- **T19.**
  - Only `find_spec` for OEq, never an import.
  - Set `CUDA_CACHE_DISABLE=1` in the doctor process before CUDA starts.
  - Exit codes as proposed, plus `--no-hash`.
  - Provide a subcommand table. T22 is the second editor of
    `cli/__init__.py`.
- **T20.**
  - Draw and record the seed.
  - The output default is `./<structure stem>.fastmace/`, or
    `./fastmace-run/` for an in-memory `Atoms`; T21 uses the same.
  - "D3: phase 3b" is a `PlanRefused` raised in `fm.md` after the
    expansion. NVE raises `PlanRefused`.
  - An explicit `integrator=` wins. A conflicting non-default `ensemble`
    raises `ValueError`.
  - Skip the gate test if T16 is not merged.
- **T21.**
  - Refuse a non-default `comm` or `communicator`, and `master` (SPEC
    amended). Walk the whole class chain.
  - Refuse fixed-cell BAOAB ("use `Langevin`"), NVE/NPH ("phase 4") and an
    explicit `P_mass`.
  - `hydrostatic=False` maps to `"full"`.
  - Convert the stress shapes, and refuse a non-symmetric 3×3.
  - A 1-ulp difference in friction is acceptable: compare plans with that
    tolerance.
- **T22.**
  - Second editor of `cli/__init__.py`.
  - The `schedule`, `integrator` and `mc` tables as proposed.
  - `run` reads a TOML of `fm.md` arguments (D5).
  - The conductor writes the CI job; T22 documents the commands.
- **T23.**
  - A renamed `fastmace_1a` package in one process is accepted only if the
    tool handles the duplicated process-wide patches (the custom-pass slot,
    `mask_head`) the way `bench.ab` does. Otherwise use alternating
    processes and say so.
  - The tag `fastmace-1a` is set on `bb35363` by the conductor.
  - POLAR-1-S on NbCl5 stays supplementary.
  - The tool may go in `tools/` and `fastmace/bench/`.
  - Adapter pytest runs in ResearchModule with `PYTHONDONTWRITEBYTECODE=1`,
    `-p no:cacheprovider` and `--basetemp` outside the tree.
  - Machines as listed; the go message confirms the reserved list.

### After T06 and T12 (19 October 2026)

- **1b-T08b added.** 1b-T12 found that the public `fm.load` imports MACE
  before OpenEquivariance is prepared, so the minimal planner never picks
  the hybrid backend after it (`backend-cueq-mace-imported-first`). The fix
  goes in `model/loading.py`: `load_pristine` prepares OEq first when it is
  installed and CUDA is visible. The T08 session does it on branch
  `fastmace/1b-T08b`. It blocks acceptance item 4, not the other tasks.
- T12: each plan reloads and converts the checkpoint, because sharing one
  converted model across plans would break §8.7 isolation; this costs load
  time. A superset beyond the structure is `rule=8`,
  `superset-beyond-structure`. Explicit option conflicts carry `rule=None`.
  The GPU comparison against `FixedPlanner` (500 MH-1 steps) is left to
  1b-T23.
- T06: the chunks and the restart file are flushed but not fsynced. That
  covers SIGKILL, not power loss (phase 2 safety).

### After T09, T15, T19 (19 October 2026)

- T09: the engines match their corrected references to about 1e-13 Å on CPU
  and GPU (murphy); the two BAOAB variants differ by 6e-6 Å on the GPU run.
  The per-bin temperature limit (0.1 K) and the BAOAB position-ratio limit
  are T09's own thresholds.
- T15: no atom-count warning, because a fixed atom threshold would be a
  per-GPU rule (§9.1); only the small-cell warning (cell height below twice
  the cutoff). The report reads the gate keys `status`, `reason`, `copies`,
  `energy_meV_per_atom`, `force_eV_per_A` and `stress_eV_per_A3`; T16
  provides them.
- T19 went back to its worker: on the NAS environment, `doctor`'s import of
  `cuequivariance_ops_torch` created `~/.cache/cuequivariance-triton`, and
  `doctor` must change nothing.

### After T04, T13, T18 (19 October 2026)

- T13: the POLAR-1-S reference matches 1a-T19's stored float64 references
  to 5e-15 on murphy (the same hardware as leto, where they were made). The
  finite-difference self-test gives 6.9e-9.
- **T18 finding, a follow-up for the compiled tier:** `fm.calc` compiles a
  second time at its second call (4.5 s on plain cuEq, 19 s on hybrid,
  NaCl 2×2×2). The likely cause is `torch.compile` switching to dynamic
  shapes when the edge count changes. After that a call takes about 15 ms.
  Few-call tools (phonopy, NEB) mostly pay compile time. The fix (compile
  the evaluation plan dynamic from the first call) belongs in the planner
  or `model/compile.py`; it is phase 2 planner work unless 1b-T23 shows it
  matters for acceptance.

### After T14 (19 October 2026)

- Canonical restart: plain cuEq float64 resumes to 3.7e-13 Å (MH-1 NaCl
  3×3×3, 500 steps, resumed every 100). The hybrid is statistical: resumed
  minus uninterrupted is about 1e-5 K and 1e-5 meV/atom, against 2.7 K and
  0.48 meV/atom between seeds (RTX 5060 Ti).
- Conductor decisions: T14's narrow `reset_polar` edit in `model/forces.py`
  is accepted. `canonicalise()` clears the sticky polar overflow flag at a
  canonical point and records the step; it does not refuse the restart.
- Not covered: compiled PolarMACE NPT on the GPU, which may recompile if
  its k-vector set changes shape at a restart. That is for 1b-T23.

### After T16 (19 October 2026)

- The gate passes 46 CPU and 7 GPU tests (murphy). The registered state is
  bitwise unchanged after the gate. An E0 offset of 1 meV per Na atom fails
  on energy alone.
- Conductor decision: on NaCl 3×3×3 the 0.05 Å rattle pulls the 6.306 Å
  shell inside the 6.3 Å build radius, so the neighbour capacity grows
  during the gate. The gate keeps the larger capacity, recaptures, and
  records it in `GateRecord.notes`. That is accepted: the gate is opt-in,
  and the run's first thermal rebuild would grow it anyway. A gated plain
  cuEq float64 run therefore need not stay within 1e-12 Å of an ungated
  one on the GPU.

### After T17 (19 October 2026)

- `Simulation` is complete: plan, opt-in gate, chunked output, the report,
  `metadata.json`, canonical restart and resume, and host writes. Its GPU
  smoke test (60 steps of MH-1) passes on plain cuEq and hybrid, within
  float32 noise of `FixedPlanner`.
- Deviations accepted: a resumed run records its trust decision as
  `sha256` even when the original used `trusted=True`, because resume
  verifies the hash. Restart files are written at every multiple of
  `restart_every`, the last step of a run included (`main.py` skips the
  last step).
- **1b-T25 added** (19 October 2026): two fast-tier GPU tests from a
  session started by 1b-T24, on tiny models: no compile at a later edge
  bucket, and a planned engine equal to a hand-built one. New test files
  only.

### After T20, T25 (19 October 2026)

- T25: two tiny-model fast-tier GPU tests: no compile at a later edge
  bucket, and planned equals hand-built to 2.2e-15 Å. A fresh process costs
  about 25 s even with warm caches, so the 20 s target per test is not met.
- The fast GPU tier had grown to 27 min warm, against T24's 3.4 min,
  because GPU tests merged after T24 lack the slow mark.
  **1b-T26 added:** re-tier them (markers and tiny tests only).
- **Flaky test, unidentified:** the T20+T25 trial on murphy had 1 failure in
  1964 that two reruns did not repeat (both 1964 passed). The conductor's
  command chain merged anyway, because `| tail` hid pytest's exit status.
  Since then trial runs use `set -o pipefail` and `-rf`. 1b-T23 watches for
  a repeat.
- T21 is the second editor of `simulation.py`, for `Simulation.add_sink`
  only (approved). It lets the ASE shim write trajectory and log at
  `loginterval` without ending a `run()` chunk.

### After T21, T22, T23 tools, T26 (19 October 2026)

- Every 1b implementation task is merged. The trial passed in both CPU
  environments (murphy `mace_cueq_latest`: 2128 passed; local
  `fastmace_cpu`: 2081 passed).
- T26: the warm fast GPU tier went from 24.3 to 8.0 min (186 tests); 13
  tests were marked slow and 2 tiny tests added. Tiny versions of the
  others cost almost as much, because processes, compiles and child
  processes dominate, and two fail on a random model. Those behaviours are
  listed as full-tier only in `docs/development.md`.
- **Possible library issue for 1b-T23 to measure:** "MACE imported first"
  takes about 310 s even with a tiny model, against 10 s for the hybrid
  case. A user who imports MACE before FastMACE would pay this.

### During 1b-T23 (19 October 2026)

- Passing:
  - the `main.py` smoke run (`--engine gpu` against `--engine fastmace`,
    NaNbO3, 2000 steps): final temperature 161.48171 against 161.48177 K,
    with a 0.035 meV/atom energy offset from D2;
  - compiled PolarMACE NPT across five canonical restarts: no recompile,
    resumed to 2.1e-7 Å.
- 116 previously skipped GPU tests pass. **Three in `test_engine_gpu.py`
  fail and are assigned to 1b-T03b:** the plain cuEq float64 research
  comparisons (stage B's kernel sequence differs, likely the float64
  energy), and a hybrid capture crash ("stale reference to the default
  stream from warmup" in the float64-energy `ScaleShiftMACE.forward`)
  inside that test's eager warm-up check.
- Setup takes 4–10 s longer than 1a in single-version processes on NaCl
  (murphy 19.5 s against 15.3 s; local 24.0 s against 14.4 s). A screen
  candidate after 1b, beside 1a's setup finding.
- The local RTX 5060 Ti has no noise floor of its own (1a-T18 measured
  murphy only), and its per-block-pair ratios scatter widely (NaCl
  0.987–1.014).

### After 1b-T23 (19 October 2026)

- The acceptance report is `docs/validation/phase-1b.md` (merged). It
  passes the gate with energy on all 19 labels, the in-library gate (19
  NVT, 18 NPT), the full GPU tier on murphy and the RTX 5060 Ti (176 each),
  restart, `doctor`/`md`, and compatibility with `main.py`.
- Its failure 4 (the research-engine comparisons) is resolved by 1b-T03b,
  merged:
  - the kernel lists differ because the float64 reduction changes
    Inductor's fusion, so those comparisons run with
    `float64_energy=False`;
  - the hybrid crash was the test's warm-up clones of a leaf that requires
    grad, fixed with `detach()`; no library change.
  Results: NVT 2.5e-13 Å, NPT 7.1e-15 Å, hybrid 1.0e-7 Å against a run
  scatter of 8.7e-8.
- Failure 2 (the ASE shim costs about 0.26 s per `run()` call; per step it
  is equal) goes to **1b-T21b**. A wrong "naive cuEq" warning from
  `fastmace md` on hybrid runs goes to **1b-T22b**.
- Failures 1 (POLAR-1-S NPT stress at the floor) and 3 (NaCl timing in
  both directions) are owner items 5 and 6.

### After T21b, T22b (19 October 2026)

- T21b: the shim no longer builds a second metadata record per `run()`
  call. Ratio to `Simulation` with 300-step calls on the RTX 5060 Ti: 1.015
  → 1.000 (ext4), 1.90 → 0.986 (output on `/mnt/d`). Still open for
  phase 2: `Simulation.run` itself rebuilds provenance at every call.
- T22b: the planner's rule 6 record used the code `allow-naive-cueq` on
  every run, and the report read it as a downgrade. Now it reads
  `fused-cueq` when the fused kernels load. Confirmed by a hybrid
  `fastmace md` run on the RTX 5060 Ti.

### For the owner's review (not blocking)

1. `fm.md`'s friction default follows `main.py` (ASE units), not SPEC's
   former "0.02/fs".
2. Resume refuses a different FastMACE or MACE version, reading SPEC §10.6
   literally. A later relaxation could allow version changes with a
   recorded warning.
3. Public `fm.load` returns a pristine model, and the planner chooses the
   backend.
4. **D2 measured (1b-T03, merged 19 October 2026).** MH-1 r²SCAN's
   production energy changes by 0.12 meV/atom on NbCl5 3×3×3 (0.236 eV in
   total), and by 0.005 meV/atom on NaCl. D2 expected about 0.01 meV/atom.
   The change is the removed float32 summation error: against the float64
   reference, the float64 sum is off by 0.00014 meV/atom, the 1a sum by
   0.122. On NbCl5 3×3×3 the POLAR-1-S and OMOL-0 XL sums differ by 83 and
   154 meV/atom; OMOL's is above the 45–102 quoted in SPEC draft 0.3. The
   float64 references of both run out of memory at 3×3×3, so they were
   checked at 2×2×2, where the float64 sum is within 0.0014 meV/atom. Step
   time is unchanged (NaCl MH-1 hybrid, 1.003). ResearchModule's
   `main.py --engine fastmace` energies move by these amounts.
5. **POLAR-1-S NPT fails the in-library gate on stress, at the floor
   (1b-T23).** NaCl: the production stress error is 1.18e-7 eV/Å³, the
   float32 yardstick's 3.89e-8, so the limit is the 1e-7 floor (owner
   decision 7). The NVT half passes. The external 19-label driver passes the
   same model at ratio 1.57 with 3 configurations per cell mode. The
   production error matches 1a's production-versus-finite-difference value.
   Options: keep the floor (the gate refuses PolarMACE NPT, which is
   experimental anyway); raise the stress floor for polar models; or use
   more copies in the gate. Not changed by the conductor.
6. **NaCl timing, 1b against 1a (1b-T23):** 0.5 % faster on murphy and
   0.3 % slower on the RTX 5060 Ti, consistently within each machine, so
   outside the 0.16 % noise floor in both directions. NbCl5 is within
   noise on both. Most likely not a 1b regression; the verdict is the
   owner's.

## 1a items deferred to 1b

| Item (where decided) | 1b task |
| --- | --- |
| Exceptions into `errors.py` with the §7.6 hierarchy (after wave 1, T05 Q1) | T02 |
| `"models"` in `_cache.CACHE_KINDS`, with the trust rule (after wave 1, T05 Q3) | T08 |
| Head functionals of MH-1's `spice_wB97M`, `oc20_usemppbe`, `omol` checked before the dispersion rule (after wave 1, T05 Q6) | T08 (check), T20 (rule) |
| `p_eps` and `force_eps` in `SimulationState` (after wave 0, interfaces) | T01 (field), T05 |
| `externalstress` and `disable_cell_langevin` in `IntegratorConfig` (after T07b/T12) | T01 (fields), T05 (integrator), T12 (planner) |
| `float64_energy` config field (after wave 0, switch map Q7) | T01 (field), T03 |
| Schema-versioned provenance; additions under `"fastmace"` until then (after wave 1b) | T10 |
| `output_steps` into `fastmace.protocols` (after wave 1b) | T07 |
| Public `fm.ForceField` as a factory over `MACEForceField` (after wave 0, interfaces) | T12 |
| The runtime-state registry seeded by `runtime_state` (interfaces; T13 seams) | T04 |
| Re-capture and overflow-before-commit seams (T13) | T04 (flags only); the policies are phase 3 |
| The gate driver's same-process lessons: references after production (second outside review); polar references correct by accident (after T17); OMOL-0 scatter (T19) | T13, T16, T23 |
| CPU float64 references through naive cuEq (after T11, done in T15) | used by T13 |

## 1a open findings

| Finding (`docs/validation/phase-1a.md`) | Disposition |
| --- | --- |
| MP-0 medium 1.4–1.6 % slower than the research engine | **1b task T11** (parity diagnosis; fix only if it restores the research engine's numerics), else a screen after 1b |
| Setup 4–33 s longer (self-checks, per-process OEq build) | **1b task T11** measures the stages and T15 reports them. Speed-ups go **after 1b**: cached qualification keyed by checkpoint hash and software versions (general, not shape-keyed), and the persistent OEq extension cache of 1a's decision after wave 0. Phase 6 removes the build with prebuilt OEq wheels |
| coraline soft-lock check | **Later, opportunistic**: one FastMACE-only control (`tools/acceptance/launch.sh speed --machine coraline --only control-mh1-nvt-fastmace`) as soon as coraline's GPU is idle, before any FastMACE production there. Not a 1b acceptance item. The conductor checks at each wave start |
| PolarMACE NbCl5 3×3×3 stress against finite differences: 4.4e-4 eV/Å³ | **Later, phase 3b** (before PolarMACE NPT is recommended). In 1b, T13's record states the finite-difference validation on NaCl 3×3×3 only, and T15's report carries the "experimental" line for PolarMACE NPT |
| New-optimisation screens (folded species embedding; `inplace_block_linear` timing) | **After 1b**: screens need the gate with energy (T16, T23) and should not move the 1b no-regression baseline. They may start in a free GPU slot once T16 merges, outside 1b's acceptance, under 1a's screen rules |

## Deviations from SPEC wording, for the conductor's review

- `fastmace/simulation.py` holds `Simulation`. §5 lists no module for it.
- `planner/workload.py` holds `Workload`. §5 calls it `WorkloadSpec`;
  §7.2 and §9.2 say `Workload`.
- The minimal planner is `planner/minimal.py`, next to 1a's `fixed.py`.
  §5 puts the rules in `plan.py`, but `fixed.py` must stay importable for
  `main.py`.
- `model/forcefield.py` holds the public `fm.ForceField`. §5's `forces.py`
  is 1a's `MACEForceField` execution copy.
- The independent BAOAB reference lives in `fastmace/reference/integrators.py`.

## Risks specific to 1b

| Risk | Mitigation |
| --- | --- |
| The float64 reduction needs MACE's forward source replaced, and a later MACE changes it | source-hash pin like the rewrites; refusal with a reason; T03 tests per family |
| Float64 energy costs step time | T03 A/B against main; acceptance 5 against the 1a tag |
| The in-library gate's references are disturbed by process state (1a "after T10") | references first; refused in-process after a production engine; in-process against fresh-process equality test (T16) |
| A gate that is slow or runs out of memory makes the default unusable | references one at a time and freed; time in the report; `unavailable` handled by D1 |
| Canonical restart points change trajectories compared to 1a (an extra rebuild at each restart) | defined behaviour (§10.6); `main.py` keeps 1a's restart path until its adapter moves; T14 documents the difference |
| Moving 1a names breaks `main.py`, the production default since 19 October | research-surface test (T01); re-exports; acceptance 6 |
| `fastmace.ase` diverges from ASE in a way scripts notice | signature tests against ASE 3.27; trajectory and log read-back; refusal instead of reinterpretation |
| Wave 1 is wide and touches shared files (`forces.py`, `polar.py`, `graph.py`, `integrators/`) | one owner per file with a named second editor after merge; the conductor merges T03 before T04's edits to those files |
| GPU contention with other users on coraline and leeloo | ours only: RTX 5060 Ti, leto, murphy; idle checks; one GPU per task |
