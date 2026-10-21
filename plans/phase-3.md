# Phase 3: dynamic topology and Monte Carlo

Status: **draft for the owner**, 22 October 2026. Spec: [SPEC.md](../SPEC.md)
§16 row "3. Dynamic topology and MC", and the sections it cites:
- §8.1: overflow before commit, the re-capture policy, the energy-only graph;
- §8.2: the compiled tier;
- §8.5: the species superset, the derived-state registry, the transaction contract and random numbers per outcome;
- §9.2–§9.4: `Workload`, rules 2, 8, 9 and 11, and the MC energy-precision dimension of the fallback;
- §10.4: Monte Carlo;
- §10.5: the swap-MC annealing protocol;
- §10.6: MC restart state;
- §12.2: the mutation-event, transition, exact-enumeration and restart test classes;
- §12.3: the move-energy gate.

It also carries phase 2's leftovers and the owner's decisions of
21–22 October 2026 (1 fs production timestep, cloud GPUs allowed).

Requires phase 2 and optimisation rounds 1–3 merged. Phase 2 is done and
tagged `fastmace-2` (e52c482). The round-2 defaults are on main (072fa0a).
Round 3 is on main (5c48e26). The `fused_integrator` default (OPT3-T03-default)
is merged before the phase 3 go messages.

"Source" names the FastMACE code a task builds on, or a ResearchModule path
(`G:\Margraf\ResearchModule`, read-only except in the cross-repository
task). Research-engine paths are relative to `Experiments/Langevin_Thermostat/`.

## Goal

Phases 1–2 run one fixed composition at graph speed, safely. Phase 3 makes
the composition **change in place** without leaving the fast tier:
- a declared **species superset** is built into every species-indexed
  table, so an atom can become any superset species. Absent species work
  too, built from the identity basis and never from "the first atom of that
  species";
- every in-place change (a species swap, a transmutation, a mass or charge
  change, a displacement, a friction change) is one **transaction**. The
  transaction quiesces, snapshots, applies, refreshes every derived buffer,
  evaluates, then accepts or restores exactly, and resumes with consistent
  forces;
- a capacity **overflow never commits** anything. The state before it is
  restored, the graphs are re-captured with a larger bucket, and the step or
  attempt is redone with the same random numbers;
- **Monte Carlo** (swap, transmutation, displacement) runs in captured graphs
  where it can. It has device-side selection, acceptance and counters, the
  momentum treatments of §10.4, and a move-energy gate that decides float32
  or float64 MC energies;
- a **compiled tier** (no CUDA graphs) joins the planner's mode dimension,
  for inexact padding, frequent host callbacks and frequent re-captures;
- MC state survives a restart, and a swap-MC annealing protocol runs end to
  end.

Kept out (later phases or 1.x):
- atom-count changes: insertion and deletion, and ghost-atom buckets (§8.6;
  rule 7 keeps refusing);
- local-energy MC updates (§10.4, 1.x);
- runtime adaptation between segments (§9.6, phase 5). A frequent
  re-capture downgrades to the compiled tier before the first step or at a
  canonical point, never silently mid-segment;
- D3 and NPT for the chlorides (phase 3b), apart from the transaction
  contract's NPT parts (`force_eps` refresh after an accepted move), which
  §8.5 puts here.

## Acceptance (SPEC §16 row 3, made concrete)

1. **Event speed.** With MH-1 hybrid float32 on NaCl and NbCl5 3×3×3, at a
   1500 K target, species-change events every 100 steps keep at least 90 % of
   the fixed-species step rate. The events are swaps of two atoms' species
   with `exchange` momenta and no energy evaluation. Measured in a
   same-process A/B, 12 blocks, both orders.
2. **MC attempt cost.** Reported separately, as a fraction of one MD step:
   - swaps in float32 and float64 MC energy precision;
   - displacements with and without a trial rebuild.
   This is a measurement with no pass mark (§10.4 "Cost").
3. **ΔE precision.** float32 forward ΔE errors against float64, measured on
   MH-1 (`matpes_r2scan`) and on an all-electron model (MH-1 `omol` head,
   and OMOL-0 XL where it fits). Swaps, transmutations to every superset
   species and displacements, on NaCl, NbCl5 and NaNbO3. The result decides
   what `mc_energy_precision="auto"` picks for those cases, and the result
   is recorded.
4. **Fresh-construction equality.** The first MD step after an accepted and
   after a rejected event equals a fresh construction from the committed
   state. That holds bitwise on CPU float64 and in reproducible mode, and
   within the gate on the hybrid. It is checked for swaps, transmutations,
   displacements (rebuild and no rebuild), mass and charge changes and a
   friction change, and NPT (`force_eps`) in hydrostatic and full-cell mode.
5. **Exact enumeration** (§12.2) passes on CPU (tiny model) and GPU:
   - 70 swap configurations and 256 transmutation configurations;
   - acceptance probabilities against the reference ΔE, including the first
     attempt after MD;
   - a χ² test of the populations;
   - the kinetic temperature with MC on and off, for each momentum choice;
   - zero total momentum under `fixcm`;
   - a negative control that must fail;
   - the charge/spin parity refusal.
6. **Restart across MC.** An interrupted run equals an uninterrupted one at
   the same cadence across accepted and rejected swaps and transmutations, a
   displacement with a trial rebuild, and a capacity-overflow redo.
   Bitwise on CPU float64 and in reproducible mode; statistically for the
   hybrid.
7. **Overflow and re-capture.** A forced edge-bucket overflow in an MD step
   and in an MC attempt is redone after re-capture from the state before it.
   No frame, thermo row, restart or acceptance comes from the overflowed
   evaluation. The re-capture count is logged and the variant cap is honoured.
8. **Compiled tier.** `mode="compiled"` runs MD for MH-1 and a generic
   model:
   - it equals graph mode bitwise on CPU float64, and within float32 noise
     on the GPU;
   - it is planned for inexact padding (rule 4) and frequent host callbacks
     (rule 9);
   - it joins the fallback's mode dimension: graph → compiled → eager →
     reference.
9. **Swap-MC annealing protocol** (owner decision D3) runs end to end on its
   crystal at 1 fs. That means MD with swap blocks, the move-energy gate
   passing, the restart, the quench, and the attempt and acceptance counters
   in the output.
10. **No regression.** On a run without species changes or MC, default plans
    and steady-state step times equal those of main at the phase 3 freeze. MH-1 NaCl
    and NbCl5 hybrid NVT, same-process A/B, 12 blocks, judged against each
    machine's noise floor. Trajectories are unchanged.
11. **Carry-overs:**
    - `fastmace.ase` observers apply species, mass and charge changes as
      events (decision 17; the §7.7 observer item);
    - ResearchModule `main.py` runs production at 1 fs (owner, 22 October
      2026);
    - a fallback's reason names the innermost compile error (2-T21d note);
    - the decisions on the old adapter and on cloud production (D1, D2).

Each item runs on the RTX 5060 Ti and one Ada GPU. For the Ada GPU see D4:
leeloo or murphy when the workstations are back and idle, or the Modal L4.

## Working principles

These are phase 2's principles, unchanged where they still apply:
- **Contracts first.** 3-T01 fixes the new config fields, public names,
  stubs and the merge order of shared files. Like 2-T01, it is the only
  task that edits `config.py`.
- **One owner per file**, listed per task, with second editors named.
- **Same trajectories without MC.** No phase 3 task changes a default plan,
  force, stress or trajectory of a run without species changes, MC or
  observers. That holds bitwise on CPU float64 against main, and a test
  shows the new paths are absent from the default path.
- **Transactions only.** State changes outside an integrator step go
  through `events` (SPEC §5 layering). No component edits engine buffers
  directly, and no event changes a shape or an address inside a captured
  graph.
- **Fault injection lives in tests only** (monkeypatched capacities,
  kernels or captures), never as a hook in the library.
- **Exactness by construction, then by test.** Each derived buffer
  registers its refresh function. A CI test compares the whole registry with
  a fresh construction, so a forgotten buffer fails.
- **General, not tuned.** The skin and headroom rule (rule 11), the
  re-capture budget and the MC defaults are general rules computed from the
  workload. There are no per-GPU or per-crystal tables.
- **Tests first,** in the right GPU tier (fast, slow, acceptance), checked
  by `tools/test_budget.py`.

### Resource rules (learned in phase 2 and rounds 1–3)

- **Prestarted workers** (`CONDUCTOR.md`): every task starts at once,
  prepares read-only, reports "prepared, waiting" with its questions, and
  starts on the conductor's go. At most five workers are active, one GPU
  job per reservation.
- **Local RTX 5060 Ti (WSL):**
  - WSL has 40 GB RAM since 21 October 2026 (`.wslconfig`);
  - check `free -g` before every job, and wait if less than 6 GiB is
    available;
  - workers run only targeted test files at `-n 2`, niced; full suites are
    the conductor's (trial merges);
  - the GPU goes through `tools/ci/local_gpu.py` reservations of at most
    15 minutes (the queue deadlock is fixed in a780878);
  - launch detached (`setsid -f`).
- **Workstations:**
  - none from Thursday 22 October 2026 evening until the owner says they're
    back (power outage on Friday 23 October);
  - before that: leeloo (RTX 4000 Ada) and murphy only when no other user's
    process is on the GPU;
  - files only under `/mnt/homes/zwingel`.
- **Modal** (`tools/modal/fastmace_modal.py`; owner, 21–22 October 2026):
  - L4 (sm_89, the stand-in for the Ada workstations), A100 and H100;
  - pinned source archives;
  - every job has a timeout and a cost estimate in its `modal.json`;
  - **hard cap: the owner's free $30 credit per calendar month** (owner,
    22 October 2026), everything Modal meters included. `modal billing
    summary` is the truth. The GPU list price alone underestimates it: on
    21 October it showed $3.65 against about $3 estimated. The guard in
    `fastmace_modal.py` (MODAL-T02) refuses a job whose worst case would
    pass $27 (a $3 reserve). Until it is merged, the conductor approves each
    Modal job;
  - budget per task as in its brief;
  - the noise floors are 0.2 % for NaCl and 0.6 % for NbCl5 on the L4, and
    only same-container ratios count.
- **Timing** only on an otherwise idle machine, against that machine's own
  noise floor: local 1.1 % NaCl and 0.4 % NbCl5; murphy 0.16 %; L4 as above.
- **Trial merges** run on the local PC (`trial_merge_local.sh`, `-n 6`) while the
  workstations are off, or on murphy's CPUs when they are on.

## What exists today (survey of main 5c48e26, 22 October 2026)

The code cites the 1b plan's decision IDs: D3 = SPEC decision 16, D4 = 17,
D6 = 19.

**Superset:**
- `ForceField(species=)`, `ModelElementError` and the rule-2 refusal exist
  (`model/forcefield.py`, `planner/minimal.py` 305–313).
- A superset beyond the structure is refused (`superset-beyond-structure`).
- `node_attrs` is built once at setup at full z-table width, so the
  superset columns already exist, but they aren't registered.
- Fixed at setup from the species *present*:
  - `species_columns` (the compile key);
  - the reassociation columns;
  - the species-skip columns;
  - sender groups (`max()+1`);
  - `node_e0`.
- The §8.5 "first atom of each species" bug is live at
  `rewrites/mh1/modules.py:155`.
- The OPT-T08 `species_tables` already use all element rows, so they are
  superset-safe.

**Registries and events:**
- The runtime-state registry exists (`engine/registry.py`).
- There is no derived-state registry and no `fastmace/events.py`.
- Not registered:
  - Langevin `c3/c4/c5_unit`;
  - BAOAB `inv_sqrt_masses`;
  - the fused-integrator constants;
  - `node_attrs`.
- Host writes of positions and momenta go through
  `canonicalise(reevaluate=True)`.
- `fastmace.ase` observers refuse species, mass and charge changes
  (`ase/md.py:413`).

**Overflow and capture:**
- Capacities grow only on a host rebuild, with lazy re-capture.
- The pipelined and speculative redo paths exist.
- Missing: a re-capture log, budget, variant cap, and wired provenance
  counters (`counters.recaptures` and `overflows_redone` are defined but
  unused).
- The polar k-vector overflow flag is read only at canonical points, not per
  step.

**Skin:** fixed at 0.3 Å. Rule 11 is only recorded. The fused integrator
bakes the skin thresholds in.

**Tiers:**
- Compiled MD is refused ("phase 3", D6).
- A compiled path exists only for one-off evaluation (`fm.calc`,
  `EngineConfig.dynamic_edges`).
- The fallback's mode dimension is graph → eager → reference.

**Energy and MC:**
- There is no energy-only graph: `_outputs` always computes forces.
- Float64 accumulation exists (`model/energy.py`).
- `Workload.mc_energy_precision` is never read.
- `fastmace/mc/` is empty. The `Workload` MC fields and the `run(mc=…)`
  checks exist, ending in "MC: phase 3" refusals in `simulation.py`,
  `quick.py`, `cli/run.py`, `provenance.py` and `report.py`.

**Restart:**
- The format is version 2.
- Resume rebuilds `Atoms` from the *initial* structure, and `load_state`
  refuses other species. After an accepted swap that is a blocker.

**Tests:** only `tests/cpu` and `tests/gpu`. Ten tests pin phase 3 refusals
and will flip; they are named in the tasks below.

## Tasks

Each task gives: ID, size (S < ½ day, M ≈ 1 day, L 2–4 days),
dependencies, spec sections, source, files owned, tests first, and
done-when. A task whose GPU part is named runs its CPU tests first and then
asks the conductor for that GPU.

### Wave 0: contracts

#### 3-T01 Contracts: names, fields, stubs, merge order (M) — after the `fused_integrator` default merges

Spec: §7.2, §7.6, §8.1, §8.5, §9.2, §10.4, §10.6.

- **`config.py`** (only editor in phase 3):
  - neighbour fields for rule 11 (`skin="auto" | float`, headroom policy);
  - re-capture fields (budget per run, variant cap ≤ 64);
  - `mode="compiled"` admitted for MD;
  - MC energy precision plumbing;
  - the MC driver's options (block size limits and stream seeds derived
    from `seed`).
- **Public names:**
  - `fm.mc.Swap(pairs, every, attempts, velocities="exchange")`;
  - `fm.mc.Transmute(species, every, attempts, velocities, mu, formal_charges)`;
  - `fm.mc.Displace(delta_A, every, attempts, adaptive=False)`;
  - `fm.mc.configurational_mu(...)`;
  - the host mutation API on `Simulation`: `set_species`, `set_masses`,
    `set_charges`, `set_friction`, beside `set_positions`/`set_momenta`.
- **Stubs** with signatures:
  - `fastmace/events.py`: `Event` kinds, `Transaction`, and
    `DerivedStateRegistry.register(name, buffer, refresh, depends_on)`;
  - `engine/compiled.py`;
  - `fastmace/mc/moves.py` and `fastmace/mc/driver.py`;
  - the engine's energy-only entry point and a public `captured(key, fn)`
    for other modules' graphs;
  - restart schema version 3 (current species, masses and charges; the MC
    record).
- **Errors** (`errors.py`, new classes only): a parity refusal for
  transmutable sets, reported as a `PlanRefused` reason. Overflows stay
  internal.
- **`docs/design/contracts-3.md`:** names, fields, the event lifecycle, the
  restart v3 keys, which of the ten refusal-pinning tests each task flips,
  and the merge order of the shared files (below).

Tests first: a contract test pinning signatures and fields, and the
research-surface test unchanged.

Done when both CPU suites and ruff are green, and every stub raises
`NotImplementedError` naming its task.

### Wave 1: foundations (parallel after T01)

#### 3-T02 Species superset in the force field (L, GPU: local, short) — after T01

Spec: §8.5 (superset), §9.3 rules 2 and 8. Source: `model/forces.py`
276–426, `rewrites/mh1/__init__.py` 306–309, `rewrites/mh1/modules.py`
153–155, `rewrites/generic/__init__.py` 81–84, `neighbors/skin.py` 98 and
247–249.

- **Build every species-indexed buffer for the superset, never from the
  atoms present:**
  - the compile-key columns;
  - the reassociation `[S, C]` table, from the node embedding (plus OMOL-0's
    per-structure joint embedding) applied to the identity basis. This fixes
    `modules.py:155`;
  - the species-skip columns;
  - the sender groups, with a fixed count equal to the superset size;
  - `node_e0`;
  - `node_attrs`.
- **Register each one** with its refresh function in T03's derived-state
  registry, through the T01 contract.
- **Lift the `superset-beyond-structure` refusal** (rule 8: graph mode,
  rewrites built for the superset). Keep rule 2 and `ModelElementError`.
- **Membership checks** at setup and on every host mutation: an atomic
  number outside the superset, or a one-hot row that doesn't sum to 1,
  raises `SpeciesNotInSuperset`.
- **Measure the cost** of a superset larger than the structure (SPEC §18
  risk row): step time and memory for NaCl with superset {Na, Cl, Nb, O}
  against {Na, Cl}.

Files: `model/forces.py` (owner in phase 3; T07 second editor), `rewrites/mh1`
and `rewrites/generic` (the parts named), `neighbors/skin.py` (the
sender-group part; second editor, before T05), `planner/minimal.py` (the
rule 8 part; merge order below).

Tests first:
- a superset species with no atoms gets its table row from the identity
  basis, not from atom 0;
- with the last superset species absent, the sender-group count is still
  the superset size;
- an atom outside the superset raises;
- after a host species change, energies and forces equal a fresh
  construction (CPU float64 bitwise; GPU within float32 noise);
- flips `test_minimal_planner.py:155-160`.

Done when a structure can gain any superset species without a new
`ForceField`, and the default plan (superset = structure) is unchanged
bitwise.

#### 3-T03 Derived-state registry and the transaction contract (L, GPU: local, short) — after T01

Spec: §8.5 (registry, transaction, random numbers per outcome), §8.1
(graph-safe events). Source: `engine/registry.py`, `engine/graph.py`
310–352 (`canonicalise`), `simulation.py` 989–1031 (host writes),
`integrators/langevin.py` 68–97, `integrators/baoab.py` 124–184,
`kernels/triton/fused_integrator.py` 203–235.

- **`fastmace/events.py`** (owner): the lifecycle of §8.5.
  1. Quiesce the pipeline, async output and speculative steps.
  2. Snapshot the affected state and the random-number cursors.
  3. Apply.
  4. Refresh every registered derived buffer.
  5. Evaluate (a hook; T07 and T09 fill it).
  6. Accept or restore exactly. Restored forces are restored, never
     recomputed.
  7. Resume in the fixed order: species and momenta, then forces and
     stress, then integrator caches derived from them (BAOAB `force_eps`).
- **Register the integrator buffers:**
  - Langevin `c3/c4/c5_unit`;
  - BAOAB `inv_sqrt_masses` and `force_eps`;
  - the fused-integrator constants.
  A friction change alters `c1`/`c2`, which are baked into the captured
  graphs as Python floats, so that event drops the affected graphs (a
  recorded re-capture).
- **The host mutation API on `Simulation`:** `set_species`, `set_masses`,
  `set_charges`, `set_friction`, each as one event.
- **Ramps stay device-scalar fills**, never events.

Files: `fastmace/events.py` (new, owner), `engine/registry.py` (the derived
policy), `integrators/langevin.py`, `integrators/baoab.py` and
`kernels/triton/fused_integrator.py` (registration only), `simulation.py`
(the mutation API; first of the merge order below), `engine/graph.py`
(quiesce hook; second editor after T04).

Tests first:
- the registry enumerated against a fresh construction, so a forgotten
  buffer fails;
- the first MD step after an accepted and after a rejected species, mass,
  charge or friction event equals a fresh construction (CPU float64 bitwise;
  reproducible mode on the GPU; hybrid within the gate);
- the first NPT step after an accepted move equals a fresh construction
  (`force_eps`), in hydrostatic and full-cell mode;
- a ramp step changes the target without an event.

Done when every in-place change goes through `events`, and the CI registry
check is in the CPU suite.

#### 3-T04 Overflow before commit and the re-capture policy (M, GPU: local, short) — after T01

Spec: §8.1 (re-capture, overflow before commit). Source: `engine/graph.py`
44–47, 412–575, 658–743, `model/forces.py` 618–651 (the polar flag),
`provenance.py` 446–488.

- **Check before commit:** every captured stage that can overflow (edges, and
  polar k-vectors *per step*, which today is read only at canonical points)
  sets a device flag. The host checks it before the step is committed
  (`state.step += 1`).
- **Redo:** an overflowed step is redone after re-capture from the state
  before it. The engine exposes the same restore-and-redo to MC attempts
  (with the same random numbers), for T09 and T10.
- **Re-capture policy:**
  - a log line per re-capture;
  - the budget and the variant cap from T01;
  - wired provenance counters (`recaptures`, `overflows_redone`).
- **When the budget is exceeded:** warn, record, and recommend the compiled
  tier in the report. The automatic mid-run downgrade is phase 5's runtime
  adaptation (§9.6); see "Conductor decisions" below.

Files: `engine/graph.py` (owner in phase 3; T03, T07 and T10 second editors
in that order), `provenance.py` (counters; before T15).

Tests first:
- a forced edge-bucket overflow (monkeypatched capacity) in an MD step is
  redone, and no frame, thermo row or restart comes from the overflowed
  evaluation;
- a forced polar k-vector overflow is caught per step;
- the variant cap and the counters work.

Done when no observable can come from an overflowed evaluation, with tests.

#### 3-T05 Adaptive skin and headroom, rule 11 (M, GPU: local timing, Ada via Modal ≤ $2) — after T01, T02's sender-group part

Spec: §9.3 rule 11, §10.4 (displacement skin), §8.1 (buckets). Source:
`neighbors/skin.py` 90–366, `neighbors/buckets.py`, `planner/minimal.py`
560–563, `kernels/triton/fused_integrator.py` 207–209.

- **Skin and edge-bucket headroom** follow a general rule from
  `max_temperature_K`, the lightest mass and the timestep: the expected
  displacement per rebuild interval. No tables.
- **Displacement workloads:** a skin that keeps trial rebuilds rare
  (≥ 2δ plus the MD drift).
- **The re-capture budget** of T04.
- **The fused integrator's thresholds** follow the planned skin (registered
  constants).

Files: `neighbors/skin.py` and `neighbors/buckets.py` (owner),
`planner/minimal.py` (the rule 11 part).

Tests first:
- the rule's values for a few workloads;
- the default plan at 300 K is unchanged. Today's 0.3 Å is the rule's
  lower bound, or the rule is off without MC and high T. Pick one and state
  it;
- flips `test_minimal_planner.py:245-250`.

GPU:
- rebuild frequency and step time at 1500 K for NaCl and NbCl5, fixed
  against adaptive skin;
- local A/B in both orders;
- one Modal L4 confirmation under the guard.

Done when rule 11 decides skin and headroom, and the 1500 K numbers are in
`docs/validation/3-T05.md`.

#### 3-T06 Compiled MD tier (L, GPU: local) — after T01

Spec: §8.2, §9.3 rules 4 and 9, §9.4 (mode dimension). Source: the one-off
compiled path (`planner/fixed.py` 404 and 502, `planner/minimal.py` 579–600
and 763+, `EngineConfig.dynamic_edges`, `ase/calculator.py`), the eager mode
of 2-T02.

- **`engine/compiled.py`:** MD on `torch.compile` (default mode) without
  CUDA graphs, with bucketed or dynamic edge shapes and the same integrators,
  output, restart and events.
- **Planner:**
  - `mode="compiled"` for MD;
  - rule 4: inexact padding goes to compiled with exact edge counts;
  - rule 9: small `host_callbacks_every` goes to compiled;
  - the fallback's mode dimension becomes graph → compiled → eager →
    reference.
- **The report:** "compiled tier: no CUDA graphs" with the reason.

Files: `engine/compiled.py` (new, owner), `planner/minimal.py` (owner of the
file in phase 3; merge order below), `planner/fallback.py` (the mode
dimension; before T11 and T17).

Tests first:
- compiled equals graph bitwise on CPU float64 (tiny model, 50 steps);
- on the GPU, within float32 noise for MH-1 hybrid and a generic model
  (MP-0) on NaCl;
- an inexact-padding model and frequent observers plan compiled;
- flips `test_minimal_planner.py:196-210`.

Done when `Simulation(..., mode="compiled")` runs, is gated and is
recorded, and the fallback can choose it.

#### 3-T07 Energy-only graph and the MC evaluator (L, GPU: local, timing) — after T01, T02

Spec: §8.1 (energy-only graph), §10.4 (baseline energy, energy precision).
Source: `model/forces.py` 497–508 (`_outputs`), `model/energy.py`,
`polar_gpu._Call` in the research engine (the `grad_fn` failure to avoid).

- **A forward-only, compiled and captured energy graph** under
  `torch.no_grad()` (capture key "E"), with float64 accumulation, for MH-1,
  generic models and PolarMACE.
- **`mc_energy_precision="float64"`:** a float64 forward of the same plan
  for MC energies while MD forces keep their precision. Memory and time are
  reported.
- **The MC evaluator:** graph, precision and state version. The energy
  cache is tagged with them. The block baseline is always evaluated with the
  MC evaluator, never taken from the MD energy.
- **Timing:** energy-only cost as a fraction of one MD step in float32 and
  float64 on NaCl and NbCl5, local (acceptance item 2's evaluation part).

Files: `model/forces.py` (second editor after T02), `engine/graph.py` (the
"E" key; after T04 and T03), a new `fastmace/mc/evaluator.py` (owner).

Tests first:
- energy-only equals the force path's energy within rounding (CPU float64
  bitwise where the graph is the same);
- no `grad_fn` on the output buffers;
- a later force capture is unaffected;
- the baseline after MD equals a fresh evaluation with the MC evaluator.

Done when the evaluator gives float32 and float64 energies from captured
graphs, with costs reported.

#### 3-T08 Restart v3: current composition and MC state (M) — after T01

Spec: §10.6 (physical state, random state, MC state). Source:
`io/restart.py` (version 2, `load_state` 366–367), `simulation.py` 773–852
and 1141–1148, `state.py` 29–31 and 165–181.

- **The restart holds the current per-atom species, masses, charges, and
  total charge and spin.** Resume uses them, not the initial structure's.
- **The MC record** (filled by T09 and T10): the attached move set, the
  position in the MD/MC cycle, the counters per move type, every MC
  stream's state and position, and adaptive δ.
- **Restart writes stay at committed steps only,** never inside an MC block
  or from an overflowed step.
- Reading v2 files keeps working.

Files: `io/restart.py` (owner), `simulation.py` (record and resume; after
T03, before T15).

Tests first:
- round trip of a v3 file;
- resume after a host species change equals an uninterrupted run (CPU
  float64 bitwise);
- v2 still loads;
- flips `test_restart.py:108`.

Done when a run whose composition changed resumes exactly.

### Wave 2: Monte Carlo (after wave 1's pieces it names)

#### 3-T09 MC driver: swaps and transmutations (L, GPU: local) — after T02, T03, T07, T08

Spec: §10.4 (all but displacements), §8.5 (random numbers per outcome).
Source: T01's stubs, T03's events, T07's evaluator.

- **`mc/moves.py` and `mc/driver.py`** (owner):
  - `Swap` and `Transmute`, and `configurational_mu`;
  - fixed-shape selection: uniform keys, +∞ for other species, `argmin`,
    and "impossible" read from the minimum value;
  - three seeded device streams (proposal, acceptance, velocity) with fixed
    counts per attempt;
  - device counters (attempted, impossible, accepted per move type), read
    only with the per-block flags;
  - one captured attempt graph: propose → apply → refresh → energy →
    accept mask → restore.
- **Momenta:** `exchange`, `rescale`, `redraw` and `redraw_all`, with the
  `fixcm` rules of §10.4.
- **Charge and spin:** formal charges q_s, the total charge as a registered
  buffer, and the parity rule ΔZ − Δq even. The planner refuses violating
  sets.
- **The block:** n MD steps, then m attempts against the per-block baseline;
  forces are recomputed once at the end of a block with an acceptance
  (T03's resume).
- MC state goes into T08's restart record.

Files: `fastmace/mc/` (owner; T10 second editor), the parity part of
`planner/minimal.py` (after T06).

Tests first:
- an accepted and a rejected swap and transmutation each leave the state
  equal to a fresh construction;
- impossible attempts consume their stream counts;
- stream positions never depend on decisions;
- the counters match;
- the parity refusal.

GPU: capture of the attempt graph for MH-1 hybrid NaCl and NbCl5; swap
attempt cost in float32 and float64 (acceptance item 2).

Done when `Simulation.run(mc=Swap(...))` and `Transmute(...)` run captured,
with T15 wiring the front ends.

#### 3-T10 Displacement moves (M, GPU: local, timing) — after T04, T05, T09

Spec: §10.4 (displacement, fixed positions versus displacements), §8.5
step 4.

- **`Displace`:**
  - uniform in a sphere of radius δ;
  - δ fixed while sampling, adaptive only during equilibration, and
    recorded;
  - the trial re-compacts the list and runs the rebuild test inside the
    captured attempt.
- **A trial that needs a rebuild** leaves the graph. It builds eagerly in
  trial buffers around the trial geometry, then runs the captured energy
  graph. Accept commits the trial list; reject restores the neighbour-list
  state.
- **Overflow** restores, grows and redoes the attempt with the same random
  numbers (T04).

Files: `fastmace/mc/` (second editor), `neighbors/skin.py` (trial buffers;
after T05), `engine/graph.py` (after T07).

Tests first (§12.2 "mutation events"):
- a displacement beyond skin/2 is evaluated on a list built around the trial
  geometry, accepted and rejected, without looping, and its ΔE equals a
  direct evaluation;
- a rejected displacement leaves the committed list;
- an overflowing displacement is redone with the same numbers.

GPU: displacement attempt cost with and without a trial rebuild (acceptance
item 2).

Done when displacements run on both paths, with costs reported.

#### 3-T11 Move-energy gate and MC energy precision (L, GPU: local; Modal ≤ $3) — after T07, T09; owner decision D5

Spec: §12.3 (move-energy differences), §10.4 (energy precision, accuracy
prerequisite), §9.3 (fifth dimension), §9.4 (MC energy-precision dimension of
the fallback).

- **The ΔE gate** for sample moves of every enabled type: each swap pair,
  transmutation to every superset species (absent ones included) and
  displacements, `|ΔE_prod − ΔE_ref| ≤ 0.01 k_B T_min`.
- **`mc_energy_precision="auto"`:** the force precision if that gate passes,
  float64 otherwise. The fallback's MC energy-precision dimension retries in
  float64 before anything else.
- **When the gate runs** for MC plans: owner decision D5.
- **Measurement for acceptance item 3:** float32 ΔE errors against float64:
  - models: MH-1 `matpes_r2scan`, the MH-1 `omol` head (all-electron E0),
    and OMOL-0 XL where it fits;
  - structures: NaCl, NbCl5 and NaNbO3, at 300 K and 1500 K snapshots;
  - output: tables in `docs/validation/3-T11.md`.

Files: `planner/selfcheck.py` (owner), `planner/minimal.py` (the precision
dimension; after T06 and T09), `planner/fallback.py` (after T06).

Tests first:
- a model with an injected species-dependent constant energy error fails
  the ΔE gate and moves MC to float64;
- an absent superset species is gated;
- the gate record has the new fields.

Done when MC plans carry a gated MC energy precision and the measurement
table is in.

#### 3-T12 Exact enumeration and hybrid MD/MC statistics (M, GPU: local) — prepare now, run after T09

Spec: §12.2 "MC exact enumeration".

New test files only:
- 8 sites with 4 + 4 atoms (70 configurations), and transmutation (256);
- acceptance probabilities per move against the reference ΔE, including the
  first attempt after MD;
- a χ² test of the sampled populations (fixed seed);
- hybrid MD/MC: the kinetic temperature with MC on and off for `exchange`,
  `rescale`, `redraw` and (with `fixcm`) `redraw_all`, and zero total
  momentum under `fixcm`;
- a negative control with unchanged velocities, which must fail;
- the parity refusal;
- CPU tiny model and GPU MH-1. Mark long runs `acceptance`.

Files: `tests/cpu/test_mc_enumeration.py`, `tests/gpu/test_mc_statistics_gpu.py`
(new).

Done when acceptance item 5 passes on the CPU and the local GPU.

#### 3-T13 `fastmace.ase` observers apply composition changes as events (S) — after T03

Spec: §7.7 (observers), decision 17. Source: `ase/md.py` 413–418.

- Changes to species, masses and charges become events through T03.
- Atom count, PBC, cell, constraints and charge/spin `info` keys still raise.

Files: `ase/md.py` (owner).

Tests first:
- an observer's change gives the same state as the same `Simulation` event;
- flips `test_ase_md.py:526-528`.

Done when the §7.7 observer acceptance item passes in full.

#### 3-T15 Front ends: `run(mc=…)`, `fm.md`, CLI, report, provenance, thermo (M) — after T09

Spec: §7.2, §7.6 (MC blocks attach and detach), §7.7, §13.

- `Simulation.run(mc=…)` attaches a move set, which stays attached until
  `mc=None`.
- `fm.md(mc=…)` and the CLI `[[mc]]` tables run.
- **Report:** MC lines (moves, precision, gate, cost). **Provenance:**
  `counters.mc`. **Thermo:** columns for attempts and acceptances per move
  type at the thermo cadence.
- Every "MC: phase 3" refusal is removed.

Files: `simulation.py` (the run path; last of the merge order), `quick.py`,
`cli/run.py`, `cli/init.py`, `report.py`, `provenance.py` (after T04).

Tests first: flips `test_simulation.py:368-432`, `test_quick.py:343-356`,
`test_cli.py:307`, `test_metadata_schema.py:118` and
`test_interfaces.py:146` (if T01 assigns it here).

Done when all three front ends run MC, report it, and restart it.

### Wave 3: protocol, cross-repository, carry-overs

#### 3-T14 Swap-MC annealing protocol (M, GPU: local, then Ada) — after T15; owner decision D3

Spec: §10.5.

- **The protocol as data** (TOML, YAML through `safe_load`, or Python): the
  anneal profile with an attached swap block on the crystal and pairs of
  D3, at 1 fs.
- **`fastmace run`** reads the protocol file.
- **End to end:**
  - pre-relaxation;
  - MD with swap blocks;
  - the move-energy gate;
  - a restart in the middle;
  - the quench;
  - counters in the output.

Files: `protocols/` (owner), `cli/run.py` (second editor after T15).

Done when acceptance item 9 runs on the local GPU, and once on Ada (leeloo,
murphy or Modal L4 under the guard).

#### 3-T16 ResearchModule `main.py`: 1 fs production, and the adapter decision (S + S) — part a now; part b after T14; owner decisions D1 and D2; ResearchModule worktree by the conductor

- **Part a, now:**
  - production defaults to 1 fs (owner, 22 October 2026);
  - the output cadence keeps physical time (D6). That gives every 10th step
    for the anneal's 10 fs frames, and the quench bins unchanged;
  - update `docs/protocol.md` and `docs/configuration.md`, and record the
    change as a documented protocol change;
  - earlier results stay labelled 0.5 fs;
  - remove the old adapter if D1 says so;
  - a documented Modal recipe for one crystal under the budget guard (D2).
- **Part b:** `main.py --profile swap-anneal` on top of T14.

#### 3-T17 A fallback's reason names the innermost compile error (S) — after T06

From 2-T21d's note: the reason stops at "BackendCompilerFailed: backend='inductor'
raised:". It must carry the innermost message, here "Attempting to use a
closed pool", into `plan.reason` and the report line. Files:
`planner/fallback.py` (last of the merge order), with a test.

#### 3-T18 Coraline soft-lock check (S, conductor-scheduled) — when the workstations are back and coraline's GPU is idle

This is 2-T22, carried over.

### Wave 4: acceptance

#### 3-T20 Phase 3 benchmarks (M, GPU: local timing; Ada) — after T09, T10, T11

- **Acceptance item 1:** the event-speed A/B, species-change events every
  100 steps at 1500 K against fixed species. NaCl and NbCl5, 12 blocks, both
  orders.
- **Item 2:** the MC attempt cost table (swap float32 and float64;
  displacement with and without a trial rebuild).
- **Item 3:** the ΔE table from T11.

Files: `tools/` and `bench/` (new entry points), and `docs/validation/3-T20.md`.

#### 3-T21 Acceptance runs and report (L, GPU, conductor-scheduled) — after all except T16b and T18

- **Items 1–11** on the RTX 5060 Ti and one Ada GPU (D4).
- **The no-regression A/B** against main at the freeze.
- **Report:** `docs/validation/phase-3.md`, with pass or fail per item, run
  IDs, hardware, environments, costs (Modal) and open findings.

## Order and parallelism

```text
Wave 0   T01 contracts
             │
Wave 1   T02 superset   T03 events   T04 overflow   T05 skin (after T02's groups)
         T06 compiled   T07 energy-only (after T02)   T08 restart v3   T16a main.py 1 fs   T17 (after T06)
             │
Wave 2   T09 swaps/transmutations (T02 T03 T07 T08)   T13 observers (T03)
         T10 displacements (T04 T05 T09)   T11 ΔE gate (T07 T09; D5)   T12 enumeration (T09)
         T15 front ends (T09)
             │
Wave 3   T14 swap-MC protocol (T15; D3)   T16b main.py profile (T14)   T18 coraline (when back)
Wave 4   T20 benchmarks (T09 T10 T11)   T21 acceptance
```

- **Workers:** at most five active, one GPU job per reservation.
  1. Start order in wave 1: T02, T03 and T07 (the critical path), then T04,
     T06 and T08, then T05.
  2. T16a can start at once, since it doesn't depend on the library.
  3. T12 prepares from the start and runs after T09.
- **Critical path:** T01 → T02 → T07 → T09 → T11 → T20 → T21. That is about
  15 working days for one developer, plus reviews (SPEC: 5–6 weeks for the
  whole phase). Give T03, T09 and T10 the most review. At phase 2's pace,
  which planned 4–5 weeks and finished in 3 days, expect about a week of
  wall time.
- **Size:** 20 tasks (3-T19 was left unused when T16 took the cloud recipe),
  about 37 worker-days.
- **Merge order of the shared files:**
  - `planner/minimal.py`: T02 (rule 8), T05 (rule 11), T06 (owner: modes,
    rules 4 and 9), T09 (parity), T11 (MC energy precision);
  - `engine/graph.py`: T04 (owner), T03, T07, T10;
  - `simulation.py`: T03 (mutation API), T08 (restart), T15 (run path);
  - `model/forces.py`: T02 (owner), T07;
  - `neighbors/skin.py`: T02 (sender groups), T05 (owner), T10 (trial
    buffers);
  - `planner/fallback.py`: T06, T11, T17;
  - `provenance.py`: T04, T15;
  - `cli/run.py`: T15, T14.
- **GPU slots:**
  - **local RTX 5060 Ti:** everything short. The timing runs (T05, T07,
    T10, T20, T21) go one at a time with the PC quiet;
  - **Ada:**
    - leeloo or murphy only before Thursday 22 October 16:00 UTC, when idle,
      and again after the owner says the workstations are back;
    - otherwise the Modal L4 under the budget guard.
  - **Modal budget:** about $10 for the whole phase. T05 ≤ $2, T11 ≤ $3,
    T14 ≤ $1, T20 ≤ $2 and T21 ≤ $3, all inside the $27 monthly limit.
    The month rolls over on 1 November.

## Conductor decisions (no owner decision needed)

- **Re-capture budget exceeded at run time:** warn, record and recommend
  the compiled tier. No automatic mid-run downgrade, because runtime
  adaptation is phase 5 (§9.6). The planner chooses compiled up front where
  rules 4, 9 and 11 predict it.
- **All-electron model for item 3:** the MH-1 `omol` head, cheap and on the
  same architecture, plus OMOL-0 XL where it fits on 16 GB.
- **Superset cost:** measured in T02. If it isn't small, the superset
  default stays "the structure's species" (SPEC already says so), and the
  report shows the cost.
- **The 2 fs heating transient noted in phase 2** (the "finding for the
  phase 3 notes") was the `prepare()` defect fixed in 2-T21d, not physics.
  1 fs production rests on the October 1 fs checks of the research engine
  and on the fixed FastMACE path.

## Checkpoints for the owner

1. Before the go messages: D1–D6 below.
2. After T11's measurement: whether float32 MC energies are accepted where
   the ΔE gate passes. That is SPEC's rule; the checkpoint is information,
   not a new decision.
3. After T21: the acceptance report, and the tag `fastmace-3`.

## Owner decisions

**D1. The old `main.py` adapter** (`--fastmace-api adapter`, `FixedPlanner`
plus the research loop). Decision 28 kept it "for one phase"; phase 2's
checkpoint 3 asked about it.
- **Options:**
  - (a) remove it in phase 3. Keep `--engine gpu` (the research engine) for
    A/B baselines and for the few runs `Simulation` refuses (PolarMACE
    Fukui and dipole-cap, a friction without an exact per-fs value);
  - (b) keep it one more phase;
  - (c) remove it and port those options to `Simulation` first.
- **Recommendation: (a).** `Simulation` passed phase 2's A/B, and two paths
  double the test surface (the `prepare()` bug only hit one). The refused
  options stay reachable through `--engine gpu`.

**D2. Production on cloud GPUs** (owner, 22 October 2026: "cloud is fine").
- **Options:**
  - (a) a documented recipe plus a thin wrapper that runs one `main.py`
    crystal per Modal job, from a pinned commit, under the budget guard,
    with results copied back;
  - (b) a full `main.py --modal GPU` integration with resume across
    container limits;
  - (c) nothing now.
- **Recommendation: (a), in T16a.** The $30 monthly cap allows roughly 8–10
  NbCl5 anneals at 1 fs on an A100 (about 1 hour and $2.5–3 each, CPU and
  memory included), so a light wrapper fits the scale. (b) can follow if
  cloud runs become routine.

**D3. The swap-MC annealing protocol** (acceptance item 9). It needs a
crystal with two exchangeable species and a schedule.
- **Options:**
  - (a) O ↔ Cl anion swaps in NaNbOCl4 (the Ma 2026 I4/m polymorph from
    `Structures/NaMOCl4`, 3×3×3). That is the oxychloride electrolyte
    family itself, where anion disorder bears on Na transport;
  - (b) Nb ↔ Ta B-site swaps in a 50/50 Na(Nb,Ta)Cl6 or Na(Nb,Ta)O3,
    derived from the benchmark NaNbCl6 or NaNbO3 by substitution;
  - (c) both.
- **Schedule, either way:**
  - the anneal profile at 1 fs (0 → 1000 K in 20 ps, then 3 K/ps to
    250 K);
  - `Swap(every=100, attempts=50, velocities="exchange")`, with
    `min_mc_temperature_K=250`;
  - MH-1 `matpes_r2scan`, hybrid float32, MC energy precision per the gate;
  - the quench as in the anneal protocol.
- **Recommendation: (a).** It is the research question's own chemistry, and
  it keeps the superset to existing species {Na, Nb, O, Cl}. The outcome is
  recorded, not judged: it is a model result.

**D4. The Ada GPU for phase 3 acceptance.** SPEC §16 asks for "the RTX 5060
Ti and one workstation GPU". The workstations are off from Thursday evening
until you say they're back.
- **Options:**
  - (a) leeloo or murphy only;
  - (b) the Modal L4 (sm_89, same architecture) counts as the Ada GPU;
  - (c) whichever is available at acceptance time, preferring a
    workstation for the timing items 1 and 10. The L4 is power-limited, and
    its graph replays run 4–5 ms slower than its kernel sums on the large
    cells (OPT-T10).
- **Recommendation: (c).** The L4 matched the local results for OPT-T08,
  and ratios such as item 1's 90 % hold on it.

**D5. MC and the opt-in gate.** §10.4 enables MC "only after the energy and
move-energy gate passes", while decision 3 made the gate opt-in (off by
default).
- **Options:**
  - (a) MC plans always run the move-energy part of the gate (seconds to a
    minute; reference ΔE on a few sample moves). The energy, force and
    stress parts stay opt-in. If the reference is `unavailable`, MC energies
    use float64;
  - (b) the gate stays fully opt-in. When skipped, "auto" uses float64 MC
    energies (safe, slower: FP64 is slow on these GPUs);
  - (c) fully opt-in, and "auto" uses float32 with a report warning.
- **Recommendation: (a).** Acceptance probabilities depend on ΔE, the check
  is cheap, and it is MC-specific, so decision 3 stands for MD.

**D6. 1 fs beyond `main.py`.**
- **Questions:**
  - Should FastMACE's own defaults (`fm.md`, CLI, the `init` template;
    today 0.5 fs) change too?
  - How does `main.py`'s output cadence follow the timestep?
- **Options:**
  - (a) library defaults stay 0.5 fs. MACE-OFF's organics with hydrogen
    need ≤ 0.5 fs, and a library default must be safe for every model. The
    `init` template for inorganic crystals suggests 1 fs. `main.py` keeps
    physical intervals: frames every 10 fs, quench bins and restart
    intervals in time, not steps;
  - (b) library defaults move to 1 fs too;
  - (c) `main.py` keeps step-based cadences (every 20th step, which at 1 fs
    is 20 fs).
- **Recommendation: (a).** It changes only the production timestep, which
  is what you decided.
