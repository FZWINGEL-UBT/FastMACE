# Phase 5: Planner completion

Status: **draft**, 23 October 2026, written while phase 4's acceptance
(4-T13) runs. D1–D6 below are open.
Spec: [SPEC.md](../SPEC.md) §16 row "5. Planner completion", and the
sections it cites:
- §9.1: rules, not measurement tables; the optional calibration, never
  persisted, no per-GPU or per-shape table;
- §9.2–§9.3: the inputs, the fifteen rules, their kinds and their strict
  order;
- §9.4 step 5: mid-run fallbacks between segments;
- §9.5: the frozen `Plan`, its revisions, `str(plan)`;
- §9.6: runtime adaptation;
- §7.7: the run report ("until runtime calibration (phase 5) there is no
  comparison between tiers"), and the review table's "calibration is
  phase 5";
- §13: provenance (tier switches);
- §18: decisions 3, 29, 35, 38, 39 (amended 23 October 2026) and 50.

Acceptance (§16): **each rule and the precedence order have a test.**
Estimate: 1–2 weeks (§16); with D1–D6 at (a), about 2 weeks.

Why it matters:
- Most of §9.3 exists, spread over phases 1b–4, but nothing checks the
  rules as a set. Rule numbers on reasons are missing in places, and one
  decision text is stale ("rule 8 is not in 1b").
- Rule 12 covers only the D3 list. `Plan.predicted_memory_bytes` is
  always `None`, so an out-of-memory run fails late (POLAR-1-L NPT on
  NbCl5 ran out of memory on 20 GB).
- A run whose re-capture budget is exceeded only warns and recommends the
  compiled tier (3-T04, `docs/design/contracts-3.md`). A failure during a
  run (a capture at a re-capture) stops it, because mid-run fallbacks were
  left to phase 5 (phase 2's "Kept out").
- `str(plan)` is a flat dump of every decision, one line each, in
  resolution order. §9.5 asks for a human-readable plan.

## Inventory

### The rules of §9.3 today

| Rule | Kind | State | Where | Missing in phase 5 |
| --- | --- | --- | --- | --- |
| 1 §4.4, variable cell without barostat | hard | done (1b, 4-T01 scheme records, decision 50) | `planner/minimal.py`, `integrators/registry.py` | conformance test with the rule number |
| 2 species outside the superset | hard | done (1b, 3-T02) | `minimal._common`, `planner/minimal.check_mc_moves` | test for the move-set half (swap pairs, transmutable sets) under the rule's code |
| 3 PolarMACE on plain compiled cuEq | hard | done (1b) | `fixed._resolve`, `longrange.route_support` | test |
| 4 inexact padding: no graph mode | hard | done (1b refusal, 3-T06 compiled tier) | `minimal._mode`, `_compiled`, `_eager` | precedence tests against rules 8 and 9 |
| 5 `reproducible=True` | hard | done (2-T16; MH-1 plain cuEq qualified) | `planner/reproducible.py` | test against every preference that could change the plan (12, adaptation) |
| 6 naive cuEq kernels | hard | done (1b) | `minimal._common` | test that `allow_naive_cueq` lifts rule 6 only |
| 7 atom-count changes | hard | done (1b) | `minimal.plan_traits` | test |
| 8 species changes keep graph mode | preference | done in substance (3-T02: superset build) | `minimal._common` | the `workload.species_changes` decision still says "rule 8 is not in 1b"; precedence under rule 4 |
| 9 host callbacks keep graph | preference | done (3-T06b, decision 38) | `minimal._workload_decisions` | override test (`mode="compiled"`) |
| 10 stress path and k-headroom from the integrator or optimiser | preference | done (1b, 4-T05, 4-T07) | `fixed._resolve` | test per cell mode |
| 11 skin, headroom, re-capture budget; D3 skin; displacement skin | preference | done (3-T05, 3-T10, 3b-T02) for MC, species changes and D3; MD-only keeps 1a's 0.3 Å by the conductor's answer of 22 October (3-T05: rebuilds 2.2–2.7× rarer, step time within noise) | `neighbors/adaptive.py`, `minimal._rule_11`, `_d3_list` | test that MD-only keeps 0.3 Å and records the rule's value; override test |
| 12 memory: drop memory-heavy rewrites or warn | preference | **partial**: D3 list only (3b-T02, `rule_12_d3`) | `minimal._d3_list` | the MACE model's memory (edge bucket × feature width), `predicted_memory_bytes`, the action (D1) |
| 13 self-check fails: disable it and its dependants | reaction | done (2-T04) | `planner/reactions.py`, `rewrites/registry.py` | test of its order before rule 14 |
| 14 gate fails: fallback | reaction | done before the first step (2-T15); **missing mid-run** (§9.4 step 5) | `planner/fallback.py`, `simulation._failure` | mid-run fallback (D4) |
| 15 known model–structure warning | reaction | **in the report, not the plan**: `report.KNOWN_ISSUES` | `report.py` | a plan decision with rule 15's code, read by the report |

**Precedence today.** The book (`fixed._Book`) collects every hard refusal
and raises them together, as §9.3 asks. What has no test is the order
itself: that no preference or override weakens a hard rule except through
its `allow_*` flag, that a later preference does not undo an earlier one,
and that reactions run after the plan is made (13 before 14). Overrides
exist one by one (explicit `RewriteConfig` values, `NeighbourConfig.skin`,
`mode=`, `allow_*`). They are recorded in `Plan.requested`, but a decision
does not say "overridden: the rule would give X" in a uniform way.

### Runtime adaptation (§9.6), concretely

§9.6 lists four monitors and one action: "switch tiers between segments
(never mid-step), as a new plan revision with a logged event". §9.4 step 5
adds mid-run fallbacks "between segments, from the last committed state".

| Monitor | Exists | Where |
| --- | --- | --- |
| re-capture count, budget, step it was exceeded | yes (3-T04) | `GraphEngine.recaptures`, `recapture_budget_exceeded_at`, `info()["recapture_log"]` |
| time spent re-capturing | no | — |
| rebuilds and build time per list (MACE, D3) | yes | `counters.neighbour_lists` (provenance) |
| bucket overflows redone | yes (3-T04) | `overflows_redone` |
| non-finite observables | yes: the run stops with `NonFiniteError` | `simulation._advance` |

What does not exist:
- a monitor that reads these per segment and decides;
- a tier switch during a run: a new revision built from the pristine model,
  the state handed over, then validated and activated;
- `counters.tier_switches` (provenance keeps it `null`, "phase 5");
- the persistent worker's parent-side revision path, which ends the worker
  (`restart_worker()`, OPT-T07) and has no caller yet.

"Segment" is not a `schedule` segment here. Conductor's reading (choice 2):
a switch happens at a **canonical restart point** (§10.6). There the
neighbour lists are rebuilt and the state is complete, which is what a
resume needs. A revision there is a resume onto another plan, in the same
process.

The non-finite rule stays as it is: the run stops, nothing is "fixed".

### What "optional calibration" may and may not do (§9.1)

May:
- run only when asked (`calibrate=True`); off by default;
- measure, on the user's own structure in this run, the step time of a
  few candidate plans that all pass the hard rules, about 50 steps each
  after warm-up;
- break a tie: choose among candidates the rules rank equal. It never
  overrides a hard rule, an `allow_*` flag or a choice the user made;
- record what it measured in the plan (`Plan.calibration`) and in
  provenance, as a record of this run.

May not:
- persist anything across runs: no cache file, no table keyed by GPU,
  shape or crystal, no reading of a previous run's numbers (§9.1, owner's
  generality rule);
- change the physics, the force precision, the MC energy precision or the
  rewrite set (rewrites are exact and chosen by rules 12–13);
- print predicted speed-ups (§7.7). The report shows measured numbers
  only.

The candidate dimensions are D5's question. Today the rules leave almost
no tie: graph unless rule 4; hybrid whenever OEq builds; rule 9 keeps
graph with callbacks. The mode is not a tie either (graph beat compiled in 3-T06 and
4-T05). The one place a rule's choice could differ by
architecture is the backend (hybrid against plain cuEq), which nobody has
timed off Blackwell and Ada. In one process both cannot compile (§8.7, one
compile family), so a backend candidate runs in a child.

### What `str(plan)` must show (§9.5)

- a header: revision, tier, backend (plain: fused convolution or not),
  force precision, MC energy precision, runner, and why this tier;
- the hard rules that shaped the plan (for example "rule 4: compiled
  tier") and the `allow_*` flags used;
- rewrites on, and rewrites off with their reasons;
- sizes: skin, pair-buffer headroom, edge bucket, re-capture budget and
  variant cap, D3 skin and list, k-space headroom;
- kernel availability: OEq, Triton, NVRTC, fused cuEq;
- predicted memory against the GPU's (rule 12);
- the workload and its rule 8–11 decisions, overrides marked;
- model–structure warnings (rule 15);
- the fallback candidates tried, the calibration record, and the revision
  chain with step and reason;
- at most about 30 lines by default. `plan.describe(full=True)` keeps
  today's list of every decision. Provenance stores both.

## Scope

- The rule table as data, the rule numbers on every reason, and a
  conformance suite: one test per rule and a precedence matrix.
- Rule 12 for the MACE model: a memory estimate from the model's own
  dimensions, `predicted_memory_bytes`, and D1's action.
- Rule 15 in the planner.
- `str(plan)` by sections.
- Runtime adaptation: the monitor, the tier switch at a canonical point
  (D2, D3), and mid-run fallbacks (D4).
- Optional calibration, as D5 decides.

## Out of scope

- **New rules** beyond §9.3's fifteen. A gap found in the conformance
  suite goes to the conductor as a SPEC question.
- **Rule 11 for MD-only runs as a default.** 3-T05 measured no step-time
  gain. It stays a recorded value (conductor, 22 October 2026).
- **Atom-count changes** (rule 7 keeps refusing; §17).
- **Batching or concurrent structures** (owner): calibration measures one
  structure's candidates one after another.
- **Qualifying `reproducible=True` for another family.**
- **ResearchModule's `main.py`:** no change. Its `run.json` gains no key
  unless an adaptation fires there.
- **1a's two-stage `auto` resolution:** stays unless D6 says otherwise.

## Owner decisions

**Decided (user, 23 October 2026): the recommendation (a) of each, D1–D6; SPEC §18 decisions 51–56. 5-T10 is closed (D6 (a)).**

Each has a recommendation (a). The task list says which tasks each blocks;
everything else can start before the answers.

**D1. Rule 12: what the planner does when the estimate does not fit.**
The estimate (5-T03) adds the model's activations for the edge bucket and
the node features, forward and backward, at the force precision, plus
the D3 list where it is on. It uses the model's dimensions (channels, L,
interactions, radial basis) and the plan's bucket. It contains no GPU
table. Each rewrite record declares its own term, so a rewrite that holds
an extra edge-sized buffer says so.
- **Recommended (a):**
  - estimate above 90 % of the free GPU memory: turn off the rewrites
    whose records declare an edge-sized extra buffer, in a fixed order
    (largest term first), until it fits, each with rule 12's reason;
  - still above: keep the plan, warn with the estimate, and let an
    out-of-memory error carry rule 12's estimate in its message;
  - never shrink rule 11's skin or headroom (that would trade memory
    for re-captures, which rule 11 sized). Never change the tier, the
    precision or anything a user set explicitly (an explicit `"on"`
    warns, as today);
  - the 90 % margin and the estimate's accuracy come from 5-T03's
    measurements on the 19 labels. The estimate must not be below the
    measured peak on any of them.
- **Alternatives:**
  - (b) warn only, never drop a rewrite: the user decides;
  - (c) refuse above the free memory, naming the rewrites that would
    have to go;
  - (d) (a), and an out-of-memory error at build or first capture also
    re-plans once through rule 12 with the measured peak in place of the
    estimate (a fallback trigger; today out-of-memory is not one).
- **Blocks:** the action part of 5-T03. The estimate and its validation
  start before.

**D2. Runtime adaptation: which monitor triggers which switch.**
Background: the compiled tier is 1.3–4.7× slower per step than graph
(3-T06). A re-capture costs a capture of the step's graphs, so switching
pays only when re-captures are frequent. 3-T05 found the rebuild rate
does not change the step time while `overlap_rebuild` is on.
- **Recommended (a):**
  - **one trigger:** the re-capture budget is exceeded (3-T04's rule) **and**,
    over the segments since the last switch, the time spent re-capturing
    is above 20 % of the wall time. The monitor measures both in this run,
    so the trigger is general (no per-GPU number);
  - **one switch:** graph → compiled, same backend, rewrites and
    precision, at the next canonical point (choice 2);
  - rebuild rate, build time and overflows are recorded per segment, in
    `counters` and the report, with no automatic action;
  - the 20 % is checked in 5-T07's timed measurement: on a forced
    re-capture storm the switch must make the run faster, and on the
    production ramps (NaCl, NbCl5 anneal profile) it must never fire.
- **Alternatives:**
  - (b) (a), plus a skin increase when the MACE or D3 list rebuilds more
    often than every n*/2 steps: rule 11 re-planned with the measured
    rate (a re-capture follows);
  - (c) switch on the budget alone, as 3-T04's warning recommends today
    (no wall-time measure; deterministic, but it fires where a few
    re-captures cost little);
  - (d) no automatic switch: keep the warning and the recommendation.
    Phase 5 then does only mid-run fallbacks (D4).
- **Blocks:** the trigger in 5-T07. The monitor (5-T06) and the switch
  machinery start before.

**D3. Runtime adaptation: limits.**
- **Recommended (a):**
  - **one way, once per run:** no switch back to graph, and at most one
    tier switch per run (a resume counts the revisions already in the
    chain);
  - **never** under `reproducible=True` (a switch would break the bitwise
    promise across restart cadences), with an explicit `mode=` (a user
    override of rules 4 and 9 is kept: the warning stays), during an MC
    block, or on the subprocess runner's child without the parent's
    `restart_worker()` hand-off;
  - **validated before activation** (§9.4 step 4): self-checks always; the
    gate when the run requested one (`check="gate"`, decision 3), and the
    move-energy part for MC plans (decision 35). A candidate that fails
    goes to the fallback (D4) or, with `fallback="refuse"`, stays on the
    old revision with a warning;
  - logged: plan revision with step and reason, report line,
    `counters.tier_switches`, and a `fastmace.planner` log line.
- **Alternatives:**
  - (b) reversible with hysteresis: back to graph after k quiet segments
    (no re-capture), at most two switches per run;
  - (c) allowed under `reproducible=True` with a deterministic trigger
    only (D2 (c)), the monitor's counters in the restart (format 5);
  - (d) no limit on count; a minimum of N canonical points between
    switches.
- **Blocks:** 5-T07's limits and tests.

**D4. Mid-run fallbacks (§9.4 step 5): which failures.**
- **Recommended (a):**
  - triggers: a capture error at a re-capture, and a compile error of the
    compiled tier at a new edge count, raised during `run()`;
  - from the last committed canonical point: the steps after it are
    redone on the validated candidate. Output after that point is handled
    as on resume (chunks deleted, never rewritten; §10.6);
  - the same algorithm and budget as before the first step, with the
    reduced gate on the committed state;
  - not triggers: out-of-memory (stops with rule 12's estimate, unless
    D1 (d)), non-finite values (stop, §9.6), refusals and process
    constraints, as today.
- **Alternatives:**
  - (b) (a) plus a failed overflow redo (an overflow that a re-capture
    cannot absorb);
  - (c) mid-run failures stop the run with a record and a resumable
    restart point; the user resumes with `allow_plan_change=True`.
- **Blocks:** 5-T08.

**D5. Calibration: whether it exists, and what it may measure.**
- **Recommended (a):**
  - `calibrate=True` (in `PlannerOptions`, `fm.md`, `fastmace run
    --calibrate`), off by default;
  - candidates: the backend dimension only (hybrid against plain cuEq)
    where both pass the hard rules and the user set no backend. The mode
    dimension is not a tie: graph beat compiled 1.3–4.7× in MD (3-T06) and
    1.1–3.9× in relaxations (4-T05);
  - each candidate in a child process (§8.7; it also keeps the parent's
    compile family free), 50 steps after warm-up, alternating children
    A B A B;
  - it changes the rule's choice only when the other candidate is faster
    by more than 5 % (above the 3 % cross-process noise) in every pair;
  - `Plan.calibration` records each candidate's measured step time; the
    report shows them as measured. Nothing is written outside the run's
    directory, nothing is read from earlier runs;
  - its cost (setup time) is reported. Production, one crystal after
    another, would pay it per crystal, so it is for unfamiliar GPUs and
    benchmarks, not a default.
- **Alternatives:**
  - (b) measure and report only: `calibrate=True` records the candidates'
    times and never changes the plan. The rules stay the only deciders;
  - (c) no calibration in 1.0: §9.1 amended, §7.7's "until runtime
    calibration" removed, calibration listed in §17;
  - (d) (a) plus the precision dimension. Not recommended: the force
    precision is the `ForceField`'s, a physics choice (§9.2).
- **Blocks:** 5-T09 (under (c) it is dropped and only the SPEC changes).

**D6. 1a's two-stage `auto` resolution** (`FixedPlanner._stage1`,
`_stage2`, which reproduce `main.py`'s `parse_args` then
`adapt_options_to_model`; phase 1a deferred "simplification" to phase 5).
- **Recommended (a):** keep it in phase 5. The conformance suite covers its
  outputs (rewrite decisions per model family), and it stays one function
  with its source cited. Rewriting it risks changing default plans and
  gives users nothing. Reconsider in phase 6, when the upstream rewrite PRs
  may change it anyway.
- **Alternative (b):** rewrite it as rule-table entries in phase 5 (task
  5-T10, size L), guarded by a plan-equality corpus: every label × the
  `main.py` configurations, identical choices before and after.
- **Blocks:** 5-T10 only.

## Conductor's choices (the owner may override)

1. **Rule table as data** (`fastmace/planner/rules.py`): order, kind, a
   short title, the codes each rule uses and its `allow_*` flag. Every
   `Reason` the planner emits for a rule carries `rule=` and a code from
   that table. A meta-test fails if a code is unknown, a rule has no test,
   or a rule's code appears without its number.
2. **Where a switch happens:** the next canonical restart point after the
   trigger, forced if the run's restart cadence would put it later. The
   switch is an in-memory resume: the committed state as a restart record,
   a new engine from a pristine model copy, the record restored. No file
   is written unless the run writes restarts anyway. Under the subprocess
   runner the parent calls `restart_worker()` with the new revision.
3. **"Gated like any other"** (§9.3 on overrides) reads as: the same gate
   policy as an unoverridden plan, so opt-in for MD (decision 3), always
   the move-energy part for MC (decision 35). No new gate runs.
4. **Overrides are recorded uniformly:** a decision a user replaced reads
   `choice (override; rule N would give X)`, and `Plan.requested` keeps
   the value. The overrides that already exist keep their keys.
5. **Restart format:** stays 4 unless 5-T01 finds that a switch needs
   state the revision chain does not carry. D3 (a) needs none, since the
   active revision's chain counts the switches. D3 (c) would need format 5.
6. **Rule 15's data** moves from `report.KNOWN_ISSUES` to
   `planner/warnings.py` unchanged. The report reads the plan's rule 15
   decisions, so its lines stay the same.
7. **Default plans do not change.** On a run where no trigger fires,
   phase 5 changes no tier, rewrite, size or step time. Rule 12 drops a
   rewrite only where the estimate says it would not fit. 5-T11's plan
   corpus and timed A/B check this.
8. **Acceptance hardware** (decision 39 as amended): full acceptance on
   the RTX 5060 Ti; `-m compat` on Blackwell (local), Ada (Colab L4) and
   the Colab A100 if phase 4's check made it a production type. Untimed
   runs may use the Colab A100 to finish sooner; timed runs stay local.

## Tasks

| Task | Size | GPU | After | Files (main ones) |
| --- | --- | --- | --- | --- |
| 5-T01 Contracts | S–M | no | — | `planner/rules.py` (new), `planner/plan.py`, `config.py` (`AdaptationConfig`, `CalibrationConfig`), `errors.py` (`Reason` checks), `simulation.py` (`PlannerOptions` fields), `docs/design/metadata-schema.md`, `provenance.py` (schema), `pyproject.toml` (marker), `docs/design/planner-5.md` |
| 5-T02 Rule conformance and precedence suite | M | no | T01 | `tests/cpu/test_rules_*.py`, `planner/minimal.py`, `planner/fixed.py` (reason texts and numbers only) |
| 5-T03 Rule 12: memory estimate and action | M–L | yes (untimed peaks) | T01; D1 for the action | `planner/memory.py` (new), `planner/minimal.py`, `rewrites/registry.py` (memory terms), `neighbors/adaptive.py`, `docs/validation/5-T03.md` |
| 5-T04 Rule 15 in the planner | S | no | T01 | `planner/warnings.py` (new), `planner/minimal.py`, `report.py` |
| 5-T05 `str(plan)` | S–M | no | T01 | `planner/plan.py`, `provenance.py` (plan text), `tests/cpu/test_plan_text.py`, golden files |
| 5-T06 Runtime monitor | M | smoke | T01 | `planner/adapt.py` (new), `engine/graph.py` (re-capture time), `simulation.py` (segment hook), `provenance.py` (counters), `report.py` |
| 5-T07 Tier switch between segments | L | yes (timed) | T06; D2, D3 | `planner/adapt.py`, `simulation.py`, `planner/minimal.py` (revision for the compiled tier), `engine/subprocess.py`, `report.py`, `docs/validation/5-T07.md` |
| 5-T08 Mid-run fallbacks | M–L | yes | T07; D4 | `planner/fallback.py`, `simulation.py` |
| 5-T09 Calibration | M–L | yes (timed smoke) | T01, T05; D5 | `planner/calibrate.py` (new), `planner/minimal.py`, `simulation.py`, `quick.py`, `cli/run.py`, `report.py` |
| 5-T10 Two-stage `auto` as rules (only under D6 (b)) | L | no | T02; D6 | `planner/fixed.py`, `tools/plan_corpus.py` |
| 5-T11 Acceptance runbook and report | M | yes | all | `docs/validation/phase-5-runbook.md`, `docs/validation/phase-5.md`, `tools/plan_corpus.py` (new), `tools/acceptance/phase5_local.sh`, `tests/_support/tiers.py` |

**5-T01 Contracts.**
- `planner/rules.py`: the table of §9.3 as data (choice 1), with
  `rule(n)`, `codes(n)`, `kind(n)` and the `allow_*` flag per hard rule.
- `Reason`: a rule number must be one of 1–15, and a code a rule claims
  must be in its table entry (checked in tests, not at run time).
- The override record (choice 4) and how `Plan.requested` relates to it.
- `AdaptationConfig` (frozen, in `config.py`): `adapt="auto"|"off"`, the
  limits D3 decides, and the trigger's thresholds as fields with D2's
  defaults. No module-level switch.
- `CalibrationConfig`: `calibrate: bool = False`, steps, margin. It exists
  under every D5 option except (c).
- `Plan.calibration` (a record or `None`) and `Plan.adaptation` (the
  active limits); `counters.tier_switches` as a list of `{step, from,
  to, reason, revision}`; the metadata schema minor version bumped.
- The `str(plan)` section layout in `docs/design/planner-5.md`.
- A pytest marker `rule(n)` and the meta-test skeleton, with the rules
  that have no test yet listed as expected failures. 5-T02 empties the
  list.
- Merges first. T02–T06 start on its contract.

**5-T02 Rule conformance and precedence suite.**
- One test file per group (hard, preferences, reactions), each test
  marked `rule(n)`, on the tiny model with injected `SoftwareTraits` and
  `Hardware` (no GPU):
  - each hard rule refuses with its code and number, alone;
  - several hard rules together give one `PlanRefused` with every reason;
  - each `allow_*` flag lifts its own rule only;
  - each preference's default, and its override recorded (choice 4);
  - reactions: 13 before 14 (a self-check failure disables the rewrite
    before any gate runs);
  - rule 15 (after T04) as a warning, never a refusal.
- **Precedence matrix:** pairs where an earlier rule wins:
  - rule 4 against 8 and 9 (inexact padding with species changes and
    callbacks: compiled tier);
  - rule 5 against 9, 11, 12 and adaptation;
  - rule 3 against an explicit plain compiled backend;
  - rule 1 against `allow_reference` with MC and constraints (decision
    10);
  - a preference against an earlier preference (11 then 12: rule 12
    never shrinks rule 11's sizes under D1 (a));
  - a user override against a hard rule: refused, never downgraded.
- Fixes found on the way, text and numbers only: the stale rule 8 text,
  missing `rule=` numbers. Any change of a choice goes to the conductor.
- **Tests:** the suite itself, and the meta-test green.

**5-T03 Rule 12: memory estimate and action.**
- `planner/memory.py`: the estimate of D1's paragraph from
  `capabilities.ModelTraits` (channels, L, interactions, radial basis,
  heads), the plan's edge bucket and pair-buffer headroom, the force
  precision, and the D3 list's existing estimate. Each rewrite record
  gains an optional memory term (bytes per edge or per node), declared by
  the rewrite's author and checked by a CPU test against a tiny-model
  measurement.
- `Plan.predicted_memory_bytes` filled; the decision `memory` with rule
  12's code; the action per D1.
- **Validation (GPU, untimed):** the measured peak (`torch.cuda.
  max_memory_allocated` over build, capture and 100 steps) against the
  estimate, for the 19 labels on NaCl 3×3×3 and NbCl5 3×3×3, NVT and NPT.
  The large models (OMOL-0 XL, POLAR-1-L) run on the Colab A100 (40 GB);
  the rest local, after phase 4's acceptance has drained. Pass: the
  estimate is never below the peak, and at most 1.5× above it.
  `docs/validation/5-T03.md`.
- **Tests:** CPU: the formula on the tiny model against its measured
  peak; the action per D1 with injected free memory; an explicit `"on"`
  kept and warned.

**5-T04 Rule 15 in the planner.**
- `KNOWN_ISSUES` moves to `planner/warnings.py` (choice 6). The planner
  adds a `warn.model_structure` decision per matching issue with rule 15's
  code; the report reads the plan.
- **Tests:** the existing report tests unchanged; one planner test per
  issue row; no refusal.

**5-T05 `str(plan)`.**
- `Plan.describe()` by the sections of the inventory, at most about 30
  lines; `describe(full=True)` keeps today's every-decision list.
  Provenance stores both (`plan.description`, `plan.description_full`, and both in each
  `plan_revisions` entry).
- Lines that T03, T07 and T09 fill (memory, revision chain with switches,
  calibration) exist with a placeholder ("not estimated", "none"). Those
  tasks update the golden files.
- **Tests:** golden texts for: hybrid NVT; NPT with D3; MC with a
  superset; the compiled tier by rule 4; the reference tier; a fallback
  revision; a reproducible plan. Each section present, no line over 120
  characters.

**5-T06 Runtime monitor.**
- `planner/adapt.py`: `Monitor` reads `engine.info()` at each canonical
  point: re-captures and the time they took (new in `engine/graph.py`,
  host timestamps around a capture, no device sync added to the step),
  rebuilds and build time per list, overflows. It keeps a per-segment
  window.
- `Simulation` calls it at canonical points. It decides nothing yet:
  T07 adds the trigger.
- Provenance: per-segment counters in `counters.segments`, totals as
  today. The report adds a line only when re-captures or overflows
  happened.
- **Tests:** CPU with a fake engine; a GPU smoke that the counters move
  on a forced re-capture (monkeypatched variant cap) and that the step
  time is unchanged (one block each, local, sized to a 1 % effect: the
  full alternating design; this is the only timed item of T06).

**5-T07 Tier switch between segments.**
- The trigger (D2) and the limits (D3) in `planner/adapt.py`.
- The switch (choice 2):
  - `MinimalPlanner` makes the compiled revision of the active plan
    (`Plan.revise`, reason and step);
  - `Simulation` builds it from a pristine model copy, hands the
    committed state over as an in-memory restart record, validates
    (self-checks, gate per D3), then activates;
  - the subprocess runner's parent calls `restart_worker()`;
  - `counters.tier_switches`, report line, log line.
- **Tests:**
  - CPU: the trigger and every limit with a fake monitor; the revision's
    chain; a resume after a switch runs the compiled revision and does
    not switch again;
  - GPU (local): a forced re-capture storm on MH-1 hybrid NaCl switches at
    the next canonical point; energies and temperatures statistically
    equal to an unswitched compiled run; interrupted against uninterrupted
    across the switch, statistical;
  - **timed (local, quiet PC):** on the storm, the switched run is faster
    than the unswitched one (effect expected large: one block each); on the
    NaCl and NbCl5 anneal profiles the trigger never fires.
- `docs/validation/5-T07.md`.

**5-T08 Mid-run fallbacks.**
- `Simulation.run` catches D4's failures, returns to the last committed
  canonical point, runs §9.4 (reduced gate, ddmin, terminal sequence) on
  the committed state, and continues on the validated revision; output
  chunks after the point are deleted, as on resume.
- `fallback="refuse"` raises with the record, and the last restart point
  stays resumable.
- **Tests:** CPU with an injected capture error at step k: the run
  continues from the canonical point before k, the chunks are not
  duplicated, the chain records both revisions; GPU: the same with a
  deliberately broken capture on MH-1 hybrid NaCl.

**5-T09 Calibration (D5).**
- Under (a): `planner/calibrate.py` lists the tied candidates, runs each in
  a child through the subprocess runner's machinery (50 steps after
  warm-up, A B A B), applies the margin, and returns the chosen plan with
  `Plan.calibration`. Under (b) the same without the choice.
- `PlannerOptions(calibrate=)`, `fm.md(calibrate=)`, `fastmace run
  --calibrate`; the report's measured line.
- **Tests:** CPU with fake timers: the margin, the "every pair" rule,
  no tie means no calibration, a user-set backend is never calibrated;
  that no file is written outside the run directory and no earlier
  record is read (a test with a planted record in the cache directories).
  GPU smoke (local, timed only as a check): on NaCl with MH-1 the
  calibration keeps hybrid, and its setup cost is recorded.

**5-T10 Two-stage `auto` as rules** (only under D6 (b)).
- `_stage1` and `_stage2` as rule-table entries; `tools/plan_corpus.py`
  shows the choices identical for every label × `main.py` configuration.

**5-T11 Acceptance (decision 39 as amended).** Outline below; the runbook
`docs/validation/phase-5-runbook.md` comes first and is accepted before the
freeze. Report `docs/validation/phase-5.md`; the owner tags `fastmace-5`.

## Order and parallelism

```text
Wave 0   T01 contracts
Wave 1   T02 conformance   T03 rule 12 (estimate first; action after D1)   T04 rule 15   T05 str(plan)   T06 monitor
Wave 2   T07 tier switch (T06, D2, D3)   T09 calibration (T05, D5)   T10 (D6 (b) only)
Wave 3   T08 mid-run fallbacks (T07, D4)
Wave 4   T11 acceptance
```

- **Start without any decision:** T01 at once; T02, T04, T05 and T06 on
  T01's merge; T03's estimate and its GPU validation; T07's switch
  machinery (the revision, the hand-over, the subprocess path) with the
  trigger stubbed.
- **Wait for a decision:** T03's action (D1), T07's trigger and limits (D2,
  D3), T08 (D4), T09 (D5), T10 (D6).
- **Critical path:** T01 → T06 → T07 → T08 → T11.
- **GPU queue:** phase 4's acceptance first. Then T03's local peaks
  (untimed) and T07's GPU tests; the timed items (T06's A/B, T07's storm and
  anneal checks, T09's smoke) in one quiet stretch. T03's large models go
  to the Colab A100 in parallel.
- At most four workers at once.
- Estimate: 2 weeks with D1–D6 at (a); D5 (c) and D6 (a) take about 3
  days off.

## Merge orders where files overlap

| File | Editors, in merge order |
| --- | --- |
| `planner/plan.py` | T01, T05 |
| `planner/minimal.py` | T01, T02, T04, T03, T07, T09, T10 |
| `planner/fixed.py` | T02, T10 |
| `simulation.py` | T01, T06, T07, T08, T09 |
| `provenance.py` | T01, T05, T06 |
| `report.py` | T04, T06, T07, T09 |
| `engine/graph.py` | T06 |
| `planner/adapt.py` | T06, T07 |
| golden `str(plan)` files | T05, then each of T03, T07, T09 updates its section |

T02 changes reason texts and numbers only, so it merges before the tasks
that add decisions. A later task that adds a rule reason adds its
conformance test in the same branch.

## Rules carried over (owner, 22 October 2026)

- **Tests and merges:** workers run only the tests their change touches;
  the full CPU suite runs once per merge, in the conductor's batched trial
  merges; one full CPU suite on the PC at a time.
- **GPU tests:** trials run on the CPU only, so a task whose change is
  covered by GPU tests runs the affected ones itself.
- **GPU jobs:** through the worktree's `tools/ci/local_gpu.py run`, as
  background tasks of the worker's session. A refused launch waits for the
  owner.
- **Timing:** repeats sized to the effect (decision 39). Cross-process
  controls use the 3 % noise floor or are left out. Timed runs on the RTX
  5060 Ti only.
- **Colab:** through `tools/colab/` (branch `fastmace/ops-colab` until it
  merges) for untimed runs:
  T03's large-model peaks and acceptance's compatibility checks. Each VM
  is stopped afterwards.
- **Workstations:** off until the owner says they are back.
- **Branches:** workers push their own `fastmace/5-Tnn` branches.
- **Small, low-risk tasks** (T04, T05) skip the prestart report and the
  separate validation document.

## Acceptance runbook outline (5-T11)

- **Freeze and baseline:** the main commit with every phase 5 task merged;
  baseline the tag `fastmace-4` (or 4-T13's freeze, named in the report).
- **§16's item, on the CPU:**
  1. the conformance suite passes, and the meta-test shows every rule
     1–15 with at least one test and every precedence pair of 5-T02's
     matrix;
  2. the CPU suite green, `-m acceptance` on the CPU included.
- **No regression (choice 7):**
  3. **plan corpus** (CPU, `tools/plan_corpus.py`): every label × the
     standard workloads (NVT, NPT, NPT + D3, MC swap, relaxation, fixed and
     variable cell) planned at the baseline and at the freeze with the same
     injected hardware and software. Tier, backend, rewrites, sizes and
     precision identical; only texts and new fields differ. A difference is
     listed and explained by a rule (rule 12 or 15) or fails;
  4. **bitwise:** Langevin and BAOAB trajectories against the baseline,
     CPU float64 and reproducible on the GPU (as 4-T13);
  5. **timed, local, quiet PC:** MH-1 hybrid Langevin NVT, NaCl and NbCl5,
     the full alternating design (effect sought under 1 %), since T06 adds
     host timestamps around captures.
- **The new behaviour, on the RTX 5060 Ti:**
  6. adaptation end to end: T07's storm test, the switch at a canonical
     point, the report line, `counters.tier_switches`, a resume after the
     switch; the production anneal profiles never switch (timed items as
     in T07, one block each);
  7. mid-run fallback: T08's broken capture ends on a validated revision
     from the committed point, no duplicate output;
  8. rule 12: T03's validation cited, rerun only if `planner/memory.py` or
     a memory term changed since;
  9. calibration (D5 (a) or (b)): the NaCl smoke, its record and setup
     cost; its choice is judged only on the reference GPU;
  10. `str(plan)`: the golden texts, and one real plan per item 6–9
     printed into the report.
- **Full acceptance tiers:** `-m acceptance` and `-m "slow and not
  acceptance"` on the RTX 5060 Ti, after the stale-test listing against
  `fastmace-4` (phase 3's lesson). Their untimed share may split by file
  with the Colab A100; step-time assertions stay local.
- **Compatibility check** (`-m compat`, 30–60 min each): local Blackwell,
  Colab L4 for Ada, Colab A100 if it is a production type. The compat tier
  gains: the planner's rule 6 check on that GPU (fused cuEq loads), rule
  12's estimate against the measured peak on MH-1 NaCl, and the forced
  re-capture switch smoke (correctness only).
- **Order:** CPU items and the corpus first (no GPU); preflight; the timed
  stretch (5, 6's timed parts, 9) alone; then the tiers overnight. Colab
  checks in parallel from the freeze.
- **Recorded, not judged:** calibration's measured step times off the
  reference GPU; the re-capture time fractions of the production profiles.

## Checkpoints for the owner

1. Before the go messages: D1–D6. T01, T02, T04, T05, T06 and T03's
   estimate start before the answers.
2. After T03's validation: whether the estimate's accuracy supports D1's
   margin.
3. After T07's timed storm check: whether the trigger's 20 % holds, or
   D2 moves to (c) or (d).
4. After T11: the acceptance report, and the tag `fastmace-5`.
