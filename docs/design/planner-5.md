# Contracts of phase 5: planner completion

Plan task 5-T01, 23 October 2026. Spec: SPEC §9.1, §9.3, §9.4 step 5, §9.5, §9.6, §13; decisions 3, 29,
35, 38, 39 and 50 (SPEC §18); `plans/phase-5.md`. Owner decisions D1–D6 of that plan are **open**: no
contract here chooses one of their options. Where a contract depends on one, every option fits it, and a
value that only a decided option could give is refused at run time (code `adaptation-pending` or
`calibration-pending`) until the task implementing it lands.

Tests: `tests/cpu/test_contracts_5.py` pins the names below, `tests/cpu/test_rules_meta.py` checks the rule
table against the code and the tests, `tests/cpu/test_config.py` the new fields against `switch-map.md`
(section 12), and `tests/cpu/test_metadata_schema.py` the schema additions.

**Default plans do not change** (conductor's choice 7): every new field defaults to today's behaviour. No
planner decision, tier, rewrite, size or step changes with this task.

## The rule table (`fastmace/planner/rules.py`; choice 1)

`RULES` holds §9.3's fifteen rows as `Rule(number, kind, title, codes, allow)`:

| Rule | Kind | `allow` | Codes |
| --- | --- | --- | --- |
| 1 | hard | `allow_reference` | `non-3d-pbc`, `constraints`, `mc-with-constraints`, `variable-cell-without-barostat`, `fix-atoms-variable-cell`, `fix-atoms-with-fixcm`, `scheme-unknown`, `scheme-ensemble`, `scheme-cell-mode`, `scheme-dof`, `scheme-cell-filter`, `polar-variable-cell-optimizer`, `polar-unqualified-source`, `allow-reference`, `no-allow-reference` |
| 2 | hard | — | `species-not-in-superset` |
| 3 | hard | — | `polar-plain-compiled-cueq` |
| 4 | hard | — | `inexact-padding`, `mode-compiled-rule-4`, `compiled-exact-edges`, `eager-exact-edges` |
| 5 | hard | — | `reproducible-tier`, `reproducible-family-not-qualified`, `reproducible-backend`, `reproducible-conv-fusion`, `reproducible-cueq-version`, `reproducible-kernel-not-deterministic`, `reproducible-qualified`, `reproducible-environment`, `backend-cueq-reproducible` |
| 6 | hard | `allow_naive_cueq` | `naive-cueq`, `fused-cueq`, `allow-naive-cueq` |
| 7 | hard | — | `atom-count-changes` |
| 8 | preference | — | `superset`, `species-changes` (5-T02) |
| 9 | preference | — | `host-callbacks-graph-sync`, `host-callbacks-compiled`, `no-host-callbacks` |
| 10 | preference | — | `rule-10-optimizer` |
| 11 | preference | — | `rule-11`, `rule-11-fixed`, `rule-11-md-only`, `rule-11-optimizer`, `rule-11-d3` (an explicit value is an `override`, 5-T02) |
| 12 | preference | — | `rule-12-d3`, `rule-12-d3-memory`, `rule-12-d3-warn`, `rule-12-memory`, `rule-12-memory-drop`, `rule-12-memory-warn` (5-T03: the estimate and decision 51's action) |
| 13 | reaction | — | `selfcheck-failed`, `selfcheck-dependant` |
| 14 | reaction | — | `fallback-exhausted`, `fallback-rewrite-off`, `fallback-rewrite-on`, `runner-subprocess-fallback` |
| 15 | reaction | — | `model-structure-warning` (reserved for 5-T04) |

The codes are those the library emits today for each rule (a static scan of `fastmace/`), plus the two
reserved ones. A code belongs to one rule. Codes of no rule stay outside the table: option requests
(`hybrid-unavailable`, `rewrite-requirement-missing`, ...), process constraints (`runner-subprocess-*`
of §8.7) and run requests (`schedule-above-max-temperature`, ...).

Functions: `rule(n)` (`KeyError` outside 1–15), `codes(n)`, `kind(n)`, `allow_flag(n)`, `numbers(kind)`,
`rule_of(code)`, `check_reason(reason)`, `override(n, would_give, detail=None)`, `overridden(reason)`;
constants `NUMBERS`, `KINDS`, `OVERRIDE_CODE = "override"`.

Codes later tasks add, each in the branch that emits them:
- 5-T03: rule 12's action codes, once **D1** is decided (for example a rewrite dropped, a warning, a
  refusal under D1 (c), or the out-of-memory re-plan of D1 (d));
- 5-T04: emits `model-structure-warning` (decisions `warn.model_structure...`);
- 5-T08: rule 14's mid-run codes, once **D4** is decided.

### `Reason` checks (`errors.py`)

`Reason.rule` is `None` or one of 1–15. With a rule, `Reason.code` is one of that rule's codes, or
`OVERRIDE_CODE` for a preference. A rule's code never comes without its number. `check_reason` returns
what is wrong (`()` when nothing is). The planner never calls it, so a reason costs nothing at run time;
`test_rules_meta.py` does:

- `test_every_numbered_reason_uses_a_code_of_its_rule`: passes now;
- `test_a_rule_code_carries_its_number[n]`: expected failure (strict) for rules 1, 5, 6, 8, 9 and 14,
  whose codes are still emitted without the number somewhere (`NUMBERLESS`);
- `test_every_rule_has_a_marked_test[n]`: expected failure (strict) for every rule (`UNTESTED`); no test
  carries the marker yet.

5-T02 numbers the reasons and marks its conformance tests, and empties both lists. A strict expected
failure that passes fails the suite, so neither list can go stale.

### Marker

`pytest.mark.rule(n)` (`pyproject.toml`): the test checks rule n. A test may carry several. The meta-test
finds them statically in `tests/`.

`pytest.mark.precedence(a, b)` (5-T02): the test checks that rule `a` wins over `b`, a later rule,
`"override"` (a user's explicit value) or an `allow_*` flag. `test_rules_meta.PAIRS` lists 5-T02's matrix;
the meta-test checks that every pair has a marked test and every marked pair is listed. 5-T03 (rule 5
against the MACE model's rule 12) and 5-T07 (rule 5 against adaptation) add their pairs.

**After 5-T02:** `NUMBERLESS` is empty; `UNTESTED` holds rule 15 until 5-T04 emits and tests it. The suite is
`tests/cpu/test_rules_{hard,preferences,reactions,precedence}.py`; `tests/_support/rule_cases.assert_conforms`
runs `check_reason` on every reason of a plan or refusal, which covers reasons built elsewhere and passed
through `_Book.refuse` (the static scan cannot read those). The overrides of rule 11 (skin, headroom,
re-capture budget, D3 skin) use `rules.override`; an explicit `mode="compiled"` keeps `mode-compiled-tier`,
which the report reads, and says that it was requested.

## Overrides (choice 4)

A decision the caller replaced, for a preference (rules 8–12), has the reason `rules.override(n, X)`:
text `override; rule n would give X` (optionally `: detail`), `rule=n`, code `override`. In `str(plan)`
it reads `key = choice  (override; rule n would give X)`. `Plan.overrides()` lists
`(key, choice, rule, X)`.

`Plan.requested` keeps every value the caller set other than `"auto"`, as today. An override decision's
key has a `requested` entry; a requested value equal to the rule's choice is no override. A hard rule has
no override: `rules.override` raises for rules 1–7 and 13–15, and a request against a hard rule stays a
refusal. The overrides that exist today keep their decision keys; 5-T02 moves their reasons to this form
(texts only, no choice changes).

## Config records (`config.py`)

`config.PENDING = "pending"` marks a field whose meaning an open owner decision fixes.

### `AdaptationConfig` (SPEC §9.6; D2, D3)

| Field | Default | Meaning | Decided by |
| --- | --- | --- | --- |
| `adapt` | `"auto"` | `"auto"`: adapt as §9.6 and the fields say; until 5-T07 this records only. `"off"`: nothing switches | — |
| `trigger` | `"pending"` | `"recapture-time"` (D2 (a), (b)), `"recapture-budget"` (D2 (c)), `"none"` (D2 (d)) | D2 |
| `recapture_time_fraction` | `0.2` | the re-capture share of wall time that, with the budget exceeded, triggers; read by `"recapture-time"` only | D2 (a), (b) |
| `skin_replan` | `"pending"` | rule 11 re-planned from the measured rebuild rate | D2 (b) |
| `max_switches` | `"pending"` | switches per run, a resume counting its chain's; `None`: no limit | D3 |
| `switch_back_after` | `"pending"` | quiet segments before switching back to graph; `None`: one way | D3 (b) |
| `min_points_between` | `"pending"` | canonical points between switches; `None`: no minimum | D3 (d) |
| `reproducible` | `"pending"` | switches allowed under `reproducible=True` | D3 (c) |

`ADAPTATION_TRIGGERS`, `ADAPTATION_DECISION_FIELDS`, `AdaptationConfig.decided()` (the decision fields
not `"pending"`).

D3 (a)'s other limits are not fields, since every option keeps them: no switch during an MC block, none with
an explicit `mode=` (the user's override of rules 4 and 9), none on a subprocess child without the
parent's `restart_worker()` hand-off. Validation before activation follows choice 3: the self-checks
always, the gate by the same policy as an unoverridden plan (opt-in for MD, decision 3; the move-energy
part for MC, decision 35).

### `CalibrationConfig` (SPEC §9.1; D5)

| Field | Default | Meaning | Decided by |
| --- | --- | --- | --- |
| `calibrate` | `False` | `True` measures the tied candidates; refused until 5-T09 | — |
| `steps` | `50` | measured steps per candidate after warm-up (§9.1) | — |
| `margin` | `0.05` | the "faster in every pair" margin of D5 (a); unused under (b) | D5 (a) |
| `choose` | `"pending"` | `True`: the measurement may change the plan (a); `False`: record only (b) | D5 |

Under D5 (c) (no calibration in 1.0) 5-T09 removes the record and its switch-map rows.

### `PlannerOptions` and `Simulation` (`simulation.py`)

`PlannerOptions.adaptation` and `.calibration` (defaults `AdaptationConfig()`, `CalibrationConfig()`),
`planner_options(adaptation=, calibration=)` and `Simulation(adaptation=, calibration=)`. `to_record` writes
both into `simulation.json`; `from_record` reads a record without them as the defaults.

The two keywords come after `neighbours`, so `Simulation`'s planner inputs stay its last keyword-only
parameters, as phase 2's contract pins. `test_contracts_2.test_simulation_takes_the_planner_inputs` now
checks phase 2's list followed by `PHASE_5_PLANNER_OPTIONS`, and `contracts-2.md` says so. Putting them
before `mode` would have left the old test as it was, but it would have split the planner inputs. `fm.md`
takes them in 5-T09, so its contract test is unchanged. `test_contracts.test_simulation_signature`, which
pins the whole keyword-only list, lists them last as well. `contracts-1b.md` shows only 1b's signature, so
it needs no note.

**Placeholders.** `planner_options` raises `PlanRefused` (`simulation.phase_5_pending`, codes in
`PHASE_5_PENDING_CODES`):
- `adaptation-pending`: any field of `ADAPTATION_DECISION_FIELDS` other than `"pending"` (D2, D3; 5-T07
  removes it);
- `calibration-pending`: `calibrate=True` or `choose` other than `"pending"` (D5; 5-T09 removes it).

`fm.md(calibrate=)` and `fastmace run --calibrate` are 5-T09's.

## Plan (`planner/plan.py`)

- `Plan.calibration: CalibrationRecord | None = None`. `CalibrationRecord(candidates, chosen,
  changed_plan, steps, margin, setup_s)` with `CalibrationCandidate(change, step_ms)`; `change == ""` is the
  rules' own plan, `step_ms` the per-block means in the order measured. Under D5 (b) `changed_plan` is
  always `False`.
- `Plan.adaptation: AdaptationConfig | None = None`: the limits active for the plan (5-T07 sets it for MD
  plans).
- `Plan.overrides()`.
- `TierSwitch(step, from_tier, to_tier, reason, revision)`, `record()` → `{step, from, to, reason,
  revision}`.

Neither new field is in the restart record (as `runner` and `selfchecks` are not): a resume takes the
limits from the recorded `PlannerOptions`, and a calibration is a record of the run that measured.
`PLAN_RECORD_VERSION` stays 2, and the restart format stays 4 (choice 5): under D3 (a) the active
revision's chain counts the switches already made. D3 (c) would need the monitor's counters in the restart
(format 5); 5-T07 bumps it then.

## Provenance (`provenance.py`, `metadata-schema.md`)

- `schema` stays `"fastmace.metadata/1"`; the new key `schema_minor` (`SCHEMA_MINOR = 1`, second key)
  counts additions within version 1. A record without it is minor 0.
- `counters.tier_switches`: a list of `TIER_SWITCH_KEYS` records (`TierSwitch` or mappings), `[]` without a
  switch. It is a `COUNTER_KEYS` entry now; `LATER_COUNTER_KEYS` is empty.
- `plan.calibration`, `plan.adaptation`: the records above, `null` when absent.
- `SPEC_13_KEYS` lists the three.

## `str(plan)` layout (5-T05 implements it)

`Plan.describe()` prints these sections in this order, at most about 30 lines, no line over 120
characters. A section with nothing to say is one line ("none") or left out where marked.
`describe(full=True)` keeps today's list of every decision. Provenance stores both (`plan.text`,
`plan.text_full`).

1. **Header** (2 lines): `Plan revision N, tier T (why this tier)`, then backend (plain: fused convolution
   or not), force precision, MC energy precision, runner.
2. **Hard rules** (1–3 lines): the hard rules that shaped the plan, with numbers (`rule 4: compiled tier,
   exact edge counts`), and the `allow_*` flags used. Left out when none applied.
3. **Rewrites** (2–4 lines): on, by name; off, with the reason grouped by reason code.
4. **Sizes** (2 lines): skin, pair-buffer headroom, edge bucket, re-capture budget and variant cap; D3 skin
   and list; k-space headroom.
5. **Kernels** (1 line): OEq, Triton, NVRTC, fused cuEq, each available or not.
6. **Memory** (1 line, rule 12): the prediction against the GPU's memory; `not estimated` until 5-T03.
7. **Workload** (1–3 lines): the workload and its rule 8–11 decisions; overrides as `override; rule n
   would give X`.
8. **Warnings** (rule 15; 0–3 lines): each with its diagnosis. Left out when none.
9. **History** (1–4 lines): the fallback candidates tried, the calibration record (`none` until 5-T09),
   the revision chain with step and reason (switches from 5-T07).

5-T05 writes the golden files; 5-T03, 5-T07 and 5-T09 update their sections.

## Runtime monitor (5-T06; `planner/adapt.py`)

`Monitor(adaptation)` keeps the counters of one process's engine per **segment**: from one reading to the
next. `Simulation._observe(reason, canonical=True)` reads at moments that synchronise anyway, so the step
path gains no synchronisation (conductor's answers to 5-T06, 23 October 2026):

| `reason` | Where | Canonical |
| --- | --- | --- |
| `restart` | after a restart write (`save_restart`, in or out of `run()`) | yes |
| `mc-block` | after an MC block that ran | yes |
| `host-write` | after a host write's transaction (`update`, `set_*`) | yes |
| `run-end` | at the end of each `run()` call, also when it stops (non-finite, cell diverged) | no |

The first `run()` takes the baseline, so the setup's captures are not in any segment; a reading without
a baseline takes it. A reading at the same step as the previous one closes nothing.

`Segment` (`counters.segments`, `provenance.SEGMENT_KEYS`): `first_step`, `last_step`, `reason`,
`canonical`, `wall_s` (the run's loop clock between the readings), `recaptures` and `recapture_s` (the
captures counted against the budget, and their host time), `captures` and `capture_s` (all of them),
`overflows_redone`, `budget_exceeded_at` (cumulative), and `lists`: `{"mace": {rebuilds, build_time_s:
null}}`, plus `"d3"` with its measured build time when the force field has the D3 term. MACE's list is
rebuilt on a side stream (`overlap_rebuild`), so its host time is not its build time; no CUDA events are
added for it.

The graph engine times each capture on the host, from the synchronisation in `_capture` that ends the
step's queued work to the one that ends the capture (`capture_seconds`, `recapture_seconds`,
`recapture_log[...]["s"]`), and `runtime_counters()` returns what the monitor reads (cheaper than
`info()`). Other engines' counters read as 0.

API for 5-T07: `observe(info, *, step, wall_s, reason, canonical=True) -> Segment | None`;
`decide(window) -> None` (5-T07's trigger; `_observe` keeps its return value as `request`); `mark(step)`
starts the window; `window(since_step=None)` sums the kept segments since the mark (or `since_step`);
`totals()`, `segments` (the last `SEGMENT_LIMIT` = 1,000), `records()`, `neighbour_lists(...)`;
`start(info, *, step, wall_s)` re-baselines (a new engine). A switch may act only on a canonical reading.

The monitor starts empty in every process: a resume starts a new one (restart format 4, choice 5), and
takes `AdaptationConfig` from the recorded `PlannerOptions`. Under the subprocess runner the child's
monitor `export(cursor)`s each call's segments with their sum, the parent `absorb`s them, and the
parent's `metadata.json` has every call's segments. The report's `runtime` line appears only when
re-captures or overflows happened: `re-captures N (S s, P % of the loop; budget B exceeded at step K);
overflows redone M`. The step-time effect of the host timestamps is measured in 5-T11 (item 5).

## What this unblocks

- **5-T02** (conformance suite): the table, the marker, the meta-test's two lists.
- **5-T03** (rule 12): `rule-12-memory`, `Plan.predicted_memory_bytes`, section 6 of the layout; the action
  waits for D1. As built (decision 51): the prediction is max(build, run). The run term covers a force
  call, the rewrites' terms (`Rewrite.memory`) and the lists. Both terms hold the two weight copies, the
  rewrites' resident buffers (`Rewrite.resident_memory`: `block_linear`, `merged_residual`) and the
  workspace. The build term adds the largest self-check transient (`Rewrite.selfcheck_memory`, with
  `Hardware.sms`). PolarMACE's long range needs no term of its own: its 64 MiB at the build's peak were
  cuBLAS workspaces. Deviations from the
  file list: `model/capabilities.py`, `planner/fallback.py`, `planner/plan.py` and `planner/fixed.py`
  (`Hardware.sms`), and `rewrites/fusion/__init__.py` (the `segment_gemm` self-check frees each
  interaction's tensors; `docs/validation/5-T03.md`).
- **5-T04** (rule 15): `model-structure-warning`.
- **5-T05** (`str(plan)`): the layout above.
- **5-T06** (monitor): `AdaptationConfig`, `counters.tier_switches`, `TierSwitch`.
- 5-T07 and 5-T09 build on the placeholders once D2/D3 and D5 are decided.

## SPEC wording for the owner

These follow from the plan's choices but are not in SPEC.md yet:

1. §9.1 names the API `plan(calibrate=True)`; the contract is `CalibrationConfig(calibrate=True)` through
   `Simulation(calibration=)` (D5 (a) adds `fm.md(calibrate=)` and `fastmace run --calibrate`).
2. §9.6 "between segments" and §9.4 step 5 "between segments, from the last committed state": read as
   "at a canonical restart point" (choice 2).
3. §9.3 and §9.5 "the overridden plan is self-checked and gated like any other": read as the same gate
   policy as an unoverridden plan (choice 3), so an override of an MD plan runs no gate unless
   `check="gate"`.
4. §9.5: the override record's form, `choice (override; rule n would give X)` (choice 4).
5. §13: the minor version `schema_minor` within `fastmace.metadata/1`.
6. D1 (c) would make rule 12, a preference, refuse a plan. §9.3 lets only hard rules refuse, so that
   option needs a §9.3 change too.
7. Rule 1's table entry includes `polar-unqualified-source` (a PolarMACE long-range source without
   qualification), because the planner emits it with `rule=1`. §9.3's rule 1 does not name it; 5-T02 should
   confirm the number or move the code.
