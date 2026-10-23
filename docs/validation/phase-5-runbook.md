# Runbook: phase 5 acceptance

Plan task 5-T11. **Revision 1, a draft**, written 23 October 2026 before any phase 5 task other than 5-T01 was
merged, and before the owner's decisions D1–D6. Items that depend on an open decision are written for each option
and marked **(D*n*)**; revision 2, at the go, keeps the decided branch and folds in the task reports. Every rule
below is fixed before any run at the freeze, and nothing runs before it. Results go to `phase-5.md` and
`results/5-T11-*/`. 5-T11 changes no library code; its tool and test changes are labelled tool-side or test-side.

Sources: SPEC §16 row 5, §7.7, §9.1–§9.6, §10.6, §13, §18 decisions 3, 29, 35, 38, 39 (amended) and 50;
`plans/phase-5.md` (D1–D6, the conductor's choices 1–8, task 5-T11 and its outline); `docs/design/planner-5.md`
(5-T01); the task reports `5-T02.md` to `5-T10.md` as they land; `phase-4-runbook.md` and `phase-4.md` (template
and baseline).

## Decisions in force

- **§16's acceptance for phase 5:** each rule of §9.3 and the precedence order have a test.
- **Choice 7, default plans do not change:** on a run where no trigger fires, phase 5 changes no tier, rewrite,
  size or step time. Rule 12 may drop a rewrite only where its estimate says the plan does not fit.
- **Choice 8 and decision 39 (amended):** full acceptance on the RTX 5060 Ti; the compatibility check (`-m compat`)
  on Blackwell (local), Ada (Colab L4) and the Colab A100 if phase 4 made it a production type. Untimed runs may
  use the Colab A100; timed runs stay local.
- **D1–D6:** open at revision 1. The items they shape: D1 item 8 and the corpus's rule 12 class; D2 and D3 item 6;
  D4 item 7; D5 item 9; D6 item 3's scope (under D6 (b) 5-T10's corpus is the same tool, run at the freeze).

## Freeze, baseline and copies

- **Freeze:** the main commit with every phase 5 task merged, named by the conductor. The report records it and the
  commit whose `fastmace/` it equals.
- **Baseline:** tag `fastmace-4` if the owner has set it by the freeze; otherwise phase 4's acceptance freeze,
  main **dae3848** (conductor, 23 October 2026), and the report says so. Exported as the renamed package with
  `python -m fastmace.bench.versions export --ref <baseline> --name fastmace_4 --destination ~/fastmace-5-T11/versions`.
- **Pinned copies:** `git -c core.autocrlf=false archive` of the 5-T11 head with `.source_commit`, `fastmace/`
  equal to the freeze's, at `~/fastmace-5-T11/src-<sha>` (ext4, not `/mnt/d`: 1b-T21b and 4-T10 measured `/mnt/d`
  distorting timing). A later test correction makes a new copy; the report lists each copy and what ran from it.
- **Environments:** CPU `~/environments/fastmace_cpu` (suite, corpus, CPU trajectories); local `mace_cueq_oeq`
  (MH-1 hybrid, timed items) and `mace_cueq_latest` (plain cuEq, pytest, float64 and reproducible). The report
  records torch, Triton, cuEquivariance, openequivariance, MACE and ASE versions per environment.
- **Every result names** the machine, GPU, driver, environment, backend, precision, model label (MH-1
  `matpes_r2scan` unless stated), the integrator with its arguments, `adapt` and `calibrate` settings, and the
  commits of the freeze and the baseline.

## Hardware and budget

- **The full acceptance runs on the local RTX 5060 Ti** (workstations off), through
  `tools/ci/local_gpu.py run --wait-minutes 600` from the pinned copy, as background tasks of the 5-T11 session. The
  queue is FIFO; no other session's job is stopped.
- **Colab** (`tools/colab/`, each VM stopped afterwards): the L4 compatibility check (Ada, correctness only); the
  A100's compatibility check if A100 is a production type after phase 4; on the A100 also the untimed share of the
  tiers, split by file. A step-time assertion that runs on the A100 is reported, not judged.
- **Timed jobs alone, on a quiet PC:** no other GPU job (the reservation) and no heavy CPU job, checked and recorded
  at each job's start; the conductor is told the window so that no CPU suite runs in it.
- **CPU work** (corpus capture, CPU suite) never during a timed stretch, the 5-T11 ones or another task's.

## Before any run: stale expectations (phase 3 lesson)

Trial merges are CPU-only, so GPU tests go stale.

1. **List** every GPU test marked `slow` or `acceptance`, and for each whether its file or the library code it
   exercises changed between the baseline and the freeze.
2. **Read** each changed-path test's expectations. Expected from the plan's file list:
   - reason texts and rule numbers (5-T02): tests that match a refusal or decision text, e.g. `test_planner_gpu`,
     `test_simulation_gpu`, `test_reference_tier_gpu`, `test_polar_gpu`;
   - plan and provenance shapes (5-T01, 5-T05, 5-T06): `schema_minor`, `counters.tier_switches`,
     `counters.segments`, `plan.text`, `plan.text_full`, `plan.calibration`, `plan.adaptation`; tests that compare
     `str(plan)` or a provenance key set;
   - the report's lines (5-T04 rule 15 from the plan, 5-T06 re-capture lines, 5-T07 switch, 5-T09 calibration);
   - rule 12's decisions (5-T03): a plan on a large label whose rewrite set rule 12 now changes;
   - `engine/graph.py`'s re-capture timestamps (5-T06): `test_engine_gpu`, `test_recapture*`;
   - the segment hook and the switch in `simulation.py` (5-T06, 5-T07, 5-T08): resume and restart tests
     (`test_canonical_restart_gpu`, `test_reproducible_gpu`, `test_reproducible_schemes_gpu`), the subprocess-runner
     and worker tests (`restart_worker()` gains a caller).
3. **Collect** which of them each phase 5 task ran on the GPU at its own head (task reports). A test whose covered
   code changed after that goes into `preflight`.
4. **Run `preflight`** (untimed). A failure is triaged before the timed jobs: a stale expectation is corrected
   test-side, in its own commit, with the reason and the change that made it stale; a library defect stops the run
   and goes to the conductor. A failure is reproduced on the production path (`Simulation`, `fastmace run`) before
   it is called a library defect.

Pytest jobs run through `tools/acceptance/stoppable_pytest.sh` (results per finished test in `results.jsonl`,
`junit.xml` also when stopped); a stop is `kill -INT "$(cat OUTDIR/pytest.pid)"`.

## The plan corpus (item 3; tool-side `tools/plan_corpus.py`)

Choice 7 made checkable on the CPU: the same inputs planned by the baseline and by the freeze.

- **One tool with 5-T10** (conductor, 23 October 2026): `tools/plan_corpus.py` is 5-T10's under D6 (b) and
  5-T11's under D6 (a); both use one format (cases with stable ids in named sets, a `package=` argument so the
  renamed `fastmace_4` export plans the same cases, canonical JSON records, `dump` and `compare
  --explain-codes`). 5-T11 adds the `"acceptance"` set.
- **Model inputs:** per label of `fastmace.model.labels` (19), the stand-ins of `tests/cpu/test_planner.py`'s
  `LABEL_FACTS` (read from the released checkpoints with `model_traits`), with the ForceField's facts the planner
  reads (precision, superset, `rewrites="auto"`, the D3 record where it applies). The dimensions rule 12's
  estimate needs (5-T03's names) are captured **once** from the checkpoints, on the CPU in WSL, one label per
  process (free memory checked first; not during a timed stretch), into a JSON fixture with each checkpoint's
  sha256. A field one side lacks is filled with that side's default and listed.
- **Hardware and software (injected, documented):** `Hardware("NVIDIA GeForce RTX 5060 Ti", (12, 0), 16 GiB)` with
  12 GiB free (the desktop takes about 4 GiB; conductor, 23 October 2026), and `SoftwareTraits` with OEq, Triton,
  NVRTC and fused cuEq available. **One low-memory case** (a 4 GiB GPU, 3 GiB free) on the largest labels so that
  rule 12's path is exercised.
- **Workloads,** per label (where the label's family allows; a refusal is a result, compared like a plan): MD NVT
  (Langevin), NPT (BAOAB), NPT with D3 (labels without a D3 record: D3-BJ with r2SCAN's parameters, explicit), MC
  swap with a superset, relaxation at fixed cell (LBFGS), relaxation at variable cell (LBFGS with
  `FrechetCellFilter`; POLAR is refused on both sides by decision 50, expected); each with the default backend and,
  where both pass the hard rules, with the other backend set explicitly. Structures: NaCl and NbCl5 3×3×3.
- **Compared:** refusal codes; tier and mode; backend and convolution fusion; runner; the rewrites on and off
  (names, not texts); force and MC energy precision; skin, pair-buffer headroom, edge bucket, re-capture budget and
  variant cap; D3 skin and list; k-space headroom; the stress path. **Ignored:** reason texts, rule numbers, and the
  phase 5 fields (`calibration`, `adaptation`, `predicted_memory_bytes`, `text`).
- **Verdict per row:** identical, or **explained**: every difference is carried by a decision whose code belongs to
  rule 12's action **(D1)** or rule 15's warning. Anything else fails and goes to the conductor. Expected
  explained rows: none on the 16 GiB record unless 5-T03 finds a label that does not fit; the low-memory case's
  rule 12 rows.
- **Output:** `results/5-T11-corpus/corpus.json` (every row, both sides) and a table in the report (counts per
  verdict, every non-identical row).

## Jobs

Local RTX 5060 Ti, run ID `5-T11-local-<date>`, `python3 <copy>/tools/ci/local_gpu.py run --wait-minutes 600
--purpose "5-T11 <job>" -- bash <copy>/tools/acceptance/phase5_local.sh <copy> <job>` (tool-side, modelled on
`phase4_local.sh`); outputs in `~/fastmace-acceptance/5-T11-local-<date>/`. CPU jobs run in WSL from the same copy.

| Job | Item | GPU | Timed | What runs | Estimate |
| --- | --- | --- | --- | --- | --- |
| `conformance` | 1 | no | no | `pytest tests/cpu -m rule` and `tests/cpu/test_rules_meta.py`; the meta-test's listing (rules → tests, precedence pairs) saved | 5 min |
| `cpu-suite` | 2 | no | no | the conductor's merge-trial CPU suite at the freeze, cited by commit; `pytest tests/cpu -m acceptance` here | 20 min |
| `corpus` | 3 | no | no | `tools/plan_corpus.py capture` (once), then `compare --baseline fastmace_4` | 1 h capture, 5 min compare |
| `preflight` | – | yes | no | the stale-listing selection | 1–3 h |
| `trajectories-cpu` | 4 | no | no | `tools/trajectory_versions.py --against 4`, CPU float64, tiny MH-1-class model, NaCl 2×2×2, 100 steps, every 10, Langevin NVT and BAOAB NPT, D3 off and on; `compare --bitwise` | 15 min |
| `trajectories-reproducible` | 4 | yes | no | the same, reproducible plan, plain cuEq float64, MH-1 NaCl, 500 steps, every 50, NVT and NPT, D3 off | 20 min |
| `restart-across` | 4 | no | no | `tools/restart_versions.py`: a Langevin restart written by the baseline (format 4, step 50) resumed at the freeze to step 100, CPU float64, tiny model; bitwise against the baseline's uninterrupted run, without `allow_plan_change` and without a new revision | 10 min |
| `regression` | 5 | yes | yes | `tools/ab_fastmace.py --release 4`: MH-1 hybrid Langevin NVT, D3 off, NaCl and NbCl5, 300 K, 12 blocks of 300 steps after 300; ABAB 4 first, ABAB 5 first, ABBA 4 first (4-T13's protocol) | 40 min |
| `adapt-storm` | 6 | yes | yes (D2) | 5-T07's forced re-capture storm, MH-1 hybrid NaCl, switched against unswitched, one block each | 30 min |
| `adapt-production` | 6 | yes | no | the NaCl and NbCl5 anneal profiles (4-T13's `npt_anneal_NbCl5_short` and its NaCl counterpart) through `fastmace run` with the default `adapt`; resume included | 1.5 h |
| `adapt-tests` | 6, 7 | yes | no | 5-T07's and 5-T08's GPU tests (switch at a canonical point, statistics against an unswitched compiled run, resume after a switch, broken capture) | 1 h |
| `calibration` | 9 | yes | yes (D5) | 5-T09's NaCl smoke with MH-1: the record, the choice, the setup cost | 20 min |
| `plans` | 10 | yes | no | one real plan printed per item 6–9 (`str(plan)` and `describe(full=True)`), from the runs above | 5 min |
| `compat` | – | yes | no | `tools/acceptance/compat.py --provider local --label "RTX 5060 Ti"` | 45 min |
| `acceptance-tier` | – | yes | (step-time assertions) | `pytest tests/gpu -m acceptance` | 5–6 h |
| `slow-tier` | – | yes | no | `pytest tests/gpu -m "slow and not acceptance"` | 2–3 h |

**Order:** the CPU jobs and the corpus first (no GPU); `preflight`; the timed jobs (`regression`, `adapt-storm`,
`calibration`) in one quiet stretch; then the rest, the acceptance and slow tiers overnight. Colab, in parallel from
the freeze: the L4 compatibility check (`5-T11-colab-l4-<date>`) and, if it applies, the A100's
(`5-T11-colab-a100-<date>`) with its untimed share.

## Timing design (decision 39: repeats sized to the effect)

- **Item 5** (effect sought: under 1 %, since 5-T06 adds host timestamps around captures and a segment hook): the
  full same-process alternating design, three processes, the local floors (NaCl 1.1 %, NbCl5 0.4 %), as 4-T13.
- **Item 6, the storm (D2 (a), (b), (c)):** the effect expected is large (compiled against re-capturing every few
  steps): one block each side, one process per side; the switched run must be faster beyond the 3 % cross-process
  floor. Under D2 (d) there is no switch and the job only records the warning.
- **Item 9, calibration:** a check, not an A/B: the smoke's own measured candidates are recorded; its choice is
  judged only on the RTX 5060 Ti. Off the reference GPU its numbers are recorded, not judged.
- **Cross-process controls** are bounded at 3 % or left out.

## Pass rules (fixed before the runs at the freeze)

**Item 1: conformance** (§16).
- `pytest -m rule` green at the freeze; the meta-test (`test_rules_meta.py`) green with its expected-failure list
  empty; its listing shows every rule 1–15 with at least one test, every code of `planner/rules.py` emitted with
  its number, and every pair of 5-T02's precedence matrix: rule 4 against 8 and 9; rule 5 against 9, 11, 12 and
  adaptation; rule 3 against an explicit plain compiled backend; rule 1 against `allow_reference` with MC and
  constraints; 11 then 12; a user override against a hard rule (refused, never downgraded).

**Item 2: CPU suite.** The conductor's trial at the freeze green (cited by commit and counts); `pytest tests/cpu -m
acceptance` green here.

**Item 3: plan corpus.** Every row identical or explained (above). Zero unexplained rows.

**Item 4: bitwise against the baseline.**
- CPU float64 (NVT and NPT, D3 off and on) and the reproducible GPU plan (NVT and NPT, D3 off): every row's
  positions, energy, temperature (under NPT, cell and pressure) equal.
- `restart-across`: bitwise equal to the baseline's uninterrupted run, resumed without `allow_plan_change` and with
  no new plan revision.
- The freeze side runs with the default `AdaptationConfig`; under a reproducible plan no switch happens (rule 5;
  D3 (a), (b), (d)). **(D3 (c))**: a switch is allowed under `reproducible=True` with a deterministic trigger; the
  runs here do not trigger it, and the job checks `counters.tier_switches == []`.

**Item 5: step time, no regression.**
- Per structure, the median of the three processes' ratio 5/4 within ± the local floor, and no process slower than
  1 + 2 × floor.
- Plans: only the phase 5 fields may differ (`adaptation`, `calibration`, `predicted_memory_bytes`, the plan text,
  `schema_minor`), and no tier switch, fallback or calibration in either side's run; any other engine-option
  difference makes the process invalid.

**Item 6: runtime adaptation** (§9.6; D2, D3).
- **D2 (a), (b) or (c):** on the storm the switch happens at the first canonical point after the trigger (forced if
  the cadence puts it later; choice 2), as a new revision with step and reason; `counters.tier_switches` has one
  entry `{step, from, to, reason, revision}`; the report line and the `fastmace.planner` log line are present; the
  candidate was validated before activation (self-checks; the gate where the run asked for one; D3 (a)).
  Energies and temperatures after the switch statistically equal to an unswitched compiled run (5-T07's measure);
  interrupted against uninterrupted across the switch, statistical. A resume after the switch runs the compiled
  revision and, under D3 (a), does not switch again. Timed: the switched run is faster (above).
  **D2 (b)** adds: the skin re-plan's revision, recorded the same way.
- **The production profiles** (NaCl, NbCl5 anneal): no switch, `counters.tier_switches == []`, on every option.
  The re-capture time fraction per segment is recorded.
- **D3's limits:** checked by 5-T07's CPU tests in the suite (cited), plus on the GPU: no switch with an explicit
  `mode=` and none under `reproducible=True` (D3 (a), (b), (d)).
- **D2 (d):** no switch exists; the item checks the warning and its recommendation on the storm, and that the
  production profiles do not warn.

**Item 7: mid-run fallbacks** (§9.4 step 5; D4).
- **D4 (a) or (b):** the broken capture (5-T08's GPU test, MH-1 hybrid NaCl) ends on a validated revision that
  started from the last committed canonical point; the chain records both revisions; output chunks after the point
  were deleted and written once (no duplicate or missing frame). **(b)** adds the failed overflow redo.
- **D4 (c):** the run stops with the record and a resumable restart point; a resume with `allow_plan_change=True`
  completes.
- On every option: an out-of-memory error and a non-finite value are not fallback triggers (the run stops, with
  rule 12's estimate in the out-of-memory message under D1 (a), (d)).

**Item 8: rule 12** (D1).
- 5-T03's validation (`docs/validation/5-T03.md`) cited: the estimate never below the measured peak on any of the
  19 labels × NaCl, NbCl5 × NVT, NPT, and at most 1.5× above. Rerun (the same tool, the same cases) only if
  `planner/memory.py` or a rewrite's memory term changed between 5-T03's head and the freeze; the report gives the
  diff.
- The action per D1, from 5-T03's CPU tests (cited) and the corpus's low-memory rows: **(a)** rewrites with an
  edge-sized term dropped largest first until it fits, each with rule 12's reason, then a warning; never a change
  of rule 11's sizes, the tier, the precision or an explicit user setting; **(b)** a warning only; **(c)** a
  refusal naming the rewrites (with §9.3 amended); **(d)** (a) plus the one re-plan after an out-of-memory error.
- In `compat` on each GPU: the estimate against the measured peak on MH-1 NaCl (5-T03's test).

**Item 9: calibration** (D5).
- **D5 (a):** on NaCl with MH-1 the smoke keeps hybrid (or changes it only with a margin above 5 % in every pair,
  judged on the RTX 5060 Ti only); `Plan.calibration` records each candidate's measured step times; the setup cost
  is in the report; nothing is written outside the run directory and nothing is read from earlier runs (5-T09's
  CPU test with a planted record, cited).
- **D5 (b):** the same record; the plan is the rules' plan (`changed_plan` false).
- **D5 (c):** no item; the report cites the SPEC change (§9.1, §7.7, §17).

**Item 10: `str(plan)`.** 5-T05's golden texts green in the CPU suite (cited), each section present, no line over
120 characters, at most about 30 lines; the real plans of items 6–9 printed in the report, each with the section
its task filled (switch in the history, fallback revision, memory line, calibration record).

**Tests:** every acceptance and slow GPU test passes on the RTX 5060 Ti at the freeze or at a corrected copy
(corrections listed with reasons; the A100's share counts there); the compatibility check passes on the RTX 5060
Ti and the Colab L4 (and on the A100 if that leg runs).

## The compatibility tier's additions (test-side, `tests/_support/tiers.py`)

`COMPAT_TESTS` gains the GPU tests the tasks write (conductor, 23 October 2026: 5-T03 and 5-T07 write them,
5-T11 lists them and checks that `tests/cpu/test_tiers.py` collects them):
- rule 6 on that GPU: the planner's fused-cuEq check loads (`test_loading_gpu.py` already holds the plain cuEq case;
  the planner-side check is added if no existing test asserts the decision);
- rule 12: the estimate against the measured peak on MH-1 NaCl (5-T03's test);
- the forced re-capture switch smoke, correctness only (5-T07's test; under D2 (d) the warning smoke).

## Tool-side additions (before the freeze, CPU-tested on `fastmace/5-T11`)

- `tools/plan_corpus.py` (`capture`, `compare`) and its CPU test on the tiny models (both sides the freeze's
  package, so every row is identical; a planted difference is caught; an explained one is classified).
- `tools/acceptance/phase5_local.sh`: the jobs above.
- `tools/ab_fastmace.py --release 4`, `tools/trajectory_versions.py --against 4`, `tools/restart_versions.py` with
  the baseline `fastmace_4` (format 4 on both sides).
- `tests/_support/tiers.py`: the compat additions.

## Recorded, not judged

- Calibration's measured step times off the reference GPU.
- The re-capture time fractions and rebuild rates per segment of the production profiles.
- The corpus's rule 12 decisions on the 16 GiB record (which labels come close to the free memory).

## Not in this runbook

- New rules beyond §9.3's fifteen (plan, "Out of scope"); a gap the conformance suite found goes to the conductor.
- ResearchModule's `main.py` (no change in phase 5).
- Judging physics (decision 6).
- 5-T03's large-model peaks on the Colab A100 are 5-T03's and are cited.
