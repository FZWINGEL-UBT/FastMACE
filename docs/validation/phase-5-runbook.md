# Runbook: phase 5 acceptance

Plan task 5-T11. **Revision 1, a draft**, written 23 October 2026 before any phase 5 task other than 5-T01 was
merged. The owner decided D1–D6 the same evening (recommendation (a) of each; SPEC decisions 51–56, main
f26bd41), and this draft keeps only those branches. Revision 2, at the go, folds in the task reports. Every rule
below is fixed before any run at the freeze, and nothing runs before it. Results go to `phase-5.md` and
`results/5-T11-*/`. 5-T11 changes no library code; its tool and test changes are labelled tool-side or test-side.

Sources: SPEC §16 row 5, §7.7, §9.1–§9.6, §10.6, §13, §18 decisions 3, 29, 35, 38, 39 (amended), 50 and
51–56; `plans/phase-5.md` (D1–D6, the conductor's choices 1–8, task 5-T11 and its outline); `docs/design/planner-5.md`
(5-T01); the task reports `5-T02.md` to `5-T10.md` as they land; `phase-4-runbook.md` and `phase-4.md` (template
and baseline).

## Decisions in force

- **§16's acceptance for phase 5:** each rule of §9.3 and the precedence order have a test.
- **Choice 7, default plans do not change:** on a run where no trigger fires, phase 5 changes no tier, rewrite,
  size or step time. Rule 12 may drop a rewrite only where its estimate says the plan does not fit.
- **Choice 8 and decision 39 (amended):** full acceptance on the RTX 5060 Ti; the compatibility check (`-m compat`)
  on Blackwell (local), Ada (Colab L4) and the Colab A100 if phase 4 made it a production type. Untimed runs may
  use the Colab A100; timed runs stay local.
- **51 (D1 (a)), rule 12's action:** above 90 % of the free GPU memory, the rewrites whose records declare an
  edge-sized buffer are turned off, largest term first, each with rule 12's reason; still above, the plan is kept
  with a warning, and an out-of-memory error carries the estimate. Rule 11's sizes, the tier, the precision and an
  explicit user setting never change. Shapes item 8 and the corpus's rule 12 class.
- **52 (D2 (a)), the trigger:** the re-capture budget exceeded **and** re-capturing above 20 % of the wall time over
  the segments since the last switch; the one switch is graph → compiled (same backend, rewrites, precision) at the
  next canonical point. Rebuild rate, build time and overflows are recorded, not acted on. Item 6.
- **53 (D3 (a)), the limits:** one way, at most one switch per run; none under `reproducible=True`, with an
  explicit `mode=`, during an MC block, or in a subprocess child without the parent's hand-off; validated before
  activation (self-checks; the gate when requested; the move-energy part for MC) and logged. Items 4 and 6.
- **54 (D4 (a)), mid-run fallbacks:** a capture error at a re-capture and a compile error of the compiled tier
  during `run()`, from the last committed canonical point, output as on resume. Not triggers: out-of-memory,
  non-finite values, refusals, process constraints. Item 7.
- **55 (D5 (a)), calibration:** `calibrate=True`, off by default; the backend dimension only, each candidate in a
  child, 50 steps after warm-up, alternating; it changes the choice only above 5 % in every pair; recorded as
  measured, nothing read from earlier runs. Item 9.
- **56 (D6 (a)):** `FixedPlanner`'s two stages stay; 5-T10 is closed, so 5-T11 creates `tools/plan_corpus.py`.

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

- **The tool** (`tools/plan_corpus.py`, written by 5-T11 since decision 56 closed 5-T10, in the format agreed
  with 5-T10 in the session notes, `5-T11-plan-corpus-format.md`): cases with stable ids in named sets (5-T11's
  `"acceptance"`; a phase 6 set can be added beside it), a `--package` argument so the renamed `fastmace_4` export
  plans the same cases, canonical JSON records with the reason texts apart (`texts`), `dump` and `compare
  --explain-rules 12`. Every API difference between versions sits in `_plan`; the export's renaming of string
  values (`IntegratorConfig.dof == "fastmace"`) is read back before comparing.
- **Model inputs:** per label of `fastmace.model.labels` (19), the stand-ins of `tools/plan_corpus_labels.py`
  (moved from `tests/cpu/test_planner.py`'s `LABEL_FACTS`, which imports them from there; read from the released
  checkpoints with `model_traits`), `precision="float32"`, `rewrites="auto"`. The sizes rule 12's estimate needs
  were read **once** from the released checkpoints by `tools/plan_corpus_capture.py`, on the CPU in WSL, one label
  per process, free memory checked first (24 October 2026, 00:00–00:10 CEST: 10 labels with `fastmace_cpu`, the
  POLAR, OMOL and OFF labels with `mace_cueq_latest` on the CPU, since PolarMACE needs `graph_longrange`), into
  `tools/plan_corpus_dims.json` with each file's sha256, checked against the label table. They reach the plan
  through 5-T03's `ModelDims`: the same tool stored `capabilities.model_dims(model).record()` per label, run with
  5-T03's branch (f0ddbc1, 24 October 2026, after WSL's restart), and `_with_dims` attaches
  `ModelDims.from_record` where a version's traits have `dims`. If 5-T03's record changes before it merges, the
  capture is rerun at the freeze (about 3 minutes).
- **Hardware and software (injected, documented):** `Hardware("NVIDIA GeForce RTX 5060 Ti", (12, 0), 16 GiB)` with
  12 GiB free (the desktop takes about 4 GiB; conductor, 23 October 2026), and `SoftwareTraits` with OEq, Triton,
  NVRTC and fused cuEq available; the free memory is injected through the planner's `_free_memory_bytes`. **The
  low-memory cases** (a 4 GiB GPU, 3 GiB free) on the four largest labels (OMOL-0 XL, POLAR-1-L, MP-0 large,
  OFF23 large), NVT and NPT, so that rule 12's action is exercised.
- **Workloads,** per label, NaCl and NbCl5 3×3×3 (NbCl5: the protocols' r2SCAN cell), with `backend` `auto`,
  `hybrid` and `cueq` (a refusal is a result, compared like a plan): MD NVT (Langevin, 1 fs, 300 K), NPT (BAOAB),
  NPT with the D3 list sized for a 40 bohr cutoff, MC swap (`species_changes`, minimum 250 K), LBFGS at fixed cell,
  LBFGS with `FrechetCellFilter` (POLAR refused on both sides by decision 50, expected). 700 cases.
- **Compared:** refusals as a multiset of (rule, code) with the error type; tier; runner; superset; allowances;
  every resolved config (`rewrites`, `engine`, `precision`, `neighbours`, `polar`, `integrator`, `optimizer`, field
  by field: backend, fusion, rewrites, precisions, skin, headroom, bucket, re-capture budget, variant cap, D3 skin,
  k headroom, stress path); `auto_options_off` verbatim; `requested`; the choice of every decision both sides
  have. **Not compared:** reason texts, rule numbers, decisions only the new side has (listed per row), and the
  phase 5 fields (`calibration`, `adaptation`, `predicted_memory_bytes`).
- **Verdict per row:** identical; **explained**: only the rewrite config differs and the new side carries a
  decision with a code of rule 12 (decision 51 turns rewrites off and changes nothing else; rule 15 changes nothing,
  so its decisions are only listed as added); anything else is **different**, fails, and goes to the conductor.
  Expected explained rows: none on the 16 GiB record unless 5-T03 finds a label that does not fit; the low-memory
  rows.
- **Dry run** (24 October 2026, CPU): baseline dae3848 against main b95c1f5 (5-T01 merged), 700 of 700 identical,
  8 s per side.
- **Output:** `results/5-T11-corpus/` (`old.json`, `new.json`, `report.json`) and a table in the report (counts
  per verdict, every non-identical row).

## Jobs

Local RTX 5060 Ti, run ID `5-T11-local-<date>`, `python3 <copy>/tools/ci/local_gpu.py run --wait-minutes 600
--purpose "5-T11 <job>" -- bash <copy>/tools/acceptance/phase5_local.sh <copy> <job>` (tool-side, modelled on
`phase4_local.sh`); outputs in `~/fastmace-acceptance/5-T11-local-<date>/`. CPU jobs run in WSL from the same copy.

| Job | Item | GPU | Timed | What runs | Estimate |
| --- | --- | --- | --- | --- | --- |
| `conformance` | 1 | no | no | `pytest tests/cpu -m rule` and `tests/cpu/test_rules_meta.py`; the meta-test's listing (rules → tests, precedence pairs) saved | 5 min |
| `cpu-suite` | 2 | no | no | the conductor's merge-trial CPU suite at the freeze, cited by commit; `pytest tests/cpu -m acceptance` here | 20 min |
| `corpus` | 3 | no | no | `tools/plan_corpus.py dump` with `fastmace_4` and with the freeze, then `compare --explain-rules 12` (the sizes were captured once, before the freeze) | 1 min |
| `preflight` | – | yes | no | the stale-listing selection | 1–3 h |
| `trajectories-cpu` | 4 | no | no | `tools/trajectory_versions.py --against 4`, CPU float64, tiny MH-1-class model, NaCl 2×2×2, 100 steps, every 10, Langevin NVT and BAOAB NPT, D3 off and on; `compare --bitwise` | 15 min |
| `trajectories-reproducible` | 4 | yes | no | the same, reproducible plan, plain cuEq float64, MH-1 NaCl, 500 steps, every 50, NVT and NPT, D3 off | 20 min |
| `restart-across` | 4 | no | no | `tools/restart_versions.py`: a Langevin restart written by the baseline (format 4, step 50) resumed at the freeze to step 100, CPU float64, tiny model; bitwise against the baseline's uninterrupted run, without `allow_plan_change` and without a new revision | 10 min |
| `regression` | 5 | yes | yes | `tools/ab_fastmace.py --release 4`: MH-1 hybrid Langevin NVT, D3 off, NaCl and NbCl5, 300 K, 12 blocks of 300 steps after 300; ABAB 4 first, ABAB 5 first, ABBA 4 first (4-T13's protocol) | 40 min |
| `adapt-storm` | 6 | yes | yes | 5-T07's forced re-capture storm, MH-1 hybrid NaCl, switched against unswitched, one block each | 30 min |
| `adapt-production` | 6 | yes | no | through `fastmace run` with the default `adapt`, resume included: the shipped `npt_anneal_NbCl5_short` and `swap_anneal_NaNbOCl4_short` (the MC path), and an NVT anneal on NaCl 3×3×3 defined here from `main.py`'s `--profile anneal` (0 → 1000 K in 20 ps, then 3 K/ps to 250 K; samples and frames every 10 fs; no shipped profile is added) | 2 h |
| `adapt-tests` | 6, 7 | yes | no | 5-T07's and 5-T08's GPU tests (switch at a canonical point, statistics against an unswitched compiled run, resume after a switch, broken capture) | 1 h |
| `calibration` | 9 | yes | yes | 5-T09's NaCl smoke with MH-1: the record, the choice, the setup cost | 20 min |
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
- **Item 6, the storm:** the effect expected is large (compiled against re-capturing every few steps): one block
  each side, one process per side; the switched run must be faster beyond the 3 % cross-process floor. This is
  also the owner's checkpoint 3 (whether decision 52's 20 % holds); 5-T07's own timed check is cited with it.
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
- The freeze side runs with the default `AdaptationConfig`; under a reproducible plan no switch happens (rule 5,
  decision 53), and every job checks `counters.tier_switches == []`.

**Item 5: step time, no regression.**
- Per structure, the median of the three processes' ratio 5/4 within ± the local floor, and no process slower than
  1 + 2 × floor.
- Plans: only the phase 5 fields may differ (`adaptation`, `calibration`, `predicted_memory_bytes`, the plan text,
  `schema_minor`), and no tier switch, fallback or calibration in either side's run; any other engine-option
  difference makes the process invalid.

**Item 6: runtime adaptation** (§9.6; decisions 52, 53).
- On the storm the switch happens at the first canonical point after the trigger (forced if the cadence puts it
  later; choice 2), graph → compiled with the same backend, rewrites and precision, as a new revision with step and
  reason; `counters.tier_switches` has one entry `{step, from, to, reason, revision}`; the report line and the
  `fastmace.planner` log line are present; the candidate was validated before activation (self-checks; the gate
  where the run asked for one). Energies and temperatures after the switch statistically equal to an unswitched
  compiled run (5-T07's measure); interrupted against uninterrupted across the switch, statistical. A resume after
  the switch runs the compiled revision and does not switch again. Timed: the switched run is faster (above).
- **The production profiles** (`npt_anneal_NbCl5_short`, `swap_anneal_NaNbOCl4_short`, the NaCl NVT anneal): no
  switch, `counters.tier_switches == []`, and none during an MC block in any case. The re-capture time fraction,
  rebuild rate, build time and overflows per segment are recorded, with no action.
- **Decision 53's limits:** checked by 5-T07's CPU tests in the suite (cited: one way, at most one switch, none
  under `reproducible=True`, with an explicit `mode=`, during an MC block, or in a child without the hand-off),
  plus on the GPU: no switch with an explicit `mode=` and none under `reproducible=True` on the storm.

**Item 7: mid-run fallbacks** (§9.4 step 5; decision 54).
- The broken capture (5-T08's GPU test, MH-1 hybrid NaCl) ends on a validated revision that started from the last
  committed canonical point; the chain records both revisions; output chunks after the point were deleted and
  written once (no duplicate or missing frame). A compile error of the compiled tier at a new edge count, injected
  during `run()`, ends the same way (5-T08's test; on the GPU if 5-T08 has one, else its CPU test cited).
- With `fallback="refuse"` the run raises with the record and the last restart point stays resumable.
- An out-of-memory error and a non-finite value are not fallback triggers: the run stops, the out-of-memory
  message carrying rule 12's estimate (decision 51).

**Item 8: rule 12** (decision 51).
- 5-T03's validation (`docs/validation/5-T03.md`) cited: the estimate never below the measured peak on any of the
  19 labels × NaCl, NbCl5 × NVT, NPT, and at most 1.5× above. Rerun (the same tool, the same cases) only if
  `planner/memory.py` or a rewrite's memory term changed between 5-T03's head and the freeze; the report gives the
  diff.
- The estimate's safety factor, if 5-T03 uses one, is a single global number, the same for every model and GPU,
  recorded with its fit (§9.3 rule 12 as amended, b8ef2ae); the run checks that `planner/memory.py` holds no
  per-model, per-GPU or per-shape factor or table.
- The action, from 5-T03's CPU tests (cited) and the corpus's low-memory rows: above 90 % of the free memory, the
  rewrites with an edge-sized term dropped largest first until it fits, each with rule 12's reason, then a warning
  if it still does not fit; never a change of rule 11's sizes, the tier, the precision or an explicit user setting
  (an explicit `"on"` is kept and warned). The 90 % margin is the owner's checkpoint 2, judged on 5-T03's numbers.
- In `compat` on each GPU: the estimate against the measured peak on MH-1 NaCl (5-T03's test).

**Item 9: calibration** (decision 55).
- On NaCl with MH-1 (`calibrate=True`, no backend set) the smoke measures hybrid and plain cuEq, each in a child,
  50 steps after warm-up, alternating; it keeps hybrid, or changes it only with a margin above 5 % in every pair,
  judged on the RTX 5060 Ti only; `Plan.calibration` records each candidate's measured step times; the setup cost
  is in the report.
- Without `calibrate`, and with a backend set, no calibration runs (5-T09's CPU tests, cited); nothing is written
  outside the run directory and nothing is read from earlier runs (5-T09's CPU test with a planted record, cited).

**Item 10: `str(plan)`.** 5-T05's golden texts green in the CPU suite (cited), each section present, no line over
120 characters, at most about 30 lines; the real plans of items 6–9 printed in the report, each with the section
its task filled (switch in the history, fallback revision, memory line, calibration record).

**Tests:** every acceptance and slow GPU test passes on the RTX 5060 Ti at the freeze or at a corrected copy
(corrections listed with reasons; the A100's share counts there); the compatibility check passes on the RTX 5060
Ti and the Colab L4 (and on the A100 if that leg runs).

## The compatibility tier's additions (test-side, `tests/_support/tiers.py`)

`COMPAT_TESTS` gains the GPU tests the tasks write (conductor, 23 October 2026: 5-T03 and 5-T07 write them,
5-T11 lists them when they merge and checks that `tests/cpu/test_tiers.py` collects them):
- rule 6 on that GPU (5-T11, written): `test_compat_gpu.py`'s graph-kernel test, already in the tier, also asserts
  that the planner's software check found cuEquivariance's fused kernels and the plan's `allow_naive_cueq`
  decision is `off` with code `fused-cueq` (the loading side, `fused_cueq["naive"]`, is `test_loading_gpu.py`'s);
- rule 12: the estimate against the measured peak on MH-1 NaCl (5-T03's `test_memory_gpu.py`, requested by the
  conductor; listed when it lands);
- the forced re-capture switch smoke, correctness only: 5-T07's
  `test_adapt_switch_gpu.py::test_a_forced_recapture_storm_switches_once_to_the_compiled_tier` (listed).

## Tool-side additions (before the freeze, CPU-tested on `fastmace/5-T11`)

- `tools/plan_corpus.py`, `plan_corpus_labels.py`, `plan_corpus_capture.py`, `plan_corpus_dims.json` and
  `tests/cpu/test_plan_corpus.py` (written: the set, planning and refusals, compare's verdicts on planted
  differences, the renamed export, the dims of all 19 checkpoints, the job script).
- `tools/acceptance/phase5_local.sh` (written): the CPU jobs, the corpus, the pytest jobs and the tiers; the jobs
  that need the `--release 4` side or 5-T07's and 5-T09's tools exit 3 until they are filled in at the freeze.
- After 4-T13's tools merge: `tools/ab_fastmace.py --release 4`, `tools/trajectory_versions.py --against 4`,
  `tools/restart_versions.py` with the baseline `fastmace_4` (format 4 on both sides).
- `tests/_support/tiers.py`: 5-T03's and 5-T07's compat tests, when they merge.

## Recorded, not judged

- Calibration's measured step times off the reference GPU.
- The re-capture time fractions and rebuild rates per segment of the production profiles.
- The corpus's rule 12 decisions on the 16 GiB record (which labels come close to the free memory).

## Not in this runbook

- New rules beyond §9.3's fifteen (plan, "Out of scope"); a gap the conformance suite found goes to the conductor.
- ResearchModule's `main.py` (no change in phase 5).
- Judging physics (decision 6).
- 5-T03's large-model peaks on the Colab A100 are 5-T03's and are cited.
