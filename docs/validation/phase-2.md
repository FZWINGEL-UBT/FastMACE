# Phase 2 acceptance report

**Status: interim, 20 October 2026, about 17:50 UTC. Accuracy acceptance deferred by the owner.** Plan task
2-T23, branch `fastmace/2-T23`.

On 20 October 2026 the owner decided: no more large accuracy runs on murphy for now. Phase 2's accuracy
acceptance moves to after the optimisation round (`plans/opt-1.md`), as one combined acceptance on the
optimised main. This draft records the state at that point:

- **done and passed:** item 1 (MACE 0.3.15, re-run on `85360cc`); the local acceptance tier (30 / 30);
  item 8's default plans and default trajectories on both machines; item 8's step times on the local card
  and on murphy for NbCl5; items 9b, 9d, 9e from their tasks;
- **done, not passed:** item 4's hybrid clause by 1b-T14's measure from the 0 K crystal (narrow, explained
  by the heating transient; owner's repeat pending); item 8's murphy NaCl, one of three processes 0.004
  points over the floor (three more processes pending); the local full tier's one failure is the suite
  definition's, not the library's;
- **stopped:** the murphy acceptance tier, stopped at 17:53 UTC by the conductor on the owner's order,
  after 14 of its 30 tests, all passed (see [Stopped](#stopped-and-deferred));
- **deferred to after the optimisation round:** the murphy acceptance tier, item 4's repeat
  (its protocol is below, written before any run), the final re-run on main after the outside-review
  fixes, and (held by the conductor) the three murphy NaCl processes, whose rule is below;
- **deferred by the owner, not a blocker:** item 7, GPU CI.

**Library.** Every 2-T23 run imports a `git archive` of a 2-T23 tools commit whose `fastmace/` tree is
main `85360cc`'s. The tools commits are `d885b12` (batch 1), `37c4014` (local batches 2 and 3, murphy
untimed suites), `4b35b60` (local batch 4, murphy timed suites, item 4's first comparison),
`a8eee21` (murphy NaCl processes and item 4's repeat, queued, not run). Only `tools/`, `tests/` and `docs/` differ between
them and `85360cc`.

**Evidence.** The merged phase 2 tasks' own records are reused where the plan allows (conductor's go
message). 2-T23 reran what needs final main: item 1's GPU check, items 4 (hybrid) and 8, the acceptance
and full GPU tiers on the local card, and the acceptance tier on murphy until it was stopped.

**Data.** Summaries, metadata, manifests, junit files, thermo rows and the batch scripts and logs are in
`results/2-T23-local-20261020/`, `results/2-T23-murphy-20261020/` and
`results/2-T23-murphy-20261020-runner2/` (`summary.md` and `summary.json` from
`tools/acceptance/summarize2.py`; item 4's comparison in `compare.json`). Large files (forces, trajectories,
caches) stay on the machines: `~/fastmace-acceptance/2-T23-local-20261020/` on the PC and
`/mnt/homes/zwingel/fastmace-acceptance/2-T23-murphy-20261020*/` on the NAS.

## Verdict

| # | Item (plans/phase-2.md) | Local RTX 5060 Ti | murphy RTX 4000 Ada | Result |
| --- | --- | --- | --- | --- |
| 1 | Unknown MACE version (0.3.15) | 2-T04d, re-run on `85360cc`: passed | not run (local only, conductor) | **pass** |
| 2 | Fallback, injected faults | acceptance tier: 6 fallback tests passed | 2-T15 (rewrite, pair, capture) | **pass**; the exhausted budget on CPU only |
| 3 | Gate statuses and resources | 2-T05; acceptance tier: both OOM cases passed | (the premise is the 16 GB card) | **pass**; CPU float64 never succeeded on a GPU-sized cell |
| 4 | Subprocess runner | acceptance tier: both runner tests passed; runner statistics as murphy | 2-T07; runner statistics | plain cuEq, chunks, the parent's plan: **pass**. Hybrid statistics: **narrow fail** by 1b-T14's measure from 0 K; the owner's equilibrated repeat is **deferred** |
| 5 | Instance isolation | acceptance tier: 3 paths passed | 2-T08, 2-T08b | **pass**; no lease needed |
| 6 | Reproducible mode | 2-T16; acceptance tier: 5 passed | 2-T16 | **pass** (qualified, with an undocumented cuEq variable) |
| 7 | GPU CI on two architectures | – | – | **deferred by the owner** (20 October 2026, SPEC draft 0.12 eleventh amendment): `gpu.yml` runs by hand only; not a phase 2 blocker. GPU coverage comes from the trial tiers and these acceptance runs |
| 8 | No regression against `fastmace-1b` | plans equal; trajectories pass; NaCl within floor; NbCl5 2 of 3 within floor, one process main 0.55 % *faster* (accepted) | plans equal; trajectories pass; NbCl5 3 of 3 within floor; NaCl 2 of 3, one process 1.00164 against 1.0016 | plans and trajectories **pass**; step times **pass** except murphy NaCl, **open** until the 3 held processes run |
| 9 | Carry-overs | | | `main.py` on `Simulation`: from 2-T21's report once merged (not merged at this draft); resume warning (2-T09, 2-T09b), doctor's stale locks (2-T11), `fm.calc` one compile (2-T13): **pass**; coraline (2-T22): **still waiting** for an idle GPU |
| – | Acceptance GPU tier | **30 / 30 passed** | **stopped** after 14 of 30, all passed, including the kernel-count comparison that failed under 2-T08b | murphy deferred |
| – | Full GPU tier | 244 passed, 2 skipped, 1 failed (suite definition, below) | merge trials (below) | |

## Runs

| Run ID | Machine | GPU | What | Tools commit | UTC, 20 October |
| --- | --- | --- | --- | --- | --- |
| `2-T23-local-20261020` | PC (WSL2) | RTX 5060 Ti 16 GB | item 1, trajectories (batch 1) | `d885b12` | 11:54–12:08 |
| same | PC | same | item 8 A/B (batch 2, timed) | `37c4014` | 12:08–12:29 |
| same | PC | same | acceptance tier (batch 3, timed) | `37c4014` | 12:30–15:29 |
| same | PC | same | runner statistics, full tier (batch 4) | `4b35b60` | 15:29–16:18 |
| `2-T23-murphy-20261020` | murphy | RTX 4000 Ada 20 GB | trajectories; runner statistics with backend "auto" (invalid, below) | `37c4014` | 15:04–15:28 |
| `2-T23-murphy-20261020-runner2` | murphy | same | item 4, first comparison | `4b35b60` | 15:29–15:40 |
| `2-T23-murphy-20261020` | murphy | same | item 8 A/B (timed), acceptance tier (timed) | `4b35b60` | 15:42– |
| `2-T23-murphy-20261020` (acceptance tier) | murphy | same | stopped by the owner's decision | `4b35b60` | 16:05–17:53 |
| – | murphy | – | NaCl processes, item 4's repeat: queued, not run | `a8eee21` | – |

- **Environments:** `mace_cueq_oeq` (torch 2.14.1+cu130, MACE 0.3.16, cuEquivariance 0.12.0, OEq 0.7.0)
  for every hybrid process; `mace_cueq_latest` for plain cuEq processes and the pytest processes of the
  tiers; `fastmace_mace_other` (MACE 0.3.15, local only) for item 1. The NAS environments on murphy,
  `~/environments/…` on the PC.
- **Release side of item 8:** tag `fastmace-1b` (`673a8a1`), exported with
  `python -m fastmace.bench.versions export --ref fastmace-1b --name fastmace_1b`; on murphy the copy is
  `/mnt/homes/zwingel/fastmace-2-T23/versions`.
- **Research-engine baseline** (the acceptance tier's comparisons): tag
  `research-engine-baseline-20261017` (`d2faa9fd`), export `fastmace-baseline/1a-T18`.
- **Run settings:** hybrid FP32, TF32 off, `TORCHINDUCTOR_COMPILE_THREADS=4`; local GPU runs under the
  reservation (`tools/ci/local_gpu.py`), murphy runs only with no process of anyone on its GPU.
- **Timing conditions.** Local: batches 2 and 3 with nothing else on the PC (conductor kept the other
  workers off). Murphy: the first window waited for user `song`'s CPU jobs (load average about 30) to
  end (conductor). A logger recorded murphy's load every 60 s: during item 8's A/B the 1-minute load
  stayed ≤ 1.65 and other users' %CPU ≤ 3.3; during the acceptance tier the load stayed about 1–1.6.
- **Data:** `results/2-T23-*/` (summaries, metadata, manifests, junit files). Large files stay on the
  machines.

## 1. Unknown MACE version

Evidence: 2-T04 to 2-T04d ([2-T04-unknown-mace.md](2-T04-unknown-mace.md)), and a re-run of 2-T04d's driver
(`results/2-T04-acc1-0315-local-20261020b/acc1.py`) on main `85360cc` (copy `d885b12`), local, MACE 0.3.15
in `fastmace_mace_other`, 11:54 UTC. Item 1 runs locally only (conductor's answer Q2).

- MH-1 `matpes_r2scan` on NaCl 3×3×3, NVT 300 K, 200 steps, `check="gate"`: **gate passed** (energy
  6.97e-4 meV/atom against a yardstick of 6.90e-4, force 2.62e-6 eV/Å), 53 s.
- Plan: revision 0 graph, plain cuEq, no compilation ("this MACE lacks the compile API"), float32 with the
  patched float64 energy. Its capture failed, and revision 1 is the eager tier: "fallback: mode eager: eager
  passes the reduced gate: the fault is in capture or compilation". One reduced-gate run for the
  diagnosis, one full-gate run for the activation.
- Records whose source does not hash on 0.3.15 are off with their reasons (`factor_species`,
  `fold_cutoff`, `polar_uvu`, `single_graph_strain`). Treatments: float64 energy patched (its source
  hashes), PolarMACE refused, MACE's own conv-fusion rule. The only patch in force is
  `capture_safe_mask_head`, whose source hashes.
- The promotion path for an unhashed float64-energy source and the PolarMACE refusal on a changed polar
  forward are shown with fake hashes in the CPU suite only (2-T04); MACE 0.3.15 does not trigger them.

## 2. Fallback

Evidence: 2-T15 on murphy ([2-T15-fallback.md](2-T15-fallback.md), `results/2-T15-murphy-20261020/`): ddmin
found the broken rewrite (8 reduced-gate runs) and the broken pair (19), and the broken capture went to
eager (1). On the local card, final main's acceptance tier passed all six `test_fallback_gpu.py` tests:
both ddmin cases, the broken capture, 2-T15b's two capture-error cases (`skip` and `gate`), and the test
that the default trajectory is unchanged by the forced first capture. The exhausted budget and the
terminal sequence are shown on CPU only (`tests/cpu/test_fallback_search.py`), as is the pristine-copy
rule (decision 24). Item 1 shows a real, not injected, capture failure ending in eager.

## 3. Gate statuses and resources

Evidence: 2-T05 attempt-5 on the local card ([2-T05-gate-resources.md](2-T05-gate-resources.md)); final
main's local acceptance tier passed both `test_gate_resources_gpu.py` cases (POLAR-1-L and OMOL-0 XL on
NbCl5). Both end `passed-reduced` after a GPU out-of-memory, with the replay check on the production
cell. The CPU float64 step was tried and either declined by its estimate or stopped by the memory cap; a
successful CPU float64 reference and `unavailable` are shown on CPU only. Murphy's 20 GB card fits these
references, so the item's premise ("the 16 GB card") is local.

## 4. Subprocess runner

- **Plain cuEq float64 after a compiled hybrid, chunks, the parent's plan:** 2-T07 on murphy (2.56e-13 Å
  against a fresh process); final main's local acceptance tier passed both `test_subprocess_runner_gpu.py`
  cases. `reproducible=True` after CUDA started runs in a child on both machines (2-T16).
- **Hybrid, statistically (1b-T14's measure), first comparison:** `tools/runner_statistics.py`, MH-1 NaCl
  3×3×3 hybrid, 2 fs, ramp 300 → 600 K over 20,000 steps in two `run()` calls, seeds 20261001 and 20261101,
  from the 0 K crystal. Murphy, run `2-T23-murphy-20261020-runner2` (tools `4b35b60`): the runner case
  really ran in children (3 per seed: setup, then one per `run()`), backend hybrid.

  | Seed | ΔT (K) | ΔE (meV/atom) | bins rms (meV/atom) |
  | --- | ---: | ---: | ---: |
  | 20261001 | 2.20 | **0.735** | 1.67 |
  | 20261101 | 0.76 | **0.924** | 2.10 |
  | limit (2 × between seeds) | 9.34 | 0.655 | 2.92 |

  **Narrow fail on the mean potential energy.** The local card gave the same numbers to three decimals
  (local batch 4), so the comparison is deterministic across the two cards.
- **Why** (supporting analysis, not the verdict; the owner chose the repeat over a cut chosen after the
  result):
  - the difference comes entirely from the first ~1,000 steps. Excluding them, runner and fresh differ by
    0.04 / 0.02 K and 0.0033 / 0.0029 meV/atom per seed, about 300 times under the limits;
  - the fresh in-process run is bitwise identical in two separate processes, and the child that resumes
    the step-0 restart leaves it at step 60, as float32 runs in different processes can;
  - in that window every 2 fs run here heats from 0 K violently (finding 1), so two seeds cannot judge it.
- **The repeat** (owner's decision): protocol below, written before any run; **deferred** to after the
  optimisation round.
- An earlier run (`2-T23-murphy-20261020`, `runner-statistics-*`) is invalid: the driver asked for backend
  `"auto"`, which kept the process's plain-cuEq family in-process, so no child ran. Fixed in `4b35b60`.

## 5. Instance isolation

Evidence: 2-T08 and 2-T08b on murphy ([2-T08-isolation.md](2-T08-isolation.md)): A → B → A passes on plain
cuEq, MH-1 hybrid and POLAR-1-S, and no lease was needed, so the lease clause was not exercised. Final
main's local acceptance tier passed the three `test_isolation_gpu.py` cases.

## 6. Reproducible mode

Evidence: 2-T16 on both machines ([2-T16-reproducible.md](2-T16-reproducible.md)): repeated and
interrupted runs bitwise equal for NaCl and NbCl5, float64 and float32. Final main's local acceptance tier
passed all five `test_reproducible_gpu.py` cases. Qualification relies on the undocumented
`CUEQUIVARIANCE_OPS_U1D_ALGOS=1,1` of cuequivariance_ops 0.12.0, recorded and reported as such.

## 7. GPU CI

Deferred by the owner on 20 October 2026 (SPEC draft 0.12, eleventh amendment): `gpu.yml` runs by hand
only, and the item does not block phase 2. GPU coverage comes from the trial tiers and these runs.

## 8. No regression against `fastmace-1b`

Tools: `tools/ab_fastmace.py --release 1b` (this checkout, side "2", against the renamed `fastmace_1b`,
both through their own minimal planner, in one process) and `tools/trajectory_versions.py` (one version
per process). MH-1 `matpes_r2scan`, NaCl and NbCl5 3×3×3, hybrid FP32 NVT, 300 warm-up steps, 12
alternating blocks of 300, three processes per machine (ABAB 1b first, ABAB 2 first, ABBA), plus one
control per version.

**Default plans.** In every process the engine options of the two versions are equal. The plan
descriptions differ by one line: phase 2 adds the decision `allow_reference = off (not requested)`.

**Default trajectories.**

| Machine | Plain cuEq float64, 500 steps (1b-T14) | Repeat of main (floor) | Hybrid, 20,000 steps × 2 seeds |
| --- | ---: | ---: | --- |
| RTX 5060 Ti | 4.4e-13 Å | 3.3e-13 Å | ΔT ≤ 9.4e-6 K, ΔE ≤ 1.0e-7 meV/atom (limits 29 K, 3.7 meV/atom) |
| murphy | 3.3e-13 Å | 3.9e-14 Å | ΔT ≤ 1.4e-5 K, ΔE ≤ 2.0e-6 meV/atom |

Limit 1e-12 Å for plain cuEq; plans equal in every run. These runs use the planner's engine at 0.5 fs.
Starting forces in the A/B processes: 1b against main ≤ 1.9e-6 eV/Å, the same as each side against its
own control.

**Step times, main/1b** (12 blocks; each machine's own floor: RTX 5060 Ti 1.1 % NaCl, 0.4 % NbCl5; murphy
0.16 %):

| Machine | Process | NaCl | NbCl5 |
| --- | --- | ---: | ---: |
| RTX 5060 Ti | ABAB, 1b first | 1.0043 | 0.9986 |
| RTX 5060 Ti | ABAB, 2 first | 0.9893 | 1.0039 |
| RTX 5060 Ti | ABBA | 1.0011 | **0.9945** (main faster) |
| murphy | ABAB, 1b first | **1.00164** | 1.00059 |
| murphy | ABAB, 2 first | 1.00096 | 0.99996 |
| murphy | ABBA | 1.00042 | 1.00015 |

Every process is valid: nothing compiled and no process-wide patch changed while timed. Controls (one
version per process, 4 blocks): local NaCl 3.486 → 3.471 ms, NbCl5 27.137 → 26.947 ms; murphy NaCl
3.499 → 3.485 ms, NbCl5 30.863 → 30.751 ms.

- Local: the NbCl5 ABBA process is outside the 0.4 % floor with main 0.55 % faster, which is not a
  regression (conductor).
- Murphy NaCl: 2 of 3 within the floor; the ABAB process with 1b built first is at 1.00164, 0.004 points
  over, with every pair above 1. 1b showed the same process dependence on NaCl
  ([phase-1b.md](phase-1b.md), finding 3). Three more processes under a rule fixed before they run
  (below) are held by the conductor.

## 9. Carry-overs

- `main.py --engine fastmace` on `Simulation` (2-T21): its A/B passed (conductor); not merged at this
  draft. Counted from its report once merged.
- `Simulation.resume` warns on a version-only change: 2-T09, 2-T09b, CPU suite.
- coraline (2-T22): still waiting for an idle GPU, as acceptance 9 allows.
- `fastmace doctor` reports stale build locks: 2-T11, CPU suite.
- `fm.calc` compiles once when the edge count changes: 2-T13 ([2-T13-calc-compile.md](2-T13-calc-compile.md));
  `test_ase_calculator_gpu.py` passed in the local full tier.

## GPU tiers

- **Acceptance tier, local:** 30 / 30 passed in 2 h 59 min (12:30–15:29 UTC, quiet PC), including the
  planner's two step-time tests (2-T14d) and all three research-engine comparisons.
- **Acceptance tier, murphy:** stopped (next section) after 14 of 30, all passed. That includes
  `test_mh1_plain_cueq_float64_against_the_research_engine`, which failed under 2-T08b on murphy (stage-B
  kernel count 249 against 237, [2-T08-isolation.md](2-T08-isolation.md)).
- **Full tier, local:** 244 passed, 2 skipped (the OEq streaming kernels need OEq in the pytest
  environment), 1 failed: `test_reference_gpu.py::test_polar1s_reference_equals_1a_t19`. The suite set
  `FASTMACE_1A_T19_POLAR1S` to a directory that exists only on the NAS, so the test read a missing file
  instead of skipping. Suite fixed in `30ad506`; no library change. Murphy's full tier is the merge
  trials' (244 passed at the T16 + T15b + T17 + T19b trial).

## Stopped and deferred

**Stopped (owner's decision, 20 October 2026):** the murphy acceptance tier
(`2-T23-murphy-20261020/acceptance-tier`, tools `4b35b60`), at 17:53 UTC, in its gate-resources file.
By then 14 of 30 tests had passed, none failed: ASE equivalence 2/2, canonical restart 1/1,
`test_engine_gpu.py` 3/3, `test_fallback_gpu.py` 6/6, `test_forces_gpu.py` 2/2. The permission check
refused 2-T23's own stop command on murphy. On the owner's direct instruction the conductor killed both
2-T23 process groups (the tier, and the waiter that would have started the NaCl processes and item 4's
repeat) and cleared the queue; murphy's GPU was then empty. The partial log and results stay on murphy.

**Deferred to the combined acceptance after the optimisation round (`plans/opt-1.md`):**

- the murphy acceptance tier;
- item 4's equilibrated repeat (protocol below);
- the final re-run on the main that includes the outside-review fixes merged after `85360cc`:
  `e552181` (2-T14e, a test duplicate), `a8cd333` (2-T07b, durable subprocess checkpoint), `05ecbe6`
  (2-T04e, torch 2.13 patch applicability), `edba9f3` (2-T14e follow-up: `test_ase_equivalence_gpu.py`
  at 4,000 steps with limits of 1e-6 K and 1e-7 meV/atom, SPEC tenth amendment), and 2-T15c when it
  merges. Each has its own trial evidence (conductor). The re-run covers the ASE-equivalence file (new
  limits, untested on sm_89), the subprocess and restart tests, the fallback tests and the
  versioned-registry CPU tests;
- held by the conductor: the three murphy NaCl processes of item 8 (short, timing, not accuracy).

## Findings

1. **A violent heating transient at 2 fs from the 0 K crystal** (MH-1, NaCl 3×3×3, hybrid and plain cuEq
   float32 alike). In the first ~300 steps the potential energy rises 35–100 meV/atom above
   equipartition at 120–240 K, and the temperature of the fresh run spikes to 466 K at step 280. The same
   crystal at 0.5 fs, the production step (`trajectory2`), heats smoothly. Carried into the phase 3 notes
   by the conductor; no task now.
2. **Hybrid float32 repeats across processes and across the two cards here:** the fresh hybrid run gave
   identical rows in two murphy processes, and the local card reproduced murphy's comparison to three
   decimals.
3. The stage-B kernel-count failure of 2-T08b on murphy did not recur on final main (above); its cause
   is still unrecorded.


## Item 8 on murphy, NaCl: the rule for three more processes (written before they ran)

The first three murphy A/B processes (run `2-T23-murphy-20261020`, 15:42–16:05 UTC) gave main/1b on NaCl
1.00096, 1.00164 and 1.00042 against the 0.16 % floor. The 1.00164 process (ABAB, 1b built first) is
0.004 percentage points over the floor, with every pair above 1. This is the same process dependence
that 1b's open NaCl item showed ([phase-1b.md](phase-1b.md), finding 3). On 20 October 2026 the conductor
fixed this rule before more data was taken:

- three more NaCl-only A/B processes on murphy, timed and quiet, right after the acceptance tier and
  before item 4's repeat: ABAB with 1b built first, ABAB with main built first, ABBA with 1b built first
  (suite `regression2-nacl`);
- **murphy NaCl passes** if the median main/1b ratio over all six processes is ≤ 1.0016 (the floor), no
  single process exceeds 1.0032 (twice the floor), and the one-version controls show main not slower;
- otherwise it is reported as failing for the owner, with every number.

## Item 4, hybrid: the repeat's protocol (written before it ran)

**Why there is a repeat.** The first comparison (run `2-T23-murphy-20261020-runner2`, tools `4b35b60`)
followed 1b-T14's measure: MH-1 NaCl 3×3×3, hybrid float32, Langevin 2 fs, target ramp 300 → 600 K over
20,000 steps in two `run()` calls, two seeds, from the 0 K crystal. It failed narrowly on the mean
potential energy (0.73 and 0.92 meV/atom against a limit of 0.65), and the whole difference came from
the heating transient of the first ~1,000 steps (section [4](#4-subprocess-runner)). On 20 October
2026, after that result, **the owner decided: "Repeat, equilibrated"**. Item 4's hybrid clause is
judged on the repeat; the first result stays in this report as it was.

**Protocol of the repeat** (`tools/runner_statistics.py`, suite `runner-repeat`, murphy, after the timed
window):

1. **Start state, made once** (`runner_statistics.py equilibrate`): MH-1 `matpes_r2scan`, NaCl 3×3×3
   (216 atoms) from the 0 K crystal, one in-process hybrid float32 `Simulation`, Langevin at a constant
   300 K target, friction 0.02 × `ase.units.fs` per fs (as the first comparison and 2-T07):
   - 4,000 steps at 0.5 fs (2 ps; the production step, which has no transient);
   - then 5,000 steps at 2 fs (10 ps).
   The positions, momenta and cell of the last step are written to `start.extxyz`. The mean temperature
   and potential energy over the second half of each stage are recorded, to show the state is
   equilibrated at 300 K.
2. **Runs**: both cases start from `start.extxyz`, with its momenta, and are otherwise the first
   comparison's protocol: 2 fs, the target ramp 300 → 600 K over 20,000 steps in two `run()` calls of
   10,000 (one hand-off), thermo rows every 20 steps.
   - `fresh`: the hybrid plan in this process;
   - `runner`: a plain-cuEq `Simulation` runs first (its compile fixes the process's family), then the
     hybrid plan (`backend="hybrid"`) runs in subprocess-runner children.
   - **Four seeds**: 20261001, 20261101, 20261201, 20261301; each case runs each seed.
3. **Limit**, 1b-T14's rule extended to four seeds. Per seed, for the mean temperature, the mean
   potential energy and the rms over 50 K target bins of the bin-mean potential energy:
   |runner − fresh| ≤ max(2 × D, floor), where D is the median of that quantity's difference over the six
   pairs of fresh seeds (with two seeds, D is their one difference: 1b-T14's rule), and the floors are
   1b-T14's: 1 K, 0.05 meV/atom, 0.05 meV/atom. Every row from step 20 to 20,000 counts; nothing is
   excluded.
4. **Pass**: all four seeds within the limit for all three quantities, every runner run reports
   `plan.runner == "subprocess"` and every fresh run `"in-process"`, and every run's backend is hybrid.

## Reproducing

From a FastMACE checkout (`$FM`) with the tag `fastmace-1b`:

```bash
python -m fastmace.bench.versions export --repository $FM --ref fastmace-1b --name fastmace_1b --destination DIR
python3 $FM/tools/ci/local_gpu.py run --purpose "phase 2 acceptance" -- \
  python3 $FM/tools/acceptance/run.py regression2 acceptance-tier trajectory2 runner-statistics full-tier \
  --machine local --run-id <id> --fastmace <pinned copy> --versions DIR
python3 $FM/tools/acceptance/summarize2.py ~/fastmace-acceptance/<id> --machine local
```

On murphy the same `run.py` line runs detached (`tools/acceptance/launch.sh`), with nothing on the GPU.
