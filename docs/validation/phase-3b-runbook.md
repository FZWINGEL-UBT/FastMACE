# Runbook: phase 3b acceptance

Plan task 3b-T10, written 22 October 2026 as a prestart, before any phase 3b result exists and before the
other 3b tasks are merged. Results go to `phase-3b.md` and `results/3b-T10-*/`. 3b-T10 changes no library
code; its tool changes are labelled tool-side. Nothing here runs before the conductor's go.

Sources: SPEC §16 row 3b, §11.2 ("Correctness criterion", "Cost target"), §18 decisions 6, 39 and 40–43;
`plans/phase-3b.md` (choices 1–8); `phase-3.md` and its runbook, with their open findings 1, 2 and 7.

## Freeze, baseline and copies

- **Freeze:** the main commit with every 3b task merged except 3b-T09 (ResearchModule), named by the conductor.
  The report records it and the commit whose `fastmace/` it equals.
- **Baseline for item 5:** tag `fastmace-3` (phase 3's release, after its slow-tier appendix). If the tag is
  not set at the freeze, the conductor names the commit, and the report says so. Exported as the renamed
  package with
  `python -m fastmace.bench.versions export --ref fastmace-3 --name fastmace_3 --destination ~/fastmace-3b-T10/versions`.
- **Pinned copies:** `git archive` of the 3b-T10 head with `.source_commit`, `fastmace/` equal to the
  freeze's: local `~/fastmace-3b-T10/src-<sha>`; for Modal, the same commit uploaded by
  `tools/modal/fastmace_modal.py`; for a workstation (if back), `/mnt/homes/zwingel/fastmace-3b-T10-<sha>`.
  A later test correction makes a new copy; the report lists each copy and what ran from it.
- **Environments:** local `mace_cueq_oeq` (MH-1 hybrid, timed items, NPT) and `mace_cueq_latest` (plain cuEq,
  pytest, references) with torch-dftd 0.5.1, `dftd3` (simple-dftd3) 1.6.0 and, optionally, tad-dftd3 0.7.0
  (choice 1). The report records each package version and how `dftd3` was installed (wheel or cmake build).
- **Every result names** the machine, GPU, driver, environment, backend, precision, model label
  (MH-1 `matpes_r2scan`), the D3 parameters and constants (choice 2), and the commits of the freeze and of
  `fastmace-3`.

## Scope and hardware (decision 39, choice 8)

- **Items run in full on the RTX 5060 Ti** (local, through `tools/ci/local_gpu.py run` from this worktree,
  as background tasks of the 3b-T10 session).
- **One Ada spot check:** the GPU acceptance tier at the freeze on the Modal L4 (Ada, sm_89), under the budget
  guard with `--limit-usd 3` and a function timeout whose worst case stays under $3 (at $0.80/h for the L4
  plus CPU and memory, about 150 min). If the workstations are back first, leeloo (RTX 4000 Ada) replaces the
  L4, after the GPU-idle and no-other-user checks, files only under `/mnt/homes/zwingel`.
  - Order on the L4: the new 3b acceptance tests first (D3 references, NPT + D3, D3 list), then the rest of
    the tier. Tests that don't fit are listed; the RTX 5060 Ti covers them.
  - Timed items don't run on the L4 (shared cloud GPU, noise unknown).
- **The local leg** runs items 1–5 in full, the preflight below, the GPU slow tier and the acceptance tier.
  If the Ada check runs the whole acceptance tier and passes, the local acceptance tier shrinks to the tests
  corrected at acceptance (decision 39's last clause).
- **Item 4's length:** the short protocol at the freeze, citing 3b-T07's full-length run, if the files on its
  path (`fastmace/longrange`, `neighbors`, `integrators`, `protocols`, `engine`, `model`, `simulation.py`,
  `planner`) are unchanged between that run's commit and the freeze. Otherwise the full protocol reruns at
  the freeze (about 3 h; overnight).

## Before any run: stale expectations (phase 3 lesson)

Trial merges are CPU-only, so GPU tests go stale (phase-3.md open finding 1: seven did). At the freeze, before
the timed jobs:

1. **List** every GPU test marked `slow` or `acceptance`, and for each whether its file or the library code it
   exercises changed between `fastmace-3` and the freeze (`git diff --stat fastmace-3 <freeze> -- fastmace tests`).
2. **Read** each changed-path test's expectations against the 3b changes, at least:
   - the NPT tests (BAOAB's stress path and the pressure setter, 3b-T05): `test_ase_equivalence_gpu`,
     `test_engine_gpu` (kernel sequences: the D3-off force graph must not change; open finding 8),
     `test_events_gpu`, `test_fused_integrator_gpu`, `test_integrators_gpu`, `test_overflow_commit_gpu`,
     `test_registry_gpu`, `test_selfcheck_gpu`;
   - tests that pin plan fields, provenance or run-report shapes (new D3 fields), the planner's rules 10–12
     (`test_planner_gpu`, `test_gate_resources_gpu`), restart (`test_canonical_restart_gpu`,
     `test_reproducible_gpu`) and the reference tier (`test_reference_tier_gpu`);
   - any test that still expects the `d3-phase-3b` refusal (none on the GPU at 8680d5b; the CPU ones run at
     merges).
3. **Collect** which of them each 3b task ran on the GPU at its own head (from the task reports). A test whose
   covered code changed after the head it last ran at goes into the `preflight` job.
4. **Run `preflight`** (untimed): those tests only. A failure is triaged before the timed jobs: a stale
   expectation is corrected test-side, in its own commit, with the reason and the change that made it stale
   (as phase-3.md's "Test corrections at acceptance"); a library defect stops the run and goes to the
   conductor. A failure is reproduced on the production path (`Simulation`, `fastmace run`) before it is called
   a library defect.

## Stoppable pytest runs (phase 3 lesson)

Phase 3's partial local tier lost its junit when stopping the chain's shell ended pytest before
`pytest_sessionfinish` (open finding 7). Tool-side additions, written and CPU-tested before the go:

- **A per-test result log:** a small plugin (`tools/acceptance/resultlog.py`, loaded with `-p`) appends one
  JSON line per finished test (node id, outcome, duration, time) and flushes it, so even a killed run keeps
  every finished test.
- **junit on SIGINT:** every pytest job runs pytest as a child with its PID recorded in the job log; the job
  script traps INT, TERM and HUP, forwards SIGINT to pytest, and waits up to 120 s for it to write
  `junit.xml`. A stop is `kill -INT <pytest pid>`, never a kill of the shell or the reservation first.
  `timeout` wrappers use `--signal=INT --kill-after=120`.
- **A CPU test** of both: a dummy suite stopped by SIGINT mid-run leaves a junit file and the JSON lines of
  the finished tests.

## Jobs

Local RTX 5060 Ti, run ID `3b-T10-local-<date>`, through `tools/ci/local_gpu.py run` and
`tools/acceptance/phase3b_local.sh <copy> <job>` (tool-side, modelled on `phase3_local.sh`); outputs in
`~/fastmace-acceptance/3b-T10-local-<date>/`. The timed jobs run first, alone, on a quiet PC (no other GPU
or heavy CPU job; checked and recorded at the start of each).

| Job | Item | Timed | What runs |
| --- | --- | --- | --- |
| `preflight` | – | no | the tests selected above |
| `regression` | 5 | yes | `tools/ab_fastmace.py --release 3` (tool-side: release `3`, tag `fastmace-3`, package `fastmace_3`): MH-1 hybrid NVT, D3 off, NaCl and NbCl5, 300 K, 12 blocks of 300 steps after 300; processes ABAB 3 first, ABAB 3b first, ABBA 3 first |
| `regression-npt` | 5 | yes | NPT step time, D3 off, NbCl5, one version per process (two-sided NPT is refused in one process), 4 alternating processes 3, 3b, 3b, 3 |
| `d3-cost` | 3 | yes | 3b-T06's tool and design at the freeze: NbCl5 3×3×3, MH-1 hybrid float32, the production ramp at 1 fs, D3 on against off in one process, NVT and NPT at 0 GPa, builds included |
| `d3-references` | 1 | no | 3b-T03's and 3b-T04's reference tests on the GPU (float64 and float32), plus the CPU float64 comparison, on the rattled and strained NaCl, NbCl5 and NaNbCl6 fixtures |
| `npt-compare` | 2 | no | 3b-T05's comparison tool, FastMACE side at the freeze (choice 7) |
| `protocol` | 4 | no | the D1 protocol through `fastmace run`: the short variant at the freeze (start, then resume to the quench); the full one if required above |
| `trajectories-cpu` | 5 | no | `tools/trajectory_versions.py --against 3` (tool-side), CPU float64, the tests' tiny MH-1-class model, NaCl 2×2×2, 100 steps, positions every 10, NVT and NPT (tool-side `--ensemble npt`), D3 off; `compare --bitwise` |
| `trajectories-reproducible` | 5 | no | the same against `3`, reproducible plan, plain cuEq float64 on the GPU, MH-1 NaCl, 500 steps, every 50, NVT and NPT, D3 off; `compare --bitwise` |
| `acceptance-tier` | 1, 2, all | (step-time assertions) | `pytest tests/gpu -m acceptance` |
| `slow-tier` | – | no | `pytest tests/gpu -m "slow and not acceptance"` |

Ada: run ID `3b-T10-ada-<date>`, `pytest tests/gpu -m acceptance` on the Modal L4 (or leeloo), in the order
above, with the result log and junit.

## Timing design (decision 39: repeats sized to the effect)

- **Item 5, NVT** (effect sought: under 1 %): the full same-process alternating design, three processes, the
  local floors measured in phase 3 (NaCl 1.1 %, NbCl5 0.4 %).
- **Item 5, NPT:** a check, not a measurement. One version per process, so only the cross-process floor of
  about 3 % applies.
- **Item 3** (target 10 %): one process with alternating on/off blocks at each ramp station first. If the
  amortised ratio lies within 3 points of 1.10 (1.07–1.13), two more processes (on first, and ABBA) and their
  median decide; outside that band one process suffices.
- **Cross-process controls** (phase 3 lesson, open finding 2): bounded at the documented 3 % or left out.
  The same-process A/Bs are the measurement; no control clause tighter than 3 %.

## Pass rules (written before the runs)

**Item 1: D3 against torch-dftd and simple-dftd3** (§11.2, plan 3b-T03).
- Fixtures: rattled and strained NaCl, NbCl5 and NaNbCl6, every one with cells narrower than twice the cutoff;
  all three codes called with the explicit numbers of choice 2 (BJ, `r2scan`, 40 bohr pair and CN cutoffs,
  ATM off, no smoothing, torch-dftd's constants where the code takes them).
- **float64 (GPU and CPU):** for energy, forces and stress, FastMACE's deviation from each reference is at
  most twice the two references' deviation from each other, plus a rounding floor of 1e-10 eV/atom (energy),
  1e-8 eV/Å (forces, max-abs) and 1e-9 eV/Å³ (stress, max-abs). Where simple-dftd3 uses its own unit
  constants, the comparison converts them, and the report states the constant-only difference.
- **The CN chain term:** a negative control (pass 3 left out) fails the force comparison.
- **float32:** within the gate (§12.3) against the float64 reference tier.
- **Capture:** a captured replay equals the eager evaluation bitwise.
- **Tests:** every D3 test of the acceptance and slow tiers passes on the RTX 5060 Ti, and the ones the Ada
  check reached pass on sm_89.

**Item 2: GPU NPT + D3 against ASE** (§11.2, choice 7).
- NbCl5 1×1×3 (216 atoms), MH-1 `matpes_r2scan` + D3-BJ, 0 GPa hydrostatic, 300 K and 600 K, 1 fs, the
  library's barostat defaults on both engines, 3 seeds per engine; duration and discarded equilibration as
  3b-T05 fixes them before its runs, the same on both sides.
- ASE side: `LangevinBAOAB` + MACE + torch-dftd (float64). Its runs come from 3b-T05 and are cited, not rerun,
  if the ASE script, its environment and its settings are unchanged; the FastMACE side reruns at the freeze.
- **Per temperature and observable** (mean V, V fluctuation as the standard deviation of V, mean P, mean PE
  per atom): |FastMACE − ASE| ≤ 3 √(SE_F² + SE_A²). SE of each engine: the standard error over its seeds of
  the per-seed block-averaged value.
- **One pre-declared extension:** if a comparison misses, 3 more seeds per engine run for that temperature,
  and the 6-seed comparison decides. Still missing: the item fails and goes to the conductor.
- Not judged: whether NbCl5 stays solid at 600 K (decision 6).

**Item 3: amortised D3 cost** (§11.2, decision 42).
- Cost = T_on / T_off − 1 over the production ramp (0 → 1000 K in 20 ps, 3 K/ps to 250 K, 1 fs), NbCl5 3×3×3,
  MH-1 hybrid float32, with every D3 build and its time included, weighted by the time the ramp spends at
  each station. NVT and NPT (0 GPa) reported separately; the D3 skin as rule 11 chooses it.
- **≤ 10 %: met. Above: a miss, recorded with the measured number, the rebuild count and the build share,
  and the phase continues** (decision 42). Neither is a library failure.
- Recorded with it: the D3 rebuild count and mean interval, the build time, the pair count against rule 12's
  estimate, and the step time without D3 against §3.3's 29 ms.

**Item 4: the NbCl5 NPT melt–quench protocol** (decision 40).
- NbCl5 3×3×3, MH-1 `matpes_r2scan` + D3-BJ (`r2scan`), hydrostatic NPT at 0 GPa, the anneal profile at
  1 fs, barostat defaults (P_tau = 20 T_tau, mass factor 1), through `fastmace run`.
- **Pass:** every stage completes and is recorded: the fixed-cell LBFGS relaxation; the NPT ramp with D3;
  a restart and resume across a canonical restart point (D3 list rebuilt there); the quench frames relaxed at
  their NPT cells with the cell fixed; outputs with volume, pressure, temperature and energy; provenance with
  the D3 parameters, the D3 skin, its rebuild counters and the barostat settings.
- **Fail:** an uncaught exception, a NaN, a silent stop, a missing stage or record. A documented refusal
  caused by the physics (for example rule 12's memory estimate after the cell grew) goes to the conductor
  before it is reported either way.
- **Recorded, not judged:** volume and density along the ramp, whether the crystal melts, swells or
  evaporates, the quenched energies.

**Item 5: no regression against `fastmace-3`, D3 off.**
- **NVT step times:** per structure, the median of the three processes' ratio 3b/3 within ± the local floor
  (NaCl 1.1 %, NbCl5 0.4 %), and no process slower than 1 + 2 × floor. Plans: only the new 3b fields may
  differ, each resolving to `fastmace-3`'s behaviour (no dispersion, no D3 list); any other engine-option
  difference makes the process invalid.
- **NPT step time:** the 3b processes' median within 3 % of the 3 processes' median.
- **Trajectories, bitwise:** CPU float64 and the reproducible GPU plan, NVT and NPT: every recorded row's
  positions, energy, temperature (and, under NPT, cell and pressure) equal (`compare --bitwise`).

**Tests:** every acceptance and slow GPU test passes on the RTX 5060 Ti at the freeze or at a corrected copy
(corrections listed with reasons); the tests the Ada check reached pass on sm_89.

## Not in this runbook

- 3b-T08's PolarMACE stress investigation (decision 41) is reported by its own task and cited.
- 3b-T09 (`main.py --dispersion`, decision 43) is outside the freeze; the report cites its merge.
- The barostat qualification and equilibrium volumes (plan, "Out of scope").
