# Runbook: phase 4 acceptance

Plan task 4-T13, written 23 October 2026 as a prestart, before any phase 4 result exists and before any phase 4
task is merged. Results go to `phase-4.md` and `results/4-T13-*/`. 4-T13 changes no library code; its tool
changes are labelled tool-side. Nothing here runs before the freeze and the conductor's go.

**Accepted** by the conductor on 23 October 2026 (078462b), with the additions beyond the plan: the float64 optimiser
tolerances, FIRE variable cell on NaCl and NbCl5 only, the widened item 2 rerun trigger, the `protocol` and
`restart-across` jobs, the shim A/B screen, ASE 3.27 in both environments, and the sampling-check escalation.

Sources: SPEC §16 row 4, §7.7 (adoption layer), §10.2–§10.3 and §10.6, §12.2 ("ASE equivalence", "Restart"),
§18 decisions 6, 37 and 39; `plans/phase-4.md` (D1–D6, choices 1–10, task 4-T13); `phase-3.md` with its open
findings 1, 2, 3 and 7; `phase-3b-runbook.md` and 3b-T10's tool-side additions (`fastmace/3b-T10`, 3ea191a).

**Decisions this runbook assumes.** D1–D6 are open. The pass rules below follow each recommendation (a). If the
owner picks another option, the affected rule is amended here before the freeze, and the change is listed at the
top of the report. The affected rules: D1 the `FixAtoms` rows of items 1 and 4; D2 the `dof` setting of item 1
(the equivalence tests use `dof="ase"` either way); D3 item 2; D4 the `protocol` job; D5 the optimiser classes of
item 3; D6 the gate of the PolarMACE tests.

## Freeze, baseline and copies

- **Freeze:** the main commit with every phase 4 task merged (4-T01 to T12), named by the conductor. The report
  records it and the commit whose `fastmace/` it equals.
- **Baseline:** tag `fastmace-3b` (phase 3b's release). If it is not set at the freeze, the conductor names the
  commit and the report says so. Exported as the renamed package with
  `python -m fastmace.bench.versions export --ref fastmace-3b --name fastmace_3b --destination ~/fastmace-4-T13/versions`.
- **Pinned copies:** `git archive` of the 4-T13 head with `.source_commit`, `fastmace/` equal to the freeze's:
  local `~/fastmace-4-T13/src-<sha>`; for Modal, the same commit uploaded by `tools/modal/fastmace_modal.py`;
  for a workstation (if back), `/mnt/homes/zwingel/fastmace-4-T13-<sha>`. A later test correction makes a new
  copy; the report lists each copy and what ran from it.
- **Environments:** local `mace_cueq_oeq` (MH-1 hybrid, timed items) and `mace_cueq_latest` (plain cuEq, pytest,
  float64 equivalence, the ASE sides). Both must have **ASE 3.27**, the version the integrators and optimisers
  port (choice 1). Another ASE version stops the run before any job, and goes to the conductor. The report
  records ASE, torch, cuEquivariance, openequivariance and MACE versions per environment.
- **Every result names** the machine, GPU, driver, environment, backend, precision, model label (MH-1
  `matpes_r2scan` unless stated), the integrator or optimiser with every argument (choice 1's records), the
  `dof` convention, and the commits of the freeze and of `fastmace-3b`.

## Scope and hardware (decision 39, choice 9)

- **Items run in full on the RTX 5060 Ti:** local, through `tools/ci/local_gpu.py run` from this worktree, as
  background tasks of the 4-T13 session.
- **One Ada spot check:** the GPU acceptance tier at the freeze on the Modal L4 (Ada, sm_89), under the budget
  guard, with `--limit-usd 3` and a function timeout whose worst case stays under $3 (about 150 min at the L4's
  rate plus CPU and memory). `modal billing summary` is read first and recorded. If the workstations are back
  first, leeloo (RTX 4000 Ada) replaces the L4, after the GPU-idle and no-other-user checks, with files only under
  `/mnt/homes/zwingel`.
  - Order on the L4: the new phase 4 acceptance tests first (the equivalence tests of the new integrators and
    of `FixAtoms`, the optimiser tests, restart and reproducibility of the new schemes), then the rest of the
    tier. Tests that don't fit are listed; the RTX 5060 Ti covers them.
  - Timed items don't run on the L4 (shared cloud GPU, noise unknown).
- **The local leg** runs the preflight, items 1–4, the no-regression check, the recorded measurements, the GPU
  slow tier and the acceptance tier. If the Ada check ran the whole acceptance tier at the freeze and passed, the
  local acceptance tier shrinks to the tests corrected at acceptance (decision 39's last clause).
- **Long untimed ASE sides** (items 1 and 2's optimiser references) may run on Modal if the local GPU is needed
  for timing, within the same $3 (plan, "Rules carried over"). Timed items stay on the RTX 5060 Ti.

## Before any run: stale expectations (phase 3 lesson)

Trial merges are CPU-only, so GPU tests go stale (phase-3.md open finding 1: seven did). Phase 4 changes the
integrator contract, the restart format, the support table and the fused Langevin stage, so the risk is higher
than in 3b. At the freeze, before the timed jobs:

1. **List** every GPU test marked `slow` or `acceptance`, and for each whether its file or the library code it
   exercises changed between `fastmace-3b` and the freeze (`git diff --stat fastmace-3b <freeze> -- fastmace tests`).
2. **Read** each changed-path test's expectations against the phase 4 changes, at least:
   - the restart format: tests that pin the format version or the key set of a restart file (v3 → v4, 4-T01):
     `test_canonical_restart_gpu`, `test_reproducible_gpu`, `test_io_gpu`;
   - the plan and provenance shapes: new `IntegratorConfig` fields (`ensemble`, `scheme`), the support table
     (rule 1), the optimiser rules 10 and 11, new report lines: `test_planner_gpu`, `test_simulation_gpu`,
     `test_selfcheck_gpu`, `test_gate_resources_gpu`;
   - the fused Langevin stage with the `FixAtoms` mask (4-T04): `test_fused_integrator_gpu`, `test_engine_gpu`
     (kernel sequences: AC and ACP are already left out under `fused_integrator`; the B graph must not change;
     phase-3.md open finding 8), `test_integrators_gpu`, `test_registry_gpu` (new runtime and derived state);
   - refusals that phase 4 lifts: `FixAtoms`, `nve-phase-4`, the `LATER` list in `ase/md.py`, variable-cell
     relaxation; any GPU test that still expects one of them;
   - the protocols' quench (D4: GPU LBFGS by default): `test_simulation_gpu` and any protocol GPU test;
   - the PolarMACE gate and the frozen k-set (4-T11, T12): `test_polar_gpu`, `test_reference_tier_gpu`,
     `test_isolation_gpu[polar1-s-hybrid]`, `test_selfcheck_gpu`.
3. **Collect** which of them each phase 4 task ran on the GPU at its own head (from the task reports). A test
   whose covered code changed after the head it last ran at goes into the `preflight` job.
4. **Run `preflight`** (untimed): those tests only. A failure is triaged before the timed jobs: a stale
   expectation is corrected test-side, in its own commit, with the reason and the change that made it stale (as
   phase-3.md's "Test corrections at acceptance"); a library defect stops the run and goes to the conductor. A
   failure is reproduced on the production path (`Simulation`, `fastmace run`, `fastmace relax`) before it is
   called a library defect.

## Stoppable pytest runs (phase 3 lesson)

Phase 3's partial local tier lost its junit when stopping the chain's shell ended pytest first (open finding 7).
3b-T10 added the fix (tool-side, 3ea191a): `tools/acceptance/stoppable_pytest.sh OUTDIR PYTHON ARGS` with the
plugin `tools/acceptance/resultlog.py`, which keep `OUTDIR/results.jsonl` per finished test and write
`OUTDIR/junit.xml` also when stopped. Every pytest job here uses them. A stop is
`kill -INT "$(cat OUTDIR/pytest.pid)"`, never a kill of the shell or the reservation first; `timeout` wrappers use
`--signal=INT --kill-after=180`. If 3b-T10 is not merged at the freeze, 4-T13 takes the two files from its branch
with their CPU test, unchanged.

## Jobs

Local RTX 5060 Ti, run ID `4-T13-local-<date>`, through `tools/ci/local_gpu.py run` and
`FASTMACE_3B_COMMIT=<sha> tools/acceptance/phase4_local.sh <copy> <job>` (tool-side, modelled on
`phase3b_local.sh`; the jobs of items 1–4 get their commands at the freeze, from 4-T02 to T10's tests and tools);
outputs in `~/fastmace-acceptance/4-T13-local-<date>/`. The timed jobs run first, alone, on a quiet PC (no other
GPU or heavy CPU job; checked and recorded at the start of each).

| Job | Item | Timed | What runs |
| --- | --- | --- | --- |
| `preflight` | – | no | the tests selected above |
| `regression` | 5 | yes | `tools/ab_fastmace.py --release 3b` (tool-side: release `3b`, tag `fastmace-3b`, package `fastmace_3b`): MH-1 hybrid Langevin NVT, D3 off, NaCl and NbCl5, 300 K, 12 blocks of 300 steps after 300; processes ABAB 3b first, ABAB 4 first, ABBA 3b first |
| `regression-npt` | 5 | yes | BAOAB NPT step time, D3 off, NbCl5, one version per process, 4 alternating processes 3b, 4, 4, 3b |
| `shim-ab` | 3 | yes | `fastmace.ase` class against the equivalent `Simulation`, same process (4-T10's A/B), screen then full design (timing design) |
| `step-costs` | rec. | yes | each new integrator against Langevin (NVT) or BAOAB (NPT), same process, 4 blocks of 300 steps each, NaCl and NbCl5 (choice 7) |
| `equivalence` | 1 | no | the ASE-equivalence tests of every scheme, plain cuEq float64 (choice 2), including the two phase 1b ones |
| `optimisers` | 1 | no | GPU LBFGS and FIRE against ASE on `fm.calc`, fixed and variable cell, with and without `FixAtoms` (4-T05 to T07's tests and tools) |
| `minima` | 2 | no | `tools/validation/anneal_minima.py` (4-T08) at the freeze, only if required below; GPU side, then the ASE side if required |
| `restart` | 4 | no | the reproducible-mode restart tests of each new scheme on the GPU |
| `protocol` | 2 | no | the short anneal protocol through `fastmace run` with the GPU quench (D4), start, then resume to the quench |
| `nve-drift` | rec. | no | NVE float32 hybrid, NaCl and NbCl5 3×3×3, 20 ps at 1 fs, conserved quantity every 10 fs (choice 7) |
| `trajectories-cpu` | 5 | no | `tools/trajectory_versions.py --against 3b`, CPU float64, the tests' tiny MH-1-class model, NaCl 2×2×2, 100 steps, positions every 10, Langevin NVT and BAOAB NPT, D3 off and on; `compare --bitwise` |
| `trajectories-reproducible` | 5 | no | the same against `3b`, reproducible plan, plain cuEq float64 on the GPU, MH-1 NaCl, 500 steps, every 50, NVT and NPT, D3 off; `compare --bitwise` |
| `restart-across` | 5 | no | a Langevin restart written by `fastmace_3b` (v3, step 50) resumed at the freeze to step 100, CPU float64, tiny model; bitwise against `fastmace_3b`'s uninterrupted run (tool-side `--resume-across`) |
| `cpu-acceptance` | rec. | no | `pytest tests/cpu -m acceptance`: choice 8's sampling checks |
| `acceptance-tier` | 1–4 | (step-time assertions) | `pytest tests/gpu -m acceptance` |
| `slow-tier` | – | no | `pytest tests/gpu -m "slow and not acceptance"` |

Ada: run ID `4-T13-ada-<date>`, `pytest tests/gpu -m acceptance` on the Modal L4
(`fastmace_modal.py::job --gpu L4 --limit-usd 3`) or leeloo, in the order above, with the result log and junit.

## Timing design (decision 39: repeats sized to the effect)

- **Item 5, NVT** (effect sought: under 1 %): the full same-process alternating design, three processes, the
  local floors measured in phase 3 (NaCl 1.1 %, NbCl5 0.4 %). 4-T04 touches the fused Langevin stage, so this is
  the one item that needs the full design.
- **Item 5, NPT:** a check, not a measurement. One version per process, so only the cross-process floor of about
  3 % applies.
- **Item 3, the shim against `Simulation`** (4-T10, "one block each unless the gap is under 3 %"): a screen of
  one block of 300 steps after 300 per side, same process, every new class on NaCl (the per-step overhead is
  largest there) and `VelocityVerlet` on NbCl5. Ratio shim / `Simulation` at or below 1 + floor: pass. Above
  1.03: a miss, to the conductor. Between: the full alternating design, 12 blocks, one process; its median
  decides.
- **Recorded step costs** (choice 7): one process, 4 alternating blocks per pair. A recorded number has no pass
  mark, so no more repeats.
- **Quench time** GPU against ASE (choice 7): one block, from 4-T08's run or the `minima` job (the effect is a
  multiple, decision 39).
- **Cross-process controls** (phase 3 open finding 2): bounded at the documented 3 % or left out. The
  same-process A/Bs are the measurement; no control clause tighter than 3 %.

## Pass rules (written before the runs)

**Item 1: ASE equivalence** (§12.2, choice 2).
- **Integrators:** NVE (`VelocityVerlet`), CSVR (`Bussi`), NHC (`NoseHooverChainNVT`), MTK (`IsotropicMTKNPT`;
  `MTKNPT` with `cell_mode="full"`), Berendsen (`NVTBerendsen`, `NPTBerendsen`; `Inhomogeneous_NPTBerendsen` where
  4-T02's test covers it), Langevin with `FixAtoms` (D1); and phase 1b's Langevin and BAOAB tests unchanged.
- **Setting:** MH-1 `matpes_r2scan`, NaCl 3×3×3, plain cuEq float64, 4,000 steps, each scheme against its ASE
  3.27 class with MACE's calculator in float64, `dof="ase"`, the same seed where the scheme draws noise (CSVR's
  host stream, choice 3; Langevin's device stream).
- **Pass:** |ΔT| < 1e-6 K and |ΔPE| < 1e-7 meV/atom in every 50 K bin (ramps) or every 500-step block
  (constant targets). NPT schemes: the volume and pressure differences per block are recorded with them.
- **The shortened horizon** (choice 2, checkpoint 2): a deterministic scheme whose chaotic divergence exceeds
  the limits before 4,000 steps is compared over its first 1,000 steps at the same limits, and its growth rate
  is recorded, only if the conductor accepted that at checkpoint 2 for that scheme. The report lists each such
  scheme and the checkpoint's answer. Otherwise a miss is a failure.
- **Optimisers, fixed cell** (one NaCl and one NbCl5 frame of `captured.traj`, fmax 0.01 eV/Å, choice 1's
  defaults, GPU LBFGS and FIRE against ASE 3.27 `LBFGS` and `FIRE` on `fm.calc`):
  - plain cuEq float64: the same number of steps to fmax, and every atom within 1e-8 Å of ASE at each of the
    first 50 steps; the largest position difference per step is recorded to the end;
  - MH-1 hybrid float32: both converge; |ΔE| ≤ 0.025 meV/atom at the minima; every atom within 0.1 Å (D3's
    scale); steps recorded.
- **Optimisers, variable cell** (`FrechetCellFilter`, D5; 4-T07): LBFGS on the ten benchmark crystals 3×3×3 and
  FIRE on NaCl and NbCl5, MH-1 hybrid float32, at 0 GPa, against ASE on `fm.calc` with the same filter: both
  converge; every cell length within 1e-3 Å; volume within 0.01 %; energy within 0.025 meV/atom. NbCl5 once more
  with D3-BJ, same limits.
- **Optimisers with `FixAtoms`** (D1, fixed cell): one NbCl5 frame, a third of the atoms fixed, plain cuEq
  float64, LBFGS and FIRE: the fixed-cell float64 rule, and the fixed atoms' positions bitwise unchanged.
- **Tests:** every test of these rows in the acceptance and slow tiers passes on the RTX 5060 Ti, and the ones the
  Ada check reached pass on sm_89.

**Item 2: the annealing-benchmark minima** (D3, 4-T08).
- **Rule (D3 (a)):** the 210 `captured.traj` frames (NaCl and the nine production crystals, 21 each, 3×3×3),
  MH-1 `matpes_r2scan` hybrid float32; GPU LBFGS (fmax 0.01 eV/Å, memory 100, maxstep 0.2 Å, fixed cell, at
  most 5,000 steps) against ASE's LBFGS on `fm.calc`, same force field and precision. **Pass:** all 210 converge
  on both sides; per frame |ΔE| ≤ 0.025 meV/atom; every atom within 0.1 Å in at least 20 of 21 frames per
  crystal.
- **Citation or rerun:** item 2 cites 4-T08's run if the freeze's code on its path equals 4-T08's run commit.
  - The GPU side reruns at the freeze (minutes) if `git diff <4-T08 run> <freeze>` is not empty in
    `fastmace/optimize`, `engine`, `neighbors`, `model`, `rewrites`, `planner` or `simulation.py`.
  - The ASE side (about 1.1 h) is cited unless the force path behind `fm.calc` changed (`fastmace/ase/calculator*`,
    `model`, `rewrites`, `engine/compiled*`, `planner`) or ASE's version did. Then it reruns too, locally or on
    Modal.
  - The plan names only `optimize/`; the wider list follows phase 3b's item 4 rule (the files on the item's
    path), and the report gives each diff.
- **Recorded, not judged:** the 13 October research-engine minima (`quench.csv`, `comparison.csv`) against the
  GPU minima; quench time GPU against ASE.
- **Production path** (`protocol` job, D4): the short anneal protocol through `fastmace run` at the freeze.
  **Pass:** every stage completes and is recorded (relaxation and quench with GPU LBFGS, `protocol.json` naming
  the optimiser, a restart and resume, the 21 quench frames, all converged or each one's non-convergence
  recorded). **Fail:** an uncaught exception, a NaN, a silent stop, a missing stage or record. The quenched
  energies are recorded, not judged.

**Item 3: §7.7's acceptance for the new classes** (4-T10).
- **Classes:** `VelocityVerlet`, `Bussi`, `NoseHooverChainNVT`, `IsotropicMTKNPT`, `MTKNPT`, `NVTBerendsen`,
  `NPTBerendsen`, `Inhomogeneous_NPTBerendsen`; with D5 (a), `fastmace.ase.LBFGS` and `FIRE` on `Atoms` and on
  `FrechetCellFilter`.
- **Pass, each class:** its argument list equals ASE 3.27's `inspect.signature` split into supported and refused
  (CPU); `rng` and the other refused arguments raise naming the argument; a ported script gets the same plan as
  the equivalent `Simulation`; its `trajectory=` file reads back frame for frame; an observer change gives the
  same state as the `Simulation` event; the step-time rule of the timing design. `fm.md(ensemble="NVE")` and
  `fastmace md --ensemble NVE` run NVE; `fastmace relax` relaxes; the report lines (Berendsen "non-canonical:
  equilibration only", MTK "experimental: not yet qualified") are present.

**Item 4: restart for each new integrator** (§10.6, §12.2 "Restart").
- **Schemes:** NVE, CSVR (with a partly consumed noise batch), NHC, MTK hydrostatic and full (h ≠ h₀ at the
  restart step), Berendsen NVT and NPT, Langevin with `FixAtoms`.
- **Pass:** round trip, and interrupted against uninterrupted at the same restart cadence, **bitwise** on CPU
  float64 (4-T02 to T04's tests, in the freeze's CPU suite) and on the GPU in reproducible mode (plain cuEq
  float64 and hybrid float32, NaCl); the restart record holds the scheme in the identity and the fixed indices in
  the physical record; v0–v3 files still load (CPU). A scheme without a GPU reproducible restart test at the
  freeze goes to the conductor before the runs.

**Item 5: no regression against `fastmace-3b`.**
- **NVT step times:** per structure, the median of the three processes' ratio 4/3b within ± the local floor
  (NaCl 1.1 %, NbCl5 0.4 %), and no process slower than 1 + 2 × floor. Plans: only the new phase 4 fields may
  differ, each resolving to `fastmace-3b`'s behaviour (Langevin scheme, no fixed atoms, `fixcm` and its
  projection as before); any other engine-option difference makes the process invalid.
- **NPT step time:** the 4 processes' median within 3 % of the 3b processes' median.
- **Trajectories, bitwise:** CPU float64 (NVT and NPT, D3 off and on) and the reproducible GPU plan (NVT and NPT,
  D3 off): every recorded row's positions, energy, temperature (and, under NPT, cell and pressure) equal.
- **Restart across versions:** the `restart-across` run equals `fastmace_3b`'s uninterrupted run bitwise.

**Recorded, not acceptance** (choices 7 and 8, decision 6).
- NVE energy drift (float32 hybrid, NaCl and NbCl5, 20 ps): the conserved quantity's linear drift per ps and
  atom, and its RMS fluctuation.
- Each new integrator's step time against Langevin or BAOAB.
- GPU against ASE quench time on the 210 frames.
- The kinetic-energy distributions of CSVR and NHC against the canonical gamma distribution (Kolmogorov–Smirnov,
  fixed seed, tiny model, CPU): the statistic and the p-value. A p-value below 0.001 goes to the conductor before
  the report: with item 1 passing it would point at a test or a degrees-of-freedom error.

**Tests:** every acceptance and slow GPU test passes on the RTX 5060 Ti at the freeze or at a corrected copy
(corrections listed with reasons); the tests the Ada check reached pass on sm_89.

## Tool-side additions (before the go, CPU-tested on `fastmace/4-T13`)

- `tools/acceptance/phase4_local.sh`: the jobs above, modelled on `phase3b_local.sh` (`FASTMACE_3B_COMMIT`, the
  stoppable pytest runner, a GPU and quiet-PC record at each job's start).
- `tools/ab_fastmace.py --release 3b` and `tools/trajectory_versions.py --against 3b`, as 3b-T10 added `3`.
- `tools/trajectory_versions.py --resume-across`: write a restart with the baseline at step k, resume with the
  freeze, compare with the baseline's uninterrupted run.
- Written after 3b-T10 merges (they extend its versions of these files). Jobs that call phase 4 task tools
  (`equivalence`, `optimisers`, `minima`, `restart`, `shim-ab`, `step-costs`, `nve-drift`, `protocol`) get their
  commands at the freeze, as in 3b.

## Not in this runbook

- The qualification of MTK, NPT Berendsen and thermostat settings (plan, "Out of scope").
- Judging physics: NVE drift, quench minima against the research engine, whether a crystal melts (decision 6).
- ResearchModule `main.py` (no change in phase 4); a GPU quench there is a follow-up after checkpoint 3.
- 4-T11 and T12's GPU checks are their own tasks' and are cited; their tests run in the tiers above.
