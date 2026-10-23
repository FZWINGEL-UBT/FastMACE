# Runbook: phase 4 acceptance

Plan task 4-T13. First written 23 October 2026 as a prestart, before any phase 4 task was merged, and accepted by
the conductor then (078462b). **Revision 2**, the same evening, at the go (main 184f572, every phase 4 build task
merged): it folds in decisions 44–50, the owner's slow-tier ruling (8fe335e) and the task reports' findings, and
sets the rules the plan left to this runbook (variable-cell cells, item 2's citation). The phase 4 results that
already exist are the tasks' own runs; every rule below is fixed before any run at the freeze. Results go to
`phase-4.md` and `results/4-T13-*/`. 4-T13 changes no library code; its tool and test changes are labelled
tool-side or test-side. Nothing runs before the freeze.

**Review (23 October 2026):** the conductor accepted revision 2 and relayed the owner's approval of the soft-cell
re-relaxation and of FIRE with the cell filter judged on NaCl only. **Revision 3**, the same evening: decision 39 is
amended (ff6b4ec; owner, relayed by the conductor). Acceptance splits into the full acceptance on the RTX 5060 Ti
and the GPU compatibility check (`-m compat`) on every production GPU type. The Ada spot check becomes the
compatibility check on a Colab L4, and the Modal launch is dropped. An A100 leg may take the untimed runs.

Sources: SPEC §16 row 4, §7.7, §10.2–§10.3, §10.6, §12.2 ("ASE equivalence", "Restart"), §18 decisions 6, 37,
39 and 44–50; `plans/phase-4.md` (choices 1–10, task 4-T13); the task reports `4-T02.md` to `4-T12b.md`;
`phase-3.md` with its open findings 1, 2, 3 and 7; `phase-3b-runbook.md` and 3b-T10's tool-side additions
(`fastmace/3b-T10`).

## Decisions in force

- **44** `FixAtoms` with NVE, Langevin, CSVR, NVT Berendsen, and FIRE and LBFGS at fixed cell; refused with the
  reason for NHC, MTK, every variable cell and `fixcm=True`.
- **45** `fixcm=True`: 3N − 3 and the momentum projected out; `FixAtoms`: 3(N − n_fixed), no projection;
  `dof="ase"` is ASE's count and the equivalence tests use it.
- **46, amended** the minima: both clauses (energy 0.025 meV/atom, structure 0.1 Å) each in at least 20 of 21
  frames per crystal; all 210 converge.
- **47** `[relax]` and `[quench]` default to `gpu-LBFGS`.
- **48** `fastmace.ase.LBFGS` and `FIRE` on `Atoms` or `FrechetCellFilter`.
- **49** PolarMACE stress floor 5e-7 eV/Å³ in the reference record.
- **50** PolarMACE with a variable-cell optimiser is refused (`polar-variable-cell-optimizer`).
- **Owner, 23 October 2026 (8fe335e):** the GPU slow tier at this freeze also stands in for phase 3b's, which was
  dropped.
- **39, amended (ff6b4ec):** full acceptance once per phase on the reference GPU (the RTX 5060 Ti): the acceptance
  and slow tiers, the physics comparisons and the timed A/Bs; its untimed runs may run elsewhere (a Colab A100).
  The GPU compatibility check (`-m compat`, about 30–60 min) on every production GPU type: Ada (a Colab L4 stands
  in, for correctness only), Blackwell (the local PC), and the Colab A100 if its check passes. 4-T13 defines the
  tier (branch `fastmace/4-T13-compat`, merged before the freeze).

## Freeze, baseline and copies

- **Freeze:** the main commit with every phase 4 task merged, named by the conductor (main 184f572 has 4-T01 to
  T12b and 4-T08b). The report records it and the commit whose `fastmace/` it equals.
- **Baseline:** tag `fastmace-3b`; if it is not set at the freeze, the 3b freeze commit ff80ca2 (conductor), and the
  report says so. Exported as the renamed package with
  `python -m fastmace.bench.versions export --ref <baseline> --name fastmace_3b --destination ~/fastmace-4-T13/versions`.
- **Pinned copies:** `git archive` of the 4-T13 head with `.source_commit`, `fastmace/` equal to the freeze's, at
  `~/fastmace-4-T13/src-<sha>` (ext4, not `/mnt/d`: 1b-T21b and 4-T10 measured `/mnt/d` distorting timing). A
  later test correction makes a new copy; the report lists each copy and what ran from it.
- **Environments:** local `mace_cueq_oeq` (MH-1 hybrid, timed items) and `mace_cueq_latest` (plain cuEq, pytest,
  float64 equivalence, the ASE sides). Both have ASE 3.27.0 (task reports); the first job checks it again and
  stops on any other version. The report records ASE, torch, Triton, cuEquivariance, openequivariance and MACE
  versions per environment.
- **Every result names** the machine, GPU, driver, environment, backend, precision, model label (MH-1
  `matpes_r2scan` unless stated), the integrator or optimiser with every argument, the `dof` convention, and the
  commits of the freeze and the baseline.

## Hardware and budget

- **The full acceptance runs on the local RTX 5060 Ti** (workstations off), after 3b-T10's queued jobs drain:
  through `tools/ci/local_gpu.py run --wait-minutes 600` from the pinned copy, as background tasks of the 4-T13
  session. The queue is FIFO; 3b-T10's queued jobs run first, and no other session's job is stopped. The local
  compatibility check (`compat` job) runs here too, since Blackwell is a production type.
- **The compatibility check on a Colab L4** (Ada, sm_89, standing in for the workstations, correctness only):
  `tools/acceptance/compat.py --provider colab --label "Colab L4"` at the freeze, through the `colab` CLI in WSL
  (`colab run --gpu L4`, or `colab new` / `exec -f` / `stop` for a longer session, always stopped afterwards), with
  the setup script of the conductor's A100 pilot adapted to the L4. The owner set no cap on Colab compute units. No
  Modal job runs in phase 4. The tasks' Modal L4 runs (4-T04, 4-T07, 4-T08, 4-T09) stay context, each at its own
  commit.
- **The Colab A100 leg**, only if the conductor's A100 pilot shows the stack working on sm_80: the compatibility
  check there (A100 then becomes a production type), and the untimed parts of the full acceptance to finish
  sooner: `acceptance-tier` and `slow-tier` split between the A100 and the local GPU by file, `optimisers`,
  `minima` (if required), `protocol`, `trajectories-reproducible` (both versions on the same A100). Timed jobs
  (`regression`, `regression-npt`, `ase-speed`, the step-time assertions of the acceptance tier) stay on the RTX
  5060 Ti; a step-time assertion that runs on the A100 is reported, not judged. Results from the A100 are judged
  by the same pass rules and are labelled with GPU and provider.
- **Every result is labelled** with GPU and provider (local, colab), as the driver's `compat.json` and the job
  logs record them.
- **The A100 pilot** (conductor's helper; Colab A100-SXM4-40GB, sm_80, FastMACE fbbbf86): smoke, the hybrid gate,
  `library_gate` and the statistical restart passed; the fast tier had 324 passed, 2 failed, 18 skipped (16 D3
  tests without `dftd3`, which fm-ops-colab adds; OEq needs `OEQ_CUDA_DRIVER_LIBRARY=/usr/lib64-nvidia/libcuda.so.1`).
  The two failures are rounding-level expectations that sm_80 breaks, judged here as architecture differences and
  corrected test-side for every GPU (each in its own commit; they still pass on sm_120 in the local `compat` job):
  - `test_polar_gpu::test_fixed_k_vectors_on_cuda[float64]`: 16 of 88 `n2 = |k|^2` values 1 ulp off graph_longrange's
    kernel on the same GPU; `n2` within 2 ulp in float64, `k` still bitwise;
  - `test_energy_gpu::test_float64_energy_replays_and_keeps_forces[default-dtype0]` (float32): the two fields
    differed by 1.6e-9 eV/Å with almost no scatter noise (7.5e-10); the bound's relative floor follows the precision
    (1e-8 in float32, 1e-12 in float64), and the 1e-6 eV/Å negative control still fails it.
- **Timed jobs first, alone, on a quiet PC:** no other GPU job (the reservation) and no heavy CPU job (checked and
  recorded at each job's start). No CPU suite of the conductor's during the timed jobs (the conductor is told the
  window).

## Before any run: stale expectations (phase 3 lesson)

Trial merges are CPU-only, so GPU tests go stale (phase-3.md open finding 1: seven did). Phase 3b dropped its slow
tier, so the stale-test listing starts at `fastmace-3`, not at 3b.

1. **List** every GPU test marked `slow` or `acceptance` (33 files at 184f572), and for each whether its file or
   the library code it exercises changed between `fastmace-3` and the freeze.
2. **Read** each changed-path test's expectations, at least:
   - restart format 4 (4-T01; `state_keys` checked exactly): `test_canonical_restart_gpu`, `test_reproducible_gpu`;
   - plan and provenance shapes (`IntegratorConfig.ensemble`, `scheme`, `Plan.optimizer`, rule 1's table, rules 10
     and 11 for optimisers, the D3 fields of 3b): `test_planner_gpu`, `test_simulation_gpu`, `test_selfcheck_gpu`,
     `test_gate_resources_gpu`, `test_isolation_gpu`;
   - the fused Langevin stage with the `FIXED` drift variant (4-T04): `test_fused_integrator_gpu`,
     `test_engine_gpu` (the B graph must not change; phase-3.md open finding 8), `test_registry_gpu`;
   - `reset_cell` and `update(cell=)` (4-T05, 4-T09) and the D3 list: `test_fixed_costs_gpu`, `test_events_gpu`;
   - the PolarMACE gate (decisions 49 and 50, 4-T11): `test_polar_gpu`, `test_reference_tier_gpu`,
     `test_reference_gpu`, `test_selfcheck_gpu`;
   - refusals phase 4 lifts (`FixAtoms`, `nve-phase-4`, `LATER`, the cell filter);
   - 3b's own changes, since its slow tier never ran (NPT's stress path and pressure setter, D3).
3. **Collect** which of them each phase 3b and 4 task ran on the GPU at its own head (task reports). A test whose
   covered code changed after that goes into `preflight`.
4. **Run `preflight`** (untimed). A failure is triaged before the timed jobs: a stale expectation is corrected
   test-side, in its own commit, with the reason and the change that made it stale; a library defect stops the run
   and goes to the conductor. A failure is reproduced on the production path (`Simulation`, `fastmace run`,
   `fastmace relax`) before it is called a library defect.

**Pre-declared correction:** `test_subprocess_worker_gpu.py`'s two `_listed(pid)` assertions cannot pass under
WSL2: `nvidia-smi --query-compute-apps=pid` reports PIDs from outside WSL's namespace (or none), so a worker's PID
never matches. Test-side fix (written before the go, verified in `preflight`): `_listed` stays where `nvidia-smi`
lists PIDs; under WSL (`microsoft` in `/proc/version`) "holds the GPU" becomes "the process is alive
(`/proc/<pid>`) and the GPU's used memory is more than 128 MiB above the reading taken before the worker started",
and "released" becomes "the process is gone and the used memory is back within 128 MiB". The margin is wider than
the tests' 64 MiB free-memory check because WSL's reading is the whole GPU's, the Windows desktop's included; a
worker's CUDA context alone is several hundred MiB. The killed-parent check then asserts that the worker exits and
the memory returns within 30 s, which is what the test means by "kept the GPU". Both tests' other assertions are
unchanged.

## Stoppable pytest runs (phase 3 lesson)

3b-T10's `tools/acceptance/stoppable_pytest.sh` and `resultlog.py` (results per finished test in
`results.jsonl`, `junit.xml` also when stopped) run every pytest job. A stop is
`kill -INT "$(cat OUTDIR/pytest.pid)"`, never a kill of the shell or the reservation first; `timeout` wrappers use
`--signal=INT --kill-after=180`. If 3b-T10 is not merged at the freeze, 4-T13 takes the two files and their CPU
test from its branch unchanged.

## Coverage gaps closed before the freeze (test-side)

1. **The new schemes through the engine against ASE.** 4-T02's and 4-T03's equivalence tests drive the
   integrators eagerly with the state on the GPU ("The engine's run of the new schemes is 4-T10's and 4-T13's"),
   and 4-T10 timed the engine path but did not compare it with ASE. Only Langevin with `FixAtoms` (4-T04) ran the
   graph engine against ASE. New acceptance test `tests/gpu/test_schemes_engine_ase_gpu.py`: each of NVE, CSVR,
   NHC, MTK hydrostatic and full, NVT, NPT and inhomogeneous NPT Berendsen through `fm.Simulation` (planner, graph
   engine, captured, plain cuEq float64, `dof="ase"`, `fixcm=False`), with 4-T02's and 4-T03's setting (NaCl
   3×3×3, 4,000 steps of 0.5 fs at 300 K from the same zero-momentum start; NHC tdamp 50 fs; MTK 1 GPa, tdamp
   50 fs, pdamp 500 fs; Berendsen as 4-T02), against the ASE class on MACE's calculator, choice 2's limits per
   500-step block. Expected GPU time about 2 h (8 × about 15 min, the ASE side dominating).
2. **Restart on the GPU for the new schemes.** `test_reproducible_gpu.py` covers Langevin only. Test-side
   addition: the interrupted-against-uninterrupted test parametrised over NVE, CSVR (restart inside a partly
   consumed noise batch), NHC, MTK hydrostatic and full (h ≠ h₀ at the restart step), NVT and NPT Berendsen and
   Langevin with `FixAtoms`, reproducible plan, MH-1 NaCl, float64 and float32, 200 steps with a restart at 100.

3. **The sampling checks of choice 8 did not exist.** New CPU acceptance test `tests/cpu/test_sampling_canonical.py`:
   CSVR and NHC (`fixcm=True`, `dof="auto"`, N_f = 3N − 3) on 4-T03's Lennard-Jones argon (108 atoms, 150 K, 2 fs,
   τ 20 fs), 2 ps equilibration, then 400 kinetic-energy samples every 100 fs, one-sample KS against
   Gamma(N_f/2, k_B T); NVT Berendsen as the negative control (the same test must reject it).

The GPU files are written on `fastmace/4-T13` and CPU-collected. Gap 2's runs first in `preflight` at the
freeze; gap 1's (about 2 h) runs in the acceptance tier after the timed stretch (changed at the freeze for speed: a
defect it finds cannot change the Langevin and BAOAB timings). A failure is triaged as above. The CPU file runs in
`cpu-acceptance`.

## Jobs

Local RTX 5060 Ti, run ID `4-T13-local-<date>`, `python3 <copy>/tools/ci/local_gpu.py run --wait-minutes 600
--purpose "4-T13 <job>" -- bash <copy>/tools/acceptance/phase4_local.sh <copy> <job>` (tool-side, modelled on
`phase3b_local.sh`); outputs in `~/fastmace-acceptance/4-T13-local-<date>/`.

| Job | Item | Timed | What runs | Estimate |
| --- | --- | --- | --- | --- |
| `preflight` | – | no | the selected tests, the two new test files, the `_listed` correction | 3 h |
| `regression` | 5 | yes | `tools/ab_fastmace.py --release 3b`: MH-1 hybrid Langevin NVT, D3 off, NaCl and NbCl5, 300 K, 12 blocks of 300 steps after 300; ABAB 3b first, ABAB 4 first, ABBA 3b first | 40 min |
| `regression-npt` | 5 | yes | BAOAB NPT, D3 off, NbCl5, one version per process, 4 processes 3b, 4, 4, 3b | 30 min |
| `ase-speed` | rec. | yes | each new ASE class and `Langevin` on stock ASE + MACE's calculator against its `fastmace.ase` class, NaCl and NbCl5, one side per process (tool-side `tools/bench/ase_integrators_speed.py`); the FastMACE sides are also the step costs against Langevin | 1 h |
| `optimisers` | 1 | no | fixed cell float32 and variable cell: `tools/bench/optimizer_frames.py` (ASE side, then the GPU side with `--compare`); float64 step for step and `FixAtoms`: `tools/validation/anneal_frame_probe.py run` per side (`--precision float64`, `--optimizer`, `--fix-every 3 --ase-calculator mace`), then `compare`; `fastmace relax` once | 2 h |
| `minima` | 2 | no | `tools/validation/anneal_minima.py` at the freeze, only if required below | 0.5–1.5 h |
| `protocol` | 2 | no | the short NbCl5 NPT anneal protocol through `fastmace run` with `gpu-LBFGS` (decision 47), start, then resume to the quench | 1.5 h |
| `nve-drift` | rec. | no | `tools/bench/nve_drift.py` (tool-side): NVE in the default plan, NaCl and NbCl5 3×3×3, 20 ps at 1 fs, conserved quantity every 10 fs | 15 min |
| `trajectories-cpu` | 5 | no | `tools/trajectory_versions.py --against 3b`, CPU float64, tiny MH-1-class model, NaCl 2×2×2, 100 steps, every 10, Langevin NVT and BAOAB NPT, D3 off and on; `compare --bitwise` | 15 min |
| `trajectories-reproducible` | 5 | no | the same, reproducible plan, plain cuEq float64 on the GPU, MH-1 NaCl, 500 steps, every 50, NVT and NPT, D3 off | 20 min |
| `restart-across` | 5 | no | a Langevin restart written by the baseline (format 3, step 50; a `git archive` of it run as the plain package) resumed at the freeze to step 100, CPU float64, tiny model; bitwise against the baseline's uninterrupted run (tool-side `tools/restart_versions.py`) | 10 min |
| `compat` | – | no | `tools/acceptance/compat.py --provider local --label "RTX 5060 Ti"`: the compatibility tier on Blackwell | 30 min |
| `cpu-acceptance` | rec. | no | `pytest tests/cpu -m acceptance`: the sampling checks (choice 8, gap 3) and the CPU acceptance tests already there | 20 min |
| `acceptance-tier` | 1–4 | (step-time assertions) | `pytest tests/gpu -m acceptance` | 5–6 h |
| `slow-tier` | – | no | `pytest tests/gpu -m "slow and not acceptance"`, also for phase 3b | 2–3 h |

Order: `preflight`; the timed jobs (`regression`, `regression-npt`, `ase-speed`) in one quiet
stretch; then the rest, the acceptance and slow tiers overnight. About 18–20 h of local GPU in all, less with the
A100 leg. Colab, in parallel from the freeze: the L4 compatibility check (Colab run ID `4-T13-colab-l4-<date>`),
and the A100's (`4-T13-colab-a100-<date>`) with its untimed share.

## Timing design (decision 39: repeats sized to the effect)

- **Item 5, NVT** (effect sought: under 1 %): the full same-process alternating design, three processes, the
  local floors (NaCl 1.1 %, NbCl5 0.4 %). 4-T04 put the `FIXED` variant into the fused Langevin stage (taken only
  with fixed atoms), so this is the item that needs the full design.
- **Item 5, NPT:** a check, one version per process, the cross-process floor of about 3 %.
- **Item 3, the shim against `Simulation`:** cited from 4-T10's run 3 (c10b4c0: 0.997–1.005, alternating blocks in
  one process, 12 blocks), if `fastmace/ase` and `simulation.py` are unchanged between c10b4c0 and the freeze;
  otherwise rerun as 4-T10 ran it, and the median of 12 blocks decides.
- **Against stock ASE** (recorded; the effect is a multiple): one process per side, one block of 300 steps after
  300 warm-up, per class and structure. Cross-process, so a ratio within 3 % of 1 is reported as "no difference".
- **Recorded step costs** of each new integrator against Langevin (NVT classes) and BAOAB (`ase-speed`'s
  `Langevin` side and 1b's BAOAB step time): one block per class and process, cross-process, so a difference
  within 3 % is reported as none; no pass mark, no more repeats.
- **Quench time GPU against ASE:** cited from 4-T08 (one block per crystal, separate processes); not rerun.
- **Cross-process controls** (phase 3 open finding 2): bounded at 3 % or left out.

## Pass rules (fixed before the runs at the freeze)

**Item 1: ASE equivalence** (§12.2, choice 2).
- **MD schemes:** NVE, CSVR, NHC, MTK hydrostatic and full, NVT, NPT and inhomogeneous NPT Berendsen, Langevin with
  `FixAtoms`, and phase 1b's Langevin and BAOAB tests unchanged. NPT Berendsen is compared with ASE's class
  reordered to one force call per step, and its distance from ASE's own two-call step is recorded (4-T02's
  deviation, accepted by the conductor).
- **Pass, each scheme:** |ΔT| < 1e-6 K and |ΔPE| < 1e-7 meV/atom in every 500-step block (constant targets) or
  50 K bin (ramps), over the full 4,000 steps, in **both** the eager test (4-T02, 4-T03) and the engine test (gap
  1). The shortened horizon is not used: every scheme met the limits over 4,000 steps in its task (checkpoint 2
  needs no answer). NPT schemes: the volume and pressure differences per block are recorded.
- **Optimisers, fixed cell,** NaCl and NbCl5 frames 0, 10 and 20 of `captured.traj`, fmax 0.01 eV/Å, ASE 3.27
  defaults, GPU LBFGS and FIRE against ASE's on `fm.calc`:
  - plain cuEq float64 (NaCl and NbCl5 frame 10): the same number of steps to fmax, and every atom within 1e-8 Å
    of ASE at each of the first 50 steps (4-T08's float64 probe: 1.6e-12 Å over 310 steps); the largest position
    difference per step recorded to the end;
  - MH-1 hybrid float32: both converge; |ΔE| ≤ 0.025 meV/atom; every atom within 0.1 Å; steps recorded.
- **Optimisers with `FixAtoms`** (decision 44, fixed cell): NbCl5 frame 10, a third of the atoms fixed, plain cuEq
  float64, LBFGS and FIRE. **The ASE side runs on MACE's own calculator** (float64, cuEq), because `fm.calc`
  refuses every constraint (4-T04, `planner/fixed.py`): the float64 rule above, and the fixed atoms' positions
  bitwise unchanged.
- **Optimisers, variable cell** (`FrechetCellFilter`, pressure 0; 4-T07's set-up: the ten benchmark crystals
  3×3×3, cell ×1.02, rattled 0.02 Å, MH-1 hybrid float32, fmax 0.01 eV/Å, ASE's optimiser with ASE's filter on
  `fm.calc`):
  - **LBFGS on the ten crystals, and NbCl5 with D3-BJ.** Pass, each: both converge; |ΔE| ≤ 0.025 meV/atom; every
    atom within 0.1 Å (minimum image, after scaling to a common cell); and cell lengths within 1e-3 Å and volume
    within 0.01 %.
  - **The soft-cell extension (pre-declared; 4-T07 found NaNbO3 at 1.2e-3 Å and 1.3e-4, NbCl5 + D3 at 9.1e-3 Å
    and 2.4e-4, both at the same minima):** a crystal that passes the energy and atom clauses and misses only a
    cell clause is relaxed again on both sides from the same start to fmax 0.002 eV/Å (`--fmax 0.002`). It passes
    if the cells are then within the limits (the miss was the fmax tolerance on a flat cell landscape); still
    missing, it fails and goes to the conductor. The report gives both comparisons.
  - **FIRE on NaCl.** NbCl5 has no FIRE reference: ASE's own FIRE with the filter does not converge there in 5,000
    steps (4-T07, 447 switch flips). The GPU side runs and is recorded, not judged.
  - **PolarMACE:** the refusal of decision 50 (CPU tests); no comparison (4-T12b: the model's own surface,
    119.6 eV/Å largest force on NbCl5 against MH-1's 1.7).
- **CPU basis, cited from the freeze's CPU suite:** each MD scheme against ASE within 1e-12 Å (1e-9 Å for NHC and
  MTK on Lennard-Jones, 4-T03's limit); optimisers step for step within 1e-12 Å (FIRE, LBFGS at fixed cell) and
  1e-10 Å (**LBFGS with the cell filter**: ASE's cell rows sit at |x| ≈ 505, where ASE drifts 2.5e-11 Å from
  itself under one ulp of noise; 4-T07's deviation, accepted).

**Item 2: the annealing-benchmark minima** (decision 46 amended, 4-T08).
- **Rule:** the 210 frames, GPU LBFGS against ASE's LBFGS on `fm.calc`, MH-1 hybrid float32. Pass: all 210
  converge on both sides; per crystal |ΔE| ≤ 0.025 meV/atom in at least 20 of 21 frames, and every atom within
  0.1 Å in at least 20 of 21 frames.
- **4-T08's result (6eb0b76):** passes; NaTaCl6 frame 12 is the one frame outside the energy clause (0.188
  meV/atom, two stationary minima; float64 GPU and ASE agree to 1.6e-12 Å).
- **Citation or rerun:**
  - the GPU side reruns at the freeze (about 30 min with the engine builds) if `git diff 6eb0b76 <freeze>` is not
    empty in `fastmace/optimize`, `engine`, `neighbors`, `model`, `rewrites`, `planner` or `simulation.py`;
  - the ASE side (about 40 min of optimiser time plus compiles) reruns only if the force path behind `fm.calc`
    changed (`fastmace/ase/calculator*`, `model`, `rewrites`, `engine/compiled*`, `planner/fixed.py`) or ASE did;
    otherwise the rerun GPU side is compared with 4-T08's ASE minima (`.npz` in `~/fastmace-runs/`);
  - the report gives each diff. 4-T08's eight Modal crystals rerun locally if they rerun at all.
- **Recorded, not judged:** the 13 October research-engine minima; quench time (4-T08).
- **Production path** (`protocol` job; decisions 40 and 47): the short NbCl5 NPT anneal (`npt_anneal_NbCl5_short`)
  through `fastmace run` at the freeze. **Pass:** every stage completes and is recorded: the `gpu-LBFGS`
  relaxation; the NPT ramp with D3; a restart and resume; the quench on one engine (`quench.json` `engines` 1), each
  frame at its own NPT cell; `protocol.json` naming `gpu-LBFGS`; each frame converged or its non-convergence
  recorded (4-T09: the vapour frames stop at the step cap, as with ASE). **Fail:** an uncaught exception, a NaN,
  a silent stop, a missing stage or record. Energies and convergence are recorded, not judged.

**Item 3: §7.7 for the new classes** (4-T10).
- **Classes:** `VelocityVerlet`, `Bussi`, `NoseHooverChainNVT`, `IsotropicMTKNPT`, `MTKNPT`, `NVTBerendsen`,
  `NPTBerendsen`, `Inhomogeneous_NPTBerendsen`, `LBFGS`, `FIRE`.
- **Pass:** 4-T10's CPU tests in the freeze's CPU suite (signatures against ASE 3.27, refusals, the same plan as
  `Simulation`, `trajectory=` frame for frame, observers equal to events, `FixAtoms` where it runs and where it is
  refused, the report lines); the step time within ±3 % of `Simulation`'s per class in alternating blocks (cited
  or rerun as the timing design says); `fm.md(ensemble="NVE")`, `fastmace md --ensemble NVE` and `fastmace relax`
  run (CPU tests, and `fastmace relax` once on the GPU in `optimisers`).
- **Recorded:** `fm.calc` refuses `FixAtoms` (4-T04), so ASE's own optimisers cannot relax a constrained structure
  on it. A known limitation in the report, not a failure: `fastmace.ase.LBFGS` and `FIRE` take `FixAtoms`.

**Item 4: restart for each new integrator** (§10.6, §12.2).
- **Pass:** round trip and interrupted against uninterrupted, **bitwise**, on CPU float64 (4-T02 to T04's tests in
  the CPU suite) and on the GPU in reproducible mode, float64 and float32 (gap 2), for NVE, CSVR inside a noise
  batch, NHC, MTK hydrostatic and full with h ≠ h₀, NVT and NPT Berendsen, Langevin with `FixAtoms`; the restart
  record holds the scheme and the fixed indices; v0–v3 files load (CPU).

**Item 5: no regression against the baseline.**
- **NVT step times:** per structure, the median of the three processes' ratio 4/3b within ± the local floor, and
  no process slower than 1 + 2 × floor. Plans: only the phase 4 fields may differ, each resolving to the baseline's
  behaviour (Langevin scheme, no fixed atoms, `fixcm` and its projection as before); any other engine-option
  difference makes the process invalid.
- **NPT step time:** the 4 processes' median within 3 % of the 3b processes' median.
- **Trajectories, bitwise:** CPU float64 (NVT and NPT, D3 off and on) and the reproducible GPU plan (NVT and NPT,
  D3 off): every row's positions, energy, temperature (under NPT, cell and pressure) equal.
- **Restart across versions:** `restart-across` equals the baseline's uninterrupted run bitwise.
  - **And it resumes without `allow_plan_change`, with no new plan revision** (conductor, 23 October 2026: SPEC
    §10.6 promises that older files load). The tools' CPU dry run before the freeze found the refusal ("the plan
    differs from the restart's in integrator": a format-3 plan record has no `dof`, read back as `"auto"` against
    the fresh plan's resolved `"fastmace"`). The fix is on `fastmace/4-T13-restart-compat` (de9fc21), merged
    before the freeze. The job records which way it resumed (`plan_change`).

**Recorded, not acceptance** (choices 7 and 8, decision 6).
- NVE energy drift (float32 hybrid, NaCl and NbCl5, 20 ps): linear drift per ps and atom, RMS fluctuation;
  4-T03's 20 ps extended energies of NHC and MTK cited.
- Each new integrator's step time against Langevin or BAOAB (`ase-speed`; 4-T03b's NHC and MTK numbers as
  context).
- **End to end against stock ASE** (the conductor's item; so far only the optimisers were measured: 4-T08 1.3–4.9×
  on the quench): each new MD class on stock ASE 3.27 with MACE's own calculator (cuEq float32, MACE's defaults,
  the way a user runs it today) against its `fastmace.ase` class with the planner's default (MH-1 hybrid float32),
  MH-1 `matpes_r2scan`, NaCl and NbCl5 3×3×3, 1 fs, 300 K, the time constants of 4-T10's A/B, pressure 0. ms/step
  and the ratio per class and structure.
- Quench time GPU against ASE (4-T08).
- The kinetic-energy distributions of CSVR and NHC against the canonical gamma distribution (gap 3: KS, fixed
  seeds, Lennard-Jones argon, CPU): statistic and p-value, and the Berendsen control's. A p-value below 0.001
  (or a control that passes) goes to the conductor before the report.
- **Model fitness, for the owner (4-T12b):** POLAR-1-S's forces on NbCl5 3×3×3 are about 70× MH-1's at the same
  structure (largest 119.6 against 1.7 eV/Å; 1,250 of 1,944 atoms above 10 eV/Å). Whether POLAR-1-S is fit for
  NbCl5 is a model question, recorded, not a FastMACE finding.

**Tests:** every acceptance and slow GPU test passes on the RTX 5060 Ti at the freeze or at a corrected copy
(corrections listed with reasons; tests of the split tiers on the A100, if that leg runs, count there); the
compatibility check passes on the RTX 5060 Ti and the Colab L4 (and on the A100 if that leg runs).

## The CPU flake from 4-T10's trial

`tests/cpu/test_subprocess_runner.py::test_mc_attaches_keeps_and_detaches_through_the_runner[fresh-children]`:
under `-n 8` the child was killed by SIGABRT ("terminate called without an active exception") after it had written
an `ok` result; 3 of 3 passed alone. That message is C++'s `std::terminate` from destroying a joinable
`std::thread`, here at the child's interpreter exit. **Found** (23 October 2026, CPU): the child's `_Heartbeat` is a
daemon thread whose loop calls `int(engine.state.step)`, a torch call that releases the GIL; `main()`'s `finally`
calls `heartbeat.stop()`, which sets the event but does not join. If the thread is inside that call when the
interpreter finalises, CPython 3.10 ends it with `pthread_exit`, and the forced unwind through the noexcept
GIL-reacquire calls `std::terminate`. Reproduced: a daemon thread in `int(tensor)` at exit aborts 40 of 40 runs
(0 of 40 when joined); the real `_Heartbeat` with its window widened aborts 29 of 30 with `stop()` as written,
0 of 30 with a join. Proposed fix (library code, the conductor's call): `stop()` joins the thread, and `main()`
clears `heartbeat.simulation` first. The parent rightly fails such a call: it accepts a result only with exit
code 0. It does not block the acceptance; the report records it with the fix's state. Note:
`4-T13-finding-runner-child-abort.md` in the session notes.

## Tool-side additions (before the freeze, CPU-tested on `fastmace/4-T13`)

- `tools/acceptance/phase4_local.sh`: the jobs above.
- `tools/bench/ase_integrators_speed.py`: stock ASE and `fastmace.ase` sides, one per process; `tools/bench/nve_drift.py`.
- `tools/validation/anneal_frame_probe.py`: `--optimizer fire`, `--fix-every`, `--ase-calculator mace`, and the
  first-50-steps maximum and fixed-atom check in `compare`.
- `tools/ab_fastmace.py --release 3b`, `tools/trajectory_versions.py --against 3b --dispersion none|d3bj` (D3(BJ)
  with r2SCAN's parameters, explicit for the tiny model), and `tools/restart_versions.py` (the cross-version
  restart), after 3b-T10's merge.
- Test-side: the `_listed` correction and the three coverage-gap test files.

## Not in this runbook

- The qualification of MTK, NPT Berendsen and thermostat settings (plan, "Out of scope").
- Judging physics: NVE drift, quench minima against the research engine, whether a crystal melts (decision 6).
- ResearchModule `main.py` (no change in phase 4); a GPU quench there is a follow-up after checkpoint 3.
- 4-T11, T12 and T12b's GPU checks are their tasks' and are cited; their tests run in the tiers above.
