# Phase 4 acceptance report

**Status: complete, 24 October 2026.** Plan task 4-T13, branch `fastmace/4-T13`. The runbook and pass rules were
written before any result: [phase-4-runbook.md](phase-4-runbook.md) (revision 3 and its notes).

## Freeze, copies and scope

- **Freeze:** main **dae3848**: phase 4 (4-T01 to 4-T12b) and four fixes found by this acceptance, each on its own
  branch from main and merged by the conductor:
  - `fastmace/4-T13-runner-abort` (c041326, merged fbbbf86): the subprocess runner's child joins its heartbeat
    thread before exit. Without that a child sometimes ended in SIGABRT after a correct run (the CPU flake from
    4-T10's trial, [4-T13.md](4-T13.md)).
  - `fastmace/4-T13-compat` (135accd, merged 018db4f): the GPU compatibility tier (`-m compat`,
    `tests/_support/tiers.py`) and `tools/acceptance/compat.py` (decision 39 amended).
  - `fastmace/4-T13-restart-compat` (de9fc21, merged 5c12ee3): a format-3 restart resumes without
    `allow_plan_change`. Its plan record has no `dof`, which now reads as the resolved default.
  - `fastmace/4-T13-csvr-resume` (79654e5, merged 5ff4796): a CSVR run resumes. The restart's `Atoms` carry its
    momenta when the engine is built, so CSVR no longer refuses to start "at rest".
  - dae3848 also carries fm-ops-colab's Colab driver (`tools/colab`) and one line in
    `fastmace/kernels/_oeq_runtime.py` (the OEq driver path on Colab VMs; no effect on the local GPU).
- **First freeze, 5c12ee3,** moved to dae3848 after the CSVR fix. The Colab legs ran at 2696741 (5c12ee3 plus
  tool- and test-side commits). `git diff 2696741 dae3848 -- fastmace` is the CSVR resume fix and the OEq driver
  line, neither on a path the Colab legs measured (no CSVR resume ran there; the driver line is what made the
  later Colab runs work out of the box).
- **Baseline for item 5:** **ff80ca2**, the 3b freeze (the tag `fastmace-3b` is not set; the conductor named
  ff80ca2). Exported as the renamed package `fastmace_3b` (`~/fastmace-4-T13/versions`), and as a plain-package
  `git archive` for the restart across versions (`~/fastmace-4-T13/release-ff80ca2`, below).
- **Pinned copies** (`git archive` of the 4-T13 head, `~/fastmace-4-T13/src-<sha>`, each with `.source_commit`):
  7e6c3ec and 636787b (preflight at the first freeze), a1962e9 (dae3848: preflight part 3, the resume reruns,
  item 5, the extended-energy check), 4c574d2 (the ASE-speed tool fix: `ase-speed`, `gate_resources[omol0-xl]`),
  ba1d5be (two test corrections: the CPU suite, item 3, the reproducible trajectories, the local compat and
  loading tests). The Colab legs archive a commit themselves: 2696741 (the 23 October legs), ba1d5be (the G4
  fix-up leg), d6d2f9d (the A100 float64 probe; ba1d5be plus the compat job's interpreter, tool-side). Every
  copy's `fastmace/` equals its freeze's; later copies differ only in tool- and test-side files.
- **Scope decisions** (owner, relayed by the conductor):
  1. Decision 39 amended (ff6b4ec): the full acceptance on the RTX 5060 Ti, plus a compatibility check per
     production GPU type (Ada through a Colab L4, Blackwell locally, the A100 since its pilot passed).
  2. "Go ahead and move the untime parts to whatever gpu is fastest for its type"; "Dont cap anything for colab".
     The untimed float32 and hybrid work went to a Colab G4 (RTX PRO 6000 Blackwell Server), the float64 work to a
     Colab A100. Timed runs and the research-engine comparisons stayed local. Modal was dropped.
  3. The slow tier of this acceptance also stands in for phase 3b's (dropped there).
  4. Workstations were off: everything local ran on the RTX 5060 Ti, through `tools/ci/local_gpu.py run
     --wait-minutes 600`.

## Verdict

| # | Item | Result |
| --- | --- | --- |
| 1 | ASE equivalence: 8 MD schemes, Langevin with `FixAtoms`, optimisers | **pass**: every scheme in the eager and the engine test; optimisers at fixed cell (float32 and float64 step for step, `FixAtoms`); variable cell after the soft-cell extension, NbCl5 + D3 with the conductor's float32 note (float64 agrees to 3e-6 Å) |
| 2 | Annealing-benchmark minima; the production protocol | **pass**: 210 / 210 converged, every crystal within 20 of 21 frames (NaTaCl6 frame 12, as in 4-T08); the short NbCl5 protocol complete through `fastmace run` |
| 3 | §7.7 for the new classes | **pass**: every ASE class within 1.2 % of `Simulation` (rerun); 4-T10's CPU tests |
| 4 | Restart for each new integrator | **pass** (after the CSVR resume fix) |
| 5 | No regression against ff80ca2 | **pass**: NVT medians 0.9993 (NaCl) and 1.0006 (NbCl5), NPT 0.998; trajectories bitwise (CPU and the reproducible GPU plan); restart across versions bitwise without `allow_plan_change` |
| – | Tests | every acceptance and slow GPU test passes on the RTX 5060 Ti or its Colab leg, after six test-side corrections; two pass only when run alone (`test_extended_energy_over_20_ps`, order dependence, open finding 3; `gate_resources[omol0-xl]`, a timeout under CPU load); CPU suite 3,976 passed, 0 failed |
| – | Compatibility check | Colab L4 (Ada) pass after one test correction; Colab G4 (Blackwell) pass after one test correction; RTX 5060 Ti **pass** (356 / 356 run) |

## Runs

| Run ID | Provider | GPU | Environment | What |
| --- | --- | --- | --- | --- |
| `4-T13-local-20261023`, `-20261024*` (`-timed`, `-speed`, `-item3`, `-cpu`, `-compat`, `-final`, …) | PC, WSL2 | RTX 5060 Ti (sm_120, 16 GiB), driver 610.88 | `~/environments/mace_cueq_oeq` (hybrid), `mace_cueq_latest` (plain cuEq, pytest) | preflight, item 5, timed jobs, resume reruns, targeted reruns, local compat |
| `4-T13-colab-g4-20261023` | Colab | G4: RTX PRO 6000 Blackwell Server Edition (sm_120, 96 GiB), driver 580.82.07, 48 CPUs | Colab venv: torch 2.14.1+cu130, cuEq 0.12.0, OEq 0.7.0, MACE 0.3.16, ASE 3.27.0 | acceptance subset, slow tier, optimisers, minima (failed to start, below), protocol, NVE drift, compat |
| `4-T13-colab-g4-fix-20261024` | Colab | G4, as above | as above | ba1d5be, 03:34–04:08 UTC: the two corrected test files, the soft-cell reruns, minima; 4.45 compute units, VM stopped |
| `4-T13-colab-a100-equivalence-20261023` | Colab | A100-SXM4-40GB (sm_80), driver 580.82.07 | as above | MH-1 equivalence tests (float64 references), `FixAtoms`, 1b Langevin and BAOAB; 56.9 compute units |
| `4-T13-colab-a100-engine-20261023` | Colab | A100, as above | as above | engine against ASE (gap 1); float64 optimiser probes and `FixAtoms`; 39.2 compute units (plus 14.9 for a run lost to a driver poll timeout) |
| `4-T13-colab-l4-20261023`, `-l4-combo-` | Colab | L4 (Ada, sm_89, 23 GiB), driver 580.82.07 | as above | compatibility tier; the combo-kernel rerun; 48.1 + 3.5 compute units |

Model everywhere: MH-1 `matpes_r2scan` (`mace-mh-1.model`) unless a test names another label. Times are UTC unless
marked CEST. Records (JSON, JSONL and logs under 200 KB; no trajectories, `.npz` or models): `results/<run-id>/`
for each run ID above, with the Colab launcher (`results/4-T13-colab-launch.sh`) and item 3's driver
(`results/4-T13-item3.sh`).

## Before the runs: stale expectations (preflight)

The slow and acceptance tests of the 16 GPU files the phase 3 and 3b acceptances found stale or that cover what
phase 4 changed (restart, the planner, the engine graph, selfcheck, the subprocess runner and worker, the reference
tier; `PREFLIGHT_FILES` in `tools/acceptance/phase4_local.sh`) ran first, locally, in three parts. Part 1 ran at
7e6c3ec. Part 2 at 636787b was stopped by SIGTERM in `test_gate_resources_gpu[polar1-l]` when the owner asked for
the GPU. Part 3 at a1962e9 held the gate-resources tests.

- **69 passed, 3 failed**, of which none in the end stands:
  - `test_reproducible_schemes_gpu[csvr-float64]` and `[csvr-float32]` (gap 2's new test): a CSVR run did not
    resume ("CSVR refuses to start at rest"). A real defect, fixed by `fastmace/4-T13-csvr-resume`; the freeze
    moved to dae3848, and the rerun of every resume test there passed 29 of 29 (item 4).
  - `test_gate_resources_gpu[omol0-xl]` hit its 5,400 s timeout while three phase 5 pytest runs loaded the CPU.
    Rerun alone at 4c574d2 (24 October, 03:26–03:29 UTC, the CPU quiet): **passed in 200 s**.
- **Test-side corrections from the preflight:** `test_subprocess_worker_gpu`'s `_listed` (phase 3b's open finding
  5): under WSL2, `nvidia-smi` lists PIDs outside the WSL namespace, so the check now asks whether the worker
  holds GPU memory and whether that memory was released (`WSL_MARGIN_MIB = 128`). Both tests pass locally.

## 1. ASE equivalence

**MD schemes** (pass: |ΔT| < 1e-6 K and |ΔPE| < 1e-7 meV/atom in every block, 4,000 steps, eager test and engine
test). All on the Colab A100 (the float64 references):

| Test | Schemes | Result |
| --- | --- | --- |
| `test_nve_csvr_berendsen_gpu::test_matches_ase_on_mh1` (eager, 4-T02) | NVE, CSVR, NVT, NPT and inhomogeneous NPT Berendsen | 5 / 5 |
| `test_nose_hoover_mtk_mh1_gpu::test_matches_ase_on_mh1` (eager, 4-T03) | NHC, MTK hydrostatic and full | 3 / 3 |
| `test_schemes_engine_ase_gpu` (engine, gap 1, `FixedPlanner` and the graph engine) | the same 8 | 8 / 8 |
| `test_fix_atoms_gpu` (4-T04) | Langevin with `FixAtoms` against ASE; fused against PyTorch stage bytes; bitwise trajectories (Langevin, NVE then CSVR) | 4 / 4 |
| `test_ase_equivalence_gpu` (1b, unchanged) | Langevin in every 50 K bin; the BAOAB variants | 2 / 2 |

The NPT Berendsen tests compare with ASE's class reordered to one force call per step (4-T02's deviation,
accepted).

**Optimisers, fixed cell, float32** (MH-1 hybrid, NaCl and NbCl5 frames 0, 10, 20; pass: both converge,
|ΔE| ≤ 0.025 meV/atom, every atom within 0.1 Å). Colab G4: **pass**, all 12. Every pair converged in exactly
ASE's number of steps, with |ΔE| and the largest distance both 0.0000 at the printed precision:

| Optimiser | NaCl 0 / 10 / 20 (steps, both sides) | NbCl5 0 / 10 / 20 (steps, both sides) |
| --- | --- | --- |
| LBFGS | 51 / 56 / 44 | 169 / 314 / 168 |
| FIRE | 50 / 60 / 49 | 181 / 288 / 133 |

**Optimisers, fixed cell, float64 step for step** (plain cuEq, frame 10; pass: same step count, every atom within
1e-8 Å at each of the first 50 steps). Colab A100: **pass**.

| Probe | Steps ASE / GPU | Largest distance, first 50 steps | At the minimum |
| --- | --- | --- | --- |
| LBFGS NaCl | 56 / 56 | 6.2e-14 Å | 3.8e-14 Å |
| LBFGS NbCl5 | 314 / 314 | 3.9e-13 Å | 7.2e-13 Å |
| FIRE NaCl | 60 / 60 | 4.0e-15 Å | 4.1e-15 Å |
| FIRE NbCl5 | 288 / 288 | 2.0e-14 Å | 4.5e-14 Å |
| LBFGS NbCl5, `FixAtoms` (648 of 1,944) | 151 / 151 | 2.2e-13 Å | 9.4e-12 Å |
| FIRE NbCl5, `FixAtoms` (648 of 1,944) | 258 / 258 | 1.6e-14 Å | 6.2e-14 Å |

With `FixAtoms` the ASE side ran on MACE's own calculator (float64, cuEq), because `fm.calc` refuses every
constraint. On both sides the fixed atoms' positions are bitwise unchanged.

**Optimisers, variable cell** (`FrechetCellFilter`, pressure 0, the ten benchmark crystals 3×3×3, cell ×1.02,
rattled 0.02 Å; MH-1 hybrid float32; Colab G4). Every crystal converged on both sides, in the same number of steps
or within two, and |ΔE| ≤ 0.001 meV/atom:

| Crystal | Steps ASE / GPU | Largest distance | Cell length | Volume (relative) |
| --- | --- | --- | --- | --- |
| NaCl | 48 / 48 | 0.0000 Å | 1.5e-6 Å | 2.6e-7 |
| NaNbCl6 | 189 / 189 | 0.0000 Å | 6.0e-7 Å | 3.1e-8 |
| NaTaCl6 | 186 / 186 | 0.0000 Å | 5.2e-6 Å | 3.0e-7 |
| NaNbO3 | 225 / 227 | 0.0027 Å | **1.23e-3 Å** | **1.31e-4** |
| NaTaO3 | 131 / 131 | 0.0000 Å | 9.3e-7 Å | 5.7e-8 |
| NbCl3O | 209 / 209 | 0.0000 Å | 5.7e-6 Å | 5.6e-7 |
| TaCl3O | 203 / 203 | 0.0000 Å | 2.5e-6 Å | 2.8e-7 |
| NbCl2O | 218 / 218 | 0.0000 Å | 5.3e-6 Å | 1.1e-7 |
| NbCl5 | 251 / 251 | 0.0000 Å | 6.8e-6 Å | 3.8e-7 |
| TaCl5 | 284 / 284 | 0.0002 Å | 2.2e-5 Å | 1.2e-6 |
| NbCl5 + D3-BJ | 658 / 658 | 0.0051 Å | **4.0e-3 Å** | 1.2e-5 |

- NaNbO3 and NbCl5 with D3 pass the energy and atom clauses and miss only the cell clauses, as in 4-T07
  (NaNbO3 1.2e-3 Å and 1.3e-4 there, NbCl5 + D3 9.1e-3 Å). The pre-declared soft-cell extension relaxed both
  again on both sides from the same start to fmax 0.002 eV/Å (G4 fix-up leg, ba1d5be, 24 October):

| Crystal, fmax 0.002 | Steps ASE / GPU | |ΔE| | Largest distance | Cell length | Volume (relative) | Result |
| --- | --- | --- | --- | --- | --- | --- |
| NaNbO3 | 277 / 277 | 0.0000 meV/atom | 0.0000 Å | 4.3e-6 Å | 1.4e-7 | **pass** |
| NbCl5 + D3-BJ | 870 / 872 | 0.0001 meV/atom | 0.0033 Å | **2.83e-3 Å** | 4.4e-5 | **fails the cell-length clause** |

  NaNbO3's miss was the fmax tolerance on a flat cell landscape. NbCl5 with D3 still missed the cell-length
  limit (1e-3 Å) at five times the precision, though the gap shrank (4.0e-3 → 2.8e-3 Å). The energy, atom and
  volume clauses pass, and both sides end at the same minimum (0.0001 meV/atom apart). As the rule says, it went
  to the conductor, who asked for a float64 probe: the same set-up (cell filter, D3-BJ, the same start), plain
  cuEq float64 on both sides (Colab A100, d6d2f9d, 24 October 04:13–04:25 UTC, 1.2 compute units, VM stopped):

| NbCl5 + D3-BJ, float64 | Steps ASE / GPU | |ΔE| | Largest distance | Cell length | Volume (relative) |
| --- | --- | --- | --- | --- | --- |
| fmax 0.01 | 660 / 660 | 0.0000 meV/atom | 0.0000 Å | 6.9e-6 Å | 1.2e-7 |
| fmax 0.002 | 876 / 876 | 0.0000 meV/atom | 0.0000 Å | 2.9e-6 Å | 1.3e-7 |

  In float64 the two optimisers take the same steps and end at the same cell, within 1e-5 Å. The float32 miss is
  precision: on D3's flat cell direction in NbCl5, float32 forces and stress fix the cell only to a few 1e-3 Å.
  **Pass, with that note** (the conductor's ruling): a known float32 limit of variable-cell relaxation with D3.
- **FIRE with the cell filter, NaCl:** 43 steps on both sides, |ΔE| 0.0000 meV/atom, cell 3.3e-8 Å, volume 1.1e-9: **pass**.
  NbCl5, recorded, not judged: the GPU side stopped at 5,000 steps with 446 switch flips, as ASE's own FIRE did in
  4-T07.
- `fastmace relax` (NaCl 3×3×3, `gpu-LBFGS`, fmax 0.01) ran on the G4: the CIF is already at its fixed-cell
  minimum (fmax 0.0000 eV/Å, 0 steps); output and summary written.
- **CPU basis:** the freeze's CPU suite (ba1d5be, 24 October: 3,976 passed, 244 skipped, 0 failed), which holds each MD scheme against ASE (`test_nve_csvr_berendsen`,
  `test_nose_hoover_mtk`) and the optimisers step for step (`test_optimize_lbfgs`, `test_fire`,
  `test_optimize_cell`): passed.

## 2. The annealing-benchmark minima and the production path

**Minima** (decision 46 amended). `git diff 6eb0b76 dae3848` is not empty in the GPU side's paths (nine files:
`planner/selfcheck.py`, `simulation.py` and the phase 4 merges after 4-T08), so the GPU side reruns. It is empty
in the ASE side's force path (`fastmace/ase/calculator*`, `model`, `rewrites`, `engine/compiled*`,
`planner/fixed.py`), and ASE is 3.27 on both, so the GPU side is compared with 4-T08's ASE minima.

- First attempt (G4, 23 October): did not run. The upload's `minima-ase/<crystal>` entries were symlinks into
  `~/fastmace-runs/4-T08-rtx5060ti-20261023/` and dangled on the VM, so no reference was copied and no GPU side ran
  (evaluate: "0 of 21 frames relaxed on both sides" for every crystal). My error, not FastMACE's: the upload now
  holds real copies.
- Rerun (G4 fix-up leg, ba1d5be, 24 October 03:49–04:07 UTC; every one of the 210 GPU relaxations converged,
  compared with 4-T08's ASE minima by `tools/validation/anneal_minima.py evaluate`): **pass**.

| Crystal | Same structure | Largest |ΔE| (meV/atom) | Largest distance (Å) |
| --- | ---: | ---: | ---: |
| NaCl | 21 / 21 | 0.0002 | 1.9e-6 |
| NaNbCl6 | 21 / 21 | 0.0058 | 0.033 |
| NaTaCl6 | 20 / 21 | **0.1922** (frame 12) | 2.70 (frame 12) |
| NaNbO3 | 21 / 21 | 0.0015 | 0.0073 |
| NaTaO3 | 21 / 21 | 0.0008 | 0.0008 |
| NbCl3O | 21 / 21 | 0.0023 | 0.0064 |
| TaCl3O | 21 / 21 | 0.0002 | 8.4e-6 |
| NbCl2O | 21 / 21 | 0.0003 | 0.0006 |
| NbCl5 | 21 / 21 | 0.0016 | 0.0076 |
| TaCl5 | 21 / 21 | 0.0073 | 0.051 |

  NaTaCl6 frame 12 is the one frame outside the limits, as in 4-T08 (0.188 meV/atom there: two stationary
  minima, which the float64 GPU and ASE agree on to 1.6e-12 Å). Each crystal is within 20 of 21 frames on both
  clauses (decision 46 amended).

**Production path** (`npt_anneal_NbCl5_short` through `fastmace run`, start then resume, Colab G4): **pass**.
- `protocol.json`: status `complete`, relaxation `gpu-LBFGS` (fixed cell, converged in 17 steps, fmax 0.0087
  eV/Å), D3 (r2SCAN BJ), two pieces of 13,500 steps (the second through `fastmace run --resume`), D3 build counters
  per piece.
- Quench: 21 frames on one engine (`quench.json` `engines` 1), each at its own NPT cell, `gpu-LBFGS`. Ten frames
  (the cooling ramp from 925 K and the final frame) stopped at the 5,000-step cap, and `quench.json` records
  them (`not_converged` 10), as 4-T09 found for the melt frames. Final quenched −10.633878 eV/atom.
- No exception, NaN or silent stop.

## 3. §7.7 for the new classes

- **CPU tests** (4-T10's: `test_ase_md_4`, `test_ase_optimize` and the rest): passed in the freeze's CPU suite (ba1d5be, 24 October: 3,976 passed, 244 skipped, 0 failed).
- **Step time against `Simulation`:** 4-T10's run 3 (c10b4c0) could not be cited as the runbook allows: since
  then `simulation.py` changed (79654e5, `_restart_atoms` only, a resume path), and run 3 timed three classes, not
  all eight. Rerun as 4-T10 ran it (`tools/ab_frontends.py --integrator <class> --sides ase simulation md`, MH-1
  hybrid float32, NaCl 3×3×3, 0.5 fs, 300 K, 12 alternating blocks of 300 steps after 300 in one process; local,
  ba1d5be, `--quiet`, 24 October 04:28–04:44 UTC, nothing else on the GPU or the CPU suites):

| Class | ASE class ms/step | `Simulation` ms/step | `fm.md` ms/step | ASE class / `Simulation` |
| --- | ---: | ---: | ---: | ---: |
| Langevin (1b) | 2.968 | 2.968 | 2.968 | 1.000 |
| VelocityVerlet | 2.970 | 2.953 | 2.964 | 1.006 |
| Bussi | 3.047 | 3.011 | 3.045 | 1.012 |
| NoseHooverChainNVT | 3.079 | 3.057 | 3.056 | 1.007 |
| IsotropicMTKNPT | 3.728 | 3.724 | 3.721 | 1.001 |
| MTKNPT | 3.771 | 3.770 | 3.788 | 1.000 |
| NVTBerendsen | 3.030 | 3.028 | 3.029 | 1.001 |
| NPTBerendsen | 3.571 | 3.556 | 3.570 | 1.004 |
| Inhomogeneous_NPTBerendsen | 3.590 | 3.568 | 3.583 | 1.006 |

  Every class within 1.2 % of `Simulation` (limit ±3 %), every side with the same plan: **pass**.
- `fastmace relax` on the GPU: ran (item 1).
- **Recorded:** `fm.calc` refuses `FixAtoms`, so ASE's own optimisers cannot relax a constrained structure on it.
  `fastmace.ase.LBFGS` and `FIRE` take `FixAtoms`.

## 4. Restart for each new integrator

- **GPU, reproducible mode** (gap 2, `test_reproducible_schemes_gpu`: an interrupted run against the uninterrupted
  one, bitwise, for NVE, CSVR inside a noise batch, NHC, MTK hydrostatic and full with h ≠ h₀, NVT and NPT
  Berendsen, Langevin with `FixAtoms`, in float64 and float32). The first run at 5c12ee3 found the CSVR resume
  defect (16 / 18). At dae3848, with every other resume test of the preflight (`test_reproducible_gpu` 5,
  `test_canonical_restart_gpu` 2, `test_subprocess_runner_gpu` 2, `test_subprocess_worker_gpu` 2): **29 / 29**
  (24 October, 00:31–01:29 UTC). **Pass.**
- **CPU** (round trip, interrupted against uninterrupted, v0–v3 files load: `test_restart`, `test_restart_v3`,
  `test_canonical_restart`, `test_csvr_resume`, `test_restart_compat`): passed in the freeze's CPU suite (ba1d5be, 24 October: 3,976 passed, 244 skipped, 0 failed).

## 5. No regression against ff80ca2

**NVT step times** (`tools/ab_fastmace.py --release 3b`: MH-1 hybrid Langevin NVT, D3 off, 300 K, 12 blocks of 300
steps after 300, three processes; dae3848 against `fastmace_3b`; 24 October 01:47–02:01 UTC, 03:47–04:01 CEST,
`--quiet`):

| Process | NaCl 4/3b | NbCl5 4/3b |
| --- | ---: | ---: |
| ABAB, 3b built first | 1.0023 | 1.0014 |
| ABAB, 4 built first | 0.9993 | 0.9969 |
| ABBA, 3b built first | 0.9965 | 1.0006 |
| **median** (floor) | **0.9993** (1.1 %) | **1.0006** (0.4 %) |

All six processes valid; no plan difference; no compile or patch change while timed. Medians within the floor, no
process above 1 + 2 × floor (1.022, 1.008): **pass**.

**NPT step time** (BAOAB, NbCl5, one version per process, order 3b, 4, 4, 3b): 3b 20.84 and 21.01 ms, 4 20.90
and 20.86 ms; medians 20.92 and 20.88, ratio **0.998**, within 3 %: **pass**.

**Trajectories, bitwise** (`tools/trajectory_versions.py --against 3b`):
- CPU float64 (the tiny MH-1-class model, NaCl 2×2×2, 100 steps, every 10), Langevin NVT and BAOAB NPT, D3 off and
  on (`--dispersion`, an explicit `D3(functional="r2scan")`: the tiny model has no training functional): all four
  `compare --bitwise` **passed**, at the first freeze (7e6c3ec) and again at dae3848 (a1962e9).
- Reproducible GPU plan (plain cuEq float64, MH-1 NaCl, 500 steps, every 50), NVT and NPT, D3 off, local,
  ba1d5be (24 October 03:37–03:44 UTC): both `compare --bitwise` **passed**.

**Restart across versions** (`tools/restart_versions.py`): a Langevin restart written by ff80ca2 (format 3, step
50; the release run as the plain package from `git archive`, because the renamed `fastmace_3b` writes engine
"fastmace_3b", which resume refuses as a different engine) resumed at the freeze to step 100, CPU float64, tiny
model: **bitwise** equal to ff80ca2's uninterrupted run, resumed with `plan_change` False and plan revision 0, at
both freezes. **Pass.**

## Recorded, not acceptance

**NVE energy drift** (`tools/bench/nve_drift.py`, the default plan, MH-1 hybrid float32 forces, 3×3×3, 300 K,
20 ps at 1 fs, conserved quantity every 10 fs; Colab G4):

| Structure | Atoms | Drift (meV/atom/ps) | RMS (meV/atom) | Range (meV/atom) |
| --- | ---: | ---: | ---: | ---: |
| NaCl | 216 | 2.1e-6 | 2.7e-4 | 6.7e-3 |
| NbCl5 | 1,944 | 2.3e-6 | 3.8e-4 | 1.3e-2 |

**Extended energies over 20 ps** (`test_extended_energy_over_20_ps`, MH-1, local, a1962e9, run alone): NHC 7.1e-5,
MTK hydrostatic 6.6e-5, MTK full 1.4e-4 meV/atom/ps; 3 / 3 passed. (On the A100 the three failed after other CUDA
tests in the same process; open finding 3.)

**End to end against stock ASE** (`tools/bench/ase_integrators_speed.py`: stock ASE 3.27 with MACE's calculator,
cuEq float32, against the `fastmace.ase` class in the default plan; one 300-step block after 300 per side and
process; NaCl and NbCl5 3×3×3, 1 fs, 300 K; local, 4c574d2, `--quiet`, 24 October 02:47–03:24 UTC). Cross-process:
ratios within 3 % of 1 count as no difference.

| Class | NaCl stock / FastMACE (ms) | ×  | NbCl5 stock / FastMACE (ms) | × | FastMACE / Langevin NaCl, NbCl5 |
| --- | --- | ---: | --- | ---: | --- |
| Langevin | 44.08 / 2.99 | 14.7 | 135.19 / 21.53 | 6.3 | 1, 1 |
| VelocityVerlet | 42.61 / 2.95 | 14.5 | 133.81 / 21.03 | 6.4 | 0.986, 0.977 |
| Bussi | 43.81 / 3.05 | 14.3 | 136.10 / 21.77 | 6.3 | 1.022, 1.011 |
| NoseHooverChainNVT | 41.43 / 3.09 | 13.4 | 134.75 / 21.33 | 6.3 | 1.035, 0.991 |
| IsotropicMTKNPT | 42.23 / 3.85 | 11.0 | 128.02 / 21.01 | 6.1 | 1.289, 0.976 |
| MTKNPT | 41.14 / 3.94 | 10.4 | 127.82 / 21.34 | 6.0 | 1.318, 0.991 |
| NVTBerendsen | 41.33 / 3.06 | 13.5 | 134.99 / 21.24 | 6.4 | 1.023, 0.987 |
| NPTBerendsen | 85.64 / 3.73 | 23.0 | 269.87 / 22.02 | 12.3 | 1.247, 1.023 |
| Inhomogeneous_NPTBerendsen | 79.99 / 3.76 | 21.3 | 270.06 / 22.11 | 12.2 | 1.258, 1.027 |

- FastMACE is 10–23× faster than the stock ASE route on NaCl (216 atoms) and 6–12× on NbCl5 (1,944 atoms). ASE's
  NPT Berendsen calls the forces twice per step, hence its doubled stock time.
- Against Langevin: the NVT classes are within 3.5 % on NaCl, and within 3 % (no difference) on NbCl5. On NaCl the
  variable-cell classes cost 1.25–1.32× Langevin (the stress path at 216 atoms). On NbCl5 the difference is gone.
- The first attempt's FastMACE sides failed after the first class: the shim's default output `./fastmace-run`
  refuses a second run. Fixed tool-side (4c574d2, one output directory per run), and the whole job reran.

**Kinetic-energy distributions** (gap 3, `tests/cpu/test_sampling_canonical.py`, Lennard-Jones argon, fixed seeds,
KS against the canonical gamma distribution): CSVR p = 0.26, NHC p = 0.77; the Berendsen control p = 7e-33
(correctly rejected). Nothing below 0.001 for the canonical schemes.

**Model fitness, for the owner (4-T12b):** POLAR-1-S's forces on NbCl5 3×3×3 are about 70× MH-1's (largest 119.6
against 1.7 eV/Å). A model question, recorded.

## Tests

| Tier | Where | Result |
| --- | --- | --- |
| Preflight (16 files, slow and acceptance) | local, 7e6c3ec / 636787b / a1962e9 | 69 passed, 3 failed: CSVR resume (fixed), `gate_resources[omol0-xl]` timeout under load (passes alone) |
| Resume tests at dae3848 | local, a1962e9 | 29 / 29 |
| Acceptance tier minus preflight and float64 files | Colab G4, 2696741 | 17 / 17 (`test_fallback_gpu` 8, `test_forces_gpu` 2, `test_mc_statistics_gpu` 5, `test_subprocess_runner_gpu` 2) |
| Float64 acceptance files | Colab A100, 2696741 | 14 passed, 3 failed (extended energy, order dependence; pass alone locally) |
| Engine against ASE (gap 1) | Colab A100, 2696741 | 8 / 8 |
| Slow tier, not acceptance, minus preflight and float64 files | Colab G4, 2696741 | 26 passed, 1 failed (`test_template_cache_gpu[hybrid-float32]`, probe timeout; corrected test-side) |
| The two corrected files (`test_mc_evaluator_gpu`, `test_template_cache_gpu`) | Colab G4, ba1d5be | 8 / 8 |
| `test_loading_gpu.py`, whole file (the 3 slow every-label tests included; neither the preflight nor a Colab leg held it) | local, ba1d5be | 22 / 22 |
| Compatibility tier | Colab L4 | 356 passed, 2 skipped, 1 failed (`test_combo_kernels_change_no_value`, corrected test-side; the rerun passed) |
| Compatibility tier | Colab G4 | 356 passed, 2 skipped, 1 failed (`test_energy_only_graph_and_evaluator_tiny_plain_cueq`, 1 float64 ulp; corrected test-side, passes in the rerun above) |
| Compatibility tier | local RTX 5060 Ti, ba1d5be | 356 passed, 3 skipped, 0 failed (35 min; pytest in `mace_cueq_latest`, hybrid children in `mace_cueq_oeq`). Skipped: the two OEq streaming-kernel tests (no OEq in the pytest interpreter; they passed on the G4, also sm_120) and `test_polar1s_reference_equals_1a_t19` (needs `FASTMACE_1A_T19_POLAR1S`, skipped on the L4 too) |
| CPU suite at the freeze, all of `tests/cpu` (`-n 8`, nice) | local, ba1d5be, `fastmace_cpu` | 3,976 passed, 244 skipped, 0 failed (35.5 min) |

**Test-side corrections** (each its own commit on `fastmace/4-T13`, each a test assumption, not a library change):
1. `test_subprocess_worker_gpu`'s `_listed` under WSL2 (above).
2. `test_polar_gpu`: the n2 check within 2 float64 ulps, not equal (A100 pilot, sm_80).
3. `test_energy_gpu`: the float64-energy cross check's floor 1e-8 in float32 (1e-12 in float64) (A100 pilot).
4. `test_combo_kernels_cuda`: three "off" evaluations as the noise estimate, plus 4 float32 ulps. One pair
   was too small an estimate on the G4 and the L4 (a repeat of 4.8e-7 against a change of 1.2e-6). A fusion that
   changed the arithmetic moves values by far more. The L4 rerun passed.
5. `test_mc_evaluator_gpu`: the after-MD baseline within 4 float64 ulps of a fresh evaluation, not equal (G4:
   101.1376298522307 against 101.13762985223069).
6. `test_template_cache_gpu`: the OpenEquivariance probe waits 900 s, not 120 s. A first import on a fresh VM may
   build the extension and wait up to 5 minutes for its lock. The `cueq-float32` case of the same test passed.

## Open findings

1. **Float32 limit for D3 variable-cell relaxation:** NbCl5 + D3's cell agrees with ASE's only to 2.8e-3 Å in
   float32 (limit 1e-3), and to 3e-6 Å in float64 with identical steps. Recorded as a known limit (item 1,
   conductor's ruling). A user who needs the cell to 1e-3 Å with D3 should relax in float64.
2. **The plan record names `Langevin` for every fixed-cell MD scheme.** `planner/fixed.py` writes
   `integrator = Langevin (NVT, fixcm=…)` for any non-NPT scheme. The NVE drift run's plan says Langevin, though it
   ran `fm.integrators.NVE` (the built `IntegratorConfig.scheme` is right, and the drift is NVE's). The record line
   predates phase 4. Follow-up: name the scheme (and keep the baseline's text for Langevin, which item 5's plan
   comparison relies on).
3. **Extended-energy order dependence:** `test_extended_energy_over_20_ps` fails with `ProcessConstraintError`
   (the allocator) after other CUDA tests in one process (A100, and locally at dae3848). It passes alone.
   Follow-up: run it in a fresh child (`run_fresh`), as other process-sensitive tests do.
4. **A100 margin case** (5-T03, Colab A100, fb67bdc; not this acceptance's legs): on omol0-xl NaCl hybrid float32
   the self-check found `reassociate_first` at a relative error of 1.1e-4 against 1e-4. Rule 13 drops
   `reassociate_first` and `segmented_first_sum`, and the runs proceed without them. NbCl5 and polar1-l are fine.
   My A100 legs did not cover omol0-xl.
5. **Quench convergence:** 10 of 21 quench relaxations of the short protocol hit the 5,000-step cap (the cooling
   ramp's liquid frames), as in 4-T09 and 3b.
6. **Infrastructure, found in this acceptance:**
   - WSL's VM restarted at 00:41 CEST on 24 October under memory pressure (several CPU suites at once). Since then
     one heavy CPU job at a time runs alongside a GPU job.
   - A pattern kill (`pkill -f "sleep [0-9]"`, another session's) ended one of my Colab chains. Kill by PID only.
   - Two Colab legs launched in the same second shared a session name and orphaned a VM; the owner stopped it.
     The driver now names sessions uniquely (fm-ops-colab 4c239b0), and a poll timeout that crashed a leg is fixed
     (e1bbb96).
   - Colab uploads must not contain symlinks (the minima's first attempt).
   - `FastMACE unknown` in `fastmace run`'s piece log on Colab: the source is a `git archive` without `.git`.
     Cosmetic.

## Reproducing

From a FastMACE checkout `$FM`, in WSL, with a pinned copy `$SRC` (`git archive <sha>` plus `.source_commit`):

```bash
FASTMACE_3B_COMMIT=ff80ca2a33d3c1ce25b3181095ff7b2a42a7f00d python3 $SRC/tools/ci/local_gpu.py run --wait-minutes 600 --purpose "4-T13 regression" -- bash $SRC/tools/acceptance/phase4_local.sh $SRC regression
```

The other jobs are listed in the runbook's table and at the top of `tools/acceptance/phase4_local.sh`. The Colab
legs are `~/fastmace-4-T13/colab-launch.sh <leg>` (g4, a100-equivalence, a100-engine, l4, l4-combo, g4-fix) on
fm-ops-colab's driver (`tools/colab/fastmace_colab.py`, e1bbb96).
