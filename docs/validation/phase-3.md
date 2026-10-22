# Phase 3 acceptance report

**Status: complete, 22 October 2026, with the GPU slow tier as the appendix.** Plan
task 3-T21, branch `fastmace/3-T21`. Every item passes on the frozen code, with test corrections that are
listed below and leave `fastmace/` unchanged. Two clauses miss as written: item 10's cross-process NbCl5
control (the same-process A/Bs pass), and item 3's OMOL-0 XL row (3-T11's open finding, not rerun). The
runbook and pass rules were written before the timed results: [phase-3-runbook.md](phase-3-runbook.md).

## Freeze, copies and scope

- **Freeze:** main **c3bf93f**. Every phase 3 task is merged in it except 3-T16b (ResearchModule, merged there
  since at 3cc4207f) and 3-T18 (coraline). Its `fastmace/` equals **cd43463**'s (`git diff cd43463 c3bf93f --
  fastmace` is empty), and no 3-T21 commit changes `fastmace/`.
- **Baseline for item 10:** main **156ab76**, just before the 3-T01 merge, tag `phase-3-baseline`
  (conductor's answer 1), exported as the renamed package `fastmace_p3b`.
- **Pinned copies:** `git archive` of the 3-T21 head with `.source_commit`, each with `fastmace/` equal to the
  freeze's:
  - local `~/fastmace-3-T21/src-<sha>`: 1d76cc1 (timed items), 208f09a (the stopped local tier), 676a8a3 (the
    corrected tests, the slow tier, items 3 and 9), 57d924b (`test_fixed_costs_gpu` and item 10's CPU check);
  - workstations `/mnt/homes/zwingel/fastmace-t21-cd43463` (`git archive cd43463`).
- **Scope, cut by the owner on 22 October 2026 in three steps:**
  1. "leeloo's acceptance tier is the one Ada spot check (no L4 runs), item 9 as the short protocol at the
     freeze citing 3-T14's full runs, and fewer timing repeats where the effect is large."
  2. The workstations stopped at 16:40 CEST (14:40 UTC) for Friday's power outage; coraline (RTX A4000) took the
     acceptance files leeloo could not reach by then.
  3. "Yes: stop the full local tier and run only the 7 corrected tests, the GPU slow tier, item 9's short
     protocol, item 3 and item 10's tiny-model bitwise check." All 39 acceptance tests had then run on
     workstation GPUs at the frozen code (leeloo 26, coraline 13). Later the same day: "Report first: pause
     the slow tier, … write the report, then finish the slow tier tonight as an appendix before the tag."
- **Modal:** not used. Cost **$0** (budget $6).

## Verdict

| # | Item | RTX 5060 Ti (local) | Ada / workstation | Result |
| --- | --- | --- | --- | --- |
| 1 | Event speed ≥ 90 % (swap-back, 1500 K) | NaCl 0.939 / 0.948, NbCl5 0.991 / 0.993 | 3-T20's leeloo series (context only, 4a1dfa8) | **pass** |
| 2 | MC attempt cost (no pass mark) | measured, one round per case | 3-T20's L4 table (context) | **measured** |
| 3 | float32 ΔE against float64 | MH-1 `matpes_r2scan` and `omol`, worst 0.0023 k_B·T_min (limit 0.01) | – | **pass** (OMOL-0 XL not measured: open finding 4) |
| 4 | Fresh construction after events | tests below | leeloo: the acceptance tests; coraline: reproducible mode | **pass** |
| 5 | Exact enumeration | CPU suite (trial merges); [fixcm] case rerun after its correction | leeloo: populations (swap, transmute), T_kin free and negative | **pass** |
| 6 | Restart across MC | CPU suite (trial merges) | leeloo: canonical restart (hybrid, statistical); coraline: reproducible restart bitwise | **pass** |
| 7 | Overflow and re-capture | CPU suite (trial merges) | – | **pass** (CPU; see "Coverage") |
| 8 | Compiled tier | slow tier: MH-1 hybrid and MP-0 within float32 noise; fallback order | leeloo: fallback falls to compiled | **pass** |
| 9 | Swap-MC protocol at 1 fs | short protocol at the freeze: every stage | 3-T14's full leeloo run (context, ed55026) | **pass** |
| 10 | No regression against 156ab76 | A/B: NaCl median 0.9996, NbCl5 0.9965; CPU float64 bitwise | – | **pass**, except the cross-process NbCl5 control clause (1.017 against 1.008) |
| 11 | Carry-overs | CPU tests 119 / 119; `main.py` at 1 fs read at ResearchModule main | – | **pass** |

The test files behind items 4–8 are listed under "Coverage"; the ones that needed a correction are under
"Test corrections at acceptance".

## Runs

| Run ID | Machine | GPU | Environment | FastMACE | What |
| --- | --- | --- | --- | --- | --- |
| `3-T21-local-20261022` | PC, WSL | RTX 5060 Ti (Blackwell, sm_120), driver 610.88 | `~/environments/mace_cueq_oeq` (hybrid), `mace_cueq_latest` (plain cuEq, pytest) | copies above | timed: items 10, 1, 2 and rule 9, 14:30–15:12 CEST (12:30–13:12 UTC), PC quiet; then the stopped acceptance tier, the corrected tests, the paused slow tier, items 3 and 9, the B-graph diagnosis, item 10 on the CPU |
| `3-T21-leeloo-20261022` | leeloo | RTX 4000 Ada (sm_89), driver 595.84 | NAS `mace_cueq_latest` (pytest), `mace_cueq_oeq` (hybrid children) | cd43463 | acceptance tier, 13:57–16:12 CEST (11:57–14:12 UTC), stopped by SIGINT after `test_mc_statistics_gpu.py` |
| `3-T21-coraline-20261022` | coraline | RTX A4000 (Ampere, sm_86), driver 595.84 | as leeloo | cd43463 | the 5 acceptance files leeloo did not reach, 15:41–16:25 CEST (13:41–14:25 UTC) |

- Every workstation job ran after an idle check (no compute process on the GPU, no other user's active job),
  as a background task of this session, under a `timeout` ending before the 16:40 CEST (14:40 UTC) shutdown. Both GPUs were
  empty afterwards. Files only under `/mnt/homes/zwingel`.
- Every local GPU job went through `tools/ci/local_gpu.py`. The conductor kept the PC quiet for the timed
  stretch.
- Small result files are in `results/3-T21-local-20261022`, `results/3-T21-leeloo-20261022` and
  `results/3-T21-coraline-20261022`. Force dumps and trajectories stay on the machines.

## 1. Event speed

`tools/bench/event_speed.py --swap-back --settle 5` (3-T20's tool and protocol): MH-1 `matpes_r2scan` hybrid
float32, 1 fs, 1500 K, swaps of two atoms' species every 100 steps with `exchange` momenta and no energy
evaluation, the previous pair swapped back on odd events; 12 blocks of 300 steps after 300, same process.

| Structure | Build order, pattern | fixed (ms/step) | events (ms/step) | events / fixed rate |
| --- | --- | ---: | ---: | ---: |
| NaCl (216) | fixed first, ABAB | 3.003 | 3.197 | **0.939** |
| NaCl | events first, ABBA | 3.004 | 3.169 | **0.948** |
| NbCl5 (1944) | fixed first, ABAB | 21.761 | 21.954 | **0.991** |
| NbCl5 | events first, ABBA | 21.730 | 21.894 | **0.993** |

All four runs valid, 36 events timed each. **Pass** (≥ 0.90). The superset rows were dropped by the owner's
cut; 3-T20's superset rows (local 0.949–0.990, leeloo 0.933–0.989, at 4a1dfa8) are context only.
3-T20's leeloo series ran at 4a1dfa8, whose `fastmace/` differs from the freeze in 14 files, so it is
**context, not a citation**; the same holds for 3-T14's leeloo runs (ed55026).

## 2. MC attempt cost

`tools/bench/mc_cost.py`, 3-T20's cases at 1500 K, **one round per case** (owner's cut), so each value carries
about one noise floor more spread than 3-T20's eight rounds. Cost as a fraction of one MD step of the same
process:

| Structure | MD step | swap f32 | swap f64 | displacement, no trial rebuild | displacement with trial rebuild |
| --- | ---: | ---: | ---: | ---: | ---: |
| NaCl (216) | 2.93–3.09 ms | 0.59–0.60 | 11.1–11.5 | 0.71 (2.10 ms graph attempt) | 5.09 (14.9 ms) |
| NbCl5 (1944) | 21.0–21.1 ms | 0.46 | 12.7–12.8 | 0.48 (10.0 ms) | 1.51 (31.8 ms) |

Energy-only evaluation: float32 0.40–0.43 of an MD step, float64 11.2–12.8. A measurement with no pass mark
(§10.4 "Cost"). The float64 MC energies cost 11–13 MD steps per evaluation on this card.

## 3. ΔE precision

`tools/validation/move_energy_precision.py` (3-T11's tool and protocol) at the freeze (676a8a3), local RTX 5060 Ti:
MH-1 hybrid float32 graph plan, MC workload over the superset {Na, Nb, O, Cl}; per structure NVT at 300 K and
then 1500 K, 2000 steps of 1 fs each, a canonical point, then 4 swaps per superset pair, 2 atoms per present
species transmuted to every other superset species and 8 displacements of 0.15 Å. ΔE_ref is the gate's float64
reference. Worst |ΔE_prod − ΔE_ref| per move type in k_B·T_min, T_min = 250 K, so the gate's limit is **0.01**:

| Model | Structure | T (K) | Swap | Transmute | Displace | float64 copy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MH-1 `matpes_r2scan` | NaCl (216) | 300 | 0.00041 | 0.00030 | 0.00022 | 0 |
| | | 1500 | 0.00042 | 0.00017 | 0.00030 | 0 |
| | NbCl5 (1944) | 300 | 0.0013 | 0.00034 | 0.00050 | 0 |
| | | 1500 | 0.00092 | 0.00064 | 0.00077 | 0 |
| | NaNbO3 (1080) | 300 | **0.0023** | 0.0013 | 0.0012 | 0 |
| | | 1500 | 0.0021 | 0.0015 | 0.00067 | 0 |
| MH-1 `omol` (all-electron E0) | NaCl | 300 | 0.00033 | 0.00019 | 0.00026 | 0 |
| | | 1500 | 0.00056 | 0.00034 | 0.00019 | 0 |
| | NbCl5 | 300 | 0.0012 | 0.00058 | 0.00063 | 2e-06 |
| | | 1500 | 0.0015 | 0.00046 | 0.00045 | 2e-06 |
| | NaNbO3 | 300 | 0.0012 | 0.0010 | 0.00064 | 0 |
| | | 1500 | 0.0016 | 0.0015 | 0.00096 | 0 |

- Every float32 error is below 0.01 k_B·T_min; the worst is 0.0023 (NaNbO3 swaps at 300 K), a margin of about 4.
  The values match 3-T11's local table and 3-T20's L4 table (worst 0.0035 and 0.0028) in size.
- **What `mc_energy_precision="auto"` picks:** float32, the force precision, for MH-1 (`matpes_r2scan` and
  `omol`) on NaCl, NbCl5 and NaNbO3 at T_min ≥ 250 K. The gate decides it per run (decision 35).
- **OMOL-0 XL:** not measured (open finding 4).

## 4–8. The GPU tests

The items' GPU tests ran in three places (CPU tests run in every trial merge's CPU suite and are not repeated
here). "Coverage" below lists every acceptance test with where it ran.

- **Item 4 (fresh construction):** the acceptance tests ran on leeloo (ASE equivalence, canonical restart,
  isolation A-B-A, forces) and coraline (reproducible mode). The events' fast-tier GPU tests use the
  Lennard-Jones force field (3-T03's open finding); the hybrid "within the gate" part is covered by
  `test_mc_statistics_gpu` and `test_compiled_gpu`.
- **Item 5 (exact enumeration):** leeloo passed the χ² populations for swaps and transmutations and the
  kinetic temperature without fixcm (free) and the negative control. The fixcm case failed there on a stale
  expectation and passes locally after its correction, with the redraw_all check strict now.
- **Item 6 (restart across MC):** leeloo's hybrid canonical restart (statistical) and coraline's reproducible
  restarts (bitwise, NaCl and NbCl5, float32 and float64, and a child process).
- **Item 7 (overflow and re-capture):** 3-T04's tests (forced edge-bucket overflow in an MD step and in an MC
  attempt, the variant cap, the counters) are in the CPU suite; the GPU counters appear in every engine's
  `info()` (`overflows_redone`, `recaptures`, `graph_variant_cap` 64).
- **Item 8 (compiled tier):** the slow tier's `test_compiled_gpu.py` passed locally (MH-1 hybrid and MP-0
  medium, compiled equals graph within float32 noise); the fallback's mode order graph → compiled → eager is
  what the corrected fallback tests check (leeloo and local).

## 9. Swap-MC annealing protocol

The short protocol (`swap_anneal_NaNbOCl4_short`: NaNbOCl4 I4/m 3×3×3, 756 atoms; 0 → 1000 K in 2 ps, then
30 K/ps to 250 K, 27,000 steps at 1 fs; MH-1 `matpes_r2scan` hybrid float32) at the freeze (676a8a3), run as
`fastmace run` in two pieces through 3-T14's `run.sh`:

| Stage | Result |
| --- | --- |
| Relaxation | LBFGS to fmax 0.01 eV/Å |
| MD with swap blocks | O ↔ Cl, every 100 steps, 50 attempts, `exchange`; attached over steps 499–27,000 |
| Move-energy gate | passed in both pieces: worst swap ΔE error 0.00075 and 0.0011 k_B T_min (limit 0.01 at 250 K, 13 configurations); MC energies float32 |
| Restart | paused at step 13,500, `fastmace run --resume` to 27,000 |
| Counters | 13,300 attempted, 0 impossible, 0 accepted; `mc_swap_attempted` and `mc_swap_accepted` in every thermo row |
| Quench | 21 frames, 0 not converged; final frame −10.240258 eV/atom |
| Speed | 9.09 and 10.3 ms/step (MD), 4.35 ms per attempt |

**Pass.** The zero acceptance is recorded, not judged (owner, 22 October 2026: the code is judged; choosing
swap species is another project). 3-T14's two full runs give the full-length behaviour
([3-T14.md](3-T14.md)): local at 2f1cfee and leeloo at ed55026, both every stage passing, 0 of 132,550 swaps
accepted, 21 of 21 frames quenched. Their `fastmace/` differs from the freeze by 3-T06b (rule 9), 3-T22c (the
fixcm default) and 3-T14's YAML input, each with its own tests.

## 10. No regression against main before phase 3

**Step times** (`tools/ab_fastmace.py --release p3b`, tool-side addition; MH-1 hybrid NVT, 300 K, 12 blocks of
300 steps after 300, same process; side 3 pins `fixcm_projection="ase"`, the baseline's projection):

| Process | NaCl 3/p3b | NbCl5 3/p3b |
| --- | ---: | ---: |
| ABAB, p3b built first | 0.9996 | 0.9937 |
| ABAB, 3 built first | 1.0015 | 1.0043 |
| ABBA, p3b built first | 0.9982 | 0.9965 |
| **median** (floor) | **0.9996** (1.1 %) | **0.9965** (0.4 %) |
| one-version controls, 4 blocks, separate processes | 2.871 / 2.821 ms = 1.018 | 21.386 / 21.033 ms = **1.017** |

- All six A/Bs valid: no compile while timing, no patch change, engine options equal. The plan descriptions
  differ only in the new phase 3 fields, each resolving to the baseline's behaviour: `NeighbourConfig.skin`
  0.3 Å and `headroom` fixed (rule 11 sizes only MC and species-change workloads), `recapture_budget` 64,
  `PrecisionConfig.mc_energy` float32 (no MC), and the pinned `fixcm_projection`.
- **Rule as written:** median within ± floor (pass), no process slower than 1 + 2 × floor (pass), controls not
  slower than that. The NbCl5 control misses this last clause: 1.017 against 1.008. The bound was tighter than
  the documented cross-process noise of about 3 % (AGENTS.md), and the same-process A/Bs are the measurement;
  the conductor kept the clause as written and added a phase 4 finding (below).

**Trajectories:**
- **CPU float64, bitwise:** `tools/trajectory_versions.py --against p3b --device cpu` with the tests' tiny
  MH-1-class model (`tests/_support/tiny_models.py`, family `mh1`, Z 11 and 17; unchanged since the baseline)
  on NaCl 2×2×2 (64 atoms), 100 steps, positions every 10 steps, seed 20261022 (conductor, decision 39: size
  the test to the risk). Positions, energies and temperatures **bitwise equal** (largest position difference
  0.0), plans equal. MH-1 itself took over 15 min per 4 CPU steps with Inductor, which the CPU tests' planner
  never uses (`compile_mode="none"`).
- **GPU, reproducible mode:** not run as a separate job (owner's cut). Coraline's `test_reproducible_gpu.py`
  covers the freeze's own reproducibility (bitwise, NaCl and NbCl5, float32 and float64), and the GPU fast tier
  pins the default path; the cross-version GPU comparison is the A/B's observables: temperature within
  1.3e-4 K and potential energy within 1.8e-8 eV/atom between the versions after the timed blocks.

## 11. Carry-overs

- `fastmace.ase` observers apply species, mass and charge changes as events (3-T13): `tests/cpu/test_ase_md.py`
  passes; with `test_fallback_reason.py` (a fallback's reason names the innermost compile error, 3-T17) and the
  contract pins, 119 of 119 at 676a8a3. The GPU fallback record shows it too: "… (AcceleratorError: … ;
  innermost AcceleratorError: CUDA error: operation not permitted when stream is capturing)".
- ResearchModule `main.py` at 1 fs: `langevin/workflow.py` at ResearchModule main has
  `PRODUCTION_TIMESTEP_FS = 1.0` behind `--timestep-fs auto` (3-T16a, merged at 90a1e7b1, with physical output
  cadences, the 1a adapter removed and the Modal recipe); `--profile swap-anneal` is merged at 3cc4207f (3-T16b).
- D1 and D2 are recorded as SPEC decisions 31 and 32 and done in 3-T16a.

## Test corrections at acceptance

Every correction is test-side; `fastmace/` is unchanged. Each test was acceptance-only or slow-only, so no merge
ran it after the change it missed.

| Test | Old expectation | New expectation | Reason | Commit |
| --- | --- | --- | --- | --- |
| `test_fallback_gpu.py`: `test_a_broken_capture_drops_the_graph_mode`, `test_a_capture_error_before_the_first_step_falls_back_to_*` [skip, gate] | tier "eager", reason "fallback: mode eager: " | tier "compiled", reason "fallback: mode compiled: "; the eager diagnosis unchanged | 3-T06's mode order graph → compiled → eager → reference (contracts-3.md) | 208f09a |
| `test_engine_gpu.py`: the three research-engine comparisons | every graph's kernel sequence equal to the research engine's | the integrator graphs AC and ACP left out under `fused_integrator`; every trajectory check kept | the fused integrator (OPT3-T03, default since ab09b51, in the baseline): AC 8 nodes against 64, ACP 8 against 73 | 208f09a |
| `test_engine_gpu.py`: the B graph | B's kernel sequence equal after cuEq's variant normalisation | the named known differences taken out (`known_b_differences`), nothing else | plain cuEq float64 (NVT, NPT): Inductor orders one group of three fused pointwise kernels and two cutlass float64 GEMMs differently, already at 156ab76 (before phase 3); MH-1 hybrid: 3-T02 (f045a3e) replaced the research engine's "first atom of each species" gather (an argmax reduction and an index) with the superset's identity basis (SPEC §8.5), intended | e8e1317 |
| `test_mc_statistics_gpu.py::test_mh1_kinetic_temperature_with_mc_on_and_off[fixcm]` | 3N degrees of freedom; redraw_all a non-strict xfail | 3N − 3 in total and 3n_s − 3M_s/M per species under the plan's `fixcm_projection="mass_weighted"`; 3N otherwise; tolerances unchanged; redraw_all strict | decision 37 (3-T22c): the mass-weighted projection holds the zero-momentum subspace | bdcc312, 08b8bd9 |
| `test_fixed_costs_gpu.py` | the default plan rebuilds on the step's stream (`beside` None) | the default rebuilds beside the step; a new `no_overlap` case keeps the other path; all five cases bitwise equal to the default | OPT-DEF-2 (953aa72, before phase 3) made `overlap_rebuild` the default | 57d924b |
| `test_planner_gpu.py::test_mh1_hybrid_float32_planned_equals_hand_built_and_minimal` | the planned MH-1 hybrid records equal `HYBRID_REWRITES` | `HYBRID_REWRITES` plus the named `ROUND_DEFAULTS`; `same_plan` and the trajectory limits unchanged | the optimisation rounds' defaults, all `"auto"` before phase 3: `gated_rows` and `segment_gemm` (opt-1), `species_tables`, `fold_readout_tail`, `fold_product_linear` and `folded_backward_tiles` (OPT-DEF-2, 953aa72) | da1f05a |
| `test_mc_statistics_gpu.py::test_mh1_first_attempt_after_md` (helper) | its helper builds a float64 reference evaluator with a model | it uses the engine's float64 energy field when there is one | 3-T07's evaluator allows one energy field per precision, and the test's workload (`min_mc_temperature_K=1`) makes the move-energy gate pick float64 MC energies (decision 35), so the engine already holds one | da1f05a |
| `test_mc_statistics_gpu.py::test_mh1_first_attempt_after_md` (baseline) | the block baseline equals a fresh MC-evaluator energy bitwise | within 1e-5 eV (2e-4 kT at 600 K, a tenth of the gate's 0.01 kT) | the test reads "fresh" right after MD and the driver its baseline after quiescing; after an accepted block (a canonical rebuild) the per-atom float32 energies can be summed in another order. Measured: 2.4e-7 and 4.8e-7 eV on −118.6 eV in 2 of 6 forced blocks, every one of them accepted (scratch diagnostic, `results/3-T21-local-20261022/diag-mc.log`). This check first ran at the freeze once the helper was fixed | 2093be8 |

Evidence:
- **Fallback:** leeloo at cd43463 and local at 676a8a3 both activate the compiled tier after eager passes the
  reduced gate; the corrected tests pass locally (targeted run).
- **B graph:** the kernel names of both engines were dumped from instrumented scratch copies at the freeze and at
  156ab76 (`results/3-T21-local-20261022/diag*/results/kinds-*.json`). NVT and NPT differ identically at both
  commits; the hybrid B graph equals the research engine's at 156ab76 and differs at the freeze only by the
  gather. The new comparison passes every dump and fails a negative case (one kernel dropped). The three tests'
  GPU rerun with it at e8e1317: **3 passed** (local, 18:52–19:05 CEST).
- **Kernel sequences:** C equal everywhere; trajectories within their limits (local: cuEq float64 2.5e-13 Å
  against a repeat of 3.9e-13 Å, NPT 7e-15 Å, hybrid 9.7e-8 Å against a research-engine scatter of 8.5e-8 Å).
- **[fixcm]:** leeloo, with 3N: MC off 1881 ± 12.5 K and swap_exchange 1894 ± 17.6 K against 2000 × 45/48 =
  1875 K (16 atoms). Locally, corrected: MC off 2006 ± 13 K; swap_exchange 2021 ± 19 K (0.62 σ),
  swap_redraw_all 1996 ± 12 K (0.58 σ), transmute_redraw_all 2000 ± 13 K (0.35 σ); per species and the Na/K
  split within 1.4 σ. The redraw_all xfail documented ASE's projection and is gone.
- **Fixed costs:** before, all four cases at step 600 with the same energy (−1461.4094561957781 eV) and 161
  rebuilds, but the failing assertion came before the bitwise one. After: **1 passed** locally (57d924b), the
  bitwise positions and momenta comparison running for all five cases, for the first time since OPT-DEF-2.

## Coverage

Every acceptance test (`-m acceptance`, 39) and where it ran. leeloo and coraline at cd43463 (the freeze's
`fastmace/`), with the test files as frozen; local at 208f09a (the partial tier, from its log) and 676a8a3 (the
targeted run of the corrected tests). A "fail" on leeloo is a stale expectation; its corrected version is in the last column.

| Test | leeloo (Ada, sm_89) | coraline (Ampere, sm_86) | local (sm_120), partial tier | local, targeted |
| --- | --- | --- | --- | --- |
| `test_ase_equivalence_gpu::test_langevin_reproduces_ase_in_every_50_K_bin` | pass | – | pass | – |
| `test_ase_equivalence_gpu::test_baoab_variants_reproduce_their_references` | pass | – | pass | – |
| `test_canonical_restart_gpu::test_mh1_hybrid_resumes_statistically` | pass | – | pass | – |
| `test_engine_gpu::test_mh1_plain_cueq_float64_against_the_research_engine` | fail | – | fail | fail |
| `test_engine_gpu::test_mh1_plain_cueq_float64_npt_against_the_research_engine` | fail | – | fail | fail |
| `test_engine_gpu::test_mh1_hybrid_float32_against_the_research_engine` | fail | – | fail | fail |
| `test_fallback_gpu::test_ddmin_finds_the_broken_rewrites[rewrite]` | pass | – | pass | – |
| `test_fallback_gpu::test_ddmin_finds_the_broken_rewrites[pair]` | pass | – | pass | – |
| `test_fallback_gpu::test_a_broken_capture_drops_the_graph_mode` | fail | – | pass | pass |
| `test_fallback_gpu::test_a_capture_error_before_the_first_step_falls_back_to_eager[skip]` | fail | – | pass | pass |
| `test_fallback_gpu::test_a_capture_error_before_the_first_step_falls_back_to_eager[gate]` | fail | – | pass | pass |
| `test_fallback_gpu::test_prepare_leaves_the_default_trajectory_unchanged` | pass | – | pass | – |
| `test_fallback_gpu::test_prepared_graphs_keep_every_edge_after_the_edge_count_grows` | pass | – | pass | – |
| `test_fallback_gpu::test_a_failed_candidates_memory_is_freed_before_the_next_build` | pass | – | pass | – |
| `test_forces_gpu::test_mh1_agrees_with_maceforces[cueq-float64-compiled-production]` | pass | – | pass | – |
| `test_forces_gpu::test_mh1_agrees_with_maceforces[hybrid-float32-compiled-production]` | pass | – | pass | – |
| `test_gate_resources_gpu::test_out_of_memory_ends_on_the_cpu_or_a_reduced_cell[polar1-l]` | pass | – | – | – |
| `test_gate_resources_gpu::test_out_of_memory_ends_on_the_cpu_or_a_reduced_cell[omol0-xl]` | pass | – | – | – |
| `test_isolation_gpu::test_a_b_a_equals_fresh_instances[mh1-plain-cueq]` | pass | – | – | – |
| `test_isolation_gpu::test_a_b_a_equals_fresh_instances[mh1-hybrid]` | pass | – | – | – |
| `test_isolation_gpu::test_a_b_a_equals_fresh_instances[polar1-s-hybrid]` | pass | – | – | – |
| `test_mc_statistics_gpu::test_mh1_populations_follow_boltzmann[swap-20261022]` | pass | – | – | – |
| `test_mc_statistics_gpu::test_mh1_populations_follow_boltzmann[transmute-20261023]` | pass | – | – | – |
| `test_mc_statistics_gpu::test_mh1_kinetic_temperature_with_mc_on_and_off[fixcm]` | fail | – | – | pass |
| `test_mc_statistics_gpu::test_mh1_kinetic_temperature_with_mc_on_and_off[free]` | pass | – | – | – |
| `test_mc_statistics_gpu::test_mh1_kinetic_temperature_with_mc_on_and_off[negative]` | pass | – | – | – |
| `test_planner_gpu::test_mh1_plain_cueq_float64_planned_is_as_fast_as_hand_built` | – | pass | – | – |
| `test_planner_gpu::test_mh1_hybrid_float32_planned_is_as_fast_as_hand_built` | – | pass | – | – |
| `test_selfcheck_gpu::test_gate_passes_in_graph_mode[omol0-xl]` | – | pass | – | – |
| `test_selfcheck_gpu::test_gate_time_on_nbcl5` | – | pass | – | – |
| `test_reproducible_gpu::test_repeated_and_interrupted_runs_are_bitwise_equal[NaCl-float64]` | – | pass | – | – |
| `test_reproducible_gpu::test_repeated_and_interrupted_runs_are_bitwise_equal[NaCl-float32]` | – | pass | – | – |
| `test_reproducible_gpu::test_repeated_and_interrupted_runs_are_bitwise_equal[NbCl5-float64]` | – | pass | – | – |
| `test_reproducible_gpu::test_repeated_and_interrupted_runs_are_bitwise_equal[NbCl5-float32]` | – | pass | – | – |
| `test_reproducible_gpu::test_reproducible_after_cuda_started_runs_in_a_child_bitwise` | – | pass | – | – |
| `test_subprocess_worker_gpu::test_the_worker_equals_fresh_children_bitwise_and_releases_the_gpu` | – | pass | – | – |
| `test_subprocess_worker_gpu::test_a_killed_parent_releases_the_workers_gpu_memory` | – | pass | – | – |
| `test_subprocess_runner_gpu::test_plain_cueq_float64_after_a_hybrid_runs_in_a_child_within_1e_12` | – | pass | – | – |
| `test_subprocess_runner_gpu::test_hybrid_after_a_plain_cueq_runs_in_a_child` | – | pass | – | – |

Totals: leeloo 19 passed, 7 failed (all seven on stale expectations, corrected above); coraline 13 / 13;
the corrected tests locally: fallback 3 / 3, [fixcm] 1 / 1, `test_engine_gpu` 3 / 3 after the B-graph correction (e8e1317).

## Appendix: GPU slow tier

`pytest tests/gpu -m "slow and not acceptance"` on the local RTX 5060 Ti (acceptance tests are also slow and are
not run twice), in two parts: from 676a8a3 until the owner's pause by SIGINT at 17:39 CEST (15:39 UTC), then the
remaining files from 25f6dd3, 19:05–19:58 CEST. Both copies' `fastmace/` equals the freeze's.

| File | Result |
| --- | --- |
| `test_ase_calculator_gpu.py` | 2 / 2 |
| `test_canonical_restart_gpu.py` | 1 / 1 (plain cuEq float64 resume within 1e-12) |
| `test_combo_kernels_cuda.py` | 1 / 1 |
| `test_compiled_gpu.py` | 2 / 2 (item 8) |
| `test_eager_gpu.py` | 2 / 2 |
| `test_engine_gpu.py` (slow, not acceptance) | 2 / 2 |
| `test_fixed_costs_gpu.py` | stale expectation; corrected (57d924b), rerun 1 / 1 |
| `test_forces_gpu.py` | 3 / 3 |
| `test_fused_integrator_gpu.py` | 4 / 4 (second part, whole file) |
| `test_loading_gpu.py` | 21 / 21 |
| `test_mc_displace_gpu.py` | 1 / 1 |
| `test_mc_statistics_gpu.py` (slow part) | 1 / 2: `test_mh1_first_attempt_after_md` stale; corrected (da1f05a, 2093be8), rerun 1 / 1 |
| `test_move_energy_gate_gpu.py` | 1 / 1 |
| `test_planner_gpu.py` (slow part) | 1 / 2: the hybrid three-engine test stale; corrected (da1f05a), rerun 1 / 1 with `same_plan` and the trajectory limits (hand-built 9.0e-8 Å, minimal 8.8e-8 Å, limit 1e-6) |
| `test_polar_gpu.py` | 3 / 3 |
| `test_reference_tier_gpu.py` | 1 / 1 |
| `test_registry_gpu.py` | 4 / 4 |
| `test_rewrites_generic_gpu.py` | 6 / 6 |
| `test_rewrites_mh1_gpu.py` | 3 / 3 |
| `test_selfcheck_gpu.py` (slow part) | 5 / 5 |
| `test_simulation_gpu.py` | 2 / 2 |
| `test_template_cache_gpu.py` | 2 / 2 |

In all: 68 of the 71 slow tests passed as frozen; the three failures were stale test expectations from before phase 3 or
from a contract the test predates, listed under "Test corrections at acceptance", and their corrected versions
pass. Results: `results/3-T21-local-20261022/slow-tier`, `slow-tier-2`, `fixed-costs` and `slow-rerun`.

## Decision 38: rule 9 on this card

`tools/ab_host_callbacks.py`, MH-1 hybrid, one block per mode (owner's cut); compiled over graph, ms/step:

| Callback every | NaCl graph / compiled | ratio | NbCl5 graph / compiled | ratio |
| ---: | --- | ---: | --- | ---: |
| 1 step | 3.73 / 15.80 | 4.24 | 23.06 / 33.97 | 1.47 |
| 5 | 3.01 / 15.94 | 5.30 | 24.55 / 31.71 | 1.29 |
| 10 | 2.94 / 15.22 | 5.18 | 22.23 / 26.78 | 1.21 |

The graph tier stays faster with host callbacks, as decision 38 assumed. The Ada leg of this measurement did not
run (the workstations stopped).

## Open findings

1. **GPU tests don't run at merges** (conductor, for phase 4). Trial merges are CPU-only, so acceptance and slow
   GPU tests went stale: 3-T06's fallback order, the fused-integrator default, decision 37's projection and
   OPT-DEF-2's `overlap_rebuild` default each left tests behind (above). A task that changes behaviour covered
   by GPU tests must run the affected ones.
2. **Cross-process control clauses** (conductor, for phase 4): they should use the documented ≈ 3 % noise floor,
   or be left out where the same-process alternating A/B is the measurement. Item 10's NbCl5 control missed a
   0.8 % bound by 0.9 points.
3. **The Ada leg is partial.** Leeloo ran 26 of the 39 acceptance tests, coraline (Ampere, not Ada) the other
   13; the timing items 1, 2 and 10 and rule 9 ran on the RTX 5060 Ti only. 3-T20's and 3-T14's leeloo runs are
   context, at trees whose `fastmace/` differs from the freeze.
4. **OMOL-0 XL's ΔE precision** is still unmeasured (3-T11: its float64 references did not finish locally or on
   a Modal A100; its move-energy gate is far slower than §12.3's "seconds to a minute").
5. **MH-1 on the CPU with Inductor** takes over 15 min for 4 steps of NaCl 3×3×3 (`compile_mode="default"`, the
   EngineConfig default, on `device="cpu"`). The CPU tests avoid it with `compile_mode="none"`. Whether
   `compile_mode="none"` becomes the CPU default is a question for phase 4: a performance default, so the owner's
   call.
6. Carried from the merged tasks (plans/phase-3.md, "Open findings"):
   - 3-T03: the friction refreshes copy host floats, so they are not capturable (a friction event drops the
     graphs anyway);
   - 3-T08: resume checks total charge and spin, 3-T09 restores them when it registers the total charge;
   - 3-T13: observer charge changes can only be refused today (no model takes per-atom charges); an observer
     attached after the first `run` does not change `host_callbacks_every`; frames after a species change made
     directly through `dyn.simulation` keep the old species until the next observer call (3-T15 owns
     per-frame species).
7. **The local acceptance tier's partial run** (208f09a) lost its junit: stopping the chain's shell ended
   pytest before it wrote it. The log kept the results (ASE equivalence 2, canonical restart 1, fallback 8,
   forces 2 passed; `test_engine_gpu.py` 3 failed on the B graph, corrected above).
8. **The research-engine kernel comparison is brittle by design:** it now names three known differences of the
   B graph. Any later change to the force graph (a rewrite, Inductor's version) will need its own named entry
   or a decision to compare kernel multisets instead.

## Reproducing

From a FastMACE checkout `$FM` with the tag `phase-3-baseline`, in WSL:

```bash
python -m fastmace.bench.versions export --repository $FM --ref phase-3-baseline --name fastmace_p3b --destination ~/fastmace-3-T21/versions
```

```bash
python3 $FM/tools/ci/local_gpu.py run --purpose "phase 3 acceptance" -- bash $FM/tools/acceptance/phase3_local.sh <pinned copy> regression3
```

The other jobs are `events`, `mc-cost`, `rule9`, `targeted`, `slow-tier`, `protocol`, `delta-e` and
`trajectories-cpu` (`tools/acceptance/phase3_local.sh`).
