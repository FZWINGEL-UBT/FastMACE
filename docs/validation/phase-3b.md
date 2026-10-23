# Phase 3b acceptance report

**Status: complete, 23 October 2026.** Plan task 3b-T10, branch `fastmace/3b-T10`. Items 1, 2, 4 and 5 pass (item 5
with one per-process timing clause missed); item 3 misses the 10 % cost target, recorded under decision 42. The runbook and pass rules were written before any result:
[phase-3b-runbook.md](phase-3b-runbook.md).

## Freeze, copies and scope

- **Freeze:** main **ff80ca2**, with 3b-T01 to T09 merged (T05 at a07ff5a, T06 at daacb9f, T07 at 2769b89). No 3b-T10
  commit changes `fastmace/`: every pinned copy's `fastmace/` equals the freeze's.
- **Baseline for item 5:** tag `fastmace-3` (**897fa83**), exported as the renamed package `fastmace_3`.
- **Pinned copies** (`git archive` of the 3b-T10 head, `~/fastmace-3b-T10/src-<sha>`): 074ffa1 (preflight, item 1),
  8c61731 (item 1 table, item 5 NPT processes 1–2), f99add0 (items 3, 4, 5 NVT, item 2's FastMACE side, CPU
  trajectories, acceptance tier, CPU suite), 424b839 (reproducible trajectories), aea4bb9 (item 5 NPT processes
  3–4), cba68dd (item 2's extension). Each later copy differs from the earlier ones only in tool-side files.
- **Scope decisions (owner, 23 October 2026, relayed by the conductor):**
  1. Item 4 at **0.1 GPa**: "Yes, rerun the full D1 protocol at 0.1 GPa (alternative b) for the acceptance item."
     This is D1's alternative (b), chosen after 3b-T07's 0 GPa run vaporised. The 0 GPa results are context.
  2. "Go ahead and follow your new recommendation about where to run what": the 3b GPU slow tier is dropped.
     Phase 4's acceptance runs the slow tier once at its freeze; this report cites the preflight for slow coverage.
  3. The Modal cap for October raised to $29 ("Im fine with upping the cap to 29$"); item 2's ASE seed 104 ran on
     Modal under that override ("Modal please", in the 3b-T10 session).
- **Modal:** $1.63 (Ada spot check) + $1.30 (ASE seed 104) metered; the month stood at $24.21 afterwards.

## Verdict

| # | Item | Result |
| --- | --- | --- |
| 1 | D3 against torch-dftd and simple-dftd3 | **pass** |
| 2 | GPU NPT + D3 against ASE (NbCl5 1×1×3, 300 and 600 K) | **pass**: 600 K on 3 seeds (non-stationary rule); 300 K on 6 seeds after the pre-declared extension (every observable within 0.56 combined SE) |
| 3 | Amortised D3 cost against 10 % | **missed, recorded** (decision 42): 24 % dense NVT, about 30 % on the D1 ramp at 0.1 GPa |
| 4 | NbCl5 NPT melt–quench, end to end | **pass** (0.1 GPa, the owner's alternative (b)); outcome recorded, not judged |
| 5 | No regression against `fastmace-3`, D3 off | **pass**, except one NbCl5 NVT per-process clause (1.0091 against 1.008), recorded as missed; bitwise CPU float64 and reproducible GPU, NVT and NPT: pass |
| – | Tests | preflight 36/36; acceptance tier 20/22 (2 WSL-only failures, below); CPU suite 3376 passed |
| – | Ada spot check (Modal L4, sm_89) | met by the new 3b D3 GPU files, 38 passed; acceptance tier partial (6 tests, 2 B-graph kernel-count failures, open finding 8) |

## Runs

| Run ID | Machine | GPU | Environment | What |
| --- | --- | --- | --- | --- |
| `3b-T10-local-20261023` | PC, WSL2 | RTX 5060 Ti (sm_120), driver 610.88 | `~/environments/mace_cueq_oeq` (hybrid: items 2–5), `mace_cueq_latest` (plain cuEq, pytest, references: torch-dftd 0.5.1, dftd3 1.6.0), `fastmace_cpu` (CPU suite) | everything local, 23 October 2026, 03:49–18:xx UTC, through `tools/ci/local_gpu.py run` (FIFO, shared with phase 4 workers) |
| `3b-T10-ada-20261023` | Modal | L4 (Ada, sm_89), driver 610.57.04 | Modal hybrid image (torch 2.14.1, cuEq 0.12.0, OEq 0.7.0, MACE 0.3.16, torch-dftd 0.5.1; no dftd3) | 03:40–05:05 UTC: the 3b D3 GPU files, then the acceptance tier until its 80 min budget |
| item 2 seed 104 | Modal | L4 | as above | ASE side, 300 K, 16:3x UTC |

Model everywhere: MH-1 `matpes_r2scan` (`mace-mh-1.model`); D3 is `fm.D3("r2scan")` (BJ, s6 1, s8 0.7898,
a1 0.4948, a2 5.7308, 40 bohr pair and CN cutoffs, no ATM, torch-dftd's constants). Records:
[`results/3b-T10-local-20261023/`](../../results/3b-T10-local-20261023/),
[`results/3b-T10-ada-20261023/`](../../results/3b-T10-ada-20261023/).

## Before the runs: stale expectations (preflight)

The slow and acceptance tests of the 13 GPU test files covering BAOAB, the planner, restart, the engine graph and
the reference tier ran first, at the freeze (copy 074ffa1): **36 of 36 passed**, nothing stale. Files:
`test_ase_equivalence_gpu` (2), `test_engine_gpu` (5, the plain-cuEq B graph included), `test_fused_integrator_gpu`
(4), `test_registry_gpu` (4), `test_selfcheck_gpu` (7), `test_planner_gpu` (4), `test_gate_resources_gpu` (2),
`test_canonical_restart_gpu` (2), `test_reproducible_gpu` (5), `test_reference_tier_gpu` (1); `test_events_gpu`,
`test_integrators_gpu` and `test_overflow_commit_gpu` have no slow or acceptance tests. The acceptance tier then left
these files out.

## 1. D3 against torch-dftd and simple-dftd3

- **Tests:** 103 of 103 (`test_d3_list_gpu`, `test_d3_kernel_gpu`, `test_d3_integration_gpu`, `test_npt_d3_gpu`, and the
  CPU reference tests `test_d3_references`, `test_d3_kernel`), `mace_cueq_latest`.
- **Numbers** (`tools/validation/d3_reference_table.py`, tool-side), six rattled and strained fixtures, CPU and CUDA,
  energy per atom / max force component / max stress component:

| Comparison | Worst over fixtures and devices |
| --- | --- |
| torch-dftd against simple-dftd3 (the references' own gap) | 1.8e-9 eV/atom, 4.4e-10 eV/Å, 7.2e-11 eV/Å³ |
| FastMACE float64 against torch-dftd | 4.2e-17 eV/atom, 4.9e-16 eV/Å, 6.1e-18 eV/Å³ |
| FastMACE float64 against simple-dftd3 | equal to the references' gap (within 2× gap + floors) |
| Negative control, pass 3 left out, against torch-dftd (NbCl5, NaNbCl6) | 1.9–3.0e-5 eV/Å, over 1000× the bound (about 1.1e-8 eV/Å) |
| float32 against float64 | 2.2e-9 eV/atom, 1.7e-7 eV/Å, 3.1e-9 eV/Å³, within the gate |

  NaCl's chain term vanishes (its C6 weights sit on one reference, dE/dCN ≈ 0), so its negative control is recorded,
  not judged, as in the tests. The tool's first run judged it and reported a failure; fixed tool-side (8c61731).
- **Capture:** the replay tests are part of the 103 (bitwise equal to the eager call).
- **Ada:** the same GPU files on the L4: 38 passed, 16 skipped (simple-dftd3 is not in the Modal image).

## 2. GPU NPT + D3 against ASE

- Settings as 3b-T05 (the runbook's item 2): NbCl5 1×1×3 (216 atoms), 0 GPa hydrostatic, 1 fs, P_tau 2 ps (test
  setting), mass factor 1, friction 0.02; 5 ps discarded, 20 ps production, a sample every 10 steps.
- **FastMACE side rerun at the freeze** (copy f99add0), seeds 1–3 at 300 and 600 K, 2.5 min each. **ASE side cited**
  from 3b-T05 (seeds 101–103, Modal L4, commit 298585a): its `run-ase` path and the torch-dftd wrapper are unchanged
  up to the freeze.
- **Rules:** the 23 October rules (one common block length of at least 2 τ of the slowest of V, P and PE; the
  non-stationary rule) apply to these runs at the freeze only. They were written on 23 October after 3b-T05's data;
  3b-T05's own numbers are context under the rules it ran under.
- **Result on 3 seeds:**

| T | Observable | Difference | Combined SE |
| --- | --- | ---: | ---: |
| 300 K | V (Å³/atom) | −0.0457 | 1.97 |
| 300 K | P (GPa) | +0.00286 | 1.12 |
| 300 K | PE (eV/atom) | −0.00077 | **3.45** |
| 300 K | V fluctuation | +0.0061 | 0.24 |
| 600 K (non-stationary) | V drift | +0.30 | 1.62 |
| 600 K | V in windows 1–4 | −3.53 / −3.45 / −1.75 / +1.13 | 2.93 / 1.73 / 0.82 / 0.47 |

- **Pre-declared extension (300 K):** 3 more seeds per engine; the 6-seed comparison decides. FastMACE seeds 4–6
  (local), ASE seeds 104 (Modal L4) and 105–106 (local, the RTX 5060 Ti; `run-ase` with MACE cuEq float32 and
  torch-dftd float64, as 3b-T05). The ASE side therefore spans two GPUs (L4 for 101–104, RTX 5060 Ti for 105–106);
  both run the same float32 cuEq MACE and float64 torch-dftd.
- **Result on 6 seeds per engine at 300 K: pass.** Blocks of 99 samples (2 τ of PE, τ = 49 samples; τ(V) = 39,
  τ(P) = 5), 120 blocks per engine:

| Observable | FastMACE | ASE | Difference | Combined SE |
| --- | ---: | ---: | ---: | ---: |
| V (Å³/atom) | 28.9992 | 29.0002 | −0.00095 | 0.06 |
| P (GPa) | −0.00140 | −0.00079 | −0.00061 | 0.32 |
| PE (eV/atom) | −10.67746 | −10.67737 | −0.00009 | 0.56 |
| V fluctuation (Å³/atom) | 0.3064 | 0.3021 | +0.0043 | 0.26 |
| T (K), context | 298.5 | 299.3 | | |

  Both engines are stationary at 300 K (ASE's mean V drift 0.0003 ± 0.0018 Å³/atom/ps). The 3-seed PE miss
  (3.45 SE) does not persist: the PE difference fell from −0.00077 to −0.00009 eV/atom with 6 seeds, as 3b-T05's
  diagnosis (a seed-set temperature difference, paired runs equivalent) expected. 600 K stays on its 3-seed
  verdict (pass).

## 3. Amortised D3 cost

`tools/bench/d3_cost.py ab` (3b-T06's tool), NbCl5 3×3×3, MH-1 hybrid float32, D3 on against off in one process,
ABBA blocks of 1000 steps with every rebuild in the clock. One process per route: the effect (≥ 2× the target) is far
outside the 1.07–1.13 band that would have called for more.

| Route | Off ms | On ms | Overhead |
| --- | ---: | ---: | ---: |
| NVT heat 0 → 1000 K (5 ps) | 21.02 | 25.75 | 22.5 % |
| **NVT 1000 K, dense** | 21.21 | 26.30 | **24.0 %** |
| NPT 0.1 GPa heat (20 ps) | 20.38 | 27.17 | 33.3 % |
| NPT 0.1 GPa 1000 K | 18.39 | 23.30 | 26.7 % |
| NPT 0.1 GPa 600 K | 18.92 | 24.66 | 30.4 % |
| NPT 0.1 GPa 300 K | 19.76 | 26.10 | 32.1 % |
| NPT 0 GPa heat / 1000 / 600 / 300 K | 19.30 / 16.38 / 16.20 / 16.05 | 26.37 / 19.92 / 17.92 / 17.40 | 36.7 / 21.6 / 10.6 / 8.4 % |

- **Amortised over the D1 ramp** (20 ps heating, then 3 K/ps to 250 K, cooling interpolated between the windows):
  **about 30 % at 0.1 GPa** (condensed), about 15 % at 0 GPa only because the box vaporises (to 3.7 × 10⁸ Å³) and the
  D3 pairs vanish.
- **The 10 % target is missed** (24 % dense, about 30 % on the production ramp). Recorded, and the phase continues
  (decision 42). 3b-T06 attributes the cost to the three passes (about 4.2 ms per step at 1.8 M pairs), not the
  builds; the follow-up is optimisation.
- Item 4's run records the builds along the real protocol: 1751, 1588 and 1141 D3 builds in its three 90,000-step
  pieces, 19, 18 and 16 s of build time (under 1 % of each piece's 2,200–2,370 s).

## 4. The NbCl5 NPT melt–quench protocol

- **Config:** `npt_anneal_NbCl5.toml` as shipped, with only `pressure_GPa = 0.1` (and the structure path made
  absolute) in a copy; no protocol file changed. Run through `fastmace run` in three pieces (`results/3-T14-protocol/run.sh`):
  start (relaxation and steps 0–90,000), then two resumes from the restart files (to 180,000 and 270,000, then the
  quench). Copy f99add0, 07:26–12:25 UTC (09:26–14:25 CEST).
- **Pass:** status `complete`. Every stage ran and is recorded:
  - the fixed-cell LBFGS relaxation with D3;
  - 270,000 NPT + D3 steps along the anneal schedule, resumed twice across canonical restart points;
  - the quench of 21 frames at their own NPT cells, cell fixed;
  - 27,001 thermo rows (volume, pressure, temperature, energies), none non-finite;
  - provenance: the D3 record, the D3 skin 2.07 Å (rule 11), externalstress −0.1 GPa, barostat mass factor 1 with
    the default tau, D3 build counters per piece.
- **Outcome, recorded and not judged (decisions 6 and 40):** at 0.1 GPa the system stays condensed.

| Step | T (K) | V (Å³/atom) | P (GPa) | PE (eV/atom) |
| ---: | ---: | ---: | ---: | ---: |
| 0 (relaxed) | 0 | 28.76 | −0.40 | −10.716 |
| 10,000 | 473 | 29.45 | 0.17 | −10.653 |
| 20,000 (1000 K reached) | 944 | 49.37 | −0.03 | −10.542 |
| 120,000 | 724 | 44.11 | −0.01 | −10.568 |
| 200,000 | 461 | 37.03 | 0.13 | −10.610 |
| 270,000 | 257 | 32.14 | 0.17 | −10.654 |

  The crystal melts on the plateau (about 50 Å³/atom, 1.7 × V0) and densifies on cooling to 32.1 Å³/atom, 12 % above
  the relaxed crystal. Quenched energies: heating ramp −10.697 ± 0.017 eV/atom, cooling ramp −10.667 ± 0.003,
  final frame −10.687. 10 of the 21 quench relaxations stopped at the protocol's 5,000-step limit before fmax
  0.01 eV/Å: the two heating frames at 925 and 975 K and 8 of the 10 cooling-ramp frames (the liquid and the
  glass quenched from it; 825 and 575 K converged at 4,831 and 4,930 steps). All heating frames up to 870 K and the
  final frame converged. Recorded.
- **Context, 0 GPa (3b-T07):** the same protocol vaporised; the cell expanded without bound on the cooling ramp,
  and the run stopped cleanly with `CellDivergedError`.

## 5. No regression against `fastmace-3`, D3 off

**NVT step times** (`tools/ab_fastmace.py --release 3`, tool-side; MH-1 hybrid NVT, 300 K, 12 blocks of 300 steps
after 300, same process; all six A/Bs valid, plans equal):

| Process | NaCl 3b/3 | NbCl5 3b/3 |
| --- | ---: | ---: |
| ABAB, 3 built first | 0.9992 | 0.9992 |
| ABAB, 3b built first | 0.9962 | **1.0091** |
| ABBA, 3 built first | 1.0058 | 0.9958 |
| **median** (floor) | **0.9992** (1.1 %) | **0.9992** (0.4 %) |

- **Rule as written:** median within ± floor: pass for both. No process slower than 1 + 2 × floor: NaCl pass
  (1.0058 against 1.022); **NbCl5 missed** (1.0091 against 1.008), by 0.11 points. In that process the 3 side ran
  unusually fast (20.51 ms against 20.70–21.00 ms in the other processes); the medians are equal.

**NPT step time** (NbCl5, one version per process, order 3, 3b, 3b, 3; processes 3 and 4 rerun after a session stop
ended process 3, below): 3 = 20.785 and 20.837 ms, 3b = 20.780 and 20.929 ms; median ratio **1.002**, within the
3 % cross-process bound: **pass**.

**Trajectories, bitwise** (`tools/trajectory_versions.py --against 3`, tool-side `--ensemble npt`): CPU float64 (the
tests' tiny MH-1-class model, NaCl 2×2×2, 100 steps, positions every 10) and the reproducible plan on the GPU (plain
cuEq float64, MH-1 NaCl, 500 steps, every 50), NVT and NPT: positions, energy, temperature, and for NPT cell and
stress **bitwise equal**, plans equal: **pass**.

## Tests

| Tier | Where | Result |
| --- | --- | --- |
| Preflight (slow and acceptance of 13 files) | local, 074ffa1 | 36 / 36 |
| D3 files and CPU references (item 1) | local, 074ffa1 | 103 / 103 |
| Acceptance tier, minus the preflight files | local, f99add0 | 20 / 22 |
| CPU suite, `-m "not acceptance"` | local, f99add0, `fastmace_cpu` | 3376 passed, 243 skipped |
| `test_mc_enumeration.py -m acceptance` (CPU) | local | at the freeze's test: 5 passed, 3 failed (stale); with a1edf4b's test: 8 / 8 |
| Slow tier | – | dropped (owner); phase 4's acceptance runs it |
| Ada: D3 files, acceptance tier | Modal L4 | 38 passed, 16 skipped; acceptance 4 passed, 2 failed of 6 reached |

- **Acceptance tier, local:** `test_fallback_gpu` 8, `test_forces_gpu` 2, `test_isolation_gpu` 3,
  `test_mc_statistics_gpu` 5, `test_subprocess_runner_gpu` 2: passed. `test_subprocess_worker_gpu` 2: **failed, a
  WSL2 limitation.** Both fail only at `_listed(pid)`, which asks `nvidia-smi --query-compute-apps` for the worker's
  PID. Under WSL2 nvidia-smi reports PIDs outside the WSL namespace (the PID it listed during the check is not a WSL
  process), so a WSL PID never matches. The checks before it passed: the persistent worker's trajectory equals fresh
  children's bitwise, and the GPU memory after close changed by +0 MiB. In phase 3 both passed on the workstations.
- **Stale CPU expectation:** `test_kinetic_temperature_with_mc_on_and_off`'s three NaK `fixcm` variants divided by
  3N where decision 37's projection leaves 3N − 3; fixed test-side in a1edf4b (3-T12-dof, 30e00a2). Not a 3b
  failure.

## Ada spot check

- Met by the new 3b D3 GPU files on the L4 (38 passed, 16 skipped), as the conductor recorded.
- The acceptance tier was partial: 6 tests in its 80 min budget, stopped by SIGINT with junit and the interrupted
  test kept. Passed: `test_langevin_reproduces_ase_in_every_50_K_bin`, `test_baoab_variants_reproduce_their_references`,
  `test_mh1_hybrid_resumes_statistically`, `test_mh1_hybrid_float32_against_the_research_engine`. Failed:
  `test_mh1_plain_cueq_float64_against_the_research_engine` and its NPT twin, only on the B graph's kernel count
  (237 against 238 nodes, NPT 294 against 295). Every trajectory check passed within 1e-12 Å, and both pass on the
  RTX 5060 Ti. Recorded as an sm_89-specific cuEq variant under phase 3's open finding 8 (the known-difference list
  was validated on sm_120 only); no kernel dump.

## Open findings

1. **D3 cost** (item 3): 24–30 % against 10 %. Follow-up: kernel optimisation (3b-T06, "Why it misses").
2. **Item 2 at 300 K:** passed only after the pre-declared extension; the 3-seed miss was seed noise in PE.
3. **Item 5's NbCl5 per-process clause** missed by 0.11 points in one same-process A/B; the medians are equal.
4. **Ada:** the full acceptance tier on an Ada workstation (murphy or leto) once the workstations are back. It also
   covers the two WSL-limited subprocess-worker tests.
5. **`test_subprocess_worker_gpu` under WSL2:** `_listed` cannot see WSL PIDs. A phase 4 test-side candidate (skip
   the listing check under WSL, or check by GPU memory).
6. **Tooling, found in this acceptance:**
   - `trajectory_versions.py --reproducible` planned after `planner.hardware()` had started CUDA, and the
     reproducible plan refused it (SPEC §8.7); this path had never run. Fixed tool-side (424b839).
   - A local GPU booking that waits longer than `--wait-minutes` (default 60) drops its job: item 5's first NVT A/B
     was skipped that way. Every booking now passes `--wait-minutes 600` (conductor's rule).
   - Stopping a chain's `wsl -e` command ends every process of that WSL session: item 5's NPT process 3 died with it.
     Jobs to keep must run in their own session.
   - The budget guard needs `--owner-override` above $27, even after the owner raised the cap.
7. **Quench convergence:** 10 of 21 quench relaxations hit the protocol's 5,000-step limit (melt frames).

## Reproducing

From a FastMACE checkout `$FM` with the tag `fastmace-3`, in WSL:

```bash
python -m fastmace.bench.versions export --repository $FM --ref fastmace-3 --name fastmace_3 --destination ~/fastmace-3b-T10/versions
```

```bash
FASTMACE_3_COMMIT=897fa83 python3 $FM/tools/ci/local_gpu.py run --wait-minutes 600 --purpose "3b-T10 regression" -- bash $FM/tools/acceptance/phase3b_local.sh <pinned copy> regression
```

The other jobs are `preflight`, `d3-references`, `d3-table`, `d3-cost` (`D3_COST_PRESSURE_GPA`, `D3_COST_NVT`),
`npt-compare` (`SEEDS`, `TEMPERATURES`), `npt-ase`, `regression-npt` (`NPT_PROCESSES`), `protocol-start` and
`protocol-resume` (`PROTOCOL_PRESSURE_GPA`, `PIECE_STEPS`), `trajectories-cpu`, `trajectories-reproducible` and
`acceptance-tier`. The Ada check is `tools/modal/3b_t10_ada.sh`.
