# Phase 5 acceptance report

**Status: all items pass, 25 October 2026.** Plan task 5-T11, branch `fastmace/5-T11-tools`. Runbook and pass rules,
fixed before the runs: [phase-5-runbook.md](phase-5-runbook.md). The release decision (merge, tag `fastmace-5`)
is the owner's.

## Freeze, copies and scope

- **Freeze:** main **533b0a6e** (every phase 5 task, optimisation round 4, decisions 57–59 and the pre-acceptance
  fixes). Its `fastmace/` tree is `95d8558071b236d6fcc563c4cfa2d4e1547d7f62`.
- **Copies used by the runs:** 8b786e06 (first sequence), 9e9996d2 (slow tier, energy and slow reruns, Colab L4),
  a2077787 (recovery: timing, adaptation, calibration, local compat). All three have the same `fastmace/` tree as
  533b0a6e. Between 533b0a6e and a2077787 only tests, acceptance tools and the runbook changed (11 files).
  **No production code changed during the acceptance.**
- **Baseline:** tag `fastmace-4` (`dae3848`).
- **Hardware:** RTX 5060 Ti (sm_120, 16 GiB, driver 610.88) in WSL2 for every judged item. Colab L4 (sm_89,
  driver 580.82.07) for compatibility. No A100 item (owner, 24 October 2026).
- **Environments:** `~/environments/mace_cueq_oeq` (hybrid) and `~/environments/mace_cueq_latest` (plain cuEq,
  CPU). Colab: torch 2.14.1+cu130, cuEq 0.12.0, OEq 0.7.0, MACE 0.3.16, ASE 3.27.0
  (`env_matches_manifest: true`).

## Verdict

| # | Item | Result |
| --- | --- | --- |
| 1 | Conformance (§16) | **pass**: rule and meta-tests are part of the CPU suite (item 2), no failure |
| 2 | CPU suite | **pass**: 4,651 passed, 12 skipped, 0 failed (`cpu-full.exit` 0) |
| 3 | Plan corpus | **pass**: 228 baseline / 228 freeze cases; only decision 57 (84), decision 59 (210), new phase 5 decisions and rules 12/15 (222), and fields new in phase 5 (666) differ |
| 4 | Bitwise trajectories | **pass**: CPU float64 NVT/NPT with and without D3(BJ), and reproducible GPU NVT/NPT, all `equal: true` against `fastmace-4` |
| 5 | Timed regression | **pass** (recovery run): freeze/baseline ratios NaCl 0.9987 and 1.0021, NbCl5 1.0031 and 0.9940, every record valid. No gain and no regression (round 4's gains do not reach MH-1 NaCl/NbCl5 under decisions 57–58) |
| 6 | Adaptation end to end | **pass**: storm switches graph → compiled once, at step 1 (27.1 ms/step against 219.2 ms/step unswitched, 2,000 steps). Production profiles (NbCl5 NPT short, NaNbOCl4 swap short, NaCl NVT anneal), each started and resumed: **0 tier switches in all six records**, re-capture fraction 0.02–0.65 % |
| 7 | Mid-run fallback | **pass** (preflight, 5-T08 test): broken capture at step 51 falls back from the restart point at step 50 to plan revision 1 (compiled) |
| 8 | Rule 12 | cited: the 24 October 2026 rerun in [5-T03.md](5-T03.md). POLAR-1-L NbCl5 NVT at 0.96 is the owner-accepted known limitation |
| 9 | Calibration | **pass**: NaCl smoke keeps hybrid (3.95/4.01 ms against cueq 7.25/7.24 ms), record setup 99.6 s |
| 10 | `str(plan)` | golden texts pass in item 2; real plan below |
| T | Full tiers | **pass after triage**: acceptance tier 52 + 3, slow tier 69 + 6 (below) |
| C | Compatibility | **pass**: RTX 5060 Ti 365 passed, 3 skipped (27.9 min); Colab L4 365 passed, 2 skipped (82.7 min, 2.56 CU) |

## Tier failures and triage

The first sequence's outer `sequence.log` said `DONE` while two tiers had failed. The runner now stops on a required
child failure (a2077787), and the JUnit records are the evidence.

- **Acceptance tier, 3 failures,** `test_extended_energy_over_20_ps[nhc|hydrostatic|full]`:
  `ProcessConstraintError`, the allocator was configured after CUDA was already in use (order dependence, phase 4
  open finding 3). **Stale test.** 9e9996d2 runs the case in a fresh child. Rerun: 3 / 3 pass (8 min).
- **Slow tier, 6 failures:** the first was a Triton CUDA out of memory in the POLAR-1-S NbCl5 stress reference. The
  other five then hit a sticky out-of-memory error during CUDA initialisation, so that process had no usable GPU left.
  **Resource exhaustion in one long process, not a defect.** Each test was rerun alone: 6 / 6 pass, with
  unchanged limits (generic rewrites float64 errors ≤ 1.2e-15).

## Runs

| Run ID | GPU | What |
| --- | --- | --- |
| `5-T11-local-20261025` | RTX 5060 Ti | first sequence: items 1–4, preflight, both tiers (original failures preserved) |
| `5-T11-local-20261025-recovery-energy`, `-slow-0`…`-slow-5` | RTX 5060 Ti | tier reruns |
| `5-T11-local-20261025-recovery-regression` | RTX 5060 Ti | item 5 (alternating blocks, `--allow-option-changes grouped_launch fx_e3nn_linear polar_graph_sums`) |
| `5-T11-local-20261025-recovery-adapt-storm`, `-adapt-production` | RTX 5060 Ti | item 6 |
| `5-T11-local-20261025-recovery-calibration`, `-compat` | RTX 5060 Ti | item 9, local compat |
| `5-T11-colab-l4-recovery-20261025` (`fm-l4-20261025-101853-76fb5a`) | Colab L4 | compat; CU 34.67 → 32.11, VM stopped |

Model: MH-1 (`mace-mh-1.model`) unless a test names another. Records (no trajectories, restart files or models) are in
`results/5-T11-20261025/`. `original/` holds the first sequence; `recovery-*` and `colab-l4-recovery` the rest.

## A real plan (item 10)

The NaCl NVT anneal profile (item 6), plan revision 0. Abridged; the full text is in
`results/5-T11-20261025/recovery-adapt-production/adapt-production/nvt_anneal_NaCl/metadata.json`:

```text
Plan revision 0, tier graph: minimal plan (phase 1b)
  EngineConfig.backend = hybrid  (requested 'hybrid')
  PrecisionConfig.dtype = float32  (requested 'float32': hybrid backend)
  RewriteConfig.grouped_launch = off  (... stage 2: off, not the generic MACE family, decision 57 ...)
  RewriteConfig.fx_e3nn_linear = off  (... not a qualified POLAR-1 checkpoint, decision 58 ...)
  IntegratorConfig.fixcm_projection = mass_weighted  (... decision 37)
  EngineConfig.dynamic_edges = on  (... the model compiles once ... (decision 59))
  memory = fits  (rule 12: predicted 1.53 GiB ... needs 1.53 of 14.80 GiB free)
```

## Open findings (not blocking)

1. The runner's outer `DONE` marker masked child failures (fixed in tools, a2077787).
2. The slow tier runs out of GPU memory in one process on 16 GiB. Run POLAR stress references in a child process,
   or split the tier.
3. From the pre-acceptance list, deferred: the integrator label in plan records (4-T13), extended-energy
   `run_fresh`, the slow POLAR cold compile, and the NAS sync (until the workstations are back).
4. OMOL-0 XL on the A100: still open. The A100 is not a production GPU.
