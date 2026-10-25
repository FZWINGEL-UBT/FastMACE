# Runbook: phase 5 acceptance

Plan task 5-T11, written by the conductor on 25 October 2026 (owner: "After all current tasks go ahead and do the
acceptance of Phase 5"; "just continue as you see fit"). The outline it follows is `plans/phase-5.md`
("Acceptance runbook outline (5-T11)"), as amended on 24 October 2026 for optimisation round 4.

## Freeze and baseline

- **Freeze:** main at the commit named in the report (`docs/validation/phase-5.md`), with every phase 5 task,
  optimisation round 4 and the pre-acceptance fixes merged. Runs use a `git -c core.autocrlf=false archive` of it
  with `.source_commit`.
- **Baseline:** the tag `fastmace-4` (`dae3848`), archived the same way.
- **Hardware:** the local RTX 5060 Ti (sm_120) in WSL2 for everything judged; the Colab L4 (sm_89) for the
  compatibility tier. The A100 is not a production type (owner, 24 October 2026) and has no judged item. Colab
  budget: about 20 CU.
- **Environments:** `~/environments/mace_cueq_oeq` (hybrid) and `~/environments/mace_cueq_latest` (plain cuEq,
  CPU), as recorded in each result.

## Items and pass rules (fixed before the runs)

| # | Item | How | Pass |
| --- | --- | --- | --- |
| 1 | Conformance (§16) | `pytest -m rule` and the rules meta-test (`tests/cpu/test_rules_*.py`, `test_rule_cases*`) | every rule 1–15 has a test, every precedence pair of 5-T02's matrix has one, all pass |
| 2 | CPU suite | `pytest -n 8 tests/cpu` with `-m acceptance` included, `FASTMACE_BASELINE` set | no failure |
| 3 | Plan corpus | `tools/plan_corpus.py run` on the baseline and the freeze trees, then `compare` | no unexplained difference: only decisions 57–59 on their cases, rule 12/15 decisions, and fields new in phase 5 |
| 4 | Bitwise trajectories | `tools/trajectory_versions.py run --against 4 --side 4\|5`, CPU float64 and reproducible GPU, then `compare` | bitwise equal, as at 4-T13 (decisions 57–59 are off in float64 and reproducible plans or exact) |
| 5 | Timed regression | `tools/ab_fastmace.py --release 4`, MH-1 hybrid Langevin NVT, NaCl and NbCl5, quiet PC, alternating blocks | the freeze is not slower than the baseline beyond the A/B's 95 % interval; a gain is recorded |
| 6 | Adaptation end to end | 5-T07's GPU tests (storm, switch at a canonical point, report line, `counters.tier_switches`, resume) | pass; the production anneal profile never switches |
| 7 | Mid-run fallback | 5-T08's GPU test (broken capture ends on a validated revision from the committed point, no duplicate output) | pass |
| 8 | Rule 12 | the dated rerun section of `docs/validation/5-T03.md` (24 October 2026) | cited; POLAR-1-L NbCl5 NVT at 0.96 is the owner-accepted known limitation (25 October 2026) |
| 9 | Calibration | 5-T09's NaCl smoke, its record and setup cost | runs; its choice judged on the reference GPU only |
| 10 | `str(plan)` | the golden texts (in item 2) and one real plan per item 6–9 in the report | printed |
| T | Full tiers | `-m acceptance` and `-m "slow and not acceptance"` on the RTX 5060 Ti, after the stale-test listing against `fastmace-4` | no failure, or each failure triaged as stale expectation (corrected) or defect (fixed, rerun) |
| C | Compatibility | `-m compat` on local Blackwell and the Colab L4 (`tools/acceptance/compat.py`) | pass |

Recorded, not judged: calibration's step times off the reference GPU; the re-capture time fractions of the
production profiles; the FastMACE against TorchSim comparison of 25 October 2026.

## Order

1. CPU items 1–3 (no GPU).
2. Preflight: the GPU tests whose code changed since `fastmace-4`, untimed.
3. The timed stretch alone on a quiet PC: item 5, then 6's timed parts and 9.
4. Items 4, 6, 7 and the tiers, untimed, in FIFO reservations.
5. The compatibility tier on the Colab L4 in parallel from the freeze.

## Tool-side additions (on `fastmace/5-T11-tools`)

- `tools/plan_corpus.py` (new): item 3.
- `tools/ab_fastmace.py` and `tools/trajectory_versions.py`: release `4` (`fastmace-4`, `dae3848`), sides `4`
  and `5`.

## Recovery and completion (25 October 2026)

The first local sequence finished, but its outer `sequence.log` masked child failures. Use the pytest
JUnit records and `jobs.log`, not the outer `DONE` marker. The runner now stops on a required child failure.
The original records remain under `~/fastmace-acceptance/5-T11-local-20261025`; recovery jobs use separate
directories, so they cannot overwrite the failed evidence.

- Rerun the three extended-energy cases with `9e9996d2`'s fresh-child test correction.
- Rerun each of the six slow-tier failures in a separate process. The first failure was a Triton CUDA OOM
  in the POLAR stress reference; later failures saw the same unusable CUDA context. Classify the failure
  from these reruns, without weakening numerical limits.
- Repeat all three local alternating timing jobs. The tool must receive
  `--allow-option-changes grouped_launch fx_e3nn_linear polar_graph_sums` for decisions 57–58; all other
  validity checks remain enabled. A zero process exit alone is insufficient: every timing record must be valid.
- Complete `adapt-storm` (2,000 steps per side) and judge the switched side against the unswitched side,
  including the recorded switch and the 3% cross-process noise floor used by the worker's runbook.
- Complete the worker's `adapt-production` jobs: the shipped short NbCl5 NPT and NaNbOCl4 swap profiles,
  plus the NaCl NVT anneal. Each runs in two pieces with a resume. Preserve each piece's root MD metadata
  before continuing; check all six records for zero tier switches. These are the named acceptance profiles,
  not a claim about every material or longer annealing protocol.
- The compatibility tier includes the rule-6 fused-cuEq assertion, rule-12 MH-1 memory peak check, and
  forced re-capture switch. If a tier started before the test-only additions, run the added tests separately
  on the same GPU type and record both source commits.
- Run the calibration check during the quiet local window, collect real plan records for items 6–9,
  and write `phase-5.md` before the release decision.

Production source remains frozen: the `fastmace/` tree at `533b0a6e` and `9e9996d2` is
`95d8558071b236d6fcc563c4cfa2d4e1547d7f62`. Recovery changes are acceptance tools, tests and documentation.
Any production defect requiring a code change must be listed with its affected reruns.
