# Runbook: phase 3 acceptance

Plan task 3-T21, written 22 October 2026 at about 12:35 UTC, before any timed result came in. Results go to
[phase-3.md](phase-3.md) and `results/`. 3-T21 changes no library code; its tool changes are labelled
tool-side.

## Freeze and copies

- Freeze: main **c3bf93f**, every phase 3 task merged except 3-T16b (ResearchModule) and 3-T18 (coraline).
  Its `fastmace/` equals **cd43463**'s (`git diff cd43463 c3bf93f -- fastmace` is empty).
- Baseline for item 10: **B** = main **156ab76**, just before the 3-T01 merge, tag `phase-3-baseline`
  (conductor's answer 1), exported with `python -m fastmace.bench.versions export --ref phase-3-baseline
  --name fastmace_p3b` to `~/fastmace-3-T21/versions`.
- Local copy: `git archive` of the 3-T21 head 1d76cc1 to `~/fastmace-3-T21/src-1d76cc1`. Its `fastmace/`
  equals c3bf93f's; the job script runs from the worktree and imports that copy.
- leeloo copy: `git archive cd43463` to `/mnt/homes/zwingel/fastmace-t21-cd43463`.

## Scope (owner, 22 October 2026, after the leeloo go)

- **Ada:** leeloo's (RTX 4000 Ada) GPU acceptance tier at cd43463 is the phase's one Ada spot check, until
  15:58 UTC (the workstations stop at 16:00 for Friday's power outage). Nothing runs on Modal. Acceptance
  tests leeloo didn't reach are listed; the RTX 5060 Ti covers them.
- **Item 9:** the short protocol (`swap_anneal_NaNbOCl4_short`, 27,000 steps, every stage) at the freeze,
  locally. 3-T14's two full runs are cited for the full length (local 2f1cfee, leeloo ed55026).
- **Timing repeats:** item 1 the 4 main swap-back runs (no superset rows); item 2 and rule 9 one block per
  case; item 10 unchanged.

## Jobs

Local RTX 5060 Ti, run ID `3-T21-local-20261022`, every job through `tools/ci/local_gpu.py run` and
`tools/acceptance/phase3_local.sh <copy> <job>`; outputs in `~/fastmace-acceptance/3-T21-local-20261022/`.

| Job | Items | Timed | What runs |
| --- | --- | --- | --- |
| `regression3` | 10 | yes | `tools/ab_fastmace.py --release p3b`: MH-1 NaCl and NbCl5 hybrid NVT, 300 K, 12 blocks of 300 steps after 300; ABAB p3b first, ABAB 3 first, ABBA p3b first; then one 4-block control per side |
| `events` | 1 | yes | `tools/bench/event_speed.py --swap-back --settle 5`, 1500 K, 1 fs, events every 100 steps; NaCl and NbCl5, build orders ab (ABAB) and ba (ABBA) |
| `mc-cost` | 2 | yes | `tools/bench/mc_cost.py`, 3-T20's cases at 1500 K, one round each |
| `rule9` | decision 38 | yes | `tools/ab_host_callbacks.py`, NaCl and NbCl5, intervals 1, 5 and 10, one block per mode |
| `acceptance-tier` | 3–8, 11 | (step-time assertions) | `pytest tests/gpu -m acceptance` |
| `slow-tier` | 4–8 | no | `pytest tests/gpu -m "slow and not acceptance"` (acceptance tests are also slow; not run twice) |
| `protocol` | 9 | no | the short protocol: start 13,500 steps, resume 13,500 (relaxation, swap blocks, gate, restart, quench) |
| `delta-e` | 3 | no | `tools/validation/move_energy_precision.py`, MH-1 `matpes_r2scan` and `omol`, NaCl, NbCl5, NaNbO3 |
| trajectories | 10 | no | `tools/trajectory_versions.py --against p3b`: CPU float64 (`--device cpu --backend auto`) and the reproducible plan on the GPU (`--reproducible --backend cueq`), positions recorded; `compare --bitwise` |

leeloo, run ID `3-T21-leeloo-20261022`: `pytest tests/gpu -m acceptance` at cd43463, NAS `mace_cueq_latest`,
every cache under `/mnt/homes/zwingel/fastmace-cache/3-T21-leeloo`, launched from this session as a
background task, after the GPU-empty and no-other-user checks, under `timeout` to 15:58 UTC.

## Pass rules (written before the runs)

- **Item 1:** in every run, events/fixed step rate ≥ 0.90 (the tool's "rate fixed/events" line, inverted
  where needed): NaCl and NbCl5, both build orders, on the RTX 5060 Ti. The Ada row is 3-T20's leeloo series,
  reported as context because its `fastmace/` (4a1dfa8) differs from the freeze.
- **Item 2:** a measurement, no pass mark (§10.4 "Cost"). One round per case, so the numbers carry about one
  noise floor more spread than 3-T20's eight.
- **Item 10, step times:** per structure, the median of the three A/B processes' ratio 3/p3b within ± the
  local floor (1.1 % NaCl, 0.4 % NbCl5), no process slower than 1 + 2 × floor, and the one-version controls
  not slower than that either. Plans: the only differences allowed are the new phase 3 fields and their
  decisions, which resolve to the baseline's values (conductor's answer 1); any other engine-option difference
  makes the process invalid. Side 3 pins `fixcm_projection="ase"` (decision 37).
- **Item 10, trajectories:** bitwise on CPU float64 and in the reproducible plan: every recorded row's
  positions, energy and temperature equal (`compare --bitwise`).
- **Items 3–8, 11 (tests):** every acceptance and slow GPU test passes on the RTX 5060 Ti; the leeloo tests
  that ran pass on sm_89. A failure is first reproduced with the production path (`Simulation`, `fastmace run`)
  before it is reported as a library defect.
- **Item 3:** the float32 ΔE error ≤ 0.01 k_B·T_min for every case of MH-1's two heads; OMOL-0 XL is 3-T11's
  open finding (its float64 references didn't finish), not rerun.
- **Item 9:** every stage of the short protocol completes at the freeze: relaxation, MD with swap blocks, the
  move-energy gate passing, a restart and resume, the quench, and the attempt and acceptance counters in the
  output. The acceptance rate is recorded, not judged (owner, 22 October 2026).
