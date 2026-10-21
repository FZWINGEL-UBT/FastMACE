# Runbook: combined acceptance of phase 2 and optimisation round 1

Plan task 2-T23, written 20 October 2026, about 23:00 UTC, before the freeze. The owner's decision of the
same day: phase 2's accuracy acceptance runs once, combined with optimisation round 1's
(`plans/opt-1.md`), on the optimised main. Results go to [phase-2.md](phase-2.md) and `results/`. If
anything fails, the error and run IDs are recorded and the rest goes on. 2-T23 changes no library code.

## 0. The freeze (about 03:00 UTC)

1. The conductor sends the freeze commit `F` and the go.
2. Bring `fastmace/2-T23` up to `F` (`sync_with_base_branch`), and check:
   - `git rev-parse origin/main` is `F`;
   - `git diff F HEAD -- fastmace` is empty (only `tools/`, `tests/`, `docs/` and `results/` may differ);
   - `git diff --name-only 6551085 F -- tests/gpu`: OPT-T06/T07 GPU files merged by `F` are added to
     `ROUND_TEST_FILES` in `tools/acceptance/suites.py` if they are not acceptance-marked (acceptance-marked
     files run in the acceptance tier anyway); commit and push.
3. Pin: `git archive` of that head to `~/fastmace-2-T23/src-<sha>` (PC) and
   `/mnt/homes/zwingel/fastmace-2-T23/src-<sha>` (murphy), each with `.source_commit`. Every job below
   uses that copy (`run.py --fastmace`). Record `F` and `<sha>` in the report.
4. The release side of item 8 is unchanged: tag `fastmace-1b` (`673a8a1`), the renamed `fastmace_1b` in
   `~/fastmace-versions` (PC) and `/mnt/homes/zwingel/fastmace-2-T23/versions` (murphy).
5. **Preflight (the defaults):** the round's rewrites are `"auto"` in `RewriteConfig` on main at the
   time of writing, so the first item 8 process checks what auto does at `F`: its metadata must show
   `gated_rows` on for MH-1 on both crystals, and `segment_gemm` on for NbCl5 (NaCl takes the stock path:
   `plans/opt-1.md`). If either is off, stop that machine's queue and ask the conductor. The gate and
   item 8 test the defaults at `F`, whatever they are.

## Rules for every job

- **One job at a time per machine**; the two machines run in parallel.
- **Local RTX 5060 Ti:** every GPU job under `python3 tools/ci/local_gpu.py run --purpose "2-T23 …"`,
  `TORCHINDUCTOR_COMPILE_THREADS=4`, detached with `setsid -f`. Timed jobs only with nothing else on the
  PC (the conductor keeps the other workers off); the conductor is told when a timed block starts and ends.
- **murphy:** only with no process of anyone on the GPU (`run.py`'s idle check), files only under
  `/mnt/homes/zwingel`, detached with `setsid nohup`. Timed jobs also need a quiet host: other users'
  load about 4 or below before each timed suite. A logger records the load and other users' %CPU every
  60 s; blocks timed while it was above that are marked and reported.
- Own processes are stopped only by their own process group, and only on the conductor's word.
- Floors (12-block same-process A/B): murphy 0.16 %; RTX 5060 Ti 1.1 % NaCl, 0.4 % NbCl5 (2-T12).

## Order, machines and durations

Durations come from 20 October's runs (2-T23) and 1b-T23's manifests. The ASE-equivalence file is now
4,000 steps instead of 18,000 (`edba9f3`), so the acceptance tier is estimated about 70 min shorter.

**Local RTX 5060 Ti** (run ID `2-T23-local-<date>-combined`):

| # | Job (`run.py` suite) | Timed | Estimate | What it decides |
| --- | --- | --- | ---: | --- |
| L1 | item 1: 2-T04d's driver on MACE 0.3.15 (`fastmace_mace_other`) | no | 2 min | item 1 |
| L2 | `regression2` | **yes** | 21 min | item 8, step times; preflight (0.5) |
| L3 | `acceptance-tier` | **yes** | ~1 h 50 | items 2–6 on final main; the round's ASE equivalence, restart, reproducible |
| L4 | `gate-local` (10 labels) | no | ~1 h | 19-label gate, local part |
| L5 | `trajectory2` | no | 14 min | item 8, default trajectories |
| L6 | `round-tests` | no | ~15 min | the round's restart, reproducible and kernel files outside the acceptance tier |
| L7 | `full-tier` | no | 37 min | merge tier on final main (last; may run past 07:30) |

Total about 4 h 20 min, so about 07:20 UTC from a 03:00 start.

**murphy RTX 4000 Ada** (run ID `2-T23-murphy-<date>-combined`):

| # | Job | Timed | Estimate | What it decides |
| --- | --- | --- | ---: | --- |
| M1 | `regression2`, then `regression2-nacl` | **yes** | 23 + 16 min | item 8, step times; the NaCl rule (phase-2.md) |
| M2 | `acceptance-tier` | **yes** | ~2 h | items 2, 4–6 on sm_89; the round's acceptance on sm_89 |
| M3 | `runner-repeat` | no | ~40 min | item 4, hybrid (the owner's repeat) |
| M4 | `gate-murphy` (9 labels) | no | ~1 h 10 | 19-label gate, murphy part |

Total about 4 h 30 min, so about 07:30 UTC. If it overruns, the gate (last) finishes after the 07:30 report:
the report then gives the labels done and is completed when the gate ends.

**Why this split.** Timed work goes first, while both hosts are quietest. The 19-label gate is split by
reference memory: ten labels whose float64 references are expected to fit 16 GB (MH-1 and OMOL-0 XL ran
on the PC in 1b-T23; the other MP-0, MPA-0, OMAT-0 and MatPES models are medium-sized) run locally; MP-0 large, the three POLAR-1 sizes and the five MACE-OFF labels need murphy's 20 GB
(1b-T03's peaks: 18.6 GB for POLAR-1-S on NbCl5, 17.1 GB for OFF24-M on paracetamol 3×3×3). A local
label whose reference runs out of memory is rerun on murphy after M4. Item 4's repeat runs on murphy,
which ran 2-T07's runner cases. Item 1 runs locally only (conductor's answer Q2). The in-library 19-label
gate (`library-gate`, 36 min in 1b) is not in the plan; it runs at the end only if time is left.

## Pass criteria

| Item | Rule | Source |
| --- | --- | --- |
| 1 | the run completes on the reported path, gate passed, unhashed records off with reasons, only hashed patches in force | phase-2.md, item 1 |
| 2, 3, 5, 6 | the acceptance tier's tests pass on both cards (item 3 is decided by the 16 GB card) | plans/phase-2.md |
| 4, plain cuEq and chunks | `test_subprocess_runner_gpu.py` passes on both cards | 2-T07 |
| 4, hybrid | the equilibrated repeat: all 4 seeds, all 3 quantities within max(2 × D, floor); runner = subprocess, fresh = in-process, backend hybrid | phase-2.md, "Item 4, hybrid: the repeat's protocol" (written before the first run) |
| 7 | deferred by the owner (SPEC draft 0.12, eleventh amendment) | – |
| 8, plans | every A/B process: the only option differences against 1b are `gated_rows` and `segment_gemm` (`--allow-option-changes`); anything else makes the process invalid | `tools/ab_fastmace.py` |
| 8, trajectories | plain cuEq float64 within 1e-12 Å of 1b (stage A is bitwise, stage B keeps the stock path in float64); hybrid within 1b-T14's statistical limits | `tools/trajectory_versions.py` |
| 8, step times | no process slower than 1 + floor (main may be faster: that is the round's point). murphy NaCl: the median main/1b over all 6 processes ≤ 1.0016, none above 1.0032, controls not slower | phase-2.md, "Item 8 on murphy, NaCl" (written before the run) |
| 9 | `main.py` from 2-T21's report; the others as in phase-2.md | – |
| Gate, 19 labels | per label, on the structures that count: production error ≤ max(2 × float32 yardstick, floor) for energy, forces and stress (1b's rule) | `tools/force_stress_gate.py` |
| Round | ASE equivalence with the tenth amendment's limits (1e-6 K, 1e-7 meV/atom); restart and reproducible tests with the rewrites on | `plans/opt-1.md`, SPEC |

**A failing gate label** is recorded with its ratios. The conductor switches the rewrite off for that
model (`plans/opt-1.md`); 2-T23 does not.

## Item 8 against an optimised main: what changes

1b's rule was "equal plans, equal step times". After the round, the plan differs from 1b by owner
decision in exactly the two rewrites, and main is expected to be faster: murphy −4.4 % (NaCl) and
−9.0 % (NbCl5) from stage A, NbCl5 another −4.8 to −5.2 % from stage B; locally −3.7 % and −10.6 %, then
−4.2 to −4.3 %. So the A/B now shows the round's gain against 1b, and the regression check is one-sided:
no process may be slower than its floor allows. The trajectory checks still hold the rest of the plan to
1b, and the gate holds the new rewrites to float64.

## After the runs

- `tools/acceptance/summarize2.py <run root> --machine local|murphy --allow-option-changes gated_rows segment_gemm`
  for item 8; `tools/runner_statistics.py compare --fresh … --runner …` for item 4; the gate drivers'
  summaries for the 19 labels; `summarize2.py` also reads the tiers' junit files.
- The verdict per item goes into phase-2.md by about 07:30 UTC, with run IDs, `F` and `<sha>`.
- Summaries and small files are copied into `results/`; large files stay on the machines.
