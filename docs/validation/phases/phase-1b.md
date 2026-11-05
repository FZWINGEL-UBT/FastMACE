# Phase 1b acceptance report

**Status: runs complete, 19 October 2026.** Plan task 1b-T23, branch `fastmace/1b-T23`.

- **Pass:** items 1 (19-label gate), 2, 3, 4a, 4c, 6 and 7.
- **Fail:**
  - item 1c: POLAR-1-S NPT in-library gate, at the stress floor;
  - item 4b: the ASE shim's per-call cost;
  - item 5 on NaCl: outside the noise floor in both directions, depending on the machine.
- **Three test failures** in research-engine comparisons that the first suite runs skipped.

Each failure and finding is listed with its numbers. None of them changed library code: the library
in every run is main `591be54`, and every 1b-T23 commit after it touches `tools/`, `tests/` and
`docs/` only. The owner decides at checkpoint 3.

**Update, 20 October 2026:** follow-up tasks 1b-T03b, 1b-T21b and 1b-T22b resolved findings 2, 4, 5
and 6. Two failures remain for the owner: 1c (POLAR-1-S NPT stress at the floor) and 5 (NaCl timing).
See [Addendum](#addendum-20-october-2026) for the final table. The sections below record the runs of
19 October as they were.

## Verdict

| # | Item (plans/phase-1b.md) | Result |
| --- | --- | --- |
| 1a | Gate with energy, 19 labels, on the structures that count | **pass** on murphy, every counting run: worst ratio energy 2.65 (below the 0.01 meV/atom floor), force 1.39, stress 1.57 |
| 1b | Energy headline | **pass**: POLAR-1 and OMOL-0 XL production energy at most 0.44 meV/atom from float64 (1a: 45–102); MACE-OFF at most 0.0004 meV/atom (1a: 3.9–15) |
| 1c | In-library reduced gate, once per label | **fail, 1 of 38 cases**: POLAR-1-S NPT fails at the stress floor (reproduced); the other 18 labels pass NVT and NPT; POLAR-1-S NVT passes |
| 1d | Gate subset on the RTX 5060 Ti | **pass** where the float64 reference fits 16 GB: MH-1, POLAR-1-S (NaCl), OMOL-0 XL; OFF24-M with paracetamol 2×2×2 (`passed-reduced`, as 1a for OFF23-L) |
| 2 | Registry and capture-state checks | **pass**: `test_registry_gpu.py` 4/4 and the tiny registry test, on murphy and the RTX 5060 Ti, full GPU suite in one process |
| 3 | Restart | **pass**: round trip, bitwise CPU and the kill test in the CPU suites; plain cuEq float64 within 1e-12 Å and the hybrid statistics in both GPU suites; compiled PolarMACE NPT across restart points: no recompile |
| 4a | `fastmace doctor`, then `fastmace md` | **pass**, with one finding: the report prints a wrong "naive cuEq" warning |
| 4b | `fm.md`, `fastmace.ase.Langevin` and `Simulation`: same plan, same gate record, equal step time | **plan and gate pass; time: `fm.md` passes, the ASE shim fails** by a fixed ~0.26 s per `run()` call (+20 % in 300-step blocks, +2.4 % in 3000-step blocks). Timed on the RTX 5060 Ti instead of murphy (below) |
| 4c | The other §7.7 items | **pass** in the CPU suites (2128 / 2081 passed, conductor's trial) |
| 5 | 1b against the 1a tag in one process | NbCl5 **pass** on murphy (0.9996–1.0004), within 0.18 % locally; NaCl **outside the floor in both directions**: 1b 0.4–0.6 % faster on murphy, 0.06–0.44 % slower locally |
| 5b | `MinimalPlanner` against `FixedPlanner`, 500 steps | **pass** in both GPU suites (`test_minimal_planner_gpu.py` 2/2) |
| 5c | `fm.calc` recompile | **measured**: one compile at the first call, none in five more calls on the same cell |
| 6 | `main.py` compatibility | **pass**: adapter tests 24 passed + 55 subtests; smoke equal to `--engine gpu` to float32 divergence; energy offset 0.035 meV/atom, D2's float64 sum |
| 7 | Rules | **pass**: layering, switch and exception tests in the CPU suites; `ruff` clean |
| – | GPU suite, full tier | murphy and RTX 5060 Ti: 176 passed, 0 failed. The 77 tests that first skipped for missing environment variables then ran on murphy: 116 passed, **3 failed** (findings 4, 5) |
| – | MACE imported first (setup, measured) | 48 s against 34 s for MH-1 NaCl; the ~310 s of 1b-T26 did not appear |
| – | Flaky test (watch) | no unexplained failure in 4 GPU-suite runs |

## Addendum (20 October 2026)

Three follow-up tasks, merged into main before this addendum (`11596b3`), resolved four of the
findings below. This addendum ran nothing new; its numbers come from those tasks' records.

- **Findings 4 and 5, the research-engine comparisons in `test_engine_gpu.py`: resolved by 1b-T03b**
  ([1b-T03-energy.md](1b-T03-energy.md), section "1b-T03b"; murphy).
  - Kernel lists: the float64 energy reduction is part of the compiled stage B, so Inductor fuses and
    numbers its kernels differently (same node count). These comparisons now run FastMACE with
    `float64_energy=False`, as the other research-engine comparisons do. The trajectory limits are
    unchanged.
  - Hybrid capture crash: the test's own warm-up check cloned every registered tensor without
    detaching. One of them is a leaf that requires grad, and its `AccumulateGrad` node stayed on the
    default stream. The test now detaches; the library's `registry.snapshot()` already did. No
    library change.
  - Results: the three tests pass. NVT 2.5e-13 Å after 500 steps, NPT 7.1e-15 Å, hybrid 1.0e-7 Å
    against a research-engine run-to-run scatter of 8.7e-8 Å.
- **Finding 2, the ASE shim's cost per `run()` call: resolved by 1b-T21b**
  ([1b-T21b-ase-shim-overhead.md](1b-T21b-ase-shim-overhead.md)).
  - Cause: the shim built a second `metadata.json` record at every call, about 0.25 s, almost all of it
    `provenance.fastmace_block`.
  - Fix: it now adds its `"ase"` block to the record the `Simulation` has just written.
  - Ratio to `Simulation` in 300-step calls on the RTX 5060 Ti:
    - FastMACE source on ext4: 1.015 before, 1.000 after;
    - checkout on `/mnt/d`: 1.90 before, 0.986 after.
  - **Open for phase 2:** `Simulation.run` itself rebuilds that provenance record at the end of every
    call. That costs about 0.13 s per call from a worktree, and about 13 s from the main checkout on
    `/mnt/d`, where `git_commit` is slow over WSL's file system. `fm.md`, the CLI and the shim pay it
    too.
- **Finding 6, the wrong "naive cuEq" warning from `fastmace md`: fixed by 1b-T22b.**
  - Cause: the planner's rule 6 record used the code `allow-naive-cueq` on every run, and the report
    read it as a downgrade.
  - Fix: it now reads `fused-cueq` when the fused kernels load.
  - Confirmed by a hybrid `fastmace md` run on the RTX 5060 Ti.

**Final table.** The items that passed on 19 October are unchanged.

| # | Item | Final result |
| --- | --- | --- |
| 1a, 1b | Gate with energy, 19 labels; energy headline | **pass** |
| 1c | In-library reduced gate | **fail, open for the owner**: POLAR-1-S NPT, stress 1.18e-7 and 1.20e-7 eV/Å³ against the 1e-7 floor (owner decision 7); the other 37 cases pass |
| 1d | Gate subset on the RTX 5060 Ti | **pass** (OFF24-M `passed-reduced`) |
| 2 | Registry and capture state | **pass** |
| 3 | Restart | **pass** |
| 4a | `fastmace doctor`, `fastmace md` | **pass** (warning fixed by 1b-T22b) |
| 4b | Three front ends: plan, gate, step time | **pass** after 1b-T21b (shim / `Simulation` 1.000 on ext4); `Simulation.run`'s per-call provenance is open for phase 2 |
| 4c | Other §7.7 items | **pass** |
| 5 | 1b against 1a | NbCl5 **pass**; NaCl **open for the owner**: 1b 0.4–0.6 % faster on murphy, 0.06–0.44 % slower on the RTX 5060 Ti, outside the 0.16 % floor in both directions |
| 5b, 5c | `MinimalPlanner` against `FixedPlanner`; `fm.calc` recompile | **pass**; measured |
| 6 | `main.py` compatibility | **pass** |
| 7 | Rules | **pass** |
| – | GPU suite, full tier | **pass**: 176/176 on both machines; the 3 research-engine comparisons pass after 1b-T03b |

## Failures and findings for the conductor

1. **POLAR-1-S NPT fails the in-library gate at the stress floor** (item 1c). Run `1b-T23-murphy-20261019`, reproduced in `1b-T23-murphy-rerun2-20261019`:

   | Run | Production stress error (eV/Å³) | Float32 yardstick | Limit (floor) |
   | --- | ---: | ---: | ---: |
   | first | 1.18e-7 | 3.89e-8 | 1e-7 |
   | repeat | 1.20e-7 | 3.26e-8 | 1e-7 |

   - The external driver passes the same model's stress with ratio 1.57. Its 3 configurations per cell mode give a larger yardstick error, so its limit is above the floor.
   - The production error equals 1a-T19's production-against-finite-difference value (1.1e-7) for POLAR-1-S.
   - The floor is owner decision 7 (SPEC §12.3), and PolarMACE NPT is reported as "experimental: not yet qualified". Neither the floor nor the gate's copies were changed; the conductor has put this on the owner's list.
   - Production about 3× the float32 yardstick on PolarMACE stress means the hybrid polar stress path is less precise than stock float32 here. That deserves phase 3b's attention, beside 1a's NbCl5 finite-difference finding (4.4e-4 eV/Å³).
2. **The ASE shim costs about 0.26 s per `run()` call** (item 4b).
   - 300-step blocks: 4.957 against 4.113 ms/step, slower in all 12 blocks.
   - 3000-step blocks: 3.748 against 3.660 ms/step (+2.4 %).
   - Per step the engine is the same. The cost is per call: after each `run(n)` the shim writes the atoms, its records and the speed line.
   - A script that calls `dyn.run(10000)` once pays 0.6 %. One that calls `run()` in a loop of short chunks pays much more.
   - `fm.md` against `Simulation` is +0.43 % and +0.48 % in the two protocols. That is inside this machine's block scatter (pair ratios 0.92–1.08 with 300-step blocks).
3. **NaCl in item 5 depends on the machine.**

   | Machine | NaCl, 1b/1a (three processes) | NbCl5, 1b/1a |
   | --- | --- | --- |
   | murphy | 0.9957, 0.9940, 0.9951 (every process; pair medians 0.994–0.996) | 0.9996–1.0004 |
   | RTX 5060 Ti | 1.0044, 1.0030, 1.0006 | 0.9982–1.0015 |

   - Every process is valid: equal engine options, nothing compiled and no process-wide patch changed while timed.
   - The 0.16 % noise floor was measured on murphy (1a-T18); the RTX 5060 Ti has none of its own, and its pair ratios scatter more (NaCl 0.987–1.014).
   - The opposite signs on the two machines argue against a regression in the step itself. Strictly, NaCl does not meet "within the noise floor"; the owner decides.
4. **Plain cuEq float64 against the research engine: the captured B graph has different kernels.** This is `test_engine_gpu.py::test_mh1_plain_cueq_float64_against_the_research_engine` and its NPT twin.
   - The trajectories still agree within the float64 limit.
   - The node count is equal (234 and 292), the kernel sequence differs.
   - Most likely 1b-T03's float64 energy reduction changes stage B, while the test still expects 1a's sequence.
5. **Hybrid float32 against the research engine: CUDA graph capture fails in the test's own FastMACE child.** This is `test_engine_gpu.py::test_mh1_hybrid_float32_against_the_research_engine`.
   - The error: "AccumulateGrad … stale reference to the default stream from warmup", raised in the float64-energy `ScaleShiftMACE.forward` during `warmup_check`.
   - Production hybrid captures worked everywhere today (A/B, gate, in-library gate, front ends, CLI), so it is tied to that test's eager warm-up check. It is still a capture failure.
6. **`fastmace md` prints a wrong warning.** The report says "naive cuEq allowed (allow_naive_cueq=True): cuEquivariance's fused kernels do not load, so this run is slow".
   - The same run is planned on the hybrid and runs at 3.55 ms/step.
   - `fastmace doctor` finds the fused kernels.
   - Either the CLI passes `allow_naive_cueq=True` and the report prints it as a fault, or the report reads the flag instead of the fact.
7. **1b's setup is 4–10 s longer than 1a's** in single-version processes, MH-1 NaCl:

   | Machine | 1b | 1a |
   | --- | ---: | ---: |
   | murphy | 19.5 s | 15.3 s |
   | RTX 5060 Ti | 24.0 s | 14.4 s |

   - On NbCl5 the two are within 1 s.
   - 1b-T12 recorded that each plan reloads and converts the checkpoint; the stages are below.
8. **MACE imported first.**
   - 48.4 s from process start to a built `Simulation`, against 33.8 s with FastMACE first.
   - With MACE first the plan falls back to plain cuEq with the reason `backend-cueq-mace-imported-first`.
   - 1b-T26's ~310 s with a tiny model did not appear for MH-1 on NaCl. It may belong to the tiny-model test path rather than to the library.
9. **`fm.calc` compiles once on same-size copies.**
   - The first call took 31 s on the hybrid and 46 s on plain cuEq; the next five took 0.01–0.02 s.
   - 1b-T18's second compile did not appear here: it depends on the edge count changing between calls, which rattled copies of one cell did not do.

## Runs

The library in every run is main `591be54`. The commit column is the tools' commit.

| Run ID | Machine | GPU, driver | What | Tools commit | Dates (UTC) |
| --- | --- | --- | --- | --- | --- |
| `1b-T23-smoke-murphy-20261019` | murphy | RTX 4000 Ada 20 GB, 595.58.03 | gate driver smoke (MH-1, NaCl 2×2×2) | `c4f5cbd` | 19 Oct, 07:47–07:50 |
| `1b-T23-murphy-20261019` | murphy | same | 1b against 1a, compile checks, gate (19 labels), in-library gate, GPU suite | `c4f5cbd` | 19 Oct, 07:50–12:58 |
| `1b-T23-murphy-rerun2-20261019` | murphy | same | POLAR-1-S in-library gate, PolarMACE NPT restart, `main.py` smoke, the skipped tests | `f6b7b86` | 19 Oct, 13:03–13:37 |
| `1b-T23-local-20261019` | PC (WSL2) | RTX 5060 Ti 16 GB, 610.88 | 1b against 1a, gate subset, GPU suite (invalid: basetemp, below), adapter tests | `c4f5cbd` | 19 Oct, 07:51–10:37 |
| `1b-T23-local-suite-20261019` | PC | same | GPU suite | `cc0a5fc` | 19 Oct, 10:40–13:02 |
| `1b-T23-local-timing-20261019` | PC | same | front ends (300-step blocks), MACE-first setup | `f6b7b86` | 19 Oct, 13:38–13:43 |
| `1b-T23-local-frontends-long-20261019` | PC | same | front ends (3000-step blocks) | `f6b7b86` | 19 Oct, 13:51–13:55 |
| `1b-T23-local-off24-reduced-20261019` | PC | same | OFF24-M gate, paracetamol 2×2×2 | `f6b7b86` | 19 Oct, 13:43–13:50 |
| `1b-T23-local-cli-20261019` | PC | same | `fastmace doctor`, `fastmace md` from a clean `git archive` of `591be54` | `591be54` | 19 Oct, 13:58–14:01 |

- **Environments:**
  - `mace_cueq_oeq` (torch 2.14.1+cu130, MACE 0.3.16, cuEquivariance 0.12.0, OEq 0.7.0) for every process that builds a hybrid engine;
  - `mace_cueq_latest` for the gate's references and driver and the pytest process of the GPU suites;
  - the NAS environments on murphy, `~/environments/…` on the PC.
- **How FastMACE ran:** from copies of the tools commit, through `PYTHONPATH`, never installed. On murphy those copies are `/mnt/homes/zwingel/fastmace-t23*`; the shared NAS checkout was not touched.
- **Run settings:** hybrid FP32, TF32 off, `TORCHINDUCTOR_COMPILE_THREADS=4`.
- **1a side of item 5:** tag `fastmace-1a` (`bb35363`), exported with `python -m fastmace.bench.versions` as `fastmace_1a` (tree digest `b817ed07…`).
- **Research-engine baseline:** tag `research-engine-baseline-20261017` (`d2faa9fd`), export `fastmace-baseline/1a-T18`.
- **ResearchModule:** `aa495277` locally (adapter tests); the NAS tree for the `main.py` smoke.
- **Data:** `results/<run-id>/`: each job's `metadata.json`, the run's `manifest.json`, the junit files, the smoke runs' thermo files and the CLI output. Large files stay on the machines.

**Item 4 was timed on the RTX 5060 Ti instead of murphy.** User margraf started FHI-aims CPU jobs (six or more processes, load about 13) on murphy at 14:58 local time (12:58 UTC). That was just as the front-end timing began, so it was stopped there. The conductor moved item 4 and the MACE-first setup to the RTX 5060 Ti, with nothing else running locally. The non-timing GPU reruns stayed on murphy, whose GPU those jobs did not use.

**leto was not used:** user cam's training job held its GPU all day.

**WSL's load average** reads about 130 while every CPU is idle (`top`: 100 % idle). It is a counting artefact of WSL2, not load.

## 1. Gate with energy

`tools/force_stress_gate.py --engine fastmace --planner minimal`, rebuilt in part 1:

- references from `fastmace.reference`, each in a fresh process before production;
- `selfcheck`'s ratio rule with energy gated;
- 3 configurations per structure and cell mode (hydrostatic, full), with ±2 % strain, a 0.05 Å rattle and seed 20261015;
- the 1a cases: the structures that count for the long-range labels, MACE-OFF on the organic set with the water box unrepeated, and paracetamol at 2×2×2 for OFF23-L.

Ratios are production error over yardstick error, worst over the structures. "E max" is the largest production energy error against float64, meV/atom.

| Label | E ratio | F ratio | S ratio | E max | Pass |
| --- | ---: | ---: | ---: | ---: | --- |
| mp0-medium | 2.65* | 1.04 | 1.11 | 0.0002 | yes |
| mp0-large | 1.52 | 1.24 | 1.27 | 0.0002 | yes |
| mp0b3-medium | 1.68 | 1.17 | 1.23 | 0.0003 | yes |
| mpa0-medium | 1.76 | 1.11 | 1.33 | 0.0004 | yes |
| omat0-medium | 1.52 | 1.15 | 1.10 | 0.0003 | yes |
| matpes-pbe | 1.12 | 1.27 | 1.49 | 0.0002 | yes |
| matpes-r2scan | 1.66 | 1.39 | 1.54 | 0.0002 | yes |
| mh1-omat | 1.00 | 1.13 | 0.94 | 0.0002 | yes |
| mh1-mp | 1.47 | 1.22 | 0.98 | 0.0002 | yes |
| mh1-r2scan | 1.10 | 1.10 | 0.90 | 0.0002 | yes |
| polar1-s (NaCl) | 1.10 | 0.99 | 1.57 | 0.0004 | yes |
| polar1-m (NaCl, NaNbO3; NbCl5 ×4) | 1.44 / 0.84 | 1.15 / 0.84 | 0.81 / 0.86 | 0.44 / 0.0004 | yes |
| polar1-l (NaCl, NaNbO3 2×2×2; NbCl5 2×2×2 ×4) | 0.93 / 1.12 | 0.91 / 0.98 | 0.94 / 0.70 | 0.0034 / 0.0019 | yes |
| omol0-xl (NaCl, NbCl5 2×2×2) | 1.85 | 1.25 | 1.48 | 0.0081 | yes |
| off23-small | 1.26 | 1.36 | 1.00 | 0.0003 | yes |
| off23-medium | 1.41 | 1.02 | 0.98 | 0.0002 | yes |
| off23-large (paracetamol 2×2×2) | 1.08 | 1.03 | 0.95 | 0.0003 | yes |
| off23b-medium | 1.59 | 1.04 | 1.20 | 0.0002 | yes |
| off24-medium | 1.40 | 1.06 | 1.33 | 0.0004 | yes |

\* below the 0.01 meV/atom floor (production 0.0002 against yardstick 0.0003 meV/atom), so it passes.

- **Finite-difference self-test** of each POLAR run's float64 reference: 6.9e-9 (S), 8.1e-9 (M), 9.4e-9 (L) eV/Å³, against a tolerance of 1e-6.
- **Outside the count, as in 1a:**
  - POLAR-1-S on NbCl5 (supplementary): force ratio 2.86 (1a: 2.55);
  - OMOL-0 XL on NaNbO3 2×2×2 (model-limited): energy errors of 3.2e4 meV/atom in production and 1.2e4 in the float32 yardstick.
- **Energy headline:**
  - the largest production energy error of a counting long-range run is 0.44 meV/atom, POLAR-1-M on NaNbO3, where float32 is 0.50;
  - in 1a the sums were 45–102 meV/atom off.
- **Subset on the RTX 5060 Ti:**
  - MH-1: 0.20 / 1.11 / 0.96;
  - POLAR-1-S NaCl: 0.28 / 1.08 / 0.53;
  - OMOL-0 XL: 1.31 / 1.19 / 0.93;
  - OFF24-M with paracetamol 2×2×2: 1.40 / 1.03 / 1.70.

  The float64 references of POLAR-1-S on NbCl5 and OFF24-M on paracetamol 3×3×3 need 18.6 and 17.1 GB (peaks on murphy); on 16 GB they stopped with "CUDA driver error: device not ready".

**In-library gate** (`tools/library_gate.py`): `Simulation(check="gate")`, NVT (2 rattled copies) and NPT hydrostatic (2 rattled and 2 strained copies), captured replay, NaCl (urea for MACE-OFF).

- References ran in-process in every case (no production engine existed yet).
- All 19 labels pass NVT; all but POLAR-1-S pass NPT (finding 1).
- Gate time 20–32 s per case. MH-1: setup 37 s = plan 0.0 s, build 13.6 s, gate 23.4 s (references 13.0 s, production 10.4 s).

## 2 and 3. Registry, capture state and restart: the GPU suites

Full tier, one process per machine, with hybrid children in `mace_cueq_oeq`:

| Machine | Run | Result | Wall |
| --- | --- | --- | --- |
| murphy | `1b-T23-murphy-20261019` | 176 passed, 80 skipped, 0 failed | 2 h 15 min |
| RTX 5060 Ti | `1b-T23-local-suite-20261019` | 176 passed, 80 skipped, 0 failed | 2 h 22 min |

They include:

- `test_registry_gpu.py` (MH-1 hybrid pipelined NVT, MH-1 NPT BAOAB full cell, POLAR-1-S, plain cuEq float64);
- `test_canonical_restart_gpu.py`: plain cuEq float64 within 1e-12 Å, and the hybrid statistics over 20,000 steps and two seeds;
- `test_minimal_planner_gpu.py`;
- `test_selfcheck_gpu.py`;
- `test_load_hybrid_gpu.py`.

**Skips, and the rerun that covered them:**

- 77 of the 80 skips were the suite environment's fault. `FASTMACE_BASELINE` and `FASTMACE_1A_T19_POLAR1S` were not set; fixed in `a066220`.
- The skipped tests then ran on murphy (`1b-T23-murphy-rerun2-20261019`): 116 passed, 3 failed (findings 4, 5).
- The remaining 2 skips are the OEq streaming-kernel tests, which need OEq in the pytest environment.

**First local suite run invalid** (`1b-T23-local-20261019`): its `--basetemp` sat inside a directory that did not exist yet, so every `tmp_path` test errored at setup (21 errors). The cause was in the suite definition; it was fixed in `cc0a5fc` and the suite was rerun.

**Compiled PolarMACE NPT across restarts** (`tools/compile_checks.py polar-npt`; POLAR-1-S NaCl 3×3×3, BAOAB hydrostatic):

- no recompile at any of 5 canonical restart points, 4.64–4.67 ms/step;
- resumed in a fresh process from the last restart: no recompile after its first step;
- 2.1e-7 Å from the uninterrupted run in positions and 4.7e-10 Å in cell, as expected for the float32 hybrid.

CPU items (3a, 3d, 4c, 7) are covered by the CPU suites of the conductor's trial on main: murphy `mace_cueq_latest` 2128 passed, local `fastmace_cpu` 2081 passed.

## 4. Adoption layer

**CLI** (`1b-T23-local-cli-20261019`), a clean `git archive` of `591be54`, `python -m fastmace.cli`:

- `doctor`: every required check passes, with one warning (torch-dftd, D3 reference only). All 17 release files match their SHA-256.
- `md --model mh1 --head matpes_r2scan --structure NaCl.cif --repeat 3 3 3 --temperature 300 --steps 2000 --check gate`, a complete run:
  - plan: graph, hybrid, float32, 12 rewrites;
  - gate passed: 2.2e-5 meV/atom, 3.4e-6 eV/Å;
  - setup 48 s (plan 0.008 s, build 21 s, gate 27 s);
  - speed 3.55 ms/step, whole-run mean;
  - `metadata.json` written.
  - Finding 6 concerns its "why" line.

**Front ends**, MH-1 NaCl 3×3×3 hybrid NVT in one process (`tools/ab_frontends.py`; `fm.md`'s defaults, the same seed and output cadences):

- All three give the same `Plan` and the same gate record.

| Protocol | `Simulation` (ms) | `fm.md` | `fastmace.ase.Langevin` |
| --- | ---: | ---: | ---: |
| 12 blocks of 300 steps | 4.113 | 4.131 (+0.43 %) | 4.957 (+20.5 %) |
| 6 blocks of 3000 steps | 3.660 | 3.678 (+0.48 %) | 3.748 (+2.4 %) |

Each block is one `run(n)` of the front end, so these times include each front end's per-call work (output rows, energy checks; for the shim also the atoms and its records). Finding 2 has the reading.

## 5. No regression against 1a

`tools/ab_fastmace.py` setup:

- 1b (minimal planner) against the renamed `fastmace_1a` (fixed planner) in one process;
- MH-1 `matpes_r2scan` on NaCl and NbCl5 3×3×3, hybrid FP32 NVT;
- 300 warm-up steps, then 12 alternating blocks of 300;
- three processes per machine: ABAB with 1a built first, ABAB with 1b built first, ABBA.

Both versions' process-wide patches are swapped before each side runs (`fastmace.bench.versions.ProcessPatches`), and the OEq runtime is shared.

| Machine | Process | NaCl 1b/1a (pair range) | NbCl5 1b/1a (pair range) |
| --- | --- | --- | --- |
| murphy | ABAB, 1a first | 0.9957 (0.9948–0.9994) | 0.9997 (0.9994–1.0023) |
| murphy | ABAB, 1b first | 0.9940 (0.9935–1.0001) | 1.0004 (0.9973–1.0006) |
| murphy | ABBA | 0.9951 (0.9895–0.9999) | 0.9996 (0.9970–1.0019) |
| RTX 5060 Ti | ABAB, 1a first | 1.0044 (0.9992–1.0132) | 1.0010 (0.9963–1.0075) |
| RTX 5060 Ti | ABAB, 1b first | 1.0030 (0.9987–1.0126) | 1.0015 (0.9931–1.0037) |
| RTX 5060 Ti | ABBA | 1.0006 (0.9873–1.0139) | 0.9982 (0.9953–0.9999) |

**Validity:**

- every process is valid: equal engine options, nothing compiled while timed, no patch changed while timed;
- the two versions' potential energies differ by D2's float64-sum shift: 0.12–0.13 meV/atom on NbCl5, 0.005 on NaCl, as 1b-T03 measured.

**Single-version controls** (4 blocks, one version per process):

| Machine | NaCl, 1a / 1b | NbCl5, 1a / 1b |
| --- | --- | --- |
| murphy | 3.494 / 3.497 ms | 31.006 / 30.867 ms |
| RTX 5060 Ti | 3.541 / 3.544 ms | 27.323 / 26.925 ms |

**Setup per stage, gate included:**

- in-library gate on murphy, per case: plan under 0.3 s, build 14–20 s, gate 20–32 s;
- `fastmace md`: 48 s in total (plan 0.008 s, build 21 s, gate 27 s);
- setup against 1a: finding 7.

## 6. Compatibility

**Adapter tests**, ResearchModule `aa495277` `tests/core/test_fastmace_engine.py` against `591be54`, read only:

- run with `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`, and `--basetemp` outside the tree; no file in ResearchModule's tree changed;
- result: 24 passed and 55 subtests passed (CPU float64, tiny models).
- They compare against the research engine in float64, where the float64 sum changes nothing, so D2's shift cannot show there.

**`main.py` smoke**, NaNbO3 3×3×3, MH-1 hybrid FP32, 0.5 ps to 400 K and 0.5 ps back, 2000 steps, from the NAS tree on murphy:

| Measure | `--engine gpu` | `--engine fastmace` |
| --- | --- | --- |
| completed steps / rows | 2000 / 21 | 2000 / 21 |
| final temperature (K) | 161.48171 | 161.48177 |
| largest difference over the rows: temperature | – | 8.6e-5 K (1a: 1.4e-4) |
| potential energy at step 0 (identical positions) | – | 3.5e-5 eV/atom (D2's float64 sum) |
| largest potential-energy difference | – | 5.6e-5 eV/atom |
| wall time (s) | 79.5 | 79.6 |

The energy differs by the float64 accumulation alone: 0.035 meV/atom at step 0, before any trajectory difference.

## 7. Rules

These are in the CPU suites of the conductor's trial on main:

- layering test (no `fastmace.engine` imports from `model` or `rewrites`, no research-engine imports);
- switch allow-list (no new module-level switches);
- exception hierarchy (every public exception derives from `FastMACEError`, with the §7.7 message shape).

`ruff check fastmace tests tools` is clean on this branch.

## Tool fixes during the runs (no library code)

- `bf35017`: the front-end harness and the compile checks asked for the GPU name before the planner set the CUDA allocator, and the library refused with `ProcessConstraintError`. Both now ask only after the first build.
- `cc0a5fc`: pytest's `--basetemp` sits directly under the run root.
- `a066220`: the GPU suite gets `FASTMACE_BASELINE` and 1a-T19's POLAR-1-S references.
- `f6b7b86`: the MACE-first setup measurement.
- Gate job names for explicit structure lists use file stems, not paths (this commit).

## Reproducing

From a FastMACE checkout on the machine (`$FM`), with an idle GPU and nothing else running for timing items:

```bash
python -m fastmace.bench.versions export --repository $FM --ref fastmace-1a --destination ~/fastmace-versions
bash $FM/tools/acceptance/launch.sh regression frontends compile-checks gate-1b library-gate gpu-suite \
  --machine murphy --run-id 1b-T23-murphy-<date> --fastmace $FM
bash $FM/tools/acceptance/launch.sh regression gate-1b-subset gpu-suite --machine local --run-id 1b-T23-local-<date>
bash $FM/tools/acceptance/launch.sh workflow --machine murphy --run-id 1b-T23-workflow-<date> \
  --research-module /mnt/homes/zwingel/ResearchModule --only main-smoke-NaNbO3-gpu main-smoke-NaNbO3-fastmace
```

The workstations need the renamed copy under `/mnt/homes/zwingel/fastmace-versions`: export it on the PC and copy it there.
