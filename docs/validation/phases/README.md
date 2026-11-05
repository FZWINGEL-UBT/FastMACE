# Phase reports

Task reports, phase acceptance reports and runbooks of phases 0–6 (17–26 October 2026; plans in `plans/phase-*.md`), the 1.0.0 release check and the CPU suite-speed work.

Columns: file, date added, title, and the report's own verdict line or first sentence.

| File | Added | Title | Verdict / first sentence |
| --- | --- | --- | --- |
| [1a-T07-T08-gpu](1a-T07-T08-gpu.md) | 2026-10-18 | GPU confirmation of the merged MH-1 and generic rewrites (1a-T07, 1a-T08, 1a-T07b) | 18 October 2026. |
| [1a-T08-generic-rewrites](1a-T08-generic-rewrites.md) | 2026-10-17 | Phase 1a: generic rewrites on released models (plan task 1a-T08) | verdict: "the first interaction RealAgnosticInteractionBlock is not a generic class". The other |
| [1a-T13-engine](1a-T13-engine.md) | 2026-10-18 | 1a-T13: CUDA-graph engine against the research engine | Plan task 1a-T13, 18 October 2026. |
| [1a-T13b-bucket-compile](1a-T13b-bucket-compile.md) | 2026-10-18 | 1a-T13b: the compile at T19's first new edge bucket | Follow-up to 1a-T13, 18 October 2026. |
| [1a-T15-planner](1a-T15-planner.md) | 2026-10-18 | 1a-T15: fixed-plan planner | Plan task 1a-T15, `fastmace/planner/fixed.py`. |
| [1a-noise-floor](1a-noise-floor.md) | 2026-10-17 | Phase 1a: noise floor of the A/B step-time criterion | Phase 1a's acceptance needs FastMACE's MD step time within 3 % of the frozen research engine, measured in alternating A/B blocks in one process (SPEC §12.2 Performance … |
| [1b-T03-energy](1b-T03-energy.md) | 2026-10-19 | 1b-T03: float64 energy accumulation | Plan task 1b-T03 (SPEC §7.3 "Energy" and "Energy and forces are one scalar", §6.2 `float64_energy`, §4.3 energy scale). |
| [1b-T04-registry](1b-T04-registry.md) | 2026-10-19 | 1b-T04: runtime-state registry and complete capture restore | Result: 1192 passed, 176 skipped in 314 s. `tests/cpu/test_registry.py` alone: 38 passed, |
| [1b-T08b-hybrid-load](1b-T08b-hybrid-load.md) | 2026-10-19 | 1b-T08b: the hybrid after a plain `fm.load` (GPU check) | Plan task 1b-T08b. |
| [1b-T09-integrators](1b-T09-integrators.md) | 2026-10-19 | 1b-T09: integrator qualification (ASE equivalence and the BAOAB correction) | Plan task 1b-T09, 19 October 2026. |
| [1b-T11-mp0-setup](1b-T11-mp0-setup.md) | 2026-10-19 | 1b-T11: MP-0 medium speed gap and setup time | Plan task 1b-T11 (phase 1b), branch `fastmace/1b-T11`. |
| [1b-T13-reference](1b-T13-reference.md) | 2026-10-19 | 1b-T13: reference records and corrected reference calculators | finding: a forward stored on the model before the polar fix was installed made the references run |
| [1b-T14-canonical-restart](1b-T14-canonical-restart.md) | 2026-10-19 | 1b-T14: canonical restart points on the GPU | Plan task 1b-T14 (SPEC §10.6 "Canonical restart points", §12.2 restart row). |
| [1b-T16-gate](1b-T16-gate.md) | 2026-10-19 | 1b-T16: the gate on the user's structure | Result: 7 passed (694 s). JSON: `results/1b-T16-gate-murphy-20261019/cases.jsonl`. |
| [1b-T17-simulation](1b-T17-simulation.md) | 2026-10-19 | 1b-T17 `Simulation`: GPU smoke check | Plan task 1b-T17, 19 October 2026. |
| [1b-T18-calc](1b-T18-calc.md) | 2026-10-19 | 1b-T18: `fm.calc` on the GPU | Plan task 1b-T18 (SPEC §7.4, §7.7 level 0). |
| [1b-T21b-ase-shim-overhead](1b-T21b-ase-shim-overhead.md) | 2026-10-19 | 1b-T21b: the ASE shim's cost per `run()` call | 1b-T23 (acceptance item 4b) measured `fastmace.ase.Langevin` about 0.26 s slower per `run()` call than `Simulation.run` (+20 % in 300-step blocks), with equal per-step … |
| [1b-T24-test-suite-speed](1b-T24-test-suite-speed.md) | 2026-10-19 | Test-suite speed (plan task 1b-T24) | Measured on 19 October 2026, on the local PC only. |
| [1b-T25-tiny-fast-tier-tests](1b-T25-tiny-fast-tier-tests.md) | 2026-10-19 | 1b-T25 Tiny-model fast-tier GPU tests (19 October 2026) | Two GPU behaviours were tested only in the full tier (1b-T24), by MH-1 tests marked `slow`. |
| [1b-T26-gpu-retiering](1b-T26-gpu-retiering.md) | 2026-10-19 | 1b-T26 Re-tiering the GPU suite (19 October 2026) | Since 1b-T24, merged tasks had added GPU tests that run released checkpoints without the `slow` mark, so the fast tier (`pytest tests/gpu -m "not slow"`) had grown from … |
| [1b-T27-gpu-tiers](1b-T27-gpu-tiers.md) | 2026-10-19 | 1b-T27 GPU acceptance tier (19 October 2026) | Goal: the full GPU tier back to 30–40 minutes without losing a check. |
| [1b-T28-software-block](1b-T28-software-block.md) | 2026-10-19 | 1b-T28: the software block once per process | answers over 9P. After the change, step time is the same as from ext4 (1b-T21b: 3.62–3.67 ms/step), |
| [2-T02-eager-tier](2-T02-eager-tier.md) | 2026-10-20 | 2-T02: eager tier for MD | Plan task 2-T02 (`plans/phase-2.md`); SPEC §8.3, §9.3 (mode dimension, rules 3 and 4), §9.4 step 2. |
| [2-T03-reference-tier](2-T03-reference-tier.md) | 2026-10-20 | 2-T03: the reference tier for MD | Plan task 2-T03 (SPEC §8.4, §9.3 rule 1, §9.4 step 5, §12.1), 20 October 2026. |
| [2-T04-unknown-mace](2-T04-unknown-mace.md) | 2026-10-20 | 2-T04: unknown MACE versions, always-on treatments, the self-check reaction | Plan task 2-T04 (`plans/phase-2.md`), SPEC §6.3, §6.4, §9.3 rule 13, §12.1 and §18 decision 2: MACE 0.3.16 is the minimum and the only qualified version. |
| [2-T05-gate-resources](2-T05-gate-resources.md) | 2026-10-20 | 2-T05: gate resource policy and statuses (acceptance 3) | Plan task 2-T05 (plans/phase-2.md); SPEC §12.3 "Status" and "Resources". |
| [2-T07-subprocess-runner](2-T07-subprocess-runner.md) | 2026-10-20 | 2-T07: the subprocess runner | Results: `results/2-T07-murphy-20261020/`. |
| [2-T08-isolation](2-T08-isolation.md) | 2026-10-20 | 2-T08: instance isolation, A → B → A (phase 2 acceptance 5) | Result: 9 passed, 1 skipped (PolarMACE: `fastmace_cpu` has no |
| [2-T08c-stage-b-kernels](2-T08c-stage-b-kernels.md) | 2026-10-20 | 2-T08c: stage-B kernel count against the research engine on murphy | Plan task 2-T08c (conductor, 20 October 2026). |
| [2-T10-durable-output](2-T10-durable-output.md) | 2026-10-20 | 2-T10 Durable output and restart files | answer Q4). At every restart point the order is: |
| [2-T12-local-noise-floor](2-T12-local-noise-floor.md) | 2026-10-20 | Phase 2: noise floor of the local RTX 5060 Ti | Plan task 2-T12 repeats 1a-T18's noise-floor measurement on the local PC (RTX 5060 Ti 16 GB, WSL2), so that later local timings (2-T13, 2-T19, 2-T20, 2-T23) can be … |
| [2-T13-calc-compile](2-T13-calc-compile.md) | 2026-10-20 | 2-T13: `fm.calc` without a second compile when the edge count changes | Plan task 2-T13 (SPEC §8.2). |
| [2-T14-flaky-test](2-T14-flaky-test.md) | 2026-10-20 | 2-T14: the flaky CPU test | result depends on which tests ran earlier in the same xdist worker, and the test failed in |
| [2-T14e-reference-reuse](2-T14e-reference-reuse.md) | 2026-10-20 | 2-T14e: reusing float64 reference results in the GPU test tiers | Recommendation in short: |
| [2-T15-fallback](2-T15-fallback.md) | 2026-10-20 | 2-T15: the fallback algorithm (acceptance 2) | Plan task 2-T15 (SPEC §9.4, §9.3 rules 13–14, §9.5; decision 29). |
| [2-T16-reproducible](2-T16-reproducible.md) | 2026-10-20 | 2-T16: reproducible mode, plain cuEq MH-1 (phase 2 acceptance item 6) | Plan task 2-T16; SPEC §9.3 rule 5 and "`reproducible=True`", §8.7 "Reproducible processes", owner decisions 13 and 14, §18 risk "plain cuEq float64 is not bitwise … |
| [2-T19-setup](2-T19-setup.md) | 2026-10-20 | 2-T19: setup time per stage, 1b against 1a | Status: measured, 20 October 2026. Plan task 2-T19, branch `fastmace/2-T19`. Local RTX 5060 Ti only; |
| [2-T20a-step-breakdown](2-T20a-step-breakdown.md) | 2026-10-20 | 2-T20a: step timing breakdown of FastMACE (baseline of optimisation round 1) | Plan task 2-T20a (owner request, 20 October 2026) and the baseline of `plans/opt-1.md`. |
| [2-T21d](2-T21d.md) | 2026-10-21 | 2-T21d: `fm.Simulation` dropped edges from the forces after 0.2 ps | 21 October 2026. |
| [3-T02](3-T02.md) | 2026-10-22 | 3-T02: the species superset on the GPU | Conclusion. The default superset stays "the structure's species" (SPEC §8.5). A declared superset of four species on NaCl costs about 1 % of the step. A quiet-machine … |
| [3-T03-events](3-T03-events.md) | 2026-10-22 | 3-T03: mutation events and the derived-state registry, validation | Plan task 3-T03 (SPEC §8.5, §8.1 "Graph-safe events", §12.2 "Mutation events"), 22 October 2026. |
| [3-T05](3-T05.md) | 2026-10-22 | 3-T05: adaptive skin and headroom (rule 11) | Phase 3 (`plans/phase-3.md`, task 3-T05; conductor answers of 22 October 2026). |
| [3-T06](3-T06.md) | 2026-10-22 | 3-T06: the compiled MD tier | Plan task 3-T06 (plans/phase-3.md): MD on `torch.compile` (default mode) without CUDA graphs, `mode="compiled"`, rules 4 and 9, and the fallback's mode dimension graph → … |
| [3-T07](3-T07.md) | 2026-10-22 | 3-T07: energy-only graph and the MC evaluator | Plan task 3-T07 (`plans/phase-3.md`), 22 October 2026. |
| [3-T09](3-T09.md) | 2026-10-22 | 3-T09: MC driver, swaps and transmutations | Plan task 3-T09 (`plans/phase-3.md`), 22 October 2026. |
| [3-T10](3-T10.md) | 2026-10-22 | 3-T10: displacement moves | Plan task 3-T10 (`plans/phase-3.md`), 22 October 2026. |
| [3-T11](3-T11.md) | 2026-10-22 | 3-T11: the move-energy gate and the MC energy precision | decision 35 (owner D5). Acceptance item 3 of phase 3. |
| [3-T12](3-T12.md) | 2026-10-22 | 3-T12: MC exact enumeration and hybrid MD/MC statistics | Plan task 3-T12 (`plans/phase-3.md`), 22 October 2026. |
| [3-T14](3-T14.md) | 2026-10-22 | 3-T14: the swap-MC annealing protocol, end to end | Result: the protocol ran end to end at 1 fs, on the local RTX 5060 Ti and on an Ada GPU (leeloo): |
| [3-T15](3-T15.md) | 2026-10-22 | 3-T15 front ends: local GPU smoke | Plan task 3-T15 (`plans/phase-3.md`): `Simulation.run(mc=…)`, `fm.md(mc=…)`, the CLI `[[mc]]` tables, the report's MC line, `counters.mc` and the thermo MC columns. |
| [3-T20](3-T20.md) | 2026-10-22 | 3-T20: phase 3 benchmarks | Status, 22 October 2026: |
| [3-T22-fixcm](3-T22-fixcm.md) | 2026-10-22 | 3-T22: the Langevin `fixcm` projection with unequal masses, measurements | Plan task 3-T22, 22 October 2026. |
| [3-T22b-fused-projection](3-T22b-fused-projection.md) | 2026-10-22 | 3-T22b: the fixcm projection in the planner, the record and the fused stage | Plan task 3-T22b, 22 October 2026, follow-up of 3-T22. |
| [3-T22c](3-T22c.md) | 2026-10-22 | 3-T22c: the mass-weighted fixcm projection as the MD default | Plan task 3-T22c, 22 October 2026, owner decision 37 (SPEC §10.4, §18). |
| [3b-T04](3b-T04.md) | 2026-10-22 | 3b-T04: D3 integration, GPU checks | 22 October 2026. |
| [3b-T05](3b-T05.md) | 2026-10-22 | 3b-T05: NPT with D3, the pressure setter, the independent comparison | Status (23 October 2026): |
| [3b-T06](3b-T06.md) | 2026-10-23 | 3b-T06: the D3 cost on the production ramp | Result: the 10 % target is missed. |
| [3b-T07](3b-T07.md) | 2026-10-23 | 3b-T07: the NPT melt–quench protocol, end to end | 23 October 2026. |
| [3b-T08](3b-T08.md) | 2026-10-22 | 3b-T08: PolarMACE stress against finite differences (owner decision D2, SPEC decision 41) | Status: investigation only. There is no qualification, and PolarMACE + D3 stays refused in 3b. |
| [4-T02](4-T02.md) | 2026-10-23 | 4-T02: NVE, CSVR and Berendsen | Plan task 4-T02 (`plans/phase-4.md`), 23 October 2026. |
| [4-T03](4-T03.md) | 2026-10-23 | 4-T03: Nose-Hoover chains and MTK NPT | Plan task 4-T03 (`plans/phase-4.md`): ports of ASE 3.27's `NoseHooverChainNVT`, `IsotropicMTKNPT` and `MTKNPT` (`ase/md/nose_hoover_chain.py`) as graph-capturable … |
| [4-T03b](4-T03b.md) | 2026-10-23 | 4-T03b: the fused Nose-Hoover chain stage | Follow-up to 4-T03, agreed with the conductor on 23 October 2026. |
| [4-T04](4-T04.md) | 2026-10-23 | 4-T04: `FixAtoms` | Plan task 4-T04 (`plans/phase-4.md`), 23 October 2026, owner decisions 44 (D1 (a)) and 45 (degrees of freedom 3(N − n_fixed), no projection). |
| [4-T05](4-T05.md) | 2026-10-23 | 4-T05: GPU LBFGS and the optimiser driver | Plan task 4-T05 (plans/phase-4.md); SPEC §9.3 rules 10 and 11, §10.3. |
| [4-T06](4-T06.md) | 2026-10-23 | 4-T06: GPU FIRE, checks | Results in `results/4-T06-fabi-pc-20261023/` (`ase.json`, `gpu.json`; the `.npz` minima stay out |
| [4-T07](4-T07.md) | 2026-10-23 | 4-T07: variable-cell relaxations | Plan task 4-T07 (plans/phase-4.md); SPEC §9.3 rule 10, §10.3. |
| [4-T08](4-T08.md) | 2026-10-23 | 4-T08: Annealing-benchmark minima | Result: passes decision 46 as amended (owner, 23 October 2026), with one recorded exception. |
| [4-T09](4-T09.md) | 2026-10-23 | 4-T09: protocol relaxations, `fastmace relax`, `fastmace quench` | Plan task 4-T09 (plans/phase-4.md); SPEC §7.5, §10.3, §10.5; decision 47. |
| [4-T10](4-T10.md) | 2026-10-23 | 4-T10: front ends for the phase 4 integrators and optimisers | Plan task 4-T10 (`plans/phase-4.md`), SPEC §7.5–§7.7, decisions 44 (FixAtoms) and 48 (the ASE optimisers). |
| [4-T12b](4-T12b.md) | 2026-10-23 | 4-T12b: PolarMACE under cell changes (4-T07's diverging cell relaxation) | Task: the conductor's follow-up to 4-T07, 23 October 2026. |
| [4-T13](4-T13.md) | 2026-10-23 | 4-T13: the subprocess runner's child aborts at exit | Found during the phase 4 acceptance preparation (plan task 4-T13), 23 October 2026. |
| [5-T03](5-T03.md) | 2026-10-24 | 5-T03: rule 12, the memory estimate and its action | Status: done. 60 measured cases set `SAFETY = 0.90` and `LIBRARY_WORKSPACES = 5` (160 MiB). The |
| [5-T06](5-T06.md) | 2026-10-24 | 5-T06 runtime monitor: GPU smoke | Plan task 5-T06 (`plans/phase-5.md`), untimed. |
| [5-T07](5-T07.md) | 2026-10-24 | 5-T07: tier switch between segments | Plan task 5-T07 (`plans/phase-5.md`), SPEC §9.6, §18 decisions 52 and 53. |
| [5-T08](5-T08.md) | 2026-10-24 | 5-T08: mid-run fallbacks | decision 53's limit (conductor's answer Q4). After a fallback off the graph tier, adaptation does not switch |
| [5-T09](5-T09.md) | 2026-10-24 | 5-T09 calibration: GPU smoke (untimed) | Plan task 5-T09 (`plans/phase-5.md`), SPEC §9.1, decision 55 (D5 (a)). |
| [6-T04](6-T04.md) | 2026-10-25 | 6-T04 validation: prebuilt OEq host-extension wheels (25 October 2026) | Local PC, WSL2 Ubuntu 24.04 (glibc 2.39), RTX 5060 Ti, driver as in oeq-runtime.md; Docker Desktop 29.6.1 (Windows client, WSL integration off). |
| [6-T15](6-T15.md) | 2026-10-25 | 6-T15: CPython 3.13 prebuilt OEq wheel | Result: passed the local clean-install GPU check, 25 October 2026. |
| [6-T16](6-T16.md) | 2026-10-25 | 6-T16: CPU suite scheduling | Result: 4,655 passed, 553 skipped in 831.34 s (13 min 51 s), versus 1,998.13 s (33 min 18 s) |
| [combined-acceptance-runbook](combined-acceptance-runbook.md) | 2026-10-20 | Runbook: combined acceptance of phase 2 and optimisation round 1 | Plan task 2-T23, written 20 October 2026, about 23:00 UTC, before the freeze. |
| [phase-0](phase-0.md) | 2026-10-17 | Phase 0 validation | Result: `92 passed, 878 warnings in 13.72s` (24 workers). The warnings |
| [phase-1a](phase-1a.md) | 2026-10-18 | Phase 1a acceptance report | Status: complete. All four acceptance items pass. Plan task 1a-T19, branch `fastmace/1a-T19`, |
| [phase-1b](phase-1b.md) | 2026-10-19 | Phase 1b acceptance report | Status: runs complete, 19 October 2026. Plan task 1b-T23, branch `fastmace/1b-T23`. |
| [phase-2](phase-2.md) | 2026-10-20 | Phase 2 acceptance report | Status: combined acceptance complete, 21 October 2026, 09:25 UTC. Every item passes except item 7, |
| [phase-3-runbook](phase-3-runbook.md) | 2026-10-22 | Runbook: phase 3 acceptance | decisions, which resolve to the baseline's values (conductor's answer 1); any other engine-option difference |
| [phase-3](phase-3.md) | 2026-10-22 | Phase 3 acceptance report | Status: complete, 22 October 2026, with the GPU slow tier as the appendix. Plan |
| [phase-3b-runbook](phase-3b-runbook.md) | 2026-10-22 | Runbook: phase 3b acceptance | Plan task 3b-T10, written 22 October 2026 as a prestart, before any phase 3b result exists and before the other 3b tasks are merged. |
| [phase-3b](phase-3b.md) | 2026-10-23 | Phase 3b acceptance report | Status: complete, 23 October 2026. Plan task 3b-T10, branch `fastmace/3b-T10`. Items 1, 2, 4 and 5 pass (item 5 |
| [phase-4-runbook](phase-4-runbook.md) | 2026-10-23 | Runbook: phase 4 acceptance | Plan task 4-T13. |
| [phase-4](phase-4.md) | 2026-10-24 | Phase 4 acceptance report | Status: complete, 24 October 2026. Plan task 4-T13, branch `fastmace/4-T13`. The runbook and pass rules were |
| [phase-5-runbook](phase-5-runbook.md) | 2026-10-23 | Runbook: phase 5 acceptance | Plan task 5-T11, written by the conductor on 25 October 2026 (owner: "After all current tasks go ahead and do the acceptance of Phase 5"; "just continue as you see fit"). |
| [phase-5](phase-5.md) | 2026-10-25 | Phase 5 acceptance report | Status: all items pass, 25 October 2026. Plan task 5-T11, branch `fastmace/5-T11-tools`. Runbook and pass rules, |
| [phase-6-runbook](phase-6-runbook.md) | 2026-10-25 | Runbook: phase 6 acceptance (1.0) | Plan task 6-T13, 25 October 2026. |
| [phase-6](phase-6.md) | 2026-10-25 | Phase 6 acceptance (1.0) | Verdict: pass, with one finding for the owner (F1). Every judged item passes. The clean install works without |
| [release-1.0.0](release-1.0.0.md) | 2026-10-25 | FastMACE 1.0.0 release validation | The owner authorized merging the Python 3.13 wheel, final validation, the release tag, PyPI uploads and documentation deployment. |
| [suite-speed](suite-speed.md) | 2026-10-28 | Suite speed: the CPU merge tier | Result on murphy, `-n 16`, `OMP_NUM_THREADS=2`, warm caches: 483 s before and 350–363 s after, about |
