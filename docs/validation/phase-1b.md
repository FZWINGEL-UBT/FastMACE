# Phase 1b acceptance report

**Status: tools ready, runs pending.** Plan task 1b-T23, branch `fastmace/1b-T23`, part 1 (tools,
19 October 2026). Part 2 runs the checklist below on the GPUs the conductor's go message names,
after 1b-T20, T21 and T22 are merged, and fills in every number. The template is
[phase-1a.md](phase-1a.md).

## Acceptance checklist

Machines (conductor answer to the prestart report):

- leto only when cam's job has gone; otherwise murphy for the gate and the GPU suite;
- the RTX 5060 Ti for its gate subset;
- timing only on an otherwise idle machine, after an idle check before each job (no compute process
  of anyone; `nvidia-smi --query-compute-apps=pid` plus `ps -o user= -p`).

Suites are in `tools/acceptance/suites.py` (`launch.sh SUITE --machine NAME --run-id ID`).

| # | Item | Machine | Tool (suite) | Pass criterion |
| --- | --- | --- | --- | --- |
| 1a | Gate with energy, 19 labels, on the structures that count (1a rule) | murphy (leto if free) | `force_stress_gate.py --engine fastmace --planner minimal` (`gate-1b`): `fastmace.reference` float64 reference and float32 yardstick with float64 energy, each in a fresh process before production; 3 configurations per structure and cell mode; MACE-OFF on the organic set, water box unrepeated | Per label, energy, forces and stress all pass: production error ≤ max(2 × yardstick error, floor). Floors: 0.01 meV/atom, 1e-5 eV/Å, 1e-7 eV/Å³. Nothing non-finite. Polar references pass the finite-difference self-test (< 1e-6 eV/Å³) |
| 1b | Energy headline | as 1a | as 1a | POLAR-1 S/M/L and OMOL-0 XL production energies within the gate (1a: 45–102 meV/atom off on NbCl5, 2.4–95 on the counting structures); MACE-OFF within the gate (1a: 3.9–15). POLAR-1-S on NbCl5 stays supplementary |
| 1c | In-library reduced gate, once per label | murphy (leto if free) | `library_gate.py` (`library-gate`): `Simulation(check="gate")`, NVT and NPT (hydrostatic), NaCl (urea for MACE-OFF), captured replay | `status == "passed"` for all 19 labels; gate time recorded ("seconds to a minute") |
| 1d | Gate subset | RTX 5060 Ti | `gate-1b-subset`: MH-1, POLAR-1-S, OMOL-0 XL, OFF24-M | as 1a |
| 2 | Registry and capture-state checks | murphy (leto if free), then RTX 5060 Ti | full GPU suite in one process (`gpu-suite`, T24's full tier) | `test_registry_gpu.py` passes for MH-1 hybrid pipelined NVT, MH-1 NPT BAOAB (full cell), POLAR-1-S and plain cuEq float64; the whole suite green |
| 3a | Restart round trip and bitwise CPU restart | local CPU (`fastmace_cpu`) | CPU suite (T05, T06, T14 tests) | save → load → save identical; interrupted = uninterrupted bitwise across a partly used noise batch, h ≠ h₀ at the restart step, a count-overflow redo before it |
| 3b | Plain cuEq float64 restart on the GPU | as 2 | `gpu-suite` (`test_canonical_restart_gpu.py`) | within 1e-12 Å over 500 steps, restarts every 100 (T14: 3.7e-13) |
| 3c | Hybrid restart statistics | RTX 5060 Ti | `gpu-suite` (`test_mh1_hybrid_resumes_statistically`) | T and E_pot means and bin means over 20,000 steps, two seeds: resumed − uninterrupted well inside the seed scatter (T14: 1e-5 K against 2.7 K) |
| 3d | Kill between a restart write and the next flush | local CPU | CPU suite (T06 SIGKILL test) | no duplicate or missing frames or rows; later chunks deleted, nothing rewritten |
| 3e | **Compiled PolarMACE NPT across restarts** (left by T14) | murphy | `compile_checks.py polar-npt` (`compile-checks`): POLAR-1-S NaCl 3×3×3 BAOAB, 5 restart points; resume in a fresh process | no recompile at a canonical restart point (or recorded with its cause); the resumed run completes, and its deviation is reported |
| 4a | CLI without Python | local GPU, clean checkout (and the CPU CI `.[full]` job) | `fastmace doctor`, then `fastmace md` | a complete run with plan, gate (`--check gate`) and speed report |
| 4b | Three front ends agree | murphy, idle | `ab_frontends.py` (`frontends`): `Simulation`, `fm.md`, `fastmace.ase.Langevin` on MH-1 NaCl 3×3×3 hybrid NVT, 12 round-robin blocks of 300 | the same `Plan` (friction within 1 ulp) and the same gate record; step-time ratios within the noise floor (0.16 % for 12 blocks) |
| 4c | The other §7.7 items | CPU suite; one ASE script on the GPU | T17, T20, T21, T22 tests: the ASE port, observer under D4, signatures, trajectory and log read-back, the `d3bj` refusal, not-graph reasons | tests green; a ported ASE Langevin script runs on the graph engine with the plan of the equivalent `Simulation` |
| 5 | No regression against tag `fastmace-1a` (`bb35363`) | murphy and RTX 5060 Ti, idle | `ab_fastmace.py` (`regression`): 1b against the renamed 1a copy in one process, MH-1 NaCl and NbCl5, hybrid NVT, 12 blocks; ABAB, 1b built first, ABBA; then one control per version | \|1b/1a − 1\| ≤ noise floor in every valid process; valid means nothing compiled, no patch changed while timed, equal engine options; setup time per stage (gate included) reported |
| 5b | **MinimalPlanner against FixedPlanner, 500 steps** (left by T12) | as 2 | `tests/gpu/test_minimal_planner_gpu.py` (in `gpu-suite`) | plain cuEq float64 within 1e-12 Å; hybrid within 3 × the scatter of two `FixedPlanner` runs; equal records, options, edge sizes and rebuilds |
| 5c | **`fm.calc` recompile** (T18 finding, measured only) | murphy | `compile_checks.py calc` (`compile-checks`), hybrid and plain cuEq, NaCl 2×2×2, 6 calls | reported: which calls compile, and their wall time (T18: a second compile, 4.5 s cuEq, 19 s hybrid) |
| 6 | `main.py` compatibility | local (adapter tests), murphy or leto (smoke from the NAS tree) | `adapter-tests`: ResearchModule `tests/core/test_fastmace_engine.py` against 1b's main, read only (`PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`, `--basetemp` outside the tree); `workflow` suite smoke | the adapter tests pass, or fail only on the energy tolerance, by D2's measured shift (**D2 energy shift in the adapter tests**: 0.005 meV/atom on NaCl, 0.12 on NbCl5, recorded); the smoke run equals `--engine gpu` to float32 divergence (1a: 1.4e-4 K, 1.6e-5 eV/atom), with the energy offset the float64 sum alone |
| 7 | Rules | local CPU | `test_layering.py`, the exception hierarchy tests, `ruff` | pass; no new module-level switches; every public exception derives from `FastMACEError` with the §7.7 message shape |
| – | coraline soft-lock control (not acceptance) | coraline, only if idle | `launch.sh speed --machine coraline --only control-mh1-nvt-fastmace` | the conductor runs it and records it here |

## Tools (part 1)

- **`tools/force_stress_gate.py`**, rebuilt.
  - The references are `fastmace.reference`'s (1b-T13): the float64 reference, and the float32
    yardstick with float64 energy accumulation. The label resolves to `--checkpoint`'s file, and its
    SHA-256 is the trust decision.
  - The rule is `selfcheck.verdict` (1b-T16), with energy gated.
  - Production goes through the minimal planner (`--planner minimal`, the default) or main.py's
    fixed one.
  - `--structures-root` replaces the research-engine export for the structures. `--engine baseline`
    still builds the research engine from `--baseline`; its energy is reported, not gated (float32
    sum).
  - Unchanged from 1a: references in fresh processes before production, the finite-difference
    self-test for PolarMACE on NaCl 3×3×3, and a driver that imports neither torch nor MACE.
- **`fastmace/bench/versions.py`**: the 1a release beside 1b in one process (conductor answer 1).
  - `export_renamed` writes the tag's `fastmace` package as `fastmace_1a`. It renames every
    standalone word `fastmace` in its Python files: imports, module-name strings, the `fastmace::`
    custom-op namespace (a second registration of the same names would fail) and the Inductor cache
    key. It records the commit and a tree digest.
  - `ProcessPatches` generalises `ab_step_time.CustomPassSlot` to every object both versions patch
    process-wide: `mask_head`, `compute_forces(_virials)`, the one-graph strain, the unfused-addmm rule
    and the Inductor custom-pass slot, and the cuEq conversion. Each side gets MACE's own objects
    before its first load (so its qualification hashes MACE's source), then its own, before every
    load, build, warm-up and timed block. That swap runs outside the clock: `run_ab(before=...)`.
  - `share_record` gives the 1a copy this process's OEq runtime. There is one host extension per
    process, and 1a's loader would refuse the hybrid once MACE is imported.
  - PolarMACE (a class patch), plain cuEq and NPT are refused with two sides. They need alternating
    processes, one version each.
  - Checked on the CPU: the real `fastmace-1a` export imports beside 1b, custom-op modules included,
    and the two keep separate custom-op namespaces (`torch.ops.fastmace`, `torch.ops.fastmace_1a`).
- **`tools/ab_fastmace.py`**: 1b against 1a.
  - It records each side's plan and patches, Dynamo's compile counters and any patch change while
    timed, and computes the ratio 1b / 1a.
  - A run counts as valid only when nothing compiled, no patch changed and the engine options are
    equal.
- **`tools/ab_frontends.py`**: `Simulation`, `fm.md` and `fastmace.ase.Langevin`, round-robin blocks,
  each block one `run(n)`. The `fm.md` and ASE sides wait for 1b-T20 and 1b-T21 and stop naming them
  until then.
- **`tools/library_gate.py`**: the in-library gate per label, NVT and NPT, each in a fresh child
  process.
- **`tools/compile_checks.py`**: compiled PolarMACE NPT across restart points, and `fm.calc`'s
  compiles.
- **`tests/gpu/test_minimal_planner_gpu.py`**: T12's 500-step GPU comparison.

Prepare the renamed copy once per machine:

```bash
python -m fastmace.bench.versions export --repository /mnt/d/Margraf/FastMACE --ref fastmace-1a --destination ~/fastmace-versions
```

For a workstation, export it on the PC and copy it to `/mnt/homes/zwingel/fastmace-versions/`. The NAS
checkout has no git. `python -m fastmace.bench.versions check --destination DIR --expected-commit bb35363`
verifies it.

## Open risks for part 2 (GPU)

1. **Two versions in one process.** It is untested on a GPU so far. These must hold:
   - 1a's allocator rule must accept being applied second;
   - the shared OEq runtime must build 1a's streaming kernels;
   - no side may compile while timed.

   The tool records all three. If any fails, item 5 falls back to alternating processes, with the
   cross-process noise, and the report says so.
2. **The front-end harness's `fm.md` and ASE builders are written against the plan.** `fm.md` follows
   T20's pushed branch (`b689017`). The ASE side's keyword names (`force_field=`, `output=`) follow
   §7.7 level 1. Both are re-checked after T20 and T21 merge.
