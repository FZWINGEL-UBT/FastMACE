# Phase 1a acceptance report

**Status: prepared, not run.** Plan task 1a-T19. The run scripts and the
gate driver are on branch `fastmace/1a-T19`. The speed and accuracy runs
start after the planner (1a-T15) merges, and the workflow checks after
`main.py` on FastMACE (1a-T17) merges. Every "…" below is filled from the
runs. The owner then decides at checkpoint 3 whether `main.py` switches to
`--engine fastmace`.

## Verdict

| Acceptance item (plans/phase-1a.md) | Criterion | Result |
| --- | --- | --- |
| 1. Speed | steady-state MD step time within 3 % of the frozen baseline, or within the noise floor if that is larger (both reported); same-process A/B; RTX 5060 Ti and one workstation | murphy: **passes** for every valid case (MH-1 within 0.03 %, NPT 0.23 %, MP-0 medium 1.4–1.6 % slower, POLAR-1-S 0.03 %, OMOL-0 XL NbCl5 0.01 %); MH-1 NaCl hybrid and cuEq to be confirmed after the 1a-T13b fix: … ; RTX 5060 Ti: **deferred: owner reserved the local GPU (18 October 2026)** |
| 1. Setup time and peak memory | reported, not judged | murphy: FastMACE's setup 4–33 s longer (self-checks), peak memory equal or lower |
| 2. Accuracy | today's force/stress gate for all 19 labels | … |
| 3. Compatibility | `main.py --engine fastmace`: smoke, `--profile anneal` on NaCl and `--quench` on its output, statistically equal to the baseline | … |
| 4. Rules | layering test; no `gpu_langevin`-style globals; no `oeq_no_nvcc.sh` for FastMACE | … |

## Runs

| Run ID | Machine | GPU, driver | Suites | FastMACE commit | Baseline | Dates (UTC) |
| --- | --- | --- | --- | --- | --- | --- |
| … | PC (WSL2) | RTX 5060 Ti 16 GB, … | speed | … | tag `research-engine-baseline-20261017`, commit `d2faa9fd`, export `~/fastmace-baseline/1a-T18` | … |
| … | murphy or leto | RTX 4000 Ada 20 GB, 595.58.03 | speed, gate, workflow | … | same tag, export `/mnt/homes/zwingel/fastmace-baseline/1a-T18` | … |

Environments: `mace_cueq_oeq` (torch 2.14.1+cu130, MACE 0.3.16,
cuEquivariance 0.12.0, OEq 0.7.0) for every process that builds a hybrid
engine, and `mace_cueq_latest` for the gate's references and driver. On the
PC these are `~/environments/…`; on workstations they are the NAS
environments `/mnt/homes/zwingel/environments/…`. FastMACE runs from a
checkout through `PYTHONPATH` and is never installed. Hybrid FP32, TF32 off,
`torch.compile` default.

Data: `results/<run-id>/summary.json` (from `tools/acceptance/summarize.py`)
and each job's `metadata.json`. Large files (forces dumps, compile caches,
logs) stay on the machine under the run root.

## 1. Speed

Protocol (SPEC §12.5; conductor decision after wave 1b): one process per
row group, both engines built and warmed up (300 steps), then 12
alternating timed blocks of 300 steps per side. Primary set: three
processes, one ABAB, one ABAB with FastMACE built first, and one ABBA.
Both sides start from the same structure with the same seed, at 0 K, with
the thermostat at 300 K. Ratio = FastMACE median / baseline median. Pass:
|ratio − 1| ≤ max(3 %, noise floor). The noise floor of
[1a-noise-floor.md](1a-noise-floor.md) is 0.16 % for 12 blocks and 0.32 %
for 4 blocks, so the criterion is 3 %.

### RTX 5060 Ti (PC)

**Deferred: owner reserved the local GPU (18 October 2026).** The local half
of acceptance item 1 runs once the owner frees the RTX 5060 Ti, with the same
suite (`tools/acceptance/launch.sh speed --machine local`). It has not failed and
was not skipped for any other reason. The table below is filled then.

| Job | Model | Ensemble | Structure | Atoms | Baseline (ms) | FastMACE (ms) | FastMACE/baseline | Pair ratios | Criterion | Pass |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| ab-mh1-nvt-r1 (ABAB) | mh1-r2scan | NVT | NaCl 3×3×3 | 216 | … | … | … | … | ±3 % (noise floor 0.16 %) | … |
| | | | NaNbO3 3×3×3 | 1080 | … | … | … | … | | … |
| | | | NbCl5 3×3×3 | 1944 | … | … | … | … | | … |
| ab-mh1-nvt-r2 (FastMACE built first) | mh1-r2scan | NVT | NaCl / NaNbO3 / NbCl5 | | … | … | … | … | | … |
| ab-mh1-nvt-r3 (ABBA) | mh1-r2scan | NVT | NaCl / NaNbO3 / NbCl5 | | … | … | … | … | | … |
| ab-mh1-npt | mh1-r2scan | NPT, hydrostatic | NaCl 3×3×3 | 216 | … | … | … | … | | … |
| ab-mp0-medium-nvt | mp0-medium | NVT | NaCl / NbCl5 | | … | … | … | … | | … |
| ab-polar1-s-nvt | polar1-s | NVT | NaCl / NbCl5 | | … | … | … | … | | … |
| ab-omol0-xl-nvt | omol0-xl | NVT | NaCl / NbCl5 | | … | … | … | … | | … |

### Workstation: murphy

Run `1a-T19-speed-murphy-20261018`, murphy, NVIDIA RTX 4000 Ada Generation 20 GB, driver 595.58.03,
NAS `mace_cueq_oeq` (hybrid processes) and `mace_cueq_latest` (plain cuEq), 18 October 2026,
11:33–15:10 UTC. FastMACE checkouts `5de02c5` (r1), `b084660` (r2, r3, NPT), `2f88d25`/`4791ec7`
(the rest): the library code is the same in all of them, only the tools changed. Baseline
`d2faa9fd`. Data: `results/1a-T19-speed-murphy-20261018/` (`summary.json` and every job's
`metadata.json`). Murphy was otherwise idle for every timed job (idle check in `manifest.json`).

| Job | Model | Ensemble | Structure | Atoms | Baseline (ms) | FastMACE (ms) | FastMACE/baseline | Pair ratios | Valid | Pass (±3 %) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| ab-mh1-nvt-r1 (ABAB, baseline first) | mh1-r2scan | NVT | NaCl | 216 | 3.560 | 3.562 | 1.0004 | 0.9995–10.36 | **no**: compiled while timed (`mask_head`) | – |
| | | | NaNbO3 | 1080 | 26.031 | 26.021 | 0.9996 | 0.9992–1.0025 | yes | yes |
| | | | NbCl5 | 1944 | 31.305 | 31.315 | 1.0003 | 1.0000–1.0031 | yes | yes |
| ab-mh1-nvt-r2 (ABAB, FastMACE first) | mh1-r2scan | NVT | NaCl | 216 | 3.616 | 3.608 | 0.9978 | 0.9964–21.80 | **no**: compiled while timed (`mask_head`) | – |
| | | | NaNbO3 | 1080 | 26.087 | 26.081 | 0.9998 | 0.9976–1.0003 | yes | yes |
| | | | NbCl5 | 1944 | 31.312 | 31.312 | 1.0000 | 0.9962–1.0004 | yes | yes |
| ab-mh1-nvt-r3 (ABBA, baseline first) | mh1-r2scan | NVT | NaCl | 216 | 3.640 | 3.641 | 1.0002 | 0.9956–1.0030 | yes | yes |
| | | | NaNbO3 | 1080 | 26.105 | 26.097 | 0.9997 | 0.9979–1.0017 | yes | yes |
| | | | NbCl5 | 1944 | 31.435 | 31.444 | 1.0003 | 0.9982–1.0025 | yes | yes |
| ab-mh1-npt | mh1-r2scan | NPT, hydrostatic | NaCl | 216 | 3.877 | 3.886 | 1.0023 | 1.0016–1.0040 | yes | yes |
| ab-mp0-medium-nvt | mp0-medium | NVT | NaCl | 216 | 1.488 | 1.508 | 1.0139 | 0.9938–1.0148 | yes | yes |
| | | | NbCl5 | 1944 | 10.065 | 10.230 | 1.0164 | 1.0149–1.0237 | yes | yes |
| ab-omol0-xl-nvt | omol0-xl | NVT | NaCl | 216 | 14.636 | 14.555 | 0.9944 | 0.9930–0.9985 | model-limited: both trajectories reach NaN | – |
| | | | NbCl5 | 1944 | 161.759 | 161.744 | 0.9999 | 0.9996–1.0016 | yes | yes |
| ab-mh1-cueq-nvt (plain cuEq float64) | mh1-r2scan | NVT | NaCl | 216 | 96.981 | 96.980 | 1.0000 | 0.8036–1.0000 | **no**: one compile per side while timed | – |

POLAR-1-S, alternating single-engine processes (the engines cannot share a process, see below),
12 blocks each, order baseline, FastMACE, FastMACE, baseline, baseline, FastMACE. The ratio is the
median of FastMACE's process medians over the baseline's. Its noise floor is the larger relative
spread of the per-process medians (0.29 %), so the criterion stays 3 %:

| Model | Structure | Baseline medians (ms) | FastMACE medians (ms) | FastMACE/baseline | Noise floor | Valid | Pass |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| polar1-s | NaCl | 3.995, 4.007, 4.006 | 4.008, 4.006, 4.008 | 1.0003 | 0.29 % | yes | yes |
| polar1-s | NbCl5 | 31.559, 31.633, 31.650 | 31.625, 31.643, 31.653 | 1.0003 | 0.29 % | yes | yes |

Reading:
- **MH-1, the primary set, is at parity.** Every valid ratio lies within 0.03 % of 1 (NaNbO3
  0.9996–0.9998, NbCl5 1.0000–1.0003, NaCl 1.0002). That is inside the 0.16 % noise floor of
  1a-T18, in both build orders and both patterns. NPT: 1.0023.
- **MP-0 medium is 1.4–1.6 % slower** (pair ratios 0.994–1.024). That passes the 3 % criterion but
  is ten times the noise floor, so it is a real difference for the screens after T19.
- **OMOL-0 XL on NaCl is model-limited.** Both engines' trajectories fall apart (edge counts down
  to 512, NaN temperature), in line with ResearchModule's screen (P = −20.7 GPa for unstrained
  NaCl). The timing of a broken trajectory says nothing about the engines.
- **Plain cuEq float64** gives 1.0000 but is invalid: each side compiled once while timed
  (`stats.unique_graphs` 3 → 5), probably the same `mask_head` interaction. Its rerun belongs with
  the post-fix NaCl rerun.
- **Still open:** the MH-1 hybrid NaCl NVT A/B and the plain cuEq NaCl A/B, rerun on murphy with
  1a-T13b's `mask_head` fix, after it merges: …

### The two engines clash on Inductor's custom-pass slot

Both engines keep Inductor's `mm + full-matrix add` unfused on the GPU.
Each also puts a no-op pass into Inductor's single
`post_grad_custom_pre_pass` config slot, where the pass's uuid keys the
compiler cache:
- the research engine: `gpu_langevin`, "langevin-unfused-full-addmm-v2";
- FastMACE: `_patches.set_unfuse_full_addmm`,
  "fastmace-unfused-full-addmm-v1".

Each engine refuses a slot that holds the other's key. The engines
therefore **cannot share a process unchanged**, in either build order. The
first A/B smoke run on murphy (18 October 2026) failed with "Inductor's
post_grad_custom_pre_pass is already in use".

The conductor chose a fix in the measuring tool only, with no library
change ("option (a)"). `tools/ab_step_time.CustomPassSlot` puts each side's
own key into the slot before that side loads, builds and warms up, and
keeps what the side left there:
- Each side's 300 warm-up steps, where it compiles and captures, run right
  after its build.
- The 12 timed blocks then alternate with no warm-up of their own.
- The key matters only at compile time. Dynamo's compile counters are
  therefore recorded before and after the timed blocks
  (`compile_during_timing`). A run in which anything compiled while timed
  is invalid; it is rerun, or that crystal falls back to alternating
  processes.
- Because the second engine's key differs from a production run's, the
  build order is swapped in one repeat per crystal (`ab-mh1-nvt-r2`).

In every A/B process each side kept its own key: baseline "langevin-unfused-full-addmm-v2",
FastMACE "fastmace-unfused-full-addmm-v1" (`custom_pass_uuids`). Counters changed while timed only in
these runs:

| Job | Structure | Counters changed while timed | Cause | Valid |
| --- | --- | --- | --- | --- |
| ab-mh1-nvt-r1 | NaCl | `unique_graphs` 5 → 6, `frames.ok` 10 → 11 | FastMACE recompile at the first new bucket (`mask_head`, below) | no |
| ab-mh1-nvt-r2 | NaCl | `unique_graphs` 5 → 6, `frames.ok` 10 → 11 | same | no |
| ab-mh1-cueq-nvt | NaCl | `unique_graphs` 3 → 5, `frames.ok` 4 → 6 | one compile per side at the first new bucket | no |

Everything else, including NbCl5's two bucket changes (51200 → 53760 → 56320) in every run, compiled
nothing while timed.

### Other same-process interactions found during the runs

- **`mask_head` capture patch (r1 and r2 NaCl invalid).** Both engines
  patch `mace.modules.blocks.mask_head` process-wide with a capture-safe
  version. The research engine reinstalls its function before every
  capture. FastMACE installed its own only once. After a research-engine
  capture, FastMACE's compiled model therefore failed its dynamo guard at
  the next eager call. That call was the warm-up of its first capture at a
  new edge bucket: NaCl's 7168 → 7680 transition, in timed block 4.
  - The result was one recompile while timed: 38.0 ms/step in r1 and
    80.5 ms/step in r2 for that block, against about 3.6 ms/step otherwise.
  - Under the no-compile rule, NaCl in r1 and r2 is invalid.
  - r3's NaCl, where the research engine did not recapture before
    FastMACE's transition, is valid.
  - Each engine alone, with fresh caches, does not compile there: NaCl's
    exact initial count (6912) and its padded size (7168) make the edge
    dimension dynamic during warm-up.
  - Root cause found by 1a-T13b. The fix, branch `fastmace/1a-T13b`
    `ae3e758`, makes FastMACE reinstall its function before each capture.
    One NaCl A/B rerun on murphy with that fix follows its merge: …
- **PolarMACE patches (POLAR-1-S A/B impossible in one process).** The
  research engine's `polar_gpu` replaces MACE's and graph_longrange's
  PolarMACE functions process-wide. FastMACE's source qualification then
  correctly refuses its GPU-only polar path ("unqualified
  MACE/graph_longrange source").
  - Swapping the build order would not help: the research engine would
    replace the functions under FastMACE's engine.
  - POLAR-1-S speed therefore comes from alternating single-engine
    processes (baseline, FastMACE, FastMACE, baseline, baseline, FastMACE;
    12 blocks each).
  - Its noise floor is the spread of the per-process medians.
- **A bug in the A/B tool, fixed (`2f88d25`).** The tool's list of
  `GPUEngine` options, inherited from 1a-T18, lacked `species_skip`. The
  baseline side therefore ran MP-0 medium without `main.py`'s
  species-restricted skip, and FastMACE appeared 12–15 % faster. MH-1 does
  not support the option, so the MH-1 results and the 1a-T18 noise floor
  stand. The MP-0 run is kept as `ab-mp0-medium-nvt.invalid-species-skip`.
  The tool now passes every option `main.py` passes, and a test pins the
  list.

### Interference checks in the A/B processes

- **Same trajectory.** After 300 + 12 × 300 steps the two sides' temperatures differ by at most
  2.4e-4 K (MH-1; MP-0 4.1e-5 K) and their potential energies by at most 2.0e-5 eV/atom. That is the
  FP32 divergence of a chaotic trajectory, the same size as in 1a-T18's baseline-against-baseline
  runs. Plain cuEq float64: 1.2e-10 K and 8e-15 eV/atom. Both sides used the same edge sizes. The
  research engine counts one more neighbour rebuild in every run (155 against 154 on NaCl, 207
  against 206 on NaNbO3), so its count apparently includes the initial build. OMOL-0 XL NbCl5
  diverges more (1.5e-2 K, 1.0e-3 eV/atom); its forces scatter 100 times more between processes
  (next item).
- **FastMACE's forces in an A/B process against a FastMACE-only process**
  (decision after wave 1b; both engines patch MACE process-wide). These are
  the starting forces of the engine's initial evaluation, the same
  structure in both processes. Tolerance 1e-5 eV/Å (the gate's force
  floor):

  | A/B job | Structure | Max \|ΔF\| against the FastMACE-only control (eV/Å) | Pass |
  | --- | --- | ---: | --- |
  | ab-mh1-nvt-r1 / r2 / r3 | NaCl | 2.8e-7 / 3.1e-7 / 2.6e-7 | yes |
  | | NaNbO3 | 2.4e-6 / 2.1e-6 / 2.5e-6 | yes |
  | | NbCl5 | 1.8e-6 / 1.5e-6 / 1.8e-6 | yes |
  | ab-mh1-npt | NaCl | 6.2e-7 | yes |
  | ab-mp0-medium-nvt | NaCl / NbCl5 | 3.6e-7 / 1.7e-6 | yes |
  | ab-mh1-cueq-nvt (float64) | NaCl | 7.0e-16 | yes |
  | alt-polar1-s p1 / p2 / p3 (FastMACE-only processes) | NaCl | 1.6e-6 / 2.0e-6 / 1.6e-6 | yes |
  | | NbCl5 | 5.1e-7 / 4.2e-7 / 3.7e-7 | yes |
  | ab-omol0-xl-nvt | NaCl / NbCl5 | 9.1e-5 / 1.6e-4 | above 1e-5, but within OMOL-0 XL's process scatter |

  For OMOL-0 XL a second FastMACE-only process (`scatter-omol0-xl-fastmace-2`) differs from the
  first by 1.2e-4 (NaCl) and 1.8e-4 eV/Å (NbCl5). That is as much as the A/B process differs from
  either (0.8–2.0e-4). So this is FP32 process-to-process scatter of the 1024-channel model (its
  reassociation self-check is already 7e-5 relative), not interference. No process changed
  FastMACE's starting forces beyond what a second FastMACE-only process does.

- **FastMACE's plan in the A/B process** equals the plan of the
  FastMACE-only process (`plans` in `metadata.json`; the rewrite set must
  not change because the research engine patched MACE first): equal in every pair checked,
  including OMOL-0 XL. The rewrite lists and FastMACE's process patches (`oeq_runtime`,
  `capture_safe_mask_head`, `unfuse_full_addmm`, `single_graph_strain`) are identical too.
- **Dynamo counters and other GPU processes:** the counters are in the table above. No other
  compute process was on the GPU after any structure (`other_gpu_processes` empty in every job).

## Setup time and peak memory

Single-engine processes (`--sides baseline`, `--sides fastmace`, SPEC
protocol of 4 blocks), because an A/B process mixes both engines (decision
after wave 1b). Load = checkpoint loading and conversion. Setup = engine
construction (rewrites, compilation is lazy in both engines and lands in
the warm-up). Peak memory = `torch.cuda.max_memory_reserved()` per
structure, after warm-up and timing. FastMACE-only processes run without
the hybrid launcher.

murphy, run `1a-T19-speed-murphy-20261018`. Step times from single processes are a cross-check
only (cross-process scatter); the A/B tables above carry the speed verdict.

| Model | Structure | Load (s) baseline / FastMACE | Setup (s) baseline / FastMACE | Peak memory (GiB) baseline / FastMACE | Step (ms) baseline / FastMACE |
| --- | --- | --- | --- | --- | --- |
| mh1-r2scan NVT | NaCl | 23.8 / 24.0 | 27.3 / 45.2 | 0.34 / 0.32 | 3.492 / 3.519 |
| | NaNbO3 | | 28.8 / 42.5 | 1.54 / 1.52 | 25.717 / 25.832 |
| | NbCl5 | | 28.5 / 36.9 | 1.31 / 1.31 | 30.934 / 30.989 |
| mh1-r2scan NPT | NaCl | 4.5 / 23.4 | 27.6 / 31.5 | 0.34 / 0.32 | 3.807 / 3.823 |
| mp0-medium | NaCl | 23.6 / 22.8 | 20.8 / 29.1 | 0.25 / 0.23 | 1.453 / 1.469 |
| | NbCl5 | | 19.2 / 24.1 | 0.96 / 0.94 | 9.880 / 10.115 |
| polar1-s | NaCl | 23.2 / 23.2 | 25.1 / 50.9 | 0.27 / 0.25 | 3.952 / 3.948 |
| | NbCl5 | | 48.0 / 47.9 | 0.83 / 0.81 | 31.422 / 31.454 |
| omol0-xl | NaCl | 22.2 / 22.0 | 36.0 / 68.6 | 1.35 / 1.27 | 14.450 / 14.323 |
| | NbCl5 | | 36.8 / 59.0 | 4.68 / 4.66 | 162.643 / 161.710 |
| mh1-r2scan cuEq float64 | NaCl | 6.3 / 6.2 | 20.8 / 41.3 | 0.83 / 0.83 | 95.505 / 95.255 |

- Load: checkpoint and conversion. The baseline's MH-1 NPT load (4.5 s) hit the warm OEq extension
  of the same process type; the research engine records its OEq JIT in load, FastMACE in its
  in-process runtime.
- Setup: FastMACE's force-field construction runs the per-rewrite self-checks of SPEC §6.4, the
  research engine has none. It takes 4–33 s longer, most for the MH-1 package and the large models.
- Peak memory: equal or lower for FastMACE in every case (up to −0.08 GiB).
- No FastMACE-only process compiled while timed, NaCl's 7168 → 7680 transition included. That
  confirms 1a-T13b's diagnosis.
- `hybrid_launcher` in these processes' `metadata.json` reads true because the flag was read after
  FastMACE's in-process OEq runtime had set the same variables (fixed in `d7773c6`). Their
  commands in `manifest.json` show no launcher.

## 2. Accuracy: force/stress gate, 19 labels

`tools/force_stress_gate.py --engine fastmace`, one run per label. Three
configurations per structure and cell mode (hydrostatic, full): ±2 %
strain, 0.05 Å rattle, seed 20261015. The rule: the production error
against float64 must be at most max(2 × the float32 calculator's error,
floor), with floors of 1e-5 eV/Å for forces and 1e-7 eV/Å³ for stress
(SPEC §12.3). Energy per atom is reported, not gated (1b).

**Processes (rewritten in 1a-T19).** The driver builds no model. Every
float64 and float32 reference runs in its own fresh process (one per
structure and calculator, in `mace_cueq_latest`). That process builds no
production engine and evaluates its configurations one at a time. All
references run before the production process (FastMACE through the
planner, in `mace_cueq_oeq`, without the launcher). This follows the rule
"after T10". There, a research-engine process that had built the hybrid
engine first moved the float64 POLAR-1-S reference on NbCl5 by up to
7.8e-7 eV/Å and 5.4e-9 eV/atom. Run alone, the same reference equals
FastMACE and stock MACE to 1e-13. Gate results from before this rewrite
count as provisional ("Second outside review").

**The old gate's polar references were correct only by accident.** The
references' PolarMACE stress needs the k-space correction for MACE issue
1642 (SPEC §12.1). The research engine's `polar_gpu` installs it on the
class at the first `enable()`. `workflow.configure_md_calculator` stores
each model's bound `forward`. The old gate called it before `enable()`,
yet its references were corrected anyway: they shared a process with the
production engine, whose setup had already installed the class patch.

1a-T19's fresh reference processes had no such engine. Their first run
called `configure_md_calculator` first, so both references, float64 and
float32, silently computed MACE 0.3.16's uncorrected stress.
- Every POLAR row then failed on stress: NaCl 3.0–4.6e-5 eV/Å³, the size
  of the uncorrected error.
- Two production engines that agree with each other both failed. FastMACE
  gave 4.6e-5, and so did the research engine
  (`gate-polar1-s-NaCl-baseline.uncorrected-reference`).

Fixed in `5e5019d`: `enable()` first, and the reference raises unless the
corrected `forward` is the active one. Since `e05dd55`, the first float64
reference of each run also checks its stress against central finite
differences of its energy, for PolarMACE: strain ±1e-4, six Voigt
components, tolerance 1e-6 eV/Å³. That tolerance lies between the corrected
path's 7.5e-9 and the uncorrected 2e-5 on NaCl. The driver also records the
production stress against the same finite differences:

| Run | Configuration | Reference − FD (eV/Å³) | Production − FD (eV/Å³) |
| --- | --- | ---: | ---: |
| … | … | … | … |

The invalid runs are kept as `*.uncorrected-reference`.

| Label | Structures | Worst force ratio | Worst stress ratio | Max energy error, production / float32 (meV/atom) | Pass |
| --- | --- | ---: | ---: | --- | --- |
| mp0-medium | NaCl, NaNbO3, NbCl5 (3×3×3) | … | … | … | … |
| mp0-large | | … | … | … | … |
| mp0b3-medium | | … | … | … | … |
| mpa0-medium | | … | … | … | … |
| omat0-medium | | … | … | … | … |
| matpes-pbe | | … | … | … | … |
| matpes-r2scan | | … | … | … | … |
| mh1-omat | | … | … | … | … |
| mh1-mp | | … | … | … | … |
| mh1-r2scan | | … | … | … | … |
| polar1-s | | … | … | … | … |
| polar1-m | | … | … | … | … |
| polar1-l | | … | … | … | … |
| omol0-xl | | … | … | … | … |
| off23-small | urea, naphthalene, paracetamol (3×3×3), water box (648 atoms, unrepeated) | … | … | … | … |
| off23-medium | | … | … | … | … |
| off23-large | | … | … | … | … |
| off23b-medium | | … | … | … | … |
| off24-medium | | … | … | … | … |

## 3. Compatibility: `main.py` on FastMACE

**RTX 5060 Ti (PC): from 1a-T17.** Following the conductor decision after
T17, these runs are not repeated here. ResearchModule main `fb81217b` ran
`main.py --engine fastmace` against FastMACE main `43ce026`, on the RTX
5060 Ti (WSL2, `mace_cueq_oeq`). The workload was hybrid FP32 MH-1
(`matpes_r2scan`) on NaCl 3×3×3: a shortened anneal (2 ps heat, 25 ps
cool, 54,000 steps) and `--quench` on its output, two seeds per engine,
each against `--engine gpu`.

The data are in `results/1a-T17-workflow-20261018/`: T17's `run.json`,
metadata, temperature and quench CSVs, logs and scripts, without the
trajectories. `comparison.txt` is the output of `scripts/t17_compare.py`,
which uses the measures of ResearchModule's `studies/annealing/README.md`.

| Measure | `--engine gpu` | `--engine fastmace` | Equal? |
| --- | --- | --- | --- |
| smoke (60 steps), largest position / energy difference | – | 7e-9 Å, 2.3e-6 eV/atom; wall time 23.2 s against 23.1 s | yes |
| MD potential energy per 100 K bin, both ramps (16 bins), max \|Δ\| between engines, same seed | – | 0.01 meV/atom (both seeds) | yes: between seeds 5.93 meV/atom for either engine |
| peak (meV/atom, at target K), seed 20261001 / 20261101 | +143.6 (961) / +135.0 (923) | +143.6 (961) / +135.0 (923) | yes |
| neighbour rebuilds, seed 20261001 / 20261101 | 3522 / 3556 | 3522 / 3556 | identical |
| quench, heating ramp (eV/atom over the relaxed start, n = 10) | +0.0140 ± 0.0039 / +0.0124 ± 0.0018 | +0.0140 ± 0.0038 / +0.0125 ± 0.0016 | within errors |
| quench, cooling ramp | +0.0119 ± 0.0013 / +0.0122 ± 0.0019 | +0.0121 ± 0.0014 / +0.0123 ± 0.0024 | within errors |
| MD wall time (s) | 211 / 210 | 212 / 212 | – |
| peak GPU memory (GB) | 0.46 | 0.49 | reported |

The full 270 ps anneal through FastMACE was not run. The conductor decision
after T17 leaves it to checkpoint 3, if the owner asks for it.

**Workstation half:** one `main.py --engine fastmace` smoke run on leto,
after the gate, with the NAS FastMACE copy and its `.source_commit`: …

## 4. Rules

- Layering test (`tests/cpu/test_layering.py`) on the tested commit: …
- No module-level switches: …
- No launcher: every FastMACE-only process above (controls, gate
  production) ran without `oeq_no_nvcc.sh` (`hybrid_launcher: false` in
  their `metadata.json`). The A/B processes use the launcher only because
  the research engine needs it.
- In-process OEq runtime on the workstations (decision after wave 0): a
  FastMACE-only hybrid process on coraline and on leto, before T17's
  default switch: …

## Reproducing

From a FastMACE checkout on the machine (`$FM`; on workstations
`/mnt/homes/zwingel/FastMACE`, synced to the tested commit), with an idle
GPU and nothing else running:

```bash
bash $FM/tools/acceptance/launch.sh speed --machine local --run-id 1a-T19-speed-pc-<date>
bash $FM/tools/acceptance/launch.sh speed gate --machine murphy --run-id 1a-T19-murphy-<date>
bash $FM/tools/acceptance/launch.sh workflow --machine murphy --run-id 1a-T19-workflow-<date> \
  --research-module /mnt/homes/zwingel/<ResearchModule tree with T17>
# in-process OEq runtime on coraline and leto (decision after wave 0): one FastMACE-only process each
bash $FM/tools/acceptance/launch.sh speed --machine coraline --run-id 1a-T19-oeq-coraline-<date> \
  --only control-mh1-nvt-fastmace
python3 $FM/tools/acceptance/summarize.py <root>/<run id> --output $FM/results/<run id>/summary.json
```

`tools/acceptance/run.py … --dry-run` lists every job and its command. Each
job checks that the GPU has no compute process before it starts. Timing jobs
also check that GPU utilization is at most 5 %; on the PC that check also
uses Windows' `nvidia-smi.exe`, which sees desktop programs. A busy GPU
stops the run.
