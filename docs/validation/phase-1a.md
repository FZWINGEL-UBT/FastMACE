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
| 1. Speed | steady-state MD step time within 3 % of the frozen baseline, or within the noise floor if that is larger (both reported); same-process A/B; RTX 5060 Ti and one workstation | … |
| 1. Setup time and peak memory | reported, not judged | … |
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

### Workstation (murphy or leto)

Same table.

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

| Job | Structure | Keys (baseline / FastMACE) | Counters changed while timed | Valid |
| --- | --- | --- | --- | --- |
| … | … | … | … | … |

### Interference checks in the A/B processes

- **Same trajectory.** Final temperature and potential energy of the two
  sides, and their rebuild counts and edge sizes: …
- **FastMACE's forces in an A/B process against a FastMACE-only process**
  (decision after wave 1b; both engines patch MACE process-wide). These are
  the starting forces of the engine's initial evaluation, the same
  structure in both processes. Tolerance 1e-5 eV/Å (the gate's force
  floor):

  | A/B job | Control | Structure | Max \|ΔF\| (eV/Å) | Bitwise | Pass |
  | --- | --- | --- | ---: | --- | --- |
  | … | … | … | … | … | … |

- **FastMACE's plan in the A/B process** equals the plan of the
  FastMACE-only process (`plans` in `metadata.json`; the rewrite set must
  not change because the research engine patched MACE first): …
- **Dynamo counters and other GPU processes:** …

## Setup time and peak memory

Single-engine processes (`--sides baseline`, `--sides fastmace`, SPEC
protocol of 4 blocks), because an A/B process mixes both engines (decision
after wave 1b). Load = checkpoint loading and conversion. Setup = engine
construction (rewrites, compilation is lazy in both engines and lands in
the warm-up). Peak memory = `torch.cuda.max_memory_reserved()` per
structure, after warm-up and timing. FastMACE-only processes run without
the hybrid launcher.

| Machine | Job | Engine | Model | Structure | Load (s) | Setup (s) | Peak memory (GiB) | Step (ms) |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| … | control-mh1-nvt-baseline | baseline | mh1-r2scan | NaCl | … | … | … | … |
| … | control-mh1-nvt-fastmace | fastmace | mh1-r2scan | NaCl | … | … | … | … |

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
