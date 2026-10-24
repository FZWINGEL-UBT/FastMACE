# OPT4-T05: setup and capture time of an `fm.md` start

**Status: measured, one option built, 24 October 2026.** Plan `plans/opt-4.md` §3 candidate 5, task OPT4-T05,
branch `fastmace/opt4-T05`.

## Verdict

| Item | Cost per start (local) | Rule (≥ 5 s warm or ≥ 15 s cold) | Outcome |
| --- | --- | --- | --- |
| **Second compile at the first B capture** | warm 8.0–9.7 s, cold 18–24 s | passes both | **built: `EngineConfig.dynamic_edges` as an opt-in on graph-tier MD plans** (exact) |
| OEq extension build | cold 16 s, warm 0 | passes cold | not built: it is cached under `FASTMACE_CACHE` already; prebuilt wheels are SPEC §14's 1.0 plan |
| Inductor's cold compile (first compiled call) | cold 24–30 s, warm 9–10 s | passes cold | not built here; its cache lies outside `FASTMACE_CACHE` (finding 1) |
| Checkpoint load, conversion | 3 + 4 s, warm and cold | fails | none |
| Self-checks | MH-1 3.7–4.0 s warm, 7 s cold; generic 0.8 / 3 s | fails (and every check stays) | none |
| Imports, plan, private copies, output | 2.5–3 s | fails | none |

With the option, a warm start is **7–9 s shorter** and a cold start **16–22 s shorter**:

| `fm.md` start, 300 steps, process wall time | MH-1 NaCl | MH-1 NbCl5 | `omat0-medium` NaCl |
| --- | ---: | ---: | ---: |
| warm, default (n = 2) | 35.2 s | 39.7 s | 28.7 s |
| warm, `dynamic_edges=True` (n = 2) | **26.7 s** | **31.3 s** | **21.6 s** |
| cold, default (n = 3) | 86.7 s | 92.9 s | 71.7 s |
| cold, `dynamic_edges=True` (n = 2) | **65.4 s** | **70.5 s** | **55.4 s** |

The option is exact. The steps replay the same compiled code, and `reproducible=True` trajectories are bitwise
the default's. The default is the owner's call.

## Method

- **Probe:** `results/opt4-T05-probes/setup_md.py` uses 2-T19's method. It wraps CUDA-synchronised timers
  around the functions of one `fm.md` call (MD, NVT 1000 K, 300 steps, `check="skip"`) and changes nothing
  that runs. It records each stage's total and self time, dynamo's compile metrics, Inductor's cache hits and
  the engine's capture log.
- **One start per process.** The run is `results/opt4-T05-probes/run_setup.sh` and the tables come from
  `summarize.py`. Records: `results/opt4-T05-local-20261024/` (`setup.json`, log and machine state per
  process, `summary.md`).
- **Cells:** MH-1 `mh1-r2scan` (multi-head `mace-mh-1.model`) on NaCl (216 atoms) and NbCl5 (1,944 atoms),
  and the generic `omat0-medium` on NaCl. All are 3×3×3, hybrid backend, float32, every plan default.
- **Cache states:**
  - **Cold:** a fresh cache root per process, holding every cache (OEq, Inductor, Triton, XDG, TMPDIR).
  - **Warm:** one shared root, filled by one process per cell first.
  - The Inductor and Triton directories are set explicitly (finding 1). The page cache was not dropped, so
    "cold" means cold compiler caches, not a cold file system.
- **Source:** main `13a1983`, as a `git archive` export on ext4 (`~/fastmace-opt4-t05/src-13a1983`), so 2-T19's
  9p stat costs are absent. Tests ran at `4c93fd9` on this branch.
- **Machine:** the local PC (WSL2), RTX 5060 Ti, driver 610.88, `~/environments/mace_cueq_oeq` (torch 2.14.1+cu130).
  Every process ran inside a `tools/ci/local_gpu.py` reservation, 09:55–13:18 UTC.
- **Untimed.** The PC was shared with other round-4 workers' CPU work, and WSL reported a load average of
  about 210 throughout (`*.idle.txt`), with no busy process in `ps`.
  - Setup times are seconds. Their spread between repeats is 0.1–1.5 s, far below the 7–22 s effects.
  - The step time is not measured here (see "Exactness").
- **Discarded runs:**
  - The first cold set (`runs/run/cold-1-*`) shared one Inductor cache in `TMPDIR` (finding 1). Its NaCl
    process is clean and agrees with the later sets.
  - One `cold-dyn-1` set ran the default by a script error. It is kept as `cold-4`, a fourth default cold
    sample.
  - Its rerun found that set's filled roots (`reused-cold-dyn-1-*`) and is excluded.

## Stages (seconds, medians, min–max)

"Captures" is every CUDA-graph capture of the run. "Steps" is `Simulation.run` without its captures. "Rest" is
the process wall time minus the listed stages: interpreter and imports (about 1 s), plan, private copies,
output and the gaps between stages.

### Cold, default (n = 3: `cold-2`, `cold-3`, `cold-4`)

| Stage | MH-1 NaCl | MH-1 NbCl5 | `omat0` NaCl |
| --- | ---: | ---: | ---: |
| OEq runtime (extension build) | 16.4 (16.3–16.4) | 16.5 (15.9–16.6) | 16.1 (16.0–16.4) |
| load (hash, `torch.load`, e3nn model) | 3.0 (3.0–3.4) | 3.0 (2.9–3.6) | 3.5 (3.1–3.5) |
| conversion (hybrid execution model) | 3.9 (3.8–4.1) | 3.9 (3.8–3.9) | 3.7 (3.6–3.8) |
| self-checks | 6.8 (6.6–7.3) | 7.0 (6.7–7.1) | 3.1 (2.7–3.2) |
| rewrites, rest | 0.3 (0.3–0.7) | 0.3 | 0.3 (0.2–0.3) |
| **first compiled call** | **29.8** (29.7–29.8) | **29.2** (29.0–29.5) | **23.9** (23.6–24.0) |
| **captures** | **22.8** (22.6–23.0) | **24.2** (23.9–24.4) | **18.5** (18.0–18.7) |
| of which B's first capture | 22.3 (22.1–22.4) | 23.6 (23.4–23.8) | 18.1 (17.6–18.2) |
| steps (300) | 0.9 | 6.3 | 0.5 |
| rest | 3.0 (2.7–3.2) | 2.8 (2.8–3.0) | 2.8 (2.8–3.0) |
| **process** | **86.7** (86.4–88.4) | **92.9** (92.6–93.6) | **71.7** (71.7–72.8) |

### Warm, default and with the option (the `ab-1` set: off, on, on, off per cell, one shared warm root)

| Stage | NaCl off | NaCl on | NbCl5 off | NbCl5 on | `omat0` off | `omat0` on |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OEq runtime | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| load | 3.2 | 3.0 | 2.9 | 2.9 | 3.1 | 3.1 |
| conversion | 3.9 | 4.1 | 3.9 | 3.8 | 3.7 | 3.7 |
| self-checks | 3.7 | 3.9 | 4.0 | 3.9 | 0.8 | 0.8 |
| first compiled call | 10.3 | 11.4 | 10.2 | 11.0 | 9.4 | 10.2 |
| **captures** | **9.9** | **0.3** | **9.4** | **0.5** | **8.1** | **0.3** |
| of which B's first capture | 9.7 | 0.1 | 9.2 | 0.3 | 8.0 | 0.1 |
| steps (300) | 0.9 | 0.9 | 6.3 | 6.3 | 0.6 | 0.6 |
| rest | 2.9 | 2.8 | 2.6 | 2.6 | 2.6 | 2.7 |
| **process** | **35.2** | **26.7** | **39.7** | **31.3** | **28.7** | **21.6** |

- Each value is the median of 2. The ranges are within 0.5 s, except NaCl "rest" off (2.6–3.2).
- The single warm-default processes of the other sets agree: `warm-1` 35.5 / 39.6 / 29.8 s, `warm-2`
  34.2 / 39.4 / 28.2 s.

### Cold with the option (n = 2: `cold-dyn-2`, `cold-dyn-3`)

| Stage | MH-1 NaCl | MH-1 NbCl5 | `omat0` NaCl |
| --- | ---: | ---: | ---: |
| OEq runtime | 15.7 (15.3–16.1) | 15.6 | 15.8 (15.6–16.0) |
| load + conversion | 7.1 | 6.7 | 7.1 |
| self-checks | 6.6 (6.0–7.3) | 6.5 (6.4–6.6) | 2.5 (2.1–3.0) |
| first compiled call | 31.3 (30.7–31.9) | 30.8 (30.6–31.0) | 26.1 (24.8–27.3) |
| captures | 0.6 | 1.8 | 0.6 |
| steps, rest | 3.6 | 8.9 | 3.1 |
| **process** | **65.4** (63.5–67.4) | **70.5** (70.3–70.8) | **55.4** (54.2–56.5) |

## The second compile, and the option

**Cause.** `TORCH_LOGS=recompiles` (`runs/run/diag-1-mh1-NaCl.log`):
`tensor 'data['shifts']' size mismatch at index 0. expected 6912, actual 7168`.

- The graph engine's first force call (`GraphEngine._initial_forces`) runs on the exact initial edge count,
  6,912 for NaCl.
- Dynamo specialises the compiled model to it.
- `prepare()` then captures B at the padded bucket size (7,168). That guard fails, and dynamo compiles the
  whole model again, now with a symbolic edge dimension (automatic dynamic shapes).
- From then on every bucket size reuses the dynamic graph: the run's later `('B', 7680)` capture took 0.07 s.
- So every graph-tier start compiles the model twice. The first, static compile serves exactly one call.
- Warm, the second compile still costs 8–10 s: dynamo traces again and Inductor loads its cached graphs.
  Cold, it costs 18–24 s, because the backward is compiled there (`bw_compiler` 17 s of the cold NaCl
  start's metrics).

**The option.** `EngineConfig.dynamic_edges` already marks the edge dimension dynamic from the first compiled
call (2-T13, `MACEForceField.use_edges` → `compile.mark_edges_dynamic`). Evaluation and compiled-tier plans
set it. On a graph-tier MD plan it now works on request: `fm.md(..., engine=EngineConfig(dynamic_edges=True))`
or `Simulation(engine=...)`.

- The edges stay padded to buckets, and every graph is still captured.
- The first compiled call is already the dynamic compile, and B's capture reuses it (0.1 s).
- The plan records the request as `EngineConfig.dynamic_edges: on`, with the reason code
  `dynamic-edges-graph`, and it appears in `plan.requested`.
- The eager tier compiles nothing and records nothing.

**Exactness.**

- **Same compiled code.** The first process ever to start with the option (`dyn-fill-*`, in the warm root that
  only default runs had filled) had no Inductor or AOT-autograd cache miss. All 4 FX graphs and both autograd
  entries hit the entries the default's second compile had written. The steps therefore replay the same
  generated kernels as the default's.
- **The only change is the step-0 force call.** It now runs the dynamic graph instead of the static one.
- **`reproducible=True`, bitwise:** MH-1 plain cuEq float64, NaCl 3×3×3, 300 steps at 1000 K, 76 neighbour
  rebuilds. The option's positions and momenta are bitwise the default's, and a repeat is bitwise too
  (`tests/gpu/test_dynamic_edges_md_gpu.py::test_dynamic_edges_reproducible_trajectories`).
- **Hybrid float32 runs** are not bitwise even default against default (the OEq and cuEq atomics):
  - after 300 steps at 1000 K, off against off differs by up to 6e-8 Å (NaCl) and 5e-7 Å (NbCl5);
  - on against off differs by the same amounts (`summary.md`, "Final states").
- **No step-time A/B.** With the same kernels nothing per step can change, and the 300 steps took the same
  0.9 / 6.3 / 0.6 s in both. A same-process A/B (`tools/ab_rewrites.py`) is not needed for an exact,
  setup-only option. The conductor can schedule one if the owner wants it for a default.
- **First compiled call +0.8–1.1 s warm.** It now compiles the dynamic graph, which takes a little longer
  than the static one. The capture saves 8–9 s of that.

## Not built

- **OEq extension build, 16 s cold.**
  - It lives under `FASTMACE_CACHE/oeq` already. Any persistent cache root (a workstation's home, Colab's
    Drive) pays it once.
  - SPEC §14 plans prebuilt wheels for 1.0, and that is the general fix.
- **Inductor's cold compile, 24–31 s, and `torch.compiler` portable cache artifacts.**
  - Artifacts would only help a fresh machine that receives artifacts saved on an identical torch, Triton
    and GPU type.
  - A persistent `FASTMACE_CACHE` gives the same, once Inductor's cache actually lives there (finding 1).
  - So the cache location is the general fix, not an artifact format.
- **Self-checks** (MH-1 3.7–4.0 s warm, 6.5–7 s cold; `segment_gemm`'s check alone 2.3 s warm, 2.8 s cold,
  mostly its Triton autotuning). Every check stays (plan rule). 2-T19 kept the self-check cache out at
  checkpoint 2.
- **Load and conversion**, 3 + 4 s, warm and cold alike.
  - The conversion is MACE's cuEq/OEq construction from the checkpoint.
  - The template cache (OPT-T06) already removes it from every later plan in a process.
  - Below the rule for a first plan.
- **Fewer graph variants up front.** `prepare()` captures 2 graphs (ACP, B), and ACP costs 0.04–0.4 s.
  There is nothing to remove.

## Findings (reported, not fixed)

1. **Inductor's cache is not under `FASTMACE_CACHE`.**
   - `fastmace.kernels.triton.cache.configure()`, which puts `TORCHINDUCTOR_CACHE_DIR` and
     `TRITON_CACHE_DIR` under the cache root (SPEC §14 "Caches"), has no caller in `fastmace/` since 1a-T04
     added it.
   - Inductor therefore writes to its default `$TMPDIR/torchinductor_$USER`. Triton writes to
     `~/.triton/cache` unless the environment says otherwise.
   - Consequences:
     - a fresh `FASTMACE_CACHE` is not a cold compile cache, which contaminated this task's first cold set;
     - a persistent `FASTMACE_CACHE` on Colab's Drive or the NAS does not carry compiled kernels: about 20 s of
       cold compile per start with the option, about 33 s without (MH-1 NaCl, cold against warm);
     - on the workstations, the compile cache lands in `/tmp`, which AGENTS.md forbids.
   - The fix is small: call `configure()` once per process before the first compile, for example in the
     planner's `configure_process`, unless the caller set the variables. It changes where caches live, so it
     is the conductor's or owner's decision, and it needs its own tests.
2. **The double compile also affects ResearchModule-style use.** Any graph-tier plan pays it: `fm.md`,
   `Simulation`, MC and the protocols. Test suites that start many short graph-tier runs pay 8–10 s each, and
   the option would shorten them.

## Changes on the branch

- `fastmace/planner/minimal.py`: a graph-tier MD plan with a requested `dynamic_edges` and a compiling engine
  records the decision `EngineConfig.dynamic_edges: on` (`DYNAMIC_EDGES_GRAPH`, code `dynamic-edges-graph`).
  Nothing else changes: the force field already marked the edges whenever the engine config asked.
- `fastmace/config.py`: the field's comment now says what it does on each tier. Its stale "no effect until
  2-T13" is gone.
- `docs/design/switch-map.md`, `docs/rewrites.md`: the row and the screen record.
- Tests:
  - `tests/cpu/test_calc_dynamic_edges.py`: MD plans keep it on request, record the decision, and record
    nothing by default or on the eager tier.
  - `tests/gpu/test_dynamic_edges_md_gpu.py` (slow, MH-1):
    - one compile per start with the option (3 unique graphs against the default's 4, none during the steps);
    - reproducible-mode trajectories bitwise equal to the default's, and repeatable.
- `results/opt4-T05-probes/`: the probe, the run script and the summary script.

## Tests

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 nice ~/environments/fastmace_cpu/bin/python -m pytest -n 2 -q tests/cpu/test_calc_dynamic_edges.py tests/cpu/test_contracts_2.py tests/cpu/test_contracts_3.py -k "not slow"
```

102 passed. The planner suites (`test_planner.py`, `test_plan_text.py`, `test_unknown_mace_planner.py`,
`-m "not slow"`): 241 passed, 20 skipped.

GPU (local, one reservation, from the ext4 export of `4c93fd9`):

```bash
TORCHINDUCTOR_COMPILE_THREADS=4 FASTMACE_HYBRID_PYTHON=~/environments/mace_cueq_oeq/bin/python FASTMACE_MODELS=~/models PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 ~/environments/mace_cueq_latest/bin/python -m pytest -q -s tests/gpu/test_dynamic_edges_md_gpu.py
```

2 passed in 239 s. The reports are in `results/opt4-T05-local-20261024/gpu-tests/`.

## Reproduce

```bash
python3 tools/ci/local_gpu.py run --wait-minutes 600 --purpose "OPT4-T05 setup" -- bash results/opt4-T05-probes/run_setup.sh ~/fastmace-opt4-t05/run2 cold-2
```

Sets: `cold-N`, `fill`, `warm-N`, `dyn-fill`, `ab-N`, `cold-dyn-N`. Use a new N for each cold set, because a
cache root is named after its set. `SRC` selects the source export.
