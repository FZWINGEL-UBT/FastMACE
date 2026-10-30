# Engine setup across compositions (float32, 30 October 2026)

Owner request via the conductor, 30 October 2026. Energy_Hull-type work relaxes many phases of different
composition (101 competing phases in 22 species sets). Every new composition builds its own engine, and the
same-composition reuse of [relax-throughput](relax-throughput.md) does not help there. This check measures
each build stage by stage, finds what changes between compositions and what is cached, and answers the
CUDA-graph update question. It is measurement and inspection only: no library code or default changed.

## Summary

- **A new composition costs 16.3 s** (median) in a process that has already built one, with a warm compile
  cache. It costs **22.3 s** in a fresh process with a warm cache, and **49.7 s** in a fresh process with an
  empty cache. The model load comes on top, once per process: 3.6 s warm, 19.8 s cold, which includes the
  OEq extension JIT build.
- **New composition, same process, cache cold for it: 36.6 s** (26.7–40.5). This is the case of a new
  Energy_Hull phase set. Each composition generates new Inductor code (150–190 files) and new Triton
  kernels (up to 1,600 files), and fits its own radial tables (3.4–11.3 s per species set).
- **What changes between compositions in the generated code:**
  - the radial-table row counts, as static shapes (53 against 52 rows for NbCl5 and TaCl5);
  - an integer constant passed to `fused_folded_forward` (9 against 11);
  - the atom count, as a static shape or as a size hint;
  - the species count, through the rewrite structure: joined products for 2 species, grouped products for
    3.

  The OEq NVRTC sources (5) and the fused-forward sources (2) are identical across all eight compositions.
  TaCl5 after NbCl5 (same atom count and species count, different tables) generated no new Triton binaries
  and built in 26.7 s against 36–40 s for the others. That is the gain on offer if the per-composition
  constants became data.
- **Same composition, warm cache, same process:** the cost is mostly composition-independent work done
  again for each build:
  - checkpoint conversion, 4.0 s;
  - Dynamo tracing, 6.4 s, including an abandoned first attempt of 2.6 s;
  - AOTAutograd and the Inductor cache load, 3.9 s;
  - self-checks, 1.3 s.
- **Graph update:** `cudaGraphExecUpdate` cannot move the force graph to another composition. The topology
  differs (416–421 nodes), and so do the Inductor kernels and the cuBLAS/CUTLASS GEMMs. Within one
  composition, two padded edge buckets give the same topology and the same kernel functions, with 21 launch
  shapes changed, so an update would apply. It would save at most the 0.3 s capture. Recaptures cost at most
  1.2 % of relaxation wall time (below). **The question is closed: not worth building.**
- **Setup-time ceiling:** fixes F2 and F3 below bring a warm same-process build from 16.3 s to about
  10.5 s, or about 9 s with F4. F1 to F3 bring a cold-cache new composition from 36.6 s to about 21 s. Removing all composition-independent work
  (conversion, tracing, self-checks) would leave about 2 s: capture, species tables, the plan and the
  private model copy. That needs whole-frame compile artefacts, which the force function's graph break
  blocks today.

## Run

| | |
| --- | --- |
| Machine, GPU | local PC, NVIDIA GeForce RTX 5060 Ti (WSL2), through the local GPU FIFO (`tools/ci/local_gpu.py run`, one reservation per mode) |
| Environment | `~/environments/mace_cueq_oeq` (hybrid backend); torch 2.14.1+cu130 |
| Model | MH-1 (`mace-mh-1.model`), head `matpes_r2scan` |
| Settings | float32, the default plan and rewrites (radial tables on; the radial-table cache merged on main is in use), gpu-LBFGS with `FrechetCell(scalar_pressure_GPa=0)` (Energy_Hull step 2 in float32), 5 relaxation steps after each build |
| Structures | eight Energy_Hull step-2 phases as 3×3×3 supercells ([results/setup-across-compositions/structures/](../../results/setup-across-compositions/structures/), from ResearchModule `Structures/MaterialsProject/hull/`): NaCl (54 atoms), Na2O (81), NaNbO3 (270), NbOCl3 (540), NbCl5 (972), TaCl5 (972), NaTaCl6 (864), NaClO4 (324) |
| FastMACE | `c5cca71b` (main `d35658be` plus the tools; no library change) |
| Tools | `tools/bench/setup_compositions.py` (`fp32_setup.py`'s timers, plus Dynamo/Inductor metrics and counters, cache-file deltas, generated-source hashes, per-self-check times and CUDA-graph DOT dumps), `tools/bench/setup_compositions_report.py`; launchers `results/setup-across-compositions/run.sh` and `run_precompile.sh` |
| Time | 30 October 2026, 01:02 to 02:06 CET |
| Raw data | [results/setup-across-compositions/rtx5060ti-20261030/](../../results/setup-across-compositions/rtx5060ti-20261030/) (`*/setup.json`, logs). The DOT dumps (3.8 MB) are not in git; they are kept at `~/fastmace-runs/setup-compositions-20261030/graphs*/graphs/` in WSL |

Modes, each in its own process:

- **cold:** each phase alone, with fresh empty caches (`FASTMACE_CACHE`, Inductor, Triton);
- **warm:** each phase alone again, with its own cold run's caches;
- **chain-cold:** all eight in one process with one fresh shared cache, so builds 2–8 run after another
  composition with nothing cached for themselves;
- **chain-warm:** the same again in a new process, with the cache now warm;
- **graphs** and **graphs-edges:** CUDA-graph dumps, not used for timing.

## Per-stage setup (seconds, median over the eight phases; the chains use builds 2–8)

| Stage | cold (fresh process, empty cache) | warm (fresh process, own warm cache) | same process, cache cold for this composition | same process, warm cache |
| --- | ---: | ---: | ---: | ---: |
| model load and conversion (checkpoint conversion, template and private copies) | 4.3 | 4.5 | 4.2 | 4.0 |
| planner and rewrites (plan, resolution, species tables, folds, other host work) | 4.1 | 3.6 | 0.3 | 0.3 |
| self-checks (23 records) | 8.1 | 2.7 | 1.8 | 1.3 |
| radial tables: fit + certify | 7.6 | 0.0 | 7.8 | 0.0 |
| OEq source render + NVRTC compile | 0.2 | 0.2 | 0.1 | 0.1 |
| Dynamo trace (incl. the restart) | 6.4 | 6.2 | 7.1 | 6.4 |
| — of which the abandoned attempt 0 | 2.6 | 2.4 | 2.8 | 2.6 |
| AOTAutograd + Inductor lowering + Triton compile (or cache load) | 1.8 | 2.9 | 2.8 | 3.9 |
| Inductor codegen (source generation) | 4.2 | 0.0 | 4.4 | 0.0 |
| Triton autotune benchmarks | 3.9 | 0.0 | 3.8 | 0.0 |
| module loading (`PyCodeCache`) | 1.3 | 0.1 | 0.5 | 0.1 |
| first execution of the compiled call (warm-up) | 6.4 | 1.9 | 4.1 | 0.0 |
| CUDA-graph capture (incl. `cudaGraphInstantiate`, ≤ 0.01 s) | 1.1 | 0.3 | 0.3 | 0.3 |
| **Simulation construction (build + capture)** | **49.7** | **22.3** | **36.6** | **16.3** |
| range | 44.9–54.3 | 21.8–23.6 | 26.7–40.5 | 16.1–17.1 |
| plus the model load, once per process (`fm.load`) | 19.8 | 3.6 | — | — |

Notes on the stages:

- **The first build in a process** pays more for planner, self-checks and first execution than later builds
  do: 3.6–4.1 s, 2.4–8.1 s and 1.8–6.4 s. These are lazy initialisations (the rewrite registry, cuBLAS,
  first kernel launches). The chain's first build: 21.6 s warm, 44.4 s cold.
- **Cold model load (19.8 s)** is mostly the OEq extension's JIT build ("Precompiled extension shared object
  not found ... falling back to JIT"), cached in `FASTMACE_CACHE` afterwards.
- **Cache hits:** in the warm runs every FX graph hits (4/4) and AOTAutograd hits 2/2. In the chain-cold
  builds 2–8, FX misses 2 of 2 for every composition. The radial-table cache hits whenever the species set
  was fitted before (0.0 s warm).
- The fp32-profile (28 October) measured `fm.md` warm at 34–41 s and the first compiled call at 11.5–12 s.
  Here the first compiled call is 10.3–12.3 s warm, which agrees. The build is cheaper because the
  radial-table cache now hits.

Per phase and run, the totals and the cache counters are in the raw records. `setup_compositions_report.py`
prints them.

## What changes between compositions

Generated sources compared by hash, with Inductor's `PyCodeCache` keys and SHA-256 of the OEq and
fused-forward sources:

| Generated code | Distinct over the 8 compositions | Shared by all |
| --- | ---: | ---: |
| OEq streaming sources compiled with NVRTC | 5 | 5 (identical everywhere) |
| fused-forward CUDA sources | 2 | 2 (identical everywhere) |
| Inductor modules (forward, backward and kernels) | 30 (cold chain), 20 (warm chain) | 0 |

The diff of two Inductor forward modules shows what makes them differ:

| Pair | Lines that differ | What differs |
| --- | ---: | --- |
| NbCl5 → TaCl5 (972 atoms, 2 species each) | 54 of 4,649 | radial-table shapes `(4, 53, 11, 66)` → `(4, 52, 11, 66)` and `(212, 10, 1280)` → `(208, …)`: the tables' row counts are static shapes; the integer argument of `torch.ops.fastmace.fused_folded_forward` (9 → 11) is a constant in the graph |
| NbOCl3 → NaClO4 (3 species, 540 vs 324 atoms) | 774 of 4,880 | atom-count size hints (`size_hints={'x': 1024}` → `512`) in every Triton kernel; the table shapes as above |
| NaCl → Na2O (first build vs second, in one process) | 4,779 | the first build is specialised to 54 atoms (static), the second has a dynamic atom count (`s23`) after Dynamo's automatic-dynamic recompile: different code throughout |
| NbCl5 → NaNbO3 (2 vs 3 species) | (structure) | the rewrite structure differs: joined products against grouped products, other GEMM shapes |

- **Could be data rather than code:**
  - Pad the radial-table rows to a bucket and pass them as a tensor of dynamic size.
  - Pass `fused_folded_forward`'s integer as data, or as a value with a bounded set of variants.

  Compositions with the same species count would then share code. This is fix F1.
- **Stays code:** the species count and the folds it selects (2 against 3 species are different programs).
- **Atom count:** a dynamic dimension already, after the first build. Its size hints still select different
  Triton configurations, so each new atom count misses the FX cache. In this set, the atom counts and the
  table shapes always changed together, so their shares are not separable.
- **Edge count:** dynamic. But NbCl5 at 0.95 strain (33,426 edges, bucket 34,048) in a new build missed the
  FX cache against the cached NbCl5 (25,542 edges) and cost 23.4 s instead of 16.6 s. The edges' size hint
  also reaches the cache key.

## CUDA-graph update (inspection only)

The `graphs` and `graphs-edges` runs keep each captured graph (`CUDAGraph.enable_debug_mode`) and dump it
with `cudaGraphDebugDotPrint`. The report script compares node count, edges, node types, kernel functions
and launch shapes (grid, block, shared memory):

| Graph | Pair | Nodes | Topology | Kernel functions | Launch shapes changed |
| --- | --- | --- | --- | --- | ---: |
| force (B) | NaCl → NaCl 0.97 (same bucket 1,792) | 421 / 421 | same | same | 0 |
| force (B) | NaCl → NaCl 0.90 (bucket 1,792 → 3,072) | 419 / 419 | same | same | 21 |
| force (B) | NbCl5 → NbCl5 0.95 (bucket 25,600 → 34,048) | 417 / 417 | same | same | 21 |
| force (B) | NaCl → NbCl5, NaCl → NaNbO3, NbCl5 → NaNbO3 | 421 / 419 / 416 | **different** | different Inductor fusions, other CUTLASS GEMM tiles (`128x32` ↔ `128x64` ↔ `128x256`), joined vs grouped products | – |
| integrator/optimiser (AC) | every pair | 300 / 300 | same | same | 25–62 |

- **Across compositions,** `cudaGraphExecUpdate` does not apply to the force graph. The node counts and
  edges differ, and the kernels are different functions. The library GEMMs change their kernels with the
  shapes, and the Inductor code differs anyway (above).
- **Within one composition,** across edge buckets, the topology and functions are equal and only kernel
  parameters and grids change, so an update would apply. It would replace a 0.3 s capture.
- **Recaptures in relaxations.** In the murphy relaxation benchmark (relax-throughput,
  `results/relax-throughput/murphy-20261029/`, float64, 20 glass frames), each block of 20 frames had 30
  recaptures in 744 neighbour rebuilds. At the measured capture time (0.27 s per fresh build, all graphs),
  that is at most 8 s of a 699 s block (≤ 1.2 %). This is just over the 1 % threshold the conductor set as
  an upper bound; one recapture replaces only the force graph, so the real share is lower.
  **Recommendation: close the graph-update question.**

## Conductor additions

### 1. The first compiled force call

- **Where the 2.4–2.9 s abandoned attempt goes.**
  - `compile_attempt_0` is Dynamo tracing the force function up to `torch.autograd.grad`. Dynamo has
    marked that function as skipped (counter `graph_break … function marked as skipped: grad`, 3 per
    build).
  - The graph break raises a restart: Dynamo discards the attempt and traces again (`compile_attempt_1`) to
    compile the part before the break. Three graphs result (`stats.unique_graphs` 3), resuming at
    `forward:136`, `get_outputs:187` and `compute_forces_virials:50`.
  - The attempt is a tracing cost on every build, in every mode (2.4–2.9 s), warm or cold.
- **The AOTAutograd cache is already on.**
  - `torch._functorch.config.enable_autograd_cache` is True by default in torch 2.14.1, and the warm runs
    hit it (2/2).
  - What remains warm is Dynamo's bytecode tracing (about 3.8 s plus the restart). Neither the FX cache nor
    the AOT cache skips it.
- **`torch.compile(...).aot_compile()` and `torch.compiler.precompile` need `fullgraph=True`.**
  - The force function has the graph break at `torch.autograd.grad`, so fullgraph capture fails before any
    custom-op question arises.
  - Not tested further: the stop rule's "breaks" applies until the graph break is gone (F2).
- **Dynamo's `caching_precompile=True`** (the whole-frame cache in torch 2.14) was tested in fresh
  processes:
  - Runs: NaCl cold, then NaCl at 0.97 strain warm, twice, and the warm default as a baseline. The runs are
    `pre-*`, launcher `run_precompile.sh`.
  - **The graph tier failed at compile in all three precompile runs.** The plan fell back to the eager tier
    ("mode compiled: failed at compile; mode eager: activated as revision 1").
  - Setup took 100 s cold, and 62 s and 60 s warm, against 26 s for the default warm build.
  - Dynamo also warned that it drops a `CLOSURE_MATCH` guard (`gated_rows.act_scalar`) under this setting.
  - **Stop: it breaks the compiled tier and saves nothing.**

### 2. The duplicate checkpoint conversion

- In the Energy_Hull flow (`fm.load` once, then a `ForceField` and a `Simulation` per structure), a build
  converts the checkpoint once: `loading.load`, 2.8–3.4 s. The template's private copy adds 0.4 s and
  `private_copy` 0.9–1.3 s, which makes the 4.0–4.5 s "load and conversion" row.
- The fp32-profile's "twice" was `fm.md`: the pristine load plus the conversion, in every call.
- The conversion does not depend on the composition. `templates.conversion_key` holds the checkpoint hash,
  path, head, backend, dtype, device, the conversion options, the process patches and the package
  versions.
- But the `TemplateCache` belongs to each `ForceField` (`ForceField._template_cache`). A new composition,
  which means a new `ForceField`, converts again.
- **One immutable template could feed every plan of every composition** of one loaded model, if the cache
  were owned by the `LoadedModel` (or a session object) instead of the `ForceField`.
  - Isolation holds as today: the template never reaches a plan; each plan gets its private copy (0.4–0.6 s;
    `templates.private`); `TemplateChangedError` guards the template.
  - `tests/cpu/test_template_cache.py` and `tests/cpu/test_isolation.py` cover these per force field. They
    would need the same cases across two force fields.
  - Saving: about 3 s per composition after the first. This is fix F3.
- The reference model the gate compares against is the pristine model (`fm.load`), not a conversion. There
  is no second conversion to merge there.

### 3. Self-checks

- 23 records per build:
  - warm, later builds: 1.2–1.4 s in all;
  - the first build of a process: 2.4 s warm, 7–8 s cold, where single checks reach 1.2–1.5 s
    (`fused_folded_backward`, `grouped_launch`) on first use.
- What a check reads (`rewrites.registry.Context`, the records' `selfcheck` functions):
  - the model and the record's rewrite;
  - inputs drawn from `torch.Generator(seed)`;
  - `Context.inputs`: the structure's one-hot `node_attrs` (per-atom rows) and the superset's
    `species_columns`, and for OMOL-type heads the per-graph joint features.
- **What decides their outcome:** model (checkpoint hash and head), the record and its version, precision,
  GPU and software stack, and the species set: the columns, joint features and table rows. The one
  structure-dependent input is `node_attrs`. Its row count follows the atom count, but its content is the
  species one-hot. No check reads positions, the cell or edges.
- **Could be cached:** all 23, keyed by (checkpoint hash, head, record name and version, precision, device
  name, software versions, species columns, joint features). This would apply only if `node_attrs` were
  reduced to one row per species, so the cache key does not carry the atom count. A cache would save
  1.2–1.4 s per composition in a running process and 2–8 s on the first build.
- **Owner decision:** `fastmace/model/templates.py` records it: "Self-checks are not cached: every plan runs
  all of them on its own copy (owner, checkpoint 2)". A cache reopens that, so it is not proposed as a
  default here (fix F4).

### 4. Recaptures

See the CUDA-graph section above: at most 1.2 % of relaxation wall time on murphy, 0.3 s per capture here.
**Graph update closed.**

## Fixes, ranked by expected seconds saved

Seconds are per new composition on the RTX 5060 Ti. Each fix needs a same-process A/B in both build orders
and the usual gates; none is a default change until the owner says so.

| Rank | Fix | Saving | Stop rule |
| ---: | --- | --- | --- |
| F1 | **Composition-specific constants as data.** Pad the radial-table rows to a bucket (by the species-pair table size, not per crystal) and pass them as dynamic-size tensors. Pass `fused_folded_forward`'s integer argument as data or a bounded variant. Then compositions with the same species count share Inductor code. | cold cache: up to about 10 s per new composition (TaCl5 after NbCl5: 26.7 s against 36–40 s); warm cache: fewer distinct cache entries | stop if two same-species-count compositions with different atom counts still miss the FX cache, or the step time rises by more than 1 % |
| F2 | **Remove the Dynamo restart.** Take `torch.autograd.grad` out of the compiled frame, or compile the forward and its backward without the graph break. This also unblocks testing `aot_compile` and `precompile` (which need `fullgraph`). | 2.4–2.9 s per build, every build (MD too); with frame artefacts afterwards, up to the 6.4 s of tracing | stop if forces or stresses change beyond the gate, or the saving is under 2 s |
| F3 | **Template cache per loaded model.** Move the `TemplateCache` from `ForceField` to the `LoadedModel` (or an explicit session object) so later compositions copy the converted model instead of converting again. | 2.8–3.4 s per composition after the first (the conversion; the private copy stays) | stop if the isolation tests across two force fields fail or `TemplateChangedError` fires in a multi-composition run |
| F4 | **Self-check cache** keyed as in addition 3, with `node_attrs` reduced to one row per species. Needs the owner (checkpoint 2 decided against caching). | 1.2–1.4 s per composition; 2–8 s on a process's first build | owner decision; then stop if any check reads more than the key |
| F5 | **Edge and atom size hints in the cache key.** A new edge count or atom count misses the FX cache although both are dynamic (NbCl5 at 0.95: +6.8 s). Test Inductor's options for hint-independent code (for example a fixed size hint per bucket) before anything else. | 4–7 s per new size across processes | stop if the step time rises by more than 1 % on the eight phases |
| – | `caching_precompile` (Dynamo whole-frame cache) | none: the compiled tier failed | rejected |
| – | `cudaGraphExecUpdate` for recaptures or compositions | ≤ 0.3 s per recapture, not applicable across compositions | closed |

**Ceiling:**
- Warm same-process build: F2 and F3 take 16.3 s to about 10.5 s; with F4, about 9 s. F1 changes little
  warm.
- Cold-cache new composition: F1 to F3 take 36.6 s to about 21 s.
- Removing all composition-independent work leaves about 2 s: capture, species tables and plan, and the
  private model copy. That needs whole-frame compile artefacts after F2.

## Side finding

`setup_compositions.py` first called `fm.load` and then built a float32 engine. The engine refused with
`ProcessConstraintError: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True was set after the first CUDA
allocation`, because `torch.cuda.get_device_name()` had initialised CUDA first. The tool now calls
`fastmace.model.compile.configure_allocator()` before anything touches CUDA, as other tools do. A user script
that loads the model and queries the GPU before its first float32 `Simulation` hits the same refusal. The
float64 relaxation benchmark (relax-throughput) did not, because only float32 plans want expandable
segments. The message says what to do; this is a usability follow-up, not a bug in the check.
