# CAP-T07: recompute the per-edge tensor-product weights when memory requires it

Plan task CAP-T07 ([plans/capacity.md](../../plans/capacity.md)). The activation profile
(`tools/bench/activation_profile.py`, NbCl5 float32, 9,504 atoms) found the largest single item alive at the force
call's memory peak:

- `wprime_backward` keeps the second interaction's per-edge weights `[E, 1,280]` for the backward;
- that is 5,120 B per edge, 1.26 GiB, a third of the peak.

The conductor's instruction (29 October 2026) was that recompute costs step time, so the planner turns it on only
when the plan would not fit otherwise, as in decision 67. It must also be exact or pass the gate.

## Verdict

**New limit on the RTX 5060 Ti 16 GB (WSL2), MH-1 NbCl5 float32, default plan, 220 MD steps:**

| Before (CAP-T05) | After | ms per step at the new limit |
| --- | --- | --- |
| 34,200 atoms | **36,720 atoms (0.96M edges at the cutoff), +7 %** | 411 |

- The next size the scan tried, 44,064 atoms (6×6×17), runs out of memory in the warm-up.
- The scan's grid has no repeat in between, so the true limit lies somewhere from 36,720 to 44,064 atoms.

**Recompute runs only where memory would not be enough without it** (owner, decision 67). In order:

1. **The plan turns it on only for a cell that cannot fit.** That means rule 12's estimate, divided by its largest
   overstatement in decision 67's fit (1.34), still does not fit 90 % of the free memory (`memory.short_by_far`).
2. **A cell that merely does not fit runs without it.** If it then runs out of memory at build or capture, and the
   CUDA context is still usable, `Simulation` releases the failed engine and rebuilds once with recompute on
   (`fallback.memory_rescue`). The plan revision records it: "fallback: rewrites recompute_conv_weights on: out of
   memory at …; recompute_conv_weights auto -> on".
3. **If the error left the context unusable, the run stops with an explanation instead of crashing.** It names the
   fix: `recompute_conv_weights="on"`, or a smaller cell if recompute was already on.

**Default plans on the RTX 5060 Ti** (`tools/bench/rescue_plans.py`: planned as `fm.Simulation` plans, on the idle GPU's
14.80 GiB free; the 28,800-36,720 points then ran with 20 warm-up and 200 timed steps):

| Cell | Atoms | Edges | Plan's `recompute_conv_weights` | Ran |
| --- | ---: | ---: | --- | --- |
| NaCl 3×3×3 | 216 | 9,408 | off | |
| NaNbO3 3×3×3 | 1,080 | 85,152 | off | |
| NbCl5 3×3×3 | 1,944 | 64,215 | off | |
| NbCl5 4×4×8 | 9,216 | 0.30M | off | |
| NbCl5 5×5×12 to 5×5×15 | 21,600–27,000 | 0.71M–0.89M | off | 27,000: 281 ms |
| NbCl5 5×5×16 | 28,800 | 0.95M | off | 302 ms, 12.51 GiB reserved |
| NbCl5 5×5×18 | 32,400 | 1.07M | off | 320 ms, 13.80 GiB |
| NbCl5 5×5×19 | 34,200 | 1.13M | off | 398–437 ms, 14.72 GiB |
| NbCl5 5×6×17 | 36,720 | 1.21M | **on** (the plan) | 411 ms, 12.75 GiB |

**Two other designs were tried first and failed:**

- **The estimate-only rescue switched on too early.** Commit `c6c4099f` turned recompute on as soon as the plan did
  not fit 90 % of the free memory. That happened from 27,000 atoms, while plans without recompute run up to 34,200.
  The estimate overstates the peak by about 1.2× there, within decision 67's band.
- **A retry alone does not reach 36,720 atoms.** With the floor compared against the whole free memory instead of
  its 90 %, 36,720 atoms was planned with recompute off. Its build then ran out of memory in the driver: the error
  arrived as a `CUDA error: out of memory` or, without blocking launches, as `CUBLAS_STATUS_EXECUTION_FAILED`, and it
  left the CUDA context sticky.
  - So the in-process retry could not run, and the fallback search then failed in `cudaMemGetInfo`.
  - Both are fixed: the driver's out-of-memory error is never a fallback trigger, and `Simulation` checks the context
    before the search (`fallback.raise_if_context_unusable`).
  - The 46,656-atom cell now stops with "the CUDA context is not usable after this build error … run again in a
    fresh process with a smaller cell".
  - On this GPU under WSL2 an out-of-memory error in the build was always the driver's, so the in-process retry
    (step 2) never ran on real hardware here. Its tests use injected errors
    (`tests/cpu/test_memory_rescue_retry.py`). It covers the caching allocator's recoverable
    `torch.cuda.OutOfMemoryError`, which native Linux raises for most allocations.

**Memory at 32,400 atoms (5×5×18), recompute forced off and on, one process each:**

| | Off | On | Change |
| --- | --- | --- | --- |
| Largest reserved memory | 13.92 GiB | 11.12 GiB | **−2.80 GiB (−20 %)** |
| Force call's bytes per edge | 15,582 | 12,173 | −3,409 |
| ms per step | 313 | 338 | +8 % (separate processes) |

The measured saving is 3,409 B per edge, two thirds of the 5,120 B the weights take. Without them the peak moves to
other tensors. Rule 12's term (`fastmace.rewrites.memory.recompute_conv_weights`) declares the full 5,120 B. At
36,720 atoms the estimate with recompute predicted 13.75 GiB, and the run reserved at most 13.69 GiB.

**Step time:** recompute costs 9–10 % in the same-process A/B, so it stays a memory rescue, never a default.

**Exact:** the recomputed weights equal the stored ones bitwise. The force/stress gate passed with recompute on and
with the default plan.

## Step-time cost

`tools/ab_rewrites.py`, a = `recompute_conv_weights=off`, b = `on`, everything else the default plan. MH-1 hybrid
float32 NVT at 300 K. Median step in ms. Order `ab` uses the abab block pattern and order `ba` uses abba.

| Cell | Blocks | order ab: a / b, b/a | order ba: a / b, b/a | block pairs |
| --- | --- | --- | --- | --- |
| NbCl5 3×3×3, 1,944 atoms | 12 × 300 steps | 17.94 / 19.66, **1.096** | 18.04 / 19.78, **1.097** | 1.071–1.116 |
| NbCl5 4×4×8, 9,216 atoms | 8 × 100 steps | 89.28 / 96.99, **1.086** | 89.08 / 97.19, **1.091** | 1.059–1.103 |

Every run was valid: nothing compiled while timed, and no other process used the GPU. The cost is the second pass
over the edges in the backward: the radial table and the `[rows, 1,280]` GEMM per chunk, plus one kernel launch per
chunk instead of one in all. It does not shrink with size, so recompute stays a memory rescue, never a default.

## What changed

- **Kernel** (`fastmace/kernels/oeq_streaming/wprime_backward.py`):
  - New custom op `streaming_conv_wprime_backward_recompute`. The backward runs over the edges in chunks of the
    interaction's chunked-backward rows.
  - For each chunk it builds the weights again from the edge lengths: the radial table, then `a @ weight.T`, the
    same chain as the forward. They go into one chunk-sized buffer, and the existing wprime kernel runs on that
    chunk.
  - `_WPrimeConvolution` saves an empty placeholder instead of the `[E, W]` weights when `recompute_rows` is set.
- **Rewrite record** `recompute_conv_weights` (`fastmace/rewrites/radial`):
  - exact; requires `wprime_backward`; not deterministic (atomic sender additions, like `wprime_backward`);
  - its self-check compares message, density, dr, dx and dsh against the stored-weights path in 256-edge chunks;
  - memory term: minus the streaming interactions' weight floats per edge.
- **Config:** `RewriteConfig.recompute_conv_weights`, default `"auto"`. Its auto rule is
  `on_by_rule_12_when_the_plan_does_not_fit_without_it`.
- **Fixed planner:** `"auto"` resolves to off, and `"on"` without `wprime_backward` is refused.
- **Rule 12's memory rescue** (`planner.minimal._rule_12`, `planner.memory.MEMORY_RESCUE`):
  - It applies only to a plan that does not fit 90 % of the free memory after decision 51's drops. An explicit
    value is kept. Reason code `rule-12-memory-rescue`.
  - If even the estimate divided by `memory.RESCUE_OVERSTATEMENT` (1.34) does not fit (`memory.short_by_far`), it
    turns on each auto record whose family applies and whose requirements are on, if that lowers the estimate. It
    stops as soon as the plan fits.
  - Otherwise it leaves them off, with a reason saying that an out-of-memory error rebuilds with them on.
- **The rebuild** (`planner.fallback.memory_rescue`, `Simulation._build_and_prepare`):
  - It applies to an out-of-memory error at build or capture, from the caching allocator or the driver's
    `CUDA error: out of memory`.
  - If a rescue record is off, was left "auto" and has its requirements on, and the context is usable,
    `Simulation` drops the failed engine (its error is detached, its graphs and pools released) and builds the
    revision with the records on, once.
  - If the context is sticky, it raises the error with the fix.
- **Out-of-memory errors are never fallback triggers** (`fallback._propagates`): this now includes the driver's
  `CUDA error: out of memory`. Before the §9.4 search, `Simulation` checks that the context is usable
  (`fallback.raise_if_context_unusable`). If not, it raises the failure with an explanation (and the fix when rule
  12 had warned) instead of crashing in the search's `cudaMemGetInfo`.
- **Docs:** switch map (field and rule rows), `docs/reference/rewrites.md`, `docs/design/planner-5.md` and the
  CHANGELOG.
- **Tools:** `tools/bench/activation_profile.py` (what the force call keeps alive at its peak, by allocating site)
  and `tools/bench/rescue_plans.py` (the default plan's choice per cell, without running it).

## Checks

- **GPU exactness** (`tests/gpu/test_recompute_conv_weights_cuda.py`, local RTX 5060 Ti): passed, 8 cases. dsh and
  dr are bitwise equal to the stored-weights backward; dx differs only by the atomic order.
- **Force/stress gate, default plan** (commit `f3ed415e`, the rescue off at these sizes): **passed** on NaCl, NbCl5
  and NaNbO3 (forces 1.05× and 1.01× the float32 yardstick's error on NbCl5 and NaNbO3).
- **Force/stress gate, recompute on** (`tools/force_stress_gate.py --engine fastmace --rewrites
  recompute_conv_weights=on`, local RTX 5060 Ti, commit `47bb25a4`): **passed** on all three cells. The plan resolved
  `recompute_conv_weights = on` (requested). Error against float64, with its ratio to the float32 ASE calculator's
  error in brackets:

  | Cell | energy (eV) | forces (eV/Å) | stress (eV/Å³) |
  | --- | ---: | ---: | ---: |
  | NaCl | 1.3e-4 (0.17) | 3.6e-6 (0.95) | 3.4e-8 (0.38) |
  | NbCl5 | 5.8e-5 (0.08) | 8.3e-5 (1.04) | 3.8e-8 (0.36) |
  | NaNbO3 | 6.1e-4 (0.50) | 1.4e-4 (1.02) | 2.3e-7 (0.76) |

- **NVE drift:** not run. The numerics do not change: the per-edge values are bitwise the same, and only the
  atomic order differs, as between any two runs.
- **CPU tests on leto** (GPU hidden, `-n 8`, commit `61d92a35`, merged with main): 702 passed, 30 skipped.
  - The files: planner, memory estimate and fit, rewrites, report, config, docs options, plan texts, fallback
    (memory, simulation, capture, candidates), rules and `test_memory_rescue_retry`.
  - The golden plan texts were regenerated on leto (commit `47bb25a4`): only the "+N more" count of rewrites off
    changed. They did not change again after the retry.

## Setup

- **Local:** Fabi-PC, WSL2, RTX 5060 Ti 16 GB (driver 610.88), `mace_cueq_oeq` (torch 2.14.1+cu130),
  `tools/ci/local_gpu.py run` reservations with the idle check before every job. Pinned source copies of FastMACE:
  - `47bb25a4`: the first points, the A/Bs and the gate with recompute on;
  - `f3ed415e`: the retry, the default gate and the per-cell plans;
  - `d317cf15`: the floor at 90 % and the scan;
  - `61d92a35`: the unusable-context check at 46,656 atoms.

  No baseline, since both sides are FastMACE.
- **Model:** MH-1 (`mh1-r2scan`), hybrid backend, float32, Langevin NVT at 300 K.
- **Capacity points:** 20 warm-up and 200 timed steps, one process per point (`tools/bench/capacity_scan.py`).
- **Raw records** in `results/cap-t07-20261029/`:
  - the first design: `points/`, `ab/`, `ab/gate/metadata.json` and `plans.json`;
  - the retry with the floor against the whole free memory: `retry/`, with the default gate;
  - the final floor: `floor/`;
  - the 46,656-atom failures: `guard/`;
  - the run script: `local.sh`.
