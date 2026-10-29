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
| 34,200 atoms | **36,720 atoms (0.96M edges at the cutoff), +7 %** | 417 |

- At 36,720 atoms rule 12 turned `recompute_conv_weights` on by itself. Its reason in the plan: "on by rule 12:
  the estimate does not fit 90% of the free GPU memory without it; turning recompute_conv_weights on saves about
  5330 MiB".
- The next size the scan tried, 44,064 atoms (6×6×17), failed during the build. The scan's grid has no repeat in
  between, so the true limit lies somewhere from 36,720 to 44,064 atoms.

**Memory at 32,400 atoms (5×5×18), recompute forced off and on, one process each:**

| | Off | On | Change |
| --- | --- | --- | --- |
| Largest reserved memory | 13.92 GiB | 11.12 GiB | **−2.80 GiB (−20 %)** |
| Force call's bytes per edge | 15,582 | 12,173 | −3,409 |
| ms per step | 313 | 338 | +8 % (separate processes) |

The measured saving is 3,409 B per edge, two thirds of the 5,120 B the weights take. Without them the peak moves to
other tensors. Rule 12's term (`fastmace.rewrites.memory.recompute_conv_weights`) declares the full 5,120 B, so the
estimate overstates the saving. This only matters once the rescue is already on. At 36,720 atoms the estimate after
the rescue predicted 13.75 GiB, and the run reserved at most 13.69 GiB.

**Step time:** the same-process A/B costs 9–10 %. The planner keeps recompute off at normal sizes, but it turns it on
from 27,000 atoms, about 7,000 atoms before it is needed. That is still open (next section).

**Exact:** the recomputed weights equal the stored ones bitwise. The force/stress gate passed with recompute on.

## Off wherever the plan fits

**Open: the rescue turns on earlier than needed.** `tools/bench/rescue_plans.py` plans each cell as `fm.Simulation`
does, on the idle GPU's free memory (14.80 GiB), without running it:

| Cell | Atoms | Edges | Plan's `recompute_conv_weights` | Rule 12's prediction |
| --- | ---: | ---: | --- | --- |
| NaCl 3×3×3 | 216 | 9,408 | off | 1.53 GiB |
| NaNbO3 3×3×3 | 1,080 | 85,152 | off | 1.53 GiB |
| NbCl5 3×3×3 | 1,944 | 64,215 | off | 1.53 GiB |
| NbCl5 4×4×8 | 9,216 | 0.30M | off | 4.96 GiB |
| NbCl5 5×5×12 | 21,600 | 0.71M | off | 11.26 GiB |
| NbCl5 5×5×13 | 23,400 | 0.77M | off | 12.18 GiB |
| NbCl5 5×5×14 | 25,200 | 0.83M | off | 13.09 GiB |
| NbCl5 5×5×15 | 27,000 | 0.89M | **on** | 10.18 GiB with it |
| NbCl5 5×5×16 | 28,800 | 0.95M | **on** | 10.84 GiB with it |
| NbCl5 5×5×18 | 32,400 | 1.07M | **on** | 12.16 GiB with it |
| NbCl5 5×6×17 | 36,720 | 1.21M | **on** | 13.75 GiB with it |

- **It stays off at normal sizes.** All cells up to 25,200 atoms keep it off.
- **From 27,000 atoms it is on.** Without recompute, the plan runs up to 34,200 atoms: the CAP-T05 limit. At
  32,400 atoms it ran at 313 ms per step with 13.92 GiB reserved.
- **So from 27,000 to 34,200 atoms recompute costs 9 % for nothing.** The estimate without recompute overstates the
  peak by about 1.2 there: about 16.8 GiB predicted at 32,400 atoms against 13.92 GiB reserved. That is within
  decision 67's fitted band of 1.05–1.34, but it is enough to trip the rescue about 7,000 atoms early.

This does not meet the condition that the cost applies only to sizes that would not run otherwise. Ways to close the
gap are listed for the conductor (CAP-T07 is not pushed until one is chosen).

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
  - It runs only when the estimate does not fit 90 % of the free memory.
  - It turns on each auto record in `MEMORY_RESCUE` whose family applies and whose requirements are on, if that
    lowers the estimate, and stops as soon as the plan fits.
  - It runs before decision 51's drops, and an explicit value is kept. Reason code `rule-12-memory-rescue`.
- **Docs:** switch map (field and rule rows), `docs/reference/rewrites.md`, `docs/design/planner-5.md` and the
  CHANGELOG.
- **Tools:** `tools/bench/activation_profile.py` (what the force call keeps alive at its peak, by allocating site)
  and `tools/bench/rescue_plans.py` (the default plan's choice per cell, without running it).

## Checks

- **GPU exactness** (`tests/gpu/test_recompute_conv_weights_cuda.py`, local RTX 5060 Ti): passed, 8 cases. dsh and
  dr are bitwise equal to the stored-weights backward; dx differs only by the atomic order.
- **Force/stress gate** (`tools/force_stress_gate.py --engine fastmace --rewrites recompute_conv_weights=on`,
  local RTX 5060 Ti, commit `47bb25a4`): **passed** on all three cells. The plan resolved
  `recompute_conv_weights = on` (requested). Error against float64, with its ratio to the float32 ASE calculator's
  error in brackets:

  | Cell | energy (eV) | forces (eV/Å) | stress (eV/Å³) |
  | --- | ---: | ---: | ---: |
  | NaCl | 1.3e-4 (0.17) | 3.6e-6 (0.95) | 3.4e-8 (0.38) |
  | NbCl5 | 5.8e-5 (0.08) | 8.3e-5 (1.04) | 3.8e-8 (0.36) |
  | NaNbO3 | 6.1e-4 (0.50) | 1.4e-4 (1.02) | 2.3e-7 (0.76) |

- **NVE drift:** not run. The numerics do not change: the per-edge values are bitwise the same, and only the
  atomic order differs, as between any two runs.
- **CPU tests on leto** (GPU hidden, `-n 8`, commit `4fa63838`): 589 passed in the targeted files (planner, memory
  estimate, rewrites, report, config, docs options, plan texts). The golden plan texts were regenerated on leto
  (commit `47bb25a4`): only the "+N more" count of rewrites off changed.

## Setup

- **Local:** Fabi-PC, WSL2, RTX 5060 Ti 16 GB (driver 610.88), `mace_cueq_oeq` (torch 2.14.1+cu130),
  `tools/ci/local_gpu.py run` reservations with the idle check before every job. FastMACE commit `47bb25a4`
  (pinned source copy); no baseline, since both sides are FastMACE.
- **Model:** MH-1 (`mh1-r2scan`), hybrid backend, float32, Langevin NVT at 300 K.
- **Capacity points:** 20 warm-up and 200 timed steps, one process per point (`tools/bench/capacity_scan.py`).
- **Raw records:** `results/cap-t07-20261029/` (`points/`, `ab/`, `ab/gate/metadata.json`, `plans.json`) and the
  run script `results/cap-t07-20261029/local.sh`.
