# 2-T13: `fm.calc` without a second compile when the edge count changes

Plan task 2-T13 (SPEC §8.2). 1b-T18 measured that `fm.calc` compiled again at its second call
(4.5 s plain cuEq, 19.1 s hybrid). The edge count had changed, and dynamo's automatic dynamic shapes
compiled the model again with a symbolic edge dimension.

## The change

- Evaluation plans (`MinimalPlanner.plan_evaluation`) set `EngineConfig.dynamic_edges` when they
  compile. The decision is recorded as `EngineConfig.dynamic_edges`. MD (graph-tier) plans keep it
  off, so they are unchanged.
- With `dynamic_edges`, `MACEForceField.use_edges` marks the edge dimension of `edge_index`,
  `shifts` and `unit_shifts` dynamic (`model.compile.mark_edges_dynamic`, dynamo's
  `maybe_mark_dynamic`). The first call then compiles once for every edge count.
- The compiled-model cache key includes `dynamic_edges`, so a static and a dynamic model are never
  shared.
- `fm.ForceField.evaluate` uses the same planner, so it gains the same benefit.
- Uncompiled plans (the CPU default, `compile_mode="none"`) are unaffected.

## Setup

- Machine: local PC (WSL2), NVIDIA GeForce RTX 5060 Ti 16 GB, driver 610.88, otherwise idle
  (`tools/ci/local_gpu.py` reservation, conductor informed).
- Environments: `~/environments/mace_cueq_latest` (plain cuEq) and `~/environments/mace_cueq_oeq`
  (hybrid), torch 2.14.1+cu130, `TORCHINDUCTOR_COMPILE_THREADS=4`.
- FastMACE: "before" is main `f8a3a8a` (its library equals the base `8c6e22e` of this branch);
  "after" is branch `fastmace/2-T13` at `31f42c5`. No research-engine baseline is involved.
- Model: MH-1, head `matpes_r2scan`, float32 forces, float64 energy, compiled tier.
- Structure: NaCl 2×2×2 (64 atoms), rattled 0.08 Å (seed 21), as 1b-T18. `ase.optimize.LBFGS`,
  `fmax=2e-3`.
- Script: [`results/2-T13-local-20261020/calc_compile.py`](../../results/2-T13-local-20261020/calc_compile.py).
  The raw JSON is in the same directory.
- Date: 20 October 2026.

## Compiles in an LBFGS relaxation

All runs take 41 steps (42 calls) and reach the minimum. The edge count takes 3 values (2062, 2060,
2048). "Calls that compiled" are the calls after which dynamo's graph count grew.

| Backend | Version | Calls that compiled | Graphs, after call 1 → end | Call 1 | Call 2 | Median of calls 3–42 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| plain cuEq | before | 2 | 1 → 2 | 37.3 s | 33.3 s | 13.05 ms |
| plain cuEq | after | **1** | 1 → 1 | 20.0 s | **0.016 s** | 12.58 ms |
| hybrid | before | 2 | 3 → 4 | 37.7 s | 23.1 s | 14.77 ms |
| hybrid | after | **1** | 3 → 3 | 22.8 s | **0.024 s** | 14.68 ms |

- **The second compile is gone** for both backends. The GPU test
  `tests/gpu/test_ase_calculator_gpu.py` now asserts this: no graph added after the first call while
  the edge count changes. It passed for both backends (`gpu_test.log`; hybrid: call 1 30.0 s, call 2
  0.03 s).
- Do not compare the call-1 times across versions. They depend on how warm the Inductor caches were:
  the "after" runs followed runs that had already compiled dynamic graphs.
- The median per call is unchanged, because main also runs a dynamic graph from call 3 on.

## Per-call time: static against dynamic edges at one edge count

The per-call cost only matters where main never compiles again: calls whose edge count stays
constant, so main keeps its static graph. This A/B used one process and one calculator on the start
structure (2062 edges), with 12 blocks in alternating order.
- Before each side, `torch._dynamo.reset()` dropped the compiled code, and the force field's
  `dynamic_edges` decided whether its edges were marked. So the static side ran static graphs and the
  dynamic side dynamic ones.
- Each side took 5 warm-up calls (which compiled) and then 100 timed calls.
- `calc` is `calculate` including the neighbour-list rebuild, which is what an optimiser pays.
  `model` is the force field's call alone, synchronised.

| Backend | Timing | Dynamic / static, mean of 12 | Median | 4-block windows | Standard error of the mean |
| --- | --- | ---: | ---: | --- | ---: |
| plain cuEq | calc | 1.005 | 1.007 | 1.027, 1.046, 0.942 | 2.1 % |
| plain cuEq | model | 1.017 | 1.009 | 1.024, 1.019, 1.009 | 1.4 % |
| hybrid | calc | 1.013 | 1.007 | 0.991, 0.993, 1.056 | 1.2 % |
| hybrid | model | 0.975 | 0.969 | 1.035, 0.895, 0.996 | 2.4 % |

- **No per-call slowdown can be told apart from noise here.** Every mean ratio is within about one
  standard error of 1. The largest is plain cuEq's model call: +1.7 % ± 1.4 %, which is above the
  12-block floor of 1.1 % (2-T12) but is not resolved.
- This A/B is noisier than the floor. Each block had 100 calls per side and a recompile between the
  sides, and block medians drifted by about 10 %. So it resolves about 2 %, not 2-T12's 1.1 %.
- An effect of 1 to 2 % on the model call alone, at a constant edge count, would need murphy (floor
  0.16 %) when it is idle.
- With warm caches, a dynamic compile took 5.3 s against 11 s static on plain cuEq, and 11 s against
  10 s on the hybrid.

## Tests

- `tests/cpu/test_calc_dynamic_edges.py`, fast tier:
  - the evaluation plan sets `dynamic_edges` (and not when uncompiled); MD plans do not;
  - `mark_edges_dynamic` gives one compile across three edge counts (`CompileCounter`);
  - the force field marks its edges and keys its compiled model by `dynamic_edges`.
- `tests/cpu/test_calc_dynamic_edges.py`, slow tier: a tiny MACE compiled on the CPU across three
  strained cells (three edge counts).
  - With `dynamic_edges`, it compiles no graph after the first call.
  - Without it, the second call compiles again.
  - Both match the uncompiled model within 1e-10.
- `tests/gpu/test_ase_calculator_gpu.py`: LBFGS with `fm.calc` on MH-1, one compile, both backends.
