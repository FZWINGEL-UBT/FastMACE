# OPT2-T04: NaCl's fixed costs (opt-2)

21 October 2026. Branch `fastmace/opt2-T04` on main 9914807 (with 2-T21d). Local RTX 5060 Ti (WSL2),
`mace_cueq_oeq`, MH-1 r²SCAN, NaCl 3×3×3 and NbCl5 3×3×3; murphy A/B pending. Results:
`results/opt2-T04-local-20261021/`.

## Where the fixed cost is

`tools/opt2_t04_probe.py`, the round's baseline (minimal planner default plus the OPT-T08 records),
hybrid FP32, NaCl at 1000 K, 1,000 pipelined steps after 300 of warm-up: 3.209 ms per step. Host
time per call, by wrapping the engine's methods:

| Part | per call | calls / 1000 steps | per step |
| --- | ---: | ---: | ---: |
| `_validate_pipelined` (the wait on the previous ACP's flags, early rebuilds included) | 2.96 ms | 1001 | 2.96 ms |
| early neighbour rebuild (`_rebuild`) | **4.19 ms** | 76 | 0.32 ms |
| graph replay (launch, host side, WSL) | 0.064 ms | 2003 | 0.13 ms |
| noise (`integrator.prepare_step`) | 0.061 ms | 1000 | 0.06 ms |

The validation wait, the replay launches and the noise run while the previous B graph still runs, so
they cost little GPU time. An early rebuild does not: its host syncs (the pair search's counter, the
kept-pair count) drain the stream, so it waits for the force graph in flight (about 2.9 ms on NaCl),
and the GPU then idles while the host launches the rest of the rebuild and the next step.

## Candidates (both opt-in `EngineConfig` fields, `False` by default; exact)

- **`overlap_rebuild`**: an early rebuild of a pipelined fixed-cell CUDA step runs on a side stream
  beside the force graph in flight, and the current stream waits for it before the next AC. The
  rebuild reads the positions, which the second half of a Langevin step does not write. A growth of
  the edge buffers waits for the force call and allocates on its stream, so the old buffers are not
  freed while it reads them. Hard rebuilds (a redo) and NPT keep the current path.
- **`step_graph`**: one `("ACPB", edges)` graph per pipelined step: ACP, the flags' copy to the
  host, an external event (a node of the graph, which the next validation waits on), then B. Saves
  one replay and the eager copy and event per step. The redo path keeps its B graphs; `prepare()`
  captures ACPB with its key's edge views, and the 2-T21d guard covers the key.

## Exactness

- CPU (`tests/cpu/test_overlap_rebuild.py`, `tests/cpu/test_step_graph.py`): the options are carried
  by the plan; on the CPU `overlap_rebuild` is a no-op and bitwise; `step_graph` eager and with the
  2-T21d recorder in place of the capture (padded edges) is bitwise the default, one ACPB capture per
  edge size, each reading its key's edges; the guard refuses a mismatched ACPB capture.
- GPU (`tests/gpu/test_fixed_costs_gpu.py`, local RTX 5060 Ti): `reproducible=True` (plain cuEq
  float64, unfused convolution, deterministic process), NaCl at 1000 K, 2 fs, 600 steps, pipelined,
  one fresh process per case: default, `overlap_rebuild`, `step_graph`, both. Positions and momenta
  bitwise equal in all four; 162 rebuilds, 122 of them early and all 122 beside the force graph with
  `overlap_rebuild`; the step-graph runs replayed ACPB graphs (B graphs only for redos). Passed in
  441 s.
- CPU suite (`tests/cpu`, `-n 2`, murphy CPUs): 2667 passed, 182 skipped. ruff clean.

## Timing (local RTX 5060 Ti, noise floor 1.1 % NaCl, 0.4 % NbCl5)

`tools/ab_rewrites.py` (now with `--a-engine`/`--b-engine` for boolean `EngineConfig` fields), same
process, both sides with the OPT-T08 records, 300 warm-up steps, 12 blocks of 300 steps, ABAB, one
structure and one build order per reservation; ratio of medians on/off:

| Option | Structure, T | order ab | order ba | all 12 pair ratios |
| --- | --- | ---: | ---: | --- |
| `overlap_rebuild` | NaCl, 1000 K | **0.9635** | **0.9674** | every pair 0.953–0.981 |
| `overlap_rebuild` | NbCl5, 1000 K | 0.9994 | 0.9927 | 0.990–1.016 |
| `step_graph` | NaCl, 1000 K | 0.9973 | 0.9999 | 0.984–1.042 (noise) |
| `step_graph` | NbCl5, 1000 K | 1.0015 | 1.0039 | 0.999–1.017 |
| both | NaCl, 300 K | **0.9839** | **0.9807** | 0.968–1.068 (one outlier pair each) |

- `overlap_rebuild`: NaCl −3.3 to −3.7 % at 1000 K (76 early rebuilds per 1000 steps), −1.6 to
  −1.9 % with both options at 300 K (fewer rebuilds); NbCl5 within or near its noise floor (rebuilds
  are under 1 % of its step). The gain scales with the rebuild rate, so with temperature and timestep.
- `step_graph`: no gain locally. The replay it saves runs while the previous B graph still runs, so
  its launch cost was hidden; NbCl5 +0.15/+0.39 % is at the noise floor. Kept as an opt-in only for
  the murphy A/B (native Linux, floor 0.16 %), where a launch costs differently; if it shows nothing
  there either, it should be dropped.
- Not built: noise selection inside the graph (0.06 ms of host time per step, hidden like the
  replays), and fusing kernels inside the B replay (147 µs of gaps between 264 kernels): that is a
  model-compilation change, not an engine one.

## Murphy A/B (pending, before Thursday 16:00 UTC)

The same four A/Bs (each option off/on, NaCl and NbCl5, 1000 K, both orders) and both options at
300 K, from a pinned copy of this branch, when murphy has no other compute process.
