# 2-T02: eager tier for MD

Plan task 2-T02 (`plans/phase-2.md`); SPEC §8.3, §9.3 (mode dimension, rules 3 and 4), §9.4 step 2.

## What it is

`Simulation(..., mode="eager")` (and `MinimalPlanner.plan(..., mode="eager")`) plans the graph
plan's backend, rewrite set, precision, neighbours and integrator, and runs them on the graph
engine without CUDA graphs and without compilation:

- plan tier `"eager"`; decisions `capture=off` ("eager tier: no CUDA graphs") and
  `EngineConfig.compile_mode=none`, so the plan claims no compile family (§8.7);
- no speculative combined graphs (they need capture); `pipelined` keeps its resolved value
  (conductor answer), so on the CPU eager and graph run the same code;
- exact edge counts, so rule 4 (inexact padding) is lifted for the eager tier only;
- PolarMACE on plain cuEq plans in eager mode (rule 3: "hybrid, or eager plain");
- the requested options stay the caller's: the eager engine's `compile_mode` is not recorded as
  requested;
- `FixedPlanner.build(..., capture=False)` builds it; a tier `"eager"` plan with any other
  `capture` is refused. 2-T04b can force eager from `capture_support` by planning tier `"eager"`.
- The report's `why:` line says "eager tier: no CUDA graphs, no compilation".

Same integrators, output, restart (`Simulation.resume` re-plans eager from `simulation.json`) and
gate.

## CPU

`tests/cpu/test_eager_tier.py` (11 tests), on murphy's CPUs (`mace_cueq_latest`,
`CUDA_VISIBLE_DEVICES=`, `-n 0`): all pass. Eager equals graph **bitwise** over 50 steps on a tiny
MH-1-class model, float64, NaCl, NVT Langevin and NPT BAOAB (positions, forces, energy, cell).
A gated `Simulation(mode="eager")` passes the gate, saves a restart, and resumes on the eager tier.
The related files (`test_minimal_planner`, `test_planner`, `test_report`, `test_simulation`,
`test_simulation_options`, `test_contracts*`, `test_research_surface`, `test_engine`,
`test_selfcheck`, `test_layering`, `test_restart`) pass: 565 passed, 20 skipped.

## GPU (murphy)

murphy, NVIDIA RTX 4000 Ada Generation 20 GB, torch 2.14.1+cu130, NAS environments
`mace_cueq_latest` with `mace_cueq_oeq` as `FASTMACE_HYBRID_PYTHON`,
`FASTMACE_MODELS=/mnt/homes/zwingel/ResearchModule/Models`; FastMACE at a87a402 (branch
`fastmace/2-T02`); no baseline involved. Untimed: 2-T07's tests shared the GPU (zwingel's own
processes, allowed for untimed work). `tests/gpu/test_eager_gpu.py`, 2 passed in 155 s; raw JSON
and junit in `results/2-T02-murphy-20261020/`.

**MH-1 hybrid float32, NaCl 3×3×3 (216 atoms)**, NVT Langevin 300 K, 2 fs, 100 steps, one fresh
hybrid process: eager with `check="gate"`, then graph twice from the same seed.

| Quantity | Eager vs graph | Graph repeat vs graph |
| --- | ---: | ---: |
| max position deviation (every 10 steps) | 5.4e-7 Å | 2.5e-7 Å |
| max energy deviation | 9.6e-5 meV/atom | 4.3e-5 meV/atom |

Eager is within the float32 divergence of two graph runs. Gate in eager mode: passed (2 rattled
copies; 7e-5 meV/atom, 3e-6 eV/Å). Eager plan: tier `eager`, hybrid, float32, `compile_mode`
none, no capture; graph plan: compiled and captured.

**POLAR-1-S, plain cuEq (fused convolution) float32, eager**, same structure and integrator,
20 steps: gate passed (0.00037 meV/atom, 3.4e-6 eV/Å), energies finite. Plain compiled cuEq is
refused for PolarMACE (rule 3); eager plain is the combination rule 3 allows.

Step times in the JSON are whole-run means on a shared GPU, including setup effects. They are
not timing results.
