# 3-T22b: the fixcm projection in the planner, the record and the fused stage

Plan task 3-T22b, 22 October 2026, follow-up of [3-T22](3-T22-fixcm.md). The design is in
[design/fixcm-projection.md](../design/fixcm-projection.md). Branch `fastmace/3-T22b`, from main eb64374.

## What changed

- **`IntegratorConfig.fixcm_projection` defaults to `"auto"`** (the conductor's decision). Every plan resolves
  it and records the decision `IntegratorConfig.fixcm_projection`:
  - `"mass_weighted"` under NVT `fixcm` with MC moves declared (`fixcm-projection-mc`);
  - otherwise `"ase"` (`fixcm-projection-ase`), so MD plans are unchanged bit for bit;
  - an explicit value is kept (`fixcm-projection-given`).

  Outside a plan, `"auto"` reads as `"ase"`.
- **`MCDriver.attach`** refuses `redraw_all` under `fixcm` unless the projection is `"mass_weighted"`
  (`mc-redraw-all-ase-projection`).
- **The public `fm.integrators.Langevin` record** has `fixcm_projection`, default `"auto"`.
- **The fused Triton stage implements both projections**, bit for bit against the PyTorch stage. For
  `"mass_weighted"`:
  - the noise kernel also stores rnd_pos·m;
  - the two centre-of-mass sums stay PyTorch reductions;
  - the drift kernel subtracts sum/M, with M the derived buffer `fused.total_mass`, which a mass event
    refreshes.

  `fixcm_projection` no longer turns the fused stage off.

## Tests

- **CPU:** WSL, `fastmace_cpu`, float64, `-n 2`: 692 passed, 29 skipped over the planner, MC driver, fixcm,
  contracts, config, simulation, restart, fused-integrator, events, ASE, integrator, quick, CLI and
  subprocess-runner files. Ruff is clean.
  - `tests/cpu/test_fixcm_projection.py`:
    - the plan's decision for MD, MC, fixcm off, NPT and the explicit values;
    - MD plans are bit for bit the same with `"auto"` and an explicit `"ase"`;
    - the `redraw_all` refusal with an explicit `"ase"`, `exchange` allowed, and `"auto"`'s MC plan
      allowed.
- **CPU, the kernels themselves:** `tests/cpu/test_fused_projection_interpreter.py` runs the Triton kernels
  under `TRITON_INTERPRET=1`, with the GPU environment's Python and no GPU; 1 passed.
  - `rnd_pos` and `rnd_vel` are bit for bit equal to `Langevin.first_half` for both projections.
  - Positions and velocities agree to 1e-15 relative. On the CPU PyTorch divides by dt exactly, while the
    kernel multiplies by 1/dt, as PyTorch's CUDA kernel does.
  - The CPU environment has no Triton, so the test is skipped there.
- **GPU:** RTX 5060 Ti, `mace_cueq_latest`, through `tools/ci/local_gpu.py`;
  `tests/gpu/test_fused_projection_gpu.py` (fast tier) with `tests/gpu/test_events_gpu.py`.
  - Argon with alternating masses (40 and 4 amu), the deterministic Lennard-Jones force field, captured
    graphs, 200 steps with rebuilds, checked and pipelined: the fused stage on and off give the same
    positions and velocities bit for bit, for both projections.
  - With `"mass_weighted"` the total momentum stays at zero to rounding.
  - After a mass event, the refreshed `fused.total_mass` keeps the fused stage equal to PyTorch's.

## Step time on NaCl

`tools/fused_projection_timing.py`, results in `results/3-T22b/timing.json`.
- **Setup:** RTX 5060 Ti, `mace_cueq_latest` (torch 2.14.1+cu130), MH-1 (`mh1-r2scan`), the planner's
  default plan in this environment (plain cuEq, float64, pipelined, captured, fused stage on). NaCl 3×3×3
  (216 atoms), Langevin 1 fs, 600 K.
- **Method:** two engines in one process, alternating blocks of 500 steps, 10 blocks each after a warm-up.
  The fused stage alone was also timed with CUDA events, called eagerly, 200 calls per round, 10 rounds.

| Projection | Step, median (min–max) | Fused stage alone, median (min–max) |
| --- | --- | --- |
| `ase` | 6220.0 µs (6185.2–6242.2) | 151.0 µs (96.3–215.3) |
| `mass_weighted` | 6214.7 µs (6183.9–6235.0) | 133.9 µs (102.1–200.0) |

**Reading:**
- The step times differ by 0.09 %, below this machine's NaCl noise floor (1.1 %).
- The eager stage calls are dominated by launch overhead, so their spread is wide and their medians
  overlap. In the captured step they are part of the step time above.
- The mass-weighted projection costs nothing measurable, so a later default change keeps the round-3
  fused stage.
- The plain-cuEq float64 plan is this environment's planner choice (no OEq here); the hybrid production
  plan was not timed.
