# 1b-T17 `Simulation`: GPU smoke check

Plan task 1b-T17, 19 October 2026. Test: `tests/gpu/test_simulation_gpu.py`; results:
`results/1b-T17-rtx5060ti-20261019/` (JSON per backend, launch script, idle check).

## Setup

- Machine: local PC, WSL2; GPU NVIDIA GeForce RTX 5060 Ti (idle: desktop processes only, 0 %
  utilisation before the run); torch 2.14.1+cu130.
- Environments: `mace_cueq_latest` (plain cuEq) and `mace_cueq_oeq` (hybrid, through
  `FASTMACE_HYBRID_PYTHON`), FastMACE from the checkout with `PYTHONPATH`.
- FastMACE: branch `fastmace/1b-T17` at `af63c19` plus the test fix committed with this note. No
  research-engine baseline is involved: the comparison is with FastMACE's own `FixedPlanner`, the
  engine `main.py --engine fastmace` builds.
- Model `mh1-r2scan` (MACE-MH-1, head `matpes_r2scan`), float32 forces, float64 energy
  accumulation. NaCl 3×3×3 (216 atoms), Maxwell–Boltzmann start at 300 K, NVT Langevin 300 K,
  2 fs, friction 0.02 per ASE time unit (`main.py`'s), seed 20261019, 60 steps.

One fresh process per backend runs `fm.Simulation` first (with output: thermo and frames every 10
steps through the asynchronous chunked writer), then twice the `FixedPlanner` engine
(`reference`, `repeat`). Positions and energies are compared every 10 steps.

## Results

| Backend | Simulation vs FixedPlanner, max |Δx| (Å) | repeat vs FixedPlanner (Å) | Energy (meV/atom): Simulation / repeat | Rewrites and engine options |
| --- | --- | --- | --- | --- |
| plain cuEq float32 | 1.7e-7 | 1.2e-7 | 2.5e-5 / 2.4e-5 | equal |
| hybrid float32 | 1.3e-7 | 1.6e-7 | 4.2e-5 / 4.2e-5 | equal |

`Simulation` stays within float32 divergence of the `main.py` path: no further from it than a
second `FixedPlanner` run. Both runs are pipelined, the gate is `skipped` (default, D1), the
thermo file has rows 0, 10, …, 60, its energies equal the committed ones (1e-6 eV/atom, CSV
rounding), and the last frame equals the committed positions.

Setup took 47 s (plain cuEq) and 27 s (hybrid) in these processes. The whole-run step-time mean
over 60 steps (0.55 s and 0.17 s) is dominated by compilation and capture, so it is not a speed
figure. The report's steady-state window (1,000 steps after 1,000 warm-up steps) was not reached.
