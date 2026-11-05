# 1b-T18: `fm.calc` on the GPU

Plan task 1b-T18 (SPEC §7.4, §7.7 level 0). Test:
`tests/gpu/test_ase_calculator_gpu.py::test_lbfgs_with_fm_calc_reaches_maces_minimum`.

- Machine: local PC (WSL2), NVIDIA GeForce RTX 5060 Ti, driver 610.88, idle (desktop only).
- pytest in `~/environments/mace_cueq_latest`; the `hybrid` case's child in
  `~/environments/mace_cueq_oeq` (`FASTMACE_HYBRID_PYTHON`). `TORCHINDUCTOR_COMPILE_THREADS=4`.
- FastMACE: branch `fastmace/1b-T18` on main `c21ff19`. No research-engine baseline is involved.
- Model: MH-1, head `matpes_r2scan` (`FASTMACE_MODELS=~/models`). Our side: `fm.calc`,
  float32 forces, float64 energy accumulation, compiled tier. MACE's side: `MACECalculator`,
  float64, same checkpoint.
- Structure: NaCl 2x2x2 (64 atoms), rattled 0.08 Å (seed 21). `ase.optimize.LBFGS`,
  `fmax=2e-3`, at most 300 steps, the same start for both.
- Date: 19 October 2026.

## Results

| Child | Backend chosen | LBFGS steps (ours / MACE) | Energy, ours − MACE (meV/atom) | Max position difference (Å) | Our total time | MACE total time |
| --- | --- | --- | --- | --- | --- | --- |
| `mace_cueq_latest` | plain cuEq | 41 / 41 | 6.7e-4 | 6.3e-7 | 34.8 s | 5.3 s |
| `mace_cueq_oeq` | hybrid | 41 / 41 | −1.3e-4 | 5.4e-7 | 62.6 s | 5.5 s |

Both relaxations reach MACE's minimum. Before this run, a plain-cuEq run without call timing gave
the same numbers within rounding: 41 / 41 steps and a 4.2e-7 Å position difference.

## Time per call (42 calls per relaxation)

| Backend | Call 1 (plan, build, compile) | Call 2 | Median of the rest |
| --- | --- | --- | --- |
| plain cuEq | 29.7 s | 4.5 s | 15 ms |
| hybrid | 42.8 s | 19.1 s | 16 ms |

MACE's float64 calculator takes about 130 ms per call on this structure. That is not a speed
comparison: its precision differs, and the GPU was not reserved for timing.

The second call compiles again: probably the edge count changed and `torch.compile` switched to
dynamic shapes. For short tools such as phonopy or NEB, which make few calls, the two compiles take
most of the time. Making the evaluation plan's compile dynamic from the first call would avoid the
second compile. That is a decision for the planner and `model/compile.py`, outside T18's files.
