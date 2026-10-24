# FastMACE against TorchSim: MD step time (25 October 2026)

Owner request (24 October 2026): "Can you also do a speed comparison test compared to torchsim?" Recorded, not
judged (phase 5 runbook).

## Setup

- **GPU:** local RTX 5060 Ti (sm_120), WSL2, one quiet `local_gpu.py` reservation. Every run is its own process,
  in two alternating rounds (FastMACE, then TorchSim's three compile modes, per case).
- **FastMACE:** main `5a3166c`, `fm.Simulation` with the planner's default plan (hybrid cuEq/OEq, float32),
  `fm.integrators.Langevin`, in `~/environments/mace_cueq_oeq` (Python 3.10, torch 2.14.1+cu130, cuEq 0.12.0,
  OEq 0.7.0, MACE 0.3.16).
- **TorchSim:** `torch-sim-atomistic` 0.6.2 with MACE's own `MaceTorchSimModel` (MACE 0.3.16), cuEquivariance
  on, float32, no stress. `nvt_langevin_init` and `nvt_langevin_step` (BAOAB) run in a loop, without reporting
  overhead. Compile modes: `none`, `default`, `reduce-overhead`.
  - TorchSim needs Python 3.12 or newer, so it runs in a separate evaluation venv,
    `~/environments/torchsim_eval` (Python 3.12.3, the same torch, cuEq and MACE pins, NumPy 2.5.4; its list
    is in `results/torchsim-comparison-20261025/`).
  - MACE's TorchSim wrapper fails with OEq ("TensorProduct.forward() takes from 3 to 4 positional arguments
    but 5 were given"), so TorchSim ran cuEq only.
- **Workload:** 3x3x3 supercells of `Structures/Benchmark/R2SCAN` NaCl (216 atoms) and NbCl5 (1,944 atoms),
  Maxwell-Boltzmann at 300 K (seed 0), 1 fs, friction 0.01/fs, one structure (production's case). FastMACE ran
  300 warm-up steps and 3 blocks of 1,000; TorchSim 50 warm-up steps and 3 blocks of 200.
- **Scripts and records:** `results/torchsim-comparison-20261025/`.

## Result (median ms per step over 6 blocks)

| Model | Structure | FastMACE | TorchSim none | TorchSim default | TorchSim reduce-overhead | FastMACE speed-up (vs TorchSim's best) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MH-1 (r2scan head) | NaCl | 2.90 | 39.5 | 25.1 | 22.2 | 7.7× |
| MH-1 (r2scan head) | NbCl5 | 21.3 | 116.7 | 82.5 | 81.5 | 3.8× |
| OMAT-0 medium | NaCl | 1.62 | 36.9 | 24.7 | 16.3 | 10× |
| OMAT-0 medium | NbCl5 | 9.79 | 47.2 | 39.5 | 39.4 | 4.0× |

- FastMACE's blocks agree within 1 %, TorchSim's within 3–10 %. Every run completed, and TorchSim's thermostat
  ended at 290–301 K.
- **Batched, recorded once:** TorchSim on 8 copies of NaCl (OMAT-0 medium) takes 38.0 ms per step (`default`),
  4.7 ms per structure. That is still 2.9× FastMACE's single-structure step.
- **Setup:** TorchSim 8–39 s (the compile modes cost more), FastMACE 22–32 s with its caches warm.

## Reading

TorchSim targets batched high-throughput MD; FastMACE targets one long trajectory at graph speed (SPEC's risk
table). On that single-structure workload FastMACE is 3.8–10× faster here. The gap is largest on small cells,
where launch overhead dominates and FastMACE's CUDA graphs, exact model rewrites and cuEq/OEq hybrid count most.
Not covered: other GPUs, TorchSim's neighbour-list options, float64 and NPT.
