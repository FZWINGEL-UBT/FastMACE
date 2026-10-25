# Benchmarks

MD step times of FastMACE 0.x on its default plan: hybrid backend (cuEquivariance + OpenEquivariance), float32 with
float64 energies, TF32 off, CUDA graphs, NVT Langevin at 300 K, 3×3×3 supercells. Lower is better. Each number is
the median of 4 timed blocks of 300 steps after 300 warm-up steps, in one process per model.

## NVIDIA L4 (Colab)

Run `6-T08-colab-l4-20261025`, 25 October 2026. Provider Google Colab, NVIDIA L4 (Ada, sm_89, 23 GiB), driver
580.82.07, torch 2.14.1+cu130, cuEquivariance 0.12.0, OpenEquivariance 0.7.0, MACE 0.3.16, FastMACE `c1f4de9c`.
Tool: `tools/ab_step_time.py --sides fastmace`. Records: `results/6-T08-colab-l4-20261025/`. Cost 0.92 compute
units.

| Model | NaCl (216 atoms) | NaNbO3 (1,080) | NbCl5 (1,944) | Peak memory | Setup (NaCl) |
| --- | --- | --- | --- | --- | --- |
| MH-1 (`matpes_r2scan`) | 3.95 ms | 31.5 ms | 35.2 ms | 3.7 GiB | 58 s |
| MP-0 medium | 1.71 ms | 17.6 ms | 13.6 ms | 2.0 GiB | 42 s |
| MPA-0 medium | 1.93 ms | 20.0 ms | 15.9 ms | 2.0 GiB | 36 s |
| OMAT-0 medium | 1.93 ms | 20.0 ms | 16.2 ms | 2.0 GiB | 36 s |
| POLAR-1-S | 3.66 ms | 12.4 ms | 32.3 ms | 2.5 GiB | 45 s |

At 1 fs per step, 3.95 ms per step is about 22 ns of MD per day for NaCl with MH-1.

**Caveats.** A cloud VM is not a quiet machine: the block spread was up to 20–43 % (one slow block per model, the
first to see a neighbour-list bucket change), so the medians are the honest summary and small differences between
models are not significant. These are FastMACE-only times. Speed-ups against stock ASE and the research engine come
from the alternating A/B runs on the RTX 5060 Ti (`docs/validation/`, SPEC §3.3). MH-1, OMAT-0 and POLAR-1
weights are under the Academic Software License (non-commercial); MP-0 and MPA-0 are MIT (label table).

## RTX 5060 Ti (local reference GPU)

The timed acceptance numbers are in the phase reports, for example the phase 5 regression A/B: MH-1 NaCl 2.87 ms,
NbCl5 21.0–21.3 ms (`docs/validation/phase-5.md`). A full table on this GPU is still to be measured.
