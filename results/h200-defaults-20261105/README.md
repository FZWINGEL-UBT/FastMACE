# H200 defaults under decision 66 (5 November 2026)

Modal NVIDIA H200 (sm_90, 132 SMs), OMAT-0 medium, float32 hybrid, Benchmark_loup SrTiO3 10x10x10 (5,000 atoms,
373,760 edges), 1 fs. Profiles `second`, `fzwingel` and `third`; total about $12.2. Every job went through
`tools/modal/fastmace_modal.py` with the budget guard and a timeout.

Before this batch the H200 was in no decision-66 GPU set and ran the generic plan (13.12 ms per step,
`results/omat0-h200-profile-0ab2bd9a/`). Under the A100-SXM4 plan (alias) it ran 5.66 ms
(`results/hopper-sweep-h200-20261105/`).

## Result

**Benchmark_loup Langevin on the real device name at 62556704: 5.778 / 5.756 / 5.731 ms per step (1.146–1.156 µs per
atom and step)** (`G/`, three runs). The force/stress gate passes: force ratios 0.74 / 0.97 / 0.99, stress
1.02 / 1.93 / 0.93 (NaCl, NbCl5, SrTiO3). Symmetrix-XL on the same H200: 14.86 ms.

The H200 is now in these sets; the MD-step b/a in both build orders (b = switch off, unless noted):

| Set | b/a |
| --- | --- |
| planner `DECISION_66_GENERIC_RADIAL_TABLES_GPUS` | 2.030 / 2.027 |
| planner `GENERIC_WPRIME` | 1.771 / 1.768 |
| w′ `FACTORED` | 1.160 / 1.159 |
| w′ `TABLE_W` | 1.142 / 1.141 |
| fused forward `TENSOR_CORES` | 1.107 / 1.105 |
| `SKIP_ZERO_COLUMN_GPUS` (02a81e13) | 1.092 / 1.090 |
| planner `TF32X3_OMAT0_GPUS` | 1.068 / 1.069 |
| w′ `BLOCK_MESSAGE` | 1.066 / 1.065 |
| generic `FUSED_FIRST_BACKWARD` | 1.051 / 1.052 |
| fused forward `VEC_A` | 1.047 / 1.046 |
| w′ `HORNER` | 1.046 / 1.046 |
| planner `GENERIC_FOLD_PRODUCT` | 1.033 / 1.032 |
| w′ `SCALAR_PATHS` | 1.030 / 1.030 |
| fused forward `TC_WFRAG` | 1.022 / 1.024 |
| generic `EDGE_SH` | 1.019 / 1.018 |
| energy `HEAD_SLICE` | 1.016 / 1.017 |
| radial `SYMMETRIC_PAIRS` | 1.005 / 1.004 |
| energy `GRAPH_SUM` | 1.005 / 1.004 |
| fused forward `VEC_SH` | 1.003 / 1.003 |
| w′ `BUTTERFLY` (b = on) | 0.994 / 0.994 |
| radial-table forward 8 warps (`WARPS_GPUS`, 62556704; b = on) | 0.9946 / 0.9950 |

Not adopted: `BACKWARD_WALK` 1.0000 / 1.0001, `SEGMENT_ROWS` 1.0006 / 1.0000, radial-table forward 2 warps
0.9984 / 0.9990, column block 32 1.0025 / 1.0035 (b = candidate). `TABLED_FORWARD`: ab order 1.0124, but the ba
order failed twice on the H200 with a CUDA-graph capture error (stale AccumulateGrad on recapture in the pipelined
step; not reproduced locally at 6x6x6), so it stays off there.

## Folders

- `abA/`, `abB/`, `abC/`, `abD/`: the decision-66 MD-step A/Bs (`tools/modal/hopper_sweep.sh`).
- `explore/` (27d27f21): loup, gate, **step breakdown** (`explore/out/breakdown/`, before skip-zero: force call
  6.36 ms; fused forward 2,328 µs, w′ backward 1,606 µs, tabled outer-sum backward 240 µs, tf32x3 GEMMs about
  0.9 ms, radial tables 2 x 115 µs) and the kernel-lab launch shapes and falsifier bounds.
  - Fused forward (base 2,323 µs): blocks per SM 1/3/4 1.141/1.112/1.076; receivers per chunk 1/3/4
    0.992/1.038/1.094. Bounds: no product 0.527, no MMA 0.705, one pass 0.717, no tensor product 0.669, no x load
    0.830, no redistribute 0.972.
  - w′ (base 1,593 µs): no atomics 1.089, one slab 0.925.
- `abD/` (profile `fzwingel`): retests of ideas rejected elsewhere: w transposed rows 80 1.0193 / 1.0198,
  species-grouped GEMM 1.0130 / 1.0139, cross-step overlap 1.029, batch 4 errored in both orders; the
  producer/consumer microbench was 1.1–1.5x slower than serial at every split; the megakernel probe failed.
- `E/`: radial-table warps (adopted 8), column block 32, full first interaction 1.1110 / 1.1129 (loses).
- `F/`: no-a-load bound 0.864; SIMT slab with the real weights 0.887 (A100 1.333), `simt-slab3` 0.679 (base without
  skip-zero).
- `G/` (02a81e13 defaults): loup x3, gate, lab against the skip-zero base (1,786 µs): `simt-slab-real` 1.076
  (loses now), `simt-slab3` 0.838 (a lab bound with values from the table, not a correct kernel; a real
  implementation that reads w from the value table in the forward would be needed to claim it).
- `int8/`, `int8-cublas-only/`: cuBLASDx INT8 emulation of the float32 GEMMs (Ozaki slices). 4 slices 3.0–3.3x
  slower than SGEMM (13.7k x 512 x 512: 545 µs against 182 µs); 2 and 3 slices look like a fallback. Rejected.

Trace files and binary profiles are left out (Modal volume / local `~/fm-runs/` only).
