# Codex H200 report (8 November): COMPOSE_SKIP fast path, generic readout-tail fold, linear_div split, cuEq projection, staged edges

8 November 2026. Modal H200 (`NVIDIA H200`, driver 610.57.04, sm_90), profile `sixth`, packed jobs. OMAT-0 medium,
SrTiO3 10x10x10 (5,000 atoms), Benchmark_loup settings (Langevin 300 K, 1 fs, skin 0.5, seed 1). A/B: a = the H200
defaults of main `909bc8cd` (plus the switches adopted before the round), b = the variant, `--warmup 200 --block 256
--blocks 6`, both build orders.

Branches: `fastmace/h200-fastpath` (bitwise class: `COMPOSE_FAST`, `LINEAR_DIV_SPLIT`, `CUEQ_PROJECTED`,
`STAGE_EDGES`, `STAGE_SH`) and `fastmace/h200-fold-tail` (rounding class: `GENERIC_FOLD_READOUT_TAIL`). The jobs ran on
a merge of both (`fastmace/h200-fastpath-job`, not for merging).

## Round 1 (`tools/modal/fastpath_job.sh`, source `3c5ed795`, [round1/](round1/run.log))

| Switch | What | MD step b/a (ab / ba) | Class, check | Decision |
| --- | --- | --- | --- | --- |
| `compose_skip.COMPOSE_FAST` | COMPOSE_SKIP's composed weights bound once per table (refreshed in place), the four degree products as one `species_mm_joined` (CUTLASS species_into per path into one buffer), the division returns a view | **0.9614 / 0.9656** (4.015 → 3.860 ms) | bitwise; forces 3.3e-6 against same-build 2.7e-6 eV/Å | **adopted (H200)** |
| `linear_div.LINEAR_DIV_SPLIT` | linear_div's backward writes the divided path blocks contiguously in one Triton kernel | 0.9972 / 1.0025 | bitwise | not faster in both orders: off, closed |
| `cueq_projection.CUEQ_PROJECTED` | cuEq SymmetricContraction's `einsum(weight, projection)` kept as a buffer instead of recomputed per call | **0.9902 / 0.9954** | rounding (einsum order); gate passed: forces 0.922 / 0.900 / 0.979, stress 0.425 / 0.852 / 1.206 | **adopted (H200)** |
| `species.GENERIC_FOLD_READOUT_TAIL` | the generic product's species skip composed with the following readout linear | **0.9855 / 0.9901** | rounding; gate passed: forces 0.848 / 0.920 / 0.968, stress 0.437 / 0.824 / 1.043 | **adopted (H200)** |
| all four | | 0.9497 / 0.9509 (4.018 → 3.816 ms) | | |

The joined species_mm is bitwise against per-path `species_mm` (`tests/gpu/test_species_mm_joined_cuda.py`, eager and
compiled, ieee / tf32x3 / tf32x3-cutlass: 6 passed on the H200, [pytest-joined.log](round1/pytest-joined.log)). The
gate ratios are SrTiO3 / NaCl / NbCl5 against the ASE float32 reference, limit 3
([gate-tail](round1/gate-tail/summary.txt), [gate-cueq](round1/gate-cueq/summary.txt)).
