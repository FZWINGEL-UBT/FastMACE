# Kernel review runs (5–6 November 2026)

The report is [docs/validation/kernel-review-20261105.md](../../docs/validation/kernel-review-20261105.md).

**Workload:** OMAT-0 medium, float32 hybrid, Benchmark_loup SrTiO3 10×10×10 (5,000 atoms), 1 fs, skin 0.5 Å.
**Modal profile:** `third` (fzwingel54).

| Directory | Commit | Machine | What |
| --- | --- | --- | --- |
| `runs/h200-lab-ab/` | `c675028a` | Modal NVIDIA H200 (driver 610.57.04), hybrid image | `hopper_sweep.sh` with `ALIAS='NVIDIA H200'`: lab forward (base, F-x-nodep, sh-lds4, F-tc-no-redistribute, F-no-xload), lab w′ (base, restrict, sh-hoist (pattern gone, not run), F-no-atomics), MD A/Bs `speciescut` (SPECIES_GEMM + species_gemm.CUTLASS), `tabfw2` / `tabbw2` (tabled forward / backward 2 warps); about $1.11 |
| `runs/rtx5060ti-lab/` | `c675028a` | local RTX 5060 Ti, WSL `mace_cueq_oeq` | the same lab variants |
| `runs/rtx5060ti-ab-shlds4/` | `5b4431fe` | local RTX 5060 Ti | MD A/B, SH_LDS4 on (b) |
| `runs/h200-ab-shlds4/` | `5b4431fe` | Modal NVIDIA H200 | lab base / sh-lds4 (0.989) and MD A/B, SH_LDS4 on (b); about $0.31 |
| `runs/a100-ab-shlds4/` | `5b4431fe` | Modal A100-SXM4-80GB | the same (lab 0.993; base 64 B local); about $0.18 |

Lab ratios (variant / base, fused forward or w′ kernel):

| Variant | H200 | RTX 5060 Ti |
| --- | ---: | ---: |
| forward base | 1,684 µs | 6,845 µs |
| sh-lds4 (bitwise) | 0.989 | 0.972 |
| F-x-nodep | 1.094 (8 B spill) | 0.989 |
| F-tc-no-redistribute | 0.998 | n/a (SIMT) |
| F-no-xload | 0.900 | 0.930 |
| w′ base | 1,564 µs | 6,944 µs |
| restrict (bitwise) | 1.031 | 1.016 |
| F-no-atomics | 1.080 | 0.979 |

MD step b/a (ab / ba):

| A/B | H200 | RTX 5060 Ti | A100 |
| --- | --- | --- | --- |
| SH_LDS4 | 1.0014 / 1.0027 | 0.9932 / 0.9902 | 1.0008 / 0.9949 (A100-SXM4-80GB) |
| SPECIES_GEMM + CUTLASS | 1.0141 / 1.0175 | | |
| tabled forward 2 warps | 1.0147 / 1.0168 | | |
| tabled backward 2 warps | 1.0270 / 1.0295 | | |
