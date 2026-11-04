# OMAT-0 tensor-product backward screen (4 November 2026)

Report: `docs/validation/omat0-tp-backward.md`. Model MACE-OMAT-0 medium (checkpoint sha256 d4b14be9…), float32 default
plan, Benchmark_loup SrTiO3 10×10×10 (5,000 atoms, 373,760 edges). Branch `fastmace/omat0-tp-backward`, from
`fastmace/omat0-strided-tables` at 4b70e3f2.

leto: NVIDIA RTX 4000 Ada Generation, driver 595.99.02, NAS `mace_cueq_oeq` (torch 2.14.1+cu130), FastMACE run from a
`git archive` copy with `PYTHONPATH` (`/mnt/homes/zwingel/fastmace-branches/tp-bwd-<sha>`), idle check
(`idle_check.txt`) before every job. A100: Modal A100-SXM4-80GB, driver 610.57.04, image of `requirements-hybrid.txt`
(`pip-freeze` hash in `modal.json`), workspace profile `second`, `MODAL_IMAGE_BUILDER_VERSION=2025.06`.

| Folder | FastMACE | What |
|---|---|---|
| `rtx4000ada-lab-falsifiers` | 51906e31 | F-tp-no-coef 0.947, F-tp-wg-only 0.737, F-no-tp 0.640, sh-hoist 1.095, F-no-slope 1.109, F-no-atomics 1.063 |
| `rtx4000ada-lab-factored` | e9980251 | factored 0.903 |
| `rtx4000ada-checks` | 78b27fd2 | GPU test (pass), force gates OMAT-0 / MH-1 default vs FACTORED (`gate-table.txt`, all pass), MD step A/B b = FACTORED 0.9832 / 0.9801 |
| `rtx4000ada-cpu-suite` | 78b27fd2 | CPU suite 5,128 passed, 202 skipped (`mace_cueq_latest`, 16 workers) |
| `rtx4000ada-lab-maxreg` | 52c13b2b | factored 0.901; maxreg 80 / 72 / 64: 1.243 / 1.467 / 2.700 (spills) |
| `rtx4000ada-lab-pair` | 02a6ccd0 | channel pair 0.898 (128 registers), its maxreg 128 0.913, maxreg 96 1.548 |
| `rtx4000ada-lab-diagnostics` | 6852e188 | on factored: one slab 0.751, no table 0.599, no dout 0.667, no x 0.879, no atomics 0.901 |
| `rtx4000ada-lab-locality` | c01b31c6 | degree 5: factored+unordered 0.877, unordered 0.984 (bitwise); degree 7 float4: base 7.97 ms, factored 1.009 |
| `rtx4000ada-ab-factored-sym` | bcb1bf28 | MD step b = FACTORED + SYMMETRIC_PAIRS 0.9792 / 0.9747 |
| `a100-sxm4-bcb1bf28` | bcb1bf28 | lab: factored 0.727, +unordered 0.721, pair 0.713, one slab 0.683, wg-only 0.588; MD step FACTORED 0.9381 / 0.9397, FACTORED + SYMMETRIC_PAIRS 0.9365 / 0.9307 ($0.52) |
| `a100-sxm4-scale-dcde706e` | dcde706e | GPU test pass; lab factored 0.722, factored-scaled 0.830, pair 0.708; MD step a = FACTORED, b = + FACTORED_SCALE 1.0247 / 1.0238 ($0.29) |
| `a100-sxm4-gate-8aec363b` | 8aec363b | force gates on the A100 default plan, default vs FACTORED |
| `rtx4000ada-cpu-suite-final` | 8aec363b | CPU suite 5,128 passed, 202 skipped; ruff: 3 long lines (in main) + 1 in `wprime_backward.__all__`, fixed after |

Lab ratios are medians of 21 round-robin rounds against the default kernel captured in the same process. MD step A/Bs:
same process, 200 warm-up, 8 blocks of 200 steps, abab and abba, skin 0.5 Å, 1 fs.
