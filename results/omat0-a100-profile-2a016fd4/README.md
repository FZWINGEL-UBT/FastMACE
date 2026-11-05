# OMAT-0 medium A100 profile at 2a016fd4 (5 November 2026)

Modal A100-SXM4-80GB (driver 610.57.04), FastMACE `2a016fd4` (round 5 start), torch 2.14.1+cu130, float32 hybrid.
Benchmark_loup SrTiO3 10x10x10 (5,000 atoms): NVE and Langevin step timings (`loup-*.log`, about 11.7 ms per step) and
the step breakdown (`breakdown/`: per-category, per-op, GEMM and kernel-sequence tables; `summary.json` has the
versions). This profile chose the round-5 work: the GEMM layout glue (`clone_slice` and `cat` copies, about 285 us per
force call) led to `SEGMENT_ROWS`, and the energy-head and graph-sum glue to `HEAD_SLICE` and `GRAPH_SUM`; see
`plans/ideas-20261105-round2.md` and `docs/optimisation-ledger.md`.
