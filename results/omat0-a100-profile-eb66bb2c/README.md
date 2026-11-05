# OMAT-0 medium A100 profile at eb66bb2c (5 November 2026)

Modal A100-SXM4-80GB, FastMACE main `eb66bb2c` (round 5 merged: HEAD_SLICE, SEGMENT_ROWS, GRAPH_SUM on the A100),
torch 2.14.1+cu130, float32 hybrid. Benchmark_loup SrTiO3 10x10x10 (5,000 atoms): NVE 11.43 / 11.76 / 11.65 ms and
Langevin 10.90 / 11.39 / 11.85 ms per step (`loup-*.log`; 2a016fd4: 11.7 ms). Force-call kernels 11.04 ms
(`breakdown/`). The neighbour-list rebuild costs 11.35 ms per call (24 in 200 steps, 1.36 ms per step, mostly host
time; 7.69 ms per call at 2a016fd4). This profile is the basis of `plans/research-prompt-20261105-skip-work.md`.
