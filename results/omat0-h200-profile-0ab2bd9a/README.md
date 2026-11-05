# OMAT-0 medium H200 profile at 0ab2bd9a (5 November 2026)

Modal NVIDIA H200 (sm_90, 132 SMs, 141 GB HBM3e, driver 610.57.04), FastMACE main `0ab2bd9a` (round 6 merged),
torch 2.14.1+cu130, float32 hybrid; `tools/modal/omat0_a100_profile.sh` plus `tools/bench/vram_compare.py fastmace`
in the same job (Modal app ap-NKnKDzmLzcfTLzhApxt1tg, profile `second`, wall 597 s, metered $0.65).

Benchmark_loup SrTiO3 10x10x10 (5,000 atoms), `tools/bench/loup_md.py`, three fresh processes each:

| Ensemble | ms per step | µs per atom and step |
| --- | --- | --- |
| NVE | 13.32 / 13.33 / 13.32 | 2.664 / 2.666 / 2.665 |
| Langevin 300 K | 13.12 / 13.12 / 13.13 | 2.625 / 2.624 / 2.625 |

For comparison: A100-SXM4-80GB 10.91 ms (2.18 µs) at round 6, A100 80GB PCIe about 12.4 ms (2.48 µs) before the
interaction-1 fold, RTX 5060 Ti 24.4 ms; Symmetrix-XL 5.00 µs (A100 PCIe, Benchmark_loup) and 5.80 µs (paper, SXM4).

**The H200 runs the generic plan.** Every per-GPU default (decision 66) is keyed by exact device names, and none
lists the H200, so it gets none of the A100 or Ada rewrites: no radial tables (the folded last radial layer is still
a GEMM, 2.0 ms forward and 1.6 ms input gradient), no generic fused forward or w′ backward (interaction 2's streaming
convolution backward 3.05 ms), no TF32x3 GEMMs, no interaction-1 fold. Force-call kernels 13.0 ms (`breakdown/`,
198 kernels per call). Memory follows from the same plan: torch peak 4,867 MiB allocated and 5,160 MiB reserved,
5,983 MiB of device memory over the idle baseline (`vram.json`; A100 profile: 1,127 / 1,180 MiB), because the
per-edge radial weights are materialised.

Next step if the H200 matters: decision-66 MD A/Bs on the H200 of the A100 plan (radial tables, generic fused
forward and w′ backward, VEC_A/VEC_SH, TC_WFRAG, TF32x3, fold), both build orders.
