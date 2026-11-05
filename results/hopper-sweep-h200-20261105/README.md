# H200 under the A100 plan, and the Hopper sweep (5 November 2026)

Modal NVIDIA H200 (sm_90, 132 SMs, driver 610.57.04), profile `second`, OMAT-0 medium, float32 hybrid, Benchmark_loup
SrTiO3 10x10x10 (5,000 atoms). The H200 is in no decision-66 GPU set, so on its own it runs the generic plan
(`results/omat0-h200-profile-0ab2bd9a/`: 13.12 ms per step, 2.62 µs per atom and step). Here it runs under the
A100-SXM4 plan through `tools/bench/gpu_alias` (`FASTMACE_GPU_ALIAS="NVIDIA H200=NVIDIA A100-SXM4-80GB"`).

## `alias-name-only/` (c03f791e): the alias did not take

The first alias patched only `torch.cuda.get_device_name`; the planner's `detect_hardware` reads
`get_device_properties().name`, so the plan stayed generic (no fused forward, no radial tables; the kernel switches
read by name did change): Langevin 12.79 ms (2.56 µs), NVE 13.00 ms, torch peak 4.87 GB. Gate passes. Kept as a
record; b03fc94e aliases both.

## `sweep-1/` (b03fc94e, `tools/modal/hopper_sweep.sh`): the A100 plan with the sm_90 CUTLASS cubin

- **CUTLASS 3xTF32 on sm_90** (`fastmace/kernels/cutlass/tf32x3_sm90.cubin`, built from the sm_80 source,
  3cc1ebd2): `tests/gpu/test_cutlass_tf32x3_tiles_cuda.py` and `test_tf32x3_gemm_cuda.py` 12 passed.
- **Benchmark_loup Langevin: 5.658 / 5.661 ms per step, 1.132 µs per atom and step** (A100-SXM4 10.91 ms, 2.18 µs;
  the H200's generic plan 13.12 ms). Torch peak 1,177 MiB. Plan: block_linear, chunked_conv_backward,
  fold_product_linear, fused_folded_backward, fused_folded_forward, grouped_linear, merged_residual, radial_tables,
  reassociate_first, segmented_first_sum, single_graph_strain, species_skip, streaming_conv, tf32x3_gemms,
  wprime_backward.
- **Force/stress gate** (NaCl, NbCl5, SrTiO3 3x3x3; ratio to MACE's float32 error against float64): forces 0.69 /
  0.98 / 1.02, stress 1.02 / **2.05** / 0.88, energy 2.04 / 4.32 / 9.50 (energies pass the 0.01 eV absolute
  floor: 3.2e-4, 9.8e-4, 3.7e-3 eV). NbCl5's stress (1.30e-7) misses the 2x ratio, the same class as the A100's
  accepted NbCl5 stress exception (ratio 2.34 there).
- **MD-step A/Bs on SrTiO3** (`ab_rewrites.py`, 6 blocks of 256 steps, both orders; side a the aliased plan):
  - Triton 3xTF32 instead of the sm_90 CUTLASS GEMMs (`CUTLASS_CAPABILITIES=()`): b/a 1.133 / 1.129, so the
    sm_90 cubin is worth 12-13 %.
  - GEMM tile 32x64: 1.009 / 1.006 (64x64 stays).
- **Kernel lab, fused forward** (A100 default `base` 1,786 µs): every variant the A100 rejected loses on the H200 too:
  tc-warps8 1.076, tc-warps10 1.081, tc-prefetch 1.193, tc-prefetch-x 1.328, tc-prefetch-w10 1.265, tc-presplit
  1.297 (and wrong: the lab rule mismatched the re-render), tc-hybrid16 1.207, tc-fp16 1.701, simt 1.340.
- **Kernel lab, w′ backward** (A100 default `base`, 96 registers, 1,602 µs): maxreg-none 1.439, maxreg80 1.252,
  maxreg72 1.083 (spills), redux 1.129, **butterfly 0.975** (bitwise in dsh and dr), F-split2 1.250, pair 1.126
  (falsifier accuracy). Butterfly is the one candidate for an MD A/B on the H200.

Cost: $0.65 (generic profile) + $0.99 (alias-name-only) + about $2.1 (sweep-1), profile `second`.

Not a default: per decision 66 an H200 plan needs MD A/Bs on the H200 in both build orders, keyed by its name.
