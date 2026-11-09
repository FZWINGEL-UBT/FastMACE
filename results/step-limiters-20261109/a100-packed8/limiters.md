# Limiters: NVIDIA A100 80GB PCIe

ncu kernels: 97 (97 aligned with the profiler's sequence); kernel time 6511.7 µs per force call (ncu, running clocks, warm caches).

## Step stages (main pass)

```
stage,calls,ms_per_call,ms_per_step,host_ms_per_step
B: forces + second half kick,300,7.1270,7.1270,0.0526
Neighbour-list rebuild (GPU),39,8.3687,1.0879,1.0272
"ACP: Langevin half kick, positions, neighbour check, edge compaction",300,0.0844,0.0844,0.0427
Noise upload (every step a slice; a new batch every noise_batch),300,0.0189,0.0189,0.1044
GPU idle between stages,,,-1.0234,
```

## Kernel time by limiter

| Limiter | µs | Share |
| --- | ---: | ---: |
| pipe:fma | 3875.6 | 59.5% |
| latency | 1464.7 | 22.5% |
| l1 | 444.0 | 6.8% |
| dram | 334.9 | 5.1% |
| underfilled | 210.6 | 3.2% |
| tiny | 86.2 | 1.3% |
| issue | 67.2 | 1.0% |
| l2 | 28.4 | 0.4% |

## By category

| Category | µs | Share | Limiters (µs) |
| --- | ---: | ---: | --- |
| Edge elementwise (geometry, radial basis, gathers, scatters) | 4610.1 | 70.8% | pipe:fma 3511, latency:long_scoreboard 475, l1 444, issue 67, latency:mio_throttle 53, dram 36, underfilled 24 |
| Other | 1050.6 | 16.1% | latency:math_pipe_throttle 500, latency:wait 437, underfilled 67, dram 29, l2 16, tiny 3 |
| Atom-level cuEq tensor products (symmetric contraction) | 414.6 | 6.4% | pipe:fma 365, dram 37, l2 13 |
| Atom-level elementwise (gates, products, residuals, readouts) | 328.0 | 5.0% | dram 200, tiny 66, underfilled 62 |
| Copies and fills | 81.0 | 1.2% | underfilled 34, dram 33, tiny 14 |
| Atom-level linear layers (GEMM) | 27.4 | 0.4% | underfilled 23, tiny 4 |

## 25 largest kernels

| # | Kernel | Op | µs | Limiter | SM fill | DRAM | L2 | L1 | Issue | Pipes | Stalls | Warps act/theo | Regs | Occ. limits r/s/w/b | Sect/ld | Smem confl | Thr/inst |
| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| 71 | oeq_projected_backward_wprime | fastmace::streaming_conv_wprime_backward | 2255.9 | pipe:fma | 99 | 7 | 53 | 60 | 61 | fma=85 lsu=40 alu=9 | long_scoreboard=2.25 math_pipe_throttle=1.92 wait=1.07 | 37/38 | 80 | 6/6/16/32 | 12.6 | 0.02 | 31.6 |
| 41 | spline_forward0 | fastmace::spline_message_blocks | 1254.9 | pipe:fma | 99 | 7 | 42 | 33 | 71 | fma=84 alu=43 alu_realtime=43 | math_pipe_throttle=2.11 wait=1.13 long_scoreboard=0.89 | 36/38 | 80 | 6/17/16/32 | 6.8 | 0.00 | 32.0 |
| 87 | kernel | fastmace::tabled_outer_sum_backward | 444.0 | l1 | 99 | 15 | 31 | 64 | 55 | fma=51 lsu=48 alu=18 | long_scoreboard=4.91 wait=1.37 short_scoreboard=1.25 | 35/38 | 80 | 24/28/64/32 | 3.9 | 0.01 | 32.0 |
| 27 | kernel | fastmace::tabled_outer_sum_forward | 399.7 | latency:long_scoreboard | 97 | 19 | 22 | 30 | 28 | fma=30 lsu=17 alu=8 | long_scoreboard=10.01 wait=1.47 math_pipe_throttle=0.28 | 23/25 | 128 | 16/16/64/32 | 3.0 | 0.00 | 32.0 |
| 80 | segmented_polynomial_bwd_algo1_4 | cuequivariance::uniform_1d | 203.3 | pipe:fma | 98 | 24 | 19 | 5 | 50 | fma=96 lsu=4 alu=1 | math_pipe_throttle=3.17 wait=1.05 long_scoreboard=0.36 | 30/31 | 96 | 5/16/16/32 | 3.9 | 0.00 | 32.0 |
| 35 | segmented_polynomial_algo1_2 | cuequivariance::uniform_1d | 103.5 | pipe:fma | 96 | 28 | 20 | 6 | 49 | fma=92 lsu=7 lsu_mem_lg_cmd_read=2 | math_pipe_throttle=5.68 wait=1.12 long_scoreboard=1.02 | 46/56 | 56 | 18/32/32/32 | 4.0 | 0.00 | 32.0 |
| 44 | fm_tf32x3_remap_div_rr_64x64 | fastmace::tf32x3_linear_div | 95.1 | latency:math_pipe_throttle | 89 | 52 | 39 | 23 | 48 | tensor=57 tensor_op_hmma=57 shared=57 | math_pipe_throttle=1.49 wait=1.18 long_scoreboard=0.45 | 17/19 | 134 | 3/3/16/32 | 14.8 | 0.01 | 32.0 |
| 67 | fm_tf32x3_rr_64x64 | ## Call CompiledFxGraph faooswkjhpky3bpt | 90.6 | latency:math_pipe_throttle | 96 | 32 | 34 | 31 | 50 | tensor=55 tensor_op_hmma=55 shared=55 | math_pipe_throttle=1.38 wait=1.03 long_scoreboard=0.44 | 18/19 | 134 | 3/3/16/32 | 16.0 | 0.03 | 32.0 |
| 45 | fm_tf32x3_remap_div_rr_64x64 | fastmace::tf32x3_linear_div | 90.2 | latency:math_pipe_throttle | 91 | 56 | 40 | 24 | 50 | tensor=55 tensor_op_hmma=55 shared=55 | math_pipe_throttle=1.28 wait=1.23 long_scoreboard=0.58 | 18/19 | 134 | 3/3/16/32 | 14.3 | 0.01 | 32.0 |
| 69 | fm_tf32x3_rr_64x64 | ## Call CompiledFxGraph faooswkjhpky3bpt | 86.2 | latency:math_pipe_throttle | 95 | 41 | 39 | 30 | 50 | tensor=55 tensor_op_hmma=55 shared=55 | math_pipe_throttle=1.29 wait=1.03 long_scoreboard=0.58 | 18/19 | 134 | 3/3/16/32 | 16.0 | 0.03 | 32.0 |
| 86 | fm_tf32x3_species_rr_64x64 | fastmace::species_mm_rows_grad | 74.0 | latency:wait | 93 | 44 | 38 | 29 | 51 | tensor=49 tensor_op_hmma=49 shared=49 | wait=1.31 math_pipe_throttle=0.97 long_scoreboard=0.74 | 18/19 | 134 | 3/3/16/32 | 9.1 | 0.02 | 32.0 |
| 33 | fm_tf32x3_species_rr_64x64 | fastmace::species_mm_rows | 71.6 | latency:wait | 92 | 52 | 39 | 26 | 49 | tensor=52 tensor_op_hmma=52 shared=52 | wait=1.31 math_pipe_throttle=1.16 long_scoreboard=0.67 | 18/19 | 134 | 3/3/16/32 | 9.7 | 0.01 | 32.0 |
| 81 | triton_red_fused_add_div_mul_neg_species_mm_rows_g | triton_red_fused_add_div_mul_neg_species | 70.8 | dram | 95 | 85 | 64 | 20 | 19 | fma=18 lsu=12 adu=10 | long_scoreboard=35.41 mio_throttle=14.13 barrier=9.31 | 93/100 | 32 | 4/14/4/32 | 11.0 | 0.01 | 32.0 |
| 20 | triton_poi_fused__to_copy_add_arange_div_exp_full_ | triton_poi_fused__to_copy_add_arange_div | 63.2 | latency:long_scoreboard | 80 | 18 | 25 | 16 | 30 | fma=26 alu=21 alu_realtime=21 | long_scoreboard=3.11 wait=1.51 drain=1.50 | 23/25 | 89 | 2/6/8/32 | 10.5 | 0.04 | 32.0 |
| 43 | fm_tf32x3_remap_div_rr_64x64 | fastmace::tf32x3_linear_div | 62.7 | latency:math_pipe_throttle | 85 | 36 | 29 | 21 | 46 | tensor=55 tensor_op_hmma=55 shared=55 | math_pipe_throttle=1.46 wait=1.16 long_scoreboard=0.46 | 16/19 | 134 | 3/3/16/32 | 14.8 | 0.01 | 32.0 |
| 65 | fm_tf32x3_rr_64x64 | ## Call CompiledFxGraph faooswkjhpky3bpt | 58.4 | latency:math_pipe_throttle | 92 | 22 | 27 | 29 | 49 | tensor=54 tensor_op_hmma=54 shared=54 | math_pipe_throttle=1.44 wait=1.03 barrier=0.39 | 18/19 | 134 | 3/3/16/32 | 16.0 | 0.03 | 32.0 |
| 59 | segmented_polynomial_bwd_algo1_1 | cuequivariance::uniform_1d | 58.0 | pipe:fma | 88 | 72 | 57 | 18 | 44 | fma=77 lsu=12 alu=2 | long_scoreboard=5.10 math_pipe_throttle=2.56 wait=1.13 | 37/50 | 64 | 32/32/64/32 | 4.0 | 0.00 | 32.0 |
| 32 | fm_tf32x3_species_rr_64x64 | fastmace::species_mm_rows | 54.2 | latency:wait | 89 | 39 | 32 | 25 | 48 | tensor=50 tensor_op_hmma=50 shared=50 | wait=1.30 math_pipe_throttle=1.22 long_scoreboard=0.58 | 17/19 | 134 | 3/3/16/32 | 9.7 | 0.01 | 32.0 |
| 85 | fm_tf32x3_species_rr_64x64 | fastmace::species_mm_rows_grad | 53.4 | latency:wait | 95 | 33 | 33 | 29 | 51 | tensor=48 tensor_op_hmma=48 shared=48 | wait=1.31 math_pipe_throttle=1.07 long_scoreboard=0.54 | 18/19 | 134 | 3/3/16/32 | 9.1 | 0.02 | 32.0 |
| 94 | triton_poi_fused_add_div_eq_index_put_masked_fill_ | triton_poi_fused_add_div_eq_index_put_ma | 53.2 | latency:mio_throttle | 84 | 18 | 31 | 22 | 15 | adu=14 fma=9 alu=9 | mio_throttle=35.13 barrier=16.03 lg_throttle=10.50 | 88/100 | 32 | 8/20/8/32 | 5.7 | 0.00 | 32.0 |
| 60 | triton_red_fused_add_div_mul_neg_sum_2 | triton_red_fused_add_div_mul_neg_sum_2 | 47.2 | dram | 91 | 90 | 66 | 25 | 26 | fma=20 lsu=17 adu=17 | long_scoreboard=39.01 short_scoreboard=5.59 mio_throttle=5.08 | 96/100 | 31 | 4/14/4/32 | 11.0 | 0.00 | 32.0 |
| 34 | triton_poi_fused_add_div_view_20 | triton_poi_fused_add_div_view_20 | 45.0 | dram | 91 | 81 | 60 | 22 | 24 | adu=22 fma=16 alu=15 | long_scoreboard=31.75 lg_throttle=7.93 mio_throttle=6.78 | 88/100 | 25 | 16/32/16/32 | 4.0 | 0.00 | 32.0 |
| 46 | segmented_polynomial_algo1_4 | cuequivariance::uniform_1d | 37.0 | dram | 90 | 66 | 48 | 12 | 35 | fma=58 lsu=13 alu=5 | long_scoreboard=13.86 math_pipe_throttle=4.88 lg_throttle=2.32 | 66/75 | 40 | 12/32/16/32 | 3.9 | 0.00 | 32.0 |
| 92 | triton_poi_fused_add_edge_sh3_backward_14 | triton_poi_fused_add_edge_sh3_backward_1 | 36.3 | dram | 89 | 81 | 66 | 17 | 5 | lsu=3 fma=2 alu=2 | long_scoreboard=174.96 lg_throttle=30.51 drain=26.18 | 83/100 | 24 | 21/32/16/32 | 16.0 | 0.00 | 32.0 |
| 31 | fm_tf32x3_species_rr_64x64 | fastmace::species_mm_rows | 36.2 | latency:wait | 84 | 31 | 26 | 22 | 46 | tensor=48 tensor_op_hmma=48 shared=48 | wait=1.28 math_pipe_throttle=1.23 long_scoreboard=0.60 | 16/19 | 134 | 3/3/16/32 | 9.7 | 0.01 | 32.0 |

## Main-pass summary keys

```
{
 "tool": "step_breakdown",
 "created_utc": "2026-11-09T08:38:48.835292+00:00",
 "host": "modal",
 "python": "/usr/local/bin/python",
 "fastmace_commit": "0df2241750fb0d813b6f79f249590ee341b06eac",
 "packages": {
  "torch": "2.14.1+cu130",
  "triton": "3.8.0",
  "mace-torch": "0.3.16",
  "cuequivariance": "0.12.0",
  "cuequivariance-torch": "0.12.0",
  "cuequivariance-ops-torch-cu13": "0.12.0",
  "openequivariance": "0.7.0",
  "e3nn": "0.4.4",
  "ase": "3.27.0",
  "numpy": "2.2.6"
 },
 "tf32": false,
 "constants": {},
 "gpu": "NVIDIA A100 80GB PCIe",
 "cuda": "13.0",
 "oeq_runtime": "PatchRecord(name='oeq_runtime', value='/__modal/volumes/vo-xuPLzFN4jQ2xJCvj0J69aS/cache/A100-80GB/oeq/cuda-runtime', targets=('env:CMAKE_CUDA_COMPILER', 'env:CPATH', 'env:CUDACXX', 'env:CUDA_HOME', 'env:CUDA_PATH', 'env:MAX_JOBS', 'env:OEQ_JIT_EXTENSION', 'env:OEQ_NVCC_ATTEMPTS', 'env:PATH', 'env:PYTORCH_NVCC', 'env:TORCH_EXTENSIONS_DIR'), detail='OEq host-extension build environment; preloaded libcudart.so.13, libnvrtc.so.13, libcublasLt.so.13, libcublas.so.13; host extension: reused')",
 "other_gpu_processes": [],
 "compute_capability": "8.0",
 "sms": 108,
 "max_sm_clock_mhz": 1410.0,
 "fp32_peak_tflops": 19.49184,
 "model": "omat0-medium",
 "checkpoint": "/__modal/volumes/vo-xuPLzFN4jQ2xJCvj0J69aS/models/mace-omat-0-medium.model",
 "checkpoint_sha256": "d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a",
 "backend": "hybrid"
}
```
