# Limiters: NVIDIA H200

ncu kernels: 98 (98 aligned with the profiler's sequence); kernel time 3257.6 µs per force call (ncu, running clocks, warm caches).

## Step stages (main pass)

```
stage,calls,ms_per_call,ms_per_step,host_ms_per_step
B: forces + second half kick,300,3.3296,3.3296,0.0292
Neighbour-list rebuild (GPU),39,4.3061,0.5598,0.5619
"ACP: Langevin half kick, positions, neighbour check, edge compaction",300,0.0537,0.0537,0.0284
Noise upload (every step a slice; a new batch every noise_batch),300,0.0164,0.0164,0.0573
GPU idle between stages,,,-0.4655,
```

## Kernel time by limiter

| Limiter | µs | Share |
| --- | ---: | ---: |
| issue | 2034.1 | 62.4% |
| latency | 569.8 | 17.5% |
| l1 | 226.2 | 6.9% |
| underfilled | 196.9 | 6.0% |
| l2 | 117.0 | 3.6% |
| tiny | 113.5 | 3.5% |

## By category

| Category | µs | Share | Limiters (µs) |
| --- | ---: | ---: | --- |
| Edge elementwise (geometry, radial basis, gathers, scatters) | 2197.6 | 67.5% | issue 1891, l1 211, underfilled 46, latency:mio_throttle 31, l2 16, tiny 4 |
| Other | 587.2 | 18.0% | latency:math_pipe_throttle 385, underfilled 93, latency:wait 73, latency:long_scoreboard 19, l1 15, tiny 2 |
| Atom-level elementwise (gates, products, residuals, readouts) | 203.5 | 6.2% | l2 87, tiny 66, underfilled 45, issue 6 |
| Atom-level cuEq tensor products (symmetric contraction) | 199.6 | 6.1% | issue 137, latency:long_scoreboard 62 |
| Copies and fills | 36.1 | 1.1% | tiny 21, l2 15 |
| Atom-level linear layers (GEMM) | 33.6 | 1.0% | tiny 21, underfilled 13 |

## 25 largest kernels

| # | Kernel | Op | µs | Limiter | SM fill | DRAM | L2 | L1 | Issue | Pipes | Stalls | Warps act/theo | Regs | Occ. limits r/s/w/b | Sect/ld | Smem confl | Thr/inst |
| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| 73 | oeq_projected_backward_wprime | fastmace::streaming_conv_wprime_backward | 1088.7 | issue | 98 | 5 | 66 | 73 | 77 | fmalite=60 fmalite_realtime=60 fma=52 | long_scoreboard=2.23 wait=0.89 dispatch_stall=0.82 | 43/44 | 72 | 7/7/16/32 | 11.7 | 0.02 | 31.6 |
| 43 | spline_forward0 | fastmace::spline_message_blocks | 628.6 | issue | 99 | 5 | 51 | 34 | 82 | fmalite=56 fmalite_realtime=56 fma=51 | wait=0.79 math_pipe_throttle=0.74 barrier=0.67 | 36/38 | 80 | 6/17/16/32 | 6.8 | 0.00 | 32.0 |
| 88 | kernel | fastmace::tabled_outer_sum_backward | 211.0 | l1 | 98 | 13 | 26 | 71 | 62 | lsu=58 fmalite=37 fmalite_realtime=37 | long_scoreboard=3.60 short_scoreboard=1.40 wait=1.21 | 35/38 | 80 | 24/56/64/32 | 3.1 | 0.00 | 32.0 |
| 28 | kernel | fastmace::tabled_outer_sum_forward | 111.8 | issue | 94 | 25 | 49 | 57 | 63 | lsu=44 fmaheavy=38 fma=35 | long_scoreboard=6.32 wait=1.52 dispatch_stall=0.53 | 44/50 | 64 | 32/32/64/32 | 2.8 | 0.00 | 32.0 |
| 81 | segmented_polynomial_bwd_algo1_1 | cuequivariance::uniform_1d | 97.2 | issue | 91 | 18 | 24 | 7 | 70 | fmalite=67 fmalite_realtime=67 fma=66 | no_instruction=1.28 dispatch_stall=0.90 long_scoreboard=0.30 | 27/31 | 96 | 20/32/64/32 | 4.0 | 0.00 | 32.0 |
| 47 | fm_tf32x3_remap_div_rr_64x64 | fastmace::tf32x3_linear_div | 51.7 | latency:math_pipe_throttle | 88 | 30 | 39 | 27 | 58 | tensor=47 tensor_op_hmma=47 shared=47 | math_pipe_throttle=0.92 wait=0.89 dispatch_stall=0.40 | 17/19 | 158 | 3/4/16/32 | 14.3 | 0.01 | 32.0 |
| 69 | fm_tf32x3_rr_64x64 | ## Call CompiledFxGraph fqhdtwk74whasb7c | 49.9 | latency:math_pipe_throttle | 93 | 18 | 34 | 35 | 60 | tensor=49 tensor_op_hmma=49 shared=49 | math_pipe_throttle=0.96 wait=0.72 dispatch_stall=0.44 | 18/19 | 158 | 3/4/16/32 | 16.0 | 0.03 | 32.0 |
| 46 | fm_tf32x3_remap_div_rr_64x64 | fastmace::tf32x3_linear_div | 49.6 | latency:math_pipe_throttle | 93 | 35 | 42 | 28 | 57 | tensor=50 tensor_op_hmma=50 shared=50 | math_pipe_throttle=1.05 wait=0.85 dispatch_stall=0.40 | 18/19 | 158 | 3/4/16/32 | 14.8 | 0.01 | 32.0 |
| 71 | fm_tf32x3_rr_64x64 | ## Call CompiledFxGraph fqhdtwk74whasb7c | 47.6 | latency:math_pipe_throttle | 93 | 23 | 38 | 35 | 59 | tensor=49 tensor_op_hmma=49 shared=49 | math_pipe_throttle=0.94 wait=0.71 dispatch_stall=0.42 | 18/19 | 158 | 3/4/16/32 | 16.0 | 0.02 | 32.0 |
| 87 | fm_tf32x3_species_rr_64x64 | fastmace::species_mm_rows_grad | 42.0 | latency:wait | 92 | 22 | 35 | 32 | 58 | tensor=42 tensor_op_hmma=42 shared=42 | wait=0.88 math_pipe_throttle=0.87 long_scoreboard=0.48 | 18/19 | 156 | 3/4/16/32 | 9.1 | 0.02 | 32.0 |
| 33 | fm_tf32x3_species_rr_64x64 | fastmace::species_mm_rows | 40.6 | latency:math_pipe_throttle | 87 | 27 | 38 | 29 | 58 | tensor=46 tensor_op_hmma=46 shared=46 | math_pipe_throttle=0.96 wait=0.86 long_scoreboard=0.41 | 18/19 | 156 | 3/4/16/32 | 9.7 | 0.02 | 32.0 |
| 29 | triton_poi_fused_scatter_add_select_unsqueeze_zero | triton_poi_fused_scatter_add_select_unsq | 40.2 | underfilled | 69 | 4 | 24 | 14 | 2 | lsu=2 alu=2 alu_realtime=2 | long_scoreboard=57.15 short_scoreboard=40.11 drain=7.62 | 16/44 | 72 | 7/14/16/32 | 20.0 | 0.00 | 32.0 |
| 35 | segmented_polynomial_algo1_2 | cuequivariance::uniform_1d | 40.2 | issue | 94 | 28 | 33 | 10 | 76 | fmalite=73 fmalite_realtime=73 fma=72 | dispatch_stall=1.59 long_scoreboard=1.00 wait=0.43 | 46/56 | 56 | 18/32/32/32 | 4.0 | 0.00 | 32.0 |
| 61 | segmented_polynomial_bwd_algo2_2 | cuequivariance::uniform_1d | 36.6 | latency:long_scoreboard | 93 | 48 | 57 | 17 | 55 | fmaheavy=45 fma=42 fmaheavy_realtime=40 | long_scoreboard=3.12 wait=0.45 dispatch_stall=0.39 | 23/25 | 128 | 4/32/16/32 | 3.9 | 0.00 | 32.0 |
| 45 | fm_tf32x3_remap_div_rr_64x64 | fastmace::tf32x3_linear_div | 34.7 | latency:math_pipe_throttle | 84 | 24 | 34 | 24 | 55 | tensor=48 tensor_op_hmma=48 shared=48 | math_pipe_throttle=1.03 wait=0.84 barrier=0.38 | 17/19 | 158 | 3/4/16/32 | 14.8 | 0.01 | 32.0 |
| 67 | fm_tf32x3_rr_64x64 | ## Call CompiledFxGraph fqhdtwk74whasb7c | 32.3 | latency:math_pipe_throttle | 90 | 10 | 31 | 33 | 58 | tensor=48 tensor_op_hmma=48 shared=48 | math_pipe_throttle=0.99 wait=0.71 dispatch_stall=0.41 | 18/19 | 158 | 3/4/16/32 | 16.0 | 0.02 | 32.0 |
| 95 | triton_poi_fused_add_div_eq_index_put_masked_fill_ | triton_poi_fused_add_div_eq_index_put_ma | 30.9 | latency:mio_throttle | 85 | 12 | 37 | 23 | 15 | adu=11 alu=10 alu_realtime=10 | mio_throttle=36.04 long_scoreboard=15.54 short_scoreboard=11.03 | 85/100 | 32 | 8/20/8/32 | 5.8 | 0.02 | 32.0 |
| 86 | fm_tf32x3_species_rr_64x64 | fastmace::species_mm_rows_grad | 30.6 | latency:wait | 91 | 20 | 34 | 31 | 58 | tensor=42 tensor_op_hmma=42 shared=42 | wait=0.88 math_pipe_throttle=0.84 long_scoreboard=0.45 | 18/19 | 156 | 3/4/16/32 | 9.1 | 0.02 | 32.0 |
| 82 | triton_red_fused_add_div_mul_neg_species_mm_rows_g | triton_red_fused_add_div_mul_neg_species | 29.6 | l2 | 91 | 74 | 85 | 28 | 28 | lsu=17 fmalite=15 fmalite_realtime=15 | long_scoreboard=29.31 barrier=8.74 short_scoreboard=5.67 | 92/100 | 32 | 4/14/4/32 | 11.0 | 0.02 | 32.0 |
| 32 | fm_tf32x3_species_rr_64x64 | fastmace::species_mm_rows | 28.9 | latency:math_pipe_throttle | 91 | 25 | 37 | 30 | 57 | tensor=45 tensor_op_hmma=45 shared=45 | math_pipe_throttle=0.97 wait=0.85 long_scoreboard=0.42 | 17/19 | 156 | 3/4/16/32 | 9.7 | 0.02 | 32.0 |
| 31 | fm_tf32x3_species_rr_64x64 | fastmace::species_mm_rows | 23.5 | underfilled | 69 | 18 | 28 | 21 | 54 | tensor=43 tensor_op_hmma=43 shared=43 | math_pipe_throttle=0.97 wait=0.83 long_scoreboard=0.54 | 17/19 | 156 | 3/4/16/32 | 9.7 | 0.01 | 32.0 |
| 85 | fm_tf32x3_species_rr_64x64 | fastmace::species_mm_rows_grad | 21.8 | latency:math_pipe_throttle | 79 | 16 | 30 | 26 | 55 | tensor=40 tensor_op_hmma=40 shared=40 | math_pipe_throttle=0.90 wait=0.86 long_scoreboard=0.51 | 17/19 | 156 | 3/4/16/32 | 9.1 | 0.01 | 32.0 |
| 62 | triton_red_fused_add_div_mul_neg_sum_2 | triton_red_fused_add_div_mul_neg_sum_2 | 20.4 | l2 | 89 | 83 | 85 | 32 | 35 | lsu=24 fmalite=18 fmalite_realtime=18 | long_scoreboard=28.83 barrier=6.52 short_scoreboard=2.08 | 94/100 | 32 | 4/14/4/32 | 11.0 | 0.01 | 32.0 |
| 34 | triton_poi_fused_add_div_view_21 | triton_poi_fused_add_div_view_21 | 18.1 | l2 | 84 | 56 | 84 | 31 | 34 | alu=19 alu_realtime=19 xu=16 | long_scoreboard=24.69 drain=3.25 lg_throttle=2.02 | 82/100 | 24 | 21/32/16/32 | 4.0 | 0.00 | 32.0 |
| 48 | segmented_polynomial_algo1_4 | cuequivariance::uniform_1d | 17.7 | latency:long_scoreboard | 86 | 53 | 48 | 17 | 56 | fmalite=47 fmalite_realtime=47 fma=45 | long_scoreboard=8.85 dispatch_stall=1.33 wait=1.21 | 78/75 | 40 | 12/32/16/32 | 3.9 | 0.00 | 32.0 |

## Main-pass summary keys

```
{
 "tool": "step_breakdown",
 "created_utc": "2026-11-09T07:50:01.176658+00:00",
 "host": "modal",
 "python": "/usr/local/bin/python",
 "fastmace_commit": "c520e07403e5d0101556729b6b28a26cdf8f57e0",
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
 "gpu": "NVIDIA H200",
 "cuda": "13.0",
 "oeq_runtime": "PatchRecord(name='oeq_runtime', value='/__modal/volumes/vo-4AvnRSqwjTBUKJJdNFtTo4/cache/H200/oeq/cuda-runtime', targets=('env:CMAKE_CUDA_COMPILER', 'env:CPATH', 'env:CUDACXX', 'env:CUDA_HOME', 'env:CUDA_PATH', 'env:MAX_JOBS', 'env:OEQ_JIT_EXTENSION', 'env:OEQ_NVCC_ATTEMPTS', 'env:PATH', 'env:PYTORCH_NVCC', 'env:TORCH_EXTENSIONS_DIR'), detail='OEq host-extension build environment; preloaded libcudart.so.13, libnvrtc.so.13, libcublasLt.so.13, libcublas.so.13; host extension: reused')",
 "other_gpu_processes": [],
 "compute_capability": "9.0",
 "sms": 132,
 "max_sm_clock_mhz": 1980.0,
 "fp32_peak_tflops": 66.90816,
 "model": "omat0-medium",
 "checkpoint": "/__modal/volumes/vo-4AvnRSqwjTBUKJJdNFtTo4/models/mace-omat-0-medium.model",
 "checkpoint_sha256": "d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a",
 "backend": "hybrid"
}
```
