# Limiters: NVIDIA A100 80GB PCIe

ncu kernels: 97 (97 aligned with the profiler's sequence); kernel time 6807.6 µs per force call (ncu, running clocks, warm caches).

## Step stages (main pass)

```
stage,calls,ms_per_call,ms_per_step,host_ms_per_step
B: forces + second half kick,300,7.3588,7.3588,0.0329
Neighbour-list rebuild (GPU),39,8.6312,1.1221,1.0556
"ACP: Langevin half kick, positions, neighbour check, edge compaction",300,0.0821,0.0821,0.0293
Noise upload (every step a slice; a new batch every noise_batch),300,0.0193,0.0193,0.0637
GPU idle between stages,,,-1.0628,
```

## Kernel time by limiter

| Limiter | µs | Share |
| --- | ---: | ---: |
| pipe:fma | 2624.3 | 38.5% |
| l2 | 1568.0 | 23.0% |
| latency | 1453.3 | 21.3% |
| l1 | 447.8 | 6.6% |
| dram | 337.1 | 5.0% |
| underfilled | 227.7 | 3.3% |
| tiny | 82.0 | 1.2% |
| issue | 67.5 | 1.0% |

## By category

| Category | µs | Share | Limiters (µs) |
| --- | ---: | ---: | --- |
| Edge elementwise (geometry, radial basis, gathers, scatters) | 4899.4 | 72.0% | pipe:fma 2259, l2 1539, latency:long_scoreboard 458, l1 448, issue 67, latency:mio_throttle 52, dram 41, underfilled 34 |
| Other | 1057.8 | 15.5% | latency:math_pipe_throttle 502, latency:wait 441, underfilled 66, dram 30, l2 16, tiny 3 |
| Atom-level cuEq tensor products (symmetric contraction) | 415.8 | 6.1% | pipe:fma 365, dram 38, l2 13 |
| Atom-level elementwise (gates, products, residuals, readouts) | 326.7 | 4.8% | dram 197, underfilled 69, tiny 61 |
| Copies and fills | 80.0 | 1.2% | underfilled 35, dram 31, tiny 14 |
| Atom-level linear layers (GEMM) | 27.9 | 0.4% | underfilled 24, tiny 4 |

## 25 largest kernels

| # | Kernel | Op | µs | Limiter | SM fill | DRAM | L2 | L1 | Issue | Pipes | Stalls | Warps act/theo | Regs | Occ. limits r/s/w/b | Sect/ld | Smem confl | Thr/inst |
| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| 71 | oeq_projected_backward_wprime | fastmace::streaming_conv_wprime_backward | 2259.4 | pipe:fma | 99 | 7 | 53 | 60 | 61 | fma=85 lsu=40 alu=9 | long_scoreboard=2.27 math_pipe_throttle=1.92 wait=1.07 | 37/38 | 80 | 6/6/16/32 | 12.6 | 0.02 | 31.6 |
| 41 | spline_forward0 | fastmace::spline_message_blocks | 1539.4 | l2 | 99 | 6 | 70 | 50 | 38 | fma=58 lsu=18 alu=6 | long_scoreboard=10.36 wait=1.23 math_pipe_throttle=0.86 | 37/38 | 80 | 6/17/16/32 | 12.4 | 0.00 | 32.0 |
| 87 | kernel | fastmace::tabled_outer_sum_backward | 447.8 | l1 | 99 | 16 | 30 | 64 | 54 | fma=51 lsu=48 alu=18 | long_scoreboard=4.98 wait=1.37 short_scoreboard=1.25 | 35/38 | 80 | 24/28/64/32 | 3.9 | 0.01 | 32.0 |
| 27 | kernel | fastmace::tabled_outer_sum_forward | 397.6 | latency:long_scoreboard | 98 | 19 | 22 | 30 | 28 | fma=30 lsu=17 alu=8 | long_scoreboard=9.98 wait=1.47 math_pipe_throttle=0.28 | 23/25 | 128 | 16/16/64/32 | 3.0 | 0.00 | 32.0 |
| 80 | segmented_polynomial_bwd_algo1_4 | cuequivariance::uniform_1d | 204.5 | pipe:fma | 98 | 23 | 19 | 5 | 50 | fma=96 lsu=4 alu=1 | math_pipe_throttle=3.17 wait=1.05 long_scoreboard=0.38 | 30/31 | 96 | 5/16/16/32 | 3.9 | 0.00 | 32.0 |
| 35 | segmented_polynomial_algo1_2 | cuequivariance::uniform_1d | 104.5 | pipe:fma | 95 | 29 | 20 | 6 | 49 | fma=92 lsu=7 lsu_mem_lg_cmd_read=2 | math_pipe_throttle=5.66 wait=1.12 long_scoreboard=1.11 | 46/56 | 56 | 18/32/32/32 | 4.0 | 0.00 | 32.0 |
| 44 | fm_tf32x3_remap_div_rr_64x64 | fastmace::tf32x3_linear_div | 96.3 | latency:math_pipe_throttle | 89 | 52 | 38 | 23 | 48 | tensor=56 tensor_op_hmma=56 shared=56 | math_pipe_throttle=1.47 wait=1.18 long_scoreboard=0.52 | 17/19 | 134 | 3/3/16/32 | 14.8 | 0.01 | 32.0 |
| 67 | fm_tf32x3_rr_64x64 | ## Call CompiledFxGraph fubhakiesrre5jmr | 90.8 | latency:math_pipe_throttle | 97 | 32 | 34 | 31 | 50 | tensor=55 tensor_op_hmma=55 shared=55 | math_pipe_throttle=1.38 wait=1.03 long_scoreboard=0.44 | 18/19 | 134 | 3/3/16/32 | 16.0 | 0.03 | 32.0 |
| 45 | fm_tf32x3_remap_div_rr_64x64 | fastmace::tf32x3_linear_div | 90.1 | latency:math_pipe_throttle | 92 | 56 | 41 | 24 | 49 | tensor=55 tensor_op_hmma=55 shared=55 | math_pipe_throttle=1.28 wait=1.22 long_scoreboard=0.61 | 18/19 | 134 | 3/3/16/32 | 14.3 | 0.01 | 32.0 |
| 69 | fm_tf32x3_rr_64x64 | ## Call CompiledFxGraph fubhakiesrre5jmr | 87.0 | latency:math_pipe_throttle | 95 | 42 | 41 | 30 | 50 | tensor=55 tensor_op_hmma=55 shared=55 | math_pipe_throttle=1.30 wait=1.03 long_scoreboard=0.60 | 18/19 | 134 | 3/3/16/32 | 16.0 | 0.03 | 32.0 |
| 86 | fm_tf32x3_species_rr_64x64 | fastmace::species_mm_rows_grad | 74.4 | latency:wait | 94 | 43 | 38 | 28 | 51 | tensor=49 tensor_op_hmma=49 shared=49 | wait=1.31 math_pipe_throttle=0.96 long_scoreboard=0.75 | 18/19 | 134 | 3/3/16/32 | 9.1 | 0.02 | 32.0 |
| 81 | triton_red_fused_add_div_mul_neg_species_mm_rows_g | triton_red_fused_add_div_mul_neg_species | 73.4 | dram | 92 | 83 | 62 | 20 | 19 | fma=18 lsu=12 adu=10 | long_scoreboard=36.88 mio_throttle=13.94 barrier=9.41 | 95/100 | 32 | 4/14/4/32 | 11.0 | 0.01 | 32.0 |
| 33 | fm_tf32x3_species_rr_64x64 | fastmace::species_mm_rows | 72.5 | latency:wait | 92 | 52 | 39 | 26 | 49 | tensor=51 tensor_op_hmma=51 shared=51 | wait=1.31 math_pipe_throttle=1.15 long_scoreboard=0.74 | 18/19 | 134 | 3/3/16/32 | 9.7 | 0.01 | 32.0 |
| 43 | fm_tf32x3_remap_div_rr_64x64 | fastmace::tf32x3_linear_div | 62.8 | latency:math_pipe_throttle | 85 | 36 | 29 | 21 | 46 | tensor=54 tensor_op_hmma=54 shared=54 | math_pipe_throttle=1.45 wait=1.16 long_scoreboard=0.49 | 16/19 | 134 | 3/3/16/32 | 14.8 | 0.01 | 32.0 |
| 20 | triton_poi_fused__to_copy_add_arange_div_exp_full_ | triton_poi_fused__to_copy_add_arange_div | 59.9 | latency:long_scoreboard | 83 | 20 | 27 | 17 | 30 | fma=27 alu=22 alu_realtime=22 | long_scoreboard=3.08 wait=1.51 drain=1.44 | 23/25 | 89 | 2/6/8/32 | 10.5 | 0.03 | 32.0 |
| 65 | fm_tf32x3_rr_64x64 | ## Call CompiledFxGraph fubhakiesrre5jmr | 58.6 | latency:math_pipe_throttle | 92 | 20 | 28 | 29 | 50 | tensor=54 tensor_op_hmma=54 shared=54 | math_pipe_throttle=1.44 wait=1.03 barrier=0.38 | 18/19 | 134 | 3/3/16/32 | 16.0 | 0.03 | 32.0 |
| 59 | segmented_polynomial_bwd_algo1_4 | cuequivariance::uniform_1d | 55.9 | pipe:fma | 94 | 75 | 61 | 19 | 44 | fma=77 lsu=12 alu=3 | long_scoreboard=4.18 math_pipe_throttle=3.99 wait=1.18 | 42/50 | 64 | 8/32/16/32 | 3.9 | 0.00 | 32.0 |
| 32 | fm_tf32x3_species_rr_64x64 | fastmace::species_mm_rows | 55.1 | latency:wait | 89 | 40 | 34 | 24 | 48 | tensor=50 tensor_op_hmma=50 shared=50 | wait=1.30 math_pipe_throttle=1.21 long_scoreboard=0.67 | 17/19 | 134 | 3/3/16/32 | 9.7 | 0.01 | 32.0 |
| 85 | fm_tf32x3_species_rr_64x64 | fastmace::species_mm_rows_grad | 54.0 | latency:wait | 94 | 34 | 33 | 28 | 50 | tensor=48 tensor_op_hmma=48 shared=48 | wait=1.31 math_pipe_throttle=1.07 long_scoreboard=0.58 | 18/19 | 134 | 3/3/16/32 | 9.1 | 0.02 | 32.0 |
| 94 | triton_poi_fused_add_div_eq_index_put_masked_fill_ | triton_poi_fused_add_div_eq_index_put_ma | 52.4 | latency:mio_throttle | 80 | 18 | 32 | 22 | 15 | adu=14 fma=10 alu=10 | mio_throttle=32.48 barrier=14.74 lg_throttle=9.65 | 86/100 | 32 | 8/20/8/32 | 5.7 | 0.00 | 32.0 |
| 60 | triton_red_fused_add_div_mul_neg_sum_2 | triton_red_fused_add_div_mul_neg_sum_2 | 47.1 | dram | 94 | 90 | 67 | 25 | 25 | fma=20 lsu=17 adu=17 | long_scoreboard=38.09 short_scoreboard=5.69 mio_throttle=5.58 | 93/100 | 31 | 4/14/4/32 | 11.0 | 0.00 | 32.0 |
| 92 | triton_poi_fused_add_edge_sh3_backward_14 | triton_poi_fused_add_edge_sh3_backward_1 | 41.0 | dram | 85 | 81 | 62 | 15 | 4 | lsu=3 fma=2 alu=2 | long_scoreboard=223.07 drain=30.88 lg_throttle=24.69 | 87/100 | 24 | 21/32/16/32 | 16.0 | 0.00 | 32.0 |
| 34 | triton_poi_fused_add_div_view_20 | triton_poi_fused_add_div_view_20 | 38.3 | dram | 88 | 72 | 67 | 25 | 29 | adu=27 fma=19 alu=19 | long_scoreboard=22.42 lg_throttle=6.88 mio_throttle=6.67 | 89/100 | 25 | 16/32/16/32 | 4.0 | 0.00 | 32.0 |
| 46 | segmented_polynomial_algo1_4 | cuequivariance::uniform_1d | 37.8 | dram | 89 | 64 | 47 | 12 | 35 | fma=57 lsu=13 alu=5 | long_scoreboard=13.91 math_pipe_throttle=4.93 lg_throttle=2.35 | 66/75 | 40 | 12/32/16/32 | 3.9 | 0.00 | 32.0 |
| 31 | fm_tf32x3_species_rr_64x64 | fastmace::species_mm_rows | 36.5 | latency:wait | 84 | 33 | 27 | 22 | 45 | tensor=47 tensor_op_hmma=47 shared=47 | wait=1.28 math_pipe_throttle=1.23 long_scoreboard=0.66 | 16/19 | 134 | 3/3/16/32 | 9.7 | 0.01 | 32.0 |

## Main-pass summary keys

```
{
 "tool": "step_breakdown",
 "created_utc": "2026-11-09T07:55:13.287458+00:00",
 "host": "modal",
 "python": "/usr/local/bin/python",
 "fastmace_commit": "18fe4b328098c8025bdffecc501c7c93349e8fe2",
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
 "oeq_runtime": "PatchRecord(name='oeq_runtime', value='/__modal/volumes/vo-QuZYWI1jH3gh2H8SolNdI6/cache/A100-80GB/oeq/cuda-runtime', targets=('env:CMAKE_CUDA_COMPILER', 'env:CPATH', 'env:CUDACXX', 'env:CUDA_HOME', 'env:CUDA_PATH', 'env:MAX_JOBS', 'env:OEQ_JIT_EXTENSION', 'env:OEQ_NVCC_ATTEMPTS', 'env:PATH', 'env:PYTORCH_NVCC', 'env:TORCH_EXTENSIONS_DIR'), detail='OEq host-extension build environment; preloaded libcudart.so.13, libnvrtc.so.13, libcublasLt.so.13, libcublas.so.13; host extension: reused')",
 "other_gpu_processes": [],
 "compute_capability": "8.0",
 "sms": 108,
 "max_sm_clock_mhz": 1410.0,
 "fp32_peak_tflops": 19.49184,
 "model": "omat0-medium",
 "checkpoint": "/__modal/volumes/vo-QuZYWI1jH3gh2H8SolNdI6/models/mace-omat-0-medium.model",
 "checkpoint_sha256": "d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a",
 "backend": "hybrid"
}
```
