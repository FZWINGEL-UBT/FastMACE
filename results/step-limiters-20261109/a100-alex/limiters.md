# Limiters: NVIDIA A100-SXM4-80GB

ncu kernels: 95 (95 aligned with the profiler's sequence); kernel time 6208.5 µs per force call (ncu, running clocks, warm caches).

## Step stages (main pass)

```
stage,calls,ms_per_call,ms_per_step,host_ms_per_step
B: forces + second half kick,300,6.3740,6.3740,0.0711
Neighbour-list rebuild (GPU),39,7.9929,1.0391,1.0422
"ACP: Langevin half kick, positions, neighbour check, edge compaction",300,0.0830,0.0830,0.0644
Noise upload (every step a slice; a new batch every noise_batch),300,0.0373,0.0373,0.1273
GPU idle between stages,,,-0.8801,
```

## Kernel time by limiter

| Limiter | µs | Share |
| --- | ---: | ---: |
| pipe:fma | 3675.7 | 59.2% |
| latency | 1402.2 | 22.6% |
| l1 | 470.0 | 7.6% |
| underfilled | 301.2 | 4.9% |
| dram | 184.5 | 3.0% |
| tiny | 89.5 | 1.4% |
| issue | 55.5 | 0.9% |
| l2 | 29.9 | 0.5% |

## By category

| Category | µs | Share | Limiters (µs) |
| --- | ---: | ---: | --- |
| Edge elementwise (geometry, radial basis, gathers, scatters) | 4525.0 | 72.9% | pipe:fma 3432, latency:long_scoreboard 451, l1 442, latency:mio_throttle 56, issue 55, underfilled 54, dram 35 |
| Other | 1235.9 | 19.9% | latency:math_pipe_throttle 496, latency:wait 400, pipe:fma 188, underfilled 105, l1 28, l2 16, tiny 3 |
| Atom-level elementwise (gates, products, residuals, readouts) | 241.6 | 3.9% | dram 84, underfilled 78, tiny 66, l2 14 |
| Atom-level cuEq tensor products (symmetric contraction) | 89.6 | 1.4% | pipe:fma 56, dram 34 |
| Copies and fills | 86.6 | 1.4% | underfilled 39, dram 32, tiny 16 |
| Atom-level linear layers (GEMM) | 29.9 | 0.5% | underfilled 25, tiny 5 |

## 25 largest kernels

| # | Kernel | Op | µs | Limiter | SM fill | DRAM | L2 | L1 | Issue | Pipes | Stalls | Warps act/theo | Regs | Occ. limits r/s/w/b | Sect/ld | Smem confl | Thr/inst |
| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| 70 | oeq_projected_backward_wprime | fastmace::streaming_conv_wprime_backward | 2177.0 | pipe:fma | 99 | 7 | 55 | 52 | 60 | fma=83 lsu=32 alu=12 | long_scoreboard=2.51 math_pipe_throttle=1.88 wait=1.11 | 37/38 | 79 | 6/6/16/32 | 12.6 | 0.03 | 31.9 |
| 40 | spline_forward0 | fastmace::spline_message_blocks | 1255.2 | pipe:fma | 98 | 7 | 42 | 33 | 71 | fma=84 alu=43 alu_realtime=43 | math_pipe_throttle=2.11 wait=1.13 long_scoreboard=0.89 | 36/38 | 80 | 6/17/16/32 | 6.8 | 0.00 | 32.0 |
| 85 | kernel | fastmace::tabled_outer_sum_backward | 442.0 | l1 | 98 | 15 | 31 | 64 | 55 | fma=52 lsu=49 alu=18 | long_scoreboard=4.84 wait=1.37 short_scoreboard=1.25 | 35/38 | 80 | 24/28/64/32 | 3.9 | 0.01 | 32.0 |
| 27 | kernel | fastmace::tabled_outer_sum_forward | 390.3 | latency:long_scoreboard | 98 | 19 | 22 | 31 | 28 | fma=31 lsu=18 alu=9 | long_scoreboard=9.78 wait=1.47 math_pipe_throttle=0.28 | 23/25 | 128 | 16/16/64/32 | 3.0 | 0.00 | 32.0 |
| 79 | sc_generated_backward | fastmace::sc_generated_backward | 110.0 | pipe:fma | 97 | 41 | 35 | 10 | 50 | fma=93 lsu=9 lsu_mem_lg_cmd_read=2 | math_pipe_throttle=2.93 wait=1.16 long_scoreboard=0.82 | 30/31 | 96 | 5/14/16/32 | 3.9 | 0.00 | 31.9 |
| 43 | fm_tf32x3_remap_div_rr_64x64 | fastmace::tf32x3_linear_div | 94.1 | latency:math_pipe_throttle | 89 | 51 | 39 | 23 | 48 | tensor=57 tensor_op_hmma=57 shared=57 | math_pipe_throttle=1.48 wait=1.18 long_scoreboard=0.48 | 17/19 | 134 | 3/3/16/32 | 14.8 | 0.01 | 32.0 |
| 44 | fm_tf32x3_remap_div_rr_64x64 | fastmace::tf32x3_linear_div | 89.7 | latency:math_pipe_throttle | 90 | 53 | 41 | 24 | 49 | tensor=55 tensor_op_hmma=55 shared=55 | math_pipe_throttle=1.28 wait=1.22 long_scoreboard=0.57 | 18/19 | 134 | 3/3/16/32 | 14.3 | 0.01 | 32.0 |
| 66 | fm_tf32x3_rr_64x64 | ## Call CompiledFxGraph fs3bhxg2xpwsc4nz | 89.5 | latency:math_pipe_throttle | 96 | 31 | 34 | 31 | 50 | tensor=55 tensor_op_hmma=55 shared=55 | math_pipe_throttle=1.38 wait=1.03 long_scoreboard=0.42 | 18/19 | 134 | 3/3/16/32 | 16.0 | 0.03 | 32.0 |
| 68 | fm_tf32x3_rr_64x64 | ## Call CompiledFxGraph fs3bhxg2xpwsc4nz | 86.2 | latency:math_pipe_throttle | 94 | 41 | 41 | 30 | 50 | tensor=55 tensor_op_hmma=55 shared=55 | math_pipe_throttle=1.29 wait=1.03 long_scoreboard=0.60 | 18/19 | 134 | 3/3/16/32 | 16.0 | 0.03 | 32.0 |
| 34 | sc_generated_forward | fastmace::sc_generated_forward | 77.6 | pipe:fma | 95 | 36 | 27 | 9 | 50 | fma=93 lsu=9 lsu_mem_lg_cmd_read=2 | math_pipe_throttle=3.56 long_scoreboard=1.54 wait=1.13 | 35/38 | 79 | 6/32/16/32 | 3.9 | 0.00 | 32.0 |
| 84 | fm_tf32x3_species_rr_64x64 | fastmace::species_mm_rows_grad | 73.5 | latency:wait | 93 | 41 | 38 | 28 | 51 | tensor=49 tensor_op_hmma=49 shared=49 | wait=1.31 math_pipe_throttle=0.96 long_scoreboard=0.73 | 18/19 | 134 | 3/3/16/32 | 9.1 | 0.02 | 32.0 |
| 33 | fm_tf32x3_species_rr_64x64 | fastmace::species_mm_rows | 71.4 | latency:wait | 91 | 49 | 39 | 26 | 49 | tensor=52 tensor_op_hmma=52 shared=52 | wait=1.31 math_pipe_throttle=1.16 long_scoreboard=0.67 | 18/19 | 134 | 3/3/16/32 | 9.7 | 0.01 | 32.0 |
| 42 | fm_tf32x3_remap_div_rr_64x64 | fastmace::tf32x3_linear_div | 61.8 | latency:math_pipe_throttle | 85 | 35 | 30 | 21 | 46 | tensor=55 tensor_op_hmma=55 shared=55 | math_pipe_throttle=1.45 wait=1.16 long_scoreboard=0.47 | 16/19 | 134 | 3/3/16/32 | 14.8 | 0.01 | 32.0 |
| 20 | triton_poi_fused__to_copy_add_arange_div_exp_full_ | triton_poi_fused__to_copy_add_arange_div | 60.5 | latency:long_scoreboard | 83 | 19 | 27 | 16 | 29 | fma=27 alu=21 alu_realtime=21 | long_scoreboard=3.20 wait=1.51 drain=1.43 | 23/25 | 89 | 2/6/8/32 | 10.5 | 0.04 | 32.0 |
| 64 | fm_tf32x3_rr_64x64 | ## Call CompiledFxGraph fs3bhxg2xpwsc4nz | 57.8 | latency:math_pipe_throttle | 92 | 22 | 27 | 29 | 49 | tensor=54 tensor_op_hmma=54 shared=54 | math_pipe_throttle=1.44 wait=1.03 long_scoreboard=0.41 | 18/19 | 134 | 3/3/16/32 | 16.0 | 0.03 | 32.0 |
| 58 | segmented_polynomial_bwd_algo1_4 | cuequivariance::uniform_1d | 55.9 | pipe:fma | 93 | 72 | 60 | 18 | 44 | fma=76 lsu=12 alu=3 | long_scoreboard=4.44 math_pipe_throttle=3.94 wait=1.18 | 44/50 | 64 | 8/32/16/32 | 3.9 | 0.00 | 32.0 |
| 92 | triton_poi_fused_add_div_eq_index_put_masked_fill_ | triton_poi_fused_add_div_eq_index_put_ma | 55.6 | latency:mio_throttle | 80 | 16 | 29 | 21 | 15 | adu=14 fma=9 alu=9 | mio_throttle=36.33 barrier=15.84 lg_throttle=10.32 | 93/100 | 32 | 8/20/8/32 | 5.7 | 0.00 | 32.0 |
| 32 | fm_tf32x3_species_rr_64x64 | fastmace::species_mm_rows | 53.7 | latency:wait | 88 | 38 | 33 | 24 | 48 | tensor=50 tensor_op_hmma=50 shared=50 | wait=1.30 math_pipe_throttle=1.22 long_scoreboard=0.61 | 17/19 | 134 | 3/3/16/32 | 9.7 | 0.01 | 32.0 |
| 83 | fm_tf32x3_species_rr_64x64 | fastmace::species_mm_rows_grad | 53.2 | latency:wait | 93 | 32 | 33 | 28 | 51 | tensor=48 tensor_op_hmma=48 shared=48 | wait=1.31 math_pipe_throttle=1.06 long_scoreboard=0.58 | 18/19 | 134 | 3/3/16/32 | 9.1 | 0.02 | 32.0 |
| 59 | triton_red_fused_add_div_mul_neg_sum_2 | triton_red_fused_add_div_mul_neg_sum_2 | 45.3 | dram | 93 | 89 | 69 | 25 | 26 | fma=21 lsu=18 adu=18 | long_scoreboard=37.37 short_scoreboard=5.34 mio_throttle=4.91 | 95/100 | 31 | 4/14/4/32 | 11.0 | 0.00 | 32.0 |
| 31 | fm_tf32x3_species_rr_64x64 | fastmace::species_mm_rows | 36.2 | latency:wait | 83 | 30 | 27 | 22 | 46 | tensor=48 tensor_op_hmma=48 shared=48 | wait=1.28 math_pipe_throttle=1.22 long_scoreboard=0.59 | 17/19 | 134 | 3/3/16/32 | 9.7 | 0.01 | 32.0 |
| 82 | fm_tf32x3_species_rr_64x64 | fastmace::species_mm_rows_grad | 36.2 | latency:wait | 86 | 26 | 28 | 25 | 49 | tensor=46 tensor_op_hmma=46 shared=46 | wait=1.29 math_pipe_throttle=1.14 long_scoreboard=0.61 | 18/19 | 134 | 3/3/16/32 | 9.1 | 0.02 | 32.0 |
| 90 | triton_poi_fused_add_edge_sh3_backward_13 | triton_poi_fused_add_edge_sh3_backward_1 | 35.0 | dram | 85 | 80 | 68 | 18 | 5 | lsu=4 fma=3 alu=2 | long_scoreboard=168.62 lg_throttle=29.21 drain=25.94 | 88/100 | 24 | 21/32/16/32 | 16.0 | 0.00 | 32.0 |
| 45 | segmented_polynomial_algo1_1 | cuequivariance::uniform_1d | 33.7 | dram | 84 | 68 | 52 | 13 | 41 | fma=67 lsu=14 alu=4 | long_scoreboard=7.77 math_pipe_throttle=1.56 wait=1.34 | 39/50 | 46 | 40/32/64/32 | 4.0 | 0.00 | 32.0 |
| 26 | void at::vectorized_elementwise_kernel<4, at::Fill | aten::fill_ | 31.9 | dram | 88 | 85 | 71 | 40 | 8 | fma=6 alu=5 alu_realtime=5 | mio_throttle=57.96 lg_throttle=56.23 drain=31.11 | 89/100 | 16 | 32/32/16/32 |  | 0.00 | 32.0 |

## Main-pass summary keys

```
{
 "tool": "step_breakdown",
 "created_utc": "2026-11-09T12:22:32.202505+00:00",
 "host": "a0731",
 "python": "/tmp/4969862.alex/fm/env/bin/python",
 "fastmace_commit": "735158a2dce2971f81a189e95921ed5ac90538de",
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
 "gpu": "NVIDIA A100-SXM4-80GB",
 "cuda": "13.0",
 "oeq_runtime": "PatchRecord(name='oeq_runtime', value='/tmp/4969862.alex/fm/cache/oeq/cuda-runtime', targets=('env:CMAKE_CUDA_COMPILER', 'env:CPATH', 'env:CUDACXX', 'env:CUDA_HOME', 'env:CUDA_PATH', 'env:MAX_JOBS', 'env:OEQ_JIT_EXTENSION', 'env:OEQ_NVCC_ATTEMPTS', 'env:PATH', 'env:PYTORCH_NVCC', 'env:TORCH_EXTENSIONS_DIR'), detail='OEq host-extension build environment; preloaded libcudart.so.13, libnvrtc.so.13, libcublasLt.so.13, libcublas.so.13; host extension: built')",
 "other_gpu_processes": [],
 "compute_capability": "8.0",
 "sms": 108,
 "max_sm_clock_mhz": 1410.0,
 "fp32_peak_tflops": 19.49184,
 "model": "omat0-medium",
 "checkpoint": "/home/atuin/v130cb/v130cb18/fm-a100omat/models/mace-omat-0-medium.model",
 "checkpoint_sha256": "d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a",
 "backend": "hybrid"
}
```
