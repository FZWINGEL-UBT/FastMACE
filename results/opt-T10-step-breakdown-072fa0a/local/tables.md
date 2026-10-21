
## NaCl (216 atoms, 8192 padded edges)

| Quantity | Value | Source |
| --- | ---: | --- |
| Wall time per MD step | 3.028 ms | un-profiled window (1000 steps) |
| GPU span per step (stage events) | 3.032 ms | CUDA events around stages |
| GPU idle between stages | -0.111 ms | CUDA events |
| B graph replay (forces + 2nd half kick) | 2.827 ms | CUDA events, un-profiled |
| Force-graph replay span / kernel time | 2.874 / 2.720 ms | profiler, 10 replays |
| Kernels per replay | 264 | profiler |
| Gaps inside a replay | 151 µs (263 gaps) | profiler |
| Gaps between graph nodes (nsys) | 174 µs (239 gaps) | nsys, every node traced |
| Eager force call kernel sum | 2.745 ms (256 kernels) | profiler, eager calls |

Stages (ms per step):

| Stage | ms | % of GPU span |
| --- | ---: | ---: |
| B: forces + second half kick | 2.826 | 93.2 |
| Neighbour-list rebuild (GPU) | 0.163 | 5.4 |
| ACP: Langevin half kick, positions, neighbour check, edge compaction | 0.144 | 4.7 |
| Noise upload (every step a slice; a new batch every noise_batch) | 0.009 | 0.3 |
| GPU idle between stages | -0.111 | -3.7 |

Force call by part (profiler, eager calls: 2.745 ms of kernels; forward/backward split from ncu's per-kernel durations, boundary at kernel 143 of 256):

| Part | ms | % of force kernels | kernels | forward / backward |
| --- | ---: | ---: | ---: | --- |
| Atom-level linears (cuBLAS GEMMs and FastMACE Triton GEMMs) | 1.306 | 47.6 | 95 | 49 / 51 % |
| Radial networks and embeddings | 0.613 | 22.3 | 31 | 45 / 55 % |
| Convolution (interaction 1 reassociated sums, interaction 2 streaming OEq) | 0.426 | 15.5 | 7 | 31 / 69 % |
| Atom-level elementwise (gates, products, residuals, readouts, species tables) | 0.247 | 9.0 | 95 | 44 / 56 % |
| Symmetric contraction (cuEq) | 0.077 | 2.8 | 4 | – |
| Neighbour list and edge prep (edge geometry, radial basis, gathers, scatters) | 0.057 | 2.1 | 12 | 54 / 46 % |
| Copies and fills | 0.018 | 0.7 | 12 | 31 / 69 % |
| **All (ncu durations)** | | | | **44 / 56 %** (1.67 / 2.17 ms) |

Top 15 kernels by ncu duration (one eager force call, 4.03 ms of ncu kernel time; DRAM peak 448 GB/s, FP32 peak 28.5 TFLOPS):

| # | Kernel (op; inputs) | Part | Launches | µs | % | DRAM GB/s (% peak) | TFLOPS (% peak) | SM / memory SOL % | Occupancy achieved / theoretical % | Bound |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | `oeq_projected_backward (fastmace::streaming_conv_backward)`; [[216, 512], [8192, 16], [8192, 1280], [8192], [8192], [217] | Convolution | 1 | 406 | 10.1 | 196 (44) | 2.03 (7) | 23 / 87 | 63 / 67 | memory |
| 2 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(cutl (aten::mm)`; [[1512, 512], [512, 512], [1512, 512]] | Atom-level linears | 4 | 275 | 6.8 | 66 (15) | 11.80 (41) | 65 / 62 | 23 / 25 | compute |
| 3 | `Kernel (fastmace::folded_linear_input_grad (trit)`; "" | Radial networks and embeddings | 1 | 230 | 5.7 | 195 (43) | 5.59 (20) | 27 / 87 | 28 / 33 | memory |
| 4 | `oeq_projected_forward (fastmace::streaming_conv_forward)`; [[216, 512], [8192, 16], [8192, 1280], [8192], [217], []] | Convolution | 1 | 210 | 5.2 | 202 (45) | 1.21 (4) | 9 / 90 | 47 / 50 | memory |
| 5 | `void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3 (aten::mm)`; [[8192, 68], [68, 1280], [8192, 1280]] | Radial networks and embeddings | 1 | 204 | 5.1 | 146 (33) | 7.81 (27) | 54 / 73 | 47 / 50 | memory |
| 6 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(cutl (aten::mm)`; [[648, 512], [512, 512], [648, 512]] | Atom-level linears | 6 | 203 | 5.0 | 79 (18) | 11.00 (39) | 59 / 55 | 22 / 25 | latency / occupancy |
| 7 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(cutl (aten::mm)`; [[1080, 512], [512, 512], [1080, 512]] | Atom-level linears | 4 | 201 | 5.0 | 82 (18) | 11.43 (40) | 61 / 58 | 23 / 25 | compute |
| 8 | ` (?)`;  | Symmetric contraction | 4 | 188 | 4.7 | 180 (40) | 4.40 (15) | 28 / 81 | 59 / 83 | memory |
| 9 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_tn_align1>(cutl (aten::addmm)`; [[64], [8192, 64], [64, 64], [], [], [8192, 64]] | Radial networks and embeddings | 4 | 81 | 2.0 | 97 (22) | 6.38 (22) | 38 / 51 | 23 / 25 | latency / occupancy |
| 10 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>(cutl (aten::mm)`; [[216, 512], [512, 512], [216, 512]] | Atom-level linears | 4 | 80 | 2.0 | 96 (21) | 5.94 (21) | 42 / 51 | 18 / 25 | latency / occupancy |
| 11 | `void magma_sgemmEx_kernel<float, float, float, false, false, 6, 3, 5,  (aten::mm)`; [[8192, 64], [64, 64], [8192, 64]] | Radial networks and embeddings | 4 | 77 | 1.9 | 125 (28) | 3.34 (12) | 23 / 56 | 27 / 58 | latency / occupancy |
| 12 | `triton_per_fused_add_div_gather_mm_mul_native_layer_norm_native_layer_`; [[8192, 64], [8192, 64], [], [8192, 1], [8192, 1], [8192, 64 | Radial networks and embeddings | 1 | 70 | 1.7 | 167 (37) | 1.39 (5) | 54 / 75 | 80 / 83 | memory |
| 13 | `triton_poi_fused_mul_sum_view_8`; [[1512, 512], [1512, 512], [216, 1, 512], []] | Atom-level elementwise | 2 | 67 | 1.7 | 203 (45) | 0.04 (0) | 2 / 91 | 62 / 100 | memory |
| 14 | `Kernel (fastmace::edge_outer_sum_backward (trito)`; "" | Convolution | 1 | 58 | 1.4 | 60 (13) | 1.37 (5) | 47 / 47 | 22 / 25 | latency / occupancy |
| 15 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_nn_align1>(cutla (aten::mm)`; [[8192, 128], [128, 10], [8192, 10]] | Radial networks and embeddings | 2 | 54 | 1.3 | 169 (38) | 4.67 (16) | 27 / 75 | 26 / 33 | memory |

## NbCl5 (1944 atoms, 56320 padded edges)

| Quantity | Value | Source |
| --- | ---: | --- |
| Wall time per MD step | 21.434 ms | un-profiled window (1000 steps) |
| GPU span per step (stage events) | 21.575 ms | CUDA events around stages |
| GPU idle between stages | -0.511 ms | CUDA events |
| B graph replay (forces + 2nd half kick) | 21.273 ms | CUDA events, un-profiled |
| Force-graph replay span / kernel time | 21.188 / 21.098 ms | profiler, 10 replays |
| Kernels per replay | 230 | profiler |
| Gaps inside a replay | 102 µs (229 gaps) | profiler |
| Gaps between graph nodes (nsys) | 110 µs (215 gaps) | nsys, every node traced |
| Eager force call kernel sum | 21.013 ms (222 kernels) | profiler, eager calls |

Stages (ms per step):

| Stage | ms | % of GPU span |
| --- | ---: | ---: |
| B: forces + second half kick | 21.295 | 98.7 |
| Neighbour-list rebuild (GPU) | 0.593 | 2.7 |
| ACP: Langevin half kick, positions, neighbour check, edge compaction | 0.175 | 0.8 |
| Noise upload (every step a slice; a new batch every noise_batch) | 0.024 | 0.1 |
| GPU idle between stages | -0.511 | -2.4 |

Force call by part (profiler, eager calls: 21.013 ms of kernels; forward/backward split from ncu's per-kernel durations, boundary at kernel 128 of 222):

| Part | ms | % of force kernels | kernels | forward / backward |
| --- | ---: | ---: | ---: | --- |
| Atom-level linears (cuBLAS GEMMs and FastMACE Triton GEMMs) | 9.025 | 42.9 | 71 | 49 / 51 % |
| Radial networks and embeddings | 3.880 | 18.5 | 31 | 60 / 40 % |
| Convolution (interaction 1 reassociated sums, interaction 2 streaming OEq) | 3.570 | 17.0 | 7 | 42 / 58 % |
| Atom-level elementwise (gates, products, residuals, readouts, species tables) | 2.746 | 13.1 | 83 | 44 / 56 % |
| Symmetric contraction (cuEq) | 1.155 | 5.5 | 4 | 100 / 0 % |
| Neighbour list and edge prep (edge geometry, radial basis, gathers, scatters) | 0.524 | 2.5 | 12 | 74 / 26 % |
| Copies and fills | 0.112 | 0.5 | 14 | 61 / 39 % |
| **All (ncu durations)** | | | | **50 / 50 %** (10.85 / 10.66 ms) |

Top 15 kernels by ncu duration (one eager force call, 22.39 ms of ncu kernel time; DRAM peak 448 GB/s, FP32 peak 28.5 TFLOPS):

| # | Kernel (op; inputs) | Part | Launches | µs | % | DRAM GB/s (% peak) | TFLOPS (% peak) | SM / memory SOL % | Occupancy achieved / theoretical % | Bound |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | `gemm (fastmace::joined_products_input_grads)`; [[1944, 8192], [[512, 512], [512, 512], [512, 512], [512, 51 | Atom-level linears | 6 | 2413 | 10.8 | 227 (51) | 12.71 (45) | 69 / 69 | 24 / 25 | compute |
| 2 | `gemm (fastmace::joined_products)`; [[[1944, 512], [5832, 512], [9720, 512], [13608, 512]], [[51 | Atom-level linears | 6 | 2398 | 10.7 | 120 (27) | 12.79 (45) | 69 / 62 | 24 / 25 | compute |
| 3 | `void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3 (aten::mm)`; [[56320, 68], [68, 1280], [56320, 1280]] | Radial networks and embeddings | 1 | 1990 | 8.9 | 166 (37) | 5.87 (21) | 38 / 66 | 49 / 50 | memory |
| 4 | `oeq_projected_backward (fastmace::streaming_conv_backward)`; [[1944, 512], [56320, 16], [56320, 1280], [56320], [56320],  | Convolution | 1 | 1830 | 8.2 | 368 (82) | 3.31 (12) | 37 / 83 | 66 / 67 | memory |
| 5 | `oeq_projected_forward (fastmace::streaming_conv_forward)`; [[1944, 512], [56320, 16], [56320, 1280], [56320], [1945], [ | Convolution | 1 | 1681 | 7.5 | 210 (47) | 1.09 (4) | 8 / 94 | 49 / 50 | memory |
| 6 | ` (?)`;  | Symmetric contraction | 3 | 886 | 4.0 | 364 (81) | 7.71 (27) | 56 / 83 | 77 / 83 | memory |
| 7 | `Kernel (fastmace::folded_linear_input_grad (trit)`; "" | Radial networks and embeddings | 1 | 866 | 3.9 | 371 (83) | 10.89 (38) | 60 / 84 | 25 / 25 | memory |
| 8 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(cutl (aten::mm)`; [[5832, 512], [512, 512], [5832, 512]] | Atom-level linears | 2 | 483 | 2.2 | 79 (18) | 12.79 (45) | 69 / 64 | 23 / 25 | compute |
| 9 | `triton_per_fused_add_div_gather_mm_mul_native_layer_norm_native_layer_`; [[56320, 64], [56320, 64], [], [56320, 1], [56320, 1], [5632 | Radial networks and embeddings | 1 | 383 | 1.7 | 361 (81) | 1.86 (7) | 76 / 82 | 79 / 83 | memory |
| 10 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(cutl (aten::mm)`; [[1944, 512], [512, 512]] | Atom-level linears | 4 | 353 | 1.6 | 67 (15) | 11.88 (42) | 67 / 61 | 24 / 25 | compute |
| 11 | `Kernel (fastmace::edge_outer_sum_backward (trito)`; "" | Convolution | 1 | 329 | 1.5 | 171 (38) | 1.78 (6) | 61 / 61 | 24 / 25 | compute |
| 12 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(cutl (aten::mm)`; [[9720, 384], [384, 512], [9720, 512]] | Atom-level linears | 1 | 316 | 1.4 | 87 (19) | 12.13 (43) | 66 / 72 | 24 / 25 | memory |
| 13 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(cutl (aten::mm)`; [[9720, 512], [512, 384], [9720, 384]] | Atom-level linears | 1 | 302 | 1.3 | 96 (21) | 12.67 (44) | 68 / 70 | 24 / 25 | memory |
| 14 | `void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3 (aten::addmm)`; [[64], [56320, 64], [64, 64], [], [], [56320, 64]] | Radial networks and embeddings | 4 | 293 | 1.3 | 232 (52) | 6.39 (22) | 46 / 84 | 47 / 50 | memory |
| 15 | `triton_poi_fused_mul_sum_view_5`; [[13608, 512], [13608, 512], [1944, 1, 512], []] | Atom-level elementwise | 2 | 277 | 1.2 | 423 (94) | 0.09 (0) | 2 / 96 | 75 / 83 | memory |

## NaNbO3 (1080 atoms, 82688 padded edges)

| Quantity | Value | Source |
| --- | ---: | --- |
| Wall time per MD step | 18.788 ms | un-profiled window (1000 steps) |
| GPU span per step (stage events) | 18.856 ms | CUDA events around stages |
| GPU idle between stages | -0.713 ms | CUDA events |
| B graph replay (forces + 2nd half kick) | 18.556 ms | CUDA events, un-profiled |
| Force-graph replay span / kernel time | 18.452 / 18.344 ms | profiler, 10 replays |
| Kernels per replay | 240 | profiler |
| Gaps inside a replay | 115 µs (239 gaps) | profiler |
| Gaps between graph nodes (nsys) | 121 µs (223 gaps) | nsys, every node traced |
| Eager force call kernel sum | 18.242 ms (232 kernels) | profiler, eager calls |

Stages (ms per step):

| Stage | ms | % of GPU span |
| --- | ---: | ---: |
| B: forces + second half kick | 18.556 | 98.4 |
| Neighbour-list rebuild (GPU) | 0.817 | 4.3 |
| ACP: Langevin half kick, positions, neighbour check, edge compaction | 0.183 | 1.0 |
| Noise upload (every step a slice; a new batch every noise_batch) | 0.013 | 0.1 |
| GPU idle between stages | -0.713 | -3.8 |

Force call by part (profiler, eager calls: 18.242 ms of kernels; forward/backward split from ncu's per-kernel durations, boundary at kernel 135 of 232):

| Part | ms | % of force kernels | kernels | forward / backward |
| --- | ---: | ---: | ---: | --- |
| Radial networks and embeddings | 6.048 | 33.2 | 31 | 53 / 47 % |
| Atom-level linears (cuBLAS GEMMs and FastMACE Triton GEMMs) | 4.891 | 26.8 | 75 | 50 / 50 % |
| Convolution (interaction 1 reassociated sums, interaction 2 streaming OEq) | 4.623 | 25.3 | 7 | 27 / 73 % |
| Atom-level elementwise (gates, products, residuals, readouts, species tables) | 1.191 | 6.5 | 85 | 45 / 55 % |
| Neighbour list and edge prep (edge geometry, radial basis, gathers, scatters) | 0.768 | 4.2 | 12 | 77 / 23 % |
| Symmetric contraction (cuEq) | 0.588 | 3.2 | 4 | 33 / 67 % |
| Copies and fills | 0.133 | 0.7 | 18 | 58 / 42 % |
| **All (ncu durations)** | | | | **46 / 54 %** (8.80 / 10.46 ms) |

Top 15 kernels by ncu duration (one eager force call, 19.52 ms of ncu kernel time; DRAM peak 448 GB/s, FP32 peak 28.5 TFLOPS):

| # | Kernel (op; inputs) | Part | Launches | µs | % | DRAM GB/s (% peak) | TFLOPS (% peak) | SM / memory SOL % | Occupancy achieved / theoretical % | Bound |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | `oeq_projected_backward (fastmace::streaming_conv_backward)`; [[1080, 512], [82688, 16], [82688, 1280], [82688], [82688],  | Convolution | 1 | 2867 | 14.7 | 359 (80) | 3.26 (11) | 36 / 81 | 66 / 67 | memory |
| 2 | `void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3 (aten::mm)`; [[82688, 68], [68, 1280], [82688, 1280]] | Radial networks and embeddings | 1 | 1996 | 10.2 | 248 (55) | 9.04 (32) | 67 / 92 | 49 / 50 | memory |
| 3 | `Kernel (fastmace::folded_linear_input_grad (trit)`; "" | Radial networks and embeddings | 1 | 1226 | 6.3 | 396 (88) | 11.88 (42) | 67 / 90 | 25 / 25 | memory |
| 4 | `oeq_projected_forward (fastmace::streaming_conv_forward)`; [[1080, 512], [82688, 16], [82688, 1280], [82688], [1081], [ | Convolution | 1 | 1159 | 5.9 | 415 (93) | 2.39 (8) | 18 / 94 | 49 / 50 | memory |
| 5 | `gemm (fastmace::joined_products_input_grads)`; [[1080, 8192], [[512, 512], [512, 512], [512, 512], [512, 51 | Atom-level linears | 4 | 1041 | 5.3 | 78 (17) | 13.15 (46) | 71 / 61 | 24 / 25 | compute |
| 6 | `gemm (fastmace::joined_products)`; [[[1080, 512], [3240, 512], [5400, 512], [7560, 512]], [[512 | Atom-level linears | 4 | 1040 | 5.3 | 91 (20) | 13.16 (46) | 71 / 61 | 24 / 25 | compute |
| 7 | `triton_per_fused_add_div_gather_mm_mul_native_layer_norm_native_layer_`; [[82688, 64], [82688, 64], [], [82688, 1], [82688, 1], [8268 | Radial networks and embeddings | 1 | 643 | 3.3 | 340 (76) | 2.69 (9) | 44 / 77 | 16 / 17 | memory |
| 8 | `void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3 (aten::addmm)`; [[64], [82688, 64], [64, 64], [], [], [82688, 64]] | Radial networks and embeddings | 4 | 565 | 2.9 | 214 (48) | 5.13 (18) | 37 / 87 | 47 / 50 | memory |
| 9 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(cutl (aten::mm)`; [[3240, 512], [512, 512]] | Atom-level linears | 4 | 544 | 2.8 | 76 (17) | 12.59 (44) | 67 / 64 | 22 / 25 | compute |
| 10 | `Kernel (fastmace::edge_outer_sum_backward (trito)`; "" | Convolution | 1 | 463 | 2.4 | 157 (35) | 1.95 (7) | 67 / 67 | 24 / 25 | compute |
| 11 | `triton_per_fused_native_layer_norm_native_layer_norm_backward_silu_bac`; [[82688, 64], [82688, 64], [82688, 1], [82688, 1], [64], [64 | Radial networks and embeddings | 2 | 432 | 2.2 | 194 (43) | 0.60 (2) | 12 / 70 | 16 / 17 | memory |
| 12 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x128_8x5_nn_align1>(cutl (aten::mm)`; [[82688, 64], [64, 64], [82688, 64]] | Radial networks and embeddings | 4 | 364 | 1.9 | 335 (75) | 7.89 (28) | 51 / 76 | 24 / 25 | memory |
| 13 | `void magma_sgemmEx_kernel<float, float, float, false, false, 6, 3, 5,  (aten::mm)`; [[82688, 128], [128, 10], [82688, 10]] | Radial networks and embeddings | 2 | 356 | 1.8 | 276 (62) | 8.04 (28) | 52 / 63 | 55 / 58 | memory |
| 14 | `triton_poi_fused_add_index_select_14`; [[82688, 128], [2, 82688], [1080], [89, 128], [89, 128], [], | Neighbour list and edge prep | 2 | 342 | 1.8 | 343 (76) | 0.13 (0) | 30 / 78 | 93 / 100 | memory |
| 15 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_tn_align1>(cutl (aten::mm)`; [[82688, 10], [10, 128], [82688, 128]] | Radial networks and embeddings | 2 | 339 | 1.7 | 164 (37) | 2.17 (8) | 24 / 73 | 24 / 25 | memory |
