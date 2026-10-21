
## NaCl (216 atoms, 8192 padded edges)

| Quantity | Value | Source |
| --- | ---: | --- |
| Wall time per MD step | 3.196 ms | un-profiled window (1000 steps) |
| GPU span per step (stage events) | 3.229 ms | CUDA events around stages |
| GPU idle between stages | 0.068 ms | CUDA events |
| B graph replay (forces + 2nd half kick) | 2.872 ms | CUDA events, un-profiled |
| Force-graph replay span / kernel time | 2.937 / 2.781 ms | profiler, 10 replays |
| Kernels per replay | 264 | profiler |
| Gaps inside a replay | 153 µs (263 gaps) | profiler |
| Gaps between graph nodes (nsys) | 147 µs (237 gaps) | nsys, every node traced |
| Eager force call kernel sum | 2.769 ms (256 kernels) | profiler, eager calls |

Stages (ms per step):

| Stage | ms | % of GPU span |
| --- | ---: | ---: |
| B: forces + second half kick | 2.872 | 88.9 |
| ACP: Langevin half kick, positions, neighbour check, edge compaction | 0.150 | 4.6 |
| Neighbour-list rebuild (GPU) | 0.124 | 3.8 |
| Noise upload (every step a slice; a new batch every noise_batch) | 0.017 | 0.5 |
| GPU idle between stages | 0.068 | 2.1 |

Force call by part (profiler, eager calls: 2.769 ms of kernels; forward/backward split from ncu's per-kernel durations, boundary at kernel 143 of 256):

| Part | ms | % of force kernels | kernels | forward / backward |
| --- | ---: | ---: | ---: | --- |
| Atom-level linears (cuBLAS GEMMs and FastMACE Triton GEMMs) | 1.303 | 47.0 | 95 | 49 / 51 % |
| Radial networks and embeddings | 0.658 | 23.8 | 31 | 42 / 58 % |
| Convolution (interaction 1 reassociated sums, interaction 2 streaming OEq) | 0.425 | 15.3 | 7 | 18 / 82 % |
| Atom-level elementwise (gates, products, residuals, readouts, species tables) | 0.230 | 8.3 | 95 | 44 / 56 % |
| Symmetric contraction (cuEq) | 0.078 | 2.8 | 4 | – |
| Neighbour list and edge prep (edge geometry, radial basis, gathers, scatters) | 0.057 | 2.1 | 12 | 53 / 47 % |
| Copies and fills | 0.018 | 0.7 | 12 | 32 / 68 % |
| **All (ncu durations)** | | | | **40 / 60 %** (1.51 / 2.23 ms) |

Top 15 kernels by ncu duration (one eager force call, 3.90 ms of ncu kernel time; DRAM peak 448 GB/s, FP32 peak 28.5 TFLOPS):

| # | Kernel (op; inputs) | Part | Launches | µs | % | DRAM GB/s (% peak) | TFLOPS (% peak) | SM / memory SOL % | Occupancy achieved / theoretical % | Bound |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | `oeq_projected_backward (fastmace::streaming_conv_backward)`; [[216, 512], [8192, 16], [8192, 1280], [8192], [8192], [217] | Convolution | 1 | 490 | 12.5 | 197 (44) | 1.69 (6) | 19 / 88 | 63 / 67 | memory |
| 2 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(cutl (aten::mm)`; [[1512, 512], [512, 512], [1512, 512]] | Atom-level linears | 4 | 281 | 7.2 | 99 (22) | 11.54 (41) | 63 / 63 | 23 / 25 | compute |
| 3 | `Kernel (fastmace::folded_linear_input_grad (trit)`; "" | Radial networks and embeddings | 1 | 236 | 6.1 | 171 (38) | 5.44 (19) | 27 / 76 | 24 / 25 | memory |
| 4 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(cutl (aten::mm)`; [[648, 512], [512, 512], [648, 512]] | Atom-level linears | 6 | 204 | 5.2 | 93 (21) | 10.95 (38) | 59 / 55 | 22 / 25 | latency / occupancy |
| 5 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(cutl (aten::mm)`; [[1080, 512], [512, 512], [1080, 512]] | Atom-level linears | 4 | 200 | 5.1 | 74 (16) | 11.47 (40) | 61 / 58 | 23 / 25 | compute |
| 6 | `void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3 (aten::mm)`; [[8192, 68], [68, 1280], [8192, 1280]] | Radial networks and embeddings | 1 | 174 | 4.5 | 166 (37) | 9.15 (32) | 65 / 87 | 48 / 50 | memory |
| 7 | ` (?)`;  | Symmetric contraction | 4 | 166 | 4.3 | 184 (41) | 4.67 (16) | 27 / 78 | 57 / 77 | memory |
| 8 | `oeq_projected_forward (fastmace::streaming_conv_forward)`; [[216, 512], [8192, 16], [8192, 1280], [8192], [217], []] | Convolution | 1 | 111 | 2.8 | 400 (89) | 2.29 (8) | 18 / 91 | 46 / 50 | memory |
| 9 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>(cutl (aten::mm)`; [[216, 512], [512, 512], [216, 512]] | Atom-level linears | 4 | 80 | 2.0 | 92 (20) | 5.92 (21) | 42 / 51 | 18 / 25 | latency / occupancy |
| 10 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_tn_align1>(cutl (aten::addmm)`; [[64], [8192, 64], [64, 64], [], [], [8192, 64]] | Radial networks and embeddings | 4 | 76 | 1.9 | 101 (22) | 6.77 (24) | 41 / 51 | 23 / 25 | latency / occupancy |
| 11 | `triton_per_fused_add_div_gather_mm_mul_native_layer_norm_native_layer_`; [[8192, 64], [8192, 64], [], [8192, 1], [8192, 1], [8192, 64 | Radial networks and embeddings | 1 | 76 | 1.9 | 194 (43) | 1.28 (4) | 50 / 87 | 80 / 83 | memory |
| 12 | `void magma_sgemmEx_kernel<float, float, float, false, false, 6, 3, 5,  (aten::mm)`; [[8192, 64], [64, 64], [8192, 64]] | Radial networks and embeddings | 4 | 71 | 1.8 | 121 (27) | 3.58 (13) | 25 / 49 | 27 / 58 | latency / occupancy |
| 13 | `Kernel (fastmace::edge_outer_sum_backward (trito)`; "" | Convolution | 1 | 59 | 1.5 | 59 (13) | 1.35 (5) | 46 / 46 | 22 / 25 | latency / occupancy |
| 14 | `triton_poi_fused_mul_sum_view_8`; [[1512, 512], [1512, 512], [216, 1, 512], []] | Atom-level elementwise | 2 | 52 | 1.3 | 261 (58) | 0.06 (0) | 2 / 90 | 63 / 100 | memory |
| 15 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_nn_align1>(cutla (aten::mm)`; [[8192, 128], [128, 10], [8192, 10]] | Radial networks and embeddings | 2 | 52 | 1.3 | 160 (36) | 4.85 (17) | 28 / 72 | 26 / 33 | memory |

## NbCl5 (1944 atoms, 56320 padded edges)

| Quantity | Value | Source |
| --- | ---: | --- |
| Wall time per MD step | 21.461 ms | un-profiled window (1000 steps) |
| GPU span per step (stage events) | 21.629 ms | CUDA events around stages |
| GPU idle between stages | 0.101 ms | CUDA events |
| B graph replay (forces + 2nd half kick) | 21.122 ms | CUDA events, un-profiled |
| Force-graph replay span / kernel time | 21.202 / 21.112 ms | profiler, 10 replays |
| Kernels per replay | 230 | profiler |
| Gaps inside a replay | 103 µs (229 gaps) | profiler |
| Gaps between graph nodes (nsys) | 109 µs (215 gaps) | nsys, every node traced |
| Eager force call kernel sum | 20.942 ms (222 kernels) | profiler, eager calls |

Stages (ms per step):

| Stage | ms | % of GPU span |
| --- | ---: | ---: |
| B: forces + second half kick | 21.143 | 97.8 |
| ACP: Langevin half kick, positions, neighbour check, edge compaction | 0.182 | 0.8 |
| Neighbour-list rebuild (GPU) | 0.165 | 0.8 |
| Noise upload (every step a slice; a new batch every noise_batch) | 0.038 | 0.2 |
| GPU idle between stages | 0.101 | 0.5 |

Force call by part (profiler, eager calls: 20.942 ms of kernels; forward/backward split from ncu's per-kernel durations, boundary at kernel 128 of 222):

| Part | ms | % of force kernels | kernels | forward / backward |
| --- | ---: | ---: | ---: | --- |
| Atom-level linears (cuBLAS GEMMs and FastMACE Triton GEMMs) | 9.001 | 43.0 | 71 | 53 / 47 % |
| Radial networks and embeddings | 3.899 | 18.6 | 31 | 40 / 60 % |
| Convolution (interaction 1 reassociated sums, interaction 2 streaming OEq) | 3.560 | 17.0 | 7 | 29 / 71 % |
| Atom-level elementwise (gates, products, residuals, readouts, species tables) | 2.707 | 12.9 | 83 | 44 / 56 % |
| Symmetric contraction (cuEq) | 1.181 | 5.6 | 4 | – |
| Neighbour list and edge prep (edge geometry, radial basis, gathers, scatters) | 0.486 | 2.3 | 12 | 76 / 24 % |
| Copies and fills | 0.108 | 0.5 | 14 | 57 / 43 % |
| **All (ncu durations)** | | | | **45 / 55 %** (9.97 / 12.15 ms) |

Top 15 kernels by ncu duration (one eager force call, 23.26 ms of ncu kernel time; DRAM peak 448 GB/s, FP32 peak 28.5 TFLOPS):

| # | Kernel (op; inputs) | Part | Launches | µs | % | DRAM GB/s (% peak) | TFLOPS (% peak) | SM / memory SOL % | Occupancy achieved / theoretical % | Bound |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | `gemm (fastmace::joined_products)`; [[[1944, 512], [5832, 512], [9720, 512], [13608, 512]], [[51 | Atom-level linears | 6 | 3020 | 13.0 | 195 (43) | 10.16 (36) | 62 / 68 | 24 / 25 | memory |
| 2 | `gemm (fastmace::joined_products_input_grads)`; [[1944, 8192], [[512, 512], [512, 512], [512, 512], [512, 51 | Atom-level linears | 6 | 2362 | 10.2 | 169 (38) | 12.98 (46) | 71 / 62 | 24 / 25 | compute |
| 3 | `oeq_projected_backward (fastmace::streaming_conv_backward)`; [[1944, 512], [56320, 16], [56320, 1280], [56320], [56320],  | Convolution | 1 | 1777 | 7.6 | 372 (83) | 3.40 (12) | 38 / 84 | 66 / 67 | memory |
| 4 | `Kernel (fastmace::folded_linear_input_grad (trit)`; "" | Radial networks and embeddings | 1 | 1742 | 7.5 | 208 (46) | 5.41 (19) | 26 / 93 | 25 / 25 | memory |
| 5 | `void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3 (aten::mm)`; [[56320, 68], [68, 1280], [56320, 1280]] | Radial networks and embeddings | 1 | 1288 | 5.5 | 227 (51) | 9.06 (32) | 67 / 91 | 49 / 50 | memory |
| 6 | ` (?)`;  | Symmetric contraction | 4 | 1140 | 4.9 | 344 (77) | 6.74 (24) | 48 / 82 | 70 / 75 | memory |
| 7 | `oeq_projected_forward (fastmace::streaming_conv_forward)`; [[1944, 512], [56320, 16], [56320, 1280], [56320], [1945], [ | Convolution | 1 | 822 | 3.5 | 413 (92) | 2.24 (8) | 17 / 94 | 49 / 50 | memory |
| 8 | `triton_per_fused_add_div_gather_mm_mul_native_layer_norm_native_layer_`; [[56320, 64], [56320, 64], [], [56320, 1], [56320, 1], [5632 | Radial networks and embeddings | 1 | 756 | 3.3 | 213 (47) | 0.94 (3) | 62 / 79 | 80 / 83 | memory |
| 9 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(cutl (aten::mm)`; [[5832, 512], [512, 512], [5832, 512]] | Atom-level linears | 2 | 484 | 2.1 | 71 (16) | 12.77 (45) | 69 / 63 | 23 / 25 | compute |
| 10 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(cutl (aten::mm)`; [[1944, 512], [512, 512]] | Atom-level linears | 4 | 353 | 1.5 | 73 (16) | 11.87 (42) | 67 / 61 | 24 / 25 | compute |
| 11 | `Kernel (fastmace::edge_outer_sum_backward (trito)`; "" | Convolution | 1 | 318 | 1.4 | 139 (31) | 1.84 (6) | 63 / 63 | 24 / 25 | compute |
| 12 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(cutl (aten::mm)`; [[9720, 384], [384, 512], [9720, 512]] | Atom-level linears | 1 | 307 | 1.3 | 102 (23) | 12.49 (44) | 67 / 73 | 24 / 25 | memory |
| 13 | `void magma_sgemmEx_kernel<float, float, float, false, false, 6, 3, 5,  (aten::mm)`; [[56320, 64], [64, 64], [56320, 64]] | Radial networks and embeddings | 4 | 305 | 1.3 | 214 (48) | 6.16 (22) | 46 / 76 | 53 / 58 | memory |
| 14 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(cutl (aten::mm)`; [[9720, 512], [512, 384], [9720, 384]] | Atom-level linears | 1 | 304 | 1.3 | 103 (23) | 12.61 (44) | 68 / 70 | 24 / 25 | memory |
| 15 | `triton_poi_fused_joined_products_mul_sigmoid_silu_slice_unsqueeze_view`; [[13608, 512], [1944, 2048], [13608, 512], []] | Atom-level elementwise | 2 | 301 | 1.3 | 277 (62) | 0.23 (1) | 12 / 92 | 83 / 100 | memory |

## NaNbO3 (1080 atoms, 82688 padded edges)

| Quantity | Value | Source |
| --- | ---: | --- |
| Wall time per MD step | 18.810 ms | un-profiled window (1000 steps) |
| GPU span per step (stage events) | 18.863 ms | CUDA events around stages |
| GPU idle between stages | 0.078 ms | CUDA events |
| B graph replay (forces + 2nd half kick) | 18.365 ms | CUDA events, un-profiled |
| Force-graph replay span / kernel time | 18.346 / 18.233 ms | profiler, 10 replays |
| Kernels per replay | 241 | profiler |
| Gaps inside a replay | 123 µs (240 gaps) | profiler |
| Gaps between graph nodes (nsys) | 124 µs (224 gaps) | nsys, every node traced |
| Eager force call kernel sum | 18.194 ms (233 kernels) | profiler, eager calls |

Stages (ms per step):

| Stage | ms | % of GPU span |
| --- | ---: | ---: |
| B: forces + second half kick | 18.365 | 97.4 |
| Neighbour-list rebuild (GPU) | 0.213 | 1.1 |
| ACP: Langevin half kick, positions, neighbour check, edge compaction | 0.178 | 0.9 |
| Noise upload (every step a slice; a new batch every noise_batch) | 0.029 | 0.2 |
| GPU idle between stages | 0.077 | 0.4 |

Force call by part (profiler, eager calls: 18.194 ms of kernels; forward/backward split from ncu's per-kernel durations, boundary at kernel 135 of 233):

| Part | ms | % of force kernels | kernels | forward / backward |
| --- | ---: | ---: | ---: | --- |
| Radial networks and embeddings | 5.999 | 33.0 | 31 | 52 / 48 % |
| Atom-level linears (cuBLAS GEMMs and FastMACE Triton GEMMs) | 4.900 | 26.9 | 75 | 50 / 50 % |
| Convolution (interaction 1 reassociated sums, interaction 2 streaming OEq) | 4.638 | 25.5 | 7 | 28 / 72 % |
| Atom-level elementwise (gates, products, residuals, readouts, species tables) | 1.185 | 6.5 | 86 | 40 / 60 % |
| Neighbour list and edge prep (edge geometry, radial basis, gathers, scatters) | 0.768 | 4.2 | 12 | 77 / 23 % |
| Symmetric contraction (cuEq) | 0.571 | 3.1 | 4 | 0 / 100 % |
| Copies and fills | 0.133 | 0.7 | 18 | 45 / 55 % |
| **All (ncu durations)** | | | | **45 / 55 %** (8.47 / 10.32 ms) |

Top 15 kernels by ncu duration (one eager force call, 19.13 ms of ncu kernel time; DRAM peak 448 GB/s, FP32 peak 28.5 TFLOPS):

| # | Kernel (op; inputs) | Part | Launches | µs | % | DRAM GB/s (% peak) | TFLOPS (% peak) | SM / memory SOL % | Occupancy achieved / theoretical % | Bound |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | `oeq_projected_backward (fastmace::streaming_conv_backward)`; [[1080, 512], [82688, 16], [82688, 1280], [82688], [82688],  | Convolution | 1 | 2722 | 14.2 | 368 (82) | 3.43 (12) | 38 / 83 | 66 / 67 | memory |
| 2 | `void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3 (aten::mm)`; [[82688, 68], [68, 1280], [82688, 1280]] | Radial networks and embeddings | 1 | 1991 | 10.4 | 241 (54) | 9.06 (32) | 67 / 92 | 49 / 50 | memory |
| 3 | `Kernel (fastmace::folded_linear_input_grad (trit)`; "" | Radial networks and embeddings | 1 | 1193 | 6.2 | 394 (88) | 12.21 (43) | 69 / 89 | 25 / 25 | memory |
| 4 | `oeq_projected_forward (fastmace::streaming_conv_forward)`; [[1080, 512], [82688, 16], [82688, 1280], [82688], [1081], [ | Convolution | 1 | 1183 | 6.2 | 414 (92) | 2.34 (8) | 17 / 94 | 49 / 50 | memory |
| 5 | `gemm (fastmace::joined_products_input_grads)`; [[1080, 8192], [[512, 512], [512, 512], [512, 512], [512, 51 | Atom-level linears | 4 | 1089 | 5.7 | 80 (18) | 12.57 (44) | 67 / 58 | 24 / 25 | compute |
| 6 | `gemm (fastmace::joined_products)`; [[[1080, 512], [3240, 512], [5400, 512], [7560, 512]], [[512 | Atom-level linears | 4 | 1056 | 5.5 | 67 (15) | 12.97 (46) | 69 / 60 | 24 / 25 | compute |
| 7 | `triton_per_fused_add_div_gather_mm_mul_native_layer_norm_native_layer_`; [[82688, 64], [82688, 64], [], [82688, 1], [82688, 1], [8268 | Radial networks and embeddings | 1 | 572 | 3.0 | 366 (82) | 1.92 (7) | 79 / 83 | 78 / 83 | memory |
| 8 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(cutl (aten::mm)`; [[3240, 512], [512, 512]] | Atom-level linears | 4 | 547 | 2.9 | 83 (18) | 12.53 (44) | 67 / 64 | 22 / 25 | compute |
| 9 | `Kernel (fastmace::edge_outer_sum_backward (trito)`; "" | Convolution | 1 | 467 | 2.4 | 151 (34) | 1.93 (7) | 66 / 66 | 24 / 25 | compute |
| 10 | `void magma_sgemmEx_kernel<float, float, float, true, false, 6, 4, 6, 3 (aten::addmm)`; [[64], [82688, 64], [64, 64], [], [], [82688, 64]] | Radial networks and embeddings | 4 | 416 | 2.2 | 325 (72) | 6.96 (24) | 51 / 88 | 47 / 50 | memory |
| 11 | `triton_per_fused_native_layer_norm_native_layer_norm_backward_silu_bac`; [[82688, 64], [82688, 1], [82688, 1], [64], [64], [82688, 64 | Radial networks and embeddings | 2 | 404 | 2.1 | 272 (61) | 0.64 (2) | 12 / 95 | 70 / 75 | memory |
| 12 | `void magma_sgemmEx_kernel<float, float, float, false, false, 6, 3, 5,  (aten::mm)`; [[82688, 128], [128, 10], [82688, 10]] | Radial networks and embeddings | 2 | 368 | 1.9 | 279 (62) | 7.78 (27) | 50 / 63 | 55 / 58 | memory |
| 13 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_64x128_8x5_nn_align1>(cutl (aten::mm)`; [[82688, 64], [64, 64], [82688, 64]] | Radial networks and embeddings | 4 | 365 | 1.9 | 328 (73) | 7.87 (28) | 51 / 74 | 24 / 25 | memory |
| 14 | `triton_poi_fused_add_index_select_14`; [[82688, 128], [2, 82688], [1080], [89, 128], [89, 128], [], | Neighbour list and edge prep | 2 | 361 | 1.9 | 346 (77) | 0.12 (0) | 28 / 79 | 93 / 100 | memory |
| 15 | ` (?)`;  | Symmetric contraction | 3 | 343 | 1.8 | 379 (85) | 4.75 (17) | 33 / 87 | 74 / 83 | memory |
