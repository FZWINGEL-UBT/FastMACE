# fp32 profile tables: RTX 5060 Ti, local (.)

GPU NVIDIA GeForce RTX 5060 Ti (sm_120, 36 SMs, max clock 3090.0 MHz, FP32 peak 28.5 TFLOPS); FastMACE 312942df5850f7cf9be7dc96438eecef77662eff; torch 2.14.1+cu130

## Step stages

| Stage | NaCl | NaNbO3 | NbCl5 |
| --- | ---: | ---: | ---: |
| Wall time, uninstrumented (ms/step) | 2.716 | 14.358 | 18.675 |
| GPU span, instrumented (ms/step) | 2.761 | 14.290 | 19.287 |
| Noise upload (every step a slice; a new batch every noise_batch) | 0.018 | 0.028 | 0.052 |
| ACP: Langevin half kick, positions, neighbour check, edge compaction | 0.050 | 0.052 | 0.046 |
| B: forces + second half kick | 2.612 | 14.162 | 19.080 |
| Neighbour-list rebuild (GPU) | 0.358 | 1.228 | 1.244 |
| C: edge compaction after a rebuild | – | 0.000 | 0.056 |
| GPU idle between stages | -0.276 | -1.180 | -1.191 |
| Rebuilds per 1000 steps (instrumented window) | 161 | 214 | 160 |
| Padded edges | 8192 | 82688 | 58880 |
| Atoms | 216 | 1080 | 1944 |
| B replay: kernels | 233 | 317 | 275 |
| B replay: kernel ms / span ms | 2.471 / 2.616 | 16.681 / 13.991 | 20.813 / 18.793 |
| Eager force call: kernel ms (kernels) | 2.469 (225) | 16.415 (309) | 20.489 (267) |

## Force call by part (compiled kernels, refined)

| Part (compiled eager force call, refined): ms (share, kernels) | NaCl | NaNbO3 | NbCl5 |
| --- | ---: | ---: | ---: |
| Atom GEMMs: segment_gemm (joined products, Triton) | – | 2.238 (14%, 16) | 5.376 (26%, 20) |
| Atom GEMMs: cuBLAS (incl. split-K reduce, memset) | 1.339 (54%, 95) | 2.991 (18%, 67) | 4.073 (20%, 59) |
| Int. 2: convolution backward, chunked (oeq_projected_backward) | 0.277 (11%, 4) | 3.015 (18%, 57) | 2.530 (12%, 42) |
| Atom-level elementwise (gates, products, residuals, readouts) | 0.234 (9%, 91) | 1.146 (7%, 75) | 2.512 (12%, 71) |
| Int. 2: radial last layer, input gradient (Triton) | 0.117 (5%, 1) | 3.090 (19%, 54) | 2.373 (12%, 39) |
| Int. 2: radial last layer + convolution forward (fused_folded_forward) | 0.244 (10%, 2) | 1.883 (11%, 2) | 1.389 (7%, 2) |
| Symmetric contraction (cuEq) | 0.098 (4%, 4) | 0.621 (4%, 4) | 1.132 (6%, 4) |
| Int. 1: reassociated edge sums (Triton) | 0.063 (3%, 3) | 0.588 (4%, 3) | 0.545 (3%, 3) |
| Radial tables (hidden radial chain, forward + backward) | 0.059 (2%, 5) | 0.540 (3%, 5) | 0.341 (2%, 5) |
| Copies and fills | 0.015 (1%, 11) | 0.141 (1%, 17) | 0.116 (1%, 13) |
| Edge elementwise (geometry, basis, gathers, scatters) | 0.024 (1%, 9) | 0.162 (1%, 9) | 0.102 (0%, 9) |
| Total eager kernel time | 2.468 | 16.415 | 20.489 |

## Force call by step_breakdown category

| Part (compiled eager force call): ms (share, kernels) | NaCl | NaNbO3 | NbCl5 |
| --- | ---: | ---: | ---: |
| Other | – | 2.078 (13%, 8) | 4.883 (24%, 12) |
| Atom-level linear layers (GEMM) | 1.338 (54%, 95) | 2.991 (18%, 67) | 4.073 (20%, 59) |
| Edge elementwise (geometry, radial basis, gathers, scatters) | 0.584 (24%, 16) | 5.435 (33%, 69) | 4.040 (20%, 54) |
| Atom-level elementwise (gates, products, residuals, readouts) | 0.253 (10%, 95) | 1.471 (9%, 87) | 3.326 (16%, 83) |
| Radial networks: folded last layer, input gradient | 0.117 (5%, 1) | 3.090 (19%, 54) | 2.373 (12%, 39) |
| Atom-level cuEq tensor products (symmetric contraction) | 0.098 (4%, 4) | 0.622 (4%, 4) | 1.133 (6%, 4) |
| Interaction 1: reassociated edge sums | 0.063 (3%, 3) | 0.588 (4%, 3) | 0.545 (3%, 3) |
| Copies and fills | 0.015 (1%, 11) | 0.141 (1%, 17) | 0.116 (1%, 13) |

## Top operators, NaCl

| µs/call | launches | category | op | kernel | grid | est. occ. % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 294.5 | 4 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [96, 1, 3] | 25.0 |
| 260.5 | 1 | Edge elementwise (geometry, radial basi… | fastmace::streaming_conv_folded_backward | oeq_projected_backward | [2048, 1, 1] | 67.0 |
| 241.2 | 1 | Edge elementwise (geometry, radial basi… | fastmace::fused_folded_forward | fused_folded_forward | [9, 4, 2] | 0.0 |
| 203.3 | 6 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [44, 1, 4] | 25.0 |
| 202.6 | 4 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [68, 1, 3] | 25.0 |
| 116.7 | 1 | Radial networks: folded last layer, inp… | fastmace::folded_linear_input_grad (tri… | Kernel | [64, 1, 1] | 30.0 |
| 77.6 | 4 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_a… | [28, 1, 3] | 19.0 |
| 57.0 | 1 | Atom-level cuEq tensor products (symmet… | cuequivariance::uniform_1d | segmented_polynomial_bwd_algo1_1 | [216, 1, 1] | 13.0 |
| 46.4 | 1 | Interaction 1: reassociated edge sums | fastmace::edge_outer_sum_backward (trit… | Kernel | [256, 1, 1] | 25.0 |
| 42.2 | 1 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [68, 1, 3] | 25.0 |
| 41.9 | 1 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [68, 1, 4] | 25.0 |
| 35.8 | 1 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [16, 1, 8] | 25.0 |
| 35.4 | 1 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [32, 2, 1] | 15.0 |
| 35.0 | 1 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [48, 1, 3] | 25.0 |
| 34.7 | 1 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [96, 1, 1] | 22.0 |
| 27.2 | 1 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_a… | [48, 1, 3] | 25.0 |
| 26.9 | 2 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [44, 1, 1] | 10.0 |
| 23.5 | 1 | Atom-level cuEq tensor products (symmet… | cuequivariance::uniform_1d | segmented_polynomial_algo1_2 | [216, 1, 1] | 25.0 |
| 23.3 | 1 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [32, 2, 1] | 15.0 |
| 22.2 | 2 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [11, 1, 3] | 8.0 |

## GEMMs, NaCl

| category | op | µs/call | calls | TFLOPS | of peak | shapes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Atom-level linear layers (GEMM) | aten::mm | 296.3 | 4.0 | 10.70 | 0.38 | [[1512, 512], [512, 512], [1512, 512]] |
| Atom-level linear layers (GEMM) | aten::mm | 206.6 | 6.0 | 9.87 | 0.35 | [[648, 512], [512, 512], [648, 512]] |
| Atom-level linear layers (GEMM) | aten::mm | 204.5 | 4.0 | 11.07 | 0.39 | [[1080, 512], [512, 512], [1080, 512]] |
| Atom-level linear layers (GEMM) | aten::mm | 79.6 | 4.0 | 5.69 | 0.20 | [[216, 512], [512, 512], [216, 512]] |
| Atom-level linear layers (GEMM) | aten::mm | 42.7 | 1.0 | 9.95 | 0.35 | [[1080, 512], [512, 384], [1080, 384]] |
| Atom-level linear layers (GEMM) | aten::mm | 42.6 | 1.0 | 9.98 | 0.35 | [[1080, 384], [384, 512], [1080, 512]] |
| Atom-level linear layers (GEMM) | aten::mm | 38.8 | 1.0 | 8.76 | 0.31 | [[216, 2048], [2048, 384], [216, 384]] |
| Atom-level linear layers (GEMM) | aten::mm | 35.4 | 1.0 | 9.60 | 0.34 | [[216, 384], [384, 2048], [216, 2048]] |
| Atom-level linear layers (GEMM) | aten::mm | 35.3 | 1.0 | 11.22 | 0.39 | [[1512, 512], [512, 256], [1512, 256]] |
| Atom-level linear layers (GEMM) | aten::mm | 34.7 | 1.0 | 11.43 | 0.40 | [[1512, 256], [256, 512], [1512, 512]] |
| Atom-level linear layers (GEMM) | aten::mm | 28.1 | 2.0 | 6.05 | 0.21 | [[648, 512], [512, 128], [648, 128]] |
| Atom-level linear layers (GEMM) | aten::mm | 27.5 | 1.0 | 7.20 | 0.25 | [[1512, 512], [512, 128], [1512, 128]] |

## Top operators, NaNbO3

| µs/call | launches | category | op | kernel | grid | est. occ. % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3089.9 | 54 | Radial networks: folded last layer, inp… | fastmace::folded_linear_input_grad (tri… | Kernel | [24, 1, 1] | 6.0 |
| 2853.5 | 54 | Edge elementwise (geometry, radial basi… | fastmace::streaming_conv_folded_backward | oeq_projected_backward | [384, 1, 1] | 67.0 |
| 1880.2 | 1 | Edge elementwise (geometry, radial basi… | fastmace::fused_folded_forward | fused_folded_forward | [9, 4, 2] | 0.0 |
| 1057.5 | 4 | Other | fastmace::joined_products_input_grads | gemm | [85, 8, 1] | 25.0 |
| 1021.0 | 4 | Other | fastmace::joined_products | gemm | [85, 8, 1] | 25.0 |
| 543.9 | 4 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [204, 1, 1] | 25.0 |
| 416.6 | 1 | Interaction 1: reassociated edge sums | fastmace::edge_outer_sum_backward (trit… | Kernel | [2584, 1, 1] | 25.0 |
| 261.4 | 2 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [204, 1, 1] | 25.0 |
| 210.4 | 1 | Atom-level cuEq tensor products (symmet… | cuequivariance::uniform_1d | segmented_polynomial_bwd_algo1_4 | [1080, 1, 1] | 58.0 |
| 198.0 | 4 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [68, 1, 3] | 25.0 |
| 194.0 | 1 | Atom-level cuEq tensor products (symmet… | cuequivariance::uniform_1d | segmented_polynomial_bwd_algo6_4 | [1080, 1, 1] | 100.0 |
| 193.7 | 1 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [238, 1, 1] | 25.0 |
| 174.1 | 1 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [340, 1, 1] | 25.0 |
| 171.4 | 1 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [340, 1, 1] | 25.0 |
| 158.0 | 1 | Atom-level elementwise (gates, products… | triton_red_fused__unsafe_view_add_clone… | triton_red_fused__unsafe_view_add_clone_div_mul_neg_select_… | [1080, 1, 1] | 100.0 |
| 152.0 | 2 | Atom-level elementwise (gates, products… | triton_poi_fused_mul_sigmoid_slice_unsq… | triton_poi_fused_mul_sigmoid_slice_unsqueeze_view_11 | [3780, 1, 1] | 100.0 |
| 151.5 | 1 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [476, 1, 1] | 25.0 |
| 150.7 | 2 | Atom-level elementwise (gates, products… | triton_poi_fused_mul_sum_view_6 | triton_poi_fused_mul_sum_view_6 | [540, 1, 1] | 83.0 |
| 144.0 | 1 | Edge elementwise (geometry, radial basi… | fastmace::radial_table_forward | forward | [2584, 1, 1] | 17.0 |
| 143.1 | 1 | Edge elementwise (geometry, radial basi… | fastmace::radial_table_backward | backward | [2584, 1, 1] | 25.0 |

## GEMMs, NaNbO3

| category | op | µs/call | calls | TFLOPS | of peak | shapes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Atom-level linear layers (GEMM) | aten::mm | 543.9 | 4.0 | 12.49 | 0.44 | [[3240, 512], [512, 512]] |
| Atom-level linear layers (GEMM) | aten::mm | 261.4 | 2.0 | 13.00 | 0.46 | [[3240, 512], [512, 512], [3240, 512]] |
| Atom-level linear layers (GEMM) | aten::mm | 200.2 | 4.0 | 11.31 | 0.40 | [[1080, 512], [512, 512]] |
| Atom-level linear layers (GEMM) | aten::mm | 193.7 | 1.0 | 10.23 | 0.36 | [[7560, 512], [512, 256], [7560, 256]] |
| Atom-level linear layers (GEMM) | aten::mm | 174.1 | 1.0 | 12.19 | 0.43 | [[5400, 512], [512, 384], [5400, 384]] |
| Atom-level linear layers (GEMM) | aten::mm | 171.4 | 1.0 | 12.39 | 0.43 | [[5400, 384], [384, 512], [5400, 512]] |
| Atom-level linear layers (GEMM) | aten::mm | 151.5 | 1.0 | 13.08 | 0.46 | [[7560, 256], [256, 512], [7560, 512]] |
| Atom-level linear layers (GEMM) | aten::mm | 142.1 | 1.0 | 11.95 | 0.42 | [[1080, 384], [384, 2048], [1080, 2048]] |
| Atom-level linear layers (GEMM) | aten::mm | 134.2 | 1.0 | 12.66 | 0.44 | [[1080, 2048], [2048, 384], [1080, 384]] |
| Atom-level linear layers (GEMM) | aten::mm | 92.6 | 1.0 | 12.23 | 0.43 | [[1080, 2048], [2048, 256], [1080, 256]] |
| Atom-level linear layers (GEMM) | aten::mm | 89.8 | 1.0 | 11.04 | 0.39 | [[7560, 512], [512, 128], [7560, 128]] |
| Atom-level linear layers (GEMM) | aten::mm | 88.0 | 1.0 | 12.87 | 0.45 | [[1080, 256], [256, 2048], [1080, 2048]] |

## Top operators, NbCl5

| µs/call | launches | category | op | kernel | grid | est. occ. % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2483.9 | 6 | Other | fastmace::joined_products | gemm | [92, 8, 1] | 25.0 |
| 2399.5 | 6 | Other | fastmace::joined_products_input_grads | gemm | [92, 8, 1] | 25.0 |
| 2372.8 | 39 | Radial networks: folded last layer, inp… | fastmace::folded_linear_input_grad (tri… | Kernel | [24, 1, 1] | 5.9 |
| 2212.1 | 39 | Edge elementwise (geometry, radial basi… | fastmace::streaming_conv_folded_backward | oeq_projected_backward | [384, 1, 1] | 66.6 |
| 1385.9 | 1 | Edge elementwise (geometry, radial basi… | fastmace::fused_folded_forward | fused_folded_forward | [9, 4, 2] | 0.0 |
| 462.3 | 2 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [368, 1, 1] | 25.0 |
| 428.7 | 1 | Atom-level cuEq tensor products (symmet… | cuequivariance::uniform_1d | segmented_polynomial_bwd_algo2_2 | [1944, 1, 1] | 75.0 |
| 375.4 | 1 | Interaction 1: reassociated edge sums | fastmace::edge_outer_sum_backward (trit… | Kernel | [1840, 1, 1] | 25.0 |
| 356.8 | 4 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [124, 1, 3] | 25.0 |
| 341.8 | 1 | Atom-level cuEq tensor products (symmet… | cuequivariance::uniform_1d | segmented_polynomial_bwd_algo6_4 | [1944, 1, 1] | 100.0 |
| 314.8 | 1 | Atom-level elementwise (gates, products… | triton_red_fused__unsafe_view_add_clone… | triton_red_fused__unsafe_view_add_clone_div_mul_neg_select_… | [1944, 1, 1] | 100.0 |
| 307.2 | 2 | Atom-level elementwise (gates, products… | triton_poi_fused_joined_products_mul_si… | triton_poi_fused_joined_products_mul_sigmoid_silu_slice_uns… | [6804, 1, 1] | 100.0 |
| 298.1 | 2 | Atom-level elementwise (gates, products… | triton_poi_fused_mul_sum_view_6 | triton_poi_fused_mul_sum_view_6 | [972, 1, 1] | 83.0 |
| 288.6 | 1 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [608, 1, 1] | 25.0 |
| 278.9 | 1 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [608, 1, 1] | 25.0 |
| 277.0 | 2 | Atom-level elementwise (gates, products… | triton_poi_fused_mul_sigmoid_slice_unsq… | triton_poi_fused_mul_sigmoid_slice_unsqueeze_view_11 | [13608, 1, 1] | 100.0 |
| 275.7 | 1 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [426, 1, 1] | 25.0 |
| 241.6 | 1 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_a… | [248, 2, 1] | 25.0 |
| 240.0 | 2 | Atom-level elementwise (gates, products… | triton_poi_fused_mul_sum_view_5 | triton_poi_fused_mul_sum_view_5 | [972, 1, 1] | 83.0 |
| 235.5 | 1 | Atom-level linear layers (GEMM) | aten::mm | void cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nn_… | [428, 1, 1] | 17.0 |

## GEMMs, NbCl5

| category | op | µs/call | calls | TFLOPS | of peak | shapes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Atom-level linear layers (GEMM) | aten::mm | 462.3 | 2.0 | 13.23 | 0.46 | [[5832, 512], [512, 512], [5832, 512]] |
| Atom-level linear layers (GEMM) | aten::mm | 359.4 | 4.0 | 11.34 | 0.40 | [[1944, 512], [512, 512]] |
| Atom-level linear layers (GEMM) | aten::mm | 288.6 | 1.0 | 13.24 | 0.47 | [[9720, 384], [384, 512], [9720, 512]] |
| Atom-level linear layers (GEMM) | aten::mm | 278.9 | 1.0 | 13.71 | 0.48 | [[9720, 512], [512, 384], [9720, 384]] |
| Atom-level linear layers (GEMM) | aten::mm | 275.7 | 1.0 | 12.94 | 0.45 | [[13608, 512], [512, 256], [13608, 256]] |
| Atom-level linear layers (GEMM) | aten::mm | 241.6 | 1.0 | 12.65 | 0.44 | [[1944, 384], [384, 2048], [1944, 2048]] |
| Atom-level linear layers (GEMM) | aten::mm | 235.9 | 1.0 | 12.96 | 0.46 | [[1944, 2048], [2048, 384], [1944, 384]] |
| Atom-level linear layers (GEMM) | aten::mm | 235.5 | 1.0 | 15.14 | 0.53 | [[13608, 256], [256, 512], [13608, 512]] |
| Atom-level linear layers (GEMM) | aten::mm | 200.2 | 1.0 | 8.91 | 0.31 | [[13608, 512], [512, 128], [13608, 128]] |
| Atom-level linear layers (GEMM) | aten::mm | 164.1 | 1.0 | 12.42 | 0.44 | [[1944, 2048], [2048, 256], [1944, 256]] |
| Atom-level linear layers (GEMM) | aten::mm | 162.4 | 1.0 | 12.55 | 0.44 | [[1944, 256], [256, 2048], [1944, 2048]] |
| Atom-level linear layers (GEMM) | aten::mm | 161.7 | 1.0 | 7.88 | 0.28 | [[9720, 512], [512, 128], [9720, 128]] |

## Nsight Compute

| Part (ncu, one eager call): DRAM, bandwidth, occupancy | NaCl | NaNbO3 | NbCl5 |
| --- | ---: | ---: | ---: |
| Other | – | 221 MB, 97 GB/s, occ 24 % | 1388 MB, 206 GB/s, occ 25 % |
| Atom-level linear layers (GEMM) | 119 MB, 82 GB/s, occ 21 % | 313 MB, 99 GB/s, occ 23 % | 566 MB, 104 GB/s, occ 27 % |
| Unattributed (not aligned with the profiler) | – | 603 MB, 157 GB/s, occ 32 % | 805 MB, 155 GB/s, occ 28 % |
| Atom-level elementwise (gates, products, residuals, readouts) | 124 MB, 164 GB/s, occ 62 % | 552 MB, 220 GB/s, occ 75 % | 1166 MB, 231 GB/s, occ 80 % |
| Edge elementwise (geometry, radial basis, gathers, scatters) | 185 MB, 170 GB/s, occ 44 % | 1187 MB, 174 GB/s, occ 33 % | 660 MB, 158 GB/s, occ 28 % |
| Atom-level cuEq tensor products (symmetric contraction) | 32 MB, 163 GB/s, occ 46 % | 215 MB, 198 GB/s, occ 74 % | 473 MB, 179 GB/s, occ 80 % |
| Interaction 1: reassociated edge sums | 6 MB, 77 GB/s, occ 27 % | 173 MB, 176 GB/s, occ 33 % | 92 MB, 146 GB/s, occ 33 % |
| Radial networks: folded last layer, input gradient | 46 MB, 209 GB/s, occ 29 % | 290 MB, 151 GB/s, occ 8 % | 70 MB, 145 GB/s, occ 8 % |
| Copies and fills | 2 MB, 65 GB/s, occ 43 % | 45 MB, 187 GB/s, occ 85 % | 25 MB, 168 GB/s, occ 82 % |

## Whole steps under Nsight Systems

**NaCl** (300 steps, 42 rebuilds)

| NVTX range (GPU projection) | instances | ms per step | µs per instance |
| --- | ---: | ---: | ---: |
| :md step | 300 | 3.185 | 3184.9 |
| :stage B | 305 | 2.604 | 2561.6 |
| :neighbour-list rebuild | 42 | 0.388 | 2768.8 |
| :stage ACP | 300 | 0.047 | 47.5 |
| CCCL:cub::DeviceRadixSort | 42 | 0.033 | 236.9 |
| :noise upload | 300 | 0.005 | 5.5 |

**NaNbO3** (300 steps, 57 rebuilds)

| NVTX range (GPU projection) | instances | ms per step | µs per instance |
| --- | ---: | ---: | ---: |
| :md step | 300 | 16.977 | 16976.8 |
| :stage B | 300 | 14.193 | 14193.2 |
| :neighbour-list rebuild | 57 | 1.085 | 5709.5 |
| CCCL:cub::DeviceRadixSort | 57 | 0.161 | 846.1 |
| :stage ACP | 300 | 0.048 | 48.5 |
| :noise upload | 300 | 0.013 | 12.7 |

**NbCl5** (300 steps, 41 rebuilds)

| NVTX range (GPU projection) | instances | ms per step | µs per instance |
| --- | ---: | ---: | ---: |
| :md step | 300 | 20.979 | 20979.0 |
| :stage B | 300 | 18.394 | 18393.5 |
| :neighbour-list rebuild | 41 | 1.040 | 7613.0 |
| CCCL:cub::DeviceRadixSort | 41 | 0.179 | 1310.4 |
| :stage ACP | 300 | 0.043 | 42.9 |
| :noise upload | 300 | 0.019 | 18.8 |

## Model parts, eager tier

| Module (eager tier, inclusive; depth ≤ 2): ms (share, kernels) | NaCl | NaNbO3 | NbCl5 |
| --- | ---: | ---: | ---: |
| bwd <root> | 1.907 (57%, 382) | 14.677 (69%, 473) | 21.177 (70%, 434) |
| fwd <root> | 1.390 (42%, 309) | 6.546 (31%, 303) | 9.020 (30%, 298) |
| fwd atomic_energies_fn | 0.003 (0%, 2) | 0.005 (0%, 2) | 0.006 (0%, 1) |
| bwd interactions | 1.645 (49%, 240) | 13.949 (66%, 334) | 19.990 (66%, 295) |
| fwd interactions | 1.118 (34%, 151) | 5.909 (28%, 145) | 8.055 (27%, 141) |
| bwd interactions.0 | 0.556 (17%, 86) | 2.818 (13%, 78) | 5.293 (18%, 76) |
| fwd interactions.0 | 0.419 (13%, 72) | 1.791 (8%, 70) | 2.924 (10%, 68) |
| bwd interactions.1 | 1.089 (33%, 154) | 11.131 (52%, 256) | 14.697 (49%, 219) |
| fwd interactions.1 | 0.699 (21%, 79) | 4.119 (19%, 75) | 5.131 (17%, 73) |
| bwd pair_repulsion_fn | 0.075 (2%, 58) | 0.092 (0%, 58) | 0.081 (0%, 58) |
| fwd pair_repulsion_fn | 0.082 (2%, 59) | 0.095 (0%, 59) | 0.087 (0%, 59) |
| bwd products | 0.098 (3%, 7) | 0.444 (2%, 4) | 0.875 (3%, 4) |
| fwd products | 0.070 (2%, 19) | 0.300 (1%, 19) | 0.578 (2%, 19) |
| bwd products.0 | 0.073 (2%, 3) | 0.246 (1%, 2) | 0.452 (2%, 2) |
| fwd products.0 | 0.046 (1%, 9) | 0.189 (1%, 9) | 0.361 (1%, 9) |
| bwd products.1 | 0.025 (1%, 4) | 0.198 (1%, 2) | 0.422 (1%, 2) |
| fwd products.1 | 0.024 (1%, 10) | 0.111 (1%, 10) | 0.216 (1%, 10) |
| bwd readouts | 0.022 (1%, 13) | 0.058 (0%, 13) | 0.128 (0%, 13) |
| fwd readouts | 0.035 (1%, 16) | 0.065 (0%, 16) | 0.097 (0%, 16) |
| bwd readouts.0 | 0.014 (0%, 6) | 0.047 (0%, 6) | 0.114 (0%, 6) |
| fwd readouts.0 | 0.020 (1%, 6) | 0.049 (0%, 6) | 0.072 (0%, 6) |
| bwd readouts.1 | 0.008 (0%, 7) | 0.011 (0%, 7) | 0.014 (0%, 7) |
| fwd readouts.1 | 0.015 (0%, 10) | 0.017 (0%, 10) | 0.025 (0%, 10) |
| bwd scale_shift | 0.001 (0%, 1) | 0.001 (0%, 1) | 0.001 (0%, 1) |
| fwd scale_shift | 0.005 (0%, 4) | 0.006 (0%, 4) | 0.022 (0%, 4) |
| bwd spherical_harmonics | 0.066 (2%, 63) | 0.133 (1%, 63) | 0.102 (0%, 63) |
| fwd spherical_harmonics | 0.027 (1%, 24) | 0.077 (0%, 24) | 0.057 (0%, 24) |
| eager kernel ms without hooks / with hooks / outside every range | 3.282 / 3.326 / 0.027 | 21.191 / 21.258 / 0.032 | 30.269 / 30.239 / 0.039 |

## Eager ranges, NaCl

| Range (exclusive) | ms/call | kernels | share |
| --- | ---: | ---: | ---: |
| bwd interactions.1.gated_rows | 0.561 | 88 | 16.9% |
| fwd interactions.1.gated_rows | 0.399 | 35 | 12.0% |
| bwd interactions.1 | 0.399 | 13 | 12.0% |
| bwd interactions.0.gated_rows | 0.398 | 56 | 11.9% |
| fwd interactions.0.gated_rows | 0.299 | 24 | 9.0% |
| fwd interactions.1 | 0.223 | 21 | 6.7% |
| bwd interactions.0.first_reassociation | 0.117 | 20 | 3.5% |
| fwd pair_repulsion_fn | 0.082 | 59 | 2.5% |
| bwd interactions.1.radial_table | 0.080 | 38 | 2.4% |
| bwd pair_repulsion_fn | 0.075 | 58 | 2.2% |
| fwd interactions.0.first_reassociation | 0.072 | 29 | 2.2% |
| bwd products.0.symmetric_contractions.f.m | 0.069 | 2 | 2.1% |
| bwd spherical_harmonics | 0.066 | 63 | 2.0% |
| fwd <root> | 0.050 | 34 | 1.5% |
| fwd interactions.1.radial_table | 0.034 | 8 | 1.0% |
| bwd interactions.1.linear_up.folded | 0.031 | 8 | 0.9% |
| bwd interactions.0 | 0.030 | 9 | 0.9% |
| fwd products.0.symmetric_contractions.f.m | 0.027 | 1 | 0.8% |
| fwd spherical_harmonics | 0.027 | 24 | 0.8% |
| (no range) | 0.027 | 21 | 0.8% |
| fwd interactions.0.radial_table | 0.025 | 8 | 0.8% |
| fwd interactions.1.linear_up.folded | 0.025 | 6 | 0.8% |
| bwd products.1.symmetric_contractions.f.m | 0.020 | 3 | 0.6% |
| fwd interactions.0 | 0.013 | 7 | 0.4% |
| fwd readouts.0.linear.folded | 0.012 | 3 | 0.4% |

## Eager ranges, NaNbO3

| Range (exclusive) | ms/call | kernels | share |
| --- | ---: | ---: | ---: |
| bwd interactions.1 | 6.622 | 119 | 31.1% |
| bwd interactions.1.gated_rows | 3.945 | 80 | 18.6% |
| fwd interactions.1 | 2.008 | 21 | 9.4% |
| fwd interactions.1.gated_rows | 1.809 | 31 | 8.5% |
| bwd interactions.0.gated_rows | 1.703 | 49 | 8.0% |
| fwd interactions.0.gated_rows | 1.262 | 22 | 5.9% |
| bwd interactions.0.first_reassociation | 0.837 | 19 | 3.9% |
| bwd interactions.1.radial_table | 0.397 | 42 | 1.9% |
| fwd interactions.0.first_reassociation | 0.319 | 29 | 1.5% |
| bwd products.0.symmetric_contractions.f.m | 0.214 | 1 | 1.0% |
| bwd products.1.symmetric_contractions.f.m | 0.187 | 1 | 0.9% |
| fwd interactions.1.radial_table | 0.182 | 8 | 0.9% |
| bwd interactions.0 | 0.160 | 9 | 0.8% |
| fwd interactions.0.radial_table | 0.147 | 8 | 0.7% |
| bwd spherical_harmonics | 0.133 | 63 | 0.6% |
| fwd products.0.symmetric_contractions.f.m | 0.123 | 1 | 0.6% |
| bwd interactions.0.radial_table | 0.118 | 1 | 0.5% |
| bwd interactions.1.linear_up.folded | 0.096 | 8 | 0.4% |
| fwd pair_repulsion_fn | 0.095 | 59 | 0.4% |
| bwd pair_repulsion_fn | 0.092 | 58 | 0.4% |
| fwd <root> | 0.088 | 34 | 0.4% |
| fwd products.1.symmetric_contractions.f.m | 0.083 | 1 | 0.4% |
| fwd spherical_harmonics | 0.077 | 24 | 0.4% |
| fwd interactions.1.linear_up.folded | 0.074 | 6 | 0.4% |
| fwd readouts.0.linear.folded | 0.040 | 3 | 0.2% |

## Eager ranges, NbCl5

| Range (exclusive) | ms/call | kernels | share |
| --- | ---: | ---: | ---: |
| bwd interactions.1.gated_rows | 8.188 | 79 | 27.1% |
| bwd interactions.1 | 5.644 | 83 | 18.7% |
| bwd interactions.0.gated_rows | 3.899 | 47 | 12.9% |
| fwd interactions.1.gated_rows | 3.274 | 30 | 10.8% |
| fwd interactions.0.gated_rows | 2.346 | 21 | 7.8% |
| fwd interactions.1 | 1.538 | 21 | 5.1% |
| bwd interactions.0.first_reassociation | 0.907 | 19 | 3.0% |
| bwd interactions.0 | 0.418 | 9 | 1.4% |
| bwd products.0.symmetric_contractions.f.m | 0.409 | 1 | 1.4% |
| bwd products.1.symmetric_contractions.f.m | 0.401 | 1 | 1.3% |
| bwd interactions.1.radial_table | 0.375 | 42 | 1.2% |
| fwd interactions.0.first_reassociation | 0.368 | 28 | 1.2% |
| bwd interactions.1.linear_up.folded | 0.291 | 8 | 1.0% |
| fwd products.0.symmetric_contractions.f.m | 0.193 | 1 | 0.6% |
| fwd products.1.symmetric_contractions.f.m | 0.179 | 1 | 0.6% |
| fwd interactions.1.radial_table | 0.122 | 8 | 0.4% |
| fwd interactions.1.linear_up.folded | 0.120 | 5 | 0.4% |
| fwd <root> | 0.118 | 34 | 0.4% |
| bwd spherical_harmonics | 0.102 | 63 | 0.3% |
| fwd interactions.0.radial_table | 0.099 | 8 | 0.3% |
| fwd products.0 | 0.089 | 4 | 0.3% |
| fwd pair_repulsion_fn | 0.087 | 59 | 0.3% |
| bwd interactions.1.skip_tp | 0.086 | 2 | 0.3% |
| bwd pair_repulsion_fn | 0.081 | 58 | 0.3% |
| bwd readouts.0.linear | 0.076 | 2 | 0.2% |

Gaps NaCl: profiler {"gaps_per_replay": 232.0, "total_gap_us_per_replay": 142.3106000000008, "histogram_per_replay": {"<2us": {"count": 202.6, "us": 58.127100000002976}, "2-5us": {"count": 29.1, "us": 82.605699999998}, "5-20us": {"count": 0.3, "us": 1.5777999999998429}, ">=20us": {"count": 0.0, "us": 0.0}}, "largest": [{"gap_us": 5.4, "before": "void cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, flo; nsys {"gaps_per_replay": 207.0, "total_gap_us_per_replay": 146.7740999999847, "histogram_per_replay": {"<2us": {"count": 174.7, "us": 32.12089999998625}, "2-5us": {"count": 31.2, "us": 108.81819999999861}, "5-20us": {"count": 1.1, "us": 5.834999999999854}, ">=20us": {"count": 0.0, "us": 0.0}}, "largest": [{"gap_us": 5.92, "before": "triton_poi_fused_mul_sigmoid_slice_unsqueeze_view_28", "after": "void 

Gaps NaNbO3: profiler {"gaps_per_replay": 316.0, "total_gap_us_per_replay": 151.94840000000687, "histogram_per_replay": {"<2us": {"count": 291.7, "us": 30.76780000000341}, "2-5us": {"count": 22.1, "us": 66.69449999999838}, "5-20us": {"count": 1.2, "us": 14.624600000001283}, ">=20us": {"count": 1.0, "us": 39.8615000000038}}, "largest": [{"gap_us": 140.68, "before": "kernel", "after": "Memset (Device)"}, {"gap_us": 36.15; nsys {"gaps_per_replay": 301.0, "total_gap_us_per_replay": 175.15560000003006, "histogram_per_replay": {"<2us": {"count": 281.3, "us": 41.895300000039605}, "2-5us": {"count": 15.2, "us": 56.50879999999088}, "5-20us": {"count": 3.4, "us": 20.05289999999841}, ">=20us": {"count": 1.1, "us": 56.69860000000117}}, "largest": [{"gap_us": 156.41, "before": "triton_poi_fused__to_copy_add_index_mul_scatter_add_s

Gaps NbCl5: profiler {"gaps_per_replay": 274.0, "total_gap_us_per_replay": 146.9558999999915, "histogram_per_replay": {"<2us": {"count": 253.2, "us": 25.937599999999044}, "2-5us": {"count": 17.1, "us": 52.62859999999655}, "5-20us": {"count": 2.7, "us": 28.795199999994292}, ">=20us": {"count": 1.0, "us": 39.59450000000161}}, "largest": [{"gap_us": 136.69, "before": "kernel", "after": "Memset (Device)"}, {"gap_us": 38.7; nsys {"gaps_per_replay": 256.0, "total_gap_us_per_replay": 140.8578999999816, "histogram_per_replay": {"<2us": {"count": 239.4, "us": 37.61829999999045}, "2-5us": {"count": 12.7, "us": 46.528099999999725}, "5-20us": {"count": 2.9, "us": 17.010899999996035}, ">=20us": {"count": 1.0, "us": 39.70059999999539}}, "largest": [{"gap_us": 63.54, "before": "kernel", "after": "void cutlass::Kernel2<cutlass_80_si

## Setup

**setup-NaCl-cold** (NaCl, cold): fm.md call → first step done 70.9 s; process start → first step 75.1 s (import fastmace 0.1 s, torch 1.6 s); first step 0.01 s; steady state with output 3.128 ms/step

Simulation._setup_s: plan 1.94 s, build 44.73 s, capture 0.57 s, provenance 0.12 s

| Stage (nested timers) | s | calls | inside |
| --- | ---: | ---: | ---: |
| fm.md (total) | 76.927 | 1 |  |
| Simulation.__init__ | 47.660 | 1 | fm.md (total) |
| build: MinimalPlanner.build | 44.727 | 1 | Simulation.__init__ |
| engine: GraphEngine.__init__ | 24.433 | 1 | build: MinimalPlanner.build |
| engine: first compiled force call | 24.394 | 1 | engine: GraphEngine.__init__ |
| load: load_pristine | 22.516 | 1 | fm.md (total) |
| OEq: prepare_hybrid_if_possible | 17.784 | 1 | load: load_pristine |
| OEq: prepare_hybrid | 17.767 | 2 | OEq: prepare_hybrid_if_possible, load: loading.load (MACE ca |
| OEq: reuse_extension (build or cache hit) | 17.710 | 1 | OEq: prepare_hybrid |
| build: MACEForceField.__init__ | 14.611 | 1 | build: MinimalPlanner.build |
| compile: compile_model (wrap; compiles lazily) | 14.327 | 1 | build: MACEForceField.__init__ |
| rewrites: apply (incl. self-checks) | 13.544 | 1 | compile: compile_model (wrap; compiles lazily) |
| load: loading.load (MACE calculator, conversion) | 8.234 | 2 | build: template cache, load: load_pristine |
| rewrites: self-checks | 7.832 | 21 | rewrites: apply (incl. self-checks) |
| Simulation.run | 6.392 | 1 | fm.md (total) |
| radial tables: certify | 4.530 | 2 | rewrites: apply (incl. self-checks) |
| build: execution model | 3.987 | 1 | build: MinimalPlanner.build |
| build: ForceField.execution_model | 3.987 | 1 | build: execution model |
| build: template cache | 3.980 | 1 | build: ForceField.execution_model |
| plan: MinimalPlanner.plan | 1.837 | 1 | Simulation.__init__ |
| rewrites: resolve | 1.692 | 1 | build: MinimalPlanner.build |
| compile: private_copy | 0.934 | 2 | build: templates.private (copy), compile: compile_model (wra |
| radial tables: fit | 0.657 | 2 | rewrites: apply (incl. self-checks) |
| engine: prepare (captures) | 0.566 | 1 | Simulation.__init__ |
| engine: graph capture | 0.566 | 2 | engine: prepare (captures) |
| build: templates.private (copy) | 0.478 | 1 | build: template cache |
| plan: hardware | 0.296 | 1 | plan: MinimalPlanner.plan |
| OEq streaming: render | 0.112 | 2 | rewrites: apply (incl. self-checks) |
| output: metadata | 0.093 | 1 | Simulation.__init__ |
| OEq streaming: NVRTC compile_cubin | 0.088 | 4 | rewrites: apply (incl. self-checks) |

torch.compile phases:

```
TorchDynamo compilation metrics:
Function                                                            Runtimes (s)
----------------------------------------------------------------  --------------
_compile.compile_inner                                                   17.4399
compile_attempt_0                                                         2.6339
bytecode_tracing                                                          4.4231
get_fake_value                                                            0.6749
create_proxy                                                              0.315
wrap_to_fake_tensor_and_record                                            0.1145
variable_builder_call                                                     2.2514
compile_attempt_1                                                        13.0201
insert_deferred_runtime_asserts                                           0.0518
OutputGraph.call_user_compiler                                           10.6981
inductor_codecache_torch_key                                              0.0632
_recursive_pre_grad_passes                                                0.1507
pass.pre_grad_passes.group_batch_fusion_passes                            0.0007
pass.pre_grad_passes.efficient_conv_bn_eval_pass                          0.0022
pass.pre_grad_passes.apply_gumbel_max_trick_pass                          0.0009
create_aot_dispatcher_function                                           10.5288
aot_collec
```

**setup-NaCl-cprofile** (NaCl, warm-cprofile): fm.md call → first step done 47.9 s; process start → first step 52.3 s (import fastmace 0.1 s, torch 1.3 s); first step 0.01 s; steady state with output 3.096 ms/step

Simulation._setup_s: plan 1.77 s, build 38.47 s, capture 0.42 s, provenance 0.47 s

| Stage (nested timers) | s | calls | inside |
| --- | ---: | ---: | ---: |
| fm.md (total) | 54.136 | 1 |  |
| Simulation.__init__ | 41.513 | 1 | fm.md (total) |
| build: MinimalPlanner.build | 38.468 | 1 | Simulation.__init__ |
| engine: GraphEngine.__init__ | 19.054 | 1 | build: MinimalPlanner.build |
| engine: first compiled force call | 18.948 | 1 | engine: GraphEngine.__init__ |
| build: MACEForceField.__init__ | 11.548 | 1 | build: MinimalPlanner.build |
| compile: compile_model (wrap; compiles lazily) | 11.390 | 1 | build: MACEForceField.__init__ |
| load: loading.load (MACE calculator, conversion) | 10.243 | 2 | build: template cache, load: load_pristine |
| rewrites: apply (incl. self-checks) | 9.820 | 1 | compile: compile_model (wrap; compiles lazily) |
| Simulation.run | 6.592 | 1 | fm.md (total) |
| build: execution model | 5.806 | 1 | build: MinimalPlanner.build |
| build: ForceField.execution_model | 5.806 | 1 | build: execution model |
| build: template cache | 5.799 | 1 | build: ForceField.execution_model |
| load: load_pristine | 5.667 | 1 | fm.md (total) |
| rewrites: self-checks | 4.334 | 21 | rewrites: apply (incl. self-checks) |
| radial tables: certify | 3.822 | 2 | rewrites: apply (incl. self-checks) |
| compile: private_copy | 2.212 | 2 | build: templates.private (copy), compile: compile_model (wra |
| rewrites: resolve | 2.054 | 1 | build: MinimalPlanner.build |
| plan: MinimalPlanner.plan | 1.667 | 1 | Simulation.__init__ |
| build: templates.private (copy) | 1.187 | 1 | build: template cache |
| radial tables: fit | 1.152 | 2 | rewrites: apply (incl. self-checks) |
| engine: prepare (captures) | 0.424 | 1 | Simulation.__init__ |
| engine: graph capture | 0.424 | 2 | engine: prepare (captures) |
| plan: hardware | 0.303 | 1 | plan: MinimalPlanner.plan |
| OEq streaming: render | 0.121 | 2 | rewrites: apply (incl. self-checks) |
| OEq streaming: NVRTC compile_cubin | 0.095 | 4 | rewrites: apply (incl. self-checks) |
| output: metadata | 0.081 | 1 | Simulation.__init__ |
| load: sha256_of | 0.035 | 2 | load: loading.load (MACE calculator, conversion) |
| OEq: prepare_hybrid_if_possible | 0.035 | 1 | load: load_pristine |
| OEq: prepare_hybrid | 0.028 | 2 | OEq: prepare_hybrid_if_possible, load: loading.load (MACE ca |

torch.compile phases:

```
TorchDynamo compilation metrics:
Function                                               Runtimes (s)
---------------------------------------------------  --------------
_compile.compile_inner                                      16.8619
compile_attempt_0                                            4.1183
bytecode_tracing                                             7.6335
get_fake_value                                               1.4545
create_proxy                                                 0.5386
wrap_to_fake_tensor_and_record                               0.1862
variable_builder_call                                        3.3456
compile_attempt_1                                           10.2174
insert_deferred_runtime_asserts                              0.1068
OutputGraph.call_user_compiler                               6.2252
inductor_codecache_torch_key                                 0.1044
_recursive_pre_grad_passes                                   0.2217
pass.pre_grad_passes.group_batch_fusion_passes               0.0016
pass.pre_grad_passes.efficient_conv_bn_eval_pass             0.0043
pass.pre_grad_passes.apply_gumbel_max_trick_pass             0.0019
create_aot_dispatcher_function                               5.9469
aot_collect_metadata                                         0.6763
aot_trace_joint_graph                                        2.4777
_recursive_joint_graph_passes                                0.9602
pass.joint_graph_passes.remove_noop_ops
```

**setup-NaCl-warm1** (NaCl, warm): fm.md call → first step done 37.1 s; process start → first step 41.8 s (import fastmace 0.1 s, torch 1.7 s); first step 0.01 s; steady state with output 3.114 ms/step

Simulation._setup_s: plan 1.62 s, build 28.90 s, capture 0.32 s, provenance 0.24 s

| Stage (nested timers) | s | calls | inside |
| --- | ---: | ---: | ---: |
| fm.md (total) | 43.173 | 1 |  |
| Simulation.__init__ | 31.406 | 1 | fm.md (total) |
| build: MinimalPlanner.build | 28.901 | 1 | Simulation.__init__ |
| engine: GraphEngine.__init__ | 11.479 | 1 | build: MinimalPlanner.build |
| engine: first compiled force call | 11.445 | 1 | engine: GraphEngine.__init__ |
| build: MACEForceField.__init__ | 11.137 | 1 | build: MinimalPlanner.build |
| compile: compile_model (wrap; compiles lazily) | 10.974 | 1 | build: MACEForceField.__init__ |
| rewrites: apply (incl. self-checks) | 10.173 | 1 | compile: compile_model (wrap; compiles lazily) |
| load: loading.load (MACE calculator, conversion) | 8.309 | 2 | build: template cache, load: load_pristine |
| Simulation.run | 6.438 | 1 | fm.md (total) |
| load: load_pristine | 4.719 | 1 | fm.md (total) |
| rewrites: self-checks | 4.470 | 21 | rewrites: apply (incl. self-checks) |
| build: execution model | 4.249 | 1 | build: MinimalPlanner.build |
| build: ForceField.execution_model | 4.249 | 1 | build: execution model |
| build: template cache | 4.243 | 1 | build: ForceField.execution_model |
| radial tables: certify | 3.668 | 2 | rewrites: apply (incl. self-checks) |
| rewrites: resolve | 2.033 | 1 | build: MinimalPlanner.build |
| radial tables: fit | 1.512 | 2 | rewrites: apply (incl. self-checks) |
| plan: MinimalPlanner.plan | 1.506 | 1 | Simulation.__init__ |
| compile: private_copy | 1.110 | 2 | build: templates.private (copy), compile: compile_model (wra |
| build: templates.private (copy) | 0.632 | 1 | build: template cache |
| plan: hardware | 0.341 | 1 | plan: MinimalPlanner.plan |
| engine: prepare (captures) | 0.324 | 1 | Simulation.__init__ |
| engine: graph capture | 0.324 | 2 | engine: prepare (captures) |
| OEq streaming: render | 0.110 | 2 | rewrites: apply (incl. self-checks) |
| OEq streaming: NVRTC compile_cubin | 0.099 | 4 | rewrites: apply (incl. self-checks) |
| output: metadata | 0.070 | 1 | Simulation.__init__ |
| load: sha256_of | 0.034 | 2 | load: loading.load (MACE calculator, conversion) |
| output: open | 0.025 | 1 | Simulation.__init__ |
| OEq: prepare_hybrid_if_possible | 0.021 | 1 | load: load_pristine |

torch.compile phases:

```
TorchDynamo compilation metrics:
Function                                               Runtimes (s)
---------------------------------------------------  --------------
_compile.compile_inner                                       9.4638
compile_attempt_0                                            2.3838
bytecode_tracing                                             4.1631
get_fake_value                                               0.679
create_proxy                                                 0.266
wrap_to_fake_tensor_and_record                               0.1169
variable_builder_call                                        2.0288
compile_attempt_1                                            5.3934
insert_deferred_runtime_asserts                              0.2886
OutputGraph.call_user_compiler                               2.9829
inductor_codecache_torch_key                                 0.0625
_recursive_pre_grad_passes                                   0.1421
pass.pre_grad_passes.group_batch_fusion_passes               0.0008
pass.pre_grad_passes.efficient_conv_bn_eval_pass             0.0025
pass.pre_grad_passes.apply_gumbel_max_trick_pass             0.0009
create_aot_dispatcher_function                               2.8121
aot_collect_metadata                                         0.3443
aot_trace_joint_graph                                        1.2964
_recursive_joint_graph_passes                                0.3823
pass.joint_graph_passes.remove_noop_ops  
```

**setup-NaCl-warm2** (NaCl, warm): fm.md call → first step done 34.1 s; process start → first step 38.1 s (import fastmace 0.0 s, torch 1.2 s); first step 0.01 s; steady state with output 3.121 ms/step

Simulation._setup_s: plan 1.49 s, build 27.51 s, capture 0.31 s, provenance 0.08 s

| Stage (nested timers) | s | calls | inside |
| --- | ---: | ---: | ---: |
| fm.md (total) | 40.113 | 1 |  |
| Simulation.__init__ | 29.616 | 1 | fm.md (total) |
| build: MinimalPlanner.build | 27.507 | 1 | Simulation.__init__ |
| engine: GraphEngine.__init__ | 11.747 | 1 | build: MinimalPlanner.build |
| engine: first compiled force call | 11.705 | 1 | engine: GraphEngine.__init__ |
| build: MACEForceField.__init__ | 9.839 | 1 | build: MinimalPlanner.build |
| compile: compile_model (wrap; compiles lazily) | 9.708 | 1 | build: MACEForceField.__init__ |
| rewrites: apply (incl. self-checks) | 8.904 | 1 | compile: compile_model (wrap; compiles lazily) |
| load: loading.load (MACE calculator, conversion) | 7.367 | 2 | build: template cache, load: load_pristine |
| Simulation.run | 6.332 | 1 | fm.md (total) |
| rewrites: self-checks | 4.326 | 21 | rewrites: apply (incl. self-checks) |
| build: execution model | 4.119 | 1 | build: MinimalPlanner.build |
| build: ForceField.execution_model | 4.119 | 1 | build: execution model |
| build: template cache | 4.112 | 1 | build: ForceField.execution_model |
| load: load_pristine | 3.909 | 1 | fm.md (total) |
| radial tables: certify | 3.197 | 2 | rewrites: apply (incl. self-checks) |
| rewrites: resolve | 1.799 | 1 | build: MinimalPlanner.build |
| plan: MinimalPlanner.plan | 1.437 | 1 | Simulation.__init__ |
| compile: private_copy | 1.102 | 2 | build: templates.private (copy), compile: compile_model (wra |
| radial tables: fit | 0.900 | 2 | rewrites: apply (incl. self-checks) |
| build: templates.private (copy) | 0.619 | 1 | build: template cache |
| plan: hardware | 0.343 | 1 | plan: MinimalPlanner.plan |
| engine: prepare (captures) | 0.306 | 1 | Simulation.__init__ |
| engine: graph capture | 0.306 | 2 | engine: prepare (captures) |
| OEq streaming: render | 0.095 | 2 | rewrites: apply (incl. self-checks) |
| OEq streaming: NVRTC compile_cubin | 0.088 | 4 | rewrites: apply (incl. self-checks) |
| output: metadata | 0.076 | 1 | Simulation.__init__ |
| OEq: prepare_hybrid_if_possible | 0.034 | 1 | load: load_pristine |
| load: sha256_of | 0.033 | 2 | load: loading.load (MACE calculator, conversion) |
| OEq: prepare_hybrid | 0.024 | 2 | OEq: prepare_hybrid_if_possible, load: loading.load (MACE ca |

torch.compile phases:

```
TorchDynamo compilation metrics:
Function                                               Runtimes (s)
---------------------------------------------------  --------------
_compile.compile_inner                                       9.4253
compile_attempt_0                                            2.3967
bytecode_tracing                                             4.1879
get_fake_value                                               0.6885
create_proxy                                                 0.3485
wrap_to_fake_tensor_and_record                               0.1091
variable_builder_call                                        2.0278
compile_attempt_1                                            5.3316
insert_deferred_runtime_asserts                              0.2951
OutputGraph.call_user_compiler                               2.9337
inductor_codecache_torch_key                                 0.0621
_recursive_pre_grad_passes                                   0.1374
pass.pre_grad_passes.group_batch_fusion_passes               0.0007
pass.pre_grad_passes.efficient_conv_bn_eval_pass             0.0032
pass.pre_grad_passes.apply_gumbel_max_trick_pass             0.001
create_aot_dispatcher_function                               2.7656
aot_collect_metadata                                         0.3445
aot_trace_joint_graph                                        1.2479
_recursive_joint_graph_passes                                0.4031
pass.joint_graph_passes.remove_noop_ops 
```

**setup-NaNbO3-warm1** (NaNbO3, warm): fm.md call → first step done 41.2 s; process start → first step 45.3 s (import fastmace 0.0 s, torch 1.1 s); first step 0.03 s; steady state with output 14.846 ms/step

Simulation._setup_s: plan 1.75 s, build 33.70 s, capture 0.53 s, provenance 0.15 s

| Stage (nested timers) | s | calls | inside |
| --- | ---: | ---: | ---: |
| fm.md (total) | 70.737 | 1 |  |
| Simulation.__init__ | 36.477 | 1 | fm.md (total) |
| build: MinimalPlanner.build | 33.695 | 1 | Simulation.__init__ |
| Simulation.run | 29.735 | 1 | fm.md (total) |
| build: MACEForceField.__init__ | 16.316 | 1 | build: MinimalPlanner.build |
| compile: compile_model (wrap; compiles lazily) | 16.115 | 1 | build: MACEForceField.__init__ |
| rewrites: apply (incl. self-checks) | 15.300 | 1 | compile: compile_model (wrap; compiles lazily) |
| engine: GraphEngine.__init__ | 11.642 | 1 | build: MinimalPlanner.build |
| engine: first compiled force call | 11.605 | 1 | engine: GraphEngine.__init__ |
| load: loading.load (MACE calculator, conversion) | 7.770 | 2 | build: template cache, load: load_pristine |
| radial tables: certify | 7.661 | 2 | rewrites: apply (incl. self-checks) |
| rewrites: self-checks | 4.509 | 21 | rewrites: apply (incl. self-checks) |
| load: load_pristine | 4.306 | 1 | fm.md (total) |
| build: execution model | 3.959 | 1 | build: MinimalPlanner.build |
| build: ForceField.execution_model | 3.959 | 1 | build: execution model |
| build: template cache | 3.954 | 1 | build: ForceField.execution_model |
| radial tables: fit | 2.603 | 2 | rewrites: apply (incl. self-checks) |
| rewrites: resolve | 1.775 | 1 | build: MinimalPlanner.build |
| plan: MinimalPlanner.plan | 1.659 | 1 | Simulation.__init__ |
| compile: private_copy | 0.926 | 2 | build: templates.private (copy), compile: compile_model (wra |
| engine: prepare (captures) | 0.530 | 1 | Simulation.__init__ |
| engine: graph capture | 0.530 | 2 | engine: prepare (captures) |
| build: templates.private (copy) | 0.461 | 1 | build: template cache |
| plan: hardware | 0.264 | 1 | plan: MinimalPlanner.plan |
| output: metadata | 0.096 | 1 | Simulation.__init__ |
| OEq streaming: NVRTC compile_cubin | 0.090 | 4 | rewrites: apply (incl. self-checks) |
| OEq streaming: render | 0.086 | 2 | rewrites: apply (incl. self-checks) |
| load: sha256_of | 0.035 | 2 | load: loading.load (MACE calculator, conversion) |
| OEq: prepare_hybrid_if_possible | 0.029 | 1 | load: load_pristine |
| output: open | 0.028 | 1 | Simulation.__init__ |

torch.compile phases:

```
TorchDynamo compilation metrics:
Function                                               Runtimes (s)
---------------------------------------------------  --------------
_compile.compile_inner                                       9.2591
compile_attempt_0                                            2.4257
bytecode_tracing                                             4.3252
get_fake_value                                               0.6595
create_proxy                                                 0.2614
wrap_to_fake_tensor_and_record                               0.4892
variable_builder_call                                        2.2932
compile_attempt_1                                            5.1887
insert_deferred_runtime_asserts                              0.0515
OutputGraph.call_user_compiler                               2.9539
inductor_codecache_torch_key                                 0.0595
_recursive_pre_grad_passes                                   0.1215
pass.pre_grad_passes.group_batch_fusion_passes               0.0006
pass.pre_grad_passes.efficient_conv_bn_eval_pass             0.002
pass.pre_grad_passes.apply_gumbel_max_trick_pass             0.0009
create_aot_dispatcher_function                               2.7951
aot_collect_metadata                                         0.3453
aot_trace_joint_graph                                        1.2493
_recursive_joint_graph_passes                                0.4059
pass.joint_graph_passes.remove_noop_ops 
```

**setup-NaNbO3-warm2** (NaNbO3, warm): fm.md call → first step done 41.4 s; process start → first step 45.9 s (import fastmace 0.1 s, torch 1.6 s); first step 0.03 s; steady state with output 14.734 ms/step

Simulation._setup_s: plan 1.61 s, build 33.45 s, capture 0.42 s, provenance 0.11 s

| Stage (nested timers) | s | calls | inside |
| --- | ---: | ---: | ---: |
| fm.md (total) | 70.746 | 1 |  |
| Simulation.__init__ | 35.871 | 1 | fm.md (total) |
| build: MinimalPlanner.build | 33.451 | 1 | Simulation.__init__ |
| Simulation.run | 29.688 | 1 | fm.md (total) |
| build: MACEForceField.__init__ | 15.936 | 1 | build: MinimalPlanner.build |
| compile: compile_model (wrap; compiles lazily) | 15.714 | 1 | build: MACEForceField.__init__ |
| rewrites: apply (incl. self-checks) | 14.900 | 1 | compile: compile_model (wrap; compiles lazily) |
| engine: GraphEngine.__init__ | 11.541 | 1 | build: MinimalPlanner.build |
| engine: first compiled force call | 11.497 | 1 | engine: GraphEngine.__init__ |
| load: loading.load (MACE calculator, conversion) | 8.617 | 2 | build: template cache, load: load_pristine |
| radial tables: certify | 7.729 | 2 | rewrites: apply (incl. self-checks) |
| load: load_pristine | 5.037 | 1 | fm.md (total) |
| rewrites: self-checks | 4.552 | 21 | rewrites: apply (incl. self-checks) |
| build: execution model | 4.065 | 1 | build: MinimalPlanner.build |
| build: ForceField.execution_model | 4.065 | 1 | build: execution model |
| build: template cache | 4.059 | 1 | build: ForceField.execution_model |
| radial tables: fit | 2.129 | 2 | rewrites: apply (incl. self-checks) |
| rewrites: resolve | 1.906 | 1 | build: MinimalPlanner.build |
| plan: MinimalPlanner.plan | 1.501 | 1 | Simulation.__init__ |
| compile: private_copy | 0.939 | 2 | build: templates.private (copy), compile: compile_model (wra |
| build: templates.private (copy) | 0.458 | 1 | build: template cache |
| engine: prepare (captures) | 0.416 | 1 | Simulation.__init__ |
| engine: graph capture | 0.416 | 2 | engine: prepare (captures) |
| plan: hardware | 0.294 | 1 | plan: MinimalPlanner.plan |
| OEq streaming: NVRTC compile_cubin | 0.089 | 4 | rewrites: apply (incl. self-checks) |
| OEq streaming: render | 0.079 | 2 | rewrites: apply (incl. self-checks) |
| output: metadata | 0.057 | 1 | Simulation.__init__ |
| load: sha256_of | 0.035 | 2 | load: loading.load (MACE calculator, conversion) |
| OEq: prepare_hybrid_if_possible | 0.020 | 1 | load: load_pristine |
| output: open | 0.018 | 1 | Simulation.__init__ |

torch.compile phases:

```
TorchDynamo compilation metrics:
Function                                               Runtimes (s)
---------------------------------------------------  --------------
_compile.compile_inner                                       9.4636
compile_attempt_0                                            2.5137
bytecode_tracing                                             4.4449
get_fake_value                                               0.6599
create_proxy                                                 0.2625
wrap_to_fake_tensor_and_record                               0.4983
variable_builder_call                                        2.4033
compile_attempt_1                                            5.3057
insert_deferred_runtime_asserts                              0.0483
OutputGraph.call_user_compiler                               2.9598
inductor_codecache_torch_key                                 0.0609
_recursive_pre_grad_passes                                   0.124
pass.pre_grad_passes.group_batch_fusion_passes               0.0006
pass.pre_grad_passes.efficient_conv_bn_eval_pass             0.0018
pass.pre_grad_passes.apply_gumbel_max_trick_pass             0.0008
create_aot_dispatcher_function                               2.8017
aot_collect_metadata                                         0.358
aot_trace_joint_graph                                        1.2858
_recursive_joint_graph_passes                                0.3745
pass.joint_graph_passes.remove_noop_ops  
```

**setup-NbCl5-warm1** (NbCl5, warm): fm.md call → first step done 35.5 s; process start → first step 39.1 s (import fastmace 0.1 s, torch 1.2 s); first step 0.18 s; steady state with output 19.315 ms/step

Simulation._setup_s: plan 1.54 s, build 27.93 s, capture 0.51 s, provenance 0.07 s

| Stage (nested timers) | s | calls | inside |
| --- | ---: | ---: | ---: |
| fm.md (total) | 73.619 | 1 |  |
| Simulation.run | 38.640 | 1 | fm.md (total) |
| Simulation.__init__ | 30.240 | 1 | fm.md (total) |
| build: MinimalPlanner.build | 27.924 | 1 | Simulation.__init__ |
| engine: GraphEngine.__init__ | 12.028 | 1 | build: MinimalPlanner.build |
| engine: first compiled force call | 11.975 | 1 | engine: GraphEngine.__init__ |
| build: MACEForceField.__init__ | 10.137 | 1 | build: MinimalPlanner.build |
| compile: compile_model (wrap; compiles lazily) | 9.978 | 1 | build: MACEForceField.__init__ |
| rewrites: apply (incl. self-checks) | 9.131 | 1 | compile: compile_model (wrap; compiles lazily) |
| load: loading.load (MACE calculator, conversion) | 7.969 | 2 | build: template cache, load: load_pristine |
| load: load_pristine | 4.493 | 1 | fm.md (total) |
| rewrites: self-checks | 4.444 | 21 | rewrites: apply (incl. self-checks) |
| build: execution model | 4.030 | 1 | build: MinimalPlanner.build |
| build: ForceField.execution_model | 4.030 | 1 | build: execution model |
| build: template cache | 4.023 | 1 | build: ForceField.execution_model |
| radial tables: certify | 3.485 | 2 | rewrites: apply (incl. self-checks) |
| rewrites: resolve | 1.726 | 1 | build: MinimalPlanner.build |
| plan: MinimalPlanner.plan | 1.497 | 1 | Simulation.__init__ |
| compile: private_copy | 0.965 | 2 | build: templates.private (copy), compile: compile_model (wra |
| radial tables: fit | 0.693 | 2 | rewrites: apply (incl. self-checks) |
| engine: graph capture | 0.630 | 3 | Simulation.run, engine: prepare (captures) |
| engine: prepare (captures) | 0.510 | 1 | Simulation.__init__ |
| build: templates.private (copy) | 0.466 | 1 | build: template cache |
| plan: hardware | 0.207 | 1 | plan: MinimalPlanner.plan |
| OEq streaming: render | 0.092 | 2 | rewrites: apply (incl. self-checks) |
| OEq streaming: NVRTC compile_cubin | 0.086 | 4 | rewrites: apply (incl. self-checks) |
| OEq: prepare_hybrid_if_possible | 0.081 | 1 | load: load_pristine |
| output: metadata | 0.067 | 1 | Simulation.__init__ |
| OEq: prepare_hybrid | 0.054 | 2 | OEq: prepare_hybrid_if_possible, load: loading.load (MACE ca |
| load: sha256_of | 0.033 | 2 | load: loading.load (MACE calculator, conversion) |

torch.compile phases:

```
TorchDynamo compilation metrics:
Function                                               Runtimes (s)
---------------------------------------------------  --------------
_compile.compile_inner                                       9.6116
compile_attempt_0                                            2.6016
bytecode_tracing                                             4.4619
get_fake_value                                               0.6714
create_proxy                                                 0.2922
wrap_to_fake_tensor_and_record                               0.1078
variable_builder_call                                        2.1651
compile_attempt_1                                            5.3311
insert_deferred_runtime_asserts                              0.0513
OutputGraph.call_user_compiler                               2.9509
inductor_codecache_torch_key                                 0.0603
_recursive_pre_grad_passes                                   0.1401
pass.pre_grad_passes.group_batch_fusion_passes               0.0006
pass.pre_grad_passes.efficient_conv_bn_eval_pass             0.0022
pass.pre_grad_passes.apply_gumbel_max_trick_pass             0.0008
create_aot_dispatcher_function                               2.7845
aot_collect_metadata                                         0.3241
aot_trace_joint_graph                                        1.2612
_recursive_joint_graph_passes                                0.3994
pass.joint_graph_passes.remove_noop_ops
```

**setup-NbCl5-warm2** (NbCl5, warm): fm.md call → first step done 34.4 s; process start → first step 37.1 s (import fastmace 0.1 s, torch 1.1 s); first step 0.19 s; steady state with output 19.322 ms/step

Simulation._setup_s: plan 1.54 s, build 27.61 s, capture 0.48 s, provenance 0.10 s

| Stage (nested timers) | s | calls | inside |
| --- | ---: | ---: | ---: |
| fm.md (total) | 72.776 | 1 |  |
| Simulation.run | 38.730 | 1 | fm.md (total) |
| Simulation.__init__ | 30.016 | 1 | fm.md (total) |
| build: MinimalPlanner.build | 27.602 | 1 | Simulation.__init__ |
| engine: GraphEngine.__init__ | 11.762 | 1 | build: MinimalPlanner.build |
| engine: first compiled force call | 11.715 | 1 | engine: GraphEngine.__init__ |
| build: MACEForceField.__init__ | 10.420 | 1 | build: MinimalPlanner.build |
| compile: compile_model (wrap; compiles lazily) | 10.272 | 1 | build: MACEForceField.__init__ |
| rewrites: apply (incl. self-checks) | 9.492 | 1 | compile: compile_model (wrap; compiles lazily) |
| load: loading.load (MACE calculator, conversion) | 7.318 | 2 | build: template cache, load: load_pristine |
| rewrites: self-checks | 4.445 | 21 | rewrites: apply (incl. self-checks) |
| build: execution model | 4.103 | 1 | build: MinimalPlanner.build |
| build: ForceField.execution_model | 4.103 | 1 | build: execution model |
| build: template cache | 4.095 | 1 | build: ForceField.execution_model |
| load: load_pristine | 3.709 | 1 | fm.md (total) |
| radial tables: certify | 3.411 | 2 | rewrites: apply (incl. self-checks) |
| plan: MinimalPlanner.plan | 1.414 | 1 | Simulation.__init__ |
| rewrites: resolve | 1.313 | 1 | build: MinimalPlanner.build |
| radial tables: fit | 1.201 | 2 | rewrites: apply (incl. self-checks) |
| compile: private_copy | 0.920 | 2 | build: templates.private (copy), compile: compile_model (wra |
| engine: graph capture | 0.596 | 3 | Simulation.run, engine: prepare (captures) |
| engine: prepare (captures) | 0.479 | 1 | Simulation.__init__ |
| build: templates.private (copy) | 0.456 | 1 | build: template cache |
| plan: hardware | 0.308 | 1 | plan: MinimalPlanner.plan |
| OEq streaming: render | 0.095 | 2 | rewrites: apply (incl. self-checks) |
| OEq streaming: NVRTC compile_cubin | 0.089 | 4 | rewrites: apply (incl. self-checks) |
| output: metadata | 0.051 | 1 | Simulation.__init__ |
| load: sha256_of | 0.034 | 2 | load: loading.load (MACE calculator, conversion) |
| OEq: prepare_hybrid_if_possible | 0.030 | 1 | load: load_pristine |
| OEq: prepare_hybrid | 0.024 | 2 | OEq: prepare_hybrid_if_possible, load: loading.load (MACE ca |

torch.compile phases:

```
TorchDynamo compilation metrics:
Function                                               Runtimes (s)
---------------------------------------------------  --------------
_compile.compile_inner                                       9.5333
compile_attempt_0                                            2.5665
bytecode_tracing                                             4.4838
get_fake_value                                               0.6813
create_proxy                                                 0.3206
wrap_to_fake_tensor_and_record                               0.1126
variable_builder_call                                        2.1466
compile_attempt_1                                            5.2818
insert_deferred_runtime_asserts                              0.048
OutputGraph.call_user_compiler                               2.8863
inductor_codecache_torch_key                                 0.0629
_recursive_pre_grad_passes                                   0.1175
pass.pre_grad_passes.group_batch_fusion_passes               0.0006
pass.pre_grad_passes.efficient_conv_bn_eval_pass             0.0018
pass.pre_grad_passes.apply_gumbel_max_trick_pass             0.0007
create_aot_dispatcher_function                               2.7276
aot_collect_metadata                                         0.3229
aot_trace_joint_graph                                        1.2653
_recursive_joint_graph_passes                                0.3542
pass.joint_graph_passes.remove_noop_ops 
```

