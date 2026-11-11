#include <cuda_fp16.h>
typedef int int32_t; typedef long long int64_t;
constexpr int INPUT_DIM = 512, EDGE_DIM = 16, WEIGHT_DIM = 1280, CHANNEL_DIM = 128, TERMS = 4, PADDED_TERMS = 4;
constexpr int LOGICAL_GROUP_SIZE = 32;
typedef float scalar_t;
typedef float acc_t;

extern "C" __global__ void __launch_bounds__(256) census_wprime(
    const float* x, float* dx, const float* __restrict__ gprime, const int32_t* senders, const int32_t* slabs,
    const float* ts, const float* scs, const unsigned* flags, const float* shs, const float* dout, float* out,
    int n_edges, int store) {
    __shared__ __align__(16) scalar_t stage_row[5120];
    __shared__ __align__(16) scalar_t stage_esh[8][EDGE_DIM];
    for (int i = threadIdx.x; i < 5120; i += blockDim.x) stage_row[i] = dout[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int e = blockIdx.x * 8 + w; e < n_edges; e += gridDim.x * 8) {
    if (lane < EDGE_DIM) stage_esh[w][lane] = shs[int64_t(e) * EDGE_DIM + lane];
    __syncwarp();
    const int64_t input_base = int64_t(senders[e]) * INPUT_DIM;
    const scalar_t* sh_edge = stage_esh[w];
    // an interior, sloped edge (the common path): the loop is then one basic block, as half2's
    const bool sloped = true, inside = true;
    const float t = ts[e];
    const scalar_t sc = scs[e];
    const scalar_t* g = gprime + int64_t(slabs[e]) * PADDED_TERMS * WEIGHT_DIM;
    acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};
    acc_t radial_gradient = acc_t(0);
    const int64_t weight_base = int64_t(e) * WEIGHT_DIM;  // unused by the body's arithmetic
    for (int channel = lane; channel < CHANNEL_DIM;
         channel += LOGICAL_GROUP_SIZE) {
            acc_t input_gradient_0_0 = acc_t(0);
            acc_t input_gradient_128_0 = acc_t(0);
            acc_t input_gradient_128_1 = acc_t(0);
            acc_t input_gradient_128_2 = acc_t(0);

        
             {
                const int weight_index =
                    0 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (inside) {
                    const float4* gq =
                        reinterpret_cast<const float4*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 4; ++k) {
                        const float4 v = __ldg(gq + k);
                        gv[4 * k + 0] = v.x;
                        gv[4 * k + 1] = v.y;
                        gv[4 * k + 2] = v.z;
                        gv[4 * k + 3] = v.w;
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = scalar_t(0);
                }
                scalar_t radial_weight = gv[TERMS - 1];
                scalar_t radial_dt = scalar_t(0);
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t + radial_weight;
                    radial_weight = radial_weight * t + gv[k];
                }
                acc_t weight_gradient = acc_t(0);
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                acc_t u_0 = acc_t(0);
                // SCALAR_PATHS: an l1 = 0 path's edge gradient w x0 c g_o added per term, without v_j.
                const scalar_t weighted_input =
                    radial_weight * x[input_base + 0 + channel];

                 {
                    const scalar_t output_gradient = stage_row[0
                        + 0 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh_edge[0];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[0]
                            += weighted_input * scaled;
                    }
                }
                const scalar_t scaled_weight = radial_weight;
                weight_gradient +=
                    x[input_base + 0 + channel] * u_0;
                input_gradient_0_0 += scaled_weight * u_0;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    128 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (inside) {
                    const float4* gq =
                        reinterpret_cast<const float4*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 4; ++k) {
                        const float4 v = __ldg(gq + k);
                        gv[4 * k + 0] = v.x;
                        gv[4 * k + 1] = v.y;
                        gv[4 * k + 2] = v.z;
                        gv[4 * k + 3] = v.w;
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = scalar_t(0);
                }
                scalar_t radial_weight = gv[TERMS - 1];
                scalar_t radial_dt = scalar_t(0);
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t + radial_weight;
                    radial_weight = radial_weight * t + gv[k];
                }
                acc_t weight_gradient = acc_t(0);
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                acc_t u_0 = acc_t(0);
                acc_t u_1 = acc_t(0);
                acc_t u_2 = acc_t(0);
                acc_t v_0 = acc_t(0);
                acc_t v_1 = acc_t(0);
                acc_t v_2 = acc_t(0);

                 {
                    const scalar_t output_gradient = stage_row[0
                        + 128 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[1];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[2];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[3];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    }
                }
                const scalar_t scaled_weight = radial_weight;
                weight_gradient +=
                    x[input_base + 128 + channel] * u_0;
                input_gradient_128_0 += scaled_weight * u_0;
                weight_gradient +=
                    x[input_base + 256 + channel] * u_1;
                input_gradient_128_1 += scaled_weight * u_1;
                weight_gradient +=
                    x[input_base + 384 + channel] * u_2;
                input_gradient_128_2 += scaled_weight * u_2;
                edge_gradient[1] +=
                    scaled_weight * v_0;
                edge_gradient[2] +=
                    scaled_weight * v_1;
                edge_gradient[3] +=
                    scaled_weight * v_2;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    256 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (inside) {
                    const float4* gq =
                        reinterpret_cast<const float4*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 4; ++k) {
                        const float4 v = __ldg(gq + k);
                        gv[4 * k + 0] = v.x;
                        gv[4 * k + 1] = v.y;
                        gv[4 * k + 2] = v.z;
                        gv[4 * k + 3] = v.w;
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = scalar_t(0);
                }
                scalar_t radial_weight = gv[TERMS - 1];
                scalar_t radial_dt = scalar_t(0);
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t + radial_weight;
                    radial_weight = radial_weight * t + gv[k];
                }
                acc_t weight_gradient = acc_t(0);
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                acc_t u_0 = acc_t(0);
                // SCALAR_PATHS: an l1 = 0 path's edge gradient w x0 c g_o added per term, without v_j.
                const scalar_t weighted_input =
                    radial_weight * x[input_base + 0 + channel];

                 {
                    const scalar_t output_gradient = stage_row[256
                        + 0 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh_edge[1];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[1]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = stage_row[256
                        + 384 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh_edge[2];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[2]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = stage_row[256
                        + 768 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh_edge[3];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[3]
                            += weighted_input * scaled;
                    }
                }
                const scalar_t scaled_weight = radial_weight;
                weight_gradient +=
                    x[input_base + 0 + channel] * u_0;
                input_gradient_0_0 += scaled_weight * u_0;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    384 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (inside) {
                    const float4* gq =
                        reinterpret_cast<const float4*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 4; ++k) {
                        const float4 v = __ldg(gq + k);
                        gv[4 * k + 0] = v.x;
                        gv[4 * k + 1] = v.y;
                        gv[4 * k + 2] = v.z;
                        gv[4 * k + 3] = v.w;
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = scalar_t(0);
                }
                scalar_t radial_weight = gv[TERMS - 1];
                scalar_t radial_dt = scalar_t(0);
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t + radial_weight;
                    radial_weight = radial_weight * t + gv[k];
                }
                acc_t weight_gradient = acc_t(0);
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                acc_t u_0 = acc_t(0);
                acc_t u_1 = acc_t(0);
                acc_t u_2 = acc_t(0);
                acc_t v_0 = acc_t(0);

                 {
                    const scalar_t output_gradient = stage_row[256
                        + 128 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[0];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = stage_row[256
                        + 512 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[0];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = stage_row[256
                        + 896 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[0];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                }
                const scalar_t scaled_weight = radial_weight;
                weight_gradient +=
                    x[input_base + 128 + channel] * u_0;
                input_gradient_128_0 += scaled_weight * u_0;
                weight_gradient +=
                    x[input_base + 256 + channel] * u_1;
                input_gradient_128_1 += scaled_weight * u_1;
                weight_gradient +=
                    x[input_base + 384 + channel] * u_2;
                input_gradient_128_2 += scaled_weight * u_2;
                edge_gradient[0] +=
                    scaled_weight * v_0;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    512 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (inside) {
                    const float4* gq =
                        reinterpret_cast<const float4*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 4; ++k) {
                        const float4 v = __ldg(gq + k);
                        gv[4 * k + 0] = v.x;
                        gv[4 * k + 1] = v.y;
                        gv[4 * k + 2] = v.z;
                        gv[4 * k + 3] = v.w;
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = scalar_t(0);
                }
                scalar_t radial_weight = gv[TERMS - 1];
                scalar_t radial_dt = scalar_t(0);
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t + radial_weight;
                    radial_weight = radial_weight * t + gv[k];
                }
                acc_t weight_gradient = acc_t(0);
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                acc_t u_0 = acc_t(0);
                acc_t u_1 = acc_t(0);
                acc_t u_2 = acc_t(0);
                acc_t v_2 = acc_t(0);
                acc_t v_4 = acc_t(0);
                acc_t v_1 = acc_t(0);
                acc_t v_0 = acc_t(0);
                acc_t v_3 = acc_t(0);

                 {
                    const scalar_t output_gradient = stage_row[256
                        + 256 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a745903946p-1f);
                        u_0 -= scaled * edge_value;
                        v_2 -= scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient;
                        u_0 -= scaled * edge_value;
                        v_4 -= scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = stage_row[256
                        + 640 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a745903947p+0f);
                        u_1 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = stage_row[256
                        + 1024 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a745903946p-1f);
                        u_2 -= scaled * edge_value;
                        v_2 -= scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    }
                }
                const scalar_t scaled_weight = radial_weight;
                weight_gradient +=
                    x[input_base + 128 + channel] * u_0;
                input_gradient_128_0 += scaled_weight * u_0;
                weight_gradient +=
                    x[input_base + 256 + channel] * u_1;
                input_gradient_128_1 += scaled_weight * u_1;
                weight_gradient +=
                    x[input_base + 384 + channel] * u_2;
                input_gradient_128_2 += scaled_weight * u_2;
                edge_gradient[6] +=
                    scaled_weight * v_2;
                edge_gradient[8] +=
                    scaled_weight * v_4;
                edge_gradient[5] +=
                    scaled_weight * v_1;
                edge_gradient[4] +=
                    scaled_weight * v_0;
                edge_gradient[7] +=
                    scaled_weight * v_3;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    640 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (inside) {
                    const float4* gq =
                        reinterpret_cast<const float4*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 4; ++k) {
                        const float4 v = __ldg(gq + k);
                        gv[4 * k + 0] = v.x;
                        gv[4 * k + 1] = v.y;
                        gv[4 * k + 2] = v.z;
                        gv[4 * k + 3] = v.w;
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = scalar_t(0);
                }
                scalar_t radial_weight = gv[TERMS - 1];
                scalar_t radial_dt = scalar_t(0);
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t + radial_weight;
                    radial_weight = radial_weight * t + gv[k];
                }
                acc_t weight_gradient = acc_t(0);
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                acc_t u_0 = acc_t(0);
                // SCALAR_PATHS: an l1 = 0 path's edge gradient w x0 c g_o added per term, without v_j.
                const scalar_t weighted_input =
                    radial_weight * x[input_base + 0 + channel];

                 {
                    const scalar_t output_gradient = stage_row[1408
                        + 0 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[4]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = stage_row[1408
                        + 384 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[5]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = stage_row[1408
                        + 768 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[6]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = stage_row[1408
                        + 1152 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[7]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = stage_row[1408
                        + 1536 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[8]
                            += weighted_input * scaled;
                    }
                }
                const scalar_t scaled_weight = radial_weight;
                weight_gradient +=
                    x[input_base + 0 + channel] * u_0;
                input_gradient_0_0 += scaled_weight * u_0;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    768 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (inside) {
                    const float4* gq =
                        reinterpret_cast<const float4*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 4; ++k) {
                        const float4 v = __ldg(gq + k);
                        gv[4 * k + 0] = v.x;
                        gv[4 * k + 1] = v.y;
                        gv[4 * k + 2] = v.z;
                        gv[4 * k + 3] = v.w;
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = scalar_t(0);
                }
                scalar_t radial_weight = gv[TERMS - 1];
                scalar_t radial_dt = scalar_t(0);
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t + radial_weight;
                    radial_weight = radial_weight * t + gv[k];
                }
                acc_t weight_gradient = acc_t(0);
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                acc_t u_0 = acc_t(0);
                acc_t u_2 = acc_t(0);
                acc_t u_1 = acc_t(0);
                acc_t v_2 = acc_t(0);
                acc_t v_0 = acc_t(0);
                acc_t v_1 = acc_t(0);

                 {
                    const scalar_t output_gradient = stage_row[1408
                        + 128 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[3];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[1];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = stage_row[1408
                        + 512 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[2];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[1];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = stage_row[1408
                        + 896 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[1];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a74590261dp-1f);
                        u_0 -= scaled * edge_value;
                        v_0 -= scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[2];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a74590261dp+0f);
                        u_1 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[3];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a74590261dp-1f);
                        u_2 -= scaled * edge_value;
                        v_2 -= scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = stage_row[1408
                        + 1280 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[3];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[2];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = stage_row[1408
                        + 1664 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[1];
                        const scalar_t scaled = output_gradient;
                        u_0 -= scaled * edge_value;
                        v_0 -= scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[3];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    }
                }
                const scalar_t scaled_weight = radial_weight;
                weight_gradient +=
                    x[input_base + 128 + channel] * u_0;
                input_gradient_128_0 += scaled_weight * u_0;
                weight_gradient +=
                    x[input_base + 384 + channel] * u_2;
                input_gradient_128_2 += scaled_weight * u_2;
                weight_gradient +=
                    x[input_base + 256 + channel] * u_1;
                input_gradient_128_1 += scaled_weight * u_1;
                edge_gradient[3] +=
                    scaled_weight * v_2;
                edge_gradient[1] +=
                    scaled_weight * v_0;
                edge_gradient[2] +=
                    scaled_weight * v_1;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    896 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (inside) {
                    const float4* gq =
                        reinterpret_cast<const float4*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 4; ++k) {
                        const float4 v = __ldg(gq + k);
                        gv[4 * k + 0] = v.x;
                        gv[4 * k + 1] = v.y;
                        gv[4 * k + 2] = v.z;
                        gv[4 * k + 3] = v.w;
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = scalar_t(0);
                }
                scalar_t radial_weight = gv[TERMS - 1];
                scalar_t radial_dt = scalar_t(0);
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t + radial_weight;
                    radial_weight = radial_weight * t + gv[k];
                }
                acc_t weight_gradient = acc_t(0);
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                acc_t u_0 = acc_t(0);
                acc_t u_1 = acc_t(0);
                acc_t u_2 = acc_t(0);
                acc_t v_4 = acc_t(0);
                acc_t v_6 = acc_t(0);
                acc_t v_1 = acc_t(0);
                acc_t v_0 = acc_t(0);
                acc_t v_2 = acc_t(0);
                acc_t v_3 = acc_t(0);
                acc_t v_5 = acc_t(0);

                 {
                    const scalar_t output_gradient = stage_row[1408
                        + 256 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[13];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362485537p-2f);
                        u_0 -= scaled * edge_value;
                        v_4 -= scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[15];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3988e14092cf4p+0f);
                        u_0 -= scaled * edge_value;
                        v_6 -= scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[10];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[9];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3988e14092cf4p+0f);
                        u_2 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[11];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362485537p-2f);
                        u_2 -= scaled * edge_value;
                        v_2 -= scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = stage_row[1408
                        + 640 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[12];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.8c97ef43f812bp-1f);
                        u_0 -= scaled * edge_value;
                        v_3 -= scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[14];
                        const scalar_t scaled = output_gradient;
                        u_0 -= scaled * edge_value;
                        v_5 -= scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[11];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362485537p+0f);
                        u_1 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[10];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = stage_row[1408
                        + 1024 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[11];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f88efap+0f);
                        u_0 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[12];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.5775c544fff49p+0f);
                        u_1 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[13];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f88efap+0f);
                        u_2 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = stage_row[1408
                        + 1408 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[10];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[13];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362485537p+0f);
                        u_1 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[12];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.8c97ef43f812bp-1f);
                        u_2 -= scaled * edge_value;
                        v_3 -= scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[14];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_5 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = stage_row[1408
                        + 1792 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[9];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3988e14092cf4p+0f);
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[11];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362485537p-2f);
                        u_0 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[14];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_5 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[13];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362485537p-2f);
                        u_2 -= scaled * edge_value;
                        v_4 -= scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[15];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3988e14092cf4p+0f);
                        u_2 += scaled * edge_value;
                        v_6 += scaled * input_value;
                    }
                }
                const scalar_t scaled_weight = radial_weight;
                weight_gradient +=
                    x[input_base + 128 + channel] * u_0;
                input_gradient_128_0 += scaled_weight * u_0;
                weight_gradient +=
                    x[input_base + 256 + channel] * u_1;
                input_gradient_128_1 += scaled_weight * u_1;
                weight_gradient +=
                    x[input_base + 384 + channel] * u_2;
                input_gradient_128_2 += scaled_weight * u_2;
                edge_gradient[13] +=
                    scaled_weight * v_4;
                edge_gradient[15] +=
                    scaled_weight * v_6;
                edge_gradient[10] +=
                    scaled_weight * v_1;
                edge_gradient[9] +=
                    scaled_weight * v_0;
                edge_gradient[11] +=
                    scaled_weight * v_2;
                edge_gradient[12] +=
                    scaled_weight * v_3;
                edge_gradient[14] +=
                    scaled_weight * v_5;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    1024 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (inside) {
                    const float4* gq =
                        reinterpret_cast<const float4*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 4; ++k) {
                        const float4 v = __ldg(gq + k);
                        gv[4 * k + 0] = v.x;
                        gv[4 * k + 1] = v.y;
                        gv[4 * k + 2] = v.z;
                        gv[4 * k + 3] = v.w;
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = scalar_t(0);
                }
                scalar_t radial_weight = gv[TERMS - 1];
                scalar_t radial_dt = scalar_t(0);
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t + radial_weight;
                    radial_weight = radial_weight * t + gv[k];
                }
                acc_t weight_gradient = acc_t(0);
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                acc_t u_0 = acc_t(0);
                // SCALAR_PATHS: an l1 = 0 path's edge gradient w x0 c g_o added per term, without v_j.
                const scalar_t weighted_input =
                    radial_weight * x[input_base + 0 + channel];

                 {
                    const scalar_t output_gradient = stage_row[3328
                        + 0 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh_edge[9];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[9]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = stage_row[3328
                        + 256 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh_edge[10];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[10]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = stage_row[3328
                        + 512 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh_edge[11];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[11]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = stage_row[3328
                        + 768 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh_edge[12];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[12]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = stage_row[3328
                        + 1024 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh_edge[13];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[13]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = stage_row[3328
                        + 1280 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh_edge[14];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[14]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = stage_row[3328
                        + 1536 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh_edge[15];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[15]
                            += weighted_input * scaled;
                    }
                }
                const scalar_t scaled_weight = radial_weight;
                weight_gradient +=
                    x[input_base + 0 + channel] * u_0;
                input_gradient_0_0 += scaled_weight * u_0;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    1152 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (inside) {
                    const float4* gq =
                        reinterpret_cast<const float4*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 4; ++k) {
                        const float4 v = __ldg(gq + k);
                        gv[4 * k + 0] = v.x;
                        gv[4 * k + 1] = v.y;
                        gv[4 * k + 2] = v.z;
                        gv[4 * k + 3] = v.w;
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = scalar_t(0);
                }
                scalar_t radial_weight = gv[TERMS - 1];
                scalar_t radial_dt = scalar_t(0);
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t + radial_weight;
                    radial_weight = radial_weight * t + gv[k];
                }
                acc_t weight_gradient = acc_t(0);
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                acc_t u_0 = acc_t(0);
                acc_t u_2 = acc_t(0);
                acc_t u_1 = acc_t(0);
                acc_t v_4 = acc_t(0);
                acc_t v_0 = acc_t(0);
                acc_t v_3 = acc_t(0);
                acc_t v_1 = acc_t(0);
                acc_t v_2 = acc_t(0);

                 {
                    const scalar_t output_gradient = stage_row[3328
                        + 128 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3988e14091337p+0f);
                        u_0 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3988e14091337p+0f);
                        u_2 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = stage_row[3328
                        + 384 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = stage_row[3328
                        + 640 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f877f5p+0f);
                        u_0 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362483aa3p-2f);
                        u_0 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362483aa3p+0f);
                        u_1 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362483aa3p-2f);
                        u_2 -= scaled * edge_value;
                        v_0 -= scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = stage_row[3328
                        + 896 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.8c97ef43f609cp-1f);
                        u_0 -= scaled * edge_value;
                        v_1 -= scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.5775c544fe318p+0f);
                        u_1 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.8c97ef43f609cp-1f);
                        u_2 -= scaled * edge_value;
                        v_3 -= scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = stage_row[3328
                        + 1152 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362483aa3p-2f);
                        u_0 -= scaled * edge_value;
                        v_0 -= scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362483aa3p+0f);
                        u_1 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f877f5p+0f);
                        u_2 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362483aa3p-2f);
                        u_2 -= scaled * edge_value;
                        v_4 -= scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = stage_row[3328
                        + 1408 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient;
                        u_0 -= scaled * edge_value;
                        v_1 -= scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = stage_row[3328
                        + 1664 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3988e14091337p+0f);
                        u_0 -= scaled * edge_value;
                        v_0 -= scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3988e14091337p+0f);
                        u_2 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    }
                }
                const scalar_t scaled_weight = radial_weight;
                weight_gradient +=
                    x[input_base + 128 + channel] * u_0;
                input_gradient_128_0 += scaled_weight * u_0;
                weight_gradient +=
                    x[input_base + 384 + channel] * u_2;
                input_gradient_128_2 += scaled_weight * u_2;
                weight_gradient +=
                    x[input_base + 256 + channel] * u_1;
                input_gradient_128_1 += scaled_weight * u_1;
                edge_gradient[8] +=
                    scaled_weight * v_4;
                edge_gradient[4] +=
                    scaled_weight * v_0;
                edge_gradient[7] +=
                    scaled_weight * v_3;
                edge_gradient[5] +=
                    scaled_weight * v_1;
                edge_gradient[6] +=
                    scaled_weight * v_2;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
            atomicAdd(
                dx + input_base + 0 + channel,
                scalar_t(input_gradient_0_0));
            atomicAdd(
                dx + input_base + 128 + channel,
                scalar_t(input_gradient_128_0));
            atomicAdd(
                dx + input_base + 256 + channel,
                scalar_t(input_gradient_128_1));
            atomicAdd(
                dx + input_base + 384 + channel,
                scalar_t(input_gradient_128_2));
    }

    if (store) {  // per-lane values for the numbers check; off in the timing (the real kernel stores 17 per edge)
    #pragma unroll
    for (int i = 0; i < EDGE_DIM; ++i) out[(int64_t(e) * 32 + lane) * 17 + i] = edge_gradient[i];
    out[(int64_t(e) * 32 + lane) * 17 + 16] = radial_gradient;
    }
    __syncwarp();
    }
}
