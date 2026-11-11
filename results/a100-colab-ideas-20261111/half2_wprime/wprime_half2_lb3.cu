#include <cuda_fp16.h>
typedef int int32_t; typedef long long int64_t;
constexpr int INPUT_DIM = 512, EDGE_DIM = 16, WEIGHT_DIM = 1280, CHANNEL_DIM = 128, TERMS = 4, PADDED_TERMS = 4;
constexpr int LOGICAL_GROUP_SIZE = 32;
typedef float scalar_t;
typedef float acc_t;

#define Z2 __half2(__half2_raw{0, 0})
#define H2C(bits) __half2(__half2_raw{bits, bits})
#define LX(k) __halves2half2(xh[input_base0 + (k)], xh[input_base1 + (k)])
extern "C" __global__ void __launch_bounds__(256, 3) census_wprime(
    const __half* xh, float* dx, const __half* __restrict__ gh, const int32_t* senders, const int32_t* slabs,
    const float* ts, const float* scs, const unsigned* flags, const __half2* shs, const __half2* dout, float* out,
    int n_pairs) {
    __shared__ __align__(16) __half2 stage_row2[5120];
    __shared__ __align__(16) __half2 stage_esh2[8][EDGE_DIM];
    for (int i = threadIdx.x; i < 5120; i += blockDim.x) stage_row2[i] = dout[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int p = blockIdx.x * 8 + w; p < n_pairs; p += gridDim.x * 8) {
    const int e0 = 2 * p, e1 = 2 * p + 1;
    if (lane < EDGE_DIM) stage_esh2[w][lane] = shs[int64_t(p) * EDGE_DIM + lane];
    __syncwarp();
    const int64_t input_base0 = int64_t(senders[e0]) * INPUT_DIM, input_base1 = int64_t(senders[e1]) * INPUT_DIM;
    const __half2* sh_edge2 = stage_esh2[w];
    // an edge outside the table reads a zero slab; the slope is masked per half (no branch on two edges)
    const __half* g0 = gh + int64_t(slabs[e0]) * PADDED_TERMS * WEIGHT_DIM;
    const __half* g1 = gh + int64_t(slabs[e1]) * PADDED_TERMS * WEIGHT_DIM;
    const __half2 t2 = __floats2half2_rn(ts[e0], ts[e1]);
    const __half2 sc2 = __floats2half2_rn((flags[e0] & 2u) ? scs[e0] : 0.f, (flags[e1] & 2u) ? scs[e1] : 0.f);
    __half2 edge_gradient[EDGE_DIM];
    #pragma unroll
    for (int i = 0; i < EDGE_DIM; ++i) edge_gradient[i] = Z2;
    __half2 radial_gradient = Z2;
    const int64_t weight_base = int64_t(e0) * WEIGHT_DIM;
    for (int channel = lane; channel < CHANNEL_DIM;
         channel += LOGICAL_GROUP_SIZE) {
            __half2 input_gradient_0_0 = Z2;
            __half2 input_gradient_128_0 = Z2;
            __half2 input_gradient_128_1 = Z2;
            __half2 input_gradient_128_2 = Z2;

        
             {
                const int weight_index =
                    0 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                __half2 gv[PADDED_TERMS];
                {
                    const uint2 a = __ldg(reinterpret_cast<const uint2*>(g0 + weight_index * PADDED_TERMS));
                    const uint2 b = __ldg(reinterpret_cast<const uint2*>(g1 + weight_index * PADDED_TERMS));
                    const unsigned q[4] = {__byte_perm(a.x, b.x, 0x5410), __byte_perm(a.x, b.x, 0x7632),
                                           __byte_perm(a.y, b.y, 0x5410), __byte_perm(a.y, b.y, 0x7632)};
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = *reinterpret_cast<const __half2*>(&q[k]);
                }
                __half2 radial_weight = gv[TERMS - 1];
                __half2 radial_dt = Z2;
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t2 + radial_weight;
                    radial_weight = radial_weight * t2 + gv[k];
                }
                __half2 weight_gradient = Z2;
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                __half2 u_0 = Z2;
                // SCALAR_PATHS: an l1 = 0 path's edge gradient w x0 c g_o added per term, without v_j.
                const __half2 weighted_input =
                    radial_weight * LX(0 + channel);

                 {
                    const __half2 output_gradient = stage_row2[0
                        + 0 + channel]; {
                        const __half2 input_value =
                            LX(0 + channel);
                        const __half2 edge_value =
                            sh_edge2[0];
                        const __half2 scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[0]
                            += weighted_input * scaled;
                    }
                }
                const __half2 scaled_weight = radial_weight;
                weight_gradient +=
                    LX(0 + channel) * u_0;
                input_gradient_0_0 += scaled_weight * u_0;
                {
                    __half2 slope = Z2;
                    slope = radial_dt * sc2;
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    128 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                __half2 gv[PADDED_TERMS];
                {
                    const uint2 a = __ldg(reinterpret_cast<const uint2*>(g0 + weight_index * PADDED_TERMS));
                    const uint2 b = __ldg(reinterpret_cast<const uint2*>(g1 + weight_index * PADDED_TERMS));
                    const unsigned q[4] = {__byte_perm(a.x, b.x, 0x5410), __byte_perm(a.x, b.x, 0x7632),
                                           __byte_perm(a.y, b.y, 0x5410), __byte_perm(a.y, b.y, 0x7632)};
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = *reinterpret_cast<const __half2*>(&q[k]);
                }
                __half2 radial_weight = gv[TERMS - 1];
                __half2 radial_dt = Z2;
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t2 + radial_weight;
                    radial_weight = radial_weight * t2 + gv[k];
                }
                __half2 weight_gradient = Z2;
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                __half2 u_0 = Z2;
                __half2 u_1 = Z2;
                __half2 u_2 = Z2;
                __half2 v_0 = Z2;
                __half2 v_1 = Z2;
                __half2 v_2 = Z2;

                 {
                    const __half2 output_gradient = stage_row2[0
                        + 128 + channel]; {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[1];
                        const __half2 scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(256 + channel);
                        const __half2 edge_value =
                            sh_edge2[2];
                        const __half2 scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[3];
                        const __half2 scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    }
                }
                const __half2 scaled_weight = radial_weight;
                weight_gradient +=
                    LX(128 + channel) * u_0;
                input_gradient_128_0 += scaled_weight * u_0;
                weight_gradient +=
                    LX(256 + channel) * u_1;
                input_gradient_128_1 += scaled_weight * u_1;
                weight_gradient +=
                    LX(384 + channel) * u_2;
                input_gradient_128_2 += scaled_weight * u_2;
                edge_gradient[1] +=
                    scaled_weight * v_0;
                edge_gradient[2] +=
                    scaled_weight * v_1;
                edge_gradient[3] +=
                    scaled_weight * v_2;
                {
                    __half2 slope = Z2;
                    slope = radial_dt * sc2;
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    256 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                __half2 gv[PADDED_TERMS];
                {
                    const uint2 a = __ldg(reinterpret_cast<const uint2*>(g0 + weight_index * PADDED_TERMS));
                    const uint2 b = __ldg(reinterpret_cast<const uint2*>(g1 + weight_index * PADDED_TERMS));
                    const unsigned q[4] = {__byte_perm(a.x, b.x, 0x5410), __byte_perm(a.x, b.x, 0x7632),
                                           __byte_perm(a.y, b.y, 0x5410), __byte_perm(a.y, b.y, 0x7632)};
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = *reinterpret_cast<const __half2*>(&q[k]);
                }
                __half2 radial_weight = gv[TERMS - 1];
                __half2 radial_dt = Z2;
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t2 + radial_weight;
                    radial_weight = radial_weight * t2 + gv[k];
                }
                __half2 weight_gradient = Z2;
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                __half2 u_0 = Z2;
                // SCALAR_PATHS: an l1 = 0 path's edge gradient w x0 c g_o added per term, without v_j.
                const __half2 weighted_input =
                    radial_weight * LX(0 + channel);

                 {
                    const __half2 output_gradient = stage_row2[256
                        + 0 + channel]; {
                        const __half2 input_value =
                            LX(0 + channel);
                        const __half2 edge_value =
                            sh_edge2[1];
                        const __half2 scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[1]
                            += weighted_input * scaled;
                    }
                } {
                    const __half2 output_gradient = stage_row2[256
                        + 384 + channel]; {
                        const __half2 input_value =
                            LX(0 + channel);
                        const __half2 edge_value =
                            sh_edge2[2];
                        const __half2 scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[2]
                            += weighted_input * scaled;
                    }
                } {
                    const __half2 output_gradient = stage_row2[256
                        + 768 + channel]; {
                        const __half2 input_value =
                            LX(0 + channel);
                        const __half2 edge_value =
                            sh_edge2[3];
                        const __half2 scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[3]
                            += weighted_input * scaled;
                    }
                }
                const __half2 scaled_weight = radial_weight;
                weight_gradient +=
                    LX(0 + channel) * u_0;
                input_gradient_0_0 += scaled_weight * u_0;
                {
                    __half2 slope = Z2;
                    slope = radial_dt * sc2;
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    384 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                __half2 gv[PADDED_TERMS];
                {
                    const uint2 a = __ldg(reinterpret_cast<const uint2*>(g0 + weight_index * PADDED_TERMS));
                    const uint2 b = __ldg(reinterpret_cast<const uint2*>(g1 + weight_index * PADDED_TERMS));
                    const unsigned q[4] = {__byte_perm(a.x, b.x, 0x5410), __byte_perm(a.x, b.x, 0x7632),
                                           __byte_perm(a.y, b.y, 0x5410), __byte_perm(a.y, b.y, 0x7632)};
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = *reinterpret_cast<const __half2*>(&q[k]);
                }
                __half2 radial_weight = gv[TERMS - 1];
                __half2 radial_dt = Z2;
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t2 + radial_weight;
                    radial_weight = radial_weight * t2 + gv[k];
                }
                __half2 weight_gradient = Z2;
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                __half2 u_0 = Z2;
                __half2 u_1 = Z2;
                __half2 u_2 = Z2;
                __half2 v_0 = Z2;

                 {
                    const __half2 output_gradient = stage_row2[256
                        + 128 + channel]; {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[0];
                        const __half2 scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const __half2 output_gradient = stage_row2[256
                        + 512 + channel]; {
                        const __half2 input_value =
                            LX(256 + channel);
                        const __half2 edge_value =
                            sh_edge2[0];
                        const __half2 scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const __half2 output_gradient = stage_row2[256
                        + 896 + channel]; {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[0];
                        const __half2 scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                }
                const __half2 scaled_weight = radial_weight;
                weight_gradient +=
                    LX(128 + channel) * u_0;
                input_gradient_128_0 += scaled_weight * u_0;
                weight_gradient +=
                    LX(256 + channel) * u_1;
                input_gradient_128_1 += scaled_weight * u_1;
                weight_gradient +=
                    LX(384 + channel) * u_2;
                input_gradient_128_2 += scaled_weight * u_2;
                edge_gradient[0] +=
                    scaled_weight * v_0;
                {
                    __half2 slope = Z2;
                    slope = radial_dt * sc2;
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    512 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                __half2 gv[PADDED_TERMS];
                {
                    const uint2 a = __ldg(reinterpret_cast<const uint2*>(g0 + weight_index * PADDED_TERMS));
                    const uint2 b = __ldg(reinterpret_cast<const uint2*>(g1 + weight_index * PADDED_TERMS));
                    const unsigned q[4] = {__byte_perm(a.x, b.x, 0x5410), __byte_perm(a.x, b.x, 0x7632),
                                           __byte_perm(a.y, b.y, 0x5410), __byte_perm(a.y, b.y, 0x7632)};
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = *reinterpret_cast<const __half2*>(&q[k]);
                }
                __half2 radial_weight = gv[TERMS - 1];
                __half2 radial_dt = Z2;
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t2 + radial_weight;
                    radial_weight = radial_weight * t2 + gv[k];
                }
                __half2 weight_gradient = Z2;
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                __half2 u_0 = Z2;
                __half2 u_1 = Z2;
                __half2 u_2 = Z2;
                __half2 v_2 = Z2;
                __half2 v_4 = Z2;
                __half2 v_1 = Z2;
                __half2 v_0 = Z2;
                __half2 v_3 = Z2;

                 {
                    const __half2 output_gradient = stage_row2[256
                        + 256 + channel]; {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[6];
                        const __half2 scaled = output_gradient * H2C(0x389e);
                        u_0 -= scaled * edge_value;
                        v_2 -= scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[8];
                        const __half2 scaled = output_gradient;
                        u_0 -= scaled * edge_value;
                        v_4 -= scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(256 + channel);
                        const __half2 edge_value =
                            sh_edge2[5];
                        const __half2 scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[4];
                        const __half2 scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const __half2 output_gradient = stage_row2[256
                        + 640 + channel]; {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[5];
                        const __half2 scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(256 + channel);
                        const __half2 edge_value =
                            sh_edge2[6];
                        const __half2 scaled = output_gradient * H2C(0x3c9e);
                        u_1 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[7];
                        const __half2 scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    }
                } {
                    const __half2 output_gradient = stage_row2[256
                        + 1024 + channel]; {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[4];
                        const __half2 scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(256 + channel);
                        const __half2 edge_value =
                            sh_edge2[7];
                        const __half2 scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[6];
                        const __half2 scaled = output_gradient * H2C(0x389e);
                        u_2 -= scaled * edge_value;
                        v_2 -= scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[8];
                        const __half2 scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    }
                }
                const __half2 scaled_weight = radial_weight;
                weight_gradient +=
                    LX(128 + channel) * u_0;
                input_gradient_128_0 += scaled_weight * u_0;
                weight_gradient +=
                    LX(256 + channel) * u_1;
                input_gradient_128_1 += scaled_weight * u_1;
                weight_gradient +=
                    LX(384 + channel) * u_2;
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
                {
                    __half2 slope = Z2;
                    slope = radial_dt * sc2;
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    640 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                __half2 gv[PADDED_TERMS];
                {
                    const uint2 a = __ldg(reinterpret_cast<const uint2*>(g0 + weight_index * PADDED_TERMS));
                    const uint2 b = __ldg(reinterpret_cast<const uint2*>(g1 + weight_index * PADDED_TERMS));
                    const unsigned q[4] = {__byte_perm(a.x, b.x, 0x5410), __byte_perm(a.x, b.x, 0x7632),
                                           __byte_perm(a.y, b.y, 0x5410), __byte_perm(a.y, b.y, 0x7632)};
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = *reinterpret_cast<const __half2*>(&q[k]);
                }
                __half2 radial_weight = gv[TERMS - 1];
                __half2 radial_dt = Z2;
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t2 + radial_weight;
                    radial_weight = radial_weight * t2 + gv[k];
                }
                __half2 weight_gradient = Z2;
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                __half2 u_0 = Z2;
                // SCALAR_PATHS: an l1 = 0 path's edge gradient w x0 c g_o added per term, without v_j.
                const __half2 weighted_input =
                    radial_weight * LX(0 + channel);

                 {
                    const __half2 output_gradient = stage_row2[1408
                        + 0 + channel]; {
                        const __half2 input_value =
                            LX(0 + channel);
                        const __half2 edge_value =
                            sh_edge2[4];
                        const __half2 scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[4]
                            += weighted_input * scaled;
                    }
                } {
                    const __half2 output_gradient = stage_row2[1408
                        + 384 + channel]; {
                        const __half2 input_value =
                            LX(0 + channel);
                        const __half2 edge_value =
                            sh_edge2[5];
                        const __half2 scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[5]
                            += weighted_input * scaled;
                    }
                } {
                    const __half2 output_gradient = stage_row2[1408
                        + 768 + channel]; {
                        const __half2 input_value =
                            LX(0 + channel);
                        const __half2 edge_value =
                            sh_edge2[6];
                        const __half2 scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[6]
                            += weighted_input * scaled;
                    }
                } {
                    const __half2 output_gradient = stage_row2[1408
                        + 1152 + channel]; {
                        const __half2 input_value =
                            LX(0 + channel);
                        const __half2 edge_value =
                            sh_edge2[7];
                        const __half2 scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[7]
                            += weighted_input * scaled;
                    }
                } {
                    const __half2 output_gradient = stage_row2[1408
                        + 1536 + channel]; {
                        const __half2 input_value =
                            LX(0 + channel);
                        const __half2 edge_value =
                            sh_edge2[8];
                        const __half2 scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[8]
                            += weighted_input * scaled;
                    }
                }
                const __half2 scaled_weight = radial_weight;
                weight_gradient +=
                    LX(0 + channel) * u_0;
                input_gradient_0_0 += scaled_weight * u_0;
                {
                    __half2 slope = Z2;
                    slope = radial_dt * sc2;
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    768 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                __half2 gv[PADDED_TERMS];
                {
                    const uint2 a = __ldg(reinterpret_cast<const uint2*>(g0 + weight_index * PADDED_TERMS));
                    const uint2 b = __ldg(reinterpret_cast<const uint2*>(g1 + weight_index * PADDED_TERMS));
                    const unsigned q[4] = {__byte_perm(a.x, b.x, 0x5410), __byte_perm(a.x, b.x, 0x7632),
                                           __byte_perm(a.y, b.y, 0x5410), __byte_perm(a.y, b.y, 0x7632)};
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = *reinterpret_cast<const __half2*>(&q[k]);
                }
                __half2 radial_weight = gv[TERMS - 1];
                __half2 radial_dt = Z2;
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t2 + radial_weight;
                    radial_weight = radial_weight * t2 + gv[k];
                }
                __half2 weight_gradient = Z2;
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                __half2 u_0 = Z2;
                __half2 u_2 = Z2;
                __half2 u_1 = Z2;
                __half2 v_2 = Z2;
                __half2 v_0 = Z2;
                __half2 v_1 = Z2;

                 {
                    const __half2 output_gradient = stage_row2[1408
                        + 128 + channel]; {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[3];
                        const __half2 scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[1];
                        const __half2 scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const __half2 output_gradient = stage_row2[1408
                        + 512 + channel]; {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[2];
                        const __half2 scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(256 + channel);
                        const __half2 edge_value =
                            sh_edge2[1];
                        const __half2 scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const __half2 output_gradient = stage_row2[1408
                        + 896 + channel]; {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[1];
                        const __half2 scaled = output_gradient * H2C(0x389e);
                        u_0 -= scaled * edge_value;
                        v_0 -= scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(256 + channel);
                        const __half2 edge_value =
                            sh_edge2[2];
                        const __half2 scaled = output_gradient * H2C(0x3c9e);
                        u_1 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[3];
                        const __half2 scaled = output_gradient * H2C(0x389e);
                        u_2 -= scaled * edge_value;
                        v_2 -= scaled * input_value;
                    }
                } {
                    const __half2 output_gradient = stage_row2[1408
                        + 1280 + channel]; {
                        const __half2 input_value =
                            LX(256 + channel);
                        const __half2 edge_value =
                            sh_edge2[3];
                        const __half2 scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[2];
                        const __half2 scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    }
                } {
                    const __half2 output_gradient = stage_row2[1408
                        + 1664 + channel]; {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[1];
                        const __half2 scaled = output_gradient;
                        u_0 -= scaled * edge_value;
                        v_0 -= scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[3];
                        const __half2 scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    }
                }
                const __half2 scaled_weight = radial_weight;
                weight_gradient +=
                    LX(128 + channel) * u_0;
                input_gradient_128_0 += scaled_weight * u_0;
                weight_gradient +=
                    LX(384 + channel) * u_2;
                input_gradient_128_2 += scaled_weight * u_2;
                weight_gradient +=
                    LX(256 + channel) * u_1;
                input_gradient_128_1 += scaled_weight * u_1;
                edge_gradient[3] +=
                    scaled_weight * v_2;
                edge_gradient[1] +=
                    scaled_weight * v_0;
                edge_gradient[2] +=
                    scaled_weight * v_1;
                {
                    __half2 slope = Z2;
                    slope = radial_dt * sc2;
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    896 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                __half2 gv[PADDED_TERMS];
                {
                    const uint2 a = __ldg(reinterpret_cast<const uint2*>(g0 + weight_index * PADDED_TERMS));
                    const uint2 b = __ldg(reinterpret_cast<const uint2*>(g1 + weight_index * PADDED_TERMS));
                    const unsigned q[4] = {__byte_perm(a.x, b.x, 0x5410), __byte_perm(a.x, b.x, 0x7632),
                                           __byte_perm(a.y, b.y, 0x5410), __byte_perm(a.y, b.y, 0x7632)};
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = *reinterpret_cast<const __half2*>(&q[k]);
                }
                __half2 radial_weight = gv[TERMS - 1];
                __half2 radial_dt = Z2;
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t2 + radial_weight;
                    radial_weight = radial_weight * t2 + gv[k];
                }
                __half2 weight_gradient = Z2;
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                __half2 u_0 = Z2;
                __half2 u_1 = Z2;
                __half2 u_2 = Z2;
                __half2 v_4 = Z2;
                __half2 v_6 = Z2;
                __half2 v_1 = Z2;
                __half2 v_0 = Z2;
                __half2 v_2 = Z2;
                __half2 v_3 = Z2;
                __half2 v_5 = Z2;

                 {
                    const __half2 output_gradient = stage_row2[1408
                        + 256 + channel]; {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[13];
                        const __half2 scaled = output_gradient * H2C(0x350f);
                        u_0 -= scaled * edge_value;
                        v_4 -= scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[15];
                        const __half2 scaled = output_gradient * H2C(0x3ce6);
                        u_0 -= scaled * edge_value;
                        v_6 -= scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(256 + channel);
                        const __half2 edge_value =
                            sh_edge2[10];
                        const __half2 scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[9];
                        const __half2 scaled = output_gradient * H2C(0x3ce6);
                        u_2 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[11];
                        const __half2 scaled = output_gradient * H2C(0x350f);
                        u_2 -= scaled * edge_value;
                        v_2 -= scaled * input_value;
                    }
                } {
                    const __half2 output_gradient = stage_row2[1408
                        + 640 + channel]; {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[12];
                        const __half2 scaled = output_gradient * H2C(0x3a32);
                        u_0 -= scaled * edge_value;
                        v_3 -= scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[14];
                        const __half2 scaled = output_gradient;
                        u_0 -= scaled * edge_value;
                        v_5 -= scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(256 + channel);
                        const __half2 edge_value =
                            sh_edge2[11];
                        const __half2 scaled = output_gradient * H2C(0x3d0f);
                        u_1 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[10];
                        const __half2 scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    }
                } {
                    const __half2 output_gradient = stage_row2[1408
                        + 1024 + channel]; {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[11];
                        const __half2 scaled = output_gradient * H2C(0x3c62);
                        u_0 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(256 + channel);
                        const __half2 edge_value =
                            sh_edge2[12];
                        const __half2 scaled = output_gradient * H2C(0x3d5e);
                        u_1 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[13];
                        const __half2 scaled = output_gradient * H2C(0x3c62);
                        u_2 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    }
                } {
                    const __half2 output_gradient = stage_row2[1408
                        + 1408 + channel]; {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[10];
                        const __half2 scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(256 + channel);
                        const __half2 edge_value =
                            sh_edge2[13];
                        const __half2 scaled = output_gradient * H2C(0x3d0f);
                        u_1 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[12];
                        const __half2 scaled = output_gradient * H2C(0x3a32);
                        u_2 -= scaled * edge_value;
                        v_3 -= scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[14];
                        const __half2 scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_5 += scaled * input_value;
                    }
                } {
                    const __half2 output_gradient = stage_row2[1408
                        + 1792 + channel]; {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[9];
                        const __half2 scaled = output_gradient * H2C(0x3ce6);
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[11];
                        const __half2 scaled = output_gradient * H2C(0x350f);
                        u_0 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(256 + channel);
                        const __half2 edge_value =
                            sh_edge2[14];
                        const __half2 scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_5 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[13];
                        const __half2 scaled = output_gradient * H2C(0x350f);
                        u_2 -= scaled * edge_value;
                        v_4 -= scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[15];
                        const __half2 scaled = output_gradient * H2C(0x3ce6);
                        u_2 += scaled * edge_value;
                        v_6 += scaled * input_value;
                    }
                }
                const __half2 scaled_weight = radial_weight;
                weight_gradient +=
                    LX(128 + channel) * u_0;
                input_gradient_128_0 += scaled_weight * u_0;
                weight_gradient +=
                    LX(256 + channel) * u_1;
                input_gradient_128_1 += scaled_weight * u_1;
                weight_gradient +=
                    LX(384 + channel) * u_2;
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
                {
                    __half2 slope = Z2;
                    slope = radial_dt * sc2;
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    1024 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                __half2 gv[PADDED_TERMS];
                {
                    const uint2 a = __ldg(reinterpret_cast<const uint2*>(g0 + weight_index * PADDED_TERMS));
                    const uint2 b = __ldg(reinterpret_cast<const uint2*>(g1 + weight_index * PADDED_TERMS));
                    const unsigned q[4] = {__byte_perm(a.x, b.x, 0x5410), __byte_perm(a.x, b.x, 0x7632),
                                           __byte_perm(a.y, b.y, 0x5410), __byte_perm(a.y, b.y, 0x7632)};
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = *reinterpret_cast<const __half2*>(&q[k]);
                }
                __half2 radial_weight = gv[TERMS - 1];
                __half2 radial_dt = Z2;
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t2 + radial_weight;
                    radial_weight = radial_weight * t2 + gv[k];
                }
                __half2 weight_gradient = Z2;
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                __half2 u_0 = Z2;
                // SCALAR_PATHS: an l1 = 0 path's edge gradient w x0 c g_o added per term, without v_j.
                const __half2 weighted_input =
                    radial_weight * LX(0 + channel);

                 {
                    const __half2 output_gradient = stage_row2[3328
                        + 0 + channel]; {
                        const __half2 input_value =
                            LX(0 + channel);
                        const __half2 edge_value =
                            sh_edge2[9];
                        const __half2 scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[9]
                            += weighted_input * scaled;
                    }
                } {
                    const __half2 output_gradient = stage_row2[3328
                        + 256 + channel]; {
                        const __half2 input_value =
                            LX(0 + channel);
                        const __half2 edge_value =
                            sh_edge2[10];
                        const __half2 scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[10]
                            += weighted_input * scaled;
                    }
                } {
                    const __half2 output_gradient = stage_row2[3328
                        + 512 + channel]; {
                        const __half2 input_value =
                            LX(0 + channel);
                        const __half2 edge_value =
                            sh_edge2[11];
                        const __half2 scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[11]
                            += weighted_input * scaled;
                    }
                } {
                    const __half2 output_gradient = stage_row2[3328
                        + 768 + channel]; {
                        const __half2 input_value =
                            LX(0 + channel);
                        const __half2 edge_value =
                            sh_edge2[12];
                        const __half2 scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[12]
                            += weighted_input * scaled;
                    }
                } {
                    const __half2 output_gradient = stage_row2[3328
                        + 1024 + channel]; {
                        const __half2 input_value =
                            LX(0 + channel);
                        const __half2 edge_value =
                            sh_edge2[13];
                        const __half2 scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[13]
                            += weighted_input * scaled;
                    }
                } {
                    const __half2 output_gradient = stage_row2[3328
                        + 1280 + channel]; {
                        const __half2 input_value =
                            LX(0 + channel);
                        const __half2 edge_value =
                            sh_edge2[14];
                        const __half2 scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[14]
                            += weighted_input * scaled;
                    }
                } {
                    const __half2 output_gradient = stage_row2[3328
                        + 1536 + channel]; {
                        const __half2 input_value =
                            LX(0 + channel);
                        const __half2 edge_value =
                            sh_edge2[15];
                        const __half2 scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[15]
                            += weighted_input * scaled;
                    }
                }
                const __half2 scaled_weight = radial_weight;
                weight_gradient +=
                    LX(0 + channel) * u_0;
                input_gradient_0_0 += scaled_weight * u_0;
                {
                    __half2 slope = Z2;
                    slope = radial_dt * sc2;
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    1152 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                __half2 gv[PADDED_TERMS];
                {
                    const uint2 a = __ldg(reinterpret_cast<const uint2*>(g0 + weight_index * PADDED_TERMS));
                    const uint2 b = __ldg(reinterpret_cast<const uint2*>(g1 + weight_index * PADDED_TERMS));
                    const unsigned q[4] = {__byte_perm(a.x, b.x, 0x5410), __byte_perm(a.x, b.x, 0x7632),
                                           __byte_perm(a.y, b.y, 0x5410), __byte_perm(a.y, b.y, 0x7632)};
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = *reinterpret_cast<const __half2*>(&q[k]);
                }
                __half2 radial_weight = gv[TERMS - 1];
                __half2 radial_dt = Z2;
                #pragma unroll
                for (int k = TERMS - 2; k >= 0; --k) {
                    radial_dt = radial_dt * t2 + radial_weight;
                    radial_weight = radial_weight * t2 + gv[k];
                }
                __half2 weight_gradient = Z2;
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                __half2 u_0 = Z2;
                __half2 u_2 = Z2;
                __half2 u_1 = Z2;
                __half2 v_4 = Z2;
                __half2 v_0 = Z2;
                __half2 v_3 = Z2;
                __half2 v_1 = Z2;
                __half2 v_2 = Z2;

                 {
                    const __half2 output_gradient = stage_row2[3328
                        + 128 + channel]; {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[8];
                        const __half2 scaled = output_gradient * H2C(0x3ce6);
                        u_0 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[4];
                        const __half2 scaled = output_gradient * H2C(0x3ce6);
                        u_2 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const __half2 output_gradient = stage_row2[3328
                        + 384 + channel]; {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[7];
                        const __half2 scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(256 + channel);
                        const __half2 edge_value =
                            sh_edge2[4];
                        const __half2 scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[5];
                        const __half2 scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    }
                } {
                    const __half2 output_gradient = stage_row2[3328
                        + 640 + channel]; {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[6];
                        const __half2 scaled = output_gradient * H2C(0x3c62);
                        u_0 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[8];
                        const __half2 scaled = output_gradient * H2C(0x350f);
                        u_0 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(256 + channel);
                        const __half2 edge_value =
                            sh_edge2[5];
                        const __half2 scaled = output_gradient * H2C(0x3d0f);
                        u_1 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[4];
                        const __half2 scaled = output_gradient * H2C(0x350f);
                        u_2 -= scaled * edge_value;
                        v_0 -= scaled * input_value;
                    }
                } {
                    const __half2 output_gradient = stage_row2[3328
                        + 896 + channel]; {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[5];
                        const __half2 scaled = output_gradient * H2C(0x3a32);
                        u_0 -= scaled * edge_value;
                        v_1 -= scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(256 + channel);
                        const __half2 edge_value =
                            sh_edge2[6];
                        const __half2 scaled = output_gradient * H2C(0x3d5e);
                        u_1 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[7];
                        const __half2 scaled = output_gradient * H2C(0x3a32);
                        u_2 -= scaled * edge_value;
                        v_3 -= scaled * input_value;
                    }
                } {
                    const __half2 output_gradient = stage_row2[3328
                        + 1152 + channel]; {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[4];
                        const __half2 scaled = output_gradient * H2C(0x350f);
                        u_0 -= scaled * edge_value;
                        v_0 -= scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(256 + channel);
                        const __half2 edge_value =
                            sh_edge2[7];
                        const __half2 scaled = output_gradient * H2C(0x3d0f);
                        u_1 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[6];
                        const __half2 scaled = output_gradient * H2C(0x3c62);
                        u_2 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[8];
                        const __half2 scaled = output_gradient * H2C(0x350f);
                        u_2 -= scaled * edge_value;
                        v_4 -= scaled * input_value;
                    }
                } {
                    const __half2 output_gradient = stage_row2[3328
                        + 1408 + channel]; {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[5];
                        const __half2 scaled = output_gradient;
                        u_0 -= scaled * edge_value;
                        v_1 -= scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(256 + channel);
                        const __half2 edge_value =
                            sh_edge2[8];
                        const __half2 scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[7];
                        const __half2 scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    }
                } {
                    const __half2 output_gradient = stage_row2[3328
                        + 1664 + channel]; {
                        const __half2 input_value =
                            LX(128 + channel);
                        const __half2 edge_value =
                            sh_edge2[4];
                        const __half2 scaled = output_gradient * H2C(0x3ce6);
                        u_0 -= scaled * edge_value;
                        v_0 -= scaled * input_value;
                    } {
                        const __half2 input_value =
                            LX(384 + channel);
                        const __half2 edge_value =
                            sh_edge2[8];
                        const __half2 scaled = output_gradient * H2C(0x3ce6);
                        u_2 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    }
                }
                const __half2 scaled_weight = radial_weight;
                weight_gradient +=
                    LX(128 + channel) * u_0;
                input_gradient_128_0 += scaled_weight * u_0;
                weight_gradient +=
                    LX(384 + channel) * u_2;
                input_gradient_128_2 += scaled_weight * u_2;
                weight_gradient +=
                    LX(256 + channel) * u_1;
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
                {
                    __half2 slope = Z2;
                    slope = radial_dt * sc2;
                    radial_gradient += weight_gradient * slope;
                }
            }
            atomicAdd(dx + input_base0 + 0 + channel, __low2float(input_gradient_0_0));
            atomicAdd(dx + input_base1 + 0 + channel, __high2float(input_gradient_0_0));
            atomicAdd(dx + input_base0 + 128 + channel, __low2float(input_gradient_128_0));
            atomicAdd(dx + input_base1 + 128 + channel, __high2float(input_gradient_128_0));
            atomicAdd(dx + input_base0 + 256 + channel, __low2float(input_gradient_128_1));
            atomicAdd(dx + input_base1 + 256 + channel, __high2float(input_gradient_128_1));
            atomicAdd(dx + input_base0 + 384 + channel, __low2float(input_gradient_128_2));
            atomicAdd(dx + input_base1 + 384 + channel, __high2float(input_gradient_128_2));
    }

    #pragma unroll
    for (int i = 0; i < EDGE_DIM; ++i) {
        out[(int64_t(e0) * 32 + lane) * 17 + i] = __low2float(edge_gradient[i]);
        out[(int64_t(e1) * 32 + lane) * 17 + i] = __high2float(edge_gradient[i]);
    }
    out[(int64_t(e0) * 32 + lane) * 17 + 16] = __low2float(radial_gradient);
    out[(int64_t(e1) * 32 + lane) * 17 + 16] = __high2float(radial_gradient);
    __syncwarp();
    }
}
