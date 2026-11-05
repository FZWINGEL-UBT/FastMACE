// oeq_projected_backward_wprime (FastMACE opt-9 OPT9-T03), adapted from the vendored
// oeq_projected_backward (OpenEquivariance PR 223, BSD 3-Clause, see NOTICE.md).
using int64_t = signed long long;
using int32_t = int;
using scalar_t = float;
using acc_t = float;

constexpr int INPUT_DIM = 512;
constexpr int EDGE_DIM = 16;
constexpr int OUTPUT_DIM = 5120;
constexpr int WEIGHT_DIM = 1280;
constexpr int CHANNEL_DIM = 128;
constexpr int LOGICAL_GROUP_SIZE =
    32;
constexpr int TERMS = 6;
constexpr int PADDED_TERMS = 6;  // a q-major G' row: TERMS, then zero terms (SLOPE_VECTOR)
constexpr unsigned int FULL_MASK = 0xffffffffu;



// One logical thread group owns an edge; each lane owns channels separated by LOGICAL_GROUP_SIZE.
// dx and dsh as in oeq_projected_backward; in place of dweights, dr[e] = sum over the edge's weights of
// dweights * w'(r_e), w' from the derivative table gprime [pairs * pieces, TERMS, WEIGHT_DIM] (q-major with
// SLOPE_VECTOR: [pairs * pieces, WEIGHT_DIM, PADDED_TERMS]).
extern "C" __global__ void oeq_projected_backward_wprime(
    int64_t node_count,
    int64_t edge_count,
    const scalar_t* x,
    const scalar_t* sh,
    const scalar_t* weights,
    const int32_t* senders,
    const int32_t* receivers,
    const scalar_t* __restrict__ dout0,
    const scalar_t* __restrict__ dout1,
    const scalar_t* __restrict__ dout2,
    const scalar_t* __restrict__ dout3,
    scalar_t* dx,
    scalar_t* dsh,
    const scalar_t* lengths,
    const int32_t* pairs,
    const scalar_t* __restrict__ gprime,
    const scalar_t* starts,
    const scalar_t* scales,
    const int32_t* bins,
    scalar_t r_lo,
    scalar_t bin_scale,
    scalar_t r_max,
    int32_t n_bins,
    int32_t pieces,
    scalar_t* dr) {

    const int lane = threadIdx.x % LOGICAL_GROUP_SIZE;
    const int logical_group = threadIdx.x / LOGICAL_GROUP_SIZE;
    const int64_t e =
        int64_t(blockIdx.x) * (blockDim.x / LOGICAL_GROUP_SIZE) + logical_group;
    if (e >= edge_count) return;

    const int32_t sender = senders[e];
    const int32_t receiver = receivers[e];
    if (sender < 0 || sender >= node_count
        || receiver < 0 || receiver >= node_count) {
        if (lane == 0) dr[e] = scalar_t(0);
        return;
    }

    const int64_t input_base = int64_t(sender) * INPUT_DIM;
    const int64_t output_base = int64_t(receiver) * OUTPUT_DIM;
    const int64_t edge_base = e * EDGE_DIM;
    const int64_t weight_base = e * WEIGHT_DIM;

    // The radial tables' piece of r and its Chebyshev argument; no slope at or beyond the cutoff, below r_lo.
    const scalar_t r = lengths[e];
    const bool sloped = r < r_max && r >= r_lo;
    const scalar_t clamped = r > r_lo ? r : r_lo;
    int32_t bin = int32_t((clamped - r_lo) * bin_scale);
    bin = bin < n_bins - 1 ? bin : n_bins - 1;
    const int32_t piece = bins[bin];
    const scalar_t t = (clamped - starts[piece]) * scales[piece] - scalar_t(1);
    const scalar_t* g = gprime + (int64_t(pairs[e]) * pieces + piece) * PADDED_TERMS * WEIGHT_DIM;
    // HORNER: the value table holds power-basis coefficients in t; w and dw/dt by one dual Horner pass per weight.
    const bool inside = r < r_max;
    const scalar_t sc = scales[piece];

    acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};
    acc_t radial_gradient = acc_t(0);
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
                    const float2* gq =
                        reinterpret_cast<const float2*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 2; ++k) {
                        const float2 v = __ldg(gq + k);
                        gv[2 * k + 0] = v.x;
                        gv[2 * k + 1] = v.y;
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
                    const scalar_t output_gradient = dout0[int64_t(receiver) * 256
                        + 0 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 0];
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
                    const float2* gq =
                        reinterpret_cast<const float2*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 2; ++k) {
                        const float2 v = __ldg(gq + k);
                        gv[2 * k + 0] = v.x;
                        gv[2 * k + 1] = v.y;
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
                    const scalar_t output_gradient = dout0[int64_t(receiver) * 256
                        + 128 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 1];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 2];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a74590331dp-1f);
                        u_1 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 3];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a74590331cp-1f);
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
                    const float2* gq =
                        reinterpret_cast<const float2*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 2; ++k) {
                        const float2 v = __ldg(gq + k);
                        gv[2 * k + 0] = v.x;
                        gv[2 * k + 1] = v.y;
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
                    const scalar_t output_gradient = dout1[int64_t(receiver) * 1152
                        + 0 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 1];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[1]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dout1[int64_t(receiver) * 1152
                        + 384 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 2];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[2]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dout1[int64_t(receiver) * 1152
                        + 768 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 3];
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
                    const float2* gq =
                        reinterpret_cast<const float2*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 2; ++k) {
                        const float2 v = __ldg(gq + k);
                        gv[2 * k + 0] = v.x;
                        gv[2 * k + 1] = v.y;
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
                    const scalar_t output_gradient = dout1[int64_t(receiver) * 1152
                        + 128 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 0];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dout1[int64_t(receiver) * 1152
                        + 512 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 0];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dout1[int64_t(receiver) * 1152
                        + 896 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 0];
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
                    const float2* gq =
                        reinterpret_cast<const float2*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 2; ++k) {
                        const float2 v = __ldg(gq + k);
                        gv[2 * k + 0] = v.x;
                        gv[2 * k + 1] = v.y;
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
                    const scalar_t output_gradient = dout1[int64_t(receiver) * 1152
                        + 256 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                        u_0 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 8];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.186f174f88471p-1f);
                        u_0 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 5];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        u_1 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        u_2 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dout1[int64_t(receiver) * 1152
                        + 640 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 5];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        u_0 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362484910p-1f);
                        u_1 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 7];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        u_2 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dout1[int64_t(receiver) * 1152
                        + 1024 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 7];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        u_1 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                        u_2 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 8];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f88471p-1f);
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
                    const float2* gq =
                        reinterpret_cast<const float2*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 2; ++k) {
                        const float2 v = __ldg(gq + k);
                        gv[2 * k + 0] = v.x;
                        gv[2 * k + 1] = v.y;
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
                    const scalar_t output_gradient = dout2[int64_t(receiver) * 1920
                        + 0 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 4];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[4]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dout2[int64_t(receiver) * 1920
                        + 384 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 5];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[5]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dout2[int64_t(receiver) * 1920
                        + 768 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 6];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[6]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dout2[int64_t(receiver) * 1920
                        + 1152 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 7];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[7]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dout2[int64_t(receiver) * 1920
                        + 1536 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 8];
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
                    const float2* gq =
                        reinterpret_cast<const float2*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 2; ++k) {
                        const float2 v = __ldg(gq + k);
                        gv[2 * k + 0] = v.x;
                        gv[2 * k + 1] = v.y;
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
                    const scalar_t output_gradient = dout2[int64_t(receiver) * 1920
                        + 128 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 3];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        u_0 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 1];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        u_2 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dout2[int64_t(receiver) * 1920
                        + 512 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 2];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        u_0 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 1];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        u_1 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dout2[int64_t(receiver) * 1920
                        + 896 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 1];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 2];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.a20bd700c2c3fp-1f);
                        u_1 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 3];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                        u_2 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dout2[int64_t(receiver) * 1920
                        + 1280 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 3];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        u_1 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 2];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        u_2 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dout2[int64_t(receiver) * 1920
                        + 1664 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 1];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 3];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
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
                    const float2* gq =
                        reinterpret_cast<const float2*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 2; ++k) {
                        const float2 v = __ldg(gq + k);
                        gv[2 * k + 0] = v.x;
                        gv[2 * k + 1] = v.y;
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
                    const scalar_t output_gradient = dout2[int64_t(receiver) * 1920
                        + 256 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 13];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        u_0 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 15];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.31fa808c55b43p-1f);
                        u_0 += scaled * edge_value;
                        v_6 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 10];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                        u_1 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 9];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        u_2 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 11];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        u_2 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dout2[int64_t(receiver) * 1920
                        + 640 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 12];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                        u_0 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 14];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.f3a92ca2f4b7bp-2f);
                        u_0 += scaled * edge_value;
                        v_5 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 11];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                        u_1 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 10];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        u_2 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dout2[int64_t(receiver) * 1920
                        + 1024 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 11];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                        u_0 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 12];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.4f2ec413cb52bp-1f);
                        u_1 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 13];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                        u_2 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dout2[int64_t(receiver) * 1920
                        + 1408 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 10];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        u_0 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 13];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                        u_1 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 12];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                        u_2 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 14];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        u_2 += scaled * edge_value;
                        v_5 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dout2[int64_t(receiver) * 1920
                        + 1792 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 9];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 11];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3c03650e00e03p-3f);
                        u_0 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 14];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                        u_1 += scaled * edge_value;
                        v_5 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 13];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        u_2 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 15];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
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
                    const float2* gq =
                        reinterpret_cast<const float2*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 2; ++k) {
                        const float2 v = __ldg(gq + k);
                        gv[2 * k + 0] = v.x;
                        gv[2 * k + 1] = v.y;
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
                    const scalar_t output_gradient = dout3[int64_t(receiver) * 1792
                        + 0 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 9];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[9]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dout3[int64_t(receiver) * 1792
                        + 256 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 10];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[10]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dout3[int64_t(receiver) * 1792
                        + 512 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 11];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[11]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dout3[int64_t(receiver) * 1792
                        + 768 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 12];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[12]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dout3[int64_t(receiver) * 1792
                        + 1024 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 13];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[13]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dout3[int64_t(receiver) * 1792
                        + 1280 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 14];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[14]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dout3[int64_t(receiver) * 1792
                        + 1536 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 0 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 15];
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
                    const float2* gq =
                        reinterpret_cast<const float2*>(g + weight_index * PADDED_TERMS);
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS / 2; ++k) {
                        const float2 v = __ldg(gq + k);
                        gv[2 * k + 0] = v.x;
                        gv[2 * k + 1] = v.y;
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
                    const scalar_t output_gradient = dout3[int64_t(receiver) * 1792
                        + 128 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 8];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        u_0 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        u_2 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dout3[int64_t(receiver) * 1792
                        + 384 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 7];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        u_0 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        u_1 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 5];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        u_2 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dout3[int64_t(receiver) * 1792
                        + 640 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                        u_0 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 8];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.75e9746a0b099p-3f);
                        u_0 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 5];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                        u_1 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        u_2 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dout3[int64_t(receiver) * 1792
                        + 896 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 5];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                        u_0 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.8c97ef43f7249p-1f);
                        u_1 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 7];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                        u_2 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dout3[int64_t(receiver) * 1792
                        + 1152 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 7];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                        u_1 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                        u_2 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 8];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        u_2 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dout3[int64_t(receiver) * 1792
                        + 1408 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 5];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.279a74590331cp-1f);
                        u_0 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 8];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        u_1 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 7];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        u_2 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dout3[int64_t(receiver) * 1792
                        + 1664 + channel]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh[edge_base
                               + 8];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
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
    // Reduce within the logical 32-thread edge owner. Lane 0 writes dsh[e, :] and dr[e].
    for (int offset = 16; offset > 0; offset >>= 1) {
        for (int component = 0; component < EDGE_DIM; ++component)
            edge_gradient[component] +=
                __shfl_down_sync(FULL_MASK, edge_gradient[component], offset, 32);
        radial_gradient += __shfl_down_sync(FULL_MASK, radial_gradient, offset, 32);
    }

    if (lane == 0) {
        for (int component = 0; component < EDGE_DIM; ++component)
            dsh[edge_base + component] = scalar_t(edge_gradient[component]);
        dr[e] = scalar_t(radial_gradient);
    }
}