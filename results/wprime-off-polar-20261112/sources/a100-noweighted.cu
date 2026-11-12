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
    const scalar_t* dout,
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
        // An edge with an out-of-range index contributes nothing: its dsh row and dr are written as zeros (the
        // wrappers allocate dsh with empty_like, so a skipped row would hold garbage; kernel review, 5 November 2026).
        for (int c = lane; c < EDGE_DIM; c += LOGICAL_GROUP_SIZE) dsh[e * EDGE_DIM + c] = scalar_t(0);
        if (lane == 0) dr[e] = scalar_t(0);
        return;
    }

    const int64_t input_base = int64_t(sender) * INPUT_DIM;
    const int64_t output_base = int64_t(receiver) * OUTPUT_DIM;
    const int64_t edge_base = e * EDGE_DIM;
    const int64_t weight_base = e * WEIGHT_DIM;
    // STAGE_SH: the edge's features in shared memory once per edge; the channel passes read them from there.
    __shared__ __align__(16) scalar_t stage_sh[4][EDGE_DIM];
    __syncwarp();  // the previous edge's reads are done
    for (int c = lane; c < EDGE_DIM; c += LOGICAL_GROUP_SIZE) stage_sh[logical_group][c] = __ldg(sh + edge_base + c);
    __syncwarp();
    const scalar_t* sh_edge = stage_sh[logical_group];

    // The radial tables' piece of r and its Chebyshev argument; no slope at or beyond the cutoff, below r_lo.
    const scalar_t r = lengths[e];
    const bool sloped = r < r_max && r >= r_lo;
    const scalar_t clamped = r > r_lo ? r : r_lo;
    int32_t bin = int32_t((clamped - r_lo) * bin_scale);
    bin = bin < n_bins - 1 ? bin : n_bins - 1;
    const int32_t piece = bins[bin];
    const scalar_t t = (clamped - starts[piece]) * scales[piece] - scalar_t(1);
    const scalar_t* g = gprime + (int64_t(pairs[e]) * pieces + piece) * PADDED_TERMS * WEIGHT_DIM;
    scalar_t cheb[PADDED_TERMS];
    cheb[0] = scalar_t(1);
    if (PADDED_TERMS > 1) cheb[1] = t;
    #pragma unroll
    for (int k = 2; k < PADDED_TERMS; ++k) cheb[k] = scalar_t(2) * t * cheb[k - 1] - cheb[k - 2];
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
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                acc_t u_0 = acc_t(0);
                // SCALAR_PATHS: an l1 = 0 path's edge gradient w x0 c g_o added per term, without v_j.
                const scalar_t weighted_input =
                    radial_weight * x[input_base + 0 + channel];

                 {
                    const int64_t output_index =
                        output_base + 0 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
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
                    #pragma unroll
                    for (int k = 0; k < TERMS; ++k)
                        slope += cheb[k] * __ldg(g + k * WEIGHT_DIM + weight_index);
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    128 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
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
                    const int64_t output_index =
                        output_base + 128 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[1];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[2];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a74590331dp-1f);
                        u_1 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[3];
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
                    #pragma unroll
                    for (int k = 0; k < TERMS; ++k)
                        slope += cheb[k] * __ldg(g + k * WEIGHT_DIM + weight_index);
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    256 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                acc_t u_0 = acc_t(0);
                // SCALAR_PATHS: an l1 = 0 path's edge gradient w x0 c g_o added per term, without v_j.
                const scalar_t weighted_input =
                    radial_weight * x[input_base + 0 + channel];

                 {
                    const int64_t output_index =
                        output_base + 256 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
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
                    const int64_t output_index =
                        output_base + 384 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
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
                    const int64_t output_index =
                        output_base + 512 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
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
                    #pragma unroll
                    for (int k = 0; k < TERMS; ++k)
                        slope += cheb[k] * __ldg(g + k * WEIGHT_DIM + weight_index);
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    384 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                acc_t u_0 = acc_t(0);
                acc_t u_1 = acc_t(0);
                acc_t u_2 = acc_t(0);
                acc_t v_0 = acc_t(0);

                 {
                    const int64_t output_index =
                        output_base + 640 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
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
                    const int64_t output_index =
                        output_base + 768 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
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
                    const int64_t output_index =
                        output_base + 896 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
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
                    #pragma unroll
                    for (int k = 0; k < TERMS; ++k)
                        slope += cheb[k] * __ldg(g + k * WEIGHT_DIM + weight_index);
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    512 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
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
                    const int64_t output_index =
                        output_base + 1024 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                        u_0 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.186f174f88471p-1f);
                        u_0 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        u_1 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        u_2 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const int64_t output_index =
                        output_base + 1152 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        u_0 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362484910p-1f);
                        u_1 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        u_2 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    }
                } {
                    const int64_t output_index =
                        output_base + 1280 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        u_1 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                        u_2 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[8];
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
                    #pragma unroll
                    for (int k = 0; k < TERMS; ++k)
                        slope += cheb[k] * __ldg(g + k * WEIGHT_DIM + weight_index);
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    640 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                acc_t u_0 = acc_t(0);
                // SCALAR_PATHS: an l1 = 0 path's edge gradient w x0 c g_o added per term, without v_j.
                const scalar_t weighted_input =
                    radial_weight * x[input_base + 0 + channel];

                 {
                    const int64_t output_index =
                        output_base + 1408 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
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
                    const int64_t output_index =
                        output_base + 1536 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
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
                    const int64_t output_index =
                        output_base + 1664 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
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
                    const int64_t output_index =
                        output_base + 1792 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
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
                    const int64_t output_index =
                        output_base + 1920 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
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
                    #pragma unroll
                    for (int k = 0; k < TERMS; ++k)
                        slope += cheb[k] * __ldg(g + k * WEIGHT_DIM + weight_index);
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    768 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
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
                    const int64_t output_index =
                        output_base + 2048 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[3];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        u_0 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[1];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        u_2 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const int64_t output_index =
                        output_base + 2176 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[2];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        u_0 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[1];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        u_1 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const int64_t output_index =
                        output_base + 2304 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[1];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[2];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.a20bd700c2c3fp-1f);
                        u_1 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[3];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                        u_2 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    }
                } {
                    const int64_t output_index =
                        output_base + 2432 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[3];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        u_1 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[2];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        u_2 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    }
                } {
                    const int64_t output_index =
                        output_base + 2560 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[1];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[3];
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
                    #pragma unroll
                    for (int k = 0; k < TERMS; ++k)
                        slope += cheb[k] * __ldg(g + k * WEIGHT_DIM + weight_index);
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    896 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
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
                    const int64_t output_index =
                        output_base + 2688 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[13];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        u_0 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[15];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.31fa808c55b43p-1f);
                        u_0 += scaled * edge_value;
                        v_6 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[10];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                        u_1 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[9];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        u_2 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[11];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        u_2 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    }
                } {
                    const int64_t output_index =
                        output_base + 2816 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[12];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                        u_0 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[14];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.f3a92ca2f4b7bp-2f);
                        u_0 += scaled * edge_value;
                        v_5 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[11];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                        u_1 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[10];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        u_2 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    }
                } {
                    const int64_t output_index =
                        output_base + 2944 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[11];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                        u_0 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[12];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.4f2ec413cb52bp-1f);
                        u_1 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[13];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                        u_2 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    }
                } {
                    const int64_t output_index =
                        output_base + 3072 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[10];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        u_0 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[13];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                        u_1 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[12];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                        u_2 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[14];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        u_2 += scaled * edge_value;
                        v_5 += scaled * input_value;
                    }
                } {
                    const int64_t output_index =
                        output_base + 3200 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[9];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[11];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3c03650e00e03p-3f);
                        u_0 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[14];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                        u_1 += scaled * edge_value;
                        v_5 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[13];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        u_2 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[15];
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
                    #pragma unroll
                    for (int k = 0; k < TERMS; ++k)
                        slope += cheb[k] * __ldg(g + k * WEIGHT_DIM + weight_index);
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    1024 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
                acc_t weight_gradient = acc_t(0);
                // FACTORED: per path u_i = sum c g_o y_j and v_j = sum c g_o x_i, then the weight gradient
                // sum_i x_i u_i, dx_i += w u_i and dsh_j += w v_j (the same sums, regrouped).
                acc_t u_0 = acc_t(0);
                // SCALAR_PATHS: an l1 = 0 path's edge gradient w x0 c g_o added per term, without v_j.
                const scalar_t weighted_input =
                    radial_weight * x[input_base + 0 + channel];

                 {
                    const int64_t output_index =
                        output_base + 3328 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
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
                    const int64_t output_index =
                        output_base + 3456 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
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
                    const int64_t output_index =
                        output_base + 3584 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
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
                    const int64_t output_index =
                        output_base + 3712 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
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
                    const int64_t output_index =
                        output_base + 3840 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
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
                    const int64_t output_index =
                        output_base + 3968 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
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
                    const int64_t output_index =
                        output_base + 4096 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
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
                    #pragma unroll
                    for (int k = 0; k < TERMS; ++k)
                        slope += cheb[k] * __ldg(g + k * WEIGHT_DIM + weight_index);
                    radial_gradient += weight_gradient * slope;
                }
            }
             {
                const int weight_index =
                    1152 + channel;
                const int64_t radial_weight_index = weight_base + weight_index;
                const scalar_t radial_weight = weights[radial_weight_index];
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
                    const int64_t output_index =
                        output_base + 4224 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        u_0 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        u_2 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const int64_t output_index =
                        output_base + 4352 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        u_0 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        u_1 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        u_2 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    }
                } {
                    const int64_t output_index =
                        output_base + 4480 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                        u_0 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.75e9746a0b099p-3f);
                        u_0 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                        u_1 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        u_2 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const int64_t output_index =
                        output_base + 4608 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                        u_0 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.8c97ef43f7249p-1f);
                        u_1 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                        u_2 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    }
                } {
                    const int64_t output_index =
                        output_base + 4736 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                        u_1 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                        u_2 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        u_2 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    }
                } {
                    const int64_t output_index =
                        output_base + 4864 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.279a74590331cp-1f);
                        u_0 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 256 + channel];
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        u_1 += scaled * edge_value;
                        v_4 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        u_2 += scaled * edge_value;
                        v_3 += scaled * input_value;
                    }
                } {
                    const int64_t output_index =
                        output_base + 4992 + channel;
                    const scalar_t output_gradient = dout[output_index]; {
                        const scalar_t input_value =
                            x[input_base
                              + 128 + channel];
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            x[input_base
                              + 384 + channel];
                        const scalar_t edge_value =
                            sh_edge[8];
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
                    #pragma unroll
                    for (int k = 0; k < TERMS; ++k)
                        slope += cheb[k] * __ldg(g + k * WEIGHT_DIM + weight_index);
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
    // BUTTERFLY: reduce-scatter the edge gradient; lane c << SHIFT writes dsh[e, c], lane 0 dr[e].
    constexpr int SHIFT = 1;
    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        const int half = (EDGE_DIM / 2) >> step, offset = 16 >> step;
        const bool upper = (lane & offset) != 0;
        #pragma unroll
        for (int i = 0; i < EDGE_DIM / 2; ++i) {
            if (i < half) {
                const acc_t send = upper ? edge_gradient[i] : edge_gradient[i + half];
                const acc_t keep = upper ? edge_gradient[i + half] : edge_gradient[i];
                edge_gradient[i] = keep + __shfl_xor_sync(FULL_MASK, send, offset, 32);
            }
        }
    }
    #pragma unroll
    for (int offset = (1 << SHIFT) >> 1; offset > 0; offset >>= 1)
        edge_gradient[0] += __shfl_xor_sync(FULL_MASK, edge_gradient[0], offset, 32);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        radial_gradient += __shfl_xor_sync(FULL_MASK, radial_gradient, offset, 32);
    if ((lane & ((1 << SHIFT) - 1)) == 0) dsh[edge_base + (lane >> SHIFT)] = scalar_t(edge_gradient[0]);
    if (lane == 0) dr[e] = scalar_t(radial_gradient);
}