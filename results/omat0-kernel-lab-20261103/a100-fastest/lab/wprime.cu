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
constexpr int TERMS = 10;
constexpr int PADDED_TERMS = 10;  // a q-major G' row: TERMS, then zero terms (SLOPE_VECTOR)
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

                 {
                    const int64_t output_index =
                        output_base + 0 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 0 + channel;
                        const int edge_component_index =
                            0
                            + 0 * 1;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                }
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

                 {
                    const int64_t output_index =
                        output_base + 128 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            1
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                    } {
                        const int64_t input_index = input_base
                            + 256 + channel;
                        const int edge_component_index =
                            2
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.279a74590331dp-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.279a74590331dp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.279a74590331dp-1f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            3
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                    }
                }
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

                 {
                    const int64_t output_index =
                        output_base + 256 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 0 + channel;
                        const int edge_component_index =
                            1
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 384 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 0 + channel;
                        const int edge_component_index =
                            2
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 512 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 0 + channel;
                        const int edge_component_index =
                            3
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                    }
                }
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

                 {
                    const int64_t output_index =
                        output_base + 640 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            0
                            + 0 * 1;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 768 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 256 + channel;
                        const int edge_component_index =
                            0
                            + 0 * 1;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 896 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            0
                            + 0 * 1;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.fffffffffffffp-1f);
                    }
                }
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

                 {
                    const int64_t output_index =
                        output_base + 1024 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                    } {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.186f174f88471p-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.186f174f88471p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.186f174f88471p-1f);
                    } {
                        const int64_t input_index = input_base
                            + 256 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 1152 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                    } {
                        const int64_t input_index = input_base
                            + 256 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.43d1362484910p-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.43d1362484910p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.43d1362484910p-1f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 1280 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                    } {
                        const int64_t input_index = input_base
                            + 256 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.43d136248490fp-2f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.186f174f88471p-1f);
                    }
                }
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

                 {
                    const int64_t output_index =
                        output_base + 1408 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 0 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 1536 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 0 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 1664 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 0 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000001p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000001p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000001p+0f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 1792 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 0 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 1920 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 0 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                }
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

                 {
                    const int64_t output_index =
                        output_base + 2048 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            3
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            1
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 2176 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            2
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                    } {
                        const int64_t input_index = input_base
                            + 256 + channel;
                        const int edge_component_index =
                            1
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 2304 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            1
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                    } {
                        const int64_t input_index = input_base
                            + 256 + channel;
                        const int edge_component_index =
                            2
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.a20bd700c2c3fp-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.a20bd700c2c3fp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.a20bd700c2c3fp-1f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            3
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.a20bd700c2c3ep-2f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 2432 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 256 + channel;
                        const int edge_component_index =
                            3
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            2
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bcdp-1f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 2560 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            1
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            3
                            + 0 * 3;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                    }
                }
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

                 {
                    const int64_t output_index =
                        output_base + 2688 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            13
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                    } {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            15
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.31fa808c55b43p-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.31fa808c55b43p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.31fa808c55b43p-1f);
                    } {
                        const int64_t input_index = input_base
                            + 256 + channel;
                        const int edge_component_index =
                            10
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            9
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            11
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 2816 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            12
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                    } {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            14
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.f3a92ca2f4b7bp-2f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.f3a92ca2f4b7bp-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.f3a92ca2f4b7bp-2f);
                    } {
                        const int64_t input_index = input_base
                            + 256 + channel;
                        const int edge_component_index =
                            11
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            10
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 2944 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            11
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                    } {
                        const int64_t input_index = input_base
                            + 256 + channel;
                        const int edge_component_index =
                            12
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.4f2ec413cb52bp-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.4f2ec413cb52bp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.4f2ec413cb52bp-1f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            13
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.11acee560242ap-1f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 3072 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            10
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                    } {
                        const int64_t input_index = input_base
                            + 256 + channel;
                        const int edge_component_index =
                            13
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.3c03650e00e03p-1f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            12
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.83091e6a7f7e6p-2f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            14
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7bp-2f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 3200 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            9
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                    } {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            11
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.3c03650e00e03p-3f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.3c03650e00e03p-3f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.3c03650e00e03p-3f);
                    } {
                        const int64_t input_index = input_base
                            + 256 + channel;
                        const int edge_component_index =
                            14
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.f3a92ca2f4b7dp-2f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            13
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.3c03650e00e03p-3f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            15
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.31fa808c55b43p-1f);
                    }
                }
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

                 {
                    const int64_t output_index =
                        output_base + 3328 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 0 + channel;
                        const int edge_component_index =
                            9
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 3456 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 0 + channel;
                        const int edge_component_index =
                            10
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 3584 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 0 + channel;
                        const int edge_component_index =
                            11
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 3712 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 0 + channel;
                        const int edge_component_index =
                            12
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000001p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000001p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000001p+0f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 3840 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 0 + channel;
                        const int edge_component_index =
                            13
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 3968 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 0 + channel;
                        const int edge_component_index =
                            14
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 4096 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 0 + channel;
                        const int edge_component_index =
                            15
                            + 0 * 7;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        input_gradient_0_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.0000000000000p+0f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.0000000000000p+0f);
                    }
                }
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

                 {
                    const int64_t output_index =
                        output_base + 4224 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 4352 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                    } {
                        const int64_t input_index = input_base
                            + 256 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 4480 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                    } {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.75e9746a0b099p-3f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.75e9746a0b099p-3f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.75e9746a0b099p-3f);
                    } {
                        const int64_t input_index = input_base
                            + 256 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 4608 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                    } {
                        const int64_t input_index = input_base
                            + 256 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.8c97ef43f7249p-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.8c97ef43f7249p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.8c97ef43f7249p-1f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.c9f25c5bfedd8p-2f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 4736 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                    } {
                        const int64_t input_index = input_base
                            + 256 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.75e9746a0b099p-1f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            6
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.43d136248490fp-1f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.75e9746a0b099p-3f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 4864 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            5
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.279a74590331cp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.279a74590331cp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.279a74590331cp-1f);
                    } {
                        const int64_t input_index = input_base
                            + 256 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        input_gradient_128_1 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            7
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.279a74590331cp-1f);
                    }
                } {
                    const int64_t output_index =
                        output_base + 4992 + channel;
                    const scalar_t output_gradient = dout[output_index];

                     {
                        const int64_t input_index = input_base
                            + 128 + channel;
                        const int edge_component_index =
                            4
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                        input_gradient_128_0 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(-0x1.6a09e667f3bccp-1f);
                    } {
                        const int64_t input_index = input_base
                            + 384 + channel;
                        const int edge_component_index =
                            8
                            + 0 * 5;

                        const int64_t edge_index = edge_base + edge_component_index;
                        const scalar_t input_value = x[input_index];
                        const scalar_t edge_value = sh[edge_index];

                        weight_gradient += output_gradient * input_value
                            * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        input_gradient_128_2 += output_gradient
                            * radial_weight * edge_value * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                        edge_gradient[edge_component_index] +=
                            output_gradient * radial_weight * input_value
                            * static_cast<scalar_t>(0x1.6a09e667f3bccp-1f);
                    }
                }
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