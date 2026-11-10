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
constexpr int TERMS = 4;
constexpr int PADDED_TERMS = 4;  // a q-major G' row: TERMS, then zero terms (SLOPE_VECTOR)
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
    // STAGE_DOUT: this block's edges [c0, c1), walked by runs of equal receiver; per run the receiver's dout row is
    // copied to shared memory once, then the warps take the run's edges.
    __shared__ __align__(16) scalar_t stage_row[5120];
    const int64_t stage_c0 = int64_t(blockIdx.x) * 32;
    const int64_t stage_c1 = stage_c0 + 32 < edge_count ? stage_c0 + 32 : edge_count;
    // STAGE_EDGES: the chunk's edge features (the whole block, float4 where aligned) and per edge (warp 0, a lane per
    // edge) sender, receiver, spline interval, t, 1 / h and the inside / sloped flags, once per block.
    __shared__ __align__(16) scalar_t stage_esh[32][EDGE_DIM];
    __shared__ int32_t stage_snd[32], stage_rcv[32], stage_slab[32];
    __shared__ float stage_t[32], stage_sx[32];
    __shared__ unsigned stage_flag[32];
    const int stage_n = int(stage_c1 - stage_c0);
    {
        const scalar_t* src = sh + stage_c0 * EDGE_DIM;
        scalar_t* dst = &stage_esh[0][0];
        if (EDGE_DIM % 4 == 0 && (reinterpret_cast<unsigned long long>(src) & 15) == 0) {
            for (int i = threadIdx.x; i < stage_n * (EDGE_DIM / 4); i += blockDim.x)
                reinterpret_cast<float4*>(dst)[i] = __ldg(reinterpret_cast<const float4*>(src) + i);
        } else {
            for (int i = threadIdx.x; i < stage_n * EDGE_DIM; i += blockDim.x) dst[i] = __ldg(src + i);
        }
    }
    if (threadIdx.x < 32) {
        const int k = threadIdx.x;
        int32_t snd_k = -1, rcv_k = -1, slab_k = 0;
        float t_k = 0.f, sx_k = 0.f;
        unsigned flag_k = 0u;
        if (k < stage_n) {
            const int64_t e = stage_c0 + k;
            snd_k = senders[e];
            rcv_k = receivers[e];
            const scalar_t r = lengths[e];
            const scalar_t clamped = r > r_lo ? r : r_lo;
    int iv; float t, sx_inv;
    if (clamped < 0x1.0000000000000p-1f) { sx_inv = 0x1.8880000000000p+9f; float u = (clamped - 0x1.3333333333333p-2f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 156 ? il : 156; t = u - (float)il; iv = 0 + il; }
    else if (clamped < 0x1.8000000000000p-1f) { sx_inv = 0x1.c000000000000p+8f; float u = (clamped - 0x1.0000000000000p-1f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 111 ? il : 111; t = u - (float)il; iv = 157 + il; }
    else if (clamped < 0x1.0000000000000p+0f) { sx_inv = 0x1.5c00000000000p+8f; float u = (clamped - 0x1.8000000000000p-1f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 86 ? il : 86; t = u - (float)il; iv = 269 + il; }
    else if (clamped < 0x1.8000000000000p+0f) { sx_inv = 0x1.f000000000000p+7f; float u = (clamped - 0x1.0000000000000p+0f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 123 ? il : 123; t = u - (float)il; iv = 356 + il; }
    else if (clamped < 0x1.0000000000000p+1f) { sx_inv = 0x1.b000000000000p+6f; float u = (clamped - 0x1.8000000000000p+0f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 53 ? il : 53; t = u - (float)il; iv = 480 + il; }
    else if (clamped < 0x1.4000000000000p+1f) { sx_inv = 0x1.f000000000000p+5f; float u = (clamped - 0x1.0000000000000p+1f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 30 ? il : 30; t = u - (float)il; iv = 534 + il; }
    else { sx_inv = 0x1.7b6db6db6db6ep+5f; float u = (clamped - 0x1.4000000000000p+1f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 165 ? il : 165; t = u - (float)il; iv = 565 + il; }
            t_k = t;
            sx_k = sx_inv;
            slab_k = int32_t(int64_t(pairs[e]) * pieces + iv);
            flag_k = (r < r_max ? 1u : 0u) | (r < r_max && r >= r_lo ? 2u : 0u);
        }
        stage_snd[k] = snd_k;
        stage_rcv[k] = rcv_k;
        stage_slab[k] = slab_k;
        stage_t[k] = t_k;
        stage_sx[k] = sx_k;
        stage_flag[k] = flag_k;
    }
    __syncthreads();
    for (int64_t stage_s = stage_c0; stage_s < stage_c1;) {
    int64_t stage_t1 = stage_c1;
    {
        const int32_t stage_r = stage_rcv[stage_s - stage_c0];
        const int64_t stage_i = stage_s + 1 + lane;
        const unsigned stage_b = __ballot_sync(FULL_MASK,
                                               stage_i < stage_c1 && stage_rcv[stage_i - stage_c0] != stage_r);
        if (stage_b) stage_t1 = stage_s + __ffs(stage_b);
        if (stage_r >= 0 && stage_r < node_count) {
            {
                const scalar_t* src = dout0 + int64_t(stage_r) * 256;
                if (0 == 0 && (reinterpret_cast<unsigned long long>(src) & 15) == 0) {
                    for (int i = threadIdx.x; i < 64; i += blockDim.x)
                        reinterpret_cast<float4*>(stage_row + 0)[i] =
                            __ldg(reinterpret_cast<const float4*>(src) + i);
                } else {
                    for (int i = threadIdx.x; i < 256; i += blockDim.x)
                        stage_row[0 + i] = __ldg(src + i);
                }
            }
            {
                const scalar_t* src = dout1 + int64_t(stage_r) * 1152;
                if (0 == 0 && (reinterpret_cast<unsigned long long>(src) & 15) == 0) {
                    for (int i = threadIdx.x; i < 288; i += blockDim.x)
                        reinterpret_cast<float4*>(stage_row + 256)[i] =
                            __ldg(reinterpret_cast<const float4*>(src) + i);
                } else {
                    for (int i = threadIdx.x; i < 1152; i += blockDim.x)
                        stage_row[256 + i] = __ldg(src + i);
                }
            }
            {
                const scalar_t* src = dout2 + int64_t(stage_r) * 1920;
                if (0 == 0 && (reinterpret_cast<unsigned long long>(src) & 15) == 0) {
                    for (int i = threadIdx.x; i < 480; i += blockDim.x)
                        reinterpret_cast<float4*>(stage_row + 1408)[i] =
                            __ldg(reinterpret_cast<const float4*>(src) + i);
                } else {
                    for (int i = threadIdx.x; i < 1920; i += blockDim.x)
                        stage_row[1408 + i] = __ldg(src + i);
                }
            }
            {
                const scalar_t* src = dout3 + int64_t(stage_r) * 1792;
                if (0 == 0 && (reinterpret_cast<unsigned long long>(src) & 15) == 0) {
                    for (int i = threadIdx.x; i < 448; i += blockDim.x)
                        reinterpret_cast<float4*>(stage_row + 3328)[i] =
                            __ldg(reinterpret_cast<const float4*>(src) + i);
                } else {
                    for (int i = threadIdx.x; i < 1792; i += blockDim.x)
                        stage_row[3328 + i] = __ldg(src + i);
                }
            }
        }
    }
    __syncthreads();
    for (int64_t e = stage_s + logical_group; e < stage_t1; e += blockDim.x / LOGICAL_GROUP_SIZE) {
    // STAGE_EDGES: the edge's staged sender, receiver, features and spline interval.
    const int stage_k = int(e - stage_c0);
    const int32_t sender = stage_snd[stage_k];
    const int32_t receiver = stage_rcv[stage_k];
    if (sender < 0 || sender >= node_count
        || receiver < 0 || receiver >= node_count) {
        // An edge with an out-of-range index contributes nothing: its dsh row and dr are written as zeros (the
        // wrappers allocate dsh with empty_like, so a skipped row would hold garbage; kernel review, 5 November 2026).
        for (int c = lane; c < EDGE_DIM; c += LOGICAL_GROUP_SIZE) dsh[e * EDGE_DIM + c] = scalar_t(0);
        if (lane == 0) dr[e] = scalar_t(0);
        continue;
    }

    const int64_t input_base = int64_t(sender) * INPUT_DIM;
    const int64_t output_base = int64_t(receiver) * OUTPUT_DIM;
    const int64_t edge_base = e * EDGE_DIM;
    const int64_t weight_base = e * WEIGHT_DIM;
    const scalar_t* sh_edge = stage_esh[stage_k];

    // The radial tables' piece of r and its Chebyshev argument; no slope at or beyond the cutoff, below r_lo.
    const unsigned stage_f = stage_flag[stage_k];
    const bool sloped = (stage_f & 2u) != 0u;
    const float t = stage_t[stage_k], sx_inv = stage_sx[stage_k];
    __syncwarp();  // the warp is converged for the end-of-edge reduction
    const scalar_t* g = gprime + int64_t(stage_slab[stage_k]) * PADDED_TERMS * WEIGHT_DIM;
    // HORNER: the value table holds power-basis coefficients in t; w and dw/dt by one dual Horner pass per weight.
    const bool inside = (stage_f & 1u) != 0u;
    const scalar_t sc = sx_inv;

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
    }  // STAGE_DOUT: the run's edges
    __syncthreads();  // every warp is done with the row before the next copy
    stage_s = stage_t1;
    }  // STAGE_DOUT: the runs
}