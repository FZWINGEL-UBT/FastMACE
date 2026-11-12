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
    for (int64_t stage_s = stage_c0; stage_s < stage_c1;) {
    int64_t stage_t1 = stage_c1;
    {
        const int32_t stage_r = receivers[stage_s];
        const int64_t stage_i = stage_s + 1 + lane;
        const unsigned stage_b = __ballot_sync(FULL_MASK, stage_i < stage_c1 && receivers[stage_i] != stage_r);
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
    const int32_t sender = senders[e];
    const int32_t receiver = receivers[e];
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
    // SPLINE_W: the cubic spline's interval and t in [0, 1) (rewrites.radial.spline; pieces = its intervals).
    int iv; float t, sx_inv;
    if (clamped < 0x1.0000000000000p-1f) { sx_inv = 0x1.8880000000000p+9f; float u = (clamped - 0x1.3333333333333p-2f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 156 ? il : 156; t = u - (float)il; iv = 0 + il; }
    else if (clamped < 0x1.8000000000000p-1f) { sx_inv = 0x1.c000000000000p+8f; float u = (clamped - 0x1.0000000000000p-1f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 111 ? il : 111; t = u - (float)il; iv = 157 + il; }
    else if (clamped < 0x1.0000000000000p+0f) { sx_inv = 0x1.5c00000000000p+8f; float u = (clamped - 0x1.8000000000000p-1f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 86 ? il : 86; t = u - (float)il; iv = 269 + il; }
    else if (clamped < 0x1.8000000000000p+0f) { sx_inv = 0x1.f000000000000p+7f; float u = (clamped - 0x1.0000000000000p+0f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 123 ? il : 123; t = u - (float)il; iv = 356 + il; }
    else if (clamped < 0x1.0000000000000p+1f) { sx_inv = 0x1.b000000000000p+6f; float u = (clamped - 0x1.8000000000000p+0f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 53 ? il : 53; t = u - (float)il; iv = 480 + il; }
    else if (clamped < 0x1.4000000000000p+1f) { sx_inv = 0x1.f000000000000p+5f; float u = (clamped - 0x1.0000000000000p+1f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 30 ? il : 30; t = u - (float)il; iv = 534 + il; }
    else { sx_inv = 0x1.7b6db6db6db6ep+5f; float u = (clamped - 0x1.4000000000000p+1f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 165 ? il : 165; t = u - (float)il; iv = 565 + il; }
    const int32_t piece = iv;
    const scalar_t* g = gprime + (int64_t(pairs[e]) * pieces + piece) * PADDED_TERMS * WEIGHT_DIM;
    // HORNER: the value table holds power-basis coefficients in t; w and dw/dt by one dual Horner pass per weight.
    const bool inside = r < r_max;
    const scalar_t sc = sx_inv;

    if (!inside) {  // skip: at or beyond the cutoff (padding edges) w = 0: nothing but zeros
        for (int c = lane; c < EDGE_DIM; c += LOGICAL_GROUP_SIZE) dsh[edge_base + c] = scalar_t(0);
        if (lane == 0) dr[e] = scalar_t(0);
        continue;
    }
    acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};
    acc_t radial_gradient = acc_t(0);
    for (int cbase = lane * 2; cbase < CHANNEL_DIM; cbase += LOGICAL_GROUP_SIZE * 2) {  // vec2
            acc_t input_gradient_0_0_0 = acc_t(0);
            acc_t input_gradient_0_0_1 = acc_t(0);
            acc_t input_gradient_128_0_0 = acc_t(0);
            acc_t input_gradient_128_0_1 = acc_t(0);
            acc_t input_gradient_128_1_0 = acc_t(0);
            acc_t input_gradient_128_1_1 = acc_t(0);
            acc_t input_gradient_128_2_0 = acc_t(0);
            acc_t input_gradient_128_2_1 = acc_t(0);
            const float2 xv_0 = *reinterpret_cast<const float2*>(x + input_base + 0 + cbase);
            const float2 xv_128 = *reinterpret_cast<const float2*>(x + input_base + 128 + cbase);
            const float2 xv_256 = *reinterpret_cast<const float2*>(x + input_base + 256 + cbase);
            const float2 xv_384 = *reinterpret_cast<const float2*>(x + input_base + 384 + cbase);
            {
            const float2 dv_0 = *reinterpret_cast<const float2*>(stage_row + 0 + 0 + cbase);
{
                const int weight_index =
                    0 + (cbase + 0);
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (true) {
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
                    radial_weight * xv_0.x;

                 {
                    const scalar_t output_gradient = dv_0.x; {
                        const scalar_t input_value =
                            xv_0.x;
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
                    xv_0.x * u_0;
                input_gradient_0_0_0 += scaled_weight * u_0;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
{
                const int weight_index =
                    0 + (cbase + 1);
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (true) {
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
                    radial_weight * xv_0.y;

                 {
                    const scalar_t output_gradient = dv_0.y; {
                        const scalar_t input_value =
                            xv_0.y;
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
                    xv_0.y * u_0;
                input_gradient_0_0_1 += scaled_weight * u_0;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
            }
            {
            const float2 dv_0 = *reinterpret_cast<const float2*>(stage_row + 0 + 128 + cbase);
{
                const int weight_index =
                    128 + (cbase + 0);
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (true) {
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
                    const scalar_t output_gradient = dv_0.x; {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[1];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            xv_256.x;
                        const scalar_t edge_value =
                            sh_edge[2];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[3];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    }
                }
                const scalar_t scaled_weight = radial_weight;
                weight_gradient +=
                    xv_128.x * u_0;
                input_gradient_128_0_0 += scaled_weight * u_0;
                weight_gradient +=
                    xv_256.x * u_1;
                input_gradient_128_1_0 += scaled_weight * u_1;
                weight_gradient +=
                    xv_384.x * u_2;
                input_gradient_128_2_0 += scaled_weight * u_2;
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
                    128 + (cbase + 1);
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (true) {
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
                    const scalar_t output_gradient = dv_0.y; {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[1];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            xv_256.y;
                        const scalar_t edge_value =
                            sh_edge[2];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[3];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    }
                }
                const scalar_t scaled_weight = radial_weight;
                weight_gradient +=
                    xv_128.y * u_0;
                input_gradient_128_0_1 += scaled_weight * u_0;
                weight_gradient +=
                    xv_256.y * u_1;
                input_gradient_128_1_1 += scaled_weight * u_1;
                weight_gradient +=
                    xv_384.y * u_2;
                input_gradient_128_2_1 += scaled_weight * u_2;
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
            }
            {
            const float2 dv_0 = *reinterpret_cast<const float2*>(stage_row + 256 + 0 + cbase);
            const float2 dv_1 = *reinterpret_cast<const float2*>(stage_row + 256 + 384 + cbase);
            const float2 dv_2 = *reinterpret_cast<const float2*>(stage_row + 256 + 768 + cbase);
{
                const int weight_index =
                    256 + (cbase + 0);
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (true) {
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
                    radial_weight * xv_0.x;

                 {
                    const scalar_t output_gradient = dv_0.x; {
                        const scalar_t input_value =
                            xv_0.x;
                        const scalar_t edge_value =
                            sh_edge[1];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[1]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dv_1.x; {
                        const scalar_t input_value =
                            xv_0.x;
                        const scalar_t edge_value =
                            sh_edge[2];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[2]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dv_2.x; {
                        const scalar_t input_value =
                            xv_0.x;
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
                    xv_0.x * u_0;
                input_gradient_0_0_0 += scaled_weight * u_0;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
{
                const int weight_index =
                    256 + (cbase + 1);
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (true) {
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
                    radial_weight * xv_0.y;

                 {
                    const scalar_t output_gradient = dv_0.y; {
                        const scalar_t input_value =
                            xv_0.y;
                        const scalar_t edge_value =
                            sh_edge[1];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[1]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dv_1.y; {
                        const scalar_t input_value =
                            xv_0.y;
                        const scalar_t edge_value =
                            sh_edge[2];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[2]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dv_2.y; {
                        const scalar_t input_value =
                            xv_0.y;
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
                    xv_0.y * u_0;
                input_gradient_0_0_1 += scaled_weight * u_0;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
            }
            {
            const float2 dv_0 = *reinterpret_cast<const float2*>(stage_row + 256 + 128 + cbase);
            const float2 dv_1 = *reinterpret_cast<const float2*>(stage_row + 256 + 512 + cbase);
            const float2 dv_2 = *reinterpret_cast<const float2*>(stage_row + 256 + 896 + cbase);
{
                const int weight_index =
                    384 + (cbase + 0);
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (true) {
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
                    const scalar_t output_gradient = dv_0.x; {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[0];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dv_1.x; {
                        const scalar_t input_value =
                            xv_256.x;
                        const scalar_t edge_value =
                            sh_edge[0];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dv_2.x; {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[0];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                }
                const scalar_t scaled_weight = radial_weight;
                weight_gradient +=
                    xv_128.x * u_0;
                input_gradient_128_0_0 += scaled_weight * u_0;
                weight_gradient +=
                    xv_256.x * u_1;
                input_gradient_128_1_0 += scaled_weight * u_1;
                weight_gradient +=
                    xv_384.x * u_2;
                input_gradient_128_2_0 += scaled_weight * u_2;
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
                    384 + (cbase + 1);
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (true) {
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
                    const scalar_t output_gradient = dv_0.y; {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[0];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dv_1.y; {
                        const scalar_t input_value =
                            xv_256.y;
                        const scalar_t edge_value =
                            sh_edge[0];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dv_2.y; {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[0];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                }
                const scalar_t scaled_weight = radial_weight;
                weight_gradient +=
                    xv_128.y * u_0;
                input_gradient_128_0_1 += scaled_weight * u_0;
                weight_gradient +=
                    xv_256.y * u_1;
                input_gradient_128_1_1 += scaled_weight * u_1;
                weight_gradient +=
                    xv_384.y * u_2;
                input_gradient_128_2_1 += scaled_weight * u_2;
                edge_gradient[0] +=
                    scaled_weight * v_0;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
            }
            {
            const float2 dv_0 = *reinterpret_cast<const float2*>(stage_row + 256 + 256 + cbase);
            const float2 dv_1 = *reinterpret_cast<const float2*>(stage_row + 256 + 640 + cbase);
            const float2 dv_2 = *reinterpret_cast<const float2*>(stage_row + 256 + 1024 + cbase);
{
                const int weight_index =
                    512 + (cbase + 0);
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (true) {
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

                 {
                    const scalar_t output_gradient = dv_0.x; {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a745903946p-1f);
                        u_0 -= scaled * edge_value;
                        edge_gradient[6] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient;
                        u_0 -= scaled * edge_value;
                        edge_gradient[8] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.x;
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        edge_gradient[5] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        edge_gradient[4] += scaled * (radial_weight * input_value);
                    }
                } {
                    const scalar_t output_gradient = dv_1.x; {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[5] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.x;
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a745903947p+0f);
                        u_1 += scaled * edge_value;
                        edge_gradient[6] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        edge_gradient[7] += scaled * (radial_weight * input_value);
                    }
                } {
                    const scalar_t output_gradient = dv_2.x; {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[4] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.x;
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        edge_gradient[7] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a745903946p-1f);
                        u_2 -= scaled * edge_value;
                        edge_gradient[6] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        edge_gradient[8] += scaled * (radial_weight * input_value);
                    }
                }
                const scalar_t scaled_weight = radial_weight;
                weight_gradient +=
                    xv_128.x * u_0;
                input_gradient_128_0_0 += scaled_weight * u_0;
                weight_gradient +=
                    xv_256.x * u_1;
                input_gradient_128_1_0 += scaled_weight * u_1;
                weight_gradient +=
                    xv_384.x * u_2;
                input_gradient_128_2_0 += scaled_weight * u_2;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
{
                const int weight_index =
                    512 + (cbase + 1);
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (true) {
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

                 {
                    const scalar_t output_gradient = dv_0.y; {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a745903946p-1f);
                        u_0 -= scaled * edge_value;
                        edge_gradient[6] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient;
                        u_0 -= scaled * edge_value;
                        edge_gradient[8] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.y;
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        edge_gradient[5] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        edge_gradient[4] += scaled * (radial_weight * input_value);
                    }
                } {
                    const scalar_t output_gradient = dv_1.y; {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[5] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.y;
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a745903947p+0f);
                        u_1 += scaled * edge_value;
                        edge_gradient[6] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        edge_gradient[7] += scaled * (radial_weight * input_value);
                    }
                } {
                    const scalar_t output_gradient = dv_2.y; {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[4] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.y;
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        edge_gradient[7] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a745903946p-1f);
                        u_2 -= scaled * edge_value;
                        edge_gradient[6] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        edge_gradient[8] += scaled * (radial_weight * input_value);
                    }
                }
                const scalar_t scaled_weight = radial_weight;
                weight_gradient +=
                    xv_128.y * u_0;
                input_gradient_128_0_1 += scaled_weight * u_0;
                weight_gradient +=
                    xv_256.y * u_1;
                input_gradient_128_1_1 += scaled_weight * u_1;
                weight_gradient +=
                    xv_384.y * u_2;
                input_gradient_128_2_1 += scaled_weight * u_2;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
            }
            {
            const float2 dv_0 = *reinterpret_cast<const float2*>(stage_row + 1408 + 0 + cbase);
            const float2 dv_1 = *reinterpret_cast<const float2*>(stage_row + 1408 + 384 + cbase);
            const float2 dv_2 = *reinterpret_cast<const float2*>(stage_row + 1408 + 768 + cbase);
            const float2 dv_3 = *reinterpret_cast<const float2*>(stage_row + 1408 + 1152 + cbase);
            const float2 dv_4 = *reinterpret_cast<const float2*>(stage_row + 1408 + 1536 + cbase);
{
                const int weight_index =
                    640 + (cbase + 0);
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (true) {
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
                    radial_weight * xv_0.x;

                 {
                    const scalar_t output_gradient = dv_0.x; {
                        const scalar_t input_value =
                            xv_0.x;
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[4]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dv_1.x; {
                        const scalar_t input_value =
                            xv_0.x;
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[5]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dv_2.x; {
                        const scalar_t input_value =
                            xv_0.x;
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[6]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dv_3.x; {
                        const scalar_t input_value =
                            xv_0.x;
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[7]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dv_4.x; {
                        const scalar_t input_value =
                            xv_0.x;
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
                    xv_0.x * u_0;
                input_gradient_0_0_0 += scaled_weight * u_0;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
{
                const int weight_index =
                    640 + (cbase + 1);
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (true) {
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
                    radial_weight * xv_0.y;

                 {
                    const scalar_t output_gradient = dv_0.y; {
                        const scalar_t input_value =
                            xv_0.y;
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[4]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dv_1.y; {
                        const scalar_t input_value =
                            xv_0.y;
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[5]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dv_2.y; {
                        const scalar_t input_value =
                            xv_0.y;
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[6]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dv_3.y; {
                        const scalar_t input_value =
                            xv_0.y;
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[7]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dv_4.y; {
                        const scalar_t input_value =
                            xv_0.y;
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
                    xv_0.y * u_0;
                input_gradient_0_0_1 += scaled_weight * u_0;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
            }
            {
            const float2 dv_0 = *reinterpret_cast<const float2*>(stage_row + 1408 + 128 + cbase);
            const float2 dv_1 = *reinterpret_cast<const float2*>(stage_row + 1408 + 512 + cbase);
            const float2 dv_2 = *reinterpret_cast<const float2*>(stage_row + 1408 + 896 + cbase);
            const float2 dv_3 = *reinterpret_cast<const float2*>(stage_row + 1408 + 1280 + cbase);
            const float2 dv_4 = *reinterpret_cast<const float2*>(stage_row + 1408 + 1664 + cbase);
{
                const int weight_index =
                    768 + (cbase + 0);
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (true) {
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
                    const scalar_t output_gradient = dv_0.x; {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[3];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[1];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dv_1.x; {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[2];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            xv_256.x;
                        const scalar_t edge_value =
                            sh_edge[1];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dv_2.x; {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[1];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a74590261dp-1f);
                        u_0 -= scaled * edge_value;
                        v_0 -= scaled * input_value;
                    } {
                        const scalar_t input_value =
                            xv_256.x;
                        const scalar_t edge_value =
                            sh_edge[2];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a74590261dp+0f);
                        u_1 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[3];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a74590261dp-1f);
                        u_2 -= scaled * edge_value;
                        v_2 -= scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dv_3.x; {
                        const scalar_t input_value =
                            xv_256.x;
                        const scalar_t edge_value =
                            sh_edge[3];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[2];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dv_4.x; {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[1];
                        const scalar_t scaled = output_gradient;
                        u_0 -= scaled * edge_value;
                        v_0 -= scaled * input_value;
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[3];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    }
                }
                const scalar_t scaled_weight = radial_weight;
                weight_gradient +=
                    xv_128.x * u_0;
                input_gradient_128_0_0 += scaled_weight * u_0;
                weight_gradient +=
                    xv_384.x * u_2;
                input_gradient_128_2_0 += scaled_weight * u_2;
                weight_gradient +=
                    xv_256.x * u_1;
                input_gradient_128_1_0 += scaled_weight * u_1;
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
                    768 + (cbase + 1);
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (true) {
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
                    const scalar_t output_gradient = dv_0.y; {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[3];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[1];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dv_1.y; {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[2];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            xv_256.y;
                        const scalar_t edge_value =
                            sh_edge[1];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_0 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dv_2.y; {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[1];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a74590261dp-1f);
                        u_0 -= scaled * edge_value;
                        v_0 -= scaled * input_value;
                    } {
                        const scalar_t input_value =
                            xv_256.y;
                        const scalar_t edge_value =
                            sh_edge[2];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a74590261dp+0f);
                        u_1 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[3];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.279a74590261dp-1f);
                        u_2 -= scaled * edge_value;
                        v_2 -= scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dv_3.y; {
                        const scalar_t input_value =
                            xv_256.y;
                        const scalar_t edge_value =
                            sh_edge[3];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[2];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_1 += scaled * input_value;
                    }
                } {
                    const scalar_t output_gradient = dv_4.y; {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[1];
                        const scalar_t scaled = output_gradient;
                        u_0 -= scaled * edge_value;
                        v_0 -= scaled * input_value;
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[3];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        v_2 += scaled * input_value;
                    }
                }
                const scalar_t scaled_weight = radial_weight;
                weight_gradient +=
                    xv_128.y * u_0;
                input_gradient_128_0_1 += scaled_weight * u_0;
                weight_gradient +=
                    xv_384.y * u_2;
                input_gradient_128_2_1 += scaled_weight * u_2;
                weight_gradient +=
                    xv_256.y * u_1;
                input_gradient_128_1_1 += scaled_weight * u_1;
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
            }
            {
            const float2 dv_0 = *reinterpret_cast<const float2*>(stage_row + 1408 + 256 + cbase);
            const float2 dv_1 = *reinterpret_cast<const float2*>(stage_row + 1408 + 640 + cbase);
            const float2 dv_2 = *reinterpret_cast<const float2*>(stage_row + 1408 + 1024 + cbase);
            const float2 dv_3 = *reinterpret_cast<const float2*>(stage_row + 1408 + 1408 + cbase);
            const float2 dv_4 = *reinterpret_cast<const float2*>(stage_row + 1408 + 1792 + cbase);
{
                const int weight_index =
                    896 + (cbase + 0);
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (true) {
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

                 {
                    const scalar_t output_gradient = dv_0.x; {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[13];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362485537p-2f);
                        u_0 -= scaled * edge_value;
                        edge_gradient[13] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[15];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3988e14092cf4p+0f);
                        u_0 -= scaled * edge_value;
                        edge_gradient[15] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.x;
                        const scalar_t edge_value =
                            sh_edge[10];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        edge_gradient[10] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[9];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3988e14092cf4p+0f);
                        u_2 += scaled * edge_value;
                        edge_gradient[9] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[11];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362485537p-2f);
                        u_2 -= scaled * edge_value;
                        edge_gradient[11] -= scaled * (radial_weight * input_value);
                    }
                } {
                    const scalar_t output_gradient = dv_1.x; {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[12];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.8c97ef43f812bp-1f);
                        u_0 -= scaled * edge_value;
                        edge_gradient[12] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[14];
                        const scalar_t scaled = output_gradient;
                        u_0 -= scaled * edge_value;
                        edge_gradient[14] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.x;
                        const scalar_t edge_value =
                            sh_edge[11];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362485537p+0f);
                        u_1 += scaled * edge_value;
                        edge_gradient[11] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[10];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        edge_gradient[10] += scaled * (radial_weight * input_value);
                    }
                } {
                    const scalar_t output_gradient = dv_2.x; {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[11];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f88efap+0f);
                        u_0 += scaled * edge_value;
                        edge_gradient[11] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.x;
                        const scalar_t edge_value =
                            sh_edge[12];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.5775c544fff49p+0f);
                        u_1 += scaled * edge_value;
                        edge_gradient[12] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[13];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f88efap+0f);
                        u_2 += scaled * edge_value;
                        edge_gradient[13] += scaled * (radial_weight * input_value);
                    }
                } {
                    const scalar_t output_gradient = dv_3.x; {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[10];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[10] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.x;
                        const scalar_t edge_value =
                            sh_edge[13];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362485537p+0f);
                        u_1 += scaled * edge_value;
                        edge_gradient[13] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[12];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.8c97ef43f812bp-1f);
                        u_2 -= scaled * edge_value;
                        edge_gradient[12] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[14];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        edge_gradient[14] += scaled * (radial_weight * input_value);
                    }
                } {
                    const scalar_t output_gradient = dv_4.x; {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[9];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3988e14092cf4p+0f);
                        u_0 += scaled * edge_value;
                        edge_gradient[9] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[11];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362485537p-2f);
                        u_0 += scaled * edge_value;
                        edge_gradient[11] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.x;
                        const scalar_t edge_value =
                            sh_edge[14];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        edge_gradient[14] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[13];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362485537p-2f);
                        u_2 -= scaled * edge_value;
                        edge_gradient[13] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[15];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3988e14092cf4p+0f);
                        u_2 += scaled * edge_value;
                        edge_gradient[15] += scaled * (radial_weight * input_value);
                    }
                }
                const scalar_t scaled_weight = radial_weight;
                weight_gradient +=
                    xv_128.x * u_0;
                input_gradient_128_0_0 += scaled_weight * u_0;
                weight_gradient +=
                    xv_256.x * u_1;
                input_gradient_128_1_0 += scaled_weight * u_1;
                weight_gradient +=
                    xv_384.x * u_2;
                input_gradient_128_2_0 += scaled_weight * u_2;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
{
                const int weight_index =
                    896 + (cbase + 1);
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (true) {
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

                 {
                    const scalar_t output_gradient = dv_0.y; {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[13];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362485537p-2f);
                        u_0 -= scaled * edge_value;
                        edge_gradient[13] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[15];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3988e14092cf4p+0f);
                        u_0 -= scaled * edge_value;
                        edge_gradient[15] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.y;
                        const scalar_t edge_value =
                            sh_edge[10];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        edge_gradient[10] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[9];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3988e14092cf4p+0f);
                        u_2 += scaled * edge_value;
                        edge_gradient[9] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[11];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362485537p-2f);
                        u_2 -= scaled * edge_value;
                        edge_gradient[11] -= scaled * (radial_weight * input_value);
                    }
                } {
                    const scalar_t output_gradient = dv_1.y; {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[12];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.8c97ef43f812bp-1f);
                        u_0 -= scaled * edge_value;
                        edge_gradient[12] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[14];
                        const scalar_t scaled = output_gradient;
                        u_0 -= scaled * edge_value;
                        edge_gradient[14] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.y;
                        const scalar_t edge_value =
                            sh_edge[11];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362485537p+0f);
                        u_1 += scaled * edge_value;
                        edge_gradient[11] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[10];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        edge_gradient[10] += scaled * (radial_weight * input_value);
                    }
                } {
                    const scalar_t output_gradient = dv_2.y; {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[11];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f88efap+0f);
                        u_0 += scaled * edge_value;
                        edge_gradient[11] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.y;
                        const scalar_t edge_value =
                            sh_edge[12];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.5775c544fff49p+0f);
                        u_1 += scaled * edge_value;
                        edge_gradient[12] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[13];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f88efap+0f);
                        u_2 += scaled * edge_value;
                        edge_gradient[13] += scaled * (radial_weight * input_value);
                    }
                } {
                    const scalar_t output_gradient = dv_3.y; {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[10];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[10] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.y;
                        const scalar_t edge_value =
                            sh_edge[13];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362485537p+0f);
                        u_1 += scaled * edge_value;
                        edge_gradient[13] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[12];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.8c97ef43f812bp-1f);
                        u_2 -= scaled * edge_value;
                        edge_gradient[12] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[14];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        edge_gradient[14] += scaled * (radial_weight * input_value);
                    }
                } {
                    const scalar_t output_gradient = dv_4.y; {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[9];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3988e14092cf4p+0f);
                        u_0 += scaled * edge_value;
                        edge_gradient[9] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[11];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362485537p-2f);
                        u_0 += scaled * edge_value;
                        edge_gradient[11] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.y;
                        const scalar_t edge_value =
                            sh_edge[14];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        edge_gradient[14] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[13];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362485537p-2f);
                        u_2 -= scaled * edge_value;
                        edge_gradient[13] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[15];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3988e14092cf4p+0f);
                        u_2 += scaled * edge_value;
                        edge_gradient[15] += scaled * (radial_weight * input_value);
                    }
                }
                const scalar_t scaled_weight = radial_weight;
                weight_gradient +=
                    xv_128.y * u_0;
                input_gradient_128_0_1 += scaled_weight * u_0;
                weight_gradient +=
                    xv_256.y * u_1;
                input_gradient_128_1_1 += scaled_weight * u_1;
                weight_gradient +=
                    xv_384.y * u_2;
                input_gradient_128_2_1 += scaled_weight * u_2;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
            }
            {
            const float2 dv_0 = *reinterpret_cast<const float2*>(stage_row + 3328 + 0 + cbase);
            const float2 dv_1 = *reinterpret_cast<const float2*>(stage_row + 3328 + 256 + cbase);
            const float2 dv_2 = *reinterpret_cast<const float2*>(stage_row + 3328 + 512 + cbase);
            const float2 dv_3 = *reinterpret_cast<const float2*>(stage_row + 3328 + 768 + cbase);
            const float2 dv_4 = *reinterpret_cast<const float2*>(stage_row + 3328 + 1024 + cbase);
            const float2 dv_5 = *reinterpret_cast<const float2*>(stage_row + 3328 + 1280 + cbase);
            const float2 dv_6 = *reinterpret_cast<const float2*>(stage_row + 3328 + 1536 + cbase);
{
                const int weight_index =
                    1024 + (cbase + 0);
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (true) {
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
                    radial_weight * xv_0.x;

                 {
                    const scalar_t output_gradient = dv_0.x; {
                        const scalar_t input_value =
                            xv_0.x;
                        const scalar_t edge_value =
                            sh_edge[9];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[9]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dv_1.x; {
                        const scalar_t input_value =
                            xv_0.x;
                        const scalar_t edge_value =
                            sh_edge[10];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[10]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dv_2.x; {
                        const scalar_t input_value =
                            xv_0.x;
                        const scalar_t edge_value =
                            sh_edge[11];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[11]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dv_3.x; {
                        const scalar_t input_value =
                            xv_0.x;
                        const scalar_t edge_value =
                            sh_edge[12];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[12]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dv_4.x; {
                        const scalar_t input_value =
                            xv_0.x;
                        const scalar_t edge_value =
                            sh_edge[13];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[13]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dv_5.x; {
                        const scalar_t input_value =
                            xv_0.x;
                        const scalar_t edge_value =
                            sh_edge[14];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[14]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dv_6.x; {
                        const scalar_t input_value =
                            xv_0.x;
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
                    xv_0.x * u_0;
                input_gradient_0_0_0 += scaled_weight * u_0;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
{
                const int weight_index =
                    1024 + (cbase + 1);
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (true) {
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
                    radial_weight * xv_0.y;

                 {
                    const scalar_t output_gradient = dv_0.y; {
                        const scalar_t input_value =
                            xv_0.y;
                        const scalar_t edge_value =
                            sh_edge[9];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[9]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dv_1.y; {
                        const scalar_t input_value =
                            xv_0.y;
                        const scalar_t edge_value =
                            sh_edge[10];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[10]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dv_2.y; {
                        const scalar_t input_value =
                            xv_0.y;
                        const scalar_t edge_value =
                            sh_edge[11];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[11]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dv_3.y; {
                        const scalar_t input_value =
                            xv_0.y;
                        const scalar_t edge_value =
                            sh_edge[12];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[12]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dv_4.y; {
                        const scalar_t input_value =
                            xv_0.y;
                        const scalar_t edge_value =
                            sh_edge[13];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[13]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dv_5.y; {
                        const scalar_t input_value =
                            xv_0.y;
                        const scalar_t edge_value =
                            sh_edge[14];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[14]
                            += weighted_input * scaled;
                    }
                } {
                    const scalar_t output_gradient = dv_6.y; {
                        const scalar_t input_value =
                            xv_0.y;
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
                    xv_0.y * u_0;
                input_gradient_0_0_1 += scaled_weight * u_0;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
            }
            {
            const float2 dv_0 = *reinterpret_cast<const float2*>(stage_row + 3328 + 128 + cbase);
            const float2 dv_1 = *reinterpret_cast<const float2*>(stage_row + 3328 + 384 + cbase);
            const float2 dv_2 = *reinterpret_cast<const float2*>(stage_row + 3328 + 640 + cbase);
            const float2 dv_3 = *reinterpret_cast<const float2*>(stage_row + 3328 + 896 + cbase);
            const float2 dv_4 = *reinterpret_cast<const float2*>(stage_row + 3328 + 1152 + cbase);
            const float2 dv_5 = *reinterpret_cast<const float2*>(stage_row + 3328 + 1408 + cbase);
            const float2 dv_6 = *reinterpret_cast<const float2*>(stage_row + 3328 + 1664 + cbase);
{
                const int weight_index =
                    1152 + (cbase + 0);
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (true) {
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

                 {
                    const scalar_t output_gradient = dv_0.x; {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3988e14091337p+0f);
                        u_0 += scaled * edge_value;
                        edge_gradient[8] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3988e14091337p+0f);
                        u_2 += scaled * edge_value;
                        edge_gradient[4] += scaled * (radial_weight * input_value);
                    }
                } {
                    const scalar_t output_gradient = dv_1.x; {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[7] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.x;
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        edge_gradient[4] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        edge_gradient[5] += scaled * (radial_weight * input_value);
                    }
                } {
                    const scalar_t output_gradient = dv_2.x; {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f877f5p+0f);
                        u_0 += scaled * edge_value;
                        edge_gradient[6] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362483aa3p-2f);
                        u_0 += scaled * edge_value;
                        edge_gradient[8] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.x;
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362483aa3p+0f);
                        u_1 += scaled * edge_value;
                        edge_gradient[5] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362483aa3p-2f);
                        u_2 -= scaled * edge_value;
                        edge_gradient[4] -= scaled * (radial_weight * input_value);
                    }
                } {
                    const scalar_t output_gradient = dv_3.x; {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.8c97ef43f609cp-1f);
                        u_0 -= scaled * edge_value;
                        edge_gradient[5] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.x;
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.5775c544fe318p+0f);
                        u_1 += scaled * edge_value;
                        edge_gradient[6] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.8c97ef43f609cp-1f);
                        u_2 -= scaled * edge_value;
                        edge_gradient[7] -= scaled * (radial_weight * input_value);
                    }
                } {
                    const scalar_t output_gradient = dv_4.x; {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362483aa3p-2f);
                        u_0 -= scaled * edge_value;
                        edge_gradient[4] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.x;
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362483aa3p+0f);
                        u_1 += scaled * edge_value;
                        edge_gradient[7] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f877f5p+0f);
                        u_2 += scaled * edge_value;
                        edge_gradient[6] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362483aa3p-2f);
                        u_2 -= scaled * edge_value;
                        edge_gradient[8] -= scaled * (radial_weight * input_value);
                    }
                } {
                    const scalar_t output_gradient = dv_5.x; {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient;
                        u_0 -= scaled * edge_value;
                        edge_gradient[5] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.x;
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        edge_gradient[8] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        edge_gradient[7] += scaled * (radial_weight * input_value);
                    }
                } {
                    const scalar_t output_gradient = dv_6.x; {
                        const scalar_t input_value =
                            xv_128.x;
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3988e14091337p+0f);
                        u_0 -= scaled * edge_value;
                        edge_gradient[4] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.x;
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3988e14091337p+0f);
                        u_2 += scaled * edge_value;
                        edge_gradient[8] += scaled * (radial_weight * input_value);
                    }
                }
                const scalar_t scaled_weight = radial_weight;
                weight_gradient +=
                    xv_128.x * u_0;
                input_gradient_128_0_0 += scaled_weight * u_0;
                weight_gradient +=
                    xv_384.x * u_2;
                input_gradient_128_2_0 += scaled_weight * u_2;
                weight_gradient +=
                    xv_256.x * u_1;
                input_gradient_128_1_0 += scaled_weight * u_1;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
{
                const int weight_index =
                    1152 + (cbase + 1);
                const int64_t radial_weight_index = weight_base + weight_index;
                scalar_t gv[PADDED_TERMS];
                if (true) {
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

                 {
                    const scalar_t output_gradient = dv_0.y; {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3988e14091337p+0f);
                        u_0 += scaled * edge_value;
                        edge_gradient[8] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3988e14091337p+0f);
                        u_2 += scaled * edge_value;
                        edge_gradient[4] += scaled * (radial_weight * input_value);
                    }
                } {
                    const scalar_t output_gradient = dv_1.y; {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient;
                        u_0 += scaled * edge_value;
                        edge_gradient[7] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.y;
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        edge_gradient[4] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        edge_gradient[5] += scaled * (radial_weight * input_value);
                    }
                } {
                    const scalar_t output_gradient = dv_2.y; {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f877f5p+0f);
                        u_0 += scaled * edge_value;
                        edge_gradient[6] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362483aa3p-2f);
                        u_0 += scaled * edge_value;
                        edge_gradient[8] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.y;
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362483aa3p+0f);
                        u_1 += scaled * edge_value;
                        edge_gradient[5] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362483aa3p-2f);
                        u_2 -= scaled * edge_value;
                        edge_gradient[4] -= scaled * (radial_weight * input_value);
                    }
                } {
                    const scalar_t output_gradient = dv_3.y; {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.8c97ef43f609cp-1f);
                        u_0 -= scaled * edge_value;
                        edge_gradient[5] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.y;
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.5775c544fe318p+0f);
                        u_1 += scaled * edge_value;
                        edge_gradient[6] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.8c97ef43f609cp-1f);
                        u_2 -= scaled * edge_value;
                        edge_gradient[7] -= scaled * (radial_weight * input_value);
                    }
                } {
                    const scalar_t output_gradient = dv_4.y; {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362483aa3p-2f);
                        u_0 -= scaled * edge_value;
                        edge_gradient[4] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.y;
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362483aa3p+0f);
                        u_1 += scaled * edge_value;
                        edge_gradient[7] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[6];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.186f174f877f5p+0f);
                        u_2 += scaled * edge_value;
                        edge_gradient[6] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.43d1362483aa3p-2f);
                        u_2 -= scaled * edge_value;
                        edge_gradient[8] -= scaled * (radial_weight * input_value);
                    }
                } {
                    const scalar_t output_gradient = dv_5.y; {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[5];
                        const scalar_t scaled = output_gradient;
                        u_0 -= scaled * edge_value;
                        edge_gradient[5] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_256.y;
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient;
                        u_1 += scaled * edge_value;
                        edge_gradient[8] += scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[7];
                        const scalar_t scaled = output_gradient;
                        u_2 += scaled * edge_value;
                        edge_gradient[7] += scaled * (radial_weight * input_value);
                    }
                } {
                    const scalar_t output_gradient = dv_6.y; {
                        const scalar_t input_value =
                            xv_128.y;
                        const scalar_t edge_value =
                            sh_edge[4];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3988e14091337p+0f);
                        u_0 -= scaled * edge_value;
                        edge_gradient[4] -= scaled * (radial_weight * input_value);
                    } {
                        const scalar_t input_value =
                            xv_384.y;
                        const scalar_t edge_value =
                            sh_edge[8];
                        const scalar_t scaled = output_gradient * static_cast<scalar_t>(0x1.3988e14091337p+0f);
                        u_2 += scaled * edge_value;
                        edge_gradient[8] += scaled * (radial_weight * input_value);
                    }
                }
                const scalar_t scaled_weight = radial_weight;
                weight_gradient +=
                    xv_128.y * u_0;
                input_gradient_128_0_1 += scaled_weight * u_0;
                weight_gradient +=
                    xv_384.y * u_2;
                input_gradient_128_2_1 += scaled_weight * u_2;
                weight_gradient +=
                    xv_256.y * u_1;
                input_gradient_128_1_1 += scaled_weight * u_1;
                if (sloped) {
                    acc_t slope = acc_t(0);
                    slope = radial_dt * sc;
                    radial_gradient += weight_gradient * slope;
                }
            }
            }
#if __CUDA_ARCH__ >= 900
            atomicAdd(reinterpret_cast<float2*>(dx + input_base + 0 + cbase), make_float2(input_gradient_0_0_0, input_gradient_0_0_1));
#else
            atomicAdd(dx + input_base + 0 + cbase + 0, input_gradient_0_0_0);
            atomicAdd(dx + input_base + 0 + cbase + 1, input_gradient_0_0_1);
#endif
#if __CUDA_ARCH__ >= 900
            atomicAdd(reinterpret_cast<float2*>(dx + input_base + 128 + cbase), make_float2(input_gradient_128_0_0, input_gradient_128_0_1));
#else
            atomicAdd(dx + input_base + 128 + cbase + 0, input_gradient_128_0_0);
            atomicAdd(dx + input_base + 128 + cbase + 1, input_gradient_128_0_1);
#endif
#if __CUDA_ARCH__ >= 900
            atomicAdd(reinterpret_cast<float2*>(dx + input_base + 256 + cbase), make_float2(input_gradient_128_1_0, input_gradient_128_1_1));
#else
            atomicAdd(dx + input_base + 256 + cbase + 0, input_gradient_128_1_0);
            atomicAdd(dx + input_base + 256 + cbase + 1, input_gradient_128_1_1);
#endif
#if __CUDA_ARCH__ >= 900
            atomicAdd(reinterpret_cast<float2*>(dx + input_base + 384 + cbase), make_float2(input_gradient_128_2_0, input_gradient_128_2_1));
#else
            atomicAdd(dx + input_base + 384 + cbase + 0, input_gradient_128_2_0);
            atomicAdd(dx + input_base + 384 + cbase + 1, input_gradient_128_2_1);
#endif
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
    }  // STAGE_DOUT: the run's edges
    __syncthreads();  // every warp is done with the row before the next copy
    stage_s = stage_t1;
    }  // STAGE_DOUT: the runs
}