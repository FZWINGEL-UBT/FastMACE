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

            // LATE_PULLBACK: the receiver row staged per channel; direct paths as q = M g.
            for (int channel = threadIdx.x; channel < CHANNEL_DIM; channel += blockDim.x) {
                {  // path 0 (0, 0, 0) direct
                const float g0 = __ldg(dout0 + int64_t(stage_r) * 256 + 0 + channel);
                stage_row[0 + channel] = 1.0f * g0;
                }
                {  // path 1 (1, 1, 0) direct
                const float g0 = __ldg(dout0 + int64_t(stage_r) * 256 + 128 + channel);
                stage_row[128 + channel] = 1.73205081f * g0;
                }
                {  // path 2 (0, 1, 1) direct
                const float g0 = __ldg(dout1 + int64_t(stage_r) * 1152 + 0 + channel);
                const float g1 = __ldg(dout1 + int64_t(stage_r) * 1152 + 384 + channel);
                const float g2 = __ldg(dout1 + int64_t(stage_r) * 1152 + 768 + channel);
                stage_row[256 + channel] = 1.73205081f * g0;
                stage_row[640 + channel] = 1.73205081f * g1;
                stage_row[1024 + channel] = 1.73205081f * g2;
                }
                {  // path 3 (1, 0, 1) direct
                const float g0 = __ldg(dout1 + int64_t(stage_r) * 1152 + 128 + channel);
                const float g1 = __ldg(dout1 + int64_t(stage_r) * 1152 + 512 + channel);
                const float g2 = __ldg(dout1 + int64_t(stage_r) * 1152 + 896 + channel);
                stage_row[384 + channel] = 1.0f * g0;
                stage_row[768 + channel] = 1.0f * g1;
                stage_row[1152 + channel] = 1.0f * g2;
                }
                {  // path 4 (1, 2, 1) direct
                const float g0 = __ldg(dout1 + int64_t(stage_r) * 1152 + 256 + channel);
                const float g1 = __ldg(dout1 + int64_t(stage_r) * 1152 + 640 + channel);
                const float g2 = __ldg(dout1 + int64_t(stage_r) * 1152 + 1024 + channel);
                stage_row[512 + channel] = 3.87298335f * g0;
                stage_row[896 + channel] = 3.87298335f * g1;
                stage_row[1280 + channel] = 3.87298335f * g2;
                }
                {  // path 5 (0, 2, 2) direct
                const float g0 = __ldg(dout2 + int64_t(stage_r) * 1920 + 0 + channel);
                const float g1 = __ldg(dout2 + int64_t(stage_r) * 1920 + 384 + channel);
                const float g2 = __ldg(dout2 + int64_t(stage_r) * 1920 + 768 + channel);
                const float g3 = __ldg(dout2 + int64_t(stage_r) * 1920 + 1152 + channel);
                const float g4 = __ldg(dout2 + int64_t(stage_r) * 1920 + 1536 + channel);
                stage_row[1408 + channel] = -1.11803399f * g2 + -1.93649167f * g4;
                stage_row[1792 + channel] = 1.93649167f * g1;
                stage_row[2176 + channel] = 1.93649167f * g0;
                stage_row[2560 + channel] = 2.23606798f * g2;
                stage_row[2944 + channel] = 1.93649167f * g3;
                }
                {  // path 6 (1, 1, 2) direct
                const float g0 = __ldg(dout2 + int64_t(stage_r) * 1920 + 128 + channel);
                const float g1 = __ldg(dout2 + int64_t(stage_r) * 1920 + 512 + channel);
                const float g2 = __ldg(dout2 + int64_t(stage_r) * 1920 + 896 + channel);
                const float g3 = __ldg(dout2 + int64_t(stage_r) * 1920 + 1280 + channel);
                const float g4 = __ldg(dout2 + int64_t(stage_r) * 1920 + 1664 + channel);
                stage_row[1536 + channel] = -0.5f * g2 + -0.866025404f * g4;
                stage_row[1920 + channel] = 0.866025404f * g1;
                stage_row[2304 + channel] = 0.866025404f * g0;
                stage_row[2688 + channel] = 1.0f * g2;
                stage_row[3072 + channel] = 0.866025404f * g3;
                }
                {  // path 7 (1, 3, 2) direct
                const float g0 = __ldg(dout2 + int64_t(stage_r) * 1920 + 256 + channel);
                const float g1 = __ldg(dout2 + int64_t(stage_r) * 1920 + 640 + channel);
                const float g2 = __ldg(dout2 + int64_t(stage_r) * 1920 + 1024 + channel);
                const float g3 = __ldg(dout2 + int64_t(stage_r) * 1920 + 1408 + channel);
                const float g4 = __ldg(dout2 + int64_t(stage_r) * 1920 + 1792 + channel);
                stage_row[1664 + channel] = -2.95803989f * g2 + -5.12347538f * g4;
                stage_row[2048 + channel] = 5.12347538f * g1;
                stage_row[2432 + channel] = 5.12347538f * g0;
                stage_row[2816 + channel] = 5.91607978f * g2;
                stage_row[3200 + channel] = 5.12347538f * g3;
                }
                {  // path 8 (0, 3, 3) direct
                const float g0 = __ldg(dout3 + int64_t(stage_r) * 1792 + 0 + channel);
                const float g1 = __ldg(dout3 + int64_t(stage_r) * 1792 + 256 + channel);
                const float g2 = __ldg(dout3 + int64_t(stage_r) * 1792 + 512 + channel);
                const float g3 = __ldg(dout3 + int64_t(stage_r) * 1792 + 768 + channel);
                const float g4 = __ldg(dout3 + int64_t(stage_r) * 1792 + 1024 + channel);
                const float g5 = __ldg(dout3 + int64_t(stage_r) * 1792 + 1280 + channel);
                const float g6 = __ldg(dout3 + int64_t(stage_r) * 1792 + 1536 + channel);
                stage_row[3328 + channel] = -2.09165007f * g0 + -1.62018517f * g2;
                stage_row[3584 + channel] = -1.32287566f * g3 + -1.70782513f * g5;
                stage_row[3840 + channel] = -0.540061725f * g4 + -2.09165007f * g6;
                stage_row[4096 + channel] = 2.1602469f * g2;
                stage_row[4352 + channel] = 1.70782513f * g1;
                stage_row[4608 + channel] = 2.64575131f * g3;
                stage_row[4864 + channel] = 2.1602469f * g4;
                }
                {  // path 9 (1, 2, 3) direct
                const float g0 = __ldg(dout3 + int64_t(stage_r) * 1792 + 128 + channel);
                const float g1 = __ldg(dout3 + int64_t(stage_r) * 1792 + 384 + channel);
                const float g2 = __ldg(dout3 + int64_t(stage_r) * 1792 + 640 + channel);
                const float g3 = __ldg(dout3 + int64_t(stage_r) * 1792 + 896 + channel);
                const float g4 = __ldg(dout3 + int64_t(stage_r) * 1792 + 1152 + channel);
                const float g5 = __ldg(dout3 + int64_t(stage_r) * 1792 + 1408 + channel);
                const float g6 = __ldg(dout3 + int64_t(stage_r) * 1792 + 1664 + channel);
                stage_row[3456 + channel] = -0.790569415f * g0 + -0.612372436f * g2;
                stage_row[3712 + channel] = -0.5f * g3 + -0.645497224f * g5;
                stage_row[3968 + channel] = -0.204124145f * g4 + -0.790569415f * g6;
                stage_row[4224 + channel] = 0.816496581f * g2;
                stage_row[4480 + channel] = 0.645497224f * g1;
                stage_row[4736 + channel] = 1.0f * g3;
                stage_row[4992 + channel] = 0.816496581f * g4;
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

    if (!inside) {  // skip: at or beyond the cutoff (padding edges) w = 0: nothing but zeros
        for (int c = lane; c < EDGE_DIM; c += LOGICAL_GROUP_SIZE) dsh[edge_base + c] = scalar_t(0);
        if (lane == 0) dr[e] = scalar_t(0);
        continue;
    }
    acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};
    acc_t radial_gradient = acc_t(0);
    // LATE_PULLBACK: G = sum w grad phi and R = sum (dw/dt) phi over the direct paths; n in the sh frame.
    const float nx = sh_edge[1] * 0.57735026918962576451f;
    const float ny = sh_edge[2] * 0.57735026918962576451f;
    const float nz = sh_edge[3] * 0.57735026918962576451f;
    float pull_Gx = 0.f, pull_Gy = 0.f, pull_Gz = 0.f, pull_R = 0.f;
    scalar_t* dx_lane = dx + input_base + lane;  // hoist
    for (int channel = lane; channel < CHANNEL_DIM;
         channel += LOGICAL_GROUP_SIZE) {

            acc_t input_gradient_0_0 = acc_t(0);
            acc_t input_gradient_128_0 = acc_t(0);
            acc_t input_gradient_128_1 = acc_t(0);
            acc_t input_gradient_128_2 = acc_t(0);{
                const int weight_index =
                    0 + channel;
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
                // LATE_PULLBACK path 0 (0, 0, 0)
                const float w = radial_weight;
                const float q0 = stage_row[0 + channel];
                const float x0 = x[input_base + 0 + channel];
                input_gradient_0_0 = fmaf(w, q0, input_gradient_0_0);
                pull_R = fmaf(radial_dt, x0 * q0, pull_R);
            }{
                const int weight_index =
                    128 + channel;
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
                // LATE_PULLBACK path 1 (1, 1, 0)
                const float w = radial_weight;
                const float q0 = stage_row[128 + channel];
                const float xx = x[input_base + 128 + channel];
                const float xy = x[input_base + 256 + channel];
                const float xz = x[input_base + 384 + channel];
                const float A = xx * nx + xy * ny + xz * nz;
                const float wq = w * q0;
                pull_Gx = fmaf(wq, xx, pull_Gx);
                pull_Gy = fmaf(wq, xy, pull_Gy);
                pull_Gz = fmaf(wq, xz, pull_Gz);
                input_gradient_128_0 = fmaf(wq, nx, input_gradient_128_0);
                input_gradient_128_1 = fmaf(wq, ny, input_gradient_128_1);
                input_gradient_128_2 = fmaf(wq, nz, input_gradient_128_2);
                pull_R = fmaf(radial_dt, A * q0, pull_R);
            }{
                const int weight_index =
                    256 + channel;
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
                // LATE_PULLBACK path 2 (0, 1, 1)
                const float w = radial_weight;
                const float q0 = stage_row[256 + channel];
                const float q1 = stage_row[640 + channel];
                const float q2 = stage_row[1024 + channel];
                const float x0 = x[input_base + 0 + channel];
                const float h = q0 * nx + q1 * ny + q2 * nz;
                const float wx = w * x0;
                pull_Gx = fmaf(wx, q0, pull_Gx);
                pull_Gy = fmaf(wx, q1, pull_Gy);
                pull_Gz = fmaf(wx, q2, pull_Gz);
                input_gradient_0_0 = fmaf(w, h, input_gradient_0_0);
                pull_R = fmaf(radial_dt, x0 * h, pull_R);
            }{
                const int weight_index =
                    384 + channel;
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
                // LATE_PULLBACK path 3 (1, 0, 1)
                const float w = radial_weight;
                const float q0 = stage_row[384 + channel];
                const float q1 = stage_row[768 + channel];
                const float q2 = stage_row[1152 + channel];
                const float xx = x[input_base + 128 + channel];
                const float xy = x[input_base + 256 + channel];
                const float xz = x[input_base + 384 + channel];
                const float h = q0 * nx + q1 * ny + q2 * nz;
                pull_R = fmaf(radial_dt, xx * q0 + xy * q1 + xz * q2, pull_R);
                input_gradient_128_0 = fmaf(w, q0, input_gradient_128_0);
                input_gradient_128_1 = fmaf(w, q1, input_gradient_128_1);
                input_gradient_128_2 = fmaf(w, q2, input_gradient_128_2);
            }{
                const int weight_index =
                    512 + channel;
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
                // LATE_PULLBACK path 4 (1, 2, 1)
                const float w = radial_weight;
                const float q0 = stage_row[512 + channel];
                const float q1 = stage_row[896 + channel];
                const float q2 = stage_row[1280 + channel];
                const float xx = x[input_base + 128 + channel];
                const float xy = x[input_base + 256 + channel];
                const float xz = x[input_base + 384 + channel];
                const float h = q0 * nx + q1 * ny + q2 * nz;
                const float A = xx * nx + xy * ny + xz * nz;
                const float B = xx * q0 + xy * q1 + xz * q2;
                pull_R = fmaf(radial_dt, A * h - 0.333333333f * B, pull_R);
                const float wh = w * h, wA = w * A, wk = w * 0.333333333f;
                pull_Gx = fmaf(wh, xx, fmaf(wA, q0, pull_Gx));
                pull_Gy = fmaf(wh, xy, fmaf(wA, q1, pull_Gy));
                pull_Gz = fmaf(wh, xz, fmaf(wA, q2, pull_Gz));
                input_gradient_128_0 = fmaf(wh, nx, fmaf(-wk, q0, input_gradient_128_0));
                input_gradient_128_1 = fmaf(wh, ny, fmaf(-wk, q1, input_gradient_128_1));
                input_gradient_128_2 = fmaf(wh, nz, fmaf(-wk, q2, input_gradient_128_2));
            }{
                const int weight_index =
                    640 + channel;
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
                // LATE_PULLBACK path 5 (0, 2, 2)
                const float w = radial_weight;
                const float q0 = stage_row[1408 + channel];
                const float q1 = stage_row[1792 + channel];
                const float q2 = stage_row[2176 + channel];
                const float q3 = stage_row[2560 + channel];
                const float q4 = stage_row[2944 + channel];
                const float x0 = x[input_base + 0 + channel];
                const float qzz = -q0 - q3;
                const float vx = q0 * nx + q1 * ny + q2 * nz, vy = q1 * nx + q3 * ny + q4 * nz, vz = q2 * nx + q4 * ny + qzz * nz;
                const float h = nx * vx + ny * vy + nz * vz;
                const float wx = 2.0f * w * x0;
                pull_Gx = fmaf(wx, vx, pull_Gx);
                pull_Gy = fmaf(wx, vy, pull_Gy);
                pull_Gz = fmaf(wx, vz, pull_Gz);
                input_gradient_0_0 = fmaf(w, h, input_gradient_0_0);
                pull_R = fmaf(radial_dt, x0 * h, pull_R);
            }{
                const int weight_index =
                    768 + channel;
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
                // LATE_PULLBACK path 6 (1, 1, 2)
                const float w = radial_weight;
                const float q0 = stage_row[1536 + channel];
                const float q1 = stage_row[1920 + channel];
                const float q2 = stage_row[2304 + channel];
                const float q3 = stage_row[2688 + channel];
                const float q4 = stage_row[3072 + channel];
                const float xx = x[input_base + 128 + channel];
                const float xy = x[input_base + 256 + channel];
                const float xz = x[input_base + 384 + channel];
                const float qzz = -q0 - q3;
                const float vx = q0 * nx + q1 * ny + q2 * nz, vy = q1 * nx + q3 * ny + q4 * nz, vz = q2 * nx + q4 * ny + qzz * nz;
                const float h = nx * vx + ny * vy + nz * vz;
                pull_R = fmaf(radial_dt, 2.f * (xx * vx + xy * vy + xz * vz), pull_R);
                const float w2 = 2.f * w;
                input_gradient_128_0 = fmaf(w2, vx, input_gradient_128_0);
                input_gradient_128_1 = fmaf(w2, vy, input_gradient_128_1);
                input_gradient_128_2 = fmaf(w2, vz, input_gradient_128_2);
                pull_Gx = fmaf(w2, q0 * xx + q1 * xy + q2 * xz, pull_Gx);
                pull_Gy = fmaf(w2, q1 * xx + q3 * xy + q4 * xz, pull_Gy);
                pull_Gz = fmaf(w2, q2 * xx + q4 * xy + qzz * xz, pull_Gz);
            }{
                const int weight_index =
                    896 + channel;
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
                // LATE_PULLBACK path 7 (1, 3, 2)
                const float w = radial_weight;
                const float q0 = stage_row[1664 + channel];
                const float q1 = stage_row[2048 + channel];
                const float q2 = stage_row[2432 + channel];
                const float q3 = stage_row[2816 + channel];
                const float q4 = stage_row[3200 + channel];
                const float xx = x[input_base + 128 + channel];
                const float xy = x[input_base + 256 + channel];
                const float xz = x[input_base + 384 + channel];
                const float qzz = -q0 - q3;
                const float vx = q0 * nx + q1 * ny + q2 * nz, vy = q1 * nx + q3 * ny + q4 * nz, vz = q2 * nx + q4 * ny + qzz * nz;
                const float h = nx * vx + ny * vy + nz * vz;
                const float A = xx * nx + xy * ny + xz * nz;
                const float B2 = xx * vx + xy * vy + xz * vz;  // B / 2
                pull_R = fmaf(radial_dt, A * h - 0.4f * B2, pull_R);
                const float wh = w * h, wA2 = 2.f * w * A, wk2 = 0.4f * w;
                pull_Gx = fmaf(wh, xx, fmaf(wA2, vx, fmaf(-wk2, q0 * xx + q1 * xy + q2 * xz, pull_Gx)));
                pull_Gy = fmaf(wh, xy, fmaf(wA2, vy, fmaf(-wk2, q1 * xx + q3 * xy + q4 * xz, pull_Gy)));
                pull_Gz = fmaf(wh, xz, fmaf(wA2, vz, fmaf(-wk2, q2 * xx + q4 * xy + qzz * xz, pull_Gz)));
                input_gradient_128_0 = fmaf(wh, nx, fmaf(-wk2, vx, input_gradient_128_0));
                input_gradient_128_1 = fmaf(wh, ny, fmaf(-wk2, vy, input_gradient_128_1));
                input_gradient_128_2 = fmaf(wh, nz, fmaf(-wk2, vz, input_gradient_128_2));
            }{
                const int weight_index =
                    1024 + channel;
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
                // LATE_PULLBACK path 8 (0, 3, 3)
                const float w = radial_weight;
                const float q0 = stage_row[3328 + channel];
                const float q1 = stage_row[3584 + channel];
                const float q2 = stage_row[3840 + channel];
                const float q3 = stage_row[4096 + channel];
                const float q4 = stage_row[4352 + channel];
                const float q5 = stage_row[4608 + channel];
                const float q6 = stage_row[4864 + channel];
                const float x0 = x[input_base + 0 + channel];
                const float q7 = -q0 - q3, q8 = -q1 - q5;
                const float Qxx = q0 * nx + q1 * ny + q2 * nz, Qxy = q1 * nx + q3 * ny + q4 * nz, Qxz = q2 * nx + q4 * ny + q7 * nz;
                const float Qyy = q3 * nx + q5 * ny + q6 * nz, Qyz = q4 * nx + q6 * ny + q8 * nz, Qzz = -Qxx - Qyy;
                const float vx = Qxx * nx + Qxy * ny + Qxz * nz, vy = Qxy * nx + Qyy * ny + Qyz * nz, vz = Qxz * nx + Qyz * ny + Qzz * nz;
                const float h = nx * vx + ny * vy + nz * vz;
                const float wx = 3.0f * w * x0;
                pull_Gx = fmaf(wx, vx, pull_Gx);
                pull_Gy = fmaf(wx, vy, pull_Gy);
                pull_Gz = fmaf(wx, vz, pull_Gz);
                input_gradient_0_0 = fmaf(w, h, input_gradient_0_0);
                pull_R = fmaf(radial_dt, x0 * h, pull_R);
            }{
                const int weight_index =
                    1152 + channel;
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
                // LATE_PULLBACK path 9 (1, 2, 3)
                const float w = radial_weight;
                const float q0 = stage_row[3456 + channel];
                const float q1 = stage_row[3712 + channel];
                const float q2 = stage_row[3968 + channel];
                const float q3 = stage_row[4224 + channel];
                const float q4 = stage_row[4480 + channel];
                const float q5 = stage_row[4736 + channel];
                const float q6 = stage_row[4992 + channel];
                const float xx = x[input_base + 128 + channel];
                const float xy = x[input_base + 256 + channel];
                const float xz = x[input_base + 384 + channel];
                const float q7 = -q0 - q3, q8 = -q1 - q5;
                const float Qxx = q0 * nx + q1 * ny + q2 * nz, Qxy = q1 * nx + q3 * ny + q4 * nz, Qxz = q2 * nx + q4 * ny + q7 * nz;
                const float Qyy = q3 * nx + q5 * ny + q6 * nz, Qyz = q4 * nx + q6 * ny + q8 * nz, Qzz = -Qxx - Qyy;
                const float vx = Qxx * nx + Qxy * ny + Qxz * nz, vy = Qxy * nx + Qyy * ny + Qyz * nz, vz = Qxz * nx + Qyz * ny + Qzz * nz;
                const float h = nx * vx + ny * vy + nz * vz;
                pull_R = fmaf(radial_dt, 3.f * (xx * vx + xy * vy + xz * vz), pull_R);
                const float w3 = 3.f * w, w6 = 6.f * w;
                input_gradient_128_0 = fmaf(w3, vx, input_gradient_128_0);
                input_gradient_128_1 = fmaf(w3, vy, input_gradient_128_1);
                input_gradient_128_2 = fmaf(w3, vz, input_gradient_128_2);
                pull_Gx = fmaf(w6, Qxx * xx + Qxy * xy + Qxz * xz, pull_Gx);
                pull_Gy = fmaf(w6, Qxy * xx + Qyy * xy + Qyz * xz, pull_Gy);
                pull_Gz = fmaf(w6, Qxz * xx + Qyz * xy + Qzz * xz, pull_Gz);
            }
            atomicAdd(
                dx_lane + 0,
                scalar_t(input_gradient_0_0));
            atomicAdd(
                dx_lane + 128,
                scalar_t(input_gradient_128_0));
            atomicAdd(
                dx_lane + 256,
                scalar_t(input_gradient_128_1));
            atomicAdd(
                dx_lane + 384,
                scalar_t(input_gradient_128_2));
            dx_lane += LOGICAL_GROUP_SIZE;

    }
    // LATE_PULLBACK: the l = 1 slots take G / sqrt(3) (edge_sh3's backward: (I - n n^T) G / r); dr takes R.
    edge_gradient[1] += pull_Gx * 0.57735026918962576451f;
    edge_gradient[2] += pull_Gy * 0.57735026918962576451f;
    edge_gradient[3] += pull_Gz * 0.57735026918962576451f;
    if (sloped) radial_gradient += pull_R * sc;
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