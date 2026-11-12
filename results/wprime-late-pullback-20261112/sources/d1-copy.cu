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
    // late-pullback design 1: per lane G = sum w grad phi (3) and R = sum dw/dt phi over paths and channels;
    // dL/dv once per edge after the reduction. n in the sh frame (sh_1 = sqrt(3) n).
    const float nx = sh_edge[1] * 0.57735026918962576451f;
    const float ny = sh_edge[2] * 0.57735026918962576451f;
    const float nz = sh_edge[3] * 0.57735026918962576451f;
    float Gx = 0.f, Gy = 0.f, Gz = 0.f, R = 0.f;
    scalar_t* dx_lane = dx + input_base + lane;  // hoist
    for (int channel = lane; channel < CHANNEL_DIM; channel += LOGICAL_GROUP_SIZE) {
        const float x0 = x[input_base + 0 + channel];
        float dx0 = 0.f;
        const float x128x = x[input_base + 128 + channel];
        float dx128x = 0.f;
        const float x128y = x[input_base + 256 + channel];
        float dx128y = 0.f;
        const float x128z = x[input_base + 384 + channel];
        float dx128z = 0.f;
        {
        // path 0: (0, 0, 0)
        const int weight_index = 0 + channel;
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

        const float w = radial_weight;
        const float q0 = stage_row[0 + channel];
        dx0 = fmaf(w, q0, dx0);
        R = fmaf(radial_dt, x0 * q0, R);
        }
        {
        // path 1: (1, 1, 0)
        const int weight_index = 128 + channel;
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

        const float w = radial_weight;
        const float q0 = stage_row[128 + channel];
        const float A = x128x * nx + x128y * ny + x128z * nz;
        const float wq = w * q0;
        Gx = fmaf(wq, x128x, Gx); Gy = fmaf(wq, x128y, Gy); Gz = fmaf(wq, x128z, Gz);
        dx128x = fmaf(wq, nx, dx128x); dx128y = fmaf(wq, ny, dx128y); dx128z = fmaf(wq, nz, dx128z);
        R = fmaf(radial_dt, A * q0, R);
        }
        {
        // path 2: (0, 1, 1)
        const int weight_index = 256 + channel;
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

        const float w = radial_weight;
        const float q0 = stage_row[256 + channel];
        const float q1 = stage_row[640 + channel];
        const float q2 = stage_row[1024 + channel];
        const float h = q0 * nx + q1 * ny + q2 * nz;
        const float wx = w * x0;
        Gx = fmaf(wx, q0, Gx); Gy = fmaf(wx, q1, Gy); Gz = fmaf(wx, q2, Gz);
        dx0 = fmaf(w, h, dx0);
        R = fmaf(radial_dt, x0 * h, R);
        }
        {
        // path 3: (1, 0, 1)
        const int weight_index = 384 + channel;
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

        const float w = radial_weight;
        const float q0 = stage_row[384 + channel];
        const float q1 = stage_row[768 + channel];
        const float q2 = stage_row[1152 + channel];
        R = fmaf(radial_dt, x128x * q0 + x128y * q1 + x128z * q2, R);
        dx128x = fmaf(w, q0, dx128x); dx128y = fmaf(w, q1, dx128y); dx128z = fmaf(w, q2, dx128z);
        }
        {
        // path 4: (1, 2, 1)
        const int weight_index = 512 + channel;
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

        const float w = radial_weight;
        const float q0 = stage_row[512 + channel];
        const float q1 = stage_row[896 + channel];
        const float q2 = stage_row[1280 + channel];
        const float A = x128x * nx + x128y * ny + x128z * nz;
        const float h = q0 * nx + q1 * ny + q2 * nz;
        const float B = x128x * q0 + x128y * q1 + x128z * q2;
        R = fmaf(radial_dt, A * h - 0.333333333f * B, R);
        const float wh = w * h, wA = w * A, wk = w * 0.333333333f;
        Gx = fmaf(wh, x128x, fmaf(wA, q0, Gx)); Gy = fmaf(wh, x128y, fmaf(wA, q1, Gy)); Gz = fmaf(wh, x128z, fmaf(wA, q2, Gz));
        dx128x = fmaf(wh, nx, fmaf(-wk, q0, dx128x)); dx128y = fmaf(wh, ny, fmaf(-wk, q1, dx128y)); dx128z = fmaf(wh, nz, fmaf(-wk, q2, dx128z));
        }
        {
        // path 5: (0, 2, 2)
        const int weight_index = 640 + channel;
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

        const float w = radial_weight;
        const float q0 = stage_row[1408 + channel];
        const float q1 = stage_row[1792 + channel];
        const float q2 = stage_row[2176 + channel];
        const float q3 = stage_row[2560 + channel];
        const float q4 = stage_row[2944 + channel];
        const float qzz = -q0 - q3;
        const float vx = q0 * nx + q1 * ny + q2 * nz, vy = q1 * nx + q3 * ny + q4 * nz, vz = q2 * nx + q4 * ny + qzz * nz;
        const float h = nx * vx + ny * vy + nz * vz;
        const float wx = 2.0f * w * x0;
        Gx = fmaf(wx, vx, Gx); Gy = fmaf(wx, vy, Gy); Gz = fmaf(wx, vz, Gz);
        dx0 = fmaf(w, h, dx0);
        R = fmaf(radial_dt, x0 * h, R);
        }
        {
        // path 6: (1, 1, 2)
        const int weight_index = 768 + channel;
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

        const float w = radial_weight;
        const float q0 = stage_row[1536 + channel];
        const float q1 = stage_row[1920 + channel];
        const float q2 = stage_row[2304 + channel];
        const float q3 = stage_row[2688 + channel];
        const float q4 = stage_row[3072 + channel];
        const float qzz = -q0 - q3;
        const float vx = q0 * nx + q1 * ny + q2 * nz, vy = q1 * nx + q3 * ny + q4 * nz, vz = q2 * nx + q4 * ny + qzz * nz;
        const float h = nx * vx + ny * vy + nz * vz;
        R = fmaf(radial_dt, 2.f * (x128x * vx + x128y * vy + x128z * vz), R);
        const float w2 = 2.f * w;
        dx128x = fmaf(w2, vx, dx128x); dx128y = fmaf(w2, vy, dx128y); dx128z = fmaf(w2, vz, dx128z);
        Gx = fmaf(w2, q0 * x128x + q1 * x128y + q2 * x128z, Gx);
        Gy = fmaf(w2, q1 * x128x + q3 * x128y + q4 * x128z, Gy);
        Gz = fmaf(w2, q2 * x128x + q4 * x128y + qzz * x128z, Gz);
        }
        {
        // path 7: (1, 3, 2)
        const int weight_index = 896 + channel;
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

        const float w = radial_weight;
        const float q0 = stage_row[1664 + channel];
        const float q1 = stage_row[2048 + channel];
        const float q2 = stage_row[2432 + channel];
        const float q3 = stage_row[2816 + channel];
        const float q4 = stage_row[3200 + channel];
        const float qzz = -q0 - q3;
        const float vx = q0 * nx + q1 * ny + q2 * nz, vy = q1 * nx + q3 * ny + q4 * nz, vz = q2 * nx + q4 * ny + qzz * nz;
        const float h = nx * vx + ny * vy + nz * vz;
        const float A = x128x * nx + x128y * ny + x128z * nz;
        const float B2 = x128x * vx + x128y * vy + x128z * vz;  // B / 2
        R = fmaf(radial_dt, A * h - 0.4f * B2, R);
        const float wh = w * h, wA2 = 2.f * w * A, wk2 = 0.4f * w;
        Gx = fmaf(wh, x128x, fmaf(wA2, vx, fmaf(-wk2, q0 * x128x + q1 * x128y + q2 * x128z, Gx)));
        Gy = fmaf(wh, x128y, fmaf(wA2, vy, fmaf(-wk2, q1 * x128x + q3 * x128y + q4 * x128z, Gy)));
        Gz = fmaf(wh, x128z, fmaf(wA2, vz, fmaf(-wk2, q2 * x128x + q4 * x128y + qzz * x128z, Gz)));
        dx128x = fmaf(wh, nx, fmaf(-wk2, vx, dx128x)); dx128y = fmaf(wh, ny, fmaf(-wk2, vy, dx128y)); dx128z = fmaf(wh, nz, fmaf(-wk2, vz, dx128z));
        }
        {
        // path 8: (0, 3, 3)
        const int weight_index = 1024 + channel;
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

        const float w = radial_weight;
        const float q0 = stage_row[3328 + channel];
        const float q1 = stage_row[3584 + channel];
        const float q2 = stage_row[3840 + channel];
        const float q3 = stage_row[4096 + channel];
        const float q4 = stage_row[4352 + channel];
        const float q5 = stage_row[4608 + channel];
        const float q6 = stage_row[4864 + channel];
        const float q7 = -q0 - q3, q8 = -q1 - q5;
        const float Qxx = q0 * nx + q1 * ny + q2 * nz, Qxy = q1 * nx + q3 * ny + q4 * nz, Qxz = q2 * nx + q4 * ny + q7 * nz;
        const float Qyy = q3 * nx + q5 * ny + q6 * nz, Qyz = q4 * nx + q6 * ny + q8 * nz, Qzz = -Qxx - Qyy;
        const float vx = Qxx * nx + Qxy * ny + Qxz * nz, vy = Qxy * nx + Qyy * ny + Qyz * nz, vz = Qxz * nx + Qyz * ny + Qzz * nz;
        const float h = nx * vx + ny * vy + nz * vz;
        const float wx = 3.0f * w * x0;
        Gx = fmaf(wx, vx, Gx); Gy = fmaf(wx, vy, Gy); Gz = fmaf(wx, vz, Gz);
        dx0 = fmaf(w, h, dx0);
        R = fmaf(radial_dt, x0 * h, R);
        }
        {
        // path 9: (1, 2, 3)
        const int weight_index = 1152 + channel;
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

        const float w = radial_weight;
        const float q0 = stage_row[3456 + channel];
        const float q1 = stage_row[3712 + channel];
        const float q2 = stage_row[3968 + channel];
        const float q3 = stage_row[4224 + channel];
        const float q4 = stage_row[4480 + channel];
        const float q5 = stage_row[4736 + channel];
        const float q6 = stage_row[4992 + channel];
        const float q7 = -q0 - q3, q8 = -q1 - q5;
        const float Qxx = q0 * nx + q1 * ny + q2 * nz, Qxy = q1 * nx + q3 * ny + q4 * nz, Qxz = q2 * nx + q4 * ny + q7 * nz;
        const float Qyy = q3 * nx + q5 * ny + q6 * nz, Qyz = q4 * nx + q6 * ny + q8 * nz, Qzz = -Qxx - Qyy;
        const float vx = Qxx * nx + Qxy * ny + Qxz * nz, vy = Qxy * nx + Qyy * ny + Qyz * nz, vz = Qxz * nx + Qyz * ny + Qzz * nz;
        const float h = nx * vx + ny * vy + nz * vz;
        R = fmaf(radial_dt, 3.f * (x128x * vx + x128y * vy + x128z * vz), R);
        const float w3 = 3.f * w, w6 = 6.f * w;
        dx128x = fmaf(w3, vx, dx128x); dx128y = fmaf(w3, vy, dx128y); dx128z = fmaf(w3, vz, dx128z);
        Gx = fmaf(w6, Qxx * x128x + Qxy * x128y + Qxz * x128z, Gx);
        Gy = fmaf(w6, Qxy * x128x + Qyy * x128y + Qyz * x128z, Gy);
        Gz = fmaf(w6, Qxz * x128x + Qyz * x128y + Qzz * x128z, Gz);
        }
        atomicAdd(dx_lane + 0, dx0);
        atomicAdd(dx_lane + 128, dx128x);
        atomicAdd(dx_lane + 256, dx128y);
        atomicAdd(dx_lane + 384, dx128z);
        dx_lane += LOGICAL_GROUP_SIZE;
    }
    #pragma unroll
    for (int o = 16; o; o >>= 1) {
        Gx += __shfl_xor_sync(FULL_MASK, Gx, o);
        Gy += __shfl_xor_sync(FULL_MASK, Gy, o);
        Gz += __shfl_xor_sync(FULL_MASK, Gz, o);
        R += __shfl_xor_sync(FULL_MASK, R, o);
    }
    if (lane == 0) {
        const float ri = 1.f / r;
        const float Rs = sloped ? R * sc : 0.f;
        const float nG = nx * Gx + ny * Gy + nz * Gz;
        dsh[edge_base + 0] = nx * Rs + (Gx - nx * nG) * ri;
        dsh[edge_base + 1] = ny * Rs + (Gy - ny * nG) * ri;
        dsh[edge_base + 2] = nz * Rs + (Gz - nz * nG) * ri;
        dr[e] = 0.f;
    }
    }  // STAGE_DOUT: the run's edges
    __syncthreads();  // every warp is done with the row before the next copy
    stage_s = stage_t1;
    }  // STAGE_DOUT: the runs
}