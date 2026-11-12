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

            // late-pullback: stage the receiver row converted to N1's Cartesian coefficients q = M g.
            for (int channel = threadIdx.x; channel < CHANNEL_DIM; channel += blockDim.x) {
                {  // path 0: (0, 0, 0)
                const float g0 = __ldg(dout0 + int64_t(stage_r) * 256 + 0 + channel);
                stage_row[0 + channel] = 1.0f * g0;
                }
                {  // path 1: (1, 1, 0)
                const float g0 = __ldg(dout0 + int64_t(stage_r) * 256 + 128 + channel);
                stage_row[128 + channel] = 1.0f * g0;
                }
                {  // path 2: (0, 1, 1)
                const float g0 = __ldg(dout1 + int64_t(stage_r) * 1152 + 0 + channel);
                const float g1 = __ldg(dout1 + int64_t(stage_r) * 1152 + 384 + channel);
                const float g2 = __ldg(dout1 + int64_t(stage_r) * 1152 + 768 + channel);
                stage_row[256 + channel] = 1.73205081f * g0;
                stage_row[640 + channel] = 1.73205081f * g1;
                stage_row[1024 + channel] = 1.73205081f * g2;
                }
                {  // path 3: (1, 0, 1)
                const float g0 = __ldg(dout1 + int64_t(stage_r) * 1152 + 128 + channel);
                const float g1 = __ldg(dout1 + int64_t(stage_r) * 1152 + 512 + channel);
                const float g2 = __ldg(dout1 + int64_t(stage_r) * 1152 + 896 + channel);
                stage_row[384 + channel] = 1.0f * g0;
                stage_row[768 + channel] = 1.0f * g1;
                stage_row[1152 + channel] = 1.0f * g2;
                }
                {  // path 4: (1, 2, 1)
                const float g0 = __ldg(dout1 + int64_t(stage_r) * 1152 + 256 + channel);
                const float g1 = __ldg(dout1 + int64_t(stage_r) * 1152 + 640 + channel);
                const float g2 = __ldg(dout1 + int64_t(stage_r) * 1152 + 1024 + channel);
                stage_row[512 + channel] = 2.12132034f * g0;
                stage_row[896 + channel] = 2.12132034f * g1;
                stage_row[1280 + channel] = 2.12132034f * g2;
                }
                {  // path 5: (0, 2, 2)
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
                {  // path 6: (1, 1, 2)
                const float g0 = __ldg(dout2 + int64_t(stage_r) * 1920 + 128 + channel);
                const float g1 = __ldg(dout2 + int64_t(stage_r) * 1920 + 512 + channel);
                const float g2 = __ldg(dout2 + int64_t(stage_r) * 1920 + 896 + channel);
                const float g3 = __ldg(dout2 + int64_t(stage_r) * 1920 + 1280 + channel);
                const float g4 = __ldg(dout2 + int64_t(stage_r) * 1920 + 1664 + channel);
                stage_row[1536 + channel] = -0.353553391f * g2 + -0.612372436f * g4;
                stage_row[1920 + channel] = 0.612372436f * g1;
                stage_row[2304 + channel] = 0.612372436f * g0;
                stage_row[2688 + channel] = 0.707106781f * g2;
                stage_row[3072 + channel] = 0.612372436f * g3;
                }
                {  // path 7: (1, 3, 2)
                const float g0 = __ldg(dout2 + int64_t(stage_r) * 1920 + 256 + channel);
                const float g1 = __ldg(dout2 + int64_t(stage_r) * 1920 + 640 + channel);
                const float g2 = __ldg(dout2 + int64_t(stage_r) * 1920 + 1024 + channel);
                const float g3 = __ldg(dout2 + int64_t(stage_r) * 1920 + 1408 + channel);
                const float g4 = __ldg(dout2 + int64_t(stage_r) * 1920 + 1792 + channel);
                stage_row[1664 + channel] = -1.44337567f * g2 + -2.5f * g4;
                stage_row[2048 + channel] = 2.5f * g1;
                stage_row[2432 + channel] = 2.5f * g0;
                stage_row[2816 + channel] = 2.88675135f * g2;
                stage_row[3200 + channel] = 2.5f * g3;
                }
                {  // path 8: (0, 3, 3)
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
                {  // path 9: (1, 2, 3)
                const float g0 = __ldg(dout3 + int64_t(stage_r) * 1792 + 128 + channel);
                const float g1 = __ldg(dout3 + int64_t(stage_r) * 1792 + 384 + channel);
                const float g2 = __ldg(dout3 + int64_t(stage_r) * 1792 + 640 + channel);
                const float g3 = __ldg(dout3 + int64_t(stage_r) * 1792 + 896 + channel);
                const float g4 = __ldg(dout3 + int64_t(stage_r) * 1792 + 1152 + channel);
                const float g5 = __ldg(dout3 + int64_t(stage_r) * 1792 + 1408 + channel);
                const float g6 = __ldg(dout3 + int64_t(stage_r) * 1792 + 1664 + channel);
                stage_row[3456 + channel] = -0.456435465f * g0 + -0.353553391f * g2;
                stage_row[3712 + channel] = -0.288675135f * g3 + -0.372677996f * g5;
                stage_row[3968 + channel] = -0.11785113f * g4 + -0.456435465f * g6;
                stage_row[4224 + channel] = 0.471404521f * g2;
                stage_row[4480 + channel] = 0.372677996f * g1;
                stage_row[4736 + channel] = 0.577350269f * g3;
                stage_row[4992 + channel] = 0.471404521f * g4;
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

    // N1 (lab_cartesian): Cartesian harmonic adjoints; dout already converted to q per path.
    const float nx = sh[edge_base + 1] * 0.57735026918962576451f;
    const float ny = sh[edge_base + 2] * 0.57735026918962576451f;
    const float nz = sh[edge_base + 3] * 0.57735026918962576451f;
    const float ri = 1.f / r;
    float fx = 0.f, fy = 0.f, fz = 0.f;
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

        const float q0 = stage_row[0 + channel];
        const float h = q0, hx = 0.f, hy = 0.f, hz = 0.f;
        const float f = x0 * h, gx = x0 * hx, gy = x0 * hy, gz = x0 * hz;
        dx0 = fmaf(radial_weight, h, dx0);
        const float wr = radial_weight * ri;
        const float radial = (sloped ? radial_dt * sc : 0.f) * f - 0.0f * wr * f;
        fx = fmaf(wr, gx, fmaf(radial, nx, fx));
        fy = fmaf(wr, gy, fmaf(radial, ny, fy));
        fz = fmaf(wr, gz, fmaf(radial, nz, fz));
        }
        {
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

        const float q0 = stage_row[128 + channel];
        const float h = q0, hx = 0.f, hy = 0.f, hz = 0.f;
        const float xx = x128x, xy = x128y, xz = x128z;
        const float hhx = 0.f, hhy = 0.f, hhz = 0.f;
        const float A = xx * nx + xy * ny + xz * nz, B = xx * hx + xy * hy + xz * hz;
        const float ik = 1.0f;
        const float f = A * h - ik * B;
        const float gx = xx * h + A * hx - ik * (2.f * nx * B + hhx), gy = xy * h + A * hy - ik * (2.f * ny * B + hhy), gz = xz * h + A * hz - ik * (2.f * nz * B + hhz);
        dx128x = fmaf(radial_weight, nx * h - ik * hx, dx128x);
        dx128y = fmaf(radial_weight, ny * h - ik * hy, dx128y);
        dx128z = fmaf(radial_weight, nz * h - ik * hz, dx128z);
        const float wr = radial_weight * ri;
        const float radial = (sloped ? radial_dt * sc : 0.f) * f - 1.0f * wr * f;
        fx = fmaf(wr, gx, fmaf(radial, nx, fx));
        fy = fmaf(wr, gy, fmaf(radial, ny, fy));
        fz = fmaf(wr, gz, fmaf(radial, nz, fz));
        }
        {
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

        const float q0 = stage_row[256 + channel];
        const float q1 = stage_row[640 + channel];
        const float q2 = stage_row[1024 + channel];
        const float h = q0 * nx + q1 * ny + q2 * nz, hx = q0, hy = q1, hz = q2;
        const float f = x0 * h, gx = x0 * hx, gy = x0 * hy, gz = x0 * hz;
        dx0 = fmaf(radial_weight, h, dx0);
        const float wr = radial_weight * ri;
        const float radial = (sloped ? radial_dt * sc : 0.f) * f - 1.0f * wr * f;
        fx = fmaf(wr, gx, fmaf(radial, nx, fx));
        fy = fmaf(wr, gy, fmaf(radial, ny, fy));
        fz = fmaf(wr, gz, fmaf(radial, nz, fz));
        }
        {
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

        const float q0 = stage_row[384 + channel];
        const float q1 = stage_row[768 + channel];
        const float q2 = stage_row[1152 + channel];
        const float h = q0 * nx + q1 * ny + q2 * nz, hx = q0, hy = q1, hz = q2;
        const float xx = x128x, xy = x128y, xz = x128z;
        const float hhx = 0.f, hhy = 0.f, hhz = 0.f;
        const float f = xx * hx + xy * hy + xz * hz, gx = hhx, gy = hhy, gz = hhz;
        dx128x = fmaf(radial_weight, hx, dx128x);
        dx128y = fmaf(radial_weight, hy, dx128y);
        dx128z = fmaf(radial_weight, hz, dx128z);
        const float wr = radial_weight * ri;
        const float radial = (sloped ? radial_dt * sc : 0.f) * f - 0.0f * wr * f;
        fx = fmaf(wr, gx, fmaf(radial, nx, fx));
        fy = fmaf(wr, gy, fmaf(radial, ny, fy));
        fz = fmaf(wr, gz, fmaf(radial, nz, fz));
        }
        {
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

        const float q0 = stage_row[512 + channel];
        const float q1 = stage_row[896 + channel];
        const float q2 = stage_row[1280 + channel];
        const float h = q0 * nx + q1 * ny + q2 * nz, hx = q0, hy = q1, hz = q2;
        const float xx = x128x, xy = x128y, xz = x128z;
        const float hhx = 0.f, hhy = 0.f, hhz = 0.f;
        const float A = xx * nx + xy * ny + xz * nz, B = xx * hx + xy * hy + xz * hz;
        const float ik = 0.333333333f;
        const float f = A * h - ik * B;
        const float gx = xx * h + A * hx - ik * (2.f * nx * B + hhx), gy = xy * h + A * hy - ik * (2.f * ny * B + hhy), gz = xz * h + A * hz - ik * (2.f * nz * B + hhz);
        dx128x = fmaf(radial_weight, nx * h - ik * hx, dx128x);
        dx128y = fmaf(radial_weight, ny * h - ik * hy, dx128y);
        dx128z = fmaf(radial_weight, nz * h - ik * hz, dx128z);
        const float wr = radial_weight * ri;
        const float radial = (sloped ? radial_dt * sc : 0.f) * f - 2.0f * wr * f;
        fx = fmaf(wr, gx, fmaf(radial, nx, fx));
        fy = fmaf(wr, gy, fmaf(radial, ny, fy));
        fz = fmaf(wr, gz, fmaf(radial, nz, fz));
        }
        {
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

        const float q0 = stage_row[1408 + channel];
        const float q1 = stage_row[1792 + channel];
        const float q2 = stage_row[2176 + channel];
        const float q3 = stage_row[2560 + channel];
        const float q4 = stage_row[2944 + channel];
        const float qzz = -q0 - q3;
        const float vx = q0 * nx + q1 * ny + q2 * nz, vy = q1 * nx + q3 * ny + q4 * nz, vz = q2 * nx + q4 * ny + qzz * nz;
        const float h = nx * vx + ny * vy + nz * vz, hx = 2.f * vx, hy = 2.f * vy, hz = 2.f * vz;
        const float f = x0 * h, gx = x0 * hx, gy = x0 * hy, gz = x0 * hz;
        dx0 = fmaf(radial_weight, h, dx0);
        const float wr = radial_weight * ri;
        const float radial = (sloped ? radial_dt * sc : 0.f) * f - 2.0f * wr * f;
        fx = fmaf(wr, gx, fmaf(radial, nx, fx));
        fy = fmaf(wr, gy, fmaf(radial, ny, fy));
        fz = fmaf(wr, gz, fmaf(radial, nz, fz));
        }
        {
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

        const float q0 = stage_row[1536 + channel];
        const float q1 = stage_row[1920 + channel];
        const float q2 = stage_row[2304 + channel];
        const float q3 = stage_row[2688 + channel];
        const float q4 = stage_row[3072 + channel];
        const float qzz = -q0 - q3;
        const float vx = q0 * nx + q1 * ny + q2 * nz, vy = q1 * nx + q3 * ny + q4 * nz, vz = q2 * nx + q4 * ny + qzz * nz;
        const float h = nx * vx + ny * vy + nz * vz, hx = 2.f * vx, hy = 2.f * vy, hz = 2.f * vz;
        const float xx = x128x, xy = x128y, xz = x128z;
        const float hhx = 2.f * (q0 * xx + q1 * xy + q2 * xz), hhy = 2.f * (q1 * xx + q3 * xy + q4 * xz), hhz = 2.f * (q2 * xx + q4 * xy + qzz * xz);
        const float f = xx * hx + xy * hy + xz * hz, gx = hhx, gy = hhy, gz = hhz;
        dx128x = fmaf(radial_weight, hx, dx128x);
        dx128y = fmaf(radial_weight, hy, dx128y);
        dx128z = fmaf(radial_weight, hz, dx128z);
        const float wr = radial_weight * ri;
        const float radial = (sloped ? radial_dt * sc : 0.f) * f - 1.0f * wr * f;
        fx = fmaf(wr, gx, fmaf(radial, nx, fx));
        fy = fmaf(wr, gy, fmaf(radial, ny, fy));
        fz = fmaf(wr, gz, fmaf(radial, nz, fz));
        }
        {
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

        const float q0 = stage_row[1664 + channel];
        const float q1 = stage_row[2048 + channel];
        const float q2 = stage_row[2432 + channel];
        const float q3 = stage_row[2816 + channel];
        const float q4 = stage_row[3200 + channel];
        const float qzz = -q0 - q3;
        const float vx = q0 * nx + q1 * ny + q2 * nz, vy = q1 * nx + q3 * ny + q4 * nz, vz = q2 * nx + q4 * ny + qzz * nz;
        const float h = nx * vx + ny * vy + nz * vz, hx = 2.f * vx, hy = 2.f * vy, hz = 2.f * vz;
        const float xx = x128x, xy = x128y, xz = x128z;
        const float hhx = 2.f * (q0 * xx + q1 * xy + q2 * xz), hhy = 2.f * (q1 * xx + q3 * xy + q4 * xz), hhz = 2.f * (q2 * xx + q4 * xy + qzz * xz);
        const float A = xx * nx + xy * ny + xz * nz, B = xx * hx + xy * hy + xz * hz;
        const float ik = 0.2f;
        const float f = A * h - ik * B;
        const float gx = xx * h + A * hx - ik * (2.f * nx * B + hhx), gy = xy * h + A * hy - ik * (2.f * ny * B + hhy), gz = xz * h + A * hz - ik * (2.f * nz * B + hhz);
        dx128x = fmaf(radial_weight, nx * h - ik * hx, dx128x);
        dx128y = fmaf(radial_weight, ny * h - ik * hy, dx128y);
        dx128z = fmaf(radial_weight, nz * h - ik * hz, dx128z);
        const float wr = radial_weight * ri;
        const float radial = (sloped ? radial_dt * sc : 0.f) * f - 3.0f * wr * f;
        fx = fmaf(wr, gx, fmaf(radial, nx, fx));
        fy = fmaf(wr, gy, fmaf(radial, ny, fy));
        fz = fmaf(wr, gz, fmaf(radial, nz, fz));
        }
        {
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
        const float h = nx * vx + ny * vy + nz * vz, hx = 3.f * vx, hy = 3.f * vy, hz = 3.f * vz;
        const float f = x0 * h, gx = x0 * hx, gy = x0 * hy, gz = x0 * hz;
        dx0 = fmaf(radial_weight, h, dx0);
        const float wr = radial_weight * ri;
        const float radial = (sloped ? radial_dt * sc : 0.f) * f - 3.0f * wr * f;
        fx = fmaf(wr, gx, fmaf(radial, nx, fx));
        fy = fmaf(wr, gy, fmaf(radial, ny, fy));
        fz = fmaf(wr, gz, fmaf(radial, nz, fz));
        }
        {
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
        const float h = nx * vx + ny * vy + nz * vz, hx = 3.f * vx, hy = 3.f * vy, hz = 3.f * vz;
        const float xx = x128x, xy = x128y, xz = x128z;
        const float hhx = 6.f * (Qxx * xx + Qxy * xy + Qxz * xz), hhy = 6.f * (Qxy * xx + Qyy * xy + Qyz * xz), hhz = 6.f * (Qxz * xx + Qyz * xy + Qzz * xz);
        const float f = xx * hx + xy * hy + xz * hz, gx = hhx, gy = hhy, gz = hhz;
        dx128x = fmaf(radial_weight, hx, dx128x);
        dx128y = fmaf(radial_weight, hy, dx128y);
        dx128z = fmaf(radial_weight, hz, dx128z);
        const float wr = radial_weight * ri;
        const float radial = (sloped ? radial_dt * sc : 0.f) * f - 2.0f * wr * f;
        fx = fmaf(wr, gx, fmaf(radial, nx, fx));
        fy = fmaf(wr, gy, fmaf(radial, ny, fy));
        fz = fmaf(wr, gz, fmaf(radial, nz, fz));
        }
        atomicAdd(dx + input_base + 0 + channel, dx0);
        atomicAdd(dx + input_base + 128 + channel, dx128x);
        atomicAdd(dx + input_base + 256 + channel, dx128y);
        atomicAdd(dx + input_base + 384 + channel, dx128z);
    }
    for (int o = 16; o; o >>= 1) {
        fx += __shfl_down_sync(FULL_MASK, fx, o);
        fy += __shfl_down_sync(FULL_MASK, fy, o);
        fz += __shfl_down_sync(FULL_MASK, fz, o);
    }
    if (lane == 0) { dsh[edge_base] = fx; dsh[edge_base + 1] = fy; dsh[edge_base + 2] = fz; dr[e] = 0.f; }
    }  // STAGE_DOUT: the run's edges
    __syncthreads();  // every warp is done with the row before the next copy
    stage_s = stage_t1;
    }  // STAGE_DOUT: the runs
}