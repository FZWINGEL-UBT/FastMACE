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

        const float q0 = dout0[int64_t(receiver) * 256 + 0 + channel];
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

        const float q0 = dout0[int64_t(receiver) * 256 + 128 + channel];
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

        const float q0 = dout1[int64_t(receiver) * 1152 + 0 + channel];
        const float q1 = dout1[int64_t(receiver) * 1152 + 384 + channel];
        const float q2 = dout1[int64_t(receiver) * 1152 + 768 + channel];
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

        const float q0 = dout1[int64_t(receiver) * 1152 + 128 + channel];
        const float q1 = dout1[int64_t(receiver) * 1152 + 512 + channel];
        const float q2 = dout1[int64_t(receiver) * 1152 + 896 + channel];
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

        const float q0 = dout1[int64_t(receiver) * 1152 + 256 + channel];
        const float q1 = dout1[int64_t(receiver) * 1152 + 640 + channel];
        const float q2 = dout1[int64_t(receiver) * 1152 + 1024 + channel];
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

        const float q0 = dout2[int64_t(receiver) * 1920 + 0 + channel];
        const float q1 = dout2[int64_t(receiver) * 1920 + 384 + channel];
        const float q2 = dout2[int64_t(receiver) * 1920 + 768 + channel];
        const float q3 = dout2[int64_t(receiver) * 1920 + 1152 + channel];
        const float q4 = dout2[int64_t(receiver) * 1920 + 1536 + channel];
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

        const float q0 = dout2[int64_t(receiver) * 1920 + 128 + channel];
        const float q1 = dout2[int64_t(receiver) * 1920 + 512 + channel];
        const float q2 = dout2[int64_t(receiver) * 1920 + 896 + channel];
        const float q3 = dout2[int64_t(receiver) * 1920 + 1280 + channel];
        const float q4 = dout2[int64_t(receiver) * 1920 + 1664 + channel];
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

        const float q0 = dout2[int64_t(receiver) * 1920 + 256 + channel];
        const float q1 = dout2[int64_t(receiver) * 1920 + 640 + channel];
        const float q2 = dout2[int64_t(receiver) * 1920 + 1024 + channel];
        const float q3 = dout2[int64_t(receiver) * 1920 + 1408 + channel];
        const float q4 = dout2[int64_t(receiver) * 1920 + 1792 + channel];
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

        const float q0 = dout3[int64_t(receiver) * 1792 + 0 + channel];
        const float q1 = dout3[int64_t(receiver) * 1792 + 256 + channel];
        const float q2 = dout3[int64_t(receiver) * 1792 + 512 + channel];
        const float q3 = dout3[int64_t(receiver) * 1792 + 768 + channel];
        const float q4 = dout3[int64_t(receiver) * 1792 + 1024 + channel];
        const float q5 = dout3[int64_t(receiver) * 1792 + 1280 + channel];
        const float q6 = dout3[int64_t(receiver) * 1792 + 1536 + channel];
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

        const float q0 = dout3[int64_t(receiver) * 1792 + 128 + channel];
        const float q1 = dout3[int64_t(receiver) * 1792 + 384 + channel];
        const float q2 = dout3[int64_t(receiver) * 1792 + 640 + channel];
        const float q3 = dout3[int64_t(receiver) * 1792 + 896 + channel];
        const float q4 = dout3[int64_t(receiver) * 1792 + 1152 + channel];
        const float q5 = dout3[int64_t(receiver) * 1792 + 1408 + channel];
        const float q6 = dout3[int64_t(receiver) * 1792 + 1664 + channel];
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
}
