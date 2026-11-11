// spline_forward0 (SPLINE_W + STAGE_EDGES, 8 November 2026): one receiver per block, its edges staged per tile.
using int64_t = long long;
constexpr int C = 128;
constexpr int64_t Q = 1280;
constexpr int IN_DIM = 512;
constexpr int ED = 16;
constexpr int TILE = 32;
constexpr float R_MAX = 0x1.8000000000000p+2f;

extern "C" __global__ void __launch_bounds__(128, 6) spline_forward0(
    int64_t N, int64_t E, const float* __restrict__ x, const float* __restrict__ sh,
    const int* __restrict__ senders, const int* __restrict__ row_ptr, const float* __restrict__ lengths,
    const int* __restrict__ pairs, const uint2* __restrict__ table, int intervals, const int* __restrict__ order, float* __restrict__ out0, float* __restrict__ out1, float* __restrict__ out2, float* __restrict__ out3) {
    __shared__ int s_snd[TILE], s_k[TILE], s_count;
    __shared__ float s_t[TILE];
    __shared__ int64_t s_tq[TILE];
    constexpr int GD = 76;
    __shared__ __align__(16) float s_g[TILE * GD];
    if (blockIdx.x >= N) return;  // the whole block
    const int rcv = order[blockIdx.x], ch = threadIdx.x;
    float acc[40];
    #pragma unroll
    for (int i = 0; i < 40; ++i) acc[i] = 0.f;
    int64_t eb = row_ptr[rcv], ee = row_ptr[rcv + 1];
    eb = eb < 0 ? 0 : eb;
    ee = ee > E ? E : ee;
    for (int64_t base = eb; base < ee; base += TILE) {
        const int count_in = (int)(ee - base < TILE ? ee - base : TILE);
        if (ch >= 32 && ch < 32 + count_in) {  // GHOIST: warp 1 forms the tile's g_oi = sum_j c e_j
            const float* hp = sh + (base + (ch - 32)) * ED;
            float* gp = s_g + (ch - 32) * GD;
            const float h0 = __ldg(hp + 0);
            const float h1 = __ldg(hp + 1);
            const float h2 = __ldg(hp + 2);
            const float h3 = __ldg(hp + 3);
            const float h4 = __ldg(hp + 4);
            const float h5 = __ldg(hp + 5);
            const float h6 = __ldg(hp + 6);
            const float h7 = __ldg(hp + 7);
            const float h8 = __ldg(hp + 8);
            const float h9 = __ldg(hp + 9);
            const float h10 = __ldg(hp + 10);
            const float h11 = __ldg(hp + 11);
            const float h12 = __ldg(hp + 12);
            const float h13 = __ldg(hp + 13);
            const float h14 = __ldg(hp + 14);
            const float h15 = __ldg(hp + 15);
            gp[0] = h0 * 0x1.0000000000000p+0f;
            gp[1] = h1 * 0x1.fffffffffe931p-1f;
            gp[2] = h2 * 0x1.fffffffffe933p-1f;
            gp[3] = h3 * 0x1.fffffffffe931p-1f;
            gp[4] = h1 * 0x1.fffffffffffffp-1f;
            gp[5] = h2 * 0x1.0000000000000p+0f;
            gp[6] = h3 * 0x1.fffffffffffffp-1f;
            gp[7] = h0 * 0x1.fffffffffffffp-1f;
            gp[8] = h0 * 0x1.0000000000000p+0f;
            gp[9] = h0 * 0x1.fffffffffffffp-1f;
            gp[10] = h6 * -0x1.279a745903946p-1f + h8 * -0x1.0000000000555p+0f;
            gp[11] = h5 * 0x1.0000000000555p+0f;
            gp[12] = h4 * 0x1.0000000000555p+0f;
            gp[13] = h5 * 0x1.0000000000555p+0f;
            gp[14] = h6 * 0x1.279a745903947p+0f;
            gp[15] = h7 * 0x1.0000000000555p+0f;
            gp[16] = h4 * 0x1.0000000000555p+0f;
            gp[17] = h7 * 0x1.0000000000555p+0f;
            gp[18] = h6 * -0x1.279a745903946p-1f + h8 * 0x1.0000000000555p+0f;
            gp[19] = h4 * 0x1.0000000000000p+0f;
            gp[20] = h5 * 0x1.0000000000000p+0f;
            gp[21] = h6 * 0x1.0000000000001p+0f;
            gp[22] = h7 * 0x1.0000000000000p+0f;
            gp[23] = h8 * 0x1.0000000000000p+0f;
            gp[24] = h3 * 0x1.fffffffffe97cp-1f;
            gp[25] = h1 * 0x1.fffffffffe97cp-1f;
            gp[26] = h2 * 0x1.fffffffffe97dp-1f;
            gp[27] = h1 * 0x1.fffffffffe97dp-1f;
            gp[28] = h1 * -0x1.279a74590261dp-1f;
            gp[29] = h2 * 0x1.279a74590261dp+0f;
            gp[30] = h3 * -0x1.279a74590261dp-1f;
            gp[31] = h3 * 0x1.fffffffffe97dp-1f;
            gp[32] = h2 * 0x1.fffffffffe97dp-1f;
            gp[33] = h1 * -0x1.fffffffffe97cp-1f;
            gp[34] = h3 * 0x1.fffffffffe97cp-1f;
            gp[35] = h13 * -0x1.43d1362485537p-2f + h15 * -0x1.3988e14092cf4p+0f;
            gp[36] = h10 * 0x1.000000000099dp+0f;
            gp[37] = h9 * 0x1.3988e14092cf4p+0f + h11 * -0x1.43d1362485537p-2f;
            gp[38] = h12 * -0x1.8c97ef43f812bp-1f + h14 * -0x1.000000000099cp+0f;
            gp[39] = h11 * 0x1.43d1362485537p+0f;
            gp[40] = h10 * 0x1.000000000099cp+0f;
            gp[41] = h11 * 0x1.186f174f88efap+0f;
            gp[42] = h12 * 0x1.5775c544fff49p+0f;
            gp[43] = h13 * 0x1.186f174f88efap+0f;
            gp[44] = h10 * 0x1.000000000099cp+0f;
            gp[45] = h13 * 0x1.43d1362485537p+0f;
            gp[46] = h12 * -0x1.8c97ef43f812bp-1f + h14 * 0x1.000000000099cp+0f;
            gp[47] = h9 * 0x1.3988e14092cf4p+0f + h11 * 0x1.43d1362485537p-2f;
            gp[48] = h14 * 0x1.000000000099dp+0f;
            gp[49] = h13 * -0x1.43d1362485537p-2f + h15 * 0x1.3988e14092cf4p+0f;
            gp[50] = h9 * 0x1.0000000000000p+0f;
            gp[51] = h10 * 0x1.0000000000000p+0f;
            gp[52] = h11 * 0x1.0000000000000p+0f;
            gp[53] = h12 * 0x1.0000000000001p+0f;
            gp[54] = h13 * 0x1.0000000000000p+0f;
            gp[55] = h14 * 0x1.0000000000000p+0f;
            gp[56] = h15 * 0x1.0000000000000p+0f;
            gp[57] = h8 * 0x1.3988e14091337p+0f;
            gp[58] = h4 * 0x1.3988e14091337p+0f;
            gp[59] = h7 * 0x1.fffffffffe931p-1f;
            gp[60] = h4 * 0x1.fffffffffe931p-1f;
            gp[61] = h5 * 0x1.fffffffffe931p-1f;
            gp[62] = h6 * 0x1.186f174f877f5p+0f + h8 * 0x1.43d1362483aa3p-2f;
            gp[63] = h5 * 0x1.43d1362483aa3p+0f;
            gp[64] = h4 * -0x1.43d1362483aa3p-2f;
            gp[65] = h5 * -0x1.8c97ef43f609cp-1f;
            gp[66] = h6 * 0x1.5775c544fe318p+0f;
            gp[67] = h7 * -0x1.8c97ef43f609cp-1f;
            gp[68] = h4 * -0x1.43d1362483aa3p-2f;
            gp[69] = h7 * 0x1.43d1362483aa3p+0f;
            gp[70] = h6 * 0x1.186f174f877f5p+0f + h8 * -0x1.43d1362483aa3p-2f;
            gp[71] = h5 * -0x1.fffffffffe931p-1f;
            gp[72] = h8 * 0x1.fffffffffe931p-1f;
            gp[73] = h7 * 0x1.fffffffffe931p-1f;
            gp[74] = h4 * -0x1.3988e14091337p+0f;
            gp[75] = h8 * 0x1.3988e14091337p+0f;
        }
        if (ch < 32) {  // warp 0: the tile's edges at once, the live ones compacted in order
            bool ok = false;
            int snd_k = 0;
            float t_k = 0.f;
            int64_t tq_k = 0;
            if (ch < count_in) {
                const int64_t e = base + ch;
                const float re = lengths[e];
                if (re < R_MAX) {  // w = 0 at and beyond the cutoff (and the padding edges)
                    snd_k = senders[e];
                    if (snd_k >= 0 && snd_k < N) {
                        int iv; float t, sx_inv;
                        if (re < 0x1.0000000000000p-1f) { sx_inv = 0x1.8880000000000p+9f; float u = (re - 0x1.3333333333333p-2f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 156 ? il : 156; t = u - (float)il; iv = 0 + il; }
                        else if (re < 0x1.8000000000000p-1f) { sx_inv = 0x1.c000000000000p+8f; float u = (re - 0x1.0000000000000p-1f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 111 ? il : 111; t = u - (float)il; iv = 157 + il; }
                        else if (re < 0x1.0000000000000p+0f) { sx_inv = 0x1.5c00000000000p+8f; float u = (re - 0x1.8000000000000p-1f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 86 ? il : 86; t = u - (float)il; iv = 269 + il; }
                        else if (re < 0x1.8000000000000p+0f) { sx_inv = 0x1.f000000000000p+7f; float u = (re - 0x1.0000000000000p+0f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 123 ? il : 123; t = u - (float)il; iv = 356 + il; }
                        else if (re < 0x1.0000000000000p+1f) { sx_inv = 0x1.b000000000000p+6f; float u = (re - 0x1.8000000000000p+0f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 53 ? il : 53; t = u - (float)il; iv = 480 + il; }
                        else if (re < 0x1.4000000000000p+1f) { sx_inv = 0x1.f000000000000p+5f; float u = (re - 0x1.0000000000000p+1f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 30 ? il : 30; t = u - (float)il; iv = 534 + il; }
                        else { sx_inv = 0x1.7b6db6db6db6ep+5f; float u = (re - 0x1.4000000000000p+1f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 165 ? il : 165; t = u - (float)il; iv = 565 + il; }
                        t_k = t;
                        tq_k = ((int64_t)pairs[e] * intervals + iv) * Q;
                        ok = true;
                    }
                }
            }
            const unsigned live = __ballot_sync(0xffffffffu, ok);
            if (ok) {
                const int slot = __popc(live & ((1u << ch) - 1u));
                s_snd[slot] = snd_k;
                s_t[slot] = t_k;
                s_tq[slot] = tq_k;
                s_k[slot] = ch;
            }
            if (ch == 0) s_count = __popc(live);
        }
        __syncthreads();
        const int count = s_count;
        for (int k = 0; k < count; ++k) {
            const int snd = s_snd[k];
            const float t = s_t[k];
            const int gk = s_k[k] * GD;
            const uint2* tq = table + s_tq[k] + ch;
            const float s = t - 0.5f, s2 = s * s;
            if (k + 2 < count) {  // PREFETCH_AHEAD
                const uint2* ahead = table + s_tq[k + 2] + ch;
                asm volatile("prefetch.global.L1 [%0];" :: "l"(ahead + 0));
                asm volatile("prefetch.global.L1 [%0];" :: "l"(ahead + 128));
                asm volatile("prefetch.global.L1 [%0];" :: "l"(ahead + 256));
                asm volatile("prefetch.global.L1 [%0];" :: "l"(ahead + 384));
            }
            const float x0 = __ldg(x + (int64_t)snd * IN_DIM + 0 + ch);
            const float x1 = __ldg(x + (int64_t)snd * IN_DIM + 128 + ch);
            const float x2 = __ldg(x + (int64_t)snd * IN_DIM + 256 + ch);
            const float x3 = __ldg(x + (int64_t)snd * IN_DIM + 384 + ch);
            {
                const uint2 c = __ldg(tq + 0);
                const unsigned eb = ((c.y & 63u) << 23) + 813694976u;  // 2^(E + 1)
                const float hb = __uint_as_float(eb | 0x400000u);
                const float a0 = __uint_as_float(((c.x >> 9) & 0x7FFFFEu) | eb) - hb;
                const float a1 = __uint_as_float(((c.y >> 3) & 0x7FFFF8u) | eb) - hb;
                const float a2 = __uint_as_float((__funnelshift_l(c.y, c.x, 13) & 0x7FFF80u) | eb) - hb;
                const float w = fmaf(s2, a2, fmaf(s, a1, a0));
                acc[0] = fmaf(w * x0, s_g[gk + 0], acc[0]);
            }
            {
                const uint2 c = __ldg(tq + 128);
                const unsigned eb = ((c.y & 63u) << 23) + 813694976u;  // 2^(E + 1)
                const float hb = __uint_as_float(eb | 0x400000u);
                const float a0 = __uint_as_float(((c.x >> 9) & 0x7FFFFEu) | eb) - hb;
                const float a1 = __uint_as_float(((c.y >> 3) & 0x7FFFF8u) | eb) - hb;
                const float a2 = __uint_as_float((__funnelshift_l(c.y, c.x, 13) & 0x7FFF80u) | eb) - hb;
                const float w = fmaf(s2, a2, fmaf(s, a1, a0));
                acc[1] = fmaf(w * x1, s_g[gk + 1], acc[1]);
                acc[1] = fmaf(w * x2, s_g[gk + 2], acc[1]);
                acc[1] = fmaf(w * x3, s_g[gk + 3], acc[1]);
            }
            {
                const uint2 c = __ldg(tq + 256);
                const unsigned eb = ((c.y & 63u) << 23) + 813694976u;  // 2^(E + 1)
                const float hb = __uint_as_float(eb | 0x400000u);
                const float a0 = __uint_as_float(((c.x >> 9) & 0x7FFFFEu) | eb) - hb;
                const float a1 = __uint_as_float(((c.y >> 3) & 0x7FFFF8u) | eb) - hb;
                const float a2 = __uint_as_float((__funnelshift_l(c.y, c.x, 13) & 0x7FFF80u) | eb) - hb;
                const float w = fmaf(s2, a2, fmaf(s, a1, a0));
                acc[2] = fmaf(w * x0, s_g[gk + 4], acc[2]);
                acc[3] = fmaf(w * x0, s_g[gk + 5], acc[3]);
                acc[4] = fmaf(w * x0, s_g[gk + 6], acc[4]);
            }
            {
                const uint2 c = __ldg(tq + 384);
                const unsigned eb = ((c.y & 63u) << 23) + 813694976u;  // 2^(E + 1)
                const float hb = __uint_as_float(eb | 0x400000u);
                const float a0 = __uint_as_float(((c.x >> 9) & 0x7FFFFEu) | eb) - hb;
                const float a1 = __uint_as_float(((c.y >> 3) & 0x7FFFF8u) | eb) - hb;
                const float a2 = __uint_as_float((__funnelshift_l(c.y, c.x, 13) & 0x7FFF80u) | eb) - hb;
                const float w = fmaf(s2, a2, fmaf(s, a1, a0));
                acc[5] = fmaf(w * x1, s_g[gk + 7], acc[5]);
                acc[6] = fmaf(w * x2, s_g[gk + 8], acc[6]);
                acc[7] = fmaf(w * x3, s_g[gk + 9], acc[7]);
            }
            {
                const uint2 c = __ldg(tq + 512);
                const unsigned eb = ((c.y & 63u) << 23) + 813694976u;  // 2^(E + 1)
                const float hb = __uint_as_float(eb | 0x400000u);
                const float a0 = __uint_as_float(((c.x >> 9) & 0x7FFFFEu) | eb) - hb;
                const float a1 = __uint_as_float(((c.y >> 3) & 0x7FFFF8u) | eb) - hb;
                const float a2 = __uint_as_float((__funnelshift_l(c.y, c.x, 13) & 0x7FFF80u) | eb) - hb;
                const float w = fmaf(s2, a2, fmaf(s, a1, a0));
                acc[8] = fmaf(w * x1, s_g[gk + 10], acc[8]);
                acc[8] = fmaf(w * x2, s_g[gk + 11], acc[8]);
                acc[8] = fmaf(w * x3, s_g[gk + 12], acc[8]);
                acc[9] = fmaf(w * x1, s_g[gk + 13], acc[9]);
                acc[9] = fmaf(w * x2, s_g[gk + 14], acc[9]);
                acc[9] = fmaf(w * x3, s_g[gk + 15], acc[9]);
                acc[10] = fmaf(w * x1, s_g[gk + 16], acc[10]);
                acc[10] = fmaf(w * x2, s_g[gk + 17], acc[10]);
                acc[10] = fmaf(w * x3, s_g[gk + 18], acc[10]);
            }
            {
                const uint2 c = __ldg(tq + 640);
                const unsigned eb = ((c.y & 63u) << 23) + 813694976u;  // 2^(E + 1)
                const float hb = __uint_as_float(eb | 0x400000u);
                const float a0 = __uint_as_float(((c.x >> 9) & 0x7FFFFEu) | eb) - hb;
                const float a1 = __uint_as_float(((c.y >> 3) & 0x7FFFF8u) | eb) - hb;
                const float a2 = __uint_as_float((__funnelshift_l(c.y, c.x, 13) & 0x7FFF80u) | eb) - hb;
                const float w = fmaf(s2, a2, fmaf(s, a1, a0));
                acc[11] = fmaf(w * x0, s_g[gk + 19], acc[11]);
                acc[12] = fmaf(w * x0, s_g[gk + 20], acc[12]);
                acc[13] = fmaf(w * x0, s_g[gk + 21], acc[13]);
                acc[14] = fmaf(w * x0, s_g[gk + 22], acc[14]);
                acc[15] = fmaf(w * x0, s_g[gk + 23], acc[15]);
            }
            {
                const uint2 c = __ldg(tq + 768);
                const unsigned eb = ((c.y & 63u) << 23) + 813694976u;  // 2^(E + 1)
                const float hb = __uint_as_float(eb | 0x400000u);
                const float a0 = __uint_as_float(((c.x >> 9) & 0x7FFFFEu) | eb) - hb;
                const float a1 = __uint_as_float(((c.y >> 3) & 0x7FFFF8u) | eb) - hb;
                const float a2 = __uint_as_float((__funnelshift_l(c.y, c.x, 13) & 0x7FFF80u) | eb) - hb;
                const float w = fmaf(s2, a2, fmaf(s, a1, a0));
                acc[16] = fmaf(w * x1, s_g[gk + 24], acc[16]);
                acc[16] = fmaf(w * x3, s_g[gk + 25], acc[16]);
                acc[17] = fmaf(w * x1, s_g[gk + 26], acc[17]);
                acc[17] = fmaf(w * x2, s_g[gk + 27], acc[17]);
                acc[18] = fmaf(w * x1, s_g[gk + 28], acc[18]);
                acc[18] = fmaf(w * x2, s_g[gk + 29], acc[18]);
                acc[18] = fmaf(w * x3, s_g[gk + 30], acc[18]);
                acc[19] = fmaf(w * x2, s_g[gk + 31], acc[19]);
                acc[19] = fmaf(w * x3, s_g[gk + 32], acc[19]);
                acc[20] = fmaf(w * x1, s_g[gk + 33], acc[20]);
                acc[20] = fmaf(w * x3, s_g[gk + 34], acc[20]);
            }
            {
                const uint2 c = __ldg(tq + 896);
                const unsigned eb = ((c.y & 63u) << 23) + 813694976u;  // 2^(E + 1)
                const float hb = __uint_as_float(eb | 0x400000u);
                const float a0 = __uint_as_float(((c.x >> 9) & 0x7FFFFEu) | eb) - hb;
                const float a1 = __uint_as_float(((c.y >> 3) & 0x7FFFF8u) | eb) - hb;
                const float a2 = __uint_as_float((__funnelshift_l(c.y, c.x, 13) & 0x7FFF80u) | eb) - hb;
                const float w = fmaf(s2, a2, fmaf(s, a1, a0));
                acc[21] = fmaf(w * x1, s_g[gk + 35], acc[21]);
                acc[21] = fmaf(w * x2, s_g[gk + 36], acc[21]);
                acc[21] = fmaf(w * x3, s_g[gk + 37], acc[21]);
                acc[22] = fmaf(w * x1, s_g[gk + 38], acc[22]);
                acc[22] = fmaf(w * x2, s_g[gk + 39], acc[22]);
                acc[22] = fmaf(w * x3, s_g[gk + 40], acc[22]);
                acc[23] = fmaf(w * x1, s_g[gk + 41], acc[23]);
                acc[23] = fmaf(w * x2, s_g[gk + 42], acc[23]);
                acc[23] = fmaf(w * x3, s_g[gk + 43], acc[23]);
                acc[24] = fmaf(w * x1, s_g[gk + 44], acc[24]);
                acc[24] = fmaf(w * x2, s_g[gk + 45], acc[24]);
                acc[24] = fmaf(w * x3, s_g[gk + 46], acc[24]);
                acc[25] = fmaf(w * x1, s_g[gk + 47], acc[25]);
                acc[25] = fmaf(w * x2, s_g[gk + 48], acc[25]);
                acc[25] = fmaf(w * x3, s_g[gk + 49], acc[25]);
            }
            {
                const uint2 c = __ldg(tq + 1024);
                const unsigned eb = ((c.y & 63u) << 23) + 813694976u;  // 2^(E + 1)
                const float hb = __uint_as_float(eb | 0x400000u);
                const float a0 = __uint_as_float(((c.x >> 9) & 0x7FFFFEu) | eb) - hb;
                const float a1 = __uint_as_float(((c.y >> 3) & 0x7FFFF8u) | eb) - hb;
                const float a2 = __uint_as_float((__funnelshift_l(c.y, c.x, 13) & 0x7FFF80u) | eb) - hb;
                const float w = fmaf(s2, a2, fmaf(s, a1, a0));
                acc[26] = fmaf(w * x0, s_g[gk + 50], acc[26]);
                acc[27] = fmaf(w * x0, s_g[gk + 51], acc[27]);
                acc[28] = fmaf(w * x0, s_g[gk + 52], acc[28]);
                acc[29] = fmaf(w * x0, s_g[gk + 53], acc[29]);
                acc[30] = fmaf(w * x0, s_g[gk + 54], acc[30]);
                acc[31] = fmaf(w * x0, s_g[gk + 55], acc[31]);
                acc[32] = fmaf(w * x0, s_g[gk + 56], acc[32]);
            }
            {
                const uint2 c = __ldg(tq + 1152);
                const unsigned eb = ((c.y & 63u) << 23) + 813694976u;  // 2^(E + 1)
                const float hb = __uint_as_float(eb | 0x400000u);
                const float a0 = __uint_as_float(((c.x >> 9) & 0x7FFFFEu) | eb) - hb;
                const float a1 = __uint_as_float(((c.y >> 3) & 0x7FFFF8u) | eb) - hb;
                const float a2 = __uint_as_float((__funnelshift_l(c.y, c.x, 13) & 0x7FFF80u) | eb) - hb;
                const float w = fmaf(s2, a2, fmaf(s, a1, a0));
                acc[33] = fmaf(w * x1, s_g[gk + 57], acc[33]);
                acc[33] = fmaf(w * x3, s_g[gk + 58], acc[33]);
                acc[34] = fmaf(w * x1, s_g[gk + 59], acc[34]);
                acc[34] = fmaf(w * x2, s_g[gk + 60], acc[34]);
                acc[34] = fmaf(w * x3, s_g[gk + 61], acc[34]);
                acc[35] = fmaf(w * x1, s_g[gk + 62], acc[35]);
                acc[35] = fmaf(w * x2, s_g[gk + 63], acc[35]);
                acc[35] = fmaf(w * x3, s_g[gk + 64], acc[35]);
                acc[36] = fmaf(w * x1, s_g[gk + 65], acc[36]);
                acc[36] = fmaf(w * x2, s_g[gk + 66], acc[36]);
                acc[36] = fmaf(w * x3, s_g[gk + 67], acc[36]);
                acc[37] = fmaf(w * x1, s_g[gk + 68], acc[37]);
                acc[37] = fmaf(w * x2, s_g[gk + 69], acc[37]);
                acc[37] = fmaf(w * x3, s_g[gk + 70], acc[37]);
                acc[38] = fmaf(w * x1, s_g[gk + 71], acc[38]);
                acc[38] = fmaf(w * x2, s_g[gk + 72], acc[38]);
                acc[38] = fmaf(w * x3, s_g[gk + 73], acc[38]);
                acc[39] = fmaf(w * x1, s_g[gk + 74], acc[39]);
                acc[39] = fmaf(w * x3, s_g[gk + 75], acc[39]);
            }
        }
        __syncthreads();
    }
    out0[(int64_t)rcv * 256 + 0 + ch] = acc[0];
    out0[(int64_t)rcv * 256 + 128 + ch] = acc[1];
    out1[(int64_t)rcv * 1152 + 0 + ch] = acc[2];
    out1[(int64_t)rcv * 1152 + 384 + ch] = acc[3];
    out1[(int64_t)rcv * 1152 + 768 + ch] = acc[4];
    out1[(int64_t)rcv * 1152 + 128 + ch] = acc[5];
    out1[(int64_t)rcv * 1152 + 512 + ch] = acc[6];
    out1[(int64_t)rcv * 1152 + 896 + ch] = acc[7];
    out1[(int64_t)rcv * 1152 + 256 + ch] = acc[8];
    out1[(int64_t)rcv * 1152 + 640 + ch] = acc[9];
    out1[(int64_t)rcv * 1152 + 1024 + ch] = acc[10];
    out2[(int64_t)rcv * 1920 + 0 + ch] = acc[11];
    out2[(int64_t)rcv * 1920 + 384 + ch] = acc[12];
    out2[(int64_t)rcv * 1920 + 768 + ch] = acc[13];
    out2[(int64_t)rcv * 1920 + 1152 + ch] = acc[14];
    out2[(int64_t)rcv * 1920 + 1536 + ch] = acc[15];
    out2[(int64_t)rcv * 1920 + 128 + ch] = acc[16];
    out2[(int64_t)rcv * 1920 + 512 + ch] = acc[17];
    out2[(int64_t)rcv * 1920 + 896 + ch] = acc[18];
    out2[(int64_t)rcv * 1920 + 1280 + ch] = acc[19];
    out2[(int64_t)rcv * 1920 + 1664 + ch] = acc[20];
    out2[(int64_t)rcv * 1920 + 256 + ch] = acc[21];
    out2[(int64_t)rcv * 1920 + 640 + ch] = acc[22];
    out2[(int64_t)rcv * 1920 + 1024 + ch] = acc[23];
    out2[(int64_t)rcv * 1920 + 1408 + ch] = acc[24];
    out2[(int64_t)rcv * 1920 + 1792 + ch] = acc[25];
    out3[(int64_t)rcv * 1792 + 0 + ch] = acc[26];
    out3[(int64_t)rcv * 1792 + 256 + ch] = acc[27];
    out3[(int64_t)rcv * 1792 + 512 + ch] = acc[28];
    out3[(int64_t)rcv * 1792 + 768 + ch] = acc[29];
    out3[(int64_t)rcv * 1792 + 1024 + ch] = acc[30];
    out3[(int64_t)rcv * 1792 + 1280 + ch] = acc[31];
    out3[(int64_t)rcv * 1792 + 1536 + ch] = acc[32];
    out3[(int64_t)rcv * 1792 + 128 + ch] = acc[33];
    out3[(int64_t)rcv * 1792 + 384 + ch] = acc[34];
    out3[(int64_t)rcv * 1792 + 640 + ch] = acc[35];
    out3[(int64_t)rcv * 1792 + 896 + ch] = acc[36];
    out3[(int64_t)rcv * 1792 + 1152 + ch] = acc[37];
    out3[(int64_t)rcv * 1792 + 1408 + ch] = acc[38];
    out3[(int64_t)rcv * 1792 + 1664 + ch] = acc[39];
}
