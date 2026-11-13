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
    const int* __restrict__ pairs, const uint2* __restrict__ table, int intervals, float* __restrict__ out0, float* __restrict__ out1, float* __restrict__ out2, float* __restrict__ out3) {
    __shared__ int s_snd[TILE], s_k[TILE], s_count;
    __shared__ float s_t[TILE];
    __shared__ int64_t s_tq[TILE];
    __shared__ __align__(16) float s_sh[TILE * ED];
    const int rcv = blockIdx.x, ch = threadIdx.x;
    if (rcv >= N) return;  // the whole block
    float acc[40];
    #pragma unroll
    for (int i = 0; i < 40; ++i) acc[i] = 0.f;
    int64_t eb = row_ptr[rcv], ee = row_ptr[rcv + 1];
    eb = eb < 0 ? 0 : eb;
    ee = ee > E ? E : ee;
    for (int64_t base = eb; base < ee; base += TILE) {
        const int count_in = (int)(ee - base < TILE ? ee - base : TILE);
        for (int i = ch; i < count_in * ED; i += C) s_sh[i] = __ldg(sh + base * ED + i);
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
            const float* she = s_sh + s_k[k] * ED;
            const uint2* tq = table + s_tq[k] + ch;
            const float s = t - 0.5f, s2 = s * s;
            const float x0 = __ldg(x + (int64_t)snd * IN_DIM + 0 + ch);
            const float x1 = __ldg(x + (int64_t)snd * IN_DIM + 128 + ch);
            const float x2 = __ldg(x + (int64_t)snd * IN_DIM + 256 + ch);
            const float x3 = __ldg(x + (int64_t)snd * IN_DIM + 384 + ch);
            {
                const float e0 = she[0];
                const uint2 c = __ldg(tq + 0);
                const unsigned eb = ((c.y & 63u) << 23) + 822083584u;  // 2^(E + 1)
                const float hb = __uint_as_float(eb | 0x400000u);
                const float a0 = __uint_as_float(((c.x >> 9) & 0x7FFFFEu) | eb) - hb;
                const float a1 = __uint_as_float(((c.y >> 3) & 0x7FFFF8u) | eb) - hb;
                const float a2 = __uint_as_float((__funnelshift_l(c.y, c.x, 13) & 0x7FFF80u) | eb) - hb;
                const float w = fmaf(s2, a2, fmaf(s, a1, a0));
                acc[0] = fmaf(w, x0 * e0 * 0x1.0000000000000p+0f, acc[0]);
            }
            {
                const float e0 = she[1];
                const float e1 = she[2];
                const float e2 = she[3];
                const uint2 c = __ldg(tq + 128);
                const unsigned eb = ((c.y & 63u) << 23) + 822083584u;  // 2^(E + 1)
                const float hb = __uint_as_float(eb | 0x400000u);
                const float a0 = __uint_as_float(((c.x >> 9) & 0x7FFFFEu) | eb) - hb;
                const float a1 = __uint_as_float(((c.y >> 3) & 0x7FFFF8u) | eb) - hb;
                const float a2 = __uint_as_float((__funnelshift_l(c.y, c.x, 13) & 0x7FFF80u) | eb) - hb;
                const float w = fmaf(s2, a2, fmaf(s, a1, a0));
                acc[1] = fmaf(w, x1 * e0 * 0x1.279a74590331cp-1f + x2 * e1 * 0x1.279a74590331dp-1f + x3 * e2 * 0x1.279a74590331cp-1f, acc[1]);
            }
            {
                const float e0 = she[1];
                const float e1 = she[2];
                const float e2 = she[3];
                const uint2 c = __ldg(tq + 256);
                const unsigned eb = ((c.y & 63u) << 23) + 822083584u;  // 2^(E + 1)
                const float hb = __uint_as_float(eb | 0x400000u);
                const float a0 = __uint_as_float(((c.x >> 9) & 0x7FFFFEu) | eb) - hb;
                const float a1 = __uint_as_float(((c.y >> 3) & 0x7FFFF8u) | eb) - hb;
                const float a2 = __uint_as_float((__funnelshift_l(c.y, c.x, 13) & 0x7FFF80u) | eb) - hb;
                const float w = fmaf(s2, a2, fmaf(s, a1, a0));
                acc[2] = fmaf(w, x0 * e0 * 0x1.fffffffffffffp-1f, acc[2]);
                acc[3] = fmaf(w, x0 * e1 * 0x1.0000000000000p+0f, acc[3]);
                acc[4] = fmaf(w, x0 * e2 * 0x1.fffffffffffffp-1f, acc[4]);
            }
            {
                const float e0 = she[0];
                const uint2 c = __ldg(tq + 384);
                const unsigned eb = ((c.y & 63u) << 23) + 822083584u;  // 2^(E + 1)
                const float hb = __uint_as_float(eb | 0x400000u);
                const float a0 = __uint_as_float(((c.x >> 9) & 0x7FFFFEu) | eb) - hb;
                const float a1 = __uint_as_float(((c.y >> 3) & 0x7FFFF8u) | eb) - hb;
                const float a2 = __uint_as_float((__funnelshift_l(c.y, c.x, 13) & 0x7FFF80u) | eb) - hb;
                const float w = fmaf(s2, a2, fmaf(s, a1, a0));
                acc[5] = fmaf(w, x1 * e0 * 0x1.fffffffffffffp-1f, acc[5]);
                acc[6] = fmaf(w, x2 * e0 * 0x1.0000000000000p+0f, acc[6]);
                acc[7] = fmaf(w, x3 * e0 * 0x1.fffffffffffffp-1f, acc[7]);
            }
            {
                const float e0 = she[4];
                const float e1 = she[5];
                const float e2 = she[6];
                const float e3 = she[7];
                const float e4 = she[8];
                const uint2 c = __ldg(tq + 512);
                const unsigned eb = ((c.y & 63u) << 23) + 822083584u;  // 2^(E + 1)
                const float hb = __uint_as_float(eb | 0x400000u);
                const float a0 = __uint_as_float(((c.x >> 9) & 0x7FFFFEu) | eb) - hb;
                const float a1 = __uint_as_float(((c.y >> 3) & 0x7FFFF8u) | eb) - hb;
                const float a2 = __uint_as_float((__funnelshift_l(c.y, c.x, 13) & 0x7FFF80u) | eb) - hb;
                const float w = fmaf(s2, a2, fmaf(s, a1, a0));
                acc[8] = fmaf(w, x1 * e2 * -0x1.43d136248490fp-2f + x1 * e4 * -0x1.186f174f88471p-1f + x2 * e1 * 0x1.186f174f88471p-1f + x3 * e0 * 0x1.186f174f88471p-1f, acc[8]);
                acc[9] = fmaf(w, x1 * e1 * 0x1.186f174f88471p-1f + x2 * e2 * 0x1.43d1362484910p-1f + x3 * e3 * 0x1.186f174f88471p-1f, acc[9]);
                acc[10] = fmaf(w, x1 * e0 * 0x1.186f174f88471p-1f + x2 * e3 * 0x1.186f174f88471p-1f + x3 * e2 * -0x1.43d136248490fp-2f + x3 * e4 * 0x1.186f174f88471p-1f, acc[10]);
            }
            {
                const float e0 = she[4];
                const float e1 = she[5];
                const float e2 = she[6];
                const float e3 = she[7];
                const float e4 = she[8];
                const uint2 c = __ldg(tq + 640);
                const unsigned eb = ((c.y & 63u) << 23) + 822083584u;  // 2^(E + 1)
                const float hb = __uint_as_float(eb | 0x400000u);
                const float a0 = __uint_as_float(((c.x >> 9) & 0x7FFFFEu) | eb) - hb;
                const float a1 = __uint_as_float(((c.y >> 3) & 0x7FFFF8u) | eb) - hb;
                const float a2 = __uint_as_float((__funnelshift_l(c.y, c.x, 13) & 0x7FFF80u) | eb) - hb;
                const float w = fmaf(s2, a2, fmaf(s, a1, a0));
                acc[11] = fmaf(w, x0 * e0 * 0x1.0000000000000p+0f, acc[11]);
                acc[12] = fmaf(w, x0 * e1 * 0x1.0000000000000p+0f, acc[12]);
                acc[13] = fmaf(w, x0 * e2 * 0x1.0000000000001p+0f, acc[13]);
                acc[14] = fmaf(w, x0 * e3 * 0x1.0000000000000p+0f, acc[14]);
                acc[15] = fmaf(w, x0 * e4 * 0x1.0000000000000p+0f, acc[15]);
            }
            {
                const float e0 = she[1];
                const float e1 = she[2];
                const float e2 = she[3];
                const uint2 c = __ldg(tq + 768);
                const unsigned eb = ((c.y & 63u) << 23) + 822083584u;  // 2^(E + 1)
                const float hb = __uint_as_float(eb | 0x400000u);
                const float a0 = __uint_as_float(((c.x >> 9) & 0x7FFFFEu) | eb) - hb;
                const float a1 = __uint_as_float(((c.y >> 3) & 0x7FFFF8u) | eb) - hb;
                const float a2 = __uint_as_float((__funnelshift_l(c.y, c.x, 13) & 0x7FFF80u) | eb) - hb;
                const float w = fmaf(s2, a2, fmaf(s, a1, a0));
                acc[16] = fmaf(w, x1 * e2 * 0x1.6a09e667f3bccp-1f + x3 * e0 * 0x1.6a09e667f3bccp-1f, acc[16]);
                acc[17] = fmaf(w, x1 * e1 * 0x1.6a09e667f3bcdp-1f + x2 * e0 * 0x1.6a09e667f3bcdp-1f, acc[17]);
                acc[18] = fmaf(w, x1 * e0 * -0x1.a20bd700c2c3ep-2f + x2 * e1 * 0x1.a20bd700c2c3fp-1f + x3 * e2 * -0x1.a20bd700c2c3ep-2f, acc[18]);
                acc[19] = fmaf(w, x2 * e2 * 0x1.6a09e667f3bcdp-1f + x3 * e1 * 0x1.6a09e667f3bcdp-1f, acc[19]);
                acc[20] = fmaf(w, x1 * e0 * -0x1.6a09e667f3bccp-1f + x3 * e2 * 0x1.6a09e667f3bccp-1f, acc[20]);
            }
            {
                const float e0 = she[9];
                const float e1 = she[10];
                const float e2 = she[11];
                const float e3 = she[12];
                const float e4 = she[13];
                const float e5 = she[14];
                const float e6 = she[15];
                const uint2 c = __ldg(tq + 896);
                const unsigned eb = ((c.y & 63u) << 23) + 822083584u;  // 2^(E + 1)
                const float hb = __uint_as_float(eb | 0x400000u);
                const float a0 = __uint_as_float(((c.x >> 9) & 0x7FFFFEu) | eb) - hb;
                const float a1 = __uint_as_float(((c.y >> 3) & 0x7FFFF8u) | eb) - hb;
                const float a2 = __uint_as_float((__funnelshift_l(c.y, c.x, 13) & 0x7FFF80u) | eb) - hb;
                const float w = fmaf(s2, a2, fmaf(s, a1, a0));
                acc[21] = fmaf(w, x1 * e4 * -0x1.3c03650e00e03p-3f + x1 * e6 * -0x1.31fa808c55b43p-1f + x2 * e1 * 0x1.f3a92ca2f4b7dp-2f + x3 * e0 * 0x1.31fa808c55b43p-1f + x3 * e2 * -0x1.3c03650e00e03p-3f, acc[21]);
                acc[22] = fmaf(w, x1 * e3 * -0x1.83091e6a7f7e6p-2f + x1 * e5 * -0x1.f3a92ca2f4b7bp-2f + x2 * e2 * 0x1.3c03650e00e03p-1f + x3 * e1 * 0x1.f3a92ca2f4b7bp-2f, acc[22]);
                acc[23] = fmaf(w, x1 * e2 * 0x1.11acee560242ap-1f + x2 * e3 * 0x1.4f2ec413cb52bp-1f + x3 * e4 * 0x1.11acee560242ap-1f, acc[23]);
                acc[24] = fmaf(w, x1 * e1 * 0x1.f3a92ca2f4b7bp-2f + x2 * e4 * 0x1.3c03650e00e03p-1f + x3 * e3 * -0x1.83091e6a7f7e6p-2f + x3 * e5 * 0x1.f3a92ca2f4b7bp-2f, acc[24]);
                acc[25] = fmaf(w, x1 * e0 * 0x1.31fa808c55b43p-1f + x1 * e2 * 0x1.3c03650e00e03p-3f + x2 * e5 * 0x1.f3a92ca2f4b7dp-2f + x3 * e4 * -0x1.3c03650e00e03p-3f + x3 * e6 * 0x1.31fa808c55b43p-1f, acc[25]);
            }
            {
                const float e0 = she[9];
                const float e1 = she[10];
                const float e2 = she[11];
                const float e3 = she[12];
                const float e4 = she[13];
                const float e5 = she[14];
                const float e6 = she[15];
                const uint2 c = __ldg(tq + 1024);
                const unsigned eb = ((c.y & 63u) << 23) + 822083584u;  // 2^(E + 1)
                const float hb = __uint_as_float(eb | 0x400000u);
                const float a0 = __uint_as_float(((c.x >> 9) & 0x7FFFFEu) | eb) - hb;
                const float a1 = __uint_as_float(((c.y >> 3) & 0x7FFFF8u) | eb) - hb;
                const float a2 = __uint_as_float((__funnelshift_l(c.y, c.x, 13) & 0x7FFF80u) | eb) - hb;
                const float w = fmaf(s2, a2, fmaf(s, a1, a0));
                acc[26] = fmaf(w, x0 * e0 * 0x1.0000000000000p+0f, acc[26]);
                acc[27] = fmaf(w, x0 * e1 * 0x1.0000000000000p+0f, acc[27]);
                acc[28] = fmaf(w, x0 * e2 * 0x1.0000000000000p+0f, acc[28]);
                acc[29] = fmaf(w, x0 * e3 * 0x1.0000000000001p+0f, acc[29]);
                acc[30] = fmaf(w, x0 * e4 * 0x1.0000000000000p+0f, acc[30]);
                acc[31] = fmaf(w, x0 * e5 * 0x1.0000000000000p+0f, acc[31]);
                acc[32] = fmaf(w, x0 * e6 * 0x1.0000000000000p+0f, acc[32]);
            }
            {
                const float e0 = she[4];
                const float e1 = she[5];
                const float e2 = she[6];
                const float e3 = she[7];
                const float e4 = she[8];
                const uint2 c = __ldg(tq + 1152);
                const unsigned eb = ((c.y & 63u) << 23) + 822083584u;  // 2^(E + 1)
                const float hb = __uint_as_float(eb | 0x400000u);
                const float a0 = __uint_as_float(((c.x >> 9) & 0x7FFFFEu) | eb) - hb;
                const float a1 = __uint_as_float(((c.y >> 3) & 0x7FFFF8u) | eb) - hb;
                const float a2 = __uint_as_float((__funnelshift_l(c.y, c.x, 13) & 0x7FFF80u) | eb) - hb;
                const float w = fmaf(s2, a2, fmaf(s, a1, a0));
                acc[33] = fmaf(w, x1 * e4 * 0x1.6a09e667f3bccp-1f + x3 * e0 * 0x1.6a09e667f3bccp-1f, acc[33]);
                acc[34] = fmaf(w, x1 * e3 * 0x1.279a74590331cp-1f + x2 * e0 * 0x1.279a74590331cp-1f + x3 * e1 * 0x1.279a74590331cp-1f, acc[34]);
                acc[35] = fmaf(w, x1 * e2 * 0x1.43d136248490fp-1f + x1 * e4 * 0x1.75e9746a0b099p-3f + x2 * e1 * 0x1.75e9746a0b099p-1f + x3 * e0 * -0x1.75e9746a0b099p-3f, acc[35]);
                acc[36] = fmaf(w, x1 * e1 * -0x1.c9f25c5bfedd8p-2f + x2 * e2 * 0x1.8c97ef43f7249p-1f + x3 * e3 * -0x1.c9f25c5bfedd8p-2f, acc[36]);
                acc[37] = fmaf(w, x1 * e0 * -0x1.75e9746a0b099p-3f + x2 * e3 * 0x1.75e9746a0b099p-1f + x3 * e2 * 0x1.43d136248490fp-1f + x3 * e4 * -0x1.75e9746a0b099p-3f, acc[37]);
                acc[38] = fmaf(w, x1 * e1 * -0x1.279a74590331cp-1f + x2 * e4 * 0x1.279a74590331cp-1f + x3 * e3 * 0x1.279a74590331cp-1f, acc[38]);
                acc[39] = fmaf(w, x1 * e0 * -0x1.6a09e667f3bccp-1f + x3 * e4 * 0x1.6a09e667f3bccp-1f, acc[39]);
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
