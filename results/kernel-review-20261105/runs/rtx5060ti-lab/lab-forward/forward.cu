// fused_folded_forward (FastMACE opt-8): folded last layer + receiver-streaming forward, w stored.
using int64_t = long long;
constexpr bool STORE_W = false;
constexpr int K = 65;  // the folded inputs the k-loop reads
constexpr int KA = 65;  // W's row stride (and a's, without VEC_A)
constexpr int AS = 65;  // a's row stride
constexpr int KP = 65;  // the k-loop's padded length (tensor cores: steps of 8)
constexpr int RS = 36;
constexpr int CB = 32;
constexpr int WS = 33;
constexpr int ED = 16;
constexpr int IN_DIM = 512;
constexpr int OUT_DIM = 5120;
constexpr int64_t Q = 1280;
constexpr int B = 8;
constexpr int RC = 2;
constexpr int TAIL = 64;
constexpr unsigned FULL = 0xffffffffu;
constexpr int A_BASE = 10728;  // the warps' staging: a_s of warp v at A_BASE + v * A_STRIDE
constexpr int A_STRIDE = 648;

extern "C" __global__ void __launch_bounds__(384, 1) fused_folded_forward(
    int64_t N, int64_t E, const float* __restrict__ a, const float* __restrict__ W,
    const float* __restrict__ x, const float* __restrict__ sh, const int* __restrict__ senders,
    const int* __restrict__ receivers, const int* __restrict__ row_ptr, int* __restrict__ counters,
    float* __restrict__ out0, float* __restrict__ out1, float* __restrict__ out2, float* __restrict__ out3, float* __restrict__ wout) {
    extern __shared__ float smem[];
    float* Ws = smem;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float* a_s = Ws + A_BASE + warp * A_STRIDE;
    float* sh_s = a_s + KP * B;
    float* r_s = sh_s + B * ED;  // tensor cores: the warp's redistribution tile [paths][B][RS]
    const int cb = blockIdx.y;
    const int ch = cb * CB + lane;
    int* ctr = counters + blockIdx.y * gridDim.z + blockIdx.z;
    // Receiver chunks first, then chunks of the padding edges past the last row.
    const int64_t receiver_chunks = (N + RC - 1) / RC;
    int64_t tail_begin = row_ptr[N] < 0 ? 0 : row_ptr[N];
    tail_begin = tail_begin > E ? E : tail_begin;
    switch (blockIdx.z) {
    case 0: {
        
        {
            const int wst[5] = {0, 256, 512, 896, 1024};
            for (int i = threadIdx.x; i < 5 * K * CB; i += blockDim.x) {
                const int p = i / (K * CB), rem = i - p * (K * CB), k = rem / CB, l = rem - k * CB;
                Ws[(p * K + k) * WS + l] = W[(int64_t)(wst[p] + cb * CB + l) * KA + k];
            }
        }
        __syncthreads();
        while (true) {
            int cc = 0;
            if (lane == 0) cc = atomicAdd(ctr, 1);
            cc = __shfl_sync(FULL, cc, 0);
            int64_t r0, r1, ebeg, eend;
            const bool tail = cc >= receiver_chunks;
            if (!tail) {
                r0 = (int64_t)cc * RC;
                r1 = r0 + RC < N ? r0 + RC : N;
                ebeg = row_ptr[r0] < 0 ? 0 : row_ptr[r0];
                eend = row_ptr[r1] < ebeg ? ebeg : row_ptr[r1];
                eend = eend > E ? E : eend;
            } else {
                r0 = r1 = N;
                ebeg = tail_begin + (cc - receiver_chunks) * TAIL;
                if (ebeg >= E || !STORE_W) break;
                eend = ebeg + TAIL < E ? ebeg + TAIL : E;
            }
            float acc[19];
            #pragma unroll
            for (int i = 0; i < 19; ++i) acc[i] = 0.f;
            int64_t cur = r0;
            for (int64_t e0 = ebeg; e0 < eend; e0 += B) {
                const int nb = (int)((eend - e0) < B ? (eend - e0) : B);
                __syncwarp();
                for (int i = lane; i < B * K; i += 32) {
                    const int b = i / K, k = i - b * K;
                    a_s[k * B + b] = (b < nb) ? a[(e0 + b) * KA + k] : 0.f;
                }
                for (int i = lane; i < B * ED / 4; i += 32)
                    reinterpret_cast<float4*>(sh_s)[i] = (4 * i / ED < nb)
                        ? __ldg(reinterpret_cast<const float4*>(sh + e0 * ED) + i) : make_float4(0.f, 0.f, 0.f, 0.f);
                float xv[B][4];
                int rv[B];
                #pragma unroll
                for (int b = 0; b < B; ++b) {
                    rv[b] = (b < nb && !tail) ? receivers[e0 + b] : 0x7fffffff;
                    int snd = (b < nb && !tail) ? senders[e0 + b] : -1;
                    if (snd < 0 || snd >= N) snd = -1;
                    xv[b][0] = (snd >= 0) ? x[(int64_t)snd * IN_DIM + 0 + ch] : 0.f;
                    xv[b][1] = (snd >= 0) ? x[(int64_t)snd * IN_DIM + 128 + ch] : 0.f;
                    xv[b][2] = (snd >= 0) ? x[(int64_t)snd * IN_DIM + 256 + ch] : 0.f;
                    xv[b][3] = (snd >= 0) ? x[(int64_t)snd * IN_DIM + 384 + ch] : 0.f;
                }
                __syncwarp();

                float w[5][B];
                #pragma unroll
                for (int g = 0; g < 5; ++g)
                    #pragma unroll
                    for (int b = 0; b < B; ++b) w[g][b] = 0.f;
                #pragma unroll 5
                for (int k = 0; k < K; ++k) {
                    const float4 av0 = *reinterpret_cast<const float4*>(&a_s[k * B + 0]);
                    const float4 av1 = *reinterpret_cast<const float4*>(&a_s[k * B + 4]);
                    { const float wk = Ws[(0 * K + k) * WS + lane];
                      w[0][0] = fmaf(av0.x, wk, w[0][0]);
                      w[0][1] = fmaf(av0.y, wk, w[0][1]);
                      w[0][2] = fmaf(av0.z, wk, w[0][2]);
                      w[0][3] = fmaf(av0.w, wk, w[0][3]);
                      w[0][4] = fmaf(av1.x, wk, w[0][4]);
                      w[0][5] = fmaf(av1.y, wk, w[0][5]);
                      w[0][6] = fmaf(av1.z, wk, w[0][6]);
                      w[0][7] = fmaf(av1.w, wk, w[0][7]);
                    }
                    { const float wk = Ws[(1 * K + k) * WS + lane];
                      w[1][0] = fmaf(av0.x, wk, w[1][0]);
                      w[1][1] = fmaf(av0.y, wk, w[1][1]);
                      w[1][2] = fmaf(av0.z, wk, w[1][2]);
                      w[1][3] = fmaf(av0.w, wk, w[1][3]);
                      w[1][4] = fmaf(av1.x, wk, w[1][4]);
                      w[1][5] = fmaf(av1.y, wk, w[1][5]);
                      w[1][6] = fmaf(av1.z, wk, w[1][6]);
                      w[1][7] = fmaf(av1.w, wk, w[1][7]);
                    }
                    { const float wk = Ws[(2 * K + k) * WS + lane];
                      w[2][0] = fmaf(av0.x, wk, w[2][0]);
                      w[2][1] = fmaf(av0.y, wk, w[2][1]);
                      w[2][2] = fmaf(av0.z, wk, w[2][2]);
                      w[2][3] = fmaf(av0.w, wk, w[2][3]);
                      w[2][4] = fmaf(av1.x, wk, w[2][4]);
                      w[2][5] = fmaf(av1.y, wk, w[2][5]);
                      w[2][6] = fmaf(av1.z, wk, w[2][6]);
                      w[2][7] = fmaf(av1.w, wk, w[2][7]);
                    }
                    { const float wk = Ws[(3 * K + k) * WS + lane];
                      w[3][0] = fmaf(av0.x, wk, w[3][0]);
                      w[3][1] = fmaf(av0.y, wk, w[3][1]);
                      w[3][2] = fmaf(av0.z, wk, w[3][2]);
                      w[3][3] = fmaf(av0.w, wk, w[3][3]);
                      w[3][4] = fmaf(av1.x, wk, w[3][4]);
                      w[3][5] = fmaf(av1.y, wk, w[3][5]);
                      w[3][6] = fmaf(av1.z, wk, w[3][6]);
                      w[3][7] = fmaf(av1.w, wk, w[3][7]);
                    }
                    { const float wk = Ws[(4 * K + k) * WS + lane];
                      w[4][0] = fmaf(av0.x, wk, w[4][0]);
                      w[4][1] = fmaf(av0.y, wk, w[4][1]);
                      w[4][2] = fmaf(av0.z, wk, w[4][2]);
                      w[4][3] = fmaf(av0.w, wk, w[4][3]);
                      w[4][4] = fmaf(av1.x, wk, w[4][4]);
                      w[4][5] = fmaf(av1.y, wk, w[4][5]);
                      w[4][6] = fmaf(av1.z, wk, w[4][6]);
                      w[4][7] = fmaf(av1.w, wk, w[4][7]);
                    }
                }

                if (STORE_W) {
                    const int wst[5] = {0, 256, 512, 896, 1024};
                    #pragma unroll
                    for (int b = 0; b < B; ++b)
                        if (b < nb) {
                            #pragma unroll
                            for (int g = 0; g < 5; ++g) wout[(e0 + b) * Q + wst[g] + ch] = w[g][b];
                        }
                }
                if (!tail) {
                    #pragma unroll
                    for (int b = 0; b < B; ++b) {
                        if (b >= nb) break;
                        while (cur < rv[b]) {
                out0[cur * 256 + 0 + ch] = acc[0];
                out1[cur * 1152 + 0 + ch] = acc[1];
                out1[cur * 1152 + 384 + ch] = acc[2];
                out1[cur * 1152 + 768 + ch] = acc[3];
                out1[cur * 1152 + 256 + ch] = acc[4];
                out1[cur * 1152 + 640 + ch] = acc[5];
                out1[cur * 1152 + 1024 + ch] = acc[6];
                out2[cur * 1920 + 256 + ch] = acc[7];
                out2[cur * 1920 + 640 + ch] = acc[8];
                out2[cur * 1920 + 1024 + ch] = acc[9];
                out2[cur * 1920 + 1408 + ch] = acc[10];
                out2[cur * 1920 + 1792 + ch] = acc[11];
                out3[cur * 1792 + 0 + ch] = acc[12];
                out3[cur * 1792 + 256 + ch] = acc[13];
                out3[cur * 1792 + 512 + ch] = acc[14];
                out3[cur * 1792 + 768 + ch] = acc[15];
                out3[cur * 1792 + 1024 + ch] = acc[16];
                out3[cur * 1792 + 1280 + ch] = acc[17];
                out3[cur * 1792 + 1536 + ch] = acc[18];
                #pragma unroll
                for (int i = 0; i < 19; ++i) acc[i] = 0.f;
                ++cur;
            }
                        {
                        const float e0 = sh_s[b * ED + 0];
                        acc[0] = fmaf(w[0][b], xv[b][0] * e0 * 0x1.0000000000000p+0f, acc[0]);
                        }
                        {
                        const float e0 = sh_s[b * ED + 1];
                        const float e1 = sh_s[b * ED + 2];
                        const float e2 = sh_s[b * ED + 3];
                        acc[1] = fmaf(w[1][b], xv[b][0] * e0 * 0x1.fffffffffffffp-1f, acc[1]);
                        acc[2] = fmaf(w[1][b], xv[b][0] * e1 * 0x1.0000000000000p+0f, acc[2]);
                        acc[3] = fmaf(w[1][b], xv[b][0] * e2 * 0x1.fffffffffffffp-1f, acc[3]);
                        }
                        {
                        const float e0 = sh_s[b * ED + 4];
                        const float e1 = sh_s[b * ED + 5];
                        const float e2 = sh_s[b * ED + 6];
                        const float e3 = sh_s[b * ED + 7];
                        const float e4 = sh_s[b * ED + 8];
                        acc[4] = fmaf(w[2][b], xv[b][1] * e2 * -0x1.43d136248490fp-2f + xv[b][1] * e4 * -0x1.186f174f88471p-1f + xv[b][2] * e1 * 0x1.186f174f88471p-1f + xv[b][3] * e0 * 0x1.186f174f88471p-1f, acc[4]);
                        acc[5] = fmaf(w[2][b], xv[b][1] * e1 * 0x1.186f174f88471p-1f + xv[b][2] * e2 * 0x1.43d1362484910p-1f + xv[b][3] * e3 * 0x1.186f174f88471p-1f, acc[5]);
                        acc[6] = fmaf(w[2][b], xv[b][1] * e0 * 0x1.186f174f88471p-1f + xv[b][2] * e3 * 0x1.186f174f88471p-1f + xv[b][3] * e2 * -0x1.43d136248490fp-2f + xv[b][3] * e4 * 0x1.186f174f88471p-1f, acc[6]);
                        }
                        {
                        const float e0 = sh_s[b * ED + 9];
                        const float e1 = sh_s[b * ED + 10];
                        const float e2 = sh_s[b * ED + 11];
                        const float e3 = sh_s[b * ED + 12];
                        const float e4 = sh_s[b * ED + 13];
                        const float e5 = sh_s[b * ED + 14];
                        const float e6 = sh_s[b * ED + 15];
                        acc[7] = fmaf(w[3][b], xv[b][1] * e4 * -0x1.3c03650e00e03p-3f + xv[b][1] * e6 * -0x1.31fa808c55b43p-1f + xv[b][2] * e1 * 0x1.f3a92ca2f4b7dp-2f + xv[b][3] * e0 * 0x1.31fa808c55b43p-1f + xv[b][3] * e2 * -0x1.3c03650e00e03p-3f, acc[7]);
                        acc[8] = fmaf(w[3][b], xv[b][1] * e3 * -0x1.83091e6a7f7e6p-2f + xv[b][1] * e5 * -0x1.f3a92ca2f4b7bp-2f + xv[b][2] * e2 * 0x1.3c03650e00e03p-1f + xv[b][3] * e1 * 0x1.f3a92ca2f4b7bp-2f, acc[8]);
                        acc[9] = fmaf(w[3][b], xv[b][1] * e2 * 0x1.11acee560242ap-1f + xv[b][2] * e3 * 0x1.4f2ec413cb52bp-1f + xv[b][3] * e4 * 0x1.11acee560242ap-1f, acc[9]);
                        acc[10] = fmaf(w[3][b], xv[b][1] * e1 * 0x1.f3a92ca2f4b7bp-2f + xv[b][2] * e4 * 0x1.3c03650e00e03p-1f + xv[b][3] * e3 * -0x1.83091e6a7f7e6p-2f + xv[b][3] * e5 * 0x1.f3a92ca2f4b7bp-2f, acc[10]);
                        acc[11] = fmaf(w[3][b], xv[b][1] * e0 * 0x1.31fa808c55b43p-1f + xv[b][1] * e2 * 0x1.3c03650e00e03p-3f + xv[b][2] * e5 * 0x1.f3a92ca2f4b7dp-2f + xv[b][3] * e4 * -0x1.3c03650e00e03p-3f + xv[b][3] * e6 * 0x1.31fa808c55b43p-1f, acc[11]);
                        }
                        {
                        const float e0 = sh_s[b * ED + 9];
                        const float e1 = sh_s[b * ED + 10];
                        const float e2 = sh_s[b * ED + 11];
                        const float e3 = sh_s[b * ED + 12];
                        const float e4 = sh_s[b * ED + 13];
                        const float e5 = sh_s[b * ED + 14];
                        const float e6 = sh_s[b * ED + 15];
                        acc[12] = fmaf(w[4][b], xv[b][0] * e0 * 0x1.0000000000000p+0f, acc[12]);
                        acc[13] = fmaf(w[4][b], xv[b][0] * e1 * 0x1.0000000000000p+0f, acc[13]);
                        acc[14] = fmaf(w[4][b], xv[b][0] * e2 * 0x1.0000000000000p+0f, acc[14]);
                        acc[15] = fmaf(w[4][b], xv[b][0] * e3 * 0x1.0000000000001p+0f, acc[15]);
                        acc[16] = fmaf(w[4][b], xv[b][0] * e4 * 0x1.0000000000000p+0f, acc[16]);
                        acc[17] = fmaf(w[4][b], xv[b][0] * e5 * 0x1.0000000000000p+0f, acc[17]);
                        acc[18] = fmaf(w[4][b], xv[b][0] * e6 * 0x1.0000000000000p+0f, acc[18]);
                        }
                    }
                }
            }
            while (cur < r1) {
                out0[cur * 256 + 0 + ch] = acc[0];
                out1[cur * 1152 + 0 + ch] = acc[1];
                out1[cur * 1152 + 384 + ch] = acc[2];
                out1[cur * 1152 + 768 + ch] = acc[3];
                out1[cur * 1152 + 256 + ch] = acc[4];
                out1[cur * 1152 + 640 + ch] = acc[5];
                out1[cur * 1152 + 1024 + ch] = acc[6];
                out2[cur * 1920 + 256 + ch] = acc[7];
                out2[cur * 1920 + 640 + ch] = acc[8];
                out2[cur * 1920 + 1024 + ch] = acc[9];
                out2[cur * 1920 + 1408 + ch] = acc[10];
                out2[cur * 1920 + 1792 + ch] = acc[11];
                out3[cur * 1792 + 0 + ch] = acc[12];
                out3[cur * 1792 + 256 + ch] = acc[13];
                out3[cur * 1792 + 512 + ch] = acc[14];
                out3[cur * 1792 + 768 + ch] = acc[15];
                out3[cur * 1792 + 1024 + ch] = acc[16];
                out3[cur * 1792 + 1280 + ch] = acc[17];
                out3[cur * 1792 + 1536 + ch] = acc[18];
                #pragma unroll
                for (int i = 0; i < 19; ++i) acc[i] = 0.f;
                ++cur;
            }
        }
    } break;
    case 1: {
        
        {
            const int wst[5] = {128, 384, 640, 768, 1152};
            for (int i = threadIdx.x; i < 5 * K * CB; i += blockDim.x) {
                const int p = i / (K * CB), rem = i - p * (K * CB), k = rem / CB, l = rem - k * CB;
                Ws[(p * K + k) * WS + l] = W[(int64_t)(wst[p] + cb * CB + l) * KA + k];
            }
        }
        __syncthreads();
        while (true) {
            int cc = 0;
            if (lane == 0) cc = atomicAdd(ctr, 1);
            cc = __shfl_sync(FULL, cc, 0);
            int64_t r0, r1, ebeg, eend;
            const bool tail = cc >= receiver_chunks;
            if (!tail) {
                r0 = (int64_t)cc * RC;
                r1 = r0 + RC < N ? r0 + RC : N;
                ebeg = row_ptr[r0] < 0 ? 0 : row_ptr[r0];
                eend = row_ptr[r1] < ebeg ? ebeg : row_ptr[r1];
                eend = eend > E ? E : eend;
            } else {
                r0 = r1 = N;
                ebeg = tail_begin + (cc - receiver_chunks) * TAIL;
                if (ebeg >= E || !STORE_W) break;
                eend = ebeg + TAIL < E ? ebeg + TAIL : E;
            }
            float acc[21];
            #pragma unroll
            for (int i = 0; i < 21; ++i) acc[i] = 0.f;
            int64_t cur = r0;
            for (int64_t e0 = ebeg; e0 < eend; e0 += B) {
                const int nb = (int)((eend - e0) < B ? (eend - e0) : B);
                __syncwarp();
                for (int i = lane; i < B * K; i += 32) {
                    const int b = i / K, k = i - b * K;
                    a_s[k * B + b] = (b < nb) ? a[(e0 + b) * KA + k] : 0.f;
                }
                for (int i = lane; i < B * ED / 4; i += 32)
                    reinterpret_cast<float4*>(sh_s)[i] = (4 * i / ED < nb)
                        ? __ldg(reinterpret_cast<const float4*>(sh + e0 * ED) + i) : make_float4(0.f, 0.f, 0.f, 0.f);
                float xv[B][4];
                int rv[B];
                #pragma unroll
                for (int b = 0; b < B; ++b) {
                    rv[b] = (b < nb && !tail) ? receivers[e0 + b] : 0x7fffffff;
                    int snd = (b < nb && !tail) ? senders[e0 + b] : -1;
                    if (snd < 0 || snd >= N) snd = -1;
                    xv[b][0] = (snd >= 0) ? x[(int64_t)snd * IN_DIM + 0 + ch] : 0.f;
                    xv[b][1] = (snd >= 0) ? x[(int64_t)snd * IN_DIM + 128 + ch] : 0.f;
                    xv[b][2] = (snd >= 0) ? x[(int64_t)snd * IN_DIM + 256 + ch] : 0.f;
                    xv[b][3] = (snd >= 0) ? x[(int64_t)snd * IN_DIM + 384 + ch] : 0.f;
                }
                __syncwarp();

                float w[5][B];
                #pragma unroll
                for (int g = 0; g < 5; ++g)
                    #pragma unroll
                    for (int b = 0; b < B; ++b) w[g][b] = 0.f;
                #pragma unroll 5
                for (int k = 0; k < K; ++k) {
                    const float4 av0 = *reinterpret_cast<const float4*>(&a_s[k * B + 0]);
                    const float4 av1 = *reinterpret_cast<const float4*>(&a_s[k * B + 4]);
                    { const float wk = Ws[(0 * K + k) * WS + lane];
                      w[0][0] = fmaf(av0.x, wk, w[0][0]);
                      w[0][1] = fmaf(av0.y, wk, w[0][1]);
                      w[0][2] = fmaf(av0.z, wk, w[0][2]);
                      w[0][3] = fmaf(av0.w, wk, w[0][3]);
                      w[0][4] = fmaf(av1.x, wk, w[0][4]);
                      w[0][5] = fmaf(av1.y, wk, w[0][5]);
                      w[0][6] = fmaf(av1.z, wk, w[0][6]);
                      w[0][7] = fmaf(av1.w, wk, w[0][7]);
                    }
                    { const float wk = Ws[(1 * K + k) * WS + lane];
                      w[1][0] = fmaf(av0.x, wk, w[1][0]);
                      w[1][1] = fmaf(av0.y, wk, w[1][1]);
                      w[1][2] = fmaf(av0.z, wk, w[1][2]);
                      w[1][3] = fmaf(av0.w, wk, w[1][3]);
                      w[1][4] = fmaf(av1.x, wk, w[1][4]);
                      w[1][5] = fmaf(av1.y, wk, w[1][5]);
                      w[1][6] = fmaf(av1.z, wk, w[1][6]);
                      w[1][7] = fmaf(av1.w, wk, w[1][7]);
                    }
                    { const float wk = Ws[(2 * K + k) * WS + lane];
                      w[2][0] = fmaf(av0.x, wk, w[2][0]);
                      w[2][1] = fmaf(av0.y, wk, w[2][1]);
                      w[2][2] = fmaf(av0.z, wk, w[2][2]);
                      w[2][3] = fmaf(av0.w, wk, w[2][3]);
                      w[2][4] = fmaf(av1.x, wk, w[2][4]);
                      w[2][5] = fmaf(av1.y, wk, w[2][5]);
                      w[2][6] = fmaf(av1.z, wk, w[2][6]);
                      w[2][7] = fmaf(av1.w, wk, w[2][7]);
                    }
                    { const float wk = Ws[(3 * K + k) * WS + lane];
                      w[3][0] = fmaf(av0.x, wk, w[3][0]);
                      w[3][1] = fmaf(av0.y, wk, w[3][1]);
                      w[3][2] = fmaf(av0.z, wk, w[3][2]);
                      w[3][3] = fmaf(av0.w, wk, w[3][3]);
                      w[3][4] = fmaf(av1.x, wk, w[3][4]);
                      w[3][5] = fmaf(av1.y, wk, w[3][5]);
                      w[3][6] = fmaf(av1.z, wk, w[3][6]);
                      w[3][7] = fmaf(av1.w, wk, w[3][7]);
                    }
                    { const float wk = Ws[(4 * K + k) * WS + lane];
                      w[4][0] = fmaf(av0.x, wk, w[4][0]);
                      w[4][1] = fmaf(av0.y, wk, w[4][1]);
                      w[4][2] = fmaf(av0.z, wk, w[4][2]);
                      w[4][3] = fmaf(av0.w, wk, w[4][3]);
                      w[4][4] = fmaf(av1.x, wk, w[4][4]);
                      w[4][5] = fmaf(av1.y, wk, w[4][5]);
                      w[4][6] = fmaf(av1.z, wk, w[4][6]);
                      w[4][7] = fmaf(av1.w, wk, w[4][7]);
                    }
                }

                if (STORE_W) {
                    const int wst[5] = {128, 384, 640, 768, 1152};
                    #pragma unroll
                    for (int b = 0; b < B; ++b)
                        if (b < nb) {
                            #pragma unroll
                            for (int g = 0; g < 5; ++g) wout[(e0 + b) * Q + wst[g] + ch] = w[g][b];
                        }
                }
                if (!tail) {
                    #pragma unroll
                    for (int b = 0; b < B; ++b) {
                        if (b >= nb) break;
                        while (cur < rv[b]) {
                out0[cur * 256 + 128 + ch] = acc[0];
                out1[cur * 1152 + 128 + ch] = acc[1];
                out1[cur * 1152 + 512 + ch] = acc[2];
                out1[cur * 1152 + 896 + ch] = acc[3];
                out2[cur * 1920 + 0 + ch] = acc[4];
                out2[cur * 1920 + 384 + ch] = acc[5];
                out2[cur * 1920 + 768 + ch] = acc[6];
                out2[cur * 1920 + 1152 + ch] = acc[7];
                out2[cur * 1920 + 1536 + ch] = acc[8];
                out2[cur * 1920 + 128 + ch] = acc[9];
                out2[cur * 1920 + 512 + ch] = acc[10];
                out2[cur * 1920 + 896 + ch] = acc[11];
                out2[cur * 1920 + 1280 + ch] = acc[12];
                out2[cur * 1920 + 1664 + ch] = acc[13];
                out3[cur * 1792 + 128 + ch] = acc[14];
                out3[cur * 1792 + 384 + ch] = acc[15];
                out3[cur * 1792 + 640 + ch] = acc[16];
                out3[cur * 1792 + 896 + ch] = acc[17];
                out3[cur * 1792 + 1152 + ch] = acc[18];
                out3[cur * 1792 + 1408 + ch] = acc[19];
                out3[cur * 1792 + 1664 + ch] = acc[20];
                #pragma unroll
                for (int i = 0; i < 21; ++i) acc[i] = 0.f;
                ++cur;
            }
                        {
                        const float e0 = sh_s[b * ED + 1];
                        const float e1 = sh_s[b * ED + 2];
                        const float e2 = sh_s[b * ED + 3];
                        acc[0] = fmaf(w[0][b], xv[b][1] * e0 * 0x1.279a74590331cp-1f + xv[b][2] * e1 * 0x1.279a74590331dp-1f + xv[b][3] * e2 * 0x1.279a74590331cp-1f, acc[0]);
                        }
                        {
                        const float e0 = sh_s[b * ED + 0];
                        acc[1] = fmaf(w[1][b], xv[b][1] * e0 * 0x1.fffffffffffffp-1f, acc[1]);
                        acc[2] = fmaf(w[1][b], xv[b][2] * e0 * 0x1.0000000000000p+0f, acc[2]);
                        acc[3] = fmaf(w[1][b], xv[b][3] * e0 * 0x1.fffffffffffffp-1f, acc[3]);
                        }
                        {
                        const float e0 = sh_s[b * ED + 4];
                        const float e1 = sh_s[b * ED + 5];
                        const float e2 = sh_s[b * ED + 6];
                        const float e3 = sh_s[b * ED + 7];
                        const float e4 = sh_s[b * ED + 8];
                        acc[4] = fmaf(w[2][b], xv[b][0] * e0 * 0x1.0000000000000p+0f, acc[4]);
                        acc[5] = fmaf(w[2][b], xv[b][0] * e1 * 0x1.0000000000000p+0f, acc[5]);
                        acc[6] = fmaf(w[2][b], xv[b][0] * e2 * 0x1.0000000000001p+0f, acc[6]);
                        acc[7] = fmaf(w[2][b], xv[b][0] * e3 * 0x1.0000000000000p+0f, acc[7]);
                        acc[8] = fmaf(w[2][b], xv[b][0] * e4 * 0x1.0000000000000p+0f, acc[8]);
                        }
                        {
                        const float e0 = sh_s[b * ED + 1];
                        const float e1 = sh_s[b * ED + 2];
                        const float e2 = sh_s[b * ED + 3];
                        acc[9] = fmaf(w[3][b], xv[b][1] * e2 * 0x1.6a09e667f3bccp-1f + xv[b][3] * e0 * 0x1.6a09e667f3bccp-1f, acc[9]);
                        acc[10] = fmaf(w[3][b], xv[b][1] * e1 * 0x1.6a09e667f3bcdp-1f + xv[b][2] * e0 * 0x1.6a09e667f3bcdp-1f, acc[10]);
                        acc[11] = fmaf(w[3][b], xv[b][1] * e0 * -0x1.a20bd700c2c3ep-2f + xv[b][2] * e1 * 0x1.a20bd700c2c3fp-1f + xv[b][3] * e2 * -0x1.a20bd700c2c3ep-2f, acc[11]);
                        acc[12] = fmaf(w[3][b], xv[b][2] * e2 * 0x1.6a09e667f3bcdp-1f + xv[b][3] * e1 * 0x1.6a09e667f3bcdp-1f, acc[12]);
                        acc[13] = fmaf(w[3][b], xv[b][1] * e0 * -0x1.6a09e667f3bccp-1f + xv[b][3] * e2 * 0x1.6a09e667f3bccp-1f, acc[13]);
                        }
                        {
                        const float e0 = sh_s[b * ED + 4];
                        const float e1 = sh_s[b * ED + 5];
                        const float e2 = sh_s[b * ED + 6];
                        const float e3 = sh_s[b * ED + 7];
                        const float e4 = sh_s[b * ED + 8];
                        acc[14] = fmaf(w[4][b], xv[b][1] * e4 * 0x1.6a09e667f3bccp-1f + xv[b][3] * e0 * 0x1.6a09e667f3bccp-1f, acc[14]);
                        acc[15] = fmaf(w[4][b], xv[b][1] * e3 * 0x1.279a74590331cp-1f + xv[b][2] * e0 * 0x1.279a74590331cp-1f + xv[b][3] * e1 * 0x1.279a74590331cp-1f, acc[15]);
                        acc[16] = fmaf(w[4][b], xv[b][1] * e2 * 0x1.43d136248490fp-1f + xv[b][1] * e4 * 0x1.75e9746a0b099p-3f + xv[b][2] * e1 * 0x1.75e9746a0b099p-1f + xv[b][3] * e0 * -0x1.75e9746a0b099p-3f, acc[16]);
                        acc[17] = fmaf(w[4][b], xv[b][1] * e1 * -0x1.c9f25c5bfedd8p-2f + xv[b][2] * e2 * 0x1.8c97ef43f7249p-1f + xv[b][3] * e3 * -0x1.c9f25c5bfedd8p-2f, acc[17]);
                        acc[18] = fmaf(w[4][b], xv[b][1] * e0 * -0x1.75e9746a0b099p-3f + xv[b][2] * e3 * 0x1.75e9746a0b099p-1f + xv[b][3] * e2 * 0x1.43d136248490fp-1f + xv[b][3] * e4 * -0x1.75e9746a0b099p-3f, acc[18]);
                        acc[19] = fmaf(w[4][b], xv[b][1] * e1 * -0x1.279a74590331cp-1f + xv[b][2] * e4 * 0x1.279a74590331cp-1f + xv[b][3] * e3 * 0x1.279a74590331cp-1f, acc[19]);
                        acc[20] = fmaf(w[4][b], xv[b][1] * e0 * -0x1.6a09e667f3bccp-1f + xv[b][3] * e4 * 0x1.6a09e667f3bccp-1f, acc[20]);
                        }
                    }
                }
            }
            while (cur < r1) {
                out0[cur * 256 + 128 + ch] = acc[0];
                out1[cur * 1152 + 128 + ch] = acc[1];
                out1[cur * 1152 + 512 + ch] = acc[2];
                out1[cur * 1152 + 896 + ch] = acc[3];
                out2[cur * 1920 + 0 + ch] = acc[4];
                out2[cur * 1920 + 384 + ch] = acc[5];
                out2[cur * 1920 + 768 + ch] = acc[6];
                out2[cur * 1920 + 1152 + ch] = acc[7];
                out2[cur * 1920 + 1536 + ch] = acc[8];
                out2[cur * 1920 + 128 + ch] = acc[9];
                out2[cur * 1920 + 512 + ch] = acc[10];
                out2[cur * 1920 + 896 + ch] = acc[11];
                out2[cur * 1920 + 1280 + ch] = acc[12];
                out2[cur * 1920 + 1664 + ch] = acc[13];
                out3[cur * 1792 + 128 + ch] = acc[14];
                out3[cur * 1792 + 384 + ch] = acc[15];
                out3[cur * 1792 + 640 + ch] = acc[16];
                out3[cur * 1792 + 896 + ch] = acc[17];
                out3[cur * 1792 + 1152 + ch] = acc[18];
                out3[cur * 1792 + 1408 + ch] = acc[19];
                out3[cur * 1792 + 1664 + ch] = acc[20];
                #pragma unroll
                for (int i = 0; i < 21; ++i) acc[i] = 0.f;
                ++cur;
            }
        }
    } break;
    }
}
