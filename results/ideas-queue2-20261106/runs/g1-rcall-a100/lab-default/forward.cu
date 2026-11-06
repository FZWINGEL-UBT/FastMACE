// fused_folded_forward (FastMACE opt-8): folded last layer + receiver-streaming forward, w stored.
using int64_t = long long;
constexpr bool STORE_W = false;
constexpr int K = 64;  // the folded inputs the k-loop reads
constexpr int KA = 65;  // W's row stride (and a's, without VEC_A)
constexpr int AS = 68;  // a's row stride
constexpr int KP = 64;  // the k-loop's padded length (tensor cores: steps of 8)
constexpr int RS = 36;
constexpr int CB = 32;
constexpr int WS = 40;
constexpr int ED = 16;
constexpr int IN_DIM = 512;
constexpr int OUT_DIM = 5120;
constexpr int64_t Q = 1280;
constexpr int B = 8;
constexpr int RC = 2;
constexpr int TAIL = 64;
constexpr unsigned FULL = 0xffffffffu;
constexpr int A_BASE = 12288;  // the warps' staging: a_s of warp v at A_BASE + v * A_STRIDE
constexpr int A_STRIDE = 2368;
// 3xTF32: x = hi + lo, both TF32 (TC_SPLIT); a product keeps hi*hi + hi*lo + lo*hi.
__device__ __forceinline__ void split_tf32(float x, unsigned& hi, unsigned& lo) {
    hi = __float_as_uint(x) & 0xffffe000u;
    lo = __float_as_uint(x - __uint_as_float(hi));
}
__device__ __forceinline__ void mma_tf32(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                 "{%8, %9}, {%0, %1, %2, %3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// W_GROUP: a warp pair's named barrier (64 threads) and its vote (bar.red.or: any lane's pred).
__device__ __forceinline__ void group_sync(int barrier) {
    asm volatile("bar.sync %0, 64;" :: "r"(barrier) : "memory");
}
__device__ __forceinline__ int group_any(int pred, int barrier) {
    int any;
    asm volatile("{\n.reg .pred p, q;\nsetp.ne.b32 q, %1, 0;\nbar.red.or.pred p, %2, 64, q;\n"
                 "selp.b32 %0, 1, 0, p;\n}\n" : "=r"(any) : "r"(pred), "r"(barrier) : "memory");
    return any;
}

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
            const int wst[4] = {0, 256, 640, 1024};
            for (int i = threadIdx.x; i < 4 * KP * CB; i += blockDim.x) {
                const int p = i / (KP * CB), rem = i - p * (KP * CB), k = rem / CB, l = rem - k * CB;
                const int fr = l & 15, fk = k & 7;
                Ws[((p * (KP / 8) + (k >> 3)) * 2 + (l >> 4)) * 128 + ((fr & 7) * 4 + (fk & 3)) * 4 + (fr >> 3)
                   + 2 * (fk >> 2)] = k < K ? W[(int64_t)(wst[p] + cb * CB + l) * KA + k] : 0.f;
            }
        }
        __syncthreads();
        {
            const int gr = warp & 1, gbar = 1 + (warp >> 1);  // the warp's place in its pair and the pair's barrier
            bool have = false, done = false, tail = false;
            int64_t r1 = 0, eend = 0, cur = 0, e0 = 0;
            float acc[16];
            #pragma unroll
            for (int i = 0; i < 16; ++i) acc[i] = 0.f;
            while (true) {
                if (!done && (!have || e0 >= eend)) {  // the next chunk with edges; empty ones flushed on the way
                    if (have) while (cur < r1) {
                out0[cur * 256 + 0 + ch] = acc[0];
                out1[cur * 1152 + 0 + ch] = acc[1];
                out1[cur * 1152 + 384 + ch] = acc[2];
                out1[cur * 1152 + 768 + ch] = acc[3];
                out2[cur * 1920 + 0 + ch] = acc[4];
                out2[cur * 1920 + 384 + ch] = acc[5];
                out2[cur * 1920 + 768 + ch] = acc[6];
                out2[cur * 1920 + 1152 + ch] = acc[7];
                out2[cur * 1920 + 1536 + ch] = acc[8];
                out3[cur * 1792 + 0 + ch] = acc[9];
                out3[cur * 1792 + 256 + ch] = acc[10];
                out3[cur * 1792 + 512 + ch] = acc[11];
                out3[cur * 1792 + 768 + ch] = acc[12];
                out3[cur * 1792 + 1024 + ch] = acc[13];
                out3[cur * 1792 + 1280 + ch] = acc[14];
                out3[cur * 1792 + 1536 + ch] = acc[15];
                #pragma unroll
                for (int i = 0; i < 16; ++i) acc[i] = 0.f;
                ++cur;
            }
                    have = false;
                    while (true) {
                        int cc = 0;
                        if (lane == 0) cc = atomicAdd(ctr, 1);
                        cc = __shfl_sync(FULL, cc, 0);
                        int64_t r0, ebeg;
                        tail = cc >= receiver_chunks;
                        if (!tail) {
                            r0 = (int64_t)cc * RC;
                            r1 = r0 + RC < N ? r0 + RC : N;
                            ebeg = row_ptr[r0] < 0 ? 0 : row_ptr[r0];
                            eend = row_ptr[r1] < ebeg ? ebeg : row_ptr[r1];
                            eend = eend > E ? E : eend;
                        } else {
                            r0 = r1 = N;
                            ebeg = tail_begin + (cc - receiver_chunks) * TAIL;
                            if (ebeg >= E || !STORE_W) {
                                done = true;
                                break;
                            }
                            eend = ebeg + TAIL < E ? ebeg + TAIL : E;
                        }
                        cur = r0;
                        if (ebeg < eend) {
                            have = true;
                            e0 = ebeg;
                            break;
                        }
                        while (cur < r1) {
                out0[cur * 256 + 0 + ch] = acc[0];
                out1[cur * 1152 + 0 + ch] = acc[1];
                out1[cur * 1152 + 384 + ch] = acc[2];
                out1[cur * 1152 + 768 + ch] = acc[3];
                out2[cur * 1920 + 0 + ch] = acc[4];
                out2[cur * 1920 + 384 + ch] = acc[5];
                out2[cur * 1920 + 768 + ch] = acc[6];
                out2[cur * 1920 + 1152 + ch] = acc[7];
                out2[cur * 1920 + 1536 + ch] = acc[8];
                out3[cur * 1792 + 0 + ch] = acc[9];
                out3[cur * 1792 + 256 + ch] = acc[10];
                out3[cur * 1792 + 512 + ch] = acc[11];
                out3[cur * 1792 + 768 + ch] = acc[12];
                out3[cur * 1792 + 1024 + ch] = acc[13];
                out3[cur * 1792 + 1280 + ch] = acc[14];
                out3[cur * 1792 + 1536 + ch] = acc[15];
                #pragma unroll
                for (int i = 0; i < 16; ++i) acc[i] = 0.f;
                ++cur;
            }
                    }
                }
                const int nb = done ? 0 : (int)((eend - e0) < B ? (eend - e0) : B);  // done: an empty batch
                __syncwarp();
                for (int i = lane; i < B * (KP / 8); i += 32) {  // A_LD128: edge b, k-step s
                    const int b = i >> 3, s = i & 7;
                    float4 v0 = make_float4(0.f, 0.f, 0.f, 0.f), v1 = v0;
                    if (b < nb) {
                        const float4* ab = reinterpret_cast<const float4*>(a + (e0 + b) * AS) + 2 * s;
                        v0 = __ldg(ab);
                        v1 = __ldg(ab + 1);
                    }
                    // k-step s of the pair's two batches in region s >> 2 (warp 0's a_s or warp 1's), lane 4 b + tq's
                    // float4 at (s & 3) * 32 + ((4 b + tq) ^ (s >> 1)), this batch's k-groups 2 s and 2 s + 1 in its
                    // half 2 (gr ^ (s & 1)): conflict-free STS.64 here, LDS.128 in the k-loop.
                    float* pt = a_s + ((s >> 2) - gr) * A_STRIDE + (s & 3) * 128 + 2 * (gr ^ (s & 1));
                    *reinterpret_cast<float2*>(pt + ((4 * b + 0) ^ (s >> 1)) * 4) = make_float2(v0.x, v1.x);
                    *reinterpret_cast<float2*>(pt + ((4 * b + 1) ^ (s >> 1)) * 4) = make_float2(v0.y, v1.y);
                    *reinterpret_cast<float2*>(pt + ((4 * b + 2) ^ (s >> 1)) * 4) = make_float2(v0.z, v1.z);
                    *reinterpret_cast<float2*>(pt + ((4 * b + 3) ^ (s >> 1)) * 4) = make_float2(v0.w, v1.w);
                }
                for (int i = lane; i < B * ED / 4; i += 32)
                    reinterpret_cast<float4*>(sh_s)[i] = (4 * i / ED < nb)
                        ? __ldg(reinterpret_cast<const float4*>(sh + e0 * ED) + i) : make_float4(0.f, 0.f, 0.f, 0.f);
                __syncwarp();
                if (!group_any(!done, gbar)) break;  // the group's a_s staged and its r_s free; or both warps done
                float wacc[4][2][4];  // [path][the group's batch t][C fragment] of m-tile gr
                #pragma unroll
                for (int g = 0; g < 4; ++g)
                    #pragma unroll
                    for (int t = 0; t < 2; ++t)
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) wacc[g][t][i] = 0.f;
                {
                    const int gid = lane >> 2, tq = lane & 3;
                    #pragma unroll 2
                    for (int k0 = 0; k0 < KP; k0 += 8) {
                        unsigned bh[2][2], bl[2][2];  // a^T fragments of batch t: k = k0 + tq (+4), edge = gid
                        {  // A_LD128: the lane's four a values of k-step s in one float4
                            const int s = k0 >> 3;
                            const float4 av = reinterpret_cast<const float4*>(a_s + ((s >> 2) - gr) * A_STRIDE)[
                                (s & 3) * 32 + (lane ^ (s >> 1))];
                            const bool odd = (s & 1) != 0;  // batch t's half: 2 (t ^ (s & 1))
                            split_tf32(odd ? av.z : av.x, bh[0][0], bl[0][0]);
                            split_tf32(odd ? av.w : av.y, bh[0][1], bl[0][1]);
                            split_tf32(odd ? av.x : av.z, bh[1][0], bl[1][0]);
                            split_tf32(odd ? av.y : av.w, bh[1][1], bl[1][1]);
                        }
                        #pragma unroll
                        for (int g = 0; g < 4; ++g) {  // W fragment: channel gr*16 + gid (+8), k0 + tq (+4)
                            const float4 wv = reinterpret_cast<const float4*>(Ws)[
                                ((g * (KP / 8) + (k0 >> 3)) * 2 + gr) * 32 + lane];
                            unsigned ah[4], al[4];
                            split_tf32(wv.x, ah[0], al[0]);
                            split_tf32(wv.y, ah[1], al[1]);
                            split_tf32(wv.z, ah[2], al[2]);
                            split_tf32(wv.w, ah[3], al[3]);
                            #pragma unroll
                            for (int t = 0; t < 2; ++t) {
                                mma_tf32(wacc[g][t], al, bh[t]);
                                mma_tf32(wacc[g][t], ah, bl[t]);
                                mma_tf32(wacc[g][t], ah, bh[t]);
                            }
                        }
                    }
                    #pragma unroll
                    for (int g = 0; g < 4; ++g)
                        #pragma unroll
                        for (int t = 0; t < 2; ++t) {  // C fragment: channel gr*16 + gid (+8), edge 2 tq (+1) of t
                            float* rp = r_s + (t - gr) * A_STRIDE + (g * B + 2 * tq) * RS + gr * 16 + gid;
                            rp[0] = wacc[g][t][0];
                            rp[RS] = wacc[g][t][1];
                            rp[8] = wacc[g][t][2];
                            rp[RS + 8] = wacc[g][t][3];
                        }
                }
                float xv[B][1];
                int rv[B];
                #pragma unroll
                for (int b = 0; b < B; ++b) {
                    rv[b] = (b < nb && !tail) ? receivers[e0 + b] : 0x7fffffff;
                    int snd = (b < nb && !tail) ? senders[e0 + b] : -1;
                    if (snd < 0 || snd >= N) snd = -1;
                    xv[b][0] = (snd >= 0) ? x[(int64_t)snd * IN_DIM + 0 + ch] : 0.f;
                }
                group_sync(gbar);  // the group's r_s rows written
                float w[4][B];
                #pragma unroll
                for (int g = 0; g < 4; ++g)
                    #pragma unroll
                    for (int b = 0; b < B; ++b) w[g][b] = r_s[(g * B + b) * RS + lane];
                if (STORE_W) {
                    const int wst[4] = {0, 256, 640, 1024};
                    #pragma unroll
                    for (int b = 0; b < B; ++b)
                        if (b < nb) {
                            #pragma unroll
                            for (int g = 0; g < 4; ++g) wout[(e0 + b) * Q + wst[g] + ch] = w[g][b];
                        }
                }
                if (!tail) {
                    if (nb == B && rv[B - 1] <= cur) {  // TP_NOFLUSH: a full batch inside row cur (CSR order), no flush checks
                        #pragma unroll
                        for (int b = 0; b < B; ++b) {
                            {
                            const float sh0 = sh_s[b * ED + 0];
                            const float sh1 = sh_s[b * ED + 1];
                            const float sh2 = sh_s[b * ED + 2];
                            const float sh3 = sh_s[b * ED + 3];
                            const float sh4 = sh_s[b * ED + 4];
                            const float sh5 = sh_s[b * ED + 5];
                            const float sh6 = sh_s[b * ED + 6];
                            const float sh7 = sh_s[b * ED + 7];
                            const float sh8 = sh_s[b * ED + 8];
                            const float sh9 = sh_s[b * ED + 9];
                            const float sh10 = sh_s[b * ED + 10];
                            const float sh11 = sh_s[b * ED + 11];
                            const float sh12 = sh_s[b * ED + 12];
                            const float sh13 = sh_s[b * ED + 13];
                            const float sh14 = sh_s[b * ED + 14];
                            const float sh15 = sh_s[b * ED + 15];
                            const float s0 = w[0][b] * xv[b][0];  // 24 (b): w x0 first
                            const float s1 = w[1][b] * xv[b][0];  // 24 (b): w x0 first
                            const float s2 = w[2][b] * xv[b][0];  // 24 (b): w x0 first
                            const float s3 = w[3][b] * xv[b][0];  // 24 (b): w x0 first
                            acc[0] = fmaf(s0, sh0, acc[0]);
                            acc[1] = fmaf(s1, sh1, acc[1]);
                            acc[2] = fmaf(s1, sh2, acc[2]);
                            acc[3] = fmaf(s1, sh3, acc[3]);
                            acc[4] = fmaf(s2, sh4, acc[4]);
                            acc[5] = fmaf(s2, sh5, acc[5]);
                            acc[6] = fmaf(s2, sh6, acc[6]);
                            acc[7] = fmaf(s2, sh7, acc[7]);
                            acc[8] = fmaf(s2, sh8, acc[8]);
                            acc[9] = fmaf(s3, sh9, acc[9]);
                            acc[10] = fmaf(s3, sh10, acc[10]);
                            acc[11] = fmaf(s3, sh11, acc[11]);
                            acc[12] = fmaf(s3, sh12, acc[12]);
                            acc[13] = fmaf(s3, sh13, acc[13]);
                            acc[14] = fmaf(s3, sh14, acc[14]);
                            acc[15] = fmaf(s3, sh15, acc[15]);
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int b = 0; b < B; ++b) {
                            if (b >= nb) break;
                            while (cur < rv[b]) {
                    out0[cur * 256 + 0 + ch] = acc[0];
                    out1[cur * 1152 + 0 + ch] = acc[1];
                    out1[cur * 1152 + 384 + ch] = acc[2];
                    out1[cur * 1152 + 768 + ch] = acc[3];
                    out2[cur * 1920 + 0 + ch] = acc[4];
                    out2[cur * 1920 + 384 + ch] = acc[5];
                    out2[cur * 1920 + 768 + ch] = acc[6];
                    out2[cur * 1920 + 1152 + ch] = acc[7];
                    out2[cur * 1920 + 1536 + ch] = acc[8];
                    out3[cur * 1792 + 0 + ch] = acc[9];
                    out3[cur * 1792 + 256 + ch] = acc[10];
                    out3[cur * 1792 + 512 + ch] = acc[11];
                    out3[cur * 1792 + 768 + ch] = acc[12];
                    out3[cur * 1792 + 1024 + ch] = acc[13];
                    out3[cur * 1792 + 1280 + ch] = acc[14];
                    out3[cur * 1792 + 1536 + ch] = acc[15];
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) acc[i] = 0.f;
                    ++cur;
                }
                            {
                            const float sh0 = sh_s[b * ED + 0];
                            const float sh1 = sh_s[b * ED + 1];
                            const float sh2 = sh_s[b * ED + 2];
                            const float sh3 = sh_s[b * ED + 3];
                            const float sh4 = sh_s[b * ED + 4];
                            const float sh5 = sh_s[b * ED + 5];
                            const float sh6 = sh_s[b * ED + 6];
                            const float sh7 = sh_s[b * ED + 7];
                            const float sh8 = sh_s[b * ED + 8];
                            const float sh9 = sh_s[b * ED + 9];
                            const float sh10 = sh_s[b * ED + 10];
                            const float sh11 = sh_s[b * ED + 11];
                            const float sh12 = sh_s[b * ED + 12];
                            const float sh13 = sh_s[b * ED + 13];
                            const float sh14 = sh_s[b * ED + 14];
                            const float sh15 = sh_s[b * ED + 15];
                            const float s0 = w[0][b] * xv[b][0];  // 24 (b): w x0 first
                            const float s1 = w[1][b] * xv[b][0];  // 24 (b): w x0 first
                            const float s2 = w[2][b] * xv[b][0];  // 24 (b): w x0 first
                            const float s3 = w[3][b] * xv[b][0];  // 24 (b): w x0 first
                            acc[0] = fmaf(s0, sh0, acc[0]);
                            acc[1] = fmaf(s1, sh1, acc[1]);
                            acc[2] = fmaf(s1, sh2, acc[2]);
                            acc[3] = fmaf(s1, sh3, acc[3]);
                            acc[4] = fmaf(s2, sh4, acc[4]);
                            acc[5] = fmaf(s2, sh5, acc[5]);
                            acc[6] = fmaf(s2, sh6, acc[6]);
                            acc[7] = fmaf(s2, sh7, acc[7]);
                            acc[8] = fmaf(s2, sh8, acc[8]);
                            acc[9] = fmaf(s3, sh9, acc[9]);
                            acc[10] = fmaf(s3, sh10, acc[10]);
                            acc[11] = fmaf(s3, sh11, acc[11]);
                            acc[12] = fmaf(s3, sh12, acc[12]);
                            acc[13] = fmaf(s3, sh13, acc[13]);
                            acc[14] = fmaf(s3, sh14, acc[14]);
                            acc[15] = fmaf(s3, sh15, acc[15]);
                            }
                        }
                    }
                }
                e0 += B;
            }
        }
    } break;
    case 1: {
        {
            const int wst[6] = {128, 384, 512, 768, 896, 1152};
            const float cpf[6] = {0x1.279a740000000p-1f, 0x1.0000000000000p+0f, 0x1.186f180000000p-1f, 0x1.6a09e60000000p-1f, 0x1.f3a92c0000000p-2f, 0x1.279a740000000p-1f};  // idea 24: c_p per path, into W
            for (int i = threadIdx.x; i < 6 * KP * CB; i += blockDim.x) {
                const int p = i / (KP * CB), rem = i - p * (KP * CB), k = rem / CB, l = rem - k * CB;
                const int fr = l & 15, fk = k & 7;
                Ws[((p * (KP / 8) + (k >> 3)) * 2 + (l >> 4)) * 128 + ((fr & 7) * 4 + (fk & 3)) * 4 + (fr >> 3)
                   + 2 * (fk >> 2)] = k < K ? cpf[p] * W[(int64_t)(wst[p] + cb * CB + l) * KA + k] : 0.f;
            }
        }
        __syncthreads();
        {
            const int gr = warp & 1, gbar = 1 + (warp >> 1);  // the warp's place in its pair and the pair's barrier
            bool have = false, done = false, tail = false;
            int64_t r1 = 0, eend = 0, cur = 0, e0 = 0;
            float acc[24];
            #pragma unroll
            for (int i = 0; i < 24; ++i) acc[i] = 0.f;
            while (true) {
                if (!done && (!have || e0 >= eend)) {  // the next chunk with edges; empty ones flushed on the way
                    if (have) while (cur < r1) {
                out0[cur * 256 + 128 + ch] = acc[0];
                out1[cur * 1152 + 128 + ch] = acc[1];
                out1[cur * 1152 + 512 + ch] = acc[2];
                out1[cur * 1152 + 896 + ch] = acc[3];
                out1[cur * 1152 + 256 + ch] = acc[4];
                out1[cur * 1152 + 640 + ch] = acc[5];
                out1[cur * 1152 + 1024 + ch] = acc[6];
                out2[cur * 1920 + 128 + ch] = acc[7];
                out2[cur * 1920 + 512 + ch] = acc[8];
                out2[cur * 1920 + 896 + ch] = acc[9];
                out2[cur * 1920 + 1280 + ch] = acc[10];
                out2[cur * 1920 + 1664 + ch] = acc[11];
                out2[cur * 1920 + 256 + ch] = acc[12];
                out2[cur * 1920 + 640 + ch] = acc[13];
                out2[cur * 1920 + 1024 + ch] = acc[14];
                out2[cur * 1920 + 1408 + ch] = acc[15];
                out2[cur * 1920 + 1792 + ch] = acc[16];
                out3[cur * 1792 + 128 + ch] = acc[17];
                out3[cur * 1792 + 384 + ch] = acc[18];
                out3[cur * 1792 + 640 + ch] = acc[19];
                out3[cur * 1792 + 896 + ch] = acc[20];
                out3[cur * 1792 + 1152 + ch] = acc[21];
                out3[cur * 1792 + 1408 + ch] = acc[22];
                out3[cur * 1792 + 1664 + ch] = acc[23];
                #pragma unroll
                for (int i = 0; i < 24; ++i) acc[i] = 0.f;
                ++cur;
            }
                    have = false;
                    while (true) {
                        int cc = 0;
                        if (lane == 0) cc = atomicAdd(ctr, 1);
                        cc = __shfl_sync(FULL, cc, 0);
                        int64_t r0, ebeg;
                        tail = cc >= receiver_chunks;
                        if (!tail) {
                            r0 = (int64_t)cc * RC;
                            r1 = r0 + RC < N ? r0 + RC : N;
                            ebeg = row_ptr[r0] < 0 ? 0 : row_ptr[r0];
                            eend = row_ptr[r1] < ebeg ? ebeg : row_ptr[r1];
                            eend = eend > E ? E : eend;
                        } else {
                            r0 = r1 = N;
                            ebeg = tail_begin + (cc - receiver_chunks) * TAIL;
                            if (ebeg >= E || !STORE_W) {
                                done = true;
                                break;
                            }
                            eend = ebeg + TAIL < E ? ebeg + TAIL : E;
                        }
                        cur = r0;
                        if (ebeg < eend) {
                            have = true;
                            e0 = ebeg;
                            break;
                        }
                        while (cur < r1) {
                out0[cur * 256 + 128 + ch] = acc[0];
                out1[cur * 1152 + 128 + ch] = acc[1];
                out1[cur * 1152 + 512 + ch] = acc[2];
                out1[cur * 1152 + 896 + ch] = acc[3];
                out1[cur * 1152 + 256 + ch] = acc[4];
                out1[cur * 1152 + 640 + ch] = acc[5];
                out1[cur * 1152 + 1024 + ch] = acc[6];
                out2[cur * 1920 + 128 + ch] = acc[7];
                out2[cur * 1920 + 512 + ch] = acc[8];
                out2[cur * 1920 + 896 + ch] = acc[9];
                out2[cur * 1920 + 1280 + ch] = acc[10];
                out2[cur * 1920 + 1664 + ch] = acc[11];
                out2[cur * 1920 + 256 + ch] = acc[12];
                out2[cur * 1920 + 640 + ch] = acc[13];
                out2[cur * 1920 + 1024 + ch] = acc[14];
                out2[cur * 1920 + 1408 + ch] = acc[15];
                out2[cur * 1920 + 1792 + ch] = acc[16];
                out3[cur * 1792 + 128 + ch] = acc[17];
                out3[cur * 1792 + 384 + ch] = acc[18];
                out3[cur * 1792 + 640 + ch] = acc[19];
                out3[cur * 1792 + 896 + ch] = acc[20];
                out3[cur * 1792 + 1152 + ch] = acc[21];
                out3[cur * 1792 + 1408 + ch] = acc[22];
                out3[cur * 1792 + 1664 + ch] = acc[23];
                #pragma unroll
                for (int i = 0; i < 24; ++i) acc[i] = 0.f;
                ++cur;
            }
                    }
                }
                const int nb = done ? 0 : (int)((eend - e0) < B ? (eend - e0) : B);  // done: an empty batch
                __syncwarp();
                for (int i = lane; i < B * (KP / 8); i += 32) {  // A_LD128: edge b, k-step s
                    const int b = i >> 3, s = i & 7;
                    float4 v0 = make_float4(0.f, 0.f, 0.f, 0.f), v1 = v0;
                    if (b < nb) {
                        const float4* ab = reinterpret_cast<const float4*>(a + (e0 + b) * AS) + 2 * s;
                        v0 = __ldg(ab);
                        v1 = __ldg(ab + 1);
                    }
                    // k-step s of the pair's two batches in region s >> 2 (warp 0's a_s or warp 1's), lane 4 b + tq's
                    // float4 at (s & 3) * 32 + ((4 b + tq) ^ (s >> 1)), this batch's k-groups 2 s and 2 s + 1 in its
                    // half 2 (gr ^ (s & 1)): conflict-free STS.64 here, LDS.128 in the k-loop.
                    float* pt = a_s + ((s >> 2) - gr) * A_STRIDE + (s & 3) * 128 + 2 * (gr ^ (s & 1));
                    *reinterpret_cast<float2*>(pt + ((4 * b + 0) ^ (s >> 1)) * 4) = make_float2(v0.x, v1.x);
                    *reinterpret_cast<float2*>(pt + ((4 * b + 1) ^ (s >> 1)) * 4) = make_float2(v0.y, v1.y);
                    *reinterpret_cast<float2*>(pt + ((4 * b + 2) ^ (s >> 1)) * 4) = make_float2(v0.z, v1.z);
                    *reinterpret_cast<float2*>(pt + ((4 * b + 3) ^ (s >> 1)) * 4) = make_float2(v0.w, v1.w);
                }
                for (int i = lane; i < B * ED / 4; i += 32)
                    reinterpret_cast<float4*>(sh_s)[i] = (4 * i / ED < nb)
                        ? __ldg(reinterpret_cast<const float4*>(sh + e0 * ED) + i) : make_float4(0.f, 0.f, 0.f, 0.f);
                __syncwarp();
                if (!group_any(!done, gbar)) break;  // the group's a_s staged and its r_s free; or both warps done
                float wacc[6][2][4];  // [path][the group's batch t][C fragment] of m-tile gr
                #pragma unroll
                for (int g = 0; g < 6; ++g)
                    #pragma unroll
                    for (int t = 0; t < 2; ++t)
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) wacc[g][t][i] = 0.f;
                {
                    const int gid = lane >> 2, tq = lane & 3;
                    #pragma unroll 2
                    for (int k0 = 0; k0 < KP; k0 += 8) {
                        unsigned bh[2][2], bl[2][2];  // a^T fragments of batch t: k = k0 + tq (+4), edge = gid
                        {  // A_LD128: the lane's four a values of k-step s in one float4
                            const int s = k0 >> 3;
                            const float4 av = reinterpret_cast<const float4*>(a_s + ((s >> 2) - gr) * A_STRIDE)[
                                (s & 3) * 32 + (lane ^ (s >> 1))];
                            const bool odd = (s & 1) != 0;  // batch t's half: 2 (t ^ (s & 1))
                            split_tf32(odd ? av.z : av.x, bh[0][0], bl[0][0]);
                            split_tf32(odd ? av.w : av.y, bh[0][1], bl[0][1]);
                            split_tf32(odd ? av.x : av.z, bh[1][0], bl[1][0]);
                            split_tf32(odd ? av.y : av.w, bh[1][1], bl[1][1]);
                        }
                        #pragma unroll
                        for (int g = 0; g < 6; ++g) {  // W fragment: channel gr*16 + gid (+8), k0 + tq (+4)
                            const float4 wv = reinterpret_cast<const float4*>(Ws)[
                                ((g * (KP / 8) + (k0 >> 3)) * 2 + gr) * 32 + lane];
                            unsigned ah[4], al[4];
                            split_tf32(wv.x, ah[0], al[0]);
                            split_tf32(wv.y, ah[1], al[1]);
                            split_tf32(wv.z, ah[2], al[2]);
                            split_tf32(wv.w, ah[3], al[3]);
                            #pragma unroll
                            for (int t = 0; t < 2; ++t) {
                                mma_tf32(wacc[g][t], al, bh[t]);
                                mma_tf32(wacc[g][t], ah, bl[t]);
                                mma_tf32(wacc[g][t], ah, bh[t]);
                            }
                        }
                    }
                    #pragma unroll
                    for (int g = 0; g < 6; ++g)
                        #pragma unroll
                        for (int t = 0; t < 2; ++t) {  // C fragment: channel gr*16 + gid (+8), edge 2 tq (+1) of t
                            float* rp = r_s + (t - gr) * A_STRIDE + (g * B + 2 * tq) * RS + gr * 16 + gid;
                            rp[0] = wacc[g][t][0];
                            rp[RS] = wacc[g][t][1];
                            rp[8] = wacc[g][t][2];
                            rp[RS + 8] = wacc[g][t][3];
                        }
                }
                float xv[B][3];
                int rv[B];
                #pragma unroll
                for (int b = 0; b < B; ++b) {
                    rv[b] = (b < nb && !tail) ? receivers[e0 + b] : 0x7fffffff;
                    int snd = (b < nb && !tail) ? senders[e0 + b] : -1;
                    if (snd < 0 || snd >= N) snd = -1;
                    xv[b][0] = (snd >= 0) ? x[(int64_t)snd * IN_DIM + 128 + ch] : 0.f;
                    xv[b][1] = (snd >= 0) ? x[(int64_t)snd * IN_DIM + 256 + ch] : 0.f;
                    xv[b][2] = (snd >= 0) ? x[(int64_t)snd * IN_DIM + 384 + ch] : 0.f;
                }
                group_sync(gbar);  // the group's r_s rows written
                float w[6][B];
                #pragma unroll
                for (int g = 0; g < 6; ++g)
                    #pragma unroll
                    for (int b = 0; b < B; ++b) w[g][b] = r_s[(g * B + b) * RS + lane];
                if (STORE_W) {
                    const int wst[6] = {128, 384, 512, 768, 896, 1152};
                    #pragma unroll
                    for (int b = 0; b < B; ++b)
                        if (b < nb) {
                            #pragma unroll
                            for (int g = 0; g < 6; ++g) wout[(e0 + b) * Q + wst[g] + ch] = w[g][b];
                        }
                }
                if (!tail) {
                    if (nb == B && rv[B - 1] <= cur) {  // TP_NOFLUSH: a full batch inside row cur (CSR order), no flush checks
                        #pragma unroll
                        for (int b = 0; b < B; ++b) {
                            {
                            const float sh0 = sh_s[b * ED + 0];
                            const float sh1 = sh_s[b * ED + 1];
                            const float sh2 = sh_s[b * ED + 2];
                            const float sh3 = sh_s[b * ED + 3];
                            const float sh4 = sh_s[b * ED + 4];
                            const float sh5 = sh_s[b * ED + 5];
                            const float sh6 = sh_s[b * ED + 6];
                            const float sh7 = sh_s[b * ED + 7];
                            const float sh8 = sh_s[b * ED + 8];
                            const float sh9 = sh_s[b * ED + 9];
                            const float sh10 = sh_s[b * ED + 10];
                            const float sh11 = sh_s[b * ED + 11];
                            const float sh12 = sh_s[b * ED + 12];
                            const float sh13 = sh_s[b * ED + 13];
                            const float sh14 = sh_s[b * ED + 14];
                            const float sh15 = sh_s[b * ED + 15];
                            const float p0_1 = xv[b][0] * sh1;
                            const float p0_4 = xv[b][0] * sh4;
                            const float p0_5 = xv[b][0] * sh5;
                            const float p0_6 = xv[b][0] * sh6;
                            const float p0_8 = xv[b][0] * sh8;
                            const float p0_9 = xv[b][0] * sh9;
                            const float p0_11 = xv[b][0] * sh11;
                            const float p0_12 = xv[b][0] * sh12;
                            const float p0_13 = xv[b][0] * sh13;
                            const float p0_15 = xv[b][0] * sh15;
                            const float p1_2 = xv[b][1] * sh2;
                            const float p1_5 = xv[b][1] * sh5;
                            const float p1_6 = xv[b][1] * sh6;
                            const float p1_7 = xv[b][1] * sh7;
                            const float p1_11 = xv[b][1] * sh11;
                            const float p1_12 = xv[b][1] * sh12;
                            const float p1_13 = xv[b][1] * sh13;
                            const float p2_3 = xv[b][2] * sh3;
                            const float p2_4 = xv[b][2] * sh4;
                            const float p2_6 = xv[b][2] * sh6;
                            const float p2_7 = xv[b][2] * sh7;
                            const float p2_8 = xv[b][2] * sh8;
                            const float p2_9 = xv[b][2] * sh9;
                            const float p2_11 = xv[b][2] * sh11;
                            const float p2_12 = xv[b][2] * sh12;
                            const float p2_13 = xv[b][2] * sh13;
                            const float p2_15 = xv[b][2] * sh15;
                            float t0 = p0_1;
                            t0 = t0 + p1_2;
                            t0 = t0 + p2_3;
                            acc[0] = fmaf(w[0][b], t0, acc[0]);
                            const float s1 = w[1][b] * sh0;  // 24 (c): w e0 first
                            acc[1] = fmaf(s1, xv[b][0], acc[1]);
                            acc[2] = fmaf(s1, xv[b][1], acc[2]);
                            acc[3] = fmaf(s1, xv[b][2], acc[3]);
                            float t4 = -p0_8;
                            t4 = t4 + p1_5;
                            t4 = t4 + p2_4;
                            t4 = fmaf(p0_6, -0x1.279a740000000p-1f, t4);
                            acc[4] = fmaf(w[2][b], t4, acc[4]);
                            float t5 = p0_5;
                            t5 = t5 + p2_7;
                            t5 = fmaf(p1_6, 0x1.279a740000000p+0f, t5);
                            acc[5] = fmaf(w[2][b], t5, acc[5]);
                            float t6 = p0_4;
                            t6 = t6 + p1_7;
                            t6 = t6 + p2_8;
                            t6 = fmaf(p2_6, -0x1.279a740000000p-1f, t6);
                            acc[6] = fmaf(w[2][b], t6, acc[6]);
                            float t7 = xv[b][0] * sh3;
                            t7 = fmaf(xv[b][2], sh1, t7);
                            acc[7] = fmaf(w[3][b], t7, acc[7]);
                            float t8 = xv[b][0] * sh2;
                            t8 = fmaf(xv[b][1], sh1, t8);
                            acc[8] = fmaf(w[3][b], t8, acc[8]);
                            float t9 = p0_1 * -0x1.279a760000000p-1f;
                            t9 = fmaf(p1_2, 0x1.279a760000000p+0f, t9);
                            t9 = fmaf(p2_3, -0x1.279a760000000p-1f, t9);
                            acc[9] = fmaf(w[3][b], t9, acc[9]);
                            float t10 = xv[b][1] * sh3;
                            t10 = fmaf(xv[b][2], sh2, t10);
                            acc[10] = fmaf(w[3][b], t10, acc[10]);
                            float t11 = -p0_1;
                            t11 = t11 + p2_3;
                            acc[11] = fmaf(w[3][b], t11, acc[11]);
                            float t12 = xv[b][1] * sh10;
                            t12 = fmaf(p0_13, -0x1.43d1380000000p-2f, t12);
                            t12 = fmaf(p0_15, -0x1.3988e20000000p+0f, t12);
                            t12 = fmaf(p2_9, 0x1.3988e20000000p+0f, t12);
                            t12 = fmaf(p2_11, -0x1.43d1380000000p-2f, t12);
                            acc[12] = fmaf(w[4][b], t12, acc[12]);
                            float t13 = -xv[b][0] * sh14;
                            t13 = fmaf(xv[b][2], sh10, t13);
                            t13 = fmaf(p0_12, -0x1.8c97f00000000p-1f, t13);
                            t13 = fmaf(p1_11, 0x1.43d1380000000p+0f, t13);
                            acc[13] = fmaf(w[4][b], t13, acc[13]);
                            float t14 = p0_11 * 0x1.186f180000000p+0f;
                            t14 = fmaf(p1_12, 0x1.5775c60000000p+0f, t14);
                            t14 = fmaf(p2_13, 0x1.186f180000000p+0f, t14);
                            acc[14] = fmaf(w[4][b], t14, acc[14]);
                            float t15 = xv[b][0] * sh10;
                            t15 = fmaf(xv[b][2], sh14, t15);
                            t15 = fmaf(p1_13, 0x1.43d1380000000p+0f, t15);
                            t15 = fmaf(p2_12, -0x1.8c97f00000000p-1f, t15);
                            acc[15] = fmaf(w[4][b], t15, acc[15]);
                            float t16 = xv[b][1] * sh14;
                            t16 = fmaf(p0_9, 0x1.3988e20000000p+0f, t16);
                            t16 = fmaf(p0_11, 0x1.43d1380000000p-2f, t16);
                            t16 = fmaf(p2_13, -0x1.43d1380000000p-2f, t16);
                            t16 = fmaf(p2_15, 0x1.3988e20000000p+0f, t16);
                            acc[16] = fmaf(w[4][b], t16, acc[16]);
                            float t17 = p0_8 * 0x1.3988e20000000p+0f;
                            t17 = fmaf(p2_4, 0x1.3988e20000000p+0f, t17);
                            acc[17] = fmaf(w[5][b], t17, acc[17]);
                            float t18 = xv[b][0] * sh7;
                            t18 = fmaf(xv[b][1], sh4, t18);
                            t18 = fmaf(xv[b][2], sh5, t18);
                            acc[18] = fmaf(w[5][b], t18, acc[18]);
                            float t19 = p0_6 * 0x1.186f180000000p+0f;
                            t19 = fmaf(p0_8, 0x1.43d1360000000p-2f, t19);
                            t19 = fmaf(p1_5, 0x1.43d1360000000p+0f, t19);
                            t19 = fmaf(p2_4, -0x1.43d1360000000p-2f, t19);
                            acc[19] = fmaf(w[5][b], t19, acc[19]);
                            float t20 = p0_5 * -0x1.8c97f00000000p-1f;
                            t20 = fmaf(p1_6, 0x1.5775c60000000p+0f, t20);
                            t20 = fmaf(p2_7, -0x1.8c97f00000000p-1f, t20);
                            acc[20] = fmaf(w[5][b], t20, acc[20]);
                            float t21 = p0_4 * -0x1.43d1360000000p-2f;
                            t21 = fmaf(p1_7, 0x1.43d1360000000p+0f, t21);
                            t21 = fmaf(p2_6, 0x1.186f180000000p+0f, t21);
                            t21 = fmaf(p2_8, -0x1.43d1360000000p-2f, t21);
                            acc[21] = fmaf(w[5][b], t21, acc[21]);
                            float t22 = -p0_5;
                            t22 = t22 + p2_7;
                            t22 = fmaf(xv[b][1], sh8, t22);
                            acc[22] = fmaf(w[5][b], t22, acc[22]);
                            float t23 = p0_4 * -0x1.3988e20000000p+0f;
                            t23 = fmaf(p2_8, 0x1.3988e20000000p+0f, t23);
                            acc[23] = fmaf(w[5][b], t23, acc[23]);
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int b = 0; b < B; ++b) {
                            if (b >= nb) break;
                            while (cur < rv[b]) {
                    out0[cur * 256 + 128 + ch] = acc[0];
                    out1[cur * 1152 + 128 + ch] = acc[1];
                    out1[cur * 1152 + 512 + ch] = acc[2];
                    out1[cur * 1152 + 896 + ch] = acc[3];
                    out1[cur * 1152 + 256 + ch] = acc[4];
                    out1[cur * 1152 + 640 + ch] = acc[5];
                    out1[cur * 1152 + 1024 + ch] = acc[6];
                    out2[cur * 1920 + 128 + ch] = acc[7];
                    out2[cur * 1920 + 512 + ch] = acc[8];
                    out2[cur * 1920 + 896 + ch] = acc[9];
                    out2[cur * 1920 + 1280 + ch] = acc[10];
                    out2[cur * 1920 + 1664 + ch] = acc[11];
                    out2[cur * 1920 + 256 + ch] = acc[12];
                    out2[cur * 1920 + 640 + ch] = acc[13];
                    out2[cur * 1920 + 1024 + ch] = acc[14];
                    out2[cur * 1920 + 1408 + ch] = acc[15];
                    out2[cur * 1920 + 1792 + ch] = acc[16];
                    out3[cur * 1792 + 128 + ch] = acc[17];
                    out3[cur * 1792 + 384 + ch] = acc[18];
                    out3[cur * 1792 + 640 + ch] = acc[19];
                    out3[cur * 1792 + 896 + ch] = acc[20];
                    out3[cur * 1792 + 1152 + ch] = acc[21];
                    out3[cur * 1792 + 1408 + ch] = acc[22];
                    out3[cur * 1792 + 1664 + ch] = acc[23];
                    #pragma unroll
                    for (int i = 0; i < 24; ++i) acc[i] = 0.f;
                    ++cur;
                }
                            {
                            const float sh0 = sh_s[b * ED + 0];
                            const float sh1 = sh_s[b * ED + 1];
                            const float sh2 = sh_s[b * ED + 2];
                            const float sh3 = sh_s[b * ED + 3];
                            const float sh4 = sh_s[b * ED + 4];
                            const float sh5 = sh_s[b * ED + 5];
                            const float sh6 = sh_s[b * ED + 6];
                            const float sh7 = sh_s[b * ED + 7];
                            const float sh8 = sh_s[b * ED + 8];
                            const float sh9 = sh_s[b * ED + 9];
                            const float sh10 = sh_s[b * ED + 10];
                            const float sh11 = sh_s[b * ED + 11];
                            const float sh12 = sh_s[b * ED + 12];
                            const float sh13 = sh_s[b * ED + 13];
                            const float sh14 = sh_s[b * ED + 14];
                            const float sh15 = sh_s[b * ED + 15];
                            const float p0_1 = xv[b][0] * sh1;
                            const float p0_4 = xv[b][0] * sh4;
                            const float p0_5 = xv[b][0] * sh5;
                            const float p0_6 = xv[b][0] * sh6;
                            const float p0_8 = xv[b][0] * sh8;
                            const float p0_9 = xv[b][0] * sh9;
                            const float p0_11 = xv[b][0] * sh11;
                            const float p0_12 = xv[b][0] * sh12;
                            const float p0_13 = xv[b][0] * sh13;
                            const float p0_15 = xv[b][0] * sh15;
                            const float p1_2 = xv[b][1] * sh2;
                            const float p1_5 = xv[b][1] * sh5;
                            const float p1_6 = xv[b][1] * sh6;
                            const float p1_7 = xv[b][1] * sh7;
                            const float p1_11 = xv[b][1] * sh11;
                            const float p1_12 = xv[b][1] * sh12;
                            const float p1_13 = xv[b][1] * sh13;
                            const float p2_3 = xv[b][2] * sh3;
                            const float p2_4 = xv[b][2] * sh4;
                            const float p2_6 = xv[b][2] * sh6;
                            const float p2_7 = xv[b][2] * sh7;
                            const float p2_8 = xv[b][2] * sh8;
                            const float p2_9 = xv[b][2] * sh9;
                            const float p2_11 = xv[b][2] * sh11;
                            const float p2_12 = xv[b][2] * sh12;
                            const float p2_13 = xv[b][2] * sh13;
                            const float p2_15 = xv[b][2] * sh15;
                            float t0 = p0_1;
                            t0 = t0 + p1_2;
                            t0 = t0 + p2_3;
                            acc[0] = fmaf(w[0][b], t0, acc[0]);
                            const float s1 = w[1][b] * sh0;  // 24 (c): w e0 first
                            acc[1] = fmaf(s1, xv[b][0], acc[1]);
                            acc[2] = fmaf(s1, xv[b][1], acc[2]);
                            acc[3] = fmaf(s1, xv[b][2], acc[3]);
                            float t4 = -p0_8;
                            t4 = t4 + p1_5;
                            t4 = t4 + p2_4;
                            t4 = fmaf(p0_6, -0x1.279a740000000p-1f, t4);
                            acc[4] = fmaf(w[2][b], t4, acc[4]);
                            float t5 = p0_5;
                            t5 = t5 + p2_7;
                            t5 = fmaf(p1_6, 0x1.279a740000000p+0f, t5);
                            acc[5] = fmaf(w[2][b], t5, acc[5]);
                            float t6 = p0_4;
                            t6 = t6 + p1_7;
                            t6 = t6 + p2_8;
                            t6 = fmaf(p2_6, -0x1.279a740000000p-1f, t6);
                            acc[6] = fmaf(w[2][b], t6, acc[6]);
                            float t7 = xv[b][0] * sh3;
                            t7 = fmaf(xv[b][2], sh1, t7);
                            acc[7] = fmaf(w[3][b], t7, acc[7]);
                            float t8 = xv[b][0] * sh2;
                            t8 = fmaf(xv[b][1], sh1, t8);
                            acc[8] = fmaf(w[3][b], t8, acc[8]);
                            float t9 = p0_1 * -0x1.279a760000000p-1f;
                            t9 = fmaf(p1_2, 0x1.279a760000000p+0f, t9);
                            t9 = fmaf(p2_3, -0x1.279a760000000p-1f, t9);
                            acc[9] = fmaf(w[3][b], t9, acc[9]);
                            float t10 = xv[b][1] * sh3;
                            t10 = fmaf(xv[b][2], sh2, t10);
                            acc[10] = fmaf(w[3][b], t10, acc[10]);
                            float t11 = -p0_1;
                            t11 = t11 + p2_3;
                            acc[11] = fmaf(w[3][b], t11, acc[11]);
                            float t12 = xv[b][1] * sh10;
                            t12 = fmaf(p0_13, -0x1.43d1380000000p-2f, t12);
                            t12 = fmaf(p0_15, -0x1.3988e20000000p+0f, t12);
                            t12 = fmaf(p2_9, 0x1.3988e20000000p+0f, t12);
                            t12 = fmaf(p2_11, -0x1.43d1380000000p-2f, t12);
                            acc[12] = fmaf(w[4][b], t12, acc[12]);
                            float t13 = -xv[b][0] * sh14;
                            t13 = fmaf(xv[b][2], sh10, t13);
                            t13 = fmaf(p0_12, -0x1.8c97f00000000p-1f, t13);
                            t13 = fmaf(p1_11, 0x1.43d1380000000p+0f, t13);
                            acc[13] = fmaf(w[4][b], t13, acc[13]);
                            float t14 = p0_11 * 0x1.186f180000000p+0f;
                            t14 = fmaf(p1_12, 0x1.5775c60000000p+0f, t14);
                            t14 = fmaf(p2_13, 0x1.186f180000000p+0f, t14);
                            acc[14] = fmaf(w[4][b], t14, acc[14]);
                            float t15 = xv[b][0] * sh10;
                            t15 = fmaf(xv[b][2], sh14, t15);
                            t15 = fmaf(p1_13, 0x1.43d1380000000p+0f, t15);
                            t15 = fmaf(p2_12, -0x1.8c97f00000000p-1f, t15);
                            acc[15] = fmaf(w[4][b], t15, acc[15]);
                            float t16 = xv[b][1] * sh14;
                            t16 = fmaf(p0_9, 0x1.3988e20000000p+0f, t16);
                            t16 = fmaf(p0_11, 0x1.43d1380000000p-2f, t16);
                            t16 = fmaf(p2_13, -0x1.43d1380000000p-2f, t16);
                            t16 = fmaf(p2_15, 0x1.3988e20000000p+0f, t16);
                            acc[16] = fmaf(w[4][b], t16, acc[16]);
                            float t17 = p0_8 * 0x1.3988e20000000p+0f;
                            t17 = fmaf(p2_4, 0x1.3988e20000000p+0f, t17);
                            acc[17] = fmaf(w[5][b], t17, acc[17]);
                            float t18 = xv[b][0] * sh7;
                            t18 = fmaf(xv[b][1], sh4, t18);
                            t18 = fmaf(xv[b][2], sh5, t18);
                            acc[18] = fmaf(w[5][b], t18, acc[18]);
                            float t19 = p0_6 * 0x1.186f180000000p+0f;
                            t19 = fmaf(p0_8, 0x1.43d1360000000p-2f, t19);
                            t19 = fmaf(p1_5, 0x1.43d1360000000p+0f, t19);
                            t19 = fmaf(p2_4, -0x1.43d1360000000p-2f, t19);
                            acc[19] = fmaf(w[5][b], t19, acc[19]);
                            float t20 = p0_5 * -0x1.8c97f00000000p-1f;
                            t20 = fmaf(p1_6, 0x1.5775c60000000p+0f, t20);
                            t20 = fmaf(p2_7, -0x1.8c97f00000000p-1f, t20);
                            acc[20] = fmaf(w[5][b], t20, acc[20]);
                            float t21 = p0_4 * -0x1.43d1360000000p-2f;
                            t21 = fmaf(p1_7, 0x1.43d1360000000p+0f, t21);
                            t21 = fmaf(p2_6, 0x1.186f180000000p+0f, t21);
                            t21 = fmaf(p2_8, -0x1.43d1360000000p-2f, t21);
                            acc[21] = fmaf(w[5][b], t21, acc[21]);
                            float t22 = -p0_5;
                            t22 = t22 + p2_7;
                            t22 = fmaf(xv[b][1], sh8, t22);
                            acc[22] = fmaf(w[5][b], t22, acc[22]);
                            float t23 = p0_4 * -0x1.3988e20000000p+0f;
                            t23 = fmaf(p2_8, 0x1.3988e20000000p+0f, t23);
                            acc[23] = fmaf(w[5][b], t23, acc[23]);
                            }
                        }
                    }
                }
                e0 += B;
            }
        }
    } break;
    }
}
