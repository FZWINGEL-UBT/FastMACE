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
constexpr int A_BASE = 15360;  // the warps' staging: a_s of warp v at A_BASE + v * A_STRIDE
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

__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mb_init(unsigned bar, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mb_arrive(unsigned bar) {
    asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ unsigned mb_try(unsigned bar, unsigned parity) {
    unsigned ok;
    asm volatile("{\n.reg .pred P1;\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n"
                 "selp.b32 %0, 1, 0, P1;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok;
}
__device__ __forceinline__ void mb_wait(unsigned bar, unsigned parity) {  // a watchdog: trap after ~10 s (a deadlock)
    if (mb_try(bar, parity)) return;
    const long long t0 = clock64();
    while (!mb_try(bar, parity))
        if (clock64() - t0 > 20000000000LL) asm volatile("trap;");
}
__device__ __forceinline__ void cp_async16(float* dst, const float* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(smem_u32(dst)), "l"(src), "r"(bytes)
                 : "memory");
}
__device__ __forceinline__ void cp_async_arrive(unsigned bar) {
    asm volatile("cp.async.mbarrier.arrive.shared::cta.b64 [%0];" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void producer_sync() {
    asm volatile("bar.sync 1, 128;" ::: "memory");
}
__device__ __forceinline__ unsigned long long gmma_desc(unsigned addr) {
    return (unsigned long long)((addr & 0x3FFFFu) >> 4) | ((unsigned long long)(128 >> 4) << 16)
        | ((unsigned long long)((64 * 32) >> 4) << 32);
}
__device__ __forceinline__ void wgmma_n96(float* d, const unsigned (&a)[4],
                                                unsigned long long desc, int scale) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %53, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n96k8.f32.tf32.tf32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, {%48, %49, %50, "
                 "%51}, %52, p, 1, 1;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale) : "memory");
}
__device__ __forceinline__ void wgmma_n128(float* d, const unsigned (&a)[4],
                                                unsigned long long desc, int scale) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, "
                 "%67}, %68, p, 1, 1;\n}\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale) : "memory");
}
// idea 30: sm_90a (WS_FORWARD: warp-specialized wgmma.mma_async forward)
// a is read with VEC_A's row stride AS: reinterpret_cast<const float4*>(a + (e0 + b) * AS)
constexpr int NC = 8;  // consumer warps (warps 4 .. 4 + NC - 1)
constexpr int ROWS = 8 * NC;  // edges per round: one 8-edge batch per consumer warp
constexpr int AW = KP + 4;  // the a tile's row stride

extern "C" __global__ void __launch_bounds__(384, 1) fused_folded_forward(
    int64_t N, int64_t E, const float* __restrict__ a, const float* __restrict__ W,
    const float* __restrict__ x, const float* __restrict__ sh, const int* __restrict__ senders,
    const int* __restrict__ receivers, const int* __restrict__ row_ptr, int* __restrict__ counters,
    float* __restrict__ out0, float* __restrict__ out1, float* __restrict__ out2, float* __restrict__ out3, float* __restrict__ wout) {
    extern __shared__ float smem[];
    // ORDER swap: part 1 first
    const int bz_ = (int)gridDim.z - 1 - (int)blockIdx.z, by_ = blockIdx.y;
    float* Ws = smem;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float* a_s = Ws + A_BASE + warp * A_STRIDE;
    float* sh_s = a_s + KP * B;
    float* r_s = sh_s + B * ED;  // tensor cores: the warp's redistribution tile [paths][B][RS]
    const int cb = by_;
    const int ch = cb * CB + lane;
    int* ctr = counters + by_ * gridDim.z + bz_;
    // Receiver chunks first, then chunks of the padding edges past the last row.
    const int64_t receiver_chunks = (N + RC - 1) / RC;
    int64_t tail_begin = row_ptr[N] < 0 ? 0 : row_ptr[N];
    tail_begin = tail_begin > E ? E : tail_begin;
    switch (bz_) {
    case 0: {  // WS_FORWARD: a producer warpgroup, 8 consumers, 2 stages
        constexpr int NW = 128;  // the part's W rows (paths x 32 channels): the product's N
        constexpr int S = 2;  // ring stages; the TP runs S - 1 rounds behind the staging
        constexpr int WREG = 4 * ROWS * RS;  // a stage's w tile [paths][ROWS][RS]; the a tile [ROWS][AW] on it
        constexpr int STG = WREG + ROWS * ED;  // w tile, then the sh tile [ROWS][ED]
        constexpr int W_FLOATS = 16384;  // W's high and low TF32 planes
        constexpr int BAR_OFF = W_FLOATS + S * STG;
        static_assert(ROWS * AW <= WREG, "the a tile fits inside the w tile");
        int* flags = reinterpret_cast<int*>(smem + BAR_OFF + 6 * S);  // active[S], fin[S]
        const unsigned bar0 = smem_u32(smem + BAR_OFF);  // full[s] at + 8 s, empty[s] + 8 (S + s), wready + 8 (2 S + s)
        {  // W's TF32 planes (high, low) for wgmma's B: [N][KP], 8 x 4 core matrices, no swizzle
            const int wst[4] = {0, 256, 640, 1024};
            
            float* Wl = Ws + NW * KP;
            for (int i = threadIdx.x; i < NW * KP; i += blockDim.x) {
                const int nn = i / KP, k = i - nn * KP, p = nn >> 5, l = nn & 31;
                unsigned hi, lo;
                split_tf32(k < K ? W[(int64_t)(wst[p] + cb * CB + l) * KA + k] : 0.f, hi, lo);
                const int o = ((nn >> 3) * (KP / 4) + (k >> 2)) * 32 + (nn & 7) * 4 + (k & 3);
                Ws[o] = __uint_as_float(hi);
                Wl[o] = __uint_as_float(lo);
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        if (threadIdx.x == 0) {
            for (int s = 0; s < S; ++s) {
                mb_init(bar0 + 8 * s, NC * 32);
                mb_init(bar0 + 8 * (S + s), NC * 32);
                mb_init(bar0 + 8 * (2 * S + s), 128);
                flags[s] = 0;
                flags[S + s] = 0;
            }
        }
        __syncthreads();
        if (warp < 4) {  // the producer warpgroup
        const int pw = warp, gid = lane >> 2, tq = lane & 3;
        const unsigned wh = smem_u32(Ws), wl = wh + NW * KP * 4;
        (void)wh; (void)wl;
        for (int n = 0;; ++n) {
            const int s = n % S;
            mb_wait(bar0 + 8 * s, (n / S) & 1);  // full: the round's a and sh staged
            float* st = smem + W_FLOATS + s * STG;
            const int act = *(volatile int*)&flags[s];
            producer_sync();
            if (threadIdx.x == 0) {
                flags[s] = 0;
                flags[S + s] = act == 0;
            }
            if (act == 0) {  // no consumer had a batch: the end
                mb_arrive(bar0 + 8 * (2 * S + s));
                break;
            }
            unsigned ah[KP / 8][4], al[KP / 8][4];  // A: edge 16 pw + gid (+8), k 8 ks + tq
            #pragma unroll
            for (int ks = 0; ks < KP / 8; ++ks) {
                const float* ar = st + (16 * pw + gid) * AW + 8 * ks + tq;
                split_tf32(ar[0], ah[ks][0], al[ks][0]);
                split_tf32(ar[8 * AW], ah[ks][1], al[ks][1]);
                split_tf32(ar[4], ah[ks][2], al[ks][2]);
                split_tf32(ar[8 * AW + 4], ah[ks][3], al[ks][3]);
            }
            producer_sync();  // the whole a tile is in registers: D may overwrite it
            #pragma unroll
            for (int h = 0; h < 1; ++h) {
                float d[64];
                asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int ks = 0; ks < KP / 8; ++ks) {  // two 128-byte core matrices per k step
                    const unsigned off = h * 16 * (KP * 32) + 256 * ks;
                    wgmma_n128(d, al[ks], gmma_desc(wh + off), ks > 0);
                    wgmma_n128(d, ah[ks], gmma_desc(wl + off), 1);
                    wgmma_n128(d, ah[ks], gmma_desc(wh + off), 1);
                }
                asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                #pragma unroll
                for (int j = 0; j < 16; ++j) {  // D: edge 16 pw + gid (+8), column h N/2 + 8 j + 2 tq (+1)
                    const int col = h * 128 + 8 * j + 2 * tq, p = col >> 5, c = col & 31;
                    float* rp = st + (p * ROWS + 16 * pw + gid) * RS + c;
                    *reinterpret_cast<float2*>(rp) = make_float2(d[4 * j], d[4 * j + 1]);
                    *reinterpret_cast<float2*>(rp + 8 * RS) = make_float2(d[4 * j + 2], d[4 * j + 3]);
                }
            }
            mb_arrive(bar0 + 8 * (2 * S + s));  // wready: the round's w tile written
        }
        } else {  // consumer warps: today's chunks and TP, S - 1 rounds behind their staging
        const int c = warp - 4;
        bool have = false, done = false;
        int64_t r0 = 0, r1 = 0, eend = 0, e0 = 0, cur = 0;
        int q0_nb = 0;
        int q0_rv = 0x7fffffff;
        int q0_snd = -1;
        int64_t q0_r0 = 0;
        int64_t q0_r1 = 0;
        bool q0_first = false;
        bool q0_last = false;
        float acc[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) acc[i] = 0.f;
        for (int n = 0;; ++n) {
            const int s = n % S;
            bool first = false;
            if (!done && (!have || e0 >= eend)) {  // the next chunk with edges; empty chunks' rows written as zeros
                have = false;
                while (true) {
                    int cc = 0;
                    if (lane == 0) cc = atomicAdd(ctr, 1);
                    cc = __shfl_sync(FULL, cc, 0);
                    if (cc >= receiver_chunks) {
                        done = true;
                        break;
                    }
                    r0 = (int64_t)cc * RC;
                    r1 = r0 + RC < N ? r0 + RC : N;
                    const int64_t ebeg = row_ptr[r0] < 0 ? 0 : row_ptr[r0];
                    eend = row_ptr[r1] < ebeg ? ebeg : row_ptr[r1];
                    eend = eend > E ? E : eend;
                    if (ebeg < eend) {
                        have = true;
                        first = true;
                        e0 = ebeg;
                        break;
                    }
                    for (int64_t zr = r0; zr < r1; ++zr) {
                            out0[zr * 256 + 0 + ch] = 0.f;
                            out1[zr * 1152 + 0 + ch] = 0.f;
                            out1[zr * 1152 + 384 + ch] = 0.f;
                            out1[zr * 1152 + 768 + ch] = 0.f;
                            out2[zr * 1920 + 0 + ch] = 0.f;
                            out2[zr * 1920 + 384 + ch] = 0.f;
                            out2[zr * 1920 + 768 + ch] = 0.f;
                            out2[zr * 1920 + 1152 + ch] = 0.f;
                            out2[zr * 1920 + 1536 + ch] = 0.f;
                            out3[zr * 1792 + 0 + ch] = 0.f;
                            out3[zr * 1792 + 256 + ch] = 0.f;
                            out3[zr * 1792 + 512 + ch] = 0.f;
                            out3[zr * 1792 + 768 + ch] = 0.f;
                            out3[zr * 1792 + 1024 + ch] = 0.f;
                            out3[zr * 1792 + 1280 + ch] = 0.f;
                            out3[zr * 1792 + 1536 + ch] = 0.f;
                    }
                }
            }
            const int nb = done ? 0 : (int)((eend - e0) < B ? (eend - e0) : B);
            if (n >= S) mb_wait(bar0 + 8 * (S + s), ((n / S) - 1) & 1);  // empty: round n - S's tiles read
            float* st = smem + W_FLOATS + s * STG;
            if (nb > 0) {
                for (int i = lane; i < B * (KP / 4); i += 32) {
                    const int b = i / (KP / 4), q = i - b * (KP / 4);
                    cp_async16(st + (8 * c + b) * AW + 4 * q, b < nb ? a + (e0 + b) * AS + 4 * q : a, b < nb ? 16 : 0);
                }
                for (int i = lane; i < B * ED / 4; i += 32)
                    cp_async16(st + WREG + 8 * c * ED + 4 * i, (4 * i / ED < nb) ? sh + e0 * ED + 4 * i : sh,
                               (4 * i / ED < nb) ? 16 : 0);
                if (lane == 0) atomicAdd(&flags[s], 1);
            }
            cp_async_arrive(bar0 + 8 * s);  // full: lands when this thread's copies have
            mb_arrive(bar0 + 8 * s);
            const int n_rv = (lane < nb) ? receivers[e0 + lane] : 0x7fffffff;
            const int n_snd = (lane < nb) ? senders[e0 + lane] : -1;
            const bool n_last = nb > 0 && e0 + nb >= eend;
            e0 += nb;
            if (n >= S - 1) {  // the TP of round m = n - (S - 1)
                const int m = n - (S - 1), ps = m % S;
                const int lane_rv = q0_rv, lane_snd = q0_snd;
                float xv[B][1];
                int rv[B];
                #pragma unroll
                for (int b = 0; b < B; ++b) {
                    rv[b] = __shfl_sync(FULL, lane_rv, b);
                    int snd = __shfl_sync(FULL, lane_snd, b);
                    if (snd < 0 || snd >= N) snd = -1;
                    xv[b][0] = (snd >= 0) ? x[(int64_t)snd * IN_DIM + 0 + ch] : 0.f;
                }
                mb_wait(bar0 + 8 * (2 * S + ps), (m / S) & 1);  // wready
                if (*(volatile int*)&flags[S + ps]) break;
                if (q0_nb > 0) {
                    const float* pst = smem + W_FLOATS + ps * STG;
                    const float* sh_s = pst + WREG + 8 * c * ED;
                    float w[4][B];
                    #pragma unroll
                    for (int g = 0; g < 4; ++g)
                        #pragma unroll
                        for (int b = 0; b < B; ++b) w[g][b] = pst[(g * ROWS + 8 * c + b) * RS + lane];
                    if (q0_first) cur = q0_r0;
                    {
                    const int nb = q0_nb;
                    const bool tail = false;
                if (!tail) {
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
                    if (q0_last) while (cur < q0_r1) {
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
                mb_arrive(bar0 + 8 * (S + ps));  // empty
            }

            q0_nb = nb;
            q0_rv = n_rv;
            q0_snd = n_snd;
            q0_r0 = r0;
            q0_r1 = r1;
            q0_first = first;
            q0_last = n_last;
        }
        }
    } break;
    case 1: {  // WS_FORWARD: a producer warpgroup, 8 consumers, 2 stages
        constexpr int NW = 192;  // the part's W rows (paths x 32 channels): the product's N
        constexpr int S = 2;  // ring stages; the TP runs S - 1 rounds behind the staging
        constexpr int WREG = 6 * ROWS * RS;  // a stage's w tile [paths][ROWS][RS]; the a tile [ROWS][AW] on it
        constexpr int STG = WREG + ROWS * ED;  // w tile, then the sh tile [ROWS][ED]
        constexpr int W_FLOATS = 24576;  // W's high and low TF32 planes
        constexpr int BAR_OFF = W_FLOATS + S * STG;
        static_assert(ROWS * AW <= WREG, "the a tile fits inside the w tile");
        int* flags = reinterpret_cast<int*>(smem + BAR_OFF + 6 * S);  // active[S], fin[S]
        const unsigned bar0 = smem_u32(smem + BAR_OFF);  // full[s] at + 8 s, empty[s] + 8 (S + s), wready + 8 (2 S + s)
        {  // W's TF32 planes (high, low) for wgmma's B: [N][KP], 8 x 4 core matrices, no swizzle
            const int wst[6] = {128, 384, 512, 768, 896, 1152};
            const float cpf[6] = {0x1.279a740000000p-1f, 0x1.0000000000000p+0f, 0x1.186f180000000p-1f, 0x1.6a09e60000000p-1f, 0x1.f3a92c0000000p-2f, 0x1.279a740000000p-1f};
            float* Wl = Ws + NW * KP;
            for (int i = threadIdx.x; i < NW * KP; i += blockDim.x) {
                const int nn = i / KP, k = i - nn * KP, p = nn >> 5, l = nn & 31;
                unsigned hi, lo;
                split_tf32(k < K ? cpf[p] * W[(int64_t)(wst[p] + cb * CB + l) * KA + k] : 0.f, hi, lo);
                const int o = ((nn >> 3) * (KP / 4) + (k >> 2)) * 32 + (nn & 7) * 4 + (k & 3);
                Ws[o] = __uint_as_float(hi);
                Wl[o] = __uint_as_float(lo);
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        if (threadIdx.x == 0) {
            for (int s = 0; s < S; ++s) {
                mb_init(bar0 + 8 * s, NC * 32);
                mb_init(bar0 + 8 * (S + s), NC * 32);
                mb_init(bar0 + 8 * (2 * S + s), 128);
                flags[s] = 0;
                flags[S + s] = 0;
            }
        }
        __syncthreads();
        if (warp < 4) {  // the producer warpgroup
        const int pw = warp, gid = lane >> 2, tq = lane & 3;
        const unsigned wh = smem_u32(Ws), wl = wh + NW * KP * 4;
        (void)wh; (void)wl;
        for (int n = 0;; ++n) {
            const int s = n % S;
            mb_wait(bar0 + 8 * s, (n / S) & 1);  // full: the round's a and sh staged
            float* st = smem + W_FLOATS + s * STG;
            const int act = *(volatile int*)&flags[s];
            producer_sync();
            if (threadIdx.x == 0) {
                flags[s] = 0;
                flags[S + s] = act == 0;
            }
            if (act == 0) {  // no consumer had a batch: the end
                mb_arrive(bar0 + 8 * (2 * S + s));
                break;
            }
            unsigned ah[KP / 8][4], al[KP / 8][4];  // A: edge 16 pw + gid (+8), k 8 ks + tq
            #pragma unroll
            for (int ks = 0; ks < KP / 8; ++ks) {
                const float* ar = st + (16 * pw + gid) * AW + 8 * ks + tq;
                split_tf32(ar[0], ah[ks][0], al[ks][0]);
                split_tf32(ar[8 * AW], ah[ks][1], al[ks][1]);
                split_tf32(ar[4], ah[ks][2], al[ks][2]);
                split_tf32(ar[8 * AW + 4], ah[ks][3], al[ks][3]);
            }
            producer_sync();  // the whole a tile is in registers: D may overwrite it
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                float d[48];
                asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int ks = 0; ks < KP / 8; ++ks) {  // two 128-byte core matrices per k step
                    const unsigned off = h * 12 * (KP * 32) + 256 * ks;
                    wgmma_n96(d, al[ks], gmma_desc(wh + off), ks > 0);
                    wgmma_n96(d, ah[ks], gmma_desc(wl + off), 1);
                    wgmma_n96(d, ah[ks], gmma_desc(wh + off), 1);
                }
                asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                #pragma unroll
                for (int j = 0; j < 12; ++j) {  // D: edge 16 pw + gid (+8), column h N/2 + 8 j + 2 tq (+1)
                    const int col = h * 96 + 8 * j + 2 * tq, p = col >> 5, c = col & 31;
                    float* rp = st + (p * ROWS + 16 * pw + gid) * RS + c;
                    *reinterpret_cast<float2*>(rp) = make_float2(d[4 * j], d[4 * j + 1]);
                    *reinterpret_cast<float2*>(rp + 8 * RS) = make_float2(d[4 * j + 2], d[4 * j + 3]);
                }
            }
            mb_arrive(bar0 + 8 * (2 * S + s));  // wready: the round's w tile written
        }
        } else {  // consumer warps: today's chunks and TP, S - 1 rounds behind their staging
        const int c = warp - 4;
        bool have = false, done = false;
        int64_t r0 = 0, r1 = 0, eend = 0, e0 = 0, cur = 0;
        int q0_nb = 0;
        int q0_rv = 0x7fffffff;
        int q0_snd = -1;
        int64_t q0_r0 = 0;
        int64_t q0_r1 = 0;
        bool q0_first = false;
        bool q0_last = false;
        float acc[24];
        #pragma unroll
        for (int i = 0; i < 24; ++i) acc[i] = 0.f;
        for (int n = 0;; ++n) {
            const int s = n % S;
            bool first = false;
            if (!done && (!have || e0 >= eend)) {  // the next chunk with edges; empty chunks' rows written as zeros
                have = false;
                while (true) {
                    int cc = 0;
                    if (lane == 0) cc = atomicAdd(ctr, 1);
                    cc = __shfl_sync(FULL, cc, 0);
                    if (cc >= receiver_chunks) {
                        done = true;
                        break;
                    }
                    r0 = (int64_t)cc * RC;
                    r1 = r0 + RC < N ? r0 + RC : N;
                    const int64_t ebeg = row_ptr[r0] < 0 ? 0 : row_ptr[r0];
                    eend = row_ptr[r1] < ebeg ? ebeg : row_ptr[r1];
                    eend = eend > E ? E : eend;
                    if (ebeg < eend) {
                        have = true;
                        first = true;
                        e0 = ebeg;
                        break;
                    }
                    for (int64_t zr = r0; zr < r1; ++zr) {
                            out0[zr * 256 + 128 + ch] = 0.f;
                            out1[zr * 1152 + 128 + ch] = 0.f;
                            out1[zr * 1152 + 512 + ch] = 0.f;
                            out1[zr * 1152 + 896 + ch] = 0.f;
                            out1[zr * 1152 + 256 + ch] = 0.f;
                            out1[zr * 1152 + 640 + ch] = 0.f;
                            out1[zr * 1152 + 1024 + ch] = 0.f;
                            out2[zr * 1920 + 128 + ch] = 0.f;
                            out2[zr * 1920 + 512 + ch] = 0.f;
                            out2[zr * 1920 + 896 + ch] = 0.f;
                            out2[zr * 1920 + 1280 + ch] = 0.f;
                            out2[zr * 1920 + 1664 + ch] = 0.f;
                            out2[zr * 1920 + 256 + ch] = 0.f;
                            out2[zr * 1920 + 640 + ch] = 0.f;
                            out2[zr * 1920 + 1024 + ch] = 0.f;
                            out2[zr * 1920 + 1408 + ch] = 0.f;
                            out2[zr * 1920 + 1792 + ch] = 0.f;
                            out3[zr * 1792 + 128 + ch] = 0.f;
                            out3[zr * 1792 + 384 + ch] = 0.f;
                            out3[zr * 1792 + 640 + ch] = 0.f;
                            out3[zr * 1792 + 896 + ch] = 0.f;
                            out3[zr * 1792 + 1152 + ch] = 0.f;
                            out3[zr * 1792 + 1408 + ch] = 0.f;
                            out3[zr * 1792 + 1664 + ch] = 0.f;
                    }
                }
            }
            const int nb = done ? 0 : (int)((eend - e0) < B ? (eend - e0) : B);
            if (n >= S) mb_wait(bar0 + 8 * (S + s), ((n / S) - 1) & 1);  // empty: round n - S's tiles read
            float* st = smem + W_FLOATS + s * STG;
            if (nb > 0) {
                for (int i = lane; i < B * (KP / 4); i += 32) {
                    const int b = i / (KP / 4), q = i - b * (KP / 4);
                    cp_async16(st + (8 * c + b) * AW + 4 * q, b < nb ? a + (e0 + b) * AS + 4 * q : a, b < nb ? 16 : 0);
                }
                for (int i = lane; i < B * ED / 4; i += 32)
                    cp_async16(st + WREG + 8 * c * ED + 4 * i, (4 * i / ED < nb) ? sh + e0 * ED + 4 * i : sh,
                               (4 * i / ED < nb) ? 16 : 0);
                if (lane == 0) atomicAdd(&flags[s], 1);
            }
            cp_async_arrive(bar0 + 8 * s);  // full: lands when this thread's copies have
            mb_arrive(bar0 + 8 * s);
            const int n_rv = (lane < nb) ? receivers[e0 + lane] : 0x7fffffff;
            const int n_snd = (lane < nb) ? senders[e0 + lane] : -1;
            const bool n_last = nb > 0 && e0 + nb >= eend;
            e0 += nb;
            if (n >= S - 1) {  // the TP of round m = n - (S - 1)
                const int m = n - (S - 1), ps = m % S;
                const int lane_rv = q0_rv, lane_snd = q0_snd;
                float xv[B][3];
                int rv[B];
                #pragma unroll
                for (int b = 0; b < B; ++b) {
                    rv[b] = __shfl_sync(FULL, lane_rv, b);
                    int snd = __shfl_sync(FULL, lane_snd, b);
                    if (snd < 0 || snd >= N) snd = -1;
                    xv[b][0] = (snd >= 0) ? x[(int64_t)snd * IN_DIM + 128 + ch] : 0.f;
                    xv[b][1] = (snd >= 0) ? x[(int64_t)snd * IN_DIM + 256 + ch] : 0.f;
                    xv[b][2] = (snd >= 0) ? x[(int64_t)snd * IN_DIM + 384 + ch] : 0.f;
                }
                mb_wait(bar0 + 8 * (2 * S + ps), (m / S) & 1);  // wready
                if (*(volatile int*)&flags[S + ps]) break;
                if (q0_nb > 0) {
                    const float* pst = smem + W_FLOATS + ps * STG;
                    const float* sh_s = pst + WREG + 8 * c * ED;
                    float w[6][B];
                    #pragma unroll
                    for (int g = 0; g < 6; ++g)
                        #pragma unroll
                        for (int b = 0; b < B; ++b) w[g][b] = pst[(g * ROWS + 8 * c + b) * RS + lane];
                    if (q0_first) cur = q0_r0;
                    {
                    const int nb = q0_nb;
                    const bool tail = false;
                if (!tail) {
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
                    if (q0_last) while (cur < q0_r1) {
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
                mb_arrive(bar0 + 8 * (S + ps));  // empty
            }

            q0_nb = nb;
            q0_rv = n_rv;
            q0_snd = n_snd;
            q0_r0 = r0;
            q0_r1 = r1;
            q0_first = first;
            q0_last = n_last;
        }
        }
    } break;
    }
}
