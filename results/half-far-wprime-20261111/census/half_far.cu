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

// HALF_FAR: packed half precision by PTX (no cuda_fp16.h: NVRTC finds no headers here). h2: edge e0 in the low half.
typedef unsigned int h2;
__device__ __forceinline__ h2 hfma(h2 a, h2 b, h2 c) {
    h2 d; asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }
__device__ __forceinline__ h2 hfms(h2 a, h2 b, h2 c) {  // c - a b
    h2 d; asm("{.reg .b32 n; neg.f16x2 n, %1; fma.rn.f16x2 %0, n, %2, %3;}" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d; }
__device__ __forceinline__ h2 hmul(h2 a, h2 b) {
    h2 d; asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
__device__ __forceinline__ h2 pack(float lo, float hi) {
    h2 d; asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo)); return d; }
__device__ __forceinline__ float lo2f(h2 a) {
    float f; asm("{.reg .f16 l, h; mov.b32 {l, h}, %1; cvt.f32.f16 %0, l;}" : "=f"(f) : "r"(a)); return f; }
__device__ __forceinline__ float hi2f(h2 a) {
    float f; asm("{.reg .f16 l, h; mov.b32 {l, h}, %1; cvt.f32.f16 %0, h;}" : "=f"(f) : "r"(a)); return f; }
__device__ __forceinline__ h2 dbc(float v) {  // one value in both halves
    h2 d; asm("cvt.rn.f16x2.f32 %0, %1, %1;" : "=r"(d) : "f"(v)); return d; }

// HALF_FAR: one thread per receiver run (edges in order) pairs consecutive far edges; every edge's skip flag, and
// each receiver's flag (1: it has a pair; zeroed by the caller).
extern "C" __global__ void wprime_half_pairs(
    int64_t edge_count, int64_t node_count, const int32_t* __restrict__ senders,
    const int32_t* __restrict__ receivers, const float* __restrict__ lengths, float r_half, float r_max,
    unsigned char* __restrict__ skip, int2* __restrict__ pairs, int* __restrict__ count,
    unsigned char* __restrict__ paired) {
    const int64_t e = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= edge_count) return;
    const int32_t rc = receivers[e];
    if (e > 0 && receivers[e - 1] == rc) return;  // not the start of a run
    const bool valid_rc = rc >= 0 && rc < node_count;
    int64_t open = -1;
    for (int64_t k = e; k < edge_count && receivers[k] == rc; ++k) {
        const int32_t s = senders[k];
        const float r = lengths[k];
        unsigned char flag = 0;
        if (valid_rc && s >= 0 && s < node_count && r > r_half && r < r_max) {
            if (open < 0) {
                open = k;
            } else {
                const int slot = atomicAdd(count, 1);
                pairs[slot] = make_int2(int(open), int(k));
                paired[rc] = 1;
                skip[open] = 1;
                flag = 1;
                open = -1;
            }
        }
        if (k != open) skip[k] = flag;
        else skip[k] = 0;
    }
}

// HALF_FAR: each paired receiver's dout row scale, a power of two taking its largest |value| into [2^DOUT_TOP,
// 2^(DOUT_TOP + 1)) (a third of the far rows' values are otherwise float16-subnormal). A warp per node.
extern "C" __global__ void wprime_half_scale(
    int64_t node_count,
    const unsigned char* __restrict__ paired,
    const float* __restrict__ dout0,
    const float* __restrict__ dout1,
    const float* __restrict__ dout2,
    const float* __restrict__ dout3,
    float* __restrict__ dscale) {
    const int lane = threadIdx.x & 31;
    const int64_t node = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    if (node >= node_count || !paired[node]) return;
    float top = 0.f;
        const float* row0 = dout0 + int64_t(node) * 256;
        for (int j = lane; j < 256; j += 32) top = fmaxf(top, fabsf(__ldg(row0 + j)));
        const float* row1 = dout1 + int64_t(node) * 1152;
        for (int j = lane; j < 1152; j += 32) top = fmaxf(top, fabsf(__ldg(row1 + j)));
        const float* row2 = dout2 + int64_t(node) * 1920;
        for (int j = lane; j < 1920; j += 32) top = fmaxf(top, fabsf(__ldg(row2 + j)));
        const float* row3 = dout3 + int64_t(node) * 1792;
        for (int j = lane; j < 1792; j += 32) top = fmaxf(top, fabsf(__ldg(row3 + j)));
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) top = fmaxf(top, __shfl_xor_sync(FULL_MASK, top, offset, 32));
    int e;
    frexpf(top, &e);  // top = f 2^e, f in [0.5, 1)
    if (lane == 0) dscale[node] = top > 0.f ? ldexpf(1.f, 4 + 1 - e) : 1.f;
}

extern "C" __global__ void __launch_bounds__(256, 3) wprime_half_far(
    int64_t node_count,
    const int2* __restrict__ pairs,
    const int* __restrict__ count,
    const float* __restrict__ x,
    const float* __restrict__ sh,
    const int32_t* __restrict__ senders,
    const int32_t* __restrict__ receivers,
    const float* __restrict__ dout0,
    const float* __restrict__ dout1,
    const float* __restrict__ dout2,
    const float* __restrict__ dout3,
    float* dx,
    float* dsh,
    const float* __restrict__ lengths,
    const int32_t* __restrict__ species,
    const unsigned short* __restrict__ gh,
    const float* __restrict__ ginv,
    int32_t pieces,
    const float* __restrict__ dscale,
    float* dr) {
    __shared__ h2 sh2s[8][EDGE_DIM];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int n = *count;
    for (int pidx = blockIdx.x * 8 + w; pidx < n; pidx += gridDim.x * 8) {
        const int2 pr = pairs[pidx];
        const int64_t e0 = pr.x, e1 = pr.y;
        const int32_t rcv = receivers[e0];
        const int64_t ib0 = int64_t(senders[e0]) * INPUT_DIM, ib1 = int64_t(senders[e1]) * INPUT_DIM;
        if (lane < EDGE_DIM) sh2s[w][lane] = pack(sh[e0 * EDGE_DIM + lane], sh[e1 * EDGE_DIM + lane]);
        float t0, t1, sx0, sx1;
        int iv0, iv1;
        {
            const float clamped = lengths[e0];
            int iv; float t, sx_inv;
            if (clamped < 0x1.0000000000000p-1f) { sx_inv = 0x1.6800000000000p+5f; float u = (clamped - 0x1.3333333333333p-2f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 8 ? il : 8; t = u - (float)il; iv = 0 + il; }
            else if (clamped < 0x1.4000000000000p+1f) { sx_inv = 0x1.6800000000000p+5f; float u = (clamped - 0x1.0000000000000p-1f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 89 ? il : 89; t = u - (float)il; iv = 9 + il; }
            else { sx_inv = 0x1.66db6db6db6dbp+5f; float u = (clamped - 0x1.4000000000000p+1f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 156 ? il : 156; t = u - (float)il; iv = 99 + il; }
            t0 = t; sx0 = sx_inv; iv0 = iv;
        }
        {
            const float clamped = lengths[e1];
            int iv; float t, sx_inv;
            if (clamped < 0x1.0000000000000p-1f) { sx_inv = 0x1.6800000000000p+5f; float u = (clamped - 0x1.3333333333333p-2f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 8 ? il : 8; t = u - (float)il; iv = 0 + il; }
            else if (clamped < 0x1.4000000000000p+1f) { sx_inv = 0x1.6800000000000p+5f; float u = (clamped - 0x1.0000000000000p-1f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 89 ? il : 89; t = u - (float)il; iv = 9 + il; }
            else { sx_inv = 0x1.66db6db6db6dbp+5f; float u = (clamped - 0x1.4000000000000p+1f) * sx_inv; u = u > 0x0.0p+0f ? u : 0x0.0p+0f; int il = (int)u; il = il < 156 ? il : 156; t = u - (float)il; iv = 99 + il; }
            t1 = t; sx1 = sx_inv; iv1 = iv;
        }
        const h2 t2 = pack(t0, t1);
        const int64_t slab0 = int64_t(species[e0]) * pieces + iv0, slab1 = int64_t(species[e1]) * pieces + iv1;
        // the receiver's dout row times a power of two (dscale) and each slab's table scale: undone in float32
        const float dsc = dscale[rcv];
        const float inv0 = ginv[slab0] / dsc, inv1 = ginv[slab1] / dsc;
        const unsigned short* g0 = gh + slab0 * int64_t(WEIGHT_DIM) * 4;
        const unsigned short* g1 = gh + slab1 * int64_t(WEIGHT_DIM) * 4;
        const float* drow0 = dout0 + int64_t(rcv) * 256;
        const float* drow1 = dout1 + int64_t(rcv) * 1152;
        const float* drow2 = dout2 + int64_t(rcv) * 1920;
        const float* drow3 = dout3 + int64_t(rcv) * 1792;
        __syncwarp();
        const h2* sh2 = sh2s[w];
        h2 edge_gradient[EDGE_DIM];
        #pragma unroll
        for (int j = 0; j < EDGE_DIM; ++j) edge_gradient[j] = 0u;
        float rg0 = 0.f, rg1 = 0.f;
        for (int channel = lane; channel < CHANNEL_DIM; channel += LOGICAL_GROUP_SIZE) {
            h2 edge_value, input_gradient_0_0, input_gradient_128_0, input_gradient_128_1, input_gradient_128_2, input_value, output_gradient, scaled, scaled_weight, u_0, u_1, u_2, v_0, v_1, v_2, v_3, v_4, v_5, v_6, weight_gradient, weighted_input;
            h2 radial_weight, radial_dt;
            float slope0, slope1;
            const float* xa = x + ib0 + channel;
            const float* xb = x + ib1 + channel;
            float* dxa = dx + ib0 + channel;
            float* dxb = dx + ib1 + channel;
            const uint2* gca = reinterpret_cast<const uint2*>(g0) + channel;
            const uint2* gcb = reinterpret_cast<const uint2*>(g1) + channel;
            const float* dc0 = drow0 + channel;
            const float* dc1 = drow1 + channel;
            const float* dc2 = drow2 + channel;
            const float* dc3 = drow3 + channel;
            input_gradient_0_0 = 0u;
            input_gradient_128_0 = 0u;
            input_gradient_128_1 = 0u;
            input_gradient_128_2 = 0u;
            // path at weight row 0
            {
                const uint2 ga = __ldg(gca + 0);
                const uint2 gb = __ldg(gcb + 0);
                radial_weight = __byte_perm(ga.y, gb.y, 0x7632);
                radial_dt = radial_weight;
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.y, gb.y, 0x5410));
                radial_dt = hfma(radial_dt, t2, radial_weight);
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.x, gb.x, 0x7632));
                radial_dt = hfma(radial_dt, t2, radial_weight);
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.x, gb.x, 0x5410));
            }
            weight_gradient = 0u;
            u_0 = 0u;
            weighted_input = hmul(radial_weight, pack(__ldg(xa + 0), __ldg(xb + 0)));
            output_gradient = dbc(__ldg(dc0 + 0) * dsc);
            input_value = pack(__ldg(xa + 0), __ldg(xb + 0));
            edge_value = sh2[0];
            scaled = output_gradient;
            u_0 = hfma(scaled, edge_value, u_0);
            edge_gradient[0] = hfma(weighted_input, scaled, edge_gradient[0]);
            scaled_weight = radial_weight;
            weight_gradient = hfma(pack(__ldg(xa + 0), __ldg(xb + 0)), u_0, weight_gradient);
            input_gradient_0_0 = hfma(scaled_weight, u_0, input_gradient_0_0);
            slope0 = lo2f(radial_dt) * sx0; slope1 = hi2f(radial_dt) * sx1;
            rg0 = fmaf(lo2f(weight_gradient), slope0, rg0); rg1 = fmaf(hi2f(weight_gradient), slope1, rg1);
            // path at weight row 128
            {
                const uint2 ga = __ldg(gca + 128);
                const uint2 gb = __ldg(gcb + 128);
                radial_weight = __byte_perm(ga.y, gb.y, 0x7632);
                radial_dt = radial_weight;
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.y, gb.y, 0x5410));
                radial_dt = hfma(radial_dt, t2, radial_weight);
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.x, gb.x, 0x7632));
                radial_dt = hfma(radial_dt, t2, radial_weight);
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.x, gb.x, 0x5410));
            }
            weight_gradient = 0u;
            u_0 = 0u;
            u_1 = 0u;
            u_2 = 0u;
            v_0 = 0u;
            v_1 = 0u;
            v_2 = 0u;
            output_gradient = dbc(__ldg(dc0 + 128) * dsc);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[1];
            scaled = output_gradient;
            u_0 = hfma(scaled, edge_value, u_0);
            v_0 = hfma(scaled, input_value, v_0);
            input_value = pack(__ldg(xa + 256), __ldg(xb + 256));
            edge_value = sh2[2];
            scaled = output_gradient;
            u_1 = hfma(scaled, edge_value, u_1);
            v_1 = hfma(scaled, input_value, v_1);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[3];
            scaled = output_gradient;
            u_2 = hfma(scaled, edge_value, u_2);
            v_2 = hfma(scaled, input_value, v_2);
            scaled_weight = radial_weight;
            weight_gradient = hfma(pack(__ldg(xa + 128), __ldg(xb + 128)), u_0, weight_gradient);
            input_gradient_128_0 = hfma(scaled_weight, u_0, input_gradient_128_0);
            weight_gradient = hfma(pack(__ldg(xa + 256), __ldg(xb + 256)), u_1, weight_gradient);
            input_gradient_128_1 = hfma(scaled_weight, u_1, input_gradient_128_1);
            weight_gradient = hfma(pack(__ldg(xa + 384), __ldg(xb + 384)), u_2, weight_gradient);
            input_gradient_128_2 = hfma(scaled_weight, u_2, input_gradient_128_2);
            edge_gradient[1] = hfma(scaled_weight, v_0, edge_gradient[1]);
            edge_gradient[2] = hfma(scaled_weight, v_1, edge_gradient[2]);
            edge_gradient[3] = hfma(scaled_weight, v_2, edge_gradient[3]);
            slope0 = lo2f(radial_dt) * sx0; slope1 = hi2f(radial_dt) * sx1;
            rg0 = fmaf(lo2f(weight_gradient), slope0, rg0); rg1 = fmaf(hi2f(weight_gradient), slope1, rg1);
            // path at weight row 256
            {
                const uint2 ga = __ldg(gca + 256);
                const uint2 gb = __ldg(gcb + 256);
                radial_weight = __byte_perm(ga.y, gb.y, 0x7632);
                radial_dt = radial_weight;
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.y, gb.y, 0x5410));
                radial_dt = hfma(radial_dt, t2, radial_weight);
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.x, gb.x, 0x7632));
                radial_dt = hfma(radial_dt, t2, radial_weight);
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.x, gb.x, 0x5410));
            }
            weight_gradient = 0u;
            u_0 = 0u;
            weighted_input = hmul(radial_weight, pack(__ldg(xa + 0), __ldg(xb + 0)));
            output_gradient = dbc(__ldg(dc1 + 0) * dsc);
            input_value = pack(__ldg(xa + 0), __ldg(xb + 0));
            edge_value = sh2[1];
            scaled = output_gradient;
            u_0 = hfma(scaled, edge_value, u_0);
            edge_gradient[1] = hfma(weighted_input, scaled, edge_gradient[1]);
            output_gradient = dbc(__ldg(dc1 + 384) * dsc);
            input_value = pack(__ldg(xa + 0), __ldg(xb + 0));
            edge_value = sh2[2];
            scaled = output_gradient;
            u_0 = hfma(scaled, edge_value, u_0);
            edge_gradient[2] = hfma(weighted_input, scaled, edge_gradient[2]);
            output_gradient = dbc(__ldg(dc1 + 768) * dsc);
            input_value = pack(__ldg(xa + 0), __ldg(xb + 0));
            edge_value = sh2[3];
            scaled = output_gradient;
            u_0 = hfma(scaled, edge_value, u_0);
            edge_gradient[3] = hfma(weighted_input, scaled, edge_gradient[3]);
            scaled_weight = radial_weight;
            weight_gradient = hfma(pack(__ldg(xa + 0), __ldg(xb + 0)), u_0, weight_gradient);
            input_gradient_0_0 = hfma(scaled_weight, u_0, input_gradient_0_0);
            slope0 = lo2f(radial_dt) * sx0; slope1 = hi2f(radial_dt) * sx1;
            rg0 = fmaf(lo2f(weight_gradient), slope0, rg0); rg1 = fmaf(hi2f(weight_gradient), slope1, rg1);
            // path at weight row 384
            {
                const uint2 ga = __ldg(gca + 384);
                const uint2 gb = __ldg(gcb + 384);
                radial_weight = __byte_perm(ga.y, gb.y, 0x7632);
                radial_dt = radial_weight;
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.y, gb.y, 0x5410));
                radial_dt = hfma(radial_dt, t2, radial_weight);
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.x, gb.x, 0x7632));
                radial_dt = hfma(radial_dt, t2, radial_weight);
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.x, gb.x, 0x5410));
            }
            weight_gradient = 0u;
            u_0 = 0u;
            u_1 = 0u;
            u_2 = 0u;
            v_0 = 0u;
            output_gradient = dbc(__ldg(dc1 + 128) * dsc);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[0];
            scaled = output_gradient;
            u_0 = hfma(scaled, edge_value, u_0);
            v_0 = hfma(scaled, input_value, v_0);
            output_gradient = dbc(__ldg(dc1 + 512) * dsc);
            input_value = pack(__ldg(xa + 256), __ldg(xb + 256));
            edge_value = sh2[0];
            scaled = output_gradient;
            u_1 = hfma(scaled, edge_value, u_1);
            v_0 = hfma(scaled, input_value, v_0);
            output_gradient = dbc(__ldg(dc1 + 896) * dsc);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[0];
            scaled = output_gradient;
            u_2 = hfma(scaled, edge_value, u_2);
            v_0 = hfma(scaled, input_value, v_0);
            scaled_weight = radial_weight;
            weight_gradient = hfma(pack(__ldg(xa + 128), __ldg(xb + 128)), u_0, weight_gradient);
            input_gradient_128_0 = hfma(scaled_weight, u_0, input_gradient_128_0);
            weight_gradient = hfma(pack(__ldg(xa + 256), __ldg(xb + 256)), u_1, weight_gradient);
            input_gradient_128_1 = hfma(scaled_weight, u_1, input_gradient_128_1);
            weight_gradient = hfma(pack(__ldg(xa + 384), __ldg(xb + 384)), u_2, weight_gradient);
            input_gradient_128_2 = hfma(scaled_weight, u_2, input_gradient_128_2);
            edge_gradient[0] = hfma(scaled_weight, v_0, edge_gradient[0]);
            slope0 = lo2f(radial_dt) * sx0; slope1 = hi2f(radial_dt) * sx1;
            rg0 = fmaf(lo2f(weight_gradient), slope0, rg0); rg1 = fmaf(hi2f(weight_gradient), slope1, rg1);
            // path at weight row 512
            {
                const uint2 ga = __ldg(gca + 512);
                const uint2 gb = __ldg(gcb + 512);
                radial_weight = __byte_perm(ga.y, gb.y, 0x7632);
                radial_dt = radial_weight;
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.y, gb.y, 0x5410));
                radial_dt = hfma(radial_dt, t2, radial_weight);
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.x, gb.x, 0x7632));
                radial_dt = hfma(radial_dt, t2, radial_weight);
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.x, gb.x, 0x5410));
            }
            weight_gradient = 0u;
            u_0 = 0u;
            u_1 = 0u;
            u_2 = 0u;
            v_2 = 0u;
            v_4 = 0u;
            v_1 = 0u;
            v_0 = 0u;
            v_3 = 0u;
            output_gradient = dbc(__ldg(dc1 + 256) * dsc);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[6];
            scaled = hmul(output_gradient, 0x389e389eu);
            u_0 = hfms(scaled, edge_value, u_0);
            v_2 = hfms(scaled, input_value, v_2);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[8];
            scaled = output_gradient;
            u_0 = hfms(scaled, edge_value, u_0);
            v_4 = hfms(scaled, input_value, v_4);
            input_value = pack(__ldg(xa + 256), __ldg(xb + 256));
            edge_value = sh2[5];
            scaled = output_gradient;
            u_1 = hfma(scaled, edge_value, u_1);
            v_1 = hfma(scaled, input_value, v_1);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[4];
            scaled = output_gradient;
            u_2 = hfma(scaled, edge_value, u_2);
            v_0 = hfma(scaled, input_value, v_0);
            output_gradient = dbc(__ldg(dc1 + 640) * dsc);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[5];
            scaled = output_gradient;
            u_0 = hfma(scaled, edge_value, u_0);
            v_1 = hfma(scaled, input_value, v_1);
            input_value = pack(__ldg(xa + 256), __ldg(xb + 256));
            edge_value = sh2[6];
            scaled = hmul(output_gradient, 0x3c9e3c9eu);
            u_1 = hfma(scaled, edge_value, u_1);
            v_2 = hfma(scaled, input_value, v_2);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[7];
            scaled = output_gradient;
            u_2 = hfma(scaled, edge_value, u_2);
            v_3 = hfma(scaled, input_value, v_3);
            output_gradient = dbc(__ldg(dc1 + 1024) * dsc);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[4];
            scaled = output_gradient;
            u_0 = hfma(scaled, edge_value, u_0);
            v_0 = hfma(scaled, input_value, v_0);
            input_value = pack(__ldg(xa + 256), __ldg(xb + 256));
            edge_value = sh2[7];
            scaled = output_gradient;
            u_1 = hfma(scaled, edge_value, u_1);
            v_3 = hfma(scaled, input_value, v_3);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[6];
            scaled = hmul(output_gradient, 0x389e389eu);
            u_2 = hfms(scaled, edge_value, u_2);
            v_2 = hfms(scaled, input_value, v_2);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[8];
            scaled = output_gradient;
            u_2 = hfma(scaled, edge_value, u_2);
            v_4 = hfma(scaled, input_value, v_4);
            scaled_weight = radial_weight;
            weight_gradient = hfma(pack(__ldg(xa + 128), __ldg(xb + 128)), u_0, weight_gradient);
            input_gradient_128_0 = hfma(scaled_weight, u_0, input_gradient_128_0);
            weight_gradient = hfma(pack(__ldg(xa + 256), __ldg(xb + 256)), u_1, weight_gradient);
            input_gradient_128_1 = hfma(scaled_weight, u_1, input_gradient_128_1);
            weight_gradient = hfma(pack(__ldg(xa + 384), __ldg(xb + 384)), u_2, weight_gradient);
            input_gradient_128_2 = hfma(scaled_weight, u_2, input_gradient_128_2);
            edge_gradient[6] = hfma(scaled_weight, v_2, edge_gradient[6]);
            edge_gradient[8] = hfma(scaled_weight, v_4, edge_gradient[8]);
            edge_gradient[5] = hfma(scaled_weight, v_1, edge_gradient[5]);
            edge_gradient[4] = hfma(scaled_weight, v_0, edge_gradient[4]);
            edge_gradient[7] = hfma(scaled_weight, v_3, edge_gradient[7]);
            slope0 = lo2f(radial_dt) * sx0; slope1 = hi2f(radial_dt) * sx1;
            rg0 = fmaf(lo2f(weight_gradient), slope0, rg0); rg1 = fmaf(hi2f(weight_gradient), slope1, rg1);
            // path at weight row 640
            {
                const uint2 ga = __ldg(gca + 640);
                const uint2 gb = __ldg(gcb + 640);
                radial_weight = __byte_perm(ga.y, gb.y, 0x7632);
                radial_dt = radial_weight;
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.y, gb.y, 0x5410));
                radial_dt = hfma(radial_dt, t2, radial_weight);
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.x, gb.x, 0x7632));
                radial_dt = hfma(radial_dt, t2, radial_weight);
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.x, gb.x, 0x5410));
            }
            weight_gradient = 0u;
            u_0 = 0u;
            weighted_input = hmul(radial_weight, pack(__ldg(xa + 0), __ldg(xb + 0)));
            output_gradient = dbc(__ldg(dc2 + 0) * dsc);
            input_value = pack(__ldg(xa + 0), __ldg(xb + 0));
            edge_value = sh2[4];
            scaled = output_gradient;
            u_0 = hfma(scaled, edge_value, u_0);
            edge_gradient[4] = hfma(weighted_input, scaled, edge_gradient[4]);
            output_gradient = dbc(__ldg(dc2 + 384) * dsc);
            input_value = pack(__ldg(xa + 0), __ldg(xb + 0));
            edge_value = sh2[5];
            scaled = output_gradient;
            u_0 = hfma(scaled, edge_value, u_0);
            edge_gradient[5] = hfma(weighted_input, scaled, edge_gradient[5]);
            output_gradient = dbc(__ldg(dc2 + 768) * dsc);
            input_value = pack(__ldg(xa + 0), __ldg(xb + 0));
            edge_value = sh2[6];
            scaled = output_gradient;
            u_0 = hfma(scaled, edge_value, u_0);
            edge_gradient[6] = hfma(weighted_input, scaled, edge_gradient[6]);
            output_gradient = dbc(__ldg(dc2 + 1152) * dsc);
            input_value = pack(__ldg(xa + 0), __ldg(xb + 0));
            edge_value = sh2[7];
            scaled = output_gradient;
            u_0 = hfma(scaled, edge_value, u_0);
            edge_gradient[7] = hfma(weighted_input, scaled, edge_gradient[7]);
            output_gradient = dbc(__ldg(dc2 + 1536) * dsc);
            input_value = pack(__ldg(xa + 0), __ldg(xb + 0));
            edge_value = sh2[8];
            scaled = output_gradient;
            u_0 = hfma(scaled, edge_value, u_0);
            edge_gradient[8] = hfma(weighted_input, scaled, edge_gradient[8]);
            scaled_weight = radial_weight;
            weight_gradient = hfma(pack(__ldg(xa + 0), __ldg(xb + 0)), u_0, weight_gradient);
            input_gradient_0_0 = hfma(scaled_weight, u_0, input_gradient_0_0);
            slope0 = lo2f(radial_dt) * sx0; slope1 = hi2f(radial_dt) * sx1;
            rg0 = fmaf(lo2f(weight_gradient), slope0, rg0); rg1 = fmaf(hi2f(weight_gradient), slope1, rg1);
            // path at weight row 768
            {
                const uint2 ga = __ldg(gca + 768);
                const uint2 gb = __ldg(gcb + 768);
                radial_weight = __byte_perm(ga.y, gb.y, 0x7632);
                radial_dt = radial_weight;
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.y, gb.y, 0x5410));
                radial_dt = hfma(radial_dt, t2, radial_weight);
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.x, gb.x, 0x7632));
                radial_dt = hfma(radial_dt, t2, radial_weight);
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.x, gb.x, 0x5410));
            }
            weight_gradient = 0u;
            u_0 = 0u;
            u_2 = 0u;
            u_1 = 0u;
            v_2 = 0u;
            v_0 = 0u;
            v_1 = 0u;
            output_gradient = dbc(__ldg(dc2 + 128) * dsc);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[3];
            scaled = output_gradient;
            u_0 = hfma(scaled, edge_value, u_0);
            v_2 = hfma(scaled, input_value, v_2);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[1];
            scaled = output_gradient;
            u_2 = hfma(scaled, edge_value, u_2);
            v_0 = hfma(scaled, input_value, v_0);
            output_gradient = dbc(__ldg(dc2 + 512) * dsc);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[2];
            scaled = output_gradient;
            u_0 = hfma(scaled, edge_value, u_0);
            v_1 = hfma(scaled, input_value, v_1);
            input_value = pack(__ldg(xa + 256), __ldg(xb + 256));
            edge_value = sh2[1];
            scaled = output_gradient;
            u_1 = hfma(scaled, edge_value, u_1);
            v_0 = hfma(scaled, input_value, v_0);
            output_gradient = dbc(__ldg(dc2 + 896) * dsc);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[1];
            scaled = hmul(output_gradient, 0x389e389eu);
            u_0 = hfms(scaled, edge_value, u_0);
            v_0 = hfms(scaled, input_value, v_0);
            input_value = pack(__ldg(xa + 256), __ldg(xb + 256));
            edge_value = sh2[2];
            scaled = hmul(output_gradient, 0x3c9e3c9eu);
            u_1 = hfma(scaled, edge_value, u_1);
            v_1 = hfma(scaled, input_value, v_1);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[3];
            scaled = hmul(output_gradient, 0x389e389eu);
            u_2 = hfms(scaled, edge_value, u_2);
            v_2 = hfms(scaled, input_value, v_2);
            output_gradient = dbc(__ldg(dc2 + 1280) * dsc);
            input_value = pack(__ldg(xa + 256), __ldg(xb + 256));
            edge_value = sh2[3];
            scaled = output_gradient;
            u_1 = hfma(scaled, edge_value, u_1);
            v_2 = hfma(scaled, input_value, v_2);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[2];
            scaled = output_gradient;
            u_2 = hfma(scaled, edge_value, u_2);
            v_1 = hfma(scaled, input_value, v_1);
            output_gradient = dbc(__ldg(dc2 + 1664) * dsc);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[1];
            scaled = output_gradient;
            u_0 = hfms(scaled, edge_value, u_0);
            v_0 = hfms(scaled, input_value, v_0);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[3];
            scaled = output_gradient;
            u_2 = hfma(scaled, edge_value, u_2);
            v_2 = hfma(scaled, input_value, v_2);
            scaled_weight = radial_weight;
            weight_gradient = hfma(pack(__ldg(xa + 128), __ldg(xb + 128)), u_0, weight_gradient);
            input_gradient_128_0 = hfma(scaled_weight, u_0, input_gradient_128_0);
            weight_gradient = hfma(pack(__ldg(xa + 384), __ldg(xb + 384)), u_2, weight_gradient);
            input_gradient_128_2 = hfma(scaled_weight, u_2, input_gradient_128_2);
            weight_gradient = hfma(pack(__ldg(xa + 256), __ldg(xb + 256)), u_1, weight_gradient);
            input_gradient_128_1 = hfma(scaled_weight, u_1, input_gradient_128_1);
            edge_gradient[3] = hfma(scaled_weight, v_2, edge_gradient[3]);
            edge_gradient[1] = hfma(scaled_weight, v_0, edge_gradient[1]);
            edge_gradient[2] = hfma(scaled_weight, v_1, edge_gradient[2]);
            slope0 = lo2f(radial_dt) * sx0; slope1 = hi2f(radial_dt) * sx1;
            rg0 = fmaf(lo2f(weight_gradient), slope0, rg0); rg1 = fmaf(hi2f(weight_gradient), slope1, rg1);
            // path at weight row 896
            {
                const uint2 ga = __ldg(gca + 896);
                const uint2 gb = __ldg(gcb + 896);
                radial_weight = __byte_perm(ga.y, gb.y, 0x7632);
                radial_dt = radial_weight;
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.y, gb.y, 0x5410));
                radial_dt = hfma(radial_dt, t2, radial_weight);
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.x, gb.x, 0x7632));
                radial_dt = hfma(radial_dt, t2, radial_weight);
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.x, gb.x, 0x5410));
            }
            weight_gradient = 0u;
            u_0 = 0u;
            u_1 = 0u;
            u_2 = 0u;
            v_4 = 0u;
            v_6 = 0u;
            v_1 = 0u;
            v_0 = 0u;
            v_2 = 0u;
            v_3 = 0u;
            v_5 = 0u;
            output_gradient = dbc(__ldg(dc2 + 256) * dsc);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[13];
            scaled = hmul(output_gradient, 0x350f350fu);
            u_0 = hfms(scaled, edge_value, u_0);
            v_4 = hfms(scaled, input_value, v_4);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[15];
            scaled = hmul(output_gradient, 0x3ce63ce6u);
            u_0 = hfms(scaled, edge_value, u_0);
            v_6 = hfms(scaled, input_value, v_6);
            input_value = pack(__ldg(xa + 256), __ldg(xb + 256));
            edge_value = sh2[10];
            scaled = output_gradient;
            u_1 = hfma(scaled, edge_value, u_1);
            v_1 = hfma(scaled, input_value, v_1);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[9];
            scaled = hmul(output_gradient, 0x3ce63ce6u);
            u_2 = hfma(scaled, edge_value, u_2);
            v_0 = hfma(scaled, input_value, v_0);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[11];
            scaled = hmul(output_gradient, 0x350f350fu);
            u_2 = hfms(scaled, edge_value, u_2);
            v_2 = hfms(scaled, input_value, v_2);
            output_gradient = dbc(__ldg(dc2 + 640) * dsc);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[12];
            scaled = hmul(output_gradient, 0x3a323a32u);
            u_0 = hfms(scaled, edge_value, u_0);
            v_3 = hfms(scaled, input_value, v_3);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[14];
            scaled = output_gradient;
            u_0 = hfms(scaled, edge_value, u_0);
            v_5 = hfms(scaled, input_value, v_5);
            input_value = pack(__ldg(xa + 256), __ldg(xb + 256));
            edge_value = sh2[11];
            scaled = hmul(output_gradient, 0x3d0f3d0fu);
            u_1 = hfma(scaled, edge_value, u_1);
            v_2 = hfma(scaled, input_value, v_2);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[10];
            scaled = output_gradient;
            u_2 = hfma(scaled, edge_value, u_2);
            v_1 = hfma(scaled, input_value, v_1);
            output_gradient = dbc(__ldg(dc2 + 1024) * dsc);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[11];
            scaled = hmul(output_gradient, 0x3c623c62u);
            u_0 = hfma(scaled, edge_value, u_0);
            v_2 = hfma(scaled, input_value, v_2);
            input_value = pack(__ldg(xa + 256), __ldg(xb + 256));
            edge_value = sh2[12];
            scaled = hmul(output_gradient, 0x3d5e3d5eu);
            u_1 = hfma(scaled, edge_value, u_1);
            v_3 = hfma(scaled, input_value, v_3);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[13];
            scaled = hmul(output_gradient, 0x3c623c62u);
            u_2 = hfma(scaled, edge_value, u_2);
            v_4 = hfma(scaled, input_value, v_4);
            output_gradient = dbc(__ldg(dc2 + 1408) * dsc);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[10];
            scaled = output_gradient;
            u_0 = hfma(scaled, edge_value, u_0);
            v_1 = hfma(scaled, input_value, v_1);
            input_value = pack(__ldg(xa + 256), __ldg(xb + 256));
            edge_value = sh2[13];
            scaled = hmul(output_gradient, 0x3d0f3d0fu);
            u_1 = hfma(scaled, edge_value, u_1);
            v_4 = hfma(scaled, input_value, v_4);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[12];
            scaled = hmul(output_gradient, 0x3a323a32u);
            u_2 = hfms(scaled, edge_value, u_2);
            v_3 = hfms(scaled, input_value, v_3);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[14];
            scaled = output_gradient;
            u_2 = hfma(scaled, edge_value, u_2);
            v_5 = hfma(scaled, input_value, v_5);
            output_gradient = dbc(__ldg(dc2 + 1792) * dsc);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[9];
            scaled = hmul(output_gradient, 0x3ce63ce6u);
            u_0 = hfma(scaled, edge_value, u_0);
            v_0 = hfma(scaled, input_value, v_0);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[11];
            scaled = hmul(output_gradient, 0x350f350fu);
            u_0 = hfma(scaled, edge_value, u_0);
            v_2 = hfma(scaled, input_value, v_2);
            input_value = pack(__ldg(xa + 256), __ldg(xb + 256));
            edge_value = sh2[14];
            scaled = output_gradient;
            u_1 = hfma(scaled, edge_value, u_1);
            v_5 = hfma(scaled, input_value, v_5);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[13];
            scaled = hmul(output_gradient, 0x350f350fu);
            u_2 = hfms(scaled, edge_value, u_2);
            v_4 = hfms(scaled, input_value, v_4);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[15];
            scaled = hmul(output_gradient, 0x3ce63ce6u);
            u_2 = hfma(scaled, edge_value, u_2);
            v_6 = hfma(scaled, input_value, v_6);
            scaled_weight = radial_weight;
            weight_gradient = hfma(pack(__ldg(xa + 128), __ldg(xb + 128)), u_0, weight_gradient);
            input_gradient_128_0 = hfma(scaled_weight, u_0, input_gradient_128_0);
            weight_gradient = hfma(pack(__ldg(xa + 256), __ldg(xb + 256)), u_1, weight_gradient);
            input_gradient_128_1 = hfma(scaled_weight, u_1, input_gradient_128_1);
            weight_gradient = hfma(pack(__ldg(xa + 384), __ldg(xb + 384)), u_2, weight_gradient);
            input_gradient_128_2 = hfma(scaled_weight, u_2, input_gradient_128_2);
            edge_gradient[13] = hfma(scaled_weight, v_4, edge_gradient[13]);
            edge_gradient[15] = hfma(scaled_weight, v_6, edge_gradient[15]);
            edge_gradient[10] = hfma(scaled_weight, v_1, edge_gradient[10]);
            edge_gradient[9] = hfma(scaled_weight, v_0, edge_gradient[9]);
            edge_gradient[11] = hfma(scaled_weight, v_2, edge_gradient[11]);
            edge_gradient[12] = hfma(scaled_weight, v_3, edge_gradient[12]);
            edge_gradient[14] = hfma(scaled_weight, v_5, edge_gradient[14]);
            slope0 = lo2f(radial_dt) * sx0; slope1 = hi2f(radial_dt) * sx1;
            rg0 = fmaf(lo2f(weight_gradient), slope0, rg0); rg1 = fmaf(hi2f(weight_gradient), slope1, rg1);
            // path at weight row 1024
            {
                const uint2 ga = __ldg(gca + 1024);
                const uint2 gb = __ldg(gcb + 1024);
                radial_weight = __byte_perm(ga.y, gb.y, 0x7632);
                radial_dt = radial_weight;
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.y, gb.y, 0x5410));
                radial_dt = hfma(radial_dt, t2, radial_weight);
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.x, gb.x, 0x7632));
                radial_dt = hfma(radial_dt, t2, radial_weight);
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.x, gb.x, 0x5410));
            }
            weight_gradient = 0u;
            u_0 = 0u;
            weighted_input = hmul(radial_weight, pack(__ldg(xa + 0), __ldg(xb + 0)));
            output_gradient = dbc(__ldg(dc3 + 0) * dsc);
            input_value = pack(__ldg(xa + 0), __ldg(xb + 0));
            edge_value = sh2[9];
            scaled = output_gradient;
            u_0 = hfma(scaled, edge_value, u_0);
            edge_gradient[9] = hfma(weighted_input, scaled, edge_gradient[9]);
            output_gradient = dbc(__ldg(dc3 + 256) * dsc);
            input_value = pack(__ldg(xa + 0), __ldg(xb + 0));
            edge_value = sh2[10];
            scaled = output_gradient;
            u_0 = hfma(scaled, edge_value, u_0);
            edge_gradient[10] = hfma(weighted_input, scaled, edge_gradient[10]);
            output_gradient = dbc(__ldg(dc3 + 512) * dsc);
            input_value = pack(__ldg(xa + 0), __ldg(xb + 0));
            edge_value = sh2[11];
            scaled = output_gradient;
            u_0 = hfma(scaled, edge_value, u_0);
            edge_gradient[11] = hfma(weighted_input, scaled, edge_gradient[11]);
            output_gradient = dbc(__ldg(dc3 + 768) * dsc);
            input_value = pack(__ldg(xa + 0), __ldg(xb + 0));
            edge_value = sh2[12];
            scaled = output_gradient;
            u_0 = hfma(scaled, edge_value, u_0);
            edge_gradient[12] = hfma(weighted_input, scaled, edge_gradient[12]);
            output_gradient = dbc(__ldg(dc3 + 1024) * dsc);
            input_value = pack(__ldg(xa + 0), __ldg(xb + 0));
            edge_value = sh2[13];
            scaled = output_gradient;
            u_0 = hfma(scaled, edge_value, u_0);
            edge_gradient[13] = hfma(weighted_input, scaled, edge_gradient[13]);
            output_gradient = dbc(__ldg(dc3 + 1280) * dsc);
            input_value = pack(__ldg(xa + 0), __ldg(xb + 0));
            edge_value = sh2[14];
            scaled = output_gradient;
            u_0 = hfma(scaled, edge_value, u_0);
            edge_gradient[14] = hfma(weighted_input, scaled, edge_gradient[14]);
            output_gradient = dbc(__ldg(dc3 + 1536) * dsc);
            input_value = pack(__ldg(xa + 0), __ldg(xb + 0));
            edge_value = sh2[15];
            scaled = output_gradient;
            u_0 = hfma(scaled, edge_value, u_0);
            edge_gradient[15] = hfma(weighted_input, scaled, edge_gradient[15]);
            scaled_weight = radial_weight;
            weight_gradient = hfma(pack(__ldg(xa + 0), __ldg(xb + 0)), u_0, weight_gradient);
            input_gradient_0_0 = hfma(scaled_weight, u_0, input_gradient_0_0);
            slope0 = lo2f(radial_dt) * sx0; slope1 = hi2f(radial_dt) * sx1;
            rg0 = fmaf(lo2f(weight_gradient), slope0, rg0); rg1 = fmaf(hi2f(weight_gradient), slope1, rg1);
            // path at weight row 1152
            {
                const uint2 ga = __ldg(gca + 1152);
                const uint2 gb = __ldg(gcb + 1152);
                radial_weight = __byte_perm(ga.y, gb.y, 0x7632);
                radial_dt = radial_weight;
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.y, gb.y, 0x5410));
                radial_dt = hfma(radial_dt, t2, radial_weight);
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.x, gb.x, 0x7632));
                radial_dt = hfma(radial_dt, t2, radial_weight);
                radial_weight = hfma(radial_weight, t2, __byte_perm(ga.x, gb.x, 0x5410));
            }
            weight_gradient = 0u;
            u_0 = 0u;
            u_2 = 0u;
            u_1 = 0u;
            v_4 = 0u;
            v_0 = 0u;
            v_3 = 0u;
            v_1 = 0u;
            v_2 = 0u;
            output_gradient = dbc(__ldg(dc3 + 128) * dsc);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[8];
            scaled = hmul(output_gradient, 0x3ce63ce6u);
            u_0 = hfma(scaled, edge_value, u_0);
            v_4 = hfma(scaled, input_value, v_4);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[4];
            scaled = hmul(output_gradient, 0x3ce63ce6u);
            u_2 = hfma(scaled, edge_value, u_2);
            v_0 = hfma(scaled, input_value, v_0);
            output_gradient = dbc(__ldg(dc3 + 384) * dsc);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[7];
            scaled = output_gradient;
            u_0 = hfma(scaled, edge_value, u_0);
            v_3 = hfma(scaled, input_value, v_3);
            input_value = pack(__ldg(xa + 256), __ldg(xb + 256));
            edge_value = sh2[4];
            scaled = output_gradient;
            u_1 = hfma(scaled, edge_value, u_1);
            v_0 = hfma(scaled, input_value, v_0);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[5];
            scaled = output_gradient;
            u_2 = hfma(scaled, edge_value, u_2);
            v_1 = hfma(scaled, input_value, v_1);
            output_gradient = dbc(__ldg(dc3 + 640) * dsc);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[6];
            scaled = hmul(output_gradient, 0x3c623c62u);
            u_0 = hfma(scaled, edge_value, u_0);
            v_2 = hfma(scaled, input_value, v_2);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[8];
            scaled = hmul(output_gradient, 0x350f350fu);
            u_0 = hfma(scaled, edge_value, u_0);
            v_4 = hfma(scaled, input_value, v_4);
            input_value = pack(__ldg(xa + 256), __ldg(xb + 256));
            edge_value = sh2[5];
            scaled = hmul(output_gradient, 0x3d0f3d0fu);
            u_1 = hfma(scaled, edge_value, u_1);
            v_1 = hfma(scaled, input_value, v_1);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[4];
            scaled = hmul(output_gradient, 0x350f350fu);
            u_2 = hfms(scaled, edge_value, u_2);
            v_0 = hfms(scaled, input_value, v_0);
            output_gradient = dbc(__ldg(dc3 + 896) * dsc);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[5];
            scaled = hmul(output_gradient, 0x3a323a32u);
            u_0 = hfms(scaled, edge_value, u_0);
            v_1 = hfms(scaled, input_value, v_1);
            input_value = pack(__ldg(xa + 256), __ldg(xb + 256));
            edge_value = sh2[6];
            scaled = hmul(output_gradient, 0x3d5e3d5eu);
            u_1 = hfma(scaled, edge_value, u_1);
            v_2 = hfma(scaled, input_value, v_2);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[7];
            scaled = hmul(output_gradient, 0x3a323a32u);
            u_2 = hfms(scaled, edge_value, u_2);
            v_3 = hfms(scaled, input_value, v_3);
            output_gradient = dbc(__ldg(dc3 + 1152) * dsc);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[4];
            scaled = hmul(output_gradient, 0x350f350fu);
            u_0 = hfms(scaled, edge_value, u_0);
            v_0 = hfms(scaled, input_value, v_0);
            input_value = pack(__ldg(xa + 256), __ldg(xb + 256));
            edge_value = sh2[7];
            scaled = hmul(output_gradient, 0x3d0f3d0fu);
            u_1 = hfma(scaled, edge_value, u_1);
            v_3 = hfma(scaled, input_value, v_3);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[6];
            scaled = hmul(output_gradient, 0x3c623c62u);
            u_2 = hfma(scaled, edge_value, u_2);
            v_2 = hfma(scaled, input_value, v_2);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[8];
            scaled = hmul(output_gradient, 0x350f350fu);
            u_2 = hfms(scaled, edge_value, u_2);
            v_4 = hfms(scaled, input_value, v_4);
            output_gradient = dbc(__ldg(dc3 + 1408) * dsc);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[5];
            scaled = output_gradient;
            u_0 = hfms(scaled, edge_value, u_0);
            v_1 = hfms(scaled, input_value, v_1);
            input_value = pack(__ldg(xa + 256), __ldg(xb + 256));
            edge_value = sh2[8];
            scaled = output_gradient;
            u_1 = hfma(scaled, edge_value, u_1);
            v_4 = hfma(scaled, input_value, v_4);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[7];
            scaled = output_gradient;
            u_2 = hfma(scaled, edge_value, u_2);
            v_3 = hfma(scaled, input_value, v_3);
            output_gradient = dbc(__ldg(dc3 + 1664) * dsc);
            input_value = pack(__ldg(xa + 128), __ldg(xb + 128));
            edge_value = sh2[4];
            scaled = hmul(output_gradient, 0x3ce63ce6u);
            u_0 = hfms(scaled, edge_value, u_0);
            v_0 = hfms(scaled, input_value, v_0);
            input_value = pack(__ldg(xa + 384), __ldg(xb + 384));
            edge_value = sh2[8];
            scaled = hmul(output_gradient, 0x3ce63ce6u);
            u_2 = hfma(scaled, edge_value, u_2);
            v_4 = hfma(scaled, input_value, v_4);
            scaled_weight = radial_weight;
            weight_gradient = hfma(pack(__ldg(xa + 128), __ldg(xb + 128)), u_0, weight_gradient);
            input_gradient_128_0 = hfma(scaled_weight, u_0, input_gradient_128_0);
            weight_gradient = hfma(pack(__ldg(xa + 384), __ldg(xb + 384)), u_2, weight_gradient);
            input_gradient_128_2 = hfma(scaled_weight, u_2, input_gradient_128_2);
            weight_gradient = hfma(pack(__ldg(xa + 256), __ldg(xb + 256)), u_1, weight_gradient);
            input_gradient_128_1 = hfma(scaled_weight, u_1, input_gradient_128_1);
            edge_gradient[8] = hfma(scaled_weight, v_4, edge_gradient[8]);
            edge_gradient[4] = hfma(scaled_weight, v_0, edge_gradient[4]);
            edge_gradient[7] = hfma(scaled_weight, v_3, edge_gradient[7]);
            edge_gradient[5] = hfma(scaled_weight, v_1, edge_gradient[5]);
            edge_gradient[6] = hfma(scaled_weight, v_2, edge_gradient[6]);
            slope0 = lo2f(radial_dt) * sx0; slope1 = hi2f(radial_dt) * sx1;
            rg0 = fmaf(lo2f(weight_gradient), slope0, rg0); rg1 = fmaf(hi2f(weight_gradient), slope1, rg1);
            atomicAdd(dxa + 0, lo2f(input_gradient_0_0) * inv0);
            atomicAdd(dxb + 0, hi2f(input_gradient_0_0) * inv1);
            atomicAdd(dxa + 128, lo2f(input_gradient_128_0) * inv0);
            atomicAdd(dxb + 128, hi2f(input_gradient_128_0) * inv1);
            atomicAdd(dxa + 256, lo2f(input_gradient_128_1) * inv0);
            atomicAdd(dxb + 256, hi2f(input_gradient_128_1) * inv1);
            atomicAdd(dxa + 384, lo2f(input_gradient_128_2) * inv0);
            atomicAdd(dxb + 384, hi2f(input_gradient_128_2) * inv1);
        }
        // the 32 per-lane values (16 per edge) reduce-scattered in float32: lane l ends with value l
        float v[32];
        #pragma unroll
        for (int j = 0; j < EDGE_DIM; ++j) { v[j] = lo2f(edge_gradient[j]); v[EDGE_DIM + j] = hi2f(edge_gradient[j]); }
        #pragma unroll
        for (int step = 0; step < 5; ++step) {
            const int half = 16 >> step;
            const bool upper = (lane & half) != 0;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                if (i < half) {
                    const float send = upper ? v[i] : v[i + half];
                    const float keep = upper ? v[i + half] : v[i];
                    v[i] = keep + __shfl_xor_sync(FULL_MASK, send, half, 32);
                }
            }
        }
        dsh[(lane < EDGE_DIM ? e0 : e1) * EDGE_DIM + (lane & (EDGE_DIM - 1))] = v[0] * (lane < EDGE_DIM ? inv0 : inv1);
        float r0 = rg0, r1 = rg1;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            r0 += __shfl_xor_sync(FULL_MASK, r0, offset, 32);
            r1 += __shfl_xor_sync(FULL_MASK, r1, offset, 32);
        }
        if (lane == 0) { dr[e0] = r0 * inv0; dr[e1] = r1 * inv1; }
        __syncwarp();
    }
}
