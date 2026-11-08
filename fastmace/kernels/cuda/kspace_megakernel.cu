// POLAR k-space field features and structure factors as persistent CUDA kernels (polar-megakernel, 7 November 2026).
//
// The same sums as fastmace/kernels/triton/kspace_megakernel.py (module docstring there), for one structure (full
// PBC): theta_kn = k_k . r_n,
//
//   Yc[k, c, j] = mask_k sum_n cos(theta_kn) x[n, c, j]          (Ys with sin)
//   ur, ui, a, b: the per-k glue (density basis dr, di; Coulomb and density scale w; projection basis pr, pi)
//   out[n, c, f] = s0 sum_k (a cos(theta_kn) + b sin(theta_kn))
//
// and the structure factor alone (Yc, Ys) for the Coulomb energy. The Triton kernel lost to the dense path on the
// RTX 5060 Ti (1.3-1.9x) because its phases ran one 64-row tile per block over all columns: 4 of 36 SMs busy on
// NaCl's 243 k-vectors, 4 warps per SM, and the 8- and 16-column products padded to 16 for tl.dot. Here:
//
// - every sum over pairs is split both ways: a work item is 32 rows (one per lane) times a chunk of the other
//   index, its 8 warps striding the chunk; the chunks are sized so there are at least two items per resident block.
//   Each item writes its own partial sums; a later phase adds them in chunk order (deterministic, no atomics);
// - the narrow per-entry factors of the strided index (positions and sources, or k-vectors and the per-k glue) are
//   staged in shared memory, FM_STAGE entries at a time, so each loop step reads them with a broadcast shared load
//   instead of an L2 round trip (__ldcg, ~300 cycles at 16 warps per SM);
// - dL/dk (stress) needs a sum over the atoms of each k-vector: the lanes write their terms to shared memory
//   (rows padded to 33) and the block adds each staged step's rows once, instead of a 5-step shuffle tree per
//   k-vector and warp;
// - each pair's phase is one precise sincosf (not the approximate __sincosf: |theta| reaches ~100 rad), then plain
//   FFMA with the narrow factors as warp-broadcast loads; nothing [k, atoms] leaves the registers;
// - the phases of one direction are separated by a grid barrier (an arrival counter; the grid is at most the
//   resident blocks, so every block is running).
//
// Data written in this launch and read after a barrier goes through L2 (__ldcg). Float32, row-major, contiguous.
// Entries, all with blockDim 256 and a grid of the resident blocks (fm_kspace_info: threads, rows per item, warps):
//   fm_kspace_field_forward_C_J_F, fm_kspace_field_backward_C_J_F_{factors}{grad_k},
//   fm_kspace_sf_forward_W, fm_kspace_sf_backward_W_{grad_k}.

#include <cstddef>

#define FM_THREADS 256
#define FM_WARPS 8
#define FM_ROWS 32
#define FM_STAGE 64  // entries of the strided index staged per step (8 per warp)
#define FM_STAGE_WIDTH 51  // the widest staged entry: k, dyc, dys (2 x 8), a, b (2 x 16)
#define FM_GK_OFFSET (FM_STAGE * FM_STAGE_WIDTH)  // dL/dk terms of a staged step: [FM_STAGE][3][33] (padded)

extern "C" __device__ int fm_kspace_info[3] = {FM_THREADS, FM_ROWS, FM_WARPS};

namespace {

__device__ __forceinline__ void grid_barrier(unsigned long long* counter) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned long long blocks = gridDim.x;
        __threadfence();
        const unsigned long long old = atomicAdd(counter, 1ULL);
        const unsigned long long target = (old / blocks + 1ULL) * blocks;
        while (atomicAdd(counter, 0ULL) < target) {
            __nanosleep(64);
        }
        __threadfence();
    }
    __syncthreads();
}

__device__ __forceinline__ float dot3(float ax, float ay, float az, float bx, float by, float bz) {
    return fmaf(az, bz, fmaf(ay, by, ax * bx));
}

// part[s, row, 0:W] = sum over the chunk of cos(theta) x[n, :], part[s, row, W:2W] the same with sin.
// rows: the k-vectors; chunks s of `chunk` atoms.
template <int W>
__device__ void ksum(const float* __restrict__ k, const float* __restrict__ r, const float* __restrict__ x,
                     float* __restrict__ part, int K, int N, int splits, int chunk, float* smem) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tiles = (K + FM_ROWS - 1) / FM_ROWS;
    for (int item = blockIdx.x; item < tiles * splits; item += gridDim.x) {
        const int tile = item % tiles, s = item / tiles;
        const int row = tile * FM_ROWS + lane;
        float kx = 0.f, ky = 0.f, kz = 0.f;
        if (row < K) {
            kx = k[row * 3];
            ky = k[row * 3 + 1];
            kz = k[row * 3 + 2];
        }
        float ac[W], as[W];
#pragma unroll
        for (int w = 0; w < W; ++w) ac[w] = as[w] = 0.f;
        const int n0 = s * chunk, n1 = min(N, (s + 1) * chunk);
        constexpr int E = 3 + W;  // staged per atom: r, x
        for (int base = n0; base < n1; base += FM_STAGE) {
            const int count = min(FM_STAGE, n1 - base);
            for (int idx = threadIdx.x; idx < count * E; idx += FM_THREADS) {
                const int e = idx / E, col = idx % E;
                smem[idx] = col < 3 ? r[(base + e) * 3 + col] : x[(size_t)(base + e) * W + col - 3];
            }
            __syncthreads();
            for (int e = warp; e < count; e += FM_WARPS) {
                const float* st = smem + e * E;
                float sn, cs;
                sincosf(dot3(kx, ky, kz, st[0], st[1], st[2]), &sn, &cs);
#pragma unroll
                for (int w = 0; w < W; ++w) {
                    ac[w] = fmaf(cs, st[3 + w], ac[w]);
                    as[w] = fmaf(sn, st[3 + w], as[w]);
                }
            }
            __syncthreads();
        }
        float* mine = smem + (warp * FM_ROWS + lane) * 2 * W;
#pragma unroll
        for (int w = 0; w < W; ++w) {
            mine[w] = ac[w];
            mine[W + w] = as[w];
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < FM_ROWS * 2 * W; idx += FM_THREADS) {
            const int l = idx / (2 * W), col = idx % (2 * W);
            float v = 0.f;
#pragma unroll
            for (int q = 0; q < FM_WARPS; ++q) v += smem[(q * FM_ROWS + l) * 2 * W + col];
            const int rr = tile * FM_ROWS + l;
            if (rr < K) part[((size_t)s * K + rr) * 2 * W + col] = v;
        }
        __syncthreads();
    }
}

// part[s, n, :] = sum over the chunk of k-vectors of cos(theta) a[k, :] + sin(theta) b[k, :] (a, b written in this
// launch: through L2). rows: the atoms; chunks s of `chunk` k-vectors.
template <int W>
__device__ void nsum(const float* __restrict__ k, const float* __restrict__ r, const float* a, const float* b,
                     float* __restrict__ part, int K, int N, int splits, int chunk, float* smem) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tiles = (N + FM_ROWS - 1) / FM_ROWS;
    for (int item = blockIdx.x; item < tiles * splits; item += gridDim.x) {
        const int tile = item % tiles, s = item / tiles;
        const int atom = tile * FM_ROWS + lane;
        float rx = 0.f, ry = 0.f, rz = 0.f;
        if (atom < N) {
            rx = r[atom * 3];
            ry = r[atom * 3 + 1];
            rz = r[atom * 3 + 2];
        }
        float acc[W];
#pragma unroll
        for (int w = 0; w < W; ++w) acc[w] = 0.f;
        const int k0 = s * chunk, k1 = min(K, (s + 1) * chunk);
        constexpr int E = 3 + 2 * W;  // staged per k-vector: k, a, b
        for (int base = k0; base < k1; base += FM_STAGE) {
            const int count = min(FM_STAGE, k1 - base);
            for (int idx = threadIdx.x; idx < count * E; idx += FM_THREADS) {
                const int e = idx / E, col = idx % E, kk = base + e;
                if (col < 3) smem[idx] = k[kk * 3 + col];
                else if (col < 3 + W) smem[idx] = __ldcg(a + (size_t)kk * W + col - 3);
                else smem[idx] = __ldcg(b + (size_t)kk * W + col - 3 - W);
            }
            __syncthreads();
            for (int e = warp; e < count; e += FM_WARPS) {
                const float* st = smem + e * E;
                float sn, cs;
                sincosf(dot3(st[0], st[1], st[2], rx, ry, rz), &sn, &cs);
#pragma unroll
                for (int w = 0; w < W; ++w) acc[w] = fmaf(cs, st[3 + w], fmaf(sn, st[3 + W + w], acc[w]));
            }
            __syncthreads();
        }
        float* mine = smem + (warp * FM_ROWS + lane) * W;
#pragma unroll
        for (int w = 0; w < W; ++w) mine[w] = acc[w];
        __syncthreads();
        for (int idx = threadIdx.x; idx < FM_ROWS * W; idx += FM_THREADS) {
            const int l = idx / W, col = idx % W;
            float v = 0.f;
#pragma unroll
            for (int q = 0; q < FM_WARPS; ++q) v += smem[(q * FM_ROWS + l) * W + col];
            const int nn = tile * FM_ROWS + l;
            if (nn < N) part[((size_t)s * N + nn) * W + col] = v;
        }
        __syncthreads();
    }
}

// The phase gradient pass: per atom over a chunk of k-vectors,
//   dtheta = cos (x . dys + g . b) - sin (x . dyc + g . a)      (g already scaled by s0; the g terms with HAS_G)
//   dx[n] += cos dyc + sin dys,  dL/dr[n] += dtheta k,  dL/dk[k] += dtheta r (per atom tile, GK)
// part[s, n, 0:CJ] = dx, part[s, n, CJ:CJ+3] = dL/dr; gk_part[tile, k, 3] (each (tile, k) written once).
template <int CJ, int CF, bool HAS_G, bool GK>
__device__ void phase_backward(const float* __restrict__ k, const float* __restrict__ r, const float* __restrict__ x,
                               const float* __restrict__ g, const float* dyc, const float* dys, const float* a,
                               const float* b, float* __restrict__ part, float* __restrict__ gk_part, int K, int N,
                               int splits, int chunk, float s0, float* smem) {
    constexpr int WP = CJ + 3;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tiles = (N + FM_ROWS - 1) / FM_ROWS;
    for (int item = blockIdx.x; item < tiles * splits; item += gridDim.x) {
        const int tile = item % tiles, s = item / tiles;
        const int atom = tile * FM_ROWS + lane;
        const bool valid = atom < N;
        const float rx = valid ? r[atom * 3] : 0.f, ry = valid ? r[atom * 3 + 1] : 0.f;
        const float rz = valid ? r[atom * 3 + 2] : 0.f;
        float xn[CJ], dx[CJ];
#pragma unroll
        for (int j = 0; j < CJ; ++j) {
            xn[j] = valid ? x[(size_t)atom * CJ + j] : 0.f;
            dx[j] = 0.f;
        }
        float gn[HAS_G ? CF : 1];
        if (HAS_G) {
#pragma unroll
            for (int f = 0; f < CF; ++f) gn[f] = valid ? s0 * g[(size_t)atom * CF + f] : 0.f;
        }
        float grx = 0.f, gry = 0.f, grz = 0.f;
        const int k0 = s * chunk, k1 = min(K, (s + 1) * chunk);
        constexpr int E = 3 + 2 * CJ + (HAS_G ? 2 * CF : 0);  // staged per k-vector: k, dyc, dys (, a, b)
        static_assert(E <= FM_STAGE_WIDTH, "the staged entry is wider than FM_STAGE_WIDTH");
        float* gks = smem + FM_GK_OFFSET;
        for (int base = k0; base < k1; base += FM_STAGE) {
            const int count = min(FM_STAGE, k1 - base);
            for (int idx = threadIdx.x; idx < count * E; idx += FM_THREADS) {
                const int e = idx / E, col = idx % E, kk = base + e;
                float v;
                if (col < 3) v = k[kk * 3 + col];
                else if (col < 3 + CJ) v = __ldcg(dyc + (size_t)kk * CJ + col - 3);
                else if (col < 3 + 2 * CJ) v = __ldcg(dys + (size_t)kk * CJ + col - 3 - CJ);
                else if (col < 3 + 2 * CJ + CF) v = __ldcg(a + (size_t)kk * CF + col - 3 - 2 * CJ);
                else v = __ldcg(b + (size_t)kk * CF + col - 3 - 2 * CJ - CF);
                smem[idx] = v;
            }
            __syncthreads();
            for (int e = warp; e < count; e += FM_WARPS) {
                const float* st = smem + e * E;
                const float kx = st[0], ky = st[1], kz = st[2];
                float sn, cs;
                sincosf(dot3(kx, ky, kz, rx, ry, rz), &sn, &cs);
                float p = 0.f, q = 0.f;
#pragma unroll
                for (int j = 0; j < CJ; ++j) {
                    const float vc = st[3 + j], vs = st[3 + CJ + j];
                    p = fmaf(xn[j], vs, p);
                    q = fmaf(xn[j], vc, q);
                    dx[j] = fmaf(cs, vc, fmaf(sn, vs, dx[j]));
                }
                if (HAS_G) {
#pragma unroll
                    for (int f = 0; f < CF; ++f) {
                        p = fmaf(gn[f], st[3 + 2 * CJ + CF + f], p);
                        q = fmaf(gn[f], st[3 + 2 * CJ + f], q);
                    }
                }
                const float dth = fmaf(cs, p, -sn * q);
                grx = fmaf(dth, kx, grx);
                gry = fmaf(dth, ky, gry);
                grz = fmaf(dth, kz, grz);
                if (GK) {
                    float* row = gks + e * 3 * 33 + lane;
                    row[0] = dth * rx;
                    row[33] = dth * ry;
                    row[66] = dth * rz;
                }
            }
            __syncthreads();
            if (GK) {
                for (int idx = threadIdx.x; idx < count * 3; idx += FM_THREADS) {
                    const float* row = gks + idx * 33;
                    float v = 0.f;
#pragma unroll
                    for (int l = 0; l < 32; ++l) v += row[l];
                    gk_part[((size_t)tile * K + base) * 3 + idx] = v;
                }
                __syncthreads();
            }
        }
        float* mine = smem + (warp * FM_ROWS + lane) * WP;
#pragma unroll
        for (int j = 0; j < CJ; ++j) mine[j] = dx[j];
        mine[CJ] = grx;
        mine[CJ + 1] = gry;
        mine[CJ + 2] = grz;
        __syncthreads();
        for (int idx = threadIdx.x; idx < FM_ROWS * WP; idx += FM_THREADS) {
            const int l = idx / WP, col = idx % WP;
            float v = 0.f;
#pragma unroll
            for (int q2 = 0; q2 < FM_WARPS; ++q2) v += smem[(q2 * FM_ROWS + l) * WP + col];
            const int nn = tile * FM_ROWS + l;
            if (nn < N) part[((size_t)s * N + nn) * WP + col] = v;
        }
        __syncthreads();
    }
}

// dx [N, CJ], dL/dr [N, 3] from the phase pass's partials (chunk order); dL/dk [K, 3] from the atom tiles' (tile
// order).
template <int CJ, bool GK>
__device__ void finish_backward(const float* part, const float* gk_part, float* __restrict__ dx,
                                float* __restrict__ gr, float* __restrict__ gk, int K, int N, int splits) {
    constexpr int WP = CJ + 3;
    const int stride = gridDim.x * FM_THREADS;
    for (int idx = blockIdx.x * FM_THREADS + threadIdx.x; idx < N * WP; idx += stride) {
        const int n = idx / WP, col = idx % WP;
        float v = 0.f;
        for (int s = 0; s < splits; ++s) v += __ldcg(part + (size_t)s * N * WP + idx);
        if (col < CJ) dx[(size_t)n * CJ + col] = v;
        else gr[n * 3 + col - CJ] = v;
    }
    if (GK) {
        const int tiles = (N + FM_ROWS - 1) / FM_ROWS;
        for (int idx = blockIdx.x * FM_THREADS + threadIdx.x; idx < K * 3; idx += stride) {
            float v = 0.f;
            for (int t = 0; t < tiles; ++t) v += __ldcg(gk_part + (size_t)t * K * 3 + idx);
            gk[idx] = v;
        }
    }
}

// the widest per-item reduction (2 x 16 columns in ksum, 16 + 3 in the backward) or a staged step with its dL/dk rows
#define FM_SMEM_MAX(a, b) ((a) > (b) ? (a) : (b))
__shared__ float fm_smem[FM_SMEM_MAX(FM_WARPS * FM_ROWS * 35, FM_GK_OFFSET + FM_STAGE * 3 * 33)];

template <int C, int J, int F>
__device__ void field_forward(const float* k, const float* r, const float* mask, const float* x, const float* dr,
                              const float* di, const float* w, const float* pr, const float* pi, float* out, float* yc,
                              float* ys, float* a, float* b, float* part1, float* part2, unsigned long long* counter,
                              int K, int N, int splits1, int chunk1, int splits2, int chunk2, float s0) {
    constexpr int CJ = C * J, CF = C * F;
    ksum<CJ>(k, r, x, part1, K, N, splits1, chunk1, fm_smem);
    grid_barrier(counter);
    const int stride = gridDim.x * FM_THREADS;
    for (int kk = blockIdx.x * FM_THREADS + threadIdx.x; kk < K; kk += stride) {
        const float keep = mask[kk], wk = w[kk];
        float vc[CJ], vs[CJ];
#pragma unroll
        for (int col = 0; col < CJ; ++col) {
            float c0 = 0.f, s1 = 0.f;
            for (int s = 0; s < splits1; ++s) {
                c0 += __ldcg(part1 + ((size_t)s * K + kk) * 2 * CJ + col);
                s1 += __ldcg(part1 + ((size_t)s * K + kk) * 2 * CJ + CJ + col);
            }
            vc[col] = keep * c0;
            vs[col] = keep * s1;
            yc[(size_t)kk * CJ + col] = vc[col];
            ys[(size_t)kk * CJ + col] = vs[col];
        }
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float ur = 0.f, ui = 0.f;
#pragma unroll
            for (int j = 0; j < J; ++j) {
                const float drj = dr[kk * J + j], dij = di[kk * J + j];
                ur += drj * vc[c * J + j] + dij * vs[c * J + j];
                ui += dij * vc[c * J + j] - drj * vs[c * J + j];
            }
            ur *= wk;
            ui *= wk;
#pragma unroll
            for (int f = 0; f < F; ++f) {
                const float prf = pr[kk * F + f], pif = pi[kk * F + f];
                a[(size_t)kk * CF + c * F + f] = ur * prf + ui * pif;
                b[(size_t)kk * CF + c * F + f] = ur * pif - ui * prf;
            }
        }
    }
    grid_barrier(counter);
    nsum<CF>(k, r, a, b, part2, K, N, splits2, chunk2, fm_smem);
    grid_barrier(counter);
    for (int idx = blockIdx.x * FM_THREADS + threadIdx.x; idx < N * CF; idx += stride) {
        float v = 0.f;
        for (int s = 0; s < splits2; ++s) v += __ldcg(part2 + (size_t)s * N * CF + idx);
        out[idx] = s0 * v;
    }
}

template <int C, int J, int F, bool FACTORS, bool GK>
__device__ void field_backward(const float* k, const float* r, const float* mask, const float* x, const float* g,
                               const float* yc, const float* ys, const float* a, const float* b, const float* dr,
                               const float* di, const float* w, const float* pr, const float* pi, float* dyc,
                               float* dys, float* part1, float* part2, float* gk_part, float* gk, float* gr, float* dx,
                               float* ddr, float* ddi, float* dw, float* dpr, float* dpi, unsigned long long* counter,
                               int K, int N, int splits1, int chunk1, int splits2, int chunk2, float s0) {
    constexpr int CJ = C * J, CF = C * F;
    ksum<CF>(k, r, g, part1, K, N, splits1, chunk1, fm_smem);
    grid_barrier(counter);
    const int stride = gridDim.x * FM_THREADS;
    for (int kk = blockIdx.x * FM_THREADS + threadIdx.x; kk < K; kk += stride) {
        const float keep = mask[kk], wk = w[kk];
        float da[CF], db[CF];
#pragma unroll
        for (int col = 0; col < CF; ++col) {
            float c0 = 0.f, s1 = 0.f;
            for (int s = 0; s < splits1; ++s) {
                c0 += __ldcg(part1 + ((size_t)s * K + kk) * 2 * CF + col);
                s1 += __ldcg(part1 + ((size_t)s * K + kk) * 2 * CF + CF + col);
            }
            da[col] = s0 * c0;
            db[col] = s0 * s1;
        }
        float dwk = 0.f;
        float gpr[FACTORS ? F : 1], gpi[FACTORS ? F : 1], gdr[FACTORS ? J : 1], gdi[FACTORS ? J : 1];
        if (FACTORS) {
#pragma unroll
            for (int f = 0; f < F; ++f) gpr[f] = gpi[f] = 0.f;
#pragma unroll
            for (int j = 0; j < J; ++j) gdr[j] = gdi[j] = 0.f;
        }
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float ur = 0.f, ui = 0.f, drr = 0.f, dri = 0.f;
#pragma unroll
            for (int j = 0; j < J; ++j) {
                const float drj = dr[kk * J + j], dij = di[kk * J + j];
                const float ycj = yc[(size_t)kk * CJ + c * J + j], ysj = ys[(size_t)kk * CJ + c * J + j];
                ur += drj * ycj + dij * ysj;
                ui += dij * ycj - drj * ysj;
            }
#pragma unroll
            for (int f = 0; f < F; ++f) {
                const float prf = pr[kk * F + f], pif = pi[kk * F + f];
                drr += da[c * F + f] * prf + db[c * F + f] * pif;
                dri += da[c * F + f] * pif - db[c * F + f] * prf;
            }
            dwk += drr * ur + dri * ui;
            const float wr = wk * drr, wi = wk * dri;
#pragma unroll
            for (int j = 0; j < J; ++j) {
                const float drj = dr[kk * J + j], dij = di[kk * J + j];
                dyc[(size_t)kk * CJ + c * J + j] = keep * (wr * drj + wi * dij);
                dys[(size_t)kk * CJ + c * J + j] = keep * (wr * dij - wi * drj);
            }
            if (FACTORS) {
                const float rr = wk * ur, ri = wk * ui;
#pragma unroll
                for (int f = 0; f < F; ++f) {
                    gpr[f] += da[c * F + f] * rr - db[c * F + f] * ri;
                    gpi[f] += da[c * F + f] * ri + db[c * F + f] * rr;
                }
#pragma unroll
                for (int j = 0; j < J; ++j) {
                    const float ycj = yc[(size_t)kk * CJ + c * J + j], ysj = ys[(size_t)kk * CJ + c * J + j];
                    gdr[j] += wr * ycj - wi * ysj;
                    gdi[j] += wr * ysj + wi * ycj;
                }
            }
        }
        if (FACTORS) {
            dw[kk] = dwk;
#pragma unroll
            for (int f = 0; f < F; ++f) {
                dpr[kk * F + f] = gpr[f];
                dpi[kk * F + f] = gpi[f];
            }
#pragma unroll
            for (int j = 0; j < J; ++j) {
                ddr[kk * J + j] = gdr[j];
                ddi[kk * J + j] = gdi[j];
            }
        }
    }
    grid_barrier(counter);
    phase_backward<CJ, CF, true, GK>(k, r, x, g, dyc, dys, a, b, part2, gk_part, K, N, splits2, chunk2, s0, fm_smem);
    grid_barrier(counter);
    finish_backward<CJ, GK>(part2, gk_part, dx, gr, gk, K, N, splits2);
}

template <int W>
__device__ void sf_forward(const float* k, const float* r, const float* mask, const float* x, float* yc, float* ys,
                           float* part1, unsigned long long* counter, int K, int N, int splits1, int chunk1) {
    ksum<W>(k, r, x, part1, K, N, splits1, chunk1, fm_smem);
    grid_barrier(counter);
    const int stride = gridDim.x * FM_THREADS;
    for (int idx = blockIdx.x * FM_THREADS + threadIdx.x; idx < K * W; idx += stride) {
        const int kk = idx / W, col = idx % W;
        float c0 = 0.f, s1 = 0.f;
        for (int s = 0; s < splits1; ++s) {
            c0 += __ldcg(part1 + ((size_t)s * K + kk) * 2 * W + col);
            s1 += __ldcg(part1 + ((size_t)s * K + kk) * 2 * W + W + col);
        }
        const float keep = mask[kk];
        yc[idx] = keep * c0;
        ys[idx] = keep * s1;
    }
}

template <int W, bool GK>
__device__ void sf_backward(const float* k, const float* r, const float* x, const float* dyc, const float* dys,
                            float* dx, float* gr, float* gk, float* part2, float* gk_part, unsigned long long* counter,
                            int K, int N, int splits2, int chunk2) {
    phase_backward<W, 1, false, GK>(k, r, x, nullptr, dyc, dys, nullptr, nullptr, part2, gk_part, K, N, splits2,
                                    chunk2, 1.f, fm_smem);
    grid_barrier(counter);
    finish_backward<W, GK>(part2, gk_part, dx, gr, gk, K, N, splits2);
}

}  // namespace

#define FM_FIELD_ARGS_F                                                                                              \
    const float *k, const float *r, const float *mask, const float *x, const float *dr, const float *di,            \
        const float *w, const float *pr, const float *pi, float *out, float *yc, float *ys, float *a, float *b,      \
        float *part1, float *part2, unsigned long long *counter, int K, int N, int splits1, int chunk1, int splits2, \
        int chunk2, float s0
#define FM_FIELD_ARGS_B                                                                                              \
    const float *k, const float *r, const float *mask, const float *x, const float *g, const float *yc,             \
        const float *ys, const float *a, const float *b, const float *dr, const float *di, const float *w,           \
        const float *pr, const float *pi, float *dyc, float *dys, float *part1, float *part2, float *gk_part,        \
        float *gk, float *gr, float *dx, float *ddr, float *ddi, float *dw, float *dpr, float *dpi,                  \
        unsigned long long *counter, int K, int N, int splits1, int chunk1, int splits2, int chunk2, float s0

#define FM_FIELD(C, J, F)                                                                                            \
    extern "C" __global__ void __launch_bounds__(FM_THREADS) fm_kspace_field_forward_##C##_##J##_##F(              \
        FM_FIELD_ARGS_F) {                                                                                           \
        field_forward<C, J, F>(k, r, mask, x, dr, di, w, pr, pi, out, yc, ys, a, b, part1, part2, counter, K, N,    \
                               splits1, chunk1, splits2, chunk2, s0);                                                \
    }                                                                                                                \
    FM_FIELD_BACKWARD(C, J, F, 0, 0, false, false)                                                                   \
    FM_FIELD_BACKWARD(C, J, F, 0, 1, false, true)                                                                    \
    FM_FIELD_BACKWARD(C, J, F, 1, 0, true, false)                                                                    \
    FM_FIELD_BACKWARD(C, J, F, 1, 1, true, true)

#define FM_FIELD_BACKWARD(C, J, F, FA, GKN, FACTORS, GK)                                                             \
    extern "C" __global__ void __launch_bounds__(FM_THREADS) fm_kspace_field_backward_##C##_##J##_##F##_##FA##GKN( \
        FM_FIELD_ARGS_B) {                                                                                           \
        field_backward<C, J, F, FACTORS, GK>(k, r, mask, x, g, yc, ys, a, b, dr, di, w, pr, pi, dyc, dys, part1,    \
                                             part2, gk_part, gk, gr, dx, ddr, ddi, dw, dpr, dpi, counter, K, N,     \
                                             splits1, chunk1, splits2, chunk2, s0);                                  \
    }

#define FM_SF(W)                                                                                                     \
    extern "C" __global__ void __launch_bounds__(FM_THREADS) fm_kspace_sf_forward_##W(                              \
        const float *k, const float *r, const float *mask, const float *x, float *yc, float *ys, float *part1,      \
        unsigned long long *counter, int K, int N, int splits1, int chunk1) {                                        \
        sf_forward<W>(k, r, mask, x, yc, ys, part1, counter, K, N, splits1, chunk1);                                 \
    }                                                                                                                \
    extern "C" __global__ void __launch_bounds__(FM_THREADS) fm_kspace_sf_backward_##W##_0(                         \
        const float *k, const float *r, const float *x, const float *dyc, const float *dys, float *dx, float *gr,   \
        float *gk, float *part2, float *gk_part, unsigned long long *counter, int K, int N, int splits2,             \
        int chunk2) {                                                                                                \
        sf_backward<W, false>(k, r, x, dyc, dys, dx, gr, gk, part2, gk_part, counter, K, N, splits2, chunk2);        \
    }                                                                                                                \
    extern "C" __global__ void __launch_bounds__(FM_THREADS) fm_kspace_sf_backward_##W##_1(                         \
        const float *k, const float *r, const float *x, const float *dyc, const float *dys, float *dx, float *gr,   \
        float *gk, float *part2, float *gk_part, unsigned long long *counter, int K, int N, int splits2,             \
        int chunk2) {                                                                                                \
        sf_backward<W, true>(k, r, x, dyc, dys, dx, gr, gk, part2, gk_part, counter, K, N, splits2, chunk2);         \
    }

// POLAR-1: J = 4 density components (sigma 1 x m 4), F = 8 feature components (sigma 2 x m 4); one channel, or
// both spins with SPIN_BATCHED. The Coulomb energy's structure factor has sigma x m = 4 columns.
FM_FIELD(1, 4, 8)
FM_FIELD(2, 4, 8)
FM_SF(1)
FM_SF(2)
FM_SF(4)
FM_SF(8)
FM_SF(16)
