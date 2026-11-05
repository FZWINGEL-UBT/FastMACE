"""Producer/consumer warp-specialisation microbenchmark for the A100 tensor-core forward (round 5, 5 November 2026).

Question (outside proposal, plans/ideas-20261105-round2.md, third set, idea 1): can MMA warps that project a onto W
(3xTF32 mma.sync, as fused_forward's tensor-core product) overlap FP32 warps that contract the result (the tensor
product's FMAs), inside one CTA? The A100 stage bounds at 6c2f2967 (kernel lab, thermalised SrTiO3 10x10x10): the
projection 1.71 ms and the contraction 1.26 ms of a 3.25 ms forward, adding up, i.e. no overlap now.

This is a synthetic CTA, not the convolution: the projection is fused_forward's MMA loop (npl path groups, two
16-channel m-tiles, KP inputs, batches of 8 edges, mask split, its redistribution tile r_s), the contraction a block
of independent FMA chains (F per path group and edge, 16 accumulators) on a gathered x, like the TP's arithmetic.
12 warps, one CTA per SM (160 KB of shared memory; the forward has 151 KB).

Modes, each with the same total work (NBATCH batches per CTA):
  P       serial warps, projection only (the contraction skipped)
  C       serial warps, contraction only (w read from a fixed tile)
  serial  12 warps, each projects and then contracts its batch (today's structure)
  pcS     6 producer warps project into a ring of S tiles per pair, 6 consumer warps contract (mbarrier handoff)

Stop rule (the proposal's): drop the idea unless pc beats serial by more than 10 % of serial, with no spills.

Usage: python tools/pc_microbench.py [--fmas 64 128 256] [--stages 2 3] [--output DIR]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

SOURCE = r"""
typedef unsigned long long u64;
constexpr int NPL = {npl}, KP = {kp}, B = 8, F = {fmas}, NX = {nx}, S = {stages}, NBATCH = {nbatch};
constexpr int WARPS = 12, PAIRS = WARPS / 2, MODE = {mode}, DOP = {dop}, DOC = {doc};
constexpr int WS = 40, RS = 36, TILE = NPL * B * RS;
constexpr int NTILES = MODE == 1 ? PAIRS * S : WARPS;

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
__device__ __forceinline__ unsigned smem(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void bar_init(u64* bar, unsigned count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(smem(bar)), "r"(count));
}
__device__ __forceinline__ void bar_arrive(u64* bar) {
    asm volatile("{ .reg .b64 st; mbarrier.arrive.shared.b64 st, [%0]; }" :: "r"(smem(bar)) : "memory");
}
__device__ __forceinline__ void bar_wait(u64* bar, unsigned parity) {
    unsigned done = 0;
    while (!done)
        asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                     : "=r"(done) : "r"(smem(bar)), "r"(parity) : "memory");
}

__device__ __forceinline__ long first_edge(int batch, int nedges) {
    return ((long)(blockIdx.x * NBATCH + batch) * B) % (long)(nedges - B);
}

__device__ __forceinline__ void produce(int batch, float* tile, float* a_s, const float* __restrict__ A,
                                        const float* Ws, int nedges, int lane) {
    const long e0 = first_edge(batch, nedges);
    for (int i = lane; i < KP * B; i += 32) {
        const int k = i / B, b = i - k * B;
        a_s[k * B + b] = A[(e0 + b) * KP + k];
    }
    __syncwarp();
    float wacc[NPL][2][4];
    #pragma unroll
    for (int g = 0; g < NPL; ++g)
        #pragma unroll
        for (int m = 0; m < 2; ++m)
            #pragma unroll
            for (int i = 0; i < 4; ++i) wacc[g][m][i] = 0.f;
    const int gid = lane >> 2, tq = lane & 3;
    #pragma unroll 2
    for (int k0 = 0; k0 < KP; k0 += 8) {
        unsigned bh[2], bl[2];
        split_tf32(a_s[(k0 + tq) * B + gid], bh[0], bl[0]);
        split_tf32(a_s[(k0 + tq + 4) * B + gid], bh[1], bl[1]);
        #pragma unroll
        for (int g = 0; g < NPL; ++g)
            #pragma unroll
            for (int m = 0; m < 2; ++m) {
                const float* wp = Ws + (g * KP + k0 + tq) * WS + m * 16 + gid;
                unsigned ah[4], al[4];
                split_tf32(wp[0], ah[0], al[0]);
                split_tf32(wp[8], ah[1], al[1]);
                split_tf32(wp[4 * WS], ah[2], al[2]);
                split_tf32(wp[4 * WS + 8], ah[3], al[3]);
                mma_tf32(wacc[g][m], al, bh);
                mma_tf32(wacc[g][m], ah, bl);
                mma_tf32(wacc[g][m], ah, bh);
            }
    }
    #pragma unroll
    for (int g = 0; g < NPL; ++g)
        #pragma unroll
        for (int m = 0; m < 2; ++m) {
            float* rp = tile + (g * B + 2 * tq) * RS + m * 16 + gid;
            rp[0] = wacc[g][m][0];
            rp[RS] = wacc[g][m][1];
            rp[8] = wacc[g][m][2];
            rp[RS + 8] = wacc[g][m][3];
        }
}

__device__ __forceinline__ void consume(int batch, const float* tile, const float* __restrict__ X,
                                        const int* __restrict__ snd, int nedges, int lane, float (&acc)[16]) {
    const long e0 = first_edge(batch, nedges);
    #pragma unroll 1
    for (int b = 0; b < B; ++b) {
        const long s = snd[e0 + b];
        float xr[NX];
        #pragma unroll
        for (int j = 0; j < NX; ++j) xr[j] = __ldg(X + (s * NX + j) * 32 + lane);
        #pragma unroll
        for (int g = 0; g < NPL; ++g) {
            const float wv = tile[(g * B + b) * RS + lane];
            #pragma unroll
            for (int j = 0; j < F; ++j) acc[j & 15] = fmaf(wv, xr[j % NX], acc[j & 15]);
        }
    }
}

extern "C" __global__ void __launch_bounds__(WARPS * 32, 1)
pc_bench(const float* __restrict__ A, const float* __restrict__ Wg, const float* __restrict__ X,
         const int* __restrict__ snd, int nedges, float* __restrict__ out) {
    extern __shared__ __align__(16) float sm[];
    u64* bars = reinterpret_cast<u64*>(sm);  // full[PAIRS][S], empty[PAIRS][S]
    float* Ws = sm + 4 * PAIRS * S;
    float* a_all = Ws + NPL * KP * WS;
    float* ring = a_all + (MODE == 1 ? PAIRS : WARPS) * KP * B;  // pc: a staging for the producers only
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < NPL * KP * 32; i += blockDim.x) Ws[(i >> 5) * WS + (i & 31)] = Wg[i];
    for (int i = threadIdx.x; i < NTILES * TILE; i += blockDim.x) ring[i] = 1e-3f * (float)(i & 255);
    if (MODE == 1 && threadIdx.x < 2 * PAIRS * S) bar_init(bars + threadIdx.x, 32);
    __syncthreads();
    float acc[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) acc[j] = 0.f;
    float* a_s = a_all + warp * KP * B;
    if (MODE == 0) {
        float* tile = ring + warp * TILE;
        for (int batch = warp; batch < NBATCH; batch += WARPS) {
            if (DOP) produce(batch, tile, a_s, A, Ws, nedges, lane);
            __syncwarp();
            if (DOC) consume(batch, tile, X, snd, nedges, lane, acc);
            __syncwarp();
        }
        if (!DOC) acc[0] += tile[lane];
    } else {
        const int pair = warp % PAIRS;
        u64* full = bars + pair * S;
        u64* empty = bars + PAIRS * S + pair * S;
        const int count = NBATCH / PAIRS;
        if (warp < PAIRS) {
            for (int n = 0; n < count; ++n) {
                const int s = n % S;
                if (n >= S) bar_wait(empty + s, ((n / S) - 1) & 1);
                produce(pair + n * PAIRS, ring + (pair * S + s) * TILE, a_s, A, Ws, nedges, lane);
                __syncwarp();
                bar_arrive(full + s);
            }
        } else {
            for (int n = 0; n < count; ++n) {
                const int s = n % S;
                bar_wait(full + s, (n / S) & 1);
                consume(pair + n * PAIRS, ring + (pair * S + s) * TILE, X, snd, nedges, lane, acc);
                __syncwarp();
                bar_arrive(empty + s);
            }
        }
    }
    float t = 0.f;
    #pragma unroll
    for (int j = 0; j < 16; ++j) t += acc[j];
    out[blockIdx.x * blockDim.x + threadIdx.x] = t;
}
"""

SHARED_BYTES = 160 * 1024  # one CTA per SM, as the forward (151 KB); the A100 allows 163 KB


def build(params, shared_bytes):
    import torch
    from cuda.bindings import driver

    from fastmace.kernels.oeq_streaming.runtime import _check, compile_cubin
    source = SOURCE.replace("{", "{{").replace("}", "}}")
    for key in params:
        source = source.replace("{{" + key + "}}", "{" + key + "}")
    source = source.format(**params)
    cubin = compile_cubin(source, torch.cuda.get_device_capability(), ())
    module = _check(driver.cuModuleLoadData(cubin))
    function = _check(driver.cuModuleGetFunction(module, b"pc_bench"))
    attr = driver.CUfunction_attribute
    _check(driver.cuFuncSetAttribute(function, attr.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_bytes))
    regs = _check(driver.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_NUM_REGS, function))
    local = _check(driver.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function))
    return module, function, int(regs), int(local)


def shared_floats(p):
    pairs = 6
    tiles = pairs * p["stages"] if p["mode"] == 1 else 12
    staging = (pairs if p["mode"] == 1 else 12) * p["kp"] * 8
    return 4 * pairs * p["stages"] + p["npl"] * p["kp"] * 40 + staging + tiles * p["npl"] * 8 * 36


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npl", type=int, default=5)
    ap.add_argument("--kp", type=int, default=72)  # OMAT-0 medium's folded radial input (K = 65, KP = 72)
    ap.add_argument("--nx", type=int, default=4)
    ap.add_argument("--fmas", type=int, nargs="+", default=[32, 64, 96, 128, 192])
    ap.add_argument("--stages", type=int, nargs="+", default=[2, 3])  # 3 fits only with fewer path groups
    ap.add_argument("--nbatch", type=int, default=3456)  # about the forward's batches per CTA (46,720 x 8 / 108)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(argv)

    import torch
    from cuda.bindings import driver

    from fastmace.kernels.oeq_streaming.runtime import _check
    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    shared_bytes = min(SHARED_BYTES, int(props.shared_memory_per_block_optin))
    gpu = torch.cuda.get_device_name(0)
    nedges, nnodes = 392448, 5000
    g = torch.Generator(device="cuda").manual_seed(0)
    A = torch.randn(nedges, args.kp, device="cuda", generator=g)
    W = torch.randn(args.npl * args.kp * 32, device="cuda", generator=g) / args.kp ** 0.5
    X = torch.randn(nnodes * args.nx * 32, device="cuda", generator=g)
    snd = torch.randint(0, nnodes, (nedges,), device="cuda", dtype=torch.int32, generator=g)
    out = torch.empty(sms * 384, device="cuda")
    print(f"{gpu}, {sms} SMs; npl {args.npl}, KP {args.kp}, NX {args.nx}, {args.nbatch} batches per CTA", flush=True)

    def timed(params):
        if 4 * shared_floats(params) > shared_bytes:
            return dict(error="shared memory")
        module, function, regs, local = build(params, shared_bytes)
        values = (A.data_ptr(), W.data_ptr(), X.data_ptr(), snd.data_ptr(), nedges, out.data_ptr())
        types = (ctypes.c_void_p,) * 4 + (ctypes.c_int, ctypes.c_void_p)
        stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)

        def launch():
            _check(driver.cuLaunchKernel(function, sms, 1, 1, 384, 1, 1, shared_bytes, stream, (values, types), 0))
        for _ in range(3):
            launch()
        torch.cuda.synchronize()
        times = []
        for _ in range(args.rounds):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            launch()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)
        _check(driver.cuModuleUnload(module))
        times.sort()
        return dict(us=times[len(times) // 2], regs=regs, local=local, checksum=float(out.double().sum()))

    rows = []
    for fmas in args.fmas:
        base = dict(npl=args.npl, kp=args.kp, nx=args.nx, fmas=fmas, nbatch=args.nbatch, stages=2)
        r = {"fmas": fmas}
        r["P"] = timed(dict(base, mode=0, dop=1, doc=0))
        r["C"] = timed(dict(base, mode=0, dop=0, doc=1))
        r["serial"] = timed(dict(base, mode=0, dop=1, doc=1))
        for s in args.stages:
            r[f"pc{s}"] = timed(dict(base, stages=s, mode=1, dop=1, doc=1))
        ser = r["serial"]["us"]
        line = (f"F {fmas:4d}: P {r['P']['us']:8.1f}  C {r['C']['us']:8.1f}  (C/P {r['C']['us'] / r['P']['us']:.2f})"
                f"  serial {ser:8.1f} (P+C {r['P']['us'] + r['C']['us']:8.1f}, regs {r['serial']['regs']} "
                f"local {r['serial']['local']})")
        for s in args.stages:
            pc = r[f"pc{s}"]
            line += (f"  pc{s} {pc.get('us', float('nan')):8.1f} = {pc.get('us', float('nan')) / ser:.3f} of serial"
                     f" (regs {pc.get('regs')} local {pc.get('local')})")
        print(line, flush=True)
        rows.append(r)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "pc_microbench.json").write_text(json.dumps(dict(gpu=gpu, args=vars(args) | {"output": None},
                                                                          rows=rows), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
