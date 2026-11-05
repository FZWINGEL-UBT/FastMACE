"""Hopper wgmma 3xTF32 GEMM against the CUTLASS mma.sync 3xTF32 kernel (``fastmace.kernels.cutlass.tf32x3``) on
the node-level GEMM shapes of OMAT-0 on Benchmark_loup (``tools/bench/gemm_shapes.py``): ``C = A @ B``, A row-major
[M, K], B row-major [K, N] ("rr") or column-major ("rc"). Round 5, 5 November 2026 (owner: "What about the backward
pass can we use wgmma there too"): the tf32x3 GEMMs take about 0.85 ms of the H200's 5.75 ms step.

The wgmma kernel: a block of BM / 64 warpgroups computes a BM x BN tile; per BK slice of K the block stages A
(row-major, padded rows) and B's TF32 high and low planes (8-row x 16-byte core matrices, K-major, no swizzle) in
shared memory; each warp's A fragments are split in registers; per k step three m64nBNk8 tf32 wgmmas (a_lo b_hi,
a_hi b_lo, a_hi b_hi; the mask split). ``--stages 2`` double-buffers the shared tiles: the next slice's global loads
are issued before the current slice's wgmmas and stored after them.

    python tools/bench/wgmma_gemm_bench.py --output wgmma_gemm.json
"""

import argparse
import ctypes
import json
import sys
from pathlib import Path

SHAPES = [  # (M, K, N, layout): the Benchmark_loup tf32x3 products (gemm_shapes.py, H200 plan)
    (5000, 128, 128, "rr"), (5000, 128, 384, "rr"), (5000, 384, 128, "rc"), (5000, 128, 256, "rr"),
    (5000, 128, 192, "rc"), (5000, 192, 128, "rr"), (5000, 256, 128, "rr"),
    (15000, 128, 128, "rr"), (15000, 128, 384, "rr"), (15000, 384, 128, "rc"), (15000, 384, 128, "rr"),
    (15000, 128, 192, "rc"), (15000, 192, 128, "rr"),
    (25000, 128, 384, "rr"), (25000, 384, 128, "rc"), (25000, 384, 128, "rr"), (25000, 128, 192, "rc"),
    (25000, 192, 128, "rr"),
    (35000, 128, 384, "rr"), (35000, 128, 256, "rr"), (35000, 384, 128, "rc"), (35000, 256, 128, "rr"),
    (35000, 128, 192, "rc"), (35000, 192, 128, "rr"),
]


def wgmma_fn(n):
    regs = n // 2
    d = ", ".join(f"%{i}" for i in range(regs))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(regs))
    return f"""__device__ __forceinline__ void wgmma_n{n}(float* d, const unsigned (&a)[4], unsigned long long desc,
                                                int scale) {{
    asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{regs + 5}, 0;\\n"
                 "wgmma.mma_async.sync.aligned.m64n{n}k8.f32.tf32.tf32 {{{d}}}, {{%{regs}, %{regs + 1}, %{regs + 2}, "
                 "%{regs + 3}}}, %{regs + 4}, p, 1, 1;\\n}}\\n"
                 : {outs}
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale) : "memory");
}}
"""


def source(bm, bn, bk, layout, stages):
    """CUDA source of ``wg_gemm`` for one tile (bm x bn, K slices of bk) and B layout."""
    col = layout == "rc"
    threads = bm * 2  # bm / 64 warpgroups of 128 threads
    a_floats = bm * (bk + 4)
    b_floats = bn * bk
    stage = a_floats + 2 * b_floats
    if col:
        bload = f"""        #pragma unroll
        for (int r_ = 0; r_ < ({bn} * ({bk} / 4) + {threads} - 1) / {threads}; ++r_) {{
            const int i = tid + r_ * {threads};
            if (i < {bn} * ({bk} / 4)) {{
                const int n = i / ({bk} / 4), q = i - n * ({bk} / 4), gn = n0 + n, gk = k0 + 4 * q;
                breg[r_] = (gn < N && gk < K) ? __ldg(reinterpret_cast<const float4*>(B + (size_t)gn * ldb + gk))
                                                      : make_float4(0.f, 0.f, 0.f, 0.f);
            }}
        }}"""
        bstore = f"""        #pragma unroll
        for (int r_ = 0; r_ < ({bn} * ({bk} / 4) + {threads} - 1) / {threads}; ++r_) {{
            const int i = tid + r_ * {threads};
            if (i < {bn} * ({bk} / 4)) {{
                const int n = i / ({bk} / 4), q = i - n * ({bk} / 4);
                const float4 v = breg[r_];
                unsigned h0, l0, h1, l1, h2, l2, h3, l3;
                split_tf32(v.x, h0, l0); split_tf32(v.y, h1, l1); split_tf32(v.z, h2, l2); split_tf32(v.w, h3, l3);
                const int o = ((n >> 3) * ({bk} / 4) + q) * 32 + (n & 7) * 4;
                *reinterpret_cast<uint4*>(bh_s + o) = make_uint4(h0, h1, h2, h3);
                *reinterpret_cast<uint4*>(bl_s + o) = make_uint4(l0, l1, l2, l3);
            }}
        }}"""
    else:
        bload = f"""        #pragma unroll
        for (int r_ = 0; r_ < ({bk} * ({bn} / 4) + {threads} - 1) / {threads}; ++r_) {{
            const int i = tid + r_ * {threads};
            if (i < {bk} * ({bn} / 4)) {{
                const int k = i / ({bn} / 4), q = i - k * ({bn} / 4), gk = k0 + k, gn = n0 + 4 * q;
                breg[r_] = (gk < K && gn < N) ? __ldg(reinterpret_cast<const float4*>(B + (size_t)gk * ldb + gn))
                                                      : make_float4(0.f, 0.f, 0.f, 0.f);
            }}
        }}"""
        bstore = f"""        #pragma unroll
        for (int r_ = 0; r_ < ({bk} * ({bn} / 4) + {threads} - 1) / {threads}; ++r_) {{
            const int i = tid + r_ * {threads};
            if (i < {bk} * ({bn} / 4)) {{
                const int k = i / ({bn} / 4), q = i - k * ({bn} / 4);
                const float4 v = breg[r_];
                const float vals[4] = {{v.x, v.y, v.z, v.w}};
                #pragma unroll
                for (int c = 0; c < 4; ++c) {{
                    const int n = 4 * q + c;
                    unsigned h, l;
                    split_tf32(vals[c], h, l);
                    const int o = ((n >> 3) * ({bk} / 4) + (k >> 2)) * 32 + (n & 7) * 4 + (k & 3);
                    bh_s[o] = h;
                    bl_s[o] = l;
                }}
            }}
        }}"""
    aload = f"""        #pragma unroll
        for (int r_ = 0; r_ < ({bm} * ({bk} / 4) + {threads} - 1) / {threads}; ++r_) {{
            const int i = tid + r_ * {threads};
            if (i < {bm} * ({bk} / 4)) {{
                const int r = i / ({bk} / 4), q = i - r * ({bk} / 4), gm = m0 + r, gk = k0 + 4 * q;
                areg[r_] = (gm < M && gk < K) ? __ldg(reinterpret_cast<const float4*>(A + (size_t)gm * lda + gk))
                                                      : make_float4(0.f, 0.f, 0.f, 0.f);
            }}
        }}"""
    astore = f"""        #pragma unroll
        for (int r_ = 0; r_ < ({bm} * ({bk} / 4) + {threads} - 1) / {threads}; ++r_) {{
            const int i = tid + r_ * {threads};
            if (i < {bm} * ({bk} / 4)) {{
                const int r = i / ({bk} / 4), q = i - r * ({bk} / 4);
                *reinterpret_cast<float4*>(a_s + r * {bk + 4} + 4 * q) = areg[r_];
            }}
        }}"""
    nareg = -(-bm * bk // 4 // threads)
    nbreg = -(-bn * bk // 4 // threads)
    setbuf = f"""        float* a_s = smem + buf * {stage};
        unsigned* bh_s = reinterpret_cast<unsigned*>(a_s + {a_floats});
        unsigned* bl_s = bh_s + {b_floats};"""
    compute = f"""        {{
            const unsigned bh = (unsigned)__cvta_generic_to_shared(bh_s), bl = bh + {b_floats} * 4;
            unsigned ah[{bk // 8}][4], al[{bk // 8}][4];
            const float* ar = a_s + (wg * 64 + wi * 16 + gid) * {bk + 4} + tq;
            #pragma unroll
            for (int ks = 0; ks < {bk // 8}; ++ks) {{
                split_tf32(ar[8 * ks], ah[ks][0], al[ks][0]);
                split_tf32(ar[8 * ks + 8 * {bk + 4}], ah[ks][1], al[ks][1]);
                split_tf32(ar[8 * ks + 4], ah[ks][2], al[ks][2]);
                split_tf32(ar[8 * ks + 8 * {bk + 4} + 4], ah[ks][3], al[ks][3]);
            }}
            asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
            #pragma unroll
            for (int ks = 0; ks < {bk // 8}; ++ks) {{
                wgmma_n{bn}(d, al[ks], desc(bh + 256 * ks), first == 0 ? (ks > 0) : 1);
                wgmma_n{bn}(d, ah[ks], desc(bl + 256 * ks), 1);
                wgmma_n{bn}(d, ah[ks], desc(bh + 256 * ks), 1);
            }}
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        }}"""
    if stages == 1:
        loop = f"""    for (int k0 = 0, first = 0; k0 < K; k0 += {bk}, first = 1) {{
        const int buf = 0;
{setbuf}
{aload}
{bload}
{astore}
{bstore}
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
{compute}
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        __syncthreads();
    }}"""
    else:
        loop = f"""    {{
        int k0 = 0, buf = 0;
{setbuf}
{aload}
{bload}
{astore}
{bstore}
    }}
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    for (int k0 = 0, buf = 0, first = 0; k0 < K; k0 += {bk}, buf ^= 1, first = 1) {{
        {{
{setbuf}
{compute}
        }}
        const bool next = k0 + {bk} < K;
        if (next) {{
            const int k0n = k0 + {bk};
            {{
                const int k0 = k0n;
{aload}
{bload}
            }}
            {{
                const int buf2 = buf ^ 1;
                float* a_s = smem + buf2 * {stage};
                unsigned* bh_s = reinterpret_cast<unsigned*>(a_s + {a_floats});
                unsigned* bl_s = bh_s + {b_floats};
{astore}
{bstore}
            }}
        }}
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
    }}"""
    return f"""// wg_gemm: 3xTF32 C = A @ B on Hopper wgmma (tools/bench/wgmma_gemm_bench.py)
__device__ __forceinline__ void split_tf32(float x, unsigned& hi, unsigned& lo) {{
    hi = __float_as_uint(x) & 0xffffe000u;
    lo = __float_as_uint(x - __uint_as_float(hi));
}}
__device__ __forceinline__ unsigned long long desc(unsigned addr) {{
    return (unsigned long long)((addr & 0x3FFFFu) >> 4) | ((unsigned long long)(128 >> 4) << 16)
        | ((unsigned long long)(({bk} * 32) >> 4) << 32);
}}
{wgmma_fn(bn)}
extern "C" __global__ void __launch_bounds__({threads}) wg_gemm(const float* __restrict__ A, const float* __restrict__ B,
                                                         float* __restrict__ C, int M, int N, int K, int lda, int ldb,
                                                         int ldc) {{
    extern __shared__ __align__(128) float smem[];
    const int tid = threadIdx.x, wg = tid >> 7, wi = (tid >> 5) & 3, lane = tid & 31, gid = lane >> 2, tq = lane & 3;
    const int m0 = blockIdx.x * {bm}, n0 = blockIdx.y * {bn};
    float4 areg[{nareg}], breg[{nbreg}];
    float d[{bn // 2}];
    #pragma unroll
    for (int i = 0; i < {bn // 2}; ++i) d[i] = 0.f;
{loop}
    const int row = m0 + wg * 64 + wi * 16 + gid;
    #pragma unroll
    for (int j = 0; j < {bn // 8}; ++j) {{
        const int col = n0 + 8 * j + 2 * tq;
        if (col < N) {{
            if (row < M) *reinterpret_cast<float2*>(C + (size_t)row * ldc + col) = make_float2(d[4 * j], d[4 * j + 1]);
            if (row + 8 < M)
                *reinterpret_cast<float2*>(C + (size_t)(row + 8) * ldc + col) = make_float2(d[4 * j + 2], d[4 * j + 3]);
        }}
    }}
}}
""", threads, stages * stage * 4


class Kernel:
    def __init__(self, bm, bn, bk, layout, stages):
        import torch
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check, compile_cubin
        self.src, self.threads, self.shared = source(bm, bn, bk, layout, stages)
        self.bm, self.bn = bm, bn
        cubin = compile_cubin(self.src, torch.cuda.get_device_capability(), suffix="a")
        torch.cuda.current_stream()
        self.module = _check(driver.cuModuleLoadData(cubin))
        self.function = _check(driver.cuModuleGetFunction(self.module, b"wg_gemm"))
        attribute = driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        _check(driver.cuFuncSetAttribute(self.function, attribute, self.shared))
        self.regs = _check(driver.cuFuncGetAttribute(driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
                                                     self.function))

    def __call__(self, a, b, out):
        import torch
        from cuda.bindings import driver

        from fastmace.kernels.oeq_streaming.runtime import _check
        m, k = a.shape
        n = b.shape[1]
        ldb = b.stride(1) if b.stride(0) == 1 else b.stride(0)
        p = ctypes.c_void_p
        values = (a.data_ptr(), b.data_ptr(), out.data_ptr(), m, n, k, a.stride(0), ldb, out.stride(0))
        types = (p, p, p) + (ctypes.c_int,) * 6
        stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
        _check(driver.cuLaunchKernel(self.function, -(-m // self.bm), -(-n // self.bn), 1, self.threads, 1, 1,
                                     self.shared, stream, (values, types), 0))


def timed(fn, reps=30):
    import torch
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) * 1000)
    times.sort()
    return times[len(times) // 2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles", nargs="*", default=["64x128x32x1", "64x128x32x2", "128x128x32x1", "128x128x32x2",
                                                         "64x64x32x2", "128x64x32x2", "64x192x32x2", "128x192x32x2"],
                        help="BMxBNxBKxSTAGES")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    import fastmace  # noqa: F401
    import torch

    from fastmace.kernels.cutlass import tf32x3

    torch.manual_seed(0)
    kernels = {}
    rows = []
    for m, k, n, layout in SHAPES:
        a = torch.randn(m, k, device="cuda")
        b = torch.randn(k, n, device="cuda") if layout == "rr" else torch.randn(n, k, device="cuda").t()
        exact = a.double() @ b.double()
        scale = a.double().abs() @ b.double().abs()
        ref = torch.empty(m, n, device="cuda")
        ca, cb = tf32x3.fit(a, b)
        cut_us = timed(lambda: tf32x3.mm_into(ca, cb, ref))
        cut_err = float(((ref.double() - exact).abs() / scale).max())
        row = dict(m=m, k=k, n=n, layout=layout, cutlass_us=round(cut_us, 2), cutlass_err=cut_err, wgmma={})
        for tile in args.tiles:
            bm, bn, bk, st = (int(x) for x in tile.split("x"))
            key = (bm, bn, bk, layout, st)
            try:
                if key not in kernels:
                    kernels[key] = Kernel(bm, bn, bk, layout, st)
                kern = kernels[key]
                out = torch.empty(m, n, device="cuda")
                us = timed(lambda: kern(a, b, out))
                err = float(((out.double() - exact).abs() / scale).max())
                row["wgmma"][tile] = dict(us=round(us, 2), err=err, regs=kern.regs, ratio=round(us / cut_us, 3))
            except Exception as error:  # noqa: BLE001
                row["wgmma"][tile] = dict(error=repr(error)[:300])
        good = [(v["us"], t) for t, v in row["wgmma"].items() if "us" in v and v["err"] < 4 * cut_err + 1e-6]
        if good:
            best = min(good)
            row["best"] = dict(tile=best[1], us=best[0], ratio=round(best[0] / cut_us, 3))
        else:
            row["best"] = dict(tile=None, us=float("nan"), ratio=float("nan"))
        rows.append(row)
        print(json.dumps({x: row[x] for x in ("m", "k", "n", "layout", "cutlass_us", "cutlass_err", "best")}), flush=True)
        if not good:
            print(json.dumps({t: {x: v.get(x) for x in ("us", "err", "error")} for t, v in row["wgmma"].items()}),
                  flush=True)
    total_cut = sum(r["cutlass_us"] for r in rows)
    total_best = sum(r["best"]["us"] for r in rows if r["best"]["tile"])
    per_tile = {t: sum(r["wgmma"][t].get("us", float("inf")) for r in rows) for t in args.tiles}
    summary = dict(gpu=torch.cuda.get_device_name(), total_cutlass_us=round(total_cut, 1),
                   total_best_us=round(total_best, 1), per_tile_total_us={t: round(v, 1) for t, v in per_tile.items()})
    print(json.dumps(summary), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
