"""Idea 1, stage 2 (11 Nov 2026): offline sm_80 census of a two-far-edges-per-half2 w' channel walk.

Takes the channel loop of today's A100 w' kernel (the captured ``wprime-real.cu``: FACTORED, HORNER, SCALAR_PATHS,
BUTTERFLY, STAGE_EDGES) and wraps it twice in the same minimal harness (one warp per edge unit, the dout row and
the harmonics in shared memory as in the real kernel):

  fp32   the body unchanged: one edge per warp.
  half2  the same body with every channel value an ``__half2`` holding two far edges of one receiver: x of the two
         senders packed from a float16 copy of x (one PRMT per value), dout broadcast in a staged __half2 row (one
         LDS as before), the harmonics as __half2 pairs in shared memory, the table from a float16 copy (two 8-byte
         loads and four PRMTs per path), Horner, u/v, the weight and edge gradients in HFMA2; the slope masked per
         half instead of the ``sloped`` branch; dx by two float atomics per value (two senders).

Per edge: fp32 channel-loop instructions against half2's / 2. The conductor's rule: stop if the half2 walk does not
cut issue slots per edge, or if it spills. Writes half2_wprime/*.cu and half2_wprime_census.json beside this file;
run with the NVRTC env (WSL ``mace_cueq_oeq``).
"""
import json
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "bench"))
from cu_census import census  # noqa: E402

SOURCE = HERE / "screen1" / "ghoist" / "sources" / "wprime-real.cu"

HEAD = """#include <cuda_fp16.h>
typedef int int32_t; typedef long long int64_t;
constexpr int INPUT_DIM = 512, EDGE_DIM = 16, WEIGHT_DIM = 1280, CHANNEL_DIM = 128, TERMS = 4, PADDED_TERMS = 4;
constexpr int LOGICAL_GROUP_SIZE = 32;
typedef float scalar_t;
typedef float acc_t;
"""

FP32 = HEAD + """
extern "C" __global__ void __launch_bounds__(256) census_wprime(
    const float* x, float* dx, const float* __restrict__ gprime, const int32_t* senders, const int32_t* slabs,
    const float* ts, const float* scs, const unsigned* flags, const float* shs, const float* dout, float* out,
    int n_edges, int store) {
    __shared__ __align__(16) scalar_t stage_row[5120];
    __shared__ __align__(16) scalar_t stage_esh[8][EDGE_DIM];
    for (int i = threadIdx.x; i < 5120; i += blockDim.x) stage_row[i] = dout[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int e = blockIdx.x * 8 + w; e < n_edges; e += gridDim.x * 8) {
    if (lane < EDGE_DIM) stage_esh[w][lane] = shs[int64_t(e) * EDGE_DIM + lane];
    __syncwarp();
    const int64_t input_base = int64_t(senders[e]) * INPUT_DIM;
    const scalar_t* sh_edge = stage_esh[w];
    // an interior, sloped edge (the common path): the loop is then one basic block, as half2's
    const bool sloped = true, inside = true;
    const float t = ts[e];
    const scalar_t sc = scs[e];
    const scalar_t* g = gprime + int64_t(slabs[e]) * PADDED_TERMS * WEIGHT_DIM;
    acc_t edge_gradient[EDGE_DIM] = {acc_t(0)};
    acc_t radial_gradient = acc_t(0);
    const int64_t weight_base = int64_t(e) * WEIGHT_DIM;  // unused by the body's arithmetic
@BODY@
    if (store) {  // per-lane values for the numbers check; off in the timing (the real kernel stores 17 per edge)
    #pragma unroll
    for (int i = 0; i < EDGE_DIM; ++i) out[(int64_t(e) * 32 + lane) * 17 + i] = edge_gradient[i];
    out[(int64_t(e) * 32 + lane) * 17 + 16] = radial_gradient;
    }
    __syncwarp();
    }
}
"""

HALF2 = HEAD + """
#define Z2 __half2(__half2_raw{0, 0})
#define H2C(bits) __half2(__half2_raw{bits, bits})
#define LX(k) __halves2half2(xh[input_base0 + (k)], xh[input_base1 + (k)])
extern "C" __global__ void __launch_bounds__(256) census_wprime(
    const __half* xh, float* dx, const __half* __restrict__ gh, const int32_t* senders, const int32_t* slabs,
    const float* ts, const float* scs, const unsigned* flags, const __half2* shs, const __half2* dout, float* out,
    int n_pairs, int store) {
    __shared__ __align__(16) __half2 stage_row2[5120];
    __shared__ __align__(16) __half2 stage_esh2[8][EDGE_DIM];
    for (int i = threadIdx.x; i < 5120; i += blockDim.x) stage_row2[i] = dout[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int p = blockIdx.x * 8 + w; p < n_pairs; p += gridDim.x * 8) {
    const int e0 = 2 * p, e1 = 2 * p + 1;
    if (lane < EDGE_DIM) stage_esh2[w][lane] = shs[int64_t(p) * EDGE_DIM + lane];
    __syncwarp();
    const int64_t input_base0 = int64_t(senders[e0]) * INPUT_DIM, input_base1 = int64_t(senders[e1]) * INPUT_DIM;
    const __half2* sh_edge2 = stage_esh2[w];
    // an edge outside the table reads a zero slab; the slope is masked per half (no branch on two edges)
    const __half* g0 = gh + int64_t(slabs[e0]) * PADDED_TERMS * WEIGHT_DIM;
    const __half* g1 = gh + int64_t(slabs[e1]) * PADDED_TERMS * WEIGHT_DIM;
    const __half2 t2 = __floats2half2_rn(ts[e0], ts[e1]);
    const __half2 sc2 = __floats2half2_rn((flags[e0] & 2u) ? scs[e0] : 0.f, (flags[e1] & 2u) ? scs[e1] : 0.f);
    __half2 edge_gradient[EDGE_DIM];
    #pragma unroll
    for (int i = 0; i < EDGE_DIM; ++i) edge_gradient[i] = Z2;
    __half2 radial_gradient = Z2;
    const int64_t weight_base = int64_t(e0) * WEIGHT_DIM;
@BODY@
    if (store) {
    #pragma unroll
    for (int i = 0; i < EDGE_DIM; ++i) {
        out[(int64_t(e0) * 32 + lane) * 17 + i] = __low2float(edge_gradient[i]);
        out[(int64_t(e1) * 32 + lane) * 17 + i] = __high2float(edge_gradient[i]);
    }
    out[(int64_t(e0) * 32 + lane) * 17 + 16] = __low2float(radial_gradient);
    out[(int64_t(e1) * 32 + lane) * 17 + 16] = __high2float(radial_gradient);
    }
    __syncwarp();
    }
}
"""

GV_HALF = """__half2 gv[PADDED_TERMS];
                {
                    const uint2 a = __ldg(reinterpret_cast<const uint2*>(g0 + weight_index * PADDED_TERMS));
                    const uint2 b = __ldg(reinterpret_cast<const uint2*>(g1 + weight_index * PADDED_TERMS));
                    const unsigned q[4] = {__byte_perm(a.x, b.x, 0x5410), __byte_perm(a.x, b.x, 0x7632),
                                           __byte_perm(a.y, b.y, 0x5410), __byte_perm(a.y, b.y, 0x7632)};
                    #pragma unroll
                    for (int k = 0; k < PADDED_TERMS; ++k) gv[k] = *reinterpret_cast<const __half2*>(&q[k]);
                }
                __half2 radial_weight = gv[TERMS - 1];"""


def body(src):
    start = src.index("    acc_t radial_gradient = acc_t(0);\n") + len("    acc_t radial_gradient = acc_t(0);\n")
    end = src.index("    // BUTTERFLY")
    return src[start:end]


def half_bits(c):
    import numpy as np
    return int(np.array([c], dtype=np.float16).view(np.uint16)[0])


def to_half2(b):
    b = re.sub(r"static_cast<scalar_t>\(([-0-9a-fx.p+]+)f\)",
               lambda m: f"H2C({half_bits(float.fromhex(m.group(1))):#06x})", b)
    b = re.sub(r"scalar_t gv\[PADDED_TERMS\];.*?scalar_t radial_weight = gv\[TERMS - 1\];", GV_HALF, b, flags=re.S)
    b = re.sub(r"atomicAdd\(\s*dx \+ input_base \+ (\d+) \+ channel,\s*scalar_t\((\w+)\)\);",
               r"atomicAdd(dx + input_base0 + \1 + channel, __low2float(\2));\n"
               r"            atomicAdd(dx + input_base1 + \1 + channel, __high2float(\2));", b)
    b = re.sub(r"x\[input_base\s*\+\s*(\d+) \+ channel\]", r"LX(\1 + channel)", b)
    b = b.replace("stage_row[", "stage_row2[").replace("sh_edge[", "sh_edge2[")
    b = b.replace("acc_t(0)", "Z2").replace("scalar_t(0)", "Z2")
    b = b.replace("* t +", "* t2 +").replace("radial_dt * sc;", "radial_dt * sc2;")
    b = b.replace("if (sloped) {", "{")
    b = re.sub(r"\b(scalar_t|acc_t)\b", "__half2", b)
    for bad in ("x[", "inside", "sloped", "static_cast", "float4", "input_base +"):
        assert bad not in b, bad
    return b


def main():
    src = SOURCE.read_text()
    b = body(src)
    out = HERE / "half2_wprime"
    out.mkdir(exist_ok=True)
    files = {"fp32": out / "wprime_fp32.cu", "half2": out / "wprime_half2.cu"}
    files["fp32"].write_text(FP32.replace("@BODY@", b), newline="\n")
    files["half2"].write_text(HALF2.replace("@BODY@", to_half2(b)), newline="\n")
    rows = {}
    for arch in ("80", "90"):
        for name, f in files.items():
            r = census(f, arch)
            r["file"] = f.name
            rows[f"{name}_sm{arch}"] = r
            print(name, arch, {k: v for k, v in r.items() if k != "longest_block_ops"}, flush=True)
            print("   loop", r["longest_block_ops"], flush=True)
    summary = {}
    for arch in ("80", "90"):
        a, h = rows[f"fp32_sm{arch}"], rows[f"half2_sm{arch}"]
        summary[f"sm{arch}"] = dict(fp32_loop_per_edge=a["longest_block"], half2_loop_per_edge=h["longest_block"] / 2,
                                     ratio=round(h["longest_block"] / 2 / a["longest_block"], 3),
                                     fp32_registers=a["registers"], half2_registers=h["registers"],
                                     half2_local=h["local"])
    print(json.dumps(summary, indent=1))
    (HERE / "half2_wprime_census.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=1) + "\n",
                                                    newline="\n")


if __name__ == "__main__":
    main()
