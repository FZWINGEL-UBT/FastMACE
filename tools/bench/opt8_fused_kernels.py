# ruff: noqa: E501  (generated CUDA lines)
"""OPT8-T01 prototype: fused folded-layer + streaming convolution kernels (plain FP32 SIMT), probe only.

Generated from the vendored streaming schedule (``fastmace.kernels.oeq_streaming.schedule``) for any uvu problem
with ``edge_mul == 1`` in the ``ir_mul`` layout. ``a [E, K]`` is the folded layer's input and ``W [Q, K]`` its
weight (``Q = paths × channels``); the per-edge weights ``w = a Wᵀ`` and their gradient are never stored.

Work split: a block owns one slice = (``CB`` channels, ``blockIdx.y``) × (one part of the paths, ``blockIdx.z``)
and keeps that slice of ``W`` in shared memory (``paths_in_part × K × CB`` floats; MH-1 with CB = 32 and two parts:
41.6 KB), read from L2 once per block. One warp owns one receiver at a time (lane = channel of the slice) and walks
its edges in batches:
- forward: the batch's ``a`` rows (transposed, so four edges' values are one float4 broadcast) and ``sh`` go to
  warp-private shared memory, the sender features to registers; per group of paths, ``w[g][b]`` is built by a
  k-loop that reads each ``W`` value once for the whole batch, then the angular terms accumulate into the
  receiver's outputs (registers), written once per receiver (outputs of different slices are disjoint);
- backward: the same ``w`` rebuild for the part's paths; per edge the weight gradient ``dw`` (registers), the
  sender-gradient atomics (as in the streaming backward) and the edge gradient (transpose-reduced over the lanes,
  one atomic add per component and slice); then ``da = dw W`` per k-block: lane partials of a batch's edges are
  transpose-reduced over the lanes and added atomically (one add per element and slice).
"""

from __future__ import annotations

import math

import torch


def lit(v):
    return f"{float(v).hex()}f"


def transpose_reduce(var, V, indent):
    """Code: sum ``var[0..V-1]`` (V a power of two <= 32) over the warp's lanes; afterwards ``var[0]`` of lane
    ``l`` holds the sum of index ``l >> (5 - log2 V)``."""
    steps = int(math.log2(V))
    assert 1 << steps == V and V <= 32
    out, n, off = [], V // 2, 16
    for _ in range(steps):
        out.append(f"{indent}{{ const bool up = (lane & {off}) != 0;")
        out.append(f"{indent}  #pragma unroll\n{indent}  for (int i = 0; i < {n}; ++i) {{")
        out.append(f"{indent}    const float snd = up ? {var}[i] : {var}[i + {n}];")
        out.append(f"{indent}    const float kp = up ? {var}[i + {n}] : {var}[i];")
        out.append(f"{indent}    {var}[i] = kp + __shfl_xor_sync(FULL, snd, {off}); }} }}")
        n //= 2
        off //= 2
    while off >= 1:
        out.append(f"{indent}{var}[0] += __shfl_xor_sync(FULL, {var}[0], {off});")
        off //= 2
    return "\n".join(out)


class FusedConv:
    def __init__(self, schedule, layout, W, cfg, module_cls=None, sms=None):
        if module_cls is None:
            from opt8_fused_conv import Module as module_cls
        if layout != "ir_mul":
            raise ValueError("prototype: ir_mul only")
        self.s, self.cfg = schedule, cfg
        C = schedule.channels
        self.C, self.K = C, int(W.shape[1])
        self.CB = int(cfg.get("cb", 32))
        self.NPARTS = int(cfg.get("parts", 2))
        self.B = int(cfg.get("b", 8))
        self.PG = int(cfg.get("pg", 5))
        self.NW = int(cfg.get("nw", 16))
        self.GX = int(cfg.get("gx", 0))
        self.BW = int(cfg.get("bw", 16))
        self.BB = int(cfg.get("bb", 4))
        self.KB = int(cfg.get("kb", 32 // self.BB))
        self.WS = self.CB + int(cfg.get("pad", 1))  # Ws row stride: CB + 1 makes both lane patterns conflict-free
        self.DA = cfg.get("da", "bcast")  # bcast: dw broadcast through shared memory, lanes = k; tr: transpose-reduce
        self.ws_floats = None  # set once the parts are known
        # span: warps take chunks of RC receivers (dynamic, one counter per slice) and their edge batches run across
        # receiver boundaries (no padding per receiver); 0: one receiver per warp, batches padded per receiver
        self.SPAN = int(cfg.get("span", 1))
        self.RC = int(cfg.get("rc", 4))
        # OPT8-T01b: the forward also stores the w it builds (wout [E, Q]) for today's chunked backward
        self.STORE = int(cfg.get("store_w", 0))
        assert self.CB == 32 and C % self.CB == 0 and self.B % 4 == 0 and self.BB % 4 == 0
        paths = schedule.paths
        assert all(p.edge_mul == 1 for p in paths)
        self.NP = len(paths)
        assert int(W.shape[0]) == self.NP * C, (W.shape, self.NP, C)
        self.W = W.contiguous()
        per = -(-self.NP // self.NPARTS)
        if int(cfg.get("balance", 1)):
            # equal path counts (equal W slices and GEMM work); angular terms balanced greedily, heaviest first
            nparts = -(-self.NP // per)
            cost = [sum(len(o.terms) for o in p.outputs) for p in paths]
            parts, load = [[] for _ in range(nparts)], [0] * nparts
            for pi in sorted(range(self.NP), key=lambda i: (-cost[i], i)):
                q = min((q for q in range(nparts) if len(parts[q]) < per), key=lambda q: (load[q], q))
                parts[q].append(pi)
                load[q] += cost[pi]
            self.parts = [sorted(q) for q in parts]
        else:
            self.parts = [list(range(i, min(self.NP, i + per))) for i in range(0, self.NP, per)]
        self.NPARTS = len(self.parts)
        self.NPP = max(len(q) for q in self.parts)
        self.ws_floats = -(-self.NPP * self.K * self.WS // 4) * 4  # keeps the warp scratch 16-byte aligned
        self.slot_off = {sl.name: sl.start + sl.component * C for sl in schedule.output_slots}
        owners = {}
        for q, part in enumerate(self.parts):  # plain stores: every output slot must belong to one part
            for pi in part:
                for o in paths[pi].outputs:
                    assert owners.setdefault(o.accumulator.name, q) == q, "output slot shared across parts"
        self.src = self.forward_source() + "\n" + self.backward_source()
        ED = schedule.edge_dim
        self.smem_f = 4 * (self.ws_floats + self.NW * (self.K * self.B + self.B * ED + self.B * 32))
        self.smem_b = 4 * (self.ws_floats + self.BW * (self.K * self.BB + self.BB * ED + self.BB * 32))
        self.mod = module_cls(self.src, ["fused_forward", "fused_backward"],
                              extra_options=["--use_fast_math"] if cfg.get("fast") else [])
        self.mod.set_shared("fused_forward", self.smem_f)
        self.mod.set_shared("fused_backward", self.smem_b)
        self.sms = sms or torch.cuda.get_device_properties(W.device).multi_processor_count

    def describe(self):
        return dict(cb=self.CB, parts=self.NPARTS, b=self.B, pg=self.PG, nw=self.NW, gx=self.GX, bw=self.BW,
                    bb=self.BB, kb=self.KB, store_w=self.STORE, span=self.SPAN, rc=self.RC, pad=self.WS - self.CB, da=self.DA, fast=bool(self.cfg.get("fast")), smem_f=self.smem_f, smem_b=self.smem_b)

    # ------------------------------------------------------------------ pieces
    def head(self):
        s = self.s
        return f"""
using int64_t = long long;
constexpr int C = {self.C};
constexpr int K = {self.K};
constexpr int CB = {self.CB};
constexpr int64_t Q = {self.NP * self.C};
constexpr int WS = {self.WS};
constexpr int ED = {s.edge_dim};
constexpr int IN_DIM = {s.input_dim};
constexpr int OUT_DIM = {s.output_dim};
constexpr unsigned FULL = 0xffffffffu;
"""

    def xoffs(self, part):
        C, paths = self.C, self.s.paths
        return sorted({paths[pi].input_start + t.input_component * C for pi in part
                       for o in paths[pi].outputs for t in o.terms})

    def stage_w(self, part):
        ws = ", ".join(str(self.s.paths[pi].weight_start) for pi in part)
        n = len(part)
        return f"""
        {{
        const int wst[{n}] = {{{ws}}};
        for (int i = threadIdx.x; i < {n} * K * CB; i += blockDim.x) {{
            const int p = i / (K * CB), rem = i - p * (K * CB), k = rem / CB, l = rem - k * CB;
            Ws[(p * K + k) * WS + l] = W[(int64_t)(wst[p] + cb * CB + l) * K + k];
        }}
        }}
        __syncthreads();
"""

    def stage_batch(self, B, xo):
        nx = len(xo)
        xs = ", ".join(str(v) for v in xo)
        return f"""
            __syncwarp();
            for (int i = lane; i < {B} * K; i += 32) {{
                const int b = i / K, k = i - b * K;
                a_s[k * {B} + b] = (b < nb) ? a[(e0 + b) * K + k] : 0.f;
            }}
            for (int i = lane; i < {B} * ED; i += 32) {{
                const int b = i / ED;
                sh_s[i] = (b < nb) ? sh[e0 * ED + i] : 0.f;
            }}
            const int xoff[{nx}] = {{{xs}}};
            float xv[{B}][{nx}];
            int sv[{B}];
            int rv[{B}];
            #pragma unroll
            for (int b = 0; b < {B}; ++b) {{
                rv[b] = (b < nb) ? receivers[e0 + b] : 0x7fffffff;
                int snd = (b < nb) ? senders[e0 + b] : -1;
                if (snd < 0 || snd >= N) snd = -1;
                sv[b] = snd;
                #pragma unroll
                for (int i = 0; i < {nx}; ++i)
                    xv[b][i] = (snd >= 0) ? x[(int64_t)snd * IN_DIM + xoff[i] + ch] : 0.f;
            }}
            __syncwarp();
"""

    @staticmethod
    def wgemm(local, B, wname="w"):
        """w[g][b] = sum_k a_s[k][b] * Ws[local p][k][lane] for the local path indices ``local``."""
        n = len(local)
        lines = [f"            float {wname}[{n}][{B}];",
                 f"            #pragma unroll\n            for (int g = 0; g < {n}; ++g)"
                 f" for (int b = 0; b < {B}; ++b) {wname}[g][b] = 0.f;",
                 "            #pragma unroll 5\n            for (int k = 0; k < K; ++k) {"]
        for q in range(B // 4):
            lines.append(f"                const float4 av{q} = *reinterpret_cast<const float4*>(&a_s[k * {B} + {4 * q}]);")
        for g, lp in enumerate(local):
            lines.append(f"                {{ const float wk = Ws[({lp} * K + k) * WS + lane];")
            for q in range(B // 4):
                for c, comp in enumerate("xyzw"):
                    lines.append(f"                  {wname}[{g}][{4 * q + c}] ="
                                 f" fmaf(av{q}.{comp}, wk, {wname}[{g}][{4 * q + c}]);")
            lines.append("                }")
        lines.append("            }")
        return "\n".join(lines)

    # ------------------------------------------------------------------ forward

    def work_loop_open(self, counter_base):
        """Code opening the warp's loop over work units; defines r0, r1, ebeg, eend (span) or r, begin, end."""
        if not self.SPAN:
            return """
        for (int64_t r = (int64_t)blockIdx.x * nw + warp; r < N; r += (int64_t)gridDim.x * nw) {
            const int64_t r0 = r, r1 = r + 1;
"""
        return f"""
        int* ctr = counters + {counter_base} + blockIdx.y * gridDim.z + blockIdx.z;
        while (true) {{
            int cc = 0;
            if (lane == 0) cc = atomicAdd(ctr, 1);
            cc = __shfl_sync(FULL, cc, 0);
            const int64_t r0 = (int64_t)cc * {self.RC};
            if (r0 >= N) break;
            const int64_t r1 = r0 + {self.RC} < N ? r0 + {self.RC} : N;
"""

    @staticmethod
    def edge_range():
        return """
            int64_t ebeg = row_ptr[r0] < 0 ? 0 : row_ptr[r0];
            int64_t eend = row_ptr[r1] < ebeg ? ebeg : row_ptr[r1];
            eend = eend > E ? E : eend;
"""

    def forward_part(self, part):
        paths, B, C = self.s.paths, self.B, self.C
        xo = self.xoffs(part)
        xi = {v: i for i, v in enumerate(xo)}
        names = list(dict.fromkeys(o.accumulator.name for pi in part for o in paths[pi].outputs))
        si = {n: i for i, n in enumerate(names)}
        offs = ", ".join(str(self.slot_off[n]) for n in names)
        n = len(names)
        local = list(range(len(part)))
        tp = []
        for lp in local:
            p = paths[part[lp]]
            tp.append("                    {")
            for j in range(p.edge_irrep_dim):
                tp.append(f"                    const float e{j} = sh_s[b * ED + {p.edge_start + j}];")
            for o in p.outputs:
                terms = " + ".join(f"xv[b][{xi[p.input_start + t.input_component * C]}] * e{t.edge_component}"
                                   f" * {lit(t.coefficient)}" for t in o.terms) or "0.f"
                k = si[o.accumulator.name]
                tp.append(f"                    acc[{k}] = fmaf(w[{lp}][b], {terms}, acc[{k}]);")
            tp.append("                    }")
        flush = f"""while (cur < rv[b]) {{
                        #pragma unroll
                        for (int i = 0; i < {n}; ++i) {{ out[cur * OUT_DIM + slot_off[i] + ch] = acc[i]; acc[i] = 0.f; }}
                        ++cur;
                    }}"""
        return f"""
    {{ // part {part}
{self.stage_w(part)}
        const int slot_off[{n}] = {{{offs}}};
{self.work_loop_open(0)}
{self.edge_range()}
            float acc[{n}];
            #pragma unroll
            for (int i = 0; i < {n}; ++i) acc[i] = 0.f;
            int64_t cur = r0;
            for (int64_t e0 = ebeg; e0 < eend; e0 += {B}) {{
                const int nb = (int)((eend - e0) < {B} ? (eend - e0) : {B});
{self.stage_batch(B, xo)}
{self.wgemm(local, B)}
{self.store_w(part, B)}
                #pragma unroll
                for (int b = 0; b < {B}; ++b) {{
                    if (b >= nb) break;
                    {flush}
{chr(10).join(tp)}
                }}
            }}
            while (cur < r1) {{
                #pragma unroll
                for (int i = 0; i < {n}; ++i) {{ out[cur * OUT_DIM + slot_off[i] + ch] = acc[i]; acc[i] = 0.f; }}
                ++cur;
            }}
        }}
    }}
"""

    def store_w(self, part, B):
        if not self.STORE:
            return ""
        ws = ", ".join(str(self.s.paths[pi].weight_start) for pi in part)
        n = len(part)
        return f"""
                {{ const int wst[{n}] = {{{ws}}};
                #pragma unroll
                for (int b = 0; b < {B}; ++b) {{
                    if (b < nb) {{
                        #pragma unroll
                        for (int g = 0; g < {n}; ++g) wout[(e0 + b) * Q + wst[g] + ch] = w[g][b];
                    }}
                }} }}
"""

    def forward_source(self):
        B = self.B
        cases = "\n".join(f"    case {q}:{self.forward_part(part)}    break;" for q, part in enumerate(self.parts))
        return self.head() + f"""
extern "C" __global__ void __launch_bounds__({self.NW * 32}, 1) fused_forward(
    int64_t N, int64_t E, const float* __restrict__ a, const float* __restrict__ W,
    const float* __restrict__ x, const float* __restrict__ sh, const int* __restrict__ senders,
    const int* __restrict__ receivers, const int* __restrict__ row_ptr, int* __restrict__ counters,
    float* __restrict__ out, float* __restrict__ wout) {{
    extern __shared__ float smem[];
    float* Ws = smem;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    float* a_s = Ws + {self.ws_floats} + warp * (K * {B} + {B} * ED + {B} * 32);
    float* sh_s = a_s + K * {B};
    float* dw_s = sh_s + {B} * ED;
    const int cb = blockIdx.y;
    const int ch = cb * CB + lane;
    switch (blockIdx.z) {{
{cases}
    }}
}}
"""

    # ------------------------------------------------------------------ backward
    def backward_part(self, part):
        paths, B, KB, C = self.s.paths, self.BB, self.KB, self.C
        ED = self.s.edge_dim
        EDP = 1 << max(0, (ED - 1).bit_length())
        xo = self.xoffs(part)
        xi = {v: i for i, v in enumerate(xo)}
        nx = len(xo)
        names = list(dict.fromkeys(o.accumulator.name for pi in part for o in paths[pi].outputs))
        si = {n: i for i, n in enumerate(names)}
        offs = ", ".join(str(self.slot_off[n]) for n in names)
        local = list(range(len(part)))
        npl = len(local)
        p1 = self.wgemm(local, B)
        p2 = [f"            float dw[{npl}][{B}];",
              f"            #pragma unroll\n            for (int b = 0; b < {B}; ++b) {{",
              f"                float dxa[{nx}];\n                float dsha[{EDP}];",
              f"                #pragma unroll\n                for (int i = 0; i < {nx}; ++i) dxa[i] = 0.f;",
              f"                #pragma unroll\n                for (int j = 0; j < {EDP}; ++j) dsha[j] = 0.f;"]
        for lp in local:
            p = paths[part[lp]]
            p2.append(f"                {{ float dwv = 0.f; const float wv = w[{lp}][b];")
            for j in range(p.edge_irrep_dim):
                p2.append(f"                const float e{j} = sh_s[b * ED + {p.edge_start + j}];")
            for o in p.outputs:
                p2.append(f"                {{ const float gg = g[{si[o.accumulator.name]}]; const float gw = gg * wv;"
                          " float ang = 0.f;")
                for t in o.terms:
                    i = xi[p.input_start + t.input_component * C]
                    c, j = lit(t.coefficient), p.edge_start + t.edge_component
                    if self.cfg.get("tp", "v2") == "v1":
                        p2.append(f"                  dwv = fmaf(gg * xv[b][{i}], e{t.edge_component} * {c}, dwv);")
                        p2.append(f"                  dxa[{i}] = fmaf(gw, e{t.edge_component} * {c}, dxa[{i}]);")
                        p2.append(f"                  dsha[{j}] = fmaf(gw, xv[b][{i}] * {c}, dsha[{j}]);")
                    else:  # the forward's angular value once per output, then dw = g * ang (fewer multiplies)
                        p2.append(f"                  {{ const float ce = e{t.edge_component} * {c};"
                                  f" ang = fmaf(xv[b][{i}], ce, ang); dxa[{i}] = fmaf(gw, ce, dxa[{i}]);"
                                  f" dsha[{j}] = fmaf(gw * {c}, xv[b][{i}], dsha[{j}]); }}")
                if self.cfg.get("tp", "v2") != "v1":
                    p2.append("                  dwv = fmaf(gg, ang, dwv);")
                p2.append("                }")
            p2.append(f"                dw[{lp}][b] = dwv; }}")
        p2 += ["                if (sv[b] >= 0) {",
               f"                    #pragma unroll\n                    for (int i = 0; i < {nx}; ++i)"
               " atomicAdd(&dx[(int64_t)sv[b] * IN_DIM + xoff[i] + ch], dxa[i]);",
               "                }",
               transpose_reduce("dsha", EDP, "                ")]
        shift = 5 - int(math.log2(EDP))
        p2 += [f"                {{ const int j = lane >> {shift};",
               f"                  if (b < nb && j < ED && (lane & {(1 << shift) - 1}) == 0)"
               " atomicAdd(&dsh[(e0 + b) * ED + j], dsha[0]); }",
               "            }"]
        if self.DA == "tr":
            p3 = self.da_transpose(local, B, KB)
        else:
            p3 = self.da_broadcast(local, B)
        n = len(names)
        p2 = [ln.replace("            for (int b = 0; b < {B}; ++b) {{".format(B=B),
                         "            for (int b = 0; b < {B}; ++b) {{ if (b < nb) {{ if (rv[b] != cur) {{ cur = rv[b];"
                         " _Pragma(\"unroll\") for (int i = 0; i < {n}; ++i) g[i] = dout[(int64_t)cur * OUT_DIM + slot_off[i] + ch]; }} }}"
                         .format(B=B, n=n), 1) for ln in p2]
        return f"""
    {{ // part {part}
{self.stage_w(part)}
        const int slot_off[{n}] = {{{offs}}};
{self.work_loop_open(self.slices())}
{self.edge_range()}
            float g[{n}];
            #pragma unroll
            for (int i = 0; i < {n}; ++i) g[i] = 0.f;
            int64_t cur = -1;
            for (int64_t e0 = ebeg; e0 < eend; e0 += {B}) {{
                const int nb = (int)((eend - e0) < {B} ? (eend - e0) : {B});
{self.stage_batch(B, xo)}
{p1}
{chr(10).join(p2)}
{chr(10).join(p3)}
            }}
        }}
    }}
"""

    def da_transpose(self, local, B, KB):
        V = B * KB
        assert V <= 32 and V & (V - 1) == 0
        vshift = 5 - int(math.log2(V))
        p3 = [f"            #pragma unroll 1\n            for (int k0 = 0; k0 < K; k0 += {KB}) {{",
              f"                float part[{V}];",
              f"                #pragma unroll\n                for (int i = 0; i < {V}; ++i) part[i] = 0.f;",
              f"                #pragma unroll\n                for (int kk = 0; kk < {KB}; ++kk) {{",
              "                    const int k = k0 + kk;",
              "                    if (k < K) {"]
        for lp in local:
            p3.append(f"                        {{ const float wk = Ws[({lp} * K + k) * WS + lane];")
            for b in range(B):
                p3.append(f"                          part[{b * KB} + kk] = fmaf(dw[{lp}][{b}], wk, part[{b * KB} + kk]);")
            p3.append("                        }")
        p3 += ["                    }", "                }",
               transpose_reduce("part", V, "                "),
               f"                {{ const int idx = lane >> {vshift}; const int b = idx / {KB}, k = k0 + idx % {KB};",
               f"                  if ((lane & {(1 << vshift) - 1}) == 0 && b < nb && k < K)"
               " atomicAdd(&da[(e0 + b) * K + k], part[0]); }",
               "            }"]
        return p3

    def da_broadcast(self, local, B):
        """da[b][k] for k = lane + 32 j (lanes = k): dw of each path goes through warp-private shared memory and is
        read back as float4 broadcasts over 4 channels; W from the padded slice (conflict-free with lanes = k).
        The remainder columns K % 32 are lane-local partials, transpose-reduced at the end."""
        K = self.K
        J, R = K // 32, K % 32
        assert R * B <= 32 and (R == 0 or (R * B) & (R * B - 1) == 0), "prototype: remainder columns"
        p3 = [f"            float dak[{J}][{B}];",
              f"            #pragma unroll\n            for (int j = 0; j < {J}; ++j) for (int b = 0; b < {B}; ++b) dak[j][b] = 0.f;"]
        if R:
            p3.append(f"            float dar[{R * B}];")
            p3.append(f"            #pragma unroll\n            for (int i = 0; i < {R * B}; ++i) dar[i] = 0.f;")
        for lp in local:
            p3.append("            __syncwarp();")
            for b in range(B):
                p3.append(f"            dw_s[{b} * 32 + lane] = dw[{lp}][{b}];")
            p3.append("            __syncwarp();")
            for rr in range(R):
                p3.append(f"            {{ const float wr = Ws[({lp} * K + {32 * J + rr}) * WS + lane];")
                for b in range(B):
                    p3.append(f"              dar[{b * R + rr}] = fmaf(dw[{lp}][{b}], wr, dar[{b * R + rr}]);")
                p3.append("            }")
            p3.append("            #pragma unroll 2\n            for (int u4 = 0; u4 < 32; u4 += 4) {")
            for j in range(J):
                p3.append(f"                float W{j}[4];")
                p3.append(f"                #pragma unroll\n                for (int c = 0; c < 4; ++c) W{j}[c] = Ws[({lp} * K + lane + {32 * j}) * WS + u4 + c];")
            p3.append(f"                #pragma unroll\n                for (int b = 0; b < {B}; ++b) {{")
            p3.append("                    const float4 d = *reinterpret_cast<const float4*>(&dw_s[b * 32 + u4]);")
            for j in range(J):
                p3.append(f"                    dak[{j}][b] = fmaf(d.w, W{j}[3], fmaf(d.z, W{j}[2], fmaf(d.y, W{j}[1], fmaf(d.x, W{j}[0], dak[{j}][b]))));")
            p3.append("                }")
            p3.append("            }")
        p3.append(f"            #pragma unroll\n            for (int b = 0; b < {B}; ++b) if (b < nb) {{")
        for j in range(J):
            p3.append(f"                atomicAdd(&da[(e0 + b) * K + lane + {32 * j}], dak[{j}][b]);")
        p3.append("            }")
        if R:
            V = R * B
            vshift = 5 - int(math.log2(V))
            p3.append(transpose_reduce("dar", V, "            "))
            p3.append(f"            {{ const int idx = lane >> {vshift}; const int b = idx / {R}, k = {32 * J} + idx % {R};")
            p3.append(f"              if ((lane & {(1 << vshift) - 1}) == 0 && b < nb) atomicAdd(&da[(e0 + b) * K + k], dar[0]); }}")
        return p3

    def backward_source(self):
        B = self.BB
        cases = "\n".join(f"    case {q}:{self.backward_part(part)}    break;" for q, part in enumerate(self.parts))
        return f"""
extern "C" __global__ void __launch_bounds__({self.BW * 32}, 1) fused_backward(
    int64_t N, int64_t E, const float* __restrict__ a, const float* __restrict__ W,
    const float* __restrict__ x, const float* __restrict__ sh, const int* __restrict__ senders,
    const int* __restrict__ receivers, const int* __restrict__ row_ptr, int* __restrict__ counters,
    const float* __restrict__ dout, float* __restrict__ dx,
    float* __restrict__ dsh, float* __restrict__ da) {{
    extern __shared__ float smem[];
    float* Ws = smem;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    float* a_s = Ws + {self.ws_floats} + warp * (K * {B} + {B} * ED + {B} * 32);
    float* sh_s = a_s + K * {B};
    float* dw_s = sh_s + {B} * ED;
    const int cb = blockIdx.y;
    const int ch = cb * CB + lane;
    switch (blockIdx.z) {{
{cases}
    }}
}}
"""

    # ------------------------------------------------------------------ launch
    def slices(self):
        return (self.C // self.CB) * self.NPARTS
    def grid(self, nw, nodes):
        slices = (self.C // self.CB) * self.NPARTS
        gx = self.GX or max(1, -(-self.sms * 2 // slices))
        return (gx, self.C // self.CB, self.NPARTS)

    def counters(self, device):
        if getattr(self, "_ctr", None) is None:
            self._ctr = torch.zeros(2 * self.slices(), dtype=torch.int32, device=device)
        return self._ctr

    def forward(self, a, x, sh, senders, receivers, row_ptr, out, wout=None):
        N, E = x.shape[0], sh.shape[0]
        ctr = self.counters(x.device)
        ctr.zero_()
        self.mod.launch("fused_forward", self.grid(self.NW, N), self.NW * 32,
                        (N, E, a, self.W, x, sh, senders, receivers, row_ptr, ctr, out,
                         out if wout is None else wout), shared=self.smem_f)

    def backward(self, a, x, sh, senders, receivers, row_ptr, dout, dx, dsh, da):
        N, E = x.shape[0], sh.shape[0]
        dx.zero_()
        dsh.zero_()
        da.zero_()
        ctr = self.counters(x.device)
        ctr.zero_()
        self.mod.launch("fused_backward", self.grid(self.BW, N), self.BW * 32,
                        (N, E, a, self.W, x, sh, senders, receivers, row_ptr, ctr, dout, dx, dsh, da), shared=self.smem_b)
