"""The warp-specialized wgmma forward (H200 ideas plan, idea 30; ``fused_forward.WS_FORWARD``; Hopper sm_90a only).

A text transform of the fused forward's message render (the H200 plan: TC 3xTF32, TC_WFRAG, W_GROUP with X_LATE,
VEC_A, VEC_SH, DEGREE_PARTS, CG_FOLD; STORE_W false). Every part's ``case`` body is replaced; the TP, x gathers and
flushes are the render's own text. Per block (12 warps, one block per SM as today):

- warpgroup 0 is the producer, warps 4..11 are eight consumer warps;
- each consumer warp walks its own receiver chunks in 8-edge batches exactly as today's warps do (same chunk counter,
  same CSR order per receiver: no atomics); one round is the eight batches together, 64 edges, a wgmma's M;
- a ring of S stages in shared memory (``STAGES``: S per part, the largest that fits the part-1 budget). Per stage
  the w tile [paths][64][RS], the a tile [64][KP + 4] over its start, and the sh tile [64][ED]; W's TF32 high and low
  planes (core-matrix order, no swizzle) once per block;
- consumers stage their a and sh rows with cp.async and arrive on the stage's ``full`` mbarrier (through
  cp.async.mbarrier.arrive and a plain arrive), then run the TP of the round S - 1 back (x gathers, wait on
  ``wready``, today's TP and flushes) and arrive on ``empty``; each round carries its chunk bounds (rows of empty
  chunks are written as zeros when claimed);
- the producer waits on ``full``, loads the a tile as wgmma A fragments (registers, today's mask split), issues per
  k step today's three passes (lo.hi, hi.lo, hi.hi) as ``wgmma.mma_async m64nNk8.f32.tf32`` (N <= 128 per half)
  against W's planes, stores D into the w tile and arrives on ``wready``; a round without batches ends the block.
- ``ORDER``: which part a block runs. "z" is today's (blockIdx.z; all part-0 blocks launch before part 1's), "swap"
  runs part 1 (the heavier) first, "interleave" alternates the parts over the linear block index.

The same 3xTF32 terms as the mma.sync product, accumulated in the wgmma's order: rounding class (kernel lab on the
H200, OMAT-0 medium: messages within 3e-8 to 3e-7 relative of today's). An mbarrier wait traps after about 10 s (a
deadlock becomes an error, not a hang). Lab: ``tools/lab_idea30.py`` (variants ``ws30``, ``ws-both``, ...), the
protocol's local check ``tools/check_idea30_local.py --run`` (the debug render: ``product="simt"``,
``w_source="global"``, four consumers); results in ``results/h200-ideas-20261105/idea30-*.md``.

``apply_ws_forward(source)`` returns (new source, dynamic shared memory in bytes) or raises ValueError when the render
is not of this form (``fused_forward`` then builds today's kernel).
"""
from __future__ import annotations

import re

MARK = "// idea 30: sm_90a (WS_FORWARD: warp-specialized wgmma.mma_async forward)"
VEC_A_MARK = "// a is read with VEC_A's row stride AS: reinterpret_cast<const float4*>(a + (e0 + b) * AS)"
CONSUMERS = 8
WARPS = 4 + CONSUMERS
BARRIER_BYTES = 128  # full[S], empty[S], wready[S] (uint64), active[S], fin[S] (int); S <= 4
STAGES = {0: 3, 1: 2}  # ring stages per part (OMAT-0 medium on the H200: part 0 has room for a third)
ORDER = "z"
SHARED_LIMIT = 232448  # the H200's opt-in shared memory per block (the stages' budget)

HELPERS = r"""__device__ __forceinline__ unsigned smem_u32(const void* p) {
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
"""

_ORDERS = {
    "z": "",
    "swap": ("    // ORDER swap: part 1 first\n"
             "    const int bz_ = (int)gridDim.z - 1 - (int)blockIdx.z, by_ = blockIdx.y;\n"),
    "interleave": ("    // ORDER interleave: the parts alternate over the linear block index\n"
                   "    const int lin_ = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);\n"
                   "    const int bz_ = lin_ % (int)gridDim.z, by_ = (lin_ / (int)gridDim.z) % (int)gridDim.y;\n"),
}


def _wgmma_helpers(widths, kp):
    """The no-swizzle descriptor (LBO 128 B between the two k core matrices, SBO kp * 32 B between 8-row groups) and
    one ``wgmma_n{N}`` per width, A from registers (``fused_forward.wgmma_helpers``' form)."""
    out = [f"""__device__ __forceinline__ unsigned long long gmma_desc(unsigned addr) {{
    return (unsigned long long)((addr & 0x3FFFFu) >> 4) | ((unsigned long long)(128 >> 4) << 16)
        | ((unsigned long long)(({kp} * 32) >> 4) << 32);
}}
"""]
    for n in sorted(set(widths)):
        regs = n // 2
        d = ", ".join(f"%{i}" for i in range(regs))
        outs = ", ".join(f'"+f"(d[{i}])' for i in range(regs))
        out.append(f"""__device__ __forceinline__ void wgmma_n{n}(float* d, const unsigned (&a)[4],
                                                unsigned long long desc, int scale) {{
    asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{regs + 5}, 0;\\n"
                 "wgmma.mma_async.sync.aligned.m64n{n}k8.f32.tf32.tf32 {{{d}}}, {{%{regs}, %{regs + 1}, %{regs + 2}, "
                 "%{regs + 3}}}, %{regs + 4}, p, 1, 1;\\n}}\\n"
                 : {outs}
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale) : "memory");
}}
""")
    return "".join(out)


def _between(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i + len(start))]


def case_text(source, part):
    """The text of ``case part: { ... } break;`` in the render."""
    start = source.index(f"    case {part}: {{")
    return source[start:source.index("    } break;\n", start) + len("    } break;\n")]


def parts_of(source):
    return [int(v) for v in re.findall(r"^    case (\d+): \{", source, re.M)]


def kp_of(source):
    return int(re.search(r"constexpr int KP = (\d+);", source).group(1))


def _int(source, name):
    found = re.search(rf"constexpr int {name} = (\d+);", source)
    if not found:
        raise ValueError(f"ws_forward: no {name}")
    return int(found.group(1))


def extract(source, part):
    """The pieces of the render that the warp-specialized kernel reuses: the kernel's header, signature and prologue,
    and part ``part``'s W fill expression, path weight starts, CG factors, x gathers, TP and flush."""
    if "constexpr bool STORE_W = false;" not in source:
        raise ValueError("ws_forward: needs the message kernel (STORE_W false)")
    head = source[:source.index('extern "C"')]
    start = source.index('extern "C"')
    signature = source[start:source.index("{\n", start) + 2]
    prologue = _between(source, "    extern __shared__ float smem[];", "    switch (blockIdx.z) {")
    case = case_text(source, part)
    wst = re.search(r"const int wst\[(\d+)\] = \{([^}]*)\};", case)
    cpf = re.search(r"const float cpf\[\d+\] = \{[^}]*\};", case)
    wexpr = re.search(r"\] = (k < K \? .*?W\[\(int64_t\)\(wst\[p\] \+ cb \* CB \+ l\) \* KA \+ k\] : 0\.f);",
                      case, re.S)
    nacc = re.search(r"float acc\[(\d+)\];", case)
    if not (wst and wexpr and nacc and "group_sync(gbar);" in case and "if (STORE_W) {" in case
            and "                float xv[B][" in case):
        raise ValueError(f"ws_forward: part {part} is not of the W_GROUP / X_LATE form")
    nacc = int(nacc.group(1))
    flush_lines = re.findall(r"out\d*\[cur \* \w+ \+ \d+ \+ ch(?: \* \d+)?\] = acc\[\d+\];", case)[:nacc]
    if len(flush_lines) != nacc:
        raise ValueError(f"ws_forward: part {part}'s flush not found")
    xv = _between(case, "                float xv[B][", "                group_sync(gbar);")
    tp_start = case.index("                if (!tail) {\n", case.index("if (STORE_W) {"))
    tp = case[tp_start:case.index("                e0 += B;", tp_start)]
    if "sh_s" not in tp or "w[0][b]" not in tp or "rv[b]" not in tp or "lane_rv" not in xv:
        raise ValueError(f"ws_forward: part {part}'s TP does not read sh_s / w / rv as expected")
    return dict(head=head, signature=signature, prologue=prologue, part=part, npl=int(wst.group(1)),
                wst=wst.group(0), cpf=cpf.group(0) if cpf else "", wexpr=wexpr.group(1), nacc=nacc,
                nx=int(re.search(r"float xv\[B\]\[(\d+)\];", xv).group(1)), flush_lines=flush_lines, xv=xv, tp=tp)


def default_halves(npl):
    """The product's N split: halves of at most 128 columns (OMAT-0: part 1 2 x 96, part 0 1 x 128)."""
    nw = 32 * npl
    halves = 1
    while nw // halves > 128 or nw % halves or (nw // halves) % 8:
        halves += 1
    return halves


def part_shared_bytes(npl, kp, rs, ed, stages=2, consumers=CONSUMERS, w_source="smem"):
    """Dynamic shared memory of one warp-specialized part: W's two TF32 planes, ``stages`` stages, the barriers."""
    rows = 8 * consumers
    w_floats = 2 * npl * 32 * kp if w_source == "smem" else 0
    total = 4 * (w_floats + stages * (npl * rows * rs + rows * ed)) + BARRIER_BYTES
    return -(-total // 128) * 128


def part_body(info, kp, consumers=CONSUMERS, product="wgmma", w_source="smem", tp=True, halves=None, stages=2):
    """The ``case`` body of one part as the warp-specialized kernel (producer warpgroup + consumer warps).
    ``product``: "wgmma" (sm_90a), "simt" (FP32 FMAs: the debug render) or "none" (a timing falsifier);
    ``w_source``: "smem" (W's TF32 planes) or "global" (the simt debug product); ``tp``: False for a timing
    falsifier; ``stages``: the ring's depth S (the TP runs S - 1 rounds behind the staging)."""
    npl, nacc = info["npl"], info["nacc"]
    nw = npl * 32
    halves = halves or default_halves(npl)
    if product == "wgmma" and (consumers != 8 or w_source != "smem"):
        raise ValueError("ws_forward: wgmma needs eight consumer warps (M = 64) and W in shared memory")
    if nw % halves or (nw // halves) % 8 or nw // halves > 256:
        raise ValueError(f"ws_forward: N = {nw} does not split into {halves} wgmma widths")
    if not 2 <= stages <= 4:
        raise ValueError("ws_forward: 2 to 4 stages")
    lag = stages - 1
    hn = nw // halves
    flush = "\n".join(f"                        {line}" for line in info["flush_lines"])
    zero = "\n".join("                            " + re.sub(r"acc\[\d+\]", "0.f", line).replace("cur", "zr")
                     for line in info["flush_lines"])
    wfill = ""
    if w_source == "smem":
        wfill = f"""        {{  // W's TF32 planes (high, low) for wgmma's B: [N][KP], 8 x 4 core matrices, no swizzle
            {info['wst']}
            {info['cpf']}
            float* Wl = Ws + NW * KP;
            for (int i = threadIdx.x; i < NW * KP; i += blockDim.x) {{
                const int nn = i / KP, k = i - nn * KP, p = nn >> 5, l = nn & 31;
                unsigned hi, lo;
                split_tf32({info['wexpr']}, hi, lo);
                const int o = ((nn >> 3) * (KP / 4) + (k >> 2)) * 32 + (nn & 7) * 4 + (k & 3);
                Ws[o] = __uint_as_float(hi);
                Wl[o] = __uint_as_float(lo);
            }}
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }}
"""
    if product == "wgmma":
        product_code = f"""            unsigned ah[KP / 8][4], al[KP / 8][4];  // A: edge 16 pw + gid (+8), k 8 ks + tq
            #pragma unroll
            for (int ks = 0; ks < KP / 8; ++ks) {{
                const float* ar = st + (16 * pw + gid) * AW + 8 * ks + tq;
                split_tf32(ar[0], ah[ks][0], al[ks][0]);
                split_tf32(ar[8 * AW], ah[ks][1], al[ks][1]);
                split_tf32(ar[4], ah[ks][2], al[ks][2]);
                split_tf32(ar[8 * AW + 4], ah[ks][3], al[ks][3]);
            }}
            producer_sync();  // the whole a tile is in registers: D may overwrite it
            #pragma unroll
            for (int h = 0; h < {halves}; ++h) {{
                float d[{hn // 2}];
                asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int ks = 0; ks < KP / 8; ++ks) {{  // two 128-byte core matrices per k step
                    const unsigned off = h * {hn // 8} * (KP * 32) + 256 * ks;
                    wgmma_n{hn}(d, al[ks], gmma_desc(wh + off), ks > 0);
                    wgmma_n{hn}(d, ah[ks], gmma_desc(wl + off), 1);
                    wgmma_n{hn}(d, ah[ks], gmma_desc(wh + off), 1);
                }}
                asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                #pragma unroll
                for (int j = 0; j < {hn // 8}; ++j) {{  // D: edge 16 pw + gid (+8), column h N/2 + 8 j + 2 tq (+1)
                    const int col = h * {hn} + 8 * j + 2 * tq, p = col >> 5, c = col & 31;
                    float* rp = st + (p * ROWS + 16 * pw + gid) * RS + c;
                    *reinterpret_cast<float2*>(rp) = make_float2(d[4 * j], d[4 * j + 1]);
                    *reinterpret_cast<float2*>(rp + 8 * RS) = make_float2(d[4 * j + 2], d[4 * j + 3]);
                }}
            }}
"""
    elif product == "simt":
        product_code = f"""            {info['wst']}
            {info['cpf']}
            float dv[{halves}][{hn // 8}][4];
            #pragma unroll
            for (int h = 0; h < {halves}; ++h)
                #pragma unroll
                for (int j = 0; j < {hn // 8}; ++j)
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) {{
                        const int row = 16 * pw + gid + 8 * (q >> 1), col = h * {hn} + 8 * j + 2 * tq + (q & 1);
                        const int p = col >> 5, l = col & 31;
                        float sum = 0.f;
                        if (row < ROWS)
                            for (int k = 0; k < KP; ++k) sum = fmaf(st[row * AW + k], {info['wexpr']}, sum);
                        dv[h][j][q] = sum;
                    }}
            producer_sync();  // the whole a tile is read: D may overwrite it
            #pragma unroll
            for (int h = 0; h < {halves}; ++h)
                #pragma unroll
                for (int j = 0; j < {hn // 8}; ++j) {{
                    const int col = h * {hn} + 8 * j + 2 * tq, p = col >> 5, c = col & 31;
                    if (16 * pw + gid < ROWS) {{
                        float* rp = st + (p * ROWS + 16 * pw + gid) * RS + c;
                        *reinterpret_cast<float2*>(rp) = make_float2(dv[h][j][0], dv[h][j][1]);
                    }}
                    if (16 * pw + gid + 8 < ROWS) {{
                        float* rp = st + (p * ROWS + 16 * pw + gid + 8) * RS + c;
                        *reinterpret_cast<float2*>(rp) = make_float2(dv[h][j][2], dv[h][j][3]);
                    }}
                }}
"""
    elif product == "none":
        product_code = "            // timing falsifier: no product, no D\n"
    else:
        raise ValueError(product)
    tp_code = ""
    if tp:
        tp_code = f"""                    if (q0_first) cur = q0_r0;
                    {{
                    const int nb = q0_nb;
                    const bool tail = false;
{info['tp']}
                    }}
                    if (q0_last) while (cur < q0_r1) {{
{flush}
                        #pragma unroll
                        for (int i = 0; i < {nacc}; ++i) acc[i] = 0.f;
                        ++cur;
                    }}
"""
    else:
        tp_code = "                    (void)sh_s; (void)w; (void)xv;\n"
    # The rounds between staging and TP: q0 (the oldest, whose TP runs now) .. q{lag - 1}.
    fields = (("int", "nb", "0"), ("int", "rv", "0x7fffffff"), ("int", "snd", "-1"), ("int64_t", "r0", "0"),
              ("int64_t", "r1", "0"), ("bool", "first", "false"), ("bool", "last", "false"))
    queue = "\n".join(f"        {t} " + ", ".join(f"q{i}_{f} = {v}" for i in range(lag)) + ";" for t, f, v in fields)
    new = dict(nb="nb", rv="n_rv", snd="n_snd", r0="r0", r1="r1", first="first", last="n_last")
    shift = "\n".join(f"            q{i}_{f} = q{i + 1}_{f};" for i in range(lag - 1) for _, f, _ in fields)
    shift += "\n" + "\n".join(f"            q{lag - 1}_{f} = {new[f]};" for _, f, _ in fields)
    return f"""    case {info['part']}: {{  // WS_FORWARD: a producer warpgroup, {consumers} consumers, {stages} stages
        constexpr int NW = {nw};  // the part's W rows (paths x 32 channels): the product's N
        constexpr int S = {stages};  // ring stages; the TP runs S - 1 rounds behind the staging
        constexpr int WREG = {npl} * ROWS * RS;  // a stage's w tile [paths][ROWS][RS]; the a tile [ROWS][AW] on it
        constexpr int STG = WREG + ROWS * ED;  // w tile, then the sh tile [ROWS][ED]
        constexpr int W_FLOATS = {2 * nw * kp if w_source == "smem" else 0};  // W's high and low TF32 planes
        constexpr int BAR_OFF = W_FLOATS + S * STG;
        static_assert(ROWS * AW <= WREG, "the a tile fits inside the w tile");
        int* flags = reinterpret_cast<int*>(smem + BAR_OFF + 6 * S);  // active[S], fin[S]
        const unsigned bar0 = smem_u32(smem + BAR_OFF);  // full[s] at + 8 s, empty[s] + 8 (S + s), wready + 8 (2 S + s)
{wfill}        if (threadIdx.x == 0) {{
            for (int s = 0; s < S; ++s) {{
                mb_init(bar0 + 8 * s, NC * 32);
                mb_init(bar0 + 8 * (S + s), NC * 32);
                mb_init(bar0 + 8 * (2 * S + s), 128);
                flags[s] = 0;
                flags[S + s] = 0;
            }}
        }}
        __syncthreads();
        if (warp < 4) {{  // the producer warpgroup
        const int pw = warp, gid = lane >> 2, tq = lane & 3;
        const unsigned wh = smem_u32(Ws), wl = wh + NW * KP * 4;
        (void)wh; (void)wl;
        for (int n = 0;; ++n) {{
            const int s = n % S;
            mb_wait(bar0 + 8 * s, (n / S) & 1);  // full: the round's a and sh staged
            float* st = smem + W_FLOATS + s * STG;
            const int act = *(volatile int*)&flags[s];
            producer_sync();
            if (threadIdx.x == 0) {{
                flags[s] = 0;
                flags[S + s] = act == 0;
            }}
            if (act == 0) {{  // no consumer had a batch: the end
                mb_arrive(bar0 + 8 * (2 * S + s));
                break;
            }}
{product_code}            mb_arrive(bar0 + 8 * (2 * S + s));  // wready: the round's w tile written
        }}
        }} else {{  // consumer warps: today's chunks and TP, S - 1 rounds behind their staging
        const int c = warp - 4;
        bool have = false, done = false;
        int64_t r0 = 0, r1 = 0, eend = 0, e0 = 0, cur = 0;
{queue}
        float acc[{nacc}];
        #pragma unroll
        for (int i = 0; i < {nacc}; ++i) acc[i] = 0.f;
        for (int n = 0;; ++n) {{
            const int s = n % S;
            bool first = false;
            if (!done && (!have || e0 >= eend)) {{  // the next chunk with edges; empty chunks' rows written as zeros
                have = false;
                while (true) {{
                    int cc = 0;
                    if (lane == 0) cc = atomicAdd(ctr, 1);
                    cc = __shfl_sync(FULL, cc, 0);
                    if (cc >= receiver_chunks) {{
                        done = true;
                        break;
                    }}
                    r0 = (int64_t)cc * RC;
                    r1 = r0 + RC < N ? r0 + RC : N;
                    const int64_t ebeg = row_ptr[r0] < 0 ? 0 : row_ptr[r0];
                    eend = row_ptr[r1] < ebeg ? ebeg : row_ptr[r1];
                    eend = eend > E ? E : eend;
                    if (ebeg < eend) {{
                        have = true;
                        first = true;
                        e0 = ebeg;
                        break;
                    }}
                    for (int64_t zr = r0; zr < r1; ++zr) {{
{zero}
                    }}
                }}
            }}
            const int nb = done ? 0 : (int)((eend - e0) < B ? (eend - e0) : B);
            if (n >= S) mb_wait(bar0 + 8 * (S + s), ((n / S) - 1) & 1);  // empty: round n - S's tiles read
            float* st = smem + W_FLOATS + s * STG;
            if (nb > 0) {{
                for (int i = lane; i < B * (KP / 4); i += 32) {{
                    const int b = i / (KP / 4), q = i - b * (KP / 4);
                    cp_async16(st + (8 * c + b) * AW + 4 * q, b < nb ? a + (e0 + b) * AS + 4 * q : a, b < nb ? 16 : 0);
                }}
                for (int i = lane; i < B * ED / 4; i += 32)
                    cp_async16(st + WREG + 8 * c * ED + 4 * i, (4 * i / ED < nb) ? sh + e0 * ED + 4 * i : sh,
                               (4 * i / ED < nb) ? 16 : 0);
                if (lane == 0) atomicAdd(&flags[s], 1);
            }}
            cp_async_arrive(bar0 + 8 * s);  // full: lands when this thread's copies have
            mb_arrive(bar0 + 8 * s);
            const int n_rv = (lane < nb) ? receivers[e0 + lane] : 0x7fffffff;
            const int n_snd = (lane < nb) ? senders[e0 + lane] : -1;
            const bool n_last = nb > 0 && e0 + nb >= eend;
            e0 += nb;
            if (n >= S - 1) {{  // the TP of round m = n - (S - 1)
                const int m = n - (S - 1), ps = m % S;
                const int lane_rv = q0_rv, lane_snd = q0_snd;
{info['xv']}                mb_wait(bar0 + 8 * (2 * S + ps), (m / S) & 1);  // wready
                if (*(volatile int*)&flags[S + ps]) break;
                if (q0_nb > 0) {{
                    const float* pst = smem + W_FLOATS + ps * STG;
                    const float* sh_s = pst + WREG + 8 * c * ED;
                    float w[{npl}][B];
                    #pragma unroll
                    for (int g = 0; g < {npl}; ++g)
                        #pragma unroll
                        for (int b = 0; b < B; ++b) w[g][b] = pst[(g * ROWS + 8 * c + b) * RS + lane];
{tp_code}                }}
                mb_arrive(bar0 + 8 * (S + ps));  // empty
            }}
{shift}
        }}
        }}
    }} break;
"""


def render_parts(source, parts, others="return", consumers=CONSUMERS, w_source="smem", order="z"):
    """The render with the parts in ``parts`` ({part: dict(product=, tp=, halves=, stages=)}) warp-specialized.
    ``others``: "return" (the other parts' blocks return at once, the lab's part-only variants) or "keep" (today's
    code); ``order``: ``ORDER``'s choices."""
    kp = kp_of(source)
    first = extract(source, next(iter(parts))) if parts else None
    head = source[:source.index('extern "C"')]
    start = source.index('extern "C"')
    signature = source[start:source.index("{\n", start) + 2]
    prologue = _between(source, "    extern __shared__ float smem[];", "    switch (blockIdx.z) {")
    if parts:
        signature, count = re.subn(r"__launch_bounds__\(\d+, 1\)", f"__launch_bounds__({32 * (4 + consumers)}, 1)",
                                   signature)
        if count != 1:
            raise ValueError("ws_forward: launch bounds not found")
    switch = "    switch (blockIdx.z) {\n"
    if order != "z":
        if "blockIdx.x" in prologue or prologue.count("blockIdx.y") != 2 or prologue.count("blockIdx.z") != 1:
            raise ValueError("ws_forward: the prologue's block indices are not of the expected form")
        prologue = prologue.replace("= blockIdx.y;", "= by_;").replace("blockIdx.y * gridDim.z + blockIdx.z",
                                                                       "by_ * gridDim.z + bz_")
        if "blockIdx" in prologue:
            raise ValueError("ws_forward: the prologue's block indices are not of the expected form")
        lines = prologue.split("\n")
        prologue = "\n".join([lines[0], _ORDERS[order].rstrip("\n"), *lines[1:]])
        switch = "    switch (bz_) {\n"
    cases, widths, wgmma = [], [], False
    for part in parts_of(source):
        if part in parts:
            options = dict(parts[part])
            info = first if first["part"] == part else extract(source, part)
            product = options.get("product", "wgmma")
            halves = options.get("halves") or default_halves(info["npl"])
            wgmma |= product == "wgmma"
            widths.append(info["npl"] * 32 // halves)
            cases.append(part_body(info, kp, consumers, product, w_source, options.get("tp", True), halves,
                                   options.get("stages", 2)))
        elif others == "keep":
            if parts and consumers != 8:
                raise ValueError("ws_forward: today's parts need 12 warps")
            cases.append(case_text(source, part))
        else:
            cases.append(f"    case {part}: return;  // idea 30: part {part} not run\n")
    if not parts:
        return f"{head}{signature}{prologue}{switch}{''.join(cases)}    }}\n}}\n"
    marks = (MARK + "\n" if wgmma else "") + VEC_A_MARK + "\n"
    helpers = HELPERS + (_wgmma_helpers(widths, kp) if wgmma else "")
    return (f"{head}{helpers}{marks}constexpr int NC = {consumers};  // consumer warps (warps 4 .. 4 + NC - 1)\n"
            "constexpr int ROWS = 8 * NC;  // edges per round: one 8-edge batch per consumer warp\n"
            "constexpr int AW = KP + 4;  // the a tile's row stride\n\n"
            f"{signature}{prologue}{switch}{''.join(cases)}    }}\n}}\n")


def shared_bytes(source, parts, consumers=CONSUMERS, w_source="smem"):
    """The largest warp-specialized part's dynamic shared memory ({part: options} as ``render_parts``)."""
    kp, rs, ed = kp_of(source), _int(source, "RS"), _int(source, "ED")
    return max(part_shared_bytes(extract(source, p)["npl"], kp, rs, ed, dict(o).get("stages", 2), consumers,
                                 w_source) for p, o in parts.items())


def fit_stages(source, stages=None, limit=SHARED_LIMIT):
    """{part: stages}: ``stages`` (default ``STAGES``) per part, lowered to 2 where a part's stages would need more
    shared memory than the largest two-stage part (so that the deeper ring never raises the kernel's budget) or than
    ``limit``."""
    stages = STAGES if stages is None else stages
    kp, rs, ed = kp_of(source), _int(source, "RS"), _int(source, "ED")
    npl = {p: extract(source, p)["npl"] for p in parts_of(source)}
    budget = min(limit, max(part_shared_bytes(n, kp, rs, ed, 2) for n in npl.values()))
    return {p: (stages.get(p, 2) if part_shared_bytes(n, kp, rs, ed, stages.get(p, 2)) <= budget else 2)
            for p, n in npl.items()}


def apply_ws_forward(source, stages=None, order=None):
    """(the warp-specialized render of ``source``, its dynamic shared memory in bytes); ValueError when ``source``
    is not the H200 plan's message render (STORE_W false, VEC_A, W_GROUP with X_LATE, every part's TP of that form)
    or ``KP``, ``RS``, ``ED`` are of another form."""
    if MARK in source:
        raise ValueError("ws_forward: already applied")
    if "constexpr bool STORE_W = false;" not in source:
        raise ValueError("ws_forward: needs the message kernel (STORE_W false)")
    if "reinterpret_cast<const float4*>(a + (e0 + b) * AS)" not in source:
        raise ValueError("ws_forward: needs VEC_A's a rows")
    if "switch (blockIdx.z) {" not in source:
        raise ValueError("ws_forward: no part switch")
    kp, rs, ed = kp_of(source), _int(source, "RS"), _int(source, "ED")
    if kp % 8 or rs % 2 or ed % 4:
        raise ValueError("ws_forward: KP, RS or ED of another form")
    parts = parts_of(source)
    if any(32 * extract(source, p)["npl"] > 256 for p in parts):
        raise ValueError("ws_forward: more than 8 paths in a part")
    plan = {p: dict(stages=s) for p, s in fit_stages(source, stages).items()}
    return render_parts(source, plan, order=ORDER if order is None else order), shared_bytes(source, plan)
