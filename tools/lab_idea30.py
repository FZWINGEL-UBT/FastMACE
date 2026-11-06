"""Idea 30's falsifier (H200 ideas plan, revision 14; user approval 6 November 2026): a warp-specialized wgmma
forward for part 1 alone, as a kernel-lab variant of the fused forward. Lab prototype only; nothing for production.

The plan's text: "one producer warpgroup stages a, x and sh (TMA or cp.async) and issues wgmma.mma_async for
w = a . W, with W pre-split (big and small parts) in shared memory; two consumer warpgroups run the TP from the
finished w, synchronized by mbarriers ... Falsifier: a lab prototype of part 1 alone with the TP's FP32 work on
consumer warps and the product on a producer, against F-tc-no-product's bound. Stop if it is not <= 0.85 of the
part's time."

Design of ``ws30`` (a new kernel body for part 1, generated from the plan's render; part-0 blocks return at once,
as in the lab's F-part1-only):
- 12 warps per block as today: warpgroup 0 is the producer, warpgroups 1 and 2 hold eight consumer warps.
- Each consumer warp walks its own receiver chunks in 8-edge batches exactly as today's warps do (the same chunk
  counter, the same CSR order per receiver), so no message needs atomics. One round is the eight consumer warps'
  batches together: 64 edges, the M of one wgmma.
- A ring of two stages in shared memory. Per stage: the w tile [paths][64 edges][RS] (the producer's D, read by the
  consumers with lane = channel), the a tile [64][KP + 4] laid over the start of the w tile (the producer reads all
  of a into registers before it writes D), and the sh tile [64][ED].
- Consumers stage their batch's a rows and sh rows with cp.async (16 B, zero-filled past the batch) and arrive on the
  stage's ``full`` mbarrier through cp.async.mbarrier.arrive (the arrival lands when the copies have) and a plain
  arrive. Then they run the TP of the previous round (x gathers first, then wait on that stage's ``wready``), and
  arrive on its ``empty``. The staging side claims chunks one round ahead of the TP side; each round carries its
  batch's chunk bounds, so the receiver flushes are today's (rows of empty chunks are written as zeros at claim).
- The producer waits on ``full``, loads the 64 x KP a tile as wgmma A fragments (registers; split into TF32 high
  and low parts by the mask split, as today), and issues per half of the part's paths (N = 96) and per k step the
  three passes lo.hi, hi.lo, hi.hi as wgmma.mma_async m64n96k8 against W's high and low planes in shared memory
  (filled once per block in core-matrix order, no swizzle; round 1's WGMMA layout and descriptors). It writes D to
  the w tile and arrives on ``wready``. A round with no active consumer ends the producer and, through a flag, the
  consumers.
- The products are the same 3xTF32 terms as today's mma.sync product but accumulated in the wgmma's order:
  rounding class, not bitwise.

Falsifiers of the prototype itself (wrong outputs): ``F-ws30-no-tp`` (consumers skip the TP: staging and product
alone) and ``F-ws30-no-product`` (the producer issues no wgmma and stores no D: staging and TP alone).
``F-tc-no-product-p1`` is the lab's F-tc-no-product restricted to part 1 (the bound the plan names).

Debug render (``render_ws(..., consumers=4, product="simt", w_source="global")``): the same protocol with four
consumer warps (32-edge rounds) and an FP32 SIMT product reading W from global memory, which fits the shared memory
of an sm_120 GPU; used by ``check_idea30_local.py`` against a naive reference kernel generated from the same TP text.

Hook (``tools/omat0_kernel_lab.py``): ``_idea30_variants(kernel)``, which also adds the references (ws30 and its
falsifiers are compared with F-part1-only).
"""
from __future__ import annotations

import re

SM90A_MARK = "// idea 30: sm_90a (wgmma.mma_async)"
VEC_A_MARK = "// a is read with VEC_A's row stride AS: reinterpret_cast<const float4*>(a + (e0 + b) * AS)"
BARRIER_BYTES = 64  # full[2], empty[2], wready[2] (uint64), active[2], fin[2] (int)


def _between(text, start, end, start_at=0):
    i = text.index(start, start_at)
    j = text.index(end, i + len(start))
    return text[i:j]


def extract(source):
    """The pieces of the plan's render that ws30 reuses: the kernel's header and prologue, and part 1's W fill
    expression, path weight starts, CG factors, x gathers, TP and flush."""
    if "constexpr bool STORE_W = false;" not in source:
        raise ValueError("ws30 needs STORE_W false (the H200 plan with CG_FOLD)")
    head = source[:source.index('extern "C"')]
    kernel_start = source.index('extern "C"')
    signature = source[kernel_start:source.index("{\n", kernel_start) + 2]
    prologue = _between(source, "    extern __shared__ float smem[];", "    switch (blockIdx.z) {")
    prologue = "\n".join(line for line in prologue.splitlines()
                         if not re.match(r"\s*float\* (a_s|sh_s|r_s) =", line)) + "\n"
    case1 = source[source.index("    case 1: {"):]
    wst = re.search(r"const int wst\[(\d+)\] = \{([^}]*)\};", case1)
    npl = int(wst.group(1))
    cpf = re.search(r"const float cpf\[\d+\] = \{[^}]*\};", case1)
    wexpr = re.search(r"\] = (k < K \? .*?W\[\(int64_t\)\(wst\[p\] \+ cb \* CB \+ l\) \* KA \+ k\] : 0\.f);",
                      case1, re.S).group(1)
    nacc = int(re.search(r"float acc\[(\d+)\];", case1).group(1))
    flush_lines = re.findall(r"out\d\[cur \* \d+ \+ \d+ \+ ch\] = acc\[\d+\];", case1)[:nacc]
    if len(flush_lines) != nacc:
        raise ValueError("ws30: part 1's flush not found")
    xv = _between(case1, "                float xv[B][", "                group_sync(gbar);")
    nx = int(re.search(r"float xv\[B\]\[(\d+)\];", xv).group(1))
    tp_start = case1.index("                if (!tail) {\n", case1.index("if (STORE_W) {"))
    tp = case1[tp_start:case1.index("                e0 += B;", tp_start)]
    if "sh_s[b * ED + " not in tp or "w[0][b]" not in tp:
        raise ValueError("ws30: part 1's TP does not read sh_s / w as expected")
    return dict(head=head, signature=signature, prologue=prologue, npl=npl, wst=wst.group(0),
                cpf=cpf.group(0) if cpf else "", wexpr=wexpr, nacc=nacc, nx=nx, flush_lines=flush_lines, xv=xv,
                tp=tp)


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
    asm volatile("{\n.reg .pred P1;\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\nselp.b32 %0, 1, 0, P1;\n}\n"
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


def _wgmma_helpers(widths, kp):
    """Round 1's wgmma helpers (``fused_forward.wgmma_helpers``), copied so that the lab does not depend on them
    staying: the no-swizzle descriptor (LBO 128 B between the two k core matrices, SBO kp * 32 B between 8-row
    groups) and one ``wgmma_n{N}`` per width, A from registers."""
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


def _kp(source):
    return int(re.search(r"constexpr int KP = (\d+);", source).group(1))


def shared_bytes(info, consumers=8, w_source="smem", kp=64, rs=36, ed=16):
    rows = 8 * consumers
    w_floats = 2 * info["npl"] * 32 * kp if w_source == "smem" else 0
    stage = info["npl"] * rows * rs + rows * ed
    total = 4 * (w_floats + 2 * stage) + BARRIER_BYTES
    return -(-total // 128) * 128


def render_ws(source, consumers=8, product="wgmma", w_source="smem", tp=True, halves=2):
    """Part 1 of the plan's render ``source`` as the warp-specialized kernel (see the module docstring).
    ``product``: "wgmma" (sm_90a), "simt" (FP32 FMAs, the debug render) or "none" (F-ws30-no-product);
    ``w_source``: "smem" (W's TF32 planes in shared memory; needed by wgmma) or "global" (debug); ``tp``: False for
    F-ws30-no-tp; ``halves``: the product's N split (2: two m64n96k8 per pass and k step)."""
    info = extract(source)
    kp = _kp(source)
    npl, nacc = info["npl"], info["nacc"]
    nw = npl * 32
    if product == "wgmma" and (consumers != 8 or w_source != "smem"):
        raise ValueError("ws30: wgmma needs eight consumer warps (M = 64) and W in shared memory")
    if nw % halves or (nw // halves) % 8 or nw // halves > 256:
        raise ValueError(f"ws30: N = {nw} does not split into {halves} wgmma widths")
    hn = nw // halves
    threads = 32 * (4 + consumers)
    signature = re.sub(r"__launch_bounds__\(\d+, 1\)", f"__launch_bounds__({threads}, 1)", info["signature"])
    flush = "\n".join(f"                        {line}" for line in info["flush_lines"])
    zero = "\n".join("                            " + re.sub(r"acc\[\d+\]", "0.f", line).replace("cur", "zr")
                     for line in info["flush_lines"])
    flush_block = f"""{{
{flush}
                        #pragma unroll
                        for (int i = 0; i < {nacc}; ++i) acc[i] = 0.f;
                        ++cur;
                    }}"""
    wfill = ""
    if w_source == "smem":
        wfill = f"""    {{  // W's TF32 planes (high, low) for wgmma's B: [N rows][KP] in 8 x 4 core matrices, no swizzle
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
        product_code = f"""                unsigned ah[KP / 8][4], al[KP / 8][4];  // A: edge 16 pw + gid (+8), k 8 ks + tq (+4)
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
        wval = info["wexpr"]
        product_code = f"""                {info['wst']}
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
                                for (int k = 0; k < KP; ++k) sum = fmaf(st[row * AW + k], {wval}, sum);
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
        product_code = "                // F-ws30-no-product: no product, no D\n"
    else:
        raise ValueError(product)
    tp_code = ""
    if tp:
        tp_code = f"""                    if (p_first) cur = p_r0;
                    {{
                    const int nb = p_nb;
                    const bool tail = false;
{info['tp']}
                    }}
                    if (p_last) while (cur < p_r1) {flush_block}
"""
    marks = (SM90A_MARK + "\n" if product == "wgmma" else "") + VEC_A_MARK + "\n"
    helpers = HELPERS + (_wgmma_helpers([hn], kp) if product == "wgmma" else "")
    xv = info["xv"]
    return f"""{info['head']}{helpers}{marks}constexpr int NC = {consumers};  // consumer warps (warps 4 .. 4 + NC - 1)
constexpr int ROWS = 8 * NC;  // edges per round: one 8-edge batch per consumer warp
constexpr int AW = KP + 4;  // the a tile's row stride
constexpr int NW = {nw};  // the part's W rows (paths x 32 channels): the product's N
constexpr int WREG = {npl} * ROWS * RS;  // a stage's w tile [paths][ROWS][RS]; its a tile [ROWS][AW] overlaps it
constexpr int STG = WREG + ROWS * ED;  // w tile, then the sh tile [ROWS][ED]
constexpr int W_FLOATS = {2 * nw * kp if w_source == "smem" else 0};
constexpr int BAR_OFF = W_FLOATS + 2 * STG;
static_assert(ROWS * AW <= WREG, "the a tile fits inside the w tile");

{signature}{info['prologue']}    if (blockIdx.z != 1) return;  // idea 30: part 1 only
    int* flags = reinterpret_cast<int*>(smem + BAR_OFF + 12);  // active[2], fin[2]
    const unsigned bar0 = smem_u32(smem + BAR_OFF);  // full[s] at + 8 s, empty[s] + 16 + 8 s, wready[s] + 32 + 8 s
{wfill}    if (threadIdx.x == 0) {{
        for (int s = 0; s < 2; ++s) {{
            mb_init(bar0 + 8 * s, NC * 32);
            mb_init(bar0 + 16 + 8 * s, NC * 32);
            mb_init(bar0 + 32 + 8 * s, 128);
            flags[s] = 0;
            flags[2 + s] = 0;
        }}
    }}
    __syncthreads();
    if (warp < 4) {{  // the producer warpgroup
        const int pw = warp, gid = lane >> 2, tq = lane & 3;
        const unsigned wh = smem_u32(Ws), wl = wh + NW * KP * 4;
        (void)wh; (void)wl; (void)gid; (void)tq;
        for (int n = 0;; ++n) {{
            const int s = n & 1;
            mb_wait(bar0 + 8 * s, (n >> 1) & 1);  // full: the round's a and sh staged
            float* st = smem + W_FLOATS + s * STG;
            const int act = *(volatile int*)&flags[s];
            producer_sync();
            if (threadIdx.x == 0) {{
                flags[s] = 0;
                flags[2 + s] = act == 0;
            }}
            if (act == 0) {{  // no consumer had a batch: the end
                mb_arrive(bar0 + 32 + 8 * s);
                break;
            }}
{product_code}            mb_arrive(bar0 + 32 + 8 * s);  // wready: the round's w tile written
        }}
    }} else {{  // consumer warps: today's chunks and TP, one round behind their staging
        const int c = warp - 4;
        bool have = false, done = false;
        int64_t r0 = 0, r1 = 0, eend = 0, e0 = 0, cur = 0;
        int p_nb = 0, p_rv = 0x7fffffff, p_snd = -1;
        int64_t p_r0 = 0, p_r1 = 0;
        bool p_first = false, p_last = false;
        float acc[{nacc}];
        #pragma unroll
        for (int i = 0; i < {nacc}; ++i) acc[i] = 0.f;
        for (int n = 0;; ++n) {{
            const int s = n & 1;
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
            if (n >= 2) mb_wait(bar0 + 16 + 8 * s, ((n >> 1) - 1) & 1);  // empty: round n - 2's w tile read
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
            if (n >= 1) {{  // the TP of round n - 1
                const int ps = s ^ 1;
                const int lane_rv = p_rv, lane_snd = p_snd;
{xv}                mb_wait(bar0 + 32 + 8 * ps, ((n - 1) >> 1) & 1);  // wready
                if (*(volatile int*)&flags[2 + ps]) break;
                if (p_nb > 0) {{
                    const float* pst = smem + W_FLOATS + ps * STG;
                    const float* sh_s = pst + WREG + 8 * c * ED;
                    float w[{npl}][B];
                    #pragma unroll
                    for (int g = 0; g < {npl}; ++g)
                        #pragma unroll
                        for (int b = 0; b < B; ++b) w[g][b] = pst[(g * ROWS + 8 * c + b) * RS + lane];
                    (void)sh_s; (void)w; (void)rv; (void)xv;
{tp_code}                }}
                mb_arrive(bar0 + 16 + 8 * ps);  // empty
            }}
            p_nb = nb;
            p_rv = n_rv;
            p_snd = n_snd;
            p_first = first;
            p_last = n_last;
            p_r0 = r0;
            p_r1 = r1;
        }}
    }}
}}
"""


def reference_kernel(source):
    """A naive reference of part 1's messages (debug check): one block per receiver row and channel block, one thread
    per channel; w by FP32 FMAs in k order (as the simt debug product); the same TP text with one edge per batch."""
    info = extract(source)
    flush = "\n".join(f"    {line}" for line in info["flush_lines"])
    return f"""{info['head']}
extern "C" __global__ void ref_part1(int64_t N, int64_t E, const float* __restrict__ a, const float* __restrict__ W,
    const float* __restrict__ x, const float* __restrict__ sh, const int* __restrict__ senders,
    const int* __restrict__ receivers, const int* __restrict__ row_ptr,
    float* __restrict__ out0, float* __restrict__ out1, float* __restrict__ out2, float* __restrict__ out3) {{
    const int lane = threadIdx.x, cb = blockIdx.y, ch = cb * CB + lane;
    int64_t cur = blockIdx.x;
    {info['wst']}
    {info['cpf']}
    float acc[{info['nacc']}];
    for (int i = 0; i < {info['nacc']}; ++i) acc[i] = 0.f;
    for (int64_t e0 = row_ptr[cur]; e0 < row_ptr[cur + 1]; ++e0) {{
        const int nb = 1;
        const bool tail = false;
        const int lane_rv = (int)cur, lane_snd = senders[e0];
        float w[{info['npl']}][B];
        for (int p = 0; p < {info['npl']}; ++p) {{
            const int l = lane;
            float sum = 0.f;
            for (int k = 0; k < KP; ++k) sum = fmaf(a[e0 * AS + k], {info['wexpr']}, sum);
            w[p][0] = sum;
        }}
        const float* sh_s = sh + e0 * ED;
{info['xv']}{info['tp']}
    }}
{flush}
}}
"""


def idea30_variants(kernel, rules, part_only, no_product_pattern):
    """name -> (kind, transform, extra) for the lab's ``forward_variants``; ``rules``: the lab's ``_RULES`` (each
    ws30 source gets its launch rule: 12 warps, its shared memory); ``part_only``: the lab's ``_part_only``;
    ``no_product_pattern``: (old, new) of the lab's F-tc-no-product."""
    import dataclasses

    def ws(**options):
        def transform(s):
            out = render_ws(s, **options)
            info = extract(s)
            rule = kernel.rule if kernel is not None else None
            if rule is not None:
                rules[out] = dataclasses.replace(rule, warps=4 + options.get("consumers", 8),
                                                 shared_bytes=shared_bytes(info, kp=_kp(s)))
            return out
        return transform

    def no_product_p1(s):
        old, new = no_product_pattern
        if s.count(old) != 2:
            raise ValueError("F-tc-no-product-p1: the k-loop pattern is not in both parts")
        return part_only(1)(s.replace(old, new))

    return {
        "ws30": ("rounding", ws(), ()),
        "ws30-n192": ("rounding", ws(halves=1), ()),
        "F-ws30-no-tp": ("falsifier", ws(tp=False), ()),
        "F-ws30-no-product": ("falsifier", ws(product="none"), ()),
        "F-tc-no-product-p1": ("falsifier", no_product_p1, ()),
    }


def idea30_references():
    return {"ws30": "F-part1-only", "ws30-n192": "F-part1-only"}
