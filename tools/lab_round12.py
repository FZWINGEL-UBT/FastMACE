"""Round 12 kernel-lab variants of the fused forward (H200 ideas, revision 12: ideas 22(iii) and 24), written as text
transforms of the plan's render for ``tools/omat0_kernel_lab.py``'s ``forward_variants``. The plan's render has X_LATE on
both GPUs since FastMACE ``fc0d7531`` (H200, A100-SXM4-80GB), and these transforms are written for that render.

Two ways to use it, as round 11's:
- paste these definitions into ``tools/omat0_kernel_lab.py`` and add ``**round12_variants(kernel)`` to the dict that
  ``forward_variants`` returns, or
- import this file and add ``**round12_variants(kernel)``.
No variant changes the launch rule or the shared memory (``_RULES`` is not touched).

Every transform first rebuilds each part of the render from the pieces it reads (the W fill, the accumulator and path
counts, the flush stores, the x gathers, the k-loop block and the TP blocks) and raises ValueError, as the lab's ``_sub``
does, unless the rebuild equals the part byte for byte. A transform therefore applies only to the form it was written
for: W_GROUP, VEC_A, STORE_W false, X_LATE (with or without K_PIPE), and LANE_INDICES (H200) or the A100-SXM4's
broadcast index loads.

Idea 22(iii), exact (bitwise): batch n's TP runs inside batch n + 1's k-loop, edge b in k-step b (KP / 8 = B = 8).
- x of the TP's next edge is gathered one k-step ahead; batch n + 1's sh waits in one float4 per lane until batch n's
  TP is done; batch n's receivers and senders sit in one register each per lane and are shuffled per edge (on the
  A100-SXM4 too, which loads them by broadcast today);
- the chunk-end flush moves behind the k-loop; chunks without edges get zero rows directly, as acc would have given;
- r_s once: batch n's TP reads its w before batch n + 1's C writes, behind one more pair barrier (gbar2: 3 barriers
  per batch instead of 2);
- branch-free (``iii``, ``iii-u1``): the TP runs for all 8 edges after the k-step's HMMAs in one basic block, so that
  ptxas can interleave them; a missing edge has x = 0, sh = 0 and w = 0 (its a row is zero) and no flush (receiver -1).
  It adds w * (x * e * c) = 0 to acc, which changes nothing unless an accumulator is exactly -0.0 (only an underflow
  makes one). sh_s and r_s are zeroed once per part, so the first TP, which has no batch n, reads zeros;
- ``iii-nf``: ``iii`` with a second copy of the k-loop that has no flush checks at all, taken when no receiver of
  batch n lies beyond the current row (``!__any_sync(FULL, p_rv > cur)``: then none of the 8 checks would flush), so
  a batch inside one row runs without a branch in its k-loop;
- ``iii-u1``: ``iii`` with the k-loop's ``#pragma unroll 2`` lowered to 1 (8 trips of one k-step instead of 4 of two);
- ``iii-br``: round 9's prototype form, the TP behind ``b < p_nb`` after the k-step's HMMAs (the control for what the
  branch-free form buys).
Each edge's TP keeps its operations and their order, every accumulator sees the same sequence of fmaf, and the flushes
write the same rows: bitwise against the plan render.

Idea 24, rounding (the force/stress gate before any default): the TP's CG factors folded into W.
- part 1: one factor c_p per path moves into the path's W rows as the block fills its W slice (``cpf[p] * W``), the
  terms become (c / c_p) x e, products are shared across paths, and a unit term costs an FADD or an FFMA at most;
- part 0 multiplies w by x0 first (s = w x0, acc += s e), and part 1's 1 (x) 0 -> 1 path w by e0 first.
Round 9's SASS count on the H200 render: 169 -> 133 FP32 instructions per edge-slice. ``fold`` is 24 on the plan
render, ``iii-fold`` and ``iii-nf-fold`` 24 on ``iii`` and ``iii-nf`` (every copy of the TP folded alike).

CPU check (no GPU, no torch): ``python -I lab_round12.py RENDER.cu OUTDIR`` writes every variant that applies to RENDER
into OUTDIR (LF) and says why the others do not apply.
"""
import itertools
import re
import struct
import sys


def _r12_sub(source, old, new, count=None):
    """``source`` with ``old`` replaced by ``new``; raises unless it occurs (``count`` times if given)."""
    found = source.count(old)
    if not found or (count is not None and found != count):
        raise ValueError(f"round-12 variant pattern found {found} times: {old[:80]!r}")
    return source.replace(old, new)


# The lab's forward_launcher reads a variant's source for this text to tell rows of a padded to AS floats (VEC_A) from
# rows of KA floats (round 11's stage-ahead lost it: CUDA_ERROR_ILLEGAL_ADDRESS). Every variant keeps it; checked.
_R12_VEC_A = "reinterpret_cast<const float4*>(a + (e0 + b) * AS)"


def _r12_guard(transform):
    """``transform``, refusing a result that lost the render's VEC_A text (``_R12_VEC_A``)."""
    def guarded(s):
        out = transform(s)
        if _R12_VEC_A in s and _R12_VEC_A not in out:
            raise ValueError("round-12 variant lost the text the lab's forward_launcher reads as VEC_A "
                             f"({_R12_VEC_A}); the lab would pass a compacted a")
        return out
    return guarded


# ------------------------------------------------------------------------------- the render's parts and their pieces
_R12_CASE = re.compile(r"^    case (\d+): \{\n", re.M)
_R12_BREAK = "    } break;\n"
_R12_TAIL = _R12_BREAK + "    }\n}"


def _r12_constant(source, name):
    match = re.search(rf"constexpr int {name} = (\d+);", source)
    if match is None:
        raise ValueError(f"round 12: constant {name} not found in the render")
    return int(match.group(1))


def _r12_warps(source):
    match = re.search(r"__launch_bounds__\((\d+), 1\)", source)
    if match is None:
        raise ValueError("round 12: no __launch_bounds__(threads, 1) in the render")
    return int(match.group(1)) // 32


def _r12_parts(source):
    """(text before the first part, [each part from its ``case`` line through its ``} break;``], the rest)."""
    if "constexpr bool STORE_W = false;" not in source:
        raise ValueError("round 12 needs the render with STORE_W false (22(iii) has no w stores)")
    if _r12_constant(source, "B") * _r12_constant(source, "ED") != 128 or _r12_constant(source, "KP") != 64:
        raise ValueError("round 12 needs B = 8, ED = 16 and KP = 64 (one sh float4 per lane, one edge per k-step)")
    starts = [m.start() for m in _R12_CASE.finditer(source)]
    if len(starts) != 2:
        raise ValueError(f"round 12 needs the render's two parts (DEGREE_PARTS); found {len(starts)}")
    end = source.rindex(_R12_TAIL) + len(_R12_BREAK)
    parts = [source[starts[0]:starts[1]], source[starts[1]:end]]
    for part in parts:
        if not part.endswith(_R12_BREAK):
            raise ValueError("round 12: a part does not end with its break")
    return source[:starts[0]], parts, source[end:]


_R12_SEARCH = """\
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
"""
_R12_STAGING = """\
                for (int i = lane; i < B * (KP / 4); i += 32) {
                    const int b = i / (KP / 4), q = i - b * (KP / 4);
                    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (b < nb) v = __ldg(reinterpret_cast<const float4*>(a + (e0 + b) * AS) + q);
                    *reinterpret_cast<float4*>(&a_s[b * KP + ((q ^ b) << 2)]) = v;
                }
"""
_R12_SH_STAGING = """\
                for (int i = lane; i < B * ED / 4; i += 32)
                    reinterpret_cast<float4*>(sh_s)[i] = (4 * i / ED < nb)
                        ? __ldg(reinterpret_cast<const float4*>(sh + e0 * ED) + i) : make_float4(0.f, 0.f, 0.f, 0.f);
"""
# The index forms: LANE_INDICES (H200: one load per lane at the staging, shuffles in the gathers) and the A100-SXM4's
# broadcast loads in the gathers.
_R12_IDX = {
    "lane": ("                const int lane_rv = (lane < nb && !tail) ? receivers[e0 + lane] : 0x7fffffff;\n"
             "                const int lane_snd = (lane < nb && !tail) ? senders[e0 + lane] : -1;\n",
             "                    rv[b] = __shfl_sync(FULL, lane_rv, b);\n"
             "                    int snd = __shfl_sync(FULL, lane_snd, b);\n"),
    "broadcast": ("",
                  "                    rv[b] = (b < nb && !tail) ? receivers[e0 + b] : 0x7fffffff;\n"
                  "                    int snd = (b < nb && !tail) ? senders[e0 + b] : -1;\n"),
}
# The plan's part (X_LATE): the gathers after the k-loop block, before the C-write barrier.
_R12_BASE = """\
    case @P@: {
@WFILL@        {
            const int gr = warp & 1, gbar = 1 + (warp >> 1);  // the warp's place in its pair and the pair's barrier
            bool have = false, done = false, tail = false;
            int64_t r1 = 0, eend = 0, cur = 0, e0 = 0;
            float acc[@NACC@];
            #pragma unroll
            for (int i = 0; i < @NACC@; ++i) acc[i] = 0.f;
            while (true) {
                if (!done && (!have || e0 >= eend)) {  // the next chunk with edges; empty ones flushed on the way
                    if (have) while (cur < r1) {
@FLUSH16@                    have = false;
                    while (true) {
@SEARCH@                        cur = r0;
                        if (ebeg < eend) {
                            have = true;
                            e0 = ebeg;
                            break;
                        }
                        while (cur < r1) {
@FLUSH16@                    }
                }
                const int nb = done ? 0 : (int)((eend - e0) < B ? (eend - e0) : B);  // done: an empty batch
                __syncwarp();
@STAGING@@SH_STAGING@@IDX@                __syncwarp();
                if (!group_any(!done, gbar)) break;  // the group's a_s staged and its r_s free; or both warps done
@KBLOCK@                float xv[B][@NX@];
                int rv[B];
                #pragma unroll
                for (int b = 0; b < B; ++b) {
@RVSND@                    if (snd < 0 || snd >= N) snd = -1;
@XV@                }
                group_sync(gbar);  // the group's r_s rows written
                float w[@NP@][B];
                #pragma unroll
                for (int g = 0; g < @NP@; ++g)
                    #pragma unroll
                    for (int b = 0; b < B; ++b) w[g][b] = r_s[(g * B + b) * RS + lane];
                if (STORE_W) {
                    const int wst[@NP@] = {@WST@};
                    #pragma unroll
                    for (int b = 0; b < B; ++b)
                        if (b < nb) {
                            #pragma unroll
                            for (int g = 0; g < @NP@; ++g) wout[(e0 + b) * Q + wst[g] + ch] = w[g][b];
                        }
                }
                if (!tail) {
                    #pragma unroll
                    for (int b = 0; b < B; ++b) {
                        if (b >= nb) break;
                        while (cur < rv[b]) {
@FLUSH16@@BLOCKS@                    }
                }
                e0 += B;
            }
        }
    } break;
"""
_R12_FLUSH_LINE = re.compile(r"^ {16}(out\d+\[cur \* \d+ \+ \d+ \+ ch\] = acc\[\d+\];)\n", re.M)
_R12_XV_LINE = re.compile(r"^ {20}xv\[b\]\[(\d+)\] = (\(snd >= 0\) \? x\[\(int64_t\)snd \* IN_DIM \+ \d+ \+ ch\] : 0\.f);\n",
                          re.M)


def _r12_fill(template, pieces):
    out = template
    for key, value in pieces.items():
        out = out.replace("@" + key + "@", value)
    if re.search(r"@[A-Z0-9_]+@", out):
        raise ValueError(f"round 12: unfilled template fields {re.findall(r'@[A-Z0-9_]+@', out)}")
    return out


def _r12_flush(stores, nacc, indent, row="cur"):
    """The flush of the accumulators into row ``row`` (and its zeroing and ``++row``) at ``indent`` spaces; the stores'
    own lines one level deeper."""
    ind, inner = " " * indent, " " * (indent + 4)
    return ("".join(f"{inner}{s.replace('cur', row)}\n" for s in stores)
            + f"{inner}#pragma unroll\n{inner}for (int i = 0; i < {nacc}; ++i) acc[i] = 0.f;\n{inner}++{row};\n"
            + f"{ind}}}\n")


def _r12_pieces(part):
    """The pieces of one part of the plan's render; raises unless they rebuild the part byte for byte."""
    pc = {"P": _R12_CASE.match(part).group(1)}
    sync = "        __syncthreads();\n"
    first = part.index("\n") + 1
    pc["WFILL"] = part[first:part.index(sync) + len(sync)]
    nacc = int(re.search(r"float acc\[(\d+)\];", part).group(1))
    npaths = int(re.search(r"float wacc\[(\d+)\]\[2\]\[4\];", part).group(1))
    nx = int(re.search(r"float xv\[B\]\[(\d+)\];", part).group(1))
    pc["NACC"], pc["NP"], pc["NX"] = str(nacc), str(npaths), str(nx)
    stores = _R12_FLUSH_LINE.findall(part)[:nacc]
    if len(stores) != nacc:
        raise ValueError(f"round 12: {len(stores)} flush stores for {nacc} accumulators")
    gathers = _R12_XV_LINE.findall(part)
    if [int(i) for i, _ in gathers] != list(range(nx)):
        raise ValueError(f"round 12: the x gathers do not match xv[B][{nx}]")
    pc["XV"] = "".join(f"                    xv[b][{i}] = {e};\n" for i, e in gathers)
    pc["FLUSH16"] = _r12_flush(stores, nacc, 12)
    pc["SEARCH"], pc["STAGING"], pc["SH_STAGING"] = _R12_SEARCH, _R12_STAGING, _R12_SH_STAGING
    form = "lane" if "lane_rv" in part else "broadcast"
    pc["IDX"], pc["RVSND"] = _R12_IDX[form]
    k0 = part.index("                float wacc[")
    k1 = part.index("                float xv[B][")
    if k1 < k0:
        raise ValueError("round 12 needs the X_LATE render: the x gathers after the k-loop block")
    pc["KBLOCK"] = part[k0:k1]
    pc["WST"] = re.search(r"const int wst\[\d+\] = \{([^}]*)\};\n {20}#pragma unroll\n", part).group(1)
    tp = part.index("                        while (cur < rv[b]) {\n")
    b0 = part.index("            }\n", tp) + len("            }\n")
    b1 = part.index("                    }\n                }\n                e0 += B;\n", b0)
    pc["BLOCKS"] = part[b0:b1]
    rebuilt = _r12_fill(_R12_BASE, pc)
    if rebuilt != part:
        line = next((i for i, (u, v) in enumerate(zip(rebuilt.splitlines(), part.splitlines())) if u != v), None)
        raise ValueError(f"round 12: part {pc['P']} is not the X_LATE plan render this file was written for "
                         f"(first difference at its line {line})")
    pc["STORES"], pc["GATHERS"], pc["FORM"] = stores, gathers, form
    return pc


def _r12_close(source, open_index):
    depth = 0
    for i in range(open_index, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("round 12: unbalanced braces after the k-loop")


# ---------------------------------------------------------------------------------------------------------- 22(iii)
_R12_KLOOP = re.compile(r"^ {20}#pragma unroll (\d+)\n {20}for \(int k0 = 0; k0 < KP; k0 \+= 8\) \{\n", re.M)
_R12_IIIB = """\
    case @P@: {
@WFILL@        {
            // 22(iii) (H200 ideas, revision 12): batch n's TP inside batch n + 1's k-loop, edge b in k-step b. Pair
            // barriers per batch: the vote (gbar), gbar2 (both warps' TP reads of r_s done before the C writes into
            // them) and gbar (the group's r_s rows written).
            const int gr = warp & 1, gbar = 1 + (warp >> 1), gbar2 = @G2@ + (warp >> 1);
            bool have = false, done = false, tail = false;
            int64_t r1 = 0, eend = 0, e0 = 0;  // the staging stream: batch n + 1
            int64_t cur = 0, tp_r1 = 0;  // the TP stream: batch n's next row and the end of its chunk
            int p_nb = 0, p_rv = -1, p_snd = -1;  // batch n: its edges; per lane its receiver and sender (-1: none)
            float acc[@NACC@];
            #pragma unroll
            for (int i = 0; i < @NACC@; ++i) acc[i] = 0.f;
@ZERO_INIT@            while (true) {
                bool opens = false;  // batch n + 1 opens a chunk, whose first row is o_r0
                int64_t o_r0 = 0;
                if (!done && (!have || e0 >= eend)) {  // the next chunk with edges; empty ones get zero rows
                    have = false;
                    while (true) {
@SEARCH@                        if (ebeg < eend) {
                            have = true;
                            e0 = ebeg;
                            opens = true;
                            o_r0 = r0;
                            break;
                        }
                        for (int64_t c = r0; c < r1; ++c) {  // a chunk without edges: zero rows, acc untouched
@ZERO@                        }
                    }
                }
                const int nb = done ? 0 : (int)((eend - e0) < B ? (eend - e0) : B);  // done: an empty batch
                __syncwarp();
@STAGING@                float4 shn = make_float4(0.f, 0.f, 0.f, 0.f);  // batch n + 1's sh: stored once batch n's TP is done
                if (4 * lane / ED < nb) shn = __ldg(reinterpret_cast<const float4*>(sh + e0 * ED) + lane);
                float xn[@NX@];  // x of the TP's next edge: edge 0 of batch n
                {
                    int snd = __shfl_sync(FULL, p_snd, 0);
                    if (snd < 0 || snd >= N) snd = -1;
@XN20@                }
                __syncwarp();
                if (!group_any(!done || p_nb > 0, gbar)) break;  // the group's a_s staged and its r_s free; or drained
@KBLOCK@                __syncwarp();
                reinterpret_cast<float4*>(sh_s)[lane] = shn;  // batch n + 1's sh: batch n's TP is done
                group_sync(gbar);  // the group's r_s rows written
                if (opens) {  // batch n + 1 opened a chunk: the rows left in batch n's chunk first
                    while (cur < tp_r1) {
@FLUSH20@                    cur = o_r0;
                    tp_r1 = r1;
                }
                p_nb = tail ? 0 : nb;
                p_rv = (lane < nb && !tail) ? receivers[e0 + lane] : -1;
                p_snd = (lane < nb && !tail) ? senders[e0 + lane] : -1;
                e0 += B;
            }
            while (cur < tp_r1) {  // the rows left in the last chunk
@FLUSH12@        }
    } break;
"""
_R12_ZERO_INIT = ("            for (int i = lane; i < A_STRIDE - KP * B; i += 32) sh_s[i] = 0.f;  "
                  "// sh_s and r_s: the first TP reads zeros\n")
_R12_TP_FLUSH = """\
                        {
                            const int rvb = __shfl_sync(FULL, p_rv, b);
                            while (cur < rvb) {  // batch n's receiver advances: its finished rows first
@FLUSH28@                        }
"""
_R12_TP_HEAD = """\
                        float xc[@NX@];  // x of edge b, gathered one k-step ahead
                        #pragma unroll
                        for (int i = 0; i < @NX@; ++i) xc[i] = xn[i];
                        {  // x of edge b + 1 (b = 7: lane 8's sender, -1, so zeros)
                            int snd = __shfl_sync(FULL, p_snd, b + 1);
                            if (snd < 0 || snd >= N) snd = -1;
@XN28@                        }
                        float wb[@NP@];  // batch n's w of edge b: r_s keeps batch n's rows until the C writes
                        #pragma unroll
                        for (int g = 0; g < @NP@; ++g) wb[g] = r_s[(g * B + b) * RS + lane];
"""
_R12_B_LINE = "                        const int b = k0 >> 3;  // 22(iii): batch n's edge b, in this k-step\n"
_R12_GBAR2 = "                    group_sync(gbar2);  // both warps' TP reads of r_s done before the C writes into them\n"


def _r12_indent(text, spaces):
    return "".join((" " * spaces + line if line.strip() else line) for line in text.splitlines(keepends=True))


_R12_NF_IF = ("                    if (__any_sync(FULL, p_rv > cur)) {  // a receiver of batch n advances: the k-loop "
              "with its flush checks\n")
_R12_NF_ELSE = ("                    } else {  // every edge of batch n adds to row cur: the k-loop without them, "
                "branch-free\n")


def _r12_iii_part(pc, warps, unroll, branch_free, no_flush_path=False):
    nacc, npaths, nx = int(pc["NACC"]), int(pc["NP"]), int(pc["NX"])
    kb = pc["KBLOCK"]
    loops = list(_R12_KLOOP.finditer(kb))
    if len(loops) != 1:
        raise ValueError(f"round 12: the k-loop header found {len(loops)} times in the k-loop block")
    loop = loops[0]
    header = loop.group(0)
    if unroll is not None:
        header = header.replace(f"#pragma unroll {loop.group(1)}", f"#pragma unroll {unroll}")
    closing = _r12_close(kb, loop.end() - 2)
    line = kb.rindex("\n", 0, closing) + 1
    if kb[line:closing].strip():
        raise ValueError("round 12: the k-loop's closing brace is not on a line of its own")
    body = kb[loop.end():line]
    blocks = re.sub(r"xv\[b\]\[(\d+)\]", r"xc[\1]", re.sub(r"\bw\[(\d+)\]\[b\]", r"wb[\1]", pc["BLOCKS"]))
    if re.search(r"\bw\[|xv\[", blocks):
        raise ValueError("round 12: the TP blocks read w or xv in another form")
    fields = {"NX": str(nx), "NP": str(npaths),
              "FLUSH28": _r12_flush(pc["STORES"], nacc, 28),
              "XN28": "".join(f"                            xn[{i}] = {e};\n" for i, e in pc["GATHERS"])}
    flush, head = _r12_fill(_R12_TP_FLUSH, fields), _r12_fill(_R12_TP_HEAD, fields)
    end = kb[line:closing + 2]  # the k-loop's closing line
    if branch_free:
        new_loop = header + _R12_B_LINE + flush + head + body + blocks + end
        if no_flush_path:  # the same loop without the flush checks, for the batches that need none
            plain = header + _R12_B_LINE + head + body + blocks + end
            new_loop = (_R12_NF_IF + _r12_indent(new_loop, 4) + _R12_NF_ELSE + _r12_indent(plain, 4)
                        + "                    }\n")
    else:  # round 9's form: the TP behind a branch, after the k-step
        new_loop = (header + _R12_B_LINE + body + "                        if (b < p_nb) {  // round 9's form: the TP "
                    "behind a branch\n" + _r12_indent(flush + head + blocks, 4) + "                        }\n" + end)
    if not end.endswith("}\n") or end.strip() != "}":
        raise ValueError("round 12: the k-loop's closing line is not a lone brace")
    kb = kb[:loop.start()] + new_loop + kb[closing + 2:]
    cwrites = (f"                    #pragma unroll\n                    for (int g = 0; g < {npaths}; ++g)\n"
               "                        #pragma unroll\n                        for (int t = 0; t < 2; ++t) {  // C fragment")
    kb = _r12_sub(kb, cwrites, _R12_GBAR2 + cwrites, 1)
    zeros = [re.sub(r"acc\[\d+\]", "0.f", s).replace("cur", "c") for s in pc["STORES"]]
    zero = "".join(f"                            {s}\n" for s in zeros)
    fields = {"P": pc["P"], "WFILL": pc["WFILL"], "G2": str(1 + warps // 2), "NACC": str(nacc), "NX": str(nx),
              "ZERO_INIT": _R12_ZERO_INIT if branch_free else "", "SEARCH": _R12_SEARCH, "ZERO": zero,
              "STAGING": _R12_STAGING, "KBLOCK": kb,
              "XN20": "".join(f"                    xn[{i}] = {e};\n" for i, e in pc["GATHERS"]),
              "FLUSH20": _r12_flush(pc["STORES"], nacc, 20), "FLUSH12": _r12_flush(pc["STORES"], nacc, 12)}
    return _r12_fill(_R12_IIIB, fields)


def _r12_iii(unroll=None, branch_free=True, no_flush_path=False):
    """22(iii) on both parts of the plan's render; ``unroll``: the k-loop's unroll (None: the render's);
    ``no_flush_path``: a second copy of the k-loop without the flush checks, taken when no receiver of batch n
    advances past row cur (one vote per batch)."""
    def transform(s):
        warps = _r12_warps(s)
        if warps % 2 or 1 + warps // 2 + warps // 2 > 15:
            raise ValueError(f"round 12: {warps} warps; 22(iii)'s second pair barriers need an even count up to 14")
        pre, parts, post = _r12_parts(s)
        return pre + "".join(_r12_iii_part(_r12_pieces(p), warps, unroll, branch_free, no_flush_path)
                             for p in parts) + post
    return transform


# -------------------------------------------------------------------------------------------------------------- 24
def _r12_f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


_R12_BLOCK = re.compile(r"^ *\{\n((?: *const float e\d+ = sh_s\[b \* ED \+ \d+\];\n)+)((?: *acc\[\d+\] = fmaf\(.*\);\n)+)"
                        r" *\}\n", re.M)
_R12_ACC = re.compile(r"^ *acc\[(\d+)\] = fmaf\((w\[(\d)\]\[b\]|wb\[(\d)\]), (.*), acc\[(\d+)\]\);$")
_R12_TERM = re.compile(r"^(xv\[b\]\[(\d)\]|xc\[(\d)\]) \* e(\d+) \* (-?0x[0-9a-f]+(?:\.[0-9a-f]*)?p[-+]\d+)f$")


def _r12_parse_runs(part):
    """The part's TP: every run of contiguous TP blocks (22(iii)'s no-flush path has two copies) as (start, end); and
    the TP they all hold, [(accumulator, path, [(x index, sh index, float32 c)])] and its (w, x) style."""
    blocks = list(_R12_BLOCK.finditer(part))
    if not blocks:
        raise ValueError("round 12 (24): no TP blocks in the part")
    runs = [[blocks[0]]]
    for u, v in zip(blocks, blocks[1:]):
        if u.end() == v.start():
            runs[-1].append(v)
        else:
            runs.append([v])
    parsed = []
    for run in runs:
        outs, style = [], None
        for block in run:
            emap = {int(k): int(j)
                    for k, j in re.findall(r"const float e(\d+) = sh_s\[b \* ED \+ (\d+)\];", block.group(1))}
            for line in block.group(2).splitlines():
                am = _R12_ACC.match(line)
                if not am or am.group(1) != am.group(6):
                    raise ValueError(f"round 12 (24): unexpected TP line {line!r}")
                g = int(am.group(3) or am.group(4))
                first = _R12_TERM.match(am.group(5).split(" + ")[0])
                style = ("w" if am.group(3) else "wb", "xv" if first and first.group(2) else "xc")
                terms = []
                for t in am.group(5).split(" + "):
                    tm = _R12_TERM.match(t)
                    if not tm:
                        raise ValueError(f"round 12 (24): unexpected TP term {t!r}")
                    terms.append((int(tm.group(2) or tm.group(3)), emap[int(tm.group(4))],
                                  _r12_f32(float.fromhex(tm.group(5)))))
                outs.append((int(am.group(1)), g, terms))
        parsed.append((outs, style))
    if any(p != parsed[0] for p in parsed):
        raise ValueError("round 12 (24): the part's copies of the TP differ")
    return [(run[0].start(), run[-1].end()) for run in runs], parsed[0][0], parsed[0][1]


def _r12_lit(v):
    return float(v).hex() + "f"


def _r12_plan_part1(outs):
    """c_p per path minimising the FP32 count: per output (terms - 1) + first + 1 (w), plus the shared products."""
    by_g = {}
    for k, g, ts in outs:
        by_g.setdefault(g, []).append((k, ts))
    c_paths = [g for g, os in by_g.items()
               if all(len(ts) == 1 for _, ts in os) and len({ts[0][1] for _, ts in os}) == 1
               and all(abs(ts[0][2]) == 1.0 for _, ts in os)]
    a_paths = [g for g in by_g if g not in c_paths]
    candidates = [sorted({abs(c) for _, ts in by_g[g] for _, _, c in ts}) for g in a_paths]
    best = None
    for combo in itertools.product(*candidates):
        cp = dict(zip(a_paths, combo))
        prods = {(i, j) for g in a_paths for _, ts in by_g[g] for i, j, c in ts if abs(c) != cp[g]}
        ops = len(prods)
        for g in a_paths:
            for _, ts in by_g[g]:
                free = any(abs(c) == cp[g] and (i, j) in prods for i, j, c in ts)
                ops += len(ts) - 1 + (0 if free else 1) + 1
        if best is None or ops < best[0]:
            best = (ops, cp, prods)
    ops, cp, prods = best
    ops += sum(1 + len(by_g[g]) for g in c_paths)
    return ops, cp, prods, c_paths, by_g


def _r12_today_ops(outs):
    """Today's FP32 count: products (shared), per output a first coefficient unless 1.0f, the other terms, w."""
    prods = {(i, j) for _, _, ts in outs for i, j, _ in ts}
    return len(prods) + sum((0 if abs(ts[0][2]) == 1.0 else 1) + len(ts) - 1 + 1 for _, _, ts in outs)


def _r12_fold_part(part, report):
    number = int(_R12_CASE.match(part).group(1))
    runs, outs, (ws, xs) = _r12_parse_runs(part)
    W = (lambda g: f"w[{g}][b]") if ws == "w" else (lambda g: f"wb[{g}]")
    X = (lambda i: f"xv[b][{i}]") if xs == "xv" else (lambda i: f"xc[{i}]")
    ind = "@IND@"  # each copy's own indentation, below
    sh_used = sorted({j for _, _, ts in outs for _, j, _ in ts})
    lines = [ind + "{"] + [f"{ind}const float sh{j} = sh_s[b * ED + {j}];" for j in sh_used]
    if number == 0:
        if not all(len(ts) == 1 and abs(ts[0][2]) == 1.0 and ts[0][0] == 0 for _, _, ts in outs):
            raise ValueError("round 12 (24): part 0 is not four 0 (x) l -> l paths with unit coefficients")
        for g in sorted({g for _, g, _ in outs}):
            lines.append(f"{ind}const float s{g} = {W(g)} * {X(0)};  // 24 (b): w x0 first")
        for k, g, ts in outs:
            i, j, c = ts[0]
            lines.append(f"{ind}acc[{k}] = fmaf({'-' if c < 0 else ''}s{g}, sh{j}, acc[{k}]);")
        ops, cp = 4 + len(outs), None
    else:
        ops, cp, prods, c_paths, by_g = _r12_plan_part1(outs)
        for i, j in sorted(prods):
            lines.append(f"{ind}const float p{i}_{j} = {X(i)} * sh{j};")
        for g in sorted(by_g):
            if g in c_paths:
                j = by_g[g][0][1][0][1]
                lines.append(f"{ind}const float s{g} = {W(g)} * sh{j};  // 24 (c): w e0 first")
                for k, ts in by_g[g]:
                    i, _, c = ts[0]
                    lines.append(f"{ind}acc[{k}] = fmaf({'-' if c < 0 else ''}s{g}, {X(i)}, acc[{k}]);")
                continue
            for k, ts in by_g[g]:
                unit = [(i, j, c) for i, j, c in ts if abs(c) == cp[g]]
                order = ([t for t in unit if (t[0], t[1]) in prods] + [t for t in unit if (t[0], t[1]) not in prods]
                         + [t for t in ts if abs(t[2]) != cp[g]])
                for n, (i, j, c) in enumerate(order):
                    u, neg = abs(c) == cp[g], "-" if c < 0 else ""
                    if n == 0:
                        if u and (i, j) in prods:
                            expr = f"{neg}p{i}_{j}"
                        elif u:
                            expr = f"{neg}{X(i)} * sh{j}"
                        else:
                            expr = f"p{i}_{j} * {_r12_lit(_r12_f32(c / cp[g]))}"
                        lines.append(f"{ind}float t{k} = {expr};")
                    elif u and (i, j) in prods:
                        lines.append(f"{ind}t{k} = t{k} {'-' if c < 0 else '+'} p{i}_{j};")
                    elif u:
                        lines.append(f"{ind}t{k} = fmaf({neg}{X(i)}, sh{j}, t{k});")
                    else:
                        lines.append(f"{ind}t{k} = fmaf(p{i}_{j}, {_r12_lit(_r12_f32(c / cp[g]))}, t{k});")
                lines.append(f"{ind}acc[{k}] = fmaf({W(g)}, t{k}, acc[{k}]);")
    lines.append(ind + "}")
    report.append((number, _r12_today_ops(outs), ops, cp))
    folded = "\n".join(lines) + "\n"
    new = part
    for s, e in reversed(runs):
        new = new[:s] + folded.replace("@IND@", " " * (len(part[s:].split("\n")[0]) - 1)) + new[e:]
    if number == 1:
        paths = int(re.search(r"const int wst\[(\d+)\] = \{", part).group(1))
        cps = [cp.get(g, 1.0) for g in range(paths)]
        decl = re.search(r"^            const int wst\[\d+\] = \{[^}]*\};\n", new, re.M)
        new = (new[:decl.end()] + "            const float cpf[" + str(paths) + "] = {"
               + ", ".join(_r12_lit(c) for c in cps) + "};  // idea 24: c_p per path, into W\n" + new[decl.end():])
        new = _r12_sub(new, "= k < K ? W[(int64_t)(wst[p] + cb * CB + l) * KA + k] : 0.f;",
                       "= k < K ? cpf[p] * W[(int64_t)(wst[p] + cb * CB + l) * KA + k] : 0.f;", 1)
    return new


def _r12_fold(then=None, report=None):
    """Idea 24 on both parts (``then``: 22(iii) first)."""
    def transform(s):
        if then is not None:
            s = then(s)
        pre, parts, post = _r12_parts(s)
        rows = [] if report is None else report
        return pre + "".join(_r12_fold_part(p, rows) for p in parts) + post
    return transform


# ---------------------------------------------------------------------------------------------------------- variants
def round12_variants(kernel=None, report=None):
    """name -> (kind, source transform, extra NVRTC options) for the lab's ``forward_variants``. ``kernel`` is not
    needed (no launch-rule change); ``report`` collects idea 24's FP32 model counts (CPU check)."""
    variants = {
        "iii": ("exact", _r12_iii()),
        "iii-nf": ("exact", _r12_iii(no_flush_path=True)),
        "iii-u1": ("exact", _r12_iii(unroll=1)),
        "iii-br": ("exact", _r12_iii(branch_free=False)),
        "fold": ("rounding", _r12_fold(report=report)),
        "iii-fold": ("rounding", _r12_fold(_r12_iii(), report)),
        "iii-nf-fold": ("rounding", _r12_fold(_r12_iii(no_flush_path=True), report)),
    }
    return {name: (kind, _r12_guard(transform), ()) for name, (kind, transform) in variants.items()}


def _r12_main(argv):
    """``python -I lab_round12.py RENDER.cu OUTDIR``: every variant that applies to RENDER, written to OUTDIR (LF)."""
    import pathlib

    if len(argv) != 2:
        raise SystemExit("usage: python -I lab_round12.py RENDER.cu OUTDIR")
    render = pathlib.Path(argv[0]).read_text()
    outdir = pathlib.Path(argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    for name in round12_variants():
        report = []  # idea 24's FP32 model per part, filled by the rounding variants
        kind, transform, extra = round12_variants(report=report)[name]
        try:
            source = transform(render)
        except ValueError as error:
            print(f"{name}: does not apply: {error}")
            continue
        (outdir / f"{name}.cu").write_text(source, newline="\n")
        model = ""
        if report:
            model = (", FP32 per edge-slice " + " + ".join(str(r[1]) for r in report) + " -> "
                     + " + ".join(str(r[2]) for r in report)
                     + f" ({sum(r[1] for r in report)} -> {sum(r[2] for r in report)})")
            cp = next((r[3] for r in report if r[3]), None)
            if cp:
                model += ", c_p " + ", ".join(f"{g}: {v:.6f}" for g, v in sorted(cp.items()))
        print(f"{name}: {kind}, {len(source.splitlines())} lines{model}")


if __name__ == "__main__":
    _r12_main(sys.argv[1:])
