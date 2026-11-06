"""Round 11 kernel-lab variants of the fused forward (H200 ideas, revision 11: ideas 27 and 26), written as text
transforms of the plan's render for ``tools/omat0_kernel_lab.py``'s ``forward_variants``.

Two ways to use it:
- paste these definitions into ``tools/omat0_kernel_lab.py`` and add ``**round11_variants(kernel)`` to the dict that
  ``forward_variants`` returns (the lab's own ``_RULES`` is found then), or
- import this file and pass the lab's ``_RULES``: ``**round11_variants(kernel, _RULES)``.

Every transform is a chain of exact-text substitutions with checked occurrence counts; a pattern that does not match
raises ValueError, as the lab's ``_sub`` does. All nine variants are exact (bitwise): the same loads, the same products
in the same order per accumulator, and the TP unchanged.

Idea 27, the k-loop's chains (x and its indices out of the registers live across the k-loop):
- ``xlate``: the x gathers, with their index shuffles or loads, moved from the staging to just before the C-write
  barrier;
- ``xlate-kp``: xlate plus kpipe (the next k-step's a values and first W fragment loaded one k-step ahead);
- ``xlate-kpk4``: xlate, kpipe and the k-loop unrolled by 4 instead of 2;
- ``xmid``: the gathers moved to before the k-loop's last two k-steps (the k-loop split at KP - 16);
- ``xmid-kp``: xmid plus kpipe.
These apply to the H200 render (W_GROUP, LANE_INDICES) and to the A100-SXM4's (W_GROUP, WFRAG_REGION, no
LANE_INDICES). On sm_80 the xmid forms spill (64 and 72 B of spill stores at 168 registers; compile only), so the
A100-SXM4 runs the xlate forms (159, 165 and 166 registers, no spill): ``--variants xlate xlate-kp xlate-kpk4``.

Idea 26, the staging one batch ahead (H200 render with LANE_INDICES only). Each warp gets two sh buffers and two
halves of an index buffer after all warps' staging regions: + 1,152 B of shared memory per warp, + 13,824 B at
12 warps, registered as the variant's launch rule (``_RULES``). The A100-SXM4 has about 4 KB left; there the second
sh buffer would have to go into r_s's padding, a form that is not built.
- ``stage-ahead``: after the C-write barrier, a and sh of the next batch of the same chunk are copied by cp.async into
  a_s and the other sh buffer; at the top of each batch the next batch's receivers and senders are copied into the
  other index half; the TP reads its indices by broadcast LDS instead of shuffles;
- ``stage-ahead-noidx``: a and sh only (the indices loaded and shuffled as today);
- ``stage-ahead-xlate-kpk4`` and ``stage-ahead-xmid-kp``: 26 with 27's two best.

CPU check (no GPU, no torch): ``python -I lab_round11.py RENDER.cu OUTDIR`` writes every variant that applies to
RENDER into OUTDIR and says why the others do not apply. On round 9's H200 render (``base.cu``) the outputs are byte
for byte the ``r11_*.cu`` of this folder and the ``r10_sa4*.cu`` of ``h200-ideas-round10-cpu/``.
"""
import dataclasses
import re
import sys


def _r11_sub(source, old, new, count=None):
    """``source`` with ``old`` replaced by ``new``; raises unless it occurs (``count`` times if given)."""
    found = source.count(old)
    if not found or (count is not None and found != count):
        raise ValueError(f"round-11 variant pattern found {found} times: {old[:80]!r}")
    return source.replace(old, new)


# ------------------------------------------------------------------------------------- idea 27: the k-loop's chains
_R11_LOOP = "                    #pragma unroll 2\n                    for (int k0 = 0; k0 < KP; k0 += 8) {\n"
_R11_A_OLD = """                    #pragma unroll 2
                    for (int k0 = 0; k0 < KP; k0 += 8) {
                        unsigned bh[2][2], bl[2][2];  // a^T fragments of batch t: k = k0 + tq (+4), edge = gid
                        #pragma unroll
                        for (int t = 0; t < 2; ++t) {
                            const float* at = a_s + (t - gr) * A_STRIDE;
                            split_tf32(at[gid * KP + (((k0 >> 2) ^ gid) << 2) + tq],
                                       bh[t][0], bl[t][0]);
                            split_tf32(at[gid * KP + ((((k0 >> 2) + 1) ^ gid) << 2) + tq],
                                       bh[t][1], bl[t][1]);
                        }
"""
_R11_A_NEW = """                    float pa[2][2];  // the next k-step's a values, loaded one k-step ahead
                    float4 pw;       // the next k-step's first W fragment (g = 0)
                    #pragma unroll
                    for (int t = 0; t < 2; ++t) {
                        const float* at = a_s + (t - gr) * A_STRIDE;
                        pa[t][0] = at[gid * KP + ((0 ^ gid) << 2) + tq];
                        pa[t][1] = at[gid * KP + ((1 ^ gid) << 2) + tq];
                    }
                    pw = reinterpret_cast<const float4*>(Ws)[gr * 32 + lane];
                    #pragma unroll 2
                    for (int k0 = 0; k0 < KP; k0 += 8) {
                        unsigned bh[2][2], bl[2][2];  // a^T fragments of batch t: k = k0 + tq (+4), edge = gid
                        #pragma unroll
                        for (int t = 0; t < 2; ++t) {
                            split_tf32(pa[t][0], bh[t][0], bl[t][0]);
                            split_tf32(pa[t][1], bh[t][1], bl[t][1]);
                        }
                        const float4 w0 = pw;
                        const int k1 = (k0 + 8 < KP) ? k0 + 8 : k0;  // the last k-step reloads its own (unused)
                        #pragma unroll
                        for (int t = 0; t < 2; ++t) {
                            const float* at = a_s + (t - gr) * A_STRIDE;
                            pa[t][0] = at[gid * KP + (((k1 >> 2) ^ gid) << 2) + tq];
                            pa[t][1] = at[gid * KP + ((((k1 >> 2) + 1) ^ gid) << 2) + tq];
                        }
                        pw = reinterpret_cast<const float4*>(Ws)[((k1 >> 3) * 2 + gr) * 32 + lane];
"""
_R11_W_OLD = """                            const float4 wv = reinterpret_cast<const float4*>(Ws)[
                                ((g * (KP / 8) + (k0 >> 3)) * 2 + gr) * 32 + lane];
"""
_R11_W_NEW = """                            const float4 wv = (g == 0) ? w0 : reinterpret_cast<const float4*>(Ws)[
                                ((g * (KP / 8) + (k0 >> 3)) * 2 + gr) * 32 + lane];
"""
_R11_START = "                float xv[B]"
_R11_RV = "                int rv[B];\n"
_R11_END = "                __syncwarp();\n                if (!group_any(!done, gbar)) break;"
_R11_BAR = "                group_sync(gbar);  // the group's r_s rows written\n"


def _r11_group(s):
    if s.count(_R11_START) != 2 or s.count(_R11_END) != 2 or s.count(_R11_BAR) != 2:
        raise ValueError("round-11 variants need the W_GROUP render with both parts (fused_forward.W_GROUP; found "
                         f"{s.count(_R11_START)} x loads, {s.count(_R11_END)} votes, {s.count(_R11_BAR)} C barriers)")


def _r11_kpipe(s):
    """The next k-step's a values and first W fragment loaded one k-step ahead (8 floats across the back-edge)."""
    return _r11_sub(_r11_sub(s, _R11_A_OLD, _R11_A_NEW, 2), _R11_W_OLD, _R11_W_NEW, 2)


def _r11_ku4(s):
    """Both k-loops unrolled by 4 instead of 2 (apply after kpipe, which keeps the loop header)."""
    return _r11_sub(s, _R11_LOOP, _R11_LOOP.replace("unroll 2", "unroll 4"), 2)


def _r11_xlate(s):
    """Both parts' x gathers (from ``float xv[B]`` to the group vote) moved to just before the C-write barrier."""
    _r11_group(s)
    out, pos = s, 0
    for _ in range(2):
        i = out.index(_R11_START, pos)
        j = out.index(_R11_END, i)
        b = out.index(_R11_BAR, j)
        out = out[:i] + out[j:b] + out[i:j] + out[b:]
        pos = b + len(_R11_BAR)
    return out


def _r11_close(s, open_index):
    depth = 0
    for i in range(open_index, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("round-11 xmid: unbalanced braces after the k-loop")


def _r11_xmid(s):
    """Both parts' x gathers (after the ``xv`` and ``rv`` declarations, which stay) moved to before the k-loop's last
    two k-steps; the k-loop split at KP - 16 into two loops with the same body."""
    _r11_group(s)
    if s.count(_R11_LOOP) != 2:
        raise ValueError(f"round-11 xmid: the k-loop header found {s.count(_R11_LOOP)} times, not 2")
    out, pos = s, 0
    for _ in range(2):
        i = out.index(_R11_RV, out.index(_R11_START, pos)) + len(_R11_RV)
        j = out.index(_R11_END, i)
        gather = out[i:j]
        out = out[:i] + out[j:]
        h = out.index(_R11_LOOP, i)
        opening = h + len(_R11_LOOP) - 2
        closing = _r11_close(out, opening)
        body = out[opening:closing + 1]
        first = _R11_LOOP.replace("k0 < KP;", "k0 < KP - 16;")
        second = _R11_LOOP.replace("int k0 = 0; k0 < KP;", "int k0 = KP - 16; k0 < KP;")
        new = first[:-2] + body + "\n" + gather + second[:-2] + body
        out = out[:h] + new + out[closing + 1:]
        pos = h + len(new)
    return out


# ---------------------------------------------------------------------------- idea 26: the staging one batch ahead
_R11_HELPERS = r'''
// Idea 26 (stage-ahead): cp.async copies (16 B .cg for a and sh, 4 B .ca for the indices), zero-filled when off.
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool on) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d), "l"(src), "r"(on ? 16 : 0) : "memory");
}
__device__ __forceinline__ void cp_async4(void* dst, const void* src, bool on) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(d), "l"(src), "r"(on ? 4 : 0) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
'''
_R11_IDX_LINES = ("                const int lane_rv = (lane < nb && !tail) ? receivers[e0 + lane] : 0x7fffffff;\n"
                  "                const int lane_snd = (lane < nb && !tail) ? senders[e0 + lane] : -1;\n")
_R11_STAGE_OLD = """                __syncwarp();
                for (int i = lane; i < B * (KP / 4); i += 32) {
                    const int b = i / (KP / 4), q = i - b * (KP / 4);
                    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (b < nb) v = __ldg(reinterpret_cast<const float4*>(a + (e0 + b) * AS) + q);
                    *reinterpret_cast<float4*>(&a_s[b * KP + ((q ^ b) << 2)]) = v;
                }
                for (int i = lane; i < B * ED / 4; i += 32)
                    reinterpret_cast<float4*>(sh_s)[i] = (4 * i / ED < nb)
                        ? __ldg(reinterpret_cast<const float4*>(sh + e0 * ED) + i) : make_float4(0.f, 0.f, 0.f, 0.f);
""" + _R11_IDX_LINES
_R11_DECL_OLD = "    float* r_s = sh_s + B * ED;  // tensor cores: the warp's redistribution tile [paths][B][RS]\n"
_R11_DECL_NEW = _R11_DECL_OLD + """\
    // Idea 26: per warp two sh buffers and two halves of receivers and senders, chosen by a parity bit.
    float* const xb = smem + A_BASE + (blockDim.x >> 5) * A_STRIDE + warp * (2 * B * ED + 4 * B);
    int* const ib = reinterpret_cast<int*>(xb + 2 * B * ED);
    int* idx_s = ib;
"""
_R11_LOOP_OLD = "            bool have = false, done = false, tail = false;\n"
_R11_LOOP_NEW = "            bool have = false, done = false, tail = false, staged = false;\n            int par = 0;\n"
_R11_SH_NEXT = "(xb + (par ^ 1) * (B * ED))"


def _r11_copies(e1, nb1, shb, ind):
    """cp.async of a batch's a tile (into a_s, swizzled as today's staging) and its sh (into ``shb``)."""
    return (f"{ind}for (int i = lane; i < B * (KP / 4); i += 32) {{\n"
            f"{ind}    const int b = i / (KP / 4), q = i - b * (KP / 4);\n"
            f"{ind}    cp_async16(&a_s[b * KP + ((q ^ b) << 2)],\n"
            f"{ind}               reinterpret_cast<const float4*>(a + ({e1} + (b < {nb1} ? b : 0)) * AS) + q, "
            f"b < {nb1});\n"
            f"{ind}}}\n"
            f"{ind}for (int i = lane; i < B * ED / 4; i += 32)\n"
            f"{ind}    cp_async16({shb} + 4 * i, reinterpret_cast<const float4*>(sh + {e1} * ED) + "
            f"(4 * i / ED < {nb1} ? i : 0),\n"
            f"{ind}               4 * i / ED < {nb1});\n")


def _r11_idx_copy(e1, nb1, dst, ind):
    """cp.async of a batch's receivers (lanes 0..B-1) and senders (lanes B..2B-1) into ``dst``."""
    return (f"{ind}if (lane < 2 * B) {{\n"
            f"{ind}    const int j = lane & (B - 1);\n"
            f"{ind}    cp_async4({dst} + lane, (lane < B ? receivers : senders) + {e1} + (j < {nb1} ? j : 0),\n"
            f"{ind}              j < {nb1} && !tail);\n"
            f"{ind}}}\n")


def _r11_stage_new(idx):
    ind = " " * 20
    s = ("                __syncwarp();\n"
         "                {  // idea 26: this batch's copies, unless the previous batch issued them; then wait\n"
         "                    par ^= 1;\n"
         "                    sh_s = xb + par * (B * ED);\n")
    if idx:
        s += "                    idx_s = ib + par * (2 * B);\n"
    s += ("                    if (!staged) {\n"
          "                    const int64_t es = nb > 0 ? e0 : 0;\n"
          + _r11_copies("es", "nb", "sh_s", ind))
    if idx:
        s += _r11_idx_copy("es", "nb", "idx_s", ind)
    s += (f"{ind}cp_async_commit();\n"
          "                    }\n"
          "                    staged = false;\n"
          "                    cp_async_wait_all();\n"
          "                    __syncwarp();\n")
    if idx:
        s += ("                    if (!done && e0 + B < eend) {  // the next batch's indices now, its a and sh after "
              "the k-loop\n"
              "                        const int64_t e1 = e0 + B;\n"
              "                        const int nb1 = (int)((eend - e1) < B ? (eend - e1) : B);\n"
              + _r11_idx_copy("e1", "nb1", "(ib + (par ^ 1) * (2 * B))", " " * 24)
              + "                        cp_async_commit();\n"
              "                    }\n")
    s += "                }\n"
    if not idx:
        s += _R11_IDX_LINES
    return s


def _r11_prefetch():
    ind = " " * 20
    return ("                group_sync(gbar);  // the group's r_s rows written\n"
            "                if (!done && e0 + B < eend) {  // idea 26: the next batch is in this chunk; a_s is free\n"
            "                    const int64_t e1 = e0 + B;\n"
            "                    const int nb1 = (int)((eend - e1) < B ? (eend - e1) : B);\n"
            + _r11_copies("e1", "nb1", _R11_SH_NEXT, ind)
            + f"{ind}cp_async_commit();\n"
            "                    staged = true;\n"
            "                }\n")


def _r11_stage_ahead_text(s, idx):
    """Idea 26 on the render's text (``idx``: the indices too); the launch rule is registered by the caller."""
    _r11_group(s)
    if s.count("rv[b] = __shfl_sync(FULL, lane_rv, b);") != 2:
        raise ValueError("stage-ahead needs the render with LANE_INDICES and W_GROUP (the H200 plan); the A100-SXM4's "
                         "has no LANE_INDICES and too little shared memory (its r_s-padding form is not built)")
    s = _r11_sub(s, 'extern "C" __global__', _R11_HELPERS + '\nextern "C" __global__', 1)
    s = _r11_sub(s, _R11_DECL_OLD, _R11_DECL_NEW, 1)
    s = _r11_sub(s, _R11_LOOP_OLD, _R11_LOOP_NEW, 2)
    s = _r11_sub(s, _R11_STAGE_OLD, _r11_stage_new(idx), 2)
    if idx:
        s = _r11_sub(s, "rv[b] = __shfl_sync(FULL, lane_rv, b);",
                     "rv[b] = (b < nb && !tail) ? idx_s[b] : 0x7fffffff;", 2)
        s = _r11_sub(s, "int snd = __shfl_sync(FULL, lane_snd, b);",
                     "int snd = (b < nb && !tail) ? idx_s[B + b] : -1;", 2)
    return _r11_sub(s, _R11_BAR, _r11_prefetch(), 2)


def _r11_constant(source, name):
    match = re.search(rf"constexpr int {name} = (\d+);", source)
    if match is None:
        raise ValueError(f"stage-ahead: constant {name} not found in the render")
    return int(match.group(1))


def _r11_stage_ahead_bytes(source, warps):
    """The plan's shared memory (W region and staging, A_BASE + warps * A_STRIDE floats) plus 26's per-warp region
    (two sh buffers of B * ED floats and two index halves of 2 * B ints)."""
    a_base, a_stride = _r11_constant(source, "A_BASE"), _r11_constant(source, "A_STRIDE")
    b, ed = _r11_constant(source, "B"), _r11_constant(source, "ED")
    return 4 * (a_base + warps * a_stride) + warps * 4 * (2 * b * ed + 4 * b)


def _r11_optin():
    """This GPU's opt-in shared memory per block, or None without torch or CUDA (CPU checks)."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.get_device_properties(0).shared_memory_per_block_optin)


def _r11_rules(rules):
    """The lab's ``_RULES`` (a variant's source -> its launch rule): given, pasted into the lab, or the lab's module."""
    if rules is not None:
        return rules
    table = globals().get("_RULES")
    for name in ("__main__", "omat0_kernel_lab"):
        if table is None and isinstance(getattr(sys.modules.get(name), "_RULES", None), dict):
            table = sys.modules[name]._RULES
    if table is None:
        raise ValueError("stage-ahead needs the lab's _RULES: round11_variants(kernel, _RULES)")
    return table


def _r11_stage_ahead(kernel, rules, idx, then=None):
    """Idea 26 (``then``: idea 27's moves on top), with the larger shared request registered as the launch rule."""
    def transform(s):
        source = _r11_stage_ahead_text(s, idx)
        if then is not None:
            source = then(source)
        need = _r11_stage_ahead_bytes(source, kernel.rule.warps)
        optin = _r11_optin()
        if optin is not None and need > optin:
            raise ValueError(f"stage-ahead needs {need} B of shared memory per block; this GPU allows {optin} B (the "
                             "A100-SXM4 would need the r_s-padding form, not built)")
        table = _r11_rules(rules)
        table[source] = dataclasses.replace(kernel.rule, shared_bytes=max(need, kernel.rule.shared_bytes))
        print(f"stage-ahead rule {table[source]} (plan {kernel.rule})", flush=True)
        return source
    return transform


def _r11_chain(*steps):
    def transform(s):
        for step in steps:
            s = step(s)
        return s
    return transform


def round11_variants(kernel=None, rules=None):
    """name -> (kind, source transform, extra NVRTC options) for the lab's ``forward_variants``. ``kernel`` (the
    plan's FusedFoldedForward) and the lab's ``_RULES`` are needed by the stage-ahead variants only, when they run."""
    return {
        "xlate": ("exact", _r11_xlate, ()),
        "xlate-kp": ("exact", _r11_chain(_r11_kpipe, _r11_xlate), ()),
        "xlate-kpk4": ("exact", _r11_chain(_r11_kpipe, _r11_ku4, _r11_xlate), ()),
        "xmid": ("exact", _r11_xmid, ()),
        "xmid-kp": ("exact", _r11_chain(_r11_kpipe, _r11_xmid), ()),
        "stage-ahead": ("exact", _r11_stage_ahead(kernel, rules, True), ()),
        "stage-ahead-noidx": ("exact", _r11_stage_ahead(kernel, rules, False), ()),
        "stage-ahead-xlate-kpk4": ("exact", _r11_stage_ahead(kernel, rules, True,
                                                             _r11_chain(_r11_xlate, _r11_kpipe, _r11_ku4)), ()),
        "stage-ahead-xmid-kp": ("exact", _r11_stage_ahead(kernel, rules, True, _r11_chain(_r11_kpipe, _r11_xmid)), ()),
    }


def _r11_main(argv):
    """``python -I lab_round11.py RENDER.cu OUTDIR``: every variant that applies to RENDER, written to OUTDIR (LF)."""
    import pathlib
    import types

    if len(argv) != 2:
        raise SystemExit("usage: python -I lab_round11.py RENDER.cu OUTDIR")
    render = pathlib.Path(argv[0]).read_text()
    outdir = pathlib.Path(argv[1])
    outdir.mkdir(parents=True, exist_ok=True)

    @dataclasses.dataclass(frozen=True)
    class Rule:  # the fields of the lab's launch rule that the variants read
        warps: int
        shared_bytes: int

    warps = int(re.search(r"__launch_bounds__\((\d+), 1\)", render).group(1)) // 32
    plan = 4 * (_r11_constant(render, "A_BASE") + warps * _r11_constant(render, "A_STRIDE"))
    kernel = types.SimpleNamespace(rule=Rule(warps=warps, shared_bytes=plan))
    rules = {}
    for name, (kind, transform, extra) in round11_variants(kernel, rules).items():
        try:
            source = transform(render)
        except ValueError as error:
            print(f"{name}: does not apply: {error}")
            continue
        (outdir / f"{name}.cu").write_text(source, newline="\n")
        rule = rules.get(source)
        print(f"{name}: {kind}, {len(source.splitlines())} lines, shared "
              f"{rule.shared_bytes if rule else plan} B (plan {plan} B at {warps} warps)")


if __name__ == "__main__":
    _r11_main(sys.argv[1:])
