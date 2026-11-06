"""Two k-loop rewrites of the fused forward's W_GROUP render, from the H200 ideas plan's revision 13 (6 November
2026; ``tools/lab_round13.py``), both bitwise:

- ``A_LD128`` (idea 17(ii)): a is staged so that a lane's four a values of a k-step lie in one float4 (STS.64 at the
  staging, conflict-free), and the k-loop reads them with one LDS.128 instead of four LDS.32. Needs KP = 64, B = 8.
- ``TP_NOFLUSH`` (idea 29): the TP loop gets a copy of its blocks without the flush checks, taken when the batch is
  full and its last receiver is the current row (CSR order: the whole batch adds to row ``cur``).

Each pattern's count is checked; a render of another form raises ValueError (``fused_forward`` then builds without).
"""
import re

from fastmace.kernels.oeq_streaming.cg_fold import _CASE, _constant, _parts


def _sub(source, old, new, count):
    found = source.count(old)
    if found != count:
        raise ValueError(f"kloop_rewrites pattern found {found} times, not {count}: {old[:80]!r}")
    return source.replace(old, new)


def _indent(text, spaces):
    return "".join((" " * spaces + line if line.strip() else line) for line in text.splitlines(keepends=True))


_STAGING = """\
                for (int i = lane; i < B * (KP / 4); i += 32) {
                    const int b = i / (KP / 4), q = i - b * (KP / 4);
                    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (b < nb) v = __ldg(reinterpret_cast<const float4*>(a + (e0 + b) * AS) + q);
                    *reinterpret_cast<float4*>(&a_s[b * KP + ((q ^ b) << 2)]) = v;
                }
"""


_TP_IF = "                if (!tail) {\n"
_TP_HEAD = (_TP_IF + "                    #pragma unroll\n"
                "                    for (int b = 0; b < B; ++b) {\n"
                "                        if (b >= nb) break;\n"
                "                        while (cur < rv[b]) {\n")
_FLUSH_END = "\n                ++cur;\n            }\n"
_LOOP_END = "                    }\n"
_TP_END = _LOOP_END + "                }\n                e0 += B;\n"


_A_READ = """\
                        #pragma unroll
                        for (int t = 0; t < 2; ++t) {
                            const float* at = a_s + (t - gr) * A_STRIDE;
                            split_tf32(at[gid * KP + (((k0 >> 2) ^ gid) << 2) + tq],
                                       bh[t][0], bl[t][0]);
                            split_tf32(at[gid * KP + ((((k0 >> 2) + 1) ^ gid) << 2) + tq],
                                       bh[t][1], bl[t][1]);
                        }
"""
_A_READ_128 = """\
                        {  // A_LD128: the lane's four a values of k-step s in one float4
                            const int s = k0 >> 3;
                            const float4 av = reinterpret_cast<const float4*>(a_s + ((s >> 2) - gr) * A_STRIDE)[
                                (s & 3) * 32 + (lane ^ (s >> 1))];
                            const bool odd = (s & 1) != 0;  // batch t's half: 2 (t ^ (s & 1))
                            split_tf32(odd ? av.z : av.x, bh[0][0], bl[0][0]);
                            split_tf32(odd ? av.w : av.y, bh[0][1], bl[0][1]);
                            split_tf32(odd ? av.x : av.z, bh[1][0], bl[1][0]);
                            split_tf32(odd ? av.y : av.w, bh[1][1], bl[1][1]);
                        }
"""


_A_STAGING_128 = """\
                for (int i = lane; i < B * (KP / 8); i += 32) {  // A_LD128: edge b, k-step s
                    const int b = i >> 3, s = i & 7;
                    float4 v0 = make_float4(0.f, 0.f, 0.f, 0.f), v1 = v0;
                    if (b < nb) {
                        const float4* ab = reinterpret_cast<const float4*>(a + (e0 + b) * AS) + 2 * s;
                        v0 = __ldg(ab);
                        v1 = __ldg(ab + 1);
                    }
                    // k-step s of the pair's two batches in region s >> 2 (warp 0's a_s or warp 1's), lane 4 b + tq's
                    // float4 at (s & 3) * 32 + ((4 b + tq) ^ (s >> 1)), this batch's k-groups 2 s and 2 s + 1 in its
                    // half 2 (gr ^ (s & 1)): conflict-free STS.64 here, LDS.128 in the k-loop.
                    float* pt = a_s + ((s >> 2) - gr) * A_STRIDE + (s & 3) * 128 + 2 * (gr ^ (s & 1));
                    *reinterpret_cast<float2*>(pt + ((4 * b + 0) ^ (s >> 1)) * 4) = make_float2(v0.x, v1.x);
                    *reinterpret_cast<float2*>(pt + ((4 * b + 1) ^ (s >> 1)) * 4) = make_float2(v0.y, v1.y);
                    *reinterpret_cast<float2*>(pt + ((4 * b + 2) ^ (s >> 1)) * 4) = make_float2(v0.z, v1.z);
                    *reinterpret_cast<float2*>(pt + ((4 * b + 3) ^ (s >> 1)) * 4) = make_float2(v0.w, v1.w);
                }
"""


def _tpnf_part(part, number):
    """The TP loop with a copy of its blocks without the flush checks, for full batches inside row cur."""
    if part.count(_TP_HEAD) != 1 or part.count(_TP_END) != 1:
        raise ValueError(f"TP_NOFLUSH: part {number}'s TP loop is not the X_LATE plan's")
    h = part.index(_TP_HEAD)
    f = part.index(_FLUSH_END, h) + len(_FLUSH_END)
    e = part.index(_TP_END, f)
    blocks = part[f:e]
    if re.search(r"\b(cur|while|break|rv)\b", blocks) or "acc[" not in blocks:
        raise ValueError(f"TP_NOFLUSH: part {number}'s TP blocks are not plain blocks")
    loop = part[h + len(_TP_IF):e + len(_LOOP_END)]
    tp = (_TP_IF
          + "                    if (nb == B && rv[B - 1] <= cur) {  // TP_NOFLUSH: a full batch inside row cur "
            "(CSR order), no flush checks\n"
          + "                        #pragma unroll\n"
          + "                        for (int b = 0; b < B; ++b) {\n"
          + _indent(blocks, 4)
          + "                        }\n"
          + "                    } else {\n"
          + _indent(loop, 4)
          + "                    }\n"
          + "                }\n                e0 += B;\n")
    return part[:h] + tp + part[e + len(_TP_END):]


def apply_tp_noflush(source):
    """``TP_NOFLUSH`` on both DEGREE_PARTS parts of an X_LATE render (STORE_W false)."""
    pre, parts, post = _parts(source)
    return pre + "".join(_tpnf_part(p, int(_CASE.match(p).group(1))) for p in parts) + post


def apply_a_ld128(source):
    """``A_LD128`` on a W_GROUP render with VEC_A's a staging (both parts' staging and k-loop reads; not K_PIPE)."""
    if _constant(source, "KP") != 64 or _constant(source, "B") != 8:
        raise ValueError("A_LD128 needs KP = 64 and B = 8 (8 k-steps, 4 lanes per edge)")
    return _sub(_sub(source, _STAGING, _A_STAGING_128, 2), _A_READ, _A_READ_128, 2)
