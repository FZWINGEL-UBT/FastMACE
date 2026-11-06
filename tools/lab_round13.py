"""Round 13 kernel-lab variants of the fused forward (H200 ideas, revision 13): what is left after round 12 rejected
22(iii) and ported idea 24 as ``fused_forward.CG_FOLD`` (lab ``fold``: H200 0.969, A100-SXM4 0.951). Text transforms
of the plan's render for ``tools/omat0_kernel_lab.py``'s ``forward_variants``, in round 12's format: every pattern's
count is checked (ValueError otherwise, as the lab's ``_sub``), and every variant keeps the text the lab's
forward_launcher reads as VEC_A. Written for the plan's render with X_LATE (both GPUs since FastMACE ``fc0d7531``)
and without K_PIPE, with or without CG_FOLD (on both GPUs since ``1388c3a1``). On a CG_FOLD base the fold-*
variants work on the base's own folded TP (round 12's ``fold`` no longer applies there) and rowfold and rcfold
re-render the plan kernel without CG_FOLD, accepted only if folding it again gives the base byte for byte;
``--constant fastmace.kernels.oeq_streaming.fused_forward:CG_FOLD=False`` gives round 12's unfolded base instead.

Hook, as round 12's (``tools/omat0_kernel_lab.py``; this file imports ``lab_round12``):

    def _round13_variants(kernel):
        import lab_round13  # tools/ is on sys.path when the lab runs as a script
        FORWARD_REFERENCES.update(lab_round13.round13_references(kernel))
        return lab_round13.round13_variants(kernel)

and ``**_round13_variants(kernel),`` after ``**_round12_variants(kernel),`` in the dict ``forward_variants``
returns. ``round13_references`` maps a variant to the variant it must match bitwise (the lab runs that one first)
instead of the base: rcfold-tpnf and rc-all match rcfold, and on an unfolded base the fold-* variants match fold (on
a CG_FOLD base they match the base). No variant changes the launch rule or the shared memory.

Fold further (rounding; compared with the base). Part 1's TP (the paths 1 (x) l -> l') per edge-slice, in FP32
instructions: today 137, ``fold`` 113 (round 12: one factor c_p per path, into W).
- ``rowfold``: one factor F_k per accumulator instead of one per path, applied once per row when the flush stores it
  (``out = acc * F_k``; the W fill keeps W as it is). 24 accumulators instead of 6 paths make more terms units: 100.
- ``rcfold``: ``rowfold`` with a factor S_j per sh component as well, applied once per batch in part 1's sh staging
  (sh_s holds S_j sh_j; part 1's TP is its only reader): 92. The S_j make as many terms as possible consistent
  (F_k S_j = |c|; a seeded search over spanning forests of the term graph), the F_k follow by a descent on the
  FP32 count. A term counts as a unit within ``_R13_TOL`` of 1; the CPU check reports every coefficient's
  deviation from today's.
Part 0 (0 (x) l -> l, unit coefficients) keeps round 12's w x0 first in both.

The registers the fold freed on the H200 (161 -> 154 in the lab), for scheduling (exact; compared with their fold):
- ``fold-tpnf``, ``rcfold-tpnf``: a copy of the TP loop without the flush checks, taken when the batch is full and
  its last receiver is the current row (CSR order: then every edge adds to row cur): its eight folded blocks form
  one basic block that ptxas can interleave;
- ``fold-kp``: ``fold`` on the K_PIPE render (the next k-step's a values and first W fragment loaded one k-step
  ahead; ``fused_forward.apply_k_pipe``'s text, copied here; the CPU check compares it with fused_forward's render).
The folded TP's sh reads as float4 loads (SH_LDS4's form) are not a variant: ptxas 13.3 already reads them with
LDS.128 (that form built the same SASS on sm_80 and the same instructions in 13 other places on sm_90).

Idea 17(ii), the a path's shared loads:
- ``a-ld128`` (exact; compared with the base) and ``fold-a-ld128`` (exact; with fold): the a staging stores each
  k-step's four a values of a lane (both batches of the warp pair, k-groups 2 s and 2 s + 1 of k-step s) as one
  float4, so the k-loop reads them with one LDS.128 instead of four LDS.32. Each warp writes its batch's half into
  both warps' regions with STS.64 (warp gr's half at 2 (gr ^ (s & 1)) of the float4); the pair's barriers already
  order every write after the previous k-loop's reads. Stores and loads are free of bank conflicts (CPU emulation).
- ``F-a-ld128`` (falsifier): the four LDS.32 replaced by one LDS.128 from the plan's layout (wrong values): the read
  side's share alone.
- ``rc-all`` (exact; compared with rcfold): rcfold with tpnf and a-ld128.

CPU check (no GPU, no torch): ``python -I -B lab_round13.py RENDER.cu OUTDIR`` writes every variant that applies to
RENDER into OUTDIR (LF) and says why the others do not apply.
"""
import math
import random
import re
import sys

try:
    import lab_round12 as _r12
except ImportError:  # python -I: lab_round12.py next to this file (tools/), or in round 12's notes folder
    import pathlib

    _here = pathlib.Path(__file__).resolve().parent
    for _d in (_here, _here.parent / "h200-ideas-round12-cpu"):
        if (_d / "lab_round12.py").is_file():
            sys.path.insert(0, str(_d))
            break
    import lab_round12 as _r12

_R13_VEC_A = _r12._R12_VEC_A


def _r13_sub(source, old, new, count):
    """``source`` with ``old`` replaced by ``new``; raises unless it occurs exactly ``count`` times."""
    found = source.count(old)
    if found != count:
        raise ValueError(f"round-13 variant pattern found {found} times, not {count}: {old[:80]!r}")
    return source.replace(old, new)


def _r13_guard(transform):
    """``transform``, refusing a result that lost the render's VEC_A text."""
    def guarded(s):
        out = transform(s)
        if _R13_VEC_A in s and _R13_VEC_A not in out:
            raise ValueError("round-13 variant lost the text the lab's forward_launcher reads as VEC_A "
                             f"({_R13_VEC_A}); the lab would pass a compacted a")
        return out
    return guarded


def _r13_then(*transforms):
    def transform(s):
        for t in transforms:
            s = t(s)
        return s
    return transform


def _r13_per_part(fn):
    """The transform applying ``fn(part, number)`` to each of the render's two parts (``_r12_parts``' checks)."""
    def transform(s):
        pre, parts, post = _r12._r12_parts(s)
        return pre + "".join(fn(p, int(_r12._R12_CASE.match(p).group(1))) for p in parts) + post
    return transform


# ----------------------------------------------------------------------------------- fold further: rowfold, rcfold
# A scaled coefficient within _R13_TOL of +-1 is a unit (dropped). Coefficients that are equal in exact arithmetic
# differ by their float32 roundings (a few 1e-8), and every other magnitude in OMAT-0's part 1 is more than 1e-3 away.
_R13_TOL = 2.0 ** -21


def _r13_unit(c):
    return abs(abs(c) - 1.0) <= _R13_TOL


def _r13_special(outs):
    """Paths whose outputs are single unit terms on one sh component: w times that component first (1 + outputs)."""
    by_g = {}
    for k, g, ts in outs:
        by_g.setdefault(g, []).append(ts)
    return tuple(sorted(g for g, tss in by_g.items()
                        if all(len(ts) == 1 and _r13_unit(ts[0][2]) for ts in tss)
                        and len({ts[0][1] for ts in tss}) == 1))


def _r13_products(outs, special):
    """The shared products x_i sh_j: every non-unit term's, then, greedily, a unit term's while it gives a free first
    term to two outputs or more (ties: the smallest (i, j))."""
    P = {(i, j) for k, g, ts in outs if g not in special for i, j, c in ts if not _r13_unit(c)}
    need = {k: {(i, j) for i, j, c in ts if _r13_unit(c)} for k, g, ts in outs
            if g not in special and not any(_r13_unit(c) and (i, j) in P for i, j, c in ts)}
    while True:
        gains = {}
        for cand in need.values():
            for q in cand:
                gains[q] = gains.get(q, 0) + 1
        if not gains:
            return P
        q, gain = max(sorted(gains.items()), key=lambda kv: kv[1])
        if gain < 2:
            return P
        P.add(q)
        need = {k: c for k, c in need.items() if q not in c}


def _r13_cost(outs):
    """(FP32 per edge-slice, special paths, products) of part 1's TP with the scaled terms ``outs``: the products; per
    output its terms and the w FFMA, plus one for its first term unless that is a unit term with a shared product;
    a special path 1 + its outputs."""
    special = _r13_special(outs)
    P = _r13_products(outs, special)
    ops = len(P)
    for k, g, ts in outs:
        if g in special:
            continue
        ops += len(ts) + (0 if any(_r13_unit(c) and (i, j) in P for i, j, c in ts) else 1)
    for g in special:
        ops += 1 + sum(1 for _, gg, _ in outs if gg == g)
    return ops, special, P


def _r13_scaled(outs, rows, cols):
    """The terms with c' = c / (F_k S_j), the factors as the kernel applies them (float32)."""
    return [(k, g, [(i, j, c / (rows.get(k, 1.0) * cols.get(j, 1.0))) for i, j, c in ts]) for k, g, ts in outs]


def _r13_rows(outs, cols):
    """Row factors F_k (float32): each output's from its terms' magnitudes |c| / S_j, starting from the one most of
    its terms share (ties: the largest), then the best single change while one lowers the FP32 count. Returns
    (factors, count)."""
    f32 = _r12._r12_f32
    cand = {}
    for k, g, ts in outs:
        tally = {}
        for i, j, c in ts:
            m = f32(abs(c) / cols.get(j, 1.0))
            tally[m] = tally.get(m, 0) + 1
        cand[k] = sorted(tally, key=lambda m: (-tally[m], -m))
    rows = {k: v[0] for k, v in cand.items()}
    best = _r13_cost(_r13_scaled(outs, rows, cols))[0]
    while True:
        move = None
        for k in sorted(cand):
            for m in cand[k]:
                if m != rows[k]:
                    trial = dict(rows)
                    trial[k] = m
                    ops = _r13_cost(_r13_scaled(outs, trial, cols))[0]
                    if ops < best:
                        best, move = ops, (k, m)
        if move is None:
            return rows, best
        rows[move[0]] = move[1]


def _r13_potentials(order):
    """A weighted union-find over the terms (k, j, log |c|) in ``order``: u(k) - u(j) = log |c| along a spanning
    forest; returns (terms consistent with it, u)."""
    parent, pot = {}, {}

    def find(n):
        parent.setdefault(n, n)
        pot.setdefault(n, 0.0)
        if parent[n] == n:
            return n
        r = find(parent[n])
        if parent[n] != r:
            pot[n] += pot[parent[n]]
            parent[n] = r
        return r

    sat = 0
    for a, b, L in order:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            pot[rb] = pot[a] - pot[b] - L
            sat += 1
        elif abs(pot[a] - pot[b] - L) <= _R13_TOL:
            sat += 1
    for n in list(parent):
        find(n)
    return sat, {n: (pot[n] if parent[n] != n else 0.0) for n in parent}, {n: find(n) for n in parent}


def _r13_cols(outs, trials=2000, seed=13):
    """Column factors S_j (float32): u(k) = log F_k and u(j) = -log S_j with F_k S_j = |c| on as many terms as a
    seeded random search over spanning forests finds (u is free per connected component: each is scaled so that its
    smallest sh component has S = 1)."""
    edges = [(("k", k), ("j", j), math.log(abs(c))) for k, g, ts in outs for i, j, c in ts]
    rng = random.Random(seed)
    best = None
    for _ in range(trials):
        order = edges[:]
        rng.shuffle(order)
        sat, u, root = _r13_potentials(order)
        if best is None or sat > best[0]:
            best = (sat, u, root)
    sat, u, root = best
    lead = {}
    for (kind, j), r in sorted(root.items()):
        if kind == "j":
            lead.setdefault(r, j)
    return {j: _r12._r12_f32(math.exp(u[("j", lead[root[("j", j)]])] - u[("j", j)]))
            for (kind, j) in u if kind == "j"}


_R13_PLANS = {}


def _r13_plan(outs, columns):
    """(rows, cols, scaled terms, special paths, products, FP32 count) of part 1, memoized per TP."""
    key = (tuple((k, g, tuple(ts)) for k, g, ts in outs), columns)
    if key not in _R13_PLANS:
        cols = _r13_cols(outs) if columns else {}
        rows, ops = _r13_rows(outs, cols)
        scaled = _r13_scaled(outs, rows, cols)
        count, special, P = _r13_cost(scaled)
        if count != ops:
            raise ValueError("round 13: the row search's count is not the plan's")
        _R13_PLANS[key] = (rows, cols, scaled, special, P, ops)
    return _R13_PLANS[key]


def _r13_lit(v):
    return _r12._r12_lit(_r12._r12_f32(v))


def _r13_rc_part(part, columns, report):
    """Part 1 with its TP blocks replaced by the row- (and column-) scaled form, the flush stores times F_k and, with
    ``columns``, the sh staging times S_j."""
    runs, outs, (ws, xs) = _r12._r12_parse_runs(part)
    if len({k for k, _, _ in outs}) != len(outs):
        raise ValueError("round 13: an accumulator takes several TP lines; row factors need one line per accumulator")
    rows, cols, scaled, special, P, ops = _r13_plan(outs, columns)
    W = (lambda g: f"w[{g}][b]") if ws == "w" else (lambda g: f"wb[{g}]")
    X = (lambda i: f"xv[b][{i}]") if xs == "xv" else (lambda i: f"xc[{i}]")
    ind = "@IND@"  # each copy's own indentation, below
    lines = [ind + "{"]
    lines += [f"{ind}const float sh{j} = sh_s[b * ED + {j}];" for j in sorted({j for _, _, ts in outs for _, j, _ in ts})]
    lines += [f"{ind}const float p{i}_{j} = {X(i)} * sh{j};" for i, j in sorted(P)]
    by_g = {}
    for k, g, ts in scaled:
        by_g.setdefault(g, []).append((k, ts))
    for g in sorted(by_g):
        if g in special:
            j = by_g[g][0][1][0][1]
            lines.append(f"{ind}const float s{g} = {W(g)} * sh{j};  // w e first")
            for k, ts in by_g[g]:
                i, _, c = ts[0]
                lines.append(f"{ind}acc[{k}] = fmaf({'-' if c < 0 else ''}s{g}, {X(i)}, acc[{k}]);")
            continue
        for k, ts in by_g[g]:
            unit = [t for t in ts if _r13_unit(t[2])]
            order = ([t for t in unit if t[:2] in P] + [t for t in unit if t[:2] not in P]
                     + [t for t in ts if not _r13_unit(t[2])])
            for n, (i, j, c) in enumerate(order):
                u, neg = _r13_unit(c), "-" if c < 0 else ""
                if n == 0:
                    expr = (f"{neg}p{i}_{j}" if u and (i, j) in P else f"{neg}{X(i)} * sh{j}" if u
                            else f"p{i}_{j} * {_r13_lit(c)}")
                    lines.append(f"{ind}float t{k} = {expr};")
                elif u and (i, j) in P:
                    lines.append(f"{ind}t{k} = t{k} {'-' if c < 0 else '+'} p{i}_{j};")
                elif u:
                    lines.append(f"{ind}t{k} = fmaf({neg}{X(i)}, sh{j}, t{k});")
                else:
                    lines.append(f"{ind}t{k} = fmaf(p{i}_{j}, {_r13_lit(c)}, t{k});")
            lines.append(f"{ind}acc[{k}] = fmaf({W(g)}, t{k}, acc[{k}]);")
    lines.append(ind + "}")
    folded = "\n".join(lines) + "\n"
    new = part
    for s, e in reversed(runs):
        new = new[:s] + folded.replace("@IND@", " " * (len(part[s:].split("\n")[0]) - 1)) + new[e:]
    ks = sorted(k for k, _, _ in outs)
    counts = {k: new.count(f"= acc[{k}];\n") for k in ks}
    flushes = counts[ks[0]]
    if not flushes or any(v != flushes for v in counts.values()):
        raise ValueError(f"round 13: part 1's flush stores per accumulator differ: {counts}")
    for k in ks:
        if rows[k] != 1.0:
            new = _r13_sub(new, f"= acc[{k}];\n", f"= acc[{k}] * {_r13_lit(rows[k])};  // F_k\n", flushes)
    if columns:
        quads =["make_float4(" + ", ".join(_r13_lit(cols.get(4 * q + c, 1.0)) for c in range(4)) + ")"
                 for q in range(4)]
        staging = (
            "                for (int i = lane; i < B * ED / 4; i += 32) {  // round 13 (rcfold): sh_j times S_j\n"
            "                    const int q = i & 3;  // the float4's place in its edge's row (ED = 16)\n"
            f"                    const float4 f = q == 0 ? {quads[0]}\n"
            f"                                   : q == 1 ? {quads[1]}\n"
            f"                                   : q == 2 ? {quads[2]}\n"
            f"                                   : {quads[3]};\n"
            "                    const float4 v = (4 * i / ED < nb)\n"
            "                        ? __ldg(reinterpret_cast<const float4*>(sh + e0 * ED) + i) : "
            "make_float4(0.f, 0.f, 0.f, 0.f);\n"
            "                    reinterpret_cast<float4*>(sh_s)[i] = make_float4(v.x * f.x, v.y * f.y, v.z * f.z, "
            "v.w * f.w);\n"
            "                }\n")
        new = _r13_sub(new, _r12._R12_SH_STAGING, staging, 1)
    units = sum(1 for _, _, ts in scaled for _, _, c in ts if _r13_unit(c))
    report.append((1, _r12._r12_today_ops(outs), ops,
                   dict(rows={k: v for k, v in rows.items() if v != 1.0}, cols={j: v for j, v in cols.items() if v != 1.0},
                        special=special, products=len(P), units=units, terms=sum(len(ts) for _, _, ts in outs))))
    return new


def _r13_rcfold(columns, report=None):
    """rowfold (``columns`` false) or rcfold: part 1 as ``_r13_rc_part``, part 0 as round 12's fold."""
    def fn(part, number):
        rows = [] if report is None else report
        if number == 0:
            return _r12._r12_fold_part(part, rows)
        if number != 1:
            raise ValueError(f"round 13: unexpected part {number}")
        return _r13_rc_part(part, columns, rows)

    per_part = _r13_per_part(fn)

    def transform(s):
        if columns and _r12._r12_constant(s, "ED") != 16:
            raise ValueError("round 13 (rcfold) needs ED = 16 (four sh float4 per edge in the staging)")
        return per_part(s)
    return transform


# ------------------------------------------------------------------------------------- the freed registers: tpnf
_R13_TP_IF = "                if (!tail) {\n"
_R13_TP_HEAD = (_R13_TP_IF + "                    #pragma unroll\n"
                "                    for (int b = 0; b < B; ++b) {\n"
                "                        if (b >= nb) break;\n"
                "                        while (cur < rv[b]) {\n")
_R13_FLUSH_END = "\n                ++cur;\n            }\n"
_R13_LOOP_END = "                    }\n"
_R13_TP_END = _R13_LOOP_END + "                }\n                e0 += B;\n"


def _r13_tpnf_part(part, number):
    """The TP loop with a copy of its blocks without the flush checks, for full batches inside row cur."""
    if part.count(_R13_TP_HEAD) != 1 or part.count(_R13_TP_END) != 1:
        raise ValueError(f"round 13 (tpnf): part {number}'s TP loop is not the X_LATE plan's")
    h = part.index(_R13_TP_HEAD)
    f = part.index(_R13_FLUSH_END, h) + len(_R13_FLUSH_END)
    e = part.index(_R13_TP_END, f)
    blocks = part[f:e]
    if re.search(r"\b(cur|while|break|rv)\b", blocks) or "acc[" not in blocks:
        raise ValueError(f"round 13 (tpnf): part {number}'s TP blocks are not plain blocks")
    loop = part[h + len(_R13_TP_IF):e + len(_R13_LOOP_END)]
    tp = (_R13_TP_IF
          + "                    if (nb == B && rv[B - 1] <= cur) {  // round 13 (tpnf): a full batch inside row cur "
            "(CSR order), no flush checks\n"
          + "                        #pragma unroll\n"
          + "                        for (int b = 0; b < B; ++b) {\n"
          + _r12._r12_indent(blocks, 4)
          + "                        }\n"
          + "                    } else {\n"
          + _r12._r12_indent(loop, 4)
          + "                    }\n"
          + "                }\n                e0 += B;\n")
    return part[:h] + tp + part[e + len(_R13_TP_END):]


# ------------------------------------------------------------------------------- the freed registers: fold-kp
# fused_forward's K_PIPE (apply_k_pipe), verbatim: the next k-step's a values and first W fragment one k-step ahead.
_R13_KP_A_OLD = """                    #pragma unroll 2
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
_R13_KP_A_NEW = """                    float pa[2][2];  // K_PIPE: the next k-step's a values, loaded one k-step ahead
                    float4 pw;       // and its first W fragment (g = 0)
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
_R13_KP_W_OLD = """                            const float4 wv = reinterpret_cast<const float4*>(Ws)[
                                ((g * (KP / 8) + (k0 >> 3)) * 2 + gr) * 32 + lane];
"""
_R13_KP_W_NEW = """                            const float4 wv = (g == 0) ? w0 : reinterpret_cast<const float4*>(Ws)[
                                ((g * (KP / 8) + (k0 >> 3)) * 2 + gr) * 32 + lane];
"""


def _r13_kp(s):
    """K_PIPE (fused_forward.apply_k_pipe) on the X_LATE render: both parts' k-loops."""
    return _r13_sub(_r13_sub(s, _R13_KP_A_OLD, _R13_KP_A_NEW, 2), _R13_KP_W_OLD, _R13_KP_W_NEW, 2)


# ------------------------------------------------------------------------------------------ 17(ii): a-ld128
_R13_A_READ = """\
                        #pragma unroll
                        for (int t = 0; t < 2; ++t) {
                            const float* at = a_s + (t - gr) * A_STRIDE;
                            split_tf32(at[gid * KP + (((k0 >> 2) ^ gid) << 2) + tq],
                                       bh[t][0], bl[t][0]);
                            split_tf32(at[gid * KP + ((((k0 >> 2) + 1) ^ gid) << 2) + tq],
                                       bh[t][1], bl[t][1]);
                        }
"""
_R13_A_READ_128 = """\
                        {  // round 13 (a-ld128): the lane's four a values of k-step s in one float4
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
_R13_A_READ_F = """\
                        {  // round 13 (F-a-ld128, falsifier): one LDS.128 from the plan's layout, wrong values
                            const float4 av = reinterpret_cast<const float4*>(a_s)[((k0 >> 3) & 3) * 32 + lane];
                            split_tf32(av.x, bh[0][0], bl[0][0]);
                            split_tf32(av.y, bh[0][1], bl[0][1]);
                            split_tf32(av.z, bh[1][0], bl[1][0]);
                            split_tf32(av.w, bh[1][1], bl[1][1]);
                        }
"""
_R13_A_STAGING_128 = """\
                for (int i = lane; i < B * (KP / 8); i += 32) {  // round 13 (a-ld128): edge b, k-step s
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


def _r13_a_ld128(s):
    """17(ii) exact: both parts' a staging and k-loop reads in the float4-per-lane layout (not with K_PIPE)."""
    if _r12._r12_constant(s, "KP") != 64 or _r12._r12_constant(s, "B") != 8:
        raise ValueError("round 13 (a-ld128) needs KP = 64 and B = 8 (8 k-steps, 4 lanes per edge)")
    return _r13_sub(_r13_sub(s, _r12._R12_STAGING, _R13_A_STAGING_128, 2), _R13_A_READ, _R13_A_READ_128, 2)


def _r13_f_a_ld128(s):
    return _r13_sub(s, _R13_A_READ, _R13_A_READ_F, 2)


# ------------------------------------------------------------------------------------ the base: CG_FOLD or not
_R13_CPF = re.compile(r"^ *const float cpf\[\d+\] = \{", re.M)


def _r13_is_folded(s):
    """True for a CG_FOLD render (idea 24 applied: part 1 declares its c_p per path)."""
    return _R13_CPF.search(s) is not None


def _r13_fold_if_needed(report=None):
    """Round 12's fold, unless the render is already folded (CG_FOLD, the plan's default since FastMACE 1388c3a1)."""
    fold = _r12._r12_fold(report=report)

    def transform(s):
        return s if _r13_is_folded(s) else fold(s)
    return transform


def _r13_unfolded(kernel, rerender=None):
    """The render without CG_FOLD: the source itself if unfolded; else the lab's plan kernel re-rendered with
    ``cg_fold=False`` and its own flags (``rerender``: a stand-in for the CPU check), accepted only if folding it
    again gives the source byte for byte (``cg_fold.apply_cg_fold``)."""
    def transform(s):
        if not _r13_is_folded(s):
            return s
        if rerender is not None:
            unfolded, fold_back = rerender(), _r12._r12_fold()
        else:
            if kernel is None or getattr(kernel, "source", None) != s:
                raise ValueError("round 13: rowfold and rcfold need the unfolded TP; on a CG_FOLD render only the "
                                 "plan's own source can be re-rendered (or build the base with --constant "
                                 "fastmace.kernels.oeq_streaming.fused_forward:CG_FOLD=False)")
            from fastmace.kernels.oeq_streaming import fused_forward as ff
            from fastmace.kernels.oeq_streaming.cg_fold import apply_cg_fold as fold_back
            unfolded = ff.render(kernel.schedule, kernel.k, kernel.rule, kernel.store_w, kernel.used, tc=kernel.tc,
                                 blocks=kernel.blocks, prefetch=kernel.prefetch, vec_sh=kernel.vec_sh,
                                 vec_a=kernel.vec_a, wfrag=kernel.wfrag, tctp=kernel.tctp, wgmma=kernel.wgmma,
                                 groups=kernel.groups, sh_lds4=kernel.sh_lds4, lane_idx=kernel.lane_idx,
                                 w_group=kernel.w_group, wfrag_region=kernel.wfrag_region, a_split=kernel.a_split,
                                 x_late=kernel.x_late, k_pipe=kernel.k_pipe, cg_fold=False)
        if _r13_is_folded(unfolded) or fold_back(unfolded) != s:
            raise ValueError("round 13: the re-render without CG_FOLD does not fold back to the plan's source")
        return unfolded
    return transform


# ---------------------------------------------------------------------------------------------------- variants
ROUND13_REFERENCES = {"fold-tpnf": "fold", "fold-kp": "fold", "fold-a-ld128": "fold", "rcfold-tpnf": "rcfold",
                      "rc-all": "rcfold"}  # variant -> the variant it must match bitwise


def round13_references(kernel=None):
    """ROUND13_REFERENCES for the lab's base: on a CG_FOLD base the fold-* variants match the base itself (round 12's
    fold does not apply there), so they are left to the lab's comparison with the base."""
    refs = dict(ROUND13_REFERENCES)
    if kernel is not None and _r13_is_folded(kernel.source):
        refs = {v: r for v, r in refs.items() if r != "fold"}
    return refs


def round13_variants(kernel=None, report=None, rerender=None):
    """name -> (kind, source transform, extra NVRTC options) for the lab's ``forward_variants``. ``kernel``: the
    lab's plan kernel (rowfold and rcfold re-render it without CG_FOLD when the base is folded; no launch-rule
    change); ``report`` collects the fold variants' FP32 model per part; ``rerender``: the CPU check's stand-in for
    that re-render. On a CG_FOLD base ``a-ld128`` is ``fold-a-ld128`` and is left out."""
    fold = _r13_fold_if_needed(report)
    unfold = _r13_unfolded(kernel, rerender)
    row, rc = _r13_then(unfold, _r13_rcfold(False, report)), _r13_then(unfold, _r13_rcfold(True, report))
    tpnf = _r13_per_part(_r13_tpnf_part)
    variants = {
        "rowfold": ("rounding", row),
        "rcfold": ("rounding", rc),
        "fold-tpnf": ("exact", _r13_then(fold, tpnf)),
        "rcfold-tpnf": ("exact", _r13_then(rc, tpnf)),
        "fold-kp": ("exact", _r13_then(_r13_kp, fold)),
        "a-ld128": ("exact", _r13_a_ld128),
        "fold-a-ld128": ("exact", _r13_then(fold, _r13_a_ld128)),
        "F-a-ld128": ("falsifier", _r13_f_a_ld128),
        "rc-all": ("exact", _r13_then(rc, tpnf, _r13_a_ld128)),
    }
    if kernel is not None and _r13_is_folded(kernel.source):
        del variants["a-ld128"]
    return {name: (kind, _r13_guard(transform), ()) for name, (kind, transform) in variants.items()}


def _r13_main(argv):
    """``python -I -B lab_round13.py RENDER.cu OUTDIR``: every variant that applies to RENDER, written to OUTDIR."""
    import pathlib

    if len(argv) != 2:
        raise SystemExit("usage: python -I -B lab_round13.py RENDER.cu OUTDIR")
    render = pathlib.Path(argv[0]).read_text()
    outdir = pathlib.Path(argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    for name in round13_variants():
        report = []  # the fold variants' FP32 model per part
        kind, transform, extra = round13_variants(report=report)[name]
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
            info = next((r[3] for r in report if isinstance(r[3], dict) and "rows" in r[3]), None)
            if info:
                model += (f"; part 1: {info['units']} of {info['terms']} terms units, {info['products']} products, "
                          f"{len(info['rows'])} row factors, {len(info['cols'])} column factors")
        print(f"{name}: {kind}, {len(source.splitlines())} lines{model}")


if __name__ == "__main__":
    _r13_main(sys.argv[1:])
