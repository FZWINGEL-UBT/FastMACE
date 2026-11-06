"""Idea 24 of the H200 ideas plan (revision 12, 6 November 2026): the fused forward's TP with its CG factors folded
into W (``fused_forward.CG_FOLD``). Moved here from ``tools/lab_round12.py`` (the ideas session's lab transform).

- Part 1 (the paths 1 (x) l -> l'): one factor c_p per path moves into the path's W rows as the block fills its W
  slice (``cpf[p] * W``); the terms become (c / c_p) x e, products are shared across paths, and a unit term costs an
  FADD or an FFMA at most. Its 1 (x) 0 -> 1 path multiplies w by e0 first.
- Part 0 (0 (x) l -> l, unit coefficients): w x0 first (s = w x0, acc += s e).

Rounding class: (c_p w)(sum (c / c_p) x e) is the same sum in exact arithmetic, rounded differently. The transform
parses each part's TP blocks and refuses (ValueError) any render it was not written for: STORE_W must be false (w
would be stored scaled), and the render must hold both DEGREE_PARTS parts with B = 8, ED = 16 and KP = 64.
"""
import itertools
import re
import struct


def _sub(source, old, new, count=None):
    """``source`` with ``old`` replaced by ``new``; raises unless it occurs (``count`` times if given)."""
    found = source.count(old)
    if not found or (count is not None and found != count):
        raise ValueError(f"cg_fold pattern found {found} times: {old[:80]!r}")
    return source.replace(old, new)


_CASE = re.compile(r"^    case (\d+): \{\n", re.M)
_BREAK = "    } break;\n"
_TAIL = _BREAK + "    }\n}"


def _constant(source, name):
    match = re.search(rf"constexpr int {name} = (\d+);", source)
    if match is None:
        raise ValueError(f"cg_fold: constant {name} not found in the render")
    return int(match.group(1))


def _warps(source):
    match = re.search(r"__launch_bounds__\((\d+), 1\)", source)
    if match is None:
        raise ValueError("cg_fold: no __launch_bounds__(threads, 1) in the render")
    return int(match.group(1)) // 32


def _parts(source):
    """(text before the first part, [each part from its ``case`` line through its ``} break;``], the rest)."""
    if "constexpr bool STORE_W = false;" not in source:
        raise ValueError("cg_fold needs the render with STORE_W false: w would be stored scaled")
    if _constant(source, "B") * _constant(source, "ED") != 128 or _constant(source, "KP") != 64:
        raise ValueError("cg_fold needs B = 8, ED = 16 and KP = 64 (one sh float4 per lane, one edge per k-step)")
    starts = [m.start() for m in _CASE.finditer(source)]
    if len(starts) != 2:
        raise ValueError(f"cg_fold needs the render's two parts (DEGREE_PARTS); found {len(starts)}")
    end = source.rindex(_TAIL) + len(_BREAK)
    parts = [source[starts[0]:starts[1]], source[starts[1]:end]]
    for part in parts:
        if not part.endswith(_BREAK):
            raise ValueError("cg_fold: a part does not end with its break")
    return source[:starts[0]], parts, source[end:]


def _f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


_BLOCK = re.compile(r"^ *\{\n((?: *const float e\d+ = sh_s\[b \* ED \+ \d+\];\n)+)((?: *acc\[\d+\] = fmaf\(.*\);\n)+)"
                        r" *\}\n", re.M)
_ACC = re.compile(r"^ *acc\[(\d+)\] = fmaf\((w\[(\d)\]\[b\]|wb\[(\d)\]), (.*), acc\[(\d+)\]\);$")
_TERM = re.compile(r"^(xv\[b\]\[(\d)\]|xc\[(\d)\]) \* e(\d+) \* (-?0x[0-9a-f]+(?:\.[0-9a-f]*)?p[-+]\d+)f$")


def _parse_runs(part):
    """The part's TP: every run of contiguous TP blocks (22(iii)'s no-flush path has two copies) as (start, end); and
    the TP they all hold, [(accumulator, path, [(x index, sh index, float32 c)])] and its (w, x) style."""
    blocks = list(_BLOCK.finditer(part))
    if not blocks:
        raise ValueError("cg_fold: no TP blocks in the part")
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
                am = _ACC.match(line)
                if not am or am.group(1) != am.group(6):
                    raise ValueError(f"cg_fold: unexpected TP line {line!r}")
                g = int(am.group(3) or am.group(4))
                first = _TERM.match(am.group(5).split(" + ")[0])
                style = ("w" if am.group(3) else "wb", "xv" if first and first.group(2) else "xc")
                terms = []
                for t in am.group(5).split(" + "):
                    tm = _TERM.match(t)
                    if not tm:
                        raise ValueError(f"cg_fold: unexpected TP term {t!r}")
                    terms.append((int(tm.group(2) or tm.group(3)), emap[int(tm.group(4))],
                                  _f32(float.fromhex(tm.group(5)))))
                outs.append((int(am.group(1)), g, terms))
        parsed.append((outs, style))
    if any(p != parsed[0] for p in parsed):
        raise ValueError("cg_fold: the part's copies of the TP differ")
    return [(run[0].start(), run[-1].end()) for run in runs], parsed[0][0], parsed[0][1]


def _lit(v):
    return float(v).hex() + "f"


def _plan_part1(outs):
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


def _today_ops(outs):
    """Today's FP32 count: products (shared), per output a first coefficient unless 1.0f, the other terms, w."""
    prods = {(i, j) for _, _, ts in outs for i, j, _ in ts}
    return len(prods) + sum((0 if abs(ts[0][2]) == 1.0 else 1) + len(ts) - 1 + 1 for _, _, ts in outs)


def _fold_part(part, report):
    number = int(_CASE.match(part).group(1))
    runs, outs, (ws, xs) = _parse_runs(part)
    W = (lambda g: f"w[{g}][b]") if ws == "w" else (lambda g: f"wb[{g}]")
    X = (lambda i: f"xv[b][{i}]") if xs == "xv" else (lambda i: f"xc[{i}]")
    ind = "@IND@"  # each copy's own indentation, below
    sh_used = sorted({j for _, _, ts in outs for _, j, _ in ts})
    lines = [ind + "{"] + [f"{ind}const float sh{j} = sh_s[b * ED + {j}];" for j in sh_used]
    if number == 0:
        if not all(len(ts) == 1 and abs(ts[0][2]) == 1.0 and ts[0][0] == 0 for _, _, ts in outs):
            raise ValueError("cg_fold: part 0 is not four 0 (x) l -> l paths with unit coefficients")
        for g in sorted({g for _, g, _ in outs}):
            lines.append(f"{ind}const float s{g} = {W(g)} * {X(0)};  // 24 (b): w x0 first")
        for k, g, ts in outs:
            i, j, c = ts[0]
            lines.append(f"{ind}acc[{k}] = fmaf({'-' if c < 0 else ''}s{g}, sh{j}, acc[{k}]);")
        ops, cp = 4 + len(outs), None
    else:
        ops, cp, prods, c_paths, by_g = _plan_part1(outs)
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
                            expr = f"p{i}_{j} * {_lit(_f32(c / cp[g]))}"
                        lines.append(f"{ind}float t{k} = {expr};")
                    elif u and (i, j) in prods:
                        lines.append(f"{ind}t{k} = t{k} {'-' if c < 0 else '+'} p{i}_{j};")
                    elif u:
                        lines.append(f"{ind}t{k} = fmaf({neg}{X(i)}, sh{j}, t{k});")
                    else:
                        lines.append(f"{ind}t{k} = fmaf(p{i}_{j}, {_lit(_f32(c / cp[g]))}, t{k});")
                lines.append(f"{ind}acc[{k}] = fmaf({W(g)}, t{k}, acc[{k}]);")
    lines.append(ind + "}")
    report.append((number, _today_ops(outs), ops, cp))
    folded = "\n".join(lines) + "\n"
    new = part
    for s, e in reversed(runs):
        new = new[:s] + folded.replace("@IND@", " " * (len(part[s:].split("\n")[0]) - 1)) + new[e:]
    if number == 1:
        paths = int(re.search(r"const int wst\[(\d+)\] = \{", part).group(1))
        cps = [cp.get(g, 1.0) for g in range(paths)]
        decl = re.search(r"^            const int wst\[\d+\] = \{[^}]*\};\n", new, re.M)
        new = (new[:decl.end()] + "            const float cpf[" + str(paths) + "] = {"
               + ", ".join(_lit(c) for c in cps) + "};  // idea 24: c_p per path, into W\n" + new[decl.end():])
        new = _sub(new, "= k < K ? W[(int64_t)(wst[p] + cb * CB + l) * KA + k] : 0.f;",
                       "= k < K ? cpf[p] * W[(int64_t)(wst[p] + cb * CB + l) * KA + k] : 0.f;", 1)
    return new


def apply_cg_fold(source, report=None):
    """``CG_FOLD`` on a fused-forward render (STORE_W false, both DEGREE_PARTS parts): the TP's CG factors folded
    into W (part 1: one factor per path into its W rows as the block fills its slice) and part 0's w x0 first. Raises
    ValueError where the render is not of that form. ``report`` collects (part, FP32 ops today, after, factors)."""
    pre, parts, post = _parts(source)
    rows = [] if report is None else report
    return pre + "".join(_fold_part(p, rows) for p in parts) + post
