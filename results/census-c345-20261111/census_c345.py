"""Offline NVRTC census (no GPU) for next-steps C3, C4 and C5 (plans/next-steps-20261111.md, section C).

Sources: the production kernels captured on the A100 on 11 November 2026
(results/a100-colab-ideas-20261111/screen1/ghoist/sources/): real-0.cu (spline_forward0: SPLINE_W, R128,
STAGE_EDGES, PACKED8, PREFETCH_AHEAD) and wprime-real.cu (w': STAGE_DOUT, STAGE_EDGES, STAGE_SH, SPLINE_VECTOR 4,
HORNER, FACTORED, SCALAR_PATHS, BUTTERFLY). Each variant is a text transform of the captured source, compiled in the
same NVRTC build for sm_80 and sm_90 with tools/bench/cu_census.py's compiler.

  C3  one PACKED8 decode for two consecutive staged edges on the same table row (same pair and interval).
  C4  joint rescaling, staging-only form: staged dout scaled per output component (s_o), staged harmonics per
      component (b_j), x unscaled; coefficients c * b_j / s_o; dsh restored by 1 / b_j at the edge's write.
  C5  w' accumulators started from their first product (acc = p instead of acc = 0; acc += p).

Per-iteration instruction counts come from the SASS loop ranges (backward branches): the innermost loop that holds
nearly all of the kernel's FFMA.

    python results/census-c345-20261111/census_c345.py
"""
import collections
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "bench"))
from cu_census import compile_cubin, tool  # noqa: E402

SOURCES = ROOT / "results/a100-colab-ideas-20261111/screen1/ghoist/sources"
OUT = HERE / "sources"
FMA_PIPE = ("FFMA", "FMUL", "FADD", "IMAD")  # Ampere's FMA pipe (IMAD and its MOV form issue there too)


# --------------------------------------------------------------------------------------------------------- SASS
def sass_of(source, arch):
    cubin = compile_cubin(source, arch)
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "k.cubin")
        Path(f).write_bytes(cubin)
        sass = subprocess.run([tool("cuobjdump"), "-sass", f], capture_output=True, text=True).stdout
        res = subprocess.run([tool("cuobjdump"), "-res-usage", f], capture_output=True, text=True).stdout
    return sass, res


def analyse(source, arch, loop_share=0.9):
    sass, res = sass_of(source, arch)
    inst = []  # (address, opcode, text)
    for line in sass.splitlines():
        m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)(.*)", line)
        if m:
            inst.append((int(m.group(1), 16), m.group(3), m.group(4)))
    ops = collections.Counter(op.split(".")[0] for _, op, _ in inst)
    addr = [a for a, _, _ in inst]
    loops = []
    for a, op, rest in inst:
        if op.startswith("BRA"):
            t = re.search(r"0x([0-9a-f]+)", rest)
            if t and int(t.group(1), 16) < a:
                lo = int(t.group(1), 16)
                body = [o.split(".")[0] for x, o, _ in inst if lo <= x <= a]
                loops.append((lo, a, collections.Counter(body)))
    total_ffma = ops["FFMA"]
    loop = None
    for lo, hi, c in sorted(loops, key=lambda L: L[1] - L[0]):  # innermost first
        if c["FFMA"] >= loop_share * total_ffma:
            loop = (lo, hi, c)
            break
    regs = re.search(r"REG:(\d+)", res)
    local = re.search(r"LOCAL:(\d+)", res)
    shared = re.search(r"SHARED:(\d+)", res)
    out = dict(arch=arch, registers=int(regs.group(1)), local=int(local.group(1)) if local else 0,
               shared=int(shared.group(1)) if shared else 0, total=len(inst),
               spill_ops=ops["LDL"] + ops["STL"], **{k: ops[k] for k in ("FFMA", "FMUL", "FADD", "IMAD", "LDG", "LDS")})
    if loop:
        lo, hi, c = loop
        out["arms"] = arms(inst, lo, hi)
        out.update(loop_len=sum(c.values()), loop_fma_pipe=sum(c[k] for k in FMA_PIPE),
                   loop_ops={k: c[k] for k in ("FFMA", "FMUL", "FADD", "IMAD", "LOP3", "SHF", "LDG", "LDS", "IADD3")})
    return out


def arms(inst, lo, hi):
    """The loop's two-way branch (if / else covering most of the body): header, arm lengths, tail, spills per arm."""
    at = {a: (op, rest) for a, op, rest in inst}
    size = lambda r: (r[1] - r[0]) // 16 + 1
    best = None
    for a, op, rest in inst:
        if not (lo <= a <= hi and op.startswith("BRA")):
            continue
        t = re.search(r"0x([0-9a-f]+)", rest)
        if not t:
            continue
        t = int(t.group(1), 16)
        before = at.get(t - 16)
        if not (a < t <= hi and before and before[0].startswith("BRA")):
            continue
        u = re.search(r"0x([0-9a-f]+)", before[1])
        if not u or not (t < int(u.group(1), 16) <= hi + 16):
            continue
        u = int(u.group(1), 16)
        first, second = (a + 16, t - 16), (t, u - 16)
        if best is None or min(size(first), size(second)) > best[0]:
            spill = lambda r: sum(1 for x, o, _ in inst if r[0] <= x <= r[1] and o.split(".")[0] in ("LDL", "STL"))
            best = (min(size(first), size(second)), dict(header=(a - lo) // 16 + 1, arm_first=size(first),
                                                         arm_second=size(second), tail=(hi - u) // 16 + 1,
                                                         spill_first=spill(first), spill_second=spill(second)))
    return best[1] if best else None


# ----------------------------------------------------------------------------------------------- C3 (forward)
def block_end(lines, start):
    depth = 0
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth == 0:
            return i
    raise ValueError("unbalanced")


def forward_parts(src, prefetch):
    lines = src.split("\n")
    head = next(i for i, l in enumerate(lines) if l.strip() == "for (int k = 0; k < count; ++k) {")
    end = block_end(lines, head)
    body = lines[head + 1:end]
    if not prefetch:
        p = next((i for i, l in enumerate(body) if "PREFETCH_AHEAD" in l), None)
        if p is not None:
            body = body[:p] + body[block_end(body, p) + 1:]
    first = next(i for i, l in enumerate(body) if l == "            {")
    header, blocks, i = body[:first], [], first
    while i < len(body):
        if body[i] == "            {":
            j = block_end(body, i)
            blocks.append(body[i:j + 1])
            i = j + 1
        else:
            i += 1
    return lines[:head], lines[end + 1:], header, blocks, body


def suffix(expr, s):
    expr = re.sub(r"\bx(\d)\b", rf"x\1{s}", expr)
    expr = re.sub(r"\be(\d)\b", rf"e\1{s}", expr)
    return re.sub(r"\bw\b", f"w_{s}", expr)


def paired_body(header, blocks, ind="            "):
    out = [ind + "const int snd_a = s_snd[k], snd_b = s_snd[k + 1];",
           ind + "const float t_a = s_t[k], t_b = s_t[k + 1];",
           ind + "const float* she_a = s_sh + s_k[k] * ED;",
           ind + "const float* she_b = s_sh + s_k[k + 1] * ED;",
           ind + "const uint2* tq = table + s_tq[k] + ch;  // the same table row for both edges",
           ind + "const float s_a = t_a - 0.5f, s2_a = s_a * s_a;",
           ind + "const float s_b = t_b - 0.5f, s2_b = s_b * s_b;"]
    for l in header:
        m = re.match(r"\s*const float x(\d) = __ldg\(x \+ \(int64_t\)snd \* IN_DIM \+ (\d+) \+ ch\);", l)
        if m:
            for s in "ab":
                out.append(ind + f"const float x{m.group(1)}{s} = __ldg(x + (int64_t)snd_{s} * IN_DIM + {m.group(2)} + ch);")
    for b in blocks:
        out.append(ind + "{")
        accs = []
        for l in b[1:-1]:
            t = l.strip()
            m = re.match(r"const float e(\d) = she\[(\d+)\];", t)
            if m:
                out += [ind + f"    const float e{m.group(1)}a = she_a[{m.group(2)}];",
                        ind + f"    const float e{m.group(1)}b = she_b[{m.group(2)}];"]
            elif t.startswith("const float w = "):
                out += [ind + "    const float w_a = fmaf(s2_a, a2, fmaf(s_a, a1, a0));",
                        ind + "    const float w_b = fmaf(s2_b, a2, fmaf(s_b, a1, a0));"]
            elif t.startswith("acc["):
                accs.append(t)
            else:
                out.append(l)
        # each output's accumulation order kept: edge k, then edge k + 1
        out += [ind + "    " + suffix(t, "a") for t in accs] + [ind + "    " + suffix(t, "b") for t in accs]
        out.append(ind + "}")
    return out


def c3_sources(src):
    pre, post, header, blocks, body = forward_parts(src, prefetch=False)
    base = "\n".join(pre + ["        for (int k = 0; k < count; ++k) {"] + body + ["        }"] + post)
    pair = paired_body(header, blocks)
    variant = "\n".join(pre + ["        for (int k = 0; k < count;) {",
                               "            if (k + 1 < count && s_tq[k] == s_tq[k + 1]) {  // C3: one decode, two edges"]
                        + ["    " + l for l in pair] + ["                k += 2;", "            } else {"]
                        + ["    " + l for l in body] + ["                k += 1;", "            }", "        }"] + post)
    paired_only = "\n".join(pre + ["        for (int k = 0; k + 1 < count; k += 2) {  // census only: always paired"]
                            + pair + ["        }"] + post)
    return {"c3_base": base, "c3_variant": variant, "c3_paired_only": paired_only}


# ---------------------------------------------------------------------------------------------------- C4 (w')
TOKEN = re.compile(r"stage_row\[(\d+\s*\+\s*\d+) \+ channel\]|sh_edge\[(\d+)\]|"
                   r"const scalar_t scaled = output_gradient(?: \* static_cast<scalar_t>\((\S+?)\))?;|"
                   r"input_value =\s*x\[input_base\s*\+\s*(\d+) \+ channel\]")


def c4_terms(src):
    o = j = i = None
    terms = []
    for m in TOKEN.finditer(src):
        if m.group(1) is not None:
            o = sum(int(v) for v in m.group(1).split("+")) // 128  # stage_row[base + offset + channel]
        elif m.group(2) is not None:
            j = int(m.group(2))
        elif m.group(4) is not None:
            i = int(m.group(4)) // 128
        else:
            c = float.fromhex(m.group(3)[:-1]) if m.group(3) else 1.0
            terms.append(dict(o=o, j=j, i=i, c=c, span=m.span()))
    return terms


def products(terms, b, a=None, s=None):
    """Distinct non-unit coefficient products g_o * c (the compiler shares one per output and magnitude). With s
    None each output takes its best scale (one of its own values), which removes one distinct value per row."""
    rows = collections.defaultdict(set)
    for t in terms:
        v = abs(t["c"]) * b.get(t["j"], 1.0) * (a.get(t["i"], 1.0) if a else 1.0) / (s.get(t["o"], 1.0) if s else 1.0)
        rows[t["o"]].add(round(v, 9))
    if s is not None:
        return sum(len(vals - {1.0}) for vals in rows.values()), None
    best_s, n = {}, 0
    for o, vals in rows.items():
        best_s[o] = 1.0 if 1.0 in vals else max(vals)
        n += len(vals) - 1
    return n, best_s


def solve_scales(terms, use_x=False, trials=4000, seed=20261111):
    """Candidate gauges from greedy random maximal consistent subsets of log a_i + log b_j - log s_o = -log c
    (a: input component i, only with use_x; b: harmonic j; s: output o), each scored by the distinct non-unit
    products left after every output takes its best scale. A screen, not a proof of the minimum."""
    rng = random.Random(seed)
    n0, s0 = products(terms, {}, {} if use_x else None)
    best = (n0, -1, {}, {} if use_x else None, s0)  # the identity gauge (today's coefficients) as candidate -1
    for trial in range(trials + 1):
        order = list(range(len(terms))) if trial == 0 else rng.sample(range(len(terms)), len(terms))
        parent, pot = {}, {}

        def potential(n):  # (root, log scale(n) - log scale(root)); no path compression (a few dozen nodes)
            if n not in parent:
                parent[n], pot[n] = n, 0.0
            p, x = 0.0, n
            while parent[x] != x:
                p += pot[x]
                x = parent[x]
            return x, p

        def link(u, v, d):  # require potential(u) - potential(v) = d; False if inconsistent
            ru, pu = potential(u)
            rv, pv = potential(v)
            if ru == rv:
                return abs(pu - pv - d) < 1e-9
            parent[rv] = ru
            pot[rv] = pu - d - pv
            return True

        for t in order:
            T = terms[t]
            d = -math.log(abs(T["c"]))
            if not use_x:
                link(("j", T["j"]), ("o", T["o"]), d)
            else:
                # a_i b_j / s_o: tie i and o through j with the i's own unknown; an edge (i, o) of weight d - log b_j
                rj, pj = potential(("j", T["j"]))
                ri, pi = potential(("i", T["i"]))
                if rj == ri:  # the pair's sum is known relative to one root: one constraint on o
                    ro, po = potential(("o", T["o"]))
                    if ro != rj:
                        parent[ro] = rj
                        pot[ro] = pi + pj - d - po
                else:  # fix j and i together by linking i under j at zero offset first (one gauge choice)
                    parent[ri] = rj
                    pot[ri] = pj - pi
        b = {n[1]: math.exp(potential(n)[1]) for n in list(parent) if n[0] == "j"}
        a = {n[1]: math.exp(potential(n)[1]) for n in list(parent) if n[0] == "i"} if use_x else None
        n, s = products(terms, b, a)
        if best is None or n < best[0]:
            best = (n, trial, b, a, s)
    return best


def f32(v):
    import struct
    return struct.unpack("f", struct.pack("f", v))[0]


def c4_source(src, terms, b, s):
    out, last = [], 0
    for t in terms:
        c = abs(t["c"]) * b.get(t["j"], 1.0) / s.get(t["o"], 1.0)
        line = ("const scalar_t scaled = output_gradient;" if abs(c - 1) < 1e-6 else
                f"const scalar_t scaled = output_gradient * static_cast<scalar_t>({float(f32(c)).hex()}f);")
        out += [src[last:t["span"][0]], line]
        last = t["span"][1]
    text = "".join(out) + src[last:]
    sv = [f32(s.get(o, 1.0)) for o in range(40)]
    bv = [f32(b.get(j, 1.0)) for j in range(16)]
    consts = ("// C4 (staging-only joint rescaling): staged dout x s_o, staged harmonics x b_j, dsh x 1 / b_j.\n"
              f"__constant__ float c4_s[40] = {{{', '.join(v.hex() + 'f' for v in sv)}}};\n"
              f"__constant__ float c4_b[16] = {{{', '.join(v.hex() + 'f' for v in bv)}}};\n"
              f"__constant__ float c4_binv[16] = {{{', '.join(f32(1 / v).hex() + 'f' for v in bv)}}};\n")
    text = text.replace('extern "C" __global__', consts + 'extern "C" __global__', 1)
    # staged dout: four elements of one float4 share an output component (128 channels per component)
    text, n1 = re.subn(r"reinterpret_cast<float4\*>\(stage_row \+ (\d+)\)\[i\] =\s*__ldg\(reinterpret_cast<const float4\*>\(src\) \+ i\);",
                       lambda m: (f"{{ float4 v = __ldg(reinterpret_cast<const float4*>(src) + i); "
                                  f"const float sc = c4_s[({m.group(1)} + 4 * i) >> 7]; v.x *= sc; v.y *= sc; v.z *= sc; "
                                  f"v.w *= sc; reinterpret_cast<float4*>(stage_row + {m.group(1)})[i] = v; }}"), text)
    text, n2 = re.subn(r"stage_row\[(\d+) \+ i\] = __ldg\(src \+ i\);",
                       lambda m: f"stage_row[{m.group(1)} + i] = __ldg(src + i) * c4_s[({m.group(1)} + i) >> 7];", text)
    # staged harmonics
    text, n3 = re.subn(r"reinterpret_cast<float4\*>\(dst\)\[i\] = __ldg\(reinterpret_cast<const float4\*>\(src\) \+ i\);",
                       "{ float4 v = __ldg(reinterpret_cast<const float4*>(src) + i); const int j0 = (4 * i) & 15; "
                       "v.x *= c4_b[j0]; v.y *= c4_b[j0 + 1]; v.z *= c4_b[j0 + 2]; v.w *= c4_b[j0 + 3]; "
                       "reinterpret_cast<float4*>(dst)[i] = v; }", text)
    text, n4 = re.subn(r"for \(int i = threadIdx.x; i < stage_n \* EDGE_DIM; i \+= blockDim.x\) dst\[i\] = __ldg\(src \+ i\);",
                       "for (int i = threadIdx.x; i < stage_n * EDGE_DIM; i += blockDim.x) dst[i] = __ldg(src + i) * c4_b[i & 15];",
                       text)
    # dsh restored at the edge's write
    text, n5 = re.subn(r"dsh\[edge_base \+ \(lane >> SHIFT\)\] = scalar_t\(edge_gradient\[0\]\);",
                       "dsh[edge_base + (lane >> SHIFT)] = scalar_t(edge_gradient[0] * c4_binv[lane >> SHIFT]);", text)
    assert (n1, n2, n3, n4, n5) == (4, 4, 1, 1, 1), (n1, n2, n3, n4, n5)
    return text


# ---------------------------------------------------------------------------------------------------- C5 (w')
def conditional_open(text, brace):
    k = brace - 1
    while k >= 0 and text[k] not in ";{}":
        k -= 1
    lead = text[k + 1:brace]
    return bool(re.search(r"\b(if|for|while|else|do)\b", lead))


def c5_source(src):
    decls = list(re.finditer(r"acc_t (\w+) = acc_t\(0\);", src))
    text, changed = src, []
    for d in reversed(decls):
        name = d.group(1)
        start = d.start()
        m = re.compile(r"acc_t " + name + r" = acc_t\(0\);").match(text, start)
        assert m
        nxt = re.compile(r"\b" + name + r"\s*(\+|-)=\s*([^;]*);").search(text, m.end())
        redecl = re.compile(r"acc_t " + name + r"\b").search(text, m.end())
        other = re.compile(r"\b" + name + r"\b").search(text, m.end())
        if not nxt or (redecl and redecl.start() < nxt.start()) or other.start() != nxt.start():
            continue  # read before its first update, or never updated in this scope
        stack, ok = [], True
        for i in range(m.end(), nxt.start()):
            ch = text[i]
            if ch == "{":
                stack.append(conditional_open(text, i))
            elif ch == "}":
                if stack:
                    stack.pop()
                else:  # left the declaring scope
                    ok = False
                    break
        if not ok or any(stack):
            continue
        sign, expr = nxt.group(1), nxt.group(2)
        new = f"{name} = {'-(' + expr + ')' if sign == '-' else expr};"
        text = text[:nxt.start()] + new + text[nxt.end():]
        text = text[:m.start()] + f"acc_t {name};" + text[m.end():]
        changed.append(name)
    return text, collections.Counter(changed)


# ------------------------------------------------------------------------------------------------------- main
def main():
    OUT.mkdir(exist_ok=True)
    fwd = (SOURCES / "real-0.cu").read_text()
    wp = (SOURCES / "wprime-real.cu").read_text()
    files = c3_sources(fwd)
    terms = c4_terms(wp)
    today, _ = products(terms, {}, None, {})
    staged, trial, b, _, s = solve_scales(terms)
    joint, trial_x, _, _, _ = solve_scales(terms, use_x=True)
    files["w_base"] = wp
    files["c4_staging"] = c4_source(wp, terms, b, s)
    files["c5_first_product"], c5_changed = c5_source(wp)
    for name, text in files.items():
        (OUT / f"{name}.cu").write_text(text, newline="\n")
    rows = {}
    for arch in ("80", "90"):
        for name, text in files.items():
            r = analyse(text, arch)
            rows[f"{name}_sm{arch}"] = r
            print(name, arch, json.dumps(r), flush=True)
    meta = dict(c4=dict(terms=len(terms), unit_terms_today=sum(1 for t in terms if abs(abs(t["c"]) - 1) < 1e-6),
                        products_today=today, products_staging_only=staged, products_joint_with_x=joint,
                        trial=trial, trial_joint=trial_x,
                        b=[b.get(j, 1.0) for j in range(16)], s=[s.get(o, 1.0) for o in range(40)],
                        scale_range=[min(list(b.values()) + list(s.values())), max(list(b.values()) + list(s.values()))]),
                c5=dict(accumulators_changed=dict(c5_changed), count=sum(c5_changed.values())),
                rows=rows)
    print(json.dumps({k: v for k, v in meta.items() if k != "rows"}))
    (HERE / "census_c345.json").write_text(json.dumps(meta, indent=1) + "\n")


if __name__ == "__main__":
    main()
