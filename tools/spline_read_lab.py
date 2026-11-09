"""Lab only: other table read paths for the SPLINE_W forward, timed by ``tools/spline_table_bound.py`` (9 Nov 2026).

Two ways of fetching fewer table bytes into L1, both on the STAGE_EDGES kernel. The production kernel is unchanged;
these variants exist only here:

- ``bspline`` (Codex 9 Nov v2, proposal 2): a cubic B-spline on the C2 spline's refined grid (851 intervals,
  ``results/bspline-reuse-bound-20261109/c2_certificate.json``). The table holds one row of Q floats per knot,
  ``[pairs * (intervals + 3), Q]``. An edge in interval k reads rows k .. k + 3, four scalar loads per weight, and
  weighs them with the uniform cubic basis of t. Neighbouring intervals share three rows. Non-uniform band edges
  would need per-interval basis weights; those cost a few FMAs per edge, not per weight, and are left out. The table
  is random (only the time counts).
- ``packed8`` (Codex 9 Nov v3, proposal 2): production's grid with 8 bytes per weight instead of 16, in the format
  that passes the value certificate (``tools/spline_packed_certificate.py``, S12): the cubic in s = t - 1/2,
  economized to a quadratic (s^3 -> 3 s / 16), then a 6-bit shared exponent E and signed fixed-point a0 (22 bits),
  a1 (20), a2 (16) in steps 2^(E - 21), 2^(E - 19), 2^(E - 15). Layout per weight: x = a0 | (a2 >> 6) << 22,
  y = a1 | (a2 & 63) << 20 | (E + bias) << 26. The table is encoded from the captured production table, so its
  message can be compared with production's.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

BASIS = ("            const float u1 = 1.f - t, tt = t * t;\n"
         "            const float b0 = u1 * u1 * u1 * (1.f / 6.f);\n"
         "            const float b1 = fmaf(tt, fmaf(t, 0.5f, -1.f), 2.f / 3.f);\n"
         "            const float b2 = fmaf(tt, fmaf(t, -0.5f, 0.5f), fmaf(t, 0.5f, 1.f / 6.f));\n"
         "            const float b3 = tt * t * (1.f / 6.f);\n")
CENTRE = "            const float s = t - 0.5f, s64 = 64.f * s;\n"
CENTRE2 = "            const float s = t - 0.5f, s2 = s * s;\n"  # packed8v2
BITS = (22, 20, 16)  # a0, a1, a2 (packed8)


def c2_grid(path=Path("results/bspline-reuse-bound-20261109/c2_certificate.json")):
    """The C2 spline's refined grid (the last ``c2_refined`` round)."""
    return [tuple(g) for g in json.loads(Path(path).read_text())["c2_refined"][-1]["grid"]]


def _replace(source, old, new, count=1):
    if source.count(old) != count:
        raise ValueError(f"spline_read_lab: expected {count} of {old!r}, found {source.count(old)}")
    return source.replace(old, new)


def transform(source, read, bias=0):
    """The STAGE_EDGES forward source with the table read of ``read`` ("bspline" or "packed8"; ``bias``: packed8's
    exponent bias)."""
    if "s_tq[TILE]" not in source:
        raise ValueError("spline_read_lab: the variants need the STAGE_EDGES kernel")
    kind = {"bspline": "float", "packed8": "uint2", "packed8v2": "uint2"}[read]
    source = _replace(source, "const float4* __restrict__ table", f"const {kind}* __restrict__ table")
    source = _replace(source, "const float4* tq = table + s_tq[k] + ch;",
                      f"const {kind}* tq = table + s_tq[k] + ch;\n" + (BASIS if read == "bspline" else CENTRE2 if read == "packed8v2" else CENTRE))
    source = source.replace("const float4* ahead = table", f"const {kind}* ahead = table")  # PREFETCH_AHEAD
    if read == "bspline":  # (intervals + 3) knot rows per pair
        source = _replace(source, "tq_k = ((int64_t)pairs[e] * intervals + iv) * Q;",
                          "tq_k = ((int64_t)pairs[e] * (intervals + 3) + iv) * Q;")
    pattern = (r"const float4 c = __ldg\(tq \+ (\d+)\);\n\s*const float w = fmaf\(t, fmaf\(t, fmaf\(t, c\.w, c\.z\), "
               r"c\.y\), c\.x\);")
    loads = re.findall(pattern, source)
    if not loads:
        raise ValueError("spline_read_lab: no table reads found")

    def bspline(m):
        o = m.group(1)
        return (f"const float w = fmaf(b3, __ldg(tq + 3 * Q + {o}), fmaf(b2, __ldg(tq + 2 * Q + {o}), "
                f"fmaf(b1, __ldg(tq + Q + {o}), b0 * __ldg(tq + {o}))));")

    def packed8(m):  # w = 2^(E - 21) (q0 + s (4 q1 + s 64 q2))
        o = m.group(1)
        return (f"const uint2 c = __ldg(tq + {o});\n"
                f"                const int q0 = (int)(c.x << 10) >> 10, q1 = (int)(c.y << 12) >> 12;\n"
                f"                const int q2 = ((int)c.x >> 22) * 64 + (int)((c.y >> 20) & 63u);\n"
                f"                const float sc = __uint_as_float(((c.y >> 26) + {106 - bias}u) << 23);\n"
                f"                const float w = sc * fmaf(s, fmaf(s64, (float)q2, 4.f * (float)q1), (float)q0);")

    def packed8v2(m):  # the same values; each field decoded as (exponent | mantissa field) - 1.5 * 2^e, exact
        o = m.group(1)
        return (f"const uint2 c = __ldg(tq + {o});\n"
                f"                const unsigned eb = ((c.y & 63u) << 23) + {(128 - bias) << 23}u;\n"
                f"                const float hb = __uint_as_float(eb | 0x400000u);\n"
                f"                const float f0 = __uint_as_float(((c.x >> 9) & 0x7FFFFEu) | eb) - hb;\n"
                f"                const float f1 = __uint_as_float(((c.y >> 3) & 0x7FFFF8u) | eb) - hb;\n"
                f"                const float f2 = __uint_as_float((__funnelshift_l(c.y, c.x, 13) & 0x7FFF80u) | eb) - hb;\n"
                f"                const float w = fmaf(s2, f2, fmaf(s, f1, f0));")

    reads = {"bspline": bspline, "packed8": packed8, "packed8v2": packed8v2}
    return re.sub(pattern, reads[read], source), len(loads)


def ghoist(source):
    """The ``ghoist`` falsifier (third 9 Nov report, "G-hoisting"): every output's CG sum regrouped by sender feature,
    acc_o += (w x_i) g_oi, with the channel-independent g_oi = sum_j c_oij e_j read from a per-edge shared array. The
    instruction shape of the real rewrite (which would compute g at staging from the tile's harmonics); here g is
    filled with harmonics, so the output is wrong. Returns (source, number of g values per edge)."""
    if "s_tq[TILE]" not in source:
        raise ValueError("spline_read_lab: the variants need the STAGE_EDGES kernel")
    count = [0]

    def output(m):
        o, expr = m.group(1), m.group(2)
        xs = sorted(set(re.findall(r"\bx(\d)\b", expr)))
        lines = []
        for i in xs:
            lines.append(f"acc[{o}] = fmaf(w * x{i}, s_g[gk + {count[0]}], acc[{o}]);")
            count[0] += 1
        return "\n                ".join(lines)

    source = re.sub(r"acc\[(\d+)\] = fmaf\(w, (.*), acc\[\1\]\);", output, source)
    gd = (count[0] + 3) // 4 * 4
    source = _replace(source, "__shared__ __align__(16) float s_sh[TILE * ED];",
                      f"__shared__ __align__(16) float s_sh[TILE * ED];\n    constexpr int GD = {gd};\n"
                      "    __shared__ __align__(16) float s_g[TILE * GD];")
    fill = "for (int i = ch; i < count_in * ED; i += C) s_sh[i] = __ldg(sh + base * ED + i);"
    source = _replace(source, fill, fill + "\n        for (int i = ch; i < count_in * GD; i += C) "
                      "s_g[i] = __ldg(sh + base * ED + (i & 15));")
    she = "const float* she = s_sh + s_k[k] * ED;"
    source = _replace(source, she, she + "\n            const int gk = s_k[k] * GD;")
    return source, count[0]


def _f32_unit(literal):
    """+1, -1 or None: the coefficient literal ("0x1.fff...p-1f") as float32 is exactly +-1."""
    import struct

    v = struct.unpack("f", struct.pack("f", float.fromhex(literal.rstrip("f"))))[0]
    return 1 if v == 1.0 else -1 if v == -1.0 else None


def geom(source):
    """The shared-geometry forward (9 Nov external proposal 1, "shared geometric expressions across the ten paths"),
    correct and selective, unlike ``ghoist``: per path the cheapest of three forms, by FP32 operations per (edge,
    channel), unit coefficients (float32 +-1) free:

    - as today: acc_o = fmaf(w, sum_t x_i e_j c_t, acc_o);
    - one sender component (the scalar-input paths): wx = w x_i once, acc_o = fmaf(wx, e_j c, acc_o);
    - G-hoisted: g_oi = sum_j c_oij e_j per edge, computed once per tile edge at staging into shared memory (it does
      not depend on the channel), and acc_o += (w x_i) g_oi.

    Rounding only (the CG sum is taken before the multiply by w x_i). Returns (source, number of g values per edge)."""
    if "s_tq[TILE]" not in source:
        raise ValueError("spline_read_lab: the variants need the STAGE_EDGES kernel")
    head, sep, loop = source.partition("        for (int k = 0; k < count; ++k) {")
    if not sep:
        raise ValueError("spline_read_lab: no edge loop")
    parts = re.split(r"(\n            \{\n)", loop)
    fills, ng, chosen = [], 0, []
    out = [parts[0]]
    for marker, block in zip(parts[1::2], parts[2::2]):
        she = dict(re.findall(r"const float (e\d+) = she\[(\d+)\];", block))
        rows = re.findall(r"acc\[(\d+)\] = fmaf\(w, (.*), acc\[\1\]\);", block)
        if not rows or not she:
            out += [marker, block]
            continue
        terms = {o: [(int(xi), ej, c) for xi, ej, c in TERM.findall(expr)] for o, expr in rows}
        xs = sorted({t[0] for ts in terms.values() for t in ts})
        nonunit = sum(1 for ts in terms.values() for t in ts if _f32_unit(t[2]) is None)
        cur = sum(len(ts) + 1 for ts in terms.values()) + nonunit
        factor = 1 + sum(len(ts) for ts in terms.values()) + nonunit if len(xs) == 1 else cur + 1
        pairs = sum(len({t[0] for t in ts}) for ts in terms.values())
        hoist = len(xs) + pairs
        best = min(cur, factor, hoist)
        new_rows = {}
        if best == cur:
            chosen.append("cur")
            out += [marker, block]
            continue
        if best == factor:
            chosen.append("factor")
            i = xs[0]
            for o, ts in terms.items():
                expr = f"acc[{o}]"
                for _, ej, c in ts:
                    unit = _f32_unit(c)
                    e = f"-e{ej}" if unit == -1 else f"e{ej}" if unit == 1 else f"e{ej} * {c}"
                    expr = f"fmaf(wx{i}, {e}, {expr})"
                new_rows[o] = f"acc[{o}] = {expr};"
        else:
            chosen.append("hoist")
            for o, ts in terms.items():
                expr = f"acc[{o}]"
                for i in sorted({t[0] for t in ts}):
                    g = " + ".join(f"q[{she['e' + ej]}] * {c}" for xi, ej, c in ts if xi == i)
                    fills.append(f"g[{ng}] = {g};")
                    expr = f"fmaf(wx{i}, s_g[gk + {ng}], {expr})"
                    ng += 1
                new_rows[o] = f"acc[{o}] = {expr};"
        wx = ", ".join(f"wx{i} = w * x{i}" for i in xs)
        block = re.sub(r"(const float w = fmaf\(t, .*;\n)",
                       lambda m: m.group(1) + f"                const float {wx};\n", block, count=1)
        block = re.sub(r"acc\[(\d+)\] = fmaf\(w, (.*), acc\[\1\]\);", lambda m: new_rows[m.group(1)], block)
        block = re.sub(r"\s*const float e\d+ = she\[\d+\];", "", block) if best != factor else block
        out += [marker, block]
    source = head + sep + "".join(out)
    if ng:
        gd = (ng + 3) // 4 * 4
        source = _replace(source, "__shared__ __align__(16) float s_sh[TILE * ED];",
                          f"__shared__ __align__(16) float s_sh[TILE * ED];\n    constexpr int GD = {gd};\n"
                          "    __shared__ __align__(16) float s_g[TILE * GD];")
        body = "\n                ".join(fills)
        source = _replace(source, "        const int count = s_count;\n",
                          "        for (int kk = ch; kk < count_in; kk += C) {  // geom: the tile's g, once per edge\n"
                          "            const float* q = s_sh + kk * ED;\n"
                          "            float* g = s_g + kk * GD;\n"
                          f"                {body}\n"
                          "        }\n"
                          "        __syncthreads();\n"
                          "        const int count = s_count;\n")
        she = "const float* she = s_sh + s_k[k] * ED;"
        source = _replace(source, she, she + "\n            const int gk = s_k[k] * GD;")
    source = source.replace("// spline_forward", f"// geom lab variant (paths: {' '.join(chosen)}); spline_forward", 1)
    return source, ng


TERM = re.compile(r"x(\d+) \* e(\d+) \* (-?0x[0-9a-fA-F.]+p[+-]?\d+f)")


def deferred(source, keep=None):
    """The deferred-CG forward (round 1 proposal 3, Symmetrix compute_Phi1): per path p and (i, j) pair,
    phi_p[i, j] += (w_p x_i) e_j over the receiver's edges, and the CG coefficients applied once per receiver after the
    edge loop, acc_o = sum c_oij phi_p[i, j]. Rounding only (the coefficient is applied after the edge sum).
    ``keep``: the path indices to compute (round 1's two-thread split, timed as two kernels: the paths left out are
    dropped and their outputs written as zero). Returns (source, number of phi accumulators)."""
    if "s_tq[TILE]" not in source:
        raise ValueError("spline_read_lab: the variants need the STAGE_EDGES kernel")
    lines = source.split("\n")
    path, phi, finals, out = -1, {}, {}, []
    pending = []  # the current path's acc lines: (indent, o, terms)

    def flush():
        if not pending:
            return
        indent = pending[0][0]
        if keep is None or path in keep:
            used = []
            for _, o, terms in pending:
                for i, j, c in terms:
                    key = (path, i, j)
                    if key not in phi:
                        phi[key] = len(phi)
                        used.append(key)
                    finals.setdefault(o, []).append((c, phi[key]))
            for i in sorted({i for _, i, _ in used}):
                out.append(f"{indent}const float u{i} = w * x{i};")
            for key in used:
                out.append(f"{indent}phi[{phi[key]}] = fmaf(u{key[1]}, e{key[2]}, phi[{phi[key]}]);")
        pending.clear()

    acc_line = re.compile(r"^(\s*)acc\[(\d+)\] = fmaf\(w, (.*), acc\[\2\]\);$")
    stored = False
    for line in lines:
        if "const float w = " in line:
            flush()
            path += 1
        m = acc_line.match(line)
        if m:
            terms = [(int(a), int(b), c) for a, b, c in TERM.findall(m.group(3))]
            if not terms or len(terms) != m.group(3).count("*") // 2:
                raise ValueError(f"spline_read_lab: cannot parse {line.strip()!r}")
            pending.append((m.group(1), int(m.group(2)), terms))
            continue
        if line.strip() == "}" and pending:
            flush()
        if not stored and re.match(r"\s*out0\[", line):
            stored = True
            for o in sorted(finals):
                expr = " + ".join(f"{c} * phi[{n}]" for c, n in finals[o])
                out.append(f"    acc[{o}] = {expr};")
        out.append(line)
    if not stored:
        raise ValueError("spline_read_lab: no output stores found")
    source = "\n".join(out)
    source = _replace(source, "    float acc[40];",
                      f"    float acc[40];\n    float phi[{max(1, len(phi))}];\n    #pragma unroll\n"
                      f"    for (int i = 0; i < {max(1, len(phi))}; ++i) phi[i] = 0.f;")
    return source, len(phi)


# round 1's two-thread split of each channel's paths (41 and 35 phi accumulators), timed as two kernels
DEFERRED_KEEP = {"deferred": None, "deferred-a": {7, 8, 9}, "deferred-b": {0, 1, 2, 3, 4, 5, 6}}


def encode_packed8(table, layout="v1"):
    """The packed8 table ``int32 [rows, Q, 2]`` of a production table ``float32 [rows, Q, 4]`` (t basis) and its
    exponent bias."""
    import torch
    c = table.double()
    c0, c1, c2, c3 = c.unbind(-1)
    a0, a1, a2 = c0 + c1 / 2 + c2 / 4 + c3 / 8, c1 + c2 + 0.75 * c3 + 3.0 / 16.0 * c3, c2 + 1.5 * c3
    top = torch.stack([a0, a1, a2]).abs().amax(dim=0)
    e = torch.floor(torch.log2(top.clamp_min(1e-300))) + 1
    live = top > 0
    emin, emax = int(e[live].min()), int(e[live].max())
    e = torch.where(live, e, torch.full_like(e, emin))
    bias = -emin
    if emax + bias > 63 or 106 - bias < 1:
        raise ValueError(f"packed8: exponents {emin} .. {emax} do not fit 6 bits")
    q = []
    for a, n in zip((a0, a1, a2), BITS):
        lim = 2 ** (n - 1) - 1
        q.append(torch.round(a / torch.exp2(e - n + 1)).clamp(-lim, lim).to(torch.int64))
    q0, q1, q2 = q
    if layout == "v2":  # biased fields: x = Q0 << 10 | Q2 >> 6, y = (Q2 & 63) << 26 | Q1 << 6 | E6
        u0, u1, u2 = q0 + 2 ** 21, q1 + 2 ** 19, q2 + 2 ** 15
        x = u0 << 10 | u2 >> 6
        y = (u2 & 63) << 26 | u1 << 6 | (e.to(torch.int64) + bias)
    else:
        x = (q0 & 0x3FFFFF) | ((q2 >> 6) & 0x3FF) << 22
        y = (q1 & 0xFFFFF) | (q2 & 63) << 20 | (e.to(torch.int64) + bias) << 26
    packed = torch.stack([x, y], dim=-1)
    packed = torch.where(packed >= 2 ** 31, packed - 2 ** 32, packed).to(torch.int32)
    return packed.contiguous(), bias


def variant(kernel, read, table=None, grid=None):
    """A copy of the captured ``SplineForward`` with the table read of ``read``, compiled, and its table: for
    "bspline" a random one on ``grid`` (default: the C2 grid); for "packed8" ``table`` (production's) encoded.
    Returns (kernel, table, sources)."""
    import torch
    from cuda.bindings import driver

    from fastmace.kernels.oeq_streaming.runtime import _check, compile_cubin
    from fastmace.kernels.oeq_streaming.spline_forward import SplineForward

    schedule, blocks, old_grid, r_max, device, path_scale, shared_locator = kernel.init_args
    grid = (c2_grid() if grid is None else grid) if read == "bspline" else old_grid
    k = SplineForward(schedule, blocks, grid, r_max, device, path_scale, shared_locator,
                      block_order=kernel.block_order)
    k.order = kernel.order
    if not k.stage_edges:
        raise SystemExit("spline_read_lab: STAGE_EDGES is off (set spline_forward:STAGE_EDGES=True)")
    q = len(schedule.paths) * schedule.channels
    bias = 0
    if read in ("ghoist", "geom") or read in DEFERRED_KEEP:
        new_table = table
    elif read == "bspline":
        pairs = table.shape[0] // kernel.intervals
        g = torch.Generator(device=device).manual_seed(0)
        new_table = torch.randn(pairs * (k.intervals + 3), q, device=device, generator=g) * 0.05
    else:
        new_table, bias = encode_packed8(table, "v2" if read == "packed8v2" else "v1")
    capability = torch.cuda.get_device_capability(device)
    functions, loads, sources = [], 0, []
    k.lab_registers = []
    for i, source in enumerate(k.sources):
        if read == "ghoist":
            new, n = ghoist(source)
        elif read == "geom":
            new, n = geom(source)
        elif read in DEFERRED_KEEP:
            new, n = deferred(source, DEFERRED_KEEP[read])
        else:
            new, n = transform(source, read, bias)
        sources.append(new)
        loads += n
        module = _check(driver.cuModuleLoadData(compile_cubin(new, capability, ())))
        function = _check(driver.cuModuleGetFunction(module, f"spline_forward{i}".encode()))
        k._modules.append(module)
        functions.append(function)
        k.lab_registers.append(int(_check(driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, function))))
    k._functions = functions
    print(f"{read}: {k.intervals} intervals, {loads} table reads (ghoist: g values; deferred: phi accumulators) per edge, "
          f"registers {k.lab_registers} "
          f"(production {kernel.registers}), table {new_table.numel() * 4 / 2 ** 20:.1f} MiB, bias {bias}",
          flush=True)
    return k, new_table, sources


def _fake(n, var="channel"):
    """A per-thread float that costs one integer op and differs per n (so nothing is folded or merged)."""
    return f"__int_as_float(0x3c000000 | (({var} + {n}) & 0xfff))"


def wprime_zero(source, what):
    """w′'s source with the loads of ``what`` ("table", "dout", "x" or "all") replaced by register values of one
    integer op each, the arithmetic kept (report 5's zero-byte test: what the operand's L1/shared wavefronts cost,
    not only its misses). The output is wrong."""
    parts = ("table", "dout", "x") if what == "all" else (what,)
    count = 0
    if "table" in parts:
        old = "const float4 v = __ldg(gq + k);"
        if old not in source:
            raise ValueError("spline_read_lab: no float4 table loads in w′")
        z = _fake(0, "weight_index")
        source = source.replace(old, f"const float zq = {z}; const float4 v = make_float4(zq, zq + 1.f, zq + 2.f, "
                                     "zq + 3.f);")
        count += 1
    if "dout" in parts:
        n = [0]

        def dout(m):
            n[0] += 1
            return f"const scalar_t output_gradient = {_fake(100 + n[0])};"
        source, k = re.subn(r"const scalar_t output_gradient =\s*stage_row\[[^\]]*\];", dout, source)
        if not k:
            raise ValueError("spline_read_lab: no staged dout reads in w′")
        count += k
    if "x" in parts:
        names = {}

        def xread(m):
            key = re.sub(r"\s+", "", m.group(1))
            names.setdefault(key, len(names))
            return _fake(300 + names[key])
        source, k = re.subn(r"\bx\[input_base([^\]]*)\]", xread, source)
        if not k:
            raise ValueError("spline_read_lab: no sender-row reads in w′")
        count += k
    return source, count


def wprime_split(source, min_terms=4):
    """w′'s source with each FACTORED u_i / v_j sum of at least ``min_terms`` terms split into two partial sums
    (alternate terms), added where the sum is read (round 2 follow-up: are the serial multiply-add chains what w′
    waits on?). Rounding only. Returns (source, number of split sums)."""
    lines = source.split("\n")
    declare = re.compile(r"acc_t ([uv]_\d+) = acc_t\(0\);")
    update = re.compile(r"\b([uv]_\d+) ([+-]=) scaled")
    starts = [(k, m.group(1)) for k, line in enumerate(lines) for m in [declare.search(line)] if m]
    split = set()  # declaration lines of the sums with enough terms
    for n, (k, name) in enumerate(starts):
        end = next((k2 for k2, name2 in starts[n + 1:] if name2 == name), len(lines))
        if sum(1 for line in lines[k:end] if (m := update.search(line)) and m.group(1) == name) >= min_terms:
            split.add(k)
    live, count = set(), {}
    for k, line in enumerate(lines):
        m = declare.search(line)
        if m:
            name = m.group(1)
            live.discard(name)
            if k in split:
                live.add(name)
                count[name] = 0
                lines[k] = line.replace(m.group(0), f"acc_t {name} = acc_t(0), {name}_b = acc_t(0);")
            continue
        m = update.search(line)
        if m and m.group(1) in live:
            name = m.group(1)
            if count[name] % 2:
                lines[k] = line.replace(m.group(0), f"{name}_b {m.group(2)} scaled", 1)
            count[name] += 1
            continue
        for name in [n for n in live if f"* {n};" in line]:
            lines[k] = lines[k].replace(f"* {name};", f"* ({name} + {name}_b);")
    return "\n".join(lines), len(split)


def wprime_variant(kernel, what):
    """A copy of the captured ``WPrimeBackward`` compiled from ``wprime_zero``'s source."""
    import copy

    import torch
    from cuda.bindings import driver

    from fastmace.kernels.oeq_streaming.runtime import _check, compile_cubin

    source, n = wprime_split(kernel.source) if what == "split" else wprime_zero(kernel.source, what)
    options = (f"--maxrregcount={kernel.max_registers}",) if kernel.max_registers else ()
    k = copy.copy(kernel)
    k.source = source
    k._module = _check(driver.cuModuleLoadData(compile_cubin(source, torch.cuda.get_device_capability(), options)))
    k._function = _check(driver.cuModuleGetFunction(k._module, b"oeq_projected_backward_wprime"))
    registers = int(_check(driver.cuFuncGetAttribute(
        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, k._function)))
    print(f"wprime {what}: {n} loads replaced or sums split, registers {registers}", flush=True)
    return k, source
