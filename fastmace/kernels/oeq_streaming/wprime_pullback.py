"""LATE_PULLBACK: the w' backward's "direct" paths with delayed geometry differentiation (outside review, 12 November
2026; results/wprime-late-pullback-20261112). A post-render transform of the float32 FACTORED kernel with
STAGE_DOUT and BLOCK_MESSAGE, like ``wprime_tuning``.

A direct path (input degree a in {0, 1}, edge degree l <= 3, output degree L <= 3) contracts the receiver's adjoint
g with N1's Cartesian harmonic polynomial: h(n) = H_L(q; n) with q = M g (``conversion``, a fixed (2L+1) x (2L+1)
map fitted in float64 against e3nn's harmonics and the schedule's own coupling coefficients). Per lane the kernel
accumulates over the direct paths and its channels only

    G = sum w grad phi (3 components, sh frame)  and  R = sum (dw/dt) phi,

with phi = x0 h (a = 0), x . grad h (a = 1, L = l + 1) or (x.n) h - (x.grad h) / (2L + 1) (a = 1, L = l - 1; the
unit-sphere extension, so no Euler term). The STAGE_DOUT prologue stages each direct path's q in place of g (one
thread per channel). After the channel loop G / sqrt(3) is added to the l = 1 edge-gradient slots and R times the
piece scale to the radial gradient (``sloped`` edges): edge_sh3's backward maps an l = 1 slot's gradient d to
sqrt(3) (I - n n^T) d / r, so the tangential part of G / r comes out exactly, and the radial part goes through dr.
Same signature, same outputs (dsh [E, 16] and dr [E]), same autograd graph: rounding class (the sums regroup).

``direct``: the edge degrees whose paths go direct; the others keep the rendered (spherical) blocks (design 2's
hybrids). A path of a form N1 does not cover (a > 1, or a = 1 with L = l) raises ``ValueError``.
"""
import math
import re

from fastmace.kernels.oeq_streaming.wprime_tuning import LOOP, _block

PROLOGUE_START = "        if (stage_r >= 0 && stage_r < node_count) {\n"
PROLOGUE_END = "    __syncthreads();\n    for (int64_t e = stage_s"
SQRT3_INV = "0.57735026918962576451f"
BLOCK_START = r"\{\s*\n\s*const int weight_index ="
_FITS = {}
APPLIED = 0  # sources transformed in this process (force_stress_gate --production-expect proof counter)


# --------------------------------------------------------------------------------------------- algebra (float64)


def sh_frame():
    """S [3, 3] with sh_1(v) = sqrt(3) v @ S (e3nn, component normalisation): the frame of the l = 1 slots."""
    import torch
    from e3nn import o3
    eye = torch.eye(3, dtype=torch.float64)
    return o3.spherical_harmonics(1, eye, normalize=False, normalization="component") / math.sqrt(3.0)


def _sym2(q):
    import torch
    q0, q1, q2, q3, q4 = q.unbind(-1)
    return torch.stack([torch.stack([q0, q1, q2], -1), torch.stack([q1, q3, q4], -1),
                        torch.stack([q2, q4, -q0 - q3], -1)], -2)


def _sym3n(q, n):
    import torch
    q0, q1, q2, q3, q4, q5, q6 = q.unbind(-1)
    nx, ny, nz = n.unbind(-1)
    qxx = q0 * nx + q1 * ny + q2 * nz
    qxy = q1 * nx + q3 * ny + q4 * nz
    qxz = q2 * nx + q4 * ny + (-q0 - q3) * nz
    qyy = q3 * nx + q5 * ny + q6 * nz
    qyz = q4 * nx + q6 * ny + (-q1 - q5) * nz
    qzz = -qxx - qyy
    return torch.stack([torch.stack([qxx, qxy, qxz], -1), torch.stack([qxy, qyy, qyz], -1),
                        torch.stack([qxz, qyz, qzz], -1)], -2)


def cartesian_h(L, q, n):
    """(H_L(q; n), grad H_L) in the kernel's Cartesian form; q [..., 2L+1], n [..., 3]."""
    import torch
    if L == 0:
        return q[..., 0], torch.zeros_like(n)
    if L == 1:
        return (q * n).sum(-1), q.expand_as(n)
    if L == 2:
        v = torch.einsum("...ij,...j->...i", _sym2(q), n)
        return (n * v).sum(-1), 2.0 * v
    if L == 3:
        v = torch.einsum("...ij,...j->...i", _sym3n(q, n), n)
        return (n * v).sum(-1), 3.0 * v
    raise ValueError(f"LATE_PULLBACK: output degree {L} > 3")


def form(a, l, L):
    """The path's Cartesian form: 'scalar', 'raise' (L = l + 1) or 'lower' (L = l - 1); ValueError otherwise."""
    if l > 3 or L > 3:
        raise ValueError(f"LATE_PULLBACK: degrees ({a}, {l}, {L}) above 3")
    if a == 0 and L == l:
        return "scalar"
    if a == 1 and L == l + 1:
        return "raise"
    if a == 1 and L == l - 1:
        return "lower"
    raise ValueError(f"LATE_PULLBACK: path ({a}, {l}, {L}) not of a form it covers")


def cartesian_f(a, l, L, x, n, q):
    """The path's contraction as a homogeneous polynomial of degree l in n."""
    h, gh = cartesian_h(L, q, n)
    kind = form(a, l, L)
    if kind == "scalar":
        return x[..., 0] * h
    if kind == "raise":
        return (x * gh).sum(-1)
    return (x * n).sum(-1) * h - (n * n).sum(-1) * (x * gh).sum(-1) / (2 * L + 1)


def conversion(a, l, L, coef, samples=96, seed=20261106):
    """(M [2L+1, 2L+1] with q = M g, relative residual): sum_ijo coef[i,j,o] x_i Y_lj(n) g_o == cartesian_f(x, n, q)
    for all x, n (least squares in float64; e3nn's component-normalised harmonics in the sh frame)."""
    import torch
    from e3nn import o3
    frame = sh_frame()
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(samples, 2 * a + 1, generator=gen, dtype=torch.float64)
    n = torch.randn(samples, 3, generator=gen, dtype=torch.float64)
    y = o3.spherical_harmonics(l, n @ frame.T, normalize=False, normalization="component")
    target = torch.einsum("ijo,si,sj->so", torch.as_tensor(coef, dtype=torch.float64), x, y)
    eye = torch.eye(2 * L + 1, dtype=torch.float64)
    phi = torch.stack([cartesian_f(a, l, L, x, n, eye[k].expand(samples, -1)) for k in range(2 * L + 1)], -1)
    m = torch.linalg.lstsq(phi, target).solution
    residual = float((phi @ m - target).abs().max() / target.abs().max().clamp_min(1e-300))
    return m.contiguous(), residual  # q_k = sum_o m[k, o] g_o (phi @ m == target: column o is g = e_o)


# ------------------------------------------------------------------------------------------------ the schedule


def paths(schedule, blocks, fold_alpha):
    """Per schedule path: degrees, form, weight / input starts, each output component's staged-row column and dout
    address, and the conversion (with FOLD_ALPHA the table carries the path's alpha: the kernel's coupling is
    coefficient / alpha, so is the fit's)."""
    import numpy as np

    from fastmace.kernels.oeq_streaming.fused_forward import block_map
    from fastmace.kernels.oeq_streaming.wprime_backward import _stage_layout, factored_paths
    if blocks is None:
        raise ValueError("LATE_PULLBACK needs BLOCK_MESSAGE")
    channels = int(schedule.channels)
    bmap = block_map(blocks, channels)
    _, offsets, _ = _stage_layout(schedule, blocks)
    alphas = [f["alpha"] for f in factored_paths(schedule, True)]
    out = []
    for index, path in enumerate(schedule.paths):
        if path.edge_mul != 1:
            raise ValueError("LATE_PULLBACK: edge_mul != 1")
        outputs = list(path.outputs)
        dim_out = outputs[0].accumulator.irrep_dim
        a, l, L = (path.input_irrep_dim - 1) // 2, (path.edge_irrep_dim - 1) // 2, (dim_out - 1) // 2
        kind = form(a, l, L)
        coef = np.zeros((path.input_irrep_dim, path.edge_irrep_dim, dim_out))
        for output in outputs:
            for term in output.terms:
                coef[term.input_component, term.edge_component, output.output_component] += term.coefficient
        if fold_alpha:
            coef = coef / alphas[index]
        key = (a, l, L, coef.round(15).tobytes())
        if key not in _FITS:
            _FITS[key] = conversion(a, l, L, coef)
        m, residual = _FITS[key]
        if residual > 1e-10:
            raise ValueError(f"LATE_PULLBACK: path {index} ({a}, {l}, {L}) fit residual {residual:.2e}")
        g, width, rank, stride = bmap[int(outputs[0].accumulator.start)]
        dout = [(g, width, rank + k * stride) for k in range(dim_out)]
        out.append(dict(index=index, a=a, l=l, L=L, form=kind, weight_start=int(path.weight_start),
                        input_start=int(path.input_start), dout=dout,
                        stage=[offsets[g] + column for g, _, column in dout], M=m.numpy()))
    return out


def supported(schedule, scalar, blocks):
    """LATE_PULLBACK applies: float32, BLOCK_MESSAGE, edge features l <= 3 (16) and every path of a covered form."""
    if scalar != "float" or blocks is None or int(schedule.edge_dim) != 16:
        return False
    for path in schedule.paths:
        dim_out = list(path.outputs)[0].accumulator.irrep_dim
        try:
            form((path.input_irrep_dim - 1) // 2, (path.edge_irrep_dim - 1) // 2, (dim_out - 1) // 2)
        except ValueError:
            return False
        if path.edge_mul != 1:
            return False
    return True


# --------------------------------------------------------------------------------------------- the transform


def _loop_parts(source):
    """(head, declarations, path blocks, tail, rest) of the channel loop (``wprime_tuning.loop_parts``, but the first
    path block may follow the declarations on the same line)."""
    begin = source.index(LOOP)
    open_, end = _block(source, begin + len(LOOP) - 2)
    body = source[open_ + 1:end - 1]
    found = re.search(BLOCK_START, body)
    if not found:
        raise ValueError("LATE_PULLBACK: no path block in the channel loop")
    decl, rest = body[:found.start()], body[found.start():]
    blocks, pos = [], 0
    while True:
        m = re.search(BLOCK_START, rest[pos:])
        if not m:
            break
        a, b = _block(rest, pos + m.start())
        blocks.append(rest[pos:a] + rest[a:b])  # keep what lies between blocks
        pos = b
    return source[:begin], decl, blocks, rest[pos:], source[end:]


def _f(v):
    text = f"{float(v):.9g}"
    if not any(c in text for c in ".en"):
        text += ".0"
    return text + "f"


def _prologue(path_list, channels):
    lines = [PROLOGUE_START,
             "            // LATE_PULLBACK: the receiver row staged per channel; direct paths as q = M g.",
             "            for (int channel = threadIdx.x; channel < CHANNEL_DIM; channel += blockDim.x) {"]
    for p in path_list:
        lines.append(f"                {{  // path {p['index']} ({p['a']}, {p['l']}, {p['L']})"
                     f"{' direct' if p['direct'] else ''}")
        for o, (g, width, column) in enumerate(p["dout"]):
            lines.append(f"                const float g{o} = __ldg(dout{g} + int64_t(stage_r) * {width} + {column}"
                         " + channel);")
        for k, column in enumerate(p["stage"]):
            if p["direct"]:
                terms = [f"{_f(p['M'][k, o])} * g{o}" for o in range(2 * p["L"] + 1) if abs(float(p["M"][k, o])) > 1e-12]
                value = " + ".join(terms) or "0.f"
            else:
                value = f"g{k}"
            lines.append(f"                stage_row[{column} + channel] = {value};")
        lines.append("                }")
    lines += ["            }", "        }", "    }", ""]
    return "\n".join(lines)


def _direct_body(p, accumulators, channels):
    """Design 1's code for one direct path (after the block's Horner: radial_weight, radial_dt)."""
    a, l, L, kind = p["a"], p["l"], p["L"], p["form"]
    s = p["input_start"]
    q = [f"stage_row[{c} + channel]" for c in p["stage"]]
    b = ["                const float w = radial_weight;"]
    b += [f"                const float q{k} = {q[k]};" for k in range(2 * L + 1)]
    if a == 0:
        b.append(f"                const float x0 = x[input_base + {s} + channel];")
        dx = [accumulators[(s, 0)]]
    else:
        b += [f"                const float x{c} = x[input_base + {s + m * channels} + channel];"
              for m, c in enumerate("xyz")]
        dx = [accumulators[(s, m)] for m in range(3)]
    if L == 1:
        b.append("                const float h = q0 * nx + q1 * ny + q2 * nz;")
    elif L == 2:
        b += ["                const float qzz = -q0 - q3;",
              "                const float vx = q0 * nx + q1 * ny + q2 * nz, vy = q1 * nx + q3 * ny + q4 * nz,"
              " vz = q2 * nx + q4 * ny + qzz * nz;",
              "                const float h = nx * vx + ny * vy + nz * vz;"]
    elif L == 3:
        b += ["                const float q7 = -q0 - q3, q8 = -q1 - q5;",
              "                const float Qxx = q0 * nx + q1 * ny + q2 * nz, Qxy = q1 * nx + q3 * ny + q4 * nz,"
              " Qxz = q2 * nx + q4 * ny + q7 * nz;",
              "                const float Qyy = q3 * nx + q5 * ny + q6 * nz, Qyz = q4 * nx + q6 * ny + q8 * nz,"
              " Qzz = -Qxx - Qyy;",
              "                const float vx = Qxx * nx + Qxy * ny + Qxz * nz, vy = Qxy * nx + Qyy * ny + Qyz * nz,"
              " vz = Qxz * nx + Qyz * ny + Qzz * nz;",
              "                const float h = nx * vx + ny * vy + nz * vz;"]

    def acc(target, value):
        return f"                {target} = fmaf({value}, {target});"

    if kind == "scalar":
        d = dx[0]
        if L == 0:
            b += [acc(d, "w, q0"), "                pull_R = fmaf(radial_dt, x0 * q0, pull_R);"]
        elif L == 1:
            b += ["                const float wx = w * x0;",
                  acc("pull_Gx", "wx, q0"), acc("pull_Gy", "wx, q1"), acc("pull_Gz", "wx, q2"),
                  acc(d, "w, h"), "                pull_R = fmaf(radial_dt, x0 * h, pull_R);"]
        else:
            b += [f"                const float wx = {_f(L)} * w * x0;",
                  acc("pull_Gx", "wx, vx"), acc("pull_Gy", "wx, vy"), acc("pull_Gz", "wx, vz"),
                  acc(d, "w, h"), "                pull_R = fmaf(radial_dt, x0 * h, pull_R);"]
    elif kind == "raise":  # phi = x . grad h, grad phi = H x, dx = w grad h
        if L == 1:
            b += ["                pull_R = fmaf(radial_dt, xx * q0 + xy * q1 + xz * q2, pull_R);",
                  acc(dx[0], "w, q0"), acc(dx[1], "w, q1"), acc(dx[2], "w, q2")]
        elif L == 2:
            b += ["                pull_R = fmaf(radial_dt, 2.f * (xx * vx + xy * vy + xz * vz), pull_R);",
                  "                const float w2 = 2.f * w;",
                  acc(dx[0], "w2, vx"), acc(dx[1], "w2, vy"), acc(dx[2], "w2, vz"),
                  acc("pull_Gx", "w2, q0 * xx + q1 * xy + q2 * xz"),
                  acc("pull_Gy", "w2, q1 * xx + q3 * xy + q4 * xz"),
                  acc("pull_Gz", "w2, q2 * xx + q4 * xy + qzz * xz")]
        else:
            b += ["                pull_R = fmaf(radial_dt, 3.f * (xx * vx + xy * vy + xz * vz), pull_R);",
                  "                const float w3 = 3.f * w, w6 = 6.f * w;",
                  acc(dx[0], "w3, vx"), acc(dx[1], "w3, vy"), acc(dx[2], "w3, vz"),
                  acc("pull_Gx", "w6, Qxx * xx + Qxy * xy + Qxz * xz"),
                  acc("pull_Gy", "w6, Qxy * xx + Qyy * xy + Qyz * xz"),
                  acc("pull_Gz", "w6, Qxz * xx + Qyz * xy + Qzz * xz")]
    else:  # lower: phi = A h - B / (2L + 1) on the unit sphere
        k = 1.0 / (2 * L + 1)
        b.append("                const float A = xx * nx + xy * ny + xz * nz;")
        if L == 0:
            b += ["                const float wq = w * q0;",
                  acc("pull_Gx", "wq, xx"), acc("pull_Gy", "wq, xy"), acc("pull_Gz", "wq, xz"),
                  acc(dx[0], "wq, nx"), acc(dx[1], "wq, ny"), acc(dx[2], "wq, nz"),
                  "                pull_R = fmaf(radial_dt, A * q0, pull_R);"]
        elif L == 1:
            b += ["                const float B = xx * q0 + xy * q1 + xz * q2;",
                  f"                pull_R = fmaf(radial_dt, A * h - {_f(k)} * B, pull_R);",
                  f"                const float wh = w * h, wA = w * A, wk = w * {_f(k)};",
                  "                pull_Gx = fmaf(wh, xx, fmaf(wA, q0, pull_Gx));",
                  "                pull_Gy = fmaf(wh, xy, fmaf(wA, q1, pull_Gy));",
                  "                pull_Gz = fmaf(wh, xz, fmaf(wA, q2, pull_Gz));",
                  f"                {dx[0]} = fmaf(wh, nx, fmaf(-wk, q0, {dx[0]}));",
                  f"                {dx[1]} = fmaf(wh, ny, fmaf(-wk, q1, {dx[1]}));",
                  f"                {dx[2]} = fmaf(wh, nz, fmaf(-wk, q2, {dx[2]}));"]
        elif L == 2:
            b += ["                const float B2 = xx * vx + xy * vy + xz * vz;  // B / 2",
                  f"                pull_R = fmaf(radial_dt, A * h - {_f(2 * k)} * B2, pull_R);",
                  f"                const float wh = w * h, wA2 = 2.f * w * A, wk2 = {_f(2 * k)} * w;",
                  "                pull_Gx = fmaf(wh, xx, fmaf(wA2, vx, fmaf(-wk2, q0 * xx + q1 * xy + q2 * xz, pull_Gx)));",
                  "                pull_Gy = fmaf(wh, xy, fmaf(wA2, vy, fmaf(-wk2, q1 * xx + q3 * xy + q4 * xz, pull_Gy)));",
                  "                pull_Gz = fmaf(wh, xz, fmaf(wA2, vz, fmaf(-wk2, q2 * xx + q4 * xy + qzz * xz, pull_Gz)));",
                  f"                {dx[0]} = fmaf(wh, nx, fmaf(-wk2, vx, {dx[0]}));",
                  f"                {dx[1]} = fmaf(wh, ny, fmaf(-wk2, vy, {dx[1]}));",
                  f"                {dx[2]} = fmaf(wh, nz, fmaf(-wk2, vz, {dx[2]}));"]
        else:
            raise ValueError("LATE_PULLBACK: lower path with L = 3 needs l = 4")
    return "\n".join(b) + "\n"


def apply(source, schedule, blocks, fold_alpha, direct=(0, 1, 2, 3)):
    """``source`` (a rendered float32 FACTORED w' kernel with STAGE_DOUT and BLOCK_MESSAGE, before wprime_tuning)
    with the paths of edge degree in ``direct`` in the late-pullback form."""
    direct = {int(l) for l in direct}
    path_list = paths(schedule, blocks, fold_alpha)
    for p in path_list:
        p["direct"] = p["l"] in direct
    if not any(p["direct"] for p in path_list):
        return source
    channels = int(schedule.channels)
    if "stage_row[" not in source or PROLOGUE_START not in source or PROLOGUE_END not in source:
        raise ValueError("LATE_PULLBACK needs STAGE_DOUT's row prologue")
    if "STAGE_QUADS" in source or "cbase" in source:  # HOIST_DX's dx_lane is fine: the atomics stay as they are
        raise ValueError("LATE_PULLBACK: not with CHANNEL_VECTOR or STAGE_QUADS")
    if "edge_sum" in source or "edge_force" in source.lower():
        raise ValueError("LATE_PULLBACK: not with EDGE_FORCE")
    accumulators = {(int(s.start), int(s.component)): s.name for s in schedule.input_gradient_slots}
    head, decl, path_blocks, tail, rest = _loop_parts(source)
    by_start = {p["weight_start"]: p for p in path_list}
    out_blocks = []
    for blk in path_blocks:
        m = re.search(r"const int weight_index =\s*(\d+) \+ channel;", blk)
        if not m or int(m.group(1)) not in by_start:
            raise ValueError("LATE_PULLBACK: a path block without a known weight_index")
        p = by_start[int(m.group(1))]
        if not p["direct"]:
            out_blocks.append(blk)
            continue
        cut = blk.index("                acc_t weight_gradient")
        if "radial_weight" not in blk[:cut] or "radial_dt" not in blk[:cut]:
            raise ValueError("LATE_PULLBACK needs HORNER's radial_weight and radial_dt")
        out_blocks.append(blk[:cut] + f"                // LATE_PULLBACK path {p['index']} "
                          f"({p['a']}, {p['l']}, {p['L']})\n" + _direct_body(p, accumulators, channels) + "            }")
    hoisted = "radial_gradient *= sc;  // HOIST_SCALE" in source
    if not hoisted and "slope = radial_dt * sc;" not in source:
        raise ValueError("LATE_PULLBACK needs HORNER's slope")
    nread = ("sh_edge[{}]" if "const scalar_t* sh_edge" in source else "sh[edge_base + {}]")
    pre = ("    // LATE_PULLBACK: G = sum w grad phi and R = sum (dw/dt) phi over the direct paths; n in the sh frame.\n"
           + "".join(f"    const float n{c} = {nread.format(k + 1)} * {SQRT3_INV};\n" for k, c in enumerate("xyz"))
           + "    float pull_Gx = 0.f, pull_Gy = 0.f, pull_Gz = 0.f, pull_R = 0.f;\n")
    post = ("\n    // LATE_PULLBACK: the l = 1 slots take G / sqrt(3) (edge_sh3's backward: (I - n n^T) G / r); dr takes R.\n"
            f"    edge_gradient[1] += pull_Gx * {SQRT3_INV};\n"
            f"    edge_gradient[2] += pull_Gy * {SQRT3_INV};\n"
            f"    edge_gradient[3] += pull_Gz * {SQRT3_INV};\n"
            f"    if (sloped) radial_gradient += {'pull_R' if hoisted else 'pull_R * sc'};")
    new = head + pre + LOOP[:-2] + "{" + decl + "".join(out_blocks) + tail + "}" + post + rest
    # the prologue
    first, after = new.split(PROLOGUE_START, 1)
    after = after[after.index(PROLOGUE_END):]
    global APPLIED
    APPLIED += 1
    return first + _prologue(path_list, channels) + after


__all__ = ["apply", "conversion", "paths", "supported"]
