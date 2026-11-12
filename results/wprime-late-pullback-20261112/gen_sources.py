"""Step 2 sources: design 1 (four-component backward, delayed geometry differentiation) and the N1 control, both
cut from today's H200 default render (results/wprime-h200-20261111/sources/wprime-h200-skip+hoist.cu: same
signature, STAGE_DOUT prologue, STAGE_SH, SPLINE_W + HORNER table lookup, SKIP_OUTSIDE).

  d1-conv     design 1, complete: the STAGE_DOUT prologue converts the receiver row (q = M g per path and channel,
              one thread per channel) while it stages it; channel loop accumulates G (3) and R (1) per lane, dx via
              the hoisted pointer; epilogue reduces G, R and writes dL/dv = n R sc + (G - n (n.G)) / r to
              dsh[e, 0:3] (lab slot; production would write [E, 3] and drop the sh backward and the dr chain).
  d1-copy     the same channel loop and epilogue with today's plain copy prologue (rows converted elsewhere).
  n1-copy     the control: lab_cartesian.n1_source unchanged (per-path radial/angular force assembly with the
              Euler term, unhoisted dx) on today's head and tail; rows converted by a separate node kernel.
  n1-conv     the control with the in-prologue conversion of d1-conv.

    python gen_sources.py   (writes sources/*.cu)
"""
import re
from types import SimpleNamespace

from common import H200, HERE, setup

BASE = (H200 / "sources" / "wprime-h200-skip+hoist.cu").read_text()
PROLOGUE_START = "        if (stage_r >= 0 && stage_r < node_count) {\n"
PROLOGUE_END = "    __syncthreads();\n    for (int64_t e = stage_s"


def _f(v):
    text = f"{float(v):.9g}"
    if not any(c in text for c in ".en"):
        text += ".0"
    return text + "f"


def stage_offsets(s):
    from fastmace.kernels.oeq_streaming import wprime_backward as wp
    _, offsets, _ = wp._stage_layout(s.schedule, s.blocks)
    return offsets


def conversion_prologue(s, src):
    """Replace the plain copy with copy + conversion: thread = channel; per path q_k = sum_o M[k, o] g_o."""
    off = stage_offsets(s)
    lines = [PROLOGUE_START,
             "            // late-pullback: stage the receiver row converted to N1's Cartesian coefficients q = M g.",
             "            for (int channel = threadIdx.x; channel < CHANNEL_DIM; channel += blockDim.x) {"]
    for index, p in enumerate(s.paths):
        dim = 2 * p["L"] + 1
        lines.append(f"                {{  // path {index}: ({p['a']}, {p['l']}, {p['L']})")
        for o, (g, width, column) in enumerate(p["dout"]):
            lines.append(f"                const float g{o} = __ldg(dout{g} + int64_t(stage_r) * {width} + {column}"
                         " + channel);")
        m = p["M"].numpy()
        for k, (g, width, column) in enumerate(p["dout"]):
            terms = [f"{_f(m[k, o])} * g{o}" for o in range(dim) if abs(float(m[k, o])) > 1e-12]
            lines.append(f"                stage_row[{off[g] + column} + channel] = {' + '.join(terms) or '0.f'};")
        lines.append("                }")
    lines += ["            }", "        }", "    }", ""]
    head, rest = src.split(PROLOGUE_START, 1)
    rest = rest[rest.index(PROLOGUE_END):]
    return head + "\n".join(lines) + rest


def d1_loop(s):
    """Design 1's channel loop and epilogue (replaces everything from the edge-gradient declaration to the dsh/dr
    stores)."""
    off = stage_offsets(s)
    base = BASE
    horner = base[base.index("                scalar_t gv[PADDED_TERMS];"):base.index("                acc_t weight_gradient")]
    ch = int(s.schedule.channels)
    inputs = sorted({(p["input_start"], 2 * p["a"] + 1) for p in s.paths})
    L_ = ["    // late-pullback design 1: per lane G = sum w grad phi (3) and R = sum dw/dt phi over paths and channels;",
          "    // dL/dv once per edge after the reduction. n in the sh frame (sh_1 = sqrt(3) n).",
          "    const float nx = sh_edge[1] * 0.57735026918962576451f;",
          "    const float ny = sh_edge[2] * 0.57735026918962576451f;",
          "    const float nz = sh_edge[3] * 0.57735026918962576451f;",
          "    float Gx = 0.f, Gy = 0.f, Gz = 0.f, R = 0.f;",
          "    scalar_t* dx_lane = dx + input_base + lane;  // hoist",
          "    for (int channel = lane; channel < CHANNEL_DIM; channel += LOGICAL_GROUP_SIZE) {"]
    xs = {}
    for start, dim in inputs:
        if dim == 1:
            xs[start] = f"x{start}"
            L_ += [f"        const float x{start} = x[input_base + {start} + channel];", f"        float dx{start} = 0.f;"]
        else:
            xs[start] = f"x{start}"
            for m, c in enumerate("xyz"):
                L_ += [f"        const float x{start}{c} = x[input_base + {start + m * ch} + channel];",
                       f"        float dx{start}{c} = 0.f;"]
    for index, p in enumerate(s.paths):
        a, l, L = p["a"], p["l"], p["L"]
        b = ["        {", f"        // path {index}: ({a}, {l}, {L})",
             f"        const int weight_index = {p['weight_start']} + channel;", horner,
             "        const float w = radial_weight;"]
        for m, (g, width, column) in enumerate(p["dout"]):
            b.append(f"        const float q{m} = stage_row[{off[g] + column} + channel];")
        # h, v (grad h = L v for L >= 2), and Q for the Hessian
        if L == 2:
            b += ["        const float qzz = -q0 - q3;",
                  "        const float vx = q0 * nx + q1 * ny + q2 * nz, vy = q1 * nx + q3 * ny + q4 * nz,"
                  " vz = q2 * nx + q4 * ny + qzz * nz;",
                  "        const float h = nx * vx + ny * vy + nz * vz;"]
        elif L == 3:
            b += ["        const float q7 = -q0 - q3, q8 = -q1 - q5;",
                  "        const float Qxx = q0 * nx + q1 * ny + q2 * nz, Qxy = q1 * nx + q3 * ny + q4 * nz,"
                  " Qxz = q2 * nx + q4 * ny + q7 * nz;",
                  "        const float Qyy = q3 * nx + q5 * ny + q6 * nz, Qyz = q4 * nx + q6 * ny + q8 * nz,"
                  " Qzz = -Qxx - Qyy;",
                  "        const float vx = Qxx * nx + Qxy * ny + Qxz * nz, vy = Qxy * nx + Qyy * ny + Qyz * nz,"
                  " vz = Qxz * nx + Qyz * ny + Qzz * nz;",
                  "        const float h = nx * vx + ny * vy + nz * vz;"]
        x = xs[p["input_start"]]
        if a == 0:
            if L == 0:
                b += [f"        dx{p['input_start']} = fmaf(w, q0, dx{p['input_start']});",
                      f"        R = fmaf(radial_dt, {x} * q0, R);"]
            elif L == 1:
                b += ["        const float h = q0 * nx + q1 * ny + q2 * nz;",
                      f"        const float wx = w * {x};",
                      "        Gx = fmaf(wx, q0, Gx); Gy = fmaf(wx, q1, Gy); Gz = fmaf(wx, q2, Gz);",
                      f"        dx{p['input_start']} = fmaf(w, h, dx{p['input_start']});",
                      f"        R = fmaf(radial_dt, {x} * h, R);"]
            else:
                b += [f"        const float wx = {_f(L)} * w * {x};",
                      "        Gx = fmaf(wx, vx, Gx); Gy = fmaf(wx, vy, Gy); Gz = fmaf(wx, vz, Gz);",
                      f"        dx{p['input_start']} = fmaf(w, h, dx{p['input_start']});",
                      f"        R = fmaf(radial_dt, {x} * h, R);"]
        elif L == l + 1:  # phi = x . grad h, grad phi = H x, dx = w grad h
            d = p["input_start"]
            if L == 1:
                b += [f"        R = fmaf(radial_dt, {x}x * q0 + {x}y * q1 + {x}z * q2, R);",
                      f"        dx{d}x = fmaf(w, q0, dx{d}x); dx{d}y = fmaf(w, q1, dx{d}y); dx{d}z = fmaf(w, q2, dx{d}z);"]
            elif L == 2:  # grad h = 2 v, H x = 2 Q x
                b += [f"        R = fmaf(radial_dt, 2.f * ({x}x * vx + {x}y * vy + {x}z * vz), R);",
                      "        const float w2 = 2.f * w;",
                      f"        dx{d}x = fmaf(w2, vx, dx{d}x); dx{d}y = fmaf(w2, vy, dx{d}y); dx{d}z = fmaf(w2, vz, dx{d}z);",
                      f"        Gx = fmaf(w2, q0 * {x}x + q1 * {x}y + q2 * {x}z, Gx);",
                      f"        Gy = fmaf(w2, q1 * {x}x + q3 * {x}y + q4 * {x}z, Gy);",
                      f"        Gz = fmaf(w2, q2 * {x}x + q4 * {x}y + qzz * {x}z, Gz);"]
            else:  # L = 3: grad h = 3 v, H x = 6 Q(n) x
                b += [f"        R = fmaf(radial_dt, 3.f * ({x}x * vx + {x}y * vy + {x}z * vz), R);",
                      "        const float w3 = 3.f * w, w6 = 6.f * w;",
                      f"        dx{d}x = fmaf(w3, vx, dx{d}x); dx{d}y = fmaf(w3, vy, dx{d}y); dx{d}z = fmaf(w3, vz, dx{d}z);",
                      f"        Gx = fmaf(w6, Qxx * {x}x + Qxy * {x}y + Qxz * {x}z, Gx);",
                      f"        Gy = fmaf(w6, Qxy * {x}x + Qyy * {x}y + Qyz * {x}z, Gy);",
                      f"        Gz = fmaf(w6, Qxz * {x}x + Qyz * {x}y + Qzz * {x}z, Gz);"]
        else:  # L = l - 1: phi = A h - B / (2L + 1) on the unit sphere
            d = p["input_start"]
            k = 1.0 / (2 * L + 1)
            b.append(f"        const float A = {x}x * nx + {x}y * ny + {x}z * nz;")
            if L == 0:
                b += ["        const float wq = w * q0;",
                      f"        Gx = fmaf(wq, {x}x, Gx); Gy = fmaf(wq, {x}y, Gy); Gz = fmaf(wq, {x}z, Gz);",
                      f"        dx{d}x = fmaf(wq, nx, dx{d}x); dx{d}y = fmaf(wq, ny, dx{d}y); dx{d}z = fmaf(wq, nz, dx{d}z);",
                      "        R = fmaf(radial_dt, A * q0, R);"]
            elif L == 1:  # grad h = q, H = 0
                b += ["        const float h = q0 * nx + q1 * ny + q2 * nz;",
                      f"        const float B = {x}x * q0 + {x}y * q1 + {x}z * q2;",
                      f"        R = fmaf(radial_dt, A * h - {_f(k)} * B, R);",
                      "        const float wh = w * h, wA = w * A, wk = w * " + _f(k) + ";",
                      f"        Gx = fmaf(wh, {x}x, fmaf(wA, q0, Gx)); Gy = fmaf(wh, {x}y, fmaf(wA, q1, Gy));"
                      f" Gz = fmaf(wh, {x}z, fmaf(wA, q2, Gz));",
                      f"        dx{d}x = fmaf(wh, nx, fmaf(-wk, q0, dx{d}x)); dx{d}y = fmaf(wh, ny, fmaf(-wk, q1, dx{d}y));"
                      f" dx{d}z = fmaf(wh, nz, fmaf(-wk, q2, dx{d}z));"]
            else:  # L = 2: grad h = 2 v, H x = 2 Q x
                b += [f"        const float B2 = {x}x * vx + {x}y * vy + {x}z * vz;  // B / 2",
                      f"        R = fmaf(radial_dt, A * h - {_f(2 * k)} * B2, R);",
                      "        const float wh = w * h, wA2 = 2.f * w * A, wk2 = " + _f(2 * k) + " * w;",
                      f"        Gx = fmaf(wh, {x}x, fmaf(wA2, vx, fmaf(-wk2, q0 * {x}x + q1 * {x}y + q2 * {x}z, Gx)));",
                      f"        Gy = fmaf(wh, {x}y, fmaf(wA2, vy, fmaf(-wk2, q1 * {x}x + q3 * {x}y + q4 * {x}z, Gy)));",
                      f"        Gz = fmaf(wh, {x}z, fmaf(wA2, vz, fmaf(-wk2, q2 * {x}x + q4 * {x}y + qzz * {x}z, Gz)));",
                      f"        dx{d}x = fmaf(wh, nx, fmaf(-wk2, vx, dx{d}x)); dx{d}y = fmaf(wh, ny, fmaf(-wk2, vy, dx{d}y));"
                      f" dx{d}z = fmaf(wh, nz, fmaf(-wk2, vz, dx{d}z));"]
        b.append("        }")
        L_ += b
    for start, dim in inputs:
        if dim == 1:
            L_.append(f"        atomicAdd(dx_lane + {start}, dx{start});")
        else:
            for m, c in enumerate("xyz"):
                L_.append(f"        atomicAdd(dx_lane + {start + m * ch}, dx{start}{c});")
    L_ += ["        dx_lane += LOGICAL_GROUP_SIZE;",
           "    }",
           "    #pragma unroll",
           "    for (int o = 16; o; o >>= 1) {",
           "        Gx += __shfl_xor_sync(FULL_MASK, Gx, o);",
           "        Gy += __shfl_xor_sync(FULL_MASK, Gy, o);",
           "        Gz += __shfl_xor_sync(FULL_MASK, Gz, o);",
           "        R += __shfl_xor_sync(FULL_MASK, R, o);",
           "    }",
           "    if (lane == 0) {",
           "        const float ri = 1.f / r;",
           "        const float Rs = sloped ? R * sc : 0.f;",
           "        const float nG = nx * Gx + ny * Gy + nz * Gz;",
           "        dsh[edge_base + 0] = nx * Rs + (Gx - nx * nG) * ri;",
           "        dsh[edge_base + 1] = ny * Rs + (Gy - ny * nG) * ri;",
           "        dsh[edge_base + 2] = nz * Rs + (Gz - nz * nG) * ri;",
           "        dr[e] = 0.f;",
           "    }"]
    return "\n".join(L_) + "\n"


def d1_source(s, convert):
    src = BASE
    head = src[:src.index("    acc_t edge_gradient[EDGE_DIM]")]
    tail = src[src.index("    }  // STAGE_DOUT: the run's edges"):]
    out = head + d1_loop(s) + tail
    return conversion_prologue(s, out) if convert else out


def n1_control(s, convert):
    import lab_cartesian as lc
    lc._render = lambda kernel, stage: BASE  # today's render as N1's head and tail
    kernel = SimpleNamespace(table_w=True, horner=True, table_pairs=False, hoist_scale=False,
                             channels=s.kernel.channels, blocks=s.blocks, schedule=s.schedule)
    src = lc.n1_source(kernel, s.paths, True)
    return conversion_prologue(s, src) if convert else src


def main():
    s = setup()
    out = HERE / "sources"
    out.mkdir(exist_ok=True)
    (out / "skip+hoist.cu").write_text(BASE)
    for name, text in {"d1-conv": d1_source(s, True), "d1-copy": d1_source(s, False),
                       "n1-copy": n1_control(s, False), "n1-conv": n1_control(s, True)}.items():
        (out / f"{name}.cu").write_text(text)
        print("wrote", name, len(text.splitlines()), "lines")


if __name__ == "__main__":
    main()
