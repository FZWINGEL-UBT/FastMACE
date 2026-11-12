"""CPU check of the GENERATED design-1 source (sources/d1-conv.cu): its prologue conversion statements and its
channel-loop statements are translated line by line to Python and executed in float64 for each edge and channel
(Horner replaced by the exact w and w' = dw/dr, sc = 1); the epilogue formula is applied after the channel sum.
Compared with the direct CG contraction (autograd, float64), dE/dv in the sh frame and dE/dx.

    python emulate_check.py [SOURCE]
"""
import json
import math
import re
import sys

import torch

from common import HERE, setup


def split_top(text, sep):
    out, depth, cur = [], 0, ""
    for ch in text:
        depth += ch in "(["
        depth -= ch in ")]"
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def translate(block):
    """C statements (declarations, assignments, fmaf) -> Python statements."""
    stmts = []
    for line in block.splitlines():
        line = line.split("//")[0].strip()
        if not line or line in ("{", "}") or line.startswith(("for ", "#pragma", "atomicAdd", "dx_lane")):
            continue
        line = line.strip("{} ")
        for st in split_top(line, ";"):
            st = st.strip()
            if not st:
                continue
            st = re.sub(r"^(const )?(float|int) ", "", st)
            st = re.sub(r"__ldg\((dout\d) \+ int64_t\(stage_r\) \* (\d+) \+ (\d+) \+ channel\)",
                        r"\1[\2 * stage_r + \3 + channel]", st)
            st = re.sub(r"(\d+\.\d*(?:e[+-]?\d+)?|\d+\.)f\b", r"\1", st)
            st = re.sub(r"\b(\d+)\.(?=\D|$)", r"\1.0", st)
            for part in split_top(st, ","):  # "a = X, b = Y"
                stmts.append(part.strip())
    return "\n".join(stmts)


def main(path):
    from lab_cartesian import harmonics
    s = setup()
    src = (HERE / path).read_text()
    prologue = src[src.index("for (int channel = threadIdx.x"):src.index("    __syncthreads();\n    for (int64_t e")]
    prologue = prologue[prologue.index("\n") + 1:]
    loop = src[src.index("    for (int channel = lane; channel < CHANNEL_DIM"):src.index("        atomicAdd(dx_lane")]
    loop = loop[loop.index("\n") + 1:]
    horner = src[src.index("                scalar_t gv[PADDED_TERMS];"):src.index("        const float w = radial_weight;")]
    paths_code = loop.split("        {\n        // path ")
    pre_code, path_blocks = paths_code[0], paths_code[1:]
    py_pre = translate(pre_code)
    py_paths = []
    for blk in path_blocks:
        index = int(blk.split(":")[0])
        body = blk[blk.index("\n") + 1:]
        hstart = body.index("                scalar_t gv[PADDED_TERMS];")
        hend = body.index("        const float w = radial_weight;")
        body = body[:hstart] + f"        radial_weight = W[{index}]\n        radial_dt = WP[{index}]\n" + body[hend:]
        py_paths.append(compile(translate(body), f"path{index}", "exec"))
    py_prologue = compile(translate(prologue), "prologue", "exec")
    py_pre = compile(py_pre, "pre", "exec")

    E, C = 12, 128
    gen = torch.Generator().manual_seed(7)
    f64 = torch.float64
    v = torch.randn(E, 3, generator=gen, dtype=f64)
    v = v / v.norm(dim=-1, keepdim=True) * (0.8 + 4.5 * torch.rand(E, 1, generator=gen, dtype=f64))
    xrow = torch.randn(E, 512, generator=gen, dtype=f64)  # the sender's x row, kernel layout (ir_mul)
    widths = {}
    for p in s.paths:
        for g, width, _ in p["dout"]:
            widths[g] = width
    dout = {g: torch.randn(E, w, generator=gen, dtype=f64) for g, w in widths.items()}
    rc = [torch.randn(C, 4, generator=gen, dtype=f64) * torch.tensor([1.0, 0.5, 0.2, 0.05], dtype=f64)
          for _ in s.paths]

    def radial(coef, r):
        rp = torch.stack([torch.ones_like(r), r, r * r, r ** 3], -1)
        dp = torch.stack([torch.zeros_like(r), torch.ones_like(r), 2 * r, 3 * r * r], -1)
        return rp @ coef.T, dp @ coef.T

    # reference: E = sum w C x Y g with x, g gathered from the kernel layouts
    vv = v.clone().requires_grad_(True)
    xr = xrow.clone().requires_grad_(True)
    r = vv.norm(dim=-1)
    energy = 0
    for p, cp in zip(s.paths, rc):
        a, ch = p["a"], torch.arange(C)
        xi = torch.stack([xr[:, p["input_start"] + m * C + ch] for m in range(2 * a + 1)], -1)  # [E, C, 2a+1]
        go = torch.stack([dout[g][:, col + ch] for g, _, col in p["dout"]], -1)  # [E, C, 2L+1]
        y = harmonics(p["l"], vv, s.frame, normalize=True)
        w, _ = radial(cp, r)
        energy = energy + (w * torch.einsum("ijo,eci,ej,eco->ec", torch.as_tensor(p["coef"]), xi, y, go)).sum()
    ref_v, ref_x = torch.autograd.grad(energy, [vv, xr])

    def fmaf(a, b, c):
        return a * b + c

    got_v = torch.zeros(E, 3, dtype=f64)
    got_x = torch.zeros(E, 512, dtype=f64)
    stage_size = max(int(k) for k in re.findall(r"stage_row\[(\d+) \+ channel\]", src)) + C
    for e in range(E):
        rr = float(v[e].norm())
        n = (v[e] / rr).tolist()
        env = dict(fmaf=fmaf, stage_r=0, stage_row=[0.0] * stage_size, x=xrow[e].tolist(), input_base=0,
                   sh_edge=[0.0, math.sqrt(3) * n[0], math.sqrt(3) * n[1], math.sqrt(3) * n[2]])
        for g in dout:
            env[f"dout{g}"] = dout[g][e].tolist()
        for channel in range(C):
            env["channel"] = channel
            exec(py_prologue, env)
        rt = torch.tensor([rr], dtype=f64)
        W, WP = zip(*[radial(cp, rt) for cp in rc])
        exec(compile("nx = sh_edge[1] * 0.57735026918962576451\nny = sh_edge[2] * 0.57735026918962576451\n"
                     "nz = sh_edge[3] * 0.57735026918962576451\nGx = Gy = Gz = R = 0.0", "init", "exec"), env)
        for channel in range(C):
            env["channel"] = channel
            env["W"] = [float(w[0, channel]) for w in W]
            env["WP"] = [float(w[0, channel]) for w in WP]
            exec(py_pre, env)
            for code in py_paths:
                exec(code, env)
            for name in [k for k in env if re.fullmatch(r"dx\d+[xyz]?", k)]:
                start = int(re.findall(r"\d+", name)[0])
                m = "xyz".index(name[-1]) if name[-1] in "xyz" else 0
                got_x[e, start + m * C + channel] += env[name]
        G = torch.tensor([env["Gx"], env["Gy"], env["Gz"]], dtype=f64)
        nn = torch.tensor(n, dtype=f64)
        got_v[e] = nn * env["R"] + (G - nn * (nn @ G)) / rr
    res = dict(source=path, edges=E, channels=C,
               dv_max_rel=float((got_v - ref_v).abs().max() / ref_v.abs().max()),
               dx_max_rel=float((got_x - ref_x).abs().max() / ref_x.abs().max()))
    print(json.dumps(res))
    (HERE / "emulate_check.json").write_text(json.dumps(res, indent=1) + "\n")
    return 0 if res["dv_max_rel"] < 1e-12 and res["dx_max_rel"] < 1e-12 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "sources/d1-conv.cu"))
