"""CPU check of a LATE_PULLBACK-transformed w' source (any mask; spherical and direct path blocks): the STAGE_DOUT
prologue, the per-edge setup after the edge-gradient declaration, the channel-loop body and the post-loop lines are
translated statement by statement to Python and run in float64 per edge and channel (Horner replaced by the exact
table value: FOLD_ALPHA's alpha w and its r-derivative, sc = 1, sloped). Output: dsh [16] and dr per edge, mapped to
dE/dv = J^T dsh + n dr with e3nn's harmonics (autograd), and dx; compared with the direct CG contraction (float64).

    python emulate2.py SOURCE.cu [SOURCE.cu ...]   (sources before wprime_tuning, i.e. without dx_lane)
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


def hexfloat(m):
    return repr(float.fromhex(m.group(1)))


def translate(text):
    text = "\n".join(line.split("//")[0] for line in text.splitlines() if not line.strip().startswith("#"))
    text = re.sub(r"\bif \((sloped|true)\)", "", text)
    text = text.replace("{", ";").replace("}", ";")
    stmts = []
    for st in split_top(text, ";"):
        st = " ".join(st.split())
        if not st or st.startswith(("atomicAdd", "for ", "else")):
            continue
        st = re.sub(r"static_cast<scalar_t>\(([^()]*)\)", r"(\1)", st)
        st = re.sub(r"(0x[0-9a-fA-F.]+p[+-]?\d+)f?", hexfloat, st)
        st = re.sub(r"\b(acc_t|scalar_t)\(([^()]*)\)", r"(\2)", st)
        st = re.sub(r"^(const )?(float|int|int64_t|acc_t|scalar_t) ", "", st)
        st = re.sub(r"__ldg\((dout\d) \+ int64_t\(stage_r\) \* (\d+) \+ (\d+) \+ channel\)",
                    r"\1[\2 * stage_r + \3 + channel]", st)
        st = re.sub(r"(\d+\.\d*(?:e[+-]?\d+)?|\d+\.)f\b", r"\1", st)
        st = re.sub(r"\b(\d+)\.(?=\D|$)", r"\1.0", st)
        for part in split_top(st, ","):
            stmts.append(part.strip())
    return "\n".join(stmts)


def horner_out(body):
    """Each path block's Horner (gv loads to the weight gradient or the LATE_PULLBACK body) -> the exact table
    values W / WP of the block's weight_start."""
    out, pos = "", 0
    for m in re.finditer(r"const int weight_index =\s*(\d+) \+ channel;", body):
        a = body.index("scalar_t gv[PADDED_TERMS];", m.end())
        ends = [body.find(k, a) for k in ("acc_t weight_gradient", "// LATE_PULLBACK path")]
        b = min(e for e in ends if e >= 0)
        out += body[pos:m.end()] + f"\nradial_weight = W[{m.group(1)}];\nradial_dt = WP[{m.group(1)}];\n"
        pos = b
    return out + body[pos:]


def check(s, path, E=8, C=128, seed=11, only=None):
    from fastmace.kernels.oeq_streaming.wprime_backward import factored_paths
    from fastmace.kernels.oeq_streaming.wprime_tuning import LOOP, _block
    from lab_cartesian import harmonics
    src = (HERE / path).read_text() if not str(path).startswith("/") else open(path).read()
    assert "dx_lane" not in src
    pro = src[src.index("        if (stage_r >= 0 && stage_r < node_count) {"):src.index("    __syncthreads();\n    for (int64_t e")]
    if "for (int channel = threadIdx.x" in pro:
        pro = pro[pro.index("for (int channel = threadIdx.x"):]
        pro = pro[pro.index("\n") + 1:]
        py_pro, plain = compile(translate(pro), "prologue", "exec"), False
    else:
        py_pro, plain = None, True  # the plain copy: stage_row = the row
    setup_text = src[src.index("    acc_t radial_gradient = acc_t(0);"):src.index(LOOP)]
    begin = src.index(LOOP)
    open_, end = _block(src, begin + len(LOOP) - 2)
    body = horner_out(src[open_ + 1:end - 1])
    post = src[end:src.index("    // BUTTERFLY", end)]
    py_setup = compile(translate(setup_text.replace("acc_t radial_gradient = acc_t(0);", "")), "setup", "exec")
    py_body = compile(translate(body), "body", "exec")
    py_post = compile(translate(post), "post", "exec")

    alphas = [f["alpha"] for f in factored_paths(s.schedule, True)]
    gen = torch.Generator().manual_seed(seed)
    f64 = torch.float64
    v = torch.randn(E, 3, generator=gen, dtype=f64)
    v = v / v.norm(dim=-1, keepdim=True) * (0.8 + 4.5 * torch.rand(E, 1, generator=gen, dtype=f64))
    xrow = torch.randn(E, 512, generator=gen, dtype=f64)
    widths = {g: w for p in s.paths for g, w, _ in p["dout"]}
    dout = {g: torch.randn(E, w, generator=gen, dtype=f64) for g, w in widths.items()}
    rc = [torch.randn(C, 4, generator=gen, dtype=f64) * torch.tensor([1.0, 0.5, 0.2, 0.05], dtype=f64)
          for _ in s.paths]
    if only is not None:
        rc = [c if k in only else c * 0 for k, c in enumerate(rc)]
    from fastmace.kernels.oeq_streaming.wprime_backward import _stage_layout
    _, offsets, _ = _stage_layout(s.schedule, s.blocks)

    def radial(coef, r):
        rp = torch.stack([torch.ones_like(r), r, r * r, r ** 3], -1)
        dp = torch.stack([torch.zeros_like(r), torch.ones_like(r), 2 * r, 3 * r * r], -1)
        return rp @ coef.T, dp @ coef.T

    vv = v.clone().requires_grad_(True)
    xr = xrow.clone().requires_grad_(True)
    r = vv.norm(dim=-1)
    energy = 0
    for p, cp in zip(s.paths, rc):
        a, ch = p["a"], torch.arange(C)
        xi = torch.stack([xr[:, p["input_start"] + m * C + ch] for m in range(2 * a + 1)], -1)
        go = torch.stack([dout[g][:, col + ch] for g, _, col in p["dout"]], -1)
        y = harmonics(p["l"], vv, s.frame, normalize=True)
        w, _ = radial(cp, r)
        energy = energy + (w * torch.einsum("ijo,eci,ej,eco->ec", torch.as_tensor(p["coef"]), xi, y, go)).sum()
    ref_v, ref_x = torch.autograd.grad(energy, [vv, xr])

    def fmaf(a, b, c):
        return a * b + c

    got_v = torch.zeros(E, 3, dtype=f64)
    got_x = torch.zeros(E, 512, dtype=f64)
    stage_size = 5120
    for e in range(E):
        rr = float(v[e].norm())
        sh = torch.cat([harmonics(l, v[e:e + 1], s.frame, normalize=True)[0] for l in range(4)]).tolist()
        env = dict(fmaf=fmaf, stage_r=0, stage_row=[0.0] * stage_size, x=xrow[e].tolist(), input_base=0,
                   sh_edge=sh, sh=sh, edge_base=0, sloped=True, sc=1.0, edge_gradient=[0.0] * 16,
                   radial_gradient=0.0)
        for g in dout:
            env[f"dout{g}"] = dout[g][e].tolist()
            if plain:
                env["stage_row"][offsets[g]:offsets[g] + widths[g]] = dout[g][e].tolist()
        if not plain:
            for channel in range(C):
                env["channel"] = channel
                exec(py_pro, env)
        exec(py_setup, env)
        rt = torch.tensor([rr], dtype=f64)
        tab = [radial(cp, rt) for cp in rc]
        for channel in range(C):
            env["channel"] = channel
            env["W"] = {p["weight_start"]: alphas[k] * float(tab[k][0][0, channel]) for k, p in enumerate(s.paths)}
            env["WP"] = {p["weight_start"]: alphas[k] * float(tab[k][1][0, channel]) for k, p in enumerate(s.paths)}
            exec(py_body, env)
            for slot in s.schedule.input_gradient_slots:
                got_x[e, int(slot.start) + int(slot.component) * C + channel] += env[slot.name]
        exec(py_post, env)
        ve = v[e:e + 1].clone().requires_grad_(True)
        yall = torch.cat([harmonics(l, ve, s.frame, normalize=True)[0] for l in range(4)])
        (jt,) = torch.autograd.grad((yall * torch.tensor(env["edge_gradient"], dtype=f64)).sum(), [ve])
        got_v[e] = jt[0] + v[e] / rr * env["radial_gradient"]
    return dict(source=str(path), dv_max_rel=float((got_v - ref_v[:E]).abs().max() / ref_v.abs().max()),
                dx_max_rel=float((got_x - ref_x).abs().max() / ref_x.abs().max()))


if __name__ == "__main__":
    s = setup()
    res = [check(s, p) for p in sys.argv[1:]]
    for r in res:
        print(json.dumps(r))
    sys.exit(0 if all(r["dv_max_rel"] < 1e-8 and r["dx_max_rel"] < 1e-8 for r in res) else 1)
