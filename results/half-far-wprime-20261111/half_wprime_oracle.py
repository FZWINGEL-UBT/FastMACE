"""HALF_FAR oracle (11 Nov 2026): interaction 2's far edges through a float16 emulation of the w' kernel itself.

The production plan (conductor, 11 Nov): only the backward's w' kernel runs far edges in half2, with two far edges of
one receiver per ``__half2`` (consecutive far edges in the receiver's run; an odd last one stays float32). The forward
stays float32. So the rounding to model is exactly w''s arithmetic for those edges:

* the kernel's own channel loop (``wprime-real.cu``, captured on the A100: FACTORED, HORNER, SCALAR_PATHS) is
  translated statement by statement into numpy, so the paths, the CG terms, their order and the accumulators are the
  kernel's; each statement rounds once to float16 (``a += b * c`` as one HFMA2: exact product, one rounding);
* inputs as the kernel holds them: x, dout, the harmonics, the CG constants, t and 1/h in float16; the table rows
  in float16 (power basis of the radial spline on its knots, Hermite from float64 values and slopes as
  ``rewrites.radial.spline``), w and dw/dt by the dual Horner pass in float16;
* per lane (channel = lane + 32 k, k = 0..3) the edge gradient and radial gradient accumulate in float16 over the
  four channels; the 32 lanes are reduced in float32; dx gets one float32 atomic per value.

The difference (float16 emulation - the same translated loop in float64) in dx, dsh and dr per far half edge is
added to the float32 MACE model's backward (a tap on interaction 2's tensor product and on the lengths). Score: max
force / stress error against the float64 model over the float32 model's own error (the gate allows 3x; the
conductor's stop rule for this stage is 2x). Self-check: the float64 translation's dx, dsh and weight gradient
against autograd's (the layout mapping and FOLD_ALPHA's per-path factor, fitted per path, must reproduce them).

CPU only. Usage: python half_wprime_oracle.py STRUCTURE.cif REPEAT OUT.json
"""
import copy
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from ase.build import make_supercell
from ase.io import read

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "a100-colab-ideas-20261111" / "screen1" / "ghoist" / "sources" / "wprime-real.cu"
CHECKPOINT = Path.home() / "models/mace-omat-0-medium.model"
THRESHOLDS = [5.0, 5.5, 5.8]
VARIANTS = [("none", ""), ("all", ""), ("dx", ""), ("dsh", ""), ("dr", ""), ("all", "w"), ("all", "wd")]
if len(sys.argv) > 5:
    VARIANTS = [tuple((v + "/").split("/")[:2]) for v in sys.argv[5].split(",")]
# (parts injected, scaling): "w" each far edge's table slab times a power of two (max |coef| -> [2^10, 2^11)), the
# outputs divided by it ("w4": max |coef| -> [2^4, 2^5)); "r": slope and radial_gradient in float32 (the half
# weight gradient converted, one float32 FMA per channel and path); "wd" also each receiver's dout row (max -> [2^4, 2^5)). "none": the float32 run again (noise).
SIGMA, SEED = 0.05, 1
R_LO, BASE_INTERVALS, BAND = 0.3, 255, 2.5  # rewrites.radial: the far band [2.5, r_max) at the base spacing


# ---------------------------------------------------------------------------------------------------------------
# the kernel's channel loop as numpy

def translate(src):
    """Python source of ``loop(X, D, SH, G, T, SC, R, c0, st)`` for one lane block (channels c0..c0+31)."""
    a = src.index("    acc_t radial_gradient = acc_t(0);\n") + len("    acc_t radial_gradient = acc_t(0);\n")
    body = re.sub(r"//[^\n]*", "", src[a:src.index("    // BUTTERFLY")])
    out, path = [], -1
    lanes = "c0:c0 + 32"

    def expr(e):
        e = re.sub(r"\s+", " ", e).strip()
        e = re.sub(r"x\[input_base \+ (\d+) \+ channel\]", lambda m: f"X[:, {m.group(1)} + c0:{m.group(1)} + c0 + 32]", e)
        e = re.sub(r"stage_row\[(\d+) \+ (\d+) \+ channel\]",
                   lambda m: f"D[:, {int(m.group(1)) + int(m.group(2))} + c0:{int(m.group(1)) + int(m.group(2))} + c0 + 32]", e)
        e = re.sub(r"sh_edge\[(\d+)\]", r"SH[:, \1:\1 + 1]", e)
        e = re.sub(r"static_cast<scalar_t>\(([-0-9a-fx.p+]+)f\)", lambda m: f"R({float.fromhex(m.group(1))!r})", e)
        e = re.sub(r"edge_gradient\[(\d+)\]", r"st['eg'][\1]", e)
        e = e.replace("radial_gradient", "st['rg']")
        return e

    for raw in re.split(r"[;{}]", body):
        t = re.sub(r"\s+", " ", raw).strip()
        if not t:
            continue
        if m := re.fullmatch(r"const int weight_index = (\d+) \+ channel", t):
            path += 1
            out.append(f"wb = {m.group(1)}; p = {path}")
        elif t == "scalar_t radial_weight = gv[TERMS - 1]":
            out.append("radial_weight, radial_dt = horner(G, T, R, wb, c0)")
        elif m := re.fullmatch(r"acc_t (\w+) = acc_t\(0\)", t):
            if m.group(1) not in ("slope",):
                out.append(f"{m.group(1)} = 0.0")
        elif m := re.fullmatch(r"const scalar_t (\w+) = (.*)", t):
            rhs = expr(m.group(2))
            out.append(f"{m.group(1)} = R({rhs})" if "*" in rhs.replace("R(", "") else f"{m.group(1)} = {rhs}")
        elif (m := re.fullmatch(r"(\S+) ([+-])= (.+)", t)) and " * " in m.group(3) and m.group(1) not in (
                "radial_gradient",):
            lhs = expr(m.group(1))
            b, c = m.group(3).split(" * ", 1)
            out.append(f"{lhs} = R({lhs} {m.group(2)} {expr(b)} * {expr(c)})")
        elif t == "slope = radial_dt * sc":
            out.append("slope = st['R2'](radial_dt * SC)")
        elif t == "radial_gradient += weight_gradient * slope":
            out.append("st['wg'][p][:, c0:c0 + 32] = weight_gradient")
            out.append("st['rg'] = st['R2'](st['rg'] + weight_gradient * slope)")
        elif m := re.fullmatch(r"atomicAdd\( dx \+ input_base \+ (\d+) \+ channel, scalar_t\((\w+)\)\)", t):
            out.append(f"st['dx'][:, {m.group(1)} + c0:{m.group(1)} + c0 + 32] += {m.group(2)}")
        elif re.match(r"(const int64_t radial_weight_index|scalar_t gv|if \(inside\)|const float4|#pragma|k [<>]|\+\+k|--k|"
                      r"gv\[|else|radial_dt = radial_dt|radial_weight = radial_weight|scalar_t radial_dt|"
                      r"if \(sloped\)|for \(int channel|channel [<+])", t):
            continue
        else:
            raise SystemExit(f"untranslated statement: {t!r}")
    code = "def loop(X, D, SH, G, T, SC, R, c0, st):\n" + "".join(f"    {line}\n" for line in out)
    return code, path + 1


def horner(G, T, R, wb, c0):
    """The dual Horner pass: G [E, 4, Q] power-basis coefficients (as the kernel holds them), T [E, 1]."""
    g = G[:, :, wb + c0:wb + c0 + 32]
    w, dt = g[:, 3], 0.0
    for k in (2, 1, 0):
        dt = R(dt * T + w)
        w = R(w * T + g[:, k])
    return w, dt


def emulate(loop, X, D, SH, G, T, SC, half, paths, rg32=False):
    """dx [E, 512], dsh [E, 16], dr [E] and the weight gradients [paths][E, 128] of the translated loop."""
    if half:
        def R(v):
            return np.asarray(v, dtype=np.float64).astype(np.float16).astype(np.float64)
        X, D, SH, G, T, SC = (R(a) for a in (X, D, SH, G, T, SC))
    else:
        def R(v):
            return v
    E = X.shape[0]
    st = dict(eg=[np.zeros((E, 32)) for _ in range(16)], rg=np.zeros((E, 32)), dx=np.zeros((E, 512)),
              wg=[np.zeros((E, 128)) for _ in range(paths)], R2=(lambda v: v) if rg32 else R)
    for c0 in (0, 32, 64, 96):
        loop(X, D, SH, G, T, SC, R, c0, st)
    return st["dx"], np.stack([e.sum(1) for e in st["eg"]], 1), st["rg"].sum(1), st["wg"]


# ---------------------------------------------------------------------------------------------------------------
# the model side

def structure(cif, repeat):
    atoms = read(cif)
    atoms = make_supercell(atoms, np.eye(3, dtype=int) * repeat) if len(atoms) < 10 else atoms.repeat(repeat)
    rng = np.random.default_rng(SEED)
    atoms.positions += rng.normal(0.0, SIGMA, atoms.positions.shape)
    return atoms


def to_ir_mul(a, irreps):
    """mul_ir rows -> the kernel's ir_mul layout (each segment: component-major, then channel)."""
    out, s = [], 0
    for mul, ir in irreps:
        d = ir.dim
        out.append(a[:, s:s + mul * d].reshape(-1, mul, d).transpose(0, 2, 1).reshape(-1, mul * d))
        s += mul * d
    return np.concatenate(out, 1)


def from_ir_mul(a, irreps):
    out, s = [], 0
    for mul, ir in irreps:
        d = ir.dim
        out.append(a[:, s:s + mul * d].reshape(-1, d, mul).transpose(0, 2, 1).reshape(-1, mul * d))
        s += mul * d
    return np.concatenate(out, 1)


class AddGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, store, key):
        ctx.store, ctx.key = store, key
        return t.clone()

    @staticmethod
    def backward(ctx, g):
        d = ctx.store.pop(ctx.key, None)
        return (g if d is None else g + torch.as_tensor(d, dtype=g.dtype).reshape(g.shape)), None, None


class Tap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, hook):
        ctx.hook = hook
        return t.clone()

    @staticmethod
    def backward(ctx, g):
        ctx.hook(g.detach())
        return g, None


def run(model, atoms, dtype, setup=None):
    """Forces and stress of ``model`` in ``dtype``; ``setup(ctx)`` -> (tp_hook, ...) installs the far-edge taps."""
    from mace import data
    from mace.tools import torch_geometric, utils
    torch.set_default_dtype(dtype)
    model = model.to(dtype)
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    config = data.config_from_atoms(atoms)
    loader = torch_geometric.dataloader.DataLoader(
        [data.AtomicData.from_config(config, z_table=z_table, cutoff=float(model.r_max), heads=["default"])],
        batch_size=1, shuffle=False)
    batch = next(iter(loader)).to_dict()
    ctx = dict(batch=batch, model=model)
    block = model.interactions[1]
    tp, radial = block.conv_tp, model.radial_embedding
    original_tp, original_radial = tp.forward, radial.forward
    store = {}

    def radial_forward(lengths, *args):
        ctx["r"] = lengths.detach().reshape(-1).double().numpy()
        ctx["radial_args"] = args
        if setup is not None:
            lengths = AddGrad.apply(lengths, store, "r")
        return original_radial(lengths, *args)

    def tp_forward(x, y, w, *rest):
        ctx["x"], ctx["y"], ctx["w"] = (t.detach().double().numpy() for t in (x, y, w))
        if setup is not None:
            x, y = AddGrad.apply(x, store, "x"), AddGrad.apply(y, store, "y")
        out = original_tp(x, y, w, *rest)
        if setup is not None:
            out = Tap.apply(out, lambda g: store.update(setup(ctx, g.double().numpy())))
        return out

    radial.forward, tp.forward = radial_forward, tp_forward
    try:
        out = model(batch, training=False, compute_force=True, compute_stress=True)
    finally:
        radial.forward, tp.forward = original_radial, original_tp
    return (out["forces"].detach().double().numpy(), out["stress"].detach().double().numpy().reshape(3, 3), ctx)


def weights_at(m, ctx, r, edges):
    """tp_weights [len(edges), Q] at lengths r for those edges (float64 copy ``m`` of the model's radial chain)."""
    node_attrs, edge_index, numbers = ctx["radial_args"]
    with torch.no_grad():
        feats, cutoff = m.radial_embedding(torch.as_tensor(r, dtype=torch.float64).reshape(-1, 1),
                                           node_attrs.double(), edge_index[:, edges], numbers)
        w = m.interactions[1].conv_tp_weights(feats)
        if cutoff is not None:
            w = w * cutoff
    return w.numpy()


def main():
    cif, repeat, out_path = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    code, paths = translate(SOURCE.read_text())
    scope = dict(horner=horner, np=np)
    exec(code, scope)
    loop = scope["loop"]
    model = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    atoms = structure(cif, repeat)
    model64 = copy.deepcopy(model).double()
    tp64 = model64.interactions[1].conv_tp
    f64, s64, _ = run(model, atoms, torch.float64)
    f32, s32, ctx32 = run(model, atoms, torch.float32)
    rf, rs = float(np.abs(f32 - f64).max()), float(np.abs(s32 - s64).max())
    tp = model.interactions[1].conv_tp
    irreps_x, irreps_out = tp.irreps_in1, tp.irreps_out.simplify()  # the kernel merges equal irreps (384x1o)
    r_max = float(model.r_max)
    n_band = int(np.ceil((r_max - BAND) / ((r_max - R_LO) / BASE_INTERVALS)))
    h = (r_max - BAND) / n_band
    avg = float(model.interactions[1].avg_num_neighbors)
    senders, receivers = (ctx32["batch"]["edge_index"][k].numpy() for k in (0, 1))
    print(dict(atoms=len(atoms), edges=int(senders.size), f32_force=rf, f32_stress=rs, paths=paths, h=h, avg=avg),
          flush=True)
    calib = {}

    def far_pairs(r, r_h):
        """Far edges in half2 pairs: consecutive far edges of each receiver's run (edge order), odd last stays."""
        far = np.nonzero(r > r_h)[0]
        keep = []
        for rc in np.unique(receivers[far]):
            e = far[receivers[far] == rc]
            keep.extend(e[: 2 * (len(e) // 2)])
        return np.sort(np.array(keep, dtype=np.int64))

    def inputs(ctx, g, edges, dscale):
        X = to_ir_mul(ctx["x"][edges], irreps_x)
        D = to_ir_mul(g[edges], irreps_out) * dscale
        SH = ctx["y"][edges]
        r = ctx["r"][edges]
        iv = np.floor((r - BAND) / h)
        x0 = BAND + iv * h
        T, SC = ((r - x0) / h)[:, None], np.full((len(edges), 1), 1.0 / h)
        d = 1e-6
        y0, y1 = weights_at(model64, ctx, x0, edges), weights_at(model64, ctx, x0 + h, edges)
        m0 = (weights_at(model64, ctx, x0 + d, edges) - weights_at(model64, ctx, x0 - d, edges)) / (2 * d) * h
        m1 = (weights_at(model64, ctx, x0 + h + d, edges) - weights_at(model64, ctx, x0 + h - d, edges)) / (2 * d) * h
        G = np.stack([y0, m0, 3 * (y1 - y0) - 2 * m0 - m1, 2 * (y0 - y1) + m0 + m1], 1)  # [E, 4, Q]
        return X, D, SH, G, T, SC

    results = []
    thresholds = [float(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else THRESHOLDS
    for r_h in thresholds:
        for parts, scaling in VARIANTS:
            dscale = 1.0
            stats = {}

            def setup(ctx, g, r_h=r_h, dscale=dscale, stats=stats, parts=parts, scaling=scaling):
                if parts == "none":
                    return {}
                edges = far_pairs(ctx["r"], r_h)
                if not len(edges):
                    return {}
                X, D, SH, G, T, SC = inputs(ctx, g, edges, dscale)
                # autograd's per-edge weight gradient of the model, to fit the per-path w factor
                if "factor" not in calib:
                    _, _, _, wg = emulate(loop, X, D / dscale, SH, G, T, SC, False, paths)
                    w = ctx["w"][edges]
                    wt = torch.tensor(w, requires_grad=True)
                    xt, yt = (torch.as_tensor(a) for a in (ctx["x"][edges], ctx["y"][edges]))
                    with torch.enable_grad():
                        o = tp64(xt, yt, wt)
                        dw, = torch.autograd.grad(o, wt, torch.as_tensor(g[edges]))
                    dw = dw.numpy()
                    factor = []
                    for p in range(paths):
                        a, b = dw[:, p * 128:(p + 1) * 128].ravel(), wg[p].ravel()
                        factor.append(float(np.dot(a, b) / np.dot(b, b)))
                    calib["factor"] = factor
                    calib["wg_fit_residual"] = [float(np.abs(dw[:, p * 128:(p + 1) * 128] - factor[p] * wg[p]).max()
                                                      / np.abs(dw[:, p * 128:(p + 1) * 128]).max())
                                                for p in range(paths)]
                    # dx / dsh with the kernel's w (= model w / factor) against autograd
                    Gk = G * np.repeat(np.array(factor), 128)[None, None, :]
                    dxk, dshk, _, _ = emulate(loop, X, D / dscale, SH, Gk, T, SC, False, paths)
                    xt2 = torch.tensor(ctx["x"][edges], requires_grad=True)
                    yt2 = torch.tensor(ctx["y"][edges], requires_grad=True)
                    ws = torch.as_tensor(np.einsum("etq,et->eq", G, T[:, [0, 0, 0, 0]] ** np.arange(4)))
                    with torch.enable_grad():
                        o = tp64(xt2, yt2, ws)
                        gx, gy = torch.autograd.grad(o, (xt2, yt2), torch.as_tensor(g[edges]))
                    gx = to_ir_mul(gx.numpy(), irreps_x)
                    calib["dx_check"] = float(np.abs(dxk - gx).max() / np.abs(gx).max())
                    calib["dsh_check"] = float(np.abs(dshk - gy.numpy()).max() / np.abs(gy.numpy()).max())
                    print("calibration", calib, flush=True)
                Gk = G * np.repeat(np.array(calib["factor"]), 128)[None, None, :]
                sw = np.ones(len(edges))
                if "w" in scaling:  # per edge (its slab): a power of two, max |coef| -> [2^top, 2^(top+1))
                    top = int(re.search(r"w(\d*)", scaling).group(1) or 10)
                    s_w = np.exp2(top - np.floor(np.log2(np.abs(Gk).reshape(len(edges), -1).max(1))))
                    Gk, sw = Gk * s_w[:, None, None], sw * s_w
                if "d" in scaling:  # per receiver row (the pair's shared dout), max -> [2^4, 2^5)
                    s_d = np.exp2(4 - np.floor(np.log2(np.abs(D).max(1))))
                    D, sw = D * s_d[:, None], sw * s_d
                lo = emulate(loop, X, D, SH, Gk, T, SC, False, paths)
                hi = emulate(loop, X, D, SH, Gk, T, SC, True, paths, rg32="r" in scaling)
                stats["overflow"] = int(sum(int((~np.isfinite(hi[k])).sum()) for k in range(3)))
                hi = tuple(np.nan_to_num(hi[k], nan=0.0, posinf=0.0, neginf=0.0) for k in range(3))
                ddx, ddsh, ddr = ((hi[k] - lo[k]) / sw.reshape((-1,) + (1,) * (hi[k].ndim - 1)) for k in range(3))
                if parts != "all":
                    ddx, ddsh, ddr = (d if parts == n else np.zeros_like(d) for d, n in zip((ddx, ddsh, ddr), ("dx", "dsh", "dr")))
                stats.update(half_edges=int(len(edges)), dx_rel=float(np.abs(ddx).max() / np.abs(lo[0]).max()),
                             dsh_rel=float(np.abs(ddsh).max() / np.abs(lo[1]).max()),
                             dr_rel=float(np.abs(ddr).max() / np.abs(lo[2]).max()),
                             g16_subnormal=float(((np.abs(Gk) < 6.1e-5) & (Gk != 0)).mean()),
                             dout16_subnormal=float(((np.abs(D) < 6.1e-5) & (D != 0)).mean()))
                E = len(ctx["r"])
                dx_full = np.zeros((E, 512)); dx_full[edges] = from_ir_mul(ddx, irreps_x)
                dy_full = np.zeros((E, 16)); dy_full[edges] = ddsh
                dr_full = np.zeros(E); dr_full[edges] = ddr
                return dict(x=dx_full, y=dy_full, r=dr_full)

            f, s, _ = run(model, atoms, torch.float32, setup)
            ef, es = float(np.abs(f - f64).max()), float(np.abs(s - s64).max())
            row = dict(r_h=r_h, parts=parts, scaling=scaling, force_over_f32=ef / rf, stress_over_f32=es / rs,
                       half_edge_fraction=stats.get("half_edges", 0) / senders.size, **stats)
            results.append(row)
            print(row, flush=True)
    out = dict(cif=cif, repeat=repeat, atoms=len(atoms), edges=int(senders.size), f32_force_rounding=rf,
               f32_stress_rounding=rs, spline_h=h, calibration=calib, stop_rule=2.0, rows=results)
    out_path.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
