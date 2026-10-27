"""Falsifier: can MH-1's radial networks be replaced, per species pair, by piecewise Chebyshev series in r
whose float32 evaluation is at least as accurate as the float32 network itself?

Truth: float64 evaluation of the network (MACE 0.3.16 RadialEmbeddingBlock + RadialMLP, weights from the checkpoint).
FP32 path: FastMACE's float32 arithmetic (projected embeddings added after the radial columns' GEMM).
Surrogate: Chebyshev interpolants fitted in float64 on n equal-width intervals of [r_lo, r_max], evaluated in float32.
Compared on the per-edge TP weights w = c(r) (W h(r) + b) (the folded last layer applied in float32 to either input),
and on dw/dr (forward-mode in the same precision), on a dense grid.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ckpt  # noqa: E402

P, M = ckpt.params, ckpt.modules


def silu(x):
    return x / (1 + np.exp(-x))


def dsilu(x):
    s = 1 / (1 + np.exp(-x))
    return s * (1 + x * (1 - s))


def layernorm(x, dx, g, b, eps):
    """Value and forward-mode derivative of LayerNorm over the last axis, in x's dtype."""
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(-1, keepdims=True)
    inv = 1 / np.sqrt(var + eps)
    xh = xc * inv
    y = xh * g + b
    if dx is None:
        return y, None
    dxc = dx - dx.mean(-1, keepdims=True)
    dxh = inv * (dxc - xh * (xh * dxc).mean(-1, keepdims=True))
    return y, dxh * g


class Radial:
    def __init__(self, model_path):
        m = ckpt.load(model_path)
        re = M(m)["radial_embedding"]
        b, d, c = M(re)["bessel_fn"], M(re)["distance_transform"], M(re)["cutoff_fn"]
        self.w = P(b)["bessel_weights"].astype(np.float64)
        self.pref = float(P(b)["prefactor"])
        self.q, self.p, self.a = (float(P(d)[k]) for k in ("q", "p", "a"))
        self.cov = P(d)["covalent_radii"].astype(np.float64)
        self.pc = float(P(c)["p"])
        self.rmax = float(P(c)["r_max"])
        self.z = list(P(m)["atomic_numbers"])
        self.blocks = []
        for name, inter in M(M(m)["interactions"]).items():
            se, te = M(inter)["source_embedding"], M(inter)["target_embedding"]
            n_el = len(self.z)
            pw_s = se.__dict__["instructions"][0]._args[3]
            pw_t = te.__dict__["instructions"][0]._args[3]
            emb_s = P(se)["weight"].reshape(n_el, -1).astype(np.float64) * pw_s
            emb_t = P(te)["weight"].reshape(n_el, -1).astype(np.float64) * pw_t
            nets = {}
            for net_name in ("conv_tp_weights", "density_fn"):
                seq = M(M(M(inter)[net_name])["net"])
                layers = []
                for k in sorted(seq, key=int):
                    mod = seq[k]
                    kind = mod._qualname.split(".")[-1]
                    pr = {a: v.astype(np.float64) for a, v in P(mod).items()}
                    if kind == "LayerNorm":
                        pr["eps"] = mod.__dict__["eps"]
                    layers.append((kind, pr))
                nets[net_name] = layers
            self.blocks.append({"name": name, "emb_s": emb_s, "emb_t": emb_t, "nets": nets})

    # --- geometry pieces (value and d/dr), in dtype of r ---
    def transform(self, r, r0):
        dt = r.dtype
        q, p, a = dt.type(self.q), dt.type(self.p), dt.type(self.a)
        x = r / dt.type(r0)
        xq, xqp = x ** q, x ** (q - p)
        den = 1 + xqp
        frac = a * xq / den
        t = 1 / (1 + frac)
        dx = 1 / dt.type(r0)
        dfrac = a * (q * x ** (q - 1) * den - xq * (q - p) * x ** (q - p - 1)) / (den * den) * dx
        dtdr = -dfrac * t * t
        return t, dtdr

    def bessel(self, t, dtdr):
        dt = t.dtype
        w = self.w.astype(dt)
        arg = t[:, None] * w[None, :]
        s, c = np.sin(arg), np.cos(arg)
        val = dt.type(self.pref) * s / t[:, None]
        dval = dt.type(self.pref) * (w[None, :] * c * t[:, None] - s) / (t[:, None] ** 2) * dtdr[:, None]
        return val, dval

    def cutoff(self, r):
        dt = r.dtype
        p = dt.type(self.pc)
        x = r / dt.type(self.rmax)
        env = (1 - (p + 1) * (p + 2) / 2 * x ** p + p * (p + 2) * x ** (p + 1) - p * (p + 1) / 2 * x ** (p + 2))
        denv = (-(p + 1) * (p + 2) / 2 * p * x ** (p - 1) + p * (p + 2) * (p + 1) * x ** p
                - p * (p + 1) / 2 * (p + 2) * x ** (p + 1)) / dt.type(self.rmax)
        return env * (x < 1), denv * (x < 1)

    def hidden(self, block, net, r, zs, zr, dtype):
        """[h c, c] (conv net) or edge density (density net) and d/dr, all in dtype."""
        r = r.astype(dtype)
        r0 = 0.5 * (self.cov[zs] + self.cov[zr])
        t, dtdr = self.transform(r, r0)
        phi, dphi = self.bessel(t, dtdr)
        layers = block["nets"][net]
        kind, first = layers[0]
        nb = phi.shape[1]
        W = first["weight"]
        s_idx, t_idx = self.z.index(zs), self.z.index(zr)
        es, et = block["emb_s"][s_idx].astype(dtype), block["emb_t"][t_idx].astype(dtype)
        ns = es.shape[0]
        # projected embeddings, as FastMACE's ProjectedEmbedding (bias joins the target side)
        proj_s = (W[:, nb:nb + ns].astype(dtype) @ es)
        proj_t = (W[:, nb + ns:].astype(dtype) @ et) + first["bias"].astype(dtype)
        x = phi @ W[:, :nb].T.astype(dtype)
        x = x + proj_s[None, :] + proj_t[None, :]
        dx = dphi @ W[:, :nb].T.astype(dtype)
        for kind, pr in layers[1:-1]:
            if kind == "LayerNorm":
                x, dx = layernorm(x, dx, pr["weight"].astype(dtype), pr["bias"].astype(dtype), dtype(pr["eps"]))
            elif kind == "SiLU":
                dx = dx * dsilu(x)
                x = silu(x)
            elif kind == "Linear":
                dx = dx @ pr["weight"].T.astype(dtype)
                x = x @ pr["weight"].T.astype(dtype) + pr["bias"].astype(dtype)
        c, dc = self.cutoff(r)
        if net == "conv_tp_weights":
            g = np.concatenate([x * c[:, None], c[:, None]], axis=1)
            dg = np.concatenate([dx * c[:, None] + x * dc[:, None], dc[:, None]], axis=1)
            return g, dg
        kind, last = layers[-1]
        y = x @ last["weight"].T.astype(dtype) + last["bias"].astype(dtype)
        dy = dx @ last["weight"].T.astype(dtype)
        th = np.tanh(y * y)
        dens = th * c[:, None]
        ddens = (1 - th * th) * 2 * y * dy * c[:, None] + th * dc[:, None]
        return dens, ddens

    def folded_last(self, block, dtype):
        kind, last = block["nets"]["conv_tp_weights"][-1]
        return np.concatenate([last["weight"], last["bias"][:, None]], axis=1).astype(dtype)  # [N, 65]


# --- piecewise Chebyshev ---
def cheb_nodes(n):
    k = np.arange(n)
    return np.cos(np.pi * (k + 0.5) / n)


def cheb_fit(f, a, b, deg):
    """Interpolate f (vector valued, columns) at deg+1 Chebyshev points of [a, b]; returns coefficients [deg+1, d]."""
    n = deg + 1
    x = cheb_nodes(n)
    r = 0.5 * (b - a) * x + 0.5 * (b + a)
    vals = f(r)  # [n, d]
    k = np.arange(n)
    T = np.cos(np.outer(k, np.arccos(x)))  # [n_coeff, n_nodes]
    coef = (2.0 / n) * (T @ vals)
    coef[0] *= 0.5
    return coef


def clenshaw(coef, x, dtype):
    """Evaluate sum_k coef[k] T_k(x) (coef [n, d], x [m]) in dtype; also d/dx via the derivative recurrence."""
    coef = coef.astype(dtype)
    x = x.astype(dtype)[:, None]
    b1 = np.zeros((x.shape[0], coef.shape[1]), dtype)
    b2 = np.zeros_like(b1)
    db1 = np.zeros_like(b1)
    db2 = np.zeros_like(b1)
    two = dtype(2)
    for k in range(coef.shape[0] - 1, 0, -1):
        b0 = two * x * b1 - b2 + coef[k]
        db0 = two * b1 + two * x * db1 - db2
        b2, b1 = b1, b0
        db2, db1 = db1, db0
    val = x * b1 - b2 + coef[0]
    dval = b1 + x * db1 - db2
    return val, dval


class Piecewise:
    def __init__(self, f, a, b, n_int, deg):
        self.edges = np.linspace(a, b, n_int + 1)
        self.a, self.b, self.n = a, b, n_int
        self.coefs = [cheb_fit(f, self.edges[i], self.edges[i + 1], deg) for i in range(n_int)]

    def __call__(self, r, dtype):
        r = r.astype(dtype)
        h = dtype((self.b - self.a) / self.n)
        idx = np.clip(((r - dtype(self.a)) / h).astype(np.int64), 0, self.n - 1)
        val = np.empty((r.shape[0], self.coefs[0].shape[1]), dtype)
        dval = np.empty_like(val)
        for i in range(self.n):
            sel = idx == i
            if not sel.any():
                continue
            lo, hi = dtype(self.edges[i]), dtype(self.edges[i + 1])
            x = (dtype(2) * r[sel] - lo - hi) / (hi - lo)
            v, dv = clenshaw(self.coefs[i], x, dtype)
            val[sel] = v
            dval[sel] = dv * (dtype(2) / (hi - lo))
        return val, dval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mace-mh-1.model")
    ap.add_argument("--pairs", default="11-17,17-11,11-11,17-17,41-17,17-41,41-41,8-41,41-8,8-11,11-8,8-8")
    ap.add_argument("--r-lo", type=float, default=1.0)
    ap.add_argument("--grid", type=int, default=60000)
    ap.add_argument("--configs", default="1x24,1x32,1x40,2x20,4x14,4x16,8x10,8x12,16x8,16x10")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    R = Radial(args.model)
    rng = np.random.default_rng(0)
    r = np.sort(np.concatenate([rng.uniform(args.r_lo, R.rmax, args.grid),
                                np.linspace(args.r_lo, R.rmax, 2001)]))
    r = r.astype(np.float32).astype(np.float64)  # both paths start from the same float32 distance
    r = r[r < R.rmax]
    configs = [tuple(int(v) for v in c.split("x")) for c in args.configs.split(",")]
    results = []
    for block in R.blocks:
        Wl32 = R.folded_last(block, np.float32)
        Wl64 = R.folded_last(block, np.float64)
        for pair in args.pairs.split(","):
            zs, zr = (int(v) for v in pair.split("-"))
            for net in ("conv_tp_weights", "density_fn"):
                f64 = lambda rr: R.hidden(block, net, rr, zs, zr, np.float64)[0]  # noqa: E731
                g64, dg64 = R.hidden(block, net, r, zs, zr, np.float64)
                g32, dg32 = R.hidden(block, net, r, zs, zr, np.float32)
                if net == "conv_tp_weights":  # compare on w = g @ W^T, the per-edge TP weights
                    def to_w(g, dg, W):
                        return g @ W.T, dg @ W.T
                    w64, dw64 = to_w(g64, dg64, Wl64)
                    w32, dw32 = to_w(g32, dg32, Wl32)
                else:
                    w64, dw64, w32, dw32 = g64, dg64, g32, dg32
                scale = np.abs(w64).max(0) + 1e-30
                dscale = np.abs(dw64).max(0) + 1e-30
                e32 = (np.abs(w32 - w64) / scale).max()
                de32 = (np.abs(dw32 - dw64) / dscale).max()
                rms32 = np.sqrt(((w32 - w64) / scale) ** 2).mean()
                drms32 = np.sqrt(((dw32 - dw64) / dscale) ** 2).mean()
                row = {"block": block["name"], "pair": pair, "net": net, "fp32_max": float(e32),
                       "fp32_dmax": float(de32), "fp32_rms": float(rms32), "fp32_drms": float(drms32), "fits": {}}
                for n_int, deg in configs:
                    pw = Piecewise(f64, args.r_lo, R.rmax, n_int, deg)
                    gc, dgc = pw(r, np.float32)
                    if net == "conv_tp_weights":
                        wc, dwc = gc @ Wl32.T, dgc @ Wl32.T
                    else:
                        wc, dwc = gc, dgc
                    ec = (np.abs(wc - w64) / scale).max()
                    dec = (np.abs(dwc - dw64) / dscale).max()
                    rmsc = np.sqrt(((wc - w64) / scale) ** 2).mean()
                    drmsc = np.sqrt(((dwc - dw64) / dscale) ** 2).mean()
                    row["fits"][f"{n_int}x{deg}"] = {"max": float(ec / e32), "dmax": float(dec / de32),
                                                     "rms": float(rmsc / rms32), "drms": float(drmsc / drms32)}
                results.append(row)
                best = {k: round(max(v["max"], v["dmax"]), 2) for k, v in row["fits"].items()}
                print(f"int{block['name']} {pair:>6} {net[:7]:>7}  fp32 max {e32:.2e} dmax {de32:.2e} | "
                      f"surrogate/fp32 (worst of value, derivative): {best}", flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"r_lo": args.r_lo, "results": results}, fh, indent=1)




# --- variant: fit h as a function of the transformed distance t (h depends on r only through t; c(r) is exact) ---
def hidden_of_t(R, block, net, t, zs, zr, dtype):
    """Network output before the cutoff, as a function of t, and dh/dt (forward mode), in dtype."""
    t = t.astype(dtype)
    phi, dphi = R.bessel(t, np.ones_like(t))
    layers = block["nets"][net]
    kind, first = layers[0]
    nb = phi.shape[1]
    W = first["weight"]
    es = block["emb_s"][R.z.index(zs)].astype(dtype)
    et = block["emb_t"][R.z.index(zr)].astype(dtype)
    ns = es.shape[0]
    proj_s = W[:, nb:nb + ns].astype(dtype) @ es
    proj_t = W[:, nb + ns:].astype(dtype) @ et + first["bias"].astype(dtype)
    x = phi @ W[:, :nb].T.astype(dtype) + proj_s[None, :] + proj_t[None, :]
    dx = dphi @ W[:, :nb].T.astype(dtype)
    for kind, pr in layers[1:-1]:
        if kind == "LayerNorm":
            x, dx = layernorm(x, dx, pr["weight"].astype(dtype), pr["bias"].astype(dtype), dtype(pr["eps"]))
        elif kind == "SiLU":
            dx = dx * dsilu(x)
            x = silu(x)
        elif kind == "Linear":
            dx = dx @ pr["weight"].T.astype(dtype)
            x = x @ pr["weight"].T.astype(dtype) + pr["bias"].astype(dtype)
    return x, dx


def surrogate_g_t(R, pw, r, zs, zr, dtype):
    """[h c, c] and d/dr from a piecewise Chebyshev fit of h(t), with t(r) and c(r) evaluated in dtype."""
    r = r.astype(dtype)
    r0 = 0.5 * (R.cov[zs] + R.cov[zr])
    t, dtdr = R.transform(r, r0)
    h, dhdt = pw(t, dtype)
    dh = dhdt * dtdr[:, None]
    c, dc = R.cutoff(r)
    g = np.concatenate([h * c[:, None], c[:, None]], axis=1)
    dg = np.concatenate([dh * c[:, None] + h * dc[:, None], dc[:, None]], axis=1)
    return g, dg


def main_t():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mace-mh-1.model")
    ap.add_argument("--pairs", default="11-17,17-11,11-11,17-17,41-17,17-41,41-41,8-41,41-8,8-11,11-8,8-8")
    ap.add_argument("--r-lo", type=float, default=1.0)
    ap.add_argument("--grid", type=int, default=60000)
    ap.add_argument("--configs", default="1x16,1x24,1x32,2x16,4x12,4x16,8x10,8x12")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(sys.argv[2:])
    R = Radial(args.model)
    rng = np.random.default_rng(0)
    r = np.sort(np.concatenate([rng.uniform(args.r_lo, R.rmax, args.grid), np.linspace(args.r_lo, R.rmax, 2001)]))
    r = r.astype(np.float32).astype(np.float64)
    r = r[r < R.rmax]
    configs = [tuple(int(v) for v in c.split("x")) for c in args.configs.split(",")]
    results = []
    for block in R.blocks:
        Wl32, Wl64 = R.folded_last(block, np.float32), R.folded_last(block, np.float64)
        for pair in args.pairs.split(","):
            zs, zr = (int(v) for v in pair.split("-"))
            net = "conv_tp_weights"
            r0 = 0.5 * (R.cov[zs] + R.cov[zr])
            t_hi = float(R.transform(np.array([args.r_lo]), r0)[0][0])
            t_lo = float(R.transform(np.array([R.rmax]), r0)[0][0])
            g64, dg64 = R.hidden(block, net, r, zs, zr, np.float64)
            g32, dg32 = R.hidden(block, net, r, zs, zr, np.float32)
            w64, dw64 = g64 @ Wl64.T, dg64 @ Wl64.T
            w32, dw32 = g32 @ Wl32.T, dg32 @ Wl32.T
            scale, dscale = np.abs(w64).max(0) + 1e-30, np.abs(dw64).max(0) + 1e-30
            e32 = (np.abs(w32 - w64) / scale).max()
            de32 = (np.abs(dw32 - dw64) / dscale).max()
            row = {"block": block["name"], "pair": pair, "t_range": [t_lo, t_hi], "fp32_max": float(e32),
                   "fp32_dmax": float(de32), "fits": {}}
            f64 = lambda tt: hidden_of_t(R, block, net, tt, zs, zr, np.float64)[0]  # noqa: E731
            for n_int, deg in configs:
                pw = Piecewise(f64, t_lo, t_hi, n_int, deg)
                gc, dgc = surrogate_g_t(R, pw, r, zs, zr, np.float32)
                wc, dwc = gc @ Wl32.T, dgc @ Wl32.T
                ec = (np.abs(wc - w64) / scale).max()
                dec = (np.abs(dwc - dw64) / dscale).max()
                row["fits"][f"{n_int}x{deg}"] = {"max": float(ec / e32), "dmax": float(dec / de32)}
            results.append(row)
            best = {k: round(max(v["max"], v["dmax"]), 3) for k, v in row["fits"].items()}
            print(f"int{block['name']} {pair:>6} t in [{t_lo:.3f},{t_hi:.3f}]  fp32 w-err {e32:.1e} dw-err {de32:.1e} | "
                  f"surrogate/fp32 worst(value,deriv): {best}", flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"r_lo": args.r_lo, "variant": "t", "results": results}, fh, indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "t":
        main_t()
    else:
        main()
