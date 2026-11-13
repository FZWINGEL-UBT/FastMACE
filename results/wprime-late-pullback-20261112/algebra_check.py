"""Step 1 of design 1 (four-component backward, delayed geometry differentiation): CPU algebra check.

All ten paths x 128 channels, nonconstant cubic radial weights w_pc(r) = a + b r + c r^2 + d r^3. Reference: the
direct CG contraction E = sum_e,p,c w_pc(r) sum_ijo C_p[i,j,o] x_i Y_lj(v/r) g_o with autograd in float64 (dE/dv in
the sh frame, dE/dx per input block). Design 1:
  q_p = M_p g_p (N1's conversion), per path phi and grad phi of an extension of phi off the unit sphere
    a = 0:              phi = x0 h,               grad = x0 grad h,                         dx0 = w h
    a = 1, L = l + 1:   phi = x . grad h,         grad = H x,                               dx  = w grad h
    a = 1, L = l - 1:   phi = A h - B / (2L + 1)  (A = x.n, B = x.grad h; unit-sphere extension, no Euler term)
                        grad = x h + A grad h - H x / (2L + 1),                             dx  = w (n h - grad h / (2L + 1))
  G = sum w grad phi, R = sum w' phi (over paths and channels), then once per edge dE/dv = n R + (G - n (n.G)) / r.
Run in float64 and in float32 (design 1's arithmetic in float32 against the float64 reference); the float32 direct
contraction (autograd) is the rounding baseline.
"""
import json
import sys

import torch

from common import HERE, setup


def h_terms(L, q, n):
    """h, grad h, and a function x -> H x (Hessian of h times x) in N1's Cartesian form."""
    from lab_cartesian import _sym2, _sym3n
    if L == 0:
        return q[..., 0], torch.zeros_like(n), lambda x: torch.zeros_like(x)
    if L == 1:
        return (q * n).sum(-1), q, lambda x: torch.zeros_like(x)
    if L == 2:
        Q = _sym2(q)
        v = torch.einsum("...ij,...j->...i", Q, n)
        return (n * v).sum(-1), 2 * v, lambda x: 2 * torch.einsum("...ij,...j->...i", Q, x)
    Q = _sym3n(q, n)
    v = torch.einsum("...ij,...j->...i", Q, n)
    return (n * v).sum(-1), 3 * v, lambda x: 6 * torch.einsum("...ij,...j->...i", Q, x)


def run(s, dtype, E=64, C=128, seed=20261112):
    from lab_cartesian import harmonics
    gen = torch.Generator().manual_seed(seed)
    f64 = torch.float64
    v = torch.randn(E, 3, generator=gen, dtype=f64)
    v = v / v.norm(dim=-1, keepdim=True) * (0.8 + 4.5 * torch.rand(E, 1, generator=gen, dtype=f64))
    blocks = sorted({(p["input_start"], 2 * p["a"] + 1) for p in s.paths})
    x = {st: torch.randn(E, C, d, generator=gen, dtype=f64) for st, d in blocks}
    g = [torch.randn(E, C, 2 * p["L"] + 1, generator=gen, dtype=f64) for p in s.paths]
    rc = [torch.randn(C, 4, generator=gen, dtype=f64) * torch.tensor([1.0, 0.5, 0.2, 0.05], dtype=f64)
          for _ in s.paths]

    def radial(coef, r):  # [C, 4], [E] -> w, w' [E, C]
        rp = torch.stack([torch.ones_like(r), r, r * r, r ** 3], -1)
        dp = torch.stack([torch.zeros_like(r), torch.ones_like(r), 2 * r, 3 * r * r], -1)
        return rp @ coef.T, dp @ coef.T

    def reference(dt):  # the direct contraction, autograd
        vv = v.to(dt).clone().requires_grad_(True)
        xs = {k: t.to(dt).clone().requires_grad_(True) for k, t in x.items()}
        r = vv.norm(dim=-1)
        energy = 0
        for p, gp, cp in zip(s.paths, g, rc):
            y = harmonics(p["l"], vv, s.frame.to(dt), normalize=True)  # [E, 2l+1]
            w, _ = radial(cp.to(dt), r)
            coef = torch.as_tensor(p["coef"], dtype=dt)
            energy = energy + (w * torch.einsum("ijo,eci,ej,eco->ec", coef, xs[p["input_start"]], y,
                                                gp.to(dt))).sum()
        dv, *dx = torch.autograd.grad(energy, [vv] + [xs[k] for k, _ in blocks])
        return dv.to(f64), {k: d.to(f64) for (k, _), d in zip(blocks, dx)}

    def design1(dt):
        vv, r = v.to(dt), v.norm(dim=-1).to(dt)
        n = vv / r[:, None]
        nn = n[:, None, :].expand(E, C, 3)
        G = torch.zeros(E, C, 3, dtype=dt)
        R = torch.zeros(E, C, dtype=dt)
        dx = {k: torch.zeros(E, C, d, dtype=dt) for k, d in blocks}
        for p, gp, cp in zip(s.paths, g, rc):
            a, l, L = p["a"], p["l"], p["L"]
            q = torch.einsum("ko,eco->eck", p["M"].to(dt), gp.to(dt))
            w, wp = radial(cp.to(dt), r)
            h, gh, hess = h_terms(L, q, nn)
            xv = x[p["input_start"]].to(dt)
            if a == 0:
                phi, grad, dxp = xv[..., 0] * h, xv[..., :1] * gh, h[..., None]
            elif L == l + 1:
                phi, grad, dxp = (xv * gh).sum(-1), hess(xv), gh
            else:
                k = 1.0 / (2 * L + 1)
                A, B = (xv * nn).sum(-1), (xv * gh).sum(-1)
                phi = A * h - k * B
                grad = xv * h[..., None] + A[..., None] * gh - k * hess(xv)
                dxp = nn * h[..., None] - k * gh
            G += w[..., None] * grad
            R += wp * phi
            dx[p["input_start"]] += w[..., None] * dxp
        G, R = G.sum(1), R.sum(1)  # the channel reduction
        dv = n * R[:, None] + (G - n * (n * G).sum(-1, keepdim=True)) / r[:, None]
        return dv.to(f64), {k: d.to(f64) for k, d in dx.items()}

    ref_v, ref_x = reference(f64)

    def err(dv, dx):
        ev = float((dv - ref_v).abs().max() / ref_v.abs().max())
        emed = float(((dv - ref_v).norm(dim=-1) / ref_v.norm(dim=-1)).median())
        ex = {str(k): float((dx[k] - ref_x[k]).abs().max() / ref_x[k].abs().max()) for k in ref_x}
        return dict(dv_max_rel=ev, dv_median_edge_rel=emed, dx_max_rel=ex)

    out = dict(dtype=str(dtype), edges=E, channels=C, fit_residual=[p["fit_residual"] for p in s.paths],
               paths=[(p["a"], p["l"], p["L"]) for p in s.paths], design1=err(*design1(dtype)))
    if dtype == torch.float32:
        out["direct_f32_baseline"] = err(*reference(torch.float32))
    return out


if __name__ == "__main__":
    s = setup()
    res = [run(s, torch.float64), run(s, torch.float32)]
    print(json.dumps(res, indent=1))
    (HERE / "algebra_check.json").write_text(json.dumps(res, indent=1) + "\n")
    ok = res[0]["design1"]["dv_max_rel"] < 1e-12 and max(res[0]["design1"]["dx_max_rel"].values()) < 1e-12
    sys.exit(0 if ok else 1)
