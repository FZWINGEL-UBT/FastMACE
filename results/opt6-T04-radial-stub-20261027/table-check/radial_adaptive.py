"""Adaptive piecewise Chebyshev tables for MH-1's conv radial networks: bisect intervals until the float64
truncation error of value and r-derivative is below tol (relative to each component's max over the domain),
then compare the float32-evaluated table with the float32 network (both against float64 truth) at the w level.
Reports table sizes. Usage: python -I radial_adaptive.py [--r-lo 0.6] [--deg 12] [--tol 2e-7]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import radial_cheb as rc  # noqa: E402


class AdaptiveTable:
    def __init__(self, f, a, b, deg, tol_v, tol_d, scale, dscale, n0=16, max_depth=12):
        self.deg, self.pieces = deg, []
        stack = [(e0, e1, 0) for e0, e1 in zip(np.linspace(a, b, n0 + 1)[:-1], np.linspace(a, b, n0 + 1)[1:])]
        while stack:
            lo, hi, depth = stack.pop()
            coef = rc.cheb_fit(f, lo, hi, deg)
            rr = np.linspace(lo, hi, 64)
            gv, gd = f(rr, True)
            x = (2 * rr - lo - hi) / (hi - lo)
            v, d = rc.clenshaw(coef, x, np.float64)
            d = d * 2 / (hi - lo)
            ok = (np.abs(v - gv) / scale).max() <= tol_v and (np.abs(d - gd) / dscale).max() <= tol_d
            if ok or depth >= max_depth:
                self.pieces.append((lo, hi, coef))
            else:
                mid = 0.5 * (lo + hi)
                stack += [(lo, mid, depth + 1), (mid, hi, depth + 1)]
        self.pieces.sort(key=lambda p: p[0])
        self.starts = np.array([p[0] for p in self.pieces])

    def __call__(self, r, dtype):
        r = r.astype(dtype)
        idx = np.clip(np.searchsorted(self.starts.astype(dtype), r, side="right") - 1, 0, len(self.pieces) - 1)
        out_v = np.empty((r.shape[0], self.pieces[0][2].shape[1]), dtype)
        out_d = np.empty_like(out_v)
        for i, (lo, hi, coef) in enumerate(self.pieces):
            sel = idx == i
            if not sel.any():
                continue
            lo_, hi_ = dtype(lo), dtype(hi)
            x = (dtype(2) * r[sel] - lo_ - hi_) / (hi_ - lo_)
            v, d = rc.clenshaw(coef, x, dtype)
            out_v[sel], out_d[sel] = v, d * (dtype(2) / (hi_ - lo_))
        return out_v, out_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mace-mh-1.model")
    ap.add_argument("--pairs", default="11-17,17-11,11-11,17-17,41-17,17-41,41-41,8-41,41-8,8-11,11-8,8-8")
    ap.add_argument("--r-lo", type=float, default=0.6)
    ap.add_argument("--deg", type=int, default=12)
    ap.add_argument("--tol", type=float, default=2e-7)
    ap.add_argument("--grid", type=int, default=30000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    R = rc.Radial(args.model)
    rng = np.random.default_rng(1)
    r = np.sort(np.concatenate([rng.uniform(args.r_lo, R.rmax, args.grid), np.linspace(args.r_lo, R.rmax, 3001)]))
    r = r.astype(np.float32).astype(np.float64)
    r = r[r < R.rmax]
    rows = []
    for block in R.blocks:
        Wl32, Wl64 = R.folded_last(block, np.float32), R.folded_last(block, np.float64)
        for pair in args.pairs.split(","):
            zs, zr = (int(v) for v in pair.split("-"))
            net = "conv_tp_weights"

            def f(rr, with_d=False, block=block, zs=zs, zr=zr):
                g, dg = R.hidden(block, net, rr, zs, zr, np.float64)
                return (g, dg) if with_d else g

            g64, dg64 = f(r, True)
            gscale, gdscale = np.abs(g64).max(0) + 1e-30, np.abs(dg64).max(0) + 1e-30
            table = AdaptiveTable(f, args.r_lo, R.rmax, args.deg, args.tol, args.tol, gscale, gdscale)
            g32, dg32 = R.hidden(block, net, r, zs, zr, np.float32)
            gc, dgc = table(r, np.float32)
            w64, dw64 = g64 @ Wl64.T, dg64 @ Wl64.T
            sc, dsc = np.abs(w64).max(0) + 1e-30, np.abs(dw64).max(0) + 1e-30
            res = {}
            for name, (wv, wd) in {"fp32_net": (g32 @ Wl32.T, dg32 @ Wl32.T),
                                   "table": (gc @ Wl32.T, dgc @ Wl32.T)}.items():
                ev, ed = np.abs(wv - w64) / sc, np.abs(wd - dw64) / dsc
                res[name] = {"max": float(ev.max()), "dmax": float(ed.max()),
                             "rms": float(np.sqrt((ev ** 2).mean())), "drms": float(np.sqrt((ed ** 2).mean()))}
            n_pieces = len(table.pieces)
            kb = n_pieces * (args.deg + 1) * g64.shape[1] * 4 / 1024
            ratio = max(res["table"]["max"] / res["fp32_net"]["max"], res["table"]["dmax"] / res["fp32_net"]["dmax"])
            rms_ratio = max(res["table"]["rms"] / res["fp32_net"]["rms"], res["table"]["drms"] / res["fp32_net"]["drms"])
            rows.append({"block": block["name"], "pair": pair, "pieces": n_pieces, "kib": kb, "errors": res,
                         "worst_max_ratio": ratio, "worst_rms_ratio": rms_ratio})
            print(f"int{block['name']} {pair:>6}: {n_pieces:3d} pieces x deg {args.deg} = {kb:6.0f} KiB | "
                  f"table/fp32-net error, worst of value & d/dr: max {ratio:.2f}, rms {rms_ratio:.3f}", flush=True)
    if args.out:
        json.dump({"r_lo": args.r_lo, "deg": args.deg, "tol": args.tol, "rows": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
