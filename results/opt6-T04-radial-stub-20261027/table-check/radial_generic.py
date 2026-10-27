"""The same falsifier for generic MACE (MP-0 medium): radial net = e3nn FullyConnectedNet on Bessel(r) * cutoff(r),
no species input, so one table per interaction. Table output: the last hidden layer h(r) (64), compared at w = h W3."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ckpt  # noqa: E402
import radial_adaptive as ra  # noqa: E402
import radial_cheb as rc  # noqa: E402

P, M = ckpt.params, ckpt.modules


class Generic:
    def __init__(self, path):
        m = ckpt.load(path)
        re = M(m)["radial_embedding"]
        b, c = M(re)["bessel_fn"], M(re)["cutoff_fn"]
        self.w = P(b)["bessel_weights"].astype(np.float64)
        self.pref = float(P(b)["prefactor"])
        self.pc, self.rmax = float(P(c)["p"]), float(P(c)["r_max"])
        self.blocks = []
        for name, inter in M(M(m)["interactions"]).items():
            fc = M(inter)["conv_tp_weights"]
            layers = []
            for k in sorted(M(fc)):
                lay = M(fc)[k]
                act = M(lay).get("act")
                layers.append((P(lay)["weight"].astype(np.float64), lay.__dict__["h_in"],
                               None if act is None else act.__dict__["cst"]))
            self.blocks.append({"name": name, "layers": layers})

    def hidden(self, block, r, dtype):
        r = r.astype(dtype)
        arg = r[:, None] * self.w.astype(dtype)[None, :]
        s, co = np.sin(arg), np.cos(arg)
        pref = dtype(self.pref)
        bes = pref * s / r[:, None]
        dbes = pref * (self.w.astype(dtype)[None, :] * co * r[:, None] - s) / (r[:, None] ** 2)
        p = dtype(self.pc)
        x = r / dtype(self.rmax)
        env = (1 - (p + 1) * (p + 2) / 2 * x ** p + p * (p + 2) * x ** (p + 1) - p * (p + 1) / 2 * x ** (p + 2)) * (x < 1)
        denv = ((-(p + 1) * (p + 2) / 2 * p * x ** (p - 1) + p * (p + 2) * (p + 1) * x ** p
                 - p * (p + 1) / 2 * (p + 2) * x ** (p + 1)) / dtype(self.rmax)) * (x < 1)
        h = bes * env[:, None]
        dh = dbes * env[:, None] + bes * denv[:, None]
        for W, h_in, cst in block["layers"][:-1]:
            Wd = (W / np.sqrt(h_in)).astype(dtype)
            h, dh = h @ Wd, dh @ Wd
            dh = dh * rc.dsilu(h) * dtype(cst)
            h = rc.silu(h) * dtype(cst)
        return h, dh

    def last(self, block, dtype):
        W, h_in, _ = block["layers"][-1]
        return (W / np.sqrt(h_in)).astype(dtype)  # [64, 512]


def main():
    G = Generic(sys.argv[1] if len(sys.argv) > 1 else "2023-12-03-mace-128-L1_epoch-199.model")
    lo = 0.6
    rng = np.random.default_rng(3)
    r = np.sort(np.concatenate([rng.uniform(lo, G.rmax, 30000), np.linspace(lo, G.rmax, 3001)]))
    r = r.astype(np.float32).astype(np.float64)
    r = r[r < G.rmax]
    for block in G.blocks:
        def f(rr, with_d=False, block=block):
            h, dh = G.hidden(block, rr, np.float64)
            return (h, dh) if with_d else h
        h64, dh64 = f(r, True)
        table = ra.AdaptiveTable(f, lo, G.rmax, 12, 2e-7, 2e-7, np.abs(h64).max(0) + 1e-30, np.abs(dh64).max(0) + 1e-30)
        W64, W32 = G.last(block, np.float64), G.last(block, np.float32)
        w64, dw64 = h64 @ W64, dh64 @ W64
        sc, dsc = np.abs(w64).max(0) + 1e-30, np.abs(dw64).max(0) + 1e-30
        h32, dh32 = G.hidden(block, r, np.float32)
        hc, dhc = table(r, np.float32)
        e = {}
        for name, (a, b) in {"fp32": (h32 @ W32, dh32 @ W32), "table": (hc @ W32, dhc @ W32)}.items():
            e[name] = ((np.abs(a - w64) / sc).max(), (np.abs(b - dw64) / dsc).max())
        print(f"MP-0 int{block['name']}: {len(table.pieces)} pieces x deg 12 = "
              f"{len(table.pieces) * 13 * 64 * 4 / 1024:.0f} KiB (one table for all species) | fp32 net max err value "
              f"{e['fp32'][0]:.1e} d/dr {e['fp32'][1]:.1e}; table/fp32: value {e['table'][0] / e['fp32'][0]:.2f}, "
              f"d/dr {e['table'][1] / e['fp32'][1]:.2f}")


if __name__ == "__main__":
    main()
