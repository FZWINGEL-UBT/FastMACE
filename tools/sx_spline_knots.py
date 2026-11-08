"""CPU only: cubic Hermite splines of interaction 2's w on piecewise-uniform knots, scored like the certificate.

sx-layout lab, follow-up (7 November 2026, results/sx-layout-lab-20261107). A uniform 256-knot spline is fast but
fails the radial certificate (slope 74x at r >= 1 A); 2,048 uniform knots pass at r >= 1 A but cost L2 traffic.
Here the knots are dense where w curves most (short r) and coarse elsewhere: a grid is a few segments, each uniform,
so the kernel lookup stays a compare and one multiply (segment s: iv = off_s + floor((r - r_s) * inv_h_s)).

``--diagnose`` prints, for a uniform grid, each r band's largest value and slope error over the float32 network's
own (whole-domain and r >= 1 A) error, which says how fine each band must be. Without it, scores the named grids
(``--grids "0.3:1.0:256,1.0:6.0:256"``: start:end:intervals per segment) as ``sx_spline_error.py`` scores a table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastmace.rewrites import radial as rt  # noqa: E402

SPECIES = (38, 22, 8)  # Sr, Ti, O


def parse_grid(text):
    """``[(start, end, intervals), ...]`` from "a:b:n,b:c:m"; contiguous."""
    grid = []
    for part in text.split(","):
        a, b, n = part.split(":")
        grid.append((float(a), float(b), int(n)))
    for (_, b, _), (a, _, _) in zip(grid, grid[1:]):
        assert abs(a - b) < 1e-12, "segments must be contiguous"
    return grid


def grid_knots(grid):
    """float64 knot positions of all segments (shared end knots once)."""
    xs = [torch.linspace(a, b, n + 1, dtype=torch.float64) for a, b, n in grid]
    return torch.cat([xs[0]] + [x[1:] for x in xs[1:]])


def grid_coefficients(chain64, pairs, grid):
    """``[P, intervals, 4, Q]`` float64 power-basis coefficients in each interval's own t; the per-interval h."""
    x = grid_knots(grid)
    h = x[1:] - x[:-1]
    folded = chain64.folded
    out = []
    for s, t in pairs:
        with torch.no_grad():
            g, dg = chain64.with_derivative(x, s, t)
        y = g[:, :-1].double() @ folded.T
        dy = dg[:, :-1].double() @ folded.T
        y0, y1 = y[:-1], y[1:]
        m0, m1 = dy[:-1] * h[:, None], dy[1:] * h[:, None]
        c2 = 3 * (y1 - y0) - 2 * m0 - m1
        c3 = 2 * (y0 - y1) + m0 + m1
        out.append(torch.stack([y0, m0, c2, c3], dim=1))
    return torch.stack(out), h


def grid_eval(coef32, grid, r32, pair):
    """float32 value and r-slope as the lab kernel computes them: segment by compare, then one multiply."""
    iv = torch.zeros(r32.shape, dtype=torch.long)
    t = torch.zeros(r32.shape, dtype=torch.float32)
    inv = torch.zeros(r32.shape, dtype=torch.float32)
    offset = 0
    for k, (a, b, n) in enumerate(grid):
        inv_h = torch.tensor(n / (b - a), dtype=torch.float32)
        mask = r32 >= torch.tensor(a, dtype=torch.float32) if k else torch.ones_like(r32, dtype=torch.bool)
        u = torch.clamp((r32 - torch.tensor(a, dtype=torch.float32)) * inv_h, min=0)
        i = torch.clamp(torch.floor(u).long(), 0, n - 1)
        iv = torch.where(mask, offset + i, iv)
        t = torch.where(mask, u - i.float(), t)
        inv = torch.where(mask, inv_h, inv)
        offset += n
    t = t[:, None]
    c = coef32[pair, iv]
    value = c[:, 0] + t * (c[:, 1] + t * (c[:, 2] + t * c[:, 3]))
    slope = (c[:, 1] + t * (2 * c[:, 2] + 3 * t * c[:, 3])) * inv[:, None]
    return value, slope


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("output", type=Path)
    p.add_argument("--diagnose", type=int, nargs="*", default=[], help="uniform interval counts to map against r")
    p.add_argument("--grids", nargs="*", default=[])
    p.add_argument("--bands", type=float, nargs="*",
                   default=[0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])
    args = p.parse_args(argv)
    torch.set_num_threads(8)
    started = time.time()
    model = torch.load(args.checkpoint, map_location="cpu", weights_only=False).float()
    numbers = [int(z) for z in model.atomic_numbers]
    columns = [numbers.index(z) for z in SPECIES]
    pairs = [(s, t) for s in range(len(columns)) for t in range(len(columns))]
    r_max, r_lo = float(model.r_max), rt.R_LO
    chain64 = rt._GenericChain(model, model.interactions[1], columns, torch.float64, torch.device("cpu"))
    chain32 = rt._GenericChain(model, model.interactions[1], columns, torch.float32, torch.device("cpu"))
    folded = chain64.folded

    gen = torch.Generator().manual_seed(0)
    n = rt.CERTIFY_POINTS
    r64 = torch.cat([torch.linspace(r_lo, r_max, n + 1, dtype=torch.float64)[:-1],
                     r_lo + (r_max - r_lo) * torch.rand(n, generator=gen, dtype=torch.float64)])
    r64 = r64.to(torch.float32).to(torch.float64)
    r32 = r64.to(torch.float32)
    physical = r64 >= rt.R_PHYSICAL

    def to_w(g):
        return g.detach().to(torch.float64)[:, :-1] @ folded.T

    truth, net = {}, {}
    with torch.no_grad():
        for k, (s, t) in enumerate(pairs):
            truth[k] = [to_w(v) for v in chain64.with_derivative(r64, s, t)]
            net[k] = [to_w(v) for v in chain32.with_derivative(r32, s, t)]

    regions = {"all": torch.ones_like(physical), "physical": physical}
    scale = {(k, j, rn): truth[k][j][m].abs().amax(dim=0).clamp_min(1e-300)
             for k in truth for j in range(2) for rn, m in regions.items()}
    net_err = {(k, j, rn): ((net[k][j][m] - truth[k][j][m]).abs() / scale[k, j, rn]).max().item()
               for k in truth for j in range(2) for rn, m in regions.items()}

    def errors(evaluate):
        """per (pair, kind, region): relative error per point, over the network's region error."""
        out = {}
        for k in range(len(pairs)):
            got = evaluate(k)
            for j in range(2):
                for rn in regions:
                    e = ((got[j].to(torch.float64) - truth[k][j]).abs() / scale[k, j, rn]).amax(dim=1)
                    out[k, j, rn] = e / net_err[k, j, rn]
        return out

    def score(rel):
        parts = {}
        for (k, j, rn), v in rel.items():
            key = f"{('value', 'slope')[j]}/{rn}"
            parts[key] = max(parts.get(key, 0.0), v[regions[rn]].max().item())
        return dict(worst=max(parts.values()), parts=parts)

    report = dict(checkpoint=args.checkpoint, r_lo=r_lo, r_max=r_max, physical_from=rt.R_PHYSICAL,
                  diagnose={}, grids={})
    edges = args.bands
    for count in args.diagnose:
        grid = [(r_lo, r_max, count)]
        coef, _ = grid_coefficients(chain64, pairs, grid)
        rel = errors(lambda k: grid_eval(coef.to(torch.float32), grid, r32, torch.full(r32.shape, k)))
        rows = []
        for a, b in zip(edges, edges[1:]):
            band = (r64 >= a) & (r64 < b)
            row = dict(band=[a, b])
            for j, kind in enumerate(("value", "slope")):
                for rn in regions:
                    row[f"{kind}/{rn}"] = max(rel[k, j, rn][band & regions[rn]].max().item()
                                              if (band & regions[rn]).any() else 0.0 for k in range(len(pairs)))
            rows.append(row)
            print(f"uniform {count} [{a:.2f},{b:.2f})", {k: round(v, 3) for k, v in row.items() if k != "band"},
                  flush=True)
        report["diagnose"][str(count)] = dict(h=(r_max - r_lo) / count, bands=rows)

    for text in args.grids:
        grid = parse_grid(text)
        assert abs(grid[0][0] - r_lo) < 1e-12 and abs(grid[-1][1] - r_max) < 1e-12, (grid, r_lo, r_max)
        coef, h = grid_coefficients(chain64, pairs, grid)
        coef32 = coef.to(torch.float32)
        result = score(errors(lambda k: grid_eval(coef32, grid, r32, torch.full(r32.shape, k))))
        intervals = sum(g[2] for g in grid)
        result.update(grid=grid, intervals=intervals, MB_6pairs=6 * intervals * 4 * folded.shape[0] * 4 / 1e6)
        report["grids"][text] = result
        print(text, intervals, round(result["MB_6pairs"], 1), "MB",
              {k: round(v, 3) for k, v in result["parts"].items()}, flush=True)

    report["seconds"] = time.time() - started
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print("done", round(report["seconds"], 1), "s")


if __name__ == "__main__":
    main()
