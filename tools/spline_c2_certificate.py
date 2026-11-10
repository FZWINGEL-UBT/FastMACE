"""CPU only: does a C2 cubic spline (a cubic B-spline) of interaction 2's radial weights pass SPLINE_W's certificate?
(9 November 2026)

Production fits a cubic Hermite spline per ordered species pair: exact values and exact r-slopes at the knots, so it is
C1 (``fastmace.rewrites.radial.spline``). A cubic B-spline table, one control point per knot
(results/bspline-reuse-bound-20261109), holds a C2 cubic spline on the same knots. On piecewise-uniform knots that is
the C2 interpolant of the exact knot values with the exact end slopes (clamped): its interior slopes come from the
continuity of the second derivative, not from the network.

This script scores that C2 spline with production's certificate (``spline._Scorer``: per pair and column, the error
over the column's largest exact value, divided by the float32 network's own error, values and r-slopes, whole domain
and r >= R_PHYSICAL; passes at <= 1, production's per-band target 0.5). It runs production's band refinement with the
C2 coefficients from the base grid, and also scores the C2 spline on production's own grid, so the knot count the
B-spline needs can be compared with the reuse model's assumption (production's 736 intervals).

The C2 coefficients are evaluated in float32 power basis as production's are; a kernel would evaluate the B-spline
basis instead (rounding only).

    python tools/spline_c2_certificate.py CHECKPOINT OUTPUT.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastmace.rewrites import radial as rt  # noqa: E402
from fastmace.rewrites.radial import spline as sp  # noqa: E402

SPECIES = (38, 22, 8)  # Sr, Ti, O


def c2_coefficients(chain64, weight64, pairs, grid):
    """``[P, intervals, 4, Q]`` float64 power-basis coefficients of the clamped C2 cubic interpolant per pair."""
    x = sp.grid_knots(grid)
    n = len(x) - 1
    h = x[1:] - x[:-1]
    out = []
    for s, t in pairs:
        with torch.no_grad():
            g, dg = chain64.with_derivative(x, s, t)
        y = g[:, :-1].double() @ weight64.T  # [n + 1, Q]
        end = dg[:, :-1].double() @ weight64.T
        # interior knots i = 1 .. n - 1: h[i] m[i-1] + 2 (h[i-1] + h[i]) m[i] + h[i-1] m[i+1] = rhs[i]
        a = torch.zeros(n + 1, n + 1, dtype=torch.float64)
        rhs = torch.zeros_like(y)
        a[0, 0] = a[n, n] = 1.0
        rhs[0], rhs[n] = end[0], end[n]
        for i in range(1, n):
            a[i, i - 1], a[i, i], a[i, i + 1] = h[i], 2 * (h[i - 1] + h[i]), h[i - 1]
            rhs[i] = 3 * (h[i] * (y[i] - y[i - 1]) / h[i - 1] + h[i - 1] * (y[i + 1] - y[i]) / h[i])
        m = torch.linalg.solve(a, rhs)
        hh = h[:, None]
        y0, y1, m0, m1 = y[:-1], y[1:], m[:-1] * hh, m[1:] * hh
        out.append(torch.stack([y0, m0, 3 * (y1 - y0) - 2 * m0 - m1, 2 * (y0 - y1) + m0 + m1], dim=1))
    return torch.stack(out)


def score(scorer, chain64, weight64, pairs, grid, kind):
    coef = (c2_coefficients if kind == "c2" else sp.coefficients)(chain64, weight64, pairs, grid).to(torch.float32)
    relative = scorer.relative(coef, grid)
    return scorer.certificate(relative), scorer.bands(relative, grid)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("output", type=Path)
    p.add_argument("--interaction", type=int, default=1)
    args = p.parse_args(argv)
    torch.set_num_threads(8)
    model = torch.load(args.checkpoint, map_location="cpu", weights_only=False).float()
    numbers = [int(z) for z in model.atomic_numbers]
    columns = [numbers.index(z) for z in SPECIES]
    pairs = [(s, t) for s in range(len(columns)) for t in range(len(columns))]
    interaction = model.interactions[args.interaction]
    chain64 = rt._GenericChain(model, interaction, columns, torch.float64, torch.device("cpu"))
    chain32 = rt._GenericChain(model, interaction, columns, torch.float32, torch.device("cpu"))
    weight = chain64.folded
    weight64 = weight.detach().double()
    r_max = float(model.r_max)
    started = time.time()
    spline, cert = sp.fit(chain64, chain32, weight, pairs, rt.R_LO, r_max, rt.CERTIFY_POINTS, rt.R_PHYSICAL,
                          seed=args.interaction)
    prod_grid = [tuple(g) for g in spline["grid"]]
    print(f"production Hermite: {cert['intervals']} intervals, certificate {cert['worst']:.3f} "
          f"({time.time() - started:.0f} s)", flush=True)
    scorer = sp._Scorer(chain64, chain32, weight64, pairs, rt.R_LO, r_max, rt.CERTIFY_POINTS, rt.R_PHYSICAL,
                        args.interaction)
    result = dict(tool="spline_c2_certificate", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  hermite=dict(grid=prod_grid, intervals=cert["intervals"], worst=cert["worst"], parts=cert["parts"]))
    c2_cert, c2_bands = score(scorer, chain64, weight64, pairs, prod_grid, "c2")
    result["c2_on_production_grid"] = dict(worst=c2_cert["worst"], parts=c2_cert["parts"], bands=c2_bands)
    print(f"C2 on production's grid: certificate {c2_cert['worst']:.3f} {c2_cert['parts']}", flush=True)
    print("  bands: " + ", ".join(f"[{a}, {b}) {n}: {s:.3f}" for (a, b, n), s in zip(prod_grid, c2_bands)), flush=True)
    # production's refinement rule with the C2 coefficients, from the base grid
    edges = [rt.R_LO] + [e for e in sp.BAND_EDGES if rt.R_LO < e < r_max] + [r_max]
    base = (r_max - rt.R_LO) / sp.BASE_INTERVALS
    grid = [(a, b, max(1, math.ceil((b - a) / base - 1e-9))) for a, b in zip(edges, edges[1:])]
    rounds = []
    for k in range(1, 2 * sp.ROUNDS + 1):
        c, bands = score(scorer, chain64, weight64, pairs, grid, "c2")
        intervals = sum(n for _, _, n in grid)
        rounds.append(dict(round=k, intervals=intervals, worst=c["worst"], grid=[list(g) for g in grid], bands=bands))
        print(f"C2 refinement round {k}: {intervals} intervals, certificate {c['worst']:.3f}", flush=True)
        if all(s <= sp.TARGET for s in bands):
            break
        grid = [(a, b, n if s <= sp.TARGET else math.ceil(n * 1.05 * (s / sp.TARGET) ** (1 / 3)))
                for (a, b, n), s in zip(grid, bands)]
    result["c2_refined"] = rounds
    result["c2_passes_target"] = all(s <= sp.TARGET for s in rounds[-1]["bands"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
