"""CPU only: the error of Symmetrix-XL-style cubic Hermite splines for interaction 2's w, next to our certified table.

sx-layout lab (7 November 2026, results/sx-layout-lab-20261107). For MACE-OMAT-0 medium and SrTiO3's species
(9 ordered pairs), interaction ``--interaction`` (default 1, the second):

- our table: ``radial.fit`` (degree 5, the production fit) evaluated by ``radial._table_values`` on the CPU;
- splines with N uniform knots on [R_LO, r_max): float64 values and exact r-derivatives of the radial chain at the
  knots (``_GenericChain.with_derivative``), the segments' power-basis coefficients in t = (r - x_i) / h computed in
  float64 and stored in float32, evaluated in float32 (the lab kernels' arithmetic: w = c0 + t (c1 + t (c2 + t c3)),
  w' = (c1 + t (2 c2 + 3 t c3)) / h).

Every candidate is scored as ``radial.certify`` scores a table: per pair and output column, the largest error over the
column's largest exact value, against the float64 chain, divided by the float32 network's own error; at the w level
(the chain's [h c, c] times the folded last layer). Values and slopes, the whole domain and r >= R_PHYSICAL. A table
passes at <= 1. Writes JSON.
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


def spline_coefficients(chain64, pairs, r_lo, r_max, knots):
    """``[P, knots - 1, 4, Q]`` float64 power-basis coefficients in t of the cubic Hermite spline of w (values and
    exact slopes at the knots); ``h``."""
    h = (r_max - r_lo) / (knots - 1)
    x = r_lo + h * torch.arange(knots, dtype=torch.float64)
    folded = chain64.folded
    out = []
    for s, t in pairs:
        with torch.no_grad():
            g, dg = chain64.with_derivative(x, s, t)
        y = g[:, :-1].double() @ folded.T  # [knots, Q]
        m = dg[:, :-1].double() @ folded.T * h  # slopes in t
        y0, y1, m0, m1 = y[:-1], y[1:], m[:-1], m[1:]
        c2 = 3 * (y1 - y0) - 2 * m0 - m1
        c3 = 2 * (y0 - y1) + m0 + m1
        out.append(torch.stack([y0, m0, c2, c3], dim=1))
    return torch.stack(out), h


def spline_eval(coef32, h, r_lo, r32, pair):
    """float32 values and r-derivatives ``[n, Q]`` of the float32 spline table at float32 lengths."""
    inv_h = torch.tensor(1.0 / h, dtype=torch.float32)
    u = (r32 - torch.tensor(r_lo, dtype=torch.float32)) * inv_h
    i = torch.clamp(torch.floor(u).long(), 0, coef32.shape[1] - 1)
    t = (u - i.float())[:, None]
    c = coef32[pair, i]  # [n, 4, Q]
    c0, c1, c2, c3 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    value = c0 + t * (c1 + t * (c2 + t * c3))
    slope = (c1 + t * (2 * c2 + 3 * t * c3)) * inv_h
    return value, slope


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("output", type=Path)
    p.add_argument("--interaction", type=int, default=1)
    p.add_argument("--knots", type=int, nargs="*", default=[256, 512, 1024, 2048, 4096])
    args = p.parse_args(argv)
    torch.set_num_threads(8)
    started = time.time()
    model = torch.load(args.checkpoint, map_location="cpu", weights_only=False).float()
    numbers = [int(z) for z in model.atomic_numbers]
    columns = [numbers.index(z) for z in SPECIES]
    pairs = [(s, t) for s in range(len(columns)) for t in range(len(columns))]
    r_max, r_lo = float(model.r_max), rt.R_LO
    interaction = model.interactions[args.interaction]
    chain64 = rt._GenericChain(model, interaction, columns, torch.float64, torch.device("cpu"))
    chain32 = rt._GenericChain(model, interaction, columns, torch.float32, torch.device("cpu"))
    folded = chain64.folded

    gen = torch.Generator().manual_seed(0)
    n = rt.CERTIFY_POINTS
    r64 = torch.cat([torch.linspace(r_lo, r_max, n + 1, dtype=torch.float64)[:-1],
                     r_lo + (r_max - r_lo) * torch.rand(n, generator=gen, dtype=torch.float64)])
    r64 = r64.to(torch.float32).to(torch.float64)
    r32 = r64.to(torch.float32)
    physical = r64 >= rt.R_PHYSICAL

    def to_w(g):
        return g.detach().to(torch.float64)[:, :-1] @ folded.T  # [n, Q]

    truth, net = {}, {}
    with torch.no_grad():
        for k, (s, t) in enumerate(pairs):
            truth[k] = [to_w(v) for v in chain64.with_derivative(r64, s, t)]
            net[k] = [to_w(v) for v in chain32.with_derivative(r32, s, t)]

    def score(evaluate):
        """{kind/region: worst ratio over pairs}, the worst, and the largest relative error."""
        parts, largest = {}, 0.0
        for k in range(len(pairs)):
            got = evaluate(k)
            for j, kind in enumerate(("value", "slope")):
                exact, network, tabled = truth[k][j], net[k][j], got[j].to(torch.float64)
                for region_name, region in (("all", slice(None)), ("physical", physical)):
                    e = exact[region]
                    scale = e.abs().amax(dim=0).clamp_min(1e-300)
                    en = ((network[region] - e).abs() / scale).max().item()
                    et = ((tabled[region] - e).abs() / scale).max().item()
                    largest = max(largest, et)
                    key = f"{kind}/{region_name}"
                    parts[key] = max(parts.get(key, 0.0), et / en if en > 0 else float("inf"))
                    parts["abs:" + key] = max(parts.get("abs:" + key, 0.0), et)
                    parts["net:" + key] = max(parts.get("net:" + key, 0.0), en)
        worst = max(v for key, v in parts.items() if ":" not in key)
        return dict(worst=worst, largest_error=largest, parts=parts)

    report = dict(checkpoint=args.checkpoint, interaction=args.interaction, species=SPECIES, r_lo=r_lo, r_max=r_max,
                  points=int(r64.numel()), physical_from=rt.R_PHYSICAL, candidates={})

    tables = rt.fit(chain64, pairs, r_max)
    module = rt._table_module(tables, columns, torch.device("cpu"))

    def table_eval_w(k):
        pair = torch.full(r32.shape, k, dtype=torch.int32)
        values, slopes = rt._table_values(module, r32, pair)  # [n, Q_in + 2]: [h c, c, density]
        return to_w(values), to_w(slopes)  # to_w drops the density column

    pieces = int(module.pieces)
    report["candidates"]["table (degree 5, certified)"] = dict(
        score(table_eval_w), pieces=pieces, terms=int(tables["coef"].shape[2]),
        w_table_MB=len(pairs) * pieces * 6 * folded.shape[0] * 4 / 1e6)
    print("table", json.dumps(report["candidates"]["table (degree 5, certified)"]["parts"]), flush=True)

    for knots in args.knots:
        coef, h = spline_coefficients(chain64, pairs, r_lo, r_max, knots)
        coef32 = coef.to(torch.float32)
        result = score(lambda k: spline_eval(coef32, h, r_lo, r32, torch.full(r32.shape, k, dtype=torch.long)))
        result.update(spacing_A=h, w_table_MB_9pairs=coef32.numel() * 4 / 1e6,
                      w_table_MB_6pairs=coef32.numel() * 4 / 1e6 * 6 / 9)
        report["candidates"][f"spline {knots}"] = result
        print(f"spline {knots}", json.dumps(result["parts"]), flush=True)

    report["seconds"] = time.time() - started
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print("done", round(report["seconds"], 1), "s")


if __name__ == "__main__":
    main()
