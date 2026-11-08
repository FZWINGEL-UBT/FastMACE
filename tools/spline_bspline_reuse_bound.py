"""CPU only: L1 reuse of a cubic B-spline table against today's Hermite slabs, SPLINE_W forward (9 November 2026).

Codex's 9 November report (proposal 2) suggests storing interaction 2's radial weights as a uniform cubic B-spline:
one 5.12 KB control point (1,280 floats) per knot instead of a 20 KB slab of four coefficients per interval. An edge in
interval k reads control points k .. k + 3, so it still reads 20 KB, but neighbouring intervals share 3 of their 4
points. That can raise L1 reuse between co-resident receivers whose edges sit in nearby intervals.

The model is ``tools/spline_table_ideas_bound.py``'s ``reuse``: SrTiO3 10x10x10 with Gaussian displacements, the cached
production grid, one receiver per 128-thread block, ``--resident`` blocks per SM, an LRU per SM, and the resident
blocks' edges interleaved. Two table layouts are compared at the same L1 bytes:

- ``hermite``: one 20 KB slab per (ordered pair, interval), an LRU of ``C`` slabs;
- ``bspline``: four 5.12 KB rows per edge, (pair, knot k + o) for o = 0..3 with a continuous knot index across bands,
  an LRU of ``4 C`` rows (a refit: a C2 spline, so the certificate and gate again);
- ``knot-hermite``: today's Hermite spline stored per knot as (value, slope), 10.24 KB per knot, an edge in interval k
  reading knots k and k + 1, an LRU of ``2 C`` rows. The same function as production (the four coefficients are
  formed from the two knots in registers: rounding only).

Bytes fetched into L1 per edge = misses x unit size. The B-spline also needs four separate loads per weight where the
Hermite slab needs one float4; this bound counts bytes only.

    python tools/spline_bspline_reuse_bound.py OUTPUT.json [--grid-from TABLE_IDEAS_BOUND.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spline_table_ideas_bound as base  # noqa: E402

SLAB_KB, ROW_KB, KNOT_KB = 20.48, 5.12, 10.24


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", type=Path)
    p.add_argument("--grid-from", type=Path,
                   default=Path("results/omat0-breakthroughs-assessment-20261108/table_ideas_bound.json"))
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resident", type=int, default=6)
    p.add_argument("--l1-slabs", type=int, nargs="*", default=[5, 10])
    args = p.parse_args(argv)
    grid = [tuple(g) for g in json.loads(args.grid_from.read_text())["certify"]["grid"]]
    atoms, i, j, d = base.structure(args.sigma, args.seed)
    order_edges = np.lexsort((j, i))
    i, j, d = i[order_edges], j[order_edges], d[order_edges]
    slab, si = base.slab_ids(atoms.numbers, i, j, d, grid)  # pair * 100000 + interval
    bounds = np.searchsorted(i, np.arange(len(atoms) + 1))
    edges = [sorted(slab[bounds[a]:bounds[a + 1]].tolist()) for a in range(len(atoms))]  # SORT_BY_LENGTH
    rows = [[s + o for s in x for o in range(4)] for x in edges]  # knots k .. k + 3 of interval k
    knots = [[s + o for s in x for o in range(2)] for x in edges]  # Hermite per knot: (value, slope) at k, k + 1
    code = base.morton(atoms.positions, atoms.cell.array)
    orders = {"natural": (list(range(len(atoms))), False),
              "species (BLOCK_ORDER)": (list(np.argsort(si, kind="stable")), False),
              "species+morton, SM-compact": (list(np.lexsort((code, si))), True)}
    result = dict(tool="spline_bspline_reuse_bound", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  sigma=args.sigma, resident=args.resident, grid=grid, edges=int(len(slab)),
                  touched_slabs=len(set(slab.tolist())), cases={})
    for cap in args.l1_slabs:
        for name, (order, compact) in orders.items():
            h_slab = base.simulate(order, edges, args.resident, cap, compact)
            h_row = base.simulate(order, rows, args.resident, 4 * cap, compact)
            h_knot = base.simulate(order, knots, args.resident, 2 * cap, compact)
            kb_slab = (1 - h_slab) * SLAB_KB
            kb_row = 4 * (1 - h_row) * ROW_KB
            kb_knot = 2 * (1 - h_knot) * KNOT_KB
            key = f"L1 {cap * SLAB_KB:.0f} KB | {name}"
            result["cases"][key] = dict(hermite_hit=round(h_slab, 4), bspline_hit=round(h_row, 4),
                                        knot_hermite_hit=round(h_knot, 4),
                                        hermite_kb_per_edge=round(kb_slab, 3), bspline_kb_per_edge=round(kb_row, 3),
                                        knot_hermite_kb_per_edge=round(kb_knot, 3),
                                        bspline_over_hermite=round(kb_row / kb_slab, 4),
                                        knot_hermite_over_hermite=round(kb_knot / kb_slab, 4))
            print(f"{key:44s} hermite {kb_slab:.2f} KB/edge (hit {h_slab:.4f})  knot-hermite {kb_knot:.2f} "
                  f"(x {kb_knot / kb_slab:.3f})  bspline {kb_row:.2f} (x {kb_row / kb_slab:.3f})", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
