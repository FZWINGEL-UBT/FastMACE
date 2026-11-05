"""Pieces and certificate of the radial tables per polynomial degree and domain (5 November 2026, H200 research
answers: "split-domain tables").

The w' backward reads, per edge and weight, a table row of ``DEGREE + 1`` terms padded to the float2 vector
(``PADDED_TERMS``: degree 5 -> 6, degree 4 -> 6, degree 3 -> 4), and its value table costs about 40 % of the kernel
on the H200 (lab variant F-no-table-horner). A lower degree on the physical range ``[R_PHYSICAL, r_max)``, with the
current degree kept below it, would cut those loads if it does not need many more pieces. This fits each generic
interaction's tables offline with ``fastmace.rewrites.radial.fit`` for every requested (degree, r_lo) and reports
the piece count, the table bytes, the fit error and the float32 certificate (``certify``, CPU arithmetic).

Usage: python tools/radial_degree_lab.py --checkpoint mace-omat-0-medium.model --elements Sr Ti O \
    [--degrees 3 4 5] [--r-lo 0.3 1.0] [--output FILE]
"""

from __future__ import annotations

import argparse
import json
import time

import fastmace  # noqa: F401  (before torch: soft file locks)


def main(argv=None):
    import torch
    from ase.data import atomic_numbers

    from fastmace.rewrites import radial

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--elements", nargs="+", default=["Sr", "Ti", "O"])
    p.add_argument("--degrees", nargs="+", type=int, default=[3, 4, 5])
    p.add_argument("--r-lo", nargs="+", type=float, default=[radial.R_LO, radial.R_PHYSICAL])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)

    model = torch.load(args.checkpoint, map_location="cpu", weights_only=False).float()
    zs = [int(z) for z in model.atomic_numbers]
    columns = [zs.index(atomic_numbers[e]) for e in args.elements]
    per_pair = radial._species_radial(model)
    pairs = [(s, t) for s in range(len(columns)) for t in range(len(columns))] if per_pair else [(0, 0)]
    r_max = float(model.r_max)
    defaults = (radial.DEGREE, radial.R_LO)
    report = {"checkpoint": args.checkpoint, "elements": args.elements, "per_pair": per_pair, "r_max": r_max,
              "tolerance": radial.TOLERANCE, "default": {"degree": defaults[0], "r_lo": defaults[1]}, "rows": []}
    try:
        for index, interaction in radial._generic_interactions(model):
            chain64 = radial._GenericChain(model, interaction, columns, torch.float64, torch.device("cpu"))
            chain32 = radial._GenericChain(model, interaction, columns, torch.float32, torch.device("cpu"))
            for r_lo in args.r_lo:
                for degree in args.degrees:
                    radial.DEGREE, radial.R_LO = degree, r_lo
                    start = time.perf_counter()
                    tables = radial.fit(chain64, pairs, r_max, table_dtype=torch.float32)
                    module = radial._table_module(tables, columns, torch.device("cpu"), single=not per_pair)
                    certificate = radial.certify(module, chain64, chain32, pairs, r_max, seed=index)
                    pieces = tables["coef"].shape[1]
                    terms = degree + 1
                    padded = terms + terms % 2
                    weights = tables["coef"].shape[-1]
                    row = {"interaction": index, "degree": degree, "r_lo": r_lo, "pieces": pieces,
                           "padded_terms": padded, "fit_error": tables["fit_error"],
                           "certificate": certificate["worst"], "largest_error": certificate["largest_error"],
                           "table_mb": len(pairs) * pieces * padded * weights * 4 / 2**20,
                           "seconds": round(time.perf_counter() - start, 1)}
                    report["rows"].append(row)
                    print(json.dumps(row), flush=True)
    finally:
        radial.DEGREE, radial.R_LO = defaults
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
