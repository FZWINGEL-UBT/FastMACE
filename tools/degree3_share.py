"""Round 2, outside idea 4 (5 November 2026): how many (pair, piece)s of a degree-5 radial table could drop to degree 3.

For each cached table (``FASTMACE_CACHE/radial-tables``, picked by shape) and each (pair, piece), the Chebyshev series
truncated after T_3 is compared with the full degree-5 series on a grid of each piece, for every column, by the fit's
own criterion (``radial.fit``): the value error over max(the piece's largest |value|, FLOOR x the domain's largest)
and the same for the r-derivative, both at most TOLERANCE. A (pair, piece) qualifies when every column passes. The
edge-weighted share uses the edge lengths of a structure (perfect lattice plus Gaussian jitter).

Usage: python tools/degree3_share.py --structure results/loup-a100-20261103/SrTiO3.cif@10x10x10 --output FILE
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def series(coef, x):
    """sum_k coef[..., k, j] T_k(x) for x [n] -> [..., n, j]."""
    terms = coef.shape[-2]
    t = np.ones((terms, x.size))
    if terms > 1:
        t[1] = x
    for k in range(2, terms):
        t[k] = 2 * x * t[k - 1] - t[k - 2]
    return np.einsum("...kj,kn->...nj", coef, t)


def analyse(entry, tolerance, floor_factor, keep):
    tables = entry["tables"]
    coef = tables["coef"].double().numpy()  # [P, K, T, J]
    scales = tables["scales"].double().numpy()  # [K]
    pairs, pieces, terms, columns = coef.shape
    x = np.linspace(-1.0, 1.0, 33)
    full = series(coef, x)  # [P, K, n, J]
    trunc = series(coef[:, :, :keep], x)
    dfull = series(np.polynomial.chebyshev.chebder(coef, axis=2), x) * scales[None, :, None, None]
    dtrunc = series(np.polynomial.chebyshev.chebder(coef[:, :, :keep], axis=2), x) * scales[None, :, None, None]
    floor = floor_factor * np.abs(full).max(axis=(1, 2), keepdims=True)  # [P, 1, 1, J]
    dfloor = floor_factor * np.abs(dfull).max(axis=(1, 2), keepdims=True)
    err = np.abs(full - trunc).max(axis=2) / np.maximum(np.abs(full).max(axis=2), floor[:, :, 0])
    derr = np.abs(dfull - dtrunc).max(axis=2) / np.maximum(np.abs(dfull).max(axis=2), dfloor[:, :, 0])
    # A column (or its derivative) that is zero on the whole domain has nothing to fit (``radial.fit``): in float32
    # tables a constant cutoff column's derivative series is rounding noise, so "zero" means below 1e-9 of the
    # largest column's.
    live = floor[:, :, 0] > 1e-9 * floor.max()
    dlive = dfloor[:, :, 0] > 1e-9 * dfloor.max()
    err = np.where(live, err, 0.0)
    derr = np.where(dlive, derr, 0.0)
    worst = np.maximum(err, derr).max(axis=2)  # [P, K]
    return worst <= tolerance, worst, tables


def edge_pieces(structure, tables, jitter, seed):
    from ase.io import read
    from ase.neighborlist import neighbor_list

    path, _, repeat = structure.partition("@")
    atoms = read(path)
    if repeat:
        atoms = atoms.repeat(tuple(int(v) for v in repeat.split("x")))
    rng = np.random.default_rng(seed)
    atoms.positions += rng.normal(0.0, jitter, atoms.positions.shape)
    r_max = float(tables["r_max"])
    i, j, d = neighbor_list("ijd", atoms, r_max)
    r_lo, bin_scale = float(tables["r_lo"]), float(tables["bin_scale"])
    bins = tables["bins"].numpy()
    b = np.clip(((np.maximum(d, r_lo) - r_lo) * bin_scale).astype(np.int64), 0, bins.size - 1)
    return bins[b], atoms.numbers[i], atoms.numbers[j]


def main(argv=None):
    import torch

    from fastmace import _cache
    from fastmace.rewrites import radial

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", default="results/loup-a100-20261103/SrTiO3.cif@10x10x10")
    p.add_argument("--shapes", nargs="*", default=["9x896x6x66", "9x624x6x66"],
                   help="coef shapes P x K x T x J of the tables to analyse (OMAT-0 medium, three species)")
    p.add_argument("--keep", type=int, default=4, help="terms kept (4: degree 3)")
    p.add_argument("--jitter", type=float, default=0.07, help="Gaussian displacement (Å) for the edge lengths")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    directory = _cache.cache_dir(radial.CACHE_KIND, create=False)
    found = {}
    for path in sorted(directory.glob("*.pt"), key=lambda f: f.stat().st_mtime):
        entry = torch.load(path, map_location="cpu", weights_only=True)
        shape = "x".join(str(v) for v in entry["tables"]["coef"].shape)
        if shape in args.shapes:
            found[shape] = (path, entry)  # the newest of each shape
    result = {}
    for shape, (path, entry) in found.items():
        ok, worst, tables = analyse(entry, radial.TOLERANCE, radial.FLOOR, args.keep)
        pieces_, _, _ = edge_pieces(args.structure, tables, args.jitter, seed=0)
        all_pairs = ok.all(axis=0)  # a piece usable for every pair
        any_pair = ok.any(axis=0)
        counts = np.bincount(pieces_, minlength=ok.shape[1])
        result[shape] = dict(
            file=path.name, pairs=ok.shape[0], pieces=ok.shape[1],
            pair_piece_share=float(ok.mean()),
            edge_share_all_pairs=float((counts * all_pairs).sum() / counts.sum()),
            edge_share_any_pair=float((counts * any_pair).sum() / counts.sum()),
            median_worst_ratio_to_tolerance=float(np.median(worst) / radial.TOLERANCE),
            edges=int(counts.sum()))
        print(shape, json.dumps(result[shape]), flush=True)
    args.output.write_text(json.dumps(dict(structure=args.structure, keep=args.keep, jitter=args.jitter,
                                           tolerance=radial.TOLERANCE, floor=radial.FLOOR, tables=result),
                                      indent=1) + "\n")


if __name__ == "__main__":
    main()
