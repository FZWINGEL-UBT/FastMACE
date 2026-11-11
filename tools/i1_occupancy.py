"""Edges per (receiver, sender species, table piece) group for the interaction tables (9 November 2026, external
LLM proposal "polynomial moments for the first interaction").

The tabled outer sum accumulates ``S[receiver, sender species] += y_e a(r_e)^T`` with ``a(r)`` a degree-D
Chebyshev series on the piece of ``r``. Within one (receiver, sender species, piece) group the coefficients are
shared, so ``S`` could take the moments ``M_k = sum_e y_e T_k(x_e)`` per group and one ``(D + 1)``-term contraction
with the piece's coefficients per group instead of one per edge. That only wins when groups hold several edges
(the proposal's break-even: about 4.4 edges per group). This reports the group occupancy for each fitted table on a
rattled SrTiO3 supercell: CPU only, the tables fitted as ``radial_degree_lab`` does (default degree and r_lo).

Usage: python tools/i1_occupancy.py --checkpoint mace-omat-0-medium.model [--repeat 10] [--sigma 0 0.05 0.08 0.12]
    [--output FILE]
"""

from __future__ import annotations

import argparse
import json
import time

import fastmace  # noqa: F401  (before torch: soft file locks)


def srtio3(repeat, sigma, seed=0):
    import numpy as np
    from ase import Atoms

    cell = Atoms("SrTiO3", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5),
                                             (0, 0.5, 0.5)], cell=[3.905] * 3, pbc=True)
    atoms = cell.repeat(repeat)
    if sigma:  # Gaussian displacements: 300 K thermal RMS in SrTiO3 is about 0.06-0.09 A per axis
        atoms.positions = atoms.positions + np.random.default_rng(seed).normal(0.0, sigma, atoms.positions.shape)
    atoms.wrap()
    return atoms


def occupancy(atoms, tables, r_max, species_of):
    import numpy as np
    from ase.neighborlist import neighbor_list

    i, j, d = neighbor_list("ijd", atoms, r_max)  # i receives from j (the sum is symmetric in direction)
    bins = tables["bins"].numpy()
    r_lo, bin_scale = tables["r_lo"], tables["bin_scale"]
    index = np.clip(((np.maximum(d, r_lo) - r_lo) * bin_scale).astype(np.int64), 0, bins.size - 1)
    piece = bins[index].astype(np.int64)
    pieces = int(piece.max()) + 1
    sender = species_of[j]
    key = (i.astype(np.int64) * 8 + sender) * (pieces + 1) + piece
    _, counts = np.unique(key, return_counts=True)
    edges = d.size
    hist = np.bincount(counts)
    # edge-weighted: the fraction of edges that sit in a group of size n
    weighted = {int(n): float(n * c / edges) for n, c in enumerate(hist) if c}
    return {"edges": int(edges), "groups": int(counts.size), "mean": float(edges / counts.size),
            "edge_weighted_mean": float((counts.astype(np.float64) ** 2).sum() / edges),
            "max": int(counts.max()), "groups_by_size": {int(n): int(c) for n, c in enumerate(hist) if c},
            "edge_fraction_by_size": weighted,
            "edges_in_groups_ge_5": float(counts[counts >= 5].sum() / edges),
            "pieces_used": int(np.unique(piece).size)}


def main(argv=None):
    import numpy as np
    import torch
    from ase.data import atomic_numbers

    from fastmace.rewrites import radial

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--elements", nargs="+", default=["Sr", "Ti", "O"])
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--sigma", nargs="+", type=float, default=[0.0, 0.05, 0.08, 0.12])
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)

    model = torch.load(args.checkpoint, map_location="cpu", weights_only=False).float()
    zs = [int(z) for z in model.atomic_numbers]
    columns = [zs.index(atomic_numbers[e]) for e in args.elements]
    per_pair = radial._species_radial(model)
    pairs = [(s, t) for s in range(len(columns)) for t in range(len(columns))] if per_pair else [(0, 0)]
    r_max = float(model.r_max)
    report = {"checkpoint": args.checkpoint, "elements": args.elements, "repeat": args.repeat,
              "degree": radial.DEGREE, "r_lo": radial.R_LO, "r_max": r_max, "tables": []}
    for index, interaction in radial._generic_interactions(model):
        start = time.perf_counter()
        chain64 = radial._GenericChain(model, interaction, columns, torch.float64, torch.device("cpu"))
        tables = radial.fit(chain64, pairs, r_max, table_dtype=torch.float32)
        row = {"interaction": index, "pieces": int(tables["coef"].shape[1]), "fit_s": time.perf_counter() - start,
               "cases": []}
        for sigma in args.sigma:
            atoms = srtio3(args.repeat, sigma)
            symbols = atoms.get_chemical_symbols()
            species_of = np.array([args.elements.index(s) for s in symbols], dtype=np.int64)
            case = {"sigma": sigma, "atoms": len(atoms), **occupancy(atoms, tables, r_max, species_of)}
            row["cases"].append(case)
            print(f"interaction {index} ({row['pieces']} pieces) sigma {sigma:.2f} A: {case['edges']} edges, "
                  f"{case['groups']} groups, mean {case['mean']:.2f}, edge-weighted {case['edge_weighted_mean']:.2f},"
                  f" max {case['max']}, edges in groups >= 5: {100 * case['edges_in_groups_ge_5']:.1f} %",
                  flush=True)
        report["tables"].append(row)
    text = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
