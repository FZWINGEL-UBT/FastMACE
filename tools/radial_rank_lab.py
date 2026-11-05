"""Numerical rank of the radial features ``a(r)`` (5 November 2026, round 5 idea "low-rank K").

The fused forward's tensor-core product ``w = a W^T`` has K = 64 (the radial network's hidden width). If the radial
features of every species pair, over the whole distance range, lie in an R-dimensional subspace to float32 precision,
then ``w = (a U)(W U)^T`` with ``U`` the leading R right singular vectors cuts K to R, and the product's MMAs in
proportion. This evaluates each interaction's radial chain in float64 on a dense grid for every species pair of the
given elements and reports, per R, the worst error of ``w`` and of ``dw/dr`` relative to their maxima.

Usage: python tools/radial_rank_lab.py --checkpoint mace-omat-0-medium.model --elements Sr Ti O [--r-min 0.5]
"""

from __future__ import annotations

import argparse
import json

import fastmace  # noqa: F401  (before torch: soft file locks)


def main(argv=None):
    import torch
    from ase.data import atomic_numbers

    from fastmace.rewrites.generic import modules as gm
    from fastmace.rewrites.radial import _GenericChain, _generic_interactions, _species_radial

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--elements", nargs="+", default=["Sr", "Ti", "O"])
    p.add_argument("--r-min", type=float, default=0.5)
    p.add_argument("--points", type=int, default=4000)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)

    model = torch.load(args.checkpoint, map_location="cpu", weights_only=False).double()
    zs = [int(z) for z in model.atomic_numbers]
    columns = [zs.index(atomic_numbers[e]) for e in args.elements]
    per_pair = _species_radial(model)
    pairs = [(s, t) for s in range(len(columns)) for t in range(len(columns))] if per_pair else [(0, 0)]
    r_max = float(model.r_max)
    k = torch.arange(args.points, dtype=torch.float64)
    x = torch.cos(torch.pi * (k + 0.5) / args.points)  # Chebyshev nodes on [r_min, r_max)
    r = 0.5 * (args.r_min + r_max) + 0.5 * (r_max - args.r_min) * x
    report = {"checkpoint": args.checkpoint, "elements": args.elements, "per_pair": per_pair, "r_min": args.r_min,
              "r_max": r_max, "points": args.points, "interactions": {}}
    for index, interaction in _generic_interactions(model):
        chain = _GenericChain(model, interaction, columns, torch.float64, torch.device("cpu"))
        hidden, last = gm.fc_split(interaction.conv_tp_weights)
        weight = gm.fc_last_weight(last).detach().double()
        hidden_width = hidden[-1].weight.shape[-1] if hasattr(hidden[-1], "weight") else None
        if weight.shape[0] not in (hidden_width, 64) and weight.shape[1] in (hidden_width, 64):
            weight = weight.t()  # [Q_in, W]: w = a @ weight
        values, slopes = [], []
        for s, t in pairs:
            v, d = chain.with_derivative(r, s, t)
            q = weight.shape[0]  # [h c] columns; the cutoff and density columns follow
            values.append(v[:, :q])
            slopes.append(d[:, :q])
        a = torch.cat(values)
        da = torch.cat(slopes)
        w, dw = a @ weight, da @ weight
        scale = float(da.abs().max() / a.abs().max())
        u, sv, vh = torch.linalg.svd(torch.cat([a, da / scale]), full_matrices=False)
        rows = []
        for rank in (8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64):
            basis = vh[:rank].t()
            proj = basis @ basis.t()
            ew = float(((a @ proj - a) @ weight).abs().max() / w.abs().max())
            edw = float(((da @ proj - da) @ weight).abs().max() / dw.abs().max())
            rows.append({"rank": rank, "w_err": ew, "dw_err": edw})
        # float32 rounding of the current path for scale: a rounded to float32, times W in float64
        e32 = float(((a.float().double() - a) @ weight).abs().max() / w.abs().max())
        report["interactions"][index] = {"q": a.shape[1], "singular_values_rel": (sv / sv[0]).tolist(),
                                         "ranks": rows, "a_float32_rounding_w_err": e32}
        print(f"interaction {index}: Q {a.shape[1]}, pairs {len(pairs)}, a float32 rounding -> w err {e32:.2e}")
        for row in rows:
            print(f"  rank {row['rank']:3d}  w {row['w_err']:.2e}  dw/dr {row['dw_err']:.2e}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
