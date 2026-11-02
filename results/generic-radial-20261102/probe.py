"""Where the generic radial tables of a model differ from its float32 network: per interaction, along r, the table's
and the float32 network's errors against float64 at the w level (``[h c, c] @ [W, 0]^T``) and in the density, as
the self-check measures them (largest absolute difference over the largest value of each case).

Run as ``run.sh NAME results/generic-radial-20261102/probe.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import fastmace  # noqa: F401  (before torch: soft file locks, SPEC §14)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="omat0-medium")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    import torch
    from ase import Atoms

    import fastmace as fm
    from fastmace.rewrites import radial

    atoms = Atoms("SrTiO3", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5),
                                              (0, 0.5, 0.5)], cell=[3.905] * 3, pbc=True).repeat((6, 6, 6))
    field = fm.ForceField(fm.load(args.model), atoms, precision="float32")
    built = field._evaluator(atoms)[1]
    model = getattr(built.model, "_orig_mod", built.model)
    device = next(model.parameters()).device
    columns = list(built.species_columns)
    per_pair = radial._species_radial(model)
    columns = columns if per_pair else columns[:1]
    pairs = [(s, t) for s in range(len(columns)) for t in range(len(columns))]
    r_max = float(model.r_max)
    record = dict(model=args.model, per_pair=per_pair, r_max=r_max, device=torch.cuda.get_device_name(),
                  source_commit=os.environ.get("FASTMACE_SOURCE_COMMIT"),
                  embedding=type(model.radial_embedding).__name__,
                  apply_cutoff=getattr(model.radial_embedding, "apply_cutoff", None),
                  transform=type(getattr(model.radial_embedding, "distance_transform", None)).__name__,
                  interactions=[])
    r64 = torch.linspace(radial.R_LO, r_max, 20001, dtype=torch.float64)[:-1]
    bands = [(radial.R_LO, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, r_max)]
    for index, interaction in radial._generic_interactions(model):
        chain64 = radial._GenericChain(model, interaction, columns, torch.float64, torch.device("cpu"))
        chain32 = radial._GenericChain(model, interaction, columns, torch.float32, device)
        tables = radial.fit(chain64, pairs, r_max, table_dtype=torch.float32)
        table = radial._table_module(tables, columns, device, single=not per_pair)
        certificate = radial.certify(table, chain64, chain32, pairs, r_max, seed=index)
        entry = dict(index=index, pieces=table.pieces, fit_error=tables["fit_error"],
                     certificate=certificate["worst"], bands=[])
        truth = chain64(r64, 0, 0)
        net = chain32(r64.to(device, torch.float32), 0, 0).double().cpu()
        pair = torch.zeros(r64.numel(), dtype=torch.int32, device=device)
        tab, _ = radial._table_values(table, r64.to(device, torch.float32), pair)
        tab = tab.double().cpu()
        folded = chain64.folded

        def w(g):
            return g[:, :-1] @ folded.T

        scale = w(truth).abs().max()
        for lo, hi in bands:
            band = (r64 >= lo) & (r64 < hi)
            entry["bands"].append(dict(
                r=[lo, hi], net=float((w(net)[band] - w(truth)[band]).abs().max() / scale),
                table=float((w(tab)[band] - w(truth)[band]).abs().max() / scale),
                table_vs_net=float((w(tab)[band] - w(net)[band]).abs().max() / scale),
                w_largest=float(w(truth)[band].abs().max() / scale)))
        record["interactions"].append(entry)
        print(json.dumps(entry))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"probe-{args.model}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: record[k] for k in ("per_pair", "r_max", "embedding", "apply_cutoff", "transform")}))


if __name__ == "__main__":
    main()
