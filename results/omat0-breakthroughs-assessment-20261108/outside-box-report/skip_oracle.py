"""Oracle falsifier for certified adaptive evaluation (idea 2): how much interaction-2 edge work can be omitted at all
while forces stay inside the gate's budget, with a *perfect* certificate that knows each omission's effect?

Interaction 2's per-edge messages (conv_tp) are masked for edges longer than r_s; everything else is unchanged
(interaction 1, the density, the products and readouts). Forces in float64 against the full float64 model. The
budget is the production rounding error, float32 against float64 on the same structure (the gate allows 3x that,
so omission may add at most about 2x). Distance is the natural ranking for omission: the cutoff envelope makes the
longest edges the smallest; any certificate's omitted set is at best as cheap as the oracle's.

CPU only (CUDA_VISIBLE_DEVICES empty). Writes skip_oracle.json beside this file.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ase.build import make_supercell
from ase.io import read

CHECKPOINT = Path.home() / "models/mace-omat-0-medium.model"
CIF = sys.argv[1]  # results/loup-a100-20261103/SrTiO3.cif
REPEAT, SIGMA, SEED = 4, 0.05, 1
THRESHOLDS = [5.0, 5.5, 5.8, 5.9, 5.95, 5.98, 5.99]


def structure():
    atoms = read(CIF)
    atoms = make_supercell(atoms, np.eye(3, dtype=int) * REPEAT) if len(atoms) < 10 else atoms.repeat(REPEAT)
    rng = np.random.default_rng(SEED)
    atoms.positions += rng.normal(0.0, SIGMA, atoms.positions.shape)
    return atoms


def forces(model, atoms, dtype, r_s=None):
    from mace import data
    from mace.tools import torch_geometric, utils
    torch.set_default_dtype(dtype)
    model = model.to(dtype)
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    config = data.config_from_atoms(atoms)
    loader = torch_geometric.dataloader.DataLoader(
        [data.AtomicData.from_config(config, z_table=z_table, cutoff=float(model.r_max), heads=["default"])],
        batch_size=1, shuffle=False)
    batch = next(iter(loader)).to_dict()
    captured = {}
    hooks = [model.radial_embedding.register_forward_pre_hook(
        lambda mod, args: captured.__setitem__("r", args[0].detach().reshape(-1)))]
    tp = model.interactions[1].conv_tp
    original = tp.forward

    def masked(x, y, w, *rest):
        keep = (captured["r"] <= r_s).to(w.dtype)[:, None]
        return original(x, y, w * keep, *rest)

    if r_s is not None:
        tp.forward = masked
    try:
        out = model(batch, training=False, compute_force=True)
    finally:
        tp.forward = original
        for h in hooks:
            h.remove()
    return out["forces"].detach().double().numpy(), captured["r"].double().numpy()


def main():
    model = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    atoms = structure()
    f64, r = forces(model, atoms, torch.float64)
    f32, _ = forces(model, atoms, torch.float32)
    rounding = float(np.abs(f32 - f64).max())
    rows = []
    for r_s in THRESHOLDS:
        fs, _ = forces(model, atoms, torch.float64, r_s)
        err = float(np.abs(fs - f64).max())
        rows.append(dict(r_s=r_s, omitted_edge_fraction=float((r > r_s).mean()), max_force_error_eV_A=err,
                         error_over_float32_rounding=err / rounding, error_over_gate_floor=err / 1e-5))
        print(rows[-1], flush=True)
    out = dict(checkpoint=str(CHECKPOINT), cif=CIF, repeat=REPEAT, sigma_A=SIGMA, seed=SEED, atoms=len(atoms),
               edges=int(r.size), max_abs_force_eV_A=float(np.abs(f64).max()),
               float32_rounding_max_force_error_eV_A=rounding, gate_force_floor_eV_A=1e-5, thresholds=rows)
    Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "thresholds"}, indent=1))


if __name__ == "__main__":
    main()
