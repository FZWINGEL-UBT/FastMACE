"""Idea 1 oracle: interaction 2's far-shell edges (r > r_h) in float16 instead of omitted.

Variant of results/omat0-breakthroughs-assessment-20261108/outside-box-report/skip_oracle.py. Instead of masking the
far edges' messages, it rounds what a packed-half2 kernel would hold in float16:
  io   -- the far edges' inputs x[sender], Y (edge_attrs), w (tp_weights, cutoff folded, as the tables hold them),
          the per-edge message mji, and in the backward dout (the message gradient) and the input gradients
          (Round16 rounds the gradient in its backward too).
  acc  -- io, plus the far messages summed per receiver in a float16 accumulator (sequential, edge by edge, as one
          thread's __hfma2 accumulator would hold them), merged into the float64/float32 sum once per receiver.
  *_s  -- the same with w rounded after a per-edge power-of-two scale (max |w_row| -> [0.5, 1)), so the cutoff's
          small far-edge weights do not lose bits to float16 subnormals.
Not modelled: the float16 rounding of intermediate products inside one edge's tensor product (the 10 paths and
their CG sums run in the run's dtype on rounded inputs) -- so the numbers are a lower bound on a kernel's error.

The model runs in float32 (production) with the far edges in float16, against the float64 model; the score is the
max force / stress error over the float32 run's own error (the force/stress gate allows 3x).
CPU only. Usage: python half_far_oracle.py SrTiO3.cif [out.json [repeat]]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ase.build import make_supercell
from ase.io import read

import mace.modules.blocks as blocks

CHECKPOINT = Path.home() / "models/mace-omat-0-medium.model"
CIF = sys.argv[1]
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_suffix(".json")
REPEAT, SIGMA, SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 4, 0.05, 1
THRESHOLDS = [5.0, 5.5, 5.8]
VARIANTS = ["io", "acc", "io_s", "acc_s"]


def structure():
    atoms = read(CIF)
    atoms = make_supercell(atoms, np.eye(3, dtype=int) * REPEAT) if len(atoms) < 10 else atoms.repeat(REPEAT)
    rng = np.random.default_rng(SEED)
    atoms.positions += rng.normal(0.0, SIGMA, atoms.positions.shape)
    return atoms


def r16(t):
    return t.to(torch.float16).to(t.dtype)


class Round16(torch.autograd.Function):
    """Round the masked rows to float16 in the forward and their gradient in the backward."""

    @staticmethod
    def forward(ctx, t, mask, scaled):
        ctx.save_for_backward(mask)
        ctx.scaled = scaled
        return _round_rows(t, mask, scaled)

    @staticmethod
    def backward(ctx, g):
        (mask,) = ctx.saved_tensors
        return _round_rows(g, mask, ctx.scaled), None, None


def _round_rows(t, mask, scaled):
    out = t.clone()
    rows = t[mask]
    if scaled:
        m = rows.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
        s = torch.exp2(-torch.floor(torch.log2(m)) - 1)  # max |row| * s in [0.5, 1)
        out[mask] = r16(rows * s) / s
    else:
        out[mask] = r16(rows)
    return out


class FarSum16(torch.autograd.Function):
    """scatter_sum with the masked (far) rows summed per receiver in a float16 accumulator; backward = exact gather."""

    @staticmethod
    def forward(ctx, src, index, mask, dim_size):
        ctx.save_for_backward(index)
        near = torch.zeros((dim_size,) + src.shape[1:], dtype=src.dtype)
        near.index_add_(0, index[~mask], src[~mask])
        fsrc, fidx = src[mask], index[mask]
        order = torch.argsort(fidx, stable=True)
        fsrc, fidx = fsrc[order], fidx[order]
        start = torch.searchsorted(fidx, fidx, right=False)
        rank = torch.arange(fidx.numel()) - start
        acc = torch.zeros((dim_size,) + src.shape[1:], dtype=torch.float16)
        for k in range(int(rank.max()) + 1 if rank.numel() else 0):
            sel = rank == k
            acc[fidx[sel]] = acc[fidx[sel]] + fsrc[sel].to(torch.float16)
        return near + acc.to(src.dtype)

    @staticmethod
    def backward(ctx, g):
        (index,) = ctx.saved_tensors
        return g[index], None, None, None


def run(model, atoms, dtype, r_h=None, variant=None):
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
    block = model.interactions[1]
    tp = block.conv_tp
    original_tp, original_scatter = tp.forward, blocks.scatter_sum
    stats = {}

    def half_tp(x, y, w, *rest):
        far = captured["r"] > r_h
        scaled = variant.endswith("_s")
        stats["w_far_absmax"] = float(w.detach()[far].abs().max()) if far.any() else 0.0
        stats["w_far_frac_subnormal16"] = float(((w[far].abs() < 6.1e-5) & (w[far] != 0)).double().mean())
        stats["x_absmax"] = float(x.abs().max())
        x, y = Round16.apply(x, far, False), Round16.apply(y, far, False)
        w = Round16.apply(w, far, scaled)
        out = original_tp(x, y, w, *rest)
        stats["msg_far_absmax"] = float(out[far].abs().max()) if far.any() else 0.0
        captured["far"] = far
        return Round16.apply(out, far, False)

    def half_scatter(src, index, dim, dim_size=None, **kw):
        if "far" in captured and src.shape[0] == captured["far"].shape[0] and dim == 0:
            far = captured.pop("far")
            return FarSum16.apply(src, index, far, dim_size)
        return original_scatter(src=src, index=index, dim=dim, dim_size=dim_size, **kw)

    if r_h is not None:
        tp.forward = half_tp
        if variant.startswith("acc"):
            blocks.scatter_sum = half_scatter
    try:
        out = model(batch, training=False, compute_force=True, compute_stress=True)
    finally:
        tp.forward = original_tp
        blocks.scatter_sum = original_scatter
        for h in hooks:
            h.remove()
    return (out["forces"].detach().double().numpy(), out["stress"].detach().double().numpy().reshape(3, 3),
            captured["r"].double().numpy(), stats)


def main():
    model = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    atoms = structure()
    f64, s64, r, _ = run(model, atoms, torch.float64)
    f32, s32, _, _ = run(model, atoms, torch.float32)
    rf, rs = float(np.abs(f32 - f64).max()), float(np.abs(s32 - s64).max())
    print(dict(f32_force_rounding=rf, f32_stress_rounding=rs), flush=True)
    rows = []
    for r_h in THRESHOLDS:
        for variant in VARIANTS:
            for dtype in (torch.float32, torch.float64):
                f, s, _, st = run(model, atoms, dtype, r_h, variant)
                ef, es = float(np.abs(f - f64).max()), float(np.abs(s - s64).max())
                rows.append(dict(r_h=r_h, variant=variant, dtype=str(dtype).split(".")[-1],
                                 half_edge_fraction=float((r > r_h).mean()),
                                 max_force_error_eV_A=ef, force_over_f32_rounding=ef / rf,
                                 max_stress_error=es, stress_over_f32_rounding=es / rs, **st))
                print(rows[-1], flush=True)
    out = dict(checkpoint=str(CHECKPOINT), cif=CIF, repeat=REPEAT, sigma_A=SIGMA, seed=SEED, atoms=len(atoms),
               edges=int(r.size), float32_force_rounding_eV_A=rf, float32_stress_rounding=rs, gate_ratio=3.0,
               rows=rows)
    OUT.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
