"""Check the NumPy re-implementation (radial_cheb.Radial) against mace-torch 0.3.16 itself, in float64:
the per-edge TP weights w(r) = conv_tp_weights(...) * cutoff of every interaction, for a few species pairs."""
import sys

import numpy as np
import torch

import radial_cheb as rc

path = sys.argv[1]
torch.set_default_dtype(torch.float64)
model = torch.load(path, map_location="cpu", weights_only=False).double().eval()
R = rc.Radial(path)
z_table = [int(z) for z in model.atomic_numbers]
r = np.linspace(0.6, float(model.r_max) - 1e-3, 400)
worst = 0.0
for zs, zr in ((11, 17), (41, 8), (8, 8), (41, 41)):
    n = len(r)
    node_attrs = torch.zeros((2, len(z_table)))
    node_attrs[0, z_table.index(zs)] = 1
    node_attrs[1, z_table.index(zr)] = 1
    edge_index = torch.stack([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])  # sender 0 -> receiver 1
    lengths = torch.tensor(r)[:, None]
    atomic_numbers = torch.tensor(z_table)
    with torch.no_grad():
        out = model.radial_embedding(lengths, node_attrs, edge_index, atomic_numbers)
        edge_feats, cutoff = out if isinstance(out, tuple) else (out, None)
        for index, (block, interaction) in enumerate(zip(R.blocks, model.interactions)):
            src = interaction.source_embedding(node_attrs)[edge_index[0]]
            tgt = interaction.target_embedding(node_attrs)[edge_index[1]]
            w = interaction.conv_tp_weights(torch.cat([edge_feats, src, tgt], dim=-1))
            if cutoff is not None:
                w = w * cutoff
            g, _ = R.hidden(block, "conv_tp_weights", r, zs, zr, np.float64)
            w_np = g @ R.folded_last(block, np.float64).T
            rel = float(np.abs(w_np - w.numpy()).max() / np.abs(w.numpy()).max())
            worst = max(worst, rel)
            print(f"interaction {index} pair {zs}-{zr}: max |w_numpy - w_mace| / max |w| = {rel:.2e}")
print(f"worst {worst:.2e}")
