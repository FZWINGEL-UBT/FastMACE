import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import torch
import fastmace as fm

ckpt = "/home/fzwin/models/mace-omat-0-medium.model"
loaded = fm.load(ckpt, trusted=True)
model = loaded.model

info = {
    "r_max": float(model.r_max),
    "num_interactions": int(model.num_interactions),
    "atomic_numbers": model.atomic_numbers.tolist(),
    "num_elements": len(model.atomic_numbers),
    "node_embedding": str(model.node_embedding),
    "radial_embedding": str(model.radial_embedding),
    "spherical_harmonics": str(model.spherical_harmonics),
    "interactions": [],
    "products": [],
    "readouts": [],
}

for i, inter in enumerate(model.interactions):
    d = {
        "index": i,
        "type": type(inter).__name__,
        "node_attrs_irreps": str(inter.node_attrs_irreps),
        "node_feats_irreps": str(inter.node_feats_irreps),
        "edge_attrs_irreps": str(inter.edge_attrs_irreps),
        "edge_feats_irreps": str(inter.edge_feats_irreps),
        "target_irreps": str(inter.target_irreps),
        "hidden_irreps": str(inter.hidden_irreps),
        "avg_num_neighbors": float(inter.avg_num_neighbors),
        "linear_up": str(inter.linear_up),
        "conv_tp": str(inter.conv_tp),
        "conv_tp_instructions": [str(inst) for inst in inter.conv_tp.instructions],
        "conv_tp_weights": str(inter.conv_tp_weights),
        "linear": str(inter.linear),
        "skip_tp": str(inter.skip_tp),
        "density_fn": str(getattr(inter, "density_fn", None)),
    }
    info["interactions"].append(d)

for i, prod in enumerate(model.products):
    sc = prod.symmetric_contractions
    d = {
        "index": i,
        "type": type(prod).__name__,
        "irreps_in": str(sc.irreps_in),
        "irreps_out": str(sc.irreps_out),
        "correlation": int(prod.correlation) if hasattr(prod, "correlation") else None,
        "linear": str(prod.linear),
        "contractions": [],
    }
    for c in sc.contractions:
        c_info = {
            "irreps_in": str(getattr(c, "irreps_in", "")),
            "irrep_out": str(getattr(c, "irrep_out", getattr(c, "irreps_out", ""))),
            "correlation": int(c.correlation),
            "weights_max": list(c.weights_max.shape),
            "weights": [list(w.shape) for w in c.weights],
            "U_tensors": [list( getattr(c, f"U_matrix_{nu}").shape ) for nu in range(1, c.correlation + 1)],
        }
        d["contractions"].append(c_info)
    info["products"].append(d)

for i, ro in enumerate(model.readouts):
    info["readouts"].append({
        "index": i,
        "type": type(ro).__name__,
        "repr": str(ro),
    })

print(json.dumps(info, indent=2))
