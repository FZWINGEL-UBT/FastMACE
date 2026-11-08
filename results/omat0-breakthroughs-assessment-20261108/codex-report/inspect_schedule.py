import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import torch
from fastmace.kernels.oeq_streaming.schedule import build_schedule

ckpt = "/home/fzwin/models/mace-omat-0-medium.model"
import fastmace as fm
loaded = fm.load(ckpt, trusted=True)
model = loaded.model

inter1 = model.interactions[1]
sched = build_schedule(inter1.conv_tp, layout="ir_mul")
print("Schedule input_dim:", sched.input_dim)
print("Schedule edge_dim:", sched.edge_dim)
print("Schedule output_dim:", sched.output_dim)
print("Schedule weight_numel:", sched.weight_numel)
print("Schedule channels:", sched.channels)
print("Num paths:", len(sched.paths))
for idx, p in enumerate(sched.paths):
    terms = [(o.accumulator.name, o.accumulator.start, o.accumulator.component, o.accumulator.irrep_dim,
              [(t.input_component, t.edge_component, float(t.coefficient)) for t in o.terms])
             for o in p.outputs]
    print(f"Path {idx}: in_start={p.input_start}, in_dim={p.input_irrep_dim}, edge_start={p.edge_start}, edge_dim={p.edge_irrep_dim}, w_start={p.weight_start}, outputs={len(p.outputs)}, total_terms={sum(len(o.terms) for o in p.outputs)}")
    for t in terms:
        print("  out:", t)
