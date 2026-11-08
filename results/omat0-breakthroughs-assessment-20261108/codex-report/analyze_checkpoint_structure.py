import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import numpy as np
import fastmace as fm
from fastmace.rewrites.radial import spline as sp_mod
from fastmace.rewrites.generic.modules import fc_last_weight

ckpt = "/home/fzwin/models/mace-omat-0-medium.model"
loaded = fm.load(ckpt, trusted=True)
model = loaded.model

# 1. Inspect Interaction 1 and Interaction 2 radial weights & spline
inter0 = model.interactions[0]
inter1 = model.interactions[1]

w0_last = fc_last_weight(list(inter0.conv_tp_weights.children())[-1]).detach().double()
w1_last = fc_last_weight(list(inter1.conv_tp_weights.children())[-1]).detach().double()
print("w0_last shape:", w0_last.shape)
print("w1_last shape:", w1_last.shape)

# Check singular values of w1_last (1280 x 64) and per-path blocks (128 x 64)
U, S, Vh = torch.linalg.svd(w1_last, full_matrices=False)
print("w1_last top 10 singular values:", S[:10].tolist())
print("w1_last bottom 10 singular values:", S[-10:].tolist())

for p in range(10):
    wp = w1_last[p*128:(p+1)*128, :]
    sp = torch.linalg.svdvals(wp)
    print(f"Path {p} (128x64) sv max={sp[0]:.4f}, sv[31]={sp[31]:.4f}, sv[63]={sp[63]:.4f}")

# 2. Check interaction 1 linear and product 1 linear & contractions
print("\n--- Interaction 1 (2nd layer) modules ---")
print("inter1.linear:", inter1.linear)
print("inter1.skip_tp:", inter1.skip_tp)
print("prod1:", model.products[1])
print("readout0:", model.readouts[0])
print("readout1:", model.readouts[1])
print("scale_shift:", model.scale_shift)
for name, param in model.scale_shift.named_parameters():
    print("scale_shift param:", name, param.shape, param)
for name, buf in model.scale_shift.named_buffers():
    print("scale_shift buffer:", name, buf.shape, buf[:10])
