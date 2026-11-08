import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import fastmace as fm
from fastmace.rewrites.radial import _GenericChain, CERTIFY_POINTS, R_LO, R_PHYSICAL, spline as sp_mod

ckpt = "/home/fzwin/models/mace-omat-0-medium.model"
loaded = fm.load(ckpt, trusted=True)
model = loaded.model
inter0 = model.interactions[0]
inter1 = model.interactions[1]

species_cols = [7, 21, 37]  # O, Ti, Sr
pairs_6 = [(i, j) for i in range(3) for j in range(i, 3)]

chain64 = _GenericChain(model, inter1, species_cols, torch.float64, torch.device("cpu"))
chain32 = _GenericChain(model, inter1, species_cols, torch.float32, torch.device("cpu"))
weight1 = chain64.folded  # [1280, 65]

# 1. Check the numerical rank of the 1D curve r -> w(r) in R^1280 (and per path in R^128)
# over physical distances r in [1.5, 6.0] A (and [1.0, 6.0] A)!
r_phys = torch.linspace(1.5, 6.0, 4000, dtype=torch.float64)
all_w = []
all_dw = []
all_a = []
for s, t in pairs_6:
    with torch.no_grad():
        g, dg = chain64.with_derivative(r_phys, s, t)
    a = g[:, :-2]  # [4000, 64]
    da = dg[:, :-2]
    w = g[:, :-1] @ weight1.T  # [4000, 1280]
    dw = dg[:, :-1] @ weight1.T
    all_a.append(a)
    all_w.append(w)
    all_dw.append(dw)

A_all = torch.cat(all_a, dim=0)  # [24000, 64]
s_A = torch.linalg.svdvals(A_all)
print("SVD of a(r) [24000 x 64] on r in [1.5, 6.0] A:")
for k in [0, 4, 7, 11, 15, 19, 23, 27, 31, 39, 47, 63]:
    print(f"  sv[{k:2d}] = {s_A[k]:.6e} (ratio to sv[0]: {s_A[k]/s_A[0]:.6e})")

# Check per species pair SVD of a(r) [4000 x 64]:
s_A0 = torch.linalg.svdvals(all_a[0])
print("\nSVD of single-pair a(r) [4000 x 64] on r in [1.5, 6.0] A:")
for k in [0, 4, 7, 11, 15, 19, 23, 27, 31, 39, 47, 63]:
    print(f"  sv[{k:2d}] = {s_A0[k]:.6e} (ratio to sv[0]: {s_A0[k]/s_A0[0]:.6e})")

# Check Interaction 0 a(r) SVD as well!
chain0_64 = _GenericChain(model, inter0, species_cols, torch.float64, torch.device("cpu"))
all_a0 = []
for s, t in pairs_6:
    with torch.no_grad():
        g, dg = chain0_64.with_derivative(r_phys, s, t)
    all_a0.append(g[:, :-2])
A0_all = torch.cat(all_a0, dim=0)
s_A0_inter0 = torch.linalg.svdvals(A0_all)
print("\nSVD of Interaction 0 a(r) [24000 x 64] on r in [1.5, 6.0] A:")
for k in [0, 4, 7, 11, 15, 19, 23, 27, 31, 39, 47, 63]:
    print(f"  sv[{k:2d}] = {s_A0_inter0[k]:.6e} (ratio to sv[0]: {s_A0_inter0[k]/s_A0_inter0[0]:.6e})")
