import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import math
import torch
import fastmace as fm
from fastmace.rewrites.radial import _GenericChain, CERTIFY_POINTS, R_LO, R_PHYSICAL, spline as sp_mod
from fastmace.rewrites.generic.modules import fc_last_weight

ckpt = "/home/fzwin/models/mace-omat-0-medium.model"
loaded = fm.load(ckpt, trusted=True)
model = loaded.model
inter0 = model.interactions[0]
inter1 = model.interactions[1]

# SrTiO3 species indices in atomic_numbers: O=8 (idx 7), Ti=22 (idx 21), Sr=38 (idx 37)
species_cols = [7, 21, 37]
pairs_9 = [(s, t) for s in range(len(species_cols)) for t in range(len(species_cols))]

chain64 = _GenericChain(model, inter1, species_cols, torch.float64, torch.device("cpu"))
chain32 = _GenericChain(model, inter1, species_cols, torch.float32, torch.device("cpu"))
weight1 = chain64.folded  # [1280, 65]

sp, cert = sp_mod.fit(chain64, chain32, weight1, pairs_9, R_LO, float(model.r_max), CERTIFY_POINTS, R_PHYSICAL, seed=1)
print("Spline grid:", sp["grid"])
print("Spline coef shape:", sp["coef"].shape, "size MB (one table):", sp["coef"].numel() * 4 / 1e6)
print("Spline cert worst:", cert["worst"], "parts:", cert["parts"], "intervals:", cert["intervals"])

# Check magnitude of c0, c1, c2, c3 in sp["coef"] [9, 731, 4, 1280]
coef = sp["coef"]  # [9, 731, 4, 1280]
for term in range(4):
    c_t = coef[:, :, term, :].abs()
    print(f"Term c{term}: max={c_t.max().item():.6e}, median={c_t.median().item():.6e}, p99={torch.quantile(c_t.flatten(), 0.99).item():.6e}")

# Also check in physical region r >= 1.0 (intervals >= 356)
print("\n--- Physical region (r >= 1.0 A, intervals 356..730, 375 intervals) ---")
for term in range(4):
    c_t = coef[:, 356:, term, :].abs()
    print(f"Physical Term c{term}: max={c_t.max().item():.6e}, median={c_t.median().item():.6e}")

# Now test 8-byte (64-bit) compressed spline representations on the EXACT _Scorer!
scorer = sp_mod._Scorer(chain64, chain32, weight1, pairs_9, R_LO, float(model.r_max), CERTIFY_POINTS, R_PHYSICAL, seed=1)

# Test 1: c0 in float32 (32b), c1 in float32 or bf16/fp16, c2/c3 in fp16 or int8/int16
# Wait: what if we store 8 bytes (64 bits) per (interval, weight) as:
# Format A (float32 c0 + fp16 c1 + fp16 c2, c3=0):
coef_A = coef.clone()
coef_A[:, :, 1, :] = coef[:, :, 1, :].to(torch.float16).to(torch.float32)
coef_A[:, :, 2, :] = coef[:, :, 2, :].to(torch.float16).to(torch.float32)
coef_A[:, :, 3, :] = 0.0
cert_A = scorer.certificate(scorer.relative(coef_A, sp["grid"]))
print("\nFormat A (f32 c0, fp16 c1, fp16 c2, c3=0) cert:", cert_A)

# Format B: What if we keep c0, c1, c2, c3 all non-zero in 8 bytes?
# How? Look at c1, c2, c3:
# If we store (c0 in float32 [32 bits]) + (c1 in 16-bit, c2 in 10-bit, c3 in 6-bit) OR
# Wait: at the knots, y_i = c0[i] and m_i * h_i = c1[i] are SHARED between interval i and interval i+1!
# WAIT A SECOND!
# Look at sp_mod.coefficients (spline.py lines 49-50):
# On uniform-band intervals (inside each of the 7 bands, h_i is CONSTANT!):
# y0 = y[i], y1 = y[i+1], m0 = m[i]*h, m1 = m[i+1]*h!
# Notice that AT KNOT i, there are ONLY TWO NUMBERS: y[i] and m[i]!
# Wait: for N intervals, there are N+1 knots!
# At each knot i, storing (y[i], m[i]) is ONLY ONE FLOAT2 (8 BYTES) PER KNOT!
print("\nWait: Hermite representation at knots stores (y[i], m[i]) = 2 floats (8 bytes) per knot!")
