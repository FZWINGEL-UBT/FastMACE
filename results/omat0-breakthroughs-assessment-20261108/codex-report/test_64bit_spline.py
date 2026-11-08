import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import math
import torch
import fastmace as fm
from fastmace.rewrites.radial import _GenericChain, R_LO, R_PHYSICAL, spline as sp_mod

ckpt = "/home/fzwin/models/mace-omat-0-medium.model"
loaded = fm.load(ckpt, trusted=True)
model = loaded.model
inter1 = model.interactions[1]
species_cols = [7, 21, 37]
pairs_6 = [(i, j) for i in range(3) for j in range(i, 3)]

chain64 = _GenericChain(model, inter1, species_cols, torch.float64, torch.device("cpu"))
chain32 = _GenericChain(model, inter1, species_cols, torch.float32, torch.device("cpu"))
weight1 = chain64.folded

POINTS = 1000
scorer = sp_mod._Scorer(chain64, chain32, weight1, pairs_6, R_LO, float(model.r_max), POINTS, R_PHYSICAL, seed=1)

# Test quadratic spline (c3 = 0) fitted by least-squares / Chebyshev on each interval!
def fit_quadratic(chain64, weight64, pairs, grid):
    # On each interval [x_i, x_{i+1}], fit quadratic c0 + c1*t + c2*t^2 matching derivative at endpoints m0, m1
    # and midpoint value!
    # Wait: If w'(t) = c1 + 2*c2*t matches m0 at t=0 and m1 at t=1:
    # c1 = m0, c2 = 0.5 * (m1 - m0)!
    # And if c0 is chosen to match value at t=0.5 (or average of endpoints):
    # w(0.5) = y_mid => c0 = y_mid - 0.5*c1 - 0.25*c2 (or c0 = y0)!
    x = sp_mod.grid_knots(grid)
    x_mid = 0.5 * (x[:-1] + x[1:])
    h = (x[1:] - x[:-1])[:, None]
    out = []
    for s, t in pairs:
        with torch.no_grad():
            g, dg = chain64.with_derivative(x, s, t)
            g_mid = chain64(x_mid, s, t)
        y = g[:, :-1].double() @ weight64.T
        y_mid = g_mid[:, :-1].double() @ weight64.T
        m = dg[:, :-1].double() @ weight64.T
        m0, m1 = m[:-1] * h, m[1:] * h
        c1 = m0
        c2 = 0.5 * (m1 - m0)
        c0 = y_mid - 0.5 * c1 - 0.25 * c2
        c3 = torch.zeros_like(c0)
        out.append(torch.stack([c0, c1, c2, c3], dim=1))
    return torch.stack(out)

def quantize_mantissa(x, mant_bits):
    if mant_bits >= 23:
        return x
    shift = 23 - mant_bits
    bits = x.view(torch.int32)
    half = 1 << (shift - 1)
    rounded = (bits + half) & ~((1 << shift) - 1)
    return rounded.view(torch.float32)

# Refine grid for quadratic spline until certificate <= 0.5
edges = [R_LO] + [e for e in sp_mod.BAND_EDGES if R_LO < e < 6.0] + [6.0]
base = (6.0 - R_LO) / 255
grid = [(a, b, max(1, math.ceil((b - a) / base - 1e-9))) for a, b in zip(edges, edges[1:])]
for r_iter in range(1, 12):
    coef_q = fit_quadratic(chain64, weight1, pairs_6, grid).to(torch.float32)
    rel = scorer.relative(coef_q, grid)
    scores = scorer.bands(rel, grid)
    cert_q = scorer.certificate(rel)
    if all(s <= 0.65 for s in scores):
        break
    grid = [(a, b, n if s <= 0.65 else math.ceil(n * 1.05 * (s / 0.65) ** 0.5)) for (a, b, n), s in zip(grid, scores)]

print(f"Quadratic spline intervals={sum(n for _,_,n in grid)}, cert worst={cert_q['worst']:.4f}, parts={cert_q['parts']}", flush=True)

# Now test quantizing (c0, c1, c2) of this quadratic spline into 64 bits (uint2)!
# In 64 bits:
# Suppose c0 has 19 mantissa bits (1s + 8e + 19m = 28 bits)
# c1 has 16 mantissa bits (1s + 8e + 16m = 25 bits) -> wait: 28 + 25 = 53 bits!
# Remaining 11 bits for c2: 1s + 4e_diff (relative to c1 exp) + 6m = 11 bits! Total = 64 bits!
for b0, b1, b2 in [(19, 16, 6), (20, 16, 7), (19, 17, 7), (18, 16, 8)]:
    cq = coef_q.clone()
    cq[:, :, 0, :] = quantize_mantissa(coef_q[:, :, 0, :], b0)
    cq[:, :, 1, :] = quantize_mantissa(coef_q[:, :, 1, :], b1)
    cq[:, :, 2, :] = quantize_mantissa(coef_q[:, :, 2, :], b2)
    res = scorer.certificate(scorer.relative(cq, grid))
    print(f"  Quadratic 64-bit mantissa ({b0}, {b1}, {b2}) -> worst={res['worst']:.4f}, parts={res['parts']}", flush=True)

# Also check cubic spline with mantissa quantization into 64 bits when grid is slightly finer (TARGET=0.35):
sp_mod.TARGET = 0.35
sp35, cert35 = sp_mod.fit(chain64, chain32, weight1, pairs_6, R_LO, float(model.r_max), POINTS, R_PHYSICAL, seed=1)
c35 = sp35["coef"]
print(f"\nCubic TARGET=0.35 intervals={cert35['intervals']}, base cert={cert35['worst']:.4f}")
for b0, b1, b2, b3 in [(19, 16, 8, 4), (20, 17, 8, 4), (20, 18, 9, 5)]:
    cq = c35.clone()
    cq[:, :, 0, :] = quantize_mantissa(c35[:, :, 0, :], b0)
    cq[:, :, 1, :] = quantize_mantissa(c35[:, :, 1, :], b1)
    cq[:, :, 2, :] = quantize_mantissa(c35[:, :, 2, :], b2)
    cq[:, :, 3, :] = quantize_mantissa(c35[:, :, 3, :], b3)
    res = scorer.certificate(scorer.relative(cq, sp35["grid"]))
    print(f"  Cubic TARGET=0.35 mantissa ({b0}, {b1}, {b2}, {b3}) -> worst={res['worst']:.4f}, parts={res['parts']}", flush=True)
