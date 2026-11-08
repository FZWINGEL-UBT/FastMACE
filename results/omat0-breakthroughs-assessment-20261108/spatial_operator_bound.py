"""Counting bound for a fast spatial operator (mesh/FFT or H2/FMM) in place of OMAT-0 medium's interaction-2 edge walks.

Codex report of 8 November 2026, route 1. The budget is the report's: the whole replacement (forward, transpose,
spatial derivative, near field) must cost at most 1,176.7 us on the H200 for a 20 % step saving. Plain arithmetic, no
GPU; writes spatial_operator_bound.json next to this file. Every count is a lower bound on the operator it describes
(peak FP32, no memory traffic, no accuracy requirement).
"""
import json
import math
from pathlib import Path

# SrTiO3 10x10x10 (results/loup-a100-20261103), OMAT-0 medium interaction 2.
ATOMS, BOX, EDGES, CUTOFF = 5000, 39.05, 373_760, 6.0
SPECIES, CHANNELS = 3, 128
INPUT_COMPONENTS = 512      # 128x0e + 128x1o per sender
RAW_COMPONENTS = 40         # path-output components per channel (5120 per receiver)
PRODUCTS = 62               # distinct input x SH products per channel (checkpoint CG census)
DIRECT_FORWARD_GFLOP = 11.865
BUDGET_US = 1176.7
H200_FP32 = 67e12

density = ATOMS / BOX ** 3
out = {"density_per_A3": density, "edges_per_atom": EDGES / ATOMS, "budget_us": BUDGET_US}

# Mesh / FFT. Forward needs one FFT per source field and one inverse per receiver output field; in between, per
# grid point, channel and species pair, the same 62 products as an edge (complex on a real-FFT half grid: 4 real
# FLOP per product per real grid point, counted as 2 to stay a lower bound).
mesh = {}
for spacing in (0.5, 0.25):
    side = math.ceil(BOX / spacing)
    points = side ** 3
    fft = 2.5 * points * math.log2(points)                   # real FFT, FLOP
    sources = SPECIES * INPUT_COMPONENTS
    outputs = SPECIES * CHANNELS * RAW_COMPONENTS
    fft_gflop = (sources + outputs) * fft / 1e9
    pointwise_gflop = points * CHANNELS * SPECIES ** 2 * PRODUCTS * 2 / 1e9
    forward = fft_gflop + pointwise_gflop
    mesh[f"{spacing} A"] = {
        "points": points, "points_per_atom": points / ATOMS,
        "pair_sites_per_edge": points * SPECIES ** 2 / EDGES,
        "fft_gflop": fft_gflop, "pointwise_gflop": pointwise_gflop,
        "forward_gflop": forward, "forward_over_direct": forward / DIRECT_FORWARD_GFLOP,
        "forward_us_at_fp32_peak": forward * 1e9 / H200_FP32 * 1e6,
        "over_budget": forward * 1e9 / H200_FP32 * 1e6 / BUDGET_US,
    }
out["mesh"] = mesh

# H2 / black-box FMM. An admissible box pair costs rank^2 per field (M2L) against n^2 direct (n atoms per box), so
# compression needs n > rank = p^3 (Chebyshev p per dimension). The box side for that n, against the cutoff.
fmm = {}
for p in (2, 3, 4):
    rank = p ** 3
    side = (rank / density) ** (1 / 3)
    fmm[f"p={p}"] = {"rank": rank, "box_side_A_for_n_equal_rank": side,
                     "non_adjacent_pairs_within_cutoff": side < CUTOFF,
                     "neighbour_volume_beyond_one_box_side": max(0.0, 1 - (side / CUTOFF) ** 3)}
out["fmm"] = fmm
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
