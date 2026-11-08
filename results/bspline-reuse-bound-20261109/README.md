# Spline-table layout and L1 reuse in the SPLINE_W forward: CPU model (9 November 2026)

**Question.** The 9 November Codex report (second version, proposal 2) suggests storing interaction 2's radial
weights as a uniform cubic B-spline, one 5.12 KB control point per knot. Each edge still reads 20 KB (four points),
but neighbouring intervals share three of their four points. Does that raise L1 reuse enough to matter?

**Model** ([`tools/spline_bspline_reuse_bound.py`](../../tools/spline_bspline_reuse_bound.py)). It is the 8 November L1
model (`tools/spline_table_ideas_bound.py`): SrTiO3 10×10×10, σ = 0.05 Å, the production grid of 736 intervals, one
receiver per 128-thread block, 6 resident blocks per SM, an LRU per SM, and each receiver's edges sorted by length
(SORT_BY_LENGTH). It compares three layouts at the same L1 bytes:

- **hermite**: today's layout, one 20 KB slab per (pair, interval).
- **knot-hermite**: today's spline stored per knot as (value, slope), 10.24 KB per knot, two knots per edge. It is
  the same function, with the coefficients formed in registers, so only rounding changes.
- **bspline**: four 5.12 KB control points per edge. This is a refit (a C² spline) and needs the certificate and
  the gate again.

The CPU run was in WSL (`fastmace_cpu`). Results are in [`bspline_reuse_bound.json`](bspline_reuse_bound.json).

| L1 for the table | Receiver order | hermite KB/edge (hit) | knot-hermite | bspline |
| --- | --- | --- | --- | --- |
| 102 KB | natural | 11.86 (0.42) | 6.84 (× 0.576) | 3.99 (× 0.336) |
| 102 KB | species (BLOCK_ORDER) | 7.14 (0.65) | 3.88 (× 0.543) | 2.27 (× 0.317) |
| 205 KB | natural | 9.33 (0.54) | 6.15 (× 0.659) | 3.90 (× 0.418) |
| 205 KB | **species (BLOCK_ORDER)**, production | 6.06 (0.70) | 3.74 (× 0.617) | 2.26 (× 0.373) |
| 205 KB | species + Morton, SM-compact | 5.19 (0.75) | 3.13 (× 0.605) | 1.87 (× 0.360) |

**What the model shows.** Shared knots cut the bytes fetched into L1 by about 40 % (knot-hermite) or 63 % (bspline)
at the production order. Most of the extra hits come from consecutive edges of one receiver, which fall in nearby
intervals within one neighbour shell, and from co-resident receivers of the same species. The table's size in L2 does
not matter: the run touches only 567 slabs (11.6 MB).

**Bound.** The table share of the H200 forward is about 170 µs (602 µs real against 432 µs with every edge on one
slab). If that time scales with L1 fill bytes, the ceilings are:

- knot-hermite: (1 − 0.617) × 170 ≈ **65 µs** (1.9 % of the 3.42 ms step);
- bspline: (1 − 0.373) × 170 ≈ **107 µs** (3.1 %).

Both have costs the model does not count:

- More load instructions per weight. knot-hermite needs two float2 loads where today's slab needs one float4, and
  bspline needs four scalar loads. The forward already issues many loads.
- Forming the coefficients takes a few FMAs per weight (knot-hermite).

**Triage.** knot-hermite is exact up to rounding, but its ceiling of 65 µs is just under the 68 µs lab bar. bspline
passes the bound stage. Its next step is the CPU falsifier the report names: a cubic B-spline fit on the production
knots (or up to about 750) that must pass the radial certificate at ≤ 0.5. If it passes, the step after that is a
5060 Ti kernel lab of the four-load forward against `tools/spline_table_bound.py`'s real and one-slab variants.
