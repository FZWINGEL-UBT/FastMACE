# H200 packs: A100 switches never measured on the H200 (11–12 November 2026)

Modal NVIDIA H200 (sm_90, driver 610.57.04), torch 2.14.1+cu130, float32 hybrid. Owner approvals in the conductor
chat: "i approve the h200 pack" (pack 1) and "run it now" (pack 2), both beyond the month's $30 cap per workspace.
b/a = switch on over the H200's default plan, MD step, same process, both build orders (ab / ba); below 1 is faster.
Gate: `tools/force_stress_gate.py`, 3× the float32 error.

| Folder | Profile | Wall | Cost (metered) | Content |
| --- | --- | --- | --- | --- |
| `pack1/` | sixth | 810 s | about $1.19 | `tf32x3.HYBRID`, `spline_forward.GEOM`: OMAT-0 medium A/B (SrTiO3 10x10x10, NbCl5 3x3x3) and gates |
| `pack2/` | fourth | 1,200 s (timeout) | $1.76 | `fused_forward.TP_NOFLUSH`, `WFRAG_REGION` (OMAT-0 medium A/B); MACE-OFF23-medium offall A/B and its default gate |
| `pack2-offgate/` | third | 362 s | $0.82 | MACE-OFF23-medium offall candidate gate (pack 2 hit its timeout during it) |

## Results

| Switch | Structure | b/a (ab / ba) | Gate | Verdict |
| --- | --- | --- | --- | --- |
| `tf32x3.HYBRID` | SrTiO3 10³ | 0.9909 / 0.9940 | passes (forces ≤ 0.99, stress ≤ 1.04) | **on** |
| | NbCl5 | 0.9892 / 0.9885 | | |
| `spline_forward.GEOM` | SrTiO3 10³ | 0.9857 / 0.9890 | passes (forces ≤ 0.95, stress ≤ 1.30) | **on** |
| | NbCl5 | 0.9891 / 0.9860 | | |
| `fused_forward.WFRAG_REGION` (bitwise) | SrTiO3 10³ | 0.9877 / 0.9921 | not needed | **on** (6 Nov: a tie against older defaults) |
| | NbCl5 | 0.9695 / 0.9913 | | |
| `fused_forward.TP_NOFLUSH` | SrTiO3 10³ | 0.9968 / 1.0030 | – | off (mixed) |
| | NbCl5 | 0.9397 / 1.0038 | | (the ab baseline ran 5 % slow) |
| MACE-OFF offall: radial tables with the stock first block, `STOCK_FIRST_SKIP`, `RADIAL_TF32X3` | OFF23-medium water_216 | 0.7417 / 0.7422 (2.43 → 1.80 ms) | passes, see below | **on** |

MACE-OFF23-medium gate ratios (energy / forces / stress), default against candidate:

| Structure | Default | Candidate |
| --- | --- | --- |
| urea | 1.78 / 1.00 / 0.72 | 2.40 / 1.00 / 0.78 |
| naphthalene | 1.46 / 1.03 / 1.13 | 2.24 / 1.00 / 1.50 |
| paracetamol | 1.07 / 0.97 / 0.79 | 1.86 / 1.25 / 1.51 |
| water_216 | 0.65 / 0.97 / 1.00 | 0.95 / 0.95 / 1.04 |

The energy ratios are against a 0.01 absolute floor; all pass. The MACE-OFF switches were measured on OFF23-medium
only; the A100 found them faster on all five OFF models (results/off-polar-alex-20261106), so the H200 takes them
for the family.

Not measured here: the tiled-mode switches (`TILE_LARGE_CAPS`, `TILE_FIRST_CALL_PROBE`, `COMPILED_GRAPH`), which
matter only for systems beyond one device's memory. Everything else that is on for the A100 only was already
measured on the H200 (h200-defaults-20261105, step-limiters-20261109, slab-sorted-h200-20261108).
