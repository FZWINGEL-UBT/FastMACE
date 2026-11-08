# Grouped GEMM for the per-path products: H200 bound (9 November 2026)

**Question.** The 9 November Codex report ("Step-change mechanisms", proposal 3) suggests one CUTLASS grouped GEMM per
product group instead of one launch per path. Its claimed saving is 75–110 µs on the H200, from the partial last
wave of each launch and the launch gaps.

**The groups.** OMAT-0 medium, SrTiO3 10×10×10 (5,000 atoms). Each group has four paths with M = 5,000 × (1, 3, 5, 7)
rows (H200 round-6 breakdown):

| Group | K | N | Production µs |
| --- | --- | --- | ---: |
| interaction 2 `linear_div` | 256 / 384 / 384 / 256 | 128 | 145 |
| its input gradient | 128 | 256 / 384 / 384 / 256 | four of the six `fm_tf32x3_rr` launches |
| interaction 1 species GEMM | 192 | 128 | 99 |
| its gradient | 128 | 192 | 101 |

**Bound** ([`tools/grouped_gemm_bound.py`](../../tools/grouped_gemm_bound.py), Modal H200, profile `sixth`, source
dfc40393, 31 alternating rounds in CUDA graphs).

- `seq`: the four products as four `tf32x3.mm_into` launches, the same 64×64 3xTF32 mainloop as production.
- `one`: one launch of M = 80,000 rows with the group's mean K (or N). It does the same multiply-adds as one
  well-filled grid.

A grouped kernel can at best reach `one`. The per-group problem bookkeeping is not counted, and the epilogues cancel
in the difference.

| Group | seq µs | one µs | Saving µs | one / seq |
| --- | ---: | ---: | ---: | ---: |
| `linear_div` | 137.3 | 118.2 | 19.1 | 0.861 |
| its gradient | 148.5 | 135.5 | 13.1 | 0.912 |
| species | 91.4 | 75.0 | 16.4 | 0.820 |
| species gradient | 93.4 | 80.9 | 12.5 | 0.866 |
| **total** | | | **61.0** | |

The proxies' `seq` times match production well: 137 against 145 µs and 91 / 93 against 99 / 101 µs. Round spreads are
below ±5 µs.

**Verdict: not built (bound).** The ceiling is 61 µs, 1.8 % of the 3.42 ms step. That is above the 1 % stop rule but
below the 68 µs that a lab has to show. A real grouped GEMM would also add per-tile problem lookup and lose some of
this, so the realistic gain is about 40–50 µs. The report's 75–110 µs exceeds the ceiling.
