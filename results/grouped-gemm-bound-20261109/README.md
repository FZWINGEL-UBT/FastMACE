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

## The report's other proposals

These were assessed against the code, the ledger and earlier labs on 9 November. Nothing was built, and each has a
row in the ledger. Proposal 5, distillation, was set aside by the owner.

| Proposal | Claim | Verdict | Evidence |
| --- | --- | --- | --- |
| 1: precomputed locator in w′ | 180–240 µs | Rejected: already measured | Under STAGE_EDGES the locator is computed at staging, not in the edge loop. The band chain is warp-uniform (one edge per warp). The precomputed record is `shared-locator`, H200 0.9966 / 0.9974. The cited SASS file does not exist, and w′ is limited by load latency, not instruction issue. |
| 2: spatial (Hilbert) slab walking | 110–150 µs | Rejected: already modelled | Morton order moved the 8 Nov L1 model by 0.02–0.04. Sorted edges plus species blocks (0.75) are SORT_BY_LENGTH + BLOCK_ORDER, both default. |
| 4: interaction-1 outer sum fused with its species GEMM | 45–65 µs | Rejected | The 61 MB intermediate is written at about 0.5 TB/s, so the kernel is not traffic bound. Fusing would give up the CUTLASS GEMM's rate. |
| 6: interaction 2 every 2 fs | 840–950 µs | Owner decision, not pursued | It changes the dynamics, and as described it is not symplectic. Plain 2 fs Langevin is the simpler test of whether 1 fs is required. |
