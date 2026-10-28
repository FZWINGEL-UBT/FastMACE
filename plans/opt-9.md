# Optimisation round 9: the radial backward as a dot with tabulated `w'(r)`

Owner request, 28 October 2026 ("Yes screen it"), on the fp32 profile's idea (branch
`fastmace/fp32-profile-20261028`, 28 October 2026): the backward half of the fused table → `w` → convolution
kernel, which OPT8 left at today's chunked backward.

## Idea

Today, float32 MH-1 (`radial_tables`, `fused_folded_forward`, `chunked_conv_backward` on): the convolution
backward writes `dw [E, 1280]` in L2-sized chunks; the Triton input gradient reads each chunk on a side stream for
`da = dw W` (`W [1280, 65]`, 83k FMA per edge); the radial tables' backward then forms `dr = <da, a'(r)>` from the
table's derivative series.

Within one table piece `a(r) = Σ_t T_t(x) C_t`, so `w(r) = Σ_t T_t(x) G_t` with `G = C Wᵀ` per (pair, piece), and
`dr = Σ_c dw_c w_c'(r)` with `w' = Σ_t T_t(x) G'_t`, `G' = C' Wᵀ` (ideas D2, `fastmace.kernels.triton.w_table`,
float64 only so far). In float32 the tables have degree 10, so `w'` costs 10 × 1280 FMA per edge instead of
65 × 1280. Inside the convolution backward the dot needs no `dw` write: it would drop the `dw` chunk stores, the
input gradient and the table backward.

**Known limits before measuring.**
- The input gradient already runs mostly hidden beside the convolution backward: the fp32 profile's F0 ceiling
  (removing it) is 2.6 % of the NaNbO3 step and 0.5 % of NbCl5.
- `G'` is 50 KB per (pair, piece) in float32: NaNbO3 586 groups, 29 MiB; NbCl5 213 groups, 10 MiB. A
  receiver-owned kernel visits the groups in edge order, so each edge reads its group's 50 KB from L2 (or DRAM
  where `G'` does not fit), against the 5 KB `dw` row it saves.
- The values change at rounding level (the series and the product are reassociated), so a record would be
  rounding-class, like D2.

## Tasks

| Task | What | Stop rule | Status |
| --- | --- | --- | --- |
| **OPT9-T01** | Falsifier, no library code (`tools/bench/opt9_wprime_dot.py`). Synthetic operands with MH-1's float32 shapes, edges in receiver order with random groups. Standalone, CUDA graphs of 20 calls with the L2 flushed before each call (flush subtracted), 15 alternating blocks. What the idea removes: the input gradient over 1,536-edge chunks from two L2 slots plus the table backward (`chain`) and the `dw` chunk stores (`dw_store`). What a fused kernel must add: `w'` from `G'` and the dot, `dw` in registers (`wdot`; also edges sorted by group, and `dw` loaded) | Go to T02 only if `wdot` ≤ 0.5 × (`chain` + `dw_store`) on NaNbO3 **and** NbCl5. The removed work is mostly hidden today, so a fused kernel that adds half of it back cannot clear the F0-sized ceiling. Otherwise report in `docs/validation/opt9-T01.md`, commit, push, stop | **done, not passed**, [report](../docs/validation/opt9-T01.md): exact to rounding (≤ 3.6e-8). `wdot` / removed: A100 **0.21 / 0.17** (NaNbO3 / NbCl5), RTX 4000 Ada **0.76 / 0.71**. The receiver-order `G'` gathers (50 KB per edge) are L2-bound on the Ada; sorted by group it would be 0.14, but a receiver-owned kernel cannot sort. **Stop** |
| OPT9-T02 | Only after T01 passes: a step-level ceiling (the default plan with the input gradient, the table backward and the `dw` stores removed, wrong forces, timing only), then a prototype in the vendored streaming backward | ceiling ≥ 3 % of the NaNbO3 step and ≥ 1.5 % of NbCl5 | not started (T01 stopped) |
