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
| OPT9-T02 | Only after T01 passes: a step-level ceiling (the default plan with the input gradient, the table backward and the `dw` stores removed, wrong forces, timing only), then a prototype in the vendored streaming backward | ceiling ≥ 3 % of the NaNbO3 step and ≥ 1.5 % of NbCl5 | **ceiling measured on the RTX 4000 Ada (owner request), no prototype**, [report](../docs/validation/opt9-T01.md#step-level-ceiling-opt9-t02-part-1-owner-request-29-october-2026): NaNbO3 0.930 (1.19 ms, 7.0 %), NbCl5 0.965 (0.72 ms, 3.5 %), NaCl 0.926. Above the bar, but below T01's added `wdot` work on this GPU (2.46 / 1.45 ms), so a fused kernel would lose there. A100 (Modal): NaNbO3 **0.828** (17.2 %), NbCl5 0.907 (9.3 %), NaCl 0.939; with T01's `wdot` an estimated 11 % / 7 % net gain there. A100-only, so a prototype needs an owner decision on a hardware rule |

## OPT9-T03 onward: the record `wprime_backward` (decision 66, 29 October 2026)

Decision 66 allows a default by exact GPU name where both build orders of a same-process whole-step A/B measured a
speed-up. The conductor asked for the record and its per-GPU table (29 October 2026); the owner decides the
default.

**Design.**
- **Scope:** float32 MH-1 interactions whose radial table feeds the chunked streaming convolution (in MH-1, the
  second interaction; the first is reassociated). Opt-in record `RewriteConfig.wprime_backward` in
  `fastmace.rewrites.radial`, requiring `radial_tables` and `chunked_conv_backward`; refused off float32 and in
  reproducible plans. Rounding class (the series and the folded product are reassociated), not deterministic
  (the sender gradients' atomics, as today).
- **Forward unchanged:** the table gives `a = [h c, c]` and the edge density; `fused_folded_forward` (or the
  cuBLAS product when it is off) builds and stores `w` and the message. One autograd function takes the edge
  lengths and returns the message and the edge density, so the length gradient has one owner.
- **Backward:** one edge-owned launch over all edges (no chunks, no side stream). The kernel is generated in
  Python from the streaming schedule like `fused_forward.py` (the vendored `.cuh` stays byte-identical). It
  computes `dx` and `dsh` as the vendored backward, keeps each lane's weight gradients in registers and dots them
  with `w'(r) = sum_t T_t(x) G'_t`, `G' = C' [W, b]^T` per (pair, piece) in float32 (`D = 10` terms; the
  derivative series of `w_table.fold_tables`, without its padding term). The lane sums are reduced like `dsh`;
  lane 0 writes `dr_w[e]`. The kernel locates the piece itself with the tables' rule (one bin gather, `x =
  (max(r, r_lo) - start) * scale - 1`); edges at or beyond the cutoff, below `r_lo`, and the padding have a zero
  slope and skip the `G'` reads.
- **Edge density:** its gradient still reaches `r` through the table's own backward on the density column alone
  (`dcoef[..., -1:]`, an `[E, 0]` folded gradient): one small launch replaces the 66-column one. `dL/dr = dr_w +
  dr_density`.
- **Dropped:** the `dw` chunk stores, `skinny_linear.input_grad_into` on the side stream, and the 66-column table
  backward of that interaction.

**Checks.** CPU: the `G'` dot against `dw W -> radial_table_backward` (float64 random tables, `dcoef` the
Chebyshev derivative of `coef`), out-of-domain edges zero; a float64 `gradcheck` of the radial chain with the
density column, and the dot plus the density-only backward equal to its gradient; the generated source and
NVRTC compilation; record and planner. GPU: the kernel's `dx`, `dsh` and `dr` against the chain (streaming
backward, `dw W`, table backward), with padding edges, edges beyond the cutoff, both forwards, and a CUDA graph;
the record's self-check does the same per interaction at setup.

| Task | What | Stop rule | Status |
| --- | --- | --- | --- |
| **OPT9-T03** | The record, its kernel, CPU and GPU tests | GPU exactness: `dr` within float32 rounding of the chain (relative to each edge's largest term), `dx`, `dsh` as the chain's. Otherwise stop and report | **done**, [report](../docs/validation/opt9-T03.md): `dr` 2.5–3.2e-7 of a float64 reference (the chain 4.5–5.3e-7), `dsh` bitwise, `dx` ≤ 2.1e-7; sm_120 and sm_80 |
| **OPT9-T04** | Force/stress gate on NaCl, NbCl5, NaNbO3; 20 ps NVE drift on NaCl and NbCl5 | gate passes and drift within the record-off run's order. Otherwise stop | **done, passed**: forces ratio 0.92 / 1.04 / 1.02; drift NaCl −3.7e-7 (off −1.3e-6), NbCl5 7.8e-7 (off 1.3e-6) meV/atom/ps |
| **OPT9-T05** | Same-process whole-step A/Bs (`tools/ab_rewrites.py`, a = off, b = on), both build orders, on every reachable GPU (A100 on Modal, RTX 4000 Ada on a workstation, local RTX 5060 Ti) | per GPU: proposed on only where both orders give b/a < 1 with every block pair below 1. Report the table to the conductor; the owner decides the default | **done for two GPUs**: A100-SXM4-80GB NaNbO3 0.815/0.812, NbCl5 0.889/0.889; RTX 5060 Ti NaNbO3 0.937/0.940, NbCl5 0.970/0.971, NaCl 0.946/0.935; every pair below 1. RTX 4000 Ada (leto, conductor) NaNbO3 0.938/0.939, NbCl5 0.964/0.964, NaCl 0.934/0.934. **Default on these three device names** (owner, decision 66, 29 October 2026; `DECISION_66_WPRIME_GPUS`) |
