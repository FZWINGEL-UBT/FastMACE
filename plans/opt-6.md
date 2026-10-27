# Optimisation round 6: L2-resident per-edge weights

Owner request, 26 October 2026: after reviewing rounds 1–5, try the L2-chunk idea first.

## Idea

The second MH-1 interaction writes its per-edge weights `w = a Wᵀ` (`[E, 1280]`, 268 MB on NbCl5 and 430 MB on
NaNbO3) to DRAM, then reads them in the streaming forward and again in the backward. The backward also writes
`dw` (the same size), which the folded layer's input gradient reads back. Processing receiver-sorted edges in
chunks small enough for L2 keeps `w` and `dw` on chip between their producer and consumer.

- **Earlier attempt:** the research engine tried this on 14 October 2026 with the unpatched OEq, which
  returns a full-size output per call. It was 1.4–2× slower (`docs/rewrites.md`).
- **What differs now:** the streaming kernels are vendored and address edges through CSR rows, so that
  limitation does not apply.
- **Ceiling:** at most 16.0 % (NaNbO3) and 9.1 % (NbCl5) of the research engine's MD step, from the
  shared-weight bound in ResearchModule `studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md`.

## Tasks

| Task | What | Stop rule | Status |
| --- | --- | --- | --- |
| **OPT6-T01** | Falsifier: `tools/bench/opt6_l2_chunked_conv.py`. The real operands of the subgraph (folded last-layer GEMM, streaming forward and backward, Triton input gradient), unchunked against chunked variants in CUDA-graph replays with an L2 flush | forward + backward ≥ 1.0 ms on NaNbO3 and ≥ 0.6 ms on NbCl5 | **done**, [report](../docs/validation/opt6-T01.md): exact. Chunked backward −0.61 / −0.93 ms (NbCl5 / NaNbO3); with the pipelined chunked forward, NbCl5 −1.06 ms. NbCl5 passes, NaNbO3 misses narrowly (0.93), NaCl 0 |
| **OPT6-T02** | Build the backward-only chunking as an opt-in rewrite: one autograd function over the folded last layer and the streaming convolution, with fixed edge chunks, two `dw` slots and the input gradient on a side stream. Then the force gate and same-process A/Bs | full-step b/a ≤ 0.98 on NbCl5 and NaNbO3 in both orders, NaCl neutral | **done**, [report](../docs/validation/opt6-T02.md): record `chunked_conv_backward` (exact, opt-in), gate passed. b/a NbCl5 0.977 / 0.980, NaNbO3 0.964 / 0.965, NaCl 1.006 / 1.011 (slower). Default and a size rule for small cells: owner decision |
| OPT6-T03 | Chunked forward for low-degree cells (a device-side receiver range, or fixed edge ranges with `+=` at boundary receivers), behind a degree rule | NbCl5 a further ≥ 2 % in the full step | not started |
| **OPT6-T04** | Radial tables (owner, 27 October 2026, from an external proposal): per species pair and interaction, the chain from r to the folded layer's input `[h c, c]` and to the edge density as piecewise Chebyshev tables, fitted in float64 at setup. First the accuracy premise and the full-step **stub ceiling**: the radial embedding, both radial networks' hidden layers and the joint first layer replaced by a stand-in of the table op's traffic (an `[E, 65]` write in the forward, its read and a per-edge reduction in the backward) | stub ceiling ≥ 5 % of the force call on NbCl5 and NaNbO3 in a same-process A/B; otherwise stop | **done, passed**, [report](../docs/validation/opt6-T04.md): NumPy re-implementation = MACE to 1e-14; float32 table error 0.13–0.57 of the float32 network's (24 of 24). Stub per force call: NaCl 0.897 / 0.898, NbCl5 **0.928 / 0.928**, NaNbO3 **0.839 / 0.843** (−0.28 / −1.50 / −3.0 ms) |
| OPT6-T05 | Build the table op as an opt-in record: setup fit per interaction and ordered species pair (superset), self-check against the float64 network, one Triton op `r → [h c, c]` and edge density with a recomputing backward; gate, NVE drift, A/B | full-step b/a ≤ 0.95 on NbCl5 and NaNbO3, gate passes | next |
