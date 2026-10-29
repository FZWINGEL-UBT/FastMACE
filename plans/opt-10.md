# Optimisation round 10: four small screens from the 29 October idea list

Owner, 29 October 2026: "Try them out in the order you proposed". The conductor's order, from an external idea list
(OPT9 already screened its first item):

1. the gate's backward folded into `linear_2`'s input-gradient GEMM (OPT10-T01);
2. the interaction-0 per-species residual table (`species_residual`, the fp32-profile session) and a readout head
   slice (OPT10-T02);
3. an edge-count-gated larger skin for small cells (OPT10-T03);
4. evict-first cache hints on the streaming kernels' `w` loads and stores (OPT10-T04).

Exact or rounding-class only; general rules only (no per-GPU tables). Nothing becomes default without the owner.

## Tasks

| Task | What | Stop rule | Status |
| --- | --- | --- | --- |
| **OPT10-T01** | Falsifier (`tools/bench/opt10_gate_epilogue.py`), synthetic operands with MH-1's `linear_2` shapes (512 channels; 0e, 1o, 2e, 3o). Chain: `joined_products_input_grads` (the library's), then the gate's elementwise backward compiled by Inductor (`dgated = drows * gate`, `dgate = sum_m drows * gated`). Fused: one Triton kernel per gated segment tiling atoms, one K-loop per component m, the gate tile loaded once, `dgate` summed in registers (no `drows` round trip, no atomics). CUDA graphs of 20 calls, L2 flushed per call, 15 alternating blocks | ≥ 150 µs per interaction on NbCl5 (≥ 300 µs per step, two gated interactions) | **done, passed on the RTX 4000 Ada**, [report](../docs/validation/opt10-T01.md) |
| OPT10-T02 | `species_residual` to a decision (fp32-profile session); readout head slice if cheap | A/B < 0.992 on two cells and ≤ 1.000 on all three; drift unchanged | with the fp32-profile session |
| OPT10-T03 | Skin from the candidate-edge count for small cells | NaCl b/a ≤ 0.990, large cells unchanged | not started |
| OPT10-T04 | `__ldcs` / `__stcs` on `w` in the vendored streaming kernels | subgraph −200 µs on NaNbO3 or NbCl5 | not started |
