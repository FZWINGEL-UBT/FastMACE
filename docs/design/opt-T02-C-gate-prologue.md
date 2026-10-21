# OPT-T02 stage C: the gate inside `linear_2`'s segment GEMM

Plan: `plans/opt-1.md` OPT-T02, stage C, after B (`segment_gemm`, main `20118db`). Inputs:
`docs/design/opt-T02-BC-go-no-go.md` and `docs/validation/opt-T02b-segment-gemm.md`.

## 1. What is left to take

With A and B on, each gated segment of a `gated_rows` chain still goes through one elementwise kernel
before `linear_2`. The kernel reads the residual product's rows `[atoms · (2l + 1), mul]` and the atom's
gates `[atoms, mul]`, then writes the gated rows. Only `linear_2` reads the gated rows: MD needs no weight
gradients, and the gate's backward needs only the ungated rows and the gates.

murphy NbCl5 after A (go/no-go §1): the per-l gate forward kernels take **about 1.4 ms** per force call,
counted twice because Inductor recomputes them for the backward. B's rule sends only the l = 2 and l = 3
products to Triton (9,720 and 13,608 rows); together they are 12 of 15 gated components, so about
**1.1 ms** of that is in reach. NaCl has no product on Triton, so there is nothing to take there.

## 2. Design: the gate in the GEMM's prologue

`fastmace::joined_products` gets an optional gate per product. On the Triton path, the kernel loads the
A tile `[BLOCK_M, BLOCK_K]` of ungated rows and the matching gate tile, gathered by atom (`row // dim`).
It multiplies the two in registers before `tl.dot`. The gate tile is 1/dim of the A tile's distinct data
(rows of one atom repeat it), so it comes mostly from L1 and L2.

- **Why not the epilogue of the residual GEMM.** The gate's backward needs the ungated rows, so that GEMM
  must write them anyway. Gating in its epilogue would add a second output, not remove one. The prologue
  removes the gated tensor entirely.
- **Rounding.** The product `x · gate` is formed in float32 exactly as the stock kernel forms it, then
  rounded once, and goes into the same IEEE `tl.dot`. So the result equals B's within the GEMM's own
  summation-order rounding. Class `rounding`, deterministic (no atomics).
- **Backward.** Unchanged from B. The custom input-gradient op returns `dL/d(gated rows)` per product.
  The gate's backward (`dx = g · gate`, `dgate = Σ_m g · x`) stays in PyTorch, where Inductor fuses it
  as it does now. Stage D (the reduction in the epilogue) stays rejected: no atomics.
- **The cuBLAS path** (products below the rule, CPU, float64): the gate is applied in PyTorch as now, so
  nothing changes there. NaCl and every product below the rule are unaffected.
- **Record:** `gate_prologue`, `requires=("segment_gemm",)`, opt-in (`never_automatic`) until its A/B on
  both GPUs. **Self-check (§6.4):** the joined products with the gates against B's joined products of the
  gated rows, values and input gradients (x and gates), on enough atoms that the rule picks Triton.

## 3. Estimate and price

| | murphy NbCl5 | NaCl |
| --- | --- | --- |
| Removable | ≈ 1.1 ms (gate forward on the Triton products, and its recompute) | 0 |
| Added | the gate tile loads in two GEMMs per interaction, a few % of ≈ 1.4 ms each, ≈ 0.1–0.2 ms | 0 |
| Net | **≈ 0.6–0.9 ms (2–3 % of 27 ms)** if Inductor drops the recompute; ≈ 0.3 ms (1 %) if not | neutral |

## 4. Decision rule

- **Go to build:** yes. It reuses B's kernel and needs one optional input and a prologue multiply.
- **Adopt (propose as default):** NbCl5 b/a ≤ 0.985 in both orders on **both** GPUs, NaCl within the
  noise floor, and the force gate passes.
- **Reject:** below that, or any NaCl loss. Record it in `docs/rewrites.md` either way.

## 5. Result (21 October 2026): rejected

Built opt-in on `fastmace/opt-T02c` (`35e51a6`, launchers `3dc0ca5`): the kernel's `HAS_GATE` prologue, the
custom op `fastmace::gated_joined_products` with the gate's backward in PyTorch, and the record
`gate_prologue`. CPU tests passed, and on the RTX 5060 Ti `tests/gpu/test_segment_gemm_cuda.py` passed
(3 tests): values bitwise equal to `segment_gemm`'s, gradients of rows and gates within 1e-5.

Same-process A/B on the local RTX 5060 Ti (sm_120, driver 610.88, `~/environments/mace_cueq_oeq`, torch
2.14.1+cu130, Triton 3.8.0), pinned copy `~/fastmace-runs/src-3dc0ca5`. a = `gated_rows` + `segment_gemm`,
b = a + `gate_prologue`; MH-1 hybrid float32 NVT 300 K, 3×3×3 cells, 12 blocks of 300 steps. Idle checks
passed, 21:55–22:08 UTC 20 October 2026:

| b/a | a-b (ABAB) | b-a (ABBA) |
| --- | --- | --- |
| NaCl | 0.9979 | 0.9970 |
| NbCl5 | **1.0196** (23.58 → 24.04 ms) | **1.0157** (23.38 → 23.75 ms) |

NbCl5 is 1.6–2.0 % slower. The gather of the gate tile by atom and the multiply in the K loop cost the
two Triton GEMMs more than the elementwise gate kernel they replace, and the estimate in §3 did not hold.
NaCl is unchanged, as expected, since no product there reaches the shape rule.

**Decision (conductor, per §4):** rejected. The rule needs both GPUs, and sm_120 alone fails it, so
murphy was not measured. The code stays unmerged on its branch.
