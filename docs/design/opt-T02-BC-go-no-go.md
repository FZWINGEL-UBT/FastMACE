# OPT-T02 stages B and C: go/no-go after stage A

Input: the 2-T20a main pass on murphy (RTX 4000 Ada, sm_89) with `gated_rows` on (`results/opt-T02-trace/murphy/`,
20 October 2026 19:27–19:32 UTC, GPU empty, load 0.3–1.2, commit `8cc9860` = main `036c0fa` + the step-breakdown
tool), against the same pass without it (`results/2-T20a-step-breakdown-20261020/murphy/`).
Design: `docs/design/opt-T02-fusion.md`. B and C share one Triton GEMM: C, the gate in the epilogue, needs B's
kernel, because cuBLAS's epilogues cannot apply a sigmoid gate taken from another tensor.

## 1. What stage A left (murphy, µs per force call)

| | NaCl before A | NaCl after A | NbCl5 before A | NbCl5 after A |
| --- | ---: | ---: | ---: | ---: |
| Step (GPU span) | 3,681 | 3,537 | 31,580 | 28,682 |
| Atom-level GEMMs | 1,562 | 1,537 | 11,192 | 11,238 |
| Atom-level elementwise | 442 | 349 | 8,948 | 5,867 |
| – forward copies (clone/cat of per-l blocks) | 136 | 80 | 3,006 | 1,032 |
| – backward slice-gradient scatters | 119 | 57 | 2,466 | 1,442 |
| – gate forward (now per-l kernels) | 41 | 58 | 901 | 1,396 |
| – gate backward | 68 | 14 | 1,917 | 283 |
| – density division (fused with slicing) | 39 | 37 | 390 | 440 |
| – other (incl. the gate's per-atom `mul_sum` reductions) | 41 | 102 | 268 | 1,274 |
| Kernels per force call | 286 | 272 | 294 | 278 |

A removed 3.1 ms of NbCl5's elementwise time and 2.9 ms of its step. Its gate kernels now read the rows
directly, so some gate work moved between the buckets.

**What remains for B on NbCl5:**

- `cat_slice_view_31`, 771 µs: the per-l outputs of `linear_2` joined into the `[N, 16, 512]` input of the
  cuEq symmetric contraction.
- Its backward, `add_slice_slice_backward_view_11`, 662 µs.
- The product block's slice copies and their backward (`add_clone_slice_slice_backward_*`), 569 µs, plus
  250 µs more.
- The density division fused with slicing, 268 µs.

That is about **2.5 ms**. For C, the per-l gate forward kernels, about 1.37 ms (they appear twice: forward,
and recomputed for the backward).

## 2. The price: Triton IEEE GEMMs against cuBLAS (OPT-T01 bench, murphy)

| Atom-level GEMMs per force call | cuBLAS | Triton IEEE (fastest fixed config) | Difference |
| --- | ---: | ---: | ---: |
| NbCl5 | 10,433 µs | 11,194 µs | +761 µs (+7 %) |
| NaCl | 1,856 µs | 2,561 µs | +705 µs (+38 %) |

The bench used contiguous operands. B's kernel reads gathered rows and writes strided rows. Each row is
512 contiguous floats, so the loads stay coalesced, but the cost is unmeasured.

## 3. Net estimate

| Crystal (murphy) | Removable (B; C) | Triton GEMM cost | Net, if fully realised | Realistic (70 % of removable) |
| --- | --- | --- | --- | --- |
| NbCl5 | 2.5 ms; 1.0 ms | +0.8 ms | +2.7 ms (9 %) | +1.6 ms (5–6 %) |
| NaCl | 0.15 ms; 0.04 ms | +0.7 ms | **−0.5 ms (loss)** | loss |

So B and C pay only on large cells, with cuBLAS kept on small ones. A shape rule decides, shared with
OPT-T03 as agreed:

- Triton where a GEMM's M ≥ 2 · SMs · 64 rows, with the SM count read from the device, not tabulated per
  GPU. On murphy's 48 SMs that is 6,144 rows: NbCl5's l = 2 and 3 products (9,720 and 13,608 rows) qualify,
  NaCl's largest (1,512 rows) does not.
- cuBLAS elsewhere.

With that rule NaCl is unchanged. NbCl5's gain shrinks to the large products' share of the removable copies,
roughly 1.0–1.5 ms (3–5 %).

## 4. Recommendation

**Conditional go: a one-day prototype first, with a stop rule.**

1. **Micro-bench.** A Triton segment GEMM (gathered A rows, strided C rows, the density scale in the
   prologue), forward and input gradient, on NbCl5's `linear_2` and residual shapes. It is untimed in the
   model, about 10 minutes on murphy.
   - **Stop** if it is slower than cuBLAS + its share of the removed copies; then B is not built and C
     cannot be built.
2. **If the bench passes:** B in the model behind its own record (`segment_gemm`, opt-in), with the shape
   rule.
   - Then the same checks as A: CPU float64 bitwise is no longer possible (rounding class); §6.4
     self-check; reproducible-mode determinism; the gate; A/B on both GPUs.
3. **C on top of B,** if B lands.

**No-go alternative:** leave B and C and move to OPT-T03 (small-M grids, NaCl's l = 0 GEMMs at 6–10
TFLOPS) and OPT-T04 (exact skips). Those help the small cells, which B and C cannot.

My preference is the prototype (step 1) first. It is cheap and settles the main unknown, the gathered-row
GEMM's speed.
