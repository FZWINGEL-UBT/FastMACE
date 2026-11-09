# Tensor-core work beside the edge walks: bound (2026-11-08)

**Question.** Could the tensor-core GEMMs run beside the CUDA-core (SIMT) edge walks? In production they run one
after another: the step is one CUDA graph on one stream.

**Method.** I took the round-6 H200 breakdown
([`../h200-fastpath-20261108/round6/breakdown/`](../h200-fastpath-20261108/round6/breakdown/), omat0-medium,
SrTiO3, 5,000 atoms, 98 launches per force call) and followed each kernel's dataflow. I looked for kernels that need
nothing from the two big edge walks:

- the interaction-2 spline forward (816.6 µs);
- w′ (1,089.2 µs), which writes the sender gradient `dx` that interaction 1's backward needs.

Only these kernels qualify:

| Beside | Kernel (sequence) | Unit | µs |
| --- | --- | --- | ---: |
| spline forward | readout-1 GEMV (37) | SIMT | 2.6 |
| spline forward | skip GEMM `fm_tf32x3_remap_select` (38) | tensor cores | 7.3 |
| w′ | w′ output zeroing (72; it must precede w′ but can move earlier) | SIMT | 3.2 |
| w′ | readout-1 backward SGEMM (74) | SIMT | 3.7 |
| w′ | skip backward: elementwise (75) and `fm_tf32x3_rc` (76) | SIMT and tensor cores | 6.6 |

All the other GEMMs sit on the critical chain:

- `linear_up` feeds the forward edge walk;
- `linear_div` and the node backward feed w′;
- interaction 1's species GEMMs need w′'s `dx`.

**Bound** ([`bound.py`](bound.py), [`bound.json`](bound.json)). Assume perfect hiding: every listed kernel runs
fully in the edge walks' shadow and also loses its graph launch gap. The gap is 2.43 µs per launch, an over-estimate
(stage time minus kernel time, divided by 98 launches). That gives:

- 23.4 µs of kernel time plus 14.6 µs of launch gaps, **38 µs = 1.07 % of the 3.56 ms stage**;
- of that, tensor-core time is only 12 µs (the skip GEMM and its backward).

**Verdict: not worth building.**

- Even the ceiling falls short of the lab bar, which is twice the 1 % stop rule (about 68 µs).
- The real gain would be well below the ceiling:
  - Both edge walks run 6 blocks × 128 threads × 80 registers per SM (61,440 of 65,536 registers), and a TF32×3
    GEMM block needs 158 × 128 = 20,224. So the GEMMs could only run in the walks' ragged last wave.
  - Each fork and join in the graph adds an event dependency.
  - Earlier two-stream measurements were 0.77–0.98× ([prec-tc-overlap](../../docs/optimisation-ledger.md)) and 0.68–1.11× (lin-concurrent-branches).
