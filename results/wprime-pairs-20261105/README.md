# w′ backward: term-pair value table and hoisted piece scale (5 November 2026)

These are ideas 1 and 9 of the H200 ideas plan (`plans/h200-ideas-20261105.md`, local branch
`fastmace/h200-ideas-20261105`). They are built as A/B switches in `fastmace/kernels/oeq_streaming/wprime_backward.py`
at `41e9ad28`. Both stay off.

- **`TABLE_PAIRS`:** TABLE_W's value table `[groups, Q, terms]` stored as `[groups, terms / 2, Q, 2]` (`pair_table`).
  A warp's `float2` load of one term pair then covers 256 contiguous bytes instead of 768 bytes over six lines. The
  values and registers are the same, so the result is bitwise.
- **`HOIST_SCALE`:** HORNER's slope without the piece scale per weight; the edge's sum is multiplied by the scale
  once. Only dL/dr changes, at rounding level.

## Accuracy

`tests/gpu/test_wprime_backward_cuda.py` adds the cases `table-pairs`, `table-pairs-horner` and `hoist-scale`, in
both layouts. On the RTX 5060 Ti, `TABLE_PAIRS` gives bitwise dsh and dL/dr (dx up to the atomics' order), also under
CUDA-graph replay. `HOIST_SCALE`'s dL/dr error against float64 is 2.11e-6 (unhoisted 2.17e-6, ir_mul) and 1.51e-6
(1.54e-6, mul_ir).

The test's earlier `ir_mul-cublas` dsh check already fails on the RTX 5060 Ti without this change: 1 of 24,000
elements at 1.4e-5 relative against rtol 1e-6. That GPU's REDUX default sums dsh in fixed point. These runs set REDUX
off for that case.

## MD-step A/Bs

OMAT-0 medium, float32, SrTiO3 10×10×10, 1 fs, skin 0.5 Å, `tools/ab_rewrites.py`, abba, b with the switch on.

| Switch | RTX 5060 Ti (local) | A100-SXM4-80GB (Modal) | H200 (Modal) |
| --- | --- | --- | --- |
| `TABLE_PAIRS` | 1.0024 / 0.9998 | 1.0058 / 1.0062 | 1.0008 / 1.0003 |
| `HOIST_SCALE` | 0.9985 / 0.9991 | 1.0104 / 0.9988 | 1.0057 / 1.0017 |

Base step: 24.7 ms (RTX 5060 Ti), 10.9 ms (A100), 5.84 ms (H200).

**Verdict:** both are rejected and stay off.
- **`TABLE_PAIRS`:** the plan's L1-wavefront model does not hold. Either the q-major `float2` reads already coalesce
  within the lines they touch, or the w′ is bound by instruction issue rather than L1 data. The A100 loses 0.6 %.
- **`HOIST_SCALE`:** RTX 5060 Ti 0.1 %, inside the noise. It is mixed on the A100 and slower on the H200.

## Runs

Commit `41e9ad28`; Modal profile third (fzwingel54).

| Directory | What |
| --- | --- |
| `runs/rtx5060ti/` | local, `mace_cueq_oeq`, both switches, both orders |
| `runs/a100/` | `tools/modal/hopper_sweep.sh` with `AB=pairs;hoist`, alias = the real name |
| `runs/h200/` | the same with `ALIAS='NVIDIA H200'` |
