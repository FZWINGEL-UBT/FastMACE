# Slab-sorted edges and species-grouped blocks on the H200 (8 November 2026)

Two switches that make the SPLINE_W forward's table reads hit L1 more often. OMAT-0 medium, Benchmark_loup
(SrTiO3), Modal profile `sixth`, NVIDIA H200 (driver 610.57.04). Background:
[omat0-breakthroughs-assessment-20261108](../omat0-breakthroughs-assessment-20261108/README.md) measured the forward
at about 45 % table traffic and modelled L1 hits 0.06 -> 0.54 (slab sort) -> 0.70 (species blocks).

## SORT_BY_LENGTH (`fastmace.neighbors.skin`)

With `sort_receivers`, each receiver's candidates (within its sender group) are ordered by length bin (4096 bins
over the reach) at the skin-list build. The compaction between rebuilds is stable, so the order holds until the
next build. Summation order changes: a rounding switch.

`h200/`, source 768bb729, b = on, against the H200 defaults with `JOINED_LAYOUT`:

| run | a (ms) | b (ms) | b/a |
| --- | --- | --- | --- |
| ab | 3.585 | 3.514 | 0.9807 |
| ba | | | 0.9791 |

Gate (`tools/force_stress_gate.py --production-constant`, `h200/gate-sorted/`): passes; forces 0.83 / 0.92 / 0.97,
stress 0.48 / 0.94 / 1.07. Adopted: `SORT_BY_LENGTH_GPUS = ("NVIDIA H200",)`.

## BLOCK_ORDER (`fastmace.kernels.oeq_streaming.spline_forward`)

The forward's owner rows walk the receivers through an int32 permutation grouped by species (stable argsort,
built once by the force field and bound to each interaction's forward). The arithmetic per receiver is unchanged:
bitwise.

`h200-order/`, source 2a8845d6, b = on, on top of `SORT_BY_LENGTH`:

| run | a (ms) | b (ms) | b/a |
| --- | --- | --- | --- |
| ab | 3.5121 | 3.4254 | 0.9753 |
| ba | 3.5159 | 3.4189 | 0.9724 |

Check (`tools/spline_w_check.py`, `check.json`): the order-bound forward agrees with the reference to 2.56e-6
against the same build's 3.34e-6 (the w' atomics). Adopted: `BLOCK_ORDER_GPUS = ("NVIDIA H200",)`.

## Lab (`h200-lab/`, `tools/spline_table_bound.py` on the sorted defaults)

| forward variant | warm µs | x real |
| --- | --- | --- |
| real | 717.3 | 1.000 |
| one slab (bound) | 432.5 | 0.603 |
| slab-sorted (exact sort) | 680.6 | 0.949 |
| species blocks | 623.1 | 0.869 (bitwise) |
| slab-sorted + species blocks | 601.6 | 0.839 |

The order costs 89 µs per call if made eagerly; the force field makes it once per species set, so it is free per
step. w' (1084 µs) is not table-bound (one slab 0.978).

## A100 (Alex pack 7, job 4965032, `a100/pack7/`)

Each switch on its own and both together, against the A100 defaults of 59c1034e. OMAT-0 medium, SrTiO3 10×10×10,
`alex_ab_pack`, both build orders:

| b | ab: a → b (ms) | b/a | ba: b/a |
| --- | --- | --- | --- |
| `SORT_BY_LENGTH` | 7.3593 → 7.3034 | 0.9924 | 0.9936 |
| `BLOCK_ORDER` | 7.3734 → 7.3388 | 0.9953 | 0.9945 |
| both | 7.3911 → 7.3323 | 0.9920 | 0.9904 |

`BLOCK_ORDER` is bitwise and faster in both orders, so it is adopted under decision 66:
`BLOCK_ORDER_GPUS = ("NVIDIA H200", "NVIDIA A100-SXM4-80GB")`.

`SORT_BY_LENGTH` changes rounding, so it needed the A100 force/stress gate before adoption.

**A100 gate** (`a100/gate-sorted/`, Alex job 4967026, 9 November). The run used `tools/force_stress_gate.py` with
production running `SORT_BY_LENGTH` and `BLOCK_ORDER`, the H200 gate's structures and seed 20261015, source
59c1034e. All three structures pass. Ratios are against the float32 yardstick, limit 3:

| | SrTiO3 | NaCl | NbCl5 |
| --- | --- | --- | --- |
| forces | 0.92 | 0.93 | 0.96 |
| stress | 0.64 | 1.02 | 1.15 |

Energy is under its 0.01 eV floor. Adopted under decision 66: `SORT_BY_LENGTH_GPUS` now includes the A100 family.
The A100 step with both switches on is 0.9920 / 0.9904 (7.39 → 7.33 ms).
