# OPT2-T03: the folded input gradient's tiles by a waves rule

Optimisation round 2 (`plans/opt-2.md`), candidate 3 of OPT-T09 (`docs/design/opt-T09-step-breakdown.md`): the
folded radial last layer's Triton input gradient (`fastmace/kernels/triton/skinny_linear.py`) ran one fixed tile
configuration at about 25 % occupancy. Date: 21 October 2026.

## What changed

- `skinny_linear.tile_choice(rows, sms, rule)`. With `rule="waves"`, it picks 128-row tiles with 8 warps where the
  64-row grid has 2 to 8 waves over the SMs, and the fixed 64 × 32 tiles (4 warps, 3 stages) elsewhere. The rule
  uses only the edge count and the SM count; no per-GPU or per-shape table.
- **Exact.** BLOCK_K stays 32, so every row is reduced in the same order and the results are bitwise the same.
  The kernel screen found every BLOCK_K = 32 configuration bitwise equal to the fixed one, and every other
  BLOCK_K within FP32 rounding only.
- **Record `folded_backward_tiles`** (MH-1 family): exact, deterministic, default off, requires
  `fused_folded_backward`; the planner's rule is `never_automatic` (opt-in, the owner decides defaults).
- The generic family (MP-0, the density family) keeps the fixed tiles. Its `folded_linear` call site was left
  alone, as agreed with OPT2-T02.

## Where the rule acts

On the RTX 5060 Ti (36 SMs), with MH-1 3×3×3 cells and their padded edges:

| Structure | Padded edges | 64-row tiles / SMs | Tiles chosen |
| --- | --- | --- | --- |
| NaCl | 8,192 | 3.6 waves | 128 rows, 8 warps |
| NbCl5 | 56,320 | 24 waves | fixed |
| NaNbO3 | 82,688 | 36 waves | fixed |

On murphy's RTX 4000 Ada (48 SMs) NaCl is 2.7 waves, inside the rule too. Only small cells change; on NbCl5 and
NaNbO3 the kernel, and so the step, is the same program as before.

## Kernel screen (local RTX 5060 Ti, kernel alone)

Interleaved, 12 rounds, median, against the fixed tiles (`local-20261021/kernel-screen-interleaved.json`):

| Edges | 64-row waves | 128 rows, 8 warps, 3 stages |
| --- | --- | --- |
| 2,048 | 0.9 | 1.67× (slower) |
| 4,096 | 1.8 | 1.01× |
| 8,203 | 3.6 | 0.71× |
| 16,384 | 7.1 | 0.90× |
| 32,768 | 14.2 | 1.03× |
| 56,250 | 24.4 | 1.00× |
| 82,688 | 35.9 | 0.98× |

Above about 8 waves the fixed tiles are as fast or faster, and below 2 waves the larger tiles leave SMs idle. That
gives the rule's bounds.

## ncu, before and after (warm launch)

`local-20261021/ncu-table.md`, from `results/opt2-T03-folded-tiles/ncu_kernel.py`:

| Structure | Tiles | Duration (µs) | DRAM % | Occupancy achieved / theoretical % | Registers |
| --- | --- | --- | --- | --- | --- |
| NaCl | fixed | 170.2 | 62.7 | 21.7 / 25.0 | 144 |
| NaCl | waves | 134.8 | 77.1 | 28.6 / 33.3 | 122 |
| NbCl5 | fixed / waves (same tiles) | 841.5 / 823.6 | 86.5 / 87.4 | 24.3 / 24.3 | 144 |
| NaNbO3 | fixed / waves (same tiles) | 1200.3 / 1204.4 | 88.4 / 87.9 | 24.6 / 24.7 | 144 |

On NaCl the kernel is 0.79× with more DRAM throughput and higher occupancy. The NbCl5 and NaNbO3 rows run the same
configuration twice; they show the measurement's own scatter.

## Accuracy

- CPU: 15 tests (`tests/cpu/test_folded_backward_tiles.py`): the rule's boundaries, the refusal without
  `fused_folded_backward`, the planner's `never_automatic`, and the record on a tiny MH-1 model with energy and
  forces bitwise unchanged. The surrounding test files (registry, planner, report, MH-1 rewrites, fusion, config,
  contracts) pass at `-n 2`.
- CUDA: `tests/gpu/test_folded_tiles_cuda.py`, 6 passed. The waves tiles equal the fixed tiles bit for bit at
  2,048, 4,096, 8,203, 16,384 and 56,250 edges, and so does the gradient through `folded_linear`; both are within
  1e-5 relative of the float64 product.
- An exact candidate needs no force gate; its results are the fixed tiles' bit for bit.

## Step A/B (local RTX 5060 Ti)

- Machine: the local PC, NVIDIA GeForce RTX 5060 Ti, driver 610.88, WSL2, desktop idle (utilisation 0 % in
  both views before each job).
- Software: `mace_cueq_oeq` (hybrid), torch 2.14.1+cu130, MH-1 (`mace-mh-1.model`) float32, NaCl 3×3×3, NVT 300 K.
- FastMACE commit `27e7718` (a pinned copy); no baseline involved.
- Protocol: `tools/ab_rewrites.py`, same process, 12 blocks of 300 steps after 300 warm-up steps, both build
  orders, one order per reservation.
- Sides: a = the round-2 baseline (`species_tables`, `fold_readout_tail` and `fold_product_linear` on); b = a +
  `folded_backward_tiles`.

| Order | a (ms/step, median) | b (ms/step, median) | b / a | Pairs with b faster |
| --- | --- | --- | --- | --- |
| ab (abab) | 3.109 | 3.056 | 0.983 | 12 / 12 |
| ba (abba) | 3.103 | 3.044 | 0.981 | 12 / 12 |

**Result: NaCl −1.7 to −1.9 %**, beyond the local NaCl noise floor (1.1 %), every pair in both orders. Within the
task's ceiling (2–4 %, NaCl). NbCl5 and NaNbO3 were not A/B-timed, because the rule leaves their tiles unchanged and
so runs the identical kernel.

Records: `results/opt2-T03-folded-tiles/local-20261021/` (`nacl-ab.json`, `nacl-ba.json`, the run logs, the ncu
CSV and table, and both kernel screens). The kernel screens ran from an unpinned copy of the branch before the
record existed; they call the kernel directly, whose code is unchanged.

## Open

- A murphy A/B (Ada, 48 SMs) when murphy is reachable and free, before Thursday 22 October 16:00 UTC; NaCl is
  2.7 waves there.
- The owner decides the default. Recommendation: on (`auto`) with `fused_folded_backward`. It is exact, and it
  changes only small cells.
- If OPT2-T01 (fused radial last layer and streaming convolution) lands, this kernel disappears, and the record
  with it.
