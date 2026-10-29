# OPT2-T05: split-K for small-cell atom products

Plan task OPT2-T05 (`plans/opt-2.md`). Design: `docs/design/opt2-T05-split-k.md`. Branch
`fastmace/opt2-T05`. Record `split_k`, opt-in (`never_automatic`).

**Result:** the candidate is rejected on sm_120. The gates pass, but NaCl does not get faster in
the step (b/a 1.0028 and 1.0049). The record stays opt-in, so it does not change any plan.

## What was built

- **The rule** `tile_rule.split_k_parts(m, n, k, sms)`, from the shapes and the SM count only. A
  product splits along K into a power of two of parts when three things hold:
  - its 64 × 64 output tiles do not fill one wave;
  - K ≤ N;
  - every part keeps at least 128 of K.
  The parts aim for one wave. On NaCl only the l = 0 products `[216, 512] @ [512, 512]` split,
  into 2 parts, forward and input gradient. No NbCl5 or NaNbO3 product splits.
- **The product:** `split_k_mm` is one batched cuBLAS product of `[parts, M, K/parts]` by
  `[parts, K/parts, N]`, then a sum. It makes no copy and is deterministic. `SplitKMatmul` splits
  the forward and the input gradient by their own shapes.
- **The record** `split_k`: rounding, deterministic, requires `block_linear`, conflicts with
  `grouped_launch`. It acts on float32 CUDA only.

## Checks

| Check | Result |
| --- | --- |
| CPU (`tests/cpu/test_rewrites_split_k.py`, 13 tests) | the rule's table and limits; `split_k_mm` and its input gradient equal `a @ b` within 1e-12 in float64 (strided rows included); the record's applicability, refusals (no `block_linear`; with `grouped_launch`) and self-check; with every product split, the tiny MH-1-type model's energy within 1e-12 and forces within 1e-10 of the unsplit model in float64 |
| CPU suite, murphy, `-n 2` (commit `c9f0dec`) | 2674 passed, 182 skipped; ruff clean |
| Micro-benchmark (`tools/split_k_bench.py`, `bench-local.json`) | `[216, 512] @ [512, 512]`: cuBLAS 20.8 µs, 2 parts 18.5 µs, 4 parts 19.4 µs. `[216, 2048] @ [2048, 512]`, which cuBLAS splits itself: 45.8 µs, split 56.9–62.1 µs. The other shapes lose too. This led to the rule's K ≤ N condition and its one-wave target (commit `2c6db72`) |
| Force, stress and energy gate (`tools/force_stress_gate.py --engine fastmace --rewrites split_k=on`; plan shows `split_k = on`) | **passed**. Ratios to the float32 calculator: NaCl energy 0.14, forces 1.13, stress 0.36; NbCl5 energy 0.11, forces 1.02, stress 0.78 (limit 2) |

## Speed

**Setup:** a same-process A/B (`tools/ab_rewrites.py`), MH-1 hybrid float32 NVT at 300 K, 3×3×3, 300
warm-up steps, 12 alternating blocks of 300 steps. Side a has the OPT-T08 records on, the round's
presumed defaults (`species_tables`, `fold_readout_tail`, `fold_product_linear`). Side b adds
`split_k`. All blocks are valid, with nothing compiled while timed.

| Structure | Order, pattern | a (ms/step) | b (ms/step) | b/a |
| --- | --- | ---: | ---: | ---: |
| NaCl (216 atoms) | a-b, ABAB | 3.1089 | 3.1175 | **1.0028** |
| NaCl | b-a, ABBA | 3.1034 | 3.1186 | **1.0049** |
| NbCl5 (1944 atoms) | a-b, ABAB | 21.3973 | 21.4760 | 1.0037 |

- **Noise floors:** 1.1 % on NaCl and 0.4 % on NbCl5 for a 12-block same-process A/B
  (`docs/validation/2-T12-local-noise-floor.md`).
- **NaCl:** a little slower in both orders, within noise. The product's 2 µs saving does not
  survive in the step. The batched product and the separate sum kernel replace one cuBLAS kernel,
  and in the force graph the extra launch and the partial sums' traffic cost about as much as
  the split saves.
- **NbCl5:** within noise, as expected. The rule never splits there, so both sides run the same
  products.

**Machine and run:**
- local RTX 5060 Ti (sm_120, 36 SMs), driver 610.88, WSL2;
- `~/environments/mace_cueq_oeq` (torch 2.14.1+cu130, Triton 3.8.0, cuEquivariance 0.12.0,
  OpenEquivariance 0.7.0);
- pinned source copy `~/fastmace-runs/src-2c6db72`;
- each job under its own `tools/ci/local_gpu.py` reservation of at most 15 minutes, 21 October
  2026, 14:10–14:37 UTC.

The results are in `results/opt2-T05-local-20261021/`: the A/B and benchmark JSONs and the gates'
`metadata.json`.

## Not done

- **The murphy A/B.** Locally the candidate does not gain in the step. Its ceiling is the
  micro-benchmark's 2 µs per product, about 0.3 % of the NaCl step. Spending murphy time before the
  Thursday cut-off is the conductor's call.
- **A Triton split-K kernel that reduces in its epilogue,** which would save the separate sum
  kernel. With a ceiling of about 0.3 % of the NaCl step it is not worth building.
