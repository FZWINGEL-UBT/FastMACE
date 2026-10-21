# OPT3-T02: the cuEquivariance symmetric contraction, measured and screened

Plan: `plans/opt-3.md`, OPT3-T02 (owner: "Go ahead and try all kernel opportunities"). Baseline main `62aa6d4`;
its code is that of `072fa0a`, which OPT-T10 measured. Build rule: only for a ceiling of at least 3 % of the step.

**Result: not built.** The symmetric contraction is 2.5 % (NaCl), 5.4 % (NbCl5) and 3.1 % (NaNbO3) of the step.
Its kernels already run at 72–96 % of memory speed-of-light, close to the bytes their input needs. No screened
option reaches a 3 % ceiling. No GPU was used: the measurement re-joins OPT-T10's ncu exports with the fixed
attribution below.

## 1. Attribution fix

OPT-T10's ncu join (`tools/step_breakdown.py --summarize-ncu`) left these kernels unnamed.
- **Why:** cuEquivariance's `uniform_1d` method picks its kernel variant at run time. The variants are
  `segmented_polynomial_algo1_4`, `_algo1_2`, `_algo2_2`, `_algo1_1`, `_bwd_algo4_2`, `_bwd_algo6_4` and
  `_bwd_algo1_4`. The ncu process often picked another variant than the profiler process, so the launch-order
  name alignment lost them.
- **Fix:** `_base_name` now maps every `segmented_polynomial[_bwd]_algo<N>[_<M>]` to one name per direction
  (`tests/cpu/test_step_breakdown_tool.py::test_cueq_kernel_variants_join_as_one_name`).
- **Result:** re-joined, every ncu kernel aligns: 231/231 (NaCl), 209/209 (NbCl5), 217/217 (NaNbO3).
- **The variants differ between processes, too:** the forward used `algo1_2` and `algo1_4` in one NbCl5
  process, and `algo2_2` in the profiler's. This is cuEquivariance's own choice; its timing effect is part of
  the run-to-run spread.

## 2. Measurement

Shares are from the local profiler (OPT-T10, un-profiled wall times as the denominator). Kernel details come from
local ncu, one eager force call, via `results/opt3-T02-symmetric-contraction/symmetric_contraction_kernels.txt`.

| | NaCl | NbCl5 | NaNbO3 |
| --- | ---: | ---: | ---: |
| Symmetric contraction per force call (profiler) | 0.077 ms | 1.155 ms | 0.588 ms |
| Share of the MD step | **2.5 %** | **5.4 %** | **3.1 %** |
| Kernels per force call | 2 forward (product blocks of both interactions) + 2 backward | same | same |

| NbCl5 kernel (ncu) | µs | DRAM | memory / SM speed-of-light | Occupancy | DRAM bytes |
| --- | ---: | ---: | --- | --- | ---: |
| forward, interaction 1 (`algo1_2`, 64 threads) | 185 | 383 GB/s (85 %) | 87 / 65 % | 74 % | 71 MB |
| forward, interaction 2 (`algo1_4`) | 161 | **422 GB/s (94 %)** | 96 / 15 % | 91 % | 68 MB |
| backward, interaction 2 (`bwd_algo4_2`) | 321 | 369 GB/s (82 %) | 84 / 22 % | 94 % | 119 MB |
| backward, interaction 1 (`bwd_algo4_2`) | 379 | 351 GB/s (78 %) | 79 / 81 % | 63 % | 133 MB |

NaNbO3's kernels run at 338–399 GB/s with memory speed-of-light 77–90 %. NaCl's tiny kernels (36–68 µs, one block
per atom) run at 160–203 GB/s with memory speed-of-light 72–91 %.

**The traffic floor:**
- **Forward:** the input is the `[atoms, 16, 512]` float32 tensor of `linear_2`'s joined output: 64 MB on NbCl5,
  read once. Add the output (4–16 MB) and the per-species weights. The forward kernels move 68–71 MB, so they
  are at the floor.
- **Backward:** it reads the input and the output gradient, and writes the input gradient (64 MB):
  119–133 MB against a floor of about 130 MB.

So the kernels are memory-bound at their traffic floor. A different kernel computing the same thing can save at
most the remaining 5–20 % of their time, under 1 % of the step on NbCl5.

## 3. Screen

| Option | Finding | Ceiling (of the step) |
| --- | --- | --- |
| Other cuEquivariance methods | `cuet.SymmetricContraction` has `uniform_1d` (used; MH-1's segments have the same shape) and `naive` (PyTorch fallback, slower). The variant inside `uniform_1d` is cuEq's own run-time choice, not an option we pass. | ≤ 1 % (the kernels' distance to the floor) |
| OpenEquivariance 0.7.0's symmetric contraction | It exists (`openequivariance/_torch/symmetric_contraction`): e3nn-style contractions in the `mul_ir` layout, built on a grouped GEMM per element (`libtorch_tp_jit.group_gemm`). It cannot read less than the input either, and needs layout transposes around it (the model is `ir_mul`). | ≤ 1 %, likely negative |
| Fusing neighbours: the input materialisation | The input is the joined output of `linear_2`'s four per-l products (`segment_gemm`). A fused contraction needs all 16 components of an atom, from four separate products, in one tile: no single GEMM epilogue holds them. Even removing the input's write and read entirely saves about half the forward and backward traffic. | ≈ 2.7 % NbCl5 at best (half the share), 1.2–1.5 % elsewhere; not buildable as an epilogue |
| Fusing neighbours: the output side | The contraction's outputs (4–16 MB) feed `products.*.linear`; `products.0.linear` is already folded into its consumers by OPT-T08 C3. | < 0.5 % |
| Skipping zero paths exactly | OPT-T04: the coupling tensors (`U_matrix_*`) are 94–99.6 % zero, which cuEq's kernel already exploits. The learned per-species weights have no zero blocks. | 0 |
| Reassociation as in the first interaction | The first interaction's reassociation works because its inputs are species-only. The contraction's input depends on the geometry (messages of every edge), so there is nothing to precompute per species. | not applicable |
| Lower precision for the input | Changes accuracy, like `w` in the 14 October screen. Not general; excluded. | – |

**Verdict:** no option shows a ceiling of 3 % on any structure, and the plan's build rule is not met, so nothing
was built. Recorded in `docs/rewrites.md` as screened. The attribution fix in `tools/step_breakdown.py` stays,
so later breakdowns name these kernels.
