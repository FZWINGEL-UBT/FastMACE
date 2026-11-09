# Interaction 0's symmetric contraction as one generated kernel: bound (9 November 2026)

**Question.** Codex report v3, proposal 5 (`FUSED_LAYER0_SC`): replace interaction 0's cuEquivariance `uniform_1d`
symmetric contraction (OMAT-0 medium) with a generated channel-diagonal NVRTC kernel (one block per atom,
`ch = threadIdx.x`), and fuse the 1 / (density + 1) division that feeds it. Is the ceiling above the 68 µs lab bar
(2× the 1 % triage floor of a 3.42 ms H200 step)?

**Verdict: LAB.** The traffic floor is not the limit; cuEq's kernels are FP32-issue bound on the H200. They also
execute more FP32 instructions than the polynomial needs: 1.33× forward and 1.56× backward against a straight-line
evaluation with shared monomials. Counting the instructions a generated kernel needs gives these estimates:

- paper ceiling, at full FP32 rate: about **119–123 µs**;
- at cuEq's own 70 % FMA-pipe use: about **95–99 µs** (2.8 % of the step).

Both clear 68 µs. Without fewer instructions than cuEq, only the two division kernels go (about 48 µs), which is
below the bar. The lab must therefore first show the instruction cut (see "Next step").

## The bucket (H200 round-6 breakdown, Benchmark_loup SrTiO3 10×10×10, 5,000 atoms)

| Kernel | µs | Interaction |
| --- | ---: | --- |
| `triton_poi_fused_add_div_view_21` (x = m / (D + 1), [5000, 2048]) | 18.1 | 0 |
| `segmented_polynomial_algo1_4` (forward) | 40.1 | 0 |
| `segmented_polynomial_bwd_algo1_4` (backward, 96 registers, 31 % occupancy) | 87.4 | 0 |
| `triton_red_fused_add_div_mul_neg_species_mm_rows_grad_sum_view_11` (dD, and dm in the species GEMM's [5000, 16, 128] layout) | 29.5 | 0 |
| **interaction-0 bucket** | **175.1** | |

The brief listed `triton_red_fused_add_div_mul_neg_sum_2` (18.0 µs) as the division's backward. In
`kernel_sequence.csv` it runs right after interaction 1's contraction backward (#61, weights [89, 1664],
dy [5000, 128]): it is interaction 1's density gradient (its division sits in `linear_div`'s epilogue). Interaction
0's division backward is `…species_mm_rows_grad_sum_view_11` (#82, right after interaction 0's backward #81),
29.5 µs.

## The bound ([`tools/sc0_bound.py`](../../tools/sc0_bound.py))

The tool times, on interaction 0's real operands:

- `products.0.symmetric_contractions` of the hybrid MACE calculator, `uniform_1d`;
- the checkpoint's weights [89, 3712];
- SrTiO3's species indices;
- x [5000, 2048] captured from an eager call on the rattled 10×10×10 cell.

It compares three kernel sets (kernel time from torch.profiler; L2 flushed before the forward pair and before the
backward pair):

- `proxy`: the division (Inductor), cuEq's forward and backward (the module's own `SegmentedPolynomial`), and the
  division's backward (Inductor);
- `floor`: two Triton kernels that only do a fused kernel's reads and writes: forward reads m and D and writes y;
  backward reads m, D and dy and writes dm and dD;
- `floor+w`: the same, plus each atom's species weight row.

D is drawn in [0, 2): its values do not change the timings.

H200 on Modal (profile `sixth`, source `bf670317`, 15 alternating rounds; [h200/](h200/)):

| Set | µs (median) | Spread | Kernels (µs) |
| --- | ---: | --- | --- |
| proxy | **179.4** | 178.8–180.3 | division 21.9, forward 40.2, backward 87.2, division backward 30.2 |
| floor | **41.8** | 41.7–42.0 | forward 15.9, backward 25.9 (143 MB, 3.43 TB/s) |
| floor+w | 44.8 | 44.4–45.3 | 17.1, 27.9 |
| copy [5000, 2048] | 21.6 | | 3.80 TB/s |

The proxy matches production: 40.2 / 87.2 against 40.1 / 87.4 µs for cuEq, and the division kernels within 4 µs.
**Traffic alone would allow 137.5 µs, but it is not the limit.**

### What bounds cuEq's kernels: Nsight Compute, one launch each (same job)

| Kernel | µs | FFMA / FMUL (thread inst.) | FP32 per atom-channel | FMA pipe | DRAM | Main stall |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| forward (`algo1_2` in this process) | 40.0 | 307 M / 577 M | 1,381 | 70 % | 23 % | not selected (5.3) |
| backward (`bwd_algo1_4`, 96 regs) | 85.9 | 744 M / 1,129 M | 2,926 | 70 % | 18 % | not selected (2.7) |
| division | 19.4 | | | 7 % | 63 % | long scoreboard |
| division backward | 28.9 | | | 18 % | 70 % | long scoreboard |
| `floor` forward / backward | 13.0 / 22.8 | | | 4 / 9 % | 70 / 66 % | long scoreboard |

The contraction kernels are issue bound: 72–74 % of peak instruction issue, and their main stall is "not selected"
(eligible warps waiting for the issue slot). cuEq's FP32 work alone, at the H200's FP32 rate (132 SMs × 128 lanes ×
1.96 GHz = 33.1 T inst/s), is 26.7 + 56.6 = 83.3 µs. A generated kernel with cuEq's arithmetic would only save the
two division kernels.

### What the polynomial needs ([term_count.txt](term_count.txt), [tools/sc_term_count.py](tools/sc_term_count.py))

The cuEq descriptor (degree 1–3, 16 input and 4 output components per channel, 29 weights) lists 4 + 86 + 1,949 paths.
They collapse to:

- **480** distinct (monomial, output, weight) terms: cuEq's FFMA count per channel is exactly 480;
- **410** distinct monomials (51 of degree 2 and 355 of degree 3), built from 478 products;
- 61 (weight, output) groups.

cuEq executes 901 FMULs per channel. That is about two per term: it rebuilds each monomial product instead of
sharing it.

| Per atom-channel | cuEq (measured) | Generated, straight-line |
| --- | ---: | --- |
| forward | 1,381 | 478 FMUL (shared monomials) + 480 FFMA (immediate coefficients) + 61 FFMA (weights) + 16 (x = m·s) ≈ **1,035** |
| backward | 2,926 | 61 (q = w·dy) + 480 (adjoint h per monomial) + 136 (pairs) + 3·355 + 2·51 + 4 (∂ monomials) + 32 (dm, dD) ≈ **1,880** |

The two floors are taken per pass; the larger one counts, since traffic and arithmetic overlap:

| | Forward | Backward | Total |
| --- | ---: | ---: | ---: |
| traffic floor | 15.9 µs | 25.9 µs | 41.8 µs |
| generated arithmetic at 100 % FP32 rate | 20.0 µs | 36.4 µs | 56.4 µs |
| at cuEq's 70 % FMA-pipe use | 28.6 µs | 52.0 µs | 80.6 µs |

| Ceiling against | Paper (100 %) | Realistic (70 %) |
| --- | ---: | ---: |
| proxy 179.4 µs | 123 µs | 99 µs |
| production bucket 175.1 µs | 119 µs | 95 µs (2.8 %) |

## The local 5060 Ti and the earlier screen

The tool ran first on the RTX 5060 Ti ([local-5060ti/](local-5060ti/), 3 rounds, through `local_gpu.py run`).
There proxy is 898 µs (division 224, forward 121, backward 240, division backward 312) and floor is 414 µs
(0.35 TB/s against a 0.37 TB/s copy). At 448 GB/s the contraction is memory bound. That is the regime of
[sc-opt3-t02](../../docs/optimisation-ledger.md#sc-opt3-t02), which screened MH-1 on this GPU and found the
kernels at 72–96 % of memory speed-of-light. On the H200 (4.8 TB/s, 128 channels) the same kernels are FP32-issue
bound, so that screen's "at its traffic floor" does not carry over.

## Risks for the lab

- **Registers.** The straight-line backward wants 16 x, 136 pair products, 16 dx accumulators and the h values at
  once. cuEq already uses 96 registers at 31 % occupancy, so the generated code has to stage by monomial groups to
  avoid spills.
- **Rounding.** The summation order differs from cuEq's, so this is not bitwise. Adoption needs decision 66 (faster
  in both A/B orders) and the 3× float32 force/stress gate.
- **The division's backward** writes dm in the species GEMM's [5000, 16, 128] row layout and reduces dD per atom
  (one block per atom: one block reduction). This is small next to the arithmetic.
- **cuEq's run-time variant choice** (`algo1_2` / `algo1_4`) moves the forward by under 1 µs here.

## Next step (lab)

Build only the forward first: NVRTC straight-line code from the descriptor (shared monomials, immediate
coefficients, x = m·s fused), timed against cuEq's 40 µs on the H200 with an ncu instruction count. It has to come in
at about 30 µs or less, and confirm about 1,035 FP32 instructions per channel, before the backward is worth writing.
If the forward does not beat cuEq, stop: the divisions alone (about 48 µs) are below the bar.
