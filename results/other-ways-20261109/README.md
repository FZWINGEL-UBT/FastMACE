# Three proposals from the external "other ways" research brief (9 November 2026)

The brief ([plans/notes/llm-prompt-20261109-other-ways.md](../../plans/notes/llm-prompt-20261109-other-ways.md))
asked an external LLM for ways to make MACE MD faster that are not kernel tuning. Of its five proposals, three were
worth a measurement. Wigner-6j edge-to-node has a cancellation problem. Exact CG factorisation (0–2 %) is a subset of
proposal 1. This page records those three measurements.

## 1. Polynomial moments for the first interaction: rejected (occupancy)

**Idea.** The tabled outer sum accumulates `S[receiver, sender species] += y_e a(r_e)ᵀ`. Here `a(r)` is a degree-5
Chebyshev series on the table piece of `r`. Within one (receiver, sender species, piece) group the coefficients are
shared. So `S` could take the moments `M_k = Σ_e y_e T_k(x_e)` per group, then one 6-term contraction with the
piece's coefficients per group instead of one per edge. The proposal's break-even was about 4.4 edges per group.

**Measurement.** [`tools/i1_occupancy.py`](../../tools/i1_occupancy.py), Alex job 4972594, CPU only:

- the interaction tables, fitted as production fits them (degree 5, default r_lo; 894 and 619 pieces);
- SrTiO3 10×10×10 (5,000 atoms), Gaussian displacements σ per axis;
- [occupancy-alex/](occupancy-alex/).

300 K thermal motion in SrTiO3 is about 0.06–0.09 Å RMS per axis.

| σ (Å) | Interaction | Groups | Mean edges per group | Edge-weighted mean | Edges in groups ≥ 5 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (perfect lattice) | 0 / 1 | 38,000 | 9.74 | 13.42 | 95.1 % |
| 0.05 | 0 | 271,460 | 1.36 | 1.78 | 3.7 % |
| 0.08 | 0 | 300,821 | 1.23 | 1.48 | 1.0 % |
| 0.08 | 1 | 286,198 | 1.29 | 1.59 | 1.4 % |
| 0.12 | 0 | 322,375 | 1.15 | 1.31 | 0.3 % |

On the perfect lattice every shell falls into one piece, but that is not an MD state. At thermal displacements,
67 % (interaction 0) and 61 % (interaction 1) of edges are alone in their group, and the mean group holds 1.2–1.3
edges, against a break-even of 4.4. The pieces are narrow (894 over 0.3–6 Å) because the certificate needs them, and a
0.08 Å displacement spreads each neighbour shell over several of them. Moments would add a per-group pass and save
nothing. **Rejected.**

## 2. Species-only work in the force graph: sized at about 1 % (second external pass, idea 2)

**Idea.** Some kernels in every force call depend only on the atoms' species, or only on the checkpoint, which do not
change during MD. They could be computed once per structure and passed in, which is bitwise.

**Sizing.** The kernel sequences of the step-limiter profiles are
[a100-alex](../step-limiters-20261109/a100-alex/main/kernel_sequence.csv) and h200 (round 3). At the start of the force
graph the following kernels read only `node_attrs [5000, 89]`, its argmax, or checkpoint weights (going by their
operand shapes and fused op names):

| Kernel (operands) | Likely role | A100 µs | H200 µs |
| --- | --- | ---: | ---: |
| `gemv2T` ([5000, 89] × [89, 1]) | per-species reference energies | 6.0 | 4.1 |
| `argmax_4`, `argmax_slice_..._skip_select_6`, `argmax_index_7`, `index_5` | species ids, column selections, skip-table rows | 21.2 | 14.0 |
| `arange_cumsum_eq_sum_8`, `cumsum_9`, the two `arange_gather_index_scatter` kernels | species plan: counts, order, inverse | 19.9 | 12.7 |
| `add_mul_unsqueeze_11`, `arange_mul_unsqueeze_12/13/14` ([5000, 1/3/5/7]) | species-GEMM row maps per irrep dimension | 13.3 | 8.9 |
| `clone_15/16` ([89, 13 / 29, 128]) | per-call weight copies (checkpoint constants) | 9.1 | about 5 |
| **total** | | **≈ 70 (1.1 % of 6.51 ms)** | **≈ 44 (1.3 % of 3.37 ms)** |

These numbers are an upper bound from names and shapes. The Inductor sources were not read, so a kernel may also
carry per-step work that has to stay. The bound is just above the 1 % rule on both GPUs and costs nothing in accuracy.
The work would be an engine change: the plan builds a per-structure species state, and the modules that call
`argmax` / `species_plan` today read it. Not built yet.

## 3. Factored symmetric contraction: faster on the A100 (follow-up 2)

**Idea.** Each output irrep of the correlation-3 contraction is a weighted sum of basis polynomials. Their span is 13
invariants plus 16 vector covariants.

[`sc_factored.py`](../../fastmace/kernels/cuda/sc_factored.py):

- picks a cheaper basis of memoised CG pair products, greedily, with a rank check at 1e-5 (the checkpoint's descriptor
  is exact only to float32);
- folds the change of basis into the weights at plan time (w′ = w K);
- generates the backward by reverse mode.

For OMAT-0 medium the factored form costs 413 operations plus 61 output FMAs. The generated kernels it replaces cost
about 1,040 FP32 operations per channel in the forward and about 1,500 in the backward.

**Lab** ([`tools/sc0_lab.py`](../../tools/sc0_lab.py), cases `fac-*`). All checks pass at float32 rounding: the
forward agrees with cuEq to 2.6e-7 rms; dm and dD agree with autograd to 2.5e-7 and 3.1e-7.

| Case (µs; × today) | Alex A100-SXM4-80GB, job 4972956 | Local RTX 5060 Ti (A100 defaults) |
| --- | ---: | ---: |
| gen-fused (today's forward) | 93.0 | 141.3 |
| fac-fused | 55.1 (0.593) | 140.5 (0.994) |
| gen-pair (today's forward + backward) | 226.0 | 397.9 |
| fac-pair | 162.4 (0.719) | 396.7 (0.997) |
| fac-pair4 (backward capped at 128 registers) | 147.0 (0.650) | 399.0 (1.003) |

Registers:

| Kernel | Registers |
| --- | ---: |
| gen-fused | 79 |
| fac-fused | 64 |
| fac-bwd | 161 |
| fac-bwd4 | 128, plus 24 bytes of local memory |

The 5060 Ti is memory-bound here, so the two forms tie. The A100 is FMA-bound, so the lower operation count shows.
On the A100 the saving is about 79 µs per force call, about 1.2 % of 6.51 ms.

**Next:** an engine switch, off by default, then the gate and the MD A/B.

## 4. Selective shared geometry in the interaction-2 forward: faster on the A100 (follow-up 3)

**Idea.** For each CG path, [`geom()`](../../tools/spline_read_lab.py) picks the cheapest of three forms:

- today's code;
- factoring w x out of a single-input path;
- hoisting g = Σ_j c Y_j into shared memory once per edge tile, for all 128 channels.

FMULs per edge and channel fall from 69 to 48 and shared loads rise from 9 to 28, at 80 registers.

**Lab** ([`tools/spline_table_bound.py`](../../tools/spline_table_bound.py) `--table-reads geom`). The geom forward
matches the real forward to 6.0e-7.

| Variant (µs; × real) | Alex A100, job 4972955 | Local RTX 5060 Ti (A100 defaults) |
| --- | ---: | ---: |
| real | 1,255.4 | 2,162.1 |
| geom | 1,088.5 (0.867) | 2,334.5 (1.080) |
| one-slab | 1,314.8 (1.047) | 1,699.6 (0.786) |
| one-slab + geom | 1,227.8 (0.978) | 1,740.7 (0.805) |

On the A100 geom saves about 167 µs per force call, about 2.6 % of the step. That meets the condition for building the
receiver-adjoint w′ variant (section 5).

**Next:** an engine switch, then the MD A/B.
