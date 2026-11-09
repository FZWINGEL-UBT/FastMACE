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
