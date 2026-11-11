# New force-call ideas, 11 November 2026

Read-only research against `D:/Margraf/FastMACE` main `b951e804`, the working optimisation ledger, the 9 November H200/A100 limiter records, `plans/review-20261111-rejections.md`, and the concurrently added `plans/ideas-20261111-round3.md`. No production edits, GPU execution, timing measurements, model fitting or checkpoint changes. Two small CPU screens used the existing `fastmace_cpu` environment with CUDA disabled and one CPU thread. Files in this directory are separate from the checkout.

These are hypotheses. The CPU checks below establish algebra and reuse opportunities, not a speedup or a force/stress gate pass. The current ledger includes the previous half-far, half-system, formulation-search, hybrid GEMM and ghoist directions; they are not counted as new here. Today's rejection review disputes parts of the half-far and hybrid tests. The new round-three note covers first-interaction ownership/packing and first-term accumulator initialization; those are also excluded here.

## 1. Joint angular-coordinate rescaling

For a tensor-product path, change coordinates with diagonal scales:

`x_tilde = x / a`, `Y_tilde = Y / b`, `m_tilde = m / s`,

so `C_tilde[o,i,j] = C[o,i,j] * a[i] * b[j] / s[o]`.

Choose the scales jointly to make as many nonzero CG coefficients as possible exactly +1 or -1. The search changes the representation of the same local contraction, not its rank or its weights. Share the input and harmonic scales across paths; allow a distinct output scale for each path/component. Restore original coordinates before the existing node operations. In the backward, use `g_tilde = s*g` and restore input gradients with the inverse input scales.

The expensive edge loop then avoids many coefficient multiplications. Input feature scaling can occur per node; harmonic scaling per edge is shared over channels; receiver-adjoint scaling belongs in existing staging. Output scaling must be explicitly undone at the node boundary: arbitrary component/path scales do not commute through all current linears. Accumulate the scaled sender gradient before restoring it, to avoid paying its inverse scale per edge.

### CPU evidence

`gauge_probe.py` enumerates OMAT-0 medium's ten angular paths, 86 nonzero e3nn CG terms, and searches consistent subsets of the logarithmic scaling equations. This is a heuristic search, not a proof of optimality.

- Symbolic nontrivial coefficient pre-multiplications per edge/channel: 25 after a common factor per path, 18 after output scaling alone, **9 after joint input/harmonic/output scaling**.
- All coefficients cannot simultaneously be made unit magnitude by this diagonal transformation; the all-equations least-squares residual is nonzero.
- Twenty random 128-channel inputs per path: local message and derivatives with respect to x, Y and w match the untransformed float64 contraction to at most **3.4e-14 relative to the maximum output magnitude**.
- No float32 MACE model, force gate, generated kernel or compiled instruction count was tested.

This count covers coefficient pre-multiplications only. Saving 16 operations in a several-hundred-instruction edge loop is a roughly 1%-class whole-call opportunity, not a large demonstrated gain; boundary transforms and occupancy can consume it. The first next test is a production-generator offline census including all transforms. Only proceed if its bound survives.

### Novelty boundary

`CG_FOLD`/`FOLD_ALPHA` absorb a common path factor; `GEOM` precomputes geometry combinations; the rejected receiver-adjoint proposal expands the staged adjoint. This proposal preserves the sparse topology and does not expand the 40-component path output. Round three considered scaling dout rows alone; the additional freedom here is the joint input and harmonic scaling. General CG factorisation has been mentioned before, so the claim is about this specific diagonal transformation.

## 2. Decode once for two consecutive edges on one spline piece

The forward currently decodes PACKED8 coefficients on every edge. With edges sorted by sender species and length, consecutive edges can use the same table row despite having different lengths, directions and sender features.

When two adjacent edges have the same table row, load/decode one path's three coefficients, evaluate both lengths, process those two edges for that path, then move to the next path. This avoids retaining all paths' decoded coefficients across an entire run. It also avoids changing the sums into polynomial moments. Preserve each output's edge accumulation order and retain a normal single-edge path for unmatched edges. Register pressure and a larger generated body remain real risks.

### CPU evidence

`decode_pairs_probe.py` uses the saved 736-interval production spline grid in `results/omat0-breakthroughs-assessment-20261108/table_ideas_bound.json`, 5,000-atom SrTiO3, Gaussian positional displacements, and receiver/species/length ordering. The pairing is restricted to existing 32-edge staging tiles.

| Displacement sigma per axis | Directed edges | Coefficient decodes removable by pairing |
| --- | ---: | ---: |
| 0.05 A | 369,244 | 24.83% |
| 0.08 A | 368,866 | 19.16% |
| 0.30 A | 374,906 | 9.19% |

These are synthetic snapshot counts, not temperature assignments, production captures or general results for other compositions. The exact current fitted grid must be checked in a real prototype.

The existing PACKED8 report counts approximately 14 decode instructions per weight, with ten weights per channel. The counts above imply about **13-35 decode instructions removed per edge/channel on average**, before pairing overhead. This is only a fraction of the forward, which is about one fifth of the recorded force-call kernel time. H200 is the stronger initial target because its forward is issue-bound; A100's FMA bottleneck may hide much of the benefit.

First next test: offline compile a paired body, ensuring no spill or harmful occupancy change, then a real-operand forward-kernel falsifier. This is instruction reuse, so a table-byte-only locality bound does not decide it. No new edge sorting should be charged to the gain; use the existing order.

## 3. Texture interpolation with an explicit quantization correction

This is the deliberately speculative candidate, and a new extension of the earlier uncorrected texture proposal.

NVIDIA texture interpolation uses eight fractional bits in its interpolation coordinates. A cubic can be represented as a trilinear interpolation along a 3D cell's diagonal, with Bernstein control points arranged by corner Hamming weight. The returned value corresponds approximately to the hardware-quantized coordinate q, not the requested t.

Let delta = t-q. For the exact polynomial arithmetic,

`p(t) = p(q) + delta*p'(q) + delta^2*p''(q)/2 + delta^3*c3`.

The forward PACKED8 polynomial is quadratic, so the expansion ends after the quadratic term. Explore a texture representation providing the leading value and enough derivative information to apply a short correction on the SM. The objective is to move interpolation work off the ordinary issue/FMA path. Any omitted correction needs a certificate; no omission is assumed safe here.

Hardware interpolation arithmetic adds error beyond coordinate quantization, and correcting q alone does not establish accuracy. Texture storage can expand substantially, extra fetches may dominate, and a corresponding radial derivative must be checked. There is **no defensible speed estimate yet**. First derive a concrete layout and count fetches/corrections, then emulate value/derivative error; a later isolated TEX lab must measure the actual interpolation error and throughput.

Primary hardware reference: [CUDA C++ Programming Guide, Texture Fetching / Linear Filtering](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf).

## Recommendation and sources

The first two justify cheap offline compiler screens; the texture idea is much lower confidence. Nothing in these checks supports a large whole-call speedup yet.

Local source records:

- `D:/Margraf/FastMACE/docs/optimisation-ledger.md`, detailed November rows.
- `D:/Margraf/FastMACE/plans/ideas-20261111-round3.md` and `plans/review-20261111-rejections.md`.
- `D:/Margraf/FastMACE/results/step-limiters-20261109/{h200,a100-packed8}/limiters.md`.
- `D:/Margraf/FastMACE/results/packed-table-lab-20261109/README.md`, packed8v2 instruction counts.
- `D:/Margraf/FastMACE/results/other-ways-20261109/README.md`, GEOM and receiver-adjoint mechanisms.
- `D:/Margraf/FastMACE/fastmace/kernels/oeq_streaming/{wprime_backward,spline_forward}.py`.

Raw new evidence is in `gauge_probe.json` and `decode_pairs_probe.json` beside their scripts. The local algebra probe is not a production numerical qualification.
