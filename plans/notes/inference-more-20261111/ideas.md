# More force-call ideas, 11 November 2026

Read-only research started with FastMACE main `f31d4f50`; the shared checkout advanced to `bb13d0a7` during the audit. Sources included the working optimisation ledger, `plans/next-steps-20261111.md`, current first-interaction and spline kernels, prior Gaunt/deferred-CG reports, and saved limiter profiles. No production edits or GPU jobs. CPU probes used the existing CPU environment with CUDA disabled and one thread. These files live outside the checkout.

The strongest new proposal is a **nested radial basis with a shorter active prefix at longer distances**. Two other directions below are a compact exact angular-product construction and shared-denominator rational radial tables. All runtime savings remain unmeasured. Numerical screens use OMAT-0 medium's Sr/Ti/O cache; they do not establish general MACE accuracy or performance.

## 1. A nested radial basis: 64 coordinates nearby, 32 farther out

### Mechanism

The first interaction forms `S[i,s,m,k] = sum_edges Y_m a_k(r)` with 64 radial coordinates, then applies frozen node matrices. A single globally low-rank approximation is restrictive. Instead, build one complete 64-dimensional basis per species pair, arranging its first coordinates to approximate the radial functions at larger r. Retain all coordinates nearby, and evaluate/accumulate only a certified prefix farther out.

With row vectors, channel normalization D, and orthonormal row basis Q:

`z = a D^-1 Q^T`, and `a = z Q D` when all coordinates are retained.

Compose `Q D` into the existing first-interaction species projection at setup. The node GEMM still has 64 radial input columns; its adjoint already arrives in z coordinates. The edge loops alone use the shorter prefix. Sender and receiver species dependencies fit the current slot/species-matrix ownership. The scalar density channel stays separate.

Initialize the smallest prefix with the values and slopes at every broad-band boundary. Subsequent basis vectors are orthogonal to those directions. In real arithmetic, truncations on either side then meet in both value and slope. This is a controlled approximation away from boundaries, not an exact low-rank identity.

### New CPU evidence

Source: cached interaction-0 table `c697fb752157aea2cccd6085ebe9843e283d630c4388f8e419a4c3e6fcebff11.pt`, shape `[9,893,6,66]`; use the first 64 coordinates. This is the same table identified in the saved first-adjoint study. The probe evaluates its float32 coefficients in float64, rather than executing the original neural network.

`nested_radial_basis.py` uses 8,193 fit points and 8,209 offset test points per pair, with values and derivatives in the basis construction. Errors are normalized per original column, separately for values/slopes, using physical-region maxima for r >= 1 A and whole-domain maxima below it. Test points select ranks, so they are a sampling screen rather than an independent production certificate.

| Bands (A) | Maximum selected ranks over nine ordered pairs, at 1e-7 | Prefix widths rounded to multiples of 8 | Node width |
| --- | --- | --- | --- |
| 0.3–3.15 / 3.15–6 | 64 / 29 | **64 / 32** | **64** |
| 0.3–2 / 2–4 / 4–6 | 64 / 49 / 25 | 64 / 56 / 32 | 64 |
| Four equal bands | 64 / 55 / 33 / 24 | 64 / 56 / 40 / 24 | 64 |

The simple two-band design is currently the clearest candidate. Both sampled value and derivative residuals meet 1e-7 in every tested pair/band. The original-network radial certificate and force/stress gate have not been run.

`radial_cost_bound.py` counts synthetic 5,000-atom SrTiO3 edges. At displacement sigma 0.08 A per axis, **83.67%** have r >= 3.15 A. Half width on those edges removes at most **41.83% of first-interaction edge work**. With 9 November kernel profiles, optimistic ceilings are **135 us / 4.15%** of H200 force-kernel time and **353 us / 5.42%** on A100. At sigma 0.05 and 0.30 A the corresponding edge-work fractions are 41.87% and 42.75%.

These are ceilings, not expected speedups. Fixed output/fill/scatter work remains. A runtime mask in the current compile-time K=64 Triton body would retain arithmetic and storage; the implementation must actually run a narrower branch/body or equivalent compact work. Charge any edge partitioning, extra launches, table layout and rounded composed matrices.

### Why the older closures do not decide this

- Fine-piece moments need several edges on the same one of 893 spline pieces. This proposal does not group edges by fine piece.
- Full rank over the entire distance range does not prevent a nested far-range subspace.
- Separate local bases were also screened (`radial_subspaces.py`): two independent bands need node width 96, four need 136 at 1e-7. The nested construction removes that width penalty.
- This is related to numerical radial compression, but no prior nested-prefix/composed-projection experiment was found. The historical MH-1 rank59–61 first-interaction result and rank65 second-interaction result should not be conflated.

**Next falsifier:** transform actual tables/composed matrices, evaluate them in float32 and run the existing w/w-prime radial certificate against the original float64/float32 chains. If that passes, compile a genuinely narrower edge body and bound its saved work before a GPU lab.

## 2. Accumulate angular products in a compact interpolation basis

### Mechanism

For each selected angular path, construct an exact decomposition

`C[o,i,j] = sum_r D[o,r] A[r,i] B[r,j]`.

Transform sender features by A once per atom, geometry by B once per edge, accumulate the independent path's radial weight times those two transformed quantities, then reconstruct the physical message with D once per receiver. Every path keeps its own original radial weights.

Homogeneous quadratic/cubic polynomial interpolation gives real decompositions using:

- six products for `1 x 1 -> 2`, and six for its index permutation `1 x 2 -> 1`;
- ten products for `1 x 2 -> 3`, and ten for `1 x 3 -> 2`.

These are constructive upper bounds on tensor rank, not proofs of minimum rank. Shared evaluation directions limit the sender vector to ten forms. Apply this to the forward only initially, reconstructing the physical message before the existing node chain; keep the physical-coordinate backward unchanged. It is the same real-arithmetic function, with different float32 rounding.

### New CPU evidence

`angular_products.py` constructs constant A/B/D matrices from e3nn harmonics and CG tensors. Maximum tensor-coefficient residual is **4.0e-16**. Twenty random 128-channel inputs per path agree in float64 to at most **2.3e-15 relative to maximum output magnitude**. Random local float32 cases reach **4.8e-7**; this is not a force/stress qualification.

The four paths' simple term-product schematic counts **148 operations versus 64** in the transformed edge body, excluding node/geometry transforms and reconstruction. Against geometry-hoisted arithmetic the same count is **66 versus 64**. Thus this count primarily motivates a screen where GEOM is off, such as the current H200 default. It does not justify an A100 speed claim against GEOM.

Whole-forward accumulators become **52 instead of 40**. The earlier literal deferred-CG implementation needed 76 and spilled. Additional costs are substantial: 32 transformed geometry forms across these four paths, sender vectors expanding from three to ten components (17.92 MB extra at 5,000 atoms x 128 channels), extra gathers and reconstruction. Load forms in short-lived groups to avoid holding all ten next to 52 accumulators.

### Novelty and stop rule

This is a new compact construction within the earlier expanded-Gaunt/deferred-CG direction. The prior Gaunt assessment explicitly left independent-filter expansions open; its 48.6x bound moved dense channel projection onto each edge. This design retains independent filters and reconstructs only at nodes. The rejected 76-coordinate body did not perform this interpolation compression.

**Next falsifier:** generate a forward-only body including all transforms, compile for sm_90, and stop if its live registers/spills or actual instruction count consume the reduction. No runtime gain is estimated from the schematic operation count.

## 3. Let 64 radial functions share a denominator

### Mechanism

Replace a piece's separate degree-five polynomials with

`a_k(r) ~= N_k(r) / Q(r)`,

where each column has its own numerator and all 64 columns share one quadratic denominator. Evaluate the reciprocal and its derivative once per edge. This is inspired by [multi-function rational approximation](https://par.nsf.gov/servlets/purl/10210126), but the CPU experiment here uses a simple simultaneous Chebyshev coefficient least-squares construction, not that paper's complete algorithm.

The first backward can avoid per-column quotient-rule corrections. With node adjoint matrix G, define `u = G N` and `v = Y^T G`. Then

`dY = u / Q`,

`dr = (v dot N') / Q - (Q' / Q) * (Y dot dY)`.

The second term reuses the harmonic adjoint and needs a 16-component dot, rather than a quotient correction on every radial coordinate. The scalar density channel remains separate.

### New CPU evidence

`shared_denominator.py` fits one denominator per pair/piece on the same cached first table. Denominators are constrained by `abs(q1)+abs(q2) <= 0.8`, ensuring Q >= 0.2 throughout the piece. Analytical Chebyshev coefficient bounds include both value and r-derivative residuals relative to the cached polynomial, normalized by column scales; they exclude new float32 rounding and the original network/table discrepancy.

At a **3e-7 bound threshold**, over the 527 physical pieces x nine pairs:

| Numerator degree | Coefficients per column, plus two shared denominator coefficients | Fraction meeting the bound |
| --- | ---: | ---: |
| 2 | 3 | 2.36% |
| 3 | 4 | 22.41% |
| 4 | 5 | 99.85% |

The aggressive three-coefficient design fails this screen for most pieces. A model-certified mix of four/five/six-coefficient pieces is the remaining possibility. The simple bound counts suggest roughly 20% fewer per-column coefficients at that threshold, before the shared denominator and metadata. This is not a performance estimate, and 3e-7 of a column scale is not the production certificate threshold.

**Next falsifier:** apply the real network-relative radial certificate, count representation bytes and all quotient arithmetic. Stop if the certified format does not clear the first-interaction bound. This is lower priority than the nested basis, and their gains overlap.

## Evidence files and source pointers

- `nested_radial_basis.py/.json`: nested-prefix ranks and sampled residuals.
- `radial_subspaces.py/.json`: separate broad-band bases and their larger node widths.
- `radial_cost_bound.py/.json`: synthetic edge counts and optimistic ceilings.
- `angular_products.py/.json/.npz`: exact angular factors and local checks.
- `shared_denominator.py/.json`: rational-table coefficient bounds.
- Shell launchers set CUDA visibility empty and CPU library thread counts to one.

Owning FastMACE sources read: `fastmace/rewrites/mh1/modules.py` (first interaction projection), `fastmace/kernels/triton/compose_skip.py`, `tabled_outer_sum.py`, `radial_table.py`, `fastmace/rewrites/radial/__init__.py` (chain and certificate), `spline_forward.py`. Relevant historical reports: `results/other-ways-20261109/README.md`, `results/packed-table-lab-20261109/README.md`, and `results/omat0-breakthroughs-assessment-20261108/codex-report/report.md:137-145`.
