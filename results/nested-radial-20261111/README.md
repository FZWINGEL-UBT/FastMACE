# Nested radial basis for the first interaction (11 November 2026)

Item §1 of `plans/next-steps-20261111-rev2.md` (idea 1 of the outside note `inference-more-20261111/ideas.md`,
probe `nested_radial_basis.py`).

**The idea.**
- Per species pair, rotate the first interaction's 64 radial columns with an orthonormal basis. Its first directions
  span the radial functions beyond 3.15 Å.
- The rotation is composed into the node matrices at setup (COMPOSE_SKIP, `_composed` in
  `fastmace/rewrites/mh1/modules.py`). Edges beyond 3.15 Å then use only the first 32 columns.
- **Ceiling:** 5.4 % of the A100 call, 4.1 % of the H200's.
- **Scope:** only where COMPOSE_SKIP is on (H200, A100, Ada), not the 5060 Ti.

**Status.**
- Step 1 (certificate): passes.
- Step 2 (timing bound): passes on the A100. The 5060 Ti run is queued.
- Step 3 (real kernels): next.
- **The 3× float32 gate (`tools/force_stress_gate.py`) stays mandatory in step 3.** This is a rounding change under
  a certificate: the certificate covers the table, the gate covers the whole call.

## Step 1: certificate (leto, CPU)

`tools/nested_radial_certificate.py` (commits 1a5d0c77 and 12b4fc20).

**Inputs.**
- OMAT-0 medium (`mace-omat-0-medium.model`), Sr Ti O, interaction 0.
- The production fit (`radial.fit`, float32): 9 pairs, 892 pieces, fit error 3.85e-5. The folded matrix is
  (512, 65).

**Construction.**
- Per pair: z = a D⁻¹ Qᵀ, with D the physical column maximum. Q is seeded with the value and slope at 3.15 Å, then
  takes the far band's SVD and is completed to 64 directions.
- The rotation is applied to `coef` and `dcoef` (float64 matmul, stored as float32), and the table is evaluated in
  float32.
- Columns at or past the far width are zeroed for r ≥ 3.15 Å.
- The node side uses C = float32(Q D Fᵀ).
- The comparison is `radial.certify`'s own logic (values and slopes, whole domain and from 1 Å, w columns and density).
  This path reproduces the production certificate exactly when unrotated (0.3011).

**Results.** Pairs are in the order 00 01 02 10 11 12 20 21 22. "Far error" is the max over pairs of the value and
slope projection errors, as fractions of each column's scale.

| Far width | Far columns per pair | Far error | Certificate worst | w parts: value all / phys, slope all / phys |
| --- | --- | ---: | ---: | --- |
| unrotated | – | – | 0.3011 | 0.049 / 0.0055, 0.146 / 0.040 |
| tolerance 1e-7 | 29 28 27 28 27 26 27 26 29 | 9.5e-8 | 0.3011 | 0.0496 / 0.0055, 0.146 / 0.040 |
| tolerance 5e-8 | 32 33 29 33 28 28 29 28 29 | – | 0.3011 | 0.0495 / 0.0055, 0.146 / 0.039 |
| tolerance 2e-8 | max 58 | – | 0.3011 | 0.049 / 0.0057, 0.142 / 0.039 |
| tolerance 1e-8 | max 60 | – | 0.3011 | 0.049 / 0.0076, 0.145 / 0.039 |
| **exactly 32 for every pair (the kernel's width)** | 32 × 9 | **5.4e-8** | **0.3011** | 0.0495 / 0.0055, 0.154 / 0.039 |

**Reading the table.**
- The worst part is the density column, which the rotation leaves alone. The w parts change only within the float32
  network noise.
- Conductor's decision (11 November): the certificate decides the tolerance, so 1e-7 passes. The kernel ships 32 far
  columns, certified in the last row.

Files: `certificate/` (four tolerances) and `certificate-k32/` (1e-7 and fixed 32).

## Step 2: timing bound (every edge at 32 columns)

`tools/first_interaction_lab.py --only 'copy|k32'` (commit 5b3fb555). The `k32` variants are the lab's copies of the
production kernels with K = 32, J = 34 and a [16, 32] slot block, run on random tables. The lab is the C1/C2 one:
SrTiO3 10×10×10, 368,752 edges, timed as 10 back-to-back calls between CUDA events.

**A100-SXM4-80GB, Alex job 4984771 (node a0831), two repeats agreeing within 0.5 µs.**

| Kernel (µs) | K = 64 copy | K = 32 | Difference |
| --- | ---: | ---: | ---: |
| forward | 465.4 | 264.3 | 201.1 |
| backward, recompute (the A100's default) | 467.5 | 307.8 | 159.7 |
| backward, power | 472.4 | 264.4 | 208.0 |

**Gain.**
- 0.84 × (201 + 160) = **303 µs** with the A100's default backward, about 4.7 % of the 6,512 µs call. With the power
  backward it is 343 µs.
- That clears the 130 µs stop line.

**Caveat for step 3.** The bound halves the slot block too: the [16, 32] atomics and the zero fill. The real kernel
keeps a [16, 64] slot block, because near and far edges share slots, and skips columns 32–63 only for far edges.
Part of the forward's 201 µs is therefore out of reach (compare C1, where the slot atomics alone were worth 202 µs).
The backward reads a [16, 64] gradient row per edge either way.

**Census** (`bound-census/`, sm_80 registers / SASS instructions):

| Kernel | K = 64 | K = 32 |
| --- | --- | --- |
| forward | 128 / 508 | 63 / 228 |
| backward, recompute | 80 / 512 | 62 / 339 |
| backward, power | 80 / 480 | 64 / 317 |

**RTX 5060 Ti:** queued behind another session's reservation of the local GPU (`bound-rtx5060ti/`). It is
informative only: COMPOSE_SKIP is off there.
