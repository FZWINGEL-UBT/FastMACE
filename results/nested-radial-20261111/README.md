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

**Status: REJECTED on the A100 and the RTX 4000 Ada (step 3b); opt-in only (`NESTED_RADIAL = True`).**
- Step 1 (certificate): passes.
- Step 2 (timing bound): passes on the A100. The 5060 Ti run was dropped (conductor).
- Step 3a (the split kernels in isolation): passes on the A100.
- Step 3b (production build, MD-step A/B, gate): the gate passes on both GPUs, but the A/B loses NbCl5 on both, and
  SrTiO3 too on the Ada. `NESTED_RADIAL_GPUS` stays empty. The H200 was not measured (Modal budget; its split forward
  rises to 118 registers).
- **The 3× float32 gate (`tools/force_stress_gate.py`) stays mandatory.** This is a rounding change under a
  certificate: the certificate covers the table, the gate covers the whole call.

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

## Step 3a: the real kernel shape, timed in isolation (A100)

`tools/nested_radial_lab.py` (commit a9bf5993) runs the K = 64 columns as two halves of 32.

**The split kernels.**
- Every edge evaluates the low half and the density column. Edges with r < r_far also evaluate the high half,
  through a scalar branch per edge.
- The forward keeps [16, 64] slots. A run's high block is added only if the run had a near edge.
- The backward loads a slot's high gradient block only for its first near edge.

**The lab.**
- r_far is 3.1518 Å, a piece start, and the far fraction on SrTiO3 10×10×10 is 0.837.
- Random tables are zero in columns 32–63 from r_far. On them the split kernels match the K = 64 copies to
  1.6e-7 relative.
- The savings below are measured at the real far fraction, so no 0.84 factor applies.

**A100-SXM4-80GB, Alex job 4984793 (a0831), two repeats agreeing within 1 µs:**

| Kernel (µs) | K = 64 copy | Split | Saving |
| --- | ---: | ---: | ---: |
| forward | 463.5 | 326.7 | **137** |
| backward, recompute (today's A100 default) | 466.3 | 414.7 | 52 |
| backward, power | 471.2 | 326.8 | **144** |

**Reading the table.**
- The recompute split keeps little: it runs two Clenshaw series per half, values and slopes.
- The power split does one dual Horner per half, and its saving holds.
- Against the A100's default backward (recompute, 466 µs), the power split saves 139 µs. With nested, the A100's
  backward would therefore switch to POWER_BACKWARD.
- POWER_BACKWARD alone is a tie on the A100 (1.0020 / 0.9977). It is already a gated rounding switch on the H200,
  Ada and 5060 Ti.
- Forward plus backward: 137 + 139 = **276 µs**, about 4.2 % of the 6,512 µs call.

**Census** (`split-census/`; registers / SASS instructions):

| Kernel | sm_80 | sm_90 | sm_120 |
| --- | --- | --- | --- |
| forward, split (copy) | **72** / 393 (128 / 508) | 118 / 521 (64 / 372) | 125 / 515 (80 / 390) |
| backward, recompute split | 80 / 500 | 72 / 506 | 80 / 515 |
| backward, power split | 80 / 452 | 72 / 474 | 80 / 491 |

- No spills; shared memory is 0–512 B.
- The sm_80 forward drops to 72 registers, because each half's atomics are 32 scalar `atom` instructions.
- The sm_90 forward rises from 64 to 118 registers. That matters for the H200's occupancy, so it is for the H200
  census step.

## Step 3b: production build, MD-step A/Bs and gates (A100, RTX 4000 Ada)

**The build** (`fastmace/rewrites/radial/nested.py`; routed by `precision.route_nested` once the generic radial
tables exist):
- `nested.install` fits the rotation per table row with 32 far columns and checks it (1e-7); `activate` swaps the
  rotated rows into the table, so the device holds one table. The plain rows wait on the host.
- The composed node matrices take the rotation in float64 (`compose_skip.composed(..., rotation=...)`).
- The tiled path calls `nested.release` (the plain rows and products back): tiled and nested are exclusive at plan
  time. A whole-system call that is not composed raises.
- `NESTED_RADIAL = True` with a table that needs more than 32 far columns raises at setup (`nested.require`).
- `tabled_outer_sum.NESTED_POWER` (default True) runs the split's backward in the power basis; False follows
  `POWER_BACKWARD`.

**The first A100 runs were void.** The precision record (compose_skip) applies before the radial record installs the
tables, so the route found no table and nested stayed off; the gate and A/B of job 4984870 measured the default
twice. Fixed in 8e51aef5. Since 7839954f every run carries a proof: `tabled_outer_sum.split_walks()` counts the split
walks' launches; `tools/ab_rewrites.py --b-expect [ENTRY@]MODULE:NAME` marks an entry INVALID (exit 3) unless the
counter advances on its side b and stays at 0 on side a; `tools/force_stress_gate.py --production-expect` fails a
structure whose builds did not advance it. All runs below passed their proof (a 0; b 10 SrTiO3 / 18 NbCl5; gate 8
per build).

**Structures** (`tools/far_fraction.py`, cutoff 6 Å, r_far 3.1518 Å; equilibrium / jitter 0.07 Å):

| Structure | Atoms | Edges | Edges per atom | Far fraction |
| --- | ---: | ---: | ---: | ---: |
| SrTiO3 10×10×10 | 5,000 | 370,000 / 368,752 | 74.0 / 73.8 | 0.838 / 0.837 |
| NbCl5 | 1,944 | 51,084 / 52,628 | 26.3 / 27.1 | 0.924 / 0.923 |

**MD-step A/B** (OMAT-0 medium float32, same process, 1 fs, skin 0.5, 6 × 256 steps; b/a, orders ab / ba; a =
today's defaults; all entries valid, no recompiles during timing):

| GPU, job | Side | SrTiO3 10×10×10 | NbCl5 |
| --- | --- | --- | --- |
| A100-SXM4-80GB, Alex 4985143 (7839954f) | b: NESTED_RADIAL (power split) | 0.9834 / 0.9831 | 1.0569 / 1.0568 |
| A100-SXM4-80GB, Alex 4985314 (a81ed9cc) | b: NESTED_RADIAL (power split) | 0.9843 / 0.9823 | 1.0570 / 1.0572 |
| | c: POWER_BACKWARD alone | 0.9980 / 0.9958 | 0.9985 / 0.9964 |
| | d: NESTED_RADIAL, recompute split | 0.9942 / 0.9914 | 1.0633 / 1.0648 |
| RTX 4000 Ada, murphy (938e5eab) | b: NESTED_RADIAL (power split; the Ada's default backward is power) | 1.0263 / 1.0211 | 1.1185 / 1.1178 |
| RTX 4000 Ada, murphy (a81ed9cc) | b: NESTED_RADIAL | 1.0264 / 1.0240 | 1.1192 / 1.1135 |
| | c: POWER_BACKWARD False | 1.0057 / 1.0079 | 1.0072 / 1.0083 |
| | d: NESTED_RADIAL, POWER_BACKWARD and NESTED_POWER False | 1.0282 / 1.0252 | 1.1224 / 1.1198 |

Step times (a → b): A100 SrTiO3 6.20 → 6.09 ms, NbCl5 1.67 → 1.77 ms; Ada SrTiO3 17.6 → 18.1 ms, NbCl5 3.88 →
4.34 ms.

**3× float32 gate** (SrTiO3 3×3×3 and NaCl; max-error ratios forces / stress / energy; all pass):

| GPU, side | SrTiO3 | NaCl |
| --- | --- | --- |
| A100, b (4985143) | 0.942 / 0.631 / 6.83 (2.1e-3 eV, under the 0.01 floor) | 0.922 / 1.195 / 0.786 |
| A100, d (4985314) | 0.877 / 0.616 / 6.59 (2.1e-3 eV) | 0.887 / 1.081 / 0.822 |
| Ada, b (murphy) | 0.794 / 0.565 / 3.33 (1.2e-3 eV) | 0.868 / 0.959 / 0.818 |

**Reading it.**
- The power basis does not explain the NbCl5 loss: POWER_BACKWARD alone ties on the A100 (c), and on the Ada both
  sides of b run it. With the recompute split (d against c) nested still loses NbCl5 on both GPUs (A100 about +6.6 %,
  Ada about +11 %).
- The loss does not follow the far fraction: NbCl5 has more far edges (0.92) than SrTiO3 (0.84) and loses more. It
  follows the edges per receiver (26 against 74): with short receiver runs, the split forward's per-run work (two
  accumulator halves, the high block's atomics when a run has a near edge) and the backward's lazily loaded high
  block cost more than the 32 skipped columns save.
- On the Ada even SrTiO3 loses (about +2.2 % in the recompute basis), so a size rule could serve the A100 only.
  With one structure on each side of any threshold, the A/B does not support one, so none is proposed.
- The isolated kernel saving of step 3a (276 µs on the A100 SrTiO3 call) shows in the A100 SrTiO3 step as about
  110 µs.

**Census:** step 3a's table holds; the sm_80 split forward is 72 registers (≤ 128), w′ unchanged.

**Decision (conductor, 11 November 2026):** the A100 and the Ada stay off; `NESTED_RADIAL_GPUS` is empty. The H200
would need its own census and a Modal pack.
