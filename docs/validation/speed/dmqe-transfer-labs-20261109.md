# dm_qe transfer ideas: local and workstation labs, 9 November 2026

The species plan is reusable, but its measured benefit is smaller than the proposed whole-step estimates. The
geometry derivative is numerically sound in an isolated lab and saves tens of microseconds. Neither result is a
production adoption. The tested handwritten first-sum layouts are rejected on both free GPUs.

## Scope and provenance

Production FastMACE source and baseline: `a7a8211e2f740c5b95ebcdcd6ed2beee76098718`. Branch:
`fastmace/dm-qe-ideas-20261109`. No production implementation, configuration or default changed.
The original main checkout's uncommitted ledger and `first-adjoint-map` evidence were preserved.

Local: Fabi-PC, RTX 5060 Ti 16 GB, WSL, torch 2.14.1+cu130. Workstations: Murphy and Leto, RTX 4000 Ada Generation
20 GB, torch 2.14.1+cu130, approved NAS environments. Standalone kernels use `mace_cueq_latest`; force-call
comparisons use `mace_cueq_oeq`, hybrid float32. No environments were changed and no paid GPUs were used.
GPU reservations, workstation GPU/CPU idle checks and other-user monitoring were used. Murphy became occupied
before its full-force job; that job was skipped, and the test moved to Leto. Coraline was occupied and unused.

The force-call model is OMAT-0 medium, checkpoint SHA256
`d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a`.
Standalone outer/species inputs are synthetic values with the model's dimensions; geometry uses installed MACE's
`ZBLBasis` as reference. They are not checkpoint accuracy gates or MD benchmarks. Exact commands, raw JSON,
source hashes and failed attempts are in [the evidence folder](../../../results/dmqe-ideas-20261109/README.md).

## Handwritten first-interaction sums

The current target is `tabled_outer_sum_forward/backward`, not the already rejected fusion of the species GEMM.
Two receiver-owned CUDA variants preserve the separate projection: a shared-accumulator control and a register
forward with three fixed eight-value accumulator arrays per thread. The second backward assigns one 8-lane
subgroup per spherical-harmonic component and performs butterfly reductions without the control's redundant work.
Both eliminate the global S zero fill and output atomics. Receiver-density output is also checked numerically.

Tables have the OMAT-0 shape: 9 pairs, 896 pieces, 6 Chebyshev terms, 66 columns, 64 used channels, 16 components,
3 species. Coefficients are synthetic and their derivatives are generated consistently. Edge topology and lengths
come from real SrTiO3 10x10x10, both ideal and jitter 0.07 A. Graph replay outputs are validated before timing.

| GPU / geometry | Triton forward / register CUDA (us) | Triton backward / butterfly CUDA (us) |
| --- | --- | --- |
| Ada / ideal | 702.5 / 1225.6 | 748.5 / 2101.2 |
| Ada / jitter 0.07 A | 752.6 / 1453.1 | 912.4 / 2242.6 |
| 5060 Ti / jitter 0.07 A | 708.2 / 2066.1 | 845.4 / 3040.0 |

The shared-accumulator control is worse (Ada jitter: forward 750.3 -> 2636.8 us, backward 894.0 -> 4573.2 us).
The register version still loses 1.7–2.9x forward and 2.5–3.6x backward. It fails the free-hardware lab criterion;
no target-GPU or full-model rounding gate is warranted for these layouts. This rejects the tested receiver-owned
implementation, not every possible handwritten first-sum algorithm.

Both layouts and captured replays pass normalized parity within 1.3e-6, including receiver density. Ada NVRTC
reports 56 registers for the register forward and 40 for the new backward, with zero stack and spill bytes; local
memory spills are therefore not the explanation. Receiver-serial edge walks and CTA barriers remain structural
costs. Static PTX/resource counts do not verify the proposed dynamic instruction-per-edge estimates.

## Species plan

The current composed first interaction recomputes `onehot.argmax`, `species_plan`, and joined row maps per call.
`COUNT_PLAN` is a cheaper stable counting sort, not a hoist. Its default is on H200/A100; the composed path itself
is on Ada/H200/A100 and off on the local 5060 Ti.

The isolated screen keeps the same `species_mm_rows` forward/input-gradient consumers and replaces only their
plan inputs with precomputed, byte-identical tensors. Shape: 5,000 atoms, counts 1,000/1,000/3,000, input width 192,
output channels 128, path sizes 1/3/5/7, 3x TF32. Ten alternating blocks, 100 graph replays per sample, both capture
orders. These percentages are of the isolated consumer, not the full force call.

| GPU | Current plan | Frozen/current, ab / ba | Removed time, ab / ba |
| --- | --- | --- | --- |
| RTX 4000 Ada (Murphy) | stable sort | 0.9664 / 0.9694 | 72.2 / 65.8 us |
| RTX 4000 Ada (Murphy) | counting sort | 0.9958 / 0.9975 | 8.8 / 5.2 us |
| RTX 5060 Ti | stable sort | 0.9364 / 0.9501 | 149.0 / 116.0 us |
| RTX 5060 Ti | counting sort | 0.9980 / 0.9964 | 4.4 / 8.0 us |

Plans, consumer outputs and input gradients were byte-identical. Changed composition and same-count species
swaps rejected stale snapshots; rebuilding restored byte-identical outputs. Eight focused CPU lifecycle tests pass.
The lab's host validation is outside capture and timing. It is not a production invalidation implementation.

The full-force test uses independent hybrid models, verifies disjoint first-interaction modules/storage, and
patches only side b's private composed method. Leto, SrTiO3 6x6x6 (1,080 atoms), NPT-planned full-cell stress,
ten alternating blocks of 50 raw force-graph replays, both model build orders:

| Geometry | Baseline / frozen, ab (us) | b/a, ab | Baseline / frozen, ba (us) | b/a, ba |
| --- | --- | --- | --- | --- |
| Ideal | 3555.4 / 3552.7 | 0.99925 | 3478.6 / 3494.8 | 1.00465 |
| Rattle 0.05 A | 4283.2 / 4260.2 | 0.99462 | 4294.5 / 4274.8 | 0.99541 |

This establishes about 0.5% on the rattled force call and a tie/build-order sensitivity on the ideal lattice. It
does not establish 1–2% of an MD step. Maximum cross-build force difference was 2.11e-6 eV/A; repeated calls of the
same build also varied by roughly 2e-6 eV/A. Stress differences were at most 1.87e-8 eV/A^3 and energy differences
at most 6.39e-5 eV. These are comparisons, not a formal `force_stress_gate.py` pass.

Production integration should use per-force-field derived buffers and capturable, in-place species refresh.
Refresh must precede MC trial evaluation and cover acceptance, rejection/rollback, restart, model/head changes,
atom count, device and precision. The existing `DerivedStateRegistry` is the appropriate owner. The isolated
check proves why checking composition counts alone is insufficient. E0/clones/other skip-select plans were not
hoisted by this experiment. Per-atom float64 E0 is already a species-derived force-field buffer; this does not
prove that every original model E0 operation disappears from the compiled call. `SC_GENERATED` also changes the
target prologue on H200/A100. The old 57/67 us list is therefore not a measured removable cost of today's full
prologue. No current H200/A100 benefit was measured.

## Geometry derivative and energy reduction

The candidate computes ZBL value and analytical radial derivative together. Both current and candidate are
compiled. The current side is the installed `mace.modules.radial.ZBLBasis`, including its polynomial cutoff and
directed-edge 1/2. Upstream is constant 0.37 to test the general constant-scale case: in `ScaleShiftMACE` it need not
be 1 because pair energy is scaled with the interaction energy.

Local RTX 5060 Ti, synthetic periodic SrTiO3 (5,000 atoms, 370,000 directed edges), alternating graph blocks,
31 samples of 50 replays per variant:

| Isolated ZBL operation | Median (us) |
| --- | --- |
| Current forward + autograd derivative + scalar energy | 80.33 |
| Analytical value/derivative, scalar energy retained | 51.89 |
| Analytical value/derivative, force-only scalar omission | 49.68 |

The energy-preserving saving is 28.44 us; omitting the scalar sum adds only 2.21 us here. Float64 analytical versus
autograd derivative error is below 1e-12 at the small test points. Compiled graph radial differences are 6.06e-9,
per-atom energy differences 7.55e-8 and total energy difference 5.58e-5 eV. No full model force/stress gate or drift
was run for this rounding change.

For scale, the fresh default hybrid profile on the same local GPU at 5,000 atoms measured 24.78 ms/MD step; 28.44 us
is only 0.115% of that time, not a measured MD gain. A first plain-cuEq float64 control is excluded from that
comparison. Removing all step energy reductions would change the engine's returned-energy contract and needs
explicit demand/lazy-energy support for output, MC, restart and cell-changing callers. It is not a safe drop-in
work removal. Full geometry-tail fusion was not implemented: its quoted target bound is already borderline.

## Qualification limit

These are free-hardware lab results. H200/A100 speed estimates and the claimed per-edge instruction surplus remain
unverified. No measured instruction census of the current Triton first-sum kernels was obtained; the outer lab
records candidate PTX/NVRTC resource data only. No new default is justified by this report.
