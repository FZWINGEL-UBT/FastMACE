# Symmetrix-XL branches: what is not on main (9 November 2026)

Source: github.com/bonan-group/symmetrix-xl, fetched 9 November 2026 (main e25067c, 8 November). 17 remote branches
carry commits that are not on main. Most are release, CI, wheel or HIP-port work (fix/v020-release-dry-runs,
v0.2.0-prepare, release/cuda-wheel-audit, ci/*, feature/sphericart-hip-migration, integrate-sphericart-hip,
metatrain-mace-extraction, docs/freeze-v0.2.0-release-scope). The commits below are the ones that touch the speed or
the memory of the GPU path.

| Commit (branches) | What it does | FastMACE today | Take? |
| --- | --- | --- | --- |
| 68d68a6 "Optimize MH-1 CUDA execution stages" (optimize/mh1-polar-shared-backbone, many-small-structures-rtc, v0.2.0-prepare) | (1) `linear_2` through cuBLAS `SgemmStridedBatched` / `SgemmBatched` (strict FP32 or TF32) instead of generated kernels; equal-shape irrep blocks grouped into one GEMM. (2) The output mask is dropped when it is the identity and folded into effective weights. (3) `shared_source_features`: for 128 channels, l_max(source) = 1, 16 edge harmonics (float32, CUDA), a block per source node loads that node's features into shared memory once for all its edges. (4) Profiling names only (`R+A`, `M+H`, NVTX). | (1) Already: CUTLASS 3×TF32 / cuBLAS GEMMs, `segment_gemm`, `grouped_launch` (equal K, N), `gated_rows`. (2) Already: `fold_product_linear`, `fold_readout_tail`, species tables. (3) Not built: this is the source-owned edge loop ([conv-source-owned](../../docs/optimisation-ledger.md), NOT BUILT, stop rule). (4) Have Nsight ranges. | (3) only, see below |
| 902777f "Extend MH-1 optimizations to POLAR-1" (optimize/mh1-polar-shared-backbone, v0.2.0-prepare) | The same vendor GEMM for `linear_1` (message linear) on POLAR-1, if the workspace fits; a standard-MACE versus MH-1 SrTiO3 5,000-atom benchmark. | Already: `fx_e3nn_linear`, `polar_uvu`, grouped GEMMs on POLAR-1. | No |
| 0d9e6cc "Align generated radial and scalar gate arithmetic" | Exactness: geometry kept in double for the radial input; SiLU as `x / (1 + e^-x)` rather than `x · sigmoid`. | Exactness only; FastMACE's gate check is the float32 force/stress gate. | No (not speed) |
| 1d96b4f, 123653e (goofy-mole) | MACEField BEC / polarisability response: radial splines evaluated inline instead of edge caches; Phi1 Clebsch–Gordan reductions fused into consumers via a per-row CSR. Capacity: 8,788 → 87,808 atoms (FP64, RTX 5090), cost flat. | FastMACE has no field-response path; its radials are already evaluated inline from tables (spline forward, w′). | No (no such path) |
| ddff820 "Reuse joined direct schedules" | Python calculator: one schedule for many small structures. | Not a FastMACE use case (one large cell, MD). | No |
| ef9f894, eb46f5b | H1 affine export format v4; host (CPU) A1 GEMM blocking. | CPU / export only. | No |

## The one lead: a source-owned edge loop on data-centre GPUs

Symmetrix's `shared_source_features` is a source-owned message loop: one block per sender, the sender's features
read once into shared memory, and its gradient written without atomics. FastMACE streams edges in receiver order; the
sender side pays a gather in the forward and atomics in the backward.

FastMACE rejected this on 27 October on consumer GPUs ([conv-source-owned](../../docs/optimisation-ledger.md),
[conv-opt7-atomics](../../docs/optimisation-ledger.md): atomics ≤ 5 % of the backward), and the ledger's reopen
condition is "a data-centre GPU (H100-class bandwidth balance) becomes a target". The H200 and A100 now are targets.
Today's bounds ([step-limiters-20261109](../step-limiters-20261109/README.md)) cap what it can win:

| Part removed (w′ kernel lab) | H200 | A100-SXM4 |
| --- | ---: | ---: |
| dx atomics | 0.996 | 0.962 |

So the ceiling is ≈ 0.4 % of w′ on the H200 and ≈ 4 % of w′ (≈ 1 % of the step) on the A100, before the cost of a
second, sender-sorted edge order (the receivers' output gradients lose their locality, which the 27 October work
found costs more than it saves). Not worth building on the H200; on the A100 it is a ≤ 1 % step lead behind the
uncertain locality cost. Recorded, not started.

## Conclusion

No Symmetrix-XL branch has a speed idea that FastMACE does not already have, except the source-owned edge loop, whose
ceiling on today's defaults is ≤ 0.4 % (H200) and ≤ 1 % of the step (A100).
