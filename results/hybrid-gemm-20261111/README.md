# Hybrid GEMM: TF32 big·big plus one FP16 cross product, in CUTLASS (11 November 2026)

This is the review retry of the A100 screen's idea 4 (`results/a100-colab-ideas-20261111`). The review is in
`results/half-far-wprime-20261111/review-20261111-rejections.md`. The first test had failed on accuracy, but that
error came from the separate single-pass cuBLAS TF32 GEMM it used for big·big. Here the scheme runs inside
production's CUTLASS 3×TF32 kernel.

## The kernel

The warp op is in `hybrid_sm80.cu` (lab) and `fastmace/kernels/cutlass/hybrid_f16.h` (production build).
- Same tiles (64×64×16, warp 32×32×16), 4 stages and staged accumulation as production's `tf32x3_sm80.cu`; only the
  warp-level op changes.
- Split: Ah = x with its low 13 significand bits cleared (the TF32 big part, as in 3×TF32); Al′ = (x − Ah)·2^11.
- Per k8 step and m16n8 tile:
  - one TF32 `mma.m16n8k8` for Ah·Bh;
  - one FP16 `mma.m16n8k16` for [Ah | Al′]·[Bl′; Bh] = 2^11 × cross;
  - four FFMAs to add 2^-11 × cross.
  - 3×TF32 needs three TF32 mma per k8; on sm_80 one FP16 k16 mma costs the same as one TF32 k8 mma.
- The k16 slot pairing permutes k identically for A and B (comment in the header).
- Range: the cross parts are float16, so operands must stay below 32,752 in magnitude.

**Offline census** (`census.sh`, sm_80 SASS, one k16 tile per warp):

| | tensor-pipe units | extra instructions | registers | spills |
| --- | --- | --- | --- | --- |
| 3×TF32 (same build) | 48 TF32 HMMA | — | 134 | none |
| Hybrid | 16 TF32 + 16 FP16 HMMA | 64 FFMA, 48 FMUL, 48 F2FP | 124 | none |

## Lab (`hybrid_lab.py`)

Shapes: tools/fp16x3_gemm_bench.py's seven OMAT-0 shapes plus 5000×128×{128, 384}. A has row-varying magnitudes.
Error is max |C − exact| / (|A||B|) against float64. Time is the median of 7 alternating blocks of 50 calls.

Local RTX 5060 Ti (sm_120 build, errors only; `smoke_5060ti.json`): the hybrid's error is 1.1–2.4e-7, against
1.1–2.1e-7 for the 3×TF32 reference.

**Alex A100-SXM4-80GB** (node a0535, driver 615.71.10, job 4983917, source 8da24384; `alex-lab/`).
- `ref` is 3×TF32 from the same CUTLASS build.
- `prod` is production's `tf32x3.mm_into`; it includes its Python launch path, which is why it is 34 µs on the
  smallest shapes.
- An earlier job (4983700) crashed in the column-major call, which passed a leading dimension of 1; fixed.

| M, K, N | hybrid µs | ref µs | prod µs | cuBLAS FP32 µs | hyb/ref | hyb err | ref err | cuBLAS FP32 err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5000, 2048, 128 | 121.1 | 139.9 | 140.6 | 210.9 | 0.866 | 1.06e-7 | 1.08e-7 | 8.34e-8 |
| 5000, 512, 128 | 29.3 | 33.0 | 34.8 | 56.0 | 0.886 | 1.23e-7 | 1.52e-7 | 8.40e-8 |
| 35000, 128, 128 | 42.9 | 47.5 | 47.8 | 78.1 | 0.903 | 2.65e-7 | 2.67e-7 | 3.33e-7 |
| 35000, 128, 192 | 61.4 | 68.3 | 68.9 | 113.4 | 0.898 | 2.26e-7 | 2.51e-7 | 4.24e-7 |
| 35000, 384, 128 | 113.3 | 128.0 | 128.9 | 229.3 | 0.885 | 1.44e-7 | 1.59e-7 | 3.64e-7 |
| 35000, 192, 384 | 162.1 | 190.1 | 188.7 | 327.0 | 0.853 | 2.37e-7 | 2.37e-7 | 3.61e-7 |
| 15000, 1024, 192 | 179.8 | 211.8 | 209.0 | 407.2 | 0.849 | 1.16e-7 | 1.39e-7 | 3.31e-7 |
| 5000, 128, 128 | 22.1 | 22.0 | 34.0 | 19.1 | 1.003 | 2.00e-7 | 2.30e-7 | 3.05e-7 |
| 5000, 128, 384 | 23.1 | 24.5 | 34.1 | 38.2 | 0.944 | 2.21e-7 | 2.70e-7 | 3.39e-7 |
| **sum** | **754.9** | **865.1** | | | **0.873** (0.851 of prod) | | | |

**Stop rules (review):**
- *Shape-weighted time below 3×TF32's:* **passes** (0.873 summed; 0.85–0.90 on every large shape).
- *Error at or below plain float32 GEMM on each shape:* passes on 7 of 9. The two 5000-row shapes with K = 2048 and
  512 miss it: cuBLAS FP32 is unusually accurate there (8.3e-8), and production 3×TF32 misses it on the same two
  shapes. Against today's default (3×TF32) the hybrid is at or below on every shape, so it goes on to the A/B and
  the gate.

## Production switch

- `fastmace.kernels.cutlass.tf32x3.HYBRID` is None, with an empty `HYBRID_GPUS`, and is read at call time like `TILE`.
- With it on, every `tf32x3` entry loads from `tf32x3_hyb_sm80.cubin` / `_sm90.cubin`: plain and species GEMMs,
  `linear_div`, `skip_select` and `segment_rows`.
- These cubins are built by `tools/build_cutlass_tf32x3.sh` with `-DFM_HYBRID_F16`; the shipped 3×TF32 cubins are
  unchanged (sha256 as recorded).
- Test: `tests/cpu/test_tf32x3_hybrid.py`.

## MD-step A/B and gate

Alex job 4984171 runs `tools/jobs/alex_ab_pack.sh` with `alex/pack.env` (SrTiO3 10×10×10 and NbCl5 3×3×3, both
orders). Alex job 4984172 runs gate_pack with `alex/gate.sh` (SrTiO3 3×3×3, NaCl, NbCl5). Results below.
