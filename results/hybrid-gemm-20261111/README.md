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

Alex A100-SXM4-80GB, source 4195be0d, OMAT-0 medium (`omat0-medium`), float32, hybrid backend, today's default A100
plan (side a) against the same plan with `tf32x3:HYBRID=True` (side b).
- Job 4984171: `tools/jobs/alex_ab_pack.sh` with `alex/pack.env`. 1 fs, skin 0.5, warm-up 200, 6 blocks of 256
  steps per side. Records in `alex-ab/`.
- Job 4984172: gate_pack with `alex/gate.sh`. Records in `alex-gate/`.

| | ab | ba |
| --- | --- | --- |
| SrTiO3 10×10×10 | a 6.3081, b 6.2425 ms: **0.9896** | b 6.2345, a 6.3119 ms: **0.9877** |
| NbCl5 3×3×3 | a 1.6934, b 1.6650 ms: **0.9832** | b 1.6694, a 1.6955 ms: **0.9846** |

Gate (`force_stress_gate.py`, ratio to the ASE float32 error, limit 3×; every structure passes):

| side | SrTiO3 F / S | NaCl F / S | NbCl5 F / S |
| --- | --- | --- | --- |
| default | 0.98 / 0.67 | 0.86 / 0.89 | 0.99 / 1.31 |
| HYBRID | 0.94 / 0.55 | 0.87 / 0.96 | 1.00 / 1.25 |

**Adopted on the A100 family** (decision 66: a rounding switch, faster in both orders, gate passes):
`HYBRID_GPUS = GPUModels(("NVIDIA A100-SXM4-80GB",))`, which covers every A100 model. The H200 has an sm_90 hybrid
cubin, but nothing was measured there, so it stays on 3×TF32.

## Range check before the merge (CPU, murphy)

HYBRID's cross parts are float16, so operands above 32,752 would overflow to inf with no runtime guard. `range_probe.py`
runs MACE's e3nn OMAT-0 medium model in float32 on the CPU under a TorchDispatchMode. It records max |A| and |B| of
every mm / addmm / bmm / baddbmm in energy, forces and stress, forward and backward. That covers every product FastMACE
sends to tf32x3, plus extra ones. Records: `range_probe.json`, `range_probe.log`.

| case | atoms | closest pair Å | max operand, all products | max, feature GEMMs (mm) |
| --- | --- | --- | --- | --- |
| SrTiO3 3×3×3 | 135 | 1.95 | 23.4 | 23.4 |
| NaCl 3×3×3 | 216 | 2.79 | 23.4 | 23.4 |
| NbCl5 3×3×3 | 1944 | 2.28 | 56.9 | 23.4 |
| SrTiO3, volume ×0.7 | 135 | 1.73 | 102.2 | 23.4 |
| SrTiO3, cell lengths ×0.7 | 135 | 1.37 | 778.7 | 60.7 |
| NaCl, volume ×0.7 | 216 | 2.48 | 35.3 | 23.4 |
| NaCl, cell lengths ×0.7 | 216 | 1.96 | 285.4 | 23.4 |
| NaCl, one pair pushed to 0.8 Å | 216 | 0.80 | 362.8 | 101.5 |
| NaCl 2×2×2 after 300 steps of 2,000 K Langevin MD | 64 | 2.23 | 23.4 | 23.4 |

- Model parameters: max |w| = 81.8. Buffers: 10.8.
- The values above 102 all come from 3×3 stress and displacement bmm products, which never reach tf32x3.
- Worst operand: 778.7, which is 42× below the limit.
- **Bar met:** every max is at or below 3,000 (10× margin).
- `tests/cpu/test_tf32x3_hybrid.py` checks this record, and checks the weights' margin when the checkpoint is under
  `FASTMACE_MODELS`.
