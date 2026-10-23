# OPT-T01: FP32-accurate tensor-core GEMMs (optimisation round 1)

Plan: `plans/opt-1.md`, OPT-T01. **Result: rejected on sm_89; the local sm_120 rerun is still to do.**
On the production GEMM shapes, no FP32-accurate tensor-core mode our stack exposes is faster than
cuBLAS's SIMT FP32 GEMM. The only one available, 3×TF32 (Triton `tf32x3`), is accurate but slower on 89
of 92 shapes. The 3 it wins are tiny.

## 1. Survey (step 1, 20 October 2026; code reading, no GPU)

| Mode | How it is enabled | sm_89 | sm_120 | Accuracy |
| --- | --- | --- | --- | --- |
| torch 2.14.1 `torch.backends.cuda.matmul.fp32_precision` | `ieee`, `tf32` or `none`; `set_float32_matmul_precision("high"/"medium")` gives `tf32` | – | – | 1×TF32 is not FP32-accurate. No 3× or emulated mode: `bf16x3`/`tf32x3` are "unknown precision", `bf16` is refused for CUDA |
| cuBLAS BF16×9 emulation (`CUBLAS_COMPUTE_32F_EMULATED_16BFX9`, `CUBLAS_EMULATE_SINGLE_PRECISION`) | environment variable or cuBLAS API (no torch binding) | **no** | **no** | cuBLAS docs, "Floating Point Emulation Support Overview": BF16×9 on compute capability **10.0 and 10.3** only |
| Triton `tl.dot(input_precision="tf32x3")` | per `tl.dot` | yes | yes (same rule, `capability // 10 >= 8`) | 3×TF32 (big·big + big·small + small·big) |
| CUTLASS 3×TF32 (`OpMultiplyAddFastF32`, example 27) | C++ extension | Sm80 kernels | Sm80 kernels | as above |

- Torch's TF32 flag does not govern custom Triton kernels. FastMACE's four `tl.dot` calls set
  `input_precision="ieee"` explicitly.
- `bf16x3` is not FP32-accurate. `bf16x6` has an open bug for K < 16 (triton#12252), which our K = 10 layer
  would hit.
- Sources and the full table are in the survey note of 20 October 2026 (cuBLAS documentation sections
  1.5, 2.2.12 and 2.4.26; Triton's `F32DotTC`; CUTLASS example 27 and discussion #385).

## 2. Micro-benchmark (step 2)

`tools/gemm_precision_bench.py`, launched by `results/opt-T01-gemm-precision-20261020/murphy.sh`:

- **Shapes:** every GEMM operator of one MH-1 force call from 2-T20a's `gemms.csv` (NaCl and NbCl5
  3×3×3), with the strides the compiled model uses. The force call includes the backward, so these cover
  the forward products and their input gradients.
- **Modes:** cuBLAS IEEE (production), cuBLAS 1×TF32 (rate only), and one plain Triton GEMM with
  `input_precision="ieee"` (the control) and `"tf32x3"`. Each Triton mode reports the fastest of 7 fixed
  tile configurations. That is a screen's upper bound, not a production rule.
- **Errors:** against a float64 product of the same float32 inputs.
- **Run:** murphy, RTX 4000 Ada (sm_89), driver 595.58.03, NAS `mace_cueq_oeq` (torch 2.14.1+cu130,
  triton 3.8.0) through the hybrid launcher. FastMACE commit `80e2c47`.
- **Conditions:** GPU empty, load 0.32 at 18:22:58 UTC 20 October 2026 (`runs/murphy-20261020/run.log`).
- **Raw data:** `results/opt-T01-gemm-precision-20261020/murphy/`.

### Peak, 4096³

| Mode | TFLOPS | Max. relative error vs float64 | Frobenius relative error |
| --- | ---: | ---: | ---: |
| cuBLAS IEEE FP32 (SIMT) | **16.3** | 1.5e-6 | 8.1e-7 |
| cuBLAS 1×TF32 | 38.6 | 2.8e-4 | 2.9e-4 |
| Triton IEEE | 13.5 | 2.7e-6 | 1.2e-6 |
| Triton tf32x3 | 8.4 | 4.9e-7 | 2.6e-7 |

The survey assumed from third-party GeForce figures that dense TF32 runs at the FP32 rate. That is wrong
for this workstation card: TF32 runs at 2.4× FP32 here. Even so, three TF32 products have a ceiling of
38.6 / 3 ≈ 12.9 TFLOPS, below cuBLAS FP32's 16.3. **No 3×TF32 kernel can beat cuBLAS FP32 on sm_89.**

### Production shapes: projected GEMM time per force call (µs)

Each shape's measured time is multiplied by its invocations per force call.

| Structure | GEMM group | cuBLAS IEEE | cuBLAS TF32 | Triton IEEE | Triton tf32x3 |
| --- | --- | ---: | ---: | ---: | ---: |
| NaCl | atom-level linear layers | 1,856 | 1,250 | 2,561 (1.38×) | 3,295 (1.78×) |
| NaCl | radial folded last layer | 144 | 142 | 333 | 541 |
| NaCl | radial other GEMMs | 199 | 175 | 380 | 396 |
| NbCl5 | atom-level linear layers | 10,433 | 4,680 | 11,194 (1.07×) | 15,573 (1.49×) |
| NbCl5 | radial folded last layer | 1,067 | 1,040 | 2,119 | 3,432 |
| NbCl5 | radial other GEMMs | 818 | 623 | 1,050 | 1,164 |

- tf32x3 is faster than cuBLAS IEEE on 3 of 92 shapes, all tiny (e.g. `[56320, 64] @ [64, 64]`).
- Its worst error over all shapes, 5.3e-7, is smaller than cuBLAS IEEE's 1.3e-6.
- A tf32x3 kernel replayed in a CUDA graph equals its eager result bitwise.

## 3. Decision and what it means for the round

- **OPT-T01 is closed on sm_89, and no model-level screen is run.** CUTLASS 3×TF32 is not built: it is the
  same arithmetic with the same ceiling. Making a mode the default would be the owner's decision. There
  is none to propose.
- **The local sm_120 rerun is still to do** (`results/opt-T01-gemm-precision-20261020/run.sh`, about 10
  minutes), for the record. It could change the outcome only if sm_120's TF32 rate were well above 3×
  its FP32 rate.
- **Triton IEEE against cuBLAS** decides OPT-T02 stage B (`docs/design/opt-T02-fusion.md`):
  - On NbCl5's atom-level GEMMs a plain Triton IEEE GEMM is 7 % slower; its largest shapes are within
    3–4 %, and some are faster.
  - On NaCl's small-M GEMMs it is 38 % slower.
  - Stage B's ≤ 2 % criterion is therefore not met by this kernel. That question goes to the conductor
    with stage A's trace.
- **Where the GEMM time can still go:** on sm_89, cuBLAS runs NbCl5's large atom-level shapes
  (`[5832–13608, 512] @ [512, 512]`) at 12.5–13.0 TFLOPS, 77–80 % of its own 16.3 TFLOPS peak at 4096³.
  On sm_120, 2-T20a measures them at 12.6–15.1 TFLOPS.
  - The remaining GEMM gain has to come from less arithmetic or less traffic (OPT-T02, OPT-T04).
  - On small cells it can come from fuller grids (OPT-T03).

## 4. A100 rerun (Colab, 23 October 2026)

Same tool and shapes as section 2 (2-T20a's `gemms.csv` for NaCl and NbCl5), on a Google Colab
A100-SXM4-40GB (sm_80), driver 580.82.07, torch 2.14.1+cu130, Triton 3.8.0, FastMACE `5c12ee3`. A fresh VM
was used and released after the run; it used 1.19 compute units. The driver scripts and raw data are in
`results/opt-T01-gemm-precision-20261020/colab-a100/`.

| | cuBLAS IEEE | cuBLAS 1×TF32 | Triton IEEE | Triton tf32x3 |
| --- | ---: | ---: | ---: | ---: |
| Peak 4096³, TFLOPS | 14.8 | **132.2** | 18.2 | 16.8 |
| NaCl GEMMs per force call, µs | 3,911 | 3,221 | 7,099 | 7,776 |
| NbCl5 GEMMs per force call, µs | 10,946 | 4,383 | 14,935 | 16,929 |
| Worst error vs float64 | 1.1e-6 | 4.2e-4 | 2.1e-6 | 5.9e-7 |

- **Triton's tf32x3 is faster on 0 of 92 shapes.** The capture check is bitwise.
- **Even at 4096³ it reaches only 16.8 TFLOPS**, while cuBLAS TF32 runs at 132. The ceiling for three TF32
  products is about 44 TFLOPS, so Triton's 3×TF32 kernel with the screen's fixed tiles does not use the A100's
  tensor cores well. The limit is the kernel, not the hardware. On sm_80, unlike sm_89 and sm_120, a well-built
  3×TF32 kernel could beat FP32 in principle.
- **Production shapes are the real test.** cuBLAS 1×TF32 takes NbCl5's GEMMs from 10.9 to 4.4 ms, but NaCl's
  only from 3.9 to 3.2 ms, because its small shapes are not compute-bound. A fused 3×TF32 kernel, which shares
  its operand loads between the three MMAs, would land between 1× and 3× the TF32 time: 4.4–13 ms on NbCl5.
  Only a measurement decides it.
- **Next test, not yet run:** CUTLASS's 3×TF32 (`OpMultiplyAddFastF32`, example 27, Sm80 kernels) on the same
  shapes. It is the open item of the 23 October research prompt
  (ResearchModule `studies/research_prompts/2026-10-23-fp32-tensor-core-research.md`).
- **Also measured:** cuBLAS IEEE on this A100 is slower than the local RTX 5060 Ti on NaCl's small GEMMs
  (3.9 against 2.6 ms per force call) and faster on NbCl5's (10.9 against 12.5 ms).
