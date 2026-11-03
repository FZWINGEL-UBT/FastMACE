# BF16×3 atom GEMMs (FACTO, arXiv 2611.03276): accuracy and speed (3 November 2026)

**Question (owner, 3 November 2026).** FACTO (ByteDance) runs its dense atom projections as BF16×3 — each float32
operand split into two bfloat16 parts, the three products hi·hi + hi·lo + lo·hi on BF16 tensor cores with float32
accumulation — and reports MD stability close to float32 (NVE fluctuation 4.14 against 4.28 µeV/atom; TF32 31.7)
and a 1.34× speed-up for its largest model. Does it hold for MACE in FastMACE, and is it faster on our GPUs?

**Result: rejected on all three GPUs measured. It stays within the force gate but is slower than the default
plan.**
BF16×3 keeps about 16 mantissa bits: per product it is 7–15× (median) less accurate than float32, and the gate
passes with less margin (force ratio up to 1.55, energy errors 5–20× the default plan's). Whole MD steps are 8–24 %
slower on the RTX 5060 Ti and the RTX 4000 Ada. On the A100, NaCl is 29 % slower and NbCl5 0.6 % faster, and the
gate there passes on NaCl only through its absolute floors (energy ratio 4.4, stress 2.2). FACTO's gain needs BF16 tensor-core rates far above
float32 (H-class GPUs: about 15×); ours have 3× (sm_120) and 5× (sm_89). The A100 (13.6×) is the one GPU where the
split GEMMs themselves win (-25 % at 13608×512×512), and the step still does not.

## 1. What ran

- **Source:** branch `fastmace/bf16x3-screen` (`4b3b9cbf`, `e00ed09f`, `1d50496c`) from main `7430fb51`.
  `FASTMACE_SPLIT_PRECISION` (environment, read at import) sets the `tl.dot` precision of the opt-in
  `tf32x3_gemms` record's GEMMs (`tf32x3_gemm` and `segment_gemm`): `tf32x3` (default), `bf16x3` or `bf16x6`.
  The record needs `grouped_launch=off` (both reroute the block linears). No change to any default.
- **Model and cells:** MH-1, hybrid float32, NaCl and NbCl5 3×3×3 (the opt4-T03 setup).
- **Tools:**
  - `tools/tf32x3_accuracy.py`: every rerouted product of one force call on real operands, against cuBLAS IEEE
    on the same float32 operands, both against float64.
  - `tools/force_stress_gate.py`: the production rule (error ≤ 2× the float32 calculator's against float64).
  - the GEMM probe of OPT-T01 §6 (`results/opt-T01-gemm-precision-20261020/colab-bf16x6/a100/probe_bf16x6.py`).
  - `tools/ab_rewrites.py`: same-process A/B, default plan (a) against `tf32x3_gemms=on grouped_launch=off` in
    bf16x3 (b), 1 fs, 300 K, 12 blocks of 300 steps, both build orders.
- **Runners:** `results/opt4-T03/local.sh` (accuracy, gates), `results/opt4-T03/bf16x3_speed.sh` (probe, A/B),
  `tools/modal/bf16x3_a100.sh` (all on the Modal A100).
- **Hardware:** local RTX 5060 Ti (sm_120, WSL2, driver 610.88); leto RTX 4000 Ada (sm_89, driver 595.99, idle
  check); Modal A100-SXM4-40GB (driver 610.57, job `1d50496c`, 25 min, about $1.22 metered). Environment `mace_cueq_oeq` (torch 2.14.1+cu130).

## 2. Accuracy

**Per product (5060 Ti, 52 products of one force call):**

| Structure | bf16x3 max rel. | IEEE max rel. | bf16x3 / IEEE: median, max | worse on | bf16x6 / IEEE: median, max |
| --- | --- | --- | --- | --- | --- |
| NaCl | 1.6e-5 | 9.1e-7 | 14.6, 58 | 48 of 52 | 0.50, 1.97 |
| NbCl5 | 1.4e-5 | 1.8e-6 | 6.8, 41 | 48 of 52 | 0.25, 1.28 |

BF16×3 is not float32-class (about 2⁻¹⁶ relative: the lo·lo term and the second split are missing); BF16×6 is.
In the GEMM probe under cancellation, bf16x3's worst error is 9.9e-3 against cuBLAS IEEE's 4.8e-5.

**Force gate (5060 Ti; ratio = production error / float32 calculator's error, limit 2):**

| Structure | Plan | Energy (eV) | Forces (eV/Å) | Stress (eV/Å³) | Result |
| --- | --- | --- | --- | --- | --- |
| NaCl | default | 1.2e-4 (0.16) | 3.6e-6 (0.95) | 3.6e-8 (0.39) | passed |
| NaCl | tf32x3 | 2.4e-4 (0.31) | 3.7e-6 (0.97) | 4.0e-8 (0.44) | passed |
| NaCl | **bf16x3** | 6.3e-4 (0.82) | 5.8e-6 (**1.55**) | 7.0e-8 (0.74) | passed |
| NbCl5 | default | 5.7e-5 (0.08) | 8.3e-5 (1.05) | 3.9e-8 (0.40) | passed |
| NbCl5 | tf32x3 | 8.8e-5 (0.12) | 8.4e-5 (1.04) | 2.8e-8 (0.28) | passed |
| NbCl5 | **bf16x3** | 1.15e-3 (**1.61**) | 9.1e-5 (1.14) | 1.7e-7 (**1.66**) | passed |

**A100 (Modal):** bf16x3 NaCl energy 8.5e-4 eV (ratio **4.38**), forces 5.8e-6 (1.42), stress 7.5e-8 (**2.20**):
passed only because both are below the gate's absolute floors (0.01 eV, 1e-7); NbCl5 ratios 1.47 / 1.17 / 1.78.
The default plan there: NaCl 0.25 / 0.91 / 1.02, NbCl5 0.14 / 1.01 / 0.35. Per product as on the 5060 Ti
(bf16x3 / IEEE median 19.2 and 7.7, worse on 48 of 52).

So FACTO's observation carries over in the narrow sense: BF16×3 stays within the production rule. But it uses most
of the rule's margin (NbCl5 energy and stress 1.6–1.7 of 2), where the default plan sits at 0.1–1.05.

## 3. Speed

**GEMM probe, µs (fastest of 7 Triton tiles; cuBLAS heuristics):**

| GPU | Shape | cuBLAS IEEE | cuBLAS 1×BF16 | Triton tf32x3 | Triton bf16x3 | bf16x3 / IEEE |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 5060 Ti | 4096³ | 8,316 | 2,782 | 19,455 | 13,678 | 1.64 |
| RTX 5060 Ti | 1512×512×512 | 70 | 26 | 144 | 90 | 1.28 |
| RTX 5060 Ti | 13608×512×512 | 505 | 165 | 1,058 | 620 | 1.23 |
| RTX 4000 Ada | 4096³ | 8,603 | 1,672 | 16,302 | 11,053 | 1.28 |
| RTX 4000 Ada | 1512×512×512 | 55 | 18 | 106 | 77 | 1.39 |
| RTX 4000 Ada | 13608×512×512 | 536 | 99 | 794 | 594 | 1.11 |
| A100 | 4096³ | 7,245 | 533 | 8,167 | 4,763 | 0.66 |
| A100 | 1512×512×512 | 86 | 32 | 123 | 84 | 0.98 |
| A100 | 13608×512×512 | 407 | 54 | 483 | 306 | 0.75 |

The dense BF16 rate is 3.0× cuBLAS float32 at 4096³ on the 5060 Ti, 5.1× on the Ada and 13.6× on the A100. Only
the A100 leaves room for three products and the split; there Triton's bf16x3 GEMMs take 0.66–0.98× of cuBLAS IEEE's
time, which the whole step loses again by giving up the grouped launch (NaCl) or nearly so (NbCl5).

**Whole MD step (median ms; b/a > 1: bf16x3 slower):**

| GPU | Structure | a default | b bf16x3 | b/a (order ab) | b/a (order ba) |
| --- | --- | --- | --- | --- | --- |
| RTX 5060 Ti | NaCl | 2.485 | 3.058 | 1.231 | 1.245 |
| RTX 5060 Ti | NbCl5 | 17.57 | 20.67 | 1.176 | 1.181 |
| RTX 4000 Ada | NaCl | 2.555 | 2.799 | 1.095 | 1.089 |
| RTX 4000 Ada | NbCl5 | 19.85 | 21.47 | 1.082 | 1.082 |
| A100 (Modal) | NaCl | 2.315 | 2.978 | 1.286 | 1.285 |
| A100 (Modal) | NbCl5 | 12.87 | 12.79 | 0.994 | 0.994 |

All runs valid (no compile or foreign GPU process while timed).

## 4. Library support (docs of CUDA 13.4, searched 3 November 2026)

- **cuBLAS / cuBLASLt:** no BF16×3 mode. `CUBLAS_COMPUTE_32F_FAST_16BF` is a single BF16 product (8 mantissa bits);
  `CUBLAS_COMPUTE_32F_FAST_TF32` a single TF32 one. The float32 emulation `CUBLAS_COMPUTE_32F_EMULATED_16BFX9`
  (BF16×9, exact float32, CUDA 12.9+) runs on compute capability 10.0 and 10.3 only; the docs say it pays only where
  BF16 is more than 9× float32.
- **cuBLASDx 0.7.0+:** `RequiredMantissaBits<X>` emulates float32/float64 GEMMs with ceil((X+1)/8) INT8 slices
  (shared exponent per row of A and column of B); with the operator present, sm_89 and sm_120 always emulate. X can
  be lowered (X = 16: 3 slices, about BF16×3's accuracy). Our own INT8-slice screen on sm_120
  ([prec-int8-f32](../optimisation-ledger.md#prec-int8-f32)) measured 3 slices at 599 µs against SGEMM's 518 µs on
  13608×512×512; the local MathDx is 25.6.0 (before 0.7.0), so cuBLASDx's own kernel is not measured.
- **cuBLASXt:** multi-GPU host-memory BLAS over cuBLAS; nothing new for this.

## 5. Records

`results/bf16x3-facto-20261103/`: `rtx5060ti-local/accuracy-gates/{bf16x3,bf16x6,tf32x3,default}/` and
`rtx5060ti-local/speed/` (probe, A/B both orders), `rtx4000ada-leto/leto-speed/`, `modal-a100/`.
