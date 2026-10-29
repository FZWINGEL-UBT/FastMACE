# TF32 screen for MH-1 MD (29 October 2026)

The owner asked on 29 October 2026 whether FastMACE's MH-1 MD could use TF32 after all, on one idle workstation GPU.
TF32 has been off since the owner's production decision of 9 October 2026: hybrid FP32 with TF32 off, on accuracy
grounds. This is a measurement only. **No default changes.**

**Result.**
- **Plain TF32 is much faster: 17–32 % per MD step on the RTX 4000 Ada.**
- **It fails the force gate by 3,500–5,100×.**
- **No scheme gives back FP32-class accuracy and still beats FP32 on this GPU:**
  - 3×TF32 passes the gate but is 12–22 % slower than FP32.
  - cuBLAS's BF16×9 emulation does nothing on sm_89.
  - No single GEMM group tolerates TF32 on its own (OPT4-T08).
- **Nothing to propose for adoption.** The 9 October decision stands on the evidence.

| Structure (atoms) | FP32 ms/step | TF32 ms/step | TF32 / FP32 (ab, ba) | Gate force ratio, TF32 (limit 2) | 3×TF32 / FP32 (ab, ba) |
| --- | ---: | ---: | ---: | ---: | ---: |
| NaCl (216) | 2.48–2.51 | 1.70–1.71 | **0.686, 0.681** | 5,138 | 1.218, 1.221 |
| NaNbO3 (1,080) | 15.84–15.88 | 13.08–13.11 | **0.826, 0.826** | 3,482 | 1.125, 1.128 |
| NbCl5 (1,944) | 20.29–20.32 | 16.71–16.73 | **0.825, 0.822** | 3,665 | 1.202, 1.204 |

## 1. Setup

- **GPU and machine:**
  - leeloo, NVIDIA RTX 4000 Ada Generation (sm_89), driver 595.99.02.
  - The GPU had no compute process and no other user had an active job, checked before every job.
  - murphy was taken by another of our sessions (the user moved the run to leeloo).
  - The conductor confirmed leeloo is within the three-workstation-GPU cap.
- **Environment:** NAS `mace_cueq_oeq`, the hybrid environment: torch 2.14.1+cu130, CUDA 13.0, Triton 3.8.0. The
  references used `mace_cueq_latest`.
- **Model and plan:**
  - MH-1 (`mh1-r2scan`, `mace-mh-1.model`), hybrid cuEq/OEq, float32 forces with float64 energy sums.
  - The minimal planner's default plan, NVT at 300 K, 0.5 fs.
  - Structures are the 1a-T18 baseline export's r2SCAN cells, 3×3×3.
- **Commits:**
  - FastMACE `fastmace/tf32-screen-20261029`: the speed, control, breakdown, gate and NVE runs at `eaa45e24`; the 3×TF32
    phase at `3412d7e3`. The archive's `.source_commit` names them per job in `run.log`.
  - The research engine is not involved: no baseline A/B.
- **Times:** 20:47–22:06 CET (19:47–21:06 UTC).
- **Raw records:** `results/tf32-screen-20261029/`, written by the run script `ada.sh` with its phases speed, control,
  gemms, acc and x3. The values `.npz` files, the job logs and the kernel sequences stay on the NAS under
  `/mnt/homes/zwingel/fastmace-work/tf32-screen/run`.

### How TF32 was switched on

FastMACE itself never sets PyTorch's matmul flag, so the tools gained screen-only switches:
- `tools/ab_rewrites.py --a-tf32/--b-tf32`, and `--a-recapture/--b-recapture` as their control;
- `tools/force_stress_gate.py --production-tf32`;
- `tools/bench/nve_drift.py --tf32`;
- `tools/step_breakdown.py --tf32`.

All of them call `tools/ab_step_time.py`'s `tf32_after_build`.

**First finding: under TF32 the default plan does not build.** The rewrite self-checks (SPEC §6.4) compare each
rewrite with the original module at a tolerance of 1e-4. Under TF32 they fail on TF32's own rounding:
- `factor_species`: relative error 4.4e-4 to 1.1e-3;
- `block_linear`: 1.6e-4 to 4.8e-4. This was measured with `Registry.selfcheck` itself wrapped in IEEE, so
  something in the reference path keeps TF32 rounding.
- Rule 13 would then re-plan without most of the default rewrites.
- Record: `build-under-tf32-selfcheck-failures.txt`.

**The method the screen therefore uses:**
- Each engine is built in IEEE FP32, as production builds it, so every self-check passes.
- The tool then sets `torch.backends.cuda.matmul.fp32_precision = "tf32"` and drops the engine's CUDA graphs
  (`invalidate_graphs`). They re-capture on the next warm-up step with TF32 cuBLAS kernels.
- On this GPU Inductor sends the model's GEMMs to cuBLAS ("Not enough SMs to use max_autotune_gemm"), so this
  reaches every atom-level GEMM.
- It does not reach FastMACE's own Triton kernels, whose `tl.dot` is IEEE explicitly: the radial MLP and the gated
  products. Nor does it reach OEq's convolution or cuEquivariance's symmetric contraction.
- In an A/B the TF32 side sets its mode before each of its steps; the IEEE side sets IEEE.

**Control.** NaCl, with side b's graphs dropped and re-captured in IEEE by the same path: b/a 1.0036 (ab) and 1.0014
(ba). The re-capture itself costs nothing; the gain below is TF32's.

## 2. Speed: whole MD step

- **Protocol:** same process, 300 warm-up steps, 12 alternating blocks of 300 steps, ABAB for build order ab and ABBA
  for ba. a is IEEE (production), b is TF32.
- **Validity:** every run is valid. Nothing compiled while timed, no other GPU process ran, and each side had the
  same neighbour-list rebuild count.
- **TF32 is active:**
  - The TF32 side's potential energy after the run differs from IEEE's by 0.1–0.2 meV/atom.
  - The step breakdown shows `ampere_sgemm_*` replaced by `cutlass_80_tensorop_*` kernels.

| Structure | Order | a (IEEE), ms | b (TF32), ms | b/a | Per-pair range |
| --- | --- | ---: | ---: | ---: | --- |
| NaCl | ab | 2.4810 | 1.7019 | 0.6860 | 0.679–0.723 |
| NaCl | ba | 2.5107 | 1.7103 | 0.6812 | 0.679–0.714 |
| NaNbO3 | ab | 15.8778 | 13.1112 | 0.8258 | 0.825–0.828 |
| NaNbO3 | ba | 15.8405 | 13.0842 | 0.8260 | 0.821–0.826 |
| NbCl5 | ab | 20.2887 | 16.7274 | 0.8245 | 0.821–0.825 |
| NbCl5 | ba | 20.3240 | 16.7109 | 0.8222 | 0.818–0.823 |

The research engine's 8 October TF32 study, separate processes on the RTX 5060 Ti, had a 19.9 % shorter force call on
NaCl and 7.1 % on NaNbO3. FastMACE's step is GEMM-bound to a larger share since the rewrites, so TF32 gains more here.

## 3. Per-GEMM breakdown

**Method:**
- `tools/step_breakdown.py` main pass with and without `--tf32`: 300 timed steps and 10 profiled eager force calls.
- Kernel time is attributed by operator and shape (`sb-fp32/`, `sb-tf32/`); the analysis script is
  `gemm_breakdown.py`.

**Where the force call's eager kernel time goes (ms per force call):**

| Category | NaCl FP32 → TF32 | NaNbO3 FP32 → TF32 | NbCl5 FP32 → TF32 |
| --- | --- | --- | --- |
| Atom-level linear layers (GEMM) | 1.253 → 0.613 (0.49×) | 4.247 → 1.941 (0.46×) | 5.540 → 2.360 (0.43×) |
| Edge elementwise | 0.543 → 0.551 | 5.830 → 5.645 | 3.959 → 3.838 |
| Atom-level elementwise | 0.311 → 0.297 | 1.970 → 1.764 | 4.001 → 3.770 |
| cuEq symmetric contraction | 0.086 → 0.083 | 0.774 → 0.724 | 1.418 → 1.408 |
| Total kernel time | 2.268 → 1.620 | 15.102 → 12.252 | 19.913 → 16.253 |

- **The atom-level GEMMs give 98 % (NaCl), 81 % (NaNbO3) and 87 % (NbCl5) of the saving.** The rest is a few
  percent off the elementwise kernels around them. Stage B, the graph replay, shrinks by about as much as the eager
  kernel time: 2.48 → 1.66, 15.61 → 12.82 and 20.08 → 16.49 ms per step.
- **The largest single wins** are the `[N, 512] @ [512, 512]` linears (`linear_up`, `linear_1`, `linear_2`, the product
  and residual linears) and their input gradients.
  - On NbCl5, `[5832, 512] @ [512, 512]` goes from 1,084 to 366 µs per force call (0.34×).
  - The `K = 2048` products `[1944, 2048] @ [2048, 384]` go from 259 to 83 µs.
  - Small-M products such as NaCl's `[216, 512] @ [512, 512]` gain about 2×.
- **Shape replay** (`tools/gemm_precision_bench.py` on the FP32 pass's 111 shapes, `gemm-bench/`), cuBLAS IEEE →
  cuBLAS TF32 per force call:

  | Structure | cuBLAS IEEE, µs | cuBLAS TF32, µs | Triton tf32x3, µs |
  | --- | ---: | ---: | ---: |
  | NaCl | 1,478 | 892 | 2,549 |
  | NaNbO3 | 4,105 | 2,026 | 6,028 |
  | NbCl5 | 5,170 | 2,281 | 7,568 |

  - Peak at 4096³: IEEE 16.0, TF32 41.2 and Triton tf32x3 8.4 TFLOPS. TF32 runs at 2.6× FP32 here, so three
    TF32 products (13.7 TFLOPS at best) cannot beat SIMT FP32. This confirms OPT-T01 §2 on murphy.
  - Worst error against float64: IEEE 1.5e-6, TF32 4.5e-4, tf32x3 5.7e-7. tf32x3 is faster than cuBLAS IEEE on 3 of
    111 shapes, all tiny.

## 4. Accuracy of plain TF32

### Force, stress and energy gate

**Setup:**
- `tools/force_stress_gate.py --engine fastmace --production-tf32` against the IEEE control run the same evening.
- 3 rattled and 3 strained copies per structure. The float64 reference and the float32 yardstick ran in separate
  processes.
- Rule: at most 2× the float32 calculator's error against float64.

| Structure | Quantity | TF32 max error | Yardstick max error | TF32 ratio | IEEE ratio (control) |
| --- | --- | ---: | ---: | ---: | ---: |
| NaCl | forces, eV/Å | 0.0202 | 3.9e-6 | **5,138** | 0.93 |
| NaCl | stress, eV/Å³ | 2.9e-4 | 4.1e-8 | **7,211** | 0.97 |
| NaCl | energy, meV/atom | 0.630 | 1.1e-4 | **5,822** | 0.70 |
| NbCl5 | forces | 0.295 | 8.0e-5 | **3,665** | 1.03 |
| NbCl5 | stress | 2.3e-4 | 9.5e-8 | **2,378** | 0.35 |
| NbCl5 | energy | 0.453 | 7.8e-4 | **583** | 0.12 |
| NaNbO3 | forces | 0.466 | 1.3e-4 | **3,482** | 1.00 |
| NaNbO3 | stress | 1.8e-3 | 3.1e-7 | **5,847** | 0.54 |
| NaNbO3 | energy | 0.810 | 1.1e-3 | **749** | 0.30 |

- **RMS force error against float64 (`gate_rms.py`):**

  | Structure | TF32, eV/Å | Yardstick, eV/Å | Ratio |
  | --- | ---: | ---: | ---: |
  | NaCl | 3.3e-3 | 8.8e-7 | 3,800× |
  | NbCl5 | 1.6e-2 | 1.0e-5 | 1,600× |
  | NaNbO3 | 3.6e-2 | 1.2e-5 | 3,000× |

  The IEEE control is at 1.0× on all three.
- **The error is systematic, not a few outliers.**
- **It is about 3–6× the research engine's TF32 error** on the same structures (8 October: RMS 9.7e-4 on NaCl, 6e-3
  on NaNbO3; max error/FP32 error 1,020–1,500× and 301–327×).
  - The likely cause is FastMACE's folded and tabulated rewrites, whose products carry larger operands into the
    TF32 rounding.
  - This was not isolated further: the gate fails either way.

### NVE drift

**Setup:** `tools/bench/nve_drift.py`, the default plan's NVE, 20 ps at 1 fs from 300 K, sampled every 10 steps.

| Structure | Mode | Drift, meV/atom/ps | RMS about the fit, meV/atom | Range, meV/atom |
| --- | --- | ---: | ---: | ---: |
| NaCl | IEEE | 7.9e-7 | 2.7e-4 | 0.0070 |
| NaCl | TF32 | −8.9e-6 | 1.19e-2 | 0.234 |
| NbCl5 | IEEE | −1.9e-6 | 3.8e-4 | 0.0127 |
| NbCl5 | TF32 | 8.6e-5 | 5.1e-3 | 0.151 |

- **Drift stays negligible** under TF32: below 1e-4 meV/atom/ps.
- **The conserved quantity fluctuates 13–43× more.** That is the force noise of the gate, seen in the dynamics.
- Short NVE alone would not have caught what the gate shows.

## 5. Accuracy-recovered TF32

Plain TF32 fails the gate and is clearly faster, so the brief's three schemes were screened.

| Scheme | Available on sm_89 | Speed against FP32 | Against plain TF32 | Gate |
| --- | --- | --- | --- | --- |
| 3×TF32, Triton (`RewriteConfig.tf32x3_gemms=on`, `grouped_launch=off`) | yes | **slower**: 1.218 / 1.221 (NaCl), 1.125 / 1.128 (NaNbO3), 1.202 / 1.204 (NbCl5), same-process A/Bs in both orders | 1.37–1.82× slower (NaCl 3.10 / 1.70, NaNbO3 17.93 / 13.10, NbCl5 24.47 / 16.72 ms) | **passes**: force ratios 0.98, 1.05, 1.02; stress 1.14, 0.29, 0.75 |
| 3×TF32, CUTLASS (`OpMultiplyAddFastF32`) | builds for Sm80 | not built: same arithmetic, ceiling 41.2 / 3 = 13.7 TFLOPS < cuBLAS FP32's 16.0 | – | (as Triton's: FP32-class, OPT-T01 §5) |
| cuBLAS BF16×9 FP32 emulation (`CUBLAS_EMULATE_SINGLE_PRECISION=1`) | **no** | no effect | – | – |
| TF32 in the tolerated GEMMs only | no GEMM group qualifies | – | – | – |

**3×TF32:**
- The FP32 side's plan has `grouped_launch` off on this GPU already (decision 66: MH-1 has not been measured on it),
  so b differs from a only by the record.
- The plan lines are in `x3-*.json`. Gate: `gate-x3/`.

**BF16×9:**
- `bf16x9_probe.py` ran the same SGEMM with and without the environment variable in fresh processes.
- Both ran the same `ampere_sgemm_64x64_nn` kernel at the same rate (17.0 / 16.8 TFLOPS at 4096³), with bitwise-equal
  results (`bf16x9-{off,on}.json`).
- cuBLAS documents the emulation for compute capability 10.0 and 10.3 only. That holds on cuBLAS 13.0 here.

**Selective TF32, ranked by error sensitivity:**
- OPT4-T08 (`docs/validation/opt4-T08-precision-sensitivity.md`) measured each stage of the force call with TF32 in
  that stage alone and every other stage in FP32, forward and backward separately, through the gate.
- The worst force ratio over NaCl, NbCl5 and NaNbO3 for the stages with GEMMs that cuBLAS runs in production:

  | Stage (production GEMM) | TF32 forward | TF32 backward |
  | --- | ---: | ---: |
  | i1 product linear | 2.60 (passes only through NaCl's force floor) | 5.46 |
  | i1 readout | 3.07 | 24.3 |
  | i0 readout | 1.03 (energy fails) | 34.0 |
  | i1 skip / residual | 27.3 | 6.16 |
  | i0 product linear | 23.1 | 16.4 |
  | i1 linears (`linear_up`, `linear_1`, `linear_2`) | 35.2 | 17.3 |
  | i0 linears | 51.2 | 23.5 |
  | i0 / i1 radial MLP (FP32 Triton in production, not TF32-reachable) | 599 / 476 | 44.7 / 25.0 |

  - The only stages that pass are i0's symmetric-contraction backward (1.25) and, on forces alone, i1's conv-TP
    backward (1.95, but stress 3.42). Neither is a cuBLAS GEMM in production: cuEquivariance's contraction kernel
    and OEq's streaming convolution have no TF32 switch.
- **The stages with real GEMM time fail the gate alone by 17–51×.** These are the 512-wide linears, which carry most
  of section 3's saving.
- The small ones are closest: i1's product linear, the readouts. Even they fail, and their GEMMs are among the
  smallest of section 3's shapes, so TF32 there could buy little.
- Production already gates at force ratio 0.93–1.03 (§4 control), so there is no headroom to spend.

The OPT4-T08 numbers are from its e3nn stand-in path (24 October 2026, local RTX 5060 Ti). Section 4 shows the production
path is more sensitive to TF32 than the research engine, not less. So the ranking stands and was not re-measured.

## 6. Conclusion and decision points

- **TF32 would be the largest single speed-up still on the table on Ada:** 0.68× NaCl, 0.83× NaNbO3 and NbCl5. The
  gain is almost all in the atom-level linear GEMMs.
- **It fails the force gate by three to four orders of magnitude.** Its NVE energy noise is 13–43× FP32's, and the
  default plan's own self-checks refuse it at build time.
- **No accuracy-recovered scheme beats FP32 on sm_89:**
  - 3×TF32 is accurate but 12–22 % slower;
  - BF16×9 does not exist below compute capability 10.0;
  - no GEMM group tolerates TF32 on its own.
- **Nothing becomes default.** The 9 October precision decision is the owner's and is unchanged.
- **The tools' TF32 switches** are measurement options for screens only; production never sets the flag.
- **The owner decides** whether a reduced-accuracy MD mode, for screening rather than production trajectories, is
  wanted at all. Its error would be about 3e-3 to 4e-2 eV/Å RMS on these cells. That is SPEC §1.2's "TF32 never
  default" and a §18 question, not something this screen can settle.
