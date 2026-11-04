# OMAT-0 kernel lab and follow-up screens (3–4 November 2026)

Owner request (3 November 2026): "Can you try to come up with more ideas to speed up omat-0 on a100 (test ideally on
workstation first)". The A100 profile (`results/omat0-a100-profile-20261103`, A100-SXM4-80GB) puts 65 % of the
Benchmark_loup force call (MACE-OMAT-0 medium, SrTiO3 10×10×10, 5,000 atoms, 373,760 edges, float32 default plan)
in two kernels: the wprime backward (6.92 ms) and the fused folded forward (5.06 ms). The launch-setting screen on
the A100 (`results/omat0-a100-launch-screen-20261103`) found nothing. This round looks inside the two kernels.

Everything here ran on the workstation leto (RTX 4000 Ada, sm_89, `mace_cueq_oeq`, after the idle check) and the
local PC (RTX 5060 Ti, sm_120, WSL `mace_cueq_oeq`); the winners then on the A100 (Modal, section 4).

## 1. Kernel lab (`tools/omat0_kernel_lab.py`)

The lab captures one eager force call's arguments of both kernels in the default plan (the largest eager call,
373,760 edges) and times rendered-source variants round robin with CUDA events (15 rounds, median). Exact variants
are compared with the base right after their warm-up: `out`, `w`, `dsh`, `dr` bit for bit, `dx` (atomic sums) to
a relative tolerance. Falsifiers (`F-…`) drop one part of the work and give a bound, not a result.

Base kernels: wprime 10.47 ms (RTX 5060 Ti, 80 registers) and 10.33 ms (RTX 4000 Ada, 96 registers); fused forward
8.70 ms (168 registers) and 8.67 ms (168 registers, 40 bytes of local memory on sm_89). Ratios are variant / base.

| Kernel | Variant | What | RTX 5060 Ti | RTX 4000 Ada | Check |
|---|---|---|---|---|---|
| wprime | qmajor-f2 | derivative table q-major, float2 term loads | 0.913 | – | dsh, dr bitwise |
| wprime | qmajor-f4 | the same, float4 (TERMS padded to 12) | 0.974 | – | dsh, dr bitwise |
| wprime | qmajor-f2+ldcs-w | and the stored w read with `__ldcs` | 1.013 | – | dsh, dr bitwise |
| wprime | ldcs-w | the stored w read with `__ldcs` (evict-first) | 0.964–0.966 | 1.127 | dsh, dr bitwise |
| wprime | unordered | mirror species pairs read one slab | 0.980–0.982 | 0.991 | dsh, dr bitwise |
| wprime | evict-last-g | L2 evict-last policy on the derivative table | 1.105 | 1.140 | dsh, dr bitwise |
| wprime | hints | both cache hints | 1.015 | 1.192 | dsh, dr bitwise |
| wprime | unordered+hints | | 1.005 | 1.167 | dsh, dr bitwise |
| wprime | maxreg64 / maxreg80 | register caps | 1.508 / 1.308 | 1.442 / 1.108 | spills (120 / 48 bytes) |
| wprime | F-no-slope | no derivative-table dot | 0.597 | 0.734 | falsifier |
| wprime | F-half-terms | 5 of 10 terms | 0.805 | 0.770 | falsifier |
| wprime | F-one-slab | every edge reads group 0 | 1.073 | 1.029 | falsifier |
| wprime | F-no-atomics | plain stores for dx | 1.101 | 0.999 | falsifier |
| wprime | F-no-wload | w not loaded | 0.858 | 0.912 | falsifier |
| forward | stcs | w stored with `__stcs` | 0.995 | 0.965 | out, w bitwise |
| forward | ldcs-a | a read with `__ldcs` | 1.036 | 0.989 | out, w bitwise |
| forward | k64 | the k-loop skips W's all-zero last column | 1.010 | 0.909 | out, w bitwise |
| forward | k64+stcs | both | 1.000 | 0.896 | out, w bitwise |
| forward | F-no-store | w not stored | 0.787 | 0.871 | falsifier |
| forward | F-no-gemm | no a·W product | 0.997 | 0.893 | falsifier |
| forward | F-no-tp | no tensor product | 0.978 | 0.900 | falsifier |
| forward | F-gemm-only | the a·W product alone | 0.356 | 0.193 | falsifier |

Reading:

- **wprime** is bound by load instructions in L1, not by DRAM or the atomics. The slope (the 10-term derivative-table
  dot per weight) is 27–40 % of the kernel, and each term about 2–5 %. One slab for every edge is not faster, so
  the table's L2 locality is not the limit. Fewer load instructions help: float2 vector loads give −8.7 % on the
  RTX 5060 Ti. Fewer terms help on both GPUs. Cache hints help on sm_120 and hurt on sm_89.
- **The fused forward** is bound by the 1.9 GB w store on the RTX 5060 Ti. On the RTX 4000 Ada the parts overlap.
  Dropping W's all-zero last column there (OMAT-0's last radial layer has no bias) gives −9.1 %, more than the
  1/65 of the product it saves; the loop and staging arithmetic simplify. On the A100 (FP32 SIMT 19.5 TFLOPS,
  2 TB/s) the a·W product (62 GFLOP per call) should dominate the kernel instead.

Files: `rtx5060ti-wprime/lab.log` (its `lab.json` was not written: the run crashed with `CUDA driver error: device
not ready` when the forward variants each allocated a 1.9 GB w; fixed in the tool), `rtx5060ti-forward/`,
`rtx5060ti-qmajor/`, `rtx4000ada-leto/` (with the idle check), and the generated base sources (`*.cu`).

## 2. Switches built from the lab

All are module constants, off by default (A/B switches; `tests/cpu/test_layering.py` lists them), and bitwise
unless stated:

| Switch | Module | Effect | Tests |
|---|---|---|---|
| `SLOPE_VECTOR` 2 / 4 | `kernels/oeq_streaming/wprime_backward.py` | q-major derivative table `[groups, Q, TERMS padded]`, float2/float4 term loads; refused by the rebuild backward (`recompute_conv_weights`) | CPU layout and render; GPU bitwise dr, dsh (leto) |
| `STREAM_W` | same | stored w read with `__ldcs` | – |
| `SKIP_ZERO_COLUMN` | `kernels/oeq_streaming/fused_forward.py` | k-loop skips an all-zero last weight column (`used_columns`), rows keep their stride | CPU render; GPU bitwise |
| `STREAM_STORE_W` | same | w stored with `__stcs` | CPU render; GPU bitwise |
| `SYMMETRIC_PAIRS` | `rewrites/radial/__init__.py` | mirror species pairs share one table row when coef and dcoef rows are bitwise equal (OMAT-0's are) | – |
| `DEGREE` (existing) | same | Chebyshev degree of the radial tables (default 10) | certificate |

Engagement on OMAT-0 medium (local, SrTiO3 10×10×10, 20 MD steps): the fused forward reads 64 of 65 columns,
`symmetric_rows` is true for all four tables, and the wprime derivative table is q-major `[1584, 1280, 10]`.

## 3. MD step screens (`tools/modal/omat0_a100_launch_screen.sh`)

Same-process MD step A/Bs on Benchmark_loup (SrTiO3 10×10×10), default plan on both sides; b/a below 1 is faster.
One order (ab, 6 blocks of 200 steps) unless both orders are given (ab / ba, 8 blocks). Cases: `degN` radial
tables of degree N (default 10); `dN-x`: switch x on top of degree N on both sides; `dzN`: degree N and
`SKIP_ZERO_COLUMN`; `tc`: `TENSOR_CORES`; `dztcN`: dzN and tc; `allN`: degree N, `SLOPE_VECTOR` 2, skip-zero and
`STREAM_STORE_W`; `allnsN`: the same without slopes; `bundle`: degree 6 and slopes2; `tolN-X`: degree N with fit
tolerance X (default 1e-7).

| Case | RTX 4000 Ada (leto) | RTX 5060 Ti (local) | A100-SXM4-80GB (Modal) |
|---|---|---|---|
| deg4 | 0.980 | 0.999 | 1.026 |
| deg5 | 0.908; 0.910 / 0.907 | 0.935 | 0.903 / 0.905 |
| deg6 | 0.934 | – | 0.920 |
| skipzero | 0.989 | 1.000 | 0.991 |
| d5-skipzero | 0.981 | – | 0.986 |
| **dz5** | – | – | **0.896 / 0.895** |
| allns5 | **0.890** | – | – |
| all5 | 0.917 / 0.917 | 0.989 | – |
| slopes2 / slopes4 | 0.991 / 1.028 | 0.983 / – | – |
| d5-slopes2 | 1.027 | 1.070 | 1.040 |
| d6-slopes2 | 0.999 | – | – |
| streamw | – | 0.987 | – |
| stcs / d5-stcs | 1.001 / 1.001 | 0.991 / – | – |
| sympairs | 0.997 | – | – |
| bundle / bundle-skipzero | 0.930 / 0.917 | – | – |
| eos-fc64 / eos-be64 | 1.002 / 1.004 | – | – |
| tc | 1.124 | 1.272 | 1.040 |
| d5-tc | – | – | 1.042 |
| dztc5 | – | 1.090 | 0.910 / 0.911 |
| tcre, dztcre5 | refused | – | refused |
| tol4-2e-7 / tol4-3e-7 / tol5-3e-7 | 0.966 / 0.958 / 0.905 | – | – |

Reading:

- **Degree 5 tables** are the largest gain on every GPU (−7 to −10 %): fewer Chebyshev terms per edge in the radial
  table forward, its backward and the wprime slope. Degree 4 needs too many pieces, degree 6 lies between. A looser
  fit tolerance adds nothing at degree 5 (0.905 against 0.908) and does not rescue degree 4.
- **Skip-zero column** adds 1–2 % on top of degree 5 on the Ada and the A100, nothing on the RTX 5060 Ti.
- **Slope vectors** lose on top of degree 5 everywhere (with 5 terms the q-major float2 loads no longer pay).
- **Tensor cores (3×TF32)** lose on all three GPUs, even on the A100 (1.04): the kernel split every W element into
  its TF32 high and low parts per edge batch, about as much ALU work as the FP32 product it replaced. A version that
  splits W once per block into shared memory followed (section 5).
- **tcre / dztcre5** (`recompute_conv_weights=on`) are refused by the planner: the generic wprime backward needs the
  stored w (`generic_wprime_backward: no generic interaction with float32 radial tables and the chunked streaming
  convolution`). Not pursued.

Files: `screens/` (RTX 4000 Ada with the idle check; RTX 5060 Ti) and `a100-fastest/ab-*`.

## 4. Benchmark_loup on the A100 and the force gate

A100 job (Modal, A100-SXM4-80GB, commit e804b246, 4 November 2026, 47 min, about $2.49; `a100-fastest/`). Its
first step, the fused forward's GPU test body, failed the tensor-core accuracy bound (1.02e-6 of the row scale
against an FP32 chain of 2.8e-7 at k = 65; on the RTX 4000 Ada 9.5e-7). The bound was too tight for the tensor
cores' FP32 accumulation, which does not round to nearest; f49b0421 adds one unit of the row's scale per mma
accumulation (3 per 8 columns), and the test passes on the Ada and the RTX 5060 Ti. The best candidate over both
orders, dz5, then ran the benchmark protocol (`tools/bench/loup_md.py`: 5 warm-up and 20 timed steps, 3 fresh
processes per ensemble):

| Set | NVE µs/atom/step (3 runs) | Langevin µs/atom/step (3 runs) | Median ms/step NVE / Langevin |
|---|---|---|---|
| default | 3.614, 3.606, 3.614 | 3.696, 3.543, 3.541 | 18.07 / 17.71 |
| **dz5** (`radial.DEGREE=5`, `fused_forward.SKIP_ZERO_COLUMN=True`) | **3.172, 3.171, 3.172** | **3.175, 3.365, 3.127** | **15.86 / 15.88** |

dz5 is the fastest Benchmark_loup time so far on the A100: 3.17 µs/atom/step, 12 % below the default plan. Its
step breakdown (`a100-fastest/breakdown`, 16.1 ms of kernels per force call): wprime backward 5.17 ms (6.92 at
degree 10), fused forward 4.91 ms, the 3×TF32 GEMMs 1.87 ms. The kernel lab on the A100 (`a100-fastest/lab`, at
degree 10): the a·W product is 56 % of the fused forward (`F-no-gemm` 0.442; the product alone 0.120), and the
kernel holds 168 registers with 48 bytes of local memory; in wprime, `qmajor-f2+ldcs-w` gives 0.805 (`qmajor-f2`
0.898, `ldcs-w` alone 1.184).

**Force gate** (`tools/force_stress_gate.py --engine fastmace --model omat0-medium`, production against float64
MACE, rule 1b-T16; RTX 5060 Ti, e804b246, `gate-rtx5060ti/`): every set passes all 12 checks on SrTiO3 3×3×3, NaCl
and NbCl5. Error ratios to ASE's float32 calculator (energy / forces / stress):

| Structure | default | dz5 | dztc5 |
|---|---|---|---|
| SrTiO3 3×3×3 | 1.52 / 0.87 / 0.68 | 1.68 / 0.79 / 0.72 | 2.60 / 0.87 / 0.48 |
| NaCl | 1.21 / 0.81 / 0.84 | 1.34 / 0.79 / 0.70 | 0.96 / 0.87 / 0.70 |
| NbCl5 | 0.84 / 0.98 / 0.55 | 0.92 / 0.97 / 0.38 | 0.66 / 0.97 / 1.06 |

Energy errors are far below the gate's 0.01 eV floor in every case. The gate ran on the local GPU because the other
workstations were busy with other users' jobs.

## 5. Tensor cores with W split once per block (not committed)

The tensor-core forward of e804b246 splits every W element into its TF32 high and low parts for each 8-edge batch.
A variant split the block's W slice once, into a high and a low plane in shared memory (the slice doubles), so the
inner loop only loads fragments. It passes the GPU test body (RTX 5060 Ti: 4.4–6.4e-7 of the row scale against FP32
chains of 2.2–3.4e-7, the message bitwise) but the MD step on top of degree 5 is 1.730 on the RTX 5060 Ti
(`presplit-5060`, one order): the doubled slice halves the warps per block. Each W fragment serves one mma triple
of 8 edges, so the tensor-core product needs twice the shared loads of the FP32 path either way; a version that
pays would need larger edge batches per warp. Reverted; `TENSOR_CORES` stays off and rejected.

## 6. Kernel lab at dz5 (RTX 4000 Ada)

`tools/omat0_kernel_lab.py --constant fastmace.rewrites.radial:DEGREE=5 fastmace.kernels.oeq_streaming.fused_forward:SKIP_ZERO_COLUMN=True`
(0130eca2, leto with the idle check; `rtx4000ada-dz5/`), without the q-major variants (they assume an even term count).
Base wprime 7.94 ms, fused forward 7.80 ms (167 registers, 40 bytes of local memory). Ratios to the base:

| Kernel | Variant | Ratio | Check |
|---|---|---|---|
| wprime | ldcs-w | 0.982 | dsh, dr bitwise |
| wprime | unordered | 0.993 | dsh, dr bitwise |
| wprime | maxreg80 | 1.034 | dsh, dr bitwise |
| wprime | F-no-slope / F-half-terms | 0.956 / 1.065 | falsifiers |
| wprime | F-no-atomics / F-no-wload | 0.967 / 0.901 | falsifiers |
| forward | stcs | 0.999 | out, w bitwise |
| forward | F-no-store / F-no-gemm / F-no-tp | 0.821 / 0.988 / 0.994 | falsifiers |
| forward | F-gemm-only | 0.183 | falsifier |

At degree 5 the slope is no longer the limit of wprime (4 % of it); the w load (10 %) and the tensor-product
backward are. The fused forward on the Ada is bound by its 1.9 GB w store (18 %); its product and tensor product
overlap. Neither kernel has a cheap lever left on the Ada; the A100 forward is bound by its FP32 product instead
(section 4), which only tensor cores with larger edge batches could move.

## 7. MH-1 with degree-5 tables (RTX 4000 Ada)

The check a global `radial.DEGREE` change needs: MH-1 (`mh1-r2scan`, default plan) with degree 5 against degree 10,
same-process MD step A/B, 1 fs, 8 blocks of 300 steps, both build orders (0130eca2, leto with the idle check;
`screens/rtx4000ada-mh1-deg5/`):

| Crystal | ab | ba |
|---|---|---|
| NaCl | 0.978 | 0.978 |
| NaNbO3 | 1.006 | 0.998 |
| NbCl5 | 1.001 | 0.999 |

Degree 5 is neutral on MH-1 (NaCl −2 %); the planner certifies every table at any degree (certificate at most 1),
so the accuracy rule is unchanged.

## 8. The A100 80GB PCIe with the new defaults

The recheck job (`tools/modal/omat0_a100_recheck.sh`, 4bc2ffeb, `a100-pcie-recheck/`) was meant to re-time the
defaults of 4 November on the A100-SXM4-80GB, but Modal assigned an **A100 80GB PCIe**. There the plan has only
the generic radial tables of decision 66; `tf32x3_gemms`, `wprime_backward`, `fused_folded_forward`,
`chunked_conv_backward` and the skipped zero column are listed for the SXM4 card alone. Benchmark protocol on the
PCIe card: NVE 4.606, 4.619, 4.612 and Langevin 4.556, 4.562, 4.569 µs/atom/step (23.0 ms per step), against 3.17
on the SXM4 card with them. MargrafGroup's Benchmark_loup ran on the PCIe card, so section 9 measures those
switches there. The screen `ffoff` (fused folded forward off) was 1.0006 on the PCIe card, where it was not on
anyway, and 1.108 on the RTX 5060 Ti (`screens/rtx5060ti-ffoff/`).

## 9. The decision-66 switches on the A100 80GB PCIe

`tools/modal/omat0_a100_pcie.sh` (d671614e; Modal, rerun until it was assigned the PCIe card: five SXM4 attempts
exited at once, about $0.06 each; the run itself about $1.07; `a100-pcie/`). Same-process MD step A/Bs, default plan
on side a (degree-5 tables, no tf32x3, no wprime set, no skip on this card):

| Case | b | b/a |
|---|---|---|
| pcie-all (both orders) | `tf32x3_gemms`, `wprime_backward`, `fused_folded_forward`, `chunked_conv_backward` on, `SKIP_ZERO_COLUMN` | **0.7490 / 0.7484** |
| pcie-tf32x3 | `tf32x3_gemms` on | 0.9503 |
| pcie-wprime | the wprime set (wprime backward, fused forward, chunked carrier) | 0.8086 |
| pcie-skip | the skip on top of the wprime set (both sides with the set) | 0.9938 |

Benchmark protocol on the PCIe card with pcie-all: NVE 3.324, 3.311, 3.305 and Langevin 3.277, 3.257, 3.260
µs/atom/step (medians 3.311 and 3.260), against 4.612 and 4.562 for the default plan there (section 8): 28 % less.
The switches are the arithmetic the SXM4 card already runs by default (same sm_80 kernels, force gates passed there).
Adding "NVIDIA A100 80GB PCIe" to their decision-66 lists needs the owner.

## 10. The defaults re-timed on the A100-SXM4-80GB

`omat0_a100_recheck.sh` with `SCREEN=streamw`, guarded to the SXM4 card (2913d44c, first attempt; about $0.40;
`a100-sxm4-retime/`). The default plan with no constants (degree 5 and the skipped zero column now default there):
NVE 3.187, 3.182, 3.183 and Langevin 3.242, 3.146, 3.199 µs/atom/step (medians 3.183 and 3.199), the same as dz5 set
by constants in section 4 (3.172 / 3.175) within run noise. `streamw` (wprime reads w with `__ldcs`) on top: 1.0048,
rejected on the A100 too.

## 11. Tensor cores with a cheaper split (A100-SXM4-80GB)

Where the tensor-core forward loses on the A100: the kernel lab captured its forward kernel with falsifiers
(e366b8e5, `a100-sxm4-tc-lab/`; 3fd6757f, `a100-sxm4-tc-lab2/`; `tools/modal/omat0_a100_tc_lab.sh`). FP32-SIMT
base 4.92 / 4.89 ms; tensor-core base 5.14 / 5.13 ms; one mma instead of three 3.08; no mma 3.25; W unsplit 3.15 /
3.18; no product at all 2.21 / 2.23. The mma triples cost little: the time goes to splitting each value into its
TF32 high and low parts with two `cvt.rna.tf32` per value. Masked splits: the high part masked (truncated) and the
low part rounded 4.20 ms; both by masking, the low part passed raw (the tensor core truncates it) 3.67 ms. On the
RTX 5060 Ti (`rtx5060ti-tc-lab/`) the tensor-core forward is slower whatever the split (TF32 there is 2x FP32).

`fused_forward.TC_SPLIT` (b62a42d7, 91d81ee6) selects the split: `round` (both parts rounded to nearest by an
integer add and mask, the default split), `rna` (the former `cvt.rna`) or `mask`. Accuracy (GPU test body, max
error of w over the row scale |a| |W| against float64, units of 2^-24; RTX 5060 Ti): `rna` and `round` identical,
7.6 / 10.7 at k = 9 / 65 (FP32 chain 3.7 / 5.5); `mask` 19.3 / 16.3. The test bound
`2 chain + (3 ceil(k/8) + 4) units` (17.5 / 41.9) holds for `round`, not for `mask` at k = 9, so `mask` is not
covered by the GPU test. The RTX 4000 Ada passes the test body with `round` (`tests on leto`, run
tests-split-b62a42d7: ruff clean on the changed files, 44 CPU tests).

Lab, A100-SXM4-80GB, b62a42d7 (`a100-sxm4-tc-split/`, $0.13): FP32-SIMT 4.90 ms, `rna` 5.11, `round` 4.18 (0.854),
`mask` 3.65 ms (0.744). RTX 5060 Ti (`rtx5060ti-tc-split/`): SIMT 16.09 ms, `rna` 18.51, `round` 17.83,
`mask` 17.49.

MD step A/B on the A100-SXM4-80GB, both orders, 8 blocks of 200 steps (9a9575fb, `screens/tc-split-sxm4/`, $0.38):
`tc` (`round`) b/a 0.9632 / 0.9647 (17.03 -> 16.42 ms), `tcmask` 0.9327 / 0.9330 (17.06 -> 15.92 ms).

Force gate (`tools/force_stress_gate.py`, production against float64 MACE, RTX 5060 Ti, 91d81ee6,
`gate-rtx5060ti-tc/`), max |error| and ratio to the ASE float32 calculator; all pass:

| set | SrTiO3 3x3x3 energy (eV) | forces (eV/Å) | NaCl energy | NbCl5 energy | NbCl5 forces |
| --- | --- | --- | --- | --- | --- |
| default | 7.3e-4 (1.73) | 2.45e-5 (0.82) | 2.6e-4 (1.20) | 2.1e-4 (0.99) | 7.85e-5 (0.98) |
| tc (`round`) | 1.15e-3 (2.95) | 2.54e-5 (0.90) | 2.1e-4 (1.34) | 1.6e-4 (0.74) | 7.90e-5 (0.96) |
| tcmask | 1.39e-3 (3.80) | 2.78e-5 (0.91) | 1.9e-4 (0.99) | 1.0e-4 (0.45) | 7.97e-5 (0.96) |

The SrTiO3 energy errors exceed twice the float32 calculator's but stay far below the gate's 0.01 eV floor; forces
and stress are within the ratio rule everywhere (stress ratios 0.41-1.35, all below their floors).

Benchmark_loup under its protocol on the A100-SXM4-80GB (7bd74981, `a100-sxm4-tc-loup/`, $0.49; the sets alternate in
one job, 2 fresh processes each), µs/atom/step:

| set | NVE | Langevin |
| --- | --- | --- |
| default | 3.174, 3.173 | 3.123, 3.125 |
| tc (`round`) | 3.028, 3.028 | 3.129, 2.981 |
| tcmask | 2.926, 2.919 | 2.928, 2.875 |

`tcmask` is the fastest Benchmark_loup yet on the A100 (2.92 / 2.90 µs/atom/step, -8 % / -7 %). `TENSOR_CORES` stays
off by default; a per-GPU default on the A100-SXM4-80GB (decision 66), and which split, is the owner's decision.

## 12. After the tensor-core forward: profile, wprime at degree 5, GEMM tiles (A100-SXM4-80GB)

**Step breakdown with tcmask** (`omat0_a100_profile.sh`, `LOUP=0`, `CONSTANTS` = `TENSOR_CORES=True`,
`TC_SPLIT='mask'`; 71db0f06, `a100-sxm4-tc-profile/`, $0.06): 14.85 ms of kernels per force call. wprime backward
5.16 ms (35 %), fused forward 3.63 ms, CUTLASS 3xTF32 GEMMs 1.85 ms (44 launches, grid 79 x 2 of 64x64 tiles),
`edge_outer_sum` backward 0.49 ms, radial tables 0.71 ms (forward and backward).

**wprime lab at degree 5** (`omat0_a100_wprime_lab.sh`, 211d3d3f, `a100-sxm4-wprime-deg5/`, $0.05): base 4.70 ms (96
registers, no local memory). `ldcs-w` 1.023 and `unordered` 0.997 (bitwise, no gain); falsifiers: no slope 0.615,
half the terms 0.952, w not loaded 0.884, no atomics 0.931. The slope costs 1.8 ms although its 6 terms halve to
3 for only 5 %: the per-weight cost is the gather of each edge's `G'` slab (term-major, 25.6 KB per edge from a
table of 9 pairs x 896 pieces x 5 terms x 1280 weights, about 206 MB, far beyond the 40 MB L2), not the
arithmetic. (`F-one-slab` did not match the kernel since `PADDED_TERMS`; fixed in 607135c5, not rerun.) An `a'
W^T` reformulation (dr = a' . (dw W), the shared W instead of the per-group `G'`) would need the forward's product
again on tensor cores (about 1.4 ms with `mask`), so at most about 0.4 ms; not built.

**CUTLASS tiles** (`cutlass.tf32x3.TILE`, 607135c5): Benchmark_loup's products of 5,000 rows x 128 columns launch
158 blocks of 64x64 on 108 SMs. The cubin now also holds 32x64 and 64x32 tiles (two warps of 32x32x16, 4 stages;
nvcc 13.3, CUTLASS 4.8.1; the 64x64 entries' SASS identical to the shipped cubin's). They are bitwise the 64x64
tile's (`tests/gpu/test_cutlass_tf32x3_tiles_cuda.py` on the RTX 4000 Ada: 4 shapes x 2 B layouts, largest error
2.5e-7 of the row scale). MD step A/B on the A100-SXM4-80GB, both orders, 8 blocks (`screens/tile-sxm4/`, $0.40):
`tile32x64` b/a 1.0144 / 1.0116, `tile64x32` 1.0167 / 1.0173. Slower: the two-warp tiles lose more in mainloop
efficiency than the even load gains. Rejected; `TILE` stays 64x64.

## 13. The A100 defaults (owner, 4 November 2026: "Yes to both")

`fused_forward.TENSOR_CORES` is None (per GPU): on for "NVIDIA A100-SXM4-80GB" and "NVIDIA A100 80GB PCIe", with
`TC_SPLIT = "mask"`; the A100 80GB PCIe joins the SXM4's OMAT-0 switches (`tf32x3_gemms` for OMAT-0 medium only,
the generic wprime set, `SKIP_ZERO_COLUMN_GPUS`). Before the default, MH-1 (which uses the same fused forward):

- MD step A/B on the A100-SXM4-80GB, default against `TENSOR_CORES=True`, `TC_SPLIT='mask'`, both orders, 8 blocks
  (`tools/modal/mh1_a100_tc.sh`, 3ac4f16b, `screens/mh1-tc-sxm4/`, $0.52): NaCl 0.9776 / 0.9697, NaNbO3 0.9660 /
  0.9665, NbCl5 0.9825 / 0.9820.
- Force gate (MH-1 r2SCAN, RTX 5060 Ti, `gate-rtx5060ti-mh1-tc/`), max |error| and ratio to the ASE float32
  calculator, default / tcmask: energy NaCl 1.3e-4 (0.18) / 6.2e-4 (0.81) eV, NaNbO3 6.0e-4 (0.49) / 5.6e-4 (0.46),
  NbCl5 5.0e-5 (0.07) / 5.1e-4 (0.71); forces 0.98 / 0.94, 1.04 / 1.08, 0.99 / 1.00; stress 0.31 / 0.42,
  0.57 / 0.95, 0.29 / 0.48. All pass.

Tests (leto, RTX 4000 Ada, working tree bfae89c2): merge-tier CPU suite 5,103 passed, 202 skipped; the fused forward
GPU test body with both splits (mask 1.09e-6 / 1.45e-6 of the row scale at k = 9 / 65, round 5.0e-7 / 9.5e-7,
chains 2.6e-7 / 2.8e-7); the CUTLASS tile test; ruff only pre-existing long lines.

**Edge order for the wprime slope (idea, bounded):** sorting a receiver's edges by species pair and piece would let
neighbouring edges share G' slabs. `F-one-slab` (every edge reads group 0; perfect locality) on the RTX 4000 Ada
(leto, `rtx4000ada-oneslab/`): wprime 7.73 ms base, 9.03 ms one slab (1.169), no slope 0.960, w not loaded 0.872.
Perfect locality is slower (every warp hits the same lines), so the gathers' locality is not the limit; not built.

**The A100 80GB PCIe** (`tools/modal/omat0_a100_pcie_tc.sh`, 1d20b01c, second Modal account, `a100-pcie-tc/`, $0.58),
on top of the SXM4's switches (tf32x3_gemms, the generic wprime set, the skipped zero column): MD step A/B with
`TENSOR_CORES=True`, `TC_SPLIT='mask'`, both orders, b/a 0.9314 / 0.9301 (18.40 -> 17.13 ms). Benchmark_loup,
µs/atom/step (2 processes each, sets alternating):

| set | NVE | Langevin |
| --- | --- | --- |
| switches | 3.400, 3.342 | 3.315, 3.315 |
| tc (`round`) | 3.237, 3.221 | 3.185, 3.207 |
| tcmask (now the default) | 3.125, 3.123 | 3.078, 3.108 |

On MargrafGroup's benchmark card, FastMACE's default plan now runs Benchmark_loup at 3.12 / 3.09 µs/atom/step, against
4.61 / 4.56 with the defaults before 4 November.

## 14. The W slice pre-split (rejected)

`fused_forward.TC_PRESPLIT` stores the tensor-core forward's W slice as float2 {hi, lo} once per block instead of
splitting the fragments in every batch (bitwise). Kernel lab, forward kernel of the default plan, round-robin:

| GPU | base | pre-split | F-tc-no-wsplit | launch |
| --- | --- | --- | --- | --- |
| A100-SXM4-80GB (`a100-sxm4-presplit/`, Modal 3de2e597, $0.06) | 3.688 ms | 5.023 (1.362) | 3.178 (0.862) | 12 -> 8 warps, 168 -> 194 registers |
| RTX 5060 Ti, tensor cores forced (`rtx5060ti-presplit/`) | 18.17 ms | 31.02 (1.707) | 17.99 (0.990) | 2 -> 4 parts |

The split instructions cost at most 14 % of the kernel on the A100, but holding the split slice costs a third of the
block's warps. Rejected; the switch stays for A/Bs.

## 15. Species skip: grouped GEMM, our own tensor-core kernel, the narrow first table (4 November 2026)

`SpeciesSkip` multiplies every atom's rows by all present species' concatenated [128, species x 128] matrices and keeps
its own block. `SPECIES_GEMM` replaces that with one grouped GEMM (`fastmace/kernels/triton/species_gemm.py`): atoms
sorted by species on the device, 64-row species-pure tiles on a fixed grid (CUDA-graph safe when atoms change
species), rows gathered and results scattered. Benchmark_loup MD step (`ab_rewrites.py`, 8 blocks, b/a):

| GPU | SPECIES_GEMM (count tiles) | first version | folder |
| --- | --- | --- | --- |
| NVIDIA RTX 4000 Ada Generation (leto) | 0.9548 / 0.9522 | 0.9695 / 0.9682 | `rtx4000ada-species-gemm-2/`, `rtx4000ada-species-gemm/` |
| NVIDIA GeForce RTX 5060 Ti | 0.9504 (one order) | 0.9621 | `rtx5060ti-species-gemm/`, `rtx5060ti-species-gemm-first/` |
| A100-SXM4-80GB (Modal) | 0.9919 / 0.9914 | 1.0402 / 1.0363 | `a100-sxm4-species-gemm-b/`, `a100-sxm4-species-gemm/` |

The 5060 Ti gate (`gate-rtx5060ti-species-gemm/`) passes. On the A100 the skip runs through tf32x3, and Triton's
tf32x3 dot is about 3x slower per FLOP than the CUTLASS kernels, so the third of the products only ties.

**Our own tensor-core kernel** (`species_gemm.TENSOR_CORES`, NVRTC, 3xTF32 mma.sync m16n8k8, K in chunks of
`TC_CHUNK` in shared memory; `TC_PRESPLIT` splits each chunk once). Species skip lab, size 16 (interaction 0),
forward + input backward, median us:

| GPU | Triton | tc16 | tc32 | tc64 | tc16p | tc32p | bound | folder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A100-SXM4-80GB (Modal 97250a3f) | 591.5 | 594.7 | 585.3 | 693.2 | 553.3 | 659.6 | 204.8 | `a100-sxm4-species-gemm-tc/` |
| RTX 4000 Ada (leto) | 946.3 | 863.3 | 872.2 | 1526.5 | 894.8 | 1548.3 | 831.5 | `rtx4000ada-species-gemm-tc/` |
| RTX 5060 Ti | 1189.2 | 1027.1 | 1091.6 | 2072.6 | 1128.2 | 1948.5 | 910.8 | `rtx5060ti-species-gemm-tc/` |

("bound": one product of all rows through one species' matrix, no gather, wrong result; on the A100 CUTLASS.) The
first, unchunked kernel (a whole W half per block, 71 KB shared memory, one block per SM) took 3125 us on the 5060 Ti.
MD step: A100-SXM4 default vs SPECIES_GEMM with tensor cores (16p, the lab's fastest) 1.0031 / 1.0017; Ada
SPECIES_GEMM vs SPECIES_GEMM with tensor cores 1.0006 / 0.9986. Rejected: a plain mma.sync kernel only reaches
Triton's rate, while CUTLASS is about 2.7x faster per product on the A100; closing that needs a multi-stage cp.async
pipeline with larger warp tiles (a CUTLASS-class grouped GEMM) for at most about 5 % of the A100 step.

**Narrow first table** (`FIRST_TABLE_COLUMNS`: the first radial table stores only the columns the first product
uses; bitwise): A100-SXM4 1.0018 / 0.9993, RTX 5060 Ti 1.002. Rejected.

Lab folders of the first round: `rtx5060ti-skiplab/`, `a100-sxm4-skiplab/` (A100 lab times differ between Modal
machines; compare only within one job).
