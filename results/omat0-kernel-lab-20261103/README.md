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
