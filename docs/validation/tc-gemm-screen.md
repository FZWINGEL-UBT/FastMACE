# Tensor-core GEMM screen (29 October 2026)

Owner request, 29 October 2026: screen whether FP32-accurate tensor-core GEMMs, and tensor-core work running
beside FP32 CUDA-core work, give whole-step MD gains on our GPUs (A100 on Modal, RTX A4000 on trillian, the local
RTX 5060 Ti, an RTX 4000 Ada if one is free). The input was an external review of main `b6421afc` with three ideas
in priority order. Branch `fastmace/tc-gemm-screen-20261029`; raw records in `results/tc-gemm-20261029/`.
No default changed; nothing here is in the library (`fastmace/`). Everything is screen code under
`tools/bench/tc_gemm/`.

## Summary

- **Done on the local RTX 5060 Ti (sm_120) only.** The A100 and the A4000 runs are prepared but did not run:
  - **Modal:** the WSL client has only one profile, `fzwingel`, which is the old account ($26.73 metered this
    month). No profile for the new $30 account exists, so nothing was spent. Spend: **$0.00**.
  - **trillian:** the A4000 was idle, but copying the source to `/mnt/homes/zwingel` was refused by the session's
    permission check (AGENTS.md: NAS sync and workstation jobs paused since 26 October until the owner allows
    them). Nothing ran on any workstation.
- **Review claims checked against the repository** (§1). Two of them are out of date:
  - the "2.9 ms of overlap on NaNbO3" predates `wprime_backward`. Under today's default on a decision-66 GPU the step
    has **no concurrent branch left**: 7 µs of concurrent kernel time per NaNbO3 step on the 5060 Ti (§4);
  - `segment_gemm` already writes the interleaved (joined) layout straight from its epilogue (OPT-T02 stage B).
- **Built:** a CUTLASS v4.8.1 3×TF32 library with three splits (CUTLASS's own, and two corrected
  round-to-nearest splits with 3 or 4 products), batched-strided so it writes `segment_gemm`'s joined layout
  directly. It plugs into the opt-in `tf32x3_gemms` record's launches, under CUDA graphs, per A/B side. Checks pass
  (float64, graph replay bitwise). A screen tool replays every block-linear GEMM of a real force call through
  each backend with float64 errors.
- **5060 Ti verdicts** (expected from the hardware, now measured on real operands):
  - every 3×TF32 variant is 1.6–2.3× slower than the default's GEMMs;
  - idea 2's IEEE alternatives to the Triton `segment_gemm` are slower: CUTLASS SIMT 1.13–1.44×, cuBLAS
    strided-batched into the joined layout 1.15–1.24×. Even pure cuBLAS on contiguous operands, with no layout
    cost at all, is only 2–5 % below the default's GEMM time, a ceiling under 2.5 % of the step. **Stop.**
- **Accuracy of the corrected splits (real MH-1 operands and opt-T01 §6's synthetic cancellation set):** see §3.
  The round-to-nearest split cuts CUTLASS 3×TF32's synthetic cancellation error from 5.5× to 3.5× IEEE (2.6× with
  the fourth product) at the same speed. The model-level gate and NVE drift are in §3.3.
- **Ideas 1 and 3 remain open on the A100 and the A4000/Ada:** the scripts are ready (§6).

## 1. The review's claims against the repository

| Claim | Check | Status |
| --- | --- | --- |
| CUTLASS 3×TF32 was 1.82× faster on NbCl5's GEMMs and 1.34× on NaCl's, standalone only | opt-T01 §5: GEMM sums 0.55× and 0.75× of cuBLAS IEEE (C++ binary, Colab A100-SXM4-40GB) | correct |
| It was never measured end to end | ledger `prec-cutlass-a100`: NOT BUILT | correct |
| TF32×3 is 7–14× less accurate than FP32 on cancellation-heavy inputs | opt-T01 §7: median 7×, worst 14× on the cancellation-scaled metric | correct, for CUTLASS's own split (§3: a round-to-nearest split halves it) |
| The FP32 default now has `wprime_backward`, so a fresh baseline is needed | opt9-T03: default on for MH-1 float32 on `NVIDIA A100-SXM4-80GB`, the 5060 Ti and the RTX 4000 Ada (decision 66) | correct |
| The fp32 profile shows about 2.9 ms of overlap on NaNbO3 that tensor-core GEMMs could join | fp32-profile §1.2: 2.9 ms between the main stream and the chunked convolution backward's side stream, measured **before** `wprime_backward`. The record replaced that backward with one launch, "no chunks, no side stream" (opt9-T03) | **out of date**: §4 measures 7 µs of concurrent kernel time per step under today's default |
| `segment_gemm` should get an epilogue that writes the interleaved layout directly | `fastmace/kernels/triton/segment_gemm.py`: the forward stores row (atom, m) at `atom * width + offset + m * mul_out`, the backward reads the gradient in place (OPT-T02 stage B) | **already so** |
| A Tensor Core `segment_gemm` exists to correct | `tf32x3_gemms=on` runs every joined product as Triton `tf32x3` (OPT4-T03, OPT5-T05) | correct; on the A100 the Triton kernel reaches 16.8 TFLOPS at 4096³ against CUTLASS's 40.8 (opt-T01 §4–5), so a corrected kernel is worth building in CUTLASS, not Triton |
| The packed FP16 hybrid was 13 % slower at [13608, 512, 512] | design doc §3.3 (5060 Ti, packing included) | correct; the Triton FP16×3 kernel was also 1.12–1.71× slower on the A100, A4000 and Ada (opt-5 §3.11) |
| Do not reopen the split-column SIMT+MMA hybrid on the 5060 Ti | opt-5 §3.3–3.8 | not reopened |

## 2. What was built (`tools/bench/tc_gemm/`)

- `cutlass_tc.cu`, `gen_variants.py`, `build.sh`: `libfm_tc_gemm.so`, CUTLASS v4.8.1 (`2c85f497`) `GemmUniversal`
  in batched mode (one launch per product, no workspace, capturable), A and C row-major with arbitrary leading
  dimensions and batch strides, B row- or column-major. Variants:
  - `rz3`: CUTLASS's `OpMultiplyAddFastF32` (big rounded toward zero, small half-ulp truncated, three products);
  - `rn3`: both parts rounded to nearest (Ootomo and Yokota's split), three products;
  - `rn4`: `rn3` plus small × small;
  - `tf32`: plain 1×TF32, a rate reference (not FP32-accurate);
  - `simt`: IEEE FP32 on the CUDA cores (Sm80 multistage SIMT).

  `gen_variants.py` derives `rn3`/`rn4` from CUTLASS's own `MmaTensorOpFastF32`, whose `FastF32` traits already
  support those roundings and the fourth product but hard-code them. Tiles: 64×64×16 and 64×128×16 (the two that won
  all 92 shapes in opt-T01 §5), chosen by a waves rule (`cutlass_tc.tile_for`: the wider tile only when N ≥ 128
  and its grid fills two waves). Built with nvcc 13.3 (local WSL), SASS for sm_80, 86, 89 and 120, static CUDA
  runtime.
- **The joined layout as a batched GEMM.** Product i's row (atom, m) of A sits at `atom * a_n + m * a_m`: with
  batch = m, `lda = a_n` and batch stride `a_m`, it is a strided batched GEMM over the components, and the output
  rows likewise. So the kernel writes the joined layout itself, forward and input gradients.
- `cutlass_tc.py`: ctypes wrapper and a dispatcher on the opt-in `tf32x3_gemms` record's two launch functions
  (`tf32x3_gemm._launch`, `segment_gemm._gemm`). They run at capture time, so the CUDA graphs hold the chosen kernels.
  Sub-tile products (K or N < 32) stay on SIMT IEEE as the record routes them (OPT5-T05). IEEE modes reroute the
  default plan's joined products instead (`simt-…`, `cublas-…`).
- `check.py`: every variant, tile and B layout against float64; joined writes leave the other columns untouched;
  graph replay = eager bitwise; the patched record against Triton's `tf32x3`. **ALL OK on the 5060 Ti.**
- `tc_gemm_screen.py`: records the real operands of one MH-1 force call's block-linear GEMMs, as
  `tools/tf32x3_accuracy.py` does (`tf32x3_gemms=on`, 48 products per cell), and times each mode on them, one CUDA
  graph per product and mode, rotated mode order, 7 blocks. It also computes float64 errors and runs opt-T01 §6's
  synthetic cancellation set on the two largest products.
- `ab_tc.py`: `tools/ab_rewrites.py` with a backend per side (both sides drop their graphs after the build and
  re-capture alike). `run_with_backend.py`: a launcher prefix that reaches the gate's production child.
  `overlap_trace.py`: torch.profiler over MD steps, per-stream and stream-agnostic concurrency, and the GEMM time
  that overlaps other kernels.

## 3. Results on the local RTX 5060 Ti

Local PC, WSL2, RTX 5060 Ti 16 GB (sm_120, 36 SMs), driver 610.88, `~/environments/mace_cueq_oeq` (torch
2.14.1+cu130, Triton 3.8.0), FastMACE `b48912ac` (this branch on main `fe8d7d43`), MH-1 `mh1-r2scan`, quiet local
GPU reservation. Raw data: `results/tc-gemm-20261029/local/`.

### 3.1 Block-linear GEMM time per force call (µs; ratio to the default's routing)

`ieee_default` is what the default plan runs for these products (cuBLAS IEEE for dense ones, the Triton
`segment_gemm` or cuBLAS plus a copy by `tile_rule.segment_gemm_choice` for joined ones; `grouped_launch`, which is
on by default on sm_120, is not modelled).

| Mode | NaCl (216 atoms) | NaNbO3 (1,080) | NbCl5 (1,944) |
| --- | ---: | ---: | ---: |
| default routing (IEEE) | 1,278 | 4,739 | 8,365 |
| cuBLAS IEEE, contiguous operands, no layout (ceiling of an IEEE path) | 0.952 | 0.983 | 0.944 |
| cuBLAS strided batched into the joined layout | 1.149 | 1.224 | 1.243 |
| CUTLASS SIMT FP32 into the joined layout | 1.442 | 1.126 | 1.168 |
| cuBLAS 1×TF32 (rate reference) | 0.582 | 0.595 | 0.612 |
| Triton `tf32x3` (the record as it is) | 1.999 | 2.116 | 2.090 |
| CUTLASS 3×TF32 `rz3`, tile rule / 64×64 only | 1.755 / 1.637 | 1.713 / 1.655 | 1.723 / 1.665 |
| CUTLASS 3×TF32 `rn3` | 1.753 | 1.704 | 1.710 |
| CUTLASS 3×TF32 `rn4` | 2.292 | 2.248 | 2.261 |
| FP16×3 prepacked (two cuBLAS FP16 GEMMs, split outside the timing) | 0.982 | 1.041 | 1.135 |
| FP16×3 activation split alone (unfused) | 0.838 | 1.004 | 1.486 |

The joined-layout rows come from the second run (`local-ieee`, NaCl 1,292 / NaNbO3 4,906 / NbCl5 8,587 µs default);
the rest from `local-screen`. The two runs' defaults differ by 1–3 %, from clocks.

- **3×TF32 on sm_120 cannot win** (TF32 = FP32 rate here; opt-T01 §3). CUTLASS is 15–20 % faster than Triton's
  `tf32x3`, still 1.6–1.7× slower than IEEE. On sm_120, unlike the A100, the 64×64 tile beats the tile rule's
  64×128.
- **Idea 2's IEEE strategy on the 5060 Ti: stop.** Both joined-layout alternatives are slower than the Triton
  `segment_gemm`. The only faster path is cuBLAS on contiguous operands: 2–5 % less GEMM time, before any
  layout copy. With the block linears at 40–50 % of the step, that ceiling is under 2.5 % of the step; the
  layout copies it needs would take part of it. The fp32 profile's 2–6 % estimate assumed cuBLAS's 13–15 TFLOPS on
  the joined layout, which neither cuBLAS's batched kernels nor CUTLASS SIMT reach. The earlier screens agree: 12
  Triton tile configurations within 3 % in isolation and neutral 128-row tiles in the step (fp32 profile §5.2).
- **FP16×3 on the 5060 Ti** (GeForce: FP16 with FP32 accumulation at half rate): even with both operands split
  outside the timing it only ties cuBLAS on NaCl and is 4–14 % slower on the large cells.

### 3.2 Accuracy against float64

Cancellation-scaled error max |c − c64| / Σ|a·b|, ratio to cuBLAS IEEE on the same operands. Real operands: the
48 products of one force call, median / max over products. Synthetic: opt-T01 §6's set on the largest product
(`[13608, 512] @ [512, 512]` on NbCl5; NaCl and NaNbO3 alike).

| Mode | Real, NaCl | Real, NaNbO3 | Real, NbCl5 | Synthetic cancellation, NbCl5 (abs. / ratio) |
| --- | --- | --- | --- | --- |
| cuBLAS IEEE | 1 / 1 | 1 / 1 | 1 / 1 | 7.4e-9 / 1 |
| Triton `tf32x3` | 0.63 / 1.35 | 0.32 / 1.35 | 0.30 / 2.05 | 2.5e-8 / 3.4 |
| CUTLASS `rz3` (CUTLASS's split) | 0.91 / 2.29 | 0.53 / 2.30 | 0.48 / 2.40 | 4.1e-8 / 5.5 |
| CUTLASS `rn3` (round to nearest) | 0.69 / 1.81 | 0.36 / 1.63 | 0.35 / 2.23 | 2.6e-8 / 3.5 |
| CUTLASS `rn4` (+ small × small) | 0.77 / 2.03 | 0.39 / 1.78 | 0.37 / 2.23 | 1.9e-8 / 2.6 |
| CUTLASS SIMT FP32 | 2.12 / 6.53 | 1.00 / 6.06 | 1.00 / 8.39 | 7.4e-9 / 1.0 |
| FP16×3 prepacked | 2.26 / 6.01 | 1.03 / 10.1 | 0.96 / 5.06 | 2.7e-8 / 3.6 |

- **The round-to-nearest split is the better 3×TF32:** same speed as CUTLASS's own (`rn3` 1.70–1.75× against `rz3`
  1.71–1.76× of IEEE here), and its cancellation error is lower (synthetic 3.5× IEEE against 5.5×). The fourth
  product helps a little more (2.6×) at 30 % more time. Worst over real products, every 3×TF32 variant is within
  2.4× of IEEE, and their absolute worst (4.4e-7 to 7.7e-7) is below cuBLAS IEEE's own worst (7.2e-7 to 1.2e-6).
- **FP16×3** (power-of-two normalised per row and column, 2^11-scaled small parts, two GEMMs) is up to 10× worse
  than IEEE on one real NaNbO3 product. Its 11 + 11 bits do not cover float32's 24, so per-element rounding of the
  operands is about 2^-22. Idea 3's "full-range numerical pass" would need a third FP16 part or a different split.
- The SIMT kernel is IEEE FMA arithmetic; its larger maximum ratio comes from one or two products where cuBLAS's
  own error is unusually small (a different summation order), not from a split.

### 3.3 Model-level gate and NVE drift (CUTLASS `rz3` and `rn3` through `tf32x3_gemms`)

PENDING (local run in progress).

## 4. Concurrency under today's default

`overlap_trace.py`, NaNbO3, 20 MD steps after 100 warm-up steps under torch.profiler, default plan
(`wprime_backward` on), 5060 Ti:

| Per step | µs |
| --- | ---: |
| Kernel span | 13,054 |
| Union of all kernels | 12,861 |
| Sum of kernel times | 12,867 |
| **Concurrent kernel time (sum − union)** | **7** |
| GEMM kernels (55 per step) | 4,850 |
| GEMM time overlapping a non-GEMM kernel | 2 |

The pre-`wprime` profile's 2.9 ms of overlap was the chunked convolution backward on its side stream; that
backward is now one launch on the main stream. There is no second branch for tensor-core GEMMs to run beside.
The second idea of the owner's request, tensor-core work beside FP32 CUDA-core work, would need a branch
created for the purpose (one stream for TC GEMMs, one for the SIMT kernels). The step's dependency chain leaves
little to split that way: the GEMM inputs are the previous kernels' outputs. The same trace on the A100, with the
CUTLASS variants, is part of the prepared Modal job (§6).

## 5. Verdicts

| Idea | GPU | Result | Verdict |
| --- | --- | --- | --- |
| 1. CUTLASS 3×TF32 in the force graph | A100 | not run (Modal profile) | **open**; integration and scripts ready |
| 1. | 5060 Ti | GEMMs 1.6–1.8× slower (rate ceiling) | rejected (as opt-T01 predicted) |
| 2. Corrected tensor-core `segment_gemm` | A100 | not run | open; the corrected split exists as `rn3`/`rn4` in the same library, joined layout included |
| 2. IEEE tile strategy for `segment_gemm` | 5060 Ti | SIMT 1.13–1.44×, cuBLAS batched 1.15–1.24× slower; ceiling < 2.5 % of the step | **stop** |
| 3. Prepacked FP16×3 | A4000, Ada | not run (trillian refused; no Ada GPU free within the cap) | open; the first check is `fp16x3_prepacked` in the screen |
| 3. | 5060 Ti (for the record) | 0.98–1.14× of default, without activation preparation | not a target GPU |
| 4. Split-column SIMT+MMA hybrid on the 5060 Ti | – | not reopened | – |
| TC beside CUDA-core work | all decision-66 GPUs | no concurrent branch in today's step (7 µs per NaNbO3 step) | no overlap to use without restructuring the step |

## 6. To finish (prepared, not run)

**A100 (Modal, new account).** Tarball = `git archive` of this branch plus `tc_gemm_lib/libfm_tc_gemm.so` (built
locally, SASS sm_80):

```bash
modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 40 --tarball T --command "LIMIT=2000 bash results/tc-gemm-20261029/modal_a100.sh screen" --out ~/fastmace-runs/tc-gemm-20261029/modal-a100-screen
```

Then `ab rn3` (or the variant the screen favours; 6 A/Bs, about 30 minutes) and `acc rn3` (gate and NVE). The stop
rule: whole step ≤ 2 % faster in either build order, or a failed gate or NVE check, rejects it. Estimated cost at
$2.50/h: about $1.5 (screen) + $1.5 (A/Bs) + $1.5 (gate, NVE).

**A4000 (trillian) and an Ada card if free.** `results/tc-gemm-20261029/workstation.sh ROOT FM LIB check screen`
with the source in `/mnt/homes/zwingel/fastmace-work/tc-gemm-screen/` (about 15 minutes per GPU). Idea 3's first
check is the `fp16x3_prepacked` column on the two largest shapes against `ieee_default`; stop unless it beats it.
Needs the owner's permission for workstation jobs (AGENTS.md, NAS pause of 26 October).

## Reproducing (local)

```bash
bash tools/bench/tc_gemm/build.sh ~/fastmace-work/cutlass-v4.8.1 ~/fastmace-work/tc_gemm/build /usr/local/cuda/bin/nvcc
FASTMACE_SCREEN_TC_GEMM_LIB=~/fastmace-work/tc_gemm/build/libfm_tc_gemm.so python3 tools/ci/local_gpu.py run --quiet --purpose "tc-gemm screen" -- ~/environments/mace_cueq_oeq/bin/python tools/bench/tc_gemm/tc_gemm_screen.py --output OUT --provider local
```
