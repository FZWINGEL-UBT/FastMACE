# Tensor-core GEMM screen (29–30 October 2026)

Owner request, 29 October 2026: screen whether FP32-accurate tensor-core GEMMs, and tensor-core work running
beside FP32 CUDA-core work, give whole-step MD gains on our GPUs. The input was an external review of main
`b6421afc` with three ideas in priority order. Branch `fastmace/tc-gemm-screen-20261029`; raw records in
`results/tc-gemm-20261029/`. **No default changed and nothing is in the library (`fastmace/`):** the CUTLASS
kernels and their hook are screen code under `tools/bench/tc_gemm/`.

## Summary

- **Idea 1 passes on the A100-SXM4-80GB:** CUTLASS 3×TF32 (`rz3`, CUTLASS's own split) in the real force graph,
  under CUDA graphs, through the opt-in `tf32x3_gemms` record. Same-process whole-step A/Bs against the FP32
  default with `wprime_backward` (on in both sides), 12 blocks of 300 steps, both build orders:

  | Cell | order ab: a / b ms, b/a | order ba: b/a | block pairs |
  | --- | --- | --- | --- |
  | NaCl | 2.191 / 1.887, **0.861** | **0.874** | 0.859–0.906 |
  | NaNbO3 | 9.229 / 7.259, **0.787** | **0.788** | 0.784–0.791 |
  | NbCl5 | 11.153 / 7.921, **0.710** | **0.709** | 0.700–0.745 |

  That is 13–29 % shorter MD steps, every block pair favouring rz3 and no compile during timing.
- **Accuracy on the A100:** the force/stress gate passes (forces 0.86–1.03× the float32 calculator's error
  against float64, limit 2×; the default's 0.89–1.02×). NVE drift over 20 ps is unchanged on NaCl and NbCl5
  (§4). On real operands every product is FP32-class. On opt-T01 §6's synthetic cancellation set rz3 is 7× worse
  than IEEE, as opt-T01 §7 found.
- **No concurrent branch is left to overlap.** The review's "2.9 ms of overlap on NaNbO3" predates
  `wprime_backward`. Under today's default the step is one chain: 55–69 µs of concurrent kernel time per step on
  the A100 and 7 µs on the 5060 Ti. The rz3 gain is the GEMM time alone: 4.14 → 2.34 ms per NaNbO3 step and
  6.60 → 3.67 ms per NbCl5 step (§3). A hybrid inside one kernel fails the co-execution probe on the A100
  (keep 0.57–0.75); only a split-producer variant reaches 1.18× in the inner loop, at most about 5 % of the step.
- **Idea 2:** the corrected (round-to-nearest) splits `rn3`/`rn4` are more accurate under cancellation but 2.8–4.9×
  slower than IEEE on the A100, so stock `rz3` is the variant to use. `segment_gemm` already writes the
  interleaved layout from its epilogue; the CUTLASS kernels write it too (batched-strided). On the 5060 Ti every
  IEEE alternative to the Triton `segment_gemm` is slower. **Stop.**
- **Idea 3 (Ada, A4000): not run.** The copy to the workstation home was refused by the session's permission
  check (AGENTS.md: NAS work paused), and later all permitted workstation GPUs were taken. The prepacked-FP16
  check is ready (`fp16x3_prepacked` in the screen). On the A100 it already halves the GEMM time (0.50×), but its
  accuracy is up to 11× worse than IEEE on real operands (§2.2), so it fails the "full-range numerical pass".
- **Modal (new account `fzwingel-ubt`): $4.58 metered,** including two failed starts of my own making (a library
  that did not load, then a hung step until the job timeout; $2.04 of the total).
- **Adoption** (decision 66) needs the owner: rz3 changes the 9 October precision decision (FP32, TF32 off), so
  it is neither a default nor in the library. An opt-in library backend is the follow-up (§6).

## 1. The review's claims against the repository

| Claim | Check | Status |
| --- | --- | --- |
| CUTLASS 3×TF32 was 1.82× faster on NbCl5's GEMMs and 1.34× on NaCl's, standalone only | opt-T01 §5: GEMM sums 0.55× and 0.75× of cuBLAS IEEE (C++ binary, Colab A100-SXM4-40GB) | correct; in the model graph now 0.50–0.53× (NaNbO3) and 0.75× (NaCl) of the default's GEMM time (§2.1) |
| It was never measured end to end | ledger `prec-cutlass-a100`: NOT BUILT | correct; now measured (summary) |
| TF32×3 is 7–14× less accurate than FP32 on cancellation-heavy inputs | opt-T01 §7 | correct (§2.2: 7× on the synthetic set); on real MH-1 operands FP32-class |
| The default now has `wprime_backward`, so a fresh baseline is needed | opt9-T03, decision 66 | correct; both A/B sides set it explicitly, because Modal's `A100-80GB` was once an `A100 80GB PCIe`, where the name rule leaves it off |
| About 2.9 ms of overlap on NaNbO3 that tensor-core GEMMs could join | fp32-profile §1.2: measured **before** `wprime_backward`, which removed the chunked backward's side stream | **out of date** (§3) |
| `segment_gemm` needs an epilogue that writes the interleaved layout | `segment_gemm.py` already stores rows at `atom * width + offset + m * mul_out` (OPT-T02 stage B) | **already so** |
| The packed FP16 hybrid was 13 % slower at [13608, 512, 512] | design doc §3.3 (5060 Ti, packing included) | correct |

## 2. GEMM screen: every block-linear GEMM of one real force call

`tools/bench/tc_gemm/tc_gemm_screen.py` records the 48 block-linear products of one MH-1 force call
(`tf32x3_gemms=on`, as `tools/tf32x3_accuracy.py` does; forward products and input gradients, real operands) and
replays each through every backend: one CUDA graph per product and mode, rotated mode order, 7 blocks, medians.
`default` is the default plan's routing (cuBLAS IEEE for dense products, the Triton `segment_gemm` or cuBLAS plus a
copy for joined ones). The joined-layout modes write `segment_gemm`'s layout directly.

### 2.1 Time per force call, ratio to the default's routing

| Mode | A100-SXM4-80GB NaCl | A100 NaNbO3 | 5060 Ti NaCl | 5060 Ti NaNbO3 | 5060 Ti NbCl5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| default routing, µs | 1,188 | 4,254 | 1,278 | 4,739 | 8,365 |
| cuBLAS IEEE, contiguous operands (no layout cost) | 0.944 | 0.944 | 0.952 | 0.983 | 0.944 |
| cuBLAS strided batched into the joined layout | 1.527 | 1.019 | 1.149 | 1.224 | 1.243 |
| CUTLASS SIMT FP32 into the joined layout | 2.203 | 1.368 | 1.442 | 1.126 | 1.168 |
| cuBLAS 1×TF32 (rate reference, not FP32-accurate) | 0.368 | 0.251 | 0.582 | 0.595 | 0.612 |
| Triton `tf32x3` (the record as it was) | 1.658 | 1.239 | 1.999 | 2.116 | 2.090 |
| **CUTLASS 3×TF32 `rz3`** (tile rule) | **0.747** | **0.528** | 1.755 | 1.713 | 1.723 |
| `rz3`, 64×64 tile only | 0.748 | 0.504 | 1.637 | 1.655 | 1.665 |
| CUTLASS `rn3` (round-to-nearest split) | 4.890 | 2.818 | 1.753 | 1.704 | 1.710 |
| CUTLASS `rn4` (`rn3` + small × small) | 4.991 | 2.901 | 2.292 | 2.248 | 2.261 |
| FP16×3 prepacked (two cuBLAS FP16 GEMMs; split outside the timing) | 0.705 | 0.496 | 0.982 | 1.041 | 1.135 |
| FP16×3 activation split alone (unfused) | 1.175 | 0.651 | 0.838 | 1.004 | 1.486 |

- The A100 NbCl5 screen stopped at an illegal memory access in the replay of the screen's own emulated
  `default` mode (product 10, `debug-NbCl5.tail.log`). Eager warm-ups of every mode passed, and the MD graphs with
  rz3 run correctly (A/Bs, gate, NVE), so it is a harness bug, not a kernel one. It was not chased further; the
  whole-step A/B covers NbCl5.
- The 5060 Ti joined-layout rows come from a second run (default 1,292 / 4,906 / 8,587 µs).
- **On sm_80 the round-to-nearest variants are 5–7× slower than rz3**, though on sm_120 they cost the same. CUTLASS's
  round-to-nearest TF32 conversion is a software sequence, against rz3's truncation, and is not worth more work.
- **On the A100 the 64×64 tile is as fast as or faster than the tile rule's 64×128.** The rule (wider tile only
  when it fills two waves) costs 4.8 % on NaNbO3. A production rule would use 64×64 throughout.

### 2.2 Accuracy against float64

Cancellation-scaled error max |c − c64| / Σ|a·b|, ratio to cuBLAS IEEE on the same operands; real operands median
/ max over the 48 products; synthetic: opt-T01 §6's set on the largest product (`[1512 or 7560 or 13608, 512] @
[512, 512]`).

| Mode | A100 real NaCl | A100 real NaNbO3 | 5060 Ti real NbCl5 | synthetic (A100 NaNbO3) |
| --- | --- | --- | --- | --- |
| cuBLAS IEEE | 1 / 1 | 1 / 1 | 1 / 1 | 6.0e-9 (1×) |
| Triton `tf32x3` | 0.96 / 3.28 | 0.63 / 1.58 | 0.30 / 2.05 | 2.5e-8 (4.2×) |
| **CUTLASS `rz3`** | 1.59 / 4.42 | 0.96 / 2.26 | 0.48 / 2.40 | 4.1e-8 (6.8×) |
| CUTLASS `rn3` | 1.19 / 3.80 | 0.67 / 1.64 | 0.35 / 2.23 | 2.4e-8 (4.0×) |
| CUTLASS `rn4` | – | – | 0.37 / 2.23 | 1.7e-8 (2.8×) |
| FP16×3 prepacked | 3.51 / 11.4 | 1.86 / 8.11 | 0.96 / 5.06 | – |

- rz3 is FP32-class on real operands, but not IEEE-equivalent: on NaCl its median product is 1.6× IEEE and its
  worst 4.4×. The model-level gate decides whether that matters (§4).
- The round-to-nearest split almost halves the cancellation error, but costs 5–7× rz3's time on the A100 (§2.1).
- **FP16×3** (11 + 11 bits, power-of-two normalised per row and column) is up to 11× worse than IEEE on real
  operands: two FP16 parts do not cover float32's 24 bits. Idea 3's "full-range numerical pass" would need a third
  part, about 1.5× the work.

## 3. Concurrency under today's default

`overlap_trace.py`: 20 MD steps after 300 warm-up steps under torch.profiler; per step, stream-agnostic (CUPTI may
file graph kernels under the launch stream). `wprime_backward` on.

| Per step (µs) | A100 NaNbO3 default | A100 NaNbO3 rz3 | A100 NbCl5 default | A100 NbCl5 rz3 | 5060 Ti NaNbO3 default |
| --- | ---: | ---: | ---: | ---: | ---: |
| kernel span | 9,312 | 7,298 | 11,083 | 7,813 | 13,054 |
| **concurrent kernel time** (kernel sum − union) | **55** | 48 | **69** | 45 | **7** |
| GEMM kernels | 4,142 | 2,336 | 6,599 | 3,670 | 4,850 |
| GEMM time overlapping a non-GEMM kernel | 31 | 19 | 48 | 25 | 2 |

- The step has no second branch any more: `wprime_backward` replaced the chunked convolution backward and its side
  stream with one launch. Tensor-core GEMMs therefore cannot hide beside CUDA-core work between kernels; the rz3
  gain is its GEMM time saved (1.8 ms of 2.0 ms on NaNbO3, 2.9 ms of 3.3 ms on NbCl5).
- **A hybrid inside one GEMM** (SIMT FP32 work beside 3×TF32 MMA in one kernel): the single-SM co-execution probe
  (`tools/megakernel_probe.py`, opt4-T07's gate: keep = SIMT rate kept ≥ 0.90 and gain over the better unit alone
  ≥ 1.10) on the Modal A100-SXM4-80GB, 30 October, commit `afc82168`
  (`results/tc-gemm-20261029/modal-a100-sxm4-probe/`):

  | Variant (4×4 warps, one block per SM) | keep | gain | Verdict |
  | --- | ---: | ---: | --- |
  | `gemm` tf32 (SIMT SGEMM loop beside a 64×64 MMA tile) | 0.75 | 0.87 | fail |
  | `x3` tf32 (3×TF32 with the split on the MMA warps, as CUTLASS) | 0.65 | 0.95 | fail |
  | `x3` fp16 (3×FP16) | 0.57 | 0.78 | fail |
  | `x3_pipe` tf32 (SIMT warps produce the split, MMA warps only multiply) | 0.63 | **1.18** | fail (keep) |
  | `x3_pipe` fp16 | 0.64 | 1.11 | fail (keep) |

  - The plain hybrid loses: on sm_80 the tensor cores alone (3×TF32 at 258 FLOP/cycle/SM, split included) are
    twice the CUDA cores' 126, and sharing the issue slots costs more than the SIMT side adds. This confirms opt4-T07
    on the A100-SXM4-40GB (keep 0.58, gain 0.92).
  - `x3_pipe` is not a FP32 + TF32 hybrid in the original sense: the CUDA cores do the split for the MMA warps,
    which then run at their full rate (MMA share 1.00 against 0.73). Its 1.18 combined rate fails the rule's
    keep ≥ 0.90. That threshold assumes the CUDA cores are the main unit, which on the A100 they are not, so the
    gain is the relevant number there. It is an inner-loop upper bound without global memory. On sm_120 a 1.22×
    inner-loop gain vanished in the real GEMM (opt-5 §3.4). With rz3 the GEMMs are 32 % of the NaNbO3 step, so
    even the full 1.18 would be worth at most about 5 % of the step.
  - **Verdict: no hybrid kernel is worth building now.** A producer/consumer 3×TF32 GEMM on the A100 (CUTLASS
    warp-specialised, split on SIMT warps) would be the only candidate, at ≤ 5 % of the step.

## 4. Accuracy gate and NVE drift

`tools/force_stress_gate.py` (production child through `run_with_backend.py`, which installs the CUTLASS backend;
its log names it), ratios to the float32 ASE calculator's error against float64 (pass ≤ 2×, with absolute floors):

| | NaCl E / F / S | NbCl5 E / F / S | NaNbO3 E / F / S |
| --- | --- | --- | --- |
| A100-SXM4-80GB, default (`wprime_backward`) | 0.25 / 0.89 / 1.07 | 0.14 / 1.02 / 0.39 | 0.17 / 1.01 / 0.60 |
| **A100-SXM4-80GB, rz3** | 2.31* / 0.86 / 1.55 | 0.49 / 1.03 / 0.36 | 0.84 / 1.02 / 0.83 |
| 5060 Ti, rz3 | 0.56 / 0.93 / 0.49 | 0.51 / 1.05 / 0.31 | 0.28 / 1.03 / 0.88 |
| 5060 Ti, rn3 | 0.32 / 0.94 / 0.44 | 0.17 / 1.05 / 0.34 | 0.11 / 1.01 / 0.71 |

\* NaCl's energy error on the A100 is above twice the calculator's, but far below the gate's absolute floor
(0.01), so it passes. It is the largest change; the energy sums every atom's readout, so a small rounding bias of the split would
add up there (not verified). All three cells pass.

NVE drift (`tools/bench/nve_drift.py`, 20 ps at 1 fs, 300 K; meV/atom/ps, rms in meV/atom):

| | NaCl drift / rms | NbCl5 drift / rms |
| --- | --- | --- |
| A100 default | −7.2e-8 / 2.750e-4 | −1.5e-6 / 3.799e-4 |
| **A100 rz3** | 1.0e-6 / 2.743e-4 | −8.0e-6 / 3.804e-4 |
| 5060 Ti default | 5.2e-7 / 2.739e-4 | 7.7e-7 / 3.795e-4 |
| 5060 Ti rz3 / rn3 | −1.4e-7 / 2.733e-4, 2.0e-6 / 2.735e-4 | −6.3e-6 / 3.800e-4, −5.9e-6 / 3.804e-4 |

The drifts are at most 8e-6 meV/atom/ps, so at most 1.6e-4 meV/atom over the 20 ps, below the 3.8e-4 rms
fluctuation; the rms is the same to three digits. opt9-T03's default runs drifted −1.1e-6 to 2.5e-6. NbCl5's −8e-6 is the largest; a longer run would
show whether it is a trend or noise.

## 5. What was built (`tools/bench/tc_gemm/`)

- `cutlass_tc.cu`, `gen_variants.py`, `glibc_shim.c`, `build.sh`: `libfm_tc_gemm.so`, CUTLASS v4.8.1 (`2c85f497`)
  `GemmUniversal` in batched mode: one launch per product, no workspace, capturable. A and C are row-major with
  any leading dimension and batch stride, B row- or column-major. Variants:
  - `rz3`: `OpMultiplyAddFastF32` (big rounded toward zero, small half-ulp truncated, three products);
  - `rn3`, `rn4`: round-to-nearest split, generated from CUTLASS's `MmaTensorOpFastF32`;
  - `tf32`: 1×TF32 (rate reference);
  - `simt`: IEEE FP32 SIMT.

  Built with nvcc 13.3 in WSL: SASS for sm_80/86/89/120, static CUDA runtime, static libstdc++ with its symbols
  hidden, and a glibc 2.38 shim, so it loads on Modal's Debian image.
- **The joined layout as a batched GEMM:** product i's row (atom, m) of A sits at `atom * a_n + m * a_m`. With
  batch = m, `lda = a_n` and batch stride `a_m`, it is a strided batched GEMM over the components, and the output
  rows likewise.
- `cutlass_tc.py`: ctypes wrapper and a per-call dispatcher on the `tf32x3_gemms` record's launches
  (`tf32x3_gemm._launch`, `segment_gemm._gemm`). They run at capture time, so the CUDA graphs hold the chosen
  kernels. Sub-tile products (K or N < 32) stay on SIMT IEEE (OPT5-T05).
- `check.py` (float64, graph replay bitwise, the patched record): ALL OK on the 5060 Ti and the A100.
- `ab_tc.py` (ab_rewrites with a backend per side; both sides re-capture after the build), `run_with_backend.py`,
  `overlap_trace.py`, `tc_gemm_screen.py`.
- Scripts: `results/tc-gemm-20261029/modal_a100.sh` (phases `screen`, `debug`, `ab V`, `acc V`), `workstation.sh`,
  `make_tarball.sh`.

## 6. Open

- **Adoption (owner decision).** rz3 changes the 9 October precision decision, so it needs the owner. A default
  by exact device name (decision 66) would start with `NVIDIA A100-SXM4-80GB`, the GPU measured here. Modal's
  `A100-80GB` also served an `NVIDIA A100 80GB PCIe` once, which would need its own A/Bs.
- **A library backend (follow-up task):**
  - a build step for the CUTLASS library (or a JIT build);
  - a `RewriteConfig` value choosing the tf32x3 backend;
  - the 64×64 tile throughout;
  - GPU tests;
  - the planner refusal off sm_80-class tensor cores.

  The 5060 Ti and the Ada/A4000 cannot gain from 3×TF32 (TF32 below 3× their FP32 rate), so the rule is per GPU,
  not per capability.
- **Not run:**
  - idea 3 on the A4000 and RTX 4000 Ada (workstations unavailable);
  - the A100 NaCl/NbCl5 numbers for the PCIe card;
  - the A100 NbCl5 GEMM screen (harness bug above).
- **Intra-GEMM hybrid on the A100:** probed, fails (§3); a split-producer 3×TF32 GEMM is worth at most about 5 % of the step.

## Reproducing

```bash
bash tools/bench/tc_gemm/build.sh ~/fastmace-work/cutlass-v4.8.1 ~/fastmace-work/tc_gemm/build /usr/local/cuda/bin/nvcc
FASTMACE_SCREEN_TC_GEMM_LIB=~/fastmace-work/tc_gemm/build/libfm_tc_gemm.so python3 tools/ci/local_gpu.py run --quiet --purpose "tc-gemm screen" -- ~/environments/mace_cueq_oeq/bin/python tools/bench/tc_gemm/tc_gemm_screen.py --output OUT --provider local
MODAL_CONFIG_PATH=/mnt/c/Users/fzwin/.modal.toml MODAL_PROFILE=fzwingel-ubt MODAL_IMAGE_BUILDER_VERSION=2025.06 modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 60 --tarball T --command "STEP=900 LIMIT=2900 bash results/tc-gemm-20261029/modal_a100.sh ab rz3" --out OUT
```

The tarball is a `git archive` with `.source_commit` plus `tc_gemm_lib/` (`make_tarball.sh`). The new Modal
workspace needs `MODAL_IMAGE_BUILDER_VERSION=2025.06`: it defaults to builder 2026.10, which the pinned client
rejects.
