# w′ backward on the H200: formulation search and instruction census (11–12 Nov 2026)

Kernel: `oeq_projected_backward_wprime` (fastmace/kernels/oeq_streaming/wprime_backward.py) with the H200 plan,
OMAT-0 medium, interaction 2 (128x0e+128x1o ⊗ 0e+1o+2e+3o, 10 uvu paths, 86 CG terms). Base: main a3d75ae3.
H200 profile (9 Nov): w′ is 1,092 µs, 33.5 % of the force call. It is issue-bound (issue 77 %, FMA pipe 52 %).
H200 step (SrTiO3 10x10x10) is 3.369 ms, so the 1 % gate is 34 µs.

## Sources

- `render_sources.py` renders the A100 and H200 kernels. Its A100 render equals the captured
  `results/a100-colab-ideas-20261111/batch2/ghoist-prod/sources/wprime-real.cu`, blank lines aside.
- `variants.py` writes `sources/wprime-h200-<variant>.cu` via `fastmace.kernels.oeq_streaming.wprime_tuning`.
  The kernel lab's `h2-*` variants use the same transforms.

## Part 1: formulation search (closed)

`formulation_search2.py` → `formulation_search2.json`; `complex_basis.py` → `complex_basis.json`.

- Model: 339 FP instructions per channel pass. The tensor-product part is 269 today.
- The best per-path mix (Horner / factored / K forms, plus the x·c pool of 13) gives 257.75. Without K forms it gives 264.75.
  Every (o, j) has one i, so the K forms are single products.
- The h-form (362) and the complex basis (70 complex terms, at least 140 real FFMA against 86 real) are worse.
- The bound is 45 instructions per edge, about 22 µs on the H200 (0.66 % of the step). That is below the 1 % gate,
  so nothing was built.

## Part 2: sm_90 SASS census (`census_sass.py`, NVRTC + cuobjdump)

The base kernel's channel pass is 481 instructions: FFMA 290, FMUL 49, FADD 20, LDS 44, CS2R 20 (the gv zero fill
for `inside`), LDG 14, IADD3 13, MOV 10, REDG 4. The butterfly reduction is lean (21 SHFL).
Per edge that is 2,198 instructions: FP 1,479 and other 719. There are no spills (LDL/STL 0) in any uncapped build.
Register caps (`census_cap{72,80,96}.json`) spill and recompute addresses, which makes them worse.

| variant | kind | sm_90 instr/edge | Δ | regs (H200 blocks/SM) | sm_80 instr/edge | regs |
|---|---|---:|---:|---:|---:|---:|
| base | | 2,198 | | 72 (7) | 2,169 | 96 |
| skip | rounding (FMA order) | 2,026 | −7.8 % | 96 (5) | 1,999 | 128 |
| hoist | exact | 2,141 | −2.6 % | 80 (6) | 2,131 | 96 |
| skip+hoist | rounding (FMA order) | 1,990 | −9.5 % | 96 (5) | 1,976 | 128 |
| vec2 | rounding | 2,046 | −6.9 % | 128 (4) | 2,050 | 128 |
| skip+vec2 | rounding | 1,864 | −15.2 % | 128 (4) | 1,860 | 128 |
| vec4 | rounding | 2,489 | +13.2 % | 141 (3) | 2,435 | 140 |
| skip+vec4 | rounding | 1,909 | −13.1 % | 167 (3) | 2,092 | 254 |

- **skip** (`SKIP_OUTSIDE`): an edge at or beyond the cutoff writes zeros and leaves before the channel loop.
  The table loads then become unconditional. The skin list is compacted to the cutoff, so only padding edges skip.
- **hoist** (`HOIST_DX`): the dx atomics' address becomes one pointer per lane.
- **vecN** (`CHANNEL_VECTOR`): each lane takes N consecutive channels, with float-N loads of the staged dout row and x,
  and vector atomics on sm_90 (scalar below). The per-lane dsh and dr sums change order, so this is a rounding change.

The three switches were built off by default. After the H200 job, `SKIP_OUTSIDE` and `HOIST_DX` are on for the H200
(see the decision below).

## Lab runs

- Colab A100 (`colab_lab_a100.sh`, H200 plan via the GPU alias, commit ccafdcf2, 11 Nov 23:20 UTC,
  A100-SXM4-40GB, SrTiO3 10x10x10, 21 rounds; `a100-lab/`). The goal was to find register and occupancy losses.
  A100 blocks/SM: 96 regs → 5, 128 → 4 (shared memory caps it at 7).

  | variant | µs | ratio | regs (blocks/SM) | dsh / dr vs base |
  |---|---:|---:|---:|---|
  | base | 2,840.6 | 1.000 | 96 (5) | |
  | skip | 2,831.4 | 0.997 | 128 (4) | 4.4e-9 / 6.9e-8 (not bitwise) |
  | hoist | 2,873.3 | 1.012 | 96 (5) | bitwise |
  | skip+hoist | 2,911.2 | 1.025 | 128 (4) | 4.4e-9 / 6.9e-8 |
  | vec2, skip+vec2, skip+vec4 | did not compile on sm_80 at ccafdcf2 (no vector atomicAdd; fixed in d68d06a6) | | | |

  The census cut does not show up on the A100. skip has 7.8 % fewer instructions but runs 0.3 % faster, because its
  registers rise from 96 to 128 and it loses a block per SM. hoist and skip+hoist are slower. The A100 is not issue-bound
  the way the H200 is (H200 base: 72 regs and 7 blocks/SM), so this does not rule out an H200 gain. It does show that
  occupancy can eat the cut.
  skip is not bitwise: for inside edges the arithmetic is the same, but the compiler schedules the FMAs differently.
  So the lab and the H200 picker now treat it as rounding, and a skip winner also gets the force/stress gate.
  dx is never bitwise (its atomics' order varies).
- H200 (Modal, `tools/modal/wprime_h200_lab_ab.sh` at 1ad25fa8, 11 Nov 23:51 UTC, NVIDIA H200, 547 s, $0.79
  metered; `h200-job/`). The job runs the lab of base and the 7 variants. Then `tools/modal/wprime_h200_pick.py` takes
  the fastest valid variant under 0.99 and runs its MD-step A/B in both orders. A variant that is not bitwise in
  dsh / dr also gets the force/stress gate.

  | variant | µs | ratio | regs (H200 blocks/SM) | dsh / dr vs base |
  |---|---:|---:|---:|---|
  | base | 1,042.3 | 1.000 | 72 (7) | |
  | skip | 1,064.4 | 1.021 | 96 (5) | 5.5e-10 / 6.9e-8 |
  | hoist | 1,041.1 | 0.999 | 80 (6) | bitwise |
  | **skip+hoist** | **1,014.2** | **0.973** | 96 (5) | 5.5e-10 / 6.9e-8 |
  | vec2 | 1,152.5 | 1.106 | 128 (4) | 1.4e-7 / 6.2e-7 |
  | skip+vec2 | 1,127.9 | 1.082 | 128 (4) | 1.4e-7 / 6.2e-7 |
  | vec4 | 1,674.4 | 1.606 | 141 (3) | 2.1e-7 / 6.5e-7 |
  | skip+vec4 | 1,527.6 | 1.466 | 167 (3) | 2.1e-7 / 6.5e-7 |

  MD step b/a for skip+hoist, OMAT-0 medium, Benchmark_loup, both build orders:

  | structure | a → b | b/a |
  |---|---|---|
  | SrTiO3 10x10x10, ab | 3.3599 → 3.3227 ms | 0.9889 |
  | SrTiO3 10x10x10, ba | 3.3537 → 3.3288 ms | 0.9926 |
  | NbCl5 3x3x3, ab | 1.0692 → 1.0503 ms | 0.9823 |
  | NbCl5 3x3x3, ba | 1.0674 → 1.0472 ms | 0.9811 |

  Force/stress gate (`h200-job/gate-skip-hoist/`, production against ASE float32, both against float64) passes:

  | structure | forces | stress |
  |---|---:|---:|
  | NaCl | 0.86 | 0.81 |
  | SrTiO3 3x3x3 | 0.88 | 0.46 |

  NbCl5 was not in the gate.
  The variant counts as rounding only because the compiler orders the inside edges' FMAs differently (dsh 5.5e-10
  relative). The arithmetic is the same and the padding edges' zeros are exact. The conductor accepted it on that
  basis (12 Nov).

## Decision (conductor, decision 66, 12 Nov)

- `SKIP_OUTSIDE` and `HOIST_DX` are on for the H200, together only: alone they lose (1.021) or tie (0.999).
  `HOIST_DX_GPUS` is `SKIP_OUTSIDE_GPUS`. `WPrime` turns `HOIST_DX`'s default on only when `SKIP_OUTSIDE` is on
  and the kernel is float32 FACTORED with STAGE_DOUT.
- `CHANNEL_VECTOR` stays off. Its register growth (128–167 registers) costs more than the cut in instructions.
- The A100 keeps both off: skip+hoist was 1.025 there.
- This was measured on OMAT-0 medium only. Other model families on the H200 that use TABLE_W get it through the GPU
  set; that cell is listed as open in `docs/current/defaults-by-gpu.md`.
- Takeaway: on the H200 the instruction count only pays at the same occupancy class. skip alone cuts 7.8 % of the
  instructions but falls from 7 to 5 blocks per SM. With hoist on top, the cut is larger (9.5 %) at the same 96
  registers, and that wins.
