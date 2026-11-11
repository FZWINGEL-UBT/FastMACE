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
| skip | exact | 2,026 | −7.8 % | 96 (5) | 1,999 | 128 |
| hoist | exact | 2,141 | −2.6 % | 80 (6) | 2,131 | 96 |
| skip+hoist | exact | 1,990 | −9.5 % | 96 (5) | 1,976 | 128 |
| vec2 | rounding | 2,046 | −6.9 % | 128 (4) | 2,050 | 128 |
| skip+vec2 | rounding | 1,864 | −15.2 % | 128 (4) | 1,860 | 128 |
| vec4 | rounding | 2,489 | +13.2 % | 141 (3) | 2,435 | 140 |
| skip+vec4 | rounding | 1,909 | −13.1 % | 167 (3) | 2,092 | 254 |

- **skip** (`SKIP_OUTSIDE`): an edge at or beyond the cutoff writes zeros and leaves before the channel loop.
  The table loads then become unconditional. The skin list is compacted to the cutoff, so only padding edges skip.
- **hoist** (`HOIST_DX`): the dx atomics' address becomes one pointer per lane.
- **vecN** (`CHANNEL_VECTOR`): each lane takes N consecutive channels, with float-N loads of the staged dout row and x,
  and vector atomics on sm_90 (scalar below). The per-lane dsh and dr sums change order, so this is a rounding change.

All three switches are off by default (`SKIP_OUTSIDE_GPUS`, `HOIST_DX_GPUS`, `CHANNEL_VECTOR_GPUS` are empty).
They are only turned on after a measured H200 win.

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
- H200: `tools/modal/wprime_h200_lab_ab.sh` runs the lab of base plus the 7 variants, then the MD-step A/B in both
  orders for the best variant (it must be at least 1 % faster than base, and bitwise for exact variants or within
  1e-5 for rounding ones). A rounding winner also gets the 3× float32 force/stress gate.
