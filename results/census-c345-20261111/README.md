# Offline census of C3, C4 and C5 (11 November 2026)

This is the compile screen from `plans/next-steps-20261111.md` section C. It covers C3 and C4 from the outside note
(`plans/notes/inference-ideas-20261111/ideas.md`) and C5 from round three. The census runs on the CPU with NVRTC
for sm_80 and sm_90; no GPU time was used.

**Verdict: all three stop.** None gets GPU time.

## Method

`census_c345.py` holds the transforms, the SASS analysis and the gauge search. Its outputs are `census_c345.json` and
`census_c345.log`, with the compiled sources in `sources/`.

- **Base sources:** the production kernels captured on the A100 on 11 November, in
  `results/a100-colab-ideas-20261111/screen1/ghoist/sources/`:
  - `real-0.cu`: `spline_forward0` with SPLINE_W, R128, STAGE_EDGES and PACKED8;
  - `wprime-real.cu`: w′ with STAGE_DOUT, STAGE_EDGES, STAGE_SH, SPLINE_VECTOR 4, HORNER, FACTORED, SCALAR_PATHS
    and BUTTERFLY.
- **Variants:** each is a text transform of the captured source, compiled in the same NVRTC build with
  `tools/bench/cu_census.py`'s compiler.
- **"Per edge and channel":** the instructions in the SASS loop range that executes once per edge and channel. That
  is the innermost backward-branch loop holding at least 90 % of the kernel's FFMA:
  - in the forward, the per-edge `k` loop (one thread per channel);
  - in w′, the per-channel loop.
- **Converting to time:** a saving becomes µs as kernel time × the share of the loop saved. The kernel times are from
  the 9 November step limiters, per SrTiO3 force call:
  - A100-SXM4 (Alex), FMA-pipe bound: w′ 2,177 µs, scaled by the FMA-pipe instructions (FFMA, FMUL, FADD, IMAD);
  - H200, issue bound: w′ 1,088.7 µs, scaled by all instructions.
  - This is an upper bound. Earlier small-kernel gains showed up at a third to a half in the MD step.

**Approximations:**
- The H200's own plan differs from the captured A100 sources: its forward has PREFETCH_AHEAD off, and its w′
  differs in STAGE_EDGES and STAGE_SH. C3 removes the prefetch for both targets. The w′ counts on sm_90 are for the
  A100-form kernel.
- C3's pairing rates are the outside note's synthetic SrTiO3 snapshots (σ 0.05 / 0.08 / 0.30 Å), not production
  captures.

## C3: one spline decode for two edges on the same table row (target H200)

In the variant, the staged loop takes edges k and k + 1 together when `s_tq` matches, meaning the same pair and
interval.
- The PACKED8 row is loaded and decoded once per path for both edges.
- Each output still accumulates edge k before edge k + 1.
- Unmatched edges take today's body.

Let p be the share of decodes that pairing removes (pairs per edge).

| | sm_90 | sm_80 |
| --- | --- | --- |
| registers (cap 80 from `__launch_bounds__(128, 6)`) | 80 | 80 |
| spills (LDL/STL) | **2** (`STL.64` / `LDL.64` in the paired arm) | 0 |
| today's loop, per edge | 348 | 340 |
| variant: header + single arm + tail, per unpaired edge | 27 + 345 + 3 = 375 | 23 + 338 + 4 = 365 |
| variant: header + paired arm + tail, per pair | 27 + 539 + 3 = 569 | 23 + 524 + 4 = 551 |
| saved per edge and channel = 348 − (569p + 375(1 − 2p)) | −27 + 181p | −25 + 179p |
| p = 0.248 / 0.192 / 0.092 (σ 0.05 / 0.08 / 0.30 Å) | **17.9 / 7.7 / −10.4** | 19.4 / 9.3 / −8.5 |
| bound with no pairing overhead, p (2 × base − paired arm) | 39.0 / 30.1 / 14.4 | 38.7 / 29.9 / 14.3 |

The always-paired loop (`c3_paired_only.cu`, for counting only) spills 45 to 65 instructions.

**Stop** on two counts:
- It spills on sm_90.
- It saves at most 18 instructions per edge and channel, against a bar of 27. Without any pairing overhead the bound
  reaches 27 only on the synthetic snapshots with σ ≤ 0.08 Å, and spills are still there.

## C4: joint rescaling, staging-only form (dout and harmonics, x unscaled)

In this form, coefficients become c·b_j / s_o, where s_o scales the staged dout per output component (40 of them)
and b_j scales the staged harmonics per component (16).
- dsh is restored by 1/b_j when the edge is written.
- dx and the weight gradient need no restore.
- The gauge comes from a random search of consistent subsets, 4,000 orders plus the identity, scored by distinct
  non-unit products, since the compiler shares one product per output and coefficient magnitude.
- This is a candidate search, not a proven minimum.

| | count |
| --- | --- |
| CG terms in w′ / unit today | 86 / 50 |
| distinct non-unit coefficient products per edge and channel: today | 25 |
| … with output scaling only | 18 (matches the outside note) |
| … staging-only, best gauge found (scales 0.204–2.32) | **10** |

| | sm_80 | sm_90 |
| --- | --- | --- |
| registers | 81 (today 79; uncapped kernel, under about 85) | 80 (80) |
| spills | 0 | 0 |
| shared per block | 23,296 B (unchanged; limit about 27 KB) | 24,320 B |
| loop, per edge and channel | 475 → **457 (−18 FMUL)** | 477 → **459 (−18)** |
| FMA pipe in the loop | 359 → 341 | 359 → 341 |

**Boundary transforms**, estimated per edge and channel:
- dout staging: one FMUL per staged element, so 5,120 × 1.4 receiver runs per 32-edge chunk ÷ (32 × 128) ≈ 1.75;
- harmonics staging: 0.125;
- dsh restore: 0.125.
- Total about 2, so the net is about **16 instructions** per edge and channel.

| | estimate | lab bar |
| --- | --- | --- |
| A100: 2,177 µs × 16 / 359 | **≈ 97 µs** (113 µs with no boundary cost) | 130 µs |
| H200: 1,088.7 µs × 16 / 477 | **≈ 37 µs** (41 µs) | 68 µs |

**Stop:** it falls under the lab bar on both GPUs, even as an upper bound.
- The full form (x scaled too) cannot change that. The outside note's best is 9 products, one fewer than staging-only.
- It would also need per-node x transforms and a restore of the forward's message.
- If it had cleared the bar, it would be a rounding change needing the gate.

## C5: w′ accumulators started from their first product

60 accumulators (u_i, v_j, weight_gradient, input_gradient_*) take `acc = p` instead of `acc = 0; acc += p`.

| | sm_80 | sm_90 |
| --- | --- | --- |
| loop | 475 → 475 | 477 → **477** |
| FFMA / FMUL in the loop | 290 / 49 → 230 / 109 | 290 / 49 → 230 / 109 |
| registers | 79 → 80 | 80 → 80 |

**Stop:** there is no instruction drop on sm_90. ptxas already emits the zero-started FFMA as a single
instruction; the change only turns 60 FFMA into FMUL.

## Resource limits on the A100

Every kernel compiled here stays within about 85 registers and 27 KB of shared memory per block:
- forward: 80 registers, 2.6 KB;
- w′: 79–81 registers, 22.75 KB.

C4's 81 registers on sm_80 are one above today's.
