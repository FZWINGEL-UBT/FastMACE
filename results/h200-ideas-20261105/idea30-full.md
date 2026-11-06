# Idea 30: toward a full version, both parts (6 November 2026)

This continues [the part-1 prototype](idea30-prototype.md), which passed at 0.80 of part 1's time. The coordinator
set four steps, each with a stop on failure:

1. Improve part 1 in the lab: take A from shared memory, use setmaxnreg for 12 consumer warps, shrink or swizzle
   the w tile for a third ring stage.
2. Apply the structure to part 0 too. Target: the **whole forward ≤ 0.85 of base**.
3. Only if step 2 holds: port to production as `fused_forward.WS_FORWARD` (sm_90 only).
4. Then the force/stress gate and the MD-step A/B on a Modal H200.

**Result: step 2 first missed its target.** The whole forward with both parts warp-specialized took **0.859** of
base (1,292.1 against 1,504.8 µs), and the work stopped there. The coordinator then ruled that 0.85 was only the
falsifier line and asked for the full version anyway. That is now done:
- r4 tested the extras;
- block order "swap" was adopted, reaching **0.850**;
- the production port is `fused_forward.WS_FORWARD`;
- r5 ran the force/stress gate and the MD-step A/B: **0.958 / 0.961** in both build orders.

See "The full version (r4, r5)" at the end.

- **Branch:** `fastmace/idea30-ws`. Lab at `0477660f`, adding the `ws30-p0`, `ws-both` and `ws-p1-keep0` variants to
  `tools/lab_idea30.py`; the debug check in `tools/check_idea30_local.py` now covers both parts.
- **Run:** r3 on Modal H200, profile `fourth`, `runs/idea30/r3`. Same lab settings as the prototype: OMAT-0 medium,
  Benchmark_loup SrTiO3 10×10×10 (373,760 edges), median of 15 with an L2 flush. Cost about $0.14 (ledger
  $11.2151 → $11.3552).

## Step 1: the improvements do not fit in shared memory

None of the three part-1 improvements fits the H200's 232,448 B opt-in limit per block, so no lab job was spent on
them. Part 1 has six paths, so N = 192 and W's TF32 high and low planes take 2 × 192 × 64 × 4 = 98,304 B. Both
planes are needed for 3xTF32, and wgmma reads B only from shared memory.

| Layout | Per stage | Stages | Total with W | Fits 232,448 B? |
| --- | ---: | ---: | ---: | --- |
| Today's ws30: w tile [6][64][36], sh [64][16] | 59,392 | 2 | 217,216 | yes |
| A third stage, RS = 36 | 59,392 | 3 | 276,480 | no |
| A third stage, unpadded w tile (RS = 32) | 53,248 | 3 | 258,048 | no |
| 12 consumers (96-edge rounds), RS = 36 | 89,088 | 2 | 276,480 | no |
| 12 consumers (96-edge rounds), RS = 32 | 79,872 | 2 | 258,048 | no |

- 96-edge rounds are also not a multiple of wgmma's M = 64. With 12 consumers the round would have to be 128 edges
  or the batches uneven, which is larger still.
- Taking A from shared memory instead of registers adds a's high and low planes: 64 × 64 × 4 × 2 = 32 KB per
  stage. It could share the w tile's space, because D lands only after `wgmma.wait_group 0`. It only pays,
  though, if setmaxnreg then gives the registers to 12 consumers, which needs the 96-edge rounds above.
- A swizzle does not shrink the tile. It only removes RS's padding, which is the RS = 32 rows above.

What would fit: a third stage for **part 0 alone**. Its W planes are 65,536 B and a stage is 40,960 B, so three
stages take 188,416 B, inside part 1's 217,216 B budget. This was not tried, because it cannot close the gap
computed below.

## Step 2: both parts (H200, µs; median of 15; r3)

| Variant | Time | / its reference | / base |
| --- | ---: | ---: | ---: |
| base (today's whole forward) | 1,504.8 | | 1.000 |
| `F-part1-only` (today's part 1) | 971.9 | | 0.646 |
| `F-part0-only` (today's part 0) | 600.4 | | 0.399 |
| `F-tc-no-product` (the plan's bound, both parts) | 663.6 | | 0.441 |
| `F-tc-no-product-p1` | 417.8 | 0.430 of part 1 | 0.278 |
| `F-tc-no-product-p0` | 258.3 | 0.430 of part 0 | 0.172 |
| `ws30` (part 1 only) | 778.0 | **0.800** of part 1 | 0.517 |
| `ws30-p0` (part 0 only, N = 128) | 524.2 | **0.873** of part 0 | 0.348 |
| `ws30-p0-n64` (part 0, two N = 64 halves) | 521.8 | 0.869 of part 0 | 0.347 |
| `F-ws30-p0-no-tp` (part 0: staging and product) | 305.2 | 0.508 of part 0 | 0.203 |
| **`ws-both`** (both parts ws) | **1,292.1** | | **0.859** |
| `ws-p1-keep0` (part 1 ws, part 0 today's code) | 1,346.0 | | 0.894 |

- **Part 1** reproduces the prototype: 0.800, against 0.803 and 0.802 in r1 and r2.
- **Part 0 gains less: 0.873.** Its TP is light (four paths, x row 0), so today's 12 warps already hide part 0's
  product better. The structure's fixed costs weigh more there: W's planes, the barrier round trip, and
  staging a and sh through shared memory. Staging and product alone (`F-ws30-p0-no-tp`) already take 305 µs, so
  the producer side does not leave room for much more.
- **The savings do not add up.** Separately, part 1 saves 194 µs and part 0 saves 76 µs, 270 µs together, which
  would give 0.82. Measured, `ws-both` saves 213 µs. The two parts' blocks share the 132 SMs over the launch
  (grid [33, 4, 2]), so the forward's time is not the sum of the parts' times. A part-only run also has the
  other part's blocks returning at once, which frees SMs.
- `ws-p1-keep0` (0.894) shows that mixing today's part 0 with ws part 1 is worse than both ws. Its launch carries
  ws's 217,216 B of shared memory and 12 warps for part 0 too.

**Correctness (r3):** against today's kernel, from the same captured call:

| Variant | out0 | out1 | out2 | out3 |
| --- | --- | --- | --- | --- |
| `ws30` (against `F-part1-only`) | 4.18e-8 | 3.27e-8 | 2.99e-8 | 3.04e-8 |
| `ws30-p0` (against `F-part0-only`) | 1.09e-7 | 2.91e-7 | 8.07e-8 | 1.67e-7 |
| `ws-both` (against base) | 1.09e-7 | 2.91e-7 | 8.07e-8 | 1.67e-7 |

All values are max |d| / max |a|. `w` is bitwise in every row: it is not stored, because STORE_W is false. Part 0's
differences are larger than part 1's because today's part 0 computes w·x0 first (CG_FOLD) in mma.sync's order, but
they are inside CG_FOLD's 3–7e-7 class. Locally, `tools/check_idea30_local.py --run` (the debug render, both parts,
four consumers, FP32 SIMT product) matched the naive reference bitwise in every column over three launches on the
RTX 5060 Ti.

Registers on sm_90a (ptxas, no spill stores):

| Variant | Registers |
| --- | ---: |
| `ws30` | 154 |
| `ws30-p0` | 154 |
| `ws30-p0-n64` | 122 |
| `ws-both` | 154 |
| `ws-p1-keep0` | 161 |

The lab reports 16–64 B of local memory, a stack frame.

## Verdict and what would be needed

**Idea 30 as designed reaches 0.859 of the forward, just short of the 0.85 target.** That is −213 µs of the
1,505 µs forward, about 4 % of the 5.40 ms MD step if it carried over to the MD unchanged.

To clear 0.85 the forward must lose about 13 µs more. Candidate levers, none measured:

- **A third stage for part 0 only.** It fits (above) and would give part 0's TP more slack.
- **Per-half `wready` for part 1.** The consumers would start on the first 96 columns while the second half's wgmma
  runs.
- **Rebalancing the grid.** Today's [33, 4, 2] split assumes today's per-part costs. With ws, part 1 is 1.48× part 0
  (778 / 524) instead of 1.62×, so the slices per part could shift.

Each would need another lab job. The plan's larger lever, more TP throughput via 12 consumer warps, does not fit in
shared memory on the H200 (step 1).

## The full version (r4, r5)

The coordinator answered that 0.85 was only the falsifier line and that 0.859 is worth taking. He asked for:
- one lab job with the cheap extras;
- the production port;
- one job with the force/stress gate and the MD-step A/B.

**Generator.** It now lives in production, in `fastmace/kernels/oeq_streaming/ws_forward.py`. `tools/lab_idea30.py`
wraps it and keeps the naive reference of the debug check. Two options are new:
- `stages`: an S-stage ring, with the TP S − 1 rounds behind the staging;
- `order`: which part a block runs.
  - "z" is today's order: all part-0 blocks launch before part 1's, since the grid is [33, 4, 2] with one block per
    SM.
  - "swap" runs part 1, the heavier part, first.
  - "interleave" alternates the parts over the linear block index.

Locally (`tools/check_idea30_local.py --run`, RTX 5060 Ti), the debug render matched the naive reference bitwise in
every column over three launches in each configuration:

| Stages | Order |
| ---: | --- |
| 2 | z |
| 3 | z |
| 3 | swap |
| 2 | swap |
| 2 | interleave |

### Lab extras (r4, `710c26e0`; H200, µs, median of 15)

| Variant | Time | / base | Notes |
| --- | ---: | ---: | --- |
| base | 1,491.9 | 1.000 | |
| `ws-both` (z, 2 stages) | 1,283.5 | 0.860 | as r3 (0.859) |
| `ws30-p0` / `ws30-p0-s3` (part 0 alone, 2 / 3 stages) | 521.2 / 525.2 | 0.349 / 0.352 | third stage: no gain |
| `ws-both-s3` | 1,284.6 | 0.861 | |
| **`ws-both-swap`** | **1,268.4** | **0.850** | adopted |
| `ws-both-s3-swap` | 1,267.9 | 0.850 | a tie with swap; 2 stages kept |
| `ws-both-inter` / `ws-both-s3-inter` | 1,520.4 / 1,508.8 | 1.019 / 1.011 | rejected |
| `base-swap` (today's kernel, part 1 first) | 1,483.0 | 0.994 | bitwise today's |
| `base-inter` | 1,563.3 | 1.048 | |

- **The per-half ready barrier was not built.** The TP text reads every path's w for each edge (the render's
  `w[g][b]` loop), so the consumers cannot start on the first half's paths before the second half lands.
  Splitting it would mean rewriting the TP, not the protocol.
- **The third stage gains nothing.** Part 0's ring is not the limit, so production keeps two stages.
- **Block order is the useful lever.** With part 1 first, part 0's short blocks fill the end of the launch.
  Against today's kernel with the same swap, ws is 1,268.4 / 1,483.0 = 0.855.
- **A side result:** `base-swap` makes today's kernel 0.6 % faster in the lab, bitwise. It is not adopted here.
- **Correctness:** each ws variant matched base to between 8.1e-8 (out2) and 2.9e-7 (out1), as max |d| / max |a|
  per output.

### Production: `fused_forward.WS_FORWARD` (`ac4b37e4`)

- **Switch:**
  - `WS_FORWARD = None`, `WS_FORWARD_GPUS = GPUModels(())` (empty) and `ws_forward_default(name)`, as for CG_FOLD and
    A_LD128;
  - layering allow-list entry in `tests/cpu/test_layering.py`.
- **When it applies:** `FusedFoldedForward` applies it only when every condition holds:
  - compute capability 9.0;
  - STORE_W is false;
  - it is not round 5's WGMMA kernel;
  - VEC_A and VEC_SH are on;
  - `apply_ws_forward` accepts the render.
- **What it changes:** the launch rule becomes 12 warps with ws's shared memory (217,216 B, checked against the
  GPU's opt-in limit), and the kernel compiles for sm_90a. TP_NOFLUSH and A_LD128 are not applied on top.
- **Fallback:** a render that `apply_ws_forward` refuses (ValueError) is built as today's kernel.
- **Defaults:** `ws_forward.STAGES = {}` (2 stages) and `ORDER = "swap"`.
- **GPU test:** `tests/gpu/test_ws_forward_cuda.py`, skipped unless compute capability is 9.0.
  - **Setup:** the H200 plan's message kernel with and without WS_FORWARD, on a synthetic C = 128 problem. The
    graph is uneven: 15 % empty rows, empty receiver chunks, receivers with 150 edges and padding edges.
  - **Check:** both layouts, three launches each, max |d| / max |ref| < 1e-5.
  - **Result:** it passed on the H200 in r4 and in r5, with a max rel of 7.4e-8 (ir_mul) and 6.3e-8 (mul_ir).
- **Lint:** ruff is clean on the new files. fused_forward.py has its 6 earlier E501 lines and no new ones.
- **Job script:** `tools/modal/ws_forward_check.sh`, which runs the lab variants, the GPU test, the gate and the
  A/B.

### Gate and MD A/B (r5, `ac4b37e4`; Modal H200, one job)

Force/stress gate: OMAT-0 medium, `results/loup-a100-20261103/SrTiO3.cif@3x3x3`, NaCl and NbCl5. The H200's
default plan was run against `--production-constant fastmace.kernels.oeq_streaming.fused_forward:WS_FORWARD=True`
in the same job. Each cell is the error ratio to ASE float32.

| Structure | Quantity | Default | WS_FORWARD | Limit | Pass (default / ws) |
| --- | --- | ---: | ---: | --- | --- |
| SrTiO3 3×3×3 | energy (max abs, eV) | 3.09e-3 | 3.14e-3 | 0.01 | yes / yes |
| | forces | 0.992 | 0.992 | ≤ 2 | yes / yes |
| | stress | 0.877 | 0.979 | ≤ 2 | yes / yes |
| NaCl | energy (max abs, eV) | 4.02e-4 | 4.01e-4 | 0.01 | yes / yes |
| | forces | 0.746 | 0.751 | ≤ 2 (floor) | yes / yes |
| | stress | 0.996 | 0.954 | ≤ 2 (floor) | yes / yes |
| NbCl5 | energy (max abs, eV) | 9.16e-4 | 9.12e-4 | 0.01 | yes / yes |
| | forces | 0.977 | 0.966 | ≤ 2 | yes / yes |
| | **stress** | **2.152** | **2.062** | ≤ 2 | **no / no** |

WS_FORWARD fails exactly what the default fails: NbCl5 stress, the H200's known exception (2.06–2.10 in earlier
jobs). It does so at a slightly lower ratio, 2.06 against the default's 2.15 in this job. Everything else passes as
the default does.

MD-step A/B: `hopper_sweep.sh`, `AB='ws|fastmace.kernels.oeq_streaming.fused_forward:WS_FORWARD=True'`, SrTiO3
10×10×10, 1 fs, 6 blocks of 256 steps after 200 warm-up steps, both build orders. a is the default, b is
WS_FORWARD.

| Order | a (ms) | b (ms) | b/a |
| --- | ---: | ---: | ---: |
| abab | 5.5584 | 5.3274 | **0.9584** |
| abba | 5.5501 | 5.3337 | **0.9610** |

That is −0.22 ms per step, about 4 %, as the lab predicted (−215 µs on the forward). Both runs are marked valid.

**Cost:** r4 $0.14 GPU (ledger $11.93 → $12.12), r5 about $0.92 with the gates (→ $13.04), on profile `fourth`.

**Open:**
- the default decision: `WS_FORWARD_GPUS` stays empty, and the coordinator decides;
- the CPU suite on leto: test_layering's allow-list check included;
- an H100 run, if wanted: the switch applies on any compute capability 9.0 GPU, but only the H200 was measured.
