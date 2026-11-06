# Idea 30: warp-specialized wgmma forward, part-1 lab prototype (6 November 2026)

This is the falsifier for idea 30 in the H200 ideas plan (revision 14). The user approved it on 6 November 2026,
which lifted the brief's exclusion of wgmma a·W and of producer/consumer warps. The question: does a lab prototype
of part 1 alone (the six 1 ⊗ l paths), with the product on a producer warpgroup and the TP on consumer warps, reach
0.85 or less of today's part-1 time? It is a kernel-lab variant only. Nothing changed in production code.

- **Branch:** `fastmace/idea30-ws`, cut from `fastmace/round13-prod` (`17212f86`).
- **Code:**
  - `tools/lab_idea30.py`: the variants;
  - `tools/check_idea30_local.py`: the local checks;
  - two hooks in `tools/omat0_kernel_lab.py`: `_idea30_variants`, and `_arch_suffix`, which compiles sources
    marked sm_90a for `sm_90a`.
- **Runs:** Modal H200 on profile `fourth`, lab at `1b5479a6`: `runs/idea30/r1` and `runs/idea30/r2`.
- **Lab settings:**
  - OMAT-0 medium, Benchmark_loup SrTiO3 10×10×10 (373,760 edges), captured after 300 MD steps;
  - 15 rounds round-robin per variant, with a 256 MB L2 flush before each launch;
  - the H200 default plan (TC 3xTF32 mma.sync, W_GROUP, LANE_INDICES, X_LATE, CG_FOLD, DEGREE_PARTS);
  - its render is byte for byte round 13's H200 lab render; A_LD128 and TP_NOFLUSH are off there.
- **Cost:** about $0.33 per job, under $0.70 for both.

## Verdict

**Pass: 0.80 of part 1's time, against the threshold of 0.85.** ws30 took 780.1 µs in r1 and 778.8 µs in r2.
Today's part 1 (`F-part1-only`) took 971.6 µs in both jobs, so the ratio is 0.803 and 0.802. That is about
−192 µs on the forward, ≈ 1,504 → ≈ 1,312 µs if part 0 stayed as it is, ≈ 3.5 % of the 5.40 ms MD step. The
messages agree with today's to 3–6e-8 relative.

The prototype is still well short of the plan's ceiling: it is 1.86× F-tc-no-product's part-1 bound (418 µs). Its
falsifiers say why. With the product removed, the TP and staging on eight consumer warps alone take 588–598 µs,
which is already 0.61 of part 1. ws30 is about 1.31× that, so the product is not fully hidden either.

## Design (`ws30`)

The kernel body for part 1 is generated from the plan's render. Part-0 blocks return at once, as in
`F-part1-only`. The grid is unchanged: [33, 4, 2], one block per SM. Each block has 384 threads (12 warps) and
217,216 B of dynamic shared memory. The kernel uses 154 registers and spills nothing. The lab reports 48 B of local
memory, a stack frame; ptxas reports no spill stores.

- **Warp roles:**
  - warpgroup 0 is the producer;
  - warpgroups 1 and 2 hold eight consumer warps;
  - setmaxnreg is not used, so every warp has the launch bound's 168 registers.
- **Work split:** each consumer warp walks its own receiver chunks in 8-edge batches, exactly as today's warps do:
  - the same chunk counter, RC = 2, and CSR order within each receiver;
  - so no message needs atomics, and the TP and flush code is today's text, copied from the render;
  - one round is the eight consumers' batches together: 64 edges, the M of one wgmma.
- **Shared memory:**
  - W's TF32 high and low planes, 2 × 48 KB, filled once per block with CG_FOLD's c_p. They use core-matrix order
    with no swizzle, round 1's WGMMA layout and descriptor (LBO 128 B, SBO KP·32 B).
  - A ring of two stages, 59,392 B each. A stage holds:
    - the w tile [6 paths][64 edges][RS = 36];
    - the a tile [64][KP + 4], laid over the start of the w tile;
    - the sh tile [64][16].
- **Consumers, per round:**
  1. Claim the next chunk if needed. Rows of empty chunks are written as zeros at claim time.
  2. Wait on the stage's `empty` barrier from two rounds back.
  3. Stage the batch's a rows and sh rows with cp.async (16 B, zero-filled past the batch). Arrive on `full`
     through `cp.async.mbarrier.arrive`, so the arrival lands when the copies have, plus one plain arrive.
  4. Run the previous round's TP:
     - gather x first;
     - wait on `wready`;
     - read w (lane = channel) and sh from the stage;
     - run today's TP and receiver flushes. Each round carries its chunk bounds, because staging runs one round
       ahead of the TP.
  5. Arrive on `empty`.
- **Producer, per round:**
  1. Wait on `full`.
  2. Load the 64 × 64 a tile into registers as wgmma A fragments, split by today's mask split.
  3. Synchronize the warpgroup with a named barrier. After it, D may overwrite the a tile.
  4. For each half of the part's paths (N = 96), issue 24 `wgmma.mma_async m64n96k8 tf32`. Per k step these are
     today's three passes in today's order: lo·hi, hi·lo, hi·hi.
  5. Wait for the wgmma, store D into the w tile, and arrive on `wready`.

  A round in which no consumer has a batch ends the producer, and a flag ends the consumers.
- **Watchdog:** each mbarrier wait traps after about 10 s, so a deadlock cannot burn a Modal job's timeout.
- **`ws30-n192`:** one m64n192k8 per pass instead of two halves. It uses 168 registers and spills 16 B, and was
  slightly slower (788.6 µs).

The falsifiers below are timing only; their outputs are wrong:
- `F-ws30-no-tp`: consumers skip the TP, leaving staging, the product and the protocol;
- `F-ws30-no-product`: the producer issues no wgmma and stores no D, leaving staging, the TP and the protocol;
- `F-tc-no-product-p1`: the lab's F-tc-no-product (k-loop removed) restricted to part 1. This is the plan's bound.
  It runs today's 12 warps with staging, TP and redistribution, but no product.

## Part-1 times (H200, µs; median of 15)

| Variant | r1 | r2 | / part 1 (r1, r2) | / whole forward (r1) |
| --- | ---: | ---: | ---: | ---: |
| base (whole forward, both parts) | 1,503.8 | 1,504.7 | | 1.000 |
| `F-part1-only` (today's part 1) | 971.6 | 971.6 | 1.000 | 0.646 |
| `F-tc-no-product-p1` (the plan's bound) | 418.8 | 417.7 | 0.431, 0.430 | 0.278 |
| **`ws30`** | **780.1** | **778.8** | **0.803, 0.802** | 0.519 |
| `ws30-n192` | 788.6 | | 0.812 | 0.524 |
| `F-ws30-no-product` (staging and TP on 8 warps) | 598.1 | 588.2 | 0.616, 0.605 | 0.398 |
| `F-ws30-no-tp` (staging and product) | 414.0 | 414.4 | 0.426, 0.427 | 0.275 |

Registers on sm_90a (ptxas):
- `F-part1-only`: 142;
- `ws30`: 154;
- `ws30-n192`: 168, with 16 B of spills;
- the falsifiers: 138–151.

## Correctness

Both jobs compared ws30's messages (out0–out3) with `F-part1-only`'s, from the same captured call:

| Output | Max \|d\| / max \|a\| |
| --- | --- |
| out0 | 4.18e-8 |
| out1 | 3.27e-8 |
| out2 | 4.48e-8 (r2); 5.97e-8 (r1) |
| out3 | 3.04e-8 |
| w | bitwise (not stored: STORE_W is false, and both leave it zero) |

ws30-n192 gave the same. This is the rounding class expected from wgmma's accumulation order with the same
3xTF32 terms, an order of magnitude below CG_FOLD's 3–7e-7.

Before any H200 run, a local check on the RTX 5060 Ti exercised the whole protocol: the mbarrier ring, cp.async
staging, the one-round lag between staging and TP, chunk bounds, zero rows of empty chunks, and termination. The
check was `tools/check_idea30_local.py --run`. It used the debug render: four consumer warps and an FP32 SIMT product
reading W from global memory, which fits 99 KB of shared memory. On a synthetic CSR graph with 15 % empty rows and
empty chunks, the debug render matched a naive per-receiver reference, built from the same TP text, bitwise in all
part-1 columns over three launches. The wgmma pieces (fragment layout, descriptors, D mapping) are round 1's WGMMA
code. Their check is the H200 comparison above.

## Where the time goes

- **Product and staging:** `F-ws30-no-tp` takes 414 µs, which is 177 rounds of 64 edges per SM at about 2.3 µs
  each. The producer is not the limit on its own.
- **The TP on 8 warps:** `F-ws30-no-product` takes 588–598 µs, against today's 12 warps without the product at
  418 µs. Today the TP's latency is hidden by 12 warps; here only 8 run it. Every round also couples the eight
  consumers through the `empty` barrier, so the slowest batch of each round sets its pace. With two stages there
  is only one round of slack.
- **Overlap:** ws30 takes 780 µs, against max(414, 593) ≈ 593 µs for perfect overlap. The remaining ≈ 185 µs is
  imperfect overlap. The producer's A loads, D stores and wgmma B reads share shared-memory bandwidth and issue
  slots with the consumers. The two-stage ring also lets one slow round stall the next.

## What a full version would need

1. **More TP throughput: the larger lever.**
   - The TP on 8 warps (0.61 of part 1) is the floor of this design.
   - Taking A from shared memory (descriptors) instead of registers would cut the producer to D only (48 registers
     per half). With setmaxnreg it could then drop to about 56 registers, freeing room for 12 consumer warps at
     about 150 registers (128 × 56 + 384 × 150 ≈ 64.8 K).
   - That needs a's TF32 high and low planes in core-matrix order in each stage, written by the producer's split.
2. **Shared memory: the binding constraint.**
   - W's two planes take 96 KB and each stage 58 KB, so a third stage or 96-edge rounds do not fit 227 KB.
   - Options:
     - an unpadded or swizzled w tile instead of RS = 36;
     - TMA or swizzled layouts;
     - W planes shared by the two halves of the k range;
     - a w tile reused across the paths of one half before the next half's D lands, with per-half `wready` and
       `empty`. This also lets the consumers start the first half's paths early.
3. **Pipelining in the producer:** issue the second half's wgmma before storing the first half's D. This needs
   registers, so it presupposes item 1.
4. **Part 0:** the same structure with N = 128 (4 paths, x row 0). Its TP is much lighter (20 FP32 per
   edge-slice), so the product is a larger share there and the gain may be larger.
5. **Production:** a render path in `fused_forward` (sm_90a only, `compile_cubin(..., suffix="a")`), with
   `launch_rule` for the new shared memory, then the force/stress gate and MD-step A/Bs in both build orders, as
   for every H200 default.

Even as built, part 1 alone is worth about −190 µs on the forward. Items 1 and 2 aim at the rest of the gap to the
bound (≈ 360 µs more).
