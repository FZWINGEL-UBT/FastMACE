# Idea 30: toward a full version, both parts (6 November 2026)

This continues [the part-1 prototype](idea30-prototype.md), which passed at 0.80 of part 1's time. The coordinator
set four steps, each with a stop on failure:

1. Improve part 1 in the lab: take A from shared memory, use setmaxnreg for 12 consumer warps, shrink or swizzle
   the w tile for a third ring stage.
2. Apply the structure to part 0 too. Target: the **whole forward ≤ 0.85 of base**.
3. Only if step 2 holds: port to production as `fused_forward.WS_FORWARD` (sm_90 only).
4. Then the force/stress gate and the MD-step A/B on a Modal H200.

**Result: step 2 misses its target.** The whole forward with both parts warp-specialized takes **0.859** of base
(1,292.1 against 1,504.8 µs). Per the brief, the work stops there: no production port was committed, and no gate
or MD A/B was run.

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
