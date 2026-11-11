# First interaction: falsifiers for C1 and C2 (11 November 2026)

Items C1 and C2 of `plans/next-steps-20261111.md` (ideas 18 and 19 of `plans/ideas-20261111-round3.md`): the first
interaction's two tabled kernels, `tabled_outer_sum`'s forward and fused backward, which are 13 % of the A100's force
call and 10 % of the H200's (`results/step-limiters-20261109/`, OMAT-0 medium, SrTiO3 10×10×10).

**Outcome.**
- **C1 is closed.** Its falsifier clears the bar on the A100: removing the slot atomics and the zero fill saves
  202 µs, against a bar of 130 µs.
- Of the three correct kernels built from it, the best keeps only 76 µs of that saving on the A100. That is under
  the conductor's stop line of about 130 µs.
- **C2 is closed by a bound.** Its falsifier is over the bar on the A100 only: one load per column saves 168 µs
  there, against 130.
- The real packed table cannot hold the slopes. A packed quadratic or cubic slope needs 9,000–15,000 pieces in
  the physical region alone, against today's 380 (619 in all). That puts the table at 140–330 MB, against the
  A100's 40 MB L2.
- The part that does fit, packed values for the forward, is worth at most 76 µs, under the bar.

## Method

`tools/first_interaction_lab.py` (this branch) times the production ops and copies of their kernels with
timing-only changes.

- **Edges:** the real key order and edge lengths: SrTiO3 10×10×10, positions jittered by 0.07 Å, cutoff 6 Å,
  edges sorted by receiver and then sender species. That gives 368,752 edges and 15,000 slots, 24.6 edges per
  slot.
- **Table:** random coefficients of the production shape: 9 pairs, 896 pieces, 6 terms, 66 columns. K = 64,
  D = 16.
- **Timing:**
  - Each sample is 10 calls back to back between two CUDA events. This keeps host launch time out; on Alex's
    slow CPU it was about 100 µs per call (`a100-alex-launch-incl/`, superseded).
  - 41 rounds on the 5060 Ti and 61 on the A100, in alternating variant order; medians after 3 warm-up rounds.
  - The A100 job runs the lab twice; the two repeats agree within 1 µs.
- **Checks:** every copy matches the production op (bitwise for the backward copies, 9e-8 for the forward). Every
  correct prototype matches to 2.6e-7 relative, a different summation order.
- **Census:** `--census` compiles every variant offline for sm_80, sm_90 and sm_120 (Triton 3.x, ptxas -v) and
  records `census/census.json`.
- **Machines:**
  - Local RTX 5060 Ti (WSL2, `mace_cueq_latest`), under `tools/ci/local_gpu.py` and a systemd unit, with the
    lab's watchdog: memory fraction 0.9, 14.8 GiB, 16,000 MiB, kernel-log stalls, 3× time.
  - NHR Alex A100-SXM4-80GB: jobs 4984611, 4984621, 4984642 and 4984672, on nodes a0534 and others, env
    `fm-omat0small`.
  - Sources: commits 038beaa8 (launch time included), 456f367f, d9bd725a and a24a39f0 of
    `fastmace/first-interaction`. The kernels are those of main f31d4f50.

## C1: the forward by slot (idea 18)

Forward times in µs: the zero fill plus the kernel.

| Variant | Correct sums | A100 (4984672) | 5060 Ti (r4) |
| --- | --- | ---: | ---: |
| production op (`fwd-prod`) | yes | 475 | 721 |
| this file's copy of it (`fwd-copy`, the reference below) | yes | 466 | 721 |
| zero fill alone | – | 36 | 181 |
| **falsifier: no slot atomics, no fill** (`fwd-noslot`) | no | **264 (−202)** | **401 (−320)** |
| no slot atomics, fill kept | no | 312 (−154) | 594 |
| by slot, 1 warp per slot (`byslot-w1`) | yes | **390 (−76)** | 490 (−231) |
| by slot, 2 / 4 warps | yes | 498 / 541 | 519 / 893 |
| run-aligned walk, nested slot loop (`aligned-w1/w2`) | yes | 477 / 474 | 514 / 583 |
| the same with its stores masked off | no | 413 | 411 |
| run-aligned single loop, plain stores (`walk`) | yes | 555 | 599 |

**Verdict.**
- The falsifier clears its bar on the A100: 202 µs saved against 130.
- None of the correct shapes comes close to it:
  - The best, one warp per slot, saves 76 µs on the A100, 1.2 % of the 6,512 µs force call.
  - The run-aligned walks the conductor chose are no faster than today's kernel, or slower.
- By the conductor's rule (a real kernel keeping under about 130 µs of the 202), **C1 stops here**.

**Why the falsifier overstates the gain.** The falsifier's speed comes from dropping the atomics' code, not only
their memory traffic:

- **Registers on sm_80.** sm_80 has no vector float atomics, so today's forward issues 64 scalar `atom.global`
  instructions and keeps 32 per-lane address registers. It compiles to 128 registers. Without the atomics it needs
  72 registers and 366 SASS instructions instead of 508.
- **The run-aligned walks** need the slot ranges and more registers: 64 to 95.
  - With stores masked off, the nested-loop form still takes 413 µs on the A100, against the falsifier's 264.
  - The single-loop form, with a store site at each run end and at the program end, compiles to 95 registers and
    564 SASS instructions on sm_80, and takes 555 µs.
- **Writing every slot's 61 MB block** costs about the zero fill (36 µs on the A100, 180 µs on the 5060 Ti). No
  correct kernel avoids that write.
- **The other GPUs** compile today's forward with vector atomics (16 `red.v4` on sm_90 and sm_120) to 64 and 80
  registers. The register effect is specific to sm_80.

**Ordering constraint found on the way:** the keys are sorted only before the neighbour list's device count. The
cutoff tail and the padding pairs follow it in matscipy order (`skin.py` `compact`, stable). Any slot-owning kernel
must take its slot ranges from that sorted head. Today's atomic walk is correct for any order.

## C2: the first table packed (idea 19)

The falsifier makes one load per column and edge: row 0 of the value table for every term of the value and slope
series, and one scalar for the density column. The arithmetic is unchanged.

| Kernel (µs) | A100 copy → one load | 5060 Ti copy → one load |
| --- | --- | --- |
| forward | 466 → 390 (**−76**) | 721 → 665 (−56) |
| backward, recompute (the A100's default) | 468 → 376 (**−92**) | 824 → 606 (−218) |
| backward, power (the H200's and 5060 Ti's default) | 473 → 404 (−69) | 713 → 623 (**−90**) |

Census: global loads per edge and lane fall from 8×64-bit + about 20 scalar to 3×64-bit + about 16 scalar.
Registers are unchanged or lower, except the sm_80 recompute backward, which goes from 80 to 92. Shared memory is
unchanged.

**Against the bars** (each GPU's default backward, plus the forward):

| GPU | Saving | Bar | Result |
| --- | --- | --- | --- |
| **A100** | 76 + 92 = **168 µs** | 130 µs | **over** |
| 5060 Ti | 56 + 90 = 146 µs, about 0.75 % of a 19.3 ms step | 2 % (about 390 µs) | under |
| H200 (no run) | about 33–49 µs | 68 µs | under |

How the H200 figure is estimated:
- **Backward:** the A100's measured power-backward fraction (14.6 %) applied to the H200's 211 µs gives 31 µs.
- **Forward:** the H200 forward is issue-bound (63 %), and its sm_90 instruction count falls only 2.2 % (372 →
  364). The bounds run from 2 µs (instructions only) to 18 µs (the A100's 16 % fraction) on its 112 µs.

C2 was therefore not stopped by its falsifier. The conductor scoped the real build to the A100 family (11 November):
a packed table with its own certificate, plus the 3× float32 gate.

### C2's real form: the piece-count bound (offline, CPU)

PACKED8 packs one quadratic per piece and takes the slope from it. The first table's backward needs the slope of all
64 columns at every edge, and the current fit holds values and slopes to 1e-7 of their scale. The bound uses
`c2-bound/c2_piece_bound.py` (output `c2-bound/output.txt`):

- **Input:** the newest cached 66-column table on the PC, `radial-tables/f8a363ff….pt`: 9 pairs, 619 pieces,
  6 terms, certificate worst 0.36.
- **Method:** for each piece it finds the smallest split into m equal parts after which the dropped Chebyshev terms
  fall under the tolerance. Term k of a piece shrinks as m^-k when the piece is split.
- **Normalisation:** each column's largest value (or slope) over the whole domain. This is more lenient than the
  fit's local scale, so the counts are lower bounds.

| Tolerance | Value as a quadratic | Slope as a quadratic | Slope as a cubic | Slope pieces from 1 Å (quadratic / cubic) |
| --- | ---: | ---: | ---: | ---: |
| 1e-7 (today's fit) | 1,402 pieces | 30,355 | 18,502 | 15,001 / 9,617 |
| 4.8e-7 (PACKED8's 22-bit a0) | 984 | 18,841 | 12,413 | 9,374 / 6,469 |
| 1e-6 | 825 | 14,906 | 10,501 | 7,455 / 5,435 |

**Reading the table.**
- The values pack well: 1,402 pieces × 66 columns × 8 B × 9 pairs is 6.4 MB.
- The slopes do not. Differentiation multiplies term k by about k², and the cutoff's last piece (r ≈ 5.98 Å) needs
  a 28–75-fold split.
- A packed slope table at 1e-7 therefore holds 9,600–15,000 pieces in the physical region alone. That is 61 MB as
  an 8-byte quadratic, or 140 MB with the values and as a cubic, against today's 15 MB. Below 1 Å it takes as many
  again.
- Both exceed the A100's 40 MB L2, and each edge's lookup lands on a random piece. The falsifier's one cached load
  per column cannot be reached by a certified packed slope.
- What remains is the value-only packing for the forward (6.4 MB). The falsifier bounds it at 76 µs, under the
  130 µs bar.

**Verdict: C2 stops (bound).** A table that is coarser in the slopes would need a looser certificate than today's,
and the falsifier's 168 µs would not survive the extra DRAM traffic.

## Files

- `census/`: the offline census of every variant (sm_80, sm_90, sm_120).
- `rtx5060ti/`: the first run, with host time included. `rtx5060ti-r2/` to `-r4/` add variants as they were built.
- `a100-alex-launch-incl/`: the first A100 run, with host time included; superseded.
- `a100-alex/`: the falsifiers.
- `a100-alex-aligned/`: the run-aligned walk.
- `a100-alex-walk/`: the single-loop walk.
- `c2-bound/`: C2's piece-count bound (script and output).
- `job5060.sh`: the local job script.
