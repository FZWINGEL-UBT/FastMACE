# Late pullback follow-up: do the lower registers rescue a rejected switch? (offline census, 12 Nov 2026)

Owner, via the conductor: "do the lower registers maybe allow another optimization we rejected?" CPU only, no GPU run.
`census_followup.py` → `census_followup.json` (sources `sources/fu-*.cu`). Counting as in the README's step 2: the
edge loop with the channel passes, plus the STAGE_DOUT prologue amortised (1.41 runs per 32-edge block).
Blocks per SM: 128 threads per block, from registers (per-warp allocation in units of 256), shared memory (+1 KiB per
block; 228 KiB per SM on sm_90, 164 KiB on sm_80) and the 2,048-thread limit.

| plan | source | arch | regs | channel pass × passes | per edge (incl. prologue) | vs lp | blocks/SM |
|---|---|---|---:|---|---:|---:|---:|
| H200 | lp (skip + LATE_PULLBACK + hoist; today's H200 default) | sm_90 | 56 | 372 × 4 | 1,826 | 1.000 | 9 |
| H200 | lp + CHANNEL_VECTOR 2 | sm_90 | 78 | 699 × 2 | 1,721 | 0.942 | 6 |
| H200 | lp + CHANNEL_VECTOR 4 | sm_90 | 167 | one pass (no loop) | 1,781 | 0.975 | 3 |
| H200 | lp | sm_80 | 64 | 374 × 4 | 1,799 | 1.000 | 7 |
| H200 | lp + CHANNEL_VECTOR 2 | sm_80 | 92 | 697 × 2 | 1,692 | 0.941 | 5 |
| H200 | lp + CHANNEL_VECTOR 4 | sm_80 | 168 | one pass | 1,781 | 0.990 | 3 |
| A100 | a100-lp (STAGE_EDGES + LATE_PULLBACK) | sm_80 | 64 | 428 × 4 | 1,890 | 1.000 | 6 (shared) |
| A100 | a100-lp + SKIP_OUTSIDE + HOIST_DX | sm_80 | 64 | 374 × 4 | 1,687 | **0.893** | 6 (shared) |
| A100 | a100-lp (for reference) | sm_90 | 48 | 429 × 4 | 1,938 | | 9 |
| A100 | a100-lp + SKIP_OUTSIDE + HOIST_DX | sm_90 | 48 | 372 × 4 | 1,735 | | 9 |

- **A100: SKIP_OUTSIDE + HOIST_DX on top of LATE_PULLBACK costs no registers** (64 → 64) and cuts 10.7 % of the
  instructions. Without LATE_PULLBACK the pair went 79/96 → 128 registers and lost (Colab A100 lab 1.025). The A100
  plan's occupancy stays at 6 blocks per SM, limited by shared memory (23,296 B), not registers. The float64 emulation
  of the source (before the hoist) matches the direct contraction to 1.6e-9 (`fu_check.py` → `fu_check.json`).
  The production order (render with STAGE_EDGES and `skip_outside=True`, then LATE_PULLBACK, then HOIST_DX;
  `a100_skip_render.py` → `a100_skip_render.json`) gives the same: 10 paths transformed, 64 registers, 1,643
  instructions per edge loop, emulation 1.6e-9.
  **Queued on Alex (conductor):** `jobs/alex_lp_skip.sh`, after 4988692. It runs the gate with all three on, the MD-step
  A/B in both orders (a = LATE_PULLBACK, b = + SKIP_OUTSIDE + HOIST_DX; SrTiO3 10³, NbCl5), and the stress-mode pair
  (below).
- **H200, CHANNEL_VECTOR 2:** −5.8 % instructions, but 56 → 78 registers (9 → 6 blocks per SM). Before LATE_PULLBACK,
  skip+vec2 was −6.4 % against skip+hoist (1,864 against 1,990, at 128 registers) and measured 1.082 slower in the
  lab. The register cost is smaller now, but the instruction gain is about the same and occupancy still drops by a
  third; the H200 lab measured lp at 0.859 against a census of 0.901, which suggests occupancy matters.
  **Census only, not run** (conductor: no paid run; the same shape as skip+vec2, which measured 1.082). Not
  emulated (the emulator does not read vector types).
- **H200, CHANNEL_VECTOR 4:** 167 registers and 3 blocks per SM: closed by count.
- **PREFETCH:** 5 November's PREFETCH is the fused forward's cp.async double buffer of its staging; the w′ kernel has no
  counterpart. Its per-edge operands are the sh row (staged by STAGE_SH), the receiver's dout row (staged by STAGE_DOUT)
  and the sender's x row read per channel; loading the next edge's x row early would need a second register set
  for 4 × 4 values per lane (or shared staging per sender). No easy form, so not censused: closed.


## Slim dsh output (conductor's addition; no GPU)

With LATE_PULLBACK every w′ kernel's dsh is zero in 13 of its 16 components (l = 0, 2, 3); only the l = 1 slots carry
G/√3. Both of OMAT-0 medium's w′ kernels take the transform (proof counter APPLIED = 2 per build in the H200 gate).
Per call on SrTiO3 10³ (370,000 edges, float32):

| what | today | slim | saved |
|---|---:|---:|---:|
| w′ dsh store, per w′ kernel | 64 B/edge | 12 B/edge (degree 1) | 52 B/edge; two kernels: 104 B/edge, 38.5 MB |
| autograd sum of the sh gradients (interaction 0's edge outer sum + two w′ dsh) | 3 × [E, 16] | unchanged: autograd needs [E, 16] gradients | none without new plumbing |
| edge_sh3_backward (Triton, one edge per thread) | reads v 12 B + g 60 B, writes 12 B: 84 B/edge; about 15 FMUL + 70 FP for the l ≤ 3 polynomial | unchanged: interaction 0's edge outer sum still fills all 16 | none |

- **What the slim store alone is worth:** 38.5 MB per force call, about 8 µs on the H200 (4.8 TB/s), 0.26 % of the
  3.07 ms step, and about 19 µs on the A100. That is under the 1 % bar.
- **Plumbing:** getting even those bytes needs the w′ backward to hand autograd something other than an [E, 16]
  gradient for sh. A gradient for a [E, 3] slice view gets scattered back into a zero [E, 16] by autograd (no
  saving). Returning dL/dv for the edge vectors instead is EDGE_FORCE's route, and EDGE_FORCE lost on 11 November for
  exactly this plumbing: AOT autograd filled in zero gradients for the compiled region's `vectors` and `edge_attrs`
  outputs (+76 B/edge), and interaction 0's edge outer sum kept edge_sh3's backward (A/B 1.03 slower on the A100).
  LATE_PULLBACK makes the w′ side of that route trivial (G is already dL/dv's angular part: no Jacobian epilogue,
  so EDGE_FORCE's +132 to 150 instructions per edge go away). It does not touch the two causes of the loss.
- **Where the real saving is:** only with EDGE_FORCE's step 3 (interaction 0's edge outer sum also contracted into
  dL/dv). Then every dsh and edge_sh3's backward go away (A100: 36 µs kernel time, sh-recompute), plus the 104 B/edge
  above. Not a census item; it stays an open idea under EDGE_FORCE.

## Stress mode (conductor's addition)

LATE_PULLBACK changes the edge-derivative path the virial uses. NVE MD (ab_rewrites) has no strain leaf, so the next
paid A/B adds `tools/virial_ab.py` (the force call with stress an NPT step or a cell relaxation makes), now
generalised to `--a-constant` / `--b-constant MODULE:NAME=VALUE` and `--b-expect` proof counters. In
`jobs/alex_lp_skip.sh` it compares a = today's A100 default against b = LATE_PULLBACK + SKIP_OUTSIDE + HOIST_DX on
SrTiO3 10³, in both orders, and records the stress and forces of b against a. The gate already covers stress
accuracy (hydrostatic and full cell modes; H200 stress ratios 0.85 / 1.30 / 0.54).

`loops_dump.py` prints the loop structure behind these counts.

## Local RTX 5060 Ti check (12 Nov 2026, `local-5060ti/`)

Through `tools/ci/local_gpu.py run` with the H200 plan alias (source 01bc2471; not evidence for the H200 or A100):
- Kernel lab (SrTiO3 10³ capture): base (skip+hoist) 4,122.3 µs at 96 registers, lp 4,108.0 µs = 0.997 at 64
  registers; dL/dv max 7.3e-7 (median edge 4.6e-7), dx 1.9e-7.
- 3× float32 gate with LATE_PULLBACK on (NaCl, SrTiO3 3×3×3): passes; APPLIED advanced by 2 per build, no
  expect failures; forces 0.82 / 0.80, stress 0.73 / 0.38 of the ASE float32 spread.

## Alex A100-SXM4-80GB results (12 Nov 2026, `alex/`)

**4988692** (source 01bc2471; the A100 default then: no SKIP_OUTSIDE / HOIST_DX / WEIGHTED_VECTOR; b = LATE_PULLBACK):
- Kernel lab: base 2,630.7 µs at 79 registers, lp 2,320.4 µs = **0.882** at 64; dL/dv 6.6e-7 (median edge 4.7e-7), dx 2.2e-7.
- Gate (LATE_PULLBACK on): passes on NaCl and SrTiO3 3³ (forces 0.87 / 0.89, stress 0.94 / 0.52), no expect failures.
- MD-step A/B b/a: SrTiO3 10³ **0.9633 / 0.9607** (6.19 → 5.96 ms), NbCl5 **0.9648 / 0.9670**; proof counter advanced on b only.

**4988974** (source c21ad219; a = LATE_PULLBACK, b = LATE_PULLBACK + SKIP_OUTSIDE + HOIST_DX):
- Gate with all three on: passes on NaCl, NbCl5, SrTiO3 3³, no expect failures.
- MD-step A/B b/a: SrTiO3 10³ 0.9908 / 0.9882, NbCl5 0.9841 / 0.9868. The run is flagged INVALID only because
  ab_rewrites requires side a's proof counter not to advance, and side a had LATE_PULLBACK on by design (both sides
  transformed). The timings stand: SKIP_OUTSIDE + HOIST_DX add about 1.1 to 1.6 % on top of LATE_PULLBACK. The census
  (0.893 on the kernel) suggested more. With the w′ kernel about 30 % of the step (4988692: kernel 0.882 → step 0.963),
  0.893 on the kernel would be about 0.967 on the step.
- Stress mode (virial_ab, SrTiO3 10³; a = the A100 default at c21ad219, b = all three on): force call with stress
  0.9969 / 1.0027 (18.1 ms per call: a tie; kernel time per atom 1,368 → 1,288 ns, −6 %, which the call's other work
  hides). Stress of b against a: max 1.5e-8 against a stress of 0.029; forces 2.3e-6.

**Deciding job 4990803** (source 4063180f = main 1d491a67 + this branch; the new A100 default SKIP_OUTSIDE + HOIST_DX +
WEIGHTED_VECTOR against LATE_PULLBACK, which keeps SKIP_OUTSIDE + HOIST_DX and turns WEIGHTED_VECTOR off): gate, MD-step
A/B in both orders (SrTiO3 10³, NbCl5), stress-mode pair; one job (owner). Results go here.

## Two channels per lane on top of LATE_PULLBACK (owner's follow-up via the conductor; CPU census only)

Idea: each lane takes two consecutive channels into the same pull_Gx/Gy/Gz/pull_R quartet, float2 loads of x and the
stage row, float2 dx atomics on sm_90. That is exactly `fu-h200-lp-vec2.cu` (`wprime_tuning.vec(lp, 2)` duplicates only
the dx accumulators; pull_G and pull_R are declared outside the channel loop and stay shared). Falsifier: stop above 96
registers on sm_90 or below 6 % fewer instructions per edge than LP+skip+hoist. `vec2_categories.py` →
`vec2_categories.json`:

| sm_90, per edge | lp (skip + LP + hoist) | lp + 2 channels per lane | change |
|---|---:|---:|---:|
| instructions (edge loop + channel passes) | 1,779 | 1,674 | −105 (−5.9 %) |
| with the prologue | 1,826 | 1,721 | −5.8 % |
| registers (blocks per SM) | 56 (9) | 78 (6) | |
| FP (FFMA/FMUL/FADD) | 1,274 | 1,274 | 0 |
| memory | 245 | 159 | −86 (LDS −80, LDG −8, REDG −8) |
| integer, move, branch, other | 239 | 225 | −14 |
| edge loop outside the channel passes | 291 | 276 | −15 |

sm_80: 1,754 → 1,647 (−6.1 %; with the prologue 1,799 → 1,692, −5.9 %), 64 → 92 registers (7 → 5 blocks per SM);
FP 1,270 both, memory 245 → 157.

**Stopped by the falsifier: −5.8 % on sm_90, under the 6 % line (registers 78, inside the 96 limit).** What the census
shows:
- The FP work is per channel and does not shrink: 1,274 of 1,779 instructions (72 %). Pairing channels saves only the
  shared-memory reads of the staged row (float2 instead of two scalars), a few global loads and the dx atomics.
- The per-edge items the idea counts on are not per channel: the n loads, the spline locator and table offsets, the
  edge-loop indexing and the butterfly already run once per edge per warp, so two channels per lane cannot halve
  them (the edge loop outside the channel passes goes 291 → 276).
- The register cost (56 → 78) takes the H200 from 9 to 6 blocks per SM. The H200 lab showed LP gaining more than its
  census (0.859 against 0.901), most likely from occupancy, so losing a third of it works against a 5.8 % count.
- This matches why vec2 lost before even at 72 registers (1.106): its saving was memory instructions too, while it
  added the per-pair overhead. No lab, gate or Alex entry.
