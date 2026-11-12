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
