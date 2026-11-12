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
  **Candidate for the A100 A/B** once Alex 4988692 settles LATE_PULLBACK there.
- **H200, CHANNEL_VECTOR 2:** −5.8 % instructions, but 56 → 78 registers (9 → 6 blocks per SM). Before LATE_PULLBACK,
  skip+vec2 was −6.4 % against skip+hoist (1,864 against 1,990, at 128 registers) and measured 1.082 slower in the
  lab. The register cost is smaller now, but the instruction gain is about the same and occupancy still drops by a
  third; the H200 lab measured lp at 0.859 against a census of 0.901, which suggests occupancy matters. A lab entry
  would settle it; not emulated (the emulator does not read vector types).
- **H200, CHANNEL_VECTOR 4:** 167 registers and 3 blocks per SM: rejected by count.
- **PREFETCH:** 5 November's PREFETCH is the fused forward's cp.async double buffer of its staging; the w′ kernel has no
  counterpart. Its per-edge operands are the sh row (staged by STAGE_SH), the receiver's dout row (staged by STAGE_DOUT)
  and the sender's x row read per channel; loading the next edge's x row early would need a second register set
  for 4 × 4 values per lane (or shared staging per sender). No easy form, so not censused.
- `loops_dump.py` prints the loop structure behind these counts.
