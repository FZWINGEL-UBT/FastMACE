# HALF_FAR: far edges of the w′ kernel in packed half2 (11 November 2026)

Production build of idea 1 from the A100 screen (`results/a100-colab-ideas-20261111`, SPEC §19 entry at 82a36413).
Branch `fastmace/half-far-wprime`. The switch is `wprime_backward.HALF_FAR`, off by default
(`HALF_FAR_GPUS` is empty). Kernels are in `fastmace/kernels/oeq_streaming/half_far.py`.

## Step 1: CPU oracle (stop rule: 2× the float32 error)

`half_wprime_oracle.py` turns the float32 w′ kernel's own channel loop (captured A100 source) into numpy. It rounds
every statement to float16 the way the h2 kernel does, uses the kernel's exact pairing (consecutive far edges of one
receiver; an odd last edge stays float32), and adds the differences into a float32 MACE backward.

The error is measured against a float64 run, as a ratio to the float32 run's own error ("F" force, "S" stress).
OMAT-0 medium, rattled 0.05 Å. Structures: SrTiO3 4×4×4 and NbCl5 2×.
- `none` row: a second float32 run, i.e. the noise of the ratio; it reads 0.77–1.36.
- Scaling suffixes: `w` scales each table slab by a power of two (max |coef| → 2^10; `w4` → 2^4); `d` scales each
  receiver's dout row (max → 2^4); `r` keeps slope and radial-gradient sums in float32.

| variant | SrTiO3 5.5 Å | SrTiO3 5.8 Å | NbCl5 5.5 Å | NbCl5 5.8 Å |
| --- | --- | --- | --- | --- |
| all, no scaling | F 1.69 S 2.55 | F 1.04 S 2.03 | F 1.58 S 2.38 | F 1.05 S 2.45 |
| all/w (2^10) | F 1.66 S 1.63 | F 0.87 S 1.74 | F 11.6 S 4.36 (overflow) | F 7.58 S 2.85 (overflow) |
| all/r | F 0.99 S 1.75 | F 0.99 S 2.01 | F 1.18 S 2.47 | F 1.02 S 1.75 |
| all/wr | F 1.12 S 1.93 | F 0.95 S 2.22 | F 1.21 S 1.43 | F 1.03 S 0.91 |
| all/dr | F 1.05 S 0.82 | F 0.92 S 1.58 | F 1.27 S 2.03 | F 1.09 S 1.63 |
| **all/w4dr (built)** | **F 1.11 S 1.22** | **F 1.00 S 1.51** | **F 1.27 S 1.08** | **F 1.12 S 0.95** |

The table is collected from oracle_*, oracle2_*, oracle3_* and oracle4_* (.json/.log).
- Diagnosis at 5.5 Å on SrTiO3: the radial gradient (dr) dominates (dr only: F 1.59 / S 1.67; dx only 0.82; dsh
  only 1.02).
- About 68 % of the far table coefficients on NbCl5 are float16-subnormal, so the table needs a scale. A 2^10 slab
  scale overflows the float16 radial-gradient sum; 2^4 with float32 sums does not.
- The built design is slab scale 2^4, a per-receiver dout row scale 2^4, and float32 slope and radial-gradient sums.
  It is the only variant at or under 2× on both structures at both radii.

## Step 2: the kernel

- `wprime_half_pairs`: pairs the far edges and sets their skip flags, plus a flag per receiver that has a pair.
- `wprime_half_scale`: one warp per paired receiver; computes the power-of-two dout row scale.
- `wprime_half_far`: one warp per pair. Inputs:
  - x is packed from float32;
  - dout is scaled and broadcast;
  - the table comes from a float16 copy with per-slab scales; the copy adds about 32 MB, half the float32 table.
  The dual Horner pass runs in HFMA2. Slope and radial gradient are float32. Scales are undone in float32 on dx, dsh
  and dr.
- The float32 kernel skips the paired edges (`with_skip`).

Offline census (`census_half_far.py`, sm_80, captured A100 source):
- Loop: 714 instructions per pair, against 2 × 497 for the float32 loop on two edges (0.72).
- 80 registers at 3 blocks/SM, with small LDL/STL spills; `HALF_FAR_MIN_BLOCKS=2` gives 128 registers and 692 per pair.
- The far fraction of edges is 11–12 % at 5.8 Å and 22–26 % at 5.5 Å.

CPU tests: `tests/cpu/test_half_far.py` plus `test_wprime_backward.py`, 21 passed on leto (NVRTC compile checks
included).

## Step 3: Colab A100 gate and A/B

`colab_ab.sh` (launched by `launch_ab.sh`) runs:
1. gate_pack: SrTiO3 3×3×3, NaCl, NbCl5; the default, then hf58 / hf55 / hf58b2.
2. If the gate passes: ab_pack on SrTiO3 10×10×10 and NbCl5 3×3×3, both build orders.

Colab A100-SXM4-40GB (driver 580.82.07), commit eb0cda12, OMAT-0 medium, against today's A100 default plan.
13.2 CU (184.55 → 171.32), with 69 min of VM setup. Logs are in `ab/`.

**Gate** (`force_stress_gate.py`, ratio to the ASE float32 error; limit 3×):

| side | SrTiO3 F / S | NaCl F / S | NbCl5 F / S |
| --- | --- | --- | --- |
| default | 0.83 / 0.61 | 0.89 / 0.87 | 0.97 / 1.42 |
| hf58 (5.8 Å) | 0.98 / 0.52 | 1.07 / 0.84 | 0.99 / 1.34 |
| hf55 (5.5 Å) | 1.24 / 2.36 | **8.94 / 6.95 (fail)** | 1.05 / 1.66 |
| hf58b2 (5.8 Å, 2 blocks/SM) | 1.02 / 0.49 | 1.09 / 0.98 | 0.97 / 1.09 |

The hf55 NaCl failure is not explained; the oracle did not cover NaCl. `gate_pack.sh` logged exit=0 for hf55
anyway, so its exit code does not reflect a failed structure.

**MD-step A/B** (ab_pack v1, b/a; b = HALF_FAR):

| switch | SrTiO3 10×10×10 ab / ba | NbCl5 3×3×3 ab / ba |
| --- | --- | --- |
| hf58 | 1.0521 / 1.0538 | 1.0610 / 1.0606 |
| hf55 | 1.1056 / 1.1059 | 1.0790 / 1.0777 |
| hf58b2 | 1.0455 / 1.0462 | 1.0538 / 1.0511 |

**Verdict: REJECTED at stage 4.** It is slower in both orders on both structures, and more far edges (5.5 Å) make it
slower still. The census predicted about 0.72 of the float32 loop per far edge, so the extra cost comes from
outside the half loop. Likely causes, not profiled:
- Skipped edges leave their lanes idle in the float32 kernel (STAGE_EDGES blocks), so that kernel barely gets
  shorter.
- Two extra launches: one thread walks each receiver's edges in the pairing kernel, and the scale kernel reads the
  paired rows.
- The half kernel runs after the float32 kernel, not alongside it.

Retrying would need the far edges compacted out of the float32 kernel's edge list, plus a profile of the three
kernels. Not started: Alex confirmation and the leto suite; both were for adoption only.
