# PACKED8 per structure: does a cold lattice need the float4 table? (packed8-auto, 12 November 2026)

Lane packed8-auto (branch `fastmace/packed8-auto`). Question from the ledger row `packed8-table`, which was OPEN as
"structure-dependent: crystal 1.060, disordered 0.882": can a cheap per-structure choice keep PACKED8's win and
avoid its loss?

## What the 1.060 was, and what is left of it

- The 1.060 / 0.882 pair was **packed8 S12**, the first decode (about 19 instructions per weight). Its successor
  **packed8v2** (same values, about 14 instructions, no integer-to-float conversions) gave 1.004 / 0.858 in the
  same kernel lab and is production's `spline_forward.PACKED8`. It has been on for `PACKED8_GPUS` (H200, A100,
  RTX 4000 Ada, RTX 5060 Ti) since 9–11 November. On the 300 K and 1500 K MD-step A/Bs it wins on every family
  (H200 0.993 / 0.982, A100 0.956 / 0.959).
- The lab's "crystal" was SrTiO3 after 300 MD steps from the ideal lattice at 300 K (`spline_table_bound.py
  --equilibrate-steps 300`), not the ideal lattice. A loss could survive only in a structure colder than that.

## Step 1: why the table cost depends on structure (offline census)

`slab_census.py` (numpy, CPU) counts the table slabs (species pair, spline interval; about 20 KB each) that SrTiO3
10×10×10's edges touch on production's Hermite grid (736 intervals; 0.021 Å wide beyond 2.5 Å, where most edges
are). Thermal states are modelled as Gaussian noise σ = 0.08 Å · √(T / 300 K). Tiles hold 256 edges in receiver
order, by length ([slab_census.json](slab_census.json)):

| State | Distinct slabs | Footprint | Slabs per tile | Edges per slab in a tile |
| --- | ---: | ---: | ---: | ---: |
| ideal lattice | 20 | 0.4 MB | 16.9 | 15.1 |
| 1 K (σ 0.005 Å) | 70 | 1.4 MB | 35.0 | 7.3 |
| 10 K | 189 | 3.7 MB | 68.5 | 3.7 |
| 50 K | 390 | 7.6 MB | 111.7 | 2.3 |
| 300 K | 862 | 16.8 MB | 165.3 | 1.55 |
| 1500 K | 1,646 | 32.1 MB | 201.8 | 1.27 |
| 0.3 Å rattle (lab "disordered") | 2,226 | 43.5 MB | 211.5 | 1.21 |

Edge-length disorder against the interval width drives everything. A 300 K displacement is about four
intervals wide, so the shells are already smeared and the table read is L1-cold. Only a lattice below roughly
50 K keeps several edges per slab in a tile. Edges per slab in a tile is the cheap predictor: one `unique` over
the locator's slab ids at capture time. It only matters if some GPU actually loses there, so the MD step was
measured next.

## Step 1b: the MD step on cold structures

`cold_ab.sh`: same-process MD-step A/B (`tools/ab_rewrites.py`), both build orders, OMAT-0 medium, Benchmark_loup
settings (1 fs, skin 0.5, warm-up 200, 8 blocks of 256 steps), Langevin at T from the ideal lattice, source
4f02f077 (main). Side a = today's defaults (PACKED8 on), side b = `PACKED8=False`. **b/a > 1 means PACKED8 is
faster.**

| GPU | Structure | 1 K | 50 K | 300 K |
| --- | --- | --- | --- | --- |
| RTX 4000 Ada (murphy) | SrTiO3 10×10×10 | 1.129 / 1.119 | 1.126 / 1.119 | 1.124 / 1.118 |
| RTX 4000 Ada (murphy) | NbCl5 | 1.065 / 1.059 | 1.071 / 1.074 | 1.075 / 1.076 |
| A100-SXM4-80GB (Alex job 4990120) | SrTiO3 10×10×10 | 1.072 / 1.073 | 1.087 / 1.089 | 1.092 / 1.082 |
| A100-SXM4-80GB (Alex job 4990120) | NbCl5 | 1.045 / 1.046 | 1.058 / 1.056 | 1.050 / 1.052 |
| H200 | both | not run: Modal over the month cap; the owner chose to wait (12 Nov). 300 K known: 0.9930 / 0.9945 (b = on) | | |

Ada: PACKED8 wins by the same margin at 1 K as at 300 K. Its forward is L1-traffic-bound (lab zero-table 0.716),
so halving the bytes pays even when only a few slabs are live ([ada-murphy/](ada-murphy/)).

A100: PACKED8 still wins 7 % at 1 K on SrTiO3 (300 K: 8–9 %). Cold shells take about 1.5 points of the gain, as
the census predicts, but the A100's forward is L2-bound (`step_limiters-20261109`), and the packed table stays
ahead ([a100-alex/](a100-alex/), a0731, source 4f02f077, 21 min).

## Verdict: no auto-choice; the row is closed

- On the two families measured cold (Ada, A100), PACKED8 is faster at 1, 50 and 300 K on both structures, by
  4.5–13 %. There is no loss for a per-structure choice to avoid, so none was built and no defaults change.
- The H200 is the only family where a cold-lattice loss is still possible. Its forward is issue-bound, and the lab
  put packed8v2 at 1.004 on a 300 K-start snapshot. Rough bound on that loss: the decode's own cost. S12's decode lost
  37 µs net on that snapshot, and v2's is about a quarter cheaper. If an ideal lattice saved no bytes at all, v2
  would lose about 30–40 µs, roughly 1 % of a 3.5 ms step (an estimate, not measured). That case needs only an MD run below about 50 K on the H200. The predictor,
  if it is ever needed: edges per slab in a 256-edge tile at capture (ideal 15, 1 K 7.3, 50 K 2.3, 300 K 1.55).
  An H200 cold A/B (1 K, 50 K; `cold_ab.sh`, about $1) was approved but not run: this session's permission check
  blocked a launch over the Modal month cap, and the owner chose to wait. It is the follow-up if cold H200 runs
  matter.
- Ledger: `packed8-table` (S12) closed as superseded by `packed8v2-table`, with the cold-lattice A/Bs recorded there.

## Scripts

- `slab_census.py`: the offline census (`python results/packed8-auto-20261112/slab_census.py > slab_census.json`).
- `cold_ab.sh`: the A/B; `ada_run.sh`: its workstation wrapper (environment, GPU-idle check);
  `alex_packed8_cold.sh`: its Alex sbatch wrapper (slim source tarball of 4f02f077).

## CPU suite

Full `tests/cpu` on the merged tip de9f0fab (git archive), leto, GPU hidden, `-n 8 -m "not acceptance"`: 5,739 passed,
205 skipped, 0 failed (13 min, 12 November 2026).
