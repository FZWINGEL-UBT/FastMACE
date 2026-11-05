# What limits FastMACE on NaCl 216 atoms (RTX 5060 Ti, 5 November 2026)

OMAT-0 medium, float32 hybrid, the RTX 5060 Ti's default plan, NaCl 3x3x3 (216 atoms, 7,680 edges within 6 Å, 35.6
per atom), Langevin 300 K, 1 fs, skin 0.5 Å; FastMACE `fastmace/round5` at `b03fc94e`. MD step 1.10–1.18 ms
(`results/nacl-compare-rtx5060ti-20261105/`). Tools: `tools/step_breakdown.py` (`breakdown/`, 500 steps) and
`tools/module_profile.py` (`sources/`, 20 calls).

**Where a step goes**

| Part | ms per step |
|---|---:|
| Force call (graph replay; kernels 0.89 ms, 183 kernels, 190 per replay) | 0.936 |
| of it: gaps between kernels inside the replay (189 gaps) | 0.076 |
| Langevin half kick, positions, neighbour check, edge compaction | 0.054 |
| Noise upload | 0.012 |
| Neighbour rebuilds (`rebuild_cost.py`: 31 in 800 steps at skin 0.5, fit 0.015 ms per step) | about 0.015 |

**Kernel time by kind (µs per force call, launches):**
- The two edge kernels of interaction 2: the fused forward 145 (grid 9 x 4 x 2 = 72 blocks of 384 threads on 36 SMs,
  one resident per SM) and the w′ backward 151 (1,920 blocks): 295 µs, a third.
- Everything else: about 150 small kernels, 0.55–0.6 ms. Atom-level GEMMs over 216–1,512 rows (28 launches, 128 µs;
  estimated occupancy 3–25 %), atom-level elementwise (81 launches, 128 µs, about 1.6 µs each), the first
  interaction's tabled sums and radial table (59 µs), symmetric contractions (6 launches, 30 µs), copies and fills
  (16 launches, 17 µs). 133 of the 152 ops take under 10 µs each (280 µs together).

**Verdict:** at 216 atoms the step is a chain of small, latency-bound kernels. The GPU is mostly waiting on launch
ramp-up and tails, not on arithmetic or memory: two thirds of the force call is kernels too small to fill 36 SMs,
plus 8 % gaps. The edge kernels cost per edge what they do on large cells. Neither the neighbour list nor the
host limits it.

**Skin A/Bs** (`ab_rewrites.py`, 8 blocks of 1,024 steps, both orders, side a skin 0.5): skin 1.0 0.973 / 1.005,
skin 1.5 0.972 / 0.963: within the run-to-run noise of this short step; with the rebuild at about 1 % of the step,
there is little to gain. A graph-captured asynchronous rebuild (drafted, not committed) was dropped for the same reason.

**What could help here (not built):** fewer, larger kernels for the atom-level chain of each interaction
(linear, gate, product, linear, readout in one kernel per interaction; the earlier megakernel estimate put the
NaCl ceiling at 8–16 %, `docs/design/2026-10-20-megakernel-research.md`); programmatic dependent launch between our
own kernels to hide ramp-up (never Inductor's, which gave wrong forces inside our graphs); and for the edge kernels a
launch shape that fills more than 72 blocks on small cells.
