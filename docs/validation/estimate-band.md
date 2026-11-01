# Rule 12's estimate per side and per stage (1 November 2026)

The owner's item, through the conductor: make rule 12's memory estimate more accurate. It was 1.01–1.21 times the
measured peak on the small cases, 1.03–1.28 on the capacity cells and 1.08–1.39 on the memory rescue's stages. The
target was 1.00–1.15 at every stage, never below 1.0, in this order: find the residual's cause and make the term
depend on the general quantity behind it; then a calibration probe; only then a tighter `SAFETY`.

## Verdict

**The crystal error is gone. What remains is the caching allocator's reserve, which no term can predict closer than
its own spread.** On cells of at least 15,000 atoms the estimate is never below the measured peak (reserved), and
per stage:

| Stage | Records | Before | Now |
|---|---|---:|---:|
| s0 | default plan | 1.27–1.44 | 1.19–1.28 |
| s1 | conv_weights | 1.18–2.01 | 1.00–1.05 |
| s2 | + first_table | 1.18–2.17 | 1.00–1.04 |
| s3 | + node_segments | 1.13–2.23 | 1.00–1.13 |
| s4 | gate_epilogue off | 1.08–2.06 | 1.00–1.24 |
| s5 | + node_region | 1.21–2.06 | 1.00–1.11 |
| s6 | region, gate_epilogue on | 1.17–2.08 | 1.03–1.11 |

These are five crystals (27.6–84.0 edges per atom), three GPUs on two platforms, 102 cells (`band.out`). "Before"
is `main` 35975577 on the same cells. Its upper end is the oxide perovskites: SrTiO3, NaTaO3 and NaNbO3 came out at
1.94–2.23 at the node stages, against 1.13–1.39 for NaCl and NbCl5, the crystals it was fitted on. So on `main` rule
12 turns the rescue's rungs on for these crystals at about half the cell size they need them. The small cases (60, 14 labels) stay at 1.006–1.210, and the capacity cells
(28) go from 1.03–1.28 to 1.03–1.24.

The held-out crystal, NaNbO3 (not in the fit), came out at 0.996–1.248. Its lowest point is s4 at 19,360 atoms,
0.4 % under its peak. The committed constants include it.

**The target band is not reached at s0, s3 and s4.** There the spread is the allocator's reserve, which differs
between crystals by up to 1.44/1.17 at one stage (below). The conductor's decision of 1 November: take this term
change now, and build a near-limit calibration probe as a separate item.

## Where the residual came from

**Method** (`residuals.py`, `per_unit.py`, `reserve.py`, `band.py`):
- split each case's estimate into its fixed, neighbour-list, per-edge and per-node parts;
- fit the measured live peak (max allocated over build, warm-up and steady) per stage as fixed + a × atoms + b × edges;
- compare both with the measured reserved peak, which is rule 12's target.

**1. The rescue records' fractions mixed the two sides.** estimate-refit made each record keep a fraction of the
whole remaining activations. But each record frees per-edge or per-atom tensors only. The measured live-peak slopes,
on cells of at least 15,000 atoms, fit within ±2.4 % at every stage:

| Stage | per atom (kB) | per edge (kB) |
|---|---:|---:|
| s0 default | 238 | 5.31 |
| s1 conv_weights | 231 | 0.43 |
| s2 first_table | 223 | 0.17 |
| s3 node_segments | 176 | 0.18 |
| s4 gate_epilogue off | 140 | 0.17 |
| s5 node_region | 90 | 0.19 |

- conv_weights removes 92 % of the per-edge part: the second interaction's [E, 1280] weights.
- first_table removes 260 B per edge, as the ledger says. What is left per edge is the radial basis and the indices.
- The node rungs remove per-atom tensors only.

So a crystal with more edges per atom (NaCl 33.6 against NbCl5 27.6) saved less at the node stages than a whole-
remainder fraction claimed: NaCl at node_segments was 1.36–1.39 against NbCl5's 1.16–1.18.

The live-peak model, fitted on four crystals, predicts the held-out NaNbO3's live peak within 0.99–1.01 at every
stage.

**2. MH-1's per-edge base was overstated more than its per-atom base.** At s0 the estimate counted 8.4 kB per edge
against 5.3 measured (1.58 times), but per atom only 1.36 times. Crystals with many edges per atom (SrTiO3, NaTaO3)
were therefore 9 % higher. The fused path's share of stock per-edge floats it does not keep (`_MH1_STREAMING_SHARE`,
float32) goes from 2,000 to 2,291 of 8,265. That brings the per-edge overstatement to the per-atom one. float64 is
unchanged (no float64 rescue matrix).

**3. The rest is the allocator's reserve.** Reserved/allocated is 1.07–1.44. For one crystal and stage it is stable
within ±1.5 % from about 15,000 atoms, and the same on Linux and WSL2 (NbCl5: s0 1.12 and 1.12, s2 1.21 and 1.22, s4
1.17 and 1.17 on the Ada and the 5060 Ti). It differs by stage, and at some stages by crystal:

| Stage | reserved / allocated |
|---|---:|
| s0 | 1.08–1.12 |
| s1 | 1.16–1.22 |
| s2 | 1.21–1.24 |
| s3 | 1.07–1.18 |
| s4 | 1.16–1.44 (SrTiO3 1.44, NaNbO3 1.43, NaTaO3 1.26, NbCl5 and NaCl 1.17) |
| s5 | 1.15–1.34 |
| s6 | 1.17–1.25 |

It is the graph pool's packing slack ([capacity-pool.md](capacity-pool.md)). An example: SrTiO3 at 46,305 atoms. s3
→ s4 lowers the live peak from 8.44 to 6.87 GiB, but reserved stays at 9.89 GiB and the pool at 8.95. On SrTiO3 and
NaNbO3, gate_epilogue off frees live memory that stays reserved. capacity-pool.md and capacity-workspace.md found no
allocator or Inductor setting that recovers it, so it was not tried again.

**No platform gap.** The 8–10 % "platform gap" in the first residual report came from the capacity cells, which
were measured on older code (29 October, b7e37b8b, 265 kB per atom at s0 against 223 now). With the same code, the
5060 Ti matches the Ada. The memory outside the allocator (1.2 GB on WSL2 with the desktop, 0.3 GB on Linux) is
already in the free memory rule 12 reads.

**Small cells.** Below about 15,000 atoms a fixed build transient of about 1.7 GB sets the live peak. The rescue is
never planned at such sizes, so these cells are left out of the rescue data. The build term covers them on the small
cases.

## The change

- `planner.memory.MemoryTerm`: `keep` becomes `edge_keep` and `node_keep`. `estimate` keeps per-edge and per-node
  remainders, and each rescue record scales its own side.
- `rewrites.memory`: conv_weights and first_table are edge-side rungs, keeping 0.080 and 0.382. node_segments,
  node_region and gate_epilogue are node-side: 0.644, 0.646, and 1.057 alone or 1.257 with the segments. These are
  the slope ratios between consecutive stages, all five crystals.
- `_MH1_STREAMING_SHARE[float32]`: 2,291 / 8,265.
- `planner.memory.RESCUE_RESERVE` and `rescue_stage`: a factor per stage, keyed by the deepest rung taken and not by
  crystal, on the kept remainder, shown as the "allocator reserve" term. It is the upper envelope over the matrix:
  the factor that puts each stage's lowest cell at its measured peak, rounded up to 0.005. Values: conv_weights
  0.855, first_table 0.85, node_segments 0.84, gate_epilogue off 1.0, node_region 0.9. The default plan has none
  (SAFETY's fit). Below 1 because SAFETY's overstatement of the default plan is larger than these stages' reserve.
- Rule 12's rung order and its tests are unchanged. On MH-1, turning gate_epilogue off still lowers the estimate by
  3.5–4.5 %. Two rung-order tests on the tiny model set the reserve neutral: fitted to MH-1's widths, it outweighs
  the rung on a model whose kept remainder is mostly per edge.

**Tests** (`tests/cpu/test_memory_fit.py`):
- every rescue cell lies in [1.0, its stage's highest ratio rounded up]: s0 1.30, s1 1.06, s2 1.05, s3 1.15, s4 1.25,
  s5 1.12, s6 1.12;
- the small cases lie in [1.0, 1.25] (was 1.5) and the capacity cells in [1.0, 1.25] (was 1.4);
- the data covers five crystals, at least three sizes each, three GPUs on Linux and WSL2;
- `test_memory_estimate.py` checks the per-side fractions and the reserve term.

## The ladder on leeloo

The default plans, nothing forced, with this change: leeloo, RTX 4000 Ada (20 GB), `capacity_scan.py point`, 60 steps
after 20, FastMACE 96e48685 (`ladder.sh`). The NbCl5 cells are estimate-refit's ladder; the SrTiO3 cells were added
above its default limit. Every cell planned its rungs by rule 12. None ran out of memory, and none fell back or
rebuilt.

| Cell | Atoms | Rungs planned | Estimate | Peak reserved | Ratio | ms/step | estimate-refit |
|---|---:|---|---:|---:|---:|---:|---|
| NbCl5 6×6×28 | 72,576 | conv, table, segments (gate_epilogue on) | 15.06 GiB | 14.49 GiB | 1.04 | 816 | same rungs, 17.10 GiB, 853 ms |
| NbCl5 6×6×33 | 85,536 | the same, gate_epilogue off | 16.89 | 13.63 | 1.24 | 1,015 | same rungs, 17.27, 1,016 |
| NbCl5 6×6×40 | 103,680 | all, gate_epilogue back on | 12.35 | 11.17 | 1.11 | 1,370 | same rungs, 14.07, 1,422 |
| NbCl5 6×6×47 | 121,824 | all, gate_epilogue back on | 14.46 | 13.05 | 1.11 | 1,620 | same rungs, 16.49, 1,680 |
| SrTiO3 21×21×21 | 46,305 | conv | 14.38 | 14.24 | 1.01 | 734 | |
| SrTiO3 25×25×25 | 78,125 | conv, table, segments (gate_epilogue on) | 17.31 | 16.63 | 1.04 | 1,386 | |
| SrTiO3 28×28×28 | 109,760 | all, gate_epilogue back on | 14.71 | 13.61 | 1.08 | 2,138 | |

- The peak is the largest reserved over build, warm-up and steady.
- The NbCl5 rungs are the same as estimate-refit's, and the estimates are 1.8–2.2 GiB lower.
- At 85,536 atoms the estimate is 1.24 times the peak. This is s4, where gate_epilogue off frees reserved memory on
  NbCl5 but not on SrTiO3 or NaNbO3, and the stage factor covers the latter.
- The ms/step values come from separate processes, about 3 % noise.

On `main`, SrTiO3 would take the rungs at about half these sizes. That was not run.

## What is left

- **The near-limit calibration probe** (the conductor's decision (ii)): an `EngineConfig` option, "auto" by default,
  that probes only when the term model's choice is within the reserve spread of the free memory, or when the next-
  cheaper rung might fit. It measures reserved on a 15–20k-atom replica with a warm cache: build 23–33 s per stage
  in these runs, plus about 4 s load. A separate item.
- `RESCUE_OVERSTATEMENT` (1.34, the capacity fit's largest ratio, which decides "short by far") is unchanged. The
  capacity cells now reach 1.24.

## Records

`results/estimate-band-20261101/`:
- the scripts: `est_matrix.sh`, the matrix per stage on the workstations (estimate-refit's, with the source path);
  `local_matrix.sh`, the same on the 5060 Ti through `local.sh`'s idle check; `residuals.py`, `split_fit.py`,
  `per_unit.py`, `reserve.py`, `band.py`; `ladder.sh`;
- the outputs: `band.out`, the final fit, every crystal; `band-heldout.out`, NaNbO3 held out; `band.json`;
  `reserve.out`; `per_unit.out`;
- the matrix points (JSON) stay on the NAS, `/mnt/homes/zwingel/fastmace-capacity/`: `band-SrTiO3-leeloo`,
  `band-NaTaO3-murphy`, `band-NaNbO3-trillian`, `band-NbCl5-5060ti`, and estimate-refit's `est-matrix-nbcl5` and
  `est-matrix-nacl`. Their cases are `tests/_support/data/rule12_rescue.json`.

Runs:

| Machine | GPU | Crystal | Sizes (atoms) |
|---|---|---|---|
| leeloo | RTX 4000 Ada | SrTiO3 | 10,985–78,125 |
| murphy | RTX 4000 Ada | NaTaO3 | 9,720–101,080 |
| trillian | RTX A4000 | NaNbO3 | 7,680–69,360 |
| local PC, WSL2 | RTX 5060 Ti 16 GB | NbCl5 | 10,800–28,800 |

- Environment: `mace_cueq_oeq`, hybrid, float32, MH-1 (`mh1-r2scan`).
- FastMACE `main` 35975577; 30 steps after 10.
- Earlier points: estimate-refit's NbCl5 (leeloo) and NaCl (trillian) matrices.
- The ladder: leeloo, this branch.
