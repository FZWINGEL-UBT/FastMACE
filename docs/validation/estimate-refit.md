# Rule 12's estimate refitted for the memory rescue (31 October 2026)

**Verdict: the estimate stays within 1.08–1.39 times the measured peak at every stage of the memory rescue, where it
was 2.3–2.6 times with every record on.** Rule 12 now plans only the rungs a cell needs, and gives gate_epilogue back
once the chunked region is on.

## What was wrong

On the 20 GB RTX 4000 Ada, the default plan for NbCl5 at 121,824–137,376 atoms ran at a steady 12.7–14.2 GiB
reserved. Rule 12 predicted 31.8–35.8 GiB. So it judged every cell short by far and planned all five rungs, with
gate_epilogue off, at 1,717–1,953 ms per step.

The planner's own breakdown (`est_diag.py`) and a matrix of measured peaks per rescue stage
(`est_matrix.sh`, below) showed two causes:

- **Edge count.** Rule 12 counted edges at uniform density: 33.0 per atom for NbCl5 and 43.6 for NaCl at 6 Å. The
  crystals have 26.3 and 32.0. Every per-edge term was 1.26–1.36 times too large; `SAFETY = 0.90` absorbed it for the
  default plan.
- **The rescue records as absolute bytes.** Their terms were measured savings, credited after `SAFETY`. The default
  plan's overstatement (about 85 kB per atom at 1.2–1.3 times the peak) sits in tensors the rescue never touches, so
  it stayed as an absolute excess while the rungs removed 70 % of the peak. The ratio grew to 2.3–2.6.

No term grows with the cell: estimate and peak are both constant per atom from 21,600 to 137,376 atoms.

## The change

- **Counted edges** (`planner.memory.counted_edges`): rule 12 counts the pairs within `r_max` in the structure (both
  directions, with the cell's periodicity, `ase.neighborlist`), plus the 5 % edge bucket. This takes 0.7 s at 21,600
  atoms and 7 s at 137,000 on one CPU core, once per plan. The neighbour list itself is built only by the engine,
  after the plan. `model_edges` (uniform density) remains for other callers.
- **The rescue records as fractions** (`MemoryTerm.keep`): each memory-rescue record, and gate_epilogue, declares the
  fraction of the remaining activations it keeps. The remainder is `SAFETY` times the activations and the additive
  terms. The estimate multiplies these fractions into it, so its overstatement shrinks with the peak.
- **Fitted fractions** (least squares, every stage, both crystals; target: the default plan's ratio):

| Record | Keeps | Measured on its own (NbCl5, NaCl) |
|---|---:|---|
| `recompute_conv_weights` | 0.62 | 0.68, 0.64 of the peak |
| `recompute_first_table` | 0.966 | 0.95–0.96, 0.97–0.98 |
| `recompute_node_segments` | 0.648 (gate_epilogue's extra separate) | 0.77, 0.70 with gate_epilogue on |
| `recompute_node_region` | 0.663 | 0.61–0.70, 0.70–0.73 |
| `gate_epilogue` | 1.04; 1.17 with the separate segments; 1 with the region | 12.3 kB per atom; 36 kB per atom; 0.5 kB per atom |

  The whole ladder keeps 0.281–0.283 of the default plan's peak for NbCl5 and 0.286 for NaCl. The fit gave 1.182
  for gate_epilogue with the separate segments; 1.17 keeps NaCl at 46,656 atoms under 1.4.
- **`SAFETY` 0.90 → 0.97**: the lowest value that keeps all 60 small cases of the 19-label fit at or above their peak
  with counted edges (0.96 left one at 0.996).

## Estimate over measured peak

The peak is the largest reserved over build, warm-up and steady state:
- the 60 small cases and the 28 capacity cases as before, with their edges now counted (`rule12_fit.json`,
  `rule12_capacity.json`: `uniform_edges` keeps the old count, `r_max` the cutoff it implies);
- the new rescue matrix (`rule12_rescue.json`): capacity_scan points with every rescue record set, 30 steps after 10.
  NbCl5 ran on leeloo (RTX 4000 Ada), NaCl on trillian (RTX A4000), FastMACE 242e0ef3.

| Cases | Count | Now | Before |
|---|---:|---|---|
| Fit: 19 labels, NaCl and NbCl5 3×3×3 | 60 | 1.01–1.21 | 1.05–1.27 |
| Capacity: MH-1, 2,000–30,600 atoms | 28 | 1.03–1.28 | 1.05–1.34 |
| Rescue, NbCl5, default plan | 2 | 1.30 | 1.29 |
| Rescue, NbCl5, + conv_weights / + first_table | 4 | 1.20–1.22 | 1.38–1.43 |
| Rescue, NbCl5, + node_segments / gate_epilogue off | 6 | 1.08–1.27 | 1.64–1.95 |
| Rescue, NbCl5, + node_region (gate_epilogue off or on) | 8 | 1.21–1.26 | 2.25–2.44 |
| Rescue, NaCl, default plan | 1 | 1.33 | 1.37 |
| Rescue, NaCl, + conv_weights / + first_table | 4 | 1.30–1.34 | 1.48–1.52 |
| Rescue, NaCl, + node_segments / gate_epilogue off | 5 | 1.32–1.39 | 1.94–2.07 |
| Rescue, NaCl, + node_region (gate_epilogue off or on) | 8 | 1.25–1.32 | 2.37–2.57 |

- `tests/cpu/test_memory_fit.py` keeps every rescue case in [1.0, 1.4], the capacity fit's band. It also checks that
  the data covers the whole ladder and at least three sizes per crystal with the region.
- The conductor asked for about 15 %. With general terms, the floor is the small fit cases at 1.0 (they set
  `SAFETY`), and the per-record fractions differ by up to about 7 % between the two crystals (node_segments,
  gate_epilogue). So the band is 1.08–1.39, not 1.0–1.15.

## What rule 12 plans now

`rungs.py`, the default plan without GPU work. On 20 GB, the free memory is 19.37 GiB:

| Cell | Atoms | Memory | Rungs planned |
|---|---:|---|---|
| NbCl5 6×6×28 | 72,576 | fits | conv_weights, first_table, node_segments (gate_epilogue on) |
| NbCl5 6×6×33 | 85,536 | fits | the same, gate_epilogue off |
| NbCl5 6×6×40 | 103,680 | fits | all, node_region on and gate_epilogue back on |
| NbCl5 6×6×47 | 121,824 | fits (16.49 GiB) | all, gate_epilogue back on |
| NaCl 20×20×20 | 64,000 | fits | conv_weights, first_table, node_segments |
| NaCl 22×22×22 to 24×24×24 | 85,184–110,592 | fits | all, gate_epilogue back on |

Before, NbCl5 121,824 atoms planned every rung with gate_epilogue off (estimate 31.8 GiB, "warn").

**GPU check.** These are the default plans with nothing forced: leeloo, RTX 4000 Ada (20 GB), `capacity_scan.py point`,
60 steps after 20, FastMACE 5eba9c80. The memory model is the same as the committed one, except that
gate_epilogue's fraction with the separate segments was 1.182 (now 1.17); the planned rungs are the same. Each point
planned exactly the rungs in the table above, and each satisfies the capacity check: the exact initial edge count,
and live edges equal to a fresh rebuild after the steps.

| Cell | Atoms | Rungs | Estimate | Steady peak reserved | ms/step | Before (all rungs, gate_epilogue off) |
|---|---:|---|---:|---:|---:|---:|
| NbCl5 6×6×28 | 72,576 | three, gate_epilogue on | 17.10 GiB | 14.49 GiB | 852.6 | |
| NbCl5 6×6×33 | 85,536 | three, gate_epilogue off | 17.27 GiB | 13.63 GiB | 1,016.0 | 1,163 (region forced) |
| NbCl5 6×6×40 | 103,680 | all, gate_epilogue on | 14.07 GiB | 10.74 GiB | 1,421.8 | |
| NbCl5 6×6×47 | 121,824 | all, gate_epilogue on | 16.49 GiB | 12.56 GiB | 1,679.5 | 1,716.5 |

At 85,536 atoms the plan without the region takes 1,016 ms per step, against 1,163 ms with it (0.87 times; the matrix
point with the region). At 121,824 atoms, gate_epilogue's return saves 2.2 %. The ms/step comparisons come from
separate processes, about 3 % noise, so they are not decision-66 timings.

## Records

`results/estimate-refit-20261031/`:
- `matrix-nbcl5-leeloo/`, `matrix-nacl-trillian/`: the rescue matrix points;
- `ladder-before-leeloo/`, `ladder-after-leeloo/`: the default-plan ladders before and after the refit;
- the scripts: `est_matrix.sh`, `matrix_cases.py` (points to fit cases), `data_update.py` (counted edges in the fit
  data), `refit2.py` (the least-squares fit), `est_diag.py` (the planner's breakdown) and `rungs.py`.
