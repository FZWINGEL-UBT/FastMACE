# Rebuild-`a`: the first interaction's radial table rebuilt in the backward (`recompute_first_table`)

Capacity idea rebuild-`a` ([plans/capacity-ideas-20261030.md](../../plans/capacity-ideas-20261030.md)), through the
conductor. The falsifier in [speed-def.md](speed-def.md) ("The rebuild-`a` falsifier") said to build it: with the other two rescue records on, the first interaction's
radial table output (`[E, h c + 1]`, 260 bytes per edge for MH-1 float32) is one of the largest per-edge tensors the
force call still keeps for the backward.

## Verdict

**The record.** `RewriteConfig.recompute_first_table` (`fastmace.rewrites.radial`) runs the first interaction's
radial table and reassociated message (`first_table_region`) under non-reentrant `torch.utils.checkpoint` inside the
compiled graph:

- the forward keeps only the region's inputs;
- the AOT partitioner rebuilds only what the backward needs, the table output (`radial_table_forward`, a custom op);
- the energy and forces are bitwise those of the plan without it (CPU test, and its self-check passes with a relative
  error of 0).

**Class and default.** Exact, deterministic, off by default. It needs `radial_tables` and `reassociate_first`, and it
has the fixed-species workload constraint of `radial_tables`. Rule 12 turns it on only when the plan does not fit with
`recompute_conv_weights` on. It is the second of the three rescue records
(`planner.memory.MEMORY_RESCUE = (recompute_conv_weights, recompute_first_table, recompute_node_segments)`). The
estimator credits it the table's output per edge (`rewrites.memory.recompute_first_table`). `Simulation`'s
out-of-memory rescue turns it on in that order.

**Memory.** At the same cell, with the other two rescue records on in both, the graph pool is 0.69–0.75 GB smaller.
The force call's live peak drops by 1.10–1.16 GB (ledger below).

**New limits** on the 20 GB RTX 4000 Ada, MH-1 float32. The record is forced off or on in the same ladder on the same
card; rule 12 turns on `recompute_conv_weights` and `recompute_node_segments` on both sides:

| Structure | Off | On | Change |
| --- | ---: | ---: | ---: |
| NbCl5 | 114,048 atoms (6×6×44) | **119,232 atoms (6×6×46)** | +4.5 % |
| NaCl | 108,800 atoms (20×20×34) | **115,200 atoms (20×20×36)** | +5.9 % |

On each side the next size ran out of memory: off at NbCl5 6×6×45 and NaCl 20×20×35, on at NbCl5 6×6×47 and NaCl
20×20×37. The ladder ran before main's `recompute_node_region` (chunked-region). That record now comes after this one
in the rescue order, so these limits are without it.

**Cost.** In the same-process A/B with the record forced on, b/a is 1.018–1.029 at three of the four cells on both
GPUs, but **1.036–1.041 at NaNbO3 8,640 atoms**: above the 3 % the conductor set. NaNbO3 has the most edges per atom
of the three structures, so the rebuilt table is a larger share of its step. The record is a rescue record, so only
cells that would otherwise not run pay this cost.

**The gate passed** on NaCl, NbCl5 and NaNbO3 with the record forced on, on both GPUs.

## Runs

- **Machines.**
  - Local PC: RTX 5060 Ti 16 GB (sm_120, driver 610.88, WSL2), environment `mace_cueq_oeq` (hybrid backend).
  - leeloo: RTX 4000 Ada 20 GB (sm_89, driver 595.99.02), NAS environment `mace_cueq_oeq`.
- **Model and precision.** MH-1 (`mh1-r2scan`), float32, torch 2.14.1+cu130.
- **Code.** FastMACE `a3069ef3` (fastmace/rebuild-a before its merges of main `83ab90d1` and `5c895e37`).
  - The A/Bs compare plans in one process, so there is no research-engine side.
  - Structures come from the baseline export `research-engine-baseline-20261017` (`d2faa9fd`).
- **Results.** [results/rebuild-a-20261031/](../../results/rebuild-a-20261031/): `local/`, `leeloo/` and `leeloo-ladder/`. The job
  scripts are `local.sh` and `leeloo.sh`.

### Limit ladder (`tools/bench/capacity_scan.py point`, leeloo)

Each point is 20 warm-up and 60 timed steps with `--rewrites recompute_first_table=off|on`
([leeloo-ladder/](../../results/rebuild-a-20261031/leeloo-ladder/)). Every completed point passed the edge check:
edges per atom within 3.6–4.0 % of the reference, against a 5 % tolerance.

| Cell | Atoms | Off: ms/step, peak reserved | On: ms/step, peak reserved | On/off time |
| --- | ---: | --- | --- | ---: |
| NbCl5 6×6×38 | 98,496 | 1,139.9, 17.83 GB | 1,178.1, 16.86 GB | 1.034 |
| NbCl5 6×6×40 | 103,680 | 1,212.9, 18.73 GB | 1,246.2, 17.70 GB | 1.027 |
| NbCl5 6×6×42 | 108,864 | 1,275.2, 19.61 GB | 1,310.5, 18.54 GB | 1.028 |
| NbCl5 6×6×44 | 114,048 | 1,340.7, 20.55 GB | 1,376.8, 19.44 GB | 1.027 |
| NbCl5 6×6×46 | 119,232 | out of memory | 1,439.7, 20.37 GB | |
| NaCl 20×20×28 | 89,600 | 1,094.4, 16.84 GB | 1,125.4, 15.81 GB | 1.028 |
| NaCl 20×20×30 | 96,000 | 1,176.1, 17.97 GB | 1,207.7, 16.90 GB | 1.027 |
| NaCl 20×20×32 | 102,400 | 1,258.3, 19.21 GB | 1,292.0, 18.03 GB | 1.027 |
| NaCl 20×20×34 | 108,800 | 1,342.2, 20.35 GB | 1,377.9, 19.13 GB | 1.027 |
| NaCl 20×20×36 | 115,200 | out of memory | 1,462.6, 20.26 GB | |

The record saves 0.97–1.22 GB of peak reserved memory at the same cell. The times come from separate processes; the
A/B below is the record's cost.

**16 GB RTX 5060 Ti.** With the record on through the default plan (rule 12 turned on all three rescue records):

- NaCl 20×20×22 and 20×20×23 (70,400 and 73,600 atoms) ran, above CAP-T08's 67,200.
- NbCl5 6×6×28 and 6×6×29 (72,576 and 75,168 atoms) ran at 12.68 and 13.10 GB peak reserved, but the scan marked them
  invalid. After 200 steps at 300 K their edges per atom were 5.43 % above the reference, against the 5 % tolerance.
  At step 0 they matched it exactly (26.28), so the list grew during the 200 steps; it is not a wrong list. The
  60-step ladder points stay at 3.9 %. (Main's `check_point` now checks the initial edges, with the live ones
  separately.)
- The on/off ladder for this card was withdrawn to free the local GPU; the conductor took the 20 GB card's on/off
  ratio as the measure.

### Memory ledger

**Tool.** `tools/bench/memory_ledger.py --rewrites` at the CAP-T08 limits. Both sides have `recompute_conv_weights`
and `recompute_node_segments` on; the "on" side adds `recompute_first_table`.

| GPU | Cell | Edges (padded) | Graph pool off → on | Peak live over start off → on | Max allocated off → on |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX 5060 Ti | NbCl5 6×6×27, 69,984 | 1,935,360 | 11.96 → 11.27 GB | 10.98 → 9.88 GB | 11.52 → 10.45 GB |
| RTX 5060 Ti | NaCl 20×20×21, 67,200 | 2,257,920 | 11.77 → 11.02 GB | 10.69 → 9.53 GB | 11.32 → 10.18 GB |
| RTX 4000 Ada | NbCl5 6×6×27, 69,984 | 1,935,360 | 11.96 → 11.27 GB | 10.93 → 9.83 GB | 11.42 → 10.35 GB |

- **The expected saving** is the table output, 260 bytes per padded edge: 0.50 GB for NbCl5 and 0.59 GB for NaCl.
  - The live peak dropped about twice that. I did not trace why.
- **Memory outside the allocator** is the same on leeloo (0.31 GB either way). On the PC it moves by ±0.4 GB between
  runs with the Windows display's use.
- **leeloo's max reserved** at the first captures is higher with the record (14.19 against 12.60 GB) although its
  allocations are lower. The warm-up of the second edge bucket reserved 2.2 GB on top of the first capture's pool. That
  did not happen on the PC, and the steady-state pool is smaller on both GPUs.

### A/B (`tools/ab_rewrites.py --b recompute_first_table=on`)

a is the default plan, b forces the record on. Each structure runs in both build orders (ab = abab, ba = abba).

| Cell | Atoms | RTX 5060 Ti b/a (ab, ba) | RTX 4000 Ada b/a (ab, ba) |
| --- | ---: | --- | --- |
| NaCl 3×3×3 | 216 | 1.0290, 1.0235 | 1.0187, 1.0221 |
| NbCl5 3×3×3 | 1,944 | 1.0197, 1.0199 | 1.0179, 1.0177 |
| NbCl5 4×4×8 | 9,216 | 1.0275, 1.0276 | 1.0276, 1.0277 |
| NaNbO3 6×6×6 | 8,640 | **1.0407, 1.0403** | **1.0373, 1.0364** |

All blocks are valid. The A/Bs predate main's decision 66 (`gate_epilogue` on for these two GPUs), so neither side has
the gate epilogue.

### Gate

`tools/force_stress_gate.py` with `recompute_first_table=on`: NaCl, NbCl5 and NaNbO3 pass on both GPUs. Energy, force
and stress errors are within the float32 references' limits. `tests/gpu/test_kernels_triton_cuda.py` passes on the PC
(6 passed; 11 need `FASTMACE_BASELINE`).

## Self-check

**The first GPU build failed.** The self-check ran the whole first interaction on random node features. With
`species_tables`, `linear_up` is a lookup by the argmax of its input, and random features indexed past the table: a
device-side assert.

**The fix.** The self-check now runs the region alone on its own graph:

- 37 nodes, 720 sorted edges, every 12th beyond `r_max`;
- the species taken from the reassociation's columns.

It compares the message, the density and both input gradients against the table and reassociation run directly.
`test_the_selfcheck_runs_only_the_region` keeps it that way.
