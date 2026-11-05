# Atom-chunked combined node region: `recompute_node_region` (31 October 2026)

**Verdict: GO, added as the last memory-rescue record.**

- **The record.** `recompute_node_region` joins each interaction's `gated_rows` chain and product block into one
  checkpoint region, run in 4 atom chunks.
- **Memory.** The force call's peak is 0.77 times the separate segments' (`recompute_node_segments`), 35 kB per
  atom less, at 1.22–1.24 times their force call.
- **Capacity.** On the 16 GB RTX 5060 Ti, NbCl5 goes from 82,944 to 101,088 atoms (+22 %), and NaCl from
  70,400 to 97,336 atoms (+38 %).
- **Planning.** It is the last record of `MEMORY_RESCUE` and never replaces the separate segments.
- **Exactness.** Rounding: bitwise equal in eager on the CPU. The gate passed.

Ninth list, item 2 (plans/capacity-ideas-20261030.md): the fix CAP-T08 named. There, the combined region kept
35 kB per atom less, but its whole-region recomputation was the backward's peak
([capacity-T08-chunks.md](capacity-T08-chunks.md)). Atom chunks cut that recomputation's working set.

## The change

- **`segments.modules.region_forward`** runs the region in `REGION_CHUNKS` atom chunks (`torch.tensor_split`). Each
  chunk is its own non-reentrant checkpoint, and the outputs are concatenated. The region does not mix atoms, so the
  per-atom results are those of the unchunked region.
- **A graph break, fixed.** The region built its checkpoint function with `types.MethodType` while Dynamo traced.
  That broke the compiled graph at the region, in CAP-T08's version too, and the chunked loop could not trace after
  the break. `join_region` now binds it once (`region_body`). The unchunked combined region now costs 1.20 times the
  separate segments, where CAP-T08 measured 1.24 (step time).
- **The record `recompute_node_region`.**
  - Requires `recompute_node_segments`.
  - Its apply undoes the separate segments' checkpoints (`modules.unrecompute`) and joins each interaction's chain
    and product block (`join_region`, 4 chunks).
  - Self-check: each region against the stored chain, reshape and product block.
  - Memory term: `_NODE_REGION_SHARE = 0.11` of the stock per-node floats (35.2 kB per atom of MH-1's 317,440 B).
- **Planner.**
  - `RewriteConfig.recompute_node_region`, auto = off.
  - `MEMORY_RESCUE = (recompute_conv_weights, recompute_node_segments, recompute_node_region)`.
    `recompute_first_table` from fastmace/rebuild-a goes second when it merges.
  - Rule 12 plans it only for a cell that cannot fit with the separate segments. The out-of-memory rebuild turns it
    on last, one record per rebuild.
- **Chunk count: 4 for every cell.** The screens below found the lowest peak at 4 chunks at every size, so there is
  no size rule (`segments.REGION_CHUNKS`, an int for A/B sides).

## Chunk count and peak

Setup:
- `results/chunked-region-20261031/region_probe.py`: the default float32 plan with `recompute_conv_weights` and
  `recompute_node_segments` on, one process per layout.
- One eager call of the compiled force call: the allocator's peak above the call's start, and the median of 5 calls.
- NbCl5 on the RTX 4000 Ada (leto, NAS `mace_cueq_oeq`, FastMACE bcc71592). The peak is the same on the RTX 5060 Ti
  (21,600 atoms: identical to the MiB).

| Atoms | Separate segments | Combined, 1 chunk | 2 chunks | **4 chunks** | 8 chunks | 16 chunks |
|---|---|---|---|---|---|---|
| 9,216 | 1.335 GiB, 103.0 ms | 1.348, 1.20× | 1.129, 1.22× | **1.033, 1.24×** | 1.242, 1.31× | – |
| 21,600 | 3.126 GiB, 238.5 ms | 3.157, 1.20× | 2.644, 1.21× | **2.417, 1.24×** | 2.907, 1.31× | 2.891, 1.29× |
| 43,200 | 6.251 GiB, 476.1 ms | – | 5.287, 1.20× | **4.834, 1.22×** | 5.813, 1.26× | 5.782, 1.31× |

- **The gain per atom is constant.** Four chunks peak at 0.773–0.774 times the separate segments at every size:
  35.2 kB per atom less each time.
- **More chunks keep more.** Eight or 16 chunks peak higher again, so per-chunk overhead outweighs the smaller
  working set beyond 4.
- **The call time** is about 1.2 times the separate segments' at every chunk count. The chain, reshape and product
  run twice; CAP-T08's separate segments cost 1.10.

## Correctness

- **CPU:** `tests/cpu/test_recompute_node_segments.py` on the tiny MH-1 model in float64, for the combined region
  with 1 and 3 chunks and for the record itself:
  - every region passes its self-check, and a row-mixing fault fails it;
  - a deep copy rebinds the stored and region functions;
  - energy and forces are bitwise equal to the record off, in eager.
- **Gate:** `tools/force_stress_gate.py` (leto, RTX 4000 Ada) with `recompute_conv_weights`,
  `recompute_node_segments` and `recompute_node_region` on: passed.

| Structure | Energy ratio | Forces ratio | Stress ratio |
|---|---|---|---|
| NaCl | 0.752 | 0.885 | 0.989 |
| NbCl5 | 0.113 | 1.045 | 0.339 |
| NaNbO3 | 0.278 | 1.000 | 0.517 |

## Capacity: limit points on the RTX 5060 Ti (16 GB)

Setup:
- `tools/bench/capacity_scan.py point`, default plan, 200 steps.
- `results/chunked-region-20261031/points.sh`, local WSL2, `mace_cueq_oeq`, FastMACE b5a07ab2.
- "region": `recompute_conv_weights`, `recompute_node_segments` and `recompute_node_region` on.
  "segments": the same with `recompute_node_region` off.
- Every passing point satisfies the new edge check (below): at step 0, exactly the reference edge count; after the
  steps, the live count equal to a fresh ASE rebuild.

| Cell | Atoms | Layout | Status | ms/step | Steady peak reserved |
|---|---:|---|---|---:|---:|
| NbCl5 6×6×32 | 82,944 | segments | ok | 918.5 | 14.74 GiB |
| NbCl5 6×6×35 | 90,720 | segments | out of memory at build | | |
| NbCl5 6×6×34 | 88,128 | region | ok | 1,145.4 | 12.88 GiB |
| NbCl5 6×6×37 | 95,904 | region | ok | 1,248.6 | 13.96 GiB |
| NbCl5 6×6×39 | 101,088 | region | ok | 1,345.0 | 14.63 GiB |
| NbCl5 6×6×41 | 106,272 | region | out of memory at build | | |
| NaCl 20×20×21 | 67,200 | segments | ok | 749.0 | 11.93 GiB (capture 14.54) |
| NaCl 20×20×22 | 70,400 | segments | ok | 784.2 | 12.49 GiB (capture 14.54) |
| NaCl 21×21×21 | 74,088 | segments | out of memory at capture | | |
| NaCl 23×23×23 | 97,336 | region | ok | 1,325.6 | 14.55 GiB |
| NaCl 23×23×24 | 101,568 | region | out of memory in warm-up | | |

- **NbCl5:** 82,944 atoms with the separate segments, 101,088 with the region (+22 %). CAP-T08 had found 69,984 on
  its main. Current main fits 82,944 with the separate segments, so main had already moved the limit.
- **NaCl:** 70,400 atoms with the separate segments, 97,336 with the region (+38 %). CAP-T08's main failed at
  70,400. The separate segments' capture is NaCl's peak: 14.54 GiB reserved at 70,400 atoms, against a steady
  12.49.
- **Step time at the limit** grows with the cell, as in CAP-T08. 1,345 ms at 101,088 atoms is 13.3 µs per atom.
- **The WSL driver** reports an out-of-memory error during capture as "CUDA driver error: device not ready /
  Failed to create GPU mapping".

## The capacity check (conductor, 31 October 2026)

`capacity_scan.check_point` had accepted a point whose edges per atom were within 5 % of the small copy's. Every
NbCl5 point failed it. The list starts at exactly the reference count (26.28 per atom), but grows 5.4 % in 200 steps
at 300 K by thermal motion; NaCl grows 3.1 %. No drift tolerance now:

- **(a)** at step 0 the list has exactly the small copy's count;
- **(b)** after the steps, the live at-cutoff count equals a fresh rebuild at the current positions (ASE on the CPU,
  independent of FastMACE's list);
- **forces** are nonzero.

`tools/bench/tiled_capacity.py` runs one force call, so it uses (a); the tiled session has been told.

## Records

`results/chunked-region-20261031/`:
- `local-5060ti-21600/`: the first screen;
- `ada-leto/`: the chunk screens at three sizes;
- `points/`: the limit points' JSON;
- `region_probe.py`, `local.sh` and `points.sh`.

The gate is in leto `/mnt/homes/zwingel/fastmace-capacity/chunked-region-gate/`. Its summary is above.
