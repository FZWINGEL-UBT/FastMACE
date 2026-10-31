# Tiled call: tile size and bucket screen (ninth idea list, item 3)

The compiled tiled call over R ∈ {4,096, 8,192, 16,384} × bucket ∈ {0.15, 0.05}, against the current call
(R = 4,096, bucket 0.15).

**Rule, from the conductor:**

- GO if a configuration is ≤ 0.92× the current call, with a workspace ≤ 2 GiB and ≤ 5 slot changes in a 300-step
  `tiled_md` run.
- STOP if all are within 0.97×.

## Verdict: STOP (the conductor's ruling); the tiled call keeps R = 4,096, bucket 0.15

No configuration meets GO. The R = 16,384 configurations, at about 0.90×, are outside the 0.97× STOP band, but
their workspace (about 4.2 GiB) fails GO whatever their time. So the conductor called STOP.

| NbCl5 atoms | R | Bucket | Time / current | Workspace | Slots |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 21,600 | 4,096 | 0.05 | 0.993 | 1.07 GiB | 2 |
| 21,600 | 8,192 | 0.15 | 0.906 | 2.14 GiB | 2 |
| 21,600 | 8,192 | 0.05 | 0.922 | 2.11 GiB | 2 |
| 21,600 | 16,384 | 0.15 | 0.898 | 4.26 GiB | 2 |
| 21,600 | 16,384 | 0.05 | 0.853 * | 4.17 GiB | 2 |
| 51,840 | 4,096 | 0.05 | 1.000 | 1.08 GiB | 3 |
| 51,840 | 8,192 | 0.15 | 0.972 | 1.79 GiB | 1 |
| 51,840 | 8,192 | 0.05 | 0.971 | 1.76 GiB | 1 |
| 51,840 | 16,384 | 0.15 | 0.897 | 4.27 GiB | 2 |
| 51,840 | 16,384 | 0.05 | 0.907 | 4.18 GiB | 2 |

\* The current call ran slow in that pair (0.356 s against 0.342-0.355 s in the others). The two R = 16,384
calls themselves took 0.304 s and 0.306 s.

What the screen shows:

- **A smaller bucket buys nothing.** At R = 4,096 it adds a slot at 51,840 atoms and runs at 1.00×. Otherwise it is
  within the noise of bucket 0.15.
- **Larger tiles are faster, but only above the 2 GiB workspace.**
  - R = 16,384 is about 0.90× at both sizes, at about 4.2 GiB.
  - R = 8,192 is 0.91-0.92× at 21,600 atoms but 0.97× at 51,840. It sits around 2 GiB: 2.1 GiB at 21,600 atoms
    (two slots), 1.8 GiB at 51,840 (one slot).
  - The workspace grows about linearly with R: about 0.26 MiB per receiver.
- No configuration meets GO, so no 300-step slot-change run was made.
- The R = 16,384 configurations are about 0.90×, outside 0.97×. The gain is real, but it costs workspace that
  GO does not allow.

**Reading for a rule (not done here; the conductor's call).** The gain comes from fewer, larger tiles:

- fewer launches per sweep, and better GEMM occupancy;
- the halo fraction falls with R (halo ≈ 2,300-3,300 atoms per tile at R = 4,096).

A rule would choose R from free memory: the largest R whose slot workspace, about 0.26 MiB per receiver, fits a
budget. That is the memory model the conductor asked for. Dynamo's cap is not the binding term, at 1-3 slots
and 3 specialisations per slot (cache limit 64). Whether 2 GiB is the right budget is a capacity-versus-speed
choice: on murphy's 20 GB, 4.2 GiB of workspace costs about 0.7M atoms of capacity (6.3 kB per atom).

## Setup

- murphy, NVIDIA RTX 4000 Ada (20 GB); environment `mace_cueq_oeq`.
- MH-1 float32 hybrid, model `mh1-r2scan`; the default compiled evaluation plan; `TileConfig(mode="compiled")`.
- FastMACE 238a5163 (`fastmace/tiled-t3`, the T3 code with the force field's global list).
- Tool: `tools/bench/tiled_graphs.py screen`.
  - One process per size. Per candidate: Dynamo reset, then the current call and the candidate are built and
    warmed up (compilation, then one steady call whose peak above the allocated memory is the workspace).
  - Then 6 alternating blocks of 3 calls, the order flipped every block.
  - The ratio is the candidate's median block over the current call's.
- Forces against the current call: within 2.5e-6 eV/Å for every candidate.
- Runs: `results/tiled-screen-20261031/screen.sh`; records in `results/tiled-screen-20261031/screen-NbCl5-*`.
