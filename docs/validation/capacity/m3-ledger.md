# M3: where the memory sits at the new limit (31 October 2026)

Measured only. Setup: the default float32 plan with `recompute_conv_weights`, `recompute_node_segments` and
`recompute_node_region` on; local RTX 5060 Ti (16 GB, WSL2), `mace_cueq_oeq`, FastMACE 32280b07 (main 5c895e37 plus
the launcher).

Tools:
- `tools/bench/memory_ledger.py`: the allocator's owners at steady state, the memory outside the allocator, and a
  re-capture with the allocator's history (the force call's peak live set by allocation site);
- the capacity points' per-phase peaks (results/chunked-region-20261031/points/).

The four-record ledger, with `recompute_first_table`, follows when fastmace/rebuild-a merges.

## Capture and warm-up transients: not the limit any more

Per-phase peaks of the limit points (GiB; reserved by the allocator):

| Cell | Records | Build | Warm-up and capture | Steady |
|---|---|---|---|---|
| NaCl 67,200 | segments | 11.85 | **13.91** | 11.93 |
| NaCl 70,400 | segments | 12.38 | **14.54** | 12.49 |
| NaCl 97,336 | region | 14.31 | 14.49 | 14.55 |
| NbCl5 82,944 | segments | 14.64 | 14.66 | 14.74 |
| NbCl5 101,088 | region | 14.39 | 14.57 | 14.63 |

- **Where the transient was.** Only NaCl with the separate segments had one: about 2 GiB above its steady state at
  capture. That set its limit (70,400 atoms).
- **With the region.** The capture peak equals the steady peak to within 0.1 GiB in every structure. A capture-side
  fix, such as capturing a smaller bucket first or freeing warm-up buffers, would help only the separate segments,
  which are no longer the limit.

## The ledger near the limit

| Owner (GiB) | NbCl5 95,904 atoms | NaCl 85,184 atoms |
|---|---:|---:|
| Graph pool (the force call's activations and transients) | 14.42 | 12.74 |
| Outside the allocator: context, libraries, kernels | 1.27 | 1.27 |
| Outside the allocator: graph executables | 0.31 | **1.34** |
| Neighbour list | 0.33 | 0.42 |
| Default pool: cache and other live blocks | 0.58 | 0.56 |
| Model, noise batch, state and engine buffers | 0.19 | 0.18 |

The re-captured force call peaks at 11.84 GiB live for NbCl5 (132.5 kB per atom) and 10.70 GiB for NaCl. The
largest allocation sites alive at that peak, per atom:

| Site | Per atom | NbCl5 GiB | NaCl GiB | What it is |
|---|---:|---:|---:|---|
| `gated_products_input_grads` (9 blocks) | 23.0 kB | 2.21 | 1.96 | the region input's gradient, the chain's input gradients (backward output, needed) |
| `fused_folded_message` | 20.5 kB | 1.96 | 1.74 | interaction 1's raw message `[N, 5120]`, kept for the normalization's backward |
| Inductor `mul_74, add_22, truediv_7` | 20.5 kB | 1.96 | 1.74 | interaction 1's normalized message `[N, 5120]` (message over the edge density), the region's saved input |
| Inductor `cat … message` / `mul_68, add_17, truediv_6` | 2 × 8.2 kB | 1.57 | 1.40 | interaction 0's raw and normalized message `[N, 2048]` |
| cuEquivariance `uniform_1d` | 8.2 kB | 0.79 | 0.70 | the contraction's `[N, 2048]` output |
| `radial_table_forward` | 260 B per edge | 0.69 | 0.75 | the radial table's `a [E, 65]` (rebuild-a's target) |

## Ranked targets and ceilings

The ceiling is the cell size the 16 GB card would reach if the term were gone. It scales the limit by the per-atom
part of the usage (about 14.4 GiB of the 15.9 at NbCl5 95,904) and assumes the peak stays at the same moment.

| # | Target | Saves at the peak | NbCl5 ceiling (from 101,088) | NaCl ceiling (from 97,336) | Exactness |
|---|---|---|---|---|---|
| 1 | **The message normalization inside the chunked region**, interactions 0 and 1: the region takes the raw message and the density, and recomputes the normalized message per chunk, so it is never kept | 20.5 + 8.2 kB per atom | about 120,000–129,000 (+20–28 %) | about 116,000–124,000 | rounding (Inductor fuses differently), deterministic |
| 2 | Graph executables outside the allocator on NaCl (1.34 GiB; NbCl5 0.31): two B graphs kept against one | 1.0 GiB on NaCl | – | about 105,000 (+8 %) | exact |
| 3 | The radial table's `a` (rebuild-a, `recompute_first_table`) | 260 B per edge | about 106,000 (+5 %) | about 103,000 (+6 %) | in progress, neighbour-list session |
| 4 | The contraction's `[N, 2048]` output | 8.2 kB per atom | about +6 %, only if the contraction moves into a chunked region too | – | rounding |

Ceilings are estimates from the ledger, not measurements.

## Proposal

**Target 1:** move each interaction's message normalization (`message × factor / density`, the Inductor kernels
above) into `recompute_node_region`'s chunked region.
- The region's inputs become the raw message and the density instead of the normalized message.
- The normalization runs per chunk inside the checkpoint, so its output is rebuilt in the backward, one chunk at a
  time.
- The raw message stays saved; the division's backward needs it or the density.
- Expected: 28.7 kB per atom less at the peak, about +20–28 % atoms on 16 GB.
- Rounding class; same record, no new planner rung.
- Falsifier first:
  - peak at NbCl5 21,600 and 43,200 atoms against the current region (region_probe.py);
  - GO if the peak falls by at least 20 kB per atom at no more than 1.05 times the region's call;
  - then the limit points.

Target 2 is a separate, small engine question (keep only the current edge bucket's force graph on NaCl). It is
worth a look after target 1.

Records: `results/m3-ledger-20261031/` (the ledger JSON of both cells and the launcher).
