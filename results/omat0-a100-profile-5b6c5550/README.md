# OMAT-0 A100 profile at 5b6c5550 (5 November 2026)

- **Machine:** Modal, NVIDIA A100 80GB PCIe (driver 610.57.04; Modal assigned the PCIe card for `--gpu A100-80GB`),
  image recipe C hybrid (`pip-freeze.txt`).
- **Source:** `5b6c5550` (FastMACE main after round 4: VEC_SH, VEC_A and TC_WFRAG on the A100).
- **Model and structure:** MACE-OMAT-0 medium, float32, hybrid backend, default plan; Benchmark_loup SrTiO3
  10×10×10 (5,000 atoms).
- **Command:** `modal run tools/modal/fastmace_modal.py::job --gpu A100-80GB --timeout-min 60 --commit 5b6c5550
  --command "bash tools/modal/omat0_a100_profile.sh"`; wall 373 s, about $0.26 (`modal.json`).

## Benchmark protocol

Each `loup/` run is one 20-step block in a fresh process.

| Ensemble | µs per atom and step (three runs) | At `2e9a3878` (same PCIe card type) |
|---|---|---|
| NVE | 2.377, 2.389, 2.393 | 2.50 |
| Langevin | 2.357, 2.355, 2.343 | 2.47 |

That is about 5 % faster than the previous PCIe profile.

## Step breakdown

`breakdown/` and `sources/` record the step breakdown (Langevin, 300 K, 1 fs, skin 0.3 Å):
- force graph: 12.49 ms per replay;
- eager force call: 11.72 ms in 223 kernels.

| Bucket | µs per call | Launches |
|---|---|---|
| `fused_folded_forward` (interaction 2) | 3,175 | 1 |
| `oeq_projected_backward_wprime` | 3,109 | 1 |
| CUTLASS `fm_tf32x3` GEMMs | 1,811 | 41 |
| Triton atom-level kernels: gates, residuals, layout copies around the GEMMs | 1,035 | 55 |
| `tabled_outer_sum` forward and backward (interaction 1) | 849 | 2 |
| cuEq symmetric contraction | 402 | 4 |
| Edge geometry and radial glue | 307 | 8 |
| `radial_table` forward and backward | 209 | 3 |
| Copies, fills, cats | 145 | 30 |
| Scatters | 114 | 11 |

The kernels in the Triton atom-level row include:
- the species skip's selection (`cat_index_mul_sum`, 137.5 µs);
- the pure layout copies `[N, 2048]` ↔ `[N·dim, 128]` (`clone_slice_*tf32x3_mm`, 162 µs; `cat_view`, 182 µs);
- the copies fused with residual adds or per-node divisions (93 µs).

The `IndexBackward0` of MACE's head selection `[arange, heads]` runs a radix sort, 34 µs over 4 launches.
