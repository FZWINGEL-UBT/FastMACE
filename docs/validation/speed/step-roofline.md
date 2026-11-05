# The large-cell step's heaviest kernels: roofline and the gated_input_grads tile screen (31 October 2026)

**Verdict: STOP.**
- None of the four heaviest kernel groups grows per atom with the cell.
- The outlier, `gated_input_grads` (30 % of FP32 peak), gains only 9–15 % from a larger tile. That is 1.6 % of the
  step at NbCl5 103,680 atoms and 2.4 % at 21,600 on the RTX 4000 Ada, below the 5 % GO bar. No code changed.

## Roofline

`results/step-roofline-20261101/op_profile.py` runs one eager call of the default plan's compiled force call under
the CUDA profiler, with the operators' input shapes, after releasing the captured graphs. The profiled kernels match
the captured step within 0.3 %. `roofline.py` derives each group's FLOPs and minimum DRAM traffic from the shapes:
the workstations allow no performance counters. Peaks: FP32 (SIMT) 26.7 TFLOP/s, DRAM 360 GB/s.

Setup:
- leeloo, RTX 4000 Ada, NbCl5, FastMACE d7054234;
- 21,600 atoms: the default plan has no memory-rescue record, step 206.4 ms;
- 103,680 atoms: conv_weights, first_table, node_segments and the region on, gate_epilogue on, step 1,340.7 ms.

| Group | Atoms | ms | Share | µs per atom | TFLOP/s | % of peak | Minimum GB/s | % of bandwidth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gated_input_grads` | 21,600 | 42.7 | 20.8 % | 1.977 | 7.96 | 30 % | 99 | 28 % |
| `gated_input_grads` | 103,680 | 204.9 | 15.2 % | 1.976 | 7.96 | 30 % | 99 | 28 % |
| `segment_gemm` (linear_2) | 21,600 | 27.7 | 13.5 % | 1.281 | 13.92 | 52 % | 109 | 30 % |
| `segment_gemm` (with the region's recompute) | 103,680 | 271.1 | 20.2 % | 2.614 | 13.24 | 50 % | 104 | 29 % |
| cuBLAS `sgemm` | 21,600 | 38.0 | 18.5 % | 1.758 | 12.77 | 48 % | 164 | 45 % |
| cuBLAS `sgemm` (with the recompute) | 103,680 | 264.0 | 19.6 % | 2.546 | 12.73 | 48 % | 156 | 43 % |
| `oeq_projected_backward_wprime` (w stored) | 21,600 | 20.0 | 9.7 % | 0.924 | ≥ 4.7 | ≥ 18 % | | |
| `wrebuild_backward` (w rebuilt) | 103,680 | 140.1 | 10.4 % | 1.351 | ≥ 3.2 | ≥ 12 % | | |
| `fused_folded_forward` | 103,680 | 51.9 | 3.9 % | 0.500 | ≥ 8.7 | ≥ 33 % | | |

The edge kernels count only their weight rebuild (2 × 65 × 1,280 FLOP per edge), so their figures are lower bounds.

- **No group grows per atom with the cell.** `gated_input_grads` costs 1.977 against 1.976 µs per atom. The step's
  rise from 9.6 to 12.9 µs per atom is the memory-rescue records' recompute: `segment_gemm` and cuBLAS run twice
  inside the region ([region-recompute.md](../capacity/region-recompute.md)).
- **`segment_gemm` and cuBLAS** sit where float32 SIMT matrix products usually do, about half of peak. The TF32 family
  was screened elsewhere.
- **`wrebuild_backward`** costs 0.43 µs per atom over the kernel that reads stored weights. That is
  `recompute_conv_weights`' capacity trade, and its weights are bitwise the forward's by design.
- **`gated_input_grads` is the outlier.** It runs the same product shape as `segment_gemm` (K = N = 512, atoms × D
  rows) at 0.6 times its rate. Its tile (32 atoms × 64 columns × 32 K, 4 warps, 3 stages) was chosen on 216–1,944
  atoms (OPT10-T01). At `segment_gemm`'s efficiency it would save 82 ms, 6.1 % of the step at 103,680 atoms.

## The tile screen

`results/gated-tile-20261101/tile_screen.py` covers the three gated segments of one `linear_2` input-gradient call
(D = 3, 5, 7):
- synthetic operands;
- tiles over ROWS (stacked rows) {64, 128, 256}, BLOCK_N {64, 128}, BLOCK_K {32, 64}, warps {4, 8}, stages {2, 3, 4};
- every tile checked against today's (128, 64, 32, 4, 3) within 1e-5 relative;
- interleaved blocks, medians.

The atom counts are 1,944 (OPT10-T01's cell), 21,600, and 25,920 (one region chunk at 103,680).

| GPU | Atoms | Today's tile | Best tile | Best / today | (128, 128, 32, 8, 3) / today |
|---|---:|---:|---|---:|---:|
| RTX 4000 Ada (leeloo) | 1,944 | 1.634 ms | (64, 64, 32, 4, 2) | 0.941 | |
| RTX 4000 Ada | 21,600 | 21.74 ms | (64, 128, 32, 4, 3) | 0.864 | 0.884 |
| RTX 4000 Ada | 25,920 | 25.68 ms | (128, 128, 32, 8, 4) | 0.887 | 0.897 |
| RTX A4000 (trillian) | 1,944 | 2.200 ms | (64, 64, 32, 4, 2) | 0.803 | |
| RTX A4000 | 21,600 | 29.95 ms | (128, 128, 32, 8, 4) | 0.858 | 0.859 |
| RTX A4000 | 25,920 | 36.10 ms | (128, 128, 32, 8, 3) | 0.849 | 0.849 |
| RTX 5060 Ti (local) | 1,944 | 1.547 ms | (64, 128, 32, 8, 3) | 0.984 | |
| RTX 5060 Ti | 21,600 | 18.49 ms | (64, 128, 32, 8, 3) | 0.906 | 0.923 |
| RTX 5060 Ti | 25,920 | 22.21 ms | (64, 128, 32, 8, 3) | 0.906 | 0.924 |

- **Larger cells want 128 columns per tile** on all three GPUs: 9–15 % faster. **The small cell wants smaller
  tiles** on the Ada and the A4000, and nothing on the 5060 Ti.
- **The kernel stays far from `segment_gemm`.** The best tile lifts it on the Ada from 7.8–7.9 to 9.0 TFLOP/s, 34 % of
  peak, not 50 %. So most of the gap is not the tile: the epilogue's gate and gated-row loads, the m-sum, and the
  atom-stacked rows that pad D = 3, 5, 7 to 4, 8, 8 (25 %, 37.5 % and 12.5 % of the rows are masked).
- **Step-level ceiling on the Ada:** 205 ms × 0.10 ≈ 21 ms of 1,341 ms (1.6 %) at 103,680 atoms; 42.7 ms × 0.12 ≈ 5 ms
  of 206 ms (2.4 %) at 21,600. Both are below the 5 % GO bar, so no rule and no A/B were built.

## What would reach more

The padding is the structural cost. Stacking D rows of an atom into a power of two wastes 12.5–37.5 % of the
`tl.dot` rows. A row-stacked layout without padding needs per-row atom indices for the gate and the m-sum, which is
what OPT10-T01 avoided to keep the m-sum in registers. That is a kernel redesign, not a tile choice, and it is not
proposed here.

## Records

- `results/step-roofline-20261101/`: `op_profile.py`, `roofline.py` and the two profile records.
- `results/gated-tile-20261101/`: `tile_screen.py` and each GPU's screen record.
