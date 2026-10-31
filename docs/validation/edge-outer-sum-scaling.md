# edge_outer_sum backward: why it grows faster than the edges (31 October 2026)

**Verdict: no fix proposed.**

- The superlinear kernel is Inductor's pointwise consumer of the custom backward's output, not the custom Triton
  kernel.
- It moves about 4.7 times the bytes between the two cell sizes. At 1,944 atoms its writes stay in the 32 MiB L2;
  at 9,216 atoms they go to DRAM, where it runs at 85 % of memory throughput.
- That is the L2-to-DRAM step, not contention, occupancy or a size-dependent configuration.
- Removing the kernel altogether would save at most 0.47 % of the step at 9,216 atoms, under the 1 % bar.
- The custom backward scales linearly. Its occupancy limit (140 registers) leaves at most about 0.5 % of the step.

Conductor's follow-up to the side note in [l2-gap-audit.md](l2-gap-audit.md) ("8.6× the time for 4.6× the edges").

## Measurement

Setup:
- Nsight Compute over one eager force call of the compiled default float32 plan, filtered to the kernels named
  `*edge_outer_sum*` and `kernel`;
- `--cache-control none --clock-control none`;
- NbCl5 at 1,944 atoms (3×3×3) and 9,216 atoms (4×4×8);
- local RTX 5060 Ti (sm_120, 32 MiB L2), `mace_cueq_oeq`, FastMACE 16be8208 (main f60d7e4c plus this launcher):
  `results/edge-outer-sum-20261031/run.sh`.

| Kernel | Atoms | µs | DRAM read / write MB | L2 MB | L2 hit % | Memory throughput | Warps active | Reductions (L2 sectors) |
|---|---|---|---|---|---|---|---|---|
| Inductor `add_edge_outer_sum_backward_slice_backward_transpose` | 1,944 | 47.3 | 19.4 / 0.4 | 34.6 | 44.7 | 94.8 % | 73 % | 0 |
| | 9,216 | 392.3 | 82.5 / 65.3 | 162.2 | 8.2 | 85.4 % | 71 % | 0 |
| Custom `edge_outer_sum_backward` (128 threads, 140 registers) | 1,944 | 316.9 | 36.2 / 7.3 | 89.2 | 44.9 | 63.2 % | 24 % | 0 |
| | 9,216 | 1,543.7 | 180.7 / 85.5 | 427.6 | 44.0 | 61.3 % | 25 % | 0 |
| Custom segmented forward (32 threads, atomics) | 1,944 | 103.2 | 34.0 / 4.8 | 60.1 | 37.0 | 85.3 % | 40 % | 1.03 M |
| | 9,216 | 650.3 | 168.8 / 79.3 | 256.6 | 34.0 | 86.5 % | 38 % | 5.01 M |

**Ratios from 1,944 to 9,216 atoms** (edges 4.6×):
- **Pointwise consumer:** 8.3× the time for 4.7× the L2 bytes.
  - At 1,944 atoms its about 35 MB of operands are mostly L2 hits. Its 30 MB of writes never reach DRAM (0.4 MB
    written), so it runs at 731 GB/s of L2 traffic.
  - At 9,216 atoms the operands are 162 MB: `[7N, 130]`, `[5N, 130]`, `[3N, 130]`, `[N, 130]` and `[2N, 1,040]`
    floats, the node-level gradient being sliced, transposed and added. The hit rate drops to 8 %, and 65 MB are
    written to DRAM, at 413 GB/s. That is 85 % of the card's 448 GB/s.
- **Custom backward:** 4.9×, which is linear. Its occupancy is limited by registers (24 % of warps active).
- **Segmented forward:** 6.3×, slightly above linear. Its atomic reductions grow 4.9×, so there is no contention
  growth.

## Ceilings at 9,216 atoms (step 84.0 ms)

- **Fusing away the pointwise consumer,** for example writing the backward's output straight in the transposed and
  sliced layout: at most its whole graph time, 407.5 µs per step (M2 trace), so **0.49 %**.
- **Raising the custom backward's occupancy:** at the forward's 86 % memory throughput instead of 61 %, it would
  take about 0.71 of its time. With its share of the graph's `kernel` time, that is about 0.4 ms of ncu time per
  call, **under 0.5 %** of the step.
- **Both together:** under 1 %. Below the bar, so nothing is proposed.

Records: `results/edge-outer-sum-20261031/` holds `run.sh` and the CSV exports `NbCl5-1944.csv` and
`NbCl5-9216.csv`.
