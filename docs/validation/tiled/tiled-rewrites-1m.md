# Tiled call at one million atoms: whole-system rewrites in the tile bodies (8 November 2026)

Verdict: four switches bring the tiled call with the state on the device from 8.83 to 6.84 µs/atom at 975,560 SrTiO3
atoms on the RTX 5060 Ti (same-process A/B 0.795 / 0.787). That is 1.43× the whole-system call per atom, down from
1.83×. The wavefront keeps three of them; block rows measured slower there and stay off.

Numbers, method, records and job scripts: [`results/tiled-speed-20261108`](../../../results/tiled-speed-20261108/README.md).

## What the tiles lacked

Before (main `d5de5bf7`), both paths applied the same rewrite records. The tiled bodies still did work the whole
system does not:

- The w′ convolution's BLOCK_MESSAGE was densified to `[R + H, width]`, sliced to the receivers, and its gradient
  padded back to R + H. The whole system multiplies `linear_1` on the block layout.
- The strain gradient went through a strain leaf whose backward ran as a split-K GEMM (`sgemm_largek`, about
  0.3 µs/atom per call). The whole-system plan takes it from the edge vectors (`single_graph_strain`).

Outside the bodies, two more costs:

- Each tile's graph was built four times per call (the slot-key pass and three sweeps): copy, unpack,
  `torch.unique`.
- With the state on the device, the next tile's graph was built only after the previous body had drained the GPU:
  about 1.2 s idle of 8.6 s.

## The switches

| Switch | Class | Device state b/a | Wavefront b/a | Default |
| --- | --- | --- | --- | --- |
| `tiled.TILE_PREFETCH_DEVICE` | exact | 0.939 / 0.941 | – | `TILED_SPEED_GPUS` |
| `generic.TILE_BLOCK_ROWS` | rounding | 0.918 / 0.921 | 1.030 / 1.060 | on, off under the wavefront |
| `tiled.TILE_VIRIAL` | rounding | 0.970 / 0.955 | 0.994 / 0.959 | `TILED_SPEED_GPUS` |
| `tiled.TILE_GRAPH_CACHE` | exact | 0.948 / 0.960 | off | `TILED_SPEED_GPUS` |

`TILED_SPEED_GPUS` is the RTX 5060 Ti family (decision 66: these keep memory alive or replace a kernel). The
force/stress gate with every switch on (tiled, compiled, R = 48) passes 12 of 12 checks.

## Left open

- Why block rows are slower under the wavefront (same tiles and slots, no recompiles).
- Sweep 3's recompute of interaction 0 (about 0.75 µs/atom): the three-sweep design.
- Halo rows (H ≈ 9,000 per 16,384 receivers), the float64 list filter (about 0.35 s per call), and the host list
  cache's fill for tiles the graph cache holds.
