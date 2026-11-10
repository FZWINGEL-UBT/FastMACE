# Tiled call at one million atoms: whole-system rewrites in the tile bodies (8-9 November 2026)

Verdict: eight switches bring the tiled call with the state on the device from 8.83 to 6.42 µs/atom at 975,560
SrTiO3 atoms on the RTX 5060 Ti (same-process A/B, all off against the defaults, 0.729 / 0.731). That is 1.34× the
whole-system call per atom, down from 1.83×. The wavefront went from 12.60 to 7.74 µs/atom (1.61× the whole-system
call, was 2.61×): two more exact switches (the graph cache under the wavefront, the host state's writes on a worker
thread) made it GPU-bound, and then all eight switches are faster there too (old wavefront defaults against the new:
0.766 / 0.774).

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
| `generic.TILE_BLOCK_ROWS` | rounding | 0.918 / 0.921 | 0.931 / 0.937 (1.030 / 1.060 host-bound) | on |
| `tiled.TILE_VIRIAL` | rounding | 0.970 / 0.955 | 0.994 / 0.959 | `TILED_SPEED_GPUS` |
| `tiled.TILE_GRAPH_CACHE` | exact | 0.948 / 0.960 | see `TILE_GRAPH_CACHE_WAVE` | `TILED_SPEED_GPUS` |
| `tiled.TILE_FILTER32` | exact | 0.987 / 0.993 | 0.981 / 1.012 | `TILED_SPEED_GPUS` |
| `tiled.TILE_EPILOGUE` | rounding | 0.980 / 0.981 | 0.985 / 0.983 (1.012 / 1.009 host-bound) | `TILED_SPEED_GPUS` |
| `tiled.TILE_FILL_SKIP` | exact | 0.961 / 0.966 | – | on |
| `tiled.TILE_PINNED_SUPERSET` | exact | 0.986 / 0.990 | 0.989 / 0.990 (1.012 / 1.028 host-bound) | `TILED_SPEED_GPUS` |
| `tiled.TILE_GRAPH_CACHE_WAVE` | exact | – | 0.824 / 0.817 | `TILED_SPEED_GPUS` |
| `tiled.TILE_HOST_WORKER` | exact | host state 0.969 / 0.966 | 0.975 / 0.982 | `TILED_SPEED_GPUS` |

`TILED_SPEED_GPUS` is the RTX 5060 Ti family (decision 66: these keep memory alive or replace a kernel). The
force/stress gate with every switch on (tiled, compiled, R = 48) passes 12 of 12 checks (f36adabd, and b5a59f7d with
the epilogues). The last four came from a region profile of one steady call: the model's kernels without sweep 3's
recompute matched the whole-system call, and the gap was the recompute plus about 870 ns/atom of eager glue (graph
builds, a float64 list filter, masks, copies). The glue is now about 520 ns/atom. Rejected: keeping a tile's index at
its slot's halo start (0.9999 / 1.0001) and keeping interaction 0's activations for sweep 3 (8 kB per atom).

## Left open

- Why block rows are slower under the wavefront (same tiles and slots, no recompiles).
- Sweep 3's recompute of interaction 0 (about 0.8 µs/atom): the three-sweep design.
- Halo rows (H ≈ 9,000 per 16,384 receivers in interaction 0) and the forces' scatter (115 ns/atom).
- On the wavefront, the host state's CPU work; its A/Bs spread by ±8 % between pairs.

## The write copies (10 November 2026)

`tiled.TILE_COPY_STREAM` (exact, `TILED_SPEED_GPUS`) moves the host state's device-to-host write copies off the
compute stream (device clones, a side stream copies them): wavefront 0.962 / 0.964 (7.80 → 7.51 µs/atom), host state
0.960 / 0.959. The wavefront call is GPU-bound (wall 7.24 s, GPU busy 7.15 s): 1.55× the whole-system call per atom.
Records: [`results/tiled-speed-20261110`](../../../results/tiled-speed-20261110/README.md).
