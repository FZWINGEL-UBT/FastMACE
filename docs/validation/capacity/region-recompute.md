# The combined region's recompute in the backward (31 October 2026)

**Verdict: STOP.** The region's recompute is necessary work done efficiently. Less than 2 % of the step is
avoidable, against the GO bar of 10 %.

Question (conductor): the combined node region (`recompute_node_region`) costs about 1.2 times the separate
segments' force call, and rule 12 plans it for every NbCl5 cell above about 100,000 atoms on 20 GB. How much of that
is avoidable recompute?

## Setup

`results/region-recompute-20261101/backward_profile.py`:
- default float32 plan, hybrid, every memory-rescue record set explicitly;
- the force call's forward (the compiled model) and its gradient step (MACE's `get_outputs`, run apart: F2) are
  timed with CUDA events, median of 5 calls;
- one gradient step and one whole call run under the CUDA profiler, kernels summed by name;
- the captured graphs are released first, so the calls are eager.

leeloo, RTX 4000 Ada, NbCl5, FastMACE 8b6ae7ff.

## Forward and gradient step by rescue stage

| Stage | Atoms | Forward | Gradient step | Call |
|---|---:|---:|---:|---:|
| conv_weights, first_table | 43,200 | 178.5 ms | 259.9 ms | 438.4 ms |
| + separate node segments | 43,200 | 175.6 ms | 311.3 ms | 486.9 ms |
| + combined region, 4 chunks (gate_epilogue on) | 43,200 | 177.0 ms | 386.0 ms | 563.0 ms |
| + combined region, 4 chunks (gate_epilogue on) | 103,680 | 423.8 ms | 921.8 ms | 1,345.6 ms |

The region leaves the forward unchanged. It adds 74.7 ms to the gradient step over the separate segments, 13 % of
its call, and 126 ms over the first-table stage.

## Where the region's extra time goes

The gradient step's kernels by name, region minus separate segments, at 43,200 atoms:

| Kernel | Separate segments | Region | Change |
|---|---|---|---:|
| `gemm` (`segment_gemm`, the reshape's `linear_2` per segment) | 3.2 ms, 2 launches | 58.7 ms, 40 launches | +55.5 ms |
| `ampere_sgemm_128x64_nn` (cuBLAS: the product's linears and the chain's `linear_1`) | 59.0 ms, 20 launches | 69.0 ms, 71 launches | +10.0 ms |
| The chain's elementwise kernels (`joined_products`, the gate), the divide | | | about +15 ms |
| `segmented_polynomial` backward | | | about the same |

- **The reshape's recompute is the region's memory saving.** With the separate segments, the reshape sits between the
  two checkpoints. Its output, the contraction's input, is kept, so the backward does not recompute it. In the region
  it is not kept, and the backward rebuilds it.
- **The rebuild is efficient.** The forward's `segment_gemm` takes 52.6–54.4 ms (8 to 32 launches); the region's
  recompute, chunked into 4 atom chunks, takes 58.7 ms. That is 1.08 times the forward, about 4 ms of chunking
  overhead.
- **The rest:**
  - the cuBLAS increase (10 ms) is the product's and the chain's linears rebuilt per chunk;
  - the elementwise kernels rebuild the gated rows;
  - all of them are inputs of the backward's own products.
  At most the product's last linear, whose output the backward does not read, could be skipped: under 10 ms, 1.8 %
  of the call.

## Speed at the limits

Default plans (refit ladder, leeloo, 20 GB): NbCl5 72,576 atoms 853 ms per step (no region); 85,536 atoms 1,016 ms
(no region, gate_epilogue off); 103,680 atoms 1,422 ms and 121,824 atoms 1,680 ms (region). On the 16 GB RTX 5060 Ti
with the region: NbCl5 114,048 atoms 1,482 ms, NaCl 115,200 atoms 1,506 ms. Between 85,536 and 103,680 atoms the step
grows by 1.40 times for 1.21 times the atoms: the region's 1.13–1.15 times the separate segments' call.

## Records

`results/region-recompute-20261101/`: `backward_profile.py` and its JSON records, with each stage's gradient-step
and forward kernels.
