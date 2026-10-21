
## NaCl (216 atoms, 8192 padded edges)

| Quantity | Value | Source |
| --- | ---: | --- |
| Wall time per MD step | 4.264 ms | un-profiled window (1000 steps) |
| GPU span per step (stage events) | 4.288 ms | CUDA events around stages |
| GPU idle between stages | -0.168 ms | CUDA events |
| B graph replay (forces + 2nd half kick) | 4.043 ms | CUDA events, un-profiled |
| Force-graph replay span / kernel time | 3.462 / 3.394 ms | profiler, 10 replays |
| Kernels per replay | 256 | profiler |
| Gaps inside a replay | 66 µs (255 gaps) | profiler |
| Eager force call kernel sum | 3.293 ms (248 kernels) | profiler, eager calls |

Stages (ms per step):

| Stage | ms | % of GPU span |
| --- | ---: | ---: |
| B: forces + second half kick | 4.043 | 94.3 |
| ACP: Langevin half kick, positions, neighbour check, edge compaction | 0.211 | 4.9 |
| Neighbour-list rebuild (GPU) | 0.180 | 4.2 |
| Noise upload (every step a slice; a new batch every noise_batch) | 0.022 | 0.5 |
| GPU idle between stages | -0.168 | -3.9 |

Force call by part (profiler, eager calls: 3.293 ms of kernels; forward/backward split from ncu's per-kernel durations, boundary at kernel 138 of 248):

| Part | ms | % of force kernels | kernels | forward / backward |
| --- | ---: | ---: | ---: | --- |
| Atom-level linears (cuBLAS GEMMs and FastMACE Triton GEMMs) | 1.367 | 41.5 | 85 | – |
| Radial networks and embeddings | 0.748 | 22.7 | 31 | – |
| Convolution (interaction 1 reassociated sums, interaction 2 streaming OEq) | 0.671 | 20.4 | 7 | – |
| Atom-level elementwise (gates, products, residuals, readouts, species tables) | 0.309 | 9.4 | 96 | – |
| Symmetric contraction (cuEq) | 0.112 | 3.4 | 6 | – |
| Neighbour list and edge prep (edge geometry, radial basis, gathers, scatters) | 0.059 | 1.8 | 11 | – |
| Copies and fills | 0.028 | 0.8 | 12 | – |

## NbCl5 (1944 atoms, 56320 padded edges)

| Quantity | Value | Source |
| --- | ---: | --- |
| Wall time per MD step | 33.126 ms | un-profiled window (1000 steps) |
| GPU span per step (stage events) | 33.736 ms | CUDA events around stages |
| GPU idle between stages | -0.467 ms | CUDA events |
| B graph replay (forces + 2nd half kick) | 33.293 ms | CUDA events, un-profiled |
| Force-graph replay span / kernel time | 31.475 / 31.426 ms | profiler, 10 replays |
| Kernels per replay | 237 | profiler |
| Gaps inside a replay | 65 µs (236 gaps) | profiler |
| Eager force call kernel sum | 28.658 ms (229 kernels) | profiler, eager calls |

Stages (ms per step):

| Stage | ms | % of GPU span |
| --- | ---: | ---: |
| B: forces + second half kick | 33.327 | 98.8 |
| Neighbour-list rebuild (GPU) | 0.480 | 1.4 |
| ACP: Langevin half kick, positions, neighbour check, edge compaction | 0.321 | 1.0 |
| Noise upload (every step a slice; a new batch every noise_batch) | 0.075 | 0.2 |
| GPU idle between stages | -0.467 | -1.4 |

Force call by part (profiler, eager calls: 28.658 ms of kernels; forward/backward split from ncu's per-kernel durations, boundary at kernel 128 of 229):

| Part | ms | % of force kernels | kernels | forward / backward |
| --- | ---: | ---: | ---: | --- |
| Atom-level linears (cuBLAS GEMMs and FastMACE Triton GEMMs) | 10.485 | 36.6 | 72 | – |
| Radial networks and embeddings | 5.773 | 20.1 | 31 | – |
| Convolution (interaction 1 reassociated sums, interaction 2 streaming OEq) | 5.352 | 18.7 | 7 | – |
| Atom-level elementwise (gates, products, residuals, readouts, species tables) | 4.260 | 14.9 | 86 | – |
| Symmetric contraction (cuEq) | 1.840 | 6.4 | 4 | – |
| Neighbour list and edge prep (edge geometry, radial basis, gathers, scatters) | 0.581 | 2.0 | 11 | – |
| Copies and fills | 0.368 | 1.3 | 18 | – |

## NaNbO3 (1080 atoms, 82688 padded edges)

| Quantity | Value | Source |
| --- | ---: | --- |
| Wall time per MD step | 29.589 ms | un-profiled window (1000 steps) |
| GPU span per step (stage events) | 29.963 ms | CUDA events around stages |
| GPU idle between stages | -0.483 ms | CUDA events |
| B graph replay (forces + 2nd half kick) | 29.601 ms | CUDA events, un-profiled |
| Force-graph replay span / kernel time | 28.197 / 28.129 ms | profiler, 10 replays |
| Kernels per replay | 249 | profiler |
| Gaps inside a replay | 80 µs (248 gaps) | profiler |
| Eager force call kernel sum | 25.604 ms (241 kernels) | profiler, eager calls |

Stages (ms per step):

| Stage | ms | % of GPU span |
| --- | ---: | ---: |
| B: forces + second half kick | 29.601 | 98.8 |
| Neighbour-list rebuild (GPU) | 0.496 | 1.7 |
| ACP: Langevin half kick, positions, neighbour check, edge compaction | 0.331 | 1.1 |
| Noise upload (every step a slice; a new batch every noise_batch) | 0.018 | 0.1 |
| GPU idle between stages | -0.482 | -1.6 |

Force call by part (profiler, eager calls: 25.604 ms of kernels; forward/backward split from ncu's per-kernel durations, boundary at kernel 132 of 241):

| Part | ms | % of force kernels | kernels | forward / backward |
| --- | ---: | ---: | ---: | --- |
| Radial networks and embeddings | 8.404 | 32.8 | 31 | – |
| Convolution (interaction 1 reassociated sums, interaction 2 streaming OEq) | 6.918 | 27.0 | 7 | – |
| Atom-level linears (cuBLAS GEMMs and FastMACE Triton GEMMs) | 5.750 | 22.5 | 76 | – |
| Atom-level elementwise (gates, products, residuals, readouts, species tables) | 2.107 | 8.2 | 88 | – |
| Symmetric contraction (cuEq) | 1.216 | 4.8 | 6 | – |
| Neighbour list and edge prep (edge geometry, radial basis, gathers, scatters) | 0.833 | 3.3 | 11 | – |
| Copies and fills | 0.377 | 1.5 | 22 | – |
