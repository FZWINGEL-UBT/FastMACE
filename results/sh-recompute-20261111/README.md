# Idea 13, step 1: offline census of the w′ epilogue with the edge force contracted (11 November 2026)

Idea 13 is in `plans/ideas-20261111-tiled-round2.md`. The conductor's stop rule was more than about 3 % of the force call. Nothing ran on a GPU.

**Source:** the production A100 w′ backward rendered on 11 November (`results/a100-colab-ideas-20261111/batch2/ghoist-prod/sources/wprime-real.cu`, OMAT-0 medium). It uses BUTTERFLY, STAGE_DOUT, STAGE_EDGES and STAGE_SH; EDGE_DIM is 16 and CHANNEL_DIM 128.

**Method:**
- `tools/bench/sh_recompute_census.py` replaces the `dsh[e, c]` store with `dL/dv[e]` (12 B per edge instead of 64), using the formulas of `fastmace/kernels/triton/edge_sh.py`'s backward, in two variants:
  - `gather`: 16 shuffles bring the components to every lane, and lane 0 contracts them;
  - `split`: each component's lane adds its share from a shared-memory table of `dY_c/du` monomial coefficients, then a 3-value xor reduction.
- NVRTC builds for sm_80, sm_89 and sm_120 (WSL, CPU only).
- `tools/bench/sh_recompute_epilogue.py` counts the per-edge epilogue: from the last dx atomic to the next BAR or EXIT.

| Arch | Variant | Registers | SASS total | Epilogue per edge | Added per edge |
|---|---|---:|---:|---:|---:|
| sm_80 | base | 79 | 1168 | 147 | – |
| sm_80 | gather | 80 | 1488 | 279 | +132 |
| sm_80 | split | 96 | 1488 | 261 | +114 |
| sm_89 | base | 80 | 1136 | 163 | – |
| sm_89 | gather | 80 | 1424 | 313 | +150 |
| sm_89 | split | 80 | 1440 | 295 | +132 |
| sm_120 | base | 96 | 1448 | 122 | – |
| sm_120 | gather | 96 | 1936 | 258 | +136 |
| sm_120 | split | 96 | 2000 | 260 | +138 |

**Per-edge cost:**
- **Today's kernel:** the channel loop (`channel += LOGICAL_GROUP_SIZE`, 128 / 32 = 4 trips) has a body of about 500 instructions. With the epilogue and the edge set-up, that is about 2,250 dynamic instructions per edge.
- **The contraction:** adds 115–150 instructions per edge, about 5–7 % of the w′ kernel's instructions.
- **Share of the force call:** w′ is 25.6 % of A100 GPU time (`results/tiled-speed-20261111/nsys-a100`). The cost is therefore at most about 1.5 % of the force call, if the epilogue were not overlapped at all. That is below 3 %, so step 1 passes.

**Choice:** `gather`.
- `split` saves only 18 instructions on sm_80 and costs 96 registers there (from 79). On sm_120 it is no cheaper.
- With REDUX (the 5060 Ti's epilogue) every lane already holds all 16 sums, so the 16 gather shuffles fall away there.

**Caveats:**
- The sm_89 and sm_120 builds use the A100's rendered source, not those GPUs' own switch sets. The 5060 Ti uses REDUX and an 80-register cap.
- Instruction counts are an upper bound on time.
- Step 2 removes the 64 B/edge `dsh` store but keeps `edge_sh3_backward` for interaction 0's `grad_y`. The kernel saving (36 µs on the A100) comes only with step 3.
