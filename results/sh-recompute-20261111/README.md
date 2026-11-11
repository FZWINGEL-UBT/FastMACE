# Idea 13: offline census (step 1) and EDGE_FORCE on the A100 (step 2), 11 November 2026

## Step 1: offline census of the w′ epilogue with the edge force contracted
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


## Step 2: EDGE_FORCE, the w′ backward writes dL/dv instead of dsh (A100, 11 November 2026)

**Verdict:** REJECTED. The switch stays opt-in (`wprime_backward.EDGE_FORCE`, `EDGE_FORCE_GPUS` empty), and step 3 is not started. The conductor's rule asked for a lower 36³ peak and pool bytes per edge than the 14.42 GiB / 720 B baseline. With EDGE_FORCE on, both are higher, and the step is 2–3 % slower.

**What it does:**
- The SPLINE_W w′ backward kernel computes the edge-vector gradient itself.
- Its epilogue contracts the 16 edge-feature sums with edge_sh3's Jacobian and writes dL/dv [E, 3], 12 B/edge, instead of dsh [E, 16], 64 B/edge.
- It uses the gather variant from step 1.
- The vectors come from the model's EdgeSH by weak reference (`rewrites.generic.modules.edge_vectors`).

**Code:** `fastmace/kernels/oeq_streaming/wprime_backward.py` (template `edge_force`, op `streaming_conv_wprime_backward_force`), `fastmace/rewrites/generic/modules.py` and `fastmace/rewrites/mh1/modules.py`.

**Tests:** `tests/cpu/test_wprime_backward.py` (source, NVRTC sm_80/89/120, hand-over) and `tests/gpu/test_edge_force_cuda.py`.

**Jobs:** `tools/jobs/alex_edge_force*.sh`. The logs and profiles are in `step2/`.

### Results (A100-SXM4-80GB, OMAT-0 medium float32)

| Source | GPU test | Gate (SrTiO3 F / S; NaCl F / S) | A/B b/a SrTiO3 10³ (ab, ba) | A/B b/a NbCl5 (ab, ba) | 36³ steady reserved off → on | pool B/edge off → on |
|---|---|---|---|---|---|---|
| b8d9dc04 (Alex 4983296, 4983308) | 3/3 | 0.94 / 0.66; 0.86 / 0.95 | capture failed | capture failed | 14.42 → out of memory | – |
| 44e30934 (4984748, 4984747) | 3/3 | 0.97 / 0.65; 0.86 / 0.89 | 1.0330, 1.0289 | 1.0248, 1.0239 | 14.42 → 16.12 GiB | 720 → 828 |
| 8dd5f605 (4985845, 4985844) | 3/3 | 0.90 / 0.68; 0.85 / 0.82 | 1.0302, 1.0313 | 1.0242, 1.0212 | 14.42 → 16.12 GiB | – |

- **GPU test:** forces with EDGE_FORCE on match it off within 1e-4 of max |F| for the plain shuffle-down, BUTTERFLY and REDUX epilogues.
- **Gate:** passes on every source.
- **Time per step at 36³:** 320.0 ms off, 329.4 ms on (8dd5f605).
- **Capacity scans (from 233,280 atoms, 40-minute budget; `step2/edgeforce-4985845/scan-*`):** both sides pass 49³ (588,245 atoms), and on both sides the 57³ and 53³ points ran into the scan's per-point timeout, so both limits lie between 49³ and 53³ and no limit change can be reported.

  | Steady reserved | Off | On | Change |
  |---|---|---|---|
  | 36³ | 14.42 GiB | 16.12 GiB | +12 % |
  | 42³ | 22.49 GiB | 25.20 GiB | +12 % |
  | 49³ | 35.31 GiB | 39.69 GiB | +12 % |

  At 49³ the step is 881 ms off against 1,057 ms on.

### Why memory goes up

Per-tensor profiles of SrTiO3 24³ (69,120 atoms, E = 5,114,880 edges; Alex 4985601 and 4985843; `step2/memprofile-*`):

- **Removed:** the w′ backward's dsh [E, 16] (312 MiB) is gone from the peak, as intended.
- **Added:** dL/dv [E, 3] (58.5 MiB).
- **Also added, on both sources:** two allocations with no Python stack, [E, 3] and [E, 16]: 76.0 B/edge, 371 MiB.
  - They are made at the start of `torch.autograd.grad` (`mace.modules.utils.compute_forces`), before the first compiled backward runs.
  - So they are zero gradients that AOT autograd fills in for the compiled forward region's `vectors` and `edge_attrs` outputs. My Function's own None gradient is not the source.
  - Passing sh detached to the convolution (8dd5f605) therefore changed nothing.
- **Net:** the peak allocation is +155 MB and the steady reserved +0.43 GiB at 24³; at 36³ the steady reserved is +1.7 GiB.

**The ceiling, even if the zeros went away:**
- Interaction 0's first-layer message (`FirstInteractionReassociation`, the edge outer sum) still needs the gradient of `edge_attrs`, so edge_sh3's backward and one dsh [E, 16] stay.
- Step 2 can save at most 64 − 12 = 52 B/edge, about 0.8 GiB (5.5 %) at 36³.
- The step costs 2.1–3.3 % at the normal sizes, against the conductor's 2 % budget for a plan-only switch.

### Bugs found on the way

- **b8d9dc04:** EdgeSH held its last vectors by strong reference. That kept the warmup forward's autograd graph alive:
  - into the capture: "AccumulateGrad has a stale reference to the default stream", in both A/B orders;
  - at the 36³ point: out of memory, then the job's time limit.
- **Fix (44e30934):** weak references, plus a CPU test that the tensors are freed.

### Reopen only with

- the region-boundary zero gradients removed: no `vectors` or `edge_attrs` output of a compiled region without a gradient;
- interaction 0's edge outer sum also contracting into dL/dv (step 3's scope), so that edge_sh3's backward and every dsh go away;
- a step cost of at most 2 % in both orders.
