# Research prompt: new ways to make FastMACE faster on the NVIDIA H200

*Written 5 November 2026 for an external research model (ChatGPT). FastMACE branch `fastmace/round5` at `f8facf08`.
The documents to attach are listed at the end.*

## Your task

Think creatively, and well outside the obvious, about how to make one MD step of FastMACE faster on an NVIDIA H200.
We have already worked through the standard list: tensor-core GEMMs, kernel fusion, CUDA graphs, staging and
vector loads, wgmma, and per-GPU switches. The ideas that remain are not obvious, so we want ones that change *what*
is computed or *how the work is laid out*, not just a faster version of today's kernel. Unconventional ideas are
welcome: algebraic reformulations, a different data layout, symmetry exploitation, reuse across MD steps, and
numerical tricks that keep FP32-class accuracy. Each one must survive the constraints below.

Give 8–15 ideas, ranked by expected whole-step gain on the H200. For each:

1. **Mechanism.** What changes in the computation, and why it should be faster on Hopper (sm_90, 132 SMs, 227 KB
   opt-in shared memory per block, 50 MB L2, 4.8 TB/s HBM3e, FP32 ≈ 67 TFLOPS, TF32 tensor cores, TMA, thread-block
   clusters and distributed shared memory).
2. **Target.** Which kernel in the profile below it attacks, and an Amdahl ceiling in µs per force call.
3. **Evidence.** Papers, vendor docs, or open-source kernels (cuEquivariance, OpenEquivariance, FlashTP/SevenNet,
   NequIP/Allegro, e3nn-jax, Symmetrix, NVIDIA ALCHEMI, CUTLASS/CuTe examples, FlashAttention-3 techniques, …) that
   show the technique working, with links and the numbers they report. Say plainly when an idea has no precedent.
4. **Exactness class.** Bitwise; rounding class (reassociation only); or approximate. If approximate, explain how it
   passes the accuracy gate.
5. **Falsifier.** The cheapest experiment that would kill the idea before anyone builds it: a lab variant of the
   existing kernel, a timing bound, or a microbenchmark. Give a stop rule.
6. **Overlap check.** Confirm it is not in the "already tried" lists below. If it resembles a rejected idea, say
   what is different this time.

Mark every number as measured (by us, from the data here), cited (from a source you name) or estimated (by you).

## The workload

- **Model:** MACE-OMAT-0 medium (2 interactions, 128 channels, hidden irreps up to l = 3, correlation order 3,
  cutoff 6 Å), float32, hybrid backend:
  - cuEquivariance for the symmetric contraction;
  - our own NVRTC/CUDA kernels for the convolution;
  - Triton and CUTLASS for the rest.
- **System:** Benchmark_loup, SrTiO3 10×10×10 (5,000 atoms, 373,760 edges), Langevin NVT at 1 fs, neighbour skin
  0.3 Å.
- **Execution:**
  - one force call per step (energy, then forces by autograd through a compiled, rewritten MACE);
  - the whole step is captured as CUDA graphs;
  - one structure on one GPU at a time.
- **Stack:** torch 2.14.1+cu130, CUDA 13, driver 610.57, Python 3.10, cuEquivariance 0.12, OpenEquivariance 0.7,
  MACE 0.3.16.
- **Speed today on the H200:** ≈ 5.67 ms per step (≈ 1.13 µs per atom-step), against ≈ 10.9 ms on an A100.
  Production systems range from 100 atoms to about a million; in tiled mode a single GPU has run 1M atoms.

## Where the time goes (H200, per force call)

Kernel times from the H200 profile at `27d27f21`, with the forward at `02a81e13`:

| Kernel | µs | Notes |
| --- | ---: | --- |
| `fused_folded_forward` (convolution forward, interactions 1–2) | ≈ 1,780 | see below |
| `oeq_projected_backward_wprime` (w′, the convolution backward) | ≈ 1,600 | one warp per edge, 93,440 blocks × 128 threads, 96 registers, no shared memory, atomics into dx |
| CUTLASS 3xTF32 GEMMs (`fm_tf32x3_*`, linears) | ≈ 866 in 37 launches | 64×64×16 tiles, sm_80 cubin running on sm_90 |
| `tabled_outer_sum_backward` (interaction 1) | ≈ 240 | Triton |
| radial tables, forward (interactions 1 / 2) | ≈ 115 / 115 | Triton, piecewise-polynomial tables of 896 / 624 pieces, 66 columns |
| `edge_outer_sum` forward (interaction 1) | ≈ 99 | |
| cuEq symmetric contraction ×4 | ≈ 166 | |
| Inductor glue around the GEMMs, `cat_index_mul_sum`, small kernels | ≈ 270 | |
| gaps inside the graph replay | ≈ 74 over 166 launches | |

### The fused forward

The fused forward is the convolution's forward. Receivers are sorted, and each warp takes 8 consecutive edges per
batch and a block of 32 channels. Per edge and channel it computes:

- **The weights:** the per-edge weight w = a·W, with a = 64 folded radial inputs and W the per-path weight slice in
  shared memory. This uses mma.sync with 3xTF32.
- **The tensor product:** a per-lane uvu product over 7–10 paths (l ≤ 3), summing into per-receiver accumulators
  that are flushed when the receiver changes.

Its profile: 168 registers, ≈ 151 KB shared memory, 12 warps per block, grid [33, 4, 2].

Lab bounds (the kernel's time with one part removed, as a ratio to the full kernel; measured):

| Removed part | H200 | A100 |
| --- | ---: | ---: |
| the gathers of the sender features x | 0.834 | 0.812 |
| the whole tensor product | 0.619 | 0.613 |
| only the CG-coefficient multiplies | 0.910 | 0.879 (0.945 on a PCIe A100) |
| the float4 staging of a and sh | 0.955 | 0.971 |
| the a·W product (tensor-core part), earlier base | 0.527 | – |
| an S tile (CG·sh precomputed per edge) | pending | 0.951 (PCIe A100) |

### The w′ backward

w′ is the convolution backward. It computes three things:
- dx, by atomics into the senders;
- dsh;
- dr, through a tabulated derivative of the radial weights (w′(r) value tables, Horner evaluation, butterfly and
  factored reductions).

Bounds on the H200: one slab only, 0.925; no atomics, 1.089 (slower, because of register pressure). Earlier Ada
bounds: no table 0.60, no dout 0.67, no x 0.88.

## Hard constraints (an idea that breaks one is out)

- **Accuracy gate.** A float32 path is accepted only if its force and stress errors against float64 are at most 2×
  those of the stock float32 calculator, on NaCl, NbCl5 and SrTiO3. NVE drift must also be unchanged. Plain TF32
  fails by about 3,500×. 3xTF32 passes, and so does FP32 SIMT.
- **Precision.** FP32 with TF32 off (or 3xTF32) is the production precision. BF16/FP16/FP8/INT8 storage or
  arithmetic is allowed only if the gate passes. Every such attempt so far has failed it or been slower.
- **Generality.** It must work for any crystal, composition and size without per-structure tuning (no autotuning
  tables, no shape-keyed caches). Per-GPU defaults are allowed, keyed by exact device name, but only where
  whole-step A/Bs in both build orders were faster.
- **Measurement.** A gain counts only as a same-process, alternating-block, whole-MD-step A/B under CUDA graphs, in
  both build orders. Effects of about 0.5 % and up are resolvable.
- **Single GPU.** No multi-GPU or domain decomposition for now.
- **Physics.** No larger timesteps, no change to the model (same checkpoint, same function), no neighbour-list
  approximations beyond the skin.
- **Compile time.** Setup is allowed to cost seconds, not minutes (no Inductor max-autotune).

## Already tried on the H200 (5 November 2026)

Each result is the MD-step b/a ratio in both build orders (b = with the change; below 1 is faster).

| Change | Result | Verdict |
| --- | --- | --- |
| The a·W product on wgmma (TF32x3, sm_90a), plain and software-pipelined | 1.024 / 1.025; pipelined 1.035 / 1.033 | rejected |
| A native sm_90 wgmma 3xTF32 GEMM for the linears | 1.19× slower than the sm_80 CUTLASS kernel, and less accurate | rejected |
| Term-pair interleaved w′ value table | 1.0008 / 1.0003 | rejected |
| Hoisting the piece scale out of the w′ slope | 1.0057 / 1.0017 | rejected |
| Degree-pure path parts in the fused forward (each part gathers only its own x rows) | 0.982 / 0.985 | adopted |
| Tabled outer-sum forward | 0.984 / 0.987 | adopted |
| Piece-sorted radial-table evaluation | sorted kernel 0.87×, but the sort costs more than the kernel takes | dead |
| TMA / 1-D bulk copies for the forward's staging | staging is only 3–4.5 % (bound) | dropped |
| PDL between our own adjacent kernels | ≤ 4.5 µs per group of 4 GEMMs (A100 bound) | dropped |
| One grouped launch per BlockLinear | 26–52 µs per group (A100 bound) | built, A/B pending |
| S tile (CG·sh once per edge) | 0.951 of the forward on a PCIe A100 | pending on the H200 |

Also excluded on the H200 by earlier screens:
- the A100 switches it lost on;
- grid/raster variants;
- BATCH = 4;
- cluster/DSMEM/TMA-multicast staging;
- producer/consumer warp specialisation with setmaxnreg (the microbenchmark lost 1.1–1.5×);
- INT8-slice and BF16x3 GEMMs;
- a receiver-owned w′ without atomics;
- graph-gap removal and megakernels;
- L2 chunk sizing;
- 16-bit weights;
- species/Morton atom order;
- skin 0.4 Å;
- low-rank linears (they are full rank);
- concurrent graph branches;
- Inductor PDL (wrong forces).

## Already adopted (part of the 5.67 ms)

The full list is in the attached `docs/optimisation-ledger.md`. The main items:

- **Convolution forward and model rewrites:**
  - receiver sorting;
  - species-factored, joint radial inputs;
  - folded cutoff and density-first;
  - the reassociated first interaction;
  - the streaming receiver-major convolution;
  - the fused folded forward (a·W on 3xTF32 mma.sync, W fragments, float4 staging, skip-zero-column);
  - radial-table evaluation of the radial MLP.
- **The w′ backward:**
  - its value tables;
  - Horner evaluation;
  - factored, butterfly and scalar-path reductions.
- **Linears and GEMMs:**
  - block, grouped and merged-residual linears;
  - CUTLASS 3xTF32 for every linear;
  - segment-row GEMMs without layout copies;
  - head-slice and graph-sum energy;
  - fused first-interaction backward.
- **Step machinery:**
  - GPU neighbour list;
  - pipelined steps;
  - asynchronous output;
  - one-graph strain.

## Directions we would especially like explored

1. **The convolution as a whole (forward ≈ 1.8 ms, backward ≈ 1.6 ms, together ≈ 60 % of the step).**
   - Is there a formulation that touches each edge fewer times? Examples:
     - a joint forward/backward per edge that saves what the backward needs;
     - an eSCN/SO(2)-style rotation to the edge frame that makes the l ≤ 3 tensor product sparse;
     - a Gaunt/FFT tensor product;
     - a channel-mixing order that turns the per-lane SIMT product into tensor-core work at FP32-class accuracy.
   - What would change the x-gather cost (17–19 % of the forward)? Examples: a receiver-tile/sender-tile blocking
     that reuses senders from shared memory or L2, or a sender-major second pass.
2. **Reuse across MD steps.** At 1 fs, positions change by about 10⁻³ Å per step and the neighbour list is rebuilt
   rarely. Can anything be cached or updated incrementally (radial tables, sh, per-edge weights) while keeping
   exactly the same function? Or reformulated so that exactness holds by construction?
3. **The backward's atomics and its 96-register, one-warp-per-edge shape.** Removing the atomics alone made it
   slower, because of register pressure. What layout would win on Hopper, given that it has more shared memory and
   L2?
4. **The 37 small 3xTF32 GEMM launches (≈ 0.87 ms).** Options beyond grouping: fusing them into neighbouring
   kernels (prologue/epilogue), keeping activations in shared memory or L2 across consecutive linears, or a
   persistent GEMM kernel with a stream-K schedule.
5. **Symmetry or structure in the data.**
   - Crystals have few species and repeated local environments (SrTiO3 has three species and two distinct sites).
     Can identical environments be exploited exactly?
   - Pair symmetry r_ij = −r_ji: we already share radial tables by species pair. What else can be shared?
6. **Hopper features we have not used well,** without the rejected patterns: thread-block clusters for receiver
   ownership, DSMEM reductions instead of global atomics, async transaction barriers, FP32 FFMA2-style dual issue.
7. **Anything from LLM inference engines** (FlashAttention-3, FlashInfer, ThunderKittens, the CUTLASS 3 persistent
   kernels) that maps onto a gather–transform–scatter graph workload.

## Deliverable

A ranked list of 8–15 ideas in the six-point format above, then:
- a short list of ideas you considered and found already covered above;
- the three ideas you would test first, with their falsifiers in enough detail to implement in an afternoon.

## Documents to attach (FastMACE repository, branch `fastmace/round5`)

Essential:
- `plans/h200-ideas-20261105.md` (on `fastmace/h200-ideas-20261105`): the H200 profile and the first nine ideas,
  with their reasoning.
- `docs/optimisation-ledger.md`: every optimisation tried, with its verdict. If it is too large, attach its index by
  area.
- `results/h200-defaults-20261105/README.md`, `results/h200-wgmma-20261105/README.md` and
  `results/wprime-pairs-20261105/README.md`: today's H200 measurements.
- `fastmace/kernels/oeq_streaming/fused_forward.py` and `fastmace/kernels/oeq_streaming/wprime_backward.py`: the
  two convolution kernels' generators (most of the step).

Useful:
- `SPEC.md` (architecture, rewrite catalogue, owner decisions);
- `docs/design/switch-map.md`;
- `results/omat0-h200-profile-0ab2bd9a/`;
- `tools/omat0_kernel_lab.py` (the falsifier variants, to see how bounds are measured).
