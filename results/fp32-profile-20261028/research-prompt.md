# Research prompt: further FP32 speed-ups for MACE MH-1 molecular dynamics in FastMACE

You are an expert in GPU performance engineering (CUDA, Triton, cuBLAS, CUDA graphs, PyTorch/Inductor) and in
equivariant machine-learning interatomic potentials (MACE, e3nn, cuEquivariance, OpenEquivariance). Your task is
to find **new, concrete ways to make one float32 molecular-dynamics step of MACE MH-1 faster without changing its
results beyond float32 rounding**, and to say for each one how much it could gain, how to falsify it cheaply, and
why it was not already covered by the work listed below. Prefer fewer, well-argued ideas over a long list. Every
estimate must be bounded by the measured shares below; say so where a gain would be hidden by stream overlap.

## 1. The system

**Software.** FastMACE is a GPU MD engine for MACE models (Python, PyTorch 2.14.1 + CUDA 13, torch.compile with
Inductor in default mode, CUDA graphs, cuEquivariance 0.12 and OpenEquivariance 0.7 kernels, plus its own Triton
and NVRTC kernels). One MD step is: an integrator/neighbour-check graph (positions, edge compaction), then one
captured CUDA graph for the force call (energy forward, then forces by reverse-mode autograd inside the graph),
then the second half-kick. Steps are pipelined (the host is one step ahead). Langevin NVT, 1 fs, typically
up to 1000 K, periodic crystals of 200–2000 atoms, one structure at a time.

**Model: MACE MH-1** (`matpes_r2scan` head), ScaleShiftMACE, r_max 6 Å, 2 interactions
(RealAgnosticResidualNonLinearInteractionBlock), correlation order 3.
- Interaction 0: node features 512x0e (species embedding only), edge irreps 128x0e, conv weights 512 per edge,
  radial MLP last layer [512, 64]; target irreps 512x0e+512x1o+512x2e+512x3o; gate input 2048x0e+512x1o+512x2e+512x3o.
- Interaction 1: node features 512x0e+512x1o, edge irreps 128x0e+128x1o, 10 tensor-product paths, conv weights
  1280 per edge, radial MLP last layer [1280, 64]; same target and gate irreps; hidden output 512x0e.
- Per interaction: skip_tp, linear_up, merged (linear_1 + linear_res) product, gated equivariant nonlinearity
  (silu scalars, sigmoid gates), linear_2, then a cuEquivariance symmetric contraction and a linear.
- Forces cost about 57 MFLOP per atom for the atom-level linears (forward + input gradients), plus per-edge work.

**Hardware.** Main target: NVIDIA RTX 5060 Ti 16 GB (sm_120, 36 SMs, 3.09 GHz, nominal FP32 28.5 TFLOPS,
practical SGEMM ~16 TFLOPS, 448 GB/s, 32 MB L2) under WSL2. Defaults must also hold on RTX 4000 Ada (sm_89) and
A100 (sm_80): no per-GPU or per-crystal tables; only rules from shapes and device properties.

## 2. Measured profile (28 October 2026, RTX 5060 Ti, 1 fs, 1000 K, 3×3×3 cells)

| Cell | Atoms | Padded edges | Step | Force call FLOP | DRAM per force call |
| --- | ---: | ---: | ---: | ---: | ---: |
| NaCl | 216 | 8,192 | 2.72 ms | 18 GFLOP | 0.5 GB |
| NaNbO3 | 1,080 | 82,688 | 14.36 ms | 109 GFLOP (62 atom GEMM, 40 edge) | 3.6 GB |
| NbCl5 | 1,944 | 58,880 | 18.68 ms | 148 GFLOP (111 atom GEMM, 27 edge) | 5.3 GB |

- The force graph is 96–99 % of the step. On NaNbO3 and NbCl5 a kernel runs 98 % of the time, on two graph streams
  that overlap by 2.9 and 1.9 ms. NaCl is launch/latency bound: 86–91 % busy, 233 kernels per force call,
  5 % in inter-kernel gaps, GEMMs at 21–25 % occupancy.
- NbCl5 kernel time by part (sums exceed the step because of overlap): `segment_gemm` Triton atom products 26 %
  (8.7–12.3 TFLOPS in the step, ≈cuBLAS in isolation), cuBLAS atom GEMMs 20 % (13–15 TFLOPS), streaming
  convolution backward (chunked, OEq-derived NVRTC kernel) 12 %, folded radial last-layer input gradient
  (Triton, `da = dw·W`, 8 % occupancy, side stream) 12 %, atom elementwise (gates, products) 12 % (~230 GB/s),
  fused radial-last-layer + convolution forward 7 %, symmetric contraction 6 % (72–96 % of memory speed of light),
  first-interaction edge sums 3 %, radial tables 2 %.
- NaNbO3 (77 edges per atom) is more edge-bound: input gradient 19 %, conv backward 18 %, cuBLAS 18 %,
  segment_gemm 14 %, fused forward 11 %.
- Neighbour list: skin 0.3 Å, rebuilt every 5–7 steps at 1 fs/1000 K on a side stream (0.2–0.7 ms of kernels per
  rebuild); NaCl pays ~0.1 ms per step for it, large cells 0.2–0.35 %. Edges are padded to 5 % buckets.
- Setup: 34–41 s warm, 71 s cold (not the focus, but welcome).

## 3. Already in the default plan (do not propose these again)

Species-factored radial input, cutoff folded into the radial MLP, certified piecewise-Chebyshev radial tables
(per species pair), density division before linear_1, reassociated first interaction (species-only sender
features) with segmented sums, receiver-sorted edges, receiver-streaming OEq convolution (uvu) with a fused
radial-last-layer + forward kernel that stores `w` [E,1280], an edge-owned backward chunked to fit L2 (1,536
edges per chunk) with the input gradient on a side stream, block/grouped/merged-residual linears per irrep,
gated rows (gate fused onto the merged product's rows), segment_gemm (Triton gathered-row GEMM writing straight
into the joined output, chosen by a waves rule), species tables for species-only products, folded product and
readout-tail linears, one-graph strain, fused Langevin integrator, pipelined steps, overlapped neighbour rebuilds,
async output, dynamic edge dimension, CUDA graphs per edge bucket.

## 4. Closed directions, with the evidence (do not reopen without new evidence)

- 16-bit storage or stages (bf16/fp16 weights, activations): fail the force accuracy gate. TF32 passes only for
  one tiny stage.
- Tensor-core FP32 emulation (3×TF32, 3×FP16, BF16×9, INT8 slices/Ozaki): slower than SIMT cuBLAS on sm_89 and
  A100; mixed on sm_120; BF16×9 only on sm_100.
- Persistent megakernel / whole-step kernel: co-execution probe failed on sm_120 and A100.
- Full fusion of radial last layer + convolution forward + backward (recompute `w` in backward): backward 1.12×
  slower (issue-bound, three 65×1280 products per edge). The input gradient alone has a measured whole-step
  ceiling of 0.5 % (NbCl5) and 2.6 % (NaNbO3) because it is hidden on the side stream.
- LunaTP-style source-owned or receiver-owned backward: slower (more bytes, less parallelism). Wide float2 loads:
  2–7 % kernel gain on sm_120 only, owner chose not to build.
- Low-rank radial weights: `w(r)` has full rank. Mirror-edge (i↔j) radial sharing: no gain.
- Split-K, per-l GEMMs on graph branches, combo kernels, Inductor aggressive fusion, finer edge buckets, larger
  fused-forward grid, alternative segment_gemm tiles (all 12 configurations within 3 % in isolation, neutral in
  the step), save-transcendentals partitioner in float32, cuBLASLt/cuBLASDx, own GEMMs/convolutions, programmatic
  dependent launch in Inductor (wrong forces), max-autotune, activation memory budgets, larger timesteps or any
  change of the dynamics (excluded by the owner), batching several structures (excluded by the owner).
- Host/Simulation overhead: ≤ 1.7 % (NaCl) in a same-process A/B.
- Pending, not yet default: per-species residual table in interaction 0 (−0.4 to −0.8 % on Ada), skin 0.66–1.0 Å
  (−1 to −1.7 % NaCl, neutral on large cells), sync-free neighbour rebuild (neutral on Ada).

## 5. Questions to research

1. **Algebra.** Are there exact (or float32-rounding-level) reformulations of the MH-1 forward or its reverse-mode
   force pass that remove FLOPs from the atom-level linears (58 % of their FLOPs are the two `linear_2` maps
   512→512 per l before the channel-wise symmetric contraction) or from the per-edge 65↔1280 radial products
   (≈75 % of edge FLOPs)? Consider equivariance, the channel-wise structure of the symmetric contraction, the
   gate, the fact that only forces (first derivatives) are needed, and that species never change in a run.
2. **Backward structure.** Is reverse mode doing redundant work here (gradients never consumed, recomputation,
   activations stored and re-read) that a hand-written or restructured backward for MACE's specific blocks could
   avoid? Quantify against the shares above.
3. **Memory traffic.** Where is DRAM/L2 traffic avoidable at equal FLOPs (elementwise around the gates, the
   `w`/`dw` tensors, the symmetric contraction), given the 32 MB L2 and two concurrent streams?
4. **Small cells.** For ~200-atom cells (latency/launch bound), what reduces the critical path without batching
   structures: fewer kernels per force call, different work decomposition of the 512-wide linears at M = 216–1512
   rows, graph-level scheduling?
5. **Literature and libraries since mid-2026.** Relevant work on fast equivariant tensor products, fused MACE
   kernels, SIMT FP32 GEMM on Blackwell consumer GPUs, Triton or CUTLASS techniques for gathered/scattered GEMMs,
   and MD-engine tricks (e.g. in LAMMPS-ML-IAP, ALCHEMI, TorchSim, Symmetrix, cuEquivariance releases). Cite and
   say concretely what would transfer to this setup and what was already measured (Symmetrix-XL, ALCHEMI and
   TorchSim were 1.1–10× slower than FastMACE in our comparisons).

## 6. Deliverable

A ranked table of at most ten ideas. For each: mechanism; which part and share it targets; an upper bound on the
whole-step gain per cell, after stream overlap; exactness class (bitwise / rounding / approximate, the last
excluded); why it is not in section 3 or 4; a falsifier that takes under a day (microbenchmark or stub with a
stop rule); generality across sm_120, sm_89 and sm_80. Flag any idea that conflicts with section 4 and explain
what new evidence would justify reopening it. State plainly if you believe no exact idea above ~2 % remains.
