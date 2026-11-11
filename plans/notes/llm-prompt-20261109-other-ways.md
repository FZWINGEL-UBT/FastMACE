# Research brief: entirely different ways to make MACE molecular dynamics faster

You are advising a GPU performance team that has spent about a month optimising MACE inference kernels and has
reached the hardware's limits. We are **not** asking for more kernel tuning. We are asking for **different ways to
get the same MD trajectories (or scientifically equivalent ones) in much less wall time**: changes to the algorithm,
the dataflow across MD steps, the numerics, the hardware mapping, or the simulation protocol. Search widely: papers
and preprints in ML potentials, classical MD, numerical analysis, HPC, sparse/graph ML, compilers and computer
architecture, plus engine source code and talks. Bring back concrete mechanisms with arithmetic. We do not want
general advice.

Everything you need is in this brief.

## 1. What we run

- **Engine.** Our own single-GPU inference engine for pretrained MACE potentials. It uses PyTorch for orchestration,
  custom NVRTC CUDA kernels for the edge work, CUTLASS 3×TF32 GEMMs for the node linear layers, cuEquivariance or
  generated kernels for the symmetric contractions, and runs the whole MD step as CUDA graphs. Forces and stress come
  from hand-written backward kernels (energy → forces, exact autograd equivalents). One structure at a time.
- **Model.** MACE-OMAT-0 "medium": a pretrained foundation model, 89 elements, float32.
  - Two message-passing interactions, 128 channels, cutoff 6.0 Å.
  - Edge spherical harmonics up to l = 3 (16 components); hidden features 128x0e + 128x1o.
  - Symmetric contractions with correlation order 3.
  - "Density" interaction blocks: the message is divided by (1 + a learned per-atom density sum).
  - Interaction 2's radial part is 1,280 per-edge weights (10 tensor-product paths × 128 channels), a function of
    edge length and the (sender, receiver) species pair. We evaluate it from a certified cubic spline table.
  - "Small" (64 channels) is also supported.
- **Benchmark.** SrTiO3 10×10×10 supercell, 5,000 atoms, Langevin MD at 300 K, 1 fs time step. About 370,000
  directed edges (about 74 neighbours per atom). Verlet skin neighbour list with a 0.5 Å skin, rebuilt about every 8–13
  steps on the GPU.
- **Hardware.** NVIDIA H200 (sm_90, 132 SMs) and A100-SXM4-80GB (sm_80, 108 SMs). We have access to many A100s (an
  HPC cluster, about 7 at once) and a few H200s (cloud).
- **Speed today (ms per MD step, 5,000 atoms).**

  | Model | H200 | A100 |
  | --- | ---: | ---: |
  | OMAT-0 medium | 3.37 | 6.51 |
  | OMAT-0 small | 2.12 | 4.17 |

  The next best engine we measured (Symmetrix-XL) runs OMAT-0 small at 11.5 ms on the H200.

## 2. Why kernel work is finished

The force call is 97–98 % of the step. On the A100 the kernel time per force call is 6.2 ms:

| Part | µs | What limits it |
| --- | ---: | --- |
| Interaction-2 backward through the radial weights (per edge: gradients for sender features, harmonics, length) | 2,177 | FP32 FMA pipe at 83 % of peak |
| Interaction-2 forward (per receiver, sum over edges of radial weights × sender features × harmonics, 10 CG paths) | 1,255 | FP32 FMA pipe at 84 % |
| Interaction-1 radial-table outer sums, forward and backward | 832 | L1 bandwidth, load latency |
| Node linear layers, 3×TF32 tensor-core GEMMs | ~980 | tensor-core rate for 3×TF32 |
| Symmetric contractions (generated kernels) | 188 | FMA pipe at 93 % |
| Everything else (geometry, harmonics, glue) | ~800 | small and spread out |

On the H200 the two big edge kernels are instruction-issue bound (77–82 %), not memory bound.

Things that would only buy a few percent each have been measured and closed. Do **not** propose these:

- Precision: FP16, BF16 or FP8 anywhere on the force path (fails the accuracy gate by 3× to over 1,000×); 1×TF32;
  3×FP16.
- Tensor cores for the per-edge tensor product, in any layout (3×TF32: a built version was 28 % slower; the best
  layout's bound is about 0.8 of the TP part). FP64 units or FP64 tensor cores as a second FMA pipe (register
  pressure; 1.04–1.19× slower).
- Table forms: 8-byte packed tables (adopted), B-splines, quadratic tables, a shared low-rank h(r) spline.
- Mirror edges (i→j and j→i sharing work), in every form: 1.3× to 3.8× slower.
- Ownership and fusion changes: receiver- or sender-owned backward, megakernels, persistent kernels for the glue,
  bulk copies and TMA, overlapping GEMMs with edge kernels, register caps, warp counts, prefetch depth.
- Certified skipping of edges, paths or channels (a perfect oracle skips 0.4 % of edges); FFT, mesh or FMM-type
  operators; radial tail pruning; temporal force extrapolation without correction.
- Sorting or reordering edges for locality (the kernels are compute-bound).

## 3. What we want: different ways, not faster kernels

Look for mechanisms in categories like the ones below. These are prompts for your search, not a list to fill in:

1. **Exploiting time coherence between MD steps.** Positions move about 0.001–0.01 Å per step. Can the force for
   step n+1 be computed from step n's intermediate results more cheaply, exactly or with a controlled error? Think
   of:
   - incremental or delta evaluation of messages for edges whose length changed little;
   - Taylor or Krylov updates of node features;
   - predictor-corrector schemes that evaluate the full model only every k steps and a cheap correction in between
     (with energy-conservation evidence);
   - work from classical MD (multiple time stepping, force splitting into fast and slow parts), quantum chemistry
     (extended Lagrangian Born–Oppenheimer MD, which reuses the previous step's density as the starting guess) and
     ML-potential MD.
   Be specific about what is exact, what is approximate, and how drift is controlled.
2. **Exploiting the structure of the model mathematically.**
   - Low-rank or factorised forms of the 1,280 radial weights across channels or paths, exact for this checkpoint
     (for example via the checkpoint's own MLP structure).
   - Algebraic rearrangements that move work from edges (370,000) to atoms (5,000) or species pairs (9 here).
   - Precomputation per species pair and length bin that turns the edge loop into a lookup plus a small contraction.
   - Different but exact orderings of the Clebsch–Gordan sums.
   - Anything that reduces the operation count asymptotically rather than by a constant.
3. **Changing what is computed per step without changing the physics.**
   - Larger stable time steps through constraints, mass repartitioning or thermostat choices compatible with MLIPs.
   - Multiple-time-step integrators where only part of the model is slow.
   - Evaluating forces at lower cost and correcting energy drift.
   Say which change scientific users would accept and cite where it has been validated for MLIPs.
4. **Different hardware mappings.**
   - Several GPUs per structure (domain decomposition for MACE-type message passing with two layers of halo).
   - Many replicas per GPU (throughput rather than latency) and when that is what users actually need.
   - CPU+GPU co-scheduling.
   - New GPU features: Blackwell FP32 emulation on tensor cores, Hopper clusters and distributed shared memory.
   - Other accelerators. Quantify against our per-step numbers.
5. **Numerics.** Mixed-precision schemes that keep float32-equivalent forces through error compensation:
   - compensated or split arithmetic;
   - Ozaki-style splitting;
   - error-free transformations;
   - lower precision only where the gradient is provably insensitive.
   Give the accuracy argument, not just the speed.
6. **Model-level options, flagged as such.** Distillation, pruning, smaller cutoffs, fewer channels or a cheaper
   architecture with the same accuracy on the target systems. The project owner decides these. List the best few,
   with published accuracy/speed trade-offs, but keep them separate from 1–5.
7. **Anything else genuinely different** that does not fit the categories above.

## 4. Constraints

- **Accuracy.** Exact rewrites must agree up to float32 rounding. Anything else must pass a force/stress gate: within
  3× the float32 reference's own error against float64, on the user's structure. For MD-protocol changes, state the
  observable you would check (energy drift, RDF, diffusion, phonons) and the published evidence.
- **Generality.** It must work for any structure and species set, not only SrTiO3, and on both GPUs.
- **Evidence.** Prefer mechanisms demonstrated somewhere (any field) over speculation. Say so when something is
  speculative.

## 5. What to return

For each proposal, ranked by expected gain in time per MD step (or per nanosecond of trajectory):

- **Idea and mechanism**, concrete enough to prototype: what is computed, when, and what is reused.
- **Source:** paper, repository or talk with a link, the field it comes from, and whether it has been used with
  MLIPs or equivariant networks.
- **Gain estimate with arithmetic** against section 2's numbers, including the overhead it adds.
- **Accuracy class:** exact, rounding-only, controlled approximation (with the control), or physics change.
- **Cheapest falsifying experiment:** what we would run first to kill it in a day.
- **Novelty check:** why it is not one of the closed items in section 2.

Return 5–10 well-founded proposals rather than a long list, and a short closing verdict: is a 2× or larger gain at
the same accuracy plausible for this model and system, and by which route?
