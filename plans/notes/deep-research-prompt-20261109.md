# Deep research brief: new ways to make MACE-OMAT-0 medium molecular dynamics faster on one GPU

You are doing a deep literature and technique search for a GPU performance engineering team. Everything you need is
in this brief; there are no attached files. Search widely (papers, preprints, GitHub repositories, engine source
code, GTC/SC/PPoPP/MLSys talks, NVIDIA documentation), and bring back concrete, *new* mechanisms with arithmetic,
not general advice.

## 1. What we run

- **Engine:** our own inference engine for pretrained MACE interatomic potentials (PyTorch + custom NVRTC CUDA
  kernels + CUTLASS + cuEquivariance for some ops, whole MD step in CUDA graphs). Single GPU, one structure at a
  time, no batching across structures.
- **Model:** MACE-OMAT-0 "medium" (pretrained foundation model, 89 elements), float32. Two message-passing
  interactions, 128 channels, cutoff 6.0 Å, edge spherical harmonics up to l = 3 (16 components), hidden features
  128x0e + 128x1o, symmetric contractions with correlation order 3, "RealAgnosticDensity" interaction blocks (the
  message is divided by (1 + a learned per-atom density sum)). The radial part of interaction 2 is 1,280 per-edge
  weights (10 tensor-product paths x 128 channels) as a function of edge length and the (sender, receiver) species
  pair.
- **Benchmark:** SrTiO3 10x10x10 supercell, 5,000 atoms, Langevin MD at 300 K, 1 fs step, about 369,000 directed
  edges within 6 Å (about 74 neighbours per atom), Verlet skin list (skin 0.5 Å, rebuilt about every 13 steps).
  Forces and stress come from autograd-equivalent hand-written backward kernels (energy -> forces).
- **Hardware targets:** NVIDIA H200 (sm_90, 132 SMs, 50 MB L2) and A100-SXM4-80GB (sm_80, 108 SMs, 40 MB L2).
  Smoke tests on an RTX 5060 Ti (sm_120).
- **Current speed:** 3.42 ms per MD step on the H200, about 7.06 ms on the A100.

## 2. Where the time goes now (H200, one force call, about 3.2 ms of kernel time)

| Kernel | µs | Notes |
| --- | ---: | --- |
| Interaction-2 backward through the radial weights ("w-prime" kernel: per edge, gradient w.r.t. sender features, edge harmonics and edge length) | ~1,090 | one launch; long-scoreboard stalls dominate; profile says the stalls come mostly from the sender-feature rows (512 floats per sender) and staged output-gradient rows, *not* the radial table; LSU busy 64 %, 96 registers |
| Interaction-2 forward (message per receiver) | ~600 | one block per receiver, one thread per channel, edges walked serially; w(r) read from a cubic Hermite spline table |
| Interaction-1 outer-sum forward / backward (radial table x harmonics, summed per receiver and sender species) | 115 / 214 | |
| Node-level linear layers (3xTF32 CUTLASS, 4 launches each for forward and gradient, plus per-species skip GEMMs) | ~145 + ~200 | low occupancy (18 %), 1–2 waves |
| cuEquivariance symmetric contractions (both layers, forward + backward) | ~165 | |
| Everything else (edge geometry, harmonics, density sums, small GEMVs, elementwise glue, ~80 small launches) | ~600 | |

**Key measurements you must respect:**

- **Radial table.** The interaction-2 forward reads a 20 KB slab of float4 cubic coefficients per edge (1,280
  weights x 4 floats), 736 intervals x 9 species pairs. If every edge reads the same slab, which is perfect L1
  reuse, the forward drops only from 601 µs to 433 µs, so the table costs about 170 µs. In w-prime, the same test
  saves only about 3 %. **The radial table is not the main bottleneck any more.**
- **Forward floor.** Even with perfect table reuse the forward costs 433 µs. Its remaining cost is the
  per-edge-per-channel work and the gathers: the sender features and 16 harmonics per edge, about 10 paths with
  their Clebsch-Gordan terms, accumulated into 40 output components per channel in registers.
- **w-prime reductions and atomics are cheap.** Removing w-prime's per-edge cross-channel reduction entirely saves
  3.6 %. Removing its scatter atomics does not help.

## 3. What is already done (do not propose these again)

- **Symmetrix-XL-style interaction-2 layout** (adopted): one thread per (receiver, channel), edges walked serially,
  message held in registers, w(r) from a certified cubic spline table instead of the radial MLP.
- **Edge staging and ordering:**
  - edges staged per 32-edge tile;
  - receivers processed grouped by species;
  - each receiver's edges sorted by length (better table reuse);
  - prefetch hints (A100).
- **GEMM work:**
  - 3xTF32 tensor-core GEMMs (CUTLASS) with fused division epilogues;
  - species-grouped GEMMs;
  - composed interaction-1 projection + linear + species skip;
  - folded readout tails.
- **Other:**
  - CUDA graphs for the whole step;
  - GPU skin neighbour list with per-step compaction;
  - binned pair search for large systems.

## 4. What is already ruled out (with the evidence), so do not return to it without new evidence

- **16-bit precision on the force path.** FP16/BF16/FP8 storage or arithmetic, including fp16 spline
  coefficients, fails our force/stress accuracy gate by 3x to more than 1,000x. 1xTF32 GEMMs fail too.
- **Compressed coefficients.** Spline coefficients packed in 8 bytes (shared exponent + fixed point) fail the
  value certificate so far.
- **Other spline forms.** A cubic B-spline table (shared control points) needs 851 intervals and recovered only
  11 % of the table's share in a kernel lab: four scalar loads per weight. A 64-channel h(r) spline in place of the
  per-weight table needs too many FMAs.
- **Edge-frame / SO(2) rotations, Gaunt or Cartesian tensor products.** The angular arithmetic is small next to
  memory and latency. Cartesian adjoints in w-prime were 6 % slower.
- **Ownership changes.** Receiver-owned or source-owned backward, w-prime channel splits over warps, receiver-local
  fusion of forward + node update + backward, and a whole-interaction megakernel: all measured slower or bounded
  out. A per-receiver block would re-read every node-linear weight.
- **Bulk copies.** Each edge's slab bulk-copied (TMA / cp.async.bulk) into shared memory was slower.
- **Scheduling.** Grouped GEMMs (one launch instead of four): at most 61 µs. Overlapping tensor-core GEMMs with the
  edge kernels: too little independent work.
- **Skipping or approximating work.** Certified adaptive skipping of edges/paths/channels: a perfect oracle skips
  0.4 % of edges. Spatial operators (FFT/mesh, FMM/H-matrices) for the edge sums. Global low-rank radial bases.
  Radial tail pruning. Temporal force extrapolation.
- **Dynamics changes** such as multiple-timestep / r-RESPA: owner decision, not pursued.
- **Distillation** to a smaller model: not wanted at this stage.

## 5. Constraints

- **Exactness.** Rewrites must be exact up to float32 rounding, or pass a certificate (spline tables: error below
  the float32 network's own error) and a force/stress gate: within 3x the float32 reference's own error against
  float64.
- **Generality.** It must work for any structure and species set, not tuned to SrTiO3 or one GPU model.
- **Adoption.** A change is adopted only if it is faster in both A/B build orders on the target GPU.

## 6. What to research

Find mechanisms we have not tried, in the spirit of "we adopted Symmetrix-XL's layout and it paid". In particular:

1. **Other engines' tricks for equivariant message passing on GPUs.** What do Symmetrix, cuEquivariance,
   OpenEquivariance, FlashTP, NVIDIA's MACE/LAMMPS ML-IAP Kokkos path, Allegro/NequIP kernels, SevenNet, eSEN/UMA,
   GPUMD's NEP, DeePMD-kit's model compression, TorchMD-Net and the JAX MLIP engines do that we don't?
   - Layouts, edge orderings, fused backward strategies, how forces are formed (edge-force vs autograd), radial
     tabulation, precision tricks that keep float32 accuracy.
   - Give the engine, the mechanism, and where in their code or paper it is.
2. **The w-prime problem specifically.** A gather-heavy edge backward where each edge reads a 512-float sender row
   and a receiver's output-gradient rows, and writes 512 floats of sender-feature gradient by atomics. What are the
   best-known ways to cut its load latency?
   - sender-row reuse via graph reordering (RCM, Morton, METIS) or edge blocking;
   - recomputation instead of storage;
   - transposed or source-centric formulations with deterministic reductions;
   - warp-specialized producer/consumer pipelines (cp.async, TMA, wgmma on Hopper);
   - L2 residency control (access-policy windows, persistence).
   Include anything from sparse-GNN kernel literature (SpMM/SDDMM fusion, FusedMM, GE-SpMM, edge-blocking) that
   transfers.
3. **The forward's 433 µs floor.** Ways to make per-edge, per-channel CG-term accumulation cheaper:
   - path factorization or common-subexpression schemes;
   - reusing work between mirror edges i->j and j->i (same length, same harmonics up to parity);
   - tensor-core formulations that keep float32 accuracy (3xTF32, split-float tricks, DGEMM-emulation-style
     schemes such as Ozaki);
   - instruction-level tricks on sm_90.
4. **Whole-step structure.** Anything that changes the dataflow:
   - forming forces from pairwise edge forces;
   - reusing interaction-1 work across MD steps exactly (incremental or delta updates when neighbour lists are
     unchanged);
   - exact caching across steps;
   - persistent kernels / megakernel schedulers for the ~80 small launches;
   - Hopper features (thread-block clusters, distributed shared memory, TMA multicast) that help receiver-centric
     kernels.
5. **Anything else genuinely new.** Algorithms from other fields (N-body codes, molecular-mechanics engines such as
   GROMACS/OpenMM/AMBER pair kernels, lattice/stencil codes, sparse attention kernels) that map onto
   "per-receiver sum over ~74 neighbours of a 1,280-weight radial function times a 16-component harmonic times a
   512-float neighbour feature".

## 7. What to return

For each proposal, ranked by expected H200 gain:

- **Mechanism** in enough detail to implement: data layout, which thread owns what, what moves where.
- **Source:** paper, repository and file, or talk, with a link, and whether it has been demonstrated on equivariant
  MLIPs.
- **Bound arithmetic** using the numbers in section 2: which kernel's time it attacks, its ceiling in µs, and the
  first cheap experiment that would falsify it.
- **Novelty check** against sections 3 and 4: say explicitly why it is not one of those.
- **Exactness class:** bitwise, rounding-only, or certified approximation.

Prefer 5–10 well-founded proposals over a long list. Say plainly if your research suggests the remaining gains are
small, and where the practical floor for this model on an H200 lies.
