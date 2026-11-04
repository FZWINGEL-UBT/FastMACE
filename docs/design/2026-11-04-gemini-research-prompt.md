# Research prompt: further speed-ups for MACE-OMAT-0 molecular dynamics in FastMACE (4 November 2026)

*Written for an external research agent. The only attachment is `omat0-a100-optimisations.md` ("the document"). FastMACE
main is `37e12689`. Everything you need about the current design is in the document; this prompt adds the goal, the
rules, and what has already been tried and rejected.*

## Your task

Find new, concrete ways to make FastMACE's molecular dynamics faster for **MACE-OMAT-0 medium in float32**, ranked by
expected gain. Primary target: the whole-system path on the **NVIDIA A100** (SXM4-80GB and 80GB PCIe), measured with
the Benchmark_loup protocol (SrTiO3 10×10×10, 5,000 atoms, about 374,000 edges, skin 0.5 Å, 1 fs). Secondary targets,
in this order:
- the same path on the RTX 4000 Ada (sm_89) and RTX 5060 Ti (sm_120);
- the speed of the tiled and wavefront capacity paths (document §4–§5) at 1M to 150M atoms.

Read the document first: §2 is how a step runs today, §3 what each part does and where it came from, §6 how
Symmetrix-XL differs.

## Where the time goes now (A100-SXM4-80GB, from the document §2)

The step is 13.77 ms and is entirely GPU-bound. One force call is 13.7 ms in 171 kernels, with gaps of a few µs in
total:

| Part | ms per call | Share |
| --- | ---: | ---: |
| w′ backward of interaction 2 (vendored OpenEquivariance PR 223 tensor-product backward, `w` and its slope rebuilt from radial tables) | 4.37 | 32 % |
| Fused forward of interaction 2 (3×TF32 `mma.sync` product of the folded radial layer inside the receiver-streaming convolution) | 3.70 | 27 % |
| Atom GEMMs (CUTLASS 3×TF32, 64×64×16 tiles, 44 launches) | 1.85 | 14 % |
| Atom-level elementwise (Inductor-fused, about 74 kernels) | 1.07 | 8 % |
| Interaction 1 reassociated edge sums, forward + backward | 0.72 | 5 % |
| Radial tables forward + backward (degree-5 Chebyshev) | 0.72 | 5 % |
| Symmetric contraction (cuEquivariance) | 0.40 | 3 % |

Benchmark_loup today: NVE 2.73–2.97 µs/atom/step, Langevin 2.79–2.95 on the A100-SXM4-80GB. The A100 80GB PCIe
measured about 3.1 at a slightly older commit. Symmetrix-XL under LAMMPS: 5.00 on the PCIe card; its paper gives
5.80 on the SXM4.

## Rules a proposal must respect

- **Accuracy gate.** Against a float64 reference, every proposal's energy, force and stress errors may be at most
  twice those of MACE's own float32 calculator (the "ratio rule").
- **Exactness classes.** Every rewrite carries one:
  - **exact:** the same arithmetic up to summation order;
  - **rounding:** reassociated float32;
  - **certified:** an approximation with a setup-time certificate that its float32 error is at most that of the
    float32 computation it replaces, as the radial tables have.
  - No other approximations are accepted.
- **Precision.** Plain TF32 and 16-bit arithmetic on the force path are excluded (see below). Error-compensated
  3×TF32 is allowed.
- **Generality.** No per-crystal or shape-keyed tables, and no caches of measured timings. Per-GPU defaults are
  allowed, matched by exact device name, when same-process A/Bs of the whole MD step in both build orders show a
  gain (decision 66). Per-model work generated at setup from the checkpoint is allowed.
- **Scope.** One structure on one NVIDIA GPU at a time, in production. No batching of structures, no multi-GPU, no
  retraining or distillation.
- **What counts as a result.** A measured whole-step gain, not a kernel micro-benchmark. Gains below about 1–2 % are
  within run noise.

## Already tried and rejected (do not propose again without a new angle)

**General (October 2026):**
- **16-bit storage or arithmetic on the force path**, including BF16/FP16 linears and edge weights, delta GEMMs and
  channel splits: errors 3× to more than 30,000× over the gate.
- **Plain TF32** anywhere: force error about 1,000× (owner rule: never).
- **3×TF32 or 3×FP16 GEMMs on consumer GPUs**, and Triton 3×TF32 on the A100: slower. CUTLASS 3×TF32 on the A100 is
  already in use.
- **Megakernels:** a complete force megakernel, persistent scheduling, heterogeneous co-execution of tensor and CUDA
  cores, two-stream branches. Ceilings are 1–16 %, and the gates failed.
- **Inductor options:** programmatic dependent launch (wrong forces in graph replays), combo kernels,
  `max-autotune`, GEMM autotuning, `decompose_mm_pass`.
- **Own GEMMs:** own FP32 GEMMs, split-K, cuBLASLt, cuBLASDx, a hand-written CUDA GEMM.
- **Full forward + backward fusion of the radial layer into the convolution:** 0.80–0.83 against a 0.75 stop rule.
  It became the current fused forward plus the table-based backward.
- **Low-rank radial or atom-linear bases:** full rank is needed (65 of 65, 509–512 of 512).
- **Other backward and convolution designs:**
  - receiver-owned and source-owned streaming backwards (slower);
  - LunaTP (1.4–1.5× slower);
  - OEq launch-shape, L2-hint and register or block tuning (ceiling 0.3 %).
- **Symmetric contraction rewrites:** cuEquivariance runs at 72–96 % of memory speed of light.
- **Gate VJP and prologue/epilogue fusions** (≤ 1 %).
- **Smaller variations:** edge-bucket width and skin re-tuning (noise), Morton or species atom order (no gain), and a
  speculative step graph (the pipelined step won).

**OMAT-0 A100 kernel round (3–4 November 2026):**
- **Launch settings:** threads, blocks per SM, register caps. No setting beats the defaults (within 0.3 %).
- **w′ slope loads:** vectorised slope loads (slower at degree 5); streaming loads of `w` (1.005).
- **Radial table degree and tolerance:** degrees 3, 4, 6 and 7 (degree 3 fails the certificate; the others are
  slower); looser fit tolerance (no gain).
- **Edge order:** sorting edges by species pair and table piece for the w′ backward is 1.51× slower, and even perfect
  slab locality is slower.
- **Rebuilding the input gradient:** the `da = dw·W` path costs 2.27 ms against the 1.81 ms slope it would replace.
- **Tensor-core forward variants:**
  - the first split rounded every value twice and was slower;
  - splitting W once per block is 1.36× slower (12 → 8 warps);
  - the mask split is the one in use.
- **CUTLASS tiles:** 32×64 and 64×32 tiles, no gain over 64×64.
- **Species-grouped GEMM for the skip product on the A100:** 0.99, not adopted there.
- **`TABLE_W` follow-ups:** float4 slabs, degree changes, and computing `w` from the slab in the forward (spills;
  slower).

**In progress elsewhere (do not duplicate):**
- **A model-specialised tensor-product backward:** a generated kernel with constant Clebsch–Gordan coefficients and
  unrolled paths, replacing the vendored generic one in the w′ backward. Its falsifier and A/B are running.
- **A tensor-core forward that batches 32 edges per warp,** to reuse each W fragment. Estimated at most 10 % on the
  A100; not built.

## Questions to research

1. **The w′ backward (32 %).** It runs the vendored OpenEquivariance receiver-streaming UVU backward, with atomic
   sender-gradient sums. Profiles say it is bound by that tensor-product backward itself. Beyond model
   specialisation, what is known about faster equivariant-convolution backward schedules? Look at
   OpenEquivariance, cuEquivariance, FlashTP, the Sobek paper (arXiv 2607.18074), e3nn-jax, Symmetrix-XL and NequIP
   or Allegro kernels. Give each one's mechanism and whether it applies to a UVU product with per-edge weights
   rebuilt in registers.
2. **The fused forward (27 %).** After the 3×TF32 product and writing only the message, what is left? Consider
   memory traffic for the message, the shared-memory schedule, and splitting work between tensor and CUDA cores
   within one kernel. Use A100 numbers: about 19.5 TFLOPS FP32, 156 TFLOPS TF32, 2.0 TB/s, 40 MB L2, 164 KB shared
   memory per SM.
3. **Fewer edge passes.** Interaction 2's forward and backward each stream all edges, and the w′ backward re-reads
   the table. Is there an exact or certified reformulation that shares one pass, or that moves edge work to atom
   work, as the reassociation of interaction 1 did? Interaction 2's sender features are not species-only.
4. **The atom side (about 25 %).** It is GEMMs, Inductor elementwise and the symmetric contraction. Look for
   reformulations that fold products together across interaction boundaries, readouts or residuals, at setup from
   the frozen weights, and that are exact or rounding-class.
5. **The A100 specifically.** Which A100 features are unused? Candidates: asynchronous copies (`cp.async`), L2
   residency control, larger shared memory per block, and TF32 tensor cores in other kernels where 3×TF32 still
   pays.
6. **Capacity paths (secondary).** The tiled and wavefront paths run compiled tiles in three sweeps, with host
   state. Name the main ideas for making a tiled force call cheaper per atom, at 1M atoms and at 150M (document §4–§5).
   Today the tiled call costs about 15 µs/atom/step at 1M on an RTX 4000 Ada, against about 13 for the
   whole-system path on the same GPU.

## Deliverable

A ranked list of at most ten proposals. For each:
- **The mechanism,** precise enough to implement: which tensors, which kernel, what changes.
- **The exactness class** and why it holds.
- **An estimated ceiling** in ms per force call on the A100-SXM4, derived from the table above and stated as an
  estimate.
- **A cheap falsifier** that would kill it in a day: a stub, a bound, or a micro-benchmark with a stop rule.
- **Sources,** with links. Say where the idea comes from (a paper, a project, or your own), and how it differs from
  every rejected item above that it resembles.

Mark proposals you are unsure about as such. Leave out anything that breaks a rule above, or that only helps batched
or multi-GPU runs.
