# Persistent megakernel for the force call: external research report (20 October 2026)

The owner ran the research prompt from the doc "FastMACE: speeding up
small-cell MD steps" through an external research model. Its report is kept
below unchanged as a record. Its sources were not re-checked here, and its
numbers are estimates or literature figures, not our measurements. The
conductor's assessment and what the plan takes from it come first.

## Conductor assessment (20 October 2026)

- **Agrees with our own estimate.** No complete force megakernel now. Its
  persistence-only budget of 0–15 % (regression possible) matches our fit,
  which gives a fixed part of 0.32 ms, a ceiling of 8–16 % on NaCl.
- **New and important:** it reports that cuBLAS's BF16×9 FP32 emulation is
  restricted to compute capabilities 10.0 and 10.3, so it would not exist on
  either of our GPUs (sm_89, sm_120). That leaves 3×TF32 (Triton
  `input_precision="tf32x3"`, CUTLASS) for OPT-T01. To be verified in OPT-T01's
  survey against the cuBLAS documentation.
- **Useful for OPT-T01:**
  - export the real GEMM shapes and strides, since irrep layouts can fold
    components into rows (M is not always the atom count);
  - test forward products and their input gradients against float64;
  - set the Triton precision per kernel explicitly, since torch's TF32 flag
    does not govern custom kernels.
- **Useful for OPT-T02:**
  - cuBLASDx as a device-callable block GEMM for fused linear + epilogue
    kernels;
  - count the backward-pass savings, and check that saved preactivations do
    not cancel the forward traffic savings.
- **Its ranking:** fusion of atom linears with their epilogues and removal of
  layout copies (5–15 %) first, 3×TF32 (10–22 %) second. That differs from the
  owner's order (tensor cores first) but changes nothing in the plan: both are
  OPT-T01 and OPT-T02, measured separately.
- **Persistence:** a staged experiment with stopping rules (its milestones
  A–E), recorded in `plans/opt-1.md` as optional OPT-T05. It starts only after
  OPT-T02, if 2-T20a's trace shows removable transitions.
- **Corrections to note:**
  - FastMACE's fused radial hidden block exists, but is off by default (no
    gain, 15 October). The report rightly says availability is not proof of
    use.
  - Symmetrix-XL was screened on 14 October and was 1.9–2.5× slower than our
    hybrid default on our cells (ResearchModule
    `studies/symmetrix_xl/README.md`).

## Report (verbatim)

**Recommendation: prototype one fused atom-level block first. Do not commit to a complete MACE megakernel yet.** Your measurements justify investigating small-GEMM efficiency and intermediate-memory traffic. They do not establish that persistent scheduling will recover enough time to outweigh its synchronization cost and the need to replace existing library kernels.

A useful target is **10–25% lower whole-step time from selected kernel improvements**. For persistence alone, I would budget **0–15%, including the possibility of regression**. These are engineering estimates, not published MACE results.

I inspected FastMACE at commit [`05ecbe62`](https://github.com/FZWINGEL-UBT/FastMACE/tree/05ecbe62c862928d465613bf083483bff24bb30b). No GPU benchmarks were executed for this report; your supplied timings are the performance baseline.

**1. What the state of the art actually demonstrates**

Persistence means keeping worker blocks resident and assigning them multiple tasks. It does **not** automatically keep every intermediate on chip, remove arithmetic, or make existing kernels callable from inside another kernel.

| System | Fusion and dependency mechanism | Reported evidence and relevance |
|---|---|---|
| **Mirage Persistent Kernel, MPK** | Compiles operators into tiled tasks and events; decentralized GPU schedulers dispatch ready tasks. Enables cross-operator pipelining. | Paper reports **1.0–1.7×** batch-one serving improvements over optimized systems across five BF16 LLMs on A100/H100/B200; batches 1–16. No demonstrated MACE backward or consumer-GPU result. [arxiv.org](https://arxiv.org/html/2512.22219v1) |
| **Hazy "No Bubbles"** | Full Llama-3.2-1B forward pass. Python generates per-worker instruction streams; counters enforce dependencies; shared-memory pages permit prefetch across instructions. | Single-sequence BF16 decoding: **>1.5× over SGLang**, H100 bandwidth utilization **78%**. Its B200 breakdown still spends substantial time exchanging activations and synchronizing. This supports the concept, not a transferable MACE speedup. [Hazy Research](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) |
| **ThunderKittens** | CUDA tile primitives and pipelining infrastructure for implementing fused kernels; a building block for Hazy's megakernels. | Not an automatic arbitrary-network compiler. Its attention/GEMM results are operator-specific; no verified full FP32 MACE-force benchmark. [GitHub](https://github.com/HazyResearch/ThunderKittens) |
| **FlashInfer** | Fused attention kernels, planned load balancing and split reductions, with CUDA-graph-compatible execution. | Paper reports **29–69% inter-token latency reduction** in its serving comparisons. Those measurements combine attention and serving improvements, rather than full-model persistent execution; they cannot predict MACE gains. [arxiv.org](https://arxiv.org/abs/2501.01005) |
| **CUTLASS persistent / Stream-K** | Persistent workers traverse output tiles; Stream-K distributes reduction work more evenly and combines partial outputs. | Original Stream-K study spans **32,824 GEMM geometries**, reporting peak improvements up to **14×/6.7×** against its CUTLASS/cuBLAS baselines. These are exceptional GEMM cases, not expected end-to-end gains. [arxiv.org](https://arxiv.org/abs/2301.03598) |
| **Triton persistent matmul** | A bounded grid loops over tiles; variants pipeline memory and matrix operations. | Official tutorial primarily demonstrates FP16/FP8. Its displayed 8192×8192×512 FP16 example is approximately **3% faster than cuBLAS**, and it warns of shared-memory limitations on consumer GPUs. It is an example, not a ready FP32 MACE backend. [Triton documentation](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html) |

The closest MLIP work mostly supports **operator and neighborhood fusion**, rather than one kernel for a complete force evaluation:

| Engine | Fusion boundary | Evidence and limitations |
|---|---|---|
| **cuEquivariance** | Segmented tensor products and symmetric contractions; specialized layouts and fused constituent operations. | NVIDIA's original H100 examples show symmetric-contraction improvements of **17× forward / 9× backward**; that comparison includes 10,000-input FP32 tests. Already part of your baseline, and not whole-network fusion. [NVIDIA Technical Blog](https://developer.nvidia.com/blog/accelerate-drug-and-material-discovery-with-new-math-library-nvidia-cuequivariance/) |
| **OpenEquivariance** | Generated sparse tensor products fused with gather/graph aggregation; corresponding backward kernels. Uses atomic or sorted deterministic aggregation. | The inspected paper benchmarks A100, including 50,000 tensor products, and reports up to **4.8× forward / 3× backward** for fused kernels against its early cuEq baseline. It used cuEq **0.0.2**, so these ratios do not apply to 0.12. [arxiv.org](https://arxiv.org/html/2501.13986v2) |
| **FlashTP** | Sparsity-aware, path-aggregated tensor-product fusion. | Reports peak kernel improvements **41.6× over e3nn / 60.8× over cuEq**, and **4.2× inference / 3.5× training** for SevenNet-l3i5; the reported inference example includes 4,000 atoms. These are historical architecture/backend comparisons, not incremental gains over your hybrid. [proceedings.mlr.press](https://proceedings.mlr.press/v267/lee25l.html) |
| **NequIP / Allegro** | PyTorch compilation plus custom fused Allegro tensor-product kernels and accelerator integrations. | Reports roughly **5–18× MD acceleration** over previous execution paths; distributed inference averages include 6.6×/6.9× on Frontier/Perlmutter. Strong evidence for compiler-plus-local-kernel work, but a different architecture and baseline. [pubs.rsc.org](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00423c) |
| **Symmetrix** | Alternative MACE execution through C++/Kokkos, with extracted model data. | Supports CUDA and CPU execution; the inspected documentation does not establish a complete persistent force kernel or a comparable small-system speedup. [GitHub](https://github.com/wcwitt/symmetrix/blob/main/symmetrix/README.md) |
| **Symmetrix-XL** | Checkpoint-specialized compilation, streamed edges and reusable tiled workspace. | October 2026 preprint claims **3.1–5.0× complete LAMMPS-step improvement** over ML-IAP+cuEq across A100/RTX 5090 workloads, with capacity demonstrated into millions of atoms. Relevant to avoiding edge materialization; not evidence for your 216-atom regime or numerical gate. [arxiv.org](https://arxiv.org/abs/2610.01036) |

**Evidence gap:** I found no verified demonstration combining complete MACE energy-plus-position-backward execution, FP32 round-off agreement, 200–2,000 atoms, and both sm_89/sm_120. Published headline ratios should not be multiplied together.

**2. What FastMACE already covers**

Several commonly proposed alternatives are already implemented:

- `rewrites/linear/modules.py`: per-irrep linears, channel grouping and merged message/residual products.
- `kernels/triton/fused_radial.py`: fused radial hidden block, with a recomputing input-gradient kernel.
- `kernels/triton/skinny_linear.py`: folded radial input gradient addressing the awkward 65-column output.
- `rewrites/generic/__init__.py`: receiver-streaming convolutions and capability-checked reassociation.

Their availability does not prove every rewrite was enabled in the supplied profile. Record the resolved configuration before claiming an additional improvement.

The largest implementation obstacle is **composability**: a megakernel cannot simply invoke the existing cuBLAS, cuEq and OEq host APIs as device functions. Their work must become device-callable task bodies. Opaque proprietary kernels are particularly problematic. That replacement effort may exceed the scheduler effort.

**3. A viable task-graph design**

The following is a proposed design, not an existing FastMACE capability.

**Represent the differentiated computation.** Capture an energy function with frozen parameters, differentiate with respect to positions, then lower the resulting forward/backward graph. AOTAutograd provides FX representations of differentiated operations; custom operations still require decompositions or registered lowering rules. A raw Dynamo forward graph alone is insufficient. [PyTorch](https://docs.pytorch.org/get-started/pytorch-2.0/)

Preserve semantic operators during lowering:

| Task family | Partitioning | Required completion condition |
|---|---|---|
| Periodic geometry, radial basis, harmonics | Edge tiles | Geometry for consumed edges ready |
| Atom linears | Atom × output-channel × irrep tiles | Relevant input-channel tiles ready |
| Radial MLP | Edge × channel tiles | Previous MLP stage ready |
| Tensor-product convolution | Receiver neighborhoods × channel/path groups | Sender features and edge quantities ready |
| Symmetric contraction/readout | Atom × channel/path groups | Required local features ready |
| Backward operators | Corresponding adjoint tiles | All contributing downstream adjoints accumulated |
| Final forces | Atom or endpoint-indexed edge reductions | Every incident-edge contribution ready |

Do not assume the effective GEMM row count always equals the atom count: irrep layouts can fold components into rows or introduce strided batches. Export actual dimensions and strides.

**Start with coarse dependencies.** Use phase-level barriers initially, then introduce receiver-tile readiness where useful. Fine-grained dependency metadata should be proportional to task count and graph adjacency, not every possible producer–consumer pair.

Neighbor connectivity changes after rebuilding or compacting. Maintain runtime CSR-style receiver metadata and reverse adjacency. Update index metadata when connectivity changes; do not regenerate model code every MD step.

**Use resident workers with ready queues.** Launch a residency-safe worker grid determined from compiled resource usage and device properties. Each worker:

1. Claims a ready task.
2. Executes it.
3. Publishes output with device-scope release semantics.
4. Updates dependency counters; newly satisfied consumers become ready.

Consumers need matching acquire semantics. Plain `volatile` flags are insufficient. Use replay epochs or correctly ordered resets so an old counter cannot satisfy a new evaluation.

**Avoid deadlock:** workers should claim ready work, rather than occupy the GPU spinning on unavailable producers. Do not assume CTA identity equals physical SM identity. Use cooperative launch only if the selected design actually needs its guarantees.

**Treat shared memory as local.** Direct register/shared-memory handoff requires producer and consumer execution within the same persistent CTA, or another explicitly supported sharing mechanism. Arbitrarily scheduled tasks communicate through global addresses, hopefully serviced by L2.

Consequently, the design needs two optimizations:

- **Local fusion:** remove intermediate writes within a task.
- **Persistent scheduling:** overlap or shorten transitions between tasks.

A scheduler alone delivers only the second.

**Make reductions explicit.** For forward message passing, prefer one owner per receiver/channel tile, iterating over its neighbors. For high-degree receivers, create partial sums followed by a fixed reduction tree.

Backward sender-feature accumulation needs reverse adjacency or another reduction. Position forces also combine contributions from both endpoints. Atomic accumulation offers flexibility but changes order and may introduce nondeterminism; ordered partial buffers offer reproducibility at extra cost.

For edge vector r_e = x_j + S_e H − x_i, with g_e = ∂E/∂r_e, add +g_e to F_i and −g_e to F_j. Include every position-dependent energy branch and preserve the current padding/cutoff behavior.

**Prune the backward correctly.** Frozen model parameters need no parameter gradients. However, radial networks produce edge-dependent weights, whose adjoints remain necessary to obtain forces. Remove parameter-gradient work, not those intermediates' derivatives.

Seed atom-energy readout adjoints with one; the position backward need not wait for a globally reduced energy scalar. Save expensive activations and selectively recompute cheap quantities after measuring the tradeoff.

**Generate from capabilities.** Extract irreps, contraction descriptors, normalization, activation, head selection, strides and derivative rules at load time. Choose tiles through resource-based formulas. Unsupported operators require the existing engine as fallback.

A literal guarantee that *every* future MACE-family model executes within one kernel is not realistic without maintaining a complete operator backend. General functionality is achievable through capability checks and fallback; universal megakernel coverage is a much larger commitment.

**4. Tensor cores: a separate, potentially stronger experiment**

**3×TF32 is worth testing independently of persistence.** CUTLASS implements AB ≈ A_h B_h + A_h B_l + A_l B_h, discarding the residual–residual product. Triton exposes `input_precision="tf32x3"`; its existence does not establish a force-accuracy guarantee. [GitHub](https://github.com/NVIDIA/cutlass/blob/main/examples/27_ampere_3xtf32_fast_accurate_tensorop_gemm/27_ampere_3xtf32_fast_accurate_tensorop_gemm.cu)

For normally scaled values, the omitted product is roughly of order 2^−22 relative to operand-product scale. That is **not** a bound on relative output error under cancellation, nor on propagated force error. Accumulation order and tensor-core arithmetic matter too.

At M = 200–2,000, sufficient output-channel tiles can expose useful parallelism. Small M does not rule tensor cores out; conversion cost, N, K, layouts and tile count decide.

**My estimate:** selected large-channel GEMMs might improve **1.3–2×** with a well-implemented 3×TF32 path; narrow products could regress. No retrieved source establishes this range for your actual shapes.

Test both forward products and their input gradients against FP64, then gate complete forces. Explicitly control Triton precision: PyTorch's TF32 setting is not a substitute for inspecting custom-kernel arithmetic.

**cuBLAS BF16x9 is unavailable on both GPUs.** Current documentation restricts it to compute capabilities **10.0 and 10.3**, excluding **8.9 and 12.0**. NVIDIA also explains that profitable execution needs sufficient BF16/FP32 throughput ratio and efficient scaling support. A custom BF16 decomposition is possible in principle, but is not equivalent to enabling cuBLAS BF16x9. [cuBLAS 13.4 documentation](https://docs.nvidia.com/cuda/cublas/index.html)

Keep the existing FP64 route intact. Neither split-precision scheme preserves its present bitwise results.

**5. What the supplied timings permit us to conclude**

Your baseline is T_step = 3.8 ms and the sum of force kernels = 3.3 ms.

The **0.5 ms difference is not measured launch overhead**. It also contains integration, compaction, synchronization and other work; overlapping kernels would further complicate subtraction.

Conditional Amdahl calculations:

| Hypothetical improvement | New step time | Speedup |
|---|---:|---:|
| Atom GEMMs become 2× faster | 2.97 ms | 1.28× |
| Atom GEMMs become 3× faster | 2.69 ms | 1.41× |
| Atom GEMM time disappears entirely | 2.14 ms | 1.78× |
| All 0.41 ms of elementwise time disappears | 3.39 ms | 1.12× |

These assume the stated buckets are disjoint and other costs stay fixed.

**Kernel durations already include ramp, tail and internal stalls.** Persistence could improve these, but the trace does not quantify how much. Likewise, global-address traffic is not necessarily DRAM traffic: intermediates may already hit L2.

A useful model is T_mega = T_step − G_removed − B_removed − M_avoided + S_scheduler + P_resource_loss.

For illustration only: removing 0.15 ms of gaps and 0.35 ms of internal overhead, while adding 0.15 ms of scheduling/resource penalties, yields **3.45 ms, or 1.10×**.

The modest per-atom scaling improvement across your sizes suggests that fixed launch cost alone is unlikely to explain the bottleneck. It does not distinguish arithmetic, cache traffic, insufficient tiles and register pressure.

**6. Measurements to take before implementation**

**First: a steady-state Nsight Systems trace.** Annotate force evaluation and MD stages. Measure:

- Force-call wall interval versus the **union** of kernel intervals.
- Kernel count, duration distribution and actual inter-kernel gaps.
- CPU-induced GPU starvation, graph replay boundaries and host synchronization.
- Whether costs change during rebuilds or bucket transitions.

**Second: Nsight Compute on the dominant GEMMs, both directions.** Collect dimensions, strides, achieved throughput, `launch__waves_per_multiprocessor`, registers, shared memory, occupancy limits, active/eligible warps, memory traffic and cache hit rates. NVIDIA documents partial waves as a source of tail inefficiency. [NsightCompute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

Interpret the results:

| Observation | Most useful next experiment |
|---|---|
| Few output tiles, poor SM coverage | Smaller tiles, grouping independent products, or split-K |
| High FP32 arithmetic utilization | 3×TF32 experiment |
| Heavy layout/copy traffic | Strided loads and fused layout conversion |
| Significant intermediate traffic | Producer–consumer fusion |
| Register/shared-memory occupancy limit | Smaller working set; avoid broader fusion |
| Large graph execution gaps | Persistent regional scheduling |

**Third: exact-shape microbenchmarks.** Preserve production strides, weight residency, graph replay and backward requirements. Compare individual kernels with the same kernels embedded in the full force call.

Use Nsight Compute for diagnosis and unprofiled CUDA-event/whole-step measurements for acceptance; profiler replay changes execution conditions. Benchmark alternating A/B blocks on both GPUs.

**7. Ranked alternatives**

All estimates below are **incremental whole-step reductions**, conditional on a matching measured bottleneck. They overlap and must not be added.

| Rank | Alternative | Estimated reduction | Qualification |
|---|---|---:|---|
| 1 | Improve atom-linear tiling and remove residual layout copies; fuse normalization/residual epilogues and matching backward work | **5–15%** | Highest-confidence target from your timing split |
| 2 | Optional 3×TF32 for selected atom GEMMs | **10–22%** | Assumes roughly 1.3–2× acceleration of the 1.66 ms bucket; forces must pass |
| 3 | One generated persistent atom block | **3–12%** | Tests task transitions without replacing edge kernels |
| 4 | Further edge-MLP/convolution streaming fusion | **3–10%** | Only after identifying traffic remaining beyond current rewrites |
| 5 | Group independent GEMMs or use deterministic split-K | **0–8%** | Valuable only where tile coverage is poor |
| 6 | Targeted Inductor/Triton backend comparison | **0–8%** | Measure generated kernels and startup cost; no assumed win |
| 7 | CUDA-graph plumbing changes | **0–3%** | Existing capture already removes much host dispatch work |

**Epilogues:** start with an atom linear plus operations consuming exactly its output tile. Include backward savings. Retaining all preactivations may erase forward traffic savings.

**Grouping:** group independent irrep products or branches whose inputs are already available. Sequential interaction layers cannot be grouped as independent GEMMs. FastMACE's existing channel grouping should be the baseline.

**Inductor:** test explicit IEEE Triton GEMMs and selective fusion first. If the prohibition concerns tuning *tables*, one-time load-time algorithm selection may be acceptable; if all empirical tuning is excluded, use resource-based heuristics. Keep Inductor graph ownership consistent with the outer capture.

**Device-callable GEMM:** cuBLASDx deserves consideration for a CUDA prototype. NVIDIA supplies block-level GEMMs and a two-dependent-GEMM fusion example, with documented sm_89/sm_120 resource cases. It still requires benchmarking against your SIMT path. [cuBLASDx](https://docs.nvidia.com/cuda/cublasdx/examples.html)

PDL remains excluded as requested.

**8. Prototype milestones and stopping rules**

| Milestone | Deliverable | Decision |
|---|---|---|
| **A. Establish the opportunity** | Shape/stride inventory, forward/backward attribution, gap measurements, roofline analysis on both GPUs | Stop persistence work if removable transitions are small and local kernels dominate |
| **B. Improve one atom block** | Generated FP32 linear plus epilogue and backward; conventional kernel launches | Seek **≥5% whole-step improvement** without force-gate regression |
| **C. Add regional persistence** | Same arithmetic, tiled dependencies, counters and reusable workspace | Require improvement over milestone B, isolating persistence's contribution |
| **D. Expand to one interaction** | Neighborhood reductions, reverse adjacency, contraction and their adjoints | Require **≥10% whole-step improvement** on both GPUs before full-model expansion |
| **E. Broaden coverage** | Additional models/compositions, bucket/rebuild tests, replay and restart qualification | Promote only supported, validated configurations |

Correctness tests should include thermal/distorted structures, mixed species, cutoff crossings, padded edges, repeated graph replay, translation/rotation checks, force-versus-energy finite differences and comparison with FP64. Evaluate fixed snapshots; chaotic trajectory divergence is not itself a numerical failure.

The main risks are changed reduction order, stale readiness counters, compiler-induced arithmetic changes, register spills, shared-memory constraints and growing operator coverage. sm_120 must not be treated as an interchangeable sm_100: dispatch on supported features, with a conservative common implementation and the existing engine as fallback.

**The project worth attempting is a generated fused atom-block backend with a measurable persistence experiment.** A complete force megakernel should follow only if that experiment demonstrates benefits beyond the cheaper fused kernels.
