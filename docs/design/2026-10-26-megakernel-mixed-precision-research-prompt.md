# Research prompt: further megakernel and mixed-precision opportunities for FastMACE (26 October 2026)

Paste everything below the line into a research-capable model with web access.

---

# Task

Research whether FastMACE can still become materially faster through (a) megakernel /
persistent-kernel execution, (b) operator fusion that removes memory traffic, or
(c) mixed-precision arithmetic that remains FP32-accurate on real MACE operands.

Do **not** re-propose directions already falsified below. Your job is to find what, if
anything, is genuinely new since those experiments, to quantify its ceiling from our
measured step breakdown, and to define cheap falsification experiments with stop rules.
Prefer primary sources (papers, vendor documentation, source code) over blogs and
marketing claims, and say explicitly when a claim does not transfer to our workload.

# Our system and constraints

- FastMACE: pretrained MACE interatomic potentials on one NVIDIA GPU, one structure at a
  time (no batching of structures, no multi-GPU domain decomposition), MD with forces
  from reverse-mode differentiation of the energy. Public repository:
  `FZWINGEL-UBT/FastMACE` (see `docs/rewrites.md`, `docs/validation/opt-5.md`,
  `docs/design/opt-T10-step-breakdown.md`, `docs/design/2026-10-26-fp16x3-and-hybrid-gemm.md`).
- Production baseline (the thing any proposal must beat): hybrid cuEquivariance +
  OpenEquivariance backend, FP32 with TF32 off, `torch.compile` (Inductor), CUDA graphs
  over the MD step, and a stack of adopted rewrites: per-irrep block linears, grouped and
  merged-residual products, gated rows + segment GEMM, species tables and folded linears
  (OPT-T08), receiver-streaming OEq convolution with edge-owned backward, GPU neighbour
  list, fused Langevin integrator stage, pipelined steps.
- Reference GPU: GeForce RTX 5060 Ti (`sm_120`, GB206, 36 SMs, ~448 GB/s DRAM, 180 W
  sustained cap). Secondary GPUs that matter: A100 (`sm_80`), RTX 4000 Ada / L4 (`sm_89`),
  RTX A4000 (`sm_86`). Proposals must be general: no per-crystal, per-GPU, or per-shape
  lookup tables or autotuning caches; a compact rule from M, N, K, and SM count is allowed.
- Accuracy contract: FP32 production. A candidate is admissible only if, on **real MACE
  force-call operands**, its cancellation-scaled error
  |c − c64| / Σ|a·b| stays within ~2× of cuBLAS IEEE FP32, and it can plausibly pass a
  force/stress gate against FP64 (production error at most 2× the FP32 ASE calculator's,
  with floors 0.01 meV/atom, 1e-5 eV/Å, 1e-7 eV/Å³). Synthetic Gaussian operands alone
  are not evidence. Deterministic (atomics-free or fixed-order) execution is strongly
  preferred; anything non-deterministic must be flagged.
- Timing methodology that counts: same-process alternating A/B blocks on an idle GPU,
  CUDA-graph replays for kernels, whole-MD-step medians for adoption. Cross-process or
  single-run timings scatter ~3%. An adoption candidate needs a consistent gain well
  above ~1% on every target crystal size, not a win on one shape.

# Where the step goes (MH-1, RTX 5060 Ti, current defaults)

Force-call shares are roughly: atom-level GEMMs 27–48% (NaNbO3/NbCl5/NaCl ordering
varies), radial networks 18–33%, streaming convolution 16–25%, atom elementwise
7–13%, symmetric contraction 3–5%, edge geometry 2–4%. CUDA-graph gaps inside a force
replay are ~0.10–0.15 ms (about 0.5% on NbCl5, up to ~5% on the small NaCl cell).
Steps: NaCl (216 atoms) ~3.0 ms, NaNbO3 (1,080) ~18–20 ms, NbCl5 (1,944) ~21–23 ms.
cuBLAS IEEE FP32 sustains ~13.7–16.7 TFLOPS on the large GEMM shapes at the 180 W cap
(FMA-pipe activity ~70%); DRAM-bound edge kernels run at 380–430 GB/s of the ~448 GB/s
peak. Both compute and bandwidth are close to saturated; there is no power headroom.

# Already falsified — do not re-propose without genuinely new evidence

1. **Hybrid SIMT+Tensor-Core GEMMs on `sm_120`** (FP16×3 or TF32×3 beside FP32 FMA,
   coupled/decoupled, concurrent two-kernel splits, pre-split operands, larger tiles,
   `setmaxnreg`, subgroup-barrier decoupling): all slower than cuBLAS; issue activity
   falls 76.6% → 58–62%, barrier stalls rise ~10×, unnormalized variants fail
   real-operand cancellation/range tests.
2. **Standalone Triton `3×FP16`** (row/col-normalized, best of fixed tiles): accurate
   (worst cancellation error 0.63–0.78× IEEE) but slower than cuBLAS on every tested
   GPU — `sm_120`, A100 `sm_80` (1.12–1.69×), RTX A4000 `sm_86` (1.43–1.62×),
   RTX 4000 Ada `sm_89` (1.33–1.71×) weighted per force call.
3. **Triton `3×TF32` / BF16 decompositions** as production GEMMs on `sm_89`/`sm_120`:
   arithmetic ceiling below SIMT FP32 (TF32 is 1× FP32 rate on `sm_120`); slower on
   89–92 of 92 production shapes. (CUTLASS 3×TF32 on A100 was never built; Triton's
   version was slower in untimed A100 steps.)
4. **GEMMul8 / Ozaki-II INT8 modular GEMM** (RIKEN GEMMul8, 6–8 moduli, `sm_120`):
   1.5–4.2× slower than cuBLAS even in a flawed screen whose operand reuse favoured it;
   FP64 accuracy never established.
5. **cuBLASLt**: PyTorch's `preferred_blas_library("cublaslt")` in full MD steps is
   mixed (~±1%, block ratios straddling 1). A custom shape-keyed tile dispatcher won
   isolated synthetic shapes but is unqualified and violates our generality rule.
6. **cuBLASDx** (MathDx 25.6, SM120 descriptor using the SM89 config): on its own
   64×64×64 block-tile problem it reaches 7.18 TFLOPS vs cuBLAS batched 7.63 TFLOPS
   (1.062× slower). It is a block-level, compile-time-sized API; whole production
   GEMMs would still need our own grid tiling and predication.
7. **Two-stream concurrency of independent branches** (atom GEMM vs edge/radial
   workload in CUDA-graph replays): small case 1.151× slower, NbCl5-like 0.995×,
   NaNbO3-like 0.985×. No overlap headroom on large cells. Earlier graph-branch and
   two-stream GEMM concurrency: 0.68–1.11× and 0.77–0.85×.
8. **Receiver-owned streaming-convolution backward**: slower in controlled MD A/Bs at
   8/216/1080/1944 atoms (+4.5 to +10.6%). The edge-owned backward is within
   0.05–0.45 ms of a bare weights-copy floor.
9. **Fused radial last layer + streaming convolution** (generate tensor-product
   weights in-kernel, never materialize [E,1280]): failed twice — in-kernel IEEE-FP32
   weight generation reaches ~18% of FMA peak; the backward still needs the weight
   gradient, so forward fusion alone does not remove the traffic.
10. **Gate/residual fusions**: gate inside the GEMM prologue (+1.6–2.0% full step);
    standalone fused gate VJP ceiling re-measured at 0.6–1.1% with kernels already at
    DRAM bandwidth; Inductor combo kernels −0.9–1.8% on NaCl but +1.9–2.7% on NbCl5.
11. **Reduced precision by stage** (TF32/FP16/BF16 storage or compute in radial basis,
    radial MLP, neighbour sums, weights): force-gate failures of 2–8,000×. Radial
    stages are the most precision-sensitive part of the model.
12. **Delta GEMM across MD steps and channel-magnitude splits**: real features change
    ~21% per 1 fs step; errors 400–6,000× IEEE on real operands.
13. **Scheduling-only ideas**: speculative combined graphs (−4 to −15%), activation
    recomputation (+0.4–5.9 ms), GEMM autotuning (−0.1 to −1.0% with 141–756 s setup),
    in-place batched block linears (+0.1–0.3 ms/step), Morton/species atom order,
    skin/bucket retunes, programmatic dependent launch (wrong forces — never use).
14. **Symmetric-contraction replacements**: OEq contraction 27–79× slower; cuEq
    kernels at 72–96% of memory speed-of-light; best theoretical fusion ≤2.7% on
    NbCl5 and not buildable as a single GEMM epilogue.

# What is still open (investigate these, and anything genuinely new)

- Whether any **published or vendor-supported** technique since October 2026 changes
  items 1–14 — new cuBLAS/CUTLASS/cuBLASDx capabilities on `sm_120`, new Triton or
  cuEquivariance/OpenEquivariance kernels, new MLIP engines with MACE forward **and**
  backward at 200–2,000 atoms — with numbers, not roadmaps.
- **Mixed precision with a rigorous error story**: adaptive/certified product counts,
  exponent-aware decompositions, or error-compensated schemes whose cost model survives
  K = N = 512 tall-narrow shapes and whose accuracy is demonstrated on real,
  cancellation-heavy operands. State the exact certificate or bound, its per-tile cost,
  and what fraction of real tile FLOPs must qualify for the cheap path to break even.
- **Traffic elimination that includes the backward**: any fusion whose forward saves a
  buffer but whose backward recomputes or re-materializes it must be charged for both.
  Identify the specific tensors (sizes in the profile above) and count bytes honestly,
  including atomics, saved activations, and layout conversions.
- **Small-cell (≤216 atoms) under-utilization**: l = 0 products run at ~21% of peak
  (0.8 waves). Is there a general, deterministic way to fill the GPU (tile rules,
  split-K with fixed-order reduction, persistent tiles) that survives full-step A/B,
  given that grouped launches and split-K as built did not?
- **Non-MH-1 model families** (generic MACE-MP-0/OFF, OMOL-0, POLAR-1): different
  kernel shares (e.g. OMOL-0 is 78% atom GEMMs; POLAR-1 was launch-bound before its
  fx-graph fix). Flag any opportunity that is family-specific and say so.
- **Cold-start/compile time** is a legitimate secondary target (tens of seconds per
  run start; POLAR cold compilation is slow), but keep it separate from step time.

# Required output

1. **Ranked table** of at most 8 candidates. For each: mechanism; what is new relative
   to the falsified list; expected whole-step gain on NaCl **and** NbCl5 (derive it
   from the shares above, showing the arithmetic); accuracy risk and the exact gate it
   must pass; implementation effort (days/weeks); and the cheapest falsification
   experiment with a numeric stop rule.
2. **A "do not pursue" list**: which of my open questions you consider closed by
   published evidence, and why.
3. **Claim audit**: for every external speedup figure you cite, state the GPU, problem
   size, precision, baseline, and whether forward-only or forward+backward — and
   whether it transfers to 200–2,000-atom FP32 MACE MD on `sm_120`. Discount LLM
   inference results (no backward, different operator mix) explicitly.
4. **The single experiment you would run first**, specified tightly enough to
   implement: operands and shapes, baseline, metric, pass/fail threshold, and what
   decision each outcome triggers.
5. Flag anything that would violate the constraints (batching, per-shape tables,
   non-determinism, accuracy contract) rather than silently assuming it away.

Be quantitative and skeptical. If your honest conclusion is that the remaining
headroom is a few percent in specific niches, say so and identify the niches.
