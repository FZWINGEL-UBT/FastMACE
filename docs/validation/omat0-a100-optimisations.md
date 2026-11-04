# What makes the OMAT-0 A100 profile fast, and where each idea came from (4 November 2026)

**Scope.** Every optimisation active in the fastest recorded OMAT-0 run:
- **Run:** `results/omat0-a100-profile-20261103/` (source `3501a499`, result commit `57b4f31e`).
- **Setup:** MACE-OMAT-0 medium, SrTiO3 10×10×10 (5,000 atoms), float32, hybrid cuEq/OEq backend, Modal
  A100-SXM4-80GB.
- **Protocol:** Benchmark_loup (`tools/bench/loup_md.py`: from rest, skin 0.5 Å, 1 fs, 5 warm-up and 20 timed steps,
  3 fresh processes).

**Result:**
- **NVE:** 3.64 µs/atom/step (3.641–3.645).
- **Langevin:** 3.59 µs/atom/step (3.576–3.606) (`loup/*.json`).
- **Force graph:** 18.43 of the 18.50 ms step (`breakdown/summary.json`).

For comparison, the same benchmark on the A100 80GB PCIe before 3 November gave:
- FastMACE: 5.35 / 5.29 µs/atom/step;
- Symmetrix-XL under LAMMPS: 5.00 ([loup-a100.md:7](loup-a100.md)).

Those numbers are from a different card and commit. They are context, not an A/B.

**How to read the citations.**
- `path:line` is a FastMACE path from the repository root.
- `RM:LT/path:line` is `Experiments/Langevin_Thermostat/path` in ResearchModule. It was read at ResearchModule main
  `d56b5550` (29 October 2026); the research-engine code itself is frozen at tag `research-engine-baseline-20261017`.
- Every quote was checked against its line.

**Origin classes.**

| Class | Meaning |
| --- | --- |
| **RE** | Own idea of a research-engine session (ResearchModule, 7–16 October 2026), carried into FastMACE in phase 1a |
| **FM** | Own idea of a FastMACE session (from 17 October 2026) |
| **SXL** | Taken from Symmetrix-XL ([bonan-group/symmetrix-xl](https://github.com/bonan-group/symmetrix-xl), arXiv 2610.01036), as the source document says |
| **UP** | Upstream code or a paper: MACE, OpenEquivariance, CUTLASS, PyTorch |
| **EXT** | An external research report, review or proposal given to us by the owner |
| **OWNER** | The owner asked for this specific thing |

Several entries have more than one class; the table gives the chain.

## 1. Where the force call's time goes now

From the kernels of one force call, 18.47 ms (`breakdown/ops.csv`, `breakdown/categories.csv`):

| Kernel | ms | Share | Optimisation |
| --- | ---: | ---: | --- |
| `oeq_projected_backward_wprime` (+ 0.22 ms Inductor epilogue) | 6.92 | 37 % | §2.4 `wprime_backward` |
| `fused_folded_forward` | 5.06 | 27 % | §2.3 `fused_folded_forward` |
| CUTLASS `fm_tf32x3_*_64x64`, 44 launches | 1.85 | 10 % | §3.6 `tf32x3_gemms` |
| Atom-level elementwise (79 kernels) | 1.52 | 8 % | Inductor fusion (§4.6) |
| Radial tables, forward + backward | 0.97 | 5 % | §2.1 `radial_tables` |
| Reassociated first interaction (`edge_outer_sum_*`) | 0.75 | 4 % | §3.3 |
| cuEq symmetric contraction | 0.41 | 2 % | §3.1 (hybrid backend) |
| Remaining atom GEMMs on cuBLAS IEEE | 0.06 | 0.3 % | below §3.6's shape rule |

**Outside the force graph:**
- **Cost:** noise, the integrator stage, neighbour rebuilds and the host together take about 0.07 ms per step (wall
  18.50 against the B replay's 18.43 ms; `breakdown/summary.json`).
- **Gaps:** the 178 kernels of the force graph leave 5.7 µs of gaps in total.

## 2. Radial path and convolution weights

**OMAT-0 is a generic-family model.** Its density interaction blocks are not MH-1's class, so MH-1's `factor_species`,
`fold_cutoff`, `joint_radial_input`, `density_first` and `folded_backward_tiles` are off (plan off reasons,
`breakdown/summary.json`). Instead, the generic records below do the same job.

| # | Optimisation (record) | What it does | Exactness | Origin, with evidence | Adopted, with evidence | ms here |
| --- | --- | --- | --- | --- | --- | ---: |
| 2.1 | `radial_tables` (generic record `generic_radial_tables`) | Per interaction and ordered species pair: piecewise Chebyshev tables, fitted in float64 at setup, replace the whole chain r → `[h c, c, density]` (Bessel basis, Agnesi transform, MLP, cutoff). One Triton op with a recomputing backward. | certified (decision 61): table error ≤ the float32 network's | **EXT → OWNER → FM.** The MH-1 tables: "The idea (owner, 27 October 2026, from an external proposal)" (`docs/validation/opt6-T04.md:21`; `plans/opt-6.md:26`). No document names that proposal. The generic version is an FM finding: "The one worth building: radial tables for OMAT-0-type models" (`docs/validation/tiled-omat.md:164`). **Not from SXL:** Symmetrix-XL uses "cubic Hermite splines (256 knots)", but the RE set tabulation aside on 14 October as "already measured" (`RM:LT/studies/symmetrix_xl/README.md:20`, `:26-28`). | MH-1: decision 62, 27 October. Generic on the A100s: owner "Yes go ahead", 3 November (`SPEC.md:3310`). Step b/a 0.902 / 0.901 (A100 PCIe, SrTiO3) and 0.917 / 0.917 (A100-SXM4, NaCl) (`docs/validation/loup-a100.md:46`, `:52`). | 0.97 |
| 2.2 | `fused_folded_backward` (generic `generic_fused_radial_backward`) | Triton input gradient of the last radial layer: the 64 columns and the bias column's row sum in one pass. | rounding | **RE**, third round, 14 October (`RM:LT/studies/further_optimizations/2026-10-14-third-round.md:43`). Generic version on user request, 15 October (`RM:LT/studies/other_models/README.md:269`). | RE 14 October (MH-1 −0.6 to −1.4 %, `…/2026-10-14-third-round.md:61`); generic −3 to −5 % (`docs/optimisation-ledger.md:302`) | **0**: on only as a prerequisite of 2.1; interaction 2 takes the w′ path, interaction 1 is reassociated |
| 2.3 | `fused_folded_forward` (generic `generic_fused_folded_forward`) | Generated from the streaming schedule, an NVRTC kernel computes `w = [h c, c] @ [W, b]ᵀ` per edge tile inside the receiver-streaming convolution forward. It stores `w [E, 1280]` for the backward. | exact | **SXL → FM → OWNER.** The streamed-edges idea is "Origin: the one idea from Symmetrix-XL" (`RM:LT/studies/symmetrix_xl/STREAMED_EDGES.md:7`). FM re-proposed it as "Fuse the radial network's last layer into the streaming convolution" (`docs/design/opt-T09-step-breakdown.md:183`). The rejected OPT2-T01 build is at `docs/optimisation-ledger.md:1726`. Round 8 is an "Owner request, 27 October 2026" (`plans/opt-8.md:3`); the variant that stores `w` was an "Owner follow-up" (`plans/opt-8.md:34`). Sobek's reassociation (arXiv 2607.18074) was considered there and not used (`plans/opt-8.md:22-27`). | MH-1: decision 64, 27 October. OMAT-0 on the A100-SXM4: owner, 3 November, **only as a pair with 2.4**. Alone it is 1.289 / 1.287 (slower); the pair is 0.895 / 0.894, and 0.885 / 0.885 with tf32x3 on in both sides (`docs/validation/generic-wprime.md:58`, `:71`; "never the fused forward alone", `:13`). | 5.06 |
| 2.4 | `wprime_backward` (generic `generic_wprime_backward`) | Within a table piece, `w(r) = Σ_t T_t(x) G_t` with `G = C Wᵀ`, so the streaming backward forms `dr = Σ_c dw_c · w′_c(r)` from derivative tables `G′ = C′ Wᵀ`: 10 × 1280 FMAs per edge instead of 65 × 1280. No `dw` store, no folded input gradient. | rounding | **FM → OWNER.** The idea was first written down as float64 idea D2 in the FM ideas round (`plans/ideas-20261027.md:57`, "`dr = Σ_c dw_c · w'_c(r)`"). The float32, backward-only form: "Owner request, 28 October 2026" (`plans/opt-9.md:3`). Its A100 ceiling triggered decision 66, which allows per-GPU defaults (`SPEC.md:2775`). | MH-1: decision 66, 29 October (A100 NaNbO3 0.815 / 0.812, `docs/validation/opt9-T03.md:28`). OMAT-0: owner "Yes, build it", then default on the A100-SXM4 on 3 November (`docs/validation/generic-wprime.md:3-6`). | 6.92 + 0.22 |
| 2.5 | `chunked_conv_backward` (generic `generic_chunked_conv_backward`) | MH-1 design: the convolution backward runs in L2-sized edge chunks, so `[E, 1280]` `dw` is never formed. | rounding (generic) | **SXL → OWNER → FM.** Step 1 of the Symmetrix-XL streamed-edges plan ("L2-resident edge chunks", `RM:LT/studies/symmetrix_xl/STREAMED_EDGES.md:97-102`). FM round 6: "Owner request, 26 October 2026: … try the L2-chunk idea first" (`plans/opt-6.md:3`). | **On here only as the carrier of 2.4**: "on as the carrier of wprime_backward, whose backward never chunks" (`fastmace/planner/fixed.py:1218`). Chunking itself is slower on the A100 (1.12 with the tables, `docs/validation/loup-a100.md:45`). | **0** (no chunked kernel runs) |

## 3. Convolution, interactions, linear layers and GEMM precision

| # | Optimisation | What it does | Exactness | Origin, with evidence | Adopted, with evidence | ms here |
| --- | --- | --- | --- | --- | --- | ---: |
| 3.1 | Hybrid backend, float32, TF32 off | OpenEquivariance's fused convolution replaces cuEq's gather, product and scatter; cuEq keeps the symmetric contraction and the linears. | backend | **UP (MACE 0.3.16) → RE.** "MACE 0.3.16 can run a model with two accelerated backends at once" (`RM:LT/studies/remaining_optimizations/proposal.md:96`). | User production decision, 9 October 2026 (`RM:LT/docs/status.md:7`); NaCl 9.668 → 7.075 ms, −26.8 % (`RM:LT/studies/hybrid_oeq/rtx5060ti.md:191`). | frame for all kernels |
| 3.2 | `streaming_conv` (generic `generic_streaming`) and the `ir_mul` layout | Receiver-streaming UVU convolution, NVRTC: the forward has one writer per receiver and no atomics. `ir_mul` removes the transposes around it. | exact (summation order; atomics) | **SXL + UP + OWNER, then RE changes.** The streamed-edges idea is from Symmetrix-XL (`…/symmetrix_xl/STREAMED_EDGES.md:7`). The kernels are vendored from OpenEquivariance PR 223 "by Paul Fuchs", which adapts Sobek (arXiv 2607.18074) (`fastmace/kernels/oeq_streaming/NOTICE.md:3-8`). The owner pointed at the PR: "User pointer, 14 October 2026" (`RM:LT/studies/symmetrix_xl/STREAMED_EDGES_RESULTS.md:121`). `ir_mul` is our own change (`NOTICE.md:12-15`). No document says the PR was written for this project. | RE default 14 October (`…/STREAMED_EDGES_RESULTS.md:159`). Generic: user decision, 15 October (`RM:LT/studies/other_models/README.md:242`), omat0-medium −16.0 / −23.9 / −18.6 % (`:219`). | runs inside 2.3 and 2.4 |
| 3.3 | `reassociate_first` (generic `generic_reassociate`) and `segmented_first_sum` | Interaction 1 is summed per (receiver, sender species) before its last radial layer, then projected with node GEMMs. This removes the first convolution and its per-edge weights. The sums run as a run-segmented Triton kernel, one atomic per run. | rounding / exact | **EXT → RE.** The first comes from an external research report answering the RE's 10 October prompt: "move the first interaction's final radial projection after a receiver-and-species aggregation" (`ResearchModule:GPT_6_PRO_RESEARCH_2.txt:1`), assessed in `RM:LT/studies/research_prompts/2026-10-10-hybrid-default-further-optimization-assessment.md:55`. The segmented sum is RE's own third-round candidate (`…/2026-10-14-third-round.md:45`). | User, 10 October ("items 6 and 7", `RM:LT/studies/further_optimizations/README.md:199`): NaCl 4.9 %, NaNbO3 7.2 %, NbCl5 4.9 % (`:55`). Generic: user, 15 October (`…/other_models/README.md:242`). | 0.75 |
| 3.4 | `sort_receivers` | Edges sorted by receiver at each rebuild; streaming (3.2) and the segmented sum (3.3) require it. | exact (permutation) | **RE**, proposal of 8 October: "If the neighbour list were sorted by receiver at each rebuild" (`RM:LT/studies/remaining_optimizations/proposal.md:250`). | RE hybrid default, 10 October; NaNbO3 3.3 %, NbCl5 4.9 % (`RM:LT/studies/further_optimizations/README.md:24`). | in the rebuild |
| 3.5 | `block_linear`, `grouped_linear` | cuEq `Linear` becomes per-irrep GEMMs with extracted weights; same-irrep input segments share one product. | rounding | `block_linear`: **RE**, from an op map showing cuEq's linears "using slower kernels" (`RM:LT/studies/further_optimizations/2026-10-14-third-round.md:100`). `grouped_linear`: **OWNER → RE** ("User request: optimize OMOL-0 further", `RM:LT/studies/other_models/README.md:652`). | `block_linear` 14 October: NaCl −11.8 % (`…/2026-10-14-third-round.md:117`). `grouped_linear`: user, 16 October (`…/other_models/README.md:721`). | GEMMs counted in 3.6 |
| 3.6 | `tf32x3_gemms` on the CUTLASS sm_80 backend | Atom GEMMs (forward and input gradients) as error-compensated 3×TF32 on tensor cores. Kernel: a prebuilt CUTLASS 4.8.1 cubin (`OpMultiplyAddFastF32`, 64×64×16). Shapes outside the rule (K, N multiples of 4; enough tiles or short K) stay on cuBLAS IEEE. No global TF32. | rounding, deterministic | **EXT + UP → OWNER → FM.** An external research report: "**3×TF32 is worth testing independently of persistence.**" (`docs/design/2026-10-20-megakernel-research.md:152`). The split is "the 3xTF32 split of CUTLASS example 27" (`fastmace/kernels/cutlass/NOTICE.md:6`). An external review of main `b6421afc` led to the 29 October screen (`docs/validation/tc-gemm-screen.md:3-4`). Then: "Owner request: \"build it for a100\"" (`docs/validation/cutlass-tf32x3.md:3`). Earlier the RE found Triton TF32x3 slower on an RTX A4000 (`RM:LT/docs/configuration/workstations.md:262`). | Owner, 3 November, decision 66, A100-SXM4-80GB only. OMAT-0 SrTiO3 0.949 / 0.947 (`docs/validation/cutlass-tf32x3.md:86`); 0.909 / 0.907 against Triton tf32x3 (`:88`). | 1.85 |
| 3.7 | `species_skip` (generic `generic_species_skip`) | The element-dependent `skip_tp` over all 89 one-hot elements becomes one GEMM per irrep block for the species present. | exact | **OWNER → RE.** "User request: continue the per-architecture campaign" (`RM:LT/studies/other_models/README.md:269`); "about 97% of this multiplies by zero" (`:282`). | User, 15 October (`…/other_models/README.md:346`); omat0-medium NaCl −32.5 %, NaNbO3 −20.0 %, NbCl5 −38.5 % (`:318`). | GEMMs counted in 3.6 |

**On in the plan but doing nothing in this run:**
- **`merged_residual`:** the plan text lists it, but the record stays off for this model ("this record stays off",
  `results/omat0-a100-profile-20261103/breakdown/summary.json:394`).
- **`single_graph_strain` (MACE PR 1498 backport, `RM:LT/studies/further_optimizations/2026-10-16-upstream-mace-survey.md:27`):**
  it acts only where stress is computed, and a fixed-cell run computes none.
- **`grouped_launch`:** off because "tf32x3_gemms reroutes the block products" (`fastmace/planner/fixed.py:1252-1257`).

## 4. MD engine

| # | Optimisation | What it does | Origin, with evidence | Adopted, with evidence | In this run |
| --- | --- | --- | --- | --- | --- |
| 4.1 | GPU-resident integrator and MD loop | ASE's Langevin step term for term on GPU tensors, with the same NumPy random stream. | **RE**, 7 October. The motivating line: "The larger potential optimization is reducing repeated Python/kernel dispatch" (`RM:LT/docs/protocol.md:482`). | RE 8 October; NaCl 41.4 → 30.0 ms, together with 4.4 (`docs/optimisation-ledger.md:495`) | on |
| 4.2 | Own CUDA graphs (ACP stage and force stage B), padded edge counts, 5 % edge buckets | Each step replays captured graphs. B is captured per padded edge count, and padding edges lie beyond the cutoff. Here there is one B size (373,760) and no recapture. | **UP → RE.** PyTorch's `reduce-overhead` mode, cited at `RM:LT/docs/protocol.md:484`, failed ("Every second step was 25–30× slower", `RM:LT/docs/implementation.md:159-161`), so the engine captures its own graphs. Buckets are an RE idea (`docs/optimisation-ledger.md:498`). | RE 7 October; NaCl 30.0 → 16.5 ms: "This was the single largest step" (`RM:LT/studies/optimization_history/report.md:88`) | on |
| 4.3 | `pipelined` steps | B is launched before the host reads the rebuild flags, and redone if a rebuild was needed. The list is rebuilt early at 0.8 of skin/2. | **RE**, from the measured host gap ("The host gap of section 5 …", `RM:LT/studies/further_optimizations/2026-10-14-third-round.md:229`) | RE 14 October; NaCl −3.0 % (`…/2026-10-14-third-round.md:255`) | on |
| 4.4 | Skin neighbour list built on the GPU (Triton pair search) | Pairs within cutoff + skin, rebuilt only after a displacement of skin/2, with float64 pair decisions. Each step compacts to the true cutoff. | Skin list: **RE** (standard Verlet list, 7 October). GPU build: **RE**, first raised in the RE's research prompt of 9 October, which names `torch-nl`, `vesin`, `nvalchemi` and Warp (`RM:LT/studies/research_prompts/2026-10-09-hybrid-fp32-optimization-research.md:348-349`). The kernel is our own Triton. | GPU build, RE 14 October: step −1.6 to −4.6 % (`…/2026-10-14-third-round.md:64`) | on; skin 0.5 Å is the benchmark's (`loup-a100.md:5`, the same as Symmetrix-XL's default, `RM:LT/studies/symmetrix_xl/README.md:38`); the library default is 0.3 Å |
| 4.5 | `overlap_rebuild` | An early rebuild runs on a side stream beside B. | **FM**, OPT2-T04 ("An early rebuild does not …", `docs/validation/opt2-T04.md:21`) | Owner, OPT-DEF-2, 21 October (`docs/optimisation-ledger.md:510`) | on; few or no rebuilds in a 25-step run from rest (inference, not recorded) |
| 4.6 | torch.compile mode `default` inside the outer graphs | Inductor fuses the elementwise chains; no autotuning. | **RE**, kernel-fusion study, "Idea 1" (`RM:LT/studies/kernel_fusion/report.md:1`) | RE 8 October; coraline NaCl 17.57 → 11.13 ms (`docs/optimisation-ledger.md:532`) | on |
| 4.7 | `fused_integrator` (Langevin only) | The Langevin ACP stage as four Triton kernels instead of about 45 PyTorch kernels. | **OWNER → FM**: "Go ahead and try all kernel opportunities" (`plans/opt-3.md:4-5`) | Owner, 21 October; NaCl 0.976 (`docs/optimisation-ledger.md:512`) | Langevin on (0.08 ms per step); not in NVE (`fastmace/kernels/triton/fused_integrator.py:211`) |
| 4.8 | 256-step noise batches | One NumPy draw and one copy per 256 steps, in ASE's order. | **RE**, from a profile where "the GPU waited for the CPU" (`RM:LT/studies/optimization_history/report.md:149`) | RE 7 October; −5 to −8 % (`docs/optimisation-ledger.md:500`) | Langevin only |
| 4.9 | `unfuse_full_addmm` | Stops torch 2.14's Inductor from fusing mm + full add into `addmm`, which cost two device copies per step. | **RE** diagnosis on coraline (`docs/optimisation-ledger.md:242`) | RE 8 October; local −3.7 % | installed; not measured on OMAT-0 |
| 4.10 | Software stack: torch 2.14.1, cuEq 0.12.0 | A newer stack. | **RE**: newer torch was needed for Blackwell and was also faster: 12.6 → 10.9 → 10.5 ms (`RM:LT/studies/optimization_history/report.md:116-128`) | User, 8 October (`docs/optimisation-ledger.md:533`) | on |

**Memory and setup only, no step-time effect here:**
- `expandable_segments`, `warmup_in_graph_pool`, `empty_cache_before_capture`, `dynamic_edges` and
  `model_template_cache` (`docs/optimisation-ledger.md:502-503`, `:537`, `:540`).
- `async_output`: the benchmark writes no output.
- `first_derivative_only`: not installed on the hybrid backend (`fastmace/model/compile.py:213`).

## 5. What came from where, in one view

| Source | Optimisations active here |
| --- | --- |
| **Symmetrix-XL** (via the RE's 14 October study) | The streamed-edges idea behind `streaming_conv`, `fused_folded_forward` and the L2-chunk design (3.2, 2.3, 2.5). The skin of 0.5 Å is the benchmark's and matches Symmetrix-XL's default. Not the radial tables (2.1). |
| **Upstream code and papers** | OEq PR 223 by Paul Fuchs, after Sobek (3.2); CUTLASS 4.8.1, `OpMultiplyAddFastF32` and example 27 (3.6); MACE 0.3.16's hybrid conversion (3.1); PyTorch's CUDA-graph and compile modes (4.2, 4.6) |
| **External reports, reviews, proposals** | The GPT-6 Pro report: reassociation (3.3). The 20 October external research report: 3×TF32 (3.6). The 29 October external review: the CUTLASS screen (3.6). The 27 October external proposal, author not recorded: radial tables (2.1). The RE research prompt naming GPU neighbour-list libraries (4.4). |
| **Owner requests** | Generic streaming (3.2), `species_skip` (3.7), `grouped_linear` (3.5), round 6 L2 chunks (2.5), round 8 fused forward (2.3), the float32 w′ backward (2.4), CUTLASS "build it for a100" (3.6), round 3 kernels (4.7); every default by owner decision |
| **Own work (RE)** | Hybrid adoption, `sort_receivers`, the segmented sum, `block_linear`, `fused_folded_backward`, GPU integrator, own CUDA graphs and buckets, pipelined steps, GPU neighbour list, noise batches, the compile setup, `unfuse_full_addmm` |
| **Own work (FM)** | The table op and certificate (2.1), the generic carry-overs (2.1–2.4), the fused forward kernel (2.3), the w′ derivation (2.4), the carrier rule (2.5), the CUTLASS integration and shape rule (3.6), `overlap_rebuild` (4.5), `fused_integrator` (4.7) |

## 6. Gaps and contradictions in the records (not fixed here)

1. **`chunked_conv_backward` on the A100.**
   - `docs/validation/loup-a100.md:100` and `SPEC.md:3310` say it "stays off on the A100".
   - The planner turns it on there as wprime's carrier (`fastmace/planner/fixed.py:1218`). It runs no kernel, but
     those two documents do not mention this.
2. **SPEC has no record of the 3 November OMAT-0 defaults.**
   - Nothing for `tf32x3_gemms` or CUTLASS on the A100, and nothing for the generic fused forward + wprime pair.
   - Decision 66's list (`SPEC.md:2775-2808`) stops at the generic chunked backward.
   - The §1.2 non-goal on TF32 (`SPEC.md:75`) is not amended for 3×TF32.
   - These are recorded only in [cutlass-tf32x3.md](cutlass-tf32x3.md), [generic-wprime.md](generic-wprime.md) and
     the commits.
3. **The optimisation ledger is stale for this profile.**
   - It has no `wprime_backward` entries.
   - It shows `tf32x3_gemms` as `never_automatic` and CUTLASS on the A100 as not built
     (`docs/optimisation-ledger.md:165`, `:414`).
4. **Missing results directory.** `docs/validation/generic-wprime.md:12` cites `results/a100-chunk-size-20261103`, which
   is in no branch.
5. **The radial-table proposal's author is not named** anywhere (`docs/validation/opt6-T04.md:21`).
6. **Two Symmetrix-XL URLs.**
   - The RE study names `bonan-group/symmetrix-xl` (`RM:LT/studies/symmetrix_xl/README.md:13`).
   - `docs/design/2026-10-positioning.md:27` names `amphilagus/symmetrix-xl`.
7. **The NVE/Langevin gap is not explained by a measurement.**
   - NVE has neither the fused integrator nor noise batching.
   - No NVE step breakdown exists. The breakdown in this profile is Langevin, skin 0.3 Å, so its rebuild counts do not
     describe the 0.5 Å benchmark runs.
