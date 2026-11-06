# Five further H200 experiments, and an audit of the bounds

6 November 2026. CPU-only research; no GPU execution, remote jobs, test suites, commits or pushes.

**Do not budget for 2× from this shortlist.** There is a credible new way to reduce the w′ kernel's register footprint, a reason to reopen one incorrectly closed locality question, and three smaller experiments. None has a measured speedup. A successful, non-overlapping selection might save approximately **0.1–0.30 ms** against the nominal 5.40 ms step, about **1.02–1.06×**. This is an engineering estimate, not a confidence interval or an additive promise. The known idea 30 and remaining idea-31 epilogues remain important; they are not counted as new here. Fresh results for ideas 30, 31, 33 and 36 arrived during this review and are incorporated below.

Two qualifications materially affect the target:

* The cited sx012 H200 profile explicitly has **`compute_stress: false`**. It measures a fixed-cell force graph. I could not verify a 5.40 ms **force-and-stress** baseline from these artifacts. All proposals below retain the stress derivative; none gets credit for omitting stress.
* The recorded round-12 H200 NbCl5 stress ratios are **2.1045 for default and 2.0634 for CG_FOLD**, both marked `passed: false`. “Does not worsen the existing exception” is insufficient for this brief's strict 2× gate. Every rounding-class proposal below remains unqualified until it passes that gate. There is no demonstrated error budget for reducing product precision.

## Scope and evidence conventions

I read Revision 14, the timing and ranking sections, all closed/rejected sections and all of “Considered and dropped,” including the newly added **idea 36**. I also read the specified kernels, the CUTLASS implementation, SPEC's exactness/accuracy/planner rules, the cited breakdowns and the relevant round artifacts. Neither checkout has `.codegraph/`.

Source inspection used `D:\Margraf\FastMACE`, initially at `31c394b45d29fe12759e55d4cbfd9351011559b1`, then refreshed against concurrent commit `5dc5d79914a93b10d4cc237f3729da8815a1111d`. The ideas document was last checked at worktree commit `fcc289e200fbcecc4cf541bf9d57800efd132a7c` (SHA-256 `9ce258d7dfe5ade5af31cb8898c14067897d74572c74afaf4c79f79ce1c73a6e`). Historical results are identified by their own run/commit; they are not treated as a single current profile. The local compilation experiments use the **saved round-9 H200 w′ render**, commit `ab76a9d5`, with FACTORED, TABLE_W, HORNER and SCALAR_PATHS, and the ordinary shuffle reduction. They do not regenerate or run the model. The concurrent changes add GEMM epilogues/results; they do not change this w′ source.

Evidence labels used below:

| Label | Source and what it establishes |
| --- | --- |
| P | [Ideas plan](D:/Margraf/FastMACE/.claude/worktrees/h200-ideas-20261105/plans/h200-ideas-20261105.md), sections named explicitly below. In particular, the nominal 5.40 ms step and approximately 1.49 ms forward are the plan's moving baseline. |
| B | [sx012 ops.csv](D:/Margraf/FastMACE/results/sx012-profile-20261106/fastmace-h200-breakdown/breakdown/ops.csv), [kernel_sequence.csv](D:/Margraf/FastMACE/results/sx012-profile-20261106/fastmace-h200-breakdown/breakdown/kernel_sequence.csv), and [summary.json](D:/Margraf/FastMACE/results/sx012-profile-20261106/fastmace-h200-breakdown/breakdown/summary.json), commit `d599cc70`. Shapes, individual operation times and the stress-off configuration. |
| W | [sx012 README, BUTTERFLY recheck](D:/Margraf/FastMACE/results/sx012-profile-20261106/README.md). w′ without BUTTERFLY: 1,612.4 / 1,613.4 µs; the original B profile's 1,825.7 µs w′ has BUTTERFLY on. Use approximately 1,612 µs for current-scale estimates, not 1,825.7. |
| L | [Fusion-bound README](D:/Margraf/FastMACE/results/fusion-bound-20261106/README.md), [H200 bounds JSON](D:/Margraf/FastMACE/results/fusion-bound-20261106/runs/h200/lab-wprime/lab.json), [real staging JSON](D:/Margraf/FastMACE/results/fusion-bound-20261106/runs/h200-stage/lab-wprime/lab.json). Both the promising artificial bound and the failed real implementations. |
| M | [Mirror experiment JSON](D:/Margraf/FastMACE/results/research-answers-20261105/runs/mirror-h200/lab-wprime/lab.json), with context in [research-answers README](D:/Margraf/FastMACE/results/research-answers-20261105/README.md). Base 1,563.872 µs; `F-no-table-horner` 1,152.128 µs, ratio 0.736715. |
| R | [Round-2 locality JSON](D:/Margraf/FastMACE/results/research-answers-20261105/runs/ideas-r2-h200/lab-wprime/lab.json) and its [saved wprime.cu](D:/Margraf/FastMACE/results/research-answers-20261105/runs/ideas-r2-h200/lab-wprime/wprime.cu). The alleged one-row experiment and its actual addressing. |
| C | CPU-only NVRTC/SASS experiments described in the compilation appendix. These establish compiler resource counts and source transformations, **not correctness or GPU performance**. |
| U | Concurrent updates: [research-answers README, Revision 14](D:/Margraf/FastMACE/results/research-answers-20261105/README.md), [idea-30 full-forward report](D:/Margraf/fm-idea30/results/h200-ideas-20261105/idea30-full.md), [idea-31 report](D:/Margraf/FastMACE/results/h200-ideas-20261105/idea31-glue.md), [idea-33 bound](D:/Margraf/fm-idea33/results/h200-ideas-20261105/idea33-bound.md). Read-only inspection of other worktrees; no jobs were launched by this review. |

Unless stated otherwise, MB/GB here are decimal. `E = 373,760` is the **padded edge extent** in B and the kernel labs, not a verified count of physically contributing neighbors. The 4.8 TB/s bandwidth in the brief is used only for ideal transfer-time arithmetic. A byte count divided by bandwidth is not, by itself, a measured saving or a rigorous ceiling.

## 1. Give each spherical-harmonic degree its own w′ warp, then combine dx on chip

**Mechanism and files.** In [wprime_backward.py](D:/Margraf/FastMACE/fastmace/kernels/oeq_streaming/wprime_backward.py), partition the generated paths by the degree of the **edge spherical harmonic**, rather than assigning every path to the same warp. In the saved OMAT-0 render the ten paths have edge degrees

`0, 1, 1, 0, 2, 2, 1, 3, 3, 2`.

Thus four warps cooperate on one edge, owning respectively 2, 3, 3 and 2 paths. Each still processes all 128 channels in four 32-channel passes. A warp needs only its degree's 1, 3, 5 or 7 `edge_gradient` components, rather than all 16.

Keep each degree's `dx[512]` partial in shared memory after its channel work, and one reduced `dr` scalar per degree. After one block barrier, combine the four dx partials and issue the original total of 512 atomic additions per edge. Write each degree's disjoint dsh components directly. Combine the four dr scalars on chip. Do **not** emit four dx arrays to global memory. Rotate the degree-to-warp assignment using a fixed hash of the edge/block ID to avoid tying the uneven degree workloads to one warp position. This is a model-schedule transformation, not a structure-specific rule.

**Target.** The approximately **1,612 µs w′** (W); not the 1.49 ms forward. The saved round-9 lab itself measured 1,632.416 µs, with 96 registers [round-9 JSON](D:/Margraf/FastMACE/results/research-answers-20261105/runs/round9/wp9-h200/lab-wprime/lab.json).

**CPU evidence and ceiling.** The separate degree bodies compiled at **48 / 64 / 64 / 64 registers**. More usefully, a combined four-warp prototype, including shared dx partials, dr reduction, final dx atomics and the barrier, compiled at **64 registers, 9,232 bytes shared memory, zero stack/local memory**. The hashed assignment also compiled at 64 registers. This is not just four isolated register counts assumed to compose.

For 128-thread blocks, the 65,536-register SM budget permits `floor(65536 / (128 × 96)) = 5` baseline blocks, versus 8 at 64 registers: **20 → 32 resident warps**, if other limits permit. The compiled shared-memory footprint does permit eight blocks. The register/warp limits are documented in NVIDIA's [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html).

A deliberately optimistic, latency-only occupancy model gives

`1,612 × (1 − 20/32) = 604.5 µs`.

That is a **conditional ceiling**, not a measured bound: it assumes useful throughput scales linearly with resident warps. Each edge now uses four warps, duplicates edge setup, and incurs a shared reduction. Uneven path work makes the barrier costly. The static combined body is also larger: 2,133 instructions with rotation versus 1,432 for the baseline; these are not dynamic instruction counts.

The global-memory version is a nonstarter: four dx partial arrays written and read would cost `2 × 4 × E × 512 × 4 = 6.124 GB`, or **1,275.8 µs even at 4.8 TB/s**. The on-chip combine is essential to this proposal.

**Expected gain.** **60–180 µs if successful; a regression remains possible.** This discounts most of the occupancy ceiling for duplicated loads/setup, imbalance, barrier waits and shared traffic. The fresh cap failures below further reduce confidence: more warps alone demonstrably did not help. No gain has been measured for this ownership change.

**Exactness.** **Rounding-class.** All coefficients and arithmetic remain float32. dx and dr are grouped differently; atomics also arrive in a different order. Do not claim bitwise equivalence for the complete output. The force/stress gate is required.

**Closest existing ideas and difference.** P, **“36. More warps for the w′ … a register cap, then edge_gradient in shared memory”** and **“2. Degree-pure path parts (DEGREE_PARTS): adopted.”** U now closes the cap experiment at a 1,646 µs base: **maxreg80 1.255 (no spills), maxreg72 1.078, maxreg64 1.234**. These timings are verified in the new results summary; their raw lab JSON was not located here. This is unfavorable evidence, not a reason to retry the same caps.

The new mechanism instead removes most live accumulators from each warp and shares only final dx/dr partials. In my matched CPU build, forcing the unchanged body to 64 registers creates **49 STL and 39 LDL static instructions and a 104-byte stack frame**; the harmonic-owned 64-register body has **no STL/LDL and zero stack**. At cap80 the original still has no spills, confirming that avoiding spills alone does not explain away its measured loss. The untested distinction is the changed ownership and dependency structure, not just the occupancy number. Idea 36's proposed shared edge_gradient alternative loads/stores those accumulators during their updates; this proposal does not.

Idea 2 partitions the **forward by input degree**; this partitions the **backward by harmonic degree**, chosen specifically to give disjoint dsh ownership. It also differs from **“23. A persistent w′: closed”**: no persistent edge queue or next-edge prefetch is needed. The proposal escapes those tested mechanisms by changing live state and ownership, but still needs its own falsifier.

**Cheapest falsifier.** The main CPU resource check has passed for this render. Next, render several supported schedules and reject a layout that needs global partial dx, spills, or cannot sustain at least 28 resident warps at the intended block size. A mathematical CPU comparison of the generated grouped sums can check indexing, but cannot establish the float32 force/stress gate. The smallest future GPU lab is this w′ alone, including its shared combine; **stop if total kernel time is above 0.95 of baseline**, or if accuracy fails. Count the final atomics, not just the degree bodies. Effort: **one to two weeks**, including general schedule generation and qualification.

## 2. Reopen dout staging with a 32-channel stripe and a longer receiver tile

**Mechanism and files.** Add a receiver-segment tile generator and a striped template to `wprime_backward.py`. A CTA owns `(receiver, up to 16 edges, 32 channels)`. Four warps process different edges and loop over that tile. Stage only the receiver's 32-channel stripe across the 40 message components:

`40 × 32 × 4 = 5,120 bytes`.

Use explicit shared-memory addressing in the inner path code, with **no generic global/shared pointer selection**. Four channel stripes together cover the original 128 channels. dx components are channel-disjoint; dsh and dr need a small four-stripe reduction. Keep receiver segments and tile descriptors in the existing edge order, update them with neighbor compaction/rebuild, and preserve masks for invalid/padded edges. The tile length is one H200 default, not a table keyed by crystal or composition. Short segments can use a general fallback.

**Target.** The approximately **1,612 µs w′**, especially its dout reads. B's block-message shapes and the saved source imply 5,120 float values per receiver, or **102.4 MB** for all 5,000 atoms. The logical per-edge read demand is `E × 5120 × 4 = 7.6546 GB` per call. That is **logical load traffic through the cache hierarchy**, not a claim of 7.65 GB of HBM reads.

**Ceiling from the actual bound.** L's `F-smem-dout` takes **1,326.336 µs versus 1,565.792 µs**, a **239.456 µs** reduction. It uses an artificial 4 KB array and also changes registers from 96 to 80, so it bounds a bundle of locality, addressing and occupancy effects, not dout bandwidth alone.

The proposed four-stripe dsh/dr outputs require at most

`2 × 4 × E × (16 + 1) × 4 = 203.325 MB`

for scratch writes and reads, an ideal **42.36 µs** at 4.8 TB/s, before the reduction's final output and execution overhead. Subtracting that from the artificial bound leaves only **about 197 µs** of optimistic budget. This is a useful stop-budget, not a rigorous upper bound under changed occupancy.

For illustration only, if every receiver had approximately 75 edges, five 16-edge tiles would stage `5000 × 5 × 4 × 5120 = 512 MB` of dout. The actual degree/segment histogram is **unverified** and must replace that illustration before implementation. Averages alone do not establish tile utilization.

**Expected gain.** **30–120 µs if successful; zero or negative is plausible.** The reduced working set and longer reuse interval must pay for staging, tile tails, duplicated edge setup across stripes and the final reduction. This is an alternative to idea 1, not an independently additive w′ saving.

**Exactness.** **Rounding-class.** The stripe loads preserve bits; the dsh/dr channel reduction and dx atomic order change. No lower-precision coefficients or products. Force and stress must pass.

**The closed experiment and why this is different.** P, **“12. Receiver-staged dout in the w′: dead,” “Round 3: what was measured,”** and L, **“Staging measured … stopped.”** There are two separate facts:

1. `FF-dout-one-row = 1.001` is **not a valid locality test for this render**. It changes an unused `output_base`; actual block-message loads use `receiver` directly. Original and transformed sources compile to byte-identical SASS, including for the exact saved round-2 source. Details are in the corrections below.
2. Real full-row staging genuinely lost: **1.986 for two rows, 1.935 for one row; even no-fill was 1.131**. That evidence remains valid and is the reason a simple retry is not proposed.

The new implementation changes all three costs in the failed real design: **5 KB instead of 20–40 KB per CTA, up to 16 instead of four edges amortizing the copy, and statically shared loads instead of generic pointer selection**. It retains parallel edges rather than serializing a whole receiver in one warp. The channel-reduction cost above is the explicit price of making that change.

**Cheapest falsifier.** First fix the one-row bound's **actual 40 load expressions** and check its SASS. My corrected source changes addressing and compiles at 72 rather than 96 registers; a future timing therefore needs a resource-matched control before attributing the difference to locality. Count receiver-segment lengths on CPU: **stop this 16-edge design if fewer than half of contributing edges lie in tiles containing at least eight edges**. Then compile the direct-shared striped body: stop on spills or generic global loads in place of the intended shared dout loads. Smallest future GPU test: striped w′ **plus staging, partial reduction and amortized descriptor maintenance**; stop unless the sum is **≤0.95** of baseline. Effort: **one to two weeks**.

## 3. Shorten the w′ polynomial dependency chain with paired Estrin evaluation

**Mechanism and files.** Replace only the six-coefficient dual-Horner evaluation in `wprime_backward.py`; retain the exact same power-basis table, pieces, table layout, path factoring and float32 precision. For coefficients `g0…g5` and normalized argument `t`, use:

```text
t2 = t*t; t4 = t2*t2
v0 = fma(g1,t,g0); v1 = fma(g3,t,g2); v2 = fma(g5,t,g4)
w  = fma(fma(v2,t2,v1),t2,v0)
d0 = fma(2*g2,t,g1); d1 = fma(4*g4,t,3*g3)
dw_dt = fma(5*g5,t4,fma(d1,t2,d0))
```

This is the same degree-five polynomial and its analytical derivative in real arithmetic. It exposes independent value/derivative subexpressions after the coefficient loads. The longest coefficient-to-result FMA chain is three rather than five; the `t2/t4` chain and added multiplies are real costs.

**Target.** The approximately **1,612 µs w′** (W), specifically polynomial latency, not radial-table locality or the forward product.

**Ceiling and CPU result.** M removes **all table loads and polynomial work**, saving `1563.872 − 1152.128 = 411.744 µs`. That is an upper opportunity envelope for far more than this change. If all of that time were a five-stage dependence chain, shortening it to three would save at most `411.744 × 2/5 = 164.70 µs`. This is deliberately favorable: table loads remain, and the coefficient chains already overlap with other work.

C compiled the actual replacement at **96 registers, no stack/local memory**, the same as baseline. Static counts changed as follows:

| | Baseline | Estrin |
| --- | ---: | ---: |
| Total instructions | 1,432 | 1,444 |
| LDG | 97 | 97 |
| FFMA | 331 | 341 |
| FMUL | 73 | 94 |
| FADD | 191 | 181 |

This is explicitly a **dependency-length experiment with slightly more instructions**, not an arithmetic-count optimization. The SASS census does not establish that the new order executes faster.

**Expected gain.** **0–60 µs**. A miss is quite possible because independent paths/channels may already hide the chain. It should be cheap enough to reject quickly. Do not add its full estimated gain to idea 1's latency-hiding gain.

**Exactness.** **Rounding-class**, requiring the gate. The polynomial and derivative are algebraically unchanged, but evaluation order and the `3*g3`/`5*g5` rounding change. Keep the cutoff and below-`r_lo` derivative masks. No degree reduction, coefficient quantization, TF32 conversion or approximation to the derivative is proposed.

**Closest existing idea and difference.** P, **“9. Hoist the piece scale out of the w′ slope (HOIST_SCALE)”** removed 40 multiplies per edge and was neutral; **“34. The w′'s path factors folded into its value table at plan time”** changes CG scale work. Existing HORNER converted the basis and reduced live state. This proposal keeps that basis and table, targeting the **serial FMA depth**, which those instruction-removal measurements do not isolate. It is also distinct from the dropped lower-degree/wider-piece table proposals under **“Considered and dropped.”**

**Cheapest falsifier.** The no-spill CPU compile has passed. Next evaluate both forms over the fitted pieces' CPU certification grid and compare both value and derivative to the original float64 radial chain; reject any certificate regression beyond the specified budget. A small arithmetic check can verify the formulas but cannot qualify stresses. Future GPU lab: this single template replacement, otherwise identical; **stop above 0.98 of w′ baseline**. Effort: **two to four days** including table/error analysis and qualification.

## 4. Remove the first interaction's separate derivative-table stream

**Mechanism and files.** In [tabled_outer_sum.py](D:/Margraf/FastMACE/fastmace/kernels/triton/tabled_outer_sum.py), `_kernel(..., RECOMPUTE=True)` currently evaluates `a(r)` by a six-coefficient Clenshaw recurrence and separately evaluates its slope from five derivative coefficients. Create a **backward-only power-basis buffer** at plan construction, using the same float64 conversion principle as `wprime_backward.power_table`, and obtain **value and derivative together** with dual Horner. Keep the original Chebyshev buffer for the unchanged forward. Wire the extra buffer from `RadialTableModule` in [radial/__init__.py](D:/Margraf/FastMACE/fastmace/rewrites/radial/__init__.py), through `fused_first_message` in [generic/modules.py](D:/Margraf/FastMACE/fastmace/rewrites/generic/modules.py), and `_TabledOuterSum`'s saved backward inputs. Do not convert coefficients per force call or pass power coefficients to the existing Clenshaw forward.

For the K feature columns, six coefficient loads replace eleven. Convert the dual-Horner derivative from normalized t to distance using **`a′(r) = dp/dt × scales[piece]`**, with the original cutoff/below-domain masks. Keep the density column's existing scalar derivative recurrence initially; it is not necessary to change that column to realize the main saving. Keep the existing outer-product contractions, key walk, masks and global layouts. Thus this can be built and evaluated independently of idea 5.

**Target and checked shapes.** B measures **218.2 µs** for `tabled_outer_sum_backward`; its forward is **118.0 µs**, which receives no credit here. The first interaction has `K=64`, `D=16`, value coefficients **`[9,896,6,66]`** and derivative coefficients **`[9,896,5,66]`**. The `624` pieces used by the second interaction must not be substituted into this calculation.

**Ceiling.** Eliminating five derivative coefficients in each of 64 columns removes the logical demand

`E × 64 × 5 × 4 = 478,412,800 bytes`.

The complete value and derivative tables are only **12.773 MB and 10.644 MB**, respectively. A full-size backward power buffer adds **12.773 MB of persistent plan storage**; it replaces the value-table reads inside this backward rather than adding another per-edge stream. Much of the repeated traffic may hit cache; claiming `478 MB / HBM bandwidth` as a saving would be unjustified. A traffic-proportional favorable model is

`218.2 × (11 − 6)/11 = 99.18 µs`.

That is **conditional**, because coefficient loads are only part of this kernel; G loads, the two contractions, reductions, masks and outputs remain. The absolute whole-kernel bound is 218.2 µs. No measured “remove derivative table only” falsifier exists in the inspected record.

**Expected gain.** **10–40 µs**, reflecting the limited coefficient work and cache reuse. This is the cheapest plausible implementation in the shortlist. It does not claim to remove the 61 MB S intermediate or any GEMM.

**Exactness.** **Rounding-class within the existing certified-table approach.** A derivative of rounded power-basis coefficients is not bitwise the independently rounded derivative table. Extend the CPU certificate to this evaluator's **values and derivatives**, retaining float64 coefficient conversion and float32 execution. Reject tables that fail it, and still require the end-to-end force/stress gate. Re-fitting a less accurate table is not part of the proposal.

**Closest existing idea and difference.** P, **“T2. The tabled kernels at K = 64: nothing to do”** established that K was already 64; this changes the coefficient dataflow, not K. **“35. The first interaction's S kept in L2”** targets S/grad_S traffic across operations; this eliminates a separate derivative stream **inside the existing backward**, preserving S and its producers/consumers. HORNER is already known in w′, but the untested mechanism here is eliminating the first interaction's duplicate value/derivative table evaluations and loads.

**Cheapest falsifier.** CPU-certify the converted table and count the generated coefficient loads: six feature-vector loads must replace eleven, with no spill introduced by carrying value and derivative together. The first future GPU variant is just this backward; **stop above 0.95 of its baseline** (less than approximately 11 µs saved), or on any accuracy failure. Effort: **two to four days**.

## 5. Make the first interaction's backward a small edge-tiled 3xTF32 contraction

**Mechanism and files.** Replace the per-edge scalar reductions in the backward of `tabled_outer_sum.py` with a 16-edge tile for one `(receiver, sender-species)` key. These are edges of **one structure**, not batching structures. The same key's `G = grad_S[key]` is a dense `16 × 64` matrix. Evaluate `A[e,k]=a_k(r_e)` and `A′[e,k]=a′_k(r_e)` into on-chip storage and form

```text
dY[16,16] = A[16,64]  @ G.T[64,16]
Z [16,16] = A′[16,64] @ G.T[64,16]
dr[e] = sum_d Y[e,d] * Z[e,d] + density contribution
```

The second identity is the reassociation of today's `grad_a = Y G; dr = dot(grad_a,a′)`. It avoids writing either grad_a or Z to HBM. Two A tiles plus G require `(2×16×64 + 16×64)×4 = 12,288 bytes`; outputs and Y add a small amount. Use three FP32/TF32 product passes, retaining both cross terms, in the spirit of the current CUTLASS implementation. Start with ordinary `mma.sync`; this proposal does not depend on another native-wgmma GEMM project.

Generate tiles from equal-key segments, handling tails with masking or the existing scalar kernel. The same rule applies to arbitrary species counts and geometries. Do not assume every key has 16 edges. General architecture schedules supply K and D; unsupported or inefficient small shapes retain the existing exact mathematical path.

**Target.** B's **218.2 µs first backward**. This is not the 866.4 µs existing GEMM family, and it does not speed the 118.0 µs first forward in the estimate.

**Ceiling and a measured-rate calibration.** The two contractions require `2 × E × D × K × 2 = 1.530921 GFLOP` in ordinary product arithmetic, or **4.592763 GFLOP** counting all three TF32 passes. P, **“Where the time goes on the H200,”** reports the product microbenchmark's **1,082 µs** for the forward's `E × 4 × 60 = 89.7024 million` HMMAs. At 2,048 operations per HMMA that is **169.79 TFLOP/s measured effective throughput**, including that microbenchmark's splits and loads. This is not peak-FLOP arithmetic. The source's `m16n8k8` product and three passes agree with that operation count. I verified the calculation and source, but did not locate the original raw 1,082 µs result separately from the plan; that timing's raw provenance remains **unverified here**.

At that rate, the new contractions alone would take `4.592763 GFLOP / 169.79 TFLOP/s = 27.05 µs`. Therefore **191.15 µs** is a favorable *calibrated* saving budget before radial evaluation, layout, loads and outputs. The absolute bound remains the entire 218.2 µs kernel. Neither is a prediction: skinny tiles, tails and conversion work may achieve only 25–50% of that effective rate, making the product portion **54–108 µs** before the remaining work. These efficiency fractions are assumptions, not measurements.

**Expected gain.** **15–60 µs if successful**, possibly none. This is the riskiest small-kernel rewrite: the existing one-warp kernel already reuses G over equal-key runs, and coefficient evaluation remains substantial. It could combine with idea 4, but their upper gains must not simply be summed against the same 218.2 µs.

**Exactness.** **Rounding-class**, not bitwise and not a low-precision proposal. Three-pass TF32 here replaces FP32 scalar reductions, and the dr sum is reassociated. The saved gate offers no blanket budget for that change. Retain a scalar fallback and reject the implementation if the strict force/stress gate fails; do not drop a cross term to rescue speed.

**Closest existing idea and difference.** P, **“35. The first interaction's S kept in L2”** puts the following learned projection inside the outer-sum forward or recomputes grad_S. Here S and grad_S remain unchanged; **the two edgewise VJP contractions inside the existing first backward** become tensor-core work. **“33. Hopper GEMMs for the 3xTF32 products”** replaces existing GEMMs, whereas this creates a batched contraction from a presently SIMT edge loop. The rejected **TC_TP** work and **“22. The TP inside the next batch's k-loop”** schedule additional work around the second forward's already busy product; they do not test this first-backward transformation. The dropped projection-based **w′** in “Considered and dropped” is a different interaction and requires recomputing its learned radial weights; this proposal uses G that is already materialized.

**Cheapest falsifier.** On CPU, histogram equal-key run lengths and calculate `useful_edges / (16 × tile_count)`, including all tails. **Stop the tensor-core design below 60% utilization**, and reject any design that materializes A, A′ or Z in HBM. A compile-only check must show no spills and an on-chip tile footprint consistent with the estimate. Before a full rewrite, the smallest future GPU lab should retain coefficient lookup/evaluation but remove only the two contractions through an opaque, non-DCE replacement: **stop if that bound saves less than 40 µs**. The real variant must then save at least **20 µs** including descriptor/tail handling and pass the gate. Effort: **one to two weeks**.

## Ranking and what can actually be added

| Order | Experiment | Reason to do it in this order | Conditional gain per call |
| --- | --- | --- | ---: |
| 1 | First-backward shared value/derivative evaluation, idea 4 | Small, concrete change; no new synchronization or edge layout | 10–40 µs |
| 2 | Harmonic-owned w′, idea 1 | Largest new upside; complete compile-only prototype reaches 64 registers without spills; fresh cap losses lower confidence | 60–180 µs |
| 3 | Estrin w′, idea 3 | Cheap dependency experiment; reject quickly if added instructions win nothing | 0–60 µs |
| 4 | Channel-striped dout, idea 2 | Correct the invalid locality test first; real staging failures demand caution | 30–120 µs |
| 5 | First-backward tensor contraction, idea 5 | Larger effort and accuracy risk for a 218 µs target | 15–60 µs |

The ranges are estimates for variants that work, except idea 3's explicit zero. Any can regress or fail accuracy. **Nothing in this report is qualified for adoption.**

Ideas 1 and 2 are competing w′ layouts. Idea 3 can overlap their latency benefit. Ideas 4 and 5 share one first-backward budget. Plan idea 36 tests the same broad latency hypothesis as idea 1, plan idea 35 can interfere with the first-backward layout, and idea 31 overlaps some of idea 35's packing work. For that reason I budget approximately **0.1–0.30 ms for a successful combined subset**, not the sum of the upper endpoints. At the nominal 5.40 ms this gives `5.40/5.30 = 1.019×` to `5.40/5.10 = 1.059×`. Full force-and-stress timing is still required.

The latest measured results narrow the older forecasts. U's idea-30 lab saves **212.7 µs** on the whole forward (`1504.8 − 1292.1`, ratio **0.859**), still above its 0.85 continuation threshold, with production qualification pending. Idea 31's adopted LINEAR_DIV saves **49.2 / 37.0 µs** in the H200 whole-step A/B (raw logs: [ab](D:/Margraf/FastMACE/results/h200-ideas-20261105/idea31/h200/ab-lineardiv-ab.log), [ba](D:/Margraf/FastMACE/results/h200-ideas-20261105/idea31/h200/ab-lineardiv-ba.log)). That saving is now in the inspected checkout and is not future opportunity. Idea 33's tested separate-pass native GEMM bound loses; it cannot contribute the older forecast's 350 µs. Its unbuilt in-kernel-split form remains a known idea, not a sixth new proposal here.

The 2× target needs **2.70 ms** saved. Even eliminating the entire nominal 1.49 ms forward leaves **3.91 ms**, only **1.38×**. The plan's older 30–33 forecast of 0.5–1.5 ms is not a measured aggregate and is stale relative to U. Adding this shortlist's maximum to that maximum would double-count costs and compound optimistic assumptions. I cannot justify a 2× forecast under the brief. I also cannot prove a hardware lower bound that makes 2× impossible: the record has not isolated all latency and occupancy costs.

If 2× is a firm requirement, the constraint most plausibly needing reconsideration is the **numerical-precision budget**, or the comparison must turn out not to be like for like. Neither is justified here, and the stress failures argue against spending additional accuracy. Within the current constraints, a substantially better forward **and** backward execution scheme must demonstrate savings beyond today's bounds before 2× is credible. Lifting the already-lifted wgmma exclusion again would not supply new evidence. The private build's matched model, stress setting, accuracy and complete-step timing remain unverified.

## Corrections and qualifications to numbers in the plan

### A. `FF-dout-one-row = 1.001` did not make these dout loads read one row

This changes the interpretation of P, **“12. Receiver-staged dout … dead,” “What round 3 says,”** and repeated references in “Considered and dropped.”

The transform in [omat0_kernel_lab.py](D:/Margraf/FastMACE/tools/omat0_kernel_lab.py) replaces:

```cpp
const int64_t output_base = int64_t(receiver) * OUTPUT_DIM;
```

with `output_base = 0`. But R's saved BLOCK_MESSAGE kernel reads, for example,

```cpp
dout0[int64_t(receiver) * 256 + ...]
```

and similarly for dout1–dout3. `output_base` is not used by those loads. There are 40 such expressions in the round-9 render. CPU NVRTC builds produce **byte-for-byte identical `cuobjdump -sass` output before and after the old transform**, both on R's exact round-2 source and on the saved round-9 source. The raw timing was 1,566.720 / 1,565.504 = **1.000777**, but it is effectively repeat timing of the same kernel.

The timing number itself is real. The assertions **“making every dout load an L1 hit saves nothing”** and **“dout locality is worth nothing”** are unsupported. Consequently, conclusions that uniquely attribute mirror-order gains to sh/dsh locality are not established by this falsifier. This does **not** invalidate the real 1.935–1.986 staging losses or prove that dout staging will help.

### B. 5.40 ms has not been verified as a force-and-stress measurement

B's `force_field_info.compute_stress` is `false`; its plan description says `compute_stress = off (fixed cell: graph engine computes no stress)`. The profile is a force graph used in fixed-cell MD. Its replay kernel sum is **5.746165 ms**, while the separate BUTTERFLY-off MD A/B reports **5.7060 / 5.7062 ms** (W). Neither should be relabeled as the current 5.40 ms force-and-stress result. The nominal 5.40 ms is usable for the brief's arithmetic, but the requested matched stress-on baseline is **unverified**.

The old and new forward times are likewise different checkpoints, not an error bar. Round-13's [H200 A/B summary](D:/Margraf/FastMACE/results/research-answers-20261105/runs/round13/ab13-h200/summary.txt) already records A_LD128 at **0.9925 / 0.9956** of whole-step time; TP_NOFLUSH alone is **0.9992 / 1.0018**. Revision 14's “A/Bs run”/early-result language is stale relative to those completed artifacts. Do not credit A_LD128 again as a new opportunity or multiply its kernel ratio into every older timing.

### C. The quoted stress margin includes actual gate failures

[gate-default.log](D:/Margraf/FastMACE/results/research-answers-20261105/runs/round12/cgf-h200/gate-default.log) gives NbCl5 production max stress error **1.36978767598e−7** against reference-f32 error **6.50880732263e−8**, ratio **2.1045141**, limit **1.30176146453e−7**.

[gate-cgfold.log](D:/Margraf/FastMACE/results/research-answers-20261105/runs/round12/cgf-h200/gate-cgfold.log) gives **1.30459509555e−7 / 6.32254280979e−8 = 2.0634025**, limit **1.26450856196e−7**. Both logs say `passed: false`. These are evidence for “already over the limit on these runs,” not demonstrated positive headroom. They do not prove that every current run fails, but cannot qualify a new rounding-class variant under this brief.

### D. Idea 34's prescribed H200 CPU baseline omits an enabled switch

P's **“34. The w′'s path factors …”** asks for H200 FACTORED/TABLE_W/HORNER and describes A100 as “plus SCALAR_PATHS.” Actual `SCALAR_PATHS_GPUS` in `wprime_backward.py` includes **both NVIDIA H200 and NVIDIA A100-SXM4-80GB**. Its H200 source comment records off/on MD ratios **1.030 / 1.030**. The saved H200 render also contains the scalar-input shortcut.

The relevant comparison is therefore against H200 **with SCALAR_PATHS already enabled**, not an artificial off baseline. The “50 of 86 terms” algebra is not the same as 50 additional removable instructions in today's compiled H200 kernel; a census must include the current shortcuts.

### E. Idea 35's 51 µs is transfer arithmetic, and its consumer graph is simplified too far

`4 × 15000 × 1024 × 4 = 245.76 MB`; `/4.8 TB/s = 51.2 µs` is arithmetically correct. But it is the ideal service time for an assumed HBM traffic count, **not a measured maximum saving**. Cache residency, packing, achieved bandwidth, chunk-launch overhead and dependency stalls determine the actual change.

Also, the approximately 41 MB message output is not part of that four-transfer S/grad_S count, so subtracting it from 51.2 µs double-charges an unavoidable output that was never included in the proposed saved bytes.

B's sequence is more specific than “a GEMM reads S”: index **18** produces S; **20/22/24/26** pack/slice it; **21/23/25/27** are the four following GEMMs. In reverse, **150** assembles grad_S from four `[N×d,192]` arrays, and **151** consumes it in `tabled_outer_sum_backward`. That packing/assembly work must be assigned exactly once between ideas 31 and 35. No corrected speedup number follows from the bytes alone.

### F. The precision claim for changing MMA instructions is too strong

P, **“30. A warp-specialized wgmma forward,”** calls the result bitwise if passes and their order are retained. Three passes and the same split are necessary to preserve the intended precision, but do **not guarantee bitwise identity across MMA instructions, tile shapes and reduction schedules**. NVIDIA's [PTX ISA specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) leaves relevant matrix-accumulation order/rounding behavior unspecified. Classify a new mma.sync-to-wgmma implementation as **rounding-class until exactness is demonstrated**, and apply the gate. This is a qualification error, not a measured timing correction.

### G. “Six largest GEMM shapes = 828 µs” conflates shapes with aggregate rows

B's six largest `fm_tf32x3` rows sum to **827.7 µs**: `168.1 + 166.1 + 159.0 + 158.7 + 88.7 + 87.1`. They contain **29 launches** (`7 + 4 + 6 + 4 + 4 + 4`), aggregated by the profile's kernel/grid grouping. They are not six distinct `(M,K,N,B-layout)` shapes.

U's new idea-33 study separately identifies **24 distinct shapes across 33 launches**. Its six largest distinct shapes comprise **eight launches**, totaling **382.8 µs**, with three separate native passes at **410.9 µs** (1.074×). These are different denominators; multiplying that six-shape ratio into 828 µs would be wrong. P, **“33. Hopper GEMMs …”**, should specify whether it means six aggregate profile groups or six actual matrix shapes. The new study's conclusion closes its tested separate-pass/pre-split variants, not every possible single-read, in-kernel-split native mainloop.

## CPU compilation appendix and reproducibility limits

All temporary sources, compiler outputs and scripts are outside the repository in

`C:\Users\fzwin\AppData\Local\Temp\codex-fastmace-h200-20261106\`.

Useful artifacts:

* [compile130.py](C:/Users/fzwin/AppData/Local/Temp/codex-fastmace-h200-20261106/compile130.py), [base/variant census](C:/Users/fzwin/AppData/Local/Temp/codex-fastmace-h200-20261106/nvrtc130/compile_results.json).
* [compile_combined.py](C:/Users/fzwin/AppData/Local/Temp/codex-fastmace-h200-20261106/compile_combined.py), [combined census](C:/Users/fzwin/AppData/Local/Temp/codex-fastmace-h200-20261106/combined/compile_results.json), [combined rotated source](C:/Users/fzwin/AppData/Local/Temp/codex-fastmace-h200-20261106/combined/harmonic_cta_rotated.cu).
* [compile_caps.py](C:/Users/fzwin/AppData/Local/Temp/codex-fastmace-h200-20261106/compile_caps.py), [matched cap census](C:/Users/fzwin/AppData/Local/Temp/codex-fastmace-h200-20261106/caps/compile_results.json), for the distinction between the existing forced cap and the proposed ownership change.
* [calculate.py](C:/Users/fzwin/AppData/Local/Temp/codex-fastmace-h200-20261106/calculate.py) for the byte/operation arithmetic and selected raw lab values.

Compilation used WSL CPU processes, `CUDA_VISIBLE_DEVICES` empty, NVRTC **13.0** from the local installed library cache, `--gpu-architecture=sm_90 --std=c++17 -default-device`, then CUDA **13.3** `cuobjdump -sass` and `-res-usage`. A separate NVRTC 13.3 pass gave the same main resource-count conclusions. No CUDA driver/runtime initialization, kernel launch, GPU discovery command, model execution or test suite was used. These are offline builds, not host or device validation runs.

The baseline round-9 source SHA-256 is `5565cb3386c4527520c634bf86968398f48b96739cfe1b412e5c951ea392f3ad`. The round-2 source SHA-256 is `c482280c044bf554b386b1230892d14c9052bfb2f042954e39352a73506cc8fc`. The combined rotated prototype's SHA-256 is `0232182c46c1d4aea5e36d4e4c609e055f9157aa096d344614b05382f5c5c1be`.

The census counts textual SASS instructions across the compiled function, not executed instructions per edge; branch bodies and final reductions are included. Register counts are compiler evidence for these exact renders/toolchains. The harmonic prototype has not been run, checked for numerical correctness, or integrated with the generic schedule generator. The two first-backward ideas and striped-dout design are implementation proposals, not completed kernels. Their performance, end-to-end accuracy, behavior on other models and force-and-stress step savings are all **unverified**.

Only this report is added to the requested worktree. It is left uncommitted.

## Second pass: two different backward paths (6 November, after the polynomial-forward rebuttal)

**I withdraw the earlier 2.5 ms budget. Two further mechanisms survive the novelty check, but neither establishes 0.5 µs/atom.** Both replace work *inside* the approximately 1.15 ms left by `F-no-table-horner`. They are alternatives for interaction 2's backward, not additive savings. I did CPU algebra calculations and compile-only NVRTC builds for them. No GPU, model execution, benchmark or test suite was run.

The plan now includes ideas 37–42. This section does not re-propose them. The broader optimisation ledger was also checked: exponent-scaled FP16x3 already exists and failed; so does Cartesian projection before the final reduction. These are particularly easy proposals to mislabel as new. Main advanced during the investigation; the last observed HEAD was `6561cc13`. The compiler comparison below deliberately uses the fixed, archived round-9 source identified in the appendix, not a claim that main was benchmarked.

### N1. Cartesian harmonic adjoints: differentiate the scalar contraction before emitting CG code

**Mechanism.** Replace the `FACTORED` CG body in `fastmace/kernels/oeq_streaming/wprime_backward.py` with an adjoint expressed using Cartesian harmonic polynomials. Keep the same radial value/slope tables, all channels, all neighbours, checkpoint and cutoff rules. Generate the polynomial algebra from the model's path schedule. Begin with the backward only; no wholesale rewrite of the force call is required.

The useful architectural fact is stronger than just “l is small”: the ten OMAT-0 paths have input degree 0 or 1, and every vector path has `L = l ± 1`. Let `n` be the edge direction, `x` its sender's vector feature, and `g` the receiver's adjoint for one path/channel. Define the homogeneous harmonic polynomial

`H_L(n) = sum_m g_m Y_Lm(n)`.

Up to a fixed, model-independent path normalization, the scalar contraction with the output adjoint is:

- Scalar input: `f = x0 H_l(n)`.
- Vector input, `L = l + 1`: `f = x · grad H_L(n)`.
- Vector input, `L = l - 1`: `f = (x · n) H_L(n) - |n|² (x · grad H_L(n))/(2L+1)`.

Differentiate these scalar expressions to produce `dx` and the three edge-vector derivatives. With the path normalization absorbed into H, the edge-vector contribution away from the normalization epsilon is

`w'(r) f n + (w(r)/r) [grad_n f - l f n]`.

The homogeneous derivative is taken **before** setting `|n|² = 1`. Omitting its derivative would give an incorrect force. Keep the existing normalization-epsilon branch/fallback in a real implementation. Retain edge-vector adjoints for the usual position and cell/stress chain rule; a force-only scatter is insufficient.

Convert each receiver's `g` into independent Cartesian harmonic coefficients once per force call. Degrees 0, 1, 2, 3 still need **1, 3, 5, 7 coefficients**, respectively. The missing Cartesian tensor entries follow from symmetry and zero trace; there is no increase from 40 to 49 stored components. For example, use `Gzz = -Gxx-Gyy` for degree 2. Degree 3 can be evaluated by forming the symmetric matrix `Q = G·n`, then `Qn` and `Qx`. This factorization matters: expanding all polynomials and asking a generic CSE pass to recover it was substantially worse.

The initial implementation needs a node basis-conversion kernel and a custom backward boundary that accepts three Cartesian edge adjoints. Later, the conversion can move before `linear_1`'s adjoint GEMMs: the harmonic basis transform acts on component indices, while those GEMMs mix channels, so the two commute. Fold path normalization into constant weights where appropriate. This is a change of coordinates in the numerical calculation, not retraining or a different potential.

**CPU evidence.** The scalar identities and their derivatives were checked against e3nn's CG tensors on 12 small random inputs for each of the ten paths. The largest absolute float64 discrepancy was below `4e-13` (unnormalized random vectors). This checks the algebra, not end-to-end forces or stress. A compile-only full-edge backward sketch retains the archived kernel's table lookup, dual Horner, channel loop and sender `dx` atomics, but assumes its adjoint input has already been converted to the Cartesian basis.

| Same NVRTC 13.0 / sm_90 build | Archived backward | Factored Cartesian sketch |
|---|---:|---:|
| Registers/thread | 96 | **56** |
| Stack/local/shared bytes | 0 / 0 / 0 | 0 / 0 / 0 |
| Static SASS instructions, including cold paths/reductions | 1,432 | 975 |
| Instructions in the repeated channel-loop body | 574 | **630** |
| Channel-loop FFMA / FMUL / FADD | 330 / 71 / 20 | 297 / 140 / 48 |
| Channel-loop LDG instructions | 90 | 74 |

The important positive result is the **natural reduction in live state**. The important negative result is that the hot loop is larger. The 32% reduction in whole-function static instruction count is **not a 32% runtime prediction**: much of it is a reduction tail outside the repeated loop. The sketch has not been executed; its basis-conversion wrapper, epsilon handling, node/cell gradient integration and accuracy qualification remain unimplemented.

**What it attacks and its ceiling.** The measured backward was **1,612.4 / 1,613.4 µs**, butterfly off, in `results/sx012-profile-20261106/README.md`. On 128-thread blocks, a register-only occupancy calculation changes from `floor(65536/(128×96))×4 = 20` resident warps to `floor(65536/(128×56))×4 = 36`. A hypothetical wholly latency-limited scaling would save `1612.4×(1-20/36) = 716.6 µs`. This is an **optimistic resource model, not an observed speedup or a hard time bound**; the larger hot loop, memory dependencies and converter remain.

The actual reduction-only falsifier is much stricter: `F-no-reduce = 0.964` on a 1,634 µs base, only **58.8 µs**. Any material win here must therefore come from changing the CG computation and its register lifetime, not simply reducing three values instead of seventeen.

A standalone conversion of all `[5000,5120]` adjoints reads and writes **204.8 MB**, an ideal HBM lower bound of **42.7 µs** at 4.8 TB/s. Converting the 16-component node adjoint before the channel GEMMs would instead move **81.92 MB**, **17.1 µs** at that bandwidth. These are byte bounds, not kernel-time forecasts, and the latter integration is not built.

**Expected gain.** Unverified: **0–250 µs net per force call** is a reasonable planning range; a slowdown is possible. A successful kernel could plausibly save 100–250 µs after conversion, but the offline evidence cannot yet favor that over a tie. I would not budget the 717 µs occupancy projection.

**Exactness.** Rounding-class. The real-arithmetic identities preserve the model; Cartesian conversion, zero-trace reconstruction and sum reassociation change rounding. A float64 algebra check does not establish the force/stress gate. Apply the original ≤2× reference gate, including NbCl5 stress, without substituting “same failure as baseline.”

**Closest existing ideas and the escape.**

- P, **“5. Small tail cuts,” (e)** forms the Cartesian derivative from completed `dsh`. This proposal eliminates the componentwise `v_j`/`dsh` calculation in the channel loop itself.
- Ledger **`sx-wprime-reduction`**, and `plans/ideas-20261105-round2.md`, **“Cartesian gradient before the w′ reduction”**: `F-cart3` ran **1.081×**, while no reduction ran 0.964×. That variant added a 16×3 projection to the existing derivative calculation. It did not lower the scalar CG contraction into harmonic value/gradient/Hessian contractions.
- P, **“36. More warps for the w′ …”**: a forced cap changes compiler allocation without removing live mathematical state and lost. Here the compiler independently chooses 56 registers without a cap.
- P, **“37. The w′ split by spherical-harmonic degree …”** gets more warps by changing ownership and combining `dx` on chip. N1 keeps one warp per edge and changes the representation of its derivative. Its gains would overlap 37 and the tail-removal part of 5(e).
- P, **“Considered and dropped: A rotated-frame (SO(2), eSCN-style) TP”** pays for edge-specific rotations. N1 uses a fixed node basis transformation, amortized over edges; no per-edge Wigner rotation is required.

**Cheapest falsifier.** The CPU resource check is done, including the unfavorable hot-loop count. Next make the converter and Cartesian-edge output numerically complete, then compare this **one backward plus its converter** in the kernel lab. Stop if the combined time is not at least **5% below** a same-job baseline, or if the full gate fails. Preserve a separate timing for conversion so a fast inner kernel cannot conceal its cost. Do not spend time integrating a forward Cartesian backend first.

**Effort.** Approximately 2–4 days for a complete lab variant; 1–2 weeks for generic schedule generation, autograd/stress integration and qualification.

### N2. Receiver-tiled FP64 matrix adjoints, with channels reduced on tensor cores

**Mechanism.** Replace the per-edge SIMT backward with a receiver-segmented, edge-parallel matrix engine. This is a new backward kernel, not another implementation of the radial product. For one path, one receiver and a tile of edges, define

`S[e,i,o] = sum_j CG[i,j,o] Y_j(e)` and `D[o,c] = dout[receiver,path,o,c]`.

Use two matrix contractions:

1. **Feature adjoint:** `U[(e,i),c] = S[(e,i),o] @ D[o,c]`, then `dx[e,i,c] += w[e,c] U[e,i,c]`.
2. **Angular moments:** `M[(e,i),o] = (w[e,c] x[e,i,c]) @ transpose(D)[c,o]`.

Contract the small M matrices with `dS/dr_vector` **after** reducing channels. The radial derivative is `sum_c w'[e,c] sum_i x[e,i,c] U[e,i,c]`; it reuses U. A second moments GEMM for w′ is unnecessary. Keep U in fragments only long enough to form `dx` and the radial partial. M persists across channel tiles and is contracted into Cartesian edge gradients on chip. Keep the existing sender atomics; this is not the excluded receiver-owned atomics-free design.

Use native **FP64 `mma.sync.m8n8k4`** for these contractions: FP32 inputs convert exactly to FP64, and no high/low split or three-product preparation is needed. Evaluate the existing radial tables as today. FP64 accumulation changes the rounding, so it still needs the gate. It is not an accuracy exemption.

Start with **16 edges per receiver slab**, **32 channels at a time**, and paths handled independently. Existing receiver order supplies slabs without sorting by distance or species. Every edge in the slab runs in parallel. A degree/path specialization selects the small matrix dimensions; node count, composition and neighbour distances do not select tuning tables. Handle receiver tails and padded invalid edges explicitly.

**CPU evidence.** For all ten paths, a small float64 calculation compared both matrix identities, `dx`, the radial-weight adjoint and the Cartesian geometry adjoint against direct differentiation of the CG contraction. Maximum absolute discrepancy: **2.67e-15**. This does not test a GPU implementation.

A compile-only skeleton for the largest vector-path shape (16 edges, 3 input components, up to 8 padded output components, 128 channels in 32-channel slices) builds at **56 registers, 18,432 bytes shared memory, zero stack/local storage** on sm_90. It emits FP64 DMMA instructions. It includes operand conversion, shared staging, U and M contractions, but **omits** radial-table evaluation, geometry preparation, receiver segmentation, force epilogues and node scatter. It writes U/M intermediates for a future isolated lab check; a production implementation must consume them on chip. Its resource count is not a completed-kernel occupancy result.

**What it attacks and work/traffic bounds.** The same **approximately 1,613 µs** backward, specifically its per-channel CG/gradient work and repeated receiver-adjoint loads. For E = 373,760 and C = 128, the exact path list gives `sum_p input_dim = 22` and `sum_p input_dim × round_up(output_dim,4) = 132`.

- U: `2 E C ×132 = 12.6301 GFLOP` of padded FP64 matrix work.
- M: `2 E C ×22×8 = 16.8401 GFLOP`.
- Total: **29.4702 GFLOP**, before receiver tails. For tiles of 16, `sum_receivers ceil(degree/16) < E/16 + N`; with N = 5,000, the padding multiplier is below **1.2141**, giving **35.7781 GFLOP** as a conservative tiling bound for the supplied padded edge count.
- Receiver-adjoint logical global reads become **0.4784 GB** before tails instead of **7.6546 GB**, if each path/slab reads D once. This is a reduction in *issued load volume*, **not an assertion that either amount comes from HBM**.
- The weighted-X shared-memory write plus read alone is **16.8401 GB** (`2 E C ×22×8` bytes). Operand staging is emphatically not free. The radial coefficient traffic also remains.

Do not divide these FLOPs by a peak specification and call the quotient a predicted kernel time. **No sustained H200 DMMA rate for these small, staged shapes was available in the local record.** The actual performance ceiling is therefore unresolved. The measured locality-only experiment provides one useful limit: `F-smem-dout` saved **239.46 µs** (`1565.792-1326.336`, with a register-count confound). N2 must beat that mechanism through the matrix computation as well, not claim a 16× bandwidth improvement in the whole backward.

The relevant non-radial reference remains `F-no-table-horner`: **1152.128 / 1563.872 µs = 0.736715** in `results/research-answers-20261105/runs/mirror-h200/lab-wprime/lab.json`. N2 changes that surviving computation, so this deletion experiment is not its runtime floor. Equally, the difference between the two times is not a separately measured radial kernel that can safely be added to a guessed DMMA time.

**Expected gain.** Higher risk than N1. **0–400 µs net** is a useful research planning envelope, with a substantial chance of losing. A 200–400 µs win would require the conversion, shared-memory staging and tensor contractions together to beat today's SIMT path. There is no evidence for a 750 µs complete backward yet.

**Exactness.** Rounding-class, same model and radial tables. Accumulate products in FP64 and avoid inserting unnecessary intermediate FP32 products/stores into M. The native instruction is supported by the compiler and the [NVIDIA PTX specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html); its availability establishes neither speed nor the force/stress gate. Stress still comes through the full Cartesian edge/cell derivative.

**Closest existing ideas and the escape.**

- P, **“Considered and dropped: FP64 tensor cores (DMMA) for the TP”**, and the earlier plan's **“The tensor product on tensor cores (TC_TP): correct, slower, rejected”**. TC_TP is a *forward* contraction, with the radial MMA already occupying the tensor pipe; its three-product operand preparation cost as much as the SIMT work, and it ran **1.276×** on the A100-SXM4. N2 uses the *backward*, whose radial path is scalar table/Horner work, uses one FP64 contraction without operand splitting, and puts the 128-channel force reduction inside M. These are specific differences; merely saying “Hopper is faster” would not reopen it.
- P, **“12. Receiver-staged dout in the w′: dead”** and **“38. dout locality, tested properly …”**. Real staging cost **1.94–1.99×** because it retained the consumer calculation and synchronization/serialization costs. N2 keeps a slab's edges parallel and consumes the staged adjoint with dense matrix instructions. The expensive weighted-X staging above is the corresponding new risk.
- P, **“Considered and dropped: An S tile in the w′'s FACTORED sums.”** That variant only rewrote u, leaving v's channelwise CG work. N2 rewrites both u and the angular derivative, moving the latter's channel sum before the CG/geometry contraction.
- P, **“41. The first interaction's backward contractions on tensor cores.”** N2 applies to interaction 2, includes the sender-feature adjoint and its atomics, and uses FP64 small matrices with a receiver's D as the reused operand. It is not an estimate for the existing first-interaction proposal.

**Cheapest falsifier.** Complete an isolated one-path U/M lab kernel, including real S preparation, X/D conversion, weighted-X staging, the radial partial and output handling. The compile-only skeleton is the starting point. Sum shape-specific timings for the actual ten-path edge count. Stop if this **non-radial portion already takes ≥1.0 ms**: too little room remains for Horner, coefficient loads, tails and integration to justify this rewrite. Then require a complete w′ variant at **≤0.85×** a same-job baseline before production work, and pass the original force/stress gate. These thresholds are engineering stop rules, not measured promises. No such GPU work was performed here.

**Effort.** Roughly one week for a meaningful isolated lab; 2–4 weeks for a complete generic backward, scheduling, tails and qualification.

### Ranking and what this does to the 0.5 µs/atom target

1. **N1 first:** the strongest new evidence is a full-edge algebraic sketch naturally reaching 56 registers, without spills or edge/degree handoff. Its larger hot loop makes the converter-inclusive timing decisive.
2. **N2 second:** a more different execution path, with a CPU-checked contraction identity and a spill-free matrix skeleton. Its 16.84 GB of weighted-X shared traffic and unmeasured small-shape DMMA rate are serious risks.

These two replace the same backward and **must not be added**. N1 overlaps 37 and 5(e); N2 overlaps 38 and much of 37. Neither saving can be stacked with a separate claim to eliminate the same `dout` or reduction cost. They can in principle coexist with idea 30's forward, but that requires a fresh complete-step measurement.

The nominal target needs **5,400-2,500 = 2,900 µs** saved. Even deleting today's approximately 1,613 µs backward entirely leaves approximately **3,787 µs**, above target. Therefore a successful new backward alone cannot establish 0.5 µs/atom. I have not found evidence sufficient to promise that target under the stated constraints. This is **not a proof that an exact implementation cannot reach it**, and no particular constraint has been proved necessary to relax. It does establish that another large, independent forward/node-path improvement is required; a radial-only rewrite is insufficient. The supplied 5.40 ms nominal force-and-stress target must also stay separate from the archived `compute_stress:false` breakdown, as discussed earlier.

### One further numerical correction to the rejection record

P, **“Considered and dropped: DMMA float64 tensor cores for the product,”** compares **66.9 TFLOPS** with **approximately 170 TFLOPS** from the three-pass TF32 product. Those FLOP denominators differ. The underlying product is `2×373760×1280×64 = 61.2368 GFLOP`; its three-pass emulation is **183.7105 GFLOP**. The old 1,082 µs microbenchmark corresponds to **56.60 useful TFLOPS**, or **169.79 TFLOPS counting the three passes**.

That paragraph's comparison is not a valid reason by itself to reject one-pass FP64. **It also does not establish that FP64 wins.** P's later **“The product's rate is not the HMMAs' floor”** correctly warns that the microbenchmark includes preparation and is not the fused kernel's tensor occupancy. The earlier 1,082 µs must not become a new DMMA speedup budget. I am not promoting a full-width FP64 radial product on this arithmetic correction alone.

### New CPU artifacts

All are in the same temporary directory used above; none is a production kernel or an executed GPU variant.

- [Algebra and tile census script](C:/Users/fzwin/AppData/Local/Temp/codex-fastmace-h200-20261106/new_path_census.py) and [results](C:/Users/fzwin/AppData/Local/Temp/codex-fastmace-h200-20261106/new_path_census.json).
- [Cartesian generator/compiler](C:/Users/fzwin/AppData/Local/Temp/codex-fastmace-h200-20261106/compile_cartesian.py), [factored CUDA sketch](C:/Users/fzwin/AppData/Local/Temp/codex-fastmace-h200-20261106/cartesian_vjp/cartesian_factored.cu) and [resource census](C:/Users/fzwin/AppData/Local/Temp/codex-fastmace-h200-20261106/cartesian_vjp/compile_results.json).
- [FP64 adjoint skeleton/compiler](C:/Users/fzwin/AppData/Local/Temp/codex-fastmace-h200-20261106/compile_adjoint_mma.py) and [resource census](C:/Users/fzwin/AppData/Local/Temp/codex-fastmace-h200-20261106/adjoint_mma/compile_results.json).
- [Arithmetic and hot-loop census](C:/Users/fzwin/AppData/Local/Temp/codex-fastmace-h200-20261106/new_paths_numbers.json), generated by [new_paths_numbers.py](C:/Users/fzwin/AppData/Local/Temp/codex-fastmace-h200-20261106/new_paths_numbers.py).

The factored Cartesian source SHA-256 is `c0cf71b334e47eb08b52a2ffe3d229bfe38c0ef4f5157ce4f1fe69a8632eb603`. The matrix skeleton SHA-256 is `9685ce25cfbf3b5c95fb0b535cac7bd897809afc7f019aa433694317f57e05fd`. Both used the same offline NVRTC 13.0 / cuobjdump 13.3 process as the appendix. The report remains uncommitted; no production files or shared ideas plan were changed.
