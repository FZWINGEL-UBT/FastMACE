# OMAT-0 medium: a breakthrough screen after the 3.64 ms H200 step

1. The verified H200 step is 3,643.7 µs; a 20% improvement must save 728.7 µs.
2. I found no new same-model mechanism that currently justifies that saving; the three research routes below do not yet qualify for implementation.
3. Interaction 2 is the decisive target: its forward and reverse total 1,905.4 µs and must improve by 38.2% to save 20% of the step.
4. All current 3×TF32 kernels total 547.0 µs in 22 launches, rather than the prompt's approximately 515 µs in 18 launches.
5. The checkpoint has 128 channels, input degrees 0/1 in interaction 2, ten independent radial paths, and 86 nonzero CG entries.
6. Its inspected radial and node matrices are full rank; two paths that a simple Gaunt substitution would tie are demonstrably independent.
7. A one-pass force walk still needs the node adjoint; retaining just radial values and slopes would require a 3.83 GB whole-cell tape.
8. A spatial fast operator is the most different remaining idea, but short range, low neighbour count and many independent filters make its break-even doubtful.
9. The A100's 7.36 ms is verified whole-step evidence; a current kernel breakdown and physical cache-byte measurements are missing.
10. Another 2× is not supported by this audit: it needs a new interaction formulation and substantial savings outside interaction 2, not another kernel tuning round.

## Scope and evidence

Research date: 8 November 2026. Read-only source snapshot: main `0d0eefaebf7d1e0152a8cdf80f2cb89f6239b284`, in the detached worktree containing this report. The complete 5,102-line `docs/optimisation-ledger.md`, project conventions, SPEC, defaults and the cited results were audited. No GPU work, suites, production changes, commits or pushes were performed. Checkpoint inspection used CPU-only PyTorch with `CUDA_VISIBLE_DEVICES=` empty.

**Concurrent-main update:** at final verification main had advanced to `73e421ea`. Its new round-6 results adopt JOINED_LAYOUT (0.9805/0.9815), leave w′ STAGE_EDGES off (0.9969/1.0010) and forward PREFETCH_AHEAD off (1.0046/1.0067), and report about 3.58 ms. The new breakdown includes all three variants, so it is not a clean adopted-default profile. These source/ledger changes were checked read-only. All arithmetic below deliberately retains the prompt's round-4 baseline; no per-kernel time is inferred by subtracting the new A/B delta. The new changes do not alter the checkpoint algebra or reopen a breakthrough route.

Local references below are relative to this worktree. The supporting [model.json](model.json) records the checkpoint hash, shapes, CG census and singular values. [floor_model.py](floor_model.py) generates [floor-model.json](floor-model.json), including arithmetic, byte counts and source hashes. These are inspectable calculations, not benchmark results.

The checkpoint `/home/fzwin/models/mace-omat-0-medium.model` hashes to `d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a`. It contains 9,063,204 parameters and 89 elements; cutoff 6 Å; two interactions; 128 channels; correlation order three; ten Bessel functions; radial hidden widths 64, 64, 64 with normalized SiLU; polynomial cutoff exponent five; and the Agnesi distance transform. Interaction 2 consumes `128x0e + 128x1o`, uses SH degrees 0–3 and ten paths, and produces 40 raw components per channel. The two readouts are scalar linear and `128 → 16 → 1` with SiLU. These details matter: high-angular-degree asymptotics and a wholly polynomial energy are both inappropriate assumptions here.

## 1. What the current measurements permit

H200 source: `results/h200-fastpath-20261108/round4/breakdown/{summary.json,ops.csv,kernel_sequence.csv,categories.csv}`, capture commit `978dbef5`. The eager force-call kernel sum is 3,399.16 µs, 103 launches. The MD wall time is 3,643.70 µs. The graph B timing is 3,569.64 µs and includes the second half kick; its replay has 148 kernels. These measurements are related but are not interchangeable.

| H200 bucket | µs/force call | Fraction of MD wall time | Saving if this bucket were free |
|---|---:|---:|---:|
| Interaction-2 w′ reverse | 1,099.0 | 30.2% | 1,099.0 µs |
| Interaction-2 spline forward | 806.4 | 22.1% | 806.4 µs |
| All `fm_tf32x3_*` | 547.0 | 15.0% | 547.0 µs |
| First tabled outer sum, reverse + forward | 214.3 + 115.0 | 9.0% | 329.3 µs |
| Symmetric-contraction category | 170.9 | 4.7% | 170.9 µs |
| Atom-level elementwise category | 223.9 | 6.1% | 223.9 µs |
| Remaining geometry, other kernels, copies/fills | 222.7 | 6.1% | 222.7 µs |

Rows use rounded CSV values, hence their 3,399.2 µs sum. The remaining bucket includes the **28.0 µs** actually classified as copies/fills. It is not 270 µs of copies. Four polynomial cuEq launches account for 164.8 µs of the 170.9 µs category; the rest is glue. The GEMM total is obtained by kernel name, because the CSV classifies several custom GEMMs as `Other`.

At the reported shares, forward alone must become about **10.4×** faster, or reverse alone **3.0×**, to save 728.7 µs. Together they need **1.62×**. The sum of custom 3×TF32 GEMMs and all symmetric contractions is only 717.9 µs. Adding the separate 26.8 µs ordinary GEMM category reaches 744.7 µs, requiring virtually all of that combined computation to disappear for a 20% win. This rules out ordinary replacements of those buckets as a 20% story, even though genuine algebraic elimination of a kernel class would still be valuable.

For another 2×, the target is 1,821.85 µs. Leaving everything outside interaction 2 unchanged leaves only `1821.85 − (3643.70 − 1905.4) = 83.55 µs` for interaction 2: approximately **22.8× faster**. This is an Amdahl screening calculation using the measured force-call shares, not a prediction of identical eager and replay timings.

A100 source: `results/a100-omat0-best-20261108/README.md`. The adopted pack gives 9.85 → **7.36 ms**, with build-order ratios 0.7471/0.7469 and steady windows 7.33–7.44 ms. There is no post-pack kernel breakdown in that results tree. Its 20% threshold is **1,472 µs**. I do not manufacture A100 per-kernel times by scaling H200 shares. A100 also retains Clenshaw in the first backward where H200 uses POWER_BACKWARD.

## 2. Floor model: what is known, and what is not

### Arithmetic reference, not an algebraic lower-bound proof

There is no established count of *irreducible* FLOPs for this trained function under all exact reformulations. Original radial-network FLOPs are especially unsuitable: tabulation has already removed those networks from the hot path. The following is a transparent work reference for the present formulation, using 373,760 padded edges, not a claim that every operation is necessary or every padded edge is live.

The checkpoint census gives 86 nonzero CG entries, 62 distinct input-component × SH-component products, 46 within-output additions and 40 path-output components per channel. A scalar forward reference is

`E × C × (62 + 46 + 2×40 + 6×10) = 11.865 GFLOP`,

where the last term is cubic Horner evaluation and FMA counts as two FLOPs. It omits fixed CG scale costs and differs from compiler cross-path factoring. The analogous factored reverse reference is `E × C × 578 = 27.652 GFLOP`; its component count is in the script. These are schedule estimates, not SASS counts or lower bounds.

For the three dominant dense node chains, semantic multiply/add counts are firmer: composed first projection and reverse **7.864 GFLOP**; second projection and reverse **13.107 GFLOP**; second `linear_up` and reverse **1.311 GFLOP**. Their sum is **22.282 GFLOP**, excluding minor heads. Three TF32 products execute approximately three times this tensor work; use either that issued work or a semantic rate divided by three, never both.

| Work reference | GFLOP | H200 arithmetic-only µs | A100 arithmetic-only µs |
|---|---:|---:|---:|
| Interaction-2 forward | 11.865 | 177 | 608 |
| Interaction-2 reverse | 27.652 | 413 | 1,418 |
| First outer forward | 1.053 | 16 | 54 |
| First outer reverse | 2.057 H200 | 31 | 121, including A100's extra Clenshaw work |
| Dominant dense node chains | 22.282 semantic | 135 | 429 |
| Sum, excluding SC and geometry | — | **771** | **2,630** |

Hardware references are H200 FP32 67 TFLOP/s, dense TF32 494.5 TFLOP/s and 4.8 TB/s HBM; A100 SXM 80 GB FP32 19.5 TFLOP/s, dense TF32 156 TFLOP/s and 2.039 TB/s HBM. NVIDIA's H200 table quotes TF32 with sparsity, so it must be divided by two for this dense model. Sources: [H200 specifications](https://www.nvidia.com/en-us/data-center/h200/) and [A100 datasheet](https://images.nvidia.com/data-center/a100/a100-datasheet.pdf).

**771 µs is neither a complete force floor nor an achievable forecast.** It excludes the actual SC instruction count, geometry, reductions, issue inefficiencies and physical cache traffic. Similarly 2,630 µs is not an A100 prediction. The current H200 direct-work rates are approximately 14.7 TFLOP/s forward and 25.2 TFLOP/s reverse; dominant semantic node work divided by *all* TF32 time is 40.7 TFLOP/s. Large gaps to peak exist, but peak arithmetic alone does not identify a useful replacement.

### Bytes and the cache ambiguity

| Quantity, float32 unless stated | Size | Meaning |
|---|---:|---|
| Positions plus forces | 0.120 MB | External input/output; not all internal traffic |
| Sender node features `[N,512]` | 10.24 MB | Reusable atom array |
| Edge SH `[E,16]` | 23.92 MB | Current stored geometry |
| Raw interaction-2 message `[N,5120]` | 102.40 MB | Current kernel-boundary array |
| Projected message `[N,2048]` | 40.96 MB | Current kernel-boundary array |
| First outer sum `[N,3,16,64]` | 61.44 MB | Species-separated current array |
| Allocated spline `[9,731,1280,4]` | 134.74 MB | Full table, not bytes fetched per call |
| One logical spline sweep | **7.655 GB** | `E × 1280 × 4 coefficients × 4 B`, before masking/reuse |
| Saved w and w′ per edge | **3.827 GB** | A proposed tape, not current allocation |

At 4.8 TB/s the two logical coefficient sweeps would cost **3.189 ms**, already greater than their measured combined 1.905 ms. Therefore the requests cannot all be compulsory HBM traffic. Forward's 9.49 TB/s and reverse's 6.97 TB/s are **logical requested-byte rates**, not measured L2 bandwidth. The same requested-byte calculation gives 7.508 ms on A100 and is equally unsuitable as a physical floor.

Some conditional HBM costs are useful: raw-message write plus read, 204.8 MB, is 42.7 µs H200 / 100.4 µs A100; the dominant separate GEMM input/output accesses, 532.48 MB, are 110.9/261.1 µs; separate SC input/output accesses, 271.36 MB, are 56.5/133.1 µs. These overlap the measured kernel buckets and may hit cache. They must not be added as extra savings or called compulsory traffic after a changed execution plan. Atomic payload is likewise not physical read-modify-write traffic.

A usable physical bound would be

`T ≥ max(F_SIMT/P_SIMT, 3F_MMA/P_TF32, B_HBM/BW_HBM, B_L2/BW_L2)`.

Current physical `B_HBM`, `B_L2`, and calibrated L2 bandwidth are **unverified on both GPUs**. The local 4.24 TB/s copy result (`results/research-answers-20261105/README.md:170`) measures HBM, not L2. Old NCU L2 activity percentages are not byte bandwidth. There is consequently no defensible numerical L2 floor to insert here.

### Limiter diagnosis and the first missing measurement

For interaction 2, **the limiting combination of scalar issue throughput, dependencies and cache traffic remains unmeasured**. The work references exclude an explanation based solely on saturating peak FP32 FLOPs, but are not SASS counts and cannot exclude a mixed-instruction issue limit. Forward has a serial edge accumulation, 80 registers and 38% estimated occupancy; reverse has 72 registers and 44%. Neither proves HBM or L2 saturation. The no-atomics reverse variant was **1.080× slower**, so atomic removal is not an established opportunity (`kr-wprime-vector-atomics`, ledger:4647). These atomics accumulate sender-feature gradients, not trained-weight gradients.

The dense node chains have substantial small-matrix/batching overhead relative to a saturated tensor pipe, but only 547 µs total headroom. First outer sums and SC also sit far above arithmetic-only references; their individual absolute ceilings are 329 and 171 µs. There is no evidence that the 28 µs fill/copy category is a hidden major bottleneck.

The next authorised profiling job should run `tools/step_breakdown.py` on **current defaults on both GPUs**, preserving separate eager-force, replay and MD measurements. Collect physical DRAM bytes, L2 sectors/hits, local-memory transactions, issue/stall reasons and achieved pipe rates for `forward` and `wprime_backward`; record active edges and table interval populations. On A100 first establish the post-pack bucket split. This is a recommendation for a later lab, not a run performed here. It resolves whether a new formulation should reduce coefficient traffic, shorten dependencies, or change scalar work into dense work.

## 3. Ranked architectural routes: falsify before implementing

**No route below currently satisfies the requested ≥20% evidence bar.** I cannot responsibly provide three positive forecasts. This ranking is of remaining research questions, with negative screens and explicit advancement thresholds. Their ceilings overlap and are not additive.

### 1. Replace the local edge matrix by a certified fast spatial operator

**Mechanism.** For fixed features, each message path is a translation-invariant, species-pair-dependent convolution:

`M_i,p,c = Σ_j w_p,c,Zi,Zj(|r_i−r_j|) Y_l(r̂_i−r_j) x_j,c` followed by the CG combination.

Replace the edge evaluation with near-field direct work plus an interpolated H²/FMM operator, or a particle-mesh/NUFFT operator. Apply its transpose and spatial derivative for feature gradients and forces. This would actually remove both streaming interaction-2 kernels, rather than merely reimplement their edge loop. Integration points: `spline_forward.py:292`/`:558`, `wprime_backward.py:750`/`:1011`, and the interaction dispatch in `rewrites/generic/modules.py:294`. All kernel paths here are under `fastmace/kernels/oeq_streaming/` unless otherwise specified.

[Black-box FMM](https://mc.stanford.edu/cgi-bin/images/f/fa/Darve_bbfmm_2009.pdf) provides the general numerical-kernel construction, not evidence of a MACE speedup. The proposed extension needs a value **and derivative** approximation, species handling and a compact-support near field.

**Floor screen.** The target is 1,905.4 µs H200. A 20% win requires the complete replacement—including spreading, transforms, transpose, derivatives and near field—to cost **≤1,176.7 µs**. The mathematical zero-cost ceiling is 1,905.4 µs; no positive expected gain range is justified yet. On A100 let the unmeasured current interaction-2 total be `T2,A`; the corresponding requirement is `Tnew,A ≤ T2,A−1472 µs`, not a scaled H200 number.

The serious objection is that the direct operator is already O(N) with only about 75 neighbours. The CIF's 39.05 Å box gives density 0.08397 atoms/Å³. A 2 Å leaf box contains only **0.67 atoms on average**; a box large enough for 16 atoms has side **5.76 Å**, nearly the entire cutoff. There is little smooth, well-separated, densely populated far field to aggregate. This is a distributional screening example, not a per-crystal implementation rule.

A mesh is also expensive. Even an **unqualified** 0.5 Å mesh needs 79³ points. Three sender species × 512 input components are 1,536 scalar fields: **3.03 GB** of real grid values before FFT workspace. The unprojected receiver fields would be 15,360 fields or **30.3 GB** if materialized. Streaming avoids simultaneous storage, and moving projection into Fourier space reduces output fields, but introduces dense channel mixing at every frequency. Neither is free. A 0.25 Å mesh is 7.84× larger. These spacings are examples, not accuracy settings or lower bounds.

**Exactness and certification.** A finite interpolated fast operator is approximate. At setup, bound both `|K−K̃|` and `|∇K−∇K̃|` for every path/pair, including cutoff and short-range behaviour. Those bounds must propagate through actual feature and node-Jacobian bounds; a kernel certificate alone does not certify forces. Arbitrary crystals provide no fixed small feature/degree bound, so a general method needs conservative runtime range checks and a direct fallback. Passing a handful of gate structures is not a proof of that global bound. Final adoption still requires the 3× gate, including stress.

**Novelty.** Unlike `eng-species-morton`, `omat0-atom-order` and `conv-sobek` (ledger:3276, 4363, 2033), this replaces the discrete sum by a hierarchical/spatial representation. It is not sorting the same gathers or moving the 65-term channel product across each edge.

**Cheapest falsifier / stop.** Spend **2–4 person-days** on CPU separation-rank and derivative-interpolation counts using the actual checkpoint: include partially cut-off blocks, all independent paths and representative irregular point clouds. Count near-field pairs and all source/target fields. Stop if even an optimistic roofline for the *complete* plan exceeds 1,176.7 µs H200, or if a force-error certificate needs an unsupported environment restriction. The simple uniform-mesh construction already fails to make a compelling case. A viable hierarchical result would warrant **20–40 further person-days**, replacing the convolution and adjoint backend; risk is high. **Expected saving on either GPU: unverified, plausibly negative; no lab recommendation yet.**

### 2. An exact edge-frame / expanded Gaunt operator, specialized to the trained paths

**Mechanism.** Rotate vector input onto each edge axis, where only `m=0,±1` channels couple; evaluate the resulting SO(2) paths and rotate outputs back. Alternatively construct a sum of spherical/Fourier signal products that retains all independently trained path filters. This replaces the CG representation in forward and adjoint, at `spline_forward.py:321` and `wprime_backward.py:728`. Setup uses checkpoint instructions, not a structure-specific table.

The basis-change principle is established by [eSCN](https://arxiv.org/abs/2302.03655). [Gaunt products](https://arxiv.org/abs/2401.10216) provide a Fourier route, but matching the original operation's expressivity is essential. The 2026 [vector spherical harmonics construction](https://arxiv.org/html/2602.21466v1) supplies a complete asymptotic construction; its §7 explicitly cautions that its fast-transform regime is impractical at angular degrees used by current networks.

**CPU screen already performed.** Edge-axis local terms drop from 86 to 20, but this checkpoint has input degree only 0/1; rotating ten independent path outputs back and preserving 1,280 independent radial weights remain. A single Gaunt product would tie some filters. For actual `1⊗1→0` and `1⊗1→2` channel-zero coefficient rows, rank is two, `smin/smax=0.411`. Sampling their radial functions at 128 distances from 1 to 5.8 Å also gives rank two, ratio 0.1073. This falsifies that tied-filter substitution, **not every expanded Gaunt construction**. All ten `[128,64]` radial path blocks have rank 64, with relative smallest singular values 0.033–0.099; node projections are also full rank.

Combining paths before inverse rotation by moving the channel projection per edge is worse: existing node projection is **6.5536 GFLOP** forward; even using just local dimensions 1,3,3,3 gives **318.432 GFLOP**, 48.59× more. At that projection's measured 44.8 semantic TFLOP/s it would take 7.11 ms. That is a rate extrapolation, not a hardware floor, but is enough to reject this trade.

**Ceiling / expectation.** Same overlapping H200 1,905.4 µs target, with ≤1,176.7 µs required for a qualifying replacement. Neither rotations nor expanded Gaunt have a verified total-work reduction remotely equivalent to that timing reduction. The simple tied-filter version is incorrect; the per-edge projection version loses the work screen. **No positive H200 or A100 expected range is supported.** A100 has the same checkpoint algebra, and still lacks the timing split.

**Exactness / novelty.** An exact expanded transform is rounding-class, requiring the 3× gate. Truncating its rank is approximate and cannot be labelled rounding-class. Closest closures are `cx-n1-cartesian-wprime`, which lost 1.060/1.062 including conversion, and `rad-low-rank` (ledger:4947, 1712). An older ideas plan also considered SO(2), but its count of 92 CG terms is wrong; its approximately 202 rotation FMAs are not an irreducible count because inactive local columns can be omitted. This screen uses the actual paths and does not simply repeat that argument.

**Falsifier / effort / stop.** The rank falsifier is already in `model_probe.py`. At most **1–2 further person-days** should enumerate the minimal active rotation DAG and an exact independent-filter Gaunt expansion, including its reverse. Stop if gains require tying filters, truncating nonzero singular values, or moving projection per edge. Only a surprising full value-and-VJP cost reduction warrants compilation; implementation would be **10–20 person-days**, replacing CG code generation. Current recommendation: **stop the simple versions**.

### 3. Retain radial state through the final nonlinear node calculation

**Mechanism.** A receiver tile computes the last message, runs its node tail, obtains the node adjoint and then applies the reverse to retained w/w′. Hopper clusters could distribute that state across shared memory; a larger tiled schedule could retain it in L2. This differs from saving only messages or staging `dout`: the intended elimination is the second radial evaluation. Sites: `spline_forward.py:292`, `wprime_backward.py:606`, and last-tail composition in `rewrites/species/__init__.py:527`.

There is a genuine dependency barrier:

`M_i = Σ_e K_e x_s;  E_i = F_i(M_i,H1_i)`.

`∂E_i/∂M_i` does not exist as an available value until the sum and nonlinear tail finish. Analytic spline slopes do not supply it. Forward-mode forces would need 3N global seeds, or approximately 225 receiver-local coordinate directions even before the first interaction's two-hop dependencies. A post-projection `[E,2048,3]` derivative tape alone is **9.19 GB**.

**Floor screen.** Saving w and w′ is `373760×1280×2×4 = 3.827 GB`; writing and reading that through HBM costs at least **1,594.7 µs H200 / 3,754.1 µs A100** at rated bandwidth, before useful work. H200's entire reverse currently takes 1,099 µs. Per receiver with 75 edges, the tape is **768 KB**, larger than an SM's shared memory: 228 KB Hopper and 164 KB A100. A cluster can pool memory, but then several SMs serve one receiver whose per-degree GEMMs have only 1/3/5/7 rows. Grouping receivers improves matrix batching while multiplying the retained state. Sources: [Hopper](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) and [Ampere tuning guides](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html).

L2 tiling avoids the full HBM penalty but does not automatically reduce traffic: today's two passes request eight coefficient floats per weight; the cached plan reads four coefficients then writes and reads two values, also eight floats. It exchanges reused coefficient rows for unique edge values. It needs a demonstrated latency/dataflow advantage, not just the word “cache”.

**Ceiling / expectation.** Total target again overlaps interaction 2 and part of its node tail. Eliminating the *entire* reverse would save at most 1,099 µs; radial retention eliminates only a subset. No current measurement isolates that subset under SPLINE_W. The global-tape version fails; the cluster plan has **no defensible positive H200 range**, and A100 lacks both clusters and a measured target. Do not count its ceiling in addition to route 1 or 2.

**Exactness / novelty / falsifier.** The same operations in a changed reduction schedule are rounding-class and need the gate. `fusion-receiver-local` (ledger:4683) already considered forward + node tail + reverse; its no-output forward was 0.994×, and an ideal staged-dout bound was 0.847×. Retaining radial state is the different mechanism; message-only fusion is known. Spend **1–2 person-days** on a concrete shared-memory/L2 lifetime and GEMM-row schedule. Stop if the tape spills to HBM, or complete estimated work cannot save 729 µs after lost batching. Implementing a successful cluster schedule would take **15–30 person-days** and replace the last-interaction execution region. Current recommendation: **no implementation until that resource contradiction is resolved**.

## 4. Other directions rejected by the screen

| Direction | Why it does not currently qualify |
|---|---|
| Exact temporal delta updates | `Mnew = Mold + Kold Δx + ΔK xold + ΔK Δx` requires three dense edge products unless rank/sparsity is proved. `prec-delta-gemm` (ledger:2545) found median feature changes 20.8% despite small position changes. Exact reuse of changed geometry is not free. |
| Taylor force prediction / thresholded edge reuse | Approximate. A degree-p Taylor force remainder needs `M_(p+2) ||ΔR||^(p+1)/(p+1)!`. No useful cheap high-derivative bound is established; normalized SH derivatives worsen at short separation. The gate's force/stress floors, 1e−5 eV/Å and 1e−7 eV/Å³ (`planner/selfcheck.py:85`), do not make “0.01 Å motion” an error certificate. |
| BF16/FP8 compensated node products | Naive BF16 three-product omission is order 2^-16, larger than TF32 three-product order 2^-22. Six BF16 terms reach order 2^-24 but at twice TF32 throughput have the same nominal tensor time. A triangular FP8 E4M3 expansion through total degree five needs 21 products; at four times TF32 throughput that exceeds three TF32 products. These are operand-error screens, not end-to-end gate bounds. More elaborate schemes are not disproved, but the whole node bucket is only 547 µs. |
| FP16×3 node GEMMs | At an ideal 2× tensor rate, maximum saving is 273.5 µs, 7.5% of the step. Normalization and exponent safety cost more. `omat0-fp16x3-gemm` (ledger:4342) already records raw A100 timing benefits with worse errors and expensive normalized whole paths. |
| Energy-directed polynomial compilation | Last product/skip linears are already folded into readout. Pushing dense mixing through cubics creates channel-index coefficient tensors; retaining a low-rank arithmetic DAG recovers much of the original work. Replacing cuEq with a fused value/VJP implementation is kernel fusion, not elimination of the polynomial operation. `lin-t08-paper` (ledger:1192) is the closest closed composition. No demonstrated removal of a new whole class. |
| Cartesian, FP64 adjoint GEMMs | Already tested: Cartesian lost despite fewer registers; one FP64 vector-path adjoint costs 555 µs and six paths already cost 3.33 ms (`cx-n1`, `cx-n2`, ledger:4947–4966). |
| Locality, zero channels, ordinary persistent kernels, TMA | Sorting and zero-block scans are known; there is no exact zero-channel opportunity in the inspected matrices. Moving identical loads with TMA or keeping the same dependency chain persistent does not establish a new computation. Graph B's measured inter-kernel gaps total only 64.4 µs. |
| Neighbour-list reuse | `stages.csv` records 15 rebuilds/200 instrumented steps, **4,365 µs/rebuild and 327.4 µs/step** of accumulated rebuild duration. The summary's stage values are already divided by step count (`tools/step_breakdown.py:958`); do not amortize twice. All rebuilds were early and no steps were redone, so that side-stream duration is not established exposed wall cost. Even its full elimination is below 20%. The summary's negative “GPU idle” value reflects overlapping accounting, not actual idle time. |

Literature reinforces the qualification boundary. [FlashTP](https://proceedings.mlr.press/v267/lee25l.html) and [NequIP/Allegro AOTI](https://arxiv.org/abs/2504.16068) accelerate different framework/kernel baselines; their reported factors do not predict gains over this path. [LunaTP](https://arxiv.org/html/2610.18393v1) uses node-centric CSR/register aggregation and bounded path groups, largely mechanisms already present or measured here. [Symmetrix-XL](https://arxiv.org/html/2610.01036v1), §§3/5, recomputes edge quantities in reverse; it does not supply a cheap one-pass force schedule. Its published static-coordinate throughput also excludes neighbour rebuilding. [FlashCart](https://arxiv.org/abs/2610.06409) includes a different compressed model architecture; that is a potential owner-decision direction, not an exact checkpoint rewrite.

## 5. Corrections, verdict and next allocation

The current records require four corrections/qualifications: **547 µs/22** custom TF32 launches rather than 515/18; **28 µs** classified copies/fills rather than 270; **86** CG entries rather than the old plan's 92; and Symmetrix-XL's approximately **14.2 ms is its NVE result**, whereas its local Langevin windows are 14.52–14.65 ms (`results/sx012-profile-20261106/README.md`). None changes the main conclusion.

**A further 2× is physically conceivable but not a credible engineering forecast for this fixed checkpoint today.** The partial ideal-arithmetic reference is well below current time, so this audit does not prove a mathematical floor. It does show why another isolated GEMM, contraction, layout or launch improvement cannot supply it. A successful plan must make the combined interaction-2 region far cheaper **and** remove substantial work elsewhere; making interaction 2 merely twice as fast gives roughly 26% of the current step, not another 2× overall.

I would allocate the next research effort to the current-default physical-byte/pipe profile, followed only by the CPU spatial-rank screen if a substantially different formulation is still desired. I would not spend a GPU lab on the three simple constructions rejected above. There is no evidence-backed three-to-six-item breakthrough shortlist to implement.

If the owner wants a different path toward a large reduction, **the clearest additional direction to authorise is changing or retraining the model**: for example, a model trained with a low-rank spatial/filter factorization or a cheaper equivariant interaction. That could change the dominant cost. It is outside this report's same-model scope and requires new scientific validation; no timing or accuracy claim is made for it. This is an owner-decision alternative, not a proof that all future exact inference algorithms must fail.
