# Round 2: four more outside ideas and our own, ranked together (5 November 2026)

Sources:
- the owner's second outside list (four ideas);
- our own ideas (owner: "Try to come up with new super creative ideas yourself!");
- the first round's five ideas (`plans/ideas-20261105-a100.md`), re-ranked here.

Everything was checked against the code on main (`e5ea0204`), the A100 profile
`results/omat0-a100-profile-20261104-4b70e3f2/breakdown/ops.csv` and the kernel-lab records. The profile measures
13.67 ms per force call on MACE-OMAT-0 medium, Benchmark_loup SrTiO3 10×10×10 (5,000 atoms, 373,760 edges). Budgets
are per force call on the A100-SXM4-80GB unless noted. Nothing below is built or measured yet. This ranking replaces
the first round's order.

## Ranking

| # | Idea | Source | Budget it attacks | Ceiling | First check | Verdict |
|---|---|---|---|---|---|---|
| 1 | 3×FP16 MMAs in the tensor-core forward, W pre-split at setup | own | the W split (no-W-split bound 3.69 → 3.18 ms) and the MMA issue | 0.3–0.5 ms (2–4 %) | timing-only lab variant and a range probe (A100, about $0.2) | **Screen first** |
| 2 | Slot-batched `edge_outer_sum` backward: each (receiver, sender-species) block read once | outside 2 | 501.1 + 65.6 µs | about 0.3 ms (2 %), every GPU | lab: one G block for all edges; a constant G | **Screen second** |
| 3 | Fused first-interaction pipeline: table inside the outer-sum forward, slope inside its backward | own | 1.40 ms in 7 kernels | about 0.95 ms with 2; realistic 0.5–0.8 ms | after 2 | **Build only after 2 wins** |
| 4 | Mixed degree {3, 5} per (pair, piece) inside the degree-5 piece set | outside 4 | table loads (Ada: 0.31 of the base kernel) and 4 of 11 table FMAs per weight | proportional to the edge-weighted degree-3 share, which is likely small | count on leto's CPU (free) | **Count first** |
| 5 | Dual Horner and l₁ = 0 decoupling in the w′ backward | round 1 #2 + outside 3 | 96 registers, 31 % occupancy | a few % of the kernel, only if 80 registers fit without spills | `ptxas` registers and spills for sm_80 and sm_89 (free) | **Cheap screen** |
| 6 | Dense CUTLASS species skip, selected-store epilogue | round 1 #1 | 260 µs | 1–1.5 % | forward epilogue only | as in round 1 |
| 7 | tf32x3 GEMM entry with strides, offsets and a scale epilogue | own | 348.9 µs in 23 glue kernels | up to 2.5 %, likely 1–1.5 % | attribute the glue kernels first | **Attribute first** |
| 8 | Mirror-pair w′ backward: one slab and one w, w′ evaluation per undirected edge pair | own | the w′ backward, 3.15 ms | A100 about 0.45 ms before locality losses; likely net negative | lab: two variants | **Bound only; prior negative** |
| 9 | Interaction 2's table evaluated inside the tensor-core forward | own | 207.1 µs and the a read | 0.15–0.2 ms | after 1 | **Later** |
| 10 | One geometry and force epilogue kernel | own | about 525 µs of edge glue | about 0.2 ms | attribution first | **Later** |
| — | Tensor-core weight projection da = dw W | outside 1 | — | negative | already measured (`omat0-wprime-aprime`) | **Do not build** |
| — | No redistribution tile; fully tabulated forward; Euler identity; moments | round 1 #3–#5 | — | — | — | as in round 1 |

## Outside 1: tensor-core projection da = dw W in the w′ backward (do not build)

This was measured on 4 November as `omat0-wprime-aprime` (REJECTED, `tools/wprime_da_bound.py`). The product
`da = dw W` of Benchmark_loup's shape (373,760 × 1,280 × 64) took:

| Form | Time |
|---|---|
| one CUTLASS 3xTF32 GEMM | 2.27 ms |
| cuBLAS FP32 | 4.14 ms |
| cuBLAS TF32, one pass | 1.27 ms, but about 1e-3 error |

The slope it would replace cost 1.81 ms at the time (wprime 4.70 ms, without the slope 2.89 ms).

**The fused form.** The standalone GEMM pays a 1.9 GB round trip of dw that a fused form would not, so that bound is
not strictly an upper bound. The arithmetic still rules it out:
- **The slope today** (`FACTORED` + `TABLE_W`, `wprime_backward.py:319-341`): per weight, w′ from the slab already in
  registers (5 FMAs) and one FMA into dr. That is 6 MACs. `FACTORED` has since removed most of the per-term
  weight-gradient work that the 1.81 ms included.
- **The projection:** 64 MACs per weight, one row of W, so 10× the slope's. The A100's 3xTF32 rate is 2.7× FP32
  SIMT, which makes the 64 MACs worth about 24 FP32 FMAs. Even 3×FP16 (5.3×) gives about 12, and that is before dw
  is staged into MMA fragments.
- **The table read stays.** w is still needed for dx and dsh and comes from the same slab. Dropping the slab as well
  means also computing w = a W on the tensor cores in the backward: another 64 MACs per weight.
- **The Ada:** there the kernel is bound by table loads, which the projection does not remove, and TF32 runs at
  only 2× FP32.

## Outside 2: slot-batched `edge_outer_sum` backward (screen second)

**What the kernel does** (`fastmace/kernels/triton/edge_outer_sum.py:132-160`):
- **Programs:** each covers 32 consecutive edges (`BACKWARD_BLOCK_EDGES`).
- **Gather:** per edge it reads its slot's block G[key] (D = 16 × K = 64) row by row: 4 KB per edge, 1.53 GB per call
  through L1 and L2.
- **Unique data:** 15,000 slots (5,000 receivers × 3 sender species) × 4 KB = 61 MB.
- **Order:** edges are sorted by receiver, then sender species (the segmented first sum), so about 25 consecutive
  edges share each block.
- **Profile:** 501.1 µs (80 registers, 38 % occupancy), plus a 65.6 µs glue kernel
  (`…add_edge_outer_sum_backward_slice_backward_transpose_view_28`).

**Floor and ceiling:**
- **DRAM floor:** a (96 MB), y (24 MB), G once (61 MB), and grad_a and grad_y written (96 + 24 MB): about 300 MB,
  0.16–0.19 ms. The arithmetic is 1.5 GFLOP, negligible.
- **Ceiling:** about 0.3 ms per call (2 %). The path is in the generic default on every GPU, so every GPU gains.

**Build:**
- One program per run of equal keys, or the forward's chunk walk with run detection.
- The run's G block is loaded once into registers or shared memory, then grad_a = Y G and grad_y = G a as FP32 FMAs.
- Tensor cores are not worth it at 1.5 GFLOP, and TF32 would add error to gradients that are exact to rounding today.
  So "tiny GEMMs" should mean FP32 SIMT, not MMA.

**Falsifiers** (kernel lab, captured tensors, A100 and Ada):
1. key ≡ 0 for every edge: one G block, always in L1, with the same instruction stream. This bounds perfect sharing.
2. G loads replaced by a constant: the instruction-only bound.

If (1) is not below about 0.7 of the base, the gather is not the cost and the idea dies.

The chunk-size screen `omat0-eos-chunks` (64 edges per program) is a different idea and still PENDING in the ledger.

## Own: the fused first-interaction pipeline (after 2)

Today the first interaction's edge path is seven kernels, 1.40 ms per call:

| Kernel | µs |
|---|---|
| `radial_table_forward` [9, 896, 6, 66] (153 registers) | 209.2 |
| `edge_outer_sum_forward` | 219.7 |
| `edge_outer_sum_backward` | 501.1 |
| `radial_table_backward` [9, 896, 5, 66] (176 registers, 13 % occupancy) | 298.2 |
| gather and slice backward before it | 108.5 |
| outer-sum backward glue | 65.6 |
| table backward glue | 11.5 |

Between these kernels, a ([E, 64], 96 MB) is written and read in the forward, and grad_a (96 MB) in the backward.

**The fused kernels:**
- **Forward:** evaluate the table inside the segmented outer sum. The 12.8 MB table fits the A100's 40 MB L2. The
  kernel accumulates y ⊗ a per run, and the receiver's density in the same pass, since receivers are contiguous.
  a never reaches memory.
- **Backward:** extend idea 2's kernel. a is evaluated again from the table, as `recompute_first_table` already does
  in its checkpointed variant, and dr_e = grad_a · a′(r_e) comes from the slope table. grad_a never leaves
  registers, and the gather and slice kernels go.

**Floor and ceiling:**
- **Forward:** about 90 MB of DRAM and 0.6 GB of L2 table reads, 0.15–0.2 ms.
- **Backward:** about 120 MB of DRAM and 1.1 GB of L2 (value and slope slabs), 0.25–0.3 ms.
- **Together:** about 0.45 ms against 1.40 ms, a ceiling of about 0.95 ms (7 %) including idea 2's share. Realistic
  0.5–0.8 ms, given how many registers the table kernels use today.

**Caveat:** `omat0-first-table-columns` removed the zero-padded slice copy and measured 1.002 / 0.999 on the A100.
So the 108.5 µs kernel is not counted as a saving of its own; only fusion that removes whole passes counts.

**Order:** idea 2 is this pipeline's backward core and goes first. The forward half and the slope fusion follow only
if 2 wins.

## Own: 3×FP16 MMAs in the tensor-core forward (screen first)

**Today** (`fastmace/kernels/oeq_streaming/fused_forward.py:67-107`): per warp mma.sync m16n8k8 TF32 tiles,
hi·hi + hi·lo + lo·hi. W and a are split into high and low parts on the fly in every batch (`TC_SPLIT = "mask"`).

**Where the time goes** (A100 kernel lab, §11 and §14):
- The MMA triples "cost little"; the splits are the cost.
- On the current kernel, skipping the W split takes 3.69 → 3.18 ms (0.862).
- `TC_PRESPLIT` split W once per block as float2 and lost (1.362): it doubled the slice's shared memory, which
  dropped the block from 12 to 8 warps and raised registers from 168 to 194.

**The idea:** exponent-shifted 3×FP16 (`docs/design/2026-10-26-fp16x3-and-hybrid-gemm.md`). big = fp16(x),
small = fp16((x − big)·2¹¹), C = big·big + 2⁻¹¹(big·small + small·big), on m16n8k16 MMAs.
- **W is a weight.** Split it once at plan setup, with power-of-two row scales, into two FP16 planes. Each element
  then takes the same 4 bytes as today's FP32 slice, so this gets the no-W-split saving without `TC_PRESPLIT`'s
  shared-memory cost.
- **a** is split on the fly as today, with a power-of-two scale per edge, since the cutoff makes rows near r_max tiny.
  One multiply per w element in the redistribution step undoes it.
- **MMAs:** k16 instead of k8 means 5 instead of 9 k-steps at K = 65 (80 padded against 72), at the same issue time
  per instruction on sm_80.
- **Accuracy:** FP16 and TF32 both carry 11 significant bits, so the 22-bit split matches 3xTF32. Rounding both parts
  to nearest is free for the pre-split W and one conversion for a. That is the `round` class (7.6 / 10.7 units of
  2⁻²⁴ at k = 9 / 65 on the RTX 5060 Ti), not `mask` (19.3 / 16.3). It might also shrink the accepted A100 NbCl5
  stress excess, which comes from tf32x3 plus this kernel.

**Why it differs from the rejected 3×FP16 work.** `prec-fp16x3-*` (26 October) were standalone GEMMs on
activations. There, per-tile normalisation, splitting and rescaling in every call outweighed the FP16 rate (1.12–1.71×
cuBLAS on the A100, RTX A4000 and RTX 4000 Ada). Their accuracy did qualify: worst cancellation error 5.02e-7,
against IEEE 6.44e-7 and tf32x3 5.59e-7. Here one operand is constant, and the other's split is shared by the warp's
channel tiles.

**Ceiling:** about the no-W-split bound, 0.5 ms (3.7 %), plus the fewer MMAs, minus the cost of scaling a. Realistic
0.3–0.5 ms.

**Other GPUs.** FP16 runs at 4× FP32 on sm_89, so 3×FP16 gets 1.33× the SIMT rate there, against 0.67× for 3xTF32. The
tensor-core forward, slower on the Ada today, may therefore be worth one Ada lab run. On the RTX 5060 Ti 3×FP16 gets
0.67× the SIMT rate, so no.

**Falsifier:**
- a timing-only lab variant on the A100: m16n8k16 FP16 MMAs, W pre-split in FP16 planes, no scaling, against the
  current tensor-core kernel;
- a probe of a's and W's ranges from a captured step, to size the scaling.

About $0.2 on Modal.

## Outside 4: mixed degree {3, 5} per (pair, piece) (count first)

**What we verified:**
- **No divergence:** the w′ kernel runs one edge per 32-lane group, with lanes over channels
  (`wprime_backward.py:155-158`). A degree per (pair, piece) is therefore the same for the whole warp.
- **No new layout:** a degree-3 piece is a degree-5 piece with c₄ = c₅ = 0, so one flag per (pair, piece) is enough.
  The slab stride stays 6 and no pieces are added. That is unlike the global degree-4 table (MD step 1.08 / 1.04) and
  the global degree-3 one (certificate 4.8).
- **Savings on degree-3 edges:** one of three float2 loads and 4 of 11 table FMAs per weight in the w′ backward, plus
  fewer table bytes in the two `radial_table` kernels.

**Why the share is likely small.** The fit (`fastmace/rewrites/radial/__init__.py:314`) splits each piece until
degree 5 meets 1e-7 for values and derivatives:
- Halving a piece cuts the degree-5 error about 2⁶ = 64×, so accepted pieces sit between tol/64 and tol.
- At the same width, degree 3's error is roughly the degree-5 error to the power 2/3: 5e-6 for 1e-8, far above the
  tolerance.
- Pieces can qualify only where the floor (1e-6 of the domain maximum) dominates, near the cutoff, or where the
  function is nearly cubic.
- The derivative binds harder. |T_k′| ≤ k², so the dropped terms' slope can reach 16|c₄| + 25|c₅| times the piece
  scale.
- The share has to hold for all 1,280 weights of a (pair, piece), since one warp evaluates them all.

The list's 87 % is not supported by anything in our fit data yet.

**Count (leto's CPU, free).** Fit OMAT-0 medium's tables. Mark the (pair, piece)s where dropping c₄ and c₅ keeps every
column's value and derivative within the fit's own criterion, and weight them by the piece histogram of SrTiO3 and
NbCl5 MD edges. Below about 50 %, drop the idea. Above, render the flag and re-run the certificate, the GPU test and
the gate.

## Round 1 #2 and outside 3: dual Horner and l₁ = 0 decoupling (`ptxas` first)

**The paths** (the `FACTORED` body, `wprime_backward.py:246-317`). OMAT-0 medium's interaction 2 has 10 paths ×
128 channels (1,280 weights): four with l₁ = 0 (16 of the 86 terms) and six with l₁ = 1.

**l₁ = 0 decoupling:**
- **Today:** per path, u₀ = Σ c g_j y_j, v_j = c g_j x₀, then dsh_j += w v_j.
- **Decoupled:** dsh_j += (w x₀)(c g_j). This saves 2l₂ multiplies per path (12 per channel and edge) and the v
  registers. That is about 3 % of the kernel's arithmetic, perhaps 0.06 ms on the A100.
- **Registers:** the 96-register peak comes from an l₁ = 1 path with l₂ = 3 (3 u and 7 v live), not from an l₁ = 0
  path. Decoupling alone does not lower it.

**With dual Horner,** which frees the 12 basis registers, the count gives about 86 registers, still above the
80-register occupancy step (5 → 6 blocks). The factored kernel capped at 80 spilled 80–224 bytes and ran 1.243×.

**Screen:** render both as options and read `ptxas` registers and spills for sm_80 and sm_89. Run lab timings only if
80 registers fit.

## Own: mirror-pair w′ backward (bound only; prior negative)

**What it shares.** With `SYMMETRIC_PAIRS` (default on the Ada and every A100), edges i→j and j→i read the same
(min, max) slab at the same r, so their w and w′ rows are bitwise equal. One warp per undirected pair would read one
slab and do one w, w′ evaluation for two TP backwards:
- dr is shared: dr = Σ (wg_ij + wg_ji) w′, one FMA per weight instead of two.
- dsh stays per directed edge, using Y(−r̂) = (−1)^l Y(r̂).

**Savings:** half the slab traffic, and about 14 % of the A100's arithmetic (5.5 of about 40 operations per weight
and directed edge).

**Costs: the receiver order.**
- Today each receiver's dout row is read once per run of about 25 edges.
- With pairs, the mirror edge's dout row (5,120 floats, 20 KB) is a gather from a 102 MB array that fits neither the
  A100's 40 MB L2 nor the Ada's 48 MB.
- That adds about 3.7 GB of gathers per call, against half a slab saved per pair.
- The slab-ordered edge sort lost 1.51× on the RTX 5060 Ti from losing this order.

**Not `rad-mirror-edge`.** That idea shared the research engine's radial MLP between mirror edges, where the per-edge
radial work was cheap and the pairing overhead dominated. Here the shared work is a 30 KB slab and 11 FMAs per weight.

**Bound** (if wanted):
1. Each warp processes two consecutive edges with one slab load and one w, w′ evaluation, treating the second as the
   mirror (timing only).
2. The same, with the second edge's sender, receiver and dout taken from the real mirror permutation.

## Own: later candidates

- **tf32x3 GEMM entry** (A100 plan). 23 glue kernels whose names contain `clone` and `tf32x3_mm` take 348.9 µs per
  call on atom-level shapes: contiguous copies, slices, transposes and scales around the CUTLASS custom op. An entry
  point with row strides, column offsets and a scale epilogue could remove most of them. First attribute each kernel
  in the compiled graph's output code, since some may be real arithmetic rather than layout. These kernels are
  separate from round 1 #1's 260 µs.
- **Interaction 2's table inside the tensor-core forward.** `radial_table_forward` [9, 624, 6, 66] takes 207.1 µs, and
  the forward then reads a (97 MB). The forward's staging could evaluate a from the L2-resident 8.9 MB table instead,
  with the density accumulated per receiver: about 0.15–0.2 ms. The kernel already runs one block per SM (168
  registers), so this comes after idea 1, which may free registers.
- **Geometry and force epilogue.** Outside interaction 1, about 525 µs of edge glue remains: the geometry forward
  142.5 µs, its backward 110.2, scatters 66.5 + 36.9, the force assembly 64.9, and smaller kernels. A hand-written
  kernel taking both interactions' dr and dsh could save about 0.2 ms. Inductor already fuses within each graph
  segment, though, and stress and virial correctness make it costly to qualify.

## Own ideas discarded

| Idea | Why not |
|---|---|
| Low-rank radial weights | `rad-low-rank`: rank 65 of 65 |
| eSCN / SO(2) convolutions | no gain at sh l ≤ 3 with L = 1 features; the factored kernel already uses the CG sparsity |
| Pair-major (slab-ordered) forward | atomics on 5,120-wide messages per receiver; the slab-ordered backward already lost 1.51× |
| Reassociating interaction 2 like the first | its sender features depend on the environment, so per-slot sums do not factor |
| Dropping near-cutoff edges | changes the model |
| Overlapping independent branches on side streams | `lin-concurrent-branches` rejected |

## Proposed order

1. **Free checks:**
   - `ptxas` registers for idea 5;
   - the degree-3 share for idea 4, on leto's CPU;
   - the attribution of the glue kernels for ideas 7 and 10, from one compile.
2. **One A100 lab job** (about $0.3): idea 1's timing variant and range probe, and idea 2's two falsifiers. The Ada
   counterparts run on leto after the idle check.
3. **Build** what the bounds support, starting with ideas 1 and 2. Idea 3 only after 2 wins.

## Results (5 November 2026, branch `fastmace/round2-screens`)

Kernel lab on Benchmark_loup's captured calls (OMAT-0 medium, SrTiO3 10×10×10), each variant against the GPU's
default kernel in the same process; Modal A100-SXM4-80GB (profile `second`), leto's RTX 4000 Ada (NAS
`mace_cueq_oeq`, idle check), the local RTX 5060 Ti (WSL `mace_cueq_oeq`). Records: `results/round2-screens-20261105/`.

**Free checks:**
- **Degree 3 (idea 4): dead.** By the fit's own criterion (value and slope within TOLERANCE 1e-7 of the degree-5
  series), 6.7 % of (pair, piece)s could drop to degree 3, but they carry 0 % of the edges for all pairs and 0.09 %
  for any pair (`tools/degree3_share.py`); the median (pair, piece) misses by 192× (interaction 2: 290×). Degree 4
  covers at most 20 % of the edges. The slope criterion binds.
- **ptxas (idea 5):** Horner frees no register at the uncapped 96 (the compiler spends them), but fits the cap of 80
  without spills on sm_80 with the l₁ = 0 paths (4 bytes without), against 136 bytes for the factored kernel capped
  at 80.
- **Glue attribution (ideas 7 and 10)** from the A100 profile's kernel names and shapes:
  - 36 Triton kernels fused around `tf32x3_mm` take 521 µs. 140 µs of that is the species skip (round 1 #1).
    Pure layout (the `[N, 2048]` features sliced and copied into `[N·dim, 128]` per irrep, and `cat` back) takes
    124.5 µs. Residual adds and per-node divisions fused with such copies take 229 µs, the backward's transposed
    copies for weight gradients included (79.8 µs). A strided-batched GEMM entry (batch over irrep components,
    row stride 2048) with a row-scale and add epilogue would remove most of the 354 µs: about 1.5–2 % of the
    A100 call, at the cost of a CUTLASS rebuild.
  - Edge glue outside the custom kernels, 645 µs in 12 kernels: the geometry forward 142.5, its backward 110.2,
    `gather_radial_table_backward_slice_backward` 108.5 (interaction 1's table backward on the outer-sum
    backward's grad_a, padded from 64 to 65 columns), scatters 66.5 + 36.9, the force assembly 64.9.

**Screens:**

| Idea | A100-SXM4-80GB | RTX 4000 Ada | RTX 5060 Ti | Verdict |
|---|---|---|---|---|
| 1. 3×FP16 m16n8k16 forward (exponent-shifted, scaled per slice and per edge) | 1.197 (104 B spills); its big·big product alone 0.964 | not run (leto, murphy busy) | 1.07 (TF32 TC base) | **Rejected:** the MMAs are not the limit; even the one-product bound gains 3.6 % |
| 2. Outer-sum backward run walk (16 edges, 1 warp) | 0.802 | 1.03–1.11 | 0.94 | **A100 candidate** (the RTX GPUs sit near their DRAM floor, copy 0.80 / 0.73) |
| 5. Horner in the w′ backward | 0.997 | **0.975** | 1.053 (0.977 capped at 80) | Ada candidate |
| 5. Horner + l₁ = 0 scalar paths | **0.940** | 0.987 | — | A100 candidate |
| 5. l₁ = 0 scalar paths alone | 1.036 | 1.001 | 1.003 | no |
| 5. Horner capped at 80 registers | 1.059 | 1.000 | — | no |
| FACTORED without BLOCK_LDG (the RTX 5060 Ti's default is unfactored) | — | — | **0.929**; + Horner capped at 80: **0.887** | 5060 Ti candidate |

Accuracy against the default kernel: Horner dr 5.2–6.6e-7 relative (the factored regrouping alone 3.7e-7), dsh
1.8e-7; the walk's grad_a 1.0e-7 against a float64 reference (base 1.8e-7). The forward variants' odd-l message
groups differ by up to 24 % of their maxima even between the SIMT and the TF32 kernel: SrTiO3 is centrosymmetric and
those messages cancel, so only the even groups (1e-6) measure the product's rounding.

The candidates are module switches (`wprime_backward.HORNER`, `SCALAR_PATHS`, `MAX_REGISTERS`;
`edge_outer_sum.BACKWARD_WALK`), off until the force gate and MD-step A/Bs in both build orders
(`tools/modal/round2_ab.sh`) decide them per GPU (decision 66).

**New ideas from these results:**
- **Fewer warps in the A100 tensor-core forward.** The production kernel spills 40 bytes at the 168-register bound
  of 384 threads; 8 or 10 warps allow 255 or 204 registers (lab `tc-warps8`, `tc-warps10`).
- **The table backward inside the outer-sum walk.** dr from the grad_a row in registers and the edge's derivative
  row (L2-resident 10 MB table): no grad_a write (96 MB) nor the 108.5 µs padded pass. Every GPU, the
  DRAM-bound Ada first. Lab bound `walk-dr` against `walk+table-bwd`.
- **Per-component strided products instead of the relayout copies (A100, idea 7 without a cubin rebuild).** The
  CUTLASS entry (`cutlass.tf32x3.mm_into`) already takes a row-major A with any leading dimension and a row- or
  column-major B. For an irrep of dimension `dim`, BlockLinear's left operand `[N·dim, 128]` is a copy of `dim`
  strided `[N, 128]` views of the `[N, 2048]` features (lda 2048), all multiplied by the same weight. `dim`
  products on the views, each writing a strided slice of the output, need no copy and keep the same tiles
  (79 × 2 = 158 ≥ MIN_TILES). The backward's weight gradient sums the `dim` products' `xᵀ g`. Up to the 124.5 µs
  of pure layout plus the copies under the 229 µs of fused adds and divisions; it costs more launches (up to 7
  per irrep) and a change in `rewrites/linear` (rows are built per irrep there).

### MD-step A/Bs and the defaults (5 November 2026, night)

`tools/modal/round2_ab.sh`: the force/stress gate with the candidate's module constants in the production process,
then `tools/ab_rewrites.py` in both build orders (abab and abba, 8 blocks of 200 steps after 200 warm-up, 1 fs,
skin 0.5 Å) on Benchmark_loup's SrTiO3 10×10×10 (5,000 atoms) and NbCl5, OMAT-0 medium, float32. Sides a: the GPU's
default plan at that commit. Modal A100-SXM4-80GB (profile `second`), leto's RTX 4000 Ada (idle check, NAS
`mace_cueq_oeq`), the local RTX 5060 Ti (WSL `mace_cueq_oeq`). b/a below 1 is faster.

| Candidate | GPU | SrTiO3 ab / ba | NbCl5 ab / ba | Gate | Default |
|---|---|---|---|---|---|
| HORNER + SCALAR_PATHS | A100-SXM4-80GB | 0.986 / 0.983 | 0.994 / 0.978 | A100 exception only | **on** |
| BACKWARD_WALK | A100-SXM4-80GB | 0.995 / 0.987 | 0.995 / 0.996 | A100 exception only | **on** |
| HORNER | RTX 4000 Ada | 1.004 / 1.001 | 1.000 / 1.000 | pass | off |
| FACTORED, no BLOCK_LDG, HORNER, 80 registers | RTX 5060 Ti | 0.979 / 0.982 | 0.986 / 0.989 | 12/12 | **on** |
| FUSED_FIRST_BACKWARD | RTX 5060 Ti | 0.962 / 0.963 | 0.972 / 0.969 | 13/13 | **on** |
| FUSED_FIRST_BACKWARD | RTX 4000 Ada | 0.962 / 0.963 | 0.979 / 0.975 | 13/13 | **on** |
| FUSED_FIRST_BACKWARD | A100-SXM4-80GB | 0.969 / 0.966 | 0.983 / 0.984 | A100 exception only | **on** |
| + TABLED_FORWARD | RTX 5060 Ti | 0.945 / 0.945 | 0.948 / 0.955 | pass | **on** |
| + TABLED_FORWARD | RTX 4000 Ada | 0.949 / 0.944 | 0.978 / 0.968 | pass | **on** |
| + TABLED_FORWARD | A100-SXM4-80GB | 0.966 / 0.962 | 0.970 / 0.982 | A100 exception only | **on** |

"A100 exception only": OMAT-0 NbCl5 stress misses the 2× float32 limit as the A100 default plan already does (the
accepted exception, `docs/validation/omat0-a100-optimisations.md`): 1.55–1.60e-7 eV/Å³ against the default's 1.8e-7.
Every other check passes.

The first fused A/Bs (`b70bcad9`) gave 0.997–1.004 everywhere: the fused branch sat in `first_table_region`, which
only the memory-rescue checkpoint calls. A probe counting kernel launches found none; `b4f37680` routes the default
tabled path through `fused_first_message`, after which the kernels launch once per step and the numbers above follow.

Lab, A100 (`38eae731`): 8 or 10 warps in the tensor-core forward 1.160 / 1.109 (rejected); the walk's chunk and warp
sweep (16 edges and one warp best, 0.813); `walk-dr` (the fused backward's bound) 464 µs against 872 µs for the walk
and the table backward apart.

Defaults: `03125b0c` (Horner, scalar paths, register cap, walk, the 5060 Ti's factored kernel), `8466ac92` and
`2e9a3878` (the fused first interaction). Next: a fresh A100 profile at `2e9a3878`.


### A100 profile at `2e9a3878` and the next ideas (5 November 2026)

Modal A100 80GB PCIe (`omat0-a100-profile-2e9a3878`; earlier profiles ran on the SXM4): Benchmark_loup 2.50 µs per
atom and step (NVE), 2.47 (Langevin); eager force call 12.33 ms. Largest kernels per call:

| Kernel | µs | Note |
|---|---|---|
| `fused_folded_forward` (interaction 2) | 3,722 | 384 threads, 168 registers, 151 KB shared: one block per SM |
| `oeq_projected_backward_wprime` | 3,119 | 128 threads, 96 registers, 31 % occupancy |
| `tabled_outer_sum_backward` | 459 | new; replaces about 870 µs |
| CUTLASS `fm_tf32x3` GEMMs | about 1,760 | over 37 launches |
| `tabled_outer_sum_forward` | 391 | new; replaces about 410 µs |
| cuEq symmetric contraction (forward and backward) | 409 | |
| `radial_table_forward` (interaction 2's `a` for the fused forward) | 204 | |

Checked and dropped:
- **Launch shapes of the tabled outer sum** (`tools/tabled_sum_lab.py`, RTX 5060 Ti): 8–64 edges per program and one
  warp stay within noise of the defaults (0.95–1.01); 2 or 4 warps 1.26–2.64. The kernels read about 1.5 KB of
  Chebyshev coefficients per edge from L2, which no launch shape changes.
- **Two fused-forward blocks per SM** through more parts (half the W slice in shared memory): occupancy is bound by
  registers, not shared memory: 384 threads × 168 registers fill the 64 K register file, so two blocks would need
  6 warps each, the same 12 warps.

Open:
- **Interaction 2's table inside the fused forward** (TABLED_FORWARD for the second interaction): `a` evaluated from
  r in the kernel instead of `radial_table_forward`'s [E, 65] output (204 µs plus its read). The grid reads each edge
  row in 8 blocks (4 channel blocks × 2 parts), so the evaluation must be shared or cheap; bound it first in the lab.
- **The w′ backward's value-table traffic.** Its w table for interaction 2 is about [pairs, pieces, 6, 512] floats;
  each edge loads 6 × 512 × 4 B = 12 KB, about 4.6 GB per call if nothing hits L2, which alone is about 2.4 ms at
  the A100's bandwidth. Measure its DRAM bytes (Nsight Compute) before ideas: a w table ordered so that a run of
  receivers touches fewer lines, or TABLE_W off for interaction 2 on the A100 where FP32 arithmetic is cheaper than
  that traffic.

### The w′ backward's cost, attributed (A100 80GB, Modal, `02bf534c`)

Kernel lab, Benchmark_loup interaction 2 (`results/round3-profile-20261105/round3-a100-wprime`), each variant in the
same process. The falsifiers (`FF-*`) are built on the factored kernel without Horner (2,996 µs); the default
(factored + Horner + scalar paths) takes 2,825 µs.

| Variant | µs | Against factored |
|---|---|---|
| no table evaluation (`FF-no-table`) | 2,374 | 0.792 |
| every edge on one table slab, perfect L2 hits (`FF-one-slab`) | 2,837 | 0.947 |
| no atomics (`FF-no-atomics`) | 2,898 | 0.967 |
| no x loads (`FF-no-xload`) | 2,967 | 0.990 |
| edges sorted by table piece (`B-piece-sorted`, on the default) | 3,314 | 1.173 of the default |

- **Table traffic: refuted.** The value-table traffic idea (12 KB per edge from DRAM) does not hold: the table stays
  in L2, and perfect locality gains about 5 %. The table's evaluation costs about 0.6 ms, and Horner has already
  taken part of that.
- **The rest is the tensor product's FP32 arithmetic.** About 86 CG terms × 3 FMAs × 128 channels × 374k edges is
  about 25 GFLOP, 1.3 ms at the A100's FP32 peak, so the kernel runs at about half of peak.
- **`F-no-tp` failed to apply** (its pattern predates the factored body).

Idea left from this, not started: per edge and path, form `M_io(y) = Σ_j C_ijo y_j` once, so that
`u_i = Σ_o M_io g_o` costs `nnz(M)` FMAs per channel instead of one per CG term. It pays only for paths whose CG
terms outnumber M's nonzeros (l2 = 2, 3); `v_j = Σ C_ijo x_i g_o` is bilinear in channel quantities and does not
shrink. First count terms against `nnz(M)` per path of OMAT-0 medium; build it only if the FMA count drops by more
than about 15 %.

**Counted and dropped (5 November 2026):** CG terms against `nnz(M)` per path (e3nn `wigner_3j`, l1 ∈ {0, 1},
l2 ≤ 3, l3 ≤ 3): 134 terms against 116. The saving applies only to `u`, one of the three FMAs per term, so it is
about 4 % of the arithmetic before forming M, below the 15 % threshold. Paths: (0,l,l) and (1,0,1), (1,1,x) equal;
(1,2,1) 11→9, (1,2,2) 16→14, (1,2,3) 21→19, (1,3,2) 21→15, (1,3,3) 26→20.

## Round 4: the fused forward's staging (5 November 2026)

The owner asked to try interaction 2's table inside the fused forward and to look for new ideas. Records:
`results/round4-staging-20261105/`.

### Interaction 2's table inside the fused forward: bounded, not built

The forward grid reads each edge's `a` row in 8 blocks (4 channel blocks × 2 parts). Evaluating `a` in the kernel
would read the edge's Chebyshev coefficients instead (6 terms × 64 columns = 1.5 KB against `a`'s 256 B) in every one
of those blocks: about 8 × 374k × 1.5 KB = 4.5 GB of L2 traffic per call against 0.77 GB now, plus 8 × 6 × 64 FMAs per
edge, to save the 204 µs table kernel. That loses at any L2 bandwidth, so it was not built. The table kernel's own
launch shapes (`tools/radial_table_lab.py`, A100 80GB PCIe): edge blocks 8–64, column blocks 16–64, 1–8 warps; the
best (8 edges, 2 warps) 0.925 of a 271 µs lab call, about 20 µs. Not pursued.

### Where the A100 forward's time goes (kernel lab, A100-SXM4-80GB, `9f5c02cf`)

| Variant | µs | Ratio |
|---|---|---|
| base (tensor cores, mask split) | 3,706 | 1.000 |
| no `a` loads (`F-no-aload`) | 3,334 | 0.900 |
| no x loads (`F-no-xload`) | 3,102 | 0.837 |
| no sh loads (`F-no-shload`) | 3,581 | 0.966 |
| no loads at all (`F-no-loads`) | 2,749 | 0.742 |
| no output stores (`F-no-out`) | 3,458 | 0.933 |
| no a·W product (`F-tc-no-product`) | 2,037 | 0.550 |
| no tensor product (`F-no-tp`) | 2,411 | 0.650 |

The staging loads cost a quarter of the kernel. The 16 scalar `a` loads per lane and batch read the row-major `a`
transposed, and the 4 scalar sh loads read 4-byte pieces.

### Staging double-buffered with cp.async (`PREFETCH`): rejected on the A100

While a warp computes a batch, `cp.async` copies the next batch's `a` and sh rows (`"x"`: also the sender rows of x)
into a second shared buffer. Bitwise. Kernel lab, forward alone:

| Variant | GPU | Ratio |
|---|---|---|
| tc-prefetch | A100-SXM4 | 1.071, 1.038 (56 B spills) |
| tc-prefetch-x | A100-SXM4 | 1.152 (240 registers) |
| tc-prefetch | RTX 5060 Ti | 0.861 |
| simt-prefetch | RTX 5060 Ti | 0.930 |
| simt-prefetch-x | RTX 5060 Ti | 1.110 (6 warps) |
| simt-prefetch | RTX 4000 Ada | 1.025 |

The A100 kernel runs 12 warps at 168 registers; the second buffer costs spills or warps, and 12 warps already hide
the load latency. Kept off everywhere (`PREFETCH_GPUS` empty).

### Vector staging (`VEC_SH`, `VEC_A`) and the W slice in fragment order (`TC_WFRAG`)

- `VEC_SH`: sh staged with float4 loads (4 vector loads per lane and batch instead of 16 scalar ones). Bitwise.
- `VEC_A`: `a` staged with float4 loads from rows padded to a multiple of 4 floats into an edge-major tile,
  XOR-swizzled per 4-float group so the mma fragment reads stay conflict-free. The radial table writes the padded
  rows itself (`a_columns`); any other `a` is padded by a copy. Bitwise; needs the TF32 product and `used % 32 == 0`.
- `TC_WFRAG`: the tensor-core W slice stored in mma.sync fragment order, `[paths][KP / 8][2][32 lanes][4]`, so each
  lane loads its four W values with one float4 shared load instead of four scalar loads. Bitwise.

Kernel lab (forward alone):

| Variant | GPU | Ratio |
|---|---|---|
| sh4 | A100-SXM4 | 0.984 |
| tc-a4 (a and sh) | A100-SXM4 | 0.903, 0.905 (the bound without any `a` loads: 0.900) |
| tc-wfrag | A100-SXM4 | 0.965 |
| tc-a4 + wfrag | A100-SXM4 | 0.860 |
| sh4 (tensor-core kernel) | RTX 5060 Ti | 0.865 |
| VEC_SH on the plan's SIMT kernel | RTX 5060 Ti | 0.905 |
| tc-wfrag, tc-a4 + wfrag | RTX 5060 Ti (tensor cores forced) | 0.996, 0.894 |
| sh4 | RTX 4000 Ada | 1.021, 1.005 |
| simt-a4, simt-a4 + sh4 | RTX 4000 Ada | 1.070, 1.095 |

MD A/B (SrTiO3 10×10×10 and NbCl5, both build orders): VEC_SH on the RTX 5060 Ti 0.977 / 0.977 and 0.984 / 0.984
(`~/fm-runs/round4-vecsh-5060ti`). A100-SXM4-80GB (Modal), force gate as the plan (NbCl5 stress 1.49e-7 and
1.57e-7, the accepted exception), GPU test passes:

| Switches | SrTiO3 b/a (ab / ba) | NbCl5 b/a (ab / ba) |
|---|---|---|
| VEC_A + VEC_SH | 0.973 / 0.976 | 0.976 / 0.981 |
| VEC_A + VEC_SH + TC_WFRAG | 0.960 / 0.954 | 0.970 / 0.969 |

Defaults (decision 66): VEC_SH on every A100 and the RTX 5060 Ti; VEC_A and TC_WFRAG on every A100.

### Tried and dropped

- **x in a per-warp shared tile (`xs`)**: frees 9 registers (168 → 159) but does not launch on the A100
  (`CUDA_ERROR_INVALID_VALUE`; it adds 4 KB of shared memory per warp to the plan's). The registers would not
  add a warp at 12 warps per block anyway. Dropped.
- **Cache hints**: x gathers through `__ldg` or `ld.global.nc.L1::evict_last`, `a` / sh through
  `L1::no_allocate`. Bitwise. A100-SXM4: x-ldg 1.021, x-el 1.021; tc-a4 with hints 0.912–0.923 against 0.903
  without. RTX 5060 Ti: 0.996. Rejected.
- **Atom order for the x gathers**: SrTiO3 10×10×10 in ASE repeat order, Morton (Z-order) and a seeded shuffle
  (`results/round4-staging-20261105/orders/`). RTX 5060 Ti forward 6,724, 6,737 and 6,679 µs. The 10 MB of x stay
  in L2 in any order, so locality does not matter. Dropped.
- **Fewer coefficient multiplies in the tensor product**: the falsifier without any coefficient multiply gives
  0.961 (RTX 5060 Ti). Dropped.
- **3xFP16 GEMMs instead of CUTLASS 3xTF32** (`tools/fp16x3_gemm_bench.py`, A100-SXM4-80GB): A's rows and B's columns
  scaled by powers of two, one FP16 GEMM over 3K, `[2^11 Ah, Ah, 2^11 Al] @ [Bh; 2^11 Bl; Bh]`, FP32 output. For the
  profile's shapes, the GEMM alone takes 0.49–0.66 of the 3xTF32 kernel's time (5000×2048×128: 88 against 143 µs).
  The unfused split (row max, split, concatenation) costs 78–138 µs, as much as the GEMM saves. The error relative to
  |A| |B| is 1.1–2.0e-6, against 1.1–2.7e-7 for 3xTF32 and 0.8–4.2e-7 for FP32: the FP16 MMA accumulates more
  coarsely. Even fully fused it would save at most about 0.8 ms of the 12.3 ms step, at 6–10× the GEMM error.
  Rejected on accuracy (FP32-equivalent rule).

### The tensor product on tensor cores (`TC_TP`): correct, slower, rejected

The a·W product's C fragments hold w as (channel, edge) tiles. Give each lane's edge pair 2 tq, 2 tq + 1 the k columns
tq, tq + 4. Then `(w x)[channel, (edge, m1)]` is directly the A fragment of a second 3×TF32 product against
`S_e[m1, m3] = Σ_m2 CG sh_e[m2]`, with a receiver's edges as its k dimension. The message stays in mma accumulators
until the receiver changes. The kernel goes path by path (8 product registers live) with a per-path receiver
cursor. Built as `fused_forward.TC_TP` (off), with `TC_TP_SPLIT = "round"`.

- **Correctness:**
  - w is bitwise the tensor-core kernel's.
  - The message is within 4–6e-7 rms and 1.5e-6 max (relative) of the FP32 TP, with mask split 7e-7 rms.
  - Checked for both layouts, the block layout, C = 128 and MACE's full path set.
  - The real SrTiO3 call's 1o and 3o groups show large *relative* differences only because their values are
    rounding noise: odd-l sums over a centrosymmetric lattice cancel to about 1e-5 against 8. The absolute
    differences, 2–9e-6, are the same as in the other groups.
- **A100-SXM4 kernel lab:** 1.276, at 162 registers without spills; the bound without any TP is 0.617.
  RTX 5060 Ti (tensor cores forced): 1.64.
- **Why it loses:** each A element serves one mma triple (N = 8 outputs, K = 8 edges). Its preparation (multiply
  by x, round-to-nearest split of both operands) costs about as much as the FP32 TP's terms it replaces. With the
  per-path a-fragment reloads and segment loops on top, the kernel is slower.

## Round 5: the A100 profile at `5b6c5550` (5 November 2026)

Profile: `results/omat0-a100-profile-5b6c5550/` (Modal A100 80GB PCIe). Benchmark_loup 2.38 µs per atom and step
(NVE) and 2.35 (Langevin), against 2.50 / 2.47 at `2e9a3878`. The eager force call takes 11.72 ms in 223 kernels:

| Kernel or bucket | µs |
|---|---|
| fused forward | 3,175 |
| w′ backward | 3,109 |
| CUTLASS GEMMs (41 launches) | 1,811 |
| Triton atom kernels (55) | 1,035 |
| tabled outer sums | 849 |
| symmetric contraction | 402 |
| edge geometry and radial glue | 307 |

Records: `results/round5-20261105/`.

Tried:
- **Edge spherical harmonics in Triton (`EDGE_SH`): adopted** on the RTX 5060 Ti, the RTX 4000 Ada and every A100. MD step
  0.96–0.998, faster in both orders on all three GPUs; the gate passes.
- **The species GEMM on the A100, again: rejected.** SrTiO3 gains (0.979–0.996), but NbCl5 loses 2.6–5.8 %.
- **w from the value table in the forward: rejected.** The lab bound with the table's real footprint is 1.333 of the
  forward on the A100; even with the table cached it is 0.988.

Left from the profile, by size:
- **GEMM layout glue, about 0.34 ms of pure copies** (`clone_slice_*tf32x3_mm` 162 µs, `cat_view` 182 µs) between
  `[N, 16·128]` and `[N·dim, 128]`. The cubin's gathered and scattered entry (`fm_tf32x3_species_rr_64x64`, one
  species) can read and write those rows through index arrays without a rebuild. It needs a custom op that takes
  the full feature tensor, and consumers that accept the block layout (cuEq's symmetric contraction does not).
- **The ZBL pair term in Inductor:** 48 + 27 µs plus a 37 µs scatter; a fused kernel each way.
- **MACE's head selection `[arange, heads]`:** its backward sorts (`IndexBackward0`, about 70 µs with the forward
  gathers), because the readouts run in eager frames between compiled graphs.
- **The interaction-1 outer sum's 61 MB zero fill (32 µs) and its atomics:** they disappear if each program owns
  whole receivers.

### Round 5, continued: the profile at `2a016fd4` and the w′ reduction (5 November 2026)

Profile with EDGE_SH: `results/omat0-a100-profile-2a016fd4/` (Modal A100-SXM4-80GB). Benchmark_loup NVE 2.35, 2.51,
2.44 µs per atom and step (three fresh processes; the spread is the card's, the median is close to `5b6c5550`'s
2.38 on the PCIe card). The eager force call is 11.43 ms in 217 kernels. The edge spherical harmonics now take
13.9 µs forward and 27.9 µs backward, against about 228 µs in Inductor. What is left of the edge glue (the
normalisation's backward and the force scatter 54 µs, ZBL 47 + 27 µs and two 32 µs scatters, the vectors 10 µs) is
about 0.22 ms, 2 % of the call. That is the ceiling of the review's force-epilogue fusion (priority 2); a fused
ZBL kernel each way is the part of it worth building (about 0.1 ms).

**The w′ backward's end-of-edge reduction.** Every lane shuffles all 16 edge-gradient components and dr down 5
levels: 85 SHFL, 85 WARPSYNC and 170 FADD, about half of the kernel's static SASS (sm_120: 1,448 instructions
with it, 752 without). Kernel lab (`results/round5-20261105/runs/`):

| Variant | RTX 5060 Ti | A100-SXM4-80GB |
|---|---|---|
| default | 7,370 µs | 3,050 µs |
| butterfly reduce-scatter (`BUTTERFLY`; 21 shuffles; dsh and dr bitwise) | 0.965 / 0.963 | 0.989 |
| no reduction at all (falsifier) | 0.675 | 0.955 |
| dr reduced, dsh not (falsifier) | 0.680 | |
| dsh reduced, dr not (falsifier) | 0.998 | |

On the RTX 5060 Ti the reduction of the 16 components costs a third of the kernel. The butterfly removes two
thirds of its instructions (992 static) but only a tenth of its time. So the cost is the tail's latency or
issue, not the shuffle count. On the A100 the whole reduction is 4.5 %.

### The external review's six ideas (5 November 2026), assessed

1. **Several warps per edge (channels split across 2 or 4 warps): bounded.** Registers per thread come from the
   per-edge state, not from the number of channels a lane handles: 16 edge-gradient accumulators, the Chebyshev
   and slope terms, and the input-gradient slots of one channel, reused across the loop. Splitting the channels does
   not shrink it. It multiplies the per-edge preamble and adds a cross-warp reduction, and on the A100 lowering the
   registers (MAX_REGISTERS 80) made the kernel slower (1.059). Lab bound `F-split2` / `F-split4` (partials added
   with atomics): see the table below.
2. **W as packed scaled FP16 hi/lo pairs: not built.** Storage stays 32 bits per coefficient, as in TC_WFRAG's FP32
   slice, so the shared-memory footprint gains nothing. Decoding takes two conversions plus a rescale, or a second
   accumulator for the 2^11-scaled low products, against the split's three instructions now. The split's ceiling
   was 14 % of the forward before TC_WFRAG (`F-tc-no-wsplit` 0.862); it is re-measured on the current default below.
3. **One program per receiver/species run (no atomics, no zero fill): small.** The forward's zero fill is 31 µs, and
   its atomics write the same 61 MB that plain stores would. With about 25 edges per slot and 32-edge programs, most
   runs straddle at most one boundary. The ceiling is about 30–60 µs, 0.3–0.5 % of the step, below the review's
   0.1–0.25 ms. Kept on the list (above: "whole receivers"), after ZBL.
4. **The backward split into path-group kernels: not built.** It has the same premise as 1: fewer registers buy
   occupancy. On the A100 they did not (MAX_REGISTERS 80: 1.059). The receiver-owned and path-grouping
   attempts (ledger `conv-receiver-backward`, `conv-path-grouping`) lost to the per-edge kernel.
5. **Cartesian edge gradients inside the w′ kernel: small.** After EDGE_SH the SH backward is 28 µs, and the dsh
   tensor (24 MB) is written once and read once. Interaction 1's dsh must be contracted as well, so both
   kernels would need the Jacobian. Ceiling about 50 µs (0.4 %).
6. **A symbolic joint forward/VJP generator: not started.** The counted CG algebra (134 terms against 116 for
   nnz(M), about 4 % of the arithmetic) and the arithmetic bounds already measured (`F-tp-no-coef` 0.947–0.961) leave
   less than the 15 % threshold such a generator would need.

### Interface items from the `2a016fd4` profile (5 November 2026)

- **Head selection (`energy.HEAD_SLICE`, built, A/Bs running).** MACE's `values[num_atoms_arange, node_heads]`
  becomes `values[:, 0]` for one-head models (source edit of the pinned forward). It removes the IndexBackward
  radix sort, scan, histogram, memsets and gathers, about 98 µs per force call. Bitwise.
- **One-graph energy sums (`energy.GRAPH_SUM`, built).** `scatter_sum(..., data["batch"], dim_size=1)` becomes
  `sum(keepdim=True)`. Three single-address atomic scatters (37.9 µs + 5.5 µs of fills) become reductions. Forces
  are bitwise; the float64 energy's order changes.
- **Density sums (not built).** Two `[E, 1] → [5000, 1]` atomic scatters, 31–32 µs each. The edges are
  receiver-sorted, so a CSR segment sum would take a few µs, but the row pointer is built inside
  `streaming_message` and is not available to the density. Ceiling about 45 µs.
- **GEMM layout glue (design).** `[N, 16·128]` viewed as `[N·16, 128]` makes each irrep block's rows the index
  pattern `n·16 + base_l + m`. CUTLASS's gathered and scattered entry (`species_into`, one species) can read A
  through that pattern and write D straight into the concatenated output. Attributed on the A100:
  - pure A copies (`clone_slice_*tf32x3_mm`): about 160 µs;
  - output concatenations (`cat_view`): about 135 µs;
  - the other glue kernels fuse divisions and residual adds and would stay.

  Only ungrouped 1:1 blocks qualify. Grouped products concatenate segments along K, which a row gather cannot
  express, and `species_into` has no β (no accumulate). The backward needs the same gather and scatter for
  `grad @ Wᵀ`. Ceiling about 0.25 ms (2 % on the A100).
- **ZBL (not built).** The forward is already one Inductor kernel (47 µs) and the backward 27 µs; a hand kernel
  saves perhaps 30–40 µs.

### External ideas, third set (5 November 2026)

Assessed against the ledger and the round-5 falsifiers. RTX 5060 Ti w′ falsifiers (BUTTERFLY off as the base,
7.37 ms): no reduction 0.675, dr reduced but not dsh 0.680, dsh but not dr 0.998, butterfly 0.965, two warps per edge
0.954, four warps per edge 0.891. On the A100 no reduction is 0.955. Reducing one value costs almost nothing; the 16
dsh components cost the 32 %, and the butterfly's quarter of the shuffles recovers only a ninth of it.

1. **Producer/consumer warps in the A100 forward (MMA warps feed FP32 warps).** Parked. The forward is at the
   168-register cap with 12 warps and 151 KB of shared memory. Fewer warps lost more than they saved (8 / 10 warps
   1.160 / 1.109; pre-split W 1.362 because it cost a third of the warps). An exchange ring of w for 8 edges is
   about 40 KB per stage. The tensor pipe is not the limit: the big·big product alone is 0.964. Dropping the W split
   altogether is 0.862, but the split runs on the same issue slots as the FP32 consumer, so warp roles cannot hide
   it. The 12 warps already interleave both pipes. First check, if reopened: Nsight barrier-stall share and pipe
   utilisation of the forward. Without a large barrier share there is nothing to overlap.
2. **Cartesian gradient before the w′ reduction.** Promoted. The falsifiers above make it the strongest 5060 Ti
   lead: three reduced values instead of sixteen, with dr folded in through the unit vector, so dr's own reduction
   goes too. Lab falsifiers `F-cart3`, plus `F-redux16` and `F-cart3-redux` for idea 5, are running. They time a
   16×3 projection whose Jacobian is read from sh as a stand-in. A real Jacobian costs some tens of FMAs more per
   lane and edge. Production would need the edge-SH backward to take a per-edge 3-vector from interaction 2 next to
   interaction 1's dsh. It is rounding, not bitwise; gate and stress as usual.
3. **Radial-manifold compression.** Closed today by `tools/radial_rank_lab.py`, before this proposal. That test is
   the one proposed: OMAT-0, every species pair, a dense Chebyshev grid over [0.5, r_max). Errors are output-weighted
   on w = aW and on dw/dr, with an SVD of the values and scaled slopes. Every interaction is full rank. Rank 56
   still leaves 1.6e-5 relative w error against 3–6e-8 from float32 rounding, so rank ≤ 32 is far off (ledger
   rad-low-rank: MH-1 rank 65 of 65). The six-dimensional space per piece is real. TABLE_W exploits it in the
   backward. The forward cannot: a per-edge W·C read is about 30 KB against a's 256 B.
4. **Pre-split constant weights in the CUTLASS atom GEMMs.** Low. The same trade in the forward (TC_PRESPLIT) lost
   1.362, and the producer-side splits lost too (prec-presplit-epilogue, prec-producer-split). K = 128 GEMMs are near
   their memory time, and B is half the split work. Bound first, if ever: an OpMultiplyAdd (1×TF32) build of the same
   tile gives the ceiling of all split work; stop below 10 % of the CUTLASS GEMM time.
5. **Integer warp reductions (redux.sync).** Raised from speculative to a lab falsifier, because the cost scales
   with the number of reduced components. Fixed point: one redux.sync.max of the lanes' largest exponent, 2^25 head
   room, 32-bit sums. That is deterministic, with about 2^-25 of the warp's largest partial per term, comparable to
   FP32's error against Σ|g| (the gate decides). Timed as `F-redux16` and `F-cart3-redux`.
6. **Path-specific precision in the forward.** Rejected. The bound is 0.964 even with every path on one TF32
   product, and it is approximate arithmetic on top of the accepted NbCl5 stress exception.
7. **Static receiver schedule instead of the work counter.** Low. About 2,500 atomics per call on uniform SrTiO3.
   The counter's dynamic balance is what a static list would give up. Falsifier when idle: the same arithmetic with
   a precomputed list; stop below 5 % of the forward.
8. **Structure-aware plan selection at warm-up.** Not ours to adopt. Per-crystal tuning was ruled out by the owner
   (10 October 2026); decision 66 opens per-GPU defaults only. Timing-noise selection would also make a run's plan,
   and its rounding, irreproducible. Opt-in calibration exists (decision 55). This is an owner decision.

Separately, the multi-warp falsifier (idea 1 of the second set) did not fail on the 5060 Ti: four warps per edge
gave 0.891 with atomics. Its A100 bound is running. A deterministic version would combine the four partials in
shared memory in a fixed order.

### Correction: the lab's 5060 Ti w′ base lacked the register cap (5 November 2026, about 13:30 UTC)

`tools/omat0_kernel_lab.py` compiled every w′ variant without the plan kernel's per-GPU `--maxrregcount`
(`MAX_REGISTERS_GPUS`: 80 on the RTX 5060 Ti since round 2). On the 5060 Ti its "base" was therefore the uncapped
kernel. On thermalised inputs that kernel takes 7.88 ms against the plan's 6.77 ms; a cap of 255 (242 registers)
takes 9.17 ms. The cap is right; the lab was wrong. The A100 and the Ada have no cap, so their lab numbers stand.
Two more lab fixes the same day:

- the capture now follows 300 MD steps (`--equilibrate-steps`); before, it was the first steps from the ideal lattice;
- every timed launch now follows a 256 MB L2 flush (`--flush-l2-mb`); this changed nothing measurable.

Corrected 5060 Ti w′ numbers: plan kernel (cap 80, BUTTERFLY off) 6.78 ms. Relative to it:

| Variant | Ratio |
| --- | --- |
| butterfly | 0.984 |
| redux-pc | 0.966 |
| no reduction (falsifier) | 0.954 |
| four warps per edge | 1.131 |

So the end-of-edge reduction is about 4.6 % of the kernel, not 32 %. Consequences:

- the Cartesian projection (idea 2 of the third set) has at most a few percent of the kernel to win: closed;
- several warps per edge loses on both GPUs: closed;
- the REDUX MD A/Bs stand as measured: 5060 Ti 0.998 / 0.999 (SrTiO3) and 0.999 / 0.998 (NbCl5) against the
  butterfly default; Ada 1.009 / 1.008 and 1.011 / 1.008.

Every earlier 5060 Ti w′ lab ratio was taken against the uncapped base. That covers this round's butterfly, split
and reduction falsifiers. The MD A/Bs, which decide defaults, are not affected.

Forward prefetch re-screen on the RTX 5060 Ti (thermalised capture, 5 November 2026): base (SIMT, 167 registers)
6.26 ms, simt-prefetch 1.013 (40 B spills). The round-4 lab's 0.930 came from the ideal-lattice capture; closed on
the 5060 Ti.

Forward prefetch re-screen on the A100-SXM4 (thermalised capture, 6c2f2967, Modal): base 3.18 ms; tc-prefetch 1.255,
tc-prefetch-w10 1.258, tc-prefetch-x 1.384. All three tensor-core prefetch variants also give wrong outputs on
thermal inputs: out0..3 max|d| / max|a| of 1.4 to 6.4. Round 4 called them bitwise, but only on the ideal-lattice
capture, where every receiver has the same number of edges. On the 5060 Ti the SIMT prefetch stayed bitwise on thermal
inputs, so the defect is in the tensor-core prefetch path, probably its handling of uneven batches. PREFETCH is off on
every GPU (PREFETCH_GPUS empty); it must not be enabled with tensor cores until that is fixed and tested on uneven
neighbourhoods.

A100-SXM4 w′ lab and REDUX A/B (thermalised capture, 6c2f2967, BUTTERFLY off as on the A100): base 3.14 ms at 96
registers, no spills. Register caps: 72 1.201 (88 B local), 64 1.477 (128 B local), so the A100 stays uncapped.
butterfly 1.006, redux-pc 1.104, no-reduce 0.969 (the reduction is 3 %). REDUX MD A/B (b/a): SrTiO3 1.029 / 1.025,
NbCl5 1.022 / 1.018, so slower; REDUX stays off on the A100. The A100 w′ kernel keeps its round-4 configuration.

The leto CPU suite at 6c2f2967 (5143 passed, 2 failed) caught a name clash: GRAPH_SUM's forward edit registered
`_fastmace_graph_sum`, which replaced PolarMACE's graph-sums helper of that name in its compiled forward. It now uses
`_fastmace_batch_sum` (4bfe3307).

Correction (5 November 2026, later): the "wrong outputs" of the A100 prefetch variants were a kernel-lab bug, not a
kernel defect. With `VEC_A` (A100 default) the radial table writes a with rows padded to AS = 68 floats (K = 65); the
lab gave that a to variants rendered without `VEC_A`, which read rows of KA = 65 floats, so every atom differed (all
5,000 rows). tc-rerender keeps `VEC_A` and was bitwise. Lab fixes: a compacted a for such variants (a9074c5b), output
buffers cleared before each variant (c544221a), the base's w zeroed, and the rows of a mismatch reported. Confirmed on
the A100: tc-prefetch bitwise (1.253), simt and simt-prefetch 1.5-2.4e-6 (1.510 / 1.519). Production was never
affected (its gate against float64 passes). Earlier A100 lab correctness labels of variants without VEC_A (prefetch,
simt, fp16, presplit, round 4) are void; their timings stand. PREFETCH stays rejected on timing.

A100 stage bounds at 6c2f2967 (thermalised capture): forward 3.25 ms; F-tc-no-product 0.473, F-no-tp 0.613,
F-tc-no-mma 0.664, F-tc-1pass 0.677. The projection (1.71 ms) and the contraction (1.26 ms) add up: no overlap now.
One MMA pass costs 42 us, three 1.09 ms. Chain variants: tc-mma-free 0.999 (bitwise), tc-2acc 1.306 (144 B spills),
tc-3acc 1.504 (280 B). Producer/consumer microbenchmark (tools/pc_microbench.py) running on the A100.

Producer/consumer warps (third set, idea 1), `tools/pc_microbench.py` on two A100-SXM4 containers (Modal; 12 warps,
one CTA per SM, npl 5, KP 72, the forward's MMA loop against an FMA contraction). Run 1 (bc0a84ae), 6+6 warps with a
2-tile ring: 0.915 of serial at C/P 0.76, 1.07-1.31 above. Run 2 (975e560f), another container: 6+6x2 0.988 / 1.044 /
1.079 / 1.104 at C/P 0.61 / 0.75 / 0.90 / 1.07; 6+6x1 1.70-1.74; 8+4 1.14-1.39; 9+3 1.28-1.67; 4+8 1.08-1.28. Run 1's
win came with a slower serial base (P 2.96 against 2.46 ms in run 2). No spills anywhere; a 3-tile ring does not fit.
Each side keeps only part of the 12 warps for its own latency, which costs what the overlap gains. **Closed** by the
proposal's own rule (more than 10 % of serial).

TF32 + FP16 hybrid projection (`tc-hybrid16`, kernel lab, 3e4f9310): the 3xTF32 product with W_hi a_hi kept in TF32
m16n8k8 and both small products (W_lo a_hi, W_hi a_lo) in FP16 m16n8k16, shifted by 2^11 (exact) and unshifted on the
store; the k16 slots map to k0 + 8s + tq (+4) in both operands, so the sum is unchanged. 4 MMAs per 16 k instead of 6.
Needs KP % 16 == 0 (SKIP_ZERO_COLUMN, KP 64 for OMAT-0). Accuracy FP32-class (max|d|/max|a| 7e-7 to 1.7e-6, w
bitwise as it is the base's). RTX 5060 Ti with the A100 constants: 0.851 (255 registers, 8 B local). A100-SXM4
(Modal, `results/round5-20261105/runs/a100-hybrid16-rebuild-cost/`): **1.017** (base 3.60 ms, 167 registers and 40 B
local; hybrid 168 and 48 B at the 12-warp launch bound). The per-fragment FP16 packs (12 cvt per (g, m, 16 k)) and the
register limit eat the third of the MMA time it saves. **Rejected** on the A100.

Neighbour-list rebuild cost (`tools/bench/rebuild_cost.py`, block time = steps t_step + rebuilds t_rebuild, 200 blocks
of 4 steps, SrTiO3 10x10x10, OMAT-0 medium): RTX 5060 Ti skin 0.5: 44 rebuilds, 2.04 ms each, 0.11 ms per step
(0.45 %); skin 1.0: 13 rebuilds, 3.05 ms each, 0.05 ms per step, t_step unchanged (24.707 against 24.713 ms).
A100-SXM4: skin 0.5: 44 rebuilds, -0.46 ms each (noise), t_step 11.644 ms; skin 1.0: 13 rebuilds, +0.36 ms each,
t_step 11.586 ms. The profile's 11.35 ms per rebuild runs beside the force call (`overlap_rebuild`): a rebuild adds
nothing measurable on the A100, so neither a cheaper rebuild nor a larger skin can gain anything there. Closed.

## Round 6: skipping work around the big kernels (5 November 2026)

Two answers to `plans/research-prompt-20261105-skip-work.md` (ChatGPT, Claude). ChatGPT ranked a GPU cell list first
(rejected: `rebuild_cost.py` above finds no added time per rebuild on the A100) and angular-product reuse in the forward
(open: needs a SASS check whether ptxas already shares the products). Claude ranked (1) folding interaction 1's linear
chain, (2) the Langevin noise refill, (3) hoisting species-only and output-only kernels (about 0.2 ms), (4) small items.

**Noise draw-ahead (`langevin.NOISE_DRAW_AHEAD`, default on, 349f03e3).** The 5,000-atom batch refill (7.68 M normals,
61 MB) blocked the host for about 96 ms every 256 steps: the rebuild-cost blocks above (A100-SXM4) are 45 ms each and
141 ms at steps 256, 512 and 768. After each refill the next batch is now drawn on a background thread into one of two
pinned buffers (allocated on the host thread; the thread makes no CUDA call, so it cannot disturb a capture) and copied
without blocking. Reading or setting `RandomStream.generator` (restart, event snapshot, provenance record) undoes a
pending draw-ahead, so the stream, restart files and every number are unchanged (GPU test
`test_drawn_ahead_noise_is_the_same_stream`: Langevin and BAOAB rows bitwise ASE's, with mid-run generator reads).
A100 80GB PCIe (Modal), blocks of 256 steps, side b without draw-ahead: SrTiO3 10x10x10 b/a 1.0298 / 1.0276 (a 12.41 /
12.47 ms), NbCl5 1.0412 / 1.0410. Host-only and bitwise, so on for every GPU
(`results/round5-20261105/runs/a100pcie-noise-draw-ahead-ab/`).

**Interaction-1 linear fold (`species.FOLD_FIRST_DENSITY_LINEAR`, 4fea81e0).** With the generic fold, a density first
interaction's `linear` is composed into its species skip (`compose_species_skip`; the 1 / (d + 1) row scale between
them commutes) and replaced by the identity. Rounding class; the fold's self-check covers the composed skip. RTX 5060 Ti
(fold_product_linear on for both sides): SrTiO3 b/a 0.9525 (pairs 0.949-0.953). A100-SXM4 (Modal): SrTiO3 0.9784 /
0.9772 (a 11.15 ms, b 10.91 ms: 2.18 us per atom and step), NbCl5 0.9542 / 0.9561; force gate passes for NaCl and
SrTiO3, NbCl5 stress 1.569e-7 (ratio 2.34) is the A100's accepted exception at the same value as the default
(`results/round5-20261105/runs/a100-fold-first-linear-gate-ab/`). RTX 4000 Ada (leto): SrTiO3 0.9536 / 0.9545,
NbCl5 0.9197 / 0.9197, gate passes for all three (`rtx4000ada-fold-first-linear-gate-ab/`). **Default** (f1c5dcbc)
wherever the generic fold is on (decision 66: Ada and A100-SXM4, both measured).

GPU memory against Symmetrix-XL (RTX 5060 Ti, `results/vram-compare-rtx5060ti-20261105/`): on the 5,000-atom cell
FastMACE takes 1,381 MiB at 24.4 ms per step, Symmetrix-XL under LAMMPS 767 MiB at 50.8 ms.
