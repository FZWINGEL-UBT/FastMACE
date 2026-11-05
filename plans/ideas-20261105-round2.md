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
