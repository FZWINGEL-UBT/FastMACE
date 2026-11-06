# Idea 31, step 1: the glue around the 3xTF32 GEMMs (CPU analysis, 6 November 2026)

**Question.** Idea 31 asks how much of the Inductor glue and `cat` time around FastMACE's 3xTF32 atom GEMMs on the H200
could go into CUTLASS epilogues, or into output pointers and leading dimensions, *bitwise*. "Bitwise" means the same
float32 operations in the same order on the same values. The plan's figures, from the `27d27f21` breakdown, were about
209 µs of Inductor kernels and 148 µs of `cat` kernels.

This is the plan's CPU falsifier step. It is a reading of files only: no GPU job, no test suite and no Modal run.

**Sources:**
- **Trace:** `../sx012-profile-20261106/fastmace-h200-breakdown/breakdown/` (`kernel_sequence.csv`, `ops.csv`). FastMACE
  `d599cc70`, NVIDIA H200, OMAT-0 medium, float32 hybrid, Benchmark_loup (SrTiO3 10×10×10: 5,000 atoms and 373,760
  padded edges). This is one eager force call; its kernels take 5.75 ms.
- **Code:** `fastmace/kernels/triton/tf32x3_gemm.py` (`tf32x3_mm`, the custom op) and `fastmace/kernels/cutlass/tf32x3.py`
  (`mm_into`).
- **Modules:** `fastmace/rewrites/linear/modules.py` (`BlockLinear`, `join_rows`), `fastmace/rewrites/generic/modules.py`
  (`generic_interaction_forward`, `SpeciesSkip`), `fastmace/rewrites/mh1/modules.py` (`FirstInteractionReassociation`)
  and `fastmace/rewrites/precision.py`.
- **Plan record:** the plan's `describe()` in `summary.json`. Its relevant settings are `tf32x3_gemms`, `block_linear`,
  `grouped_linear`, `species_skip` and `fold_product_linear`. `SPECIES_GEMM` is off on the H200 and `SEGMENT_ROWS` is off.

## How the numbers were obtained (and their limits)

- **Totals at `d599cc70`:** 33 GEMM launches per force call (24 `rr` and 9 `rc`) take **866.4 µs**. Kernels whose names
  contain `tf32x3_mm` (Inductor glue) take **207.0 µs**. Triton `cat` kernels take **154.1 µs**, and aten
  `CatArrayBatchedCopy` takes 9.1 µs. The same sums over `27d27f21`'s breakdown
  (`../h200-defaults-20261105/explore/out/breakdown/`) are 865.8, 208.8 and 147.6 µs. The plan's "24 launches" counts
  only the `rr` launches.
- **Glue µs** are measured: every glue kernel is its own row in `ops.csv`.
- **GEMM µs are estimates.** `ops.csv` groups the `fm_tf32x3_*` launches by compiled graph, gives only the first launch's
  grid and has no per-launch input shapes, because the custom op has no aten operator. Each launch's M×K×N comes from
  the modules (below). Each group's measured total was split as 2.8 µs per launch plus a share proportional to FLOPs. All
  ten groups then come out at 18.8–19.7 µs per GFLOP. That consistency checks the shape assignment, but the per-launch
  values are not measured.
- **No Inductor source.** The generated code is not in the repository. Kernel names (`triton_poi_fused_<ops>`), input
  dims and their position in the sequence identify what each glue kernel computes. The GPU step has to confirm the
  bodies (see "Next step").
- **`#n`** is the kernel's index in `kernel_sequence.csv`.

**Layouts the analysis relies on.** cuEquivariance's core layout is `ir_mul`: the l block of an atom is
(2l+1)·128 contiguous floats, component m outermost. A `BlockLinear` product for block l therefore has rows r = (a, m),
that is M = 5000·(2l+1), and `join_rows` concatenates the blocks into [5000, Σ(2l+1)·128].

Row r of block l lands at `dst[a·ld + off_l + (r mod (2l+1))·128 + c]` with a = r div (2l+1). An atom's rows are one
contiguous run, so the block is a row-major [5000, (2l+1)·128] matrix with leading dimension `ld`. A per-row address
computation in the epilogue, or a per-m batched launch with pointer `off_l + m·128` and `ld`, writes it in place.

## (a) One row per GEMM launch

Forward (17 launches, 428.7 µs):

| GEMM | #n | graph | model op | M×K×N | µs | glue before (µs, what) | glue after (µs, what) | cats consuming the output |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| F1 | 14 | fwms | int. 1 `linear_up` (128x0e) | 5000×128×128 | 7.5 m | – (A is the node embedding, cuBLAS #8) | none | none. **The output looks unread:** `fused_first_message` reads `species_feats`, not `node_feats`. Verify, then drop it (7.5 µs, not glue) |
| F2 | 21 | fkect | int. 1 message l=0: reassociated convolution with `linear` folded in, `summed[(a,m),(slot,k)] @ (γ x Wᵀ)` | 5000×192×128 | 7.5 | #20 3.3: A copy of the tabled outer sum (transpose slot↔m); #19 1.6: B operand γ·x·Wᵀ, shared by F2–F5 | #31 2.6: slice of the cat ÷ (density+1), the A of F6 | #28 `cat_view_7` 19.2 (F2–F5 → [5000, 2048]) |
| F3 | 23 | fkect | same, l=1 | 15000×192×128 | 17.0 | #22 8.3: A copy | #34 4.9: ÷(d+1) copy, the A of F7 | #28 |
| F4 | 25 | fkect | same, l=2 | 25000×192×128 | 26.5 | #24 13.1: A copy | #36 8.4: the A of F8 | #28 |
| F5 | 27 | fkect | same, l=3 | 35000×192×128 | 36.0 | #26 15.1: A copy | #38 12.3: the A of F9 | #28 |
| F6 | 32 | fdamfw | int. 1 `skip_tp` (`SpeciesSkip`: 3 species' matrices side by side), l=0 | 5000×128×384 | 12.0 | #31 (above) | #40 56.5 for F6–F9: own-species select Σ_s y_s·onehot_s, using #33 2.2 onehot | #40 `cat_index_mul_sum` (the select and cat in one kernel), then #41 `cat_slice_view_8` 18.9, a copy into the symmetric contraction's [5000, 16, 128] |
| F7 | 35 | fdamfw | same, l=1 | 15000×128×384 | 30.5 | #34 | #40 | #40, #41 |
| F8 | 37 | fdamfw | same, l=2 | 25000×128×384 | 48.9 | #36 | #40 | #40, #41 |
| F9 | 39 | fdamfw | same, l=3 | 35000×128×384 | 67.3 | #38 | #40 | #40, #41 |
| F10 | 51 | frht | int. 2 `linear_up` l=0 (product 1's `linear` folded in) | 5000×128×128 | 6.0 | – (A is the contraction output, ld 512, read in place) | – | #54 `cat_view_4` 5.2 (F10, F11 → [5000, 512], the fused forward's x) |
| F11 | 53 | frht | int. 2 `linear_up` l=1 | 15000×128×128 | 12.3 | #52 6.2: A copy [5000, 512] → [15000, 128] | – | #54 |
| F12 | 60 or 61 | frht | int. 2 `skip_tp` (`SpeciesSkip`), 0e only | 5000×128×384 | 12.3 | – (A in place, ld 512) | #62 2.8: own-species select, using #58 2.0 onehot | – |
| F13 | 61 or 60 | frht | int. 2 `linear` l=0 (2 paths) | 5000×256×128 | 9.1 | – (A is the fused forward's block message) | #67 31.0 for F13–F16: ÷(density+1) into [5000, 16, 128] for product 2 | #66 `cat_view_9` 23.3 (F13–F16 → [5000, 2048]); #67 `add_cat_div_slice_view_10` reads it |
| F14 | 63 | frht | int. 2 `linear` l=1 (3 paths) | 15000×384×128 | 31.2 | – | #67 | #66 |
| F15 | 64 | frht | int. 2 `linear` l=2 (3 paths) | 25000×384×128 | 50.2 | – | #67 | #66 |
| F16 | 65 | frht | int. 2 `linear` l=3 (2 paths) | 35000×256×128 | 47.0 | – | #67 | #66 |
| F17 | 73 | fhpk | product 2 `linear` (128x0e) | 5000×128×128 | 7.3 m | – | #74 2.8: `+ sc` (F12's skip); #75 3.8: aten `copy_`, destination unknown | – |

The trace cannot tell whether F12 or F13 runs first. Rows marked "m" are measured singletons.

Backward (16 launches, 437.7 µs):

| GEMM | #n | graph | model op | M×K×N (layout) | µs | glue before (µs, what) | glue after (µs, what) | cats / scatters consuming the output |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| B1 | 109 | f33db | ∂ product 2 `linear` (grad @ Wᵀ) | 5000×128×128 rr | 7.3 m | – (grad from readout 2's backward, cuBLAS #104) | – (into cuEq's backward #110) | – |
| B2 | 118 | fws43 | ∂ int. 2 `linear` l=3 | 35000×128×256 rr | 48.0 | #112 9.1: A = g ÷ (d+1) copied out of cuEq's [5000, 16, 128] gradient; #111 22.8 beside it: ∂density row reduction | – (to the w′ backward #125, layout as produced) | – |
| B3 | 119 | fws43 | same, l=2 | 25000×128×384 rr | 51.3 | #113 7.6 | – | – |
| B4 | 120 | fws43 | same, l=1 | 15000×128×384 rr | 31.9 | #114 5.2 | – | – |
| B5 | 121 | fws43 | same, l=0 | 5000×128×256 rr | 9.3 | #115 3.3 | – | – |
| B6 | 123 | fws43 | ∂ int. 2 `skip_tp` (grad_y @ Wᵀ, B a transposed view) | 5000×384×128 rc | 16.6 m | #116 4.4: A = g ⊗ onehot ([5000, 128] → [5000, 384]) | #130 (below) | – |
| B7 | 128 | fws43 | ∂ int. 2 `linear_up` l=0 | 5000×128×128 rr | 6.0 | – (grad slice of w′'s [5000, 512], in place) | #130 9.9: ∂node_feats₁ = (B7 + B6 + readout 1's grad, B8) assembled into [5000, 512]; #108 3.7: readout 1's grad zero-padded to 512 columns for #130; #131 7.1: aten `add_`, autograd's accumulation | #130 (cat via `slice_backward`) |
| B8 | 129 | fws43 | ∂ int. 2 `linear_up` l=1 | 15000×128×128 rr | 12.5 | #127 4.8: A copy [5000, 512] → [15000, 128] | #130 | #130 |
| B9 | 134 | feog | ∂ int. 1 `skip_tp` l=3 (B transposed view) | 35000×384×128 rc | 70.6 | #133 32.0: A = g ⊗ onehot | #141 29.7 for B9–B12: ∂density₁ row reduction and g ÷ (d+1) into [5000, 2048]; #143 8.9: copied back into [35000, 128], the A of B13 | – |
| B10 | 136 | feog | same, l=2 | 25000×384×128 rc | 51.2 | #135 24.6 | #141; #145 8.0 | – |
| B11 | 138 | feog | same, l=1 | 15000×384×128 rc | 31.8 | #137 13.0 | #141; #147 5.2 | – |
| B12 | 140 | feog | same, l=0 | 5000×384×128 rc | 12.5 | #139 4.3 | #141 (B16's A is read in place, ld 2048) | – |
| B13 | 144 | fkk52 | ∂ int. 1 reassociated message l=3 (grad @ (γxWᵀ)ᵀ) | 35000×128×192 rc | 36.7 | #143 | – | #150 29.7 for B13–B16: scatter into the outer sum's [15000, 1024] layout (transpose m↔slot) for `tabled_outer_sum_backward` #151 |
| B14 | 146 | fkk52 | same, l=2 | 25000×128×192 rc | 27.0 | #145 | – | #150 |
| B15 | 148 | fkk52 | same, l=1 | 15000×128×192 rc | 17.3 | #147 | – | #150 |
| B16 | 149 | fkk52 | same, l=0 | 5000×128×192 rc | 7.6 | – | – | #150 |

The readout GEMMs (`#77`, `#104` and the `gemv`/SIMT products, N or K below 32) go through `tf32x3_mm` but fall back to
cuBLAS IEEE. They are not 3xTF32 launches, so their glue (`#78` and `#103`, 2.4 µs) is left out.

## (b) Verdict per glue kernel and cat

Verdict classes:
- **E (epilogue):** the same float32 expression computed on the accumulator in the producing GEMM's epilogue. The
  accumulator is the float32 value `mm_into` stores, so every operand is bitwise the same.
- **S (output slice):** a pure copy, removed by storing the tile to its final address: pointer plus `ld`, with the row
  remap above where needed.
- **V (operand view):** a pure copy of an A operand, removed by reading A in place. It changes only the A iterator, not
  the epilogue.
- **P (prologue):** arithmetic on A before the product. It is bitwise only as an A-load transform before the TF32
  split, never as an epilogue, because the products would round differently.
- **N (not absorbable).**

| Kernel | µs | Verdict | Reason |
| --- | ---: | --- | --- |
| #28 `cat_view_7` | 19.2 | **S** | Pure cat of F2–F5 ([M_l, 128] → [5000, 2048] at off_l = 0, 128, 512, 1152). F2–F5 store there directly. |
| #31/34/36/38 `add_div_slice…tf32x3_mm` | 28.2 | **E** | `cat[a, off_l+j] / (density[a] + 1)`, one division per element on F2–F5's output. The epilogue makes a second store of v ÷ (d[row]+1) in the same [5000, 2048] layout, and F6–F9 read it per m with ld 2048. The density (#30) does not depend on F2–F5, so it runs first. |
| #40 `cat_index_mul_sum…` | 56.5 | **E**, with a new tile | Own-species select y₀·oh₀ + y₁·oh₁ + y₂·oh₂ plus the cat. A thread needs the same (row, channel) column of all 3 species, which sit 128 columns apart in different 64-wide N tiles today. Permute W's columns to [channel block of 64][species][64] and use a 64×192 (or 32×192) tile. The epilogue then evaluates the same unrolled sum in the same order and stores 128 columns into the [5000, 2048] slice. |
| #41 `cat_slice_view_8` | 18.9 | **S** | [5000, 2048] → [5000, 16, 128] is MACE's `reshape_irreps` on an `ir_mul` tensor. It has the same element order, so it is a copy Inductor could not alias. F6–F9's epilogue stores into the contraction's input buffer, plus a second store only if the [5000, 2048] tensor is also saved for the backward. |
| #54 `cat_view_4` | 5.2 | **S** | F10 and F11 into [5000, 512] (offsets 0 and 128). |
| #62 `index_mul_sum…` | 2.8 | **E**, with a new tile | Same select as #40 for F12 (l=0). |
| #66 `cat_view_9` | 23.3 | **S** | F13–F16 into [5000, 2048]. Kept as a store if the backward reads it (#141 and #111 take [5000, 2048] inputs). |
| #67 `add_cat_div_slice_view_10` | 31.0 | **E** | v ÷ (density[a] + 1) into [5000, 16, 128], the same element order as #66. A second store in F13–F16's epilogue. |
| #74 `add_3` | 2.8 | **E** | F17's v + sc: one float32 add, commutative, so the order cannot differ. |
| #130 `add_slice_backward_view_9` | 9.9 | **E** + **S** | ∂node_feats₁ assembled from B7 + B6 (+ readout 1's grad) and B8 (+ 0 from the `slice_backward` padding). It has a fixed add order. B6 and #107 run before B7, so B7's and B8's epilogues add them in Inductor's order and store into [5000, 512]. |
| #108 `slice_backward_0` | 3.7 | **E** (with #130) | A zero-padding of readout 1's [5000, 128] grad. It disappears when B7's epilogue reads the unpadded grad. B8's part adds +0.0, which the epilogue must reproduce (see the conditions). |
| #143/145/147 `clone_slice…tf32x3_mm` | 22.1 | **E** | #141 divides B9–B11's outputs by (d+1) into [5000, 2048], and these copy them back into [M_l, 128], B13–B15's A. B9–B11's epilogue computes g ÷ (d[row]+1) and stores [M_l, 128] directly. #141 keeps only its row reduction. |
| #150 `add_slice_backward_tabled_outer_sum_backward…` | 29.7 | **S** | B13–B16's rows (a, m) × (slot, k) go to `summed[a, slot, start_l+m, k]`, row stride 3072. A 64-column N tile is exactly one slot (offset slot·1024), so store per tile. The `add` is with zero padding: reproduce +0.0. |
| #20/22/24/26 `clone_div_mul_slice_t…tf32x3_mm…` | 39.8 | **V** | Inputs are only [15000, 1024], so it is a transpose copy of the outer sum. A for (l, m) has ld 3072, and K = 192 is 3 contiguous runs of 64 at stride 1024. A K-tile of 16 never straddles a run, so an A iterator with a two-level K stride, or `tabled_outer_sum_forward` writing this layout, removes it bitwise. The names also list `div`/`mul`. If the source shows a scalar scale, it becomes a **P**. |
| #52 `clone_slice_tf32x3_mm_3` | 6.2 | **V** | F11's A is the l=1 block of [5000, 512]: per-m batched with ld 512, offset 128 + m·128. The producer is cuEq's contraction, not our GEMM. |
| #127 `clone_slice_tf32x3_mm_8` | 4.8 | **V** | Same for B8, from w′'s [5000, 512]. |
| #112–#115 `add_clone_div_slice_slice_backward…` | 25.2 | **P** | A = g ÷ (d+1) from cuEq's backward output. The producer is not ours, and an epilogue on B2–B5 would divide after the product (not bitwise). |
| #116 `expand_mul_t…` | 4.4 | **P** | A = g[a, c]·onehot[a, s] for B6 (K = 3×128). |
| #133/135/137/139 `add_expand_mul…` | 73.9 | **P** | Same for B9–B12. An alternative outside this idea is a species-gathered product (K = 128, own species only). It has the same nonzero terms in the same 16-wide K tiles; the zero tiles add exactly 0, so it is bitwise except for signed zeros. |
| #111 `red_fused_add_div_mul_neg…sum` | 22.8 | **N** | ∂density₂ = −Σ over 2048 columns of g·msg/(d+1)². The row sum spans four GEMMs' tiles. |
| #141 `red_fused_add_div_mul_neg…sum` | 29.7 | **N** | ∂density₁, the same row reduction. Its elementwise half moves to B9–B11 (counted under #143–#147); the reduction stays. |
| #19 `div_mul_slice_t_unsqueeze_2` | 1.6 | **N** | Builds F2–F5's B operand γ·x·Wᵀ. It is not an epilogue; it is constant for a structure, so it could be built once outside the graph. |
| #33, #58 onehot gathers | 4.2 | **N** | Inputs of the selects (tiny). |
| #75 aten `copy_` after F17 | 3.8 | **N** (unknown) | An eager copy whose destination the trace does not show. |
| #131 aten `add_` [5000, 512] | 7.1 | **N** (conditional) | Autograd's gradient accumulation, outside the compiled graph. It is absorbable only if the graph returns the accumulated gradient. |

**Conditions for "bitwise" (to check on the GPU):**
1. **Division instruction.** The epilogue must use the same float32 division instruction as Inductor's Triton kernel:
   `div.rn.f32` or `div.full.f32`, read from the PTX. CUDA C++ `/` gives `div.rn` without fast math; `div.full` needs
   inline PTX. d + 1 must be the same float32 add.
2. **Signed zeros.** Zero-padding adds (`slice_backward` + `add`) turn −0.0 into +0.0, so the epilogue must add +0.0f
   explicitly.
3. **Species-select sum.** The select must keep Inductor's term order and FMA contraction. With one-hot values of
   exactly 0 and 1 and finite y, any order gives y_own except for the sign of a zero, but the epilogue should copy the
   generated expression anyway.
4. **Tiling.** Re-tiling M or N does not change an element's K sum (the module docstring of `tf32x3.py`: each element's
   sum is the same in every tile), so row remaps and per-m or per-slot batching are bitwise.

## (c) Absorbable time

| Class | Kernels | µs per force call |
| --- | --- | ---: |
| E: epilogue, bitwise | #31/34/36/38, #40, #62, #67, #74, #108, #130, #143/145/147 | **157.0** |
| S: output slice | #28, #41, #54, #66, #150 | **96.3** |
| **E + S (what idea 31 asks for)** | | **253.3** |
| V: operand view (A in place, also bitwise; a mainloop iterator, not an epilogue) | #20–#26, #52, #127 | 50.8 |
| P: prologue arithmetic (bitwise only as an A-load transform) | #112–#115, #116, #133–#139 | 103.5 |
| N: not absorbable | #111, #141, #19, #33, #58, #75, #131 | 69.2 |

253.3 µs is 4.4 % of the 5.75 ms force call's kernel time; with V it is 304.1 µs. These are the glue kernels' own times,
an upper bound on the gain. Each absorbed kernel saves its launch and one read-and-write pass. The epilogue still pays
its stores, and dual stores (#31–#38, #66/#67, #41 if [5000, 2048] stays saved) add a write to the GEMM.

## (d) The family to build first

**Recommendation: the interaction `linear` products (F2–F5 and F13–F16) with one epilogue functor.**

**The functor:** store the accumulator into the [5000, W] slice with the (a, m) row remap and pointer plus `ld`.
Optionally, also store acc ÷ (density[row]+1) as a second output, either in the same layout or as [M_l, 128].

**What it absorbs:**
- **Forward:** #28 + #31–#38 + #66 + #67 = **101.7 µs**.
- **Backward:** the same functor on B9–B11 (divide, store [M_l, 128]) absorbs #143–#147, another **22.1 µs**.
- **Total:** **123.8 µs**, half of the E + S total.

**Why first:**
- It keeps today's 64×64 tile and mainloop and only adds a row-broadcast division and a remapped store.
- The bitwise check reduces to condition 1 (the division instruction).
- It covers 12 of the 33 launches (F2–F5, F13–F16, B9–B12).

**Second:** the `SpeciesSkip` products (F6–F9, F12) with permuted columns, a 64×192 tile and the select epilogue: #40
+ #41 + #62 = **78.2 µs**. It needs a new tile shape and the term-order check.

**Third:** the output-slice-only stores of `linear_up` and the backward (#54, #150, and #130 with #108): 48.5 µs.

## Step 2 (6 November 2026): Inductor's code, the first family built, bitwise check and MD A/B

**Commit** `7264f56f` (branch `fastmace/idea31-glue`). Switch `fastmace.kernels.cutlass.linear_div.LINEAR_DIV`
(`None` with `LINEAR_DIV_GPUS = GPUModels(())`: off everywhere; `True` for an A/B side). Run files: `idea31/`.

### Inductor's generated code (local RTX 5060 Ti under the H200 plan)

Dump: `TORCH_LOGS=output_code`, fresh Inductor and Triton caches, one OMAT-0 medium force call on SrTiO3 10×10×10
(`tools/bench/idea31_epilogue.py --dump-only`, `FASTMACE_GPU_ALIAS="NVIDIA GeForce RTX 5060 Ti=NVIDIA H200"`; local only,
not kept: `~/fm-runs/idea31-dump`). The glue is the same custom-op boundary on every backend, so the bodies hold for the
H200 too; only the GEMM inside `tf32x3_mm` was Triton here.
- **`cat_view_9`** (#66): a pure select-and-copy of the four [M_l, 128] products into [5000, 2048] (offsets 0, 128, 512,
  1152). No arithmetic.
- **`add_cat_div_slice_view_10`** (#67): reads #66's output and `density[a]`, computes `tmp = d + 1.0` then `x / tmp`,
  writes [5000, 16, 128] (`reshape_irreps`' cat, the same element order). The same `x / (d + 1)` per element as the
  model, in fp32.
- **Division instruction:** every division in these kernels and in `add_div_slice…tf32x3_mm` (#31–#38), the backward
  `add_clone_div_slice…` (#112–#115) and the ∂density reductions is **`div.full.f32`** (approximate full-range, not
  `div.rn`); `d + 1` is `add.f32` (rn; `add.f32x2` on sm_120). The epilogue therefore uses inline PTX `div.full.f32`
  after a plain fp32 add.
- **Signed zeros:** no `+0.0` padding is involved in #66/#67 (`tl.where` selects, no add with zero).
- **∂density (int. 1, #141 `red_fused_add_div_mul_neg_slice_backward_sum_view_4`):** `-g * ((cat / (d+1)) / (d+1))`,
  summed over the 2048 columns in a looped reduction (`fma.rn` contraction of the accumulate). This is aten's
  `div_tensor_other_backward`, which the new op's backward reproduces, with `cat / (d+1)` taken from the op's output
  (the same bits).
- **#20–#26 (`clone_div_mul_slice_t…`):** no floating-point instructions in the PTX at all (integer index arithmetic
  only), so they are pure transpose copies despite the names: **V**, not **P**.
- **F1:** confirmed unread: `FirstInteractionReassociation.forward` uses `self.species_feats`, never `node_feats`, when
  `summed_fn` is set (the fused-first path), so interaction 1's `linear_up` product is dead work (7.5 µs). Not changed
  here.

### What was built (F13–F16 only)

- **Kernel:** `fm_tf32x3_remap_div_{rr,rc}_64x64` in `tf32x3_sm80.cu`. It is the 64x64 `mm_into` mainloop (the same
  `Mma` type and K iterations; alpha 1, beta 0) with CUTLASS's default epilogue around a new output iterator. Row r is
  stored at `c + (r / rows_per_atom) * atom_stride + (r % rows_per_atom) * row_stride`, each element as
  `div.full(acc, density[r / rows_per_atom] + 1)`. Both cubins are rebuilt; the SASS of every existing entry is
  unchanged (cuobjdump hash per function, sm_80 and sm_90).
- **Op:** `fastmace::tf32x3_linear_div` (`kernels/cutlass/linear_div.py`). It is called from `BlockLinear.divided` in
  `generic_interaction_forward` for `self.linear(message) / (density + 1)`. `reshape_irreps` becomes a view of its
  output, so #66 and #67 are gone.
- **Backward:** aten's formulas (`grad / (d+1)`; `-grad * (out / (d+1))` summed), and `tf32x3_mm(grad_j, weight_t_j)`
  per block, as `Tf32x3Matmul`.
- **Where it applies:** only where `precision.route` finds the CUTLASS backend and the switch on (decided at build, so
  the trace has no new graph break). It covers the tuple input of `BLOCK_MESSAGE` (the H200 and A100 default) or a
  tensor input without `SEGMENT_ROWS`. On OMAT-0 medium it fires in interaction 2 only (28 launches counted while
  warming up and capturing).
- **Not built yet:** F2–F5 (interaction 1, inside the reassociation, whose ÷(d+1) copies would stay as the F6–F9 A
  copies unless the epilogue stores [M_l, 128]) and B9–B11. The `rc` entry exists for B9–B11, but nothing calls it.

### Bitwise check

- **Op level (`tests/gpu/test_linear_div_cuda.py`):** OMAT-0's interaction-2 shapes (l = 0..3, K = 256/384/384/256 →
  128, n = 5000 and 997, zero rows). The test compiles `linear(x) / (d + 1)` (today: `mm_into` per block plus
  Inductor's cat and division) and `linear.divided(x, d)` with `torch.compile`, then compares the output, every left
  factor's gradient and the density gradient as int32 bit patterns. **Passed on the H200 and the A100-SXM4-80GB** (with
  the tile tests: 7 passed on each). It also passed locally on the RTX 5060 Ti with a local sm_120 cubin of the same
  source (not shipped).
- **Force calls and MD (`tools/bench/idea31_epilogue.py`, SrTiO3 10×10×10, 2 rattled force calls and 20 Langevin
  steps):** the run-to-run floor is not zero. A second build of the *default* (a2) already differs from the first (a):

| GPU | a2 vs a: forces (call) | md.forces after 20 steps | b vs a: forces (call) | md.forces after 20 steps |
| --- | ---: | ---: | ---: | ---: |
| H200 | 3.8e-6 | 8.1e-5 | 3.6e-6 | 8.1e-5 |
| A100-SXM4-80GB | 3.2e-6 | 8.5e-5 | 3.6e-6 | 8.1e-5 |
| RTX 5060 Ti (local, sm_120 cubin) | 3.1e-6 | 4.3e-5 | 3.7e-6 | 4.5e-5 |

  End to end, LINEAR_DIV is indistinguishable from a rebuild of today's path. The nondeterminism is upstream (the energy
  of the first force call already differs between a and a2). The bitwise statement rests on the op-level test.

### MD-step A/B (OMAT-0 medium, SrTiO3 10×10×10, 1 fs, b = LINEAR_DIV=True, `hopper_sweep.sh`)

| GPU (Modal) | b/a, build order ab (abab) | b/a, build order ba (abba) | a (ms) | b (ms) |
| --- | ---: | ---: | ---: | ---: |
| NVIDIA H200 (profile fourth) | **0.9912** | **0.9934** | 5.583 / 5.581 | 5.534 / 5.544 |
| NVIDIA A100-SXM4-80GB (profile third) | **0.9896** | **0.9891** | 10.552 / 10.555 | 10.442 / 10.440 |

All four runs are valid (no compile while timed, no other GPU process). The H200 gains 37–49 µs per step, close to
#66 + #67's 54 µs. The A100 gains about 110 µs. The first A100 attempt got an A100 80GB PCIe and stopped with WRONG-GPU
(`idea31/a100-pcie-wrong-gpu/`). Run directories: `idea31/h200/`, `idea31/a100/`, `idea31/local-rtx5060ti/`. Job:
`tools/modal/idea31_check.sh`.

## Step 3 (6 November 2026): interaction 1 (FIRST_DIV) and the species-skip products (SKIP_SELECT)

Branch `fastmace/idea31-more`, from main `5dc5d799` (LINEAR_DIV default on the H200 and A100-SXM4). Both new switches
are `None` with empty GPU lists (off everywhere). The A/B side b sets the new switch on top of LINEAR_DIV's default, so
a = today's default. Job: `tools/modal/idea31_check.sh` (`CONST`, `NAME`, `GPU_TESTS`). Model, cell and protocol are
as in step 2.

### FIRST_DIV (`a8f5f6e8`): F2–F5, #28's cat slice, #31–38's divisions, and B9–B11

- **What:** the op `fastmace::tf32x3_blocks_div(lefts, rights, density)` (`kernels/cutlass/first_div.py`). It computes
  `FirstInteractionReassociation`'s per-degree products with the remap-div epilogue straight into one flat buffer
  (blocks one after another, [M_l, 128] each). `forward` returns views, so the copies feeding F6–F9, #28's cat slice
  and the #31–38 divisions are gone.
- **B9–B11:** covered by the op's own backward, not a separate switch. The backward does the following:
  - computes the quotients per block, tied to the gradient (`other + grad[:1]*0.0`) so the partitioner keeps them in
    the backward;
  - forms ∂density as one reduction over the joined grads times the joined quotients, with the product directly
    before the sum, as in today's #141;
  - computes `tf32x3_mm(grad / (d+1), right.t())` per block.
- **Bitwise (`tests/gpu/test_first_div_cuda.py`):**
  - **Passed:** the output and every left-factor gradient on the H200 and A100-SXM4.
  - **Failed:** the **density gradient differs** in the last bits on both GPUs (n = 5000: 3684 of 5000 rows, n = 997:
    765 rows, |Δ| ≤ 4e-6). The cause is likely a different reduction split or contraction in the joined ∂density
    reduction on sm_80/sm_90.
  - Locally (sm_120, with `torch.use_deterministic_algorithms`), the density gradient is bitwise equal too.
- **MD A/B (side b = FIRST_DIV=True):**

| GPU (Modal) | b/a, order ab | b/a, order ba | a (ms) | b (ms) |
| --- | ---: | ---: | ---: | ---: |
| NVIDIA H200 (profile fourth) | **1.0305** | **1.0330** | 5.478 / 5.466 | 5.645 / 5.646 |
| NVIDIA A100-SXM4-80GB (profile third) | **1.0349** | **1.0337** | 10.284 / 10.307 | 10.643 / 10.655 |

  b vs a end to end is at the run-to-run floor (forces 3.3–3.4e-6, md.forces 8.7e-5 on the H200, as a2 vs a).
- **Verdict: rejected** (3–3.5% slower on both GPUs, and the density gradient is not bitwise). Not investigated
  further. Likely causes:
  - the density must exist before the products;
  - extra backward kernels (per-block quotients and grad/(d+1));
  - a changed split of the interaction-1 graph.
- **Pre-existing graph break (not from this branch):** `fused_first_backward` calls `torch.cuda.get_device_name` inside
  the trace (`rewrites/generic/modules.py:93`). This splits the interaction-1 graph on every backend and is worth
  hoisting to build time.
- Run files: `idea31/firstdiv-h200/`, `idea31/firstdiv-a100/`.

### SKIP_SELECT (`6718aa2a`): the species-skip products (select epilogue, 64×192 tile)

- **Kernel:** `fm_tf32x3_remap_select_{rr,rc}_64x64` in `tf32x3_sm80.cu`. It is the remap mainloop with a `Select`
  template parameter: element (r, j) is stored only if `kind[atom] == j / select_columns`, at column
  `j % select_columns`. The product with all species' weights runs as one GEMM. The epilogue keeps each atom's own
  species and writes straight into the reshaped [n, width] layout.
  - Both cubins were rebuilt (sm_80 `13e99898…`, sm_90 `91ca0924…`).
  - The SASS of every existing entry is unchanged.
- **Op:** `fastmace::tf32x3_skip_select` (`kernels/cutlass/skip_select.py`), called from `SpeciesSkip` where the layout
  allows it. In interaction 1 the message is returned as a `reshape_view`. This removes #40/#62's
  `cat_index_mul_sum` and the `cat_slice_view` copy (#41). The backward is
  `tf32x3_mm((g ⊗ onehot).reshape(m, S·cols), weight.t())`.
- **Bitwise (`tests/gpu/test_skip_select_cuda.py`):** (5000, l 0..3, 2048), (997, l 0..3, 2048) and (5000, l 0, 512),
  compiled `SpeciesSkip` off vs on. The output and gradients are compared as int32 bits. **Passed on the H200 and the
  A100-SXM4-80GB** (with LINEAR_DIV's and the tile tests; the only failures in those jobs are FIRST_DIV's two
  density-grad cases above). It also passed locally.
- **End to end:** b vs a is at the floor (H200: forces 3.8e-6, md.forces 8.8e-5; A100: 3.6e-6, 8.4e-5).
- **MD A/B (side b = SKIP_SELECT=True on top of LINEAR_DIV):**

| GPU (Modal) | b/a, order ab | b/a, order ba | a (ms) | b (ms) |
| --- | ---: | ---: | ---: | ---: |
| NVIDIA H200 (profile fourth) | **0.9876** | **0.9919** | 5.551 / 5.540 | 5.483 / 5.496 |
| NVIDIA A100-SXM4-80GB (profile third) | **0.9843** | **0.9858** | 10.437 / 10.377 | 10.273 / 10.229 |

  All four runs are valid. The H200 gains 45–69 µs per step and the A100 148–164 µs.
  - The first A100 attempt got an A100 80GB PCIe and stopped with WRONG-GPU
    (`idea31/skipselect-a100-pcie-wrong-gpu/`); the retry ran on the SXM4.
  - Run files: `idea31/skipselect-h200/`, `idea31/skipselect-a100/`.
- **Status:** a candidate default for the H200 and A100-SXM4 (decision 66). `SKIP_SELECT_GPUS` stays empty on this
  branch, for the coordinator to set.
- **Modal cost of step 3:** about $2.1 (profile third at $20.43, fourth at $14.45 metered this month).

## Next step (one GPU)

1. **Read the generated code.** Dump Inductor's output code (`TORCH_LOGS=output_code`) for Benchmark_loup on the H200.
   For each kernel above, confirm:
   - the body (whether #20–#26 are pure copies);
   - the division instruction in the PTX of #31–#38, #67 and #141;
   - the select's expression in #40 and #62.
2. **Measure per launch.** Take a per-instance trace of the 33 launches to replace the estimated GEMM µs.
3. **Check F1.** Confirm whether F1's output is read.
4. **Build and gate the first family.** Build F13–F16 with the slice-plus-division epilogue, then check:
   - **Bitwise:** compare against the current path on the force call, values and forces.
   - **Speed:** run the MD-step A/B in both build orders (decision 66).
