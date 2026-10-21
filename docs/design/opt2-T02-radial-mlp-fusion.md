# OPT2-T02: radial MLP hidden layers in one kernel per direction (design note)

**Status: design, 21 October 2026.** Plan `plans/opt-2.md`, OPT2-T02 (OPT-T09 rank 2, ceiling 4–8 %, rounding class).

## Why the 15 October fusion gave nothing

ResearchModule's `--fused-radial-hidden` (ported here as the opt-in record `fused_radial_hidden`,
`fastmace/kernels/triton/fused_radial.py`) was measured with no gain: −0.06 ms on NaCl, 0 on NaNbO3, +0.15 ms on
NbCl5 (`studies/further_optimizations/2026-10-15-step-breakdown.md`). Its scope explains that:

- **Too little of the network.** It fuses one LayerNorm, SiLU, Linear(64, 64), LayerNorm, SiLU block. MH-1's
  hidden stack is LN, SiLU, Linear, LN, SiLU, Linear, LN, SiLU, so the second Linear and the third LN/SiLU stay
  outside, as separate kernels.
- **Forward-shaped.** Its backward recomputes the block and returns the input gradient, but the gradients
  through the remaining Linear, LN and SiLU are still separate GEMM and elementwise kernels.
- **It breaks Inductor's own fusion.** The op is opaque to torch.compile. Inductor had already fused the
  species gathers, the add and the first LN/SiLU into one kernel (`triton_per_fused_add_div_gather_mm_mul_
  native_layer_norm…`), and the LN/SiLU backward into another. Around an opaque block, those inputs and outputs
  must be written to memory again, which ate the block's saving.
- **Small tiles.** 16 rows per program, with an IEEE FP32 `tl.dot` of `[16, 64] @ [64, 64]`.

## What is left to win (OPT-T09, local RTX 5060 Ti, current defaults)

Radial "other GEMMs" plus "layer norms and activations", per force call:

| | NaCl | NbCl5 | NaNbO3 |
| --- | ---: | ---: | ---: |
| ms | 0.31 | 1.83 | 3.17 (OPT-T09 radial elementwise and hidden GEMMs) |
| share of the step | ~10 % | ~8.5 % | ~15 % |

The kernels involved, NbCl5:
- `addmm [E,64]@[64,64]` forward, 4 launches (2 layers × 2 networks);
- `mm` input gradients, 4 launches;
- the fused gather + add + LN + SiLU forward;
- LN/SiLU backward kernels, 2 launches;
- the joint `[E,128]@[128,10]` gradient.

All of them are memory- or occupancy-bound (48–82 % of DRAM, 3–28 % of FP32).

**Minimum traffic of a full fusion** (one network, D = 64, float32):
- forward: read x, write h, 512 B per edge;
- backward: read x and the gradient of h, write the gradient of x, 768 B per edge.

Compute is 5 `[rows, 64] @ [64, 64]` products per edge in the backward (3 to recompute the forward, 2 for the
gradients) and 2 in the forward: about 57 kflop per edge.

For NbCl5 (56,320 edges, two networks), that is 144 MB, about 0.32 ms at 448 GB/s, and 6.4 GFLOP, about 0.6 ms
at ~10 TFLOPS FP32 FMA. So the fused version sits around 0.6–0.8 ms against 1.83 ms today, about 5 % of the
step. Compute, not memory, becomes the bound: IEEE FP32 with TF32 off is the production precision.

## Design

**Record `fused_radial_mlp`**, MH-1 family (`fastmace/rewrites/mh1`).
- Opt-in (`never_automatic`), rounding class.
- Requires `fold_cutoff`, because it acts on `FactoredRadialMLP`, the module `factor_species` makes.
- Conflicts with `fused_radial_hidden`, which covers a subset.

**Applies when** a network's hidden run `rest[:-1]` is exactly LN, SiLU, then k ≥ 1 times (Linear(D, D) with
bias, LN, SiLU):
- every LN is affine over (D,);
- D is a power of two ≤ 128;
- k ≤ 4.

MH-1: both the weight network and the density network, D = 64, k = 2. Anything else is refused with the
reason.

**Kernels** (new file `fastmace/kernels/triton/radial_mlp.py`, `torch.library` custom ops with fake
implementations, like `fused_radial.py`):
- **forward** `(x, gamma[k+1, D], beta[k+1, D], weight_t[k, D, D], bias[k, D]) → h`. One program per tile of
  rows; the layers loop in registers. Only x is read and only h written.
- **backward** `(x, grad_h, same parameters) → grad_x`. Per tile it recomputes the forward and keeps each layer's
  normalised input, rstd and SiLU sigmoid in registers (k + 1 = 3 layers at D = 64). Then it runs the chain
  backwards: SiLU' → LN' → `@ W` → … Only grad_x is written. The parameters are constants in MD, so there are
  no weight gradients, as for `fused_radial_hidden`.
- **Dots:** `tl.dot(..., input_precision="ieee")`, FP32, never TF32.
- **Rows per program** by a rule from the shapes, no table. Start at 32 and check register spills on both GPUs.
  A larger tile only if it does not spill.
- **CPU:** the PyTorch formulas, the same modules in the same order, so the CPU result is bitwise the
  original's.

**Autograd:** a `torch.autograd.Function` around the two ops that saves only x. torch.compile traces it, as it
does `_FusedRadialHidden`.

**What stays with Inductor:**
- the species gathers and the add that form x;
- the `[h·c, c]` concatenation;
- the last layer, with its folded GEMM and Triton input gradient.

The last two belong to OPT2-T01 and OPT2-T03. The contract between us is a plain contiguous `[E, 64]` h and
its gradient (both workers told, 21 October).

## Checks

1. **Applicability and refusal:**
   - MH-1's two networks apply;
   - any other layout is refused with its reason: a missing affine LN, a Linear without bias, D not a power
     of two or above 128, k = 0 or k > 4, or another activation;
   - the record conflicts with `fused_radial_hidden` and requires `fold_cutoff`.
2. **CPU float64:** the record's model against the original on random graphs (the family self-check) and on
   the MH-1 tiny model: energy, forces and stress agree to rounding.
3. **GPU kernel tests:** forward and input gradient against PyTorch autograd in float64 (≤ 1e-12 relative) and
   float32 (≤ 1e-5), for D = 64 and 32, k = 1 to 4, row counts not a multiple of the tile.
4. **Force gate:** NaCl and NbCl5, float32 (the round's rule).
5. **Timing:** a local same-process A/B, both build orders, 12 blocks; murphy when free before Thursday 16:00
   UTC. Plus a kernel-level microbenchmark of the replaced kernels against the new pair, so that a step result
   can be traced to its kernels.

**Stop rule:** if the microbenchmark saving is below 0.15 ms on NbCl5 (under 1 % of the step), stop after
the microbenchmark and report, as 15 October should have.
