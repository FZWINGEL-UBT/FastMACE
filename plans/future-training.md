# Future work: faster training and fine-tuning

Status: **future work, not scheduled** (owner, 28 October 2026: "Currently we dont have much resources to run
that so for now just document this as future work"). Nothing here is built or measured. Start with the
profiling in step 1 when compute is available.

## Goal

Make training and fine-tuning of MACE models (MH-1 above all) faster than MACE's own trainer
(`mace_run_train`, multihead fine-tuning with replay). The owner expects a lot of headroom over current
methods.

## Package boundary

- The training workflow goes in a separate package (working name `fastmace-train`): datasets, the trainer
  loop, loss weights, learning-rate schedules, EMA, checkpoints, multihead replay and an active-learning loop.
  It wraps or follows MACE's trainer instead of copying it, and depends on FastMACE one way only.
- The reusable speed-ups stay in FastMACE, in its rewrite layer. FastMACE would gain:
  - a record of which rewrites stay valid under training;
  - second derivatives for the custom kernels that need them;
  - small inference-side helpers: fast batch evaluation of a dataset to validate a fine-tuned model, and MD
    or MC frames written as labelled candidates for training data.

## Why the inference speed-ups do not all transfer as they are

Training on forces differentiates twice. The forces are the gradient with respect to positions, and the loss
on them is then differentiated with respect to the weights. The weights also change every optimiser step.
FastMACE's speed-ups fall into four groups by how they behave under that.

### 1. Exact rewrites: valid for any weights

These are algebraic reformulations, independent of the weights' values:

- the reassociated first interaction;
- density division before `linear_1`;
- `species_skip`;
- the joint radial input;
- grouped linear layers and the merged residual;
- receiver sorting;
- block linear layers.

They need a second derivative:

- In plain PyTorch they have it already.
- The Triton fused backward kernels (folded radial input gradient, `edge_outer_sum`, chunked convolution
  backward) have none.
- The OEq streaming convolution needs one. Check what upstream OpenEquivariance offers against our vendored
  copy.
- cuEquivariance's kernels support training already.

### 2. Weight-derived tables: rebuild once per optimiser step

These are computed from the weights, which are constant within one optimiser step:

- `species_tables`;
- the folded radial layer;
- `fold_readout_tail` and `fold_product_linear`.

Rebuilding them at the start of each step is cheap: it costs about one evaluation per species, while the
saving applies to every atom in the batch. Gradients flow back through the rebuild to the weights.

The radial Chebyshev tables (`radial_tables`, `w_tables`) are a special case. Their coefficients are a fixed
linear map of the radial network evaluated at a few hundred nodes, so they are differentiable with respect to
the weights. They are also far cheaper than running the network on every edge. Before adopting them, check
that the interpolation error does not change the trained model: compare its validation errors with an
exact-path training.

### 3. Frozen parts in fine-tuning: transfer completely

A fine-tune that freezes the trunk (embeddings, radial networks, early interactions) can run every inference
optimisation on the frozen part, including the certified tables and the precomputed folds. A readout-only
fine-tune still needs gradients with respect to positions through the trunk, but none with respect to its
weights there. That is close to the MD workload FastMACE already optimises.

### 4. Systems work: probably the largest headroom

- **CUDA graphs over the whole training step:** forward, force backward, loss backward, optimiser and EMA,
  with batches padded into fixed-shape buckets. FastMACE's bucket and capture machinery is built for this.
- **Data and batching:** MACE's trainer is often limited by Python data loading and batch assembly. The
  datasets are static, so neighbour graphs can be built once and kept on the GPU.
- **Precision:** MACE often trains in float64. Two options:
  - float32 with float64 energy sums, as qualified for MD;
  - FastMACE's float64 emulation on consumer GPUs (decision 65), but only after the fp64-D1 relaxation fix:
    small products outside the model must not use cuBLAS GEMMs.
- **Fused optimiser and EMA updates.**

## Steps when resumed

1. **Profile MACE's own trainer** on a realistic fine-tune, for example MH-1 on the NaMOCl4 frames from
   ResearchModule. Split the step time into data loading, forward, force backward, loss backward and
   optimiser, per GPU model. This sets the real headroom; until then the estimates above are guesses.
2. **Systems prototype:** precomputed GPU graphs, padded buckets, one captured training step. Check that the
   losses and gradients match MACE's trainer, then time it.
3. **Training safety of the rewrites:**
   - mark each rewrite as safe or unsafe to train through;
   - compare gradients with respect to the weights against the unmodified model (second derivatives
     included);
   - add second derivatives where a kernel lacks them.
4. **Per-step table rebuilds** (group 2), then the differentiable radial tables, with a convergence check
   against exact training.
5. **Frozen-trunk fine-tuning** (group 3).
6. Decide the package layout and the API with the owner (a SPEC §18 decision) before anything is released.
