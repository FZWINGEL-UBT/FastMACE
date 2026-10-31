# Rewrites and precision options

This page lists every field of `RewriteConfig` and `PrecisionConfig`, what
`"auto"` resolves to, and the limits of each option. For the history of the
screens behind these defaults (measurements, rejected ideas) see
[Rewrite screens](../rewrites.md). For how the planner works in general see
[The planner](../guide/planner.md).

## What a rewrite is

A **rewrite** replaces part of the MACE model with code that computes the same
function faster: weights folded together, sums reordered, several kernels
fused into one, or a stock module replaced by a custom kernel. The model's
weights and its physics do not change. Each rewrite is a record in the rewrite
registry (`fastmace/rewrites/`). The record names the models it fits, the
MACE source it was written against (checked by hash), the records it needs
and its exactness class.

Every rewrite runs a **self-check** when it is applied: the rewritten module
is compared with the original module on random inputs. The relative error must
be at most 1e-4 in float32 and 1e-10 in float64. A failed self-check turns the
rewrite off, together with every rewrite that needs it, and the plan is built
again from the unrewritten model.

### Toggle values

Most fields take one of three values:

- `"auto"` (the default): the planner decides, by a fixed rule per field,
  from the backend, the force precision, the model and whether the plan is
  reproducible. The rule and the outcome are recorded in the plan.
- `"on"`: you ask for the rewrite. If the model, the backend or the plan
  cannot take it, planning stops with `PlanRefused`, which lists every reason.
  An `"on"` is never silently turned off.
- `"off"`: the rewrite is not applied. Rewrites that need it are off too.

A few fields are plain booleans or have their own values; their rows say so.

### Exactness classes

Each registry record has one of these classes. The tables below give the
class of each option.

| Class | Meaning |
| --- | --- |
| exact | The same arithmetic in the same order as the original. Results are bit-for-bit identical, except where a kernel sums with atomic additions, whose order can change from run to run. |
| rounding | The same mathematical function, computed in a different order or grouping (reassociated sums, folded or factored weights, different matrix-product grouping). Results differ from the original only by floating-point rounding. Most rewrites are of this class. |
| certified | An approximation (for example a fitted table) whose error is measured when the plan is built, on the plan's own model, species and range, against a float64 evaluation of the same weights. The error must be at most the rounding error of the computation it replaces, for the values and their first derivatives. The self-check holds this certificate; a ratio above 1 fails it and turns the rewrite off. |
| approximate | Not exact to rounding. Never on under `"auto"`. No current option is of this class. |

"Deterministic" in the tables means bitwise repeatable from run to run on the
same GPU. A reproducible plan (`fm.Workload(reproducible=True)`) refuses any
rewrite that is not deterministic.

### Short names

- **cuEq**: NVIDIA cuEquivariance, the tensor-product library MACE uses.
- **OEq**: OpenEquivariance, the second tensor-product library of the hybrid
  backend. "Hybrid" means cuEq for most of the model and OEq for the
  convolution.
- **Plain cuEq**: the cuEq backend without OEq.
- **Inductor**: PyTorch's compiler (`torch.compile`).
- **MH-1 class**: models whose interaction blocks are MACE's non-linear
  residual blocks and that apply the radial cutoff inside the interaction.
  MACE-MH-1 is the main one; OMOL-0 and POLAR-1 also have these blocks.
- **Generic MACE family**: MACE-MP-0, MPA-0, OMAT-0 and MatPES, whose
  interactions are MP-0's residual block or the density-normalised block.
  MACE-OFF has a plain first interaction and is not in this family, but takes
  the generic variants of some rewrites.

### See which rewrites are on

```python
print(sim.plan)                        # rewrites on, and the ones off with a short reason
print(sim.plan.describe(full=True))    # every decision with its rule and reason
sim.plan.decision("RewriteConfig.fold_cutoff")   # (item, choice, reason)
sim.plan.overrides()                   # the options you set explicitly
```

The resolved values are also in `sim.plan.rewrites` and `sim.plan.precision`,
in the run report and in `metadata.json`.

### Set an option

`RewriteConfig` is not in the top-level `fastmace` namespace. Import it from
`fastmace.config` and pass it to the force field:

```python
import fastmace as fm
from fastmace.config import RewriteConfig

model = fm.load("mh1", head="matpes_r2scan")
ff = fm.ForceField(model, atoms, precision="float32",
                   rewrites=RewriteConfig(radial_tables="off", direct_node_feats="on"))
sim = fm.Simulation(atoms, ff, seed=1,
                    integrator=fm.integrators.Langevin(timestep_fs=1.0, temperature_K=600,
                                                       friction_per_fs=0.002))
print(sim.plan.decision("RewriteConfig.radial_tables"))
```

Fields you leave out stay at `"auto"`. `rewrites="auto"` is the same as
`RewriteConfig()`. `fm.md` always uses `rewrites="auto"`.

### How "auto" chains

Many rewrites build on another one. Under `"auto"` a rewrite is on only when
the rewrites it needs are on, so turning one off also turns off what is built
on it. The main chains are:

- `factor_species` → `fold_cutoff` → `density_first`, `joint_radial_input`,
  `reassociate_first`, `fused_folded_backward`, `streaming_conv`,
  `radial_tables`;
- `streaming_conv` + `fused_folded_backward` → `chunked_conv_backward` →
  `fused_folded_forward`; `radial_tables` + `chunked_conv_backward` →
  `w_tables`, `wprime_backward` → `recompute_conv_weights`;
- `block_linear` → `grouped_linear`, `merged_residual` → `gated_rows` →
  `segment_gemm`, `recompute_node_segments`; `block_linear` → `grouped_launch`, `fold_readout_tail`;
  `block_linear` + `species_tables` → `fold_product_linear`.

`factor_species` and `block_linear` are on under `"auto"` only with the hybrid
backend, so on plain cuEq most rewrites are off. Setting `fold_cutoff="off"`
also turns off the generic variants of `reassociate_first`, `streaming_conv`
and `fused_folded_backward`, because the planner checks `fold_cutoff` before
it checks which variant fits the model.

The reference tier (`Simulation(mode="reference")`) runs MACE's own
calculator with every rewrite off. The eager and compiled tiers keep the
graph plan's rewrites.

## Radial network

The radial network turns each edge length into the weights of that edge's
tensor product. In the MH-1 class it also reads the species of both atoms.

| Option | Values | Default (`auto` means) | What it does | Exactness / limits |
| --- | --- | --- | --- | --- |
| `factor_species` | `auto`, `on`, `off` | On with the hybrid backend where the model takes it (MH-1 class). | Splits the radial network's first layer into a species part and a distance part, so the species part is computed once per species instead of once per edge. | rounding, deterministic. Needs a factored (MH-1-class) interaction. |
| `fold_cutoff` | `auto`, `on`, `off` | On with `factor_species` where the model takes it. | Folds the cutoff function into the last radial layer instead of multiplying the edge weights by it afterwards. | rounding, deterministic. Needs `factor_species` and a model that applies its cutoff in the interaction (`apply_cutoff=False`). |
| `joint_radial_input` | `auto`, `on`, `off` | On with `fold_cutoff`. | Runs one first layer for both radial networks of an interaction (the tensor-product weights and the edge density) instead of two. | rounding, deterministic. Needs `fold_cutoff`. |
| `fused_folded_backward` | `auto`, `on`, `off` | On with `fold_cutoff` (MH-1 class); for the generic MACE family its generic variant is on with the hybrid backend. | Computes the input gradient of the folded last radial layer in one Triton kernel. | rounding, deterministic. |
| `folded_backward_tiles` | `auto`, `on`, `off` | On with `fused_folded_backward` (MH-1 class). | Picks 128-row tiles for that gradient kernel where 64-row tiles would give 2 to 8 waves over the GPU's multiprocessors. A rule from the GPU's size, not a per-GPU table. | exact (bitwise the same), deterministic. Needs `fused_folded_backward`. |
| `fused_radial_hidden` | `auto`, `on`, `off` | Off. Only `"on"` enables it. | Runs the radial network's hidden block as one pair of Triton kernels. | rounding, deterministic. Needs `fold_cutoff`. No measured gain. |
| `radial_tables` | `auto`, `on`, `off` | On with `fold_cutoff` (MH-1 class) in float32 and float64 plans that are not reproducible. | Replaces the radial chain (distance → edge weights and edge density) with per-species-pair piecewise Chebyshev tables, fitted when the plan is built. One Triton kernel evaluates them; the backward recomputes. | certified, deterministic. Needs `fold_cutoff`. A failed certificate turns it off. Float64 tables are held to float64 rounding. `"on"` is refused in reproducible float64 plans. |
| `w_tables` | `auto`, `on`, `off` | On only in float64 MH-1 plans that are not reproducible, on an NVIDIA GeForce RTX 5060 Ti or an NVIDIA RTX 4000 Ada Generation, with at most 3 species. Off elsewhere. | Builds the second interaction's per-edge weights straight from the radial tables (one table per piece already multiplied by the last layer), and their distance gradient without a separate product. | rounding, not deterministic. Needs `radial_tables` and `chunked_conv_backward`. `"on"` needs float64 outside reproducible plans. The tables grow with the square of the species count; above 2 GiB per interaction the rewrite does not apply. |
| `wprime_backward` | `auto`, `on`, `off` | On for MH-1 checkpoints in float32 plans that are not reproducible, on an NVIDIA A100-SXM4-80GB, an NVIDIA GeForce RTX 5060 Ti or an NVIDIA RTX 4000 Ada Generation (matched by exact name). Off on every other GPU, for other models, in float64 and in reproducible plans. | Computes the second interaction's distance gradient inside the streaming convolution's backward: each edge's weight gradient is multiplied with the derivative of its per-edge weights, read from per-piece tables. The weight gradient is never stored, and the folded layer's input gradient and the radial table's backward are not needed. | rounding, not deterministic. Needs `radial_tables` and `chunked_conv_backward`. `"on"` needs float32 outside reproducible plans; with it, any GPU takes it. MD step 6–19 % faster on the listed GPUs (`docs/validation/opt9-T03.md`). |
| `recompute_conv_weights` | `auto`, `on`, `off` | Off. On only when memory would not be enough: the planner turns it on for a cell that cannot fit without it, and a run that runs out of memory while it builds or captures its graphs is rebuilt once with it on. | The per-edge weights of the tensor product (1,280 floats per edge for MH-1) are not kept for the backward. With `fused_folded_forward` (MH-1's default) they never reach memory: the forward builds them on chip without storing them, and `wprime_backward`'s backward builds them on chip again (CAP-T07b). Without it, the backward builds them again with cuBLAS in edge chunks. | rounding, not deterministic: the rebuilt weights equal the stored ones bitwise, while `dsh` and the distance gradient are added over channel slices in another order. Needs `wprime_backward`. It saves about 5 kB per edge, about a third of the force call's memory on large cells, and costs 1–4 % step time. |
| `recompute_node_segments` | `auto`, `on`, `off` | Off. On only when memory would not be enough with `recompute_conv_weights` on: the planner turns it on, after `recompute_conv_weights`, for a cell that cannot fit with that one on, and a run that runs out of memory while it builds or captures its graphs is rebuilt with the memory rescue's next record on (`recompute_conv_weights` first, then `recompute_node_segments`). | Each interaction's `gated_rows` chain and product block runs under non-reentrant checkpointing: the force call keeps only their inputs, and the backward rebuilds their intermediates when it reaches them. The compiled model stays one graph. | rounding (Inductor fuses the recomputed segments differently), deterministic. Needs `gated_rows`. At NbCl5 21,600 atoms it lowered the force call's peak by 64 kB per atom (27 %) at 1.10 times the force call's time (CAP-T08). |

## Density and first interaction

| Option | Values | Default (`auto` means) | What it does | Exactness / limits |
| --- | --- | --- | --- | --- |
| `density_first` | `auto`, `on`, `off` | On with `fold_cutoff` (MH-1 class). | Divides by the edge density before the bias-free `linear_1` instead of after it. | rounding, deterministic. Needs `fold_cutoff`. |
| `reassociate_first` | `auto`, `on`, `off` | On with the hybrid backend where the first interaction's inputs depend only on species: the MH-1 variant needs `fold_cutoff`; the generic variant serves the generic MACE family and MACE-OFF. Off on plain cuEq. | Sums the first interaction per receiver and per sender species before its last radial layer, so the layer runs on far fewer rows. | rounding, deterministic. `"on"` on plain cuEq is refused. Built for the species superset. |
| `segmented_first_sum` | `auto`, `on`, `off` | On with `reassociate_first` and receiver-sorted edges. | Computes that per-species sum with a run-segmented Triton kernel over receiver-sorted edges. | exact, not deterministic (atomic additions). Needs `reassociate_first` and `NeighbourConfig.sort_receivers="on"`. |
| `species_skip` | `auto`, `on`, `off` | On with the graph engine on either backend where the model takes it (generic MACE family, MACE-OFF). | Restricts the element-dependent skip connection to the species of the structure (or the declared superset). | exact, deterministic. |

## Convolution (OEq streaming)

The convolution combines each edge's weights with its sender's features and
sums the messages per receiver.

| Option | Values | Default (`auto` means) | What it does | Exactness / limits |
| --- | --- | --- | --- | --- |
| `streaming_conv` | `auto`, `on`, `off` | On with the hybrid backend and receiver-sorted edges, where the model takes it: the MH-1 variant needs `fold_cutoff`; the generic variant serves the generic MACE family and MACE-OFF. | Runs the convolution with OEq's receiver-streaming kernels: messages are summed per receiver as they are produced, without a per-edge message tensor. | exact, not deterministic. Needs the hybrid backend and `NeighbourConfig.sort_receivers="on"`: on unsorted edges these kernels give wrong results while every self-check passes, so the planner refuses that combination. |
| `chunked_conv_backward` | `auto`, `on`, `off` | On with `streaming_conv` and `fused_folded_backward` (MH-1 class). | Runs the streaming backward in edge chunks sized to the GPU's L2 cache. Each chunk's weight gradient goes straight into the folded layer's input gradient, so the full per-edge weight gradient is never written to device memory. Cells whose weights fit in four L2 caches run unchunked inside the same kernel. | exact, not deterministic. Needs `streaming_conv` and `fused_folded_backward`. |
| `fused_folded_forward` | `auto`, `on`, `off` | On for MH-1 checkpoints (by label, or by SHA-256 and head) in float32 plans that are not reproducible. Off for other models, in float64 and in reproducible plans. | Fuses the folded last radial layer into the streaming forward: one kernel builds the per-edge weights and the message, and stores the weights for the chunked backward. | exact (the same float32 sums in the same order), not deterministic. Needs `chunked_conv_backward`. `"on"` needs float32; with it, any float32 model with `chunked_conv_backward` takes it. |
| `direct_node_feats` | `auto`, `on`, `off` | Off. Only `"on"` enables it. | In the folded MH-1 forward, skips the join of node features and attributes that MACE builds for LAMMPS, whose strided slices the streaming convolution had to copy. | exact, deterministic. Needs `fold_cutoff`. |
| `folded_constants` | `auto`, `on`, `off` | Off. Only `"on"` enables it. | Builds the folded last radial layer's weight-and-bias matrix and its halves once, not on every force call. | exact, deterministic. Needs `fold_cutoff`. |

## Linear layers

MACE's equivariant linear layers act per angular momentum `l`. cuEq runs each
as one generic kernel; these rewrites run them as ordinary matrix products.

| Option | Values | Default (`auto` means) | What it does | Exactness / limits |
| --- | --- | --- | --- | --- |
| `block_linear` | `auto`, `on`, `off` | On with the hybrid backend. | Runs each cuEq `Linear` as one matrix product per `l`, with the weights extracted once. | rounding, deterministic. |
| `grouped_linear` | `auto`, `on`, `off` | On with `block_linear`. | Puts input segments of the same irrep that feed one output segment into a single product. | rounding, deterministic. Needs `block_linear`. |
| `merged_residual` | `auto`, `on`, `off` | On with `block_linear`. | Runs `linear_1` and the residual linear as one product per output segment. | rounding, deterministic. Needs `block_linear` and `density_first`; does nothing on models without `density_first`. |
| `gated_rows` | `auto`, `on`, `off` | On with `merged_residual` (MH-1 class). | Runs the residual products, the gate and `linear_2` of MH-1-class interactions on per-segment rows, without the joins and copies between them. | exact, deterministic. Needs `block_linear` and `merged_residual`. |
| `segment_gemm` | `auto`, `on`, `off` | On with `gated_rows` in float32 plans that are not reproducible. | Writes `linear_2`'s products straight into its joined output with a gathered-row Triton matrix product, where a shared shape rule picks it. | rounding, deterministic. Needs `gated_rows`. The kernel is float32. |
| `gate_epilogue` | `auto`, `on`, `off` | On for MH-1 with `segment_gemm` on the RTX 5060 Ti and the RTX 4000 Ada (decision 66, by exact device name); off elsewhere. | Runs the gate's backward inside `linear_2`'s input-gradient matrix product in MH-1-class chains: one atom-tiled Triton kernel per gated segment stores the gated rows' gradient and sums the gates' gradient in registers, where the shape rule picks Triton. | rounding, deterministic. Needs `segment_gemm` and float32. Conflicts with `tf32x3_gemms`. |
| `grouped_launch` | `auto`, `on`, `off` | On with `block_linear` for the generic MACE family in float32 plans that are not reproducible. Off for other families. | Runs a block linear's small per-`l` products (too few row tiles to fill the GPU, by the shared shape rule) in one Triton launch. | rounding, deterministic. Needs `block_linear`. Conflicts with `tf32x3_gemms`. |
| `inplace_block_linear` | `False`, `True` (bool) | `False`. | Runs the per-`l` products as in-place strided-batched products. | `True` is refused: it was slower and is not carried over. |

## Species tables and folding

| Option | Values | Default (`auto` means) | What it does | Exactness / limits |
| --- | --- | --- | --- | --- |
| `species_tables` | `auto`, `on`, `off` | On where the model takes it, in float32 and float64 plans that are not reproducible, on either backend. | Replaces atom-level products that depend only on the species (embeddings, the first interaction's `skip_tp` and `linear_up`) with per-species tables, gathered each step. | rounding, deterministic (but not bitwise equal to the per-atom products on the GPU). Needs `ScaleShiftMACE`, a factored first interaction with species-only inputs, and no charge/spin embedding (so not OMOL-0). Tables are built for the species superset. |
| `species_residual` | `auto`, `on`, `off` | Off unless set to `"on"` (opt-in; not a default yet). | With `species_tables`, the first interaction's residual product `linear_res(linear_up(embedding))` depends only on the species, so it becomes per-species rows added to `linear_1`'s products; the merged product loses that half of its inputs and of its backward. Measured 0.4–0.8 % faster on the RTX 4000 Ada and 0.2–1 % on the RTX 5060 Ti's large cells (fp32 profile, 29 October 2026). | rounding. Needs `species_tables` and the `fold_cutoff` interaction forward. |
| `fold_readout_tail` | `auto`, `on`, `off` | On with `block_linear` where the model takes it, in plans that are not reproducible. | Folds the last readout's first linear into the last product's linear and `skip_tp`. | rounding, deterministic. Needs `block_linear`, `ScaleShiftMACE` and a factored last interaction (MH-1, OMOL-0). |
| `fold_product_linear` | `auto`, `on`, `off` | On with `species_tables` and `block_linear` where the model takes it, in plans that are not reproducible. | Folds layer 0's product linear into the linear layers that read layer 0's output, with the species carried next to the contraction. | rounding, deterministic. Needs `block_linear` and `species_tables`. |

## Compiler (Inductor)

These change how `torch.compile` builds the model. They have no effect when
nothing is compiled (`EngineConfig.compile_mode="none"`).

| Option | Values | Default (`auto` means) | What it does | Exactness / limits |
| --- | --- | --- | --- | --- |
| `combo_kernels` | `auto`, `on`, `off` | Off. Only `"on"` enables it. | Turns on Inductor's combo kernels for pointwise operations: independent adjacent pointwise kernels run in one launch. | rounding (the energy moves by about 1e-9 relative), deterministic. |
| `save_transcendentals` | `auto`, `on`, `off` | On only in float64 MH-1 plans that are not reproducible, on an NVIDIA GeForce RTX 5060 Ti or an NVIDIA RTX 4000 Ada Generation. Off elsewhere. | A custom partitioner saves the results of transcendental operations (exp, tanh, sigmoid and others) for the backward pass instead of recomputing them there. | exact, deterministic. May use more memory, since the saved tensors are kept until the backward. |

## Precision tricks

| Option | Values | Default (`auto` means) | What it does | Exactness / limits |
| --- | --- | --- | --- | --- |
| `tf32x3_gemms` | `auto`, `on`, `off` | Off. Only `"on"` enables it. | Runs every atom-level matrix product (block linears, `segment_gemm`, species skip, reassociated first interaction) and its input gradient as three TF32 tensor-core products (Triton's `tf32x3`), which together reach about float32 accuracy. | rounding, deterministic; not equivalent to IEEE float32 when terms cancel. Needs `block_linear`, float32 and compute capability 8.0 or higher. Conflicts with `grouped_launch`: set `grouped_launch="off"` with it on the generic MACE family. |

The float64 emulation on integer tensor cores, and its mantissa bits, are
engine options (`EngineConfig.fp64_emulation`,
`EngineConfig.fp64_emulation_mantissa_bits`), not rewrites.

## PolarMACE

These apply only to MACE-POLAR-1 models.

| Option | Values | Default (`auto` means) | What it does | Exactness / limits |
| --- | --- | --- | --- | --- |
| `polar_uvu` | `True`, `False` (bool) | `True`. No effect on other models. | Reassociates PolarMACE's sparse UVU tensor product (the l × l → 0 paths contract the weight first, as one matrix product) instead of forming the pair tensor. | rounding, deterministic. |
| `fx_e3nn_linear` | `auto`, `on`, `off` | On for the checkpoints `polar1-s`, `polar1-m` and `polar1-l` (identified by SHA-256) in hybrid float32 plans that are not reproducible. Off for every other model. | Builds e3nn's `o3.Linear` as an fx graph instead of the TorchScript a checkpoint restores, so the compiler can trace PolarMACE's whole forward. | rounding, deterministic. |
| `polar_graph_sums` | `auto`, `on`, `off` | As `fx_e3nn_linear`. | Computes PolarMACE's per-structure sums (Fukui norms, total charges, barycentre) as a sum over atoms and a broadcast, not with `scatter_add_` and an index. | rounding, deterministic (no atomics). Needs one structure per evaluation, which is always the case. |

## Stress

| Option | Values | Default (`auto` means) | What it does | Exactness / limits |
| --- | --- | --- | --- | --- |
| `single_graph_strain` | `auto`, `on`, `off` | On for every engine. It acts only where stress is computed (variable-cell MD, relaxations with a cell filter). | Uses MACE PR 1498's one-structure strain: the strained cell is used directly instead of being gathered per edge. Bound into the force field's own forward; nothing in MACE is patched process-wide. | rounding, deterministic. Needs one structure per evaluation. |

## PrecisionConfig

The force evaluation runs in float32 or float64. **Integration is always
float64**: positions, momenta, thermostat and barostat state are float64
whatever the force precision. The potential energy is summed in float64 in
both precisions.

Float32 is the default and the fast path: several rewrites (`segment_gemm`,
`grouped_launch`, `fused_folded_forward`, `wprime_backward`, `tf32x3_gemms`, the PolarMACE
compiler rewrites) are float32 only. Float64 is slower; how much depends on
the GPU's float64 rate. Choose float64 when:

- you need energies or forces beyond float32 rounding, for example for
  finite-difference checks or long NVE runs where float32 drift matters;
- you compare against a float64 reference or another float64 code;
- the gate (`check="gate"`) fails in float32 on your system.

On GPUs with a reduced float64 rate (consumer and workstation cards, about
1/64 of float32) the planner turns on cuBLAS's float64 emulation on integer
tensor cores (`EngineConfig.fp64_emulation`), which keeps float64 accuracy.

Not every field is set through `PrecisionConfig` by users. The table says
where each one comes from.

| Option | Values | Default (`auto` means) | What it does | Exactness / limits |
| --- | --- | --- | --- | --- |
| `dtype` | `auto`, `float32`, `float64` | Set with `fm.ForceField(model, atoms, precision=...)` (default `"float32"`); `fm.md` always uses float32. The planner's own `"auto"` rule (float32 for the hybrid backend, float64 for plain cuEq) is not used for user plans: the planner takes the force field's precision. | Precision of the model and of the force evaluation. | On a MACE install whose float64-energy patch cannot be applied (its source hash does not match), a non-PolarMACE float32 force field is promoted to float64, so the energy is still summed in float64. The plan records this as the `force_precision` decision. |
| `tf32` | `False`, `True` (bool) | `False`. Not settable through `Simulation`, `fm.md` or `ForceField`. | TF32 tensor-core matrix products. | Never on by default. `True` is only a label in the plan ("TF32" in the header); FastMACE never switches PyTorch's TF32 mode. `metadata.json` records the real flag as `precision.tf32` and this field as `precision.tf32_requested`. If your script enables TF32 (`torch.set_float32_matmul_precision("high")`), float32 forces are no longer IEEE float32. |
| `float64_energy` | `True`, `False` (bool) | `True`. Not settable through `Simulation`, `fm.md` or `ForceField`. | Accumulates the potential energy in float64 (forces and stress are MACE's either way). | `False` reproduces the research engine's float32 energy sum, for A/B comparisons only. |
| `mc_energy` | `auto`, `float32`, `float64` | Set with `fm.Workload(mc_energy_precision=...)`. `"auto"`: the force precision if the move-energy gate passes, float64 otherwise. Without MC moves it equals the force precision. | Precision of the energies that Monte Carlo moves compare. MD forces keep `dtype`. | `"float32"` with float64 forces is refused: MC energies are never evaluated below the force precision. |

## When to change these

You almost never need to. The `"auto"` rules come from same-process A/B
timings and force gates, and every rewrite is checked when it is applied.
Leave `rewrites="auto"` unless one of these applies.

**To bisect a gate failure or a suspicious result.** Turn rewrites off one at
a time, or a whole chain at its root (`factor_species="off"` or
`block_linear="off"`), and compare. The automatic fallback after a failed gate
does the same search and records it in `sim.gate`. `Simulation(mode="reference")`
runs with no rewrites at all.

**To try the opt-in rewrites.** These are off under `"auto"` because they gave
no gain, or a loss, in the measurements so far:

| Option | Measured |
| --- | --- |
| `direct_node_feats`, `folded_constants` | Together, NaCl about 0.5–1 % faster on an RTX 5060 Ti; NbCl5 and an L4 neutral. |
| `combo_kernels` | NaCl about 1–2 % faster, NbCl5 2–3 % slower on an RTX 5060 Ti and an L4. |
| `fused_radial_hidden` | No measured gain. |
| `tf32x3_gemms` | An accuracy screen; slower than cuBLAS float32 on most production shapes measured. |

Test them with a same-process A/B on your own GPU and structure before you
rely on them.

**Float64 on other GPUs.** `w_tables` and `save_transcendentals` are on only
on the two GPU models where they were measured faster in float64 MH-1 runs (an
RTX 5060 Ti and an RTX 4000 Ada): per MD step `w_tables` 0.86–0.94 and
`save_transcendentals` 0.97–0.98 of the time without them. On an A100, which
runs float64 at half the float32 rate, both were slower (`w_tables` 1.06–1.27,
`save_transcendentals` up to 1.02). On another reduced-rate GPU you may set
them `"on"` and measure. Unmeasured GPUs keep them off even when they share a
compute capability with a measured one.

**For reproducible plans.** With `fm.Workload(reproducible=True)` the planner
uses plain cuEq and turns off every rewrite that is not deterministic or that
is kept for non-reproducible plans only (`radial_tables`, `species_tables`,
`fold_readout_tail`, `fold_product_linear`, `segment_gemm`, `grouped_launch`,
`fused_folded_forward`, `w_tables`, `wprime_backward`, `recompute_conv_weights`, `save_transcendentals`,
`fx_e3nn_linear`, `polar_graph_sums`). An explicit `"on"` for a rewrite that is not
deterministic is refused.

**To save memory.** `radial_tables` and `w_tables` hold tables that grow with
the square of the species count. Turn them off if the plan's memory estimate
is close to the GPU's memory.
