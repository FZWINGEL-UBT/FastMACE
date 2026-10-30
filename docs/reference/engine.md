# Engine and neighbour list

Two records hold the options of the engine that steps the simulation:

- `EngineConfig`: the backend, compilation, CUDA graphs, pipelining,
  allocator and float64 settings;
- `NeighbourConfig`: the skin neighbour list and the padded edge buffers.

Pass them to `fm.Simulation` or `fm.md` as `engine=` and `neighbours=`. They
are not in the `fm` namespace; import them from `fastmace.config`.

```python
import fastmace as fm
from fastmace.config import EngineConfig, NeighbourConfig

model = fm.load("mh1", head="matpes_r2scan")
sim = fm.Simulation(
    atoms, model,
    integrator=fm.integrators.Langevin(temperature_K=1200, friction_per_fs=0.02),
    seed=1,
    engine=EngineConfig(backend="cueq", recapture_budget=16),
    neighbours=NeighbourConfig(skin=0.5),
)
print(sim.plan.overrides())   # the options you set, and what the planner did with them
```

`fm.md(..., engine=EngineConfig(...), neighbours=NeighbourConfig(...))`
takes the same records.

Both records are frozen. A value outside the allowed set raises `ValueError`
when you build the record, before anything is loaded. An option the plan
cannot honour raises `PlanRefused` with the reason when the `Simulation` is
built, before the first step.

**A record you pass replaces the default one as a whole**, so set every
field you need in one `EngineConfig`. Without `engine=`, a `Simulation` uses
`EngineConfig()`, on `cuda:0`. Only `fm.ForceField.evaluate`, which takes no
`engine=`, falls back to `EngineConfig(device="cpu", compile_mode="none")`
on a machine without CUDA.

For `"auto"`, `"on"` and `"off"` see the
[reference overview](index.md#auto-on-and-off). Each resolved value and its
reason is in `sim.plan.describe(full=True)`, under the key
`EngineConfig.<field>` or `NeighbourConfig.<field>`. The
[planner guide](../guide/planner.md) explains how a plan is made.

## Terms used on this page

- **CUDA graph capture.** The engine records the GPU work of one stage of a
  step (for example the force call) once as a CUDA graph and then replays
  it every step. A replay launches all kernels at once, without Python in
  between. A graph is tied to fixed tensor shapes and addresses.
- **Re-capture.** Recording a graph again, or recording a new variant of a
  stage that was captured before: for example when the number of edges
  moves into a new bucket, after a buffer had to grow, or after an event
  dropped the graphs. Each re-capture costs host time, and the step that
  needs it waits.
- **Skin.** The neighbour list stores every pair closer than cutoff + skin.
  That list stays valid until some atom has moved more than skin / 2 since
  the last build. A larger skin means fewer rebuilds and more pairs per
  step.
- **Edge buffer and edge bucket.** The model reads the pairs inside the
  cutoff (the edges), padded to a fixed length so that one captured graph
  fits many steps. That length is the edge count rounded up to a bucket.
  The padded edges lie beyond the cutoff and contribute exactly zero. Each
  bucket that the run visits costs one graph per stage.
- **Tiers.** `graph` (compiled model and CUDA graphs), `compiled` (compiled
  model, no graphs), `eager` (no compilation, no graphs) and `reference`
  (MACE's own calculator). See the
  [planner reference](planner.md#tiers).
- **Backends.** `hybrid`: cuEquivariance for most of the model and
  OpenEquivariance kernels for the convolutions. `cueq`: plain
  cuEquivariance throughout. See the
  [planner reference](planner.md#backends).

## `EngineConfig`

| Option | Values | Default (`auto` means) | What it does | Notes |
| --- | --- | --- | --- | --- |
| `backend` | `"auto"`, `"hybrid"`, `"cueq"` | `"auto"`: `hybrid` when OpenEquivariance can build, the device is CUDA, MACE was not imported before FastMACE, and the process has not compiled a plain cuEq model; else `cueq`, with the reason | Chooses the model backend. | `reproducible=True` in the workload resolves `auto` to `cueq` and refuses `hybrid`. `hybrid` on a CPU device is refused. The reference tier always uses MACE's stock cuEq model. |
| `compile_mode` | `"default"`, `"none"` | `"default"` | `"default"` compiles the model with `torch.compile`. `"none"` runs it uncompiled. | The eager and reference tiers set `"none"` themselves. If the installed MACE lacks the compile interface, the planner sets `"none"` and records why. With `"none"` on the graph tier, the uncompiled model is still captured in CUDA graphs. |
| `device` | a torch device string | `"cuda:0"` | The device of the model and the simulation state. | `fm.ForceField.evaluate` uses `"cpu"` on a machine without CUDA. |
| `pipelined` | `"auto"`, `"on"`, `"off"` | `"auto"`: on for MD, unless `speculative_graph` is set | Launches the force stage of a step before the host has read the step's neighbour flags. The step is checked while the force stage runs. A step that needed a rebuild, or more edges than were padded, is restored and redone. | Off for relaxations (an explicit `"on"` there is refused). `"on"` together with `speculative_graph=True` is refused. The reference tier turns it off. |
| `async_output` | `"auto"`, `"on"`, `"off"` | `"auto"`: on for MD and relaxations | Copies thermo samples and trajectory frames behind their step and writes them while the next steps run. | Off for one-off evaluations (`fm.calc`, `ForceField.evaluate`). |
| `speculative_graph` | `True`, `False` | `False` | One combined graph per step (integrator and force stage), checked one step later and rolled back when a rebuild was needed. | NVT with a fixed cell only: refused with NPT and for relaxations. Turns `pipelined` off. Opt-in. |
| `pipeline_edge_margin` | float, ≥ 0 | `0.0` | Pipelined steps pad the force stage to the last checked edge count × (1 + margin), rounded up to the bucket. | A margin makes count redos rarer, at the cost of more padded edges every step. |
| `pipeline_rebuild_fraction` | float, > 0 and ≤ 1 | `0.8` | Pipelined steps rebuild the neighbour list early once an atom has moved this fraction of skin / 2. | Keeps redos after a late rebuild rare. 1 rebuilds only when the list is invalid. |
| `cueq_conv_fusion` | `"auto"`, `"on"`, `"off"` | `"auto"`: on with the `cueq` backend, except in reproducible plans; off with `hybrid` | Uses cuEquivariance's fused convolution kernel in plain cuEq models. | `"on"` with `hybrid` is refused (the hybrid fuses its convolution anyway). `"on"` with `reproducible=True` is refused. If FastMACE cannot verify the installed MACE's conversion code, MACE's own rule decides and a different explicit value is refused. |
| `first_derivative_only` | `True`, `False` | `True` | Compiled cuEq models trace MACE's force function without the graph for second derivatives. Exact. | No effect on hybrid models. |
| `unfuse_full_addmm` | `True`, `False` | `True` | Keeps Inductor from fusing a matrix product with a full-matrix add. Same formulas, different kernel grouping. | Compiled models only. |
| `dynamo_cache_size_limit` | int, > 0 | `64` | Raises `torch._dynamo`'s per-function cache limit to this value (never lowers it). | Beyond the limit, dynamo silently runs the model uncompiled. Must be at least `graph_variant_cap`. |
| `expandable_segments` | `"auto"`, `"on"`, `"off"` | `"auto"`: on with torch 2.14 or later on CUDA, unless `PYTORCH_CUDA_ALLOC_CONF` is set to something without `expandable_segments` | Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before the first CUDA allocation. This lets the allocator grow memory segments instead of fragmenting them. | Off on the CPU. It acts only if set before the process makes its first CUDA allocation: plan before you allocate anything on the GPU. |
| `fp64_emulation` | `"auto"`, `"on"`, `"off"` | `"auto"`: on for float64 plans on CUDA that are not reproducible, on GPUs whose measured float64 rate is below 1/8 of their float32 rate | Runs float64 matrix products through cuBLAS's emulation on INT8 tensor cores. On consumer and workstation GPUs (float64 at about 1/64 rate) this was faster per MD step (MH-1 on an RTX 5060 Ti and an RTX 4000 Ada), with products at least as accurate as native float64. | Off in float32 plans, on the CPU and on full-rate GPUs (A100, H100). `"on"` with `reproducible=True` is refused. The rate is measured with a short probe on the device. If a cuBLAS handle existed before the plan, the products stay native and the build warns. |
| `fp64_emulation_mantissa_bits` | `"auto"`, `"dynamic"`, int 8–128 | `"auto"`: 56 for float64 MH-1 plans that are not reproducible, on the GPUs where it was measured faster (NVIDIA GeForce RTX 5060 Ti, NVIDIA RTX 4000 Ada Generation); `"dynamic"` elsewhere | The mantissa bit count of the emulation (`CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT`). `"dynamic"` lets cuBLAS choose. | No effect while `fp64_emulation` is off. A value set in the environment wins. On the measured GPUs, 56 bits gave MD steps 0.86–0.88 of the time of `"dynamic"`, with the same product errors. |
| `empty_cache_before_capture` | `True`, `False` | `True` | Releases cached free GPU memory between a stage's warm-up runs and its capture. | Lowers peak memory of the graph pool. |
| `warmup_in_graph_pool` | `True`, `False` | `True` | After the first capture of a stage kind, runs later warm-ups inside the graph memory pool. | Keeps warm-up allocations from fragmenting ordinary memory. |
| `dynamic_edges` | `True`, `False` | `True` | Marks the edge dimension as dynamic from the first compiled call. A new edge count then does not compile the model again. On the graph tier, the first padded capture reuses the first compiled graph, which saves one compile per start (7–9 s with warm compiler caches, 16–22 s cold, on an RTX 5060 Ti). | The compiled tier and one-off evaluations always use it. Plans that compile nothing turn it off. Off for PolarMACE with a variable cell on the graph tier. Trajectories are the same with and without it. |
| `model_template_cache` | `True`, `False` | `True` | Keeps each converted model as a template and gives later plans in the process a private copy, instead of loading and converting the checkpoint again. The templates belong to the `fm.load` result, so force fields of other compositions made from it share them. | CUDA with `cueq` or `hybrid` only. Saves setup time when one process builds several simulations. |
| `overlap_rebuild` | `True`, `False` | `True` | On a pipelined fixed-cell step, runs an early neighbour rebuild on a side CUDA stream next to the force stage in flight. Exact. | Acts only on pipelined fixed-cell CUDA steps. |
| `step_graph` | `True`, `False` | `False` | Pipelined steps replay one CUDA graph per step instead of two. Exact. | Opt-in. |
| `fused_integrator` | `True`, `False` | `True` | Fixed-cell Langevin on CUDA runs the integrator stage (half kick, drift, neighbour checks, compaction, flags) as four Triton kernels instead of about 45 PyTorch kernels. Exact: same float64 operations, same order, same noise. | Stays on in reproducible plans. NPT, the CPU and installations without Triton use the PyTorch stage. |
| `recapture_budget` | `"auto"` or int ≥ 0 | `"auto"`: `graph_variant_cap` (64 by default) | The number of re-captures one run may make before it warns (`RecaptureBudgetWarning`), records the step, and the report recommends the compiled tier. | The first capture of each graph does not count. `0` warns at the first re-capture. Past the budget, runtime adaptation may switch the run to the compiled tier (see [planner reference](planner.md#adaptationconfig)). |
| `force_graphs` | `"auto"`, `"keep"`, `"fit"`, `"one"` | `"auto"`: `"fit"` with CUDA graphs, else `"keep"` | How many force graphs (one per padded edge size) live at once. `"keep"` keeps every variant up to `graph_variant_cap`. `"one"` releases the captured graphs and their memory pool before capturing a new force variant. `"fit"` does that only when the new variant would not fit the free GPU memory. After a release the padded edge size is never lowered again, so the run does not capture again when the edge count falls back. | Exact: padded edges contribute nothing. Each force graph holds a force evaluation's activations, so near the memory limit a second variant is what fails first. A release costs one extra capture of every stage. |
| `graph_variant_cap` | int ≥ 1, ≤ 64 and ≤ `dynamo_cache_size_limit` | `64` | The most captured graph variants kept at once. When full, the least recently replayed one is dropped; capturing it again counts as a re-capture. | A lower cap saves GPU memory and makes re-captures more likely. |

## `NeighbourConfig`

| Option | Values | Default (`auto` means) | What it does | Notes |
| --- | --- | --- | --- | --- |
| `skin` | `"auto"` or float > 0, Å | `"auto"`: 0.3 Å for MD; for runs with MC moves or declared species changes, the skin sized from the workload; for relaxations, 3 × the optimiser's `maxstep_A` and at least 0.3 Å | The neighbour-list skin. | The workload rule takes the fastest atom expected at `Workload.max_temperature_K` (lightest mass, timestep) and sizes the skin so that a rebuild comes about every 16 steps, at least 0.3 Å. Displacement moves add 2 × `max_displacement_A`. For plain MD the plan records what that rule would give, but keeps 0.3 Å. An explicit value is kept and shown as an override. |
| `sort_receivers` | `"auto"`, `"on"`, `"off"` | `"auto"`: on with the `hybrid` backend, off with `cueq` | Orders the candidate pairs by receiver atom at every rebuild. | The hybrid's streaming convolution and the segmented first-interaction sum need it: turning it off turns them off, and asking for them with it off is refused. |
| `gpu_neighbour_list` | `"auto"`, `"on"`, `"off"` | `"auto"`: on | Builds the neighbour list on the GPU instead of with matscipy on the host. | Both find the same pairs; only the order within a receiver differs. On a CPU device the list is always built with matscipy. |
| `edge_multiple` | int > 0 | `256` | Edge buffer capacities and bucket sizes are multiples of this. | Mostly for alignment. |
| `edge_bucket_fraction` | float ≥ 0 | `0.05` | The bucket size as a fraction of the first edge count, rounded up to `edge_multiple`. | Larger buckets mean fewer graphs and re-captures, and more padded edges per step (at most one bucket). `0` uses buckets of `edge_multiple` edges. |
| `capacity_headroom` | float ≥ 1 | `1.15` | The pair buffer holds this many times the pairs of the first build. A list that outgrows it is reallocated, which drops the graphs that read it. | With `headroom="auto"` and an MC or species-change workload, the planner sizes it from the skin and the cutoff (at least 1.15). |
| `headroom` | `"auto"`, `"fixed"` | `"auto"`: resolves to `"fixed"` with `capacity_headroom` sized for MC and species-change workloads | `"fixed"` uses `capacity_headroom` and `edge_bucket_fraction` as given. | For plain MD `"auto"` keeps the given values. The plan always records `"fixed"` with the value it used. `edge_bucket_fraction` is never resized. |
| `d3_skin` | `"auto"` or float > 0, Å | `"auto"`: sized for the D3 pair list from the fastest atom and, under a barostat, the expected cell strain | The skin of the D3 dispersion pair list, which is separate from the model's list. | Read only when the force field has D3. Without D3 the plan records it as unused. |

## When to change these

- **Hot MD runs.** For plain MD `skin="auto"` stays at 0.3 Å whatever the
  temperature. Above about 1000 K, or with light atoms (H, Li) and a long
  timestep, rebuilds become frequent. Set `NeighbourConfig(skin=0.5)` or
  more, and check the rebuild count in the report.
- **Many re-captures.** If the report shows a `RecaptureBudgetWarning`,
  the edge count moves through many buckets (melting, large volume
  changes, MC). Try a larger `edge_bucket_fraction` (for example 0.1) or
  `capacity_headroom` (for example 1.3), or plan the compiled tier with
  `mode="compiled"`. Runtime adaptation may also make that switch for you.
- **Recapture budget.** Lower `recapture_budget` to be told earlier that
  graphs do not pay off. Raising it above `graph_variant_cap` has little
  use: past the cap, graphs are evicted and captured again.
- **Debugging.** Run the same structure with `mode="eager"` (see
  [planner reference](planner.md#simulation-keyword-arguments)). It keeps
  the backend, rewrites and precision but has no compilation and no CUDA
  graphs, so errors point at the line that raised them. `pipelined="off"`
  checks every step before the next one starts.
- **Memory.** Leave `expandable_segments="auto"`. If you set
  `PYTORCH_CUDA_ALLOC_CONF` yourself, include `expandable_segments:True`,
  or the planner leaves the allocator alone. Plan before anything else
  allocates on the GPU.
- **Comparing backends.** Set `backend="cueq"` or `backend="hybrid"`
  explicitly; `"auto"` may already pick plain cuEq when OpenEquivariance is
  missing or MACE was imported first. `fastmace doctor` names the reason.
- **Float64 on consumer GPUs.** `fp64_emulation="auto"` already turns on
  where it pays. Set `"off"` to compare against native float64 products.
- **CPU runs** (testing only; FastMACE's engines are for NVIDIA GPUs). Pass
  `EngineConfig(device="cpu", compile_mode="none")`.
