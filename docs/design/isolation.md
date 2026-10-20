# Instance isolation: the audit (phase 2)

Plan task 2-T08. Spec: §8.7 "Instance isolation", §12.2 "Isolation", phase 2
acceptance 5. Code audited at `main` 8c6e22e: `model/forces.py`,
`model/compile.py`, `longrange/polar.py`, `rewrites/engine.py`,
`rewrites/registry.py`, `_patches.py`, `engine/graph.py`,
`engine/constraints.py`, `kernels/oeq_streaming/runtime.py` and every
module-level object of the package. Tests: `tests/cpu/test_isolation.py` and
`tests/gpu/test_isolation_gpu.py`; results in
[docs/validation/2-T08-isolation.md](../validation/2-T08-isolation.md).

SPEC §8.7 asks that several force fields and engines live in one process and
are used in any order, A → B → A, each giving what a fresh instance gives.
Each instance owns its geometry and capture state; compiled code shared
between instances receives per-instance state as data; a component that
cannot be made per-instance is held under an exclusive lease.

## What two instances can share

"Shared" means reachable from two force fields or engines in one process.
The question for each object is whether a call through one instance can
change what another instance computes.

### Owned by each instance

| Object | Where | Why it is per instance |
| --- | --- | --- |
| Input batch: species one-hot, head, charge and spin, float64 E0 per atom, cell, volume, reciprocal cell, displacement leaf | `MACEForceField.data` | built in `__init__` from the instance's atoms; never stored on the model |
| Positions leaf the model differentiates | `MACEForceField.positions` | per instance, refilled before every call |
| Neighbour list: candidate pairs, compacted edges, shifts, device edge count, build cell, reference positions, sender groups | `SkinNeighbourList` | one per force field |
| Streaming edge count read by the streaming interactions | `neighbours.count`, bound by `_bind_streaming_count` | the model is shared, the counter is not: every `_outputs` call points the model's `streaming_count` buffers at its own counter (a host-side pointer swap). Captured graphs keep the address they were captured with. Dynamo reads the buffer as an input (no recompilation per swap, tested) |
| PolarMACE k-space state (`PolarHolder`, `PolarState`: k-vector superset, mask, bounds, sticky overflow flag) | the holder on the force field's private copy | a polar force field is never put in the compiled-model cache (`key=None`) and always enables its own copy |
| PolarMACE per-call state (`_Call`) | `polar._ACTIVE` | call context: pushed and popped within one forward, empty outside it |
| Engine state: captured graphs, graph pool, side stream, flags, saved mid-step tensors, runtime-state registry, edge buckets | `GraphEngine` | one per engine |
| Integrator state and noise stream | `SimulationState`, integrator | one per engine |

### Shared, read-only

| Object | Why sharing is safe |
| --- | --- |
| The compiled model of `CompiledModelCache` (one per key: source model, mode, backend, enabled records, species columns for species-dependent records, compile switches, energy treatment) | weights and rewritten modules are not written by any call; per-instance state reaches it through the batch dict and the streaming-count swap |
| The source model (eager plan without model rewrites uses it as it is) | evaluated, never written; the float64 energy forward is a function called on it |
| `energy._FORWARDS` | rewritten forward functions by class, stateless |
| OEq streaming kernels (`oeq_streaming.runtime._REGISTRY`) | compiled cubins by id; outputs and gradients allocated per call, no workspace |
| PolarMACE class-level wrappers (`polar.install`) and `allow_irrep_guards` | installed once, dispatch on the per-instance holder |
| Caches of facts: `provenance._DIRTY`, `provenance._SOFTWARE`, `loading._LICENCES_SHOWN`, `pair_search` and `jinja_env` caches, `engine.base._PRODUCTION_ENGINES` | facts about the process or the checkout, the same for every instance |
| Inductor, Triton and OEq disk caches | keyed by content |

### Shared and process-wide (SPEC §8.7 process constraints)

| Object | Effect on other instances | Handling |
| --- | --- | --- |
| Dynamo's code cache, reset by `torch._dynamo.reset()` before each PolarMACE compile (`forces.py`) | every compiled model in the process loses its compiled code; captured graphs keep their kernels | **correct, slower** (conductor answer Q3): an instance recompiles at its next call that is not a replay (a one-off evaluation, the warm-up of a capture at a new edge bucket). The GPU test counts it and checks the result after it |
| `dynamo.config.cache_size_limit` | raised to 64, never lowered | no isolation issue |
| `capture_safe_mask_head` patch (`_patches`) | the same replacement for every instance; reinstalled before every capture (1a-T13b) | idempotent inside FastMACE; only foreign code (the research engine in an A/B process) replaces it |
| `first_derivative_only`, `unfuse_full_addmm` patches | the same for every instance; never undone by the library | idempotent |
| `single_graph_strain` patch | since 2-T08b read by the reference calculators only; no force field reads it | closed (finding 1): each force field binds its strain path per model |
| Compile family and reproducibility of the process (`engine.constraints.PROCESS`) | refused, not shared state | 2-T07 (subprocess runner), 2-T16 |

## Findings

1. **A failing `single_graph_strain` self-check undid the patch for every
   earlier force field. Closed by 2-T08b.**
   - What happened before 2-T08b: `Registry._selfcheck_failure` called
     `_patches.restore(name)` when the process-wide `single_graph_strain`
     record failed its self-check (and a reference calculator whose record
     has the stock strain off removed it too). A force field A built earlier
     with the record kept running on MACE's function: energy and forces
     unchanged, stress at rounding level (8.7e-19 eV/Å³ on the tiny CPU
     model), a recompile at its next non-replayed call, captured graphs
     unaffected, `A.applied` stale.
   - The fix (conductor decision, 20 October 2026; SPEC §8.7 "MACE
     patches"): each force field binds its strain path into its own
     forward, on or off ([strain-per-model.md](strain-per-model.md)). The
     record installs nothing, so a failing self-check restores nothing, and
     the reference calculators' process patch reaches no force field. The
     bound path is in the force field's provenance (`info()["strain_path"]`).
   - Tests: `tests/cpu/test_isolation.py`
     (`test_failing_strain_selfcheck_leaves_earlier_force_fields_unchanged`,
     which pinned the old behaviour before, and
     `test_bound_strain_path_ignores_the_process_patch`) and
     `tests/cpu/test_strain_per_model.py`.

## Leases

None needed: every per-instance state is owned or passed as data, and the
dynamo reset leaves results correct. `rebind()` and `IsolationError` are
therefore not used (conductor answer Q1: only where a lease is needed).
Finding 1 was closed by making the strain path per model, not by a lease.
No shared state is left open.
