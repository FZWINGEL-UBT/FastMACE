# Interfaces: integrator, force field, engine, plan (phase 1a)

Plan task 1a-T02. Spec: §5 (layering), §7.1, §8.1, §9.5, §10.1, §10.6.
Source: ResearchModule tag `research-engine-baseline-20261017`,
`langevin/gpu_langevin.py` (`MACEForces`, `GPULangevin`) and
`langevin/gpu_npt.py` (`GPULangevinBAOAB`). Stubs: `fastmace/state.py`,
`fastmace/model/forces.py`, `fastmace/integrators/base.py`,
`fastmace/engine/base.py`, `fastmace/planner/plan.py`.

## The cut

`GPULangevin` is integrator and engine at once. The cut goes along its own
method boundaries: **the integrator is `_load_noise`, `_stage_a` and the kick
in `_stage_b`. The engine is everything else.** The engine also does the
three neighbour-list lines inside `_stage_a` (`set_cell`, the displacement
check) and the force call with its copies at the start of `_stage_b`. The
integrator never calls the force field. It never touches the neighbour list
and never captures or synchronises. The engine holds no MD maths. Each graph
records the same kernels in the same order as the baseline. The cut changes
only which Python object issues them, so the graph node counts must match
the baseline (T13 checks this).

## One MD step

| Baseline (`GPULangevin`) | FastMACE | Where |
| --- | --- | --- |
| `_load_noise` | `integrator.prepare_step(state)` | host, before the graphs |
| `_stage_a`: kick, drift (BAOAB: barostat, cell, O steps) | `integrator.first_half(state)` | in A |
| `force_field.set_cell(cell)` (BAOAB only) | engine: `ff.set_cell(state.cell)` if `integrator.variable_cell` | in A |
| `rebuild_needed ← max_displacement_exceeded(x)` | engine: `ff.needs_rebuild(x)` → own flag | in A |
| `_stage_c` | engine: `ff.compact(x)` | C |
| `_flags ← (rebuild, count)` | engine flags | end of AC |
| host reads flags; on a rebuild: `rebuild`, `C`, read count | engine (host) | between graphs |
| `use_edges(padded n)` | engine: `ff.use_edges(n)` | host, before B |
| `_stage_b`: `ff(x)`, copy f, E (stress) | engine: `ff(state.positions)` → `state.forces/energy/stress` | in B |
| `_stage_b`: (BAOAB `_update_force_eps`), kick(s) | `integrator.second_half(state)` | in B |

Graph keys stay as they are: `AC`, `("B", edges)`, `C`. **Pipelined**
(`ACP`): AC, then `engine.save_mid` copies `integrator.mid_step_tensors`
(Langevin `v`; BAOAB `v`, `p_eps`) into engine buffers. Then it writes
`[rebuild, count, ff.needs_rebuild(x, PIPELINE_REBUILD_FRACTION)]` to the
pipe flags and starts their non-blocking copy, and launches
`("B", predicted edges)` without waiting. Validation, one step later on the
host, restores the mid tensors and redoes C and B on a rebuild or an
overflow, or does an early rebuild. **Speculative** (`("ACB", edges, slot)`):
AC, save_mid, slot flags and B in one graph, with the same restore and redo
of B. **Async output**: `engine.snapshot()` validates the step in flight. It
then queues pinned copies of `integrator.output_values(state)` behind it and
records an event. `Snapshot.wait()` gives the values.

At construction the engine does `set_cell` if the cell is variable,
compaction, `use_edges`, the initial force call into the state, then
`integrator.start(state)` (BAOAB's starting cell force). That is the
baseline order.

## Who owns what

| Object | Owns (fixed device addresses unless noted) |
| --- | --- |
| `SimulationState` | `positions`, `velocities`, `masses (N,1)`, `cell`, `species`, `forces`, `energy`, `stress`, the committed results. `rng`: numpy generator, drawn-ahead `batch`, `cursor`. `step` (host) |
| `Integrator` | coefficients (`c1…c5`, OU factors, barostat mass), target scalar `sqrt_kT`, scratch (`x_old`, `rnd_pos`, `rnd_vel`, noise rows `xi`, `eta`, `zeta`), BAOAB `p_eps`, `force_eps`, `externalstress`. Allocates `state.rng.batch` (its layout) |
| `ForceField` | model (compiled, rewritten), input batch, skin neighbour list (edge buffers, device count, build cell, reference positions), model inputs `cell`, `volume`, `rcell`, `displacement`, the streaming edge counter (no `bind_streaming_count`) |
| `Engine` | graphs, pool, side stream, edge bucket and padded sizes, `rebuild_needed`, flags and pinned mirrors (`_flags`, `_pipe_flags`, `_spec_flags`), mid copies (`_v_mid`, `_p_mid`), the step in flight, floors, counters, capture records |
| `Plan` | frozen: tier, resolved configs (`RewriteConfig`, `EngineConfig`, `PrecisionConfig`, `NeighbourConfig`, `PolarConfig`, `IntegratorConfig`), `(item, choice, reason)` decisions, fallbacks, memory, revision chain |
| `Planner` | `plan(traits, atoms, integrator, hardware) → Plan`, `build(plan, model, atoms) → Engine`: the only builder of force fields, integrators and engines |

Capture warm-ups save and restore `integrator.buffers(state)`,
`ff.buffers()` and the engine's own flags. This covers more than the
baseline's `_graph_state`, which misses volume, rcell, `_v_mid`, `_p_mid`
and the pipe flags (§8.1). The union of the three is the seed of the 1b
runtime-state registry.

## BAOAB's cell degrees of freedom

The cell lives in `SimulationState.cell`, the same tensor for both ensembles.
BAOAB updates it in place in `first_half` and declares `variable_cell = True`.
The engine then requires `ff.compute_stress`, passes the cell to
`ff.set_cell` inside stage A, and the force stage fills `state.stress`. The
barostat momentum `p_eps` and cell force `force_eps` are integrator state in
1a (`state_dict`). §10.6 lists them as physical state, so 1b may move them
into `SimulationState` (`FIELDS_ADDED_IN_1B["barostat"]`). BAOAB's noise row
is `3N + {0, 1, 9}` wide, which is why the integrator owns the batch layout.
Speculative graphs stay refused for NPT, as today.

## Open questions

1. `SimulationState` stores velocities, not momenta: ASE stores momenta, the
   brief says momenta, the baseline integrates `v`. `momenta` is derived.
2. Name clash: `fm.ForceField(model, atoms, …)` in §7.1 is a constructor;
   here `ForceField` is the protocol. Proposal: T11's class is
   `MACEForceField`, and the public `fm.ForceField` becomes a factory (phase 3).
3. Where `p_eps` lives (integrator in 1a, state in 1b?).

Questions 2 and 3 are answered in [contracts-1b.md](contracts-1b.md) (1b-T01): the public
`fm.ForceField` is a plan-free class in `model/forcefield.py`, and every plan builds its own
`MACEForceField`; `p_eps` and `force_eps` move into `SimulationState` (1b-T05).
