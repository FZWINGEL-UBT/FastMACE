# Contracts of phase 3: dynamic topology and Monte Carlo

Plan task 3-T01, 22 October 2026. Spec: SPEC §7.2, §7.6, §8.1, §8.5, §9.2, §10.4, §10.6;
decisions 31–36 (SPEC §18); `plans/phase-3.md`. Every later phase 3 task builds against these
names and signatures. `tests/cpu/test_contracts_3.py` pins them, and `tests/cpu/test_config.py`
checks the new fields against `switch-map.md` (sections 2–4 and 11).

Every default keeps today's behaviour, so default plans, forces and trajectories are bitwise
unchanged. `tests/cpu/test_research_surface.py` is unchanged.

## Config fields and records (`config.py`; T01 is its only phase 3 editor)

| Name | Default | Meaning | Filled by |
| --- | --- | --- | --- |
| `NeighbourConfig.skin` | `"auto"` | rule 11 from the workload; an explicit float is kept as given | T05 |
| `NeighbourConfig.headroom` | `"auto"` | `"fixed"`: `capacity_headroom` and `edge_bucket_fraction` as given; `"auto"`: rule 11 sizes them | T05 |
| `EngineConfig.recapture_budget` | `"auto"` | re-captures one run may make before it warns, records and recommends the compiled tier; an int ≥ 0 | T04 (T05's rule) |
| `EngineConfig.graph_variant_cap` | `64` | captured variants kept at once, at most 64 and at most `dynamo_cache_size_limit` | T04 |
| `PrecisionConfig.mc_energy` | `"auto"` | the precision of MC energies, the planner's resolution of `Workload.mc_energy_precision` | T07, T11 |
| `Workload.max_displacement_A` | `None` | the largest displacement δ (Å) any `Displace` may use; required when `"displace"` is in `mc_moves`; rule 11 adds 2δ to the skin, and T10 refuses a larger `delta_A` (`planner/workload.py`) | T05 |
| `MCConfig.max_attempts_per_block` | `10_000` | attempts of one move per block | T09 |
| `MCConfig.max_move_sets` | `8` | move records attached at once | T09 |
| `IntegratorConfig.fixcm_projection` | `"auto"` | how `fixcm` projects the Langevin noise. `"mass_weighted"`: the Gaussian conditioned on a fixed centre of mass (mass-weighted means, no √(N/(N−1)) factor; canonical over 3N − 3 degrees of freedom, per species too; the PyTorch and the fused stage, bitwise alike). `"ase"`: ASE's, bit for bit (noise-stream equivalence, research-engine A/Bs). `"auto"` resolves in every plan with the decision `IntegratorConfig.fixcm_projection`: `"mass_weighted"` under NVT `fixcm` for MD (`fixcm-projection-md`, decision 37, 3-T22c) and MC (`fixcm-projection-mc`), `"ase"` for NPT and `fixcm` off (`fixcm-projection-ase`), an explicit value kept (`fixcm-projection-given`); outside a plan it reads as `"ase"`. `MCDriver.attach` refuses `redraw_all` under `fixcm` unless the projection is `"mass_weighted"` (`mc-redraw-all-ase-projection`). Also a field of `fm.integrators.Langevin` and `fastmace.ase.Langevin` (default `"auto"`). Reported temperatures: `io.trajectory.degrees_of_freedom` (3N − 3 under `"mass_weighted"`, else 3N), `io.trajectory.kinetic_temperature`, `Simulation.degrees_of_freedom`. Resume keeps the projection a run ran with; a record without the field reads as `"ase"` | T22, T22b, T22c (docs/design/fixcm-projection.md) |

Constants and functions:
- `DEFAULT_SKIN_A = 0.3`;
- `MC_ENERGY_PRECISIONS`, equal to `Workload.mc_energy_precision`'s choices;
- `MC_STREAMS = ("proposal", "acceptance", "velocity")`;
- `SWAP_VELOCITIES = ("exchange", "rescale", "redraw", "redraw_all")`;
- `TRANSMUTE_VELOCITIES = ("rescale", "redraw", "redraw_all")`;
- `mc_stream_seeds(seed) -> {stream: int}`: 63-bit seeds from SHA-256 of `"fastmace-mc:{seed}:{stream}"`. They are the same in every process and differ per stream.

`mode="compiled"` is already one of `config.MODES`. 3-T06 lifts the planner's `mode-compiled`
refusal.

**`mode` defaults to `None`** (3-T06, conductor answer 1) in `Simulation`, `fm.md`, `PlannerOptions`,
`planner_options` and `MinimalPlanner.plan`/`plan_traits`; `simulation.json` records `None`. Without a
mode the planner picks the graph tier, or the compiled tier where rule 4 applies: `padding_is_exact` is
false (decision code `mode-compiled-rule-4`).

**Rule 9 as amended by decision 38** (3-T06b, 22 October 2026): host callbacks, `fastmace.ase` observers
included, keep the graph tier at every interval, with a host sync at each callback. The decision
`workload.host_callbacks_every` is `"graph, host sync every n steps"` (`host-callbacks-graph-sync`, rule
9), and its reason (`minimal.RULE_9_REASON`) names decision 38 and 3-T06's measurement (compiled
1.3-4.7x slower than graph with a callback every 1-10 steps). 3-T06's `RULE_9_HOST_CALLBACKS` and the
code `mode-compiled-rule-9` are gone.

An explicit mode is the user's (code `mode-compiled-tier` for `"compiled"`). An explicit `"graph"` is
refused under rule 4 (`inexact-padding`, the reason names `mode="compiled"`). The compiled tier's decisions:
`capture` off ("compiled tier: no CUDA graphs" and why), `EngineConfig.speculative_graph` off,
`EngineConfig.dynamic_edges` on when `compile_mode` is not `"none"`, `padding` "exact edge counts"
(`compiled-exact-edges`) for an inexact model, and, with host callbacks, `workload.host_callbacks_every`
`host-callbacks-compiled` (the compiled tier was chosen by rule 4 or `mode="compiled"`). `planner.fixed.COMPILED_TIER = "compiled"`; `FixedPlanner.build` builds it
with `capture=False`.

Every plan resolves the four new `"auto"` fields, each with a decision `auto (<rule>)`
(`FixedPlanner._phase_3_fields`, `fixed.AUTO_RULES`). Until the tasks named below give the
rules, they resolve to today's behaviour:

| Field | Resolves to now | Rule from |
| --- | --- | --- |
| `NeighbourConfig.skin` | `DEFAULT_SKIN_A` (0.3 Å) | T05 |
| `NeighbourConfig.headroom` | `"fixed"` | T05 |
| `EngineConfig.recapture_budget` | `graph_variant_cap` (64): the T04 rule (below); the engine reads it | T04 |
| `PrecisionConfig.mc_energy` | the force precision (`dtype`); nothing reads it yet | T11, for MC plans only |

`SkinNeighbourList` reads a `skin="auto"` that reaches it without a plan as `DEFAULT_SKIN_A`.

## Public names

`fm.mc` is the module `fastmace.mc`, resolved lazily. Its records import neither torch, MACE nor
ASE (tested in a fresh interpreter).

| Name | Signature | State |
| --- | --- | --- |
| `fm.mc.Swap` | `Swap(pairs, every, attempts, velocities="exchange")`; `.move == "swap"`; `pairs` stored as a tuple of 2-tuples | record complete |
| `fm.mc.Transmute` | `Transmute(species, every, attempts, velocities=None, mu=None, formal_charges=None)`; `.move == "transmute"`; `mu` and `formal_charges` stored as sorted `(species, value)` pairs; `velocities=None` is the driver's default (`rescale`, or `redraw_all` under `fixcm`) | record complete |
| `fm.mc.Displace` | `Displace(delta_A, every, attempts, adaptive=False)`; `.move == "displace"` | record complete |
| `fm.mc.MOVE_TYPES` | `("swap", "transmute", "displace")`, equal to `planner.workload.MC_MOVES` | complete |
| `fm.mc.configurational_mu` | `configurational_mu(mu_full, temperature_K, reference, masses=None) -> dict` | complete |
| `fm.mc.MCDriver` | `MCDriver(moves, *, seed, config=None)`; `attach(engine)`, `run_block(engine, step)`, `state_dict()`, `load_state_dict(record)` | stub, T09 |
| `fm.mc.moves.build_attempt` | `build_attempt(move, engine, evaluator, streams)` | stub, T09 (T10 for `Displace`) |
| `Simulation.update` | `update(*, species=None, masses=None, charges=None, total_charge=None, friction_per_fs=None, positions=None, momenta=None)`: any combination as one transaction with one force evaluation; nothing given, nothing happens | complete, T03 (conductor answer) |
| `Simulation.set_species` | `set_species(species)`: atomic numbers or symbols, `(N,)`; `update(species=)` | complete, T03; needs T02's `check_species`, refused without it |
| `Simulation.set_masses` | `set_masses(masses)`: amu, `(N,)`; the momenta stay (ASE): v' = v·m_old/m_new | complete, T03 |
| `Simulation.set_charges` | `set_charges(charges, *, total_charge=None)`: per-atom charges and/or the total | complete, T03; refused for models that take none, and until T02 registers a buffer that reads them |
| `Simulation.set_friction` | `set_friction(friction_per_fs)`: 1/fs, as the public integrator records | complete, T03 |
| `engine.subprocess.HOST_WRITE_KINDS` | `("positions", "momenta", "species", "masses", "charges", "total_charge", "friction")`: through the subprocess runner, `update` queues one host write per field in its transaction's order; the child applies each by `set_<kind>` (`total_charge` by `set_charges(None, total_charge=)`), and its hand-off restart holds the result (T08) | complete, T08 and T03 |
| `Committed.species`, `.masses`, `.charges` | atomic numbers `(N,)`, amu `(N,)`, charges `(N,)` or `None`; read-only arrays, in the subprocess runner's `committed()` too | complete, T03 |

Notes on the records:
- **Species stay as given:** symbols or atomic numbers ≥ 1. The driver maps them to atomic numbers.
- **`Simulation.run(mc=…)` runs them (3-T15):** it takes `fm.mc` records only (`TypeError` otherwise; an empty set is a `ValueError`), checks the move set that will be attached (given, or kept with `mc=...`) against the workload over every step of the call (`mc-move-not-declared`, `mc-below-min-temperature`, `species-not-in-superset`), then attaches it through `MCDriver(moves, seed=sim.seed).attach(engine)`.
  - A block runs right after the MD step it is due at, before that step's output and restart, with `Simulation._mc_block_active` set.
  - Detaching (`mc=None`) keeps the driver's `state_dict()`; a later attach loads it into the new driver with the new moves (`simulation._carried_mc_record`, then T09's `load_state_dict(record, other_moves=True)`: streams, cycle position and cumulative counters carry over). A restart holds the attached set (`mc`) or, detached, `simulation.json`'s `mc` entry.
  - Move records as data: T09's `mc.driver.move_record` and `move_from_record`, through `simulation.move_record` and `simulation.move_from_record`, which also names an unknown field and takes `mu` and `formal_charges` as TOML tables (restart `mc.moves`, CLI `[[mc]]` tables, the child runner).
  - Thermo: `mc_<move>_attempted` and `mc_<move>_accepted` per `Workload.mc_moves` entry (`simulation.mc_columns`), as of the last block; none without MC moves. `counters.mc` and the report's `mc:` line and `report.mc` metadata block (moves, ranges, blocks, device time). The cost per attempt is the device time over the attempts of the timed blocks: each attach's first block, whose time includes the attempts' captures, is left out (`timed_attempts`).
  - `fm.md(mc=…)` adds the move set's species to the superset and, with a schedule, attaches it only over the steps at or above `min_mc_temperature_K` (`quick.mc_ranges`).
- **`configurational_mu` formula:** Δμ_s^conf = Δμ_s^full + (3/2) k_B T ln(m_s / m_ref), with k_B = `ase.units.kB` (8.617330337 × 10⁻⁵ eV/K in ASE 3.27), the constant the thermostats use, as in the MC acceptance. The reference's own Δμ must be 0. `masses=None` takes ASE's standard atomic masses.

## Errors (`errors.py`, new classes only)

- `ChargeParityRefused(PlanRefused)`: a transmutable set with an odd ΔZ − Δq (SPEC §10.4 "Charge and spin").
  - T09 raises it with `ChargeParityRefused.from_reasons(...)`, with a reason coded `CHARGE_PARITY_CODE = "transmute-charge-parity"`.
  - Other reasons of the same refusal come with it.
  - If rule 2 is among them, `from_reasons` raises `SpeciesNotInSuperset` instead, as for any `PlanRefused`.
- Capacity overflows stay internal (SPEC §7.6). T04 defines its internal signal in `engine/`, not in `errors.py`.

## Events (`fastmace/events.py`; owner T03)

- **Event kinds** (`EVENT_KINDS`): `species`, `masses`, `charges`, `friction`, `barostat`, `swap`, `transmute`, `displace`.
- **`Event(kind, atoms=None, values=None)`**: a frozen record.
  - `atoms` holds distinct indices; `None` means every atom, or none (a friction change).
  - `values` holds what the kind needs.
- **Lifecycle** (`LIFECYCLE`), each step from SPEC §8.5:
  1. `quiesce`: the pipeline, asynchronous output and speculative steps;
  2. `snapshot`: the affected state, the derived buffers it feeds and the random-number cursors (displacements add positions and the neighbour list);
  3. `apply`;
  4. `refresh`: every derived buffer that depends on the change;
  5. `evaluate`: the hook, with the overflow check before any decision;
  6. `decide`: accept, or restore exactly; restored forces are restored, never recomputed;
  7. `resume`, in `RESUME_ORDER`: species, masses, charges and momenta; then forces and stress; then the integrator caches (BAOAB `force_eps`).
- **Outcomes** (`OUTCOMES`): `accepted`, `rejected`, `impossible` and `redo`.
  - `redo` is an overflow. It is an execution event, not a chain outcome: the snapshot is restored, the random-number cursors included, and the attempt runs again with the same numbers.
- **Not events:**
  - schedule-driven targets, which are device-scalar fills;
  - nothing changes a shape or an address inside a captured graph.
  - A friction change drops the graphs that bake `c1`/`c2` in, and that is a recorded re-capture.
- **`DerivedStateRegistry.register(name, buffer, refresh, depends_on)` → `DerivedBuffer`**:
  - `depends_on` names some of `DEPENDENCIES`: species, masses, charges, friction, barostat, positions, cell, momenta, forces, stress;
  - one entry per name;
  - `names()`, `affected(changed)`, `in` and `len`;
  - `refresh(changed)` runs the affected entries in registration order and returns their names. An owner registers a buffer after the buffers it reads;
  - `check_against(fresh)`: the names whose buffers differ bit for bit from a fresh registry (or engine), or that only one side has.
- **Where the registry lives (T03):** `GraphEngine.derived`, built after the state, force field and integrator. The engine calls `register_derived(registry, state)` on the force field (if it has one; T02's `MACEForceField.register_derived`), then on the integrator, then on the fused integrator. Every derived tensor is also in the runtime-state registry, under its owner's name or as `derived.<name>` (policy `derived.*`: restart "derived", captured).
- **Refresh functions** take no arguments and read the committed inputs from the state: `state.species` (int64 atomic numbers), `state.masses` `(N, 1)`, `state.charges`, `state.total_charge`. They write in place at fixed addresses. Those that depend on species, masses or charges are capturable (no host sync, no new buffer; temporaries inside a graph are fine), tested in `tests/gpu/test_events_gpu.py`. A friction refresh may copy host floats: a friction event drops the graphs anyway.
- **Registered by T03:** Langevin `integrator.c3_unit`, `c4_unit`, `c5_unit` (masses, friction); BAOAB `integrator.inv_sqrt_masses` (masses) and `integrator.force_eps` (masses, momenta, forces, stress, cell); the fused integrator's `fused.coefficients` (friction) and `fused.total_mass` (masses; 3-T22b's mass-weighted projection).
- **`Transaction(engine, events, *, evaluate=None, registry=None)`**: `run()`, `quiesce()`, `snapshot()`, `apply()`, `refresh()`, `accept()`, `reject()` and `resume()`; `host_writes` (callables `write(state)` run after the events: `Simulation.update`'s positions and momenta); `changed` and `outcome`.
  - It validates every event when it is made, before anything is written. The host kinds are species, masses, charges and friction; `swap` and `transmute` raise `NotImplementedError` naming 3-T09, `displace` 3-T10, `barostat` (no host API in phase 3).
  - Values: `masses` amu and `species` atomic numbers or symbols, one per atom of `atoms`; `charges` an array or `{"charges", "total_charge"}`; `friction` in inverse ASE time units.
  - A total charge goes through the force field's host `set_total_charge(value)` (T02: its device total is base + Σ charges − Σ charges at the base, so T09 changes it in a graph through `state.charges`), then the `charges` refresh; `state.total_charge` keeps the host record. A rejection sets the old base back before the buffers are restored. A model without `set_total_charge` takes no total charge.
  - A mass change keeps the momenta, only where the mass changes. A species change sets the standard masses the same way; a masses event after it wins.
  - Accepted: `engine.transactions_committed` counts it; the resume is `canonicalise(reevaluate=True)`, then a refresh of what depends on forces and stress.
  - Rejected, impossible, redo: the runtime-state registry snapshot is restored in place, with the noise generator and cursor and the integrator's host constants.
  - A friction event calls T04's `GraphEngine.invalidate_graphs(reason)`, again on a rejection that captured meanwhile. With the default barostat τ it also changes `barostat` (`P_tau`, the barostat mass); an explicit `barostat_tau_fs` stays.
- **`events.check_against(engine, fresh)`**: the CI registry check of SPEC §12.2. It compares the derived registries, the state's composition, the neighbour list as the set of compacted edges, the integrator's host constants (the floats a stage bakes in, its config) and every other tensor reachable from the force field and integrator that the runtime-state registry does not hold. So a buffer left out of the registry fails it (`tests/cpu/test_events.py` has the negative controls).
- **`events.species_numbers(values, count)`**: atomic numbers from numbers or symbols, validated.

## Species superset in the force field (3-T02)

Added by 3-T02 under the conductor's answers (`plans/phase-3.md`, "3-T02 species superset").

| Name | Signature | Meaning |
| --- | --- | --- |
| `MACEForceField(..., species=None)` | keyword | the superset, atomic numbers; default the structure's species. The planner passes `Plan.superset`. A superset species without an embedding raises `ModelElementError`, a structure species outside it `SpeciesNotInSuperset` |
| `MACEForceField.superset`, `.species_columns` | tuples | the superset (by model column) and its `node_attrs` columns; the compile key of species-dependent rewrites |
| `MACEForceField.atomic_numbers`, `.charges` | device tensors, (N,) int64 and float64 | what a transaction writes (3-T03); every derived buffer below is refreshed from them |
| `MACEForceField.check_species(numbers)` | host | raises `SpeciesNotInSuperset` (rule 2, code `species-not-in-superset`) before anything is written; a wrong length raises `ValueError` |
| `MACEForceField.register_derived(registry, state=None, prefix="force_field")` | → list of `DerivedBuffer`; with `state` (the engine's, 3-T03) the refreshes read `state.species` and `state.charges` (`use_sources`) | registers `node_attrs`, `node_e0`, `sender_groups` (species), `total_charge`, `total_spin` and `species_feats.interactions.<i>` (charges), in that order. Each refresh runs on the device, in place, at fixed addresses, without host synchronisation |
| `MACEForceField.use_sources(species, charges=None)` | device tensors | alias the species and per-atom charge sources; their values must be the current ones |
| `MACEForceField.set_total_charge(total_charge)` | host, between replays | a host total-charge event: the next `total_charge` refresh gives exactly this value; transmutations change only the per-atom charges, and the device refresh follows them |
| `MACEForceField.derived_buffers()` | → `{name: tensor}` | the same buffers without the prefix |
| `fastmace.rewrites.registry.structure_inputs(batch, species_columns=None)` | → `Context.inputs` | `node_attrs`, the per-graph `joint_features` (`JOINT_FEATURES = ("total_charge", "total_spin")`) and the superset columns |

- **Identity basis.** The reassociated first interaction reads `FirstInteractionReassociation.species_feats`, an `[S, C]` table: `linear_up(node_embedding(I_S) + joint_embedding)` (`rewrites.mh1.modules.identity_species_feats`). It is evaluated in a batch of the structure's row count, so a present species' row is bitwise the row its atoms have. The force field binds its own table before each call, as it binds the streaming count.
- **Sender groups.** `SkinNeighbourList(..., sender_groups=<tensor>, group_count=S)` reads the tensor at every build. The count is fixed at the superset size.
- **Rule 8.** A superset beyond the structure plans graph mode. The `superset` decision is coded `superset`, rule 8.

## Engines

- `GraphEngine.derived` (T03): the derived-state registry above. `GraphEngine.quiesce()`: an event's quiesce (`flush`).
- `GraphEngine.transactions_committed` (int) and `GraphEngine.state_version == (state.step, transactions_committed)` (T03): T07 tags its energy cache with it. Not restart state.
- `engine/compiled.py`: `MODE = "compiled"`; `CompiledEngine(config, force_field, integrator, state, neighbours=None, constraints=None, dynamic_edges=True)`, a `GraphEngine` subclass that never captures; `dynamic_edges` steps on exact edge counts (`False`: bucket padding). Its `info()` adds `tier`, `dynamic_edges` and `compile_mode`; restart files keep the engine name `graph`. Only the planner builds it (T06).
- The fallback's mode dimension (`planner/fallback.py`, T06): when eager passes the reduced gate, a graph plan tries compiled first, then eager (§9.4 step 2). The terminal sequence is unchanged.
- `engine/graph.py`:
  - `ENERGY_KEY = "E"` and `ENGINE_KEYS = ("A", "AC", "ACB", "ACP", "ACPB", "B", "C", "E")`;
  - `GraphEngine.energy(*, precision=None)`: the energy-only graph. It returns a fixed-address float64 `[1]` buffer that the next call overwrites. Its key is `(ENERGY_KEY, precision, edges)`: one variant per edge size, captured at the first call. It reads the current positions and active edges, and sets `EDGE_OVERFLOW` when the count passes them. Without capture it runs directly. Complete (T07);
  - `GraphEngine.add_energy_field(field)`: the energy-only graph's force field in another precision, the float64 MC copy that follows the engine's force field. Its derived buffers join `GraphEngine.derived` as `energy.<precision>.*`. One per precision. `GraphEngine.energy_fields`: `{precision: force field}`. Complete (T07);
  - `GraphEngine.energy_stage(*, precision=None) -> (stage, out)`: the same evaluation as a plain function, for T09's attempt graph, which captures it with its own key, one per edge size. Complete (T07);
  - `GraphEngine.captured(key, fn) -> replay`: captures `fn`, which takes no arguments and has fixed shapes and addresses, with the engine's warm-up, registry restore and re-capture policy (variant cap, budget, log). It refuses `ENGINE_KEYS`, which are the engine's own, and tuple keys whose first element is one. The graph is captured at the first replay and again after an invalidation or eviction; without capture the replay calls `fn`. A tuple key's first element is its family, so `("mc.swap", bucket)` keys are variants of one key. Complete (T04), for T09's attempt graph.
  - `GraphEngine.invalidate_graphs(reason, keys=None)`: drops the captured graphs (all of them, which also retires the pool, or those of `keys`) and logs the reason. The next capture of a dropped key counts as a re-capture. T03 calls it for friction events. Complete (T04).
  - `GraphEngine.uncommitted() -> bool`: true while a pipelined or speculative step is unvalidated, or an overflowed evaluation is being redone. T08's restart write guard reads it. Complete (T04).

### Energy-only evaluation and the MC evaluator (T07)

- `MACEForceField.energy_only(positions)`: the float64 energy `[1]` from a forward pass under `torch.no_grad()`, with forces, stress and `training` off. It is the force path's float64 sum and has no `grad_fn`. A compiled model compiles it as its own variant, because grad mode and `training` are guarded. That variant is accepted (conductor answer 2).
- `MACEForceField.follow(leader)`: a copy in another precision reads the leader's compacted edge index and its unit shifts. It rebuilds the shifts in its own precision from the float64 cell, and shares the leader's species, charge and total-charge sources. `use_edges(n)` then takes the leader's padded count. `info()["follows"]` names the leader's precision.
- `fastmace/mc/evaluator.py` (owner T07):
  - `PRECISIONS = ("float32", "float64")`;
  - `float64_model(force_field, plan, backend=None)`: the plan's checkpoint in float64 (`ForceField.execution_model`), in the plan's backend or the one given;
  - `float64_copy(force_field, model, *, neighbours=None, polar=None, check=True)`: an eager float64 `MACEForceField` that follows a float32 one. It takes that force field's rewrites, superset, charge, spin and E0 table;
  - `EnergyCache`: one float64 energy, tagged `(evaluator, precision, state version)`: `get(tag)`, `store(tag, energy)`, `clear()`;
  - `MCEvaluator(engine, precision=None, *, model=None, copy=None, polar=None)`. `precision=None` means the force precision, and nothing below it is offered. `"auto"` is refused, because the planner resolves it (T11).
    - `evaluate()` goes through `GraphEngine.energy` on a graph engine, otherwise through the force field directly.
    - `stage()` returns `GraphEngine.energy_stage` for a caller's captured graph.
    - `baseline()` is always a fresh evaluation, cached under `tag()`, and never the MD energy.
    - `accept(energy)` stores the trial's energy as the new baseline; `reject()` keeps the old one.
    - `current()` returns the cached baseline, or None once the state version has moved.
    - `info()` gives the precision, force precision, backend, copy, compile mode, captured, memory bytes and evaluations.
  - `tag()` uses `GraphEngine.state_version`. An engine without one uses `("step", state.step)`.
- `tools/bench/mc_cost.py`: energy-only cost as a fraction of one MD step. T09, T10 and T20 add their rows as further sides.

### MC streams and the driver (T09)

**Streams** (`fastmace/mc/streams.py`; conductor answer 2 of 3-T09). The layout below is a restart
compatibility promise: a restart record's (seed, offset) must give the same numbers in a later version.
- Counter-based Philox4x32-10 (Random123's `philox4x32_10`). Number `n` of a stream with seed `s`
  (`config.mc_stream_seeds(seed)[stream]`) is `Philox(counter=(n mod 2³², n >> 32, 0, 0), key=(s mod 2³², s >> 32))
  = (x0, x1, x2, x3)`.
  - A **uniform** in [0, 1): `((x0 >> 5)·2²⁶ + (x1 >> 6)) / 2⁵³`.
  - A **normal**: `sqrt(−2 ln(1 − u))·cos(2π v)`, with `u` from (x0, x1) and `v` from (x2, x3) as above.
  - The same numbers on CPU and GPU (uniforms bit for bit). Plain int64 PyTorch, capturable.
- A stream's state is its seed and its **offset**, the position of its next number (`io.restart.MC_STREAM_KEYS`).
- **Fixed counts per attempt** (`streams.attempt_counts(move, velocities, N)`), whatever the outcome:

  | Stream | Swap | Transmute |
  | --- | --- | --- |
  | proposal | 2N + 1 | N + 1 |
  | acceptance | 1 | 1 |
  | velocity | 6 with `redraw`, else 0 | 3 with `redraw`, else 0 |

  `redraw_all` draws 3N velocity numbers once per block per move, after the move's attempts
  (`streams.block_velocity_count`), used only after an acceptance in them.
- **Attempt `k` of a move** in a block reads, per stream, the numbers `base + k·count … base + (k+1)·count − 1`.
  `base` is the stream's offset when the move's attempts began. `k` is a device counter (the attempt index)
  that advances only when an attempt completes; the host reads it with the block counters and advances each
  offset by `k·count` (plus 3N for `redraw_all`). Moves due at the same step run in attached order.
- **Decoding** (`streams.decode_swap`, `streams.decode_transmute`):
  - swap: proposal number 0 picks the pair type `p = min(⌊u·P⌋, P − 1)` of `Swap.pairs` (in record order);
    numbers 1…N are the keys of the first species' atoms (atom order), N+1…2N those of the second's.
  - transmute: numbers 1…N are the keys of atoms whose species is in S; number 0 picks the new species,
    `r = min(⌊u·(|S| − 1)⌋, |S| − 2)`, then `S[r + (r ≥ c)]` with `c` the current species' index in S
    (`Transmute.species` order).
  - Selection: keys of other atoms are +∞, `argmin` (the first minimum) picks the atom, and a minimum of +∞
    marks the attempt impossible. An impossible attempt changes nothing and consumes its counts.
  - Acceptance: with `u` the acceptance number, accept iff `ln u < −(ΔE − Δμ)/kT`, where ΔE = E_trial −
    E_baseline (float64, the MC evaluator's), Δμ = μ[new] − μ[old] for a transmutation (0 for swaps; absent
    species have μ = 0) and kT = `ase.units.kB`·T (the thermostat's constant) with T the integrator's target at the block. `u = 0` accepts.
  - Velocities (`fastmace/mc/momenta.py`, named functions): `redraw` writes `sqrt(kT/m_new)·z` with the
    velocity numbers of the attempt, atom i's x, y, z then atom j's; `redraw_all` writes `sqrt(kT/m)·z` for
    atom a, component c from number `base + 3a + c`, then removes the centre-of-mass velocity.

**Driver** (`fastmace/mc/driver.py`). T15 wires it into `Simulation.run(mc=…)`.
- `MCDriver(moves, *, seed, config=None)`; `attach(engine, *, evaluator=None)`: the checks below, the device
  tables, counters and captured attempts on a built engine. `evaluator=None` is
  `MCEvaluator(engine)` (the force precision); pass a float64 `MCEvaluator` for float64 MC energies.
- `due(step)`: the moves whose `every` divides `step` (step > 0). `run_block(engine, step)`: quiesce, the
  baseline (a fresh MC-evaluator energy), every due move's attempts as captured replays (key
  `("mc.<move>", driver id, index, precision, edges)`), one host read of counters, attempt indices and the overflow
  word, then, after an acceptance, `redraw_all` where chosen, the mass and momentum refreshes and
  `canonicalise(reevaluate=True)` (forces once, `integrator.start`). Returns the block record, also in
  `last_block`: `step`, `moves` (per due move: `move`, `attempts`, `attempted`, `impossible`, `accepted`),
  `baseline_eV`, `energy_eV` (the baseline after the last acceptance), `accepted`, `device_ms` (CUDA events;
  `None` on the CPU). `None` when no move is due.
- `counters` (cumulative, per move type, `MC_COUNTER_KEYS`) and `offsets` (per stream): host values,
  updated once per block. `state_dict()` is the restart's `mc` record (moves as `{"move", **fields}`, tuples as lists;
  `driver.move_from_record` reads an entry back) (`MC_RECORD_KEYS`; `cycle_position`
  is the step of the last block, `delta_A` empty for swaps and transmutations); `load_state_dict(record, *, other_moves=False)`
  restores it; `other_moves=True` lets another move set continue a record's streams, counters and cycle
  position (a re-attach). `attach` and `run_block` also put it into `engine.state.mc`.
- **Checks** (`planner.minimal.check_mc_moves(moves, *, superset, charge_inputs, fixcm)`, called by
  `attach`; raises `PlanRefused` with every reason):
  - rule 2: a species outside the superset (`species-not-in-superset`, `SpeciesNotInSuperset`);
  - `fixcm`: `rescale` and `redraw` refused (`mc-velocities-fixcm`); `exchange` is for swaps only (records);
  - parity, only for models or heads with charge or spin inputs: every pair of S needs ΔZ − Δq even
    (`transmute-charge-parity`, `ChargeParityRefused`); otherwise the decision is "parity: not applicable".
- Transmuted atoms take ASE's standard masses. A swap exchanges species, masses and per-atom charges. A
  transmutation with formal charges moves the force field's total charge by Δq inside the graph; after a block
  with an acceptance the host records the new total in `state.total_charge`.

### Overflow before commit (T04)

- **The device word** `GraphEngine.overflow` (int64 scalar, registered as `engine.overflow`, sticky, reset at a canonical point) holds the bits `EDGE_OVERFLOW = 1` and `KVECTOR_OVERFLOW = 2`.
  - `flag_edge_overflow(count, edges)`: device operations only, so an MC evaluation sets it inside its captured graph (T07, T09, T10).
  - `overflowed() -> int`: the word, plus `KVECTOR_OVERFLOW` when the PolarMACE flag is set. One host sync.
  - `clear_overflow()`: clears both.
- **MD steps:** a step is committed only after its flags are checked.
  - Edges: the count against the padded edges the force stage read. Pipelined and speculative steps can overflow; a checked step reads the count first, so it cannot.
  - K-vectors: with PolarMACE and a variable cell, the flag is read after every force stage. That is one host sync per step, only in those plans, recorded as the plan decision `overflow check: per step` (conductor answer 1). `info()["overflow_check"]` is `"per step"` there and `"edges"` otherwise.
  - An overflowed step restores stage A's mid-step tensors, grows what overflowed (a larger padded size, or the k-vector set rebuilt for the step's cell), and redoes the force stage with the same noise (answer 2).
  - A pipelined step adds to `state.step` at its launch and is committed at its validation. `snapshot`, `sync`, `canonicalise` and restart writes validate first.
- **MC attempts:** `redo_on_overflow(attempt, *, keep=(), max_redos=3) -> redos`.
  1. It quiesces and snapshots the registered state and the tensors of `keep` (the caller's random-number cursors, such as an MC stream's device offset).
  2. It runs `attempt` and reads `overflowed()`.
  3. On an overflow it restores both, grows (the neighbour list rebuilt at the restored positions, or the k-vector set at the current cell), and runs `attempt` again with the same numbers.
  4. After `max_redos` it raises `CapacityOverflow` (internal, in `engine/graph.py`, with `.word`).
  The engine draws no random numbers itself.
- **Counters:** `info()["overflows_redone"]` and `info()["overflow_log"]` (step, kinds, mode). A pipelined rebuild redo (a stale list) is not an overflow, and a speculative bucket rollback is. `provenance` takes `counters.overflows_redone` from the engine info when the caller gives none.

### Re-capture policy (T04)

- **Every capture is logged** in `info()["recapture_log"]` (step, key, reason, counted) and as a `logging` line (`fastmace.engine.graph`).
- **What counts** against `EngineConfig.recapture_budget` (answer 3): a capture that replaces or adds a variant of a key captured before. That covers a new edge size, a capture after `invalidate_graphs`, an eviction, or a pool reset (`POOL_RESET`). The first capture of each key family is logged but not counted.
- **The variant cap** (answer 4): at most `EngineConfig.graph_variant_cap` graphs are held. Before a capture that would pass it, the least recently replayed graph is evicted. That is recorded in `graph_evictions`, and its later capture counts.
- **Past the budget:**
  - the engine warns once (`RecaptureBudgetWarning`);
  - it records the step (`recapture_budget_exceeded_at`);
  - `recapture_warning` gives the report's line, which recommends the compiled tier. `Simulation.run` adds it to the report's warnings.
  - The run never downgrades mid-run (phase 5, §9.6).
- **The budget's `"auto"` rule** (`FixedPlanner._phase_3_fields`, `recapture_budget_from_workload`): the variant cap. A run that has re-captured as many graphs as it may hold at once has cycled through its whole working set, so the compiled tier is the better plan. An engine built without a plan reads `"auto"` the same way.
- `provenance` takes `counters.recaptures` from the engine info when the caller gives none. The full record stays in `counters.engine`.

## The move-energy gate and the MC energy precision (3-T11)

Added by 3-T11 under the conductor's answers (`plans/phase-3.md`, "3-T11 move-energy gate"). SPEC §12.3
"move-energy differences", §10.4 "Energy precision", §9.4 step 2, decision 35.

| Name | Signature | Meaning |
| --- | --- | --- |
| `selfcheck.sample_moves` | `sample_moves(atoms, superset, workload, seed, samples=MOVE_SAMPLES) -> MoveSet` | the sample moves (below), drawn from the seed |
| `selfcheck.move_references` | `move_references(plan, force_field, atoms, workload, *, seed, check="skip", gate=None, samples=MOVE_SAMPLES) -> MoveReferences \| None` | the float64 reference energy of every sample configuration, before any engine, on the gate's ladder (plan device, then the CPU; no reduced cell); `None` without `mc_moves`; `unavailable` is recorded, never raised |
| `selfcheck.move_gate` | `move_gate(engine, references, *, plan, force_field, record=None, requested=None) -> GateRecord` | ΔE through the engine's `MCEvaluator`; returns `record` (default `skipped`) with `move_energy` and `mc_energy_precision`; raises `GateFailed` |
| `selfcheck.move_rows`, `move_verdict`, `move_threshold` | | the rows, the rule and `0.01 k_B T_min` (eV) |
| `GateRecord.move_energy`, `.mc_energy_precision` | fields | the move-energy part: `status` (`passed`, `failed`, `unavailable`), `requested`, `precision`, `threshold_eV`, `min_mc_temperature_K`, `fraction`, `superset`, `samples`, `not_sampleable`, `moves` (per type: samples, worst eV and in k_B T_min), `tried` (each precision), `rows`, `references`, `seconds`, `reason`. `GateRecord.status` stays the energy, force and stress gate's (`skipped` under `check="skip"`) |
| `fallback.DIMENSIONS` | `("mc_energy", "mode", "rewrites", "backend")` | the MC energy precision first |
| `fallback.candidate(plan, {"mc_energy": "float64", ...})` | | the revision with float64 MC energies; nothing re-planned |
| `fallback.mc_energy_revision(plan, record)` | → plan | the gate's chosen precision as an `mc_energy` revision with its `TriedCandidate` |
| `fallback.move_energy_only(failure)` | → bool | only the move-energy part failed |
| `fallback.activate(..., moves=None)`, `Fallback(..., moves=None)` | keyword | every candidate and diagnosis run also passes the move-energy gate |

- **Sample moves.** The move records arrive only with `run(mc=…)`, so the gate samples what any record
  could declare: `MOVE_SAMPLES["swap"]` (2) swaps per unordered pair of superset species;
  `MOVE_SAMPLES["transmute"]` (1) atoms of each present species, each to every other superset species;
  `MOVE_SAMPLES["displace"]` (4) displacements of exactly `Workload.max_displacement_A`. A pair with an
  absent species is sampled on a base where another atom was transmuted into it, if transmutations are
  declared, and is "not sampleable" otherwise.
- **Precision.** `PrecisionConfig.mc_energy` for an MC plan: "auto" is the force precision pending the
  gate (decision code `mc-energy-gate-pending`); an explicit request is kept (`mc-energy-requested`),
  and float32 below float64 forces is refused (`mc-energy-below-force-precision`). `move_gate` under
  "auto" tries the force precision, then float64; its result goes into the plan through
  `mc_energy_revision`. An explicit precision that fails raises `GateFailed`.
- **When.** MC plans always run it, also under `check="skip"` (decision 35). `Simulation` calls it
  after T15 (second editor of `simulation.py`).

### Displacements (T10)

`fastmace/mc/displace.py` (owner T10, second editor of `fastmace/mc/` after T09):
- **Numbers per attempt** (a restart compatibility promise like the swap's): `PROPOSAL_COUNT = 5` proposal numbers
  from `base + 5k` (number 0 as a uniform: the atom `min(⌊u·N⌋, N − 1)`; numbers 1–3 as normals: the direction;
  number 4 as a uniform: the radius `δ·u^(1/3)`), one acceptance number at `base + k`, no velocity number
  (`attempt_counts(N)`). Acceptance as for swaps: `ln u < −ΔE/kT`, `u = 0` accepts. `decode_displace(u, g, N, delta)`.
- **`check_displace(move, *, max_displacement_A, variable_cell)`**: `PlanRefused` with every reason, codes
  `displace-variable-cell` (answer 3) and `displace-beyond-max-displacement` (`delta_A` above
  `Workload.max_displacement_A`, or none declared).
- **`DisplaceAttempt(move, engine, evaluator, block_streams)`**, built by `moves.build_attempt` for a `Displace`: the
  shape of T09's `Attempt` (`bound(stage, trial)`, `run()`, `tensors()`, `counters` `[3]`, `accepted`, `counts`,
  `block_velocity_count = 0`, `velocities = None`; `accept` binds `moves.metropolis` when built), plus `halted`,
  `trial_energy`, `delta` (device), `poll() -> (index, halted)`, `host_attempt() -> rebuilt`, `set_delta(delta_A, *,
  record=True)`, `adapt(accepted, attempted)`, `delta_A`, `delta_history`, `trial_rebuilds` (host, cumulative) and
  `max_displacement_A`.
- **Paths:** the captured attempt (key `("mc.displace", driver id, index, precision, edges)`, T09's scheme) is valid
  only when the rebuild test passes and the count fits the active edges; otherwise it restores itself, leaves the
  attempt index and sets `halted`, the later replays do nothing, and the host runs that attempt again with the same
  numbers: the committed list saved, a trial build around the trial geometry when needed, edges sized for the trial,
  the captured energy graph, then accept (keep the trial list) or reject (restore the position and the list).
- **In `MCDriver`:**
  - `attach(engine, *, evaluator=None, max_displacement_A=None)`: `check_displace` for every `Displace`, with the
    workload's maximum (T15 passes `Workload.max_displacement_A`) and the engine's `variable_cell`.
  - `run_block`: after a displacement move's replays, `_finish_displacements(index)` reads its attempt index and
    `halted` (one host read), runs a halted attempt on the host and replays the rest; one more read per halted
    attempt (answer 1). Moves due in the block still run in attached order.
  - The block record's displacement entry adds `trial_rebuilds` (answer 4; a run counter, not restart state, since
    the restart's counters are exactly `MC_COUNTER_KEYS`), `delta_A` (the δ the block used) and, with `adaptive`,
    `adapted_delta_A`.
  - Adaptive δ (answer 2) at the end of each block: δ ← δ·clamp(acc/0.3, 0.5, 2), at most `delta_A` and
    `Workload.max_displacement_A`. The restart's `delta_A` holds each displacement move's current δ by
    `"displace:<index>"`; `load_state_dict` restores it before or after attach. `info()["displacements"]` gives
    δ, its history and the trial builds per move.

Beside it:
- `SkinNeighbourList.save_list(into=None) -> dict` and `restore_list(saved)`: copies of `buffers()` and the candidate
  count, kept by the caller (not reachable from the engine, so `check_against` is unaffected); `restore_list` refuses
  once the capacity moved. Fixed cell only.
- `GraphEngine.trial_rebuild() -> grew` (in place at the trial positions; a growth drops the graphs and is redone
  as an overflow), `use_trial_edges() -> edges` (compact, one host sync, the padded count), `adopt_trial_list()`
  (the pipeline's padded sizes follow an accepted trial list), `record_mc_overflow(word)` (counts in
  `overflows_redone`, clears the flags); `GraphEngine.trial_rebuilds` counts the trial builds.

## Restart version 3 (`io/restart.py`; owner T08)

Since 3-T08 the writer is at `FORMAT_VERSION = NEXT_FORMAT_VERSION = 3`. Version 0, 1 and 2 files keep loading.
- **`COMPOSITION_KEYS`**: `species`, `masses`, `charges` and `physical` (total charge, spin).
  - These are already written. In version 3 they are the resumed state; today resume rebuilds the `Atoms` from the initial structure.
- **`V3_KEYS = ("mc",)`**: one JSON record, present when a move set is attached. Its keys (`MC_RECORD_KEYS`):
  - `moves`: the attached records;
  - `cycle_position`: the step within the MD/MC cycle;
  - `counters`: per move type, `MC_COUNTER_KEYS` = attempted, impossible, accepted;
  - `streams`: per `config.MC_STREAMS` entry, `MC_STREAM_KEYS` = seed, offset;
  - `delta_A`: adaptive displacement step sizes.
- **`mc_record(restart)`**: stub, T08.
- **Restart writes** happen only at committed steps: never inside an MC block, never from an overflowed step.

## Protocol runs (`fastmace/protocols/`; owner T14)

Added by 3-T14 under the conductor's answers (`plans/phase-3.md`, "3-T14 swap-MC annealing protocol").

| Name | Signature | Meaning |
| --- | --- | --- |
| `protocols.quench.relax` | `relax(atoms, calc, fmax, max_steps=5000, logfile=None, optimizer="ase-LBFGS", cell_filter=None) -> (Atoms, summary)` | ASE LBFGS or BFGS at fixed cell (`langevin/quench.py:relax`); 4-T09: names resolved (`"LBFGS"` is `"ase-LBFGS"`), `cell_filter` an `fm.optimize.FrechetCell` through ASE's `FrechetCellFilter` |
| `protocols.quench.select_frames` | `select_frames(frames, min_K, bin_K, per_bin, peak_K)` | the research function, unchanged |
| `protocols.quench.picks` | `picks(resolved, frame_every, settings)` | `select_frames` over a run's frames (step 0, every `frame_every`), then the final frame |
| `protocols.quench.quench_run` | `quench_run(directory, relaxer, resolved, frame_every, timestep_fs, settings, *, name="quench", log=print) -> summary` | `<run>/quench/` (or `name`, 4-T09): `settings.json` (first; a continuation needs the same), `captured.traj`, `quenched.traj`, `quench.csv` (per frame; an interrupted quench continues), `quench.json` (last). `relaxer`: `quench.relaxer(...)` (ASE or GPU, 4-T09) or an ASE calculator |
| `protocols.runner.start` | `start(structure, options, *, relax=None, quench=None, steps=None, source=None, log=print) -> record` | relaxation, then `fm.md` for at most `steps` steps, a restart, and the quench once the schedule is done |
| `protocols.runner.resume` | `resume(directory, *, steps=None, log=print) -> record` | `Simulation.resume`, at most `steps` more steps with the move set attached as `fm.md` attaches it, a restart, the quench at the end |
| `protocols.runner.split_stages` | `split_stages(data) -> ({"relax", "quench"}, rest)` | the `[relax]` (`RELAX_KEYS = ("fmax", "optimizer", "max_steps")`) and `[quench]` (`profiles.QUENCH_KEYS`) tables of a protocol |
| `protocols.runner.protocol_names`, `protocol_file`, `DATA` | | the shipped protocols: `fastmace/protocols/data/*.toml`, by stem |
| `fastmace run` | `fastmace run [CONFIG \| NAME] [--steps N] [--output DIR]`, `fastmace run --resume RUN_DIR [--steps N]` | a file with `[relax]`/`[quench]`, or `--steps`, is a protocol run; otherwise `fm.md` as before |

- `protocol.json` (`PROTOCOL_RECORD_VERSION = 1`) in the run directory: source, structure, repeat, atoms, stages, relaxation summary, move records, `min_mc_temperature_K`, pieces (`first_step`, `last_step`, `wall_s`), status (`relax`, `md`, `quench`, `complete`) and the quench summary.
- `fm.md` with `steps=` and a schedule sizes `max_temperature_K` by the whole schedule (SPEC §7.7), so the rest of the schedule runs later on the same plan (3-T14; before, by the steps of the call).

## The ten refusal-pinning tests and the task that flips each

| Test | Pins | Flipped by |
| --- | --- | --- |
| `test_minimal_planner.py::test_d3_superset_beyond_the_structure_is_phase_3` (now `test_superset_beyond_the_structure_is_planned_by_rule_8`) | `superset-beyond-structure` | T02 (flipped) |
| `test_minimal_planner.py::test_rule_4_inexact_padding_refuses_the_graph_tier` | "compiled MD tier: phase 3" in rule 4 | T06 |
| `test_minimal_planner.py::test_other_modes_are_refused_with_their_phase` | `mode-compiled` | T06 |
| `test_minimal_planner.py::test_rule_11_records_max_temperature_and_keeps_the_1a_skin` (now `test_rule_11_md_only_keeps_the_1a_skin_and_records_the_rule`) | the 1a skin, "phase 3" | T05 (flipped) |
| `test_restart.py::test_the_1b_fields_are_in_the_state` | `FIELDS_ADDED_IN_1B == {"mc"}` | T08 |
| `test_interfaces.py::test_state_fields_of_phase_1a` (`FIELDS_ADDED_IN_1B`) | the same fact | T08 |
| `test_ase_md.py::test_forbidden_observer_changes_raise_with_their_reason` (numbers, masses, charges) | "phase 3" | T13 |
| `test_simulation.py::test_a_valid_mc_block_is_refused_with_mc_phase_3` and the `mc=` section after it | `mc-phase-3` | T15 (flipped) |
| `test_quick.py::test_mc_is_checked_then_refused_phase_3` and `test_every_refusal_at_once` | `mc-phase-3` | T15 (flipped) |
| `test_cli.py::test_run_mc_is_checked_then_refused` (now `test_run_mc_tables_become_move_records`) | "MC: phase 3" | T15 (flipped) |
| `test_metadata_schema.py` `LATER_PHASES` (`counters.mc`) | `counters.mc` null | T15 (flipped) |

The plan counts the `mc=` sections of `test_simulation.py` and `test_quick.py` as one each. The
table has eleven rows because `test_interfaces.py` pins the same fact as `test_restart.py`.

## Merge order of the shared files

T01 is first everywhere it appears.

| File | Editors, in merge order |
| --- | --- |
| `config.py` | T01 only (decision 1 below) |
| `planner/minimal.py` | T02 (rule 8), T05 (rule 11), T06 (owner: modes, rules 4 and 9), T09 (parity), T11 (MC energy precision) |
| `planner/fixed.py` | T01 (`_phase_3_fields`), T04 (re-capture budget rule), T05 (rule 11), T11 (MC energy precision) |
| `engine/graph.py` | T01 (stubs), T04 (owner), T03, T07, T10 |
| `simulation.py` | T01 (stubs), T03 (mutation API), T08 (restart), T15 (run path) |
| `model/forces.py` | T02 (owner), T07 |
| `neighbors/skin.py` | T01 (`skin="auto"` read as `DEFAULT_SKIN_A`), T02 (sender groups), T05 (owner), T10 (trial buffers) |
| `planner/fallback.py` | T06, T11, T17 |
| `provenance.py` | T04, T15 |
| `cli/run.py` | T15, T14 |
| `io/restart.py` | T01 (constants, stub), T08 (owner) |
| `fastmace/events.py` | T01 (contract), T03 (owner) |
| `fastmace/mc/` | T01 (records, stubs), T09 (owner), T10 |
| `engine/compiled.py` | T01 (stub), T06 (owner) |

## Decisions made in 3-T01

These are for the conductor's review at merge. The conductor's go asked the worker to decide the
open questions of its prestart report and record them here.

1. **The four new fields default to `"auto"` and are resolved in every plan.**
   - `skin`, `headroom`, `recapture_budget` and `mc_energy` all default to `"auto"`.
   - `FixedPlanner._phase_3_fields` resolves them to today's values and records a decision for each, as the planner tests require: no `"auto"` left in a plan, and one `auto (<rule>)` decision per field.
   - Default plans are unchanged except for the new fields' values and decisions.
   - T04, T05 and T11 replace the interim resolutions inside the planner, so none of them needs to edit `config.py`.
   - One assertion of the T05 refusal test moves: `test_rule_11_records_max_temperature_and_keeps_the_1a_skin` now compares with `DEFAULT_SKIN_A`. It still pins the 1a skin and "phase 3", and T05 flips it.
   - `main.py`'s adapter passes `skin` explicitly, so its plans now list `NeighbourConfig.skin` under `requested`.
2. **MC energy precision has one owner per role.**
   - `Workload.mc_energy_precision` is the user's request.
   - `PrecisionConfig.mc_energy` is the planner's resolved value in the plan (SPEC §10.6 "Plan: … MC energy precision").
   - The planner writes it (T11); nobody else sets it.
3. **Parity refusal.** It is a subclass, `ChargeParityRefused(PlanRefused)`, plus a reason code, so `except PlanRefused` still catches it.
4. **`captured(key, fn)` returns the replay callable.** Engine stage keys are refused, and T04 owns the method, because it carries the re-capture policy.
5. **The registry's bookkeeping is complete now.** `DerivedStateRegistry.register` works in T01, so T02 can register its buffers without waiting for T03. Refreshing and the CI check stay T03's.
6. **The restart writer stays at version 2 until T08.** A version 3 file without its new content would lie. T01 only fixes the key names.
7. **The move records and `configurational_mu` are complete, not stubs.** They are pure, torch-free and validated, so T12 and T15 can build against them now.
8. **Swaps admit all four momentum treatments.** §10.4 restricts only `exchange`, to swaps. The `fixcm` refusals of `rescale` and `redraw` are T09's planner check, not record validation.
9. **`set_species` changes masses.** The new species' standard masses apply unless `set_masses` follows, so a host species change needs no second call. `set_friction` takes 1/fs, the unit of the public integrator records.
10. **`MCConfig` reaches the driver as `MCDriver(config=)`.** It is not yet a `Simulation` or `fm.md` option. T15 decides whether it becomes one; it would be a new keyword-only parameter, pinned in `test_contracts.py`. T15: not yet; `Simulation` attaches with the default `MCConfig`.
