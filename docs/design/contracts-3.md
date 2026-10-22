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
| `MCConfig.max_attempts_per_block` | `10_000` | attempts of one move per block | T09 |
| `MCConfig.max_move_sets` | `8` | move records attached at once | T09 |

Constants and functions:
- `DEFAULT_SKIN_A = 0.3`;
- `MC_ENERGY_PRECISIONS`, equal to `Workload.mc_energy_precision`'s choices;
- `MC_STREAMS = ("proposal", "acceptance", "velocity")`;
- `SWAP_VELOCITIES = ("exchange", "rescale", "redraw", "redraw_all")`;
- `TRANSMUTE_VELOCITIES = ("rescale", "redraw", "redraw_all")`;
- `mc_stream_seeds(seed) -> {stream: int}`: 63-bit seeds from SHA-256 of `"fastmace-mc:{seed}:{stream}"`. They are the same in every process and differ per stream.

`mode="compiled"` is already one of `config.MODES`. 3-T06 lifts the planner's `mode-compiled`
refusal.

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
| `Simulation.set_species` | `set_species(species)`: atomic numbers or symbols, `(N,)` | stub, T03 (superset check T02) |
| `Simulation.set_masses` | `set_masses(masses)`: amu, `(N,)` | stub, T03 |
| `Simulation.set_charges` | `set_charges(charges, *, total_charge=None)` | stub, T03 |
| `Simulation.set_friction` | `set_friction(friction_per_fs)`: 1/fs, as the public integrator records | stub, T03 |

Notes on the records:
- **Species stay as given:** symbols or atomic numbers ≥ 1. The driver maps them to atomic numbers.
- **`Simulation.run(mc=…)` reads them already:** `_refuse_mc` reads `.move`, `.pairs` and `.species`, so the workload, temperature and superset checks apply to the real records today. The call still ends in the `mc-phase-3` refusal until T15.
- **`configurational_mu` formula:** Δμ_s^conf = Δμ_s^full + (3/2) k_B T ln(m_s / m_ref), with k_B = 8.617333262 × 10⁻⁵ eV/K. The reference's own Δμ must be 0. `masses=None` takes ASE's standard atomic masses.

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
  - `names()`, `affected(changed)`, `in` and `len` are complete;
  - `refresh(changed)` and `check_against(fresh)` (the CI registry check) are T03's.
- **`Transaction(engine, events, *, evaluate=None, registry=None)`**: `run()`, `quiesce()`, `snapshot()`, `apply()`, `refresh()`, `accept()`, `reject()` and `resume()`. Stub, T03.

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

- `engine/compiled.py`: `MODE = "compiled"`; `CompiledEngine(config, force_field, integrator, state, neighbours=None, constraints=None, dynamic_edges=True)`. Stub, T06. Only the planner builds it.
- `engine/graph.py`:
  - `ENERGY_KEY = "E"` and `ENGINE_KEYS = ("A", "AC", "ACB", "ACP", "ACPB", "B", "C", "E")`;
  - `GraphEngine.energy(*, precision=None)`: the energy-only graph. Stub, T07;
  - `GraphEngine.captured(key, fn) -> replay`: captures `fn`, which takes no arguments and has fixed shapes and addresses, with the engine's warm-up, registry restore and re-capture policy (variant cap, budget, log). It refuses `ENGINE_KEYS`, which are the engine's own, and tuple keys whose first element is one. The graph is captured at the first replay and again after an invalidation or eviction; without capture the replay calls `fn`. A tuple key's first element is its family, so `("mc.swap", bucket)` keys are variants of one key. Complete (T04), for T09's attempt graph.
  - `GraphEngine.invalidate_graphs(reason, keys=None)`: drops the captured graphs (all of them, which also retires the pool, or those of `keys`) and logs the reason. The next capture of a dropped key counts as a re-capture. T03 calls it for friction events. Complete (T04).
  - `GraphEngine.uncommitted() -> bool`: true while a pipelined or speculative step is unvalidated, or an overflowed evaluation is being redone. T08's restart write guard reads it. Complete (T04).

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

## The ten refusal-pinning tests and the task that flips each

| Test | Pins | Flipped by |
| --- | --- | --- |
| `test_minimal_planner.py::test_d3_superset_beyond_the_structure_is_phase_3` (now `test_superset_beyond_the_structure_is_planned_by_rule_8`) | `superset-beyond-structure` | T02 (flipped) |
| `test_minimal_planner.py::test_rule_4_inexact_padding_refuses_the_graph_tier` | "compiled MD tier: phase 3" in rule 4 | T06 |
| `test_minimal_planner.py::test_other_modes_are_refused_with_their_phase` | `mode-compiled` | T06 |
| `test_minimal_planner.py::test_rule_11_records_max_temperature_and_keeps_the_1a_skin` | the 1a skin, "phase 3" | T05 |
| `test_restart.py::test_the_1b_fields_are_in_the_state` | `FIELDS_ADDED_IN_1B == {"mc"}` | T08 |
| `test_interfaces.py::test_state_fields_of_phase_1a` (`FIELDS_ADDED_IN_1B`) | the same fact | T08 |
| `test_ase_md.py::test_forbidden_observer_changes_raise_with_their_reason` (numbers, masses, charges) | "phase 3" | T13 |
| `test_simulation.py::test_a_valid_mc_block_is_refused_with_mc_phase_3` and the `mc=` section after it | `mc-phase-3` | T15 |
| `test_quick.py::test_mc_is_checked_then_refused_phase_3` and `test_every_refusal_at_once` | `mc-phase-3` | T15 |
| `test_cli.py::test_run_mc_is_checked_then_refused` | "MC: phase 3" | T15 |
| `test_metadata_schema.py` `LATER_PHASES` (`counters.mc`) | `counters.mc` null | T15 |

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
10. **`MCConfig` reaches the driver as `MCDriver(config=)`.** It is not yet a `Simulation` or `fm.md` option. T15 decides whether it becomes one; it would be a new keyword-only parameter, pinned in `test_contracts.py`.
