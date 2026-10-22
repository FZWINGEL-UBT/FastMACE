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
| `EngineConfig.recapture_budget` | `graph_variant_cap` (64); nothing reads it yet | T04 |
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
  - `names()`, `affected(changed)`, `in` and `len`;
  - `refresh(changed)` runs the affected entries in registration order and returns their names. An owner registers a buffer after the buffers it reads;
  - `check_against(fresh)`: the names whose buffers differ bit for bit from a fresh registry (or engine), or that only one side has.
- **Where the registry lives (T03):** `GraphEngine.derived`, built after the state, force field and integrator. The engine calls `register_derived(registry, state)` on the force field (if it has one; T02's `MACEForceField.register_derived`), then on the integrator, then on the fused integrator. Every derived tensor is also in the runtime-state registry, under its owner's name or as `derived.<name>` (policy `derived.*`: restart "derived", captured).
- **Refresh functions** take no arguments and read the committed inputs from the state: `state.species` (int64 atomic numbers), `state.masses` `(N, 1)`, `state.charges`, `state.total_charge`. They write in place at fixed addresses. Those that depend on species, masses or charges are capturable (no host sync, no new buffer; temporaries inside a graph are fine), tested in `tests/gpu/test_events_gpu.py`. A friction refresh may copy host floats: a friction event drops the graphs anyway.
- **Registered by T03:** Langevin `integrator.c3_unit`, `c4_unit`, `c5_unit` (masses, friction); BAOAB `integrator.inv_sqrt_masses` (masses) and `integrator.force_eps` (masses, momenta, forces, stress, cell); the fused integrator's `fused.coefficients` (friction).
- **`Transaction(engine, events, *, evaluate=None, registry=None)`**: `run()`, `quiesce()`, `snapshot()`, `apply()`, `refresh()`, `accept()`, `reject()` and `resume()`; `host_writes` (callables `write(state)` run after the events: `Simulation.update`'s positions and momenta); `changed` and `outcome`.
  - It validates every event when it is made, before anything is written. The host kinds are species, masses, charges and friction; `swap` and `transmute` raise `NotImplementedError` naming 3-T09, `displace` 3-T10, `barostat` (no host API in phase 3).
  - Values: `masses` amu and `species` atomic numbers or symbols, one per atom of `atoms`; `charges` an array or `{"charges", "total_charge"}`; `friction` in inverse ASE time units.
  - A total charge goes through the force field's host `set_total_charge(value)` (T02: its device total is base + Σ charges − Σ charges at the base, so T09 changes it in a graph through `state.charges`), then the `charges` refresh; `state.total_charge` keeps the host record. A rejection sets the old base back before the buffers are restored. A model without `set_total_charge` takes no total charge.
  - A mass change keeps the momenta, only where the mass changes. A species change sets the standard masses the same way; a masses event after it wins.
  - Accepted: `engine.transactions_committed` counts it; the resume is `canonicalise(reevaluate=True)`, then a refresh of what depends on forces and stress.
  - Rejected, impossible, redo: the runtime-state registry snapshot is restored in place, with the noise generator and cursor and the integrator's host constants.
  - A friction event calls T04's `GraphEngine.invalidate_graphs(reason)` (until T04 merges: `_clear_graphs()`), again on a rejection that captured meanwhile. With the default barostat τ it also changes `barostat` (`P_tau`, the barostat mass); an explicit `barostat_tau_fs` stays.
- **`events.check_against(engine, fresh)`**: the CI registry check of SPEC §12.2. It compares the derived registries, the state's composition, the neighbour list as the set of compacted edges, the integrator's host constants (the floats a stage bakes in, its config) and every other tensor reachable from the force field and integrator that the runtime-state registry does not hold. So a buffer left out of the registry fails it (`tests/cpu/test_events.py` has the negative controls).
- **`events.species_numbers(values, count)`**: atomic numbers from numbers or symbols, validated.

## Engines

- `GraphEngine.derived` (T03): the derived-state registry above. `GraphEngine.quiesce()`: an event's quiesce (`flush`).
- `GraphEngine.transactions_committed` (int) and `GraphEngine.state_version == (state.step, transactions_committed)` (T03): T07 tags its energy cache with it. Not restart state.

- `engine/compiled.py`: `MODE = "compiled"`; `CompiledEngine(config, force_field, integrator, state, neighbours=None, constraints=None, dynamic_edges=True)`. Stub, T06. Only the planner builds it.
- `engine/graph.py`:
  - `ENERGY_KEY = "E"` and `ENGINE_KEYS = ("A", "AC", "ACB", "ACP", "ACPB", "B", "C", "E")`;
  - `GraphEngine.energy(*, precision=None)`: the energy-only graph. Stub, T07;
  - `GraphEngine.captured(key, fn) -> replay`: captures `fn`, which takes no arguments and has fixed shapes and addresses, with the engine's warm-up, registry restore and re-capture policy (variant cap, budget, log). It refuses `ENGINE_KEYS`, which are the engine's own. Stub, T04, for T09's attempt graph.

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
| `test_minimal_planner.py::test_d3_superset_beyond_the_structure_is_phase_3` | `superset-beyond-structure` | T02 |
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
