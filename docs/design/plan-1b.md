# The Plan record, the minimal planner and `fm.ForceField` (phase 1b)

Plan task 1b-T12, 19 October 2026. Spec: SPEC §9.1–§9.5, §7.1, §7.6, §8.7, §4.4;
owner decisions D3 and D6 of `plans/phase-1b.md`. Code: `fastmace/planner/plan.py`,
`fastmace/planner/minimal.py`, `fastmace/planner/fixed.py` (the shared resolution),
`fastmace/model/forcefield.py`. Tests: `tests/cpu/test_minimal_planner.py`.

T15 (report), T16 (gate), T17 (`Simulation`), T18 (`fm.calc`), T20 and T21 build on
the names here.

## `Plan` (frozen)

| Field | Type | Meaning |
| --- | --- | --- |
| `tier` | str | `"graph"` (MD) or `"compiled"` (one-off evaluation) in 1b |
| `rewrites`, `engine`, `precision`, `neighbours` | config records | resolved: no `"auto"` left |
| `polar` | `PolarConfig` or `None` | `None` for non-polar models |
| `integrator` | `IntegratorConfig` or `None` | `None` for an evaluation plan |
| `decisions` | `((key, choice, reason), ...)` | every decision; `key` stable and unique; `plan.decision(key)` |
| `requested` | `((item, value), ...)` | options the caller set to a non-default value |
| `auto_options_off` | `((option, reason), ...)` | what `auto` switched off for the model (1a) |
| `fallbacks`, `predicted_memory_bytes` | `()`, `None` | phase 2 |
| `superset` | `tuple[int, ...]` | the species superset, atomic numbers sorted |
| `workload` | `Workload` or `None` | `None` for evaluation plans and 1a's fixed plans |
| `software` | `SoftwareTraits` or `None` | the traits the plan was made with |
| `allowances` | `tuple[str, ...]` | `allow_*` flags used, for example `"allow_naive_cueq"` |
| `revision`, `predecessor`, `reason`, `step` | | the revision chain (§9.5) |

- `plan.revise(reason, step, **changes)` returns revision n+1 with `predecessor` set to
  the plan it was called on. That plan does not change. `revision`, `predecessor`,
  `reason` and `step` cannot be passed in `changes`.
- `plan.chain()` gives the predecessors, oldest first.
- `str(plan)` is `describe()`.

`SoftwareTraits(oeq, triton, nvrtc, fused_cueq, mace_imported_first)`: each of the first
four is `(available, reason)`; `mace_imported_first` lists the modules that were imported
before FastMACE could prepare the OEq runtime (empty when the runtime is prepared, or
still can be). `minimal.detect_software()` probes them without importing MACE or OEq.

### Reasons and codes

A reason that reports or tests match on is a `fastmace.errors.Reason` (a `str` with
`.code`, and `.rule` for §9.3 rules). T15 matches codes, never text.

**Refusals** (`PlanRefused.reasons`; built only through `PlanRefused.from_reasons`):

| Code | Rule | Where |
| --- | --- | --- |
| `non-3d-pbc`, `constraints` | 1 | both planners |
| `variable-cell-without-barostat`, `reference-tier-not-in-1b` | 1 | minimal (D6) |
| `species-not-in-superset` | 2 | minimal, `build` → `SpeciesNotInSuperset` |
| `polar-plain-compiled-cueq` | 3 | both |
| `inexact-padding` | 4 | both (minimal text: "compiled MD tier: phase 3", D6) |
| `reproducible-backend`, `reproducible-conv-fusion`, `reproducible-family-not-qualified`, `reproducible-cueq-version`, `reproducible-kernel-not-deterministic` | 5 | minimal (2-T16; `planner/reproducible.py`) |
| `naive-cueq` | 6 | minimal (plan, or `build` after conversion) |
| `atom-count-changes` | 7 | minimal |
| `superset-beyond-structure` | 8 | minimal (D3: phase 3) |
| `mode-compiled`, `mode-eager`, `mode-reference` | – | minimal: MD modes 1b does not have |
| `hybrid-unavailable` | – | minimal: explicit `backend="hybrid"` that cannot run |
| `hybrid-needs-cuda`, `conv-fusion-needs-cueq`, `inplace-block-linear-not-carried-over`, `speculative-graph-npt`, `pipelined-excludes-speculative`, `rewrite-requirement-missing`, `rewrite-unsupported-by-model`, `reassociation-plain-cueq`, `streaming-needs-sorted-receivers`, `rewrite-needs-hybrid`, `unknown-workload-constraint` | – | both: explicit option requests (1a) |

**Decision codes** (the `reason` of these decision keys is a `Reason`):

| Key | Codes |
| --- | --- |
| `mode` | `mode-graph`, `mode-evaluation` |
| `EngineConfig.backend` | `backend-hybrid`, `backend-cueq-cpu`, `backend-cueq-oeq-unavailable`, `backend-cueq-mace-imported-first`, `backend-cueq-process-family`, `backend-cueq-requested`, `backend-cueq-reproducible` (2-T16) |
| `reproducible`, `reproducible.environment` | `reproducible-qualified` (the report names the undocumented cuEq internal), `reproducible-environment` (2-T16) |
| `runner` | `runner-subprocess-reproducible`, `runner-subprocess-family`, `runner-subprocess-not-reproducible` (2-T07, 2-T16) |
| `workload.host_callbacks_every` | `no-host-callbacks`, `host-callbacks-graph-sync` (rule 9, D6: the report says so) |
| `workload.max_temperature_K` | `max-temperature-recorded` (rule 11: 1a's skin and buckets) |
| `workload.mc_moves` | `no-mc`, `mc-phase-3` |
| `superset` | `superset` |
| `allow_naive_cueq` | `fused-cueq` (the fused kernels load), `allow-naive-cueq` (naive kernels allowed: the report warns) |

The other keys are minimal-plan facts: `force_precision`, `software.oeq`, `software.triton`,
`software.nvrtc`, `software.fused_cueq`, `software.mace_imported_first`,
`workload.species_changes` and `workload.needs_per_atom`. The rest are 1a's resolution:
`tier`, `workload`, `hardware`, `PrecisionConfig.dtype`, `EngineConfig.*`,
`RewriteConfig.*`, `NeighbourConfig.*`, `PolarConfig.kspace_headroom`, `capture`,
`integrator` and `compute_stress`.

### Restart record

`plan.to_restart(edge_bucket=, neighbour_capacity=)` returns a JSON-safe mapping for
`io.restart.state_arrays(plan=...)`:

- `plan_record_version` (1);
- `edge_bucket` and `neighbour_capacity` (`io.restart.PLAN_BUCKET_KEYS`, §10.6), the
  engine's at the restart point;
- `tier`, `revision`, `reason`, `step`, and `chain`: `[{revision, step, reason}]` of the
  predecessors;
- `rewrites`, `engine`, `precision`, `neighbours`, `polar` and `integrator` as
  dictionaries;
- `decisions` as `[key, choice, reason text, code or null, rule or null]`;
- `requested`, `auto_options_off`, `predicted_memory_bytes`, `superset`, `workload`,
  `software` and `allowances`.

`Plan.from_restart(record)` rebuilds the plan, with the predecessors as summaries, and
`from_restart(r).to_restart(...) == r`. A plan with fallback candidates has no restart
record before phase 2.

## `MinimalPlanner`

```python
planner = MinimalPlanner(engine=None, neighbours=None, polar=None, precision=None, *,
                         registry=None, environ=None, torch_version=None, software=None, process=None)
plan = planner.plan(ff, atoms, integrator_config, workload, hardware=None, *,
                    variable_cell=None, allow_naive_cueq=False, allow_reference=False, mode="graph")
engine = planner.build(plan, ff, atoms, *, compile_cache=None, check=True, temperature_K=None)
plan = planner.plan_evaluation(ff, atoms, hardware=None, *, allow_naive_cueq=False)
field = planner.build_evaluation(plan, ff, atoms, *, compile_cache=None, check=True)   # MACEForceField
```

- The force precision and rewrites come from the `ForceField`. `precision=` carries only
  `float64_energy` and `tf32`; its `dtype` must stay `"auto"`.
- `plan_traits(...)` and `plan_evaluation_traits(...)` take model traits instead of a
  `ForceField`, for tests.
- The rewrite set and the engine options come from `FixedPlanner._resolve`, 1a's
  resolution, unchanged. A `main.py` configuration therefore gets `FixedPlanner`'s
  rewrites and engine options. This includes 1b-T11's `generic_segmented_first_sum`
  record, which the registry resolution enables.
- **Process family (§8.7):** the first compiled plan claims the compile family in
  `process` (default `engine.constraints.PROCESS`). After that, `auto` picks the
  process's family. An explicit plan of the other family raises
  `ProcessConstraintError`, whose message says the subprocess runner comes in phase 2. A
  plan with `compile_mode="none"` claims nothing.
- **Rule 6** is checked at plan time from `software.fused_cueq` on CUDA, and on the loaded
  model when it already has the plan's backend. It is checked again in `build`, where the
  conversion's `NaiveCueqError` becomes `PlanRefused` with rule 6.
- `build` checks that the structure's species equal the plan's superset, converts the
  model and builds through `FixedPlanner.build`. That passes `float64_energy` and the
  process record through.
- **Evaluation plans:** tier `"compiled"`, stress on, no integrator, engine options off,
  no capture, exact padding not required.

## `fm.ForceField`

`ForceField(model, atoms=None, *, precision="float32", rewrites="auto", species=None)`:

- `model` is an `fm.load` result or a label. Any other string raises `TypeError` (load it
  with `fm.load` first).
- `species` is compared as a set. A species without an embedding raises
  `ModelElementError`.
- `ff.species` is the superset (atomic numbers, sorted), or `None` until first use.
  `ff.superset_for(atoms)` fixes it from the first structure.
- `ff.identity` is the `state.ForceFieldIdentity` record:
  - the checkpoint SHA-256, head, superset and precision;
  - physics: charge and spin;
  - the MACE and FastMACE versions;
  - `None` until the superset is known.
- `ff.traits` holds the model traits of the pristine model.
- `ff.execution_model(plan)` is for the planner only. It loads the checkpoint again with
  the plan's backend, precision and device, checking the hash. Each plan gets its own
  converted model, and the pristine model never changes.
- `ff.evaluate(atoms, stress=False)` returns `(energy, forces, stress or None)`.
  - It plans once per species sequence through `minimal.default_planner`: CUDA if there
    is one, else the CPU without compilation.
  - It keeps the evaluation plan and force field for later calls;
    `ff.evaluation_plan(atoms)` returns that plan.
