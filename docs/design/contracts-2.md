# Contracts of phase 2: safety

Plan task 2-T01, 20 October 2026. Spec: SPEC §7.6, §8.7, §9.2–§9.5, §12.3;
decisions 2, 5, 29 and 30 (SPEC §18); `plans/phase-2.md`, "Conductor answers to
the prestart reports". Every later phase 2 task builds against these names and
signatures. They are pinned by `tests/cpu/test_contracts_2.py`,
`tests/cpu/test_simulation_options.py`, `tests/cpu/test_config.py` (the switch map)
and the updated signatures in `tests/cpu/test_contracts.py`.

Every default keeps 1b's behaviour. Default plans, forces and trajectories are
bitwise unchanged.

## Config records and fields (`config.py`; T01 owns it, T20 adds only its screens' fields)

| Name | Default | Meaning | Filled by |
| --- | --- | --- | --- |
| `Mode` | `Literal["graph", "compiled", "eager", "reference"]` | the execution mode; `config.MODES` equals `planner.minimal.MODES` | T02, T03 |
| `FallbackConfig.fallback` | `"auto"` | the §9.4 search on failure (decision 29); `"refuse"` raises as in 1b, with the record | T15 |
| `FallbackConfig.diagnosis_budget` | `64` | reduced-gate runs per fallback; full gate runs are counted apart | T15 |
| `FallbackConfig.reference_last` | `True` | the reference tier may end the terminal sequence, if it fits | T15 |
| `GateConfig.cpu_float64_time_limit_s` | `300.0` | CPU float64 is tried after GPU memory runs out only within this estimated time; `0` never | T05 |
| `GateConfig.reduced_cell` | `"auto"` | the reduced-cell rule (`largest_smaller_supercell_at_least_min_height`), or `"off"` | T05 |
| `GateConfig.reduced_cell_min_height_cutoffs` | `2.0` | every height of the reduced cell is at least this × the cutoff (≥ 1) | T05 |
| `RunnerConfig.python` | `None` | the child's interpreter; `resolved_python()` gives `sys.executable` for `None` | T07 |
| `RunnerConfig.environment` | `()` | variables set in the child; a mapping is normalised to sorted `(name, value)` pairs | T07, T16 |
| `RunnerConfig.inherit_environment` | `True` | the child starts from the parent's environment | T07 |
| `RunnerConfig.timeout_s` | `None` | total limit of one child run | T07 |
| `RunnerConfig.stall_timeout_s` | `1800.0` | a child without a committed step for this long is killed | T07 |
| `RunnerConfig.persistent` | `True` | one worker child serves every `run()` of a `Simulation`; `False`: a fresh child per call (added by OPT-T07) | OPT-T07 |
| `EngineConfig.dynamic_edges` | `False` | evaluation plans compile with a dynamic edge dimension | T13 |
| `RewriteConfig.selfcheck_cache` | `"use"` | the qualification cache: `"use"`, `"off"`, `"refresh"` (inert until T19) | T19 |

`Workload.reproducible` already exists (1b-T01) and is unchanged. Each field has
a row in `switch-map.md` (section 10 for the new records).

## Planner inputs of `Simulation` and `fm.md`

```python
Simulation(atoms, force_field, *, species=None, integrator, workload=None, output=None, seed,
           check="skip", allow_naive_cueq=False,
           mode="graph", allow_reference=False, fallback=None, gate=None, runner=None,
           engine=None, neighbours=None)
fm.md(..., integrator=None, mode="graph", allow_reference=False, fallback=None, gate=None,
      runner=None, engine=None, neighbours=None)
```

- They are validated together by `simulation.planner_options(...)` into a frozen
  `simulation.PlannerOptions`, before anything is loaded (`fm.md` validates them
  right after `check`). A wrong mode raises `ValueError`, and a wrong type
  `TypeError`.
- `None` for `fallback`, `gate` and `runner` means the record's defaults. For
  `engine` and `neighbours` it means the default planner's
  (`minimal.default_planner`: CUDA if available, else the CPU without compilation).
  A given `engine` or `neighbours` replaces the default planner's in a copy of it,
  keeping its process record and software traits.
- `mode` and `allow_reference` go to `MinimalPlanner.plan`. `mode="compiled"`
  still refuses there (code `mode-compiled`, phase 3). T02 lifted `mode-eager`;
  T03 lifted `mode-reference` and `reference-tier-not-in-1b`: `mode="reference"`
  plans the reference tier, and `allow_reference=True` is recorded in
  `Plan.allowances` (it lifts no §4.4 refusal, because the tier runs FastMACE's
  integrators).
- `fallback`, `gate` and `runner` are stored and recorded, with no effect until
  T05, T07 and T15. They can be read as `sim.fallback`, `sim.gate_config` and
  `sim.runner` (configs; `sim.gate` stays the gate's result) and as `sim.options`.
- `simulation.json` (record version 2) holds `planner_options`
  (`PlannerOptions.to_record()`, JSON-safe; `engine`/`neighbours` `null` when not
  given). `Simulation.resume` re-plans with them. A version 1 record resumes
  with the defaults.
- `fastmace run` reads them from TOML: `mode` and `allow_reference` as keys, and
  `[engine]`, `[neighbours]`, `[fallback]`, `[gate]` and `[runner]` as tables of
  their records' fields (`cli/run.py`). `fastmace init` writes them commented
  (`cli/init.py`), because it names every `fm.md` argument. `fastmace md` has no
  flags for them yet.

## Errors (`errors.py`, new classes only)

All four have the §7.7 message shape (what, why, next, record, environment).

- `FallbackExhausted(PlanRefused)`: `FallbackExhausted(reasons, ..., *, revisions=())`.
  `revisions` lists the plans the fallback tried, oldest first. The record is
  `run.plan` (its fallbacks and revision chain) and `metadata.json`.
- `SubprocessRunnerError(FastMACEError, RuntimeError)`: `exit_code`
  (negative for a signal, `None` if stopped), `child_message`, `directory` (the
  hand-off directory, kept after an error).
- `IsolationError(FastMACEError, RuntimeError)`: `component` and `holder`
  (who holds the lease now). The next step is `rebind()`.
- `ReproducibilityError(FastMACEError)`: `blocking` (kernel names). It is raised
  for a failed qualification only. A `reproducible=True` plan without a
  qualified family stays `PlanRefused` (rule 5).

## Records

- `Rewrite.deterministic: bool | None = None`, the last field. `None` means
  undeclared, which a reproducible plan refuses (§9.3). `Rewrite.from_dict` does
  not require it (T16 declares it per record).
- `Plan.runner: Literal["in-process", "subprocess"] = "in-process"`
  (`plan.RUNNERS`). `"subprocess"` marks a plan that §8.7 forbids in this
  process, which the runner runs in a child (T07). It is not in the restart
  record: the child resumes the plan in its own process. `describe()` prints it
  when it is not in-process.
- Plan revision of a fallback (T06, T15): `failed_plan.revise("fallback: <change>:
  <reason>", step=0, ...)`. Every tried candidate is a
  `planner.fallback.TriedCandidate` in `Plan.fallbacks`:
  - `dimension`: `mode`, `rewrites` or `backend`;
  - `change`;
  - `outcome`: `activated`, `passed` (a diagnosis run that passed the reduced gate,
    never activated; T15), `failed` or `skipped`;
  - `reason`;
  - `stage`: `build`, `compile`, `capture`, `self-check` or `gate`;
  - `revision`;
  - `reduced_gate_runs` and `full_gate_runs`.
  `PLAN_RECORD_VERSION` 2, which encodes `fallbacks`, is T06's. Version 1
  records still read.
- Gate record (`planner.selfcheck.GateRecord`): T05 adds two fields.
  - `device`: the device the reference ran on (`"cuda:0"`, `"cpu"`).
  - `reduced_cell`: the reduced cell's repeat of the primitive cell, three ints
    (or the cell itself), or `None` on the production cell. Its status is then
    `passed-reduced`.
  The tried steps (device, cell, time, outcome) go into `notes` and `seconds`, or
  a new field, as T05 decides.

## Stubs (each raises `NotImplementedError` naming its task)

| Module | Names and signatures | Task |
| --- | --- | --- |
| `engine/subprocess.py` | `SubprocessRunner(config=None)`; `.run(simulation, *, steps=None, schedule=None)`; `main(argv=None)` for `python -m fastmace.engine.subprocess <directory>`; `HANDOFF_NAME = ".fastmace-handoff"` | T07 |
| `engine/reference.py` | `ReferenceForceField(force_field, atoms, *, device=None, dtype="float64", polar=None, compute_stress=False)`, implementing `model.forces.ForceField` (T03 added `polar` and `compute_stress`); `MinimalPlanner.build_reference(plan, force_field, atoms)` and `MinimalPlanner.reference_fits(plan, force_field, atoms, references=None) -> (bool, reason)` | T03 |
| `planner/fallback.py` | `candidate(plan, changes, force_field, atoms, *, planner=None)`, `activate(candidate, planner, force_field, atoms, *, references, reduced=False, quantities=None, budget=None, ...)` | T06 (merged) |
| `planner/fallback.py` | `Fallback(planner, force_field, atoms, integrator, references, config=None, *, seed=0, gate=None).run(failure)` with `failure` a `Failure(plan, stage, error, gate=None)`, returning a `Result(plan, engine, gate, budget, tried, runner=False)` or raising `FallbackExhausted` (with `.tried`, `.budget`, `.failure`); `diagnose(plan, failure, *, test)`; `ddmin(rewrites, test, *, requires, budget)` (raises `DiagnosisBudgetSpent`); `terminal_sequence(plan, *, reference=True)`; helpers `enabled_rewrites`, `field_requires`, `dependants`, `replay_off`, `stored_fallback_plan` | T15 |
| `planner/fallback.py` | records `TriedCandidate`, `Budget(limit, reduced_gate_runs=0, full_gate_runs=0)` with `.spent`; `DIMENSIONS`, `OUTCOMES`, `STAGES` | complete |
| `planner/reproducible.py` | `AllowListEntry(component, kind, deterministic, reason, condition="")` (kind: `cueq-kernel`, `torch-op`, `fastmace-kernel`); `Qualification(backend, family, qualified, blocking=(), entries=())`; `ALLOW_LIST = ()`; `qualify(plan, ...)` | T16 |

The four modules import neither torch nor MACE at import time (tested in a fresh
interpreter).

## Merge order of the shared files

From the conductor's answers (`plans/phase-2.md`), T01 first everywhere it
appears:

| File | Editors, in merge order |
| --- | --- |
| `config.py` | T01 (owner), then T20 for its screens' fields only |
| `simulation.py` | T01 (options), T09 (resume), T10 (`_write_json` only), T07 (runner), T15 (fallback call) |
| `quick.py` | T01 (options) |
| `cli/init.py`, `cli/run.py` | T01 (the options' keys and tables) |
| `planner/plan.py` | T01 (`runner` field), T06 (fallbacks, record version 2) |
| `rewrites/registry.py` | T01 (field), T04 (owner), T19 (cache lookup), T20 (screen records) |
| `planner/minimal.py` | T02, T03, T07 (`_claim_family` only), T13, T16, then T15 (owner) |
| `planner/fixed.py` | T02 (`capture=` on `build`), T16 (reproducible overrides) |
| `model/forces.py` | T04, T13 (`dynamic_edges` and the cache key), T08 (`rebind`, if a lease is needed) |
| `provenance.py` | T10 (`write_metadata` only), T17 (owner) |
| `reference/calculators.py` | T03, T05 |
| `report.py` | T05 (gate line), T15 (revision lines) |

## Notes for the tasks that fill the stubs

- T07: `Simulation.runner` is the `RunnerConfig` property. Keep the live runner
  in a private attribute (`_runner`), as T07's note proposes.
- T07 and T16: `RunnerConfig.environment` carries T16's variables
  (`CUBLAS_WORKSPACE_CONFIG`, `CUEQUIVARIANCE_OPS_U1D_ALGOS`,
  `TORCHINDUCTOR_DETERMINISTIC`) into the child.
- T15: `Simulation.fallback` is the config to pass into `Fallback(...)`.
