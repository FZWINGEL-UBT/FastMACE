# `metadata.json`, schema `fastmace.metadata/1`

Plan task 1b-T10, 19 October 2026. Spec: SPEC §13, §10.6 (restart points and output chunks),
§12.1 (reference record id), owner decision D1 (opt-in gate). Code: `fastmace/provenance.py`.
Tests: `tests/cpu/test_metadata_schema.py`. Phase 2 additions: plan task 2-T17, 20 October 2026
(tests also in `tests/cpu/test_metadata_simulation.py`).

Phase 1b writes version 1. Phase 2 adds keys and changes none, so the schema stays
`fastmace.metadata/1`. See "What phase 2 adds" below. OPT-T07 (20 October 2026) adds
`subprocess.calls` and `subprocess.persistent` by the same rule. `subprocess.children` still
counts child processes; before OPT-T07 there was one child per call, so it equalled the calls.
Phase 5 (plan task 5-T01, 23 October 2026) adds keys too and records it in the minor version
`schema_minor` (1); see "What phase 5 adds" below.

## How a run writes it

```python
from fastmace import provenance

record = provenance.metadata_record(model=..., checkpoint=loaded, identity=ff.identity, plan=plan,
                                    seed=seed, generators={...}, cursors={...}, gate=sim.gate,
                                    counters={...}, engine_info=engine.info(), restart_points=[...],
                                    setup={...}, timing={...}, report=report, device=device,
                                    force_field=engine.force_field, subprocess=None)
provenance.write_metadata(directory / "metadata.json", record)
```

- `Simulation` (1b-T17) and `fm.md` (1b-T20) call these two functions.
- Every input is optional. A missing input gives `null`, or an empty list for `plan_revisions`
  and `restart_points`.
- `write_metadata` writes a temporary file and then renames it. It refuses a record of another
  schema, and NaN or infinity; the old file then stays as it was.
- `main.py`'s files do not change: `model_block`, `fastmace_block` and `run_record` keep their
  1a output. The only addition is `trust` in `model_block`, right after `checkpoint_sha256`.

## Random state: hashes, never states

`random.generators` maps each generator's name to `generator_digest(...)`: the SHA-256 of
its state. The function accepts:

- a `torch.Generator`, or a state tensor;
- a numpy `Generator`, `BitGenerator` or `RandomState`;
- bytes;
- a digest already made (64 hex digits).

`random.cursors` holds the stream positions of the batches drawn ahead. Full states and
those batches live only in the restart file (§10.6).

The record refuses, anywhere in it, with `ValueError`:

- raw bytes;
- a generator object outside `generators=`;
- a tensor or array of more than 9 values.

Smaller arrays, such as a 3×3 pressure tensor, become lists.

## Keys

Top-level keys, in order. "Source" is the input of `metadata_record`.

| Key | Content | Source |
| --- | --- | --- |
| `schema` | `"fastmace.metadata/1"` | |
| `schema_minor` | `1` (5-T01): additions within version 1; a record without it is minor 0 | `provenance.SCHEMA_MINOR` |
| `created_utc` | ISO time | `created_utc` (default now) |
| `code` | `version`, `git_commit`, `git_dirty` (§13 "code version") | `fastmace_block` |
| `checkpoint` | `path`, `sha256`, `head`, `licence`, `label`, `trust` | `checkpoint`: a `LoadedModel` (`provenance()`) or a mapping |
| `model` | the 1a `model_block`, unchanged | `model` |
| `force_field_identity` | `ForceFieldIdentity.to_dict()` plus `digest` | `identity` (T05's record or its dict) |
| `reference_record` | the §12.1 record; at least `id` | `reference_record`: a mapping, `to_dict()`, or `.id`; must equal the identity's `reference_record_id` |
| `plan` | active revision: `revision`, `tier`, `reason`, `step`, `description` (`str(plan)`), `backend`, `conv_fusion`, `rewrites`, `engine`, `neighbours`, `polar`, `integrator`, `decisions` (`key`, `choice`, `reason`, `code`, `rule`), `requested`, `auto_options_off`, `superset`, `workload`, `software`, `allowances`, `fallbacks_tried` (each `TriedCandidate` of 2-T06/2-T15: `dimension`, `change`, `outcome`, `reason`, `stage`, `revision`, `reduced_gate_runs`, `full_gate_runs`; `[]` when no fallback ran), `predicted_memory_bytes`, `runner` (`"in-process"` or `"subprocess"`, 2-T07), `calibration` (`CalibrationRecord`, 5-T09; `null` when none ran), `adaptation` (the active `AdaptationConfig`, 5-T07; `null` before) | `plan` (T12's `Plan`) |
| `plan_revisions` | `[{revision, tier, step, reason, description, runner, fallback}]`, oldest first, active last (§9.5); `fallback` is the activated candidate that became the revision, else `null` | `plan.chain()` and `plan` |
| `precision` | `force`, `integration` (always `"float64"`, §7.3), `tf32`, `float64_energy` (T03, D2) | `plan.precision`; without a plan, the identity's force precision |
| `runtime` | `RuntimeInfo`: `file_locks`, `file_lock_source`, `file_lock_switch_late`, `cuda_driver_cache` (§14) | `fastmace.runtime_info()` |
| `software` | `versions` (Python and packages), `kernels` (Triton, NVRTC, streaming probes), `patches`, `triton` (version), `oeq_extension` (the OEq build variables for a hybrid plan, else `null`), `treatments` (each always-on treatment of 2-T04: `name`, `qualified`, `decision`, `reason`; `null` without a plan, so MACE is not imported) | `fastmace_block`; `rewrites.versions.treatments()` |
| `hardware` | `gpu`, `driver`, `compute_capability`; `null` off CUDA | `device` |
| `random` | `seed`, `generators` (hashes), `cursors` | `seed`, `generators`, `cursors` |
| `integrator` | integrator parameters | `integrator`, default the plan's `IntegratorConfig` |
| `schedule` | the schedule and its position | `schedule`: a mapping or `to_restart()` (T07) |
| `gate` | `status`, `reason`, `selfchecks`, `device`, `reduced_cell`, `steps`, and the rest of the gate record (2-T05: `copies`, `quantities`, `references`, `seconds`, `replay`, …). `selfchecks` defaults to the plan's per-record self-checks (2-T04 `SelfCheck`: `name`, `passed`, `max_relative_error`, `threshold`, `modules`, `dtype`, `detail`) | `gate`; default `{"status": "skipped", "reason": "opt-in: check=\"gate\"", ...}` (D1), with `device` and `reduced_cell` `null` and `steps` `[]` |
| `counters` | `recaptures`, `neighbour_lists` (`{name: {rebuilds, build_time_s, skin_A}}` for MACE and D3), `overflows_redone`; `mc` (3-T15: `{move: {attempted, impossible, accepted}}`, plus `trial_rebuilds` for `displace`; `null` when no move set was ever attached); `tier_switches` (5-T01: a list of `{step, from, to, reason, revision}`, `[]` without a switch; 5-T07 fills it); `engine` (the engine's `info()`) | `counters`, `engine_info` |
| `restart_points` | `[{step, output_chunk, output_rows, output_frames, ...}]` (T06's indices, §10.6) | `restart_points` |
| `timing` | `setup_s` (`{stage: seconds}`), `setup_total_s`, then the timing summary | `setup`, `timing` |
| `report` | the run report | `report`: T15's `to_metadata()` or a mapping |
| `force_field` | the engine's force field record: backend, dtype, rewrites applied, their decisions and self-check errors, `energy_treatment`, `strain_path` (`"one_graph"`, `"mace"`, or `null` if not bound; 2-T08b), … | `force_field`: the engine's force field (`info()`) or a mapping |
| `subprocess` | `null` in-process; for a plan in a child (2-T07): `python`, `reason`, `directory`, `children` (child processes started), `calls` (setup and `run()` calls they served; OPT-T07), `persistent` (`RunnerConfig.persistent`; OPT-T07), `seconds`, `failed`, `last` (the last call: `action`, `stage`, `pid`, `exit_code` (`null` while a persistent worker is still running), `seconds`, `steps`; or `null`) | `subprocess`: a `SubprocessRunner` or its `record()`, checked by `provenance.subprocess_block` |

Setup stages use §12.5's names: load, rewrites, compilation, capture, gate. 1b-T11 split them
further: load, plan, rewrites, self-checks, model copy, first compiled call, captures. The
record keeps whatever stage names it is given. Values must be seconds ≥ 0.

`counters` takes only the keys listed. `tier_switches` takes `planner.plan.TierSwitch` records or
mappings with exactly `provenance.TIER_SWITCH_KEYS`. `counters.mc` is checked: known move types only, exactly
those keys.

## SPEC §13 item by item

`provenance.SPEC_13_KEYS` maps every item to its key path, and the test lists them again.

| §13 item | Key |
| --- | --- |
| model path, SHA-256, head, licence | `checkpoint.path`, `.sha256`, `.head`, `.licence` |
| force-field identity, reference record | `force_field_identity`, `reference_record` |
| plan: tier, rewrites on/off with reasons | `plan.tier`, `plan.rewrites`, `plan.decisions` |
| force and integration precision, TF32 | `precision.force`, `.integration`, `.tf32` |
| backend, convolution fused, fallbacks tried | `plan.backend`, `plan.conv_fusion`, `plan.fallbacks_tried` |
| file-lock mode, CUDA driver cache | `runtime.file_locks`, `runtime.cuda_driver_cache` |
| package and kernel versions, OEq build, Triton | `software.versions`, `.kernels`, `.oeq_extension`, `.triton` |
| GPU name, driver, compute capability | `hardware.gpu`, `.driver`, `.compute_capability` |
| seeds, generator-state hashes, cursors | `random.seed`, `.generators`, `.cursors` |
| integrator parameters, schedule | `integrator`, `schedule` |
| gate status, self-check results | `gate.status`, `gate.selfchecks` |
| re-captures; rebuilds, build times, skins; overflows redone | `counters.recaptures`, `.neighbour_lists`, `.overflows_redone` |
| tier switches; MC counters | `counters.tier_switches` (a list, 5-T01); `counters.mc` (3-T15) |
| restart points with chunk indices; timing summary | `restart_points`; `timing` |
| code version (package + git hash) | `code.version`, `code.git_commit` |

What 1b-T10 adds:

- the schema version: `schema`;
- the plan revision chain: `plan_revisions`;
- the setup breakdown: `timing.setup_s`;
- the report: `report`;
- the 1a block: `model`;
- the trust: `checkpoint.trust`;
- the float64 energy: `precision.float64_energy`.

## What phase 2 adds (plan task 2-T17)

`SPEC_13_KEYS` lists these too. A run that takes the path behind a key leaves it non-null
(`test_no_phase_2_key_is_null_on_the_paths_that_fill_it`; a real run in
`test_metadata_simulation.py`).

| Item | Key | Filled by |
| --- | --- | --- |
| fallbacks tried, with reasons, and the revision each became | `plan.fallbacks_tried`, `plan_revisions[].fallback` | 2-T06, 2-T15 |
| where the plan runs | `plan.runner`, `plan_revisions[].runner` | 2-T07 |
| per-record self-checks | `gate.selfchecks` (the plan's `SelfCheck`s) | 2-T04 |
| always-on treatments | `software.treatments` | 2-T04 |
| gate reference device, reduced cell, resource steps | `gate.device`, `gate.reduced_cell`, `gate.steps` | 2-T05 |
| the subprocess runner and its children | `subprocess` | 2-T07 |
| a persistent worker: calls served, and whether the runner keeps its child | `subprocess.calls`, `subprocess.persistent` | OPT-T07 |
| strain path bound by the force field | `force_field.strain_path` | 2-T08b |
| compiled with dynamic edges | `plan.engine.dynamic_edges` (and its decision) | 2-T13 |
| reproducible mode | `plan.workload.reproducible`; the verdict and its environment are the `reproducible` decisions | 2-T16 |
| versions changed on a resume | the revision's `reason` in `plan_revisions`, and a report warning | 2-T09 |

`Simulation` records the built plan (`MinimalPlanner.build` adds the self-checks to the engine's
`plan`) when its revision is the active one.

A self-check result marked "cached" is not recorded, because 2-T19 added no qualification cache.

## What phase 5 adds (plan task 5-T01)

`SPEC_13_KEYS` lists these too (`docs/design/planner-5.md`).

| Item | Key | Filled by |
| --- | --- | --- |
| schema minor version | `schema_minor` | 5-T01 |
| calibration record | `plan.calibration` | 5-T09 |
| adaptation limits | `plan.adaptation` | 5-T07 |
| tier switches | `counters.tier_switches` (no longer `null`) | 5-T07 |

## Not in version 1

- **OEq extension build id.** The OEq runtime of 1a-T00 has no build id. The record keeps
  the build variables instead (`OEQ_JIT_EXTENSION`, `OEQ_EXTENSION_SEED` and the others of
  `OEQ_ENVIRONMENT_VARIABLES`), the same ones `model_block` writes under `oeq_wrapper`.
- **Validation against a schema file.** No phase 2 task has it. `metadata_record` checks its
  inputs, and `subprocess_block` checks the runner's block.
