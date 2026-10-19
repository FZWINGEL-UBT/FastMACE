# `metadata.json`, schema `fastmace.metadata/1`

Plan task 1b-T10, 19 October 2026. Spec: SPEC §13, §10.6 (restart points and output chunks),
§12.1 (reference record id), owner decision D1 (opt-in gate). Code: `fastmace/provenance.py`.
Tests: `tests/cpu/test_metadata_schema.py`.

Phase 1b writes version 1. Phase 2 ("provenance schema") freezes and validates it.

## How a run writes it

```python
from fastmace import provenance

record = provenance.metadata_record(model=..., checkpoint=loaded, identity=ff.identity, plan=plan,
                                    seed=seed, generators={...}, cursors={...}, gate=sim.gate,
                                    counters={...}, engine_info=engine.info(), restart_points=[...],
                                    setup={...}, timing={...}, report=report, device=device)
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
| `created_utc` | ISO time | `created_utc` (default now) |
| `code` | `version`, `git_commit`, `git_dirty` (§13 "code version") | `fastmace_block` |
| `checkpoint` | `path`, `sha256`, `head`, `licence`, `label`, `trust` | `checkpoint`: a `LoadedModel` (`provenance()`) or a mapping |
| `model` | the 1a `model_block`, unchanged | `model` |
| `force_field_identity` | `ForceFieldIdentity.to_dict()` plus `digest` | `identity` (T05's record or its dict) |
| `reference_record` | the §12.1 record; at least `id` | `reference_record`: a mapping, `to_dict()`, or `.id`; must equal the identity's `reference_record_id` |
| `plan` | active revision: `revision`, `tier`, `reason`, `step`, `description` (`str(plan)`), `backend`, `conv_fusion`, `rewrites`, `engine`, `neighbours`, `polar`, `integrator`, `decisions` (`key`, `choice`, `reason`, `code`, `rule`), `requested`, `auto_options_off`, `superset`, `workload`, `software`, `allowances`, `fallbacks_tried` (`[]` until phase 2), `predicted_memory_bytes` | `plan` (T12's `Plan`) |
| `plan_revisions` | `[{revision, tier, step, reason, description}]`, oldest first, active last (§9.5) | `plan.chain()` and `plan` |
| `precision` | `force`, `integration` (always `"float64"`, §7.3), `tf32`, `float64_energy` (T03, D2) | `plan.precision`; without a plan, the identity's force precision |
| `runtime` | `RuntimeInfo`: `file_locks`, `file_lock_source`, `file_lock_switch_late`, `cuda_driver_cache` (§14) | `fastmace.runtime_info()` |
| `software` | `versions` (Python and packages), `kernels` (Triton, NVRTC, streaming probes), `patches`, `triton` (version), `oeq_extension` (the OEq build variables for a hybrid plan, else `null`) | `fastmace_block` |
| `hardware` | `gpu`, `driver`, `compute_capability`; `null` off CUDA | `device` |
| `random` | `seed`, `generators` (hashes), `cursors` | `seed`, `generators`, `cursors` |
| `integrator` | integrator parameters | `integrator`, default the plan's `IntegratorConfig` |
| `schedule` | the schedule and its position | `schedule`: a mapping or `to_restart()` (T07) |
| `gate` | `status`, `reason`, `selfchecks`, and the rest of the gate record | `gate`; default `{"status": "skipped", "reason": "opt-in: check=\"gate\"", "selfchecks": null}` (D1) |
| `counters` | `recaptures`, `neighbour_lists` (`{name: {rebuilds, build_time_s, skin_A}}` for MACE and D3), `overflows_redone`; `tier_switches` and `mc` always `null` in 1b; `engine` (the engine's `info()`) | `counters`, `engine_info` |
| `restart_points` | `[{step, output_chunk, output_rows, output_frames, ...}]` (T06's indices, §10.6) | `restart_points` |
| `timing` | `setup_s` (`{stage: seconds}`), `setup_total_s`, then the timing summary | `setup`, `timing` |
| `report` | the run report | `report`: T15's `to_metadata()` or a mapping |

Setup stages use §12.5's names: load, rewrites, compilation, capture, gate. 1b-T11 split them
further: load, plan, rewrites, self-checks, model copy, first compiled call, captures. The
record keeps whatever stage names it is given. Values must be seconds ≥ 0.

`counters` takes only the keys listed. The phase-2 and phase-3 keys (`tier_switches`, `mc`)
are refused as inputs, so that 1b cannot fill them by accident.

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
| tier switches; MC counters | `counters.tier_switches`, `counters.mc`: `null` (phases 2 and 3) |
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

## Not in version 1

- **OEq extension build id.** The OEq runtime of 1a-T00 has no build id. The record keeps
  the build variables instead (`OEQ_JIT_EXTENSION`, `OEQ_EXTENSION_SEED` and the others of
  `OEQ_ENVIRONMENT_VARIABLES`), the same ones `model_block` writes under `oeq_wrapper`.
- **Self-check results** are part of the gate record. Until 1b-T16 they are `null`.
- **Validation against a schema file** comes in phase 2.
