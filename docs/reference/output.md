# Output files

This page lists what a run writes, file by file and column by column. The
how-to is [Output, restart and resume](../guide/restart.md).

## `fm.Output`

```python
fm.Output(directory, trajectory_every=20, thermo_every=20, restart_every=10_000)
```

| Parameter | Values | Default | What it does |
| --- | --- | --- | --- |
| `directory` | path (`str` or `os.PathLike`) | required | where the run writes; kept as given, as a string |
| `trajectory_every` | int ≥ 1 or `None` | `20` | steps between trajectory frames; `None` writes no trajectory |
| `thermo_every` | int ≥ 1 | `20` | steps between thermo rows; thermo rows are always written |
| `restart_every` | int ≥ 1 or `None` | `10000` | steps between restart points; `None` writes no restart files on its own |

`fm.md(output=...)` takes a directory or an `fm.Output`. A directory gets
these default intervals. Without `output`, `fm.md` writes to
`./<structure stem>.fastmace/`, or `./fastmace-run/` for an in-memory
`Atoms`. `fm.Simulation(output=None)` writes nothing.

Intervals count the simulation's steps from 0, across `run()` calls and
resumes. A thermo row and, with `trajectory_every`, a frame are written at
step 0 when the run starts, then at every step that is a multiple of the
interval. A restart point is written at every multiple of `restart_every`.

## The output directory

| File | Written by | Content |
| --- | --- | --- |
| `thermo.<n>.csv` | every MD run | thermo rows, one chunk per file |
| `trajectory.<n>.traj` | every MD run | ASE trajectory frames, one chunk per file |
| `restart.npz` | restart points | the complete state at the last restart point |
| `simulation.json` | restart points | the inputs a resume needs |
| `metadata.json` | setup, each `run()`, restart points, stops | plan, gate, report, counters, provenance |
| `.fastmace-handoff/` | a plan that runs in a child process | the hand-off between parent and child; deleted on a clean close |
| `protocol.json` | protocol runs | stages, pieces, status |
| `relaxation.log`, `relaxed.extxyz` | protocol runs with `[relax]` | the initial relaxation |
| `quench/`, `quench-NAME/` | protocol runs with `[quench]`, `fastmace quench` | the quenched frames |

A new run refuses a directory that already holds output chunks
(`FileExistsError`): resume the run or use a new directory.

## Chunks

The thermo table and the trajectory are written in chunks, `n = 0, 1, ...`.
Every restart point closes the current pair of chunk files and opens the
next, so a chunk boundary lies exactly at each restart point. Each chunk is
complete on its own: every thermo chunk starts with the header, and a chunk
without rows or frames still has both files.

```python
from fastmace.io.output import read_frames, read_thermo

header, rows = read_thermo("run/")      # every chunk, in order; rows as strings
frames = read_frames("run/")            # a list of ase.Atoms
ase.io.read("run/trajectory.0.traj", index=":")   # one chunk
```

## Thermo table

`thermo.<n>.csv` is a CSV file with a header line. Energies are per atom.

| Column | Unit | Content |
| --- | --- | --- |
| `step` | | the MD step |
| `time_ps` | ps | `step × timestep_fs / 1000` |
| `phase` | | `heating`, `cooling` or `hold` (the schedule segment), or `constant` without a schedule |
| `target_K` | K | the thermostat target at this step; `nan` for NVE |
| `temperature_K` | K | the kinetic temperature, 2 E_kin / (f k_B) |
| `potential_eV_atom` | eV/atom | potential energy |
| `kinetic_eV_atom` | eV/atom | kinetic energy |
| `total_eV_atom` | eV/atom | potential plus kinetic |
| `volume_A3_atom` | Å³/atom | NPT runs only: the cell volume |
| `pressure_GPa` | GPa | NPT runs only: the instantaneous pressure, kinetic part included |
| `mc_<move>_attempted`, `mc_<move>_accepted` | | runs with MC only: cumulative counts per move type, as of the last MC block |

The degrees of freedom f are 3N − 3 for Langevin NVT with the centre of
mass fixed under the `"mass_weighted"` projection, 3(N − n_fixed) with
`FixAtoms`, and otherwise the integrator's count (3N for ASE's projection,
NPT and `fixcm` off). `Simulation.degrees_of_freedom` gives the value of a
run. A non-finite value stops the run before the row is written.

The NPT columns appear for every integrator with `ensemble` NPT:
LangevinBAOAB, MTK, and Berendsen with a pressure.

## Trajectory frames

`trajectory.<n>.traj` is an ASE trajectory. Each frame holds the step's
cell, positions and momenta, and a single-point calculator with the energy
and forces; variable-cell runs add the stress (Voigt, eV/Å³). After an MC
swap or transmutation, a frame carries the step's species and masses.

## Durable writes

`restart.npz`, `simulation.json` and `metadata.json` are written durably and
atomically: a temporary file next to the target is written, flushed and
fsynced, renamed over the target, and then the directory is fsynced. A
crash leaves the old file or the new one, never a short one. The directory
fsync is skipped on Windows and on file systems that refuse it.

Thermo rows are flushed after each row; frames are on disk once written. At
a restart point, the output queue is drained, the closing chunks and the
directory are fsynced, and only then is the restart file written. So a
restart never counts rows or frames that are not on disk. Rows and frames
after the last restart point are flushed but not fsynced.

Output is asynchronous: a step's values are copied to the host behind the
step and written while the next step runs. The files are the same as with
synchronous writing.

## `restart.npz`

One `.npz` file (plain arrays and JSON text, no pickles), format version 4.
The writer makes the same bytes from the same state.

| Entry | Content |
| --- | --- |
| `step`, `engine` | the step and the writing engine (`"fastmace"`) |
| `positions`, `momenta`, `velocities`, `cell`, `forces`, `energy` | the state at the step |
| `masses`, `species`, `charges` | the current composition, after any MC change |
| `cell_state`, `stress`, `p_eps`, `force_eps` | variable-cell runs: the barostat state |
| `rng_state`, `noise`, `noise_next` | the random generator's state and the batch drawn ahead with its cursor |
| `integrator_keys` | the integrator scheme's own state (NHC, MTK, CSVR) |
| `physical` | total charge, spin, PBC, constraints, fixed atoms |
| `integrator` | class, configuration and thermostat target |
| `protocol` | the schedule and its position |
| `plan`, `plan_revision` | the plan record, with the edge bucket and neighbour capacity |
| `identity`, `identity_digest` | the force-field identity |
| `output_chunk`, `output_rows`, `output_frames` | the last closed chunk, and the rows and frames in chunks 0 to it |
| `mc` | the attached move set: moves, cycle position, counters, stream seeds, displacement step sizes |
| `fastmace_restart` | the format version |

Every restart write is a canonical point: the pipeline is quiesced and the
neighbour lists are rebuilt first, so a resumed engine is in the state of
the uninterrupted one. Restart writes happen only at committed steps,
never inside an MC block.

`restart.npz` is written:

- every `restart_every` steps;
- by `sim.save_restart(directory)` (the file name must be `restart.npz`);
- at the end of each piece of a protocol run;
- when an NPT run stops because the cell diverged.

A run that ends between restart points, for example `fm.md` with `steps` not
a multiple of `restart_every`, has no restart at its last step. Call
`sim.save_restart(...)` on `RunResult.simulation` if you need one.

## `simulation.json`

Written next to every restart: what the restart format does not hold.

| Key | Content |
| --- | --- |
| `simulation_record_version` | `2` |
| `model` | `label`, `path`, `head`, `sha256`, `trust`, `charge`, `spin_multiplicity` |
| `force_field` | `precision`, `species`, `rewrites`, and `dispersion` (the D3 record) when there is one |
| `integrator` | `class` and `fields` of the `fm.integrators` record |
| `workload` | the `fm.Workload` fields |
| `output` | the `fm.Output` fields; `directory` is `"."` when the restart lies in the output directory, so the directory can be moved |
| `seed` | the seed |
| `allow_naive_cueq` | as given |
| `planner_options` | `mode`, `allow_reference`, the fallback, gate, runner, engine, neighbour, adaptation and calibration records |
| `structure` | initial `numbers`, `masses`, `pbc` |
| `plan_revision` | the active plan revision |
| `restart_points` | `[{step, output_chunk, output_rows, output_frames}]` |
| `mc` | the MC record |
| `reference_volume_A3` | NPT only: the starting volume of the cell-divergence check |

## Resume

```python
sim = fm.Simulation.resume("run/", allow_plan_change=False, check="skip")
sim.run()                     # the rest of the schedule, or sim.run(steps=...)
```

`Simulation.resume(directory)` needs `restart.npz` and `simulation.json` in
`directory`. It:

1. loads the model again from `simulation.json`'s file, checked against its
   SHA-256;
2. compares the force-field identity with the restart's. A different
   identity raises `RestartMismatch`; a different MACE or FastMACE version
   alone warns, and is recorded as a new plan revision and in the report;
3. plans again. A plan different from the recorded one needs
   `allow_plan_change=True`, and makes a new revision. A resume never
   calibrates;
4. builds the engine at the restart's positions, cell and composition, then
   loads the state, edge bucket and neighbour capacity;
5. checks the output: every chunk up to `output_chunk` must be present with
   the recorded numbers of rows and frames, and every later chunk must be
   deletable. Otherwise it raises `RestartMismatch` before it touches
   anything;
6. deletes the later chunks (rows and frames written after the restart
   point) and continues with chunk `output_chunk + 1`. Nothing is rewritten.

`check` is as in `fm.Simulation`. The command-line resume,
`fastmace run --resume RUN_DIR`, is for protocol runs only (below).

## `metadata.json`

Schema `fastmace.metadata/1`, minor version 1. A JSON object with these
top-level keys, in this order. It holds no NaN, no infinity and no raw
random states (generator states appear as SHA-256 hashes).

| Key | Content |
| --- | --- |
| `schema`, `schema_minor` | `"fastmace.metadata/1"`, `1`; a record without `schema_minor` is minor 0 |
| `created_utc` | ISO time of the write |
| `code` | `version`, `git_commit`, `git_dirty` |
| `checkpoint` | `path`, `sha256`, `head`, `licence`, `label`, `trust` |
| `model` | the model block |
| `force_field_identity` | the force-field identity and its `digest` |
| `reference_record` | the reference record, with at least `id` |
| `plan` | the active plan: `revision`, `tier`, `reason`, `step`, `description` (`str(plan)`), `description_full` (`plan.describe(full=True)`), `backend`, `conv_fusion`, `rewrites`, `engine`, `neighbours`, `polar`, `integrator`, `decisions` (`key`, `choice`, `reason`, `code`, `rule`), `requested`, `auto_options_off`, `superset`, `workload`, `software`, `allowances`, `fallbacks_tried`, `predicted_memory_bytes`, `runner` (`"in-process"` or `"subprocess"`), `calibration`, `adaptation` |
| `plan_revisions` | every revision, oldest first, the active one last: `revision`, `tier`, `step`, `reason`, `description`, `description_full`, `runner`, `fallback` |
| `precision` | `force`, `integration` (always `"float64"`), `tf32`, `float64_energy` |
| `runtime` | `file_locks` (`"soft"` or `"kernel"`), `file_lock_source`, `file_lock_switch_late`, `cuda_driver_cache` (see [Environment and cache](environment.md#file-locks-on-nfs)) |
| `software` | `versions` (Python and packages), `kernels`, `patches`, `triton`, `oeq_extension` (the OEq build variables for a hybrid plan, else `null`), `treatments` |
| `hardware` | `gpu`, `driver`, `compute_capability`; `null` off CUDA |
| `random` | `seed`, `generators` (SHA-256 of each generator state), `cursors` |
| `integrator` | the integrator parameters |
| `schedule` | the schedule and its position |
| `gate` | `status`, `reason`, `selfchecks`, `device`, `reduced_cell`, `steps`, and the rest of the gate record; `status` `"skipped"` without `check="gate"` |
| `counters` | `recaptures`, `neighbour_lists` (`{name: {rebuilds, build_time_s, skin_A}}`), `overflows_redone`, `mc` (`{move: {attempted, impossible, accepted}}`), `tier_switches`, `segments` (the runtime monitor's last 1,000 segments), `fallbacks` (mid-run fallbacks), `engine` (the engine's `info()`) |
| `restart_points` | `[{step, output_chunk, output_rows, output_frames}]` |
| `timing` | `setup_s` (`{stage: seconds}`), `setup_total_s`, `status`, `steps`, `loop_s` |
| `report` | the run report: `text`, `speed`, `measured_window`, `speed_window`, `setup_s`, `setup_total_s`, `licence`, `gate_status`, `move_energy_status`, `mc_energy_precision`, `reasons`, `warnings`, `mc` |
| `force_field` | the engine's force field record: backend, dtype, rewrites applied with their decisions and self-check errors, `energy_treatment`, `strain_path`, and `d3` with D3 |
| `subprocess` | `null` in-process; for a plan in a child: `python`, `reason`, `directory`, `children`, `calls`, `persistent`, `seconds`, `failed`, `last` |

`timing.status` is `running`, `paused` (after a `run()` call; also after
`fm.md` returns), `complete` (after `sim.close()`), `non-finite`,
`cell-diverged` or `fallback-failed`. The `fastmace.ase` integrators add an
`ase` block.

## `RunResult`

`fm.md` returns a frozen `RunResult`:

| Field | Type | Content |
| --- | --- | --- |
| `summary` | `str` | the one-screen run report |
| `plan` | `Plan` | the plan the run used |
| `gate` | gate record | the gate status and results |
| `simulation` | `fm.Simulation` | the simulation, with its output flushed; it can run on (`run`) or write a restart (`save_restart`) |
| `metadata` | `dict` | `metadata.json` as written at the end of the run |
| `output` | `str` | the output directory |

## Protocol runs

`fastmace run` with `[relax]`, `[quench]` or `--steps` writes, besides the
files above:

| File | Content |
| --- | --- |
| `protocol.json` | the protocol record (below) |
| `relaxation.log` | the initial relaxation's optimizer log |
| `relaxed.extxyz` | the relaxed starting structure |
| `quench/` | the quench (below) |

`protocol.json` keys: `protocol_record_version` (1), `source`, `structure`,
`repeat`, `atoms`, `timestep_fs`, `relax` and `relaxation` (settings and
summary), `quench` and `quench_result`, `mc`, `min_mc_temperature_K`,
`md_steps` (the run's MD steps for a run without a schedule, so that
`--resume` continues to it; `null` with a schedule), `dispersion`, `pieces` (each `first_step`, `last_step`, `wall_s`,
`counters`), `counters` (the sums of `neighbour_rebuilds`, `d3_builds`,
`d3_rebuilds`, `d3_build_s`), and `status`:

| `status` | Meaning | `--resume` |
| --- | --- | --- |
| `relax` | the relaxation is running | no |
| `md` | the MD is running or paused | continues |
| `quench` | the quench is running | continues |
| `complete` | done | reports it is complete |
| `relaxation not converged` | the relaxation stopped | no |
| `cell diverged` | an NPT piece stopped; `stop` holds the reason | no |

## Quench directory

`quench/` (or `quench-NAME/` from `fastmace quench --into NAME`):

| File | Content |
| --- | --- |
| `settings.json` | the resolved settings, written first; an interrupted quench continues only with them |
| `captured.traj` | the picked MD frames |
| `quenched.traj` | their minima |
| `quench.csv` | one row per frame, written after each frame |
| `quench.json` | settings and summary, written last; a quench with it is complete |

`quench.csv` columns: `phase`, `bin`, `frame`, `step`, `time_ps`,
`target_K`, `md_temperature_K`, `md_potential_eV_atom`,
`singlepoint_eV_atom`, `quenched_eV_atom`, `converged`, `optimizer_steps`,
`seconds`, `fmax_eV_A`, `volume_A3_atom`.

## `fastmace relax`

`fastmace relax` writes three files, by default `<stem>.relaxed.extxyz`,
`<stem>.relaxed.json` and `<stem>.relaxed.log` (the `--output` name with
`.json` and `.log`).

| `.json` key | Content |
| --- | --- |
| `structure`, `repeat`, `atoms` | the input |
| `model`, `head`, `precision`, `output` | the model and the output file |
| `optimizer` | the optimizer that ran |
| `fixed_cell` | `false` with `--cell-filter` |
| `fmax_target_eV_A`, `max_steps` | the limits |
| `converged`, `steps`, `seconds` | the result |
| `initial_eV_atom`, `energy_eV_atom` | energy per atom before and after |
| `fmax_eV_A` | the largest force at the end |
| `energies_eV`, `fmax_history_eV_A`, `neighbour_rebuilds`, `recaptures` | `gpu-*` optimizers only |
| `cell_A`, `volume_A3`, `stress_GPa`, `scalar_pressure_GPa` | with `--cell-filter` only |

A relaxation `Simulation` in Python writes no output files and no restart.
