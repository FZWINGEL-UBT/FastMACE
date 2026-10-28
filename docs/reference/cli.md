# Command line and TOML

`fastmace` is installed as a console script. `python -m fastmace.cli` is the
same command. This page lists every subcommand, every flag and every key of
the `fastmace run` file. [FastMACE in five minutes](../quickstart.md) shows
the common runs.

```text
fastmace [--version] [--debug] COMMAND [--debug] [options]
```

| Command | What it does |
| --- | --- |
| [`doctor`](#fastmace-doctor) | checks the installation; changes nothing |
| [`models`](#fastmace-models) | lists the model labels, heads, files, SHA-256 and licences |
| [`md`](#fastmace-md) | constant-temperature or NVE MD in one command (`fm.md`) |
| [`init`](#fastmace-init) | writes a commented TOML of `fm.md`'s arguments |
| [`run`](#fastmace-run) | runs MD or a protocol from a TOML (or YAML) file |
| [`relax`](#fastmace-relax) | relaxes one structure, at fixed cell or with a cell filter |
| [`quench`](#fastmace-quench) | quenches the trajectory frames of a finished run |

## Global options and exit codes

| Flag | Values | Default | What it does |
| --- | --- | --- | --- |
| `--version` | | | prints `fastmace <version>` and exits 0 |
| `--debug` | | off | prints the traceback of a FastMACE error instead of its message alone; a `ValueError` raised inside `fm.md` or a protocol stage then shows its traceback instead of exit code 2 |
| `-h`, `--help` | | | the help of `fastmace` or of one command |

`--debug` may stand before or after the command: `fastmace --debug md ...`
and `fastmace md --debug ...` are the same.

| Exit code | Meaning |
| --- | --- |
| `0` | success |
| `1` | a FastMACE error (a refused plan, a failed gate, a missing model file, a restart that does not match, ...), a required `doctor` check failed, a relaxation did not converge, or a protocol stage stopped |
| `2` | a usage error: an unknown flag or value (argparse), a flag combination with no `fm.md` meaning, an unknown or missing TOML key, or no command at all (then the help is printed) |

A FastMACE error is printed as its message alone: what happened, why, what
to do next, and where the record is. Any other exception is a bug and keeps
its traceback.

## `fastmace doctor`

```text
fastmace doctor [--no-hash]
```

| Flag | Values | Default | What it does |
| --- | --- | --- | --- |
| `--no-hash` | | off | lists the model files without computing their SHA-256 (hashing takes seconds per file) |

Prints one line per check, `[ ok ]`, `[FAIL]` (a required check) or
`[warn]` (an optional one), and for each failed item the command or page
that fixes it. Exit code 1 if a required check failed, else 0. The checks
are listed item by item on [Environment and cache](environment.md#fastmace-doctor).
`doctor` changes nothing: it creates no directory, imports neither MACE nor
OpenEquivariance, and deletes no lock file.

## `fastmace models`

```text
fastmace models
```

No flags. Prints the label table (label, head, licence, released file name,
SHA-256), the licence texts, and the aliases that take a `head`. It reads
only the table: it looks at no file and downloads nothing. See
[Models and licences](../guide/models.md).

## `fastmace md`

```text
fastmace md --model MODEL --structure FILE [options]
```

Each flag is one `fm.md` argument. A flag that is not given is left out, so
`fm.md`'s default applies; the command has no defaults of its own. It prints
the run report and the output directory, and exits 0.

**Model flags**

| Flag | Values | Default | `fm.md` argument | What it does |
| --- | --- | --- | --- | --- |
| `--model` | label, path or URL | required | `model` | the model; labels are listed by `fastmace models` |
| `--head` | head name | none | `head` | the head of a multi-head model; required for MH-1 |
| `--charge` | int | the model's | `charge` | total charge, for models that take it |
| `--spin-multiplicity` | int | the model's | `spin_multiplicity` | spin multiplicity, for models that take it |
| `--sha256` | 64 hex digits | none | `sha256` | the expected SHA-256 of a model file outside the label table |
| `--trusted` | | off | `trusted=True` | trust a model file without a known SHA-256; it is unpickled |

**Run flags**

| Flag | Values | Default | `fm.md` argument | What it does |
| --- | --- | --- | --- | --- |
| `--structure` | file | required | `structure` | any format ASE reads |
| `--repeat NA NB NC` | three ints | `1 1 1` | `repeat` | supercell repeats; the cell is never expanded silently |
| `--ensemble` | `NVT`, `NPT`, `NVE` (any case) | `NVT` | `ensemble` | `NVT` is Langevin, `NPT` LangevinBAOAB, `NVE` velocity Verlet from the structure's momenta |
| `--temperature K` | float, K | none | `temperature_K` | the target temperature; required for NVT and NPT, refused for NVE |
| `--pressure GPA` | float, GPa | `0` | `pressure_GPa` | NPT only |
| `--cell-mode` | `hydrostatic`, `full` | `hydrostatic` | `cell_mode` | NPT only |
| `--timestep FS` | float, fs | `0.5` | `timestep_fs` | the MD timestep |
| `--steps` | int | required | `steps` | the number of MD steps |
| `--dispersion` | `none`, `d3bj` | `none` | `dispersion` | `d3bj` adds D3(BJ), with the parameters of the head's training functional from the label table |
| `--d3-functional NAME` | parameter set, e.g. `pbe`, `r2scan` | none | `dispersion=fm.D3(functional=NAME)` | the D3 parameter set; needs `--dispersion d3bj` |
| `--seed` | int ≥ 0 | drawn | `seed` | the noise seed; a drawn seed is recorded in `metadata.json` |
| `--check` | `skip`, `gate`, `allow_unavailable` | `skip` | `check` | `gate` runs the gate before the first step; `allow_unavailable` runs it but goes on if no reference can run |
| `--output DIR` | directory | `./<structure stem>.fastmace/` | `output` | the output directory, with `fm.Output`'s default intervals |

The Langevin friction is `fm.md`'s, 0.02 per ASE time unit (about
0.00196 per fs). NVE draws no velocities: a structure file without momenta
starts at rest.

Schedules, MC moves, explicit integrators, a full D3 record, output
intervals and planner options have no flag. Use `fastmace init` and
`fastmace run` for them.

`fm.md` does not download models: a label's file must be in
`$FASTMACE_MODELS` or `<cache root>/models` already. Fetch it once from
Python with `fm.load(label, download=True)` (see
[Environment and cache](environment.md)).

## `fastmace init`

```text
fastmace init [fastmace md's flags] > run.toml
```

Takes every flag of `fastmace md`, all optional, and prints a TOML file of
`fm.md`'s arguments for `fastmace run`. Every argument appears under its own
name: the ones given as set lines, the rest as commented lines
(`# key = value`) with what they do and an example. The record tables
(schedule segments, integrator, D3, MC, output, planner options) follow as
commented examples.

- Without flags, the set lines are an example run: `structure = "NaCl.cif"`,
  `model = "mh1"`, `head = "matpes_r2scan"`, `ensemble = "NVT"`,
  `temperature_K = 300.0`, `steps = 1000000`, `output = "run/"`.
- With flags, the set lines are exactly those flags, so
  `fastmace init <flags> > run.toml; fastmace run run.toml` is
  `fastmace md <flags>`.
- `--dispersion d3bj --d3-functional NAME` is written as a `[dispersion]`
  table with the record's numbers.

Exit code 2 for `--d3-functional` without `--dispersion d3bj`, else 0.

## `fastmace run`

```text
fastmace run CONFIG.toml [--steps N] [--output DIR] [--calibrate]
fastmace run --resume RUN_DIR [--steps N]
```

| Argument or flag | Values | Default | What it does |
| --- | --- | --- | --- |
| `CONFIG.toml` | a file, `-`, or a shipped protocol name | | the run file; `-` reads standard input (as TOML); a `.yaml` or `.yml` file is read as YAML |
| `--resume RUN_DIR` | directory | | continues the protocol run in `RUN_DIR`; not together with `CONFIG.toml` |
| `--steps N` | int ≥ 1 | the whole run | runs at most `N` MD steps, writes a restart and stops; makes the run a protocol run (below). A `steps` key in the file is the run's length; `--steps` shortens this piece of it |
| `--output DIR` | directory | the file's `output` | the output directory, in place of the file's (also replaces `directory` in an `[output]` table) |
| `--calibrate` | | off | the file's `calibrate = true`: times hybrid against plain cuEquivariance on this structure before the run; refused with `--resume` |

Paths in the file (the structure, the output directory) are taken relative
to the directory `fastmace run` starts in, except for a shipped protocol,
whose relative structure path is a file shipped with it. A key that is not an
`fm.md` argument is refused, naming it (exit code 2). `structure` and
`model` are required.

A YAML file needs PyYAML and is read with `safe_load` only. It takes the
same keys as TOML.

### Plain runs and protocol runs

A file without `[relax]` or `[quench]` tables, run without `--steps`, is a
plain run: `fm.md(structure, **keys)`, exactly as `fastmace md`. It prints
the report and exits.

A file with a `[relax]` or `[quench]` table, or any run with `--steps`, is a
**protocol run**, in stages in the output directory:

1. `[relax]`: an initial relaxation at fixed cell (`relaxation.log`,
   `relaxed.extxyz`). If it does not converge, the run stops with exit code 1.
2. `fm.md` on the relaxed structure, for at most `--steps` steps of the
   schedule (all of it without `--steps`), then a restart at the last step.
   If the schedule is not finished, the run pauses and prints the
   `fastmace run --resume` command.
3. `[quench]`, once the schedule is done: trajectory frames picked per
   target-temperature bin, each relaxed to 0 K (`quench/`).

`protocol.json` in the output directory records the stages, the pieces and
the status (see [Output files](output.md#protocol-runs)). A directory that
already holds a protocol run is refused; continue it with `--resume`.
`fastmace run --resume RUN_DIR [--steps N]` continues for at most `N` more
steps, then the quench. A complete run prints that it is complete and exits
0. A run stopped by a failed relaxation or a diverged cell cannot be resumed.

`--resume` continues protocol runs only. A plain run is continued from
Python with `fm.Simulation.resume(RUN_DIR)` (see
[Output, restart and resume](../guide/restart.md)).

The shipped protocols are named by their file stem:
`npt_anneal_NbCl5`, `npt_anneal_NbCl5_short`, `swap_anneal_NaNbOCl4` and
`swap_anneal_NaNbOCl4_short`. `fastmace run swap_anneal_NaNbOCl4_short`
runs one; a file of the same name in the current directory wins.

### Top-level keys

The keys are `fm.md`'s arguments under their own names. See
[`fm.md`](../api/quick.md) for the Python side.

| Key | Values | Default | What it does |
| --- | --- | --- | --- |
| `structure` | path | required | the structure, any format ASE reads |
| `model` | label, path or URL | required | the model |
| `head` | string | none | head of a multi-head model; required for MH-1 |
| `charge` | int | the model's | total charge, for models that take it |
| `spin_multiplicity` | int | the model's | spin multiplicity, for models that take it |
| `sha256` | string | none | expected SHA-256 of a model file outside the label table |
| `trusted` | bool | `false` | trust a model file without a known SHA-256 (it is unpickled) |
| `ensemble` | `"NVT"`, `"NPT"`, `"NVE"` | `"NVT"` | Langevin, LangevinBAOAB or velocity Verlet; ignored in favour of `[integrator]` when one is given (a conflicting non-default value is refused) |
| `temperature_K` | float, K | none | the target temperature; or a schedule, not both |
| `schedule` | profile name or `[[schedule]]` segments | none | a temperature schedule (below) |
| `pressure_GPa` | float, GPa | `0.0` | NPT only |
| `cell_mode` | `"hydrostatic"`, `"full"` | `"hydrostatic"` | NPT only |
| `timestep_fs` | float, fs | `0.5` | the timestep |
| `steps` | int | the whole schedule | required without a schedule; with one, at most this many steps of it |
| `mc` | `[mc]` or `[[mc]]` tables | none | MC moves (below); experimental |
| `min_mc_temperature_K` | float, K | the start temperature | MC runs only where the target is at or above it; required with a schedule and MC |
| `dispersion` | `"none"`, `"d3bj"` or a `[dispersion]` table | none | D3(BJ): `"d3bj"` takes the head's training functional |
| `repeat` | array of three ints | `[1, 1, 1]` | supercell repeats |
| `output` | directory or an `[output]` table | `"./<structure stem>.fastmace/"` | the output directory and intervals |
| `seed` | int ≥ 0 | drawn | the noise seed, recorded in `metadata.json` |
| `check` | `"skip"`, `"gate"`, `"allow_unavailable"` | `"skip"` | the gate; opt-in |
| `integrator` | an `[integrator]` table | none | an explicit integrator (below); it wins over `ensemble` |
| `mode` | `"graph"`, `"compiled"`, `"eager"`, `"reference"` | unset | the execution tier; unset lets the planner choose |
| `allow_reference` | bool | `false` | allows the reference tier where a hard rule needs it |
| `calibrate` | bool | `false` | times hybrid against plain cuEquivariance first and takes plain cuEq only if it is more than 5 % faster; MD only, refused with MC |
| `fallback` | a `[fallback]` table | defaults | the fallback options |
| `gate` | a `[gate]` table | defaults | where the gate's references may run |
| `runner` | a `[runner]` table | defaults | the subprocess runner |
| `engine` | an `[engine]` table | defaults | backend and engine options |
| `neighbours` | a `[neighbours]` table | defaults | the neighbour list |

With an explicit `[integrator]`, `temperature_K`, `timestep_fs`,
`pressure_GPa` and `cell_mode` must be left out or equal the record's
fields; set them in the table.

### `schedule`

Either a profile name or an array of segments, not both.

| Profile | Schedule |
| --- | --- |
| `"slow"` | 0 → 400 K in 30 ps, 400 → 0 K in 240 ps |
| `"fast"` | 0 → 400 K in 3 ps, 400 → 0 K in 24 ps |
| `"ultrafast"` | 0 → 400 K in 1 ps, 400 → 0 K in 8 ps |
| `"anneal"` | 0 → 1000 K in 20 ps, 1000 → 250 K in 250 ps |

In a run file a profile name sets the schedule only. The profile's output
intervals, relaxation and quench settings are not applied; set them in
`[output]`, `[relax]` and `[quench]`.

Each `[[schedule]]` table is one segment; segments follow each other without
a temperature jump.

| Key | Segment | Values | What it does |
| --- | --- | --- | --- |
| `kind` | both | `"ramp"`, `"hold"` | the segment type |
| `start_K`, `end_K` | ramp | float, K | the ramp's end points |
| `ps` | ramp or hold | float > 0, ps | the duration; a ramp takes `ps` or `rate_K_per_ps`, not both |
| `rate_K_per_ps` | ramp | float, K/ps | the rate; its sign must follow the ramp's direction |
| `temperature_K` | hold | float, K | the held temperature |

Durations are rounded half up to whole steps. On a ramp the target moves
every step.

### `[integrator]`

`type` names an `fm.integrators` record; the other keys are that record's
fields. Time constants have no defaults. For Langevin and LangevinBAOAB,
`friction_per_fs` defaults to `fm.md`'s (0.02 per ASE time unit). See
[Integrators and optimisers](integrators.md) for what each field does.

| `type` | Required fields | Optional fields (default) |
| --- | --- | --- |
| `"Langevin"` | `temperature_K` | `timestep_fs` (0.5), `friction_per_fs`, `fixcm` (`"auto"`), `fixcm_projection` (`"auto"`) |
| `"LangevinBAOAB"` | `temperature_K` | `timestep_fs` (0.5), `friction_per_fs`, `pressure_GPa` (0.0), `external_stress_GPa` (none), `cell_mode` (`"hydrostatic"`), `barostat_tau_fs` (ASE's 20 T_tau), `barostat_mass_factor` (1.0), `disable_cell_langevin` (false), `fixcm` (true), `max_volume_ratio` (20.0), `max_cell_length_A` (1000.0) |
| `"NVE"` | `timestep_fs` | `fixcm` (`"auto"`), `dof` (`"auto"`) |
| `"CSVR"` | `timestep_fs`, `temperature_K`, `tau_fs` | `fixcm` (`"auto"`), `dof` (`"auto"`) |
| `"NoseHooverChain"` | `timestep_fs`, `temperature_K`, `tdamp_fs` | `tchain` (3), `tloop` (1), `fixcm` (true), `dof` (`"auto"`) |
| `"MTK"` | `timestep_fs`, `temperature_K`, `pressure_GPa`, `tdamp_fs`, `pdamp_fs` | `tchain` (3), `pchain` (3), `tloop` (1), `ploop` (1), `cell_mode` (`"hydrostatic"`), `fixcm` (true), `dof` (`"auto"`) |
| `"Berendsen"` | `timestep_fs`, `temperature_K`, `taut_fs` | `pressure_GPa`, then `taup_fs` and `compressibility_per_GPa` (required with a pressure), `mask` (three 0/1 flags), `fixcm` (`"auto"`), `dof` (`"auto"`) |

MTK is experimental. Berendsen is not canonical: use it for equilibration
only.

### `dispersion`

`"none"` (the default), `"d3bj"`, or a `[dispersion]` table with the fields
of `fm.D3`:

| Key | Values | Default | What it does |
| --- | --- | --- | --- |
| `functional` | string | required | the parameter set, e.g. `"pbe"`, `"r2scan"`; should be the model's training functional |
| `damping` | `"bj"` | `"bj"` | Becke-Johnson damping |
| `pair_cutoff_bohr` | float | `40.0` | pair cutoff, bohr |
| `cn_cutoff_bohr` | float | `40.0` | coordination-number cutoff, bohr |
| `three_body` | bool | `false` | the ATM term; `true` is refused |
| `smoothing` | `"none"` | `"none"` | cutoff smoothing |
| `precision` | `"float64"` | `"float64"` | precision of the reductions |
| `autoang`, `autoev` | float | `0.52917726`, `27.21138505` | the unit constants |
| `s6`, `s8`, `a1`, `a2` | float | the functional's set | give all four or none |

See [Models, D3, PolarMACE and schedules](models.md).

### `mc`

One `[mc]` table or several `[[mc]]` tables, each an `fm.mc` record. `move`
picks the record. MC is experimental. See [Monte Carlo](mc.md).

| `move` | Fields (default) |
| --- | --- |
| `"swap"` | `pairs` (array of species pairs, required), `every` (required), `attempts` (required), `velocities` (`"exchange"`; or `"rescale"`, `"redraw"`, `"redraw_all"`) |
| `"transmute"` | `species` (at least two, required), `every`, `attempts` (required), `velocities` (`"rescale"`, or `"redraw_all"` when the integrator fixes the centre of mass; or `"redraw"`), `mu` (table species → eV), `formal_charges` (table species → int) |
| `"displace"` | `delta_A` (float, Å, required), `every`, `attempts` (required), `adaptive` (false) |

Species are symbols or atomic numbers. `every` is in MD steps.

### `[output]`

The fields of `fm.Output`. A missing `directory` is `fm.md`'s default,
`./<structure stem>.fastmace/`.

| Key | Values | Default | What it does |
| --- | --- | --- | --- |
| `directory` | path | `./<structure stem>.fastmace/` | the output directory |
| `trajectory_every` | int ≥ 1 | `20` | steps between trajectory frames |
| `thermo_every` | int ≥ 1 | `20` | steps between thermo rows |
| `restart_every` | int ≥ 1 | `10000` | steps between restart points |

TOML has no null, so `trajectory_every` and `restart_every` cannot be
switched off from a TOML file; in a YAML file `null` switches them off. See
[Output files](output.md).

### `[relax]` and `[quench]`

These two tables make a protocol run. They are not `fm.md` arguments.

| Table | Key | Values | Default | What it does |
| --- | --- | --- | --- | --- |
| `[relax]` | `fmax` | float > 0, eV/Å | required | converged when every force is below it |
| | `optimizer` | an optimizer name | `"gpu-LBFGS"` | see [`fastmace relax`](#fastmace-relax) |
| | `max_steps` | int ≥ 1 | `5000` | at most this many steps |
| `[quench]` | `min_K` | float, K | required | the lowest target temperature whose frames are picked |
| | `bin_K` | float > 0, K | required | the width of a target-temperature bin |
| | `per_bin` | int ≥ 1 | required | frames per bin and ramp |
| | `fmax` | float > 0, eV/Å | required | the quench's convergence |
| | `optimizer` | an optimizer name | required | the quench's optimizer |
| | `max_steps` | int ≥ 1 | required | at most this many steps per frame |

`[quench]` needs all six keys, a schedule and trajectory frames. The
relaxation keeps the cell fixed; each quenched frame keeps its own cell.

### Planner tables

`[engine]`, `[neighbours]`, `[fallback]`, `[gate]` and `[runner]` take the
fields of `fastmace.config.EngineConfig`, `NeighbourConfig`,
`FallbackConfig`, `GateConfig` and `RunnerConfig`. An unknown field is
refused, naming the known ones. The common ones:

| Table | Key | Values | Default | What it does |
| --- | --- | --- | --- | --- |
| `[engine]` | `backend` | `"auto"`, `"hybrid"`, `"cueq"` | `"auto"` | hybrid OEq/cuEq or plain cuEq |
| | `compile_mode` | `"default"`, `"none"` | `"default"` | `torch.compile` of the model |
| | `device` | device string | `"cuda:0"` | the GPU |
| `[neighbours]` | `skin` | float, Å, or `"auto"` | `"auto"` | the neighbour-list skin |
| | `d3_skin` | float, Å, or `"auto"` | `"auto"` | the D3 pair list's skin |
| `[fallback]` | `fallback` | `"auto"`, `"refuse"` | `"auto"` | fall back to a validated slower plan, or raise |
| `[gate]` | `cpu_float64_time_limit_s` | float, s | `300.0` | CPU float64 references only within this time |
| `[runner]` | `stall_timeout_s` | float, s | `1800.0` | a child without progress for this long is killed |

The full field lists are on [Engine and neighbour list](engine.md) and
[Planner, gate and fallback](planner.md). An array becomes a tuple; a
`[runner.environment]` table is a mapping of variable names to values.

### A complete example

```toml
# An NPT anneal with D3, swap MC, a relaxation and a quench.
structure = "NaNbOCl4.cif"
model = "mh1"
head = "matpes_r2scan"
repeat = [3, 3, 3]
ensemble = "NPT"
pressure_GPa = 0.0
cell_mode = "hydrostatic"
timestep_fs = 1.0
min_mc_temperature_K = 250.0     # MC only where the target is >= 250 K
dispersion = "d3bj"              # D3(BJ) of the head's functional (r2scan)
seed = 1
check = "skip"

# 0 -> 1000 K in 2 ps, then down to 250 K at -30 K/ps
[[schedule]]
kind = "ramp"
start_K = 0.0
end_K = 1000.0
ps = 2.0

[[schedule]]
kind = "ramp"
start_K = 1000.0
end_K = 250.0
rate_K_per_ps = -30.0

[[mc]]
move = "swap"
pairs = [["O", "Cl"]]
every = 100                      # a block every 100 MD steps
attempts = 50
velocities = "exchange"

[output]
directory = "anneal/"
trajectory_every = 10
thermo_every = 10
restart_every = 10000

[engine]
backend = "hybrid"

[relax]                          # before the MD, at fixed cell
fmax = 0.01
optimizer = "gpu-LBFGS"
max_steps = 5000

[quench]                         # after the MD: 2 frames per 100 K bin from 500 K
min_K = 500.0
bin_K = 100.0
per_bin = 2
fmax = 0.01
optimizer = "gpu-LBFGS"
max_steps = 5000
```

```bash
fastmace run anneal.toml --steps 100000     # first piece
fastmace run --resume anneal/ --steps 100000
```

## `fastmace relax`

```text
fastmace relax STRUCTURE --model MODEL [options]
```

| Flag | Values | Default | What it does |
| --- | --- | --- | --- |
| `STRUCTURE` | file | required | any format ASE reads |
| `--model` | label, path or URL | required | as `fastmace md` |
| `--head`, `--charge`, `--spin-multiplicity`, `--sha256`, `--trusted` | | | as `fastmace md` |
| `--precision` | `float32`, `float64` | `float32` | the force-field precision |
| `--dispersion`, `--d3-functional` | | none | as `fastmace md` |
| `--optimizer` | `gpu-LBFGS`, `gpu-FIRE`, `ase-LBFGS`, `ase-BFGS`, `LBFGS`, `BFGS` | `gpu-LBFGS` | `gpu-*` run on the graph engine; `ase-*` are ASE's on `fm.calc`; bare `LBFGS` and `BFGS` mean ASE's |
| `--fmax EV_A` | float > 0, eV/Å | `0.01` | converged when every force is below it |
| `--steps` | int ≥ 1 | `5000` | at most this many optimizer steps |
| `--repeat NA NB NC` | three ints ≥ 1 | `1 1 1` | supercell repeats |
| `--cell-filter` | `frechet` | none (fixed cell) | relaxes the cell too, with ASE's `FrechetCellFilter` (its device port for `gpu-*`) |
| `--pressure GPA` | float, GPa | `0` | the cell filter's scalar pressure; needs `--cell-filter` |
| `--output FILE` | file | `<structure stem>.relaxed.extxyz` | the relaxed structure; the `.json` summary and `.log` go next to it |

The Python side is `fm.Simulation(..., optimizer=fm.optimize.LBFGS())` and
`sim.relax(fmax=...)`; see [Relaxations and quenches](../guide/relax.md).
The files are described on [Output files](output.md#fastmace-relax).
A relaxation that does not reach `--fmax` within `--steps` still writes its
files and exits 1.

## `fastmace quench`

```text
fastmace quench RUN_DIR [options]
```

| Flag | Values | Default | What it does |
| --- | --- | --- | --- |
| `RUN_DIR` | directory | required | a finished run with a schedule, trajectory frames, `restart.npz` and `simulation.json` |
| `--into NAME` | letters, digits, `._-` | none | writes `RUN_DIR/quench-NAME/` instead of `RUN_DIR/quench/` |
| `--optimizer` | as `fastmace relax` | from the settings | the optimizer |
| `--fmax EV_A` | float, eV/Å | from the settings | converged when every force is below it |
| `--max-steps` | int | from the settings | at most this many steps per frame |
| `--min-K` | float, K | from the settings | the lowest target temperature whose frames are picked |
| `--bin-K` | float, K | from the settings | the width of a target-temperature bin |
| `--per-bin` | int | from the settings | frames per bin and ramp |

The settings are the run's `[quench]` table for a protocol run, else the
`anneal` profile's: `min_K` 500, `bin_K` 100, `per_bin` 2, `fmax` 0.01,
`optimizer` `gpu-LBFGS`, `max_steps` 5000. Each flag replaces one value.

The quench uses the run's model (checked by SHA-256), precision, D3 record
and engine options. Frames are picked by the schedule's target temperature,
on each ramp separately, evenly spaced within each bin, plus the final
frame; each is relaxed at its own cell. A complete quench (its
`quench.json` exists) only prints its summary. An interrupted one continues
with the settings it started with.
