# Planner, gate and fallback

These are the planner's inputs: the keyword arguments of `fm.Simulation`
that choose the tier and the checks, the `fm.Workload` record, and five
configuration records:

- `FallbackConfig`: what happens when a build, compile, capture, self-check
  or gate fails;
- `GateConfig`: where the gate's float64 references may run;
- `RunnerConfig`: the child process that runs a plan this process cannot;
- `AdaptationConfig`: runtime tier switches;
- `CalibrationConfig`: the optional backend measurement before a run.

The [planner guide](../guide/planner.md) explains how to read a plan. This
page lists every option. Engine and neighbour-list options are on
[Engine and neighbour list](engine.md).

```python
import fastmace as fm
from fastmace.config import AdaptationConfig, FallbackConfig, GateConfig, RunnerConfig

model = fm.load("mh1", head="matpes_r2scan")
sim = fm.Simulation(
    atoms, model,
    integrator=fm.integrators.Langevin(temperature_K=600, friction_per_fs=0.02),
    seed=1,
    workload=fm.Workload(max_temperature_K=900),
    check="gate",
    fallback=FallbackConfig(fallback="refuse"),
    gate=GateConfig(cpu_float64_time_limit_s=600),
    runner=RunnerConfig(stall_timeout_s=3600),
    adaptation=AdaptationConfig(adapt="off"),
)
print(sim.plan)        # tier, backend, precision and the decisions that matter
print(sim.gate)        # the gate record
```

`fm.Workload` is in the `fm` namespace. The other records are imported from
`fastmace.config`. Every record is frozen and checks its values when you
build it (`ValueError`). Options the planner cannot honour raise
`PlanRefused` with every reason when the `Simulation` is built, before
anything runs.

`fm.md` takes `mode=`, `allow_reference=`, `check=`, `seed=`, `fallback=`,
`gate=`, `runner=`, `engine=` and `neighbours=` and passes them on
unchanged. It builds the `Workload` from its own arguments (the highest
target temperature of the schedule, and the declared MC moves), and takes
`calibrate=True` instead of a `CalibrationConfig`. It has no `adaptation=`
or `allow_naive_cueq=`; use `fm.Simulation` for those.

The options are recorded in `simulation.json` next to the restart.
`Simulation.resume` plans with the same options.

## Tiers

A tier is the way the engine executes each step. Every tier runs FastMACE's
own integrators, noise, output and restart, so a trajectory has the same
physics on every tier.

| Tier | What runs | When the planner picks it |
| --- | --- | --- |
| `graph` | The compiled model, with each stage of a step recorded once as a CUDA graph and replayed. Edges are padded to buckets so that one graph fits many steps. Fastest. | The default for MD and relaxations. On a CPU device the same engine runs its stages without graphs. |
| `compiled` | The compiled model without CUDA graphs. Exact edge counts; a new edge count does not compile again. Each step is launched from the host. | Without `mode=`, when padded edges would change this model's energies (padding is not exact). With `mode="compiled"`. As the target of a runtime switch. |
| `eager` | The graph tier's backend, rewrites and precision, with no compilation and no CUDA graphs. Exact edge counts. | With `mode="eager"`. When the installed MACE cannot capture this model's readout (some multi-head models), even for `mode="graph"`. As a fallback candidate. |
| `reference` | MACE's own calculator through the versioned reference record: stock uncompiled cuEquivariance model, no rewrites, no padding, MACE's own neighbour list. Slowest. | With `mode="reference"`. As the last fallback candidate, if it fits on the device and `reference_last=True`. |

Host callbacks (`Workload.host_callbacks_every`) keep the graph tier, with a
host synchronisation at each callback: the compiled tier measured 1.3–4.7×
slower with a callback every 1–10 steps (MH-1, hybrid float32, NaCl and
NbCl5 on an RTX 5060 Ti).

## Backends

| Backend | What it is | When `EngineConfig.backend="auto"` picks it |
| --- | --- | --- |
| `hybrid` | cuEquivariance for the symmetric contraction and linear layers, OpenEquivariance kernels for the convolutions (message passing). Most rewrites need it. Float32 by default. | OpenEquivariance can build, the device is CUDA, MACE was not imported before FastMACE prepared the OpenEquivariance runtime, and the process has not compiled a plain cuEq model. |
| `cueq` | Plain cuEquivariance throughout, with its fused convolution unless the plan is reproducible. | Every other case, with the reason in the plan. `reproducible=True` always uses it. |

One process can hold compiled models of only one of the two backends, because
MACE's compile setting is process-wide. A plan of the other backend runs in
a child process through the subprocess runner (`RunnerConfig`). Calibration
measures the two backends against each other on your structure.

## `Simulation` keyword arguments

| Option | Values | Default (`auto` means) | What it does | Notes |
| --- | --- | --- | --- | --- |
| `mode` | `None`, `"graph"`, `"compiled"`, `"eager"`, `"reference"` | `None`: the graph tier, or the compiled tier when padding is not exact for the model | Asks for a [tier](#tiers). | An explicit `"graph"` with inexact padding is refused. An explicit mode also stops runtime tier switches. |
| `check` | `"skip"`, `"gate"`, `"allow_unavailable"` | `"skip"` | Whether the [gate](#the-gate) runs before the first step. `"gate"`: a failed gate goes to the fallback, or raises `GateFailed`; a gate whose reference cannot run raises `GateUnavailable`. `"allow_unavailable"`: as `"gate"`, but an unavailable reference is recorded and the run goes on. | MC runs always run the move-energy part of the gate, whatever `check` is. The record is `sim.gate`. |
| `allow_reference` | `True`, `False` | `False` | Recorded in the plan. | It does not lift any refusal and is not needed for `mode="reference"`. |
| `allow_naive_cueq` | `True`, `False` | `False` | Accepts cuEquivariance's slow fallback kernels when its fused kernels (`cuequivariance_ops_torch`) do not load. | Without it, such a plan is refused with the import error. Recorded. |
| `seed` | int | required | The seed of the integrator's noise. MC streams derive their own seeds from it. | `fm.md(seed=None)` draws a seed and records it in `metadata.json`. |
| `workload` | `fm.Workload` or `None` | `None`: `fm.Workload(max_temperature_K=<integrator's temperature_K>)`; for a relaxation `max_temperature_K=0` | What the run will do; see [`fm.Workload`](#fmworkload). | An integrator without a temperature (NVE) needs an explicit workload. A starting target above `max_temperature_K` is refused. |
| `engine` | `EngineConfig` or `None` | `None`: the default planner's | Engine options; see [Engine and neighbour list](engine.md#engineconfig). | A record you pass replaces the default as a whole. |
| `neighbours` | `NeighbourConfig` or `None` | `None`: the default planner's | Neighbour-list options; see [Engine and neighbour list](engine.md#neighbourconfig). | |
| `fallback` | `FallbackConfig` or `None` | `None`: `FallbackConfig()` | See [`FallbackConfig`](#fallbackconfig). | |
| `gate` | `GateConfig` or `None` | `None`: `GateConfig()` | See [`GateConfig`](#gateconfig). | The record is `sim.gate_config`; the result is `sim.gate`. |
| `runner` | `RunnerConfig` or `None` | `None`: `RunnerConfig()` | See [`RunnerConfig`](#runnerconfig). | |
| `adaptation` | `AdaptationConfig` or `None` | `None`: `AdaptationConfig()` | See [`AdaptationConfig`](#adaptationconfig). | |
| `calibration` | `CalibrationConfig` or `None` | `None`: `CalibrationConfig()` | See [`CalibrationConfig`](#calibrationconfig). | |

## `fm.Workload`

The workload holds the facts about the run that no other object knows.
Periodicity and constraints come from the `Atoms`, the ensemble from the
integrator, and the precision and species superset from the `ForceField`.

| Option | Values | Default (`auto` means) | What it does | Notes |
| --- | --- | --- | --- | --- |
| `max_temperature_K` | float ≥ 0, K | required (`Simulation` fills it from the integrator) | The highest thermostat target of the run. | `Simulation.run` refuses a schedule that goes above it. For MC and species-change runs it sizes the skin and the pair buffers; for plain MD it is recorded. |
| `species_changes` | `True`, `False` | `False` | Declares that species will change (swaps, transmutations, species events). | Required for `"swap"` and `"transmute"` moves. Keeps the graph tier and builds species-dependent tables for the whole superset. Makes the planner size the skin and buffers from the workload. |
| `atom_count_changes` | `True`, `False` | `False` | Declares that atoms will be added or removed. | Accepted by the record and refused by the planner: not supported in 1.0. |
| `min_mc_temperature_K` | float > 0, K, or `None` | `None` | The lowest MC target temperature. Sets the move-energy gate's limit, 0.01 k_B T_min. | Required when `mc_moves` is not empty. Attaching MC moves while the target temperature is below it is refused. |
| `host_callbacks_every` | int > 0 or `None` | `None` | Declares that user Python runs every n steps. | Keeps the graph tier, with a host synchronisation at each callback. The ASE front end sets it from its attached observers. |
| `needs_per_atom` | set of names, for example `{"energies", "charges"}` | empty | Per-atom outputs the run needs. | Recorded in the plan. |
| `mc_moves` | tuple of `"swap"`, `"transmute"`, `"displace"` | `()` | The MC move types the run will attach through `Simulation.run(mc=...)`. | Attaching a move type not declared here is refused. Declaring moves turns on the move-energy gate. |
| `reproducible` | `True`, `False` | `False` | Asks for bitwise reproducible trajectories on the same hardware, driver, software, model, plan, seeds and restart cadence. | Qualified only for plain cuEq with the unfused convolution, the MH-1 architecture, the graph or eager tier and the checked `cuequivariance_ops` version; everything else is refused with the reason. The process is set to deterministic mode before CUDA starts; if CUDA already started, the plan runs in a child process. |
| `mc_energy_precision` | `"auto"`, `"float32"`, `"float64"` | `"auto"`: the force precision if the move-energy gate passes, else float64 | The precision of MC trial energies. MD forces keep their own precision. | `"float32"` with float64 forces is refused. |
| `max_displacement_A` | float > 0, Å, or `None` | `None` | The largest displacement any `Displace` move of the run may use, adaptive steps included. | Required when `"displace"` is in `mc_moves`. Adds 2 × this value to the skin. A `Displace` with a larger `delta_A` is refused. |

## The gate

The gate checks the planned engine on your structure before the first step.
It is opt-in: `check="gate"`.

1. **Copies.** The gate makes 2 copies of the structure, each rattled by
   0.05 Å. For a variable-cell integrator it adds 2 copies strained by up
   to ±2 % and then rattled.
2. **References.** On each copy it evaluates MACE's own model in float64
   (the reference) and in float32 with the energy summed in float64 (the
   yardstick). These run before the production engine is built, or in a
   child process.
3. **Production.** It evaluates each copy through the planned engine, in
   the mode it will run (captured graphs on the graph tier), and restores
   the engine's state bit for bit afterwards.
4. **Rule.** For energy per atom, forces and stress, the production error
   against the float64 reference, maximised over all copies, must be at
   most 2 × the yardstick's error, or at most the floor: 0.01 meV/atom,
   1×10⁻⁵ eV/Å (largest force component), 1×10⁻⁷ eV/Å³ (largest stress
   component; 5×10⁻⁷ eV/Å³ for PolarMACE). Any NaN or infinity fails. NVT
   plans have no stress path and record "stress: not in plan".

**Move energies.** A run with MC moves always runs the move-energy part,
whatever `check` is. It samples trial moves (2 swaps, 1 transmutation,
4 displacements per declared type) and requires every
|ΔE_production − ΔE_reference| ≤ 0.01 k_B T_min, which keeps acceptance
probabilities within about 1 %. With `mc_energy_precision="auto"`, a
failure switches the MC energies to float64.

**Resources.** If the float64 reference runs out of GPU memory, the gate
tries, in order: the CPU in float64 on the full cell (if its estimated time
is within `cpu_float64_time_limit_s` and its memory fits); a reduced cell
on the GPU; the reduced cell on the CPU. A pass on a reduced cell is
recorded as `passed-reduced`. A reference that fails for any reason other
than memory makes the gate unavailable at once.

## `GateConfig`

| Option | Values | Default (`auto` means) | What it does | Notes |
| --- | --- | --- | --- | --- |
| `cpu_float64_time_limit_s` | float ≥ 0, s | `300.0` | After a GPU out-of-memory, the CPU float64 reference is tried only if its estimated time is within this limit. | `0` never tries the CPU. The estimate comes from timing two small boxes cut from your structure, in the same run. |
| `reduced_cell` | `"auto"`, `"off"` | `"auto"`: the largest supercell of the same primitive cell, smaller than the production cell, whose every height is at least `reduced_cell_min_height_cutoffs` × the cutoff | Allows the reference on a reduced cell when the full cell does not fit. | `"off"`: never; the gate is unavailable instead. On a reduced cell the production path is evaluated by a separate uncaptured force field with the same rewrites and precision, and the captured replay is checked against it on the full cell. |
| `reduced_cell_min_height_cutoffs` | float ≥ 1 | `2.0` | The reduced cell's smallest height, in cutoffs. | Below 1 a periodic image would sit inside the cutoff. |

## `FallbackConfig`

The fallback acts when something fails before the first step: the build,
compilation, the first CUDA graph capture, a rewrite self-check, or the
gate. With `fallback="auto"` it searches for the closest plan that passes:

1. **Diagnose.** It runs the failing check again on the eager tier with
   the same rewrites. If eager passes, the CUDA graphs or compilation were
   the problem and the tier changes. If not, it narrows down the rewrite
   set (turning rewrites off in groups, respecting their dependencies),
   each try checked with a reduced gate: one rattled copy, the failing
   quantity only.
2. **Budget.** At most `diagnosis_budget` reduced-gate runs. Each candidate
   that would become active also gets a full gate run, counted apart.
3. **Terminal sequence.** When the budget is spent: the same tier with
   every rewrite off; eager with every rewrite off; the reference tier if
   it fits and `reference_last=True`.
4. **Nothing passes.** `FallbackExhausted` with every candidate tried.

Candidates are built from the pristine model, never by undoing rewrites in
place. A candidate becomes active only after it passed. The physics
(integrator, force precision, structure) never changes. Every candidate is
listed in `sim.plan.fallbacks` and in `metadata.json`. With `check="skip"`
the fallback still validates its candidate against float64 references; the
gate record stays `skipped`.

During `run()`, a capture error at a re-capture or a compile error on the
compiled tier starts the same search from the last committed restart
point. A second such failure before the next restart point stops the run
with `FallbackExhausted`. Out-of-memory errors, non-finite values,
refusals and process constraints never start the fallback.

| Option | Values | Default (`auto` means) | What it does | Notes |
| --- | --- | --- | --- | --- |
| `fallback` | `"auto"`, `"refuse"` | `"auto"`: search for a validated slower plan and record it | What a failure does. `"refuse"` raises the original error, with the record. | No effect on a run where nothing fails. |
| `diagnosis_budget` | int > 0 | `64` | The most reduced-gate runs of one fallback. | Full gate runs (one per candidate that would become active) are not counted. |
| `reference_last` | `True`, `False` | `True` | Allows the reference tier as the last candidate, if it fits on the plan's device: the gate's reference ran there on the full cell, or a trial evaluation runs. | `False` ends the search with `FallbackExhausted` instead. |

## `RunnerConfig`

Some plans cannot run in the current process:

- a compiled model of the other backend after this process compiled one;
- `reproducible=True` after CUDA was started in this process, or a plan
  that is not reproducible in a process that was made reproducible.

The planner then marks the plan `runner="subprocess"`. The `Simulation`
builds nothing in this process; a child process plans, gates and builds,
and the two hand the state over through restart files in
`<output>/.fastmace-handoff/` (or the cache's `handoff/` without output).
`sim.plan`, `sim.gate`, `sim.report` and `metadata.json` reflect the
child. The runner is also used for calibration.

| Option | Values | Default (`auto` means) | What it does | Notes |
| --- | --- | --- | --- | --- |
| `python` | path or `None` | `None`: this process's `sys.executable` | The child's Python interpreter. | The child imports the same FastMACE as the parent. |
| `environment` | mapping or `(name, value)` pairs of strings | `()` | Environment variables set in the child, over the inherited ones. | A mapping is stored as sorted pairs. |
| `inherit_environment` | `True`, `False` | `True` | The child starts from the parent's environment. | `False`: from `environment` alone. |
| `timeout_s` | float > 0, s, or `None` | `None`: no limit | The total time limit of one child run. | A child past it is killed; `SubprocessRunnerError` is raised. |
| `stall_timeout_s` | float > 0, s | `1800.0` | A child with no progress (no committed step, read from its heartbeat file) for this long is killed. | The heartbeat changes when the child's phase or committed step changes. One long setup phase (compilation, the gate's reference) counts as no progress, so keep this above your longest setup phase. |
| `persistent` | `True`, `False` | `True` | One child serves every `run()` of the `Simulation` and keeps its compiled engine. | `False`: a fresh child for each `run()`, resuming from the last hand-off. Windows always uses fresh children. Both write the same restart points. |

## `AdaptationConfig`

During a run the monitor reads the engine's counters at each restart point
and at the end of each `run()`: re-captures and their time, overflows
redone, neighbour-list rebuilds. These are in `metadata.json` and the
report's `runtime` line.

The implemented adaptation is one switch from the graph tier to the
compiled tier. It happens when both:

- the run has made more re-captures than `EngineConfig.recapture_budget`;
- re-capturing took more than `recapture_time_fraction` of the wall time
  since the start (or the last switch).

The switch is made at the next committed step outside an MC block, which
becomes a restart point. The new plan has the same backend, rewrites,
precision and physics; it is a new plan revision, validated before it
becomes active. A resume rebuilds the same revision. A run that never
reaches the trigger is unchanged.

There is no switch with `reproducible=True`, with an explicit `mode=`, for
a relaxation, or on any tier other than graph.

**Refused values.** The record can hold more options than are
implemented. `Simulation` refuses, with `PlanRefused` (code
`adaptation-unsupported`), any record where:

- `trigger` is `"recapture-budget"`;
- `skin_replan` is `True`;
- `max_switches` is `None` or above 1;
- `switch_back_after` is not `None`;
- `min_points_between` is not `None`;
- `reproducible` is `True`.

This holds even with `adapt="off"`. To turn adaptation off, change only
`adapt` (or `trigger="none"`, or `max_switches=0`).

| Option | Values | Default (`auto` means) | What it does | Notes |
| --- | --- | --- | --- | --- |
| `adapt` | `"auto"`, `"off"` | `"auto"`: switch as described above | Whether the monitor may switch tiers. | `"off"`: the monitor still records. |
| `trigger` | `"recapture-time"`, `"none"`; `"recapture-budget"` is refused | `"recapture-time"` | What fires a switch. `"recapture-time"`: budget exceeded and re-capture time above `recapture_time_fraction`. `"none"`: no switch. | |
| `recapture_time_fraction` | float, > 0 and < 1 | `0.2` | The share of wall time spent re-capturing above which an exceeded budget switches tiers. | Measured in the run, so it holds on any GPU. |
| `skin_replan` | `False` only | `False` | Reserved: re-plan the skin from the measured rebuild rate. | `True` is refused. |
| `max_switches` | `0`, `1` | `1` | The most tier switches one run may make. A resume counts the switches already in its plan history. | `0` disables switches. `None` (no limit) and values above 1 are refused. |
| `switch_back_after` | `None` only | `None`: switches are one way | Reserved: go back to the graph tier after this many quiet segments. | Any other value is refused. |
| `min_points_between` | `None` only | `None` | Reserved: a minimum number of restart points between two switches. | Any other value is refused. |
| `reproducible` | `False` only | `False` | Reserved: allow switches under `reproducible=True`. | `True` is refused. |

## `CalibrationConfig`

Calibration is off by default. With `calibrate=True` the run first measures
the two backends on your structure, when the rules rank them equal: the
planner chose `hybrid` because OpenEquivariance builds, and plain cuEq is
also allowed. It does nothing when you set `EngineConfig.backend`, for
reproducible runs, for the reference tier, or when the plan runs in a child
process.

Each measurement is a fresh child process that builds the plan, runs 50
warm-up steps, then times `steps` steps. The blocks alternate hybrid, cuEq,
hybrid, cuEq (2 pairs). Plain cuEq replaces hybrid only when it is faster
by more than `margin` in **every** pair. The four child builds are the
cost; the report shows it as `setup: calibration`. Nothing is stored or
reused by another run. The chosen backend is recorded, so a resume plans
the same one; a resume never calibrates.

`calibrate=True` is refused (code `calibration-workload`) for relaxations
and for runs with MC moves: their step time is not a steady MD step.
`fm.md(calibrate=True)` sets it.

| Option | Values | Default (`auto` means) | What it does | Notes |
| --- | --- | --- | --- | --- |
| `calibrate` | `True`, `False` | `False` | Measures the backends before the run. | MD runs only. |
| `steps` | int ≥ 1 | `50` | Timed steps per block, after the warm-up. | |
| `margin` | float, > 0 and < 1 | `0.05` | How much faster plain cuEq must be, in every pair, to replace hybrid. | The default is above the cross-process timing noise of about 3 %. |
| `choose` | `True` only | `True` | The measurement may change the backend. | `False` (measure and record only) is not implemented: `Simulation` refuses it with code `calibration-unsupported`, even with `calibrate=False`. |

## When to change these

- **Debugging a failure.** Pass `mode="eager"`: same backend, rewrites and
  precision, no compilation and no graphs, so a traceback points at the
  failing operation. `fallback=FallbackConfig(fallback="refuse")` raises
  the original error instead of searching for another plan.
- **New model or new structure.** Run once with `check="gate"`. If the
  gate is unavailable on a large cell and you accept running without it,
  use `check="allow_unavailable"`. Raise `cpu_float64_time_limit_s` if the
  CPU reference is declined for time and you can wait.
- **Gate is slow.** The gate's float64 reference dominates setup on large
  cells. It is a one-off cost per run; leave `check="skip"` for production
  runs of a validated model and structure.
- **MC runs.** Declare every move type in `Workload.mc_moves`, with
  `min_mc_temperature_K` and, for displacements, `max_displacement_A`. Set
  `max_temperature_K` to the highest target of the whole run, not of the
  first segment.
- **Hot or changing structures.** If re-captures are frequent, the graph
  tier loses its advantage. Leave `adaptation` at its default, so the run
  can move to the compiled tier, or plan `mode="compiled"` yourself.
- **Long setups in a child process.** A child that compiles and runs the
  gate on a large cell can take a long time before its first step. Raise
  `stall_timeout_s` if it is killed during setup, and set `timeout_s` on
  shared machines so that a stuck child cannot hold the GPU.
- **Persistent runner.** Keep `persistent=True`: one child keeps its
  compiled engine across `run()` calls. Use `False` only to get a fresh
  process per `run()`, for example to release GPU memory between calls.
- **Comparing backends on one structure.** Use `calibrate=True` (or
  `fm.md(calibrate=True)`) for an MD run. It costs four extra builds, so use
  it for long runs only.
