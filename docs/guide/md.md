# Running MD

This page covers the choices every MD run makes: the ensemble and
integrator, the timestep, the friction, the seed, how long the run lasts and
how often it writes. It uses `fm.md` for one-call runs and `fm.Simulation`
when you want to keep the simulation and run it in pieces.

## Two ways in

`fm.md` builds and runs a simulation in one call and returns a `RunResult`:

```python
import fastmace as fm

run = fm.md("NaCl.cif", model="mh1", head="matpes_r2scan", ensemble="NVT",
            temperature_K=300, timestep_fs=0.5, steps=100_000, repeat=(3, 3, 3),
            output="run/", seed=1)
print(run.summary)
```

`fm.Simulation` takes the pieces explicitly: the structure, a force field
or model, an integrator record, a workload, an output record and a seed.
`fm.md` builds exactly this:

```python
from ase.io import read

import fastmace as fm

atoms = read("NaCl.cif").repeat((3, 3, 3))
model = fm.load("mh1", head="matpes_r2scan")
sim = fm.Simulation(
    atoms, model,
    integrator=fm.integrators.Langevin(timestep_fs=0.5, temperature_K=300, friction_per_fs=0.002),
    output=fm.Output("run/"),
    seed=1,
)
sim.run(steps=100_000)
sim.run(steps=100_000)          # 100,000 more steps, same engine
sim.flush_output()
print(sim.report.text())
sim.close()
```

`Simulation` copies `atoms`; your object is never changed. It needs
periodic boundaries in all three directions. It can also be used as a
context manager (`with fm.Simulation(...) as sim:`), which closes it at the
end.

## Choosing an ensemble

`fm.md`'s `ensemble=` and `fastmace md --ensemble` pick one of three
integrators:

| `ensemble=` | Integrator | ASE 3.27 class | Needs |
| --- | --- | --- | --- |
| `"NVT"` (default) | `fm.integrators.Langevin` | `Langevin` | `temperature_K` or `schedule` |
| `"NPT"` | `fm.integrators.LangevinBAOAB` | `LangevinBAOAB` | as NVT, plus `pressure_GPa` (default 0) and `cell_mode` (`"hydrostatic"` or `"full"`) |
| `"NVE"` | `fm.integrators.NVE` (velocity Verlet) | `VelocityVerlet` | no temperature; refuses `temperature_K`, `schedule` and `mc` |

`pressure_GPa` and `cell_mode` are for NPT only; `fm.md` refuses them with
another ensemble.

## Choosing an integrator

For any other thermostat or barostat, pass an integrator record. It decides
the ensemble, so leave out `ensemble=`, and set the temperature and
timestep in the record only:

```python
run = fm.md("NaCl.cif", model="mh1", head="matpes_r2scan", steps=100_000, repeat=(3, 3, 3),
            integrator=fm.integrators.CSVR(timestep_fs=1.0, temperature_K=300, tau_fs=100.0),
            output="csvr/", seed=1)
```

| Record | Ensemble | ASE 3.27 class | Required fields besides `timestep_fs` and `temperature_K` |
| --- | --- | --- | --- |
| `Langevin` | NVT | `Langevin` | `friction_per_fs` |
| `LangevinBAOAB` | NPT | `LangevinBAOAB` | `friction_per_fs`; `pressure_GPa` defaults to 0 |
| `NVE` | NVE | `VelocityVerlet` | none (no `temperature_K`) |
| `CSVR` | NVT | `Bussi` | `tau_fs` |
| `NoseHooverChain` | NVT | `NoseHooverChainNVT` | `tdamp_fs` |
| `MTK` | NPT | `IsotropicMTKNPT` (`cell_mode="hydrostatic"`), `MTKNPT` (`"full"`) | `pressure_GPa`, `tdamp_fs`, `pdamp_fs` |
| `Berendsen` | NVT, or NPT with `pressure_GPa` | `NVTBerendsen`, `NPTBerendsen`, `Inhomogeneous_NPTBerendsen` (`mask`) | `taut_fs`; with a pressure also `taup_fs` and `compressibility_per_GPa` |

All times are in fs, pressures in GPa and compressibilities in 1/GPa. The
other parameters keep ASE 3.27's names and defaults; for example
`NoseHooverChain` has `tchain=3` and `tloop=1`. Records are frozen
dataclasses, so a typo in a field name is an error.

Some choices are flagged in the run report:

- **NPT** (`LangevinBAOAB`, `MTK`, and `Berendsen` with a pressure) carries
  an "experimental: not yet qualified" line until its qualification study
  is recorded.
- **Berendsen** is non-canonical. Use it for equilibration only; the report
  says so.

The same records go into a TOML file as an `[integrator]` table with
`type = "CSVR"` and the record's fields (see
[Command line and TOML](../reference/cli.md)).

## Timestep

`timestep_fs` is in fs. `fm.md`'s default is 0.5 fs. The shipped anneal and
melt-quench examples use 1.0 fs. With an explicit integrator record, the
record's `timestep_fs` is the timestep, and `fm.md` refuses a different
`timestep_fs=` argument.

## Friction

The Langevin records take `friction_per_fs`, in 1/fs. `fm.md`,
`fastmace md` and a TOML `[integrator]` table without the field use 0.02
per ASE time unit, about 0.002 per fs (0.001965/fs).

The `fastmace.ase` classes take ASE's `friction` in ASE's inverse time
units, as ASE does: `friction=0.02 / units.fs` is 0.02 per fs, ten times
`fm.md`'s default.

For `LangevinBAOAB`, ASE's thermostat time `T_tau` is `1 / friction`, and
the barostat time `barostat_tau_fs` defaults to 20 `T_tau`.

`sim.set_friction(friction_per_fs)` changes the friction during a run. It
re-captures the CUDA graphs once, and the change is recorded.

## Starting velocities

By default FastMACE draws no starting velocities. A run starts from the
momenta stored on the structure; most structure files have none, so the
atoms start at rest and a Langevin thermostat heats them. CSVR cannot start
from rest (it rescales velocities, and there are none): it refuses such a
structure at once with an `InputError`, before anything is compiled.

`fm.md(..., initial_velocities=...)` (TOML key `initial_velocities`,
`fastmace md --initial-velocities`) changes this. It is opt-in and acts
**only on a structure at rest** (zero kinetic energy):

| Value | A structure at rest starts with |
| --- | --- |
| `"keep"` (default) | no velocities: it starts at rest, as in ASE |
| `"maxwell_boltzmann"` | Maxwell-Boltzmann velocities at the run's starting target: `temperature_K`, the integrator's, or the schedule's first temperature. Refused when that is 0 K (a schedule from 0 K) and for NVE, which has no target |
| a float, K | Maxwell-Boltzmann velocities at that temperature; works for NVE and for a schedule that starts at 0 K |

```python
from fastmace.protocols import ramp

# CSVR heating from 0 K: start the atoms at 10 K, then follow the ramp.
run = fm.md(atoms, model="mh1", head="matpes_r2scan",
            integrator=fm.integrators.CSVR(timestep_fs=1.0, temperature_K=0.0, tau_fs=100.0),
            schedule=ramp(0, 1000, ps=20), initial_velocities=10.0, seed=1, output="heat/")
```

What to know before using it:

- **Seeded.** The draw uses `seed=` (a drawn seed when `seed` is `None`, recorded in
  `metadata.json` as always), through its own random stream, so the thermostat noise is
  the same as without the option. The same seed gives the same starting velocities.
- **Only at rest.** A structure that already has momenta keeps them; nothing is redrawn.
- **Recorded.** The report and `metadata.json`'s warnings say which happened, for example
  `starting velocities: Maxwell-Boltzmann at 10 K, drawn from seed 1`.
- **What is drawn.** ASE's Maxwell-Boltzmann draw (`thermalize_momenta`, or `MaxwellBoltzmannDistribution` before ASE 3.29), without rescaling to the exact
  temperature; `FixAtoms` atoms get none. The integrator then treats the momenta as usual:
  with `fixcm` it removes the total momentum at the start.
- **`fm.Simulation` does not draw.** The option belongs to `fm.md` (and so `fastmace run` and
  `fastmace md`); with `Simulation`, set the momenta on the `Atoms` yourself, as below.

To set the momenta yourself instead:

```python
import numpy as np
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(1))
run = fm.md(atoms, model="mh1", head="matpes_r2scan", ensemble="NVE",
            timestep_fs=0.5, steps=100_000, output="nve/", seed=1)
```

This matters most for NVE, which has no thermostat. An NVE run sizes its
neighbour-list skin for twice the starting kinetic temperature, at least
1 K. `fm.md` with an `Atoms` writes to `./fastmace-run/` unless you pass
`output=`.

## Seeds

`seed=` seeds the thermostat noise and the MC streams. In `fm.md` it is
optional: `None` draws a seed and records it in `metadata.json`, so any
run can be repeated. `fm.Simulation` requires an int.

!!! note "Bitwise reproducibility"
    The default hybrid backend uses atomic sums, so two runs with the same
    seed are not bitwise identical; compare them statistically. See
    [Reproducibility](restart.md#reproducibility).

## Steps and schedules

A run with a constant temperature needs `steps=`. A run with a schedule
needs no `steps=`: it covers the whole schedule.

`fm.protocols.ramp(start_K, end_K, ps=...)` (or `rate_K_per_ps=`, signed)
and `fm.protocols.hold(temperature_K, ps=...)` build a schedule; `+` joins
them, and each segment must start where the previous one ended:

```python
heat_hold_cool = (fm.protocols.ramp(0, 1000, ps=20) + fm.protocols.hold(1000, ps=10)
                  + fm.protocols.ramp(1000, 250, rate_K_per_ps=-3))
run = fm.md("NaCl.cif", model="mh1", head="matpes_r2scan", schedule=heat_hold_cool,
            timestep_fs=1.0, repeat=(3, 3, 3), output="anneal/", seed=1)
```

`temperature_K=` and `schedule=` exclude each other. Durations are rounded
to whole steps at the run's timestep. The named profiles `slow`, `fast`,
`ultrafast` and `anneal` are in `fm.protocols.PROFILES`; for example
`fm.protocols.PROFILES["anneal"].schedule` is 0 → 1000 K in 20 ps, then
1000 → 250 K over 250 ps.

With a schedule, `steps=` runs only the first part. `Simulation.run`
continues it:

```python
sim = fm.Simulation(
    atoms, model,
    integrator=fm.integrators.Langevin(timestep_fs=1.0, temperature_K=0.0, friction_per_fs=0.002),
    workload=fm.Workload(max_temperature_K=1000),
    output=fm.Output("anneal/"), seed=1,
)
sim.run(schedule=heat_hold_cool, steps=50_000)   # the first 50,000 steps
sim.run()                                        # the rest of the schedule
```

`fm.Workload(max_temperature_K=...)` tells the planner the highest target,
which sizes the neighbour-list skin. `fm.md` works it out from the
schedule; with `Simulation` you pass it, and a schedule above it is refused
before the first step. Without a workload, `Simulation` takes the
integrator's `temperature_K`.

`Simulation.run(steps=..., schedule=None)` without an attached schedule
keeps the current target. Example script:
[`examples/01_nvt_anneal_nacl.py`](https://github.com/FZWINGEL-UBT/FastMACE/blob/main/examples/01_nvt_anneal_nacl.py).

## Output intervals

`fm.Output` sets the directory and the intervals, in MD steps. The defaults
are those of `fm.md`:

```python
out = fm.Output("run/", trajectory_every=20, thermo_every=20, restart_every=10_000)
run = fm.md("NaCl.cif", model="mh1", head="matpes_r2scan", temperature_K=300,
            steps=100_000, output=out, seed=1)
```

- `trajectory_every`: a trajectory frame every n steps; `None` writes no
  trajectory.
- `thermo_every`: a thermo row every n steps; always written.
- `restart_every`: a restart point every n steps; `None` writes no
  periodic restart points.

A plain string, `output="run/"`, takes these defaults. Output is written
asynchronously, so writing does not slow the GPU. The files are described
in [Output, restart and resume](restart.md).

See also: [Integrators and optimisers](../reference/integrators.md),
[Output files](../reference/output.md),
[API: quick runs](../api/quick.md), [API: Simulation](../api/simulation.md).
