# Output, restart and resume

## What a run writes

`fm.Output("run/", trajectory_every=20, thermo_every=20, restart_every=10_000)`
(the `fm.md` defaults) writes into `run/`:

| File | Content |
| --- | --- |
| `thermo.<n>.csv` | thermo rows (step, time, temperature, energies, pressure for NPT), each chunk with its header |
| `trajectory.<n>.traj` | ASE trajectory frames |
| `restart.npz` | the complete simulation state at the last restart point |
| `simulation.json` | the inputs and planner options, for `resume` |
| `metadata.json` | plan, gate, report, provenance (versions, model hash, GPU) |

The outputs are written in **chunks**, `n = 0, 1, ...`, with a chunk
boundary at every restart point. Each chunk is complete on its own:
`ase.io.read("run/trajectory.0.traj", index=":")` reads one. To read a whole
run in order:

```python
from fastmace.io.output import read_frames, read_thermo

frames = read_frames("run/")
thermo = read_thermo("run/")
```

Output is asynchronous: a snapshot is copied to the host and written while
the next step runs, so writing does not slow the GPU.

## Restart

A restart file holds the **complete** state: positions, momenta, cell,
barostat and thermostat state, the RNG state, species after MC exchanges,
the schedule position, the MC counters and the plan revision. Restart
points are written every `restart_every` steps and on a clean stop.

```python
sim.save_restart("run/restart.npz")         # an extra one by hand
sim = fm.Simulation.resume("run/")          # continue
sim.run()                                   # the rest of the schedule
```

Resume checks the chunks up to the restart point, deletes later chunks
(rows and frames written after the restart point) and continues with the
next chunk. Nothing is rewritten. If the stored chunks do not match, it
raises `RestartMismatch` before touching anything.

A resume rebuilds the recorded plan. If the software or GPU changed so that
the plan would differ, it refuses unless you pass
`allow_plan_change=True`. A resume never falls back.

## Reproducibility

Hybrid runs (the default backend) use atomic sums and are **not bitwise
reproducible**; compare them statistically. A resumed run agrees bit for
bit with an uninterrupted one in plans made with
`fm.Workload(reproducible=True)`, which the planner refuses where it cannot
guarantee it. Restart points rebuild the neighbour list, so for bitwise
agreement set `restart_every=None` too.

## Command line

`fastmace run run.toml` writes the same files. To quench frames of a
finished run, see [Relaxations and quenches](relax.md).
