# FastMACE in five minutes

This page takes you from a fresh install to a finished MD run. It runs the
same simulation three ways: with the `fastmace md` command, with `fm.md`
in Python, and as a ported ASE script. All three build the same
`Simulation`, so for the same inputs they give the same plan, gate and
trajectory.

You need Linux, an NVIDIA GPU and a FastMACE install (see
[Install](install.md)).

## 1. Check the installation

```bash
fastmace doctor
```

`doctor` checks the GPU, the driver, torch, MACE, the cuEquivariance ops
and their fused kernels, OpenEquivariance, Triton, NVRTC, the optional
packages, the cache root and the model files. For each item that fails it
names the command or page that fixes it. It changes nothing. It exits 1 if a
required check fails and 0 otherwise. Optional items, such as the GPU
and OpenEquivariance, only warn.

## 2. Get a model and a structure

`fastmace models` lists the model labels with their heads, files, SHA-256
and licences. This tutorial uses MH-1 with its r2SCAN head. Download the
checkpoint once; FastMACE checks its SHA-256 and keeps it in its cache:

```bash
python -c 'import fastmace as fm; fm.load("mh1", head="matpes_r2scan", download=True)'
```

Later runs find the file in the cache, or in `$FASTMACE_MODELS` if you keep
your checkpoints there. MH-1 is under the Academic Software License
(non-commercial); see [Models and licences](guide/models.md).

Any structure file ASE reads works. To make a rock-salt NaCl cell (8 atoms,
a = 5.64 Å):

```bash
python -c 'from ase.build import bulk; bulk("NaCl", "rocksalt", a=5.64, cubic=True).write("NaCl.cif")'
```

## 3. Run it from the command line

```bash
fastmace md --model mh1 --head matpes_r2scan --structure NaCl.cif \
  --ensemble NVT --temperature 300 --timestep 0.5 --steps 20000 \
  --repeat 3 3 3 --seed 1 --output run/
```

This is 20,000 Langevin NVT steps of 0.5 fs (10 ps) at 300 K on a 3×3×3
supercell, 216 atoms. The first start takes tens of seconds, because
FastMACE compiles the model and captures CUDA graphs; later starts reuse
the caches.

The flags you will use most:

| Flag | Meaning | Default |
| --- | --- | --- |
| `--model`, `--head` | model label, path or URL, and its head | required; the head is required for MH-1 |
| `--structure` | structure file, any format ASE reads | required |
| `--repeat NA NB NC` | supercell | `1 1 1`: the cell is never expanded silently |
| `--ensemble` | `NVT` (Langevin), `NPT` (Langevin-BAOAB) or `NVE` (velocity Verlet) | `NVT` |
| `--temperature` | target temperature, K | required, except for NVE |
| `--pressure` | NPT pressure, GPa | `0` |
| `--timestep` | timestep, fs | `0.5` |
| `--steps` | number of MD steps | required |
| `--dispersion d3bj` | add D3(BJ) dispersion | `none` |
| `--seed` | noise seed | drawn and recorded |
| `--check gate` | run the gate before the first step | `skip` |
| `--output` | output directory | `./<structure stem>.fastmace/` |

[Command line and TOML](reference/cli.md) lists every flag. An error prints
its message without a traceback and exits 1; `--debug` adds the traceback.
A wrong combination of flags exits 2.

## 4. The same run from Python

```python
import fastmace as fm

run = fm.md("NaCl.cif", model="mh1", head="matpes_r2scan", ensemble="NVT",
            temperature_K=300, timestep_fs=0.5, steps=20_000, repeat=(3, 3, 3),
            output="run-py/", seed=1)
print(run.summary)                 # the run report
print(run.plan)                    # the plan
sim = run.simulation               # the Simulation, to run on or restart
sim.run(steps=20_000)              # 20,000 more steps
```

Import `fastmace` before anything that imports torch. On NFS home
directories this is what switches PyTorch's file locks to soft locks (see
the [FAQ](faq.md#nfs-home-directories-and-soft-file-locks)).

`fm.md` returns a `RunResult` with `summary`, `plan`, `gate`, `simulation`,
`metadata` (the contents of `metadata.json`) and `output` (the directory).

## 5. The same run as an ASE script

If you have an ASE Langevin script, change the import and name the model:

```python
from ase import units
from ase.io import read

from fastmace.ase import Langevin      # was: from ase.md.langevin import Langevin

atoms = read("NaCl.cif").repeat((3, 3, 3))
dyn = Langevin(atoms, 0.5 * units.fs, temperature_K=300, friction=0.02,
               model="mh1", head="matpes_r2scan", seed=1,
               trajectory="run.traj", logfile="run.log", loginterval=20)
dyn.run(20_000)
print(dyn.summary)
```

The arguments keep ASE's names and units. `friction=0.02` is 0.02 per ASE
time unit, about 0.002 per fs: the friction `fm.md` and `fastmace md` use.
(`friction=0.02 / units.fs` would be 0.02 per fs, ten times as much.)
`trajectory=` and `logfile=` keep their ASE format; FastMACE's own files go
to `run.fastmace/`. See [Port your ASE script](guide/ase.md).

## 6. Read the report

Every run prints a one-screen report, and `metadata.json` keeps it. It
looks like this (an example; your numbers differ):

```text
FastMACE 1.0.0  |  mh1 (head matpes_r2scan)  |  NaCl, 216 atoms (3×3×3), 3D PBC, NVT Langevin 0.5 fs
physics: no dispersion; charge/spin not used by this model
plan:    graph, hybrid cuEq/OEq, float32, 18 rewrites on, 14 off (reasons in run.plan)
gate:    not run (opt-in: check="gate")
setup:   42 s (load 6 s, rewrites 9 s, compilation 17 s, capture 4 s)
speed:   4.1 ms/step, measured over steps 1,000–2,000 of this run
licence: ASL (non-commercial), from the label table
warn:    none
```

| Line | What it tells you |
| --- | --- |
| header | version, model and head, structure, atom count, supercell, ensemble and timestep |
| `physics` | dispersion, charge and spin |
| `plan` | the tier (`graph` is the CUDA-graph engine, the fastest), the backend, the force precision and the rewrites |
| `gate` | the gate result, or that it was not run |
| `setup` | the time to load, rewrite, compile, capture and gate |
| `speed` | the measured time per step; at the end of the run, the whole-run mean |
| `output` | the files written |
| `licence` | the model's licence |
| `warn` | anything you should know: a slower plan than a full install would give and why, experimental features, small cells |

If the plan line does not say `graph` and `hybrid`, a warning says why and
what would change it. `print(run.plan)` gives about 30 lines on the
decisions, and `run.plan.describe(full=True)` all of them. See
[The planner](guide/planner.md).

## 7. Find the files

`run/` holds:

| File | Content |
| --- | --- |
| `thermo.0.csv` | thermo rows: step, time, temperature, energies (and pressure for NPT) |
| `trajectory.0.traj` | ASE trajectory frames |
| `restart.npz` | the complete state at the last restart point |
| `simulation.json` | the inputs and planner options, for resume |
| `metadata.json` | the report, plan, gate and provenance (versions, model hash, GPU) |

By default FastMACE writes a frame and a thermo row every 20 steps and a
restart point every 10,000 steps. At each restart point a new chunk
starts (`thermo.1.csv`, `trajectory.1.traj`, ...). To read a whole run in
order:

```python
from fastmace.io.output import read_frames, read_thermo

frames = read_frames("run/")
thermo = read_thermo("run/")
```

To continue a run: `fm.Simulation.resume("run/")`, then `sim.run(steps=...)`.
See [Output, restart and resume](guide/restart.md).

## 8. Runs the flags cannot express

A temperature schedule, MC moves, an explicit integrator, a full D3 record
or other output intervals go in a TOML file. `fastmace init` writes one
with every `fm.md` argument, each commented with what it does:

```bash
fastmace init --model mh1 --head matpes_r2scan --structure NaCl.cif --temperature 300 --steps 20000 > run.toml
fastmace run run.toml
```

With the same flags, `fastmace run run.toml` is the same run as
`fastmace md`. The keys are `fm.md`'s argument names, and the records are
tables at the end of the file. For example, a CSVR thermostat with sparser
output:

```toml
[integrator]
type = "CSVR"
timestep_fs = 1.0
temperature_K = 300.0
tau_fs = 100.0

[output]
directory = "run/"
trajectory_every = 100
thermo_every = 20
```

A schedule is `schedule = "anneal"` (a profile: `slow`, `fast`,
`ultrafast` or `anneal`) or a list of `[[schedule]]` segments, not both.
`fastmace run` refuses a key that is not an `fm.md` argument and names it.
Paths are relative to the directory you start it in.

## Next steps

- [Running MD](guide/md.md): ensembles, integrators, timestep, friction,
  schedules and output intervals.
- [Relaxations and quenches](guide/relax.md): `fastmace relax`,
  `fastmace quench` and `Simulation.relax`.
- [NPT and D3 dispersion](guide/npt-d3.md) and
  [Monte Carlo and swaps](guide/mc.md).
- The runnable scripts in
  [`examples/`](https://github.com/FZWINGEL-UBT/FastMACE/tree/main/examples).
- [FAQ and troubleshooting](faq.md) when something fails.
