# Relaxations and quenches

## Command line

```bash
fastmace relax NaCl.cif --model mh1 --head matpes_r2scan --optimizer gpu-LBFGS --fmax 0.01
fastmace relax NaCl.cif --model mh1 --head matpes_r2scan --optimizer gpu-FIRE --cell-filter frechet --pressure 0
fastmace quench run/ --into gpu --optimizer gpu-LBFGS
```

| Optimizer name | Runs on |
| --- | --- |
| `gpu-LBFGS` (default) | the graph engine |
| `gpu-FIRE` | the graph engine |
| `ase-LBFGS`, `ase-BFGS` | ASE's optimisers on `fm.calc` |
| `LBFGS`, `BFGS` | ASE's (so existing protocol files keep their meaning) |

`gpu-LBFGS` reproduces the annealing-benchmark minima of ASE's LBFGS.

`fastmace relax` writes the relaxed structure (`--output`, default
`<stem>.relaxed.extxyz`) with a `.json` summary and a `.log`.

`fastmace quench RUN_DIR` quenches frames of a finished run, chosen as a
protocol's quench picks them. Each frame keeps its own cell. `--into NAME`
writes `RUN_DIR/quench-NAME/`, so one run can be quenched with several
optimisers side by side. A GPU quench relaxes all its frames on one engine.

## Python: `Simulation.relax`

```python
import fastmace as fm

model = fm.load("mh1", head="matpes_r2scan")
opt = fm.optimize.LBFGS()                                  # ASE 3.27's defaults
opt = fm.optimize.FIRE(cell_filter=fm.optimize.FrechetCell(scalar_pressure_GPa=0.0))
sim = fm.Simulation(atoms, model, optimizer=opt, seed=0)
relaxed, summary = sim.relax(fmax=0.01)
```

A relaxation `Simulation` takes `optimizer=` instead of `integrator=` and
writes no output files (pass no `output=`).

## Python: ASE-style optimisers

```python
from ase.filters import FrechetCellFilter
from fastmace.ase import LBFGS

opt = LBFGS(FrechetCellFilter(atoms), model="mh1", head="matpes_r2scan")
opt.run(fmax=0.01)
```

`fastmace.ase.LBFGS` and `FIRE` take ASE 3.27's arguments.
`use_line_search=True`, `position_reset_callback` and the deprecated
`maxmove` are refused. They take `Atoms` or `FrechetCellFilter`; other cell
filters are refused with the reason.

## Anneal and quench protocols

`schedule=fm.protocols.PROFILES["anneal"].schedule` (or the profiles
`slow`, `fast`, `ultrafast`; in TOML just `schedule = "anneal"`) runs a
temperature schedule; `fm.protocols.ramp` and `fm.protocols.hold` build your
own:

```python
anneal = fm.protocols.ramp(0, 1000, ps=20) + fm.protocols.ramp(1000, 250, rate_K_per_ps=-3)
sim.run(schedule=anneal)
```

A protocol file's `[relax]` and `[quench]` tables take the same optimizer
names.

See also: [Integrators and optimisers](../reference/integrators.md), [Command line and TOML](../reference/cli.md).
