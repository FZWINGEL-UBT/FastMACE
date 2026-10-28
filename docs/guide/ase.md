# Port your ASE script

FastMACE offers three ways in from ASE. They all use the same force field,
planner, gate and restart files.

| You have | Use | Speed |
| --- | --- | --- |
| a tool that needs only a calculator (phonopy, NEB, your own optimiser) | `fm.calc` | compiled tier, no CUDA graphs |
| an ASE MD or optimiser script | `fastmace.ase` classes | full graph-engine speed |
| a new run | `fm.md` or `fm.Simulation` | full graph-engine speed |

## Level 0: a calculator, `fm.calc`

```python
import fastmace as fm

atoms.calc = fm.calc("mh1", head="matpes_r2scan")
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
```

`fm.calc` wraps `fastmace.ase.FastMACECalculator`. ASE calls a calculator
once per step from Python, so a calculator cannot use CUDA graphs over the
whole step. You get the compiled model and the exact rewrites, not the full
speed-up. FastMACE prints this once: use `fm.md` or the `fastmace.ase`
integrators for full speed.

## Level 1: change the import

An ASE Langevin script:

```python
from ase import units
from ase.md.langevin import Langevin

atoms.calc = mace_mp(...)
dyn = Langevin(atoms, 0.5 * units.fs, temperature_K=300, friction=0.02 / units.fs,
               trajectory="run.traj", logfile="run.log", loginterval=20)
dyn.attach(print_energy, interval=1000)
dyn.run(1_000_000)
```

becomes:

```python
from ase import units
from fastmace.ase import Langevin

dyn = Langevin(atoms, 0.5 * units.fs, temperature_K=300, friction=0.02 / units.fs,
               model="mh1", head="matpes_r2scan", trajectory="run.traj",
               logfile="run.log", loginterval=20)
dyn.attach(print_energy, interval=1000)
dyn.run(1_000_000)
print(dyn.summary)        # the run report
```

Rules:

- **Arguments keep ASE 3.27's names and units.** `friction` is in ASE's
  inverse time units, as in ASE. The shim never reinterprets an ASE
  argument. An argument that is neither supported nor explicitly refused
  raises `TypeError` naming it.
- **Refused:** `rng` (pass `seed=`; the noise comes from FastMACE's device
  generator), a non-default `communicator`, and `master`.
- **FastMACE additions are keyword-only:** `model=` or `force_field=`,
  `head=`, `seed=`, `output=`, `check=`, `dispersion=`.
- **Files:** `trajectory=` and `logfile=` keep their ASE meaning and format,
  so your analysis scripts work unchanged. Everything else (`metadata.json`,
  the report, restart files) goes to `output=`, by default
  `<trajectory stem>.fastmace/`.
- **Observers** see the committed state in `atoms`, and
  `get_potential_energy()`, `get_forces()` and `get_stress()` return the
  engine's values without a new evaluation. An observer may change
  positions, momenta, species, masses or charges; FastMACE applies the
  change as one event. Changing the atom count, PBC, cell, constraints or
  `atoms.calc` raises, because each would invalidate the plan.
- **FastMACE members:** `summary`, `plan`, `gate`, `metadata`, `simulation`
  (the underlying `fm.Simulation`) and `output`.

### Mapped classes

| ASE 3.27 class | `fastmace.ase` class | FastMACE integrator |
| --- | --- | --- |
| `Langevin` | `Langevin` | `fm.integrators.Langevin` |
| `LangevinBAOAB` | `LangevinBAOAB` | `fm.integrators.LangevinBAOAB` |
| `VelocityVerlet` | `VelocityVerlet` | `fm.integrators.NVE` |
| `Bussi` | `Bussi` | `fm.integrators.CSVR` |
| `NoseHooverChainNVT` | `NoseHooverChainNVT` | `fm.integrators.NoseHooverChain` |
| `IsotropicMTKNPT`, `MTKNPT` | same names | `fm.integrators.MTK` |
| `NVTBerendsen`, `NPTBerendsen`, `Inhomogeneous_NPTBerendsen` | same names | `fm.integrators.Berendsen` |
| `LBFGS`, `FIRE` (optimisers) | `LBFGS`, `FIRE` | `fm.optimize.LBFGS`, `fm.optimize.FIRE` |

An ASE class without a mapping raises `PlanRefused` with the list of mapped
ones. The shim never falls back to ASE's own loop silently.

### Resume a shim run

```python
import fastmace as fm

sim = fm.Simulation.resume("run.fastmace/")
sim.run()
```

The resumed run continues in FastMACE's chunked output format in that
directory (see [Output, restart and resume](restart.md)).

## Level 2: one call, `fm.md`

```python
import fastmace as fm

run = fm.md("NaCl.cif", model="mh1", head="matpes_r2scan", ensemble="NVT",
            temperature_K=300, steps=1_000_000, repeat=(3, 3, 3),
            output="run/", seed=1)
print(run.summary)
```

See [FastMACE in five minutes](../quickstart.md) for the arguments and the
command-line equivalent, `fastmace md`.

See also: [API: ASE front end](../api/ase.md), [Integrators and optimisers](../reference/integrators.md).
