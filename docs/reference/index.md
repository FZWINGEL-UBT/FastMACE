# Reference

These pages list every option FastMACE takes, with its values, default and
effect. The [user guide](../quickstart.md) shows how to use them for common
jobs. The [API reference](../api/index.md) lists the functions and classes
with their signatures.

## How options reach a run

Every option is a field of a frozen record. Nothing is set through
module-level switches or environment variables, except the few process-wide
settings on [Environment and cache](environment.md).

| Where you set it | What it takes | Page |
| --- | --- | --- |
| `fm.load(label, head=..., ...)` | the model, head, charge, spin, trust | [Models, D3, PolarMACE and schedules](models.md) |
| `fm.ForceField(model, atoms, precision=..., rewrites=..., dispersion=..., polar=...)` | force precision, a `RewriteConfig`, D3, a `PolarConfig` | [Rewrites and precision](rewrites.md), [PolarMACE](models.md#polarconfig) |
| `fm.Simulation(..., integrator=...)` or `optimizer=...` | an `fm.integrators` or `fm.optimize` record | [Integrators and optimisers](integrators.md) |
| `fm.Simulation(..., engine=..., neighbours=...)` | `EngineConfig`, `NeighbourConfig` | [Engine and neighbour list](engine.md) |
| `fm.Simulation(..., mode=..., check=..., workload=..., fallback=..., gate=..., runner=..., adaptation=..., calibration=...)` | the planner's inputs | [Planner, gate and fallback](planner.md) |
| `fm.md(..., mc=...)`, `Simulation.run(..., mc=...)` | `fm.mc` move records | [Monte Carlo](mc.md) |
| `fm.Simulation(..., output=fm.Output(...))` | output intervals and files | [Output files](output.md) |
| `fastmace md`, `fastmace run run.toml`, ... | flags and TOML keys | [Command line and TOML](cli.md) |

`fm.md` takes most of these as keyword arguments too and passes them on.

The configuration records are not in the `fm` namespace. Import them from
`fastmace.config`:

```python
import fastmace as fm
from fastmace.config import EngineConfig, NeighbourConfig, RewriteConfig

model = fm.load("mh1", head="matpes_r2scan")
ff = fm.ForceField(model, atoms, rewrites=RewriteConfig(radial_tables="on"))
sim = fm.Simulation(atoms, ff, integrator=fm.integrators.Langevin(temperature_K=300, friction_per_fs=0.002), seed=1,
                    engine=EngineConfig(pipelined="off"), neighbours=NeighbourConfig(skin=0.5))
print(sim.plan.overrides())   # the options you set, and what the planner did with them
```

## `"auto"`, `"on"` and `"off"`

Many fields are toggles with three values.

- `"auto"` (the default) lets the planner decide, with a general rule
  based on the model, backend, precision and execution tier. The plan
  records the rule and the reason.
- `"on"` asks for the option. If the model or plan cannot use it, the
  planner raises `PlanRefused` with the reason before the first step.
  Nothing is substituted silently.
- `"off"` turns the option off.

`print(sim.plan)` shows the decisions that matter, and
`sim.plan.describe(full=True)` shows every decision with its rule. The same
text is in the run report and in `metadata.json`.

## Units

Units follow ASE: Å, eV, eV/Å, K and amu. Times are in fs at the API.
Pressures are in GPa wherever a name ends in `_GPa`.
