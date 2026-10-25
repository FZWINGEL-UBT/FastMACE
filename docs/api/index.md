# API reference

The public API is the `fastmace` namespace, imported as `import fastmace as fm`.
Importing it loads neither torch nor MACE; each name is resolved at first use.

| Name | Page |
| --- | --- |
| `fm.md` | [Quick runs](quick.md) |
| `fm.load`, `fm.ForceField`, `fm.Workload` | [Models and force fields](model.md) |
| `fm.Simulation` | [Simulation](simulation.md) |
| `fm.integrators` | [Integrators](integrators.md) |
| `fm.optimize` | [Optimisers](optimize.md) |
| `fm.mc` | [Monte Carlo](mc.md) |
| `fm.protocols` | [Protocols and schedules](protocols.md) |
| `fm.Output` | [Output](output.md) |
| `fm.D3` | [Dispersion](d3.md) |
| `fm.calc`, `fastmace.ase` | [ASE front end](ase.md) |
| exceptions | [Errors](errors.md) |

Units follow ASE: Å, eV, eV/Å, K and amu, times in fs at the API, pressure in
GPa only where a name ends in `_GPa`.

::: fastmace.runtime_info

::: fastmace.RuntimeInfo
