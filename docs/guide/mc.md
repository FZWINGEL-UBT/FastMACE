# Monte Carlo and swaps

!!! warning "Experimental"
    MC moves carry an "experimental: not yet qualified" line in the run
    report until their qualification study is recorded.

FastMACE runs hybrid MD/MC: MD steps on the graph engine, with blocks of MC
attempts in between. Species changes are **transactional events inside the
captured graph**, so a swap does not cost a re-capture. The species superset
(every species the run may contain) is declared up front.

## Move records

| Record | Move |
| --- | --- |
| `fm.mc.Swap(pairs=[("O", "Cl")], every=100, attempts=50)` | swap the species of two atoms; `velocities="exchange"` (default), `"rescale"`, `"redraw"` or `"redraw_all"` |
| `fm.mc.Transmute(species=("Na", "K"), every=..., attempts=..., mu={...})` | semi-grand canonical species change; `mu` are configurational chemical-potential differences in eV, `fm.mc.configurational_mu` converts full ones |
| `fm.mc.Displace(delta_A=0.1, every=..., attempts=...)` | displace one atom in a sphere; `adaptive=True` tunes δ during equilibration |

Each attempt is accepted by the Metropolis rule at the thermostat's current
target temperature. The thermo output has `mc_<move>_attempted` and
`mc_<move>_accepted` columns, and the counters survive restarts.

## With `fm.md`

```python
import fastmace as fm

run = fm.md("NaNbOCl4.cif", model="mh1", head="matpes_r2scan",
            temperature_K=600, steps=200_000,
            mc=fm.mc.Swap(pairs=[("O", "Cl")], every=100, attempts=50),
            output="swap/", seed=1)
```

`fm.md` infers the species superset and the workload from `mc=`. With a
constant temperature the MC temperature is that temperature, which must be
positive. With a schedule, `min_mc_temperature_K=` is required: MC blocks
attach only where the schedule's target is at or above it.

## With `Simulation`

```python
model = fm.load("mh1", head="matpes_r2scan")
ff = fm.ForceField(model, atoms, species=["Na", "Nb", "Cl", "O"])
sim = fm.Simulation(
    atoms, ff,
    integrator=fm.integrators.Langevin(timestep_fs=1.0, temperature_K=600, friction_per_fs=0.002),
    workload=fm.Workload(max_temperature_K=1000, species_changes=True, mc_moves=("swap",),
                         min_mc_temperature_K=250),
    output=fm.Output("run/"), seed=1,
)
sim.run(steps=100_000)                                    # MD only
sim.run(steps=100_000, mc=fm.mc.Swap(pairs=[("O", "Cl")], every=100, attempts=50))
sim.run(steps=50_000)                                     # the swap block stays attached
sim.run(steps=50_000, mc=None)                            # detach
```

A move set stays attached until a later `run` passes another `mc=` or
`mc=None`. A species outside the superset raises `SpeciesNotInSuperset`.

## From TOML

```toml
[[mc]]
move = "swap"
pairs = [["O", "Cl"]]
every = 100
attempts = 50
```

## Charge and spin

For models that read charges (PolarMACE) or a total charge or spin
(OMOL-0), a transmutation must keep ΔZ − Δq even (`formal_charges=`), or it
is refused with `ChargeParityRefused`.
