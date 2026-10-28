# Monte Carlo moves

This page lists every move record in `fm.mc`, how moves attach to an MD run,
and the `MCConfig` limits. For worked examples see
[Monte Carlo and swaps](../guide/mc.md). The generated API page is
[Monte Carlo](../api/mc.md).

!!! warning "Experimental"
    MC moves are not yet qualified. Every run with MC has an
    "MC: experimental: not yet qualified" line in its report.

## How hybrid MD/MC works

A run with MC alternates MD steps and **blocks** of MC attempts. A move with
`every=100` and `attempts=50` runs a block of 50 attempts after every
simulation step that is a multiple of 100. When several moves are due at
the same step, they run in the order they were attached.

Each attempt is: propose, apply the trial, evaluate the energy, accept or
reject, and restore the old state where rejected. All of it runs as device
operations inside a captured CUDA graph, so an attempt costs one energy
evaluation and no re-capture. The forces are recomputed once at the end of a
block that accepted anything.

**Acceptance.** The Metropolis rule at the thermostat's current target
temperature T: accept if ln u < −ΔE / (k_B T), with u uniform in [0, 1).
For a transmutation the exponent uses ΔE − Σ Δμ_s ΔN_s, the semi-grand
canonical excess. A NaN energy rejects. T is the target at the start of the
block, so it follows a schedule.

**Energy precision.** The MC energy is evaluated in the force precision when
the plan's move-energy gate passes, else in float64. The gate threshold is
0.01 k_B T_min, with T_min the workload's `min_mc_temperature_K`. Set
`fm.Workload(mc_energy_precision="float32" | "float64")` to choose it
yourself.

**Counters.** Each move counts `attempted`, `impossible` (a swap whose
species are not both present, for example) and `accepted`. The thermo output
has `mc_<move>_attempted` and `mc_<move>_accepted` columns, and the counters
survive restarts.

## `fm.mc.Swap`

Swap the species of two atoms. A pair type is chosen uniformly from
`pairs`, then one atom of each of its two species uniformly.

```python
fm.mc.Swap(pairs=[("O", "Cl")], every=100, attempts=50)
```

| Option | Values / units | Default | What it does | Notes |
| --- | --- | --- | --- | --- |
| `pairs` | sequence of pairs of species (symbols or atomic numbers) | required | The species pairs that may swap. | At least one pair; each pair names two different species. |
| `every` | MD steps, int ≥ 1 | required | Block interval. | |
| `attempts` | int ≥ 1 | required | Attempts per block. | At most 10,000. |
| `velocities` | `"exchange"`, `"rescale"`, `"redraw"`, `"redraw_all"` | `"exchange"` | Momentum treatment of the two atoms after an accepted swap. | See [Velocity treatments](#velocity-treatments). |

With a model that reads per-atom charges, the charges move with the species.

## `fm.mc.Transmute`

Change one atom's species within a transmutable set: semi-grand canonical
MC. One atom whose species is in the set is chosen uniformly, and a new
species is drawn uniformly from the rest of the set.

```python
fm.mc.Transmute(species=("Na", "K"), every=100, attempts=20, mu={"K": -0.15})
```

| Option | Values / units | Default | What it does | Notes |
| --- | --- | --- | --- | --- |
| `species` | sequence of species (symbols or atomic numbers) | required | The transmutable set. | At least two different species. |
| `every` | MD steps, int ≥ 1 | required | Block interval. | |
| `attempts` | int ≥ 1 | required | Attempts per block. | At most 10,000. |
| `velocities` | `None`, `"rescale"`, `"redraw"`, `"redraw_all"` | `None` | Momentum treatment of the changed atom. | `None`: `"redraw_all"` when the integrator fixes the centre of mass, `"rescale"` otherwise. `"exchange"` is for swaps only. |
| `mu` | mapping species → eV | `None` | Configurational chemical-potential differences Δμ_s, against a reference species. | Species left out have 0. Keys must be in `species`. |
| `formal_charges` | mapping species → int | `None` | Formal charge q_s of each species, for the charge and spin parity rule. | Species left out have 0. |

**Chemical potentials.** `mu` takes *configurational* differences. To convert
full chemical-potential differences (which include the kinetic part), use

```python
mu = fm.mc.configurational_mu({"K": -0.20}, temperature_K=600, reference="Na")
```

which returns Δμ_s^conf = Δμ_s^full + (3/2) k_B T ln(m_s / m_ref) for each
key. `masses=` (amu, by species) replaces ASE's standard atomic masses. The
reference's own entry, if given, must be 0.

**Charge and spin.** For a model that reads charges or a total charge or
spin (PolarMACE, MACE-OMOL-0), every pair of species in the set must have
ΔZ − Δq even, so that the spin multiplicity stays valid. Give
`formal_charges` to make it so; otherwise the move is refused with
`ChargeParityRefused`. For other models the rule does not apply.

## `fm.mc.Displace`

Move one atom, chosen uniformly, to a uniform point in a sphere of radius δ
around it. The proposal is symmetric, so the acceptance is plain Metropolis
on ΔE. Momenta are not changed.

```python
fm.mc.Displace(delta_A=0.1, every=10, attempts=100)
```

| Option | Values / units | Default | What it does | Notes |
| --- | --- | --- | --- | --- |
| `delta_A` | Å, > 0 | required | Sphere radius δ, and the largest δ an adaptive move may reach. | At most the workload's `max_displacement_A`. |
| `every` | MD steps, int ≥ 1 | required | Block interval. | |
| `attempts` | int ≥ 1 | required | Attempts per block. | At most 10,000. |
| `adaptive` | bool | `False` | Tune δ at the end of every block toward 30 % acceptance. | See below. |

With `adaptive=True`, after each block δ ← δ · clamp(acceptance / 0.3, 0.5,
2), capped at `delta_A` and at `max_displacement_A`. Every value is recorded,
and the restart keeps the current one. A changing δ breaks detailed balance,
so use an adaptive move for equilibration and a fixed one for sampling.

Displacements need `fm.Workload(max_displacement_A=...)`, because the
neighbour-list skin is sized for them (2δ is added to the skin). They are
refused with a variable cell (NPT). `fm.md` does not set
`max_displacement_A`, so run displacement moves through `fm.Simulation`.

An attempt that would need a neighbour-list rebuild, or more edges than the
captured graph holds, is redone on the host with the same random numbers.
The outcome is the same; only that attempt is slower.

## Velocity treatments

A swap or transmutation changes atomic masses. The momentum treatment says
what the velocities of the changed atoms become after an accepted move. v is
the velocity, m the mass, z a standard normal vector.

| Value | Moves | What it does | Conserves |
| --- | --- | --- | --- |
| `"exchange"` | swaps | The two atoms exchange velocities with their species: v_i' = v_j, v_j' = v_i. | Kinetic energy and total momentum. |
| `"rescale"` | swaps, transmutations | Keep each atom's kinetic energy: v' = v · sqrt(m_old / m_new). | Kinetic energy, not total momentum. |
| `"redraw"` | swaps, transmutations | Draw the changed atoms' velocities from Maxwell-Boltzmann at T with the new masses: v' = sqrt(k_B T / m_new) · z. | Neither. |
| `"redraw_all"` | swaps, transmutations | Once per block with an acceptance, draw every velocity from Maxwell-Boltzmann at T and project to zero total momentum. | Total momentum (zero). |

The integrator's centre-of-mass setting limits the choice:

- With `fixcm` on, `"rescale"` and `"redraw"` are refused, because they change
  the total momentum behind the integrator's back. Use `"exchange"` (swaps)
  or `"redraw_all"`.
- `"redraw_all"` under `fixcm` draws (3N − 3)·k_B T of kinetic energy. It
  needs the Langevin noise projection `fixcm_projection="mass_weighted"`,
  which is what `"auto"` gives under NVT `Langevin` with `fixcm` on. Under
  `LangevinBAOAB` (its `fixcm` defaults to `True` and its projection resolves
  to `"ase"`), `"redraw_all"` is refused, so swaps there use `"exchange"` and
  transmutations need `fixcm=False`.

## Random numbers and seeds

MC uses three counter-based random streams (Philox4x32-10): `proposal`,
`acceptance` and `velocity`. Their seeds are fixed 63-bit values derived by
SHA-256 from the `Simulation`'s `seed`, distinct from the integrator's noise
seed. Every attempt consumes a fixed count of numbers from each stream,
whatever its outcome, so a number is a function of the seed and its position
only. The numbers are the same on the CPU and the GPU. A stream's state is
its seed and offset, which the restart stores; a resumed run continues the
same streams.

## Attaching moves to a run

### `fm.md`

`fm.md(..., mc=...)` takes one move record or a sequence of them. It infers
the species superset (the structure's species plus every species the moves
name) and the workload from `mc=`.

| Option | Values / units | Default | What it does | Notes |
| --- | --- | --- | --- | --- |
| `mc` | an `fm.mc` record or a sequence of them | `None` | The move set. | Refused with `ensemble="NVE"`. |
| `min_mc_temperature_K` | K, > 0 | `None` | The lowest target at which MC blocks run, and T_min of the energy gate. | Without a schedule: defaults to `temperature_K`, which must be positive. With a schedule: required. |

With a schedule, the move set is attached only over the step ranges whose
target is at or above `min_mc_temperature_K`, and detached elsewhere. The
report and `metadata.json` record the ranges.

### `fm.Simulation`

`sim.run(steps=..., mc=...)` attaches a move set. It stays attached until a
later `run` passes another `mc=` or `mc=None`. The workload must declare the
moves up front, because the plan and its gate are built for them:

```python
workload = fm.Workload(max_temperature_K=1000, species_changes=True,
                       mc_moves=("swap",), min_mc_temperature_K=250)
```

| `Workload` field | Needed for |
| --- | --- |
| `mc_moves` | Every move type you will attach (`"swap"`, `"transmute"`, `"displace"`). An undeclared move is refused. |
| `species_changes=True` | Swaps and transmutations. |
| `min_mc_temperature_K` | Any MC. A block whose target falls below it is refused. |
| `max_displacement_A` | Displacements. |
| `mc_energy_precision` | Optional: `"auto"` (default), `"float32"` or `"float64"`. |

The `ForceField`'s `species=` must contain every species a move can create;
a species outside it raises `SpeciesNotInSuperset`. MC with any constraint,
`FixAtoms` included, is refused.

### TOML

One `[[mc]]` table per move, with `move = "swap"`, `"transmute"` or
`"displace"` and the record's fields:

```toml
[[mc]]
move = "swap"
pairs = [["O", "Cl"]]
every = 100
attempts = 50
velocities = "exchange"
```

## `MCConfig`

`fastmace.config.MCConfig` holds the limits of the MC driver. A `Simulation`
always uses the defaults; `fm.md`, `Simulation` and the TOML files take no
`MCConfig`. A move set over a limit raises `ValueError` when it is attached.

| Option | Values / units | Default | What it does | Notes |
| --- | --- | --- | --- | --- |
| `max_attempts_per_block` | int ≥ 1 | 10,000 | The largest `attempts` of one move. The captured attempt is replayed at most this often per block. | |
| `max_move_sets` | int ≥ 1 | 8 | The most move records attached at once. | |
