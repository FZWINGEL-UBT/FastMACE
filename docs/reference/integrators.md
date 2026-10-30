# Integrators and optimisers

This page lists every integrator record in `fm.integrators`, every optimiser
record in `fm.optimize`, and the two config records they map onto,
`IntegratorConfig` and `OptimizerConfig`. For worked examples see
[NPT and D3](../guide/npt-d3.md) and
[Relaxations and quenches](../guide/relax.md). The generated API pages are
[Integrators](../api/integrators.md) and [Optimisers](../api/optimize.md).

All records are frozen dataclasses with keyword-only arguments. They are
checked when they are built, so a wrong value raises `ValueError` before any
model is loaded. Each record ports one ASE 3.27 class, and every one of them
runs inside the CUDA-graph engine: the choice of integrator does not move a
run to a slower tier. What the planner refuses is listed under
[Constraints and refusals](#constraints-and-refusals).

## Which integrator?

| Record | Ensemble | ASE 3.27 class | Use it for | Notes |
| --- | --- | --- | --- | --- |
| `Langevin` | NVT | `Langevin` | Production NVT; `fm.md`'s default | Stochastic. Takes `FixAtoms`. |
| `LangevinBAOAB` | NPT | `LangevinBAOAB` | Production NPT; `fm.md`'s `ensemble="NPT"` | Stochastic thermostat and barostat. Experimental. |
| `NVE` | NVE | `VelocityVerlet` | Energy conservation checks, microcanonical runs | No thermostat. `fm.md`'s `ensemble="NVE"`. |
| `CSVR` | NVT | `Bussi` | Canonical NVT with little disturbance of the dynamics | Stochastic velocity rescaling. Takes `FixAtoms`. |
| `NoseHooverChain` | NVT | `NoseHooverChainNVT` | Deterministic canonical NVT | No `FixAtoms`. |
| `MTK` | NPT | `IsotropicMTKNPT`, `MTKNPT` | Deterministic NPT | Experimental. No `FixAtoms`. |
| `Berendsen` | NVT or NPT | `NVTBerendsen`, `NPTBerendsen`, `Inhomogeneous_NPTBerendsen` | Equilibration only | Not canonical; the report says so. |

Every NPT run has an "experimental: not yet qualified" line in its report.

## Common conventions

**Units.** Time in fs, temperature in K, pressure and stress in GPa,
lengths in Å. The friction of the Langevin records is per fs.

**Temperature target.** `temperature_K` is the target at the start. A
schedule (`Simulation.run(schedule=...)`, `fm.md(schedule=...)`) moves the
target every step; see [Schedules](models.md#schedules-and-profiles).

**Time constants have no defaults.** CSVR, Nose-Hoover, MTK and Berendsen
need their time constants stated, as in ASE. FastMACE never chooses a
physical time constant for you. The exception is `LangevinBAOAB`'s
barostat time constant, which follows ASE's default.

**Seed.** The records take no seed. The `Simulation`'s `seed=` seeds the
noise of the stochastic schemes (Langevin, LangevinBAOAB, CSVR).

**Friction units.** `friction_per_fs` is the friction coefficient γ in 1/fs.
ASE's `Langevin(friction=...)` takes it in inverse ASE time units; the record
converts with `friction_per_fs / ase.units.fs`. `fm.md` uses 0.02 per ASE
time unit, which is 0.001965 per fs (a velocity relaxation time 1/γ of about
509 fs). To pass ASE's value, convert it: `friction_per_fs = friction_ase *
ase.units.fs`.

### `fixcm`: centre-of-mass motion

| Value | Meaning |
| --- | --- |
| `"auto"` | Off when the structure has `FixAtoms` (the fixed atoms pin the frame already), on otherwise. The default of `Langevin`, `NVE`, `CSVR` and `Berendsen`. |
| `True` | Remove the total momentum at the start and keep it at zero. Refused with `FixAtoms`. |
| `False` | Leave the centre of mass free. |

`NoseHooverChain`, `MTK` and `LangevinBAOAB` take `True` or `False` only,
default `True`, because they refuse `FixAtoms` anyway.

### `dof`: degrees of freedom

`dof` sets the number of kinetic degrees of freedom N_f that the scheme uses
for its target kinetic energy (CSVR, Berendsen, Nose-Hoover, MTK) and for the
reported temperature (all schemes). N is the number of atoms, n_fixed the
number of fixed atoms.

| Value | N_f |
| --- | --- |
| `"fastmace"` | 3(N − n_fixed) with `FixAtoms`; 3N − 3 under `fixcm`; 3N otherwise. |
| `"ase"` | The count of the ASE class: 3(N − n_fixed) for `VelocityVerlet`, `Bussi` and the Berendsen classes; 3N for `NoseHooverChainNVT` and the MTK classes. |
| `"auto"` (default) | `"fastmace"`. |

With `dof="ase"` and `fixcm=False`, the arithmetic is ASE's. The Langevin
records have no `dof` argument. Their count is 3(N − n_fixed) with `FixAtoms`,
3N − 3 for `Langevin` with `fixcm` on and the mass-weighted projection (the
default), and 3N otherwise (so 3N for `LangevinBAOAB`).

## `fm.integrators.Langevin`

Langevin NVT, ASE's `Langevin`. Fixed cell.

```python
fm.integrators.Langevin(timestep_fs=0.5, temperature_K=300, friction_per_fs=0.002)
```

| Option | Values / units | Default | What it does | Notes |
| --- | --- | --- | --- | --- |
| `timestep_fs` | fs, > 0 | 0.5 | MD timestep. | |
| `temperature_K` | K, ≥ 0 | required | Thermostat target at the start. | A schedule moves it. |
| `friction_per_fs` | 1/fs, > 0 | required | Friction γ. | `fm.md` uses 0.001965/fs. |
| `fixcm` | `"auto"`, `True`, `False` | `"auto"` | Remove centre-of-mass motion. | See [fixcm](#fixcm-centre-of-mass-motion). |
| `fixcm_projection` | `"auto"`, `"ase"`, `"mass_weighted"` | `"auto"` | How `fixcm` removes the centre-of-mass part of the noise. | See below. |

`fixcm_projection` only matters with `fixcm` on:

- `"ase"`: ASE's projection, bit for bit with ASE's noise stream. It
  subtracts the plain mean of the random displacements and scales by
  sqrt(N/(N − 1)). It is canonical only when all masses are equal: with
  unequal masses, heavy and light species settle at slightly different
  temperatures.
- `"mass_weighted"`: the Gaussian noise conditioned on a fixed centre of
  mass, with mass-weighted means. It is canonical with 3N − 3 degrees of
  freedom for every species and every mass ratio.
- `"auto"`: `"mass_weighted"` under NVT with `fixcm` on, `"ase"` otherwise.
  The plan records the choice.

Use `"ase"` only to compare trajectories with ASE's `Langevin`.

## `fm.integrators.LangevinBAOAB`

Langevin-Hoover NPT, ASE 3.27's `LangevinBAOAB`: BAOAB Langevin dynamics for
the atoms (Leimkuhler and Matthews 2013) and a Langevin-Hoover barostat for
the cell (Quigley and Probert 2004).

```python
fm.integrators.LangevinBAOAB(timestep_fs=1.0, temperature_K=300, friction_per_fs=0.002,
                             pressure_GPa=0.0, cell_mode="hydrostatic")
```

| Option | Values / units | Default | What it does | Notes |
| --- | --- | --- | --- | --- |
| `timestep_fs` | fs, > 0 | 0.5 | MD timestep. | |
| `temperature_K` | K, ≥ 0 | required | Thermostat target at the start. | A schedule moves it. |
| `friction_per_fs` | 1/fs, > 0 | required | Thermostat friction γ. ASE's `T_tau` is 1/γ. | With `fm.md`'s 0.001965/fs, `T_tau` ≈ 509 fs. |
| `pressure_GPa` | GPa | 0.0 | Hydrostatic target pressure. | Exclusive with `external_stress_GPa`. |
| `external_stress_GPa` | GPa: a scalar or six Voigt components (xx, yy, zz, yz, xz, xy) | `None` | Target stress in ASE's `externalstress` sign (positive is tensile). | `None` means −`pressure_GPa` on the diagonal. Giving both this and a non-zero `pressure_GPa` raises `ValueError`. |
| `cell_mode` | `"hydrostatic"`, `"full"` | `"hydrostatic"` | `"hydrostatic"`: cell shape fixed, volume free (ASE `hydrostatic=True`). `"full"`: fully flexible cell. | |
| `barostat_tau_fs` | fs, > 0, or `None` | `None` | Barostat time constant, ASE's `P_tau`. | `None` is ASE's default, 20 `T_tau` (about 10.2 ps with `fm.md`'s friction). |
| `barostat_mass_factor` | > 0 | 1.0 | Factor on ASE's heuristic barostat mass, ASE's `P_mass_factor`. | ASE's mass gives a cell period of about `P_tau`/4. |
| `disable_cell_langevin` | bool | `False` | No friction and noise on the cell degrees of freedom. | As ASE. |
| `fixcm` | `False` | `False` | BAOAB does not remove centre-of-mass motion; `True` is refused. | See the note below. |
| `max_volume_ratio` | > 1, or `None` | 20.0 | Stop with `CellDivergedError` once the volume leaves [V0/ratio, V0·ratio]. V0 is the starting volume. | `None` turns the check off. |
| `max_cell_length_A` | Å, > 0, or `None` | 1000.0 | Stop with `CellDivergedError` once a cell vector is longer than this. | float32 coordinates still resolve 6e-5 Å at 1000 Å. |

The cell-divergence stop writes the restart and metadata first. Without it,
a vapour at low pressure expands without bound until the coordinates lose
their precision.

**`fixcm` in `LangevinBAOAB`.** ASE's `LangevinBAOAB` does not remove the
centre-of-mass motion, and neither does FastMACE's port, so the record takes
`fixcm=False` only. MC moves under BAOAB therefore treat the run as one
without a fixed centre of mass (see [MC](mc.md#velocity-treatments)). A
restart written before this rule, which recorded `fixcm=True`, resumes with
`False`; its dynamics were the same.

The drag term in the full-cell barostat is the corrected one. ASE 3.27's
variant, which applies a trace term to all nine drag elements, stays an
`IntegratorConfig` field (`ase_trace_drag_all_elements`) for comparisons with
ASE only.

## `fm.integrators.NVE`

Velocity Verlet, ASE's `VelocityVerlet`. The conserved quantity is
E_kin + E_pot. `fm.md(ensemble="NVE")` builds it and starts from the
structure's momenta; it draws no velocities.

| Option | Values / units | Default | What it does | Notes |
| --- | --- | --- | --- | --- |
| `timestep_fs` | fs, > 0 | required | MD timestep. | |
| `fixcm` | `"auto"`, `True`, `False` | `"auto"` | Remove the total momentum at the start and project it out after each kick. | ASE's class has no `fixcm`; it stops float drift of the force sum from accumulating. |
| `dof` | `"auto"`, `"fastmace"`, `"ase"` | `"auto"` | Count for the reported temperature. | See [dof](#dof-degrees-of-freedom). |

With `FixAtoms`, the step is ASE's constrained step: the fixed atoms' forces
and velocities are zero, and the free atoms' velocities are recomputed from
their displacement, as `VelocityVerlet.step` does.

## `fm.integrators.CSVR`

Canonical sampling through velocity rescaling (Bussi, Donadio and Parrinello,
J. Chem. Phys. 126, 014101 (2007)), ASE 3.27's `Bussi`. At the start of each
step the velocities are scaled by a stochastic factor, then one velocity
Verlet step follows. The noise is ASE's stream from the `Simulation`'s seed.
The conserved quantity is E_kin + E_pot minus the energy the thermostat
transferred, as ASE's `transferred_energy`.

| Option | Values / units | Default | What it does | Notes |
| --- | --- | --- | --- | --- |
| `timestep_fs` | fs, > 0 | required | MD timestep. | |
| `temperature_K` | K, ≥ 0 | required | Target at the start. | A schedule moves it. |
| `tau_fs` | fs, > 0 | required | Thermostat time constant, ASE's `taut`. | Larger values disturb the dynamics less. |
| `fixcm` | `"auto"`, `True`, `False` | `"auto"` | Centre-of-mass motion. | |
| `dof` | `"auto"`, `"fastmace"`, `"ase"` | `"auto"` | N_f of the target kinetic energy kT·N_f/2. | |

## `fm.integrators.NoseHooverChain`

Nose-Hoover chains, ASE 3.27's `NoseHooverChainNVT` (Martyna, Klein and
Tuckerman 1992). Deterministic. The conserved quantity is E_pot + E_kin plus
the thermostat energy, as ASE's `get_conserved_energy`.

| Option | Values / units | Default | What it does | Notes |
| --- | --- | --- | --- | --- |
| `timestep_fs` | fs, > 0 | required | MD timestep. | |
| `temperature_K` | K, ≥ 0 | required | Target at the start. | Also sets the thermostat masses; see below. |
| `tdamp_fs` | fs, > 0 | required | Thermostat damping time, ASE's `tdamp`. | ASE suggests about 100 timesteps. |
| `tchain` | int ≥ 1 | 3 | Thermostats in the chain, ASE's `tchain`. | |
| `tloop` | int ≥ 1 | 1 | Suzuki-Yoshida sub-loops per half step, ASE's `tloop`. | |
| `fixcm` | bool | `True` | Remove the total momentum at the start and after every step. | |
| `dof` | `"auto"`, `"fastmace"`, `"ase"` | `"auto"` | N_f in the first thermostat mass, its force and its energy. | ASE uses 3N. |

The thermostat masses come from the construction temperature, as in ASE. A
schedule changes the target kT, not the masses.

## `fm.integrators.MTK`

Martyna-Tobias-Klein NPT with Nose-Hoover chains on the atoms and on the
cell, ASE 3.27's `IsotropicMTKNPT` (`cell_mode="hydrostatic"`) and `MTKNPT`
(`cell_mode="full"`). Deterministic. Experimental: not yet qualified. The
conserved quantity is E_pot + E_kin plus the thermostat and barostat
energies, the cell kinetic energy and P·V, as ASE's `get_conserved_energy`.

| Option | Values / units | Default | What it does | Notes |
| --- | --- | --- | --- | --- |
| `timestep_fs` | fs, > 0 | required | MD timestep. | |
| `temperature_K` | K, ≥ 0 | required | Target at the start. | Sets the thermostat masses. |
| `pressure_GPa` | GPa | required | Target pressure, ASE's `pressure_au` in GPa. | |
| `tdamp_fs` | fs, > 0 | required | Thermostat damping time, ASE's `tdamp`. | |
| `pdamp_fs` | fs, > 0 | required | Barostat damping time, ASE's `pdamp`. | ASE suggests about 1000 timesteps. |
| `tchain` | int ≥ 1 | 3 | Thermostat chain length. | |
| `pchain` | int ≥ 1 | 3 | Barostat chain length. | |
| `tloop` | int ≥ 1 | 1 | Thermostat sub-loops. | |
| `ploop` | int ≥ 1 | 1 | Barostat sub-loops. | |
| `cell_mode` | `"hydrostatic"`, `"full"` | `"hydrostatic"` | `"hydrostatic"`: isotropic strain, cell = cell0·exp(ε). `"full"`: 3×3 cell momentum. | |
| `fixcm` | bool | `True` | Remove the total momentum at the start and after every step. | |
| `dof` | `"auto"`, `"fastmace"`, `"ase"` | `"auto"` | N_f in every coupling term (thermostat mass, drag 1 + d/N_f, barostat mass). | `"ase"` with `fixcm=False` is ASE's arithmetic. |

ASE 3.27's own inconsistencies in these classes are ported as they are; the
reference is ASE.

## `fm.integrators.Berendsen`

Berendsen thermostat and barostat (Berendsen et al., J. Chem. Phys. 81, 3684
(1984)). **Not canonical: use it for equilibration only.** The plan warns and
the report says "non-canonical: equilibration only".

| Option | Values / units | Default | What it does | Notes |
| --- | --- | --- | --- | --- |
| `timestep_fs` | fs, > 0 | required | MD timestep. | |
| `temperature_K` | K, ≥ 0 | required | Target at the start. | |
| `taut_fs` | fs, > 0 | required | Thermostat time constant, ASE's `taut`. | The velocity scale per step is clamped to [0.9, 1.1], as in ASE. |
| `pressure_GPa` | GPa, or `None` | `None` | Target pressure. `None` is NVT (`NVTBerendsen`). | A pressure needs `taup_fs` and `compressibility_per_GPa`. |
| `taup_fs` | fs, > 0, or `None` | `None` | Barostat time constant, ASE's `taup`. | Needs `pressure_GPa`. |
| `compressibility_per_GPa` | 1/GPa, > 0, or `None` | `None` | Compressibility κ, ASE's `compressibility_au` in 1/GPa. | Needs `pressure_GPa`. No default, as in ASE. |
| `mask` | three 0/1 flags, or `None` | `None` | `None`: the whole cell is scaled (`NPTBerendsen`). Flags: each cell vector is scaled on its own where its flag is 1 (`Inhomogeneous_NPTBerendsen`); the angles stay. | Needs `pressure_GPa`. |
| `fixcm` | `"auto"`, `True`, `False` | `"auto"` | Subtract the mean momentum after the first kick, as ASE. | |
| `dof` | `"auto"`, `"fastmace"`, `"ase"` | `"auto"` | N_f of the kinetic temperature. | |

One deviation from ASE: ASE's NPT step evaluates the forces twice per step,
because the cell scaling invalidates its cache. FastMACE makes one force call
per step, so the first kick uses the previous step's forces at the unscaled
cell. The difference is of order dt·κ·|P0 − P|/τ_P in one half kick. NVT
Berendsen is ASE's step exactly.

## Constraints and refusals

Only `FixAtoms` is supported; any other ASE constraint is refused.

| Record | `FixAtoms` | `fixcm=True` with `FixAtoms` | MC moves |
| --- | --- | --- | --- |
| `Langevin` | yes | refused | yes, without constraints |
| `NVE` | yes | refused | yes, without constraints |
| `CSVR` | yes | refused | yes, without constraints |
| `Berendsen` | NVT only | refused | yes, without constraints |
| `NoseHooverChain` | refused (ASE's class has no constraints, so there is no reference) | — | yes |
| `LangevinBAOAB`, `MTK` | refused (no variable cell with fixed atoms) | — | swaps and transmutations; no displacements |

MC moves with any constraint are refused. A refusal raises `PlanRefused`
with every reason before the first step.

## `IntegratorConfig`

`fastmace.config.IntegratorConfig` is the integrator's internal settings
record. You do not build it: every `fm.integrators` record turns into one
(`record.to_config(seed=...)`) when the `Simulation` is created, and
`Simulation(integrator=...)` takes records only. The fields show up in the
plan, in `metadata.json` and in restart files. Units: fs, GPa, and the
friction in inverse ASE time units (not 1/fs as on the records).

| Option | Values / units | Default | What it does | Notes |
| --- | --- | --- | --- | --- |
| `ensemble` | `"nve"`, `"nvt"`, `"npt"` | `"nvt"` | The ensemble. `"npt"` means a variable cell and a stress calculation. | |
| `scheme` | `"langevin"`, `"baoab"`, `"nve"`, `"csvr"`, `"nhc"`, `"mtk"`, `"berendsen"`, or `None` | `None` | The registered scheme the planner builds. | `None` resolves from the ensemble: `nvt` → `langevin`, `npt` → `baoab`, `nve` → `nve`. |
| `dof` | `"auto"`, `"fastmace"`, `"ase"` | `"auto"` | Degrees-of-freedom convention. | The Langevin schemes ignore it. |
| `parameters` | (name, value) pairs | `()` | Scheme settings (time constants, chain lengths, the Berendsen mask) in the record's units. | Sorted by name. |
| `timestep_fs` | fs, > 0 | 0.5 | MD timestep. | |
| `friction` | inverse ASE time units, > 0 | 0.02 | Langevin friction. NPT uses `T_tau` = 1/`friction`. | The records convert `friction_per_fs`. Unused by other schemes. |
| `seed` | int ≥ 0 | 20261001 | Noise seed. | Always the `Simulation`'s seed when built from a record. |
| `fixcm` | bool | `True` | Remove centre-of-mass motion. | Resolved from the record's `"auto"`. |
| `fixcm_projection` | `"auto"`, `"ase"`, `"mass_weighted"` | `"auto"` | Langevin's projection of the noise under `fixcm`. | The planner resolves `"auto"`. |
| `noise_batch` | int > 0 | 256 | Steps of random numbers drawn and copied to the GPU at once, at most 64 MiB of them (fewer steps above about 5,500 atoms; CAP-T06). | Same numbers in the same order for any value. A restart written with another batch size resumes bitwise. Not set by any record. |
| `pressure_GPa` | GPa | 0.0 | NPT target pressure (ASE `externalstress` = −pressure). | |
| `cell_mode` | `"hydrostatic"`, `"full"` | `"hydrostatic"` | NPT cell mode. | |
| `barostat_tau_fs` | fs, > 0, or `None` | `None` | BAOAB's `P_tau`. | `None`: 20 `T_tau`. |
| `barostat_mass_factor` | > 0 | 1.0 | Factor on ASE's barostat mass. | |
| `ase_trace_drag_all_elements` | bool | `False` | Full cell: ASE 3.27's trace term on all nine drag elements. | For comparisons with ASE only. Not set by any record. |
| `temperature_K` | K, ≥ 0, or `None` | `None` | Thermostat target at the start. | The records always set it, except `NVE`. |
| `external_stress_GPa` | GPa: scalar or six Voigt components, or `None` | `None` | NPT target stress, ASE sign. | Exclusive with a non-zero `pressure_GPa`. |
| `disable_cell_langevin` | bool | `False` | BAOAB: no friction and noise on the cell. | |

## Optimisers: `fm.optimize`

The optimisers are ASE 3.27's `LBFGS` and `FIRE` on the graph engine, with
ASE's arguments and defaults. Lengths are in Å. A relaxation is a
`Simulation` built with `optimizer=` instead of `integrator=`:

```python
opt = fm.optimize.FIRE(cell_filter=fm.optimize.FrechetCell(scalar_pressure_GPa=0.0))
sim = fm.Simulation(atoms, model, optimizer=opt, seed=0)
relaxed, summary = sim.relax(fmax=0.01, steps=5000)
```

### Convergence: `fmax` and `steps`

`Simulation.relax(fmax=0.05, steps=100_000_000)` is ASE's
`Optimizer.run(fmax=, steps=)`. The run converges when the largest per-atom
force norm (after constraints) is below `fmax`, in eV/Å. With a cell filter
the cell rows count too, as in ASE. Convergence is checked before the first
step, so a converged start takes no step. At most `steps` steps follow, one
force call each. A second `relax` call continues with the optimiser's
history. The summary records the steps, whether it converged, the per-step
energies and maximum forces, and the neighbour-list rebuilds.

### `fm.optimize.LBFGS`

| Option | Values / units | Default | What it does | Notes |
| --- | --- | --- | --- | --- |
| `maxstep_A` | Å, > 0 | 0.2 | Largest atom displacement per step, ASE's `maxstep`. | |
| `memory` | int ≥ 1 | 100 | Number of steps kept in the history. | |
| `damping` | > 0 | 1.0 | Scale on the step. | |
| `alpha` | eV/Å², > 0 | 70.0 | Initial guess of the Hessian, α·I. | |
| `cell_filter` | `None`, `FrechetCell(...)` | `None` | `None` keeps the cell fixed. | |

ASE's `use_line_search` is not offered.

### `fm.optimize.FIRE`

| Option | Values / units | Default | What it does | Notes |
| --- | --- | --- | --- | --- |
| `dt` | ASE time units, > 0 | 0.1 | Initial timestep. | 0.1 ASE time units ≈ 1.02 fs. A step-size parameter, as in ASE. |
| `maxstep_A` | Å, > 0 | 0.2 | Largest atom displacement per step, ASE's `maxstep`. | |
| `dtmax` | ASE time units, > 0 | 1.0 | Largest timestep. | ≈ 10.2 fs. |
| `Nmin` | int ≥ 0 | 5 | Downhill steps before the timestep may grow. | |
| `finc` | > 0 | 1.1 | Timestep growth factor. | |
| `fdec` | > 0 | 0.5 | Timestep shrink factor after an uphill step. | |
| `astart` | > 0 | 0.1 | Mixing parameter α after a restart. | |
| `fa` | > 0 | 0.99 | Factor on α per downhill step. | |
| `a` | ≥ 0 | 0.1 | Initial mixing parameter α. | |
| `downhill_check` | bool | `False` | Reject a step that raises the energy and go back, as ASE. | |
| `cell_filter` | `None`, `FrechetCell(...)` | `None` | `None` keeps the cell fixed. | |

ASE's deprecated `maxmove` and `position_reset_callback` are not offered.

### `fm.optimize.FrechetCell`

ASE 3.27's `FrechetCellFilter` on the GPU. The degrees of freedom are the
positions in the original cell and the matrix logarithm of the deformation
gradient; both optimisers run on them unchanged. A step that would strain the
cell by more than about 65 % per element is not applied, and the relaxation
stops with a failure reason.

| Option | Values / units | Default | What it does | Notes |
| --- | --- | --- | --- | --- |
| `scalar_pressure_GPa` | GPa | 0.0 | External hydrostatic pressure, ASE's `scalar_pressure` in GPa. | The energy becomes E + P·V. |
| `hydrostatic_strain` | bool | `False` | Only isotropic strain: the cell shape stays. | |
| `constant_volume` | bool | `False` | Keep the volume fixed. | |
| `mask` | six Voigt 0/1 flags or a 3×3 of 0/1 flags, or `None` | `None` | Which strain components may change, as ASE. | |
| `exp_cell_factor` | > 0, or `None` | `None` | Scale between the cell and position degrees of freedom. | `None` is N, the number of atoms (ASE's default). |

`FixAtoms` with a cell filter is refused. PolarMACE with a cell filter is
refused: relax PolarMACE structures at fixed cell.

### `OptimizerConfig`

`fastmace.config.OptimizerConfig` is the optimiser's internal settings
record. You do not build it: `LBFGS(...).to_config()` and
`FIRE(...).to_config()` produce it when the `Simulation` is created. The
`fmax` and `steps` of `Simulation.relax` replace the config's values at run
time.

| Option | Values / units | Default | What it does | Notes |
| --- | --- | --- | --- | --- |
| `scheme` | `"lbfgs"`, `"fire"` | required | The optimiser. | |
| `parameters` | (name, value) pairs | `()` | The record's settings, in its units. | Sorted by name. |
| `cell_filter` | `None`, `"frechet"` | `None` | `None` keeps the cell fixed. | |
| `cell_filter_parameters` | (name, value) pairs | `()` | `FrechetCell`'s settings. | Needs a `cell_filter`. |
| `fmax` | eV/Å, > 0 | 0.05 | Convergence threshold on the largest force. | `relax(fmax=)` overrides it. |
| `steps` | int ≥ 0 | 100,000,000 | Most optimiser steps (ASE's `run` default). | `relax(steps=)` overrides it. |

### Command line and protocol files

`fastmace relax` and the `[relax]` and `[quench]` tables of protocol files
take optimiser names: `gpu-LBFGS` (default) and `gpu-FIRE` run these records
on the graph engine; `ase-LBFGS` and `ase-BFGS` run ASE's own optimisers on
FastMACE's calculator; bare `LBFGS` and `BFGS` mean the ASE ones. See
[Relaxations and quenches](../guide/relax.md).
