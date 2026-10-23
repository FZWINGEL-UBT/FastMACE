# Phase 4 contracts: schemes, optimisers, constraints

Plan task 4-T01 ([plans/phase-4.md](../../plans/phase-4.md)). SPEC §4.4, §7.5–7.7, §9.3 rule 1,
§10.1–10.3, §10.6. This note is the contract that 4-T02, T03, T05, T06 and T07 build on.

## The scheme registry

SPEC §4.4's support table is data: one record per scheme, in
`fastmace/integrators/registry.py`.

- **Kinds:** `IntegratorRecord`, `OptimizerRecord` and `CellFilterRecord`, all `SchemeRecord`s.
- **Where records live:** one module per scheme, with a module-level `RECORDS` tuple:
  - integrators in `fastmace/integrators/schemes/`;
  - optimisers and cell filters in `fastmace/optimize/schemes/`.

  `registry()` finds them once per process with `pkgutil`. A task adds its scheme by adding
  its module. It changes no shared file: not the planner, not `engine/registry.py`, not
  `simulation.py`.
- **No torch on import:** the record modules import neither torch nor the implementation.
  `implementation` is a `"module:Class"` path, imported when the planner builds the scheme.

| Field | Meaning |
| --- | --- |
| `scheme` | the name in `IntegratorConfig.scheme`, `OptimizerConfig.scheme` or `OptimizerConfig.cell_filter` |
| `implementation` | `"module:Class"` |
| `constraints` | the §4.4 rows it supports: `"fixcm"`, `"fix_atoms"` |
| `refusals` | `{row: reason}` for the rows it does not support, quoted in the refusal |
| `ensembles`, `cell_modes` | integrators: `nve`/`nvt`/`npt`; `fixed`, or `hydrostatic`/`full` for `npt` |
| `cell_filters` | optimisers: `None` (fixed cell) and/or `"frechet"` |
| `non_canonical` | Berendsen: plan warning, report line "equilibration only" |
| `dof_modes`, `ase_dof`, `dof_rule` | owner decision D2, below |
| `keywords` | `"module:function"` returning extra constructor keywords from the config (BAOAB: `externalstress`) |
| `state_keys` | the `state_dict` keys a restart holds (format 4) |
| `buffers` | runtime-state policies of its own buffers: `(name, "save" \| "reset" \| "derived", recompute)` |
| `conserved` | what `output_values()["conserved"]` is, or `None` |
| `ase_class`, `report` | for reports and the equivalence tests |

Registered today: `langevin` and `baoab`, which declare `fixcm` only. 4-T04 adds their
`fix_atoms` row after D1.

**Shared buffer names (4-T01b).** A buffer name is per owner: `integrator.`, `optimizer.` or
`dof.`. Two schemes of one owner may declare the same name only if it means the same thing in
both: the same quantity, units and shape, and the same restart policy. Only one scheme runs per
engine, but restart, capture and the gate read the policy by name alone.

- **Different restart policies:** discovery refuses the pair with `RegistryError`, naming both
  schemes and the buffer. Rename the buffer in one of the records. For example, Berendsen's
  pressure target and MTK's barostat state cannot both be `integrator.pressure`.
- **Same restart policy, different `recompute` functions:** this is allowed. The runtime-state
  registry keeps one policy whose `recompute` lists both functions, sorted and joined with
  ` | `. Example: LBFGS and FIRE's `optimizer.fmax` and `optimizer.status`.
- **A name `engine.registry.POLICIES` already holds:** the table is matched first, by name or by
  pattern, so the scheme must declare the table's restart policy. For example, a record with
  `integrator.eta` must declare Langevin's `"derived"`. A different policy raises
  `RegistryError`, naming the buffer, the scheme and the table entry with its owner.

## Constructors and dispatch

Only the planner builds schemes (SPEC §5), through:

- `build_integrator(state, config, temperature_K=)`: calls
  `Class(state, config, temperature_K=…, **keywords)`;
- `build_optimizer(dof, config)`: calls `Class(dof, config)`;
- `build_dof(state, config)`: `PositionsDOF(state)`, or the cell filter's `Class(state, config)`.

`FixedPlanner.build` and `MinimalPlanner.build_reference` dispatch from the registry. The
branches by ensemble are gone.

## Rule 1

`integrator_refusals(config, fix_atoms=, other_constraints=, mc=, variable_cell=)` and
`optimizer_refusals(config, fix_atoms=, other_constraints=)` return every reason. Each reason has
rule 1 and a stable code:

| Code | When |
| --- | --- |
| `scheme-unknown` | the scheme is not registered (a public record whose implementation has not merged yet) |
| `scheme-ensemble`, `scheme-cell-mode`, `scheme-dof`, `scheme-cell-filter` | outside the record's rows |
| `variable-cell-without-barostat` | a variable cell and a non-`npt` config |
| `constraints` | `FixAtoms` with a scheme that does not declare it (with the record's reason), any other ASE constraint, or `fixcm` with a scheme that does not declare it |
| `fix-atoms-with-fixcm` | `FixAtoms` with `fixcm=True` (D1 (a); `FIX_ATOMS_WITH_FIXCM`) |
| `mc-with-constraints` | MC moves with any constraint, also in the reference tier (decision 10) |

The fixed planner's `_resolve` calls it for every MD plan, the reference tier included. The two
builders call it again on the atoms they build.

## Configs and public records

`IntegratorConfig` gains:

- `ensemble="nve"`;
- `scheme`: `None` resolves to `DEFAULT_SCHEMES` (`nvt` → `langevin`, `npt` → `baoab`,
  `nve` → `nve`), so every earlier config means what it meant;
- `dof`: `"auto"`, `"fastmace"` or `"ase"`. The planner resolves `"auto"` to `registry.DOF_AUTO`
  and records it (decision `IntegratorConfig.dof`), so plans hold no `"auto"`;
- `parameters`: sorted `(name, value)` pairs, hashable and unchanged by a JSON round trip.

The scheme settings travel in `parameters`, in the public record's units. The implementation
reads them with `config.parameter("tau_fs")`.

`OptimizerConfig` is new: `scheme`, `parameters`, `cell_filter`, `cell_filter_parameters`,
`fmax`, `steps`.

Public records (conductor's choice 1):

- `fm.integrators`: `NVE`, `CSVR`, `NoseHooverChain`, `MTK`, `Berendsen`;
- `fm.optimize`: `LBFGS`, `FIRE`, `FrechetCell`.

Each has `to_config`. Time constants have no defaults.

- **`fixcm` and `dof`:** all five new integrator records take them, following D2's "the new
  integrators".
- **FIRE's units:** `dt` and `dtmax` stay in ASE time units, as in ASE.
- **`Simulation`** accepts every record in `fm.integrators.RECORDS`, and resumes them by class
  name. A record without a temperature target (NVE) needs an explicit `workload=`, because
  `max_temperature_K` sizes the skin.

## Integrator contract additions

Additions to `fastmace/integrators/base.py`:

- **`record`:** the integrator's scheme record.
- **`constraint_support`:** the record's `constraints`.
- **`degrees_of_freedom(state)`:** the record's count for `self.config` and `state.n_fixed`.
- **`output_values()["conserved"]`** (`CONSERVED_KEY`):
  - NVE: E_kin + E_pot;
  - CSVR: E_kin + E_pot plus the transferred energy, as ASE's `Bussi`;
  - NHC and MTK: the extended energy.
- **`state_dict()` keys:** exactly the record's `state_keys`.

## Degrees of freedom (D2, recommended (a))

`registry.degrees_of_freedom(natoms, n_fixed=, fixcm=, dof=, ase_dof=)`:

- **`dof="auto"`:** `DOF_AUTO`, `"fastmace"` under D2 (a) (`"ase"` under (b)). The records'
  default `dof_modes` is `("fastmace", "ase")`.
- **`dof="fastmace"`:**
  - 3(N − n_fixed) with fixed atoms (no centre-of-mass projection then);
  - 3N − 3 under `fixcm`;
  - 3N otherwise.
- **`dof="ase"`:** the scheme's ASE class:
  - `ase_dof="constraints"`: 3(N − n_fixed). This is `Bussi`, and `Atoms.get_temperature`
    for Berendsen;
  - `"3n"`: 3N (NHC, MTK).

Langevin and BAOAB keep 3-T22c's count bitwise (`dof_rule="langevin"`). `io.trajectory.degrees_of_freedom`
now asks the record.

## Optimisers and the DOF view

`fastmace/optimize/base.py`. An optimiser never sees atoms or a cell: it moves a `DOFView`.

| `DOFView` member | Meaning |
| --- | --- |
| `rows` | number of 3-vectors: N, or N + 3 with a cell filter |
| `variable_cell` | `set_x` changes the cell (the engine needs stress and `set_cell`) |
| `x` | flat `(3 rows,)` float64 device tensor, the current values |
| `set_x(x)` | move in place; masked components keep their values |
| `gradient()` | of `energy()`, masked components zero (minus ASE's constraint-adjusted forces) |
| `energy()` | `(1,)`: E_pot, plus p V with a cell filter |
| `mask` | bool `(3 rows,)` of free components, or `None` |
| `refresh()` | derived quantities after the force call (the filter's virial gradient) |
| `max_force()` | ASE's `fmax` measure: the largest row norm of the gradient |
| `buffers()` | what `set_x` and `refresh` write |

- **`PositionsDOF(state)`:** fixed cell. `x` is a view of `state.positions`; `FixAtoms` comes
  in through `state.free_mask`.
- **4-T07's `FrechetCellFilter`** (`fastmace/optimize/cell.py`): implements the same view, so FIRE
  and LBFGS run unchanged on it.
  - `x` is ASE's (N + 3) × 3 vector: positions in the original cell, then the cell rows
    `ecf · logm(cf · F)`. ASE folds `UnitCellFilter`'s `cell_factor` cf = N into the log, so
    the cell rows carry a constant `ecf · ln(N) · I`. FastMACE evaluates both directions in a
    shifted form: `expm(X / ecf − ln(cf) I)`, and the log of F itself. Same values, but on
    matrices of norm ~|log F| instead of ~ln N. Against 40-digit mpmath, ASE's own form costs
    2e-14 relative in F; the shifted form is exact to 1e-17.
  - Device matrix functions, no host syncs and no LAPACK:
    - `expm_3x3` from `integrators/baoab.py`, now for any square size;
    - `logm_3x3`, by inverse scaling and squaring. It is used only with a `mask`; without one,
      log F is `X / ecf − ln(cf) I` itself;
    - ASE's nine `expm_frechet` calls as one 6 × 6 exponential: the adjoint of `L_exp(A, ·)`
      is `L_exp(Aᵀ, ·)`;
    - 3 × 3 inverses by cofactors.
  - `refresh` runs after each force call and in `OptimizerStepper.start`. It computes `x`'s
    atom rows, the gradient (ASE's `get_forces`) and the energy E + pV.
  - `set_x` moves positions and cell. It does nothing for a move to the current `x` (a
    converged step) or for one outside |log F| ≤ 0.5 per element. In the second case
    `cell_ok` drops, `max_force` becomes NaN, and `driver.relax` raises `FloatingPointError`
    with the filter's `failure_reason`.
  - `FixAtoms` and non-periodic cells are refused (D1).

The `Optimizer(dof, config)` methods:

- `set_fmax(fmax)`;
- `start()`;
- `propose()`: the step from the gradient and the history, then `dof.set_x`;
- `update()`: after the force call and `dof.refresh()`: the history, `max_force`, `converged`;
- `converged`: a device bool;
- `max_force()`, `buffers()`, `state_dict()` (tests only, no optimiser restart files),
  `output_values()`.

`propose` and `update` take no argument: the optimiser is bound to its view at construction,
at fixed addresses. This is a deviation from the plan's `propose(state)`. 4-T05's driver
composes `propose`, then the neighbour check and the force call, then `refresh`, then `update`.
The host reads `converged` and `max_force` each step (choice 5).

## `FixAtoms` in the state

`SimulationState.from_atoms` turns every `FixAtoms` into:

- `fixed_indices`: sorted, unique, host;
- `free_mask`: float64 `(N, 1)`, 1 for free atoms and 0 for fixed ones, or `None`.

Any other constraint raises. `n_fixed` counts the fixed atoms. The runtime-state registry
lists `state.free_mask` as derived. Whether a scheme takes fixed atoms is rule 1's decision.

## Restart format 4

- `fastmace_restart = 4`.
- The integrator record's config holds `scheme`, `dof` and `parameters`.
- The physical record adds `fixed_atoms`. A resume into a state with other fixed atoms is
  refused, as another structure is.
- Writing checks that the integrator's `state_dict` keys equal its record's `state_keys`:
  - NHC: `eta`, `p_eta`;
  - MTK: the cell momenta and the barostat chain;
  - CSVR: the transferred energy (its noise rows and cursor are `noise` and `noise_next`, as
    Langevin's);
  - NVE and Berendsen: none.
- Readers: `restart.fixed_atoms(file)` and `restart.scheme(file)`.
- Files of versions 0–3 load unchanged: their scheme is the ensemble's default, and they have
  no fixed atoms.

## `fastmace.ase` argument lists

`fastmace.ase.md.ASE_ARGUMENTS` lists the eight ASE 3.27 MD classes' constructor arguments,
along their class chain, as data. `ASE_OPTIMIZER_ARGUMENTS` does the same for LBFGS and FIRE
(D5). `tests/cpu/test_contracts_4.py` checks both against `inspect.signature`; ASE 3.29 has
the same lists. 4-T10 splits each list into `SUPPORTED` and `REFUSED`.

## Owner decisions and where to change them

| Decision | Recommendation followed | Where |
| --- | --- | --- |
| D1 | `FixAtoms` with NVE, Langevin, CSVR, NVT Berendsen, FIRE and LBFGS at fixed cell; refused for NHC, MTK and every variable cell; `fixcm` with `FixAtoms` refused | each record's `constraints` and `refusals`; `registry.FIX_ATOMS_WITH_FIXCM` |
| D2 | `dof="auto"` resolves to `"fastmace"`; `dof="ase"` for the equivalence tests | `registry.DOF_AUTO`; `registry.degrees_of_freedom` |
| D5 | LBFGS and FIRE classes, `FrechetCellFilter` only | `ASE_OPTIMIZER_ARGUMENTS`; the optimiser records' `cell_filters` |
