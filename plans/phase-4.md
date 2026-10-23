# Phase 4: Physics

Status: **draft**, 23 October 2026, written while phase 3b runs (3b-T01 to
T05 and T08 merged; T06, T07, T09 and T10 open). D1–D6 below are open.
Spec: [SPEC.md](../SPEC.md) §16 row "4. Physics", and the sections it cites:
- §4.4 and §11.3: constraints, `FixAtoms`, fixed centre of mass, degrees of
  freedom;
- §7.5–§7.7: the CLI (`fastmace relax`), the `Simulation` inputs
  (`integrator=` or `optimizer=`), the `fastmace.ase` mapping and its
  acceptance;
- §9.2–§9.3: rules 1, 10 and 11 for the new integrators and optimisers;
- §10.1–§10.3: the integrator contract, the list, the relaxations;
- §10.6: restart with the new integrators' state;
- §12.1–§12.3: the reference policy, ASE equivalence, the gate;
- §18: decisions 6, 7, 10, 26, 37–39 and 40–43.

Why it matters:
- Langevin friction perturbs the dynamics. Sodium transport needs NVE or a
  deterministic thermostat (CSVR, NHC) on the graph engine.
- Every relaxation and quench runs ASE's LBFGS on `fm.calc`, the compiled
  tier. 3-T06 measured that tier 1.3–4.7× slower per step than graph. The ten
  crystals' 210 quench frames took 1.1 h on 13 October (ResearchModule
  `studies/annealing/README.md`).
- `FixAtoms` and variable-cell relaxations are refused today
  (`ase/md.py` `_check_constraints`, `state.py` line 250,
  `planner/minimal.py` line 949).

## Scope

- **New integrators** (§10.2) on the graph engine, each a port of its ASE
  3.27 class:
  - NVE: `VelocityVerlet`;
  - CSVR: `Bussi`;
  - NHC: `NoseHooverChainNVT`;
  - MTK: `IsotropicMTKNPT` and `MTKNPT`;
  - Berendsen: `NVTBerendsen`, `NPTBerendsen` and
    `Inhomogeneous_NPTBerendsen`, flagged non-canonical.
- **GPU optimisers** (§10.3): FIRE and LBFGS as ASE 3.27 defines them, fixed
  cell and with a cell filter (D5), driven on the graph engine, with ASE's
  `fmax` convergence.
- **`FixAtoms`** where D1 puts it, with the degrees of freedom it implies (D2).
- **Restart v4:** the new integrators' state and the constraint record
  (§10.6).
- **Front ends:** the `fastmace.ase` classes (§7.7), `fm.md(ensemble="NVE")`,
  `fastmace relax` and the protocols' relaxations (D4).
- **PolarMACE stress items carried over from 3b-T08:** the frozen k-set in
  `ReferenceCalculator.check_stress_fix`, and the gate's stress rule (D6).

## Out of scope

- **Qualification of the new barostats and thermostat settings.** MTK and
  NPT Berendsen keep the "experimental: not yet qualified" line, as BAOAB
  does (§10.2). That is an experiment, not library work.
- **Stochastic cell rescaling and replica exchange:** 1.x (§10.2).
- **RATTLE and other ASE constraints:** refused in 1.0 (§4.4).
- **MC with constraints:** stays refused (decision 10).
- **ResearchModule `main.py`:** no change in phase 4. A GPU quench option there
  is a follow-up after checkpoint 3.
- **PolarMACE NPT qualification.** D6 changes the gate rule only.
- **Judging physics.** NVE drift and quench minima are recorded. Only
  agreement with the reference is acceptance (decision 6).

## Owner decisions

**Decided (owner, 23 October 2026): the recommendation (a) of each**, D1–D6,
in the owner's words "For the other decisions I trust your recommendations".
They are SPEC §18 decisions 44 (D1) to 49 (D6). The text below keeps each
option for the record.

**D1. Where `FixAtoms` is supported.** §4.4 says "all 1.0 integrators and
optimisers". ASE 3.27 is not uniform: `NoseHooverChainNVT`, `MTKNPT` and
`IsotropicMTKNPT` raise "Current implementation does not support
constraints" (`ase/md/nose_hoover_chain.py` lines 285 and 553), so they have
no reference with fixed atoms.
- **Recommended (a):**
  - `FixAtoms` with the fixed-cell integrators whose ASE class takes it:
    NVE, Langevin, CSVR and NVT Berendsen;
  - and with FIRE and LBFGS at fixed cell;
  - refused, with the reason, for NHC and MTK (ASE refuses them too) and for
    every variable cell: BAOAB, MTK, NPT Berendsen and the cell filter;
  - `fixcm=True` with `FixAtoms` refused. The fixed atoms already pin the
    frame, and ASE's `fixcm` displacement moves fixed atoms before the
    constraint resets them;
  - SPEC §4.4's table amended to match.
- **Alternatives:**
  - (b) also NHC, against a float64 CPU reference written from Martyna–Klein–
    Tuckerman with 3(N − n_fixed) degrees of freedom;
  - (c) also variable cell, with fixed atoms held in fractional coordinates
    (they move affinely with the cell, as ASE's filters treat them).
- **Blocks:** 4-T04. The other tasks start and add their `FixAtoms` row to the
  support table after D1.

**D2. Degrees of freedom and the centre of mass of the new thermostats.**
- ASE's counts: `Bussi` uses `atoms.get_number_of_degrees_of_freedom()`
  (3N minus constraints), NHC and MTK use 3N, Berendsen uses
  `atoms.get_temperature()`. None removes the centre of mass.
- Decision 37 made Langevin under `fixcm` count 3N − 3. The difference
  3N/(3N − 3) is 0.46 % on 216 atoms and 0.05 % on 1944.
- **Recommended (a):**
  - with `fixcm=True` (the default, as Langevin) the new integrators remove
    the total momentum at the start, project it out each step against
    float drift, and count 3N − 3 in their targets and temperatures;
  - with `FixAtoms`, 3(N − n_fixed) and no projection (D1);
  - `dof="ase"` reproduces ASE's counts. The ASE-equivalence tests use it.
- **Alternative (b):** ASE's counts by default, 3N − 3 as the option.
- **Blocks:** only the default value in 4-T02 and 4-T03. Both conventions are
  built either way.

**D3. "Reproduces the annealing-benchmark minima" (§16 row 4).**
- The frames: 210 `captured.traj` frames (NaCl and the nine production
  crystals, 21 each, 3×3×3), in git under ResearchModule
  `Experiments/Langevin_Thermostat/results/anneal-*-rtx5060ti-20261010/`.
  MH-1 `matpes_r2scan`, hybrid float32.
- The 13 October LBFGS–BFGS comparison set the scale: max |ΔE| 0.025
  meV/atom, and every atom within 0.1 Å in 208 of 210 frames.
- **Recommended (a):**
  - GPU LBFGS (fmax 0.01 eV/Å, memory 100, maxstep 0.2 Å, fixed cell, at most
    5,000 steps) against ASE's LBFGS on `fm.calc`, with the same force field
    and precision. Only the optimiser differs;
  - pass: all 210 converge; |ΔE| ≤ 0.025 meV/atom and the same structure
    (every atom within 0.1 Å), each in at least 20 of 21 frames per crystal
    (energy allowance added by the owner on 23 October 2026, SPEC decision 46);
  - the 13 October research-engine minima (`quench.csv`, `comparison.csv`)
    compared and recorded, not judged: other stack, and `lbfgs.traj` is on the
    NAS only.
- **Alternatives:** (b) against the 13 October minima directly (needs the NAS
  for the structures, so it waits for the workstations); (c) a subset:
  NaCl, NaNbO3 and NbCl5, 63 frames.
- **Blocks:** 4-T08.

**D4. Protocol relaxations on the GPU.**
- **Recommended (a):** once 4-T08 passes, `[relax]` and `[quench]` use GPU
  LBFGS by default. `"ase-LBFGS"` and `"ase-BFGS"` stay as options, and
  `protocol.json` records which ran.
- **Alternative (b):** GPU LBFGS opt-in; ASE stays the default until a
  production quench comparison.
- **Blocks:** only the default line in 4-T09.

**D5. Optimisers in `fastmace.ase` and the cell filter.** §7.7 maps only the
MD classes.
- **Recommended (a):**
  - `fastmace.ase.LBFGS` and `fastmace.ase.FIRE` with ASE 3.27's arguments
    (signature test as for the MD classes);
  - they take `Atoms` or `ase.filters.FrechetCellFilter(atoms, …)`, ASE's
    recommended filter. `UnitCellFilter`, `ExpCellFilter` and `StrainFilter`
    are refused with the reason;
  - `fm.optimize.FrechetCell` is the same filter in `Simulation`.
- **Alternatives:** (b) no ASE optimiser classes: `Simulation(optimizer=)` and
  `fastmace relax` only; (c) (a) plus `UnitCellFilter`.
- **Blocks:** the optimiser part of 4-T10.

**D6. The gate's stress rule for PolarMACE** (3b-T08, "What NPT with
PolarMACE would need").
- PolarMACE's float32 stress scatter is 1–3e-7 eV/Å³ (hybrid 0.9–2.1× the
  stock float32 yardstick), the order of the 1e-7 floor (decision 7). Some
  configurations fail the gate on rounding alone (1b-T23: 1.2e-7 on NaCl).
- **Recommended (a):** a PolarMACE stress floor of 5e-7 eV/Å³ (0.08 MPa), in
  the reference record, for the stress part only. The energy and force floors
  stay.
- **Alternatives:**
  - (b) more gate configurations per cell mode (4 rattled and 4 strained), the
    yardstick the largest over the draws;
  - (c) float64 PolarMACE for NPT.
- **Blocks:** 4-T12.

## Conductor's choices (the owner may override)

1. **Records and units** (§7.3): fs, K and GPa at the API, ASE's argument
   names otherwise.
   - `fm.integrators.NVE(timestep_fs, fixcm=True)`;
   - `CSVR(timestep_fs, temperature_K, tau_fs, fixcm=True, dof="auto")`;
   - `NoseHooverChain(timestep_fs, temperature_K, tdamp_fs, tchain=3, tloop=1)`;
   - `MTK(timestep_fs, temperature_K, pressure_GPa, tdamp_fs, pdamp_fs,
     tchain=3, pchain=3, tloop=1, ploop=1, cell_mode="hydrostatic"|"full")`;
   - `Berendsen(timestep_fs, temperature_K, taut_fs, pressure_GPa=None,
     taup_fs=None, compressibility_per_GPa=None, mask=None, fixcm=True)`;
   - `fm.optimize.LBFGS(maxstep_A=0.2, memory=100, damping=1.0, alpha=70.0)`
     and `FIRE(...)` with ASE's defaults; `fm.optimize.FrechetCell(
     scalar_pressure_GPa=0.0, hydrostatic_strain=False, constant_volume=False,
     mask=None, exp_cell_factor=None)`.
   - Time constants have no defaults, as in ASE: physics is never chosen
     silently.
2. **ASE equivalence** (§12.2), the owner's limits for Langevin (2-T14e):
   - MH-1 `matpes_r2scan`, NaCl 3×3×3, plain cuEq float64, 4,000 steps;
     |ΔT| < 1e-6 K and |ΔPE| < 1e-7 meV/atom per 50 K bin (ramps) or per
     500-step block (constant targets);
   - each new integrator against its ASE class with MACE's calculator,
     `dof="ase"`;
   - the deterministic integrators have no shared noise to hold the two runs
     together. If chaotic divergence exceeds the limits before 4,000 steps,
     the test compares the first 1,000 steps at the same limits and records
     the growth rate. That goes to checkpoint 2;
   - CPU tests on the tiny model against ASE, as
     `test_langevin_trajectory_matches_ase` does.
3. **CSVR noise:** ASE's `Bussi` stream from a seeded host generator: one
   `standard_normal(1)` and one `standard_gamma(0.5 (ndof − 1))` per step, in
   that order, prefetched in `NOISE_BATCH` rows with the cursor in the restart
   (§10.6). `rng=` is refused in the shim; `seed=` replaces it.
4. **NHC and MTK masses:** Q and W from the construction temperature, as ASE.
   A schedule moves only the device target kT, and the report says so.
5. **Optimiser driver:**
   - one force replay per step; the host reads the converged flag and fmax
     each step, as ASE checks. That is tens of µs against 3.75–29 ms per
     force call;
   - optimiser state in float64;
   - no optimiser restart files. Relaxations take minutes, and a quench
     already resumes per frame (`quench.csv`).
6. **Mode and skin for optimisers:** measured in 4-T05, graph with rebuilds
   against compiled, on NaCl and NbCl5 quench frames. A step moves an atom by
   up to 0.2 Å against the 0.3 Å default skin. Rule 10 records the choice.
7. **Recorded, not acceptance:**
   - NVE energy drift, float32 hybrid, NaCl and NbCl5 3×3×3, 20 ps at 1 fs;
   - step time of each new integrator against Langevin;
   - GPU against ASE quench time on the 210 frames.
8. **Sampling checks:** the kinetic-energy distribution of CSVR and NHC
   against the canonical gamma distribution (Kolmogorov–Smirnov, fixed seed)
   on the tiny model on the CPU. They run in the acceptance tier, not at
   merges.
9. **Acceptance hardware** (decision 39): items in full on the RTX 5060 Ti.
   The one Ada spot check runs on the Modal L4 (at most $3) unless the
   workstations are back first.
10. **SPEC amendments with the decisions:** §4.4's table, §7.7's mapping and
    optimiser classes, §10.2 and §10.3, §10.6 (format v4), §12.2's
    equivalence row, and the §18 entries.

## Tasks

| Task | Size | GPU | After | Files (main ones) |
| --- | --- | --- | --- | --- |
| 4-T01 Contracts | S–M | no | — | `integrators/base.py`, `integrators/__init__.py`, `config.py`, `optimize/base.py`, `optimize/__init__.py`, `state.py`, `planner/minimal.py` (rule 1 table), `io/restart.py` (v4), `io/trajectory.py` (DOF), `ase/md.py` (argument lists), `docs/design/physics.md` |
| 4-T02 NVE, CSVR, Berendsen | M–L | yes | T01 | `integrators/nve.py`, `integrators/csvr.py`, `integrators/berendsen.py` |
| 4-T03 NHC and MTK | L | yes | T01 | `integrators/nose_hoover.py`, `integrators/mtk.py` |
| 4-T04 `FixAtoms` | M | yes | T01, T02, D1 | `state.py`, `integrators/langevin.py`, `kernels/triton/fused_integrator.py`, `planner/minimal.py`, `io/trajectory.py`, `ase/md.py` |
| 4-T05 GPU LBFGS and the optimiser driver | L | yes | T01 | `optimize/lbfgs.py`, `optimize/driver.py`, `engine/graph.py`, `simulation.py` (`optimizer=`, `relax`), `planner/minimal.py` (rules 10, 11) |
| 4-T06 GPU FIRE | M | yes | T01; wiring after T05 | `optimize/fire.py` |
| 4-T07 Variable-cell relaxations | M–L | yes | T05, T06 | `optimize/cell.py`, `engine/graph.py`, `planner/minimal.py` |
| 4-T08 Annealing-benchmark minima | M | yes | T05, D3 | `tools/validation/anneal_minima.py`, `docs/validation/4-T08.md` |
| 4-T09 Protocol relaxations, `fastmace relax` | S–M | smoke | T05, T07, 3b-T07; D4 for the default | `protocols/quench.py`, `protocols/runner.py`, `protocols/profiles.py`, `cli/relax.py`, `cli/__main__.py` |
| 4-T10 Front ends | M | yes (§7.7 A/B) | T02, T03; T07 and D5 for the optimisers | `ase/md.py`, `ase/optimize.py`, `quick.py`, `cli/md.py`, `cli/init.py`, `report.py` |
| 4-T11 Frozen k-set in `check_stress_fix` | S | yes (POLAR-1-S) | — | `reference/calculators.py`, `reference/records.py`, `tools/force_stress_gate.py` |
| 4-T12 PolarMACE stress-gate rule | S | yes | T11, D6 | `reference/records.py`, `planner/selfcheck.py` |
| 4-T13 Acceptance runbook and report | M | yes | all | `docs/validation/phase-4-runbook.md`, `docs/validation/phase-4.md` |

**4-T01 Contracts.**
- `Integrator` gains:
  - `constraint_support`: the §4.4 rows it declares (`fixcm`, `fix_atoms`);
  - `degrees_of_freedom(state)` under D2's conventions (`"auto"` resolves
    after D2);
  - a conserved quantity in `output_values` (NVE: E_kin + E_pot; CSVR: plus
    the transferred energy, as ASE's `Bussi`; NHC and MTK: the extended
    energy).
- An `Optimizer` interface:
  - `propose(state)`: the step from forces and history, positions (and cell)
    moved, in the graph;
  - `update(state)` after the force call: history, fmax and a device
    `converged` flag;
  - `variable_cell`, `max_force(state)` with constraints applied,
    `state_dict()` for tests.
- The records of choice 1, and `IntegratorConfig` with `ensemble` in
  `nve`/`nvt`/`npt` plus a `scheme` name.
- `SimulationState`: `FixAtoms` becomes a device mask plus host indices.
  Other constraints still raise.
- Planner rule 1: the support table as data (integrator or optimiser, cell
  mode) × (`fixcm`, `FixAtoms`). Every refusal names its reason. MC with
  constraints stays refused (decision 10).
- Restart format v4:
  - per-scheme state keys: NHC `eta`, `p_eta`; MTK the strain or cell
    momenta and the barostat chain; CSVR its noise rows, cursor and
    transferred energy; NVE and Berendsen none;
  - the scheme in the identity, the fixed indices in the physical record;
  - v0–v3 files still load.
- `fastmace.ase` argument lists for the eight ASE classes as data, with the
  `inspect.signature` test against ASE 3.27.
- Merges first. T02, T03, T05 and T06 start at once on its contract.

**4-T02 NVE, CSVR and Berendsen.**
- NVE: `VelocityVerlet`'s half kick and drift in `first_half`, the second
  kick in `second_half`. `fixcm` per D2.
- CSVR: ASE 3.27 `Bussi`'s scaling, in its order. The noise of choice 3.
- Berendsen:
  - `NVTBerendsen`: λ from the current kinetic temperature and `taut`;
  - `NPTBerendsen` and `Inhomogeneous_NPTBerendsen` (`mask`):
    `compressibility` required, the cell scaled from the stress;
  - non-canonical: a plan warning and the report line "equilibration only".
- `FixAtoms` rows after D1.
- **Tests:**
  - CPU float64 on the tiny model against each ASE class on the same
    calculator, positions within 1e-12 Å after 200 steps;
  - GPU equivalence (choice 2);
  - restart round trip, and interrupted against uninterrupted, bitwise on CPU
    float64;
  - runtime-state and derived-state registries (CSVR's ndof).

**4-T03 NHC and MTK.**
- A port of ASE 3.27 `nose_hoover_chain.py`: `NoseHooverChainNVT` (`tchain`,
  `tloop`), `IsotropicMTKNPT` and `MTKNPT` (`pchain`, `ploop`).
- The chain updates are device scalar operations, unrolled at capture. No
  host sync inside a step (§10.1).
- MTK uses the BAOAB stress path: `variable_cell=True`, stress, the
  deformation-aware neighbour test, the D3 virial.
- Q and W as choice 4. `FixAtoms` refused (D1).
- **Tests:**
  - CPU float64 against the ASE classes, hydrostatic and full cell;
  - GPU equivalence (choice 2);
  - restart with the chain state, bitwise on CPU float64;
  - the extended energy recorded over 20 ps.

**4-T04 `FixAtoms` (D1).**
- The mask in Langevin's PyTorch stage and in the fused Triton stage
  (OPT3-T03). ASE's semantics: fixed atoms keep their positions, their momenta
  are zero, and the noise stream is drawn for every atom, as ASE draws it.
- The NVE, CSVR and NVT Berendsen rows on T02's integrators.
- Degrees of freedom 3(N − n_fixed) in the targets and thermo output (D2).
- Refusals: `fixcm=True` with `FixAtoms` (D1), and MC with constraints
  (decision 10).
- `fastmace.ase` accepts `FixAtoms` where the table allows it. An observer
  that changes constraints still raises.
- The observer cache's `get_forces()` applies the constraint, as ASE does.
  The gate compares every atom's forces, since constraints don't change the
  force field.
- **Tests:**
  - CPU against ASE `Langevin`, `VelocityVerlet` and `Bussi` with
    `FixAtoms`;
  - GPU equivalence for Langevin with `FixAtoms`;
  - restart with the indices;
  - every refusal and its reason;
  - the fused stage bitwise against the PyTorch stage with the mask.

**4-T05 GPU LBFGS and the optimiser driver.**
- `optimize/lbfgs.py`: ASE 3.27 `LBFGS` exactly.
  - Two-loop recursion, H0 = 1/alpha, `damping`, and `determine_step`'s
    scaling of the whole step to `maxstep`.
  - `use_line_search=True` is refused.
  - `FixAtoms` through the constraint-adjusted forces, as ASE's
    `atoms.get_forces()` gives them.
- `optimize/driver.py` drives the graph engine: `propose`, then the neighbour
  check, the force replay and `update` (choice 5). It records steps, time,
  fmax and energy, as `quench.relax`'s summary does.
- `Simulation(atoms, ff, optimizer=fm.optimize.LBFGS(...))` and
  `sim.relax(fmax=0.01, steps=5000)`, returning the relaxed `Atoms` and the
  summary.
- Rules 10 and 11 for optimisers; mode and skin measured (choice 6).
- **Tests:**
  - CPU float64, tiny model: positions against ASE `LBFGS` on the same
    calculator within 1e-12 Å at every step for 50 steps, and the same step
    count to fmax;
  - GPU: one NaCl and one NbCl5 frame from `captured.traj` against ASE on
    `fm.calc`, hybrid float32; energies and steps recorded.

**4-T06 GPU FIRE.**
- ASE 3.27 `FIRE` exactly: `dt`, `maxstep`, `dtmax`, `Nmin`, `finc`, `fdec`,
  `astart`, `fa`, `a`.
- `downhill_check` restores positions with `where` masks, so it stays
  capturable. `position_reset_callback` is refused.
- Written and CPU-tested against ASE on an eager loop over T01's contract.
  Wired into T05's driver after T05 merges.
- **Tests:** as T05.

**4-T07 Variable-cell relaxations.**
- `optimize/cell.py`: ASE 3.27 `FrechetCellFilter` on the device:
  - the deformation gradient, its log and exp (`expm_3x3` from
    `integrators/baoab.py`; ASE uses SciPy);
  - `exp_cell_factor` defaulting to N, `scalar_pressure`,
    `hydrostatic_strain`, `constant_volume`, `mask`;
  - the virial from the stress, and ASE's convergence on the (N + 3) × 3
    gradient.
- Both optimisers. The stress path and the deformation-aware rebuild
  (rule 10). The D3 list rebuilds on strain (3b-T02).
- PolarMACE runs. Its k-space cutoff steps (3b-T08) go into the report as a
  known limitation.
- `FixAtoms` with a cell filter is refused (D1).
- **Tests:**
  - CPU float64 against ASE `LBFGS` and `FIRE` with `FrechetCellFilter`;
  - GPU: the ten benchmark crystals 3×3×3, hybrid float32, against ASE on
    `fm.calc`: cell lengths within 1e-3 Å, volume within 0.01 %, energy
    within 0.025 meV/atom; NbCl5 once with D3.
  - The runbook fixes the final tolerances.

**4-T08 Annealing-benchmark minima (D3).**
- The 210 frames of D3, GPU LBFGS against ASE LBFGS on `fm.calc`, with D3's
  pass rule.
- The ASE side took about 1.1 h on 13 October. It runs locally, or on Modal
  if the local GPU is needed for timing.
- Quench time GPU against ASE, one block (decision 39).
- Report `docs/validation/4-T08.md`. It is an acceptance item.

**4-T09 Protocol relaxations and `fastmace relax`.**
- The optimizer names in `[relax]` and `[quench]`: `"LBFGS"` and `"FIRE"`
  (GPU), `"ase-LBFGS"` and `"ase-BFGS"`. The default follows D4.
- One engine per quench, reused across its 21 frames: positions are set as an
  event, so there is no compile or capture per frame.
- 3b's NPT quench frames relax at their NPT cells with the cell fixed
  (decision 40).
- `fastmace relax STRUCTURE --model … --head … --fmax … --optimizer …
  [--cell-filter frechet --pressure GPa]` and `fastmace quench RUN_DIR`
  (§7.5).
- **Tests:** CPU; GPU smoke: the short anneal protocol end to end with the GPU
  quench.

**4-T10 Front ends.**
- `fastmace.ase`: `VelocityVerlet`, `Bussi`, `NoseHooverChainNVT`,
  `IsotropicMTKNPT`, `MTKNPT`, `NVTBerendsen`, `NPTBerendsen` and
  `Inhomogeneous_NPTBerendsen`.
  - ASE 3.27 arguments and units (`pressure_au`, `tdamp`, `pdamp`, `taut`,
    `taup`, `compressibility_au`, …);
  - `rng` refused, `seed=` instead;
  - `md.py`'s `LATER` list emptied.
- Optimiser classes per D5: `fastmace.ase.LBFGS` and `FIRE`.
- `fm.md(ensemble="NVE")` runs NVE (the `nve-phase-4` refusal goes). The
  other integrators go through `integrator=`. `fastmace md --ensemble NVE`.
  The `fastmace init` template lists the integrators.
- Report lines: Berendsen "non-canonical: equilibration only"; MTK
  "experimental: not yet qualified", as BAOAB.
- **Tests, §7.7's acceptance for the new classes:**
  - each argument list against `inspect.signature`;
  - a ported script gets the same plan as the equivalent `Simulation`;
  - its `trajectory=` file reads back frame for frame;
  - observer changes give the same state as `Simulation` events;
  - the same steady-state step time as `Simulation` in alternating A/B
    blocks, one block each unless the gap is under 3 %.

**4-T11 Frozen k-set in `check_stress_fix`.**
- `ReferenceCalculator.check_stress_fix` holds the unstrained cell's included
  k-set during its finite differences, as `tools/polar_stress_fd.py
  --frozen-kset` (`KSetProbe(frozen=True)`) does.
- The self-test then validates the stress formula on the run's own structure.
  The NaCl-only rule (decision 26, `tools/force_stress_gate.py`) and the open
  finding in `reference/records.py` are retired.
- **Tests:**
  - CPU: the existing `test_reference_policy.py` cases;
  - GPU: POLAR-1-S on the 3b-T08 NbCl5 gate configuration passes at
    h = 1e-4 (3b-T08: 8.3e-9 eV/Å³ frozen against 4.36e-4 live); NaCl
    unchanged.
- A short addendum to `docs/validation/3b-T08.md` instead of a new document.

**4-T12 PolarMACE stress-gate rule (D6).**
- Under (a): the PolarMACE stress floor in the reference record, read by
  `planner/selfcheck.py`'s gate and recorded in the `GateRecord`. `FLOORS`
  keeps the general values.
- **Tests:** CPU unit; GPU: the POLAR-1-S NbCl5 hydrostatic gate passes on the
  hybrid, and NaCl is unchanged.

**4-T13 Acceptance (decision 39).**
- **Step 0, preflight.** The GPU slow and acceptance tiers on the frozen tree
  on the RTX 5060 Ti, before any timed item. Stale tests are corrected
  test-side and listed in the report. Phase 3 found 7 at its acceptance.
- §16 row 4's items:
  1. ASE equivalence: NVE, CSVR, NHC, MTK (hydrostatic and full), Berendsen
     NVT and NPT, Langevin with `FixAtoms`; GPU LBFGS and FIRE against ASE,
     fixed and variable cell;
  2. the annealing-benchmark minima (D3). It cites 4-T08's runs and the code
     difference in `optimize/` since; it reruns only if that difference is not
     empty;
  3. §7.7's acceptance for the new classes (T10's tests);
  4. restart round trips and interrupted against uninterrupted for each new
     integrator.
- Plus a no-regression check against the `fastmace-3b` tag: Langevin and
  BAOAB bitwise on CPU float64 and in reproducible mode, and their step time
  in alternating A/B blocks.
- Recorded only: choice 7's measurements and choice 8's sampling checks.
- One Ada spot check (choice 9). Timing repeats sized to the effect.
  Cross-process controls use the 3 % noise floor or are left out.
- When the Ada tier has passed at the freeze, the local leg reruns only: the
  tests corrected at acceptance, the GPU slow tier, and the items' own runs.
- The runbook `docs/validation/phase-4-runbook.md` comes first and is
  accepted before the freeze. Report `docs/validation/phase-4.md`; the owner
  tags `fastmace-4`.

## Order and parallelism

```text
Wave 0   T01 contracts   T11 frozen k-set
Wave 1   T02 NVE/CSVR/Berendsen   T03 NHC/MTK   T05 LBFGS + driver   T06 FIRE (wiring after T05)
Wave 2   T04 FixAtoms (T02, D1)   T07 variable cell (T05 T06)   T08 minima (T05, D3)   T12 polar gate (T11, D6)
Wave 3   T09 protocols + relax (T05 T07, after 3b-T07)   T10 front ends (T02 T03; T07 for optimisers)
Wave 4   T13 acceptance
```

- **Start without any decision** (prestart tonight, read-only until go):
  - T01 and T11 at once;
  - T02, T03, T05 and T06 on T01's merge;
  - T07 after T05 and T06;
  - T09 and T10, except their D4 and D5 parts.
- **Wait for a decision:** T04 (D1), T08 (D3), T12 (D6). D2 sets one default
  in T02 and T03, D4 one in T09; neither blocks the work.
- **Critical path:** T01 → T05 → T07 → T09 and T10 → T13. In the local GPU
  queue, T05 goes first, then T03, T02, T06, T04.
- **Phase 3b first:** 3b-T10's timed items take the local GPU before any
  phase 4 job. At most four workers at once.
- Estimate: 3–4 weeks (§16).

## Merge orders where files overlap

| File | Editors, in merge order |
| --- | --- |
| `state.py` | T01, T04 |
| `planner/minimal.py` | T01, T04, T05, T07 |
| `engine/graph.py` | T05, T07 |
| `simulation.py` | 3b-T07, T05, T10 |
| `io/trajectory.py` | T01, T04 |
| `ase/md.py` | T01, T04, T10 |
| `reference/records.py` | T11, T12 |
| `protocols/`, `cli/run.py` | 3b-T07, T09 |
| `cli/__main__.py` | T09, T10 |

## Rules carried over (owner, 22 October 2026)

- **Tests and merges:**
  - workers run only the tests their change touches;
  - the full CPU suite runs once per merge, in the conductor's batched trial
    merges;
  - one full CPU suite runs on the PC at a time.
- **GPU tests:** trials run on the CPU only, so a task that changes behaviour
  covered by GPU tests (slow or acceptance) runs the affected ones itself.
  Acceptance starts with the preflight of 4-T13.
- **GPU jobs:** through the worktree's `tools/ci/local_gpu.py run`, the FIFO
  queue (OPS-T01), as background tasks of the worker's session. A refused
  launch waits for the owner.
- **Timing:** repeats sized to the effect (decision 39). Cross-process
  control clauses use the documented 3 % noise floor or are left out.
- **Modal:** within the month's $30 under the guard in
  `tools/modal/fastmace_modal.py`. It stops at $27, keeping the $3 reserve.
  Check `modal billing summary` before planning. Long untimed runs (the ASE
  sides of T07 and T08) may move there; timed items stay on the RTX 5060 Ti.
- **Workstations:** off until the owner says they're back. Then only when
  nobody else is on the GPU or running active jobs, with files only under
  `/mnt/homes/zwingel`.
- **Branches:** workers push their own `fastmace/4-Tnn` branches.
- **Small, low-risk tasks** (T11, T12) skip the prestart report and the
  separate validation document. They keep their GPU check.

## Answers from the tasks

- **4-T01** (branch `fastmace/4-T01`, contract in
  [docs/design/physics.md](../docs/design/physics.md)):
  - `Optimizer.propose()` and `update()` take no argument: the optimiser is bound to its
    `DOFView` at construction (`Class(dof, config)`), at fixed addresses. The plan had
    `propose(state)`; accepted by the conductor on 23 October 2026.
  - The support table is a registry of scheme records, one module per scheme in
    `integrators/schemes/` and `optimize/schemes/`. T02, T03, T05, T06 and T07 add their
    records there, not in `planner/minimal.py`.
  - The new integrator records `NoseHooverChain` and `MTK` also take `fixcm` and `dof`, following
    D2 (a)'s "the new integrators"; choice 1 listed them only for NVE, CSVR and Berendsen.
  - FIRE's `dt` and `dtmax` stay in ASE time units, as in ASE (choice 1: ASE's names otherwise).
  - Merged as `352b33c` (from `fastmace/4-T01-rebased` at `bc40c19`, 23 October 2026). SPEC §10.6's
    format 4 paragraph is applied.
  - **SPEC wording held until D1, D2 and D5 are decided.** 4-T01 proposed the text below, written for
    the recommended options. The conductor applies it once the owner decides, and adjusts it if the
    decision differs.
    - §4.4, the paragraph under the heading: "Support is declared per scheme: each integrator,
      optimiser and cell filter has a record in fastmace.integrators.registry that lists its
      ensembles, cell modes and constraint rows (fixcm, fix_atoms), with a reason for each row it
      lacks. Rule 1 (§9.3) reads these records and refuses anything outside them at plan time, before
      any compilation, with a reason and a stable code. FixAtoms is held in the state as a device mask
      and host indices. Every other ASE constraint is refused, and so are MC moves with any constraint
      (decision 10)."
    - §4.4, the FixAtoms row (D1): "NVE, Langevin, CSVR and NVT Berendsen, and FIRE and LBFGS at fixed
      cell; refused for NHC, MTK and every variable cell (BAOAB, MTK, NPT Berendsen, cell filters), and
      with fixcm=True". Fixed centre of mass row: "all 1.0 integrators".
    - §4.4, degrees of freedom (D2): "Temperatures and thermostat targets count 3N − 3 degrees of
      freedom under fixcm and 3(N − n_fixed) with FixAtoms (dof="auto"), or the scheme's ASE count
      with dof="ase". Langevin and BAOAB keep their counts (decision 37)."
    - §7.7 (D5): "Phase 4 maps VelocityVerlet, Bussi, NoseHooverChainNVT, IsotropicMTKNPT, MTKNPT,
      NVTBerendsen, NPTBerendsen and Inhomogeneous_NPTBerendsen with ASE 3.27's arguments and units.
      Each argument list is held as data and checked against inspect.signature. rng is refused in
      favour of seed=. fastmace.ase.LBFGS and FIRE take ASE 3.27's arguments: use_line_search=True and
      position_reset_callback are refused, and the deprecated maxmove too. They take Atoms or
      ase.filters.FrechetCellFilter, and the other cell filters are refused with the reason. In
      Simulation the same optimisers are fm.optimize.LBFGS and FIRE, and FrechetCellFilter is
      fm.optimize.FrechetCell."
- **4-T03 and 4-T03b** (branches `fastmace/4-T03-rebased`, `fastmace/4-T03b`;
  [docs/validation/4-T03.md](../docs/validation/4-T03.md), [4-T03b.md](../docs/validation/4-T03b.md)):
  - The task's "the chain updates are device scalar operations, unrolled at capture" now reads
    **"fused"**: on CUDA with Triton, each chain half and MTK's 3 × 3 exp/exprel run as one Triton
    program each, bitwise the PyTorch stage (4-T03b, accepted by the conductor on 23 October 2026).
    The CPU keeps the PyTorch stage, within 1e-9 Å of ASE over 300 steps.
  - D2 (a) for NHC and MTK: N_f replaces ASE's 3N in every coupling term (conductor, 23 October).
  - MTK has no `eigh`: exp and exprel of the symmetric cell momentum as a matrix series.
  - The MTK record's `state_keys` serve both cell modes; the full cell's `eps` stays 0.

## Checkpoints for the owner

1. Before the go messages: D1, D3, D5 and D6. D2 and D4 can wait until
   their tasks reach the default line.
2. After T02 and T03: whether every deterministic integrator meets choice 2's
   limits over 4,000 steps, or the shortened horizon.
3. After T08: the minima result. D4's default takes effect if it passes, and
   a ResearchModule GPU quench becomes possible as a follow-up.
4. After T13: the acceptance report, and the tag `fastmace-4`.
